static const char help[] =
  "Read tetrahedral 'raw mesh' (rmesh, file format used for Parmod),\n"
  "determine the volume of dual cells, set boundary conditions, and write\n"
  "in the FUN3D binary format\n\n";

#include <petscmat.h>
#include <assert.h>
#include <voro++.hh>

#define ALEN(a) (sizeof (a) / sizeof (a)[0])
static int verbose = 0;

typedef struct {
  int nfree;
  int free[100];                // Free boundary tags
  int nsolid;
  int solid[100];               // Solid boundary tags
} BoundarySpec;
typedef struct _TetMesh *TetMesh;
struct _TetMesh {
  int ncell,nedge,nvtx,nface;
  int *cellconn;                // tetrahedral cell connectivity
  int *facetag;                 // numeric tags on faces
  int *faceconn;                // triangular face connectivity
  int *v2foff,*v2f;             // reverse links: vertex to face
  int *edge;                    // edges of tetrahedral mesh (NOT interlaced)
  double *edgevec;              // unit x,y,z, then area (NOT interlaced)
  double *vtxcoord;             // vertex coordinates (centroids of dual volumes)
  double *vdualvol;             // volume of dual volumes (corresponds to primal vertex centers)
  Mat A;                        // nodal adjacency (primal edges, dual faces)
  BoundarySpec *bs;
};

static PetscErrorCode Calloc_Private(size_t size,void *ptr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(size,(void**)ptr);CHKERRQ(ierr);
  ierr = PetscMemzero(*(void**)ptr,size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#define CallocA(n,p) Calloc_Private((n)*(sizeof(**(p))),(p))

static PetscErrorCode FileReadRaw(void *buf,size_t size,size_t nmemb,const char *fmt,...)
{
  va_list Argp;
  char fname[PETSC_MAX_PATH_LEN];
  size_t fullLen,count;
  FILE *fp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  va_start(Argp,fmt);
  ierr = PetscVSNPrintf(fname,sizeof fname,fmt,&fullLen,Argp);CHKERRQ(ierr);
  va_end(Argp);
  fp = fopen(fname,"rb");
  if (!fp) {perror(""); SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"%s",fname);}
  count = fread(buf,size,nmemb,fp);
  if (count != nmemb) {
    perror("");
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Read %D/%D items of size %D from %s",(PetscInt)count,(PetscInt)nmemb,(PetscInt)size,fname);
  }
  fclose(fp);
  PetscFunctionReturn(0);
}

#define FileReadRawA1(arr,n,fmt,arg) FileReadRaw((arr),sizeof(*(arr)),(n),(fmt),(arg))

static PetscErrorCode TetMeshLoadRMesh(MPI_Comm comm,const char *fname,TetMesh *itm)
{
  PetscErrorCode ierr;
  TetMesh tm;
  int *tmp,i,j,k,*v2f,*v2foff,*v2fcnt;
  float *ftmp;

  PetscFunctionBegin;
  ierr = CallocA(1,&tm);CHKERRQ(ierr);
  ierr = FileReadRawA1(&tm->ncell,1,"%s/nCells.bin",fname);CHKERRQ(ierr);
  ierr = FileReadRawA1(&tm->nvtx,1,"%s/nPoints.bin",fname);CHKERRQ(ierr);
  ierr = FileReadRawA1(&tm->nface,1,"%s/nFaces.bin",fname);CHKERRQ(ierr);

  ierr = CallocA(tm->nvtx*3,&ftmp);CHKERRQ(ierr);
  ierr = FileReadRawA1(ftmp,tm->nvtx*3,"%s/points.bin",fname);CHKERRQ(ierr);
  ierr = CallocA(tm->nvtx*3,&tm->vtxcoord);CHKERRQ(ierr);
  for (i=0; i<tm->nvtx; i++) {
    tm->vtxcoord[tm->nvtx*0+i] = ftmp[i*3+0];
    tm->vtxcoord[tm->nvtx*1+i] = ftmp[i*3+1];
    tm->vtxcoord[tm->nvtx*2+i] = ftmp[i*3+2];
  }
  ierr = PetscFree(ftmp);CHKERRQ(ierr);

  ierr = CallocA(tm->ncell,&tmp);CHKERRQ(ierr);
  ierr = FileReadRawA1(tmp,tm->ncell,"%s/cellSizes.bin",fname);CHKERRQ(ierr);
  for (i=0; i<tm->ncell; i++) if (tmp[i] != 4) SETERRQ1(comm,PETSC_ERR_SUP,"No support for non-tetrahedral element at %D",i);
  ierr = PetscFree(tmp);CHKERRQ(ierr);

  ierr = CallocA(tm->ncell*4,&tm->cellconn);CHKERRQ(ierr);
  ierr = FileReadRawA1(tm->cellconn,tm->ncell*4,"%s/cells.bin",fname);CHKERRQ(ierr);

  ierr = CallocA(tm->nface,&tmp);CHKERRQ(ierr);
  ierr = FileReadRawA1(tmp,tm->nface,"%s/faceSizes.bin",fname);CHKERRQ(ierr);
  for (i=0; i<tm->nface; i++) if (tmp[i] != 3) SETERRQ3(comm,PETSC_ERR_SUP,"No support for non-triangular face (%D-vertices) at face %D of %D",tmp[i],i,tm->nface);
  ierr = PetscFree(tmp);CHKERRQ(ierr);

  ierr = CallocA(tm->nface,&tm->facetag);CHKERRQ(ierr);
  ierr = FileReadRawA1(tm->facetag,tm->nface,"%s/faceTags.bin",fname);CHKERRQ(ierr);

  ierr = CallocA(tm->nface*3,&tm->faceconn);CHKERRQ(ierr);
  ierr = FileReadRawA1(tm->faceconn,tm->nface*3,"%s/faces.bin",fname);CHKERRQ(ierr);

  // Filter the faces by dropping all the interior faces (tag=0)
  k=0;
  for (i=0; i<tm->nface; i++) {
    if (tm->facetag[i] == 0) continue;
    tm->facetag[k] = tm->facetag[i];
    for (j=0; j<3; j++) tm->faceconn[k*3+j] = tm->faceconn[i*3+j];
    k++;
  }
  printf("Kept %d of %d faces\n",k,tm->nface);
  tm->nface = k;

  ierr = MatCreateSeqAIJ(comm,tm->nvtx,tm->nvtx,24,PETSC_NULL,&tm->A);CHKERRQ(ierr);
  for (i=0; i<tm->ncell; i++) {
    for (j=0; j<3; j++) {
      for (k=j+1; k<4; k++) {
        int node0,node1;
        node0 = tm->cellconn[i*4+j];
        node1 = tm->cellconn[i*4+k];
        ierr = MatSetValue(tm->A,node0,node1,1.0,ADD_VALUES);CHKERRQ(ierr);
        ierr = MatSetValue(tm->A,node1,node0,1.0,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(tm->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tm->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  { // Build the reduced edge data structure
    int m,*ai,*aj,cnt;
    PetscBool done;
    ierr = MatGetRowIJ(tm->A,0,PETSC_FALSE,PETSC_FALSE,&m,&ai,&aj,&done);CHKERRQ(ierr);
    if (!done) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Could not get RowIJ");
    tm->nedge = 0;
    for (int i=0; i<m; i++) {
      for (int j=ai[i]; j<ai[i+1]; j++) {
        if (i < aj[j]) tm->nedge++;
      }
    }
    ierr = CallocA(2*tm->nedge,&tm->edge);CHKERRQ(ierr);
    cnt = 0;
    for (int i=0; i<m; i++) {
      for (int j=ai[i]; j<ai[i+1]; j++) {
        if (i < aj[j]) {
          tm->edge[cnt]           = i;
          tm->edge[cnt+tm->nedge] = aj[j];
          cnt++;
        }
      }
    }
    ierr = MatRestoreRowIJ(tm->A,0,PETSC_FALSE,PETSC_FALSE,&m,&ai,&aj,&done);CHKERRQ(ierr);
  }

  // Create a map from vertices to faces that they are adjacent to
  ierr = CallocA(tm->nvtx,&v2fcnt);CHKERRQ(ierr);
  ierr = CallocA(tm->nvtx+1,&v2foff);CHKERRQ(ierr);
  ierr = CallocA(tm->nface*3,&v2f);CHKERRQ(ierr);
  for (i=0; i<tm->nface; i++) { // Count number of faces that each vertex interacts with
    int node0,node1,node2;
    node0 = tm->faceconn[i*3+0];
    node1 = tm->faceconn[i*3+1];
    node2 = tm->faceconn[i*3+2];
    v2fcnt[node0]++;
    v2fcnt[node1]++;
    v2fcnt[node2]++;
  }
  // Convert from a count to an offset
  v2foff[0] = 0;
  for (i=1; i<tm->nvtx+1; i++) {
    v2foff[i] = v2foff[i-1] + v2fcnt[i-1];
  }
  assert(v2foff[tm->nvtx] == tm->nface*3);
  // Add the reverse links
  ierr = PetscMemzero(v2fcnt,tm->nvtx*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<tm->nface; i++) {
    int node0,node1,node2;
    node0 = tm->faceconn[i*3+0];
    node1 = tm->faceconn[i*3+1];
    node2 = tm->faceconn[i*3+2];
    v2f[v2foff[node0] + v2fcnt[node0]++] = i;
    v2f[v2foff[node1] + v2fcnt[node1]++] = i;
    v2f[v2foff[node2] + v2fcnt[node2]++] = i;
  }
  tm->v2f = v2f;
  tm->v2foff = v2foff;
  ierr = PetscFree(v2fcnt);CHKERRQ(ierr);

  if (0) {
    IS rperm,cperm;
    Mat B;

    ierr = MatView(tm->A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = MatGetOrdering(tm->A,MATORDERINGRCM,&rperm,&cperm);CHKERRQ(ierr);
    ierr = MatPermute(tm->A,rperm,cperm,&B);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  *itm = tm;
  PetscFunctionReturn(0);
}

static int Sorted2(int *a,int *b) {
  if (*b < *a) {
    int t = *a;
    *a = *b;
    *b = t;
    return -1;
  } else return 1;
}

static PetscErrorCode FindEdge(int n,const int *edge,int x,int y,int *found,double *orientation)
{
  const int *dest = edge + n;
  *orientation = Sorted2(&x,&y);
  *found = -1;
  int low = 0;
  while (n - low > 1) {
    int mid = low + (n - low) / 2;
    if (x < edge[mid] || (x == edge[mid] && y < dest[mid]))
      n = mid;
    else low = mid;
  }
  if (!(edge[low] == x && dest[low] == y)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Could not find edge (%d -> %d)",x,y);
  *found = low;
  return 0;
}

static PetscErrorCode TetMeshComputeDualVolumes(TetMesh tm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CallocA(4*tm->nedge,&tm->edgevec);CHKERRQ(ierr);
  ierr = CallocA(tm->nvtx,&tm->vdualvol);CHKERRQ(ierr);
  for (int i=0; i<tm->nvtx; i++) {
    int j,k,nj;
    const int *aj;
    voro::voronoicell_neighbor vc;
    double edgevec[3][256];

    // Cut at the edge midpoints for each neighboring vertex
    ierr = MatGetRow(tm->A,i,&nj,&aj,PETSC_NULL);CHKERRQ(ierr);
    assert(nj < 256);
    double bb[3] = {0,0,0};
    double cc[3] = {tm->vtxcoord[tm->nvtx*0+i],tm->vtxcoord[tm->nvtx*1+i],tm->vtxcoord[tm->nvtx*2+i]};
    for (j=0; j<nj; j++) {
      for (k=0; k<3; k++) bb[k] = PetscMax(bb[k],PetscAbs(tm->vtxcoord[tm->nvtx*k+aj[j]]-cc[k]));
    }
    vc.init(-bb[0],bb[0],-bb[1],bb[1],-bb[2],bb[2]); // Initial bounding box, centered at the origin

    for (j=0; j<nj; j++) {
      for (k=0; k<3; k++) edgevec[k][j] = (tm->vtxcoord[tm->nvtx*k+aj[j]]-cc[k]); // displacement particle, cut plane bisects this vector
      // Cut the cell at the midpoint of the dual edge. The new plane ID is j, the index of this edge.
      if (!vc.nplane(edgevec[0][j],edgevec[1][j],edgevec[2][j], j)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Plane cut deleted cell");

      // Normalize edgevec
      double len = sqrt(PetscSqr(edgevec[0][j]) + PetscSqr(edgevec[1][j]) + PetscSqr(edgevec[2][j]));
      edgevec[0][j] /= len;
      edgevec[1][j] /= len;
      edgevec[2][j] /= len;
    }

    // Cut at the faces for each boundary face
    int nfaces = tm->v2foff[i+1] - tm->v2foff[i];
    for (int j=0; j<nfaces; j++) {
      int *verts = &tm->faceconn[tm->v2f[tm->v2foff[i]+j]*3];
      double coords[3][3];
      double edgeA[3],edgeB[3],normal[3];
      for (int k=0; k<3; k++) {
        coords[k][0] = tm->vtxcoord[tm->nvtx*0 + verts[k]];
        coords[k][1] = tm->vtxcoord[tm->nvtx*1 + verts[k]];
        coords[k][2] = tm->vtxcoord[tm->nvtx*2 + verts[k]];
      }
      for (int k=0; k<3; k++) {
        edgeA[k] = coords[1][k] - coords[0][k];
        edgeB[k] = coords[2][k] - coords[0][k];
      }
      // normal computed as the cross product of the edges
      normal[0] = edgeA[1]*edgeB[2] - edgeA[2]*edgeB[1];
      normal[1] = edgeA[2]*edgeB[0] - edgeA[0]*edgeB[2];
      normal[2] = edgeA[0]*edgeB[1] - edgeA[1]*edgeB[0];
      double len = sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
      normal[0] /= len;
      normal[1] /= len;
      normal[2] /= len;
      voro::wall_plane wp(normal[0],normal[1],normal[2],0); // Plane intersects the origin (since it was created relative to the vertex)
      wp.cut_cell(vc,0,0,0);                                // Dual cell is also centered at the origin
    }

    std::vector<double> vedge,varea;
    std::vector<int> neighbors;
    vc.neighbors(neighbors);
    vc.normals(vedge);
    vc.face_areas(varea);
    assert(neighbors.size() == varea.size());
    int cnt = 0;
    for (int j=0; j<(int)neighbors.size(); j++) {
      int k = neighbors[j];
      if (k >= 0) {             // This neighbor is "real"
        int node0,node1;
        node0 = i;
        node1 = aj[k];
        int edge;
        double orientation;
        ierr = FindEdge(tm->nedge,tm->edge,node0,node1,&edge,&orientation);CHKERRQ(ierr);
        tm->edgevec[0*tm->nedge+edge] = vedge[3*j+0]*orientation;
        tm->edgevec[1*tm->nedge+edge] = vedge[3*j+1]*orientation;
        tm->edgevec[2*tm->nedge+edge] = vedge[3*j+2]*orientation;
        tm->edgevec[3*tm->nedge+edge] = varea[j];

        if (verbose > 2) printf("[%5d -> %5d] neighbor %d: area=%g edgevec=(%g %g %g) vnormal=(%g %g %g)\n",node0,node1,k,varea[j],
                            edgevec[0][k],edgevec[1][k],edgevec[2][k],
                            vedge[j*3], vedge[j*3+1], vedge[j*3+2]);
        cnt++;
      }
    }
    if (cnt != nj) {
      if (verbose > 1) printf("dcell[%d] The number of neighbors %d != number of outgoing edges %d. Not Delaunay?\n",i,cnt,nj);
      //SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"The number of neighbors %d != number of outgoing edges %d. Not Delaunay?",cnt,nj);
    }
    if (verbose > 2) printf("dcell[%d] volume=%10.2e nj=%d nfaces=%d vnfaces=%zd\n",i,vc.volume(),nj,nfaces,varea.size());
    ierr = MatRestoreRow(tm->A,i,&nj,&aj,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TetMeshDestroy(TetMesh *tm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree((*tm)->cellconn);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->faceconn);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->facetag);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->v2foff);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->v2f);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->edge);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->edgevec);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->vtxcoord);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->vdualvol);CHKERRQ(ierr);
  ierr = MatDestroy(&(*tm)->A);CHKERRQ(ierr);
  ierr = PetscFree(*tm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct FUN3DHeader {
  int ncell;                    // number of tetrahedral cells in mesh
  int nnodes;                   // number of nodes (usually used as dual volumes)
  int nedge;                    // number of edges, corresponds to faces of dual volumes
  int nnbound;                  // number of solid boundary surfaces
  int nvbound;                  // number of viscous boundary surfaces
  int nfbound;                  // number of free surfaces
  int nnfacet;                  // total number of tetrahedral faces on solid boundaries
  int nvfacet;                  // total  number of tetrahedral faces on viscous boundaries
  int nffacet;                  // total  number of tetrahedral faces on free surface boundaries
  int nsnode;                   // total number of vertices on solid boundaries
  int nvnode;                   // total number of vertices on viscous boundaries
  int nfnode;                   // total number of vertices on free surface boundaries
  int ntte;                     // not used
};

static PetscErrorCode TetMeshWriteFUN3D(TetMesh tm,const char *filename)
{
  struct FUN3DHeader h;
  PetscViewer bv;
  BoundarySpec *bs = tm->bs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&bv);CHKERRQ(ierr);
  ierr = PetscMemzero(&h,sizeof h);CHKERRQ(ierr);
  h.ncell = tm->ncell;
  h.nnodes = tm->nvtx;
  h.nedge = tm->nedge;
  h.nnbound = bs->nsolid;
  h.nvbound = 0;
  h.nfbound = bs->nfree;

  // Count number of faces of each type
  h.nnfacet = 0;
  h.nvfacet = 0;
  h.nffacet = 0;
  for (int i=0; i<tm->nface; i++) {
    int ftag = tm->facetag[i];
    bool found = false;
    for (int k=0; k<bs->nsolid; k++) if (ftag == bs->solid[k]) found = true;
    if (found) {h.nnfacet++; continue;}
    for (int k=0; k<bs->nfree; k++) if (ftag == bs->free[k]) found = true;
    if (found) {h.nffacet++; continue;}
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Face tag %D not found in any boundary",ftag);
  }

  // Count number of nodes on each type of boundary
  h.nsnode = 0;
  h.nvnode = 0;
  h.nfnode = 0;
  for (int i=0; i<tm->nvtx; i++) {
    for (int j=tm->v2foff[i]; j<tm->v2foff[i+1]; j++) {
      int ftag = tm->facetag[tm->v2f[j]];
      bool found = false;
      for (int k=0; k<bs->nsolid; k++) if (ftag == bs->solid[k]) found = true;
      if (found) {h.nsnode++; continue;}
      for (int k=0; k<bs->nfree; k++) if (ftag == bs->free[k]) found = true;
      if (found) {h.nfnode++; continue;}
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Face tag %D not found in any boundary",ftag);
    }
  }

  h.ntte = -1;                  // not used

  assert(13 == sizeof(h)/sizeof(int));
  ierr = PetscViewerBinaryWrite(bv,&h,13,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);                // header
  ierr = PetscViewerBinaryWrite(bv,tm->edge,2*tm->nedge,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr); // edges
  ierr = PetscViewerBinaryWrite(bv,tm->edgevec,4*tm->nedge,PETSC_DOUBLE,PETSC_FALSE);CHKERRQ(ierr); // unit normal and area of dual faces
  ierr = PetscViewerBinaryWrite(bv,tm->vtxcoord,3*tm->nvtx,PETSC_DOUBLE,PETSC_FALSE);CHKERRQ(ierr); // coordinates of dual cell centers
  ierr = PetscViewerBinaryWrite(bv,tm->vdualvol,tm->nvtx,PETSC_DOUBLE,PETSC_FALSE);CHKERRQ(ierr);   // volume of dual volumes
  // FIXME: Write solid boundaries, viscous boundaries, and free boundaries
  ierr = PetscViewerDestroy(&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  char rmeshbase[PETSC_MAX_PATH_LEN] = "";
  PetscBool flg;
  TetMesh tm;
  PetscErrorCode ierr;
  BoundarySpec bs;

  PetscInitialize(&argc,&argv,0,help);
  ierr = PetscMemzero(&bs,sizeof bs);CHKERRQ(ierr);
  bs.nfree = ALEN(bs.free);
  bs.nsolid = ALEN(bs.solid);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","rmeshdual options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-f","Directory containing the rmesh binary files","",rmeshbase,rmeshbase,sizeof rmeshbase,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-boundary_free","Face tags corresponding to free boundaries","",bs.free,&bs.nfree,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-boundary_solid","Face tags corresponding to solid boundaries","",bs.solid,&bs.nsolid,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-verbose","Verbosity level","",verbose,&verbose,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must specify an input rmesh file with -f input.rmesh");
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TetMeshLoadRMesh(PETSC_COMM_WORLD,rmeshbase,&tm);CHKERRQ(ierr);
  tm->bs = &bs;
  ierr = TetMeshComputeDualVolumes(tm);CHKERRQ(ierr);
  ierr = TetMeshWriteFUN3D(tm,"output.msh");CHKERRQ(ierr);
  ierr = TetMeshDestroy(&tm);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
