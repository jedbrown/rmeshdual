static const char help[] =
  "Read tetrahedral 'raw mesh' (rmesh, file format used for Parmod),\n"
  "determine the volume of dual cells, set boundary conditions, and write\n"
  "in the FUN3D binary format\n\n";

#include <petscmat.h>

typedef struct _TetMesh *TetMesh;
struct _TetMesh {
  int ncell,nedge,nvtx;
  int *cellconn;                // tetrahedral cell connectivity
  int *edge;                    // edges of tetrahedral mesh
  double *vtxcoord;             // vertex coordinates (centroids of dual volumes)
  double *vdualvol;             // volume of dual volumes (corresponds to primal vertex centers)
  double *enormal;              // edge normals
  Mat A;                        // nodal adjacency (primal edges, dual faces)
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

PetscErrorCode FileReadRaw(void *buf,size_t size,size_t nmemb,const char *fmt,...)
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

PetscErrorCode TetMeshLoadRMesh(MPI_Comm comm,const char *fname,TetMesh *itm)
{
  PetscErrorCode ierr;
  TetMesh tm;
  int *tmp,i,j,k;
  float *ftmp;

  PetscFunctionBegin;
  ierr = CallocA(1,&tm);CHKERRQ(ierr);
  ierr = FileReadRawA1(&tm->ncell,1,"%s/nCells.bin",fname);CHKERRQ(ierr);
  ierr = FileReadRawA1(&tm->nvtx,1,"%s/nPoints.bin",fname);CHKERRQ(ierr);

  ierr = CallocA(tm->nvtx*3,&ftmp);CHKERRQ(ierr);
  ierr = FileReadRawA1(ftmp,tm->nvtx*3,"%s/points.bin",fname);CHKERRQ(ierr);
  ierr = CallocA(tm->nvtx*3,&tm->vtxcoord);CHKERRQ(ierr);
  for (i=0; i<tm->nvtx*3; i++) tm->vtxcoord[i] = ftmp[i];
  ierr = PetscFree(ftmp);CHKERRQ(ierr);

  ierr = CallocA(tm->ncell,&tmp);CHKERRQ(ierr);
  ierr = FileReadRawA1(tmp,tm->ncell,"%s/cellSizes.bin",fname);CHKERRQ(ierr);
  for (i=0; i<tm->ncell; i++) if (tmp[i] != 4) SETERRQ1(comm,PETSC_ERR_SUP,"No support for non-tetrahedral element at %D",i);
  ierr = PetscFree(tmp);CHKERRQ(ierr);

  ierr = CallocA(tm->ncell*4,&tm->cellconn);CHKERRQ(ierr);
  ierr = FileReadRawA1(tm->cellconn,tm->ncell*4,"%s/cells.bin",fname);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJ(comm,tm->nvtx,tm->nvtx,24,PETSC_NULL,&tm->A);CHKERRQ(ierr);
  for (i=0; i<tm->ncell; i++) {
    for (j=0; j<3; j++) {
      for (k=j+1; k<4; k++) {
        int node0,node1;
        node0 = tm->cellconn[i*4+j];
        node1 = tm->cellconn[i*4+k];
        ierr = MatSetValue(tm->A,node0,node1,1.0,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(tm->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tm->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(tm->A,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);

  if (1) {
    IS rperm,cperm;
    Mat B;
    ierr = MatGetOrdering(tm->A,MATORDERINGRCM,&rperm,&cperm);CHKERRQ(ierr);
    ierr = MatPermute(tm->A,rperm,cperm,&B);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
  }

  *itm = tm;
  PetscFunctionReturn(0);
}

PetscErrorCode TetMeshDestroy(TetMesh *tm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscFree((*tm)->cellconn);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->edge);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->vtxcoord);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->vdualvol);CHKERRQ(ierr);
  ierr = PetscFree((*tm)->enormal);CHKERRQ(ierr);
  ierr = MatDestroy(&(*tm)->A);CHKERRQ(ierr);
  ierr = PetscFree(*tm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  char rmeshbase[PETSC_MAX_PATH_LEN] = "";
  PetscBool flg;
  TetMesh tm;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,0,help);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","rmeshdual options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-f","Directory containing the rmesh binary files","",rmeshbase,rmeshbase,sizeof rmeshbase,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must specify an input rmesh file with -f input.rmesh");
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TetMeshLoadRMesh(PETSC_COMM_WORLD,rmeshbase,&tm);CHKERRQ(ierr);

  ierr = TetMeshDestroy(&tm);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
