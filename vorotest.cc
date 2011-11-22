#include <voro++.hh>

// Cell center is ALWAYS at the origin.
// Cell is cut by plane that is the midpoint of the line segment from center of initial box to the particle
static int test_cuts() {
  voro::voronoicell_neighbor vc;
  std::vector<double> varea,vnormals;
  vc.init(0,2, 0,3, 0,1);
  vc.nplane(1,1,0, 0);
  vc.nplane(0,1,0, 1);
  printf("volume = %g\n",vc.volume());
  vc.face_areas(varea); voro::voro_print_vector(varea); puts("");
  vc.normals(vnormals); voro::voro_print_positions(vnormals); puts("");
  vc.output_neighbors(); puts("");
  std::vector<int> neighbors;
  vc.neighbors(neighbors);
  for (int i=0; i<(int)neighbors.size(); i++) {
    if (neighbors[i] >= 0) {
      printf("neighbor %d: area=%g  normal=(%g %g %g)\n",neighbors[i], varea[i], vnormals[i*3], vnormals[i*3+1], vnormals[i*3+2]);
    }
  }
  return 0;
}

static int test_walls() {
  voro::voronoicell_neighbor vc;
  vc.init(-1,1,-1,1,-1,1);
  vc.nplane(1,1,0,0);
  voro::wall_plane wp(0,0,1,0);
  printf("volume = %g\n",vc.volume());
  wp.cut_cell(vc, 0,0,0);
  printf("volume = %g\n",vc.volume());
  //vc.face_areas(varea); voro::voro_print_vector(varea); puts("");
  return 0;
}

#define RUN(test) do {                          \
    puts("\n#### " #test);                      \
    int ret = test();                           \
    if (ret) return ret;                        \
  } while (0);

int main() {
  RUN(test_cuts);
  RUN(test_walls);
  return 0;
}
