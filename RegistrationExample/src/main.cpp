#include <ctime>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <OverlapTrimmer/include/trimmer.h>
#include <ScaleStretchICP/include/ssicp.h>

void main(int argc, char *argv[])
{
  clock_t start = clock();

  Eigen::MatrixXd A, B;
  Eigen::MatrixXi F;
  igl::readOBJ(argv[1], A, F);
  igl::readOBJ(argv[2], B, F);

  Trimmer::Test(A, B, 0.01);
  igl::writeOBJ("a.obj", A, Eigen::MatrixXi());
  igl::writeOBJ("b.obj", B, Eigen::MatrixXi());

  SSICP::Test("a.obj", "b.obj", "a-trans.obj");

  printf("Execution Time: %lfs\n", static_cast<double>(clock() - start) / CLOCKS_PER_SEC);
  return;
}