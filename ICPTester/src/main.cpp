#include <igl/readPLY.h>
#include <igl/writePLY.h>
#include "ScaleStretchICP/include/ssicp.h"

int main(int argc, char *argv[])
{
  Eigen::MatrixXd X, Y;
  Eigen::MatrixXi F;
  igl::readPLY(argv[1], X, F);
  igl::readPLY(argv[2], Y, F);
  Eigen::MatrixXd A = SSICP::Align(X, Y);
  igl::writePLY(argv[3], A, Eigen::MatrixXi());

  system("pause");
	return 0;
}
