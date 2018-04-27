#include "ssicp.h"

#include <cmath>
#include <algorithm>
#include <climits>

#include <Eigen/dense>
#include "kdtree.h"
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

void EigenvaluesAndEigenvectors(const Eigen::Matrix3d &A, std::vector<double> &eigenvalues,
  std::vector<Eigen::Vector3d> &eigenvectors)
{
  Eigen::EigenSolver<Eigen::Matrix3d> es(A);
  eigenvalues.resize(3), eigenvectors.resize(3);
  for (size_t i = 0; i < 3; ++i)
  {
    eigenvalues[i] = es.eigenvalues()[i].real();
    Eigen::Vector3cd ev = es.eigenvectors().col(i);
    for (size_t j = 0; j < 3; ++j)
      eigenvectors[i](j) = ev(j).real();
    eigenvectors[i].normalize();
  }

  // sort eigenvalues with eigenvectors
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = i + 1; j < 3; ++j)
      if (eigenvalues[i] > eigenvalues[j])
      {
        std::swap(eigenvalues[i], eigenvalues[j]);
        std::swap(eigenvectors[i], eigenvectors[j]);
      }
}

SSICP_PUBLIC void SSICP::SetX(const Eigen::MatrixXd &_)
{
  X = _;
}

SSICP_PUBLIC void SSICP::SetY(const Eigen::MatrixXd & _)
{
  Y = _;
}

SSICP_PUBLIC void SSICP::SetEpsilon(double e)
{
  epsilon = e;
}

SSICP_PUBLIC void SSICP::Initialize()
{
  Z = Eigen::MatrixXd::Zero(X.rows(), X.cols());
  last_e = std::numeric_limits<double>::max();

  // calculate initial values for s(a, b)
  Eigen::RowVector3d x_c = X.colwise().sum() / static_cast<double>(X.rows());
  Eigen::RowVector3d y_c = Y.colwise().sum() / static_cast<double>(Y.rows());
  Eigen::MatrixXd X_tilde(X), Y_tilde(Y);
  X_tilde.rowwise() -= x_c, Y_tilde.rowwise() -= y_c;
  Eigen::Matrix3d M_X = (X_tilde.transpose().eval()) * X_tilde;
  Eigen::Matrix3d M_Y = (Y_tilde.transpose().eval()) * Y_tilde;

  std::vector<double> evax, evay;
  std::vector<Eigen::Vector3d> evex, evey;
  EigenvaluesAndEigenvectors(M_X, evax, evex);
  EigenvaluesAndEigenvectors(M_Y, evay, evey);

  a = std::numeric_limits<double>::max(), b = std::numeric_limits<double>::min();
  for (size_t i = 0; i < 3; ++i)
  {
    double v = sqrt(evay[i] / evax[i]);
    a = std::min(a, v);
    b = std::max(b, v);
    s += v / 3;
  }

  // intial values for R
  /*
  Eigen::Matrix3d P, Q;
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
    {
      P(i, j) = evex[j](i);
      Q(i, j) = evey[j](i);
    }
  R = Q * (P.transpose());
  */
  // before fixing the orientation of the main axises
  // we use identity for initial R
  R = Eigen::Matrix3d::Identity();

  // intial values for T
  T = y_c - x_c;

  OutputParameters();
}

SSICP_PUBLIC void SSICP::Iterate()
{
  size_t counter = 0;
  do
  {
    printf("Iteration %zu:\n", ++counter);
    FindCorrespondeces();
    std::cout << "Error after the first step: " << ComputeError() << std::endl;
    FindTransformation();
    std::cout << "Error after the second step: " << ComputeError() << std::endl;
  } while (!Converged());
  printf("Iteration #%zu: %lf\n", counter, last_e);
}

SSICP_PUBLIC void SSICP::FindCorrespondeces()
{
  // use kdtree to calculate nesrest points
  kdtree *ptree = kd_create(3);
  char *data = new char('a');
  for (int i = 0; i < Y.rows(); ++i)
    kd_insert3(ptree, Y(i, 0), Y(i, 1), Y(i, 2), data);
  Eigen::MatrixXd SRXT = s * X * R.transpose();
  SRXT.rowwise() += T;
  for (int i = 0; i < X.rows(); ++i)
  {
    kdres *presults = kd_nearest3(ptree, SRXT(i, 0), SRXT(i, 1), SRXT(i, 2));
    while (!kd_res_end(presults))
    {
      double pos[3];
      kd_res_item(presults, pos);
      for (int j = 0; j < 3; ++j)
        Z(i, j) = pos[j];
      kd_res_next(presults);
    }
    kd_res_free(presults);
  }
  free(data);
  kd_free(ptree);
}

SSICP_PUBLIC void SSICP::FindTransformation()
{
  // calculate X_tilde and Z_tilde
  size_t rows = X.rows(), cols = X.cols();
  Eigen::RowVector3d x_c = X.colwise().sum() / static_cast<double>(rows);
  Eigen::RowVector3d z_c = Z.colwise().sum() / static_cast<double>(rows);
  Eigen::MatrixXd X_tilde(X), Z_tilde(Z);
  X_tilde.rowwise() -= x_c, Z_tilde.rowwise() -= z_c;

  // update R using SVD decomposition
  Eigen::Matrix3d H = (X_tilde.transpose().eval()) * Z_tilde;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
  if ((V * (U.transpose())).determinant() > 0)
    R = V * (U.transpose());
  else
  {
    Eigen::Matrix3d I;
    I << 1, 0, 0,
      0, 1, 0,
      0, 0, -1;
    R = V * I * (U.transpose());
  }

  // update s
  double num = Z_tilde.cwiseProduct(X_tilde * (R.transpose())).sum();
  double den = X_tilde.cwiseProduct(X_tilde).sum();
  s = num / den;
  if (s < a) s = a;
  if (s > b) s = b;

  T = z_c - s * x_c * (R.transpose());
}

SSICP_PUBLIC bool SSICP::Converged()
{
  double e = ComputeError();
  double theta = last_e > 0 ? 1 - e / last_e : 0;
  bool conv = (theta >= 0 && theta < epsilon);
  last_e = e;
  return conv;
}

SSICP_PUBLIC double SSICP::ComputeError()
{
  Eigen::MatrixXd E = s * X * R.transpose() - Z;
  E.rowwise() += T;
  double e = E.cwiseProduct(E).sum();
  return e;
}

SSICP_PUBLIC void SSICP::OutputParameters()
{
  std::cout << "Scale: " << s << " in [" << a << ", " << b << "]" << std::endl;
  std::cout << "Rotation: " << std::endl;
  std::cout << R << std::endl;
  std::cout << "Translation: " << std::endl;
  std::cout << T << std::endl;
}

SSICP_PUBLIC void SSICP::OutputTransformed(std::string out_filename)
{
  Eigen::MatrixXd A = s * X * R.transpose();
  A.rowwise() += T;
  igl::writeOBJ(out_filename, A, Eigen::MatrixXi());
}

SSICP_PUBLIC void SSICP::Test(const std::string &filename_x, const std::string &filename_y,
  const std::string &out_filename)
{
  Eigen::MatrixXi F;
  igl::readOBJ(filename_x, X, F);
  igl::readOBJ(filename_y, Y, F);
  SetEpsilon(0.0001);

  Initialize();
  Iterate();
  OutputTransformed(out_filename);
}
