#include "ssicp.h"

#include <cmath>
#include <algorithm>
#include <climits>

#include <Eigen/dense>
#include <flann/flann.hpp>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

void EigenvaluesAndEigenvectors(const Eigen::Matrix3d &A, std::vector<double> &eigenvalues,
  std::vector<Eigen::Vector3cd> &eigenvectors)
{
  Eigen::EigenSolver<Eigen::Matrix3d> es(A);
  eigenvalues.resize(3), eigenvectors.resize(3);
  for (size_t i = 0; i < 3; ++i)
  {
    eigenvalues[i] = es.eigenvalues()[i].real();
    eigenvectors[i] = es.eigenvectors().col(i);
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
  last_e = std::numeric_limits<double>::max();

  // calculate initial values for s(a, b)
  Eigen::RowVector3d x_c = X.colwise().sum() / static_cast<double>(X.rows());
  Eigen::RowVector3d y_c = Y.colwise().sum() / static_cast<double>(Y.rows());
  Eigen::MatrixXd X_tilde(X), Y_tilde(Y);
  X_tilde.rowwise() -= x_c, Y_tilde.rowwise() -= y_c;
  Eigen::Matrix3d M_X = (X_tilde.transpose()) * X_tilde;
  Eigen::Matrix3d M_Y = (Y_tilde.transpose()) * Y_tilde;

  std::vector<double> evax, evay;
  std::vector<Eigen::Vector3cd> evex, evey;
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
  Eigen::Matrix3d P, Q;
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
    {
      P(i, j) = evex[j](i).real();
      Q(i, j) = evey[j](i).real();
    }
  R = Q * (P.transpose());

  // intial values for T
  T = y_c - x_c;
}

SSICP_PUBLIC void SSICP::Iterate()
{
  size_t counter = 0;
  do
  {
    printf("Iteration #%zu: %lf\n", ++counter, last_e);
    FindCorrespondeces();
    FindTransformation();
  } while (Converged());
}

SSICP_PUBLIC void SSICP::FindCorrespondeces()
{
  // use flann to calculate nesrest points
  // build the data for searching
  size_t rows_data = Y.rows(), cols = Y.cols();
  double *data = static_cast<double *>(malloc(rows_data * cols * sizeof(double)));
  double *ptr = data;
  for (size_t i = 0; i < rows_data; ++i)
    for (size_t j = 0; j < cols; ++j)
      *(ptr++) = Y(i, j);

  flann::Matrix<double> mat_data(data, rows_data, cols);
  flann::Index<flann::L2<double>> index(mat_data, flann::KDTreeIndexParams(4));
  index.buildIndex();

  // build the queries
  Eigen::MatrixXd RTX = s * X * R.transpose();
  RTX.rowwise() += T;

  size_t rows_queries = X.rows();
  double *queries = static_cast<double *>(malloc(rows_queries * cols * sizeof(double)));
  ptr = queries;
  for (size_t i = 0; i < rows_queries; ++i)
    for (size_t j = 0; j < cols; ++j)
      *(ptr++) = RTX(i, j);
  flann::Matrix<double> mat_queries(queries, rows_queries, cols);

  // find the nearest points
  std::vector<std::vector<int>> indices;
  std::vector<std::vector<double>> dists;
  index.knnSearch(mat_queries, indices, dists, 1, flann::SearchParams(128));

  Z = Eigen::MatrixXd(rows_queries, cols);
  for (size_t i = 0; i < rows_queries; ++i)
    Z(i) = Y(indices[i].front());
}

SSICP_PUBLIC void SSICP::FindTransformation()
{
  // calculate X_tilde and Z_tilde
  size_t rows = X.rows(), cols = X.cols();
  Eigen::RowVector3d x_c = X.colwise().sum() / static_cast<double>(rows);
  Eigen::RowVector3d z_c = X.colwise().sum() / static_cast<double>(rows);

  Eigen::MatrixXd X_tilde(X), Z_tilde(X);
  X_tilde.rowwise() -= x_c, Z_tilde.rowwise() -= z_c;

  // update R using SVD decomposition
  Eigen::Matrix3d H = (X_tilde.transpose().eval()) * Z_tilde;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
  std::cout << U << std::endl;
  std::cout << V << std::endl;
  Eigen::Vector3d ttt = svd.singularValues();
  std::cout << ttt << std::endl;
  //std::cout << U * ttt * V << std::endl;
  std::cout << "H:" << std::endl;
  std::cout << H << std::endl;
  std::cout << (V * (U.transpose())).determinant() << std::endl;
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
  double e = 0;
  size_t rows = X.rows(), cols = X.cols();
  Eigen::RowVector3d x_c = X.colwise().sum() / static_cast<double>(rows);
  Eigen::RowVector3d z_c = Z.colwise().sum() / static_cast<double>(rows);
  Eigen::MatrixXd X_tilde(X), Z_tilde(Z);
  X_tilde.rowwise() -= x_c, Z_tilde.rowwise() -= z_c;

  e += s * s * X_tilde.cwiseProduct(X_tilde).sum();
  e -= 2 * s * Z_tilde.cwiseProduct(X_tilde * (R.transpose())).sum();
  e += Z_tilde.cwiseProduct(Z_tilde).sum();

  double theta = 1 - e / last_e;
  bool conv = (theta < epsilon);
  last_e = e;
  return conv;
}

SSICP_PUBLIC void SSICP::OutputTransformed(std::string out_filename)
{
  Eigen::Matrix3d A = s * X * R.transpose();
  A.rowwise() += T;
  igl::writeOBJ(out_filename, A, Eigen::MatrixXf());
}

SSICP_PUBLIC void SSICP::Test(const std::string file_name_x, const std::string file_name_y)
{
  Eigen::MatrixXi F;
  igl::readOBJ(file_name_x, X, F);
  igl::readOBJ(file_name_y, Y, F);
  SetEpsilon(0.01);

  Initialize();
  Iterate();
  OutputTransformed("out.obj");
}
