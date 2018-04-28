#include "gps.h"

#include <Eigen/dense>
#include "./ScaleStretchICP/include/ssicp.h"

GPS_PUBLIC void GPS::CalcualteCamerasTransformation(const Eigen::MatrixXd &X, const Eigen::MatrixXd& Y, double &s,
  Eigen::Matrix3d &R, Eigen::RowVector3d &T)
{
  double a, b;
  SSICP::Initialize(X, Y, s, a, b, R, T);
  SSICP::FindTransformation(X, Y, a, b, s, R, T);
}

GPS_PUBLIC Eigen::MatrixXd GPS::GetLocalizedPoints(const Eigen::MatrixXd &X, const double &s, const Eigen::Matrix3d &R,
  const Eigen::RowVector3d &T)
{
  return SSICP::GetTransformed(X, s, R, T);
}

GPS_PUBLIC void GPS::Localize()
{

}
