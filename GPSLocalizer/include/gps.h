#ifndef GPS_H
#define GPS_H
#include "prereq.h"

#include <Eigen/dense>

namespace GPS
{
  GPS_PUBLIC void CalcualteCamerasTransformation(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, double &s,
    Eigen::Matrix3d &R, Eigen::RowVector3d &T);
  GPS_PUBLIC Eigen::MatrixXd GetLocalizedPoints(const Eigen::MatrixXd &X, const double &s, const Eigen::Matrix3d &R,
    const Eigen::RowVector3d &T);
  GPS_PUBLIC void Localize();
}

#endif