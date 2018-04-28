#ifndef SSICP_H
#define SSICP_H
#include "prereq.h"

#include <vector>
#include <Eigen/dense>

namespace SSICP
{
  // Initialize parameters
  SSICP_PUBLIC void Initialize(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y,
    double &s, double &a, double &b, Eigen::Matrix3d &R, Eigen::RowVector3d &T);

  // The iterative algorithm
  SSICP_PUBLIC void Iterate(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const double &a, const double &b,
    const double &epsilon, double &s, Eigen::Matrix3d &R, Eigen::RowVector3d &T);
  // Perform the first step of iteration
  SSICP_PUBLIC Eigen::MatrixXd FindCorrespondeces(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y, const double &s,
    const Eigen::Matrix3d &R, const Eigen::RowVector3d &T);
  // Perform the second step of iteration
  SSICP_PUBLIC void FindTransformation(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Z, const double &a,
    const double &b, double &s, Eigen::Matrix3d &R, Eigen::RowVector3d &T);
  // Compute error for current parameters
  SSICP_PUBLIC double ComputeError(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Z, const double &s,
    const Eigen::Matrix3d &R, const Eigen::RowVector3d &T);

  // Output current parameters
  SSICP_PUBLIC void OutputParameters(const double &s, const double &a, const double &b, const Eigen::MatrixXd &R,
    const Eigen::RowVector3d T);
  // Return tranformed point cloud
  SSICP_PUBLIC Eigen::MatrixXd GetTransformed(const Eigen::MatrixXd &X, double &s, const Eigen::MatrixX3d &R,
    const Eigen::RowVector3d &T);

  // The overall process of SSICP
  SSICP_PUBLIC Eigen::MatrixXd Align(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y);
}

#endif