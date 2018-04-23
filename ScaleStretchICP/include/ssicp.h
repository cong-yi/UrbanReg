#ifndef SSICP_H
#define SSICP_H
#include "prereq.h"

#include <vector>
#include <Eigen/dense>

namespace SSICP
{
  // variables
  Eigen::MatrixXd X, Y;
  
  Eigen::MatrixXd R, Z;
  Eigen::RowVector3d T;
  double s, a, b, last_e, epsilon;
   
  // methods
  SSICP_PUBLIC void SetX(const Eigen::MatrixXd &_);
  SSICP_PUBLIC void SetY(const Eigen::MatrixXd &_);
  SSICP_PUBLIC void SetEpsilon(double e);

  SSICP_PUBLIC void Initialize();

  SSICP_PUBLIC void Iterate();
  SSICP_PUBLIC void FindCorrespondeces();
  SSICP_PUBLIC void FindTransformation();
  SSICP_PUBLIC bool Converged();

  SSICP_PUBLIC void OutputTransformed(std::string out_filename);

  SSICP_PUBLIC void Test(const std::string &filename_x, const std::string &filename_y,
    const std::string &out_filename);
}

#endif