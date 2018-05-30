#ifndef GOICP_H
#define GOICP_H

#include "prereq.h"
#include <Eigen/Dense>

namespace GOICP
{
	GOICP_PUBLIC Eigen::Matrix4d goicp(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, Eigen::MatrixXd& aligned_v_2);
}

#endif