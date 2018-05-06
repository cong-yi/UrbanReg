#ifndef FAST_GLOBAL_REGISTRATION_H
#define FAST_GLOBAL_REGISTRATION_H
#include "prereq.h"
#include <Eigen/Dense>
#include <vector>

namespace FastGlobalRegistration
{
	FGR_PUBLIC void normalize(const Eigen::VectorXd& min_corner, const Eigen::VectorXd& max_corner, Eigen::MatrixXd& v);
	FGR_PUBLIC void advanced_matching(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const Eigen::MatrixXd& fpfh_1, const Eigen::MatrixXd& fpfh_2, std::vector<std::pair<int, int> >& corres);
	FGR_PUBLIC double optimize_pairwise(bool decrease_mu, int num_iter, const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const std::vector<std::pair<int, int> >& corres, Eigen::Matrix4d& trans_mat);
}

#endif