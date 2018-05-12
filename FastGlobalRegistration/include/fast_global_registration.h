#ifndef FAST_GLOBAL_REGISTRATION_H
#define FAST_GLOBAL_REGISTRATION_H
#include "prereq.h"
#include <Eigen/Dense>
#include <vector>

namespace FastGlobalRegistration
{
	FGR_PUBLIC void advanced_matching(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const Eigen::MatrixXd& fpfh_1, const Eigen::MatrixXd& fpfh_2, std::vector<std::pair<int, int> >& corres);
	FGR_PUBLIC double optimize_pairwise(bool decrease_mu, int num_iter, const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const std::vector<std::pair<int, int> >& corres, Eigen::Matrix4d& trans_mat);

  FGR_PUBLIC Eigen::Matrix4f update_fgr(const Eigen::MatrixXd &v_1, const Eigen::MatrixXf &v_2,
    const std::vector<std::pair<int, int>> &corres, const double mu);
  FGR_PUBLIC Eigen::Matrix4f update_ssicp(const Eigen::MatrixXd &v_1, const Eigen::MatrixXf &v_2,
    const std::vector<std::pair<int, int>> &corres, const double mu);
}

#endif