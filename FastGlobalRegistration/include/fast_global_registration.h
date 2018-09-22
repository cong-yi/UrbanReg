#ifndef FAST_GLOBAL_REGISTRATION_H
#define FAST_GLOBAL_REGISTRATION_H
#include "prereq.h"
#include <Eigen/Dense>
#include <vector>
#include <map>

namespace FastGlobalRegistration
{
	FGR_PUBLIC void advanced_matching(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const Eigen::MatrixXd& fpfh_1, const Eigen::MatrixXd& fpfh_2, std::vector<std::pair<int, int> >& corres);
	FGR_PUBLIC double optimize_pairwise(bool decrease_mu, int num_iter, const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const std::vector<std::pair<int, int> >& corres, Eigen::Matrix4d& trans_mat);
	FGR_PUBLIC double optimize_global(bool decrease_mu, int num_iter, std::map<int, Eigen::MatrixXd>& v_map, std::map<int, std::map<int, std::vector<std::pair<int, int> > > >& corres_map, std::map<int, Eigen::Matrix4d>& trans_mat_map);

	FGR_PUBLIC Eigen::Matrix4f update_fgr(const Eigen::MatrixXd &v_1, const Eigen::MatrixXf &v_2, const std::vector<std::pair<int, int>> &corres, const double mu);
	FGR_PUBLIC Eigen::Matrix3d compute_r(const Eigen::MatrixXd &p_mat, const Eigen::MatrixXd &q_mat, const double mu);
	FGR_PUBLIC Eigen::Matrix4d update_ssicp(const Eigen::MatrixXd &v_1, const Eigen::MatrixXd &v_2,
		const std::vector<std::pair<int, int>> &corres, const double mu);
	FGR_PUBLIC std::map<int, Eigen::Matrix4d> update_ssicp_global(std::map<int, Eigen::MatrixXd>& v_map, std::map<int, std::map<int, std::vector<std::pair<int, int> > > >& corres_map, const double mu);
}

#endif