#ifndef FEATURE_ALG_H
#define FEATURE_ALG_H
#include "prereq.h"
#include <Eigen/Dense>
#include <vector>

namespace FeatureAlg
{
	FEATUREALG_PUBLIC void compute_fpfh(const Eigen::MatrixXd& v, const Eigen::MatrixXd& vn, Eigen::MatrixXd& fpfh);
	FEATUREALG_PUBLIC void compute_shot(const Eigen::MatrixXd& v, const Eigen::MatrixXd& vn, const Eigen::MatrixXd& vc, Eigen::MatrixXd& shot);
	FEATUREALG_PUBLIC void compute_shot(const Eigen::MatrixXd& downsampled_v, const Eigen::MatrixXd& v, const Eigen::MatrixXd& vn, const Eigen::MatrixXd& downsampled_vc, const Eigen::MatrixXd& vc, Eigen::MatrixXd& shot);
}

#endif