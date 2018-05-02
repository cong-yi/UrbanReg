#ifndef FEATURE_ALG_H
#define FEATURE_ALG_H
#include "prereq.h"
#include <Eigen/Dense>

namespace FeatureAlg
{
	FEATUREALG_PUBLIC void compute_fpfh(const Eigen::MatrixXd& v, const Eigen::MatrixXd& vn, Eigen::MatrixXd& fpfh);
}

#endif