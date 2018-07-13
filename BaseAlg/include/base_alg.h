#ifndef BASE_ALG_H
#define BASE_ALG_H
#include "prereq.h"
#include <Eigen/Dense>

namespace BaseAlg
{
    BASEALG_PUBLIC void icp(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, Eigen::MatrixXd& aligned_v_2);
	BASEALG_PUBLIC Eigen::Matrix4d normalize(const Eigen::VectorXd& min_corner, const Eigen::VectorXd& max_corner, Eigen::MatrixXd& v);
	//return the largest ev_num eigen values and corresponding eigen vectors (each column in matrix) of data_mat
	BASEALG_PUBLIC int pca(const Eigen::MatrixXd& data_mat, int ev_num, Eigen::VectorXd& eigen_values, Eigen::MatrixXd& eigen_vectors);
	BASEALG_PUBLIC double rmse(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2);
	//return the index of the nearest neighbour of each point in v_2 within the point cloud v_1
	BASEALG_PUBLIC Eigen::VectorXi find_nearest_neighbour(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, Eigen::VectorXd& distances);
	//return the sampled indices for slicing
	BASEALG_PUBLIC Eigen::VectorXi downsampling(int total_num, int downsampling_num);
}

#endif