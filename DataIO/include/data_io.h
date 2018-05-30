#ifndef DATA_IO_H
#define DATA_IO_H
#include "prereq.h"

#include <string>
#include <vector>
#include <Eigen/Dense>

struct t_ply_argument_;
typedef struct t_ply_argument_ *p_ply_argument;

namespace DataIO
{
	DATAIO_PUBLIC int read_ply(const std::string& filename, Eigen::MatrixXd& v, Eigen::MatrixXd& vc, Eigen::MatrixXd& vn);
	DATAIO_PUBLIC int write_ply(const std::string& filename, const Eigen::MatrixXd& v, const Eigen::MatrixXd& vc, const Eigen::MatrixXd& vn);
	DATAIO_PUBLIC int load_gps(const std::string &filename, Eigen::MatrixXd &local);
	DATAIO_PUBLIC int read_fgr_config(const std::string& filename, std::vector<std::string>& pointcloud_filenames, std::string& feature_type, std::vector<std::string>& feature_filenames, std::string& correspondences_filename, int& downsampling_num, int& pca_component_num);
}

#endif