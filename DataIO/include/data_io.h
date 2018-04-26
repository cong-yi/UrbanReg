#ifndef DATA_IO_H
#define DATA_IO_H
#include "prereq.h"
#include <string>
#include <Eigen/Dense>
struct t_ply_argument_;
typedef struct t_ply_argument_ *p_ply_argument;
namespace DataIO
{
	DATAIO_PUBLIC int read_ply(const std::string& filename, Eigen::MatrixXd& v, Eigen::MatrixXd& vc);
	DATAIO_PUBLIC int write_ply(const std::string& filename, const Eigen::MatrixXd& v, const Eigen::MatrixXd& vc);
}

#endif