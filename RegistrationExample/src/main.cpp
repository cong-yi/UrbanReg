#include <ctime>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <OverlapTrimmer/include/trimmer.h>
#include <ScaleStretchICP/include/ssicp.h>
#include "DataIO/include/data_io.h"
#include "FeatureAlg/include/feature_alg.h"
#include "FastGlobalRegistration/include/fast_global_registration.h"
#include "igl/writeDMAT.h"
#include "igl/readDMAT.h"
#include "igl/slice.h"
#include "igl/jet.h"
#include "GoICP/include/goicp.h"
#include <igl/slice_mask.h>
#include <random>
#include "BaseAlg/include/base_alg.h"
#include "RegistrationPipeline/include/registration_pipeline.h"
#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
//#define USE_PCA

void icp_example(const std::string& pointcloud_a, const std::string& pointcloud_b)
{
	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_a, v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_b, v_2, vc_2, vn_2);

	Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	v.topRows(v_1.rows()) = v_1;
	v.bottomRows(v_2.rows()) = v_2;

	BaseAlg::normalize(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), v);
	v_1 = v.topRows(v_1.rows());
	v_2 = v.bottomRows(v_2.rows());

	Eigen::MatrixXd aligned_v_2;
	BaseAlg::icp(v_1, v_2, aligned_v_2);

	DataIO::write_ply("icp_1.ply", v_1, vc_1, vn_1);
	DataIO::write_ply("icp_2.ply", v_2, vc_2, vn_2);
	DataIO::write_ply("icp_2_aligned.ply", aligned_v_2, vc_2, vn_2);
	return;
}

void convert_trans_to_rmse(const std::string& point_cloud_a, const std::string& point_cloud_b, const std::string& trans_prefix, int iter_num, const std::string& output_rmse)
{
	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(point_cloud_a, v_1, vc_1, vn_1);
	DataIO::read_ply(point_cloud_b, v_2, vc_2, vn_2);

	Eigen::Matrix4d gt_trans_mat;
	gt_trans_mat <<
		-0.1346397895, 0.1154806788, 0.9841424388, -0.0000000000,
		0.9804971974, 0.1590265973, 0.1154806788, -0.0000000000,
		-0.1431690361, 0.9804971974, -0.1346397895, -0.0000000000,
		0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000;

	Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	v.topRows(v_1.rows()) = v_1;
	v.bottomRows(v_2.rows()) = v_2;

	Eigen::Matrix4d normalization_trans_mat = BaseAlg::normalize(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), v);
	gt_trans_mat = normalization_trans_mat.eval() * gt_trans_mat.eval() * normalization_trans_mat.inverse().eval();

	Eigen::MatrixXd gt_v2 = (v_2.rowwise().homogeneous() * gt_trans_mat.transpose()).leftCols(3);

	v_2 = v.bottomRows(v_2.rows());
	Eigen::MatrixXd tmp_v2 = v_2;
	Eigen::VectorXd rmse(iter_num);
	for (int i = 0; i < iter_num; ++i)
	{
		Eigen::Matrix4d affine_trans_mat;
		igl::readDMAT(trans_prefix + std::to_string(i) + ".dmat", affine_trans_mat);
		tmp_v2 = (v_2.rowwise().homogeneous() * affine_trans_mat.transpose()).leftCols(3);
		rmse(i) = BaseAlg::rmse(gt_v2, tmp_v2);
		std::cout << rmse(i) << std::endl;
		
	}
	igl::writeDMAT(output_rmse, rmse);
}

void convert_trans_to_rmse(const std::string& point_cloud_a, const std::string& point_cloud_b, const std::string& trans_matrix_fname, const std::string& scales_fname, const std::string& output_rmse)
{
	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(point_cloud_a, v_1, vc_1, vn_1);

	std::cout << (v_1.colwise().maxCoeff() - v_1.colwise().minCoeff()).norm() << std::endl;
	//system("pause");

	DataIO::read_ply(point_cloud_b, v_2, vc_2, vn_2);

	Eigen::Matrix4d gt_trans_mat;
	gt_trans_mat <<
		-0.1346397895, 0.1154806788, 0.9841424388, - 0.0000000000,
		0.9804971974, 0.1590265973, 0.1154806788, - 0.0000000000,
		- 0.1431690361, 0.9804971974, - 0.1346397895, - 0.0000000000,
		0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000;
	////21_rot_05
	//gt_trans_mat <<
	//	0.3396880587, 0.4689553595, - 0.8152870007, - 0.0000000000,
	//	- 0.7223851455, 0.6851892080, 0.0931421004, - 0.0000000000,
	//	0.6023053415, 0.5573119594, 0.5715169775, - 0.0000000000,
	//	0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000;

	Eigen::MatrixXd trans_mat;
	igl::readDMAT(trans_matrix_fname, trans_mat);

	Eigen::MatrixXd scales;
	igl::readDMAT(scales_fname, scales);

	int iter_num = scales.rows();

	Eigen::MatrixXd gt_v2 = (v_2.rowwise().homogeneous() * gt_trans_mat.transpose()).leftCols(3);
	std::cout << (gt_v2.colwise().maxCoeff() - gt_v2.colwise().minCoeff()).norm() << std::endl;
	DataIO::write_ply("gt_v2.ply", gt_v2, vc_2, vn_2);
	Eigen::VectorXd rmse(iter_num);
	for (int i = 0; i < iter_num; ++i)
	{
		double gt_ratio = 1/ scales(i);
		Eigen::Matrix4d affine_trans_mat = trans_mat.block<4, 4>(i * 4, 0);
		Eigen::MatrixXd v_2_copy = v_2;// *scales(i);
		Eigen::MatrixXd tmp_v2 = (v_2_copy.rowwise().homogeneous() * affine_trans_mat.transpose()).leftCols(3);
		DataIO::write_ply("result_" + std::to_string(i) + ".ply", tmp_v2, vc_2, vn_2);
		rmse(i) = BaseAlg::rmse(gt_v2, tmp_v2);
		double det = affine_trans_mat.determinant();
		std::cout << "No." << i << ": " << gt_ratio << ": " << std::pow(det, 1.0 / 3.0) << ": " << rmse(i) << std::endl;

	}
	igl::writeDMAT(output_rmse, rmse);
}

void main()
{
	//Eigen::MatrixXd v, vc, vn;
	//DataIO::read_ply("E:\\Data\\e9\\scene_dense.ply", v, vc, vn);
	//std::cout << v.rows() << " " << v.cols() << std::endl;

	//Eigen::MatrixXd down_sampled_v, down_sampled_vc, down_sample_vn;
	//down_sampled_v.resize(v.rows() / 100, 3);
	//down_sampled_vc.resize(v.rows() / 100, 3);
	//down_sample_vn.resize(v.rows() / 100, 3);
	//std::random_device rd;
	//std::uniform_int_distribution<> dist(0, v.rows() - 1);
	//std::vector<char> has_sampled(v.rows(), false);
	//for (int i = 0; i < down_sampled_v.rows(); ++i)
	//{
	//	int row_id = dist(rd);
	//	while(has_sampled[row_id])
	//	{
	//		row_id = dist(rd);
	//		printf("has sampled.\n");
	//	}
	//	has_sampled[row_id] = true;
	//	down_sampled_v.row(i) = v.row(row_id);
	//	down_sampled_vc.row(i) = vc.row(row_id);
	//	down_sample_vn.row(i) = vn.row(row_id);
	//}
	//DataIO::write_ply("downsampled_e9.ply", down_sampled_v, down_sampled_vc, down_sample_vn);
	//return;

	//icp_example("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", "E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply");

	//fgr_example("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", "fpfh");
	//fgr_example("out_e44_vn_trimmed.ply", "out_e55_vn_trimmed.ply", "shot", "shot_corres.dmat");
	//fgr_example("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_21_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_21_rot_05\\Depth_0001.ply", "fpfh", "");


	//std::string path = "/path/to/directory";
	//for (auto & p : fs::directory_iterator(path))
	//	std::cout << p << std::endl;

	//using namespace boost::filesystem;
	//path p("E:\\Projects\\UrbanReg\\build\\bin\\Release");   // p reads clearer than argv[1] in the following code

	//if (exists(p))    // does p actually exist?
	//{
	//	if (is_regular_file(p))        // is p a regular file?   
	//		std::cout << p << " size is " << file_size(p) << '\n';

	//	else if (is_directory(p))      // is p a directory?
	//		std::cout << p << "is a directory\n";

	//	else
	//		std::cout << p << "exists, but is neither a regular file nor a directory\n";
	//}
	//else
	//	std::cout << p << "does not exist\n";

	//for (directory_iterator itr(p); itr != directory_iterator(); ++itr)
	//{
	//	std::string temp_string = itr->path().filename().string();
	//	std::cout << temp_string << ' '; // display filename only
	//	if (is_regular_file(itr->status())) std::cout << " [" << file_size(itr->path()) << ']';
	//	std::cout << '\n';
	//}

	RegPipeline::PointCloudRegistrationUsingScaleFGR("fgr_test.xml");
	////RegPipeline::PointCloudRegistrationUsingGoICP("fgr_test.xml");

	//convert_trans_to_rmse("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_10_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_10_rot_05\\Depth_0001.ply", "affine_matrix.dmat", "scales.dmat", "rmse.dmat");



	//Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	//DataIO::read_ply("E:\\Documents\\Chinagraph2018\\data\\visual_comparison\\pairwise_no_noise_21_rot_05\\Scale-FGR\\gt_v2.ply", v_1, vc_1, vn_1);
	//DataIO::read_ply("E:\\Documents\\Chinagraph2018\\data\\visual_comparison\\pairwise_no_noise_21_rot_05\\Scale-FGR\\scale_fgr_v2.ply", v_2, vc_2, vn_2);
	//Eigen::VectorXd distances = (v_1 - v_2).rowwise().norm();
	////BaseAlg::find_nearest_neighbour(v_1, v_2, distances);

	////visualize the feature correspondences
	////std::cout << distances.maxCoeff() << std::endl;
	//igl::colormap(igl::COLOR_MAP_TYPE_PARULA, distances, 0, 0.02, vc_2);
	//vc_2 *= 255;
	//DataIO::write_ply("E:\\Documents\\Chinagraph2018\\data\\visual_comparison\\pairwise_no_noise_21_rot_05\\Scale-FGR\\scale_fgr_v2_colored.ply", v_2, vc_2, vn_2);

	//convert_trans_to_rmse("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_noise_xyz_level_02_01_rot_05\\Depth_0000.ply",
	//	"E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_noise_xyz_level_02_01_rot_05\\Depth_0001.ply",
	//	"trans", 200, "goicp.dmat");

	//Eigen::Matrix4d affine_trans_mat;
	//affine_trans_mat <<
	//	-0.1346397895, 0.1154806788, 0.9841424388, -0.0000000000,
	//	0.9804971974, 0.1590265973, 0.1154806788, -0.0000000000,
	//	-0.1431690361, 0.9804971974, -0.1346397895, -0.0000000000,
	//	0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000;
	////igl::readDMAT("affine_trans.dmat", affine_trans_mat);
	//std::cout << affine_trans_mat << std::endl;
	//Eigen::Matrix3d R = affine_trans_mat.block<3, 3>(0, 0);
	//Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	//DataIO::read_ply("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", v_1, vc_1, vn_1);
	//v_2 = (v_1.rowwise().homogeneous() * affine_trans_mat.transpose()).leftCols(3);
	//vn_2 = vn_1 * R.transpose();
	//DataIO::write_ply("goicp_aligned.ply", tmp_v2, vc_1, vn_2);
	//return;

	return;
}
