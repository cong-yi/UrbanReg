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
#include "igl/readOBJ.h"
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

void convert_batch_files_to_vis_data()
{
	Eigen::MatrixXd v1, vc1, vn1, v2, vc2, vn2;
	DataIO::read_ply("E:\\Documents\\Chinagraph2018\\new_data\\pca_validation\\620\\aligned_0.ply", v1, vc1, vn1);
	DataIO::read_ply("E:\\Documents\\Chinagraph2018\\new_data\\pca_validation\\620\\aligned_1.ply", v2, vc2, vn2);
	//Eigen::VectorXi downsampled_id_1, downsampled_id_2;
	//igl::readDMAT("E:\\Projects\\UrbanReg\\build\\bin\\Release\\downsample_ids_0.dmat", downsampled_id_1);
	//igl::readDMAT("E:\\Projects\\UrbanReg\\build\\bin\\Release\\downsample_ids_1.dmat", downsampled_id_2);

	//Eigen::MatrixXd downsampled_v1 = igl::slice(v1, downsampled_id_1, 1);
	//Eigen::MatrixXd downsampled_v2 = igl::slice(v2, downsampled_id_2, 1);

	namespace fs = boost::filesystem;

	std::map<int, double> ratio_map;
	std::map<int, double> pca_time_map;
	std::map<int, double> matching_time_map;
	std::map<int, double> feature_time_map;

	std::string path = "E:\\Documents\\Chinagraph2018\\new_data\\fpfh_validation";
	for (auto & p : fs::directory_iterator(path))
	{
		//std::cout << p << std::endl;
		if (!fs::is_directory(p.path()))
		{

			continue;
		}

		for (fs::directory_iterator itr(p); itr != fs::directory_iterator(); ++itr)
		{
			std::string temp_string = itr->path().filename().string();

			std::cout << temp_string << std::endl;

			if (temp_string == "shot_0_1_corres.dmat" || temp_string == "fpfh_0_1_corres.dmat")
			{
				std::cout << std::stoi(p.path().filename().string()) << std::endl;;
				Eigen::VectorXi downsampled_id_1, downsampled_id_2;
				igl::readDMAT(p.path().string() + "\\" + "downsample_ids_0.dmat", downsampled_id_1);
				igl::readDMAT(p.path().string() + "\\" + "downsample_ids_1.dmat", downsampled_id_2);

				Eigen::MatrixXd downsampled_v1 = igl::slice(v1, downsampled_id_1, 1);
				Eigen::MatrixXd downsampled_v2 = igl::slice(v2, downsampled_id_2, 1);
				Eigen::MatrixXi corres_mat;
				std::cout << itr->path().string() << std::endl;
				igl::readDMAT(itr->path().string(), corres_mat);
				const Eigen::MatrixXd corres_v_1 = igl::slice(downsampled_v1, corres_mat.col(0), 1);
				const Eigen::MatrixXd corres_v_2 = igl::slice(downsampled_v2, corres_mat.col(1), 1);
				Eigen::VectorXd distances = (corres_v_1 - corres_v_2).rowwise().norm();
				double accept_num = (distances.array() < 2e-2).count();
				ratio_map[std::stoi(p.path().filename().string())] = accept_num / corres_mat.rows();
				std::cout << ratio_map[std::stoi(p.path().filename().string())] << std::endl;
			}
			else if (temp_string == "ratio.txt")
			{
				std::ifstream in(p.path().string() + "\\" + "ratio.txt");
				std::string word;
				//in.open(p.path().string() + "\\" + "ratio.txt", std::fstream::in);
				if (in.is_open())
				{
					getline(in, word);
					in >> word;
					word.pop_back();
					pca_time_map[std::stoi(p.path().filename().string())] = std::stod(word);
					getline(in, word);
					in >> word;
					word.pop_back();
					matching_time_map[std::stoi(p.path().filename().string())] = std::stod(word);
					getline(in, word);
					in >> word;
					word.pop_back();
					feature_time_map[std::stoi(p.path().filename().string())] = std::stod(word);
					in.close();
				}
			}
		}
	}

	Eigen::MatrixXd ratios(ratio_map.size(), 2);
	int counter = 0;
	for (auto& ele : ratio_map)
	{
		ratios(counter, 0) = ele.first;
		ratios(counter, 1) = ele.second;
		counter++;
	}

	Eigen::MatrixXd pca_time_mat(pca_time_map.size(), 2);
	counter = 0;
	for (auto& ele : pca_time_map)
	{
		pca_time_mat(counter, 0) = ele.first;
		pca_time_mat(counter, 1) = ele.second;
		counter++;
	}

	Eigen::MatrixXd matching_time_mat(matching_time_map.size(), 2);
	counter = 0;
	for (auto& ele : matching_time_map)
	{
		matching_time_mat(counter, 0) = ele.first;
		matching_time_mat(counter, 1) = ele.second;
		counter++;
	}

	Eigen::MatrixXd feature_time_mat(feature_time_map.size(), 2);
	counter = 0;
	for (auto& ele : feature_time_map)
	{
		feature_time_mat(counter, 0) = ele.first;
		feature_time_mat(counter, 1) = ele.second;
		counter++;
	}

	igl::writeDMAT(path +"\\ratios.dmat", ratios);
	igl::writeDMAT(path + "\\pca_time.dmat", pca_time_mat);
	igl::writeDMAT(path + "\\matching_time.dmat", matching_time_mat);
	igl::writeDMAT(path + "\\feature_time.dmat", feature_time_mat);
}

Eigen::Matrix4d read_gt_mat(const std::string& filename)
{
	std::ifstream in(filename);
	std::string word;
	Eigen::Matrix4d trans_mat;
	if (in.is_open())
	{
		getline(in, word);
		for(int i = 0; i < 15; ++i)
		{
			in >> word;
			trans_mat(i / 4, i % 4) = std::stod(word);
		}
	}
	return trans_mat;
}

void main()
{
	//Eigen::MatrixXd v1, vc1, vn1;
	//Eigen::MatrixXi f1;
	//DataIO::read_ply("G:\\scene_dense.ply", v1, vc1, vn1);
	//std::cout << "read ply finished" << std::endl;
	//Eigen::VectorXi downsampled_id = BaseAlg::downsampling(v1.rows(), 5e7);
	//Eigen::MatrixXd downsampled_v = igl::slice(v1, downsampled_id, 1);
	//Eigen::MatrixXd downsampled_vc = igl::slice(vc1, downsampled_id, 1);
	//Eigen::MatrixXd downsampled_vn = igl::slice(vn1, downsampled_id, 1);
	//DataIO::write_ply("E:\\scene_downsampled.ply", downsampled_v, downsampled_vc, downsampled_vn);
	//Eigen::Matrix4d gt_trans_mat;
	//gt_trans_mat <<
	//	1.356538418140138, -0.074346534882401, 0.061055720092299544, -0.11025508026773478,
	//	0.07333601385607247, 1.3577557494068475, 0.023934120403920928, -2.587773141743163,
	//	-0.06226585964002496, -0.02058168600448811, 1.3583633700227857, 1.256938874536025,
	//	0.0, 0.0, 0.0, 1.0;
	//	//-1.1269964793895708, 0.04269661470689909, -0.019881248254682987, 0.013554839733725007,
	//	//-0.047093043829193895, -1.0143285310518328, 0.4911814387914244, -0.8880703493137462,
	//	//0.0007142566033271028, 0.49158311574227515, 1.0152265065741248, 0.44941749834308586,
	//	//0.0, 0.0, 0.0, 1.0;
	//std::cout << gt_trans_mat.determinant() << std::endl;
	//
	//std::cout<< std::pow(abs(gt_trans_mat.determinant()), 1 / 3.0) <<std::endl;
	////system("pause");
	//	///////////////1
	//	//-0.265451995651203, 0.07511910571368165, 0.18013673235734284, 0.0009126750199021233,
	//	//-0.010136471391022998, 0.2983812102152643, -0.13936566909043147, 0.2511295889132222, 
	//	//0.19490863073589945, 0.11782469066397497, 0.23808580872745405, 0.2785980678803347,
	//	//0.0, 0.0, 0.0, 1.0;
	//v1 = (v1.eval().rowwise().homogeneous() * gt_trans_mat.transpose()).leftCols(3);

	//DataIO::write_ply("E:\\Models\\cloud_and_poses2_aligned.ply", v1, vc1, vn1);
	//system("pause");

	//Eigen::MatrixXd v, vc, vn;
	//DataIO::read_ply("E:\\Documents\\Chinagraph2018\\new_data\\cross_data\\Caterpillar\\Caterpillar_COLMAP_cropped.ply", v, vc, vn);
	//Eigen::Matrix4d gt_trans_mat;
	//gt_trans_mat <<
	//	1.033308928464703236e+00, 4.611365568720580260e-02, -1.176860629313055240e+00, 2.060646395531264830e+00,
	//	1.177421469408414634e+00, -7.818705290307090272e-02, 1.030737706619544447e+00, -5.257332513306570698e-01,
	//	-2.839178479957314705e-02, -1.564165933406792774e+00, -8.621827554759166345e-02, 1.575946917537108618e+02,
	//	0.0000000000, 0.0000000000, 0.0000000000, 1.0000000000;
	//Eigen::MatrixXd aligned_v = (v.rowwise().homogeneous() * gt_trans_mat.transpose()).leftCols(3);
	//Eigen::MatrixXd aligned_vn = (vn * gt_trans_mat.block<3, 3>(0, 0)).eval().leftCols(3);
	//DataIO::write_ply("E:\\Documents\\Chinagraph2018\\new_data\\cross_data\\Caterpillar\\Caterpillar_COLMAP_gt.ply", aligned_v, vc, aligned_vn);
	//return;
	////Eigen::VectorXi downsampled_id = BaseAlg::downsampling(v.rows(), 1000000);
	////Eigen::MatrixXd down_sampled_v = igl::slice(v, downsampled_id, 1);
	////Eigen::MatrixXd down_sampled_vc = igl::slice(vc, downsampled_id, 1);
	////Eigen::MatrixXd down_sample_vn = igl::slice(vn, downsampled_id, 1);
	////DataIO::write_ply("E:\\Documents\\Chinagraph2018\\new_data\\cross_data\\Caterpillar\\Caterpillar_relax_1m.ply", down_sampled_v, down_sampled_vc, down_sample_vn);
	////return;
	//system("pause");

	//icp_example("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", "E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply");

	//fgr_example("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", "fpfh");
	//fgr_example("out_e44_vn_trimmed.ply", "out_e55_vn_trimmed.ply", "shot", "shot_corres.dmat");
	//fgr_example("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_21_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_21_rot_05\\Depth_0001.ply", "fpfh", "");

	namespace fs = boost::filesystem;

	std::string path = "E:\\Projects\\FastGlobalRegistration\\dataset";
	std::string output_path = "E:\\Documents\\Chinagraph2018\\new_data\\fgr_dataset_performance";
	int step_num = 20;
	double scaling_step = std::pow(9.0, 1.0/step_num);
	for (auto & p : fs::directory_iterator(path))
	{
		//std::cout << p << std::endl;
		if (!fs::is_directory(p.path()))
		{
			continue;
		}
		std::string path_string = p.path().string();
		std::cout << path_string << std::endl;
		Eigen::MatrixXd v_source, vc_source, vn_source;
		DataIO::read_ply(path_string + "\\Depth_0001.ply", v_source, vc_source, vn_source);
		Eigen::Matrix4d gt_mat = read_gt_mat(path_string + "\\gt.log");
		Eigen::MatrixXd gt_v = (v_source.rowwise().homogeneous() * gt_mat.transpose()).leftCols(3);
		DataIO::clear_config_node("fgr_test.xml", "point_cloud", "filename");
		DataIO::add_config_node("fgr_test.xml", "point_cloud", "filename", path_string + "\\Depth_0000.ply");
		DataIO::add_config_node("fgr_test.xml", "point_cloud", "filename", path_string + "\\Depth_0001.ply");
		for(int i = 0; i < step_num + 1; ++i)
		{
			if (!boost::filesystem::exists(output_path + "\\" + std::to_string(i)))    // does p actually exist?
			{
				std::cout << output_path + "\\" + std::to_string(i) << " does not exist\n";
				boost::filesystem::create_directory(output_path + "\\" + std::to_string(i));
			}
			double scaling_factor = 1 / 3.0 * std::pow(scaling_step, i);
			DataIO::change_config_attribute("fgr_test.xml", "output_folder", "name", output_path + "\\" + std::to_string(i) + "\\" + p.path().filename().string());
			DataIO::change_config_attribute("fgr_test.xml", "parameters", "data_scaling_factor", std::to_string(scaling_factor));
			RegPipeline::MultiwayPointCloudRegistrationUsingScaleFGR("fgr_test.xml");
			Eigen::MatrixXd v_1, vc_1, vn_1;
			DataIO::read_ply(output_path + "\\" + std::to_string(i) + "\\" + p.path().filename().string() + "\\aligned_1.ply", v_1, vc_1, vn_1);
			double rmse = BaseAlg::rmse(gt_v, v_1);
			std::cout << "rmse: " << rmse << std::endl;
			DataIO::write_file(output_path + "\\" + std::to_string(i) + "\\" + p.path().filename().string() + "\\rmse.txt", rmse);
		}

		
		//DataIO::change_config_attribute("fgr_test.xml", "parameters", "downsampling_num", std::to_string(i * 10000));
		//RegPipeline::MultiwayPointCloudRegistrationUsingScaleFGR("fgr_test.xml");
	}
	

	//RegPipeline::PointCloudRegistrationUsingScaleFGR("fgr_test.xml");

	//convert_batch_files_to_vis_data();
	//system("pause");

	//for(int i = 11; i < 21; ++i)
	//{
	//	DataIO::change_config_attribute("fgr_test.xml", "parameters", "downsampling_num", std::to_string(i * 10000));
	//	RegPipeline::MultiwayPointCloudRegistrationUsingScaleFGR("fgr_test.xml");
	//}
	//convert_batch_files_to_vis_data();
	//RegPipeline::MultiwayPointCloudRegistrationUsingScaleFGR("fgr_test.xml");

	////RegPipeline::PointCloudRegistrationUsingGoICP("fgr_test.xml");

	//convert_trans_to_rmse("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_10_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_10_rot_05\\Depth_0001.ply", "affine_matrix.dmat", "scales.dmat", "rmse.dmat");



	//Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2, gt_v, gt_vc, gt_vn;

	//DataIO::read_ply("E:\\Documents\\Chinagraph2018\\new_data\\cross_data\\Caterpillar\\sfgr\\aligned_0.ply", v_1, vc_1, vn_1);
	//DataIO::read_ply("E:\\Documents\\Chinagraph2018\\new_data\\cross_data\\Caterpillar\\sfgr\\aligned_1.ply", v_2, vc_2, vn_2);
	//DataIO::read_ply("E:\\Documents\\Chinagraph2018\\new_data\\cross_data\\Caterpillar\\sfgr\\Caterpillar_COLMAP_gt.ply", gt_v, gt_vc, gt_vn);


	//DataIO::write_ply("E:\\Documents\\Chinagraph2018\\new_data\\cross_data\\Caterpillar\\sfgr\\colorized_distances_1.ply", v_2, vc_2, vn_2);

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
