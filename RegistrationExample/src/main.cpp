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

void goicp_example(const std::string& pointcloud_a, const std::string& pointcloud_b)
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
	GOICP::goicp(v_1, v_2, aligned_v_2);

	DataIO::write_ply("goicp_1.ply", v_1, vc_1, vn_1);
	DataIO::write_ply("goicp_2.ply", v_2, vc_2, vn_2);
	DataIO::write_ply("goicp_2_aligned.ply", aligned_v_2, vc_2, vn_2);
	return;
}

void fgr_example(const std::string& pointcloud_a, const std::string& pointcloud_b, const std::string& feature_type, const std::string& corres_file)
{
	clock_t begin = clock();
	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_a, v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_b, v_2, vc_2, vn_2);
	//v_2 *= 0.95;

	Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	v.topRows(v_1.rows()) = v_1;
	v.bottomRows(v_2.rows()) = v_2;

	BaseAlg::normalize(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), v);
	v_1 = v.topRows(v_1.rows());
	v_2 = v.bottomRows(v_2.rows());

  int sampling_num = 1e5;
  Eigen::MatrixXd downsampled_v1 = v_1, downsampled_v2 = v_2;
  Eigen::MatrixXd downsampled_vn1 = vn_1, downsampled_vn2 = vn_2;

  if (sampling_num > 0)
  {
	  downsampled_v1.conservativeResize(sampling_num, 3);
	  downsampled_v2.conservativeResize(sampling_num, 3);
	  downsampled_vn1.conservativeResize(sampling_num, 3);
	  downsampled_vn2.conservativeResize(sampling_num, 3);
  }

	Eigen::MatrixXd feature_1, feature_2;
	Eigen::MatrixXd output_v2;
	Eigen::MatrixXi corres_mat;
	Eigen::Matrix4d trans_mat;
	trans_mat.setIdentity();
	for (int t = 0; t < 1; ++t)
	{
		std::vector<std::pair<int, int> > corres;
		if(corres_file == "")
		{
			if (feature_type == "fpfh" || feature_type == "FPFH")
			{
				//igl::readDMAT("fpfh_1.dmat", feature_1);
				//igl::readDMAT("fpfh_2.dmat", feature_2);
				FeatureAlg::compute_fpfh(downsampled_v1, downsampled_vn1, feature_1);
				FeatureAlg::compute_fpfh(downsampled_v2, downsampled_vn2, feature_2);
			}
			else if (feature_type == "shot" || feature_type == "SHOT")
			{
				Eigen::MatrixXd downsampled_vc1 = vc_1, downsampled_vc2 = vc_2;
				if (sampling_num > 0)
				{
					downsampled_vc1.conservativeResize(sampling_num, 3);
					downsampled_vc2.conservativeResize(sampling_num, 3);
				}
				//igl::readDMAT("E:\\Documents\\report_0504\\data\\shot\\shot_1.dmat", feature_1);
				//igl::readDMAT("E:\\Documents\\report_0504\\data\\shot\\shot_2.dmat", feature_2);

				//std::cout << feature_1.rows() << std::endl;
				//feature_1.conservativeResize(sampling_num, 3);
				//feature_2.conservativeResize(sampling_num, 3);
				FeatureAlg::compute_shot(downsampled_v1, downsampled_vn1, downsampled_vc1, feature_1);
				FeatureAlg::compute_shot(downsampled_v2, downsampled_vn2, downsampled_vc2, feature_2);
			}
			//igl::writeDMAT(feature_type + "_1.dmat", feature_1, false);
			//igl::writeDMAT(feature_type + "_2.dmat", feature_2, false);

			Eigen::Array<bool, Eigen::Dynamic, 1> feature_1_mask = feature_1.col(0).array().isNaN() == false;
			Eigen::Array<bool, Eigen::Dynamic, 1> feature_2_mask = feature_2.col(0).array().isNaN() == false;

			int num_1 = feature_1_mask.count();
			std::cout << "NaN vertices in point cloud 1: " << feature_1.rows() - num_1 << std::endl;
			int num_2 = feature_2_mask.count();
			std::cout << "NaN vertices in point cloud 2: " << feature_2.rows() - num_2 << std::endl;

			Eigen::MatrixXd filtered_features(num_1 + num_2, feature_1.cols());

			filtered_features.topRows(num_1) = igl::slice_mask(feature_1, feature_1_mask, 1);
			filtered_features.bottomRows(num_2) = igl::slice_mask(feature_2, feature_2_mask, 1);

			std::cout << "start pca computation" << std::endl;

			Eigen::VectorXd eigenvalues;
			Eigen::MatrixXd eigenvectors;

			BaseAlg::pca(filtered_features, 50, eigenvalues, eigenvectors);
			Eigen::MatrixXd feature_pca = filtered_features * eigenvectors;

			std::cout << "end pca computation" << std::endl;

			feature_1 = feature_pca.topRows(num_1);
			feature_2 = feature_pca.bottomRows(num_2);

			Eigen::MatrixXd filtered_v1 = igl::slice_mask(downsampled_v1, feature_1_mask, 1);
			Eigen::MatrixXd filtered_v2 = igl::slice_mask(downsampled_v2, feature_2_mask, 1);

			FastGlobalRegistration::advanced_matching(filtered_v1, filtered_v2, feature_1, feature_2, corres);
			//An O(n) method to shift back to the original vertex indices before filering NaN elements
			Eigen::VectorXi offset_1(feature_1.rows());
			int id_offset = 0;
			for (int i = 0, j = 0; i < feature_1_mask.size(); ++i)
			{
				if (!feature_1_mask(i))
				{
					++id_offset;
					continue;
				}
				offset_1(j) = id_offset;
				++j;
			}

			Eigen::VectorXi offset_2(feature_2.rows());
			id_offset = 0;
			for (int i = 0, j = 0; i < feature_2_mask.size(); ++i)
			{
				if (!feature_2_mask(i))
				{
					++id_offset;
					continue;
				}
				offset_2(j) = id_offset;
				++j;
			}

			for(auto & ele : corres)
			{
				ele.first += offset_1(ele.first);
				ele.second += offset_2(ele.second);
			}
			
			corres_mat.resize(corres.size(), 2);
			for (int i = 0; i < corres.size(); ++i)
			{
				corres_mat(i, 0) = corres[i].first;
				corres_mat(i, 1) = corres[i].second;
			}
			igl::writeDMAT(feature_type + "_corres.dmat", corres_mat);
		}
		else
		{
			igl::readDMAT(corres_file, corres_mat);
			corres.resize(corres_mat.rows());
			for (int i = 0; i < corres_mat.rows(); ++i)
			{
				corres[i] = std::pair<int, int>(corres_mat(i, 0), corres_mat(i, 1));
			}
		}

		Eigen::Matrix4d last_trans = trans_mat;
		FastGlobalRegistration::optimize_pairwise(true, 128, downsampled_v1, downsampled_v2, corres, trans_mat);
		trans_mat = trans_mat.eval() * last_trans;
		double s = std::pow(trans_mat.determinant(), 1.0 / 3);
		std::cout << trans_mat << std::endl;
		std::cout << s << std::endl;

		Eigen::Affine3d affine_trans(trans_mat);
		Eigen::MatrixXd tmp_vn(vn_2.rows(), 3);
		output_v2.resize(v_2.rows(), 3);
		for (int i = 0; i < output_v2.rows(); ++i)
		{
			Eigen::Vector3d tmp_v = v_2.row(i).transpose();
			output_v2.row(i) = (affine_trans * tmp_v).transpose();
			tmp_vn.row(i) = (affine_trans * (tmp_v + vn_2.row(i).transpose())).transpose() - output_v2.row(i);
			tmp_vn.row(i).normalize();
		}
		downsampled_v2 = output_v2;
		downsampled_vn2 = tmp_vn;
		if(sampling_num > 0)
		{
			downsampled_v2.conservativeResize(sampling_num, 3);
			downsampled_vn2.conservativeResize(sampling_num, 3);
		}
	}

	Eigen::MatrixXd corres_v_1, corres_vc_1, corres_vn_1, corres_v_2, corres_vc_2, corres_vn_2, corres_v_aligned;
	corres_v_1 = igl::slice(downsampled_v1, corres_mat.col(0), 1);
	corres_vc_1 = igl::slice(vc_1, corres_mat.col(0), 1);
	corres_vn_1 = igl::slice(vn_1, corres_mat.col(0), 1);
	corres_v_2 = igl::slice(v_2, corres_mat.col(1), 1);
	corres_vc_2 = igl::slice(vc_2, corres_mat.col(1), 1);
	corres_vn_2 = igl::slice(vn_2, corres_mat.col(1), 1);
	corres_v_aligned = igl::slice(output_v2, corres_mat.col(1), 1);
	
	//visualize the feature correspondences
	Eigen::MatrixXd vcolor;
	igl::jet(corres_v_1.col(1), true, vcolor);
	vcolor *= 255;

	std::cout << (clock() - begin) / CLOCKS_PER_SEC << "s" << std::endl;
	DataIO::write_ply(feature_type+"_feature_1.ply", corres_v_1, vcolor, corres_vn_1);
	igl::writeDMAT(feature_type + "_feature_1.dmat", corres_v_1);
	DataIO::write_ply(feature_type + "_feature_2.ply", corres_v_2, vcolor, corres_vn_2);
	igl::writeDMAT(feature_type + "_feature_2.dmat", corres_v_2);
	DataIO::write_ply(feature_type + "_feature_3.ply", corres_v_aligned, vcolor, corres_vn_2);
	igl::writeDMAT(feature_type + "_feature_3.dmat", corres_v_2);

	DataIO::write_ply(feature_type + "_fgr_1.ply", v_1, vc_1, vn_1);
	DataIO::write_ply(feature_type + "_fgr_2.ply", v_2, vc_2, vn_2);
	DataIO::write_ply(feature_type + "_fgr_2_aligned.ply", output_v2, vc_2, vn_2);

	return;
}

void main()
{
	//Eigen::MatrixXd v, vc, vn;
	//DataIO::read_ply("E:\\Data\\m4\\scene_dense.ply", v, vc, vn);
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
	//DataIO::write_ply("downsampled_m4.ply", down_sampled_v, down_sampled_vc, down_sample_vn);
	//return;
	//icp_example("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", "E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply");

	//fgr_example("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", "fpfh");
	fgr_example("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_m3_trimmed.ply", "E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_m4_trimmed.ply", "shot", "");

	return;

	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	//DataIO::read_ply("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", v_1, vc_1, vn_1);
	//DataIO::read_ply("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", v_2, vc_2, vn_2);

	DataIO::read_ply("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", v_1, vc_1, vn_1);
	DataIO::read_ply("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply", v_2, vc_2, vn_2);

	return;
}
