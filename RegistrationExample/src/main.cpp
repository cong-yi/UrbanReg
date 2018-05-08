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
#include <random>

void icp_example(const std::string& pointcloud_a, const std::string& pointcloud_b)
{
	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_a, v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_b, v_2, vc_2, vn_2);

	Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	v.topRows(v_1.rows()) = v_1;
	v.bottomRows(v_2.rows()) = v_2;

	FastGlobalRegistration::normalize(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), v);
	v_1 = v.topRows(v_1.rows());
	v_2 = v.bottomRows(v_2.rows());

	Eigen::MatrixXd aligned_v_2;
	FeatureAlg::icp(v_1, v_2, aligned_v_2);

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

	FastGlobalRegistration::normalize(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), v);
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
	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_a, v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_b, v_2, vc_2, vn_2);
	//v_2 *= 0.95;

	Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	v.topRows(v_1.rows()) = v_1;
	v.bottomRows(v_2.rows()) = v_2;

	FastGlobalRegistration::normalize(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), v);
	v_1 = v.topRows(v_1.rows());
	v_2 = v.bottomRows(v_2.rows());

  int sampling_num = -1e5;
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
				//igl::readDMAT("shot_1.dmat", feature_1);
				//igl::readDMAT("shot_2.dmat", feature_2);
				FeatureAlg::compute_shot(downsampled_v1, downsampled_vn1, downsampled_vc1, feature_1);
				FeatureAlg::compute_shot(downsampled_v2, downsampled_vn2, downsampled_vc2, feature_2);
			}
			igl::writeDMAT(feature_type + "_1.dmat", feature_1, false);
			igl::writeDMAT(feature_type + "_2.dmat", feature_2, false);
			FastGlobalRegistration::advanced_matching(downsampled_v1, downsampled_v2, feature_1, feature_2, corres);
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

	Eigen::MatrixXd vcolor;
	igl::jet(corres_v_1.col(1), true, vcolor);
	vcolor *= 255;

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
	//DataIO::read_ply("E:\\Data\\e44\\scene_dense.ply", v, vc, vn);
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
	//DataIO::write_ply("downsampled_e44.ply", down_sampled_v, down_sampled_vc, down_sample_vn);
	
	//icp_example("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", "E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply");

	//fgr_example("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", "E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", "fpfh");
	fgr_example("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", "E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply", "shot", "shot_corres.dmat");

	return;

	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	//DataIO::read_ply("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", v_1, vc_1, vn_1);
	//DataIO::read_ply("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", v_2, vc_2, vn_2);

	DataIO::read_ply("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", v_1, vc_1, vn_1);
	DataIO::read_ply("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply", v_2, vc_2, vn_2);

	return;
}
