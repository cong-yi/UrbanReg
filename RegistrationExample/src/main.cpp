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
#include "GoICP/include/goicp.h"
#include <random>

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

void fgr_example(const std::string& pointcloud_a, const std::string& pointcloud_b, const std::string& feature_type)
{
	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_a, v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_b, v_2, vc_2, vn_2);
  v_2 *= 0.95;

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
    downsampled_v1 = downsampled_v1.topRows(sampling_num);
    downsampled_v2 = downsampled_v2.topRows(sampling_num);
    downsampled_vn1 = downsampled_vn1.topRows(sampling_num);
    downsampled_vn2 = downsampled_vn2.topRows(sampling_num);
  }

	Eigen::MatrixXd feature_1, feature_2;

	if(feature_type == "fpfh" || feature_type == "FPFH")
	{
		//igl::readDMAT("fpfh_1.dmat", feature_1);
		//igl::readDMAT("fpfh_2.dmat", feature_2);
		FeatureAlg::compute_fpfh(downsampled_v1, downsampled_vn1, feature_1);
		igl::writeDMAT("fpfh_1.dmat", feature_1, false);
		FeatureAlg::compute_fpfh(downsampled_v2, downsampled_vn2, feature_2);
		igl::writeDMAT("fpfh_2.dmat", feature_2, false);
	}
	else if(feature_type == "shot" || feature_type == "SHOT")
	{
    Eigen::MatrixXd downsampled_vc1 = vc_1, downsampled_vc2 = vc_2;
    if (sampling_num > 0)
    {
      downsampled_vc1 = vc_1.topRows(sampling_num);
      downsampled_vc2 = vc_2.topRows(sampling_num);
    }
		//igl::readDMAT("shot_1.dmat", feature_1);
		//igl::readDMAT("shot_2.dmat", feature_2);
		FeatureAlg::compute_shot(downsampled_v1, downsampled_vn1, downsampled_vc1, feature_1);
		igl::writeDMAT("shot_1.dmat", feature_1, false);
		FeatureAlg::compute_shot(downsampled_v2, downsampled_vn2, downsampled_vc2, feature_2);
		igl::writeDMAT("shot_2.dmat", feature_2, false);
	}

	std::vector<std::pair<int, int> > corres;
	FastGlobalRegistration::advanced_matching(downsampled_v1, downsampled_v2, feature_1, feature_2, corres);

	Eigen::MatrixXi corres_mat(corres.size(), 2);
	for (int i = 0; i < corres.size(); ++i)
	{
		corres_mat(i, 0) = corres[i].first;
		corres_mat(i, 1) = corres[i].second;
	}
	igl::writeDMAT("corres.dmat", corres_mat);

	Eigen::Matrix4d trans_mat;
	FastGlobalRegistration::optimize_pairwise(true, 64, downsampled_v1, downsampled_v2, corres, trans_mat);
  double s = std::pow(trans_mat.determinant(), 1.0 / 3);
  std::cout << trans_mat << std::endl;
  std::cout << s << std::endl;

	Eigen::Affine3d affine_trans(trans_mat);
	Eigen::MatrixXd output_v2;
	output_v2.resize(v_2.rows(), 3);
	for (int i = 0; i < output_v2.rows(); ++i)
	{
		Eigen::Vector3d tmp_v = v_2.row(i).transpose();
		output_v2.row(i) = (affine_trans * tmp_v).transpose();
	}

	Eigen::MatrixXd corres_v_1, corres_vc_1, corres_vn_1, corres_v_2, corres_vc_2, corres_vn_2;
	corres_v_1 = igl::slice(downsampled_v1, corres_mat.col(0), 1);
	corres_vc_1 = igl::slice(vc_1, corres_mat.col(0), 1);
	corres_vn_1 = igl::slice(vn_1, corres_mat.col(0), 1);
	corres_v_2 = igl::slice(downsampled_v2, corres_mat.col(1), 1);
	corres_vc_2 = igl::slice(vc_2, corres_mat.col(1), 1);
	corres_vn_2 = igl::slice(vn_2, corres_mat.col(1), 1);
	DataIO::write_ply("feature_1.ply", corres_v_1, corres_vc_1, corres_vn_1);
	DataIO::write_ply("feature_2.ply", corres_v_2, corres_vc_2, corres_vn_2);

	DataIO::write_ply("fgr_1.ply", v_1, vc_1, vn_1);
	DataIO::write_ply("fgr_2.ply", v_2, vc_2, vn_2);
	DataIO::write_ply("fgr_2_aligned.ply", output_v2, vc_2, vn_2);

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
	
	//goicp_example("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", "E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply");

	fgr_example("D:\\reg\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", "D:\\reg\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", "fpfh");

	return;

	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	//DataIO::read_ply("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0000.ply", v_1, vc_1, vn_1);
	//DataIO::read_ply("E:\\Projects\\FastGlobalRegistration\\dataset\\pairwise_no_noise_01_rot_05\\Depth_0001.ply", v_2, vc_2, vn_2);

	DataIO::read_ply("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e44_vn_trimmed.ply", v_1, vc_1, vn_1);
	DataIO::read_ply("E:\\Projects\\UrbanReg\\build\\bin\\Release\\out_e55_vn_trimmed.ply", v_2, vc_2, vn_2);

	return;
}
