#include "registration_pipeline.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <DataIO/include/data_io.h>
#include <GPSLocalizer/include/gps.h>
#include <OverlapTrimmer/include/trimmer.h>
#include "GoICP/include/goicp.h"
#include "BaseAlg/include/base_alg.h"
#include "ScaleStretchICP/include/ssicp.h"
#include "FastGlobalRegistration/include/fast_global_registration.h"
#include "FeatureAlg/include/feature_alg.h"
#include <igl/writeDMAT.h>
#include <igl/slice.h>
#include <igl/jet.h>
#include <igl/readDMAT.h>
#include <igl/slice_mask.h>
#include <filesystem>
#include <boost/filesystem.hpp>
#include <igl/colormap.h>
#include <random>
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"

REGPIPELINE_PUBLIC void RegPipeline::LocalizePointCloud(const std::vector<std::string> &filenames, std::string &format)
{
	if (filenames.size() < 4) return;

	Eigen::MatrixXd A, C, N;
	if (format == "OBJ" || format == "obj")
	{
		Eigen::MatrixXi F;
		igl::readOBJ(filenames[0], A, F);
	}
	if (format == "PLY" || format == "ply")
	{
		DataIO::read_ply(filenames[0], A, C, N);
	}

	Eigen::MatrixXd L, G;
	Eigen::MatrixXi F;
	igl::readOBJ(filenames[1], L, F);
	igl::readOBJ(filenames[2], G, F);

	double s;
	Eigen::Matrix3d R;
	Eigen::RowVector3d T;
	GPS::CalcualteCamerasTransformation(L, G, s, R, T);
	Eigen::MatrixXd B = GPS::GetLocalizedPoints(A, s, R, T);

	if (format == "OBJ" || format == "obj")
		igl::writeOBJ(filenames[3], B, Eigen::MatrixXi());
	if (format == "PLY" || format == "ply")
	{
		DataIO::write_ply(filenames[3], B, C, N);
	}
}

REGPIPELINE_PUBLIC void RegPipeline::TrimPointsClouds(const std::vector<std::string> &filenames,
	std::string &format)
{
	if (filenames.size() < 4) return;

	Eigen::MatrixXd A, B, CA, CB, NA, NB;
	if (format == "OBJ" || format == "obj")
	{
		Eigen::MatrixXi F;
		igl::readOBJ(filenames[0], A, F);
		igl::readOBJ(filenames[1], B, F);
	}
	if (format == "PLY" || format == "ply")
	{
		DataIO::read_ply(filenames[0], A, CA, NA);
		DataIO::read_ply(filenames[1], B, CB, NB);
	}

	std::vector<int> indices_a(A.rows()), indices_b(B.rows());
	for (size_t i = 0; i < indices_a.size(); ++i)
		indices_a[i] = i;
	for (size_t i = 0; i < indices_b.size(); ++i)
		indices_b[i] = i;

	Trimmer::Trim(A, B, 0.05, indices_a, indices_b);

	Eigen::MatrixXd AA(indices_a.size(), 3), BB(indices_b.size(), 3);
	Eigen::MatrixXd CAA(indices_a.size(), 3), CBB(indices_b.size(), 3);
	Eigen::MatrixXd NAA(indices_a.size(), 3), NBB(indices_b.size(), 3);
	for (int i = 0; i < indices_a.size(); ++i)
	{
		AA.row(i) = A.row(indices_a[i]);
		CAA.row(i) = CA.row(indices_a[i]);
		NAA.row(i) = NA.row(indices_a[i]);
	}
	for (int i = 0; i < indices_b.size(); ++i)
	{
		BB.row(i) = B.row(indices_b[i]);
		CBB.row(i) = CB.row(indices_b[i]);
		NBB.row(i) = NB.row(indices_b[i]);
	}

	if (format == "OBJ" || format == "obj")
	{
		igl::writeOBJ(filenames[2], A, Eigen::MatrixXi());
		igl::writeOBJ(filenames[3], B, Eigen::MatrixXi());
	}
	if (format == "PLY" || format == "ply")
	{
		DataIO::write_ply(filenames[2], AA, CAA, NAA);
		DataIO::write_ply(filenames[3], BB, CBB, NBB);
	}
}

void RegPipeline::PointCloudRegistrationUsingGoICP(const std::string& config_filename)
{
	std::string feature_type;
	std::vector<std::string> pointcloud_filenames;
	std::vector<std::string> feature_filenames;
	std::string correspondences_filename;
	std::string output_folder;
	int downsampling_num;
	int pca_component_num;
	double data_scaling_factor;

	DataIO::read_fgr_config(config_filename, pointcloud_filenames, feature_type, feature_filenames, correspondences_filename, downsampling_num, pca_component_num, data_scaling_factor, output_folder);

	boost::filesystem::path output_path(output_folder);
	if (boost::filesystem::exists(output_path) && boost::filesystem::is_directory(output_path))    // does p actually exist?
	{
		std::cout << output_path << " exists\n";
	}
	else
	{
		std::cout << output_path << " does not exist\n";
	}
	system("pause");

	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_filenames[0], v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_filenames[1], v_2, vc_2, vn_2);

	v_2 = v_2 * data_scaling_factor;

	Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	v.topRows(v_1.rows()) = v_1;
	v.bottomRows(v_2.rows()) = v_2;

	Eigen::Matrix4d normalization_trans_mat = BaseAlg::normalize(Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1), v);
	v_1 = v.topRows(v_1.rows());
	v_2 = v.bottomRows(v_2.rows());

	Eigen::MatrixXd aligned_v_2;
	Eigen::Matrix4d trans_mat = GOICP::goicp(v_1, v_2, aligned_v_2);
	Eigen::Matrix4d ssicp_trans = SSICP::GetOptimalTrans(aligned_v_2, v_1, 1e-1, 1e-3);//trim_thre should be 1e-2

	//aligned_v_2 = (aligned_v_2.rowwise().homogeneous() * ssicp_trans.transpose()).eval().leftCols(3);
	trans_mat = normalization_trans_mat.inverse().eval() * ssicp_trans.eval() * trans_mat.eval() * normalization_trans_mat.eval();
	igl::writeDMAT("affine_matrix.dmat", trans_mat);
	return;
}

void RegPipeline::PointCloudRegistrationUsingScaleFGR(const std::string& config_filename)
{
	std::string feature_type;
	std::vector<std::string> pointcloud_filenames;
	std::vector<std::string> feature_filenames;
	std::string correspondences_filename;
	std::string output_folder;
	int downsampling_num;
	int pca_component_num;
	double data_scaling_factor;

	DataIO::read_fgr_config(config_filename, pointcloud_filenames, feature_type, feature_filenames, correspondences_filename, downsampling_num, pca_component_num, data_scaling_factor, output_folder);

	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_filenames[0], v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_filenames[1], v_2, vc_2, vn_2);

	Eigen::MatrixXd aligned_v2;
	Eigen::MatrixXd aligned_vn_2;

	Eigen::MatrixXd affine_matrix(4, 4);

	clock_t begin = clock();

	Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	v.topRows(v_1.rows()) = v_1;
	v.bottomRows(v_2.rows()) = v_2;
	Eigen::Vector3d min_corner = Eigen::Vector3d(-1, -1, -1);
	Eigen::Vector3d max_corner = Eigen::Vector3d(1, 1, 1);
	Eigen::Matrix4d normalization_trans_mat = BaseAlg::normalize(min_corner, max_corner, v);

	v_1 = v.topRows(v_1.rows());
	v_2 = v.bottomRows(v_2.rows());

	Eigen::MatrixXd downsampled_v1 = v_1, downsampled_v2 = v_2;
	Eigen::MatrixXd downsampled_vn1 = vn_1, downsampled_vn2 = vn_2;

	bool need_downsampling = false;
	if (downsampling_num > 0 && downsampling_num < v_1.rows() && downsampling_num < v_2.rows())
	{
		need_downsampling = true;
	}
	Eigen::VectorXi downsampled_ids_1;
	Eigen::VectorXi downsampled_ids_2;
	if (need_downsampling)
	{
		downsampled_ids_1 = BaseAlg::downsampling(v_1.rows(), downsampling_num);
		downsampled_ids_2 = BaseAlg::downsampling(v_2.rows(), downsampling_num);
	}

	if (need_downsampling)
	{
		downsampled_v1 = igl::slice(v_1, downsampled_ids_1, 1);
		downsampled_v2 = igl::slice(v_2, downsampled_ids_2, 1);
		downsampled_vn1 = igl::slice(vn_1, downsampled_ids_1, 1);
		downsampled_vn2 = igl::slice(vn_2, downsampled_ids_2, 1);
		//downsampled_v1.conservativeResize(downsampling_num, 3);
		//downsampled_v2.conservativeResize(downsampling_num, 3);
		//downsampled_vn1.conservativeResize(downsampling_num, 3);
		//downsampled_vn2.conservativeResize(downsampling_num, 3);
	}

	Eigen::MatrixXd feature_1, feature_2;
	Eigen::MatrixXi corres_mat;
	std::vector<std::pair<int, int> > corres;
	if (correspondences_filename == "")
	{
		if (feature_filenames[0] == "" || feature_filenames[1] == "")
		{
			if (feature_type == "fpfh" || feature_type == "FPFH")
			{
				FeatureAlg::compute_fpfh(downsampled_v1, downsampled_vn1, feature_1);
				FeatureAlg::compute_fpfh(downsampled_v2, downsampled_vn2, feature_2);
			}
			else if (feature_type == "shot" || feature_type == "SHOT")
			{
				std::cout << "compute shot..." << std::endl;
				Eigen::MatrixXd downsampled_vc1 = vc_1, downsampled_vc2 = vc_2;
				if (need_downsampling)
				{
					downsampled_vc1 = igl::slice(vc_1, downsampled_ids_1, 1);
					downsampled_vc2 = igl::slice(vc_2, downsampled_ids_2, 1);
					//downsampled_vc1.conservativeResize(downsampling_num, 3);
					//downsampled_vc2.conservativeResize(downsampling_num, 3);
				}
				FeatureAlg::compute_shot(downsampled_v1, v_1, vn_1, downsampled_vc1, vc_1, feature_1);
				FeatureAlg::compute_shot(downsampled_v2, v_2, vn_2, downsampled_vc2, vc_2, feature_2);
			}
			//igl::writeDMAT(feature_type + "_1.dmat", feature_1, false);
			//igl::writeDMAT(feature_type + "_2.dmat", feature_2, false);
		}
		else
		{
			igl::readDMAT(feature_filenames[0], feature_1);
			igl::readDMAT(feature_filenames[1], feature_2);
		}

		Eigen::Array<bool, Eigen::Dynamic, 1> feature_1_mask = feature_1.col(0).array().isNaN() == false;
		Eigen::Array<bool, Eigen::Dynamic, 1> feature_2_mask = feature_2.col(0).array().isNaN() == false;

		int num_1 = feature_1_mask.count();
		std::cout << "NaN vertices in point cloud 1: " << feature_1.rows() - num_1 << std::endl;
		int num_2 = feature_2_mask.count();
		std::cout << "NaN vertices in point cloud 2: " << feature_2.rows() - num_2 << std::endl;

		Eigen::MatrixXd filtered_features(num_1 + num_2, feature_1.cols());

		filtered_features.topRows(num_1) = igl::slice_mask(feature_1, feature_1_mask, 1);
		filtered_features.bottomRows(num_2) = igl::slice_mask(feature_2, feature_2_mask, 1);
		if (pca_component_num > 0)
		{
			std::cout << "start pca computation" << std::endl;

			Eigen::VectorXd eigenvalues;
			Eigen::MatrixXd eigenvectors;

			BaseAlg::pca(filtered_features, pca_component_num, eigenvalues, eigenvectors);

			Eigen::MatrixXd feature_pca = filtered_features * eigenvectors;

			std::cout << "end pca computation" << std::endl;

			feature_1 = feature_pca.topRows(num_1);
			feature_2 = feature_pca.bottomRows(num_2);
		}

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

		for (auto & ele : corres)
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

		////visualize the feature correspondences
		//Eigen::MatrixXd feature_vcolor;
		//Eigen::MatrixXd feature_v1 = igl::slice(downsampled_v1, corres_mat.col(0), 1);
		//Eigen::MatrixXd feature_v2 = igl::slice(downsampled_v2, corres_mat.col(1), 1);
		//igl::colormap(igl::COLOR_MAP_TYPE_PARULA, feature_v1, true, feature_vcolor);
		//feature_vcolor *= 255;
		//DataIO::write_ply(feature_type + "_feature_1.ply", corres_v_1, vcolor, corres_vn_1);
	}
	else
	{
		igl::readDMAT(correspondences_filename, corres_mat);
		corres.resize(corres_mat.rows());
		for (int i = 0; i < corres_mat.rows(); ++i)
		{
			corres[i] = std::pair<int, int>(corres_mat(i, 0), corres_mat(i, 1));
		}
	}
	std::cout << (clock() - begin) / (double)CLOCKS_PER_SEC << "s" << std::endl;
	begin = clock();
	Eigen::Matrix4d trans_mat;
	FastGlobalRegistration::optimize_pairwise(true, 128, downsampled_v1, downsampled_v2, corres, trans_mat);
	std::cout << "FGR: " << (clock() - begin) / (double)CLOCKS_PER_SEC << "s" << std::endl;

	double s = std::pow(trans_mat.determinant(), 1.0 / 3);
	std::cout << trans_mat << std::endl;
	std::cout << s << std::endl;

	begin = clock();
	aligned_v2 = (v_2.rowwise().homogeneous() * trans_mat.transpose()).eval().leftCols(3);
	aligned_vn_2 = vn_2 * trans_mat.block<3, 3>(0, 0).transpose();

	Eigen::Matrix4d ssicp_trans = SSICP::GetOptimalTrans(aligned_v2, v_1, 1e-2, 1e-3);

	std::cout << "SSICP: " << (clock() - begin) / (double)CLOCKS_PER_SEC << "s" << std::endl;
	aligned_v2 = (aligned_v2.rowwise().homogeneous() * ssicp_trans.transpose()).eval().leftCols(3);

	affine_matrix = normalization_trans_mat.inverse().eval() * ssicp_trans * trans_mat * normalization_trans_mat.eval();

	std::cout << affine_matrix << std::endl;
	s = std::pow(affine_matrix.determinant(), 1.0 / 3);
	std::cout << "scaling factor: " << s << std::endl;

	igl::writeDMAT("affine_matrix.dmat", affine_matrix);

	Eigen::MatrixXd corres_v_1, corres_vn_1, corres_v_2, corres_vn_2, corres_v_aligned;
	corres_v_1 = igl::slice(downsampled_v1, corres_mat.col(0), 1);
	corres_vn_1 = igl::slice(downsampled_vn1, corres_mat.col(0), 1);
	corres_v_2 = igl::slice(downsampled_v2, corres_mat.col(1), 1);
	corres_vn_2 = igl::slice(downsampled_vn2, corres_mat.col(1), 1);
	corres_v_aligned = igl::slice(aligned_v2, corres_mat.col(1), 1);

	//visualize the feature correspondences
	Eigen::MatrixXd vcolor;
	igl::jet(corres_v_1.col(0), true, vcolor);
	vcolor *= 255;

	DataIO::write_ply(feature_type + "_feature_1.ply", corres_v_1, vcolor, corres_vn_1);
	DataIO::write_ply(feature_type + "_feature_2.ply", corres_v_2, vcolor, corres_vn_2);
	DataIO::write_ply(feature_type + "_feature_2_aligned.ply", corres_v_aligned, vcolor, corres_vn_2);

	DataIO::write_ply(feature_type + "_fgr_1.ply", v_1, vc_1, vn_1);
	DataIO::write_ply(feature_type + "_fgr_2.ply", v_2, vc_2, vn_2);
	DataIO::write_ply(feature_type + "_fgr_2_aligned.ply", aligned_v2, vc_2, aligned_vn_2);

	return;
}

void RegPipeline::MultiwayPointCloudRegistrationUsingScaleFGR(const std::string& config_filename)
{
	std::string feature_type;
	std::vector<std::string> pointcloud_filenames;
	std::vector<std::string> feature_filenames;
	std::string correspondences_filename;
	std::string output_folder;
	int downsampling_num;
	int pca_component_num;
	double data_scaling_factor;

	DataIO::read_fgr_config(config_filename, pointcloud_filenames, feature_type, feature_filenames, correspondences_filename, downsampling_num, pca_component_num, data_scaling_factor, output_folder);
	std::cout << output_folder << std::endl;

	//boost::filesystem::path output_path(output_folder + "\\" + std::to_string(downsampling_num));
	boost::filesystem::path output_path(output_folder);
	if (boost::filesystem::exists(output_path) && boost::filesystem::is_directory(output_path))    // does p actually exist?
	{
		std::cout << output_path << " exists\n";
		return;
	}
	else
	{
		std::cout << output_path << " does not exist\n";
		boost::filesystem::create_directory(output_path);
	}
	
	int point_cloud_num = pointcloud_filenames.size();

	std::map<int, Eigen::MatrixXd> v_map, vc_map, vn_map;

	Eigen::MatrixXd pc_colors;
	Eigen::VectorXd pc_ids = BaseAlg::generate_random_ids(pointcloud_filenames.size()).cast<double>();

	igl::jet(pc_ids, true, pc_colors);
	pc_colors *= 255;

	for(int i = 0; i < point_cloud_num; ++i)
	{
		v_map[i] = Eigen::MatrixXd();
		vc_map[i] = Eigen::MatrixXd();
		vn_map[i] = Eigen::MatrixXd();
		DataIO::read_ply(pointcloud_filenames[i], v_map[i], vc_map[i], vn_map[i]);
		if(vc_map[i].isZero())
		{
			vc_map[i].setOnes();
			vc_map[i].col(0) *= pc_colors(i, 0);
			vc_map[i].col(1) *= pc_colors(i, 1);
			vc_map[i].col(2) *= pc_colors(i, 2);
		}
	}
	v_map[1] *= data_scaling_factor;
	//for (int i = 0; i < point_cloud_num; ++i)
	//{
	//	DataIO::write_ply(output_path.string() + "\\source_" + std::to_string(i) + ".ply", v_map[i], vc_map[i], vn_map[i]);
	//}

	clock_t begin = clock();
	bool need_downsampling = true;
	if (downsampling_num < 0)
	{
		need_downsampling = false;
	}
	for(int i = 0; i < pointcloud_filenames.size(); ++i)
	{
		if (downsampling_num >= v_map[i].rows())
		{
			need_downsampling = false;
		}
	}

	//Eigen::MatrixXd v(v_1.rows() + v_2.rows(), 3);
	//v.topRows(v_1.rows()) = v_1;
	//v.bottomRows(v_2.rows()) = v_2;
	//Eigen::Vector3d min_corner = Eigen::Vector3d(-1, -1, -1);
	//Eigen::Vector3d max_corner = Eigen::Vector3d(1, 1, 1);
	//Eigen::Matrix4d normalization_trans_mat = BaseAlg::normalize(min_corner, max_corner, v);

	//v_1 = v.topRows(v_1.rows());
	//v_2 = v.bottomRows(v_2.rows());

	//Eigen::MatrixXd downsampled_v1 = v_1, downsampled_v2 = v_2;
	//Eigen::MatrixXd downsampled_vn1 = vn_1, downsampled_vn2 = vn_2;

	std::map<int, Eigen::VectorXi> downsampled_id_map;
	std::map<int, Eigen::MatrixXd> downsampled_v_map;
	std::map<int, Eigen::MatrixXd> downsampled_vn_map;
	std::map<int, Eigen::MatrixXd> features_map;
	std::map<int, Eigen::MatrixXd> filtered_features_map;
	std::map<int, Eigen::MatrixXd> filtered_v_map;
	std::map<int, Eigen::Array<bool, Eigen::Dynamic, 1> > feature_mask_map;
	std::map<int, double> shot_times;
	for(int i = 0; i < pointcloud_filenames.size(); ++i)
	{
		downsampled_v_map[i] = v_map[i];
		downsampled_vn_map[i] = vn_map[i];
		if (need_downsampling)
		{
			downsampled_id_map[i] = BaseAlg::downsampling(v_map[i].rows(), downsampling_num);
			downsampled_v_map[i] = igl::slice(v_map[i], downsampled_id_map[i], 1);
			downsampled_vn_map[i] = igl::slice(vn_map[i], downsampled_id_map[i], 1);
		}
		if(feature_filenames.empty())
		{
			features_map[i] = Eigen::MatrixXd();
			if (feature_type == "fpfh" || feature_type == "FPFH")
			{
				FeatureAlg::compute_fpfh(downsampled_v_map[i], downsampled_vn_map[i], features_map[i]);
			}
			else if (feature_type == "shot" || feature_type == "SHOT")
			{
				Eigen::MatrixXd downsampled_vc = vc_map[i];
				if (need_downsampling)
				{
					downsampled_vc = igl::slice(vc_map[i], downsampled_id_map[i], 1);
				}
				clock_t start = clock();
				FeatureAlg::compute_shot(downsampled_v_map[i], v_map[i], vn_map[i], downsampled_vc, vc_map[i], features_map[i]);
				shot_times[i] = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
				std::cout << "end shot computation" << std::endl;
			}
			igl::writeDMAT(output_path.string() + "\\downsample_ids_" + std::to_string(i) + ".dmat", downsampled_id_map[i], false);
			//igl::writeDMAT(output_path.string() + "\\" + feature_type + "_" + std::to_string(i) + ".dmat", features_map[i], false);
		}
		else
		{
			downsampled_v_map[i] = v_map[i];
			downsampled_vn_map[i] = vn_map[i];
			if (need_downsampling)
			{
				igl::readDMAT("downsample_ids_" + std::to_string(i) + ".dmat", downsampled_id_map[i]);
				downsampled_v_map[i] = igl::slice(v_map[i], downsampled_id_map[i], 1);
				downsampled_vn_map[i] = igl::slice(vn_map[i], downsampled_id_map[i], 1);
			}
			igl::readDMAT(feature_filenames[i], features_map[i]);
		}


		feature_mask_map[i] = features_map[i].col(0).array().isNaN() == false;
		const int valid_feature_num = feature_mask_map[i].count();
		std::cout << "NaN vertices in point cloud "<< i << ": " << features_map[i].rows() - valid_feature_num << std::endl;
		filtered_features_map[i] = igl::slice_mask(features_map[i], feature_mask_map[i], 1);
		filtered_v_map[i] = igl::slice_mask(downsampled_v_map[i], feature_mask_map[i], 1);
	}
	std::map<int, std::map<int, std::vector<std::pair<int, int> > > > corres_map;
	std::map<int, std::map<int, Eigen::MatrixXi> > corres_mat_map;
	std::map<int, std::map<int, double> > pca_time;
	std::map<int, std::map<int, double> > corres_time;
	if (correspondences_filename == "")
	{
		for (int i = 0; i < pointcloud_filenames.size(); ++i)
		{
			corres_map[i] = std::map<int, std::vector<std::pair<int, int> > >();
			corres_mat_map[i] = std::map<int, Eigen::MatrixXi>();
			for (int j = i + 1; j < pointcloud_filenames.size(); ++j)
			{
				int num_i = feature_mask_map[i].count();
				std::cout << "NaN vertices in point cloud " << i << ": " << features_map[i].rows() - num_i << std::endl;
				int num_j = feature_mask_map[j].count();
				std::cout << "NaN vertices in point cloud " << j << ": " << features_map[j].rows() - num_j << std::endl;

				if (pca_component_num > 0)
				{
					std::cout << "start pca computation (target dimension: " << pca_component_num << ")" << std::endl;
					Eigen::MatrixXd filtered_features(num_i + num_j, features_map[i].cols());

					filtered_features.topRows(num_i) = filtered_features_map[i];
					filtered_features.bottomRows(num_j) = filtered_features_map[j];

					Eigen::VectorXd eigenvalues;
					Eigen::MatrixXd eigenvectors;
					clock_t start = clock();
					BaseAlg::pca(filtered_features, pca_component_num, eigenvalues, eigenvectors);

					Eigen::MatrixXd feature_pca = filtered_features * eigenvectors;
					pca_time[i][j] = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
					std::cout << "end pca computation" << std::endl;

					features_map[i] = feature_pca.topRows(num_i);
					features_map[j] = feature_pca.bottomRows(num_j);
				}
				corres_map[i][j] = std::vector<std::pair<int, int> >();
				clock_t start = clock();
				std::cout << "start matching" << std::endl;
				FastGlobalRegistration::advanced_matching(filtered_v_map[i], filtered_v_map[j], features_map[i], features_map[j], corres_map[i][j]);
				corres_time[i][j] = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
				std::cout << "end matching " << corres_time[i][j] << "s" << std::endl;
				//An O(n) method to shift back to the original vertex indices before filering NaN elements
				Eigen::VectorXi offset_i(features_map[i].rows());
				int id_offset = 0;
				for (int s = 0, t = 0; s < feature_mask_map[i].size(); ++s)
				{
					if (!feature_mask_map[i](s))
					{
						++id_offset;
						continue;
					}
					offset_i(t) = id_offset;
					++t;
				}

				Eigen::VectorXi offset_j(features_map[j].rows());
				id_offset = 0;
				for (int s = 0, t = 0; s < feature_mask_map[j].size(); ++s)
				{
					if (!feature_mask_map[j](s))
					{
						++id_offset;
						continue;
					}
					offset_j(t) = id_offset;
					++t;
				}

				for (auto & ele : corres_map[i][j])
				{
					ele.first += offset_i(ele.first);
					ele.second += offset_j(ele.second);
				}
				corres_mat_map[i][j] = Eigen::MatrixXi();
				corres_mat_map[i][j].resize(corres_map[i][j].size(), 2);
				for (int k = 0; k < corres_map[i][j].size(); ++k)
				{
					corres_mat_map[i][j](k, 0) = corres_map[i][j][k].first;
					corres_mat_map[i][j](k, 1) = corres_map[i][j][k].second;
				}
				igl::writeDMAT(output_path.string() + "\\" + feature_type + "_" + std::to_string(i) + "_" +std::to_string(j) + "_corres.dmat", corres_mat_map[i][j]);

				//Eigen::MatrixXd corres_v_1, corres_vn_1, corres_v_2, corres_vn_2;
				//corres_v_1 = igl::slice(downsampled_v_map[i], corres_mat_map[i][j].col(0), 1);
				//corres_vn_1 = igl::slice(downsampled_vn_map[i], corres_mat_map[i][j].col(0), 1);
				//corres_v_2 = igl::slice(downsampled_v_map[j], corres_mat_map[i][j].col(1), 1);
				//corres_vn_2 = igl::slice(downsampled_vn_map[j], corres_mat_map[i][j].col(1), 1);

				////visualize the feature correspondences
				//Eigen::MatrixXd vcolor;
				//igl::jet(corres_v_1.col(0), true, vcolor);
				//vcolor *= 255;

				//DataIO::write_ply(output_path.string() + "\\" + feature_type + "_feature_1.ply", corres_v_1, vcolor, corres_vn_1);
				//DataIO::write_ply(output_path.string() + "\\" + feature_type + "_feature_2.ply", corres_v_2, vcolor, corres_vn_2);
				//system("pause");

				////igl::writeDMAT(feature_type + "_" + std::to_string(i) + "_" + std::to_string(j) + "_corres.dmat", corres_mat_map[i][j]);
			}
		}
	}
	else
	{
		//igl::readDMAT(correspondences_filename, corres_mat);
		//corres.resize(corres_mat.rows());
		//for (int i = 0; i < corres_mat.rows(); ++i)
		//{
		//	corres[i] = std::pair<int, int>(corres_mat(i, 0), corres_mat(i, 1));
		//}
	}
	std::cout << (clock() - begin) / (double)CLOCKS_PER_SEC << "s" << std::endl;
	std::map<int, Eigen::Matrix4d> trans_mat_map;
	
	
	begin = clock();
	FastGlobalRegistration::optimize_global(true, 128, downsampled_v_map, corres_map, trans_mat_map);
	double fgr_time = static_cast<double>(clock() - begin) / CLOCKS_PER_SEC;
	std::cout << "FGR: " << fgr_time << "s" << std::endl;

	std::map<int, Eigen::MatrixXd> aligned_v_map;
	std::map<int, Eigen::MatrixXd> aligned_downsampled_v_map;
	for(const auto& ele : trans_mat_map)
	{
		std::cout << ele.second << std::endl << std::endl;
		aligned_v_map[ele.first] = (v_map[ele.first].rowwise().homogeneous() * ele.second.transpose()).eval().leftCols(3);
		aligned_downsampled_v_map[ele.first] = (downsampled_v_map[ele.first].rowwise().homogeneous() * ele.second.transpose()).eval().leftCols(3);
		Eigen::MatrixXd aligned_vn = (vn_map[ele.first] * ele.second.block<3, 3>(0, 0)).eval().leftCols(3);
		DataIO::write_ply(output_path.string() + "\\" + "aligned_" + std::to_string(ele.first) + ".ply", aligned_v_map[ele.first], vc_map[ele.first], aligned_vn);
		igl::writeDMAT(output_path.string() + "\\" + "transmat_" + std::to_string(ele.first) + ".dmat", ele.second);
	}

	double ssicp_time = 0;
	//if (point_cloud_num == 2)
	//{
	//	Eigen::MatrixXd aligned_v2 = (v_map[1].rowwise().homogeneous() * trans_mat_map[1].transpose()).eval().leftCols(3);
	//	begin = clock();
	//	Eigen::Matrix4d ssicp_trans = SSICP::GetOptimalTrans(aligned_v2, v_map[0], 1e-2, 1e-3);
	//	ssicp_time = static_cast<double>(clock() - begin) / CLOCKS_PER_SEC;
	//	Eigen::Matrix4d final_trans = ssicp_trans * trans_mat_map[1].eval();
	//	Eigen::MatrixXd ssicp_v2 = (v_map[1].rowwise().homogeneous() * final_trans.transpose()).eval().leftCols(3);
	//	//aligned_downsampled_v_map[ele.first] = (downsampled_v_map[ele.first].rowwise().homogeneous() * ele.second.transpose()).eval().leftCols(3);
	//	Eigen::MatrixXd aligned_vn = (vn_map[1] * final_trans.block<3, 3>(0, 0)).eval().leftCols(3);
	//	DataIO::write_ply(output_path.string() + "\\" + "aligned_ssicp.ply", ssicp_v2, vc_map[1], aligned_vn);
	//	igl::writeDMAT(output_path.string() + "\\" + "final_transmat_1.dmat", final_trans);
	//}

	std::ofstream out;
	out.open(output_path.string() + "\\" + "info.txt", std::fstream::out);
	double total_accept_num = 0;
	int total_feature_num = 0;;
	for (int i = 0; i < pointcloud_filenames.size(); ++i)
	{
		for (int j = i + 1; j < pointcloud_filenames.size(); ++j)
		{
			const Eigen::MatrixXd corres_v_1 = igl::slice(aligned_downsampled_v_map[i], corres_mat_map[i][j].col(0), 1);
			const Eigen::MatrixXd corres_v_2 = igl::slice(aligned_downsampled_v_map[j], corres_mat_map[i][j].col(1), 1);
			Eigen::VectorXd distances = (corres_v_1 - corres_v_2).rowwise().norm();
			double accept_num = (distances.array() < 2e-2).count();
			total_accept_num += accept_num;
			total_feature_num += distances.rows();
			out << i << " and " << j << " accept ratio: " << accept_num / distances.rows() << " (Total Num: " << distances.rows() << ")" << std::endl;
			out << pca_time[i][j] << "s for PCA;" << std::endl;
			out << corres_time[i][j] << "s for build correspondences." << std::endl;
			out << shot_times[i] << "s for computing shot descriptors." << std::endl;
			out << fgr_time << "s for sfgr optimization." << std::endl;
			out << ssicp_time << "s for Scale-ICP." << std::endl;
			out << "scaling factor: " << data_scaling_factor << std::endl;
		}
	}
	out << "Overall ratio: " << total_accept_num / total_feature_num <<" (Total Num: " << total_feature_num << ")" << std::endl;
	out.close();
	return;
}