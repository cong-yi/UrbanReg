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
#include <random>

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
	int downsampling_num;
	int pca_component_num;

	DataIO::read_fgr_config(config_filename, pointcloud_filenames, feature_type, feature_filenames, correspondences_filename, downsampling_num, pca_component_num);

	Eigen::MatrixXd v_1, vc_1, vn_1, v_2, vc_2, vn_2;

	DataIO::read_ply(pointcloud_filenames[0], v_1, vc_1, vn_1);
	DataIO::read_ply(pointcloud_filenames[1], v_2, vc_2, vn_2);

	v_2 = v_2 * 1.2;

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
	int downsampling_num;
	int pca_component_num;

	DataIO::read_fgr_config(config_filename, pointcloud_filenames, feature_type, feature_filenames, correspondences_filename, downsampling_num, pca_component_num);

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
	if (need_downsampling)
	{
		downsampled_v1.conservativeResize(downsampling_num, 3);
		downsampled_v2.conservativeResize(downsampling_num, 3);
		downsampled_vn1.conservativeResize(downsampling_num, 3);
		downsampled_vn2.conservativeResize(downsampling_num, 3);
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
				Eigen::MatrixXd downsampled_vc1 = vc_1, downsampled_vc2 = vc_2;
				if (need_downsampling)
				{
					downsampled_vc1.conservativeResize(downsampling_num, 3);
					downsampled_vc2.conservativeResize(downsampling_num, 3);
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

	//Eigen::MatrixXd corres_v_1, corres_vc_1, corres_vn_1, corres_v_2, corres_vc_2, corres_vn_2, corres_v_aligned;
	//corres_v_1 = igl::slice(downsampled_v1, corres_mat.col(0), 1);
	//corres_vc_1 = igl::slice(vc_1, corres_mat.col(0), 1);
	//corres_vn_1 = igl::slice(vn_1, corres_mat.col(0), 1);
	//corres_v_2 = igl::slice(v_2, corres_mat.col(1), 1);
	//corres_vc_2 = igl::slice(vc_2, corres_mat.col(1), 1);
	//corres_vn_2 = igl::slice(vn_2, corres_mat.col(1), 1);
	//corres_v_aligned = igl::slice(aligned_v2, corres_mat.col(1), 1);

	//visualize the feature correspondences
	//Eigen::MatrixXd vcolor;
	//igl::jet(corres_v_1.col(1), true, vcolor);
	//vcolor *= 255;

	//DataIO::write_ply(feature_type + "_feature_1.ply", corres_v_1, vcolor, corres_vn_1);
	//DataIO::write_ply(feature_type + "_feature_2.ply", corres_v_2, vcolor, corres_vn_2);
	//DataIO::write_ply(feature_type + "_feature_3.ply", corres_v_aligned, vcolor, corres_vn_2);

	DataIO::write_ply(feature_type + "_fgr_1.ply", v_1, vc_1, vn_1);
	DataIO::write_ply(feature_type + "_fgr_2.ply", v_2, vc_2, vn_2);
	DataIO::write_ply(feature_type + "_fgr_2_aligned.ply", aligned_v2, vc_2, aligned_vn_2);

	return;
}