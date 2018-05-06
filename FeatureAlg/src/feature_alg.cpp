#include "feature_alg.h"
#include <pcl/point_types.h>
#include <pcl/io/obj_io.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <igl/writeOBJ.h>
#include <chrono>

void SearchFLANNTree(flann::Index<flann::L2<float>>* index,
	Eigen::VectorXf& input,
	std::vector<int>& indices,
	std::vector<float>& dists,
	int nn)
{
	int rows_t = 1;
	int dim = input.size();

	std::vector<float> query;
	query.resize(rows_t*dim);
	for (int i = 0; i < dim; i++)
		query[i] = input(i);
	flann::Matrix<float> query_mat(&query[0], rows_t, dim);

	indices.resize(rows_t*nn);
	dists.resize(rows_t*nn);
	flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
	flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);

	index->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
}

void FeatureAlg::compute_fpfh(const Eigen::MatrixXd& v, const Eigen::MatrixXd& vn, Eigen::MatrixXd& fpfh)
{
	pcl::PointCloud<pcl::PointNormal>::Ptr object(new pcl::PointCloud<pcl::PointNormal>);

	if(vn.rows() == v.rows())
	{
		object->points.resize(v.rows());
		for (int i = 0; i < v.rows(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				object->points[i].data[j] = static_cast<float>(v(i, j));
				object->points[i].data_n[j] = static_cast<float>(vn(i, j));
			}
			object->points[i].data[3] = 1.0f;
			object->points[i].data_n[3] = 0.0f;
		}
	}
	else
	{
		return;
		//pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_xyz(new pcl::PointCloud<pcl::PointXYZ>);
		//tmp_xyz->points.resize(v.rows());
		//for (int i = 0; i < v.rows(); ++i)
		//{
		//	for (int j = 0; j < 3; ++j)
		//	{
		//		tmp_xyz->points[i].data[j] = static_cast<float>(v(i, j));
		//	}
		//	tmp_xyz->points[i].data[3] = 1.0f;
		//}

		//// Create the normal estimation class, and pass the input dataset to it
		//pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		//ne.setInputCloud(tmp_xyz);

		//// Create an empty kdtree representation, and pass it to the normal estimation object.
		//// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		//ne.setSearchMethod(tree);

		//// Output datasets
		//pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

		//// Use all neighbors in a sphere of radius 3m
		//ne.setRadiusSearch(250);

		//// Compute the features
		//ne.compute(*cloud_normals);
		//
		//pcl::copyPointCloud(*tmp_xyz, *object);
		//pcl::copyPointCloud(*cloud_normals, *object);
		//pcl::io::savePLYFile("normal_test.ply", *object, false);

	}
	std::cout << "Start FPFH computation..." << std::endl;
	auto time_start = std::chrono::high_resolution_clock::now();
	pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fest;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_features(new pcl::PointCloud<pcl::FPFHSignature33>());
	double feature_radius = 0.1;
	
	fest.setRadiusSearch(feature_radius);
	fest.setInputCloud(object);
	fest.setInputNormals(object);
	fest.compute(*object_features);

	fpfh.resize(v.rows(), object_features->points[0].descriptorSize());
	for(int i = 0; i < fpfh.rows(); ++i)
	{
		for(int j = 0; j < fpfh.cols(); ++j)
		{
			fpfh(i, j) = static_cast<double>(object_features->points[i].histogram[j]);
		}
	}
	auto time_end = std::chrono::high_resolution_clock::now();
	std::cout << "FPFH computation complete. It costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
}

void FeatureAlg::compute_shot(const Eigen::MatrixXd& v, const Eigen::MatrixXd& vn, const Eigen::MatrixXd& vc, Eigen::MatrixXd& shot)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud< pcl::Normal>);

	if (vn.rows() == v.rows())
	{
		cloud->points.resize(v.rows());
		normals->points.resize(vn.rows());
		for (int i = 0; i < v.rows(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				cloud->points[i].data[j] = static_cast<float>(v(i, j));
				normals->points[i].data_n[j] = static_cast<float>(vn(i, j));
			}
			cloud->points[i].r = static_cast<uint8_t>(vc(i, 0));
			cloud->points[i].g = static_cast<uint8_t>(vc(i, 1));
			cloud->points[i].b = static_cast<uint8_t>(vc(i, 2));
			cloud->points[i].a = 255;
			cloud->points[i].data[3] = 1.0f;
			normals->points[i].data_n[3] = 0.0f;
		}
	}
	// Setup the SHOT features
	typedef pcl::SHOT1344 ShotFeature;
	pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, ShotFeature> shotEstimation;
	shotEstimation.setInputCloud(cloud);
	shotEstimation.setInputNormals(normals);
	//shotEstimation.setSearchSurface(cloud);

	pcl::PointCloud<ShotFeature>::Ptr shotFeatures(new pcl::PointCloud<ShotFeature>);
	shotEstimation.setRadiusSearch (0.1);

	// Actually compute the spin images
	shotEstimation.compute(*shotFeatures);
	std::cout << "SHOT output points.size (): " << shotFeatures->points.size() << std::endl;

	std::cout << "SHOT output feature length: " << shotFeatures->points[0].descriptorSize() << std::endl;

	shot.resize(v.rows(), shotFeatures->points[0].descriptorSize());
	for (int i = 0; i < v.rows(); ++i)
	{
		for (int j = 0; j < shot.cols(); ++j)
		{
			
			shot(i, j) = static_cast<double>(shotFeatures->points[i].descriptor[j]);
		}
	}
	return;
}

void FeatureAlg::icp(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, Eigen::MatrixXd& aligned_v_2)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZ>);

	cloud_1->points.resize(v_1.rows());
	for (int i = 0; i < v_1.rows(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			cloud_1->points[i].data[j] = static_cast<float>(v_1(i, j));
		}
		cloud_1->points[i].data[3] = 1.0f;
	}

	cloud_2->points.resize(v_2.rows());
	for (int i = 0; i < v_2.rows(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			cloud_2->points[i].data[j] = static_cast<float>(v_2(i, j));
		}
		cloud_2->points[i].data[3] = 1.0f;
	}

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputCloud(cloud_2);
	icp.setInputTarget(cloud_1);

	icp.setMaxCorrespondenceDistance(0.1);
	icp.setMaximumIterations(50);
	icp.setTransformationEpsilon(1e-8);
	icp.setEuclideanFitnessEpsilon(1);

	pcl::PointCloud<pcl::PointXYZ> final;
	icp.align(final);
	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore() << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;

	std::cout << icp.getFinalTransformation() * icp.getFinalTransformation().transpose().eval() << std::endl;

	aligned_v_2.resize(final.points.size(), 3);
	for(int i = 0; i < final.points.size(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			aligned_v_2(i, j) = static_cast<double>(final.points[i].data[j]);
		}
	}
}