#include "feature_alg.h"
#include <pcl/point_types.h>
#include <pcl/io/obj_io.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <chrono>

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
		pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_xyz(new pcl::PointCloud<pcl::PointXYZ>);
		tmp_xyz->points.resize(v.rows());
		for (int i = 0; i < v.rows(); ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				tmp_xyz->points[i].data[j] = static_cast<float>(v(i, j));
			}
			tmp_xyz->points[i].data[3] = 1.0f;
		}

		// Create the normal estimation class, and pass the input dataset to it
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		ne.setInputCloud(tmp_xyz);

		// Create an empty kdtree representation, and pass it to the normal estimation object.
		// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		ne.setSearchMethod(tree);

		// Output datasets
		pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

		// Use all neighbors in a sphere of radius 3m
		ne.setRadiusSearch(0.03);

		// Compute the features
		ne.compute(*cloud_normals);
		
		pcl::copyPointCloud(*tmp_xyz, *object);
		pcl::copyPointCloud(*cloud_normals, *object);
	}
	std::cout << "Start FPFH computation..." << std::endl;
	auto time_start = std::chrono::high_resolution_clock::now();
	pcl::FPFHEstimationOMP<pcl::PointNormal, pcl::PointNormal, pcl::FPFHSignature33> fest;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr object_features(new pcl::PointCloud<pcl::FPFHSignature33>());
	double feature_radius = 0.5;
	
	fest.setRadiusSearch(feature_radius);
	fest.setInputCloud(object);
	fest.setInputNormals(object);
	fest.compute(*object_features);

	fpfh.resize(v.rows(), 33);
	for(int i = 0; i < v.rows(); ++i)
	{
		for(int j = 0; j < 33; ++j)
		{
			fpfh(i, j) = static_cast<double>(object_features->points[i].histogram[j]);
		}
	}
	auto time_end = std::chrono::high_resolution_clock::now();
	std::cout << "FPFH computation complete. It costs: " << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() << " ms" << std::endl;
}
