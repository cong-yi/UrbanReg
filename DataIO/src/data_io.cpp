#include "data_io.h"

#include <iostream>
#include <fstream>
#include <vector>
#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "rapidxml_print.hpp"
#include "rply.h"
#include <json/json.h>

int DataIO::read_ply(const std::string& filename, Eigen::MatrixXd& v, Eigen::MatrixXd& vc, Eigen::MatrixXd& vn)
{
	p_ply ply = ply_open(filename.c_str(), nullptr, 0, nullptr);
	if (!ply)
	{
		return false;
	}
	if (!ply_read_header(ply))
	{
		return false;
	}
	long nvertices = ply_set_read_cb(ply, "vertex", "x", nullptr, nullptr, 0);
	v.resize(nvertices, 3);
	vc.resize(nvertices, 3);
	vc.setZero();
	vn.resize(nvertices, 3);
	vn.setZero();
	//vertex call back function
	auto read_ply_vertex_call_back = [](p_ply_argument argument)->int
	{
		static unsigned long long vcounter = 0;
		void *ptemp = nullptr;						// pointer to the custom object, store the data
		ply_get_argument_user_data(argument, &ptemp, nullptr);
		Eigen::MatrixXd* p_vmat = static_cast<Eigen::MatrixXd*>(ptemp);
		(*p_vmat)(vcounter / 3, vcounter % 3) = ply_get_argument_value(argument);
		++vcounter;
		if (vcounter == p_vmat->size())
		{
			vcounter = 0;
		}
		return 1;
	};
	//vertex color call back function
	auto read_ply_vertex_color_call_back = [](p_ply_argument argument)->int
	{
		static unsigned long long vcounter = 0;
		void *ptemp = nullptr;						// pointer to the custom object, store the data
		ply_get_argument_user_data(argument, &ptemp, nullptr);
		Eigen::MatrixXd* p_vcmat = static_cast<Eigen::MatrixXd*>(ptemp);
		(*p_vcmat)(vcounter / 3, vcounter % 3) = ply_get_argument_value(argument);
		++vcounter;
		if (vcounter == p_vcmat->size())
		{
			vcounter = 0;
		}
		return 1;
	};
	//vertex normal call back function
	auto read_ply_vertex_normal_call_back = [](p_ply_argument argument)->int
	{
		static unsigned long long vcounter = 0;
		void *ptemp = nullptr;						// pointer to the custom object, store the data
		ply_get_argument_user_data(argument, &ptemp, nullptr);
		Eigen::MatrixXd* p_vnmat = static_cast<Eigen::MatrixXd*>(ptemp);
		(*p_vnmat)(vcounter / 3, vcounter % 3) = ply_get_argument_value(argument);
		++vcounter;
		if (vcounter == p_vnmat->size())
		{
			vcounter = 0;
		}
		return 1;
	};
	// vertex
	ply_set_read_cb(ply, "vertex", "x", read_ply_vertex_call_back, &v, 0);
	ply_set_read_cb(ply, "vertex", "y", read_ply_vertex_call_back, &v, 1);
	ply_set_read_cb(ply, "vertex", "z", read_ply_vertex_call_back, &v, 2);

	//vertex color
	ply_set_read_cb(ply, "vertex", "red", read_ply_vertex_color_call_back, &vc, 0);
	ply_set_read_cb(ply, "vertex", "green", read_ply_vertex_color_call_back, &vc, 1);
	ply_set_read_cb(ply, "vertex", "blue", read_ply_vertex_color_call_back, &vc, 2);

	//vertex normal
	ply_set_read_cb(ply, "vertex", "nx", read_ply_vertex_normal_call_back, &vn, 0);
	ply_set_read_cb(ply, "vertex", "ny", read_ply_vertex_normal_call_back, &vn, 1);
	ply_set_read_cb(ply, "vertex", "nz", read_ply_vertex_normal_call_back, &vn, 2);

	// read mesh info
	if (!ply_read(ply))
	{
		return false;
	}
	ply_close(ply);
	return true;
}

int DataIO::write_ply(const std::string& filename, const Eigen::MatrixXd& v, const Eigen::MatrixXd& vc, const Eigen::MatrixXd& vn)
{
	p_ply oply = ply_create(filename.c_str(), PLY_LITTLE_ENDIAN, nullptr, 0, nullptr);
	if (!oply)
	{
		return 1;
	}
	/* Add vertex element. */
	if (!ply_add_element(oply, "vertex", v.rows())) {
		fprintf(stderr, "ERROR: Could not add element.\n");
		return EXIT_FAILURE;
	}

	/* Add vertex properties: x, y, z, r, g, b */
	if (!ply_add_property(oply, "x", PLY_FLOAT, PLY_FLOAT32, PLY_FLOAT32)) {
		fprintf(stderr, "ERROR: Could not add property x.\n");
		return EXIT_FAILURE;
	}

	if (!ply_add_property(oply, "y", PLY_FLOAT, PLY_FLOAT32, PLY_FLOAT32)) {
		fprintf(stderr, "ERROR: Could not add property y.\n");
		return EXIT_FAILURE;
	}

	if (!ply_add_property(oply, "z", PLY_FLOAT, PLY_FLOAT32, PLY_FLOAT32)) {
		fprintf(stderr, "ERROR: Could not add property z.\n");
		return EXIT_FAILURE;
	}

	if(v.rows() == vc.rows())
	{
		if (!ply_add_property(oply, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR)) {
			fprintf(stderr, "ERROR: Could not add property red.\n");
			return EXIT_FAILURE;
		}

		if (!ply_add_property(oply, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR)) {
			fprintf(stderr, "ERROR: Could not add property green.\n");
			return EXIT_FAILURE;
		}

		if (!ply_add_property(oply, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR)) {
			fprintf(stderr, "ERROR: Could not add property blue.\n");
			return EXIT_FAILURE;
		}
	}

	if (v.rows() == vn.rows())
	{
		if (!ply_add_property(oply, "nx", PLY_FLOAT, PLY_FLOAT32, PLY_FLOAT32)) {
			fprintf(stderr, "ERROR: Could not add property nx.\n");
			return EXIT_FAILURE;
		}

		if (!ply_add_property(oply, "ny", PLY_FLOAT, PLY_FLOAT32, PLY_FLOAT32)) {
			fprintf(stderr, "ERROR: Could not add property ny.\n");
			return EXIT_FAILURE;
		}

		if (!ply_add_property(oply, "nz", PLY_FLOAT, PLY_FLOAT32, PLY_FLOAT32)) {
			fprintf(stderr, "ERROR: Could not add property nz.\n");
			return EXIT_FAILURE;
		}
	}

	/* Write header to file */
	if (!ply_write_header(oply)) {
		fprintf(stderr, "ERROR: Could not write header.\n");
		return EXIT_FAILURE;
	}

	for (int i = 0; i < v.rows(); ++i)
	{
		ply_write(oply, static_cast<float>(v(i, 0))); /* x */
		ply_write(oply, static_cast<float>(v(i, 1))); /* y */
		ply_write(oply, static_cast<float>(v(i, 2))); /* z */
		if (v.rows() == vc.rows())
		{
			ply_write(oply, static_cast<unsigned char>(vc(i, 0))); /* red   */
			ply_write(oply, static_cast<unsigned char>(vc(i, 1))); /* blue  */
			ply_write(oply, static_cast<unsigned char>(vc(i, 2))); /* green */
		}
		if (v.rows() == vn.rows())
		{
			ply_write(oply, static_cast<float>(vn(i, 0))); /* nx  */
			ply_write(oply, static_cast<float>(vn(i, 1))); /* ny  */
			ply_write(oply, static_cast<float>(vn(i, 2))); /* nz  */
		}
	}

	if (!ply_close(oply)) {
		fprintf(stderr, "ERROR: Could not close file.\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int DataIO::load_gps(const std::string &filename, Eigen::MatrixXd &local)
{
  std::ifstream ifs(filename);
  if (!ifs.good()) return EXIT_FAILURE;

  Json::CharReaderBuilder reader;
  Json::Value sfm_data;
  std::string errs;
  Json::parseFromStream(reader, ifs, &sfm_data, &errs);

  

  return EXIT_SUCCESS;
}

int DataIO::read_fgr_config(const std::string& filename, std::vector<std::string>& pointcloud_filenames, std::string& feature_type, std::vector<std::string>& feature_filenames, std::string& correspondences_filename, int& downsampling_num, int& pca_component_num, double& data_scaling_factor, std::string& output_folder)
{
	std::shared_ptr<rapidxml::file<char> > xml_file = nullptr;
	try { xml_file = std::make_shared<rapidxml::file<char> >(filename.c_str()); }
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
	rapidxml::xml_document<> doc;
	doc.parse<0>(xml_file->data());

	rapidxml::xml_node<>* pc_node = doc.first_node("point_cloud");
	rapidxml::xml_node<>* pc_path_node = pc_node->first_node("filename");
	std::cout << "here" << std::endl;
	for (pointcloud_filenames.clear(); pc_path_node; pc_path_node = pc_path_node->next_sibling())
	{
		pointcloud_filenames.emplace_back(pc_path_node->value());
		std::cout << pointcloud_filenames.back() << std::endl;
	}

	rapidxml::xml_node<>* feature_node = doc.first_node("feature");
	feature_type = feature_node->first_attribute("feature_type")->value();
	rapidxml::xml_node<>* features_path_node = feature_node->first_node("filename");
	for(feature_filenames.clear(); features_path_node; features_path_node = features_path_node->next_sibling())
	{
		feature_filenames.emplace_back(features_path_node->value());
	}
	rapidxml::xml_node<>* corres_node = doc.first_node("correspondences");
	rapidxml::xml_node<>* corres_path_node = corres_node->first_node("filename");
	correspondences_filename = corres_path_node->value();
	rapidxml::xml_node<>* parameters_node = doc.first_node("parameters");
	downsampling_num = std::stoi(parameters_node->first_attribute("downsampling_num")->value());
	pca_component_num = std::stoi(parameters_node->first_attribute("pca_component_num")->value());
	data_scaling_factor = std::stod(parameters_node->first_attribute("data_scaling_factor")->value());

	rapidxml::xml_node<>* output_folder_node = doc.first_node("output_folder");
	std::cout << "here" << std::endl;
	output_folder = output_folder_node->first_attribute("name")->value();
	std::cout<< output_folder << std::endl;
	doc.clear();
	return EXIT_SUCCESS;
}

int DataIO::change_config_attribute(const std::string& filename, const std::string& node_name, const std::string& attribute_name, const std::string& new_value)
{
	std::shared_ptr<rapidxml::file<char> > xml_file = nullptr;
	try { xml_file = std::make_shared<rapidxml::file<char> >(filename.c_str()); }
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
	rapidxml::xml_document<> doc;
	doc.parse<0>(xml_file->data());

	rapidxml::xml_node<>* target_node = doc.first_node(node_name.c_str());
	rapidxml::xml_attribute<>* target_attribute = target_node->first_attribute(attribute_name.c_str());
	target_node->remove_attribute(target_attribute);
	target_node->append_attribute(doc.allocate_attribute(attribute_name.c_str(), new_value.c_str()));

	std::ofstream fs_file;
	fs_file.open(filename, std::ofstream::out | std::ofstream::trunc);
	fs_file << doc;
	fs_file.close();
	doc.clear();
	return EXIT_SUCCESS;
}

int DataIO::clear_config_node(const std::string& filename, const std::string& node_name, const std::string& subnode_name)
{
	std::shared_ptr<rapidxml::file<char> > xml_file = nullptr;
	try { xml_file = std::make_shared<rapidxml::file<char> >(filename.c_str()); }
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
	rapidxml::xml_document<> doc;
	doc.parse<0>(xml_file->data());

	rapidxml::xml_node<>* target_parent_node = doc.first_node(node_name.c_str());
	rapidxml::xml_node<>* target_node = target_parent_node->first_node(subnode_name.c_str());
	while(target_parent_node->first_node(subnode_name.c_str()))
	{
		target_parent_node->remove_node(target_parent_node->first_node(subnode_name.c_str()));
	}

	std::ofstream fs_file;
	fs_file.open(filename, std::ofstream::out | std::ofstream::trunc);
	fs_file << doc;
	fs_file.close();
	doc.clear();
	return EXIT_SUCCESS;
}

int DataIO::add_config_node(const std::string& filename, const std::string& node_name, const std::string& subnode_name, const std::string& new_value)
{
	std::shared_ptr<rapidxml::file<char> > xml_file = nullptr;
	try { xml_file = std::make_shared<rapidxml::file<char> >(filename.c_str()); }
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return false;
	}
	rapidxml::xml_document<> doc;
	doc.parse<0>(xml_file->data());

	rapidxml::xml_node<>* target_parrent_node = doc.first_node(node_name.c_str());
	rapidxml::xml_node<>* new_node = doc.allocate_node(rapidxml::node_element, subnode_name.c_str(), new_value.c_str());
	target_parrent_node->append_node(new_node);

	std::ofstream fs_file;
	fs_file.open(filename, std::ofstream::out | std::ofstream::trunc);
	fs_file << doc;
	fs_file.close();
	doc.clear();
	return EXIT_SUCCESS;
}

int DataIO::write_file(const std::string& filename, double value)
{
	std::ofstream out;
	out.open(filename, std::fstream::out);
	out << value;
	out.close();
	return EXIT_SUCCESS;
}