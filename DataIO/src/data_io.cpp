#include "data_io.h"
#include <vector>
#include "rply.h"
#include <iostream>
#include "../../ScaleStretchICP/include/ssicp.h"
#include <json/json.h>

int DataIO::read_ply(const std::string& filename, Eigen::MatrixXd& v, Eigen::MatrixXd& vc)
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
	// vertex
	ply_set_read_cb(ply, "vertex", "x", read_ply_vertex_call_back, &v, 0);
	ply_set_read_cb(ply, "vertex", "y", read_ply_vertex_call_back, &v, 1);
	ply_set_read_cb(ply, "vertex", "z", read_ply_vertex_call_back, &v, 2);

	//vertex color
	ply_set_read_cb(ply, "vertex", "red", read_ply_vertex_color_call_back, &vc, 0);
	ply_set_read_cb(ply, "vertex", "green", read_ply_vertex_color_call_back, &vc, 1);
	ply_set_read_cb(ply, "vertex", "blue", read_ply_vertex_color_call_back, &vc, 2);

	// read mesh info
	if (!ply_read(ply))
	{
		return false;
	}
	ply_close(ply);
	return true;
}

int DataIO::write_ply(const std::string& filename, const Eigen::MatrixXd& v, const Eigen::MatrixXd& vc)
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
	bool has_vertex_color = v.rows() == vc.rows() ? true : false;
	if(has_vertex_color)
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

	/* Write header to file */
	if (!ply_write_header(oply)) {
		fprintf(stderr, "ERROR: Could not write header.\n");
		return EXIT_FAILURE;
	}

	for (int i = 0; i < v.rows(); ++i)
	{
		ply_write(oply, v(i, 0)); /* x */
		ply_write(oply, v(i, 1)); /* y */
		ply_write(oply, v(i, 2)); /* z */
		if (has_vertex_color)
		{
			ply_write(oply, static_cast<unsigned char>(vc(i, 0))); /* red   */
			ply_write(oply, static_cast<unsigned char>(vc(i, 1))); /* blue  */
			ply_write(oply, static_cast<unsigned char>(vc(i, 2))); /* green */
		}
	}

	if (!ply_close(oply)) {
		fprintf(stderr, "ERROR: Could not close file.\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}