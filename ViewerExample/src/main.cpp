#include <ctime>

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <OverlapTrimmer/include/trimmer.h>
#include <ScaleStretchICP/include/ssicp.h>
#include "DataIO/include/data_io.h"
#include <igl/opengl/glfw/Viewer.h>
#include <random>

void main()
{
	Eigen::MatrixXd v1, vc1;
	DataIO::read_ply("downsampled_e4_with_color.ply", v1, vc1);

	Eigen::MatrixXd v2, vc2;
	DataIO::read_ply("downsampled_e5_with_color.ply", v2, vc2);

	igl::opengl::glfw::Viewer viewer;
	vc1 /= 255.0;
	vc2 /= 255.0;
	// Plot the points
	viewer.data().set_points(v1, vc1);
	viewer.data().add_points(v2, vc2);
	viewer.data().point_size = 2;
	viewer.launch();
}
