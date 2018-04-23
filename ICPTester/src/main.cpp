#include "ScaleStretchICP/include/ssicp.h"

int main(int argc, char *argv[])
{
  std::string input_filename_x(argv[1]), input_filename_y(argv[2]);
  std::string output_filename(argv[3]);
	SSICP::Test(input_filename_x, input_filename_y, output_filename);
	return 0;
}
