#include "ScaleStretchICP/include/ssicp.h"

int main(int argc, char *argv[])
{
  std::string input_filename_x(argv[1]), input_filename_y(argv[2]);
	SSICP::Test(input_filename_x, input_filename_y);
	return 0;
}
