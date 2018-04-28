#include "registration_pipeline.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <DataIO/include/data_io.h>
#include <OverlapTrimmer/include/trimmer.h>

REGPIPELINE_PUBLIC void RegPipeline::TrimPointsClouds(const std::vector<std::string> &filenames,
  std::string &in_format)
{
  if (filenames.size() < 4) return;

  Eigen::MatrixXd A, B;
  if (in_format == "OBJ" || in_format == "obj")
  {
    Eigen::MatrixXi F;
    igl::readOBJ(filenames[0], A, F);
    igl::readOBJ(filenames[1], B, F);
  }
  if (in_format == "PLY" || in_format == "ply")
  {
    Eigen::MatrixXd C;
    DataIO::read_ply(filenames[0], A, C);
    DataIO::read_ply(filenames[1], B, C);
  }

  Trimmer::Trim(A, B, 0.05);

  igl::writeOBJ(filenames[2], A, Eigen::MatrixXi());
  igl::writeOBJ(filenames[3], B, Eigen::MatrixXi());
}
