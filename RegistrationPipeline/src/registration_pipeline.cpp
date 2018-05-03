#include "registration_pipeline.h"

#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <DataIO/include/data_io.h>
#include <GPSLocalizer/include/gps.h>
#include <OverlapTrimmer/include/trimmer.h>

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
