#include "trimmer.h"

#include <climits>
#include <algorithm>
#include <vector>

#include <flann/flann.hpp>

Trimmer::BoundingBox::BoundingBox(const Eigen::MatrixXd &P)
{
  min_point(0) = std::numeric_limits<double>::max();
  min_point(1) = min_point(0), min_point(2) = min_point(0);
  max_point(0) = std::numeric_limits<double>::min();
  max_point(1) = max_point(0), max_point(2) = max_point(0);
  for (int i = 0; i < P.rows(); ++i)
  {
    for (size_t j = 0; j < 3; ++j)
    {
      min_point(j) = std::min(min_point(j), P(i, j));
      max_point(j) = std::max(max_point(j), P(i, j));
    }
  }
}

bool Trimmer::BoundingBox::Contains(const Eigen::Vector3d p) const
{
  for (size_t i = 0; i < 3; ++i)
    if (p(i) < min_point(i) || p(i) > max_point(i))
      return false;
  return true;
}

OVERLAPTRIMMER_PUBLIC Trimmer::BoundingBox Trimmer::Union(const BoundingBox &a, const BoundingBox &b)
{
  BoundingBox bb;
  for (size_t i = 0; i < 3; ++i)
  {
    bb.min_point(i) = std::max(a.min_point(i), b.min_point(i));
    bb.max_point(i) = std::min(a.max_point(i), b.max_point(i));
  }
  return bb;
}

OVERLAPTRIMMER_PUBLIC void Trimmer::TrimThroughBoudingBox(Eigen::MatrixXd &A, Eigen::MatrixXd &B)
{
  BoundingBox bb = Union(BoundingBox(A), BoundingBox(B));
  size_t ca = 0, cb = 0;
  for (int i = 0; i < A.rows(); ++i)
    ca += bb.Contains(A.row(i));
  for (int i = 0; i < B.rows(); ++i)
    cb += bb.Contains(B.row(i));

  Eigen::MatrixXd AA(ca, 3), BB(cb, 3);
  ca = cb = 0;
  for (int i = 0; i < A.rows(); ++i)
    if (bb.Contains(A.row(i)))
      AA.row(ca++) = A.row(i);
  for (int i = 0; i < B.rows(); ++i)
    if (bb.Contains(B.row(i)))
      BB.row(cb++) = B.row(i);
  A = AA, B = BB;
}

OVERLAPTRIMMER_PUBLIC Eigen::MatrixXd Trimmer::GetNearbyPoints(const Eigen::MatrixXd &P, const Eigen::MatrixXd &T,
  double threshold)
{
  int rt = T.rows(), c = T.cols();
  double *data = static_cast<double *>(malloc(1LL * rt * c * sizeof(double)));
  double *ptr = data;
  for (int i = 0; i < rt; ++i)
    for (int j = 0; j < c; ++j)
      *(ptr++) = T(i, j);
  flann::Matrix<double> data_mat(data, rt, c);
  flann::Index<flann::L2<double>> index(data_mat, flann::KDTreeIndexParams(4));
  index.buildIndex();

  int rp = P.rows();
  double *queries = static_cast<double *>(malloc(1LL * rp * c * sizeof(double)));
  ptr = queries;
  for (int i = 0; i < rp; ++i)
    for (int j = 0; j < c; ++j)
      *(ptr++) = P(i, j);
  flann::Matrix<double> queries_mat(queries, rp, c);

  std::vector<std::vector<int>> indices;
  std::vector<std::vector<double>> dists;
  index.radiusSearch(queries_mat, indices, dists, threshold * threshold, flann::SearchParams(128));

  int counter = 0;
  for (size_t i = 0; i < indices.size(); ++i)
    counter += (!indices[i].empty());
  int it = 0;
  Eigen::MatrixXd A(counter, c);
  for (size_t i = 0; i < indices.size(); ++i)
    if (!indices[i].empty())
      A.row(it++) = P.row(i);
  return A;
}

OVERLAPTRIMMER_PUBLIC void Trimmer::TrimThroughDistances(Eigen::MatrixXd &A, Eigen::MatrixXd &B, double threshold,
  bool percentage)
{
  if (percentage)
  {
    BoundingBox bb = Union(BoundingBox(A), BoundingBox(B));
    threshold = threshold * (bb.max_point - bb.min_point).norm();
  }
  Eigen::MatrixXd AA = GetNearbyPoints(A, B, threshold), BB = GetNearbyPoints(B, A, threshold);
  A = AA, B = BB;
}

OVERLAPTRIMMER_PUBLIC void Trimmer::Test(Eigen::MatrixXd &A, Eigen::MatrixXd &B, double threshold)
{
  TrimThroughBoudingBox(A, B);
  TrimThroughDistances(A, B, threshold, true);
}
