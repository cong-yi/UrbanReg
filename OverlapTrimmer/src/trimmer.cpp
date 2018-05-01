#include "trimmer.h"

#include <climits>
#include <algorithm>
#include <vector>

#include <kdtree.h>

Trimmer::BoundingBox::BoundingBox(const Eigen::MatrixXd &P)
{
  min_point(0) = std::numeric_limits<double>::max();
  min_point(1) = min_point(0), min_point(2) = min_point(0);
  max_point(0) = std::numeric_limits<double>::lowest();
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
  kdtree *ptree = kd_create(3);
  char *data = new char('a');
  for (int i = 0; i < T.rows(); ++i)
    kd_insert3(ptree, T(i, 0), T(i, 1), T(i, 2), data);

  std::vector<Eigen::RowVector3d> points;
  for (int i = 0; i < P.rows(); ++i)
  {
    Eigen::RowVector3d Pi;
    double dist = std::numeric_limits<double>::max();
    kdres *presults = kd_nearest3(ptree, P(i, 0), P(i, 1), P(i, 2));
    while (!kd_res_end(presults))
    {
      double pos[3];
      kd_res_item(presults, pos);
      
      Eigen::RowVector3d R;
      for (int j = 0; j < 3; ++j)
      {
        R(j) = pos[j];
        Pi(j) = P(i, j);
      }
      dist = std::min(dist, (R - Pi).norm());

      kd_res_next(presults);
    }
    kd_res_free(presults);

    if (dist < threshold)
      points.push_back(Pi);
  }

  Eigen::MatrixXd res(points.size(), 3);
  for (int i = 0; i < res.rows(); ++i)
    res.row(i) = points[i];
  return res;
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

OVERLAPTRIMMER_PUBLIC void Trimmer::Trim(Eigen::MatrixXd &A, Eigen::MatrixXd &B, double threshold)
{
  TrimThroughBoudingBox(A, B);
  // TrimThroughDistances(A, B, threshold, true);
}
