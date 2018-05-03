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

OVERLAPTRIMMER_PUBLIC void Trimmer::TrimThroughBoundingBox(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
  std::vector<int> &indices_a, std::vector<int> &indices_b)
{
  BoundingBox bb = Union(BoundingBox(A), BoundingBox(B));
  
  std::vector<int> ka, kb;
  for (size_t i = 0; i < indices_a.size(); ++i)
    if (bb.Contains(A.row(indices_a[i])))
      ka.push_back(indices_a[i]);
  for (size_t i = 0; i < indices_b.size(); ++i)
    if (bb.Contains(B.row(indices_b[i])))
      kb.push_back(indices_b[i]);

  indices_a = ka, indices_b = kb;
}

OVERLAPTRIMMER_PUBLIC void Trimmer::GetNearbyPoints(const Eigen::MatrixXd &P, const Eigen::MatrixXd &T,
  double threshold, std::vector<int> &indices_p)
{
  kdtree *ptree = kd_create(3);
  char *data = new char('a');
  for (int i = 0; i < T.rows(); ++i)
    kd_insert3(ptree, T(i, 0), T(i, 1), T(i, 2), data);

  std::vector<int> kp;
  for (int i = 0; i < indices_p.size(); ++i)
  {
    Eigen::RowVector3d Pi;
    double dist = std::numeric_limits<double>::max();
    kdres *presults = kd_nearest3(ptree, P(indices_p[i], 0), P(indices_p[i], 1), P(indices_p[i], 2));
    while (!kd_res_end(presults))
    {
      double pos[3];
      kd_res_item(presults, pos);
      
      Eigen::RowVector3d R;
      for (int j = 0; j < 3; ++j)
      {
        R(j) = pos[j];
        Pi(j) = P(indices_p[i], j);
      }
      dist = std::min(dist, (R - Pi).norm());

      kd_res_next(presults);
    }
    kd_res_free(presults);

    if (dist < threshold)
      kp.push_back(indices_p[i]);
  }
  indices_p = kp;
}

OVERLAPTRIMMER_PUBLIC void Trimmer::TrimThroughDistances(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
  double threshold, bool percentage, std::vector<int> &indices_a, std::vector<int> &indices_b)
{
  if (percentage)
  {
    BoundingBox bb = Union(BoundingBox(A), BoundingBox(B));
    threshold = threshold * (bb.max_point - bb.min_point).norm();
  }
  GetNearbyPoints(A, B, threshold, indices_a);
  GetNearbyPoints(B, A, threshold, indices_b);
}

OVERLAPTRIMMER_PUBLIC void Trimmer::Trim(Eigen::MatrixXd &A, Eigen::MatrixXd &B, double threshold,
  std::vector<int> &indices_a, std::vector<int> &indices_b)
{
  TrimThroughBoundingBox(A, B, indices_a, indices_b);
  // TrimThroughDistances(A, B, threshold, true, indices_a, indices_b);
}
