#ifndef TRIMMER_H
#define TRIMMER_H
#include "prereq.h"

#include <vector>
#include <Eigen/dense>

namespace Trimmer
{
  struct BoundingBox
  {
    Eigen::Vector3d min_point, max_point;
    
    BoundingBox() {}
    // calculate the bounding box of a point cloud
    BoundingBox(const Eigen::MatrixXd &P);
    bool Contains(const Eigen::Vector3d p) const;
  };

  OVERLAPTRIMMER_PUBLIC BoundingBox Union(const BoundingBox &a, const BoundingBox &b);

  // keep points in the union of bounding boxes of two point clouds
  OVERLAPTRIMMER_PUBLIC void TrimThroughBoundingBox(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
    std::vector<int> &indices_a, std::vector<int> &indices_b);
  // return points near to a another point cloud
  OVERLAPTRIMMER_PUBLIC void GetNearbyPoints(const Eigen::MatrixXd &P, const Eigen::MatrixXd &T,
    const double threshold, std::vector<int> &indices_p);
  // abandon points far away from the other point cloud
  OVERLAPTRIMMER_PUBLIC void TrimThroughDistances(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, double threshold,
    bool percentage, std::vector<int> &indices_a, std::vector<int> &indices_b);

  OVERLAPTRIMMER_PUBLIC void Trim(Eigen::MatrixXd &A, Eigen::MatrixXd &B, double threshold,
    std::vector<int> &indices_a, std::vector<int> &indices_b);
}

#endif