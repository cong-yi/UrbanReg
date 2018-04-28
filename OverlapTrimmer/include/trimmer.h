#ifndef TRIMMER_H
#define TRIMMER_H
#include "prereq.h"

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
  OVERLAPTRIMMER_PUBLIC void TrimThroughBoudingBox(Eigen::MatrixXd &A, Eigen::MatrixXd &B);
  // return points near to a another point cloud
  OVERLAPTRIMMER_PUBLIC Eigen::MatrixXd GetNearbyPoints(const Eigen::MatrixXd &P, const Eigen::MatrixXd &T,
    double threshold);
  // abandon points far away from the other point cloud
  OVERLAPTRIMMER_PUBLIC void TrimThroughDistances(Eigen::MatrixXd &A, Eigen::MatrixXd &B, double threshold,
    bool percentage);

  OVERLAPTRIMMER_PUBLIC void Trim(Eigen::MatrixXd &A, Eigen::MatrixXd &B, double threshold);
}

#endif