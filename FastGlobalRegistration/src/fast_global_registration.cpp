#include "fast_global_registration.h"
#include <flann/flann.hpp>

#include "./ScaleStretchICP/include/ssicp.h"
#include <pcl/sample_consensus/sac_model_registration.h>
#include <igl/slice.h>
#include <pcl/sample_consensus/ransac.h>
#include <igl/writeDMAT.h>

#define DIV_FACTOR			1.4		// Division factor used for graduated non-convexity
#define USE_ABSOLUTE_SCALE	0		// Measure distance in absolute scale (1) or in scale relative to the diameter of the model (0)
#define MAX_CORR_DIST		0.0001	// Maximum correspondence distance (also see comment of USE_ABSOLUTE_SCALE)
#define TUPLE_SCALE			0.95	// Similarity measure used for tuples of feature points.
#define TUPLE_MAX_CNT		1000	// Maximum tuple numbers.
#define SHOW_DEBUG_INFO
#define TUPLE_SIMILAR_CRITERIA
//#define USE_RANSAC

void SearchFLANNTree(flann::Index<flann::L2<float>>* index,
  Eigen::VectorXf& input,
  std::vector<int>& indices,
  std::vector<float>& dists,
  int nn)
{
  int rows_t = 1;
  int dim = input.size();

  std::vector<float> query;
  query.resize(rows_t*dim);
  for (int i = 0; i < dim; i++)
    query[i] = input(i);
  flann::Matrix<float> query_mat(&query[0], rows_t, dim);

  indices.resize(rows_t*nn);
  dists.resize(rows_t*nn);
  flann::Matrix<int> indices_mat(&indices[0], rows_t, nn);
  flann::Matrix<float> dists_mat(&dists[0], rows_t, nn);

  index->knnSearch(query_mat, indices_mat, dists_mat, nn, flann::SearchParams(128));
}

void FastGlobalRegistration::advanced_matching(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const Eigen::MatrixXd& fpfh_1, const Eigen::MatrixXd& fpfh_2, std::vector<std::pair<int, int> >& corres)
{
	clock_t start = clock();
	std::cout << "start matching" << std::endl;

  std::vector<const Eigen::MatrixXd *> pointcloud(2);
  pointcloud[0] = &v_1;
  pointcloud[1] = &v_2;
  std::vector<const Eigen::MatrixXd *> features(2);
  features[0] = &fpfh_1;
  features[1] = &fpfh_2;

  int fi = 0;
  int fj = 1;

  printf("Advanced matching : [%d - %d]\n", fi, fj);
  bool swapped = false;

  if (pointcloud[fj]->rows() > pointcloud[fi]->rows())
  {
    int temp = fi;
    fi = fj;
    fj = temp;
    swapped = true;
  }

  int nPti = pointcloud[fi]->rows();
  int nPtj = pointcloud[fj]->rows();

  ///////////////////////////
  /// BUILD FLANNTREE
  ///////////////////////////

  // build FLANNTree - fi
  int rows, dim;
  rows = features[fi]->rows();
  dim = features[fi]->cols();

  std::vector<float> dataset_fi(rows * dim);
  flann::Matrix<float> dataset_mat_fi(&dataset_fi[0], rows, dim);

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < dim; j++)
      dataset_fi[i * dim + j] = static_cast<float>((*features[fi])(i, j));

  flann::Index<flann::L2<float>> feature_tree_i(dataset_mat_fi, flann::KDTreeSingleIndexParams(15));
  feature_tree_i.buildIndex();

  // build FLANNTree - fj
  rows = features[fj]->rows();
  dim = features[fj]->cols();

  std::vector<float> dataset_fj(rows * dim);
  flann::Matrix<float> dataset_mat_fj(&dataset_fj[0], rows, dim);

  for (int i = 0; i < rows; i++)
    for (int j = 0; j < dim; j++)
      dataset_fj[i * dim + j] = static_cast<float>((*features[fj])(i, j));

  flann::Index<flann::L2<float>> feature_tree_j(dataset_mat_fj, flann::KDTreeSingleIndexParams(15));
  feature_tree_j.buildIndex();

  bool crosscheck = true;
  bool tuple = true;

  std::vector<int> corres_K, corres_K2;
  std::vector<float> dis;
  std::vector<int> ind;

  //std::vector<std::pair<int, int>> corres;
  std::vector<std::pair<int, int>> corres_cross;
  std::vector<std::pair<int, int>> corres_ij;
  std::vector<std::pair<int, int>> corres_ji;

  ///////////////////////////
  /// INITIAL MATCHING
  ///////////////////////////

  std::vector<int> i_to_j(nPti, -1);
  for (int j = 0; j < nPtj; j++)
  {
    Eigen::VectorXf query_feature_j = features[fj]->row(j).transpose().cast<float>();
    SearchFLANNTree(&feature_tree_i, query_feature_j, corres_K, dis, 1);
    int i = corres_K[0];
    if (i_to_j[i] == -1)
    {
      Eigen::VectorXf query_feature_i = features[fi]->row(i).transpose().cast<float>();
      SearchFLANNTree(&feature_tree_j, query_feature_i, corres_K, dis, 1);
      int ij = corres_K[0];
      i_to_j[i] = ij;
    }
    corres_ji.push_back(std::pair<int, int>(i, j));
  }

  for (int i = 0; i < nPti; i++)
  {
    if (i_to_j[i] != -1)
      corres_ij.push_back(std::pair<int, int>(i, i_to_j[i]));
  }

  int ncorres_ij = corres_ij.size();
  int ncorres_ji = corres_ji.size();

  // corres = corres_ij + corres_ji;
  for (int i = 0; i < ncorres_ij; ++i)
    corres.push_back(std::pair<int, int>(corres_ij[i].first, corres_ij[i].second));
  for (int j = 0; j < ncorres_ji; ++j)
    corres.push_back(std::pair<int, int>(corres_ji[j].first, corres_ji[j].second));

  printf("points are remained : %d\n", (int)corres.size());

  ///////////////////////////
  /// CROSS CHECK
  /// input : corres_ij, corres_ji
  /// output : corres
  ///////////////////////////
  if (crosscheck)
  {
    printf("\t[cross check] ");

    // build data structure for cross check
    corres.clear();
    corres_cross.clear();
    std::vector<std::vector<int>> Mi(nPti);
    std::vector<std::vector<int>> Mj(nPtj);

    int ci, cj;
    for (int i = 0; i < ncorres_ij; ++i)
    {
      ci = corres_ij[i].first;
      cj = corres_ij[i].second;
      Mi[ci].push_back(cj);
    }
    for (int j = 0; j < ncorres_ji; ++j)
    {
      ci = corres_ji[j].first;
      cj = corres_ji[j].second;
      Mj[cj].push_back(ci);
    }

    // cross check
    for (int i = 0; i < nPti; ++i)
    {
      for (int ii = 0; ii < Mi[i].size(); ++ii)
      {
        int j = Mi[i][ii];
        for (int jj = 0; jj < Mj[j].size(); ++jj)
        {
          if (Mj[j][jj] == i)
          {
            corres.push_back(std::pair<int, int>(i, j));
            corres_cross.push_back(std::pair<int, int>(i, j));
          }
        }
      }
    }
    printf("points are remained : %d\n", (int)corres.size());
  }

  ///////////////////////////
  /// TUPLE CONSTRAINT
  /// input : corres
  /// output : corres
  ///////////////////////////
  if (tuple)
  {
    srand(time(NULL));

    printf("\t[tuple constraint] ");
    int rand0, rand1, rand2;
    int idi0, idi1, idi2;
    int idj0, idj1, idj2;
    float scale = TUPLE_SCALE;
    int ncorr = corres.size();
    int number_of_trial = ncorr * 100;
    std::vector<std::pair<int, int>> corres_tuple;

    int cnt = 0;
    int i;
    for (i = 0; i < number_of_trial; i++)
    {
      rand0 = rand() % ncorr;
      rand1 = rand() % ncorr;
      rand2 = rand() % ncorr;

      idi0 = corres[rand0].first;
      idj0 = corres[rand0].second;
      idi1 = corres[rand1].first;
      idj1 = corres[rand1].second;
      idi2 = corres[rand2].first;
      idj2 = corres[rand2].second;

      // collect 3 points from i-th fragment
      Eigen::Vector3f pti0 = pointcloud[fi]->row(idi0).transpose().cast<float>();
      Eigen::Vector3f pti1 = pointcloud[fi]->row(idi1).transpose().cast<float>();
      Eigen::Vector3f pti2 = pointcloud[fi]->row(idi2).transpose().cast<float>();

      float li0 = (pti0 - pti1).norm();
      float li1 = (pti1 - pti2).norm();
      float li2 = (pti2 - pti0).norm();

      // collect 3 points from j-th fragment
      Eigen::Vector3f ptj0 = pointcloud[fj]->row(idj0).transpose().cast<float>();
      Eigen::Vector3f ptj1 = pointcloud[fj]->row(idj1).transpose().cast<float>();
      Eigen::Vector3f ptj2 = pointcloud[fj]->row(idj2).transpose().cast<float>();

      float lj0 = (ptj0 - ptj1).norm();
      float lj1 = (ptj1 - ptj2).norm();
      float lj2 = (ptj2 - ptj0).norm();
#ifndef TUPLE_SIMILAR_CRITERIA
      if ((li0 * scale < lj0) && (lj0 < li0 / scale) &&
        (li1 * scale < lj1) && (lj1 < li1 / scale) &&
        (li2 * scale < lj2) && (lj2 < li2 / scale))
      {
        corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
        corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
        corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
        cnt++;
      }
#else
	  float k0 = li0 / lj0;
	  float k1 = li1 / lj1;
	  float k2 = li2 / lj2;
	  if ((k0 * k0 / (k1*k2) > scale) && (k1*k2 / (k0*k0) > scale) &&
		  (k1 * k1 / (k0*k2) > scale) && (k0*k2 / (k1*k1) > scale) &&
		  (k2 * k2 / (k1*k0) > scale) && (k1*k0 / (k2*k2) > scale))
	  {
		  corres_tuple.push_back(std::pair<int, int>(idi0, idj0));
		  corres_tuple.push_back(std::pair<int, int>(idi1, idj1));
		  corres_tuple.push_back(std::pair<int, int>(idi2, idj2));
		  cnt++;
	  }
#endif
      if (cnt >= TUPLE_MAX_CNT)
        break;
    }

    printf("%d tuples (%d trial, %d actual).\n", cnt, number_of_trial, i);
    corres.clear();

    for (int i = 0; i < corres_tuple.size(); ++i)
      corres.push_back(std::pair<int, int>(corres_tuple[i].first, corres_tuple[i].second));
  }

  if (swapped)
  {
    std::vector<std::pair<int, int>> temp;
    for (int i = 0; i < corres.size(); i++)
      temp.push_back(std::pair<int, int>(corres[i].second, corres[i].first));
    corres.clear();
    corres = temp;
  }

  printf("\t[final] matches %d.\n", (int)corres.size());
  std::cout << "end matching " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << "s" << std::endl;
}

double FastGlobalRegistration::optimize_pairwise(bool decrease_mu, int num_iter, const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, const std::vector<std::pair<int, int> >& corres, Eigen::Matrix4d& trans_mat)
{
  printf("Pairwise rigid pose optimization\n");
  if (corres.size() < 10) return -1;

  double par = 4.0f;

  // make a float copy of v2.
  Eigen::MatrixXf pcj_copy = v_2.cast<float>();

  Eigen::Matrix4f trans;
  trans.setIdentity();

  for (int itr = 0; itr < num_iter; itr++)
  {
    // graduated non-convexity.
    if (decrease_mu)
    {
      if (itr % 4 == 0 && par > MAX_CORR_DIST) {
        par /= DIV_FACTOR;
      }
    }

    Eigen::Matrix4f delta = update_ssicp(v_1, pcj_copy, corres, par);
	//Eigen::Matrix4f delta = update_fgr(v_1, pcj_copy, corres, par);

    trans = delta * trans.eval();

    // transform point clouds
	pcj_copy = (pcj_copy.rowwise().homogeneous() * delta.transpose()).leftCols(3).eval();
  }
  trans_mat = trans.cast<double>();
  return par;
}

FGR_PUBLIC Eigen::Matrix4f FastGlobalRegistration::update_fgr(const Eigen::MatrixXd &v_1, const Eigen::MatrixXf &v_2,
  const std::vector<std::pair<int, int>> &corres, const double mu)
{
  const int nvariable = 6;	// 3 for rotation and 3 for translation
  Eigen::MatrixXd JTJ(nvariable, nvariable);
  Eigen::MatrixXd JTr(nvariable, 1);
  Eigen::MatrixXd J(nvariable, 1);
  JTJ.setZero();
  JTr.setZero();

  std::vector<double> s(corres.size(), 1.0);
  double r;
  double r2 = 0.0;
  double e = 0;
  for (int c = 0; c < corres.size(); c++) {
    int ii = corres[c].first;
    int jj = corres[c].second;
    Eigen::Vector3f p, q;
    p = v_1.row(ii).transpose().cast<float>();
    q = v_2.row(jj).transpose();
    Eigen::Vector3f rpq = p - q;

    int c2 = c;

    float temp = mu / (rpq.dot(rpq) + mu);
    s[c2] = temp * temp;

	e += temp * temp*rpq.dot(rpq) + mu * (temp - 1)*(temp - 1);

    J.setZero();
    J(1) = -q(2);
    J(2) = q(1);
    J(3) = -1;
    r = rpq(0);
    JTJ += J * J.transpose() * s[c2];
    JTr += J * r * s[c2];
    r2 += r * r * s[c2];

    J.setZero();
    J(2) = -q(0);
    J(0) = q(2);
    J(4) = -1;
    r = rpq(1);
    JTJ += J * J.transpose() * s[c2];
    JTr += J * r * s[c2];
    r2 += r * r * s[c2];

    J.setZero();
    J(0) = -q(1);
    J(1) = q(0);
    J(5) = -1;
    r = rpq(2);
    JTJ += J * J.transpose() * s[c2];
    JTr += J * r * s[c2];
    r2 += r * r * s[c2];

    r2 += (mu * (1.0 - sqrt(s[c2])) * (1.0 - sqrt(s[c2])));
  }
  std::cout << "energy: " << e << std::endl;
  Eigen::MatrixXd result(nvariable, 1);
  result = -JTJ.llt().solve(JTr);

  Eigen::Affine3d aff_mat;
  aff_mat.linear() = (Eigen::Matrix3d) Eigen::AngleAxisd(result(2), Eigen::Vector3d::UnitZ())
    * Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY())
    * Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
  aff_mat.translation() = Eigen::Vector3d(result(3), result(4), result(5));

  return aff_mat.matrix().cast<float>();
}

Eigen::Matrix3d FastGlobalRegistration::compute_r(const Eigen::MatrixXd &p_mat, const Eigen::MatrixXd &q_mat, const double mu)
{
	const int nvariable = 3;	// 3 for rotation
	Eigen::MatrixXd JTJ(nvariable, nvariable);
	Eigen::MatrixXd JTr(nvariable, 1);
	Eigen::MatrixXd J(nvariable, 1);
	JTJ.setZero();
	JTr.setZero();

	std::vector<double> s(p_mat.rows(), 1.0);
	double r;
	double r2 = 0.0;
	for (int c = 0; c < s.size(); c++) {
		Eigen::Vector3d p = p_mat.row(c).transpose();
		Eigen::Vector3d q = q_mat.row(c).transpose();
		Eigen::Vector3d rpq = p - q;

		int c2 = c;

		double temp = mu / (rpq.dot(rpq) + mu);
		s[c2] = temp * temp;

		J.setZero();
		J(1) = -q(2);
		J(2) = q(1);
		r = rpq(0);
		JTJ += J * J.transpose() * s[c2];
		JTr += J * r * s[c2];
		r2 += r * r * s[c2];

		J.setZero();
		J(2) = -q(0);
		J(0) = q(2);
		r = rpq(1);
		JTJ += J * J.transpose() * s[c2];
		JTr += J * r * s[c2];
		r2 += r * r * s[c2];

		J.setZero();
		J(0) = -q(1);
		J(1) = q(0);
		r = rpq(2);
		JTJ += J * J.transpose() * s[c2];
		JTr += J * r * s[c2];
		r2 += r * r * s[c2];

		r2 += (mu * (1.0 - sqrt(s[c2])) * (1.0 - sqrt(s[c2])));
	}

	Eigen::MatrixXd result(nvariable, 1);
	result = -JTJ.llt().solve(JTr);
	Eigen::Matrix3d r_mat;
	r_mat = Eigen::AngleAxisd(result(2), Eigen::Vector3d::UnitZ())
							* Eigen::AngleAxisd(result(1), Eigen::Vector3d::UnitY()) 
							* Eigen::AngleAxisd(result(0), Eigen::Vector3d::UnitX());
	return r_mat;
}

FGR_PUBLIC Eigen::Matrix4f FastGlobalRegistration::update_ssicp(const Eigen::MatrixXd &v_1, const Eigen::MatrixXf &v_2,
  const std::vector<std::pair<int, int>>& corres, const double mu)
{
#ifdef SHOW_DEBUG_INFO
	double e = 0;
#endif
	double sum_l = 0;
	Eigen::RowVector3d sum_l_p(0, 0, 0);
	Eigen::RowVector3d sum_l_q(0, 0, 0);
	Eigen::VectorXd sqrtl_vec(corres.size());
  Eigen::MatrixXd X(corres.size(), 3), Z(corres.size(), 3);
  for (size_t i = 0; i < corres.size(); ++i)
  {
    Eigen::Vector3f p = v_1.row(corres[i].first).transpose().cast<float>();
    Eigen::Vector3f q = v_2.row(corres[i].second).transpose();
    Eigen::Vector3f rpq = p - q;
    double sqrtl = mu / (rpq.dot(rpq) + mu);
	sqrtl_vec(i) = sqrtl;
	double l = sqrtl * sqrtl;
	sum_l += l;
	sum_l_p += l * p.transpose().cast<double>();
	sum_l_q += l * q.transpose().cast<double>();
#ifdef SHOW_DEBUG_INFO
	e += sqrtl * sqrtl*rpq.dot(rpq) + mu * (sqrtl - 1)*(sqrtl - 1);
#endif

    X.row(i) = sqrtl * v_2.row(corres[i].second).cast<double>();
    Z.row(i) = sqrtl * v_1.row(corres[i].first).cast<double>();
  }
  X -= (sqrtl_vec * sum_l_q / sum_l);
  Z -= (sqrtl_vec * sum_l_p / sum_l);

#ifdef SHOW_DEBUG_INFO
  std::cout << "energy: " << e << std::endl;
#endif
#ifdef USE_RANSAC
  Eigen::MatrixXi corres_mat(corres.size(), 2);
  for (int i = 0; i < corres.size(); ++i)
  {
	  corres_mat(i, 0) = corres[i].first;
	  corres_mat(i, 1) = corres[i].second;
  }
  Eigen::MatrixXd feature_1 = igl::slice(v_1, corres_mat.col(0), 1);
  Eigen::MatrixXf feature_2 = igl::slice(v_2, corres_mat.col(1), 1);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_1(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_2(new pcl::PointCloud<pcl::PointXYZ>);
  pointcloud_1->points.resize(feature_1.rows());
  pointcloud_2->points.resize(feature_2.rows());
  for (int i = 0; i < corres_mat.rows(); ++i)
  {
	  for (int j = 0; j < 3; ++j)
	  {
		  pointcloud_1->points[i].data[j] = static_cast<float>(feature_1(i, j));
		  pointcloud_2->points[i].data[j] = static_cast<float>(feature_2(i, j));
	  }
	  pointcloud_1->points[i].data[3] = 1.0f;
	  pointcloud_2->points[i].data[3] = 1.0f;
  }

  pcl::SampleConsensusModelRegistration<pcl::PointXYZ>::Ptr model_r(new pcl::SampleConsensusModelRegistration<pcl::PointXYZ>(pointcloud_2));
  model_r->setInputTarget(pointcloud_1);
  Eigen::VectorXf coeff;
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_r, 0.05);
  ransac.computeModel(0);
  ransac.getModelCoefficients(coeff);
  Eigen::Matrix4f transform_ransac;
  std::cout << "RANSAC transformation: " << std::endl;
  for (size_t i = 0; i<16; i++) {
	  transform_ransac(i / 4, i % 4) = coeff[i];
  }
  std::cout << transform_ransac << std::endl;
  std::vector<int> inlier_vec;
  ransac.getInliers(inlier_vec);
  Eigen::VectorXi inlier_ids = Eigen::Map<Eigen::VectorXi>(&inlier_vec[0], inlier_vec.size());
  std::cout << "inlier num: " << inlier_ids.rows() << std::endl;
  Z = igl::slice(feature_1, inlier_ids, 1);
  X = igl::slice(feature_2, inlier_ids, 1).cast<double>();
#endif

	Eigen::Matrix3d R = FastGlobalRegistration::compute_r(Z, X, mu);

	double num = (Z.array() * (X * R.transpose()).array()).sum();
	double den = X.squaredNorm();
	double s = num / den;
	Eigen::RowVector3d T = sum_l_p / sum_l - s * sum_l_q / sum_l * (R.transpose());

  Eigen::Matrix4d trans;
  trans.setZero();
  trans.block<3, 3>(0, 0) = s * R;
  trans.block<3, 1>(0, 3) = T.transpose();
  trans(3, 3) = 1;

  return trans.cast<float>();
}
