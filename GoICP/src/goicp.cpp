#include "goicp.h"
#include <jly_goicp.h>

void GOICP::goicp(const Eigen::MatrixXd& v_1, const Eigen::MatrixXd& v_2, Eigen::MatrixXd& aligned_v_2)
{
	clock_t  clockBegin, clockEnd;
	int NdDownsampled = 0;
	POINT3D *pModel, *pData;
	pModel = (POINT3D *)malloc(sizeof(POINT3D) * v_1.rows());
	for (int i = 0; i < v_1.rows(); i++)
	{
		pModel[i].x = v_1(i, 0);
		pModel[i].y = v_1(i, 1);
		pModel[i].z = v_1(i, 2);
	}

	pData = (POINT3D *)malloc(sizeof(POINT3D) * v_2.rows());
	for (int i = 0; i < v_2.rows(); i++)
	{
		pData[i].x = v_2(i, 0);
		pData[i].y = v_2(i, 1);
		pData[i].z = v_2(i, 2);
	}

	GoICP goicp;
	////////////config
	// Mean Squared Error(MSE) convergence threshold
	goicp.MSEThresh = 0.001;
	// Smallest rotation value along dimension X of rotation cube(radians)
	//goicp.initNodeRot.a = -3.1416;
	goicp.initNodeRot.a = -0.31416;
	// Smallest rotation value along dimension Y of rotation cube(radians)
	goicp.initNodeRot.b = -0.31416;
	// Smallest rotation value along dimension Z of rotation cube(radians)
	goicp.initNodeRot.c = -0.31416;
	// Side length of each dimension of rotation cube(radians)
	goicp.initNodeRot.w = 0.62832;
	// Smallest translation value along dimension X of translation cube
	goicp.initNodeTrans.x = -0.3;
	// Smallest translation value along dimension Y of translation cube
	goicp.initNodeTrans.y = -0.3;
	// Smallest translation value along dimension Z of translation cube
	goicp.initNodeTrans.z = -0.3;
	// Side length of each dimension of translation cube
	goicp.initNodeTrans.w = 0.6;
	// Set to 0.0 for no trimming
	goicp.trimFraction = 0.05;
	// If < 0.1% trimming specified, do no trimming
	if (goicp.trimFraction < 0.001)
	{
		goicp.doTrim = false;
	}
	// Nodes per dimension of distance transform
	goicp.dt.SIZE = 300;
	// DistanceTransformWidth = ExpandFactor x WidthLargestDimension
	goicp.dt.expandFactor = 2.0;

	////////////config

	goicp.pModel = pModel;
	goicp.Nm = v_1.rows();
	goicp.pData = pData;
	goicp.Nd = v_2.rows();

	// Build Distance Transform
	cout << "Building Distance Transform..." << flush;
	clockBegin = clock();
	goicp.BuildDT();
	clockEnd = clock();
	cout << (double)(clockEnd - clockBegin) / CLOCKS_PER_SEC << "s (CPU)" << endl;

	// Run GO-ICP
	if (NdDownsampled > 0)
	{
		goicp.Nd = NdDownsampled; // Only use first NdDownsampled data points (assumes data points are randomly ordered)
	}
	cout << "Model: " << " (" << goicp.Nm << "), Data: " << " (" << goicp.Nd << ")" << endl;
	cout << "Registering..." << endl;
	clockBegin = clock();
	goicp.Register();
	clockEnd = clock();
	double time = (double)(clockEnd - clockBegin) / CLOCKS_PER_SEC;
	cout << "Optimal Rotation Matrix:" << endl;
	cout << goicp.optR << endl;
	cout << "Optimal Translation Vector:" << endl;
	cout << goicp.optT << endl;
	cout << "Finished in " << time << endl;

	Eigen::Matrix4d affine_mat;
	affine_mat.setZero();
	for(int i = 0; i < 3; ++i)
	{
		for(int j = 0; j < 3; ++j)
		{
			affine_mat(i, j) = static_cast<double>(goicp.optR.val[i][j]);
		}
	}
	for(int i = 0; i < 3; ++i)
	{
		affine_mat(i, 3) = static_cast<double>(goicp.optT.val[i][0]);
	}
	affine_mat(3, 3) = 1;
	Eigen::Affine3d affine_trans(affine_mat);
	Eigen::MatrixXd output_tmp;
	output_tmp.resize(v_2.rows(), 3);
	for (int i = 0; i < output_tmp.rows(); ++i)
	{
		Eigen::Vector3d tmp_v = v_2.row(i).transpose();
		output_tmp.row(i) = (affine_trans * tmp_v).transpose();
	}
	aligned_v_2 = output_tmp;

	delete(pModel);
	delete(pData);
}
