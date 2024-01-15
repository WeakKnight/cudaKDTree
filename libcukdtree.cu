#include "libcukdtree.h"

#include "cukd/builder.h"
#include "cukd/knn.h"

using namespace cukd;

using data_t = float3;
using data_traits = default_data_traits<float3>;

#define CU_KNN_MAX_RADIUS 10.0f

__global__
void d_knn_4(
	int k,
	float3* searchPoints,
	int      numSearchPoints,
	float3* points,
	int      numPoints,
	uint32_t* indices, float* sqrDistances)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numSearchPoints)
	{
		return;
	}

	FixedCandidateList<4> stackListResults(CU_KNN_MAX_RADIUS);
	stackBased::knn(stackListResults, searchPoints[tid], points, numPoints);

	for (int i = 0; i < k; i++)
	{
		int pointId = stackListResults.get_pointID(i);
		indices[tid * k + i] = pointId >= 0 ? pointId : ~0u;
		sqrDistances[tid * k + i] = stackListResults.get_dist2(i);
	}
}

__global__
void d_knn_8(
	int k,
	float3* searchPoints,
	int      numSearchPoints,
	float3* points,
	int      numPoints,
	uint32_t* indices, float* sqrDistances)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numSearchPoints)
	{
		return;
	}

	FixedCandidateList<8> stackListResults(CU_KNN_MAX_RADIUS);
	stackBased::knn(stackListResults, searchPoints[tid], points, numPoints);

	for (int i = 0; i < k; i++)
	{
		int pointId = stackListResults.get_pointID(i);
		indices[tid * k + i] = pointId >= 0 ? pointId : ~0u;
		sqrDistances[tid * k + i] = stackListResults.get_dist2(i);
	}
}

__global__
void d_knn_12(
	int k,
	float3* searchPoints,
	int      numSearchPoints,
	float3* points,
	int      numPoints,
	uint32_t* indices, float* sqrDistances)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numSearchPoints)
	{
		return;
	}

	FixedCandidateList<12> stackListResults(CU_KNN_MAX_RADIUS);
	stackBased::knn(stackListResults, searchPoints[tid], points, numPoints);

	for (int i = 0; i < k; i++)
	{
		int pointId = stackListResults.get_pointID(i);
		indices[tid * k + i] = pointId >= 0 ? pointId : ~0u;
		sqrDistances[tid * k + i] = stackListResults.get_dist2(i);
	}
}

void cukdtree_build(int numPoints, void* points)
{
	cukd::buildTree((float3*)points, numPoints);
	CUKD_CUDA_SYNC_CHECK();
}

void cukdtree_knn(int k, int numSearchPoints, void* searchPoints, int numPoints, void* points, uint32_t* indices, float* sqrDistances)
{
	int bs = 128;
	int nb = divRoundUp((int)numSearchPoints, bs);

	if (k <= 4)
	{
		d_knn_4 << <nb, bs >> > (k, (float3*)searchPoints, numSearchPoints, (float3*)points, numPoints, indices, sqrDistances);
	}
	else if (k <= 8)
	{
		d_knn_8 << <nb, bs >> > (k, (float3*)searchPoints, numSearchPoints, (float3*)points, numPoints, indices, sqrDistances);
	}
	else if (k <= 12)
	{
		d_knn_12 << <nb, bs >> > (k, (float3*)searchPoints, numSearchPoints, (float3*)points, numPoints, indices, sqrDistances);
	}
}

void cukdtree_test()
{
	int pointNum = 10;
	float3* points;
	cudaMallocManaged((char **)&points, pointNum * sizeof(*points));
	for (int i = 0; i < pointNum; i++)
	{
		points[i].x = i;
		points[i].y = 0.0f;
		points[i].z = 0.0f;
	}

	int queryNum = 1;
	float3* queryPoints;
	cudaMallocManaged((char **)&queryPoints, queryNum * sizeof(*queryPoints));
	queryPoints[0].x = 2.0f;
	queryPoints[0].y = 0.0f;
	queryPoints[0].z = 0.0f;

	int k = 3;

	uint32_t* resultIndices;
	cudaMallocManaged((char **)&resultIndices, k * queryNum * sizeof(*resultIndices));

	float* resultSqrDistances;
	cudaMallocManaged((char **)&resultSqrDistances, k * queryNum * sizeof(*resultSqrDistances));

	cukdtree_knn(k, queryNum, queryPoints, pointNum, points, resultIndices, resultSqrDistances);

	cudaDeviceSynchronize();

	for (int searchPosIndex = 0; searchPosIndex < queryNum; searchPosIndex++)
	{
		float3 searchPos = queryPoints[searchPosIndex];
		printf("Search Index: %u, Position: %f, %f, %f \n", searchPosIndex, searchPos.x, searchPos.y, searchPos.z);
		for (int i = 0; i < k; i++)
		{
			uint32_t index = resultIndices[i];
			float sqrDistance = resultSqrDistances[i];
			printf("=== Neighbor Index: %u \n", index);
			float3 neighborPos = points[index];
			printf("=== Neighbor Position: %f, %f, %f \n", neighborPos.x, neighborPos.y, neighborPos.z);
			printf("=== Neighbor SqrDistance: %f \n", sqrDistance);
		}
	}
}