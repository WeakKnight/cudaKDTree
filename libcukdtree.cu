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