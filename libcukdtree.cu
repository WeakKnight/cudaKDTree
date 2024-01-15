#include "libcukdtree.h"

#include "cukd/builder.h"
#include "cukd/knn.h"

using namespace cukd;

#define CU_KNN_MAX_RADIUS 10.0f

struct PointPlusPayload 
{
	float3 position;
	uint32_t payload;

	operator float3() const
	{
		return position;
	} 
};

struct PointPlusPayload_traits : public cukd::default_data_traits<float3>
{
	using point_t = float3;

	static inline __device__ __host__
	float3 get_point(const PointPlusPayload &data)
	{ 
		return data.position; 
	}

	static inline __device__ __host__
	float  get_coord(const PointPlusPayload &data, int dim)
	{ 
		return cukd::get_coord(get_point(data),dim); 
	}
};

__global__
void d_knn_4(
	int k,
	float3* searchPoints,
	int      numSearchPoints,
	PointPlusPayload* points,
	int      numPoints,
	uint32_t* indices, float* sqrDistances)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numSearchPoints)
	{
		return;
	}

	FixedCandidateList<4> stackListResults(CU_KNN_MAX_RADIUS);
	stackBased::knn<FixedCandidateList<4>, PointPlusPayload, PointPlusPayload_traits>(stackListResults, searchPoints[tid], points, numPoints);

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
	PointPlusPayload* points,
	int      numPoints,
	uint32_t* indices, float* sqrDistances)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numSearchPoints)
	{
		return;
	}

	FixedCandidateList<8> stackListResults(CU_KNN_MAX_RADIUS);
	stackBased::knn<FixedCandidateList<8>, PointPlusPayload, PointPlusPayload_traits>(stackListResults, searchPoints[tid], points, numPoints);

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
	PointPlusPayload* points,
	int      numPoints,
	uint32_t* indices, float* sqrDistances)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= numSearchPoints)
	{
		return;
	}

	FixedCandidateList<12> stackListResults(CU_KNN_MAX_RADIUS);
	stackBased::knn<FixedCandidateList<12>, PointPlusPayload, PointPlusPayload_traits>(stackListResults, searchPoints[tid], points, numPoints);

	for (int i = 0; i < k; i++)
	{
		int pointId = stackListResults.get_pointID(i);
		indices[tid * k + i] = pointId >= 0 ? pointId : ~0u;
		sqrDistances[tid * k + i] = stackListResults.get_dist2(i);
	}
}

void cukdtree_build(int numPoints, void* points)
{
	cukd::buildTree<PointPlusPayload, PointPlusPayload_traits>((PointPlusPayload*)points, numPoints);
	CUKD_CUDA_SYNC_CHECK();
}

void cukdtree_knn(int k, int numSearchPoints, void* searchPoints, int numPoints, void* points, uint32_t* indices, float* sqrDistances)
{
	int bs = 128;
	int nb = divRoundUp((int)numSearchPoints, bs);

	if (k <= 4)
	{
		d_knn_4 << <nb, bs >> > (k, (float3*)searchPoints, numSearchPoints, (PointPlusPayload*)points, numPoints, indices, sqrDistances);
	}
	else if (k <= 8)
	{
		d_knn_8 << <nb, bs >> > (k, (float3*)searchPoints, numSearchPoints, (PointPlusPayload*)points, numPoints, indices, sqrDistances);
	}
	else if (k <= 12)
	{
		d_knn_12 << <nb, bs >> > (k, (float3*)searchPoints, numSearchPoints, (PointPlusPayload*)points, numPoints, indices, sqrDistances);
	}
}

void cukdtree_test()
{
	int pointNum = 10;
	PointPlusPayload* points;
	cudaMallocManaged((char **)&points, pointNum * sizeof(*points));
	for (int i = 0; i < pointNum; i++)
	{
		points[i].position.x = i;
		points[i].position.y = 0.0f;
		points[i].position.z = 0.0f;
		points[i].payload = i;
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

  	cukdtree_build(pointNum, points);

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
			PointPlusPayload neighborPos = points[index];

			printf("=== Neighbor Position: %f, %f, %f \n", neighborPos.position.x, neighborPos.position.y, neighborPos.position.z);
			printf("=== Neighbor Index: %u \n", neighborPos.payload);
			printf("=== Neighbor SqrDistance: %f \n", sqrDistance);
		}
	}
}
