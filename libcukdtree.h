#pragma once
#include <stdint.h>

#ifdef _WIN32
#define LIB_CUKDTREE_API __declspec(dllexport) 
#else
#define LIB_CUKDTREE_API
#endif

extern "C"
{
	void LIB_CUKDTREE_API cukdtree_build(int numPoints, void* points);

	void LIB_CUKDTREE_API cukdtree_knn(int k, int numSearchPoints, void* searchPoints, int numPoints, void* points, uint32_t* indices, float* sqrDistances);

    void LIB_CUKDTREE_API cukdtree_test();
}