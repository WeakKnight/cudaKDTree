// ======================================================================== //
// Copyright 2018-2023 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "cukd/builder.h"
// fcp = "find closest point" query
#include "cukd/fcp.h"

#if D_FROM_CMAKE == 2
using floatN = float2;
#elif D_FROM_CMAKE == 3
using floatN = float3;
#elif D_FROM_CMAKE == 4
using floatN = float4;
#elif D_FROM_CMAKE == 8
using floatN = cukd::vec_float<8>;
#else
#pragma error("error ... should get a value of 2, 3, or 4 from cmakefile...")
#endif

using namespace cukd;

__global__ void d_fcp(unsigned long long *d_stats,
                      int *d_results,
                      floatN *d_queries,
                      int numQueries,
#if CUKD_IMPROVED_TRAVERSAL
                      const cukd::box_t<floatN> *d_bounds,
#endif
                      floatN *d_nodes,
                      int numNodes)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;

  d_results[tid]
    = cukd::fcp
    (d_stats,d_queries[tid],
#if CUKD_IMPROVED_TRAVERSAL
     *d_bounds,
#endif
     d_nodes,numNodes);
}

void fcp(int *d_results,
         floatN *d_queries,
         int numQueries,
#if CUKD_IMPROVED_TRAVERSAL
         const cukd::box_t<floatN> *d_bounds,
#endif
         floatN *d_nodes,
         int numNodes)
{
  int bs = 128;
  int nb = divRoundUp(numQueries,bs);
  unsigned long long *d_stats = 0;
  static bool firstTime = true;
  if (firstTime) {
    cudaMallocManaged((char **)&d_stats,sizeof(*d_stats));
    *d_stats = 0;
  }
  d_fcp<<<nb,bs>>>(d_stats,d_results,d_queries,numQueries,
#if CUKD_IMPROVED_TRAVERSAL
                   d_bounds,
#endif
                   d_nodes,numNodes);
  if (firstTime) {
    cudaDeviceSynchronize();
    std::cout << "KDTREE_STATS " << *d_stats << std::endl;
    cudaFree(d_stats);
    firstTime = false;
  }
}

bool noneBelow(floatN *d_points, int N, int curr, int dim, float value)
{
  if (curr >= N) return true;
  return
    ((&d_points[curr].x)[dim] >= value)
    && noneBelow(d_points,N,2*curr+1,dim,value)
    && noneBelow(d_points,N,2*curr+2,dim,value);
}

bool noneAbove(floatN *d_points, int N, int curr, int dim, float value)
{
  if (curr >= N) return true;
  return
    ((&d_points[curr].x)[dim] <= value)
    && noneAbove(d_points,N,2*curr+1,dim,value)
    && noneAbove(d_points,N,2*curr+2,dim,value);
}

bool checkTree(floatN *d_points, int N, int curr=0)
{
  if (curr >= N) return true;

  int dim = cukd::BinaryTree::levelOf(curr)%4;
  float value = (&d_points[curr].x)[dim];
  
  if (!noneAbove(d_points,N,2*curr+1,dim,value))
    return false;
  if (!noneBelow(d_points,N,2*curr+2,dim,value))
    return false;
  
  return
    checkTree(d_points,N,2*curr+1)
    &&
    checkTree(d_points,N,2*curr+2);
}

int main(int ac, const char **av)
{
  using namespace cukd::common;

  int nPoints = 173;
  bool verify = false;
  // float maxQueryRadius = std::numeric_limits<float>::infinity();
  int nRepeats = 1;
  size_t nQueries = 10000000;
  for (int i=1;i<ac;i++) {
    std::string arg = av[i];
    if (arg[0] != '-')
      nPoints = std::stoi(arg);
    else if (arg == "-v")
      verify = true;
    else if (arg == "-nq")
      nQueries = atoi(av[++i]);
    else if (arg == "-nr")
      nRepeats = atoi(av[++i]);
    // else if (arg == "-r")
    //   maxQueryRadius = std::stof(av[++i]);
    else
      throw std::runtime_error("known cmdline arg "+arg);
  }
  
  floatN *d_points = loadPoints<floatN>("data_points",nPoints);//generatePoints(nPoints);
  // floatN *d_points = generatePoints(nPoints);
  
#if CUKD_IMPROVED_TRAVERSAL
  cukd::box_t<floatN> *d_bounds;
  cudaMalloc((void**)&d_bounds,sizeof(cukd::box_t<floatN>));
  cukd::computeBounds
    (d_bounds,d_points,nPoints);
#endif
  {
    double t0 = getCurrentTime();
    std::cout << "calling builder..." << std::endl;
    cukd::buildTree(d_points,nPoints);
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done building tree, took " << prettyDouble(t1-t0) << "s" << std::endl;
  }

  if (verify) {
    std::cout << "checking tree..." << std::endl;
    if (!checkTree(d_points,nPoints))
      throw std::runtime_error("not a valid kd-tree!?");
    else
      std::cout << "... passed" << std::endl;
  }

  // floatN *d_queries = generatePoints(nQueries);
  floatN *d_queries = loadPoints<floatN>("query_points",nQueries);
  int    *d_results;
  CUKD_CUDA_CALL(MallocManaged((void**)&d_results,nQueries*sizeof(int)));
  {
    double t0 = getCurrentTime();
    for (int i=0;i<nRepeats;i++) {
      fcp(d_results,d_queries,nQueries,
#if CUKD_IMPROVED_TRAVERSAL
          d_bounds,
#endif
          d_points,nPoints);
    }
    CUKD_CUDA_SYNC_CHECK();
    double t1 = getCurrentTime();
    std::cout << "done " << nRepeats << " iterations of " << nQueries << " fcp queries, took " << prettyDouble(t1-t0) << "s" << std::endl;
    std::cout << "that is " << prettyDouble(nQueries*nRepeats/(t1-t0)) << " queries/s" << std::endl;
  }
  
  if (verify) {
    std::cout << "verifying ..." << std::endl;
    for (int i=0;i<nQueries;i++) {
      if (d_results[i] == -1) continue;
      
      floatN qp = d_queries[i];
      float reportedDist
        = cukd::distance(qp,d_points[d_results[i]]);
      for (int j=0;j<nPoints;j++) {
        float dist_j = cukd::distance(qp,d_points[j]);
        if (dist_j < reportedDist) {
#if D_FROM_CMAKE == 2
#elif D_FROM_CMAKE == 3
#elif D_FROM_CMAKE == 4
#pragma error("error ... should get a value of 2, 3, or 4 from cmakefile...")
          printf("for query %i: found offending point %i (%f %f %f %f) with dist %f (vs %f)\n",
                 i,
                 j,
                 d_points[j].x,
                 d_points[j].y,
                 d_points[j].z,
                 d_points[j].w,
                 dist_j,
                 reportedDist);
#elif D_FROM_CMAKE == 8
#else
#endif          
          throw std::runtime_error("verification failed ...");
        }
      }
    }
    std::cout << "verification succeeded... done." << std::endl;
  }
}
