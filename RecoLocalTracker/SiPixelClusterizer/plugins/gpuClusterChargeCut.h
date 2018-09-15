#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuClusterChargeCut_h

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "gpuClusteringConstants.h"

#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"

namespace gpuClustering {

  __global__ void  clusterChargeCut(
                           uint16_t * __restrict__ id,             // module id of each pixel (modified if bad cluster)
                           uint16_t const * __restrict__ adc,              //  charge of each pixel
                           uint32_t const * __restrict__ moduleStart,    // index of the first pixel of each module
                           uint32_t * __restrict__ nClustersInModule,    // modified: number of clusters found in each module
                           uint32_t const * __restrict__ moduleId,             // module id of each module
                           int32_t * __restrict__  clusterId,            // modified: cluster id of each pixel
                           int numElements)
  {

    if (blockIdx.x >= moduleStart[0])
      return;

    auto firstPixel = moduleStart[1 + blockIdx.x];
    auto thisModuleId = id[firstPixel];
    assert(thisModuleId < MaxNumModules);

    auto nclus = nClustersInModule[thisModuleId];
    if (nclus==0) return;

    assert(nclus<=MaxNumClustersPerModules);

#ifdef GPU_DEBUG
    if (thisModuleId % 100 == 1)
      if (threadIdx.x == 0)
        printf("start clusterizer for module %d in block %d\n", thisModuleId, blockIdx.x);
#endif

    auto first = firstPixel + threadIdx.x;

    __shared__ int32_t charge[MaxNumClustersPerModules];
    for (int i=threadIdx.x; i<nclus; i += blockDim.x) {
      charge[i]=0;
    }

    for (int i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId) continue;     // not valid
      if (id[i] != thisModuleId) break;           // end of module
      atomicAdd(&charge[clusterId[i]], adc[i]);
    }
    __syncthreads();

    auto chargeCut = thisModuleId<96 ? 2000 : 4000; // move in constants (calib?)
    __shared__ uint16_t newclusId[MaxNumClustersPerModules];
    for (int i=threadIdx.x; i<nclus; i += blockDim.x) {
       newclusId[i] =  charge[i]>chargeCut ? 1 : 0;
    }

    // renumber
    __shared__ uint16_t ws[32];
    blockPrefixScan(newclusId, nclus, ws);
    
    assert(nclus>=newclusId[nclus-1]);
    
    if(nclus==newclusId[nclus-1]) return;

    nClustersInModule[thisModuleId] = newclusId[nclus-1];

    // mark bad cluster with 0 again
    for (int i=threadIdx.x+1; i<nclus; i += blockDim.x) {
      if (newclusId[i]==newclusId[i-1]) newclusId[i]=InvId+1;
    }
    __syncthreads();
    if (newclusId[0]==0) newclusId[0]=InvId+1;

    // reassign id
    for (int i = first; i < numElements; i += blockDim.x) {
      if (id[i] == InvId) continue;     // not valid
      if (id[i] != thisModuleId) break;           // end of module
      clusterId[i] = newclusId[clusterId[i]]-1;
      if(clusterId[i]==InvId) id[i] = InvId;
    }

    //done
  }


} // namespace
#endif
