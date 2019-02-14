#ifndef RecoPixelVertexing_PixelVertexFinding_sortByPt2_H
#define RecoPixelVertexing_PixelVertexFinding_sortByPt2_H


#include<cstdint>
#include<cmath>
#include <algorithm>
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#ifdef __CUDA_ARCH__
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"
#endif

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {

  __global__
  void sortByPt2(
                 ZVertices * pdata, WorkSpace * pws
                )  {
    auto & __restrict__ data = *pdata;
    auto & __restrict__ ws = *pws;
    auto nt = ws.ntrks;
    float const * __restrict__ ptt2 = ws.ptt2;
    uint32_t const & nvFinal = data.nvFinal;

    int32_t const * __restrict__ iv = ws.iv;
    float * __restrict__ ptv2 = data.ptv2;
    uint16_t * __restrict__ sortInd = data.sortInd;

    if (nvFinal<1) return;

    // can be done asynchronoisly at the end of previous event
    for (auto i = threadIdx.x; i < nvFinal; i += blockDim.x) {
      ptv2[i]=0;
    }
    __syncthreads();


    for (auto i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) continue;
      atomicAdd(&ptv2[iv[i]], ptt2[i]);
    }
    __syncthreads();

    if (1==nvFinal) {
      if (threadIdx.x==0) sortInd[0]=0;
      return;
    }
#ifdef __CUDA_ARCH__
    __shared__ uint16_t sws[1024];
    // sort using only 16 bits
    radixSort<float,2>(ptv2,sortInd,sws,nvFinal);
#else
    for(uint16_t i=0; i<nvFinal; ++i) sortInd[i]=i;
    std::sort(sortInd,sortInd+nvFinal,[&](auto i, auto j){return ptv2[i]<ptv2[j];});
#endif
  }

}

#endif
