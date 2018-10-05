#ifndef RecoPixelVertexing_PixelVertexFinding_fitVertices_H
#define RecoPixelVertexing_PixelVertexFinding_fitVertices_H

#include<cstdint>
#include<cmath>
#include <algorithm>
#include<cassert>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"


#include "gpuVertexFinder.h"

namespace gpuVertexFinder {


  __global__
  void fitVertices(int nt,
                   OnGPU * pdata
                  )  {

    constexpr bool verbose = false; // in principle the compiler should optmize out if false


    auto & __restrict__ data = *pdata;
    float const * __restrict__ zt = data.zt;
    float const * __restrict__ ezt2 = data.ezt2;
    float * __restrict__ zv = data.zv;
    float * __restrict__ wv = data.wv;
    float * __restrict__ chi2 = data.chi2;
    uint32_t & nv  = *data.nv;
    uint32_t & nv2 = *data.nv2;

    int32_t * __restrict__ nn = data.nn;
    int32_t * __restrict__ iv = data.iv;

    assert(pdata);
    assert(zt);

   assert(nv<=nv2);
   nv = nv2;
   auto foundClusters = nv2;
 
   // zero
   for (int i = threadIdx.x; i < foundClusters; i += blockDim.x) {
      zv[i]=0;
      wv[i]=0;
      chi2[i]=0;
   }

      // only for test
    __shared__ int noise;
   if(verbose && 0==threadIdx.x) noise = 0;

    __syncthreads();

    // compute cluster location
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) {
        if (verbose) atomicAdd(&noise, 1);
        continue;
      }
      assert(iv[i]>=0);
      assert(iv[i]<foundClusters);
      auto w = 1.f/ezt2[i];
      atomicAdd(&zv[iv[i]],zt[i]*w);
      atomicAdd(&wv[iv[i]],w);
    }

    __syncthreads();
    // reuse nn
    for (int i = threadIdx.x; i < foundClusters; i += blockDim.x) {
      assert(wv[i]>0.f);
      zv[i]/=wv[i];
      nn[i]=-1;  // ndof
    }
    __syncthreads();


    // compute chi2
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) continue;

      auto c2 = zv[iv[i]]-zt[i]; c2 *=c2/ezt2[i];
      // remove outliers ???? if (c2> cut) {iv[i] = 9999; continue;}????
      atomicAdd(&chi2[iv[i]],c2);
      atomicAdd(&nn[iv[i]],1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < foundClusters; i += blockDim.x) if(nn[i]>0) wv[i] *= float(nn[i])/chi2[i];

    if(verbose && 0==threadIdx.x) printf("found %d proto clusters ",foundClusters);
    if(verbose && 0==threadIdx.x) printf("and %d noise\n",noise);

  }

}

#endif
