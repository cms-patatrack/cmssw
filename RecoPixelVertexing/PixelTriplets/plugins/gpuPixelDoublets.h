#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

#include "GPUCACell.h"

namespace gpuPixelDoublets {

  constexpr uint32_t MaxNumOfDoublets = 1024*1024*256;

  template<typename Hist>
  __device__
  void doubletsFromHisto(uint8_t const * layerPairs, uint32_t nPairs, GPUCACell * cells, uint32_t * nCells,
                         int16_t const * iphi, Hist const * hist, uint32_t const * offsets, float phiCut,
                         siPixelRecHitsHeterogeneousProduct::HitsOnGPU const & hh) {
    auto iphicut = phi2short(phiCut);

    auto idx = blockIdx.x*blockDim.x + threadIdx.x;

    auto layerSize = [=](uint8_t li) { return offsets[li+1]-offsets[li]; };

    // to be optimized later
    uint32_t innerLayerCumulaliveSize[64];
    assert(nPairs<=64);
    innerLayerCumulaliveSize[0] = layerSize(layerPairs[0]);
    for (uint32_t i=1; i<nPairs; ++i) {
       innerLayerCumulaliveSize[i] = innerLayerCumulaliveSize[i-1] + layerSize(layerPairs[2*i]);
    }

    auto ntot = innerLayerCumulaliveSize[nPairs-1];
    if (idx>=ntot) return;  // will move to for(auto j=idx;j<ntot;j+=blockDim.x*gridDim.x) {
    auto j = idx; 

    uint32_t pairLayerId=0;
    while(j>=innerLayerCumulaliveSize[pairLayerId++]);  --pairLayerId; // move to lower_bound ??

    assert(pairLayerId<nPairs);
    assert(j<innerLayerCumulaliveSize[pairLayerId]);
    assert(0==pairLayerId || j>=innerLayerCumulaliveSize[pairLayerId-1]);

    uint8_t inner = layerPairs[2*pairLayerId];
    uint8_t outer = layerPairs[2*pairLayerId+1];

    auto i = (0==pairLayerId) ? 0 :  j-innerLayerCumulaliveSize[pairLayerId-1];
    i += offsets[inner];

    // printf("Hit in Layer %d %d %d %d\n", i, inner, pairLayerId, j);

    assert(i>=offsets[inner]);
    assert(i<offsets[inner+1]);

    // found hit corresponding to our cuda thread!!!!!
    // do the job

    auto mep = iphi[i];
    auto kl = hist[outer].bin(mep-iphicut);
    auto kh = hist[outer].bin(mep+iphicut);
    auto incr = [](auto & k) { return k = (k+1)%Hist::nbins();};
    int tot  = 0;
    int nmin = 0;
    auto khh = kh;
    incr(khh);
    for (auto kk=kl; kk!=khh; incr(kk)) {
      if (kk!=kl && kk!=kh) nmin+=hist[outer].size(kk);
      for (auto p=hist[outer].begin(kk); p<hist[outer].end(kk); ++p) {
        if (std::min(std::abs(int16_t(iphi[*p]-mep)), std::abs(int16_t(mep-iphi[*p]))) > iphicut)
          continue;
        auto ind = atomicInc(nCells,MaxNumOfDoublets);
        // int layerPairId, int doubletId, int innerHitId,int outerHitId)
        cells[ind].init(hh,pairLayerId,ind,i,*p);
        ++tot;
      }
    }
    if (0==hist[outer].nspills) assert(tot>=nmin);
    // look in spill bin as well....
  }

  /*
  __global__
  void getDoubletsFromHisto(uint8_t const * layerPairs, uint32_t nPairs, GPUCACell * cells, uint32_t * nCells,
                            siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp, float phiCut) {
    auto const & hh = *hhp;
    doubletsFromHisto(layerPairs, nPairs, cells, nCells, 
                      hh.iphi_d,hh.hist_d,hh.hitsLayerStart_d,phiCut, hh);
  }
 */


  __global__
  void getDoubletsFromHisto(GPUCACell * cells, uint32_t * nCells, siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp, float phiCut) {

    uint8_t const layerPairs[2*13] = {0,1 ,1,2 ,2,3 
                                     ,0,4 ,1,4 ,2,4 ,4,5 ,5,6  
                                     ,0,7 ,1,7 ,2,7 ,7,8 ,8,9
                                     };

    auto const & hh = *hhp;
    doubletsFromHisto(layerPairs, 13, cells, nCells, 
                      hh.iphi_d,hh.hist_d,hh.hitsLayerStart_d,phiCut,hh);
  }



} // namespace end

#endif // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
