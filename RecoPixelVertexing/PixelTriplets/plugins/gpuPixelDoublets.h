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

namespace gpuPixelDoublets {

  __device__
  std::pair<int,int>
  findPhiLimits(int16_t phiMe, int16_t * iphi, uint16_t * index, uint16_t size, int16_t iphicut) {

  assert(iphicut>0);

  // find extreemes in top
  int16_t minPhi = phiMe-iphicut;
  int16_t maxPhi = phiMe+iphicut;

  // std::cout << "\n phi min/max " << phiMe << ' ' << minPhi << ' ' << maxPhi << std::endl;

  // guess and adjust
  auto findLimit = [&](int16_t mPhi) {
    int jm = float(0.5f*size)*(1.f+float(mPhi)/float(std::numeric_limits<short>::max()));
    // std::cout << "jm for " << mPhi << ' ' << jm << std::endl;
    jm = std::min(size-1,std::max(0,jm));
    bool notDone=true;
    while(jm>0 && mPhi<iphi[index[--jm]]){notDone=false;}
    if (notDone) while(jm<size && mPhi>iphi[index[++jm]]){}
    jm = std::min(size-1,std::max(0,jm));
    return jm;
  };

  auto jmin = findLimit(minPhi);
  auto jmax = findLimit(maxPhi);


  /*
  std::cout << "j min/max " << jmin << ' ' << jmax << std::endl;
  std::cout << "found min/max " << iphi[index[jmin]] << ' ' << iphi[index[jmax]] << std::endl;
  std::cout << "found min/max +1 " << iphi[index[jmin+1]] << ' ' << iphi[index[jmax+1]] << std::endl;
  std::cout << "found min/max -1 " << iphi[index[jmin-1]] << ' ' << iphi[index[jmax-1]] << std::endl;
  */

  return std::make_pair(jmin,jmax);
  }


  __global__
  void getDoubletsFromSorted(int16_t * iphi, uint16_t * index, uint32_t * offsets, float phiCut) {
    auto iphicut = phi2short(phiCut);
    auto i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i>=offsets[9]) {
      // get rid of last layer
      return;
    }

    assert(0==offsets[0]);
    int top = (i>offsets[5]) ? 5: 0;
    while (i>=offsets[++top]){};
    assert(top<10);
    auto bottom = top-1;
    if (bottom == 3 or bottom == 6) {
      // do not have UP... (9 we got rid already)
      return;
    }
    assert(i >= offsets[bottom]);
    assert(i < offsets[top]);

    if (index[i]>= (offsets[top]-offsets[bottom])) {
      printf("index problem: %d %d %d %d %d\n",i, offsets[top], offsets[bottom], offsets[top]-offsets[bottom], index[i]);
      return;
    }

    assert(index[i]<offsets[top]-offsets[bottom]);

    int16_t phiMe = iphi[offsets[bottom]+index[i]];

    auto size = offsets[top+1] - offsets[top];
    assert(size<std::numeric_limits<uint16_t>::max());

    auto jLimits = findPhiLimits(phiMe, iphi+offsets[top],index+offsets[top],size,iphicut);

    auto slidingWindow = [&](uint16_t mysize, uint16_t mymin,uint16_t mymax) {
      auto topPhi = iphi+offsets[top];
      uint16_t imax =  std::numeric_limits<uint16_t>::max();
      uint16_t offset = (mymin>mymax) ? imax-(mysize-1) : 0;
      int n=0;
      for (uint16_t i = mymin+offset; i!=mymax; i++) {
        assert(i<=imax);
        uint16_t k = (i>mymax) ? i-offset : i;
        assert(k<mysize);
        assert(k>=mymin || k<mymax);
        if (int16_t(topPhi[k]-phiMe)>2*iphicut && int16_t(phiMe-topPhi[k])>2*iphicut)
          printf("deltaPhi problem: %d %d %d %d, deltas %d:%d cut %d\n",i,k,phiMe,topPhi[k],int16_t(topPhi[k]-phiMe),int16_t(phiMe-topPhi[k]),iphicut);
        n++;
      }
      int tot = (mymin>mymax) ? (mysize-mymin)+mymax : mymax-mymin;
      assert(n==tot);
    };

    slidingWindow(size,jLimits.first,jLimits.second);
  }

  template<typename Hist>
  __device__
  void doubletsFromHisto(uint8_t const * layerPairs, uint32_t nPairs, 
                         int16_t const * iphi, Hist const * hist, uint32_t const * offsets, float phiCut) {
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

    uint32_t innerLayerId=0;
    while(j>=innerLayerCumulaliveSize[innerLayerId++]);  --innerLayerId; // move to lower_bound ??

    assert(innerLayerId<nPairs);
    assert(j<innerLayerCumulaliveSize[innerLayerId]);
    assert(0==innerLayerId || j>=innerLayerCumulaliveSize[innerLayerId-1]);

    uint8_t inner = layerPairs[2*innerLayerId];
    uint8_t outer = layerPairs[2*innerLayerId+1];

    auto i = (0==innerLayerId) ? 0 :  j-innerLayerCumulaliveSize[innerLayerId-1];
    i += offsets[inner];

    // printf("Hit in Layer %d %d %d %d\n", i, inner, innerLayerId, j);

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
        ++tot;
      }
    }
    if (0==hist[outer].nspills) assert(tot>=nmin);
    // look in spill bin as well....
  }

  __global__
  void getDoubletsFromHisto(uint8_t const * layerPairs, uint32_t nPairs, 
                            siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp, float phiCut) {
    auto const & hh = *hhp;
    doubletsFromHisto(layerPairs, nPairs, hh.iphi_d,hh.hist_d,hh.hitsLayerStart_d,phiCut);
  }


  __global__
  void getDoubletsFromHisto( siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp, float phiCut) {

    uint8_t const layerPairs[2*13] = {0,1 ,1,2 ,2,3 
                                     ,0,4 ,1,4 ,2,4 ,4,5 ,5,6  
                                     ,0,7 ,1,7 ,2,7 ,7,8 ,8,9
                                     };

    auto const & hh = *hhp;
    doubletsFromHisto(layerPairs, 13, hh.iphi_d,hh.hist_d,hh.hitsLayerStart_d,phiCut);
  }



} // namespace end

#endif // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
