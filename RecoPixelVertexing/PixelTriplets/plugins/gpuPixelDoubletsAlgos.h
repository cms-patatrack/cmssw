#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoubletsAlgos_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoubletsAlgos_h

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "CAHitNtupletGeneratorKernels.h"

#include "CAConstants.h"
#include "GPUCACell.h"

//#define GPU_DEBUG 1

namespace gpuPixelDoublets {

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __device__ __forceinline__ void doubletsFromHisto(uint8_t const* __restrict__ layerPairs,
                                                    uint32_t nPairs,
                                                    GPUCACell* cells,
                                                    uint32_t* nCells,
                                                    CellNeighborsVector* cellNeighbors,
                                                    CellTracksVector* cellTracks,
                                                    TrackingRecHit2DSOAView const& __restrict__ hh,
                                                    GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                                    int16_t const* __restrict__ phicuts,
                                                    float const* __restrict__ minz,
                                                    float const* __restrict__ maxz,
                                                    float const* __restrict__ maxr,
                                                    bool ideal_cond,
                                                    bool doClusterCut,
                                                    bool doZ0Cut,
                                                    bool doPtCut,
                                                    uint32_t maxNumOfDoublets, 
						    bool upgrade
                                                    ) {
    // ysize cuts (z in the barrel)  times 8
    // these are used if doClusterCut is true
    int minYsizeB1 = upgrade ? 36 : 3;
    int minYsizeB2 = upgrade ? 28 : 3;
    int maxDYsize12 = upgrade ? 28 : 1;
    int maxDYsize = upgrade ? 20 : 0;
    int maxDYPred = 20;
    float dzdrFact = 8 * 0.0285 / 0.015;  // from dz/dr to "DY"
  
    bool isOuterLadder = ideal_cond;

    using Hist = TrackingRecHit2DSOAView::Hist;

    auto const& __restrict__ hist = hh.phiBinner();
    uint32_t const* __restrict__ offsets = hh.hitsLayerStart();
    assert(offsets);
    auto layerSize = [=](uint8_t li) { return offsets[li + 1] - offsets[li]; };   

    // nPairsMax to be optimized later (originally was 64).
    // If it should be much bigger, consider using a block-wide parallel prefix scan,
    // e.g. see  https://nvlabs.github.io/cub/classcub_1_1_warp_scan.html
    const int nPairsMax = CAConstants::maxNumberOfLayerPairs();
    assert(nPairs <= nPairsMax);
    __shared__ uint32_t innerLayerCumulativeSize[nPairsMax];
    __shared__ uint32_t ntot;
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      innerLayerCumulativeSize[0] = layerSize(layerPairs[0]);
      for (uint32_t i = 1; i < nPairs; ++i) {
        innerLayerCumulativeSize[i] = innerLayerCumulativeSize[i - 1] + layerSize(layerPairs[2 * i]);
        }
      ntot = innerLayerCumulativeSize[nPairs - 1];
    }
    __syncthreads();

    // x runs faster
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    auto first = threadIdx.x;
    auto stride = blockDim.x;

    uint32_t pairLayerId = 0;  // cannot go backward
                                    
    #ifdef GPU_DEBUG
    int nDoublets[50],nStart[50],nZ[50],nPhi[50],nz0[50],nPt[50], nReg[50];

    for (size_t i = 0; i < 50; i++) {
      nDoublets[i]=0;
      nStart[i] = 0;
      nZ[i] = 0;
      nPhi[i] = 0;
      nz0[i] = 0;
      nPt[i] = 0;
      nReg[i] = 0;
    }

    #endif
    for (auto j = idy; j < ntot; j += blockDim.y * gridDim.y) {


      while (j >= innerLayerCumulativeSize[pairLayerId++])
        ;
      --pairLayerId;  // move to lower_bound ??


      assert(pairLayerId < nPairs);
      assert(j < innerLayerCumulativeSize[pairLayerId]);
      assert(0 == pairLayerId || j >= innerLayerCumulativeSize[pairLayerId - 1]);

      uint8_t inner = layerPairs[2 * pairLayerId];
      uint8_t outer = layerPairs[2 * pairLayerId + 1];

      // printf("%d %d %d %d %d \n",j,innerLayerCumulativeSize[pairLayerId],pairLayerId,outer,inner);
      assert(outer > inner);

      auto hoff = Hist::histOff(outer);

      auto i = (0 == pairLayerId) ? j : j - innerLayerCumulativeSize[pairLayerId - 1];
      i += offsets[inner];

      assert(i >= offsets[inner]);
      assert(i < offsets[inner + 1]);

      // found hit corresponding to our cuda thread, now do the job
      auto mi = hh.detectorIndex(i);

      auto maxModules = upgrade ? 4000: 2000;
      if (mi > maxModules)
        continue;  // invalid

      /* maybe clever, not effective when zoCut is on
      auto bpos = (mi%8)/4;  // if barrel is 1 for z>0
      auto fpos = (outer>3) & (outer<7);
      if ( ((inner<3) & (outer>3)) && bpos!=fpos) continue;
      */
      #ifdef GPU_DEBUG
      nStart[pairLayerId]++;
      #endif
      auto mez = hh.zGlobal(i);
//printf("not here\n");
    
      if (mez < minz[pairLayerId] || mez > maxz[pairLayerId])
      continue;
     
      #ifdef GPU_DEBUG
      nZ[pairLayerId]++;
      #endif
      int16_t mes = -1;  // make compiler happy
      if (doClusterCut) {
        // if ideal treat inner ladder as outer
        if (inner == 0)
          assert(mi < 108);
        isOuterLadder = ideal_cond ? true : 0 == (mi / 8) % 2;  // only for B1/B2/B3 B4 is opposite, FPIX:noclue...

        // in any case we always test mes>0 ...
        mes = inner > 0 || isOuterLadder ? hh.clusterSizeY(i) : -1;

        if (inner == 0 && outer > 3)  // B1 and F1
          if (mes > 0 && mes < minYsizeB1)
            continue;                 // only long cluster  (5*8)
        if (inner == 1 && outer > 3)  // B2 and F1
          if (mes > 0 && mes < minYsizeB2)
            continue;
      }
      auto mep = hh.iphi(i);
      auto mer = hh.rGlobal(i);

      // all cuts: true if fails
      float z0cut = 12.f;//maxz0[pairLayerId];//upgrade ? 12.f : 12.f;      // cm

      constexpr float hardPtCut = 0.5f;  // GeV
      constexpr float minRadius =
          hardPtCut * 87.78f;  // cm (1 GeV track has 1 GeV/c / (e * 3.8T) ~ 87 cm radius in a 3.8T field)
      constexpr float minRadius2T4 = 4.f * minRadius * minRadius ;
       auto ptcut = [&](int j, int16_t idphi) {
     //auto ptcut = [&](int j, int32_t idphi) {
        auto r2t4 = minRadius2T4;
        auto ri = mer;
        auto ro = hh.rGlobal(j);
        auto dphi = short2phi(idphi);
        return dphi * dphi * (r2t4 - ri * ro) > (ro - ri) * (ro - ri);
      };
      auto z0cutoff = [&](int j) {
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);
        auto dr = ro - mer;
        auto ll = std::abs((mez * ro - mer * zo));
        // printf("%f\n",z0Cut); 
        return   dr > 100*maxr[pairLayerId] || dr < 0 || ll > z0cut * dr;
      };

      auto zsizeCut = [&](int j) {
        auto onlyBarrel = outer < 4;
        auto so = hh.clusterSizeY(j);
        auto dy = inner == 0 ? maxDYsize12 : maxDYsize;
        // in the barrel cut on difference in size
        // in the endcap on the prediction on the first layer (actually in the barrel only: happen to be safe for endcap as well)
        // FIXME move pred cut to z0cutoff to optmize loading of and computaiton ...
        auto zo = hh.zGlobal(j);
        auto ro = hh.rGlobal(j);

	if(!upgrade)
        {
                return onlyBarrel ? mes > 0 && so > 0 && std::abs(so - mes) > dy
                          : (inner < 4) && mes > 0 &&
                                std::abs(mes - int(std::abs((mez - zo) / (mer - ro)) * dzdrFact + 0.5f)) > maxDYPred;
        }else
        {
                float f = std::abs((mer - ro) / (mez - zo) * (so-mes));
                return onlyBarrel ? mes > 0 && so > 0 && std::abs(so - mes) > dy : (inner < 4) && mes > 0 && (f <= 1.75) && (f>=0.25);
        }

      };

      auto iphicut = phicuts[pairLayerId];

      auto kl = Hist::bin(int16_t(mep - iphicut));
      auto kh = Hist::bin(int16_t(mep + iphicut));

      auto incr = [](auto& k) { return k = (k + 1) % Hist::nbins(); };
      // bool piWrap = std::abs(kh-kl) > Hist::nbins()/2;

#ifdef GPU_DEBUG
      int tot = 0;
      int nmin = 0;
      int tooMany = 0;
#endif

      auto khh = kh;

      // if(khh<Hist::nbins()-1)
      //   incr(khh);

      for (auto kk = kl; kk != khh; incr(kk)) {
#ifdef GPU_DEBUG
        if (kk != kl && kk != kh)
          nmin += hist.size(kk + hoff);
#endif
        auto const* __restrict__ p = hist.begin(kk + hoff);
        auto const* __restrict__ e = hist.end(kk + hoff);

        p += first;

        for (; p < e; p += stride) {

          auto oi = __ldg(p);
          assert(oi >= offsets[outer]);

 // assert(oi < offsets[outer + 1]);
          if(oi>offsets[outer + 1]) continue;
          auto mo = hh.detectorIndex(oi);

          if (mo > maxModules)
            continue;  //    invalid

          bool z0Cut = false, phicut = false, ptCut = false;
          if (doZ0Cut && z0cutoff(oi))
          #ifdef GPU_DEBUG
          z0Cut = true;
          #else
            continue;
          #endif

          #ifdef GPU_DEBUG
          nz0[pairLayerId]++;
          #endif

          bool badRegion = false;
          
          if(badRegion)
          #ifdef GPU_DEBUG
          {
           //regionCut = true;
          nReg[pairLayerId]++;
          }
          #else
          continue;
          #endif

          // printf("one\n");
          auto mop = hh.iphi(oi);
          // if(mep > halfpi_p )
          uint16_t idphi = std::min(std::abs(int16_t(mop - mep)), std::abs(int16_t(mep - mop)));

          if (idphi > iphicut)
          #ifdef GPU_DEBUG
          phicut = true;
          #else
            continue;
          #endif

          // printf("two\n");
          if (doClusterCut && zsizeCut(oi))
            continue;
          if (doPtCut && ptcut(oi, idphi)) //ptCut = true;
          #ifdef GPU_DEBUG
          nPhi[pairLayerId]++;
           ptCut = true;
          #else
            continue;
          #endif


          #ifdef GPU_DEBUG
          nPt[pairLayerId]++;
  	  
          printf("Doublet %d %d - inn %.2f %.2f %.2f - out %.2f %.2f %.2f - %d %d - %d %d %d %d \n",
          inner,outer,
          hh.xGlobal(i),hh.yGlobal(i),hh.zGlobal(i),
          hh.xGlobal(oi),hh.yGlobal(oi),hh.zGlobal(oi),
          iphicut,idphi,
          z0Cut,phicut,ptCut, (z0Cut && phicut && ptCut));

          /*if(z0Cut || phicut || ptCut)
          {
            continue;
          };
*/
          #endif
          // printf("three\n");
          auto ind = atomicAdd(nCells, 1);
          #ifdef GPU_DEBUG
          nDoublets[pairLayerId]++;
          #endif
  
	if (ind >= maxNumOfDoublets) {
  
            atomicSub(nCells, 1);
            break;
          }  // move to SimpleVector??
          // int layerPairId, int doubletId, int innerHitId, int outerHitId)
          cells[ind].init(*cellNeighbors, *cellTracks, hh, pairLayerId, ind, i, oi);
          isOuterHitOfCell[oi].push_back(ind);
          //printf("%d - %d - %d - %d - %d \n",i,oi,outer,ind,int(isOuterHitOfCell[oi].size()));
#ifdef GPU_DEBUG
          if (isOuterHitOfCell[oi].full())
            ++tooMany;
          ++tot;
#endif
        }

      }
      // printf("OuterHitOfCell full for %d in layer %d/%d, %d,%d %d\n", i, inner, outer, nmin, tot, tooMany);
#ifdef GPU_DEBUG
      if (tooMany > 0)
            printf("OuterHitOfCell full for %d in layer %d/%d, %d,%d %d\n", i, inner, outer, nmin, tot, tooMany);
#endif

    }  // loop in block...
    
    #ifdef GPU_DEBUG
         printf("i in ou nD nS nZ nZ0 nPhi nPt nReg\n") ;
         for (int i = 0; i < 50; i++) {
            if(i>int(nPairs))
            continue;
            printf("pair %d %d %d %d %d %d %d %d %d %d\n", i, layerPairs[2 * i], layerPairs[2 * i + 1], nDoublets[i], nStart[i],nZ[i],nz0[i],nPhi[i],nPt[i],nReg[i]);
          }
    printf("gpuDoublets %d \n",*nCells);
    #endif
  }

}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoupletsAlgos_h
