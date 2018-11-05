#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuFishbone_h

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "DataFormats/Math/interface/approx_atan2.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"

#include "GPUCACell.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUVecArray.h"

namespace gpuPixelDoublets {

//  __device__
//  __forceinline__
  __global__
  void fishbone(
               GPUCACell::Hits const *  __restrict__ hhp,
               GPUCACell * cells, uint32_t const * __restrict__ nCells,
               GPUCACell::OuterHitOfCell const * __restrict__ isOuterHitOfCell,
               uint32_t nHits,
               uint32_t stride) {

    constexpr auto maxCellsPerHit = GPUCACell::maxCellsPerHit;


    auto const & hh = *hhp;
    uint8_t const * __restrict__ layerp =  hh.phase1TopologyLayer_d;
    auto layer = [&](uint16_t id) { return __ldg(layerp+id/phase1PixelTopology::maxModuleStride);};

    auto ldx = threadIdx.x + blockIdx.x * blockDim.x;
    auto idx = ldx/stride;
    auto first = ldx - idx*stride;
    assert(first<stride);

    if (idx>=nHits) return;
    auto const & vc = isOuterHitOfCell[idx];
    auto s = vc.size();
    if (s<2) return;
    // if alligned kill one of the two.
    auto const & c0 = cells[vc[0]];
    auto xo = c0.get_outer_x(hh);
    auto yo = c0.get_outer_y(hh);
    auto zo = c0.get_outer_z(hh);
    float x[maxCellsPerHit], y[maxCellsPerHit],z[maxCellsPerHit], n[maxCellsPerHit];
    uint16_t d[maxCellsPerHit]; uint8_t l[maxCellsPerHit];
    for (uint32_t ic=0; ic<s; ++ic) {
      auto & ci = cells[vc[ic]];
      d[ic] = ci.get_inner_detId(hh);
      l[ic] = layer(d[ic]);
      x[ic] = ci.get_inner_x(hh) -xo;
      y[ic] = ci.get_inner_y(hh) -yo;
      z[ic] = ci.get_inner_z(hh) -zo;
      n[ic] = x[ic]*x[ic]+y[ic]*y[ic]+z[ic]*z[ic];
    }

    // here we parallelize
    for (uint32_t ic=first; ic<s-1;  ic+=stride) {
      auto & ci = cells[vc[ic]];
      for    (auto jc=ic+1; jc<s; ++jc) {
        auto & cj = cells[vc[jc]];
        // must be different detectors in the same layer
        if (d[ic]==d[jc]) continue;
        // || l[ic]!=l[jc]) continue;
        auto cos12 = x[ic]*x[jc]+y[ic]*y[jc]+z[ic]*z[jc];
        if (cos12*cos12 >= 0.99999f*n[ic]*n[jc]) {
         // alligned:  kill closest
         if (n[ic]>n[jc]) {
           ci.theDoubletId=-1; 
           break;
         } else {
           cj.theDoubletId=-1;
         }
        }
      } //cj   
    } // ci
  }

}

#endif
