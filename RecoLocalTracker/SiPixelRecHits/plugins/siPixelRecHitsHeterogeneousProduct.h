#ifndef RecoLocalTracker_SiPixelRecHits_plugins_siPixelRecHitsHeterogeneousProduct_h
#define RecoLocalTracker_SiPixelRecHits_plugins_siPixelRecHitsHeterogeneousProduct_h

#include <cstdint>
#include <vector>


// #include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"


namespace siPixelRecHitsHeterogeneousProduct {

  struct HitsOnGPU{
     float * bs_d;
     uint32_t * hitsModuleStart_d;
     uint32_t * hitsLayerStart_d;
     int32_t  * charge_d;
     uint16_t * detInd_d;
     float *xg_d, *yg_d, *zg_d, *rg_d;
     float *xl_d, *yl_d;
     float *xerr_d, *yerr_d;
     int16_t * iphi_d;
     uint16_t * sortIndex_d;
     uint16_t * mr_d;

     // using Hist = HistoContainer<int16_t,7,8>;
     // Hist * hist_d;
  };

  struct GPUProduct {
     HitsOnGPU const * hits_d;
  };

}



#endif
