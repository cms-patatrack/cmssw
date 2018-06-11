#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include "EventFilter/SiPixelRawToDigi/plugins/siPixelRawToClusterHeterogeneousProduct.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"

#include <cuda/api_wrappers.h>

#include <cstdint>
#include <vector>


// #include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"


namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

namespace pixelgpudetails {
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

  struct HitsOnCPU {
    explicit HitsOnCPU(uint32_t nhits) :
      charge(nhits),xl(nhits),yl(nhits),xe(nhits),ye(nhits), mr(nhits){}
    uint32_t hitsModuleStart[2001];
    std::vector<int32_t> charge;
    std::vector<float> xl, yl;
    std::vector<float> xe, ye;
    std::vector<uint16_t> mr;
  };

  class PixelRecHitGPUKernel {
  public:
    PixelRecHitGPUKernel();
    ~PixelRecHitGPUKernel();

    PixelRecHitGPUKernel(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel(PixelRecHitGPUKernel&&) = delete;
    PixelRecHitGPUKernel& operator=(const PixelRecHitGPUKernel&) = delete;
    PixelRecHitGPUKernel& operator=(PixelRecHitGPUKernel&&) = delete;

    void makeHitsAsync(const siPixelRawToClusterHeterogeneousProduct::GPUProduct& input,
                       float const * bs,
                       pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                       cuda::stream_t<>& stream);

    HitsOnCPU getOutput(cuda::stream_t<>& stream) const;

  private:
    HitsOnGPU * gpu_d;  // copy of the structure on the gpu itself: this is the "Product" 
    HitsOnGPU gpu_;
    uint32_t hitsModuleStart_[gpuClustering::MaxNumModules+1];
  };
}

#endif // RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
