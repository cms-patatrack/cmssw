#ifndef RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h
#define RecoLocalTracker_SiPixelRecHits_plugins_PixelRecHits_h

#include "EventFilter/SiPixelRawToDigi/plugins/siPixelRawToClusterHeterogeneousProduct.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"

#include <cuda/api_wrappers.h>

#include <cstdint>
#include <vector>

#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h" 


namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

namespace pixelgpudetails {
  using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;

  struct HitsOnCPU {
    explicit HitsOnCPU(uint32_t nhits) :
      charge(nhits),xl(nhits),yl(nhits),xe(nhits),ye(nhits), mr(nhits), mc(nhits){}
    uint32_t hitsModuleStart[2001];
    std::vector<int32_t> charge;
    std::vector<float> xl, yl;
    std::vector<float> xe, ye;
    std::vector<uint16_t> mr;
    std::vector<uint16_t> mc;

    HitsOnGPU const * gpu_d=nullptr;  // does not belong here (or actually does it?) ...
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
