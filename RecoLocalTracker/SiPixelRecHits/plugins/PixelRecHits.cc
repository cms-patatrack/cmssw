// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClusteringConstants.h"

#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace pixelgpudetails {

  TrackingRecHit2DCUDA PixelRecHitGPUKernel::makeHitsAsync(SiPixelDigisCUDA const& digis_d,
                                                           SiPixelClustersCUDA const& clusters_d,
                                                           BeamSpotCUDA const& bs_d,
                                                           pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                                           cudaStream_t stream) const {
    auto nHits = clusters_d.nClusters();
    TrackingRecHit2DCUDA hits_d(nHits, cpeParams, clusters_d.clusModuleStart(), stream);

    int threadsPerBlock = 128;
    int blocks = digis_d.nModules();  // active modules (with digis)

#ifdef GPU_DEBUG
    std::cout << "launching getHits kernel for " << blocks << " blocks" << std::endl;
#endif
    if (blocks) {  // protect from empty events
      cudautils::launch(gpuPixelRecHits::getHits, { blocks, threadsPerBlock, 0, stream },
          cpeParams, bs_d.data(), digis_d.view(), digis_d.nDigis(), clusters_d.view(), hits_d.view());
    }
#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif

    // assuming full warp of threads is better than a smaller number...
    if (nHits) {
      cudautils::launch(gpuPixelRecHits::setHitsLayerStart, {1, 256, 0, stream}, clusters_d.clusModuleStart(), cpeParams, hits_d.hitsLayerStart());

      if (nHits >= TrackingRecHit2DSOAView::maxHits()) {
        edm::LogWarning("PixelRecHitGPUKernel")
            << "Hits Overflow " << nHits << " > " << TrackingRecHit2DSOAView::maxHits();
      }

      auto hws = cudautils::make_device_unique<uint8_t[]>(TrackingRecHit2DSOAView::Hist::wsSize(), stream);
      cudautils::fillManyFromVector(
          hits_d.phiBinner(), hws.get(), 10, hits_d.iphi(), hits_d.hitsLayerStart(), nHits, 256, stream);
      cudaCheck(cudaGetLastError());
    }

#ifdef GPU_DEBUG
    cudaCheck(cudaDeviceSynchronize());
#endif

    return hits_d;
  }

}  // namespace pixelgpudetails
