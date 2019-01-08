#ifndef RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#define RecoLocalTracker_SiPixelClusterizer_SiPixelFedCablingMapGPUWrapper_h
#include "CUDADataFormats/Common/interface/device_unique_ptr.h"
#include "CUDADataFormats/Common/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESManaged.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPU.h"

#include <cuda/api_wrappers.h>

#include <set>

class SiPixelFedCablingMap;
class TrackerGeometry;
class SiPixelQuality;

// TODO: since this has more information than just cabling map, maybe we should invent a better name?
class SiPixelFedCablingMapGPUWrapper {
public:
  SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMap const& cablingMap,
                                 TrackerGeometry const& trackerGeom,
                                 SiPixelQuality const *badPixelInfo);
  ~SiPixelFedCablingMapGPUWrapper();

  bool hasQuality() const { return hasQuality_; }

  // returns pointer to GPU memory
  const SiPixelFedCablingMapGPU *getGPUProductAsync(cuda::stream_t<>& cudaStream) const;

  // returns pointer to GPU memory
  const unsigned char *getModToUnpAllAsync(cuda::stream_t<>& cudaStream) const;
  edm::cuda::device::unique_ptr<unsigned char[]> getModToUnpRegionalAsync(std::set<unsigned int> const& modules, cuda::stream_t<>& cudaStream) const;

private:
  const SiPixelFedCablingMap *cablingMap_;
  CUDAESManaged helperUnp_;
  unsigned char *modToUnpDefault = nullptr;

  CUDAESManaged helper_;
  SiPixelFedCablingMapGPU *cablingGPU_ = nullptr;
  unsigned int size;
  bool hasQuality_;
};


#endif
