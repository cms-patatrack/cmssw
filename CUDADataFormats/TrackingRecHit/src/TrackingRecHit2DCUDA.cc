#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

template <>
cms::cuda::host::unique_ptr<float[]> TrackingRecHit2DGPU::localCoordToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<float[]>(5 * nHits(), stream);
  cms::cuda::copyAsync(ret, m_store32, 5 * nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint16_t[]> TrackingRecHit2DGPU::detIndexToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nHits(), stream);
  cms::cuda::copyAsync(ret, m_store16, nHits(), stream);
  return ret;
}

template <>
cms::cuda::host::unique_ptr<uint32_t[]> TrackingRecHit2DGPU::hitsModuleStartToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(2001, stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), m_hitsModuleStart, 4 * 2001, cudaMemcpyDefault, stream));
  return ret;
}

// the only specialization needed
template <>
void TrackingRecHit2DHost::copyFromGPU(TrackingRecHit2DGPU const* input, cudaStream_t stream) {
  assert(input);
  m_store32 = input->localCoordToHostAsync(stream);
}
