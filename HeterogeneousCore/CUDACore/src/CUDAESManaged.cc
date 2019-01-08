#include "HeterogeneousCore/CUDACore/interface/CUDAESManaged.h"
#include "HeterogeneousCore/CUDAServices/interface/numberOfCUDADevices.h"

CUDAESManaged::CUDAESManaged(): prefetched_(numberOfCUDADevices()) {
  for(auto& pref: prefetched_) {
    pref.store(false);
  }
}

CUDAESManaged::~CUDAESManaged() {
  for(auto& ptrSize: buffers_) {
    cudaFree(ptrSize.first);
  }
}

void CUDAESManaged::advise() const {
  for(const auto& ptrSize: buffers_) {
    cudaCheck(cudaMemAdvise(ptrSize.first, ptrSize.second, cudaMemAdviseSetReadMostly, 0)); // device is ignored for this advise
  }
}

void CUDAESManaged::prefetchAsync(cuda::stream_t<>& stream) const {
  // The boolean atomic is an optimization attempt, it doesn't really
  // matter if more than one thread/edm stream issues the prefetches
  // as long as most of the prefetches are avoided.
  auto& pref = prefetched_[stream.device_id()];
  if(pref.load())
    return;

  for(const auto& ptrSize: buffers_) {
    cudaMemPrefetchAsync(ptrSize.first, ptrSize.second, stream.device_id(), stream.id());
  }

  pref.store(true);
}
