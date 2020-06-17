#include "CUDADataFormats/Common/interface/ProductBase.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContextBase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/StreamCache.h"

#include "chooseDevice.h"

namespace cms::cuda {
  ScopedContextBase::ScopedContextBase(edm::StreamID streamID) : currentDevice_(chooseDevice(streamID)) {
    cudaCheck(cudaSetDevice(currentDevice_));
    stream_ = getStreamCache().get();
  }

  ScopedContextBase::ScopedContextBase(const ProductBase& data) : currentDevice_(data.device()) {
    cudaCheck(cudaSetDevice(currentDevice_));
    if (data.mayReuseStream()) {
      stream_ = data.streamPtr();
    } else {
      stream_ = getStreamCache().get();
    }
  }

  ScopedContextBase::ScopedContextBase(int device, SharedStreamPtr stream)
      : currentDevice_(device), stream_(std::move(stream)) {
    cudaCheck(cudaSetDevice(currentDevice_));
  }
}  // namespace cms::cuda
