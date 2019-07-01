#ifndef HeterogeneousCore_CUDACore_src_getCachingHostAllocator
#define HeterogeneousCore_CUDACore_src_getCachingHostAllocator

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CachingHostAllocator.h"

#include "getCachingDeviceAllocator.h"

namespace cudautils {
  namespace allocator {
    inline
    notcub::CachingHostAllocator& getCachingHostAllocator() {
      static notcub::CachingHostAllocator allocator {
        binGrowth, minBin, maxBin, minCachedBytes(),
          false, // do not skip cleanup
          debug
          };
      return allocator;
    }
  }
}

#endif
