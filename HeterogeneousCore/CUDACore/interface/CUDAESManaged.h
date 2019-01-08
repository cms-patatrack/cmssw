#ifndef HeterogeneousCore_CUDACore_interface_CUDAESManaged_h
#define HeterogeneousCore_CUDACore_interface_CUDAESManaged_h

#include <atomic>
#include <vector>

#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

/**
 * Class to help with memory allocations for ESProducts. Each CUDA
 * ESProduct wrapper should
 * - include an instance of this class as their member
 * - use allocate() to allocate the memory buffers
 * - call advise() after filling the buffers in CPU
 * - call prefetch() before returning the CUDA ESProduct
 *
 * It owns the allocated memory and frees it in its destructor.
 */
class CUDAESManaged {
public:
  CUDAESManaged();
  ~CUDAESManaged();

  template <typename T>
  T *allocate(size_t elements, size_t elementSize=sizeof(T)) {
    T *ptr = nullptr;
    auto size = elementSize*elements;
    cudaCheck(cudaMallocManaged(&ptr, size));
    buffers_.emplace_back(ptr, size);
    return ptr;
  }

  template <typename T>
  void allocate(T **ptr, size_t elements, size_t elementSize=sizeof(T)) {
    *ptr = allocate<T>(elements, elementSize);
  }

  // Record a buffer allocated elsewhere to be used in advise/prefetch
  /*
  void record(void *ptr, size_t size) {
    buffers_.emplace_back(ptr, size);
  }
  */

  void advise() const;

  void prefetchAsync(cuda::stream_t<>& stream) const;

private:
  std::vector<std::pair<void *, size_t> > buffers_;
  mutable std::vector<std::atomic<bool>> prefetched_;
};

#endif
