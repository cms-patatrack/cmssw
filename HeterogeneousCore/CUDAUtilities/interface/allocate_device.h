#ifndef HeterogeneousCore_CUDAUtilities_allocate_device_h
#define HeterogeneousCore_CUDAUtilities_allocate_device_h

#include <cuda_runtime.h>

namespace cms {
  namespace cuda {
    // Allocate device memory
    // This variant does not create device-side ownership
    void *allocate_device(int dev, size_t nbytes);

    // Allocate device memory
    // This variant creates device-side ownership. When freed, all work
    // in the stream up to the freeing point must be finished for the
    // memory block to be considered free (except for new allocation in
    // the same stream)
    void *allocate_device(int dev, size_t nbytes, cudaStream_t stream);

    // Free device memory (to be called from unique_ptr)
    void free_device(int device, void *ptr);
  }  // namespace cuda
}  // namespace cms

#endif
