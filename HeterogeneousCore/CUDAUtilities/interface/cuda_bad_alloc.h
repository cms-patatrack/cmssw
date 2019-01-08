#ifndef HeterogeneousCore_CUDAUtilities_interface_cuda_bad_alloc_h
#define HeterogeneousCore_CUDAUtilities_interface_cuda_bad_alloc_h

#include <memory>
#include <cuda_runtime.h>

class cuda_bad_alloc : public std::bad_alloc {
public:
  cuda_bad_alloc(cudaError_t error) noexcept :
    error_(error)
  { }

  const char* what() const noexcept override
  {
    return cudaGetErrorString(error_);
  }

private:
  cudaError_t error_;
};

#endif
