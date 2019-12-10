// system include files
#include <cmath>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "cudavectors.h"

namespace cudavectors {

  __host__ __device__ inline void convert(CylindricalVector const& cylindrical, CartesianVector & cartesian) {
    // fill here ...
  }

  __global__ void convertKernel(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    // fill here ...
  }

  void convertWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    // fill here ...
    //convertKernel<<<gridSize, blockSize>>>(cylindrical, cartesian, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace cudavectors
