// system include files
#include <cmath>

// CUDA include files
#include <cuda_runtime.h>

// CMSSW include files
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "cudavectors.h"

namespace cudavectors {

  __host__ __device__ inline void convert(CylindricalVector const& cylindrical, CartesianVector & cartesian) {
    cartesian.x = cylindrical.rho * std::cos(cylindrical.phi);
    cartesian.y = cylindrical.rho * std::sin(cylindrical.phi);
    cartesian.z = cylindrical.rho * std::sinh(cylindrical.eta);
  }

  __global__ void convertKernel(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size) {
    auto firstElement = threadIdx.x + blockIdx.x * blockDim.x;
    auto gridSize = blockDim.x * gridDim.x;

    for (size_t i = firstElement; i < size; i += gridSize) {
      convert(cylindrical[i], cartesian[i]);
    }
  }

  void convertWrapper(CylindricalVector const* cylindrical, CartesianVector* cartesian, size_t size, cudaStream_t stream) {
    auto blockSize = 512;                                // somewhat arbitrary
    auto gridSize = (size + blockSize - 1) / blockSize;  // round up to cover the sample size
    convertKernel<<<gridSize, blockSize, 0, stream>>>(cylindrical, cartesian, size);
    cudaCheck(cudaGetLastError());
  }

}  // namespace cudavectors
