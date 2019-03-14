#include <map>

#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/supportedCUDADevices.h"

__global__
void isSupported(bool * result) {
  * result = true;
}

std::map<int, std::pair<int, int>> supportedCUDADevices(bool reset) {
  std::map<int, std::pair<int, int>> capabilities;

  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (cudaSuccess != status) {
    return capabilities;
  }

  for (int i = 0; i < devices; ++i) {
    cudaCheck(cudaSetDevice(i));
    bool supported = false;
    bool * supported_d;
    cudaCheck(cudaMalloc(&supported_d, sizeof(bool)));
    cudaCheck(cudaMemset(supported_d, 0x00, sizeof(bool)));
    isSupported<<<1,1>>>(supported_d);
    // swallow any eventual error from launching the kernel on an unsupported device
    cudaGetLastError();
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(& supported, supported_d, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(supported_d));
    if (supported) {
      cudaDeviceProp properties;
      cudaCheck(cudaGetDeviceProperties(&properties, i));
      capabilities[i] = std::make_pair(properties.major, properties.minor);
    }
    if (reset) {
      cudaCheck(cudaDeviceReset());
    }
  }

  return capabilities;
}
