#ifndef HeterogeneousCore_CUDAServices_interface_supportedCudaDevices_h
#define HeterogeneousCore_CUDAServices_interface_supportedCudaDevices_h

#include <map>

std::map<int, std::pair<int, int>> supportedCudaDevices(bool reset = true);

#endif // HeterogeneousCore_CUDAServices_interface_supportedCudaDevices_h
