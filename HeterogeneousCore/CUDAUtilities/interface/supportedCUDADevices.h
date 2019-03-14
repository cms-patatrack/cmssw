#ifndef HeterogeneousCore_CUDAUtilities_interface_supportedCUDADevices_h
#define HeterogeneousCore_CUDAUtilities_interface_supportedCUDADevices_h

#include <map>

std::map<int, std::pair<int, int>> supportedCUDADevices(bool reset = true);

#endif // HeterogeneousCore_CUDAUtilities_interface_supportedCUDADevices_h
