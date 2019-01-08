#ifndef HeterogeneousCore_CUDAUtilities_interface_CUDAManagedVector_h
#define HeterogeneousCore_CUDAUtilities_interface_CUDAManagedVector_h

#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/CUDAManagedAllocator.h"

template <typename T>
using CUDAManagedVector = std::vector<T, CUDAManagedAllocator<T>>;

#endif // HeterogeneousCore_CUDAUtilities_interface_CUDAManagedVector_h
