#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include <cuda_runtime.h>
#include<cstdio>

// these are in cudaCompat: here we do just a minimal test of "launch"
#ifndef __CUDACC__
#undef __global__
#define __global__ inline __attribute__((always_inline))
#endif

inline
constexpr void printEnv() {
  printf("ENV\n");
#ifdef CUDA_KERNELS_ON_CPU
  printf("CUDA_KERNELS_ON_CPU defined\n");
#endif
#ifdef __CUDACC__
  printf("__CUDACC__ defined\n");
#endif
#ifdef __CUDA_RUNTIME_H__
  printf("__CUDA_RUNTIME_H__ defined\n");
#endif
#ifdef __CUDA_ARCH__
  printf("__CUDA_ARCH__ defined\n");
#endif
  printf("---\n");
}

__global__
void hello(float k) {

  printf("hello from kernel %f\n",k);
  printEnv();
}


inline void wrapper() {

  printf("in Wrapper\n");
  printEnv();

  cudautils::launch(hello,{1, 1},3.14);
  cudaDeviceSynchronize();

}

