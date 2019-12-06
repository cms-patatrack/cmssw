#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireCUDADevices.h"



#include <cuda_runtime.h>
#include<cstdio>

#undef __global__
#define __global__ inline __attribute__((always_inline))


__global__
void hello(float k) {

  printf("hello %f\n",k);

}




int main() {

  requireCUDADevices();

  cudautils::launch(hello,{1, 1},3.14);
  cudaDeviceSynchronize();

  return 0;
}
