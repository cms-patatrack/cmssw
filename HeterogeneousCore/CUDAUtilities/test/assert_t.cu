#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireCUDADevices.h"

__global__ void testIt(int one) { assert(one == 1); }

int main(int argc, char* argv[]) {
  requireCUDADevices();

  testIt<<<1, 1>>>(argc);
  cudaDeviceSynchronize();

  return (argc == 1) ? EXIT_SUCCESS : EXIT_FAILURE;
}
