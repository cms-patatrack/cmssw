#include "HeterogeneousCore/CUDAUtilities/interface/prefixScan.h"
#include <cassert>

template<typename T>
__global__
void testPrefixScan(uint32_t size) {

  __shared__ T ws[32];
  __shared__ T c[1024];
  auto first = threadIdx.x;
  for (auto i=first; i<size; i+=blockDim.x) c[i]=1;

  blockPrefixScan(c, size, ws);

  assert(1==c[0]);
  for (auto i=first+1; i<size; i+=blockDim.x) {
    if (c[i]!=c[i-1]+1) printf("failed %d %d %d: %d %d\n",size, i, blockDim.x, c[i],c[i-1]);
    assert(c[i]==c[i-1]+1); assert(c[i]==i+1);
  }
}


int main() {
  
  for(int bs=32; bs<=1024; bs+=32)
  for (int j=1;j<=1024; ++j) {
   testPrefixScan<uint16_t><<<1,bs>>>(j);
   testPrefixScan<float><<<1,bs>>>(j);
  }
  cudaDeviceSynchronize();

  return 0;
}
