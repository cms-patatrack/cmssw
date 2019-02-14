#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <array>

#include <cuda/api_wrappers.h>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/exitSansCUDADevices.h"

constexpr uint32_t MaxElem=64000;
constexpr uint32_t MaxTk=8000;
constexpr uint32_t MaxAssocs = 4*MaxTk;
using Assoc = OneToManyAssoc<uint16_t,MaxElem,MaxAssocs>;

using Multiplicity = OneToManyAssoc<uint16_t,8,MaxTk>;

using TK = std::array<uint16_t,4>;

__global__
void countMultiLocal(TK const * __restrict__ tk, Multiplicity * __restrict__ assoc, uint32_t n) {
   int first = blockDim.x * blockIdx.x + threadIdx.x;
   for (int i = first; i < n; i += gridDim.x*blockDim.x) {

     __shared__ Multiplicity::CountersOnly local;
     if (threadIdx.x==0) local.zero();
     __syncthreads();
     local.countDirect(2+i%4);
     __syncthreads();
     if (threadIdx.x==0) assoc->add(local);
  }
}

__global__
void countMulti(TK const * __restrict__ tk, Multiplicity * __restrict__ assoc, uint32_t n) {
   int first = blockDim.x * blockIdx.x + threadIdx.x;
   for (int i = first; i < n; i += gridDim.x*blockDim.x)
     assoc->countDirect(2+i%4);
}


__global__
void verifyMulti(Multiplicity * __restrict__ m1, Multiplicity * __restrict__ m2) {
   int first = blockDim.x * blockIdx.x + threadIdx.x;
   for (int i = first; i < Multiplicity::totbins(); i += gridDim.x*blockDim.x)
     assert(m1->off[i]==m2->off[i]);
}

__global__ 
void count(TK const * __restrict__ tk, Assoc * __restrict__ assoc, uint32_t n) {
   int first = blockDim.x * blockIdx.x + threadIdx.x;
   for (int i = first; i < 4*n; i += gridDim.x*blockDim.x) {
     auto k = i/4;
     auto j = i - 4*k;
     assert(j<4);
     if (k>=n) return;
     if (tk[k][j]<MaxElem)
     assoc->countDirect(tk[k][j]);
   }
}

__global__
void fill(TK const * __restrict__ tk, Assoc * __restrict__ assoc, uint32_t n) {
   int first = blockDim.x * blockIdx.x + threadIdx.x;
   for (int i = first; i < 4*n; i += gridDim.x*blockDim.x) {
     auto    k = i/4;
     auto    j = i - 4*k;
     assert(j<4);
     if (k>=n) return;
     if (tk[k][j]<MaxElem)
     assoc->fillDirect(tk[k][j],k);
   }
}

__global__
void verify(Assoc * __restrict__ assoc) {
   assert(assoc->size()<Assoc::capacity());
}

__global__
void fillBulk(AtomicPairCounter * apc, TK const * __restrict__ tk, Assoc * __restrict__ assoc, uint32_t n) {
   int first = blockDim.x * blockIdx.x + threadIdx.x;
   for (int k = first; k < n; k += gridDim.x*blockDim.x) {
     auto m = tk[k][3]<MaxElem ? 4 : 3;
     assoc->bulkFill(*apc,&tk[k][0],m);
   }
}

int main() {
  exitSansCUDADevices();

  std::cout << "OneToManyAssoc " << Assoc::nbins() << ' ' << Assoc::capacity() << ' '<< Assoc::wsSize() << std::endl;

  auto current_device = cuda::device::current::get();

  std::mt19937 eng;

  std::geometric_distribution<int> rdm(0.8);

  constexpr uint32_t N = 4000;

  std::vector<std::array<uint16_t,4>> tr(N);

  // fill with "index" to element
  long long ave=0;
  int imax=0;
  auto n=0U;
  auto z=0U;
  auto nz=0U;
  for (auto i=0U; i<4U; ++i) {
    auto j=0U;
    while(j<N && n<MaxElem) {
      if (z==11) { ++n; z=0; ++nz; continue;} // a bit of not assoc 
      auto x = rdm(eng);
      auto k = std::min(j+x+1,N);
      if (i==3 && z==3) { // some triplets time to time
        for (;j<k; ++j) tr[j][i] = MaxElem+1;
      } else {
        ave+=x+1;
        imax = std::max(imax,x);
        for (;j<k; ++j) tr[j][i] = n;
        ++n; 
      }
      ++z;
    }
    assert(n<=MaxElem);
    assert(j<=N);
  }
  std::cout << "filled with "<< n << " elements " << double(ave)/n <<' '<< imax << ' ' << nz << std::endl;

  auto v_d = cuda::memory::device::make_unique<std::array<uint16_t,4>[]>(current_device, N);
  assert(v_d.get());
  auto a_d = cuda::memory::device::make_unique<Assoc[]>(current_device,1);
  auto ws_d = cuda::memory::device::make_unique<uint8_t[]>(current_device, Assoc::wsSize());

  cuda::memory::copy(v_d.get(), tr.data(), N*sizeof(std::array<uint16_t,4>));

  cudautils::launchZero(a_d.get(),0);

  auto nThreads = 256;
  auto nBlocks = (4*N + nThreads - 1) / nThreads;

  count<<<nBlocks,nThreads>>>(v_d.get(),a_d.get(),N);
  
  cudautils::launchFinalize(a_d.get(),ws_d.get(),0);
  verify<<<1,1>>>(a_d.get());
  fill<<<nBlocks,nThreads>>>(v_d.get(),a_d.get(),N);

  Assoc la;
  cuda::memory::copy(&la,a_d.get(),sizeof(Assoc));
  std::cout << la.size() << std::endl;
  imax = 0;
  ave=0;
  z=0;
  for (auto i=0U; i<n; ++i) {
    auto x = la.size(i);
    if (x==0) { z++; continue;}
    ave+=x;
    imax = std::max(imax,int(x));
  }
  assert(0==la.size(n));
  std::cout << "found with "<< n << " elements " << double(ave)/n <<' '<< imax << ' ' << z << std::endl;

  // now the inverse map (actually this is the direct....)
  AtomicPairCounter * dc_d;
  cudaMalloc(&dc_d, sizeof(AtomicPairCounter));
  cudaMemset(dc_d, 0, sizeof(AtomicPairCounter));
  nBlocks = (N + nThreads - 1) / nThreads;
  fillBulk<<<nBlocks,nThreads>>>(dc_d,v_d.get(),a_d.get(),N);
  cudautils::finalizeBulk<<<nBlocks,nThreads>>>(dc_d,a_d.get());

  AtomicPairCounter dc;
  cudaMemcpy(&dc, dc_d, sizeof(AtomicPairCounter), cudaMemcpyDeviceToHost);

  std::cout << "final counter value " << dc.get().n << ' ' << dc.get().m << std::endl;

  cuda::memory::copy(&la,a_d.get(),sizeof(Assoc));
  std::cout << la.size() << std::endl;
  imax = 0;
  ave=0;
  for (auto i=0U; i<N; ++i) {
    auto x = la.size(i);
    if (!(x==4 || x==3)) std::cout << i << ' ' << x << std::endl;
    assert(x==4 || x==3);
    ave+=x;
    imax = std::max(imax,int(x));
  }
  assert(0==la.size(N));
  std::cout << "found with ave occupancy " << double(ave)/N <<' '<< imax << std::endl;


  // here verify use of block local counters
  auto m1_d = cuda::memory::device::make_unique<Multiplicity[]>(current_device,1);
  auto m2_d = cuda::memory::device::make_unique<Multiplicity[]>(current_device,1);
  cudautils::launchZero(m1_d.get(),0);
  cudautils::launchZero(m2_d.get(),0);

  nBlocks = (4*N + nThreads - 1) / nThreads;
  countMulti<<<nBlocks,nThreads>>>(v_d.get(),m1_d.get(),N);
  countMultiLocal<<<nBlocks,nThreads>>>(v_d.get(),m2_d.get(),N);
  verifyMulti<<<1,Multiplicity::totbins()>>>(m1_d.get(),m2_d.get());

  cudautils::launchFinalize(m1_d.get(),ws_d.get(),0);
  cudautils::launchFinalize(m2_d.get(),ws_d.get(),0);
  verifyMulti<<<1,Multiplicity::totbins()>>>(m1_d.get(),m2_d.get());


  cudaCheck(cudaGetLastError());
  cudaCheck(cudaDeviceSynchronize());
  return 0;
}
