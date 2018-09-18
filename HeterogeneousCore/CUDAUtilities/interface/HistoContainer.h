#ifndef HeterogeneousCore_CUDAUtilities_HistoContainer_h
#define HeterogeneousCore_CUDAUtilities_HistoContainer_h

#include <cassert>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#ifndef __CUDA_ARCH__
#include <atomic>
#endif // __CUDA_ARCH__

#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


#ifdef __CUDACC__
namespace cudautils {

  template<typename Histo>
  __global__
  void zeroMany(Histo * h, uint32_t nh) {
    auto i  = blockIdx.x * blockDim.x + threadIdx.x;
    auto ih = i / Histo::nbins();
    auto k  = i - ih * Histo::nbins();
    if (ih < nh) {
      h[ih].nspills = 0;
      if (k < Histo::nbins())
        h[ih].n[k] = 0;
    }
  }

  template<typename Histo, typename T>
  __global__
  void fillFromVector(Histo * h,  uint32_t nh, T const * __restrict__ v, uint32_t * offsets) {
     auto i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i >= offsets[nh]) return;
     auto off = cuda_std::upper_bound(offsets, offsets + nh + 1, i);
     assert((*off) > 0);
     int32_t ih = off - offsets - 1;
     assert(ih >= 0);
     assert(ih < nh); 
     h[ih].fill(v[i], i);
  }

  template<typename Histo, typename T>
  __global__
  void fillFromVector(Histo * h, T const * __restrict__ v, uint32_t size) {
     auto i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i < size) h->fill(v[i], i);
  }

  template<typename Histo>
  void zero(Histo * h, uint32_t nh, int nthreads, cudaStream_t stream) {
    auto nblocks = (nh * Histo::nbins() + nthreads - 1) / nthreads;
    zeroMany<<<nblocks, nthreads, 0, stream>>>(h, nh);
    cudaCheck(cudaGetLastError());
  }

  template<typename Histo, typename T>
  void fillOneFromVector(Histo * h, T const * __restrict__ v, uint32_t size, int nthreads, cudaStream_t stream) {
    zero(h, 1, nthreads, stream);
    auto nblocks = (size + nthreads - 1) / nthreads;
    fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, v, size);
    cudaCheck(cudaGetLastError());
  }

  template<typename Histo, typename T>
  void fillManyFromVector(Histo * h, uint32_t nh, T const * __restrict__ v, uint32_t * offsets, uint32_t totSize, int nthreads, cudaStream_t stream) {
    zero(h, nh, nthreads, stream);
    auto nblocks = (totSize + nthreads - 1) / nthreads;
    fillFromVector<<<nblocks, nthreads, 0, stream>>>(h, nh, v, offsets);
    cudaCheck(cudaGetLastError());
  }

} // namespace cudautils
#endif


// iteratate over N bins left and right of the one containing "v"
// including spillBin
template<typename Hist, typename V, typename Func>
__host__ __device__
__forceinline__
void forEachInBins(Hist const & hist, V value, int n, Func func) {
   int bs = hist.bin(value);
   int be = std::min(int(hist.nbins()),bs+n+1);
   bs = std::max(0,bs-n);
   // assert(be>bs);
//   bool tbc=false;
   for (auto b=bs; b<be; ++b){
//   tbc |= hist.full(b);
   for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
      func(*pj);
   }}
//   if (tbc)
   for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
     func(*pj);
}

// iteratate over bins containing all values in window wmin, wmax
// including spillBin
template<typename Hist, typename V, typename Func>
__host__ __device__
__forceinline__
void forEachInWindow(Hist const & hist, V wmin, V wmax, Func const & func) {
   auto bs = hist.bin(wmin);
   auto be = hist.bin(wmax);
   // be = std::min(int(hist.nbins()),be+1);
   // bs = std::max(0,bs);
   // assert(be>=bs);
//   bool tbc=false;
   for (auto b=bs; b<=be; ++b){
//   tbc |= hist.full(b);
   for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
      func(*pj);
   }}
//   if (tbc)
   for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
     func(*pj);
}


template<
  typename T, // the type of the discretized input values
  uint32_t NBINS, // number of bins 
  uint32_t M, // max number of element a bin can contain (in bits)
  uint32_t S=sizeof(T) * 8, // number of significant bits in T
  typename I=uint32_t  // type stored in the container (usually an index in a vector of the input values)
>
class HistoContainer {
public:
#ifdef __CUDACC__
  using Counter = uint32_t;
#else
  using Counter = std::atomic<uint32_t>;
#endif

  using index_type = I;
  using UT = typename std::make_unsigned<T>::type;

  static constexpr uint32_t ilog2(uint32_t v) {

    constexpr uint32_t b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    constexpr uint32_t s[] = {1, 2, 4, 8, 16};

    uint32_t r = 0; // result of log2(v) will go here
    for (auto i = 4; i >= 0; i--) if (v & b[i]) {
      v >>= s[i];
      r |= s[i];
    }
    return r;
  }


  static constexpr uint32_t sizeT()     { return S; }
  static constexpr uint32_t nbins()     { return NBINS;}
  static constexpr uint32_t nbits()     { return ilog2(NBINS-1)+1;}
  static constexpr uint32_t binSize()   { return 1 << M; }
  static constexpr uint32_t spillSize() { return 16 * binSize(); }

  static constexpr UT bin(T t) {
    constexpr uint32_t shift = sizeT() - nbits();
    constexpr uint32_t mask = (1<<nbits()) - 1;
    return (t >> shift) & mask;
  }

  void zero() {
    nspills = 0;
    for (auto & i : n)
      i = 0;
  }

  static __host__ __device__
  __forceinline__
  uint32_t atomicIncrement(Counter & x) {
    #ifdef __CUDA_ARCH__
    return atomicAdd(&x, 1);
    #else
    return x++;
    #endif
  }

  __host__ __device__
  void fill(T t, index_type j) {
    UT b = bin(t);
    assert(b<nbins());
    auto w = atomicIncrement(n[b]);
    if (w < binSize()) {
      bins[b * binSize() + w] = j;
    } else {
      auto w = atomicIncrement(nspills);
      if (w < spillSize())
        spillBin[w] = j;
    }
  }

  constexpr bool fullSpill() const {
    return nspills > spillSize();
  }

  constexpr bool full(uint32_t b) const {
    return n[b] > binSize();
  }

  constexpr auto const * begin(uint32_t b) const {
     return bins + b * binSize();
  }

  constexpr auto const * end(uint32_t b) const {
     return begin(b) + std::min(binSize(), uint32_t(n[b]));
  }

  constexpr auto size(uint32_t b) const {
     return n[b];
  }

  constexpr auto const * beginSpill() const {
     return spillBin;
  }
    
  constexpr auto const * endSpill() const {
     return beginSpill() + std::min(spillSize(), uint32_t(nspills));
  }

  Counter  n[nbins()];
  Counter  nspills;
  index_type bins[nbins()*binSize()];
  index_type spillBin[spillSize()];
};

#endif // HeterogeneousCore_CUDAUtilities_HistoContainer_h
