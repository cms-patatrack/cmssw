#ifndef HeterogeneousCoreCUDAUtilitiesAtomicPairCounter_H
#define HeterogeneousCoreCUDAUtilitiesAtomicPairCounter_H

#include<cstdint>

class AtomicPairCounter {
public:

  using c_type = unsigned long long int;

  AtomicPairCounter(){}
  AtomicPairCounter(c_type i) { counter.ac=i;}

#ifdef __CUDACC__
  __device__ __host__
  AtomicPairCounter & operator=(c_type i) { counter.ac=i; return *this;}
#endif

  struct Counters {
    uint32_t n;  // total size
    uint32_t m;  // number of elements
  };

  union Atomic2 {
    Counters counters;
    c_type ac;
  };

#ifdef __CUDACC__

  static constexpr c_type incr = 1UL<<32;

  __device__ __host__
  Counters get() const { return counter.counters;}

  // increment n by 1 and m by i.  return previous value
  __device__
  Counters add(c_type i) {
    i+=incr;
    Atomic2 ret;
    ret.ac = atomicAdd(&counter.ac,i);
    return ret.counters;
  }

#endif

private:

  Atomic2 counter;

};


#endif
