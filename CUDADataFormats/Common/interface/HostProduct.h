#ifndef CUDADataFormatsCommonHostProduct_H
#define CUDADataFormatsCommonHostProduct_H

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

// a heterogeneous unique pointer...
template<typename T>
class HostProduct {
public:


  HostProduct() = default; // make root happy
  ~HostProduct() = default;
  HostProduct(HostProduct&&) = default;
  HostProduct& operator=(HostProduct&&) = default;

  explicit HostProduct(cudautils::host::unique_ptr<T> && p) : hm_ptr(std::move(p)) {}
  explicit HostProduct(std::unique_ptr<T> && p) : std_ptr(std::move(p)) {}


  T const * get() const {
    return hm_ptr ? hm_ptr.get() : std_ptr.get();
  }
  
  T const & operator*() const {
    return *get();
  }

  T const * operator->() const {
    return get();
  }
  

private:
  cudautils::host::unique_ptr<T> hm_ptr;
  std::unique_ptr<T> std_ptr;

};

#endif
