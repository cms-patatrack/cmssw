#ifndef CUDADataFormatsCommonHostProduct_H
#define CUDADataFormatsCommonHostProduct_H

#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cpu_unique_ptr.h"

// a heterogeneous unique pointer...
template <typename T>
class HostProduct {
public:
  HostProduct() = default;  // make root happy
  ~HostProduct() = default;
  HostProduct(HostProduct&&) = default;
  HostProduct& operator=(HostProduct&&) = default;

  explicit HostProduct(cudautils::host::unique_ptr<T>&& p) : hm_ptr(std::move(p)) {}
  explicit HostProduct(cudautils::cpu::unique_ptr<T>&& p) : cm_ptr(std::move(p)) {}

  auto const* get() const { return hm_ptr ? hm_ptr.get() : cm_ptr.get(); }

  auto const& operator*() const { return *get(); }

  auto const* operator-> () const { return get(); }

private:
  cudautils::host::unique_ptr<T> hm_ptr;  //!
  cudautils::cpu::unique_ptr<T>  cm_ptr;  //!
};

#endif
