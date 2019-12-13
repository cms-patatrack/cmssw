#ifndef CUDADataFormatsCommonHeterogeneousSoA_H
#define CUDADataFormatsCommonHeterogeneousSoA_H

#include <cassert>

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cpu_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

// a heterogeneous unique pointer...
template <typename T>
class HeterogeneousSoA {
public:
  using Product = T;

  HeterogeneousSoA() = default;  // make root happy
  ~HeterogeneousSoA() = default;
  HeterogeneousSoA(HeterogeneousSoA &&) = default;
  HeterogeneousSoA &operator=(HeterogeneousSoA &&) = default;

  explicit HeterogeneousSoA(cudautils::device::unique_ptr<T> &&p) : dm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(cudautils::host::unique_ptr<T> &&p) : hm_ptr(std::move(p)) {}
  explicit HeterogeneousSoA(cudautils::cpu::unique_ptr<T> &&p) : cm_ptr(std::move(p)) {}

  auto const *get() const { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : cm_ptr.get()); }

  auto const &operator*() const { return *get(); }

  auto const *operator-> () const { return get(); }

  auto *get() { return dm_ptr ? dm_ptr.get() : (hm_ptr ? hm_ptr.get() : cm_ptr.get()); }

  auto &operator*() { return *get(); }

  auto *operator-> () { return get(); }

  // in reality valid only for GPU version...
  cudautils::host::unique_ptr<T> toHostAsync(cudaStream_t stream) const {
    assert(dm_ptr);
    auto ret = cudautils::make_host_unique<T>(stream);
    cudaCheck(cudaMemcpyAsync(ret.get(), dm_ptr.get(), sizeof(T), cudaMemcpyDefault, stream));
    return ret;
  }

private:
  // a union wan't do it, a variant will not be more efficienct
  cudautils::device::unique_ptr<T> dm_ptr;  //!
  cudautils::host::unique_ptr<T> hm_ptr;    //!
  cudautils::cpu::unique_ptr<T> cm_ptr;    //!
};

namespace cudaCompat {

  struct GPUTraits {
    static constexpr const char * name = "GPU"; 
    static constexpr bool runOnDevice = true;

    template <typename T>
    using unique_ptr = cudautils::device::unique_ptr<T>;

    template <typename T>
    static auto make_unique(cudaStream_t stream) {
      return cudautils::make_device_unique<T>(stream);
    }

    template <typename T>
    static auto make_unique(size_t size, cudaStream_t stream) {
      return cudautils::make_device_unique<T>(size, stream);
    }

    template <typename T>
    static auto make_host_unique(cudaStream_t stream) {
      return cudautils::make_host_unique<T>(stream);
    }

    template <typename T>
    static auto make_device_unique(cudaStream_t stream) {
      return cudautils::make_device_unique<T>(stream);
    }

    template <typename T>
    static auto make_device_unique(size_t size, cudaStream_t stream) {
      return cudautils::make_device_unique<T>(size, stream);
    }
  };

  struct HostTraits {
    static constexpr const char * name = "HOST";
    static constexpr bool runOnDevice = false;

    template <typename T>
    using unique_ptr = cudautils::host::unique_ptr<T>;

    template <typename T>
    static auto make_unique(cudaStream_t stream) {
      return cudautils::make_host_unique<T>(stream);
    }

    template <typename T>
    static auto make_host_unique(cudaStream_t stream) {
      return cudautils::make_host_unique<T>(stream);
    }

    template <typename T>
    static auto make_device_unique(cudaStream_t stream) {
      return cudautils::make_device_unique<T>(stream);
    }

    template <typename T>
    static auto make_device_unique(size_t size, cudaStream_t stream) {
      return cudautils::make_device_unique<T>(size, stream);
    }
  };

  struct CPUTraits {
    static constexpr const char * name = "CPU";
    static constexpr bool runOnDevice = false;

    template <typename T>
    using unique_ptr = cudautils::cpu::unique_ptr<T>;;

    template <typename T>
    static auto make_unique() {
      return cudautils::make_cpu_unique<T>(cudaStreamDefault);
    }

    template <typename T>
    static auto make_unique(size_t size) {
      return cudautils::make_cpu_unique<T>(size,cudaStreamDefault);
    }

    template <typename T>
    static auto make_unique(cudaStream_t stream) {
      return cudautils::make_cpu_unique<T>(stream);
    }

    template <typename T>
    static auto make_unique(size_t size, cudaStream_t stream) {
      return cudautils::make_cpu_unique<T>(size, stream);
    }

    template <typename T>
    static auto make_host_unique(cudaStream_t stream) {
      return cudautils::make_cpu_unique<T>(stream);
    }

    template <typename T>
    static auto make_device_unique(cudaStream_t stream) {
      return cudautils::make_cpu_unique<T>(stream);
    }

    template <typename T>
    static auto make_device_unique(size_t size, cudaStream_t stream) {
      return cudautils::make_cpu_unique<T>(size, stream);
    }
  };

}  // namespace cudaCompat

// a heterogeneous unique pointer (of a different sort) ...
template <typename T, typename Traits>
class HeterogeneousSoAImpl {
public:
  template <typename V>
  using unique_ptr = typename Traits::template unique_ptr<V>;

  HeterogeneousSoAImpl() = default;  // make root happy
  ~HeterogeneousSoAImpl() = default;
  HeterogeneousSoAImpl(HeterogeneousSoAImpl &&) = default;
  HeterogeneousSoAImpl &operator=(HeterogeneousSoAImpl &&) = default;

  explicit HeterogeneousSoAImpl(unique_ptr<T> &&p) : m_ptr(std::move(p)) {}
  explicit HeterogeneousSoAImpl(cudaStream_t stream);

  T const *get() const { return m_ptr.get(); }

  T *get() { return m_ptr.get(); }

  cudautils::host::unique_ptr<T> toHostAsync(cudaStream_t stream) const;

private:
  unique_ptr<T> m_ptr;  //!
};

template <typename T, typename Traits>
HeterogeneousSoAImpl<T, Traits>::HeterogeneousSoAImpl(cudaStream_t stream) {
  m_ptr = Traits::template make_unique<T>(stream);
}

// in reality valid only for GPU version...
template <typename T, typename Traits>
cudautils::host::unique_ptr<T> HeterogeneousSoAImpl<T, Traits>::toHostAsync(cudaStream_t stream) const {
  auto ret = cudautils::make_host_unique<T>(stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), get(), sizeof(T), cudaMemcpyDefault, stream));
  return ret;
}

template <typename T>
using HeterogeneousSoAGPU = HeterogeneousSoAImpl<T, cudaCompat::GPUTraits>;
template <typename T>
using HeterogeneousSoACPU = HeterogeneousSoAImpl<T, cudaCompat::CPUTraits>;
template <typename T>
using HeterogeneousSoAHost = HeterogeneousSoAImpl<T, cudaCompat::HostTraits>;

#endif
