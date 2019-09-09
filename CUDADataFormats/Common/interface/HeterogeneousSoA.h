#ifndef CUDADataFormatsCommonHeterogeneousSoA_H
#define CUDADataFormatsCommonHeterogeneousSoA_H

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// a heterogeneous unique pointer...
template<typename T>
class HeterogeneousSoA {
public:

  using Product = T;

  HeterogeneousSoA() = default; // make root happy
  virtual ~HeterogeneousSoA() = default;
  HeterogeneousSoA(HeterogeneousSoA&&) = default;
  HeterogeneousSoA& operator=(HeterogeneousSoA&&) = default;


  virtual T const * get() const = 0;
  virtual T * get() = 0;

  auto & operator*() {
    return *get();
  }

  auto * operator->() {
    return get();
  }

  auto const & operator*() const {
    return *get();
  }

  auto const * operator->() const {
    return get();
  }


};

namespace cudaCompat {

  struct GPUTraits {

    template<typename T>
    using unique_ptr =  cudautils::device::unique_ptr<T>;

    template<typename T>
    static auto make_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(stream);
    }

    template<typename T>
    static auto make_unique(edm::Service<CUDAService> & cs, size_t size, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(size, stream);
    }

    template<typename T>
    static auto make_host_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_host_unique<T>(stream);
    }


    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(stream);
    }

    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService> & cs, size_t size, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(size, stream);
    }


  };


  struct HostTraits {

    template<typename T>
    using unique_ptr =  cudautils::host::unique_ptr<T>;

    template<typename T>
    static auto make_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_host_unique<T>(stream);
    }


    template<typename T>
    static auto make_host_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_host_unique<T>(stream);
    }


    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(stream);
    }

    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService> & cs, size_t size, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(size, stream);
    }


  };


  struct CPUTraits {

    template<typename T>
    using unique_ptr =  std::unique_ptr<T>;

    template<typename T>
    static auto make_unique(edm::Service<CUDAService>&, cuda::stream_t<> &)    {
      return std::make_unique<T>();
    }


    template<typename T>
    static auto make_unique(edm::Service<CUDAService>&, size_t size, cuda::stream_t<> &)    {
      return std::make_unique<T>(size);
    }


    template<typename T>
    static auto make_host_unique(edm::Service<CUDAService>&, cuda::stream_t<> &)    {
      return std::make_unique<T>();
    }


    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService>&, cuda::stream_t<> &)    {
      return std::make_unique<T>();
    }

    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService>&, size_t size, cuda::stream_t<> &)    {
      return std::make_unique<T>(size);
    }


  };

}



// a heterogeneous unique pointer...
template<typename T, typename Traits>
class HeterogeneousSoAImpl : public HeterogeneousSoA<T> {
public:

  template<typename V>
  using unique_ptr = typename Traits:: template unique_ptr<V>;


  HeterogeneousSoAImpl() = default; // make root happy
  ~HeterogeneousSoAImpl() = default;
  HeterogeneousSoAImpl(HeterogeneousSoAImpl&&) = default;
  HeterogeneousSoAImpl& operator=(HeterogeneousSoAImpl&&) = default;

  explicit HeterogeneousSoAImpl(unique_ptr<T> && p) : m_ptr(std::move(p)) {}
  explicit HeterogeneousSoAImpl(cuda::stream_t<> &stream);

  T const * get() const final {
    return m_ptr.get();
  }
  
  T * get() final {
    return m_ptr.get();
  }


  cudautils::host::unique_ptr<T> toHostAsync(cuda::stream_t<>& stream) const;

private:
  unique_ptr<T> m_ptr; //!

};



template<typename T, typename Traits>
HeterogeneousSoAImpl<T,Traits>::HeterogeneousSoAImpl(cuda::stream_t<> &stream) {
  edm::Service<CUDAService> cs;
  m_ptr = Traits:: template make_unique<T>(cs,stream);
}


// in reality valid only for GPU version...
template<typename T, typename Traits>
cudautils::host::unique_ptr<T>
HeterogeneousSoAImpl<T,Traits>::toHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<T>(stream);
  cudaCheck(cudaMemcpyAsync(ret.get(), get(), sizeof(T), cudaMemcpyDefault, stream.id()));
  return ret;
}


template<typename T>
using HeterogeneousSoAGPU = HeterogeneousSoAImpl<T,cudaCompat::GPUTraits>;
template<typename T>
using HeterogeneousSoACPU =  HeterogeneousSoAImpl<T,cudaCompat::CPUTraits>;
template<typename T>
using HeterogeneousSoAHost = HeterogeneousSoAImpl<T,cudaCompat::HostTraits>;


#endif
