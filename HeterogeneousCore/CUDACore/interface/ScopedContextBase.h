#ifndef HeterogeneousCore_CUDACore_ScopedContextBase_h
#define HeterogeneousCore_CUDACore_ScopedContextBase_h

#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SharedStreamPtr.h"

namespace cms {
  namespace cuda {
    class ProductBase;

    class ScopedContextBase {
    public:
      ScopedContextBase(ScopedContextBase const&) = delete;
      ScopedContextBase& operator=(ScopedContextBase const&) = delete;
      ScopedContextBase(ScopedContextBase&&) = delete;
      ScopedContextBase& operator=(ScopedContextBase&&) = delete;

      int device() const { return currentDevice_; }

      // cudaStream_t is a pointer to a thread-safe object, for which a
      // mutable access is needed even if the ScopedContext itself
      // would be const. Therefore it is ok to return a non-const
      // pointer from a const method here.
      cudaStream_t stream() const { return stream_.get(); }
      const SharedStreamPtr& streamPtr() const { return stream_; }

      template <typename T>
      typename cms::cuda::device::impl::make_device_unique_selector<T>::non_array make_device_unique() {
        return cms::cuda::make_device_unique<T>(stream());
      }

      template <typename T>
      typename cms::cuda::device::impl::make_device_unique_selector<T>::unbounded_array make_device_unique(size_t n) {
        return cms::cuda::make_device_unique<T>(n, stream());
      }

      template <typename T, typename... Args>
      typename cms::cuda::device::impl::make_device_unique_selector<T>::bounded_array make_device_unique(Args&&...) =
          delete;

    protected:
      // The constructors set the current device, but the device
      // is not set back to the previous value at the destructor. This
      // should be sufficient (and tiny bit faster) as all CUDA API
      // functions relying on the current device should be called from
      // the scope where this context is. The current device doesn't
      // really matter between modules (or across TBB tasks).
      explicit ScopedContextBase(edm::StreamID streamID);

      explicit ScopedContextBase(const ProductBase& data);

      explicit ScopedContextBase(int device, SharedStreamPtr stream);

    private:
      int currentDevice_;
      SharedStreamPtr stream_;
    };
  }  // namespace cuda
}  // namespace cms

#endif
