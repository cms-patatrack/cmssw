#ifndef HeterogeneousCore_CUDAUtilities_interface_cpu_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_cpu_unique_ptr_h

#include <memory>
#include <functional>

#include <cstdlib>
#include <cuda_runtime.h>

namespace cudautils {
  namespace cpu {
    namespace impl {
      // Additional layer of types to distinguish from device:: and host::unique_ptr
      class CPUDeleter {
      public:
        CPUDeleter() = default;

        void operator()(void *ptr) {
            ::free(ptr);
        }
      };
    }  // namespace impl

    template <typename T>
    using unique_ptr = std::unique_ptr<T, impl::CPUDeleter>;

    namespace impl {
      template <typename T>
      struct make_cpu_unique_selector {
        using non_array = cudautils::cpu::unique_ptr<T>;
      };
      template <typename T>
      struct make_cpu_unique_selector<T[]> {
        using unbounded_array = cudautils::cpu::unique_ptr<T[]>;
      };
      template <typename T, size_t N>
      struct make_cpu_unique_selector<T[N]> {
        struct bounded_array {};
      };
    }  // namespace impl
  }    // namespace cpu

  template <typename T>
  typename cpu::impl::make_cpu_unique_selector<T>::non_array make_cpu_unique(cudaStream_t) {
    static_assert(std::is_trivially_constructible<T>::value,
                  "Allocating with non-trivial constructor on the cpu memory is not supported");
    void *mem = ::malloc(sizeof(T));
    return typename cpu::impl::make_cpu_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                            cpu::impl::CPUDeleter()};
  }

  template <typename T>
  typename cpu::impl::make_cpu_unique_selector<T>::unbounded_array make_cpu_unique(size_t n, cudaStream_t) {
    using element_type = typename std::remove_extent<T>::type;
    static_assert(std::is_trivially_constructible<element_type>::value,
                  "Allocating with non-trivial constructor on the cpu memory is not supported");
    void *mem = ::malloc(n * sizeof(element_type));
    return typename cpu::impl::make_cpu_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                  cpu::impl::CPUDeleter()};
  }

  template <typename T, typename... Args>
  typename cpu::impl::make_cpu_unique_selector<T>::bounded_array make_cpu_unique(Args &&...) = delete;

  // No check for the trivial constructor, make it clear in the interface
  template <typename T>
  typename cpu::impl::make_cpu_unique_selector<T>::non_array make_cpu_unique_uninitialized(cudaStream_t) {
    void *mem = ::malloc(sizeof(T));
    return typename cpu::impl::make_cpu_unique_selector<T>::non_array{reinterpret_cast<T *>(mem),
                                                                            cpu::impl::CPUDeleter()};
  }

  template <typename T>
  typename cpu::impl::make_cpu_unique_selector<T>::unbounded_array make_cpu_unique_uninitialized(size_t n, cudaStream_t) {
    using element_type = typename std::remove_extent<T>::type;
    void *mem = ::malloc(n * sizeof(element_type));
    return typename cpu::impl::make_cpu_unique_selector<T>::unbounded_array{reinterpret_cast<element_type *>(mem),
                                                                                  cpu::impl::CPUDeleter()};
  }

  template <typename T, typename... Args>
  typename cpu::impl::make_cpu_unique_selector<T>::bounded_array make_cpu_unique_uninitialized(Args &&...) =
      delete;
}  // namespace cudautils

#endif
