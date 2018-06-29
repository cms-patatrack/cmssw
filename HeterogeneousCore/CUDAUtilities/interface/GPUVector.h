#ifndef HeterogeneousCore_CUDAUtilities_GPUVector_h
#define HeterogeneousCore_CUDAUtilities_GPUVector_h

#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>

template <typename T>
class GPUVectorWrapper;

/**
 * This class owns the GPU memory, and provides interface to transfer
 * the data to CPU memory.
 */
template <typename T>
class GPUVector {
public:
  explicit GPUVector(int capacity) {
    static_assert(std::is_trivially_destructible<T>::value);

    m_sizeCapacity_host.m_size = 0;
    m_sizeCapacity_host.m_capacity = capacity;

    auto current_device = cuda::device::current::get();
    m_sizeCapacity = cuda::memory::device::make_unique<SizeCapacity>(current_device);
    m_data = cuda::memory::device::make_unique<T[]>(current_device, capacity);

    updateMetadataToDevice();
  }

  void updateMetadata() {
    cuda::memory::copy(&m_sizeCapacity_host, m_sizeCapacity.get(), sizeof(SizeCapacity));
  }
  void updateMetadataAsync(cudaStream_t stream) {
    cuda::memory::async::copy(&m_sizeCapacity_host, m_sizeCapacity.get(), sizeof(SizeCapacity), stream);
  }

  int size() const { return m_sizeCapacity_host.m_size; }
  int capacity() const { return m_sizeCapacity_host.m_capacity; }

  const T *data() const { return m_data.get(); }
  T *data() { return m_data.get(); }

  void copyFrom(const T *src, int num) {
    assert(num <= m_sizeCapacity_host.m_capacity);
    cuda::memory::copy(m_data.get(), src, num*sizeof(T));
    m_sizeCapacity_host.m_size = num;
    updateMetadataToDevice();
  }

  void copyFromAsync(const T *src, int num, cudaStream_t stream) {
    assert(num <= m_sizeCapacity_host.m_capacity);
    cuda::memory::async::copy(m_data.get(), src, num*sizeof(T), stream);
    m_sizeCapacity_host.m_size = num;
    updateMetadataToDeviceAsync(stream);
  }

  int copyTo(T *dst, int num) {
    updateMetadata();
    int copied = std::min(num, m_sizeCapacity_host.m_size);
    cuda::memory::copy(dst, m_data.get(), copied*sizeof(T));
    return copied;
  }
  int copyToAsync(T *dst, int num, cudaStream_t stream) {
    // calling updateMetadataAsync() or otherwise guaranteeing the host
    // and device to be in synch with the size is on the
    // responsibility of the caller
    int copied = std::min(num, m_sizeCapacity_host.m_size);
    cuda::memory::async::copy(dst, m_data.get(), copied*sizeof(T), stream);
    return copied;
  }

private:
  void updateMetadataToDevice() {
    cuda::memory::copy(m_sizeCapacity.get(), &m_sizeCapacity_host, sizeof(SizeCapacity));
  }
  void updateMetadataToDeviceAsync(cudaStream_t stream) {
    cuda::memory::async::copy(m_sizeCapacity.get(), &m_sizeCapacity_host, sizeof(SizeCapacity), stream);
  }

  friend GPUVectorWrapper<T>;

  struct SizeCapacity {
#if defined(__NVCC__) || defined(__CUDACC__)
    __device__ int addElement() {
      auto previousSize = atomicAdd(&m_size, 1);
      assert(previousSize < m_capacity);
      return previousSize;
    }

    __device__ void resize(int size) {
      assert(size <= m_capacity);
      m_size = size;
    }
#endif

    int m_size;
    int m_capacity;
  };

  SizeCapacity m_sizeCapacity_host;
  cuda::memory::device::unique_ptr<SizeCapacity> m_sizeCapacity;
  cuda::memory::device::unique_ptr<T[]> m_data;
};

/**
 * This class acts as a device wrapper of GPUVector<T> by containing
 * the pointers to GPU memory and an interface for manipulations in
 * the device. It can be passed by value to the kernels.
 */
template <typename T>
class GPUVectorWrapper {
public:
  GPUVectorWrapper(GPUVector<T>& vec): // allow automatic conversion
    m_sizeCapacity(vec.m_sizeCapacity.get()),
    m_data(vec.m_data.get())
  {}

#if defined(__NVCC__) || defined(__CUDACC__)
  // thread-safe version of the vector, when used in a CUDA kernel
  __device__ void push_back(const T &element) {
    auto index = m_sizeCapacity->addElement();
    m_data[index] = element;
  }

  template <typename... Args>
  __device__ void emplace_back(Args&&... args) {
    auto index = m_sizeCapacity->addElement();
    new (&m_data[index]) T(std::forward<Args>(args)...);
  }

  __device__ const T& back() const {
    assert(m_sizeCapacity->m_size > 0);
    return m_data[m_sizeCapacity->m_size - 1];
  }
  __device__ T& back() {
    assert(m_sizeCapacity->m_size > 0);
    return m_data[m_sizeCapacity->m_size - 1];
  }

  __device__ void reset() { m_sizeCapacity->m_size = 0; }

  __device__ int size() const { return m_sizeCapacity->m_size; }

  __device__ int capacity() const { return m_sizeCapacity->m_capacity; }

  __device__ void resize(int size) { m_sizeCapacity->resize(size); }

  __device__ const T& operator[](int i) const { return m_data[i]; }
  __device__ T& operator[](int i) { return m_data[i]; }

  __device__ const T *data() const { return m_data; }
  __device__ T *data() { return m_data; }

#endif

private:
  typename GPUVector<T>::SizeCapacity *m_sizeCapacity = nullptr;
  T *m_data = nullptr;
};


#endif
