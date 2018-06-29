#include "HeterogeneousCore/CUDAUtilities/interface/GPUVector.h"

#include <bitset>
#include <numeric>
#include <vector>

#include <cuda.h>

#include "catch.hpp"

__global__ void vector_sizeCapacity(GPUVectorWrapper<int> vec, unsigned int *ret) {
  *ret = 0;
  if(vec.capacity() == 10) {
    *ret = *ret | 1<<0;
  }
  if(vec.size() == 0) {
    *ret = *ret | 1<<1;
  }
}

__global__ void vector_elements(GPUVectorWrapper<int> vec, int *ret) {
  auto index = threadIdx.x + blockIdx.x*blockDim.x;
  ret[index] = (vec[index] == index);
}

__global__ void vector_pushback(GPUVectorWrapper<int> vec) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  vec.push_back(index);
}

__global__ void vector_emplaceback(GPUVectorWrapper<int> vec) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  vec.emplace_back(index);
}

__global__ void vector_access(GPUVectorWrapper<int> vec) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  vec[index] += index;
  atomicAdd(&vec.back(), 1);
}

__global__ void vector_resize(GPUVectorWrapper<int> vec, unsigned int *ret) {
  *ret = 0;
  if(vec.capacity() == 10) {
    *ret = *ret | 1<<0;
  }
  if(vec.size() == 10) {
    *ret = *ret | 1<<1;
  }

  vec.resize(5);

  if(vec.capacity() == 10) {
    *ret = *ret | 1<<2;
  }
  if(vec.size() == 5) {
    *ret = *ret | 1<<3;
  }
}

__global__ void vector_reset(GPUVectorWrapper<int> vec, unsigned int *ret) {
  *ret = 0;
  if(vec.capacity() == 10) {
    *ret = *ret | 1<<0;
  }
  if(vec.size() == 10) {
    *ret = *ret | 1<<1;
  }

  vec.reset();

  if(vec.capacity() == 10) {
    *ret = *ret | 1<<2;
  }
  if(vec.size() == 0) {
    *ret = *ret | 1<<3;
  }
}



TEST_CASE("Tests of GPUVector", "[GPUVector]") {
  int deviceCount = 0;
  auto ret = cudaGetDeviceCount( &deviceCount );
  if(ret != cudaSuccess || deviceCount < 1) {
    WARN("No CUDA devices, ignoring the tests");
    return;
  }

  auto current_device = cuda::device::current::get();

  auto vec_d = GPUVector<int>(10);

  SECTION("Construction") {
    REQUIRE(vec_d.size() == 0);
    REQUIRE(vec_d.capacity() == 10);

    auto res_d = cuda::memory::device::make_unique<unsigned int>(current_device);
    vector_sizeCapacity<<<1, 1>>>(vec_d, res_d.get());
    current_device.synchronize();

    unsigned int res;
    cuda::memory::copy(&res, res_d.get(), sizeof(unsigned int));
    auto ret = std::bitset<2>(res);
    for(int i=0; i<2; ++i) {
      INFO("Bit " << i);
      CHECK(ret.test(i));
    }
  }

  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);
  auto res_h = std::vector<int>(10, 0);

  SECTION("Copy to device") {
    auto vec_h = std::vector<int>(10);
    std::iota(vec_h.begin(), vec_h.end(), 0);

    SECTION("Synchronous") {
      vec_d.copyFrom(vec_h.data(), 10);

      auto res_d = cuda::memory::device::make_unique<int[]>(current_device, 10);
      vector_elements<<<1, 10>>>(vec_d, res_d.get());
      current_device.synchronize();

      cuda::memory::copy(res_h.data(), res_d.get(), 10*sizeof(int));
      for(int i=0; i<10; ++i) {
        INFO("Index " << i);
        CHECK(res_h[i] == 1); // all comparisons are true
      }
    }

    SECTION("Asynchronous") {
      vec_d.copyFromAsync(vec_h.data(), 10, stream.id());

      auto res_d = cuda::memory::device::make_unique<int[]>(current_device, 10);
      vector_elements<<<1, 10, 0, stream.id()>>>(vec_d, res_d.get());

      cuda::memory::async::copy(res_h.data(), res_d.get(), 10*sizeof(int), stream.id());
      stream.synchronize();
      for(int i=0; i<10; ++i) {
        INFO("Index " << i);
        CHECK(res_h[i] == 1); // all comparisons are true
      }
    }
  }

  SECTION("Copy from device") {
    auto vec_h = std::vector<int>(10);
    std::iota(vec_h.begin(), vec_h.end(), 0);

    SECTION("Synchronous") {
      vec_d.copyFrom(vec_h.data(), 10);

      auto ret = vec_d.copyTo(res_h.data(), 10);
      REQUIRE(ret == 10);
      for(int i=0; i<10; ++i) {
        INFO("Index " << i);
        CHECK(res_h[i] == i);
      }

      ret = vec_d.copyTo(res_h.data(), 5);
      REQUIRE(ret == 5);
      for(int i=0; i<5; ++i) {
        INFO("Index " << i);
        CHECK(res_h[i] == i);
      }

      ret = vec_d.copyTo(res_h.data(), 20);
      REQUIRE(ret == 10);
    }

    SECTION("Asynchronous") {
      vec_d.copyFromAsync(vec_h.data(), 10, stream.id());
      vec_d.updateMetadataAsync(stream.id());
      stream.synchronize();

      auto ret = vec_d.copyToAsync(res_h.data(), 10, stream.id());
      REQUIRE(ret == 10);
      stream.synchronize();
      for(int i=0; i<10; ++i) {
        INFO("Index " << i);
        CHECK(res_h[i] == i);
      }

      std::fill(res_h.begin(), res_h.end(), -1);
      ret = vec_d.copyToAsync(res_h.data(), 5, stream.id());
      REQUIRE(ret == 5);
      stream.synchronize();
      for(int i=0; i<5; ++i) {
        INFO("Index " << i);
        CHECK(res_h[i] == i);
      }

      std::fill(res_h.begin(), res_h.end(), -1);
      ret = vec_d.copyToAsync(res_h.data(), 20, stream.id());
      REQUIRE(ret == 10);
    }
  }

  SECTION("push_back") {
    vector_pushback<<<1, 10>>>(vec_d);
    current_device.synchronize();

    vec_d.updateMetadata();
    REQUIRE(vec_d.size() == 10);

    vec_d.copyTo(res_h.data(), 10);
    for(int i=0; i<10; ++i) {
      CHECK(std::find(res_h.begin(), res_h.end(), i) != res_h.end());
    }
  }

  SECTION("emplace_back") {
    vector_emplaceback<<<1, 10>>>(vec_d);
    current_device.synchronize();

    vec_d.updateMetadata();
    REQUIRE(vec_d.size() == 10);

    vec_d.copyTo(res_h.data(), 10);
    for(int i=0; i<10; ++i) {
      CHECK(std::find(res_h.begin(), res_h.end(), i) != res_h.end());
    }
  }

  SECTION("Element access") {
    auto vec_h = std::vector<int>(10);
    std::iota(vec_h.begin(), vec_h.end(), 0);
    vec_d.copyFrom(vec_h.data(), 10);

    vector_access<<<1, 9>>>(vec_d);
    current_device.synchronize();

    vec_d.copyTo(res_h.data(), 10);
    for(int i=0; i<9; ++i) {
      CHECK(res_h[i] == i*2);
    }
    CHECK(res_h[9] == 9+9);
  }

  SECTION("Resize") {
    auto vec_h = std::vector<int>(10);
    std::iota(vec_h.begin(), vec_h.end(), 0);
    vec_d.copyFrom(vec_h.data(), 10);

    auto res_d = cuda::memory::device::make_unique<unsigned int>(current_device);
    vector_resize<<<1, 1>>>(vec_d, res_d.get());

    unsigned int res;
    cuda::memory::copy(&res, res_d.get(), sizeof(unsigned int));
    auto ret = std::bitset<4>(res);
    for(int i=0; i<4; ++i) {
      INFO("Bit " << i);
      CHECK(ret.test(i));
    }
  }

  SECTION("Reset") {
    auto vec_h = std::vector<int>(10);
    std::iota(vec_h.begin(), vec_h.end(), 0);
    vec_d.copyFrom(vec_h.data(), 10);

    auto res_d = cuda::memory::device::make_unique<unsigned int>(current_device);
    vector_reset<<<1, 1>>>(vec_d, res_d.get());

    unsigned int res;
    cuda::memory::copy(&res, res_d.get(), sizeof(unsigned int));
    auto ret = std::bitset<4>(res);
    for(int i=0; i<4; ++i) {
      INFO("Bit " << i);
      CHECK(ret.test(i));
    }
  }
}
