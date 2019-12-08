#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/cpu_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"

TEST_CASE("cpu_unique_ptr", "[cudaMemTools]") {

  SECTION("Single element") {
    auto ptr = cudautils::make_cpu_unique<int>(cudaStreamDefault);
    REQUIRE(ptr != nullptr);
    *ptr = 1; 
    cudautils::memsetAsync(ptr,0,cudaStreamDefault);
    REQUIRE(0==*ptr);
  }

  SECTION("Reset") {
    auto ptr = cudautils::make_cpu_unique<int>(cudaStreamDefault);
    REQUIRE(ptr != nullptr);

    ptr.reset();
    REQUIRE(ptr.get() == nullptr);
  }

  SECTION("Multiple elements") {
    auto ptr = cudautils::make_cpu_unique<int[]>(10,cudaStreamDefault);
    REQUIRE(ptr != nullptr);
    for (int i=0; i<10; ++i) ptr[i]=1;
    cudautils::memsetAsync(ptr,0,10,cudaStreamDefault);
    int s=0; for (int i=0; i<10; ++i) s+=ptr[i];
    REQUIRE(0==s);
  }

  SECTION("Allocating too much") {
    constexpr size_t maxSize = 1 << 30;  // 8**10
    auto ptr = cudautils::make_cpu_unique<char[]>(maxSize+1,cudaStreamDefault);
    REQUIRE(ptr != nullptr);
  }
}
