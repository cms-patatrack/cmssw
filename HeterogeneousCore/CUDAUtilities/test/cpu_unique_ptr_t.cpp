#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/cpu_unique_ptr.h"

TEST_CASE("cpu_unique_ptr", "[cudaMemTools]") {

  SECTION("Single element") {
    auto ptr = cudautils::make_cpu_unique<int>(cudaStreamDefault);
    REQUIRE(ptr != nullptr);
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
  }

  SECTION("Allocating too much") {
    constexpr size_t maxSize = 1 << 30;  // 8**10
    auto ptr = cudautils::make_cpu_unique<char[]>(maxSize+1,cudaStreamDefault);
    ptr.reset();
    REQUIRE(ptr != nullptr);
  }
}
