#include "catch.hpp"

#include "HeterogeneousCore/CUDAUtilities/interface/cpu_unique_ptr.h"

TEST_CASE("cpu_unique_ptr", "[cudaMemTools]") {

  SECTION("Single element") {
    auto ptr = cudautils::make_cpu_unique<int>();
    REQUIRE(ptr != nullptr);
  }

  SECTION("Reset") {
    auto ptr = cudautils::make_cpu_unique<int>();
    REQUIRE(ptr != nullptr);

    ptr.reset();
    REQUIRE(ptr.get() == nullptr);
  }

  SECTION("Multiple elements") {
    auto ptr = cudautils::make_host_unique<int[]>(10);
    REQUIRE(ptr != nullptr);
  }

  SECTION("Allocating too much") {
    constexpr size_t maxSize = 1 << 30;  // 8**10
    auto ptr = cudautils::make_cpu_unique<char[]>(maxSize+1);
    ptr.reset();
    REQUIRE(ptr != nullptr);
  }
}
