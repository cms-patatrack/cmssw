#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

__global__ void test_constructors() {
    auto id0 = DetId();
    auto id1 = DetId(DetId::Hcal, 0);
    auto id2 = DetId(DetId::Hcal, 1);
}

__global__ void test_getters() {
    auto id1 = DetId(DetId::Hcal, 1);
    auto det = id1.det();
    auto subdet = id1.subdetId();
    auto t0 = id1();
    auto rawid = id1.rawId();
    auto isnull = id1.null();

    printf("%d\n", rawid);
}

__global__ void test_operators() {
    auto id1 = DetId(DetId::Hcal, 0);
    auto id2 = DetId(DetId::Hcal, 1);
    auto t0 = id1<id2;
    auto t1 = id1==id2;
    auto t2 = id1!=id2;
}

__global__ void test_hcaldetid_constructors() {
    HcalDetId id1();
    HcalDetId id3(0);
    HcalSubdetector sub = HcalBarrel;
    HcalDetId id2(sub, 5, 5, 0);
}

int main(int argc, char** argv) {
    std::cout << "hello world" << std::endl;

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "nDevices = " << nDevices << std::endl;

    test_constructors<<<1,1>>>();
    test_getters<<<1,1>>>();
    test_operators<<<1,1>>>();

    test_hcaldetid_constructors<<<1,1>>>();

    cudaDeviceSynchronize();

    return 0;
}
