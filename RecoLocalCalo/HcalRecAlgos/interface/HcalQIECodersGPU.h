#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalQIECodersGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalQIECodersGPU_h

#include "CondFormats/HcalObjects/interface/HcalQIEData.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#endif

class HcalQIECodersGPU {
public:
    struct Product {
        ~Product();
        float *offset00, *offset01, *offset02, *offset03,
              *offset10, *offset11, *offset12, *offset13,
              *offset20, *offset21, *offset22, *offset23,
              *offset30, *offset31, *offset32, *offset33;
        float *slope00, *slope01, *slope02, *slope03,
              *slope10, *slope11, *slope12, *slope13,
              *slope20, *slope21, *slope22, *slope23,
              *slope30, *slope31, *slope32, *slope33;
    };

#ifndef __CUDACC__
    // rearrange reco params
    HcalQIECodersGPU(HcalQIEData const&);

    // will trigger deallocation of Product thru ~Product
    ~HcalQIECodersGPU() = default;

    // get device pointers
    Product const& getProduct(cuda::stream_t<>&) const;

    // 
    static std::string name() { return std::string{"hcalQIECodersGPU"}; }

private:
    uint64_t totalChannels_;
    std::vector<float, CUDAHostAllocator<float>> 
        offset00_, offset01_, offset02_, offset03_,
        offset10_, offset11_, offset12_, offset13_,
        offset20_, offset21_, offset22_, offset23_,
        offset30_, offset31_, offset32_, offset33_;
    std::vector<float, CUDAHostAllocator<float>> 
        slope00_, slope01_, slope02_, slope03_,
        slope10_, slope11_, slope12_, slope13_,
        slope20_, slope21_, slope22_, slope23_,
        slope30_, slope31_, slope32_, slope33_;

    CUDAESProduct<Product> product_;
#endif
};

#endif
