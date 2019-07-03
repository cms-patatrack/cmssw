#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalGainsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalGainsGPU_h

#include "CondFormats/HcalObjects/interface/HcalGains.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#endif

class HcalGainsGPU {
public:
    struct Product {
        ~Product();
        float *value0;
        float *value1;
        float *value2;
        float *value3;
    };

#ifndef __CUDACC__
    // rearrange reco params
    HcalGainsGPU(HcalGains const&);

    // will trigger deallocation of Product thru ~Product
    ~HcalGainsGPU() = default;

    // get device pointers
    Product const& getProduct(cuda::stream_t<>&) const;

    // 
    static std::string name() { return std::string{"hcalGainsGPU"}; }

private:
    uint64_t totalChannels_;
    std::vector<float, CUDAHostAllocator<float>> value0_, value1_, value2_, value3_;

    CUDAESProduct<Product> product_;
#endif
};

#endif
