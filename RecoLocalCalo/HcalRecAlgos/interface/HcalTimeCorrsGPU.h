#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalTimeCorrsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalTimeCorrsGPU_h

#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#endif

class HcalTimeCorrsGPU {
public:
    struct Product {
        ~Product();
        float *value;
    };

#ifndef __CUDACC__
    // rearrange reco params
    HcalTimeCorrsGPU(HcalTimeCorrs const&);

    // will trigger deallocation of Product thru ~Product
    ~HcalTimeCorrsGPU() = default;

    // get device pointers
    Product const& getProduct(cuda::stream_t<>&) const;

    // 
    static std::string name() { return std::string{"hcalTimeCorrsGPU"}; }

private:
    std::vector<float, CUDAHostAllocator<float>> value_;

    CUDAESProduct<Product> product_;
#endif
};

#endif
