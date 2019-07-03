#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalLUTCorrsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalLUTCorrsGPU_h

#include "CondFormats/HcalObjects/interface/HcalLUTCorrs.h"

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#endif

class HcalLUTCorrsGPU {
public:
    struct Product {
        ~Product();
        float *value;
    };

#ifndef __CUDACC__
    // rearrange reco params
    HcalLUTCorrsGPU(HcalLUTCorrs const&);

    // will trigger deallocation of Product thru ~Product
    ~HcalLUTCorrsGPU() = default;

    // get device pointers
    Product const& getProduct(cuda::stream_t<>&) const;

    // 
    static std::string name() { return std::string{"hcalLUTCorrsGPU"}; }

private:
    std::vector<float, CUDAHostAllocator<float>> value_;

    CUDAESProduct<Product> product_;
#endif
};

#endif
