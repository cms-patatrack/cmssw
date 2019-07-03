#ifndef RecoLocalCalo_HcalRecAlgos_interface_HcalRecoParamsGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HcalRecoParamsGPU_h

#ifndef __CUDACC__
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAESProduct.h"
#endif

class HcalRecoParams;

class HcalRecoParamsGPU {
public:
    struct Product {
        ~Product();
        uint32_t *param1, *param2;
    };

#ifndef __CUDACC__
    // rearrange reco params
    HcalRecoParamsGPU(HcalRecoParams const&);

    // will trigger deallocation of Product thru ~Product
    ~HcalRecoParamsGPU() = default;

    // get device pointers
    Product const& getProduct(cuda::stream_t<>&) const;

    // 
    static std::string name() { return std::string{"hcalRecoParamsGPU"}; }

private:
    uint64_t totalChannels_; // hb + he
    std::vector<uint32_t, CUDAHostAllocator<uint32_t>> param1_;
    std::vector<uint32_t, CUDAHostAllocator<uint32_t>> param2_;

    CUDAESProduct<Product> product_;
#endif
};

#endif
