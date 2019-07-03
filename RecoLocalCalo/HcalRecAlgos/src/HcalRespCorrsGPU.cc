#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRespCorrsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalRespCorrsGPU::HcalRespCorrsGPU(HcalRespCorrs const& respcorrs) 
    : value_(respcorrs.getAllContainers()[0].second.size()
        + respcorrs.getAllContainers()[1].second.size())
{
    // fill in eb
    auto const& barrelValues = respcorrs.getAllContainers()[0].second;
    for (uint64_t i=0; i<barrelValues.size(); ++i) {
        value_[i] = barrelValues[i].getValue();
    }

    // fill in ee
    auto const& endcapValues = respcorrs.getAllContainers()[1].second;
    auto const offset = barrelValues.size();
    for (uint64_t i=0; i<endcapValues.size(); ++i) {
        value_[i + offset] = endcapValues[i].getValue();
    }
}

HcalRespCorrsGPU::Product::~Product() {
    // deallocation
    cudaCheck( cudaFree(value) );
}

HcalRespCorrsGPU::Product const& HcalRespCorrsGPU::getProduct(
        cuda::stream_t<>& cudaStream) const {
    auto const& product = product_.dataForCurrentDeviceAsync(cudaStream,
        [this](HcalRespCorrsGPU::Product& product, cuda::stream_t<>& cudaStream){
            // malloc
            cudaCheck( cudaMalloc((void**)&product.value, 
                this->value_.size() * sizeof(float)) );
            
            // transfer
            cudaCheck( cudaMemcpyAsync(product.value, 
                                       this->value_.data(),
                                       this->value_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
        });

    return product;
}

TYPELOOKUP_DATA_REG(HcalRespCorrsGPU);
