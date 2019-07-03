#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalsGPU.h"

#include "CondFormats/HcalObjects/interface/HcalPedestals.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// FIXME: add proper getters to conditions
HcalPedestalsGPU::HcalPedestalsGPU(HcalPedestals const& pedestals) 
    : unitIsADC_{pedestals.isADC()}
    , totalChannels_{pedestals.getAllContainers()[0].second.size()
        + pedestals.getAllContainers()[1].second.size()}
    , value0_(totalChannels_)
    , value1_(totalChannels_)
    , value2_(totalChannels_)
    , value3_(totalChannels_)
    , width0_(totalChannels_)
    , width1_(totalChannels_)
    , width2_(totalChannels_)
    , width3_(totalChannels_)
{
    // fill in eb
    auto const& barrelValues = pedestals.getAllContainers()[0].second;
    for (uint64_t i=0; i<barrelValues.size(); ++i) {
        value0_[i] = barrelValues[i].getValue(0);
        value1_[i] = barrelValues[i].getValue(1);
        value2_[i] = barrelValues[i].getValue(2);
        value3_[i] = barrelValues[i].getValue(3);

        width0_[i] = barrelValues[i].getWidth(0);
        width1_[i] = barrelValues[i].getWidth(1);
        width2_[i] = barrelValues[i].getWidth(2);
        width3_[i] = barrelValues[i].getWidth(3);
    }

    // fill in ee
    auto const& endcapValues = pedestals.getAllContainers()[1].second;
    auto const offset = barrelValues.size();
    for (uint64_t i=0; i<endcapValues.size(); ++i) {
        value0_[i + offset] = endcapValues[i].getValue(0);
        value1_[i + offset] = endcapValues[i].getValue(1);
        value2_[i + offset] = endcapValues[i].getValue(2);
        value3_[i + offset] = endcapValues[i].getValue(3);

        width0_[i + offset] = endcapValues[i].getWidth(0);
        width1_[i + offset] = endcapValues[i].getWidth(1);
        width2_[i + offset] = endcapValues[i].getWidth(2);
        width3_[i + offset] = endcapValues[i].getWidth(3);
    }
}

HcalPedestalsGPU::Product::~Product() {
    // deallocation
    cudaCheck( cudaFree(value0) );
    cudaCheck( cudaFree(value1) );
    cudaCheck( cudaFree(value2) );
    cudaCheck( cudaFree(value3) );
    
    cudaCheck( cudaFree(width0) );
    cudaCheck( cudaFree(width1) );
    cudaCheck( cudaFree(width2) );
    cudaCheck( cudaFree(width3) );
}

HcalPedestalsGPU::Product const& HcalPedestalsGPU::getProduct(
        cuda::stream_t<>& cudaStream) const {
    auto const& product = product_.dataForCurrentDeviceAsync(cudaStream,
        [this](HcalPedestalsGPU::Product& product, cuda::stream_t<>& cudaStream){
            // malloc
            cudaCheck( cudaMalloc((void**)&product.value0, 
                this->value0_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.value1, 
                this->value1_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.value2, 
                this->value2_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.value3, 
                this->value3_.size() * sizeof(float)) );
            
            cudaCheck( cudaMalloc((void**)&product.width0, 
                this->width0_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.width1, 
                this->width1_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.width2, 
                this->width2_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.width3, 
                this->width3_.size() * sizeof(float)) );

            // transfer
            cudaCheck( cudaMemcpyAsync(product.value0, 
                                       this->value0_.data(),
                                       this->value0_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.value1, 
                                       this->value1_.data(),
                                       this->value1_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.value2, 
                                       this->value2_.data(),
                                       this->value2_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.value3, 
                                       this->value3_.data(),
                                       this->value3_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            cudaCheck( cudaMemcpyAsync(product.width0, 
                                       this->width0_.data(),
                                       this->width0_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.width1,
                                       this->width1_.data(),
                                       this->width1_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.width2, 
                                       this->width2_.data(),
                                       this->width2_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.width3, 
                                       this->width3_.data(),
                                       this->width3_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
        });

    return product;
}

TYPELOOKUP_DATA_REG(HcalPedestalsGPU);
