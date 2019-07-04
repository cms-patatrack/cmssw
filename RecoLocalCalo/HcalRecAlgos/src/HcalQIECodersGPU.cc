#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIECodersGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

HcalQIECodersGPU::HcalQIECodersGPU(HcalQIEData const& qiedata) 
    : totalChannels_{qiedata.getAllContainers()[0].second.size()
        + qiedata.getAllContainers()[1].second.size()}
    , offset00_(totalChannels_)
    , offset01_(totalChannels_)
    , offset02_(totalChannels_)
    , offset03_(totalChannels_)
    , offset10_(totalChannels_)
    , offset11_(totalChannels_)
    , offset12_(totalChannels_)
    , offset13_(totalChannels_)
    , offset20_(totalChannels_)
    , offset21_(totalChannels_)
    , offset22_(totalChannels_)
    , offset23_(totalChannels_)
    , offset30_(totalChannels_)
    , offset31_(totalChannels_)
    , offset32_(totalChannels_)
    , offset33_(totalChannels_)
    , slope00_(totalChannels_)
    , slope01_(totalChannels_)
    , slope02_(totalChannels_)
    , slope03_(totalChannels_)
    , slope10_(totalChannels_)
    , slope11_(totalChannels_)
    , slope12_(totalChannels_)
    , slope13_(totalChannels_)
    , slope20_(totalChannels_)
    , slope21_(totalChannels_)
    , slope22_(totalChannels_)
    , slope23_(totalChannels_)
    , slope30_(totalChannels_)
    , slope31_(totalChannels_)
    , slope32_(totalChannels_)
    , slope33_(totalChannels_)
{
    // fill in hb
    auto const& barrelValues = qiedata.getAllContainers()[0].second;
    for (uint64_t i=0; i<barrelValues.size(); ++i) {
        offset00_[i] = barrelValues[i].offset(0, 0);
        offset01_[i] = barrelValues[i].offset(0, 1);
        offset02_[i] = barrelValues[i].offset(0, 2);
        offset03_[i] = barrelValues[i].offset(0, 3);
        offset10_[i] = barrelValues[i].offset(1, 0);
        offset11_[i] = barrelValues[i].offset(1, 1);
        offset12_[i] = barrelValues[i].offset(1, 2);
        offset13_[i] = barrelValues[i].offset(1, 3);
        offset20_[i] = barrelValues[i].offset(2, 0);
        offset21_[i] = barrelValues[i].offset(2, 1);
        offset22_[i] = barrelValues[i].offset(2, 2);
        offset23_[i] = barrelValues[i].offset(2, 3);
        offset30_[i] = barrelValues[i].offset(3, 0);
        offset31_[i] = barrelValues[i].offset(3, 1);
        offset32_[i] = barrelValues[i].offset(3, 2);
        offset33_[i] = barrelValues[i].offset(3, 3);
        
        slope00_[i] = barrelValues[i].slope(0, 0);
        slope01_[i] = barrelValues[i].slope(0, 1);
        slope02_[i] = barrelValues[i].slope(0, 2);
        slope03_[i] = barrelValues[i].slope(0, 3);
        slope10_[i] = barrelValues[i].slope(1, 0);
        slope11_[i] = barrelValues[i].slope(1, 1);
        slope12_[i] = barrelValues[i].slope(1, 2);
        slope13_[i] = barrelValues[i].slope(1, 3);
        slope20_[i] = barrelValues[i].slope(2, 0);
        slope21_[i] = barrelValues[i].slope(2, 1);
        slope22_[i] = barrelValues[i].slope(2, 2);
        slope23_[i] = barrelValues[i].slope(2, 3);
        slope30_[i] = barrelValues[i].slope(3, 0);
        slope31_[i] = barrelValues[i].slope(3, 1);
        slope32_[i] = barrelValues[i].slope(3, 2);
        slope33_[i] = barrelValues[i].slope(3, 3);
    }

    // fill in he
    auto const& endcapValues = qiedata.getAllContainers()[1].second;
    auto const offset = barrelValues.size();
    for (uint64_t i=0; i<endcapValues.size(); ++i) {
        offset00_[i + offset] = endcapValues[i].offset(0, 0);
        offset01_[i + offset] = endcapValues[i].offset(0, 1);
        offset02_[i + offset] = endcapValues[i].offset(0, 2);
        offset03_[i + offset] = endcapValues[i].offset(0, 3);
        offset10_[i + offset] = endcapValues[i].offset(1, 0);
        offset11_[i + offset] = endcapValues[i].offset(1, 1);
        offset12_[i + offset] = endcapValues[i].offset(1, 2);
        offset13_[i + offset] = endcapValues[i].offset(1, 3);
        offset20_[i + offset] = endcapValues[i].offset(2, 0);
        offset21_[i + offset] = endcapValues[i].offset(2, 1);
        offset22_[i + offset] = endcapValues[i].offset(2, 2);
        offset23_[i + offset] = endcapValues[i].offset(2, 3);
        offset30_[i + offset] = endcapValues[i].offset(3, 0);
        offset31_[i + offset] = endcapValues[i].offset(3, 1);
        offset32_[i + offset] = endcapValues[i].offset(3, 2);
        offset33_[i + offset] = endcapValues[i].offset(3, 3);
        
        slope00_[i + offset] = endcapValues[i].slope(0, 0);
        slope01_[i + offset] = endcapValues[i].slope(0, 1);
        slope02_[i + offset] = endcapValues[i].slope(0, 2);
        slope03_[i + offset] = endcapValues[i].slope(0, 3);
        slope10_[i + offset] = endcapValues[i].slope(1, 0);
        slope11_[i + offset] = endcapValues[i].slope(1, 1);
        slope12_[i + offset] = endcapValues[i].slope(1, 2);
        slope13_[i + offset] = endcapValues[i].slope(1, 3);
        slope20_[i + offset] = endcapValues[i].slope(2, 0);
        slope21_[i + offset] = endcapValues[i].slope(2, 1);
        slope22_[i + offset] = endcapValues[i].slope(2, 2);
        slope23_[i + offset] = endcapValues[i].slope(2, 3);
        slope30_[i + offset] = endcapValues[i].slope(3, 0);
        slope31_[i + offset] = endcapValues[i].slope(3, 1);
        slope32_[i + offset] = endcapValues[i].slope(3, 2);
        slope33_[i + offset] = endcapValues[i].slope(3, 3);
    }
}

HcalQIECodersGPU::Product::~Product() {
    // deallocation
    cudaCheck( cudaFree(offset00) );
    cudaCheck( cudaFree(offset01) );
    cudaCheck( cudaFree(offset02) );
    cudaCheck( cudaFree(offset03) );
    cudaCheck( cudaFree(offset10) );
    cudaCheck( cudaFree(offset11) );
    cudaCheck( cudaFree(offset12) );
    cudaCheck( cudaFree(offset13) );
    cudaCheck( cudaFree(offset20) );
    cudaCheck( cudaFree(offset21) );
    cudaCheck( cudaFree(offset22) );
    cudaCheck( cudaFree(offset23) );
    cudaCheck( cudaFree(offset30) );
    cudaCheck( cudaFree(offset31) );
    cudaCheck( cudaFree(offset32) );
    cudaCheck( cudaFree(offset33) );
    
    cudaCheck( cudaFree(slope00) );
    cudaCheck( cudaFree(slope01) );
    cudaCheck( cudaFree(slope02) );
    cudaCheck( cudaFree(slope03) );
    cudaCheck( cudaFree(slope10) );
    cudaCheck( cudaFree(slope11) );
    cudaCheck( cudaFree(slope12) );
    cudaCheck( cudaFree(slope13) );
    cudaCheck( cudaFree(slope20) );
    cudaCheck( cudaFree(slope21) );
    cudaCheck( cudaFree(slope22) );
    cudaCheck( cudaFree(slope23) );
    cudaCheck( cudaFree(slope30) );
    cudaCheck( cudaFree(slope31) );
    cudaCheck( cudaFree(slope32) );
    cudaCheck( cudaFree(slope33) );
}

HcalQIECodersGPU::Product const& HcalQIECodersGPU::getProduct(
        cuda::stream_t<>& cudaStream) const {
    auto const& product = product_.dataForCurrentDeviceAsync(cudaStream,
        [this](HcalQIECodersGPU::Product& product, cuda::stream_t<>& cudaStream){
            // malloc
            cudaCheck( cudaMalloc((void**)&product.offset00,
                this->offset00_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset01,
                this->offset01_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset02,
                this->offset02_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset03,
                this->offset03_.size() * sizeof(float)) );

            cudaCheck( cudaMalloc((void**)&product.offset10,
                this->offset10_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset11,
                this->offset11_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset12,
                this->offset12_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset13,
                this->offset13_.size() * sizeof(float)) );
            
            cudaCheck( cudaMalloc((void**)&product.offset20,
                this->offset20_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset21,
                this->offset21_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset22,
                this->offset22_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset23,
                this->offset23_.size() * sizeof(float)) );
            
            cudaCheck( cudaMalloc((void**)&product.offset30,
                this->offset30_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset31,
                this->offset31_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset32,
                this->offset32_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.offset33,
                this->offset33_.size() * sizeof(float)) );
            
            cudaCheck( cudaMalloc((void**)&product.slope00,
                this->slope00_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope01,
                this->slope01_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope02,
                this->slope02_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope03,
                this->slope03_.size() * sizeof(float)) );

            cudaCheck( cudaMalloc((void**)&product.slope10,
                this->slope10_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope11,
                this->slope11_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope12,
                this->slope12_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope13,
                this->slope13_.size() * sizeof(float)) );
            
            cudaCheck( cudaMalloc((void**)&product.slope20,
                this->slope20_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope21,
                this->slope21_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope22,
                this->slope22_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope23,
                this->slope23_.size() * sizeof(float)) );
            
            cudaCheck( cudaMalloc((void**)&product.slope30,
                this->slope30_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope31,
                this->slope31_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope32,
                this->slope32_.size() * sizeof(float)) );
            cudaCheck( cudaMalloc((void**)&product.slope33,
                this->slope33_.size() * sizeof(float)) );

            // transfer
            // offset
            cudaCheck( cudaMemcpyAsync(product.offset00, 
                                       this->offset00_.data(),
                                       this->offset00_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset01, 
                                       this->offset01_.data(),
                                       this->offset01_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset02, 
                                       this->offset02_.data(),
                                       this->offset02_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset03, 
                                       this->offset03_.data(),
                                       this->offset03_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            cudaCheck( cudaMemcpyAsync(product.offset10, 
                                       this->offset10_.data(),
                                       this->offset10_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset11, 
                                       this->offset11_.data(),
                                       this->offset11_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset12, 
                                       this->offset12_.data(),
                                       this->offset12_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset13, 
                                       this->offset13_.data(),
                                       this->offset13_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            cudaCheck( cudaMemcpyAsync(product.offset20, 
                                       this->offset20_.data(),
                                       this->offset20_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset21, 
                                       this->offset21_.data(),
                                       this->offset21_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset22, 
                                       this->offset22_.data(),
                                       this->offset22_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset23, 
                                       this->offset23_.data(),
                                       this->offset23_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            cudaCheck( cudaMemcpyAsync(product.offset30, 
                                       this->offset30_.data(),
                                       this->offset30_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset31, 
                                       this->offset31_.data(),
                                       this->offset31_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset32, 
                                       this->offset32_.data(),
                                       this->offset32_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.offset33, 
                                       this->offset33_.data(),
                                       this->offset33_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            // slope
            cudaCheck( cudaMemcpyAsync(product.slope00, 
                                       this->slope00_.data(),
                                       this->slope00_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope01, 
                                       this->slope01_.data(),
                                       this->slope01_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope02, 
                                       this->slope02_.data(),
                                       this->slope02_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope03, 
                                       this->slope03_.data(),
                                       this->slope03_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            cudaCheck( cudaMemcpyAsync(product.slope10, 
                                       this->slope10_.data(),
                                       this->slope10_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope11, 
                                       this->slope11_.data(),
                                       this->slope11_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope12, 
                                       this->slope12_.data(),
                                       this->slope12_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope13, 
                                       this->slope13_.data(),
                                       this->slope13_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            cudaCheck( cudaMemcpyAsync(product.slope20, 
                                       this->slope20_.data(),
                                       this->slope20_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope21, 
                                       this->slope21_.data(),
                                       this->slope21_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope22, 
                                       this->slope22_.data(),
                                       this->slope22_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope23, 
                                       this->slope23_.data(),
                                       this->slope23_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            
            cudaCheck( cudaMemcpyAsync(product.slope30, 
                                       this->slope30_.data(),
                                       this->slope30_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope31, 
                                       this->slope31_.data(),
                                       this->slope31_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope32, 
                                       this->slope32_.data(),
                                       this->slope32_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
            cudaCheck( cudaMemcpyAsync(product.slope33, 
                                       this->slope33_.data(),
                                       this->slope33_.size() * sizeof(float),
                                       cudaMemcpyHostToDevice,
                                       cudaStream.id()) );
        });

    return product;
}

TYPELOOKUP_DATA_REG(HcalQIECodersGPU);
