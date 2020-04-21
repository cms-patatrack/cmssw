#include "RecoLocalCalo/EcalRecAlgos/interface/EcalChannelStatusGPU.h"

#include "FWCore/Utilities/interface/typelookup.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

EcalChannelStatusGPU::EcalChannelStatusGPU(EcalChannelStatus const& values) 
: status_(values.size())
{
  // fill in eb
  auto const& barrelValues = values.barrelItems();
  for (unsigned int i=0; i<barrelValues.size(); i++) {
    status_[i] = barrelValues[i].getEncodedStatusCode();
  }
  
  // fill in ee
  auto const& endcapValues = values.endcapItems();
  auto const offset = barrelValues.size();
  for (unsigned int i=0; i<endcapValues.size(); i++) {
    status_[offset + i] = endcapValues[i].getEncodedStatusCode();
  }
}

EcalChannelStatusGPU::Product::~Product() {
  // deallocation
  cudaCheck( cudaFree(status) );
}

EcalChannelStatusGPU::Product const& EcalChannelStatusGPU::getProduct(cudaStream_t cudaStream) const { 
  auto const& product = product_.dataForCurrentDeviceAsync(
    cudaStream,
    [this](EcalChannelStatusGPU::Product& product, cudaStream_t cudaStream) {
      // malloc
      cudaCheck( cudaMalloc((void**)&product.status,
                            this->status_.size() * sizeof(uint16_t)) );
      // transfer 
      cudaCheck( cudaMemcpyAsync(product.status,
                                 this->status_.data(),
                                 this->status_.size() * sizeof(uint16_t),
                                 cudaMemcpyHostToDevice,
                                 cudaStream) );
    }
  );
  
  return product;
}

TYPELOOKUP_DATA_REG(EcalChannelStatusGPU);


