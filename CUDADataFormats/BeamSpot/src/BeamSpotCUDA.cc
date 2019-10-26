#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

BeamSpotCUDA::BeamSpotCUDA(Data const* data_h, cudaStream_t stream) {
  data_d_ = cudautils::make_device_unique<Data>(stream);
  cudaMemcpyAsync(data_d_.get(), data_h, sizeof(Data), cudaMemcpyHostToDevice, stream);
}
