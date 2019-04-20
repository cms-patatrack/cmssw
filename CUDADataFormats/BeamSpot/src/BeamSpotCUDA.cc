#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

BeamSpotCUDA::BeamSpotCUDA(Data const* data_h, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  assert(std::abs(data_h->x)<1.f);
  auto data_l =  cs->make_host_unique<Data>(stream);
  memcpy(data_l.get(),data_h,sizeof(Data));
  assert(std::abs(data_l->x)<1.f);
  data_d_ = cs->make_device_unique<Data>(stream);
  cuda::memory::async::copy(data_d_.get(), data_l.get(), sizeof(Data), stream.id());
  cudaStreamSynchronize(stream.id());
}
