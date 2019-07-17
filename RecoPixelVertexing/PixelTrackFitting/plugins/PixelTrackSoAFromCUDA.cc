#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"


#include "CUDADataFormats/Track/interface/PixelTrackHeterogeneous.h"

class PixelTrackSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;


  edm::EDGetTokenT<CUDAProduct<PixelTrackHeterogeneous>> tokenCUDA_;
  edm::EDPutTokenT<PixelTrackHeterogeneous> tokenSOA_;

  cudautils::host::unique_ptr<pixelTrack::TrackSoA> m_soa;

};

PixelTrackSoAFromCUDA::PixelTrackSoAFromCUDA(const edm::ParameterSet& iConfig) :
  tokenCUDA_(consumes<CUDAProduct<PixelTrackHeterogeneous>>(iConfig.getParameter<edm::InputTag>("src"))),
  tokenSOA_(produces<PixelTrackHeterogeneous>())
{}


void PixelTrackSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

   desc.add<edm::InputTag>("src", edm::InputTag("caHitNtupletCUDA"));
   descriptions.add("pixelTrackSoA", desc);

}


void PixelTrackSoAFromCUDA::acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  CUDAProduct<PixelTrackHeterogeneous> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  CUDAScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_soa = inputData.toHostAsync(ctx.stream());

}

void PixelTrackSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {

  /*
  auto tsoa = *m_soa;
  auto maxTracks = tdoa.stride();
  std::cout << "size of SoA" << sizeof(PixelTrackHeterogeneous::SoA) << " stride " << maxTracks << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = tsoa.nHits(it);
    assert(nHits==int(tsoa.hitIndices.size(it)));
    if (nHits == 0) break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  std::cout << "found " << nt << " tracks in cpu SoA" << std::endl;
  */

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_,PixelTrackHeterogeneous(std::move(m_soa)));

}


DEFINE_FWK_MODULE(PixelTrackSoAFromCUDA);
