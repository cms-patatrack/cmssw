#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include "CUDADataFormats/Vertex/interface/ZVertexCUDA.h"
#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"


class PixelTrackDumpCUDA : public edm::global::EDAnalyzer<> {
public:
  explicit PixelTrackDumpCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackDumpCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const & iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  edm::EDGetTokenT<CUDAProduct<PixelTrackCUDA>> tokenGPUTrack_;
  edm::EDGetTokenT<CUDAProduct<ZVertexCUDA>> tokenGPUVertex_;
  edm::EDGetTokenT<PixelTrackCUDA::SoA> tokenSoATrack_;
  edm::EDGetTokenT<ZVertexCUDA::SoA> tokenSoAVertex_;


};

PixelTrackDumpCUDA::PixelTrackDumpCUDA(const edm::ParameterSet& iConfig) :
  m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
    tokenGPUTrack_ = consumes<CUDAProduct<PixelTrackCUDA>>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenGPUVertex_ = consumes<CUDAProduct<ZVertexCUDA>>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
  } else {
    tokenSoATrack_ = consumes<PixelTrackCUDA::SoA>(iConfig.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenSoAVertex_ = consumes<ZVertexCUDA::SoA>(iConfig.getParameter<edm::InputTag>("pixelVertexSrc"));
  }
}

void PixelTrackDumpCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

   desc.add<bool>("onGPU",true);
   desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("caHitNtupletCUDA"));
   desc.add<edm::InputTag>("pixelVertexSrc", edm::InputTag("pixelVertexCUDA"));
   descriptions.add("pixelTrackDumpCUDA", desc);
}

void PixelTrackDumpCUDA::analyze(edm::StreamID streamID, edm::Event const & iEvent, const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    edm::Handle<CUDAProduct<PixelTrackCUDA>>  hTracks;
    iEvent.getByToken(tokenGPUTrack_, hTracks);

    CUDAScopedContextProduce ctx{*hTracks};
    auto const& tracks = ctx.get(*hTracks);

    auto const * tsoa = tracks.soa();
    assert(tsoa);

    edm::Handle<CUDAProduct<ZVertexCUDA>>  hVertices;
    iEvent.getByToken(tokenGPUVertex_, hVertices);

    auto const& vertices = ctx.get(*hVertices);

    auto const * vsoa = vertices.soa();
    assert(vsoa);

  } else {
    edm::Handle<PixelTrackCUDA::SoA>  hTracks;
    iEvent.getByToken(tokenSoATrack_, hTracks);

    auto const * tsoa = hTracks.product();
    assert(tsoa);

    edm::Handle<ZVertexCUDA::SoA>  hVertices;
    iEvent.getByToken(tokenSoAVertex_, hVertices);

    auto const * vsoa = hVertices.product();
    assert(vsoa);

  }

}


DEFINE_FWK_MODULE(PixelTrackDumpCUDA);

