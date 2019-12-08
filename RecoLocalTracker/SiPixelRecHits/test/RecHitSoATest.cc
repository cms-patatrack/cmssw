#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/CUDAProduct.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"

template<typename Traits>
void analyzeImpl(TrackingRecHit2DHeterogeneous<Traits> const & ghits, cudaStream_t stream);

/*
TrackingRecHit2DCUDA and TrackingRecHit2DCPU are NOT the same type (for tracks and vertices are the same type)
they are templated with the (GPU/CPU)Traits so in principle the whole Analyzer can be (partially) templated as well
they return the same view type though (so the real analyzer algo in the header file above does not need to be templated)
*/

class RecHitSoATest : public edm::global::EDAnalyzer<> {
public:

  explicit RecHitSoATest(const edm::ParameterSet& iConfig);
  ~RecHitSoATest() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const override;
  const bool m_onGPU;
  edm::EDGetTokenT<CUDAProduct<TrackingRecHit2DCUDA>> tGpuHits;
  edm::EDGetTokenT<TrackingRecHit2DCPU> tCpuHits;
};

RecHitSoATest::RecHitSoATest(const edm::ParameterSet& iConfig) : m_onGPU(iConfig.getParameter<bool>("onGPU")) {
  if (m_onGPU) {
      tGpuHits = 
          consumes<CUDAProduct<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"));
  } else {
      tCpuHits =
          consumes<TrackingRecHit2DCPU>(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"));
  }
}


void RecHitSoATest::analyze(edm::StreamID streamID, edm::Event const& iEvent, const edm::EventSetup& iSetup) const {
  if (m_onGPU) {
    auto const & gh = iEvent.get(tGpuHits);
    CUDAScopedContextProduce ctx{gh};

    analyzeImpl<cudaCompat::GPUTraits>(ctx.get(gh),ctx.stream());

  } else {
    analyzeImpl<cudaCompat::CPUTraits>(iEvent.get(tCpuHits),cudaStreamDefault);
  }
}

void RecHitSoATest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("onGPU", true);
  desc.add<edm::InputTag>("heterogeneousPixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  descriptions.add("RecHitSoATest", desc);
}

DEFINE_FWK_MODULE(RecHitSoATest);
