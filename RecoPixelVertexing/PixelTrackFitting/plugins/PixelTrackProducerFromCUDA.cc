#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

/**
 * This class will eventually be the one creating the reco::Track
 * objects from the output of GPU CA. Now it is just to produce
 * something persistable.
 */
class PixelTrackProducerFromCUDA: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
 public:

  using Output = HeterogeneousProductImpl<heterogeneous::CPUProduct<int>,
                                                                   heterogeneous::GPUCudaProduct<int> >;

  explicit PixelTrackProducerFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelTrackProducerFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginStreamGPUCuda(edm::StreamID streamId,
                          cuda::stream_t<> &cudaStream) override {
  }
  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override {}
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent,
                  const edm::EventSetup &iSetup) override;


 private:
  edm::EDGetTokenT<HeterogeneousProduct> gpuToken_;
  edm::EDGetTokenT<RegionsSeedingHitSets> srcToken_;
  bool enabled_;
};

PixelTrackProducerFromCUDA::PixelTrackProducerFromCUDA(const edm::ParameterSet& iConfig):
  HeterogeneousEDProducer(iConfig),
  gpuToken_(consumes<HeterogeneousProduct>(iConfig.getParameter<edm::InputTag>("src")))
//  srcToken_(consumes<RegionsSeedingHitSets>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<int>();
//  produces<HeterogeneousProduct>();
}

void PixelTrackProducerFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksHitQuadruplets"));

  HeterogeneousEDProducer::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

void PixelTrackProducerFromCUDA::produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) {
  iEvent.put(std::make_unique<int>(0));
}


void PixelTrackProducerFromCUDA::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup)
{
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

DEFINE_FWK_MODULE(PixelTrackProducerFromCUDA);
