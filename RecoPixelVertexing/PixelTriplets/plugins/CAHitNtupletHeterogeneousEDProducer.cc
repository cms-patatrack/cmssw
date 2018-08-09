#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/RegionsSeedingHitSets.h"

#include "CAHitQuadrupletGeneratorGPU.h"

namespace {
void fillNtuplets(RegionsSeedingHitSets::RegionFiller &seedingHitSetsFiller,
                  const OrderedHitSeeds &quadruplets) {
  for (const auto &quad : quadruplets) {
    seedingHitSetsFiller.emplace_back(quad[0], quad[1], quad[2], quad[3]);
  }
}
} // namespace

class CAHitNtupletHeterogeneousEDProducer
    : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
public:

  using PixelRecHitsH = siPixelRecHitsHeterogeneousProduct::HeterogeneousPixelRecHit;


  CAHitNtupletHeterogeneousEDProducer(const edm::ParameterSet &iConfig);
  ~CAHitNtupletHeterogeneousEDProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void beginStreamGPUCuda(edm::StreamID streamId,
                          cuda::stream_t<> &cudaStream) override;
  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent,
                  const edm::EventSetup &iSetup) override;

private:
  edm::EDGetTokenT<edm::OwnVector<TrackingRegion> > regionToken_;

  edm::EDGetTokenT<HeterogeneousProduct> tGpuHits;

  edm::RunningAverage localRA_;
  CAHitQuadrupletGeneratorGPU GPUGenerator_;

  bool emptyRegions = false;
  std::unique_ptr<RegionsSeedingHitSets> seedingHitSets_;
};

CAHitNtupletHeterogeneousEDProducer::CAHitNtupletHeterogeneousEDProducer(
    const edm::ParameterSet &iConfig)
    : HeterogeneousEDProducer(iConfig),
      regionToken_(consumes<edm::OwnVector<TrackingRegion>>(
          iConfig.getParameter<edm::InputTag>("trackingRegions"))),
      tGpuHits(consumesHeterogeneous(iConfig.getParameter<edm::InputTag>("heterogeneousPixelRecHitSrc"))),
      GPUGenerator_(iConfig, consumesCollector()) {
  produces<RegionsSeedingHitSets>();
}

void CAHitNtupletHeterogeneousEDProducer::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("doublets", edm::InputTag("dummy"))->setComment("Not really used, kept to keep the python parameters");
  desc.add<edm::InputTag>("trackingRegions", edm::InputTag("globalTrackingRegionFromBeamSpot"));

  desc.add<edm::InputTag>("heterogeneousPixelRecHitSrc", edm::InputTag("siPixelRecHitHeterogeneous"));

  CAHitQuadrupletGeneratorGPU::fillDescriptions(desc);
  HeterogeneousEDProducer::fillPSetDescription(desc);
  auto label = "caHitQuadrupletHeterogeneousEDProducer";
  descriptions.add(label, desc);
}

void CAHitNtupletHeterogeneousEDProducer::beginStreamGPUCuda(
    edm::StreamID streamId, cuda::stream_t<> &cudaStream) {
  GPUGenerator_.allocateOnGPU();
}

void CAHitNtupletHeterogeneousEDProducer::acquireGPUCuda(
    const edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup,
    cuda::stream_t<> &cudaStream) {

  seedingHitSets_ = std::make_unique<RegionsSeedingHitSets>();

  edm::Handle<edm::OwnVector<TrackingRegion>> hregions;
  iEvent.getByToken(regionToken_, hregions);
  const auto &regions = *hregions;
  assert(regions.size()<=1);

  if (regions.empty()) {
    emptyRegions = true;
    return;
  }

  const TrackingRegion &region = regions[0];


  edm::Handle<siPixelRecHitsHeterogeneousProduct::GPUProduct> gh;
  iEvent.getByToken<siPixelRecHitsHeterogeneousProduct::HeterogeneousPixelRecHit>(tGpuHits, gh);
  auto const & gHits = *gh;
//  auto nhits = gHits.nHits;

  // move inside hitNtuplets???
  GPUGenerator_.buildDoublets(gHits,cudaStream.id());

  seedingHitSets_->reserve(regions.size(), localRA_.upper());
  GPUGenerator_.initEvent(iEvent.event(), iSetup);

  LogDebug("CAHitNtupletHeterogeneousEDProducer")
        << "Creating ntuplets for " << regions.size()
        << " regions";

  GPUGenerator_.hitNtuplets(region, gHits, iSetup, cudaStream.id());
  
}

void CAHitNtupletHeterogeneousEDProducer::produceGPUCuda(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup,
    cuda::stream_t<> &cudaStream) {

  if (not emptyRegions) {
    edm::Handle<edm::OwnVector<TrackingRegion>> hregions;
    iEvent.getByToken(regionToken_, hregions);
    const auto &regions = *hregions;

    edm::Handle<HeterogeneousProduct> gh;
    iEvent.getByToken(tGpuHits, gh);
    auto const & rechits = gh->get<siPixelRecHitsHeterogeneousProduct::HeterogeneousPixelRecHit>().getProduct<HeterogeneousDevice::kCPU>();

    std::vector<OrderedHitSeeds> ntuplets(regions.size());
    for (auto &ntuplet : ntuplets) ntuplet.reserve(localRA_.upper());
    int index = 0;
    for (const auto &region : regions) {
      auto seedingHitSetsFiller = seedingHitSets_->beginRegion(&region);
      GPUGenerator_.fillResults(region, rechits.collection, ntuplets, iSetup, cudaStream.id());
      fillNtuplets(seedingHitSetsFiller, ntuplets[index]);
      ntuplets[index].clear();
      index++;
    }
    localRA_.update(seedingHitSets_->size());
  }
  iEvent.put(std::move(seedingHitSets_));
}

void CAHitNtupletHeterogeneousEDProducer::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup) {
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

DEFINE_FWK_MODULE(CAHitNtupletHeterogeneousEDProducer);
