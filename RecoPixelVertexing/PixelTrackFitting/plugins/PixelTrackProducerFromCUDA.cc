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

// track stuff
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"
#include "storeTracks.h"


/**
 * This class will eventually be the one creating the reco::Track
 * objects from the output of GPU CA. Now it is just to produce
 * something persistable.
 */
class PixelTrackProducerFromCUDA: public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
 public:

  using Input = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;
  using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;


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
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent,
                  const edm::EventSetup &iSetup) override;


 private:

  TuplesOnCPU const * tuples_=nullptr;

  edm::EDGetTokenT<HeterogeneousProduct> gpuToken_;
  edm::EDGetTokenT<RegionsSeedingHitSets> srcToken_;
  bool enableConversion_;
};

PixelTrackProducerFromCUDA::PixelTrackProducerFromCUDA(const edm::ParameterSet& iConfig):
  HeterogeneousEDProducer(iConfig),
  gpuToken_(consumes<HeterogeneousProduct>(iConfig.getParameter<edm::InputTag>("src"))),
  enableConversion_ (iConfig.getParameter<bool>("gpuEnableConversion"))
{
  if (enableConversion_) {
    srcToken_ = consumes<RegionsSeedingHitSets>(iConfig.getParameter<edm::InputTag>("src"));
    produces<reco::TrackCollection>();
    produces<TrackingRecHitCollection>();
    produces<reco::TrackExtraCollection>();
  }
  produces<int>();  // dummy
//  produces<HeterogeneousProduct>();
}

void PixelTrackProducerFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksHitQuadruplets"));
  desc.add<bool>("gpuEnableConversion", true);


  HeterogeneousEDProducer::fillPSetDescription(desc);
  descriptions.addWithDefaultLabel(desc);
}

void  PixelTrackProducerFromCUDA::acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) {

  edm::Handle<TuplesOnCPU> gh;
  iEvent.getByToken<Input>(gpuToken_, gh);
  auto const & gTuples = *gh;
  std::cout << "tuples from gpu " << gTuples.nTuples << std::endl;


  tuples_ = gh.product();

}


void PixelTrackProducerFromCUDA::produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) {
  iEvent.put(std::make_unique<int>(0));
  if (!enableConversion_) return;

  std::cout << "Converting gpu helix in reco tracks" << std::endl;

  pixeltrackfitting::TracksWithTTRHs tracks;
  edm::ESHandle<TrackerTopology> httopo;
  iSetup.get<TrackerTopologyRcd>().get(httopo);


  edm::Handle<RegionsSeedingHitSets> hhitSets;
  iEvent.getByToken(srcToken_, hhitSets);
  const auto & hitSet =  *hhitSets->begin();
  auto b = hitSet.begin();  auto e = hitSet.end(); 
  std::cout << "reading hitset " << e-b << std::endl;


  // store tracks
  storeTracks(iEvent, tracks, *httopo);
}


void PixelTrackProducerFromCUDA::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup)
{
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

DEFINE_FWK_MODULE(PixelTrackProducerFromCUDA);
