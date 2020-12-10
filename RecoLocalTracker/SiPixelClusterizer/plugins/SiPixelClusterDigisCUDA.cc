#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"
#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"
#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "SiPixelRawToClusterGPUKernel.h"

#include <memory>
#include <string>
#include <vector>

class SiPixelClusterDigisCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelClusterDigisCUDA(const edm::ParameterSet& iConfig);
  ~SiPixelClusterDigisCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:


  void acquire(const edm::Event& iEvent,
               const edm::EventSetup& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> pixelDigiToken_;

  edm::EDPutTokenT<cms::cuda::Product<SiPixelDigisCUDA>> digiPutToken_;
  edm::EDPutTokenT<cms::cuda::Product<SiPixelDigiErrorsCUDA>> digiErrorPutToken_;
  edm::EDPutTokenT<cms::cuda::Product<SiPixelClustersCUDA>> clusterPutToken_;

  cms::cuda::ContextState ctxState_;

  pixelgpudetails::SiPixelRawToClusterGPUKernel gpuAlgo_;
  PixelDataFormatter::Errors errors_;

};

SiPixelClusterDigisCUDA::SiPixelClusterDigisCUDA(const edm::ParameterSet& iConfig)
    :
      pixelDigiToken_(consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("InputLabel"))),
      digiPutToken_(produces<cms::cuda::Product<SiPixelDigisCUDA>>()),
      clusterPutToken_(produces<cms::cuda::Product<SiPixelClustersCUDA>>())
{

}

void SiPixelClusterDigisCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("simSiPixelDigis:Pixel"));
  descriptions.addWithDefaultLabel(desc);
}


void SiPixelClusterDigisCUDA::acquire(const edm::Event& iEvent,
                                      const edm::EventSetup& iSetup,
                                      edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::ScopedContextAcquire ctx{iEvent.streamID(), std::move(waitingTaskHolder), ctxState_}; 

  edm::Handle<edm::DetSetVector<PixelDigi>> digis;
  iEvent.getByToken(pixelDigiToken_, digis);
  auto const& input = *digis;

  const TrackerGeometry* geom_ = nullptr;
  edm::ESHandle<TrackerGeometry> geom;
  iSetup.get<TrackerDigiGeometryRecord>().get(geom);
  geom_ = geom.product();

  uint32_t nDigis = 0;

  std::cout << "Looping for digis"<< std::endl;
  // for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++)
  // {
  //   nDigis += uint32_t(DSViter->size()); //Is there a smareter way?
  // }
  //
  // SiPixelDigisCUDA digis_h(nDigis,nullptr);
  //
  // std::std::vector<int> v;
  // std::cout << "Filling SoA "<< nDigis << std::endl;
  // nDigis = 0;
  // std::vector<uint16_t> i, x, y, a;
  // std::vector<uint32_t> p;

  auto x = cms::cuda::make_host_unique<uint16_t[]>(10000000, ctx.stream());
  auto y = cms::cuda::make_host_unique<uint16_t[]>(10000000, ctx.stream());
  auto a = cms::cuda::make_host_unique<uint16_t[]>(10000000, ctx.stream());
  auto i = cms::cuda::make_host_unique<uint16_t[]>(10000000, ctx.stream());
  auto p = cms::cuda::make_host_unique<uint32_t[]>(10000000, ctx.stream());
  auto r = cms::cuda::make_host_unique<uint32_t[]>(10000000, ctx.stream());

  for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++)
  {
    unsigned int detid = DSViter->detId();
    DetId detIdObject(detid);
    const GeomDetUnit* genericDet = geom_->idToDetUnit(detIdObject);
    auto const gind = genericDet->index();
    for (auto const& px : *DSViter)
    {
      x[nDigis] = uint16_t(px.row());
      y[nDigis] = uint16_t(px.column());
      a[nDigis] = uint16_t(px.adc());
      p[nDigis] = uint32_t(px.packedData());
      i[nDigis] = uint16_t(gind);
      r[nDigis] = uint32_t(detid);
      nDigis++;
  
    }
  }
  gpuAlgo_.makeDigiClustersAsync(
                             i.get(),x.get(),y.get(),a.get(),p.get(),r.get(),
                             nDigis,
                             ctx.stream());
}

void SiPixelClusterDigisCUDA::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  cms::cuda::ScopedContextProduce ctx{ctxState_};

  auto tmp = gpuAlgo_.getResults();
  ctx.emplace(iEvent, digiPutToken_, std::move(tmp.first));
  ctx.emplace(iEvent, clusterPutToken_, std::move(tmp.second));
}

// define as framework plugin
DEFINE_FWK_MODULE(SiPixelClusterDigisCUDA);
