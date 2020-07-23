#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DReduced.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"

class SiPixelRecHitFromSOA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiPixelRecHitFromSOA(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitFromSOA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using HMSstorage = HostProduct<unsigned int[]>;

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<TrackingRecHit2DCUDA>> tokenHit_;  // CUDA hits
  edm::EDGetTokenT<SiPixelClusterCollectionNew> clusterToken_;           // Legacy Clusters

  uint32_t m_nHits;
  cms::cuda::host::unique_ptr<uint16_t[]> m_store16;
  cms::cuda::host::unique_ptr<float[]> m_store32;
  cms::cuda::host::unique_ptr<uint32_t[]> m_hitsModuleStart;

  bool m_clusterLess;
};

SiPixelRecHitFromSOA::SiPixelRecHitFromSOA(const edm::ParameterSet& iConfig)
    : tokenHit_(
          consumes<cms::cuda::Product<TrackingRecHit2DCUDA>>(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
      m_clusterLess(iConfig.getParameter<bool>("clusterLess")) {
  if (!m_clusterLess)
    clusterToken_ = consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("src"));
  produces<SiPixelRecHitCollectionNew>();
  produces<HMSstorage>();
  produces<TrackingRecHit2DReduced>();
}

void SiPixelRecHitFromSOA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<bool>("clusterLess", false);
  descriptions.add("siPixelRecHitFromSOA", desc);
}

void SiPixelRecHitFromSOA::acquire(edm::Event const& iEvent,
                                   edm::EventSetup const& iSetup,
                                   edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<TrackingRecHit2DCUDA> const& inputDataWrapped = iEvent.get(tokenHit_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& inputData = ctx.get(inputDataWrapped);

  m_nHits = inputData.nHits();

  // std::cout<< "converting " << m_nHits << " Hits"<< std::endl;

  if (0 == m_nHits)
    return;
  m_store32 = inputData.localCoordToHostAsync(ctx.stream());
  m_store16 = inputData.detIndexToHostAsync(ctx.stream());
  m_hitsModuleStart = inputData.hitsModuleStartToHostAsync(ctx.stream());
}

void SiPixelRecHitFromSOA::produce(edm::Event& iEvent, edm::EventSetup const& es) {
  constexpr uint32_t MaxHitsInModule = gpuClustering::MaxHitsInModule;

  // yes a unique ptr of a unique ptr so edm is happy
  auto hitsModuleStart = m_hitsModuleStart.get();
  auto hms = std::make_unique<HMSstorage>(std::move(m_hitsModuleStart));  // m_hitsModuleStart is gone
  iEvent.put(std::move(hms));                                             // hms is gone!

  auto output = std::make_unique<SiPixelRecHitCollectionNew>();
  if (0 == m_nHits) {
    iEvent.put(std::move(output));
    return;
  }

  auto xl = m_store32.get();
  auto yl = xl + m_nHits;
  auto xe = yl + m_nHits;
  auto ye = xe + m_nHits;

  auto hlp = std::make_unique<TrackingRecHit2DReduced>(std::move(m_store32),std::move(m_store16),m_nHits);  // m_store32/16 are gone!
  auto orphanHandle = iEvent.put(std::move(hlp));                 // hlp is gone
  edm::RefProd<TrackingRecHit2DReduced> refProd{orphanHandle};
  assert(refProd.isNonnull());

  edm::ESHandle<TrackerGeometry> hgeom;
  es.get<TrackerDigiGeometryRecord>().get(hgeom);
  auto const& geom = *hgeom.product();

  if (m_clusterLess) {
    // the usual mess
    auto const& dus = geom.detUnits();
    unsigned m_detectors = dus.size();
    for (unsigned int i = 1; i < 7; ++i) {
      LogDebug("PixelCPEBase:: LookingForFirstStrip")
          << "Subdetector " << i << " GeomDetEnumerator " << GeomDetEnumerators::tkDetEnum[i] << " offset "
          << geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) << " is it strip? "
          << (geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size()
                  ? dus[geom.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isOuterTracker()
                  : false);

      if (geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() &&
          dus[geom.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isOuterTracker()) {
        if (geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_detectors) {
          m_detectors = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
        }
      }
    }

    assert(m_detectors <= gpuClustering::MaxNumModules);
    for (int gind = 0; gind < int(m_detectors); ++gind) {
      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(dus[gind]);
      assert(pixDet);
      assert(pixDet->index() == gind);
      auto detid = pixDet->geographicalId();
      SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(*output, detid);
      int fc = hitsModuleStart[gind];
      int lc = hitsModuleStart[gind + 1];
      int nhits = lc - fc;
      if (0 == nhits)
        continue;

      assert(lc > fc);
      // std::cout << "in det " << gind << ": conv " << nhits << " hits " << fc <<','<<lc<<std::endl;
      if (nhits > int(MaxHitsInModule))
        printf(
            "WARNING: too many clusters %d in Module %d. Only first %d Hits converted\n", nhits, gind, MaxHitsInModule);
      nhits = std::min(nhits, int(MaxHitsInModule));
      auto jnd = [&](int k) { return fc + k; };
      for (int i = 0; i < nhits; ++i) {
        auto ij = jnd(i);
        if (ij >= int(TrackingRecHit2DSOAView::maxHits()))
          continue;  // overflow...
        LocalPoint lp(xl[ij], yl[ij]);
        LocalError le(xe[ij], 0, ye[ij]);
        SiPixelRecHitQuality::QualWordType rqw = 0;

        // Create a persistent edm::Ref to the cluster
        OmniClusterRef notCluster(refProd, ij);
        // Make a RecHit and add it to the DetSet
        SiPixelRecHit hit(lp, le, rqw, *pixDet, notCluster);
        //
        // Now save it =================
        recHitsOnDetUnit.push_back(hit);
        std::push_heap(recHitsOnDetUnit.begin(), recHitsOnDetUnit.end(), [](auto const& h1, auto const& h2) {
          return h1.localPosition().x() < h2.localPosition().x();
        });
        // =============================

      }  // hits
      std::sort_heap(recHitsOnDetUnit.begin(), recHitsOnDetUnit.end(), [](auto const& h1, auto const& h2) {
        return h1.localPosition().x() < h2.localPosition().x();
      });

    }  // detunit
  } else {
    edm::Handle<SiPixelClusterCollectionNew> hclusters;
    iEvent.getByToken(clusterToken_, hclusters);

    auto const& input = *hclusters;

    int numberOfDetUnits = 0;
    int numberOfClusters = 0;
    for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
      numberOfDetUnits++;
      unsigned int detid = DSViter->detId();
      DetId detIdObject(detid);
      const GeomDetUnit* genericDet = geom.idToDetUnit(detIdObject);
      auto gind = genericDet->index();
      const PixelGeomDetUnit* pixDet = dynamic_cast<const PixelGeomDetUnit*>(genericDet);
      assert(pixDet);
      SiPixelRecHitCollectionNew::FastFiller recHitsOnDetUnit(*output, detid);
      auto fc = hitsModuleStart[gind];
      auto lc = hitsModuleStart[gind + 1];
      auto nhits = lc - fc;

      assert(lc > fc);
      // std::cout << "in det " << gind << ": conv " << nhits << " hits from " << DSViter->size() << " legacy clusters"
      //          <<' '<< fc <<','<<lc<<std::endl;
      if (nhits > MaxHitsInModule)
        printf(
            "WARNING: too many clusters %d in Module %d. Only first %d Hits converted\n", nhits, gind, MaxHitsInModule);
      nhits = std::min(nhits, MaxHitsInModule);

      //std::cout << "in det " << gind << "conv " << nhits << " hits from " << DSViter->size() << " legacy clusters"
      //          <<' '<< lc <<','<<fc<<std::endl;

      if (0 == nhits)
        continue;
      auto jnd = [&](int k) { return fc + k; };
      assert(nhits <= DSViter->size());
      if (nhits != DSViter->size()) {
        edm::LogWarning("GPUHits2CPU") << "nhits!= nclus " << nhits << ' ' << DSViter->size() << std::endl;
      }
      for (auto const& clust : *DSViter) {
        assert(clust.originalId() >= 0);
        assert(clust.originalId() < DSViter->size());
        if (clust.originalId() >= nhits)
          continue;
        auto ij = jnd(clust.originalId());
        if (ij >= TrackingRecHit2DSOAView::maxHits())
          continue;  // overflow...
        LocalPoint lp(xl[ij], yl[ij]);
        LocalError le(xe[ij], 0, ye[ij]);
        SiPixelRecHitQuality::QualWordType rqw = 0;

        numberOfClusters++;

        /*   cpu version....  (for reference)
           std::tuple<LocalPoint, LocalError, SiPixelRecHitQuality::QualWordType> tuple = cpe_->getParameters( clust, *genericDet );
           LocalPoint lp( std::get<0>(tuple) );
           LocalError le( std::get<1>(tuple) );
           SiPixelRecHitQuality::QualWordType rqw( std::get<2>(tuple) );
      */

        // Create a persistent edm::Ref to the cluster
        edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> cluster = edmNew::makeRefTo(hclusters, &clust);
        // Make a RecHit and add it to the DetSet
        SiPixelRecHit hit(lp, le, rqw, *genericDet, cluster);
        //
        // Now save it =================
        recHitsOnDetUnit.push_back(hit);
        std::push_heap(recHitsOnDetUnit.begin(), recHitsOnDetUnit.end(), [](auto const& h1, auto const& h2) {
          return h1.localPosition().x() < h2.localPosition().x();
        });  // =============================

        // std::cout << "SiPixelRecHitGPUVI " << numberOfClusters << ' '<< lp << " " << le << std::endl;

      }  //  <-- End loop on Clusters
      std::sort_heap(recHitsOnDetUnit.begin(), recHitsOnDetUnit.end(), [](auto const& h1, auto const& h2) {
        return h1.localPosition().x() < h2.localPosition().x();
      });

      //  LogDebug("SiPixelRecHitGPU")
      //std::cout << "SiPixelRecHitGPUVI "
      //	<< " Found " << recHitsOnDetUnit.size() << " RecHits on " << detid //;
      // << std::endl;

    }  //    <-- End loop on DetUnits

    /*
  std::cout << "SiPixelRecHitGPUVI $ det, clus, lost "
    <<  numberOfDetUnits << ' '
    << numberOfClusters  << ' '
    << std::endl;
  */
  }
  iEvent.put(std::move(output));
}

DEFINE_FWK_MODULE(SiPixelRecHitFromSOA);
