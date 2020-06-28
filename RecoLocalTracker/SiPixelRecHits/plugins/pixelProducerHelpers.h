#ifndef pixelProducerHelpers_H
#define pixelProducerHelpers_H

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DHeterogeneous.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
using HMSstorage = HostProduct<unsigned int[]>;

// make    code rules happy
namespace pixelProducerHelpers {

  inline TrackingRecHit2DSOAView const* storeSoA(edm::Event& iEvent,
                                                 std::unique_ptr<TrackingRecHit2DHost>& input,
                                                 edm::EDPutTokenT<TrackingRecHit2DHost> const& outToken) {
    assert(input);
    auto view = input->view();
    iEvent.put(outToken, std::move(input));
    return view;
  }

  inline uint32_t const* StoreModuleStart(edm::Event& iEvent,
                                          cms::cuda::host::unique_ptr<uint32_t[]>& input,
                                          edm::EDPutTokenT<HMSstorage> const& outToken) {
    // yes a unique ptr of a unique ptr so edm is happy
    auto hitsModuleStart = input.get();
    iEvent.emplace(outToken, std::move(input));  // input is gone, hitsModuleStart still alive and kicking...
    return hitsModuleStart;
  }

  inline uint32_t const* buildAndStoreModuleStart(edm::Event& iEvent,
                                                  TrackerGeometry const& geom,
                                                  const edmNew::DetSetVector<SiPixelCluster>& input,
                                                  edm::EDPutTokenT<HMSstorage> const& outToken) {
    // fill cluster arrays
    int numberOfDetUnits = 0;
    int numberOfClusters = 0;
    auto hmsp = std::make_unique<uint32_t[]>(gpuClustering::MaxNumModules +
                                             1);  // is the memory zeroed? (not required if clusInModule is used)
    auto hitsModuleStart = hmsp.get();
    std::array<uint32_t, gpuClustering::MaxNumModules + 1> clusInModule{};  // One can spare it...
    for (auto DSViter = input.begin(); DSViter != input.end(); DSViter++) {
      unsigned int detid = DSViter->detId();
      DetId detIdObject(detid);
      const GeomDetUnit* genericDet = geom.idToDetUnit(detIdObject);
      auto gind = genericDet->index();
      // FIXME to be changed to support Phase2
      if (gind >= int(gpuClustering::MaxNumModules))
        continue;
      auto const nclus = DSViter->size();
      assert(nclus > 0);
      clusInModule[gind] = nclus;
      numberOfClusters += nclus;
      ++numberOfDetUnits;
    }
    hitsModuleStart[0] = 0;
    assert(clusInModule.size() > gpuClustering::MaxNumModules);
    for (int i = 1, n = clusInModule.size(); i < n; ++i)
      hitsModuleStart[i] = hitsModuleStart[i - 1] + clusInModule[i - 1];
    assert(numberOfClusters == int(hitsModuleStart[gpuClustering::MaxNumModules]));

    // yes a unique ptr of a unique ptr so edm is happy and the pointer stay still...
    iEvent.emplace(outToken, std::move(hmsp));  // hmsp is gone, hitsModuleStart still alive and kicking...

    return hitsModuleStart;  // pointing to the object now stored in the event
  }

}  // namespace pixelProducerHelpers

#endif
