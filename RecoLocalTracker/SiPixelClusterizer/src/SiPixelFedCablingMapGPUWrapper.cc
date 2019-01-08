// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMap const& cablingMap,
                                                               TrackerGeometry const& trackerGeom,
                                                               SiPixelQuality const *badPixelInfo):
  hasQuality_(badPixelInfo != nullptr)
{
  unsigned int *fedMap = nullptr;
  unsigned int *linkMap = nullptr;
  unsigned int *rocMap = nullptr;
  unsigned int *RawId = nullptr;
  unsigned int *rocInDet = nullptr;
  unsigned int *moduleId = nullptr;
  unsigned char *badRocs = nullptr;

  helper_.allocate(&fedMap,   pixelgpudetails::MAX_SIZE);
  helper_.allocate(&linkMap,  pixelgpudetails::MAX_SIZE);
  helper_.allocate(&rocMap,   pixelgpudetails::MAX_SIZE);
  helper_.allocate(&RawId,    pixelgpudetails::MAX_SIZE);
  helper_.allocate(&rocInDet, pixelgpudetails::MAX_SIZE);
  helper_.allocate(&moduleId, pixelgpudetails::MAX_SIZE);
  helper_.allocate(&badRocs,  pixelgpudetails::MAX_SIZE);

  helperUnp_.allocate(&modToUnpDefault, pixelgpudetails::MAX_SIZE);

  std::vector<unsigned int> const& fedIds = cablingMap.fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap.cablingTree();

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed   = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  int index = 1;

  for (unsigned int fed = startFed; fed <= endFed; fed++) {
    for (unsigned int link = 1; link <= pixelgpudetails::MAX_LINK; link++) {
      for (unsigned int roc = 1; roc <= pixelgpudetails::MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling->findItem(path);
        fedMap[index] = fed;
        linkMap[index] = link;
        rocMap[index] = roc;
        if (pixelRoc != nullptr) {
          RawId[index] = pixelRoc->rawId();
          rocInDet[index] = pixelRoc->idInDetUnit();
          modToUnpDefault[index] = false;
          if (badPixelInfo != nullptr)
            badRocs[index] = badPixelInfo->IsRocBad(pixelRoc->rawId(), pixelRoc->idInDetUnit());
          else
            badRocs[index] = false;
        } else { // store some dummy number
          RawId[index] = 9999;
          rocInDet[index] = 9999;
          badRocs[index] = true;
          modToUnpDefault[index] = true;
        }
        index++;
      }
    }
  } // end of FED loop

  // Given FedId, Link and idinLnk; use the following formula
  // to get the RawId and idinDU
  // index = (FedID-1200) * MAX_LINK* MAX_ROC + (Link-1)* MAX_ROC + idinLnk;
  // where, MAX_LINK = 48, MAX_ROC = 8 for Phase1 as mentioned Danek's email
  // FedID varies between 1200 to 1338 (In total 108 FED's)
  // Link varies between 1 to 48
  // idinLnk varies between 1 to 8

  for (int i = 1; i < index; i++) {
    if (RawId[i] == 9999) {
      moduleId[i] = 9999;
    } else {
      /*
      std::cout << RawId[i] << std::endl;
      */
      auto gdet = trackerGeom.idToDetUnit(RawId[i]);
      if (!gdet) {
        LogDebug("SiPixelFedCablingMapGPU") << " Not found: " << RawId[i] << std::endl;
        continue;
      }
      moduleId[i] = gdet->index();
    }
    LogDebug("SiPixelFedCablingMapGPU") << "----------------------------------------------------------------------------" << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << fedMap[i]  << std::setw(20) << linkMap[i]  << std::setw(20) << rocMap[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << RawId[i]   << std::setw(20) << rocInDet[i] << std::setw(20) << moduleId[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << static_cast<bool>(badRocs[i]) << std::setw(20) << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << "----------------------------------------------------------------------------" << std::endl;

  }

  helper_.allocate(&cablingGPU_, 1);
  cablingGPU_->size = index-1;
  cablingGPU_->fed = fedMap;
  cablingGPU_->link = linkMap;
  cablingGPU_->roc = rocMap;
  cablingGPU_->RawId = RawId;
  cablingGPU_->rocInDet = rocInDet;
  cablingGPU_->moduleId = moduleId;
  cablingGPU_->badRocs = badRocs;
  helper_.advise();
  helperUnp_.advise();
}


SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {}


const SiPixelFedCablingMapGPU *SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(cuda::stream_t<>& cudaStream) const {
  helper_.prefetchAsync(cudaStream);
  return cablingGPU_;
}

const unsigned char *SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(cuda::stream_t<>& cudaStream) const {
  helperUnp_.prefetchAsync(cudaStream);
  return modToUnpDefault;
}

edm::cuda::device::unique_ptr<unsigned char[]> SiPixelFedCablingMapGPUWrapper::getModToUnpRegionalAsync(std::set<unsigned int> const& modules, cuda::stream_t<>& cudaStream) const {
  edm::Service<CUDAService> cs;
  auto modToUnpDevice = cs->make_device_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE, cudaStream);
  auto modToUnpHost = cs->make_host_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE, cudaStream);

  std::vector<unsigned int> const& fedIds = cablingMap_->fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap_->cablingTree();

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed   = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  int index = 1;

  for (unsigned int fed = startFed; fed <= endFed; fed++) {
    for (unsigned int link = 1; link <= pixelgpudetails::MAX_LINK; link++) {
      for (unsigned int roc = 1; roc <= pixelgpudetails::MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling->findItem(path);
        if (pixelRoc != nullptr) {
          modToUnpHost[index] = (not modules.empty()) and (modules.find(pixelRoc->rawId()) == modules.end());
        } else { // store some dummy number
          modToUnpHost[index] = true;
        }
        index++;
      }
    }
  }

  cuda::memory::async::copy(modToUnpDevice.get(), modToUnpHost.get(), pixelgpudetails::MAX_SIZE * sizeof(unsigned char), cudaStream.id());
  return modToUnpDevice;
}
