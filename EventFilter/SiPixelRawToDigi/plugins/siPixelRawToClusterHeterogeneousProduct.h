#ifndef EventFilter_SiPixelRawToDigi_siPixelRawToClusterHeterogeneousProduct_h
#define EventFilter_SiPixelRawToDigi_siPixelRawToClusterHeterogeneousProduct_h

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelRawDataError.h"
#include "FWCore/Utilities/interface/typedefs.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

namespace siPixelRawToClusterHeterogeneousProduct {
  struct CPUProduct {
    edm::DetSetVector<PixelDigi> collection;
    edm::DetSetVector<SiPixelRawDataError> errorcollection;
    DetIdCollection tkerror_detidcollection;
    DetIdCollection usererror_detidcollection;
    edmNew::DetSetVector<PixelFEDChannel> disabled_channelcollection;
    SiPixelClusterCollectionNew outputClusters;
  };

  struct error_obj {
    uint32_t rawId;
    uint32_t word;
    unsigned char errorType;
    unsigned char fedId;

    constexpr
    error_obj(uint32_t a, uint32_t b, unsigned char c, unsigned char d):
      rawId(a),
      word(b),
      errorType(c),
      fedId(d)
    { }
  };

  struct GPUProduct {
    uint32_t const * pdigi_h = nullptr;
    uint32_t const * rawIdArr_h = nullptr;
    int32_t const * clus_h = nullptr;
    uint16_t const * adc_h = nullptr;
    GPU::SimpleVector<error_obj> const * error_h = nullptr;
  };

  using HeterogeneousDigiCluster = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                            heterogeneous::GPUCudaProduct<GPUProduct> >;
}

#endif
