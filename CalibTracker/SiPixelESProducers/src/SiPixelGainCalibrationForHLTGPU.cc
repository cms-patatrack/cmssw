#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTGPU.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include <cuda.h>

SiPixelGainCalibrationForHLTGPU::SiPixelGainCalibrationForHLTGPU(const SiPixelGainCalibrationForHLT& gains, const TrackerGeometry& geom)
{
  // bizzarre logic (looking for fist strip-det) don't ask
  auto const & dus = geom.detUnits();
  unsigned m_detectors = dus.size();
  for(unsigned int i=1;i<7;++i) {
    if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() &&
        dus[geom.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip()) {
      if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_detectors) m_detectors = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    }
  }

  /*
  std::cout << "caching calibs for " << m_detectors << " pixel detectors of size " << gains.data().size() << std::endl;
  std::cout << "sizes " << sizeof(char) << ' ' << sizeof(uint8_t) << ' ' << sizeof(SiPixelGainForHLTonGPU::DecodingStructure) << std::endl;
  */

  helper_.allocate(&gainForHLT_, 1);

  // do not read back from the (possibly write-combined) memory buffer
  auto minPed  = gains.getPedLow();
  auto maxPed  = gains.getPedHigh();
  auto minGain = gains.getGainLow();
  auto maxGain = gains.getGainHigh();
  auto nBinsToUseForEncoding = 253;

  // we will simplify later (not everything is needed....)
  gainForHLT_->minPed_ = minPed;
  gainForHLT_->maxPed_ = maxPed;
  gainForHLT_->minGain_= minGain;
  gainForHLT_->maxGain_= maxGain;

  gainForHLT_->numberOfRowsAveragedOver_ = 80;
  gainForHLT_->nBinsToUseForEncoding_    = nBinsToUseForEncoding;
  gainForHLT_->deadFlag_                 = 255;
  gainForHLT_->noisyFlag_                = 254;

  gainForHLT_->pedPrecision  = static_cast<float>(maxPed - minPed) / nBinsToUseForEncoding;
  gainForHLT_->gainPrecision = static_cast<float>(maxGain - minGain) / nBinsToUseForEncoding;

  /*
  std::cout << "precisions g " << gainForHLT_->pedPrecision << ' ' << gainForHLT_->gainPrecision << std::endl;
  */

  // fill the index map
  auto const & ind = gains.getIndexes();
  /*
  std::cout << ind.size() << " " << m_detectors << std::endl;
  */

  for (auto i=0U; i<m_detectors; ++i) {
    auto p = std::lower_bound(ind.begin(),ind.end(),dus[i]->geographicalId().rawId(),SiPixelGainCalibrationForHLT::StrictWeakOrdering());
    assert (p!=ind.end() && p->detid==dus[i]->geographicalId());
    assert(p->iend<=gains.data().size());
    assert(p->iend>=p->ibegin);
    assert(0==p->ibegin%2);
    assert(0==p->iend%2);
    assert(p->ibegin!=p->iend);
    assert(p->ncols>0);
    gainForHLT_->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(p->ibegin,p->iend), p->ncols);
    // if (ind[i].detid!=dus[i]->geographicalId()) std::cout << ind[i].detid<<"!="<<dus[i]->geographicalId() << std::endl;
    // gainForHLT_->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(ind[i].ibegin,ind[i].iend), ind[i].ncols);
  }

  helper_.allocate(&(gainForHLT_->v_pedestals), gains.data().size(), sizeof(char)); // override the element size because essentially we reinterpret_cast on the fly
  std::memcpy(gainForHLT_->v_pedestals, gains.data().data(), gains.data().size()*sizeof(char));

  helper_.advise();
}

SiPixelGainCalibrationForHLTGPU::~SiPixelGainCalibrationForHLTGPU() {
}

const SiPixelGainForHLTonGPU *SiPixelGainCalibrationForHLTGPU::getGPUProductAsync(cuda::stream_t<>& cudaStream) const {
  helper_.prefetchAsync(cudaStream);
  return gainForHLT_;
}
