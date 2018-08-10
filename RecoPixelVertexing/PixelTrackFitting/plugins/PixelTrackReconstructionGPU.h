#ifndef RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstructionGPU_H
#define RecoPixelVertexing_PixelTrackFitting_PixelTrackReconstructionGPU_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/TracksWithHits.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <memory>
#include <vector>
#include <cuda_runtime.h>


class PixelFitter;
class PixelTrackCleaner;
class PixelTrackFilter;
class RegionsSeedingHitSets;

namespace edm { class Event; class EventSetup; class Run; class ParameterSetDescription;}

class PixelTrackReconstructionGPU {
public:

  PixelTrackReconstructionGPU( const edm::ParameterSet& conf,
	   edm::ConsumesCollector && iC);
  ~PixelTrackReconstructionGPU();

  static void fillDescriptions(edm::ParameterSetDescription& desc);

  void run(pixeltrackfitting::TracksWithTTRHs& tah, edm::Event& ev, const edm::EventSetup& es);
  void launchKernelFit(float * hits_and_covariances, int cumulative_size, int hits_in_fit, float B,
      Rfit::helix_fit * results);

private:
  edm::EDGetTokenT<RegionsSeedingHitSets> theHitSetsToken;
  edm::EDGetTokenT<PixelFitter> theFitterToken;
  edm::EDGetTokenT<PixelTrackFilter> theFilterToken;
  std::string theCleanerName;
};
#endif
