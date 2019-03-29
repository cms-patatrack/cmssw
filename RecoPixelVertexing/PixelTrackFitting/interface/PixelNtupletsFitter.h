#ifndef RecoPixelVertexing_PixelTrackFitting_PixelNtupletsFitter_H
#define RecoPixelVertexing_PixelTrackFitting_PixelNtupletsFitter_H

#include <vector>

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"

class PixelNtupletsFitter final : public PixelFitterBase {
public:
  explicit PixelNtupletsFitter(float nominalB, const MagneticField *field, bool useRiemannFit);
  ~PixelNtupletsFitter() override = default;
  std::unique_ptr<reco::Track> run(const std::vector<const TrackingRecHit *>& hits,
                                   const TrackingRegion& region, const edm::EventSetup& es) const override;

private:
  float nominalB_;
  const MagneticField *field_;
  bool useRiemannFit_;
};

#endif // RecoPixelVertexing_PixelTrackFitting_PixelNtupletsFitter_H
