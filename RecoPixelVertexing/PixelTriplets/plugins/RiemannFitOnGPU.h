#ifndef RecoPixelVertexing_PixelTrackFitting_plugins_RiemannFitOnGPU_h
#define RecoPixelVertexing_PixelTrackFitting_plugins_RiemannFitOnGPU_h

#include "RecoPixelVertexing/PixelTrackFitting/interface/FitResult.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

namespace siPixelRecHitsHeterogeneousProduct {
   struct HitsOnCPU;
}

class RiemannFitOnGPU {
public:

   using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
   using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

   using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

   RiemannFitOnGPU() = default;
   ~RiemannFitOnGPU() { deallocateOnGPU();}

   void setBField(double bField) { bField_ = bField;}
   void launchKernels(HitsOnCPU const & hh, uint32_t nhits, uint32_t maxNumberOfTuples, cudaStream_t cudaStream);

   void allocateOnGPU(TuplesOnGPU::Container const * tuples, Rfit::helix_fit * helix_fit_results);
   void deallocateOnGPU();


private:

    static constexpr uint32_t maxNumberOfConcurrentFits_ = 10000;

    // fowarded
    TuplesOnGPU::Container const * tuples_d = nullptr;
    double bField_;
    Rfit::helix_fit * helix_fit_results_d = nullptr;



   // Riemann Fit internals
   Rfit::Matrix3xNd *hitsGPU_ = nullptr;
   Rfit::Matrix3Nd *hits_covGPU_ = nullptr;
   Eigen::Vector4d *fast_fit_resultsGPU_ = nullptr;
   Rfit::circle_fit *circle_fit_resultsGPU_ = nullptr;
   Rfit::line_fit *line_fit_resultsGPU_ = nullptr;

};

#endif
