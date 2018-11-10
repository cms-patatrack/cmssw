#ifndef RecoPixelVertexing_PixelTrackFitting_plugins_RiemannFitOnGPU_h
#define RecoPixelVertexing_PixelTrackFitting_plugins_RiemannFitOnGPU_h

#include "RecoPixelVertexing/PixelTrackFitting/interface/pixelTrackHeterogeneousProduct.h"

class RiemannFitOnGPU {
public:
   using Input = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;
   using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;
   using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;

   using TracksOnGPU = pixelTuplesHeterogeneousProduct::TracksOnGPU;
   using TracksOnCPU = pixelTuplesHeterogeneousProduct::TracksOnCPU;
   using Output = pixelTrackHeterogeneousProduct::HeterogeneousPixelTracks;

   RiemannFitOnGPU() = default;
   ~RiemannFitOnGPU() { deallocateOnGPU();}

   void launchKernels(TuplesOnCPU const & hh, bool transferToCPU, cudaStream_t);

   TracksOnCPU getOutput() const {
       return TuplesOnCPU { helix_fit_results_,  gpu_d, nTracks_};
   }

   void allocateOnGPU();
   void deallocateOnGPU();


private:

    static constexpr int maxNumberOfTracks_ = 10000;


   // input
   TuplesOnCPU const * tuplesOnCPU;

   // products
   TracksOnGPU * gpu_d = nullptr;   // copy of the structure on the gpu itself: this is the "Product"
   Rfit::helix_fit * helix_fit_results_ = nullptr;
   uint32_t nTracks_ = 0;
   TracksOnGPU gpu_;

   // Riemann Fit internals
   Rfit::Matrix3xNd *hitsGPU_ = nullptr;
   Rfit::Matrix3Nd *hits_covGPU_ = nullptr;
   Eigen::Vector4d *fast_fit_resultsGPU_ = nullptr;
   Rfit::circle_fit *circle_fit_resultsGPU_ = nullptr;
   Rfit::line_fit *line_fit_resultsGPU_ = nullptr;

};

#endif
