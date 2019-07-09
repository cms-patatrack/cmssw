//
// Author: Felice Pantaleo, CERN
//

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DCUDA.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/BrokenLine.h"

#include "HelixFitOnGPU.h"

using HitsOnGPU = TrackingRecHit2DSOAView;
using Tuples = PixelTrackCUDA::HitContainer;
using OutputSoA = PixelTrackCUDA::SoA;

using namespace Eigen;

// #define BL_DUMP_HITS

template <int N>
__global__ void kernelBLFastFit(Tuples const *__restrict__ foundNtuplets,
                                CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                                HitsOnGPU const *__restrict__ hhp,
                                double *__restrict__ phits,
                                float *__restrict__ phits_ge,
                                double *__restrict__ pfast_fit,
                                uint32_t nHits,
                                uint32_t offset) {
  constexpr uint32_t hitsInFit = N;

  assert(hitsInFit <= nHits);

  assert(pfast_fit);
  assert(foundNtuplets);
  assert(tupleMultiplicity);

  // look in bin for this hit multiplicity
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);

#ifdef BROKENLINE_DEBUG
  if (0 == local_start) {
    printf("%d total Ntuple\n",foundNtuplets->nbins());
    printf("%d Ntuple of size %d for %d hits to fit\n", tupleMultiplicity->size(nHits), nHits, hitsInFit);
  }
#endif

  auto tuple_start = local_start + offset;
  if (tuple_start >= tupleMultiplicity->size(nHits))
    return;

  // get it from the ntuple container (one to one to helix)
  auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_start);
  assert(tkid < foundNtuplets->nbins());

  assert(foundNtuplets->size(tkid) == nHits);

  Rfit::Map3xNd<N> hits(phits + local_start);
  Rfit::Map4d fast_fit(pfast_fit + local_start);
  Rfit::Map6xNf<N> hits_ge(phits_ge + local_start);

#ifdef BL_DUMP_HITS
  __shared__ int done;
  done = 0;
  __syncthreads();
  bool dump = (foundNtuplets->size(tkid) == 5 && 0 == atomicAdd(&done, 1));
#endif

  // Prepare data structure
  auto const *hitId = foundNtuplets->begin(tkid);
  for (unsigned int i = 0; i < hitsInFit; ++i) {
    auto hit = hitId[i];
    float ge[6];
    hhp->cpeParams().detParams(hhp->detectorIndex(hit)).frame.toGlobal(hhp->xerrLocal(hit), 0, hhp->yerrLocal(hit), ge);
#ifdef BL_DUMP_HITS
    if (dump) {
      printf("Hit global: %d: %d hits.col(%d) << %f,%f,%f\n",
             tkid,
             hhp->detectorIndex(hit),
             i,
             hhp->xGlobal(hit),
             hhp->yGlobal(hit),
             hhp->zGlobal(hit));
      printf("Error: %d: %d  hits_ge.col(%d) << %e,%e,%e,%e,%e,%e\n",
             tkid,
             hhp->detetectorIndex(hit),
             i,
             ge[0],
             ge[1],
             ge[2],
             ge[3],
             ge[4],
             ge[5]);
    }
#endif
    hits.col(i) << hhp->xGlobal(hit), hhp->yGlobal(hit), hhp->zGlobal(hit);
    hits_ge.col(i) << ge[0], ge[1], ge[2], ge[3], ge[4], ge[5];
  }
  BrokenLine::BL_Fast_fit(hits, fast_fit);

  // no NaN here....
  assert(fast_fit(0) == fast_fit(0));
  assert(fast_fit(1) == fast_fit(1));
  assert(fast_fit(2) == fast_fit(2));
  assert(fast_fit(3) == fast_fit(3));
}

template <int N>
__global__ void kernelBLFit(CAConstants::TupleMultiplicity const *__restrict__ tupleMultiplicity,
                            double B,
                            OutputSoA *results,
                            double *__restrict__ phits,
                            float *__restrict__ phits_ge,
                            double *__restrict__ pfast_fit,
                            uint32_t nHits,
                            uint32_t offset) {
  assert(N <= nHits);

  assert(results);
  assert(pfast_fit);

  // same as above...

  // look in bin for this hit multiplicity
  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto tuple_start = local_start + offset;
  if (tuple_start >= tupleMultiplicity->size(nHits))
    return;

  // get it for the ntuple container (one to one to helix)
  auto tkid = *(tupleMultiplicity->begin(nHits) + tuple_start);

  Rfit::Map3xNd<N> hits(phits + local_start);
  Rfit::Map4d fast_fit(pfast_fit + local_start);
  Rfit::Map6xNf<N> hits_ge(phits_ge + local_start);

  BrokenLine::PreparedBrokenLineData<N> data;
  Rfit::Matrix3d Jacob;

  BrokenLine::karimaki_circle_fit circle;
  Rfit::line_fit line;

  BrokenLine::prepareBrokenLineData(hits, fast_fit, B, data);
  BrokenLine::BL_Line_fit(hits_ge, fast_fit, B, data, line);
  BrokenLine::BL_Circle_fit(hits, hits_ge, fast_fit, B, data, circle);

  results->stateAtBS.copyFromCircle(circle.par,circle.cov,line.par,line.cov,1.f/float(B),tkid);
  results->pt(tkid) =  float(B)/float(std::abs(circle.par(2)));
  results->eta(tkid) =  asinhf(line.par(0));
  results->chi2(tkid) = (circle.chi2+line.chi2)/(2*N-5);

#ifdef BROKENLINE_DEBUG
  if (!(circle.chi2 >= 0) || !(line.chi2 >= 0))
    printf("kernelBLFit failed! %f/%f\n", circle.chi2, line.chi2);
  printf("kernelBLFit size %d for %d hits circle.par(0,1,2): %d %f,%f,%f\n",
         N,
         nHits,
         tkid,
         circle.par(0),
         circle.par(1),
         circle.par(2));
  printf("kernelBLHits line.par(0,1): %d %f,%f\n", tkid, line.par(0), line.par(1));
  printf("kernelBLHits chi2 cov %f/%f  %e,%e,%e,%e,%e\n",
         circle.chi2,
         line.chi2,
         circle.cov(0, 0),
         circle.cov(1, 1),
         circle.cov(2, 2),
         line.cov(0, 0),
         line.cov(1, 1));
#endif
}

void HelixFitOnGPU::launchBrokenLineKernels(HitsOnCPU const &hh,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            cuda::stream_t<> &stream) {
  assert(tuples_d);

  auto blockSize = 64;
  auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

  //  Fit internals
  edm::Service<CUDAService> cs;
  auto hitsGPU_ = cs->make_device_unique<double[]>(
      maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd<4>) / sizeof(double), stream);
  auto hits_geGPU_ =
      cs->make_device_unique<float[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix6x4f) / sizeof(float), stream);
  auto fast_fit_resultsGPU_ =
      cs->make_device_unique<double[]>(maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d) / sizeof(double), stream);

  for (uint32_t offset = 0; offset < maxNumberOfTuples; offset += maxNumberOfConcurrentFits_) {
    // fit triplets
    kernelBLFastFit<3><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                      tupleMultiplicity_d,
                                                                      hh.view(),
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      3,
                                                                      offset);
    cudaCheck(cudaGetLastError());

    kernelBLFit<3><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                  bField_,
                                                                  outputSoa_d,
                                                                  hitsGPU_.get(),
                                                                  hits_geGPU_.get(),
                                                                  fast_fit_resultsGPU_.get(),
                                                                  3,
                                                                  offset);
    cudaCheck(cudaGetLastError());

    // fit quads
    kernelBLFastFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                      tupleMultiplicity_d,
                                                                      hh.view(),
                                                                      hitsGPU_.get(),
                                                                      hits_geGPU_.get(),
                                                                      fast_fit_resultsGPU_.get(),
                                                                      4,
                                                                      offset);
    cudaCheck(cudaGetLastError());

    kernelBLFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                  bField_,
                                                                  outputSoa_d,
                                                                  hitsGPU_.get(),
                                                                  hits_geGPU_.get(),
                                                                  fast_fit_resultsGPU_.get(),
                                                                  4,
                                                                  offset);
    cudaCheck(cudaGetLastError());

    if (fit5as4_) {
      // fit penta (only first 4)
      kernelBLFastFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                        tupleMultiplicity_d,
                                                                        hh.view(),
                                                                        hitsGPU_.get(),
                                                                        hits_geGPU_.get(),
                                                                        fast_fit_resultsGPU_.get(),
                                                                        5,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<4><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                    bField_,
                                                                    outputSoa_d,
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    5,
                                                                    offset);
      cudaCheck(cudaGetLastError());
    } else {
      // fit penta (all 5)
      kernelBLFastFit<5><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tuples_d,
                                                                        tupleMultiplicity_d,
                                                                        hh.view(),
                                                                        hitsGPU_.get(),
                                                                        hits_geGPU_.get(),
                                                                        fast_fit_resultsGPU_.get(),
                                                                        5,
                                                                        offset);
      cudaCheck(cudaGetLastError());

      kernelBLFit<5><<<numberOfBlocks, blockSize, 0, stream.id()>>>(tupleMultiplicity_d,
                                                                    bField_,
                                                                    outputSoa_d,
                                                                    hitsGPU_.get(),
                                                                    hits_geGPU_.get(),
                                                                    fast_fit_resultsGPU_.get(),
                                                                    5,
                                                                    offset);
      cudaCheck(cudaGetLastError());
    }

  }  // loop on concurrent fits
}
