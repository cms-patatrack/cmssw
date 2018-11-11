//
// Author: Felice Pantaleo, CERN
//

#include "RiemannFitOnGPU.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RiemannFit.h"

#include <cstdint>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"


using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

using namespace Eigen;

__global__
void kernelFastFitAllHits(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    HitsOnGPU const * __restrict__ hhp,
    int hits_in_fit,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::Vector4d *fast_fit,
    uint32_t offset)
{

  assert(fast_fit); assert(foundNtuplets);

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }


  hits[local_start].resize(3, hits_in_fit);
  hits_cov[local_start].resize(3 * hits_in_fit, 3 * hits_in_fit);

  // Prepare data structure
  auto const * hitId = foundNtuplets->begin(helix_start);
  for (unsigned int i = 0; i < hits_in_fit; ++i) {
    auto hit = hitId[i];
    //  printf("Hit global_x: %f\n", hhp->xg_d[hit]);
    float ge[6];
    hhp->cpeParams->detParams(hhp->detInd_d[hit]).frame.toGlobal(hhp->xerr_d[hit], 0, hhp->yerr_d[hit], ge);
    //  printf("Error: %d: %f,%f,%f,%f,%f,%f\n",hhp->detInd_d[hit],ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);

    hits[local_start].col(i) << hhp->xg_d[hit], hhp->yg_d[hit], hhp->zg_d[hit];

    for (auto j = 0; j < 3; ++j) {
      for (auto l = 0; l < 3; ++l) {
        // Index numerology:
        // i: index of the hits/point (0,..,3)
        // j: index of space component (x,y,z)
        // l: index of space components (x,y,z)
        // ge is always in sync with the index i and is formatted as:
        // ge[] ==> [xx, xy, xz, yy, yz, zz]
        // in (j,l) notation, we have:
        // ge[] ==> [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)]
        // so the index ge_idx corresponds to the matrix elements:
        // | 0  1  2 |
        // | 1  3  4 |
        // | 2  4  5 |
        auto ge_idx = j + l + (j > 0 and l > 0);
        hits_cov[local_start](i + j * hits_in_fit, i + l * hits_in_fit) = ge[ge_idx];
      }
    }
  }
  fast_fit[local_start] = Rfit::Fast_fit(hits[local_start]);

  assert(fast_fit[local_start](0)==fast_fit[local_start](0));
  assert(fast_fit[local_start](1)==fast_fit[local_start](1));
  assert(fast_fit[local_start](2)==fast_fit[local_start](2));
  assert(fast_fit[local_start](3)==fast_fit[local_start](3));

}

__global__
void kernelCircleFitAllHits(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    int hits_in_fit,
    double B,
    Rfit::Matrix3xNd const * __restrict__ hits,
    Rfit::Matrix3Nd  const * __restrict__ hits_cov,
    Rfit::circle_fit *circle_fit,
    Rfit::Vector4d const * __restrict__ fast_fit,
    uint32_t offset)
{
  assert(circle_fit); 

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }

  auto n = hits[local_start].cols();

  Rfit::VectorNd rad = (hits[local_start].block(0, 0, 2, n).colwise().norm());

  circle_fit[local_start] =
      Rfit::Circle_fit(hits[local_start].block(0, 0, 2, n),
                       hits_cov[local_start].block(0, 0, 2 * n, 2 * n),
                       fast_fit[local_start], rad, B, true);

#ifdef GPU_DEBUG
  printf("kernelCircleFitAllHits circle.par(0): %d %f\n", helix_start, circle_fit[local_start].par(0));
  printf("kernelCircleFitAllHits circle.par(1): %d %f\n", helix_start, circle_fit[local_start].par(1));
  printf("kernelCircleFitAllHits circle.par(2): %d %f\n", helix_start, circle_fit[local_start].par(2));
#endif
}

__global__
void kernelLineFitAllHits(TuplesOnGPU::Container const * __restrict__ foundNtuplets,
    int hits_in_fit,
    double B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd const * __restrict__ hits,
    Rfit::Matrix3Nd const * __restrict__ hits_cov,
    Rfit::circle_fit * __restrict__ circle_fit,
    Rfit::Vector4d const * __restrict__ fast_fit,
    Rfit::line_fit *line_fit,
    uint32_t offset)
{

  assert(results); assert(line_fit);

  auto local_start = (blockIdx.x * blockDim.x + threadIdx.x);
  auto helix_start = local_start + offset;

  if (helix_start>=foundNtuplets->nbins()) return;
  if (foundNtuplets->size(helix_start)<hits_in_fit) {
    return;
  }

  line_fit[local_start] = Rfit::Line_fit(hits[local_start], hits_cov[local_start], circle_fit[local_start], fast_fit[local_start], B, true);

  par_uvrtopak(circle_fit[local_start], B, true);

  // Grab helix_fit from the proper location in the output vector
  auto & helix = results[helix_start];
  helix.par << circle_fit[local_start].par, line_fit[local_start].par;

  // TODO: pass properly error booleans

  helix.cov = MatrixXd::Zero(5, 5);
  helix.cov.block(0, 0, 3, 3) = circle_fit[local_start].cov;
  helix.cov.block(3, 3, 2, 2) = line_fit[local_start].cov;

  helix.q = circle_fit[local_start].q;
  helix.chi2_circle = circle_fit[local_start].chi2;
  helix.chi2_line = line_fit[local_start].chi2;

#ifdef GPU_DEBUG
  printf("kernelLineFitAllHits line.par(0): %d %f\n", helix_start, circle_fit[local_start].par(0));
  printf("kernelLineFitAllHits line.par(1): %d %f\n", helix_start, line_fit[local_start].par(1));
#endif
}


void RiemannFitOnGPU::launchKernels(HitsOnCPU const & hh, uint32_t nhits, uint32_t maxNumberOfTuples, cudaStream_t cudaStream)
{
    assert(tuples_d); assert(fast_fit_resultsGPU_);

    auto blockSize = 128;
    auto numberOfBlocks = (maxNumberOfConcurrentFits_ + blockSize - 1) / blockSize;

    for (uint32_t offset=0; offset<maxNumberOfTuples; offset+=maxNumberOfConcurrentFits_) {
      kernelFastFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, hh.gpu_d, 4,
          hitsGPU_, hits_covGPU_, fast_fit_resultsGPU_,offset);
      cudaCheck(cudaGetLastError());

      kernelCircleFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
          tuples_d, 4, bField_,
          hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_, offset);
      cudaCheck(cudaGetLastError());


      kernelLineFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
             tuples_d, 4,  bField_, helix_fit_results_d,
             hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
             line_fit_resultsGPU_, offset);
      cudaCheck(cudaGetLastError());
    }
}
