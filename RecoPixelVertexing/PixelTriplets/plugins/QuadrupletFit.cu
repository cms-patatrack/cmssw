//
// Author: Felice Pantaleo, CERN
//

#include <cstdint>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "CAHitQuadrupletGeneratorGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"


using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;
using namespace Eigen;

__global__
void kernelFastFitAllHits(GPU::SimpleVector<Quadruplet> * foundNtuplets,
    siPixelRecHitsHeterogeneousProduct::HitsOnGPU const * hhp,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
  printf("BlockDim.x: %d, BlockIdx.x: %d, threadIdx.x: %d, helix_start: %d, cumulative_size: %d\n",
      blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif

  hits[helix_start].resize(3, hits_in_fit);
  hits_cov[helix_start].resize(3 * hits_in_fit, 3 * hits_in_fit);

  // Prepare data structure
  for (unsigned int i = 0; i < hits_in_fit; ++i) {
    auto hit = (*foundNtuplets)[helix_start].hitId[i];
    //  printf("Hit global_x: %f\n", hhp->xg_d[hit]);
    float ge[6];
    hhp->cpeParams->detParams(hhp->detInd_d[hit]).frame.toGlobal(hhp->xerr_d[hit], 0, hhp->yerr_d[hit], ge);
    //  printf("Error: %d: %f,%f,%f,%f,%f,%f\n",hhp->detInd_d[hit],ge[0],ge[1],ge[2],ge[3],ge[4],ge[5]);

    hits[helix_start].col(i) << hhp->xg_d[hit], hhp->yg_d[hit], hhp->zg_d[hit];

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
        hits_cov[helix_start](i + j * hits_in_fit, i + l * hits_in_fit) = ge[ge_idx];
      }
    }
  }
  fast_fit[helix_start] = Rfit::Fast_fit(hits[helix_start]);
}

__global__
void kernelCircleFitAllHits(GPU::SimpleVector<Quadruplet> * foundNtuplets,
    int hits_in_fit,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
  printf("blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d, helix_start: %d, cumulative_size: %d\n",
         blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif
  auto n = hits[helix_start].cols();

  Rfit::VectorNd rad = (hits[helix_start].block(0, 0, 2, n).colwise().norm());

  circle_fit[helix_start] =
      Rfit::Circle_fit(hits[helix_start].block(0, 0, 2, n),
                       hits_cov[helix_start].block(0, 0, 2 * n, 2 * n),
                       fast_fit[helix_start], rad, B, true);

#ifdef GPU_DEBUG
  printf("kernelCircleFitAllHits circle.par(0): %d %f\n", helix_start, circle_fit[helix_start].par(0));
  printf("kernelCircleFitAllHits circle.par(1): %d %f\n", helix_start, circle_fit[helix_start].par(1));
  printf("kernelCircleFitAllHits circle.par(2): %d %f\n", helix_start, circle_fit[helix_start].par(2));
#endif
}

__global__
void kernelLineFitAllHits(GPU::SimpleVector<Quadruplet> * foundNtuplets,
    float B,
    Rfit::helix_fit *results,
    Rfit::Matrix3xNd *hits,
    Rfit::Matrix3Nd *hits_cov,
    Rfit::circle_fit *circle_fit,
    Vector4d *fast_fit,
    Rfit::line_fit *line_fit)
{
  int helix_start = (blockIdx.x * blockDim.x + threadIdx.x);
  if (helix_start >= foundNtuplets->size()) {
    return;
  }

#ifdef GPU_DEBUG
  printf("blockDim.x: %d, blockIdx.x: %d, threadIdx.x: %d, helix_start: %d, cumulative_size: %d\n",
         blockDim.x, blockIdx.x, threadIdx.x, helix_start, foundNtuplets->size());
#endif

  line_fit[helix_start] = Rfit::Line_fit(hits[helix_start], hits_cov[helix_start], circle_fit[helix_start], fast_fit[helix_start], B, true);

  par_uvrtopak(circle_fit[helix_start], B, true);

  // Grab helix_fit from the proper location in the output vector
  auto & helix = results[helix_start];
  helix.par << circle_fit[helix_start].par, line_fit[helix_start].par;

  // TODO: pass properly error booleans

  helix.cov = MatrixXd::Zero(5, 5);
  helix.cov.block(0, 0, 3, 3) = circle_fit[helix_start].cov;
  helix.cov.block(3, 3, 2, 2) = line_fit[helix_start].cov;

  helix.q = circle_fit[helix_start].q;
  helix.chi2_circle = circle_fit[helix_start].chi2;
  helix.chi2_line = line_fit[helix_start].chi2;

#ifdef GPU_DEBUG
  printf("kernelLineFitAllHits line.par(0): %d %f\n", helix_start, circle_fit[helix_start].par(0));
  printf("kernelLineFitAllHits line.par(1): %d %f\n", helix_start, line_fit[helix_start].par(1));
#endif
}


void CAHitQuadrupletGeneratorGPU::launchFit(int regionIndex, HitsOnCPU const & hh, uint32_t nhits,
                                            cudaStream_t cudaStream)
{

    auto blockSize = 256;
    auto numberOfBlocks = (maxNumberOfQuadruplets_ + blockSize - 1) / blockSize;

    kernelFastFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        d_foundNtupletsVec_[regionIndex], hh.gpu_d, 4, bField_, helix_fit_resultsGPU_,
        hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
        line_fit_resultsGPU_);
    cudaCheck(cudaGetLastError());

    kernelCircleFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        d_foundNtupletsVec_[regionIndex], 4, bField_, helix_fit_resultsGPU_,
        hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
        line_fit_resultsGPU_);
    cudaCheck(cudaGetLastError());

    kernelLineFitAllHits<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
        d_foundNtupletsVec_[regionIndex], bField_, helix_fit_resultsGPU_,
           hitsGPU_, hits_covGPU_, circle_fit_resultsGPU_, fast_fit_resultsGPU_,
           line_fit_resultsGPU_);
    cudaCheck(cudaGetLastError());
}
