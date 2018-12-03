#include "RiemannFitOnGPU.h"

void RiemannFitOnGPU::allocateOnGPU(TuplesOnGPU::Container const * tuples, Rfit::helix_fit * helix_fit_results) {

  tuples_d = tuples;
  helix_fit_results_d = helix_fit_results;

  assert(tuples_d); assert(helix_fit_results_d);

  cudaCheck(cudaMalloc(&hitsGPU_, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd(3, 4))));
  cudaCheck(cudaMemset(hitsGPU_, 0x00, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3xNd(3, 4))));

  cudaCheck(cudaMalloc(&hits_covGPU_, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3Nd(12, 12))));
  cudaCheck(cudaMemset(hits_covGPU_, 0x00, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::Matrix3Nd(12, 12))));

  cudaCheck(cudaMalloc(&fast_fit_resultsGPU_, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d)));
  cudaCheck(cudaMemset(fast_fit_resultsGPU_, 0x00, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::Vector4d)));

  cudaCheck(cudaMalloc(&circle_fit_resultsGPU_, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit)));
  cudaCheck(cudaMemset(circle_fit_resultsGPU_, 0x00, 48 * maxNumberOfConcurrentFits_ * sizeof(Rfit::circle_fit)));

  cudaCheck(cudaMalloc(&line_fit_resultsGPU_, maxNumberOfConcurrentFits_ * sizeof(Rfit::line_fit)));
  cudaCheck(cudaMemset(line_fit_resultsGPU_, 0x00, maxNumberOfConcurrentFits_ * sizeof(Rfit::line_fit)));

}

void RiemannFitOnGPU::deallocateOnGPU() {

  cudaFree(hitsGPU_);
  cudaFree(hits_covGPU_);
  cudaFree(fast_fit_resultsGPU_);
  cudaFree(circle_fit_resultsGPU_);
  cudaFree(line_fit_resultsGPU_);

}



