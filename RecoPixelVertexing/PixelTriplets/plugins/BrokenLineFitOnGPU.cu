#include "BrokenLineFitOnGPU.h"

void HelixFitOnGPU::launchBrokenLineKernels(HitsView const * hv,
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
                                                                      hv,
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
                                                                      hv,
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
                                                                        hv,
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
                                                                        hv,
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
