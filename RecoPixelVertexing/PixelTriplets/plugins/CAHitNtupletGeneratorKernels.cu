#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"

#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"


template <>
void CAHitNtupletGeneratorKernelsGPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, cudaStream_t cudaStream) {
  auto blockSize = 128;
  int numberOfBlocks = (HitContainer::capacity() + blockSize - 1) / blockSize;

  cudautils::launch(kernel_fillHitDetIndices,{numberOfBlocks, blockSize, 0, cudaStream},
      &tracks_d->hitIndices, hv, &tracks_d->detIndices);
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  // zero tuples
  cudautils::launchZero(tuples_d, cudaStream);

  auto nhits = hh.nHits();
  assert(nhits <= pixelGPUConstants::maxNumberOfHits);

  // std::cout << "N hits " << nhits << std::endl;
  // if (nhits<2) std::cout << "too few hits " << nhits << std::endl;

  //
  // applying conbinatoric cleaning such as fishbone at this stage is too expensive
  //

  auto nthTot = 64;
  auto stride = 4;
  auto blockSize = nthTot / stride;
  int numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  auto rescale = numberOfBlocks / 65536;
  blockSize *= (rescale + 1);
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  assert(numberOfBlocks < 65536);
  assert(blockSize > 0 && 0 == blockSize % 16);
  dim3 blks(1, numberOfBlocks, 1);
  dim3 thrs(stride, blockSize, 1);

  cudautils::launch(kernel_connect,{blks, thrs, 0, cudaStream},
      device_hitTuple_apc_,
      device_hitToTuple_apc_,  // needed only to be reset, ready for next kernel
      hh.view(),
      device_theCells_.get(),
      device_nCells_,
      device_theCellNeighbors_,
      device_isOuterHitOfCell_.get(),
      m_params.hardCurvCut_,
      m_params.ptmin_,
      m_params.CAThetaCutBarrel_,
      m_params.CAThetaCutForward_,
      m_params.dcaCutInnerTriplet_,
      m_params.dcaCutOuterTriplet_);

  if (nhits > 1 && m_params.earlyFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    int blockSize = nthTot / stride;
    int numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    cudautils::launch(fishbone,{blks, thrs, 0, cudaStream},
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, false);
  }

  blockSize = 64;
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  cudautils::launch(kernel_find_ntuplets,{numberOfBlocks, blockSize, 0, cudaStream},hh.view(),
                                                                     device_theCells_.get(),
                                                                     device_nCells_,
                                                                     device_theCellTracks_,
                                                                     tuples_d,
                                                                     device_hitTuple_apc_,
                                                                     quality_d,
                                                                     m_params.minHitsPerNtuplet_);

  if (m_params.doStats_)
    cudautils::launch(kernel_mark_used,{numberOfBlocks, blockSize, 0, cudaStream},hh.view(), device_theCells_.get(), device_nCells_);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
  blockSize = 128;
  numberOfBlocks = (HitContainer::totbins() + blockSize - 1) / blockSize;
//  cudautils::launch(cudautils::finalizeBulk<decltype(*tuples_d)>,{numberOfBlocks, blockSize, 0, cudaStream},device_hitTuple_apc_, tuples_d);
  cudautils::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(device_hitTuple_apc_, tuples_d);


  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  cudautils::launch(kernel_earlyDuplicateRemover,{numberOfBlocks, blockSize, 0, cudaStream},
      device_theCells_.get(), device_nCells_, tuples_d, quality_d);

  blockSize = 128;
  numberOfBlocks = (3 * CAConstants::maxTuples() / 4 + blockSize - 1) / blockSize;
  cudautils::launch(kernel_countMultiplicity,{numberOfBlocks, blockSize, 0, cudaStream},
      tuples_d, quality_d, device_tupleMultiplicity_.get());
  cudautils::launchFinalize(device_tupleMultiplicity_.get(), device_tmws_, cudaStream);
  cudautils::launch(kernel_fillMultiplicity,{numberOfBlocks, blockSize, 0, cudaStream},
      tuples_d, quality_d, device_tupleMultiplicity_.get());

  if (nhits > 1 && m_params.lateFishbone_) {
    auto nthTot = 128;
    auto stride = 16;
    auto blockSize = nthTot / stride;
    auto numberOfBlocks = (nhits + blockSize - 1) / blockSize;
    dim3 blks(1, numberOfBlocks, 1);
    dim3 thrs(stride, blockSize, 1);
    cudautils::launch(fishbone,{blks, thrs, 0, cudaStream},
        hh.view(), device_theCells_.get(), device_nCells_, device_isOuterHitOfCell_.get(), nhits, true);
  }

  if (m_params.doStats_) {
    numberOfBlocks = (std::max(nhits, m_params.maxNumberOfDoublets_) + blockSize - 1) / blockSize;
    cudautils::launch(kernel_checkOverflows,{numberOfBlocks, blockSize, 0, cudaStream},tuples_d,
                                                                        device_tupleMultiplicity_.get(),
                                                                        device_hitTuple_apc_,
                                                                        device_theCells_.get(),
                                                                        device_nCells_,
                                                                        device_theCellNeighbors_,
                                                                        device_theCellTracks_,
                                                                        device_isOuterHitOfCell_.get(),
                                                                        nhits,
                                                                        m_params.maxNumberOfDoublets_,
                                                                        counters_);
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::buildDoublets(HitsOnCPU const &hh, cudaStream_t stream) {
  auto nhits = hh.nHits();

#ifdef NTUPLE_DEBUG
  std::cout << "building Doublets out of " << nhits << " Hits" << std::endl;
#endif

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  // in principle we can use "nhits" to heuristically dimension the workspace...
  device_isOuterHitOfCell_ = cudautils::make_device_unique<GPUCACell::OuterHitOfCell[]>(std::max(1U, nhits), stream);
  assert(device_isOuterHitOfCell_.get());
  {
    int threadsPerBlock = 128;
    // at least one block!
    int blocks = (std::max(1U, nhits) + threadsPerBlock - 1) / threadsPerBlock;
    cudautils::launch(gpuPixelDoublets::initDoublets,{blocks, threadsPerBlock, 0, stream},device_isOuterHitOfCell_.get(),
                                                                           nhits,
                                                                           device_theCellNeighbors_,
                                                                           device_theCellNeighborsContainer_.get(),
                                                                           device_theCellTracks_,
                                                                           device_theCellTracksContainer_.get());
  }

  device_theCells_ = cudautils::make_device_unique<GPUCACell[]>(m_params.maxNumberOfDoublets_, stream);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

  if (0 == nhits)
    return;  // protect against empty events

  // FIXME avoid magic numbers
  auto nActualPairs = gpuPixelDoublets::nPairs;
  if (!m_params.includeJumpingForwardDoublets_)
    nActualPairs = 15;
  if (m_params.minHitsPerNtuplet_ > 3) {
    nActualPairs = 13;
  }

  assert(nActualPairs <= gpuPixelDoublets::nPairs);
  int stride = 4;
  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize / stride;
  int blocks = (4 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blks(1, blocks, 1);
  dim3 thrs(stride, threadsPerBlock, 1);
  cudautils::launch(gpuPixelDoublets::getDoubletsFromHisto,{blks, thrs, 0, stream},device_theCells_.get(),
                                                                    device_nCells_,
                                                                    device_theCellNeighbors_,
                                                                    device_theCellTracks_,
                                                                    hh.view(),
                                                                    device_isOuterHitOfCell_.get(),
                                                                    nActualPairs,
                                                                    m_params.idealConditions_,
                                                                    m_params.doClusterCut_,
                                                                    m_params.doZ0Cut_,
                                                                    m_params.doPtCut_,
                                                                    m_params.maxNumberOfDoublets_);

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream) {
  // these are pointer on GPU!
  auto const *tuples_d = &tracks_d->hitIndices;
  auto *quality_d = (Quality *)(&tracks_d->m_quality);

  int blockSize = 64;

  // classify tracks based on kinematics
  int numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
  cudautils::launch(kernel_classifyTracks,{numberOfBlocks, blockSize, 0, cudaStream},tuples_d, tracks_d, m_params.cuts_, quality_d);

  if (m_params.lateFishbone_) {
    // apply fishbone cleaning to good tracks
    numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
    cudautils::launch(kernel_fishboneCleaner,{numberOfBlocks, blockSize, 0, cudaStream},
        device_theCells_.get(), device_nCells_, quality_d);
  }

  // remove duplicates (tracks that share a doublet)
  numberOfBlocks = (3 * m_params.maxNumberOfDoublets_ / 4 + blockSize - 1) / blockSize;
  cudautils::launch(kernel_fastDuplicateRemover,{numberOfBlocks, blockSize, 0, cudaStream},
      device_theCells_.get(), device_nCells_, tuples_d, tracks_d);

  if (m_params.minHitsPerNtuplet_ < 4 || m_params.doStats_) {
    // fill hit->track "map"
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    cudautils::launch(kernel_countHitInTracks,{numberOfBlocks, blockSize, 0, cudaStream},
        tuples_d, quality_d, device_hitToTuple_.get());
    cudautils::launchFinalize(device_hitToTuple_.get(), device_tmws_, cudaStream);
    cudaCheck(cudaGetLastError());
    cudautils::launch(kernel_fillHitInTracks,{numberOfBlocks, blockSize, 0, cudaStream},tuples_d, quality_d, device_hitToTuple_.get());
  }
  if (m_params.minHitsPerNtuplet_ < 4) {
    // remove duplicates (tracks that share a hit)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    cudautils::launch(kernel_tripletCleaner,{numberOfBlocks, blockSize, 0, cudaStream},
        hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get());
  }

  if (m_params.doStats_) {
    // counters (add flag???)
    numberOfBlocks = (HitToTuple::capacity() + blockSize - 1) / blockSize;
    cudautils::launch(kernel_doStatsForHitInTracks,{numberOfBlocks, blockSize, 0, cudaStream},device_hitToTuple_.get(), counters_);
    numberOfBlocks = (3 * CAConstants::maxNumberOfQuadruplets() / 4 + blockSize - 1) / blockSize;
    cudautils::launch(kernel_doStatsForTracks,{numberOfBlocks, blockSize, 0, cudaStream},tuples_d, quality_d, counters_);
  }
#ifdef GPU_DEBUG
  cudaDeviceSynchronize();
  cudaCheck(cudaGetLastError());
#endif

#ifdef DUMP_GPU_TK_TUPLES
  static std::atomic<int> iev(0);
  ++iev;
  cudautils::launch(kernel_print_found_ntuplets,{1, 32, 0, cudaStream},
      hh.view(), tuples_d, tracks_d, quality_d, device_hitToTuple_.get(), 100, iev);
#endif
}

template <>
void CAHitNtupletGeneratorKernelsGPU::printCounters(Counters const *counters) {
  cudautils::launch(kernel_printCounters,{1, 1},counters);
}
