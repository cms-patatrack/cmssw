#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h

#include <cstdint>

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/SimpleVector.h"
#include "HeterogeneousCore/CUDAUtilities/interface/VecArray.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

// #define ONLY_PHICUT
#define DENSE_EVENTS

namespace CAConstants {

  // constants
#ifdef DENSE_EVENTS
  constexpr uint32_t maxNumberOfTuples() { return 256 * 1024; }
  constexpr uint32_t maxNumberOfDoublets() { return 4 * 1024 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128 * 4; } 
  constexpr uint32_t maxNumOfActiveDoublets() { return maxNumberOfDoublets() / 32; }
#else
#ifndef ONLY_PHICUT
#ifdef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumberOfTuples() { return 3 * 1024; }
#else
  constexpr uint32_t maxNumberOfTuples() { return 256 * 1024; }
#endif
#else
  constexpr uint32_t maxNumberOfTuples() { return 128 * 1024; }
#endif
 
#ifndef ONLY_PHICUT
#ifndef GPU_SMALL_EVENTS
  constexpr uint32_t maxNumberOfDoublets() { return 8 * 1024 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128 * 8; }
#else
  constexpr uint32_t maxNumberOfDoublets() { return 128 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 128 / 2; }
#endif
#else
  constexpr uint32_t maxNumberOfDoublets() { return 2 * 1024 * 1024; }
  constexpr uint32_t maxCellsPerHit() { return 12 * 128; }
#endif
  constexpr uint32_t maxNumOfActiveDoublets() { return maxNumberOfDoublets() / 32; }
#endif
  
  constexpr uint32_t maxNumberOfQuadruplets() { return maxNumberOfTuples(); }
  constexpr uint32_t maxNumberOfLayerPairs() { return 64; }
  constexpr uint32_t maxNumberOfLayers() { return 28; }
  constexpr uint32_t maxTuples() { return maxNumberOfTuples(); }

  // types
  using hindex_type = uint32_t;  // FIXME from siPixelRecHitsHeterogeneousProduct
  using tindex_type = uint32_t;  //  for tuples

#ifdef DENSE_EVENTS
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 64>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 96>;  
#else
#ifndef ONLY_PHICUT
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 36>;
  using CellTracks = cms::cuda::VdecArray<tindex_type, 48>;
#else
  using CellNeighbors = cms::cuda::VecArray<uint32_t, 64>;
  using CellTracks = cms::cuda::VecArray<tindex_type, 64>;
#endif
#endif

  using CellNeighborsVector = cms::cuda::SimpleVector<CellNeighbors>;
  using CellTracksVector = cms::cuda::SimpleVector<CellTracks>;

  using OuterHitOfCell = cms::cuda::VecArray<uint32_t, maxCellsPerHit()>;
#ifdef DENSE_EVENTS
  using TuplesContainer = cms::cuda::OneToManyAssoc<hindex_type, maxTuples(), 8 * maxTuples()>;
  using HitToTuple =
      cms::cuda::OneToManyAssoc<tindex_type, pixelGPUConstants::maxNumberOfHits, 12 * maxTuples()>;  // 3.5 should be enough
  using TupleMultiplicity = cms::cuda::OneToManyAssoc<tindex_type, 16, maxTuples()>;
#else
  using TuplesContainer = cms::cuda::OneToManyAssoc<hindex_type, maxTuples(), 5 * maxTuples()>;
  using HitToTuple = cms::cuda::OneToManyAssoc<tindex_type, pixelGPUConstants::maxNumberOfHits, 4 * maxTuples()>;  // 3.5 should be enough
  using TupleMultiplicity = cms::cuda::OneToManyAssoc<tindex_type, 8, maxTuples()>;
#endif 
}  // namespace CAConstants

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAConstants_h
