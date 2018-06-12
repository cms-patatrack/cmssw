//
// Author: Felice Pantaleo, CERN
//

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "GPUCACell.h"
#include "CAHitQuadrupletGeneratorGPU.h"

__global__ void
kernel_debug(unsigned int numberOfLayerPairs_, unsigned int numberOfLayers_,
             const GPULayerDoublets *gpuDoublets,
             const GPULayerHits *gpuHitsOnLayers, GPUCACell *cells,
             GPUSimpleVector<200, unsigned int> *isOuterHitOfCell,
             GPU::SimpleVector<Quadruplet> *foundNtuplets,
             float ptmin, float region_origin_x, float region_origin_y,
             float region_origin_radius, const float thetaCut,
             const float phiCut, const float hardPtCut,
             unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {
  if (threadIdx.x == 0 and blockIdx.x == 0)
    foundNtuplets->reset();

  printf("kernel_debug_create: theEvent contains numberOfLayerPairs_: %d\n",
         numberOfLayerPairs_);
  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs_;
       ++layerPairIndex) {

    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    int numberOfDoublets = gpuDoublets[layerPairIndex].size;
    printf(
        "kernel_debug_create: layerPairIndex: %d inner %d outer %d size %u\n",
        layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = outerLayerId * maxNumberOfHits_;
    printf("kernel_debug_create: theIdOfThefirstCellInLayerPair: %d "
           "globalFirstHitIdx %d\n",
           globalFirstDoubletIdx, globalFirstHitIdx);

    for (unsigned int i = 0; i < gpuDoublets[layerPairIndex].size; i++) {

      auto globalCellIdx = i + globalFirstDoubletIdx;
      auto &thisCell = cells[globalCellIdx];
      auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
      thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers,
                    layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,
                    region_origin_x, region_origin_y);

      isOuterHitOfCell[globalFirstHitIdx + outerHitId].push_back_ts(
          globalCellIdx);
    }
  }

  // for(unsigned int layerIndex = 0; layerIndex < numberOfLayers_;++layerIndex )
  // {
  //     auto numberOfHitsOnLayer = gpuHitsOnLayers[layerIndex].size;
  //     for(unsigned hitId = 0; hitId < numberOfHitsOnLayer; hitId++)
  //     {
  //
  //         if(isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].size()>0)
  //         {
  //             printf("\nlayer %d hit %d is outer hit of %d
  //             cells\n",layerIndex, hitId,
  //             isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].size());
  //             printf("\n\t%f %f %f
  //             \n",gpuHitsOnLayers[layerIndex].x[hitId],gpuHitsOnLayers[layerIndex].y[hitId],gpuHitsOnLayers[layerIndex].z[hitId]);
  //
  //             for(unsigned cell = 0; cell<
  //             isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].size();
  //             cell++)
  //             {
  //                 printf("cell %d\n",
  //                 isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].m_data[cell]);
  //                 auto& thisCell =
  //                 cells[isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].m_data[cell]];
  //                             float x1, y1, z1, x2, y2, z2;
  //
  //                             x1 = thisCell.get_inner_x();
  //                             y1 = thisCell.get_inner_y();
  //                             z1 = thisCell.get_inner_z();
  //                             x2 = thisCell.get_outer_x();
  //                             y2 = thisCell.get_outer_y();
  //                             z2 = thisCell.get_outer_z();
  //                 printf("\n\tDEBUG cellid %d innerhit outerhit (xyz) (%f %f
  //                 %f), (%f %f
  //                 %f)\n",isOuterHitOfCell[layerIndex*maxNumberOfHits_+hitId].m_data[cell],
  //                 x1,y1,z1,x2,y2,z2);
  //             }
  //         }
  //     }
  // }

  // starting connect

  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs_;
       ++layerPairIndex) {

    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    int numberOfDoublets = gpuDoublets[layerPairIndex].size;
    printf("kernel_debug_connect: connecting layerPairIndex: %d inner %d outer "
           "%d size %u\n",
           layerPairIndex, innerLayerId, outerLayerId, numberOfDoublets);

    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = innerLayerId * maxNumberOfHits_;
    //        printf("kernel_debug_connect: theIdOfThefirstCellInLayerPair: %d
    //        globalFirstHitIdx %d\n", globalFirstDoubletIdx,
    //        globalFirstHitIdx);

    for (unsigned int i = 0; i < numberOfDoublets; i++) {

      auto globalCellIdx = i + globalFirstDoubletIdx;

      auto &thisCell = cells[globalCellIdx];
      auto innerHitId = thisCell.get_inner_hit_id();
      auto numberOfPossibleNeighbors =
          isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
      //            if(numberOfPossibleNeighbors>0)
      //            printf("kernel_debug_connect: cell: %d has %d possible
      //            neighbors\n", globalCellIdx, numberOfPossibleNeighbors);
      float x1, y1, z1, x2, y2, z2;

      x1 = thisCell.get_inner_x();
      y1 = thisCell.get_inner_y();
      z1 = thisCell.get_inner_z();
      x2 = thisCell.get_outer_x();
      y2 = thisCell.get_outer_y();
      z2 = thisCell.get_outer_z();
      printf("\n\n\nDEBUG cellid %d innerhit outerhit (xyz) (%f %f %f), (%f %f "
             "%f)\n",
             globalCellIdx, x1, y1, z1, x2, y2, z2);

      for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
        unsigned int otherCell =
            isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];

        float x3, y3, z3, x4, y4, z4;
        x3 = cells[otherCell].get_inner_x();
        y3 = cells[otherCell].get_inner_y();
        z3 = cells[otherCell].get_inner_z();
        x4 = cells[otherCell].get_outer_x();
        y4 = cells[otherCell].get_outer_y();
        z4 = cells[otherCell].get_outer_z();

        printf("kernel_debug_connect: checking compatibility with %d \n",
               otherCell);
        printf("DEBUG \tinnerhit outerhit (xyz) (%f %f %f), (%f %f %f)\n", x3,
               y3, z3, x4, y4, z4);

        if (thisCell.check_alignment_and_tag(
                cells, otherCell, ptmin, region_origin_x, region_origin_y,
                region_origin_radius, thetaCut, phiCut, hardPtCut)) {

          printf("kernel_debug_connect: \t\tcell %d is outer neighbor of %d \n",
                 globalCellIdx, otherCell);

          cells[otherCell].theOuterNeighbors.push_back_ts(globalCellIdx);
        }
      }
    }
  }
}

__global__ void debug_input_data(unsigned int numberOfLayerPairs_,
                                 const GPULayerDoublets *gpuDoublets,
                                 const GPULayerHits *gpuHitsOnLayers,
                                 float ptmin, float region_origin_x,
                                 float region_origin_y,
                                 float region_origin_radius,
                                 unsigned int maxNumberOfHits_) {
  printf("GPU: Region ptmin %f , region_origin_x %f , region_origin_y %f , "
         "region_origin_radius  %f \n",
         ptmin, region_origin_x, region_origin_y, region_origin_radius);
  printf("GPU: numberOfLayerPairs_: %d\n", numberOfLayerPairs_);

  for (unsigned int layerPairIndex = 0; layerPairIndex < numberOfLayerPairs_;
       ++layerPairIndex) {
    printf("\t numberOfDoublets: %d \n", gpuDoublets[layerPairIndex].size);
    printf("\t innerLayer: %d outerLayer: %d \n",
           gpuDoublets[layerPairIndex].innerLayerId,
           gpuDoublets[layerPairIndex].outerLayerId);

    for (unsigned int cellIndexInLayerPair = 0;
         cellIndexInLayerPair < gpuDoublets[layerPairIndex].size;
         ++cellIndexInLayerPair) {

      if (cellIndexInLayerPair < 5) {
        auto innerhit =
            gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair];
        auto innerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .x[innerhit];
        auto innerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .y[innerhit];
        auto innerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].innerLayerId]
                          .z[innerhit];

        auto outerhit =
            gpuDoublets[layerPairIndex].indices[2 * cellIndexInLayerPair + 1];
        auto outerX = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .x[outerhit];
        auto outerY = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .y[outerhit];
        auto outerZ = gpuHitsOnLayers[gpuDoublets[layerPairIndex].outerLayerId]
                          .z[outerhit];
        printf("\t \t %d innerHit: %d %f %f %f outerHit: %d %f %f %f\n",
               cellIndexInLayerPair, innerhit, innerX, innerY, innerZ, outerhit,
               outerX, outerY, outerZ);
      }
    }
  }
}

template <int maxNumberOfQuadruplets_>
__global__ void kernel_debug_find_ntuplets(
    unsigned int numberOfRootLayerPairs_, const GPULayerDoublets *gpuDoublets,
    GPUCACell *cells,
    GPUSimpleVector<maxNumberOfQuadruplets_, Quadruplet> *foundNtuplets,
    unsigned int *rootLayerPairs, unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_) {
  printf("numberOfRootLayerPairs_ = %d", numberOfRootLayerPairs_);
  for (int rootLayerPair = 0; rootLayerPair < numberOfRootLayerPairs_;
       ++rootLayerPair) {
    unsigned int rootLayerPairIndex = rootLayerPairs[rootLayerPair];
    auto globalFirstDoubletIdx = rootLayerPairIndex * maxNumberOfDoublets_;

    GPUSimpleVector<3, unsigned int> stack;
    for (int i = 0; i < gpuDoublets[rootLayerPairIndex].size; i++) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      stack.reset();
      stack.push_back(globalCellIdx);
      cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack,
                                         minHitsPerNtuplet);
    }
    printf("found quadruplets: %d", foundNtuplets->size());
  }
}

__global__ void kernel_create(
    const unsigned int numberOfLayerPairs_, const GPULayerDoublets *gpuDoublets,
    const GPULayerHits *gpuHitsOnLayers, GPUCACell *cells,
    GPUSimpleVector<200, unsigned int> *isOuterHitOfCell,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    const float region_origin_x, const float region_origin_y,
    unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {

  unsigned int layerPairIndex = blockIdx.y;
  unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndexInLayerPair == 0 && layerPairIndex == 0) {
    foundNtuplets->reset();
  }

  if (layerPairIndex < numberOfLayerPairs_) {
    int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = outerLayerId * maxNumberOfHits_;

    for (unsigned int i = cellIndexInLayerPair;
         i < gpuDoublets[layerPairIndex].size; i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      auto &thisCell = cells[globalCellIdx];
      auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
      thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers,
                    layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId,
                    region_origin_x, region_origin_y);

      isOuterHitOfCell[globalFirstHitIdx + outerHitId].push_back_ts(
          globalCellIdx);
    }
  }
}

__global__ void
kernel_connect(unsigned int numberOfLayerPairs_,
               const GPULayerDoublets *gpuDoublets, GPUCACell *cells,
               GPUSimpleVector<200, unsigned int> *isOuterHitOfCell,
               float ptmin, float region_origin_x, float region_origin_y,
               float region_origin_radius, const float thetaCut,
               const float phiCut, const float hardPtCut,
               unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {
  unsigned int layerPairIndex = blockIdx.y;
  unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
  if (layerPairIndex < numberOfLayerPairs_) {
    int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
    auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets_;
    auto globalFirstHitIdx = innerLayerId * maxNumberOfHits_;

    for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
         i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;

      auto &thisCell = cells[globalCellIdx];
      auto innerHitId = thisCell.get_inner_hit_id();
      auto numberOfPossibleNeighbors =
          isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
      for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
        unsigned int otherCell =
            isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];

        if (thisCell.check_alignment_and_tag(
                cells, otherCell, ptmin, region_origin_x, region_origin_y,
                region_origin_radius, thetaCut, phiCut, hardPtCut)) {
          cells[otherCell].theOuterNeighbors.push_back_ts(globalCellIdx);
        }
      }
    }
  }
}

__global__ void kernel_find_ntuplets(
    unsigned int numberOfRootLayerPairs_, const GPULayerDoublets *gpuDoublets,
    GPUCACell *cells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int *rootLayerPairs, unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_) {

  if (blockIdx.y < numberOfRootLayerPairs_) {
    unsigned int cellIndexInRootLayerPair =
        threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int rootLayerPairIndex = rootLayerPairs[blockIdx.y];
    auto globalFirstDoubletIdx = rootLayerPairIndex * maxNumberOfDoublets_;
    GPUSimpleVector<3, unsigned int> stack;
    for (int i = cellIndexInRootLayerPair;
         i < gpuDoublets[rootLayerPairIndex].size;
         i += gridDim.x * blockDim.x) {
      auto globalCellIdx = i + globalFirstDoubletIdx;
      stack.reset();
      stack.push_back(globalCellIdx);
      cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack,
                                         minHitsPerNtuplet);
    }
  }
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU() {

  cudaFreeHost(h_indices_);
  cudaFreeHost(h_doublets_);
  cudaFreeHost(h_x_);
  cudaFreeHost(h_y_);
  cudaFreeHost(h_z_);
  cudaFreeHost(h_rootLayerPairs_);
  for (int i = 0; i < maxNumberOfRegions_; ++i)
  {
    cudaFreeHost(h_foundNtupletsVec_[i]);
    cudaFreeHost(h_foundNtupletsData_[i]);
    cudaFreeHost(tmp_foundNtupletsVec_[i]);
    cudaFree(d_foundNtupletsVec_[i]);
    cudaFree(d_foundNtupletsData_[i]);
  }
  cudaFreeHost(tmp_layers_);
  cudaFreeHost(tmp_layerDoublets_);
  cudaFreeHost(h_layers_);

  cudaFree(d_indices_);
  cudaFree(d_doublets_);
  cudaFree(d_x_);
  cudaFree(d_y_);
  cudaFree(d_z_);
  cudaFree(d_rootLayerPairs_);
  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU() {
  cudaCheck(cudaMallocHost(&h_doublets_, maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets)));

  cudaMallocHost(&h_indices_,
                 maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * 2 * sizeof(int));
  cudaMallocHost(&h_x_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float));
  cudaMallocHost(&h_y_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float));
  cudaMallocHost(&h_z_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float));
  cudaMallocHost(&h_rootLayerPairs_, maxNumberOfRootLayerPairs_ * sizeof(int));

  cudaMalloc(&d_indices_,
             maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * 2 * sizeof(int));
  cudaMalloc(&d_doublets_, maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets));
  cudaMalloc(&d_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits));
  cudaMalloc(&d_x_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float));
  cudaMalloc(&d_y_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float));
  cudaMalloc(&d_z_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float));
  cudaMalloc(&d_rootLayerPairs_,
             maxNumberOfRootLayerPairs_ * sizeof(unsigned int));
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * sizeof(GPUCACell)));

  cudaCheck(cudaMalloc(&device_isOuterHitOfCell_,
             maxNumberOfLayers_ * maxNumberOfHits_ *
                 sizeof(GPUSimpleVector<maxCellsPerHit_, unsigned int>)));
  cudaCheck(cudaMemset(device_isOuterHitOfCell_, 0,
             maxNumberOfLayers_ * maxNumberOfHits_ *
                 sizeof(GPUSimpleVector<maxCellsPerHit_, unsigned int>)));

  h_foundNtupletsVec_.resize(maxNumberOfRegions_);
  h_foundNtupletsData_.resize(maxNumberOfRegions_);
  d_foundNtupletsVec_.resize(maxNumberOfRegions_);
  d_foundNtupletsData_.resize(maxNumberOfRegions_);
  tmp_foundNtupletsVec_.resize(maxNumberOfRegions_);

  for (int i = 0; i < maxNumberOfRegions_; ++i) {
    cudaCheck(cudaMalloc(&d_foundNtupletsVec_[i],
               sizeof(GPU::SimpleVector<Quadruplet>)));
    cudaCheck(cudaMalloc(&d_foundNtupletsData_[i], sizeof(Quadruplet)*maxNumberOfQuadruplets_));
    cudaCheck(cudaMallocHost(&h_foundNtupletsVec_[i],
                   sizeof(GPU::SimpleVector<Quadruplet>)));
    cudaCheck(cudaMallocHost(&h_foundNtupletsData_[i], sizeof(Quadruplet)*maxNumberOfQuadruplets_));
    cudaCheck(cudaMallocHost(&tmp_foundNtupletsVec_[i],
                   sizeof(GPU::SimpleVector<Quadruplet>)));
    new (h_foundNtupletsVec_[i]) GPU::SimpleVector<Quadruplet>(maxNumberOfQuadruplets_, h_foundNtupletsData_[i]);
    new (tmp_foundNtupletsVec_[i]) GPU::SimpleVector<Quadruplet>(maxNumberOfQuadruplets_, d_foundNtupletsData_[i]);

    cudaMemcpy(d_foundNtupletsVec_[i], tmp_foundNtupletsVec_[i], sizeof(GPU::SimpleVector<Quadruplet>), cudaMemcpyDefault);

  }

  cudaMallocHost(&tmp_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits));
  cudaMallocHost(&tmp_layerDoublets_,
                 maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets));
  cudaMallocHost(&h_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits));
}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex) {

  assert(regionIndex < maxNumberOfRegions_);
  dim3 numberOfBlocks_create(32, numberOfLayerPairs_);
  dim3 numberOfBlocks_connect(16, numberOfLayerPairs_);
  dim3 numberOfBlocks_find(8, numberOfRootLayerPairs_);
  h_foundNtupletsVec_[regionIndex]->reset();
  kernel_create<<<numberOfBlocks_create, 32, 0, cudaStream_>>>(
      numberOfLayerPairs_, d_doublets_, d_layers_, device_theCells_,
      device_isOuterHitOfCell_, d_foundNtupletsVec_[regionIndex],
      region.origin().x(), region.origin().y(), maxNumberOfDoublets_,
      maxNumberOfHits_);

  kernel_connect<<<numberOfBlocks_connect, 512, 0, cudaStream_>>>(
      numberOfLayerPairs_, d_doublets_, device_theCells_,
      device_isOuterHitOfCell_,
      region.ptMin(), region.origin().x(), region.origin().y(),
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      maxNumberOfDoublets_, maxNumberOfHits_);

  kernel_find_ntuplets<<<numberOfBlocks_find, 1024, 0, cudaStream_>>>(
      numberOfRootLayerPairs_, d_doublets_, device_theCells_,
      d_foundNtupletsVec_[regionIndex],
      d_rootLayerPairs_, 4, maxNumberOfDoublets_);

  cudaMemcpyAsync(h_foundNtupletsVec_[regionIndex], d_foundNtupletsVec_[regionIndex],
                  sizeof(GPU::SimpleVector<Quadruplet>),
                  cudaMemcpyDeviceToHost, cudaStream_);
  cudaMemcpyAsync(h_foundNtupletsData_[regionIndex], d_foundNtupletsData_[regionIndex],
                                  h_foundNtupletsVec_[regionIndex]->size()*sizeof(Quadruplet),
                                  cudaMemcpyDeviceToHost, cudaStream_);

}

std::vector<std::array<std::pair<int, int>, 3>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int regionIndex) {
  //this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                  maxNumberOfLayers_ * maxNumberOfHits_ *
                      sizeof(GPUSimpleVector<maxCellsPerHit_, unsigned int>),
                  cudaStream_);

  std::vector<std::array<std::pair<int, int>, 3>> quadsInterface;
  h_foundNtupletsVec_[regionIndex]->set_data(h_foundNtupletsData_[regionIndex]);
  for (int i = 0; i < h_foundNtupletsVec_[regionIndex]->size(); ++i) {
    std::array<std::pair<int, int>, 3> tmpQuad = {
        {std::make_pair((*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[0].x,
                        (*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[0].y -
                            maxNumberOfDoublets_ *
                                (*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[0].x),
         std::make_pair((*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[1].x,
                        (*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[1].y -
                            maxNumberOfDoublets_ *
                                (*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[1].x),
         std::make_pair((*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[2].x,
                        (*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[2].y -
                            maxNumberOfDoublets_ *
                                (*h_foundNtupletsVec_[regionIndex])[i].layerPairsAndCellId[2].x)}};
    quadsInterface.push_back(tmpQuad);
  }
  std::cout << h_foundNtupletsVec_[regionIndex]->size() << std::endl;
  return quadsInterface;
}
