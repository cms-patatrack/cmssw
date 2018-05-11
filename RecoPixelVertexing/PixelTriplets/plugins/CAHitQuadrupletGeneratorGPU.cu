//
// Author: Felice Pantaleo, CERN
//

#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"


template<int maxNumberOfQuadruplets>
__global__
void kernel_create(const unsigned int numberOfLayerPairs, const GPULayerDoublets* gpuDoublets,
        const GPULayerHits* gpuHitsOnLayers, GPUCACell* cells, GPUSimpleVector<200, unsigned int> * isOuterHitOfCell,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets, const float region_origin_x,const float region_origin_y,
        unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits)
{

    unsigned int layerPairIndex = blockIdx.y;
    unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    if(cellIndexInLayerPair == 0 && layerPairIndex == 0)
    {
        foundNtuplets->reset();
    }

    if (layerPairIndex < numberOfLayerPairs)
    {
        int outerLayerId = gpuDoublets[layerPairIndex].outerLayerId;
        auto globalFirstDoubletIdx = layerPairIndex*maxNumberOfDoublets;
        auto globalFirstHitIdx = outerLayerId*maxNumberOfHits;

        for (unsigned int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto globalCellIdx = i+globalFirstDoubletIdx;
            auto& thisCell = cells[globalCellIdx];
            auto outerHitId = gpuDoublets[layerPairIndex].indices[2 * i + 1];
            thisCell.init(&gpuDoublets[layerPairIndex], gpuHitsOnLayers, layerPairIndex, globalCellIdx,
                    gpuDoublets[layerPairIndex].indices[2 * i], outerHitId, region_origin_x,  region_origin_y);

            isOuterHitOfCell[globalFirstHitIdx+outerHitId].push_back_ts(globalCellIdx);
        }
    }
}


__global__
void kernel_connect(unsigned int numberOfLayerPairs, const GPULayerDoublets* gpuDoublets, GPUCACell* cells,
        GPUSimpleVector<200, unsigned int> * isOuterHitOfCell, float ptmin, float region_origin_x, float region_origin_y, float region_origin_radius,
        const float thetaCut, const float phiCut, const float hardPtCut,
        unsigned int maxNumberOfDoublets, unsigned int maxNumberOfHits)
{
    unsigned int layerPairIndex = blockIdx.y;
    unsigned int cellIndexInLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
    if (layerPairIndex < numberOfLayerPairs)
    {
        int innerLayerId = gpuDoublets[layerPairIndex].innerLayerId;
        auto globalFirstDoubletIdx = layerPairIndex * maxNumberOfDoublets;
        auto globalFirstHitIdx = innerLayerId * maxNumberOfHits;

        for (int i = cellIndexInLayerPair; i < gpuDoublets[layerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto globalCellIdx = i + globalFirstDoubletIdx;

            auto& thisCell = cells[globalCellIdx];
            auto innerHitId = thisCell.get_inner_hit_id();
            auto numberOfPossibleNeighbors =
                    isOuterHitOfCell[globalFirstHitIdx + innerHitId].size();
            for (auto j = 0; j < numberOfPossibleNeighbors; ++j)
            {
                unsigned int otherCell = isOuterHitOfCell[globalFirstHitIdx + innerHitId].m_data[j];

                if (thisCell.check_alignment_and_tag(cells, otherCell, ptmin, region_origin_x,
                        region_origin_y, region_origin_radius, thetaCut, phiCut, hardPtCut))
                {
//                    printf("kernel_debug_connect: \t\tcell %d is outer neighbor of %d \n", globalCellIdx, otherCell);

                    cells[otherCell].theOuterNeighbors.push_back_ts(globalCellIdx);
                }
            }
        }
    }
}

template<int maxNumberOfQuadruplets>
__global__
void kernel_find_ntuplets(unsigned int numberOfRootLayerPairs,
        const GPULayerDoublets* gpuDoublets,
        GPUCACell* cells,
        GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>* foundNtuplets,
        unsigned int* rootLayerPairs,
        unsigned int minHitsPerNtuplet, unsigned int maxNumberOfDoublets)
{

    if(blockIdx.y < numberOfRootLayerPairs)
    {
        unsigned int cellIndexInRootLayerPair = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int rootLayerPairIndex = rootLayerPairs[blockIdx.y];
        auto globalFirstDoubletIdx = rootLayerPairIndex*maxNumberOfDoublets;
        GPUSimpleVector<3, unsigned int> stack;
        for (int i = cellIndexInRootLayerPair; i < gpuDoublets[rootLayerPairIndex].size;
                i += gridDim.x * blockDim.x)
        {
            auto globalCellIdx = i+globalFirstDoubletIdx;
            stack.reset();
            stack.push_back(globalCellIdx);
            cells[globalCellIdx].find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);
        }
    }
}



void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{
  cudaStreamDestroy(cudaStream_);

  cudaFreeHost(h_indices);
  cudaFreeHost(h_doublets);
  cudaFreeHost(h_x);
  cudaFreeHost(h_y);
  cudaFreeHost(h_z);
  cudaFreeHost(h_rootLayerPairs);
  cudaFreeHost(h_foundNtuplets);
  cudaFreeHost(tmp_layers);
  cudaFreeHost(tmp_layerDoublets);
  cudaFreeHost(h_layers);

  cudaFree(d_indices);
  cudaFree(d_doublets);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_rootLayerPairs);
  cudaFree(device_theCells);
  cudaFree(device_isOuterHitOfCell);
  cudaFree(d_foundNtuplets);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  cudaStreamCreateWithFlags(&cudaStream_, cudaStreamNonBlocking);
  unsigned int maxNumberOfLayerPairs = 13;
  unsigned int maxNumberOfLayers = 10;
  unsigned int maxNumberOfHits = 2000;
  unsigned int maxNumberOfRootLayerPairs = 13;

  cudaMallocHost(&h_doublets, maxNumberOfLayerPairs * sizeof(GPULayerDoublets));

  unsigned int maxNumberOfDoublets = 1;

  cudaMallocHost(&h_indices, maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(int));
  cudaMallocHost(&h_x, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_y, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_z, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMallocHost(&h_rootLayerPairs, maxNumberOfRootLayerPairs * sizeof(int));
  cudaMallocHost(&h_foundNtuplets,
                 sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>));

  cudaMalloc(&d_indices,
             maxNumberOfLayerPairs * maxNumberOfDoublets * 2 * sizeof(int));
  cudaMalloc(&d_doublets, maxNumberOfLayerPairs * sizeof(GPULayerDoublets));
  cudaMalloc(&d_layers, maxNumberOfLayers * sizeof(GPULayerHits));
  cudaMalloc(&d_x, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMalloc(&d_y, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMalloc(&d_z, maxNumberOfLayers * maxNumberOfHits * sizeof(float));
  cudaMalloc(&d_rootLayerPairs,
             maxNumberOfRootLayerPairs * sizeof(unsigned int));
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaMalloc(&device_theCells,
             maxNumberOfLayerPairs * maxNumberOfDoublets * sizeof(GPUCACell));

  cudaMalloc(&device_isOuterHitOfCell,
             maxNumberOfLayers * maxNumberOfHits *
                 sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int>));
  cudaMemset(device_isOuterHitOfCell, 0,
             maxNumberOfLayers * maxNumberOfHits *
                 sizeof(GPUSimpleVector<maxCellsPerHit, unsigned int>));

  cudaMalloc(&d_foundNtuplets,
             sizeof(GPUSimpleVector<maxNumberOfQuadruplets, Quadruplet>));

  cudaMallocHost(&tmp_layers,
                   maxNumberOfLayers * sizeof(GPULayerHits));
  cudaMallocHost(&tmp_layerDoublets,
                   maxNumberOfLayerPairs * sizeof(GPULayerDoublets));
  cudaMallocHost(&h_layers, maxNumberOfLayers * sizeof(GPULayerHits));

}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region)
{
  dim3 numberOfBlocks_create(32, numberOfLayerPairs);
  dim3 numberOfBlocks_connect(16, numberOfLayerPairs);
  dim3 numberOfBlocks_find(8, numberOfRootLayerPairs);

  kernel_create<<<numberOfBlocks_create,32,0,cudaStream_>>>(numberOfLayerPairs, d_doublets,
                          d_layers, device_theCells,
                          device_isOuterHitOfCell, d_foundNtuplets,region.origin().x(), region.origin().y(), maxNumberOfDoublets, maxNumberOfHits);

  kernel_connect<<<numberOfBlocks_connect,512,0,cudaStream_>>>(numberOfLayerPairs,
        d_doublets, device_theCells,
        device_isOuterHitOfCell, region.ptMin(), region.origin().x(), region.origin().y(), region.originRBound(),
        caThetaCut, caPhiCut,
        caHardPtCut, maxNumberOfDoublets, maxNumberOfHits);

  kernel_find_ntuplets<<<numberOfBlocks_find,1024,0,cudaStream_>>>(numberOfRootLayerPairs,
                d_doublets, device_theCells,
                d_foundNtuplets,d_rootLayerPairs, 4 , maxNumberOfDoublets);


}
