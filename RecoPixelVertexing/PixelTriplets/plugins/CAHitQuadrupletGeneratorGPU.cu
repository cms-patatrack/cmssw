//
// Author: Felice Pantaleo, CERN
//

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"

__global__ void
kernel_connect(GPUCACell *cells, uint32_t const * nCells,
               GPU::VecArray< unsigned int, 512> *isOuterHitOfCell,
               float ptmin, 
               float region_origin_radius, const float thetaCut,
               const float phiCut, const float hardPtCut,
               unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {

  float region_origin_x =0.;
  float region_origin_y =0.;

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells) ) return;
  auto &thisCell = cells[cellIndex];
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
     auto otherCell = isOuterHitOfCell[innerHitId][j];

     if (thisCell.check_alignment_and_tag(
                 cells, otherCell, ptmin, region_origin_x, region_origin_y,
                  region_origin_radius, thetaCut, phiCut, hardPtCut)
        ) {
          cells[otherCell].theOuterNeighbors.push_back(cellIndex);
     }
  }
}

__global__ void kernel_find_ntuplets(
    GPUCACell *cells, uint32_t const * nCells,
    GPU::SimpleVector<Quadruplet> *foundNtuplets,
    unsigned int *rootLayerPairs, unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_)
{

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells) ) return;
  auto &thisCell = cells[cellIndex];
  if (thisCell.theLayerPairId!=0 && thisCell.theLayerPairId!=3 && thisCell.theLayerPairId!=8) return; // inner layer is 0 FIXME
  GPU::VecArray<unsigned int, 3> stack;
  thisCell.find_ntuplets(cells, foundNtuplets, stack, minHitsPerNtuplet);

  printf("in %d found quadruplets: %d", cellIndex, foundNtuplets->size());
}

template <int maxNumberOfDoublets_>
__global__ void
kernel_print_found_ntuplets(GPU::SimpleVector<Quadruplet> *foundNtuplets) {
  for (int i = 0; i < foundNtuplets->size(); ++i) {
    printf("\nquadruplet %d: %d %d, %d %d, %d %d\n", i,
           (*foundNtuplets)[i].hitId[0],
           (*foundNtuplets)[i].hitId[1],
           (*foundNtuplets)[i].hitId[2],
           (*foundNtuplets)[i].hitId[3],

  }
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{
  cudaFreeHost(h_indices_);
  cudaFreeHost(h_doublets_);
  cudaFreeHost(h_x_);
  cudaFreeHost(h_y_);
  cudaFreeHost(h_z_);
  cudaFreeHost(h_rootLayerPairs_);
  for (size_t i = 0; i < h_foundNtupletsVec_.size(); ++i)
  {
    cudaFreeHost(h_foundNtupletsVec_[i]);
    cudaFreeHost(h_foundNtupletsData_[i]);
    cudaFree(d_foundNtupletsVec_[i]);
    cudaFree(d_foundNtupletsData_[i]);
  }
  cudaFreeHost(tmp_layers_);
  cudaFreeHost(tmp_layerDoublets_);
  cudaFreeHost(h_layers_);

  cudaFree(d_indices_);
  cudaFree(d_doublets_);
  cudaFree(d_layers_);
  cudaFree(d_x_);
  cudaFree(d_y_);
  cudaFree(d_z_);
  cudaFree(d_rootLayerPairs_);
  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
  cudaFree(device_nCells_);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  cudaCheck(cudaMallocHost(&h_doublets_, maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets)));
  cudaCheck(cudaMallocHost(&h_indices_, maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * 2 * sizeof(int)));
  cudaCheck(cudaMallocHost(&h_x_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMallocHost(&h_y_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMallocHost(&h_z_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMallocHost(&h_rootLayerPairs_, maxNumberOfRootLayerPairs_ * sizeof(int)));

  cudaCheck(cudaMalloc(&d_indices_, maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * 2 * sizeof(int)));
  cudaCheck(cudaMalloc(&d_doublets_, maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets)));
  cudaCheck(cudaMalloc(&d_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits)));
  cudaCheck(cudaMalloc(&d_x_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMalloc(&d_y_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMalloc(&d_z_, maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(float)));
  cudaCheck(cudaMalloc(&d_rootLayerPairs_, maxNumberOfRootLayerPairs_ * sizeof(unsigned int)));

  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * sizeof(GPUCACell)));
  cudaCheck(cudaMalloc(&device_nCells_,sizeof(uint32_t)));

  cudaCheck(cudaMalloc(&device_isOuterHitOfCell_,
             maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));
  cudaCheck(cudaMemset(device_isOuterHitOfCell_, 0,
             maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));

  h_foundNtupletsVec_.resize(maxNumberOfRegions_);
  h_foundNtupletsData_.resize(maxNumberOfRegions_);
  d_foundNtupletsVec_.resize(maxNumberOfRegions_);
  d_foundNtupletsData_.resize(maxNumberOfRegions_);

  // FIXME this could be rewritten with a single pair of cudaMallocHost / cudaMalloc
  for (int i = 0; i < maxNumberOfRegions_; ++i) {
    cudaCheck(cudaMallocHost(&h_foundNtupletsData_[i],  sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMallocHost(&h_foundNtupletsVec_[i],   sizeof(GPU::SimpleVector<Quadruplet>)));
    new(h_foundNtupletsVec_[i]) GPU::SimpleVector<Quadruplet>(maxNumberOfQuadruplets_, h_foundNtupletsData_[i]);
    cudaCheck(cudaMalloc(&d_foundNtupletsData_[i],      sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMemset(d_foundNtupletsData_[i], 0x00, sizeof(Quadruplet) * maxNumberOfQuadruplets_));
    cudaCheck(cudaMalloc(&d_foundNtupletsVec_[i],       sizeof(GPU::SimpleVector<Quadruplet>)));
    GPU::SimpleVector<Quadruplet> tmp_foundNtuplets(maxNumberOfQuadruplets_, d_foundNtupletsData_[i]);
    cudaCheck(cudaMemcpy(d_foundNtupletsVec_[i], & tmp_foundNtuplets, sizeof(GPU::SimpleVector<Quadruplet>), cudaMemcpyDefault));
  }

  cudaCheck(cudaMallocHost(&tmp_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits)));
  cudaCheck(cudaMallocHost(&tmp_layerDoublets_,maxNumberOfLayerPairs_ * sizeof(GPULayerDoublets)));
  cudaCheck(cudaMallocHost(&h_layers_, maxNumberOfLayers_ * sizeof(GPULayerHits)));
}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex, cudaStream_t cudaStream)
{
  assert(regionIndex < maxNumberOfRegions_);
  dim3 numberOfBlocks_create(64, numberOfLayerPairs_);
//  dim3 numberOfBlocks_connect(32, numberOfLayerPairs_);
  dim3 numberOfBlocks_find(16, numberOfRootLayerPairs_);
  h_foundNtupletsVec_[regionIndex]->reset();
  /*
  kernel_create<<<numberOfBlocks_create, 32, 0, cudaStream>>>(
      numberOfLayerPairs_, d_doublets_, d_layers_, device_theCells_,
      device_isOuterHitOfCell_, d_foundNtupletsVec_[regionIndex],
      region.origin().x(), region.origin().y(), maxNumberOfDoublets_,
      maxNumberOfHits_);
  */

  auto numberOfBlocks = (maxNumberOfDoublets_ + 512 - 1)/512;
  kernel_connect<<<numberOfBlocks, 512, 0, cudaStream>>>(
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      region.ptMin(), 
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      maxNumberOfDoublets_, maxNumberOfHits_
  );

  kernel_find_ntuplets<<<numberOfBlocks, 512, 0, cudaStream>>>(
      device_theCells_, device_nCells_,
      d_foundNtupletsVec_[regionIndex],
      d_rootLayerPairs_, 4, maxNumberOfDoublets_);

  cudaCheck(cudaMemcpyAsync(h_foundNtupletsVec_[regionIndex], d_foundNtupletsVec_[regionIndex],
                            sizeof(GPU::SimpleVector<Quadruplet>),
                            cudaMemcpyDeviceToHost, cudaStream));

  cudaCheck(cudaMemcpyAsync(h_foundNtupletsData_[regionIndex], d_foundNtupletsData_[regionIndex],
                            maxNumberOfQuadruplets_*sizeof(Quadruplet),
                            cudaMemcpyDeviceToHost, cudaStream));

}

std::vector<std::array<int, 4>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int regionIndex, cudaStream_t cudaStream)
{
  h_foundNtupletsVec_[regionIndex]->set_data(h_foundNtupletsData_[regionIndex]);
  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaCheck(cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                            maxNumberOfLayers_ * maxNumberOfHits_ * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>),
                            cudaStream));
  std::vector<std::array<int, 4>> quadsInterface(h_foundNtupletsVec_[regionIndex]->size());
  for (int i = 0; i < h_foundNtupletsVec_[regionIndex]->size(); ++i) {
    for (int j = 0; j<4; ++j) quadsInterface[i][j] = (*h_foundNtupletsVec_[regionIndex])[i].hitId[j];
  }
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(HitsOnCPU const & hh, cudaStream_t stream) {
   auto nhits = hh.nHits;

  int threadsPerBlock = 256;
  int blocks = (3*nhits + threadsPerBlock - 1) / threadsPerBlock;

  cudaCheck(cudaMemset(device_nCells_,0,sizeof(uint32_t)));
  gpuPixelDoublets::getDoubletsFromHisto<<<blocks, threadsPerBlock, 0, stream>>>(device_theCells_,device_nCells_,hh.gpu_d, device_isOuterHitOfCell_);
}
