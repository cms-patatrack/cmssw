//
// Author: Felice Pantaleo, CERN
//

#include <cstdint>
#include <cuda_runtime.h>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforGPU.h"
#include "CAHitQuadrupletGeneratorGPU.h"
#include "GPUCACell.h"
#include "gpuPixelDoublets.h"
#include"gpuFishbone.h"
using namespace gpuPixelDoublets;

using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;
using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

__global__
void kernel_checkOverflows(TuplesOnGPU::Container * foundNtuplets,
               GPUCACell const * __restrict__ cells, uint32_t const * __restrict__ nCells,
               GPUCACell::OuterHitOfCell const * __restrict__ isOuterHitOfCell,
               uint32_t nHits, uint32_t maxNumberOfDoublets) {

 __shared__ uint32_t killedCell;
 killedCell=0;
 __syncthreads();
  
 auto idx = threadIdx.x + blockIdx.x * blockDim.x;
 #ifdef GPU_DEBUG
 if (0==idx)
   printf("number of found cells %d\n",*nCells);
 #endif
 if (idx < (*nCells) ) {
   auto &thisCell = cells[idx];
   if (thisCell.theOuterNeighbors.full()) //++tooManyNeighbors[thisCell.theLayerPairId];
     printf("OuterNeighbors overflow %d in %d\n", idx, thisCell.theLayerPairId);
   if (thisCell.theDoubletId<0) atomicInc(&killedCell,maxNumberOfDoublets);
 }
 if (idx < nHits) {
   if (isOuterHitOfCell[idx].full()) // ++tooManyOuterHitOfCell;
     printf("OuterHitOfCell overflow %d\n", idx);
 }

 __syncthreads();
// if (threadIdx.x==0) printf("number of killed cells %d\n",killedCell);
}


__global__ 
void
kernel_connect(AtomicPairCounter * apc,
               GPUCACell::Hits const *  __restrict__ hhp,
               GPUCACell * cells, uint32_t const * __restrict__ nCells,
               GPUCACell::OuterHitOfCell const * __restrict__ isOuterHitOfCell,
               float ptmin,
               float region_origin_radius, const float thetaCut,
               const float phiCut, const float hardPtCut,
               unsigned int maxNumberOfDoublets_, unsigned int maxNumberOfHits_) {

  auto const & hh = *hhp;

  // 87 cm/GeV = 1/(3.8T * 0.3)
  // take less than radius given by the hardPtCut and reject everything below
  // auto hardCurvCut = 1.f/(hardPtCut * 87.f);
  constexpr auto hardCurvCut = 1.f/(0.35f * 87.f); // VI tune

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (0==cellIndex) (*apc)=0; // ready for next kernel

  if (cellIndex >= (*nCells) ) return;
  auto const & thisCell = cells[cellIndex];
  if (thisCell.theDoubletId<0) return;
  auto innerHitId = thisCell.get_inner_hit_id();
  auto numberOfPossibleNeighbors = isOuterHitOfCell[innerHitId].size();
  auto vi = isOuterHitOfCell[innerHitId].data();
  for (auto j = 0; j < numberOfPossibleNeighbors; ++j) {
     auto otherCell = __ldg(vi+j);
     if (cells[otherCell].theDoubletId<0) continue;
     if (thisCell.check_alignment(hh,
                 cells[otherCell], ptmin,
                  region_origin_radius+phiCut, thetaCut, hardCurvCut)
        ) {
          cells[otherCell].theOuterNeighbors.push_back(cellIndex);
     }
  }
}

__global__ 
void kernel_find_ntuplets(
    GPUCACell * const __restrict__ cells, uint32_t const * nCells,
    TuplesOnGPU::Container * foundNtuplets, AtomicPairCounter * apc,
    unsigned int minHitsPerNtuplet,
    unsigned int maxNumberOfDoublets_)
{

  auto cellIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (cellIndex >= (*nCells) ) return;
  auto &thisCell = cells[cellIndex];
  if (thisCell.theLayerPairId!=0 && thisCell.theLayerPairId!=3 && thisCell.theLayerPairId!=8) return; // inner layer is 0 FIXME
  GPUCACell::TmpTuple stack;
  stack.reset();
  thisCell.find_ntuplets(cells, *foundNtuplets, *apc, stack, minHitsPerNtuplet);
  assert(stack.size()==0);
  // printf("in %d found quadruplets: %d\n", cellIndex, apc->get());
}

__global__
void kernel_print_found_ntuplets(GPU::SimpleVector<Quadruplet> *foundNtuplets, int maxPrint) {
  for (int i = 0; i < std::min(maxPrint, foundNtuplets->size()); ++i) {
    printf("\nquadruplet %d: %d %d %d %d\n", i,
           (*foundNtuplets)[i].hitId[0],
           (*foundNtuplets)[i].hitId[1],
           (*foundNtuplets)[i].hitId[2],
           (*foundNtuplets)[i].hitId[3]
          );

  }
}

void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{

  fitter.deallocateOnGPU();

  cudaFree(device_theCells_);
  cudaFree(device_isOuterHitOfCell_);
  cudaFree(device_nCells_);

  //product
  cudaFree(gpu_.tuples_d);
  cudaFree(gpu_.helix_fit_results_d);
  cudaFree(gpu_.apc_d);
  cudaFree(gpu_d);
  cudaFree(tuples_);
  cudaFree(helix_fit_results_);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{
  //////////////////////////////////////////////////////////
  // ALLOCATIONS FOR THE INTERMEDIATE RESULTS (STAYS ON WORKER)
  //////////////////////////////////////////////////////////

  cudaCheck(cudaMalloc(&device_theCells_,
             maxNumberOfLayerPairs_ * maxNumberOfDoublets_ * sizeof(GPUCACell)));
  cudaCheck(cudaMalloc(&device_nCells_, sizeof(uint32_t)));
  cudaCheck(cudaMemset(device_nCells_, 0, sizeof(uint32_t)));

  cudaCheck(cudaMalloc(&device_isOuterHitOfCell_,
             PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));
  cudaCheck(cudaMemset(device_isOuterHitOfCell_, 0,
             PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>)));


  //product
  cudaCheck(cudaMalloc(&gpu_.tuples_d, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMalloc(&gpu_.apc_d, sizeof(AtomicPairCounter)));
  cudaCheck(cudaMalloc(&gpu_.helix_fit_results_d, sizeof(Rfit::helix_fit)*maxNumberOfQuadruplets_));

  cudaCheck(cudaMalloc(&gpu_d, sizeof(TuplesOnGPU)));
  gpu_.me_d = gpu_d;
  cudaCheck(cudaMemcpy(gpu_d, &gpu_, sizeof(TuplesOnGPU), cudaMemcpyDefault));

  cudaCheck(cudaMallocHost(&tuples_, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMallocHost(&helix_fit_results_, sizeof(Rfit::helix_fit)*maxNumberOfQuadruplets_));

  fitter.allocateOnGPU(gpu_.tuples_d, gpu_.helix_fit_results_d);


}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex, HitsOnCPU const & hh,
                                                bool doRiemannFit,
                                                bool transferToCPU,
                                                cudaStream_t cudaStream)
{
  assert(regionIndex < maxNumberOfRegions_);
  assert(0==regionIndex);


  auto nhits = hh.nHits;
  assert(nhits <= PixelGPUConstants::maxNumberOfHits);
  auto blockSize = 64;
  auto stride = 4;
  auto numberOfBlocks = (nhits + blockSize - 1)/blockSize;
  numberOfBlocks *=stride;
  fishbone<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      hh.gpu_d,
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      nhits, stride
  );

  numberOfBlocks = (maxNumberOfDoublets_ + blockSize - 1)/blockSize;
  kernel_connect<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      gpu_.apc_d, // needed only to be reset, ready for next kernel
      hh.gpu_d,
      device_theCells_, device_nCells_,
      device_isOuterHitOfCell_,
      region.ptMin(),
      region.originRBound(), caThetaCut, caPhiCut, caHardPtCut,
      maxNumberOfDoublets_, PixelGPUConstants::maxNumberOfHits
  );
  cudaCheck(cudaGetLastError());

  kernel_find_ntuplets<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
      device_theCells_, device_nCells_,
      gpu_.tuples_d,
      gpu_.apc_d,
      4, maxNumberOfDoublets_);
  cudaCheck(cudaGetLastError());

  cudautils::finalizeBulk<<<numberOfBlocks, blockSize, 0, cudaStream>>>(gpu_.apc_d,gpu_.tuples_d);

  numberOfBlocks = (std::max(int(nhits), maxNumberOfDoublets_) + blockSize - 1)/blockSize;
  kernel_checkOverflows<<<numberOfBlocks, blockSize, 0, cudaStream>>>(
                        gpu_.tuples_d,
                        device_theCells_, device_nCells_,
                        device_isOuterHitOfCell_, nhits,
                        maxNumberOfDoublets_
                       );
  cudaCheck(cudaGetLastError());

  // kernel_print_found_ntuplets<<<1, 1, 0, cudaStream>>>(gpu_.tuples_d, 10);

  if (doRiemannFit) {
    launchFit(hh, nhits, cudaStream);
  }

  if (transferToCPU) {
    cudaCheck(cudaMemcpyAsync(tuples_,gpu_.tuples_d,
                              sizeof(TuplesOnGPU::Container),
                              cudaMemcpyDeviceToHost, cudaStream));

    cudaCheck(cudaMemcpyAsync(helix_fit_results_,gpu_.helix_fit_results_d, 
                              sizeof(Rfit::helix_fit)*maxNumberOfQuadruplets_,
                              cudaMemcpyDeviceToHost, cudaStream));
  }

}

void CAHitQuadrupletGeneratorGPU::cleanup(cudaStream_t cudaStream) {
  // this lazily resets temporary memory for the next event, and is not needed for reading the output
  cudaCheck(cudaMemsetAsync(device_isOuterHitOfCell_, 0,
                            PixelGPUConstants::maxNumberOfHits * sizeof(GPU::VecArray<unsigned int, maxCellsPerHit_>),
                            cudaStream));
  cudaCheck(cudaMemsetAsync(device_nCells_, 0, sizeof(uint32_t), cudaStream));
}

std::vector<std::array<int, 4>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int)
{
  assert(tuples_);
  auto const & tuples = *tuples_;

  uint32_t sizes[7]={0};
  std::vector<int> ntk(10000);
  auto add = [&](uint32_t hi) { if (hi>=ntk.size()) ntk.resize(hi+1); ++ntk[hi];};

  std::vector<std::array<int, 4>> quadsInterface; quadsInterface.reserve(10000);

  nTuples_=0;
  for (auto i = 0U; i < tuples.totbins(); ++i) {
    auto sz = tuples.size(i);
    if (sz==0) break;  // we know cannot be less then 3
    ++nTuples_;
    ++sizes[sz];
    for (auto j=tuples.begin(i); j!=tuples.end(i); ++j) add(*j);
    if (sz<4) continue;
    quadsInterface.emplace_back(std::array<int, 4>());
    quadsInterface.back()[0] = tuples.begin(i)[0];
    quadsInterface.back()[1] = tuples.begin(i)[1];
    quadsInterface.back()[2] = tuples.begin(i)[2];   // [sz-2];
    quadsInterface.back()[3] = tuples.begin(i)[3];   // [sz-1];
  }

//#ifdef GPU_DEBUG
  long long ave =0; int nn=0; for (auto k : ntk) if(k>0){ave+=k; ++nn;}
  std::cout << "Q Produced " << quadsInterface.size() << " quadruplets: ";
  for (auto i=3; i<7; ++i) std::cout << sizes[i] << ' ';
  std::cout << "max/ave " << *std::max_element(ntk.begin(),ntk.end())<<'/'<<float(ave)/float(nn) << std::endl;
//#endif
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(HitsOnCPU const & hh, cudaStream_t stream) {
  auto nhits = hh.nHits;

  int threadsPerBlock = gpuPixelDoublets::getDoubletsFromHistoMaxBlockSize;
  int blocks = (3 * nhits + threadsPerBlock - 1) / threadsPerBlock;
  gpuPixelDoublets::getDoubletsFromHisto<<<blocks, threadsPerBlock, 0, stream>>>(device_theCells_, device_nCells_, hh.gpu_d, device_isOuterHitOfCell_);
  cudaCheck(cudaGetLastError());
}
