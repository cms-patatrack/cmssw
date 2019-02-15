#include "gpuClusterTracks.h"
#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"


// a macro SORRY
#define FROM(M) ((char*)(onGPU_d) +offsetof(OnGPU,M))


namespace gpuVertexFinder {


  void Producer::allocateOnGPU() {
    cudaCheck(cudaMalloc(&onGPU_d,sizeof(OnGPU)));
  }
	      
  void Producer::deallocateOnGPU() {
    cudaCheck(cudaFree(onGPU_d));
  }


  __global__
  void loadTracks(pixelTuplesHeterogeneousProduct::TuplesOnGPU const * tracks,
                  OnGPU * pdata,
                  float ptMin
                 ){

    auto const & tuples = *tracks->tuples_d;
    auto const * fit = tracks->helix_fit_results_d;
    auto const * quality = tracks->quality_d;

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>= tuples.nbins()) return;
    if (tuples.size(idx)==0) {
      return;
    }

    if(quality[idx] != pixelTuplesHeterogeneousProduct::loose ) return;

    auto const & fittedTrack = fit[idx];

    if (fittedTrack.par(2)<ptMin) return; 

    auto & data = *pdata;   
    auto it = atomicAdd(&data.ntrks,1);
    data.itrk[it] = idx;
    data.zt[it] = fittedTrack.par(4);
    data.ezt2[it] = fittedTrack.cov(4, 4);
    data.ptt2[it] = fittedTrack.par(2)*fittedTrack.par(2);
 
  }

  void Producer::produce(cudaStream_t stream, TuplesOnCPU const & tracks, float ptMin) {
    
    assert(onGPU_d);assert(tracks.gpu_d);
    init<<<1,1,0,stream>>>(onGPU_d);
    auto blockSize = 128;
    auto numberOfBlocks = (CAConstants::maxTuples() + blockSize - 1) / blockSize;
    loadTracks<<<numberOfBlocks,blockSize,0,stream>>>(tracks.gpu_d,onGPU_d, ptMin);
    cudaCheck(cudaGetLastError());

    clusterTracks<<<1,1024-256,0,stream>>>(onGPU_d,minT,eps,errmax,chi2max);
    cudaCheck(cudaGetLastError());
    fitVertices<<<1,1024-256,0,stream>>>(onGPU_d,50.);
    cudaCheck(cudaGetLastError());

    splitVertices<<<1024,128,0,stream>>>(onGPU_d,9.f);
    cudaCheck(cudaGetLastError());
    fitVertices<<<1,1024-256,0,stream>>>(onGPU_d,5000.);
    cudaCheck(cudaGetLastError());

    sortByPt2<<<1,256,0,stream>>>(onGPU_d);
    cudaCheck(cudaGetLastError());

    if(enableTransfer) {
      auto from = (char*)(onGPU_d) +offsetof(OnGPU,nvFinal);
      cudaCheck(cudaMemcpyAsync(&gpuProduct.nVertices, from, sizeof(uint32_t),
                                cudaMemcpyDeviceToHost, stream));
      from = (char*)(onGPU_d) +offsetof(OnGPU,ntrks);
      cudaCheck(cudaMemcpyAsync(&gpuProduct.nTracks, from, sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, stream));
    }
  }
  
  Producer::OnCPU const & Producer::fillResults(cudaStream_t stream) {

    if(!enableTransfer) return gpuProduct;

    // finish copy
    gpuProduct.ivtx.resize(gpuProduct.nTracks);
    cudaCheck(cudaMemcpyAsync(gpuProduct.ivtx.data(),FROM(iv),sizeof(int32_t)*gpuProduct.nTracks,
                              cudaMemcpyDeviceToHost, stream));
    gpuProduct.itrk.resize(gpuProduct.nTracks);
    cudaCheck(cudaMemcpyAsync(gpuProduct.itrk.data(),FROM(itrk),sizeof(int16_t)*gpuProduct.nTracks,
                              cudaMemcpyDeviceToHost, stream));

    gpuProduct.z.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.z.data(),FROM(zv),sizeof(float)*gpuProduct.nVertices,
			      cudaMemcpyDeviceToHost, stream));
    gpuProduct.zerr.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.zerr.data(),FROM(wv),sizeof(float)*gpuProduct.nVertices,
			      cudaMemcpyDeviceToHost, stream));
    gpuProduct.chi2.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.chi2.data(),FROM(chi2),sizeof(float)*gpuProduct.nVertices,
			      cudaMemcpyDeviceToHost, stream));
        
    gpuProduct.sortInd.resize(gpuProduct.nVertices);
    cudaCheck(cudaMemcpyAsync(gpuProduct.sortInd.data(),FROM(sortInd),sizeof(uint16_t)*gpuProduct.nVertices,
                              cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    
    return gpuProduct;
  }
	
} // end namespace

#undef FROM
