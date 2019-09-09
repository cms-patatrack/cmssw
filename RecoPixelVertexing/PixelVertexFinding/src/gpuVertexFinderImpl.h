#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksByDensity.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksDBSCAN.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksIterative.h"

#include "gpuFitVertices.h"
#include "gpuSortByPt2.h"
#include "gpuSplitVertices.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


namespace gpuVertexFinder {

  __global__ void loadTracks(TkSoA const* ptracks, ZVertexSoA * soa, WorkSpace* pws, float ptMin) {
    
    auto const & tracks = *ptracks;
    auto const & fit = tracks.stateAtBS;
    auto const* quality = tracks.qualityData();

    auto first = blockIdx.x * blockDim.x + threadIdx.x;
    for  (int idx = first, nt = TkSoA::stride(); idx<nt; idx+=gridDim.x * blockDim.x) {
      auto nHits = tracks.nHits(idx);
      if (nHits == 0) break;  // this is a guard: maybe we need to move to nTracks...

      // initialize soa...
      soa->idv[idx]=-1;

      if (nHits < 4)
        return;  // no triplets
      if (quality[idx] != trackQuality::loose)
        return;
 
      auto pt = tracks.pt(idx);

      if (pt < ptMin)
        continue;

      auto& data = *pws;
      auto it = atomicAdd(&data.ntrks, 1);
      data.itrk[it] = idx;
      data.zt[it] = tracks.zip(idx);
      data.ezt2[it] = fit.covariance(idx)(14);
      data.ptt2[it] = pt*pt;
    }  

  }

#ifdef __CUDACC__
  ZVertexGPU Producer::makeAsync(cuda::stream_t<>& stream, TkSoA const * tksoa, float ptMin) const {
    ZVertexGPU vertices(stream);
#else
  ZVertexCPU Producer::make(cuda::stream_t<>& stream, TkSoA const * tksoa, float ptMin) const {
    ZVertexCPU vertices(stream);
#endif
    assert(tksoa);
    auto * soa = vertices.get();

#ifdef __CUDACC__
    edm::Service<CUDAService> cs;
    auto ws_d = cs->make_device_unique<WorkSpace>(stream);
#else
    auto ws_d = std::make_unique<WorkSpace>();
#endif   

#ifdef __CUDACC__
    init<<<1, 1, 0, stream.id()>>>(soa, ws_d.get());
    auto blockSize = 128;
    auto numberOfBlocks = (TkSoA::stride() + blockSize - 1) / blockSize;
    loadTracks<<<numberOfBlocks, blockSize, 0, stream.id()>>>(tksoa, soa, ws_d.get(), ptMin);
    cudaCheck(cudaGetLastError());
#else
    init(soa, ws_d.get());
    loadTracks(tksoa, soa, ws_d.get(), ptMin);
#endif


#ifdef __CUDACC__
    if (useDensity_) {
      clusterTracksByDensity<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), minT, eps, errmax, chi2max);
    }
    cudaCheck(cudaGetLastError());
    fitVertices<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), 50.);
    cudaCheck(cudaGetLastError());
#else
    if (useDensity_) {
      clusterTracksByDensity(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useDBSCAN_) {
      clusterTracksDBSCAN(soa, ws_d.get(), minT, eps, errmax, chi2max);
    } else if (useIterative_) {
      clusterTracksIterative(soa, ws_d.get(), minT, eps, errmax, chi2max);
    }
    fitVertices(soa, ws_d.get(), 50.);
#endif

#ifdef __CUDACC__
    splitVertices<<<1024, 128, 0, stream.id()>>>(soa, ws_d.get(), 9.f);
    cudaCheck(cudaGetLastError());
    fitVertices<<<1, 1024 - 256, 0, stream.id()>>>(soa, ws_d.get(), 5000.);
    cudaCheck(cudaGetLastError());

    sortByPt2<<<1, 256, 0, stream.id()>>>(soa, ws_d.get());
    cudaCheck(cudaGetLastError());
#else
    splitVertices(soa, ws_d.get(), 9.f);
    fitVertices(soa, ws_d.get(), 5000.);

    sortByPt2(soa, ws_d.get());
#endif


    return vertices;
  }

}  // namespace gpuVertexFinder

#undef FROM
