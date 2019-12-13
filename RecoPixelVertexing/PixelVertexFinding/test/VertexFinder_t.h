#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/requireCUDADevices.h"
#include "HeterogeneousCore/CUDAUtilities/interface/launch.h"
#ifdef USE_DBSCAN
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksDBSCAN.h"
#define CLUSTERIZE clusterTracksDBSCAN
#elif USE_ITERATIVE
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksIterative.h"
#define CLUSTERIZE clusterTracksIterative
#else
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuClusterTracksByDensity.h"
#define CLUSTERIZE clusterTracksByDensityKernel
#endif
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuFitVertices.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuSortByPt2.h"
#include "RecoPixelVertexing/PixelVertexFinding/src/gpuSplitVertices.h"

#ifdef ONE_KERNEL
__global__ void vertexFinderOneKernel(gpuVertexFinder::ZVertices* pdata,
                                      gpuVertexFinder::WorkSpace* pws,
                                      int minT,      // min number of neighbours to be "seed"
                                      float eps,     // max absolute distance to cluster
                                      float errmax,  // max error to be "seed"
                                      float chi2max  // max normalized distance to cluster,
) {
  clusterTracksByDensity(pdata, pws, minT, eps, errmax, chi2max);
  __syncthreads();
  fitVertices(pdata, pws, 50.);
  __syncthreads();
  splitVertices(pdata, pws, 9.f);
  __syncthreads();
  fitVertices(pdata, pws, 5000.);
  __syncthreads();
  sortByPt2(pdata, pws);
}
#endif

using namespace gpuVertexFinder;

struct Event {
  std::vector<float> zvert;
  std::vector<uint16_t> itrack;
  std::vector<float> ztrack;
  std::vector<float> eztrack;
  std::vector<float> pttrack;
  std::vector<uint16_t> ivert;
};

struct ClusterGenerator {
  explicit ClusterGenerator(float nvert, float ntrack)
      : rgen(-13., 13), errgen(0.005, 0.025), clusGen(nvert), trackGen(ntrack), gauss(0., 1.), ptGen(0.001,1.) {}

  void operator()(Event& ev) {
    int nclus = clusGen(reng);
    ev.zvert.resize(nclus);
    ev.itrack.resize(nclus);
    for (auto& z : ev.zvert) {
      z = 3.5f * gauss(reng);
    }

    ev.ztrack.clear();
    ev.eztrack.clear();
    ev.pttrack.clear();
    ev.ivert.clear();    
    float ptMax=0; float pt5=0;
    for (int iv = 0; iv < nclus; ++iv) {
      auto nt = trackGen(reng);
      if (iv == 5) nt *= 3;
      ev.itrack[iv] = nt;
      float ptSum=0;
      for (int it = 0; it < nt; ++it) {
        auto err = errgen(reng);  // reality is not flat....
        ev.ztrack.push_back(ev.zvert[iv] + err * gauss(reng));
        ev.eztrack.push_back(err * err);
        ev.ivert.push_back(iv);
        ev.pttrack.push_back(std::pow(ptGen(reng),iv==5 ?-1.0f:-0.5f));
        ev.pttrack.back() *= ev.pttrack.back();
        ptSum += ev.pttrack.back();
      }
      ptMax = std::max(ptMax,ptSum);
      if (iv == 5) pt5 = ptSum;
    }
    std::cout << "PV, ptMax " << std::sqrt(pt5) << ' ' << std::sqrt(ptMax) << std::endl;

    // add noise
    auto nt = 2 * trackGen(reng);
    for (int it = 0; it < nt; ++it) {
      auto err = 0.03f;
      ev.ztrack.push_back(rgen(reng));
      ev.eztrack.push_back(err * err);
      ev.ivert.push_back(9999);
      ev.pttrack.push_back(std::pow(ptGen(reng),-0.5f));
      ev.pttrack.back() *= ev.pttrack.back();
    }
  }

  std::mt19937 reng;
  std::uniform_real_distribution<float> rgen;
  std::uniform_real_distribution<float> errgen;
  std::poisson_distribution<int> clusGen;
  std::poisson_distribution<int> trackGen;
  std::normal_distribution<float> gauss;
  std::uniform_real_distribution<float> ptGen;  // becomes a power low
};

// a macro SORRY
#define LOC_ONGPU(M) ((char*)(onGPU_d.get()) + offsetof(ZVertices, M))
#define LOC_WS(M) ((char*)(ws_d.get()) + offsetof(WorkSpace, M))

__global__ void print(ZVertices const* pdata, WorkSpace const* pws) {
  auto const& __restrict__ data = *pdata;
  auto const& __restrict__ ws = *pws;
  printf("nt,nv %d %d,%d\n", ws.ntrks, data.nvFinal, ws.nvIntermediate);
}

__global__ void linit(ZVertices * pdata, WorkSpace * pws, int nt) {
  auto & __restrict__ data = *pdata;
  auto & __restrict__ ws = *pws;
  for (int i=0; i<nt; ++i) {  ws.itrk[i]=i; data.idv[i] = -1;}
}

int main() {
#ifndef CUDA_KERNELS_ON_CPU
  requireCUDADevices();

  auto onGPU_d = cudautils::make_device_unique<ZVertices[]>(1, nullptr);
  auto ws_d = cudautils::make_device_unique<WorkSpace[]>(1, nullptr);
#else
  auto onGPU_d = std::make_unique<ZVertices>();
  auto ws_d = std::make_unique<WorkSpace>();
#endif

  Event ev;

  float eps = 0.1f;
  std::array<float, 3> par{{eps, 0.01f, 9.0f}};
  for (int nav = 30; nav < 100; nav += 20) {
    ClusterGenerator gen(nav, 10);

    for (int iii = 8; iii < 20; ++iii) {
      auto kk = iii / 4;  // M param

      gen(ev);

      std::cout << "v,t size " << ev.zvert.size() << ' ' << ev.ztrack.size() << std::endl;
      int nt = ev.ztrack.size();
      int nvori = ev.zvert.size();
      int ntori = nt;
      assert(ntori<int(ZVertexSoA::MAXTRACKS));
      assert(nvori< int(ZVertexSoA::MAXVTX));

#ifndef CUDA_KERNELS_ON_CPU
      init<<<1, 1, 0, 0>>>(onGPU_d.get(), ws_d.get());
      linit<<<1, 1, 0, 0>>>(onGPU_d.get(), ws_d.get(),ntori);
#else
      onGPU_d->init();
      ws_d->init();
      for (int16_t i=0; i<ntori; ++i) {  ws_d->itrk[i]=i; onGPU_d->idv[i] = -1;}  // FIXME do the same on GPU....
#endif

#ifndef CUDA_KERNELS_ON_CPU
      cudaCheck(cudaMemcpy(LOC_WS(ntrks), &nt, sizeof(uint32_t), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(LOC_WS(zt), ev.ztrack.data(), sizeof(float) * ev.ztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(LOC_WS(ezt2), ev.eztrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));
      cudaCheck(cudaMemcpy(LOC_WS(ptt2), ev.pttrack.data(), sizeof(float) * ev.eztrack.size(), cudaMemcpyHostToDevice));

#else
      ::memcpy(LOC_WS(ntrks), &nt, sizeof(uint32_t));
      ::memcpy(LOC_WS(zt), ev.ztrack.data(), sizeof(float) * ev.ztrack.size());
      ::memcpy(LOC_WS(ezt2), ev.eztrack.data(), sizeof(float) * ev.eztrack.size());
      ::memcpy(LOC_WS(ptt2), ev.pttrack.data(), sizeof(float) * ev.eztrack.size());
#endif

      std::cout << "M eps, pset " << kk << ' ' << eps << ' ' << (iii % 4) << std::endl;

      if ((iii % 4) == 0)
        par = {{eps, 0.02f, 12.0f}};
      if ((iii % 4) == 1)
        par = {{eps, 0.02f, 9.0f}};
      if ((iii % 4) == 2)
        par = {{eps, 0.01f, 9.0f}};
      if ((iii % 4) == 3)
        par = {{0.7f * eps, 0.01f, 9.0f}};

      uint32_t nv = 0;
#ifndef CUDA_KERNELS_ON_CPU
      print<<<1, 1, 0, 0>>>(onGPU_d.get(), ws_d.get());
      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

#ifdef ONE_KERNEL
      cudautils::launch(vertexFinderOneKernel, {1, 512 + 256}, onGPU_d.get(), ws_d.get(), kk, par[0], par[1], par[2]);
#else
      cudautils::launch(CLUSTERIZE, {1, 512 + 256}, onGPU_d.get(), ws_d.get(), kk, par[0], par[1], par[2]);
#endif
      print<<<1, 1, 0, 0>>>(onGPU_d.get(), ws_d.get());

      cudaCheck(cudaGetLastError());
      cudaDeviceSynchronize();

      cudautils::launch(fitVerticesKernel, {1, 1024 - 256}, onGPU_d.get(), ws_d.get(), 50.f);
      cudaCheck(cudaGetLastError());
      cudaCheck(cudaMemcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t), cudaMemcpyDeviceToHost));

#else
      print(onGPU_d.get(), ws_d.get());
      CLUSTERIZE(onGPU_d.get(), ws_d.get(), kk, par[0], par[1], par[2]);
      print(onGPU_d.get(), ws_d.get());
      fitVertices(onGPU_d.get(), ws_d.get(), 50.f);
      nv = onGPU_d->nvFinal;
#endif

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

      int16_t * idv = nullptr;
      float* zv = nullptr;
      float* wv = nullptr;
      float* ptv2 = nullptr;
      int32_t* nn = nullptr;
      uint16_t* ind = nullptr;

      // keep chi2 separated...
      float chi2[2 * nv];  // make space for splitting...

#ifndef CUDA_KERNELS_ON_CPU
      int16_t hidv[16000];
      float hzv[2 * nv];
      float hwv[2 * nv];
      float hptv2[2 * nv];
      int32_t hnn[2 * nv];
      uint16_t hind[2 * nv];

      idv = hidv;
      zv = hzv;
      wv = hwv;
      ptv2 = hptv2;
      nn = hnn;
      ind = hind;
#else
      idv = onGPU_d->idv;
      zv = onGPU_d->zv;
      wv = onGPU_d->wv;
      ptv2 = onGPU_d->ptv2;
      nn = onGPU_d->ndof;
      ind = onGPU_d->sortInd;
#endif

#ifndef CUDA_KERNELS_ON_CPU
      cudaCheck(cudaMemcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float), cudaMemcpyDeviceToHost));
#else
      memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));
#endif

   auto verifyMatch = [&]() {

      // matching-merging metrics
      constexpr int MAXMA = 32;
      struct Match { Match() {for (auto&e:vid)e=-1; for (auto&e:nt)e=0;} std::array<int,MAXMA> vid; std::array<int,MAXMA> nt; };

      auto nnn=0;
      Match matches[nv]; for (auto kv = 0U; kv < nv; ++kv) { matches[kv] =  Match();}
      auto iPV =  ind[nv - 1];
      for (int it=0; it<nt; ++it) {
        auto const iv = idv[it];
        if (iv>9990) continue;
        assert(iv<int(nv));
        if (iv<0) continue;
        auto const tiv = ev.ivert[it];
        if (tiv>9990) continue;
        ++nnn;
        for (int i=0; i<MAXMA; ++i) {
          if (matches[iv].vid[i]<0) { matches[iv].vid[i]=tiv; matches[iv].nt[i]=1; break;}
          else if (tiv==matches[iv].vid[i]) { ++(matches[iv].nt[i]); break;}
        }
      }

      float frac[nv];
      int nok=0; int merged50=0; int merged75=0; int nmess=0;
      float dz=0;
      for (auto kv = 0U; kv < nv; ++kv) {
        auto mx = std::max_element(matches[kv].nt.begin(),matches[kv].nt.end())-matches[kv].nt.begin();
        assert(mx>=0 && mx<MAXMA);
        if (0==matches[kv].nt[mx]) std::cout <<"????? " << kv << ' ' << matches[kv].vid[mx] << ' ' << matches[kv].vid[0] << std::endl;
        auto itv = matches[kv].vid[mx];
        frac[kv] = itv<0 ? 0.f : float(matches[kv].nt[mx])/float(ev.itrack[itv]);
        assert(frac[kv]<1.1f);
        if (frac[kv]>0.75f) ++nok;
        if (frac[kv]<0.5f) ++nmess;
        auto ldz = std::abs(zv[kv] - ev.zvert[itv]);
        dz = std::max(dz,ldz);
        int nm5=0; int nm7=0;
        int ntt=0;
        for (int i=0; i<MAXMA; ++i) {
          ntt+=matches[kv].nt[i];
          auto itv = matches[kv].vid[i];
          float f = itv<0 ? 0.f : float(matches[kv].nt[i])/float(ev.itrack[itv]);
          if (f>0.5f) ++nm5;
          if (f>0.75f) ++nm7;
        }
        if (nm5>1) ++merged50;
        if (nm7>1) ++merged75;
        if (kv ==  iPV ) std::cout << "PV " << itv << ' ' << std::sqrt(ptv2[kv]) << ' ' << float(ntt)/float(ev.itrack[itv]) << '/' <<  frac[kv] << '/' << nm5 << '/' << nm7 << ' ' << dz << std::endl;
      }
      // for (auto f: frac) std::cout << f << ' ';
      // std::cout << std::endl;
      std::cout << "ori/tot/matched/merged5//merged7/random/dz "
                << nvori << '/' << nv << '/' << nok << '/' << merged50 << '/' << merged75  << '/' << nmess
                << '/' << dz << std::endl;
      }; // verifyMatch


      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "after fit nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

#ifndef CUDA_KERNELS_ON_CPU
      cudautils::launch(fitVerticesKernel, {1, 1024 - 256}, onGPU_d.get(), ws_d.get(), 50.f);
      cudaCheck(cudaMemcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float), cudaMemcpyDeviceToHost));
#else
      fitVertices(onGPU_d.get(), ws_d.get(), 50.f);
      nv = onGPU_d->nvFinal;
      memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));
#endif

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "before splitting nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

#ifndef CUDA_KERNELS_ON_CPU
      // one vertex per block!!!
      cudautils::launch(splitVerticesKernel, {1024, 64}, onGPU_d.get(), ws_d.get(), 9.f);
      cudaCheck(cudaMemcpy(&nv, LOC_WS(nvIntermediate), sizeof(uint32_t), cudaMemcpyDeviceToHost));
#else
      gridDim.x = 1;
      assert(blockIdx.x == 0);
      splitVertices(onGPU_d.get(), ws_d.get(), 9.f);
      resetGrid();
      nv = ws_d->nvIntermediate;
#endif
      std::cout << "after split " << nv << std::endl;

#ifndef CUDA_KERNELS_ON_CPU
      cudautils::launch(fitVerticesKernel, {1, 1024 - 256}, onGPU_d.get(), ws_d.get(), 5000.f);
      cudautils::launch(sortByPt2Kernel, {1, 256}, onGPU_d.get(), ws_d.get());
      cudaCheck(cudaMemcpy(&nv, LOC_ONGPU(nvFinal), sizeof(uint32_t), cudaMemcpyDeviceToHost));
#else
      fitVertices(onGPU_d.get(), ws_d.get(), 5000.f);
      sortByPt2(onGPU_d.get(), ws_d.get());
      nv = onGPU_d->nvFinal;
      memcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float));
#endif

      if (nv == 0) {
        std::cout << "NO VERTICES???" << std::endl;
        continue;
      }

#ifndef CUDA_KERNELS_ON_CPU
      cudaCheck(cudaMemcpy(zv, LOC_ONGPU(zv), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(wv, LOC_ONGPU(wv), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(chi2, LOC_ONGPU(chi2), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ptv2, LOC_ONGPU(ptv2), nv * sizeof(float), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(nn, LOC_ONGPU(ndof), nv * sizeof(int32_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(ind, LOC_ONGPU(sortInd), nv * sizeof(uint16_t), cudaMemcpyDeviceToHost));
      cudaCheck(cudaMemcpy(idv, LOC_ONGPU(idv), nt * sizeof(uint16_t), cudaMemcpyDeviceToHost));
#endif

      for (auto j = 0U; j < nv; ++j)
        if (nn[j] > 0)
          chi2[j] /= float(nn[j]);
      {
        auto mx = std::minmax_element(chi2, chi2 + nv);
        std::cout << "nv, min max chi2 " << nv << " " << *mx.first << ' ' << *mx.second << std::endl;
      }

      {
        auto mx = std::minmax_element(wv, wv + nv);
        std::cout << "min max error " << 1. / std::sqrt(*mx.first) << ' ' << 1. / std::sqrt(*mx.second) << std::endl;
      }

      {
        auto mx = std::minmax_element(ptv2, ptv2 + nv);
        std::cout << "min max ptv2 " << *mx.first << ' ' << *mx.second << std::endl;
        std::cout << "min max ptv2 " << ptv2[ind[0]] << ' ' << ptv2[ind[nv - 1]] << " at " << ind[0] << ' '
                  << ind[nv - 1] << std::endl;
      }

      float dd[nv];
      for (auto kv = 0U; kv < nv; ++kv) {
        auto zr = zv[kv];
        auto md = 500.0f;
        for (auto zs : ev.ztrack) {
          auto d = std::abs(zr - zs);
          md = std::min(d, md);
        }
        dd[kv] = md;
      }
      if (iii == 6) {
        for (auto d : dd)
          std::cout << d << ' ';
        std::cout << std::endl;
      }
      auto mx = std::minmax_element(dd, dd + nv);
      float rms = 0;
      for (auto d : dd)
        rms += d * d;
      rms = std::sqrt(rms) / (nv - 1);
      std::cout << "min max rms " << *mx.first << ' ' << *mx.second << ' ' << rms << std::endl;


      verifyMatch();


    }  // loop on events
  }    // lopp on ave vert

  return 0;
}
