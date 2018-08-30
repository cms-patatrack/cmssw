#ifndef RecoPixelVertexing_PixelVertexFinding_clusterTracks_H
#define RecoPixelVertexing_PixelVertexFinding_clusterTracks_H

#include<cstdint>
#include<cmath>
#include<cassert>

#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"

#include "gpuVertexFinder.h"

namespace gpuVertexFinder {
  
  // this algo does not really scale as it works in a single block...
  // enough for <10K tracks we have
  __global__ 
  void clusterTracks(int nt,
		     OnGPU * pdata,
		     int minT,  // min number of neighbours to be "core"
		     float eps, // max absolute distance to cluster
		     float errmax, // max error to be "seed"
		     float chi2max   // max normalized distance to cluster
		     )  {

    constexpr bool verbose = false;
    
    auto er2mx = errmax*errmax;
    
    auto & data = *pdata;
    float const * zt = data.zt;
    float const * ezt2 = data.ezt2;
    float * zv = data.zv;
    float * wv = data.wv;
    uint32_t & nv = *data.nv;
    
    int8_t  * izt = data.izt;
    uint16_t * nn = data.nn;
    int32_t * iv = data.iv;
    
    assert(pdata);
    assert(zt);
    
    __shared__ HistoContainer<int8_t,8,5,8,uint16_t> hist;
    
    //  if(0==threadIdx.x) printf("params %d %f\n",minT,eps);    
    //  if(0==threadIdx.x) printf("booked hist with %d bins, size %d for %d tracks\n",hist.nbins(),hist.binSize(),nt);
    
    // zero hist
    hist.nspills = 0;
    for (auto k = threadIdx.x; k<hist.nbins(); k+=blockDim.x) hist.n[k]=0;
    __syncthreads();

    //  if(0==threadIdx.x) printf("histo zeroed\n");
    
    
  // fill hist
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      assert(i<64000);
      int iz =  int(zt[i]*10.);
      iz = std::max(iz,-127);
      iz = std::min(iz,127);
      izt[i]=iz;
      hist.fill(int8_t(iz),uint16_t(i));
      iv[i]=i;
      nn[i]=0;
    }
    __syncthreads();
    
    //   if(0==threadIdx.x) printf("histo filled %d\n",hist.nspills);
    if(0==threadIdx.x && hist.fullSpill()) printf("histo overflow\n");
    
    // count neighbours
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (ezt2[i]>er2mx) continue;
      auto loop = [&](int j) {
        if (i==j) return;
        auto dist = std::abs(zt[i]-zt[j]);
        if (dist>eps) return;
        if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
        nn[i]++;
      };
      
      int bs = hist.bin(izt[i]);
      int be = std::min(int(hist.nbins()),bs+2);
      bs = bs==0 ? 0 : bs-1;
      assert(be>bs);
      for (auto b=bs; b<be; ++b){
	for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
	  loop(*pj);
	}}
      for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
        loop(*pj);
    }
    
    __syncthreads();
    
    //  if(0==threadIdx.x) printf("nn counted\n");
    
    // cluster seeds only
    bool more = true;
    while (__syncthreads_or(more)) {
      more=false;
      for (int i = threadIdx.x; i < nt; i += blockDim.x) {
	if (nn[i]<minT) continue; // DBSCAN core rule
	auto loop = [&](int j) {
	  if (i==j) return;
	  if (nn[j]<minT) return;  // DBSCAN core rule
	  // look on the left
	  auto dist = zt[j]-zt[i];
	  if (dist<0 || dist>eps) return;
	  if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return;
	  auto old = atomicMin(&iv[j], iv[i]);
	  if (old != iv[i]) {
	    // end the loop only if no changes were applied
	    more = true;
	  }
	  atomicMin(&iv[i], old);
	};
	
	int bs = hist.bin(izt[i]);
	int be = std::min(int(hist.nbins()),bs+2);
	for (auto b=bs; b<be; ++b){
	  for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
       	    loop(*pj);
	  }}
	for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
	  loop(*pj);
      } // for i
    } // while
    
    
    
    // collect edges (assign to closest cluster of closest point??? here to closest point)
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      //    if (nn[i]==0 || nn[i]>=minT) continue;    // DBSCAN edge rule
      if (nn[i]>=minT) continue;    // DBSCAN edge rule
      float mdist=eps;
      auto loop = [&](int j) {
	if (nn[j]<minT) return;  // DBSCAN core rule
	auto dist = std::abs(zt[i]-zt[j]);
	if (dist>mdist) return;
	if (dist*dist>chi2max*(ezt2[i]+ezt2[j])) return; // needed?
	mdist=dist;
	iv[i] = iv[j]; // assign to cluster (better be unique??)
      };
      int bs = hist.bin(izt[i]);
      int be = std::min(int(hist.nbins()),bs+2);
      bs = bs==0 ? 0 : bs-1;
      for (auto b=bs; b<be; ++b){
	for (auto pj=hist.begin(b);pj<hist.end(b);++pj) {
	  loop(*pj);
	}}
      for (auto pj=hist.beginSpill();pj<hist.endSpill();++pj)
        loop(*pj);
    }
    
    
    __shared__ int foundClusters;
    foundClusters = 0;
    __syncthreads();
    
    // find the number of different clusters, identified by a tracks with clus[i] == i;
    // mark these tracks with a negative id.
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] == i) {
	if  (nn[i]>=minT) {
	  auto old = atomicAdd(&foundClusters, 1);
	  iv[i] = -(old + 1);
	  zv[old]=0;
	  wv[old]=0;
	} else { // noise
	  iv[i] = -9998;
	}
      }
    }
    __syncthreads();
    
    // propagate the negative id to all the tracks in the cluster.
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i] >= 0) {
	// mark each track in a cluster with the same id as the first one
	iv[i] = iv[iv[i]];
      }
    }
    __syncthreads();
    
    // adjust the cluster id to be a positive value starting from 0
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      iv[i] = - iv[i] - 1;
    }
    
    // only for test
    __shared__ int noise;
    noise = 0;
    
    __syncthreads();
    
    // compute cluster location
    for (int i = threadIdx.x; i < nt; i += blockDim.x) {
      if (iv[i]>9990) {
	if (verbose) atomicAdd(&noise, 1);
	continue;
      }
      assert(iv[i]>=0);
      assert(iv[i]<foundClusters);
      auto w = 1.f/ezt2[i];
      atomicAdd(&zv[iv[i]],zt[i]*w);
      atomicAdd(&wv[iv[i]],w); 
    }
    
    __syncthreads();
    
    if(verbose && 0==threadIdx.x) printf("found %d proto clusters ",foundClusters);
    if(verbose && 0==threadIdx.x) printf("and %d noise\n",noise);
    
    for (int i = threadIdx.x; i < foundClusters; i += blockDim.x) zv[i]/=wv[i];
    
    nv = foundClusters;
  }

}
#endif
