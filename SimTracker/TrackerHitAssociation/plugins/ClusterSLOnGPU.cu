#include "ClusterSLOnGPU.h"

// for the "packing"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include <limits>
#include<cassert>
#include<atomic>
#include <mutex>


using ClusterSLGPU = trackerHitAssociationHeterogeneousProduct::ClusterSLGPU;

__global__
void simLink(clusterSLOnGPU::DigisOnGPU const * ddp, uint32_t ndigis, clusterSLOnGPU::HitsOnGPU const * hhp, ClusterSLGPU const * slp, uint32_t n) {

  assert(slp==slp->me_d);

  constexpr int32_t invTK = 0; // std::numeric_limits<int32_t>::max();
 
  constexpr uint16_t InvId=9999; // must be > MaxNumModules
  
  auto const & dd = *ddp;
  auto const & hh = *hhp;
  auto const & sl = *slp;
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (i>ndigis) return;

  auto id = dd.moduleInd_d[i];
  if (InvId==id) return;
  assert(id<2000);

  auto ch = pixelgpudetails::pixelToChannel(dd.xx_d[i], dd.yy_d[i]);
  auto first = hh.hitsModuleStart_d[id];
  auto cl = first + dd.clus_d[i];
  assert(cl<256*2000);
  
  const std::array<uint32_t,4> me{{id,ch,0,0}};

  auto less = [] __device__ __host__ (std::array<uint32_t,4> const & a, std::array<uint32_t,4> const & b)->bool {
     return a[0]<b[0] || ( !(b[0]<a[0]) && a[1]<b[1]); // in this context we do not care of [2] 
  };

  auto equal = [] __device__ __host__ (std::array<uint32_t,4> const & a, std::array<uint32_t,4> const & b)->bool {
     return a[0]==b[0] && a[1]==b[1]; // in this context we do not care of [2]
  };

  auto const * b = sl.links_d;
  auto const * e = b+n;

  auto p = cuda_std::lower_bound(b,e,me,less);
  int32_t j = p-sl.links_d;
  assert(j>=0);

  auto getTK = [&](int i) { auto const & l = sl.links_d[i]; return l[2];};

  j = std::min(int(j),int(n-1));
  if (equal(me,sl.links_d[j])) {
    auto const itk = j;
    auto const tk = getTK(j);
    auto old = atomicCAS(&sl.tkId_d[cl],invTK,itk);
    if (invTK==old || tk==getTK(old)) { 
       atomicAdd(&sl.n1_d[cl],1);
//         sl.n1_d[cl] = tk;
    } else {
      auto old = atomicCAS(&sl.tkId2_d[cl],invTK,itk);
      if (invTK==old || tk==getTK(old)) atomicAdd(&sl.n2_d[cl],1);
    }
//    if (92==tk) printf("TK3: %d %d %d  %d: %d,%d ?%d?%d\n", j, cl, id, i, dd.xx_d[i], dd.yy_d[i], hh.mr_d[cl], hh.mc_d[cl]);
  } 
  /*
  else {
    auto const & k=sl.links_d[j];
    auto const & kk = j+1<n ? sl.links_d[j+1] : k;
    printf("digi not found %d:%d closest %d:%d:%d, %d:%d:%d\n",id,ch, k[0],k[1],k[2], kk[0],kk[1],kk[2]);
  }
  */

}

__global__
void verifyZero(int ev, clusterSLOnGPU::DigisOnGPU const * ddp, clusterSLOnGPU::HitsOnGPU const * hhp, uint32_t nhits, ClusterSLGPU const * slp) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i>nhits) return;

//  auto const & dd = *ddp;
//  auto const & hh = *hhp;
  auto const & sl = *slp;

  assert(sl.tkId_d[i]==0);
  auto const & tk = sl.links_d[0];
  assert(tk[0]==0);
  assert(tk[1]==0);
  assert(tk[2]==0);
  assert(tk[3]==0);

  // if (i==0) printf("xx_d gpu %x\n",dd.xx_d);

}


__global__
void dumpLink(int first, int ev, clusterSLOnGPU::HitsOnGPU const * hhp, uint32_t nhits, ClusterSLGPU const * slp) {
  auto i = first + blockIdx.x*blockDim.x + threadIdx.x;
  if (i>nhits) return;

  auto const & hh = *hhp;
  auto const & sl = *slp;

  auto const & tk1 = sl.links_d[sl.tkId_d[i]];
  auto const & tk2 = sl.links_d[sl.tkId2_d[i]];

  printf("HIT: %d %d %d %d %f %f %f %f %d %d %d %d %d %d %d\n",ev, i, 
         hh.detInd_d[i], hh.charge_d[i], 
         hh.xg_d[i],hh.yg_d[i],hh.zg_d[i],hh.rg_d[i],hh.iphi_d[i], 
         tk1[2],tk1[3],sl.n1_d[i],
         tk2[2],tk2[3],sl.n2_d[i]
        );

}



namespace clusterSLOnGPU {

 constexpr uint32_t invTK = 0; // std::numeric_limits<int32_t>::max();

 void printCSVHeader() {
      printf("HIT: %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s\n", "ev", "ind",
         "det", "charge",	
         "xg","yg","zg","rg","iphi", 
         "tkId","pt","n1","tkId2","pt2","n2" 
        );
     }


  std::atomic<int> evId(0);
  std::once_flag doneCSVHeader;

  Kernel::Kernel(cuda::stream_t<>& stream, bool dump) : doDump(dump) {
    if (doDump) std::call_once(doneCSVHeader,printCSVHeader);
    alloc(stream);
  }


  void
  Kernel::alloc(cuda::stream_t<>& stream) {
   cudaCheck(cudaMalloc((void**) & slgpu.links_d,(ClusterSLGPU::MAX_DIGIS)*sizeof(std::array<uint32_t,4>)));

   cudaCheck(cudaMalloc((void**) & slgpu.tkId_d,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));
   cudaCheck(cudaMalloc((void**) & slgpu.tkId2_d,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));
   cudaCheck(cudaMalloc((void**) & slgpu.n1_d,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));
   cudaCheck(cudaMalloc((void**) & slgpu.n2_d,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t)));


   cudaCheck(cudaMalloc((void**) & slgpu.me_d, sizeof(ClusterSLGPU)));
   cudaCheck(cudaMemcpyAsync(slgpu.me_d, &slgpu, sizeof(ClusterSLGPU), cudaMemcpyDefault, stream.id()));
  }

 void
  Kernel::deAlloc() {
   cudaCheck(cudaFree(slgpu.links_d));
   cudaCheck(cudaFree(slgpu.tkId_d));
   cudaCheck(cudaFree(slgpu.tkId2_d));
   cudaCheck(cudaFree(slgpu.n1_d));
   cudaCheck(cudaFree(slgpu.n2_d));
   cudaCheck(cudaFree(slgpu.me_d));
}


  void
  Kernel::zero(cudaStream_t stream) {
   cudaCheck(cudaMemsetAsync(slgpu.tkId_d,invTK,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
   cudaCheck(cudaMemsetAsync(slgpu.tkId2_d,invTK,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
   cudaCheck(cudaMemsetAsync(slgpu.n1_d,0,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
   cudaCheck(cudaMemsetAsync(slgpu.n2_d,0,(ClusterSLGPU::MaxNumModules*256)*sizeof(uint32_t), stream));
  }


  void 
  Kernel::algo(DigisOnGPU const & dd, uint32_t ndigis, HitsOnCPU const & hh, uint32_t nhits, uint32_t n, cuda::stream_t<>& stream) {
    
    /*
    size_t pfs = 16*1024*1024;
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize,pfs);
    cudaDeviceGetLimit(&pfs,cudaLimitPrintfFifoSize);
    std::cout << "cudaLimitPrintfFifoSize " << pfs << std::endl;
    */

    zero(stream.id());

    ClusterSLGPU const & sl = slgpu;

    int ev = ++evId;
    int threadsPerBlock = 256;

    int blocks = (nhits + threadsPerBlock - 1) / threadsPerBlock;
    verifyZero<<<blocks, threadsPerBlock, 0, stream.id()>>>(ev, dd.me_d, hh.gpu_d, nhits, sl.me_d);


    blocks = (ndigis + threadsPerBlock - 1) / threadsPerBlock;

    assert(sl.me_d);
    simLink<<<blocks, threadsPerBlock, 0, stream.id()>>>(dd.me_d,ndigis, hh.gpu_d, sl.me_d,n);

    if (doDump) {
      cudaStreamSynchronize(stream.id());	// flush previous printf
      // one line == 200B so each kernel can print only 5K lines....
      blocks = 16; // (nhits + threadsPerBlock - 1) / threadsPerBlock;
      for (int first=0; first<int(nhits); first+=blocks*threadsPerBlock) {
        dumpLink<<<blocks, threadsPerBlock, 0, stream.id()>>>(first, ev, hh.gpu_d, nhits, sl.me_d);
        cudaStreamSynchronize(stream.id());
      }
    }
    cudaCheck(cudaGetLastError());

  }

}
