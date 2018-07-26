#ifndef SimTrackerTrackerHitAssociationClusterHeterogeneousProduct_H
#define SimTrackerTrackerHitAssociationClusterHeterogeneousProduct_H

#ifndef	__NVCC__
#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#endif

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"


namespace trackerHitAssociationHeterogeneousProduct {

#ifndef __NVCC__
  struct CPUProduct {
    CPUProduct() = default;
    template<typename T> 
    explicit CPUProduct(T const & t) : collection(t){}
    ClusterTPAssociation collection;
  };
#endif

  struct ClusterSLGPU {

   ClusterSLGPU * me_d=nullptr;
   std::array<uint32_t,4> * links_d;
   uint32_t * tkId_d;
   uint32_t * tkId2_d;
   uint32_t * n1_d;
   uint32_t * n2_d;

   static constexpr uint32_t MAX_DIGIS = 2000*150;
   static constexpr uint32_t MaxNumModules = 2000;

  };

  struct GPUProduct {
     ClusterSLGPU *  gpu_d=nullptr;
  };

#ifndef	__NVCC__
   using ClusterTPAHeterogeneousProduct = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                                   heterogeneous::GPUCudaProduct<GPUProduct> >;
#endif

}

#endif
