#ifndef RecoPixelVertexing_PixelVertexFinding_gpuVertexFinder_H
#define RecoPixelVertexing_PixelVertexFinding_gpuVertexFinder_H

#include<cstdint>

#include "RecoPixelVertexing/PixelVertexFinding/interface/pixelVertexHeterogeneousProduct.h"


namespace gpuVertexFinder {

  struct OnGPU {

    static constexpr uint32_t MAXTRACKS = 16000;
    static constexpr uint32_t MAXVTX= 1024;
    
    float * zt;
    float * ezt2;
    float * zv;
    float * wv;
    uint32_t * nv;
    int32_t * iv;

    // workspace  
    int8_t  * izt;
    uint16_t * nn;
    
  };
  

  class Producer {
  public:
    
    using GPUProduct = pixelVertexHeterogeneousProduct::GPUProduct;
    using OnGPU = gpuVertexFinder::OnGPU;


    Producer(
	     int iminT,  // min number of neighbours to be "core"
	     float ieps, // max absolute distance to cluster
	     float ierrmax, // max error to be "seed"
	     float ichi2max   // max normalized distance to cluster
	     ) :
      minT(iminT),
      eps(ieps),
      errmax(ierrmax),
      chi2max(ichi2max)  
    {}
    
    ~Producer() { deallocateOnGPU();}
    
    void produce(cudaStream_t stream,
		 float const * zt,
		 float const * ezt2,
		 uint32_t ntrks
		 );
    
    GPUProduct const & fillResults(cudaStream_t stream);
    

    void allocateOnGPU();
    void deallocateOnGPU();

  private:
    GPUProduct gpuProduct;
    OnGPU onGPU;
    OnGPU * onGPU_d=nullptr;

    int minT;  // min number of neighbours to be "core"
    float eps; // max absolute distance to cluster
    float errmax; // max error to be "seed"
    float chi2max;   // max normalized distance to cluster

  };
  
} // end namespace

#endif
