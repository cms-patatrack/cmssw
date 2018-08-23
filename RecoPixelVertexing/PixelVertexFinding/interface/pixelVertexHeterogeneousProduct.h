#ifndef RecoPixelVertexing_PixelVertexFinding_pixelVertexHeterogeneousProduct_H
#define RecoPixelVertexing_PixelVertexFinding_pixelVertexHeterogeneousProduct_H

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"

namespace pixelVertexHeterogeneousProduct {

  struct CPUProduct {
    reco::VertexCollection; collection;
  };

  struct VerticesOnGPU{
    float * z_d;
    float * zerr_d;
    uint16_t * ntrk_d;
  };


  struct VerticesOnCPU {
    VerticesOnCPU() = default;

    explicit VerticesOnCPU(uint32_t nvtx) :
      z(nvtx),
      zerr(nvtx),
      ntk(nvtx),
      nVertices(nvtx)
    { }

    std::vector<float,    CUDAHostAllocator<float>> z,zerr;
    std::vector<uint16_t, CUDAHostAllocator<uint16_t>> ntrk;

    uint32_t nvtx;
    VerticesOnGPU const * gpu_d = nullptr;
  };


  using GPUProduct = VerticesOnCPU;  // FIXME fill cpu vectors on demand

  using HeterogeneousPixelVertices = HeterogeneousProductImpl<heterogeneous::CPUProduct<CPUProduct>,
                                                              heterogeneous::GPUCudaProduct<GPUProduct> >;
}
#endif
