#define CUDA_KERNELS_ON_CPU // not needed yet
#include "gpuVertexFinderImpl.h"
namespace gpuVertexFinder {
template<>
    ZVertexHeterogeneous Producer::make<cudaCompat::CPUTraits>(cudaStream_t stream, TkSoA const* tksoa, float ptMin, bool onGPU) const {
       assert(!onGPU);
       return makeImpl<cudaCompat::CPUTraits>(stream,tksoa,ptMin);
    }
}
