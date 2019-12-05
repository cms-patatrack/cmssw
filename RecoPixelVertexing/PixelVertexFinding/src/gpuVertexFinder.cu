#include "gpuVertexFinderImpl.h"
namespace gpuVertexFinder {
template<>
    ZVertexHeterogeneous Producer::make<cudaCompat::GPUTraits>(cudaStream_t stream, TkSoA const* tksoa, float ptMin, bool onGPU) const {
       assert(onGPU);
       return makeImpl<cudaCompat::GPUTraits>(stream,tksoa,ptMin);
    }
}
