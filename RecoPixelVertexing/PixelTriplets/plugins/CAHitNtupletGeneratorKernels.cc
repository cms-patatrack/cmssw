#include "RecoPixelVertexing/PixelTriplets/plugins/CAHitNtupletGeneratorKernelsImpl.h"


template
void CAHitNtupletGeneratorKernelsCPU::fillHitDetIndices(HitsView const *hv, TkSoA *tracks_d, cudaStream_t cudaStream);

template
void CAHitNtupletGeneratorKernelsCPU::launchKernels(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream);

template
void CAHitNtupletGeneratorKernelsCPU::buildDoublets(HitsOnCPU const &hh, cudaStream_t stream);

template
void CAHitNtupletGeneratorKernelsCPU::classifyTuples(HitsOnCPU const &hh, TkSoA *tracks_d, cudaStream_t cudaStream);

template
void CAHitNtupletGeneratorKernelsCPU::printCounters(Counters const *counters);
