#include "RiemannFitOnGPU.h"

template
void HelixFitOnGPU::launchRiemannKernels<cudaCompat::CPUTraits>(HitsView const *hv,
                                         uint32_t nhits,
                                         uint32_t maxNumberOfTuples,
                                         cudaStream_t stream);
