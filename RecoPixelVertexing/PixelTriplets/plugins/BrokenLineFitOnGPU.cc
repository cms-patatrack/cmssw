#include "BrokenLineFitOnGPU.h"

template
void HelixFitOnGPU::launchBrokenLineKernels<cudaCompat::CPUTraits>(HitsView const *hv,
                                            uint32_t hitsInFit,
                                            uint32_t maxNumberOfTuples,
                                            cudaStream_t stream);
