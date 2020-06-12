#ifndef EcalUncalibRecHitMultiFitAlgo_gpu_new_h
#define EcalUncalibRecHitMultiFitAlgo_gpu_new_h

#include <vector>

#include <cuda.h>

#include "DeclsForKernels.h"

namespace ecal {
  namespace multifit {

    void entryPoint(EventInputDataGPU const&,
                    EventOutputDataGPU&,
                    EventDataForScratchGPU&,
                    ConditionsProducts const&,
                    ConfigurationParameters const&,
                    cudaStream_t);

  }
}  // namespace ecal

#endif // EcalUncalibRecHitMultiFitAlgo_gpu_new_h
