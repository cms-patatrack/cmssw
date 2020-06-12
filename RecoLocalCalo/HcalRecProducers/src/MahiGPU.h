#ifndef MahiGPU_h
#define MahiGPU_h

#include "DeclsForKernels.h"

namespace hcal {
  namespace mahi {

    void entryPoint(InputDataGPU const&,
                    OutputDataGPU&,
                    ConditionsProducts const&,
                    ScratchDataGPU&,
                    ConfigParameters const&,
                    cudaStream_t);

  }
}  // namespace hcal

#endif // MahiGPU_h
