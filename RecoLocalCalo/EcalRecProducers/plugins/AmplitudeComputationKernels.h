#ifndef AmplitudeComputationKernels_h
#define AmplitudeComputationKernels_h

#include "EigenMatrixTypes_gpu.h"
#include "DeclsForKernels.h"
#include "Common.h"

class EcalPulseShape;
class EcalPulseCovariance;
class EcalUncalibratedRecHit;

namespace ecal {
  namespace multifit {

    namespace v1 {

      void minimization_procedure(EventInputDataGPU const& eventInputGPU,
                                  EventOutputDataGPU& eventOutputGPU,
                                  EventDataForScratchGPU& scratch,
                                  ConditionsProducts const& conditions,
                                  ConfigurationParameters const& configParameters,
                                  cudaStream_t cudaStream);

    }

  }  // namespace multifit
}  // namespace ecal

#endif  // AmplitudeComputationKernels_h
