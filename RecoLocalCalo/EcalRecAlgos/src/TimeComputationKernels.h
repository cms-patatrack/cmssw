#ifndef RecoLocalCalo_EcalRecAlgos_src_TimeComputationKernels
#define RecoLocalCalo_EcalRecAlgos_src_TimeComputationKernels

#include <iostream>
#include <limits>

#include "RecoLocalCalo/EcalRecAlgos/interface/Common.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/DeclsForKernels.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes_gpu.h"

#include "DataFormats/Math/interface/approx_exp.h"
#include "DataFormats/Math/interface/approx_log.h"

#include "cuda.h"

//#define DEBUG

//#define ECAL_RECO_CUDA_DEBUG

namespace ecal { namespace multifit {

#define RUN_NULLHYPOT
#ifdef RUN_NULLHYPOT
__global__
void kernel_time_compute_nullhypot(SampleVector::Scalar const* sample_values,
                                   SampleVector::Scalar const* sample_value_errors,
                                   bool const* useless_sample_values,
                                   SampleVector::Scalar* chi2s,
                                   SampleVector::Scalar* sum0s,
                                   SampleVector::Scalar* sumAAs,
                                   int const nchannels);
#endif

#define RUN_MAKERATIO
#ifdef RUN_MAKERATIO
//
// launch ctx parameters are 
// 45 threads per channel, X channels per block, Y blocks
// 45 comes from: 10 samples for i <- 0 to 9 and for j <- i+1 to 9
// TODO: it might be much beter to use 32 threads per channel instead of 45
// to simplify the synchronization
//
__global__
void kernel_time_compute_makeratio(SampleVector::Scalar const* sample_values,
                                   SampleVector::Scalar const* sample_value_errors,
                                   bool const* useless_sample_values,
                                   char const* pedestal_nums,
                                   SampleVector::Scalar const* amplitudeFitParameters,
                                   SampleVector::Scalar const* timeFitParameters,
                                   SampleVector::Scalar const* sumAAsNullHypot,
                                   SampleVector::Scalar const* sum0sNullHypot,
                                   SampleVector::Scalar* tMaxAlphaBetas,
                                   SampleVector::Scalar* tMaxErrorAlphaBetas,
                                   SampleVector::Scalar* g_accTimeMax,
                                   SampleVector::Scalar* g_accTimeWgt,
                                   TimeComputationState* g_state,
                                   unsigned int const timeFitParameters_size,
                                   SampleVector::Scalar const timeFitLimits_first,
                                   SampleVector::Scalar const timeFitLimits_second,
                                   int const nchannels);
#endif

/// launch ctx parameters are 
/// 10 threads per channel, N channels per block, Y blocks
/// TODO: do we need to keep the state around or can be removed?!
//#define DEBUG_FINDAMPLCHI2_AND_FINISH
#define RUN_FINDAMPLCHI2_AND_FINISH
#ifdef RUN_FINDAMPLCHI2_AND_FINISH
__global__
void kernel_time_compute_findamplchi2_and_finish(
        SampleVector::Scalar const* sample_values,
        SampleVector::Scalar const* sample_value_errors,
        bool const* useless_samples,
        SampleVector::Scalar const* g_tMaxAlphaBeta,
        SampleVector::Scalar const* g_tMaxErrorAlphaBeta,
        SampleVector::Scalar const* g_accTimeMax,
        SampleVector::Scalar const* g_accTimeWgt,
        SampleVector::Scalar const* amplitudeFitParameters,
        SampleVector::Scalar const* sumAAsNullHypot,
        SampleVector::Scalar const* sum0sNullHypot,
        SampleVector::Scalar const* chi2sNullHypot,
        TimeComputationState* g_state,
        SampleVector::Scalar* g_ampMaxAlphaBeta,
        SampleVector::Scalar* g_ampMaxError,
        SampleVector::Scalar* g_timeMax,
        SampleVector::Scalar* g_timeError,
        int const nchannels);
#endif

__global__
void kernel_time_compute_fixMGPAslew(uint16_t const* digis,
                                     SampleVector::Scalar* sample_values,
                                     SampleVector::Scalar* sample_value_errors,
                                     bool* useless_sample_values,
                                     unsigned int const sample_mask,
                                     int const nchannels);

#define RUN_AMPL
#ifdef RUN_AMPL
__global__
void kernel_time_compute_ampl(SampleVector::Scalar const* sample_values,
                              SampleVector::Scalar const* sample_value_errors,
                              bool const* useless_samples,
                              SampleVector::Scalar const* g_timeMax,
                              SampleVector::Scalar const* amplitudeFitParameters,
                              SampleVector::Scalar *g_amplitudeMax,
                              int const nchannels);
#endif

//#define ECAL_RECO_CUDA_TC_INIT_DEBUG
__global__
void kernel_time_computation_init(uint16_t const* digis,
                                  uint32_t const* dids,
                                  float const* rms_x12,
                                  float const* rms_x6,
                                  float const* rms_x1,
                                  float const* mean_x12,
                                  float const* mean_x6,
                                  float const* mean_x1,
                                  float const* gain12Over6,
                                  float const* gain6Over1,
                                  SampleVector::Scalar* sample_values,
                                  SampleVector::Scalar* sample_value_errors,
                                  SampleVector::Scalar* ampMaxError,
                                  bool* useless_sample_values,
                                  char* pedestal_nums,
                                  unsigned int const sample_mask,
                                  int nchannels);

///
/// launch context parameters: 1 thread per channel
///
//#define DEBUG_TIME_CORRECTION
__global__
void kernel_time_correction_and_finalize(
//        SampleVector::Scalar const* g_amplitude,
        float const* g_amplitude,
        float const* amplitudeBins,
        float const* shiftBins,
        SampleVector::Scalar const* g_timeMax,
        SampleVector::Scalar const* g_timeError,
        float *g_jitter,
        float *g_jitterError,
        int const amplitudeBinsSize,
        SampleVector::Scalar const timeConstantTerm,
        int const nchannels);

}}

#endif // RecoLocalCalo_EcalRecAlgos_src_TimeComputationKernels
