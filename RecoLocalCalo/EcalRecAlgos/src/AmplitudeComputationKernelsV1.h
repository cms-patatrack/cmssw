#ifndef RecoLocalCalo_EcalRecAlgos_src_AmplitudeComputationKernelsV1
#define RecoLocalCalo_EcalRecAlgos_src_AmplitudeComputationKernelsV1

#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes_gpu.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/DeclsForKernels.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/Common.h"

class EcalPulseShape;
class EcalPulseCovariance;
class EcalUncalibratedRecHit;

namespace ecal { namespace multifit {

///
/// assume kernel launch configuration is 
/// (MAXSAMPLES * nchannels, blocks)
/// TODO: is there a point to split this kernel further to separate reductions
/// 
__global__
void kernel_prep_1d_and_initialize(EcalPulseShape const* shapes_in,
                    FullSampleVector* shapes_out, 
                    uint16_t const* digis_in,
                    SampleVector* amplitudes,
                    SampleVector* amplitudesForMinimization,
                    SampleGainVector* gainsNoise,
                    SampleGainVector* gainsPedestal,
                    float const* mean_x1,
                    float const* mean_x12,
                    float const* rms_x12,
                    float const* mean_x6,
                    float const* gain6Over1,
                    float const* gain12Over6,
                    bool* hasSwitchToGain6,
                    bool* hasSwitchToGain1,
                    bool* isSaturated,
                    float* energies,
                    float* chi2,
                    char* acState,
                    bool gainSwitchUseMaxSample,
                    int nchannels);

///
/// assume kernel launch configuration is 
/// ([MAXSAMPLES, MAXSAMPLES], nchannels)
///
__global__
void kernel_prep_2d(EcalPulseCovariance const* pulse_cov_in,
                    FullSampleMatrix* pulse_cov_out,
                    SampleGainVector const* gainNoise,
                    SampleMatrixD const* noisecorrs,
                    float const* rms_x12,
                    float const* rms_x6,
                    float const* rms_x1,
                    float const* gain12Over6,
                    float const* gain6Over1,
                    SampleMatrix* noisecov,
                    PulseMatrixType* pulse_matrix,
                    FullSampleVector const* pulse_shape,
                    BXVectorType const* bxs,
                    bool const* hasSwitchToGain6,
                    bool const* hasSwitchToGain1,
                    bool const* isSaturated);

///
/// TODO: trivial impl for now, there must be a way to improve
///
/// Conventions:
///   - amplitudes -> solution vector, what we are fitting for
///   - samples -> raw detector responses
///   - passive constraint - satisfied constraint
///   - active constraint - unsatisfied (yet) constraint
///
__global__
void kernel_minimize(SampleMatrix const* noisecov,
                     FullSampleMatrix const* full_pulse_cov,
                     BXVectorType const* bxs,
                     SampleVector const* samples,
                     SampleVector* amplitudes,
                     float* energies,
                     PulseMatrixType* pulse_matrix, 
                     bool* statuses,
                     float* chi2s,
                     bool const* isSaturated,
                     bool const* hasSwitchToGain6,
                     bool const* hasSwitchToGain1,
                     float const* rms_x12,
                     char *acState,
                     int nchannels,
                     int max_iterations, 
                     bool gainSwitchUseMaxSample);

///
/// Build an Ecal RecHit.
/// TODO: Use SoA data structures on the host directly
/// the reason for removing this from minimize kernel is to isolate the minimize + 
/// again, building an aos rec hit involves strides... -> bad memory access pattern
///
#ifdef RUN_BUILD_AOS_RECHIT
__global__
void kernel_build_rechit(
    float const* energies,
    float const* chi2s,
    uint32_t* dids,
    EcalUncalibratedRecHit* rechits,
    int nchannels);
#endif

}}

#endif // RecoLocalCalo_EcalRecAlgos_src_AmplitudeComputationKernelsV1
