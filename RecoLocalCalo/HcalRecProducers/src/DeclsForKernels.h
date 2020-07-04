#ifndef RecoLocalCalo_HcalRecProducers_src_DeclsForKernels_h
#define RecoLocalCalo_HcalRecProducers_src_DeclsForKernels_h

#include <functional>
#include <optional>

#include "CUDADataFormats/HcalDigi/interface/DigiCollection.h"
#include "CUDADataFormats/HcalRecHitSoA/interface/RecHitCollection.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedEffectivePedestalWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedEffectivePedestalsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedPedestalWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalConvertedPedestalsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalLUTCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIECodersGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIETypesGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsWithPulseShapesGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRespCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSiPMCharacteristicsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSiPMParametersGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimeCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalMahiPulseOffsetsGPU.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

namespace hcal {
  namespace mahi {

    struct ConditionsProducts {
      HcalGainWidthsGPU::Product const& gainWidths;
      HcalGainsGPU::Product const& gains;
      HcalLUTCorrsGPU::Product const& lutCorrs;
      HcalConvertedPedestalWidthsGPU::Product const& pedestalWidths;
      HcalConvertedEffectivePedestalWidthsGPU::Product const& effectivePedestalWidths;
      HcalConvertedPedestalsGPU::Product const& pedestals;
      HcalQIECodersGPU::Product const& qieCoders;
      HcalRecoParamsWithPulseShapesGPU::Product const& recoParams;
      HcalRespCorrsGPU::Product const& respCorrs;
      HcalTimeCorrsGPU::Product const& timeCorrs;
      HcalQIETypesGPU::Product const& qieTypes;
      HcalSiPMParametersGPU::Product const& sipmParameters;
      HcalSiPMCharacteristicsGPU::Product const& sipmCharacteristics;
      HcalConvertedPedestalsGPU::Product const* convertedEffectivePedestals;
      HcalTopology const* topology;
      HcalDDDRecConstants const* recConstants;
      uint32_t offsetForHashes;
      HcalMahiPulseOffsetsGPU::Product const& pulseOffsets;
      std::vector<int, cms::cuda::HostAllocator<int>> const& pulseOffsetsHost;
    };

    struct ConfigParameters {
      uint32_t maxChannels;
      uint32_t maxTimeSamples;
      uint32_t kprep1dChannelsPerBlock;
      int sipmQTSShift;
      int sipmQNTStoSum;
      int firstSampleShift;
      bool useEffectivePedestals;

      float meanTime;
      float timeSigmaSiPM, timeSigmaHPD;
      float ts4Thresh;

      std::array<uint32_t, 3> kernelMinimizeThreads;

      // FIXME:
      //   - add "getters" to HcalTimeSlew calib formats
      //   - add ES Producer to consume what is produced above not to replicate.
      //   which ones to use is hardcoded, therefore no need to send those to the device
      bool applyTimeSlew;
      float tzeroTimeSlew, slopeTimeSlew, tmaxTimeSlew;
    };

    struct OutputDataGPU {
      RecHitCollection<common::DevStoragePolicy> recHits;

      void allocate(ConfigParameters const& config, cudaStream_t cudaStream) {
        recHits.energy = cms::cuda::make_device_unique<float[]>(
          config.maxChannels, cudaStream);
        recHits.chi2 = cms::cuda::make_device_unique<float[]>(
          config.maxChannels, cudaStream);
        recHits.energyM0 = cms::cuda::make_device_unique<float[]>(
          config.maxChannels, cudaStream);
        recHits.timeM0 = cms::cuda::make_device_unique<float[]>(
          config.maxChannels, cudaStream);
        recHits.did = cms::cuda::make_device_unique<uint32_t[]>(
          config.maxChannels, cudaStream);
      }
    };

    struct ScratchDataGPU {
      cms::cuda::device::unique_ptr<float[]> amplitudes, noiseTerms,
          pulseMatrices, pulseMatricesM, pulseMatricesP;
      cms::cuda::device::unique_ptr<int8_t[]> soiSamples;

      /*
      float *amplitudes = nullptr, *noiseTerms = nullptr;
      float *pulseMatrices = nullptr, *pulseMatricesM = nullptr, *pulseMatricesP = nullptr;
      int8_t* soiSamples = nullptr;

      // TODO: properly allocate for NSAMPLES VS NPULSES
      void allocate(ConfigParameters const& config) {
        cudaCheck(cudaMalloc((void**)&amplitudes, sizeof(float) * config.maxChannels * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&noiseTerms, sizeof(float) * config.maxChannels * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&pulseMatrices,
                             sizeof(float) * config.maxChannels * config.maxTimeSamples * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&pulseMatricesM,
                             sizeof(float) * config.maxChannels * config.maxTimeSamples * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&pulseMatricesP,
                             sizeof(float) * config.maxChannels * config.maxTimeSamples * config.maxTimeSamples));
        cudaCheck(cudaMalloc((void**)&soiSamples, sizeof(int8_t) * config.maxChannels));
      }

      void deallocate(ConfigParameters const& config) {
        if (amplitudes) {
          cudaCheck(cudaFree(amplitudes));
          cudaCheck(cudaFree(noiseTerms));
          cudaCheck(cudaFree(pulseMatrices));
          cudaCheck(cudaFree(pulseMatricesM));
          cudaCheck(cudaFree(pulseMatricesP));
          cudaCheck(cudaFree(soiSamples));
        }
      }*/
    };

    struct InputDataGPU {
      DigiCollection<Flavor01, common::DevStoragePolicy> const& f01HEDigis;
      DigiCollection<Flavor5, common::DevStoragePolicy> const& f5HBDigis;
      DigiCollection<Flavor3, common::DevStoragePolicy> const& f3HBDigis;
    };

  }  // namespace mahi
}  // namespace hcal

#endif  // RecoLocalCalo_HcalRecProducers_src_DeclsForKernels_h
