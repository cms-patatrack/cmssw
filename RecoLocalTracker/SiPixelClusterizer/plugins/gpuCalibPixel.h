#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <cstdint>
#include <cstdio>

#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cuda_assert.h"

#include "gpuClusteringConstants.h"

namespace gpuCalibPixel {

  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  // valid for run2
  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220
  
  constexpr float    ElectronPerADCGain = 600;
  constexpr int8_t   Phase2ReadoutMode  = -1;
  constexpr uint16_t Phase2DigiBaseline = 1000;
  constexpr uint8_t  Phase2KinkADC      = 8;

  __global__ void calibDigis(bool isRun2,
                             uint16_t* id,
                             uint16_t const* __restrict__ x,
                             uint16_t const* __restrict__ y,
                             uint16_t* adc,
                             SiPixelGainForHLTonGPU const* __restrict__ ped,
                             int numElements,
                             uint32_t* __restrict__ moduleStart,        // just to zero first
                             uint32_t* __restrict__ nClustersInModule,  // just to zero them
                             uint32_t* __restrict__ clusModuleStart     // just to zero first
  ) {
    int first = blockDim.x * blockIdx.x + threadIdx.x;

    // zero for next kernels...
    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < gpuClustering::MaxNumModules; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

    for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      if (InvId == id[i])
        continue;

      float conversionFactor = (isRun2) ? (id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain) : 1.f;
      float offset = (isRun2) ? (id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset) : 0;

      bool isDeadColumn = false, isNoisyColumn = false;

      int row = x[i];
      int col = y[i];
      auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
      float pedestal = ret.first;
      float gain = ret.second;
      // float pedestal = 0; float gain = 1.;
      if (isDeadColumn | isNoisyColumn) {
        id[i] = InvId;
        adc[i] = 0;
        printf("bad pixel at %d in %d\n", i, id[i]);
      } else {
        float vcal = adc[i] * gain - pedestal * gain;
        adc[i] = std::max(100, int(vcal * conversionFactor + offset));
      }
    }
  }
 __global__ void calibDigisUpgrade(
                             uint16_t *xx, uint16_t *yy,
                             uint16_t *adc, uint32_t *pdigi,
                             uint16_t *id,int numElements,
			     uint32_t* __restrict__ moduleStart,        // just to zero first
                             uint32_t* __restrict__ nClustersInModule,  // just to zero them
                             uint32_t* __restrict__ clusModuleStart)
   {
    int first = blockDim.x * blockIdx.x + threadIdx.x;

    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < gpuClustering::MaxNumModulesUpgrade; i += gridDim.x * blockDim.x) {
      nClustersInModule[i] = 0;
    }

   for (int i = first; i < numElements; i += gridDim.x * blockDim.x) {
      
      if (InvId == id[i])
        continue;
      int mode = (Phase2ReadoutMode < -1 ? -1 : Phase2ReadoutMode);

      if(mode < 0)
        adc[i] = std::max(100, int(adc[i] * ElectronPerADCGain));
      else
      {
        if (adc[i] < Phase2KinkADC)
          adc[i] = int((adc[i] - 0.5) * ElectronPerADCGain);
        else
        {
          constexpr int8_t dspp = (Phase2ReadoutMode < 10 ? Phase2ReadoutMode : 10);
          constexpr int8_t ds   = int8_t(dspp <= 1 ? 1 : (dspp - 1) * (dspp - 1));

          adc[i] -= (Phase2KinkADC - 1);
          adc[i] *= ds;
          adc[i] += (Phase2KinkADC - 1);

          adc[i] = uint16_t((adc[i] - 0.5 * ds) * ElectronPerADCGain);
        }

        adc[i] += int(Phase2DigiBaseline);
        adc[i] = std::max(uint16_t(100),adc[i]);
        }
    }
  }
  
}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
