#ifndef RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h
#define RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalLUTCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIECodersGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRespCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimeCorrsGPU.h"

namespace hcal { namespace mahi {

constexpr uint32_t max_samples = QIE11DigiCollection::MAXSAMPLES;

struct InputDataCPU {
    HBHEDigiCollection const& digisQ8;
    QIE11DigiCollection const& digisQ11;
};

struct ConditionsProducts {
    HcalGainWidthsGPU::Product const& gainWidths;
    HcalGainsGPU::Product const& gains;
    HcalLUTCorrsGPU::Product const& lutCorrs;
    HcalPedestalWidthsGPU::Product const& pedestalWidths;
    HcalPedestalsGPU::Product const& pedestals;
    HcalQIECodersGPU::Product const& qieCoders;
    HcalRecoParamsGPU::Product const& recoParams;
    HcalRespCorrsGPU::Product const& respCorrs;
    HcalTimeCorrsGPU::Product const& timeCorrs;
};

struct ConfigParameters {
    uint32_t maxChannels;
};

struct InputDataGPU {
    uint32_t *ids=nullptr;
    uint16_t *data=nullptr;

    void allocate(ConfigParameters const& cfg) {
        cudaCheck( cudaMalloc((void**)&ids, cfg.maxChannels * sizeof(uint32_t)) );
        cudaCheck( cudaMalloc((void**)&data, 
                              cfg.maxChannels * max_samples * sizeof(uint16_t)) );
    }

    void deallocate(ConfigParameters const& cfg) {
        if (ids) {
            cudaCheck( cudaFree(ids) );
            cudaCheck( cudaFree(data) );
        }
    }
};

}}

#endif // RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h
