#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMCharacteristicsRcd.h"
#include "CondFormats/DataRecord/interface/HcalSiPMParametersRcd.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidths.h"
#include "CondFormats/HcalObjects/interface/HcalGainWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalGainsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalLUTCorrsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalQIECodersGPU.h"
#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypes.h"
#include "CondFormats/HcalObjects/interface/HcalQIETypesGPU.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalRecoParamsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristics.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMCharacteristicsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParameters.h"
#include "CondFormats/HcalObjects/interface/HcalSiPMParametersGPU.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorrs.h"
#include "CondFormats/HcalObjects/interface/HcalTimeCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsWithPulseShapesGPU.h"

#include "HcalESProducerGPU.h"

using HcalRecoParamsGPUESProducer = HcalESProducerGPU<HcalRecoParamsRcd, HcalRecoParamsGPU, HcalRecoParams>;

using HcalRecoParamsWithPulseShapesGPUESProducer =
    HcalESProducerGPU<HcalRecoParamsRcd, HcalRecoParamsWithPulseShapesGPU, HcalRecoParams>;

using HcalPedestalsGPUESProducer = HcalESProducerGPU<HcalPedestalsRcd, HcalPedestalsGPU, HcalPedestals>;

using HcalGainsGPUESProducer = HcalESProducerGPU<HcalGainsRcd, HcalGainsGPU, HcalGains>;

using HcalLUTCorrsGPUESProducer = HcalESProducerGPU<HcalLUTCorrsRcd, HcalLUTCorrsGPU, HcalLUTCorrs>;

using HcalRespCorrsGPUESProducer = HcalESProducerGPU<HcalRespCorrsRcd, HcalRespCorrsGPU, HcalRespCorrs>;

using HcalTimeCorrsGPUESProducer = HcalESProducerGPU<HcalTimeCorrsRcd, HcalTimeCorrsGPU, HcalTimeCorrs>;

using HcalPedestalWidthsGPUESProducer =
    HcalESProducerGPU<HcalPedestalWidthsRcd, HcalPedestalWidthsGPU, HcalPedestalWidths>;

using HcalGainWidthsGPUESProducer = HcalESProducerGPU<HcalGainWidthsRcd, HcalGainWidthsGPU, HcalGainWidths>;

using HcalQIECodersGPUESProducer = HcalESProducerGPU<HcalQIEDataRcd, HcalQIECodersGPU, HcalQIEData>;

using HcalQIETypesGPUESProducer = HcalESProducerGPU<HcalQIETypesRcd, HcalQIETypesGPU, HcalQIETypes>;

using HcalSiPMParametersGPUESProducer =
    HcalESProducerGPU<HcalSiPMParametersRcd, HcalSiPMParametersGPU, HcalSiPMParameters>;

using HcalSiPMCharacteristicsGPUESProducer =
    HcalESProducerGPU<HcalSiPMCharacteristicsRcd, HcalSiPMCharacteristicsGPU, HcalSiPMCharacteristics>;

DEFINE_FWK_EVENTSETUP_MODULE(HcalRecoParamsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalRecoParamsWithPulseShapesGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalPedestalsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalGainsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalLUTCorrsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalRespCorrsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalTimeCorrsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalPedestalWidthsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalGainWidthsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalQIECodersGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalQIETypesGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalSiPMParametersGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalSiPMCharacteristicsGPUESProducer);

#include "CondFormats/HcalObjects/interface/HcalCombinedRecordsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedPedestalsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedEffectivePedestalsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedPedestalWidthsGPU.h"
#include "CondFormats/HcalObjects/interface/HcalConvertedEffectivePedestalWidthsGPU.h"

using HcalConvertedPedestalsGPUESProducer = HcalESProducerGPUWithDependencies<HcalConvertedPedestalsRcd,
                                                                              HcalConvertedPedestalsGPU,
                                                                              HcalPedestals,
                                                                              HcalQIEData,
                                                                              HcalQIETypes>;

using HcalConvertedEffectivePedestalsGPUESProducer =
    HcalESProducerGPUWithDependencies<HcalConvertedEffectivePedestalsRcd,
                                      HcalConvertedEffectivePedestalsGPU,
                                      HcalPedestals,
                                      HcalQIEData,
                                      HcalQIETypes>;

using HcalConvertedPedestalWidthsGPUESProducer = HcalESProducerGPUWithDependencies<HcalConvertedPedestalWidthsRcd,
                                                                                   HcalConvertedPedestalWidthsGPU,
                                                                                   HcalPedestals,
                                                                                   HcalPedestalWidths,
                                                                                   HcalQIEData,
                                                                                   HcalQIETypes>;

using HcalConvertedEffectivePedestalWidthsGPUESProducer =
    HcalESProducerGPUWithDependencies<HcalConvertedEffectivePedestalWidthsRcd,
                                      HcalConvertedEffectivePedestalWidthsGPU,
                                      HcalPedestals,
                                      HcalPedestalWidths,
                                      HcalQIEData,
                                      HcalQIETypes>;

DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedPedestalsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedEffectivePedestalsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedPedestalWidthsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalConvertedEffectivePedestalWidthsGPUESProducer);
