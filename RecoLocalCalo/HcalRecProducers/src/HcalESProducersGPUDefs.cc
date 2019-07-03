#include "HcalESProducerGPU.h"

#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalsGPU.h"

#include <iostream>

using HcalRecoParamsGPUESProducer = HcalESProducerGPU<HcalRecoParamsGPU,
                                                      HcalRecoParams,
                                                      HcalRecoParamsRcd>;
using HcalPedestalsGPUESProducer = HcalESProducerGPU<HcalPedestalsGPU,
                                                     HcalPedestals,
                                                     HcalPedestalsRcd>;

DEFINE_FWK_EVENTSETUP_MODULE(HcalRecoParamsGPUESProducer);
DEFINE_FWK_EVENTSETUP_MODULE(HcalPedestalsGPUESProducer);
