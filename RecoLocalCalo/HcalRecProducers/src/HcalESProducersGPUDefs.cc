#include "HcalESProducerGPU.h"

#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsGPU.h"

#include <iostream>

using HcalRecoParamsGPUESProducer = HcalESProducerGPU<HcalRecoParamsGPU,
                                                      HcalRecoParams,
                                                      HcalRecoParamsRcd>;
DEFINE_FWK_EVENTSETUP_MODULE(HcalRecoParamsGPUESProducer);
