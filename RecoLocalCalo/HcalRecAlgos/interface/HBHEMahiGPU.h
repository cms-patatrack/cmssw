#ifndef RecoLocalCalo_HcalRecAlgos_interface_HBHEMahiGPU_h
#define RecoLocalCalo_HcalRecAlgos_interface_HBHEMahiGPU_h

#include "RecoLocalCalo/HcalRecAlgos/interface/DeclsForKernels.h"

#include <cuda/api_wrappers.h>

namespace hcal { namespace mahi {

void entryPoint(
        InputDataCPU const&, InputDataGPU&, ConditionsProducts const&,
        ConfigParameters const&, cuda::stream_t<>&);

}}

#endif // RecoLocalCalo_HcalRecAlgos_interface_HBHEMahiGPU_h
