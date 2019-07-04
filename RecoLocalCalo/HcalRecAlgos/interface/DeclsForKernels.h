#ifndef RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h
#define RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

namespace hcal { namespace mahi {

struct InputDataCPU {
    HBHEDigiCollection const& digisQ8;
    QIE11DigiCollection const& digisQ11;
};

}}

#endif // RecoLocalCalo_HcalRecAlgos_interface_DeclsForKernels_h
