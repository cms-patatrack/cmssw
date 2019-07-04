#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEMahiGPU.h"

#include <cuda/api_wrappers.h>

namespace hcal { namespace mahi {

void entryPoint(
        InputDataCPU const& inputDataCPU,
        InputDataGPU& inputDataGPU,
        ConditionsProducts const& conditions,
        ConfigParameters const& configParameters,
        cuda::stream_t<>& cudaStream) {
    // TODO: data provided by the unpacker (for QIE 8 only)
    // should be in slightly different format
    // that is gpu friendly (for memory accesses coalescing)
    std::vector<uint32_t> detIdsQ8(inputDataCPU.digisQ8.size());
    std::vector<uint16_t> dataQ8(inputDataCPU.digisQ8.size() * max_samples);

    // FIXME: remove debugging
    std::cout << "nqie8 = " << detIdsQ8.size()
              << "  nqie11 = " << inputDataCPU.digisQ11.size() << std::endl;

    // TODO: see above
    // copy from hbhe digi collection
    for (uint32_t i=0; i<inputDataCPU.digisQ8.size(); ++i) {
        detIdsQ8[i] = inputDataCPU.digisQ8[i].id().rawId();
        for (uint32_t sample=0; sample<max_samples; ++sample) 
            dataQ8[i*max_samples + sample] = inputDataCPU.digisQ8[i].sample(sample).raw();
    }

    // transfer 
    cudaCheck( cudaMemcpyAsync(inputDataGPU.ids,
                               detIdsQ8.data(),
                               detIdsQ8.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice,
                               cudaStream.id()) );
    cudaCheck( cudaMemcpyAsync(inputDataGPU.ids + detIdsQ8.size(),
                               inputDataCPU.digisQ11.ids().data(),
                               inputDataCPU.digisQ11.ids().size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice,
                               cudaStream.id()) );
    cudaCheck( cudaMemcpyAsync(inputDataGPU.data,
                               dataQ8.data(),
                               dataQ8.size() * sizeof(uint16_t),
                               cudaMemcpyHostToDevice,
                               cudaStream.id()) );
    cudaCheck( cudaMemcpyAsync(inputDataGPU.data + dataQ8.size(),
                               inputDataCPU.digisQ11.data().data(),
                               inputDataCPU.digisQ11.data().size() * 
                               sizeof(uint16_t),
                               cudaMemcpyHostToDevice,
                               cudaStream.id()) );
}

}}
