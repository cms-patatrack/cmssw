// framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"
//#include "HeterogeneousCore/Producer/interface/HeterogeneousEvent.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAScopedContext.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsGPU.h"

class HBHERecHitProducerGPU : public edm::stream::EDProducer<edm::ExternalWork>
{
public:
    explicit HBHERecHitProducerGPU(edm::ParameterSet const&);
    ~HBHERecHitProducerGPU() override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
    void acquire(edm::Event const&,
                 edm::EventSetup const&,
                 edm::WaitingTaskWithArenaHolder) override;
    void produce(edm::Event&, edm::EventSetup const&) override;

    uint32_t maxChannels_;
    CUDAContextState cudaState_;
};

HBHERecHitProducerGPU::HBHERecHitProducerGPU(edm::ParameterSet const& ps) 
    : maxChannels_{ps.getParameter<uint32_t>("maxChannels")}
{}

HBHERecHitProducerGPU::~HBHERecHitProducerGPU() {}

void HBHERecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
    edm::ParameterSetDescription desc;
    desc.add<uint32_t>("maxChannels", 10000u);

    std::string label = "hcalRecHitProducerGPU";
    cdesc.add(label, desc);
}

void HBHERecHitProducerGPU::acquire(
        edm::Event const& event,
        edm::EventSetup const& setup,
        edm::WaitingTaskWithArenaHolder holder) 
{
    // raii
    CUDAScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

    // conditions
    edm::ESHandle<HcalRecoParamsGPU> recoParamsHandle;
    setup.get<HcalRecoParamsRcd>().get(recoParamsHandle);
    auto const& recoParamsProduct = recoParamsHandle->getProduct(ctx.stream());
}

void HBHERecHitProducerGPU::produce(
        edm::Event& event, 
        edm::EventSetup const& setup) 
{
    CUDAScopedContextProduce ctx{cudaState_};
}

DEFINE_FWK_MODULE(HBHERecHitProducerGPU);
