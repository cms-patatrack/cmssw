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

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalLUTCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalTimeCorrsRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRecoParamsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalGainsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalLUTCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalWidthsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalPedestalsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalQIECodersGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalRespCorrsGPU.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalTimeCorrsGPU.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/DeclsForKernels.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEMahiGPU.h"

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

    edm::EDGetTokenT<HBHEDigiCollection> digisTokenQ8_;
    edm::EDGetTokenT<QIE11DigiCollection> digisTokenQ11_;

    hcal::mahi::ConfigParameters configParameters_;
    hcal::mahi::InputDataGPU inputDataGPU_;
    CUDAContextState cudaState_;
};

HBHERecHitProducerGPU::HBHERecHitProducerGPU(edm::ParameterSet const& ps) 
    : digisTokenQ8_{consumes<HBHEDigiCollection>(ps.getParameter<edm::InputTag>(
        "digisLabelQIE8"))}
    , digisTokenQ11_{consumes<QIE11DigiCollection>(ps.getParameter<edm::InputTag>(
        "digisLabelQIE11"))}
{
    configParameters_.maxChannels = ps.getParameter<uint32_t>("maxChannels");

    inputDataGPU_.allocate(configParameters_);
}

HBHERecHitProducerGPU::~HBHERecHitProducerGPU() {
    inputDataGPU_.deallocate(configParameters_);
}

void HBHERecHitProducerGPU::fillDescriptions(edm::ConfigurationDescriptions& cdesc) {
    edm::ParameterSetDescription desc;
    desc.add<uint32_t>("maxChannels", 10000u);
    desc.add<edm::InputTag>("digisLabelQIE8", edm::InputTag{"hcalDigis"});
    desc.add<edm::InputTag>("digisLabelQIE11", edm::InputTag{"hcalDigis"});

    std::string label = "hbheRecHitProducerGPU";
    cdesc.add(label, desc);
}

void HBHERecHitProducerGPU::acquire(
        edm::Event const& event,
        edm::EventSetup const& setup,
        edm::WaitingTaskWithArenaHolder holder) 
{
    // FIXME: remove debugging
    auto start = std::chrono::high_resolution_clock::now();

    // raii
    CUDAScopedContextAcquire ctx{event.streamID(), std::move(holder), cudaState_};

    // conditions
    edm::ESHandle<HcalRecoParamsGPU> recoParamsHandle;
    setup.get<HcalRecoParamsRcd>().get(recoParamsHandle);
    auto const& recoParamsProduct = recoParamsHandle->getProduct(ctx.stream());
    
    edm::ESHandle<HcalGainWidthsGPU> gainWidthsHandle;
    setup.get<HcalGainWidthsRcd>().get(gainWidthsHandle);
    auto const& gainWidthsProduct = gainWidthsHandle->getProduct(ctx.stream());
    
    edm::ESHandle<HcalGainsGPU> gainsHandle;
    setup.get<HcalGainsRcd>().get(gainsHandle);
    auto const& gainsProduct = gainsHandle->getProduct(ctx.stream());
    
    edm::ESHandle<HcalLUTCorrsGPU> lutCorrsHandle;
    setup.get<HcalLUTCorrsRcd>().get(lutCorrsHandle);
    auto const& lutCorrsProduct = lutCorrsHandle->getProduct(ctx.stream());
    
    edm::ESHandle<HcalPedestalWidthsGPU> pedestalWidthsHandle;
    setup.get<HcalPedestalWidthsRcd>().get(pedestalWidthsHandle);
    auto const& pedestalWidthsProduct = pedestalWidthsHandle->getProduct(ctx.stream());
    edm::ESHandle<HcalPedestalsGPU> pedestalsHandle;
    setup.get<HcalPedestalsRcd>().get(pedestalsHandle);
    auto const& pedestalsProduct = pedestalsHandle->getProduct(ctx.stream());
    
    edm::ESHandle<HcalQIECodersGPU> qieCodersHandle;
    setup.get<HcalQIEDataRcd>().get(qieCodersHandle);
    auto const& qieCodersProduct = qieCodersHandle->getProduct(ctx.stream());
    
    edm::ESHandle<HcalRespCorrsGPU> respCorrsHandle;
    setup.get<HcalRespCorrsRcd>().get(respCorrsHandle);
    auto const& respCorrsProduct = respCorrsHandle->getProduct(ctx.stream());
    
    edm::ESHandle<HcalTimeCorrsGPU> timeCorrsHandle;
    setup.get<HcalTimeCorrsRcd>().get(timeCorrsHandle);
    auto const& timeCorrsProduct = timeCorrsHandle->getProduct(ctx.stream());

    // bundle up conditions
    hcal::mahi::ConditionsProducts conditions{
        gainWidthsProduct, gainsProduct, lutCorrsProduct,
        pedestalWidthsProduct, pedestalsProduct,
        qieCodersProduct, recoParamsProduct,
        respCorrsProduct, timeCorrsProduct
    };

    // event data
    edm::Handle<HBHEDigiCollection> digisQ8;
    edm::Handle<QIE11DigiCollection> digisQ11;
    event.getByToken(digisTokenQ8_, digisQ8);
    event.getByToken(digisTokenQ11_, digisQ11);

    hcal::mahi::InputDataCPU inputCPU{*digisQ8, *digisQ11};
    hcal::mahi::entryPoint(inputCPU, inputDataGPU_, conditions,
        configParameters_, ctx.stream());

    // FIXME: remove debugging
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "acquire  duration = " << duration << std::endl;
}

void HBHERecHitProducerGPU::produce(
        edm::Event& event, 
        edm::EventSetup const& setup) 
{
    CUDAScopedContextProduce ctx{cudaState_};
}

DEFINE_FWK_MODULE(HBHERecHitProducerGPU);
