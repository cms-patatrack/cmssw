#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"


// make    code rules happy
namespace {

  enum Id { legacy, fromSoA };


  struct    FromSoABase { 
     static constexpr Id s_myId = fromSoA; 
     virtual ~FromSoABase() = default;
  };
  struct   LegacyBase {
     static constexpr Id s_myId = legacy;
    virtual ~LegacyBase() = default;
    virtual  void acquire(edm::Event const& iEvent,
             edm::EventSetup const& iSetup,
             edm::WaitingTaskWithArenaHolder waitingTaskHolder) =0;

  };

}


template<typename Producer, typename Base>
class SiPixelRecHitProducer : public Producer, private Base {
public:

  explicit SiPixelRecHitProducer(const edm::ParameterSet& iConfig);
  ~SiPixelRecHitProducer() noexcept override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

};

template<typename Producer, typename Base>
SiPixelRecHitProducer<Producer,Base>::SiPixelRecHitProducer(const edm::ParameterSet& iConfig) {

}


template<typename Producer, typename Base>
void SiPixelRecHitProducer<Producer,Base>::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {


}


template<typename Producer, typename Base>
void SiPixelRecHitProducer<Producer,Base>::acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) {

}

template<typename Producer, typename Base>
void SiPixelRecHitProducer<Producer,Base>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsCUDAPreSplitting"));
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<bool>("clusterLess", false);
  if constexpr (Base::s_myId == fromSoA)
    descriptions.add("notyetsiPixelRecHitFromSOA", desc);
  else
    descriptions.add("notyesiPixelRecHitConverter", desc);

}





using NotyetSiPixelRecHitFromSOA = SiPixelRecHitProducer<edm::stream::EDProducer<edm::ExternalWork>,FromSoABase>;
using NotyeSiPixelRecHitConverter = SiPixelRecHitProducer<edm::stream::EDProducer<>,LegacyBase>;
DEFINE_FWK_MODULE(NotyetSiPixelRecHitFromSOA);
DEFINE_FWK_MODULE(NotyeSiPixelRecHitConverter);

