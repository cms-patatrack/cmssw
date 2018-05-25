#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include "siPixelRawToDigiHeterogeneousProduct.h"

class SiPixelClusterHeterogeneousConverter: public edm::global::EDProducer<> {
public:
  explicit SiPixelClusterHeterogeneousConverter(edm::ParameterSet const& iConfig);
  ~SiPixelClusterHeterogeneousConverter() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<HeterogeneousProduct> token_;
};

SiPixelClusterHeterogeneousConverter::SiPixelClusterHeterogeneousConverter(edm::ParameterSet const& iConfig):
  token_(consumes<HeterogeneousProduct>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<edm::DetSetVector<PixelDigi> >();
  produces<SiPixelClusterCollectionNew>(); 
}

void SiPixelClusterHeterogeneousConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPixelDigisHeterogeneous"));

  descriptions.addWithDefaultLabel(desc);
}

namespace {
  template <typename T>
  auto copy_unique(const edmNew::DetSetVector<T>& t) {
    auto ret = std::make_unique<edmNew::DetSetVector<T> >();
    ret->replaceFrom(t);
    return ret;
  }
}

void SiPixelClusterHeterogeneousConverter::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HeterogeneousProduct> hinput;
  iEvent.getByToken(token_, hinput);

  const auto& input = hinput->get<siPixelRawToDigiHeterogeneousProduct::HeterogeneousDigiCluster>().getProduct<HeterogeneousDevice::kCPU>();

  iEvent.put(copy_unique(input.outputClusters));
}


DEFINE_FWK_MODULE(SiPixelClusterHeterogeneousConverter);
