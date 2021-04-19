// system include files
#include <cmath>
#include <memory>
#include <vector>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

class ConvertToCartesianVectors : public edm::stream::EDProducer<> {
public:
  explicit ConvertToCartesianVectors(const edm::ParameterSet&);
  ~ConvertToCartesianVectors() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using CartesianVectors = std::vector<math::XYZVectorF>;
  using CylindricalVectors = std::vector<math::RhoEtaPhiVectorF>;

  static void convert(math::RhoEtaPhiVectorF const& cilindrical, math::XYZVectorF & cartesian);

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<CylindricalVectors> input_;
  edm::EDPutTokenT<CartesianVectors> output_;
};

ConvertToCartesianVectors::ConvertToCartesianVectors(const edm::ParameterSet& config)
    : input_(consumes<CylindricalVectors>(config.getParameter<edm::InputTag>("input"))) {
  output_ = produces<CartesianVectors>();
}

void ConvertToCartesianVectors::convert(math::RhoEtaPhiVectorF const& cilindrical, math::XYZVectorF & cartesian) {
    cartesian.SetCoordinates(cilindrical.rho() * std::cos(cilindrical.phi()),
                             cilindrical.rho() * std::sin(cilindrical.phi()),
                             cilindrical.rho() * std::sinh(cilindrical.eta()));
}

void ConvertToCartesianVectors::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& input = event.get(input_);
  auto elements = input.size();
  auto product = std::make_unique<CartesianVectors>(elements);

  // convert the vectors from cylindrical to cartesian coordinates
  for (unsigned int i = 0; i < elements; ++i) {
    convert(input[i], (*product)[i]);
  }

  event.put(output_, std::move(product));
}

void ConvertToCartesianVectors::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(ConvertToCartesianVectors);
