// system include files
#include <memory>
#include <vector>

// CMSSW include files
#include "DataFormats/Math/interface/Vector3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class PrintCartesianVectors : public edm::one::EDAnalyzer<> {
public:
  explicit PrintCartesianVectors(const edm::ParameterSet&);
  ~PrintCartesianVectors() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using CartesianVectors = std::vector<math::XYZVectorF>;

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<CartesianVectors> input_;
};

PrintCartesianVectors::PrintCartesianVectors(const edm::ParameterSet& config)
    : input_(consumes<CartesianVectors>(config.getParameter<edm::InputTag>("input"))) {}

void PrintCartesianVectors::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  for (auto const& v : event.get(input_)) {
    std::cout << std::fixed << "x: " << std::setw(6) << std::setprecision(2) << v.x() << ", y: " << std::setw(6)
              << std::setprecision(2) << v.y() << ", z: " << std::setw(8) << std::setprecision(2) << v.z() << std::endl;
  }
  std::cout << std::endl;
}

void PrintCartesianVectors::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cartesianVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(PrintCartesianVectors);
