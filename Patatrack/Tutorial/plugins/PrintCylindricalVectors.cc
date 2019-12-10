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

class PrintCylindricalVectors : public edm::one::EDAnalyzer<> {
public:
  explicit PrintCylindricalVectors(const edm::ParameterSet&);
  ~PrintCylindricalVectors() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  using CylindricalVectors = std::vector<math::RhoEtaPhiVectorF>;

  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<CylindricalVectors> input_;
};

PrintCylindricalVectors::PrintCylindricalVectors(const edm::ParameterSet& config)
    : input_(consumes<CylindricalVectors>(config.getParameter<edm::InputTag>("input"))) {}

void PrintCylindricalVectors::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  for (auto const& v : event.get(input_)) {
    std::cout << std::fixed << "pT: " << std::setw(6) << std::setprecision(2) << v.rho() << ", eta: " << std::setw(6)
              << std::setprecision(2) << v.eta() << ", phi: " << std::setw(6) << std::setprecision(2) << v.phi()
              << std::endl;
  }
  std::cout << std::endl;
}

void PrintCylindricalVectors::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("input", edm::InputTag("cylindricalVectors"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(PrintCylindricalVectors);
