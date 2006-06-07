#include <iostream>
#include <vector>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibMuon/CSCCalibration/interface/CSCAFEBThrAnalysis.h"

class CSCAFEBAnalyzer : public edm::EDAnalyzer {
public:
  explicit CSCAFEBAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  virtual void endJob();
private:
  /// variables persistent across events should be declared here.

  CSCAFEBThrAnalysis analysisthr_;
};

