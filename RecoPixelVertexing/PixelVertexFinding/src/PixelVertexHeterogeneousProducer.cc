#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <memory>
#include <string>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDHeterogeneousProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"


class PixelVertexHeterogeneousProducer : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
  explicit PixelVertexHeterogeneousProducer(const edm::ParameterSet&);
  ~PixelVertexHeterogeneousProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void beginStreamGPUCuda(edm::StreamID streamId,
                          cuda::stream_t<> &cudaStream) override;
  void acquireGPUCuda(const edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceGPUCuda(edm::HeterogeneousEvent &iEvent,
                      const edm::EventSetup &iSetup,
                      cuda::stream_t<> &cudaStream) override;
  void produceCPU(edm::HeterogeneousEvent &iEvent,
                  const edm::EventSetup &iSetup) override;
 private:
  // ----------member data ---------------------------
  // Tracking cuts before sending tracks to vertex algo
  const double ptMin_;
  const edm::InputTag trackCollName;
  const edm::EDGetTokenT<reco::TrackCollection> token_Tracks;

};


PixelVertexHeterogeneousProducer::PixelVertexHeterogeneousProducer(const edm::ParameterSet& conf) 
  , ptMin_  (conf.getParameter<double>("PtMin")  ) // 1.0 GeV
  , trackCollName  ( conf.getParameter<edm::InputTag>("TrackCollection") )
  , token_Tracks   ( consumes<reco::TrackCollection>(trackCollName) )
{
  // Register my product
  produces<reco::VertexCollection>();

  float zOffset     = conf.getParameter<double>("ZOffset"); // 5.0 sigma
  float zSeparation = conf.getParameter<double>("ZSeparation"); // 0.05 cm
  int ntrkMin        = conf.getParameter<int>("NTrkMin"); // 3
  // Tracking requirements before sending a track to be considered for vtx
  

  float track_pt_min   = ptMin_;
  float track_pt_max   = 10.;
  float track_chi2_max = 9999999.;
  float track_prob_min = -1.;

  if ( conf.exists("PVcomparer") ) {
    edm::ParameterSet PVcomparerPSet = conf.getParameter<edm::ParameterSet>("PVcomparer");
    track_pt_min   = PVcomparerPSet.getParameter<double>("track_pt_min");    
    if (track_pt_min != ptMin_) {
      if (track_pt_min < ptMin_)
	edm::LogInfo("PixelVertexHeterogeneousProducer") << "minimum track pT setting differs between PixelVertexHeterogeneousProducer (" << ptMin_ << ") and PVcomparer (" << track_pt_min << ") [PVcomparer considers tracks w/ lower threshold than PixelVertexHeterogeneousProducer does] !!!";
      else
	edm::LogInfo("PixelVertexHeterogeneousProducer") << "minimum track pT setting differs between PixelVertexHeterogeneousProducer (" << ptMin_ << ") and PVcomparer (" << track_pt_min << ") !!!";
    }
    track_pt_max   = PVcomparerPSet.getParameter<double>("track_pt_max");
    track_chi2_max = PVcomparerPSet.getParameter<double>("track_chi2_max");
    track_prob_min = PVcomparerPSet.getParameter<double>("track_prob_min");
  }

}



void PixelVertexHeterogeneousProducer::acquireGPUCuda(
                      const edm::HeterogeneousEvent & e,
                      const edm::EventSetup & es,
                      cuda::stream_t<> &cudaStream) {

  // First fish the pixel tracks out of the event
  edm::Handle<reco::TrackCollection> trackCollection;
  e.getByToken(token_Tracks,trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());
  if (verbose_ > 0) edm::LogInfo("PixelVertexHeterogeneousProducer") << ": Found " << tracks.size() << " tracks in TrackCollection called " << trackCollName << "\n";
  

  // Second, make a collection of pointers to the tracks we want for the vertex finder
  reco::TrackRefVector trks;
  for (unsigned int i=0; i<tracks.size(); i++) {
    if (tracks[i].pt() > ptMin_)     
      trks.push_back( reco::TrackRef(trackCollection, i) );
  }
  if (verbose_ > 0) edm::LogInfo("PixelVertexHeterogeneousProducer") << ": Selected " << trks.size() << " of these tracks for vertexing\n";

  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByToken(token_BeamSpot,bsHandle);
  math::XYZPoint myPoint(0.,0.,0.);
  if (bsHandle.isValid()) myPoint = math::XYZPoint(bsHandle->x0(),bsHandle->y0(), 0. ); //FIXME: fix last coordinate with vertex.z() at same time

  // Third, ship these tracks off to be vertexed
  auto vertexes = std::make_unique<reco::VertexCollection>();
  bool ok;
  if (method2) {
    ok = dvf_->findVertexesAlt(trks,       // input
			     *vertexes,myPoint); // output
    if (verbose_ > 0) edm::LogInfo("PixelVertexHeterogeneousProducer") << "Method2 returned status of " << ok;
  }
  else {
    ok = dvf_->findVertexes(trks,       // input
			    *vertexes); // output
    if (verbose_ > 0) edm::LogInfo("PixelVertexHeterogeneousProducer") << "Method1 returned status of " << ok;
  }

  if (verbose_ > 0) {
    edm::LogInfo("PixelVertexHeterogeneousProducer") << ": Found " << vertexes->size() << " vertexes\n";
    for (unsigned int i=0; i<vertexes->size(); ++i) {
      edm::LogInfo("PixelVertexHeterogeneousProducer") << "Vertex number " << i << " has " << (*vertexes)[i].tracksSize() << " tracks with a position of " << (*vertexes)[i].z() << " +- " << std::sqrt( (*vertexes)[i].covariance(2,2) );
    }
  }


  if(bsHandle.isValid())
    {
      const reco::BeamSpot & bs = *bsHandle;
      
      for (unsigned int i=0; i<vertexes->size(); ++i) {
	double z=(*vertexes)[i].z();
	double x=bs.x0()+bs.dxdz()*(z-bs.z0());
	double y=bs.y0()+bs.dydz()*(z-bs.z0()); 
	reco::Vertex v( reco::Vertex::Point(x,y,z), (*vertexes)[i].error(),(*vertexes)[i].chi2() , (*vertexes)[i].ndof() , (*vertexes)[i].tracksSize());
	//Copy also the tracks 
	for (std::vector<reco::TrackBaseRef >::const_iterator it = (*vertexes)[i].tracks_begin();
	     it !=(*vertexes)[i].tracks_end(); it++) {
	  v.add( *it );
	}
	(*vertexes)[i]=v;
	
      }
    }
  else
    {
      edm::LogWarning("PixelVertexHeterogeneousProducer") << "No beamspot found. Using returning vertexes with (0,0,Z) ";
    } 
  
  if(vertexes->empty() && bsHandle.isValid()){
    
    const reco::BeamSpot & bs = *bsHandle;
      
      GlobalError bse(bs.rotatedCovariance3D());
      if ( (bse.cxx() <= 0.) ||
	   (bse.cyy() <= 0.) ||
	   (bse.czz() <= 0.) ) {
	AlgebraicSymMatrix33 we;
	we(0,0)=10000;
	we(1,1)=10000;
	we(2,2)=10000;
	vertexes->push_back(reco::Vertex(bs.position(), we,0.,0.,0));
	
	edm::LogInfo("PixelVertexHeterogeneousProducer") <<"No vertices found. Beamspot with invalid errors " << bse.matrix() << std::endl
					       << "Will put Vertex derived from dummy-fake BeamSpot into Event.\n"
					       << (*vertexes)[0].x() << "\n"
					       << (*vertexes)[0].y() << "\n"
					       << (*vertexes)[0].z() << "\n";
      } else {
	vertexes->push_back(reco::Vertex(bs.position(),
					 bs.rotatedCovariance3D(),0.,0.,0));
	
	edm::LogInfo("PixelVertexHeterogeneousProducer") << "No vertices found. Will put Vertex derived from BeamSpot into Event:\n"
					       << (*vertexes)[0].x() << "\n"
					       << (*vertexes)[0].y() << "\n"
					       << (*vertexes)[0].z() << "\n";
      }
  }
      
  else if(vertexes->empty() && !bsHandle.isValid())
    {
      edm::LogWarning("PixelVertexHeterogeneousProducer") << "No beamspot and no vertex found. No vertex returned.";
    }
  
  e.put(std::move(vertexes));
  
}


PixelVertexHeterogeneousProducer::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup)
{
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelVertexHeterogeneousProducer);
