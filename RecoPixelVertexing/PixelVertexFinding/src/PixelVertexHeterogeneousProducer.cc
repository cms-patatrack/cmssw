#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <memory>
#include <string>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "HeterogeneousCore/CUDACore/interface/GPUCuda.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/Producer/interface/HeterogeneousEDProducer.h"

#include "RecoPixelVertexing/PixelVertexFinding/interface/pixelVertexHeterogeneousProduct.h"
#include "gpuVertexFinder.h"


class PixelVertexHeterogeneousProducer : public HeterogeneousEDProducer<heterogeneous::HeterogeneousDevices<
          heterogeneous::GPUCuda, heterogeneous::CPU>> {
  explicit PixelVertexHeterogeneousProducer(const edm::ParameterSet&);
  ~PixelVertexHeterogeneousProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void beginStreamGPUCuda(edm::StreamID streamId,
                          cuda::stream_t<> &cudaStream) override {
    m_gpuAlgo.allocateOnGPU();
  }
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
  const double m_ptMin;
  const edm::InputTag trackCollName;
  const edm::EDGetTokenT<reco::TrackCollection> token_Tracks;
  const edm::EDGetTokenT<reco::BeamSpot> token_BeamSpot;


  reco::TrackRefVector m_trks;

  gpuVertexFinder::Producer m_gpuAlgo;

  bool verbose_ = true;
  
};


void PixelVertexHeterogeneousProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("PtMin", 0.5);
  desc.add<edm::InputTag>("TrackCollection", edm::InputTag("pixelTracks"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  auto label = "pixelVertexHeterogeneousProducer";
  descriptions.add(label, desc);
}


PixelVertexHeterogeneousProducer::PixelVertexHeterogeneousProducer(const edm::ParameterSet& conf) :
  m_ptMin  (conf.getParameter<double>("PtMin")  ) // 0.5 GeV
  , trackCollName  ( conf.getParameter<edm::InputTag>("TrackCollection") )
  , token_Tracks   ( consumes<reco::TrackCollection>(trackCollName) )
  , token_BeamSpot ( consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpot") ) )
{
  // Register my product
  produces<reco::VertexCollection>();  
  
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

  // on gpu beamspot already subtracted at hit level...
  math::XYZPoint bs(0.,0.,0.);
  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByToken(token_BeamSpot,bsHandle);

  if (bsHandle.isValid()) bs = math::XYZPoint(bsHandle->x0(),bsHandle->y0(), bsHandle->z0() ); 

  // Second, make a collection of pointers to the tracks we want for the vertex finder
  // fill z,ez
  std::vector<float> z,ez;
  assert(m_trks.empty());
  for (unsigned int i=0; i<tracks.size(); i++) {
    if (tracks[i].pt() < m_ptMin) continue;    
    m_trks.push_back( reco::TrackRef(trackCollection, i) );
    z.push_back(tracks[i].dz(bs));
    ez.push_back(tracks[i].dzError());ez.back()*=ez.back();
  }
  if (verbose_ > 0) edm::LogInfo("PixelVertexHeterogeneousProducer") << ": Selected " << m_trks.size() << " of these tracks for vertexing\n";
  
  // Third, ship these tracks off to be vertexed
  m_gpuAlgo.produce(cudaStream.id(),z.data(),ez.data(),z.size());

}

void PixelVertexHeterogeneousProducer::produceGPUCuda(
    edm::HeterogeneousEvent & e, const edm::EventSetup & es,
    cuda::stream_t<> &cudaStream) {


  auto const & gpuProduct = m_gpuAlgo.fillResults(cudaStream.id());

  
  edm::Handle<reco::BeamSpot> bsHandle;
  e.getByToken(token_BeamSpot,bsHandle);
 
  auto vertexes = std::make_unique<reco::VertexCollection>(); 


  float x0=0,y0=0,z0=0,dxdz=0,dydz=0;
  if(!bsHandle.isValid()) {
     edm::LogWarning("PixelVertexHeterogeneousProducer") << "No beamspot found. Using returning vertexes with (0,0,Z) ";
  } else {
    const reco::BeamSpot & bs = *bsHandle;
    x0=bs.x0();y0=bs.y0();z0=bs.z0(); dxdz=bs.dxdz();dydz=bs.dydz();
  }
  for (unsigned int i=0; i<gpuProduct.nVertices; ++i) {
    auto z= gpuProduct.z[i];
    auto x= x0 + dxdz*z;
    auto y= y0 + dydz*z;
    z +=z0;
    reco::Vertex::Error err;
    err(2,2) = 1.f/gpuProduct.zerr[i];
    auto nt =0;
    (*vertexes).emplace_back(reco::Vertex::Point(x,y,z), err, 0, nt-1, nt );
    //Copy also the tracks (no intention to be more efficient....)
    //  v.add(m_trks[k]);
  }

  if (verbose_ > 0) {
    edm::LogInfo("PixelVertexHeterogeneousProducer") << ": Found " << vertexes->size() << " vertexes\n";
    for (unsigned int i=0; i<vertexes->size(); ++i) {
      edm::LogInfo("PixelVertexHeterogeneousProducer") << "Vertex number " << i << " has " << (*vertexes)[i].tracksSize() << " tracks with a position of " << (*vertexes)[i].z() << " +- " << std::sqrt( (*vertexes)[i].covariance(2,2) );
    }
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
      
      edm::LogInfo("PixelVertexHeterogeneousProducer") << "No vertices found. Beamspot with invalid errors " << bse.matrix()
						       << "\nWill put Vertex derived from dummy-fake BeamSpot into Event.\n"
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
  m_trks.clear();
}


PixelVertexHeterogeneousProducer::produceCPU(
    edm::HeterogeneousEvent &iEvent, const edm::EventSetup &iSetup)
{
  throw cms::Exception("NotImplemented") << "CPU version is no longer implemented";
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PixelVertexHeterogeneousProducer);
