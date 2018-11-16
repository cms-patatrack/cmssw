//
// Author: Felice Pantaleo, CERN
//

#include <functional>

#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoPixelVertexing/PixelTriplets/interface/ThirdHitPredictionFromCircle.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "CAHitQuadrupletGeneratorGPU.h"

namespace {

  template <typename T> T sqr(T x) { return x * x; }

} // namespace

using namespace std;

constexpr unsigned int CAHitQuadrupletGeneratorGPU::minLayers;

CAHitQuadrupletGeneratorGPU::CAHitQuadrupletGeneratorGPU(
    const edm::ParameterSet &cfg,
    edm::ConsumesCollector &iC)
  : extraHitRPhitolerance(cfg.getParameter<double>("extraHitRPhitolerance")), // extra window in ThirdHitPredictionFromCircle range (divide by R to get phi)
    maxChi2(cfg.getParameter<edm::ParameterSet>("maxChi2")),
    fitFastCircle(cfg.getParameter<bool>("fitFastCircle")),
    fitFastCircleChi2Cut(cfg.getParameter<bool>("fitFastCircleChi2Cut")),
    useBendingCorrection(cfg.getParameter<bool>("useBendingCorrection")),
    caThetaCut(cfg.getParameter<double>("CAThetaCut")),
    caPhiCut(cfg.getParameter<double>("CAPhiCut")),
    caHardPtCut(cfg.getParameter<double>("CAHardPtCut"))
{
  edm::ParameterSet comparitorPSet = cfg.getParameter<edm::ParameterSet>("SeedComparitorPSet");
  std::string comparitorName = comparitorPSet.getParameter<std::string>("ComponentName");
  if (comparitorName != "none") {
    theComparitor.reset(SeedComparitorFactory::get()->create(comparitorName, comparitorPSet, iC));
  }
}

void CAHitQuadrupletGeneratorGPU::fillDescriptions(edm::ParameterSetDescription &desc) {
  desc.add<double>("extraHitRPhitolerance", 0.1);
  desc.add<bool>("fitFastCircle", false);
  desc.add<bool>("fitFastCircleChi2Cut", false);
  desc.add<bool>("useBendingCorrection", false);
  desc.add<double>("CAThetaCut", 0.00125);
  desc.add<double>("CAPhiCut", 10);
  desc.add<double>("CAHardPtCut", 0);
  desc.addOptional<bool>("CAOnlyOneLastHitPerLayerFilter")->setComment(
      "Deprecated and has no effect. To be fully removed later when the "
      "parameter is no longer used in HLT configurations.");
  edm::ParameterSetDescription descMaxChi2;
  descMaxChi2.add<double>("pt1", 0.2);
  descMaxChi2.add<double>("pt2", 1.5);
  descMaxChi2.add<double>("value1", 500);
  descMaxChi2.add<double>("value2", 50);
  descMaxChi2.add<bool>("enabled", true);
  desc.add<edm::ParameterSetDescription>("maxChi2", descMaxChi2);

  edm::ParameterSetDescription descComparitor;
  descComparitor.add<std::string>("ComponentName", "none");
  descComparitor.setAllowAnything();    // until we have moved SeedComparitor to EDProducers too
  desc.add<edm::ParameterSetDescription>("SeedComparitorPSet", descComparitor);
}

void CAHitQuadrupletGeneratorGPU::initEvent(edm::Event const& ev, edm::EventSetup const& es) {
  if (theComparitor)
    theComparitor->init(ev, es);
  fitter.setBField(1 / PixelRecoUtilities::fieldInInvGev(es));
}


CAHitQuadrupletGeneratorGPU::~CAHitQuadrupletGeneratorGPU() {
    deallocateOnGPU();
}

void CAHitQuadrupletGeneratorGPU::hitNtuplets(
    TrackingRegion const& region,
    HitsOnCPU const& hh,
    edm::EventSetup const& es,
    bool doRiemannFit,
    bool transferToCPU,
    cudaStream_t cudaStream)
{
  hitsOnCPU = &hh;
  int index = 0;
  launchKernels(region, index, hh, doRiemannFit, transferToCPU, cudaStream);
}

void CAHitQuadrupletGeneratorGPU::fillResults(
    const TrackingRegion &region, SiPixelRecHitCollectionNew const & rechits,
    std::vector<OrderedHitSeeds> &result, const edm::EventSetup &es)
{
  hitmap_.clear();
  auto const & rcs = rechits.data();
  for (auto const & h : rcs) hitmap_.add(h, &h);

  assert(hitsOnCPU);
  auto nhits = hitsOnCPU->nHits;
  int index = 0;

  auto const & foundQuads = fetchKernelResult(index);
  unsigned int numberOfFoundQuadruplets = foundQuads.size();

  /*
  const QuantityDependsPtEval maxChi2Eval = maxChi2.evaluator(es);

  // re-used throughout
  std::array<float, 4> bc_r;
  std::array<float, 4> bc_z;
  std::array<float, 4> bc_errZ2;
  */
  
  std::array<GlobalPoint, 4> gps;
  std::array<GlobalError, 4> ges;
  std::array<bool, 4> barrels;
  std::array<BaseTrackerRecHit const*, 4> phits;


  int nbad=0;
  // loop over quadruplets
  for (unsigned int quadId = 0; quadId < numberOfFoundQuadruplets; ++quadId) {
    auto isBarrel = [](const unsigned id) -> bool {
      return id == PixelSubdetector::PixelBarrel;
    };
    bool bad = pixelTuplesHeterogeneousProduct::bad == quality_[quadId];
    for (unsigned int i = 0; i < 4; ++i) {
      auto k = foundQuads[quadId][i];
      assert(k<int(nhits));
      auto hp = hitmap_.get((*hitsOnCPU).detInd[k],(*hitsOnCPU).mr[k], (*hitsOnCPU).mc[k]);
      if (hp==nullptr) {
        bad=true;
        break;
      }
      phits[i] = static_cast<BaseTrackerRecHit const *>(hp);
      auto const &ahit = *phits[i];
      gps[i] = ahit.globalPosition();
      ges[i] = ahit.globalPositionError();
      barrels[i] = isBarrel(ahit.geographicalId().subdetId());

    }
    if (bad) { nbad++; quality_[quadId] = pixelTuplesHeterogeneousProduct::bad; continue;}
    if (quality_[quadId] != pixelTuplesHeterogeneousProduct::loose) continue; // FIXME remove dup
    
    /*
    // this part shall not be run anymore...
    quality_[quadId] = pixelTuplesHeterogeneousProduct::bad;

    // TODO:
    // - if we decide to always do the circle fit for 4 hits, we don't
    //   need ThirdHitPredictionFromCircle for the curvature; then we
    //   could remove extraHitRPhitolerance configuration parameter
    ThirdHitPredictionFromCircle predictionRPhi(gps[0], gps[2],
        extraHitRPhitolerance);
    const float curvature = predictionRPhi.curvature(
        ThirdHitPredictionFromCircle::Vector2D(gps[1].x(), gps[1].y()));
    const float abscurv = std::abs(curvature);
    const float thisMaxChi2 = maxChi2Eval.value(abscurv);
    if (theComparitor) {
      SeedingHitSet tmpTriplet(phits[0],  phits[1],  phits[3]);
      if (!theComparitor->compatible(tmpTriplet)) {
        continue;
      }
    }

    float chi2 = std::numeric_limits<float>::quiet_NaN();
    // TODO: Do we have any use case to not use bending correction?
    if (useBendingCorrection) {
      // Following PixelFitterByConformalMappingAndLine
      const float simpleCot = (gps.back().z() - gps.front().z()) /
        (gps.back().perp() - gps.front().perp());
      const float pt = 1.f / PixelRecoUtilities::inversePt(abscurv, es);
      for (int i = 0; i < 4; ++i) {
        const GlobalPoint &point = gps[i];
        const GlobalError &error = ges[i];
        bc_r[i] = sqrt(sqr(point.x() - region.origin().x()) +
            sqr(point.y() - region.origin().y()));
        bc_r[i] += pixelrecoutilities::LongitudinalBendingCorrection(pt, es)(
            bc_r[i]);
        bc_z[i] = point.z() - region.origin().z();
        bc_errZ2[i] =
          (barrels[i]) ? error.czz() : error.rerr(point) * sqr(simpleCot);
      }
      RZLine rzLine(bc_r, bc_z, bc_errZ2, RZLine::ErrZ2_tag());
      chi2 = rzLine.chi2();
    } else {
      RZLine rzLine(gps, ges, barrels);
      chi2 = rzLine.chi2();
    }
    if (edm::isNotFinite(chi2) || chi2 > thisMaxChi2) {
      continue;
    }
    // TODO: Do we have any use case to not use circle fit? Maybe
    // HLT where low-pT inefficiency is not a problem?
    if (fitFastCircle) {
      FastCircleFit c(gps, ges);
      chi2 += c.chi2();
      if (edm::isNotFinite(chi2))
        continue;
      if (fitFastCircleChi2Cut && chi2 > thisMaxChi2)
        continue;
    }

    */    

    result[index].emplace_back(phits[0],  phits[1],  phits[2],  phits[3]);
    // quality_[quadId] = pixelTuplesHeterogeneousProduct::loose;
  } // end loop over quads

// #ifdef GPU_DEBUG
  std::cout << "Q Final quads " << result[index].size() << ' ' << nbad << std::endl; 
// #endif

}


void CAHitQuadrupletGeneratorGPU::deallocateOnGPU()
{

  fitter.deallocateOnGPU();
  kernels.deallocateOnGPU();

  //product
  cudaFree(gpu_.tuples_d);
  cudaFree(gpu_.helix_fit_results_d);
  cudaFree(gpu_.apc_d);
  cudaFree(gpu_d);
  cudaFree(tuples_);
  cudaFree(helix_fit_results_);
}

void CAHitQuadrupletGeneratorGPU::allocateOnGPU()
{

  constexpr auto maxNumberOfQuadruplets_ = CAConstants::maxNumberOfQuadruplets();

  //product
  cudaCheck(cudaMalloc(&gpu_.tuples_d, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMalloc(&gpu_.apc_d, sizeof(AtomicPairCounter)));
  cudaCheck(cudaMalloc(&gpu_.helix_fit_results_d, sizeof(Rfit::helix_fit)*maxNumberOfQuadruplets_));
  cudaCheck(cudaMalloc(&gpu_.quality_d, sizeof(Quality)*maxNumberOfQuadruplets_));

  cudaCheck(cudaMalloc(&gpu_d, sizeof(TuplesOnGPU)));
  gpu_.me_d = gpu_d;
  cudaCheck(cudaMemcpy(gpu_d, &gpu_, sizeof(TuplesOnGPU), cudaMemcpyDefault));

  cudaCheck(cudaMallocHost(&tuples_, sizeof(TuplesOnGPU::Container)));
  cudaCheck(cudaMallocHost(&helix_fit_results_, sizeof(Rfit::helix_fit)*maxNumberOfQuadruplets_));
  cudaCheck(cudaMallocHost(&quality_, sizeof(Quality)*maxNumberOfQuadruplets_));

  kernels.allocateOnGPU();
  fitter.allocateOnGPU(gpu_.tuples_d, gpu_.helix_fit_results_d);


}

void CAHitQuadrupletGeneratorGPU::launchKernels(const TrackingRegion &region,
                                                int regionIndex, HitsOnCPU const & hh,
                                                bool doRiemannFit,
                                                bool transferToCPU,
                                                cudaStream_t cudaStream)
{
  assert(0==regionIndex);


  kernels.launchKernels(hh, gpu_, cudaStream); 
  if (doRiemannFit) 
    fitter.launchKernels(hh, hh.nHits, CAConstants::maxNumberOfQuadruplets(), cudaStream);

  if (transferToCPU) {
    cudaCheck(cudaMemcpyAsync(tuples_,gpu_.tuples_d,
                              sizeof(TuplesOnGPU::Container),
                              cudaMemcpyDeviceToHost, cudaStream));

    cudaCheck(cudaMemcpyAsync(helix_fit_results_,gpu_.helix_fit_results_d, 
                              sizeof(Rfit::helix_fit)*CAConstants::maxNumberOfQuadruplets(),
                              cudaMemcpyDeviceToHost, cudaStream));

    cudaCheck(cudaMemcpyAsync(quality_,gpu_.quality_d,
                              sizeof(Quality)*CAConstants::maxNumberOfQuadruplets(),
                              cudaMemcpyDeviceToHost, cudaStream));

  }

}

void CAHitQuadrupletGeneratorGPU::cleanup(cudaStream_t cudaStream) {
  kernels.cleanup(cudaStream);
}



std::vector<std::array<int, 4>>
CAHitQuadrupletGeneratorGPU::fetchKernelResult(int)
{
  assert(tuples_);
  auto const & tuples = *tuples_;

  uint32_t sizes[7]={0};
  std::vector<int> ntk(10000);
  auto add = [&](uint32_t hi) { if (hi>=ntk.size()) ntk.resize(hi+1); ++ntk[hi];};

  std::vector<std::array<int, 4>> quadsInterface; quadsInterface.reserve(10000);

  nTuples_=0;
  for (auto i = 0U; i < tuples.nbins(); ++i) {
    auto sz = tuples.size(i);
    if (sz==0) break;  // we know cannot be less then 3
    ++nTuples_;
    ++sizes[sz];
    for (auto j=tuples.begin(i); j!=tuples.end(i); ++j) add(*j);
    if (sz<4) continue;
    quadsInterface.emplace_back(std::array<int, 4>());
    quadsInterface.back()[0] = tuples.begin(i)[0];
    quadsInterface.back()[1] = tuples.begin(i)[1];
    quadsInterface.back()[2] = tuples.begin(i)[2];   // [sz-2];
    quadsInterface.back()[3] = tuples.begin(i)[3];   // [sz-1];
  }

//#ifdef GPU_DEBUG
  long long ave =0; int nn=0; for (auto k : ntk) if(k>0){ave+=k; ++nn;}
  std::cout << "Q Produced " << quadsInterface.size() << " quadruplets: ";
  for (auto i=3; i<7; ++i) std::cout << sizes[i] << ' ';
  std::cout << "max/ave " << *std::max_element(ntk.begin(),ntk.end())<<'/'<<float(ave)/float(nn) << std::endl;
//#endif
  return quadsInterface;
}

void CAHitQuadrupletGeneratorGPU::buildDoublets(HitsOnCPU const & hh, cudaStream_t stream) {
   kernels.buildDoublets(hh,stream);
}
