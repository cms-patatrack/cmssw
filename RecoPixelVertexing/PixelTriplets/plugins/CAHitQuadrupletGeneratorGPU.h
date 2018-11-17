#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h

#include <cuda_runtime.h>

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/PixelTrackingGPUConstants.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/RZLine.h"
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitSeeds.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/RecHitsMap.h"
#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircleFit.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitorFactory.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/RecHitsMap.h"

#include "CAHitQuadrupletGeneratorKernels.h"
#include "RiemannFitOnGPU.h"

#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

// FIXME  (split header???)
#include "GPUCACell.h"

class TrackingRegion;

namespace edm {
    class Event;
    class EventSetup;
    class ParameterSetDescription;
}

class CAHitQuadrupletGeneratorGPU {
public:

    using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
    using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;
    using hindex_type = siPixelRecHitsHeterogeneousProduct::hindex_type;

    using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;
    using TuplesOnCPU = pixelTuplesHeterogeneousProduct::TuplesOnCPU;
    using Quality = pixelTuplesHeterogeneousProduct::Quality;
    using Output = pixelTuplesHeterogeneousProduct::HeterogeneousPixelTuples;

    static constexpr unsigned int minLayers = 4;
    using  ResultType = OrderedHitSeeds;

public:

    CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC): CAHitQuadrupletGeneratorGPU(cfg, iC) {}
    CAHitQuadrupletGeneratorGPU(const edm::ParameterSet& cfg, edm::ConsumesCollector& iC);

    ~CAHitQuadrupletGeneratorGPU();

    static void fillDescriptions(edm::ParameterSetDescription& desc);
    static const char *fillDescriptionsLabel() { return "caHitQuadrupletGPU"; }

    void initEvent(const edm::Event& ev, const edm::EventSetup& es);

    void buildDoublets(HitsOnCPU const & hh, cudaStream_t stream);

    void hitNtuplets(const TrackingRegion &region, HitsOnCPU const & hh,
                     const edm::EventSetup& es,
                     bool doRiemannFit,
                     bool transferToCPU,
                     cudaStream_t stream);

    TuplesOnCPU getOutput() const {
       return TuplesOnCPU { std::move(indToEdm), hitsOnCPU->gpu_d, tuples_,  helix_fit_results_, quality_, gpu_d, nTuples_};
    }

    void cleanup(cudaStream_t stream);
    void fillResults(const TrackingRegion &region, SiPixelRecHitCollectionNew const & rechits,
                     std::vector<OrderedHitSeeds>& result,
                     const edm::EventSetup& es);

    void allocateOnGPU();
    void deallocateOnGPU();

private:

    // cpu stuff

    std::unique_ptr<SeedComparitor> theComparitor;

    class QuantityDependsPtEval {
    public:

        QuantityDependsPtEval(float v1, float v2, float c1, float c2) :
        value1_(v1), value2_(v2), curvature1_(c1), curvature2_(c2) {
        }

        float value(float curvature) const {
            if (value1_ == value2_) // not enabled
                return value1_;

            if (curvature1_ < curvature)
                return value1_;
            if (curvature2_ < curvature && curvature <= curvature1_)
                return value2_ + (curvature - curvature2_) / (curvature1_ - curvature2_) * (value1_ - value2_);
            return value2_;
        }

    private:
        const float value1_;
        const float value2_;
        const float curvature1_;
        const float curvature2_;
    };

    // Linear interpolation (in curvature) between value1 at pt1 and
    // value2 at pt2. If disabled, value2 is given (the point is to
    // allow larger/smaller values of the quantity at low pt, so it
    // makes more sense to have the high-pt value as the default).

    class QuantityDependsPt {
    public:

        explicit QuantityDependsPt(const edm::ParameterSet& pset) :
        value1_(pset.getParameter<double>("value1")),
        value2_(pset.getParameter<double>("value2")),
        pt1_(pset.getParameter<double>("pt1")),
        pt2_(pset.getParameter<double>("pt2")),
        enabled_(pset.getParameter<bool>("enabled")) {
            if (enabled_ && pt1_ >= pt2_)
                throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt1 (" << pt1_ << ") needs to be smaller than pt2 (" << pt2_ << ")";
            if (pt1_ <= 0)
                throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt1 needs to be > 0; is " << pt1_;
            if (pt2_ <= 0)
                throw cms::Exception("Configuration") << "PixelQuadrupletGenerator::QuantityDependsPt: pt2 needs to be > 0; is " << pt2_;
        }

        QuantityDependsPtEval evaluator(const edm::EventSetup& es) const {
            if (enabled_) {
                return QuantityDependsPtEval(value1_, value2_,
                        PixelRecoUtilities::curvature(1.f / pt1_, es),
                        PixelRecoUtilities::curvature(1.f / pt2_, es));
            }
            return QuantityDependsPtEval(value2_, value2_, 0.f, 0.f);
        }

    private:
        const float value1_;
        const float value2_;
        const float pt1_;
        const float pt2_;
        const bool enabled_;
    };

    // end cpu stuff


    void launchKernels(const TrackingRegion &, int, HitsOnCPU const & hh, bool doRiemannFit, bool transferToCPU, cudaStream_t);


    std::vector<std::array<int,4>> fetchKernelResult(int);


    CAHitQuadrupletGeneratorKernels kernels;
    RiemannFitOnGPU fitter;


    const float extraHitRPhitolerance;

    const QuantityDependsPt maxChi2;
    const bool fitFastCircle;
    const bool fitFastCircleChi2Cut;
    const bool useBendingCorrection;

    const float caThetaCut = 0.00125f;
    const float caPhiCut = 0.1f;
    const float caHardPtCut = 0.f;

    // products
    std::vector<uint32_t> indToEdm; // index of    tuple in reco tracks....
    TuplesOnGPU * gpu_d = nullptr;   // copy of the structure on the gpu itself: this is the "Product"
    TuplesOnGPU::Container * tuples_ = nullptr;
    Rfit::helix_fit * helix_fit_results_ = nullptr;
    Quality * quality_ =  nullptr;
    uint32_t nTuples_ = 0;
    TuplesOnGPU gpu_;

    // input
    HitsOnCPU const * hitsOnCPU=nullptr;

    RecHitsMap<TrackingRecHit const *> hitmap_ = RecHitsMap<TrackingRecHit const *>(nullptr);

};

#endif // RecoPixelVertexing_PixelTriplets_plugins_CAHitQuadrupletGeneratorGPU_h
