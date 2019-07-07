#ifndef RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
#define RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h

#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"

#include "CUDADataFormats/Track/interface/PixelTrackCUDA.h"
#include "GPUCACell.h"

// #define DUMP_GPU_TK_TUPLES


class CAHitNtupletGeneratorKernels {
public:
  // counters
  struct Counters {
    unsigned long long nEvents;
    unsigned long long nHits;
    unsigned long long nCells;
    unsigned long long nTuples;
    unsigned long long nFitTracks;
    unsigned long long nGoodTracks;
    unsigned long long nUsedHits;
    unsigned long long nDupHits;
    unsigned long long nKilledCells;
    unsigned long long nEmptyCells;
    unsigned long long nZeroTrackCells;
  };

  using HitsOnGPU = TrackingRecHit2DSOAView;
  using HitsOnCPU = TrackingRecHit2DCUDA;

  using HitToTuple = CAConstants::HitToTuple;
  using TupleMultiplicity = CAConstants::TupleMultiplicity;

  using Quality = PixelTrackCUDA::Quality;
  using TkSoA = PixelTrackCUDA::SoA;
  using HitContainer = PixelTrackCUDA::HitContainer;


  struct QualityCuts {
    // chi2 cut = chi2Scale * (chi2Coeff[0] + pT/GeV * (chi2Coeff[1] + pT/GeV * (chi2Coeff[2] + pT/GeV * chi2Coeff[3])))
    float chi2Coeff[4];
    float chi2MaxPt;    // GeV
    float chi2Scale;

    struct region {
      float maxTip;     // cm
      float minPt;      // GeV
      float maxZip;     // cm
    };

    region triplet;
    region quadruplet;
  };


  // params
  struct Params {
    Params(uint32_t minHitsPerNtuplet,
                                  bool useRiemannFit,
                                  bool fit5as4,
                                  bool includeJumpingForwardDoublets,
                                  bool earlyFishbone,
                                  bool lateFishbone,
                                  bool idealConditions,
                                  bool doStats,
                                  bool doClusterCut,
                                  bool doZCut,
                                  bool doPhiCut,
                                  bool doIterations,
                                  float ptmin,
                                  float CAThetaCutBarrel,
                                  float CAThetaCutForward,
                                  float hardCurvCut,
                                  float dcaCutInnerTriplet,
                                  float dcaCutOuterTriplet,
                                  QualityCuts const& cuts)
      : minHitsPerNtuplet_(minHitsPerNtuplet),
        useRiemannFit_(useRiemannFit),
        fit5as4_(fit5as4),
        includeJumpingForwardDoublets_(includeJumpingForwardDoublets),
        earlyFishbone_(earlyFishbone),
        lateFishbone_(lateFishbone),
        idealConditions_(idealConditions),
        doStats_(doStats),
        doClusterCut_(doClusterCut),
        doZCut_(doZCut),
        doPhiCut_(doPhiCut),
        doIterations_(doIterations),
        ptmin_(ptmin),
        CAThetaCutBarrel_(CAThetaCutBarrel),
        CAThetaCutForward_(CAThetaCutForward),
        hardCurvCut_(hardCurvCut),
        dcaCutInnerTriplet_(dcaCutInnerTriplet),
        dcaCutOuterTriplet_(dcaCutOuterTriplet),
        cuts_(cuts) { }

  const uint32_t minHitsPerNtuplet_;
  const bool useRiemannFit_;
  const bool fit5as4_;
  const bool includeJumpingForwardDoublets_;
  const bool earlyFishbone_;
  const bool lateFishbone_;
  const bool idealConditions_;
  const bool doStats_;
  const bool doClusterCut_;
  const bool doZCut_;
  const bool doPhiCut_;
  const bool doIterations_;
  const float ptmin_;
  const float CAThetaCutBarrel_;
  const float CAThetaCutForward_;
  const float hardCurvCut_;
  const float dcaCutInnerTriplet_;
  const float dcaCutOuterTriplet_;

  // quality cuts
  QualityCuts cuts_
  {
    // polynomial coefficients for the pT-dependent chi2 cut
    { 0.68177776, 0.74609577, -0.08035491, 0.00315399 },
    // max pT used to determine the chi2 cut
    10.,
    // chi2 scale factor: 30 for broken line fit, 45 for Riemann fit
    30.,
    // regional cuts for triplets
    {
      0.3,  // |Tip| < 0.3 cm
      0.5,  // pT > 0.5 GeV
      12.0  // |Zip| < 12.0 cm
    },
    // regional cuts for quadruplets
    {
      0.5,  // |Tip| < 0.5 cm
      0.3,  // pT > 0.3 GeV
      12.0  // |Zip| < 12.0 cm
    }
   };
  
  }; // Params


  CAHitNtupletGeneratorKernels(Params const & params) : m_params(params){}
  ~CAHitNtupletGeneratorKernels() = default;

  TupleMultiplicity const* tupleMultiplicity() const { return device_tupleMultiplicity_.get(); }

  void launchKernels(HitsOnCPU const& hh, TkSoA * tuples_d, cudaStream_t cudaStream);

  void classifyTuples(HitsOnCPU const& hh, TkSoA * tuples_d, cudaStream_t cudaStream);

  void fillHitDetIndices(HitsOnCPU const &hh, TkSoA * tuples_d, cuda::stream_t<>& stream);

  void buildDoublets(HitsOnCPU const& hh, cuda::stream_t<>& stream);
  void allocateOnGPU(cuda::stream_t<>& stream);
  void cleanup(cudaStream_t cudaStream);

  static void printCounters(Counters const * counters);
  Counters* counters_ = nullptr;


private:

  // workspace
  CAConstants::CellNeighborsVector* device_theCellNeighbors_ = nullptr;
  cudautils::device::unique_ptr<CAConstants::CellNeighbors[]> device_theCellNeighborsContainer_;
  CAConstants::CellTracksVector* device_theCellTracks_ = nullptr;
  cudautils::device::unique_ptr<CAConstants::CellTracks[]> device_theCellTracksContainer_;

  cudautils::device::unique_ptr<GPUCACell[]> device_theCells_;
  cudautils::device::unique_ptr<GPUCACell::OuterHitOfCell[]> device_isOuterHitOfCell_;
  uint32_t* device_nCells_ = nullptr;

  cudautils::device::unique_ptr<HitToTuple> device_hitToTuple_;
  AtomicPairCounter* device_hitToTuple_apc_ = nullptr;

  AtomicPairCounter* device_hitTuple_apc_ = nullptr;

  cudautils::device::unique_ptr<TupleMultiplicity> device_tupleMultiplicity_;
  
  uint8_t * device_tmws_;

  cudautils::device::unique_ptr<AtomicPairCounter::c_type[]> device_storage_;
  // params
  Params const & m_params;
};

#endif  // RecoPixelVertexing_PixelTriplets_plugins_CAHitNtupletGeneratorKernels_h
