#ifndef RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h
#define RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDoublets_h

#include "RecoPixelVertexing/PixelTriplets/plugins/gpuPixelDoubletsAlgos.h"

#define CONSTANT_VAR __constant__

namespace gpuPixelDoublets {

  constexpr int nPairs = 13 + 2 + 4;
  constexpr int nPairsUpgrade = 6 + 14 + 3 + 8 + 6 + 14;

  static_assert(nPairs <= CAConstants::maxNumberOfLayerPairs());
  static_assert(nPairsUpgrade <= CAConstants::maxNumberOfLayerPairs());

  // start constants
  // clang-format off

  CONSTANT_VAR const uint8_t layerPairs[2 * nPairs] = {
      0, 1, 0, 4, 0, 7,              // BPIX1 (3)
      1, 2, 1, 4, 1, 7,              // BPIX2 (5)
      4, 5, 7, 8,                    // FPIX1 (8)
      2, 3, 2, 4, 2, 7, 5, 6, 8, 9,  // BPIX3 & FPIX2 (13)
      0, 2, 1, 3,                    // Jumping Barrel (15)
      0, 5, 0, 8,                    // Jumping Forward (BPIX1,FPIX2)
      4, 6, 7, 9                     // Jumping Forward (19)
  };

    CONSTANT_VAR const uint8_t layerPairsUpgrade[2 * nPairsUpgrade] = {

    0, 1, 0, 4, 0, 16, //BPIX1 (3)
    1, 2, 1, 4, 1, 16, //BPIX2 (6)

    4 ,5 ,5 ,6 ,6 ,7 ,7 ,8 ,8 ,9 ,9 ,10,10,11, //POS (13)
    16,17,17,18,18,19,19,20,20,21,21,22,22,23, //NEG (20)

    2, 3, 2, 4, 2, 16, //Barrel Jump (23)

    11,12,12,13,13,14,14,15, //Late POS (27)
    23,24,24,25,25,26,26,27, //Late NEG (31)

    0, 2, 0, 5, 0, 17, // BPIX1 Jump (34)
    1, 3, 1, 5, 1, 17, // BPIX2 Jump (37)

    4, 6, 5, 7, 6, 8, 7, 9, 8, 10,9 ,11,10,12, //POS Jump (44)
    16,18,17,19,18,20,19,21,20,22,21,23,22,24 //NEG Jump (51)


};

  constexpr int16_t phi0p05 = 522;  // round(521.52189...) = phi2short(0.05);
  constexpr int16_t phi0p06 = 626;  // round(625.82270...) = phi2short(0.06);
  constexpr int16_t phi0p07 = 730;  // round(730.12648...) = phi2short(0.07);
  constexpr int16_t phi0p08 = 840;
  
  CONSTANT_VAR const int16_t phicutsUpgrade[nPairsUpgrade]{
  
       phi0p07, phi0p08, phi0p08,
       phi0p07, phi0p08, phi0p08,
       phi0p07, phi0p07, phi0p07,phi0p07, phi0p07, phi0p07, phi0p07,
       phi0p07, phi0p07, phi0p07,phi0p07, phi0p07, phi0p07, phi0p07, 
       phi0p07, phi0p08, phi0p08, 
       phi0p07, phi0p07, phi0p07,phi0p07,
       phi0p07, phi0p07, phi0p07,phi0p07,
       phi0p07, phi0p08, phi0p08,
       phi0p07, phi0p08, phi0p08,   
       phi0p07, phi0p07, phi0p07,phi0p07, phi0p07, phi0p07, phi0p07,
       phi0p07, phi0p07, phi0p07,phi0p07, phi0p07, phi0p07, phi0p07};

  
 CONSTANT_VAR const int16_t phicuts[nPairs]{phi0p05,
                                             phi0p07,
                                             phi0p07,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p06,
                                             phi0p06,
                                             phi0p06,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05,
                                             phi0p05};
  //   phi0p07, phi0p07, phi0p06,phi0p06, phi0p06,phi0p06};  // relaxed cuts

  CONSTANT_VAR float const minz[nPairs] = {
      -20., 0., -30., -22., 10., -30., -70., -70., -22., 15., -30, -70., -70., -20., -22., 0, -30., -70., -70.};
  CONSTANT_VAR float const maxz[nPairs] = {
      20., 30., 0., 22., 30., -10., 70., 70., 22., 30., -15., 70., 70., 20., 22., 30., 0., 70., 70.};
  CONSTANT_VAR float const maxr[nPairs] = {
      20., 9., 9., 20., 7., 7., 5., 5., 20., 6., 6., 5., 5., 20., 20., 9., 9., 9., 9.};

  CONSTANT_VAR float const minzUpgrade[nPairsUpgrade] = {
        -17.,   -1.,  -22.,
        -17.,    4.,  -22.,
         22.,   28.,   36.,  47.,   60.,   76.,   98.,
        -28.,  -36.,  -46.,  -58.,  -74., -95., -121.,
        -19.,   -2.,  -22.,

        125.,  157.,  180.,  207.,
       -155., -193., -222., -255.,
        -17.,   -1.,  -22.,
        -17.,    4.,  -22.,
         22.,   28.,   36.,   47.,   60., 76.,   98.,
        -28.,  -36.,  -46.,  -58.,  -74.,  -95., -121.};

    CONSTANT_VAR float const maxzUpgrade[nPairsUpgrade] = {
         17.,   22.,    1.,
         17.,   22.,   -4.,
         28.,   36.,   46.,   58.,   74.,   95.,  121.,
        -22.,  -28.,  -36.,  -47.,  -60.,  -76.,  -98.,
         19.,   22.,    2.,
        155.,  193.,  222.,  255.,
       -125., -157., -180., -207.,
         17.,   22.,    1.,
         17.,   22.,   -4.,
         28.,   36.,   46.,   58.,   74., 95.,  121.,
        -22.,  -28.,  -36.,  -47.,  -60.,  -76.,  -98.};

  CONSTANT_VAR float const maxrUpgrade[nPairsUpgrade] = {
        6. , 12. , 13.5,  7.5, 12. , 12. ,  7.5,  7.5,  7.5,  6. ,  6. ,
        6. ,  6. ,  7.5,  7.5,  6. ,  6. ,  6. ,  6. ,  6. ,  7.5,  9. ,
        9. ,  7.5,  6. ,  6. ,  6. ,  7.5,  6. ,  6. ,  6. , 12. , 12. ,
       10.5, 10.5, 10.5, 10.5, 15. , 12. , 12. , 10.5, 10.5, 10.5, 10.5,
       15.};
   CONSTANT_VAR float const maxZ0[nPairs] = {12.0};

    CONSTANT_VAR float const maxZ0Upgrade[nPairsUpgrade] = {
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,
        9.,  9.,  9.,  9.,  9., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
        12., 12., 12., 12., 12., 12. };

  // end constants
  // clang-format on

  using CellNeighbors = CAConstants::CellNeighbors;
  using CellTracks = CAConstants::CellTracks;
  using CellNeighborsVector = CAConstants::CellNeighborsVector;
  using CellTracksVector = CAConstants::CellTracksVector;

  __global__ void initDoublets(GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                               int nHits,
                               CellNeighborsVector* cellNeighbors,
                               CellNeighbors* cellNeighborsContainer,
                               CellTracksVector* cellTracks,
                               CellTracks* cellTracksContainer) {
    assert(isOuterHitOfCell);
    int first = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = first; i < nHits; i += gridDim.x * blockDim.x)
      isOuterHitOfCell[i].reset();

    if (0 == first) {
      cellNeighbors->construct(CAConstants::maxNumOfActiveDoublets(), cellNeighborsContainer);
      cellTracks->construct(CAConstants::maxNumOfActiveDoublets(), cellTracksContainer);
      auto i = cellNeighbors->extend();
      assert(0 == i);
      (*cellNeighbors)[0].reset();
      i = cellTracks->extend();
      assert(0 == i);
      (*cellTracks)[0].reset();
    }
  }

  constexpr auto getDoubletsFromHistoMaxBlockSize = 64;  // for both x and y
  constexpr auto getDoubletsFromHistoMinBlocksPerMP = 16;

  __global__
#ifdef __CUDACC__
  __launch_bounds__(getDoubletsFromHistoMaxBlockSize, getDoubletsFromHistoMinBlocksPerMP)
#endif
      void getDoubletsFromHisto(GPUCACell* cells,
                                uint32_t* nCells,
                                CellNeighborsVector* cellNeighbors,
                                CellTracksVector* cellTracks,
                                TrackingRecHit2DSOAView const* __restrict__ hhp,
                                GPUCACell::OuterHitOfCell* isOuterHitOfCell,
                                int nActualPairs,
                                bool ideal_cond,
                                bool doClusterCut,
                                bool doZ0Cut,
                                bool doPtCut,
                                uint32_t maxNumOfDoublets,
                                bool upgrade) {
    auto const& __restrict__ hh = *hhp;
    doubletsFromHisto(upgrade ? layerPairsUpgrade : layerPairs,
                      nActualPairs,
                      cells,
                      nCells,
                      cellNeighbors,
                      cellTracks,
                      hh,
                      isOuterHitOfCell,
                      upgrade ? phicutsUpgrade : phicuts,
                      upgrade ? minzUpgrade : minz,
                      upgrade ? maxzUpgrade : maxz,
                      upgrade ? maxrUpgrade : maxr,
                      ideal_cond,
                      doClusterCut,
                      doZ0Cut,
                      doPtCut,
                      maxNumOfDoublets,
                      upgrade);
  }

}  // namespace gpuPixelDoublets

#endif  // RecoLocalTracker_SiPixelRecHits_plugins_gpuPixelDouplets_h
