#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h

#include <cuda_runtime.h>

#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HistoContainer.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "Geometry/TrackerGeometryBuilder/interface/phase1PixelTopology.h"
#include "CUDADataFormats/TrackingRecHit/interface/SiPixelStatus.h"

namespace pixelCPEforGPU {
  struct ParamsOnGPU;
}

class TrackingRecHit2DSOAView {
public:
  using Status = SiPixelStatus;
  static_assert(sizeof(Status) == sizeof(uint8_t));

  static constexpr uint32_t maxHits() { return gpuClustering::MaxNumClusters; }
  using hindex_type = uint32_t;  // if above is <=2^32

  using PhiBinner =
      cms::cuda::HistoContainer<int16_t, 128, gpuClustering::MaxNumClusters, 8 * sizeof(int16_t), hindex_type, 10>;

  using Hist = PhiBinner;  // FIXME

  using AverageGeometry = phase1PixelTopology::AverageGeometry;

  template <typename>
  friend class TrackingRecHit2DHeterogeneous;
  friend class TrackingRecHit2DReduced;

  __device__ __forceinline__ uint32_t nHits() const { return m_nHits; }

  __device__ __forceinline__ float& xLocal(int i) { return m_xl[i]; }
  __device__ __forceinline__ float xLocal(int i) const { return __ldg(m_xl + i); }
  __device__ __forceinline__ float& yLocal(int i) { return m_yl[i]; }
  __device__ __forceinline__ float yLocal(int i) const { return __ldg(m_yl + i); }

  __device__ __forceinline__ float& xerrLocal(int i) { return m_xerr[i]; }
  __device__ __forceinline__ float xerrLocal(int i) const { return __ldg(m_xerr + i); }
  __device__ __forceinline__ float& yerrLocal(int i) { return m_yerr[i]; }
  __device__ __forceinline__ float yerrLocal(int i) const { return __ldg(m_yerr + i); }

  __device__ __forceinline__ float& xGlobal(int i) { return m_xg[i]; }
  __device__ __forceinline__ float xGlobal(int i) const { return __ldg(m_xg + i); }
  __device__ __forceinline__ float& yGlobal(int i) { return m_yg[i]; }
  __device__ __forceinline__ float yGlobal(int i) const { return __ldg(m_yg + i); }
  __device__ __forceinline__ float& zGlobal(int i) { return m_zg[i]; }
  __device__ __forceinline__ float zGlobal(int i) const { return __ldg(m_zg + i); }
  __device__ __forceinline__ float& rGlobal(int i) { return m_rg[i]; }
  __device__ __forceinline__ float rGlobal(int i) const { return __ldg(m_rg + i); }

  __device__ __forceinline__ int16_t& iphi(int i) { return m_iphi[i]; }
  __device__ __forceinline__ int16_t iphi(int i) const { return __ldg(m_iphi + i); }

  __device__ __forceinline__ void setChargeAndStatus(int i, uint32_t ich, Status is) {
    // static_assert(0xffffff == chargeMask());
    ich = std::min(ich, chargeMask());
    uint32_t w = *reinterpret_cast<uint8_t*>(&is);
    ich |= (w << 24);
    m_chargeAndStatus[i] = ich;
  }

  __device__ __forceinline__ Status status(int i) const {
    uint8_t w = __ldg(m_chargeAndStatus + i) >> 24;
    return *reinterpret_cast<Status*>(&w);
  }
  __device__ __forceinline__ uint32_t charge(int i) const { return __ldg(m_chargeAndStatus + i) & chargeMask(); }
  __device__ __forceinline__ int16_t& clusterSizeX(int i) { return m_xsize[i]; }
  __device__ __forceinline__ int16_t clusterSizeX(int i) const { return __ldg(m_xsize + i); }
  __device__ __forceinline__ int16_t& clusterSizeY(int i) { return m_ysize[i]; }
  __device__ __forceinline__ int16_t clusterSizeY(int i) const { return __ldg(m_ysize + i); }
  __device__ __forceinline__ uint16_t& detectorIndex(int i) { return m_detInd[i]; }
  __device__ __forceinline__ uint16_t detectorIndex(int i) const { return __ldg(m_detInd + i); }

  __device__ __forceinline__ pixelCPEforGPU::ParamsOnGPU const& cpeParams() const { return *m_cpeParams; }

  __device__ __forceinline__ uint32_t hitsModuleStart(int i) const { return __ldg(m_hitsModuleStart + i); }

  __device__ __forceinline__ uint32_t* hitsLayerStart() { return m_hitsLayerStart; }
  __device__ __forceinline__ uint32_t const* hitsLayerStart() const { return m_hitsLayerStart; }

  __device__ __forceinline__ Hist& phiBinner() { return *m_hist; }
  __device__ __forceinline__ Hist const& phiBinner() const { return *m_hist; }

  __device__ __forceinline__ AverageGeometry& averageGeometry() { return *m_averageGeometry; }
  __device__ __forceinline__ AverageGeometry const& averageGeometry() const { return *m_averageGeometry; }

private:
  // local coord
  float *m_xl, *m_yl;
  float *m_xerr, *m_yerr;

  // global coord
  float *m_xg, *m_yg, *m_zg, *m_rg;
  int16_t* m_iphi;

  // cluster properties
  static constexpr uint32_t chargeMask() { return (1 << 24) - 1; }
  uint32_t* m_chargeAndStatus;
  int16_t* m_xsize;
  int16_t* m_ysize;
  uint16_t* m_detInd;

  // supporting objects
  AverageGeometry* m_averageGeometry;  // owned (corrected for beam spot: not sure where to host it otherwise)
  pixelCPEforGPU::ParamsOnGPU const* m_cpeParams;  // forwarded from setup, NOT owned
  uint32_t const* m_hitsModuleStart;               // forwarded from clusters

  uint32_t* m_hitsLayerStart;

  PhiBinner* m_hist;  // FIXME use a more descriptive name consistently

  uint32_t m_nHits;
};

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DSOAView_h
