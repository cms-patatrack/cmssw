#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"
#include "CUDADataFormats/Common/interface/HeterogeneousSoA.h"

template <typename Traits>
class TrackingRecHit2DHeterogeneous {
public:
  template <typename T>
  using unique_ptr = typename Traits::template unique_ptr<T>;

  using Hist = TrackingRecHit2DSOAView::Hist;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(
      uint32_t nHits,
      pixelCPEforGPU::ParamsOnGPU const* cpeParams,
      uint32_t const* hitsModuleStart,
      cudaStream_t stream,
      TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits> const* input = nullptr);

  ~TrackingRecHit2DHeterogeneous() = default;

  TrackingRecHit2DHeterogeneous(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous& operator=(const TrackingRecHit2DHeterogeneous&) = delete;
  TrackingRecHit2DHeterogeneous(TrackingRecHit2DHeterogeneous&&) = default;
  TrackingRecHit2DHeterogeneous& operator=(TrackingRecHit2DHeterogeneous&&) = default;

  TrackingRecHit2DSOAView* view() { return m_view.get(); }
  TrackingRecHit2DSOAView const* view() const { return m_view.get(); }

  auto nHits() const { return m_nHits; }

  auto hitsModuleStart() const { return m_hitsModuleStart; }
  auto hitsLayerStart() { return m_hitsLayerStart; }
  auto phiBinner() { return m_hist; }
  auto iphi() { return m_iphi; }

  // only the local coord and detector index
  cms::cuda::host::unique_ptr<float[]> localCoordToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cudaStream_t stream) const;
  cms::cuda::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cudaStream_t stream) const;

  // needs specialization for Host
  void copyFromGPU(TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits> const* input, cudaStream_t stream);

private:
  static constexpr uint32_t n16 = 4;
  static constexpr uint32_t n32 = 9;
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  unique_ptr<uint16_t[]> m_store16;  //!
  unique_ptr<float[]> m_store32;     //!

  unique_ptr<TrackingRecHit2DSOAView::Hist> m_HistStore;                        //!
  unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;  //!

  unique_ptr<TrackingRecHit2DSOAView> m_view;  //!

  uint32_t m_nHits;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  Hist* m_hist;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};

using TrackingRecHit2DGPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits>;
using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits>;
using TrackingRecHit2DCPU = TrackingRecHit2DHeterogeneous<cms::cudacompat::CPUTraits>;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous<cms::cudacompat::HostTraits>;

#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

template <typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(
    uint32_t nHits,
    pixelCPEforGPU::ParamsOnGPU const* cpeParams,
    uint32_t const* hitsModuleStart,
    cudaStream_t stream,
    TrackingRecHit2DHeterogeneous<cms::cudacompat::GPUTraits> const* input)
    : m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  auto view = Traits::template make_host_unique<TrackingRecHit2DSOAView>(stream);

  view->m_nHits = nHits;
  m_view = Traits::template make_unique<TrackingRecHit2DSOAView>(stream);  // leave it on host and pass it by value?
  m_AverageGeometryStore = Traits::template make_unique<TrackingRecHit2DSOAView::AverageGeometry>(stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) 
    if constexpr (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
      cms::cuda::copyAsync(m_view, view, stream);
    } else {
      m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
    }
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated

  // host copy is "reduced"  (to be reviewed at some point)
  if
#ifdef __cpp_if_constexpr
      constexpr
#endif
      (std::is_same<Traits, cms::cudacompat::HostTraits>::value) {
    // it has to compile for ALL cases
    copyFromGPU(input, stream);
  } else {
    assert(input == nullptr);
    m_store16 = Traits::template make_unique<uint16_t[]>(nHits * n16, stream);
    m_store32 = Traits::template make_unique<float[]>(nHits * n32 + 11, stream);
    m_HistStore = Traits::template make_unique<TrackingRecHit2DSOAView::Hist>(stream);
  }

  auto get32 = [&](int i) { return m_store32.get() + i * nHits; };

  // copy all the pointers

  view->m_xl = get32(0);
  view->m_yl = get32(1);
  view->m_xerr = get32(2);
  view->m_yerr = get32(3);
  view->m_chargeAndStatus = reinterpret_cast<uint32_t*>(get32(4));

  if
#ifdef __cpp_if_constexpr
      constexpr
#endif
      (!std::is_same<Traits, cms::cudacompat::HostTraits>::value) {
    assert(input == nullptr);
    view->m_xg = get32(5);
    view->m_yg = get32(6);
    view->m_zg = get32(7);
    view->m_rg = get32(8);

    auto get16 = [&](int i) { return m_store16.get() + i * nHits; };
    m_iphi = view->m_iphi = reinterpret_cast<int16_t*>(get16(1));

    view->m_xsize = reinterpret_cast<int16_t*>(get16(2));
    view->m_ysize = reinterpret_cast<int16_t*>(get16(3));
    view->m_detInd = get16(0);

    m_hist = view->m_hist = m_HistStore.get();
    m_hitsLayerStart = view->m_hitsLayerStart = reinterpret_cast<uint32_t*>(get32(n32));
  }

  // transfer view
  if
#ifdef __cpp_if_constexpr
      constexpr
#endif
      (std::is_same<Traits, cms::cudacompat::GPUTraits>::value) {
    cms::cuda::copyAsync(m_view, view, stream);
  } else {
    m_view.reset(view.release());  // NOLINT: std::move() breaks CUDA version
  }
}

#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
