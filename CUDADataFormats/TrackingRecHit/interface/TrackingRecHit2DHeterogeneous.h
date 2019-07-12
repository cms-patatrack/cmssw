#ifndef CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
#define CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

#include "CUDADataFormats/TrackingRecHit/interface/TrackingRecHit2DSOAView.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"


// to be moved elsewhere
namespace cudaCompat {

  struct CUDATraits {

    template<typename T>
    using unique_ptr =  cudautils::device::unique_ptr<T>;

    template<typename T>
    static auto make_host_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_host_unique<T>(stream);
    }


    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService> & cs, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(stream);
    }

    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService> & cs, size_t size, cuda::stream_t<> &stream)    {
      return cs->make_device_unique<T>(size, stream);
    }


  };


  struct HostTraits {

    template<typename T>
    using unique_ptr =  std::unique_ptr<T>;

    template<typename T>
    static auto make_host_unique(edm::Service<CUDAService>&, cuda::stream_t<> &)    {
      return std::make_unique<T>();
    }


    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService>&, cuda::stream_t<> &)    {
      return std::make_unique<T>();
    }

    template<typename T>
    static auto make_device_unique(edm::Service<CUDAService>&, size_t size, cuda::stream_t<> &)    {
      return std::make_unique<T>(size);
    }


  };
}


template<typename Traits> 
class TrackingRecHit2DHeterogeneous {
public:

  template<typename T>
  using unique_ptr = typename Traits:: template unique_ptr<T>;

  using Hist = TrackingRecHit2DSOAView::Hist;

  TrackingRecHit2DHeterogeneous() = default;

  explicit TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                pixelCPEforGPU::ParamsOnGPU const* cpeParams,
                                uint32_t const* hitsModuleStart,
                                cuda::stream_t<>& stream);
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
  cudautils::host::unique_ptr<float[]> localCoordToHostAsync(cuda::stream_t<>& stream) const;
  cudautils::host::unique_ptr<uint16_t[]> detIndexToHostAsync(cuda::stream_t<>& stream) const;
  cudautils::host::unique_ptr<uint32_t[]> hitsModuleStartToHostAsync(cuda::stream_t<>& stream) const;

private:
  static constexpr uint32_t n16 = 4;
  static constexpr uint32_t n32 = 9;
  static_assert(sizeof(uint32_t) == sizeof(float));  // just stating the obvious

  unique_ptr<uint16_t[]> m_store16;
  unique_ptr<float[]> m_store32;

  unique_ptr<TrackingRecHit2DSOAView::Hist> m_HistStore;
  unique_ptr<TrackingRecHit2DSOAView::AverageGeometry> m_AverageGeometryStore;

  unique_ptr<TrackingRecHit2DSOAView> m_view;

  uint32_t m_nHits;

  uint32_t const* m_hitsModuleStart;  // needed for legacy, this is on GPU!

  // needed as kernel params...
  Hist* m_hist;
  uint32_t* m_hitsLayerStart;
  int16_t* m_iphi;
};


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"


template<typename Traits>
TrackingRecHit2DHeterogeneous<Traits>::TrackingRecHit2DHeterogeneous(uint32_t nHits,
                                           pixelCPEforGPU::ParamsOnGPU const *cpeParams,
                                           uint32_t const *hitsModuleStart,
                                           cuda::stream_t<> &stream)
    : m_nHits(nHits), m_hitsModuleStart(hitsModuleStart) {
  edm::Service<CUDAService> cs;


  auto view = Traits:: template make_host_unique<TrackingRecHit2DSOAView>(cs,stream);

  view->m_nHits = nHits;
  m_view = Traits:: template make_device_unique<TrackingRecHit2DSOAView>(cs,stream);
  m_AverageGeometryStore = Traits:: template make_device_unique<TrackingRecHit2DSOAView::AverageGeometry>(cs,stream);
  view->m_averageGeometry = m_AverageGeometryStore.get();
  view->m_cpeParams = cpeParams;
  view->m_hitsModuleStart = hitsModuleStart;

  // if empy do not bother
  if (0 == nHits) {
    if /*constexpr*/ (std::is_same<Traits,cudaCompat::CUDATraits>::value) {
#ifndef VIEW_ON_HOST 
      cudautils::copyAsync(m_view, view, stream);
#endif
    } else { m_view.reset(view.release());}
    return;
  }

  // the single arrays are not 128 bit alligned...
  // the hits are actually accessed in order only in building
  // if ordering is relevant they may have to be stored phi-ordered by layer or so
  // this will break 1to1 correspondence with cluster and module locality
  // so unless proven VERY inefficient we keep it ordered as generated
  m_store16 = Traits:: template make_device_unique<uint16_t[]>(cs, nHits * n16, stream);
  m_store32 = Traits:: template make_device_unique<float[]>(cs, nHits * n32 + 11, stream);
  m_HistStore = Traits:: template make_device_unique<TrackingRecHit2DSOAView::Hist>(cs, stream);

  auto get16 = [&](int i) { return m_store16.get() + i * nHits; };
  auto get32 = [&](int i) { return m_store32.get() + i * nHits; };

  // copy all the pointers
  m_hist = view->m_hist = m_HistStore.get();

  view->m_xl = get32(0);
  view->m_yl = get32(1);
  view->m_xerr = get32(2);
  view->m_yerr = get32(3);

  view->m_xg = get32(4);
  view->m_yg = get32(5);
  view->m_zg = get32(6);
  view->m_rg = get32(7);

  m_iphi = view->m_iphi = reinterpret_cast<int16_t *>(get16(0));

  view->m_charge = reinterpret_cast<int32_t *>(get32(8));
  view->m_xsize = reinterpret_cast<int16_t *>(get16(2));
  view->m_ysize = reinterpret_cast<int16_t *>(get16(3));
  view->m_detInd = get16(1);

  m_hitsLayerStart = view->m_hitsLayerStart = reinterpret_cast<uint32_t *>(get32(n32));

  // transfer view
    if /*constexpr*/ (std::is_same<Traits,cudaCompat::CUDATraits>::value) {
#ifndef VIEW_ON_HOST
      cudautils::copyAsync(m_view, view, stream);
#endif
    } else { m_view.reset(view.release());}
}


using TrackingRecHit2DCUDA = TrackingRecHit2DHeterogeneous<cudaCompat::CUDATraits>;
using TrackingRecHit2DHost = TrackingRecHit2DHeterogeneous<cudaCompat::HostTraits>;


#endif  // CUDADataFormats_TrackingRecHit_interface_TrackingRecHit2DHeterogeneous_h
