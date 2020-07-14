#ifndef TrackerRecHit2D_OmniClusterRef_H
#define TrackerRecHit2D_OmniClusterRef_H

#include "DataFormats/Common/interface/RefCoreWithIndex.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"

class OmniClusterRef {
  // OK this need to be fixed: the mask shall represent an integer, not just a bit patter as the omniref can be just ONE thing not all of them at the same time
  static const unsigned int kInvalid = 0x80000000;  // bit 31 on
  static const unsigned int kIsStrip = 0x20000000;  // bit 29 on
  // FIXME:: need to check when introducing phase2 pixel
  static const unsigned int kIsPhase2 = 0x40000000;    // bit 30 on
  static const unsigned int kIsTiming = 0x10000000;    // bit 28 on
  static const unsigned int kIsRegional = 0x60000000;  // bit 30 and 29 on  (will become fastsim???)
  static const unsigned int kIsSoA = 0x8000000;        // bit 27 on (to be fixed)

  static const unsigned int indexMask = 0xFFFFFF;
  static const unsigned int subClusMask = 0x7000000;  // 8 subcluster are enough
  static const unsigned int subClusShift = 24;

public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> ClusterPixelRef;
  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>, SiStripCluster> ClusterStripRef;
  typedef edm::Ref<edmNew::DetSetVector<Phase2TrackerCluster1D>, Phase2TrackerCluster1D> Phase2Cluster1DRef;
  typedef edm::Ref<FTLClusterCollection, FTLCluster> ClusterMTDRef;

  OmniClusterRef() : me(edm::RefCore(), kInvalid) {}
  explicit OmniClusterRef(ClusterPixelRef const& ref, unsigned int subClus = 0)
      : me(ref.refCore(), (ref.isNonnull() ? ref.key() | (subClus << subClusShift) : kInvalid)) {}
  explicit OmniClusterRef(ClusterStripRef const& ref, unsigned int subClus = 0)
      : me(ref.refCore(), (ref.isNonnull() ? (ref.key() | kIsStrip) | (subClus << subClusShift) : kInvalid)) {}
  explicit OmniClusterRef(Phase2Cluster1DRef const& ref, unsigned int subClus = 0)
      : me(ref.refCore(), (ref.isNonnull() ? (ref.key() | kIsPhase2) | (subClus << subClusShift) : kInvalid)) {}
  explicit OmniClusterRef(ClusterMTDRef const& ref)
      : me(ref.refCore(), (ref.isNonnull() ? (ref.key() | kIsTiming) : kInvalid)) {}

  explicit OmniClusterRef(edm::RefCore const& ref, int index)
      : me(ref, (ref.isNonnull() ? (index | kIsSoA) : kInvalid)) {}
  template <typename Prod>
  explicit OmniClusterRef(edm::RefProd<Prod> const& ref, int index) : OmniClusterRef(ref.refCore(), index) {}

  ClusterPixelRef cluster_pixel() const {
    return (isPixel() && isValid()) ? ClusterPixelRef(me.toRefCore(), index()) : ClusterPixelRef();
  }

  ClusterStripRef cluster_strip() const {
    return isStrip() ? ClusterStripRef(me.toRefCore(), index()) : ClusterStripRef();
  }

  Phase2Cluster1DRef cluster_phase2OT() const {
    return isPhase2() ? Phase2Cluster1DRef(me.toRefCore(), index()) : Phase2Cluster1DRef();
  }

  ClusterMTDRef cluster_mtd() const { return isTiming() ? ClusterMTDRef(me.toRefCore(), index()) : ClusterMTDRef(); }

  SiPixelCluster const& pixelCluster() const { return *ClusterPixelRef(me.toRefCore(), index()); }
  SiStripCluster const& stripCluster() const { return *ClusterStripRef(me.toRefCore(), index()); }
  Phase2TrackerCluster1D const& phase2OTCluster() const { return *Phase2Cluster1DRef(me.toRefCore(), index()); }
  FTLCluster const& mtdCluster() const { return *ClusterMTDRef(me.toRefCore(), index()); }

  bool operator==(OmniClusterRef const& lh) const {
    return (rawIndex() | kIsSoA) == (lh.rawIndex() | kIsSoA);  // in principle this is enough!
  }

  bool operator<(OmniClusterRef const& lh) const {
    return (rawIndex() | kIsSoA) < (lh.rawIndex() | kIsSoA);  // in principle this is enough!
  }

  bool operator>(OmniClusterRef const& lh) const {
    return (rawIndex() | kIsSoA) > (lh.rawIndex() | kIsSoA);  // in principle this is enough!
  }

public:
  // edm Ref interface
  edm::RefCore const& refCore() const { return me.toRefCore(); }
  edm::ProductID id() const { return me.id(); }
  unsigned int key() const { return index(); }

  unsigned int rawIndex() const { return me.index(); }

  unsigned int index() const { return rawIndex() & indexMask; }

  unsigned int subCluster() const { return (rawIndex() & subClusMask) >> subClusShift; }

  bool isValid() const { return !(rawIndex() & kInvalid); }
  bool isPixel() const { return !isStrip() && !isPhase2(); }  //NOTE: non-valid will also show up as a pixel
  bool isStrip() const { return rawIndex() & kIsStrip; }
  bool isPhase2() const { return rawIndex() & kIsPhase2; }
  bool isTiming() const { return rawIndex() & kIsTiming; }
  bool isSoA() const { return rawIndex() & kIsSoA; }

  // bool isRegional() const { return (rawIndex() & kIsRegional)==kIsRegional; }
  // bool isNonRegionalStrip() const {return (rawIndex() & kIsRegional)==kIsStrip;}

private:
  edm::RefCoreWithIndex me;
};

#endif  // TrackerRecHit2D_OmniClusterRef_H
