import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.gpu_cff import gpu

siPixelRecHits = cms.EDProducer("SiPixelRecHitConverter",
    src = cms.InputTag("siPixelClusters"),
    CPE = cms.string('PixelCPEGeneric'),
    VerboseLevel = cms.untracked.int32(0),
)

siPixelRecHitsPreSplitting = siPixelRecHits.clone(
    src = 'siPixelClustersPreSplitting'
)


from RecoLocalTracker.SiPixelRecHits.siPixelRecHitHeterogeneous_cfi import siPixelRecHitHeterogeneous

from RecoLocalTracker.SiPixelRecHits.siPixelRecHitHeterogeneousConverter_cfi import siPixelRecHitHeterogeneousConverter as _siPixelRecHitHeterogeneousConverter
gpu.toReplaceWith(siPixelRecHitsPreSplitting, _siPixelRecHitHeterogeneousConverter.clone())



