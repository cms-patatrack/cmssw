import copy
import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.CustomConfigs import MassReplaceInputTag


#
# changes required to consume RecHits instead of Clusters
#
def customizeHLTforPixelRecHits(process) :

  process = MassReplaceInputTag(process,"hltSiPixelClusters","hltSiPixelRecHits")
  process.hltSiPixelRecHits.src = "hltSiPixelClusters"

# need to be fixed
  process.hltSiStripClusters.pixelClusterProducer = "hltSiPixelRecHits"

#more mess

  process = MassReplaceInputTag(process,"hltSiPixelClustersRegForBTag","hltSiPixelRecHitsRegForBTag")
  process.hltSiStripClustersRegForBTag.pixelClusterProducer = "hltSiPixelRecHitsRegForBTag"
  process.hltSiPixelRecHitsRegForBTag.src  = cms.InputTag( "hltSiPixelClustersRegForBTag")
  process.hltFastPrimaryVertex.clusters = cms.InputTag( "hltSiPixelClustersRegForBTag")


  process = MassReplaceInputTag(process,"hltSiPixelClustersRegForTau","hltSiPixelRecHitsRegForTau")
# process.hltSiStripClustersRegForTau.pixelClusterProducer = "hltSiPixelRecHitsRegForTau"
  process.hltSiPixelRecHitsRegForTau.src  = cms.InputTag( "hltSiPixelClustersRegForTau")

  process = MassReplaceInputTag(process,"hltSiPixelClustersRegL1TauSeeded","hltSiPixelRecHitsRegL1TauSeeded")
# process.hltSiStripClustersRegL1TauSeeded.pixelClusterProducer = "hltSiPixelRecHitsRegL1TauSeeded"
  process.hltSiPixelRecHitsRegL1TauSeeded.src  = cms.InputTag( "hltSiPixelClustersRegL1TauSeeded")


  process.hltSiPixelClustersCache.src = cms.InputTag( "hltSiPixelClusters" )
  process.hltSiPixelClustersRegForBTagCache.src = cms.InputTag( "hltSiPixelClustersRegForBTag" )

  process.hltSiPixelClustersRegForTauCache.src = cms.InputTag( "hltSiPixelClustersRegForTau")
  process.hltSiPixelClustersRegL1TauSeededCache.src = cms.InputTag( "hltSiPixelClustersRegL1TauSeeded")

  return process
















    
