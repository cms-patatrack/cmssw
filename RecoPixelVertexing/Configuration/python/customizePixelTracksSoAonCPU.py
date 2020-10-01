import FWCore.ParameterSet.Config as cms

def customizePixelTracksSoAonCPU(process) :
  
  process.CUDAService = cms.Service("CUDAService",
    enabled = cms.untracked.bool(False)
  )

  from RecoLocalTracker.SiPixelRecHits.siPixelRecHitHostSoA_cfi import *
  process.siPixelRecHitsPreSplitting = siPixelRecHitHostSoA.clone(
    convertToLegacy = True
  )

  from RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi import *
  process.pixelTrackSoA = caHitNtupletCUDA.clone(
    onGPU = False,
    pixelRecHitSrc = 'siPixelRecHitsPreSplitting'
  )

  from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import *
  process.pixelVertexSoA = pixelVertexCUDA.clone(
    onGPU = False,
    pixelTrackSrc = 'pixelTrackSoA'
  )

  from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import *
  process.pixelTracks = pixelTrackProducerFromSoA.clone(
    pixelRecHitLegacySrc = 'siPixelRecHitsPreSplitting'
  )

  from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import *
  process.pixelVertices = pixelVertexFromSoA.clone()

  process.reconstruction_step += process.siPixelRecHitsPreSplitting + process.pixelTrackSoA + process.pixelVertexSoA

  return process


def customizePixelTracksForTriplets(process) :
 
  if 'caHitNtupletCUDA' in process.__dict__:
        process.caHitNtupletCUDA.includeJumpingForwardDoublets = True
        process.caHitNtupletCUDA.minHitsPerNtuplet = 3

  if 'pixelTrackSoA' in process.__dict__:
        process.pixelTrackSoA.includeJumpingForwardDoublets = True
        process.pixelTrackSoA.minHitsPerNtuplet = 3
 
  return process
 

def customizePixelTracksSoAonCPUForProfiling(process) :

  process.MessageLogger.cerr.FwkReport.reportEvery = 100

  process = customizePixelTracksSoAonCPU(process)
  process.siPixelRecHitHostSoA.convertToLegacy = False
  
  process.TkSoA = cms.Path(process.offlineBeamSpot+process.siPixelDigis+process.siPixelClustersPreSplitting+process.siPixelRecHitHostSoA+process.pixelTrackSoA+process.pixelVertexSoA)
  process.schedule = cms.Schedule(process.TkSoA)
  return process
