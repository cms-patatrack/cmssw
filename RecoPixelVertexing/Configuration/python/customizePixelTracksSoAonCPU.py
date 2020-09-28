import FWCore.ParameterSet.Config as cms

def customizePixelTracksSoAonCPU(process) :

  process.load('RecoLocalTracker/SiPixelRecHits/siPixelRecHitHostSoA_cfi')
  
  from RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi import *
  from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import *
 
  from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import *
  from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import *
  
  process.CUDAService = cms.Service("CUDAService")
  process.CUDAService.enabled = cms.untracked.bool(False)

  process.pixelTrackSoA = caHitNtupletCUDA.clone()
  process.pixelTrackSoA.onGPU = False
  process.pixelTrackSoA.pixelRecHitSrc = 'siPixelRecHitHostSoA'
  process.pixelVertexSoA = pixelVertexCUDA.clone()
  process.pixelVertexSoA.onGPU = False
  process.pixelVertexSoA.pixelTrackSrc = 'pixelTrackSoA' 

  process.pixelTracks = pixelTrackProducerFromSoA.clone()
  process.pixelVertices = pixelVertexFromSoA.clone()
  process.pixelTracks.pixelRecHitLegacySrc = 'siPixelRecHitHostSoA'
  process.siPixelRecHitHostSoA.convertToLegacy = True

  process.reconstruction_step += process.siPixelRecHitHostSoA+process.pixelTrackSoA+process.pixelVertexSoA

  return process

def customizePixelTracksSoAonCPUForWF(process) :

  from RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi import *
  from RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi import *
  from RecoLocalTracker.SiPixelRecHits.siPixelRecHitHostSoA_cfi import *
  from RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi import *
  from RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi import *

  process.CUDAService = cms.Service("CUDAService")
  process.CUDAService.enabled = cms.untracked.bool(False)
  
  process.pixelTrackSoA = caHitNtupletCUDA.clone()
  process.pixelTrackSoA.onGPU = False
  process.pixelTrackSoA.pixelRecHitSrc = 'siPixelRecHitsPreSplitting'
  process.pixelVertexSoA = pixelVertexCUDA.clone()
  process.pixelVertexSoA.onGPU = False
  process.pixelVertexSoA.pixelTrackSrc = 'pixelTrackSoA'

  process.pixelTracks = pixelTrackProducerFromSoA.clone()

  process.pixelVertices = pixelVertexFromSoA.clone()
  process.pixelTracks.pixelRecHitLegacySrc = 'siPixelRecHitsPreSplitting'

  process.siPixelRecHitsPreSplitting = siPixelRecHitHostSoA.clone()
  process.siPixelRecHitsPreSplitting.convertToLegacy = True

  process.reconstruction_step += process.pixelTrackSoA+process.pixelVertexSoA

  return process

def customizePixelTracksForTriplets(process) :
 
  if 'caHitNtupletCUDA' in process.__dict__:
        process.caHitNtupletCUDA.includeJumpingForwardDoublets = True
        process.caHitNtupletCUDA.minHitsPerNtuplet = 3
  elif 'pixelTrackSoA' in process.__dict__:
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
