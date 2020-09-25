import FWCore.ParameterSet.Config as cms

def customizePixelTracksSoAonCPU(process) :

  process.load('RecoLocalTracker/SiPixelRecHits/siPixelRecHitHostSoA_cfi')
  process.load('RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi')
  process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi')

  process.pixelTrackSoA = process.caHitNtupletCUDA.clone()
  process.pixelTrackSoA.onGPU = False
  process.pixelTrackSoA.pixelRecHitSrc = 'siPixelRecHitHostSoA'
  process.pixelVertexSoA = process.pixelVertexCUDA.clone()
  process.pixelVertexSoA.onGPU = False
  process.pixelVertexSoA.pixelTrackSrc = 'pixelTrackSoA'

  process.load('RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi')
  process.pixelTracks = process.pixelTrackProducerFromSoA.clone()
  process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi')
  process.pixelVertices = process.pixelVertexFromSoA.clone()
  process.pixelTracks.pixelRecHitLegacySrc = 'siPixelRecHitHostSoA'
  process.siPixelRecHitHostSoA.convertToLegacy = True

  process.reconstruction_step += process.siPixelRecHitHostSoA+process.pixelTrackSoA+process.pixelVertexSoA

  return process

def customizePixelTracksSoAonCPUForWF(process) :

  process.load('RecoLocalTracker/SiPixelRecHits/siPixelRecHitHostSoA_cfi')
  process.load('RecoPixelVertexing.PixelTriplets.caHitNtupletCUDA_cfi')
  process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexCUDA_cfi')
  
  process.CUDAService = cms.Service("CUDAService")
  process.CUDAService. enabled = cms.untracked.bool(False)
  
  process.pixelTrackSoA = process.caHitNtupletCUDA.clone()
  process.pixelTrackSoA.onGPU = False
  process.pixelTrackSoA.pixelRecHitSrc = 'siPixelRecHitsPreSplitting'
  process.pixelVertexSoA = process.pixelVertexCUDA.clone()
  process.pixelVertexSoA.onGPU = False
  process.pixelVertexSoA.pixelTrackSrc = 'pixelTrackSoA'

  process.load('RecoPixelVertexing.PixelTrackFitting.pixelTrackProducerFromSoA_cfi')
  process.pixelTracks = process.pixelTrackProducerFromSoA.clone()
  process.load('RecoPixelVertexing.PixelVertexFinding.pixelVertexFromSoA_cfi')
  process.pixelVertices = process.pixelVertexFromSoA.clone()
  process.pixelTracks.pixelRecHitLegacySrc = 'siPixelRecHitsPreSplitting'

  process.siPixelRecHitsPreSplitting = process.siPixelRecHitHostSoA.clone()
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
