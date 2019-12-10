import FWCore.ParameterSet.Config as cms

process = cms.Process("PRINT")

process.options = cms.untracked.PSet(
  numberOfThreads = cms.untracked.uint32( 1 ),
  numberOfStreams = cms.untracked.uint32( 1 ),
  wantSummary = cms.untracked.bool( False )
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
process.MessageLogger.categories.append("CUDAService")

process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring("file:cylindricalVectors.root"),
)

process.convertToCartesianVectors = cms.EDProducer('ConvertToCartesianVectorsCUDA',
  input = cms.InputTag('generateCylindricalVectors')
)

process.printCartesianVectors = cms.EDAnalyzer('PrintCartesianVectors',
  input = cms.InputTag('convertToCartesianVectors')
)

process.path = cms.Path(process.convertToCartesianVectors + process.printCartesianVectors)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 1 )
)
