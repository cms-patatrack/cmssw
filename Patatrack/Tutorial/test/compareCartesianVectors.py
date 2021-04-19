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

process.convertToCartesianVectorsCUDA = cms.EDProducer('ConvertToCartesianVectorsCUDA',
  input = cms.InputTag('generateCylindricalVectors')
)

process.convertToCartesianVectors = cms.EDProducer('ConvertToCartesianVectors',
  input = cms.InputTag('generateCylindricalVectors')
)

process.compareCartesianVectors = cms.EDAnalyzer('CompareCartesianVectors',
  first = cms.InputTag('convertToCartesianVectorsCUDA'),
  second = cms.InputTag('convertToCartesianVectors'),
  precision = cms.double(1.e-7)
)

process.path = cms.Path(process.convertToCartesianVectorsCUDA + process.convertToCartesianVectors + process.compareCartesianVectors)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( -1 )
)
