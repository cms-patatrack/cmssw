import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

process.options = cms.untracked.PSet(
  numberOfThreads = cms.untracked.uint32( 8 ),
  numberOfStreams = cms.untracked.uint32( 0 ),
  wantSummary = cms.untracked.bool( True )
)

process.source = cms.Source("EmptySource")

process.generateCylindricalVectors = cms.EDProducer('GenerateCylindricalVectors',
  size = cms.uint32(10000)
)

process.path = cms.Path(process.generateCylindricalVectors)

process.out = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("cylindricalVectors.root"),
  outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_generateCylindricalVectors_*_*')
)

process.endp = cms.EndPath(process.out)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 1200 )
)
