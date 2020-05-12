import FWCore.ParameterSet.Config as cms

def useTMTT(process):
    from L1Trigger.TrackerDTC.Producer_Defaults_cfi import TrackerDTCProducer_params
    from L1Trigger.TrackerDTC.Format_TMTT_cfi import TrackerDTCFormat_params
    from L1Trigger.TrackerDTC.Analyzer_Defaults_cfi import TrackerDTCAnalyzer_params
    TrackerDTCProducer_params.ParamsED.DataFormat = "TMTT"
    TrackerDTCAnalyzer_params.ParamsTP.MinPt = cms.double( 3. )
    process.TrackerDTCProducer = cms.EDProducer('trackerDTC::Producer', TrackerDTCProducer_params, TrackerDTCFormat_params, TrackerDTCAnalyzer_params)
    return process