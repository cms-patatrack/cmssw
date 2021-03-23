import FWCore.ParameterSet.Config as cms

ptSeedParameterization = cms.PSet(
    SMB_21 = cms.vdouble(0.982838, -0.181889, 0.0, 0.394683, -0.339892, 0.0),
    SMB_20 = cms.vdouble(0.89598, -0.07901, 0.0, 0.230235, -0.006089, 0.0),
    SMB_22 = cms.vdouble(1.378972, -0.736727, 0.0, -0.136466, 0.513675, 0.0),
    OL_2213 = cms.vdouble(0.653836, -0.585322, 0.0, 0.058185, 0.274112, 0.0),
    # Sigle CSC Segments 
    # CSCDet_Id         p1        p2        p3       ep1      ep2      ep3
    #------------------------------------------------------------------------    #
    SME_11 = cms.vdouble(2.74557, -1.047071, 0.0, -12.438189, 7.901606, 0.0),
    SME_13 = cms.vdouble(0.40909, -0.015486, 0.0, 0.50943, 2.419695, 0.0),
    SME_12 = cms.vdouble(-0.115064, 0.666587, 0.0, 1.864145, -0.108712, 0.0),
    SME_32 = cms.vdouble(-0.123675, 0.0724, 0.0, -61.965417, 47.181913, 0.0),
    SME_31 = cms.vdouble(-0.44301, 0.241463, 0.0, 18.782915, -6.103203, 0.0),
    SME_42 = cms.vdouble(-0.123675, 0.0724, 0.0, -61.965417, 47.181913, 0.0),
    # OL Parameters 
    # Det_Layers        p1        p2        p3       ep1      ep2      ep3
    #------------------------------------------------------------------------    #
    OL_1213 = cms.vdouble(1.02946, -0.815312, 0.0, 0.383779, -0.122652, 0.0),
    DT_13 = cms.vdouble(0.308217, 0.115029, -0.189501, 0.247424, -0.259089, 0.254165),
    # DT Parameters 
    # Det_Stations          p1        p2        p3       ep1      ep2      ep3
    #------------------------------------------------------------------------    #
    DT_12 = cms.vdouble(0.18377, 0.072535, -0.106303, 0.214783, -0.260733, 0.251975),
    DT_14 = cms.vdouble(0.388423, 0.068698, -0.145925, 0.159515, 0.124299, -0.133269),
    OL_1232 = cms.vdouble(0.162344, 0.004229, 0.0, 0.435151, 0.021102, 0.0),
    CSC_23 = cms.vdouble(-0.096102, 0.123296, -0.029944, 26.53004, -30.425446, 8.432029),
    CSC_24 = cms.vdouble(-0.291634, 0.287144, -0.061892, 24.535639, -20.958264, 4.69219),
    CSC_03 = cms.vdouble(0.333428, 0.107124, -0.076661, 0.828685, -0.809356, 0.284652),
    SMB_31 = cms.vdouble(0.472501, -0.141958, 0.0, 1.011733, -0.609072, 0.0),
    # CSC Parameters 
    # Det_Stations      p1        p2        p3       ep1      ep2      ep3
    #------------------------------------------------------------------------    #
    CSC_01 = cms.vdouble(0.164247, 0.003469, 0.0, 0.194849, 0.001297, 0.0),
    SMB_32 = cms.vdouble(0.63441, -0.384632, 0.0, 1.626055, -0.791262, 0.0),
    SMB_30 = cms.vdouble(0.399607, 0.204044, 0.0, 0.654936, -0.144121, 0.0),
    OL_2222 = cms.vdouble(0.093337, 0.010211, 0.0, 0.507105, -0.012159, 0.0),
    # Sigle DT Segments 
    # DTDet_Id          p1        p2        p3       ep1      ep2      ep3
    #------------------------------------------------------------------------    #
    SMB_10 = cms.vdouble(1.238604, 0.077387, 0.0, 0.18914, -0.019523, 0.0),
    SMB_11 = cms.vdouble(1.283128, -0.022048, 0.0, 0.113762, 0.036688, 0.0),
    SMB_12 = cms.vdouble(2.080734, -1.0151, 0.0, -0.008602, 0.184733, 0.0),
    DT_23 = cms.vdouble(0.126967, 0.034511, -0.072707, 0.346017, -0.441869, 0.477679),
    DT_24 = cms.vdouble(0.189527, 0.037328, -0.088523, 0.251936, 0.032411, 0.010984),
    SME_21 = cms.vdouble(0.427743, -0.044944, 0.0, -1.552391, 4.823953, 0.0),
    SME_22 = cms.vdouble(-0.583124, 0.605138, 0.0, 39.469726, -24.276153, 0.0),
    CSC_34 = cms.vdouble(-0.203881, 0.185196, -0.040454, -157.042998, 164.144186, -40.345536),
    CSC_02 = cms.vdouble(0.715999, -0.300442, 0.022045, 0.687261, -0.687633, 0.246137),
    SME_41 = cms.vdouble(0.047325, -0.016319, 0.0, 39.982744, -19.953719, 0.0),
    DT_34 = cms.vdouble(0.049146, -0.003494, -0.010099, 0.672095, 0.36459, -0.304346),
    CSC_14 = cms.vdouble(0.590293, -0.15007, -0.011723, 0.964342, -0.858203, 0.289029),
    OL_1222 = cms.vdouble(0.221555, 0.009098, 0.0, 0.340439, -0.035132, 0.0),
    CSC_13 = cms.vdouble(-0.526928, 0.787257, -0.233089, -11.341927, 17.836639, -6.706424),
    CSC_12 = cms.vdouble(-0.39523, 0.601674, -0.177565, 6.353887, -8.25429, 2.804363)
)


