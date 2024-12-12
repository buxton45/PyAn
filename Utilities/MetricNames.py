#!/usr/bin/env python
from enum import Enum
import Utilities


#class MetricName:
#    






# 61 elements
MetricNames_Xml_PhantomA = (
    "ReconstructedObjectLength",
    "NormalizedCTN_7cm",
    "NormalizedCTN_11cm",
    "NormalizedCTN_15cm",
    "NormalizedCTN_19cm",
    "NormalizedCTN_23cm",
    "NormalizedCTN_27cm",
    "NormalizedCTN_31cm",
    "NormalizedCTN_35cm",
    "CTNUniformity_CTRL_MeanCTN",
    "CTNUniformity_CTRL_STDCTN",
    "CTNUniformity_Al_MeanCTN",
    "CTNUniformity_Al_STDCTN",
    "CTNUniformity_Al_RatioMeanCTN",
    "CTNUniformity_Al_RatioSTDCTN",
    "CTNUniformity_Cu_MeanCTN",
    "CTNUniformity_Cu_STDCTN",
    "CTNUniformity_Cu_RatioMeanCTN",
    "CTNUniformity_Cu_RatioSTDCTN",
    "CTNUniformity_Pb_MeanCTN",
    "CTNUniformity_Pb_STDCTN",
    "CTNUniformity_Pb_RatioMeanCTN",
    "CTNUniformity_Pb_RatioSTDCTN",
    "CTNUniformity_AlMeanCTN",
    "CTNUniformity_AlSTDCTN",
    "Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Minimum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Maximum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Minimum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Maximum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL"
)


# 61 elements
MetricNames_Long_PhantomA = (
    "Reconstructed Object Length",
    "CTN Vs Path Length (7 cm)",
    "CTN Vs Path Length (11 cm)",
    "CTN Vs Path Length (15 cm)",
    "CTN Vs Path Length (19 cm)",
    "CTN Vs Path Length (23 cm)",
    "CTN Vs Path Length (27 cm)",
    "CTN Vs Path Length (31 cm)",
    "CTN Vs Path Length (35 cm)",
    "CTN Value Uniformity Acetal CTRL Absolute Mean",
    "CTN Value Uniformity Acetal CTRL Absolute Std",
    "CTN Value Uniformity Acetal Al Absolute Mean",
    "CTN Value Uniformity Acetal Al Absolute Std",
    "CTN Value Uniformity Acetal Al Relative to CTRL Mean",
    "CTN Value Uniformity Acetal Al Relative to CTRL Std",
    "CTN Value Uniformity Acetal Cu Absolute Mean",
    "CTN Value Uniformity Acetal Cu Absolute Std",
    "CTN Value Uniformity Acetal Cu Relative to CTRL Mean",
    "CTN Value Uniformity Acetal Cu Relative to CTRL Std",
    "CTN Value Uniformity Acetal Pb Absolute Mean",
    "CTN Value Uniformity Acetal Pb Absolute Std",
    "CTN Value Uniformity Acetal Pb Relative to CTRL Mean",
    "CTN Value Uniformity Acetal Pb Relative to CTRL Std",
    "CTN Uniformity Mean CTN of Al Band",
    "CTN Uniformity STD CTN of Al Band",
    "Streak Artifacts Absolute Mean CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Minimum CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Maximum CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Mean CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Minimum CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Maximum CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Mean CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Minimum CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Maximum CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Mean CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Minimum CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Maximum CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Region Measurement of CTRL Area",
    "Streak Artifacts Mean CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Minimum CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Maximum CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation Relative to Mean CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Range Relative to Mean CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Mean CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Minimum CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Maximum CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation Relative to Mean CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Range Relative to Mean CTN Region Measurement of Pin Area Relative to CTRL"
)

PathToMetricValues_PhantomA = (
    "objectLengthAccuracyResults.reconstructedObjectLength",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(7.0, {}).normalized_ctn",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(11.0, {}).normalized_ctn",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(15.0, {}).normalized_ctn",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(19.0, {}).normalized_ctn",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(23.0, {}).normalized_ctn",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(27.0, {}).normalized_ctn",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(31.0, {}).normalized_ctn",
    "ctnVsPathLengthResults.GetCTNVsPathLengthPoint(35.0, {}).normalized_ctn",
    "ctnUniformityResults.GetShieldingMaterialResult('CTRL').meanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('CTRL').stdCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Al').meanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Al').stdCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Al').ratioMeanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Al').ratioSTDCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').meanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').stdCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').ratioMeanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').ratioSTDCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Pb').meanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Pb').stdCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Pb').ratioMeanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Pb').ratioSTDCTN",
    "ctnUniformityResults.meanCTNAl",
    "ctnUniformityResults.stdCTNAl",    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_Pin_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_Pin_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_Pin_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_Pin_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_Pin_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_Pin_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_CTRL_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_CTRL_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_CTRL_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_CTRL_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_CTRL_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_CTRL_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_CTRL_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_CTRL_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Minimum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Maximum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Minimum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Maximum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').percentCTNRange"
)

PathToMetricValues_PhantomA_inSAT = (
    "objectLengthAccuracyResults.reconstructedObjectLength",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "ctnUniformityResults.GetShieldingMaterialResult('CTRL').meanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('CTRL').stdCTN",
    "",
    "",
    "",
    "",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').meanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').stdCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').ratioMeanCTN",
    "ctnUniformityResults.GetShieldingMaterialResult('Cu').ratioSTDCTN",
    "",
    "",
    "",
    "",
    "",
    "",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_Pin_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_Pin_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_Pin_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_Pin_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_Pin_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_Pin_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_CTRL_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_CTRL_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_CTRL_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_CTRL_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_CTRL_Area').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_CTRL_Area').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_CTRL_Area').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_CTRL_Area').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Minimum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Maximum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').percentCTNRange",
    
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').meanCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').stdCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Minimum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').minCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Maximum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').maxCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').percentSTDCTN",
    "streakArtifactResults.GetStreakResult('Streak_Artifacts_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').percentCTNRange"
)


# 15 elements
MetricNames_Xml_PhantomB = (
    "ReconstructedObjectLength",
    "SSPModule_MTF_0p5",
    "SSPModule_MTF_1p0",
    "NEQModule_MTF_0p5",
    "NEQModule_NEQ1_0p5",
    "NEQModule_NEQ2_0p5",
    "NEQModule_NEQ3_0p5",
    "NEQModule_MTF_1p0",
    "NEQModule_NEQ1_1p0",
    "NEQModule_NEQ2_1p0",
    "NEQModule_NEQ3_1p0",
    "CTNConsistency_MedianMeanCTN",
    "CTNConsistency_STDMeanCTN",
    "CTNConsistency_MedianSTDCTN",
    "CTNConsistency_STDSTDCTN"
)

# 15 elements
MetricNames_Long_PhantomB = (
    "Reconstructed Object Length",
    "Modulation Transfer Function of the Acetyl Slice Sensitivity test object at 0.5/cm",
    "Modulation Transfer Function of the Acetyl Slice Sensitivity test object at 1.0/cm",
    "Modulation Transfer Function at 0.5/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 1 at 0.5/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 2 at 0.5/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 3 at 0.5/cm using the Acetyl cylinder test object",
    "Modulation Transfer Function at 1.0/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 1 at 1.0/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 2 at 1.0/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 3 at 1.0/cm using the Acetyl cylinder test object",
    "CTN Consistency Median of Mean (CTN)",
    "CTN Consistency STD of Mean (CTN)",
    "CTN Consistency Median of STD (CTN)",
    "CTN Consistency STD of STD (CTN)"
)

PathToMetricValues_PhantomB = (
    "objectLengthAccuracyResults.reconstructedObjectLength",
    "sspResults.GetSSPPoint(0.5, {}).mtf",
    "sspResults.GetSSPPoint(1.0, {}).mtf",
    "neqResults.GetNEQPoint(0.5).mtf",
    "neqResults.GetNEQPoint(0.5).neqMethod1",
    "neqResults.GetNEQPoint(0.5).neqMethod2",
    "neqResults.GetNEQPoint(0.5).neqMethod3",
    "neqResults.GetNEQPoint(1.0).mtf",
    "neqResults.GetNEQPoint(1.0).neqMethod1",
    "neqResults.GetNEQPoint(1.0).neqMethod2",
    "neqResults.GetNEQPoint(1.0).neqMethod3",
    "ctnConsistencyResults.medianOfMeans",
    "ctnConsistencyResults.standardDeviationOfMeans",
    "ctnConsistencyResults.medianOfStandardDeviations",
    "ctnConsistencyResults.standardDeviationOfStandardDeviations"
)


MetricNames_Xml_SAT = (
    "ReconstructedObjectLength",
    
    "CTNUniformity_CTRL_MeanCTN",
    "CTNUniformity_CTRL_STDCTN",
    
    "CTNUniformity_Cu_MeanCTN",
    "CTNUniformity_Cu_STDCTN",
    "CTNUniformity_Cu_RatioMeanCTN",
    "CTNUniformity_Cu_RatioSTDCTN",

    "Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area",
    "Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area",
    "Streak_Artifacts_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Minimum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Maximum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Minimum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Maximum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL",
    "Streak_Artifacts_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL", 
    
    "SSPModule_MTF_0p5",
    "SSPModule_MTF_1p0",
    
    "NEQModule_MTF_0p5",
    "NEQModule_NEQ1_0p5",
    "NEQModule_NEQ2_0p5",
    "NEQModule_NEQ3_0p5",
    
    "NEQModule_MTF_1p0",
    "NEQModule_NEQ1_1p0",
    "NEQModule_NEQ2_1p0",
    "NEQModule_NEQ3_1p0",
    
    "CTNConsistency_MedianMeanCTN",
    "CTNConsistency_STDMeanCTN",
    "CTNConsistency_MedianSTDCTN",
    "CTNConsistency_STDSTDCTN"
)


MetricNames_Long_SAT = (
    "Reconstructed Object Length",
    
    "CTN Value Uniformity Acetal CTRL Absolute Mean",
    "CTN Value Uniformity Acetal CTRL Absolute Std",
    
    "CTN Value Uniformity Acetal Cu Absolute Mean",
    "CTN Value Uniformity Acetal Cu Absolute Std",
    "CTN Value Uniformity Acetal Cu Relative to CTRL Mean",
    "CTN Value Uniformity Acetal Cu Relative to CTRL Std",

    "Streak Artifacts Absolute Mean CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Minimum CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Maximum CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Line Measurement of Pin Area",
    "Streak Artifacts Absolute Mean CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Minimum CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Maximum CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Region Measurement of Pin Area",
    "Streak Artifacts Absolute Mean CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Minimum CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Maximum CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Line Measurement of CTRL Area",
    "Streak Artifacts Absolute Mean CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Minimum CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Maximum CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Standard Deviation Relative to Mean CTN Region Measurement of CTRL Area",
    "Streak Artifacts Absolute Range Relative to Mean CTN Region Measurement of CTRL Area",
    "Streak Artifacts Mean CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Minimum CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Maximum CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation Relative to Mean CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Range Relative to Mean CTN Line Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Mean CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Minimum CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Maximum CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Standard Deviation Relative to Mean CTN Region Measurement of Pin Area Relative to CTRL",
    "Streak Artifacts Range Relative to Mean CTN Region Measurement of Pin Area Relative to CTRL", 
    
    "Modulation Transfer Function of the Acetyl Slice Sensitivity test object at 0.5/cm",
    "Modulation Transfer Function of the Acetyl Slice Sensitivity test object at 1.0/cm",
    
    "Modulation Transfer Function at 0.5/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 1 at 0.5/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 2 at 0.5/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 3 at 0.5/cm using the Acetyl cylinder test object",
    
    "Modulation Transfer Function at 1.0/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 1 at 1.0/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 2 at 1.0/cm using the Acetyl cylinder test object",
    "Noise Equivalent Quanta Method 3 at 1.0/cm using the Acetyl cylinder test object",
    
    "CTN Consistency Median of Mean (CTN)",
    "CTN Consistency STD of Mean (CTN)",
    "CTN Consistency Median of STD (CTN)",
    "CTN Consistency STD of STD (CTN)"
)

assert(len(PathToMetricValues_PhantomA) == len(MetricNames_Xml_PhantomA) == len(MetricNames_Long_PhantomA))
assert(len(PathToMetricValues_PhantomB) == len(MetricNames_Xml_PhantomB) == len(MetricNames_Long_PhantomB))

MetricNames_PhantomA_dict = dict(zip(MetricNames_Xml_PhantomA, MetricNames_Long_PhantomA))
MetricNames_PhantomB_dict = dict(zip(MetricNames_Xml_PhantomB, MetricNames_Long_PhantomB))

MetricNames_PhantomA_dict_inv = {v:k for k,v in MetricNames_PhantomA_dict.items()}
MetricNames_PhantomB_dict_inv = {v:k for k,v in MetricNames_PhantomB_dict.items()}

PathToMetricValues_PhantomA_Xml_dict = dict(zip(MetricNames_Xml_PhantomA, PathToMetricValues_PhantomA))
PathToMetricValues_PhantomA_Long_dict = dict(zip(MetricNames_Long_PhantomA, PathToMetricValues_PhantomA))

PathToMetricValues_PhantomB_Xml_dict = dict(zip(MetricNames_Xml_PhantomB, PathToMetricValues_PhantomB))
PathToMetricValues_PhantomB_Long_dict = dict(zip(MetricNames_Long_PhantomB, PathToMetricValues_PhantomB))

MisspelledNames = ("Devation",  "Regioin")
CorrectNames = ("Deviation", "Region")


def ReplaceAllMisspelled(aString):
    assert(len(MisspelledNames)==len(CorrectNames))
    tReturnStr = aString
    for i in range(len(MisspelledNames)):
        tReturnStr = tReturnStr.replace(MisspelledNames[i], CorrectNames[i])
    tReplaced = (not(aString==tReturnStr))    
    return tReturnStr, tReplaced

def ReplaceBackAllMisspelled(aString):
    assert(len(MisspelledNames)==len(CorrectNames))
    tReturnStr = aString
    for i in range(len(CorrectNames)):
        tReturnStr = tReturnStr.replace(CorrectNames[i], MisspelledNames[i])
    tReplaced = (not(aString==tReturnStr))    
    return tReturnStr, tReplaced


#def CompStrings_CaseInsensitive(aStr1, aStr2):
#    return aStr1.strip().casefold()==aStr2.strip().casefold()


def FindElementIndexInVecOfStrings(aElementName, aVecOfStrings, aAssertMatch=False):
    tReturnIdx=-1
    try:
        tReturnIdx = aVecOfStrings.index(aElementName)
    except:
        tReturnIdx = -1
        
    if(aAssertMatch):
        if(tReturnIdx < 0):
            print(f"In MetricNames:FindElementIndexInVecOfStrings, cannot find index for aElementName = {aElementName}")
        assert(tReturnIdx > -1)
    return tReturnIdx


def FindElementIndexInVecOfStrings_CaseInsensitive(aElementName, aVecOfStrings, aAssertMatch=False):
    tVecOfStrings = [elem.casefold() for elem in aVecOfStrings]
    tElementName = aElementName.casefold()
    return FindElementIndexInVecOfStrings(tElementName, tVecOfStrings, aAssertMatch)



def GetMetricName_XmlFromLong(aLongName, aAssertMatch=True, aPhantType=None):
    assert((aPhantType==Utilities.PhantomType.kA) or (aPhantType==Utilities.PhantomType.kB) or (aPhantType is None))
    tLongName, tReplaced = ReplaceAllMisspelled(aLongName)
    tMetricNames_dict_inv_AB = [MetricNames_PhantomA_dict_inv, MetricNames_PhantomB_dict_inv]
    
    if(aPhantType is not None):
        tReturnStr = tMetricNames_dict_inv_AB[aPhantType.value].get(tLongName, None)
    else:
        tReturnStr = tMetricNames_dict_inv_AB[Utilities.PhantomType.kA.value].get(tLongName, None)
        if(tReturnStr is None):
            tReturnStr = tMetricNames_dict_inv_AB[Utilities.PhantomType.kB.value].get(tLongName, None)

    if(aAssertMatch):
        if(tReturnStr is None):
            print(f"In MetricNames.GetMetricName_XmlFromLong, cannot find index for aLongName = {aLongName}")
        assert(tReturnStr is not None)
    if(tReturnStr is not None):
        #If portion misspelled in Long, also misspelled in Xml, so replace
        if(tReplaced):
            tReturnStr,_ = ReplaceBackAllMisspelled(tReturnStr)
        return tReturnStr
    else:
        return ''
    
    
    
def GetMetricName_LongFromXml(aXmlName, aAssertMatch=True, aPhantType=None):
    assert((aPhantType==Utilities.PhantomType.kA) or (aPhantType==Utilities.PhantomType.kB) or (aPhantType is None))
    tXmlName, tReplaced = ReplaceAllMisspelled(aXmlName)
    
    tMetricNames_PhantomA_dict = MetricNames_PhantomA_dict.copy()
    tMetricNames_PhantomB_dict = MetricNames_PhantomB_dict.copy()

    # Include correction version of Streak_Artifacts__Standard_Deviation_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL
    # with only on _ instead of __
    toFix = [x for x in tMetricNames_PhantomA_dict if x.find('__')>-1]
    for found in toFix:
        tMetricNames_PhantomA_dict[found.replace('__', '_')] = tMetricNames_PhantomA_dict[found]
    
    tMetricNames_dict_AB = [tMetricNames_PhantomA_dict, tMetricNames_PhantomB_dict]
    
    if(aPhantType is not None):
        tReturnStr = tMetricNames_dict_AB[aPhantType.value].get(tXmlName, None)
    else:
        tReturnStr = tMetricNames_dict_AB[Utilities.PhantomType.kA.value].get(tXmlName, None)
        if(tReturnStr is None):
            tReturnStr = tMetricNames_dict_AB[Utilities.PhantomType.kB.value].get(tXmlName, None)

    if(aAssertMatch):
        if(tReturnStr is None):
            print(f"In MetricNames.GetMetricName_LongFromXml, cannot find index for aXmlName = {aXmlName}")
        assert(tReturnStr is not None)
    if(tReturnStr is not None):
        #If portion misspelled in Xml, also misspelled in Long, so replace
        if(tReplaced):
            tReturnStr,_ = ReplaceBackAllMisspelled(tReturnStr)
        return tReturnStr
    else:
        return ''    
    

def GetPathToMetricValueFromName(aMetricName, aAssertFound=True):
    #Doesn't really matter which dict we find it in
    tDictsToSearch = [PathToMetricValues_PhantomA_Xml_dict, PathToMetricValues_PhantomB_Xml_dict,
                      PathToMetricValues_PhantomA_Long_dict, PathToMetricValues_PhantomB_Long_dict]

    #NOTE: Some values in the dicts are empty strings, which are NOT the same as None
    #  Using dict.get('whatever', None), a return value of None means the key wasn't found
    #  wherease a return value of '' (an empty string) means the key was found!
    #  If I have aPath = '', aPath is None will return False
    #  If I have aPath = None, aPath is None will return True
    #  Whereas, if aPath='' or aPath=None, in either case not(aPath) return True!
    tReturnPath = None
    for aDict in tDictsToSearch:
        tReturnPath = aDict.get(aMetricName, None)
        if(tReturnPath is not None):
            break            
    if(aAssertFound):
        assert(tReturnPath is not None)
    return tReturnPath
    
    
def GetMetricNamesOfInterest():
    assert(len(PathToMetricValues_PhantomA_inSAT)==len(MetricNames_Xml_PhantomA))    
    tMetricsOfInterest = [MetricNames_Xml_PhantomA[i] for i in range(len(PathToMetricValues_PhantomA_inSAT)) 
                         if PathToMetricValues_PhantomA_inSAT[i]]

    #Skip first element in MetricNames_Xml_PhantomB because alredy obtained ReconstructedObjectLength from A
    assert(len(PathToMetricValues_PhantomB)==len(MetricNames_Xml_PhantomB))    
    tMetricsOfInterest.extend([MetricNames_Xml_PhantomB[i] for i in range(1, len(PathToMetricValues_PhantomB)) 
                               if PathToMetricValues_PhantomB[i]])
    return tMetricsOfInterest

def GetMetricNamesOfInterestA():
    assert(len(PathToMetricValues_PhantomA)==len(MetricNames_Xml_PhantomA))    
    tMetricsOfInterest = [MetricNames_Xml_PhantomA[i] for i in range(len(PathToMetricValues_PhantomA)) 
                         if PathToMetricValues_PhantomA[i]]
    return tMetricsOfInterest
    
    
def GetMetricNamesOfInterestB():
    assert(len(PathToMetricValues_PhantomB)==len(MetricNames_Xml_PhantomB))    
    tMetricsOfInterest = [MetricNames_Xml_PhantomB[i] for i in range(len(PathToMetricValues_PhantomB)) 
                         if PathToMetricValues_PhantomB[i]]
    return tMetricsOfInterest
    
def GetMetricNamesOfInterest_Long():
    assert(len(PathToMetricValues_PhantomA)==len(MetricNames_Long_PhantomA))    
    tMetricsOfInterest = [MetricNames_Long_PhantomA[i] for i in range(len(PathToMetricValues_PhantomA)) 
                         if PathToMetricValues_PhantomA[i]]

    #Skip first element in MetricNames_Long_PhantomB because alredy obtained ReconstructedObjectLength from A
    assert(len(PathToMetricValues_PhantomB)==len(MetricNames_Long_PhantomB))    
    tMetricsOfInterest.extend([MetricNames_Long_PhantomB[i] for i in range(1, len(PathToMetricValues_PhantomB)) 
                               if PathToMetricValues_PhantomB[i]])
    return tMetricsOfInterest