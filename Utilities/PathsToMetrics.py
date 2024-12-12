#!/usr/bin/env python
import Utilities

PathToMetricValues = {
    #Phantom A and B
    1:"objectLengthAccuracyResults.reconstructedObjectLength",
    
    #Phantom A
    #2-9 CTN vs Path Length
    2:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(7.0, {}).normalized_ctn",
    3:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(11.0, {}).normalized_ctn",
    4:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(15.0, {}).normalized_ctn",
    5:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(19.0, {}).normalized_ctn",
    6:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(23.0, {}).normalized_ctn",
    7:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(27.0, {}).normalized_ctn",
    8:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(31.0, {}).normalized_ctn",
    9:"ctnVsPathLengthResults.GetCTNVsPathLengthPoint(35.0, {}).normalized_ctn",
    
    #10-25 CTN Uniformity
    10:"ctnUniformityResults.GetShieldingMaterialResult('CTRL').meanCTN",
    11:"ctnUniformityResults.GetShieldingMaterialResult('CTRL').stdCTN",
    
    12:"ctnUniformityResults.GetShieldingMaterialResult('Al').meanCTN",
    13:"ctnUniformityResults.GetShieldingMaterialResult('Al').stdCTN",
    14:"ctnUniformityResults.GetShieldingMaterialResult('Al').ratioMeanCTN",
    15:"ctnUniformityResults.GetShieldingMaterialResult('Al').ratioSTDCTN",
    
    16:"ctnUniformityResults.GetShieldingMaterialResult('Cu').meanCTN",
    17:"ctnUniformityResults.GetShieldingMaterialResult('Cu').stdCTN",
    18:"ctnUniformityResults.GetShieldingMaterialResult('Cu').ratioMeanCTN",
    19:"ctnUniformityResults.GetShieldingMaterialResult('Cu').ratioSTDCTN",
    
    20:"ctnUniformityResults.GetShieldingMaterialResult('Pb').meanCTN",
    21:"ctnUniformityResults.GetShieldingMaterialResult('Pb').stdCTN",
    22:"ctnUniformityResults.GetShieldingMaterialResult('Pb').ratioMeanCTN",
    23:"ctnUniformityResults.GetShieldingMaterialResult('Pb').ratioSTDCTN",
    
    24:"ctnUniformityResults.meanCTNAl",
    25:"ctnUniformityResults.stdCTNAl",
    
    #26-31 CTN_Line_Measurement_of_Pin_Area
    26:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_Pin_Area').meanCTN",
    27:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area').stdCTN",
    28:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_Pin_Area').minCTN",
    29:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_Pin_Area').maxCTN",    
    30:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area').percentSTDCTN",    
    31:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area').percentCTNRange", 
    
    #32-37 CTN_Region_Measurement_of_Pin_Area
    32:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_Pin_Area').meanCTN",
    33:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area').stdCTN",
    34:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_Pin_Area').minCTN",
    35:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_Pin_Area').maxCTN",    
    36:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area').percentSTDCTN",    
    37:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area').percentCTNRange",    
    
    #38-43 CTN_Line_Measurement_of_CTRL_Area
    38:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Line_Measurement_of_CTRL_Area').meanCTN",
    39:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Line_Measurement_of_CTRL_Area').stdCTN",
    40:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Line_Measurement_of_CTRL_Area').minCTN",
    41:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Line_Measurement_of_CTRL_Area').maxCTN",    
    42:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area').percentSTDCTN",    
    43:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Line_Measurement_of_CTRL_Area').percentCTNRange",    
    
    #44-49 CTN_Region_Measurement_of_CTRL_Area
    44:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Mean_CTN_Region_Measurement_of_CTRL_Area').meanCTN",
    45:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_CTN_Region_Measurement_of_CTRL_Area').stdCTN",
    46:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Minimum_CTN_Region_Measurement_of_CTRL_Area').minCTN",
    47:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Maximum_CTN_Region_Measurement_of_CTRL_Area').maxCTN",    
    48:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area').percentSTDCTN",    
    49:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Absolute_Range_Relative_to_Mean_CTN_Region_Measurement_of_CTRL_Area').percentCTNRange",    
    
    #50-55 CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL
    50:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').meanCTN",    
    51:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').stdCTN",
    52:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Minimum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').minCTN",
    53:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Maximum_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').maxCTN",    
    54:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').percentSTDCTN",    
    55:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Range_Relative_to_Mean_CTN_Line_Measurement_of_Pin_Area_Relative_to_CTRL').percentCTNRange",    
    
    #56-61 CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL
    56:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').meanCTN",    
    57:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').stdCTN",
    58:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Minimum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').minCTN",
    59:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Maximum_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').maxCTN",    
    60:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Standard_Deviation_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').percentSTDCTN",    
    61:"streakArtifactResults.GetStreakResult('Streak_Artifacts_Range_Relative_to_Mean_CTN_Region_Measurement_of_Pin_Area_Relative_to_CTRL').percentCTNRange", 
    
    #Phantom A and B
    62:"objectLengthAccuracyResults.reconstructedObjectLength",
    
    #Phantom B
    #63-64 SSP
    63:"sspResults.GetSSPPoint(0.5, {}).mtf",
    64:"sspResults.GetSSPPoint(1.0, {}).mtf",
    
    #65-72 NEQ
    65:"neqResults.GetNEQPoint(0.5).mtf",
    66:"neqResults.GetNEQPoint(0.5).neqMethod1",
    67:"neqResults.GetNEQPoint(0.5).neqMethod2",
    68:"neqResults.GetNEQPoint(0.5).neqMethod3",
    69:"neqResults.GetNEQPoint(1.0).mtf",
    70:"neqResults.GetNEQPoint(1.0).neqMethod1",
    71:"neqResults.GetNEQPoint(1.0).neqMethod2",
    72:"neqResults.GetNEQPoint(1.0).neqMethod3",
    
    #73-76 CTN Consistency
    73:"ctnConsistencyResults.medianOfMeans",
    74:"ctnConsistencyResults.standardDeviationOfMeans",
    75:"ctnConsistencyResults.medianOfStandardDeviations",
    76:"ctnConsistencyResults.standardDeviationOfStandardDeviations"
}