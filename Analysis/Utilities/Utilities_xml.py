#!/usr/bin/env python

import os
import sys
import glob
import re
from pathlib import Path

import xml.etree.ElementTree as ET
import xml.parsers.expat
from xml.dom import minidom

import Utilities

#--------------------------------------------------------------------
def import_xml_etree(file_path):
    try:
        tree = ET.parse(file_path)
    except (EnvironmentError,
           xml.parsers.expat.ExpatError) as err:
        print("{0}: import error: {1}".format(
        os.path.basename(sys.argv[0]), err))
        assert(0)
    return tree

def get_xml_etree_root(file_path):
    tree = import_xml_etree(file_path)
    return tree.getroot()    
    
def element_exists(aNode, aElName):
    assert(isinstance(aNode, ET.Element))
    if(aNode.find(aElName) is None):
        return False
    else:
        return True

def element_exists_and_is_float(aNode, aElName):
    assert(isinstance(aNode, ET.Element))
    if(aNode.find(aElName) is None):
        return False
    try:
        float(aNode.find(aElName).text)
        return True
    except:
        return False
    return False

def element_exists_and_is_int(aNode, aElName):
    #If it's a float, it can also be an int
    return element_exists_and_is_float(aNode, aElName)

def convert_element_to_float(aNode, aElName, aUnset=-9999.0):
    tReturnVal = float(aNode.find(aElName).text) if element_exists_and_is_float(aNode, aElName) else aUnset
    return tReturnVal

def convert_element_to_int(aNode, aElName, aUnset=-9999):
    #Converting, for ex. str = '153.000' directly to int doesn't work
    #Need to instead do int(float('153.000'))    
    tReturnVal = int(float(aNode.find(aElName).text)) if element_exists_and_is_int(aNode, aElName) else aUnset
    return tReturnVal

def convert_element_to_str(aNode, aElName, aUnset=''):
    tReturnVal = aNode.find(aElName).text if element_exists(aNode, aElName) else aUnset
    return tReturnVal
#--------------------------------------------------------------------
def combine_nist_xmls(file_paths):
    xml_data = None
    for file_path in file_paths:
        data = get_xml_etree_root(file_path)
        if xml_data is None:
            xml_data = data
        else:
            xml_data.extend(data)
    assert(xml_data is not None)
    return ET.ElementTree(xml_data)

def combine_nist_xmls_and_output(file_paths, save_path):
    xml_data = combine_nist_xmls(file_paths)
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent)
    xml_str = minidom.parseString(ET.tostring(xml_data.getroot())).toprettyxml(encoding="utf-8", indent='\t ')
    #Take out all the newlines which are added for whatever reason...
    xml_str = b'\n'.join([s for s in xml_str.splitlines() if s.strip()])
    with open(save_path, 'wb') as f:
        f.write(xml_str)

#--------------------------------------------------------------------
def remove_nist_result_nodes_by_id_and_save(xml_path_orig, xml_path_save, ids_to_remove):
    #ids_to_remove should either be single integer or a list of integers
    if isinstance(ids_to_remove, int):
        ids_to_remove = [ids_to_remove]
    assert(isinstance(ids_to_remove, list))
    for idx in ids_to_remove:
        assert(isinstance(idx, int))
    #-----
    tree = import_xml_etree(xml_path_orig)
    root = tree.getroot()
    n_results_orig = len(root.findall('NISTResult'))
    #-----
    for i,child in enumerate(list(root)):
        if i in ids_to_remove:
            root.remove(child)
    n_results_final = len(root.findall('NISTResult'))
    assert(n_results_orig-n_results_final==len(ids_to_remove))
    #-----
    tree.write(xml_path_save)

def remove_nist_result_nodes_by_fileName_and_save(xml_path_orig, xml_path_save, fileNames_to_remove):
    #fileNames_to_remove should either be single string or a list of strings
    if isinstance(fileNames_to_remove, str):
        fileNames_to_remove = [fileNames_to_remove]
    assert(isinstance(fileNames_to_remove, list))
    for fileName in fileNames_to_remove:
        assert(isinstance(fileName, str))
    #-----
    tree = import_xml_etree(xml_path_orig)
    root = tree.getroot()
    n_results_orig = len(root.findall('NISTResult'))
    #-----
    for i,child in enumerate(list(root)):
        fileName = convert_element_to_str(child, 'FileName')
        if fileName in fileNames_to_remove:
            root.remove(child)
    n_results_final = len(root.findall('NISTResult'))
    assert(n_results_orig-n_results_final==len(fileNames_to_remove))
    #-----
    tree.write(xml_path_save)       
#--------------------------------------------------------------------
def is_spca_xml(xml_path):
    root = get_xml_etree_root(xml_path)
    if root.tag == 'ANSIComparisonDB':
        return True
    else:
        return False
    
def is_nist_xml(xml_path):
    root = get_xml_etree_root(xml_path)
    if root.tag == 'NISTPhantomDatabase':
        return True
    else:
        return False
#--------------------------------------------------------------------
def find_all_nist_paths(base_dir, glob_pattern=r'**\Results\**\*.xml', 
                        regex_pattern='NIST ?Results?', regex_ignore_case=False, recursive=True, 
                        tags_to_ignore=['error', 'temporary']):
    # regex_pattern = 'NIST ?Results?' will find e.g. 'NISTResults', 'NISTResult', 'NIST Results', 'NIST Result'
    # regex_pattern='(NIST ?Results?)|(Results?)' will find those above plus 'Results' and 'Result'
    # Some other common patterns:
    # r'**\6040CTiX\**\*.xml'
    # r'**\Results\**\NISTResults\**\*.xml'
    if base_dir.find('milky-way') and glob_pattern==r'**\Results\**\*.xml':
        glob_pattern=r'*\Results\**\*.xml'
    nist_paths = Utilities.find_all_paths(base_dir, glob_pattern, regex_pattern, regex_ignore_case, recursive)
    #nist_paths = Utilities.remove_tagged_from_list(nist_paths, tags_to_ignore)
    nist_paths = Utilities.remove_tagged_from_list_of_paths(nist_paths, tags_to_ignore, base_dir=base_dir)
    nist_paths = [x for x in nist_paths if is_nist_xml(x)]
    return nist_paths

def find_all_spca_paths(base_dir, glob_pattern=r'**\Results\**\*.xml', regex_pattern='SPCA ?Results?', regex_ignore_case=False, recursive=True):
    # See find_all_nist_paths for more info
    if base_dir.find('milky-way') and glob_pattern==r'**\Results\**\*.xml':
        glob_pattern=r'*\Results\**\*.xml'
    spca_paths = Utilities.find_all_paths(base_dir, glob_pattern, regex_pattern, regex_ignore_case, recursive)
    spca_paths = [x for x in spca_paths if is_spca_xml(x)]
    return spca_paths

#-------------------------------------------------------------------- 
def get_spca_result_from_xml(xml_path, phantom_type):
    node = 'PhantomResult'
    if phantom_type==Utilities.PhantomType.kA:
        node = 'PhantomAResult'
    if phantom_type==Utilities.PhantomType.kB:
        node = 'PhantomBResult'
        
    root = get_xml_etree_root(xml_path)
    t_sqr_val = convert_element_to_float(root,f'{node}/TSqrValue', None)
    ryg_condition = convert_element_to_str(root,f'{node}/RYGCondition', None)
    assert(t_sqr_val is not None and ryg_condition is not None)
    ryg_condition = Utilities.str_to_color_type(ryg_condition)
    return t_sqr_val, ryg_condition
    
def get_test_and_base_datasets_from_spca_xml(xml_path, phantom_type, full_path=False):
    node = 'PhantomResult'
    if phantom_type==Utilities.PhantomType.kA:
        node = 'PhantomAResult'
    if phantom_type==Utilities.PhantomType.kB:
        node = 'PhantomBResult'
        
    root = get_xml_etree_root(xml_path)
    dataset_test = convert_element_to_str(root,f'{node}/TestDataset/DatasetName', None)
    dataset_base = convert_element_to_str(root,f'{node}/BaselineDataset/DatasetName', None)
    assert(dataset_test is not None and dataset_base is not None)
    if not full_path:
        dataset_test = os.path.basename(dataset_test)
        dataset_base = os.path.basename(dataset_base)
    return dataset_test, dataset_base
    
#--------------------------------------------------------------------
def get_unique_EDSNames_in_nist(nist_path):
    assert(is_nist_xml(nist_path))
    tree = import_xml_etree(nist_path)
    eds_names = tree.findall('.//EDSName')
    unique_eds_names = []
    for name in eds_names:
        if name.text not in unique_eds_names:
            unique_eds_names.append(name.text)
    return unique_eds_names

def change_EDSNames_in_nist_path(nist_path, new_EDSName, old_EDSName=None, save_path=None):
    # if save_path==None, nist_path will be overwritten with new xml
    # This is usually what is wanted
    # If old_EDSName is not None, then name will be changed only if original name matches old_EDSName
    assert(is_nist_xml(nist_path))
    if save_path is None:
        save_path=nist_path
    tree = import_xml_etree(nist_path)
    eds_names = tree.findall('.//EDSName')
    for name in eds_names:
        if old_EDSName is None:
            name.text = new_EDSName
        else:
            if name.text==old_EDSName:
                name.text = new_EDSName
    tree.write(save_path)
    
def change_EDSNames_in_nist_paths(nist_paths, new_EDSName, old_EDSName=None, save_path=None):
    for nist_path in nist_paths:
        change_EDSNames_in_nist_path(nist_path, new_EDSName, old_EDSName, save_path)
        
        
        
#------------------------------------------------------------------------------------------------------------
def GetBaselineDatasetNameFromSpca(aSpcaFileName):
    root = ET.parse(aSpcaFileName)
    tBLs = root.findall('.//*BaselineDataset/DatasetName')
    #Each should use the same baseline...
    tBL = tBLs[0].text
    for BL in tBLs:
        assert(BL.text==tBL)
    return tBL
    
def GetBaselineDirPathFromName(aBLName, data_storage_base_dir = r'C:\Users\BUXTONJ\Documents\Analysis\LocalDataStorage'):
    if(aBLName.find('6040CTiX_SAT_FSB_Baseline_022020_1500_v204.xml')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6040CTiX\SAT\6040CTiX_SAT_FSB_Baseline_022020_1500_v204')
    elif(aBLName.find('6040CTiX_SAT_Baseline_20190926_1500_v189')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6040CTiX\SAT\6040CTiX_SAT_Baseline_20190926_1500_v189')
        
    elif(aBLName.find('6040CTiX_FAT_Baseline_20190926_1400_v188')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6040CTiX\FAT\6040CTiX_FAT_Baseline_20190926_1400_v188')
        
    elif(aBLName.find('6700_FAT_Baseline_102318_1200_v153')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700\FAT\6700_FAT_Baseline_102318_1200_v153')
        
    elif(aBLName.find('6700_IQPSAT_Baseline_050718_1002_v94')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700\SAT\6700_IQPSAT_Baseline_050718_1002_v94')
    elif(aBLName.find('6700_IQPSAT_Baseline_073018_0205_v139')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700\SAT\6700_IQPSAT_Baseline_073018_0205_v139')
    elif(aBLName.find('6700_SAT_Baseline_102318_0722_v157')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700\SAT\6700_SAT_Baseline_102318_0722_v157')
        
    elif(aBLName.find('6700ES_IQPSAT_Baseline_051518_0315_v109')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700ES\SAT\6700ES_IQPSAT_Baseline_051518_0315_v109')
    elif(aBLName.find('6700ES_IQPSAT_Baseline_073118_0432_v141')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700ES\SAT\6700ES_IQPSAT_Baseline_073118_0432_v141')
    elif(aBLName.find('6700ES_SAT_Baseline_102318_0717_v158')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700ES\SAT\6700ES_SAT_Baseline_102318_0717_v158')
    elif(aBLName.find('6700ES_SAT_Baseline_20190719_1700_v183')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700ES\SAT\6700ES_SAT_Baseline_20190719_1700_v183')
        
    elif(aBLName.find('6700ES_IQSFAT_Baseline_060118_0910_v120')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700ES\FAT\6700ES_IQSFAT_Baseline_060118_0910_v120')
    elif(aBLName.find('6700ES_IQSFAT_Baseline_073118_0427_v140')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700ES\FAT\6700ES_IQSFAT_Baseline_073118_0427_v140')
    elif(aBLName.find('6700ES_FAT_Baseline_102318_1201_v154')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\6700ES\FAT\6700ES_FAT_Baseline_102318_1201_v154')
        
    elif(aBLName.find('CT80_All_IQPSAT_Baseline_062118_1206_v127')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\SAT\CT80_All_IQPSAT_Baseline_062118_1206_v127')
    elif(aBLName.find('CT80_All_IQPSAT_Baseline_072518_0810_v133')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\SAT\CT80_All_IQPSAT_Baseline_072518_0810_v133')
    elif(aBLName.find('CT80_All_SAT_Baseline_050919_0200_v170')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\SAT\CT80_All_SAT_Baseline_050919_0200_v170')
    elif(aBLName.find('CT80_All_SAT_Baseline_102318_1217_v155')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\SAT\CT80_All_SAT_Baseline_102318_1217_v155')
    elif(aBLName.find('CT80_AllModels_IQPSAT_Baseline_01318_0959_v68')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\SAT\CT80_AllModels_IQPSAT_Baseline_01318_0959_v68')
        
    elif(aBLName.find('CT80_All_FAT_Baseline_050919_0200_v169')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\FAT\CT80_All_FAT_Baseline_050919_0200_v169')
    elif(aBLName.find('CT80_All_FAT_Baseline_102318_1207_v150')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\FAT\CT80_All_FAT_Baseline_102318_1207_v150')
    elif(aBLName.find('CT80_All_IQSFAT_Baseline_062118_0104_v126')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\FAT\CT80_All_IQSFAT_Baseline_062118_0104_v126')
    elif(aBLName.find('CT80_All_IQSFAT_Baseline_072518_0750_v132')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\FAT\CT80_All_IQSFAT_Baseline_072518_0750_v132')
    elif(aBLName.find('CT80_AllModels_IQS-FAT_Baseline_083017_0304_v30')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CT80\FAT\CT80_AllModels_IQS-FAT_Baseline_083017_0304_v30')
        
    elif(aBLName.find('9800_SEIO_SCMS_FAT_Baseline_102318_1158_v152')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CTX9800_SEIO_SCMS\FAT\9800_SEIO_SCMS_FAT_Baseline_102318_1158_v152')
        
    elif(aBLName.find('9800_SEIO_SCMS_IQPSAT_Baseline_091918_1211_v148')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CTX9800_SEIO_SCMS\SAT\9800_SEIO_SCMS_IQPSAT_Baseline_091918_1211_v148')
    elif(aBLName.find('9800_SEIO_SCMS_SAT_Baseline_102318_1229_v156')>-1):
        tBLDir = os.path.join(data_storage_base_dir, r'Baselines\CTX9800_SEIO_SCMS\SAT\9800_SEIO_SCMS_SAT_Baseline_102318_1229_v156')
        
    else:
        print(f'Cannot find BL files for {aBLName}')
        assert(0)
    return tBLDir
    
def GetLocalBaselineParamFileFromName(aBLName):
    #NOTE: aBLName can be full path from SPCA file or just filename
    tBLDir = GetBaselineDirPathFromName(aBLName)
    tParamFile = FindFileWithPattern(tBLDir, r'*.mat')
    assert(len(tParamFile)==1)
    return tParamFile[0]

def GetBaselineDir(aSpcaFileName):
    tBLName = GetBaselineDatasetNameFromSpca(aSpcaFileName)
    tBLDir = GetBaselineDirPathFromName(tBLName)
    return tBLDir