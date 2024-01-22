#!/usr/bin/env python

# READ ME: IN ALMOST EVERY CASE, THE EASIEST AND QUICKEST OPTION WILL BE TO SIMPLY USE
# gMetricInfos DECLARED AFTER CLASS DEFINITION
import os
import sys
import re
from pathlib import Path

import Utilities

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


# READ ME: IN ALMOST EVERY CASE, THE EASIEST AND QUICKEST OPTION WILL BE TO SIMPLY USE
# gMetricInfos DECLARED AFTER CLASS DEFINITION
class MetricInfos:
    def __init__(self, **kwargs):        
        self.threshold_csv_args = MetricInfos.pop_off_threshold_csv_args(kwargs)
        self.threshold_csv_args = MetricInfos.get_threshold_csv_args(self.threshold_csv_args)

        #-----------
        # Set additional kwargs if they exist, otherwise set to default values
        # e.g. var1 = kwargs.get('var1', var_1_default_value)
        #-----------
        
        self.metric_infos_df = MetricInfos.build_metric_infos_df(self.threshold_csv_args)
    
    @staticmethod
    def get_default_threshold_csv_args():
        std_csv_path = os.path.join(os.path.realpath('..'), 'ANSIMetricNamesAndThresholds.csv')
        return {
            'thresholds_csv_file_path' : std_csv_path,
            'metric_id_col' : 'MetricID', 
            'name_xml_col' : 'AbbreviatedMetricName', 
            'name_long_col' : 'FullMetricName', 
            'phantom_A_col' : 'PhantomA', 
            'phantom_B_col' : 'PhantomB', 
            'phantom_SAT_col' : 'PhantomSAT', 
            'min_col' : 'Min', 
            'max_col' : 'Max', 
            'min_OLD_col' : 'MinOLD', 
            'max_OLD_col' : 'MaxOLD'
        }
    
    @staticmethod
    def pop_off_threshold_csv_args(kwargs):
        # use get_default_threshold_csv_args just to grab all keys
        return_args = MetricInfos.get_default_threshold_csv_args()
        for key in return_args:
            return_args[key] = kwargs.pop(key, None)
        # remove elements not in kwargs (which were set to None in the above loop)
        return_args = {k:v for k,v in return_args.items() if v is not None}
        return return_args

    @staticmethod
    def get_threshold_csv_args(csv_args):
        # Any argument not specified by the user will be set to the default value
        return_args = MetricInfos.get_default_threshold_csv_args()
        for key in csv_args:
            assert(key in return_args)
            return_args[key] = csv_args[key]
        return return_args
    
    @staticmethod
    def replace_all_misspelled(string):
        MisspelledNames = ("Devation",  "Regioin")
        CorrectNames = ("Deviation", "Region")    
        assert(len(MisspelledNames)==len(CorrectNames))
        return_str = string
        for i in range(len(MisspelledNames)):
            return_str = return_str.replace(MisspelledNames[i], CorrectNames[i])
        return return_str

    @staticmethod
    def replace_back_all_misspelled(string):
        MisspelledNames = ("Devation",  "Regioin")
        CorrectNames = ("Deviation", "Region")
        assert(len(MisspelledNames)==len(CorrectNames))
        return_str = string
        for i in range(len(CorrectNames)):
            return_str = return_str.replace(CorrectNames[i], MisspelledNames[i])
        return return_str

    @staticmethod
    def replace_double_underscore_with_single(string):
        return_str = string
        return_str = return_str.replace('__', '_')
        return return_str

    @staticmethod
    def replace_double_space_with_single(string):
        return_str = string
        return_str = return_str.replace('  ', ' ')
        return return_str

    @staticmethod
    def replace_all_mistakes(string):
        return_str = string
        return_str = MetricInfos.replace_all_misspelled(return_str)
        return_str = MetricInfos.replace_double_underscore_with_single(return_str)
        return_str = MetricInfos.replace_double_space_with_single(return_str)
        return return_str
    
    @staticmethod
    def replace_all_mistakes_in_vec(metric_names):
        metric_names_clean = []
        for metric_name in metric_names:
            metric_names_clean.append(MetricInfos.replace_all_mistakes(metric_name))
        assert(len(metric_names)==len(metric_names_clean))
        return metric_names_clean
    
    @staticmethod
    def convert_minmax_str_to_float(minmax_str):
        minmax_dbl = -9999.0
        if Utilities.CompStrings_CaseInsensitive(minmax_str, 'None'):
            minmax_dbl = np.nan
        elif Utilities.CompStrings_CaseInsensitive(minmax_str, 'Removed'):
            minmax_dbl = np.nan
        elif minmax_str[-1]=='%':
            minmax_dbl = float(minmax_str[:-1])
            minmax_dbl /= 100.0
        else:
            minmax_dbl = float(minmax_str)
        return minmax_dbl

    @staticmethod
    def convert_minmax_col_str_to_float(df, minmax_col):
        df[minmax_col] = df[minmax_col].apply(lambda x: MetricInfos.convert_minmax_str_to_float(x))
        return df
    
    @staticmethod
    def build_metric_infos_df(csv_args):
        threshold_csv_args = MetricInfos.get_threshold_csv_args(csv_args)

        metric_infos_df = pd.read_csv(threshold_csv_args['thresholds_csv_file_path'])

        in_phantom_cols = [threshold_csv_args['phantom_A_col'], 
                           threshold_csv_args['phantom_B_col'], 
                           threshold_csv_args['phantom_SAT_col']]
        metric_infos_df[in_phantom_cols] = metric_infos_df[in_phantom_cols].astype('bool')

        minmax_cols = [threshold_csv_args['min_col'], 
                       threshold_csv_args['max_col'], 
                       threshold_csv_args['min_OLD_col'], 
                       threshold_csv_args['max_OLD_col']]
        for minmax_col in minmax_cols:
            metric_infos_df = MetricInfos.convert_minmax_col_str_to_float(metric_infos_df, minmax_col)

        metric_infos_df['Index'] = metric_infos_df[threshold_csv_args['metric_id_col']]
        metric_infos_df.set_index('Index', inplace=True)
        return metric_infos_df
        
    def get_metric_info_from_name(self, metric_name, 
                                  phantom_type=Utilities.PhantomType.kUnsetPhantomType, 
                                  verbose=True):
        # Returns pd.Series
        #xml or long metric_name will work
        #-------------------------
        # Although all metric names should now be correct, clean metric_name just in case
        #   e.g. if working with old CT80 Fault Test Data
        metric_name_cpy = MetricInfos.replace_all_mistakes(metric_name)
        #-------------------------
        # First, look through xml names
        name_xml_col = self.threshold_csv_args['name_xml_col']
        found_row = self.metric_infos_df[self.metric_infos_df[name_xml_col]==metric_name_cpy]
        assert(found_row.shape[0]>-1)
        if found_row.shape[0]==1:
            return found_row.squeeze() #squeeze returns pd.Series instead of pd.DataFrame
        elif found_row.shape[0]>1:
            assert(found_row.iloc[0][name_xml_col]=='ReconstructedObjectLength')
            assert(found_row.shape[0]==2)
            if phantom_type==Utilities.PhantomType.kB:
                found_row = found_row[found_row[self.threshold_csv_args['phantom_B_col']]==True]
                assert(found_row.shape[0]==1)
                return found_row.squeeze()
            else:
                found_row = found_row[found_row[self.threshold_csv_args['phantom_B_col']]==False]
                assert(found_row.shape[0]==1)
                return found_row.squeeze()
        else:
            assert(found_row.shape[0]==0)

        #-------------------------
        # If not found, look through long names
        name_long_col = self.threshold_csv_args['name_long_col']
        found_row = self.metric_infos_df[self.metric_infos_df[name_long_col]==metric_name_cpy]
        assert(found_row.shape[0]>-1)
        if found_row.shape[0]==1:
            return found_row.squeeze() #squeeze returns pd.Series instead of pd.DataFrame
        elif found_row.shape[0]>1:
            assert(found_row.iloc[0][name_long_col]=='Reconstructed Object Length')
            assert(found_row.shape[0]==2)
            if phantom_type==Utilities.PhantomType.kB:
                found_row = found_row[found_row[self.threshold_csv_args['phantom_B_col']]==True]
                assert(found_row.shape[0]==1)
                return found_row.squeeze()
            else:
                found_row = found_row[found_row[self.threshold_csv_args['phantom_B_col']]==False]
                assert(found_row.shape[0]==1)
                return found_row.squeeze()
        else:
            if verbose:
                print(f'CANNOT FIND METRIC INFO FOR:\n\tmetric_name = {metric_name_cpy}\nCRASH IMMINENT!!!!!\n')
            assert(0)


    def get_metric_info_from_id(self, metric_id):
        # Returns pd.Series
        found_row = self.metric_infos_df.loc[metric_id]
        assert(found_row.ndim==1)
        assert(found_row[self.threshold_csv_args['metric_id_col']]==metric_id)
        return found_row


    def get_metric_info(self, metric_id_or_name, 
                        phantom_type=Utilities.PhantomType.kUnsetPhantomType, 
                        verbose=True):
        if isinstance(metric_id_or_name, str):
            return self.get_metric_info_from_name(metric_id_or_name, phantom_type, verbose=verbose)
        elif isinstance(metric_id_or_name, int):
            return self.get_metric_info_from_id(metric_id_or_name) 
        else:
            assert(0)
            
    def is_metric_in_phantom(self, phantom_type, metric_id_or_name):
        metric_info = self.get_metric_info(metric_id_or_name, phantom_type)
        if(phantom_type==Utilities.PhantomType.kFAT):
            if(metric_info[self.threshold_csv_args['phantom_A_col']]==True 
            or metric_info[self.threshold_csv_args['phantom_B_col']]==True):
                return True
        elif(phantom_type==Utilities.PhantomType.kA):
            return metric_info[self.threshold_csv_args['phantom_A_col']]
        elif(phantom_type==Utilities.PhantomType.kB):
            return metric_info[self.threshold_csv_args['phantom_B_col']]    
        elif(phantom_type==Utilities.PhantomType.kSAT):
            return metric_info[self.threshold_csv_args['phantom_SAT_col']]
        elif(phantom_type==Utilities.PhantomType.kORT):
            return metric_info[self.threshold_csv_args['phantom_SAT_col']]
        else:
            assert(0)
            
    def is_metric_of_interest(self, metric_id_or_name, 
                              phantom_type, 
                              use_new_thresholds=True):
        if not self.is_metric_in_phantom(phantom_type, metric_id_or_name):
            return False
        metric_info = self.get_metric_info(metric_id_or_name, phantom_type)
        if use_new_thresholds:
            thresh_min = metric_info[self.threshold_csv_args['min_col']]
            thresh_max = metric_info[self.threshold_csv_args['max_col']]
        else:
            thresh_min = metric_info[self.threshold_csv_args['min_OLD_col']]
            thresh_max = metric_info[self.threshold_csv_args['max_OLD_col']]           
            
        return not(np.isnan(thresh_min) and np.isnan(thresh_max))
    
    def get_metrics_of_interest_df(self, phantom_type, use_new_thresholds=True):
        metric_ids = self.metric_infos_df[self.threshold_csv_args['metric_id_col']].tolist()
        assert(metric_ids == self.metric_infos_df.index.tolist())
        ids_of_interest = []
        for idx in metric_ids:
            if self.is_metric_of_interest(idx, phantom_type, use_new_thresholds):
                ids_of_interest.append(idx)
        return_metric_infos_df = self.metric_infos_df.loc[ids_of_interest].copy()
        return return_metric_infos_df
    
    def get_metrics_of_interest_xml(self, phantom_type, use_new_thresholds=True):
        metric_infos = self.get_metrics_of_interest_df(phantom_type, use_new_thresholds)
        metric_names = []
        for idx, metric_info in metric_infos.iterrows():
            metric_names.append(metric_info[self.threshold_csv_args['name_xml_col']])
        return metric_names

    def get_metrics_of_interest(self, phantom_type, use_new_thresholds=True):
        return self.get_metrics_of_interest_xml(phantom_type, use_new_thresholds)
    
    def get_metrics_of_interest_long(self, phantom_type, use_new_thresholds=True):
        metric_infos = self.get_metrics_of_interest_df(phantom_type, use_new_thresholds)
        metric_names = []
        for idx, metric_info in metric_infos.iterrows():
            metric_names.append(metric_info[self.threshold_csv_args['name_long_col']])
        return metric_names

    def get_metric_ids_of_interest(self, phantom_type, use_new_thresholds=True):
        metric_infos = self.get_metrics_of_interest_df(phantom_type, use_new_thresholds)
        metric_names = []
        for idx, metric_info in metric_infos.iterrows():
            metric_names.append(metric_info[self.threshold_csv_args['metric_id_col']])
        return metric_names
    
    def get_metric_name(self, metric_id_or_name, 
                        phantom_type=Utilities.PhantomType.kUnsetPhantomType, 
                        verbose=True):
        metric_info = self.get_metric_info(metric_id_or_name, phantom_type, verbose=verbose)
        return metric_info[self.threshold_csv_args['name_xml_col']]

    def get_metric_name_long(self, metric_id_or_name, 
                             phantom_type=Utilities.PhantomType.kUnsetPhantomType):
        metric_info = self.get_metric_info(metric_id_or_name, phantom_type)
        return metric_info[self.threshold_csv_args['name_long_col']]
    
    def get_metric_id(self, metric_id_or_name, 
                      phantom_type=Utilities.PhantomType.kUnsetPhantomType):
        metric_info = self.get_metric_info(metric_id_or_name, phantom_type)
        return metric_info[self.threshold_csv_args['metric_id_col']]
        
    def does_metric_id_and_name_make_sense(self, metric_id, metric_name):
        #-----
        try:
            name_from_metric_id = self.get_metric_name(metric_id, verbose=False)
        except:
            name_from_metric_id = 'name_from_metric_id'
        #-----
        try:
            name_from_metric_name = self.get_metric_name(metric_name, verbose=False)
        except:
            name_from_metric_name = 'name_from_metric_name'        
        #-----
        if name_from_metric_id==name_from_metric_name:
            return True
        else:
            return False
        
    def get_joined_metric_id_and_name(self, metric_id_or_name, 
                                      phantom_type=Utilities.PhantomType.kUnsetPhantomType,
                                      use_name_long=True):
        # Usefule because MATLAB outputs e.g. (1) Reconstructed Object Length
        metric_info = self.get_metric_info(metric_id_or_name, phantom_type)
        metric_id = metric_info[self.threshold_csv_args['metric_id_col']]
        if use_name_long:
            metric_name = metric_info[self.threshold_csv_args['name_long_col']]
        else:
            metric_name = metric_info[self.threshold_csv_args['name_xml_col']]
        return f'({metric_id}) {metric_name}'
        
    def split_joined_metric_id_and_name(self, metric_id_and_name, assert_result_makes_sense=True):
        # metric_id_and_name should be, e.g. '(1) Reconstructed Object Length'
        # Almost always, assert_result_makes_sense=True will be used
        # The purpose of even including this is for use with get_df_subset_by_metric_name_or_id
        # If metric_id_and_name cannot be split, (None, metric_id_and_name) is returned
        #--------------------
        # Expanded to work with lists also
        if isinstance(metric_id_and_name, list):
            return_list=[]
            for id_and_name in metric_id_and_name:
                return_list.append(self.split_joined_metric_id_and_name(id_and_name, assert_result_makes_sense))
            return return_list
        #--------------------
        metric_id_and_name_split = metric_id_and_name.split(maxsplit=1)
        if len(metric_id_and_name_split) != 2:
            return (None, metric_id_and_name)
        metric_id, metric_name = metric_id_and_name_split[0], metric_id_and_name_split[1]
        metric_name = metric_name.strip()
        metric_id = metric_id[1:-1]
        try:
            metric_id = int(metric_id)
        except:
            metric_id = metric_id
        
        # Make sure the result makes sense
        if assert_result_makes_sense:
            assert(self.does_metric_id_and_name_make_sense(metric_id, metric_name))
        return (metric_id, metric_name)
        
    def is_joined_metric_id_and_name(self, candidate_string):
        metric_id, metric_name = self.split_joined_metric_id_and_name(candidate_string, assert_result_makes_sense=False)
        if metric_id is None:
            return False
        if self.does_metric_id_and_name_make_sense(metric_id, metric_name):
            return True
        else:
            return False
 
    def set_metric_id_as_idx_in_df(self, df, metric_name_col='metric_name', metric_idx_col='metric_id', 
                                   phantom_type=Utilities.PhantomType.kUnsetPhantomType):
        # If metric_idx_col exists, simply make it the index
        # If it doesn't, then first build it and then set to index
        assert(metric_name_col in df.columns)
        if metric_idx_col not in df.columns:
            df[metric_idx_col] = df[metric_name_col].apply(lambda x: self.get_metric_id(x, phantom_type=phantom_type))
        df.set_index(metric_idx_col, inplace=True)
        return df
        
    def get_df_subset_by_metric_name_or_id(self, df, metric_name_to_find, 
                                           metric_name_col='metric_name', 
                                           phantom_type=Utilities.PhantomType.kUnsetPhantomType, 
                                           assert_single_found=True):
        # Initially designed to work when metric_name_to_find = name, name_long, or id
        #   i.e. metric_name_to_find = 'ReconstructedObjectLength', 'Reconstructed Object Length', or 1 all work
        # Expanded such that metric id and name works as well
        #   i.e. metric_name_to_find = '(1) Reconstructed Object Length' works
        #--------------------------------------------------
        # if metric_name_to_find is name, name_long, or id, then simple
        if(not self.is_joined_metric_id_and_name(metric_name_to_find) and 
           not self.is_joined_metric_id_and_name(df.iloc[0][metric_name_col])):
            return_df = df[df.apply(lambda x: self.get_metric_name_long(x[metric_name_col])==self.get_metric_name_long(metric_name_to_find), axis=1)].copy()
        #--------------------------------------------------
        # if metric_name_to_find is joined id and name, a little more effort
        else:
            metric_id_and_name = metric_name_to_find
            if not self.is_joined_metric_id_and_name(metric_id_and_name):
                metric_id_and_name = self.get_joined_metric_id_and_name(metric_id_and_name, phantom_type)

            df['tmp_metric_id_and_name'] = df[metric_name_col]
            if not self.is_joined_metric_id_and_name(df.iloc[0]['tmp_metric_id_and_name']):
                df['tmp_metric_id_and_name'] = df['tmp_metric_id_and_name'].apply(lambda x: self.get_joined_metric_id_and_name(x))

            return_df = df[df['tmp_metric_id_and_name']==metric_id_and_name].copy()
            return_df.drop(columns=['tmp_metric_id_and_name'], inplace=True)
            df.drop(columns=['tmp_metric_id_and_name'], inplace=True)

        if assert_single_found:
            assert(return_df.shape[0]==1) #make sure only one found!
        return return_df
        
        
    @staticmethod
    def split_metric_idname_to_id_name_in_df(df, 
                                             drop_metric_idname_col=True, make_metric_id_index=True, bring_metric_id_name_cols_tofront=True, 
                                             metric_idname_col='Name', 
                                             metric_id_name_cols={'metric_id_col':'MetricID', 'metric_name_col':'FullMetricName'}):
        #   df[metric_idname_col] should be e.g. '(1) Reconstructed Object Length'
        #   From df[metric_idname_col] extract metric_id_col and metric_name_col
        #
        #   Drop df[metric_idname_col] if drop_metric_idname_col==True
        #   Bring metric_id_col and metric_name_col to first columns if bring_metric_id_name_cols_tofront==True
        #   Make metric_id_col the index if make_metric_id_index==True    
        #-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!
        # LESSON: In general, vectorization with pandas is much faster than apply
        #         And, in particular, built-in Pandas functions are designed to operate on entire arrays, instead of sequentially on individual values (referred to as scalars). 
        #         Vectorization is the process of executing operations on entire arrays.
        #         So, whenever possible, use pandas built-in functions!
        # (1) Tried using apply with gMetricInfos.split_joined_metric_id_and_name, but this was VERY VERY VERY slow
        # df[[metric_id_col, metric_name_col]] = df.apply(lambda x: gMetricInfos.split_joined_metric_id_and_name(x[metric_idname_col], assert_result_makes_sense=False), axis=1, result_type='expand')
        #
        # (2) apply w/ axis=1 probably essentially goes row-by-row.  So, to try to speed things up, I grabbed df[metric_idname_col] as a list,
        #     ran the list through split_joined_metric_id_and_name (returning a list of tuples), and set df[[metric_id_col, metric_name_col]]
        #     to the list of tuples returned.  This is a kind of poor man's vectorization I guess.
        #     This sped things up SIGNIFICANTLY, ~2-3 orders of magnitude
        # df[[metric_id_col, metric_name_col]] = gMetricInfos.split_joined_metric_id_and_name(df[metric_idname_col].tolist())
        #
        # (3) However, the code used, although three lines instead of one, gives an additional ~1 order of magntidue improvement
        #     Therefore, this chosen as winner.
        # df[[metric_id_col, metric_name_col]] = df[metric_idname_col].str.split(n=1, expand=True)
        # df[metric_id_col] = df[metric_id_col].str[1:-1].astype(int)
        # df[metric_name_col] = df[metric_name_col].str.strip()
        #-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!-----!!!!!    
        df = df.copy()
        assert(metric_idname_col in df)
        assert(not any([col_name in df.columns for col_name in metric_id_name_cols.values()]))    
        metric_id_col = metric_id_name_cols['metric_id_col']
        metric_name_col = metric_id_name_cols['metric_name_col']
        
        # build metric_id_col and metric_name_col columns
        df[[metric_id_col, metric_name_col]] = df[metric_idname_col].str.split(n=1, expand=True)    
        # some formatting for metric_id_col
        df[metric_id_col] = df[metric_id_col].str[1:-1].astype(int) #makes e.g. '(1)' -> '1' -> 1    
        # some formatting for metric_name_col
        df[metric_name_col] = df[metric_name_col].str.strip() #makes e.g. ' Reconstructed Object Length' -> 'Reconstructed Object Length'
        
        if drop_metric_idname_col:
            df.drop(columns=[metric_idname_col], inplace=True)
        if bring_metric_id_name_cols_tofront:
            df.insert(0, metric_name_col, df.pop(metric_name_col))
            df.insert(0, metric_id_col, df.pop(metric_id_col))
            
        if make_metric_id_index:    
            df.set_index(metric_id_col, inplace=True)
        
        return df
    
gMetricInfos = MetricInfos()