#!/usr/bin/env python

r"""
Holds MECPOAn class.  See MECPOAn.MECPOAn for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re
import string
import json
from pathlib import Path

import pandas as pd
import numpy as np
from natsort import natsorted
import copy
import itertools

#--------------------------------------------------
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
from MECPODf import MECPODf, OutageDType
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
#--------------------------------------------------


class MECPOAn:
    r"""
    MECPOAn = Meter Event Counts Per Outage Analysis
    Class to hold a collection of Reason Counts Per Outage (RCPO) and Id (enddeviceeventtypeid) Counts Per Outage (ICPO)
      pd.DataFrame objects representing a single Meter Events Counts Per Outage Analysis (MECPOAn).
    All DFs in the class are built from the same underlying data, but differ in their normalizations.
    There are methods to project out e.g., different major/minor outage causes from the data.
    
    This class is mainly to help facilitate the plotting process.
    """
    
    def __init__(
        self                        , 
        data_type                   , 
        pkls_base_dir               ,
        days_min_max_outg_td_window = None, 
        pkls_sub_dir                = None,
        naming_tag                  = None,
        normalize_by_time_interval  = False, 
        is_no_outg                  = False, 
        outg_rec_nb_col             = 'index', 
        read_and_load_all_pickles   = False, 
        init_cpo_dfs_to             = pd.DataFrame() # Should be pd.DataFrame() or None
    ):
        #-----
        self.data_type = None
        # self.data_type will eventually be of type OutageDType.
        # However, the input data_type may be any of the following:
        #   OutageDType ==> self.data_type set directly
        #   int         ==> value must be within [e.value for e in OutageDType], but not equal to OutageDType['unset'].value
        #                   Converted via self.data_type = OutageDType(data_type)
        #   str         ==> value must be within [e.name for e in OutageDType], but not equal to 'unset'
        #                   Converted via self.data_type = OutageDType[input_data_type]
        self.set_data_type(input_data_type = data_type)
        #-----
        self.pkls_base_dir               = pkls_base_dir
        self.days_min_max_outg_td_window = days_min_max_outg_td_window
        self.pkls_sub_dir                = pkls_sub_dir
        self.naming_tag                  = naming_tag
        self.normalize_by_time_interval  = normalize_by_time_interval
        self.is_no_outg                  = is_no_outg
        self.outg_rec_nb_col             = outg_rec_nb_col
        #-----
        # Following are set in MECPOAn.read_and_load_all_pickles
        self.cpx_pkls_dir                = None
        self.end_events_dir              = None
        self.base_dir                    = None
        self.grp_by                      = None
        
        # When using self.init_cpo_dfs_to ALWAYS use copy.deepcopy(self.init_cpo_dfs_to),
        # otherwise, unintended consequences could return (self.init_cpo_dfs_to is None it
        # doesn't really matter, but when it is pd.DataFrame it does!)
        self.init_cpo_dfs_to = init_cpo_dfs_to
        #------------------------------
        self.ede_typeid_to_reason_df = pd.DataFrame()
        self.reason_to_ede_typeid_df = pd.DataFrame()
        self.time_infos_df           = copy.deepcopy(self.init_cpo_dfs_to)
        
        self.outg_rec_nbs = None
        self.mjr_mnr_causes_for_outages_df=None
        
        #--------------------------------------------------
        # NOTE: Due to the use of @property setters/getters below all elements in 
        #       rcpo_dfs_coll/icpo_dfs_coll can be treated like member attributes.
        #         e.g., one may call: self.rcpo_df_OG 
        #                 instead of: self.rcpo_dfs_coll['rcpo_df_OG']
        #       The purpose of keeping them in the dict structures is simply because it makes
        #       some other operations easier (e.g., matching columns between MECPOAn objects)
        self.rcpo_dfs_coll = dict(
            rcpo_df_OG                = copy.deepcopy(self.init_cpo_dfs_to), 
            #-----
            rcpo_df_raw               = copy.deepcopy(self.init_cpo_dfs_to), 
            rcpo_df_norm              = copy.deepcopy(self.init_cpo_dfs_to), 
            rcpo_df_norm_by_xfmr_nSNs = copy.deepcopy(self.init_cpo_dfs_to), 
            rcpo_df_norm_by_outg_nSNs = copy.deepcopy(self.init_cpo_dfs_to), 
            rcpo_df_norm_by_prim_nSNs = copy.deepcopy(self.init_cpo_dfs_to)
        )
        
        self.icpo_dfs_coll = dict(
            icpo_df_raw               = copy.deepcopy(self.init_cpo_dfs_to), 
            icpo_df_norm              = copy.deepcopy(self.init_cpo_dfs_to), 
            icpo_df_norm_by_xfmr_nSNs = copy.deepcopy(self.init_cpo_dfs_to), 
            icpo_df_norm_by_outg_nSNs = copy.deepcopy(self.init_cpo_dfs_to), 
            icpo_df_norm_by_prim_nSNs = copy.deepcopy(self.init_cpo_dfs_to)            
        )
        #--------------------------------------------------
        # In future, maybe make all these DFs MECPODf objects which all have own is_norm, 
        #   normalize_by_nSNs_included, etc. attributes.  For now, this works.
        self.is_norm_dict = dict(
            rcpo_df_OG                = False, 
            rcpo_df_raw               = False, 
            rcpo_df_norm              = True, 
            rcpo_df_norm_by_xfmr_nSNs = True, 
            rcpo_df_norm_by_outg_nSNs = True, 
            rcpo_df_norm_by_prim_nSNs = True, 

            icpo_df_raw               = False, 
            icpo_df_norm              = True, 
            icpo_df_norm_by_xfmr_nSNs = True, 
            icpo_df_norm_by_outg_nSNs = True, 
            icpo_df_norm_by_prim_nSNs = True  
        )
        #-----
        self.is_normalize_by_nSNs_included = dict(
            rcpo_df_OG                = True, 
            rcpo_df_raw               = False, 
            rcpo_df_norm              = False, 
            rcpo_df_norm_by_xfmr_nSNs = False, 
            rcpo_df_norm_by_outg_nSNs = False, 
            rcpo_df_norm_by_prim_nSNs = False, 

            icpo_df_raw               = False, 
            icpo_df_norm              = False, 
            icpo_df_norm_by_xfmr_nSNs = False, 
            icpo_df_norm_by_outg_nSNs = False, 
            icpo_df_norm_by_prim_nSNs = False  
        )
        #--------------------------------------------------
        # See @property cpo_df_name_to_norm_counts_col_dict
        self.norm_counts_df = copy.deepcopy(self.init_cpo_dfs_to)
        
        self.SNs_col  = '_SNs'
        self.nSNs_col = '_nSNs'
        
        self.outg_SNs  = '_outg_SNs'
        self.outg_nSNs = '_outg_nSNs'
        
        self.prim_SNs  = '_prim_SNs'
        self.prim_nSNs = '_prim_nSNs'
        
        self.xfmr_SNs  = '_xfmr_SNs'
        self.xfmr_nSNs = '_xfmr_nSNs'
        
        if read_and_load_all_pickles:
            self.read_and_load_all_pickles(
                base_dir                    = self.pkls_base_dir, 
                days_min_max_outg_td_window = self.days_min_max_outg_td_window, 
                sub_dir                     = self.pkls_sub_dir, 
                naming_tag                  = self.naming_tag, 
                normalize_by_time_interval  = self.normalize_by_time_interval, 
                is_no_outg                  = self.is_no_outg, 
            )
            
        #self.drop_empty_cpo_dfs()


    @staticmethod
    def get_data_type(
        input_data_type
    ):
        r"""
        data_type will eventually be of type OutageDType.
        However, the input input_data_type may be any of the following:
          OutageDType ==> data_type returned directly
          int         ==> value must be within [e.value for e in OutageDType], but not equal to OutageDType['unset'].value
                          Converted via data_type = OutageDType(input_data_type)
          str         ==> value must be within [e.name for e in OutageDType], but not equal to 'unset'
                          Converted via data_type = OutageDType[input_data_type]
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(input_data_type, [OutageDType, int, str]))
        #-------------------------
        if isinstance(input_data_type, OutageDType):
            return input_data_type
        #-------------------------
        if isinstance(input_data_type, int):
            assert(
                input_data_type in [e.value for e in OutageDType] and 
                input_data_type != OutageDType['unset'].value
            )
            return OutageDType(input_data_type)
        #-------------------------
        if isinstance(input_data_type, str):
            assert(
                input_data_type in [e.name for e in OutageDType] and 
                input_data_type != 'unset'
            )
            return OutageDType[input_data_type]
        #-------------------------
        assert(0)

    def set_data_type(
        self, 
        input_data_type
    ):
        r"""
        self.data_type will eventually be of type OutageDType.
        However, the input input_data_type may be any of the following:
          OutageDType ==> self.data_type set directly
          int         ==> value must be within [e.value for e in OutageDType], but not equal to OutageDType['unset'].value
                          Converted via self.data_type = OutageDType(input_data_type)
          str         ==> value must be within [e.name for e in OutageDType], but not equal to 'unset'
                          Converted via self.data_type = OutageDType[input_data_type]
        """
        #-------------------------
        self.data_type = MECPOAn.get_data_type(input_data_type=input_data_type)
        
        
    #--------------------------------------------------
    #--------------------PROPERTIES--------------------
    # In this case, used for setters/getters/deleters for DFs stored
    # inside of rcpo_dfs_coll and icpo_dfs_coll
    #------------------------------
    @property
    def cpo_dfs_coll(self):
        r"""
        NOTE!!!!!: Since this creates a new dict and returns it, this CANNOT BE USED TO SET DFs!!!!!
                     e.g., cpo_dfs_coll[cpo_df_name] = some_function(cpo_dfs_coll[cpo_df_name]) or even
                           cpo_dfs_coll[cpo_df_name] = some_df WILL NOT WORK
                     HOWEVER, if function performs inplace, I believe this still will change the member df
                       e.g., cpo_dfs_coll[cpo_df_name] = some_function_2(cpo_dfs_coll[cpo_df_name], inplace=True)
                   This behavior is different from e.g., rcpo_dfs, which does grab directly the underlying dict (without
                     creating a new dict before returned), and therefore rcpo_dfs[cpo_df_name] = some_df WILL WORK
        """
        return {**self.rcpo_dfs_coll, **self.icpo_dfs_coll}
    
    @property
    def cpo_dfs(self):
        r"""
        NOTE!!!!!: Since this creates a new dict and returns it, this CANNOT BE USED TO SET DFs!!!!!
                     e.g., cpo_dfs[cpo_df_name] = some_function(cpo_dfs[cpo_df_name]) or even
                           cpo_dfs[cpo_df_name] = some_df WILL NOT WORK
                     HOWEVER, if function performs inplace, I believe this still will change the member df
                       e.g., cpo_dfs[cpo_df_name] = some_function_2(cpo_dfs[cpo_df_name], inplace=True)
                   This behavior is different from e.g., rcpo_dfs, which does grab directly the underlying dict (without
                     creating a new dict before returned), and therefore rcpo_dfs[cpo_df_name] = some_df WILL WORK
        """
        return {**self.rcpo_dfs_coll, **self.icpo_dfs_coll}
        
    @property
    def rcpo_dfs(self):
        return self.rcpo_dfs_coll
        
    @property
    def icpo_dfs(self):
        return self.icpo_dfs_coll
    #------------------------------
    @property
    def rcpo_df_OG(self):
        return self.rcpo_dfs_coll['rcpo_df_OG']
    @rcpo_df_OG.setter
    def rcpo_df_OG(self, df):
        self.rcpo_dfs_coll['rcpo_df_OG'] = df
        if(
            self.rcpo_dfs_coll['rcpo_df_OG'] is not None and 
            self.rcpo_dfs_coll['rcpo_df_OG'].shape[0]>0
        ):
            self.set_outg_rec_nbs()
    @rcpo_df_OG.deleter
    def rcpo_df_OG(self):
        del self.rcpo_dfs_coll['rcpo_df_OG']
        self.rcpo_dfs_coll['rcpo_df_OG'] = copy.deepcopy(self.init_cpo_dfs_to)
    #------------------------------
    @property
    def rcpo_df_raw(self):
        return self.rcpo_dfs_coll['rcpo_df_raw']
    @rcpo_df_raw.setter
    def rcpo_df_raw(self, df):
        self.rcpo_dfs_coll['rcpo_df_raw'] = df
    @rcpo_df_raw.deleter
    def rcpo_df_raw(self):
        del self.rcpo_dfs_coll['rcpo_df_raw']
        self.rcpo_dfs_coll['rcpo_df_raw'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------
    @property
    def rcpo_df_norm(self):
        return self.rcpo_dfs_coll['rcpo_df_norm']
    @rcpo_df_norm.setter
    def rcpo_df_norm(self, df):
        self.rcpo_dfs_coll['rcpo_df_norm'] = df
    @rcpo_df_norm.deleter
    def rcpo_df_norm(self):
        del self.rcpo_dfs_coll['rcpo_df_norm']
        self.rcpo_dfs_coll['rcpo_df_norm'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------    
    @property
    def rcpo_df_norm_by_xfmr_nSNs(self):
        return self.rcpo_dfs_coll['rcpo_df_norm_by_xfmr_nSNs']
    @rcpo_df_norm_by_xfmr_nSNs.setter
    def rcpo_df_norm_by_xfmr_nSNs(self, df):
        self.rcpo_dfs_coll['rcpo_df_norm_by_xfmr_nSNs'] = df
    @rcpo_df_norm_by_xfmr_nSNs.deleter
    def rcpo_df_norm_by_xfmr_nSNs(self):
        del self.rcpo_dfs_coll['rcpo_df_norm_by_xfmr_nSNs']
        self.rcpo_dfs_coll['rcpo_df_norm_by_xfmr_nSNs'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------    
    @property
    def rcpo_df_norm_by_outg_nSNs(self):
        return self.rcpo_dfs_coll['rcpo_df_norm_by_outg_nSNs']
    @rcpo_df_norm_by_outg_nSNs.setter
    def rcpo_df_norm_by_outg_nSNs(self, df):
        self.rcpo_dfs_coll['rcpo_df_norm_by_outg_nSNs'] = df
    @rcpo_df_norm_by_outg_nSNs.deleter
    def rcpo_df_norm_by_outg_nSNs(self):
        del self.rcpo_dfs_coll['rcpo_df_norm_by_outg_nSNs']
        self.rcpo_dfs_coll['rcpo_df_norm_by_outg_nSNs'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------    
    @property
    def rcpo_df_norm_by_prim_nSNs(self):
        return self.rcpo_dfs_coll['rcpo_df_norm_by_prim_nSNs']
    @rcpo_df_norm_by_prim_nSNs.setter
    def rcpo_df_norm_by_prim_nSNs(self, df):
        self.rcpo_dfs_coll['rcpo_df_norm_by_prim_nSNs'] = df
    @rcpo_df_norm_by_prim_nSNs.deleter
    def rcpo_df_norm_by_prim_nSNs(self):
        del self.rcpo_dfs_coll['rcpo_df_norm_by_prim_nSNs']
        self.rcpo_dfs_coll['rcpo_df_norm_by_prim_nSNs'] = copy.deepcopy(self.init_cpo_dfs_to)   
    #------------------------------
    @property
    def icpo_df_raw(self):
        return self.icpo_dfs_coll['icpo_df_raw']
    @icpo_df_raw.setter
    def icpo_df_raw(self, df):
        self.icpo_dfs_coll['icpo_df_raw'] = df
    @icpo_df_raw.deleter
    def icpo_df_raw(self):
        del self.icpo_dfs_coll['icpo_df_raw']
        self.icpo_dfs_coll['icpo_df_raw'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------
    @property
    def icpo_df_norm(self):
        return self.icpo_dfs_coll['icpo_df_norm']
    @icpo_df_norm.setter
    def icpo_df_norm(self, df):
        self.icpo_dfs_coll['icpo_df_norm'] = df
    @icpo_df_norm.deleter
    def icpo_df_norm(self):
        del self.icpo_dfs_coll['icpo_df_norm']
        self.icpo_dfs_coll['icpo_df_norm'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------    
    @property
    def icpo_df_norm_by_xfmr_nSNs(self):
        return self.icpo_dfs_coll['icpo_df_norm_by_xfmr_nSNs']
    @icpo_df_norm_by_xfmr_nSNs.setter
    def icpo_df_norm_by_xfmr_nSNs(self, df):
        self.icpo_dfs_coll['icpo_df_norm_by_xfmr_nSNs'] = df
    @icpo_df_norm_by_xfmr_nSNs.deleter
    def icpo_df_norm_by_xfmr_nSNs(self):
        del self.icpo_dfs_coll['icpo_df_norm_by_xfmr_nSNs']
        self.icpo_dfs_coll['icpo_df_norm_by_xfmr_nSNs'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------    
    @property
    def icpo_df_norm_by_outg_nSNs(self):
        return self.icpo_dfs_coll['icpo_df_norm_by_outg_nSNs']
    @icpo_df_norm_by_outg_nSNs.setter
    def icpo_df_norm_by_outg_nSNs(self, df):
        self.icpo_dfs_coll['icpo_df_norm_by_outg_nSNs'] = df
    @icpo_df_norm_by_outg_nSNs.deleter
    def icpo_df_norm_by_outg_nSNs(self):
        del self.icpo_dfs_coll['icpo_df_norm_by_outg_nSNs']
        self.icpo_dfs_coll['icpo_df_norm_by_outg_nSNs'] = copy.deepcopy(self.init_cpo_dfs_to)
    #----------    
    @property
    def icpo_df_norm_by_prim_nSNs(self):
        return self.icpo_dfs_coll['icpo_df_norm_by_prim_nSNs']
    @icpo_df_norm_by_prim_nSNs.setter
    def icpo_df_norm_by_prim_nSNs(self, df):
        self.icpo_dfs_coll['icpo_df_norm_by_prim_nSNs'] = df
    @icpo_df_norm_by_prim_nSNs.deleter
    def icpo_df_norm_by_prim_nSNs(self):
        del self.icpo_dfs_coll['icpo_df_norm_by_prim_nSNs']
        self.icpo_dfs_coll['icpo_df_norm_by_prim_nSNs'] = copy.deepcopy(self.init_cpo_dfs_to)   
    #--------------------------------------------------
    @property
    def cpo_df_name_to_norm_counts_col_dict(self):
        r"""
        Used to look up which counts column to use for normalization.
          This is important when combining columns, different DFs, etc., as those which
            were normalized need to be un-normalized, combined, then re-normalized        
        """
        return_dict = dict(
            rcpo_df_OG                = self.nSNs_col, 
            rcpo_df_raw               = None, 
            rcpo_df_norm              = self.nSNs_col, 
            rcpo_df_norm_by_xfmr_nSNs = self.xfmr_nSNs, 
            rcpo_df_norm_by_outg_nSNs = self.outg_nSNs, 
            rcpo_df_norm_by_prim_nSNs = self.prim_nSNs, 

            icpo_df_raw               = None, 
            icpo_df_norm              = self.nSNs_col, 
            icpo_df_norm_by_xfmr_nSNs = self.xfmr_nSNs, 
            icpo_df_norm_by_outg_nSNs = self.outg_nSNs, 
            icpo_df_norm_by_prim_nSNs = self.prim_nSNs    
        )
        return return_dict
        
    @property
    def cpo_df_name_to_norm_list_col_dict(self):
        r"""
        Used to look up which counts column to use for normalization.
          This is important when combining columns, different DFs, etc., as those which
            were normalized need to be un-normalized, combined, then re-normalized        
        """
        return_dict = dict(
            rcpo_df_OG                = self.SNs_col, 
            rcpo_df_raw               = None, 
            rcpo_df_norm              = self.SNs_col, 
            rcpo_df_norm_by_xfmr_nSNs = self.xfmr_SNs, 
            rcpo_df_norm_by_outg_nSNs = self.outg_SNs, 
            rcpo_df_norm_by_prim_nSNs = self.prim_SNs, 

            icpo_df_raw               = None, 
            icpo_df_norm              = self.SNs_col, 
            icpo_df_norm_by_xfmr_nSNs = self.xfmr_SNs, 
            icpo_df_norm_by_outg_nSNs = self.outg_SNs, 
            icpo_df_norm_by_prim_nSNs = self.prim_SNs    
        )
        return return_dict
        
    @staticmethod
    def is_df_normalized(
        cpx_df_name
    ):
        r"""
        Given a pd.DataFrame name, this returns whether or not the DF is normalized.
        Returns: boolean.
    
        I don't love this function, the DFs themselves should know whether or not they are normalized (i.e., argument
          for storing the data as full MECPODf objects instead of pd.DataFrame object), but this works for now
        """
        #-------------------------
        is_norm_dict = dict(
            rcpo_df_OG                = False, 
            rcpo_df_raw               = False, 
            rcpo_df_norm              = True, 
            rcpo_df_norm_by_xfmr_nSNs = True, 
            rcpo_df_norm_by_outg_nSNs = True, 
            rcpo_df_norm_by_prim_nSNs = True, 
    
            icpo_df_raw               = False, 
            icpo_df_norm              = True, 
            icpo_df_norm_by_xfmr_nSNs = True, 
            icpo_df_norm_by_outg_nSNs = True, 
            icpo_df_norm_by_prim_nSNs = True  
        )
        assert(cpx_df_name in is_norm_dict.keys())
        return is_norm_dict[cpx_df_name]
        
        
    def identify_lists_and_counts_cols_in_df(
        self, 
        df, 
        addtnl_possible_lists_and_counts_cols = [
            ['_prem_nbs', '_nprem_nbs']
        ]
    ):
        r"""
        """
        #-------------------------
        assert(df.columns.nlevels<=2)
        #-------------------------
        if df.columns.nlevels==1:
            btm_lvl_cols = df.columns
        else:
            lvl_0_vals = df.columns.get_level_values(0).unique()
            # If df.columns.nlevels==2, then the DF must be of the form typical to rcpo_df_OG, where
            #   each level 0 values has the same set of level 1 columns
            lvl_1_vals = df[lvl_0_vals[0]].columns
            for lvl_0_val in lvl_0_vals:
                assert(len(set(df[lvl_0_val].columns).symmetric_difference(set(lvl_1_vals)))==0)
            btm_lvl_cols = lvl_1_vals
        #-----
        list_cols_dict = {k:v for k,v in self.cpo_df_name_to_norm_list_col_dict.items() 
                          if v in btm_lvl_cols}

        counts_cols_dict = {k:v for k,v in self.cpo_df_name_to_norm_counts_col_dict.items() 
                            if v in btm_lvl_cols}
        #-------------------------
        # For now, all list_cols must have matching counts_cols and vice versa
        assert(len(set(list_cols_dict.keys()).symmetric_difference(set(counts_cols_dict.keys())))==0)
        #-------------------------
        # Procedure below ensures the lists cols and counts cols are in the same order
        #   between the two output lists
        lists_w_counts_cols = []
        for df_name in list_cols_dict.keys():
            lists_w_counts_cols.append((list_cols_dict[df_name], counts_cols_dict[df_name]))

        # There will be duplicate entries, as, e.g., rcpo_df_norm and icpo_df_norm
        #   share the same list and counts cols
        # Eliminate duplicates
        lists_w_counts_cols = list(set(lists_w_counts_cols))
        #-------------------------
        # Look for any of the columns listed in addtnl_possible_lists_and_counts_cols
        if addtnl_possible_lists_and_counts_cols is not None:
            for addtnl_pair in addtnl_possible_lists_and_counts_cols:
                assert(len(addtnl_pair)==2)
                if addtnl_pair[0] in btm_lvl_cols or addtnl_pair[1] in btm_lvl_cols:
                    assert(addtnl_pair[0] in btm_lvl_cols and addtnl_pair[1] in btm_lvl_cols)
                    lists_w_counts_cols.append(addtnl_pair)
        #-------------------------
        # Separate out the lists and counts
        lists_cols, counts_cols = list(zip(*lists_w_counts_cols))
        # Convert tuples to lists
        lists_cols = list(lists_cols)
        counts_cols = list(counts_cols)
        #-------------------------
        return lists_cols, counts_cols
        
    #--------------------------------------------------
    def set_cpo_df(self, cpo_df_name, df):
        assert(cpo_df_name in self.cpo_dfs_coll)
        if cpo_df_name in self.rcpo_dfs_coll:
            self.rcpo_dfs_coll[cpo_df_name] = df
        else:
            self.icpo_dfs_coll[cpo_df_name] = df
    
    def drop_empty_cpo_dfs(self):
        to_del = []
        for idx, cpo_df in self.rcpo_dfs_coll.items():
            if cpo_df is None or cpo_df.shape[0]==0:
                to_del.append(idx)
        for idx in to_del:
            del self.rcpo_dfs_coll[idx]
        #-------------------------
        to_del = []
        for idx, cpo_df in self.icpo_dfs_coll.items():
            if cpo_df is None or cpo_df.shape[0]==0:
                to_del.append(idx)
        for idx in to_del:
            del self.icpo_dfs_coll[idx]
            
    def get_all_cpo_df_names(self, non_empty_only=True):
        if not non_empty_only:
            return list(self.cpo_dfs_coll.keys())
        return_names = []
        for df_name, df in self.cpo_dfs_coll.items():
            if df is None or df.shape[0]==0:
                continue
            else:
                return_names.append(df_name)
        return return_names
        
    #--------------------------------------------------
    def build_norm_counts_df(
        self                        , 
        cpo_df_name_w_counts        = 'rcpo_df_raw', 
        include_norm_lists          = True, 
        include_all_available       = True, 
        look_in_all_dfs_for_missing = True
    ):
        r"""
        Build a DF containing all of the normalization counts (and possibly lists of SNs).
        Instead of keeping these stored in the DFs themselves (or in rcpo_df_raw), it makes more sense
          to store them separately.  This way, they can be used by all, and any un-needed DFs can be dropped
          to save space without having and unintended consequences.

        cpo_df_name_w_counts:
            The name of the DF which currently stores the normalization counts information.
            Typically, this is rcpo_df_raw

        include_norm_lists:
            If True, include both the counts and the lists of SNs

        include_all_available:
            If True, include all  normalizations which can possibly be found (i.e., incluse those which are not 
              currently used anywhere (e.g., _xfmr_nPNs))
            If False, use only the norm counts and norm list cols which are currently used for various DFs (taken
              from self.cpo_df_name_to_norm_counts_col_dict and self.cpo_df_name_to_norm_list_col_dict)

        look_in_all_dfs_for_missing:
            If True, look in all DFs for any normalization columns not found in cpo_df_name_w_counts
        """
        #-------------------------
        if include_all_available:
            norm_counts_cols = [
                '_nSNs', 
                '_xfmr_nSNs', 
                '_xfmr_nPNs', 
                '_outg_nSNs', 
                '_prim_nSNs', 
                '_nprem_nbs'
            ]
            #-----
            norm_list_cols = [
                '_SNs',
                '_xfmr_SNs',
                '_xfmr_PNs',
                '_outg_SNs',
                '_prim_SNs',
                '_prem_nbs'
            ]
        else:
            norm_counts_cols = list(set(self.cpo_df_name_to_norm_counts_col_dict.values()))
            norm_list_cols   = list(set(self.cpo_df_name_to_norm_list_col_dict.values()))
        #-------------------------
        if include_norm_lists:
            norm_counts_cols = natsorted(norm_counts_cols+norm_list_cols)
        else:
            norm_counts_cols = natsorted(norm_counts_cols)
        norm_counts_cols = [x for x in norm_counts_cols if x is not None]
        #-------------------------
        # Find available_cols which are contained in self.get_cpo_df(cpo_df_name_w_counts)
        available_cols = [
            x for x in norm_counts_cols 
            if x in self.get_cpo_df(cpo_df_name_w_counts).columns
        ]
        norm_counts_df = self.get_cpo_df(cpo_df_name_w_counts)[available_cols].copy()
        #--------------------------------------------------
        #--------------------------------------------------
        # If look_in_all_dfs_for_missing, try to recover any columns from norm_counts_cols which are
        #   absent from available_cols by looking in all other cpo dfs
        if look_in_all_dfs_for_missing:
            missing_cols = list(set(norm_counts_cols).difference(set(available_cols)))
            other_cpo_df_names = [
                x for x in self.get_all_cpo_df_names(non_empty_only=True) 
                if x!=cpo_df_name_w_counts
            ]
            found_missing = []
            for missing_col in missing_cols:
                if missing_col in found_missing:
                    continue
                for cpo_df_name in other_cpo_df_names:
                    cpo_df = self.get_cpo_df(cpo_df_name)
                    # Complexity introduced below is intended for use with DFs having columns with nlevels=2
                    # But, the methods should work also for regular DFs, so two separate methods are not developed
                    found_col_idxs = Utilities_df.find_idxs_in_highest_order_of_columns(cpo_df, missing_col)
                    if len(found_col_idxs) > 0:
                        found_missing.append(missing_col)
                        found_cols = [cpo_df.columns[x] for x in found_col_idxs]
                        addtnl_norm_counts_df = cpo_df[found_cols]
                        #-------------------------
                        # If more than one column found (e.g., ('counts', '_SNs') and ('counts_norm', '_SNs') from rcpo_df_OG), 
                        #   make sure they are the same (then, only keep the first)
                        if addtnl_norm_counts_df.shape[1]>1:
                            for i in range(1, addtnl_norm_counts_df.shape[1]):
                                assert(addtnl_norm_counts_df[found_cols[0]].equals(addtnl_norm_counts_df[found_cols[i]]))
                        addtnl_norm_counts_df = addtnl_norm_counts_df[found_cols[0]].copy()
                        #-------------------------
                        # Append addtnl_norm_counts_df to norm_counts_df
                        #   As found elsewhere, if DFs are large, merge can be quite memory heavy
                        #   Therefore, check if indices match, and if so, use concat instead of merge
                        if addtnl_norm_counts_df.index.equals(norm_counts_df.index):
                            norm_counts_df = pd.concat([norm_counts_df, addtnl_norm_counts_df], axis=1)
                        else:
                            norm_counts_df = pd.merge(norm_counts_df, addtnl_norm_counts_df, how='left', left_index=True, right_index=True)
        #-------------------------
        return norm_counts_df


    def build_and_set_norm_counts_df(
        self                        , 
        remove_cols_from_dfs        = True, 
        cpo_df_name_w_counts        = 'rcpo_df_raw', 
        include_norm_lists          = True, 
        include_all_available       = True, 
        look_in_all_dfs_for_missing = True, 
        replace_if_present          = True
    ):
        r"""
        See build_norm_counts_df for more information.

        remove_cols_from_dfs:
            If True, run self.remove_SNs_cols_from_all_cpo_dfs
        """
        #-------------------------
        if(
            not replace_if_present and 
            self.norm_counts_df is not None and 
            self.norm_counts_df.shape[0]>0
        ):
            return
        #-------------------------
        norm_counts_df = self.build_norm_counts_df(
            cpo_df_name_w_counts        = cpo_df_name_w_counts, 
            include_norm_lists          = include_norm_lists, 
            include_all_available       = include_all_available, 
            look_in_all_dfs_for_missing = look_in_all_dfs_for_missing
        )
        self.norm_counts_df = norm_counts_df
        if remove_cols_from_dfs:
            self.remove_SNs_cols_from_all_cpo_dfs(
                SNs_tags           = None, 
                include_cpo_df_OG  = True,
                include_cpo_df_raw = True,
                cpo_dfs_to_ignore  = None, 
                is_long            = False
            )
        
    #--------------------------------------------------
    def get_shared_index(self):
        r"""
        Return the index of the DFs.

        Expect all to have the same indices.  
        However, I have observed that rcpo_df_norm_by_xfmr_nSNs may have only a subset of the indices.
          I believe this has to do with drop_xfmr_nSNs_eq_0 in build_rcpo_df_norm_by_xfmr_active_nSNs
        Thus, for a given DF, if the index does not equal that of rcpo_df_OG, it must equal a subset of it
        NOTE: The set operation, intended for the subset case above, should also handle the case where two
              sets of indices are the same but in different order    
        """
        #-------------------------
        cpo_df_names = self.get_all_cpo_df_names()
        # Grab the full set of indices from rcpo_df_OG
        shared_idx = self.get_cpo_df('rcpo_df_OG').index
        #-----
        # Make sure all DFs agree on the index.  See documentation above for insight into the
        # use of set.difference below
        for cpo_df_name in cpo_df_names:
            idx_i = self.get_cpo_df(cpo_df_name).index
            if not shared_idx.equals(idx_i):
                if len(set(idx_i).difference(set(shared_idx)))!=0:
                    print(f'Failure due to cpo_df_name = {cpo_df_name}')
                assert(len(set(idx_i).difference(set(shared_idx)))==0)
        #-----
        return shared_idx
        
    def add_unique_idfr_to_idx_in_all_dfs(self):
        r"""
        THIS ONLY WORKS FOR THE SPECIAL CASE WHERE THE INDEX ONLY HAS ONE LEVEL!
        After this operation, the DFs will have indices with two levels, the first equals
          the original index and the second contains the unique identifiers.
        The unique identifiers are simply a random prefix with the iloc value of the row in rcpo_df_OG

        Also, the DFs should have unique indices.

        Purpose/intent: This was built to handle the original baseline, where a single clean period was queried for
                          each transformer, and thus the results were only grouped by trsf_pole_nb.
                        The issue occurs when combining the baseline results from different years, for each year will
                          contain many of the same trsf_pole_nbs as the others, and after combining there is no way to
                          distinguish them from each other.
                        The different entries should be kept separate, not aggregated together, thus the ability to
                          distinguish them is important.
        """
        #-------------------------
        shared_idx = self.get_shared_index()

        # Indices should be unique
        assert(shared_idx.nunique()==len(shared_idx))
        #-------------------------
        # New level will simply be a random prefix with the iloc value of the row in rcpo_df_OG
        rand_pfx = Utilities.generate_random_string(str_len=5, letters=string.ascii_letters + string.digits)

        # I want the digit portion of the new indices to be uniform in length, so use zero front padding
        # NOTE: double {{ (}}) escape the { (}) at the beginning (end) of digits_frmt
        n_digits = int(np.ceil(np.log10(len(shared_idx))))
        digits_frmt = f'{{:0{n_digits}d}}'

        # Build the new index
        new_idx = [(x, rand_pfx+digits_frmt.format(i)) for i,x in enumerate(shared_idx)]
        new_idx = pd.MultiIndex.from_tuples(new_idx)

        # Build dictionary version, which will be used to ensure correct operation even if the other
        #   DFs have their indices in different order, of if they only contain a subset of the full index
        new_idx_dict = dict(new_idx.to_list())
        #-------------------------
        # Iterate over all DFs and update their indices
        cpo_df_names = self.get_all_cpo_df_names()
        for cpo_df_name_i in cpo_df_names:
            df_i = self.get_cpo_df(cpo_df_name_i)
            assert(df_i.index.nlevels==1)
            #-----
            if df_i.index.equals(new_idx.get_level_values(0)):
                df_i.index = new_idx
            else:
                new_idx_i = [(idx_ij, new_idx_dict[idx_ij]) for idx_ij in df_i.index]
                new_idx_i = pd.MultiIndex.from_tuples(new_idx_i)
                #-----
                df_i.index = new_idx_i
            self.set_cpo_df(cpo_df_name_i, df_i)
        #-------------------------
        # Make sure all indices agree again
        _ = self.get_shared_index()
        #-------------------------
        return
    
    def set_outg_rec_nbs(self):
        r"""
        Using self.rcpo_df_OG (i.e., self.rcpo_dfs_coll['rcpo_df_OG']), grab the outg_rec_nbs
        and set self.outg_rec_nbs
        """
        assert(isinstance(self.rcpo_df_OG, pd.DataFrame) and 
               self.rcpo_df_OG.shape[0]>0)
        self.outg_rec_nbs = MECPODf.get_outg_rec_nbs_list_from_cpo_df(cpo_df=self.rcpo_df_OG, idfr=self.outg_rec_nb_col, unique_only=True)
            
    def build_mjr_mnr_causes_for_outages_df(
        self                     , 
        include_equip_type       = True, 
        set_outg_rec_nb_as_index = True
    ):
        r"""
        """
        #-------------------------
        if self.outg_rec_nbs is None:
            self.set_outg_rec_nbs()
        assert(len(self.outg_rec_nbs)>0)
        #-------------------------
        self.mjr_mnr_causes_for_outages_df = DOVSOutages.get_mjr_mnr_causes_for_outages(
            outg_rec_nbs             = self.outg_rec_nbs, 
            include_equip_type       = include_equip_type, 
            set_outg_rec_nb_as_index = set_outg_rec_nb_as_index
        )
    
    @staticmethod
    def read_pickle_if_exists(
        dir_path  , 
        file_name , 
        verbose   = False
    ):
        f"""
        Simple function tries to read pickle at os.path.join(dir_path, file_name)
        If the file does not exist (or fails to open), it returns None
        """
        file_path = os.path.join(dir_path, file_name)
        if not os.path.exists(file_path):
            if verbose:
                print(f'No file exists at {file_path}')
            return None
        try:
            return_pkl = pd.read_pickle(file_path)
            return return_pkl
        except:
            if verbose:
                print(f'Unable to read file at {file_path}')
            return None
        

    @staticmethod
    def read_all_pickles(
        base_dir                    , 
        days_min_max_outg_td_window = None,
        sub_dir                     = None, 
        naming_tag                  = None, 
        normalize_by_time_interval  = False, 
        is_no_outg                  = False, 
        return_pkls_dir             = False
    ):
        r"""
        Somewhat specific function for loading all pickles needed to create a MECPOAn from a directory.
        
        base_dir:
          The base directory in which the files or file directory resides.
          If days_min_max_outg_td_window and sub_dir are both None, the pickle files should reside in base_dir
          
        days_min_max_outg_td_window/sub_dir:
          These identify the subdirectory within which the pickle files should reside.
          If both are None, the pickle files should reside in base_dir.
          If both are not None, the subdirectory to search is taken from sub_dir, and days_min_max_outg_td_window will
            only be used if normalize_by_time_interval==True
          If one is None and the other is not, the latter will be used to identify the subdirectory.
          #-----
          days_min_max_outg_td_window:
            Should be a tuple with two int values, 0th element for the min_outg_td_window, and the 1st element 
              for max_outg_td_window
            The sub-directory in which the pickle files should reside will be:
              f'outg_td_window_{days_min_max_outg_td_window[0]}_to_{days_min_max_outg_td_window[1]}_days'
              
          sub_dir:
            Should be a string representing the name of the sub-directory in which the files reside.

        naming_tag:
          Common tag identifier for all files of this type.
            e.g., for entire outage collection available, naming_tag is None
            e.g., for direct meters, naming_tag = '_prim_strict'
            e.g., for no outages sample, naming_tag = '_no_outg'
            e.g., rcpo_df_OG will be expected to be located at the following path:
              os.path.join(pkls_dir, f'rcpo{naming_tag}_df_OG.pkl')
              
        normalize_by_time_interval:
          If True, the DFs are normalized by the time interval (taken from days_min_max_outg_td_window)
        
        is_no_outg:
          Direct which DFs to load
        """
        #-------------------------
        assert(days_min_max_outg_td_window is not None or sub_dir is not None)
        #-----
        if naming_tag is None:
            naming_tag=''
        #-----    
        if normalize_by_time_interval:
            assert(days_min_max_outg_td_window is not None)
        #-------------------------
        pkls_subdir=None
        if sub_dir is not None:
            pkls_subdir = sub_dir
        else:
            assert(days_min_max_outg_td_window is not None)
            pkls_subdir = f'outg_td_window_{days_min_max_outg_td_window[0]}_to_{days_min_max_outg_td_window[1]}_days'
        #-------------------------
        if pkls_subdir is not None:
            pkls_dir = os.path.join(base_dir, pkls_subdir)
        else:
            pkls_dir=base_dir
        assert(os.path.isdir(pkls_dir))
        #----------------------------------------------------------------------------------------------------
        rcpo_df_OG                = MECPOAn.read_pickle_if_exists(pkls_dir, f'rcpo{naming_tag}_df_OG.pkl', verbose=True)
        #-------------------------
        ede_typeid_to_reason_df   = MECPOAn.read_pickle_if_exists(pkls_dir, f'ede_typeid_to_reason{naming_tag}_df_OG.pkl', verbose=True)
        reason_to_ede_typeid_df   = AMIEndEvents.invert_ede_typeid_to_reason_df(ede_typeid_to_reason_df)   
        #-------------------------
        rcpo_df_raw               = MECPOAn.read_pickle_if_exists(pkls_dir, f'rcpo{naming_tag}_df_raw.pkl', verbose=True)
        rcpo_df_norm              = MECPOAn.read_pickle_if_exists(pkls_dir, f'rcpo{naming_tag}_df_norm.pkl', verbose=True)
        rcpo_df_norm_by_xfmr_nSNs = MECPOAn.read_pickle_if_exists(pkls_dir, f'rcpo{naming_tag}_df_norm_by_xfmr_nSNs.pkl', verbose=True)
        #-----
        icpo_df_raw               = MECPOAn.read_pickle_if_exists(pkls_dir, f'icpo{naming_tag}_df_raw.pkl', verbose=False)
        icpo_df_norm              = MECPOAn.read_pickle_if_exists(pkls_dir, f'icpo{naming_tag}_df_norm.pkl', verbose=False)
        icpo_df_norm_by_xfmr_nSNs = MECPOAn.read_pickle_if_exists(pkls_dir, f'icpo{naming_tag}_df_norm_by_xfmr_nSNs.pkl', verbose=False)
        if not is_no_outg:
            rcpo_df_norm_by_outg_nSNs = MECPOAn.read_pickle_if_exists(pkls_dir, f'rcpo{naming_tag}_df_norm_by_outg_nSNs.pkl', verbose=False)
            rcpo_df_norm_by_prim_nSNs = MECPOAn.read_pickle_if_exists(pkls_dir, f'rcpo{naming_tag}_df_norm_by_prim_nSNs.pkl', verbose=False)
            #-----
            icpo_df_norm_by_outg_nSNs = MECPOAn.read_pickle_if_exists(pkls_dir, f'icpo{naming_tag}_df_norm_by_outg_nSNs.pkl', verbose=False)
            icpo_df_norm_by_prim_nSNs = MECPOAn.read_pickle_if_exists(pkls_dir, f'icpo{naming_tag}_df_norm_by_prim_nSNs.pkl', verbose=False)
        #----------------------------------------------------------------------------------------------------
        #------------------------- Remove SNs cols ------------------------- 
        if rcpo_df_norm is not None:
            rcpo_df_norm = MECPODf.remove_SNs_cols_from_rcpo_df(rcpo_df_norm)
        if rcpo_df_norm_by_xfmr_nSNs is not None:
            rcpo_df_norm_by_xfmr_nSNs = MECPODf.remove_SNs_cols_from_rcpo_df(rcpo_df_norm_by_xfmr_nSNs)
        #-----
        if icpo_df_norm is not None:
            icpo_df_norm = MECPODf.remove_SNs_cols_from_rcpo_df(icpo_df_norm)
        if icpo_df_norm_by_xfmr_nSNs is not None:
            icpo_df_norm_by_xfmr_nSNs = MECPODf.remove_SNs_cols_from_rcpo_df(icpo_df_norm_by_xfmr_nSNs)
        if not is_no_outg:
            if rcpo_df_norm_by_outg_nSNs is not None:
                rcpo_df_norm_by_outg_nSNs = MECPODf.remove_SNs_cols_from_rcpo_df(rcpo_df_norm_by_outg_nSNs)
            if rcpo_df_norm_by_prim_nSNs is not None:
                rcpo_df_norm_by_prim_nSNs = MECPODf.remove_SNs_cols_from_rcpo_df(rcpo_df_norm_by_prim_nSNs)
            #-----
            if icpo_df_norm_by_outg_nSNs is not None:
                icpo_df_norm_by_outg_nSNs = MECPODf.remove_SNs_cols_from_rcpo_df(icpo_df_norm_by_outg_nSNs)
            if icpo_df_norm_by_prim_nSNs is not None:
                icpo_df_norm_by_prim_nSNs = MECPODf.remove_SNs_cols_from_rcpo_df(icpo_df_norm_by_prim_nSNs)
        #----------------------------------------------------------------------------------------------------
        return_dict = {
            'rcpo_df_OG'                : rcpo_df_OG, 
            'ede_typeid_to_reason_df'   : ede_typeid_to_reason_df, 
            'reason_to_ede_typeid_df'   : reason_to_ede_typeid_df, 

            'rcpo_df_raw'               : rcpo_df_raw, 
            'rcpo_df_norm'              : rcpo_df_norm, 
            'rcpo_df_norm_by_xfmr_nSNs' : rcpo_df_norm_by_xfmr_nSNs,

            'icpo_df_raw'               : icpo_df_raw, 
            'icpo_df_norm'              : icpo_df_norm, 
            'icpo_df_norm_by_xfmr_nSNs' : icpo_df_norm_by_xfmr_nSNs
        }
        if not is_no_outg:
            return_dict = {
                **return_dict, 
                **{
                    'rcpo_df_norm_by_outg_nSNs' : rcpo_df_norm_by_outg_nSNs,
                    'rcpo_df_norm_by_prim_nSNs' : rcpo_df_norm_by_prim_nSNs, 

                    'icpo_df_norm_by_outg_nSNs' : icpo_df_norm_by_outg_nSNs, 
                    'icpo_df_norm_by_prim_nSNs' : icpo_df_norm_by_prim_nSNs
                }
            }
        #----------------------------------------------------------------------------------------------------
        if normalize_by_time_interval:
            for key in return_dict:
                if 'cpo' in key and return_dict[key] is not None:
                    return_dict[key] = MECPODf.normalize_rcpo_df_by_time_interval(
                        rcpo_df=return_dict[key], 
                        days_min_outg_td_window=days_min_max_outg_td_window[0], 
                        days_max_outg_td_window=days_min_max_outg_td_window[1], 
                    )
        #----------------------------------------------------------------------------------------------------
        if return_pkls_dir:
            return return_dict, pkls_dir
        return return_dict

    def load_dfs_from_dict(
        self, 
        input_dict
    ):
        r"""
        """
        #-------------------------
        self.ede_typeid_to_reason_df   = input_dict.get('ede_typeid_to_reason_df', None)
        self.reason_to_ede_typeid_df   = input_dict.get('reason_to_ede_typeid_df', None)
        #-------------------------
        self.rcpo_df_OG                = input_dict.get('rcpo_df_OG', copy.deepcopy(self.init_cpo_dfs_to))
        #-------------------------
        self.rcpo_df_raw               = input_dict.get('rcpo_df_raw', copy.deepcopy(self.init_cpo_dfs_to))
        self.rcpo_df_norm              = input_dict.get('rcpo_df_norm', copy.deepcopy(self.init_cpo_dfs_to))
        self.rcpo_df_norm_by_xfmr_nSNs = input_dict.get('rcpo_df_norm_by_xfmr_nSNs', copy.deepcopy(self.init_cpo_dfs_to))
        self.rcpo_df_norm_by_outg_nSNs = input_dict.get('rcpo_df_norm_by_outg_nSNs', copy.deepcopy(self.init_cpo_dfs_to))
        self.rcpo_df_norm_by_prim_nSNs = input_dict.get('rcpo_df_norm_by_prim_nSNs', copy.deepcopy(self.init_cpo_dfs_to))
        #-------------------------    
        self.icpo_df_raw               = input_dict.get('icpo_df_raw', copy.deepcopy(self.init_cpo_dfs_to))
        self.icpo_df_norm              = input_dict.get('icpo_df_norm', copy.deepcopy(self.init_cpo_dfs_to))
        self.icpo_df_norm_by_xfmr_nSNs = input_dict.get('icpo_df_norm_by_xfmr_nSNs', copy.deepcopy(self.init_cpo_dfs_to))
        self.icpo_df_norm_by_outg_nSNs = input_dict.get('icpo_df_norm_by_outg_nSNs', copy.deepcopy(self.init_cpo_dfs_to))
        self.icpo_df_norm_by_prim_nSNs = input_dict.get('icpo_df_norm_by_prim_nSNs', copy.deepcopy(self.init_cpo_dfs_to))
        #-------------------------
        # Making a tuple so it is immutable (so I can use methods e.g., assert(len(set(grp_by_vals))==1) when grp_by_vals is a list
        #   of such self.grp_by objects (i.e., a list of tuples)
        self.grp_by = tuple(self.rcpo_df_OG.index.names)
        #-----
        if self.rcpo_df_raw is not None and self.rcpo_df_raw.shape[0]>0:
            assert(self.grp_by==tuple(self.rcpo_df_raw.index.names))
        if self.rcpo_df_norm is not None and self.rcpo_df_norm.shape[0]>0:
            assert(self.grp_by==tuple(self.rcpo_df_norm.index.names))
        if self.rcpo_df_norm_by_xfmr_nSNs is not None and self.rcpo_df_norm_by_xfmr_nSNs.shape[0]>0:
            assert(self.grp_by==tuple(self.rcpo_df_norm_by_xfmr_nSNs.index.names))
        if self.rcpo_df_norm_by_outg_nSNs is not None and self.rcpo_df_norm_by_outg_nSNs.shape[0]>0:
            assert(self.grp_by==tuple(self.rcpo_df_norm_by_outg_nSNs.index.names))
        if self.rcpo_df_norm_by_prim_nSNs is not None and self.rcpo_df_norm_by_prim_nSNs.shape[0]>0:
            assert(self.grp_by==tuple(self.rcpo_df_norm_by_prim_nSNs.index.names))
            #-----
        if self.icpo_df_raw is not None and self.icpo_df_raw.shape[0]>0:
            assert(self.grp_by==tuple(self.icpo_df_raw.index.names))
        if self.icpo_df_norm is not None and self.icpo_df_norm.shape[0]>0:
            assert(self.grp_by==tuple(self.icpo_df_norm.index.names))
        if self.icpo_df_norm_by_xfmr_nSNs is not None and self.icpo_df_norm_by_xfmr_nSNs.shape[0]>0:
            assert(self.grp_by==tuple(self.icpo_df_norm_by_xfmr_nSNs.index.names))
        if self.icpo_df_norm_by_outg_nSNs is not None and self.icpo_df_norm_by_outg_nSNs.shape[0]>0:
            assert(self.grp_by==tuple(self.icpo_df_norm_by_outg_nSNs.index.names))
        if self.icpo_df_norm_by_prim_nSNs is not None and self.icpo_df_norm_by_prim_nSNs.shape[0]>0:
            assert(self.grp_by==tuple(self.icpo_df_norm_by_prim_nSNs.index.names))
        
    def read_and_load_all_pickles(
        self                        , 
        base_dir                    , 
        days_min_max_outg_td_window = None,
        sub_dir                     = None, 
        naming_tag                  = None, 
        normalize_by_time_interval  = False, 
        is_no_outg                  = False
    ):
        r"""
        """
        #-------------------------
        dfs_dict, pkls_dir = MECPOAn.read_all_pickles(
            base_dir                    = base_dir, 
            days_min_max_outg_td_window = days_min_max_outg_td_window, 
            sub_dir                     = sub_dir, 
            naming_tag                  = naming_tag , 
            normalize_by_time_interval  = normalize_by_time_interval, 
            is_no_outg                  = is_no_outg, 
            return_pkls_dir             = True
        )
        #-------------------------
        # e.g., pkls_dir       = ...\dovs_and_end_events_data\20231221\20230401_20231130\Outages\rcpo_dfs_GRP_BY_OUTG_AND_XFMR\outg_td_window_1_to_6_days
        #       pkls_base_dir  = ...\dovs_and_end_events_data\20231221\20230401_20231130\Outages\rcpo_dfs_GRP_BY_OUTG_AND_XFMR
        #       base_dir       = ...\dovs_and_end_events_data\20231221\20230401_20231130\Outages
        #       end_events_dir = ...\dovs_and_end_events_data\20231221\20230401_20231130\Outages\EndEvents
        self.cpx_pkls_dir = pkls_dir
        pkls_base_dir = str(Path(self.cpx_pkls_dir).parent)
        #-----
        base_dir = str(Path(pkls_base_dir).parent)
        assert(os.path.isdir(base_dir))
        self.base_dir = base_dir
        #-----
        end_events_dir = os.path.join(self.base_dir, 'EndEvents')
        assert(os.path.isdir(end_events_dir))
        self.end_events_dir = end_events_dir
        #-------------------------
        self.load_dfs_from_dict(input_dict=dfs_dict)
        
    #-----------------------------------------------------------------------------------------------------------------------------    
    @staticmethod
    def get_bsln_time_interval_infos_from_summary_file(
        summary_path , 
        date_only    = False, 
        date_col     = 'aep_event_dt'
    ):
        r"""
        Specialized function.
        TODO!!!!!!!!!!!!!!!!!!
        In the future, this stuff should probably be output at run-time somewhere    
        """
        #-------------------------
        assert(os.path.exists(summary_path))
        #-------------------------
        f = open(summary_path)
        summary_json_data = json.load(f)
        assert('sql_statement' in summary_json_data)
        sql_statement = summary_json_data['sql_statement']
        #-------------------------
        f.close()
        #-------------------------
        # Find the last instance of "SELECT * FROM USG_X" to extract how many sets of 
        # t_min,t_max,prem_nbs to expect.
        # If not found, expect only one
        pattern = r"SELECT \* FROM .*_(\d*)$"
        found_all = re.findall(pattern, sql_statement)
        if len(found_all)==0:
            n_groups_expected = 1
        else:
            assert(len(found_all)==1)
            n_groups_expected = int(found_all[0])+1
        #-------------------------
        try:
            addtnl_groupby_cols=summary_json_data['build_sql_function_kwargs']['df_args']['addtnl_groupby_cols']
        except:
            addtnl_groupby_cols=None
        #-------------------------
        # So obnoxious...using flags=re.MULTILINE|re.DOTALL with .* was causing the trailing ) and \n to match in premise numbers
        #   This also made it such that only the last occurrence of the match was returned.
        #   What I found to work was eliminating the re.DOTALL flag and [\s\S] to match a newline or any symbol.
        #     Typically, . matches everything BUT newline characters (unless using re.DOTALL).
        #     The main idea is that the opposite shorthand classes inside a character class match any symbol there is in the input string.
        # NOTE: The new pattern should find both, e.g.:
        #       (a) un_rin.aep_premise_nb IN ('102186833','102252463','106876833','108452463')
        #       (b) un_rin.aep_premise_nb = '072759453'
        #       However, now need the if prem_nbs[0]=='(' block below
        #       ALSO: (?: TIMESTAMP){0,1} needed to be included (twice) after switch to Athena
        #             See, e.g., is_timestamp in SQLWhere class

        # pattern = r"SELECT[\s\S]+?"\
        #           r"CAST.* BETWEEN '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})' AND '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'[\s\S]+?"\
        #           r"aep_premise_nb IN \((.*)\)[\s\S]+?"    
        #-------------------------
        time_intrvl_pttrn = r"CAST.* BETWEEN(?: TIMESTAMP){0,1} '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})' AND(?: TIMESTAMP){0,1} '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'[\s\S]+?"
        if date_only:
            time_intrvl_pttrn = date_col + r"\s*BETWEEN '(\d{4}-\d{2}-\d{2})' AND '(\d{4}-\d{2}-\d{2})'[\s\S]+?"
        #-------------------------
        pattern = r"SELECT[\s\S]+?"
        if addtnl_groupby_cols is not None:
            for addtnl_groupby_col in addtnl_groupby_cols:
                pattern += r"'(.*)' AS {}[\s\S]+?".format(addtnl_groupby_col)
        pattern += time_intrvl_pttrn + r"aep_premise_nb?\s*(?:IN|=)?\s*(\((?:.*)\)|(?:\'.*\'))[\s\S]+?"
        #-------------------------
        found_all = re.findall(pattern, sql_statement, flags=re.MULTILINE)
        assert(len(found_all)>0)
        #-------------------------
        return_coll=[]
        for found in found_all:
            if addtnl_groupby_cols is None:
                assert(len(found)==3)
                t_min,t_max,prem_nbs = found
            else:
                assert(len(found)==3+len(addtnl_groupby_cols))
                addtnl_groupby_cols_vals = found[:len(addtnl_groupby_cols)]
                t_min,t_max,prem_nbs = found[-3:]

            if prem_nbs[0]=='(':
                assert(prem_nbs[-1]==')')
                prem_nbs=prem_nbs[1:-1]
            prem_nbs = prem_nbs.replace('\'', '')
            prem_nbs = prem_nbs.split(',')
            return_dict_i = {
                'prem_nbs':prem_nbs, 
                't_min':t_min, 
                't_max':t_max
            }
            if addtnl_groupby_cols is not None:
                assert(len(addtnl_groupby_cols)==len(addtnl_groupby_cols_vals))
                for i_col,addtnl_groupby_col in enumerate(addtnl_groupby_cols):
                    return_dict_i[addtnl_groupby_col] = addtnl_groupby_cols_vals[i_col]
            return_coll.append(return_dict_i)
        #-------------------------
        return return_coll
        
        
        
    @staticmethod
    def get_bsln_time_interval_infos_df_from_summary_file(
        summary_path            , 
        output_prem_nbs_col     = 'prem_nbs', 
        output_t_min_col        = 't_min', 
        output_t_max_col        = 't_max', 
        make_addtnl_groupby_idx = True,
        include_summary_path    = False, 
        date_only               = False, 
        date_col                = 'aep_event_dt'
    ):
        r"""
        Returns a pd.DataFrame version of MECPOAn.get_bsln_time_interval_infos_from_summary_file
        """
        #-------------------------
        return_df = pd.DataFrame()
        bsln_time_infos = MECPOAn.get_bsln_time_interval_infos_from_summary_file(
            summary_path = summary_path, 
            date_only    = date_only, 
            date_col     = date_col
        )
        expected_cols = ['prem_nbs', 't_min', 't_max']
        for i,bsln_time_info_i in enumerate(bsln_time_infos):
            #-------------------------
            # To mirror what is done for outages, where the period of interest is taken before the outages,
            #   use t_max as DT_OFF_TS_FULL
            #   *** Technically, for both outages and non-outages, the data were collected 30 days before AND after
            #       the event (where event is outage for outages, and event is randomly selected date for non-outages)
            #       Therefore, to exactly mirror the outages, I suppose one should take DT_OFF_TS_FULL to be the mid-point
            #       betweeen t_min and t_max.  However, this doesn't really matter, all that matters is there are entries
            #       at least window_width_days = days_max_outg_td_window-days_min_outg_td_window+1 before whatever is called
            #       DT_OFF_TS_FULL
            #-------------------------
            bsln_time_info_df_i = pd.DataFrame(bsln_time_info_i)
            # Make sure expected columns are found
            if i==0:
                assert(len(set(expected_cols).difference(set(bsln_time_info_df_i.columns)))==0)
                addtnl_groupby_cols = [x for x in bsln_time_info_df_i.columns if x not in expected_cols]
                expected_cols.extend(addtnl_groupby_cols)
            assert(len(set(expected_cols).symmetric_difference(set(bsln_time_info_df_i.columns)))==0)
            return_df = pd.concat([return_df, bsln_time_info_df_i], ignore_index=True)
        #-------------------------
        # Make prem nbs into list
        return_df=return_df.groupby(addtnl_groupby_cols+['t_min', 't_max']).agg(list)
        return_df=return_df.reset_index()
        #-------------------------
        # Rename columns
        rename_dict = {
            'prem_nbs':output_prem_nbs_col, 
            't_min':output_t_min_col, 
            't_max':output_t_max_col
        }
        return_df = return_df.rename(columns=rename_dict)
        #-------------------------
        if date_only:
            return_df[output_t_min_col] = pd.to_datetime(return_df[output_t_min_col], format='%Y-%m-%d')
            return_df[output_t_max_col] = pd.to_datetime(return_df[output_t_max_col], format='%Y-%m-%d')
        else:
            return_df[output_t_min_col] = pd.to_datetime(return_df[output_t_min_col], format='%Y-%m-%d %H:%M:%S')
            return_df[output_t_max_col] = pd.to_datetime(return_df[output_t_max_col], format='%Y-%m-%d %H:%M:%S')
        #-------------------------
        if make_addtnl_groupby_idx:
            return_df=return_df.set_index(addtnl_groupby_cols)
        if include_summary_path:
            return_df['summary_path'] = summary_path
        #-------------------------
        return return_df
    
    
    @staticmethod
    def get_bsln_time_interval_infos_df_from_summary_files(
        summary_paths           , 
        output_prem_nbs_col     = 'prem_nbs', 
        output_t_min_col        = 't_min', 
        output_t_max_col        = 't_max', 
        make_addtnl_groupby_idx = True, 
        include_summary_paths   = False, 
        date_only               = False, 
        date_col                = 'aep_event_dt'
    ):
        r"""
        Handles multiple summary files

        Note: drop_duplicates will remove rows if indices are different (but all columns equal)
              Therefore, if make_addtnl_groupby_idx==True, this should only be done AFTER drop duplicates
              This explains why make_addtnl_groupby_idx=False in the call to MECPOAn.get_bsln_time_interval_infos_df_from_summary_file
        Note: The reason for drop duplicates if for the case where a collection is split over mulitple
              files/runs (i.e., the asynchronous case)
        """
        return_df = pd.DataFrame()
        for summary_path in summary_paths:
            df_i = MECPOAn.get_bsln_time_interval_infos_df_from_summary_file(
                summary_path            = summary_path, 
                output_prem_nbs_col     = output_prem_nbs_col, 
                output_t_min_col        = output_t_min_col, 
                output_t_max_col        = output_t_max_col, 
                make_addtnl_groupby_idx = False, 
                include_summary_path    = include_summary_paths, 
                date_only               = date_only, 
                date_col                = date_col
            )
            return_df = pd.concat([return_df, df_i], ignore_index=True)
        #-------------------------
        # It is possible that a group was split over multiple files/runs
        #   e.g., if not run using slim, then a particular outage/transformer group might have premises split
        #     across neighboring files
        # The method below will combine such entries
        groupby_cols = [x for x in return_df.columns if x not in [output_prem_nbs_col, 'summary_path']]
        cols_shared_by_group=None
        cols_to_collect_in_lists=[output_prem_nbs_col]
        if 'summary_path' in return_df.columns:
            cols_to_collect_in_lists.append('summary_path')
        # NOTE: Make custom_aggs_for_list_cols a dict instead of function so desired functionality is achieved
        #         regardless of presence of 'summary_path'
        custom_aggs_for_list_cols={output_prem_nbs_col : lambda x: list(set(itertools.chain(*x)))}
        #-----
        return_df = Utilities_df.consolidate_df(
            df=return_df, 
            groupby_cols=groupby_cols, 
            cols_shared_by_group=cols_shared_by_group, 
            cols_to_collect_in_lists=cols_to_collect_in_lists, 
            custom_aggs_for_list_cols=custom_aggs_for_list_cols
        )
        return_df = return_df.reset_index()
        #-------------------------
        if make_addtnl_groupby_idx:
            addtnl_groupby_cols = [x for x in return_df.columns 
                                   if x not in [output_prem_nbs_col, output_t_min_col, output_t_max_col, 'summary_path']]
            return_df=return_df.set_index(addtnl_groupby_cols)
        return return_df    
    
    
    @staticmethod
    def get_bsln_time_interval_infos_df_for_data_in_dir(
        data_dir                , 
        file_path_glob          = r'end_events_[0-9]*.csv', 
        output_prem_nbs_col     = 'prem_nbs', 
        output_t_min_col        = 't_min', 
        output_t_max_col        = 't_max', 
        make_addtnl_groupby_idx = True, 
        include_summary_paths   = False, 
        date_only               = False, 
        date_col                = 'aep_event_dt'
    ):
        r"""
        data_dir should point to the directory containing the actual data CSV files.
        It is expected that the summary files live in os.path.join(data_dir, 'summary_files')
        """
        #-------------------------
        assert(os.path.isdir(data_dir))
        data_paths = Utilities.find_all_paths(base_dir=data_dir, glob_pattern=file_path_glob)
        assert(len(data_paths)>0)
        #-----
        summary_paths = [AMIEndEvents.find_summary_file_from_csv(x) for x in data_paths]
        assert(len(summary_paths)>0)
        assert(len(summary_paths)==len(data_paths))
        #-------------------------
        bsln_time_infos_df = MECPOAn.get_bsln_time_interval_infos_df_from_summary_files(
            summary_paths           = summary_paths, 
            output_prem_nbs_col     = output_prem_nbs_col, 
            output_t_min_col        = output_t_min_col, 
            output_t_max_col        = output_t_max_col, 
            make_addtnl_groupby_idx = make_addtnl_groupby_idx, 
            include_summary_paths   = include_summary_paths, 
            date_only               = date_only, 
            date_col                = date_col
        )
        #-------------------------
        return bsln_time_infos_df
    
    
    @staticmethod
    def build_baseline_time_infos_df(
        ede_data_dir     , 
        rename_idxs_dict = None, 
        final_idx_order  = None, 
        min_req_cols     = ['t_min', 't_max'], 
        standardize      = False, 
        date_only        = False, 
        date_col         = 'aep_event_dt'
    ):
        r"""
        rename_idxs_dict:
            If supplied, this must be a dictionary mapping the original index names to the new names.
            The key values must equal the time_infos_df.index.names
        final_idx_order:
            If supplied, the output DF will have it's index re-ordered to match final_idx_order
            **** If some of the time_infos_df.index.names are not contained in final_idx_order, THE CORRESPONDING LEVELS WILL BE DROPPED FROM THE RETURNED DF
            If rename_idxs_dict is supplied:
                ==> final_idx_order and rename_idxs_dict.values() must contain the same elements (not necessarily in the same order though)
            If rename_idxs_dict is not supplied:
                ==> final_idx_order and time_infos_df.index.names loaded in function must contain the same elements (not necessarily in the same order though)
    
        standardize:
            If True, the only columns which are returned will be min_req_cols.
            Note, the index should still contain the grp_by values (e.g., typically trsf_pole_nb and no_outg_rec_nb)
        """
        #-------------------------
        time_infos_df = MECPOAn.get_bsln_time_interval_infos_df_for_data_in_dir(
            data_dir                = ede_data_dir, 
            make_addtnl_groupby_idx = True, 
            include_summary_paths   = False, 
            date_only               = date_only, 
            date_col                = date_col
        )
        #-------------------------
        assert(set(min_req_cols).difference(set(time_infos_df.columns.tolist()))==set())
        #-------------------------
        time_infos_df = Utilities_df.change_idx_names_andor_order(
            df               = time_infos_df, 
            rename_idxs_dict = rename_idxs_dict, 
            final_idx_order  = final_idx_order, 
            inplace          = True
        )
        #-------------------------
        if standardize:
            time_infos_df = time_infos_df[min_req_cols].copy()
        #-------------------------
        return time_infos_df
        
        
    @staticmethod
    def build_outg_time_infos_df_w_dovs(
        rcpx_df                 , 
        outg_rec_nb_idfr        = ('index', 'outg_rec_nb'), 
        dummy_col_levels_prefix = 'dummy_lvl_',     
    ):
        r"""
        Could use build_baseline_time_infos_df, but would need to adjust by the window collection width to find the actual event start
        """
        #-------------------------
        tmp_og_cols = rcpx_df.columns.tolist()
        #-------------------------
        time_infos_df_outg = DOVSOutages.append_outg_dt_off_ts_full_to_df(
            df                      = rcpx_df.copy(), 
            outg_rec_nb_idfr        = outg_rec_nb_idfr, 
            dummy_col_levels_prefix = dummy_col_levels_prefix, 
            include_dt_on_ts        = True
        )
        #-------------------------
        time_info_cols = list(set(time_infos_df_outg.columns.tolist()).difference(set(tmp_og_cols)))
        time_infos_df_outg = time_infos_df_outg[time_info_cols]
        if time_infos_df_outg.columns.nlevels>1:
            assert(time_infos_df_outg.columns.nlevels==2)
            assert(time_infos_df_outg.columns.get_level_values(0).nunique()==1)
            time_infos_df_outg.columns = time_infos_df_outg.columns.droplevel(0)
        #-------------------------
        assert(len(set(['DT_OFF_TS_FULL', 'DT_ON_TS']).difference(set(time_infos_df_outg.columns)))==0)
        time_infos_df_outg = time_infos_df_outg.rename(columns={
            'DT_OFF_TS_FULL':'t_min', 
            'DT_ON_TS':'t_max'
        })
        #-------------------------
        return time_infos_df_outg
        
    def save_time_infos_df(self):
        r"""
        """
        #-------------------------
        if self.time_infos_df is None or self.time_infos_df.shape[0]==0:
            print('In MECPOAn.save_time_infos_df, the self.time_infos_df object is empty, so nothing will be saved!')
            return
        #-------------------------
        assert(os.path.isdir(self.base_dir))
        pkl_path = os.path.join(self.base_dir, 'time_infos_df.pkl')
        self.time_infos_df.to_pickle(pkl_path)    
        
    def load_time_infos_df(self):
        r"""
        """
        #-------------------------
        assert(os.path.isdir(self.base_dir))
        #-----
        pkl_path = os.path.join(self.base_dir, 'time_infos_df.pkl')
        assert(os.path.exists(pkl_path))
        time_infos_df = pd.read_pickle(pkl_path)
        self.time_infos_df = time_infos_df    

    
    def build_time_infos_df(
        self, 
        save_to_pkl = False
    ):
        r"""
        !!!!!!!!!!!!!!!!!!!!!!!!!
        If self.data_type == OutageDType.outg:, this function IS NOT ALLOWED TO SAVE the resultant time_infos_df
            The reason is because the time_infos_df is essentially saved for the MECPOCollection as a whole, but the time_infos_df
              built here is only for the MECPOAn.
            This is an important distinction for OutageDType.outg becuase the time_infos_df is built using DOVS data, and the set of
              groups (i.e., the set of record numbers and transformer pole numbers) is generally only a subset of the MECPOCollection set of groups.
                This is due to the fact that a group will be excluded from a particular MECPOAn if that group does not register any events in the 
                  given time period (or whatever restrictions are placed on the analysis)
            This is not a problem for the baseline data, as their time_infos_df objects are built directly from the EndEvents summary files.
        !!!!!!!!!!!!!!!!!!!!!!!!!
        """
        #-------------------------
        if self.data_type == OutageDType.outg:
            assert('outg_rec_nb' in self.rcpo_df_OG.index.names)
            time_infos_df = MECPOAn.build_outg_time_infos_df_w_dovs(
                rcpx_df                 = self.rcpo_df_OG, 
                outg_rec_nb_idfr        = ('index', 'outg_rec_nb'), 
                dummy_col_levels_prefix = 'dummy_lvl_',     
            )
            save_to_pkl = False
        elif self.data_type == OutageDType.otbl or self.data_type == OutageDType.prbl:
            assert(os.path.isdir(self.end_events_dir))
            time_infos_df = MECPOAn.build_baseline_time_infos_df(
                ede_data_dir     = self.end_events_dir, 
                rename_idxs_dict = None, 
                final_idx_order  = None, 
                min_req_cols     = ['t_min', 't_max'], 
                standardize      = False, 
                date_only        = False, 
                date_col         = 'aep_event_dt'
            )
        else:
            assert(0)
        #-------------------------
        self.time_infos_df = time_infos_df
        #-------------------------
        if save_to_pkl:
            self.save_time_infos_df()
            
    def build_or_load_time_infos_df(
        self, 
        save_to_pkl=False, 
        verbose=True
    ):
        r"""
        If the pickle file os.path.join(self.base_dir, 'time_infos_df.pkl') exists, load it!
        Otherwise, build it
    
        save_to_pkl:
            If True and time_infos_df needed to be built, save.
        """
        #-------------------------
        # If for some reason self.base_dir does not exist, then typically self.end_events_dir will also not exist (since it is typically a
        #   sub-directory of the former).
        # But, I suppose this is not strictly required, so allow for it...
        if(
            self.base_dir is not None and 
            os.path.exists(os.path.join(self.base_dir, 'time_infos_df.pkl'))
        ):
            if verbose:
                print(f"MECPOAn: Loading time_infos_df from {os.path.join(self.base_dir, 'time_infos_df.pkl')}")
            self.load_time_infos_df()
            return
    
        #-------------------------
        # Made it to this point in the code, so the time_infos_df could not be loaded and therefore must be be built!
        if verbose:
            print(f"MECPOAn: No file found at {os.path.join(self.base_dir, 'time_infos_df.pkl')}, \n\tBuilding time_infos_df")
        self.build_time_infos_df(save_to_pkl=save_to_pkl)
        
    #-----------------------------------------------------------------------------------------------------------------------------
    def get_cpo_df_subset_by_mjr_mnr_causes(
        self, 
        cpo_df_name, 
        df_subset_slicers
    ):
        r"""
        If only single subset desired, see get_cpo_df_subset_by_mjr_mnr_cause above
        
        Called get_cpo... because works for rcpo or icpo

        cpo_df_name:
          Needs to be the key value for one of the dfs in rcpo_dfs_coll or icpo_dfs_coll
        """
        #-------------------------
        if self.mjr_mnr_causes_for_outages_df is None:
            self.build_mjr_mnr_causes_for_outages_df(
                include_equip_type=True, 
                set_outg_rec_nb_as_index=True
            )
        assert(isinstance(self.mjr_mnr_causes_for_outages_df, pd.DataFrame) and 
               self.mjr_mnr_causes_for_outages_df.shape[0]>0)
        #-------------------------
        rcpo_df = {**self.rcpo_dfs_coll, **self.icpo_dfs_coll}.get(cpo_df_name, None)
        assert(rcpo_df is not None)
        #-------------------------
        subset_dfs_dict = MECPODf.get_rcpo_df_subset_by_mjr_mnr_causes(
            rcpo_df=rcpo_df, 
            df_subset_slicers=df_subset_slicers, 
            outg_rec_nb_col=self.outg_rec_nb_col, # Only needed for check that all outg_rec_nbs in mjr_mnr_causes_df
            mjr_mnr_causes_df=self.mjr_mnr_causes_for_outages_df
        )
        #-------------------------
        return subset_dfs_dict
        
    def get_cpo_df_subset_by_mjr_mnr_cause(
        self, 
        cpo_df_name, 
        mjr_cause,
        mnr_cause, 
        addtnl_slicers=None, 
        apply_not=False, 
        mjr_cause_col='MJR_CAUSE_CD', 
        mnr_cause_col='MNR_CAUSE_CD'
    ):
        r"""
        addtnl_slicers:
          Can be a list of DataFrameSubsetSingleSlicer, or a list of dict entries, where each dict
            has the approriate keys/values to build a DataFrameSubsetSingleSlicer object.
            At the time this documentation was written, DataFrameSubsetSingleSlicer constructor accepts:
              column (REQUIRED)
              value  (REQUIRED)
              comparison_operator (optional)
          e.g., to reproduce slicer_dl_eqf_xfmr in MECPODf.get_rcpo_df_subset_by_std_mjr_mnr_causes, set:
            mjr_cause='DL'
            mnr_cause='EQF'
            addtnl_slicers=[dict(column='EQUIP_TYP_NM', value=['TRANSFORMER, OH', 'TRANSFORMER, UG'])]
        """
        #-------------------------
        slcr = DFSlicer(
            single_slicers=[
                dict(column=mjr_cause_col, value=mjr_cause), 
                dict(column=mnr_cause_col, value=mnr_cause)
            ], 
            name=None, 
            apply_not=apply_not
        )
        if addtnl_slicers is not None:
            for addtnl in addtnl_slicers:
                slcr.add_single_slicer(addtnl)
        return self.get_cpo_df_subset_by_mjr_mnr_causes(
            cpo_df_name=cpo_df_name, 
            df_subset_slicers=slcr
        )
    
    def get_cpo_df(
        self, 
        cpo_df_name, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        max_total_counts_args=None
    ):
        r"""
        cpo_df_subset_by_mjr_mnr_cause_args:
          Allows one to select subsets of the pd.DataFrames by the outage type.
          This should be a dict with arguments appropriate for the MECPOAn.get_cpo_df_subset_by_mjr_mnr_cause
            function (except for the cpo_df_name argument, which will be set to cpo_df_name)       

        max_total_counts_args:
          Allows one to further select a subset by the maximum number of total counts.
          This should be a dict with arguments appropriate for the MECPODf.get_cpo_df_subset_below_max_total_counts
            function (except for the cpo_df argument, which will be set to return_df
            *** Actually, can also be a simple int/float, which is interpreted as the max_total_counts argument for
                MECPODf.get_cpo_df_subset_below_max_total_counts (all other arguments are set to defaults)

        NOTE: If one wants to perform operations on the DF returned by this function and have those changes be reflected
                in the underlying class attribute DF, it is safest to use the set_cpo_df method.
              If the operations can be done inplace, these should be reflected regardless, otherwise, using the set_cpo_df
                method is necessary!
        """
        #-------------------------
        assert(cpo_df_name in self.cpo_dfs_coll)
        #-------------------------
        if cpo_df_subset_by_mjr_mnr_cause_args is not None:
            return_df = self.get_cpo_df_subset_by_mjr_mnr_cause(
                cpo_df_name=cpo_df_name, 
                **cpo_df_subset_by_mjr_mnr_cause_args
            )
        else:
            if cpo_df_name in self.rcpo_dfs_coll:
                return_df = self.rcpo_dfs_coll[cpo_df_name]
            else:
                return_df = self.icpo_dfs_coll[cpo_df_name]
        #-------------------------
        if max_total_counts_args is not None:
            assert(Utilities.is_object_one_of_types(max_total_counts_args, [dict, int, float]))
            if not isinstance(max_total_counts_args, dict):
                max_total_counts_args = dict(max_total_counts=max_total_counts_args)
            return_df = MECPODf.get_cpo_df_subset_below_max_total_counts(
                cpo_df=return_df, 
                **max_total_counts_args
            )
        #-------------------------
        return return_df        


    def get_cpo_df_subset_by_std_mjr_mnr_causes(
        self, 
        cpo_df_name, 
        addtnl_df_subset_slicers=None
    ):
        r"""
        Called get_cpo... because works for rcpo or icpo

        cpo_df_name:
          Needs to be the key value for one of the dfs in rcpo_dfs_coll or icpo_dfs_coll
        """
        #-------------------------
        if self.mjr_mnr_causes_for_outages_df is None:
            self.build_mjr_mnr_causes_for_outages_df(
                include_equip_type=True, 
                set_outg_rec_nb_as_index=True
            )
        assert(isinstance(self.mjr_mnr_causes_for_outages_df, pd.DataFrame) and 
               self.mjr_mnr_causes_for_outages_df.shape[0]>0)
        #-------------------------
        rcpo_df = {**self.rcpo_dfs_coll, **self.icpo_dfs_coll}.get(cpo_df_name, None)
        assert(rcpo_df is not None)
        #-------------------------
        subset_dfs_dict = MECPODf.get_rcpo_df_subset_by_std_mjr_mnr_causes(
            rcpo_df=rcpo_df, 
            addtnl_df_subset_slicers=addtnl_df_subset_slicers, 
            outg_rec_nb_col=self.outg_rec_nb_col, # Only needed for check that all outg_rec_nbs in mjr_mnr_causes_df
            mjr_mnr_causes_df=self.mjr_mnr_causes_for_outages_df
        )
        #-------------------------
        return subset_dfs_dict
        
        
    def remove_SNs_cols_from_all_cpo_dfs(
        self, 
        SNs_tags=None, 
        include_cpo_df_OG=False,
        include_cpo_df_raw=False,
        cpo_dfs_to_ignore=None, 
        is_long=False
    ):
        r"""
        Remove the SNs cols from all cpo_dfs in MECPOAn.
        By default, the SNs cols are NOT REMOVED FROM: rcpo_df_OG, rcpo_df_raw, and icpo_df_raw
          unless include_cpo_df_OG or include_cpo_df_OG is True

        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        NOTE: SNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled in MECPODf.remove_SNs_cols_from_rcpo_df.

        include_cpo_df_OG:
          If True, also remove SNs from rcpo_df_OG

        include_cpo_df_raw:
          If True, also remove SNs from rcpo_df_raw and icpo_df_raw

        cpo_dfs_to_ignore:
          List of cpo_df names.  The SNs cols will not be removed from any of these DFs
        """
        #-------------------------
        if cpo_dfs_to_ignore is None:
            cpo_dfs_to_ignore=[]
        assert(Utilities.is_object_one_of_types(cpo_dfs_to_ignore, [list, tuple]))
        #-----
        # include_cpo_df_OG means include in removal.  So, if False (default), exclude from removal
        if not include_cpo_df_OG:
            cpo_dfs_to_ignore.append('rcpo_df_OG')
        # include_cpo_df_raw means include in removal.  So, if False (default), exclude from removal
        if not include_cpo_df_raw:
            cpo_dfs_to_ignore.extend(['rcpo_df_raw', 'icpo_df_raw'])
        #-------------------------
        for cpo_df_name in self.get_all_cpo_df_names(non_empty_only=True):
            if cpo_df_name in cpo_dfs_to_ignore:
                continue
            cpo_df = self.get_cpo_df(cpo_df_name=cpo_df_name)
            cpo_df = MECPODf.remove_SNs_cols_from_rcpo_df(
                rcpo_df=cpo_df, 
                SNs_tags=SNs_tags, 
                is_long=is_long
            )
            self.set_cpo_df(
                cpo_df_name=cpo_df_name, 
                df=cpo_df
            )
    
    
    @staticmethod
    def make_cpo_columns_equal(
        mecpo_coll, 
        same_order=True, 
        cols_to_init_with_empty_lists=MECPODf.std_SNs_cols(), 
        drop_empty_cpo_dfs=False
    ):
        r"""
        mecpo_coll is a list of MECPOAn objects

        If init_cpo_dfs_to is None (not pd.DataFrame), drop_empty_cpo_dfs will need to be set to True
          if any of the members in mecpo_coll have empty cpo dfs (e.g., for the case of no outages, typically
          the norm_by_outg_nSNs and norm_by_prim_nSNs DataFrames are empty.)

        NOTE: Operations are done in place (i.e., no inplace=False option)
        """
        #-------------------------
        assert(isinstance(mecpo_coll, list))
        assert(all([isinstance(x, MECPOAn) for x in mecpo_coll]))
        #-------------------------
        cpo_dfs_dicts = []
        for mecpo_an in mecpo_coll:
            if drop_empty_cpo_dfs:
                mecpo_an.drop_empty_cpo_dfs()
            cpo_dfs_dicts.append({**mecpo_an.rcpo_dfs_coll, **mecpo_an.icpo_dfs_coll})
        #-------------------------
        AMIEndEvents.make_reason_counts_per_outg_columns_equal_list_of_dicts_of_dfs(
            rcpo_dfs_dicts=cpo_dfs_dicts, 
            df_key_tags_to_ignore=None, 
            same_order=same_order, 
            cols_to_init_with_empty_lists=cols_to_init_with_empty_lists
        )    

            
    def remove_reasons_explicit_from_all_rcpo_dfs(
        self, 
        reasons_to_remove
    ):
        r"""
        Called _explicit because exact columns to removed must be explicitly given.  
        For a more flexible version, see remove_reasons_from_all_rcpo_dfs
        
        reasons_to_remove:
          Should be a list of strings.  If a given df has MultiIndex columns, this will be handled.
          Reasons are only removed if they exist (obviously)
        """
        for rcpo_df_name in self.rcpo_dfs_coll.keys():
            rcpo_df_i = self.rcpo_dfs_coll[rcpo_df_name]
            rcpo_df_i = MECPODf.remove_reasons_explicit_from_rcpo_df(
                rcpo_df=rcpo_df_i, 
                reasons_to_remove=reasons_to_remove
            )
            self.rcpo_dfs_coll[rcpo_df_name] = rcpo_df_i
            
            
    def remove_reasons_from_all_rcpo_dfs(
        self, 
        regex_patterns_to_remove, 
        ignore_case=True
    ):
        r"""
        For each DF in rcpo_dfs, remove any columns from rcpo_df where any of the patterns in 
          regex_patterns_to_remove are found
        
        regex_patterns_to_remove:
          Should be a list of regex patterns (i.e., strings)
        """
        for rcpo_df_name in self.rcpo_dfs_coll.keys():
            rcpo_df_i = self.rcpo_dfs_coll[rcpo_df_name]
            rcpo_df_i = MECPODf.remove_reasons_from_rcpo_df(
                rcpo_df=rcpo_df_i, 
                regex_patterns_to_remove=regex_patterns_to_remove, 
                ignore_case=ignore_case
            )
            self.rcpo_dfs_coll[rcpo_df_name] = rcpo_df_i
            
    def get_counts_series(
        self, 
        cpo_df_name, 
        include_list_col=False
    ):
        r"""
        include_list_col:
          If True, include the corresponding list column.  The number of counts for a given entry is equal to the
            length of the list (e.g., if counts_col = '_nSNs', list_col = '_SNs')
          If True, the returned object will be a pd.DataFrame, not a pd.Series
        """
        #-------------------------
        # First, grab is_norm and counts_col from class methods using cpo_df_name
        is_norm                    = self.is_norm_dict[cpo_df_name]
        counts_col                 = self.cpo_df_name_to_norm_counts_col_dict[cpo_df_name] 
        list_col                   = self.cpo_df_name_to_norm_list_col_dict[cpo_df_name] 
        #-------------------------
        # If norm_counts_df is not already built, attempt to build it
        if self.norm_counts_df is None or self.norm_counts_df.shape[0]==0:
            self.build_and_set_norm_counts_df(
                remove_cols_from_dfs=False
            )
        #-------------------------
        cols_to_return = [counts_col]
        assert(counts_col in self.norm_counts_df.columns)
        if include_list_col:
            assert(list_col in self.norm_counts_df.columns)
            cols_to_return.append(list_col)
        #-------------------------
        return self.norm_counts_df[cols_to_return]
            
            
            
    def add_counts_col_to_df(
        self, 
        cpo_df_name, 
        include_list_col=False
    ):
        r"""
        In some instances, the counts_col was stripped from a cpo_df.
        If the counts_col is again needed, this function will add it back.
        The counts_col will simply be grabbed from rcpo_df_OG or rcpo_df_raw
        
        CURRENTLY THIS ONLY WORKS FOR DFs with columns.nlevels==1!
          This is mainly because counts_col returns a single string
          I suppose, for example with rcpo_df_OG, I could return a tuple,
            and then run this recursively.  BUT, rcpo_df_OG is currently the only 
            case, and it should always have the counts cols
        
        include_list_col:
          If True, include the corresponding list column.  The number of counts for a given entry is equal to the
            length of the list (e.g., if counts_col = '_nSNs', list_col = '_SNs')
        
        NOTE: If the counts_col already exists, this function will do nothing
        NOTE: If cpo_df_name==rcpo_df_OG or cpo_df_name==(r)(i)cpo_raw, this does nothing
              (as counts_col is None in the latter)
        """
        if cpo_df_name=='rcpo_df_OG':
            return
        #-------------------------
        # First, grab is_norm and counts_col from class methods using cpo_df_name
        is_norm                    = self.is_norm_dict[cpo_df_name]
        counts_col                 = self.cpo_df_name_to_norm_counts_col_dict[cpo_df_name] 
        list_col                   = self.cpo_df_name_to_norm_list_col_dict[cpo_df_name] 
        
        if counts_col is None:
            return
        #-------------------------
        rcpo_df = self.cpo_dfs[cpo_df_name]
        assert(rcpo_df.columns.nlevels==1)
        #-------------------------
        # If the counts_col already exists, then simply return
        if counts_col in rcpo_df.columns:
            if not include_list_col:
                return 
            else:
                if list_col in rcpo_df.columns:
                    return
        #-------------------------
        # If norm_counts_df is not already built, attempt to build it
        if self.norm_counts_df is None or self.norm_counts_df.shape[0]==0:
            self.build_and_set_norm_counts_df(
                remove_cols_from_dfs=False
            )
        #-------------------------
        org_shape = rcpo_df.shape
        cols_to_merge = [counts_col]
        assert(counts_col in self.norm_counts_df.columns)
        if include_list_col:
            assert(list_col in self.norm_counts_df.columns)
            cols_to_merge.append(list_col)
        #-----
        # Note: If the desire is to add list_col to rcpo_df, but counts_col already contained, then
        #       counts_col must be removed from cols_to_merge, otherwise one will end up with 
        #       counts_col_x and counts_col_y in the final DF!
        #       To be safe, I'll also check for presence of list_col
        if counts_col in rcpo_df.columns:
            cols_to_merge.remove(counts_col)
        if list_col in rcpo_df.columns:
            cols_to_merge.remove(list_col)
        assert(len(cols_to_merge)>0)
        #-----
        rcpo_df = pd.merge(rcpo_df, self.norm_counts_df[cols_to_merge], left_index=True, right_index=True, how='inner')
        assert(rcpo_df.shape[1]==org_shape[1]+len(cols_to_merge))
        assert(rcpo_df.shape[0]==org_shape[0])
        #-------------------------
        self.set_cpo_df(cpo_df_name, rcpo_df)        
        
        
    def add_counts_col_to_all_rcpo_dfs(
        self, 
        include_list_col=False
    ):
        r"""
        For each rcpo_df in each MECPOAn, add counts col (and possibly list col), as determined by 
          MECPOAn.cpo_df_name_to_norm_counts_col_dict
        """
        #-------------------------
        for rcpo_df_name in self.get_all_cpo_df_names(non_empty_only=True):
            self.add_counts_col_to_df(
                cpo_df_name=rcpo_df_name, 
                include_list_col=include_list_col
            ) 
            
            
    def combine_cpo_df_reasons(
        self, 
        cpo_df_name, 
        patterns_and_replace=None, 
        addtnl_patterns_and_replace=None, 
        initial_strip=True,
        initial_punctuation_removal=True, 
        level_0_raw_col='counts', 
        level_0_nrm_col='counts_norm', 
        return_red_to_org_cols_dict=False
    ):
        r"""
        Combine groups of reasons in DF identified by cpo_df_name according to patterns_and_replace.
        """
        #-------------------------
        # First, grab is_norm, normalize_by_nSNs_included, and counts_col from class methods using cpo_df_name
        is_norm                    = self.is_norm_dict[cpo_df_name]
        normalize_by_nSNs_included = self.is_normalize_by_nSNs_included[cpo_df_name]
        counts_col                 = self.cpo_df_name_to_norm_counts_col_dict[cpo_df_name]
        #-------------------------
        # NOTE: Since MECPODf.combine_cpo_df_reasons is not an inplace operation (and does not even have an inplace option)
        #       the set_cpo_df must be used at end!
        #-------------------------
        # Below, rcpo_df_OG should always have the counts_col
        # Raw Dfs (those where is_norm==False) don't need counts_col for the combine operation (and, in fact,
        #   counts_col=None for these)
        # All others must have counts_col for combine operation, so make sure this is so!
        #-------------------------
        # If any counts cols are added, they should be removed at the end.  Grabbing the columns here
        #   makes the identification of these columns much easier.
        org_cols = self.cpo_dfs[cpo_df_name].columns
        if cpo_df_name=='rcpo_df_OG':
            assert((level_0_raw_col, counts_col) in self.cpo_dfs[cpo_df_name].columns and 
                   (level_0_nrm_col, counts_col) in self.cpo_dfs[cpo_df_name].columns)
        elif is_norm:
            if counts_col not in self.cpo_dfs[cpo_df_name].columns:
                self.add_counts_col_to_df(
                    cpo_df_name, 
                    include_list_col=False
                )
        else:
            pass
        cols_added = list(set(self.cpo_dfs[cpo_df_name].columns).difference(set(org_cols)))
        #-------------------------
        rcpo_df = self.cpo_dfs[cpo_df_name]
        #-------------------------
        rcpo_df = MECPODf.combine_cpo_df_reasons(
            rcpo_df=rcpo_df, 
            patterns_and_replace=patterns_and_replace, 
            addtnl_patterns_and_replace=addtnl_patterns_and_replace, 
            is_norm=is_norm, 
            counts_col=counts_col, 
            normalize_by_nSNs_included=normalize_by_nSNs_included, 
            initial_strip=initial_strip,
            initial_punctuation_removal=initial_punctuation_removal, 
            level_0_raw_col=level_0_raw_col, 
            level_0_nrm_col=level_0_nrm_col, 
            return_red_to_org_cols_dict=return_red_to_org_cols_dict
        )
        if return_red_to_org_cols_dict:
            # Order of commands below matters!
            red_to_org_cols_dict = rcpo_df[1]
            rcpo_df              = rcpo_df[0]
        #-------------------------
        if len(cols_added)>0:
            rcpo_df = rcpo_df.drop(columns=cols_added)
        #-------------------------
        self.set_cpo_df(cpo_df_name, rcpo_df)
        #-------------------------
        if return_red_to_org_cols_dict:
            return red_to_org_cols_dict

    def combine_reasons_in_all_rcpo_dfs(
        self, 
        patterns_and_replace=None, 
        addtnl_patterns_and_replace=None, 
        initial_strip=True,
        initial_punctuation_removal=True, 
        level_0_raw_col='counts', 
        level_0_nrm_col='counts_norm', 
        return_red_to_org_cols_dict=False
    ):
        r"""
        Combine groups of reasons in all rcpo_dfs according to patterns_and_replace.
        NOTE: Specifically for rcpo_dfs, as icpo_dfs will have different columns
              (rcpo_dfs have reasons, icpo_dfs have enddeviceeventtypeids)
              
        NOTE!!!!!!!!!!!!!:
          Typically, one should keep patterns_and_replace=None.  When this is the case, dflt_patterns_and_replace
            will be used.
          If one wants to add to dflt_patterns_and_replace, use the addtnl_patterns_and_replace argument.

        patterns_and_replace/addtnl_patterns_and_replace:
          A list of tuples (or lists) of length 2.
          For each item in the list:
            first element should be a regex pattern for which to search 
            second element is replacement

            DON'T FORGET ABOUT BACKREFERENCING!!!!!
              e.g.
                reason_i = 'I am a string named Jesse Thomas Buxton'
                patterns_and_replace_i = (r'(Jesse) (Thomas) (Buxton)', r'\1 \3')
                  reason_i ===> 'I am a string named Jesse Buxton'              
        """
        #-------------------------
        # NOTE: If return_red_to_org_cols_dict==False, then self.combine_cpo_df_reasons will return None,
        #       and red_to_org_cols_dicts will be a dict with all values equal to None (so, no harm in building regardless)
        red_to_org_cols_dicts={}
        cpo_df_names = self.get_all_cpo_df_names(non_empty_only=True)
        for rcpo_df_name in self.rcpo_dfs_coll.keys():
            if rcpo_df_name not in cpo_df_names:
                continue
            red_to_org_cols_dict = self.combine_cpo_df_reasons(
                cpo_df_name=rcpo_df_name, 
                patterns_and_replace=patterns_and_replace, 
                addtnl_patterns_and_replace=addtnl_patterns_and_replace, 
                initial_strip=initial_strip,
                initial_punctuation_removal=initial_punctuation_removal, 
                level_0_raw_col=level_0_raw_col, 
                level_0_nrm_col=level_0_nrm_col, 
                return_red_to_org_cols_dict=return_red_to_org_cols_dict
            ) 
            red_to_org_cols_dicts[rcpo_df_name] = red_to_org_cols_dict
        if return_red_to_org_cols_dict:
            return red_to_org_cols_dicts
            
    def delta_cpo_df_reasons(
        self, 
        cpo_df_name, 
        reasons_1,
        reasons_2,
        delta_reason_name, 
        level_0_raw_col='counts', 
        level_0_nrm_col='counts_norm'
    ):
        r"""
        Combine groups of reasons in DF identified by cpo_df_name according to patterns_and_replace.
        """
        #-------------------------
        # First, grab is_norm, normalize_by_nSNs_included, and counts_col from class methods using cpo_df_name
        is_norm                    = self.is_norm_dict[cpo_df_name]
        normalize_by_nSNs_included = self.is_normalize_by_nSNs_included[cpo_df_name]
        counts_col                 = self.cpo_df_name_to_norm_counts_col_dict[cpo_df_name]
        #-------------------------
        # NOTE: Since MECPODf.combine_cpo_df_reasons is not an inplace operation (and does not even have an inplace option)
        #       the set_cpo_df must be used at end!
        #-------------------------
        # Below, rcpo_df_OG should always have the counts_col
        # Raw Dfs (those where is_norm==False) don't need counts_col for the combine operation (and, in fact,
        #   counts_col=None for these)
        # All others must have counts_col for combine operation, so make sure this is so!
        if cpo_df_name=='rcpo_df_OG':
            assert((level_0_raw_col, counts_col) in self.cpo_dfs[cpo_df_name].columns and 
                   (level_0_nrm_col, counts_col) in self.cpo_dfs[cpo_df_name].columns)
        elif is_norm:
            if counts_col not in self.cpo_dfs[cpo_df_name].columns:
                self.add_counts_col_to_df(
                    cpo_df_name, 
                    include_list_col=False
                )
        else:
            pass
        rcpo_df = self.cpo_dfs[cpo_df_name]
        #-------------------------
        rcpo_df = MECPODf.delta_cpo_df_reasons(
            rcpo_df=rcpo_df, 
            reasons_1=reasons_1,
            reasons_2=reasons_2,
            delta_reason_name=delta_reason_name, 
            is_norm=is_norm, 
            counts_col=counts_col, 
            normalize_by_nSNs_included=normalize_by_nSNs_included, 
            level_0_raw_col=level_0_raw_col, 
            level_0_nrm_col=level_0_nrm_col           
        )
        self.set_cpo_df(cpo_df_name, rcpo_df)
        
    def delta_cpo_df_reasons_in_all_rcpo_dfs(
        self,  
        reasons_1,
        reasons_2,
        delta_reason_name, 
        level_0_raw_col='counts', 
        level_0_nrm_col='counts_norm'
    ):
        r"""
        Combine groups of reasons in all rcpo_dfs according to patterns_and_replace.
        """
        #-------------------------
        for rcpo_df_name in self.rcpo_dfs_coll.keys():
            self.delta_cpo_df_reasons(
                cpo_df_name=rcpo_df_name, 
                reasons_1=reasons_1,
                reasons_2=reasons_2,
                delta_reason_name=delta_reason_name, 
                level_0_raw_col=level_0_raw_col, 
                level_0_nrm_col=level_0_nrm_col
            ) 
            
    def get_total_event_counts(
        self, 
        cpo_df_name, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        output_col='total_counts', 
        SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols(), 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        cpo_df_name:
          Needs to be the key value for one of the dfs in rcpo_dfs_coll or icpo_dfs_coll

        cpo_df_subset_by_mjr_mnr_cause_args:
          Allows one to select subsets of the pd.DataFrames by the outage type.
          This should be a dict with arguments appropriate for the MECPOAn.get_cpo_df_subset_by_mjr_mnr_cause
            function (except for the cpo_df_name argument, which will be set to cpo_df_name)
            
        The nSNs (and SNs) columns should not be included in the count, hence the need for the SNs tags argument.
        If normalize_by_nSNs_included==True, the return DF with have MultiIndex level_0 values matching those of cpo_df
          (i.e., with level_0_raw_col and level_0_nrm_col)
        """
        #-------------------------
        assert(cpo_df_name in self.cpo_dfs.keys())
        #-------------------------
        if cpo_df_subset_by_mjr_mnr_cause_args is None:
            cpo_df = self.get_cpo_df(cpo_df_name=cpo_df_name)
        else:
            cpo_df = self.get_cpo_df_subset_by_mjr_mnr_cause(
                cpo_df_name=cpo_df_name, 
                **cpo_df_subset_by_mjr_mnr_cause_args
            )
        #-------------------------
        total_df = MECPODf.get_total_event_counts(
            cpo_df=cpo_df, 
            output_col=output_col, 
            SNs_tags=SNs_tags, 
            normalize_by_nSNs_included=normalize_by_nSNs_included, 
            level_0_raw_col = level_0_raw_col, 
            level_0_nrm_col = level_0_nrm_col
        )
        #-------------------------
        return total_df
        
    def get_top_reasons_subset_from_cpo_df(
        self, 
        cpo_df_name, 
        n_reasons_to_include=10,
        combine_others=True, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        max_total_counts_args=None, 
        output_combine_others_col='Other Reasons', 
        SNs_tags=None, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        """
        #-------------------------
        is_norm    = self.is_norm_dict[cpo_df_name]
        counts_col = self.cpo_df_name_to_norm_counts_col_dict[cpo_df_name]     
        #-------------------------
        cpo_df = self.get_cpo_df(
            cpo_df_name=cpo_df_name, 
            cpo_df_subset_by_mjr_mnr_cause_args=cpo_df_subset_by_mjr_mnr_cause_args, 
            max_total_counts_args=max_total_counts_args
        )
        #-------------------------
        # NOT INTENDED FOR rcpo_df_OG!
        assert(cpo_df.columns.nlevels==1)
        #-------------------------
        if counts_col not in cpo_df.columns:
            counts_series = self.get_counts_series(
                cpo_df_name=cpo_df_name, 
                include_list_col=False
            )
            assert(len(set(cpo_df.index).difference(set(counts_series.index)))==0)
            cpo_df = pd.merge(cpo_df, counts_series.to_frame(), left_index=True, right_index=True, how='inner')
        assert(counts_col in cpo_df.columns)    
        #-------------------------
        cpo_df = MECPODf.get_top_reasons_subset_from_cpo_df(
            cpo_df=cpo_df, 
            n_reasons_to_include=n_reasons_to_include,
            combine_others=combine_others, 
            output_combine_others_col=output_combine_others_col, 
            SNs_tags=SNs_tags, 
            is_norm=is_norm, 
            counts_col=counts_col, 
            normalize_by_nSNs_included=False, 
            level_0_raw_col = level_0_raw_col, 
            level_0_nrm_col = level_0_nrm_col       
        )    
        #-------------------------
        return cpo_df


    def get_reasons_subset_from_cpo_df(
        self, 
        cpo_df_name, 
        reasons_to_include,
        combine_others=True, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        max_total_counts_args=None, 
        output_combine_others_col='Other Reasons', 
        SNs_tags=None, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        """
        #-------------------------
        is_norm    = self.is_norm_dict[cpo_df_name]
        counts_col = self.cpo_df_name_to_norm_counts_col_dict[cpo_df_name]     
        #-------------------------
        cpo_df = self.get_cpo_df(
            cpo_df_name=cpo_df_name, 
            cpo_df_subset_by_mjr_mnr_cause_args=cpo_df_subset_by_mjr_mnr_cause_args, 
            max_total_counts_args=max_total_counts_args
        )
        #-------------------------
        # NOT INTENDED FOR rcpo_df_OG!
        assert(cpo_df.columns.nlevels==1)
        #-------------------------
        if counts_col not in cpo_df.columns:
            counts_series = self.get_counts_series(
                cpo_df_name=cpo_df_name, 
                include_list_col=False
            )
            assert(len(set(cpo_df.index).difference(set(counts_series.index)))==0)
            cpo_df = pd.merge(cpo_df, counts_series.to_frame(), left_index=True, right_index=True, how='inner')
        assert(counts_col in cpo_df.columns)    
        #-------------------------
        cpo_df = MECPODf.get_reasons_subset_from_cpo_df(
            cpo_df=cpo_df, 
            reasons_to_include=reasons_to_include,
            combine_others=combine_others, 
            output_combine_others_col=output_combine_others_col, 
            SNs_tags=SNs_tags, 
            is_norm=is_norm, 
            counts_col=counts_col, 
            normalize_by_nSNs_included=False, 
            level_0_raw_col = level_0_raw_col, 
            level_0_nrm_col = level_0_nrm_col       
        )    
        #-------------------------
        return cpo_df
        
        
        
    def remove_all_cpo_dfs_except(
        self, 
        to_keep, 
        keep_rcpo_df_OG=True
    ):
        r"""
        By remove, I mean set to self.init_cpo_dfs_to (which is typically an empty DF)

        to_keep:
            Should be the name of a DF to keep, or a list of names of DFs to keep.
        """
        #-------------------------
        if not isinstance(to_keep, list):
            to_keep = [to_keep]
        #-------------------------
        # Typically want to keep rcpo_df_OG!
        if keep_rcpo_df_OG and 'rcpo_df_OG' not in to_keep:
            to_keep = ['rcpo_df_OG'] + to_keep
        #-------------------------
        # Make sure the DFs in to_keep exist (and are non_empty)
        for name in to_keep:
            assert(name in self.get_all_cpo_df_names(non_empty_only=True))
        #-----
        to_remove = list(set(self.get_all_cpo_df_names(non_empty_only=True)).difference(set(to_keep)))
        for name in to_remove:
            self.set_cpo_df(name, copy.deepcopy(self.init_cpo_dfs_to))
            

            
    @staticmethod
    def combine_two_mecpo_ans(
        mecpo_an_1, 
        mecpo_an_2, 
        append_only=True
    ):
        r"""
        NOTE: Case where append_only==False should work, but has not been super scrutinized and is somewhat hacky

        If the datasets are mutually exclusive, append_only should be set to True.
            In this case, having append_only=True or False should give the same result, but having it set to
            True will run much faster
        """
        #--------------------------------------------------
        # Get all cpo df names as the union of those found in mecpo_an_1 and mecpo_an_2
        all_cpo_df_names = list(set(mecpo_an_1.get_all_cpo_df_names()).union(set(mecpo_an_2.get_all_cpo_df_names())))
        #--------------------------------------------------
        # Build input_dict and use load_dfs_from_dict
        load_dfs_dict = {}
        #--------------------------------------------------
        # First, build all cpo_dfs
        for cpo_df_name_i in all_cpo_df_names:
            assert(cpo_df_name_i not in load_dfs_dict.keys())
            cpo_df_1_i = mecpo_an_1.get_cpo_df(cpo_df_name_i)
            cpo_df_2_i = mecpo_an_2.get_cpo_df(cpo_df_name_i)
            #-----
            is_empty_1 = cpo_df_1_i is None or cpo_df_1_i.shape[0]==0
            is_empty_2 = cpo_df_2_i is None or cpo_df_2_i.shape[0]==0    
            #-------------------------
            # Handle cases where one or both are empty
            #-----
            # If both are empty, set load_dfs_dict[cpo_df_name_i] to empty as dictated by 
            #   mecpo_an_1.init_cpo_dfs_to which should either be pd.DataFrame() or None
            if is_empty_1 and is_empty_2:
                load_dfs_dict[cpo_df_name_i] = mecpo_an_1.init_cpo_dfs_to
                continue
            #-----
            # If one is empty, set load_dfs_dict[cpo_df_name_i] to the other
            if is_empty_1 and not is_empty_2:
                load_dfs_dict[cpo_df_name_i] = cpo_df_2_i
                continue
            if is_empty_2 and not is_empty_1:
                load_dfs_dict[cpo_df_name_i] = cpo_df_1_i
                continue
            #-------------------------
            # At this point, both cpo_df_1_i and cpo_df_2_i are not empty
            if append_only:
                cpo_df_1_i, cpo_df_2_i = AMIEndEvents.make_reason_counts_per_outg_columns_equal(cpo_df_1_i, cpo_df_2_i, same_order=True, inplace=False)
                cpo_df_12_i = pd.concat([cpo_df_1_i, cpo_df_2_i])
                cpo_df_12_i = cpo_df_12_i[cpo_df_12_i.columns.sort_values()]
            else:
                # First, grab is_norm, normalize_by_nSNs_included, counts_col and list_col from class methods using cpo_df_name
                is_norm                    = mecpo_an_1.is_norm_dict[cpo_df_name_i]
                normalize_by_nSNs_included = mecpo_an_1.is_normalize_by_nSNs_included[cpo_df_name_i]
                counts_col                 = mecpo_an_1.cpo_df_name_to_norm_counts_col_dict[cpo_df_name_i] 
                list_col                   = mecpo_an_1.cpo_df_name_to_norm_list_col_dict[cpo_df_name_i] 

                # Make sure mecpo_an_2 agrees
                assert(is_norm                    == mecpo_an_2.is_norm_dict[cpo_df_name_i])
                assert(normalize_by_nSNs_included == mecpo_an_2.is_normalize_by_nSNs_included[cpo_df_name_i])
                assert(counts_col                 == mecpo_an_2.cpo_df_name_to_norm_counts_col_dict[cpo_df_name_i])
                assert(list_col                   == mecpo_an_2.cpo_df_name_to_norm_list_col_dict[cpo_df_name_i])

                # Make sure DFs have counts columns, necessary for combine below
                #TODO: IDEALLY DONT WANT TO HAVE TO INCLUDE LIST COL
                mecpo_an_1.add_counts_col_to_df(cpo_df_name_i, include_list_col=True)
                mecpo_an_2.add_counts_col_to_df(cpo_df_name_i, include_list_col=True)
                # Need to grab updated DFs
                cpo_df_1_i = mecpo_an_1.get_cpo_df(cpo_df_name_i)
                cpo_df_2_i = mecpo_an_2.get_cpo_df(cpo_df_name_i)

        #         # The raw dfs typically house all sorts of normalization information, but since they're raw
        #         #   counts_col and list_col will come back as None.
        #         # These need to be properly combined, so must be included
        #         if counts_col is None or list_col is None:
        #             assert(counts_col is None and list_col is None)
        #             assert(cpo_df_name_i=='rcpo_df_raw' or cpo_df_name_i=='icpo_df_raw')
        #             list_col, counts_col = mecpo_an_1.identify_lists_and_counts_cols_in_df(cpo_df_1_i)
        #             assert((list_col, counts_col)==mecpo_an_2.identify_lists_and_counts_cols_in_df(cpo_df_2_i))
                #-----
                list_cols = list_col
                list_counts_cols = counts_col
                if not isinstance(list_cols, list):
                    list_cols = [list_cols]
                if not isinstance(list_counts_cols, list):
                    list_counts_cols = [list_counts_cols]
                #-----
                list_cols_in_df, counts_cols_in_df = mecpo_an_1.identify_lists_and_counts_cols_in_df(cpo_df_1_i)
                assert((list_cols_in_df, counts_cols_in_df)==mecpo_an_2.identify_lists_and_counts_cols_in_df(cpo_df_2_i))
                addtnl_list_cols = list(set(list_cols_in_df).difference(set(list_cols)))
                addtnl_counts_cols = list(set(counts_cols_in_df).difference(set(list_counts_cols)))
                if len(addtnl_list_cols)>0 or len(addtnl_counts_cols)>0:
                    assert(cpo_df_name_i=='rcpo_df_OG' or cpo_df_name_i=='rcpo_df_raw' or cpo_df_name_i=='icpo_df_raw')
                    list_cols.extend(addtnl_list_cols)
                    list_counts_cols.extend(addtnl_counts_cols)
                list_cols = [x for x in list_cols if x is not None]
                list_counts_cols = [x for x in list_counts_cols if x is not None]
                #-----
                cpo_df_12_i = AMIEndEvents.combine_two_reason_counts_per_outage_dfs(
                    rcpo_df_1=cpo_df_1_i, 
                    rcpo_df_2=cpo_df_2_i, 
                    are_dfs_wide_form=True, 
                    normalize_by_nSNs_included=normalize_by_nSNs_included, 
                    is_norm=is_norm, 
                    list_cols=list_cols, 
                    list_counts_cols=list_counts_cols
                )
            load_dfs_dict[cpo_df_name_i] = cpo_df_12_i
        #--------------------------------------------------
        # Now, build ede_typeid_to_reason_df and reason_to_ede_typeid_df
        ede_typeid_to_reason_df_12 = AMIEndEvents.combine_two_ede_typeid_to_reason_dfs(
            ede_typeid_to_reason_df1=mecpo_an_1.ede_typeid_to_reason_df, 
            ede_typeid_to_reason_df2=mecpo_an_2.ede_typeid_to_reason_df, 
            sort=False
        )
        assert('ede_typeid_to_reason_df' not in load_dfs_dict.keys())
        load_dfs_dict['ede_typeid_to_reason_df'] = ede_typeid_to_reason_df_12
        #-----
        reason_to_ede_typeid_df_12 = AMIEndEvents.invert_ede_typeid_to_reason_df(ede_typeid_to_reason_df_12)
        assert('reason_to_ede_typeid_df' not in load_dfs_dict.keys())
        load_dfs_dict['reason_to_ede_typeid_df'] = reason_to_ede_typeid_df_12
        #--------------------------------------------------
        # Alert user if there are no unexpected DFs
        expected_dfs = [
            'rcpo_df_OG', 
            'ede_typeid_to_reason_df', 
            'reason_to_ede_typeid_df', 

            'rcpo_df_raw', 
            'rcpo_df_norm', 
            'rcpo_df_norm_by_xfmr_nSNs',
            'rcpo_df_norm_by_outg_nSNs',
            'rcpo_df_norm_by_prim_nSNs', 

            'icpo_df_raw', 
            'icpo_df_norm', 
            'icpo_df_norm_by_xfmr_nSNs', 
            'icpo_df_norm_by_outg_nSNs', 
            'icpo_df_norm_by_prim_nSNs'    
        ]
        unexpected_dfs = list(set(load_dfs_dict.keys()).difference(set(expected_dfs)))
        if len(unexpected_dfs)>0:
            print(f'In combine_two_mecpo_ans, unexpected dfs found, which will not be added to output MECPOAn:')
            print(unexpected_dfs)
        #--------------------------------------------------
        # Initiate mecpo_an_comb and load DFs into it
        mecpo_an_comb = MECPOAn(None)
        mecpo_an_comb.load_dfs_from_dict(load_dfs_dict)
        #--------------------------------------------------
        return mecpo_an_comb