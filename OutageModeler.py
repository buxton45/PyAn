#!/usr/bin/env python

r"""
Holds OutageModeler class.  See OutageModeler.OutageModeler for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import re
import json
import pickle
import joblib

import pandas as pd
import numpy as np
import datetime
import time
from natsort import natsorted

import copy
import itertools

#---------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing



#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from EEMSP import EEMSP
#-----
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from MECPODf import MECPODf
from MECPOAn import MECPOAn
from MECPOCollection import MECPOCollection
from DOVSOutages import DOVSOutages
from OutageDAQ import OutageDataInfo as ODI
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
#---------------------------------------------------------------------
#sys.path.insert(0, os.path.join(os.path.realpath('..'), 'Utilities'))
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Plot_General
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
from DataFrameSubsetSlicer import DataFrameSubsetSingleSlicer as DFSingleSlicer
from CustomJSON import CustomWriter
#---------------------------------------------------------------------
class DumbClassifier(BaseEstimator):
    r"""
    A dumb classifier which always predicts 1
    """
    def fit(self, X, y=None):
        return self
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


#---------------------------------------------------------------------

class OutageModeler:
    r"""
    Class to construct outage model
    """
    def __init__(
        self,
        data_evs_sum_vw              = True, 
        include_prbl                 = True, 
        save_base_dir                = None, 
        save_sub_dir                 = None, 
        init_dfs_to                  = pd.DataFrame(),  # Should be pd.DataFrame() or None
        force_fresh_data_build       = False, 
        save_data                    = True, # if True, will only save if new object built (i.e., no saving if simply read in)
        save_model                   = False, 
        extend_lists_in_mecpx_kwargs = False, 
        extend_lists_in_model_kwargs = False, 
        verbose                      = True, 
        outg_mdlr                    = None, # if copy constructor functionality is desired
        **kwargs
    ):
        r"""
        Acceptable kwargs:
            See OutageModeler.get_default_mecpx_build_info_dict, OutageModeler.get_acceptable_generic_mecpx_build_args(),  and 
              OutageModeler.get_default_model_args_dict/OutageModeler.get_model_args
            e.g., OutageModeler(red_test_size=0.70, max_total_counts=1000)
        """
        #---------------------------------------------------------------------------
        if outg_mdlr is not None:
            assert(isinstance(outg_mdlr, OutageModeler))
            self.copy_constructor(outg_mdlr)
            return
        #---------------------------------------------------------------------------
        # Split up the kwargs into those for building MECPOCollections and those for modelling
        split_kwargs_dict = OutageModeler.split_kwargs(**kwargs)
        #-----
        mecpx_kwargs = split_kwargs_dict['mecpx_kwargs']
        model_kwargs = split_kwargs_dict['model_kwargs']
        other_kwargs = split_kwargs_dict['other_kwargs']
        #-----
        if len(other_kwargs) > 0:
            print('WARNING!!!!!\nIn OutageModeler constructor\nkwargs found not assigned to mecpx_kwargs or model_kwargs')
            print('THESE WILL NOT BE IMPLEMENTED:')
            print(other_kwargs)
        
        #---------------------------------------------------------------------------
        # When using self.init_dfs_to ALWAYS use copy.deepcopy(self.init_dfs_to),
        # otherwise, unintended consequences could return (self.init_dfs_to is None it
        # doesn't really matter, but when it is pd.DataFrame it does!)
        self.init_dfs_to = init_dfs_to
        
        #---------------------------------------------------------------------------
        # MECPOCollection objects
        #     These are needed to compile the finals DFs which will be used for modelling.
        #     HOWEVER, they are EXTREMELY heavy.
        #     Therefore, they should be closed out after being built and saved, and only used when needed.
        #     E.g., after building DFs, one should save to disk so they can be loaded directly when running model
        #       instead of needing to build from MECPOCollection objects each time.
        #-------------------------
        # mecpx = Meter Event Counts Per X (e.g., per transformer)
        # outg  = OUTaGe == signal data 
        # otbl  = Outage Transformer BaseLine
        # prbl  = PRistine BaseLine
        self.mecpx_coll_outg  = None
        self.mecpx_coll_otbl  = None
        self.mecpx_coll_prbl  = None
        
        self.cpx_dfs_name_outg = 'rcpo_df_norm_by_xfmr_nSNs'
        self.cpx_dfs_name_otbl = 'rcpo_df_norm_by_xfmr_nSNs'
        self.cpx_dfs_name_prbl = 'rcpo_df_norm_by_xfmr_nSNs'
        
        self.combine_reasons_kwargs = OutageModeler.get_default_combine_reasons_kwargs()
        self.mecpx_build_info_dict  = OutageModeler.get_default_mecpx_build_info_dict()
        #--------------------------------------------------
        mecpx_kwargs['data_evs_sum_vw'] = data_evs_sum_vw
        # If any mecpx_kwargs supplied by user in kwargs, set appropriate values
        if len(mecpx_kwargs)>0:
            self.set_mecpx_build_args(
                extend_any_lists = extend_lists_in_mecpx_kwargs, 
                **mecpx_kwargs
            )
        
        #---------------------------------------------------------------------------
        self.include_prbl    = include_prbl
        #TODO!!!!!!!!!!!!!!!!!!
        # In reality, probably need two different levels of __force_build 
        self.__force_build   = force_fresh_data_build
        #-----
        self.merged_df_outg  = copy.deepcopy(self.init_dfs_to)
        self.merged_df_otbl  = copy.deepcopy(self.init_dfs_to)
        self.merged_df_prbl  = copy.deepcopy(self.init_dfs_to)
        #-----
        self.time_infos_df   = copy.deepcopy(self.init_dfs_to)
        #-----
        self.counts_series_outg = None
        self.counts_series_otbl = None
        self.counts_series_prbl = None
        #-------------------------
        self.eemsp_df           = copy.deepcopy(self.init_dfs_to)
        self.eemsp_reduce1_df   = copy.deepcopy(self.init_dfs_to)
        self.eemsp_df_info_dict = None
        #-------------------------
        self.dovs_df            = copy.deepcopy(self.init_dfs_to)
        #-------------------------
        self.target_col         = ('is_outg', 'is_outg')
        self.from_outg_col      = ('from_outg', 'from_outg')
        #-------------------------
        self.__can_model          = True
        #-------------------------
        self.__df_train           = copy.deepcopy(self.init_dfs_to)
        self.__df_test            = copy.deepcopy(self.init_dfs_to)
        self.__df_holdout         = copy.deepcopy(self.init_dfs_to)
        self.__df_target          = copy.deepcopy(self.init_dfs_to) # Same index as __df_train/_train/_holdout, but only has columns 
                                                                    #   self.target_col and self.from_outg_col
        #-------------------------
        self.__X_train            = None
        self.__y_train            = None
        self.__y_train_pred       = None
        #-----
        self.__X_test             = None
        self.__y_test             = None
        self.__y_test_pred        = None
        #-----
        self.__X_holdout          = None
        self.__y_holdout          = None
        self.__y_holdout_pred     = None
        #-----
        
        #---------------------------------------------------------------------------
        # Model arguments 
        #   NOTE: These will be made private so I can control when and how they are set
        self.__model_args_locked = False # Set to True after model built
        #--------------------------------------------------
        self.__an_keys                    = None # NOT REALLY SET BY USER AT POINT OF MODELLING, BUT TAKEN FROM MECPO DATA
        #-------------------------
        self.__random_state               = None
        #-------------------------
        self.__n_top_reasons_to_inclue    = 10
        self.__combine_others             = True
        #-------------------------
        self.__merge_eemsp                = True
        self.__eemsp_mult_strategy        = 'agg'
        self.__eemsp_enc                  = None
        #-------------------------
        self.__include_month              = True
        #-------------------------
        self.__an_keys_to_drop            = None
        #-------------------------
        # NOTE: Timestamp is not JSON serializable, hence the need for strftime below
        self.__date_0_train                  = None
        self.__date_1_train                  = None
        #-----
        self.__date_0_test                   = None
        self.__date_1_test                   = None
        #-----
        self.__date_0_holdout                = None
        self.__date_1_holdout                = None
        #-------------------------
        self.__test_size                     = 0.33
        self.__get_train_test_by_date        = False
        self.__split_train_test_by_outg      = True
        self.__addtnl_bsln_frac              = 1.0
        #-------------------------
        self.__create_validation_set         = False
        self.__val_size                      = 0.1
        #-------------------------
        self.__run_scaler                    = True
        self.__scaler                        = None
        #-------------------------
        self.__run_PCA                       = False
        self.__pca_n_components              = 0.95
        self.__pca                           = None
        #-------------------------
        self.__outgs_slicer                  = OutageModeler.get_dummy_slicer()
        self.__gnrl_slicer                   = OutageModeler.get_dummy_slicer()
        #-------------------------
        self.__remove_others_from_outages    = False
        #-------------------------
        self.__min_pct_target_1              = None
        #-------------------------
        self.__reduce_train_size             = False
        self.__red_test_size                 = 0.75
        #-------------------------
        self.__is_norm                       = None
        self.__include_total_counts_features = True
        self.__remove_scheduled_outages      = True
        self.__trsf_pole_nbs_to_remove       = ['NETWORK', 'PRIMARY']
        #---------------------------------------------------------------------------
        self.__model_clf                     = None
        #--------------------------------------------------
        # If any model_kwargs supplied by user in kwargs, set appropriate values
        if len(model_kwargs)>0:
            self.set_model_args(
                extend_any_lists = extend_lists_in_model_kwargs, 
                verbose          = verbose, 
                **model_kwargs
            )
        
        #---------------------------------------------------------------------------
        self.save_base_dir = save_base_dir
        self.save_sub_dir  = save_sub_dir
        if self.save_sub_dir is None:
            self.save_sub_dir = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if(
            save_base_dir is not None and 
            not os.path.exists(os.path.join(save_base_dir, save_sub_dir))
        ):
            os.makedirs(os.path.join(save_base_dir, save_sub_dir))
        #---------------------------------------------------------------------------
        self.__save_data                   = save_data
        self.__save_model                  = save_model
        #-------------------------
        # self.__saved_ = False
        self.__saved_mecpo_colls           = False
        self.__saved_merged_dfs_0          = False
        self.__saved_merged_dfs            = False
        self.__saved_counts_series_0       = False
        self.__saved_counts_series         = False
        self.__saved_mecpx_build_info_dict = False
        
        self.__saved_time_infos_df         = False
        self.__saved_eemsp                 = False
        self.__saved_eemsp_enc             = False
        self.__saved_data_structure_df     = False
        self.__saved_scaler                = False
        self.__saved_pca                   = False
        self.__saved_summary_dict          = False
        self.__saved_model_clf             = False

    def force_fresh_build(
        self, 
        force_build=True
    ):
        r"""
        If set to True, object will be built, even if .pkl/.json/.we found at expected locations
        """
        #-------------------------
        self.__force_build = force_build
        
        
    @staticmethod
    def general_copy(
        attr
    ):
        r"""
        """
        #-------------------------
        if attr is None:
            return None
        #-------------------------
        # pd.DataFrame, pd.Series, and others have their own, built-in, .copy methods
        try:
            cpy = attr.copy()
            return cpy
        except:
            pass
        #-------------------------
        cpy = copy.deepcopy(attr)
        return cpy
        
    def copy_constructor(
        self, 
        outg_mdlr
    ):
        r"""
        Annoyingly, the follow simple solution does not work:
            self = copy.deepcopy(outg_mdlr)
          neither does:
            self = OutageModeler()
            self.__dict__ = copy.deepcopy(outg_mdlr.__dict__)
    
        So, I guess it's back to headache C++ style...
        """
        #--------------------------------------------------
        assert(isinstance(outg_mdlr, OutageModeler))
        #--------------------------------------------------
        self.init_dfs_to                     = OutageModeler.general_copy(outg_mdlr.init_dfs_to)
        #--------------------------------------------------
        self.mecpx_coll_outg                 = OutageModeler.general_copy(outg_mdlr.mecpx_coll_outg)
        self.mecpx_coll_otbl                 = OutageModeler.general_copy(outg_mdlr.mecpx_coll_otbl)
        self.mecpx_coll_prbl                 = OutageModeler.general_copy(outg_mdlr.mecpx_coll_prbl)
    
        self.cpx_dfs_name_outg               = OutageModeler.general_copy(outg_mdlr.cpx_dfs_name_outg)
        self.cpx_dfs_name_otbl               = OutageModeler.general_copy(outg_mdlr.cpx_dfs_name_otbl)
        self.cpx_dfs_name_prbl               = OutageModeler.general_copy(outg_mdlr.cpx_dfs_name_prbl)
    
        self.combine_reasons_kwargs          = OutageModeler.general_copy(outg_mdlr.combine_reasons_kwargs)
        self.mecpx_build_info_dict           = OutageModeler.general_copy(outg_mdlr.mecpx_build_info_dict)
        #--------------------------------------------------
        self.include_prbl                    = OutageModeler.general_copy(outg_mdlr.include_prbl)
        #-----
        self.merged_df_outg                  = OutageModeler.general_copy(outg_mdlr.merged_df_outg)
        self.merged_df_otbl                  = OutageModeler.general_copy(outg_mdlr.merged_df_otbl)
        self.merged_df_prbl                  = OutageModeler.general_copy(outg_mdlr.merged_df_prbl)
        #-----
        self.time_infos_df                   = OutageModeler.general_copy(outg_mdlr.time_infos_df)
        #-----
        self.counts_series_outg              = OutageModeler.general_copy(outg_mdlr.counts_series_outg)
        self.counts_series_otbl              = OutageModeler.general_copy(outg_mdlr.counts_series_otbl)
        self.counts_series_prbl              = OutageModeler.general_copy(outg_mdlr.counts_series_prbl)
        #-------------------------
        self.eemsp_df                        = OutageModeler.general_copy(outg_mdlr.eemsp_df)
        self.eemsp_reduce1_df                = OutageModeler.general_copy(outg_mdlr.eemsp_reduce1_df)
        self.eemsp_df_info_dict              = OutageModeler.general_copy(outg_mdlr.eemsp_df_info_dict)
        #-------------------------
        self.dovs_df                         = OutageModeler.general_copy(outg_mdlr.dovs_df)
        #-------------------------
        self.target_col                      = OutageModeler.general_copy(outg_mdlr.target_col)
        self.from_outg_col                   = OutageModeler.general_copy(outg_mdlr.from_outg_col)
        #-------------------------
        self.__can_model                     = OutageModeler.general_copy(outg_mdlr.__can_model)
        #-------------------------
        self.__df_train                      = OutageModeler.general_copy(outg_mdlr.__df_train)
        self.__df_test                       = OutageModeler.general_copy(outg_mdlr.__df_test)
        self.__df_holdout                    = OutageModeler.general_copy(outg_mdlr.__df_holdout)
        self.__df_target                     = OutageModeler.general_copy(outg_mdlr.__df_target)
        #-------------------------
        self.__X_train                       = OutageModeler.general_copy(outg_mdlr.__X_train)
        self.__y_train                       = OutageModeler.general_copy(outg_mdlr.__y_train)
        self.__y_train_pred                  = OutageModeler.general_copy(outg_mdlr.__y_train_pred)
        #-----
        self.__X_test                        = OutageModeler.general_copy(outg_mdlr.__X_test)
        self.__y_test                        = OutageModeler.general_copy(outg_mdlr.__y_test)
        self.__y_test_pred                   = OutageModeler.general_copy(outg_mdlr.__y_test_pred)
        #-----
        self.__X_holdout                     = OutageModeler.general_copy(outg_mdlr.__X_holdout)
        self.__y_holdout                     = OutageModeler.general_copy(outg_mdlr.__y_holdout)
        self.__y_holdout_pred                = OutageModeler.general_copy(outg_mdlr.__y_holdout_pred)
        #--------------------------------------------------
        self.__model_args_locked             = OutageModeler.general_copy(outg_mdlr.__model_args_locked)
        #--------------------------------------------------
        self.__an_keys                       = OutageModeler.general_copy(outg_mdlr.__an_keys)
        #-------------------------
        self.__random_state                  = OutageModeler.general_copy(outg_mdlr.__random_state)
        #-------------------------
        self.__n_top_reasons_to_inclue       = OutageModeler.general_copy(outg_mdlr.__n_top_reasons_to_inclue)
        self.__combine_others                = OutageModeler.general_copy(outg_mdlr.__combine_others)
        #-------------------------
        self.__merge_eemsp                   = OutageModeler.general_copy(outg_mdlr.__merge_eemsp)
        self.__eemsp_mult_strategy           = OutageModeler.general_copy(outg_mdlr.__eemsp_mult_strategy)
        self.__eemsp_enc                     = OutageModeler.general_copy(outg_mdlr.__eemsp_enc)
        #-------------------------
        self.__include_month                 = OutageModeler.general_copy(outg_mdlr.__include_month)
        #-------------------------
        self.__an_keys_to_drop               = OutageModeler.general_copy(outg_mdlr.__an_keys_to_drop)
        #-------------------------
        self.__date_0_train                  = OutageModeler.general_copy(outg_mdlr.__date_0_train)
        self.__date_1_train                  = OutageModeler.general_copy(outg_mdlr.__date_1_train)
        #-----
        self.__date_0_test                   = OutageModeler.general_copy(outg_mdlr.__date_0_test)
        self.__date_1_test                   = OutageModeler.general_copy(outg_mdlr.__date_1_test)
        #-----
        self.__date_0_holdout                = OutageModeler.general_copy(outg_mdlr.__date_0_holdout)
        self.__date_1_holdout                = OutageModeler.general_copy(outg_mdlr.__date_1_holdout)
        #-------------------------
        self.__test_size                     = OutageModeler.general_copy(outg_mdlr.__test_size)
        self.__get_train_test_by_date        = OutageModeler.general_copy(outg_mdlr.__get_train_test_by_date)
        self.__split_train_test_by_outg      = OutageModeler.general_copy(outg_mdlr.__split_train_test_by_outg)
        self.__addtnl_bsln_frac              = OutageModeler.general_copy(outg_mdlr.__addtnl_bsln_frac)
        #-------------------------
        self.__create_validation_set         = OutageModeler.general_copy(outg_mdlr.__create_validation_set)
        self.__val_size                      = OutageModeler.general_copy(outg_mdlr.__val_size)
        #-------------------------
        self.__run_scaler                    = OutageModeler.general_copy(outg_mdlr.__run_scaler)
        self.__scaler                        = OutageModeler.general_copy(outg_mdlr.__scaler)
        #-------------------------
        self.__run_PCA                       = OutageModeler.general_copy(outg_mdlr.__run_PCA)
        self.__pca_n_components              = OutageModeler.general_copy(outg_mdlr.__pca_n_components)
        self.__pca                           = OutageModeler.general_copy(outg_mdlr.__pca)
        #-------------------------
        self.__outgs_slicer                  = OutageModeler.general_copy(outg_mdlr.__outgs_slicer)
        self.__gnrl_slicer                   = OutageModeler.general_copy(outg_mdlr.__gnrl_slicer)
        #-------------------------
        self.__remove_others_from_outages    = OutageModeler.general_copy(outg_mdlr.__remove_others_from_outages)
        #-------------------------
        self.__min_pct_target_1              = OutageModeler.general_copy(outg_mdlr.__min_pct_target_1)
        #-------------------------
        self.__reduce_train_size             = OutageModeler.general_copy(outg_mdlr.__reduce_train_size)
        self.__red_test_size                 = OutageModeler.general_copy(outg_mdlr.__red_test_size)
        #-------------------------
        self.__is_norm                       = OutageModeler.general_copy(outg_mdlr.__is_norm)
        self.__include_total_counts_features = OutageModeler.general_copy(outg_mdlr.__include_total_counts_features)
        self.__remove_scheduled_outages      = OutageModeler.general_copy(outg_mdlr.__remove_scheduled_outages)
        self.__trsf_pole_nbs_to_remove       = OutageModeler.general_copy(outg_mdlr.__trsf_pole_nbs_to_remove)
        #-------------------------
        self.__model_clf                     = OutageModeler.general_copy(outg_mdlr.__model_clf)
        #--------------------------------------------------
        self.save_base_dir                   = OutageModeler.general_copy(outg_mdlr.save_base_dir)
        self.save_sub_dir                    = OutageModeler.general_copy(outg_mdlr.save_sub_dir)
        #--------------------------------------------------
        self.__save_data                     = OutageModeler.general_copy(outg_mdlr.__save_data)
        self.__save_model                    = OutageModeler.general_copy(outg_mdlr.__save_model)
        #-------------------------
        self.__saved_mecpo_colls             = OutageModeler.general_copy(outg_mdlr.__saved_mecpo_colls)
        self.__saved_merged_dfs_0            = OutageModeler.general_copy(outg_mdlr.__saved_merged_dfs_0)
        self.__saved_merged_dfs              = OutageModeler.general_copy(outg_mdlr.__saved_merged_dfs)
        self.__saved_counts_series_0         = OutageModeler.general_copy(outg_mdlr.__saved_counts_series_0)
        self.__saved_counts_series           = OutageModeler.general_copy(outg_mdlr.__saved_counts_series)
        self.__saved_mecpx_build_info_dict   = OutageModeler.general_copy(outg_mdlr.__saved_mecpx_build_info_dict)
        #-------------------------
        self.__saved_time_infos_df           = OutageModeler.general_copy(outg_mdlr.__saved_time_infos_df)
        self.__saved_eemsp                   = OutageModeler.general_copy(outg_mdlr.__saved_eemsp)
        self.__saved_eemsp_enc               = OutageModeler.general_copy(outg_mdlr.__saved_eemsp_enc)
        self.__saved_data_structure_df       = OutageModeler.general_copy(outg_mdlr.__saved_data_structure_df)
        self.__saved_scaler                  = OutageModeler.general_copy(outg_mdlr.__saved_scaler)
        self.__saved_pca                     = OutageModeler.general_copy(outg_mdlr.__saved_pca)
        self.__saved_summary_dict            = OutageModeler.general_copy(outg_mdlr.__saved_summary_dict)
        self.__saved_model_clf               = OutageModeler.general_copy(outg_mdlr.__saved_model_clf)
        
    def copy(self):
        r"""
        """
        #-------------------------
        return_outg_mdlr = OutageModeler(outg_mdlr=self)
        return return_outg_mdlr
        
        
    def reset_model_results(
        self
    ):
        r"""
        """
        #-------------------------
        self.__df_train                      = copy.deepcopy(self.init_dfs_to)
        self.__df_test                       = copy.deepcopy(self.init_dfs_to)
        self.__df_holdout                    = copy.deepcopy(self.init_dfs_to)
        self.__df_target                     = copy.deepcopy(self.init_dfs_to)
        #-------------------------
        self.__X_train                       = None
        self.__y_train                       = None
        self.__y_train_pred                  = None
        #-----
        self.__X_test                        = None
        self.__y_test                        = None
        self.__y_test_pred                   = None
        #-----
        self.__X_holdout                     = None
        self.__y_holdout                     = None
        self.__y_holdout_pred                = None
        
        
    @staticmethod
    def split_kwargs(**kwargs):
        r"""
        Split up the kwargs into groups:
            - Building MECPOCollection objects/ merged DFs arguments
            - Model arguments
            - Any other arguments
        """
        #--------------------------------------------------
        if len(kwargs)==0:
            return dict(
                mecpx_kwargs = {}, 
                model_kwargs = {}, 
                other_kwargs = {}
            )
        #--------------------------------------------------
        dflt_mecpx_keys = list(OutageModeler.get_default_mecpx_build_info_dict().keys()) + OutageModeler.get_acceptable_generic_mecpx_build_args()
        dflt_model_keys = list(OutageModeler.get_default_model_args_dict().keys())
        #-----
        # This only works if there is no overlap in the mecpx and model kwargs!!!!!
        assert(set(dflt_mecpx_keys).intersection(set(dflt_model_keys))==set())
        #-------------------------
        mecpx_keys = set(dflt_mecpx_keys).intersection(set(kwargs.keys()))
        model_keys = set(dflt_model_keys).intersection(set(kwargs.keys()))
        other_keys = set(kwargs.keys()).difference(set(dflt_mecpx_keys+dflt_model_keys))
        #-------------------------
        mecpx_kwargs = {k:v for k,v in kwargs.items() if k in mecpx_keys}
        model_kwargs = {k:v for k,v in kwargs.items() if k in model_keys}
        other_kwargs = {k:v for k,v in kwargs.items() if k in other_keys}
        #--------------------------------------------------
        return dict(
            mecpx_kwargs = mecpx_kwargs, 
            model_kwargs = model_kwargs, 
            other_kwargs = other_kwargs
        )

    @property
    def can_model(self):
        return self.__can_model
    @property
    def df_train(self):
        return self.__df_train
    @property
    def df_test(self):
        return self.__df_test
    @property
    def df_holdout(self):
        return self.__df_holdout
    @property
    def df_target(self):
        return self.__df_target
    @property
    def df(self):
        dfs = [self.df_train, self.df_test]
        if self.df_holdout is not None and self.df_holdout.shape[0]>0:
            dfs.append(self.df_holdout)
        df = pd.concat(dfs, axis=0)
        return df
        
    @property
    def X_train(self):
        return self.__X_train
    @property
    def y_train(self):
        return self.__y_train
    @property
    def y_train_pred(self):
        return self.__y_train_pred
    @property
    def X_test(self):
        return self.__X_test
    @property
    def y_test(self):
        return self.__y_test
    @property
    def y_test_pred(self):
        return self.__y_test_pred
    @property
    def X_holdout(self):
        return self.__X_holdout
    @property
    def y_holdout(self):
        return self.__y_holdout
    @property
    def y_holdout_pred(self):
        return self.__y_holdout_pred
        
    @property
    def model_args_locked(self):
        return self.__model_args_locked
    #-------------------------
    @property
    def an_keys(self):
        self.__an_keys = self.mecpx_build_info_dict['mecpx_an_keys']
        return self.__an_keys
    #-------------------------
    @property
    def random_state(self):
        return self.__random_state
        
    def set_random_state(self, state=None):
        self.__random_state = state
    #-------------------------
    @property
    def n_top_reasons_to_inclue(self):
        return self.__n_top_reasons_to_inclue
    @property
    def combine_others(self):
        return self.__combine_others
    #-------------------------
    @property
    def merge_eemsp(self):
        return self.__merge_eemsp
    @property
    def eemsp_mult_strategy(self):
        return self.__eemsp_mult_strategy
    @property
    def eemsp_enc(self):
        return self.__eemsp_enc
    #-------------------------
    @property
    def include_month(self):
        return self.__include_month
    #-------------------------
    @property
    def an_keys_to_drop(self):
        return self.__an_keys_to_drop
    #-------------------------
    @property
    def date_0_train(self):
        return self.__date_0_train
    @property
    def date_1_train(self):
        return self.__date_1_train
    #-----
    @property
    def date_0_test(self):
        return self.__date_0_test
    @property
    def date_1_test(self):
        return self.__date_1_test
    #-----
    @property
    def date_0_holdout(self):
        return self.__date_0_holdout
    @property
    def date_1_holdout(self):
        return self.__date_1_holdout
    #-------------------------
    @property
    def test_size(self):
        return self.__test_size
    @property
    def get_train_test_by_date(self):
        return self.__get_train_test_by_date
    @property
    def split_train_test_by_outg(self):
        return self.__split_train_test_by_outg
    @property
    def addtnl_bsln_frac(self):
        return self.__addtnl_bsln_frac
    #-------------------------
    @property
    def create_validation_set(self):
        return self.__create_validation_set
    @property
    def val_size(self):
        return self.__val_size
    #-------------------------
    @property
    def run_scaler(self):
        return self.__run_scaler
    @property
    def scaler(self):
        return self.__scaler
    #-------------------------
    @property
    def run_PCA(self):
        return self.__run_PCA
    @property
    def pca_n_components(self):
        return self.__pca_n_components
    @property
    def pca(self):
        return self.__pca
    #-------------------------
    @property
    def outgs_slicer(self):
        return self.__outgs_slicer
    @property
    def gnrl_slicer(self):
        return self.__gnrl_slicer
    #-------------------------
    @property
    def remove_others_from_outages(self):
        return self.__remove_others_from_outages
    #-------------------------
    @property
    def min_pct_target_1(self):
        return self.__min_pct_target_1
    #-------------------------
    @property
    def reduce_train_size(self):
        return self.__reduce_train_size
    @property
    def red_test_size(self):
        return self.__red_test_size
    #-------------------------
    @property
    def is_norm(self):
        self.__is_norm = self.are_data_normalized()
        return self.__is_norm
    @property
    def include_total_counts_features(self):
        return self.__include_total_counts_features
    @property
    def remove_scheduled_outages(self):
        return self.__remove_scheduled_outages
    @property
    def trsf_pole_nbs_to_remove(self):
        return self.__trsf_pole_nbs_to_remove
        
    @property
    def model_clf(self):
        return self.__model_clf
        
    #NOTE: Explicitly note using the .setter property decorator because I don't want changing to be simple
    #        to the user (i.e., I want them to have to go out of their way, to better ensure they understand
    #        what they're doing when making the change)
    def set_gnrl_slicer(self, gnrl_slicer):
        self.__gnrl_slicer = gnrl_slicer

        
        
    @property
    def save_dir(self):
        r"""
        If self.save_base_dir and self.save_sub_dir both set, return os.path.join of the two
        Otherwise, return None
        """
        #-------------------------
        if self.save_base_dir is None or self.save_sub_dir is None:
            return None
        else:
            return os.path.join(self.save_base_dir, self.save_sub_dir)
            
    def is_save_base_dir_loaded(
        self, 
        verbose = True
    ):
        r"""
        """
        #-------------------------
        if self.save_base_dir is None:
            return False
        #-------------------------
        if not os.path.isdir(self.save_base_dir):
            if verbose:
                print(f"OutageModeler: save_base_dir is set ({self.save_base_dir}), but the directory does not exist!")
            return False
        #-------------------------
        return True
    
    def is_save_sub_dir_loaded(
        self, 
        verbose = True
    ):
        r"""
        """
        #-------------------------
        if self.save_base_dir is None:
            return False
        #-------------------------
        if self.save_sub_dir is None:
            return False
        #-------------------------
        
        if not os.path.isdir(self.save_dir):
            if verbose:
                print(f"OutageModeler: save_sub_dir is set ({self.save_dir}), but the directory does not exist!")
            return False
        #-------------------------
        return True
    
    def make_sub_dir(
        self
    ):
        r"""
        """
        #-------------------------
        if os.path.isdir(self.save_dir):
            return
        #-------------------------
        if self.save_base_dir is None:
            print('OutageModeler.make_sub_dir: CANNOT MAKE DIRECTORY BECAUSE save_base_dir IS None!\nCrash Imminent!')
            assert(0)
        #-------------------------
        if self.save_sub_dir is None:
            print('OutageModeler.make_sub_dir: CANNOT MAKE DIRECTORY BECAUSE save_sub_dir IS None!\nCrash Imminent!')
            assert(0)
        #-------------------------
        os.makedirs(self.save_dir)
        
        
    @staticmethod
    def get_default_combine_reasons_kwargs():
        r"""
        """
        #-------------------------
        combine_reasons_kwargs = dict(
            patterns_and_replace        = None, 
            addtnl_patterns_and_replace = None, 
            initial_strip               = True,
            initial_punctuation_removal = True, 
            level_0_raw_col             = 'counts', 
            level_0_nrm_col             = 'counts_norm', 
            return_red_to_org_cols_dict = False
        )
        return combine_reasons_kwargs
        
        
    def are_data_normalized(self):
        r"""
        """
        #-------------------------
        is_norm_list = [
            MECPOAn.is_df_normalized(cpx_df_name = self.cpx_dfs_name_outg), 
            MECPOAn.is_df_normalized(cpx_df_name = self.cpx_dfs_name_otbl)
        ]
        if self.include_prbl:
            is_norm_list.append(MECPOAn.is_df_normalized(cpx_df_name = self.cpx_dfs_name_prbl))
        #-------------------------
        # Make sure all agree...
        assert(len(set(is_norm_list))==1)
        return is_norm_list[0]
        
        
    @staticmethod
    def check_mecpx_build_info_dict(mecpx_build_info_dict):
        r"""
        """
        exit_value = 0
        try:
            #-------------------------
            # exit_value = 0
            if mecpx_build_info_dict['old_to_new_keys_dict'] is not None:
                assert(len(mecpx_build_info_dict['old_to_new_keys_dict']) == len(mecpx_build_info_dict['days_min_max_outg_td_windows']))
            exit_value+=1
            #-------------------------
            # exit_value = 1
            # Sanity check
            for window_i in mecpx_build_info_dict['days_min_max_outg_td_windows']:
                assert(pd.Timedelta(f'{window_i[1]}D')-pd.Timedelta(f'{window_i[0]}D')==pd.Timedelta(mecpx_build_info_dict['freq']))
            exit_value+=1
            #-------------------------
            # exit_value = 2
            assert(set(mecpx_build_info_dict['grp_by_cols_outg']).symmetric_difference(set(mecpx_build_info_dict['std_dict_grp_by_cols_outg'].keys()))==set())
            assert(set(mecpx_build_info_dict['grp_by_cols_otbl']).symmetric_difference(set(mecpx_build_info_dict['std_dict_grp_by_cols_otbl'].keys()))==set())
            assert(set(mecpx_build_info_dict['grp_by_cols_prbl']).symmetric_difference(set(mecpx_build_info_dict['std_dict_grp_by_cols_prbl'].keys()))==set())
            exit_value+=1
            #-----
            # exit_value = 3
            assert(
                set(mecpx_build_info_dict['std_dict_grp_by_cols_outg'].values()) == 
                set(mecpx_build_info_dict['std_dict_grp_by_cols_otbl'].values()) == 
                set(mecpx_build_info_dict['std_dict_grp_by_cols_prbl'].values())
            )
            exit_value+=1
            #-------------------------
            # Make sure the final index names of the time_info_dfs (ti_dfs) agree
            # exit_value = 4
            assert(set(mecpx_build_info_dict['std_dict_grp_by_cols_outg'].values())==set(mecpx_build_info_dict['std_dict_grp_by_cols_otbl'].values()))
            assert(set(mecpx_build_info_dict['std_dict_grp_by_cols_outg'].values())==set(mecpx_build_info_dict['std_dict_grp_by_cols_prbl'].values()))
            exit_value+=1
            #-----
            # Make sure rec_nb_idfr and trsf_pole_nb_idfr contained in final indices
            # exit_value = 5
            assert(mecpx_build_info_dict['rec_nb_idfr'][1]       in mecpx_build_info_dict['std_dict_grp_by_cols_outg'].values())
            assert(mecpx_build_info_dict['trsf_pole_nb_idfr'][1] in mecpx_build_info_dict['std_dict_grp_by_cols_outg'].values())
            exit_value+=1
            #-------------------------
            # exit_value = 6
            return True
        except:
            print(f"FAILED in OutageModeler.check_mecpx_build_info_dict\n\texit_value = {exit_value}")
            return False
            
    @staticmethod
    def get_acceptable_generic_mecpx_build_args():
        r"""
        """
        #-------------------------
        # None of these generic forms should be found in self.mecpx_build_info_dict, only the specific forms!
        #   e.g., acq_run_date should not be in self.mecpx_build_info_dict, but acq_run_date_outg, acq_run_date_otbl, and acq_run_date_prbl should
        #-------------------------
        acceptable_kwargs = [
            'acq_run_date', 
            'data_date_ranges', 
            'data_dir_base', 
            'grp_by_cols_bsln', 
            'std_dict_grp_by_cols_bsln', 
            'cpx_dfs_name'
        ]
        return acceptable_kwargs
        
    @staticmethod
    def build_old_to_new_an_keys(
        old_an_keys
    ):
        r"""
        Given a list of old analysis keys (i.e., those assigned during the data acquisition and packagaing), convert these
          to their simpler form.
                e.g., 'outg_td_window_1_to_6_days' ==> '01-06 Days'
        Returns a dictionary whose keys are the old values and values are new keys
        """
        #-------------------------
        assert(isinstance(old_an_keys, list))
        regex = r'.*_(\d*)_to_(\d*)_days'
        return_dict = {}
        for old_key_i in old_an_keys:
            assert(old_key_i not in return_dict.keys())
            #-----
            new_key_i = re.findall(regex, old_key_i)
            assert(len(new_key_i)==1 and len(new_key_i[0])==2)
            #-----
            new_key_i = "{:02d}-{:02d} Days".format(int(new_key_i[0][0]), int(new_key_i[0][1]))
            #-----
            return_dict[old_key_i] = new_key_i
        #-------------------------
        return return_dict
        
    @staticmethod
    def get_default_mecpx_build_info_dict():
        r"""
        Mainly so user doesn't have to enter all values each time.
        This will probably be used with Utilities.supplement_dict_with_default_values
        """
        #-------------------------
        mecpx_build_info_dict = dict()
        #-------------------------
        mecpx_build_info_dict['data_evs_sum_vw']              = True
        #-------------------------
        mecpx_build_info_dict['acq_run_date_outg']            = '20240101'
        mecpx_build_info_dict['data_date_ranges_outg']        = [
            ['2023-01-01', '2023-12-31']
        ]
        mecpx_build_info_dict['data_dir_base_outg']           = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
        mecpx_build_info_dict['grp_by_cols_outg']             = [
            'outg_rec_nb', 
            'trsf_pole_nb'
        ]
        mecpx_build_info_dict['std_dict_grp_by_cols_outg']    = {
            'outg_rec_nb'  : 'outg_rec_nb', 
            'trsf_pole_nb' : 'trsf_pole_nb'
        }
        mecpx_build_info_dict['coll_label_outg']              = 'Outages'
        mecpx_build_info_dict['barplot_kwargs_shared_outg']   = dict(facecolor='red')
        #-----
        mecpx_build_info_dict['acq_run_date_otbl']            = '20240101'
        mecpx_build_info_dict['data_date_ranges_otbl']        = [
            ['2023-01-01', '2023-12-31']
        ]
        mecpx_build_info_dict['data_dir_base_otbl']           = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
        mecpx_build_info_dict['grp_by_cols_otbl']             = [
            'trsf_pole_nb', 
            'no_outg_rec_nb'
        ]
        mecpx_build_info_dict['std_dict_grp_by_cols_otbl']    = {
            'no_outg_rec_nb'  : 'outg_rec_nb', 
            'trsf_pole_nb'    : 'trsf_pole_nb'
        }
        mecpx_build_info_dict['coll_label_otbl']              = 'Baseline (OTBL)'
        mecpx_build_info_dict['barplot_kwargs_shared_otbl']   = dict(facecolor='orange')
        #-----
        mecpx_build_info_dict['acq_run_date_prbl']            = '20240101'
        mecpx_build_info_dict['data_date_ranges_prbl']        = [
            ['2023-01-01', '2023-12-31']
        ]
        mecpx_build_info_dict['data_dir_base_prbl']           = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
        mecpx_build_info_dict['grp_by_cols_prbl']             = [
            'trsf_pole_nb', 
            'no_outg_rec_nb'
        ]
        mecpx_build_info_dict['std_dict_grp_by_cols_prbl']    = {
            'no_outg_rec_nb'  : 'outg_rec_nb', 
            'trsf_pole_nb'    : 'trsf_pole_nb'
        }
        mecpx_build_info_dict['coll_label_prbl']              = 'Baseline (PRBL)'
        mecpx_build_info_dict['barplot_kwargs_shared_prbl']   = dict(facecolor='orange')
        #-------------------------
        mecpx_build_info_dict['rec_nb_idfr']                  = ('index', 'outg_rec_nb')
        mecpx_build_info_dict['trsf_pole_nb_idfr']            = ('index', 'trsf_pole_nb')        
        #-------------------------
        mecpx_build_info_dict['normalize_by_time_interval']   = True
        #-------------------------
        mecpx_build_info_dict['include_power_down_minus_up']  = False
        mecpx_build_info_dict['pd_col']                       = 'Primary Power Down'
        mecpx_build_info_dict['pu_col']                       = 'Primary Power Up'
        mecpx_build_info_dict['pd_m_pu_col']                  = 'Power Down Minus Up'
        #-------------------------
        mecpx_build_info_dict['regex_to_remove_patterns']     = [
            '.*cleared.*', 
            '.*Test Mode.*'
        ]
        mecpx_build_info_dict['regex_to_remove_ignore_case']  = True
        #-------------------------
        mecpx_build_info_dict['max_total_counts']             = None
        mecpx_build_info_dict['how_max_total_counts']         = 'any'
        #-------------------------
        mecpx_build_info_dict['mecpo_idx_for_ordering']       = 0
        #-------------------------
        mecpx_build_info_dict['cpx_dfs_name_outg']            = 'rcpo_df_norm_by_xfmr_nSNs'
        mecpx_build_info_dict['cpx_dfs_name_otbl']            = 'rcpo_df_norm_by_xfmr_nSNs'
        mecpx_build_info_dict['cpx_dfs_name_prbl']            = 'rcpo_df_norm_by_xfmr_nSNs'
        #-------------------------
        mecpx_build_info_dict['freq'] = '5D'
        #-----
        mecpx_build_info_dict['days_min_max_outg_td_windows'] = [
            [1,6], [6,11], [11,16], [16,21], [21,26], [26,31]
        ]
        #-----
        mecpx_build_info_dict['mecpx_an_keys']                = None
        #-----
        mecpx_build_info_dict['old_to_new_keys_dict']         = {
            'outg_td_window_1_to_6_days'  :'01-06 Days',
            'outg_td_window_6_to_11_days' :'06-11 Days',
            'outg_td_window_11_to_16_days':'11-16 Days',
            'outg_td_window_16_to_21_days':'16-21 Days',
            'outg_td_window_21_to_26_days':'21-26 Days',
            'outg_td_window_26_to_31_days':'26-31 Days'
        }
        #-------------------------
        assert(OutageModeler.check_mecpx_build_info_dict(mecpx_build_info_dict))
        return mecpx_build_info_dict
        
        
    def set_mecpx_build_args(
        self, 
        extend_any_lists = False, 
        **kwargs
    ):
        r"""
        Set any of the arguments used to build the MECPO collections.
            e.g., outg_mdlr.set_mecpx_build_args(max_total_counts=1000)
        extend_any_lists:
            If True, any list values in self.mecpx_build_info_dict will be supplemented with the values from kwargs
              instead of being replace by them.
        -----
        !!!!! IMPORTANT !!!!!
        There are some special kwargs which can be used to set multiple arguments.
        These are typically generic keywords which are then used to populate the more specific _outg, _otbl, and _prbl versions.
        These generic keywords are contained in OutageModeler.get_acceptable_generic_mecpx_build_args()
        e.g., if one supplies acq_run_date='20240101'
            ==> acq_run_date_outg = acq_run_date_otbl = acq_run_date_prbl = '20240101'
        e.g., if one supplies acq_run_date='20240101' and acq_run_date_outg='20240102'
            ==> acq_run_date_outg = '20240102' and acq_run_date_otbl = acq_run_date_prbl = '20240101'
        NOTE: The following are slightly different.
                    grp_by_cols_bsln, std_dict_grp_by_cols_bsln, 
                e.g., grp_by_cols_bsln will only be used to set grp_by_cols_otbl and grp_by_cols_prbl
        -----
        Note, for the purposes here the input to Utilities.supplement_dict_with_default_values may seem backwards, but
         they are, in fact, correct.
        Namely:
            to_supplmnt_dict    = kwargs
            default_values_dict = self.mecpx_build_info_dict
        From Utilities.supplement_dict_with_default_values documentation:
                If a key IS NOT contained in to_supplmnt_dict
                  ==> a new key/value pair is simply created.
                If a key IS already contained in to_supplmnt_dict
                  ==> value (to_supplmnt_dict[key]) is kept 
                      UNLESS extend_any_lists==True and to_supplmnt_dict[key] and/or default_values_dict[key] is a list,
                        in which case the two lists are combined
        ==> Since to_supplmnt_dict=kwargs below, the values from kwargs are kept over those in self.mecpx_build_info_dict
        """
        #----------------------------------------------------------------------------------------------------
        if len(kwargs)==0:
            return
        #----------------------------------------------------------------------------------------------------
        addtnl_acceptable_kwargs = OutageModeler.get_acceptable_generic_mecpx_build_args()
        acceptable_kwargs = list(OutageModeler.get_default_mecpx_build_info_dict().keys()) + addtnl_acceptable_kwargs
        assert(set(kwargs.keys()).difference(set(acceptable_kwargs))==set())
        #-------------------------
        for generic_key_i in addtnl_acceptable_kwargs:
            if generic_key_i not in kwargs.keys():
                continue
            # Need to remove generic_key_i from kwargs and update the specific versions, BUT ONLY IF NOT CONTAINED IN kwargs
            generic_val_i = kwargs.pop(generic_key_i)
            if generic_key_i in ['grp_by_cols_bsln', 'std_dict_grp_by_cols_bsln']:
                kwargs[f'{generic_key_i[:-5]}_otbl'] = kwargs.get(f'{generic_key_i[:-5]}_otbl', generic_val_i)
                kwargs[f'{generic_key_i[:-5]}_prbl'] = kwargs.get(f'{generic_key_i[:-5]}_prbl', generic_val_i)
            else:
                kwargs[f'{generic_key_i}_outg'] = kwargs.get(f'{generic_key_i}_outg', generic_val_i)
                kwargs[f'{generic_key_i}_otbl'] = kwargs.get(f'{generic_key_i}_otbl', generic_val_i)
                kwargs[f'{generic_key_i}_prbl'] = kwargs.get(f'{generic_key_i}_prbl', generic_val_i)

        #----------------------------------------------------------------------------------------------------
        # NOTE: For our purposes here, self.mecpx_build_info_dict will actually be the default_values_dict parameter below!
        #       This is because we want to keep values in kwargs over those in self.mecpx_build_info_dict
        self.mecpx_build_info_dict = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = kwargs, 
            default_values_dict = self.mecpx_build_info_dict, 
            extend_any_lists    = extend_any_lists, 
            inplace             = False
        )
        #-------------------------
        assert(OutageModeler.check_mecpx_build_info_dict(self.mecpx_build_info_dict))
        
        
    @staticmethod
    def get_default_model_args_dict():
        r"""
        Mainly so user doesn't have to enter all values each time.
        This will probably be used with Utilities.supplement_dict_with_default_values
        """
        #-------------------------
        model_args = dict()
        #-------------------------
        model_args['an_keys']                       = None
        #-------------------------
        model_args['random_state']                  = None
        #-------------------------
        model_args['n_top_reasons_to_inclue']       = 10
        model_args['combine_others']                = True
        #-------------------------
        model_args['merge_eemsp']                   = True
        model_args['eemsp_mult_strategy']           = 'agg'
        model_args['eemsp_enc']                     = None
        #-------------------------
        model_args['include_month']                 = True
        #-------------------------
        model_args['an_keys_to_drop']               = None
        #-------------------------
        model_args['date_0_train']                  = None
        model_args['date_1_train']                  = None
        #-----
        model_args['date_0_test']                   = None
        model_args['date_1_test']                   = None
        #-----
        model_args['date_0_holdout']                = None
        model_args['date_1_holdout']                = None
        #-------------------------
        model_args['test_size']                     = 0.33
        model_args['get_train_test_by_date']        = False
        model_args['split_train_test_by_outg']      = True
        model_args['addtnl_bsln_frac']              = 1.0
        #-------------------------
        model_args['create_validation_set']         = False
        model_args['val_size']                      = 0.1
        #-------------------------
        model_args['run_scaler']                    = True
        model_args['scaler']                        = None
        #-------------------------
        model_args['run_PCA']                       = False
        model_args['pca_n_components']              = 0.95
        model_args['pca']                           = None
        #-------------------------
        model_args['outgs_slicer']                  = OutageModeler.get_dummy_slicer()
        model_args['gnrl_slicer']                   = OutageModeler.get_dummy_slicer()
        #-------------------------
        model_args['remove_others_from_outages']    = False
        #-------------------------
        model_args['min_pct_target_1']              = None
        #-------------------------
        model_args['reduce_train_size']             = False
        model_args['red_test_size']                 = 0.75
        #-------------------------
        model_args['is_norm']                       = None
        model_args['include_total_counts_features'] = True    
        model_args['remove_scheduled_outages']      = True
        model_args['trsf_pole_nbs_to_remove']       = ['NETWORK', 'PRIMARY']
        #-------------------------
        return model_args
        
        
    def get_model_args(
        self, 
        return_copy=True
    ):
        r"""
        """
        #--------------------------------------------------
        model_args = dict()
        #-------------------------
        model_args['an_keys']                       = self.an_keys
        #-------------------------
        model_args['random_state']                  = self.random_state
        #-------------------------
        model_args['n_top_reasons_to_inclue']       = self.n_top_reasons_to_inclue
        model_args['combine_others']                = self.combine_others
        #-------------------------
        model_args['merge_eemsp']                   = self.merge_eemsp
        model_args['eemsp_mult_strategy']           = self.eemsp_mult_strategy
        model_args['eemsp_enc']                     = self.eemsp_enc
        #-------------------------
        model_args['include_month']                 = self.include_month
        #-------------------------
        model_args['an_keys_to_drop']               = self.an_keys_to_drop
        #-------------------------
        model_args['date_0_train']                  = self.date_0_train
        model_args['date_1_train']                  = self.date_1_train
        #-----
        model_args['date_0_test']                   = self.date_0_test
        model_args['date_1_test']                   = self.date_1_test
        #-----
        model_args['date_0_holdout']                = self.date_0_holdout
        model_args['date_1_holdout']                = self.date_1_holdout
        #-------------------------
        model_args['test_size']                     = self.test_size
        model_args['get_train_test_by_date']        = self.get_train_test_by_date
        model_args['split_train_test_by_outg']      = self.split_train_test_by_outg
        model_args['addtnl_bsln_frac']              = self.addtnl_bsln_frac
        #-------------------------
        model_args['create_validation_set']         = self.create_validation_set
        model_args['val_size']                      = self.val_size
        #-------------------------
        model_args['run_scaler']                    = self.run_scaler
        model_args['scaler']                        = self.scaler
        #-------------------------
        model_args['run_PCA']                       = self.run_PCA
        model_args['pca_n_components']              = self.pca_n_components
        model_args['pca']                           = self.pca
        #-------------------------
        model_args['outgs_slicer']                  = self.outgs_slicer
        model_args['gnrl_slicer']                   = self.gnrl_slicer
        #-------------------------
        model_args['remove_others_from_outages']    = self.remove_others_from_outages
        #-------------------------
        model_args['min_pct_target_1']              = self.min_pct_target_1
        #-------------------------
        model_args['reduce_train_size']             = self.reduce_train_size
        model_args['red_test_size']                 = self.red_test_size
        #-------------------------
        model_args['is_norm']                       = self.is_norm
        model_args['include_total_counts_features'] = self.red_test_size    
        model_args['remove_scheduled_outages']      = self.remove_scheduled_outages
        model_args['trsf_pole_nbs_to_remove']       = self.trsf_pole_nbs_to_remove
        #-------------------------
        model_args['model_clf']                     = self.model_clf
        #--------------------------------------------------
        if return_copy:
            return copy.deepcopy(model_args)
        else:
            return model_args
        
        
    def get_model_args_excluding(
        self, 
        args_to_exclude
    ):
        r"""
        """
        #--------------------------------------------------
        model_args = self.get_model_args(return_copy=True)
        assert(set(args_to_exclude).difference(set(model_args.keys()))==set())
        for arg_i in args_to_exclude:
            del model_args[arg_i]
        #--------------------------------------------------
        return model_args
        
        
    def get_summary_dict(
        self
    ):
        r"""
        This is just the model arguments (self.get_model_args()) with adjustments to a few entries (so they can be easily
          output into the summary JSON file)
        """
        #--------------------------------------------------
        # Some of the larger elements of the model arguments will be output to their own files, so they will 
        #   not be included in the dict returned here
        # args_to_exclude = ['eemsp_enc', 'scaler', 'pca', 'outgs_slicer', 'gnrl_slicer', 'model_clf']
        args_to_exclude = ['eemsp_enc', 'scaler', 'pca', 'model_clf']
        #--------------------------------------------------
        summary_dict = self.get_model_args_excluding(args_to_exclude=args_to_exclude)
        #--------------------------------------------------
        # NOTE: Timestamp is not JSON serializable, hence the need for strftime below
        if summary_dict['date_0_train'] is not None:
            summary_dict['date_0_train']               = summary_dict['date_0_train'].strftime('%Y-%m-%d %H:%M:%S')
        if summary_dict['date_1_train'] is not None:
            summary_dict['date_1_train']               = summary_dict['date_1_train'].strftime('%Y-%m-%d %H:%M:%S')
        #-----
        if summary_dict['date_0_test'] is not None:
            summary_dict['date_0_test']                = summary_dict['date_0_test'].strftime('%Y-%m-%d %H:%M:%S')
        if summary_dict['date_1_test'] is not None:
            summary_dict['date_1_test']                = summary_dict['date_1_test'].strftime('%Y-%m-%d %H:%M:%S')
        #-----
        if summary_dict['date_0_holdout'] is not None:
            summary_dict['date_0_holdout']             = summary_dict['date_0_holdout'].strftime('%Y-%m-%d %H:%M:%S')
        if summary_dict['date_1_holdout'] is not None:
            summary_dict['date_1_holdout']             = summary_dict['date_1_holdout'].strftime('%Y-%m-%d %H:%M:%S')
        #--------------------------------------------------
        assert(summary_dict['outgs_slicer'] is None or Utilities.is_object_one_of_types(summary_dict['outgs_slicer'], [DFSlicer, DFSingleSlicer]))
        if summary_dict['outgs_slicer'] is not None:
            summary_dict['outgs_slicer'] = summary_dict['outgs_slicer'].as_dict()
        #--------------------------------------------------
        assert(summary_dict['gnrl_slicer'] is None or Utilities.is_object_one_of_types(summary_dict['gnrl_slicer'], [DFSlicer, DFSingleSlicer]))
        if summary_dict['gnrl_slicer'] is not None:
            summary_dict['gnrl_slicer'] = summary_dict['gnrl_slicer'].as_dict()
        #--------------------------------------------------
        return summary_dict
        
    def save_summary_dict(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_summary_dict:
            return
        #-------------------------
        summary_dict = self.get_summary_dict()
        #-----
        assert(os.path.isdir(self.save_dir))
        #-----
        CustomWriter.output_dict_to_json(
            os.path.join(self.save_dir, 'summary_dict.json'), 
            summary_dict
        )
        self.__saved_summary_dict = True    
        
        
    def set_model_args(
        self, 
        extend_any_lists = False, 
        verbose          = True, 
        **kwargs
    ):
        r"""
        Update any of the arguments used to build the model.
            e.g., one could call outg_mdlr.set_model_args(random_state=42, n_top_reasons_to_inclue=10, ...)
        extend_any_lists:
            If True, any list values in model arguments will be supplemented with the values from kwargs
              instead of being replace by them.
        -----
        I'm going to be a little lazy (or maybe clever?) here and utilize Python's setattr functionality.
            This will allow me to set attributes without explicitly coding everything out.
            This will be beneficial in the future if acceptable model arguments are changed.
        The non-lazy way would be to create setter functions for all the private members, and call the appropriate
          setters depending on what is contained in kwargs.
            This would take a lot of ugly if statements, and would need to be updated each time an attribute is changed.
            Therefore, I guess my lazy method is actually most flexible...
    
        The flow of the function will be as follows:
            1. Grab the current model arguments using self.get_model_args()
            2. Adjust the model arguments with those provided in kwargs using Utilities.supplement_dict_with_default_values
            3. Set the model attributes using setattr
        -----
        Note, for the purposes here the input to Utilities.supplement_dict_with_default_values may seem backwards, but
         they are, in fact, correct.
        Namely:
            to_supplmnt_dict    = kwargs
            default_values_dict = self.mecpx_build_info_dict
        From Utilities.supplement_dict_with_default_values documentation:
                If a key IS NOT contained in to_supplmnt_dict
                  ==> a new key/value pair is simply created.
                If a key IS already contained in to_supplmnt_dict
                  ==> value (to_supplmnt_dict[key]) is kept 
                      UNLESS extend_any_lists==True and to_supplmnt_dict[key] and/or default_values_dict[key] is a list,
                        in which case the two lists are combined
        ==> Since to_supplmnt_dict=kwargs below, the values from kwargs are kept over those in self.mecpx_build_info_dict
        """
        #--------------------------------------------------
        if self.__model_args_locked:
            print("Model arguments LOCKED!\nCANNOT SET NEW VALUES!")
            return
        
        #--------------------------------------------------
        # 1. Grab the current model arguments using self.get_model_args()
        model_args = self.get_model_args(return_copy=False)
        
        #--------------------------------------------------
        # Make sure supplied kwargs are appropriate for model_args
        assert(set(kwargs.keys()).difference(set(model_args.keys()))==set())
    
        #--------------------------------------------------
        # NOT ALL ARGUMENTS CAN BE SET BY THE USER (i.e., some, e.g., self.an_keys, are set by the input data)
        args_not_set_by_user = ['an_keys', 'eemsp_enc', 'scaler', 'pca', 'is_norm', 'model_clf']
        rejected_kwargs = list(set(kwargs.keys()).intersection(set(args_not_set_by_user)))
        if len(rejected_kwargs)>0:
            if verbose:
                print(f"The following model arguments cannot be set by user, and will be ignored:\n{rejected_kwargs}")
            for arg_i in rejected_kwargs:
                del kwargs[arg_i]
    
        #--------------------------------------------------
        # 2. Adjust the model arguments with those provided in kwargs
        model_args = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = kwargs, 
            default_values_dict = model_args, 
            extend_any_lists    = extend_any_lists, 
            inplace             = False
        )
    
        #--------------------------------------------------
        # 3. Set the model attributes using setattr
        #    NOTE: Only setting those values which changed, i.e., those contained in kwargs.keys()
        #          BUT, we still need to use model_args in case extend_any_lists=True, in which case model_args has the
        #            correct value (because Utilities.supplement_dict_with_default_values was utilized) while that in kwargs is
        #            not fully correct.
        #    NOTE: Syntax = setattr(object, attribute, value), but setattr takes no keywords
        #    NOTE: Model arguments are now private, so _OutageModeler__ needed before attr
        for attr_i in kwargs.keys():
            val_i = model_args[attr_i]
            setattr(self, f'_OutageModeler__{attr_i}', val_i)
                    
        
    @staticmethod
    def get_mecpx_base_dirs(
        dataset                    , 
        acq_run_date               , 
        data_date_ranges           , 
        data_evs_sum_vw            , 
        acq_run_date_subdir_appndx = None, 
        data_dir_base              = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data', 
        assert_found               = True
    ):
        r"""
        Find the directories housing the data to be used for build_and_combine_mecpo_colls_for_dates.
        A directory for each date range in data_date_ranges will be found as:
            rcpo_pkl_dir_i = os.path.join(
                data_dir_base, 
                acq_run_date, 
                date_pd_subdir, 
                ODI.get_subdir(dataset), 
                end_events_method or evs_sum_vw_method (depending on data_evs_sum_vw value)
            )
        where date_pd_subdir corresponds to the particular date range, dataset_subdirs can be found in OutageDataInfo class (in OutageDAQ module)
        -------------------------
        dataset:
            Must be one of ['outg', 'otbl', 'prbl']
            
        acq_run_date:
            The acquisition date of the data (i.e., the date the acquisition program was run to save the needed 
              data locally)
            
        data_date_ranges:
            The date ranges to be included in the MECPOCollection.
            This should be a list of two-element.
                
        data_dir_base:
            Tbe base directory housing the data.
            Typically this should be 
                r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data' or
                r'U:\CloudData\dovs_and_end_events_data'
        """
        #--------------------------------------------------
        ODI.assert_dataset(dataset)
        #-----
        assert(Utilities.is_object_one_of_types(data_date_ranges, [list, tuple]))
        assert(Utilities.are_list_elements_lengths_homogeneous(data_date_ranges, length=2))
        #-------------------------
        acq_run_date_subdir = acq_run_date
        if acq_run_date_subdir_appndx is not None:
            acq_run_date_subdir += acq_run_date_subdir_appndx
        #-------------------------
        if data_evs_sum_vw:
            method_subdir = 'evs_sum_vw_method'
        else:
            method_subdir = 'end_events_method'
        #-------------------------
        data_dirs_dict = {}
        for date_0, date_1 in data_date_ranges:
            date_pd_subdir = f"{date_0.replace('-','')}_{date_1.replace('-','')}"
            data_dir_i = os.path.join(
                data_dir_base, 
                acq_run_date_subdir, 
                date_pd_subdir, 
                ODI.get_subdir(dataset), 
                method_subdir
            )
            if assert_found and not os.path.isdir(data_dir_i):
                print(f'Directory DNE!\n\t{data_dir_i}\nCRASH IMMINENT!!!!!!')
                assert(0)
            assert(date_pd_subdir not in data_dirs_dict.keys())
            data_dirs_dict[date_pd_subdir] = data_dir_i
        #--------------------------------------------------
        return data_dirs_dict
    
    def get_data_base_dirs(
        self, 
        dataset      = None, 
        assert_found = True
    ):
        r"""
        Returns a dict object containing the base directory for each data_date_range
          (i.e., data_date_range keys and base directory values)
        If dataset is None, such a dict is returned for each of the accptbl_datasets
          defined below.
        -----
        dataset:
            If not None:
                Must be one of accptbl_datasets = ['outg', 'otbl', 'prbl']
                Returns a dict object containing the base directory for each data_date_range
            If None:
                Routine is run for each of the accptbl_datasets = ['outg', 'otbl', 'prbl'].
                Returns a dict with keys equal to the accptbl_datasets and values equal
                  to dict object described above.
        """
        #--------------------------------------------------
        # Acceptable datasets 
        accptbl_datasets = ['outg', 'otbl']
        if self.include_prbl:
            accptbl_datasets.append('prbl')
        #--------------------------------------------------
        if dataset is None:
            return_dict = {}
            for dataset_i in accptbl_datasets:
                base_dirs_i = self.get_data_base_dirs(
                    dataset      = dataset_i, 
                    assert_found = assert_found
                )
                assert(dataset_i not in return_dict.keys())
                return_dict[dataset_i] = base_dirs_i
            return return_dict
        #--------------------------------------------------
        assert(dataset in accptbl_datasets)
        #-------------------------
        acq_run_date     = self.mecpx_build_info_dict[f'acq_run_date_{dataset}']
        data_date_ranges = self.mecpx_build_info_dict[f'data_date_ranges_{dataset}']
        data_dir_base    = self.mecpx_build_info_dict[f'data_dir_base_{dataset}']
        #-------------------------
        base_dirs = OutageModeler.get_mecpx_base_dirs(
            dataset          = dataset, 
            acq_run_date     = acq_run_date, 
            data_date_ranges = data_date_ranges, 
            data_evs_sum_vw  = self.mecpx_build_info_dict['data_evs_sum_vw'], 
            data_dir_base    = data_dir_base, 
            assert_found     = assert_found
        )
        #-------------------------
        return base_dirs

    def find_all_data_base_dirs(
        self
    ):
        r"""
        Run self.get_data_base_dirs with dataset=None and assert_found=False to compile 
          all expected base dirs for all datasets, then return only those base_dirs 
          which actually exist.
        """
        #-------------------------
        base_dirs_by_dataset = self.get_data_base_dirs(
            dataset      = None, 
            assert_found = False
        )
        #-------------------------
        return_dict = {}
        for dataset_i, base_dirs_i in base_dirs_by_dataset.items():
            assert(dataset_i not in return_dict.keys())
            found_dirs_i = {}
            for date_pd_j, base_dir_j in base_dirs_i.items():
                assert(date_pd_j not in found_dirs_i.keys())
                if os.path.isdir(base_dir_j):
                    found_dirs_i[date_pd_j] = base_dir_j
            return_dict[dataset_i] = found_dirs_i
        #-------------------------
        return return_dict
    
    @staticmethod
    def get_rcpo_pkl_dirs_for_build_and_combine_mecpo_colls_for_dates(
        dataset, 
        acq_run_date, 
        data_date_ranges, 
        grp_by_cols, 
        data_evs_sum_vw, 
        data_dir_base = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
    ):
        r"""
        Find the directories housing the data to be used for build_and_combine_mecpo_colls_for_dates.
        A directory for each date range in data_date_ranges will be found as:
            rcpo_pkl_dir_i = os.path.join(
                data_dir_base, 
                acq_run_date, 
                date_pd_subdir, 
                ODI.get_subdir(dataset), 
                data_subdir
            )
        where date_pd_subdir corresponds to the particular date range, dataset_subdirs can be found in OutageDataInfo class (in OutageDAQ module)
        and data_subdir is determined from grp_by_cols
        -------------------------
        dataset:
            Must be one of ['outg', 'otbl', 'prbl']
            
        acq_run_date:
            The acquisition date of the data (i.e., the date the acquisition program was run to save the needed 
              data locally)
            
        data_date_ranges:
            The date ranges to be included in the MECPOCollection.
            This should be a list of two-element.
            
        grp_by_cols:
            The columns which were grouped in the packaging of the data.
            Typically, this is:
                ['outg_rec_nb', 'trsf_pole_nb'] for outage data
                ['trsf_pole_nb', 'no_outg_rec_nb'] for baseline data
                
        data_dir_base:
            Tbe base directory housing the data.
            Typically this should be 
                r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data' or
                r'U:\CloudData\dovs_and_end_events_data'
        """
        #--------------------------------------------------
        data_subdir = 'rcpo_dfs'
        if grp_by_cols==['outg_rec_nb', 'trsf_pole_nb']:
            data_subdir += '_GRP_BY_OUTG_AND_XFMR'
        elif grp_by_cols=='trsf_pole_nb':
            data_subdir += '_GRP_BY_XFMR'
        elif grp_by_cols=='outg_rec_nb':
            data_subdir += '_GRP_BY_OUTG'
        elif grp_by_cols==['trsf_pole_nb', 'no_outg_rec_nb']:
            data_subdir += '_GRP_BY_NO_OUTG_AND_XFMR'
        else:
            assert(0)
        #--------------------------------------------------
        data_dirs_dict = OutageModeler.get_mecpx_base_dirs(
            dataset          = dataset, 
            acq_run_date     = acq_run_date, 
            data_date_ranges = data_date_ranges, 
            data_evs_sum_vw  = data_evs_sum_vw, 
            data_dir_base    = data_dir_base, 
            assert_found     = True
        )
        rcpo_pkl_dirs = {}
        for date_range_i, data_dir_i in data_dirs_dict.items():
            rcpo_pkl_dir_i = os.path.join(
                data_dir_i, 
                data_subdir
            )
            if not os.path.isdir(rcpo_pkl_dir_i):
                print(f'Directory DNE!\n\t{rcpo_pkl_dir_i}\nCRASH IMMINENT!!!!!!')
                assert(0)
            assert(date_range_i not in rcpo_pkl_dirs.keys())
            rcpo_pkl_dirs[date_range_i] = rcpo_pkl_dir_i
        #--------------------------------------------------
        return rcpo_pkl_dirs
    
    @staticmethod
    def build_and_combine_mecpo_colls_for_dates(
        dataset, 
        acq_run_date, 
        data_date_ranges, 
        grp_by_cols, 
        data_evs_sum_vw, 
        days_min_max_outg_td_windows, 
        old_to_new_keys_dict, 
        coll_label=None, 
        barplot_kwargs_shared=None, 
        normalize_by_time_interval=True, 
        data_dir_base=r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
    ):
        r"""
        Build a MECPOCollection object for data which are split over multiple date ranges (e.g., in a typical case,
          data are acquired for each calendar year).
          
        NOTE: One could build all mecpo_coll objects first, and then use combine_mecpo_colls to combine them.
              However, this method saves on memory consumption by building one at a time and combining with
                the growing aggregate.
          
        See get_rcpo_pkl_dirs_for_build_and_combine_mecpo_colls_for_dates for explanation of
            dataset, 
            acq_run_date, 
            data_date_ranges, 
            grp_by_cols, 
            data_dir_base
        """
        #--------------------------------------------------
        if old_to_new_keys_dict is not None:
            assert(len(old_to_new_keys_dict)==len(days_min_max_outg_td_windows))
        #-------------------------
        rcpo_pkl_dirs = OutageModeler.get_rcpo_pkl_dirs_for_build_and_combine_mecpo_colls_for_dates(
            dataset          = dataset, 
            acq_run_date     = acq_run_date, 
            data_date_ranges = data_date_ranges, 
            grp_by_cols      = grp_by_cols, 
            data_evs_sum_vw  = data_evs_sum_vw, 
            data_dir_base    = data_dir_base
        )
        #-------------------------
        if coll_label is None:
            coll_label = ODI.get_subdir(dataset) + Utilities.generate_random_string(str_len=4)
        if barplot_kwargs_shared is None:
            barplot_kwargs_shared = dict(facecolor='tab:blue')
        #-------------------------
        mecpo_coll = None
        for i, (date_range_i, rcpo_pkl_dir_i) in enumerate(rcpo_pkl_dirs.items()):
            mecpo_coll_i = MECPOCollection(
                data_type                    = dataset, 
                mecpo_coll                   = None, 
                coll_label                   = coll_label, 
                barplot_kwargs_shared        = barplot_kwargs_shared, 
                read_and_load_all_pickles    = True, 
                pkls_base_dir                = rcpo_pkl_dir_i,
                days_min_max_outg_td_windows = days_min_max_outg_td_windows, 
                pkls_sub_dirs                = None,
                naming_tag                   = ODI.get_naming_tag(dataset),
                normalize_by_time_interval   = normalize_by_time_interval, 
                are_no_outg                  = ODI.get_is_no_outage(dataset) 
            )
            if old_to_new_keys_dict is not None:
                mecpo_coll_i.change_mecpo_an_keys(old_to_new_keys_dict)
    
            if i==0:
                mecpo_coll = copy.deepcopy(mecpo_coll_i)
            else:
                mecpo_coll = MECPOCollection.combine_two_mecpo_colls(
                    mecpo_coll_1 = mecpo_coll, 
                    mecpo_coll_2 = mecpo_coll_i, 
                    append_only  = True
                )
            del mecpo_coll_i
        #-------------------------
        return mecpo_coll
        
        
    @staticmethod
    def build_mecpx_colls_for_modeler(
        acq_run_date, 
        data_date_ranges, 
        data_dir_base, 
        days_min_max_outg_td_windows, 
        old_to_new_keys_dict, 
        data_evs_sum_vw, 
        grp_by_cols_outg = ['outg_rec_nb', 'trsf_pole_nb'], 
        grp_by_cols_bsln = ['trsf_pole_nb', 'no_outg_rec_nb'], 
        include_prbl=True, 
        normalize_by_time_interval=True, 
        **kwargs
    ):
        r"""
        Static method of OutageModeler.build_mecpx_colls (pre-dates aforementioned)
        Build the MECPOCollection objects for outage data (outg), outage transformers baseline data (otbl), and pristine baseline (prbl)
          to be used by the modeler.
        This function implicitly assumes outg, otbl, and prbl all share the same acq_run_date, data_date_ranges, and data_dir_base (and 
          that otbl and prbl share grp_by_cols_bsln).
        If outg, otbl, and/or prbl do not share the same acq_run_date, data_date_ranges, and/or data_dir_base, then the dataset specific 
          version of those parameters should be used (e.g., acq_run_date_outg, acq_run_date_otbl, acq_run_date_prbl)
    
        kwargs:
            As stated above, these are mainly used when outg, otbl, and/or prbl do not share the same input parameters.
            There are also kwargs hidden for setting coll_label, barplot_kwargs_shared
            Acceptable kwargs:
                Any of the following are acceptable with _outg, _otbl, or _prbl tags
                'acq_run_date',        (e.g., acq_run_date_outg, acq_run_date_otbl, acq_run_date_prbl)
                'data_date_ranges', 
                'data_dir_base', 
                'grp_by_cols', 
                'coll_label', 
                'barplot_kwargs_shared'
        """
        #----------------------------------------------------------------------------------------------------
        acceptable_kwargs = ['acq_run_date', 'data_date_ranges', 'data_dir_base', 'coll_label', 'barplot_kwargs_shared']
        acceptable_kwargs = list(itertools.chain.from_iterable([[f'{x}_outg', f'{x}_otbl', f'{x}_prbl'] for x in acceptable_kwargs]))
        acceptable_kwargs.extend(['grp_by_cols_otbl', 'grp_by_cols_prbl'])
        assert(set(kwargs.keys()).difference(set(acceptable_kwargs))==set())
        
        #----- Outages (_outg) ------------------------------------------------------------------------------
        acq_run_date_outg          = kwargs.get('acq_run_date_outg',          acq_run_date)
        data_date_ranges_outg      = kwargs.get('data_date_ranges_outg',      data_date_ranges)
        data_dir_base_outg         = kwargs.get('data_dir_base_outg',         data_dir_base)
        coll_label_outg            = kwargs.get('coll_label_outg',            'Outages')
        barplot_kwargs_shared_outg = kwargs.get('barplot_kwargs_shared_outg', dict(facecolor='red'))
        #-------------------------
        mecpx_coll_outg = OutageModeler.build_and_combine_mecpo_colls_for_dates(
            dataset                      = 'outg', 
            acq_run_date                 = acq_run_date_outg, 
            data_date_ranges             = data_date_ranges_outg, 
            grp_by_cols                  = grp_by_cols_outg, 
            data_evs_sum_vw              = data_evs_sum_vw, 
            days_min_max_outg_td_windows = days_min_max_outg_td_windows, 
            old_to_new_keys_dict         = old_to_new_keys_dict, 
            coll_label                   = coll_label_outg, 
            barplot_kwargs_shared        = barplot_kwargs_shared_outg, 
            normalize_by_time_interval   = normalize_by_time_interval, 
            data_dir_base                = data_dir_base_outg
        )
    
        #----- Outage Transformers Baseline (_otbl) ---------------------------------------------------------
        acq_run_date_otbl          = kwargs.get('acq_run_date_otbl',          acq_run_date)
        data_date_ranges_otbl      = kwargs.get('data_date_ranges_otbl',      data_date_ranges)
        data_dir_base_otbl         = kwargs.get('data_dir_base_otbl',         data_dir_base)
        grp_by_cols_otbl           = kwargs.get('grp_by_cols_otbl',           grp_by_cols_bsln)
        coll_label_otbl            = kwargs.get('coll_label_otbl',            'Baseline (OTBL)')
        barplot_kwargs_shared_otbl = kwargs.get('barplot_kwargs_shared_otbl', dict(facecolor='orange'))
        #-------------------------
        mecpx_coll_otbl = OutageModeler.build_and_combine_mecpo_colls_for_dates(
            dataset                      = 'otbl', 
            acq_run_date                 = acq_run_date_otbl, 
            data_date_ranges             = data_date_ranges_otbl, 
            grp_by_cols                  = grp_by_cols_otbl, 
            data_evs_sum_vw              = data_evs_sum_vw, 
            days_min_max_outg_td_windows = days_min_max_outg_td_windows, 
            old_to_new_keys_dict         = old_to_new_keys_dict, 
            coll_label                   = coll_label_otbl, 
            barplot_kwargs_shared        = barplot_kwargs_shared_otbl, 
            normalize_by_time_interval   = normalize_by_time_interval, 
            data_dir_base                = data_dir_base_otbl
        )
    
        #----- Pristine Baseline (_prbl) --------------------------------------------------------------------
        if include_prbl:
            acq_run_date_prbl          = kwargs.get('acq_run_date_prbl',          acq_run_date)
            data_date_ranges_prbl      = kwargs.get('data_date_ranges_prbl',      data_date_ranges)
            data_dir_base_prbl         = kwargs.get('data_dir_base_prbl',         data_dir_base)
            grp_by_cols_prbl           = kwargs.get('grp_by_cols_prbl',           grp_by_cols_bsln)
            coll_label_prbl            = kwargs.get('coll_label_prbl',            'Baseline (PRBL)')
            barplot_kwargs_shared_prbl = kwargs.get('barplot_kwargs_shared_prbl', dict(facecolor='orange'))
            #-------------------------
            mecpx_coll_prbl = OutageModeler.build_and_combine_mecpo_colls_for_dates(
                dataset                      = 'prbl', 
                acq_run_date                 = acq_run_date_prbl, 
                data_date_ranges             = data_date_ranges_prbl, 
                grp_by_cols                  = grp_by_cols_prbl, 
                data_evs_sum_vw              = data_evs_sum_vw, 
                days_min_max_outg_td_windows = days_min_max_outg_td_windows, 
                old_to_new_keys_dict         = old_to_new_keys_dict, 
                coll_label                   = coll_label_prbl, 
                barplot_kwargs_shared        = barplot_kwargs_shared_prbl, 
                normalize_by_time_interval   = normalize_by_time_interval, 
                data_dir_base                = data_dir_base_prbl
            )
        else:
            mecpx_coll_prbl = None
    
        #----------------------------------------------------------------------------------------------------
        return_dict = dict(
            outg = mecpx_coll_outg, 
            otbl = mecpx_coll_otbl, 
            prbl = mecpx_coll_prbl
        )
        return return_dict
        
        
    def set_mecpx_colls(
        self, 
        mecpx_coll_outg, 
        mecpx_coll_otbl, 
        mecpx_coll_prbl = None
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(mecpx_coll_outg, MECPOCollection))
        assert(isinstance(mecpx_coll_otbl, MECPOCollection))
        if self.include_prbl:
            assert(isinstance(mecpx_coll_prbl, MECPOCollection))
        else:
            assert(mecpx_coll_prbl is None)
        #-------------------------
        self.mecpx_coll_outg = mecpx_coll_outg
        self.mecpx_coll_otbl = mecpx_coll_otbl
        self.mecpx_coll_prbl = mecpx_coll_prbl
        
    def try_to_determine_and_apply_new_an_keys(self):
        r"""
        This function will only attempt to do anything if self.mecpx_build_info_dict['old_to_new_keys_dict'] is None
        """
        #--------------------------------------------------
        if self.mecpx_build_info_dict['old_to_new_keys_dict'] is not None:
            return
        #--------------------------------------------------
        try:
            old_to_new_keys_dict = OutageModeler.build_old_to_new_an_keys(old_an_keys=self.mecpx_build_info_dict['mecpx_an_keys'])
        except:
            old_to_new_keys_dict = None
        #--------------------------------------------------
        if old_to_new_keys_dict is not None:
            assert(set(self.mecpx_coll_outg.mecpo_an_keys).difference(set(old_to_new_keys_dict.keys()))==set())
            assert(set(self.mecpx_coll_otbl.mecpo_an_keys).difference(set(old_to_new_keys_dict.keys()))==set())
            #-----
            self.mecpx_coll_outg.change_mecpo_an_keys(old_to_new_keys_dict)
            self.mecpx_coll_otbl.change_mecpo_an_keys(old_to_new_keys_dict)
            #-----
            an_keys = self.mecpx_coll_outg.mecpo_an_keys
            assert(set(self.mecpx_coll_otbl.mecpo_an_keys).symmetric_difference(set(an_keys))==set())
            #-------------------------
            if self.include_prbl:
                assert(set(self.mecpx_coll_prbl.mecpo_an_keys).difference(set(old_to_new_keys_dict.keys()))==set())
                self.mecpx_coll_prbl.change_mecpo_an_keys(old_to_new_keys_dict)
                assert(set(self.mecpx_coll_otbl.mecpo_an_keys).symmetric_difference(set(an_keys))==set())
            #-------------------------
            self.mecpx_build_info_dict['old_to_new_keys_dict'] = old_to_new_keys_dict
            #-----
            assert(set(an_keys).symmetric_difference(set(old_to_new_keys_dict.values()))==set())
            self.mecpx_build_info_dict['mecpx_an_keys'] = an_keys
            
    def save_mecpx_build_info_dict(self):
        r"""
        """
        #-------------------------
        if self.__saved_mecpx_build_info_dict:
            return
        #-------------------------
        assert(os.path.isdir(self.save_base_dir))
        CustomWriter.output_dict_to_json(
            os.path.join(self.save_base_dir, 'mecpx_build_info_dict.json'), 
            self.mecpx_build_info_dict
        )
        self.__saved_mecpx_build_info_dict = True
            
    def save_mecpx_colls(self):
        r"""
        """
        #-------------------------
        if self.__saved_mecpo_colls:
            return
        #-------------------------
        assert(self.mecpx_coll_outg is not None and self.mecpx_coll_otbl is not None)
        if not os.path.isdir(self.save_base_dir):
            print(f'OutageModeler.save_mecpx_colls - Directory DNE!: {self.save_base_dir}\nCRASH IMMINENT!')
        assert(os.path.isdir(self.save_base_dir))
        #-----
        with open(os.path.join(self.save_base_dir, 'mecpo_coll_outg.pkl'), 'wb') as handle:
            pickle.dump(self.mecpx_coll_outg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.save_base_dir, 'mecpo_coll_otbl.pkl'), 'wb') as handle:
            pickle.dump(self.mecpx_coll_otbl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #-----
        if self.include_prbl:
            with open(os.path.join(self.save_base_dir, 'mecpo_coll_prbl.pkl'), 'wb') as handle:
                pickle.dump(self.mecpx_coll_prbl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #-------------------------
        self.__saved_mecpo_colls = True
        #-------------------------
        self.save_mecpx_build_info_dict()
        
    def save_merged_dfs(
        self, 
        tag=None
    ):
        r"""
        tag:
            If None,   save_name = 'merged_df_xxxx'
            Otherwise, save_name = f'merged_df_xxxx_{tag}'
        """
        #-------------------------
        if tag is None and self.__saved_merged_dfs:
            return
        if (tag=='0' or tag==0) and self.__saved_merged_dfs_0:
            return
        #-------------------------
        assert(self.merged_df_outg is not None and self.merged_df_otbl is not None)
        assert(os.path.isdir(self.save_base_dir))
        #-----
        if tag is None:
            appndx = ''
        else:
            appndx = f'_{tag}'
        #-----
        self.merged_df_outg.to_pickle(os.path.join(self.save_base_dir, f'merged_df_outg{appndx}.pkl'))
        self.merged_df_otbl.to_pickle(os.path.join(self.save_base_dir, f'merged_df_otbl{appndx}.pkl'))
        #-----
        if self.include_prbl:
            self.merged_df_prbl.to_pickle(os.path.join(self.save_base_dir, f'merged_df_prbl{appndx}.pkl'))
        #-------------------------  
        if tag is None:
            self.__saved_merged_dfs   = True
        if (tag=='0' or tag==0):
            self.__saved_merged_dfs_0 = True
    
    
    def save_counts_series(
        self, 
        tag=None
    ):
        r"""
        tag:
            If None,   save_name = 'counts_series_xxxx'
            Otherwise, save_name = f'counts_series_xxxx_{tag}'
        """
        #-------------------------
        if tag is None and self.__saved_counts_series:
            return
        if (tag=='0' or tag==0) and self.__saved_counts_series_0:
            return
        #-------------------------
        assert(self.counts_series_outg is not None and self.counts_series_otbl is not None)
        assert(os.path.isdir(self.save_base_dir))
        #-------------------------
        if tag is None:
            appndx = ''
        else:
            appndx = f'_{tag}'
        #-------------------------
        with open(os.path.join(self.save_base_dir, f'counts_series_outg{appndx}.pkl'), 'wb') as handle:
            pickle.dump(self.counts_series_outg, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.save_base_dir, f'counts_series_otbl{appndx}.pkl'), 'wb') as handle:
            pickle.dump(self.counts_series_otbl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #-----
        if self.counts_series_prbl is not None:
            with open(os.path.join(self.save_base_dir, f'counts_series_prbl{appndx}.pkl'), 'wb') as handle:
                pickle.dump(self.counts_series_prbl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #-------------------------
        if tag is None:
            self.__saved_counts_series   = True
        if (tag=='0' or tag==0):
            self.__saved_counts_series_0 = True
        
    def save_all_mecpx(
        self, 
        tag=None
    ):
        r"""
        Save all MECPOCollection objects and all associated objects (i.e., merged_dfs, counts_series, etc.)
        """
        #-------------------------
        self.save_mecpx_colls()
        self.save_merged_dfs(tag=tag)
        self.save_counts_series(tag=tag)
        
        
    def load_mecpx_build_info_dict(
        self,
        verbose=True
    ):
        r"""
        """
        #-------------------------
        assert(self.is_save_base_dir_loaded())
        tpath = os.path.join(self.save_base_dir, 'mecpx_build_info_dict.json')
        assert(os.path.exists(tpath))
        #-----
        with open(tpath, 'rb') as handle:
            self.mecpx_build_info_dict = json.load(handle)
        #-------------------------
        # Any tuples are saved/retrieved from json files as lists.
        # In most cases, this distinction does not matter, but it does for rec_nb_idfr and trsf_pole_nb_idfr
        if isinstance(self.mecpx_build_info_dict['rec_nb_idfr'], list):
            self.mecpx_build_info_dict['rec_nb_idfr'] = tuple(self.mecpx_build_info_dict['rec_nb_idfr'])
        #-----
        if isinstance(self.mecpx_build_info_dict['trsf_pole_nb_idfr'], list):
            self.mecpx_build_info_dict['trsf_pole_nb_idfr'] = tuple(self.mecpx_build_info_dict['trsf_pole_nb_idfr'])
        #-------------------------
        if verbose:
            print('Successfully loaded mecpx_build_info_dict')
        #----------------------------------------------------------------------------------------------------
        # Since mecpx_build_info_dict loaded from file, there will be no reason to try to save again.
        self.__saved_mecpx_build_info_dict = True 
        
        
    def load_mecpx_colls_from_pkls(
        self, 
        verbose=True
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        assert(os.path.isdir(self.save_base_dir))
        assert(os.path.exists(os.path.join(self.save_base_dir, 'mecpo_coll_outg.pkl')))
        assert(os.path.exists(os.path.join(self.save_base_dir, 'mecpo_coll_otbl.pkl')))
        #----------------------------------------------------------------------------------------------------
        with open(os.path.join(self.save_base_dir, 'mecpo_coll_outg.pkl'), 'rb') as handle:
            mecpx_coll_outg = pickle.load(handle)
        an_keys = mecpx_coll_outg.mecpo_an_keys
        #-------------------------
        with open(os.path.join(self.save_base_dir, 'mecpo_coll_otbl.pkl'), 'rb') as handle:
            mecpx_coll_otbl = pickle.load(handle)
        assert(set(mecpx_coll_otbl.mecpo_an_keys).symmetric_difference(set(an_keys))==set())
        #-------------------------
        loaded = [
            f"mecpo_coll_outg: {os.path.join(self.save_base_dir, 'mecpo_coll_outg.pkl')}", 
            f"mecpo_coll_otbl: {os.path.join(self.save_base_dir, 'mecpo_coll_otbl.pkl')}"
        ]
        #-------------------------
        if self.include_prbl:
            assert(os.path.exists(os.path.join(self.save_base_dir, 'mecpo_coll_prbl.pkl')))
            with open(os.path.join(self.save_base_dir, 'mecpo_coll_prbl.pkl'), 'rb') as handle:
                mecpx_coll_prbl = pickle.load(handle)
            loaded.append(f"mecpo_coll_prbl: {os.path.join(self.save_base_dir, 'mecpo_coll_prbl.pkl')}")
            assert(set(mecpx_coll_prbl.mecpo_an_keys).symmetric_difference(set(an_keys))==set())
        else:
            mecpx_coll_prbl = None
        #----------------------------------------------------------------------------------------------------
        self.set_mecpx_colls(
            mecpx_coll_outg = mecpx_coll_outg, 
            mecpx_coll_otbl = mecpx_coll_otbl, 
            mecpx_coll_prbl = mecpx_coll_prbl, 
        )
        #--------------------------------------------------
        self.load_mecpx_build_info_dict()
        #-------------------------
        if self.mecpx_build_info_dict['mecpx_an_keys'] is None:
            self.mecpx_build_info_dict['mecpx_an_keys'] = an_keys
        else:
            assert(set(self.mecpx_build_info_dict['mecpx_an_keys']).symmetric_difference(set(an_keys))==set())
        #-------------------------
        # If self.mecpx_build_info_dict['old_to_new_keys_dict'] not supplied, try to determine it and apply it
        # NOTE: If it was supplied, it would have been applied in OutageModeler.build_and_combine_mecpo_colls_for_dates calls above
        if self.mecpx_build_info_dict['old_to_new_keys_dict'] is None:
            self.try_to_determine_and_apply_new_an_keys() 
        #----------------------------------------------------------------------------------------------------
        if verbose:
            print('Successfully loaded MECPOCollection objects:')
            print(*loaded, sep='\n')
            
        #----------------------------------------------------------------------------------------------------
        # Since self.mecpx_coll_... were loaded from file, there will be no reason to try to save them again.
        self.__saved_mecpo_colls = True
        
    def build_mecpx_colls(
        self
    ):
        r"""
        Build the MECPOCollection objects for outage data (outg), outage transformers baseline data (otbl), and pristine baseline (prbl)
          to be used by the modeler.
        """
        #----------------------------------------------------------------------------------------------------
        # Since building new, set __saved... = False
        self.__saved_mecpx_build_info_dict = False
        self.__saved_mecpo_colls           = False
        #----------------------------------------------------------------------------------------------------
        # Make sure self.mecpx_build_info_dict looks good
        #----------------------------------------------------------------------------------------------------
        assert(OutageModeler.check_mecpx_build_info_dict(self.mecpx_build_info_dict))
        
        #----------------------------------------------------------------------------------------------------
        # Build the MECPOCollection objects
        #----------------------------------------------------------------------------------------------------
        #----- Outages (_outg) ----------------------------
        mecpx_coll_outg = OutageModeler.build_and_combine_mecpo_colls_for_dates(
            dataset                      = 'outg', 
            acq_run_date                 = self.mecpx_build_info_dict['acq_run_date_outg'], 
            data_date_ranges             = self.mecpx_build_info_dict['data_date_ranges_outg'], 
            grp_by_cols                  = self.mecpx_build_info_dict['grp_by_cols_outg'], 
            data_evs_sum_vw              = self.mecpx_build_info_dict['data_evs_sum_vw'], 
            days_min_max_outg_td_windows = self.mecpx_build_info_dict['days_min_max_outg_td_windows'], 
            old_to_new_keys_dict         = self.mecpx_build_info_dict['old_to_new_keys_dict'], 
            coll_label                   = self.mecpx_build_info_dict['coll_label_outg'], 
            barplot_kwargs_shared        = self.mecpx_build_info_dict['barplot_kwargs_shared_outg'], 
            normalize_by_time_interval   = self.mecpx_build_info_dict['normalize_by_time_interval'], 
            data_dir_base                = self.mecpx_build_info_dict['data_dir_base_outg']
        )
        an_keys = mecpx_coll_outg.mecpo_an_keys
    
        #----- Outage Transformers Baseline (_otbl) -------
        mecpx_coll_otbl = OutageModeler.build_and_combine_mecpo_colls_for_dates(
            dataset                      = 'otbl', 
            acq_run_date                 = self.mecpx_build_info_dict['acq_run_date_otbl'], 
            data_date_ranges             = self.mecpx_build_info_dict['data_date_ranges_otbl'], 
            grp_by_cols                  = self.mecpx_build_info_dict['grp_by_cols_otbl'], 
            data_evs_sum_vw              = self.mecpx_build_info_dict['data_evs_sum_vw'], 
            days_min_max_outg_td_windows = self.mecpx_build_info_dict['days_min_max_outg_td_windows'], 
            old_to_new_keys_dict         = self.mecpx_build_info_dict['old_to_new_keys_dict'], 
            coll_label                   = self.mecpx_build_info_dict['coll_label_otbl'], 
            barplot_kwargs_shared        = self.mecpx_build_info_dict['barplot_kwargs_shared_otbl'], 
            normalize_by_time_interval   = self.mecpx_build_info_dict['normalize_by_time_interval'], 
            data_dir_base                = self.mecpx_build_info_dict['data_dir_base_otbl']
        )
        assert(set(mecpx_coll_otbl.mecpo_an_keys).symmetric_difference(set(an_keys))==set())
    
        #----- Pristine Baseline (_prbl) ------------------
        if self.include_prbl:
            mecpx_coll_prbl = OutageModeler.build_and_combine_mecpo_colls_for_dates(
                dataset                      = 'prbl', 
                acq_run_date                 = self.mecpx_build_info_dict['acq_run_date_prbl'], 
                data_date_ranges             = self.mecpx_build_info_dict['data_date_ranges_prbl'], 
                grp_by_cols                  = self.mecpx_build_info_dict['grp_by_cols_prbl'], 
                data_evs_sum_vw              = self.mecpx_build_info_dict['data_evs_sum_vw'], 
                days_min_max_outg_td_windows = self.mecpx_build_info_dict['days_min_max_outg_td_windows'], 
                old_to_new_keys_dict         = self.mecpx_build_info_dict['old_to_new_keys_dict'], 
                coll_label                   = self.mecpx_build_info_dict['coll_label_prbl'], 
                barplot_kwargs_shared        = self.mecpx_build_info_dict['barplot_kwargs_shared_prbl'], 
                normalize_by_time_interval   = self.mecpx_build_info_dict['normalize_by_time_interval'], 
                data_dir_base                = self.mecpx_build_info_dict['data_dir_base_prbl']
            )
            assert(set(mecpx_coll_prbl.mecpo_an_keys).symmetric_difference(set(an_keys))==set())
        else:
            mecpx_coll_prbl = None
        
        #----------------------------------------------------------------------------------------------------
        self.set_mecpx_colls(
            mecpx_coll_outg = mecpx_coll_outg, 
            mecpx_coll_otbl = mecpx_coll_otbl, 
            mecpx_coll_prbl = mecpx_coll_prbl, 
        )
        self.mecpx_build_info_dict['mecpx_an_keys'] = an_keys
        #----------------------------------------------------------------------------------------------------
        # If self.mecpx_build_info_dict['old_to_new_keys_dict'] not supplied, try to determine it and apply it
        # NOTE: If it was supplied, it would have been applied in OutageModeler.build_and_combine_mecpo_colls_for_dates calls above
        if self.mecpx_build_info_dict['old_to_new_keys_dict'] is None:
            self.try_to_determine_and_apply_new_an_keys() 
                
        #----------------------------------------------------------------------------------------------------
        if self.__save_data:
            self.save_mecpx_colls()
            
    def build_or_load_mecpx_colls(
        self, 
        verbose      = True
    ):
        r"""
        Returns boolean, signifying whether (True) or not (False) the collections were built.
        """
        #--------------------------------------------------
        if self.__force_build:
            self.build_mecpx_colls()
            return True
        #--------------------------------------------------
        load_colls = True
        #-----
        expected_dir_exists = self.is_save_base_dir_loaded(verbose=False) # Want verbose always False here
        if not expected_dir_exists:
            # If self.save_base_dir doesn't exist, we can't possibly load the collections!
            load_colls = False
        else:
            # If any of the pickle files does not exist, we cannot load the collections
            if not os.path.exists(os.path.join(self.save_base_dir, 'mecpo_coll_outg.pkl')):
                load_colls = False
            if not os.path.exists(os.path.join(self.save_base_dir, 'mecpo_coll_otbl.pkl')):
                load_colls = False
            if self.include_prbl and not os.path.exists(os.path.join(self.save_base_dir, 'mecpo_coll_prbl.pkl')):
                load_colls = False
        #--------------------------------------------------
        if load_colls:
            self.load_mecpx_colls_from_pkls(verbose=verbose)
            return False
        else:
            self.build_mecpx_colls()
            return True
        
        
    def build_mecpx_colls_w_args(
        self, 
        acq_run_date, 
        data_date_ranges, 
        data_dir_base, 
        days_min_max_outg_td_windows, 
        old_to_new_keys_dict, 
        grp_by_cols_outg = ['outg_rec_nb', 'trsf_pole_nb'], 
        grp_by_cols_bsln = ['trsf_pole_nb', 'no_outg_rec_nb'], 
        include_prbl=True, 
        normalize_by_time_interval=True, 
        **kwargs
    ):
        r"""
        Build the MECPOCollection objects for outage data (outg), outage transformers baseline data (otbl), and pristine baseline (prbl)
          to be used by the modeler.
        NOTE: self.mecpx_build_info_dict will be updated with the values supplied to this function!
              To run with the values currently in self.mecpx_build_info_dict, call build_mecpx_colls!
        This function implicitly assumes outg, otbl, and prbl all share the same acq_run_date, data_date_ranges, and data_dir_base (and 
          that otbl and prbl share grp_by_cols_bsln).
        If outg, otbl, and/or prbl do not share the same acq_run_date, data_date_ranges, and/or data_dir_base, then the dataset specific 
          version of those parameters should be used (e.g., acq_run_date_outg, acq_run_date_otbl, acq_run_date_prbl)
    
        kwargs:
            As stated above, these are mainly used when outg, otbl, and/or prbl do not share the same input parameters.
            There are also kwargs hidden for setting coll_label, barplot_kwargs_shared
            Acceptable kwargs:
                Any of the following are acceptable with _outg, _otbl, or _prbl tags
                'acq_run_date',        (e.g., acq_run_date_outg, acq_run_date_otbl, acq_run_date_prbl)
                'data_date_ranges', 
                'data_dir_base', 
                'grp_by_cols', 
                'coll_label', 
                'barplot_kwargs_shared'
        """
        #----------------------------------------------------------------------------------------------------
        acceptable_kwargs = ['acq_run_date', 'data_date_ranges', 'data_dir_base', 'coll_label', 'barplot_kwargs_shared']
        acceptable_kwargs = list(itertools.chain.from_iterable([[f'{x}_outg', f'{x}_otbl', f'{x}_prbl'] for x in acceptable_kwargs]))
        acceptable_kwargs.extend(['grp_by_cols_otbl', 'grp_by_cols_prbl'])
        assert(set(kwargs.keys()).difference(set(acceptable_kwargs))==set())
    
        #----------------------------------------------------------------------------------------------------
        # Using input arguments, set the apropriate values in self.mecpx_build_info_dict[
        #----------------------------------------------------------------------------------------------------
        #----- Used by all --------------------------------
        if self.mecpx_build_info_dict is None:
            self.mecpx_build_info_dict = {}
        self.mecpx_build_info_dict['days_min_max_outg_td_windows'] = days_min_max_outg_td_windows
        self.mecpx_build_info_dict['old_to_new_keys_dict']         = old_to_new_keys_dict
        self.mecpx_build_info_dict['normalize_by_time_interval']   = normalize_by_time_interval
        self.include_prbl                                          = include_prbl
    
        #----- Outages (_outg) ----------------------------
        self.mecpx_build_info_dict['acq_run_date_outg']          = kwargs.get('acq_run_date_outg',          acq_run_date)
        self.mecpx_build_info_dict['data_date_ranges_outg']      = kwargs.get('data_date_ranges_outg',      data_date_ranges)
        self.mecpx_build_info_dict['data_dir_base_outg']         = kwargs.get('data_dir_base_outg',         data_dir_base)
        self.mecpx_build_info_dict['grp_by_cols_outg']           = grp_by_cols_outg
        self.mecpx_build_info_dict['coll_label_outg']            = kwargs.get('coll_label_outg',            'Outages')
        self.mecpx_build_info_dict['barplot_kwargs_shared_outg'] = kwargs.get('barplot_kwargs_shared_outg', dict(facecolor='red'))
    
        #----- Outage Transformers Baseline (_otbl) -------
        self.mecpx_build_info_dict['acq_run_date_otbl']          = kwargs.get('acq_run_date_otbl',          acq_run_date)
        self.mecpx_build_info_dict['data_date_ranges_otbl']      = kwargs.get('data_date_ranges_otbl',      data_date_ranges)
        self.mecpx_build_info_dict['data_dir_base_otbl']         = kwargs.get('data_dir_base_otbl',         data_dir_base)
        self.mecpx_build_info_dict['grp_by_cols_otbl']           = kwargs.get('grp_by_cols_otbl',           grp_by_cols_bsln)
        self.mecpx_build_info_dict['coll_label_otbl']            = kwargs.get('coll_label_otbl',            'Baseline (OTBL)')
        self.mecpx_build_info_dict['barplot_kwargs_shared_otbl'] = kwargs.get('barplot_kwargs_shared_otbl', dict(facecolor='orange'))
    
        #----- Pristine Baseline (_prbl) ------------------
        self.mecpx_build_info_dict['acq_run_date_prbl']          = kwargs.get('acq_run_date_prbl',          acq_run_date)
        self.mecpx_build_info_dict['data_date_ranges_prbl']      = kwargs.get('data_date_ranges_prbl',      data_date_ranges)
        self.mecpx_build_info_dict['data_dir_base_prbl']         = kwargs.get('data_dir_base_prbl',         data_dir_base)
        self.mecpx_build_info_dict['grp_by_cols_prbl']           = kwargs.get('grp_by_cols_prbl',           grp_by_cols_bsln)
        self.mecpx_build_info_dict['coll_label_prbl']            = kwargs.get('coll_label_prbl',            'Baseline (PRBL)')
        self.mecpx_build_info_dict['barplot_kwargs_shared_prbl'] = kwargs.get('barplot_kwargs_shared_prbl', dict(facecolor='orange'))
    
        #----------------------------------------------------------------------------------------------------
        self.build_mecpx_colls()
        
    def perform_similarity_operations(
        self, 
        verbose=True
    ):
        r"""
        Make the columns equal in the relevant DFs from the MECPOCollections
        """
        #----------------------------------------------------------------------------------------------------
        assert(self.mecpx_coll_outg is not None and self.mecpx_coll_otbl is not None)
        mecpo_colls_with_cpo_df_names = [
            [self.mecpx_coll_outg, self.cpx_dfs_name_outg], 
            [self.mecpx_coll_otbl, self.cpx_dfs_name_otbl]
        ]
        #-----
        if self.include_prbl:
            mecpo_colls_with_cpo_df_names.append([self.mecpx_coll_prbl, self.cpx_dfs_name_prbl])
        #----------------------------------------------------------------------------------------------------
        if verbose:
            print('\n\nIn OutageModeler.perform_similarity_operations\nStarting shapes:')
            for i_coll, mecpo_coll_i_w_df_name in enumerate(mecpo_colls_with_cpo_df_names):
                print(f"MECPO Collection {i_coll}")
                for an_key in mecpo_coll_i_w_df_name[0].mecpo_an_keys:
                    print(f"\t\t{an_key}, '{mecpo_coll_i_w_df_name[1]}' shape: {mecpo_coll_i_w_df_name[0].get_cpo_df(an_key, mecpo_coll_i_w_df_name[1]).shape}")
        #----------------------------------------------------------------------------------------------------
        # First, make columns equal between MECPOAn objects within each MECPOCollection
        for mecpo_coll_i_w_df_name in mecpo_colls_with_cpo_df_names:
            mecpo_coll_i_w_df_name[0].make_cpo_columns_equal(drop_empty_cpo_dfs=True)
    
        #-------------------------
        # Now, make columns equal between the MECPOCollections
        MECPOCollection.make_cpo_columns_equal_between_mecpo_colls(
            mecpo_colls = [x[0] for x in mecpo_colls_with_cpo_df_names], 
            drop_empty_cpo_dfs=True
        )
    
        #-------------------------
        # If not all same cpo_df names are used between collections, then one should call 
        #   MECPOCollection.make_mixed_cpo_columns_equal_between_mecpo_colls.
        if len(set([x[1] for x in mecpo_colls_with_cpo_df_names]))>1:
            MECPOCollection.make_mixed_cpo_columns_equal_between_mecpo_colls(
                mecpo_colls_with_cpo_df_names = mecpo_colls_with_cpo_df_names, 
                segregate_by_mecpo_an_keys=False
            )
            
        #----------------------------------------------------------------------------------------------------
        if verbose:
            print('\n\nAfter making columns equal amongst collections:')
            for i_coll, mecpo_coll_i_w_df_name in enumerate(mecpo_colls_with_cpo_df_names):
                print(f"MECPO Collection {i_coll}")
                for an_key in mecpo_coll_i_w_df_name[0].mecpo_an_keys:
                    print(f"\t\t{an_key}, '{mecpo_coll_i_w_df_name[1]}' shape: {mecpo_coll_i_w_df_name[0].get_cpo_df(an_key, mecpo_coll_i_w_df_name[1]).shape}")
                    
                    
    # NOTE: This is resource wasting as it performs reductions on all DFs, not just those of the appropriate name
    def perform_reduction_operations(
        self, 
        verbose=True
    ):
        r"""
        Remove and/or combine reasons
        Also, insert power_down_minus_up column if set.
        """
        #----------------------------------------------------------------------------------------------------
        assert(self.mecpx_coll_outg is not None and self.mecpx_coll_otbl is not None)
        mecpo_colls_with_cpo_df_names = [
            [self.mecpx_coll_outg, self.cpx_dfs_name_outg], 
            [self.mecpx_coll_otbl, self.cpx_dfs_name_otbl]
        ]
        #-----
        if self.include_prbl:
            mecpo_colls_with_cpo_df_names.append([self.mecpx_coll_prbl, self.cpx_dfs_name_prbl])
        #----------------------------------------------------------------------------------------------------
        if verbose:
            print('\n\nIn OutageModeler.perform_reduction_operations\nStarting shapes:')
            for i_coll, mecpo_coll_i_w_df_name in enumerate(mecpo_colls_with_cpo_df_names):
                print(f"MECPO Collection {i_coll}")
                for an_key in mecpo_coll_i_w_df_name[0].mecpo_an_keys:
                    print(f"\t\t{an_key}, '{mecpo_coll_i_w_df_name[1]}' shape: {mecpo_coll_i_w_df_name[0].get_cpo_df(an_key, mecpo_coll_i_w_df_name[1]).shape}")
    
        #----------------------------------------------------------------------------------------------------
        # Remove and/or combine reasons
        #----------------------------------------------------------------------------------------------------
        #-------------------------
        # Remove all reasons containing 'cleared'
        for mecpo_coll_i_w_df_name in mecpo_colls_with_cpo_df_names:
            mecpo_coll_i_w_df_name[0].remove_reasons_from_all_rcpo_dfs(
                regex_patterns_to_remove = self.mecpx_build_info_dict['regex_to_remove_patterns'], 
                ignore_case              = self.mecpx_build_info_dict['regex_to_remove_ignore_case']
            )
    
        #-------------------------
        # Combine reasons using the standard combine (see dflt_patterns_and_replace in MECPODf.combine_cpo_df_reasons
        #   for the list of default patterns_and_replace)
        for mecpo_coll_i_w_df_name in mecpo_colls_with_cpo_df_names:
            mecpo_coll_i_w_df_name[0].combine_reasons_in_all_rcpo_dfs(**self.combine_reasons_kwargs)
    
        #-------------------------
        # Build power down minus power up counts
        if self.mecpx_build_info_dict['include_power_down_minus_up']:
            for mecpo_coll_i_w_df_name in mecpo_colls_with_cpo_df_names:
                mecpo_coll_i_w_df_name[0].delta_cpo_df_reasons_in_all_rcpo_dfs(
                    reasons_1         = self.mecpx_build_info_dict['pd_col'],
                    reasons_2         = self.mecpx_build_info_dict['pu_col'],
                    delta_reason_name = self.mecpx_build_info_dict['pd_m_pu_col']
                )        
        #-------------------------
        # Don't want to include SNs or nSNs cols (and similar) in plotting, so remove
        for mecpo_coll_i_w_df_name in mecpo_colls_with_cpo_df_names:
            mecpo_coll_i_w_df_name[0].remove_SNs_cols_from_all_cpo_dfs()
        
        #----------------------------------------------------------------------------------------------------
        if verbose:
            print('\n\nAfter removing and/or combining reasons:')
            for i_coll, mecpo_coll_i_w_df_name in enumerate(mecpo_colls_with_cpo_df_names):
                print(f"MECPO Collection {i_coll}")
                for an_key in mecpo_coll_i_w_df_name[0].mecpo_an_keys:
                    print(f"\t\t{an_key}, '{mecpo_coll_i_w_df_name[1]}' shape: {mecpo_coll_i_w_df_name[0].get_cpo_df(an_key, mecpo_coll_i_w_df_name[1]).shape}")
                    
                    
    @staticmethod
    def standardize_index_names(
        to_adjust, 
        std_dict
    ):
        r"""
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(to_adjust, [pd.DataFrame, pd.Series]))
        #-------------------------
        if std_dict is None:
            return to_adjust
        #-------------------------
        assert(isinstance(std_dict, dict))
        assert(set(to_adjust.index.names).difference(set(std_dict.keys()))==set())
        std_idx_names = [std_dict[x] for x in to_adjust.index.names]
        #-------------------------
        to_adjust.index.names = std_idx_names
        return to_adjust
        
        
    @staticmethod
    def update_counts_series(
        merged_df, 
        counts_series
    ):
        r"""
        After a merged_df is updated (via slicing or whatever), counts_series should be updated as well.
        This is achieved by selecting using the indices.
        Therefore, the indices must be meaningfuly, and comparable, which is why we ensure they are named and
          share the same name(s)
        """
        #-------------------------
        # Make sure index names are equal and all levels are unique
        assert(merged_df.index.names==counts_series.index.names)
        assert(not any([x is None for x in merged_df.index.names]))
        assert(len(set(merged_df.index.names))==merged_df.index.nlevels)
        #-------------------------
        # Make sure counts_series contains at minimum the entries in merged_df
        assert(set(merged_df.index).difference(set(counts_series.index))==set())
        #-------------------------
        # Select subset of counts_series using index
        return_counts_series = counts_series.loc[merged_df.index].copy()
        #-----
        # Sanity check...
        assert(set(merged_df.index).symmetric_difference(set(return_counts_series.index))==set())
        #-------------------------
        return return_counts_series
        
    def reduce_time_infos_and_eemsp(
        self
    ):
        r"""
        After new model is sliced off, the full time_infos_df and eemsp_df are no longer needed.
        Remove the unnecessary rows to free up memory.
        
        This is somewhat restrictive, but is flexible enough for the foreseeable future.
        After self.gnrl_slicer is applied, grab to unique set of (rec_nb_i, trsf_pole_nb_i) which remain in the data, rec_nb_trsf_pole_tuples
        Use this subset to select the subsets of time_infos_df and eemsp_df
        The restrictions come as:
          merged_df_outg/_otbl/_prbl must have rec_nb/trsf_pole_nb in index
          merged_df_outg/_otbl/_prbl must share the same index structure as time_infos_df AND 
          the following mapping from merged_df_xxxx to eemsp_df must hold:
              ('index', rec_idx_lvl)       --> eemsp_rec_nb_col
              ('index', trsf_pole_idx_lvl) --> eemsp_location_nb_col
        """
        #--------------------------------------------------
        # First, handle reduction of time_infos_df
        if self.time_infos_df is not None and self.time_infos_df.shape[0]>0:
            assert(self.mecpx_build_info_dict['rec_nb_idfr'][0]=='index')
            rec_idx_lvl_name = self.mecpx_build_info_dict['rec_nb_idfr'][1]
            #-----
            assert(self.mecpx_build_info_dict['trsf_pole_nb_idfr'][0]=='index')
            trsf_pole_idx_lvl_name = self.mecpx_build_info_dict['trsf_pole_nb_idfr'][1]
            #--------------------------------------------------
            assert(self.merged_df_outg.index.names==self.merged_df_otbl.index.names==self.time_infos_df.index.names)
            rec_nb_trsf_pole_tuples = natsorted(set(
                self.merged_df_outg.index.unique().tolist() + 
                self.merged_df_otbl.index.unique().tolist()
            ))
            #-------------------------
            if self.include_prbl:
                assert(self.merged_df_prbl.index.names==self.time_infos_df.index.names)
                rec_nb_trsf_pole_tuples = natsorted(set(
                    rec_nb_trsf_pole_tuples + 
                    self.merged_df_prbl.index.unique().tolist()
                ))
            #--------------------------------------------------
            self.time_infos_df = self.time_infos_df[self.time_infos_df.index.isin(rec_nb_trsf_pole_tuples)].copy()
        #--------------------------------------------------
        # Next, handle eemsp_df if it exists...
        if self.eemsp_df is not None and self.eemsp_df.shape[0]>0:
            eemsp_rec_nb_col      = self.eemsp_df_info_dict['rec_nb_to_merge_col']
            eemsp_location_nb_col = self.eemsp_df_info_dict['eemsp_location_nb_col']
            #-------------------------
            rec_idx_lvl       = Utilities_df.get_idfr_loc(self.merged_df_outg, self.mecpx_build_info_dict['rec_nb_idfr'])
            trsf_pole_idx_lvl = Utilities_df.get_idfr_loc(self.merged_df_outg, self.mecpx_build_info_dict['trsf_pole_nb_idfr'])
            #-----
            assert(rec_idx_lvl[1]==trsf_pole_idx_lvl[1]==True)
            #-----
            rec_idx_lvl       = rec_idx_lvl[0]
            trsf_pole_idx_lvl = trsf_pole_idx_lvl[0]
            assert(set([rec_idx_lvl, trsf_pole_idx_lvl])==set([0,1]))
            #-------------------------
            eemsp_df = self.eemsp_df.copy()
            #-------------------------
            if rec_idx_lvl < trsf_pole_idx_lvl:
                eemsp_df = eemsp_df.set_index([eemsp_rec_nb_col, eemsp_location_nb_col])
            elif trsf_pole_idx_lvl < rec_idx_lvl:
                eemsp_df = eemsp_df.set_index([eemsp_location_nb_col, eemsp_rec_nb_col])
            else:
                assert(0) # rec_idx_lvl should never equal trsf_pole_idx_lvl!
            #-------------------------
            eemsp_df = eemsp_df[eemsp_df.index.isin(rec_nb_trsf_pole_tuples)].copy()
            eemsp_df = eemsp_df.reset_index(drop=False)
            #-------------------------
            self.eemsp_df = eemsp_df.copy()
    
    
    def apply_gnrl_slicer(
        self
    ):
        r"""
        """
        #--------------------------------------------------
        # Make sure all model data (e.g., df_train/_test, etc.) are cleared out
        # This is important because we don't want underlying merged_dfs to be updated but
        #   X_test etc. remain same
        self.reset_model_results()
        #-------------------------
        if self.gnrl_slicer is None:
            return
        #-------------------------
        assert(Utilities.is_object_one_of_types(self.gnrl_slicer, [DFSlicer, DFSingleSlicer]))
        #-------------------------
        # NOTE: counts_series don't HAVE to be updated (as long as they contain, at minimum, the entries needed, 
        #         we're good), but they should be updated to save space
        self.merged_df_outg = self.gnrl_slicer.perform_slicing(self.merged_df_outg)
        self.merged_df_otbl = self.gnrl_slicer.perform_slicing(self.merged_df_otbl)
        #-----
        if self.counts_series_outg is not None and self.counts_series_outg.shape[0]>0:
            self.counts_series_outg = OutageModeler.update_counts_series(
                merged_df     = self.merged_df_outg, 
                counts_series = self.counts_series_outg
            )
        if self.counts_series_otbl is not None and self.counts_series_otbl.shape[0]>0:
            self.counts_series_otbl = OutageModeler.update_counts_series(
                merged_df     = self.merged_df_otbl, 
                counts_series = self.counts_series_otbl
            )
        
        if self.include_prbl:
            self.merged_df_prbl = self.gnrl_slicer.perform_slicing(self.merged_df_prbl)
            #-----
            if self.counts_series_prbl is not None and self.counts_series_prbl.shape[0]>0:
                self.counts_series_prbl = OutageModeler.update_counts_series(
                    merged_df     = self.merged_df_prbl, 
                    counts_series = self.counts_series_prbl
                )
        #--------------------------------------------------
        # Update time_infos_df and eemsp_df to free up memory
        self.reduce_time_infos_and_eemsp()
        

    def compile_merged_dfs_0(
        self
    ):
        r"""
        Get merged DFs from collections
        """
        #----------------------------------------------------------------------------------------------------
        # Since building new, set __saved... = False
        self.__saved_merged_dfs_0    = False
        self.__saved_counts_series_0 = False
        #----------------------------------------------------------------------------------------------------
        merged_df_outg = self.mecpx_coll_outg.get_merged_cpo_dfs(
            cpo_df_name                         = self.cpx_dfs_name_outg, 
            cpo_df_subset_by_mjr_mnr_cause_args = None, 
            max_total_counts                    = self.mecpx_build_info_dict['max_total_counts']  , 
            how_max_total_counts                = self.mecpx_build_info_dict['how_max_total_counts']
        )
        assert(merged_df_outg.index.names==self.mecpx_build_info_dict['grp_by_cols_outg'])
        #-----
        merged_df_otbl = self.mecpx_coll_otbl.get_merged_cpo_dfs(
            cpo_df_name                         = self.cpx_dfs_name_otbl, 
            cpo_df_subset_by_mjr_mnr_cause_args = None, 
            max_total_counts                    = self.mecpx_build_info_dict['max_total_counts']  , 
            how_max_total_counts                = self.mecpx_build_info_dict['how_max_total_counts']
        )
        assert(merged_df_otbl.index.names==self.mecpx_build_info_dict['grp_by_cols_otbl'])
    
        #--------------------------------------------------
        # Make sure all SNs columns are removed
        merged_df_outg = MECPODf.remove_SNs_cols_from_rcpo_df(merged_df_outg)
        merged_df_otbl = MECPODf.remove_SNs_cols_from_rcpo_df(merged_df_otbl)
        
        #--------------------------------------------------
        # Sort columns and make sure all columns equal for all
        merged_df_outg = merged_df_outg[merged_df_outg.columns.sort_values()]
        merged_df_otbl = merged_df_otbl[merged_df_otbl.columns.sort_values()]
        #-----
        assert(all(merged_df_otbl.columns==merged_df_outg.columns))
        
        #--------------------------------------------------
        # Build counts series
        counts_series_outg = self.mecpx_coll_outg.get_counts_series(self.cpx_dfs_name_outg, False) 
        assert(counts_series_outg.index.names==self.mecpx_build_info_dict['grp_by_cols_outg'])
        #-----
        counts_series_otbl = self.mecpx_coll_otbl.get_counts_series(self.cpx_dfs_name_otbl, False)
        assert(counts_series_otbl.index.names==self.mecpx_build_info_dict['grp_by_cols_otbl'])
        #--------------------------------------------------
        # Standardize index names
        merged_df_outg     = OutageModeler.standardize_index_names(to_adjust=merged_df_outg,     std_dict=self.mecpx_build_info_dict['std_dict_grp_by_cols_outg'])
        counts_series_outg = OutageModeler.standardize_index_names(to_adjust=counts_series_outg, std_dict=self.mecpx_build_info_dict['std_dict_grp_by_cols_outg'])
        #-----
        merged_df_otbl     = OutageModeler.standardize_index_names(to_adjust=merged_df_otbl,     std_dict=self.mecpx_build_info_dict['std_dict_grp_by_cols_otbl'])
        counts_series_otbl = OutageModeler.standardize_index_names(to_adjust=counts_series_otbl, std_dict=self.mecpx_build_info_dict['std_dict_grp_by_cols_otbl'])
        #-------------------------
        idx_names_common = merged_df_outg.index.names
        assert(counts_series_outg.index.names == idx_names_common)
        merged_df_otbl     = merged_df_otbl.reorder_levels(order=idx_names_common, axis=0)
        counts_series_otbl = counts_series_otbl.reorder_levels(order=idx_names_common)
        #-------------------------
        if self.include_prbl:
            merged_df_prbl = self.mecpx_coll_prbl.get_merged_cpo_dfs(
                cpo_df_name                         = self.cpx_dfs_name_prbl, 
                cpo_df_subset_by_mjr_mnr_cause_args = None, 
                max_total_counts                    = self.mecpx_build_info_dict['max_total_counts']  , 
                how_max_total_counts                = self.mecpx_build_info_dict['how_max_total_counts']
            )
            assert(merged_df_prbl.index.names==self.mecpx_build_info_dict['grp_by_cols_prbl'])
            #-------------------------
            merged_df_prbl = MECPODf.remove_SNs_cols_from_rcpo_df(merged_df_prbl)
            merged_df_prbl = merged_df_prbl[merged_df_prbl.columns.sort_values()]
            assert(all(merged_df_prbl.columns==merged_df_outg.columns))
            #-------------------------
            # Build counts series
            counts_series_prbl = self.mecpx_coll_prbl.get_counts_series(self.cpx_dfs_name_prbl, False)
            assert(counts_series_prbl.index.names==self.mecpx_build_info_dict['grp_by_cols_prbl'])
            #-------------------------
            # Standardize index names
            merged_df_prbl     = OutageModeler.standardize_index_names(to_adjust=merged_df_prbl,     std_dict=self.mecpx_build_info_dict['std_dict_grp_by_cols_prbl'])
            counts_series_prbl = OutageModeler.standardize_index_names(to_adjust=counts_series_prbl, std_dict=self.mecpx_build_info_dict['std_dict_grp_by_cols_prbl'])
            #-----
            merged_df_prbl     = merged_df_prbl.reorder_levels(order=idx_names_common, axis=0)
            counts_series_prbl = counts_series_prbl.reorder_levels(order=idx_names_common)
        else:
            merged_df_prbl     = None
            counts_series_prbl = None
        #----------------------------------------------------------------------------------------------------
        assert(
            merged_df_outg.index.names == counts_series_outg.index.names == 
            merged_df_otbl.index.names == counts_series_otbl.index.names
        )
        if self.include_prbl:
            assert(
                merged_df_outg.index.names == counts_series_outg.index.names == 
                merged_df_prbl.index.names == counts_series_prbl.index.names
            )        
        #----------------------------------------------------------------------------------------------------
        self.merged_df_outg = merged_df_outg
        self.merged_df_otbl = merged_df_otbl
        self.merged_df_prbl = merged_df_prbl
        #-----
        self.counts_series_outg = counts_series_outg
        self.counts_series_otbl = counts_series_otbl
        self.counts_series_prbl = counts_series_prbl
        #----------------------------------------------------------------------------------------------------
        self.apply_gnrl_slicer()
        #----------------------------------------------------------------------------------------------------
        if self.__save_data:
            self.save_merged_dfs(tag='0')
            self.save_counts_series(tag='0')
            
    def compile_or_load_merged_dfs_0(
        self
    ):
        r"""
        Load merged_dfs and counts_series if they all exist.
        Otherwise, compile.
        """
        
        #--------------------------------------------------
        compile_dfs_0 = False
        if self.__force_build:
            compile_dfs_0 = True
        #--------------------------------------------------
        # If compile_dfs_0 is False, try to simply load the needed DFs
        #     If this fails, the DFs will be compiled
        # If compile_dfs_0 is True, the DFs will be compiled (duh)
        if not compile_dfs_0:
            try:
                self.load_merged_dfs(
                    tag='0', 
                    verbose=True
                )
                #-----
                self.load_counts_series_from_pkls(
                    tag='0', 
                    verbose=True
                )
                #-------------------------
                return
            except:
                pass
        #--------------------------------------------------
        
        # The MECPOCOllection objects MUST exist
        assert(
            isinstance(self.mecpx_coll_outg, MECPOCollection) and 
            isinstance(self.mecpx_coll_otbl, MECPOCollection)
        )
        if self.include_prbl:
            assert(isinstance(self.mecpx_coll_prbl, MECPOCollection))
        #-------------------------
        self.compile_merged_dfs_0()
        
    def reduce_merged_dfs_to_top_reasons(
        self
    ):
        r"""
        NOTE: Cannot do get_top_reasons_subset_from_merged_cpo_df for each, as they will then in general have unequal columns!
        """
        #--------------------------------------------------
        if self.n_top_reasons_to_inclue is None:
            return
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and self.counts_series_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 and self.counts_series_otbl.shape[0]>0
        )
        #--------------------------------------------------
        merged_df_outg     = self.merged_df_outg
        counts_series_outg = self.counts_series_outg
        #-------------------------
        other_dfs_w_counts_series = [[self.merged_df_otbl, self.counts_series_otbl]]
        if self.include_prbl:
            other_dfs_w_counts_series.append([self.merged_df_prbl, self.counts_series_prbl])
        #-------------------------
        # NOTE: Cannot do get_top_reasons_subset_from_merged_cpo_df for each, as they will then in general have unequal columns!
        merged_df_outg, other_dfs = MECPOCollection.get_top_reasons_subset_from_merged_cpo_df_and_project_from_others(
            merged_cpo_df             = merged_df_outg,
            other_dfs_w_counts_series = other_dfs_w_counts_series, 
            how                       = 'per_mecpo_an', 
            n_reasons_to_include      = self.n_top_reasons_to_inclue,
            combine_others            = self.combine_others,
            output_combine_others_col = 'Other Reasons',
            SNs_tags                  = None, 
            is_norm                   = self.is_norm, 
            counts_series             = counts_series_outg
        )
        #-------------------------
        self.merged_df_outg = merged_df_outg
        self.merged_df_otbl = other_dfs[0]
        if self.include_prbl:
            assert(len(other_dfs)==2)
            self.merged_df_prbl = other_dfs[1]
            
            
    def add_total_counts_features(
        self
    ):
        r"""
        """
        #-------------------------
        if not self.include_total_counts_features:
            return
        #-------------------------
        self.merged_df_outg=MECPOCollection.get_total_event_counts_for_merged_cpo_df(
            merged_df  = self.merged_df_outg, 
            output_col = 'total_counts', 
            SNs_tags   = None
        )
        self.merged_df_otbl=MECPOCollection.get_total_event_counts_for_merged_cpo_df(
            merged_df  = self.merged_df_otbl, 
            output_col = 'total_counts', 
            SNs_tags   = None
        )
        if self.include_prbl:
            self.merged_df_prbl=MECPOCollection.get_total_event_counts_for_merged_cpo_df(
                merged_df  = self.merged_df_prbl, 
                output_col = 'total_counts', 
                SNs_tags   = None
            )
            
    def append_nSNs_per_group(
        self
    ):
        r"""
        """
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and self.counts_series_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 and self.counts_series_otbl.shape[0]>0
        )
        #--------------------------------------------------
        assert(len(set(self.merged_df_outg.index).difference(set(self.counts_series_outg.index)))==0)
        self.merged_df_outg = pd.merge(
            self.merged_df_outg, 
            self.counts_series_outg.to_frame(name=('nSNs', 'nSNs')), 
            left_index=True, right_index=True, how='inner')
        #--------------------------------------------------
        assert(len(set(self.merged_df_otbl.index).difference(set(self.counts_series_otbl.index)))==0)
        self.merged_df_otbl = pd.merge(
            self.merged_df_otbl, 
            self.counts_series_otbl.to_frame(name=('nSNs', 'nSNs')), 
            left_index=True, right_index=True, how='inner')
        #--------------------------------------------------
        if self.include_prbl:
            assert(len(set(self.merged_df_prbl.index).difference(set(self.counts_series_prbl.index)))==0)
            self.merged_df_prbl = pd.merge(
                self.merged_df_prbl, 
                self.counts_series_prbl.to_frame(name=('nSNs', 'nSNs')), 
                left_index=True, right_index=True, how='inner')
        
        
    def combine_time_infos_df(
        self, 
        ti_df_outg, 
        ti_df_otbl, 
        ti_df_prbl, 
        min_req_cols = ['t_min', 't_max']
    ):
        r"""
        Final indices and order will always be determined by outage data, not baseline
        """
        #----------------------------Z----------------------
        assert(
            ti_df_outg is not None and ti_df_outg.shape[0]>0 and 
            ti_df_otbl is not None and ti_df_otbl.shape[0]>0 
        )
        #--------------------------------------------------
        # This is a more general sanity check, and should probably be found somewhere more basic in the class....
        assert(tuple(self.mecpx_build_info_dict['grp_by_cols_outg']) == self.mecpx_coll_outg.grp_by)
        assert(tuple(self.mecpx_build_info_dict['grp_by_cols_otbl']) == self.mecpx_coll_otbl.grp_by)
        #-----
        # Make sure the final index names of the time_info_dfs (ti_dfs) agree
        assert(set(self.mecpx_build_info_dict['std_dict_grp_by_cols_outg'].values())==set(self.mecpx_build_info_dict['std_dict_grp_by_cols_otbl'].values()))
        #--------------------------------------------------
        # Adjust index names and orders....
        ti_df_outg = Utilities_df.change_idx_names_andor_order(
            df               = ti_df_outg, 
            rename_idxs_dict = self.mecpx_build_info_dict['std_dict_grp_by_cols_outg'], 
            final_idx_order  = None, 
            inplace          = True
        )
        final_idx_order = list(ti_df_outg.index.names)
        #-------------------------
        ti_df_otbl = Utilities_df.change_idx_names_andor_order(
            df               = ti_df_otbl, 
            rename_idxs_dict = self.mecpx_build_info_dict['std_dict_grp_by_cols_otbl'], 
            final_idx_order  = final_idx_order, 
            inplace          = True
        )
        #--------------------------------------------------
        ti_dfs = [ti_df_outg, ti_df_otbl]
        #--------------------------------------------------
        if self.include_prbl:
            assert(tuple(self.mecpx_build_info_dict['grp_by_cols_prbl']) == self.mecpx_coll_prbl.grp_by)
            #-------------------------
            assert(set(self.mecpx_build_info_dict['std_dict_grp_by_cols_prbl'].values())==set(self.mecpx_build_info_dict['std_dict_grp_by_cols_outg'].values()))
            #-------------------------
            ti_df_prbl = Utilities_df.change_idx_names_andor_order(
                df               = ti_df_prbl, 
                rename_idxs_dict = self.mecpx_build_info_dict['std_dict_grp_by_cols_prbl'], 
                final_idx_order  = final_idx_order, 
                inplace          = True
            )
            #-------------------------
            ti_dfs.append(ti_df_prbl)
        #--------------------------------------------------
        return_cols = set(ti_dfs[0].columns.tolist())
        for ti_df_i in ti_dfs:
            return_cols = return_cols.intersection(set(ti_df_i.columns.tolist()))
        return_cols = list(return_cols)
        #-------------------------
        assert(set(min_req_cols).difference(set(return_cols))==set())
        #-------------------------
        # Sanity
        return_idx_names = ti_dfs[0].index.names
        for ti_df_i in ti_dfs:
            assert(ti_df_i.index.names==return_idx_names)
        #-------------------------
        return_df = pd.concat([ti_df_i[return_cols] for ti_df_i in ti_dfs], axis=0)
        return_df = Utilities_df.move_cols_to_front(df=return_df, cols_to_move=['t_min', 't_max'])
        return return_df
        
        
    def check_time_infos_df(
        self, 
        time_infos_df, 
        assert_pass=True
    ):
        r"""
        Make sure time_infos_df has all the needed entries
        """
        #-------------------------
        needed_idxs = self.merged_df_outg.index.tolist()+self.merged_df_otbl.index.tolist()
        if self.include_prbl:
            needed_idxs.extend(self.merged_df_prbl.index.tolist())
        #-------------------------
        if set(needed_idxs).difference(set(time_infos_df.index))==set():
            tpass = True
        else:
            tpass = False
        #-------------------------
        if assert_pass:
            assert(tpass)
        #-------------------------
        return tpass
        
    def save_time_infos_df(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_time_infos_df:
            return
        #-------------------------
        assert(self.time_infos_df is not None and self.time_infos_df.shape[0]>0)
        assert(self.is_save_base_dir_loaded())
        self.time_infos_df.to_pickle(os.path.join(self.save_base_dir, 'time_infos_df.pkl'))
        #-------------------------
        self.__saved_time_infos_df = True
    
    def load_time_infos_df(
        self
    ):
        r"""
        Load the full (i.e., containing outg and baseline(s) data) time_infos_df from os.path.join(self.save_base_dir, 'time_infos_df.pkl')
        """
        #-------------------------
        assert(self.is_save_base_dir_loaded())
        assert(os.path.exists(os.path.join(self.save_base_dir, 'time_infos_df.pkl')))
        #-------------------------
        time_infos_df = pd.read_pickle(os.path.join(self.save_base_dir, 'time_infos_df.pkl'))
        self.check_time_infos_df(
            time_infos_df = time_infos_df, 
            assert_pass   = True
        )
        self.time_infos_df = time_infos_df
        # Since loaded, there will be no reason to try to save them again.
        self.__saved_time_infos_df = True
        
        
    def build_time_infos_df(
        self, 
        build_all    = False, 
        min_req_cols = ['t_min', 't_max']
    ):
        r"""
        In most situations, I suggest using build_or_load_time_infos_df instead!
        Final indices and order will always be determined by outage data, not baseline
        """
        #----------------------------------------------------------------------------------------------------
        # Since building new, set __saved... = False
        self.__saved_time_infos_df = False
        #--------------------------------------------------
        assert(
            self.mecpx_coll_outg is not None and 
            self.mecpx_coll_otbl is not None
        )
        #--------------------------------------------------
        self.mecpx_coll_outg.build_time_infos_df(save_to_pkl=self.__save_data, build_all=build_all)
        self.mecpx_coll_otbl.build_time_infos_df(save_to_pkl=self.__save_data, build_all=build_all)
        #-----
        ti_df_outg = self.mecpx_coll_outg.time_infos_df
        ti_df_otbl = self.mecpx_coll_otbl.time_infos_df
        #--------------------------------------------------
        if self.include_prbl:
            self.mecpx_coll_prbl.build_time_infos_df(save_to_pkl=self.__save_data, build_all=build_all)
            ti_df_prbl = self.mecpx_coll_prbl.time_infos_df
        else:
            ti_df_prbl = None
        #--------------------------------------------------
        time_infos_df = self.combine_time_infos_df(
            ti_df_outg   = ti_df_outg, 
            ti_df_otbl   = ti_df_otbl, 
            ti_df_prbl   = ti_df_prbl, 
            min_req_cols = min_req_cols
        )
        self.check_time_infos_df(
            time_infos_df = time_infos_df, 
            assert_pass   = True
        )
        self.time_infos_df = time_infos_df  
        #--------------------------------------------------
        if self.__save_data:
            self.save_time_infos_df()
    
    
    def build_or_load_time_infos_df(
        self, 
        min_req_cols = ['t_min', 't_max'], 
        verbose=True
    ):
        r"""
        Final indices and order will always be determined by outage data, not baseline
        """
        #--------------------------------------------------
        if self.__force_build:
            self.build_time_infos_df(min_req_cols=min_req_cols)
            return 
        #--------------------------------------------------
        # First, check if full (i.e., containing outg and baseline(s) data) time_infos_df is found at os.path.join(self.save_base_dir, 'time_infos_df.pkl')
        if self.is_save_base_dir_loaded() and os.path.exists(os.path.join(self.save_base_dir, 'time_infos_df.pkl')):
            self.load_time_infos_df()
            return
        #--------------------------------------------------
        assert(
            self.mecpx_coll_outg is not None and 
            self.mecpx_coll_otbl is not None
        )
        #--------------------------------------------------
        self.mecpx_coll_outg.build_or_load_time_infos_df(save_to_pkl=self.__save_data, verbose=verbose)
        self.mecpx_coll_otbl.build_or_load_time_infos_df(save_to_pkl=self.__save_data, verbose=verbose)
        #-----
        ti_df_outg = self.mecpx_coll_outg.time_infos_df
        ti_df_otbl = self.mecpx_coll_otbl.time_infos_df
        #--------------------------------------------------
        if self.include_prbl:
            self.mecpx_coll_prbl.build_or_load_time_infos_df(save_to_pkl=self.__save_data, verbose=verbose)
            ti_df_prbl = self.mecpx_coll_prbl.time_infos_df
        else:
            ti_df_prbl = None
        #--------------------------------------------------
        time_infos_df = self.combine_time_infos_df(
            ti_df_outg   = ti_df_outg, 
            ti_df_otbl   = ti_df_otbl, 
            ti_df_prbl   = ti_df_prbl, 
            min_req_cols = min_req_cols
        )
        self.check_time_infos_df(
            time_infos_df = time_infos_df, 
            assert_pass   = True
        )
        self.time_infos_df = time_infos_df 
        #--------------------------------------------------
        if self.__save_data:
            self.save_time_infos_df()
    
    
    def load_merged_dfs(
        self, 
        tag='0', 
        verbose=True
    ):
        r"""
        tag:
            If None,   save_name = 'merged_df_xxxx'
            Otherwise, save_name = f'merged_df_xxxx_{tag}'
        """
        #-------------------------
        assert(os.path.isdir(self.save_base_dir))
        #-------------------------
        if tag is None:
            appndx = ''
        else:
            appndx = f'_{tag}'        
        #-----
        assert(os.path.exists(os.path.join(self.save_base_dir, f'merged_df_outg{appndx}.pkl')))
        assert(os.path.exists(os.path.join(self.save_base_dir, f'merged_df_otbl{appndx}.pkl')))
        #-----
        self.merged_df_outg=pd.read_pickle(os.path.join(self.save_base_dir, f'merged_df_outg{appndx}.pkl'))
        self.merged_df_otbl=pd.read_pickle(os.path.join(self.save_base_dir, f'merged_df_otbl{appndx}.pkl'))
        #-----
        loaded = [
            f"merged_df_outg{appndx}: {os.path.join(self.save_base_dir, f'merged_df_outg{appndx}.pkl')}", 
            f"merged_df_otbl{appndx}: {os.path.join(self.save_base_dir, f'merged_df_otbl{appndx}.pkl')}"
        ]
        #-------------------------
        if self.include_prbl:
            assert(os.path.exists(os.path.join(self.save_base_dir, f'merged_df_prbl{appndx}.pkl')))
            self.merged_df_prbl=pd.read_pickle(os.path.join(self.save_base_dir, f'merged_df_prbl{appndx}.pkl'))
            loaded.append(f"merged_df_prbl{appndx}: {os.path.join(self.save_base_dir, f'merged_df_prbl{appndx}.pkl')}")
        #-------------------------
        if verbose:
            print('Successfully loaded merged_dfs:')
            print(*loaded, sep='\n')
        #----------------------------------------------------------------------------------------------------
        # Since self.merged_df_... were loaded from file, there will be no reason to try to save them again.
        if tag is None:
            self.__saved_merged_dfs   = True
        if (tag=='0' or tag==0):
            self.__saved_merged_dfs_0 = True
    
    
    def load_counts_series_from_pkls(
        self, 
        tag='0', 
        verbose=True
    ):
        r"""
        """
        #-------------------------
        assert(os.path.isdir(self.save_base_dir))
        #-------------------------
        if tag is None:
            appndx = ''
        else:
            appndx = f'_{tag}'
        #-----
        assert(os.path.exists(os.path.join(self.save_base_dir, f'counts_series_outg{appndx}.pkl')))
        assert(os.path.exists(os.path.join(self.save_base_dir, f'counts_series_otbl{appndx}.pkl')))
        #-----
        with open(os.path.join(self.save_base_dir, f'counts_series_outg{appndx}.pkl'), 'rb') as handle:
            self.counts_series_outg = pickle.load(handle)
        with open(os.path.join(self.save_base_dir, f'counts_series_otbl{appndx}.pkl'), 'rb') as handle:
            self.counts_series_otbl = pickle.load(handle)
        #-----
        loaded = [
            f"counts_series_outg{appndx}: {os.path.join(self.save_base_dir, f'counts_series_outg{appndx}.pkl')}", 
            f"counts_series_otbl{appndx}: {os.path.join(self.save_base_dir, f'counts_series_otbl{appndx}.pkl')}"
        ]
        #-------------------------
        if os.path.exists(os.path.join(self.save_base_dir, f'counts_series_prbl{appndx}.pkl')):
            with open(os.path.join(self.save_base_dir, f'counts_series_prbl{appndx}.pkl'), 'rb') as handle:
                self.counts_series_prbl = pickle.load(handle)
            loaded.append(f"counts_series_prbl{appndx}: {os.path.join(self.save_base_dir, f'counts_series_prbl{appndx}.pkl')}")
        #-------------------------
        if verbose:
            print('Successfully loaded counts_series:')
            print(*loaded, sep='\n')
        #----------------------------------------------------------------------------------------------------
        # Since self.counts_series_... were loaded from file, there will be no reason to try to save them again.
        if tag is None:
            self.__saved_counts_series   = True
        if (tag=='0' or tag==0):
            self.__saved_counts_series_0 = True
            
            
    def load_all_mecpx(
        self, 
        tag='0', 
        verbose=True
    ):
        r"""
        Load all MECPOCollection objects and all associated objects (i.e., merged_dfs, counts_series, etc.)
        """
        #-------------------------
        self.load_mecpx_colls_from_pkls(verbose=verbose)
        self.load_merged_dfs(verbose=verbose, tag=tag)
        self.load_counts_series_from_pkls(verbose=verbose, tag=tag)
        self.load_mecpx_build_info_dict(verbose=verbose)

            
    def get_trsf_pole_nbs(
        self, 
        trsf_pole_idx_lvl = 'trsf_pole_nb'
    ):
        r"""
        """
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0
        )
        #-----
        assert(
            trsf_pole_idx_lvl in self.merged_df_outg.index.names and 
            trsf_pole_idx_lvl in self.merged_df_otbl.index.names
        )
        #--------------------------------------------------
        trsf_pole_nbs = list(set(
            self.merged_df_outg.index.get_level_values(trsf_pole_idx_lvl).unique().tolist()+
            self.merged_df_otbl.index.get_level_values(trsf_pole_idx_lvl).unique().tolist()
        ))
        if self.include_prbl:
            assert(trsf_pole_idx_lvl in self.merged_df_prbl.index.names)
            trsf_pole_nbs = list(set(trsf_pole_nbs + self.merged_df_prbl.index.get_level_values(trsf_pole_idx_lvl).unique().tolist()))
        #--------------------------------------------------
        return trsf_pole_nbs
        
        
    def save_eemsp(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_eemsp or not self.merge_eemsp:
            return
        #-------------------------
        assert(self.is_save_base_dir_loaded())
        assert(self.eemsp_df is not None and self.eemsp_df.shape[0]>0)
        assert(self.eemsp_df is not None and self.eemsp_df.shape[0]>0)
        assert(self.eemsp_df_info_dict is not None)
        #-------------------------
        self.eemsp_df.to_pickle(os.path.join(self.save_base_dir, f'df_eemsp_reduce2_{self.eemsp_mult_strategy}.pkl'))
        #-----
        CustomWriter.output_dict_to_json(
            os.path.join(self.save_base_dir, 'eemsp_df_info_dict.json'), 
            self.eemsp_df_info_dict
        )
        #-------------------------
        if self.eemsp_reduce1_df is not None and self.eemsp_reduce1_df.shape[0]>0:
            self.eemsp_reduce1_df.to_pickle(os.path.join(self.save_base_dir, 'df_eemsp_reduce1.pkl'))
        #-------------------------
        self.__saved_eemsp = True
        
    
    def build_eemsp_df_for_modeler(
        self, 
        trsf_pole_idx_lvl = 'trsf_pole_nb'
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        # Since building new, set __saved... = False
        self.__saved_eemsp = False
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 and 
            self.time_infos_df.shape[0]>0 
        )
        trsf_pole_nbs = self.get_trsf_pole_nbs(
            trsf_pole_idx_lvl = trsf_pole_idx_lvl
        )
        #--------------------------------------------------
        if self.__save_data:
            assert(self.is_save_base_dir_loaded())
        #--------------------------------------------------
        eemsp_dict = EEMSP.build_eemsp_df_for_xfmrs_w_time_info(
            trsf_pole_nbs               = trsf_pole_nbs, 
            time_infos_df               = self.time_infos_df, 
            mult_strategy               = self.eemsp_mult_strategy, 
            batch_size                  = 1000, 
            verbose                     = True, 
            n_update                    = 10, 
            addtnl_kwargs               = None, 
            time_infos_df_info_dict     = None, 
            eemsp_df_info_dict          = None, 
            min_pct_found_in_time_infos = 0.75, 
            save_dfs                    = self.__save_data, 
            save_dir                    = self.save_base_dir, 
            return_3_dfs                = True,  
            return_eemsp_df_info_dict   = True
        )
        #--------------------------------------------------
        self.eemsp_df           = eemsp_dict['df_eemsp_reduce2']
        self.eemsp_reduce1_df   = eemsp_dict['df_eemsp_reduce1']
        self.eemsp_df_info_dict = eemsp_dict['eemsp_df_info_dict']
        #--------------------------------------------------
        if self.__save_data:
            self.__saved_eemsp = True
        
    def build_or_load_eemsp_df(
        self, 
        trsf_pole_idx_lvl = 'trsf_pole_nb', 
        verbose           = True
    ):
        r"""
        """
        #--------------------------------------------------
        if self.__force_build:
            self.build_eemsp_df_for_modeler(trsf_pole_idx_lvl=trsf_pole_idx_lvl)
            return
        #--------------------------------------------------
        # os.path.exists doesn't like being fed None, hence the need for the following if/else statements
        if self.save_base_dir is not None:
            path_df         = os.path.join(self.save_base_dir, f'df_eemsp_reduce2_{self.eemsp_mult_strategy}.pkl')
            path_df_reduce1 = os.path.join(self.save_base_dir, 'df_eemsp_reduce1.pkl')
            path_info_dict  = os.path.join(self.save_base_dir, 'eemsp_df_info_dict.json')

        else:
            path_df         = ''
            path_df_reduce1 = ''
            path_info_dict  = ''
        #--------------------------------------------------
        if os.path.exists(path_df) and os.path.exists(path_info_dict):
            if verbose:
                print(f'Reading eemsp_df from {path_df}')
            self.eemsp_df = pd.read_pickle(path_df)
            #-----
            tmp_f = open(path_info_dict)
            self.eemsp_df_info_dict = json.load(tmp_f)
            tmp_f.close()
            #-----
            if os.path.exists(path_df_reduce1):
                self.eemsp_reduce1_df = pd.read_pickle(path_df_reduce1)
            else:
                print(f'Warning: self.eemsp_df loaded from {path_df},\n but no eemsp_reduce1_df file found at {path_df_reduce1}')
                print('\tNot a huge deal, but e.g., some EEMSP plotting method may be unavailable without it')
            #-------------------------
            # Since self.eemsp_df and self.eemsp_df_info_dict were loaded from file, there will be no reason to try to save them again.
            # ==> set self.__saved_eemsp = True
            self.__saved_eemsp = True
            #-------------------------
            return
        #--------------------------------------------------
        # No pickle file exists at expected path, so eemsp_df must be built
        if verbose:
            print(f'No file found at \n\t{path_df} and/or \n\t{path_info_dict}, \n==> building eemsp_df\n')
        self.build_eemsp_df_for_modeler(
            trsf_pole_idx_lvl = trsf_pole_idx_lvl
        )    
        
    def add_eemsp_features( 
        self, 
        verbose               = True
    ):
        r"""
        """
        #-------------------------------------------------- 
        if not self.merge_eemsp:
            return
        #--------------------------------------------------
        # build_eemsp_df (used to be an input to this function):
        #     If True, eemsp_df will be built.
        #     If False, the program will:
        #             1: Check if self.eemsp_df exists
        #             2: Check if a pickle file exists at:
        #                 os.path.join(self.save_base_dir, f'df_eemsp_reduce2_{self.eemsp_mult_strategy}.pkl')
        #         If 1 is True (regardless of 2) ==> use self.eemsp_df to add features
        #         If 2 is True (and 1 is False)  ==> set self.eemsp_df equal to file contents and use to add features
        #         If 1 and 2 both False          ==> self.eemsp_df will be built
        build_eemsp_df = False
        if self.__force_build:
            build_eemsp_df = True
        
        #--------------------------------------------------
        assert(self.mecpx_build_info_dict['rec_nb_idfr'][0]=='index')
        rec_idx_lvl = self.mecpx_build_info_dict['rec_nb_idfr'][1]
        #-----
        assert(self.mecpx_build_info_dict['trsf_pole_nb_idfr'][0]=='index')
        trsf_pole_idx_lvl = self.mecpx_build_info_dict['trsf_pole_nb_idfr'][1]
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 
        )
        assert(set([rec_idx_lvl, trsf_pole_idx_lvl]).difference(set(self.merged_df_outg.index.names))==set())
        assert(set([rec_idx_lvl, trsf_pole_idx_lvl]).difference(set(self.merged_df_otbl.index.names))==set())
        #-----
        if self.include_prbl:
            assert(set([rec_idx_lvl, trsf_pole_idx_lvl]).difference(set(self.merged_df_prbl.index.names))==set())
        #--------------------------------------------------
        # Build, load, or simply pass through (if self.eemsp_df exists and build_eemsp_df=False)
        if build_eemsp_df:
            self.build_eemsp_df_for_modeler(
                trsf_pole_idx_lvl = trsf_pole_idx_lvl, 
            )
        else:
            # If self.eemsp_df exists, use it.
            # ==> If self.eemsp_df does not exist, build it
            if self.eemsp_df is None or self.eemsp_df.shape[0]==0:
                self.build_or_load_eemsp_df(
                    trsf_pole_idx_lvl = trsf_pole_idx_lvl, 
                    verbose           = verbose
                )
        #--------------------------------------------------
        eemsp_rec_nb_col      = self.eemsp_df_info_dict['rec_nb_to_merge_col']
        eemsp_location_nb_col = self.eemsp_df_info_dict['eemsp_location_nb_col']
        assert(set([eemsp_rec_nb_col, eemsp_location_nb_col]).difference(set(self.eemsp_df.columns))==set())
        n_trsf_pole_nbs = len(self.get_trsf_pole_nbs())
        #--------------------------------------------------
        if verbose:
            print(f"self.eemsp_df['{eemsp_location_nb_col}'].nunique() = {self.eemsp_df[eemsp_location_nb_col].nunique()}")
            print(f"n_trsf_pole_nbs                        = {n_trsf_pole_nbs}")
            print(f"Diff                                   = {n_trsf_pole_nbs-self.eemsp_df[eemsp_location_nb_col].nunique()}")
            print()
            #-------------------------
            print("\nShapes BEFORE merging")
            print(f"self.merged_df_outg.shape = {self.merged_df_outg.shape}")
            print(f"self.merged_df_otbl.shape = {self.merged_df_otbl.shape}")
            if self.include_prbl:
                print(f"self.merged_df_prbl.shape = {self.merged_df_prbl.shape}")
        #--------------------------------------------------
        self.merged_df_outg = EEMSP.merge_rcpx_with_eemsp(
            df_rcpx       = self.merged_df_outg, 
            df_eemsp      = self.eemsp_df, 
            merge_on_rcpx = [('index', rec_idx_lvl), ('index', trsf_pole_idx_lvl)], 
            merge_on_eems = [eemsp_rec_nb_col, eemsp_location_nb_col], 
            set_index     = True
        )
        #-------------------------
        self.merged_df_otbl = EEMSP.merge_rcpx_with_eemsp(
            df_rcpx       = self.merged_df_otbl, 
            df_eemsp      = self.eemsp_df, 
            merge_on_rcpx = [('index', rec_idx_lvl), ('index', trsf_pole_idx_lvl)],
            merge_on_eems = [eemsp_rec_nb_col, eemsp_location_nb_col], 
            set_index     = True
        )
        #-------------------------
        if self.include_prbl:
            self.merged_df_prbl = EEMSP.merge_rcpx_with_eemsp(
                df_rcpx       = self.merged_df_prbl, 
                df_eemsp      = self.eemsp_df, 
                merge_on_rcpx = [('index', rec_idx_lvl), ('index', trsf_pole_idx_lvl)],
                merge_on_eems = [eemsp_rec_nb_col, eemsp_location_nb_col], 
                set_index     = True
            )
        #--------------------------------------------------
        if verbose:
            print("\nShapes AFTER merging")
            print(f"self.merged_df_outg.shape = {self.merged_df_outg.shape}")
            print(f"self.merged_df_otbl.shape = {self.merged_df_otbl.shape}")
            if self.include_prbl:
                print(f"self.merged_df_prbl.shape = {self.merged_df_prbl.shape}")
            
            
    @staticmethod
    def merge_cpx_df_w_time_infos(
        cpx_df, 
        time_infos_df, 
        time_infos_drop_dupls_subset=['index', 't_min'], 
        dummy_lvl_base_name = 'dummy_lvl'
    ):
        r"""
        cpx_df and time_infos_df must have same indices.
        Typically, these are no_outg_rec_nb and trsf_pole_nb.
        
        time_infos_drop_dupls_subset:
            Since data are collected over multiple files, sometimes a (no-)outage event is split over multiple files
              with, e.g., different PNs.  So, duplicates must be dropped.
            Typically, index needs to be included as well (index usually comprised of no_outg_rec_nb and trsf_pole_nb)
        """
        #-------------------------
        if time_infos_drop_dupls_subset is not None:
            assert(Utilities.is_object_one_of_types(time_infos_drop_dupls_subset, [list, tuple]))
            if 'index' in time_infos_drop_dupls_subset:
                time_infos_drop_dupls_subset.remove('index')
                time_infos_df = time_infos_df.reset_index().drop_duplicates(
                    subset=list(time_infos_df.index.names)+time_infos_drop_dupls_subset
                ).set_index(time_infos_df.index.names)
            else:
                time_infos_df = time_infos_df.drop_duplicates(subset=time_infos_drop_dupls_subset)
        #-------------------------
        # In order to merge, cpx_df and time_infos_df must have same number of levels in columns
        if cpx_df.columns.nlevels>1:
            n_levels_to_add = cpx_df.columns.nlevels - time_infos_df.columns.nlevels
            #-----
            for i_new_lvl in range(n_levels_to_add):
                # With each iteration, prepending a new level from n_levels_to_add-1 to 0
                i_level_val = f'{dummy_lvl_base_name}_{(n_levels_to_add-1)-i_new_lvl}'
                time_infos_df = Utilities_df.prepend_level_to_MultiIndex(
                    df=time_infos_df, 
                    level_val=i_level_val, 
                    level_name=None, 
                    axis=1
                )
        assert(cpx_df.columns.nlevels==time_infos_df.columns.nlevels)
        #-------------------------
        # Apparently, pd.merge is smart enough to match index level names, so the following isn't strictly necessary!
        # However, it doesn't hurt, and is good in practice
        assert(len(set(cpx_df.index.names).symmetric_difference(set(time_infos_df.index.names)))==0)
        if time_infos_df.index.names!=cpx_df.index.names:
            time_infos_df = time_infos_df.reset_index().set_index(cpx_df.index.names)
        #-----
        cpx_df_wt = pd.merge(
            cpx_df, 
            time_infos_df, 
            how='left', 
            left_index=True, 
            right_index=True
        )
        #-------------------------
        return cpx_df_wt
            
    @staticmethod
    def add_time_infos_features_to_df(
        cpx_df, 
        time_infos_df, 
        include_eemsp       = False, 
        include_month       = True, 
        t_min_col           = 't_min', 
        trsf_install_dt_col = ('EEMSP_0', 'install_dt'), 
        return_trsf_age_col = None, 
        return_month_col    = None, 
        keep_time_infos     = False
    ):
        r"""
        cpx_df and time_infos_df must have the same index structure!!!!!
            Meaning, they must have the same number of index levels and the same level names
        """
        #----------------------------------------------------------------------------------------------------
        if not(include_eemsp or include_month):
            return cpx_df
        #----------------------------------------------------------------------------------------------------
        og_cols = cpx_df.columns.tolist()
        addtnl_cols_to_keep = []
        #-----
        assert(cpx_df.index.names==time_infos_df.index.names)
        return_df = OutageModeler.merge_cpx_df_w_time_infos(
            cpx_df                       = cpx_df, 
            time_infos_df                = time_infos_df, 
            time_infos_drop_dupls_subset = ['index', t_min_col], 
            dummy_lvl_base_name          = 'time_info'
        )
        
        #----------------------------------------------------------------------------------------------------
        # Since the column names in time_infos_df could possibly overlap with those in df, 
        #   to find correct location of, e.g., t_min_col, I will:
        #     1. Find location(s) of t_min_col in highest order of column levels.
        #     2. If MultiIndex:
        #            2a. Find location(s) of dummy_lvl_base_name in lowest order of column levels.
        #            2b. Find the overlap of (1) and (2a)
        #     3. Ensure only a single column found
        #----------------------------------------------------------------------------------------------------
        # 1. Find location(s) of t_min_col in highest order of column levels.
        #-----
        t_min_idxs = Utilities_df.find_idxs_in_highest_order_of_columns(
            df            = return_df, 
            col           = t_min_col, 
            exact_match   = True, 
            assert_single = False
        )
        
        #----------------------------------------------------------------------------------------------------
        # 2. If MultiIndex
        #-----
        time_info_lvl_0_nm = None
        if return_df.columns.nlevels>1:
            #--------------------------------------------------
            # 2a. Find location(s) of dummy_lvl_base_name in lowest order of column levels.
            #-----
            time_info_cols_idxs = Utilities_df.find_idxs_in_lowest_order_of_columns(
                df            = return_df, 
                regex_pattern = r'time_info_\d*', 
                ignore_case   = False
            )
            # Should be found in the 0th level...
            assert(time_info_cols_idxs[1]==0)
            time_info_cols_idxs = time_info_cols_idxs[0]
            #-------------------------
            time_info_lvl_0_nm = return_df.columns.get_level_values(0)[time_info_cols_idxs].unique().tolist()
            assert(len(time_info_lvl_0_nm)==1)
            time_info_lvl_0_nm = time_info_lvl_0_nm[0]
            #-------------------------
            # 2b. Find the overlap of (1) and (2a)
            #-----
            t_min_idxs = list(set(t_min_idxs).intersection(set(time_info_cols_idxs)))
                
        #----------------------------------------------------------------------------------------------------
        # 3. Ensure only a single column found
        #-----
        assert(len(t_min_idxs)==1)
        
        #----------------------------------------------------------------------------------------------------
        # If including EEMSP, we typically use the age of the transformer instead of the installation date
        #   as the model feature.
        #-----
        if include_eemsp:
            assert(trsf_install_dt_col in return_df.columns.tolist())
            #-----
            if return_trsf_age_col is None:
                return_trsf_age_col    = trsf_install_dt_col
            else:
                return_trsf_age_col = Utilities_df.build_column_for_df(
                    df                  = return_df, 
                    input_col           = return_trsf_age_col, 
                    level_0_val         = None,
                    level_nm1_val       = None, 
                    dummy_lvl_base_name = 'dummy_lvl_xfmr_age', 
                    level_vals_dict     = None
                )
            #-------------------------
            return_df[return_trsf_age_col] = (return_df.iloc[:, t_min_idxs[0]] - return_df[trsf_install_dt_col]).dt.total_seconds()/(60*60*24*365)
            addtnl_cols_to_keep.append(return_trsf_age_col)
        
        #----------------------------------------------------------------------------------------------------
        if include_month:
            return_month_col = Utilities_df.build_column_for_df(
                df                  = return_df, 
                input_col           = return_month_col, 
                level_0_val         = time_info_lvl_0_nm if time_info_lvl_0_nm is not None else 'outg_month',
                level_nm1_val       = 'outg_month', 
                dummy_lvl_base_name = 'dummy_lvl_month', 
                level_vals_dict     = None
            )
            #-----
            return_df[return_month_col] = return_df.iloc[:, t_min_idxs[0]].dt.month
            addtnl_cols_to_keep.append(return_month_col)
        
        #----------------------------------------------------------------------------------------------------
        if not keep_time_infos:
            cols_to_drop = (set(return_df.columns.tolist()).difference(set(og_cols))).difference(addtnl_cols_to_keep)
            return_df = return_df.drop(columns=cols_to_drop)
    
        #----------------------------------------------------------------------------------------------------
        return return_df
        
        
    def add_time_infos_features(
        self
    ):
        r"""
        self.include_month==True:
            Include the month of the event as a feature
    
        self.merge_eemsp==True:
            Convert transformer installation date to transformer age
        """
        #--------------------------------------------------
        if not self.include_month and not self.merge_eemsp:
            return
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 and 
            self.time_infos_df.shape[0]>0 
        )
        #--------------------------------------------------
        self.merged_df_outg = OutageModeler.add_time_infos_features_to_df(
            cpx_df              = self.merged_df_outg, 
            time_infos_df       = self.time_infos_df, 
            include_eemsp       = self.merge_eemsp, 
            include_month       = self.include_month, 
            t_min_col           = 't_min', 
            trsf_install_dt_col = ('EEMSP_0', 'install_dt'), 
            return_trsf_age_col = None, 
            return_month_col    = None, 
            keep_time_infos     = False
        )
        #-------------------------
        self.merged_df_otbl = OutageModeler.add_time_infos_features_to_df(
            cpx_df              = self.merged_df_otbl, 
            time_infos_df       = self.time_infos_df, 
            include_eemsp       = self.merge_eemsp, 
            include_month       = self.include_month, 
            t_min_col           = 't_min', 
            trsf_install_dt_col = ('EEMSP_0', 'install_dt'), 
            return_trsf_age_col = None, 
            return_month_col    = None, 
            keep_time_infos     = False
        )
        #-------------------------
        if self.include_prbl:
            self.merged_df_prbl = OutageModeler.add_time_infos_features_to_df(
                cpx_df              = self.merged_df_prbl, 
                time_infos_df       = self.time_infos_df, 
                include_eemsp       = self.merge_eemsp, 
                include_month       = self.include_month, 
                t_min_col           = 't_min', 
                trsf_install_dt_col = ('EEMSP_0', 'install_dt'), 
                return_trsf_age_col = None, 
                return_month_col    = None, 
                keep_time_infos     = False
            )        
        
        
    @staticmethod
    def encode_eemsp(
        df, 
        eemsp_enc      = None, 
        eemsp_lvl_0_nm = 'EEMSP_0', 
        cols_to_encode = None, 
        numeric_cols   = ['kva_size', 'install_dt'], 
        run_transform  = True, 
        no_xfrm_df0    = False, 
    ):
        r"""
        Encode the EEMSP columns in df using the ordinal encoder OrdinalEncoder.
        If eemsp_enc is not supplied, one is built and the used to transform the data.
        Returns: df, eemsp_enc
        -----
        This function was designed expected df to have MutliIndex columns with 2 levels, where all EEMSP data are contained
        in the columns with level 0 name = eemsp_lvl_0_nm (e.g., a normal EEMSP column would be ('EEMSP_0', 'prim_voltage'))
    
        The function will still work if df.columns.nlevels<2, the user will just need to explicitly supply cols_to_encode and
        eemsp_lvl_0_nm will be ignored.
    
        I'm not sure how I will want to handle the case where df.columns.nlevels>2, so, for now, it is not permitted.
        NOTE: The main change will involve anything having to do with .get_level_values(0), or .droplevel(0, axis=1)
        -----
        df:
            pd.DataFrame object OR a list of pd.DataFrame objects
            IF LIST:
                All elements must be pd.DataFrame objects with the same columns (or empty)
                If eemsp_enc is included, it will be used to transform all.
                If eemsp_enc is not included, it will be fit with df[0] and used to transform all
                Returns: list of transformed dfs and eemsp_enc
    
        eemsp_enc:
            OrdinalEncoder object or None.
            If eemsp_enc is not supplied, one is built and the used to transform the data.
            If eemsp_enc is supplied, it is used to transform the data
    
        eemsp_lvl_0_nm:
            As mentioned above, this was designed for the case where df.columns.nlevels==2 and all EEMSP data are contained
              in the columns with level 0 name = eemsp_lvl_0_nm
            If no cols_to_encode are input, the eemsp_lvl_0_nm must be supplied as it will be used to find the columns to encode.
            If eemsp_lvl_0_nm is supplied by the 0th level value of cols_to_encode which are found do not agree with the supplied value,
              then a warning will be printed.
        
        cols_to_encode:
            Columns to encode.
            If not supplied, all EEMSP columns are used (i.e., df[eemsp_lvl_0_nm].columns.tolist())
        
        numeric_cols:
            These columns are explicitly excluded from cols_to_encode
    
        no_xfrm_df0:
            ONLY HAS EFFECT WHEN df SUPPLIED IS LIST OF DATAFRAMES!
            If True, the first DataFrame, df[0], will not be transformed (all others will be)
            If False, the first DataFrame, df[0], will be transformed with all others.
            NOTE: If eemsp_enc is not supplied, regardless of no_xfrm_df0, df[0] will be used to fit the encoder
        """
        #----------------------------------------------------------------------------------------------------
        assert(Utilities.is_object_one_of_types(df, [pd.DataFrame, list]))
        if isinstance(df, list):
            # Make sure all dfs share the same columns (or, are empty)
            all_cols = natsorted(set(itertools.chain.from_iterable([x.columns.tolist() for x in df])))
            for df_i in df:
                assert(
                    df_i is None or
                    df_i.shape[0]==0 or
                    set(df_i.columns.tolist()).symmetric_difference(set(all_cols))==set()
                )
            #-----
            # no_xfrm_df0 dictates whether or not df_0 will be transformed (all others will be)
            assert(isinstance(no_xfrm_df0, bool))
            run_transform = not no_xfrm_df0
            #-----
            return_dfs = []
            for df_i in df:
                df_i_fnl, eemsp_enc = OutageModeler.encode_eemsp(
                    df             = df_i, 
                    eemsp_enc      = eemsp_enc, 
                    eemsp_lvl_0_nm = eemsp_lvl_0_nm, 
                    cols_to_encode = cols_to_encode, 
                    numeric_cols   = numeric_cols, 
                    run_transform  = run_transform
                )
                return_dfs.append(df_i_fnl)
                #-----
                # run_transform should always be true after the 0th element
                run_transform = True
            #-----
            return return_dfs, eemsp_enc
    
        #----------------------------------------------------------------------------------------------------
        assert(df is None or isinstance(df, pd.DataFrame))
        if df is None or df.shape[0]==0:
            return df, eemsp_enc
        #-----
        assert(df.columns.nlevels<=2)
        #-------------------------
        if cols_to_encode is None:
            assert(eemsp_lvl_0_nm in df.columns.get_level_values(0))
            cols_to_encode = df[eemsp_lvl_0_nm].columns.tolist()
        #-----
        assert(isinstance(cols_to_encode, list))
        # Following code enforces that columns are found (by crashing otherwise) and returns the correct
        #   form of the columns for the case of MultiIndex
        cols_to_encode = [Utilities_df.find_single_col_in_multiindex_cols(df=df, col=x) for x in cols_to_encode]
        #-------------------------
        # numeric_cols contains a list of numeric columns expected to be found in ANY EEMSP dataframe,
        #   not all are necessarily found in this dataframe
        if numeric_cols is None:
            numeric_cols = []
        else:
            numeric_cols_fnl = []
            for col_i in numeric_cols:
                try:
                    col_i_fnl = Utilities_df.find_single_col_in_multiindex_cols(df=df, col=col_i)
                    numeric_cols_fnl.append(col_i_fnl)
                except:
                    contine
            numeric_cols = numeric_cols_fnl
        #-------------------------
        cols_to_encode = [x for x in cols_to_encode if x not in numeric_cols]
        #-------------------------
        df[cols_to_encode] = df[cols_to_encode].astype(str)
        #------------------------- 
        # If eemsp_lvl_0_nm supplied, print message if found cols_to_encode disagree with
        #   level 0 values
        if df.columns.nlevels==2 and eemsp_lvl_0_nm is not None:
            all_lvl_0_agree = all([True if x[0]==eemsp_lvl_0_nm else False for x in cols_to_encode])
            if not all_lvl_0_agree:
                print(f'WARNING!!!!!\nIn OutageModeler.encode_eemsp: The found cols_to_encode do not agree with the supplied eemsp_lvl_0_nm')
                print(f'\teemsp_lvl_0_nm = {eemsp_lvl_0_nm}\n\tcols_to_encode = {cols_to_encode}')
                print(f'\tDisagree: {[x for x in cols_to_encode if x[0]!=eemsp_lvl_0_nm]}')
        #-------------------------
        # NOTE: In order for eemsp_enc to include the feature_names_in_ attribute (which I would like to use
        #         in model deployment to ensure the EEMSP data form matches that expected by eemsp_end), the
        #         feature names must be string.
        #       Hence the need for .droplevel(0, axis=1) below (the use within the transform call is not strictly
        #         needed, but prevents a warning message from being output)
        #-----
        df_to_enc = df[cols_to_encode]
        if df.columns.nlevels==2:
            df_to_enc = df_to_enc.droplevel(0, axis=1)
        #-------------------------
        if eemsp_enc is None:
            eemsp_enc = preprocessing.OrdinalEncoder(
                handle_unknown='use_encoded_value', 
                unknown_value=-1
            )
            #-----
            eemsp_enc.fit(df_to_enc)
        #-------------------------
        if run_transform:
            df[cols_to_encode] = eemsp_enc.transform(df_to_enc)
        #-------------------------
        return df, eemsp_enc
        
        
    @staticmethod
    def build_dovs_df_for_outgs(
        merged_df_outg, 
        outg_rec_nb_idfr          = ('index', 'outg_rec_nb'), 
        contstruct_df_args        = None, 
        build_sql_function        = DOVSOutages_SQL.build_sql_std_outage, 
        build_sql_function_kwargs = None, 
        dummy_col_levels_prefix   = 'outg_dummy_lvl_', 
        remove_unnec_col_lvls     = False
    ):
        r"""
        """
        #-------------------------
        # Get DOVS info to be used for setting target values
        og_cols = merged_df_outg.columns.tolist()
        #-----
        dovs_df = DOVSOutages.append_outg_info_to_df(
            df                        = merged_df_outg.copy(), 
            outg_rec_nb_idfr          = outg_rec_nb_idfr, 
            contstruct_df_args        = contstruct_df_args, 
            build_sql_function        = build_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs, 
            dummy_col_levels_prefix = dummy_col_levels_prefix
        )
        
        #-------------------------
        # Keep only the new DOVS columns...
        dovs_df = dovs_df[list(set(dovs_df.columns.tolist()).difference(set(og_cols)))]
    
        #-------------------------
        if remove_unnec_col_lvls:
            # If merged_df_outg_w_DOVS columns are MultiIndex, check if any of the levels only have one unique value
            # Drop any levels found containing only a single value
            col_lvl_i=0
            while col_lvl_i < dovs_df.columns.nlevels-1:
                if dovs_df.columns.get_level_values(col_lvl_i).nunique()==1:
                    dovs_df = dovs_df.droplevel(level=col_lvl_i, axis=1)
                    continue
                col_lvl_i+=1
    
        #-------------------------
        return dovs_df

    def build_dovs_df(self):
        r"""
        """
        #-------------------------
        # remove_unnec_col_lvls=False below because we want dovs_df to have same number of levels in columns as self.merged_df_outg
        dovs_df = OutageModeler.build_dovs_df_for_outgs(
            merged_df_outg            = self.merged_df_outg, 
            outg_rec_nb_idfr          = self.mecpx_build_info_dict['rec_nb_idfr'], 
            contstruct_df_args        = None, 
            build_sql_function        = DOVSOutages_SQL.build_sql_std_outage, 
            build_sql_function_kwargs = None, 
            dummy_col_levels_prefix   = 'outg_dummy_lvl_', 
            remove_unnec_col_lvls     = False
        )
        self.dovs_df = dovs_df
        
        
    def remove_trsf_pole_nbs_from_merged_dfs(
        self
    ):
        r"""
        """
        #--------------------------------------------------
        if self.trsf_pole_nbs_to_remove is None or len(self.trsf_pole_nbs_to_remove)==0:
            return
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 
        )
        #--------------------------------------------------
        self.merged_df_outg = self.merged_df_outg.loc[
            ~self.merged_df_outg.index.get_level_values(
                self.mecpx_build_info_dict['trsf_pole_nb_idfr'][1]
            ).isin(self.trsf_pole_nbs_to_remove)
        ].copy()
        #-------------------------
        self.merged_df_otbl = self.merged_df_otbl.loc[
            ~self.merged_df_otbl.index.get_level_values(
                self.mecpx_build_info_dict['trsf_pole_nb_idfr'][1]
            ).isin(self.trsf_pole_nbs_to_remove)
        ].copy()
        #-------------------------
        if self.include_prbl:
            self.merged_df_prbl = self.merged_df_prbl.loc[
                ~self.merged_df_prbl.index.get_level_values(
                    self.mecpx_build_info_dict['trsf_pole_nb_idfr'][1]
                ).isin(self.trsf_pole_nbs_to_remove)
            ].copy()

    def compile_trsf_location_info_df(
        self
    ):
        r"""
        """
        #-------------------------
        found_dirs_by_dataset = self.find_all_data_base_dirs()
        #-------------------------
        dfs_name = 'trsf_location_info_df.pkl'
        trsf_loc_info_dfs = []
        for dataset_i, base_dirs_i in found_dirs_by_dataset.items():
            for date_pd_j, base_dir_j in base_dirs_i.items():
                path_ij = os.path.join(base_dir_j, dfs_name)
                assert(os.path.exists(path_ij))
                df_ij = pd.read_pickle(path_ij)
                #-----
                df_ij['dataset'] = dataset_i
                df_ij['date_pd'] = date_pd_j        
                #-----
                trsf_loc_info_dfs.append(df_ij)
        #-------------------------
        cols = trsf_loc_info_dfs[0].columns.tolist()
        for df_i in trsf_loc_info_dfs:
            assert(set(df_i.columns.tolist()).symmetric_difference(set(cols))==set())
        #-----
        trsf_loc_info_df = Utilities_df.concat_dfs(
            dfs                  = trsf_loc_info_dfs, 
            axis                 = 0, 
            make_col_types_equal = True
        )
        #-------------------------
        return trsf_loc_info_df

            
    @staticmethod
    def set_target_val_1_by_idx(
        df,
        val_1_idxs,
        remove_others_from_outages=False, 
        target_col=('is_outg', 'is_outg'), 
        from_outg_col=('from_outg', 'from_outg')
    ):
        r"""
        Set the target value to 1 for those in df with indices found in val_1_idxs.
    
        df:
            pd.DataFrame object OR a list of such objects
        
        val_1_idxs:
            A list containing the indices whose target values should be set to 1.
            Note, in general, val_1_idxs can contain more indices than those found in df, 
              as an intersection will be used in the code.
              
        remove_others_from_outages:
            If True, those with df[from_outg_col]==1 and df[target_col]==0 will be removed.
            This is useful if one wants to use a subset of outages as the target and remove all other outages from the data.
            
        target_col
        
        from_outg_col:
            Only used if remove_others_from_outages==True
            
        """
        #----------------------------------------------------------------------------------------------------
        assert(df is None or Utilities.is_object_one_of_types(df, [pd.DataFrame, list]))
        if isinstance(df, list):
            return_dfs = []
            for df_i in df:
                df_i_fnl = OutageModeler.set_target_val_1_by_idx(
                    df                         = df_i, 
                    val_1_idxs                 = val_1_idxs,
                    remove_others_from_outages = remove_others_from_outages, 
                    target_col                 = target_col, 
                    from_outg_col              = from_outg_col
                )
                return_dfs.append(df_i_fnl)
            return return_dfs
        #----------------------------------------------------------------------------------------------------
        if df is None or df.shape[0]==0:
            return df
        #----------------------------------------------------------------------------------------------------
        # First, set all target values to 0
        df[target_col] = 0
        
        #-------------------------
        # Set the target values to 1 for any indices in val_1_idxs
        df.loc[list(set(df.index).intersection(set(val_1_idxs))), target_col] = 1
        
        #-------------------------
        # Remove other outages not marked as target==1 if remove_others_from_outages==True
        if remove_others_from_outages:
            # Drop any entries which are from the outages collection but not marked as target==1
            # NOTE: The method below is a little safer than finding the indices to drop and then calling .drop()
            #         as this should be safe against duplicate indices, whereas the .drop method would not be.
            #       However, I do not expect duplicate indices to occur, so either would probably be fine.
            df = df[
                ~(
                    (df[from_outg_col]==1) & 
                    (df[target_col]==0)
                )
            ]
            
        #-------------------------
        return df
    
    
    def finalize_merged_dfs(
        self
    ):
        r"""
        Odds and ends...
        
        If self.include_month or self.merge_eemsp, call self.add_time_infos_features() to:
            - Include the month of the event as a feature, and/or
            - Convert transformer installation date to transformer age
    
        If self.remove_scheduled_outages, remove scheduled outages using MECPODf.get_cpo_df_subset_excluding_mjr_mnr_causes
          with mnr_causes_to_exclude= ['SCO', 'SO']
    
        If self.trsf_pole_nbs_to_remove, remove using remove_trsf_pole_nbs
    
        If self.an_keys_to_drop, remove the keys
    
        Set from_outg_col
    
        Encode EEMSP values
        
        Set target values, using self.outgs_slicer
        """
        #--------------------------------------------------
        assert(
            self.merged_df_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 
        )
        #--------------------------------------------------
        if self.include_month or self.merge_eemsp:
            self.add_time_infos_features()
        #--------------------------------------------------
        if self.remove_scheduled_outages:
            self.merged_df_outg =  MECPODf.get_cpo_df_subset_excluding_mjr_mnr_causes( 
                cpo_df                    = self.merged_df_outg, 
                mjr_mnr_causes_to_exclude = None, 
                mjr_causes_to_exclude     = None,
                mnr_causes_to_exclude     = ['SCO', 'SO'], 
                outg_rec_nb_col           = self.mecpx_build_info_dict['rec_nb_idfr']
            )
        #--------------------------------------------------
        if self.trsf_pole_nbs_to_remove is not None and len(self.trsf_pole_nbs_to_remove)>0:
            self.remove_trsf_pole_nbs_from_merged_dfs()
    
        #--------------------------------------------------
        if self.an_keys_to_drop is not None:
            assert(set(self.an_keys_to_drop).difference(set(self.merged_df_outg.columns.get_level_values(0).unique().tolist()))==set())
            #-----
            self.merged_df_outg = self.merged_df_outg.drop(columns=self.an_keys_to_drop, level=0)
            self.merged_df_otbl = self.merged_df_otbl.drop(columns=self.an_keys_to_drop, level=0)
            if self.include_prbl:
                self.merged_df_prbl = self.merged_df_prbl.drop(columns=self.an_keys_to_drop, level=0)  
    
        #--------------------------------------------------
        # Set from_outg_col
        # Add 'from_outg' information so I can track how many are in target=1 and target=0 
        if self.merged_df_outg.columns.nlevels>1:
            assert(len(self.from_outg_col)==self.merged_df_outg.columns.nlevels)
            #-----
            self.merged_df_outg[self.from_outg_col] = 1
            self.merged_df_otbl[self.from_outg_col] = 0
            if self.include_prbl:
                self.merged_df_prbl[self.from_outg_col] = 0
                
        #--------------------------------------------------
        # Set target values
        #-------------------------
        # First, if self.dovs_df is not built, build it.
        if self.dovs_df is None or self.dovs_df.shape[0]==0:
            self.build_dovs_df()
        #------------------------------
        self.dovs_df = self.outgs_slicer.set_simple_column_value(df=self.dovs_df, column=self.target_col, value=1)
        outg_idxs_i = self.dovs_df[self.dovs_df[self.target_col]==1].index
        #------------------------------
        [
            self.merged_df_outg, 
            self.merged_df_otbl, 
            self.merged_df_prbl
        ] = OutageModeler.set_target_val_1_by_idx(
            df                         = [
                self.merged_df_outg, 
                self.merged_df_otbl, 
                self.merged_df_prbl
            ],
            val_1_idxs                 = outg_idxs_i,
            remove_others_from_outages = self.remove_others_from_outages, 
            target_col                 = self.target_col, 
            from_outg_col              = self.from_outg_col
        )
    
        #--------------------------------------------------
        self.__df_target = pd.concat([
            self.merged_df_outg[[self.target_col, self.from_outg_col]], 
            self.merged_df_otbl[[self.target_col, self.from_outg_col]]
        ])
        if self.include_prbl:
            self.__df_target = pd.concat([
                self.__df_target, 
                self.merged_df_prbl[[self.target_col, self.from_outg_col]]
            ])
    
        #--------------------------------------------------
        # Make sure DFs end with self.from_outg_col, self.target_col
        self.merged_df_outg = Utilities_df.move_cols_to_back(df=self.merged_df_outg, cols_to_move=[self.from_outg_col, self.target_col])
        self.merged_df_otbl = Utilities_df.move_cols_to_back(df=self.merged_df_otbl, cols_to_move=[self.from_outg_col, self.target_col])
        if self.include_prbl:
            self.merged_df_prbl = Utilities_df.move_cols_to_back(df=self.merged_df_prbl, cols_to_move=[self.from_outg_col, self.target_col])
            
        #--------------------------------------------------
        # IMPORTANT: Notice, we do not call save_merged_dfs/save_counts_series here.  This is on purpose.  The final form or merged_dfs
        #              will ultimately depend on decisions such as include_month, merge_eemsp, remove_scheduled_outages, etc.
        #            We want the saved versions of merged_dfs to be able to be used regardless of what user chooses for these parameters.
        #            If we did, e.g., save here after the user opted to include_month, then the user went back to run a model without the month
        #              it would still be included (assuming __force_build==False, as is standard) 
        

            
    @staticmethod
    def get_cpx_outg_df_subset_by_outg_datetime(
        cpx_outg_df, 
        date_0,
        date_1, 
        outg_rec_nb_idfr='index', 
        return_notin_also=False
    ):
        r"""
        Returns the subset of cpx_outg_df whose associated outages are within [date_0, date_1)
        """
        #-------------------------
        if date_0 is None and date_1 is None:
            if not return_notin_also:
                return cpx_outg_df
            else:
                return cpx_outg_df, pd.DataFrame()
        #-------------------------
        if date_0 is None:
            date_0 = pd.Timestamp.min
        #-----
        if date_1 is None:
            date_1 = pd.Timestamp.max
        #-------------------------
        if not isinstance(date_0, datetime.datetime):
            date_0 = pd.to_datetime(date_0)
        if not isinstance(date_1, datetime.datetime):
            date_1 = pd.to_datetime(date_1)
        assert(isinstance(date_0, datetime.datetime))
        assert(isinstance(date_1, datetime.datetime))
        #-------------------------
        contstruct_df_args=None
        build_sql_function = DOVSOutages_SQL.build_sql_outage
        build_sql_function_kwargs=dict(
            datetime_col='DT_OFF_TS_FULL', 
            cols_of_interest=[
                'OUTG_REC_NB', 
                dict(field_desc=f"DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)", 
                     alias='DT_OFF_TS_FULL', table_alias_prefix=None)
            ]
        )
        #-----
        df_off_df = DOVSOutages.get_outg_info_for_df(
            df=cpx_outg_df, 
            outg_rec_nb_idfr=outg_rec_nb_idfr, 
            contstruct_df_args=contstruct_df_args, 
            build_sql_function=build_sql_function, 
            build_sql_function_kwargs=build_sql_function_kwargs, 
            set_outg_rec_nb_as_index=True
        )
        #-------------------------
        outg_rec_nbs = MECPODf.get_outg_rec_nbs_from_cpo_df(cpo_df=cpx_outg_df, idfr=outg_rec_nb_idfr)
        assert(len(cpx_outg_df)==len(outg_rec_nbs)) # Important in ensuring proper selection towards end of function
        #-----
        subset_outg_rec_nbs = df_off_df.loc[(df_off_df['DT_OFF_TS_FULL'] >= date_0) & 
                                            (df_off_df['DT_OFF_TS_FULL'] < date_1)].index.unique().tolist()
        cpx_outg_df_subset = cpx_outg_df[outg_rec_nbs.isin(subset_outg_rec_nbs)].copy()
        #-------------------------
        if not return_notin_also:
            return cpx_outg_df_subset
        else:
            cpx_outg_df_notin = cpx_outg_df[~outg_rec_nbs.isin(subset_outg_rec_nbs)].copy()
            return cpx_outg_df_subset, cpx_outg_df_notin
            
    @staticmethod        
    def get_cpx_baseline_df_subset_by_datetime(
        cpx_bsln_df, 
        bsln_time_infos_df, 
        date_0,
        date_1,
        bsln_time_infos_time_col='t_min', 
        return_notin_also=False, 
        merge_time_info_to_cpx_bsln_df=False
    ):
        r"""
        cpx_bsln_df and bsln_time_infos_df must have same indices.
        Typically, these are no_outg_rec_nb and trsf_pole_nb.
        
        NOTE: Have found merging can be taxing (from memory standpoint) when DFs are large.
              Hence why default merge_time_info_to_cpx_bsln_df=False
        """
        #-------------------------
        if date_0 is None and date_1 is None:
            if not return_notin_also:
                return cpx_outg_df
            else:
                return cpx_outg_df, pd.DataFrame()
        #-------------------------
        if date_0 is None:
            date_0 = pd.Timestamp.min
        #-----
        if date_1 is None:
            date_1 = pd.Timestamp.max
        #-------------------------
        if not isinstance(date_0, datetime.datetime):
            date_0 = pd.to_datetime(date_0)
        if not isinstance(date_1, datetime.datetime):
            date_1 = pd.to_datetime(date_1)
        assert(isinstance(date_0, datetime.datetime))
        assert(isinstance(date_1, datetime.datetime))
        #-------------------------
        assert(len(set(cpx_bsln_df.index.names).symmetric_difference(set(bsln_time_infos_df.index.names)))==0)
        if bsln_time_infos_df.index.names!=cpx_bsln_df.index.names:
            bsln_time_infos_df = bsln_time_infos_df.reset_index().set_index(cpx_bsln_df.index.names)
        #-------------------------
        if bsln_time_infos_df.columns.nlevels>1 and not Utilities.is_object_one_of_types(bsln_time_infos_time_col, [list, tuple]):
            bsln_time_infos_time_col = Utilities_df.find_single_col_in_multiindex_cols(
                df=bsln_time_infos_df, 
                col=bsln_time_infos_time_col
            )
        assert(bsln_time_infos_time_col in bsln_time_infos_df.columns.tolist())
        #-------------------------
        bsln_time_infos_df = bsln_time_infos_df[[bsln_time_infos_time_col]]
        # Since collected over multiple files, sometimes a no-outage 'event' is split over multiple files
        #   with, e.g., different PNs.  So, duplicates must be dropped
        # Need to called reset_index because otherwise only bsln_time_infos_time_col will be considered, whereas
        #   here a duplicate must have same no_outg_rec_nb, trsf_pole_nb, and bsln_time_infos_time_col!
        bsln_time_infos_df = bsln_time_infos_df.reset_index().drop_duplicates().set_index(bsln_time_infos_df.index.names)
        #-------------------------
        # There should be an entry in bsln_time_infos_df for each in cpx_bsln_df
        # The reverse is NOT true: If no events found for specifiec timeframe, no entries will exist in cpx_bsln_df
        assert(len(set(cpx_bsln_df.index).difference(set(bsln_time_infos_df.index)))==0)
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        if not merge_time_info_to_cpx_bsln_df:
            subset_idxs = bsln_time_infos_df.loc[
                (bsln_time_infos_df[bsln_time_infos_time_col] >= date_0) &
                (bsln_time_infos_df[bsln_time_infos_time_col] < date_1)
            ].index.unique().tolist()
            cpx_bsln_df_subset = cpx_bsln_df[cpx_bsln_df.index.isin(subset_idxs)].copy()
            #-------------------------
            if not return_notin_also:
                return cpx_bsln_df_subset
            else:
                cpx_bsln_df_notin = cpx_bsln_df[~cpx_bsln_df.index.isin(subset_idxs)].copy()
                return cpx_bsln_df_subset, cpx_bsln_df_notin
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        else:
            cpx_bsln_df_wt = OutageModeler.merge_cpx_df_w_time_infos(
                cpx_df=cpx_bsln_df, 
                time_infos_df=bsln_time_infos_df, 
                time_infos_drop_dupls_subset=['index', bsln_time_infos_time_col]
            )
            #-------------------------
            # Merging will add dummy levels if needed, so adjust bsln_time_infos_time_col if needed
            if not bsln_time_infos_time_col in cpx_bsln_df_wt.columns.tolist():
                bsln_time_infos_time_col = Utilities_df.find_single_col_in_multiindex_cols(
                    df=cpx_bsln_df_wt, 
                    col=bsln_time_infos_time_col
                )
            assert(bsln_time_infos_time_col in cpx_bsln_df_wt.columns.tolist())
            #-------------------------
            cpx_bsln_df_wt_subset = cpx_bsln_df_wt.loc[
                (cpx_bsln_df_wt[bsln_time_infos_time_col] >= date_0) &
                (cpx_bsln_df_wt[bsln_time_infos_time_col] < date_1)
            ]
            if not return_notin_also:
                return cpx_bsln_df_wt_subset
            else:
                cpx_bsln_df_wt_notin = cpx_bsln_df_wt.loc[
                    ~((cpx_bsln_df_wt[bsln_time_infos_time_col] >= date_0) &
                    (cpx_bsln_df_wt[bsln_time_infos_time_col] < date_1))
                ]
                return cpx_bsln_df_wt_subset, cpx_bsln_df_wt_notin
      
    @staticmethod
    def train_test_split_df_by_outage(
        df, 
        outg_rec_nb_idfr, 
        test_size, 
        random_state=None
        
    ):
        r"""
        This is simply a train-test split according to the outage groups.
        i.e., this enforces that all entries for a given outage remain together (either all in train or all in test)
              and never split across train/test
        e.g., if outage 1 affects 5 transformers, all 5 transformers will be in train or all will be in test, it will never
              occur that 3 are in train and 2 in test
              
              
        outg_rec_nb_idfr:
            This directs from where the outg_rec_nbs will be retrieved.
            This should be a string, list, or tuple.
            If the outg_rec_nbs are located in a column, idfr should simply be the column
                - Single index columns --> simple string
                - MultiIndex columns   --> appropriate tuple to identify column
            If the outg_rec_nbs are located in the index:
                - Single level index --> simple string 'index' or 'index_0'
                - MultiIndex index:  --> 
                    - string f'index_{level}', where level is the index level containing the outg_rec_nbs
                    - tuple of length 2, with 0th element ='index' and 1st element = idx_level_name where
                        idx_level_name is the name of the index level containing the outg_rec_nbs 
        """
        #-------------------------
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        outg_rec_nbs = DOVSOutages.get_outg_rec_nbs_list_from_df(
            df=df, 
            idfr=outg_rec_nb_idfr, 
            unique_only=False
        )
        #-----
        split = gss.split(df, groups=outg_rec_nbs)
        train_idxs, test_idxs = next(split)
        #-----
        df_train = df.iloc[train_idxs].copy()
        df_test  = df.iloc[test_idxs].copy()
        #-------------------------
        # Make sure the operation worked as expected
        outg_rec_nbs_train = DOVSOutages.get_outg_rec_nbs_list_from_df(
            df=df_train, 
            idfr=outg_rec_nb_idfr, 
            unique_only=True
        )
        outg_rec_nbs_test = DOVSOutages.get_outg_rec_nbs_list_from_df(
            df=df_test, 
            idfr=outg_rec_nb_idfr, 
            unique_only=True
        )
        assert(len(set(outg_rec_nbs_train).intersection(set(outg_rec_nbs_test)))==0)
        #-------------------------
        return df_train, df_test
        
        
    @staticmethod
    def adjust_date_range(
        date_range, 
        assert_conversion=True
    ):
        r"""
        Ensures that date_range is None or a list/tuple of length 2
        If date_range is None ==> return None
        If date_range = [None, None] ==> return None
        If date_range = [None, dt_1] ==> return [pd.Timestamp.min, dt_1]
        If date_range = [dt_0, None] ==> return [dt_0, pd.Timestamp.max]
        If date_range = [dt_0, dt_1] ==> return [dt_0, dt_1]
        """
        #-------------------------
        assert(
            date_range is None or 
            (Utilities.is_object_one_of_types(date_range, [list, tuple]) and len(date_range)==2)
        )
        #-------------------------
        if date_range is None:
            return None
        #-------------------------
        if all([x is None for x in date_range]):
            return None
        #-------------------------
        if date_range[0] is None:
            date_range[0] = pd.Timestamp.min
        if date_range[1] is None:
            date_range[1] = pd.Timestamp.max
        #-------------------------
        if assert_conversion:
            # Make sure the objects in date_range can be converted to datetime/timestamp objects
            _ = [pd.to_datetime(x) for x in date_range]
        #-------------------------
        return date_range
    
    
    @staticmethod
    def train_test_split_outg(
        df, 
        split_train_test_by_outg = True, 
        test_size                = 0.33, 
        random_state             = None, 
        get_train_test_by_date   = False,
        date_range_train         = None, 
        date_range_test          = None, 
        outg_rec_nb_idfr         = ('index', 'outg_rec_nb'), 
        date_range_holdout       = None, 
        verbose                  = True
    ):
        r"""
        This function allows one to split the data in three different ways:
            (1) Run train-test split according to the outage groups
                i.e., this enforces that all entries for a given outage remain together (either all in train or all in test)
                      and never split across train/test
                To achieve this functionality:
                    Set split_train_test_by_outg=True OR 
                    Set get_train_test_by_date=True and supply date_range_train/_test
            (2) Split by date, using date_range_train and date_range_test
                NOTE: By virtue of splitting according to when outage events occur, this method always enforces a train-test split
                        according to the outage groups (i.e., this enforces that all entries for a given outage remain together, and it does
                        this regardless of the value of split_train_test_by_outg)
                To achieve this functionality:
                    Set get_train_test_by_date=True and supply date_range_train/_test
                NOTE: If only one of date_range_train/_test supplied, then the supplied range will be split using test_size to achieve the 
                      train and test ranges
            (3) Run a standard sklearn.model_selection.train_test_split
                To achieve this functionality:
                    Set split_train_test_by_outg=False and get_train_test_by_date=False
        In any case, if one wants holdout data, those are always simply returned using date_range_holdout
    
        !!!!! IMPORTANT !!!!!
            When using get_train_test_by_date==False, the dates contained in date_range_train/_test are still used to
            perform an initial date cut on the data before splitting train and test
    
        date_range_train/_test/_holdout:
            If supplied, these must be lists (or tuples) of length 2 representing ranges of dates
            These are NOT ALLOWED TO OVERLAP!
            If get_train_test_by_date==True, date_range_train and date_range_test MUST BE SUPPLIED
            IMPORTANT: When using get_train_test_by_date==False, the dates contained in date_range_train/_test are still used to
                       perform an initial date cut on the data before splitting train and test
        """
        #----------------------------------------------------------------------------------------------------
        # NOTE: adjust_date_range makes sure input is None or a list/tuple of length 2
        date_range_train   = OutageModeler.adjust_date_range(date_range=date_range_train, assert_conversion=True)
        date_range_test    = OutageModeler.adjust_date_range(date_range=date_range_test, assert_conversion=True)
        date_range_holdout = OutageModeler.adjust_date_range(date_range=date_range_holdout, assert_conversion=True)
        #-----
        if get_train_test_by_date:
            assert(not pd.Interval(*date_range_train).overlaps(pd.Interval(*date_range_test)))
        if date_range_holdout is not None:
            assert(Utilities.is_object_one_of_types(date_range_holdout, [list, tuple]) and len(date_range_holdout)==2)
            #-----
            try:
                assert(not pd.Interval(*date_range_train).overlaps(pd.Interval(*date_range_holdout)))
                assert(not pd.Interval(*date_range_test).overlaps(pd.Interval(*date_range_holdout)))
            except:
                print("OutageModeler: date_range_holdout overlaps with date_range_train and/or date_range_test!")
                print(f"\tdate_range_train = {date_range_train}\n\tdate_range_test = {date_range_test}\n\tdate_range_holdout = {date_range_holdout}")
                print("Overlapping holdout with train or test data are strictly forbidden.\nCRASH IMMINENT!")
                assert(0)
        #----------------------------------------------------------------------------------------------------
        if get_train_test_by_date:
            # At least one of date_range_train/_test must not be None
            assert(date_range_train is not None or date_range_test is not None)
            #-------------------------
            # If one is None, then use non-None range together with test size to get train and test ranges
            if date_range_train is None or date_range_test is None:
                assert(test_size is not None and 0 < test_size < 1)
                if verbose:
                    print('WARNING: get_train_test_by_date==True but only one of date_range_train/_test supplied')
                    print(f'Will use supplied range and split according to test_size = {test_size}')
                date_0_tt = np.min([x for x in [date_range_train, date_range_test] if x])
                date_1_tt = np.max([x for x in [date_range_train, date_range_test] if x])
    
                delta = date_1_tt - date_0_tt
                train_size = ((1.0-test_size)*delta).round('D')
                #-----
                date_range_train = [date_0_tt,                               date_0_tt+train_size]
                date_range_test  = [date_0_tt+train_size+pd.Timedelta('1D'), date_1_tt]
            #-------------------------
            df_train = OutageModeler.get_cpx_outg_df_subset_by_outg_datetime(
                cpx_outg_df       = df.copy(), 
                date_0            = date_range_train[0],
                date_1            = date_range_train[1], 
                outg_rec_nb_idfr  = outg_rec_nb_idfr, 
                return_notin_also = False
            )
            df_test = OutageModeler.get_cpx_outg_df_subset_by_outg_datetime(
                cpx_outg_df       = df.copy(), 
                date_0            = date_range_test[0],
                date_1            = date_range_test[1], 
                outg_rec_nb_idfr  = outg_rec_nb_idfr, 
                return_notin_also = False
            )
        else:
            if date_range_train is None and date_range_test is None:
                df_train_test = df.copy()
            else:
                date_0_tt = np.min([x for x in [date_range_train, date_range_test] if x])
                date_1_tt = np.max([x for x in [date_range_train, date_range_test] if x])
                #-----
                df_train_test = OutageModeler.get_cpx_outg_df_subset_by_outg_datetime(
                    cpx_outg_df       = df.copy(), 
                    date_0            = date_0_tt,
                    date_1            = date_1_tt, 
                    outg_rec_nb_idfr  = outg_rec_nb_idfr, 
                    return_notin_also = False
                )
            #-----
            assert(test_size is not None and 0 < test_size < 1)
            if split_train_test_by_outg:
                df_train, df_test = OutageModeler.train_test_split_df_by_outage(
                    df               = df_train_test, 
                    outg_rec_nb_idfr = outg_rec_nb_idfr, 
                    test_size        = test_size, 
                    random_state     = random_state
                )
            else:
                df_train, df_test = train_test_split(
                    df_train_test, 
                    test_size    = test_size, 
                    random_state = random_state
                )
        #----------------------------------------------------------------------------------------------------
        if date_range_holdout is None:
            df_holdout = pd.DataFrame()
        else:
            df_holdout = OutageModeler.get_cpx_outg_df_subset_by_outg_datetime(
                cpx_outg_df       = df.copy(), 
                date_0            = date_0_holdout,
                date_1            = date_1_holdout, 
                outg_rec_nb_idfr  = outg_rec_nb_idfr, 
                return_notin_also = False
            )
        #----------------------------------------------------------------------------------------------------
        return_dict = dict(
            train   = df_train, 
            test    = df_test, 
            holdout = df_holdout
        )
        return return_dict
        
        
    @staticmethod
    def train_test_split_bsln(
        df, 
        split_train_test_by_outg = True, 
        test_size                = 0.33, 
        random_state             = None, 
        get_train_test_by_date   = False,
        bsln_time_infos_df       = None, 
        date_range_train         = None, 
        date_range_test          = None, 
        no_outg_rec_nb_idfr      = ('index', 'no_outg_rec_nb'), 
        bsln_time_infos_time_col = 't_min', 
        date_range_holdout       = None, 
        verbose                  = True
    ):
        r"""
        See train_test_split_outg for more information, as this is essentially the same function
        """
        #----------------------------------------------------------------------------------------------------
        # NOTE: adjust_date_range makes sure input is None or a list/tuple of length 2
        date_range_train   = OutageModeler.adjust_date_range(date_range=date_range_train, assert_conversion=True)
        date_range_test    = OutageModeler.adjust_date_range(date_range=date_range_test, assert_conversion=True)
        date_range_holdout = OutageModeler.adjust_date_range(date_range=date_range_holdout, assert_conversion=True)
        #-----
        if get_train_test_by_date:
            assert(not pd.Interval(*date_range_train).overlaps(pd.Interval(*date_range_test)))
        if date_range_holdout is not None:
            assert(Utilities.is_object_one_of_types(date_range_holdout, [list, tuple]) and len(date_range_holdout)==2)
            #-----
            assert(not pd.Interval(*date_range_train).overlaps(pd.Interval(*date_range_holdout)))
            assert(not pd.Interval(*date_range_test).overlaps(pd.Interval(*date_range_holdout)))
        #----------------------------------------------------------------------------------------------------
        if get_train_test_by_date:
            assert(bsln_time_infos_df is not None and isinstance(bsln_time_infos_df, pd.DataFrame))
            # At least one of date_range_train/_test must not be None
            assert(date_range_train is not None or date_range_test is not None)
            #-------------------------
            # If one is None, then use non-None range together with test size to get train and test ranges
            if date_range_train is None or date_range_test is None:
                assert(test_size is not None and 0 < test_size < 1)
                if verbose:
                    print('WARNING: get_train_test_by_date==True but only one of date_range_train/_test supplied')
                    print(f'Will use supplied range and split according to test_size = {test_size}')
                date_0_tt = np.min([x for x in [date_range_train, date_range_test] if x])
                date_1_tt = np.max([x for x in [date_range_train, date_range_test] if x])
    
                delta = date_1_tt - date_0_tt
                train_size = ((1.0-test_size)*delta).round('D')
                #-----
                date_range_train = [date_0_tt,                               date_0_tt+train_size]
                date_range_test  = [date_0_tt+train_size+pd.Timedelta('1D'), date_1_tt]
            #-------------------------
            df_train = OutageModeler.get_cpx_baseline_df_subset_by_datetime(
                cpx_bsln_df                    = df.copy(), 
                bsln_time_infos_df             = bsln_time_infos_df.copy(), 
                date_0                         = date_range_train[0],
                date_1                         = date_range_train[1], 
                bsln_time_infos_time_col       = bsln_time_infos_time_col, 
                return_notin_also              = False, 
                merge_time_info_to_cpx_bsln_df = False
            )
            df_test = OutageModeler.get_cpx_baseline_df_subset_by_datetime(
                cpx_bsln_df                    = df.copy(), 
                bsln_time_infos_df             = bsln_time_infos_df.copy(), 
                date_0                         = date_range_test[0],
                date_1                         = date_range_test[1], 
                bsln_time_infos_time_col       = bsln_time_infos_time_col, 
                return_notin_also              = False, 
                merge_time_info_to_cpx_bsln_df = False
            )
        
        else:
            if date_range_train is None and date_range_test is None:
                df_train_test = df.copy()
            else:
                assert(bsln_time_infos_df is not None and isinstance(bsln_time_infos_df, pd.DataFrame))
                #-----
                date_0_tt = np.min([x for x in [date_range_train, date_range_test] if x])
                date_1_tt = np.max([x for x in [date_range_train, date_range_test] if x])
                #-----
                df_train_test = OutageModeler.get_cpx_baseline_df_subset_by_datetime(
                    cpx_bsln_df                    = df.copy(), 
                    bsln_time_infos_df             = bsln_time_infos_df.copy(), 
                    date_0                         = date_0_tt,
                    date_1                         = date_1_tt, 
                    bsln_time_infos_time_col       = bsln_time_infos_time_col, 
                    return_notin_also              = False, 
                    merge_time_info_to_cpx_bsln_df = False
                )
            #-----
            assert(test_size is not None and 0 < test_size < 1)
            if split_train_test_by_outg:
                df_train, df_test = OutageModeler.train_test_split_df_by_outage(
                    df               = df_train_test, 
                    outg_rec_nb_idfr = no_outg_rec_nb_idfr, 
                    test_size        = test_size, 
                    random_state     = random_state
                )
            else:
                df_train, df_test = train_test_split(
                    df_train_test, 
                    test_size    = test_size, 
                    random_state = random_state
                )
        #----------------------------------------------------------------------------------------------------
        if date_range_holdout is None:
            df_holdout = pd.DataFrame()
        else:
            df_holdout = OutageModeler.get_cpx_baseline_df_subset_by_datetime(
                cpx_bsln_df                    = df.copy(), 
                bsln_time_infos_df             = bsln_time_infos_df.copy(), 
                date_0                         = date_0_holdout,
                date_1                         = date_1_holdout, 
                bsln_time_infos_time_col       = bsln_time_infos_time_col, 
                return_notin_also              = False, 
                merge_time_info_to_cpx_bsln_df = False
            )
        #----------------------------------------------------------------------------------------------------
        return_dict = dict(
            train   = df_train, 
            test    = df_test, 
            holdout = df_holdout
        )
        return return_dict
        
    def build_train_test_data(
        self,
        assert_can_model = True, 
        verbose          = True
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        if self.get_train_test_by_date:
            # Make sure the minimum required dates are supplied
            needed_dates = [self.date_0_train, self.date_1_train, self.date_0_test, self.date_1_test]
            if any([x is None for x in needed_dates]):
                self.__can_model = False
                if verbose:
                    print('Cannot model!!!!! needed_dates missing!')
                    print(f'\tself.date_0_train = {self.date_0_train}')
                    print(f'\tself.date_1_train = {self.date_1_train}')
                    print(f'\tself.date_0_test  = {self.date_0_test}')
                    print(f'\tself.date_1_test  = {self.date_1_test}')
                if assert_can_model:
                    assert(0)
                return
            #-----
            # Make sure the minimum required dates can be converted to datetime/timestamp objects
            _ = [pd.to_datetime(x) for x in needed_dates]
        #----------------------------------------------------------------------------------------------------
        if not (
            self.merged_df_outg.shape[0]>0 and 
            self.merged_df_otbl.shape[0]>0 and 
            self.time_infos_df.shape[0]>0 
        ):
            self.__can_model = False
            if verbose:
                print('Cannot model!!!!! merged_df and/or time_infos_df missing!')
                print(f'\tself.merged_df_outg.shape[0] = {self.merged_df_outg.shape[0]}')
                print(f'\tself.merged_df_otbl.shape[0] = {self.merged_df_otbl.shape[0]}')
                print(f'\tself.time_infos_df.shape[0]  = {self.time_infos_df.shape[0]}')
            if assert_can_model:
                assert(0)
            return
        #-----
        include_prbl = False
        if self.include_prbl:
            include_prbl = True
        #----------------------------------------------------------------------------------------------------
        if self.date_0_holdout is not None or self.date_1_holdout is not None:
            if(
                self.date_0_holdout is None or 
                self.date_1_holdout is None
            ):
                self.__can_model = False
                if verbose:
                    print('Cannot model!!!!! date_0/andor1_holdout missing!')
                    print(f'\tself.date_0_holdout = {self.date_0_holdout}')
                    print(f'\tself.date_1_holdout = {self.date_1_holdout}')
                if assert_can_model:
                    assert(0)
                return
            date_range_holdout = [self.date_0_holdout, self.date_1_holdout]
        else:
            date_range_holdout = None
        #----------------------------------------------------------------------------------------------------
        split_data_dict_outg = OutageModeler.train_test_split_outg(
            df                       = self.merged_df_outg, 
            split_train_test_by_outg = self.split_train_test_by_outg, 
            test_size                = self.test_size, 
            random_state             = self.random_state, 
            get_train_test_by_date   = self.get_train_test_by_date,
            date_range_train         = [self.date_0_train, self.date_1_train], 
            date_range_test          = [self.date_0_test,  self.date_1_test], 
            outg_rec_nb_idfr         = self.mecpx_build_info_dict['rec_nb_idfr'], 
            date_range_holdout       = date_range_holdout, 
            verbose                  = verbose
        )
        #-------------------------
        split_data_dict_otbl = OutageModeler.train_test_split_bsln(
            df                       = self.merged_df_otbl, 
            split_train_test_by_outg = self.split_train_test_by_outg, 
            test_size                = self.test_size, 
            random_state             = self.random_state, 
            get_train_test_by_date   = self.get_train_test_by_date,
            bsln_time_infos_df       = self.time_infos_df, 
            date_range_train         = [self.date_0_train, self.date_1_train], 
            date_range_test          = [self.date_0_test,  self.date_1_test], 
            no_outg_rec_nb_idfr      = self.mecpx_build_info_dict['rec_nb_idfr'], 
            bsln_time_infos_time_col = 't_min', 
            date_range_holdout       = date_range_holdout, 
            verbose                  = verbose
        )
        #-------------------------
        if include_prbl:
            split_data_dict_prbl = OutageModeler.train_test_split_bsln(
                df                       = self.merged_df_prbl, 
                split_train_test_by_outg = self.split_train_test_by_outg, 
                test_size                = self.test_size, 
                random_state             = self.random_state, 
                get_train_test_by_date   = self.get_train_test_by_date,
                bsln_time_infos_df       = self.time_infos_df, 
                date_range_train         = [self.date_0_train, self.date_1_train], 
                date_range_test          = [self.date_0_test,  self.date_1_test], 
                no_outg_rec_nb_idfr      = self.mecpx_build_info_dict['rec_nb_idfr'], 
                bsln_time_infos_time_col = 't_min', 
                date_range_holdout       = date_range_holdout, 
                verbose                  = verbose
            )
        #----------------------------------------------------------------------------------------------------
        if include_prbl:
            addtnl_bsln_train   = pd.concat([split_data_dict_otbl['train'],   split_data_dict_prbl['train']])
            addtnl_bsln_test    = pd.concat([split_data_dict_otbl['test'],    split_data_dict_prbl['test']])
            addtnl_bsln_holdout = pd.concat([split_data_dict_otbl['holdout'], split_data_dict_prbl['holdout']])
        else:
            addtnl_bsln_train   = split_data_dict_otbl['train']
            addtnl_bsln_test    = split_data_dict_otbl['test']
            addtnl_bsln_holdout = split_data_dict_otbl['holdout']
        #-------------------------
        if not(0 <= self.addtnl_bsln_frac <= 1.0):
            self.__can_model = False
            if verbose:
                print(f'Cannot model!!!!! self.addtnl_bsln_frac inappropriate (= {self.addtnl_bsln_frac})')
            if assert_can_model:
                assert(0)
            return
        if self.addtnl_bsln_frac < 1.0:
            addtnl_bsln_train   = addtnl_bsln_train.sample(frac=self.addtnl_bsln_frac, random_state=self.random_state)
            addtnl_bsln_test    = addtnl_bsln_test.sample(frac=self.addtnl_bsln_frac, random_state=self.random_state)
            addtnl_bsln_holdout = addtnl_bsln_holdout.sample(frac=self.addtnl_bsln_frac, random_state=self.random_state)
        #----------------------------------------------------------------------------------------------------
        df_train   = pd.concat([split_data_dict_outg['train'],   addtnl_bsln_train])
        df_test    = pd.concat([split_data_dict_outg['test'],    addtnl_bsln_test])
        df_holdout = pd.concat([split_data_dict_outg['holdout'], addtnl_bsln_holdout])
        
        #Shuffle the data
        df_train   = df_train.sample(frac=1, random_state=self.random_state)
        df_test    = df_test.sample(frac=1, random_state=self.random_state)
        df_holdout = df_holdout.sample(frac=1, random_state=self.random_state)
        
        #----------------------------------------------------------------------------------------------------
        self.__df_train   = df_train
        self.__df_test    = df_test
        self.__df_holdout = df_holdout
        
        
    @staticmethod
    def get_slicer_mjr_mnr_outg_causes(
        mjr_causes, 
        mnr_causes, 
        mjr_cause_col = ('outg_dummy_lvl_0', 'MJR_CAUSE_NM'), 
        mnr_cause_col = ('outg_dummy_lvl_0', 'MNR_CAUSE_NM')
    ):
        r"""
        For now, this slicer is only inclusive, meaning:
            - You can use it to select certain major and minor causes
            - You CANNOT use it to select all except certain major and minor causes
        Expanding the functionality would not be difficult, but it's not needed currently and leaving the
          functionality out makes this cleaner.
        Note: The reason this is only inclusive is because comparison_operator is set to None for the single slicers 
                below, so the DFSingleSlicer.set_standard_comparison_operator sets the operator to use.
        
        mjr_causes/mnr_causes:
            May be single causes (i.e., strings), lists of causes (i.e., list of strings), or None
        """
        #--------------------------------------------------
        if mjr_causes is None and mnr_causes is None:
            return DFSlicer()
        #--------------------------------------------------
        single_slicers = []
        #-------------------------
        if mjr_causes is not None:
            single_slicers.append(
                dict(
                    column              = mjr_cause_col, 
                    value               = mjr_causes, 
                    comparison_operator = None
                    
                )
            )
        #-------------------------
        if mnr_causes is not None:
            single_slicers.append(
                dict(
                    column              = mnr_cause_col, 
                    value               = mnr_causes, 
                    comparison_operator = None
                    
                )
            )
        #--------------------------------------------------
        outgs_slicer = DFSlicer(single_slicers=single_slicers)
        return outgs_slicer
    
    @staticmethod
    def get_slicer_xfmr_equip_typ_nms_of_interest(
        xfmr_equip_typ_nms_of_interest = ['TRANSFORMER, OH', 'TRANSFORMER, UG'], 
        equip_typ_nm_col               = ('outg_dummy_lvl_0', 'EQUIP_TYP_NM')
    ):
        r"""
        """
        #-------------------------
        outgs_slicer = DFSlicer(
            single_slicers = [
                dict(
                    column              = equip_typ_nm_col, 
                    value               = xfmr_equip_typ_nms_of_interest, 
                    comparison_operator = 'isin'
                )
            ]
        )
        #-------------------------
        return outgs_slicer
    
    @staticmethod
    def get_slicer_outg_rec_nbs(
        outg_rec_nbs, 
        outg_rec_nb_col = ('outg_dummy_lvl_0', 'OUTG_REC_NB')
    ):
        r"""
        """
        #-------------------------
        outgs_slicer = DFSlicer(
            single_slicers = [
                dict(
                    column              = outg_rec_nb_col, 
                    value               = outg_rec_nbs, 
                    comparison_operator = 'isin'
                )
            ]
        )
        #-------------------------
        return outgs_slicer
    
    @staticmethod
    def get_dummy_slicer():
        r"""
        Returns DFSlicer(), which won't perform any sort of slicing
        """
        #-------------------------
        return DFSlicer()        
        
    
    @staticmethod
    def ensure_target_val_1_min_pct(
        df,
        min_pct,
        target_col=('is_outg', 'is_outg'), 
        random_state=None, 
        assert_success=True, 
        return_discarded=False
    ):
        r"""
        Make sure the collection of entries with target value==1 comprises at least min_pct of the overall collection.
        If the percentage is below min_pct, entries are removed from the target value==0 collection until desired 
          percentage is reached.
        If the percentage is already above min_pct, simply return the df.
        
        min_pct:
            Should be between 0 and 100! (not, e.g., 0 and 1)
        """
        #-------------------------
        pct = 100*(df[target_col]==1).sum()/df.shape[0]
        #-------------------------
        if pct >= min_pct:
            return df
        #-------------------------
        # Need to determine how many entries from target==0 to keep 
        #   Define the number of target==1 values to be n_1 and the needed number of
        #     target==0 values to be n_0
        #   One must solve for n_0 in the following:  100*n_1/(n_1+n_0)=min_pct
        #   ==> n_0 = (100-min_pct)*n_1/min_pct
        n_1 = (df[target_col]==1).sum()
        n_0 = np.floor((100-min_pct)*n_1/min_pct).astype(int)
        #-------------------------
        df_1 = df[df[target_col]==1]
        df_0 = df[df[target_col]==0]
        df_0_sub = df_0.sample(n=n_0, replace=False, random_state=random_state)
        #-------------------------
        # Join df_1 and df_0_sub and randomize order
        return_df = pd.concat([df_1, df_0_sub])
        return_df = return_df.sample(frac=1, random_state=random_state)
        #-------------------------
        # Ensure operation was successful
        if assert_success:
            pct = 100*(return_df[target_col]==1).sum()/return_df.shape[0]
            assert(pct>=min_pct)
        #-------------------------
        if return_discarded:
            df_0_discarded = df_0[~df_0.index.isin(df_0_sub.index)]
            assert(df_0_sub.shape[0]+df_0_discarded.shape[0]==df_0.shape[0])
            return return_df, df_0_discarded
        else:
            return return_df
            
            
    @staticmethod
    def print_data_composition(
        df, 
        is_outg_col = ('is_outg', 'is_outg'), 
        from_outg_col = ('from_outg', 'from_outg'), 
        headline_str = None
    ):
        r"""
        NOTE: df and headline_str can be lists for easy multiple outputs!
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(df, [pd.DataFrame, list]))
        if isinstance(df, list):
            assert(headline_str is None or isinstance(headline_str, list))
            if headline_str is None:
                headline_str = [None for _ in range(len(df))]
            assert(len(df)==len(headline_str))
            #-----
            for i in range(len(df)):
                if df[i] is None or df[i].shape[0]==0:
                    continue
                OutageModeler.print_data_composition(
                    df            = df[i], 
                    is_outg_col   = is_outg_col, 
                    from_outg_col = from_outg_col, 
                    headline_str  = headline_str[i]
                )
            return
        #--------------------------------------------------
        assert(is_outg_col   in df.columns.tolist())
        assert(from_outg_col in df.columns.tolist())
        #--------------------------------------------------
        n_outg_target_1 = df[is_outg_col].sum()
        n_outg_target_0 = df[
            (df[from_outg_col]==1) & 
            (df[is_outg_col]==0)
        ].shape[0]
        n_bsln = df[df[from_outg_col]==0].shape[0]
        assert(df.shape[0]==n_outg_target_1+n_outg_target_0+n_bsln)
        pct_target_1 = 100*n_outg_target_1/(n_outg_target_1+n_outg_target_0+n_bsln)
        #-----
        if headline_str is not None:
            print(f'\n----- {headline_str} -----')
        else:
            print('\n-----------------')
        #-----
        print(f"%(target==1):       {pct_target_1.round(2)}%")
        print(f"#(target = 1):      {n_outg_target_1}")
        print(f"#(target = 0):      {(n_outg_target_0+n_bsln)}")
        print(f"\tFrom Outage Dataset:   {n_outg_target_0}")
        print(f"\tFrom Baseline Dataset: {n_bsln}")


    @staticmethod
    def plot_target_counts(
        ax, 
        df, 
        is_outg_col = ('is_outg', 'is_outg'), 
        title = None
    ):
        r"""
        """
        #-------------------------
        sns.countplot(
            ax   = ax, 
            x    = is_outg_col, 
            data = df
        )
        if title is not None:
            ax.set_title(title);
        ax.set_xlabel('Is Outage');
        ax.set_ylim([1.05*x for x in ax.get_ylim()])
        for p in ax.patches:
            pct = (100*(p.get_height()/df.shape[0])).round(2)
            txt = f'{pct}%'
            txt_x = p.get_x() 
            txt_y = 1.025*p.get_height()
            ax.text(txt_x,txt_y,txt)
        #-------------------------
        return ax
    
    @staticmethod
    def plot_target_counts_full_train_test(
        df_full, 
        df_train, 
        df_test, 
        df_holdout = None, 
        is_outg_col = ('is_outg', 'is_outg')
    ):
        r"""
        """
        #-------------------------
        if df_holdout is None or df_holdout.shape[0]==0:
            fig, axs = Plot_General.default_subplots(1,3, return_flattened_axes=True)
        else:
            fig, axs = Plot_General.default_subplots(2,2, return_flattened_axes=True)
        #-------------------------
        assert(Utilities.is_object_one_of_types(df_full, [pd.DataFrame, pd.Series]))
        df_full_plt = df_full
        if isinstance(df_full, pd.Series):
            df_full_plt = df_full.to_frame()
        #-----
        assert(Utilities.is_object_one_of_types(df_full, [pd.DataFrame, pd.Series]))
        df_train_plt = df_train
        if isinstance(df_train, pd.Series):
            df_train_plt = df_train.to_frame()
        #-----
        assert(Utilities.is_object_one_of_types(df_full, [pd.DataFrame, pd.Series]))
        df_test_plt = df_test
        if isinstance(df_test, pd.Series):
            df_test_plt = df_test.to_frame()
        #-----
        assert(Utilities.is_object_one_of_types(df_full, [pd.DataFrame, pd.Series]))
        df_holdout_plt = df_holdout
        if df_holdout.shape[0]>0 and isinstance(df_holdout, pd.Series):
            df_holdout_plt = df_holdout.to_frame()
        #-----
    
        #-------------------------
        axs[0] = OutageModeler.plot_target_counts(
            ax = axs[0], 
            df = df_full_plt, 
            is_outg_col = is_outg_col, 
            title = 'Full Data'
        )
        #-----
        axs[1] = OutageModeler.plot_target_counts(
            ax = axs[1], 
            df = df_train_plt, 
            is_outg_col = is_outg_col, 
            title = 'Training Data'
        )
        #-----
        axs[2] = OutageModeler.plot_target_counts(
            ax = axs[2], 
            df =  df_test_plt, 
            is_outg_col = is_outg_col, 
            title = 'Testing Data'
        )
        #-----
        if df_holdout.shape[0]>0:
            axs[3] = OutageModeler.plot_target_counts(
                ax = axs[3], 
                df = df_holdout_plt, 
                is_outg_col = is_outg_col, 
                title = 'Holdout Data'
            )
        #-------------------------
        return fig,axs
        
        
    def plot_pca_explained_var(
        self
    ):
        r"""
        """
        #-------------------------
        fig,ax = Plot_General.default_subplots()
        pca = PCA()
        pca.fit(self.X_train)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(cumsum, ax=ax)
        #-------------------------
        return fig,ax
        
        
    def build_X_y(
        self, 
        verbose=True
    ):
        r"""
        Build X_train/_test/_holdout and y_train/_test/_holdout
        """
        #---------------------------------------------------------------------------------------------------
        include_holdout=False
        if self.df_holdout is not None and self.df_holdout.shape[0]>0:
            include_holdout = True
        #---------------------------------------------------------------------------------------------------
        assert(self.df_train.columns.tolist()[-1]==self.target_col)
        assert(self.df_test.columns.tolist()[-1]==self.target_col)
        if include_holdout:
            assert(self.df_holdout.columns.tolist()[-1]==self.target_col)
            
        #---------------------------------------------------------------------------------------------------
        # Encode EEMSP values
        if self.merge_eemsp:
            [
                _, 
                df_train, 
                df_test, 
                df_holdout
            ], self.__eemsp_enc = OutageModeler.encode_eemsp(
                df             = [
                    pd.concat([self.df_train.copy(), self.df_test.copy(), self.df_holdout.copy()]), 
                    self.df_train.copy(), 
                    self.df_test.copy(), 
                    self.df_holdout.copy()
                ], 
                eemsp_enc      = None, 
                eemsp_lvl_0_nm = 'EEMSP_0', 
                cols_to_encode = None, 
                numeric_cols   = ['kva_size', 'install_dt'], 
                run_transform  = True, 
                no_xfrm_df0    = True, 
            )  
        else:
            df_train, df_test, df_holdout = self.df_train.copy(), self.df_test.copy(), self.df_holdout.copy()
            
        #--------------------------------------------------------------------------------------------------- 
        X_train_OG = df_train.iloc[:, :-1].copy()
        y_train    = df_train.iloc[:, -1].copy()
        
        X_test_OG = df_test.iloc[:, :-1].copy()
        y_test    = df_test.iloc[:, -1].copy()
        
        if include_holdout:
            X_holdout = df_holdout.iloc[:, :-1]
            y_holdout = df_holdout.iloc[:, -1]
        else:
            X_holdout = pd.DataFrame()
            y_holdout = pd.DataFrame()
        #---------------------------------------------------------------------------------------------------
        if self.create_validation_set:
            if self.split_train_test_by_outg:
                X_train_OG, X_val_OG, y_train, y_val = Utilities_df.train_test_split_df_group(
                    X            = X_train_OG, 
                    y            = y_train, 
                    groups       = X_train_OG.index.get_level_values(self.mecpx_build_info_dict['rec_nb_idfr'][1]), 
                    test_size    = self.val_size, 
                    random_state = self.random_state
                )
            else:
                X_train_OG, X_val_OG, y_train, y_val = train_test_split(
                    X_train_OG, 
                    y_train, 
                    test_size    = self.val_size, 
                    random_state = self.random_state
                )
        #---------------------------------------------------------------------------------------------------
        if self.run_scaler:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train_OG)
            X_test  = scaler.transform(X_test_OG)
            #-----
            if self.create_validation_set:
                X_val = scaler.transform(X_val_OG)
            #-----
            if include_holdout:
                X_holdout = scaler.transform(X_holdout)
            #-------------------------
            self.__scaler = scaler
        else:
            X_train = X_train_OG
            X_test  = X_test_OG
            if self.create_validation_set:
                X_val = X_val_OG
    
        #---------------------------------------------------------------------------------------------------
        if self.run_PCA:
            pca = PCA(
                n_components = self.pca_n_components, 
                random_state = self.random_state
            )
            X_train = pca.fit_transform(X_train)
            X_test  = pca.transform(X_test)
            if self.create_validation_set:
                X_val = pca.transform(X_val)
            if include_holdout:
                X_holdout = pca.transform(X_holdout)
            if verbose:
                print(f'PCA n-components       = {pca.n_components_}')
                print(f'PCA explained variance = {pca.explained_variance_ratio_.sum()}')
            #-------------------------
            self.__pca = pca
    
        #---------------------------------------------------------------------------------------------------
        self.__X_train = X_train
        self.__y_train = y_train
        #-----
        self.__X_test = X_test
        self.__y_test = y_test
        #-----
        if include_holdout:
            self.__X_holdout = X_holdout
            self.__y_holdout = y_holdout
        else:
            self.__X_holdout = None
            self.__y_holdout = None
    
    
    def finalize_train_test_data(
        self, 
        verbose=True
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        self.__saved_data_structure_df     = False
        self.__saved_eemsp_enc             = False
        self.__saved_pca                   = False
        self.__saved_scaler                = False
        self.__saved_summary_dict          = False
        #----------------------------------------------------------------------------------------------------
        if verbose:
            print('In OutageModeler.finalize_train_test_data\nBEFORE ensure_target_val_1_min_pct')
            OutageModeler.print_data_composition(
                df            = [self.df_train, self.df_test, self.df_holdout], 
                headline_str  = ['TRAIN',              'TEST',              'HOLDOUT']
            )
        #---------------------------------------------------------------------------------------------------
        include_holdout=False
        if self.df_holdout is not None and self.df_holdout.shape[0]>0:
            include_holdout = True
        #---------------------------------------------------------------------------------------------------
        if self.df_train[self.target_col].sum()==0:
            print(f"Not enough target value==1 in train\n#target==1 train:   {self.df_train[self.target_col].sum()}")
            assert(0)
        #-----
        if self.df_test[self.target_col].sum()==0:
            print(f"Not enough target value==1 in test\n#target==1 test:    {self.df_test[self.target_col].sum()}")
            assert(0)
        #-----
        if include_holdout and self.df_holdout[self.target_col].sum()==0:
            print(f"Not enough target value==1 in holdout\n#target==1 holdout: {self.df_holdout[self.target_col].sum()}")
            assert(0)
        #---------------------------------------------------------------------------------------------------
        if self.min_pct_target_1 is not None:
            self.__df_train = OutageModeler.ensure_target_val_1_min_pct(
                df               = self.__df_train,
                min_pct          = self.min_pct_target_1,
                target_col       = self.target_col, 
                random_state     = self.random_state, 
                return_discarded = False
            )
            #-----
            self.__df_test = OutageModeler.ensure_target_val_1_min_pct(
                df               = self.__df_test,
                min_pct          = self.min_pct_target_1,
                target_col       = self.target_col, 
                random_state     = self.random_state, 
                return_discarded = False
            )
            #-----
            self.__df_holdout = OutageModeler.ensure_target_val_1_min_pct(
                df               = self.__df_holdout,
                min_pct          = self.min_pct_target_1,
                target_col       = self.target_col, 
                random_state     = self.random_state, 
                return_discarded = False
            )
        #---------------------------------------------------------------------------------------------------
        if verbose:
            print('\nAFTER')
            OutageModeler.print_data_composition(
                df            = [self.df_train, self.df_test, self.df_holdout], 
                headline_str  = ['TRAIN',              'TEST',              'HOLDOUT']
            )
        #---------------------------------------------------------------------------------------------------
        if self.reduce_train_size:
            if self.split_train_test_by_outg:
                self.__df_train, _ = OutageModeler.train_test_split_df_by_outage(
                    df               = self.__df_train, 
                    outg_rec_nb_idfr = self.mecpx_build_info_dict['rec_nb_idfr'], 
                    test_size        = self.red_test_size, 
                    random_state     = self.random_state
                )
            else:
                # Use sklearn method
                self.__df_train, _ = train_test_split(
                    self.__df_train, 
                    test_size    = self.red_test_size, 
                    random_state = self.random_state
                )
            #-------------------------
            if verbose:
                print('After reduce_train_size')
                OutageModeler.print_data_composition(
                    df            = [self.df_train, self.df_test, self.df_holdout], 
                    headline_str  = ['TRAIN',              'TEST',              'HOLDOUT']
                )
        #---------------------------------------------------------------------------------------------------
        # Remove self.from_outg_col  and make sure DFs end with self.target_col
        #-------------------------
        # Remove self.from_outg_col 
        if self.from_outg_col in self.df_train.columns.tolist():
            self.__df_train = self.__df_train.drop(columns=[self.from_outg_col])
        if self.from_outg_col in self.df_test.columns.tolist():
            self.__df_test = self.__df_test.drop(columns=[self.from_outg_col])
        #-------------------------
        # Make sure DFs end with self.target_col
        self.__df_train = Utilities_df.move_cols_to_back(df=self.__df_train, cols_to_move=[self.target_col])
        self.__df_test  = Utilities_df.move_cols_to_back(df=self.__df_test,  cols_to_move=[self.target_col])
        #-------------------------
        if include_holdout:
            if self.from_outg_col in self.df_holdout.columns.tolist():
                self.__df_holdout = self.__df_holdout.drop(columns=[self.from_outg_col])
            self.__df_holdout = Utilities_df.move_cols_to_back(df=self.__df_holdout, cols_to_move=[self.target_col])
        #---------------------------------------------------------------------------------------------------
        # Build X_train/_test/_holdout and y_train/_test/_holdout
        #-------------------------
        self.build_X_y(verbose=verbose)
        #---------------------------------------------------------------------------------------------------
        if self.__save_model:
            self.save_data_structure_df()
            self.save_eemsp_enc()
            self.save_pca()
            self.save_scaler()
            self.save_summary_dict()



    @staticmethod
    def get_dumb_cross_val_scores(
        X,
        y, 
        cv      = 3, 
        scoring = 'accuracy'
    ):
        r"""
        """
        #-------------------------
        dumb_clf = DumbClassifier()
        cross_val_scores = cross_val_score(
            dumb_clf, 
            X,
            y, 
            cv      = cv, 
            scoring = scoring
        )
        return cross_val_scores
        
    def print_dumb_cross_val_scores(
        self    , 
        cv      = 3, 
        scoring = 'accuracy'
    ):
        r"""
        """
        #-------------------------
        scores = OutageModeler.get_dumb_cross_val_scores(
            X       = self.X_train,
            y       = self.y_train, 
            cv      = cv, 
            scoring = scoring
        )
        #-------------------------
        print('Dumb cross val. scores:')
        print(scores)
        
        
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    def initialize_data(
        self, 
        verbose = True
    ):
        r"""
        Methods to build MECPOCollection objects and the original (v0) version of merged DFs
        """
        #-------------------------
        colls_built = self.build_or_load_mecpx_colls(verbose=verbose)
        self.perform_similarity_operations(verbose=verbose)
        self.perform_reduction_operations(verbose=verbose)
        #-----
        # If the collections had to be re-built, then merged_dfs should be re-compiled
        if colls_built:
            self.compile_merged_dfs_0()
        else:
            # This doesn't guarantee the collections will be loaded and not compiled, only that
            #   the program will attempt to load them (if fails, then compile)
            self.compile_or_load_merged_dfs_0()
    
    def refine_data(
        self, 
        verbose = True
    ):
        r"""
        Methods to refine merged DFs down to what we want
        """
        #-------------------------
        self.build_or_load_time_infos_df(verbose=verbose)
        self.reduce_merged_dfs_to_top_reasons()
        self.add_total_counts_features()
        self.append_nSNs_per_group()
        self.add_eemsp_features(verbose=verbose)
        self.finalize_merged_dfs()
        
    def min_memory_impact_load(
        self, 
        verbose=True
    ):
        r"""
        On first run, the MECPOCollection objects must be built, and the merged_df objects are grabbed from these.
        However, on subsequent runs, the MECPOCollection objects are not really needed, and it is useful to exclude them
          from loading because they can be quite large in (memory) size.
        The purpose of this function is to try to load only the absolutely necessary elements needed for running these
          subsequent generations.

        IMPORTANT:  Notice the _0 versions of merged_dfs are loaded, not the final versions!  This is by design.
                    As outlined in OutageModeler.finalize_merged_dfs, the final form or merged_dfs will ultimately depend on 
                      decisions such as include_month, merge_eemsp, remove_scheduled_outages, etc.
                    Therefore, we start from _0 versions and implement such parameters as desired by the current run!
        """
        #-------------------------
        try:
            self.load_merged_dfs(
                tag     = '0', 
                verbose = verbose
            )
            self.load_counts_series_from_pkls(
                tag     = '0', 
                verbose = verbose            
            )
            #-----
            self.refine_data(verbose=verbose)
            #-----
            return True
        except:
            pass
        #-------------------------
        return False
    
    def finalize_data(
        self, 
        assert_can_model  = True, 
        print_dumb_scores = True, 
        verbose           = True
    ):
        r"""
        Methods to refine merged DFs down to what we want
        """
        #-------------------------
        self.build_train_test_data(
            assert_can_model = assert_can_model, 
            verbose          = verbose
        )
        if not self.can_model:
            return
        #-------------------------
        self.finalize_train_test_data(
            verbose = verbose
        )
        #-------------------------
        if print_dumb_scores:
            self.print_dumb_cross_val_scores(
                cv      = 3, 
                scoring = 'accuracy'
            )    
    
    
    def compile_data(
        self, 
        assert_can_model  = True, 
        print_dumb_scores = True, 
        verbose           = True
    ):
        r"""
        Always try to easiest with minimum memory usage method first!
        """
        #--------------------------------------------------
        if not self.__force_build:
            data_loaded = self.min_memory_impact_load(verbose = verbose)
        else:
            data_loaded = False
        #-------------------------
        if not data_loaded:
            self.initialize_data(verbose = verbose)
            self.refine_data(verbose = verbose)
        #--------------------------------------------------
        self.finalize_data(
            assert_can_model  = assert_can_model, 
            print_dumb_scores = print_dumb_scores, 
            verbose           = verbose
        )
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
        
        
    @staticmethod
    def init_random_forest(
        n_estimators = 1000, 
        max_depth    = 25, 
        criterion    = 'gini',  #'gini' or 'entropy', 
        class_weight = None,   # None or 'balanced'
        n_jobs       = None, 
        random_state = None
    ):
        r"""
        """
        #-------------------------
        forest_clf = RandomForestClassifier(
            n_estimators = n_estimators, 
            max_depth    = max_depth, 
            criterion    = criterion, 
            class_weight = class_weight, 
            n_jobs       = n_jobs, 
            random_state = random_state
        )
        #-------------------------
        return forest_clf
        
    @staticmethod
    def print_apr(
        y, 
        y_pred, 
        name = None
    ):
        r"""
        """
        #-------------------------
        if name is not None:
            print('*****'*5)
            print(name)
        print('*****'*5)
        print(f"#(target==1): {y.sum()}")
        print(f"#(target==0): {y.shape[0]-y.sum()}")
        print(f"%(target==1): {100*(y.sum()/(y.shape[0]))}")
        print('-----'*5)
        print("ACCURACY  OF THE MODEL: ", accuracy_score(y, y_pred))
        print("PRECISION OF THE MODEL: ", precision_score(y, y_pred))
        print("RECALL    OF THE MODEL: ", recall_score(y, y_pred))
        print()
        
        
    def init_model_clf(
        self, 
        model_type = 'random_forest', 
        **model_kwargs
    ):
        r"""
        """
        #-------------------------
        accptbl_model_types = ['random_forest']
        assert(model_type in accptbl_model_types)
        #-------------------------
        model_kwargs['random_state'] = self.random_state
        #-------------------------
        if model_type=='random_forest':
            self.__model_clf = OutageModeler.init_random_forest(**model_kwargs)
            return
        #-------------------------
        assert(0)
    
    def fit(
        self, 
        verbose = True
    ):
        r"""
        """
        #-------------------------
        assert(self.can_model)
        #-------------------------
        self.__saved_model_clf = False
        #-------------------------
        start = time.time()
        self.model_clf.fit(self.X_train, self.y_train)
        fit_time = time.time()-start
        if verbose:
            print(f"Fit time = {fit_time}")
        #-------------------------
        self.__model_args_locked = True
        #-------------------------
        if self.__save_model:
            self.save_model_clf()
    
    def predict(
        self, 
        verbose = True
    ):
        r"""
        """
        #-------------------------
        assert(self.can_model)
        #-------------------------
        self.__y_train_pred = self.model_clf.predict(self.X_train)
        self.__y_test_pred  = self.model_clf.predict(self.X_test)
        #-----
        if verbose:
            OutageModeler.print_apr(y=self.y_train, y_pred=self.y_train_pred, name='TRAINING DATASET')
            OutageModeler.print_apr(y=self.y_test,  y_pred=self.y_test_pred, name='TESTING DATASET')
        #-------------------------
        if self.df_holdout is not None and self.df_holdout.shape[0]>0:
            self.__y_holdout_pred  = self.model_clf.predict(self.X_holdout)
            if verbose:
                OutageModeler.print_apr(y=self.y_holdout, y_pred=self.y_holdout_pred, name='HOLDOOUT DATASET')
                
                
    def slice_off_model(
        self, 
        new_gnrl_slicer, 
        save_model       = False, 
        new_save_sub_dir = None, 
    ):
        r"""
        Intent is so I can easily clone off copies of a head analysis to build more targetted analyses (but probably useful elsewhere too)
            e.g., build head Ohio analysis
                    --> slice off individual models (i.e., partitions of the data) so each zip code can have its own model
        """
        #-------------------------
        if save_model:
            assert(self.is_save_sub_dir_loaded())
            assert(new_save_sub_dir is not None)
        #-------------------------
        return_outg_mdlr = self.copy()
        #-----
        return_outg_mdlr.set_gnrl_slicer(new_gnrl_slicer)
        return_outg_mdlr.apply_gnrl_slicer()
        #-------------------------
        # Need to re-set certain attributes (e.g., return_outg_mdlr.save_sub_dir should be re-set, to avoid overwriting head analysis results)
        return_outg_mdlr.__save_model = save_model
        return_outg_mdlr.save_sub_dir = new_save_sub_dir
        if self.is_save_sub_dir_loaded():
            return_outg_mdlr.save_base_dir = os.path.join(self.save_dir, 'Slices')
        #-------------------------
        if return_outg_mdlr.__save_model and not os.path.exists(return_outg_mdlr.save_dir):
            os.makedirs(return_outg_mdlr.save_dir)
        #-------------------------
        return return_outg_mdlr
        
        
    def get_data_structure_df(
        self
    ):
        r"""
        """
        #-------------------------
        assert(self.df_train is not None and self.df_train.shape[0]>0)
        ds_df = self.df_train.drop(columns=[self.target_col]).head().copy()
        return ds_df
    
    def save_data_structure_df(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_data_structure_df:
            return
        #-------------------------
        assert(self.is_save_sub_dir_loaded())
        ds_df = self.get_data_structure_df()
        ds_df.to_pickle(os.path.join(self.save_dir, 'data_structure_df.pkl'))
        #-------------------------
        self.__saved_data_structure_df = True
        
    def save_scaler(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_scaler or not self.run_scaler:
            return
        #-------------------------
        assert(self.is_save_sub_dir_loaded())
        joblib.dump(self.scaler, os.path.join(self.save_dir, 'scaler.joblib'))
        #-------------------------
        self.__saved_scaler = True
        
    def save_pca(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_pca or not self.run_PCA:
            return
        #-------------------------
        assert(self.is_save_sub_dir_loaded())
        joblib.dump(self.pca, os.path.join(self.save_dir, 'pca.joblib'))
        #-------------------------
        self.__saved_pca = True
        
    def save_eemsp_enc(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_eemsp_enc or not self.merge_eemsp:
            return
        #-------------------------
        assert(self.is_save_sub_dir_loaded())
        joblib.dump(self.__eemsp_enc, os.path.join(self.save_dir, 'eemsp_encoder.joblib'))
        #-------------------------
        self.__saved_eemsp_enc = True
        
    def save_model_clf(
        self
    ):
        r"""
        """
        #-------------------------
        if self.__saved_model_clf:
            return
        #-------------------------
        assert(self.is_save_sub_dir_loaded())
        assert(self.model_clf is not None)
        joblib.dump(self.model_clf, os.path.join(self.save_dir, 'model_clf.joblib'))
        #-------------------------
        self.__saved_model_clf = True
        
    def save_model(
        self, 
        mkdir_if_dne = True
    ):
        r"""
        To be used, e.g., if self.__save_model was set to False, but after running the model the user
          decides they want to save it.
        If self.__save_model if True, the elements are saved along the way as they are built.
        """
        #-------------------------
        if mkdir_if_dne:
            self.make_sub_dir()
        #-------------------------
        self.save_data_structure_df()
        self.save_summary_dict()
        self.save_scaler()
        self.save_pca()
        self.save_eemsp_enc()
        self.save_model_clf()
        
        
