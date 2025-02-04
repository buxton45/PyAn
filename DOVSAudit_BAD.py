#!/usr/bin/env python

r"""
Holds AMINonVee class.  See AMINonVee.AMINonVee for more information.
"""

__author__ = "Jesse Buxton"
__email__ = "jbuxton@aep.com"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import re
import pickle

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
from scipy import stats
import datetime
import time
from natsort import natsorted, ns, natsort_keygen
from packaging import version

import copy
import itertools
import adjustText

import pyodbc

from sklearn.cluster import DBSCAN
from sklearn import metrics
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from MeterPremise import MeterPremise
#-----
from AMI_SQL import AMI_SQL
from AMINonVee_SQL import AMINonVee_SQL
from AMIEndEvents_SQL import AMIEndEvents_SQL
from AMIUsgInst_SQL import AMIUsgInst_SQL
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from GenAn import GenAn
from AMINonVee import AMINonVee
from AMIEndEvents import AMIEndEvents
from AMIUsgInst import AMIUsgInst
from DOVSOutages import DOVSOutages
#---------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib import dates
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
import Utilities_dt
from Utilities_df import DFConstructType
import Plot_General
import DataFrameSubsetSlicer
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
#---------------------------------------------------------------------

class DOVSAudit:
    r"""
    Class to adjust DOVS outage times and CI/CMI
    """
    def __init__(
        self, 
        outg_rec_nb, 
        init_dfs_to = pd.DataFrame() # Should be pd.DataFrame() or None
    ):
        r"""
        
        """
        #-------------------------
        self.outg_rec_nb = outg_rec_nb
        #-------------------------
        self.calculate_by_PN = True
        self.combine_by_PN_likeness_thresh = pd.Timedelta('15 minutes')

        self.expand_outg_search_time_tight = pd.Timedelta('1 hours')
        self.expand_outg_search_time_loose = pd.Timedelta('12 hours')
        self.use_est_outg_times=False
        self.use_full_ede_outgs=False
        self.run_outg_inclusion_assessment=True
        self.max_pct_PNs_missing_allowed=0
        #-----
        self.expand_outg_est_search_time = self.expand_outg_search_time_loose
        if self.use_est_outg_times:
            self.expand_outg_search_time = self.expand_outg_search_time_tight
        else:
            self.expand_outg_search_time = self.expand_outg_search_time_loose
        
        #-------------------------
        # When using self.init_dfs_to ALWAYS use copy.deepcopy(self.init_dfs_to),
        # otherwise, unintended consequences could return (self.init_dfs_to is None it
        # doesn't really matter, but when it is pd.DataFrame it does!)
        self.init_dfs_to = init_dfs_to
        
        self.ami_df_i  = copy.deepcopy(self.init_dfs_to)
        self.ede_df_i  = copy.deepcopy(self.init_dfs_to)
        self.dovs_df_i = copy.deepcopy(self.init_dfs_to)
        self.mp_df_i   = copy.deepcopy(self.init_dfs_to)
        
        # In order to run analysis, ami_df_i and dovs_df_i are absolutely necessary.
        # It is nice to have ede_df_i, but the analysis can be run without it.
        self.__is_loaded_ami  = False
        self.__is_loaded_ede  = False
        self.__is_loaded_dovs = False
        self.__is_loaded_mp   = False
        
        self.ami_df_info_dict  = DOVSAudit.get_full_ami_df_info_dict()
        self.dovs_df_info_dict = DOVSAudit.get_full_dovs_df_info_dict()
        self.ede_df_info_dict  = DOVSAudit.get_full_ede_df_info_dict()
        self.mp_df_info_dict   = DOVSAudit.get_full_mp_df_info_dict()
        
        #-------------------------
        self.best_ests_df             = copy.deepcopy(self.init_dfs_to)
        self.best_ests_df_w_keep_info = copy.deepcopy(self.init_dfs_to)
        self.overlap_outgs_for_PNs_df = copy.deepcopy(self.init_dfs_to)
        
        #-------------------------
        self.ci  = -1
        self.cmi = -1
        
        #-------------------------
        self.warnings_flag = False
        self.warnings_dict = None
        self.best_est_df_entries_w_warning = copy.deepcopy(self.init_dfs_to)

    #----------------------------------------------------------------------------------------------------
    @property
    def is_loaded_ami(self):
        return self.__is_loaded_ami
    @property
    def is_loaded_ede(self):
        return self.__is_loaded_ede
    @property
    def is_loaded_dovs(self):
        return self.__is_loaded_dovs
    @property
    def is_loaded_mp(self):
        return self.__is_loaded_mp
    
    @property
    def n_SNs(self):
        return self.ami_df_i[self.ami_df_info_dict[SN_col]].nunique()
    
    @property
    def n_PNs(self):
        return self.ami_df_i[self.ami_df_info_dict[PN_col]].nunique()
    
    @property
    def dovs_outg_t_beg_end(self):
        r"""
        Get the outage time from DOVS
        """
        #-------------------------
        outg_t_beg_col = self.dovs_df_info_dict['outg_t_beg_col']
        outg_t_end_col = self.dovs_df_info_dict['outg_t_end_col']
        dovs_outg_t_beg_end = self.dovs_df_i.iloc[0][[outg_t_beg_col, outg_t_end_col]].tolist()
        assert(len(dovs_outg_t_beg_end)==2)
        return dovs_outg_t_beg_end
    
    @property
    def ci_cmi_dovs(self):
        r"""
        Get the CI and CMI from DOVS
        """
        #-------------------------
        ci_nb_col  = self.dovs_df_info_dict['ci_nb_col']
        cmi_nb_col = self.dovs_df_info_dict['cmi_nb_col']
        ci_cmi_dovs = self.dovs_df_i.iloc[0][[ci_nb_col, cmi_nb_col]].tolist()
        assert(len(ci_cmi_dovs)==2)
        return ci_cmi_dovs
    
    @property
    def PNs_dovs(self):
        r"""
        Get the number of premises from DOVS
        """
        #-------------------------
        PNs_col = self.dovs_df_info_dict['PNs_col']
        n_PNs_dovs = list(set(self.dovs_df_i.iloc[0][PNs_col]))
        return n_PNs_dovs
    @property
    def n_PNs_dovs(self):
        r"""
        Get the number of premises from DOVS
        """
        #-------------------------
        PNs_col = self.dovs_df_info_dict['PNs_col']
        n_PNs_dovs = len(set(self.dovs_df_i.iloc[0][PNs_col]))
        return n_PNs_dovs
        
    @property
    def outage_nb(self):
        r"""
        Get the outage number from DOVS
        """
        #-------------------------
        outage_nb_col = self.dovs_df_info_dict['outage_nb_col']
        outage_nb = self.dovs_df_i.iloc[0][outage_nb_col]
        return outage_nb
        
    #--------------------------------------------------    
    def get_cnsrvtv_out_t_beg_end(
        self, 
        t_min_col='conservative_min', 
        t_max_col='conservative_max', 
        keepers_only=True
    ):
        r"""
        """
        #-------------------------
        dovs_outg_t_beg_end = self.dovs_outg_t_beg_end
        #-------------------------
        if keepers_only:
            best_ests_df = self.best_ests_df
        else:
            if self.best_ests_df_w_keep_info is not None and self.best_ests_df_w_keep_info.shape[0]>0:
                best_ests_df = self.best_ests_df_w_keep_info
            else:
                best_ests_df = self.best_ests_df
        #-------------------------
        if best_ests_df.shape[0]>0:
            cnsrvtv_out_t_beg = np.min([best_ests_df[t_min_col].min(), dovs_outg_t_beg_end[0]])
            cnsrvtv_out_t_end = np.max([best_ests_df[t_max_col].max(), dovs_outg_t_beg_end[1]])
            #-----
            cnsrvtv_out_t_beg_end = [cnsrvtv_out_t_beg, cnsrvtv_out_t_end]
        else:
            cnsrvtv_out_t_beg_end = dovs_outg_t_beg_end       
        #-------------------------
        return cnsrvtv_out_t_beg_end

    @property
    def cnsrvtv_out_t_beg_end(self):
        return self.get_cnsrvtv_out_t_beg_end()
          
          
    def are_all_meters_AMI(
        self
    ):
        r"""
        """
        #-------------------------
        assert(self.__is_loaded_mp)
        return (self.mp_df_i[self.mp_df_info_dict['technology_tx_col']]=='AMI').all()
        
    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_full_ami_df_info_dict(
        ami_df_info_dict=None
    ):
        r"""
        NOTE: Can exclude an item from check_..._info_dict by setting value to None
        """
        #--------------------------------------------------
        dflt_ami_df_info_dict = dict(
            SN_col                   = 'serialnumber', 
            PN_col                   = 'aep_premise_nb', 
            t_int_beg_col_raw        = 'starttimeperiod', 
            t_int_beg_col            = 'starttimeperiod_local', 
            t_int_end_col_raw        = 'endtimeperiod', 
            t_int_end_col            = 'endtimeperiod_local', 
            value_col                = 'value', 
            aep_derived_uom_col      = 'aep_derived_uom', 
            aep_srvc_qlty_idntfr_col = 'aep_srvc_qlty_idntfr', 
            # outg_ columns will typically not be in ami_df_i, but can be
            outg_rec_nb_idfr         = 'OUTG_REC_NB_GPD_FOR_SQL', 
            outg_t_beg_col           = 'DT_OFF_TS_FULL', 
            outg_t_end_col           = 'DT_ON_TS', 
            trsf_pole_nb_col         = 'trsf_pole_nb_GPD_FOR_SQL', 
            op_unit_id_col           = None, 
            technology_tx_col        = None
        )
        #-----
        if ami_df_info_dict is None:
            ami_df_info_dict = dict()
        #-----
        assert(isinstance(ami_df_info_dict, dict))
        ami_df_info_dict = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = ami_df_info_dict, 
            default_values_dict = dflt_ami_df_info_dict, 
            extend_any_lists    = False, 
            inplace             = False
        )
        #--------------------------------------------------
        return ami_df_info_dict  
    
    @staticmethod
    def check_ami_df_info_dict(
        ami_df, 
        ami_df_info_dict
    ):
        r"""
        """
        #-------------------------
        for k,v in ami_df_info_dict.items():
            if v is None:
                continue
            if k.find('_col')>-1 and k.find('outg_')==-1:
                if v not in ami_df.columns.tolist():
                    print(f"{v} not in {ami_df.columns.tolist()}\nCRASH IMMINENT!!!!!")
                    assert(0)
                

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_full_ede_df_info_dict(
        ede_df_info_dict=None
    ):
        r"""
        NOTE: Can exclude an item from check_..._info_dict by setting value to None
        """
        #--------------------------------------------------
        dflt_ede_df_info_dict = dict(
            outg_rec_nb_idfr = 'OUTG_REC_NB_GPD_FOR_SQL', 
            SN_col           = 'serialnumber', 
            PN_col           = 'aep_premise_nb', 
            time_col_raw     = 'valuesinterval', 
            time_col         = 'valuesinterval_local', 
            reason_col       = 'reason', 
            ede_typeid_col   = 'enddeviceeventtypeid', 
            event_type_col   = 'event_type', 
            trsf_pole_nb_col = 'trsf_pole_nb_GPD_FOR_SQL',
        )
        #-----
        if ede_df_info_dict is None:
            ede_df_info_dict = dict()
        #-----
        assert(isinstance(ede_df_info_dict, dict))
        ede_df_info_dict = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = ede_df_info_dict, 
            default_values_dict = dflt_ede_df_info_dict, 
            extend_any_lists    = False, 
            inplace             = False
        )
        #--------------------------------------------------
        return ede_df_info_dict  
    
    @staticmethod
    def check_ede_df_info_dict(
        ede_df, 
        ede_df_info_dict
    ):
        r"""
        """
        #-------------------------
        for k,v in ede_df_info_dict.items():
            if v is None:
                continue
            if k.find('_col')>-1 and k.find('_raw')==-1:
                if v not in ede_df.columns.tolist():
                    print(f"{v} not in {ede_df.columns.tolist()}\nCRASH IMMINENT!!!!!")
                    assert(0)
                
    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_full_dovs_df_info_dict(
        dovs_df_info_dict=None
    ):
        r"""
        NOTE: Can exclude an item from check_..._info_dict by setting value to None
        """
        #--------------------------------------------------
        dflt_dovs_df_info_dict = dict(
            is_consolidated  = True,
            outg_rec_nb_idfr = 'index', 
            PN_col           = 'PREMISE_NB', 
            PNs_col          = 'premise_nbs', 
            outg_t_beg_col   = 'DT_OFF_TS_FULL', 
            outg_t_end_col   = 'DT_ON_TS', 
            ci_nb_col        = 'CI_NB', 
            cmi_nb_col       = 'CMI_NB', 
            outage_nb_col    = 'OUTAGE_NB'
        )
        #-----
        if dovs_df_info_dict is None:
            dovs_df_info_dict = dict()
        #-----
        assert(isinstance(dovs_df_info_dict, dict))
        dovs_df_info_dict = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = dovs_df_info_dict, 
            default_values_dict = dflt_dovs_df_info_dict, 
            extend_any_lists    = False, 
            inplace             = False
        )
        #--------------------------------------------------
        return dovs_df_info_dict  
    
    @staticmethod
    def check_dovs_df_info_dict(
        dovs_df, 
        dovs_df_info_dict
    ):
        r"""
        In this case, dovs_df can be pd.DataFrame or pd.Series
        """
        #-------------------------
        # Allow dovs_df to be frame or series
        assert(Utilities.is_object_one_of_types(dovs_df, [pd.DataFrame, pd.Series]))
        is_frame = isinstance(dovs_df, pd.DataFrame)
        #-------------------------
        is_consolidated = dovs_df_info_dict['is_consolidated']
        for k,v in dovs_df_info_dict.items():
            if v is None:
                continue
            if k.find('_col')>-1:
                if is_consolidated and k=='PN_col':
                    continue
                if not is_consolidated and k=='PNs_col':
                    continue
                #-----
                if is_frame:
                    assert(v in dovs_df.columns.tolist())
                else:
                    assert(v in dovs_df.index.tolist())
                    
                    
    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_full_mp_df_info_dict(
        mp_df_info_dict=None
    ):
        r"""
        NOTE: Can exclude an item from check_..._info_dict by setting value to None
        """
        #--------------------------------------------------
        dflt_mp_df_info_dict = dict(
            SN_col            = 'mfr_devc_ser_nbr', 
            PN_col            = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_idfr  = 'OUTG_REC_NB', 
            inst_ts_col       = 'inst_ts', 
            rmvl_ts_col       = 'rmvl_ts', 
            technology_tx_col = 'technology_tx'
        )
        #-----
        if mp_df_info_dict is None:
            mp_df_info_dict = dict()
        #-----
        assert(isinstance(mp_df_info_dict, dict))
        mp_df_info_dict = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = mp_df_info_dict, 
            default_values_dict = dflt_mp_df_info_dict, 
            extend_any_lists    = False, 
            inplace             = False
        )
        #--------------------------------------------------
        return mp_df_info_dict  
        
    @staticmethod
    def check_mp_df_info_dict(
        mp_df, 
        mp_df_info_dict
    ):
        r"""
        """
        #-------------------------
        for k,v in mp_df_info_dict.items():
            if v is None:
                continue
            if k.find('_col')>-1:
                if v not in mp_df.columns.tolist():
                    print(f"{v} not in {mp_df.columns.tolist()}\nCRASH IMMINENT!!!!!")
                    assert(0)
                    
                                 
    #****************************************************************************************************
    # Methods for finding the correct data file(s) housing the outage
    # These are used, e.g., when a large data acquisition run is done before analyzing the data
    #****************************************************************************************************
    @staticmethod
    def invert_file_to_outg_rec_nbs_dict(file_to_outg_rec_nbs_dict):
        r"""
        Input:
            keys   = file paths
            values = outg_rec_nbs contained in given file path

        Output:
            keys   = outg_rec_nbs
            values = files containing given outg_rec_nb
        """
        #-------------------------
        outg_rec_nb_to_files_dict = dict()
        for file_i, outg_rec_nbs_i in file_to_outg_rec_nbs_dict.items():
            for outg_rec_nb_ij in outg_rec_nbs_i:
                if outg_rec_nb_ij in outg_rec_nb_to_files_dict.keys():
                    outg_rec_nb_to_files_dict[outg_rec_nb_ij].append(file_i)
                else:
                    outg_rec_nb_to_files_dict[outg_rec_nb_ij] = [file_i]
        #-------------------------
        return outg_rec_nb_to_files_dict


    @staticmethod
    def build_outg_rec_nb_to_files_dict(
        base_dir, 
        glob_pattern=r'ami_nonvee_[0-9]*.csv', 
        regex_pattern=None,
        outg_rec_nb_col='OUTG_REC_NB_GPD_FOR_SQL', 
        save_path=None
    ):
        r"""
        Build a dictionary whose keys are outg_rec_nbs and values are the files in which each outg_rec_nb
          is contained (note, a given outg_rec_nb may be split across multiple files)

        glob_pattern:
            For AMI Nonvee, typical value: glob_pattern=r'ami_nonvee_[0-9]*.csv'
            For end events, typical value: glob_pattern=r'end_events_[0-9]*.csv'
        """
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir      = base_dir, 
            glob_pattern  = glob_pattern, 
            regex_pattern = regex_pattern
        )
        paths=natsorted(paths)
        assert(len(paths)>0)
        #-------------------------
        outg_rec_nbs_in_files = dict()
        for path in paths:
            assert(path not in outg_rec_nbs_in_files.keys())
            df = GenAn.read_df_from_csv(path)
            outg_rec_nbs_in_files[path] = df[outg_rec_nb_col].unique().tolist()
        outg_rec_nb_to_files_dict = DOVSAudit.invert_file_to_outg_rec_nbs_dict(outg_rec_nbs_in_files)
        #-------------------------
        if save_path is not None:
            with open(save_path, 'wb') as handle:
                pickle.dump(outg_rec_nb_to_files_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #-------------------------
        return outg_rec_nb_to_files_dict


    @staticmethod
    def find_outg_rec_nb_in_outg_rec_nbs_in_files_dict(
        outg_rec_nb, 
        outg_rec_nbs_in_files,
        crash_if_not_found=True
    ):
        #-------------------------
        for file_i, outg_rec_nbs_i in outg_rec_nbs_in_files.items():
            if outg_rec_nb in outg_rec_nbs_i:
                return file_i
        print(f'outg_rec_nb = {outg_rec_nb} not found!')
        if crash_if_not_found:
            assert(0)
        else:
            return None


    #****************************************************************************************************
    # Methods to load and initialize DOVS data
    #****************************************************************************************************        
    def load_dovs(
        self,
        dovs_df=None, 
        dovs_df_info_dict=None
    ):
        r"""
        Load/set self.dovs_df for self.outg_rec_nb.
        Note, the output format will be a single-row pd.DataFrame for the outage.
        dovs_df:
            If supplied, grab the appropriate values
            If not supplied, run SQL query to obtain needed values
        """
        #-------------------------
        dovs_df_info_dict = DOVSAudit.get_full_dovs_df_info_dict(dovs_df_info_dict=dovs_df_info_dict)
        #-------------------------
        dovs_df_i = pd.DataFrame()
        #-----
        if dovs_df is not None:
            dovs_df_i = DOVSOutages.retrieve_outage_from_dovs_df(
                dovs_df                  = dovs_df, 
                outg_rec_nb              = self.outg_rec_nb, 
                outg_rec_nb_idfr         = dovs_df_info_dict['outg_rec_nb_idfr'], 
                assert_outg_rec_nb_found = False
            )
        #-------------------------
        # If dovs_df_i is empty, run SQL query to obtain needed results
        # Note, dovs_df_i can be empty at this point for two reasons:
        #   1. dovs_df was not supplied
        #   2. dovs_df was supplied, but self.outg_rec_nb was not found in it.
        if dovs_df_i.shape[0]==0:
            dovs = DOVSOutages(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                contstruct_df_args=None, 
                init_df_in_constructor=True,
                build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                build_sql_function_kwargs=dict(
                    outg_rec_nbs=[self.outg_rec_nb], 
                    include_DOVS_PREMISE_DIM=True, 
                    include_DOVS_MASTER_GEO_DIM=True, 
                    include_DOVS_OUTAGE_ATTRIBUTES_DIM=True, 
                    include_DOVS_CLEARING_DEVICE_DIM=True, 
                    include_DOVS_EQUIPMENT_TYPES_DIM=True, 
                    include_DOVS_OUTAGE_CAUSE_TYPES_DIM=True
                ), 
                build_consolidated=True
            )
            dovs_df_i = dovs.df.copy()

        #-------------------------
        # The dovs_df_i which is to be returned should only have a single row
        # If this is not the case, run DOVSOutages.consolidate_df_outage
        # NOTE: This could only ever occur if the user supplies dovs_df which is not consolidated (but does
        #         contain self.outg_rec_nb)
        if dovs_df_i.shape[0]>1:
            outg_rec_nb_idfr_loc = Utilities_df.get_idfr_loc(
                df   = dovs_df_i, 
                idfr = dovs_df_info_dict['outg_rec_nb_idfr']

            )
            #-------------------------
            # If the outg_rec_nbs are in the index, then reset_index must be called for
            #   consolidate_df_outage to run properly
            if outg_rec_nb_idfr_loc[1]:
                outg_rec_nb_idx_lvl = outg_rec_nb_idfr_loc[0]
                #-----
                if dovs_df_i.index.names[outg_rec_nb_idx_lvl]:
                    outg_rec_nb_col = dovs_df_i.index.names[outg_rec_nb_idx_lvl]
                else:
                    outg_rec_nb_col = 'OUTG_REC_NB_'+Utilities.generate_random_string(str_len=4)
                    assert(outg_rec_nb_col not in dovs_df_i.columns.tolist())
                    assert(outg_rec_nb_col not in list(dovs_df_i.index.names))
                    dovs_df_i.index = dovs_df_i.index.set_names(outg_rec_nb_col, level=outg_rec_nb_idx_lvl)
                #-----
                # Set the outg_rec_nb_col and drop index
                dovs_df_i[outg_rec_nb_col] = dovs_df_i.index.get_level_values(outg_rec_nb_idx_lvl)
                if dovs_df_i.index.nlevels==1:
                    # NOTE: Values already placed in outg_rec_nb_col above, hence why drop=True below
                    dovs_df_i = dovs_df_i.reset_index(drop=True)
                else:
                    dovs_df_i = dovs_df_i.droplevel(outg_rec_nb_idx_lvl, axis=0)
                #-----
                assert(outg_rec_nb_col in dovs_df_i.columns.tolist())
            else:
                outg_rec_nb_col = outg_rec_nb_idfr_loc[0]
            #-------------------------
            # OFF_TM and REST_TM are premise specific, and should not be included
            cols_to_drop = ['OFF_TM', 'REST_TM']
            cols_to_drop = list(set(cols_to_drop).intersection(set(dovs_df_i.columns.tolist())))
            #-----
            # PN_col must be in dovs_df_i
            assert(dovs_df_info_dict['PN_col'] in dovs_df_i.columns.tolist())
            #-----
            # All columns exepct premise numbers (and cols_to_drop) should be shared by groups
            cols_shared_by_group = [x for x in dovs_df_i.columns.tolist() 
                                    if x not in [outg_rec_nb_col, dovs_df_info_dict['PN_col']]+cols_to_drop]
            #-------------------------
            dovs_df_i = DOVSOutages.consolidate_df_outage(
                df_outage                    = dovs_df_i, 
                outg_rec_nb_col              = outg_rec_nb_col, 
                addtnl_grpby_cols            = None, 
                cols_shared_by_group         = cols_shared_by_group, 
                cols_to_collect_in_lists     = [dovs_df_info_dict['PN_col']], 
                allow_duplicates_in_lists    = False, 
                allow_NaNs_in_lists          = False, 
                recover_uniqueness_violators = True, 
                gpby_dropna                  = False, 
                rename_cols                  = None,     
                premise_nb_col               = dovs_df_info_dict['PN_col'], 
                premise_nbs_col              = dovs_df_info_dict['PNs_col'], 
                cols_to_drop                 = cols_to_drop, 
                sort_PNs                     = True, 
                drop_null_premise_nbs        = True, 
                set_outg_rec_nb_as_index     = True,
                drop_outg_rec_nb_if_index    = True, 
                verbose                      = False
            )

        #-------------------------
        # The dovs_df_i which is to be returned MUST only have a single row
        assert(dovs_df_i.shape[0]==1)
        #-------------------------
        self.dovs_df_i         = dovs_df_i
        self.dovs_df_info_dict = dovs_df_info_dict
        self.__is_loaded_dovs  = True
        
        
    #****************************************************************************************************
    # Methods to load and initialize MeterPremise data
    # NOTE: This is not strictly necessary (e.g., if one supplies the AMI and EDE data).
    #       However, if one only supplies the outg_rec_nb, and the AMI/EDE data need to be built using
    #         SQL queries, then MP will be necessary and this methods will be used
    #****************************************************************************************************   
    def build_mp_df(
        self, 
        drop_mp_dups_fuzziness=pd.Timedelta('1 hour'), 
        addtnl_mp_df_curr_cols=['technology_tx'], 
        addtnl_mp_df_hist_cols=['technology_tx'], 
        assert_all_PNs_found=True, 
        consolidate_PNs_batch_size=1000, 
        early_return=False
    ):
        r"""
        """
        #-------------------------
        # Typically use technology_tx to determine whether or not meter is AMI, which is important for this analysis
        # Therefore, make sure technology_tx is in both addtnl_mp_df_curr_cols and addtnl_mp_df_hist_cols
        if addtnl_mp_df_curr_cols is None:
            addtnl_mp_df_curr_cols = []
        if addtnl_mp_df_hist_cols is None:
            addtnl_mp_df_hist_cols = []
        #-----
        assert(isinstance(addtnl_mp_df_curr_cols, list) and isinstance(addtnl_mp_df_hist_cols, list))
        addtnl_mp_df_curr_cols = list(set(addtnl_mp_df_curr_cols + ['technology_tx']))
        addtnl_mp_df_hist_cols = list(set(addtnl_mp_df_hist_cols + ['technology_tx']))
        #-------------------------
        dovs_df_i, outg_rec_nb_col = DOVSOutages.get_outg_rec_nb_col_from_idfr(
            dovs_df          = self.dovs_df_i.copy(), 
            outg_rec_nb_idfr = self.dovs_df_info_dict['outg_rec_nb_idfr']
        )
        #-------------------------
        prem_nb_col = (self.dovs_df_info_dict['PNs_col'] if self.dovs_df_info_dict['is_consolidated'] else 
                       self.dovs_df_info_dict['PN_col'])
        #-------------------------
        df_mp_outg_OG = DOVSOutages.build_active_MP_for_outages_df(
            df_outage                  = dovs_df_i, 
            prem_nb_col                = prem_nb_col, 
            df_mp_curr                 = None, 
            df_mp_hist                 = None, 
            assert_all_PNs_found       = assert_all_PNs_found, 
            drop_inst_rmvl_cols        = False, 
            outg_rec_nb_idfr           = outg_rec_nb_col, 
            is_slim                    = self.dovs_df_info_dict['is_consolidated'], 
            addtnl_mp_df_curr_cols     = addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols     = addtnl_mp_df_hist_cols, 
            dt_on_ts_col               = self.dovs_df_info_dict['outg_t_end_col'], 
            df_off_ts_full_col         = self.dovs_df_info_dict['outg_t_beg_col'], 
            consolidate_PNs_batch_size = consolidate_PNs_batch_size, 
            early_return               = early_return
        )
        #-------------------------
        # Make sure inst_ts_col and rmvl_ts_col are both datetime types
        inst_ts_col = 'inst_ts'
        rmvl_ts_col = 'rmvl_ts'
        #-----
        df_mp_outg_OG[inst_ts_col] = pd.to_datetime(df_mp_outg_OG[inst_ts_col])
        df_mp_outg_OG[rmvl_ts_col] = pd.to_datetime(df_mp_outg_OG[rmvl_ts_col])
        #-------------------------
        addtnl_groupby_cols = [outg_rec_nb_col]
        if addtnl_mp_df_curr_cols is not None and addtnl_mp_df_hist_cols is not None:
                addtnl_groupby_cols.extend(
                    list(set(addtnl_mp_df_curr_cols).intersection(set(addtnl_mp_df_hist_cols)))
                )
        #-----
        # Remove any duplicates in df_mp_outg_OG
        df_mp_outg = MeterPremise.drop_approx_mp_duplicates(
            mp_df                 = df_mp_outg_OG.copy(), 
            fuzziness             = drop_mp_dups_fuzziness, 
            assert_single_overlap = True, 
            addtnl_groupby_cols   = addtnl_groupby_cols, 
            gpby_dropna           = False
        )
        #-------------------------
        # Really only want one entry per meter (here, meter being a mfr_devc_ser_nbr/prem_nb combination)
        # ALthough drop_duplicates was used, multiple entries could still exist if, e.g., a meter has two
        #   non-fuzzy-overlapping intervals
        assert(all(df_mp_outg[['mfr_devc_ser_nbr', 'prem_nb', outg_rec_nb_col]].value_counts()==1))
        #-------------------------
        DOVSAudit.check_mp_df_info_dict(
            mp_df           = df_mp_outg, 
            mp_df_info_dict = self.mp_df_info_dict
        )
        #-------------------------
        self.mp_df_i        = df_mp_outg
        self.__is_loaded_mp = True
        
        
    def get_merged_dovs_and_mp(
        self
    ):
        r"""
        The non-consolidated version of dovs_df_i will be merged with mp_df_i
        """
        #-------------------------
        assert(self.__is_loaded_dovs)
        assert(self.__is_loaded_mp)
        #-------------------------
        # Life much easier if outg_rec_nb info and premise info both in columns instead of in index
        # Calling DOVSOutage.get_outg_rec_nb_col_from_idfr will accomplish this
        dovs_df, outg_rec_nb_col_dovs = DOVSOutages.get_outg_rec_nb_col_from_idfr(
            dovs_df          = self.dovs_df_i.copy(), 
            outg_rec_nb_idfr = self.dovs_df_info_dict['outg_rec_nb_idfr']
        )
        #-----
        mp_df, outg_rec_nb_col_mp = DOVSOutages.get_outg_rec_nb_col_from_idfr(
            dovs_df          = self.mp_df_i.copy(), 
            outg_rec_nb_idfr = self.mp_df_info_dict['outg_rec_nb_idfr']
        )
        #-------------------------
        # If dovs_df is consolidate, call explode
        if self.dovs_df_info_dict['is_consolidated']:
            dovs_df = dovs_df.explode(column=self.dovs_df_info_dict['PNs_col'])
            dovs_df = dovs_df.rename(columns={
                self.dovs_df_info_dict['PNs_col']:self.dovs_df_info_dict['PN_col']
            })
        #-------------------------
        dovs_w_mp_df = DOVSOutages.merge_df_outage_with_mp(
            df_outage          = dovs_df, 
            df_mp              = mp_df, 
            merge_on_outg      = [outg_rec_nb_col_dovs, self.dovs_df_info_dict['PN_col']], 
            merge_on_mp        = [outg_rec_nb_col_mp,   self.mp_df_info_dict['PN_col']], 
            cols_to_include_mp = None, 
            drop_cols          = None, 
            rename_cols        = None, 
            inplace            = True
        )
        #-------------------------
        return dovs_w_mp_df
        
    @staticmethod
    def merge_df_with_mp(
        df, 
        mp_df, 
        df_info_dict, 
        mp_df_info_dict, 
        merge_on=None, 
        merge_on_mp=None, 
        how='left'
    ):
        r"""
        Default is to merge on SN_col and PN_col
        If merge_on/merge_on_mp are not supplied, df_info_dict/mp_df_info_dict MUST be supplied
        """
        #-------------------------
        df=df.copy()
        #-------------------------
        if merge_on is None and merge_on_mp is None:
            assert(df_info_dict is not None and mp_df_info_dict is not None)
            merge_on = [
                df_info_dict['SN_col'], 
                df_info_dict['PN_col']
            ]
            merge_on_mp = [
                mp_df_info_dict['SN_col'], 
                mp_df_info_dict['PN_col']
            ]
        else:
            assert(isinstance(merge_on, list))
            assert(isinstance(merge_on_mp, list))
            assert(len(merge_on)==len(merge_on_mp))
        #-------------------------
        df, merge_on = Utilities_df.prep_df_for_merge(
            df       = df, 
            merge_on = merge_on, 
            inplace  = True
        )
        mp_df, merge_on_mp = Utilities_df.prep_df_for_merge(
            df       = mp_df, 
            merge_on = merge_on_mp, 
            inplace  = True
        )
        #----------
        assert(
            df[merge_on].dtypes.tolist() ==
            mp_df[merge_on_mp].dtypes.tolist()
        )
        #-----
        df = pd.merge(
            df, 
            mp_df, 
            left_on  = merge_on, 
            right_on = merge_on_mp, 
            how      = how
        )
        #----------
        df = df.drop(columns=list(set(merge_on_mp).difference(merge_on)))
        #-------------------------
        return df
        

    #****************************************************************************************************
    # Methods to load and initialize AMI data
    #****************************************************************************************************
    @staticmethod
    def get_dflt_ami_slicers(
        ami_df_info_dict=None
    ):
        r"""
        """
        #--------------------------------------------------
        ami_df_info_dict = DOVSAudit.get_full_ami_df_info_dict(ami_df_info_dict=ami_df_info_dict)
        #--------------------------------------------------
        instvabc_slcr = DFSlicer(
            single_slicers = [
                dict(
                    column=ami_df_info_dict['aep_derived_uom_col'], 
                    value='VOLT', 
                    comparison_operator='=='
                ), 
                dict(
                    column=ami_df_info_dict['aep_srvc_qlty_idntfr_col'], 
                    value=['INSTVA1', 'INSTVB1', 'INSTVC1'], 
                    comparison_operator='isin'
                )
            ], 
            name='VOLT, INSTV(ABC)1', 
            join_single_slicers='and'
        )
        #-------------------------
        volt_avg_slcr = DFSlicer(
            single_slicers = [
                dict(
                    column=ami_df_info_dict['aep_derived_uom_col'], 
                    value='VOLT', 
                    comparison_operator='=='
                ), 
                dict(
                    column=ami_df_info_dict['aep_srvc_qlty_idntfr_col'], 
                    value='AVG', 
                    comparison_operator='=='
                )
            ], 
            name='VOLT, AVG', 
            join_single_slicers='and'
        )
        #-------------------------
        slicers=[instvabc_slcr, volt_avg_slcr]
        return slicers
    
    @staticmethod
    def choose_best_slicer_and_perform_slicing(
        df, 
        slicers, 
        groupby_SN=False, 
        t_search_min_max=None, 
        time_col='starttimeperiod_local', 
        value_col=None, 
        SN_col='serialnumber', 
        return_sorted=True
    ):
        r"""
        From slicers, choose the best option, slice the full df, and return.
        This can be done for the entire df as a whole, or by serial number, depending on the value of groupby_SN
        If t_search_min_max is set, the best slicer is chosen from the subset of df, df_choose_slcr, with
          time values within the constraints of t_search_min_max
        Although the best slicer is determined from the subset, the slicing is performed on the entirety of df.
        How is the best slicer chosen?
            If value_col is None, the best slicer is that with the most rows after slicing df_choose_slcr.
            If value_col is set, the best slicer is that with the most not-NA value_col entries after slicing df_choose_slcr.
            NOTE: If multiple best slicers exists, the first is chosen

        return_sorted:
            If true, the returned df is sorted according to time_col, SN_col
        """
        #----------------------------------------------------------------------------------------------------
        # NOTE: Need to exercise groupby_SN==True option here (instead of, e.g., after df_choose_slcr is found),
        #         so that slicing is done on entirety of df, as described in documentation
        if groupby_SN:
            # NOTE: groupby_SN MUST BE SET TO FALSE BELOW, otherwise infinite loop!
            return_df = df.groupby([SN_col], as_index=False, group_keys=False).apply(
                lambda x: DOVSAudit.choose_best_slicer_and_perform_slicing(
                    df=x, 
                    slicers=slicers, 
                    groupby_SN=False, 
                    t_search_min_max=t_search_min_max, 
                    time_col=time_col, 
                    value_col=value_col, 
                    SN_col=SN_col, 
                    return_sorted=return_sorted
                )
            )
            #-----
            # Sort, if return_sorted==True
            if return_sorted:
                return_df = return_df.sort_values(by=[time_col, SN_col])
            #-----
            return return_df
        #----------------------------------------------------------------------------------------------------
        assert(Utilities.is_object_one_of_types(slicers, [list, tuple]))
        #-------------------------
        if t_search_min_max is not None:
            assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]))
            assert(len(t_search_min_max)==2)
            #-----
            # At least one of t_search_min_max should not be None
            assert(t_search_min_max[0] is not None or t_search_min_max[1] is not None)
            if t_search_min_max[0] is None:
                t_search_min_max[0] = pd.Timestamp.min
            if t_search_min_max[1] is None:
                t_search_min_max[1] = pd.Timestamp.max
            #-----
            df_choose_slcr = df[
                (df[time_col]>=t_search_min_max[0]) & 
                (df[time_col]<=t_search_min_max[1])
            ]
        else:
            df_choose_slcr = df
        #-------------------------
        # Construct slcrs_w_counts to choose best slicer
        slcrs_w_counts = []
        for slcr_i in slicers:
            if value_col is None:
                counts_i = slcr_i.perform_slicing(df_choose_slcr).shape[0]
            else:
                counts_i = slcr_i.perform_slicing(df_choose_slcr)[value_col].notna().sum()
            slcrs_w_counts.append((slcr_i, counts_i))
        #-------------------------
        # Find the winner from slcrs_w_counts
        slicer = max(slcrs_w_counts, key=lambda x: x[1])[0]

        #-------------------------
        # Perform the slicing
        return_df = slicer.perform_slicing(df).copy()

        #-------------------------
        # Sort, if return_sorted==True
        if return_sorted:
            return_df = return_df.sort_values(by=[time_col, SN_col])

        #-------------------------
        return return_df
    
    
    @staticmethod
    def reduce_INSTV_ABC_1_vals_in_df(
        df, 
        value_col='value', 
        aep_derived_uom_col='aep_derived_uom', 
        aep_srvc_qlty_idntfr_col='aep_srvc_qlty_idntfr', 
        output_aep_srvc_qlty_idntfr = 'INSTV(ABC)1', 
        include_index_in_shared_cols=True, 
        gpby_dropna=False
    ):
        r"""
        Function to reduce down multiple INSTVA1/INSTVB1/INSTVC1 for each timestamp into a single average value.
        -----
        In general, it is desired that each serial number have a single value per time stamp.
        For this function, df can have a mix of INSTV(ABC)1 and AVG voltage readings, HOWEVER, it is expected that for
          each SN/timestamp combination, there should only be INSTV(ABC)1 values or a single AVG value.
        This is built only for aep_derived_uom==VOLT, and aep_srvc_qlty_idntfr in [INSTVA1, INSTVB1, INSTVC1, AVG]

        include_index_in_shared_cols:
            As various elements will be grouped/rows collapsed/however you want to think about it, the index will
              necessarily be lost, 
            In many cases (e.g., when the indices are time stamps) the indices for those being combined should be 
              shared (but, pandas doesn't know that).
            If this is the case, set include_index_in_shared_cols==True
            NOTE: For the foreseeable applications, this only really makes sense if the indices are datetime objects,
                  so this will be enforced.
        """
        #-------------------------
        assert(df[aep_derived_uom_col].nunique()==1)
        assert(df[aep_derived_uom_col].unique()[0]=='VOLT')
        #-----
        assert(len(set(df[aep_srvc_qlty_idntfr_col].unique().tolist()).difference(
            set(['INSTVA1', 'INSTVB1', 'INSTVC1', 'AVG'])
        ))==0)
        #-------------------------
        # Grab og_cols to maintain column ordering in output
        og_cols = df.columns.tolist()

        #-------------------------
        grp_by_cols = [x for x in df.columns.tolist() 
                       if x not in [value_col, aep_derived_uom_col, aep_srvc_qlty_idntfr_col]]
        cols_shared_by_groups = [aep_derived_uom_col]
        agg_dict = {
            value_col:               'mean', 
            aep_derived_uom_col:     'first', 
            aep_srvc_qlty_idntfr_col: lambda x: ' '.join(natsorted(set(x)))
        }
        if include_index_in_shared_cols:
            # As various elements will be grouped/rows collapsed/however you want to think about it, the index will
            #   necessarily be lost, even though the indices for those being combined should be shared (but, pandas
            #   doesn't know that).
            # In order to retain the index, stored it in a temporary column and re-set it later
            # NOTE: This assumes there is a single index level.  If MultiIndex, method will need re-worked
            assert(df.index.nlevels==1)
            assert(is_datetime64_dtype(df.index.dtype))
            if df.index.name is not None and df.index.name not in df.columns.tolist():
                tmp_idx_col = df.index.name
            else:
                tmp_idx_col = Utilities.generate_random_string()
            df = df.reset_index(drop=False, names=tmp_idx_col)
            cols_shared_by_groups.append(tmp_idx_col)
            agg_dict[tmp_idx_col] = 'first'
        #----------
        # Make sure all columns in cols_shared_by_groups have a single value per group
        assert((df.groupby(grp_by_cols, dropna=gpby_dropna)[cols_shared_by_groups].nunique()<=1).all().all())    

        #----------
        # Do aggregation
        return_df = df.groupby(grp_by_cols, dropna=gpby_dropna, as_index=False, group_keys=False).agg(agg_dict)

        #-------------------------
        # Due to join operation about with natsorted and set, the only possible values for aep_srvc_qlty_idntfr_col
        #   in return_df are the sorted combinations of ['INSTVA1', 'INSTVB1', 'INSTVC1']
        #   e.g., 'INSTVA1', 'INSTVB1', 'INSTVC1', 'INSTVA1 INSTVB1', ... 'INSTVA1 INSTVB1 INSTVC1'
        # Assert that this is true
        accptbl_fnl_srvc_qlty_idntfrs = ['INSTVA1', 'INSTVB1', 'INSTVC1']
        #-----
        combs = []
        for i in range(1, len(accptbl_fnl_srvc_qlty_idntfrs)+1):
            els = [list(x) for x in itertools.combinations(accptbl_fnl_srvc_qlty_idntfrs, i)]
            combs.extend(els)
        #-----
        accptbl_fnl_srvc_qlty_idntfrs = [' '.join(x) for x in combs]
        accptbl_fnl_srvc_qlty_idntfrs.append('AVG')
        #-----
        assert(len(set(df[aep_srvc_qlty_idntfr_col].unique().tolist()).difference(set(accptbl_fnl_srvc_qlty_idntfrs)))==0)
        #-------------------------
        # Set rows with aep_srvc_qlty_idntfr!='VOLT' equal to output_aep_srvc_qlty_idntfr
        return_df.loc[return_df[aep_srvc_qlty_idntfr_col]!='AVG', aep_srvc_qlty_idntfr_col] = output_aep_srvc_qlty_idntfr
        #-------------------------
        if include_index_in_shared_cols:
            # Set index back to original values
            return_df = return_df.set_index(tmp_idx_col, drop=True)
        #-------------------------
        assert(len(set(og_cols).symmetric_difference(set(return_df.columns.tolist())))==0)
        return_df = return_df[og_cols]
        #-------------------------
        return return_df
    

    @staticmethod
    def perform_std_init_ami(
        outg_rec_nb, 
        ami_df_i, 
        slicers=None, 
        ami_df_info_dict=None
    ):
        r"""
        Perform standard initialization for ami_df_i
        """
        #--------------------------------------------------
        ami_df_i = ami_df_i.copy()
        #--------------------------------------------------
        ami_df_info_dict = DOVSAudit.get_full_ami_df_info_dict(ami_df_info_dict=ami_df_info_dict)
        #--------------------------------------------------
        if slicers is None:
            slicers = DOVSAudit.get_dflt_ami_slicers(ami_df_info_dict)
        #--------------------------------------------------
        # First, make sure we only keep entries in ami_df_i associated with the given outage outg_rec_nb
        outg_rec_nb_idfr_loc = Utilities_df.get_idfr_loc(
                df   = ami_df_i, 
                idfr = ami_df_info_dict['outg_rec_nb_idfr']
            )
        #-----
        if outg_rec_nb_idfr_loc[1]:
            # outg_rec_nb contained in index level outg_rec_nb_idfr_loc[0]
            outg_rec_nb_idx_lvl = outg_rec_nb_idfr_loc[0]
            assert(outg_rec_nb in ami_df_i.index.get_level_values(outg_rec_nb_idx_lvl).unique().tolist())
            ami_df_i = ami_df_i[ami_df_i.index.get_level_values(outg_rec_nb_idx_lvl)==outg_rec_nb]
        else:
            # outg_rec_nb contained in column outg_rec_nb_idfr_loc[0]
            outg_rec_nb_col = outg_rec_nb_idfr_loc[0]
            assert(outg_rec_nb in ami_df_i[outg_rec_nb_col].unique().tolist())
            ami_df_i = ami_df_i[ami_df_i[outg_rec_nb_col]==outg_rec_nb].copy()

        #--------------------------------------------------
        # Although I cannot yet call DOVSAudit.choose_best_slicer_and_perform_slicing and DOVSAudit.reduce_INSTV_ABC_1_vals_in_df, 
        #   as the standard cleaning and conversions must be done first, I am able to cut down the size of
        #   ami_df_i by joining the slicers with 'or' statements.
        # Thus, ami_df_i will be reduced to only the subset of data which will be considered in 
        #   DOVSAudit.choose_best_slicer_and_perform_slicing
        # As mentioned, this will cut down the size of ami_df_i and will also save time and resources by not having
        #   to run entire DF through cleaning and conversions procedures.
        ami_df_i = DFSlicer.combine_slicers_and_perform_slicing(
            df           = ami_df_i, 
            slicers      = slicers, 
            join_slicers = 'or', 
            apply_not    = False
        )
        if ami_df_i.shape[0]==0:
            return ami_df_i

        #--------------------------------------------------
        ami_df_i = AMINonVee.perform_std_initiation_and_cleaning(
            df             = ami_df_i, 
            drop_na_values = True, 
            inplace        = True
        )
        #-----
        # Should the following be added to AMINonVee.perform_std_initiation_and_cleaning?
        ami_df_i = Utilities_dt.strip_tz_info_and_convert_to_dt(
            df            = ami_df_i, 
            time_col      = ami_df_info_dict['t_int_beg_col_raw'], 
            placement_col = ami_df_info_dict['t_int_beg_col'], 
            run_quick     = True, 
            n_strip       = 6, 
            inplace       = False
        )
        ami_df_i = Utilities_dt.strip_tz_info_and_convert_to_dt(
            df            = ami_df_i, 
            time_col      = ami_df_info_dict['t_int_end_col_raw'], 
            placement_col = ami_df_info_dict['t_int_end_col'], 
            run_quick     = True, 
            n_strip       = 6, 
            inplace       = False
        )
        #--------------------------------------------------
        ami_df_i = DOVSAudit.choose_best_slicer_and_perform_slicing(
            df = ami_df_i, 
            slicers          = slicers, 
            groupby_SN       = True, 
            t_search_min_max = None, 
            time_col         = ami_df_info_dict['t_int_beg_col'], 
            value_col        = None, 
            SN_col           = ami_df_info_dict['SN_col'], 
            return_sorted    = True
        )

        ami_df_i = DOVSAudit.reduce_INSTV_ABC_1_vals_in_df(
            df                          = ami_df_i, 
            value_col                   = ami_df_info_dict['value_col'], 
            aep_derived_uom_col         = ami_df_info_dict['aep_derived_uom_col'], 
            aep_srvc_qlty_idntfr_col    = ami_df_info_dict['aep_srvc_qlty_idntfr_col'], 
            output_aep_srvc_qlty_idntfr = 'INSTV(ABC)1'
        )
        #--------------------------------------------------
        if ami_df_i.shape[0]==0:
            return ami_df_i
        #--------------------------------------------------
        # Each serial number should have a single value per time stamp
        assert(ami_df_i.groupby([ami_df_info_dict['SN_col'], ami_df_info_dict['t_int_beg_col']]).ngroups == ami_df_i.shape[0])    
        #--------------------------------------------------
        return ami_df_i
    
    
    def load_ami_from_csvs(
        self, 
        paths, 
        slicers=None, 
        ami_df_info_dict=None, 
        run_std_init=True, 
        cols_and_types_to_convert_dict=None, 
        to_numeric_errors='coerce', 
        drop_na_rows_when_exception=True, 
        drop_unnamed0_col=True, 
        pd_read_csv_kwargs=None, 
        make_all_columns_lowercase=False, 
        assert_all_cols_equal=True, 
        min_fsize_MB=None
    ):
        r"""
        See GenAn.read_df_from_csv_batch for more info
        """
        #-------------------------
        ami_df_i = GenAn.read_df_from_csv_batch(
            paths                          = paths, 
            cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors              = to_numeric_errors, 
            drop_na_rows_when_exception    = drop_na_rows_when_exception, 
            drop_unnamed0_col              = drop_unnamed0_col, 
            pd_read_csv_kwargs             = pd_read_csv_kwargs, 
            make_all_columns_lowercase     = make_all_columns_lowercase, 
            assert_all_cols_equal          = assert_all_cols_equal, 
            min_fsize_MB                   = min_fsize_MB
        )
        #-------------------------
        ami_df_info_dict = DOVSAudit.get_full_ami_df_info_dict(ami_df_info_dict=ami_df_info_dict)
        #-------------------------
        if run_std_init:
            ami_df_i = DOVSAudit.perform_std_init_ami(
                outg_rec_nb      = self.outg_rec_nb, 
                ami_df_i         = ami_df_i, 
                slicers          = slicers, 
                ami_df_info_dict = ami_df_info_dict
            )
            DOVSAudit.check_ami_df_info_dict(
                ami_df           = ami_df_i, 
                ami_df_info_dict = ami_df_info_dict
            )
        #-------------------------
        self.ami_df_i         = ami_df_i
        self.ami_df_info_dict = ami_df_info_dict
        self.__is_loaded_ami  = True
        
        
    def load_ami_from_df(
        self, 
        ami_df_i, 
        ami_df_info_dict=None, 
        run_std_init=False, 
        slicers=None
    ):
        r"""
        """
        #-------------------------
        ami_df_info_dict = DOVSAudit.get_full_ami_df_info_dict(ami_df_info_dict=ami_df_info_dict)
        #-------------------------
        if run_std_init:
            ami_df_i = DOVSAudit.perform_std_init_ami(
                outg_rec_nb      = self.outg_rec_nb, 
                ami_df_i         = ami_df_i, 
                slicers          = slicers, 
                ami_df_info_dict = ami_df_info_dict
            )
        #-------------------------
        DOVSAudit.check_ami_df_info_dict(
            ami_df           = ami_df_i, 
            ami_df_info_dict = ami_df_info_dict
        )
        #-------------------------
        self.ami_df_i         = ami_df_i
        self.ami_df_info_dict = ami_df_info_dict
        self.__is_loaded_ami  = True


    def run_ami_daq(
        self, 
        search_time_half_window, 
        slicers = None, 
        build_sql_fncn=AMINonVee_SQL.build_sql_usg, 
        addtnl_build_sql_kwargs=None, 
        ami_df_info_dict=None, 
        save_args=False
    ):
        r"""

        save_args:
            If one wants to output the DAQ results (with summary files, etc), supply save_args
            This should be a dict with keys:
                - save_to_file
                - save_dir
                - save_name
                - index
            See GenAn for more information
        """
        #-------------------------
        assert(self.__is_loaded_dovs)
        assert(self.__is_loaded_mp)
        #-------------------------
        ami_df_info_dict = DOVSAudit.get_full_ami_df_info_dict(ami_df_info_dict=ami_df_info_dict)
        #-------------------------
        if addtnl_build_sql_kwargs is None:
            addtnl_build_sql_kwargs = dict()
        assert(isinstance(addtnl_build_sql_kwargs, dict))
        #-------------------------
        # Grab premises from self.dovs_df_i
        if self.dovs_df_info_dict['is_consolidated']:
            assert(self.dovs_df_i.shape[0]==1)
            PNs = list(set(self.dovs_df_i.iloc[0][self.dovs_df_info_dict['PNs_col']]))
        else:
            PNs = self.dovs_df_i[self.dovs_df_info_dict['PN_col']].unique().tolist()
        #-------------------------
        # Form t_search_min_max using the DOVS outage beg./end times together with search_time_half_window
        t_search_min_max = (
            self.dovs_outg_t_beg_end[0] - search_time_half_window, 
            self.dovs_outg_t_beg_end[1] + search_time_half_window
        )
        #-------------------------
        build_sql_kwargs = copy.deepcopy(addtnl_build_sql_kwargs)
        build_sql_kwargs['premise_nbs']    = PNs
        build_sql_kwargs['datetime_range'] = t_search_min_max
        build_sql_kwargs['field_to_split'] = 'premise_nbs'
        build_sql_kwargs['batch_size']     = build_sql_kwargs.get('batch_size', 1000)
        #-------------------------
        ami = AMINonVee(
            df_construct_type         = DFConstructType.kRunSqlQuery,
            contstruct_df_args        = None, 
            init_df_in_constructor    = True, 
            build_sql_function        = build_sql_fncn, 
            build_sql_function_kwargs = build_sql_kwargs, 
            save_args                 = save_args
        )
        ami_df = ami.df.copy()
        #-------------------------
        # Merge with self.mp_df_i to include trsf_pole_nb, technology_tx, etc.
        #----------
        ami_df = DOVSAudit.merge_df_with_mp(
            df              = ami_df, 
            mp_df           = self.mp_df_i, 
            df_info_dict    = ami_df_info_dict, 
            mp_df_info_dict = self.mp_df_info_dict
        )
        #----------
        ami_df_info_dict['outg_rec_nb_idfr']  = self.mp_df_info_dict['outg_rec_nb_idfr']
        ami_df_info_dict['trsf_pole_nb_col']  = self.mp_df_info_dict['trsf_pole_nb_col']
        ami_df_info_dict['technology_tx_col'] = self.mp_df_info_dict['technology_tx_col']
        #-------------------------
        # Run initiation procedure
        ami_df = DOVSAudit.perform_std_init_ami(
            outg_rec_nb      = self.outg_rec_nb, 
            ami_df_i         = ami_df, 
            slicers          = slicers, 
            ami_df_info_dict = ami_df_info_dict
        )
        DOVSAudit.check_ami_df_info_dict(
            ami_df           = ami_df, 
            ami_df_info_dict = ami_df_info_dict
        )
        #-------------------------
        self.ami_df_i         = ami_df
        self.ami_df_info_dict = ami_df_info_dict
        self.__is_loaded_ami  = True


    #****************************************************************************************************
    # Methods to load and initialize EDE data
    #****************************************************************************************************
    @staticmethod
    def perform_std_init_ede(
        outg_rec_nb, 
        ede_df_i, 
        ede_df_info_dict=None
    ):
        r"""
        Perform standard initialization for ede_df_i
        """
        #--------------------------------------------------
        ede_df_i = ede_df_i.copy()
        #--------------------------------------------------
        ede_df_info_dict = DOVSAudit.get_full_ede_df_info_dict(ede_df_info_dict=ede_df_info_dict)
        #--------------------------------------------------
        # First, make sure we only keep entries in ede_df_i associated with the given outage outg_rec_nb
        outg_rec_nb_idfr_loc = Utilities_df.get_idfr_loc(
                df   = ede_df_i, 
                idfr = ede_df_info_dict['outg_rec_nb_idfr']
            )
        #-----
        if outg_rec_nb_idfr_loc[1]:
            # outg_rec_nb contained in index level outg_rec_nb_idfr_loc[0]
            outg_rec_nb_idx_lvl = outg_rec_nb_idfr_loc[0]
            assert(outg_rec_nb in ede_df_i.index.get_level_values(outg_rec_nb_idx_lvl).unique().tolist())
            ede_df_i = ede_df_i[ede_df_i.index.get_level_values(outg_rec_nb_idx_lvl)==outg_rec_nb]
        else:
            # outg_rec_nb contained in column outg_rec_nb_idfr_loc[0]
            outg_rec_nb_col = outg_rec_nb_idfr_loc[0]
            assert(outg_rec_nb in ede_df_i[outg_rec_nb_col].unique().tolist())
            ede_df_i = ede_df_i[ede_df_i[outg_rec_nb_col]==outg_rec_nb].copy()
        #--------------------------------------------------
        if ede_df_i.shape[0]==0:
            return ede_df_i
        #--------------------------------------------------
        ede_df_i = Utilities_dt.strip_tz_info_and_convert_to_dt(
            df            = ede_df_i, 
            time_col      = ede_df_info_dict['time_col_raw'], 
            placement_col = ede_df_info_dict['time_col'], 
            run_quick     = True, 
            n_strip       = 6, 
            inplace       = False
        )
        #-------------------------
        ede_df_i = AMIEndEvents.reduce_end_event_reasons_in_df(
            df = ede_df_i, 
            reason_col                    = ede_df_info_dict['reason_col'], 
            edetypeid_col                 = ede_df_info_dict['ede_typeid_col'], 
            patterns_to_replace_by_typeid = None, 
            addtnl_patterns_to_replace    = None, 
            placement_col                 = None, 
            count                         = 0, 
            flags                         = re.IGNORECASE,  
            inplace                       = True
        )
        #-------------------------
        ede_cols_to_keep = [
            ede_df_info_dict['time_col'], 
            ede_df_info_dict['reason_col'], 
            ede_df_info_dict['SN_col'], 
            ede_df_info_dict['PN_col'], 
            ede_df_info_dict['ede_typeid_col'], 
            ede_df_info_dict['event_type_col'], 
            ede_df_info_dict['trsf_pole_nb_col'],
        ]
        if outg_rec_nb_idfr_loc[1]==False:
            ede_cols_to_keep.append(outg_rec_nb_idfr_loc[0])
        ede_df_i = ede_df_i[ede_cols_to_keep]
        #-------------------------
        return ede_df_i

        
    def load_ede_from_csvs(
        self, 
        paths, 
        slicers=None, 
        ede_df_info_dict=None, 
        run_std_init=True, 
        cols_and_types_to_convert_dict=None, 
        to_numeric_errors='coerce', 
        drop_na_rows_when_exception=True, 
        drop_unnamed0_col=True, 
        pd_read_csv_kwargs=None, 
        make_all_columns_lowercase=False, 
        assert_all_cols_equal=True, 
        min_fsize_MB=None
    ):
        r"""
        See GenAn.read_df_from_csv_batch for more info
        """
        #-------------------------
        ede_df_i = GenAn.read_df_from_csv_batch(
            paths                          = paths, 
            cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors              = to_numeric_errors, 
            drop_na_rows_when_exception    = drop_na_rows_when_exception, 
            drop_unnamed0_col              = drop_unnamed0_col, 
            pd_read_csv_kwargs             = pd_read_csv_kwargs, 
            make_all_columns_lowercase     = make_all_columns_lowercase, 
            assert_all_cols_equal          = assert_all_cols_equal, 
            min_fsize_MB                   = min_fsize_MB
        )
        #-------------------------
        ede_df_info_dict = DOVSAudit.get_full_ede_df_info_dict(ede_df_info_dict=ede_df_info_dict)
        #-------------------------
        if run_std_init:
            ede_df_i = DOVSAudit.perform_std_init_ede(
                outg_rec_nb      = self.outg_rec_nb, 
                ede_df_i         = ede_df_i, 
                ede_df_info_dict = ede_df_info_dict
            )
            DOVSAudit.check_ede_df_info_dict(
                ede_df           = ede_df_i, 
                ede_df_info_dict = ede_df_info_dict
            )
        #-------------------------
        self.ede_df_i         = ede_df_i
        self.ede_df_info_dict = ede_df_info_dict
        self.__is_loaded_ede  = True
        
        
    def load_ede_from_df(
        self, 
        ede_df_i, 
        ede_df_info_dict=None, 
        run_std_init=False, 
    ):
        r"""
        """
        #-------------------------
        ede_df_info_dict = DOVSAudit.get_full_ede_df_info_dict(ede_df_info_dict=ede_df_info_dict)
        #-------------------------
        if ede_df_i is not None and ede_df_i.shape[0]>0:
            if run_std_init:
                ede_df_i = DOVSAudit.perform_std_init_ede(
                    outg_rec_nb      = self.outg_rec_nb, 
                    ede_df_i         = ede_df_i, 
                    ede_df_info_dict = ede_df_info_dict
                )
            #-------------------------
            DOVSAudit.check_ede_df_info_dict(
                ede_df           = ede_df_i, 
                ede_df_info_dict = ede_df_info_dict
            )
        #-------------------------
        self.ede_df_i         = ede_df_i
        self.ede_df_info_dict = ede_df_info_dict
        self.__is_loaded_ede  = True
        
        
    def run_ede_daq(
        self, 
        search_time_half_window, 
        build_sql_fncn=AMIEndEvents_SQL.build_sql_end_events, 
        addtnl_build_sql_kwargs=None, 
        ede_df_info_dict=None, 
        save_args=False
    ):
        r"""

        save_args:
            If one wants to output the DAQ results (with summary files, etc), supply save_args
            This should be a dict with keys:
                - save_to_file
                - save_dir
                - save_name
                - index
            See GenAn for more information
        """
        #-------------------------
        assert(self.__is_loaded_dovs)
        assert(self.__is_loaded_mp)
        #-------------------------
        ede_df_info_dict = DOVSAudit.get_full_ede_df_info_dict(ede_df_info_dict=ede_df_info_dict)
        #-------------------------
        if addtnl_build_sql_kwargs is None:
            addtnl_build_sql_kwargs = dict()
        assert(isinstance(addtnl_build_sql_kwargs, dict))
        #-------------------------
        # Grab premises from self.dovs_df_i
        if self.dovs_df_info_dict['is_consolidated']:
            assert(self.dovs_df_i.shape[0]==1)
            PNs = list(set(self.dovs_df_i.iloc[0][self.dovs_df_info_dict['PNs_col']]))
        else:
            PNs = self.dovs_df_i[self.dovs_df_info_dict['PN_col']].unique().tolist()
        #-------------------------
        # Form t_search_min_max using the DOVS outage beg./end times together with search_time_half_window
        t_search_min_max = (
            self.dovs_outg_t_beg_end[0] - search_time_half_window, 
            self.dovs_outg_t_beg_end[1] + search_time_half_window
        )
        #-------------------------
        build_sql_kwargs = copy.deepcopy(addtnl_build_sql_kwargs)
        build_sql_kwargs['premise_nbs']    = PNs
        build_sql_kwargs['datetime_range'] = t_search_min_max
        build_sql_kwargs['field_to_split'] = 'premise_nbs'
        build_sql_kwargs['batch_size']     = build_sql_kwargs.get('batch_size', 1000)
        #-------------------------
        ede = AMIEndEvents(
            df_construct_type         = DFConstructType.kRunSqlQuery,
            contstruct_df_args        = None, 
            init_df_in_constructor    = True, 
            build_sql_function        = build_sql_fncn, 
            build_sql_function_kwargs = build_sql_kwargs, 
            save_args                 = save_args
        )
        ede_df = ede.df.copy()
        if ede_df.shape[0]==0:
            ede_df = copy.deepcopy(self.init_dfs_to)
        else:
            #-------------------------
            # Merge with self.mp_df_i to include trsf_pole_nb, technology_tx, etc.
            #----------
            ede_df = DOVSAudit.merge_df_with_mp(
                df              = ede_df, 
                mp_df           = self.mp_df_i, 
                df_info_dict    = ede_df_info_dict, 
                mp_df_info_dict = self.mp_df_info_dict
            )
            #----------
            ede_df_info_dict['outg_rec_nb_idfr']  = self.mp_df_info_dict['outg_rec_nb_idfr']
            ede_df_info_dict['trsf_pole_nb_col']  = self.mp_df_info_dict['trsf_pole_nb_col']
            #-------------------------
            # Run initiation procedure
            ede_df = DOVSAudit.perform_std_init_ede(
                outg_rec_nb      = self.outg_rec_nb, 
                ede_df_i         = ede_df, 
                ede_df_info_dict = ede_df_info_dict
            )
            DOVSAudit.check_ede_df_info_dict(
                ede_df           = ede_df, 
                ede_df_info_dict = ede_df_info_dict
            )
        #-------------------------
        self.ede_df_i         = ede_df
        self.ede_df_info_dict = ede_df_info_dict
        self.__is_loaded_ede  = True


    #****************************************************************************************************
    #****************************************************************************************************
    def run_ami_and_ede_daq(
        self, 
        search_time_half_window,  
        #-----
        slicers_ami = None, 
        build_sql_fncn_ami=AMINonVee_SQL.build_sql_usg, 
        addtnl_build_sql_kwargs_ami=None, 
        ami_df_info_dict=None, 
        save_args_ami=False, 
        #-----
        build_sql_fncn_ede=AMIEndEvents_SQL.build_sql_end_events, 
        addtnl_build_sql_kwargs_ede=None, 
        ede_df_info_dict=None, 
        save_args_ede=False
    ):
        r"""
        """
        #-------------------------
        assert(self.__is_loaded_dovs)
        assert(self.__is_loaded_mp)
        #-------------------------
        self.run_ami_daq(
            search_time_half_window = search_time_half_window, 
            slicers                 = slicers_ami, 
            build_sql_fncn          = build_sql_fncn_ami, 
            addtnl_build_sql_kwargs = addtnl_build_sql_kwargs_ami, 
            ami_df_info_dict        = ami_df_info_dict, 
            save_args               = save_args_ami
        )
        #-------------------------
        self.run_ede_daq(
            search_time_half_window = search_time_half_window, 
            build_sql_fncn          = build_sql_fncn_ede, 
            addtnl_build_sql_kwargs = addtnl_build_sql_kwargs_ede, 
            ede_df_info_dict        = ede_df_info_dict, 
            save_args               = save_args_ede
        )    
        
    
    #****************************************************************************************************
    # Methods to assess whether or not outage should be analyzed.
    #****************************************************************************************************
    @staticmethod
    def check_found_ami_for_SN(
        df_i, 
        ts_req, 
        time_col='starttimeperiod_local', 
        SN_col='serialnumber', 
    ):
        r"""
        Intended for use in DOVSAudit.check_found_ami_for_all_SNs_in_outage.
        df_i must be for a single SN
        Just checks that all required timestamps (ts_req) are found in df_i[time_col]
        """
        #-------------------------
        assert(df_i[SN_col].nunique()==1)
        #-------------------------
        if len(set(ts_req).difference(set(df_i[time_col].tolist())))==0:
            return True
        else:
            return False
        
        
    @staticmethod
    def check_found_ami_for_all_SNs_in_outage(
        df, 
        t_search_min_max, 
        requirement='all', 
        time_col='starttimeperiod_local', 
        value_col='value', 
        SN_col='serialnumber', 
    ):
        r"""
        """
        #-------------------------
        assert(requirement in ['all', 'endpoints'])
        #-------------------------
        # Need frequency for rounding and generating required timestamps (if requirement=='all')
        freq = Utilities_df.determine_freq_in_df_col(
            df=df, 
            groupby_SN=True, 
            return_val_if_not_found='error', 
            assert_single_freq=True, 
            assert_expected_freq_found=False, 
            expected_freq=pd.Timedelta('15 minutes'), 
            time_col=time_col, 
            SN_col=SN_col
        )
        #-------------------------
        assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]) and len(t_search_min_max)==2)
        # t_search_min_max must be datetime object
        if not isinstance(t_search_min_max[0], datetime.datetime) or not isinstance(t_search_min_max[1], datetime.datetime):
            t_search_min_max = [pd.to_datetime(t_search_min_max[0]), pd.to_datetime(t_search_min_max[1])]
        # Need to round t_search_min down to nearest freq and t_search_max up to nearest freq
        t_search_min_max = [t_search_min_max[0].floor(freq=freq), t_search_min_max[1].ceil(freq=freq)]
        #-------------------------
        # Generate the needed timestamps according to requirement argument
        if requirement=='all':
            ts_req = pd.date_range(start=t_search_min_max[0], end=t_search_min_max[1], freq=freq)
        elif requirement=='endpoints':
            ts_req = t_search_min_max
        else:
            assert(0)
        #-------------------------
        # Generate the series containing all pass values (index is SN, value is boolean representing pass/fail)
        pass_srs = df.groupby(SN_col).apply(
            lambda x: DOVSAudit.check_found_ami_for_SN(
                df_i=x, 
                ts_req=ts_req, 
                time_col=time_col, 
                SN_col=SN_col
            )
        )

        #-------------------------
        # In order to pass, all SNs must have the required AMI
        if all(pass_srs):
            return True
        else:
            return False
        
        
    @staticmethod
    def check_found_ami_for_all_SNs_in_outage_2(
        df, 
        t_search_min_max, 
        requirement='all', 
        time_col='starttimeperiod_local', 
        value_col='value', 
        SN_col='serialnumber', 
    ):
        r"""
        """
        #-------------------------
        assert(requirement in ['all', 'endpoints'])
        #-------------------------
        # Need frequency for rounding and generating required timestamps (if requirement=='all')
        freq = Utilities_df.determine_freq_in_df_col(
            df=df, 
            groupby_SN=True, 
            return_val_if_not_found='error', 
            assert_single_freq=True, 
            assert_expected_freq_found=False, 
            expected_freq=pd.Timedelta('15 minutes'), 
            time_col=time_col, 
            SN_col=SN_col
        )
        #-------------------------
        assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]) and len(t_search_min_max)==2)
        # t_search_min_max must be datetime object
        if not isinstance(t_search_min_max[0], datetime.datetime) or not isinstance(t_search_min_max[1], datetime.datetime):
            t_search_min_max = [pd.to_datetime(t_search_min_max[0]), pd.to_datetime(t_search_min_max[1])]
        # Need to round t_search_min down to nearest freq and t_search_max up to nearest freq
        t_search_min_max = [t_search_min_max[0].floor(freq=freq), t_search_min_max[1].ceil(freq=freq)]
        #-------------------------
        # Generate the needed timestamps according to requirement argument
        if requirement=='all':
            ts_req = pd.date_range(start=t_search_min_max[0], end=t_search_min_max[1], freq=freq)
        elif requirement=='endpoints':
            ts_req = t_search_min_max
        else:
            assert(0)
        #-------------------------
        # Generate a list containing the pass values
        pass_vals = []
        for SN_i in df[SN_col].unique().tolist():
            df_i = df[df[SN_col]==SN_i]
            pass_i = DOVSAudit.check_found_ami_for_SN(
                df_i=df_i, 
                ts_req=ts_req, 
                time_col=time_col, 
                SN_col=SN_col, 
            )
            pass_vals.append(pass_i)
        #-------------------------
        # In order to pass, all SNs must have the required AMI
        if all(pass_vals):
            return True
        else:
            return False
    
    
    @staticmethod
    def assess_outage_inclusion_requirements(
        ami_df_i, 
        outg_rec_nb, 
        dovs_df, 
        max_pct_PNs_missing_allowed=0, 
        ami_df_info_dict=None, 
        dovs_df_info_dict=None, 
        check_found_ami_for_all_SNs_kwargs=None
    ):
        r"""
        Check whether or not outage should be included in analysis.
        Returns boolean; True if the outage should be included, False if not
        This check includes:
            - ensure minimum percentage of PNs listed in DOVS are found in AMI
                - By default, all PNs must be included for an outage to be deemed suitable
                - Value set by max_pct_PNs_missing_allowed parameter, which should be between 0 and 100
            - ensure usable AMI data are found for each meter
                - Enforced using the function DOVSAudit.check_found_ami_for_all_SNs_in_outage, which has a
                    requirement parameter which (at the time of writing) may be set to 'all' or 'endpoints'

        ami_df_i:
            AMI DataFrame for a single outage.
            This should be reduced down to only the data of interest, e.g., only data with 'aep_derived_uom'=='VOLT'
              and 'aep_srvc_qlty_idntfr'=='AVG' or instantaneous voltage data after reduction is run (using, e.g.,
              the function DOVSAudit.reduce_INSTV_ABC_1_vals_in_df)

        outg_rec_nb:
            The outage record number, used to identify the outage information in dovs_df

        dovs_df:
            DOVS DataFrame containing information about the outage.
            The preferred form for this DF is consolidated form, meaning one row per outage with the elements
              of the premise column being list objects

        max_pct_PNs_missing_allowed:
            The maximum percent of PNs found in DOVS missing from AMI allowed for the outage to be included (to return True)
            default: 0 (meaning all PNs must be found)


        ami_df_info_dict:
            Gives necessary information regarding ami_df_i.
            See default values in dflt_ami_df_info_dict below
            PN_col:
                default: 'aep_premise_nb'
            time_col:
                default: 'starttimeperiod_local', 
            value_col:
                default: 'value', 
            SN_col: 
                default: 'serialnumber', 
            outg_rec_nb_col:
                default: 'OUTG_REC_NB_GPD_FOR_SQL'
                Not necessary in ami_df_i, but if present, the value will be compared to outg_rec_nb to ensure consistency
            outg_t_beg_col:
                default: 'DT_OFF_TS_FULL'
                Not necessary in ami_df_i, but if present, the value will be compared to the value extracted from dovs_df
            outg_t_end_col:
                default: 'DT_ON_TS'
                Not necessary in ami_df_i, but if present, the value will be compared to the value extracted from dovs_df

        dovs_df_info_dict:
            Gives necessary information regarding dovs_df, such as whether or not it is consolidated and the
              names of needed columns (e.g., PN_col)
            Default values given in code below in dflt_dovs_df_info_dict variable.
                is_consolidated:
                    default: True
                PN_col:
                    default: 'premise_nbs'
                outg_rec_nb_idfr:
                    default: 'index'
                    This directs where the outg_rec_nbs are stored in dovs_df, which can be a column or the index.
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
                outg_t_beg_col:
                    default: 'DT_OFF_TS_FULL'
                outg_t_end_col:
                    default: 'DT_ON_TS'

        check_found_ami_for_all_SNs_kwargs:
            Arguments to be fed into DOVSAudit.check_found_ami_for_all_SNs_in_outage function.
            See default values in dflt_check_found_ami_for_all_SNs_kwargs in code below.
            User should really only set t_search_min_max and requirement
                df:
                    Does not need to be input by user, as it will be set to ami_df_i
                t_search_min_max:
                    If not set by user, this will be set to [out_t_beg-pd.Timedelta('1hour'), out_t_end+pd.Timedelta('1hour')]
                requirement:
                    default: 'all'
                time_col:
                    Does not need to be input by user
                value_col:
                    Does not need to be input by user
                SN_col:
                    Does not need to be input by user
        """
        #-------------------------
        if ami_df_i.shape[0]==0:
            return False

        #--------------------------------------------------
        # Grab the relevant info from dovs_df
        #-------------------------
        dovs_df_info_dict = DOVSAudit.get_full_dovs_df_info_dict(dovs_df_info_dict=dovs_df_info_dict)
        #-------------------------
        dovs_df_i = DOVSOutages.retrieve_outage_from_dovs_df(
            dovs_df=dovs_df, 
            outg_rec_nb=outg_rec_nb, 
            outg_rec_nb_idfr=dovs_df_info_dict['outg_rec_nb_idfr'], 
            assert_outg_rec_nb_found=True
        )

        #-------------------------
        # Make sure the necessary columns are in dovs_df_i
        necessary_dovs_cols = [
            dovs_df_info_dict['PNs_col'], 
            dovs_df_info_dict['outg_t_beg_col'], 
            dovs_df_info_dict['outg_t_end_col']
        ]
        assert(len(set(necessary_dovs_cols).difference(set(dovs_df_i.columns.tolist())))==0)

        #-------------------------
        # Grab the needed info from dovs_df_i
        if dovs_df_info_dict['is_consolidated']:
            assert(dovs_df_i.shape[0]==1)
            out_t_beg = dovs_df_i.iloc[0][dovs_df_info_dict['outg_t_beg_col']]
            out_t_end = dovs_df_i.iloc[0][dovs_df_info_dict['outg_t_end_col']]
            PNs_dovs  = dovs_df_i.iloc[0][dovs_df_info_dict['PNs_col']]
            assert(Utilities.is_object_one_of_types(PNs_dovs, [list, tuple, set]))
        else:
            assert(dovs_df_i[dovs_df_info_dict['outg_t_beg_col']].nunique()==1)
            assert(dovs_df_i[dovs_df_info_dict['outg_t_end_col']].nunique()==1)
            out_t_beg = dovs_df_i[dovs_df_info_dict['outg_t_beg_col']].unique().tolist()[0]
            out_t_end = dovs_df_i[dovs_df_info_dict['outg_t_end_col']].unique().tolist()[0]
            PNs_dovs  = dovs_df_i[dovs_df_info_dict['PNs_col']].unique().tolist()

        #--------------------------------------------------
        # Determine the PNs found in ami_df and compare to those from DOVS
        #-------------------------
        ami_df_info_dict = DOVSAudit.get_full_ami_df_info_dict(ami_df_info_dict=ami_df_info_dict)
        #-------------------------
        # Make sure the necessary columns are present in ami_df_i
        necessary_ami_cols = [
            ami_df_info_dict['PN_col'], 
            ami_df_info_dict['t_int_beg_col'], 
            ami_df_info_dict['value_col'], 
            ami_df_info_dict['SN_col']
        ]
        assert(len(set(necessary_ami_cols).difference(set(ami_df_i.columns.tolist())))==0)

        #-------------------------
        # If outage information also contained in ami_df, compare against expected values
        if ami_df_info_dict['outg_rec_nb_idfr'] is not None:
            outg_rec_nb_idfr_loc = Utilities_df.get_idfr_loc(
                    df   = ami_df_i, 
                    idfr = ami_df_info_dict['outg_rec_nb_idfr']
                )
            if outg_rec_nb_idfr_loc[1]:
                # outg_rec_nb contained in index level outg_rec_nb_idfr_loc[0]
                outg_rec_nb_idx_lvl = outg_rec_nb_idfr_loc[0]
                assert(ami_df_i.index.get_level_values(outg_rec_nb_idx_lvl).nunique()==1)
                assert(ami_df_i.index.get_level_values(outg_rec_nb_idx_lvl).unique().tolist()[0]==outg_rec_nb)
            else:
                # outg_rec_nb contained in column outg_rec_nb_idfr_loc[0]
                outg_rec_nb_col = outg_rec_nb_idfr_loc[0]
                assert(outg_rec_nb_col in ami_df_i.columns.tolist())
                assert(ami_df_i[outg_rec_nb_col].nunique()==1)
                assert(ami_df_i[outg_rec_nb_col].unique().tolist()[0]==outg_rec_nb)
        #-----
        if ami_df_info_dict['outg_t_beg_col'] in ami_df_i.columns.tolist():
            assert(ami_df_i[ami_df_info_dict['outg_t_beg_col']].nunique()==1)
            assert(ami_df_i[ami_df_info_dict['outg_t_beg_col']].unique().tolist()[0]==out_t_beg)
        #-----
        if ami_df_info_dict['outg_t_end_col'] in ami_df_i.columns.tolist():
            assert(ami_df_i[ami_df_info_dict['outg_t_end_col']].nunique()==1)
            assert(ami_df_i[ami_df_info_dict['outg_t_end_col']].unique().tolist()[0]==out_t_end)

        #-------------------------
        # Grab PNs_ami
        PNs_ami = ami_df_i[ami_df_info_dict['PN_col']].unique().tolist()

        #-----*****-----*****-----*****-----*****-----
        # Find difference in PNs between DOVS and AMI
        dff_pns = list(set(PNs_dovs).difference(set(PNs_ami)))

        # NOTE: For the current analysis, where data are queried according to premise numbers, there should
        #         never be a situation where PNs found in AMI not present in DOVS.
        #       For now, assert this to be true.
        #       In the future, if I instead query at the transformer level to try to pick up PNs possible missed
        #         by DOVS, this may not be the case.
        assert(len(set(PNs_ami).difference(set(PNs_dovs)))==0)

        # Check if difference within allowable range, as set by max_pct_PNs_missing_allowed
        pct_pns_missing = 100.*len(dff_pns)/len(PNs_dovs)
        if pct_pns_missing > max_pct_PNs_missing_allowed:
            return False

        #--------------------------------------------------
        # Check that usable AMI data are found for each meter
        #   At this point, the minimum percentage of PNs listed in DOVS found in AMI criterion passed
        #-------------------------
        dflt_check_found_ami_for_all_SNs_kwargs = dict(
            t_search_min_max=[out_t_beg-pd.Timedelta('1hour'), out_t_end+pd.Timedelta('1hour')], 
            requirement='all', 
            time_col=ami_df_info_dict['t_int_beg_col'], 
            value_col=ami_df_info_dict['value_col'], 
            SN_col=ami_df_info_dict['SN_col'], 
        )
        check_found_ami_for_all_SNs_kwargs = Utilities.supplement_dict_with_default_values(
            check_found_ami_for_all_SNs_kwargs, 
            dflt_check_found_ami_for_all_SNs_kwargs, 
            inplace=False
        )
        # Make sure the user didn't accidentally set time_col, value_col, or SN_col incorrectly
        assert(check_found_ami_for_all_SNs_kwargs['time_col']==ami_df_info_dict['t_int_beg_col'])
        assert(check_found_ami_for_all_SNs_kwargs['value_col']==ami_df_info_dict['value_col'])
        assert(check_found_ami_for_all_SNs_kwargs['SN_col']==ami_df_info_dict['SN_col'])

        check_found_ami_for_all_SNs_kwargs['df'] = ami_df_i

        #-------------------------
        # Run ami_df_i through DOVSAudit.check_found_ami_for_all_SNs_in_outage
        all_found_ami_i = DOVSAudit.check_found_ami_for_all_SNs_in_outage(**check_found_ami_for_all_SNs_kwargs)

        # Since minimum percentage of PNs already passed, all_found_ami_i can simply be returned here
        return all_found_ami_i
    
    def self_assess_outage_inclusion_requirements(
        self, 
        max_pct_PNs_missing_allowed=0, 
        check_found_ami_for_all_SNs_kwargs=None
    ):
        r"""
        """
        assert(self.ami_df_i.shape[0]>0)
        assert(self.dovs_df_i.shape[0]>0)
        #-------------------------
        to_include = DOVSAudit.assess_outage_inclusion_requirements(
            ami_df_i                           = self.ami_df_i, 
            outg_rec_nb                        = self.outg_rec_nb, 
            dovs_df                            = self.dovs_df_i, 
            max_pct_PNs_missing_allowed        = max_pct_PNs_missing_allowed, 
            ami_df_info_dict                   = self.ami_df_info_dict, 
            dovs_df_info_dict                  = self.dovs_df_info_dict, 
            check_found_ami_for_all_SNs_kwargs = check_found_ami_for_all_SNs_kwargs
        )
        return to_include
    
    
    @staticmethod
    def identify_outg_rec_nbs_to_remove(
        paths, 
        slicers, 
        verbose=False
    ):
        r"""
        NOT USED MUCH ANYMORE
        Probably use DOVSAudit.assess_outage_inclusion_requirements.
        But, I suppose if one wanted to simply assess which outages to include and which to remove from
          a list of paths, this would be the function to use!
        """
        #-------------------------
        # First, need to iterate through paths to retrieve list of all outg_rec_nbs
        # This is necessary as outages can be split across multiple files, so one cannot
        #   simply iterate through the files
        paths=natsorted(paths)
        outg_rec_nbs_in_files = dict()
        for path in paths:
            assert(path not in outg_rec_nbs_in_files.keys())
            df = GenAn.read_df_from_csv(path)
            outg_rec_nbs_in_files[path] = df['OUTG_REC_NB_GPD_FOR_SQL'].unique().tolist()
        outg_rec_nb_to_files_dict = DOVSAudit.invert_file_to_outg_rec_nbs_dict(outg_rec_nbs_in_files)
        all_outg_rec_nbs = list(outg_rec_nb_to_files_dict.keys())

        #-------------------------
        # Build dovs_df
        dovs = DOVSOutages(
            df_construct_type=DFConstructType.kRunSqlQuery, 
            contstruct_df_args=None, 
            init_df_in_constructor=True,
            build_sql_function=DOVSOutages_SQL.build_sql_outage, 
            build_sql_function_kwargs=dict(
                outg_rec_nbs=all_outg_rec_nbs, 
                field_to_split='outg_rec_nbs', 
                include_premise=True
            ), 
            build_consolidated=True
        )
        dovs_df = dovs.df.copy()

        #-------------------------
        # Now, iterate through all outages
        outg_rec_nbs_to_remove = []
        for i_outg, outg_rec_nb in enumerate(all_outg_rec_nbs):
            if verbose:
                print(f'\n\ti_outg: {i_outg+1}/{len(all_outg_rec_nbs)}')
                print(f'\toutg_rec_nb = {outg_rec_nb}')
            ami_df = GenAn.read_df_from_csv_batch(outg_rec_nb_to_files_dict[outg_rec_nb])

            #--------------------------------------------------
            ami_df_i = ami_df[ami_df['OUTG_REC_NB_GPD_FOR_SQL']==outg_rec_nb].copy()     

            # Although I cannot yet call DOVSAudit.choose_best_slicer_and_perform_slicing and DOVSAudit.reduce_INSTV_ABC_1_vals_in_df, 
            #   as the standard cleaning and conversions must be done first, I am able to cut down the size of
            #   ami_df_i by joining the slicers with 'or' statements.
            # Thus, ami_df_i will be reduced to only the subset of data which will be considered in 
            #   DOVSAudit.choose_best_slicer_and_perform_slicing
            # As mentioned, this will cut down the size of ami_df_i and will also save time and resources by not having
            #   to run entire DF through cleaning and conversions procedures.
            ami_df_i = DFSlicer.combine_slicers_and_perform_slicing(
                df=ami_df_i, 
                slicers=slicers, 
                join_slicers='or'
            )
            if ami_df_i.shape[0]==0:
                outg_rec_nbs_to_remove.append(outg_rec_nb)
                continue        

            #--------------------------------------------------
            ami_df_i = AMINonVee.perform_std_initiation_and_cleaning(ami_df_i)
            #-----
            # Should the following be added to AMINonVee.perform_std_initiation_and_cleaning?
            ami_df_i = Utilities_dt.strip_tz_info_and_convert_to_dt(
                df=ami_df_i, 
                time_col='starttimeperiod', 
                placement_col='starttimeperiod_local', 
                run_quick=True, 
                n_strip=6, 
                inplace=False
            )
            ami_df_i = Utilities_dt.strip_tz_info_and_convert_to_dt(
                df=ami_df_i, 
                time_col='endtimeperiod', 
                placement_col='endtimeperiod_local', 
                run_quick=True, 
                n_strip=6, 
                inplace=False
            )
            #--------------------------------------------------
            ami_df_i = DOVSAudit.choose_best_slicer_and_perform_slicing(
                df=ami_df_i, 
                slicers=slicers, 
                groupby_SN=True, 
                t_search_min_max=None, 
                time_col='starttimeperiod_local', 
                value_col=None, 
                SN_col='serialnumber', 
                return_sorted=True
            )

            ami_df_i = DOVSAudit.reduce_INSTV_ABC_1_vals_in_df(
                df=ami_df_i, 
                value_col='value', 
                aep_derived_uom_col='aep_derived_uom', 
                aep_srvc_qlty_idntfr_col='aep_srvc_qlty_idntfr', 
                output_aep_srvc_qlty_idntfr = 'INSTV(ABC)1'
            )

            if ami_df_i.shape[0]==0:
                outg_rec_nbs_to_remove.append(outg_rec_nb)
                continue

            to_include_i = DOVSAudit.assess_outage_inclusion_requirements(
                ami_df_i=ami_df_i, 
                outg_rec_nb=outg_rec_nb, 
                dovs_df=dovs_df, 
                max_pct_PNs_missing_allowed=0

            )
            if not to_include_i:
                outg_rec_nbs_to_remove.append(outg_rec_nb)
        #--------------------------------------------------
        return outg_rec_nbs_to_remove
    
    #****************************************************************************************************
    # End device events (ede) methods
    #****************************************************************************************************
    @staticmethod
    def get_pd_pu_times_for_meter(
        ede_df_i, 
        t_search_min_max=None, 
        pd_ids=['3.26.0.47', '3.26.136.47'], 
        pu_ids=['3.26.0.216', '3.26.136.216'], 
        SN_col='serialnumber', 
        valuesinterval_col='valuesinterval_local', 
        edetypeid_col='enddeviceeventtypeid'
    ):
        r"""
        Get the power-down (pd) and power-up (pu) events from ede_df_i within t_search_min_max.
        ede_df_i must be a meter events DataFrame containing data from a single meter.

        Returns a dict with two keys, pd_times and pu_times, the values of which are lists containing
          the associated times.
        """
        #-------------------------
        assert(ede_df_i[SN_col].nunique()==1)
        #-------------------------
        # Make sure valuesinterval_col contains time data
        if not is_datetime64_dtype(ede_df_i[valuesinterval_col]):
            ede_df_i[valuesinterval_col] = pd.to_datetime(ede_df_i[valuesinterval_col])
        ede_df_i = ede_df_i.sort_values(by=[valuesinterval_col])

        #------------------------- 
        if t_search_min_max is not None:
            assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]))
            assert(len(t_search_min_max)==2)
            #-----
            # At least one of t_search_min_max should not be None
            assert(t_search_min_max[0] is not None or t_search_min_max[1] is not None)
            if t_search_min_max[0] is None:
                t_search_min_max[0] = pd.Timestamp.min
            if t_search_min_max[1] is None:
                t_search_min_max[1] = pd.Timestamp.max
            #-----
            ede_df_i = ede_df_i[
                (ede_df_i[valuesinterval_col]>=t_search_min_max[0]) & 
                (ede_df_i[valuesinterval_col]<=t_search_min_max[1])
            ]

        #------------------------- 
        ede_df_i = ede_df_i.sort_values(by=valuesinterval_col)
        #-----
        pd_times = ede_df_i[ede_df_i[edetypeid_col].isin(pd_ids)][valuesinterval_col].tolist()
        pu_times = ede_df_i[ede_df_i[edetypeid_col].isin(pu_ids)][valuesinterval_col].tolist()
        #------------------------- 
        return dict(
            pd_times=pd_times, 
            pu_times=pu_times
        )
    
    @staticmethod
    def estimate_broad_outage_time_using_ede(
        ede_df_i, 
        dovs_out_t_beg, 
        dovs_out_t_end, 

        pd_ids=['3.26.0.47', '3.26.136.47'], 
        pu_ids=['3.26.0.216', '3.26.136.216'], 

        return_pct_SNs_close=False, 
        out_time_thresh=pd.Timedelta('10 minutes'), 

        SN_col='serialnumber', 
        valuesinterval_col='valuesinterval_local', 
        edetypeid_col='enddeviceeventtypeid', 
        verbose=False
    ):
        r"""
        This function essentially finds the closest power-down event to dovs_out_t_beg and the 
          closest power-up event to dovs_out_t_end
        NOTE: I use the terms power-up and power-down events broadly here.  
              In general, these also contain reasons such as last gasps, power restore, etc.
              What is considered power-up is defined by pu_ids, and what is considered power-down is defined by pd_ids,
                where pu_ids and pd_ids contain enddeviceeventtypeids.
              Might want to use 'reason' column and run a further reduce method, but it seems enddeviceeventtypeids will
                work well for this purpose.

        This can be used with an ede_df_i containing data for all meters in an outage, or for an ede_df_i containing
          events for a single meter.
        If ede_df_i contains data from multiple meters, one can check the percentage of SNs with a power-down close to the
          found outage begin time and the percentage with a power-up close to the found outage end time.
          The definition of 'close' is set by out_time_thresh
        """
        #-------------------------
        # The methodology relies on ede_df_i being sorted by time
        # But, first, make sure valuesinterval_col contains time data
        if not is_datetime64_dtype(ede_df_i[valuesinterval_col]):
            ede_df_i[valuesinterval_col] = pd.to_datetime(ede_df_i[valuesinterval_col])
        ede_df_i = ede_df_i.sort_values(by=[valuesinterval_col])
        #-------------------------
        # Get the power-up and power-down subsets from ede_df_i
        ede_df_i_pu = ede_df_i[ede_df_i[edetypeid_col].isin(pu_ids)]
        ede_df_i_pd = ede_df_i[ede_df_i[edetypeid_col].isin(pd_ids)]
        if ede_df_i_pd.shape[0]==0 or ede_df_i_pu.shape[0]==0:
            if return_pct_SNs_close:
                return [], 0., 0.
            else:
                return []
        #-----
        # Get the time values for the power-up and power-down events
        vals_pu = ede_df_i_pu[valuesinterval_col].drop_duplicates().sort_values().tolist()
        vals_pd = ede_df_i_pd[valuesinterval_col].drop_duplicates().sort_values().tolist()

        #--------------------------------------------------
        # Estimate beginning of outage, out_t_beg
        #--------------------------------------------------
        try:
            out_t_beg_le = Utilities.find_le(vals_pd, dovs_out_t_beg)
        except:
            if verbose:
                print('Warning: Unable to find out_t_beg_le, so using vals_pd[0]')
            out_t_beg_le = vals_pd[0]
        #-----
        try:
            out_t_beg_ge = Utilities.find_ge(vals_pd, dovs_out_t_beg)
        except:
            if verbose:
                print('Warning: Unable to find out_t_beg_ge, so using out_t_beg_le')
            out_t_beg_ge = out_t_beg_le
        #-----
        if out_t_beg_le==out_t_beg_ge:
            out_t_beg = out_t_beg_le
        else:
            # Note: If out_t_beg_le and out_t_beg_ge are equidistant from dovs_out_t_beg
            #         out_t_beg_le is favored (through <= operator)
            if (dovs_out_t_beg-out_t_beg_le) <= (out_t_beg_ge-dovs_out_t_beg):
                out_t_beg = out_t_beg_le
            else:
                out_t_beg = out_t_beg_ge

        #--------------------------------------------------
        # Estimate end of outage, out_t_end
        #--------------------------------------------------
        try:
            out_t_end_ge = Utilities.find_ge(vals_pu, dovs_out_t_end)
        except:
            if verbose:
                print('Warning: Unable to find out_t_end_ge, so using vals_pu[-1]')
            out_t_end_ge = vals_pu[-1]
        #-----
        try:
            out_t_end_le = Utilities.find_le(vals_pu, dovs_out_t_end)
        except:
            if verbose:
                print('Warning: Unable to find out_t_end_le, so using out_t_end_ge')
            out_t_end_le = out_t_end_ge
        #-----
        if out_t_end_le==out_t_end_ge:
            out_t_end = out_t_end_le
        else:
            # Note: If out_t_end_le and out_t_end_ge are equidistant from dovs_out_t_end
            #         out_t_beg_ge is favored (through <= operator)
            if (out_t_end_ge-dovs_out_t_end) <= (dovs_out_t_end-out_t_end_le):
                out_t_end = out_t_end_ge
            else:
                out_t_end = out_t_end_le

        #--------------------------------------------------
        # If out_t_beg is greater than or equal to out_t_end, try changing out_t_beg to out_t_beg_le and/or 
        #   out_t_end to out_t_end_ge
        #-----
        # Before vals were split into vals_pd and vals_pu, it seemed like the only way this could happen was if
        #   out_t_beg equals out_t_end, but now I supposed it could happen in general?
        #   In any case, if the assertion hits below, then I should investigate further
        #-----
        if out_t_beg>=out_t_end:
    #         assert(out_t_end==out_t_beg)
            # Change either out_t_beg to out_t_beg_le or out_t_end to out_t_end_ge first according to which 
            #   is closer (to make minimal adjustment first)
            # Note: abs calls below should really be necessary, but also no harm
            if np.abs(out_t_beg-out_t_beg_le) <= np.abs(out_t_end_ge-out_t_end):
                out_t_beg = out_t_beg_le
            else:
                out_t_end = out_t_end_ge
            #-----
            # If out_t_beg still >= out_t_end, set both out_t_beg=out_t_beg_le and out_t_end=out_t_end_ge
            # Note: Don't know which was set in the above if/else, so simply set both
            if out_t_beg>=out_t_end:
                out_t_beg = out_t_beg_le
                out_t_end = out_t_end_ge
        #-------------------------
        if out_t_beg>=out_t_end:
            if verbose:
                print('In DOVSAudit.estimate_broad_outage_time_using_ede, found out_t_beg<out_t_end.')
                print('Returning []')
            return ()
        assert(out_t_beg<out_t_end)
        #--------------------------------------------------
        if not return_pct_SNs_close:
            return (out_t_beg, out_t_end)
        #--------------------------------------------------
        # Find percentage of SNs with PD close to out_t_beg and percentage with PU close to out_t_end
        n_SNs_w_pd_close_to_beg = ede_df_i_pd[np.abs(ede_df_i_pd[valuesinterval_col]-out_t_beg)<out_time_thresh][SN_col].nunique()
        pct_SNs_w_pd_close_to_beg = 100*n_SNs_w_pd_close_to_beg/ede_df_i[SN_col].nunique()
        #-----
        n_SNs_w_pu_close_to_end = ede_df_i_pu[np.abs(ede_df_i_pu[valuesinterval_col]-out_t_end)<out_time_thresh][SN_col].nunique()
        pct_SNs_w_pu_close_to_end = 100*n_SNs_w_pu_close_to_end/ede_df_i[SN_col].nunique()
        #-------------------------
        return (out_t_beg, out_t_end), pct_SNs_w_pd_close_to_beg, pct_SNs_w_pu_close_to_end


    @staticmethod
    def estimate_outage_times_using_ede_for_meter(
        ede_df_i, 
        broad_out_t_beg, 
        broad_out_t_end, 
        expand_search_time=pd.Timedelta('10 minutes'), 

        pd_ids=['3.26.0.47', '3.26.136.47'], 
        pu_ids=['3.26.0.216', '3.26.136.216'], 

        SN_col='serialnumber', 
        valuesinterval_col='valuesinterval_local', 
        edetypeid_col='enddeviceeventtypeid', 
        verbose=False
    ):
        r"""
        Given the broad outage time, find any times during the main outage time when power was restored for a single meter.
        NOTE: ede_df_i must be a meter events DataFrame containing data from a single meter.

        Essentially, I am finding any sub-outages within the outage as a whole
        This only makes sense at a serial number level (i.e., ede_df_i must contain events for only a single SN)
        This method, in contrast to the work I put together for Patrick, doesn't care about mismatched power-up/power-down 
          pairs (i.e., e.g., multiple power down events can occur in a row)
        Basically, first find a power down event, then find the next power up.  Call this the first sub-outage
        Then, find the next power down after the first sub-outage, then find the next power up.  Call this the second sub-outage.
        Repeat.
        -----    
        """
        #-------------------------
        # As described above, designed to work for DF containing data from a single meter.
        assert(ede_df_i[SN_col].nunique()==1)
        #-------------------------
        # First, get a better estimate for the broad outage times (which are usually estimated from all meters in an outage 
        #  or simply taken from DOVS)
        out_t_beg_end = DOVSAudit.estimate_broad_outage_time_using_ede(
            ede_df_i=ede_df_i, 
            dovs_out_t_beg=broad_out_t_beg-expand_search_time, 
            dovs_out_t_end=broad_out_t_end+expand_search_time, 

            pd_ids=pd_ids, 
            pu_ids=pu_ids, 

            return_pct_SNs_close=False, 
            out_time_thresh=pd.Timedelta('10 minutes'), 

            SN_col=SN_col, 
            valuesinterval_col=valuesinterval_col, 
            edetypeid_col=edetypeid_col, 
            verbose=verbose
        )
        if out_t_beg_end:
            assert(len(out_t_beg_end)==2)
            out_t_beg = out_t_beg_end[0]
            out_t_end = out_t_beg_end[1]
        else:
            return []
    #     return [[out_t_beg, out_t_end]]
        #-------------------------
        # Get the subset of ede_df_i within the broad outage time (including the outage beginning and end) having power-up
        #   or power-down type IDs.
        ede_df_i_sub = ede_df_i[
            (ede_df_i[valuesinterval_col]>=out_t_beg) & 
            (ede_df_i[valuesinterval_col]<=out_t_end) &
            (ede_df_i[edetypeid_col].isin(pd_ids+pu_ids))
        ].copy()
        #-------------------------
        # The methodology relies on ede_df_i_sub being sorted by time
        # But, first, make sure valuesinterval_col contains time data
        if not is_datetime64_dtype(ede_df_i_sub[valuesinterval_col]):
            ede_df_i_sub[valuesinterval_col] = pd.to_datetime(ede_df_i_sub[valuesinterval_col])
        ede_df_i_sub = ede_df_i_sub.sort_values(by=[valuesinterval_col])
        #-------------------------
        # The procedure is to essentially mark and power-ups with a value of 1 and any power-downs with a value of 0
        # These values are stored in a temporary column (tmp_id_type_col).
        # Then, .diff can be called on this column, and the results (stored in tmp_diff_col) can be used to identify
        #   the beginning and end of the sub-outages.
        #-----
        tmp_id_type_col = Utilities.generate_random_string()
        ede_df_i_sub[tmp_id_type_col] = np.nan
        ede_df_i_sub.loc[ede_df_i_sub[edetypeid_col].isin(pu_ids), tmp_id_type_col] = 1
        ede_df_i_sub.loc[ede_df_i_sub[edetypeid_col].isin(pd_ids), tmp_id_type_col] = 0
        #-----
        tmp_diff_col = Utilities.generate_random_string()
        ede_df_i_sub[tmp_diff_col] = ede_df_i_sub[tmp_id_type_col].diff()
        #-------------------------
        # By definition, the first entry should be a power down event and the last should be a power up event
        #  (because of the use of >= and <= in ede_df_i_sub declaration above)
        assert(ede_df_i_sub.shape[0]>=2)
        #-----
        # If first value is not in pd_ids, remove until found
        if ede_df_i_sub.iloc[0][edetypeid_col] not in pd_ids:
            if verbose:
                print('Warning, first value of ede_df_i_sub not in pd_ids')
            while ede_df_i_sub.iloc[0][edetypeid_col] not in pd_ids:
                ede_df_i_sub = ede_df_i_sub.iloc[1:]
        # If last value is not in pu_ids, remove until found
        if ede_df_i_sub.iloc[-1][edetypeid_col] not in pu_ids:
            if verbose:
                print('Warning, last value of ede_df_i_sub not in pu_ids')
            while ede_df_i_sub.iloc[-1][edetypeid_col] not in pu_ids:
                ede_df_i_sub = ede_df_i_sub.iloc[:-1]
        #-----
        if ede_df_i_sub.shape[0]<2:
            print(f'Could not determine outage times for SN={ede_df_i[SN_col].unique()[0]}')
            return []
        assert(ede_df_i_sub.shape[0]>=2)
        assert(ede_df_i_sub.iloc[0][edetypeid_col] in pd_ids)
        assert(ede_df_i_sub.iloc[-1][edetypeid_col] in pu_ids)
        if ede_df_i_sub.shape[0]==2:
            outg_times = [(ede_df_i_sub.iloc[0][valuesinterval_col], ede_df_i_sub.iloc[-1][valuesinterval_col])]
            # return outg_times
        #-------------------------
        # The .diff operation above leaves the first entry (which is a power-down event) as NaN
        # Make it equal to -1 instead, like all other power-down events
        ede_df_i_sub.loc[ede_df_i_sub.index[0], tmp_diff_col]=-1

        # Get the sub-outages begin (diff = -1) and end (diff = +1) times
        outg_times_beg = ede_df_i_sub[ede_df_i_sub[tmp_diff_col]==-1][valuesinterval_col].tolist()
        outg_times_end = ede_df_i_sub[ede_df_i_sub[tmp_diff_col]==1][valuesinterval_col].tolist()
        assert(len(outg_times_beg)==len(outg_times_end))

        # Combine the begin and end times
        outg_times = list(zip(outg_times_beg, outg_times_end))
        #-------------------------
        # Sanity check on outage times
        for i_outg, outg_time in enumerate(outg_times):
            # Each element should contain a beginning and ending time
            assert(len(outg_time)==2)
            # The beginning time should occur before the ending time
            assert(outg_time[0]<=outg_time[1])
            # This outage should occur after the last outage
            if i_outg>0:
                assert(outg_time[0]>=outg_times[i_outg-1][1])
        #-------------------------
        return outg_times


    @staticmethod
    def estimate_cmi_using_ede_for_meter(
        ede_df_i, 
        broad_out_t_beg, 
        broad_out_t_end, 
        expand_search_time=pd.Timedelta('1 hour'), 

        pd_ids=['3.26.0.47', '3.26.136.47'], 
        pu_ids=['3.26.0.216', '3.26.136.216'], 

        SN_col='serialnumber', 
        valuesinterval_col='valuesinterval_local', 
        edetypeid_col='enddeviceeventtypeid', 
        return_est_outage_times=False, 
        verbose=False
    ):
        r"""
        Estimate the outage times using DOVSAudit.estimate_outage_times_using_ede_for_meter, then calculate the CMI for the customer
        """
        #-------------------------
        outg_times = DOVSAudit.estimate_outage_times_using_ede_for_meter(
            ede_df_i=ede_df_i, 
            broad_out_t_beg=broad_out_t_beg, 
            broad_out_t_end=broad_out_t_end,
            expand_search_time=expand_search_time, 
            pd_ids=pd_ids, 
            pu_ids=pu_ids, 
            SN_col=SN_col, 
            valuesinterval_col=valuesinterval_col, 
            edetypeid_col=edetypeid_col, 
            verbose=verbose
        )
        #-------------------------
        cmi_i = pd.Timedelta(0)
        for outg_time in outg_times:
            cmi_i += (outg_time[1]-outg_time[0])
        #-------------------------
        if return_est_outage_times:
            return cmi_i, outg_times
        else:
            return cmi_i
        
        
    #****************************************************************************************************
    # AMI nonvee (ami) methods
    #****************************************************************************************************
    @staticmethod
    def check_for_data_in_search_window(
        df_i, 
        t_search_min_max=None, 
        t_int_beg_col='starttimeperiod_local', 
        t_int_end_col='endtimeperiod_local'
    ):
        r"""
        Checks whether or not data are present within the search window
        """
        #-------------------------
        if t_search_min_max is not None:
            assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]))
            assert(len(t_search_min_max)==2)
            #-----
            # At least one of t_search_min_max should not be None
            assert(t_search_min_max[0] is not None or t_search_min_max[1] is not None)
            if t_search_min_max[0] is None:
                t_search_min_max[0] = pd.Timestamp.min
            if t_search_min_max[1] is None:
                t_search_min_max[1] = pd.Timestamp.max
            #-----
            return df_i[
                (df_i[t_int_beg_col]>=t_search_min_max[0]) & 
                (df_i[t_int_end_col]<=t_search_min_max[1])
            ].shape[0]>0
        else:
            return df_i.shape[0]>0

    @staticmethod
    def estimate_outage_times_for_meter(
        df_i, 
        t_search_min_max=None, 
        t_int_beg_col='starttimeperiod_local', 
        t_int_end_col='endtimeperiod_local', 
        value_col='value', 
        SN_col='serialnumber', 
        verbose=True,
        outg_rec_nb_col=None
    ):
        r"""
        Try to determine when outages occur by looking for periods of time where the meters show values of 0.
        df_i should contain AMI data for a single meter.
        Designed to find multipe groups of zero readings.

        Returns a list of dict objects, each having keys: cnsrvtv_t_beg, cnsrvtv_t_end, zero_t_beg, zero_t_end.

        IMPORTANT: Due to the 15-minute granularity of the data, the raw output will underestimate the length of the outage
                     between 0-30 minutes (i.e., the outage actually began 0-15 minutes before AMI data received showing meters
                     with 0 values and the outage actually ended 0-15 minutes after last data-point received with meters 
                     showing 0 values).
                    ACTUALLY, to be 100% correct, the raw output will underestimate the length of the outage by between 
                      0 and 2*freq of data!!

        REMINDER: The original method, which found zero times and combined them using Utilities.get_fuzzy_overlap, 
                    was no good because if there are gaps in the data multiple sub-outages would be split off
                  e.g., if a period of zero values lasts for an hour, but there is a 15-minute interval missing data,
                    the original method would have split this into two sub-outages

        t_search_min_max:
            Allows the user to set of time subset of the data from which to search for the outages.
            If t_search_min_max is not None, it should be a two-element list/tuple.
            However, one of the elements of the list may be None if, e.g., one wants limits on only the min or max

        outg_rec_nb_col:
            Only used for outputting warnings etc. when verbose==True
        """
        #-------------------------
        # If df_i doesn't have any data within search window, return []
        if not DOVSAudit.check_for_data_in_search_window(
            df_i=df_i, 
            t_search_min_max=t_search_min_max, 
            t_int_beg_col=t_int_beg_col, 
            t_int_end_col=t_int_end_col
        ):
            return []
        #-------------------------
        # As described above, designed to work for DF containing data from a single meter.
        assert(df_i[SN_col].nunique()==1)
        SN_i = df_i[SN_col].unique().tolist()[0] # Used only for error messages

        # outg_rec_nb only used for error messages
        if outg_rec_nb_col is not None and outg_rec_nb_col in df_i.columns.tolist():
            outg_rec_nb = df_i[outg_rec_nb_col].unique().tolist()[0]
        else:
            outg_rec_nb = None

        #-------------------------
        # For this procedure, only really need the t_int_beg_col, t_int_end_col, value_col (and, SN_col for assertion check)
        df_i = df_i.drop(
            columns=list(set(df_i.columns.tolist()).difference(set([t_int_beg_col, t_int_end_col, value_col, SN_col]))), 
            inplace=False
        )

        #-------------------------
        # First, make sure df_i is sorted by t_int_beg_col!
        df_i = df_i.sort_values(by=t_int_beg_col)

        #-------------------------
        # It is easier to work with the index in terms of numbers between 0 and df_i.shape[0], therefore
        #   I call reset_index()
        # NOTE: Is storing idx_col really necessary?  I'm thinking it's not...
        assert(df_i.index.nlevels==1)
        if df_i.index.name is not None and df_i.index.name not in df_i.columns.tolist():
            idx_col = df_i.index.name
        else:
            idx_col = Utilities.generate_random_string()
            df_i.index.name = idx_col
        assert(idx_col not in df_i.columns.tolist())
        #-----
        df_i = df_i.reset_index()


        #-------------------------
        # Find the rows in df_i in which to search for outages
        if t_search_min_max is not None:
            assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]))
            assert(len(t_search_min_max)==2)
            #-----
            # At least one of t_search_min_max should not be None
            assert(t_search_min_max[0] is not None or t_search_min_max[1] is not None)
            #-----
            if t_search_min_max[0] is None:
                idx_search_min = 0
            else:
                idx_search_min = df_i[df_i[t_int_beg_col]>=t_search_min_max[0]].index[0]
            #-----
            if t_search_min_max[1] is None:
                idx_search_max = df_i.shape[0]-1
            else:
                idx_search_max = df_i[df_i[t_int_end_col]<=t_search_min_max[1]].index[-1]
        else:
            idx_search_min = 0
            idx_search_max = df_i.shape[0]-1
            t_search_min_max=[None, None] # Needed only for verbose printing below

        #--------------------------------------------------
        # If there is no power (value==0) at the beginning or end of the search window
        #   expand the search window
        #-----
        # NOTE: Below, I want sub_df_i to include idx_search_min and idx_search_max, which is why
        #         the +1 is needed with idx_search_min
        # NOTE: Even if idx_search_max was set to df_i.shape[0]-1, there not no harm
        #         in calling df_i.iloc[i:df_i.shape[0]], whereas calling df_i.iloc[df_i.shape[0]]
        #         would result in an index out-of-bound error!
        #-------------------------
        # If no power at the beginning of the search window
        open_beg = False
        if df_i.iloc[idx_search_min][value_col]==0:
            # Portion of df_i previous to search window
            df_i_pre = df_i.iloc[:idx_search_min]
            if (df_i_pre[value_col]!=0).any():
                # I want the non-zero value closest to the search window, which, in this case
                #   means the largest index having a non-zero value
                # To achieve this, reverse the order of df_i_pre and use idxmax
                # NOTE: This only works because the indices are integers (and argmax would give the
                #       wrong answer in any case)!  
                idx_search_min = (df_i_pre.iloc[::-1][value_col]!=0).idxmax()
                if verbose:
                    print('\n\nExpanding search window to include beginning of first outage')
            else:
                idx_search_min = 0
                open_beg = True
                if verbose:
                    print('\n\nWARNING: could not find beginning of first outage!')
                    print('Search window expanded to include minimum time of full df_i')
            #-----
            if verbose:
                print(f'SN = {SN_i}')
                if outg_rec_nb is not None:
                    print(f'OUTG_REC_NB = {outg_rec_nb}')
                print(f'\tOrg = {t_search_min_max[0]}\n\tNew = {df_i.iloc[idx_search_min][t_int_beg_col]}')
        #-------------------------
        # If no power at the end of the search window
        open_end = False
        if df_i.iloc[idx_search_max][value_col]==0:
            # Portion of df_i following search window
            df_i_post = df_i.iloc[idx_search_max+1:]
            if (df_i_post[value_col]!=0).any():
                # I want the non-zero value closest to the search window, which, in this case
                #   means the smallest index having a non-zero value
                idx_search_max = (df_i_post[value_col]!=0).idxmax()
                if verbose:
                    print('\n\nExpanding search window to include ending of last outage')
            else:
                idx_search_max = df_i.shape[0]-1
                open_end = True
                if verbose:
                    print('\n\nWARNING: could not find ending of last outage')
                    print('Search window expanded to include maximum time of full df_i')
            #-----
            if verbose:
                print(f'SN = {SN_i}')
                if outg_rec_nb is not None:
                    print(f'OUTG_REC_NB = {outg_rec_nb}')
                print(f'\tOrg = {t_search_min_max[1]}\n\tNew = {df_i.iloc[idx_search_max][t_int_end_col]}')
        #-------------------------
        sub_df_i = df_i.iloc[idx_search_min:idx_search_max+1].copy()
        #--------------------------------------------------
        #-------------------------
        # If no periods with zero values, return empty list
        if not (sub_df_i[value_col]==0).any():
            return []

        #-------------------------
        # To be consistent with methods in DOVSAudit.estimate_outage_times_using_ede_for_meter, where power-up events are assigned
        #   a value of +1 and power-down event assigned a value of 0, I should use !=0 below
        tmp_neq0_col = Utilities.generate_random_string()
        tmp_diff_col = Utilities.generate_random_string()
        #-----
        sub_df_i[tmp_neq0_col] = (sub_df_i[value_col] != 0).astype(int)
        sub_df_i[tmp_diff_col] = sub_df_i[tmp_neq0_col].diff()
        #-------------------------
        # The .diff() operation always leaves the first element as a NaN
        # However, if the first element was the beginning of an outage (i.e., if the value==0)
        #   then the diff should be +1
        if sub_df_i.iloc[0][value_col]==0:
            sub_df_i.loc[sub_df_i.index[0], tmp_diff_col]=-1  

        #-------------------------
        # Find the power-down (pd) and power-up (pu) rows
        #-----
        # Outage times are defined by a power loss followed by a power restore.
        # The beginning of a power loss occurs when the value in value_col goes from non-zero to zero
        #   and is located in rows for which sub_df_i[tmp_diff_col]==-1
        # The ending of a power loss occurs when the value in value_col goes from zero to non-zero
        #   and is located in rows for which sub_df_i[tmp_diff_col]==+1
        # NOTE: The use of .reset_index() below ensures that the values returned for rows are numerical index
        #         values from 0 to sub_df_i.shape[0]-1
        # NOTE: Below pd==power-down and pu==power-up.  The pd times are when the first zero values occur (following non-zero
        #         values) and the pu times are when the first non-zero values occur (following zero values)
        #-----
        pd_row_idxs = sub_df_i.reset_index()[sub_df_i.reset_index()[tmp_diff_col]==-1].index.tolist()
        pu_row_idxs = sub_df_i.reset_index()[sub_df_i.reset_index()[tmp_diff_col]==1].index.tolist()
        #-----
        # Sanity checks
        assert((sub_df_i.iloc[pd_row_idxs][tmp_diff_col]==-1).all())
        assert((sub_df_i.iloc[pd_row_idxs][value_col]==0).all())
        #-----
        assert((sub_df_i.iloc[pu_row_idxs][tmp_diff_col]==1).all())
        assert((sub_df_i.iloc[pu_row_idxs][value_col]!=0).all())
        #-------------------------    

        #-------------------------
        # As stated above, the pd times are when the first zero values occur (following non-zero
        #   values) and the pu times are when the first non-zero values occur (following zero values)
        #-----
        # pd_row_idxs/zeros_beg_row_idxs/cnsrvtv_beg_row_idxs:
        #   - the idxs denoting when the periods of zeros begin (zeros_beg_row_idxs) is simply pd_row_idxs.
        #     At these times, we know for certain the outage is ongoing, so these are essentially the maximum
        #       times at which the outages could begin
        #   - the conservative estimates for the beginning of the outages is the time period preceding the first zero,
        #       which should be a non-zero value.
        #     At these times, we know for certain (unless the outage is ongoing at the beginning of the data) the outage is not 
        #       ongoing, so these are essentially the minumum times at which the outages could begin
        #     If the first outage is ongoing at the beginning of the data, then zeros_beg_row_idxs[0]==0, in which case the best
        #       we can do is set the conservative row equal to 0 as well
        #-----
        # pu_row_idxs/zeros_end_row_idxs/cnsrvtv_end_row_idxs:
        #   - the idxs denoting when a period of zeros ends with the first non-zero value (cnsrvtv_end_row_idxs) is simply pu_row_idxs.
        #     At these times, we know for certain the outages are no longer ongoin (unless the outage is ongoing at the end of the data).
        #     These are essentially the maximum times at which the outages could end.
        #   - The last zero value (zeros_end_row_idxs) in a group of zeros should be the time period preceding the first non-zero (which 
        #       should be a zero value).
        #     At these times, we know for certain the outage is ongoing, so these are essentially the minimum times at which the outages
        #       could end.
        #----------
        zeros_beg_row_idxs   = pd_row_idxs
        cnsrvtv_beg_row_idxs = [x-1 if x>0 else 0 for x in pd_row_idxs]
        #-----
        cnsrvtv_end_row_idxs = pu_row_idxs
        zeros_end_row_idxs = [x-1 if x>0 else 0 for x in pu_row_idxs]
        #-------------------------

        #-------------------------
        # All the conservative estimates should have non-zero values
        # NOTE: Cannot simply include, e.g.,  only values from cnsrvtv_beg_row_idxs which
        #         are greater than zero, as the first pd could be at i=1, making
        #         i=0 a perfectly acceptable value
        #       Basically, one needs to include those in cnsrvtv_beg_row_idxs not 
        #         equal to their counterparts in pd_row_idxs
        assert((sub_df_i.iloc[
            [cnsrvtv_beg_row_idxs[i] for i in range(len(pd_row_idxs)) 
             if cnsrvtv_beg_row_idxs[i]!=pd_row_idxs[i]]
        ][value_col]!=0).all())
        assert((sub_df_i.iloc[cnsrvtv_end_row_idxs][value_col]!=0).all())
        #-----
        # Similarly, the zero times should have zero values!
        assert((sub_df_i.iloc[zeros_beg_row_idxs][value_col]==0).all())
        assert((sub_df_i.iloc[zeros_end_row_idxs][value_col]==0).all())

        #-------------------------
        # Sanity check: all conservative beg(end) values should have matching zeros beg(end) value
        assert(len(cnsrvtv_beg_row_idxs)==len(zeros_beg_row_idxs))
        assert(len(zeros_end_row_idxs)==len(cnsrvtv_end_row_idxs))
        #-------------------------
        # In the case where an outage is ongoing at the end of the data, there will be
        #   one more beginning time than ending time.
        # In such a case, the best we can do is set the ending times (zeros and conservative)
        #   equal to the end of the data
        # If verbose, the user will have already been notified of this
        if len(zeros_end_row_idxs)<len(zeros_beg_row_idxs):
            assert(sub_df_i.iloc[-1][value_col]==0)
            zeros_end_row_idxs.append(sub_df_i.shape[0]-1)
            cnsrvtv_end_row_idxs.append(sub_df_i.shape[0]-1)
        assert(len(zeros_end_row_idxs)==len(zeros_beg_row_idxs))

        #-------------------------
        # Convert the row indices to times
        cnsrvtv_beg_times = sub_df_i.iloc[cnsrvtv_beg_row_idxs][t_int_beg_col].tolist()
        zeros_beg_times   = sub_df_i.iloc[zeros_beg_row_idxs][t_int_beg_col].tolist()
        #-----
        cnsrvtv_end_times = sub_df_i.iloc[cnsrvtv_end_row_idxs][t_int_end_col].tolist()
        zeros_end_times   = sub_df_i.iloc[zeros_end_row_idxs][t_int_end_col].tolist()

        # Assertion not really needed
        assert(
            len(cnsrvtv_beg_times)==
            len(zeros_beg_times)==
            len(cnsrvtv_end_times)==
            len(zeros_end_times)
        )
        #-------------------------
        # Construct return_list of dict objects
        return_list = []
        for i_outg in range(len(cnsrvtv_beg_times)):
            outg_dict_i = dict(
                cnsrvtv_t_beg = cnsrvtv_beg_times[i_outg], 
                zero_t_beg    = zeros_beg_times[i_outg], 
                zero_t_end    = zeros_end_times[i_outg], 
                cnsrvtv_t_end = cnsrvtv_end_times[i_outg], 
                open_beg      = False, 
                open_end      = False
            )
            #-----
            if i_outg==0:
                outg_dict_i['open_beg'] = open_beg
            #-----
            if i_outg==len(cnsrvtv_beg_times)-1:
                outg_dict_i['open_end'] = open_end
            #-----
            return_list.append(outg_dict_i)
        #-------------------------
        # Sanity check on outage times
        for i_outg, outg_dict_i in enumerate(return_list):
            # Time ordering should be: cnsrvtv_t_beg, zero_t_beg, zero_t_end, cnsrvtv_t_end
            assert(
                outg_dict_i['cnsrvtv_t_beg']<=
                outg_dict_i['zero_t_beg']<=
                outg_dict_i['zero_t_end']<=
                outg_dict_i['cnsrvtv_t_end']
            )
            # This outage should occur after the last outage
            # NOTE: We are using zero_t_beg/_end instead of cnsrvtv_t_beg/_end.
            #       For the case where a meter regains power briefly (briefly here meaning during a single
            #         15T interval), the cnsrvtv_t_beg of the right outage period will actually be BEFORE the
            #         cnsrvtv_t_end of the left outage.
            #       Visually, this situation will result in a triangular structure in the voltage signal data (zeros
            #         to the left, one single peak, followed by zeros to the right).
            #       We use zero_t_beg/_end because the zero_t_beg of the right outage will always be after
            #         the zero_t_end of the left outage period.
            if i_outg>0:
                assert(outg_dict_i['zero_t_beg']>=return_list[i_outg-1]['zero_t_end'])
        #-------------------------
        return return_list
    
    
    @staticmethod
    def combine_est_outg_times_with_overlap_cnsrvtv(
        est_outg_times,
        overlap_to_enforce=None
    ):
        r"""
        The conservative beginning time of an outage can overlap with the convervative ending time of
          the previous outage.
        This will occur, e.g., when, during an outage, a meter regains power for one interval (visually, this
          corresponds to a triangular shape in the voltage data).
        In such instances, this function can be used to combine the two into a single est_outg_time.

        overlap_to_enforce:
            If not None:
                Should be a pd.Timedelta.
                It is enforced that overlaps between any two est_outg_times must equal this value
                A typical value is overlap_to_enforce=pd.Timedelta('15T')

        """
        #-------------------------
        # We are going to manipulate est_outg_times (e.g., after popping off first element, subsequent changes
        #   to that object will actually change to underlying dict values as well!
        # So make a copy so original is not altered
        est_outg_times = copy.deepcopy(est_outg_times)
        #-------------------------
        if overlap_to_enforce is not None:
            assert(isinstance(overlap_to_enforce, pd.Timedelta))
        #-------------------------
        # First, make sure all elements of est_outg_times are dicts with the expected keys
        expected_keys = ['cnsrvtv_t_beg', 'zero_t_beg', 'zero_t_end', 'cnsrvtv_t_end']
        for est_outg_times_i in est_outg_times:
            assert(isinstance(est_outg_times_i, dict))
            assert(len(set(expected_keys).difference(set(est_outg_times_i.keys())))==0)
        #-----
        # Next, make sure est_outg_times is sorted
        est_outg_times = natsorted(est_outg_times, key=lambda x: x['cnsrvtv_t_beg'])
        #-------------------------
        return_est_outg_times = []
        # Set the first range in return_est_outg_times simply as the first range in the list
        est_outg_times_curr = est_outg_times.pop(0)
        return_est_outg_times.append(est_outg_times_curr)
        #-----
        for est_outg_times_i in est_outg_times:
            if est_outg_times_i['cnsrvtv_t_beg'] > est_outg_times_curr['cnsrvtv_t_end']:
                # cnsrvtv_t_beg_i is after current end, so new interval needed
                return_est_outg_times.append(est_outg_times_i)
                est_outg_times_curr = est_outg_times_i
            else:
                # cnsrvtv_t_beg_i <= current_end, so overlap.
                # The beg of return_est_outg_times[-1] remains the same, 
                #   but the end of return_est_outg_times[-1] should be changed to
                #   the max of current_end and end
                #-----
                if overlap_to_enforce is not None:
                    overlap_i = est_outg_times_curr['cnsrvtv_t_end']-est_outg_times_i['cnsrvtv_t_beg']
                    assert(overlap_i==overlap_to_enforce)
                #-----
                return_est_outg_times[-1]['zero_t_end'] = max(
                    return_est_outg_times[-1]['zero_t_end'], 
                    est_outg_times_i['zero_t_end']
                )
                return_est_outg_times[-1]['cnsrvtv_t_end'] = max(
                    return_est_outg_times[-1]['cnsrvtv_t_end'], 
                    est_outg_times_i['cnsrvtv_t_end']
                )
        return return_est_outg_times
    
    @staticmethod
    def estimate_outage_times(
        df, 
        t_search_min_max=None, 
        pct_SNs_required_for_outage=0, 
        relax_pct_SNs_required_if_no_outgs_found=False, 
        t_int_beg_col='starttimeperiod_local', 
        t_int_end_col='endtimeperiod_local', 
        value_col='value', 
        combine_overlaps=True, 
        verbose=True,
        outg_rec_nb_col=None
    ):
        r"""
        Try to determine when outages occur by looking for periods of time where some minimum percentage
          of the meters (defined by pct_SNs_required_for_outage) show values of 0.
        Designed to find multipe groups of zero readings.
        Strategy: Basically, reduce df down to one entry per timestamp (instead of multiple SNs per timestamp) whose
                    value signifies whether or not the the timestamp can be considered as a zero value or a non-zero value.
                  A throw-away SN_col is then added on so that the DOVSAudit.estimate_outage_times_for_meter method may be used.

        !!!!! IMPORTANT !!!!!
            pct_SNs_required_for_outage is the minimum percentage of available data at each timestamp needed to be zero
              for the timestamp to be considered a zero value.
            In many cases, this is equal to the percentage of SNs with a zero value, BUT NOT ALWAYS.
            If, e.g., one meter in an outage has timestamps offset by a minute compared to the others, this will 
              not be the case!

        Returns a list of dict objects, each having keys: cnsrvtv_t_beg, cnsrvtv_t_end, zero_t_beg, zero_t_end.

        IMPORTANT: Due to the 15-minute granularity of the data, the raw output will underestimate the length of the outage
                     between 0-30 minutes (i.e., the outage actually began 0-15 minutes before AMI data received showing meters
                     with 0 values and the outage actually ended 0-15 minutes after last data-point received with meters 
                     showing 0 values).
                    ACTUALLY, to be 100% correct, the raw output will underestimate the length of the outage by between 
                      0 and 2*freq of data!!

        t_search_min_max:
            Allows the user to set of time subset of the data from which to search for the outages.
            If t_search_min_max is not None, it should be a two-element list/tuple.
            However, one of the elements of the list may be None if, e.g., one wants limits on only the min or max

        outg_rec_nb_col:
            Only used for outputting warnings etc. when verbose==True
        """
        #-------------------------
        # If df doesn't have any data within search window, return []
        if not DOVSAudit.check_for_data_in_search_window(
            df_i=df, 
            t_search_min_max=t_search_min_max, 
            t_int_beg_col=t_int_beg_col, 
            t_int_end_col=t_int_end_col
        ):
            return []
        #------------------------- 
        # outg_rec_nb only used for error messages
        if outg_rec_nb_col is not None and outg_rec_nb_col in df.columns.tolist():
            outg_rec_nb = df[outg_rec_nb_col].unique().tolist()[0]
        else:
            outg_rec_nb = None

        #------------------------- 
        if t_search_min_max is not None:
            assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]))
            assert(len(t_search_min_max)==2)
            #-----
            # At least one of t_search_min_max should not be None
            assert(t_search_min_max[0] is not None or t_search_min_max[1] is not None)
            if t_search_min_max[0] is None:
                t_search_min_max[0] = pd.Timestamp.min
            if t_search_min_max[1] is None:
                t_search_min_max[1] = pd.Timestamp.max
            #-----
            df = df[
                (df[t_int_beg_col]>=t_search_min_max[0]) & 
                (df[t_int_end_col]<=t_search_min_max[1])
            ]

        #-------------------------
        # For this procedure, only really need the t_int_beg_col, t_int_end_col, value_col
        df = df.drop(
            columns=list(set(df.columns.tolist()).difference(set([t_int_beg_col, t_int_end_col, value_col]))), 
            inplace=False
        )

        #-------------------------
        # Make sure df is sorted by t_int_beg_col!
        df = df.sort_values(by=t_int_beg_col)

        #-------------------------
        # NOTE: Cannot use >= below!
        #       A typical value is to set pct_SNs_required_for_outage=0, and if using >= below,
        #         everything would come back as true, regardless of whether or not any 0 values
        #         were registered.
        zeros_df = df.groupby([t_int_beg_col, t_int_end_col]).apply(
            lambda x: 100*((x[value_col]==0).sum()/x.shape[0])>pct_SNs_required_for_outage
        )

        # To be able to feed into DOVSAudit.estimate_outage_times_for_meter, we actually want the opposite of zeros_df,
        #   where non-zero values are assigned 1 and zero values are assigned 0
        neq0_df = (~zeros_df).astype(int)

        #-------------------------
        # At this point, neq0_df is a series with indices equal to starttimeperiod_local and values equal to 0 or 1, 
        #   indicating whether or not the time period as a whole can be treated as value 0 (without power) or 1 (with power)
        # Name the series value_col, so after calling reset_index to column will equal value_col, as needed for input into
        #   DOVSAudit.estimate_outage_times_for_meter
        neq0_df.name = value_col
        neq0_df = neq0_df.reset_index()

        #-------------------------
        # DOVSAudit.estimate_outage_times_for_meter requires a SN_col with one unique value, so satisfy that
        SN_col='SN'
        if outg_rec_nb is None:
            SN_val = 'AGG_OUTAGE'
        else:
            SN_val = f'AGG_OUTAGE {outg_rec_nb}'
        neq0_df[SN_col] = SN_val

        #-------------------------
        # Now, simply feed into DOVSAudit.estimate_outage_times_for_meter
        return_list = DOVSAudit.estimate_outage_times_for_meter(
            df_i=neq0_df, 
            t_search_min_max=t_search_min_max, 
            t_int_beg_col=t_int_beg_col, 
            t_int_end_col=t_int_end_col, 
            value_col=value_col, 
            SN_col=SN_col, 
            verbose=verbose,
            outg_rec_nb_col=outg_rec_nb_col
        )

        #-------------------------
        if len(return_list)==0 and relax_pct_SNs_required_if_no_outgs_found and pct_SNs_required_for_outage>0:
            # NOTE: Important below that input relax_pct_SNs_required_if_no_outgs_found be set to False
            #       OTHERWISE INFINITE LOOP WHEN NO OUTAGE TIMES FOUND!
            print('No outage_times found in DOVSAudit.estimate_outage_times, so relaxing the pct_SNs_required contraint')
            return_list = DOVSAudit.estimate_outage_times(
                df=df, 
                t_search_min_max=t_search_min_max, 
                pct_SNs_required_for_outage=0, 
                relax_pct_SNs_required_if_no_outgs_found=False, 
                t_int_beg_col=t_int_beg_col, 
                t_int_end_col=t_int_end_col, 
                value_col=value_col, 
                verbose=verbose,
                outg_rec_nb_col=outg_rec_nb_col
            )

        #-------------------------
        if combine_overlaps and len(return_list)>0:
            return_list = DOVSAudit.combine_est_outg_times_with_overlap_cnsrvtv(
                est_outg_times=return_list,
                overlap_to_enforce=None
            )

        #-------------------------
        return return_list

    #****************************************************************************************************
    # Combine AMI and EDE
    #****************************************************************************************************
    @staticmethod
    def outg_time_audit(
        outg_est_times_i, 
        dovs_outg_t_beg_end=None, 
        outg_times_ede=None, 
        selection_method='min', 
        return_all_best_ests=False
    ):
        r"""
        Given the zero times for the (sub)outage, frequency of the data, and DOVS outage data and/or EDE outage times,
        return the best estimate for the outage time.

        NOTE: To be certain, this explicitly places preference on the zero times as found by the 15-minute interval.
              The conservative values found using that data are replaced on if dovs and/or ede data supply a value that is
                consistent with the 15-minute interval data (i.e., in the uncertainty intervals, see below for more info).
              Therefore, if dovs or ede complete disagree with 15-minute interval, the latter is always chosen.

        The conservative estimates for the outage time are found by subtracting the frequency of the data (freq) from the first
          zero time and adding freq to the last zero time.
        The uncertainty intervals are defined as:
            [first zero - freq, first zero]
            [last zero,         last zero + freq]
        If dovs and/or ede supply times within the uncertainty region, the values used are chosen according to selection_method: 
            selection_method=='min':       those minimizing the outage time are chosen
            selection_method=='max':       those maximizing the outage time are chosen
            selection_method=='dovs pref': dovs times preferred, meaning dovs times are chosen if both dovs and ede available,
                                             but ede times are taken if available when dovs are not
            selection_method=='dovs only':  dovs times are chosen if available.  
                                             If dovs not available but ede is, ede ignored
            selection_method=='ede pref':  ede times preferred, meaning ede times are chosen if both dovs and ede available
                                             but dovs times are taken if available when ede are not
            selection_method=='ede only':  ede times are chosen if available.  
                                             If ede not available but dovs is, dovs ignored

        outg_times_ede:
            this can be a list of 2-element lists representing multiple outage beg/end pairs (if, e.g., found using
              DOVSAudit.estimate_outage_times_using_ede_for_meter)
            OR
            this can be a dict with two keys, pd_times and pu_times, the values of which are lists containing the 
              associated times (if, e.g., found using DOVSAudit.get_pd_pu_times_for_meter)

        Given the conservative estimates for the outage time (and frequency of the data), and DOVS outage data and/or EDE outage times,
        return the best estimate for the outage time.

        cnsrvtv_beg_end:
            The conservative estimates for the beginning and end of the outage.
            These are typically the period before the first zero and the period after the last zero. 

        NOTE: This assumes the user wants a conservative estimate.! 
                The non-conservative estimate essentially counts to times the meter registered a zero count, so no outside audit needed.
        """
        #-------------------------
        assert(isinstance(outg_est_times_i, dict))
        assert(len(set(outg_est_times_i.keys()).difference(set(['cnsrvtv_t_beg', 'cnsrvtv_t_end', 'zero_t_beg', 'zero_t_end', 'open_beg', 'open_end'])))==0)

        #-------------------------
        # First, find the conservative outage beginning and ending
        cnsrvtv_beg_end = [outg_est_times_i['cnsrvtv_t_beg'], outg_est_times_i['cnsrvtv_t_end']]
        zero_t_beg_end  = [outg_est_times_i['zero_t_beg'], outg_est_times_i['zero_t_end']]
        #-------------------------
        if not dovs_outg_t_beg_end and not outg_times_ede:
            return cnsrvtv_beg_end
        #-------------------------
        assert(selection_method in ['min', 'max', 'dovs pref', 'dovs only', 'ede pref', 'ede only'])
        #----------------------------------------------------------------------------------------------------
        dovs_best_est = [None, None]
        if dovs_outg_t_beg_end:
            # If the outage information from DOVS is between the conservative beginning/end and the
            #   first/last recorded zero, DOVSs may be used in audit
            if dovs_outg_t_beg_end[0]>cnsrvtv_beg_end[0] and dovs_outg_t_beg_end[0]<zero_t_beg_end[0]:
                dovs_best_est[0] = dovs_outg_t_beg_end[0]
            #-----
            if dovs_outg_t_beg_end[1]<cnsrvtv_beg_end[1] and dovs_outg_t_beg_end[1]>zero_t_beg_end[1]:
                dovs_best_est[1] = dovs_outg_t_beg_end[1]
        #----------------------------------------------------------------------------------------------------
        ede_best_est = [None, None]
        if outg_times_ede is not None:
            # Expecting outg_times_ede to be a list of outage times or a dict with pd_times, pu_times keys
            assert(Utilities.is_object_one_of_types(outg_times_ede, [list, dict]))
            #----------
            if isinstance(outg_times_ede, list) and len(outg_times_ede)>0:
                # Expecting outg_times_ede to be a list of outage times, so list of two-element lists
                outg_times_ede_shape = np.array(outg_times_ede).shape
                assert(outg_times_ede_shape[1]==2)
                #-----
                # Unzip outg_times_ede, taking it from a list of outage times (2-element lists) to a list of two lists,
                #   one containing the beginning times and one containing the end times
                # This is essentially a transpose
                outg_times_beg_end_ede = list(zip(*outg_times_ede))
                #-----
                # Expecting outg_times_beg_end_ede to be a list containing two lists
                outg_times_beg_end_ede_shape = np.array(outg_times_beg_end_ede).shape
                assert(outg_times_beg_end_ede_shape[0]==2)

                # Since this is a transpose...
                assert(outg_times_ede_shape[0]==outg_times_beg_end_ede_shape[1])
                #-----
                outg_times_beg_ede = outg_times_beg_end_ede[0]
                outg_times_end_ede = outg_times_beg_end_ede[1]
            else:
                outg_times_beg_ede=[]
                outg_times_end_ede=[]
            #----------
            if isinstance(outg_times_ede, dict):
                assert('pd_times' in outg_times_ede.keys() and 'pu_times' in outg_times_ede.keys())
                outg_times_beg_ede = outg_times_ede['pd_times']
                outg_times_end_ede = outg_times_ede['pu_times']

            #--------------------------------------------------
            # If multiple suitable beg/ends are found, those which maximize the outage time are chosen
            #   i.e., min of ede_best_ests_beg and max of ede_best_ests_end
            # This keeps in line with our strategy of being conservative at each step.
            #-----
            # To be certain, if outg_times_ede is a list, meaning it is a collection of (outg_beg,outg_end pairs) 
            #   for different sub-outages, the method below allows one sub-outage to be used to resolve the ambiguity
            #   in the start time and another to resolve that of the end time.
            #-------------------------
            ede_best_ests_beg = []
            for beg_t_i in outg_times_beg_ede:
                if beg_t_i>cnsrvtv_beg_end[0] and beg_t_i<zero_t_beg_end[0]:
                    ede_best_ests_beg.append(beg_t_i)
            #-----
            if ede_best_ests_beg:
                ede_best_est_beg = np.min(ede_best_ests_beg)
            else:
                ede_best_est_beg = None #excplicitly make None for later
            #-------------------------
            ede_best_ests_end = []
            for end_t_i in outg_times_end_ede:
                if end_t_i<cnsrvtv_beg_end[1] and end_t_i>zero_t_beg_end[1]:
                    ede_best_ests_end.append(end_t_i)
            #-----
            if ede_best_ests_end:
                ede_best_est_end = np.max(ede_best_ests_end)
            else:
                ede_best_est_end = None #excplicitly make None for later
            #--------------------------------------------------
            ede_best_est = [ede_best_est_beg, ede_best_est_end]
        #----------------------------------------------------------------------------------------------------
        # NOTE: dovs always 0th element, ede 1st element
        best_ests_beg = [dovs_best_est[0], ede_best_est[0]]
        best_ests_end = [dovs_best_est[1], ede_best_est[1]]
        #--------------------------------------------------
        if np.all([x is None for x in best_ests_beg]):
            best_est_beg = None
        else:
            # NOTE: because of if statement above, we can be certain here there is at least one
            #       non-null value, so taking min/max should be safe (since not finding min/max of empty list)
            if selection_method=='min':
                best_est_beg = np.max([x for x in best_ests_beg if x])
            elif selection_method=='max':
                best_est_beg = np.min([x for x in best_ests_beg if x])
            elif selection_method=='dovs pref' or selection_method=='dovs only':
                if best_ests_beg[0]:
                    best_est_beg = best_ests_beg[0]
                else:
                    if selection_method=='dovs pref':
                        assert(best_ests_beg[1])
                        best_est_beg = best_ests_beg[1]
                    else:
                        #selection_method=='dovs only' case
                        best_est_beg = None
            elif selection_method=='ede pref' or selection_method=='ede only':
                if best_ests_beg[1]:
                    best_est_beg = best_ests_beg[1]
                else:
                    if selection_method=='ede pref':
                        assert(best_ests_beg[0])
                        best_est_beg = best_ests_beg[0]
                    else:
                        #selection_method=='ede only' case
                        best_est_beg = None
            else:
                assert(0)
        #-------------------------
        if np.all([x is None for x in best_ests_end]):
            best_est_end = None
        else:
            # NOTE: because of if statement above, we can be certain here there is at least one
            #       non-null value, so taking min/max should be safe (since not finding min/max of empty list)
            if selection_method=='min':
                best_est_end = np.min([x for x in best_ests_end if x])
            elif selection_method=='max':
                best_est_end = np.max([x for x in best_ests_end if x])
            elif selection_method=='dovs pref' or selection_method=='dovs only':
                if best_ests_end[0]:
                    best_est_end = best_ests_end[0]
                else:
                    if selection_method=='dovs pref':
                        assert(best_ests_end[1])
                        best_est_end = best_ests_end[1]
                    else:
                        #selection_method=='dovs only' case
                        best_est_end = None
            elif selection_method=='ede pref' or selection_method=='ede only':
                if best_ests_end[1]:
                    best_est_end = best_ests_end[1]
                else:
                    if selection_method=='ede pref':
                        assert(best_ests_end[0])
                        best_est_end = best_ests_end[0]
                    else:
                        #selection_method=='ede only' case
                        best_est_end = None
            else:
                assert(0)
        #----------------------------------------------------------------------------------------------------
        # By definition, if any best estimates are found, they are in the uncertainty intervals, and therefore
        #  reduce the outage time in comparison to the conservative estimates
        if best_est_beg:
            assert(best_est_beg > cnsrvtv_beg_end[0])
            final_out_t_beg = best_est_beg
        else:
            final_out_t_beg = cnsrvtv_beg_end[0]
        #-----
        if best_est_end:
            assert(best_est_end < cnsrvtv_beg_end[1])
            final_out_t_end = best_est_end
        else:
            final_out_t_end = cnsrvtv_beg_end[1]
        #-------------------------
        if return_all_best_ests:
            all_best_est_dict = dict(
                ede          = ede_best_est, 
                dovs         = dovs_best_est,
                conservative = cnsrvtv_beg_end, 
                zero_times   = zero_t_beg_end,
                winner       = [final_out_t_beg, final_out_t_end], 
                open_beg     = outg_est_times_i['open_beg'], 
                open_end     = outg_est_times_i['open_end']
            )
            return final_out_t_beg, final_out_t_end, all_best_est_dict
        else:
            return final_out_t_beg, final_out_t_end
        
        
    @staticmethod
    def calculate_mi_for_meter_ami_w_ede_help(
        df_i, 
        ede_df_i=None, 
        t_search_min_max=None, 
        conservative_estimate=True, 
        dovs_outg_t_beg_end=None, 
        est_ede_kwargs=None, 
        audit_selection_method='min', 
        t_int_beg_col='starttimeperiod_local', 
        t_int_end_col='endtimeperiod_local', 
        value_col='value', 
        SN_col='serialnumber', 
        return_all_best_ests=False
    ):
        r"""
        Find the times for which df_i registers a value of 0 and use that information to calculate the Minutes Interrupted (MI)
          for the meter.
        df_i should contain AMI data for a single meter.

        t_search_min_max:
            Allows the user to set of time subset of the data from which to search for the outages/calculate mi.
            If t_search_min_max is not None, it should be a two-element list/tuple.
            However, one of the elements of the list may be None if, e.g., one wants limits on only the min or max

        conservative_estimate:
            It appears that AMI meters only report values of 0 if all readings in the 15-minute period are zero.
            I believe that if at least one value is non-zero, the AMI will report a non-zero value.
                e.g., if power out from 12:00 to 12:14, but turns on from 12:14 to 12:15, the 12:15 reading will be non-zero
            Thus, simply calculating the period of time when the measurements are 0 inherently underestimates the actual outage time.
            The more conservative approach, which typically overestimates the MI, would be to add 2*freq to MI, where freq is the 
              frequency of the data (typically 15 minutes).
                To elaborate a bit more: the outage actually begins in the interval before the first 0 value is regiesterd and ends
                  in the interval after the last 0 is registered.
            If conservative_estimate==True, 2*freq is added to the MI UNLESS dovs_outg_t_beg_end is supplied (see description below)

        dovs_outg_t_beg_end:
            Only has an affect if conservative_estimate==True
            If the outage beginning and ending times are supplied from DOVS, they can sometimes be used in place of the conservative
              starting and ending times to help resolve the uncertainty.
            Essentially, if a DOVS time falls within the uncertainty interval, it can be used in place of the conservative time (where
              the uncertainty interval is the time before/after the first/last recorded zero value and the preceding/following interval).
                i.e., on the front end, use DOVS if time between 0-freq before last zero value
                      on the back end, use DOVS if time between 0-freq after last zero value
            If a DOVS time is well outside of the uncertainty interval (e.g., if the DOVS start time is well before or well after the first
              recorded 0 value), it cannot be used.

        est_ede_kwargs:
            Keyword arguments to input in DOVSAudit.estimate_outage_times_using_ede_for_meter.
            NOTE: ede_df_i, broad_out_t_beg, broad_out_t_end, and expand_search_time SHOULD NOT be included
                    in est_ede_kwargs, as the function will set these        
        """
        #-------------------------
        # As described above, designed to work for DF containing data from a single meter.
        assert(df_i[SN_col].nunique()==1)
        if ede_df_i is not None:
            assert(ede_df_i[SN_col].nunique()==1)
            assert(df_i[SN_col].unique().tolist()[0]==ede_df_i[SN_col].unique().tolist()[0])

        #-------------------------
        if t_search_min_max is not None:
            assert(Utilities.is_object_one_of_types(t_search_min_max, [list, tuple]))
            assert(len(t_search_min_max)==2)
            #-----
            # At least one of t_search_min_max should not be None
            assert(t_search_min_max[0] is not None or t_search_min_max[1] is not None)
            if t_search_min_max[0] is None:
                t_search_min_max[0] = pd.Timestamp.min
            if t_search_min_max[1] is None:
                t_search_min_max[1] = pd.Timestamp.max
        #-------------------------
        # Make sure values sorted
        df_i = df_i.sort_values(by=t_int_beg_col)
        #--------------------------------------------------
        # Find the time periods where all values are zero
        outg_est_times = DOVSAudit.estimate_outage_times_for_meter(
            df_i=df_i, 
            t_search_min_max=t_search_min_max, 
            t_int_beg_col=t_int_beg_col, 
            t_int_end_col=t_int_end_col, 
            value_col=value_col, 
            SN_col=SN_col
        )
        #--------------------------------------------------
        # Grab the outage times from ede
        if ede_df_i is not None:
            if not est_ede_kwargs:
                est_ede_kwargs = {}
            #broad_out_t_beg = t_search_min_max[0]
            #broad_out_t_end = t_search_min_max[1]
            # Since DOVSAudit.estimate_outage_times_for_meter is able to expand the search window, instead of
            #   using broad_out_t_beg_end = t_search_min_max, I should set these values equal to the
            #   min/max of the conservative estimates from outg_est_times
            if len(outg_est_times)>0:
                broad_out_t_beg = np.min([x['cnsrvtv_t_beg'] for x in outg_est_times])
                broad_out_t_end = np.max([x['cnsrvtv_t_end'] for x in outg_est_times])
            else:
                broad_out_t_beg = t_search_min_max[0]
                broad_out_t_end = t_search_min_max[1]
            dflt_est_ede_kwargs = dict(
                ede_df_i=ede_df_i, 
                broad_out_t_beg=broad_out_t_beg, 
                broad_out_t_end=broad_out_t_end,
                expand_search_time=pd.Timedelta(0), 
                use_full_ede_outgs=False, 
                pd_ids=['3.26.0.47', '3.26.136.47'], 
                pu_ids=['3.26.0.216', '3.26.136.216'], 
                SN_col='serialnumber', 
                valuesinterval_col='valuesinterval_local', 
                edetypeid_col='enddeviceeventtypeid', 
                verbose=False
            )
            #-----
            # Make sure none of the kwargs set by function are included in est_ede_kwargs
            supplied_est_ede_kwargs = ['ede_df_i', 'broad_out_t_beg', 'broad_out_t_end', 'expand_search_time']
            assert(len(set(est_ede_kwargs.keys()).intersection(set(supplied_est_ede_kwargs)))==0)
            #-----
            # Want est_ede_kwargs to be kept over dflt_est_ede_kwargs, so use to_supplmnt_dict=est_ede_kwargs in the following
            est_ede_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict=est_ede_kwargs, 
                default_values_dict=dflt_est_ede_kwargs, 
                extend_any_lists=False, 
                inplace=False
            )
            use_full_ede_outgs = est_ede_kwargs.pop('use_full_ede_outgs')
            #-----
            if use_full_ede_outgs:
                outg_times_ede = DOVSAudit.estimate_outage_times_using_ede_for_meter(**est_ede_kwargs)
            else:
                outg_times_ede = DOVSAudit.get_pd_pu_times_for_meter(
                    ede_df_i=ede_df_i, 
                    t_search_min_max=[est_ede_kwargs['broad_out_t_beg'], est_ede_kwargs['broad_out_t_end']], 
                    pd_ids=est_ede_kwargs['pd_ids'], 
                    pu_ids=est_ede_kwargs['pu_ids'], 
                    SN_col=est_ede_kwargs['SN_col'], 
                    valuesinterval_col=est_ede_kwargs['valuesinterval_col'], 
                    edetypeid_col=est_ede_kwargs['edetypeid_col']
                )
        else:
            outg_times_ede = None
        #--------------------------------------------------
        # Iterate through outg_est_times, calculate mi_j for each, and add to total mi
        mi=pd.Timedelta(0)
        all_best_ests = []
        for outg_est_times_j in outg_est_times:
            #-------------------------
            #TODO: Is there really any reason to find times from df_i?
            #      Can't I just directly use outg_est_times_j?
            #      Only point seems to be to enforce assertion, but probably 
            #        the definition of DOVSAudit.estimate_outage_times_for_meter requires that outcome?
            # First, grab the subset of the data corresponding to the zero time
            df_i_zr_j = df_i[
                (df_i[t_int_beg_col]>=outg_est_times_j['zero_t_beg']) & 
                (df_i[t_int_end_col]<=outg_est_times_j['zero_t_end'])
            ]
            #-------------------------
            # Since zero_t_end/zero_t_beg keys used above, all values should be zero
            assert((df_i_zr_j[value_col]==0).all())

            #-------------------------
            # Non-conservative MI, simply the difference in time between the last zero value and the first
            # As mentioned elsehwere, this inherently underestimates the MI
            mi_j = df_i_zr_j.iloc[-1][t_int_end_col]-df_i_zr_j.iloc[0][t_int_beg_col]

            if conservative_estimate:
                outg_j_beg_end = DOVSAudit.outg_time_audit(
                    outg_est_times_i=outg_est_times_j, 
                    dovs_outg_t_beg_end=dovs_outg_t_beg_end, 
                    outg_times_ede=outg_times_ede, 
                    selection_method=audit_selection_method, 
                    return_all_best_ests=return_all_best_ests
                )
                outg_j_beg, outg_j_end = outg_j_beg_end[0], outg_j_beg_end[1]
                mi_j = outg_j_end-outg_j_beg
                #-----
                if return_all_best_ests:
                    assert(len(outg_j_beg_end)==3)
                    all_best_ests_j = outg_j_beg_end[2]
                    #-----
                    all_best_ests.append(all_best_ests_j)
            #-------------------------
            mi += mi_j
        #-------------------------
        mi = mi.total_seconds()/60
        if return_all_best_ests:
            return mi, all_best_ests
        else:
            return mi
        
        
    @staticmethod
    def get_nonoverlapping_search_windows(
        est_outg_times, 
        expand_outg_search_time
    ):
        r"""
        Given a list of estimated outage times and expand search times, return a list of non-overlapping t_search_min_max values.
        Returns a list of two-element lists (0th element is t_search_min, 1st is t_search_max).
        The length of the return list will equal that of est_outg_times

        NOTE: If I ever decide to move to asymmetric expand_outg_search_time, I believe this method should still work.
              However, one should carefully verify instead of assuming.
        """
        #-------------------------
        assert(
            Utilities.is_object_one_of_types(est_outg_times, [list, tuple]) and
            Utilities.are_all_list_elements_one_of_types(est_outg_times, [list, tuple]) and
            Utilities.are_list_elements_lengths_homogeneous(est_outg_times, length=2)
        )

        #--------------------------------------------------
        # First, go through and naively set t_search values
        # To more easily find overlapping regions, define a left, center, and right search window for each outage, where:
        #   left (left)   = search period preceding outage, ending at est. outage begin
        #   center (cntr) = est. outage begin to est. outage end
        #   right (rght)  = search period following outage, beginning at est. outage end
        # The left and right windows will be of length expand_outg_search_time UNLESS they overlap with another outage, in which
        #   case the endpoint of the window will be defined by the overlapping outage.
        #     left - check if overlaps with est. outage end of preceding outage.
        #            If overlaps, minimum of left set to est. outage end of preceding outage.
        #     rght - check if overlaps with est. outage beginning of following outage.
        #            If overlaps, maximum of right set to est. outage beginning of following outage.
        # NOTE: Ensuring the search windows themselves do not overlap occurs in the next for loop, not this one!
        #       This is just a rough chopping
        # NOTE: After completing the code, no real need to keep 'cntr' windows, but no harm in keeping (might be useful later?)
        #-----
        search_windows_coll = []
        est_outg_times = natsorted(est_outg_times)
        for i_outg in range(len(est_outg_times)):
            est_outg_times_i = est_outg_times[i_outg]
            #-----
            # Simple sanity check: ensure outage end occurs after outage beginning!
            assert(len(est_outg_times_i)==2)
            assert(est_outg_times_i[1] > est_outg_times_i[0])
            # Sanity check: the given estimate outage times should not overlap!
            # Note: Sufficient to check that beginning of current is greater than ending of preceding
            if i_outg>0:
                assert(est_outg_times_i[0] >= est_outg_times[i_outg-1][1])
            #-------------------------
            # No adjustment needed on center window
            search_windows_i = dict(cntr=est_outg_times_i)
            #-------------------------
            # Left window
            if i_outg==0:
                # First element, so no preceding outage to compare
                search_windows_i['left']=[
                    est_outg_times_i[0]-expand_outg_search_time, 
                    est_outg_times_i[0]
                ]
            else:
                # Desired left window = est_outg_times_i[0]-expand_outg_search_time to est_outg_times_i[0]
                # Check if minimum value of desired left window overlaps with est. outage end of preceding outage
                # NOTE: im1 == i minus 1
                left_i = [est_outg_times_i[0]-expand_outg_search_time, est_outg_times_i[0]]
                est_outg_times_im1 = est_outg_times[i_outg-1]
                if left_i[0]<est_outg_times_im1[1]:
                    left_i[0] = est_outg_times_im1[1]
                #-----
                search_windows_i['left']=left_i
            #-------------------------
            # Right window
            if i_outg==len(est_outg_times)-1:
                # Last element, so no following outage to compare
                search_windows_i['rght']=[
                    est_outg_times_i[1], 
                    est_outg_times_i[1]+expand_outg_search_time
                ]
            else:
                # Desired right window = est_outg_times_i[1] to est_outg_times_i[1]+expand_outg_search_time
                # Check if maximum value of desired right window overlaps with est. outage beginning of following outage
                # NOTE: ip1 == i plus 1
                rght_i = [est_outg_times_i[1], est_outg_times_i[1]+expand_outg_search_time]
                est_outg_times_ip1 = est_outg_times[i_outg+1]
                if rght_i[1]>est_outg_times_ip1[0]:
                    rght_i[1] = est_outg_times_ip1[0]
                #-----
                search_windows_i['rght']=rght_i
            #-------------------------
            search_windows_coll.append(search_windows_i)
        #--------------------------------------------------
        # Now, go back through and resolve any overlaps.
        # If overlaps exist, set the boundary equal to the midpoint of the overlap.
        # It should be sufficient to go through and compare the left search window of the
        #   current element to the right of the previous
        assert(len(search_windows_coll)==len(est_outg_times))
        for i_outg in range(len(search_windows_coll)):
            if i_outg==0:
                continue
            search_windows_i   = search_windows_coll[i_outg]
            search_windows_im1 = search_windows_coll[i_outg-1]
            #-----
            left_i   = search_windows_i['left']
            rght_im1 = search_windows_im1['rght']
            #-----
            # Sanity check: Due to rough clipping done in first iteration:
            #   max of left_i (outg. beg. i) should be greater than (or eq. to) max of rght_im1
            #   min of rght_im1 (outg. end i-1) should be less than (or eq. to) min of left_i
            assert(left_i[1] >= rght_im1[1])
            assert(rght_im1[0] <= left_i[0])
            #-------------------------
            # If there is no overlap, continue
            if left_i[0] >= rght_im1[1]:
                continue
            #-------------------------
            # Overlap exists, so resolve by setting endpoints equal to midpoint of overlap
            #-----
            # The overlap min is the maximum of the mins
            # The overlap max is the minimum of the maxs
            ovrlp_min = np.max([left_i[0], rght_im1[0]])
            ovrlp_max = np.min([left_i[1], rght_im1[1]])
            #-----
            ovrlp_mid = Utilities_dt.calc_dt_mean([ovrlp_min, ovrlp_max])
            #-----
            # Update values in search_windows_coll (update min of left_i and max of right_im1 to equal ovrlp_mid)
            search_windows_coll[i_outg]['left'][0]   = ovrlp_mid
            search_windows_coll[i_outg-1]['rght'][1] = ovrlp_mid

        #--------------------------------------------------
        # Finally, get the final search windows as the min of left and max of right
        t_search_min_max_coll = []
        for search_windows_i in search_windows_coll:
            left_i = search_windows_i['left']
            rght_i = search_windows_i['rght']
            t_search_min_max_coll.append([left_i[0], rght_i[1]])

        #--------------------------------------------------
        # Final sanity check!
        assert(len(est_outg_times)==len(t_search_min_max_coll))
        for i_outg in range(len(t_search_min_max_coll)):
            assert(t_search_min_max_coll[i_outg][1]>t_search_min_max_coll[i_outg][0])
            if i_outg>0:
                assert(t_search_min_max_coll[i_outg][0]>=t_search_min_max_coll[i_outg-1][1])
        #--------------------------------------------------
        return t_search_min_max_coll
    
    
    @staticmethod
    def convert_all_best_ests_dict_to_df(all_best_ests_dict):
        r"""
        Convert all_best_ests_dict to pd.DataFrame object.
        all_best_ests_dict is returned by, e.g., DOVSAudit.calculate_ci_cmi_w_ami_w_ede_help.
        This is a pretty specific function, so it should probably only be used within DOVSAudit.calculate_ci_cmi_w_ami_w_ede_help
          or directly on an object returned by it.
        NOTE: SNs for which no suboutages were found are EXCLUDED from df!
        -----
        all_best_ests_dict:
            It should be a dictionary whose keys equal serial numbers in an outage.
            Each key is a list of dict objects.
                Each dict object represents on suboutage time, and contains keys for the various estimates.
                The length of the list equals the number of suboutages found (can be zero if none found)
                Typical keys are: 'zero_times', 'conservative', 'winner', 'ede', and 'dovs'
                For each key, there is a two-element list representing the beg/end times.
        EX:
            {'645792023': [],
             '645779036': 
               [
                 {
                  'ede': [Timestamp('2023-01-01 01:22:37'), None],
                  'dovs': [Timestamp('2023-01-01 01:22:00'), None],
                  'conservative': [Timestamp('2023-01-01 01:15:00'), Timestamp('2023-01-01 02:30:00')],
                  'winner': [Timestamp('2023-01-01 01:22:37'), Timestamp('2023-01-01 02:30:00')],
                  'zero_times': [Timestamp('2023-01-01 01:30:00'), Timestamp('2023-01-01 02:15:00')]
                  },
                 {
                  'ede': [None, None],
                  'dovs': [None, Timestamp('2023-01-01 04:13:00')],
                  'conservative': [Timestamp('2023-01-01 03:00:00'), Timestamp('2023-01-01 04:15:00')],
                  'winner': [Timestamp('2023-01-01 03:00:00'), Timestamp('2023-01-01 04:13:00')],
                  'zero_times': [Timestamp('2023-01-01 03:15:00'), Timestamp('2023-01-01 04:00:00')]
                 }
              ],
            ...
        """
        #-------------------------
        return_df = pd.DataFrame()
        #-----
        for (SN_i, PN_i), best_ests_list_i in all_best_ests_dict.items():
            if len(best_ests_list_i)==0:
                continue
            for outg_j, best_est_dict_ij in enumerate(best_ests_list_i):
                best_est_dict_ij_split=dict()
                best_est_dict_ij_split['SN'] = SN_i
                best_est_dict_ij_split['PN'] = PN_i
                best_est_dict_ij_split['i_outg'] = outg_j
                for est_key, beg_end_vals in best_est_dict_ij.items():
                    if est_key=='open_beg' or est_key=='open_end':
                        continue
                    best_est_dict_ij_split[f"{est_key}_min"] = beg_end_vals[0]
                    best_est_dict_ij_split[f"{est_key}_max"] = beg_end_vals[1]
                #-----
                best_est_dict_ij_split['open_beg'] = best_est_dict_ij['open_beg']
                best_est_dict_ij_split['open_end'] = best_est_dict_ij['open_end']
                #-----
                return_df = pd.concat([return_df, pd.DataFrame(best_est_dict_ij_split, index=[return_df.shape[0]])])
        #-------------------------
        return return_df
    
    
    @staticmethod
    def calculate_ci_cmi_w_ami_w_ede_help(
        df, 
        ede_df, 
        dovs_outg_t_beg_end, 
        expand_outg_search_time=pd.Timedelta('1 hour'), 
        conservative_estimate=True, 
        est_ede_kwargs=None, 
        audit_selection_method='min', 
        return_CI_SNs=False, 
        use_est_outg_times=False, 
        pct_SNs_required_for_outage_est=0, 
        expand_outg_est_search_time=pd.Timedelta('1 hour'), 
        use_only_overall_endpoints_of_est_outg_times=False, 
        t_int_beg_col='starttimeperiod_local', 
        t_int_end_col='endtimeperiod_local', 
        value_col='value', 
        SN_col='serialnumber', 
        PN_col='aep_premise_nb', 
        return_all_best_ests=False, 
        return_all_best_ests_type='dict'
    ):
        r"""
        Given a DF containing AMI data together with outage begin and end times from DOVS, calculate the CI (customers interrupted) 
          and CMI (customer minutes interrupted).

        NOTE: The times from DOVS are used simply as time markers around which the algorithm searches for outages events.
              The DOVS times aren't really necessary aside from setting this time search region (defined by both dovs_outg_t_beg_end
                and expand_outg_search_time)
              So, by no means is this a simple calculation using only DOVS.
              Moreover, any beg/end time together with expand_outg_search_time could be used.

        NOTE: if ede_df is supplied, it should be for the same outage as that of df.

        Return value is tuple containing (CI, CMI)
          UNLESS return_CI_SNs==True, in which case return (CI, CMI, outg_SNs)

        NOTE:  If one wants to calculate CI/CMI using PNs (premise numbers) instead of SNs (serial numbers), one must be careful
               to first drop duplicate values (after dropping SNs col), or else the same answer will be found for PNs as SNs

        dovs_outg_t_beg_end:
            Should be a list/tuple of length two containing the outage beginning and ending times from DOVS.
            This, together with expand_outg_search_time sets the region around which the algorithm searches for outages.

        expand_outg_search_time:
            Sets the region around each outage (either found using DOVSAudit.estimate_outage_times if use_est_outg_times==True, or taken
              directly from dovs_outg_t_beg_end) to search for individual meter outages (using calculate_mi_for_meter).
            If use_est_outg_times==True, probably want expand_outg_search_time tighter and expand_outg_est_search_time looser.
            If use_est_outg_times==False, probably want expand_outg_search_time looser

        conservative_estimate:
            Should likely have this set to True, otherwise one will certainly be underestimating the CI/CMI
            If False, CI/CMI are calculated only using the periods with values of 0.
              As documented elsewhere, it appears that AMI meters only report values of 0 if all readings in the 15-minute period are zero.
              I believe that if at least one value is non-zero, the AMI will report a non-zero value.
                  e.g., if power out from 12:00 to 12:14, but turns on from 12:14 to 12:15, the 12:15 reading will be non-zero
              Thus, simply calculating the period of time when the measurements are 0 inherently underestimates the actual outage time.
            If True, a more conservative approach is used, in which the CI/CMI include the period preceding the first 0 value and that
              following the last 0 value.
              If applicable, the endpoint will be taken from DOVS (via dovs_outg_t_beg_end) instead of the full period preceding/following
                as described above.
              Find more information regarding this functionality in the dovs_outg_t_beg_end section of the calculate_mi_for_meter documentation.

        return_CI_SNs:
            If True, also return outg_SNs.
            This feature is mainly included for functionality where out_t_beg_end contains multiple outage times.

        use_est_outg_times:
            If True, estimate the outage times by looking at the collection of all SNs in df
              i.e., try to determine when outages occur by looking for periods of time where the meters show values of 0.
            If a single outage is actually comprised of smaller suboutages, the suboutage times will be returned and used.
            NOTE: If no estimate outages are found, dovs_outg_t_beg_end will be used (assuming it is not None)
            -----
            pct_SNs_required_for_outage_est:
                The percentage of serial numbers which must show a reading of 0 for a time period to be considered an outage.
                  e.g., if pct_SNs_required_for_outage==0, all serial numbers must have a reading of 0
                  e.g., if pct_SNs_required_for_outage==25, 25% of the serial numbers must have a reading of zero.
                NOTE: pct_SNs_required_for_outage_est=0 essentially means at least one SN must register a value of 0

            expand_outg_est_search_time=pd.Timedelta('1 hour'):
                The interval around dovs_outg_t_beg_end to use when trying to estimate the outage times.
                This should typically be looser than expand_outg_search_time.

            use_only_overall_endpoints_of_est_outg_times:
                As stated above, if a single outage is actually comprised of smaller suboutages, the suboutage times 
                  will be returned.
                If use_only_overall_endpoints_of_est_outg_times==True, the beginning of the first suboutage and ending
                  of the last suboutage will only be used.
                If use_only_overall_endpoints_of_est_outg_times==False, all suboutages will be used fully.
        """
        #-------------------------
        if use_est_outg_times:
            # Find an estimate of the outage time using the AMI data
            # NOTE: Below, conservative_estimate should always be True (not necessarily equal to input parameter)
            est_outg_times = DOVSAudit.estimate_outage_times(
                df = df, 
                t_search_min_max=[
                    dovs_outg_t_beg_end[0]-expand_outg_est_search_time, 
                    dovs_outg_t_beg_end[1]+expand_outg_est_search_time
                ], 
                pct_SNs_required_for_outage=pct_SNs_required_for_outage_est, 
                relax_pct_SNs_required_if_no_outgs_found=True, 
                t_int_beg_col=t_int_beg_col, 
                t_int_end_col=t_int_end_col, 
                value_col=value_col, 
                verbose=False,
                outg_rec_nb_col=None
            )
            if len(est_outg_times)==0:
                if dovs_outg_t_beg_end is not None:
                    outg_times = [dovs_outg_t_beg_end]
                else:
                    outg_times = []
            else:
                if use_only_overall_endpoints_of_est_outg_times:
                    outg_times = [[est_outg_times[0]['cnsrvtv_t_beg'], est_outg_times[-1]['cnsrvtv_t_end']]]
                else:
                    outg_times = [[x['cnsrvtv_t_beg'], x['cnsrvtv_t_end']] for x in est_outg_times]
        else:
            assert(dovs_outg_t_beg_end is not None)
            outg_times = [dovs_outg_t_beg_end]
        #-------------------------
        # Get the search min/max times from the est. outage times and expand search time
        t_search_min_max_coll = DOVSAudit.get_nonoverlapping_search_windows(
            est_outg_times=outg_times, 
            expand_outg_search_time=expand_outg_search_time
        )
        assert(len(t_search_min_max_coll)==len(outg_times))
        #-----
        CMI=0
        outg_SNs=[]
        all_best_ests_dict = dict()
        for t_search_min_max_i in t_search_min_max_coll:
            df_outg_i=df 
            # Get the SNs and PNs for outg_i
            SNs_PNs_i = df_outg_i[[SN_col, PN_col]].value_counts().index.tolist()
            #-------------------------
            for SN_ij, PN_ij in SNs_PNs_i:
                df_outg_i_SN_j = df_outg_i[df_outg_i[SN_col]==SN_ij]
                assert(df_outg_i_SN_j[PN_col].nunique()==1)
                #-----
                if ede_df is not None and ede_df.shape[0]>0:
                    if SN_ij in ede_df[SN_col].tolist():
                        ede_df_outg_i_SN_j = ede_df[ede_df[SN_col]==SN_ij]
                    else:
                        ede_df_outg_i_SN_j=None
                else:
                    ede_df_outg_i_SN_j=None
                #-----
                mi_ij = DOVSAudit.calculate_mi_for_meter_ami_w_ede_help(
                    df_i=df_outg_i_SN_j, 
                    ede_df_i=ede_df_outg_i_SN_j, 
                    t_search_min_max=t_search_min_max_i, 
                    conservative_estimate=conservative_estimate, 
                    dovs_outg_t_beg_end=dovs_outg_t_beg_end, 
                    est_ede_kwargs=est_ede_kwargs, 
                    audit_selection_method=audit_selection_method, 
                    t_int_beg_col=t_int_beg_col, 
                    t_int_end_col=t_int_end_col, 
                    value_col=value_col, 
                    SN_col=SN_col, 
                    return_all_best_ests=return_all_best_ests
                )
                #-------------------------
                if return_all_best_ests:
                    assert(isinstance(mi_ij, tuple) and len(mi_ij)==2)
                    # NOTE: mi_ij must be grabbed from tuple second!
                    all_best_ests_ij = mi_ij[1]
                    mi_ij = mi_ij[0]
                    #-----
                    if (SN_ij, PN_ij) in all_best_ests_dict.keys():
                        all_best_ests_dict[(SN_ij, PN_ij)].extend(all_best_ests_ij)
                    else:
                        all_best_ests_dict[(SN_ij, PN_ij)] = all_best_ests_ij
                #-------------------------                
                CMI += mi_ij
                if mi_ij>0:
                    outg_SNs.append(SN_ij) 
        #-------------------------
        # If there are multiple sub-outage periods, likely repeat values in outg_SNs
        # So need to perform set operation
        outg_SNs = list(set(outg_SNs))
        CI = len(outg_SNs)
        #-------------------------
        if not return_CI_SNs and not return_all_best_ests:
            return CI, CMI
        else:
            return_dict = dict(
                CI=CI, 
                CMI=CMI
            )
            if return_CI_SNs:
                return_dict['CI_SNs'] = outg_SNs
            if return_all_best_ests:
                assert(return_all_best_ests_type in ['dict', 'pd.DataFrame'])
                if return_all_best_ests_type=='dict':
                    return_dict['all_best_ests'] = all_best_ests_dict
                else:
                    return_dict['all_best_ests'] = DOVSAudit.convert_all_best_ests_dict_to_df(all_best_ests_dict)
            #-----
            return return_dict
        
        

    #****************************************************************************************************
    # best_ests_df/means_df methods
    #****************************************************************************************************
    @staticmethod
    def get_mean_times_w_dbscan(
        best_ests_df, 
        eps_min=5, 
        min_samples=2, 
        ests_to_include_in_clustering=['winner_min', 'winner_max'],
        ests_to_include_in_output=[
            'winner_min', 'winner_max', 
            'conservative_min', 'conservative_max', 
            'zero_times_min', 'zero_times_max'
        ], 
        return_labelled_best_ests_df=False
    ):
        r"""
        """
        #-------------------------
        # For this procedure to work, they cannot be and NaNs in ests_to_include_in_clustering.
        # Therefore, it may be best to use winner, as there will always be winner values
        if best_ests_df.dropna(subset=ests_to_include_in_clustering).shape[0]==0:
            print("All rows contain at least one NaN in ests_to_include_in_clustering!")
            print("Resetting ests_to_include_in_clustering to only includer winners")
            ests_to_include_in_clustering=['winner_min', 'winner_max']
        #-----
        best_ests_df = best_ests_df.dropna(subset=ests_to_include_in_clustering)
        #-------------------------
        # Although it is apparently possible to feed dbscan the raw Datetime objects, it appears that they are essentially
        #   converted to timestamps, meaning they are converted to numbers on the order of 1.6e9, which makes them somewhat
        #   difficult to deal with (e.g., in setting the eps parameter for dbscan)
        # Therefore, for my purposes it is preferable to convert these to smaller numbers, such as the difference between the row's
        #   datetime and the approximate beginning of the outage (NOTE, the comparison value doesn't matter too much, we just want
        #   something in the ballpark so the differences are not too big).
        # I will express these differences in terms of minutes, so the eps value should be set in terms of minutes
        comp_val = best_ests_df[ests_to_include_in_clustering[0]].min()
        #-----
        db_cols = []
        for clust_col in ests_to_include_in_clustering:
            best_ests_df[f'{clust_col}_db'] = (best_ests_df[clust_col]-comp_val).dt.total_seconds()/60
            db_cols.append(f'{clust_col}_db')
        #-------------------------    
        db = DBSCAN(eps=eps_min, min_samples=min_samples).fit(best_ests_df[db_cols])
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        #-------------------------
        # Set the label values in best_ests_df
        best_ests_df['db_label'] = labels
        #-------------------------
        means_df = best_ests_df.groupby(['db_label'])[ests_to_include_in_output].mean()
        if n_noise_>0:
            # Unclustered data will be handled separately, and not aggregated.  So remove from means_df
            means_df=means_df.drop(index=[-1])
            #-----
            unclstrd_df = best_ests_df[best_ests_df['db_label']==-1][ests_to_include_in_output].copy()
            unclstrd_df['db_label'] = [f"Unclustered {i}" for i in range(unclstrd_df.shape[0])]
            unclstrd_df = unclstrd_df.set_index('db_label')
            #-----
            # Want to set the correct db_label value for these in best_ests_df in case return_labelled_best_ests_df
            best_ests_df.loc[best_ests_df['db_label']==-1, 'db_label'] = [f"Unclustered {i}" for i in range(best_ests_df[best_ests_df['db_label']==-1].shape[0])]
            #-----
            means_df = pd.concat([means_df, unclstrd_df])
        #-------------------------
        if return_labelled_best_ests_df:
            return means_df, best_ests_df.drop(columns=db_cols)
        else:
            return means_df
        
    @staticmethod
    def add_nPNs_to_means_df(
        means_df, 
        best_ests_df_w_db_lbl,
        db_label_col='db_label', 
        PN_col='PN', 
        n_PNs_col = 'n_PNs'
    ):
        r"""
        Add the number of premise numbers per db_label group to means_df
        NOTE: The index of means_df should be the db_labels (db_label_col is to locate the labels
                within best_ests_df_w_db_lbl)
        """
        #-------------------------
        n_PNs_dict = {}
        for db_label_i, row_i in means_df.iterrows():
            n_PNs_i = best_ests_df_w_db_lbl[best_ests_df_w_db_lbl[db_label_col]==db_label_i][PN_col].nunique()
            n_PNs_dict[db_label_i] = n_PNs_i
        #-----
        n_PNs_srs = pd.Series(n_PNs_dict)
        n_PNs_srs.name = n_PNs_col
        #-------------------------
        means_df = pd.merge(means_df, n_PNs_srs, left_index=True, right_index=True, how='inner')
        return means_df
    
    @staticmethod
    def combine_PNs_in_best_ests_df_i(
        best_ests_df_for_PN_i, 
        min_cols, 
        max_cols, 
        return_col_order, 
        likeness_thresh = pd.Timedelta('1 minutes'), 
        SN_col = 'SN', 
        PN_col = 'PN', 
        i_outg_col = 'i_outg', 
        open_beg_col = 'open_beg', 
        open_end_col = 'open_end'
    ):
        r"""
        !!!!! THIS IS A HELPER FUNCTION FOR DOVSAudit.combine_PNs_in_best_ests_df. !!!!!
        !!!!! IT IS NOT RECOMMENDED TO USE THIS FUNCTION OUTSIDE OF DOVSAudit.combine_PNs_in_best_ests_df !!!!!
        """
        #-------------------------
        if best_ests_df_for_PN_i.shape[0]==1:
            return best_ests_df_for_PN_i[return_col_order].squeeze()
        #-------------------------
        # best_ests_df_for_PN_i should come from best_ests_df being grouped by PN_col and i_outg_col
        # Therefore, there should be a single unique value for these!
        #   NOTE: Could also do nunique for each column sparately instead of using value_counts
        assert(best_ests_df_for_PN_i[[PN_col, i_outg_col]].value_counts().shape[0]==1)
        #-------------------------
        ranges = best_ests_df_for_PN_i[min_cols+max_cols].apply(lambda x: x.dropna().max()-x.dropna().min())
        #-----
        # NOTE: NaN values are possible if all values are None for given column
        if (ranges>likeness_thresh).any():
            print('Outage estimates for serial numbers in given premise are not similar!')
            print('CRASH IMMINENT')
            print(f"PN = {best_ests_df_for_PN_i[PN_col].unique()[0]}")
            print(f"threshold = {likeness_thresh}")
            print(f"Violators:\n{ranges[ranges>likeness_thresh]}")
        assert(((ranges.isna()) | (ranges<=likeness_thresh)).all())
        #-------------------------
        return_srs = best_ests_df_for_PN_i[min_cols+max_cols].agg({x:'min' for x in min_cols}|{x:'max' for x in max_cols})
        #-------------------------
        assert(best_ests_df_for_PN_i[open_beg_col].nunique()==1)
        return_srs[open_beg_col] = best_ests_df_for_PN_i[open_beg_col].unique().tolist()[0]
        #-----
        assert(best_ests_df_for_PN_i[open_end_col].nunique()==1)
        return_srs[open_end_col] = best_ests_df_for_PN_i[open_end_col].unique().tolist()[0]
        #-------------------------
        return_srs = return_srs[return_col_order]
        #-------------------------
        return return_srs


    @staticmethod
    def combine_PNs_in_best_ests_df(
        best_ests_df, 
        likeness_thresh = pd.Timedelta('1 minutes'), 
        SN_col = 'SN', 
        PN_col = 'PN', 
        i_outg_col = 'i_outg', 
        open_beg_col = 'open_beg', 
        open_end_col = 'open_end'
    ):
        r"""
        Combine all serial numbers (SNs) for each premise number (PN) in best_ests_df.
        The purpose is so that CI/CMI may be calculated at the premise level, instead of meter level.
        When there are multiple meters for a given premise:
            1. If there are multiple sub-outages, it is checked to ensure each sub-outage contains the same
                 set of serial numbers, see EXPLANATION OF STACKED GROUPBY OPERATIONS below for more info.
            2. For each sub-outage, it is ensured that the time estimates from all meters is roughly equal,
                 as defined by the likeness_thresh parameter.
               e.g., winner_min should be roughly the same for all meters on a premise, as should winner_max, etc.
            3. For each premise and sub-outage:
                   for all minimum time estimates, the minimum value amongst the meters on a premise is kept.
                   for all maximum time estimates, the maximum value amongst the meters on a premise is kept.
               This maintains our strategy of reporting the most conservative value for the outage time, although,
                 due to the tightness of likeness_thresh, this doesn't make a huge difference.

        best_ests_df:
            Should contain SN_col, PN_col, i_outg_col and various min/max time estimate columns.
            In a typical situation, best_ests_df will have the following columns:
                ['SN', 'PN', 'i_outg',
                 'ede_min', 'ede_max',
                 'dovs_min', 'dovs_max',
                 'conservative_min', 'conservative_max',
                 'zero_times_min', 'zero_times_max',
                 'winner_min', 'winner_max']
            NOTE: minimum columns must end with '_min' and maxima with '_max'
                  Also, best_ests_df should only contain 'SN', 'PN', 'i_outg' and min/max columns

        likeness_thresh:
            Sets the maximum allowed difference between time estimates for meters of a given premise.
            All SNs on a given PN should lose and regain power at the same time, although slight differences exist (typically
              only a few seconds difference, and likely due to delays in meters registering events)

        --------------------------------------------------
        EXPLANATION OF STACKED GROUPBY OPERATIONS
        --------------------------------------------------
        The purpose of the multiple groupby operations is to ensure that, when multiple sub-outages exist for a premise (i.e., 
          when i_outg=0, 1, ...), all sub-outages share the same set of meters.
        One would not expect, e.g., the first sub-outage to only affect one meter on a given premise and the second sub-outage
          to affect multiple meters on the premise.
        The full chain of groupby commands I am referencing is:

            (best_ests_df
             .groupby([PN_col, i_outg_col])[SN_col].apply(list)
             .groupby([PN_col]).apply(list)
             .apply(lambda x: Utilities.are_all_lists_eq(x))
            )  

        Suppose best_ests_df has the following form:

                  SN    PN  i_outg
            0  SN_11  PN_1       0
            1  SN_11  PN_1       1
            2  SN_12  PN_1       0
            3  SN_12  PN_1       1
            4  SN_21  PN_2       0
            5  SN_21  PN_2       1
            6  SN_22  PN_2       0
            7  SN_22  PN_2       1

        In this example, there are two premises (PN_1 and PN_2), each having two meters (SN_11, SN_12 associated with PN_1 
          and SN_21, SN_22 associated with PN_2), and each having an outage split into two sub-outages (i_outg 0 and 1).
        The output for the steps of the chain of groupby commands is as follows:
        -------------------------
        1. Get a list of the meters (SNs) associated with each premise and sub-outage
            ----------
            (best_ests_df
             .groupby([PN_col, i_outg_col])[SN_col].apply(list)
            )
            ----------
            PN    i_outg
            PN_1  0         [SN_11, SN_12]
                  1         [SN_11, SN_12]
            PN_2  0         [SN_21, SN_22]
                  1         [SN_21, SN_22]        
        -------------------------
        2. Combine these into a list of lists for each premise
            ----------
            (best_ests_df
             .groupby([PN_col, i_outg_col])[SN_col].apply(list)
             .groupby([PN_col]).apply(list)
            )
            ----------
            PN
            PN_1    [[SN_11, SN_12], [SN_11, SN_12]]
            PN_2    [[SN_21, SN_22], [SN_21, SN_22]]        
        -------------------------
        3. Use Utilities.are_all_lists_eq function to ensure the collection of meters for each premise is consistent
             for all sub-outages
            ----------
            (best_ests_df
             .groupby([PN_col, i_outg_col])[SN_col].apply(list)
             .groupby([PN_col]).apply(list)
             .apply(lambda x: Utilities.are_all_lists_eq(x))
            )
            ----------
            PN
            PN_1    True
            PN_2    True
        """
        #-------------------------
        # Make sure SN_col, PN_col, and i_outg_col all in best_ests_df
        req_cols = [SN_col, PN_col, i_outg_col, open_beg_col, open_end_col]
        assert(len(set(req_cols).difference(best_ests_df.columns.tolist()))==0)

        # Determine which cols are mins and which are maxs
        # NOTE: The method used for finding minmax_cols together with the assertion below assures that 
        #       best_ests_df contains only SN_col, PN_col, i_outg_col and min/max cols
        minmax_cols = [x for x in best_ests_df.columns.tolist() if x not in req_cols]
        min_cols = [x for x in best_ests_df.columns.tolist() if x.endswith('_min')]
        max_cols = [x for x in best_ests_df.columns.tolist() if x.endswith('_max')]
        assert(len(set(minmax_cols).symmetric_difference(set(min_cols+max_cols)))==0)
        assert(len(min_cols)==len(max_cols))

        # Maintain the original order of columns in best_ests_df in the returned df
        return_col_order = [x for x in best_ests_df.columns.tolist() if x in min_cols+max_cols]
        return_col_order.extend([open_beg_col, open_end_col])
        #-------------------------
        # All columns in minmax_cols must be datetime type
        # Below ensures all min/max columns are datetime
        for col_i in min_cols+max_cols:
            dtype_i = best_ests_df[col_i].dtype
            if not (dtype_i is datetime.datetime or is_datetime64_dtype(dtype_i)):
                best_ests_df = Utilities_df.convert_col_type_w_pd_to_datetime(
                    df=best_ests_df, 
                    column=col_i, 
                    inplace=True
                )

        #-------------------------
        # Really, the reduction operations below only need to be performed on PNs with more than one SN
        # To save resources and time, split apart those needing reduced and those not
        SNs_per_PN = best_ests_df[[SN_col, PN_col]].drop_duplicates()[PN_col].value_counts()
        PNs_w_mult_SNs = SNs_per_PN[SNs_per_PN>1].index.tolist()

        # If there are no PNs with multiple SNs, return best_ests_df (without SN_col, as the purpose of this
        #   is to eliminate SN_col by grouping by premise)
        if len(PNs_w_mult_SNs)==0:
            return best_ests_df.drop(columns=[SN_col]).sort_values(by=[PN_col, i_outg_col], ignore_index=True)

        best_ests_df_w_mult  = best_ests_df[best_ests_df[PN_col].isin(PNs_w_mult_SNs)]
        best_ests_df_wo_mult = best_ests_df[~best_ests_df[PN_col].isin(PNs_w_mult_SNs)]
        assert(best_ests_df_w_mult.shape[0]+best_ests_df_wo_mult.shape[0]==best_ests_df.shape[0])

        #-------------------------
        # Ensure that, when multiple sub-outages exist for a premise (i.e., when i_outg=0, 1, ...), 
        #   all sub-outages share the same set of meters.
        # See function documentation above for explanation of operations
        assert(
            (best_ests_df_w_mult
             .groupby([PN_col, i_outg_col])[SN_col].apply(list)
             .groupby([PN_col]).apply(list)
             .apply(lambda x: Utilities.are_all_lists_eq(x))
            ).all()
        )

        #-------------------------
        # Perform reduction
        #   For all minimum time estimates, the minimum value amongst the meters on a premise is kept.
        #   For all maximum time estimates, the maximum value amongst the meters on a premise is kept.
        return_df = best_ests_df_w_mult.groupby([PN_col, i_outg_col], as_index=False, group_keys=False).apply(
            lambda x: DOVSAudit.combine_PNs_in_best_ests_df_i(
                best_ests_df_for_PN_i = x, 
                min_cols              = min_cols, 
                max_cols              = max_cols, 
                return_col_order      = return_col_order, 
                likeness_thresh       = likeness_thresh, 
                SN_col                = SN_col, 
                PN_col                = PN_col, 
                i_outg_col            = i_outg_col, 
                open_beg_col          = open_beg_col, 
                open_end_col          = open_end_col
            )
        )

        #-------------------------
        # Combine together with collection of PNs having a single SN
        # NOTE: Since we are grouping by PN_col, i_outg_col and aggregating the min/max cols,
        #       the SN_col should not be present in the final output
        best_ests_df_wo_mult = best_ests_df_wo_mult.drop(columns=[SN_col])
        assert(return_df.columns.tolist()==best_ests_df_wo_mult.columns.tolist())
        return_df = pd.concat([return_df, best_ests_df_wo_mult])

        #-------------------------
        # Sometimes it seems the operation can change the type of data in min/max cols
        #   e.g., I have seen epoch times returned, instead of datetime object
        #     (e.g., 1673095676000000000, which is actually 2023-01-07 12:47:56)
        # This was occurred in the older version, prior to splitting best_ests_df_w_mult and 
        #   best_ests_df_wo_mult, but there's no harm in keeping this check in place
        # Below ensures all min/max columns are datetime
        for col_i in min_cols+max_cols:
            dtype_i = return_df[col_i].dtype
            if not (dtype_i is datetime.datetime or is_datetime64_dtype(dtype_i)):
                return_df = Utilities_df.convert_col_type_w_pd_to_datetime(
                    df=return_df, 
                    column=col_i, 
                    inplace=True
                )

        #-------------------------
        return_df = return_df.sort_values(by=[PN_col, i_outg_col], ignore_index=True)

        #-------------------------
        return return_df
    
    
    @staticmethod
    def alter_best_ests_df_using_dovs_outg_t_beg(
        best_ests_df,
        dovs_df, 
        outg_rec_nb, 
        outg_rec_nb_idfr='index', 
        dt_off_ts_full_col='DT_OFF_TS_FULL', 
        winner_min_col='winner_min', 
        winner_max_col='winner_max', 
        i_outg_col='i_outg'
    ):
        r"""
        Alter best_ests_df to use the outage beginning time as given by DOVS and the ending times as estimated using AMI.
        To summarize:
            Take dovs_outg_t_beg to be the outage starting time as given by DOVS.
            Any sub-outages which end before dovs_outg_t_beg will be completely removed.
            Any sub-outages which begin after dovs_outg_t_beg will be left alone.
            Sub-outages with beginning times before dovs_outg_t_beg and ending times after dovs_outg_t_beg will
              have their beginning times altered to equal dovs_outg_t_beg
        """
        #-------------------------
        # First, get the needed info from DOVS
        dovs_df_i = DOVSOutages.retrieve_outage_from_dovs_df(
            dovs_df=dovs_df, 
            outg_rec_nb=outg_rec_nb, 
            outg_rec_nb_idfr=outg_rec_nb_idfr, 
            assert_outg_rec_nb_found=True
        )
        assert(dovs_df_i.shape[0]==1)
        # Get the outage time from DOVS
        dovs_outg_t_beg = dovs_df_i.iloc[0][dt_off_ts_full_col]
        #-------------------------
        # Since we are using the DOVS beginning time, we only consider sub-outages that end after this time
        #   Put differently, get rid of any sub-outages which end before the posted DOVS beginning time
        return_df = best_ests_df[best_ests_df[winner_max_col]>dovs_outg_t_beg].copy()
        #-------------------------
        # Make sure return_df is sorted properly, and reset i_outg
        return_df=return_df.sort_values(by=[winner_min_col, winner_max_col], ignore_index=True)
        return_df[i_outg_col]=return_df.index
        #-------------------------
        # Any sub-outages which begin after dovs_outg_t_beg should be left alone
        #   Only those with starting times before dovs_outg_t_beg should be altered
        return_df.loc[return_df[winner_min_col]<dovs_outg_t_beg, winner_min_col] = dovs_outg_t_beg
        #-------------------------
        return return_df
        
    @staticmethod
    def combine_overlapping_suboutages_for_premise(
        be_df_i, 
        groupby_cols, 
        i_outg_col = 'i_outg', 
        t_min_col = 'winner_min', 
        t_max_col = 'winner_max', 
        open_beg_col = 'open_beg', 
        open_end_col = 'open_end'
    ):
        r"""
        Intended for use with a SINGLE premise, meaning a single SN and/or PN.
        -------------------------
        Overlapping suboutages can occur for a given SN/PN.
        This can happen when, e.g., the meter regains power for a short period of time, where a short period
          is defined by the resolution of the voltage data, i.e., 15T
        See OUTG_REC_NB = '13686882' and premise number = '023368334'
            For this outage, it appears all other premised were able to find a power up event to start the second sub-outage
              EXCEPT for premise 023368334.
            Thus, this premise had to use the conservative starting time for the second sub-outage, which caused it to overlap
              with the first sub-outage
        -------------------------
        groupby_cols:
            If SN col is present, should be ['SN'] or ['SN', 'PN']
            If SN col not present (because already combined via DOVSAudit.combine_PNs_in_best_ests_df), should be ['PN']
        """
        #--------------------------------------------------
        if not isinstance(groupby_cols, list):
            groupby_cols = [groupby_cols]
        assert(set(groupby_cols).difference(set(be_df_i.columns.tolist()))==set())
        #--------------------------------------------------
        # As stated in documentation above, this is intended for a single premise!
        assert(len(be_df_i[groupby_cols].value_counts())==1)
        #--------------------------------------------------
        # Make sure properly sorted
        be_df_i = be_df_i.sort_values(by=[t_min_col, t_max_col])

        #--------------------------------------------------
        # Make a column of pd.Interval objects housing pd.Interval(t_min_col, t_max_col)
        # This will allow me to use the .overlaps functionality
        tmp_intrvl_col = Utilities.generate_random_string()
        assert(tmp_intrvl_col not in be_df_i.columns.tolist())
        be_df_i[tmp_intrvl_col] = be_df_i.apply(lambda x: pd.Interval(x[t_min_col], x[t_max_col]), axis=1)

        #--------------------------------------------------
        # Using the tmp_intrvl_col, create tmp_ovrlps_col detailing whether or not the current row's interval
        #   overlaps with the following row's interval (cast as an integer instead of boolean)
        tmp_ovrlps_col = Utilities.generate_random_string()
        assert(tmp_ovrlps_col not in be_df_i.columns.tolist())
        be_df_i[tmp_ovrlps_col] = np.nan
        #-----
        tmp_ovrlps_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(df=be_df_i, col=tmp_ovrlps_col, exact_match=True, assert_single=True)
        #-----
        for i_row in range(be_df_i.shape[0]-1):
            be_df_i.iloc[i_row, tmp_ovrlps_col_idx] = int(be_df_i.iloc[i_row][tmp_intrvl_col].overlaps(be_df_i.iloc[i_row+1][tmp_intrvl_col]))

        #--------------------------------------------------
        # If no overlaps, simply return
        if be_df_i[tmp_ovrlps_col].sum()==0:
            return_df = be_df_i.drop(columns=[tmp_intrvl_col, tmp_ovrlps_col])
            return return_df

        #--------------------------------------------------
        # Take diff of tmp_ovrlps_col to find blocks of overlapping sub-outages
        # The beginning of a block will have tmp_diff_col == +1
        # The end       of a block will have tmp_diff_col == -1
        #-------------------------
        tmp_diff_col = Utilities.generate_random_string()
        assert(tmp_diff_col not in be_df_i.columns.tolist())
        be_df_i[tmp_diff_col] =  be_df_i[tmp_ovrlps_col].diff()
        #-----
        tmp_diff_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(df=be_df_i, col=tmp_diff_col, exact_match=True, assert_single=True)
        #-------------------------
        # The .diff() operation always leaves the first element as a NaN
        # However, if the first element overlaps the second (i.e., if the value==1)
        #   then the diff should be +1
        if be_df_i.iloc[0][tmp_ovrlps_col]==1:
            be_df_i.iloc[0, tmp_diff_col_idx]=1  
        #-------------------------
        # The tmp_ovrlps_col and tmp_diff_col are not set for the last entry (because, tmp_overlaps cols contains info regarding
        #   whether or not the current entry overlaps with the following entry, and, of course, the last entry has no follower)
        # If the second-to-last entry has tmp_ovrlps_col==1, then set tmp_diff_col=-1 for the last entry to signify the end
        #   of an overlapping block
        if be_df_i.iloc[-2][tmp_ovrlps_col]==1:
            be_df_i.iloc[-1, tmp_diff_col_idx]=-1
        else:
            be_df_i.iloc[-1, tmp_diff_col_idx]=0

        #--------------------------------------------------
        # Using the tmp_diff_col, determine the locations of the beginnings and endings of the blocks
        #   and combine into ovrlp_begend_ilocs
        ovrlp_beg_ilocs = be_df_i.reset_index()[be_df_i.reset_index()[tmp_diff_col]==+1].index.tolist()
        ovrlp_end_ilocs = be_df_i.reset_index()[be_df_i.reset_index()[tmp_diff_col]==-1].index.tolist()
        #-----
        assert(len(ovrlp_beg_ilocs)==len(ovrlp_end_ilocs))
        #-----
        ovrlp_begend_ilocs = list(zip(ovrlp_beg_ilocs, ovrlp_end_ilocs))

        #--------------------------------------------------
        # Determine which indices are accounted for by the overlaps
        # NOTE: overlaps are INCLUSIVE, which is why +1 is necessary on the right end of each range
        ilocs_accntd = []
        for ovrlp_begend_i in ovrlp_begend_ilocs:
            ilocs_accntd.extend(list(range(ovrlp_begend_i[0], ovrlp_begend_i[1]+1)))
        #-------------------------
        # Determine which are unaccounted for by the overlaps
        ilocs_unaccntd = list(set(range(be_df_i.shape[0])).difference(set(ilocs_accntd)))

        #--------------------------------------------------
        # Build the return_df
        # The entries in ilocs_unaccntd do not need to be adjusted as they do not have any overlaps
        return_df = be_df_i.iloc[ilocs_unaccntd].copy()
        #-------------------------
        # t_max_col, open_end_col, and any columns ending with '_max' will be set using be_df_i.iloc[ovrlp_begend_i[1]], 
        #   i.e., using the last entry in the block of overlaps.
        # All others are set using the first entry in the block.
        #-----
        max_col_idxs = Utilities_df.find_col_idxs_with_regex(df=be_df_i, regex_pattern=r'.*_max$')
        # Don't want to accidentally set t_min_col using be_df_i.iloc[ovrlp_begend_i[1]]!
        assert(t_min_col not in be_df_i.columns[max_col_idxs].tolist())
        #-----
        for ovrlp_begend_i in ovrlp_begend_ilocs:
            condensed_srs_i = be_df_i.iloc[ovrlp_begend_i[0]].copy()
            #-----
            # Set ending values
            condensed_srs_i.iloc[max_col_idxs] = be_df_i.iloc[ovrlp_begend_i[1]].iloc[max_col_idxs]
            # Set open_end_col, and make absolutely certain t_max_col was set (likely already set using max_col_idxs above)
            condensed_srs_i[open_end_col] = be_df_i.iloc[ovrlp_begend_i[1]][open_end_col]
            condensed_srs_i[t_max_col]    = be_df_i.iloc[ovrlp_begend_i[1]][t_max_col]
            #-----
            # Add condensed entry to return_df
            condensed_srs_i = condensed_srs_i.to_frame().T
            assert((condensed_srs_i.columns==return_df.columns).all())
            return_df = pd.concat([return_df, condensed_srs_i])

        #--------------------------------------------------
        return_df = return_df.drop(columns=[tmp_intrvl_col, tmp_ovrlps_col, tmp_diff_col])
        return_df = DOVSAudit.set_i_outg_in_best_ests_df(
            best_ests_df = return_df, 
            groupby_cols = groupby_cols, 
            sort_cols    = [t_min_col, t_max_col], 
            i_outg_col   = i_outg_col
        )
        #--------------------------------------------------
        return return_df

    @staticmethod
    def combine_overlapping_suboutages(
        be_df, 
        groupby_cols, 
        i_outg_col = 'i_outg', 
        t_min_col = 'winner_min', 
        t_max_col = 'winner_max', 
        open_beg_col = 'open_beg', 
        open_end_col = 'open_end'
    ):
        r"""
        Overlapping suboutages can occur for a given SN/PN.
        This can happen when, e.g., the meter regains power for a short period of time, where a short period
          is defined by the resolution of the voltage data, i.e., 15T
        See OUTG_REC_NB = '13686882' and premise number = '023368334'
            For this outage, it appears all other premised were able to find a power up event to start the second sub-outage
              EXCEPT for premise 023368334.
            Thus, this premise had to use the conservative starting time for the second sub-outage, which caused it to overlap
              with the first sub-outage
        -------------------------
        groupby_cols:
            If SN col is present, should be ['SN'] or ['SN', 'PN']
            If SN col not present (because already combined via DOVSAudit.combine_PNs_in_best_ests_df), should be ['PN']
        """
        #--------------------------------------------------
        return_df = be_df.groupby(groupby_cols, as_index=False, group_keys=False).apply(
            lambda x: DOVSAudit.combine_overlapping_suboutages_for_premise(
                be_df_i      = x, 
                groupby_cols = groupby_cols, 
                i_outg_col   = i_outg_col, 
                t_min_col    = t_min_col, 
                t_max_col    = t_max_col, 
                open_beg_col = open_beg_col, 
                open_end_col = open_end_col
            )
        )
        return return_df
        
        
    def build_best_ests_df(
        self, 
        conservative_estimate = True, 
        audit_selection_method = 'ede only', 
        pct_SNs_required_for_outage_est = 0, 
        use_only_overall_endpoints_of_est_outg_times = False
    ):
        r"""
        """
        #-------------------------
        # At a minimum, ami_df_i and dovs_df_i are absolutely necessary to run analysis.
        # It is nice to have ede_df_i, but the analysis can be run without it.
        assert(self.is_loaded_ami and self.is_loaded_dovs)
        #-------------------------
        res_dict = DOVSAudit.calculate_ci_cmi_w_ami_w_ede_help(
            df                                           = self.ami_df_i, 
            ede_df                                       = self.ede_df_i, 
            dovs_outg_t_beg_end                          = self.dovs_outg_t_beg_end, 
            expand_outg_search_time                      = self.expand_outg_search_time, 
            conservative_estimate                        = conservative_estimate, 
            est_ede_kwargs                               = dict(use_full_ede_outgs=self.use_full_ede_outgs), 
            audit_selection_method                       = audit_selection_method, 
            return_CI_SNs                                = False, 
            use_est_outg_times                           = self.use_est_outg_times, 
            pct_SNs_required_for_outage_est              = pct_SNs_required_for_outage_est, 
            expand_outg_est_search_time                  = self.expand_outg_est_search_time, 
            use_only_overall_endpoints_of_est_outg_times = use_only_overall_endpoints_of_est_outg_times, 
            t_int_beg_col                                = self.ami_df_info_dict['t_int_beg_col'], 
            t_int_end_col                                = self.ami_df_info_dict['t_int_end_col'], 
            value_col                                    = self.ami_df_info_dict['value_col'], 
            SN_col                                       = self.ami_df_info_dict['SN_col'], 
            return_all_best_ests                         = True, 
            return_all_best_ests_type                    = 'pd.DataFrame'
        )
        #-------------------------
        best_ests_df = res_dict['all_best_ests']
        if best_ests_df is None or best_ests_df.shape[0]==0:
            assert(res_dict['CI']==0 and res_dict['CMI']==0)
            self.best_ests_df = best_ests_df
            self.ci  = 0
            self.cmi = 0
            return
        #-------------------------
        # best_ests_df = DOVSAudit.combine_overlapping_suboutages(
            # be_df        = best_ests_df, 
            # groupby_cols = ['SN', 'PN'], 
            # i_outg_col   = 'i_outg', 
            # t_min_col    = 'winner_min', 
            # t_max_col    = 'winner_max', 
            # open_beg_col = 'open_beg', 
            # open_end_col = 'open_end'
        # )
        #-------------------------
        if self.calculate_by_PN and best_ests_df.shape[0]>0:
            try:
                best_ests_df = DOVSAudit.combine_PNs_in_best_ests_df(
                    best_ests_df    = best_ests_df, 
                    likeness_thresh = self.combine_by_PN_likeness_thresh, 
                    SN_col          = 'SN', 
                    PN_col          = 'PN', 
                    i_outg_col      = 'i_outg'     
                )
                #-----
                # # Probably not necessary.....
                # best_ests_df = DOVSAudit.combine_overlapping_suboutages(
                    # be_df        = best_ests_df, 
                    # groupby_cols = ['PN'], 
                    # i_outg_col   = 'i_outg', 
                    # t_min_col    = 'winner_min', 
                    # t_max_col    = 'winner_max', 
                    # open_beg_col = 'open_beg', 
                    # open_end_col = 'open_end'
                # )
            except:
                print(f'outg_rec_nb={self.outg_rec_nb} failed DOVSAudit.combine_PNs_in_best_ests_df\nCRASH IMMINENT!')
                assert(0)
        #-------------------------
        self.best_ests_df = best_ests_df
        self.ci  = self.best_ests_df['PN'].nunique()
        self.cmi = (self.best_ests_df['winner_max']-self.best_ests_df['winner_min']).sum().total_seconds()/60
        
    #****************************************************************************************************
    # Methods to build percent of premises with power as a function of time (from best_ests_df and possibly ami_df)
    #****************************************************************************************************
    @staticmethod
    def build_n_PNs_w_power_srs(
        best_ests_df, 
        ami_df_i, 
        return_pct=True, 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        i_outg_col='i_outg', 
        PN_col_ami_df='aep_premise_nb', 
        open_beg_col = 'open_beg', 
        open_end_col = 'open_end'
    ):
        r"""
        ami_df_i
            - Only needed for finding n_PNs_tot
        """
        #-------------------------
        df = best_ests_df[[PN_col, i_outg_col, t_min_col, t_max_col, open_beg_col, open_end_col]].copy()
        #-------------------------
        all_times_sorted = natsorted(set(df[t_min_col].tolist()+df[t_max_col].tolist()))
        all_times_sorted = [all_times_sorted[0]-pd.Timedelta('1T')] + all_times_sorted
        #-------------------------
        # After all times are grabbed above, adjust any entries in df having open_beg=True and/or open_end=True
        # In order for the calculation to be performed correctly, any entries with open beginning need to have
        #   t_min set to pd.Timestamp.min and those with open ending need to have t_max se to pd.Timestamp.max
        df.loc[df[open_beg_col]==True, t_min_col] = pd.Timestamp.min
        df.loc[df[open_end_col]==True, t_max_col] = pd.Timestamp.max
        #-------------------------
        # For each unique time, determine how many PNs without power
        n_PNs_tot = ami_df_i[PN_col_ami_df].nunique()
        time_and_n_PNs_w_power=dict()
        for time_i in all_times_sorted:
            df_i = df[
                (df[t_min_col]<=time_i) & 
                (df[t_max_col]>time_i)
            ].copy()
            # Should find at most one outage for a given PN!
            # assert(df_i.shape[0]==df_i[PN_col].nunique())
            n_PNs_out_i = df_i.shape[0]
            assert(time_i not in time_and_n_PNs_w_power.keys())
            if return_pct:
                time_and_n_PNs_w_power[time_i] = 100*(n_PNs_tot-n_PNs_out_i)/n_PNs_tot
            else:
                time_and_n_PNs_w_power[time_i] = n_PNs_tot-n_PNs_out_i
        #-------------------------
        n_PNs_w_power_srs = pd.Series(data=time_and_n_PNs_w_power)
        #-------------------------
        return n_PNs_w_power_srs

    @staticmethod
    def simplify_n_PNs_w_power_srs(
        n_PNs_w_power_srs, 
        freq='1T'
    ):
        r"""
        If there are a bunch of entries close together, this will keep only the one with the largest value.
        Intention: To be used when including data point information in text on plot for n_PNs_w_power_srs
        """
        #-------------------------
        assert(isinstance(n_PNs_w_power_srs, pd.Series))
        #-------------------------
        if n_PNs_w_power_srs.name:
            name = n_PNs_w_power_srs.name
        else:
            name = 'pct_w_power'
        #-------------------------
        n_PNs_w_power_srs_simp = n_PNs_w_power_srs.to_frame(name=name).reset_index().groupby(
            pd.Grouper(freq=freq, key='index')
        ).apply(
            lambda x: x.loc[x[name].idxmax()] if x.shape[0]>0 else None
        )
        n_PNs_w_power_srs_simp = n_PNs_w_power_srs_simp.dropna()
        n_PNs_w_power_srs_simp=n_PNs_w_power_srs_simp.set_index('index').squeeze()
        #-------------------------
        return n_PNs_w_power_srs_simp


    @staticmethod
    def get_periods_above_threshold(
        n_PNs_w_power_srs, 
        threshold, 
        return_indices=False
    ):
        r"""
        Given n_PNs_w_power_srs, returns the beginning and ending of periods during which the number
        of premises with power is above threshold

        n_PNs_w_power_srs:
            A series with index equal to timestamps and values equal to the percent of premises with power
        """
        #-------------------------
        assert(isinstance(n_PNs_w_power_srs, pd.Series))
        #-------------------------
        n_PNs_w_power_srs = n_PNs_w_power_srs.copy()
        if n_PNs_w_power_srs.name:
            pct_w_power_col = n_PNs_w_power_srs.name
        else:
            pct_w_power_col = Utilities.generate_random_string()
            n_PNs_w_power_srs.name = pct_w_power_col
        #-------------------------
        above_thresh_col = 'above_thresh'
        above_thresh_df = (n_PNs_w_power_srs>threshold).astype(int).to_frame()
        above_thresh_df=above_thresh_df.rename(columns={pct_w_power_col:above_thresh_col})
        #-----
        tmp_diff_col = Utilities.generate_random_string()
        above_thresh_df[tmp_diff_col] = above_thresh_df[above_thresh_col].diff()
        #-------------------------
        # The .diff() operation always leaves the first element as a NaN
        # However, if the first element pct_w_power>threshold it represents the beginning
        #   of a black and the diff should be +1
        if above_thresh_df.iloc[0][above_thresh_col]==1:
            above_thresh_df.loc[above_thresh_df.index[0], tmp_diff_col] = 1
        #-------------------------
        # Continuous blocks of True (i.e., blocks with pct_w_power > threshold) begin with diff = +1 and
        #   end at the element preceding diff = -1
        block_beg_idxs = above_thresh_df.reset_index().index[above_thresh_df[tmp_diff_col]==1].tolist()
        #-----
        block_end_idxs = above_thresh_df.reset_index().index[above_thresh_df[tmp_diff_col]==-1].tolist()
        block_end_idxs = [x-1 for x in block_end_idxs]
        # If power is ongoing at the end of the data, the procedure above will miss the last end idx
        #   If single point above threshold at end of data ==> tmp_diff_col = +1
        #   If multiple points above threshold at the end of the data, then tmp_diff_col for the last value will be 0
        #     In this case, there does not exist a tmp_diff_col=-1 to signal the end of the block, so must add by hand
        if above_thresh_df.iloc[-1][above_thresh_col]==1:
            block_end_idxs.append(above_thresh_df.shape[0]-1)
        #--------------------------------------------------
        # periods of length one should have idx in both block_beg_idxs and block_end_idxs
        len_1_pd_idxs = natsorted(set(block_beg_idxs).intersection(set(block_end_idxs)))
        #-------------------------
        # Remove the len_1 idxs so the remaineders can be matched
        # NOTE: The following procedure relies on block_beg(end)_idxs being sorting
        block_beg_idxs = natsorted(set(block_beg_idxs).difference(len_1_pd_idxs))
        block_end_idxs = natsorted(set(block_end_idxs).difference(len_1_pd_idxs))
        assert(len(block_beg_idxs)==len(block_end_idxs))
        block_begend_idxs = list(zip(block_beg_idxs, block_end_idxs))
        #-------------------------
        # Include the length 1 blocks!
        block_begend_idxs.extend([(x,x) for x in len_1_pd_idxs])
        #-------------------------
        # Sort
        block_begend_idxs = natsorted(block_begend_idxs, key=lambda x: x[0])
        #-------------------------
        # Sanity check!
        for i in range(len(block_begend_idxs)):
            assert(len(block_begend_idxs[i])==2)
            assert(block_begend_idxs[i][1]>=block_begend_idxs[i][0])
            if i>0:
                assert(block_begend_idxs[i][0]>block_begend_idxs[i-1][1])
        #-------------------------
        # Convert indices to actual values
        block_begend = [(above_thresh_df.index[x[0]], above_thresh_df.index[x[1]]) for x in block_begend_idxs]
        #-------------------------
        if return_indices:
            return block_begend_idxs
        else:
            return block_begend

    @staticmethod
    def get_first_last_above_threshold(
        n_PNs_w_power_srs, 
        threshold
    ):
        r"""
        Get the first and last time (after initial power loss) that power was regained.
        For simple outages, these should be approximately equal
        """
        #-------------------------
        pds_above_thresh = DOVSAudit.get_periods_above_threshold(
            n_PNs_w_power_srs=n_PNs_w_power_srs, 
            threshold=threshold, 
            return_indices=False
        )
        #----------------------------------------------------------------------------------------------------
        # len(pds_above_thresh)==2:
        #     For simple outages, where all initially have power, power is lost suddenly for all, and power is regained 
        #       suddenly for all, there should be two periods above threshold (len(pds_above_thresh)==2)
        #
        # len(pds_above_thresh)==1:
        #     This can arise from a few different situations (#2 is the only usable of the 3):
        #       1: - 'Power' is never lost (i.e., the percent of premises with power never dips below threshold)
        #          -  The single period spans the data
        #          -  The period data is of no use
        #       2: - 'Power' is already out at the beginning of the data 
        #          -  The beginning of the single period may be used to signify when power is restored
        #       3: - 'Power' is on at the beginning of the data, goes out, but is never restored
        #          -  The period data is of no use
        #
        # len(pds_above_thresh)==0:
        #     'Power' is out for the entirety of the data
        #
        # len(pds_above_thresh)>2:
        #     More complicated structure with sub-outages
        #
        #-------------------------
        # How we interpret the data depends on whether there is power at the beginng and end of the data
        power_at_beg = n_PNs_w_power_srs.iloc[0]>threshold
        power_at_end = n_PNs_w_power_srs.iloc[-1]>threshold
        #-------------------------
        if len(pds_above_thresh)==0:
            assert(not power_at_beg and not power_at_end)
            frst_above=None
            last_above=None
        #-------------------------
        elif len(pds_above_thresh)==1:
            assert(power_at_beg or power_at_end)
            last_above=None
            # If power_at_beg and power_at_end, then 'power' is never lost ==> situation #1 (described above)
            # If power_at_beg==False and power_at_end==True  ==> situation #2 (described above)
            # If power_at_beg==True  and power_at_end==False ==> situation #3 (described above)
            if power_at_beg==False and power_at_end==True:
                frst_above = pds_above_thresh[0][0]
            else:
                frst_above = None
        #-------------------------
        else:
            assert(power_at_beg or power_at_end)
            #-----
            # First time above threshold:
            #   If first data point is above threshold, then return beginning of the second period (at index=1)
            #     - i.e., if there is 'power' at the beginning of the data, return the first time power was restored
            #         after it was lost (where 'power' is defined as a minimum threshold of premises having power)
            #   If first data point is not above threshold, the power was initially out, so return beginning of first 
            #     period (at index=0)
            if power_at_beg:
                frst_above = pds_above_thresh[1][0]
            else:
                frst_above = pds_above_thresh[0][0]
            #-----
            # Last time above threshold:
            # If last data point is above threshold, then return beginning of last period (at index=-1)
            #     - i.e., if there is 'power' at the end of the data, return the beginning of the last period
            # If the last data point is not above threshold, the 'power' is out at the end of the data, so
            #   in this case return None
            if power_at_end:
                last_above = pds_above_thresh[-1][0]
            else:
                last_above = None
        #-------------------------
        return frst_above, last_above

    @staticmethod
    def get_n_PNs_w_power_srs_time_to_print(
        n_PNs_w_power_srs, 
        threshold
    ):
        r"""
        In figure, entries with n_PNs_w_power>threshold are typically circled and have both their pct and time
        printed next to the data point.
        However, if there are multiple in a row, this can be messy.
        In such cases, we only want to print the time for those at the beginning and end of each grouping.
        This function identifies which entries should have their times printed
        """
        #-------------------------
        assert(isinstance(n_PNs_w_power_srs, pd.Series))
        #-------------------------
        pds_above_thresh = DOVSAudit.get_periods_above_threshold(
            n_PNs_w_power_srs=n_PNs_w_power_srs, 
            threshold=threshold, 
            return_indices=True
        )
        # Get unique idxs from pds_above_thresh
        print_times = natsorted(set(itertools.chain.from_iterable(pds_above_thresh)))
        #-------------------------
        # Always print the first and last times!
        if 0 not in print_times:
            print_times = [0]+print_times
        if n_PNs_w_power_srs.shape[0]-1 not in print_times:
            print_times = print_times+[n_PNs_w_power_srs.shape[0]-1]
        #-------------------------
        return_print_times_srs = n_PNs_w_power_srs.iloc[print_times]
        #-------------------------
        return return_print_times_srs

    @staticmethod
    def plot_n_PNs_w_power_srs(
        n_PNs_w_power_srs, 
        simp_freq='1T', 
        threshold=None, 
        fig_num=0, 
        title=None, 
        adjust_texts=True, 
        fig_ax=None, 
        **kwargs
    ):
        r"""
        """
        #-------------------------
        threshold_color = kwargs.get('threshold_color', 'red')
        #-------------------------
        if not n_PNs_w_power_srs.name:
            n_PNs_w_power_srs.name = 'pct_w_power'
        #-------------------------
        if fig_ax is None:
            fig,ax = Plot_General.default_subplots(fig_num=fig_num)
        else:
            fig = fig_ax[0]
            ax  = fig_ax[1]
        #-------------------------
        # Line plot of the data
        sns.lineplot(
            ax=ax, 
            data=n_PNs_w_power_srs.to_frame().reset_index(), 
            x='index', 
            y=n_PNs_w_power_srs.name, 
            marker='o'
        )
        #-----
        # Grab txt_srs (depending on simp_freq) to be used when printing
        #   points data on plot
        if simp_freq is not None:
            txt_srs = DOVSAudit.simplify_n_PNs_w_power_srs(
                n_PNs_w_power_srs, 
                freq=simp_freq
            )
        else:
            txt_srs = n_PNs_w_power_srs

        #-------------------------
        if threshold is not None:
            # Draw red circles around any data above threshold
            sns.scatterplot(
                ax=ax, 
                data=n_PNs_w_power_srs[n_PNs_w_power_srs>threshold].to_frame().reset_index(), 
                x='index', 
                y=n_PNs_w_power_srs.name, 
                marker='o', 
                edgecolor=threshold_color, 
                facecolor='none', 
                s=100
            )
            #-----
            # Draw horizontal line indicating threshold value
            ax.axhline(threshold, color=threshold_color, linestyle='dashed', alpha=0.5)
            #-----
            # Grab print_times_srs, used to print full time information next to 
            #   some of the above-threshold values (printed at beginning and end of blocks)
            print_times_srs = DOVSAudit.get_n_PNs_w_power_srs_time_to_print(
                n_PNs_w_power_srs=n_PNs_w_power_srs, 
                threshold=threshold
            )
            #-----
            # Don't want any repeats between print_times_srs and txt_srs
            txt_srs = txt_srs.loc[list(set(txt_srs.index).difference(set(print_times_srs.index)))]           

        #----- Printing points data on plot ---------------
        texts = []
        for time_i, pct_i in txt_srs.items():
            texts.append(ax.annotate(np.round(pct_i, decimals=2), (time_i, pct_i)))
        #-----
        if threshold is not None:
            for time_i, pct_i in print_times_srs.items():
                texts.append(
                    ax.annotate(
                        f"{np.round(pct_i, decimals=2)}, {time_i.strftime('%d %H:%M:%S')}", 
                    (time_i, pct_i), 
                    color=threshold_color
                    )
                )
        if adjust_texts:
            adjustText.adjust_text(texts, ax=ax)
        #--------------------------------------------------
        if title is not None:
            ax.set_title(title, fontdict=dict(fontsize=24))
        #--------------------------------------------------
        return fig,ax

    @staticmethod
    def build_n_PNs_w_power_srs_and_plot(
        best_ests_df, 
        ami_df_i, 
        return_pct=True, 
        simp_freq='1T', 
        threshold=None, 
        fig_num=0, 
        fig_ax=None, 
        title=None, 
        adjust_texts=True, 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        i_outg_col='i_outg', 
        PN_col_ami_df='aep_premise_nb', 
        open_beg_col = 'open_beg', 
        open_end_col = 'open_end', 
        **plot_kwargs
    ):
        r"""
        ami_df_i
            - Only needed for finding n_PNs_tot
        """
        #-------------------------
        n_PNs_w_power_srs = DOVSAudit.build_n_PNs_w_power_srs(
            best_ests_df  = best_ests_df, 
            ami_df_i      = ami_df_i, 
            return_pct    = return_pct, 
            PN_col        = PN_col, 
            t_min_col     = t_min_col, 
            t_max_col     = t_max_col, 
            i_outg_col    = i_outg_col, 
            PN_col_ami_df = PN_col_ami_df, 
            open_beg_col  = open_beg_col, 
            open_end_col  = open_end_col
        )
        #-----
        if not n_PNs_w_power_srs.name:
            n_PNs_w_power_srs.name = 'pct_w_power'
        #-------------------------    
        fig,ax = DOVSAudit.plot_n_PNs_w_power_srs(
            n_PNs_w_power_srs=n_PNs_w_power_srs, 
            simp_freq=simp_freq, 
            threshold=threshold, 
            fig_num=fig_num, 
            title=title, 
            adjust_texts=adjust_texts, 
            fig_ax=fig_ax, 
            **plot_kwargs
        )
        #-------------------------
        return n_PNs_w_power_srs, fig, ax
        
        
    #****************************************************************************************************
    # Methods to identify any overlaps with other DOVS events
    #****************************************************************************************************
    @staticmethod
    def set_i_outg_helper(
        best_ests_df_i, 
        sort_cols=['winner_min', 'winner_max'], 
        i_outg_col='i_outg'
    ):
        r"""
        Intended only for use within DOVSAudit.set_i_outg_in_best_ests_df.
        """
        #-------------------------
        return_df=best_ests_df_i.sort_values(by=sort_cols)
        return_df[i_outg_col]=list(range(return_df.shape[0]))
        #-------------------------
        return return_df

    @staticmethod
    def set_i_outg_in_best_ests_df(
        best_ests_df, 
        groupby_cols=['PN'], 
        sort_cols=['winner_min', 'winner_max'], 
        i_outg_col='i_outg'
    ):
        r"""
        """
        #-------------------------
        # Make sure all the needed columns are contained in best_ests_df
        assert(set(groupby_cols+sort_cols).difference(set(best_ests_df.columns.tolist()))==set())
        #-----
        assert(isinstance(groupby_cols, list))
        assert(isinstance(sort_cols, list))
        #-------------------------
        if i_outg_col not in best_ests_df.columns.tolist():
            best_ests_df[i_outg_col] = None
        #-------------------------
        return_df = best_ests_df.groupby(groupby_cols, as_index=False, group_keys=False).apply(
            lambda x: DOVSAudit.set_i_outg_helper(
                best_ests_df_i = x, 
                sort_cols      = sort_cols, 
                i_outg_col     = i_outg_col
            )
        )
        #-------------------------
        # Make sure all values set
        assert(return_df[i_outg_col].isna().sum()==0)
        #-------------------------
        return return_df
        
    def update_best_ests_and_ci_cmi(
        self, 
        PN_col='PN', 
        winner_min_col='winner_min', 
        winner_max_col='winner_max', 
        i_outg_col='i_outg', 
        keep_col='keep', 
        best_ests_cols=[
            'PN', 
            'i_outg', 
            'winner_min', 
            'winner_max', 
            'conservative_min', 
            'conservative_max', 
            'zero_times_min', 
            'zero_times_max', 
            'open_beg', 
            'open_end'
        ]
    ):
        r"""
        """
        #-------------------------
        assert(set(best_ests_cols).difference(set(self.best_ests_df_w_keep_info.columns.tolist()))==set())
        assert(set([PN_col, i_outg_col, winner_min_col, winner_max_col]).difference(set(best_ests_cols))==set())
        #-------------------------
        self.best_ests_df = self.best_ests_df_w_keep_info[self.best_ests_df_w_keep_info[keep_col]==True][best_ests_cols].copy()
        self.best_ests_df = DOVSAudit.set_i_outg_in_best_ests_df(
            best_ests_df = self.best_ests_df, 
            groupby_cols = [PN_col], 
            sort_cols    = [winner_min_col, winner_max_col], 
            i_outg_col   = i_outg_col
        )
        #-------------------------
        self.ci  = self.best_ests_df[PN_col].nunique()
        self.cmi = (self.best_ests_df[winner_max_col]-self.best_ests_df[winner_min_col]).sum().total_seconds()/60
        

    @staticmethod
    def get_potential_overlapping_dovs(
        PNs, 
        outg_t_beg, 
        outg_t_end, 
        dovs_sql_fcn=DOVSOutages_SQL.build_sql_outage, 
        addtnl_dovs_sql_kwargs=dict(
            CI_NB_min  = 0, 
            CMI_NB_min = 0
        )
    ):
        r"""
        Find any DOVS events which potentially overlap with the premise numbers in PNs during
        the time period defined by outg_t_beg and outg_t_end.

        NOTE: outg_t_beg/_end can come from a DOVS events, or, e.g., from best_ests_df[t_min_col].min() and
              best_ests_df[t_max_col].max()
        -------------------------      
        To find potential DOVS outages overlapping with our findings, for the given PNs, collect all DOVS events whose 
          ending (dt_on_ts) is greater than (or equal to) our earliest estimate 
          and beginning (dt_off_ts_full) is less than (or equal to) our latest estimate
        It may be easier to think about the logic from the standpoint of which DOVS events to EXCLUDE:
          Any DOVS event with ending (dt_on_ts) less than our earliest estimate clearly finished after our window
            and therefore does not overlap
          Any DOVS event with beginning (dt_off_ts_full) greater than our latest estimate clearly began after our
           window and therefore does not overlap.
        -----
        ==>
            A DOVS event is considered overlapping if both of the following are True:
                - it begins before the present outage ends
                    - i.e., outg_t_beg_i <= outg_t_end (or, dt_off_ts_full_i <= outg_t_end)
                - it ends after the present outage ends
                    - i.e., out_t_end_i >= out_t_beg   (or, dt_on_ts >= out_t_beg)
        -------------------------            
        addtnl_dovs_sql_kwargs:
            Any additional arguments to use in build_sql_function_kwargs when building dovs_df
            NOTE: premise_nbs, field_to_split, dt_on_ts, and dt_off_ts_full are handled by the function and 
                    should therefore NOT be included in addtnl_dovs_sql_kwargs
                  These will be set as:
                    - premise_nbs (from the PNs argument)
                    - field_to_split (set to 'premise_nbs')
                    - dt_on_ts (using the outg_t_beg argument)
                      -- See above if confused why dt_on_ts paired with outg_t_beg
                    - dt_off_ts_full (using outg_t_end argument)
                      -- See above if confused why dt_off_ts_full paired with outg_t_end
        """
        #CI_NB_min
        #-------------------------
        assert(isinstance(PNs, list))
        #-------------------------
        dflt_dovs_sql_kwargs = dict(
            premise_nbs=PNs, 
            field_to_split='premise_nbs', 
            dt_on_ts=dict(
                value=outg_t_beg, 
                comparison_operator='>='
            ), 
            dt_off_ts_full=dict(
                value=outg_t_end, 
                comparison_operator='<='
            )        
        )
        #-----
        if addtnl_dovs_sql_kwargs is None:
            dovs_sql_kwargs = dflt_dovs_sql_kwargs
        else:
            assert(isinstance(addtnl_dovs_sql_kwargs, dict))
            #-----
            # Make sure none of the dovs_sql_kwargs handled by this function are included in addtnl_dovs_sql_kwargs
            assert(set(dflt_dovs_sql_kwargs.keys()).intersection(set(addtnl_dovs_sql_kwargs.keys()))==set())
            #-----
            dovs_sql_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = dflt_dovs_sql_kwargs, 
                default_values_dict = addtnl_dovs_sql_kwargs, 
                extend_any_lists    = True, 
                inplace             = False
            )
        #-------------------------
        dovs = DOVSOutages(
            df_construct_type=DFConstructType.kRunSqlQuery, 
            contstruct_df_args=None, 
            init_df_in_constructor=True,
            build_sql_function=dovs_sql_fcn, 
            build_sql_function_kwargs=dovs_sql_kwargs
        )
        dovs_df = dovs.df.copy()
        #-------------------------
        return dovs_df
        

    @staticmethod
    def get_outgs_in_dovs_df_overlapping_interval(
        t_min, 
        t_max, 
        dovs_df, 
        outg_rec_nb_i, 
        outg_rec_nb_col='OUTG_REC_NB', 
        t_min_col='DT_OFF_TS_FULL', 
        t_max_col='DT_ON_TS'
    ):
        r"""
        Given t_min, t_max, and dovs_df, find any entries in dovs_df which overlap with the interval [t_min, t_max].

        t_min,t_max:
            Define the interval during which to look for overlapping DOVS events

        dovs_df:
            pd.DataFrame containing DOVS outages
            There MUST ONLY BE one single row per outg_rec_nb

        outg_rec_nb_i:
            If supplied, outg_rec_nb_i is excluded from any found overlapping DOVS events.
            In this case, it is assumed t_min,t_max and outg_rec_nb_i are related, and therefore one does not
              want to include outg_rec_nb_i in the final results, because, of course, it trivially overlaps with
              itself!
        """
        #-------------------------
        # Make sure needed columns are contained in dovs_df
        assert(set([outg_rec_nb_col, t_min_col, t_max_col]).difference(set(dovs_df.columns.tolist()))==set())
        #-----
        # There should only be a single row for each outg_rec_nb
        assert(dovs_df[outg_rec_nb_col].nunique()==dovs_df.shape[0])
        #-------------------------
        # Grab the interval
        interval_1 = pd.Interval(t_min, t_max)
        #-------------------------
        # For each row (i.e., for each outage), check whether overlaps with interval_1
        # Store that information in tmp_overlap_col
        tmp_overlap_col = Utilities.generate_random_string()
        dovs_df[tmp_overlap_col] = dovs_df.apply(
            lambda x: pd.Interval(x[t_min_col], x[t_max_col]).overlaps(interval_1), 
            axis=1
        )
        #-------------------------
        # Collect the outg_rec_nbs which overlap with interval_1
        overlap_outg_rec_nbs = dovs_df[dovs_df[tmp_overlap_col]==True][outg_rec_nb_col].unique().tolist()
        # If outg_rec_nb_i was supplied, then ensure it is excluded from overlap_outg_rec_nbs
        if outg_rec_nb_i is not None:
            overlap_outg_rec_nbs = [x for x in overlap_outg_rec_nbs if x!=outg_rec_nb_i]
        #-------------------------
        # Drop tmp_overlap_col
        dovs_df = dovs_df.drop(columns=[tmp_overlap_col])    
        #-------------------------
        return overlap_outg_rec_nbs


    @staticmethod
    def get_outgs_in_dovs_df_overlapping_outg_rec_nb_i(
        outg_rec_nb_i, 
        dovs_df, 
        outg_rec_nb_col='OUTG_REC_NB', 
        t_min_col='DT_OFF_TS_FULL', 
        t_max_col='DT_ON_TS'
    ):
        r"""
        Given outg_rec_nb_i and dovs_df, find any entries in dovs_df which overlap with outg_rec_nb_i.
        If outg_rec_nb_i is contained in dovs_df, simply extract on/off times
        If outg_rec_nb_i is NOT contained in dovs_df, run SQL query to retrieve info

        dovs_df:
            pd.DataFrame containing DOVS outages
            There MUST ONLY BE one single row per outg_rec_nb
        """
        #-------------------------
        # Make sure needed columns are contained in dovs_df
        assert(set([outg_rec_nb_col, t_min_col, t_max_col]).difference(set(dovs_df.columns.tolist()))==set())
        #-----
        # There should only be a single row for each outg_rec_nb
        assert(dovs_df[outg_rec_nb_col].nunique()==dovs_df.shape[0])
        #-------------------------
        if outg_rec_nb_i in dovs_df[outg_rec_nb_col].unique().tolist():
            dovs_df_i = dovs_df[dovs_df[outg_rec_nb_col]==outg_rec_nb_i].copy()
        else:
            # Build dovs_df_i
            dovs_i = DOVSOutages(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                contstruct_df_args=None, 
                init_df_in_constructor=True,
                build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                build_sql_function_kwargs=dict(
                    outg_rec_nbs=[outg_rec_nb_i], 
                )
            )
            dovs_df_i = dovs_i.df.copy()
            #-------------------------
            if outg_rec_nb_col not in dovs_df_i.columns.tolist():
                assert('OUTG_REC_NB' in dovs_df_i.columns.tolist())
                dovs_df_i = dovs_df_i.rename(columns={'OUTG_REC_NB':outg_rec_nb_col})
            #-----
            if t_min_col not in dovs_df_i.columns.tolist():
                assert('DT_OFF_TS_FULL' in dovs_df_i.columns.tolist())
                dovs_df_i = dovs_df_i.rename(columns={'DT_OFF_TS_FULL':t_min_col})
            #-----
            if t_max_col not in dovs_df_i.columns.tolist():
                assert('DT_ON_TS' in dovs_df_i.columns.tolist())
                dovs_df_i = dovs_df_i.rename(columns={'DT_ON_TS':t_max_col})
            #-------------------------
        assert(dovs_df_i.shape[0]==1)
        #-------------------------
        overlap_outg_rec_nbs = DOVSAudit.get_outgs_in_dovs_df_overlapping_interval(
            t_min = dovs_df_i.iloc[0][t_min_col], 
            t_max = dovs_df_i.iloc[0][t_max_col], 
            dovs_df = dovs_df, 
            outg_rec_nb_i = outg_rec_nb_i, 
            outg_rec_nb_col = outg_rec_nb_col, 
            t_min_col = t_min_col, 
            t_max_col = t_max_col
        )   
        #-------------------------
        return overlap_outg_rec_nbs
        

    @staticmethod
    def build_overlap_outgs_for_PNs_df_from_srs(
        overlap_outgs_for_PNs, 
        outg_rec_nb_i=None, 
        best_ests_df=None, 
        PN_col_best_ests='PN', 
        return_outg_rec_nb_in_idx=True
    ):
        r"""
        Both functions DOVSAudit.get_outgs_in_dovs_df_overlapping_interval_by_PN and DOVSAudit.get_outgs_in_dovs_df_overlapping_outg_rec_nb_i_by_PN 
          use these methods, so makes sense to put in function
        """
        #-------------------------
        assert(isinstance(overlap_outgs_for_PNs, pd.Series))
        overlap_outgs_for_PNs_df = overlap_outgs_for_PNs.to_frame(name='overlap_outg_rec_nbs')
        overlap_outgs_for_PNs_df['n_overlap'] = overlap_outgs_for_PNs_df['overlap_outg_rec_nbs'].apply(len)
        #-------------------------
        if best_ests_df is not None:
            assert(
                isinstance(best_ests_df, pd.DataFrame) and 
                PN_col_best_ests in best_ests_df.columns.tolist()
            )
            #-----
            overlap_outgs_for_PNs_df['lost_power'] = False
            overlap_outgs_for_PNs_df.loc[
                overlap_outgs_for_PNs_df.index.isin(best_ests_df[PN_col_best_ests].unique().tolist()), 
                'lost_power'
            ] = True
            #-----
            # Re-order columns
            overlap_outgs_for_PNs_df=overlap_outgs_for_PNs_df[['lost_power', 'overlap_outg_rec_nbs', 'n_overlap']]
        #-------------------------
        if return_outg_rec_nb_in_idx and outg_rec_nb_i is not None:
            overlap_outgs_for_PNs_df = Utilities_df.prepend_level_to_MultiIndex(
                df=overlap_outgs_for_PNs_df, 
                level_val=outg_rec_nb_i, 
                level_name='OUTG_REC_NB', 
                axis=0
            )
        #-------------------------
        return overlap_outgs_for_PNs_df    

    @staticmethod
    def get_outgs_in_dovs_df_overlapping_interval_by_PN(
        t_min, 
        t_max, 
        dovs_df,
        outg_rec_nb_i=None, 
        best_ests_df=None, 
        outg_rec_nb_col='OUTG_REC_NB', 
        PN_col='PREMISE_NB', 
        t_min_col='DT_OFF_TS_FULL', 
        t_max_col='DT_ON_TS', 
        PN_col_best_ests='PN', 
        return_outg_rec_nb_in_idx=True
    ):
        r"""
        Given t_min, t_max, and dovs_df, find any entries in dovs_df which overlap with the interval [t_min, t_max].

        t_min,t_max:
            Define the interval during which to look for overlapping DOVS events

        dovs_df:
            pd.DataFrame containing DOVS outages
            There MUST ONLY BE one single row per outg_rec_nb

        outg_rec_nb_i:
            If supplied, outg_rec_nb_i is excluded from any found overlapping DOVS events.
            In this case, it is assumed t_min,t_max and outg_rec_nb_i are related, and therefore one does not
              want to include outg_rec_nb_i in the final results, because, of course, it trivially overlaps with
              itself!

        best_ests_df:
            Premises which actually suffer from an outage are found in best_ests_df.
            Thus, if the user supplies best_ests_df, then best_ests_df[PN_col_best_ests] will be used to 
              fill the 'lost_power' column, which is boolean and denotes whether or not the premise lost power.
            If best_ests_df is not supplied, then the 'lost_power' field is not included in the output
        """
        #-------------------------
        #-------------------------
        overlap_outgs_for_PNs = dovs_df.groupby(PN_col, as_index=True).apply(
            lambda x: DOVSAudit.get_outgs_in_dovs_df_overlapping_interval(
                t_min=t_min, 
                t_max=t_max, 
                dovs_df=x, 
                outg_rec_nb_i=outg_rec_nb_i, 
                outg_rec_nb_col=outg_rec_nb_col, 
                t_min_col=t_min_col, 
                t_max_col=t_max_col
            )
        )
        #-------------------------
        overlap_outgs_for_PNs_df = DOVSAudit.build_overlap_outgs_for_PNs_df_from_srs(
            overlap_outgs_for_PNs=overlap_outgs_for_PNs, 
            outg_rec_nb_i=outg_rec_nb_i, 
            best_ests_df=best_ests_df, 
            PN_col_best_ests=PN_col_best_ests, 
            return_outg_rec_nb_in_idx=return_outg_rec_nb_in_idx
        )
        #-------------------------
        return overlap_outgs_for_PNs_df

    @staticmethod
    def get_outgs_in_dovs_df_overlapping_outg_rec_nb_i_by_PN(
        outg_rec_nb_i, 
        dovs_df, 
        best_ests_df=None, 
        outg_rec_nb_col='OUTG_REC_NB', 
        PN_col='PREMISE_NB', 
        t_min_col='DT_OFF_TS_FULL', 
        t_max_col='DT_ON_TS', 
        PN_col_best_ests='PN', 
        return_outg_rec_nb_in_idx=True
    ):
        r"""
        Given outg_rec_nb_i and dovs_df, find any entries in dovs_df which overlap with outg_rec_nb_i.

        best_ests_df:
            Premises which actually suffer from an outage are found in best_ests_df.
            Thus, if the user supplies best_ests_df, then best_ests_df[PN_col_best_ests] will be used to 
              fill the 'lost_power' column, which is boolean and denotes whether or not the premise lost power.
            If best_ests_df is not supplied, then the 'lost_power' field is not included in the output
        -----
        If outg_rec_nb_i is contained in dovs_df, simply extract on/off times
        If outg_rec_nb_i is NOT contained in dovs_df, run SQL query to retrieve info

        dovs_df:
            pd.DataFrame containing DOVS outages
            There MUST ONLY BE one single row per outg_rec_nb
        """
        #-------------------------
        overlap_outgs_for_PNs = dovs_df.groupby(PN_col, as_index=True).apply(
            lambda x: DOVSAudit.get_outgs_in_dovs_df_overlapping_outg_rec_nb_i(
                outg_rec_nb_i=outg_rec_nb_i, 
                dovs_df=x, 
                outg_rec_nb_col=outg_rec_nb_col, 
                t_min_col=t_min_col, 
                t_max_col=t_max_col
            )
        )
        #-------------------------
        overlap_outgs_for_PNs_df = DOVSAudit.build_overlap_outgs_for_PNs_df_from_srs(
            overlap_outgs_for_PNs=overlap_outgs_for_PNs, 
            outg_rec_nb_i=outg_rec_nb_i, 
            best_ests_df=best_ests_df, 
            PN_col_best_ests=PN_col_best_ests, 
            return_outg_rec_nb_in_idx=return_outg_rec_nb_in_idx
        )
        #-------------------------
        return overlap_outgs_for_PNs_df
        
        
    @staticmethod
    def identify_dovs_overlaps_for_PN_i(
        best_ests_df_i, 
        dovs_df, 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        PN_col_dovs='PREMISE_NB', 
        t_min_col_dovs='DT_OFF_TS_FULL', 
        t_max_col_dovs='DT_ON_TS', 
        outg_rec_nb_col_dovs = 'OUTG_REC_NB', 
        overlap_outg_col='overlap_DOVS', 
        overlap_times_col='overlap_times', 
        keep_col='keep'
    ):
        r"""
        Built with intention of being used in a .groupby(...).apply(lambda x: ) setting.
        So, be careful if using elsewhere.
        NOTE: dovs_df MUST NOT CONTAIN the outg_rec_nb for the data contained in best_ests_df_i.
                IF IT DOES, then all entries will be identified, because there will always be an overlap!
        -----
        This is a bit different from the previous functions (e.g., DOVSAudit.get_outgs_in_dovs_df_overlapping_interval, 
           DOVSAudit.get_outgs_in_dovs_df_overlapping_outg_rec_nb_i_by_PN. etc.)
        Those are trying to find any other DOVS events whose posted beginning/ending time overlaps with the given
           interval/current outages's beg,end times.

        Here, we are dealing with the identified sub-outages found for each premise number.
        For each premise number:
          Determine if any of the identified sub-outages overlap with any of the potentially overlapping DOVS events.
          If no DOVS event overlaps with a given sub-outage, the sub-outage is kept.
          Which sub-outages to keep is tracked in the keep_srs_i series, which is updated at each iteration over
            the potential conflicting outages through keep_boolean_ij.
          The DOVS events overlapping the sub-outages are tracked in the overlap_df_i series
        Note:  At the end, keep_srs_i should equal overlap_df_i[overlap_outg_col].apply(lambda x: len(x)==0)
        """
        #-------------------------
        # Designed for single PN
        assert(best_ests_df_i[PN_col].nunique()==1)
        PN_i = best_ests_df_i[PN_col].unique().tolist()[0]
        #-------------------------
        # Make sure overlap_outg_col/overlap_times_col/keep_col aren't already contained in best_ests_df_i
        if overlap_outg_col in best_ests_df_i.columns.tolist():
            overlap_outg_col = overlap_outg_col + f'_{Utilities.generate_random_string(str_len=4)}'
            assert(overlap_outg_col not in best_ests_df_i.columns.tolist())
        #-----
        if overlap_times_col in best_ests_df_i.columns.tolist():
            overlap_times_col = overlap_times_col + f'_{Utilities.generate_random_string(str_len=4)}'
            assert(overlap_times_col not in best_ests_df_i.columns.tolist())
        #-----
        if keep_col in best_ests_df_i.columns.tolist():
            keep_col = keep_col + f'_{Utilities.generate_random_string(str_len=4)}'
            assert(keep_col not in best_ests_df_i.columns.tolist())
        #-------------------------
        # Grab relevant entries from dovs_df
        dovs_df_i = dovs_df[dovs_df[PN_col_dovs]==PN_i]

        # If no relevant entires, return
        if dovs_df_i.shape[0]==0:
            return_df = best_ests_df_i.copy()
            return_df[overlap_outg_col]  = [[] for _ in range(return_df.shape[0])]
            return_df[overlap_times_col] = [[] for _ in range(return_df.shape[0])]
            return_df[keep_col]          = True
            return return_df

        # Initiate the boolean slicing series
        keep_srs_i    = pd.Series(data=True, index=best_ests_df_i.index)
        overlap_df_i = pd.DataFrame(
            data={
                overlap_outg_col  : [[] for _ in range(best_ests_df_i.shape[0])], 
                overlap_times_col : [[] for _ in range(best_ests_df_i.shape[0])]
            }, 
            index=best_ests_df_i.index
        )

        # Iterate through (possibly) conflicting outages, finding the slicing boolean (keep_boolean_ij) for each
        #   and updating the overall slicing boolean (keep_srs_i)
        # In general, slicing booleans contain information regarding which sub-outages should be kept for this
        #   specific premise number (i.e., each row represents a sub-outage).
        for idx_j, cnflct_outg_j in dovs_df_i.iterrows():
            # I guess it's actually easier to essentially find those which overlap and then take the logical opposite.
            # To find those which don't overlap directly would need two separate and statements, I believe, whereas
            #   this only needs one
            keep_boolean_ij = ~(
                (best_ests_df_i[t_min_col]<=cnflct_outg_j[t_max_col_dovs]) & 
                (best_ests_df_i[t_max_col]>=cnflct_outg_j[t_min_col_dovs])
            )
            assert(all(keep_srs_i.index==keep_boolean_ij.index)) # Sanity, not really necessary...
            #----------
            # If cnflct_outg_j overlaps with any of the entries of best_ests_df_i, add the overlapping outg_rec_nb
            #   to the appropriate list in overlap_df_i
            # The overlap information is simply ~keep_boolean_ij (because we only keep if non-overlapping)
            overlap_df_i.loc[~keep_boolean_ij, overlap_outg_col] = overlap_df_i.loc[~keep_boolean_ij, overlap_outg_col].apply(
                lambda x: x+[cnflct_outg_j[outg_rec_nb_col_dovs]]
            )
            #-----
            overlap_df_i.loc[~keep_boolean_ij, overlap_times_col] = overlap_df_i.loc[~keep_boolean_ij, overlap_times_col].apply(
                lambda x: x+[(cnflct_outg_j[t_min_col_dovs], cnflct_outg_j[t_max_col_dovs])]
            )
            #----------
            keep_srs_i = keep_srs_i & keep_boolean_ij

        #-------------------------
        # Sanity check
        assert(overlap_df_i[overlap_outg_col].apply(lambda x: len(x)==0).equals(keep_srs_i))
        assert((overlap_df_i.index==keep_srs_i.index).all())
        #-------------------------
        df_to_merge = pd.merge(
            overlap_df_i, 
            keep_srs_i.to_frame(name=keep_col), 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        #-------------------------
        assert(set(best_ests_df_i.index).symmetric_difference(set(df_to_merge.index))==set())
        assert(set(best_ests_df_i.columns).intersection(set(df_to_merge.columns))==set())
        #-----
        return_df = pd.merge(
            best_ests_df_i, 
            df_to_merge, 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        #-------------------------        
        return return_df
        
        
    @staticmethod
    def identify_dovs_overlaps_from_best_ests(
        best_ests_df, 
        outg_rec_nb, 
        dovs_df, 
        get_ptntl_ovrlp_dovs_kwargs=None, 
        assert_no_overlaps=True, 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        PN_col_dovs='PREMISE_NB', 
        t_min_col_dovs='DT_OFF_TS_FULL', 
        t_max_col_dovs='DT_ON_TS', 
        outg_rec_nb_col_dovs='OUTG_REC_NB', 
        overlap_outg_col='overlap_DOVS', 
        overlap_times_col='overlap_times', 
        keep_col='keep'
    ):
        r"""
        If dovs_df is not supplied by user, it will built.
            Default behavior is to query using the premise number and time restrictions from best_ests_df.
            However, one could override this behavior by supplying different arguments for get_ptntl_ovrlp_dovs_kwargs.

        If dovs_df is supplied by the user:
            - it must include outg_rec_nb
            - it must contain a 'PREMISE_NB' column

        get_ptntl_ovrlp_dovs_kwargs:
            A dictionary with key/value pairs suitable for input into DOVSAudit.get_potential_overlapping_dovs.
            The keys and default values are:
                - PNs: 
                    -- premise numbers to query
                    -- default = best_ests_df[PN_col].unique().tolist()
                - outg_t_beg: 
                    -- beginning time to look for overlaps
                    -- default = best_ests_df[t_min_col].min()
                - outg_t_end: 
                    -- ending time to look for overlaps
                    -- default = best_ests_df[t_max_col].max()
                - dovs_sql_fcn: 
                    -- SQL functin to use for query 
                    -- default = DOVSOutages_SQL.build_sql_outage
                - addtnl_dovs_sql_kwargs: 
                    -- Any additional kwargs to input into dovs_sql_fcn when running query
                    -- default = dict(
                            include_DOVS_PREMISE_DIM=True, 
                            CI_NB_min  = 0, 
                            CMI_NB_min = 0
                        )
                    -- NOTE: premise_nbs, field_to_split, dt_on_ts, and dt_off_ts_full are handled by 
                             DOVSAudit.get_potential_overlapping_dovs and should therefore NOT be included in 
                             addtnl_dovs_sql_kwargs
        """
        #--------------------------------------------------
        # Build dovs, if needed
        #-----
        # As mentioned above, the functionality here allows the user to override the default behavior by supplying
        #   different arguments for get_ptntl_ovrlp_dovs_kwargs.
        #-------------------------
        if dovs_df is None:
            dflt_get_ptntl_ovrlp_dovs_kwargs = dict(
                PNs                    = best_ests_df[PN_col].unique().tolist(), 
                outg_t_beg             = best_ests_df[t_min_col].min(), 
                outg_t_end             = best_ests_df[t_max_col].max(), 
                dovs_sql_fcn           = DOVSOutages_SQL.build_sql_outage, 
                addtnl_dovs_sql_kwargs = dict(
                    include_DOVS_PREMISE_DIM = True, 
                    CI_NB_min                = 0, 
                    CMI_NB_min               = 0
                )
            )
            #-----
            if get_ptntl_ovrlp_dovs_kwargs is None:
                get_ptntl_ovrlp_dovs_kwargs = dflt_get_ptntl_ovrlp_dovs_kwargs
            else:
                get_ptntl_ovrlp_dovs_kwargs = Utilities.supplement_dict_with_default_values(
                    to_supplmnt_dict    = get_ptntl_ovrlp_dovs_kwargs, 
                    default_values_dict = dflt_get_ptntl_ovrlp_dovs_kwargs, 
                    extend_any_lists    = False, 
                    inplace             = False
                )
            #-----
            dovs_df = DOVSAudit.get_potential_overlapping_dovs(**get_ptntl_ovrlp_dovs_kwargs)
            #-------------------------
            if outg_rec_nb_col_dovs not in dovs_df.columns.tolist():
                assert('OUTG_REC_NB' in dovs_df.columns.tolist())
                dovs_df = dovs_df.rename(columns={'OUTG_REC_NB':outg_rec_nb_col_dovs})
            #-----
            if t_min_col_dovs not in dovs_df.columns.tolist():
                assert('DT_OFF_TS_FULL' in dovs_df.columns.tolist())
                dovs_df = dovs_df.rename(columns={'DT_OFF_TS_FULL':t_min_col_dovs})
            #-----
            if t_max_col_dovs not in dovs_df.columns.tolist():
                assert('DT_ON_TS' in dovs_df.columns.tolist())
                dovs_df = dovs_df.rename(columns={'DT_ON_TS':t_max_col_dovs})
        #--------------------------------------------------
        # Mainly sanity checks....
        #-----
        nec_dovs_cols = [outg_rec_nb_col_dovs, PN_col_dovs, t_min_col_dovs, t_max_col_dovs]
        assert(set(nec_dovs_cols).difference(set(dovs_df.columns.tolist()))==set())
        #-----
        outg_rec_nbs_and_times = dovs_df[
            [outg_rec_nb_col_dovs, t_min_col_dovs, t_max_col_dovs]
        ].drop_duplicates().set_index(outg_rec_nb_col_dovs)
        #-----
        # Should only be one entry per outg_rec_nb
        assert(outg_rec_nbs_and_times.index.nunique()==outg_rec_nbs_and_times.shape[0])
        #-------------------------
        # For a given PN, there definitely should not be any overlap between two DOVS OUTG_REC_NBs
        #   i.e., a given PN cannot belong to two outages at the same time!
        # ==========
        # OLD METHOD:
        #     Test this by checking whether Utilities.get_overlap_intervals returns an object with
        #       the same length as the input for each PN (if there were any overlaps, the returned
        #       object would have a shorter length)
        # REASON FOR CHANGE:
        #     This is actually a little bit too strict.
        #     For a given PN, if potentially overlapping DOVS outages overlap with each other but do not
        #       overlap with outg_rec_nb, they do not affect the processing of this outage, and therefore
        #       are not of issue.
        #     The old method was stricter, whereas the new method is more relaxed and matches the description above
        #-----
        # PNs_w_mult_outgs_smltnsly: multiple outages simultaneously
    #     PNs_w_mult_outgs_smltnsly = dovs_df.groupby(PN_col_dovs, as_index=True, group_keys=False).apply(
    #         lambda x: len(Utilities.get_overlap_intervals(x[[t_min_col_dovs, t_max_col_dovs]].values))!=x.shape[0]
    #     )
    #     if assert_no_overlaps:
    #         assert(not PNs_w_mult_outgs_smltnsly.any())
        # END: OLD METHOD
        # ==========
        overlap_outgs_for_PNs_df = DOVSAudit.get_outgs_in_dovs_df_overlapping_outg_rec_nb_i_by_PN(
            outg_rec_nb_i    = outg_rec_nb, 
            dovs_df          = dovs_df, 
            best_ests_df     = best_ests_df, 
            outg_rec_nb_col  = outg_rec_nb_col_dovs, 
            PN_col           = PN_col_dovs, 
            t_min_col        = t_min_col_dovs, 
            t_max_col        = t_max_col_dovs, 
            PN_col_best_ests = PN_col
        )
        if assert_no_overlaps:
            #assert((overlap_outgs_for_PNs_df['n_overlap']>0).sum()==0)
            assert((overlap_outgs_for_PNs_df[overlap_outgs_for_PNs_df['lost_power']==True]['n_overlap']>0).sum()==0)
        #--------------------------------------------------
        # Do not need current outg_rec_nb anymore, only the others
        dovs_df = dovs_df[dovs_df[outg_rec_nb_col_dovs] != outg_rec_nb].copy()
        #--------------------------------------------------
        # Simple-minded way would be to exclude any entries in best_ests_df which overlap 
        #   with the outage times in outg_rec_nbs_and_times.
        #   This would be correct in most all cases
        # HOWEVER, to be completely correct, this should be done at the PREMISE_NB level!
        #-----
        # I don't want to simply merge best_ests_df with dovs_df (by PN) because the former can have multiple sub-outages
        #   per PN, and the latter can have multiple (possibly) conflicting outages.
        # Merging the two would be sloppy.
        # Therefore, I'll use groupby
        return_df = best_ests_df.groupby(PN_col, as_index=False, group_keys=False).apply(
            lambda x: DOVSAudit.identify_dovs_overlaps_for_PN_i(
                best_ests_df_i       = x, 
                dovs_df              = dovs_df, 
                PN_col               = PN_col, 
                t_min_col            = t_min_col, 
                t_max_col            = t_max_col, 
                PN_col_dovs          = PN_col_dovs, 
                t_min_col_dovs       = t_min_col_dovs, 
                t_max_col_dovs       = t_max_col_dovs, 
                outg_rec_nb_col_dovs = outg_rec_nb_col_dovs, 
                overlap_outg_col     = overlap_outg_col, 
                overlap_times_col    = overlap_times_col, 
                keep_col             = keep_col
            )
        )
        #--------------------------------------------------
        return return_df
        
        
    @staticmethod
    def expand_removed_srs(
        removed_srs, 
        ami_df_i, 
        best_ests_df, 
        PN_col_ami='aep_premise_nb', 
        time_col_ami='starttimeperiod_local', 
        PN_col_be='PN', 
        t_min_col_be='winner_min', 
        t_max_col_be='winner_max', 
        keep_col_be='keep', 
    ):
        r"""
        Expand the bounds of removed_srs in DOVSAudit.set_removed_due_to_overlap_in_ami_df_i.
        -----
        The point of building the 'removed_due_to_overlap' in ami_df_i is simply for plotting purposes.
        If there is a block in the middle of ami_df_i which is marked 'removed_due_to_overlap'=True, and the data with those
          values removed is plotted, the figure will still appear as if there are data points for the removed times.
        The reason being that markers are not drawn, so it is not obvious where datapoints are.
        The value to the left of the points to the removed will be connected with a straight line to those to the right of 
          the points to be removed.
        -----
        The purpose of this function is to make the plot a little clearer.
        If a removal period happens before the first sub-outage for a PN, expand the left point of the removal period to be
          equal to the minimum time value in ami_df_i for the PN.
        If a removal period happens after the last sub-outage for a PN, expand the right point of the removal period to be
          equal to the maximum time value in ami_df_i for the PN.
        """
        #-------------------------
        winner_minmax_by_PN = best_ests_df[best_ests_df[keep_col_be]==True].groupby(PN_col_be).agg(
            {t_min_col_be:'min', t_max_col_be:'max'}
        )
        #-----
        ami_minmax_times_by_PN = ami_df_i.groupby(PN_col_ami).agg(
            {time_col_ami:['min', 'max']}
        )
        assert(set(removed_srs.index).difference(set(ami_minmax_times_by_PN.index))==set())
        #-------------------------
        # Iterate through PNs in removed_srs and build expanded values
        return_srs = dict()
        for PN_j, overlap_times_j in removed_srs.items():
            # If PN_j did not actually lose power, there will be no entry in winner_minmax_by_PN
            # In such a case, leave values as they are
            if PN_j not in winner_minmax_by_PN.index:
                overlap_times_j_exp = overlap_times_j
            else:
                overlap_times_j_exp = []
                for overlap_pd_j_k in overlap_times_j:
                    # If a removal period happens before the first sub-outage for a PN, expand the left point of 
                    #   the removal period to be equal to the minimum time value in ami_df_i for the PN.
                    if overlap_pd_j_k[1] < winner_minmax_by_PN.loc[PN_j, t_min_col_be]:
                        overlap_pd_j_k_0 = ami_minmax_times_by_PN.loc[PN_j, (time_col_ami, 'min')]
                    else:
                        overlap_pd_j_k_0 = overlap_pd_j_k[0]
                    #-----
                    # If a removal period happens after the last sub-outage for a PN, expand the right point of
                    #   the removal period to be equal to the maximum time value in ami_df_i for the PN.        
                    if overlap_pd_j_k[0] > winner_minmax_by_PN.loc[PN_j, t_max_col_be]:
                        overlap_pd_j_k_1 = ami_minmax_times_by_PN.loc[PN_j, (time_col_ami, 'max')]
                    else:
                        overlap_pd_j_k_1 = overlap_pd_j_k[1]
                    #-----
                    overlap_times_j_exp.append((overlap_pd_j_k_0, overlap_pd_j_k_1)) 
            assert(PN_j not in return_srs.keys())
            return_srs[PN_j] = overlap_times_j_exp
        #-------------------------
        return_srs = pd.Series(return_srs, name=removed_srs.name)
        return return_srs


    @staticmethod
    def set_removed_due_to_overlap_helper(
        ami_df_i, 
        PN_j, 
        overlap_times_j, 
        time_col='starttimeperiod_local', 
        PN_col='aep_premise_nb', 
        removed_due_to_overlap_col='removed_due_to_overlap'
    ):
        r"""
        ONLY INTENDED FOR USE INSIDE OF DOVSAudit.set_removed_due_to_overlap_in_ami_df_i

        In order to set removed_due_to_overlap, the logic will be: set if PN is correct and time falls within
          any one of the overlap_times taken from best_ests_df
        In code, this amounts to:
            ami_df_i[
                (ami_df_i[PN_col]==PN_j) & 
                (    
                    (        
                        (ami_df_i[time_col] >= overlap_time_min_j_0) &
                        (ami_df_i[time_col] <= overlap_time_max_j_0)
                    )
                    ||
                    .
                    .
                    .
                    (        
                        (ami_df_i[time_col] >= overlap_time_min_j_n) &
                        (ami_df_i[time_col] <= overlap_time_max_j_n)
                    )    
                )
            ]
        Due to the unknown number of overlap times, this is easiest to accomplish using DFSlicers    
        """
        #-------------------------
        # Make sure all elements in overlap_times_j are lists/tuples of length 2
        assert(Utilities.are_all_list_elements_one_of_types(overlap_times_j, [list, tuple]))
        #-----
        if len(overlap_times_j)==0:
            return ami_df_i
        #-----
        assert(Utilities.are_list_elements_lengths_homogeneous(overlap_times_j, 2))
        #-------------------------
        pn_slicer = slicer = DFSlicer(
            single_slicers = [dict(column=PN_col, value=PN_j, comparison_operator='==')]
        )
        #-----
        pn_slcr_bool_srs = pn_slicer.get_slicing_booleans(
            df=ami_df_i
        )
        #-------------------------
        time_slicers = []
        for overlap_time_min, overlap_time_max in overlap_times_j:
            ts_i = DFSlicer(
                single_slicers = [
                    dict(column=time_col, value=overlap_time_min, comparison_operator='>='), 
                    dict(column=time_col, value=overlap_time_max, comparison_operator='<=')
                ], 
                join_single_slicers='and'
            )
            time_slicers.append(ts_i)
        #-----
        time_slcr_bool_srs = DFSlicer.combine_slicers_and_get_slicing_booleans(
            df=ami_df_i, 
            slicers=time_slicers, 
            join_slicers='or', 
            apply_not=False
        )
        #-------------------------
        assert((time_slcr_bool_srs.index==pn_slcr_bool_srs.index).all())
        slcr_bool_srs = time_slcr_bool_srs & pn_slcr_bool_srs
        #-------------------------
        ami_df_i.loc[slcr_bool_srs, removed_due_to_overlap_col] = True
        #-------------------------
        return ami_df_i

    @staticmethod
    def set_removed_due_to_overlap_in_ami_df_i(
        ami_df_i, 
        best_ests_df, 
        PN_col='aep_premise_nb', 
        time_idfr='starttimeperiod_local', 
        PN_col_be='PN', 
        t_min_col_be='winner_min', 
        t_max_col_be='winner_max', 
        keep_col_be='keep', 
        overlap_times_col_be='overlap_times', 
        removed_due_to_overlap_col='removed_due_to_overlap', 
        expand_removed_times=True
    ):
        r"""
        In order to set removed_due_to_overlap, the logic will be: set if PN is correct and time falls within
          any one of the overlap_times taken from best_ests_df
        In code, this amounts to:
            ami_df_i[
                (ami_df_i['aep_premise_nb']==PN_ij) & 
                (    
                    (        
                        (ami_df_i[time_col] >= overlap_time_min_0) &
                        (ami_df_i[time_col] <= overlap_time_max_0)
                    )
                    ||
                    .
                    .
                    .
                    (        
                        (ami_df_i[time_col] >= overlap_time_min_n) &
                        (ami_df_i[time_col] <= overlap_time_max_n)
                    )    
                )
            ]
        Due to the unknown number of overlap times, this is easiest to accomplish using DFSlicers    
        """
        #--------------------------------------------------
        # In order for method to work, time information must be in column, not index.
        # If found in index, call reset_index
        time_idfr_loc = Utilities_df.get_idfr_loc(
            df=ami_df_i, 
            idfr=time_idfr
        )
        #-------------------------
        if time_idfr_loc[1]:
            # For now, make sure ami_df_i has a single index (if MultiIndex, this functionality can be built out later)
            # time information is located in index, so reset_index must be called for methods to work
            # Looking back at final code: I'm not sure anything additional needs to be done for MultiIndex case
            assert(ami_df_i.index.nlevels==1)
            #-----
            if ami_df_i.index.names[time_idfr_loc[0]]:
                time_col = ami_df_i.index.names[time_idfr_loc[0]]
            else:
                time_col = 'time_idx_'+Utilities.generate_random_string(str_len=4)
                assert(time_col not in ami_df_i.columns.tolist())
                assert(time_col not in list(ami_df_i.index.names))
                ami_df_i.index = ami_df_i.index.set_names(time_col, level=time_idfr_loc[0])
            #-----
            idx_cols = list(ami_df_i.index.names)
            ami_df_i = ami_df_i.reset_index()
        else:
            time_col = time_idfr_loc[0]
            idx_cols=None
        #-------------------------
        assert(set([PN_col, time_col]).difference(set(ami_df_i.columns.tolist()))==set())
        #--------------------------------------------------
        # From best_ests_df, find all sub-outages to be removed by PN
        assert(set([PN_col_be, keep_col_be, overlap_times_col_be]).difference(set(best_ests_df.columns.tolist()))==set())
        removed_srs = best_ests_df[best_ests_df[keep_col_be]==False].copy()
        removed_srs = removed_srs.groupby(PN_col_be)[overlap_times_col_be].sum()
        if expand_removed_times:
            removed_srs = DOVSAudit.expand_removed_srs(
                removed_srs  = removed_srs, 
                ami_df_i     = ami_df_i, 
                best_ests_df = best_ests_df, 
                PN_col_ami   = PN_col, 
                time_col_ami = time_col, 
                PN_col_be    = PN_col_be, 
                t_min_col_be = t_min_col_be, 
                t_max_col_be = t_max_col_be, 
                keep_col_be  = keep_col_be 
            )
        #--------------------------------------------------
        # Iterate through each PN in removed_srs and use DOVSAudit.set_removed_due_to_overlap_helper to set the appropriate
        #   values in ami_df_i
        ami_df_i[removed_due_to_overlap_col] = False
        for PN_j, overlap_times_j in removed_srs.items():
            ami_df_i = DOVSAudit.set_removed_due_to_overlap_helper(
                ami_df_i                   = ami_df_i, 
                PN_j                       = PN_j, 
                overlap_times_j            = overlap_times_j, 
                time_col                   = time_col, 
                PN_col                     = PN_col, 
                removed_due_to_overlap_col = removed_due_to_overlap_col
            )
        #--------------------------------------------------
        # If .reset_index was called earlier, set index back to original values
        if idx_cols is not None:
            ami_df_i = ami_df_i.set_index(idx_cols)
        #--------------------------------------------------
        return ami_df_i
        
        
    def identify_overlaps(
        self, 
        overlaps_dovs_sql_fcn=DOVSOutages_SQL.build_sql_outage, 
    ):
        r"""
        Identify and handle any overlaps with other DOVS events
        This function will:
            1. Build self.best_ests_df_w_keep_info
            2. Build self.overlap_outgs_for_PNs_df
            3. Update self.best_ests_df by removing any with overlaps
            4. In self.ami_df_i, mark any entries which were essentially removed via the 
                 identify_dovs_overlaps_from_best_ests and removal procedure
        """
        #---------------------------------------------------------------------------
        if self.best_ests_df is None or self.best_ests_df.shape[0]==0:
            return
        #---------------------------------------------------------------------------
        dovs_outg_t_beg_end = self.dovs_outg_t_beg_end
        dovs_outg_t_beg = dovs_outg_t_beg_end[0]
        dovs_outg_t_end = dovs_outg_t_beg_end[1]
        #-------------------------
        # I'll need potential overlapping DOVS df using (dovs_outg_t_beg, dovs_outg_t_end) and also
        #   using (best_ests_df[t_min_col].min(), best_ests_df[t_max_col].max())
        # Since these typically have significant overlap, grab potential overlapping DOVS events for 
        #   all, and then subset as needed
        #-----
        # NOTE: ptntl_ovrlp_dovs_df below WILL NOT BE CONSOLIDATED, meaning that, in general, each outage will have
        #         multiple rows with a single row for each premise.
        #       As the overlap procedure will be done at the PN level, this is the desired form
        #-----
        # NOTE: ptntl_ovrlp_dovs_df is built directly from DOVS database by running SQL query.
        #       Therefore, the columns are predictable, and one does NOT need to use self.dovs_df_info_dict
        ptntl_ovrlp_dovs_df = DOVSAudit.get_potential_overlapping_dovs(
            PNs          = self.ami_df_i[self.ami_df_info_dict['PN_col']].unique().tolist(), 
            outg_t_beg   = np.min([dovs_outg_t_beg, self.best_ests_df['winner_min'].min()]), 
            outg_t_end   = np.max([dovs_outg_t_end, self.best_ests_df['winner_max'].max()]), 
            dovs_sql_fcn = overlaps_dovs_sql_fcn, 
            addtnl_dovs_sql_kwargs=dict(
                CI_NB_min  = 0, 
                CMI_NB_min = 0
            )
        )
        #-----
        ptntl_ovrlp_dovs_df_out_times = ptntl_ovrlp_dovs_df[
            (ptntl_ovrlp_dovs_df['DT_OFF_TS_FULL'] <= dovs_outg_t_end) & 
            (ptntl_ovrlp_dovs_df['DT_ON_TS']       >= dovs_outg_t_beg)
        ]
        ptntl_ovrlp_dovs_df_est_times = ptntl_ovrlp_dovs_df[
            (ptntl_ovrlp_dovs_df['DT_OFF_TS_FULL'] <= self.best_ests_df['winner_max'].max()) & 
            (ptntl_ovrlp_dovs_df['DT_ON_TS']       >= self.best_ests_df['winner_min'].min())
        ]

        #---------------------------------------------------------------------------
        # First, find if any other DOVS outages overlap with the current one
        #-----
        # NOTE: Supply best_ests_df to get_outgs_in_dovs_df_overlapping_outg_rec_nb_i_by_PN below so 'lost_power'
        #       column is output
        overlap_outgs_for_PNs_df = DOVSAudit.get_outgs_in_dovs_df_overlapping_outg_rec_nb_i_by_PN(
            outg_rec_nb_i    = self.outg_rec_nb, 
            dovs_df          = ptntl_ovrlp_dovs_df_out_times, 
            best_ests_df     = self.best_ests_df, 
            outg_rec_nb_col  = 'OUTG_REC_NB', 
            PN_col           = 'PREMISE_NB', 
            t_min_col        = 'DT_OFF_TS_FULL', 
            t_max_col        = 'DT_ON_TS', 
            PN_col_best_ests = 'PN'
        )
        self.overlap_outgs_for_PNs_df = overlap_outgs_for_PNs_df
        #-------------------------
        # If any PN has one or more overlapping DOVS events, output info to file
        n_PNs_w_overlap = (overlap_outgs_for_PNs_df['n_overlap']>0).sum()
        if n_PNs_w_overlap>0:
            print(f'Need to output to new file\n\tn_PNs_w_overlap={n_PNs_w_overlap}')
        #-------------------------
        # If any PN which lost power has one or more overlapping DOVS events, stop analysis
        n_out_PNs_w_overlap = overlap_outgs_for_PNs_df[overlap_outgs_for_PNs_df['lost_power']==True]['overlap_outg_rec_nbs'].apply(lambda x: len(x)>0).sum()
        if n_out_PNs_w_overlap>0:
            print('STOP analysis')
            return

        #---------------------------------------------------------------------------
        # Procedure above ensure that the current outage, as defined by DOVS, does not overlap with any other DOVS outages.
        # Now, we are safe to check whether any of our estimates for the current outage overlap with any other DOVS outages!
        # ==> Should be safe to set assert_no_overlaps=True in identify_dovs_overlaps_from_best_ests
        #-----
        # NOTE: Both self.best_ests_df and ptntl_ovrlp_dovs_df_est_times built internally, and therefore have predictable
        #         columns, so no info_dicts are needed
        best_ests_df_w_keep_info = DOVSAudit.identify_dovs_overlaps_from_best_ests(
            best_ests_df                = self.best_ests_df, 
            outg_rec_nb                 = self.outg_rec_nb, 
            dovs_df                     = ptntl_ovrlp_dovs_df_est_times, 
            get_ptntl_ovrlp_dovs_kwargs = None, 
            assert_no_overlaps          = True, 
            PN_col                      = 'PN', 
            t_min_col                   = 'winner_min', 
            t_max_col                   = 'winner_max', 
            PN_col_dovs                 = 'PREMISE_NB', 
            t_min_col_dovs              = 'DT_OFF_TS_FULL', 
            t_max_col_dovs              = 'DT_ON_TS', 
            outg_rec_nb_col_dovs        = 'OUTG_REC_NB', 
            overlap_outg_col            = 'overlap_DOVS', 
            overlap_times_col           = 'overlap_times', 
            keep_col                    = 'keep'
        )
        self.best_ests_df_w_keep_info = best_ests_df_w_keep_info    
        
        
    #****************************************************************************************************
    # Methods to try to rectify any overlaps with other DOVS events
    #****************************************************************************************************
    @staticmethod
    def get_overlap_interval(
        intrvl_1, 
        intrvl_2
    ):
        r"""
        Find the overlap of intervals intrvl_1 and intrvl_2.
        If no overlap exists, return None.
        Otherwise, a list of length 2 containing the endpoints of the overlap region will be returned
        """
        #-------------------------
        assert(len(intrvl_1)==len(intrvl_2)==2)
        #-------------------------
        if not pd.Interval(*intrvl_1).overlaps(pd.Interval(*intrvl_2)):
            return None
        #-------------------------
        # Overlap region will range from max of mins to min of maxs
        ovrlp = [np.max([intrvl_1[0], intrvl_2[0]]), np.min([intrvl_1[1], intrvl_2[1]])]
        return ovrlp

    @staticmethod
    def get_overlap_interval_len(
        intrvl_1, 
        intrvl_2, 
        norm_by=None
    ):
        r"""
        Return the length of the overlap of intervals intrvl_1 and intrvl_2.
        If desired, this can be normalized by either the length of intrvl_1 or that of intrvl_2
          by setting norm_by equal to 1 or 2, respectively
        """
        #-------------------------
        acceptable_norm_by = [None, 1, '1', 2, '2']
        assert(norm_by in acceptable_norm_by)
        #-------------------------
        ovrlp = DOVSAudit.get_overlap_interval(
            intrvl_1 = intrvl_1, 
            intrvl_2 = intrvl_2
        )
        #-------------------------
        if ovrlp is None:
            return 0
        #-------------------------
        ovrlp_len = ovrlp[1]-ovrlp[0]
        if norm_by is None:
            return ovrlp_len
        elif norm_by==1 or norm_by=='1':
            return ovrlp_len/(intrvl_1[1]-intrvl_1[0])
        elif norm_by==2 or norm_by=='2':
            return ovrlp_len/(intrvl_2[1]-intrvl_2[0])
        else:
            assert(0)
            
    @staticmethod
    def set_initial_to_adjust_df(
        be_df, 
        be_df_cols, 
        addtnl_cols=None
    ):
        r"""
        """
        #-------------------------
        if be_df_cols is None:
            be_df_cols = be_df.columns.tolist()
        #-------------------------
        if addtnl_cols is not None:
            assert(isinstance(addtnl_cols, list))
            # Don't want to change be_df_cols outside of this function, so copy!
            be_df_cols = copy.deepcopy(be_df_cols)
            #-----
            be_df_cols.extend(addtnl_cols)
        #-------------------------
        to_adjust = be_df[be_df_cols].reset_index(drop=True)
        #-----
        to_adjust['adjustment']       = None
        to_adjust['resolved']         = np.nan
        to_adjust['resolved_details'] = ''
        #-------------------------
        return to_adjust


    @staticmethod
    def quantify_closeness_of_intervals(
        intrvl_1, 
        intrvl_2, 
        euclidean=False
    ):
        r"""
        By default, takes absolute value of difference between starting points and that of the ending points, and averages them.
        If euclidean==True, the Euclidean distance is instead used

        Helper function for DOVSAudit.handle_audits_with_same_interval_for_PN
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(intrvl_1, [list,tuple,np.ndarray]) and len(intrvl_1)==2)
        assert(Utilities.is_object_one_of_types(intrvl_2, [list,tuple,np.ndarray]) and len(intrvl_2)==2)
        #-------------------------
        delta_0 = np.abs(intrvl_1[0]-intrvl_2[0])
        delta_1 = np.abs(intrvl_1[1]-intrvl_2[1])
        #-------------------------
        if not euclidean:
            return 0.5*(delta_0+delta_1)
        else:
            if isinstance(delta_0, pd.Timedelta) or isinstance(delta_1, pd.Timedelta):
                assert(isinstance(delta_0, pd.Timedelta) and isinstance(delta_1, pd.Timedelta))
                delta_0 = delta_0.total_seconds()
                delta_1 = delta_1.total_seconds()
            return np.linalg.norm([delta_0, delta_1])
            
            
    @staticmethod
    def remove_prefix(
        inpt, 
        prefix
    ):
        r"""
        Very simple function, checks that inpt starts with prefix, then removes the prefix

        inpt:
            Can be a string, or a list of strings.
            Returned type will match input type
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(inpt, [str, list]))
        #-------------------------
        if isinstance(inpt, list):
            return [DOVSAudit.remove_prefix(inpt=x, prefix=prefix) for x in inpt]
        #-------------------------
        assert(inpt.startswith(prefix))
        return inpt[len(prefix):]



    @staticmethod
    def split_outage_interval_across_multiple_overlapping_dovs(
        outg_intrvl, 
        outg_rec_nbs, 
        dovs_df=None, 
        outg_rec_nb_col='OUTG_REC_NB', 
        dt_off_ts_full_col='DT_OFF_TS_FULL', 
        dt_on_ts_col='DT_ON_TS', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        If one continuous found outage interval overlaps with multiple DOVS events, this splits up the outg_intrvl amongst the DOVS events.
        This function assumes the following criteria are met, and will crash if not:
            1. The DOVS events do not overlap each other
            2. The DOVS events all overlap with outg_intrvl

        Since all DOVS events overlap with the found outage interval (i.e., the zero-power well), and the DOVS events do not overlap with 
          each other, all of the DOVS beginning and ending times should fall within outg_intrvl EXCEPT FOR possibly the beginning of the 
          first and/or ending of the last.

        If there exists time between the DOVS events (during which the PN was without power), this time is essentially unaccounted for by either
          of the events.
        In such a case, this time is split, with the first half given to the first DOVS event and the second to the second.

        If dovs_df is None, or not all outg_rec_nbs are found within dovs_df, it will be built on-the-fly by running a SQL query

        Returns:
            dovs_df with columns = [outg_rec_nb_col, dt_off_ts_full_col, dt_on_ts_col, intrvl_min, intrvl_max] and one entry per outg_rec_nb, 
              where intrvl_min,intrvl_max house the portion of outg_intrvl assigned to the given outg_rec_nb
        """
        #-------------------------
        if(
            dovs_df is None or 
            set(outg_rec_nbs).difference(set(dovs_df[outg_rec_nb_col].unique().tolist())) != set()
        ):
            dovs = DOVSOutages(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                contstruct_df_args=None, 
                init_df_in_constructor=True,
                build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                build_sql_function_kwargs=dict(
                    outg_rec_nbs=outg_rec_nbs, 
                    include_premise=False
                ), 
                build_consolidated=False
            )
            dovs_df = dovs.df.copy()
        #-------------------------
        # If dovs_df supplied, don't want to make changes outside of this function, so copy
        dovs_df = dovs_df.copy()
        assert(set(outg_rec_nbs).difference(set(dovs_df[outg_rec_nb_col].unique().tolist()))==set())
        #-----
        dovs_df = dovs_df[dovs_df[outg_rec_nb_col].isin(outg_rec_nbs)]
        dovs_df = dovs_df[[outg_rec_nb_col, dt_off_ts_full_col, dt_on_ts_col]].drop_duplicates()
        #-----
        # There should only be one entry in dovs_df for each outg_rec_nb
        assert(dovs_df[outg_rec_nb_col].nunique()==dovs_df.shape[0])

        #-------------------------
        # Make sure the DOVS events themselves do not overlap!
        for i in range(dovs_df.shape[0]-1):
            assert(not pd.Interval(*dovs_df.iloc[i][[dt_off_ts_full_col, dt_on_ts_col]].values).overlaps(pd.Interval(*dovs_df.iloc[i+1][[dt_off_ts_full_col, dt_on_ts_col]].values)))

        #-------------------------
        # Make sure the DOVS events are properly ordered
        dovs_df = dovs_df.sort_values(by=[dt_off_ts_full_col], key=natsort_keygen(), ignore_index=True, ascending=True)
        # Sanity check
        for i in range(dovs_df.shape[0]-1):
            assert(dovs_df.iloc[i][dt_on_ts_col] <= dovs_df.iloc[i+1][dt_off_ts_full_col])

        #-------------------------
        # Since all DOVS events overlap with the found outage interval (i.e., the zero-power well), and the DOVS events do not
        #   overlap with each other, all of the DOVS beginning and ending times should fall within outg_intrvl 
        #   EXCEPT FOR possibly the beginning of the first and/or ending of the last
        #   These are enforced via the pd.Interval(*outg_intrvl).overlaps(...) calls below
        #-----
        # If there exists time between the DOVS events (during which the PN was without power), this time is essentially 
        #   unaccounted for by either of the events.
        # This time is split, with the first half given to the first DOVS event and the second to the second.
        # The calculation can be thought of as:
        #   1. Taking the end point of the first and adding half of the separation (and, taking the beginning point of the second and
        #        subtracting half of the separation), i.e., a + 0.5*(b-a)
        #   2. Taking the average of the end point of the first and beginning point of the second, i.e. 0.5(a+b)
        #   These are mathematically equivalent, as a+0.5(b-a) = 0.5(a+b).
        #   However, the end/beginning points are timestamps, for which one cannot simply take the average.
        #   Therefore, the calculation proceeds as (1) above.
        #-------------------------
        dovs_df['intrvl_min'] = np.nan
        dovs_df['intrvl_max'] = np.nan
        dovs_df[open_beg_col] = False
        dovs_df[open_end_col] = False
        #-----
        # Super conservative here, not 100% necessary to find intrvl_min_col_idx and intrvl_max_col_idx (could use .iloc[x, -1] and .iloc[x, -2] instead), 
        #   but safest method
        intrvl_min_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(df=dovs_df, col='intrvl_min', exact_match=True, assert_single=True)
        intrvl_max_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(df=dovs_df, col='intrvl_max', exact_match=True, assert_single=True)
        open_beg_col_idx   = Utilities_df.find_idxs_in_highest_order_of_columns(df=dovs_df, col=open_beg_col, exact_match=True, assert_single=True)
        open_end_col_idx   = Utilities_df.find_idxs_in_highest_order_of_columns(df=dovs_df, col=open_end_col, exact_match=True, assert_single=True)
        #-----
        for i_dovs in range(dovs_df.shape[0]):
            dovs_i = dovs_df.iloc[i_dovs]
            if i_dovs==0:
                # First DOVS event: ending must overlap with outg_intrvl (not necessarily the beginning)
                assert(pd.Interval(*outg_intrvl).overlaps(pd.Interval(dovs_i[dt_on_ts_col], dovs_i[dt_on_ts_col])))
                #-----
                dovs_ip1 = dovs_df.iloc[i_dovs+1]
                #-----
                t_min_i = outg_intrvl[0]
                t_max_i = dovs_i[dt_on_ts_col] + 0.5*(dovs_ip1[dt_off_ts_full_col] - dovs_i[dt_on_ts_col])
                #-----
                open_beg_i = False
                open_end_i = True
            elif i_dovs==dovs_df.shape[0]-1:
                # Last DOVS event: beginning must overlap with outg_intrvl (not necessarily the ending)
                assert(pd.Interval(*outg_intrvl).overlaps(pd.Interval(dovs_i[dt_off_ts_full_col], dovs_i[dt_off_ts_full_col])))
                #-----
                dovs_im1 = dovs_df.iloc[i_dovs-1]
                #-----
                t_min_i = dovs_i[dt_off_ts_full_col] - 0.5*(dovs_i[dt_off_ts_full_col] - dovs_im1[dt_on_ts_col])
                t_max_i = outg_intrvl[1]
                #-----
                open_beg_i = True
                open_end_i = False
            else:
                # Middle DOVS event: both beginning and ending must overlap with outg_intrvl
                assert(
                    pd.Interval(*outg_intrvl).overlaps(pd.Interval(dovs_i[dt_off_ts_full_col], dovs_i[dt_off_ts_full_col])) and 
                    pd.Interval(*outg_intrvl).overlaps(pd.Interval(dovs_i[dt_on_ts_col],       dovs_i[dt_on_ts_col]))
                )
                #-----
                dovs_im1 = dovs_df.iloc[i_dovs-1]
                dovs_ip1 = dovs_df.iloc[i_dovs+1]
                #-----
                t_min_i = dovs_i[dt_off_ts_full_col] - 0.5*(dovs_i[dt_off_ts_full_col] - dovs_im1[dt_on_ts_col])
                t_max_i = dovs_i[dt_on_ts_col] + 0.5*(dovs_ip1[dt_off_ts_full_col] - dovs_i[dt_on_ts_col])
                #-----
                open_beg_i = True
                open_end_i = True
            #-------------------------
            dovs_df.iloc[i_dovs, intrvl_min_col_idx] = t_min_i
            dovs_df.iloc[i_dovs, intrvl_max_col_idx] = t_max_i
            dovs_df.iloc[i_dovs, open_beg_col_idx]   = open_beg_i
            dovs_df.iloc[i_dovs, open_end_col_idx]   = open_end_i
        #-------------------------
        assert(dovs_df.shape[0]==len(outg_rec_nbs))
        return dovs_df


    @staticmethod
    def check_overlap_DOVS_col_agreement_helper(
        be_df_i_j,
        outg_rec_nb_i, 
        ovrlp_pfx, 
        ovrlp_pct_cols, 
        overlap_DOVS_col='overlap_DOVS', 
        overlap_disagree_cols=['ovrlp_disagree_typeA', 'ovrlp_disagree_typeB']
    ):
        r"""
        How to check?
        For an entry in be_df_i, the events contained in the overlap_DOVS_col column should have non-zero values for their corresponding entries in ovrlp_pfx columns.
        The ovrlp_pfx column for outg_rec_nb_i (i.e., audit_i.outg_rec_nb) can be zero or non-zero.
        All others must be exactly zero.
        """
        #-------------------------
        # Intended to be run on a single row of best_ests_df
        assert(isinstance(be_df_i_j, pd.Series))
        #-------------------------
        ovrlp_DOVS = be_df_i_j[overlap_DOVS_col]
        #-------------------------
        nonzero_cols = [f'{ovrlp_pfx}{x}' for x in ovrlp_DOVS]
        zero_cols = [x for x in ovrlp_pct_cols if x not in nonzero_cols+[f'{ovrlp_pfx}{outg_rec_nb_i}']]
        #-------------------------
        # Sanity checks
        assert(set(nonzero_cols).intersection(set(zero_cols))==set())
        assert(set(nonzero_cols+zero_cols).symmetric_difference(set(ovrlp_pct_cols))==set([f'{ovrlp_pfx}{outg_rec_nb_i}']))
        #-------------------------
        type_a_col = overlap_disagree_cols[0]
        type_b_col = overlap_disagree_cols[1]
        # Consistency check
        if not (be_df_i_j[nonzero_cols]>0).all() or not (be_df_i_j[zero_cols]==0).all():
            if not (be_df_i_j[nonzero_cols]>0).all():
                offenders = be_df_i_j[nonzero_cols].loc[be_df_i_j[nonzero_cols]==0].index.tolist()
                offenders = DOVSAudit.remove_prefix(inpt=offenders, prefix = ovrlp_pfx)
                be_df_i_j[type_a_col] = offenders
            if not (be_df_i_j[zero_cols]==0).all():
                offenders = be_df_i_j[zero_cols].loc[be_df_i_j[zero_cols]!=0].index.tolist()
                offenders = DOVSAudit.remove_prefix(inpt=offenders, prefix = ovrlp_pfx)
                be_df_i_j[type_b_col] = offenders
        #-------------------------
        return be_df_i_j

    @staticmethod
    def check_overlap_DOVS_col_agrees_with_ovrlp_pct_cols(
        be_df_i,
        outg_rec_nb_i, 
        ovrlp_pfx, 
        overlap_DOVS_col='overlap_DOVS', 
        overlap_disagree_cols=['ovrlp_disagree_typeA', 'ovrlp_disagree_typeB']
    ):
        r"""
        This serves as a sanity check for consistency between the DOVS events housed in overlap_DOVS_col and the overlap percentages
          housed in the columns beginning with ovrlp_pfx.
        The overlap_DOVS_col was built for each premise via the DOVSAudit.identify_overlaps functionality, so it should definitely be trustworthy.
        The ovrlp_pfx columns are built in DOVSAudit.resolve_overlapping_audits, working with audit_i.best_ests_df_w_keep_info on a row-by-by basis
          (via lambda function) and should be reliable as well.
        HOWEVER, since the DOVSAudit.identify_overlaps functionality looks at the possible overlapping DOVS events for each premise independently, if the premise
          is not listed in a DOVS event, it will not be found as a potential overlap when in reality it likely is.
        WHEREAS DOVSAudit.resolve_overlapping_audits looks for overlaps amongst all potential overlapping DOVS events seen by any of the premises.
        For example, the following scenario was observed in OUTG_REC_NB=13582178 for premise 070171942:
            - In DOVSAudit.identify_overlaps, not potential overlapping events were found.  The reason being premise 070171942 is not listed in the DOVS database with 
              OUTG_REC_NB=13582332
            - When running DOVSAudit.identify_overlaps with other premises in OUTG_REC_NB=13582178, OUTG_REC_NB=13582332 was found as potentially overlapping (e.g., premise
              070012942 is contained in the DOVS database with OUTG_REC_NB=13582332)
            - Premise 070171942 does lose power during OUTG_REC_NB=13582332, therefore it should likely be included in that DOVS event
            - Since PN 070171942 is not associated with OUTG_REC_NB=13582332, but did lose power during that time (and because OUTG_REC_NB=13582332 was picked up by other premises
              in the original OUTG_REC_NB=13582178), this causes a disagreement between the contents of overlap_DOVS_col and the columns housing the overlap percentages.
            

        How to check?
        For an entry in be_df_i, the events contained in the overlap_DOVS_col column should have non-zero values for their corresponding entries in ovrlp_pfx columns.
        The ovrlp_pfx column for outg_rec_nb_i (i.e., audit_i.outg_rec_nb) can be zero or non-zero.
        All others must be exactly zero.
        """
        #-------------------------
        regex_pattern  = r'{}.*'.format(ovrlp_pfx)
        ovrlp_pct_cols = Utilities_df.find_cols_with_regex(df=be_df_i, regex_pattern=regex_pattern, ignore_case=False)
        #-------------------------
        outg_rec_nbs = [x[len(ovrlp_pfx):] for x in ovrlp_pct_cols]
        if outg_rec_nb_i not in outg_rec_nbs:
            print(f'{outg_rec_nb_i} not in {outg_rec_nbs}!!!!!')
            assert(0)
        #-------------------------
        assert(len(overlap_disagree_cols)==2)
        assert(set(overlap_disagree_cols).intersection(set(be_df_i.columns.tolist()))==set())
        be_df_i[overlap_disagree_cols] = None
        #-------------------------
        be_df_i = be_df_i.apply(
            lambda x: DOVSAudit.check_overlap_DOVS_col_agreement_helper(
                be_df_i_j             = x,
                outg_rec_nb_i         = outg_rec_nb_i, 
                ovrlp_pfx             = ovrlp_pfx, 
                ovrlp_pct_cols        = ovrlp_pct_cols, 
                overlap_DOVS_col      = overlap_DOVS_col, 
                overlap_disagree_cols = overlap_disagree_cols
            ), 
            axis=1
        )
        #-------------------------
        return be_df_i

    def resolve_overlapping_audits(
        self, 
        dovs_df=None, 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        overlap_DOVS_col='overlap_DOVS', 
        outg_rec_nb_col_dovs='OUTG_REC_NB', 
        dt_off_ts_full_col_dovs='DT_OFF_TS_FULL', 
        dt_on_ts_col_dovs='DT_ON_TS', 
        overlaps_dovs_sql_fcn=DOVSOutages_SQL.build_sql_outage, 
        overlaps_addtnl_dovs_sql_kwargs=dict(
            CI_NB_min  = 0, 
            CMI_NB_min = 0
        ), 
        overlap_disagree_cols=['ovrlp_disagree_typeA', 'ovrlp_disagree_typeB'], 
        unq_idfr_cols = ['PN', 'i_outg'], 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        
        unq_idfr_cols:
            Not necessary, only used for outputting info when disagreement between overlap_DOVS_col and found overlap percentages
        """
        #---------------------------------------------------------------------------
        if self.best_ests_df is None or self.best_ests_df.shape[0]==0:
            return
        #-------------------------
        update_df = DOVSAudit.set_initial_to_adjust_df(
            be_df       = self.best_ests_df_w_keep_info.copy(), 
            be_df_cols  = None, 
            addtnl_cols = None
        )
        #----------------------------------------------------------------------------------------------------
        outg_rec_nbs = list(set(update_df[overlap_DOVS_col].sum()))+[self.outg_rec_nb]
        #-------------------------
        if(
            dovs_df is None or 
            set(outg_rec_nbs).difference(set(dovs_df[outg_rec_nb_col_dovs].unique().tolist())) != set()
        ):
            dflt_dovs_sql_kwargs = dict(
                outg_rec_nbs=outg_rec_nbs, 
                include_premise=False
            )
            #-----
            if overlaps_addtnl_dovs_sql_kwargs is None:
                dovs_sql_kwargs = dflt_dovs_sql_kwargs
            else:
                assert(isinstance(overlaps_addtnl_dovs_sql_kwargs, dict))
                #-----
                # Make sure none of the dovs_sql_kwargs handled by this function are included in overlaps_addtnl_dovs_sql_kwargs
                assert(set(dflt_dovs_sql_kwargs.keys()).intersection(set(overlaps_addtnl_dovs_sql_kwargs.keys()))==set())
                #-----
                dovs_sql_kwargs = Utilities.supplement_dict_with_default_values(
                    to_supplmnt_dict    = dflt_dovs_sql_kwargs, 
                    default_values_dict = overlaps_addtnl_dovs_sql_kwargs, 
                    extend_any_lists    = True, 
                    inplace             = False
                )
            
            #-----
            dovs = DOVSOutages(
                df_construct_type         = DFConstructType.kRunSqlQuery, 
                contstruct_df_args        = None, 
                init_df_in_constructor    = True,
                build_sql_function        = overlaps_dovs_sql_fcn, 
                build_sql_function_kwargs = dovs_sql_kwargs, 
                build_consolidated        = False
            )
            dovs_df = dovs.df.copy()
        #-------------------------
        # If dovs_df supplied, don't want to make changes outside of this function, so copy
        dovs_df = dovs_df.copy()
        assert(set(outg_rec_nbs).difference(set(dovs_df[outg_rec_nb_col_dovs].unique().tolist()))==set())
        #-----
        dovs_df = dovs_df[dovs_df[outg_rec_nb_col_dovs].isin(outg_rec_nbs)]
        dovs_df = dovs_df[[outg_rec_nb_col_dovs, dt_off_ts_full_col_dovs, dt_on_ts_col_dovs]].drop_duplicates()
        #-----
        # There should only be one entry in dovs_df for each outg_rec_nb
        assert(dovs_df[outg_rec_nb_col_dovs].nunique()==dovs_df.shape[0])
        #----------------------------------------------------------------------------------------------------
        # Build new columns, one for each outg_rec_nb, housing the percent overlap of the found interval for each row
        #   and the possibly overlapping DOVS events
        #-----
        ovrlp_pfx = Utilities.generate_random_string(str_len=4)+'_'
        assert(not col.startswith(ovrlp_pfx) for col in update_df.columns.tolist())
        #-----
        for idx_i, dovs_i in dovs_df.iterrows():
            outg_rec_nb_i = dovs_i[outg_rec_nb_col_dovs]
            update_df[f'{ovrlp_pfx}{outg_rec_nb_i}'] = update_df.apply(
                lambda x: DOVSAudit.get_overlap_interval_len(
                    intrvl_1 = [x[t_min_col], x[t_max_col]], 
                    intrvl_2 = [dovs_i[dt_off_ts_full_col_dovs], dovs_i[dt_on_ts_col_dovs]], 
                    norm_by=2
                ), 
                axis=1
            )
        #-------------------------
        ovrlp_pct_cols = [x for x in update_df.columns.tolist() if x.startswith(ovrlp_pfx)]
        assert(len(ovrlp_pct_cols)==len(outg_rec_nbs))

        #----------------------------------------------------------------------------------------------------
        # Check agreement between overlap_DOVS and the overlap percentage columns
        update_df = DOVSAudit.check_overlap_DOVS_col_agrees_with_ovrlp_pct_cols(
            be_df_i               = update_df,
            outg_rec_nb_i         = self.outg_rec_nb, 
            ovrlp_pfx             = ovrlp_pfx, 
            overlap_DOVS_col      = overlap_DOVS_col, 
            overlap_disagree_cols = overlap_disagree_cols
        )
        if update_df[overlap_disagree_cols].notna().any().any():
            print(f"WARNING: Disagreement between {overlap_DOVS_col} and the overlap percentage columns beginning with {ovrlp_pfx} for outg_rec_nb={outg_rec_nb_i}")
            print('\tType A: DOVS claimed there was an overlapping outage, but we cannot find one')
            print('\tType B: DOVS did not find an overlapping outage, but we did')
            for disagree_col in overlap_disagree_cols:
                if update_df[disagree_col].notna().any():
                    print(f'Type disagreement found: {disagree_col}')
                    disagree_ovrlp_DOVS = list(set(update_df[update_df[disagree_col].notna()][disagree_col].sum()))
                    print(f'\tOutside DOVS events: {disagree_ovrlp_DOVS}')
                    if set(unq_idfr_cols).difference(set(update_df.columns.tolist()))==set():
                        disagree_entries = update_df[update_df[disagree_col].notna()][unq_idfr_cols].drop_duplicates()
                        print(f"\tEntries: {disagree_entries.values.tolist()}")
                        print(f'\tunq_idfr_cols = {unq_idfr_cols}')

        #----------------------------------------------------------------------------------------------------
        # RULES:
        #   - If a sub-outage only overlaps with one DOVS event, it is decided
        #   - If a sub-outage does not overlap with any of the DOVS events, it remains with self
        #   - If a sub-outage does not overlap with self, but does with others, remove from audit_i
        #-------------------------
        # There might be some clever, pythonic, way to achieve this (e.g., can definitely identify via calls
        #   like (update_df[ovrlp_pct_cols]>0).sum(axis=1) and (update_df[ovrlp_pct_cols]==0).sum(axis=1)).
        # But, the operations are light and simple, so brute-force is easier and fine in this case
        #-------------------------
        # Need unique indices, so I can safely assign using .loc
        # If one did not want to do the .reset_index call below, one could iterate over rows using 
        #   for i_row in range(update_df.shape[0]) in the for loop below
        if update_df.index.nunique()<update_df.shape[0]:
            update_df = update_df.reset_index(drop=True)
        #-----
        for idx_i, row_i in update_df.iterrows():
            gt0_srs = row_i[ovrlp_pct_cols]>0
            eq0_srs = row_i[ovrlp_pct_cols]==0
            #-----
            # If a sub-outage only overlaps with one DOVS event, it is decided
            if gt0_srs.sum()==1:
                parent_dovs = gt0_srs[gt0_srs==True].index
                assert(len(parent_dovs)==1)
                parent_dovs = DOVSAudit.remove_prefix(
                    inpt   = parent_dovs[0], 
                    prefix = ovrlp_pfx
                )
                #-------------------------
                update_df.loc[idx_i, 'resolved']         = True
                update_df.loc[idx_i, 'resolved_details'] = 'Certain'
                if parent_dovs == self.outg_rec_nb:
                    update_df.loc[idx_i, keep_col]       = True
                    update_df.loc[idx_i, 'adjustment']   = None
                else:
                    update_df.loc[idx_i, keep_col]       = False
                    update_df.loc[idx_i, 'adjustment']   = 'remove'
            # If a sub-outage does not overlap with any of the DOVS events, it remains with self
            elif eq0_srs.all():
                update_df.loc[idx_i, 'resolved']         = True
                update_df.loc[idx_i, 'resolved_details'] = 'Certain'
                update_df.loc[idx_i, keep_col]           = True
                update_df.loc[idx_i, 'adjustment']       = None
            # Sub-outage overlaps with multiple DOVS events......
            else:
                assert(gt0_srs.sum()>1)
                #-------------------------
                ovrlp_outg_rec_nbs = DOVSAudit.remove_prefix(
                    inpt   = gt0_srs[gt0_srs==True].index.tolist(), 
                    prefix = ovrlp_pfx
                )
                #-------------------------
                # If self.outg_rec_nb does not overlap with the sub-outage, no further analysis is needed,
                #   this sub-outage simply needs to be removed from audit_i
                if self.outg_rec_nb not in ovrlp_outg_rec_nbs:
                    update_df.loc[idx_i, 'resolved']         = True
                    update_df.loc[idx_i, 'resolved_details'] = 'Certain'
                    update_df.loc[idx_i, keep_col]           = False
                    update_df.loc[idx_i, 'adjustment']       = 'remove'
                # If self.outg_rec_nb does overlap sub-outage (with others), split using
                #   DOVSAudit.split_outage_interval_across_multiple_overlapping_dovs
                else:
                    outg_intrvl = [row_i[t_min_col], row_i[t_max_col]]
                    ovrlp_outg_rec_nbs = DOVSAudit.remove_prefix(
                        inpt   = gt0_srs[gt0_srs==True].index.tolist(), 
                        prefix = ovrlp_pfx
                    )
                    split_df = DOVSAudit.split_outage_interval_across_multiple_overlapping_dovs(
                        outg_intrvl = outg_intrvl, 
                        outg_rec_nbs = ovrlp_outg_rec_nbs, 
                        dovs_df=None, 
                        outg_rec_nb_col=outg_rec_nb_col_dovs, 
                        dt_off_ts_full_col=dt_off_ts_full_col_dovs, 
                        dt_on_ts_col=dt_on_ts_col_dovs, 
                        open_beg_col=open_beg_col, 
                        open_end_col=open_end_col
                    )
                    #-------------------------
                    assert(self.outg_rec_nb in split_df[outg_rec_nb_col_dovs].unique().tolist())
                    adjstmnt_i = split_df[split_df[outg_rec_nb_col_dovs]==self.outg_rec_nb]
                    assert(adjstmnt_i.shape[0]==1)
                    #-----
                    update_df.loc[idx_i, 'resolved']         = True
                    update_df.loc[idx_i, 'resolved_details'] = 'Uncertain'
                    update_df.loc[idx_i, keep_col]           = True
                    update_df.loc[idx_i, 'adjustment']       = 'adjust'
                    #-----
                    update_df.loc[idx_i, t_min_col]    = adjstmnt_i.iloc[0]['intrvl_min']
                    update_df.loc[idx_i, t_max_col]    = adjstmnt_i.iloc[0]['intrvl_max']
                    update_df.loc[idx_i, open_beg_col] = (update_df.loc[idx_i, open_beg_col] or adjstmnt_i.iloc[0][open_beg_col])
                    update_df.loc[idx_i, open_end_col] = (update_df.loc[idx_i, open_end_col] or adjstmnt_i.iloc[0][open_end_col])
        #-------------------------
        update_df = update_df.drop(columns=ovrlp_pct_cols)
        self.best_ests_df_w_keep_info = update_df.copy()
        self.update_best_ests_and_ci_cmi(
            PN_col         = 'PN', 
            winner_min_col = t_min_col, 
            winner_max_col = t_max_col, 
            i_outg_col     = 'i_outg', 
            keep_col       = keep_col
        )
        #-------------------------
        self.generate_warnings(
            resolved_col='resolved', 
            resolved_details_col='resolved_details', 
            overlap_disagree_cols=overlap_disagree_cols, 
            overlap_disagree_cols_descs = [
                'Type A: DOVS claimed there was an overlapping outage, but we cannot find one', 
                'Type B: DOVS did not find an overlapping outage, but we did'
            ]
        )
        #-------------------------
        



    #****************************************************************************************************
    # OLD Methods to try to rectify any overlaps with other DOVS events
    # !!!!!!!!! TO BE DELETED !!!!!!!!!!!
    #****************************************************************************************************
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def assign_a_b(
        ovrlps_srs, 
        pct_overlap_a_col='pct_overlap_a', 
        pct_overlap_b_col='pct_overlap_b' 
    ):
        r"""
        Helper function for DOVSAudit.handle_audits_with_same_two_intervals_for_PN
        """
        #-------------------------
        if ovrlps_srs[pct_overlap_a_col] > ovrlps_srs[pct_overlap_b_col]:
            return 'a'
        elif ovrlps_srs[pct_overlap_a_col] < ovrlps_srs[pct_overlap_b_col]:
            return 'b'
        else:
            return np.nan


	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def did_audits_find_same_interval_for_PN(
        be_df_a, 
        be_df_b, 
        t_min_col='winner_min', 
        t_max_col='winner_max'    
    ):
        r"""
        Determine whether or not two audits found the same, single, outage interval
        """
        #-------------------------
        if(
            be_df_a.shape[0] == 1 and 
            be_df_b.shape[0] == 1 and
            be_df_a[[t_min_col, t_max_col]].reset_index(drop=True).equals(
                be_df_b[[t_min_col, t_max_col]].reset_index(drop=True)
            )
        ):
            return True
        return False


	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def did_audits_find_same_two_intervals_for_PN(
        be_df_a, 
        be_df_b, 
        t_min_col='winner_min', 
        t_max_col='winner_max'    
    ):
        r"""
        Determine whether or not two audits found the same two outage interval.
        When there is an overlap between two audits, this is likely a common.
        """
        #-------------------------
        # NOTE: symmetric_difference doesn't like a list of lists or list of numpy.ndarray objects,
        #         which is why I need to convert elements to tuples instead of lists/arrays below
        if(
            be_df_a.shape[0] == 2 and 
            be_df_b.shape[0] == 2 and
            (set([tuple(x) for x in be_df_a[[t_min_col, t_max_col]].values]).symmetric_difference(
                set([tuple(x) for x in be_df_b[[t_min_col, t_max_col]].values])
            )==set())
        ):
            return True
        return False


	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def handle_audits_with_same_interval_for_PN(
        be_df_a, 
        be_df_b, 
        dovs_outg_t_beg_end_a, 
        dovs_outg_t_beg_end_b, 
        PN_i, 
        unq_idfr_cols = ['PN', 'i_outg'], 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        """
        #-------------------------
        assert(
            DOVSAudit.did_audits_find_same_interval_for_PN(
                be_df_a   = be_df_a, 
                be_df_b   = be_df_b, 
                t_min_col = t_min_col, 
                t_max_col = t_max_col
            )
        )
        #-------------------------
        outg_intrvl = be_df_a.iloc[0][[t_min_col, t_max_col]].tolist()
        ovrlps_dovs_a = pd.Interval(*outg_intrvl).overlaps(pd.Interval(*dovs_outg_t_beg_end_a))
        ovrlps_dovs_b = pd.Interval(*outg_intrvl).overlaps(pd.Interval(*dovs_outg_t_beg_end_b))
        #-------------------------
        be_df_cols = unq_idfr_cols + list(set([t_min_col, t_max_col, open_beg_col, open_end_col]).difference(set(unq_idfr_cols)))
        assert(set(be_df_cols).difference(set(be_df_a.columns.tolist()))==set())
        assert(set(be_df_cols).difference(set(be_df_b.columns.tolist()))==set())
        #-----
        return_cols = be_df_cols + ['adjustment', 'resolved', 'resolved_details']
        #-----
        resolved_i = False
        resolved_i_details = 'Unresolved'
        #-----
        to_adjust_a = DOVSAudit.set_initial_to_adjust_df(
            be_df = be_df_a, 
            be_df_cols = be_df_cols
        )
        to_adjust_b = DOVSAudit.set_initial_to_adjust_df(
            be_df = be_df_b, 
            be_df_cols = be_df_cols
        )
        #---------------------------------------------------------------------------
        # If doesn't overlap with either DOVS events, then choose closest?
        # How should I quantify closest?
        if ovrlps_dovs_a+ovrlps_dovs_b==0:
            closeness_a = DOVSAudit.quantify_closeness_of_intervals(
                intrvl_1  = be_df_a[[t_min_col, t_max_col]].values[0], 
                intrvl_2  = dovs_outg_t_beg_end_a, 
                euclidean = True
            )
            #-----
            closeness_b = DOVSAudit.quantify_closeness_of_intervals(
                intrvl_1  = be_df_b[[t_min_col, t_max_col]].values[0], 
                intrvl_2  = dovs_outg_t_beg_end_b, 
                euclidean = True
            )
            #-------------------------
            resolved_i = True
            resolved_i_details = 'Uncertain'
            if closeness_a<=closeness_b:
                to_adjust_b['adjustment'] = 'remove'
            else:
                to_adjust_a['adjustment'] = 'remove'
        #---------------------------------------------------------------------------
        # If only overlaps with one of the DOVS events, easily resolved
        elif ovrlps_dovs_a+ovrlps_dovs_b==1:
            resolved_i = True
            resolved_i_details = 'Certain'
            if ovrlps_dovs_a:
                to_adjust_b['adjustment'] = 'remove'
            else:
                to_adjust_a['adjustment'] = 'remove'
        #---------------------------------------------------------------------------
        # Case: Outage period split across two DOVS events 
        #  e.g., outg_rec_nb = 13551139 and 13551140 for PN=040587963
        #   Criteria:
        #     1. Overlaps both DOVS events
        #     2. DOVS events don't overlap each other
        #     3. First DOVS event ends and second DOVS event begins within outage period
        #-------------------------
        else:
            assert(ovrlps_dovs_a+ovrlps_dovs_b==2)
            #-------------------------
            # Determine which DOVS event occurred first
            if dovs_outg_t_beg_end_a[0] < dovs_outg_t_beg_end_b[0]:
                frst_dovs = dovs_outg_t_beg_end_a
                scnd_dovs = dovs_outg_t_beg_end_b
            else:
                frst_dovs = dovs_outg_t_beg_end_b
                scnd_dovs = dovs_outg_t_beg_end_a
            assert(frst_dovs[1] < scnd_dovs[0])
            #-------------------------
            # At this point, we know both DOVS events overlap with the found outage period, and they don't
            #   overlap with each other.
            # Therefore, it seems trivial that the first DOVS event should end and second should begin
            #   within the found outage period, but I'm not 100%
            if(
                frst_dovs[1] > outg_intrvl[0] and frst_dovs[1] < outg_intrvl[1] and 
                scnd_dovs[0] > outg_intrvl[0] and scnd_dovs[0] < outg_intrvl[1] 
            ):
                # At this point, we know the situation: Outage period split across two DOVS events 
                #   e.g., outg_rec_nb = 13551139 and 13551140 for PN=040587963
                #-----
                # The time between the first and second outages will be split; the first half given to first and
                #   second half given to second
                unaccounted_intrvl = [
                    frst_dovs[1], 
                    scnd_dovs[0]
                ]
                #-----
                # Timestamps, so can't simply take average of unaccounted_intrvl[0] and unaccounted_intrvl[1]
                #   But, calculation below gives same result as average of two
                midpoint = unaccounted_intrvl[0] + 0.5*(unaccounted_intrvl[1]-unaccounted_intrvl[0])
                #-----
                frst_intrvl = [outg_intrvl[0], midpoint]
                scnd_intrvl = [midpoint, outg_intrvl[1]]
                #-------------------------
                to_adjust_a = be_df_a[be_df_cols].reset_index(drop=True)
                to_adjust_b = be_df_b[be_df_cols].reset_index(drop=True)
                #-----
                if frst_dovs==dovs_outg_t_beg_end_a:
                    assert(scnd_dovs==dovs_outg_t_beg_end_b)
                    to_adjust_a[[t_min_col, t_max_col]] = frst_intrvl
                    to_adjust_a[open_end_col] = True
                    #-----
                    to_adjust_b[[t_min_col, t_max_col]] = scnd_intrvl
                    to_adjust_b[open_beg_col] = True
                elif frst_dovs==dovs_outg_t_beg_end_b:
                    assert(scnd_dovs==dovs_outg_t_beg_end_a)
                    to_adjust_a[[t_min_col, t_max_col]] = scnd_intrvl
                    to_adjust_a[open_beg_col] = True
                    #-----
                    to_adjust_b[[t_min_col, t_max_col]] = frst_intrvl
                    to_adjust_b[open_end_col] = True
                else:
                    assert(0)
                #-----
                resolved_i = True
                resolved_i_details = 'Uncertain'
                #-----
                to_adjust_a['adjustment'] = 'adjust'
                to_adjust_b['adjustment'] = 'adjust'
            else:
                resolved_i = False
                resolved_i_details = 'Unresolved'
        #---------------------------------------------------------------------------
        to_adjust_a['resolved']         = resolved_i
        to_adjust_a['resolved_details'] = resolved_i_details
        #-----
        to_adjust_b['resolved']         = resolved_i
        to_adjust_b['resolved_details'] = resolved_i_details
        #-------------------------
        return dict(
            resolved_i         = resolved_i, 
            resolved_i_details = resolved_i_details, 
            to_adjust_a        = to_adjust_a, 
            to_adjust_b        = to_adjust_b
        )


	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def handle_audits_with_same_two_intervals_for_PN(
        be_df_a, 
        be_df_b, 
        dovs_outg_t_beg_end_a, 
        dovs_outg_t_beg_end_b, 
        PN_i, 
        unq_idfr_cols = ['PN', 'i_outg'], 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        Simplest (and likely the most common) situation: 
          audit_a and audit_b both find the same two sub-outages for PN_i, one of which clearly belongs
            to audit_a and the other of which clearly belongs to audit_b.
          This situation results in resolved_i_details = 'Certain' below
        """
        #-------------------------
        assert(
            DOVSAudit.did_audits_find_same_two_intervals_for_PN(
                be_df_a = be_df_a, 
                be_df_b = be_df_b, 
                t_min_col = t_min_col, 
                t_max_col = t_max_col   
            )
        )
        #-------------------------
        be_df_cols = unq_idfr_cols + list(set([t_min_col, t_max_col, open_beg_col, open_end_col]).difference(set(unq_idfr_cols)))
        assert(set(be_df_cols).difference(set(be_df_a.columns.tolist()))==set())
        assert(set(be_df_cols).difference(set(be_df_b.columns.tolist()))==set())
        #-----
        return_cols = be_df_cols + ['adjustment', 'resolved', 'resolved_details']
        #-----
        resolved_i = False
        resolved_i_details = 'Unresolved'
        #-----
        to_adjust_a = DOVSAudit.set_initial_to_adjust_df(
            be_df       = be_df_a, 
            be_df_cols  = be_df_cols, 
            addtnl_cols = [keep_col]
        )
        to_adjust_b = DOVSAudit.set_initial_to_adjust_df(
            be_df       = be_df_b, 
            be_df_cols  = be_df_cols, 
            addtnl_cols = [keep_col]
        )
        #-------------------------
        # The data in tmp_ovrlp_dovs_col will make life a lot easier if the program below goes past the first if statement
        tmp_ovrlp_dovs_col = Utilities.generate_random_string()
        assert(tmp_ovrlp_dovs_col not in to_adjust_a.columns.tolist() and 
               tmp_ovrlp_dovs_col not in to_adjust_b.columns.tolist())
        to_adjust_a[tmp_ovrlp_dovs_col] = to_adjust_a.apply(
            lambda x: pd.Interval(x[t_min_col], x[t_max_col]).overlaps(
                pd.Interval(dovs_outg_t_beg_end_a[0], dovs_outg_t_beg_end_a[1])
            ), 
            axis=1
        )
        #-----
        to_adjust_b[tmp_ovrlp_dovs_col] = to_adjust_b.apply(
            lambda x: pd.Interval(x[t_min_col], x[t_max_col]).overlaps(
                pd.Interval(dovs_outg_t_beg_end_b[0], dovs_outg_t_beg_end_b[1])
            ), 
            axis=1
        )    
        #-------------------------
        # If each to_adjust_a and to_adjust_b has one entry with keep==True and one with keep==False, 
        #   and these differ from each other, then ownership can be resolved simply (the entry with keep==True
        #   will be kept for each)
        if(
            to_adjust_a[keep_col].sum() == 1 and
            to_adjust_b[keep_col].sum() == 1 and 
            not to_adjust_a[to_adjust_a[keep_col]==True][[t_min_col, t_max_col]].reset_index(drop=True).equals(
                to_adjust_b[to_adjust_b[keep_col]==True][[t_min_col, t_max_col]].reset_index(drop=True)
            )
        ):
            to_adjust_a.loc[to_adjust_a[keep_col]==False, 'adjustment'] = 'remove'
            to_adjust_b.loc[to_adjust_b[keep_col]==False, 'adjustment'] = 'remove'
            #-----
            assert((to_adjust_a['adjustment']=='remove').sum()==1)
            assert((to_adjust_b['adjustment']=='remove').sum()==1)
            #-----
            resolved_i = True
            resolved_i_details = 'Certain'
        #-------------------------
        # If only one entry in to_adjust_a overlaps with dovs_outg_t_beg_end_a and 
        #    only one entry in to_adjust_b overlaps with dovs_outg_t_beg_end_b
        # AND these differ from each other, then ownership can be resolved simply
        elif(
            to_adjust_a[tmp_ovrlp_dovs_col].sum() == 1 and
            to_adjust_b[tmp_ovrlp_dovs_col].sum() == 1 and 
            not to_adjust_a[to_adjust_a[tmp_ovrlp_dovs_col]==True][[t_min_col, t_max_col]].reset_index(drop=True).equals(
                to_adjust_b[to_adjust_b[tmp_ovrlp_dovs_col]==True][[t_min_col, t_max_col]].reset_index(drop=True)
            )
        ):
            to_adjust_a.loc[to_adjust_a[tmp_ovrlp_dovs_col]==False, 'adjustment'] = 'remove'
            to_adjust_b.loc[to_adjust_b[tmp_ovrlp_dovs_col]==False, 'adjustment'] = 'remove'
            #-----
            assert((to_adjust_a['adjustment']=='remove').sum()==1)
            assert((to_adjust_b['adjustment']=='remove').sum()==1)
            #-----
            resolved_i = True
            resolved_i_details = 'Certain'
        #-------------------------
        else:
            # Find the amount of overlap each sub-outage has with its parent DOVS event, and decide from there whether
            #   a most probably parent for each can be determined
            #-------------------------
            ovrlps_df = to_adjust_a[[t_min_col, t_max_col]].copy()
            #-----
            ovrlps_df['pct_overlap_a'] = ovrlps_df.apply(
                lambda x: DOVSAudit.get_overlap_interval_len(
                    intrvl_1 = [x[t_min_col], x[t_max_col]], 
                    intrvl_2 = dovs_outg_t_beg_end_a, 
                    norm_by=2
                ), 
                axis=1
            )
            #-----
            ovrlps_df['pct_overlap_b'] = ovrlps_df.apply(
                lambda x: DOVSAudit.get_overlap_interval_len(
                    intrvl_1 = [x[t_min_col], x[t_max_col]], 
                    intrvl_2 = dovs_outg_t_beg_end_b, 
                    norm_by=2
                ), 
                axis=1
            )
            #-----
            ovrlps_df['belongs_to_outg'] = ovrlps_df.apply(
                lambda x: DOVSAudit.assign_a_b(
                    ovrlps_srs        = x, 
                    pct_overlap_a_col = 'pct_overlap_a', 
                    pct_overlap_b_col = 'pct_overlap_b' 
                ), 
                axis=1
            )
            #-------------------------
            if ovrlps_df['belongs_to_outg'].isna().sum()>0:
                resolved_i = False
                resolved_i_details = 'Failed'
            else:
                to_a = ovrlps_df[ovrlps_df['belongs_to_outg']=='a']
                to_b = ovrlps_df[ovrlps_df['belongs_to_outg']=='b']
                #-------------------------
                # Make index columns, from 0 to df.shape[0]-1, so rows can be easily identified
                tmp_idx_col = Utilities.generate_random_string()
                assert(tmp_idx_col not in to_adjust_a.columns.tolist())
                assert(tmp_idx_col not in to_adjust_b.columns.tolist())
                to_adjust_a[tmp_idx_col] = range(to_adjust_a.shape[0])
                to_adjust_b[tmp_idx_col] = range(to_adjust_b.shape[0])
                #-------------------------
                # Determine which entries in to_adjust_a will stay with audit_a by merging to_adjust_a with to_a
                # Similarly for audit_b
                to_a = pd.merge(
                    to_adjust_a, 
                    to_a, 
                    left_on  = [t_min_col, t_max_col], 
                    right_on = [t_min_col, t_max_col], 
                    how = 'inner'
                )
                #-----
                to_b = pd.merge(
                    to_adjust_b.reset_index(drop=True), 
                    to_b, 
                    left_on  = [t_min_col, t_max_col], 
                    right_on = [t_min_col, t_max_col], 
                    how = 'inner'
                )
                #-------------------------
                # Determine which indices need to be removed from audit_a using to_a[tmp_idx_col]
                # Similarly for audit_b
                idxs_to_adjust_a = [i for i in range(to_adjust_a.shape[0]) if i not in to_a[tmp_idx_col].tolist()]
                idxs_to_adjust_b = [i for i in range(to_adjust_b.shape[0]) if i not in to_b[tmp_idx_col].tolist()]
                #-------------------------
                # Form to_adjust_a and to_adjust_b
                to_adjust_a.iloc[idxs_to_adjust_a, Utilities_df.find_single_col_in_multiindex_cols(df=to_adjust_a, col='adjustment')] = 'remove'
                to_adjust_b.iloc[idxs_to_adjust_b, Utilities_df.find_single_col_in_multiindex_cols(df=to_adjust_b, col='adjustment')] = 'remove'
                #-----
                assert((to_adjust_a['adjustment']=='remove').sum()==1)
                assert((to_adjust_b['adjustment']=='remove').sum()==1)
                #-----
                resolved_i = True
                resolved_i_details = 'Uncertain'
                #-------------------------
                # Cleanup
                to_adjust_a = to_adjust_a.drop(columns=[tmp_idx_col])
                to_adjust_b = to_adjust_b.drop(columns=[tmp_idx_col])
        #-------------------------
        # Need to remove tmp_ovrlp_dovs_col and keep_col from t_adjust_a and to_adjust_b
        to_adjust_a = to_adjust_a.drop(columns=[tmp_ovrlp_dovs_col, keep_col])
        to_adjust_b = to_adjust_b.drop(columns=[tmp_ovrlp_dovs_col, keep_col])
        #-------------------------
        to_adjust_a['resolved']         = resolved_i
        to_adjust_a['resolved_details'] = resolved_i_details
        #-----
        to_adjust_b['resolved']         = resolved_i
        to_adjust_b['resolved_details'] = resolved_i_details
        #-------------------------
        return dict(
            resolved_i         = resolved_i, 
            resolved_i_details = resolved_i_details, 
            to_adjust_a        = to_adjust_a, 
            to_adjust_b        = to_adjust_b
        )


	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def implement_adjustment_for_PN(
        audit, 
        to_adjust_df,
        unq_idfr_cols = ['PN', 'i_outg'], 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        Method to implement the needed adjustments in audit found by DOVSAudit.resolve_overlapping_audits_for_PN
        """
        #-------------------------
        if 'resolved' not in audit.best_ests_df_w_keep_info.columns.tolist():
            audit.best_ests_df_w_keep_info['resolved'] = np.nan 
        if 'resolved_details' not in audit.best_ests_df_w_keep_info.columns.tolist():
            audit.best_ests_df_w_keep_info['resolved_details'] = ''
        #-------------------------
        # Each PN can have multiple entries which need adjusted, so need to iterate
        for adjust_j in range(to_adjust_df.shape[0]):
            slicer_j = DFSlicer()
            to_adjust_j = to_adjust_df.iloc[adjust_j]
            for col in unq_idfr_cols:
                slicer_j.add_single_slicer(dict(
                    column              = col, 
                    value               = to_adjust_j[col], 
                    comparison_operator = '=='
                ))
            #-------------------------
            slcr_bool_j = slicer_j.get_slicing_booleans(audit.best_ests_df_w_keep_info)
            # Handling one entry at a time, so slcr_bool_j should sum to 1
            assert(slcr_bool_j.sum()==1)
            #-------------------------
            audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved'] = to_adjust_j['resolved']
            audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved_details'] = to_adjust_j['resolved_details']
            #TODO!!!!!!!!!!!!!!!!!!!!
            # # I don't want this adjustment to over-write the results from any previous (e.g., if event overlaps with multiple others)
            # if pd.isna(audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved']):
                # audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved'] = to_adjust_j['resolved']
            # else:
                # audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved'] = audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved'] and to_adjust_j['resolved']
            # #-----
            # if audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved_details']=='':
                # audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved_details'] = to_adjust_j['resolved_details']
            # else:
                # audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved_details'] = audit.best_ests_df_w_keep_info.loc[slcr_bool_j, 'resolved_details'] + '; ' + to_adjust_j['resolved_details']
            #-------------------------
            if to_adjust_j['adjustment']=='adjust':
                audit.best_ests_df_w_keep_info.loc[slcr_bool_j, keep_col]     = True
                audit.best_ests_df_w_keep_info.loc[slcr_bool_j, t_min_col]    = to_adjust_j[t_min_col]
                audit.best_ests_df_w_keep_info.loc[slcr_bool_j, t_max_col]    = to_adjust_j[t_max_col]
                audit.best_ests_df_w_keep_info.loc[slcr_bool_j, open_beg_col] = to_adjust_j[open_beg_col]
                audit.best_ests_df_w_keep_info.loc[slcr_bool_j, open_end_col] = to_adjust_j[open_end_col]
            elif to_adjust_j['adjustment']=='remove':
                # Not really needed, since PNs_to_resolve all have keep==False
                # However, explicitly set to False in case I open up procedure to all PNs, not just those
                #   flagged as not keeping
                audit.best_ests_df_w_keep_info.loc[slcr_bool_j, keep_col] = False
            elif to_adjust_j['adjustment'] is None or to_adjust_j['adjustment']=='':
                continue
            else:
                print('Unexpected adjustment value!')
                assert(0)    


	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def implement_adjustments_for_audit(
        audit_a,
        audit_b, 
        adjustment_dicts, 
        unq_idfr_cols = ['PN', 'i_outg'], 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        Method to implement a collection of adjustments in audit.

        audit_a, audit_b:
            Whichever is supplied will be adjusted
                i.e., if both supplied, both adjusted
                      if only one supplied, only one adjusted

        adjustment_dicts:
            A dictionary with:
                keys:   typically equal to PNs, but the value doesn't really matter
                values: dict objects with keys = ['resolved_i', 'resolved_i_details', 'to_adjust_a', 'to_adjust_b']
        """
        #-------------------------
        # audit_a or audit_b (or both) must be supplied
        assert(audit_a is not None or audit_b is not None)
        #-------------------------
        # In adjustment_dicts, the keys are typically PNs (but doesn't really matter) and 
        #   values are dict objects with keys = ['resolved_i', 'resolved_i_details', 'to_adjust_a', 'to_adjust_b']
        for k,adjstmnt_dict in adjustment_dicts.items():
            if audit_a is not None:
                if 'resolved' not in audit_a.best_ests_df_w_keep_info.columns.tolist():
                    audit_a.best_ests_df_w_keep_info['resolved'] = np.nan 
                if 'resolved_details' not in audit_a.best_ests_df_w_keep_info.columns.tolist():
                    audit_a.best_ests_df_w_keep_info['resolved_details'] = ''
                #-----
                DOVSAudit.implement_adjustment_for_PN(
                    audit         = audit_a, 
                    to_adjust_df  = adjstmnt_dict['to_adjust_a'],
                    unq_idfr_cols = unq_idfr_cols, 
                    t_min_col     = t_min_col, 
                    t_max_col     = t_max_col, 
                    keep_col      = keep_col, 
                    open_beg_col  = open_beg_col, 
                    open_end_col  = open_end_col
                )
            #-----
            if audit_b is not None:
                if 'resolved' not in audit_b.best_ests_df_w_keep_info.columns.tolist():
                    audit_b.best_ests_df_w_keep_info['resolved'] = np.nan 
                if 'resolved_details' not in audit_b.best_ests_df_w_keep_info.columns.tolist():
                    audit_b.best_ests_df_w_keep_info['resolved_details'] = ''
                #-----
                DOVSAudit.implement_adjustment_for_PN(
                    audit         = audit_b, 
                    to_adjust_df  = adjstmnt_dict['to_adjust_b'],
                    unq_idfr_cols = unq_idfr_cols, 
                    t_min_col     = t_min_col, 
                    t_max_col     = t_max_col, 
                    keep_col      = keep_col, 
                    open_beg_col  = open_beg_col, 
                    open_end_col  = open_end_col
                )


	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!  
    @staticmethod
    def resolve_overlapping_audits_for_PN(
        audit_a, 
        audit_b, 
        PN_i, 
        implement_resolutions=True, 
        unq_idfr_cols = ['PN', 'i_outg'], 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        overlap_DOVS_col='overlap_DOVS', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        implement_resolutions:
            If True AND resolutions are needed, then audit_a and audit_b will be adjusted accordingly
        """
        #-------------------------
        # Make sure best_ests_df_w_keep_info exists for each
        #  Below will fail if object is None or empty DF
        assert(audit_a.best_ests_df_w_keep_info.shape[0]>0)
        assert(audit_b.best_ests_df_w_keep_info.shape[0]>0)
        #-------------------------
        be_df_cols = unq_idfr_cols + list(set([t_min_col, t_max_col, open_beg_col, open_end_col]).difference(set(unq_idfr_cols)))
        assert(set(be_df_cols).difference(set(audit_a.best_ests_df_w_keep_info.columns.tolist()))==set())
        assert(set(be_df_cols).difference(set(audit_b.best_ests_df_w_keep_info.columns.tolist()))==set())
        #-----
        return_cols = be_df_cols + ['adjustment', 'resolved', 'resolved_details']
        #----------------------------------------------------------------------------------------------------
        outg_rec_nb_a         = audit_a.outg_rec_nb
        dovs_outg_t_beg_end_a = audit_a.dovs_outg_t_beg_end
        #-----
        outg_rec_nb_b         = audit_b.outg_rec_nb
        dovs_outg_t_beg_end_b = audit_b.dovs_outg_t_beg_end
        #--------------------------------------------------        
        # Grab the best_ests_df_w_keep_info object from each audit
        be_df_a = audit_a.best_ests_df_w_keep_info[audit_a.best_ests_df_w_keep_info[PN_col]==PN_i].copy()
        be_df_b = audit_b.best_ests_df_w_keep_info[audit_b.best_ests_df_w_keep_info[PN_col]==PN_i].copy()
        #--------------------------------------------------
        resolved_i = False
        resolved_i_details = ''
        #-----
        to_adjust_a = DOVSAudit.set_initial_to_adjust_df(
            be_df       = be_df_a, 
            be_df_cols  = be_df_cols, 
            addtnl_cols = [keep_col]
        )
        to_adjust_b = DOVSAudit.set_initial_to_adjust_df(
            be_df       = be_df_b, 
            be_df_cols  = be_df_cols, 
            addtnl_cols = [keep_col]
        )
        
        #----------------------------------------------------------------------------------------------------
        # If one of the audits does not contain PN_i, or the audits don't overlap for the given PN, so no 
        #   intervention needed
        if(
            be_df_a.shape[0] == 0 or 
            be_df_b.shape[0] == 0 or 
            outg_rec_nb_b not in list(set(audit_a.best_ests_df_w_keep_info[overlap_DOVS_col].sum())) or
            outg_rec_nb_a not in list(set(audit_b.best_ests_df_w_keep_info[overlap_DOVS_col].sum()))

        ):
            resolved_i = True
            resolved_i_details = 'No intervention needed'
            #-----
            to_adjust_a['resolved']         = resolved_i
            to_adjust_a['resolved_details'] = resolved_i_details
            #-----
            to_adjust_b['resolved']         = resolved_i
            to_adjust_b['resolved_details'] = resolved_i_details
            #-----
            res_dict_i = dict(
                resolved_i         = resolved_i, 
                resolved_i_details = resolved_i_details, 
                to_adjust_a        = to_adjust_a, 
                to_adjust_b        = to_adjust_b
            )
        #----------------------------------------------------------------------------------------------------
        # If the two DOVS events overlap, there's no way to resolve
        elif pd.Interval(*dovs_outg_t_beg_end_a).overlaps(pd.Interval(*dovs_outg_t_beg_end_b)):
            resolved_i = False
            resolved_i_details = 'DOVS events overlap, impossible to resolve!'
            print(f'{resolved_i_details}\nCANNOT RESOLVE!')
            #-----
            to_adjust_a['resolved']         = resolved_i
            to_adjust_a['resolved_details'] = resolved_i_details
            #-----
            to_adjust_b['resolved']         = resolved_i
            to_adjust_b['resolved_details'] = resolved_i_details
            #-----
            res_dict_i = dict(
                resolved_i         = resolved_i, 
                resolved_i_details = resolved_i_details, 
                to_adjust_a        = to_adjust_a, 
                to_adjust_b        = to_adjust_b
            )
        #----------------------------------------------------------------------------------------------------
        # Case: Both find the same, single, outage
        elif DOVSAudit.did_audits_find_same_interval_for_PN(
            be_df_a   = be_df_a, 
            be_df_b   = be_df_b, 
            t_min_col = t_min_col, 
            t_max_col = t_max_col
        ):
            res_dict_i = DOVSAudit.handle_audits_with_same_interval_for_PN(
                be_df_a               = be_df_a, 
                be_df_b               = be_df_b, 
                dovs_outg_t_beg_end_a = dovs_outg_t_beg_end_a, 
                dovs_outg_t_beg_end_b = dovs_outg_t_beg_end_b, 
                PN_i                  = PN_i, 
                unq_idfr_cols         = unq_idfr_cols, 
                PN_col                = PN_col, 
                t_min_col             = t_min_col, 
                t_max_col             = t_max_col, 
                open_beg_col          = open_beg_col, 
                open_end_col          = open_end_col
            )
        #----------------------------------------------------------------------------------------------------
        # Simplest (and likely the most common) situation: 
        #   audit_a and audit_b both find the same two sub-outages for PN_i, one of which clearly belongs
        #   to audit_a and the other of which clearly belongs to audit_b
        elif(
            DOVSAudit.did_audits_find_same_two_intervals_for_PN(
                be_df_a   = be_df_a, 
                be_df_b   = be_df_b, 
                t_min_col = t_min_col, 
                t_max_col = t_max_col   
            )
        ):
            res_dict_i = DOVSAudit.handle_audits_with_same_two_intervals_for_PN(
                be_df_a               = be_df_a, 
                be_df_b               = be_df_b, 
                dovs_outg_t_beg_end_a = dovs_outg_t_beg_end_a, 
                dovs_outg_t_beg_end_b = dovs_outg_t_beg_end_b, 
                PN_i                  = PN_i, 
                unq_idfr_cols         = unq_idfr_cols, 
                PN_col                = PN_col, 
                t_min_col             = t_min_col, 
                t_max_col             = t_max_col, 
                keep_col              = keep_col, 
                open_beg_col          = open_beg_col, 
                open_end_col          = open_end_col
            )
        #----------------------------------------------------------------------------------------------------
        else:
            resolved_i = False
            resolved_i_details = 'Unresolved by current methods'
            #-----
            to_adjust_a['resolved']         = resolved_i
            to_adjust_a['resolved_details'] = resolved_i_details
            #-----
            to_adjust_b['resolved']         = resolved_i
            to_adjust_b['resolved_details'] = resolved_i_details
            #-----
            res_dict_i = dict(
                resolved_i         = resolved_i, 
                resolved_i_details = resolved_i_details, 
                to_adjust_a        = to_adjust_a, 
                to_adjust_b        = to_adjust_b
            )
        #----------------------------------------------------------------------------------------------------
        if implement_resolutions:
            DOVSAudit.implement_adjustment_for_PN(
                audit         = audit_a, 
                to_adjust_df  = res_dict_i['to_adjust_a'],
                unq_idfr_cols = unq_idfr_cols, 
                t_min_col     = t_min_col, 
                t_max_col     = t_max_col, 
                keep_col      = keep_col, 
                open_beg_col  = open_beg_col, 
                open_end_col  = open_end_col
            )
            #-----
            DOVSAudit.implement_adjustment_for_PN(
                audit         = audit_b, 
                to_adjust_df  = res_dict_i['to_adjust_b'],
                unq_idfr_cols = unq_idfr_cols, 
                t_min_col     = t_min_col, 
                t_max_col     = t_max_col, 
                keep_col      = keep_col, 
                open_beg_col  = open_beg_col, 
                open_end_col  = open_end_col
            )
        #----------------------------------------------------------------------------------------------------
        return res_dict_i
        

	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!
	# Likely remove if resolve_overlapping_audits_OLD is removed!
	# *************************!!!!!!!!!!!!!!!!!!!!!!!!!        
    @staticmethod
    def handle_dovs_events_overlapping(
        audit_a, 
        audit_b, 
        PNs_to_resolve=None, 
        unq_idfr_cols = ['PN', 'i_outg'], 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        overlap_DOVS_col='overlap_DOVS', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        """
        #--------------------------------------------------
        if not pd.Interval(*audit_a.dovs_outg_t_beg_end).overlaps(pd.Interval(*audit_b.dovs_outg_t_beg_end)):
            return
        #--------------------------------------------------
        assert(PN_col in unq_idfr_cols)
        #-------------------------
        assert(audit_a.best_ests_df_w_keep_info.groupby(unq_idfr_cols).ngroups==audit_a.best_ests_df_w_keep_info.shape[0])
        assert(audit_b.best_ests_df_w_keep_info.groupby(unq_idfr_cols).ngroups==audit_b.best_ests_df_w_keep_info.shape[0])
        #-------------------------
        be_df_cols = unq_idfr_cols + list(set([t_min_col, t_max_col, open_beg_col, open_end_col]).difference(set(unq_idfr_cols)))
        assert(set(be_df_cols).difference(set(audit_a.best_ests_df_w_keep_info.columns.tolist()))==set())
        assert(set(be_df_cols).difference(set(audit_b.best_ests_df_w_keep_info.columns.tolist()))==set()) 
        #-----
        return_cols = be_df_cols + ['adjustment', 'resolved', 'resolved_details'] 
        #--------------------------------------------------
        ovrlp_PNs = list(set(audit_a.best_ests_df_w_keep_info[PN_col].unique().tolist()).intersection(set(audit_b.best_ests_df_w_keep_info[PN_col].unique().tolist())))
        resolved_i = False
        resolved_i_details = 'DOVS events overlap, impossible to resolve!'  
        #-------------------------
        to_adjust_full_a = audit_a.best_ests_df_w_keep_info[audit_a.best_ests_df_w_keep_info[PN_col].isin(ovrlp_PNs)][be_df_cols].copy()
        to_adjust_full_a['adjustment']       = ''
        to_adjust_full_a['resolved']         = resolved_i
        to_adjust_full_a['resolved_details'] = resolved_i_details
        #-----
        to_adjust_full_b = audit_b.best_ests_df_w_keep_info[audit_b.best_ests_df_w_keep_info[PN_col].isin(ovrlp_PNs)][be_df_cols].copy()
        to_adjust_full_b['adjustment']       = ''
        to_adjust_full_b['resolved']         = resolved_i
        to_adjust_full_b['resolved_details'] = resolved_i_details
        #-------------------------
        # The other methods expect a dict (of dict objects), not the full adjustment DFs
        #   However, I think one could simply feed, e.g., to_adjust_full_a and audit_a into DOVSAudit.implement_adjustment_for_PN
        #     to achieve the desired end result
        # In any case, to be consistent with methods elsewhere, convert to adjustment_dicts
        adjustment_dicts = dict()
        for PN_i in ovrlp_PNs:
            assert(PN_i not in adjustment_dicts.keys())
            res_dict_i = dict(
                resolved_i         = resolved_i, 
                resolved_i_details = resolved_i_details, 
                to_adjust_a        = to_adjust_full_a[to_adjust_full_a[PN_col]==PN_i].reset_index(drop=True), 
                to_adjust_b        = to_adjust_full_b[to_adjust_full_b[PN_col]==PN_i].reset_index(drop=True)
            )
        #-------------------------
        DOVSAudit.implement_adjustments_for_audit(
            audit_a          = audit_a,
            audit_b          = audit_b, 
            adjustment_dicts = adjustment_dicts, 
            unq_idfr_cols    = unq_idfr_cols, 
            t_min_col        = t_min_col, 
            t_max_col        = t_max_col, 
            keep_col         = keep_col, 
            open_beg_col     = open_beg_col, 
            open_end_col     = open_end_col
        )
        #-------------------------
        return adjustment_dicts


    # *************************!!!!!!!!!!!!!!!!!!!!!!!!!
    # If removing this OLD version, likely remove also:
    #     assign_a_b
    #     did_audits_find_same_interval_for_PN
    #     did_audits_find_same_two_intervals_for_PN
    #     handle_audits_with_same_interval_for_PN
    #     handle_audits_with_same_two_intervals_for_PN
    #     implement_adjustment_for_PN
    #     implement_adjustments_for_audit
    #     resolve_overlapping_audits_for_PN
    #     handle_dovs_events_overlapping
    #     resolve_overlapping_audits_OLD_v2
    # *************************!!!!!!!!!!!!!!!!!!!!!!!!!
    @staticmethod
    def resolve_overlapping_audits_OLD(
        audit_a, 
        audit_b, 
        PNs_to_resolve=None, 
        unq_idfr_cols = ['PN', 'i_outg'], 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        overlap_DOVS_col='overlap_DOVS', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        Adjustments made on-the-fly for each PN.
        Should give the same results as DOVSAudit.resolve_overlapping_audits_v2.
        The question is which version is more efficient?

        PNs_to_resolve:
            If None, taken to be any PNs in audit_a.best_ests_df_w_keep_info with keep==False 
        """
        #--------------------------------------------------
        if pd.Interval(*audit_a.dovs_outg_t_beg_end).overlaps(pd.Interval(*audit_b.dovs_outg_t_beg_end)):
            adjustment_dicts = DOVSAudit.handle_dovs_events_overlapping(
                audit_a          = audit_a, 
                audit_b          = audit_b, 
                PNs_to_resolve   = PNs_to_resolve, 
                unq_idfr_cols    = unq_idfr_cols, 
                PN_col           = PN_col, 
                t_min_col        = t_min_col, 
                t_max_col        = t_max_col, 
                keep_col         = keep_col, 
                overlap_DOVS_col = overlap_DOVS_col, 
                open_beg_col     = open_beg_col, 
                open_end_col     = open_end_col
            )
            return adjustment_dicts
        #--------------------------------------------------
        if PNs_to_resolve is None:
            PNs_to_resolve = audit_a.best_ests_df_w_keep_info[audit_a.best_ests_df_w_keep_info[keep_col]==False][PN_col].unique().tolist()
        #-------------------------
        adjustment_dicts = dict()
        for PN_i in PNs_to_resolve:
            assert(PN_i not in adjustment_dicts.keys())
            res_dict_i = DOVSAudit.resolve_overlapping_audits_for_PN(
                audit_a               = audit_a, 
                audit_b               = audit_b, 
                PN_i                  = PN_i, 
                implement_resolutions = True, 
                unq_idfr_cols         = unq_idfr_cols, 
                PN_col                = PN_col, 
                t_min_col             = t_min_col, 
                t_max_col             = t_max_col, 
                keep_col              = keep_col, 
                open_beg_col          = open_beg_col, 
                open_end_col          = open_end_col
            )
            adjustment_dicts[PN_i] = res_dict_i
        #-------------------------
        return adjustment_dicts


    # *************************!!!!!!!!!!!!!!!!!!!!!!!!!
    # If removing this OLD version, likely remove also:
    #     assign_a_b
    #     did_audits_find_same_interval_for_PN
    #     did_audits_find_same_two_intervals_for_PN
    #     handle_audits_with_same_interval_for_PN
    #     handle_audits_with_same_two_intervals_for_PN
    #     implement_adjustment_for_PN
    #     implement_adjustments_for_audit
    #     resolve_overlapping_audits_for_PN
    #     handle_dovs_events_overlapping
    #     resolve_overlapping_audits
    # *************************!!!!!!!!!!!!!!!!!!!!!!!!!
    @staticmethod
    def resolve_overlapping_audits_OLD_v2(
        audit_a, 
        audit_b, 
        PNs_to_resolve=None, 
        unq_idfr_cols = ['PN', 'i_outg'], 
        PN_col='PN', 
        t_min_col='winner_min', 
        t_max_col='winner_max', 
        keep_col='keep', 
        overlap_DOVS_col='overlap_DOVS', 
        open_beg_col='open_beg', 
        open_end_col='open_end'
    ):
        r"""
        Necessary adjustments collected for each PN, then implemented for all at end.
        Should give the same results as DOVSAudit.resolve_overlapping_audits.
        The question is which version is more efficient?

        PNs_to_resolve:
            If None, taken to be any PNs in audit_a.best_ests_df_w_keep_info with keep==False 
        """
        #--------------------------------------------------
        if pd.Interval(*audit_a.dovs_outg_t_beg_end).overlaps(pd.Interval(*audit_b.dovs_outg_t_beg_end)):
            adjustment_dicts = DOVSAudit.handle_dovs_events_overlapping(
                audit_a          = audit_a, 
                audit_b          = audit_b, 
                PNs_to_resolve   = PNs_to_resolve, 
                unq_idfr_cols    = unq_idfr_cols, 
                PN_col           = PN_col, 
                t_min_col        = t_min_col, 
                t_max_col        = t_max_col, 
                keep_col         = keep_col, 
                overlap_DOVS_col = overlap_DOVS_col, 
                open_beg_col     = open_beg_col, 
                open_end_col     = open_end_col
            )
            return adjustment_dicts
        #--------------------------------------------------
        if PNs_to_resolve is None:
            PNs_to_resolve = audit_a.best_ests_df_w_keep_info[audit_a.best_ests_df_w_keep_info[keep_col]==False][PN_col].unique().tolist()
        #-------------------------
        adjustment_dicts = dict()
        for PN_i in PNs_to_resolve:
            assert(PN_i not in adjustment_dicts.keys())
            res_dict_i = DOVSAudit.resolve_overlapping_audits_for_PN(
                audit_a               = audit_a, 
                audit_b               = audit_b, 
                PN_i                  = PN_i, 
                implement_resolutions = False, 
                unq_idfr_cols         = unq_idfr_cols, 
                PN_col                = PN_col, 
                t_min_col             = t_min_col, 
                t_max_col             = t_max_col, 
                keep_col              = keep_col, 
                open_beg_col          = open_beg_col, 
                open_end_col          = open_end_col
            )
            adjustment_dicts[PN_i] = res_dict_i
        #-------------------------
        DOVSAudit.implement_adjustments_for_audit(
            audit_a          = audit_a,
            audit_b          = audit_b, 
            adjustment_dicts = adjustment_dicts, 
            unq_idfr_cols    = unq_idfr_cols, 
            t_min_col        = t_min_col, 
            t_max_col        = t_max_col, 
            keep_col         = keep_col, 
            open_beg_col     = open_beg_col, 
            open_end_col     = open_end_col
        )
        #-------------------------
        return adjustment_dicts
    
    
    #****************************************************************************************************
    # Reporting
    #****************************************************************************************************
    def generate_warnings(
        self, 
        resolved_col='resolved', 
        resolved_details_col='resolved_details', 
        overlap_disagree_cols=['ovrlp_disagree_typeA', 'ovrlp_disagree_typeB'], 
        overlap_disagree_cols_descs = [
            'Type A: DOVS claimed there was an overlapping outage, but we cannot find one', 
            'Type B: DOVS did not find an overlapping outage, but we did'
        ]
    ):
        r"""
        """
        #-------------------------
        warnings_dict = dict()
        cumulative_bool = pd.Series(data=False, index=self.best_ests_df_w_keep_info.index)
        #----------------------------------------------------------------------------------------------------
        # Any entries which were resolved but for which the resolution was uncertain
        desc = 'Resolved but uncertain'
        bool_srs = (
            (self.best_ests_df_w_keep_info[resolved_col]==True) & 
            (self.best_ests_df_w_keep_info[resolved_details_col]!='Certain')
        )
        assert(desc not in warnings_dict.keys())
        warnings_dict[desc] = self.best_ests_df_w_keep_info.loc[bool_srs].copy()
        cumulative_bool     = cumulative_bool | bool_srs

        #----------------------------------------------------------------------------------------------------
        # Any entries left unresolved
        desc = 'Unresolved'
        bool_srs = self.best_ests_df_w_keep_info[resolved_details_col]=='Unresolved'
        assert(desc not in warnings_dict.keys())
        warnings_dict[desc] = self.best_ests_df_w_keep_info.loc[bool_srs].copy()
        cumulative_bool     = cumulative_bool | bool_srs

        #----------------------------------------------------------------------------------------------------
        # Any entries where disagreements between DOVS and found percentages
        #-----
        if overlap_disagree_cols_descs is not None:
            assert(len(overlap_disagree_cols)==len(overlap_disagree_cols_descs))
        #-----
        for i_disagree, disagree_col in enumerate(overlap_disagree_cols):
            if overlap_disagree_cols_descs is None:
                desc = f'Type disagreement found: {disagree_col}'
            else:
                desc = overlap_disagree_cols_descs[i_disagree]
            bool_srs = self.best_ests_df_w_keep_info[disagree_col].notna()
            assert(desc not in warnings_dict.keys())
            warnings_dict[desc] = self.best_ests_df_w_keep_info.loc[bool_srs].copy()
            cumulative_bool     = cumulative_bool | bool_srs

        #----------------------------------------------------------------------------------------------------
        # Cumulative
        self.best_est_df_entries_w_warning = self.best_ests_df_w_keep_info.loc[cumulative_bool].copy()
        self.warnings_dict = copy.deepcopy(warnings_dict)
        #-------------------------
        if not any([warn_df_i.shape[0]>0 for warn_df_i in self.warnings_dict.values()]):
            assert(self.best_est_df_entries_w_warning is None or self.best_est_df_entries_w_warning.shape[0]==0)
            self.warnings_flag = False
        else:
            assert(self.best_est_df_entries_w_warning.shape[0]>0)
            self.warnings_flag = True
        
    def generate_warnings_text(
        self
    ):
        r"""
        """
        #-------------------------
        if not any([warn_df_i.shape[0]>0 for warn_df_i in self.warnings_dict.values()]):
            assert(self.best_est_df_entries_w_warning is None or self.best_est_df_entries_w_warning.shape[0]==0)
            return ''
        #-------------------------
        return_text = f"{'*'*100}\nOUTG_REC_NB = {self.outg_rec_nb}\n\n"
        #-------------------------
        return_text += f"{'*'*50}\nALL estimate entries with warnings\n{'-'*10}\n"
        return_text += self.best_est_df_entries_w_warning.to_csv()
        return_text += f"{'-'*50}\n\n"
        #-------------------------
        for warning_i, entries_i in self.warnings_dict.items():
            if entries_i.shape[0]>0:
                return_text += f"{'*'*50}\nWarning: {warning_i}\n{'-'*10}\n"
                return_text += entries_i.to_csv()
                return_text += f"{'-'*50}\n\n"
        return_text += f"{'*'*100}\n\n"
        #-------------------------
        return return_text
    
    @staticmethod
    def build_detailed_summary_df(
        means_df, 
        best_ests_df_w_db_lbl,
        CI_tot, 
        CMI_tot, 
        n_PNs_ami, 
        outg_rec_nb, 
        dovs_df_i, 
        warnings_flag, 
        db_label_col='db_label', 
        winner_min_col='winner_min', 
        winner_max_col='winner_max', 
        PN_col='PN',
        i_outg_col='i_outg'
    ):
        r"""
        Build a summmary_df for outage.
        This summary is really just a single-row DataFrame (i.e., a series).
        The intent is for this to be used when running over a large number of outages.

        PN_col:
            In the most typical case, where CI/CMI is set by premise number, this should be set to the
              premise number column.
            However, if one wants to calculate CI/CMI by serial number, this should be set to the serial number column.
        """
        #-------------------------
        # Don't want to alter means_df, best_ests_df_w_db_lbl outside of function
        means_df=means_df.copy()
        best_ests_df_w_db_lbl=best_ests_df_w_db_lbl.copy()

        #--------------------------------------------------
        # Get DOVS info
        assert(
            isinstance(dovs_df_i, pd.Series) or 
            (isinstance(dovs_df_i, pd.DataFrame) and dovs_df_i.shape[0]==1)
        )
        #-------------------------
        # Make sure OUTG_REC_NB in dovs_df_i agrees with input value safecheck
        if isinstance(dovs_df_i, pd.DataFrame):
            if dovs_df_i.index.name=='OUTG_REC_NB':
                assert(dovs_df_i.index[0]==outg_rec_nb)
            else:
                assert('OUTG_REC_NB' in dovs_df_i.columns)
                assert(dovs_df_i.iloc[0]['OUTG_REC_NB']==outg_rec_nb)
        else:
            if 'OUTG_REC_NB' in dovs_df_i.index:
                assert(dovs_df_i['OUTG_REC_NB']==outg_rec_nb)
            else:
                assert(dovs_df_i.name==outg_rec_nb)

        #-------------------------
        # Make dovs_df_i a pd.Series object (if pd.DataFrame, this will collapse to pd.Series, if
        #   pd.Series, this will have no effect)
        # This isn't necessary, it just eliminate the need to use .loc[0] all over the place
        dovs_df_i = dovs_df_i.squeeze()

        #-------------------------
        # Grab needed entries from DOVS data
        dovs_outg_t_beg = dovs_df_i['DT_OFF_TS_FULL']
        dovs_outg_t_end = dovs_df_i['DT_ON_TS']
        ci_dovs         = dovs_df_i['CI_NB']
        cmi_dovs        = dovs_df_i['CMI_NB']
        n_PNs_dovs      = len(set(dovs_df_i['premise_nbs']))
        outage_nb       = dovs_df_i['OUTAGE_NB']
        mjr_cause_cd    = dovs_df_i['MJR_CAUSE_CD']
        mnr_cause_cd    = dovs_df_i['MNR_CAUSE_CD']
        dvc_typ_nb      = dovs_df_i['DVC_TYP_NM']   
        opco_id         = dovs_df_i['OPERATING_UNIT_ID']

        #--------------------------------------------------
        # If only a single sub-outage, only return 'Full Outage' entry
        if means_df.shape[0]==1:
            return_df = pd.DataFrame(
                data=dict(
                    outg_i_beg=means_df.iloc[0][winner_min_col], 
                    outg_i_end=means_df.iloc[0][winner_max_col], 
                    CI_i=CI_tot,
                    CMI_i=CMI_tot
                ), 
                index=pd.MultiIndex.from_tuples(
                    [(outage_nb, outg_rec_nb, 'Full Outage')], 
                    names=['OUTAGE_NB','OUTG_REC_NB', 'Outage Subset']
                )
            )
            # Remove fractional seconds
            return_df['outg_i_beg']=return_df['outg_i_beg'].dt.strftime('%Y-%m-%d %H:%M:%S')
            return_df['outg_i_end']=return_df['outg_i_end'].dt.strftime('%Y-%m-%d %H:%M:%S')

        #--------------------------------------------------
        else:  
            #-------------------------
            # The following protects against the case where one inputs means_df/best_ests_df_w_db_lbl which were built
            #   at the serial number level, but tries to run this function at the premise number level (by, e.g., setting
            #   PN_col='PN')
            # In such a case, the CI would still come out correct (due to how it is calculated below), but the CMI would be 
            #   larger than in reality, as premises with multiple SNs would get counted multiple times.
            if (best_ests_df_w_db_lbl.groupby([PN_col, i_outg_col]).size()>1).any():
                print('Each combination of PN_col, i_outg_col in best_ests_df_w_db_lbl should have a single entry')
                print('The input violates this!  CRASH IMMINENT!!!!!')
                print('Likely, one is trying to calculate CI/CMI by premise, but forgot to run DOVSAudit.combine_PNs_in_best_ests_df')
            assert((best_ests_df_w_db_lbl.groupby([PN_col, i_outg_col]).size()==1).all())

            #-------------------------
            # Add nPNs column to means_df, as this will be used for CI of the sub-outages
            n_PNs_col = f'n_{PN_col}s'
            #-----
            means_df = DOVSAudit.add_nPNs_to_means_df(
                means_df=means_df, 
                best_ests_df_w_db_lbl=best_ests_df_w_db_lbl,
                db_label_col=db_label_col, 
                PN_col=PN_col, 
                n_PNs_col=n_PNs_col
            )

            #-------------------------
            # Sort means_df
            means_df.sort_values(by=[winner_min_col, winner_max_col])

            #-------------------------
            # Create winner_max-winner_min column (winner_delta_col), used to calculate CMI
            winner_delta_col = Utilities.generate_random_string()
            best_ests_df_w_db_lbl[winner_delta_col] = best_ests_df_w_db_lbl[winner_max_col]-best_ests_df_w_db_lbl[winner_min_col]

            #-------------------------
            # Build a series containing the CMI values by db_label
            cmi_by_label = best_ests_df_w_db_lbl.groupby([db_label_col])[winner_delta_col].apply(lambda x: x.sum().total_seconds()/60)
            cmi_by_label.name = 'CMI_i'
            assert(len(set(cmi_by_label.index).symmetric_difference(means_df.index))==0)

            # The CMI_tot input should equal cmi_by_label.sum() (likely they won't be EXACTLY the same, hence
            #   the use of Utilities.are_approx_equal)
            # NOTE: The same cannot be said for CI, as individual meters can suffer multiple sub-outages
            assert(Utilities.are_approx_equal(CMI_tot, cmi_by_label.sum(), precision=0.00001))

            #-------------------------
            # Construct return_df
            return_df = means_df[[winner_min_col, winner_max_col, n_PNs_col]].copy()
            return_df = return_df.merge(cmi_by_label, left_index=True, right_index=True, how='left')

            # Remove fractional seconds
            return_df[winner_min_col]=return_df[winner_min_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            return_df[winner_max_col]=return_df[winner_max_col].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Adjust column names
            return_df=return_df.rename(columns={
                winner_min_col:'outg_i_beg', 
                winner_max_col:'outg_i_end', 
                n_PNs_col:'CI_i'
            })

            # Change index to sub-outage numbers
            return_df.index = [f'Sub-outage {i}' for i in range(1, return_df.shape[0]+1)]

            #-------------------------
            # Build a row containing the net outage information
            net_outg_row = pd.DataFrame(
                data=dict(
                    outg_i_beg=return_df['outg_i_beg'].min(), 
                    outg_i_end=return_df['outg_i_end'].max(), 
                    CI_i=CI_tot,
                    CMI_i=CMI_tot
                ), 
                index=['Full Outage']
            )

            # Add net_outg_row to beginning of return_df
            return_df = pd.concat([net_outg_row, return_df])
            return_df.index.name = 'Outage Subset'

            #-------------------------
            # Include the OUTG_REC_NB and OUTAGE_NB
            return_df = Utilities_df.prepend_level_to_MultiIndex(return_df, level_val=outg_rec_nb, level_name='OUTG_REC_NB', axis=0)
            return_df = Utilities_df.prepend_level_to_MultiIndex(return_df, level_val=outage_nb, level_name='OUTAGE_NB', axis=0)

        #--------------------------------------------------
        # Add n_PNs_ami, n_PNs_dovs, and %
        return_df['n_PNs_ami']     = n_PNs_ami
        return_df['n_PNs_DOVS']    = n_PNs_dovs
        return_df['pct_PNs_found'] = 100*return_df['n_PNs_ami']/return_df['n_PNs_DOVS']
        #-------------------------
        # Add DOVS
        return_df['DT_OFF_TS_FULL']    = dovs_outg_t_beg
        return_df['DT_ON_TS']          = dovs_outg_t_end
        return_df['CI_NB']             = ci_dovs
        return_df['CMI_NB']            = cmi_dovs
        return_df['MJR_CAUSE_CD']      = mjr_cause_cd
        return_df['MNR_CAUSE_CD']      = mnr_cause_cd
        return_df['DVC_TYP_NM']        = dvc_typ_nb
        return_df['OPERATING_UNIT_ID'] = opco_id
        
        #-------------------------
        # Warnings flag
        if warnings_flag==True:
            return_df['warnings_flag'] = int(warnings_flag)
        elif warnings_flag==False:
            return_df['warnings_flag'] = ''
        else:
            return_df['warnings_flag'] = warnings_flag

        #--------------------------------------------------
        # For new format, I still want 'OUTAGE_NB' and 'OUTG_REC_NB' as indices, but I'm going to move 'Outage Subset'
        #   into the columns
        # I also want to list all of the DOVS attributes first
        return_df=return_df.reset_index(level='Outage Subset')
        #-----
        return_df = Utilities_df.move_cols_to_front(
            return_df, 
            [
                'DT_OFF_TS_FULL', 
                'DT_ON_TS', 
                'CI_NB', 
                'CMI_NB', 
                'MJR_CAUSE_CD', 
                'MNR_CAUSE_CD', 
                'DVC_TYP_NM', 
                'OPERATING_UNIT_ID', 
                'n_PNs_DOVS'
            ]
        )
        #-------------------------
        return return_df