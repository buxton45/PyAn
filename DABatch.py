#!/usr/bin/env python

r"""
Holds DABatchOPCO and DABatch classes
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import time
import copy
from natsort import natsorted, natsort_keygen
#---------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from MeterPremise import MeterPremise
#-----
from AMI_SQL import DfToSqlMap
from AMINonVee_SQL import AMINonVee_SQL
from AMIEndEvents_SQL import AMIEndEvents_SQL
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from AMINonVee import AMINonVee
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
from DOVSAudit import DOVSAudit
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Plot_General
import Plot_Hist
import PDFMerger
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
#---------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------
#*****************************************************************************************************************************
#-----------------------------------------------------------------------------------------------------------------------------
class DABatchOPCO:
    r"""
    Class to adjust DOVS outage times and CI/CMI
    """
    def __init__(
        self                        , 
        date_0                      , 
        date_1                      , 
        opco                        , # e.g., 'oh'
        #--------------------
        save_dir_base               , 
        dates_subdir_appndx         = None, 
        save_results                = True, 
        #--------------------
        # DAQ arguments
        states                      = None, # e.g., ['OH']
        CI_NB_min                   = None, # e.g., 15
        mjr_mnr_cause               = None, 
        use_sql_std_outage          = True, 
        addtnl_outg_sql_kwargs      = None, 
        #--------------------
        daq_search_time_window      = pd.Timedelta('24 hours'), # Both for DAQ and DOVSAudit
        outg_rec_nbs                = None, 
        #--------------------
        # DOVSAudit arguments (if not None, it should be a dict with possible keys contained in dflt_dovsaudit_args below)
        dovs_audit_args             = None, 
        #--------------------
        init_dfs_to                 = pd.DataFrame(),  # Should be pd.DataFrame() or None
        conn_aws                    = None,
        conn_dovs                   = None
    ):
        r"""
        save_dir_base:
            The directory where all DOVSAudit results are housed.
            For me, this directory is r'C:\Users\s346557\Documents\LocalData\dovs_check'
            If save_dir_base is None and save_results==True:
                ==> os.path.join(Utilities.get_local_data_dir(), 'dovs_check') will be used

        outg_rec_nbs:
            Intended to be None, but included in case user wants to input own list of outg_rec_nbs
            The memeber self.outg_rec_nbs will be updated one self.outages_df is built

        dovs_audit_args:
            If None, dflt_dovsaudit_args (defined below) will be used.
            Otherwise, must be a dict object with keys taken from dflt_dovsaudit_args.
            e.g., if one wants combine_by_PN_likeness_thresh = pd.Timedelta('5 minutes') but all other default values,
              simple input dovs_audit_args = dict(combine_by_PN_likeness_thresh = pd.Timedelta('5 minutes'))
            NOTE: If the user inputs daq_search_time_window into dovs_audit_args, this will not throw any error (because 
                  this is an acceptable argument for DOVSAudit).  HOWEVER, the value will be replaced by the daq_search_time_window
                  value explicitly input into the constructor.
        """
        #----------------------------------------------------------------------------------------------------
        # When using self.init_dfs_to ALWAYS use copy.deepcopy(self.init_dfs_to),
        # otherwise, unintended consequences could return (self.init_dfs_to is None it
        # doesn't really matter, but when it is pd.DataFrame it does!)
        self.init_dfs_to = init_dfs_to

        #----------------------------------------------------------------------------------------------------
        # MAIN/MOST IMPORTANT ATTRIBUTE: audit_candidates_df
        # Note: For the lazy of us, one can grab via ac_df propert (i.e., one can call da_batch_obj.ac_df)
        self.__audit_candidates_df = copy.deepcopy(self.init_dfs_to)
        #-------------------------
        self.date_0 = date_0
        self.date_1 = date_1
        #-------------------------
        self.opco   = opco
        #-----
        accptbl_opcos = ['ap', 'ky', 'oh', 'im', 'pso', 'swp', 'tx']
        assert(self.opco in accptbl_opcos)
        #-------------------------
        self.save_results        = save_results
        self.dates_subdir_appndx = dates_subdir_appndx
        self.base_dirs           = None # Will be set in analyze_audits_for_opco/analyze_audits_for_opco_from_csvs
        #-----
        self.save_dir_base = save_dir_base
        if self.save_results and save_dir_base is None:
            self.save_dir_base = os.path.join(Utilities.get_local_data_dir(), 'dovs_check')
        #-----
        if self.save_dir_base is None:
            assert(not self.save_results)
            self.save_dir = None
        else:
            dates_subdir  = pd.to_datetime(date_0).strftime('%Y%m%d') + '_' + pd.to_datetime(date_1).strftime('%Y%m%d')
            if self.dates_subdir_appndx is not None:
                dates_subdir += dates_subdir_appndx
            #-----
            self.save_dir = os.path.join(self.save_dir_base, dates_subdir)
            #-----
            if self.opco is not None:
                self.save_dir = os.path.join(self.save_dir, self.opco)
            else:
                self.save_dir = os.path.join(self.save_dir, 'AllOPCOs')
        #-------------------------
        if self.save_results:
            assert(os.path.isdir(self.save_dir_base))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        #-------------------------
        # DAQ arguments
        self.states                 = states
        self.CI_NB_min              = CI_NB_min
        self.mjr_mnr_cause          = mjr_mnr_cause
        self.use_sql_std_outage     = use_sql_std_outage
        #--------------------
        self.addtnl_outg_sql_kwargs = addtnl_outg_sql_kwargs
        #-----
        if self.addtnl_outg_sql_kwargs is None:
            self.addtnl_outg_sql_kwargs = dict()
        #-----
        if not self.use_sql_std_outage:
            self.addtnl_outg_sql_kwargs['include_DOVS_PREMISE_DIM']         = True
            self.addtnl_outg_sql_kwargs['select_cols_DOVS_PREMISE_DIM']     = list(set(self.addtnl_outg_sql_kwargs.get('select_cols_DOVS_PREMISE_DIM', [])).union(set(['OFF_TM', 'REST_TM', 'PREMISE_NB'])))
            self.addtnl_outg_sql_kwargs['include_DOVS_CLEARING_DEVICE_DIM'] = True
        #--------------------
        self.daq_search_time_window = daq_search_time_window
        self.outg_rec_nbs           = outg_rec_nbs
        self.outg_rec_nbs_ami       = None

        #-------------------------
        # DOVSAudit arguments
        dflt_dovsaudit_args = dict(
            calculate_by_PN                 = True, 
            combine_by_PN_likeness_thresh   = pd.Timedelta('15 minutes'), 
            expand_outg_search_time_tight   = pd.Timedelta('1 hours'), 
            expand_outg_search_time_loose   = pd.Timedelta('12 hours'), 
            use_est_outg_times              = False, 
            use_full_ede_outgs              = False, 
            daq_search_time_window          = self.daq_search_time_window, 
            overlaps_addtnl_dovs_sql_kwargs = dict(
                CI_NB_min  = 0, 
                CMI_NB_min = 0
            ), 
        )
        #-----
        if dovs_audit_args is None:
            dovs_audit_args = dflt_dovsaudit_args
        assert(
            isinstance(dovs_audit_args, dict) and
            set(dovs_audit_args.keys()).difference(set(dflt_dovsaudit_args.keys())) == set()
        )
        #-----
        dovs_audit_args = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = dovs_audit_args, 
            default_values_dict = dflt_dovsaudit_args, 
            extend_any_lists    = True, 
            inplace             = False
        )
        #-----
        # Set daq_search_time_window attribute to agree with that input into DABatchOPCO init
        dovs_audit_args['daq_search_time_window'] = self.daq_search_time_window
        self.dovs_audit_args                      = copy.deepcopy(dovs_audit_args)

        #--------------------------------------------------
        self.outages_sql       = None
        self.outages_df        = copy.deepcopy(self.init_dfs_to)
        #-----
        self.outages_mp_df_OG  = copy.deepcopy(self.init_dfs_to)
        self.outages_mp_df     = copy.deepcopy(self.init_dfs_to)
        #-----
        self.outages_mp_df_ami = copy.deepcopy(self.init_dfs_to)
        self.outages_df_ami    = copy.deepcopy(self.init_dfs_to)
        #-----


        #---------------------------------------------------------------------------
        # Grabbing connection can take time (seconds, not hours).
        # Keep set to None, only creating if needed (see conn_aws/conn_dovs property below)
        self.__conn_aws   = conn_aws
        self.__conn_dovs  = conn_dovs
        #-------------------------

    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    @property
    def conn_aws(self):
        if self.__conn_aws is None:
            self.__conn_aws  = Utilities.get_athena_prod_aws_connection()
        return self.__conn_aws
    
    @property
    def conn_dovs(self):
        if self.__conn_dovs is None:
            self.__conn_dovs  = Utilities.get_utldb01p_oracle_connection()
        return self.__conn_dovs
    
    @property
    def audit_candidates_df(self):
        if self.__audit_candidates_df is None:
            return self.__audit_candidates_df
        else:
            return self.__audit_candidates_df.copy()
    @property
    def ac_df(self):
        return self.audit_candidates_df
    
    #----------------------------------------------------------------------------------------------------
    def set_audit_candidates_df(
        self                , 
        audit_candidates_df , 
    ):
        self.__audit_candidates_df = audit_candidates_df


    #--------------------------------------------------
    def load_audit_candidates_df_if_exist(
        self     , 
        verbose  = True
    ):
        r"""
        """
        #--------------------------------------------------
        if not os.path.isdir(self.save_dir):
            #------
            if verbose:
                print(f'No audit_candidates_df loaded, directory DNE! ({self.save_dir})')
            #-----
            return False
        #-------------------------
        ac_df_path = os.path.join(self.save_dir, 'audit_candidates_df.pkl')
        if os.path.exists(ac_df_path):
            ac_df = pd.read_pickle(ac_df_path)
            self.set_audit_candidates_df(audit_candidates_df = ac_df)
            #-----
            if verbose:
                print(f'Success! Loaded audit_candidates_df from {ac_df_path}')
            #-----
            return True
        #-------------------------
        if verbose:
            print(f'No audit_candidates_df loaded from {ac_df_path}')
        #-----
        return False

    #----------------------------------------------------------------------------------------------------
    def load_daq_prereqs(
        self       , 
        how        = 'min', 
        assert_min = True, 
        verbose    = True
    ):
        r"""
        Load all of the required attributes needed for batch data acquistion.
        The purpose here is to allow one to easily re-start a DAQ run that may have failed part way through.
    
        how:
            Should be either 'min' or 'full'
            For DAQ, (aside from any kwargs, e.g., opco) the only data really needed is audit_candidates_df.
            When 'min' is selected, only audit_candidates_df is loaded

        assert_min:
            If True, os.path.join(self.save_dir, 'audit_candidates_df.pkl') must exists
        """
        #--------------------------------------------------
        if not os.path.isdir(self.save_dir):
            #------
            if verbose:
                print(f'No prereqs loaded, directory DNE! ({self.save_dir})')
            #-----
            if assert_min:
                assert(0)
            #-----
            return False
        #--------------------------------------------------
        assert(os.path.isdir(self.save_dir))
        assert(how in ['min', 'full'])
        #-------------------------
        # In all cases, audit_candidates_df must be loaded
        ac_df_path = os.path.join(self.save_dir, 'audit_candidates_df.pkl')
        if assert_min:
            assert(os.path.exists(ac_df_path))
        if os.path.exists(ac_df_path):
            ac_df = pd.read_pickle(ac_df_path)
            self.set_audit_candidates_df(audit_candidates_df = ac_df)
            #-------------------------
            if how == 'min':
                if verbose:
                    print(f"Loaded minimum DAQ prereqs from base_dir = {self.save_dir}")
                return True
            else:
                success = {'audit_candidates_df': True}
        else:
            success = {'audit_candidates_df': False}
        #-------------------------
        other_dfs = {
            'outages_df'        : 'outages_df', 
            'outages_mp_df_OG'  : 'outages_mp_df_b4_dupl_rmvl', 
            'outages_mp_df'     : 'outages_mp_df', 
            'outages_df_ami'    : 'outages_df_ami', 
            'outages_mp_df_ami' : 'outages_mp_df_ami'
        }
        #-----
        for attr_name_i, file_name_i in other_dfs.items():
            path_i = os.path.join(self.save_dir, f'{file_name_i}.pkl')
            if os.path.exists(path_i):
                try:
                    df_i = pd.read_pickle(path_i)
                    setattr(self, attr_name_i, df_i)
                    success[attr_name_i] = True
                except:
                    success[attr_name_i] = False
            else:
                success[attr_name_i] = False
        #-------------------------
        if verbose:
            print(f"Loaded full DAQ prereqs from base_dir = {self.save_dir}")
            Utilities.print_dict_align_keys(
                dct        = success, 
                left_align = True
            )
        #-------------------------
        return np.all(list(success.values()))
    

    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    def compile_outages_sql(
        self
    ):
        r"""
        """
        #-------------------------
        if self.use_sql_std_outage:
            outages_sql = DOVSOutages_SQL.build_sql_std_outage(
                mjr_mnr_cause   = self.mjr_mnr_cause, 
                include_premise = True, 
                date_range      = [self.date_0, self.date_1], 
                states          = self.states, 
                opco            = self.opco, 
                CI_NB_min       = self.CI_NB_min, 
                outg_rec_nbs    = self.outg_rec_nbs, 
                **self.addtnl_outg_sql_kwargs
            )
        else:
            outages_sql = DOVSOutages_SQL.build_sql_outage(
                mjr_mnr_cause   = self.mjr_mnr_cause, 
                date_range      = [self.date_0, self.date_1], 
                states          = self.states, 
                opco            = self.opco, 
                CI_NB_min       = self.CI_NB_min, 
                outg_rec_nbs    = self.outg_rec_nbs, 
                **self.addtnl_outg_sql_kwargs
            )
        #-------------------------
        return outages_sql
    
    #--------------------------------------------------
    def get_outages_sql(
        self
    ):
        r"""
        """
        #-------------------------
        if self.outages_sql is None:
            return self.compile_outages_sql()
        else:
            return self.outages_sql
    
    #--------------------------------------------------
    def compile_outages_sql_statement(
        self
    ):
        r"""
        """
        #-------------------------
        outages_sql = self.compile_outages_sql()
        if isinstance(outages_sql, str):
            return outages_sql
        else:
            outages_sql_stmnt = outages_sql.get_sql_statement()
            assert(isinstance(outages_sql_stmnt, str))
            return outages_sql_stmnt
        
    #--------------------------------------------------
    def get_outages_sql_statement(
        self
    ):
        r"""
        """
        #-------------------------
        if self.outages_sql is None:
            return self.compile_outages_sql_statement()
        else:
            if isinstance(self.outages_sql, str):
                return self.outages_sql
            else:
                return_stmnt = self.outages_sql.get_sql_statement()
                assert(isinstance(return_stmnt, str))
                return return_stmnt
            

    #--------------------------------------------------
    def build_outages_df(
        self    , 
        verbose = True
    ):
        r"""
        """
        #--------------------------------------------------
        outages_sql = self.compile_outages_sql()
        #-------------------------
        if isinstance(outages_sql, str):
            outages_sql_stmnt = outages_sql
        else:
            outages_sql_stmnt = outages_sql.get_sql_statement()
        #-----
        assert(isinstance(outages_sql_stmnt, str))
        #--------------------------------------------------
        outages_df = pd.read_sql_query(
            outages_sql_stmnt, 
            self.conn_dovs
        )
        #-----
        outages_df = Utilities_df.convert_col_types(
            df                  = outages_df, 
            cols_and_types_dict = {
                'CI_NB'       : np.int64, 
                'CMI_NB'      : np.float64, 
                'OUTG_REC_NB' : [np.int64, str]
            }, 
            to_numeric_errors   = 'coerce', 
            inplace             = True
        )
        #--------------------------------------------------
        if self.save_results:
            assert(os.path.isdir(self.save_dir))
            outages_df.to_pickle(os.path.join(self.save_dir, 'outages_df.pkl'))
        #--------------------------------------------------
        if verbose:
            print(f'outages_sql_stmnt:\n{outages_sql_stmnt}\n\n')
            print(f"outages_df.shape = {outages_df.shape}")
            print(f"# OUTG_REC_NBs   = {outages_df['OUTG_REC_NB'].nunique()}")
        #--------------------------------------------------
        self.outages_sql  = outages_sql
        self.outages_df   = outages_df
        self.outg_rec_nbs = self.outages_df['OUTG_REC_NB'].unique().tolist()
        

    #----------------------------------------------------------------------------------------------------
    def build_outages_mp_df(
        self    , 
        verbose = True
    ):
        r"""
        """
        #--------------------------------------------------
        assert(self.outages_df.shape[0]>0)
        #--------------------------------------------------
        time_i = time.time()
        #-----
        outages_mp_df_OG = DOVSOutages.build_active_MP_for_outages_df(
            df_outage              = self.outages_df, 
            prem_nb_col            = 'PREMISE_NB', 
            is_slim                = False, 
            addtnl_mp_df_curr_cols = ['technology_tx'], 
            addtnl_mp_df_hist_cols = ['technology_tx'], 
            assert_all_PNs_found   = False
        )
        #-----
        time_i = time.time() - time_i
        #-----
        outages_mp_df_OG['inst_ts'] = pd.to_datetime(outages_mp_df_OG['inst_ts'])
        outages_mp_df_OG['rmvl_ts'] = pd.to_datetime(outages_mp_df_OG['rmvl_ts'])
        #-------------------------
        self.outages_mp_df_OG = outages_mp_df_OG
        #-------------------------
        if self.save_results:
            assert(os.path.isdir(self.save_dir))
            outages_mp_df_OG.to_pickle(os.path.join(self.save_dir, 'outages_mp_df_b4_dupl_rmvl.pkl'))
        #-----
        if verbose:
            print(f'Time for DOVSOutages.build_active_MP_for_outages_df: {time_i}')
        #--------------------------------------------------
        time_i = time.time()
        #-----
        outages_mp_df = MeterPremise.drop_approx_mp_duplicates(
            mp_df                 = outages_mp_df_OG.copy(), 
            fuzziness             = pd.Timedelta('1 hour'), 
            assert_single_overlap = True, 
            addtnl_groupby_cols   = ['OUTG_REC_NB', 'technology_tx'], 
            gpby_dropna           = False
        )
        #-----
        time_i = time.time() - time_i
        #-------------------------
        self.outages_mp_df = outages_mp_df
        #-------------------------
        if self.save_results:
            assert(os.path.isdir(self.save_dir))
            outages_mp_df.to_pickle(os.path.join(self.save_dir, 'outages_mp_df.pkl'))
        #-----
        if verbose:
            print(f'Time for drop_approx_mp_duplicates: {time_i}')
        #--------------------------------------------------
        if verbose:
            print('\nWARNING (I guess, more like a fact of life...): \n Some premises listed in DOVS are simply not found in AMI')
            print('For the current data:')
            print(f" #PNs DOVS: {self.outages_df['PREMISE_NB'].nunique()}")
            print(f" #PNs AMI:  {self.outages_mp_df['prem_nb'].nunique()}\n")
        #--------------------------------------------------
        # Really only want one entry per meter (here, meter being a mfr_devc_ser_nbr/prem_nb combination)
        # ALthough drop_duplicates was used, multiple entries could still exist if, e.g., a meter has two
        #   non-fuzzy-overlapping intervals
        # IF ASSERTION FAILS!:
        #   Simple-minded: Let's just keep the one with the most recent install date
        #   outages_mp_df = outages_mp_df.iloc[outages_mp_df.reset_index().groupby(['mfr_devc_ser_nbr', 'prem_nb', 'OUTG_REC_NB'])['inst_ts'].idxmax()]
        #   assert(all(outages_mp_df[['mfr_devc_ser_nbr', 'prem_nb', 'OUTG_REC_NB']].value_counts()==1))
        assert(all(outages_mp_df[['mfr_devc_ser_nbr', 'prem_nb', 'OUTG_REC_NB']].value_counts()==1))


    #--------------------------------------------------
    def build_outages_mp_df_ami(
        self    , 
        verbose = True
    ):
        r"""
        Keep only outages with all meters of typr AMI
        """
        #--------------------------------------------------
        assert(self.outages_mp_df.shape[0]>0)
        #--------------------------------------------------
        # Keep only outages with all meters of type AMI
        #     If, instead, one wanted to keep only trsf_pole_nbs with all meters of type AMI, one could do:
        #       outages_mp_df_ami = self.outages_mp_df.groupby(['trsf_pole_nb']).filter(lambda x: all(x['technology_tx']=='AMI'))
        outages_mp_df_ami = self.outages_mp_df.groupby(['OUTG_REC_NB']).filter(lambda x: all(x['technology_tx']=='AMI'))
        #-------------------------
        ami_outg_rec_nbs = outages_mp_df_ami['OUTG_REC_NB'].unique().tolist()
        outages_df_ami   = self.outages_df[self.outages_df['OUTG_REC_NB'].isin(ami_outg_rec_nbs)].copy()
        #--------------------------------------------------
        self.outages_mp_df_ami = outages_mp_df_ami
        self.outages_df_ami    = outages_df_ami
        self.outg_rec_nbs_ami  = self.outages_df_ami['OUTG_REC_NB'].unique().tolist()
        #--------------------------------------------------
        if self.save_results:
            assert(os.path.isdir(self.save_dir))
            outages_mp_df_ami.to_pickle(os.path.join(self.save_dir, 'outages_mp_df_ami.pkl'))
            outages_df_ami.to_pickle(os.path.join(self.save_dir, 'outages_df_ami.pkl'))
        #--------------------------------------------------
        if verbose:
            print(f"# Outages in outages_mp_df:     {self.outages_mp_df['OUTG_REC_NB'].nunique()}")
            print(f"# Outages in outages_mp_df_ami: {self.outages_mp_df_ami['OUTG_REC_NB'].nunique()}")


    #--------------------------------------------------
    def finalize_audit_candidates(
        self    , 
        verbose = True
    ):
        r"""
        The final audit_candidates_df pd.DataFrame object is a simple merge of self.outages_df_ami and self.outages_mp_df_ami
    
        NOTE (from development):
            - Here, when running batch DAQ, we need to merge DOVS with MP so we can group by trsf_pole_nb later when actually running the DAQ
            - This is mainly a convenience thing, so that (along with OUTG_REC_NB and operating company info) are ready in the output DAQ files 
              (otherwise, I would have to populate these data at run time, which would be annoying)
        """
        #--------------------------------------------------
        assert(
            self.outages_df_ami.shape[0]    > 0 and 
            self.outages_mp_df_ami.shape[0] > 0
        )
        #--------------------------------------------------
        ac_df = DOVSOutages.merge_df_outage_with_mp(
            df_outage          = self.outages_df_ami, 
            df_mp              = self.outages_mp_df_ami, 
            merge_on_outg      = ['OUTG_REC_NB', 'PREMISE_NB'], 
            merge_on_mp        = ['OUTG_REC_NB', 'prem_nb'], 
            cols_to_include_mp = None, 
            drop_cols          = None, 
            rename_cols        = None, 
            inplace            = True
        )
        #--------------------------------------------------
        if verbose:
            print(f"# Outages Final Candidates (before consolidation): {ac_df['OUTG_REC_NB'].nunique()}")
        #--------------------------------------------------
        # Consolidate ac_df down to slim form
        ac_df = DOVSOutages.consolidate_df_outage(
            df_outage                = ac_df, 
            addtnl_grpby_cols        = ['trsf_pole_nb'], 
            set_outg_rec_nb_as_index = False, 
            gpby_dropna              = False
        )
        #--------------------------------------------------
        if verbose:
            print(f"# Outages Final Candidates (after consolidation):  {ac_df['OUTG_REC_NB'].nunique()}")
        #--------------------------------------------------
        # Set search times
        ac_df = DOVSOutages.set_search_time_in_outage_df(
            df_outage          = ac_df, 
            search_time_window = self.daq_search_time_window
        )
        #--------------------------------------------------
        self.__audit_candidates_df = ac_df
        #--------------------------------------------------
        if self.save_results:
            assert(os.path.isdir(self.save_dir))
            ac_df.to_pickle(os.path.join(self.save_dir, 'audit_candidates_df.pkl'))

        #--------------------------------------------------
        # The following will find OUTG_REC_NB, trsf_pole_nb groups (since DOVSOutages.consolidate_df_outage was run with 
        #   addtnl_grpby_cols=['trsf_pole_nb']) for which the PNs found by DOVS are not equal to those found in MP
        # BUT, if they're missing from MP, does this necessarily mean they'll be missing from interval data?!
        #   SO, I'm not 100% sure whether or not these should be eliminated at this stage.
        #   Keeping them at this point will just lead to a slightly longer data collection time.
        #   If they should be left out, the functionality of DOVSAudit will eliminate them
        df_outg_missing_PNs = ac_df[ac_df.apply(
            lambda x: len(set(x['PREMISE_NBS']).symmetric_difference(set(x['prem_nb'])))!=0, 
            axis=1
        )]
        outg_rec_nbs_to_exclude = df_outg_missing_PNs['OUTG_REC_NB'].unique().tolist()
        #-----
        df_exclude_missing = ac_df[~ac_df['OUTG_REC_NB'].isin(outg_rec_nbs_to_exclude)]
        #-----
        if verbose:
            print('With excluding outages with premises missing from MP')
            print(f"# Outages: {df_exclude_missing['OUTG_REC_NB'].nunique()}")
            print(f"#PNs DOVS: {len(set(df_exclude_missing['PREMISE_NBS'].sum()))}")
            print(f"#PNs AMI:  {len(set(df_exclude_missing['prem_nb'].sum()))}")
            print()
            #-----
            print('Without excluding outages with premises missing from MP')
            print(f"# Outages: {ac_df['OUTG_REC_NB'].nunique()}")
            print(f"#PNs DOVS: {len(set(ac_df['PREMISE_NBS'].sum()))}")
            print(f"#PNs AMI:  {len(set(ac_df['prem_nb'].sum()))}")


    #--------------------------------------------------
    def run_ami_nonvee_daq(
        self, 
        batch_size    = 25, 
        verbose       = True, 
        n_update      = 1, 
    ):
        r"""
        """
        #--------------------------------------------------
        assert(self.save_results)
        assert(self.save_dir is not None)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        #--------------------------------------------------
        df_construct_type      = DFConstructType.kRunSqlQuery
        contstruct_df_args_ami = None
        addtnl_groupby_cols    = ['OUTG_REC_NB', 'trsf_pole_nb', 'OPERATING_UNIT_ID']
        cols_of_interest_ami   = TableInfos.AMINonVee_TI.std_columns_of_interest
        #-------------------------
        ami_sql_function_kwargs = dict(
            cols_of_interest                  = cols_of_interest_ami, 
            df_outage                         = self.audit_candidates_df, 
            build_sql_function                = AMINonVee_SQL.build_sql_usg, 
            build_sql_function_kwargs         = dict(
                states = self.states, 
                opco   = self.opco, 
            ), 
            join_mp_args                      = False, 
            df_args                           = dict(
                addtnl_groupby_cols = addtnl_groupby_cols, 
                mapping_to_ami      = DfToSqlMap(df_col='PREMISE_NBS', kwarg='premise_nbs', sql_col='aep_premise_nb'), 
                is_df_consolidated  = True
            ), 
            # GenAn - keys_to_pop in GenAn.build_sql_general
            field_to_split                    = 'df_outage', 
            field_to_split_location_in_kwargs = ['df_outage'], 
            save_and_dump                     = True,  
            sort_coll_to_split                = True,
            batch_size                        = batch_size, 
            verbose                           = verbose, 
            n_update                          = n_update
        )
        #-------------------------
        save_args = dict(
            save_to_file = True, 
            save_dir     = os.path.join(self.save_dir, 'AMINonVee'), 
            save_name    = r'ami_nonvee.csv', 
            index        = True
        )
        #--------------------------------------------------
        time_i = time.time()
        #-----
        exit_status = Utilities.run_tryexceptwhile_process(
            func                = AMINonVee,
            func_args_dict      = dict(
                df_construct_type         = df_construct_type, 
                contstruct_df_args        = contstruct_df_args_ami, 
                build_sql_function        = AMINonVee_SQL.build_sql_usg_for_outages, 
                build_sql_function_kwargs = ami_sql_function_kwargs, 
                init_df_in_constructor    = True, 
                save_args                 = save_args
            ), 
            max_calls_per_min   = 1, 
            lookback_period_min = 15, 
            max_calls_absolute  = 1000, 
            verbose             = True
        )
        #-----
        time_i = time.time() - time_i
        #-----
        print(f'exit_status = {exit_status}')
        print(f'Build Time = {time_i}')
    
    
    #--------------------------------------------------
    def run_ami_ede_daq(
        self, 
        pdpu_only     = True, 
        batch_size    = 25, 
        verbose       = True, 
        n_update      = 1, 
    ):
        r"""
        """
        #--------------------------------------------------
        assert(self.save_results)
        assert(self.save_dir is not None)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        #--------------------------------------------------
        df_construct_type              = DFConstructType.kRunSqlQuery
        contstruct_df_args_end_events  = None
        addtnl_groupby_cols            = ['OUTG_REC_NB', 'trsf_pole_nb', 'OPERATING_UNIT_ID']
        cols_of_interest_end_dev_event = TableInfos.AMIEndEvents_TI.std_columns_of_interest
        #-------------------------
        if pdpu_only:
            pd_ids   = ['3.26.0.47', '3.26.136.47', '3.26.136.66']
            pu_ids   = ['3.26.0.216', '3.26.136.216']
            pdpu_ids = pd_ids+pu_ids
            #-----
            enddeviceeventtypeids = pdpu_ids
        else:
            enddeviceeventtypeids = None
        #-------------------------
        end_events_sql_function_kwargs = dict(
            cols_of_interest                  = cols_of_interest_end_dev_event, 
            df_outage                         = self.audit_candidates_df, 
            build_sql_function                = AMIEndEvents_SQL.build_sql_end_events, 
            build_sql_function_kwargs         = dict(
                states                = self.states, 
                opco                  = self.opco, 
                enddeviceeventtypeids = enddeviceeventtypeids, 
            ), 
            join_mp_args                      = False, 
            df_args                           = dict(
                addtnl_groupby_cols = addtnl_groupby_cols, 
                mapping_to_ami      = DfToSqlMap(df_col='PREMISE_NBS', kwarg='premise_nbs', sql_col='aep_premise_nb'), 
                is_df_consolidated  = True
            ), 
            # GenAn - keys_to_pop in GenAn.build_sql_general
            field_to_split                    = 'df_outage', 
            field_to_split_location_in_kwargs = ['df_outage'], 
            save_and_dump                     = True, 
            sort_coll_to_split                = True,
            batch_size                        = batch_size, 
            verbose                           = verbose, 
            n_update                          = n_update
        )
        #-------------------------
        save_args = dict(
            save_to_file = True, 
            save_dir     = os.path.join(self.save_dir, 'EndEvents'), 
            save_name    = r'end_events.csv', 
            index        = True
        )
        #--------------------------------------------------
        time_i = time.time()
        #-----
        exit_status = Utilities.run_tryexceptwhile_process(
            func                = AMIEndEvents,
            func_args_dict      = dict(
                df_construct_type         = df_construct_type, 
                contstruct_df_args        = contstruct_df_args_end_events, 
                build_sql_function        = AMIEndEvents_SQL.build_sql_end_events_for_outages, 
                build_sql_function_kwargs = end_events_sql_function_kwargs, 
                init_df_in_constructor    = True, 
                save_args                 = save_args
            ), 
            max_calls_per_min   = 1, 
            lookback_period_min = 15, 
            max_calls_absolute  = 1000, 
            verbose             = True
        )
        #-----
        time_i = time.time() - time_i
        #-----
        print(f'exit_status = {exit_status}')
        print(f'Build Time = {time_i}')
        

    #--------------------------------------------------
    def generate_batch_daq_prereqs(
        self, 
        load_prereqs_if_exist = True, 
        verbose               = True
    ):
        r"""
        """
        #--------------------------------------------------
        # NOTE: For batch DAQ, only need audit_candidates_df to be loaded
        ac_loaded = False
        if load_prereqs_if_exist:
            ac_loaded = self.load_daq_prereqs(
                how        = 'min', 
                assert_min = False, 
                verbose    = verbose
            )
        #-----
        # If load_prereqs_if_exist is False or (load_prereqs_if_exist True but) ac_loaded is False,
        #   then full suite of prereqs must be built
        if(
            not load_prereqs_if_exist or 
            not ac_loaded
        ):
            self.build_outages_df(verbose = verbose)
            self.build_outages_mp_df(verbose = verbose)
            self.build_outages_mp_df_ami(verbose = verbose)
            self.finalize_audit_candidates(verbose = verbose)

    #--------------------------------------------------
    def run_batch_daq(
        self, 
        load_prereqs_if_exist = True, 
        batch_size_ami        = 25, 
        n_update_ami          = 1, 
        batch_size_ede        = 25, 
        n_update_ede          = 1, 
        pdpu_only_ede         = True, 
        verbose               = True
    ):
        r"""
        """
        #--------------------------------------------------
        assert(self.save_results)
        assert(self.save_dir is not None)
        #-------------------------
        self.generate_batch_daq_prereqs(
            load_prereqs_if_exist = load_prereqs_if_exist, 
            verbose               = verbose
        )
        #-------------------------
        # Perform DAQ
        self.run_ami_nonvee_daq(
            batch_size    = batch_size_ami, 
            verbose       = verbose, 
            n_update      = n_update_ami, 
        )
        self.run_ami_ede_daq(
            pdpu_only     = pdpu_only_ede, 
            batch_size    = batch_size_ede, 
            verbose       = verbose, 
            n_update      = n_update_ede, 
        )

    #----------------------------------------------------------------------------------------------------
    #--------------------------------------------------
    @staticmethod
    def build_base_dirs(
        date_0              , 
        date_1              , 
        opco                , 
        save_dir_base       = None, 
        dates_subdir_appndx = None, 
        assert_base_exists  = True, 
        assert_amiede_exist = True, 
    ):
        r"""
        date_0 & date_1:
            These define the period over which the analysis is run.
            Typically, the analysis is run for a single week (with a lag of two weeks)
                e.g, on 2025-02-04, I run the analysis for 
                    date_0 = 2025-01-19
                    date_1 = 2025-01-25
        
        save_dir_base:
            The directory where all DOVSAudit results are housed.
            For me, this directory is r'C:\Users\s346557\Documents\LocalData\dovs_check'
            If save_dir_base is None:
                ==> os.path.join(Utilities.get_local_data_dir(), 'dovs_check') will be used
        """
        #--------------------------------------------------
        if save_dir_base is None:
            save_dir_base = os.path.join(Utilities.get_local_data_dir(), 'dovs_check')
        if assert_base_exists:
            assert(os.path.isdir(save_dir_base))
        #-------------------------
        dates_subdir = pd.to_datetime(date_0).strftime('%Y%m%d') + '_' + pd.to_datetime(date_1).strftime('%Y%m%d')
        if dates_subdir_appndx is not None:
            dates_subdir += dates_subdir_appndx
        #-------------------------
        base_dir     = os.path.join(save_dir_base, dates_subdir, opco)
        base_dir_ami = os.path.join(base_dir, r'AMINonVee')
        base_dir_ede = os.path.join(base_dir, r'EndEvents')
        #-----
        save_dir        = os.path.join(base_dir, r'Results')
        dovs_audits_dir = os.path.join(save_dir, 'dovs_audits')
        #-----
        if assert_amiede_exist:
            assert(os.path.exists(base_dir_ami))
            assert(os.path.exists(base_dir_ede))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(dovs_audits_dir):
            os.makedirs(dovs_audits_dir)
        #-------------------------
        return dict(
            base_dir        = base_dir, 
            base_dir_ami    = base_dir_ami, 
            base_dir_ede    = base_dir_ede, 
            save_dir        = save_dir, 
            dovs_audits_dir = dovs_audits_dir
        )
    

    #--------------------------------------------------
    @staticmethod
    def build_pdf_paths_and_tmp_subdirs(
        save_dir                    , 
        include_suboutg_endpt_plots = True, 
        assert_dirs_dne_or_empty    = True, 
    ):
        r"""
        In some cases, keeping all of the PdfPages objects open becomes taxing in terms of memory
        Therefore, I will save all of the PDFs as separate documents, closing each at the end of each iteration
          and collecting all in their respective single files at the end.
        The following paths are for the final, single files
        """
        #--------------------------------------------------
        assert(os.path.isdir(save_dir))
        #--------------------------------------------------
        res_pdf_path   = os.path.join(save_dir, r'Results.pdf')
        #-------------------------
        res_dovs_beg_pdf_path   = Utilities.append_to_path(
            res_pdf_path, 
            '_dovs_beg', 
            ext_to_find='.pdf', 
            append_to_end_if_ext_no_found=False
        )
        #-------------------------
        res_w_endpts_pdf_path   = Utilities.append_to_path(
            res_pdf_path, 
            '_w_suboutg_endpt_plots', 
            ext_to_find='.pdf', 
            append_to_end_if_ext_no_found=False
        )
        #-------------------------
        pdf_paths = dict(
            res          = res_pdf_path, 
            res_dovs_beg = res_dovs_beg_pdf_path, 
        )
        if include_suboutg_endpt_plots:
            pdf_paths['res_w_endpts'] = res_w_endpts_pdf_path
        #--------------------------------------------------
        res_tmp_subdir          = 'TMP_Results'
        res_dovs_beg_tmp_subdir = 'TMP_Results_dovs_beg'
        res_w_endpts_tmp_subdir = 'TMP_Results_w_suboutg_endpt'
        #-----
        tmp_pdf_subdirs = dict(
            res          = res_tmp_subdir, 
            res_dovs_beg = res_dovs_beg_tmp_subdir, 
        )
        if include_suboutg_endpt_plots:
            tmp_pdf_subdirs['res_w_endpts'] = res_w_endpts_tmp_subdir
        #-----
        Utilities.make_tmp_save_dir(
            base_dir_path           = save_dir,
            tmp_dir_name            = list(tmp_pdf_subdirs.values()), 
            assert_dir_dne_or_empty = assert_dirs_dne_or_empty, 
            return_path             = False
        )
        #--------------------------------------------------
        return pdf_paths, tmp_pdf_subdirs
    

    #--------------------------------------------------
    @staticmethod
    def build_summary_paths_and_tmp_subdirs(
        save_dir                 , 
        assert_dirs_dne_or_empty = True
    ):
        r"""
        Decided to also save summary dfs in temporary directory.
        The reason is for continuing analyses which died partway through. 
        Having these available mean not needing to load the entire DOVSAudit objects to obtain the results.ss
        """
        #--------------------------------------------------
        assert(os.path.isdir(save_dir))
        #--------------------------------------------------
        ci_cmi_path            = os.path.join(save_dir, r'ci_cmi_summary')
        detailed_path          = os.path.join(save_dir, r'detailed_summary')
        detailed_dovs_beg_path = os.path.join(save_dir, r'detailed_summary_dovs_beg')
        #-------------------------
        summary_paths = dict(
            ci_cmi            = ci_cmi_path, 
            detailed          = detailed_path, 
            detailed_dovs_beg = detailed_dovs_beg_path
        )
        #--------------------------------------------------
        ci_cmi_tmp_subdir            = 'TMP_ci_cmi_summaries'
        detailed_tmp_subdir          = 'TMP_detailed_summaries'
        detailed_dovs_beg_tmp_subdir = 'TMP_detailed_summaries_dovs_beg'
        #-----
        tmp_summary_subdirs = dict(
            ci_cmi            = ci_cmi_tmp_subdir, 
            detailed          = detailed_tmp_subdir, 
            detailed_dovs_beg = detailed_dovs_beg_tmp_subdir
        )
        #-----
        Utilities.make_tmp_save_dir(
            base_dir_path           = save_dir,
            tmp_dir_name            = list(tmp_summary_subdirs.values()), 
            assert_dir_dne_or_empty = assert_dirs_dne_or_empty, 
            return_path             = False
        )
        #--------------------------------------------------
        return summary_paths, tmp_summary_subdirs
    
    
    #--------------------------------------------------
    @staticmethod
    def find_preexisting_audits(
        dovs_audits_dir , 
        glob_pattern       = r'*.pkl', 
        regex_pattern      = None
    ):
        r"""
        Find any pre-existing results in dovs_audits_dir
        """
        #-------------------------
        preex_audit_paths = Utilities.find_all_paths(
            base_dir      = dovs_audits_dir, 
            glob_pattern  = glob_pattern, 
            regex_pattern = regex_pattern
        )
        preex_audit_paths = natsorted(preex_audit_paths)
        preex_audits      = [Path(x).stem for x in preex_audit_paths]
        #-------------------------
        return preex_audits
    
    #--------------------------------------------------
    @staticmethod
    def find_preexisting_tmp_results(
        results_dir         , 
        tmp_summary_subdirs ,
        tmp_pdf_subdirs     , 
        tmp_warnings_subdir = 'TMP_warnings', 
        build_summary_dfs   = True, 
        perform_plotting    = True, 
    ):
        r"""
        Find any pre-existing results in the temporary results directories.
        These will still be around if, e.g., and analysis died partway through.
        In such a case, it is (expected to be) much quicker to simply grab the needed txt/pkl/pdf files
          from these temporary directories than loading in DOVSAudit from pkl file and then grabbing (and
          certainly much quicker than completely reanalyzing!)
    
        In order to be considered found, all results must be found!
        Hence all the opportunities to exit early and return []
        """
        #---------------------------------------------------------------------------
        assert(os.path.isdir(results_dir))
        #-------------------------
        # Store all found results in a collection, res_colls, which is just a list of lists!
        res_colls = []
        #-------------------------
        # Always need warnings, even if empty!!!!!
        if not os.path.isdir(os.path.join(results_dir, tmp_warnings_subdir)):
            return []
        #-----
        preex_warnings = DABatchOPCO.find_preexisting_audits(
            dovs_audits_dir    = os.path.join(results_dir, tmp_warnings_subdir), 
            glob_pattern       = r'*.txt', 
            regex_pattern      = None
        )
        #-----
        if len(preex_warnings)==0:
            return []
        #-----
        res_colls.append(preex_warnings)
        
        #--------------------------------------------------
        if build_summary_dfs:
            summary_colls = []
            for subdir_i in tmp_summary_subdirs.values():
                if not os.path.isdir(os.path.join(results_dir, subdir_i)):
                    return []
                preex_i = DABatchOPCO.find_preexisting_audits(
                    dovs_audits_dir    = os.path.join(results_dir, subdir_i), 
                    glob_pattern       = r'*.pkl', 
                    regex_pattern      = None
                )
                #-----
                if len(preex_i)==0:
                    return []
                #-----
                summary_colls.append(preex_i)
            #-------------------------
            preex_summaries = Utilities.get_intersection_of_lists(
                lol  = summary_colls, 
                sort = 'ascending'
            )
            #-----
            res_colls.append(preex_summaries)
        
        #--------------------------------------------------
        if perform_plotting:
            plotting_colls = []
            for type_i, subdir_i in tmp_pdf_subdirs.items():
                if not os.path.isdir(os.path.join(results_dir, subdir_i)):
                    return []
                preex_i = DABatchOPCO.find_preexisting_audits(
                    dovs_audits_dir    = os.path.join(results_dir, subdir_i), 
                    glob_pattern       = r'*.pdf', 
                    regex_pattern      = None
                )
                #-----
                if len(preex_i)==0:
                    return []
                #-----
                #-------------------------
                if type_i == 'res_w_endpts':
                    # res_w_endpts is a little annoying, as it will save two PDFs for each outg_rec_nb,
                    #   outg_rec_nb_0.pdf and outg_rec_nb_1.pdf
                    #-----
                    # Past functionality:    To be considered found, both must be found!
                    # Current functionality: To be considered found, _0 must be found!
                    #     Reason for change:   If algorithm cannot find any outage period, only _0 will be output!
                    found_0s = Utilities.replace_if_found_in_list_else_omit(
                        lst     = preex_i, 
                        pattern = r'(\d*)_0', 
                        repl    = r'\1',
                        count   = 0, 
                        flags   = 0
                    )
                    #-----
                    # found_1s = Utilities.replace_if_found_in_list_else_omit(
                    #     lst     = preex_i, 
                    #     pattern = r'(\d*)_1', 
                    #     repl    = r'\1',
                    #     count   = 0, 
                    #     flags   = 0
                    # )
                    # #-------------------------
                    # found_res_w_endpts = Utilities.get_intersection_of_lists(
                    #     lol  = [found_0s, found_1s], 
                    #     sort = 'ascending'
                    # )
                    #-----
                    found_res_w_endpts = found_0s
                    if len(found_res_w_endpts)==0:
                        return []
                    #-----
                    plotting_colls.append(found_res_w_endpts)
                else:
                    #-----
                    if len(preex_i)==0:
                        return []
                    #-----
                    plotting_colls.append(preex_i)
            #-------------------------
            preex_plots = Utilities.get_intersection_of_lists(
                lol  = plotting_colls, 
                sort = 'ascending'
            )
            #-----
            res_colls.append(preex_plots)
        
        #--------------------------------------------------
        # Finalize! Finally!!!
        preex_results = Utilities.get_intersection_of_lists(
            lol  = res_colls, 
            sort = 'ascending'
        )
        #-------------------------
        return preex_results
    
    #--------------------------------------------------
    @staticmethod
    def remove_preexisting_audits(
        all_outg_rec_nbs , 
        dovs_audits_dir  , 
        glob_pattern     = r'*.pkl', 
        regex_pattern    = None, 
        verbose          = True
    ):
        r"""
        Given a list of outage record numbers (all_outg_rec_nbs), look for pre-existing results in
          dovs_audits_dir, and remove those from list.
        Returns a subset of all_outg_rec_nbs
        """
        #-------------------------
        preex_audits = DABatchOPCO.find_preexisting_audits(
            dovs_audits_dir = dovs_audits_dir, 
            glob_pattern    = glob_pattern, 
            regex_pattern   = regex_pattern
        )
        #-----
        # Pre-existing should be a subset of all_outg_rec_nbs!
        assert(set(preex_audits).difference(set(all_outg_rec_nbs))==set())
        #-----
        outg_rec_nbs = list(set(all_outg_rec_nbs).difference(set(preex_audits)))
        #-------------------------
        if verbose:
            print(f'len(all_outg_rec_nbs): {len(all_outg_rec_nbs)}')
            print(f'len(preex_audits):     {len(preex_audits)}')
            print(f'len(outg_rec_nbs):     {len(outg_rec_nbs)}')
        #-------------------------
        return outg_rec_nbs
    
    #--------------------------------------------------
    @staticmethod
    def identify_preexisting_audits(
        all_outg_rec_nbs , 
        dovs_audits_dir  , 
        glob_pattern     = r'*.pkl', 
        regex_pattern    = None
    ):
        r"""
        Given a list of outage record numbers (all_outg_rec_nbs), look for pre-existing results in
          dovs_audits_dir, and identify those in list.
        Returns a dict object with keys equal to all_outg_rec_nbs and True/False values signifying whether
          or not pre-existing result.

        ESSENTIALLY REPLACED BY DABatchOPCO.identify_preexisting_results
        """
        #-------------------------
        preex_audits = DABatchOPCO.find_preexisting_audits(
            dovs_audits_dir = dovs_audits_dir, 
            glob_pattern    = glob_pattern, 
            regex_pattern   = regex_pattern
        )
        #-------------------------
        all_outg_rec_nbs_dict = {k:False for k in all_outg_rec_nbs}
        preex_audits_dict     = {k:True  for k in preex_audits}
        #-----
        all_outg_rec_nbs_dict = all_outg_rec_nbs_dict|preex_audits_dict
        #-------------------------
        # Sanity check
        assert(np.sum(list(all_outg_rec_nbs_dict.values())) == len(preex_audits))
        #-------------------------
        return all_outg_rec_nbs_dict
    

    #--------------------------------------------------
    @staticmethod
    def identify_preexisting_results(
        all_outg_rec_nbs    , 
        dovs_audits_dir     , 
        results_dir         , 
        tmp_summary_subdirs ,
        tmp_pdf_subdirs     , 
        tmp_warnings_subdir = 'TMP_warnings', 
        build_summary_dfs   = True, 
        perform_plotting    = True, 
    ):
        r"""
        Given a list of outage record numbers (all_outg_rec_nbs), look for pre-existing results in
          dovs_audits_dir and temporary results directories, and identify those in list.
        Returns a dict object with keys equal to all_outg_rec_nbs and 'tmp_result'/'audit'/False values signifying whether
          or not pre-existing result, and, if so, where.
        """
        #--------------------------------------------------
        preex_audits = DABatchOPCO.find_preexisting_audits(
            dovs_audits_dir = dovs_audits_dir, 
            glob_pattern    = r'*.pkl', 
            regex_pattern   = None
        )
        #-------------------------
        preex_tmp_results = DABatchOPCO.find_preexisting_tmp_results(
            results_dir         = results_dir, 
            tmp_summary_subdirs = tmp_summary_subdirs,
            tmp_pdf_subdirs     = tmp_pdf_subdirs, 
            tmp_warnings_subdir = tmp_warnings_subdir, 
            build_summary_dfs   = build_summary_dfs, 
            perform_plotting    = perform_plotting, 
        )
        #-------------------------
        # Combine preex_audits and preex_tmp_results, used for sanity check later
        preex_results = Utilities.melt_list_of_lists_and_return_unique(
            lol  = [preex_audits, preex_tmp_results], 
            sort = 'ascending'
        )
    
        #--------------------------------------------------
        all_outg_rec_nbs_dict  = {k:False        for k in all_outg_rec_nbs}
        preex_audits_dict      = {k:'audit'      for k in preex_audits}
        preex_tmp_results_dict = {k:'tmp_result' for k in preex_tmp_results}
        #-------------------------
        # NOTE: When merging dictionaries, e.g., x|y, if both dictionaries have the same keys, the values of dictionary x are 
        #         overwritten by the values of dictionary y
        # Below, we place preference on preex_tmp_results, as those are the faster to reload!
        #-----
        all_outg_rec_nbs_dict = all_outg_rec_nbs_dict|preex_audits_dict|preex_tmp_results_dict
        #-------------------------
        # Sanity check
        assert(np.sum([x != False for x in all_outg_rec_nbs_dict.values()]) == len(preex_results))
        #-------------------------
        return all_outg_rec_nbs_dict
    
    
    #--------------------------------------------------
    @staticmethod
    def perform_plotting_for_batch(
        audit_i                     , 
        n_PNs_w_power_threshold     , 
        save_dir                    , 
        tmp_pdf_subdirs             , 
        include_suboutg_endpt_plots = True, 
        fig_num                     = 0
    ):
        r"""
        """
        #--------------------------------------------------
        expected_pdf_subdirs = ['res', 'res_dovs_beg']
        if include_suboutg_endpt_plots:
            expected_pdf_subdirs.append('res_w_endpts')
        assert(set(tmp_pdf_subdirs.keys()).difference(set(expected_pdf_subdirs))==set())
        #-------------------------
        assert(os.path.isdir(save_dir))
        assert(np.all([os.path.isdir(os.path.join(save_dir, x)) for x in tmp_pdf_subdirs.values()]))
        #--------------------------------------------------
        fig, axs = audit_i.plot_results(
            include_dovs_beg_text      = True, 
            name                       = 'AMI', 
            expand_time                = pd.Timedelta('1 hour'), 
            n_PNs_w_power_threshold    = n_PNs_w_power_threshold, 
            fig_num                    = fig_num
        )    
        Plot_General.save_fig(
            fig         = fig, 
            save_dir    = os.path.join(save_dir, tmp_pdf_subdirs['res']), 
            save_name   = f"{audit_i.outg_rec_nb}.pdf", 
            bbox_inches = 'tight'
        )
        #-------------------------
        if include_suboutg_endpt_plots:
            # If creating endpoint plots, save this figure in that directory as well (before clearing out)
            Plot_General.save_fig(
                fig         = fig, 
                save_dir    = os.path.join(save_dir, tmp_pdf_subdirs['res_w_endpts']), 
                save_name   = f"{audit_i.outg_rec_nb}_0.pdf", 
                bbox_inches = 'tight'
            )
        fig.clear()
        plt.close(fig)
        fig_num += 1

        #--------------------------------------------------
        if include_suboutg_endpt_plots:
            fig_axs = audit_i.plot_zoomed_endpts(
                fig_num     = fig_num
            )
            if fig_axs is not None:
                fig = fig_axs[0]
                axs = fig_axs[1]
                #-------------------------
                Plot_General.save_fig(
                    fig         = fig, 
                    save_dir    = os.path.join(save_dir, tmp_pdf_subdirs['res_w_endpts']), 
                    save_name   = f"{audit_i.outg_rec_nb}_1.pdf", 
                    bbox_inches = 'tight'
                ) 
                fig.clear()
                plt.close(fig)
                fig_num += 1
        
        #--------------------------------------------------
        fig, axs = audit_i.plot_results_dovs_beg(
            include_full_alg_text      = True, 
            name                       = 'AMI w/ DOVS t_beg', 
            expand_time                = pd.Timedelta('1 hour'), 
            n_PNs_w_power_threshold    = n_PNs_w_power_threshold, 
            fig_num                    = fig_num
        )    
        Plot_General.save_fig(
            fig         = fig, 
            save_dir    = os.path.join(save_dir, tmp_pdf_subdirs['res_dovs_beg']), 
            save_name   = f"{audit_i.outg_rec_nb}.pdf", 
            bbox_inches = 'tight'
        )
        fig.clear()
        plt.close(fig)
        fig_num += 1
    
        #--------------------------------------------------
        return fig_num
    

    #--------------------------------------------------
    @staticmethod
    def load_needed_tmp_results_for_audit_i(
        outg_rec_nb_i       , 
        results_dir         , 
        tmp_summary_subdirs ,
        tmp_warnings_subdir = 'TMP_warnings', 
        build_summary_dfs   = True,   
        index_col           = None, 
    ):
        r"""
        NOTE: If the needed temporary files were written, this implies the run_result=='Pass'
        """
        #--------------------------------------------------
        assert(os.path.isdir(results_dir))
        #--------------------------------------------------
        warning_path_i = os.path.join(results_dir, tmp_warnings_subdir, f"{outg_rec_nb_i}.txt")
        assert(os.path.exists(warning_path_i))
        with open(warning_path_i, 'r') as f:
            warning_i = f.read()
        #--------------------------------------------------
        return_dict = dict()
        return_dict['warnings_text_i'] = warning_i
        #--------------------------------------------------
        if build_summary_dfs:
            nec_keys = ['ci_cmi', 'detailed', 'detailed_dovs_beg']
            assert(set(nec_keys).symmetric_difference(set(tmp_summary_subdirs.keys()))==set())
            for type_j, subdir_j in tmp_summary_subdirs.items():
                #-----
                path_j = os.path.join(results_dir, subdir_j, f"{outg_rec_nb_i}.pkl")
                assert(os.path.exists(path_j))
                #-----
                df_j = pd.read_pickle(path_j)
                #-------------------------
                if type_j   == 'ci_cmi':
                    return_dict['ci_cmi_summary_df_i']            = df_j
                elif type_j == 'detailed':
                    return_dict['detailed_summary_df_i']          = df_j
                elif type_j == 'detailed_dovs_beg':
                    return_dict['detailed_summary_df_dovs_beg_i'] = df_j
                else:
                    assert(0)
        else:
            return_dict['ci_cmi_summary_df_i']            = pd.DataFrame()
            return_dict['detailed_summary_df_i']          = pd.DataFrame()
            return_dict['detailed_summary_df_dovs_beg_i'] = pd.DataFrame()
        #--------------------------------------------------
        # As stated above, if the needed temporary files were written, this implies the run_result=='Pass'
        return_dict['run_result'] = 'Pass'
        #--------------------------------------------------
        return return_dict


    #--------------------------------------------------
    @staticmethod
    def analyze_audit_i_post_buildload_for_batch(
        audit_i                       , 
        preex_i                       , 
        base_dirs                     , 
        tmp_summary_subdirs           ,
        tmp_pdf_subdirs               , 
        tmp_warnings_subdir           = 'TMP_warnings', 
        build_summary_dfs             = True, 
        perform_plotting              = True, 
        #-----
        run_outg_inclusion_assessment = True, 
        max_pct_PNs_missing_allowed   = 0, 
        n_PNs_w_power_threshold       = 95, 
        include_suboutg_endpt_plots   = True, 
        fig_num                       = 0, 
    ):
        r"""
        """
        #--------------------------------------------------
        failed_return_dict = dict(
            detailed_summary_df_i          = pd.DataFrame(), 
            detailed_summary_df_dovs_beg_i = pd.DataFrame(), 
            ci_cmi_summary_df_i            = pd.DataFrame(), 
            warnings_text_i                = '', 
            fig_num                        = fig_num
        )
        #--------------------------------------------------
        if preex_i == 'tmp_result':
            return_dict = DABatchOPCO.load_needed_tmp_results_for_audit_i(
                outg_rec_nb_i       = audit_i, 
                results_dir         = base_dirs['save_dir'], 
                tmp_summary_subdirs = tmp_summary_subdirs,
                tmp_warnings_subdir = tmp_warnings_subdir, 
                build_summary_dfs   = build_summary_dfs,   
                index_col           = ['OUTAGE_NB', 'OUTG_REC_NB'], 
            )
            return_dict['audit_i'] = audit_i # outg_rec_nb, in this case
            return_dict['fig_num'] = fig_num
        else:
            if preex_i == False:
                run_result = audit_i.run_audit(
                    run_outg_inclusion_assessment = run_outg_inclusion_assessment, 
                    max_pct_PNs_missing_allowed   = max_pct_PNs_missing_allowed, 
                )
                #-------------------------
                audit_i.save(os.path.join(base_dirs['dovs_audits_dir'], f'{audit_i.outg_rec_nb}.pkl'))
                #-------------------------
                if run_result != 'Pass':
                    return_dict               = failed_return_dict
                    return_dict['run_result'] = run_result
                    return_dict['audit_i']    = audit_i
                    return return_dict
                #--------------------------------------------------
            else:
                assert(preex_i==True or preex_i=='audit')
                if audit_i.ami_df_i.shape[0]==0:
                    return_dict               = failed_return_dict
                    return_dict['run_result'] = "ami_df_i.shape[0]==0"
                    return_dict['audit_i']    = audit_i
                    return return_dict
                #-------------------------
                if run_outg_inclusion_assessment:
                    to_include_i = audit_i.self_assess_outage_inclusion_requirements(
                        max_pct_PNs_missing_allowed        = max_pct_PNs_missing_allowed, 
                        check_found_ami_for_all_SNs_kwargs = None
                    )
                    if not to_include_i:
                        return_dict               = failed_return_dict
                        return_dict['run_result'] = "Inclusion Requirements"
                        return_dict['audit_i']    = audit_i
                        return return_dict
                if not audit_i.can_analyze:
                    return_dict               = failed_return_dict
                    return_dict['run_result'] = "not can_analyze (likely overlapping DOVS)"
                    return_dict['audit_i']    = audit_i
                    return return_dict
            #----------------------------------------------------------------------------------------------------
            warnings_text_i = audit_i.generate_warnings_text()
            assert(os.path.isdir(os.path.join(base_dirs['save_dir'], tmp_warnings_subdir)))
            if warnings_text_i is None:
                warnings_text_i = ''
            with open(os.path.join(base_dirs['save_dir'], tmp_warnings_subdir, f"{audit_i.outg_rec_nb}.txt"), 'w') as f:
                f.write(warnings_text_i)
            #----------------------------------------------------------------------------------------------------
            if build_summary_dfs:
                detailed_summary_df_i, detailed_summary_df_dovs_beg_i = audit_i.get_detailed_summary_df_and_dovs_beg(
                    delta_t_off_cut         = pd.Timedelta('5min'), 
                    delta_t_on_cut          = pd.Timedelta('5min'), 
                    delta_ci_cut            = 3, 
                    delta_cmi_cut           = None, 
                    n_PNs_w_power_threshold = n_PNs_w_power_threshold, 
                )
                #-------------------------
                # If full algorithm found an outage but the _dovs_beg version did not, WE DO NOT WANT TO INCLUDE
                # Reason: It is likely that the outage began and ended before the beginning time as defined by DOVS,
                #           making it impossible for the _dovs_beg version to reconstruct the (real and nonzero) outage period
                if(
                    (audit_i.ci_dovs_beg==0 and audit_i.cmi_dovs_beg==0) and
                    not(audit_i.ci==0 and audit_i.cmi==0)
                ):
                    detailed_summary_df_dovs_beg_i = pd.DataFrame()
                #-------------------------
                ci_cmi_summary_df_i = audit_i.get_ci_cmi_summary(return_type = pd.DataFrame)
                #--------------------------------------------------
                ci_cmi_summary_df_i.to_pickle(           os.path.join(base_dirs['save_dir'], tmp_summary_subdirs['ci_cmi'],            f"{audit_i.outg_rec_nb}.pkl"))
                detailed_summary_df_i.to_pickle(         os.path.join(base_dirs['save_dir'], tmp_summary_subdirs['detailed'],          f"{audit_i.outg_rec_nb}.pkl"))
                detailed_summary_df_dovs_beg_i.to_pickle(os.path.join(base_dirs['save_dir'], tmp_summary_subdirs['detailed_dovs_beg'], f"{audit_i.outg_rec_nb}.pkl"))
            else:
                detailed_summary_df_i          = pd.DataFrame()
                detailed_summary_df_dovs_beg_i = pd.DataFrame()
                ci_cmi_summary_df_i            = pd.DataFrame()
            #----------------------------------------------------------------------------------------------------
            if perform_plotting:
                fig_num = DABatchOPCO.perform_plotting_for_batch(
                    audit_i                     = audit_i, 
                    n_PNs_w_power_threshold     = n_PNs_w_power_threshold, 
                    save_dir                    = base_dirs['save_dir'], 
                    tmp_pdf_subdirs             = tmp_pdf_subdirs, 
                    include_suboutg_endpt_plots = include_suboutg_endpt_plots, 
                    fig_num                     = fig_num
                )
            #----------------------------------------------------------------------------------------------------
            return_dict = dict(
                audit_i                        = audit_i, 
                run_result                     = 'Pass', 
                detailed_summary_df_i          = detailed_summary_df_i, 
                detailed_summary_df_dovs_beg_i = detailed_summary_df_dovs_beg_i, 
                ci_cmi_summary_df_i            = ci_cmi_summary_df_i, 
                warnings_text_i                = warnings_text_i, 
                fig_num                        = fig_num
            )
        return return_dict
    

    #--------------------------------------------------
    @staticmethod
    def analyze_audit_i_for_batch(
        outg_rec_nb_i                   , 
        preex_i                         , 
        opco                            , 
        dovs_df                         , 
        base_dirs                       , 
        tmp_summary_subdirs             ,
        tmp_pdf_subdirs                 , 
        tmp_warnings_subdir             = 'TMP_warnings', 
        build_summary_dfs               = True, 
        perform_plotting                = True, 
        #-----
        run_outg_inclusion_assessment   = True, 
        max_pct_PNs_missing_allowed     = 0, 
        n_PNs_w_power_threshold         = 95, 
        include_suboutg_endpt_plots     = True, 
        fig_num                         = 0, 
        #-----
        calculate_by_PN                 = True, 
        combine_by_PN_likeness_thresh   = pd.Timedelta('15 minutes'), 
        expand_outg_search_time_tight   = pd.Timedelta('1 hours'), 
        expand_outg_search_time_loose   = pd.Timedelta('12 hours'), 
        use_est_outg_times              = False, 
        use_full_ede_outgs              = False, 
        daq_search_time_window          = pd.Timedelta('24 hours'), 
        overlaps_addtnl_dovs_sql_kwargs = dict(
            CI_NB_min  = 0, 
            CMI_NB_min = 0
        ), 
    ):
        r"""
        """
        #--------------------------------------------------
        if preex_i == False:
            print('\tNo pre-existing audit found ==> building')
            #-------------------------
            audit_i = DOVSAudit(
                outg_rec_nb                     = outg_rec_nb_i, 
                opco                            = opco, 
                calculate_by_PN                 = calculate_by_PN, 
                combine_by_PN_likeness_thresh   = combine_by_PN_likeness_thresh, 
                expand_outg_search_time_tight   = expand_outg_search_time_tight, 
                expand_outg_search_time_loose   = expand_outg_search_time_loose, 
                use_est_outg_times              = use_est_outg_times, 
                use_full_ede_outgs              = use_full_ede_outgs, 
                daq_search_time_window          = daq_search_time_window, 
                overlaps_addtnl_dovs_sql_kwargs = overlaps_addtnl_dovs_sql_kwargs
            )
            
            #-------------------------
            audit_i.build_basic_data(
                slicers_ami                   = None, 
                build_sql_fncn_ami            = AMINonVee_SQL.build_sql_usg, 
                addtnl_build_sql_kwargs_ami   = None, 
                run_std_init_ami              = True, 
                save_args_ami                 = False, 
                #-----
                build_sql_fncn_ede            = AMIEndEvents_SQL.build_sql_end_events, 
                addtnl_build_sql_kwargs_ede   = None, 
                pdpu_only                     = True, 
                run_std_init_ede              = True, 
                save_args_ede                 = False, 
                #-----
                dovs_df                       = dovs_df, 
                assert_outg_rec_nb_in_dovs_df = True, 
                #-----
                mp_df                         = None, 
                mp_df_outg_rec_nb_col         = 'OUTG_REC_NB', 
                #-----
                drop_mp_dups_fuzziness        = pd.Timedelta('1 hour'), 
                addtnl_mp_df_cols             = ['technology_tx'], 
                assert_all_PNs_found          = True, 
                consolidate_PNs_batch_size    = 1000, 
                early_return                  = False, 
            )
    
        elif preex_i==True or preex_i=='audit':
            print('\tPre-existing audit found ==> loading')
            audit_i_path = os.path.join(base_dirs['dovs_audits_dir'], f'{outg_rec_nb_i}.pkl')
            assert(os.path.exists(audit_i_path))
            audit_i = DOVSAudit(outg_rec_nb = audit_i_path)

        else:
            print('\tPre-existing temporary results found ==> loading')
            assert(preex_i == 'tmp_result')
            audit_i = outg_rec_nb_i
        #----------------------------------------------------------------------------------------------------
        return_dict = DABatchOPCO.analyze_audit_i_post_buildload_for_batch(
            audit_i                       = audit_i, 
            preex_i                       = preex_i, 
            base_dirs                     = base_dirs, 
            tmp_summary_subdirs           = tmp_summary_subdirs, 
            tmp_pdf_subdirs               = tmp_pdf_subdirs, 
            tmp_warnings_subdir           = tmp_warnings_subdir, 
            build_summary_dfs             = build_summary_dfs, 
            perform_plotting              = perform_plotting, 
            #-----
            run_outg_inclusion_assessment = run_outg_inclusion_assessment, 
            max_pct_PNs_missing_allowed   = max_pct_PNs_missing_allowed, 
            n_PNs_w_power_threshold       = n_PNs_w_power_threshold, 
            include_suboutg_endpt_plots   = include_suboutg_endpt_plots, 
            fig_num                       = fig_num, 
        )
        #-------------------------
        return return_dict 


    #--------------------------------------------------
    @staticmethod
    def analyze_audit_i_from_csvs_for_batch(
        outg_rec_nb_i                   , 
        preex_i                         , 
        opco                            , 
        outg_rec_nb_to_files_dict       , 
        outg_rec_nb_to_files_ede_dict   , 
        dovs_df                         , 
        base_dirs                       , 
        tmp_summary_subdirs             ,
        tmp_pdf_subdirs                 , 
        tmp_warnings_subdir             = 'TMP_warnings', 
        build_summary_dfs               = True, 
        perform_plotting                = True, 
        #-----
        run_outg_inclusion_assessment   = True, 
        max_pct_PNs_missing_allowed     = 0, 
        n_PNs_w_power_threshold         = 95, 
        include_suboutg_endpt_plots     = True, 
        fig_num                         = 0, 
        #-----
        calculate_by_PN                 = True, 
        combine_by_PN_likeness_thresh   = pd.Timedelta('15 minutes'), 
        expand_outg_search_time_tight   = pd.Timedelta('1 hours'), 
        expand_outg_search_time_loose   = pd.Timedelta('12 hours'), 
        use_est_outg_times              = False, 
        use_full_ede_outgs              = False, 
        daq_search_time_window          = pd.Timedelta('24 hours'), # See note below!
        overlaps_addtnl_dovs_sql_kwargs = dict(
            CI_NB_min  = 0, 
            CMI_NB_min = 0
        )
    ):
        r"""
        NOTE:
            daq_search_time_window does nothing in this function, since data have already been acquired beforehand, and are
              simply being read in here before analyzed.
            It is kept simply to make the signature similar to that of analyze_audit_i_for_batch
            This makes life easier when using dovs_audit_args
        """
        #--------------------------------------------------
        if preex_i == False:
            print('\tNo pre-existing audit found ==> building')
            audit_i = DOVSAudit(
                outg_rec_nb                     = outg_rec_nb_i, 
                opco                            = opco, 
                calculate_by_PN                 = calculate_by_PN, 
                combine_by_PN_likeness_thresh   = combine_by_PN_likeness_thresh, 
                expand_outg_search_time_tight   = expand_outg_search_time_tight, 
                expand_outg_search_time_loose   = expand_outg_search_time_loose, 
                use_est_outg_times              = use_est_outg_times, 
                use_full_ede_outgs              = use_full_ede_outgs, 
                daq_search_time_window          = daq_search_time_window, 
                overlaps_addtnl_dovs_sql_kwargs = overlaps_addtnl_dovs_sql_kwargs
            )
    
            #--------------------------------------------------
            paths_ami = outg_rec_nb_to_files_dict[audit_i.outg_rec_nb]
            #-----
            if audit_i.outg_rec_nb in outg_rec_nb_to_files_ede_dict.keys():
                paths_ede = outg_rec_nb_to_files_ede_dict[audit_i.outg_rec_nb]
            else:
                paths_ede = None
            #-------------------------
            audit_i.load_basic_data_from_csvs(
                #-----
                paths_ami                          = paths_ami, 
                paths_ede                          = paths_ede, 
                slicers_ami                        = None, 
                run_std_init_ami                   = True, 
                run_std_init_ede                   = True, 
                #-----
                dovs_df                            = dovs_df, 
                assert_outg_rec_nb_in_dovs_df      = True, 
                #-----
                cols_and_types_to_convert_dict_ami = None, 
                to_numeric_errors_ami              = 'coerce', 
                drop_na_rows_when_exception_ami    = True, 
                drop_unnamed0_col_ami              = True, 
                pd_read_csv_kwargs_ami             = None, 
                make_all_columns_lowercase_ami     = False, 
                assert_all_cols_equal_ami          = True, 
                min_fsize_MB_ami                   = None, 
                #-----
                cols_and_types_to_convert_dict_ede = None, 
                to_numeric_errors_ede              = 'coerce', 
                drop_na_rows_when_exception_ede    = True, 
                drop_unnamed0_col_ede              = True, 
                pd_read_csv_kwargs_ede             = None, 
                make_all_columns_lowercase_ede     = False, 
                assert_all_cols_equal_ede          = True, 
                min_fsize_MB_ede                   = None, 
            )

        elif preex_i==True or preex_i=='audit':
            print('\tPre-existing audit found ==> loading')
            audit_i_path = os.path.join(base_dirs['dovs_audits_dir'], f'{outg_rec_nb_i}.pkl')
            assert(os.path.exists(audit_i_path))
            audit_i = DOVSAudit(outg_rec_nb = audit_i_path)

        else:
            print('\tPre-existing temporary results found ==> loading')
            assert(preex_i == 'tmp_result')
            audit_i = outg_rec_nb_i
        #----------------------------------------------------------------------------------------------------
        return_dict = DABatchOPCO.analyze_audit_i_post_buildload_for_batch(
            audit_i                       = audit_i, 
            preex_i                       = preex_i, 
            base_dirs                     = base_dirs, 
            tmp_summary_subdirs           = tmp_summary_subdirs, 
            tmp_pdf_subdirs               = tmp_pdf_subdirs, 
            tmp_warnings_subdir           = tmp_warnings_subdir, 
            build_summary_dfs             = build_summary_dfs, 
            perform_plotting              = perform_plotting, 
            #-----
            run_outg_inclusion_assessment = run_outg_inclusion_assessment, 
            max_pct_PNs_missing_allowed   = max_pct_PNs_missing_allowed, 
            n_PNs_w_power_threshold       = n_PNs_w_power_threshold, 
            include_suboutg_endpt_plots   = include_suboutg_endpt_plots, 
            fig_num                       = fig_num, 
        )
        #-------------------------
        return return_dict
    

    @staticmethod
    def sort_detailed_summary_dfs_helper(
        x                         , 
        how                       = 'abs_delta_ci', 
        #-----
        outg_subset_col           = 'Outage Subset', 
        full_outg_desc            = 'Full Outage', 
        #-----
        ami_ci_col                = 'CI_i', 
        ami_cmi_col               = 'CMI_i', 
        ami_t_off_col             = 'outg_i_beg', 
        ami_t_on_col              = 'outg_i_end', 
        #-----
        dovs_ci_col               = 'CI_NB', 
        dovs_cmi_col              = 'CMI_NB', 
        dovs_t_off_col            = 'DT_OFF_TS_FULL', 
        dovs_t_on_col             = 'DT_ON_TS', 
        dovs_outg_rec_nb_idfr_loc = 'OUTG_REC_NB', 
        dovs_outage_nb_idfr_loc   = 'OUTAGE_NB'
    ):
        r"""
        """
        #--------------------------------------------------
        x_full = x[x[outg_subset_col]==full_outg_desc]
        assert(x_full.shape[0]==1)
        x_full = x_full.iloc[0]
        #--------------------------------------------------
        if dovs_outg_rec_nb_idfr_loc[1]:
            outg_rec_nb = x_full.name[dovs_outg_rec_nb_idfr_loc[0]]
        else:
            outg_rec_nb = x_full[dovs_outg_rec_nb_idfr_loc[0]]
        #-------------------------
        if dovs_outage_nb_idfr_loc[1]:
            outage_nb = x_full.name[dovs_outage_nb_idfr_loc[0]]
        else:
            outage_nb = x_full[dovs_outage_nb_idfr_loc[0]]
        #--------------------------------------------------
        if how == 'abs_delta_ci':
            return abs(x_full[dovs_ci_col] - x_full[ami_ci_col]), x_full[dovs_ci_col], outg_rec_nb
        #--------------------------------------------------
        elif how == 'abs_delta_cmi':
            return abs(x_full[dovs_cmi_col] - x_full[ami_cmi_col]), x_full[dovs_cmi_col], outg_rec_nb
        #--------------------------------------------------
        elif how == 'abs_delta_ci_cmi':
            return (abs(x_full[dovs_ci_col] - x_full[ami_ci_col]), abs(x_full[dovs_cmi_col] - x_full[ami_cmi_col])), x_full[dovs_ci_col], x_full[dovs_cmi_col], outg_rec_nb
        #--------------------------------------------------
        elif how == 'abs_delta_cmi_ci':
            return (abs(x_full[dovs_cmi_col] - x_full[ami_cmi_col]), abs(x_full[dovs_ci_col] - x_full[ami_ci_col])), x_full[dovs_cmi_col], x_full[dovs_ci_col], outg_rec_nb
        #--------------------------------------------------
        else:
            assert(0)
    
    @staticmethod
    def sort_detailed_summary_dfs(
        detailed_summary_dfs , 
        how                  = 'abs_delta_ci', 
        #-----
        outg_subset_col      = 'Outage Subset', 
        full_outg_desc       = 'Full Outage', 
        #-----
        ami_ci_col           = 'CI_i', 
        ami_cmi_col          = 'CMI_i', 
        ami_t_off_col        = 'outg_i_beg', 
        ami_t_on_col         = 'outg_i_end', 
        #-----
        dovs_ci_col          = 'CI_NB', 
        dovs_cmi_col         = 'CMI_NB', 
        dovs_t_off_col       = 'DT_OFF_TS_FULL', 
        dovs_t_on_col        = 'DT_ON_TS', 
        dovs_outg_rec_nb_col = 'OUTG_REC_NB', 
        dovs_outage_nb_col   = 'OUTAGE_NB' 
    ):
        r"""
        """
        #--------------------------------------------------
        assert(Utilities.are_all_list_elements_of_type(detailed_summary_dfs, pd.DataFrame))
        if len(detailed_summary_dfs)==0:
            return detailed_summary_dfs
        #-------------------------
        accptbl_hows = [
            'abs_delta_ci', 
            'abs_delta_cmi', 
            'abs_delta_ci_cmi', 
            'abs_delta_cmi_ci'
        ]
        #-----
        assert(how in accptbl_hows)
        #--------------------------------------------------
        # Procedure assumes all fields of interest (see agruments to function) are found in the columns of the pd.DataFrame objects
        # HOWEVER, in many cases, OUTAGE_NB and OUTG_REC_NB will be found in the index.
        # This is fine, sort_detailed_summary_dfs_helper simply needs to be informed of this.
        # ALSO, all pd.DataFrame objects in all_detailed_summary_dfs must have the same structure (i.e., same columns and index names)
        #   NOTE: This is stricter than it really needs to be
        #--------------------------------------------------
        # Look for outg_rec(outage)_nb_idfr_loc in columns and index
        #-------------------------
        detailed_summary_df_i = detailed_summary_dfs[0]
        #-------------------------
        outg_rec_nb_idfr_loc = None
        #-----
        try:
            outg_rec_nb_idfr_loc  = Utilities_df.get_idfr_loc(detailed_summary_df_i, dovs_outg_rec_nb_col)
        except:
            outg_rec_nb_idfr_loc  = Utilities_df.get_idfr_loc(detailed_summary_df_i, ('index', dovs_outg_rec_nb_col))
        #-------------------------
        outage_nb_idfr_loc = None
        #-----
        try:
            outage_nb_idfr_loc  = Utilities_df.get_idfr_loc(detailed_summary_df_i, dovs_outage_nb_col)
        except:
            outage_nb_idfr_loc  = Utilities_df.get_idfr_loc(detailed_summary_df_i, ('index', dovs_outage_nb_col))
        #--------------------------------------------------
        if outg_rec_nb_idfr_loc[1]:
            for df_i in detailed_summary_dfs:
                assert(dovs_outg_rec_nb_col in df_i.index.names)
        else:
            for df_i in detailed_summary_dfs:
                assert(dovs_outg_rec_nb_col in df_i.columns.tolist())
        #-------------------------
        if outage_nb_idfr_loc[1]:
            for df_i in detailed_summary_dfs:
                assert(dovs_outage_nb_col in df_i.index.names)
        else:
            for df_i in detailed_summary_dfs:
                assert(dovs_outage_nb_col in df_i.columns.tolist())
        #--------------------------------------------------
        sorted_dfs = natsorted(
            detailed_summary_dfs, 
            reverse = True, 
            key     = lambda x: DABatchOPCO.sort_detailed_summary_dfs_helper(
                x                         = x, 
                how                       = how, 
                outg_subset_col           = outg_subset_col, 
                full_outg_desc            = full_outg_desc, 
                #-----
                ami_ci_col                = ami_ci_col, 
                ami_cmi_col               = ami_cmi_col, 
                ami_t_off_col             = ami_t_off_col, 
                ami_t_on_col              = ami_t_on_col, 
                #-----
                dovs_ci_col               = dovs_ci_col, 
                dovs_cmi_col              = dovs_cmi_col, 
                dovs_t_off_col            = dovs_t_off_col, 
                dovs_t_on_col             = dovs_t_on_col, 
                dovs_outg_rec_nb_idfr_loc = outg_rec_nb_idfr_loc, 
                dovs_outage_nb_idfr_loc   = outage_nb_idfr_loc
            )
        )
        #-------------------------
        return sorted_dfs 
    


    @staticmethod
    def sort_detailed_summary_dfs_flex_helper(
        x                    , 
        how                  = 'abs_delta_ci', 
        #-----
        outg_subset_col      = 'Outage Subset', 
        full_outg_desc       = 'Full Outage', 
        #-----
        ami_ci_col           = 'CI_i', 
        ami_cmi_col          = 'CMI_i', 
        ami_t_off_col        = 'outg_i_beg', 
        ami_t_on_col         = 'outg_i_end', 
        #-----
        dovs_ci_col          = 'CI_NB', 
        dovs_cmi_col         = 'CMI_NB', 
        dovs_t_off_col       = 'DT_OFF_TS_FULL', 
        dovs_t_on_col        = 'DT_ON_TS', 
        dovs_outg_rec_nb_col = 'OUTG_REC_NB', 
        dovs_outage_nb_col   = 'OUTAGE_NB'
    ):
        r"""
        """
        #--------------------------------------------------
        assert(isinstance(x, pd.DataFrame))
        #--------------------------------------------------
        # Look for outg_rec(outage)_nb_idfr_loc in columns and index
        #-------------------------
        outg_rec_nb_idfr_loc = None
        #-----
        try:
            outg_rec_nb_idfr_loc  = Utilities_df.get_idfr_loc(x, dovs_outg_rec_nb_col)
        except:
            outg_rec_nb_idfr_loc  = Utilities_df.get_idfr_loc(x, ('index', dovs_outg_rec_nb_col))
        #-------------------------
        outage_nb_idfr_loc = None
        #-----
        try:
            outage_nb_idfr_loc  = Utilities_df.get_idfr_loc(x, dovs_outage_nb_col)
        except:
            outage_nb_idfr_loc  = Utilities_df.get_idfr_loc(x, ('index', dovs_outage_nb_col))    
        #--------------------------------------------------
        x_full = x[x[outg_subset_col]==full_outg_desc]
        assert(x_full.shape[0]==1)
        x_full = x_full.iloc[0]
        #--------------------------------------------------
        if outg_rec_nb_idfr_loc[1]:
            outg_rec_nb = x_full.name[outg_rec_nb_idfr_loc[0]]
        else:
            outg_rec_nb = x_full[outg_rec_nb_idfr_loc[0]]
        #-------------------------
        if outage_nb_idfr_loc[1]:
            outage_nb = x_full.name[outage_nb_idfr_loc[0]]
        else:
            outage_nb = x_full[outage_nb_idfr_loc[0]]
        #--------------------------------------------------
        if how == 'abs_delta_ci':
            return abs(x_full[dovs_ci_col] - x_full[ami_ci_col]), x_full[dovs_ci_col], outg_rec_nb
        #--------------------------------------------------
        elif how == 'abs_delta_cmi':
            return abs(x_full[dovs_cmi_col] - x_full[ami_cmi_col]), x_full[dovs_cmi_col], outg_rec_nb
        #--------------------------------------------------
        elif how == 'abs_delta_ci_cmi':
            return (abs(x_full[dovs_ci_col] - x_full[ami_ci_col]), abs(x_full[dovs_cmi_col] - x_full[ami_cmi_col])), x_full[dovs_ci_col], x_full[dovs_cmi_col], outg_rec_nb
        #--------------------------------------------------
        elif how == 'abs_delta_cmi_ci':
            return (abs(x_full[dovs_cmi_col] - x_full[ami_cmi_col]), abs(x_full[dovs_ci_col] - x_full[ami_ci_col])), x_full[dovs_cmi_col], x_full[dovs_ci_col], outg_rec_nb
        #--------------------------------------------------
        else:
            assert(0)
    
    @staticmethod
    def sort_detailed_summary_dfs_flex(
        detailed_summary_dfs , 
        how                  = 'abs_delta_ci', 
        #-----
        outg_subset_col      = 'Outage Subset', 
        full_outg_desc       = 'Full Outage', 
        #-----
        ami_ci_col           = 'CI_i', 
        ami_cmi_col          = 'CMI_i', 
        ami_t_off_col        = 'outg_i_beg', 
        ami_t_on_col         = 'outg_i_end', 
        #-----
        dovs_ci_col          = 'CI_NB', 
        dovs_cmi_col         = 'CMI_NB', 
        dovs_t_off_col       = 'DT_OFF_TS_FULL', 
        dovs_t_on_col        = 'DT_ON_TS', 
        dovs_outg_rec_nb_col = 'OUTG_REC_NB', 
        dovs_outage_nb_col   = 'OUTAGE_NB' 
    ):
        r"""
        More flexible, but slower processing, compared to sort_detailed_summary_dfs
        """
        #--------------------------------------------------
        assert(Utilities.are_all_list_elements_of_type(detailed_summary_dfs, pd.DataFrame))
        #-------------------------
        accptbl_hows = [
            'abs_delta_ci', 
            'abs_delta_cmi', 
            'abs_delta_ci_cmi', 
            'abs_delta_cmi_ci'
        ]
        #-----
        assert(how in accptbl_hows)
        #-------------------------
        sorted_dfs = natsorted(
            detailed_summary_dfs, 
            reverse = True, 
            key     = lambda x: DABatchOPCO.sort_detailed_summary_dfs_flex_helper(
                x                    = x, 
                how                  = how, 
                outg_subset_col      = outg_subset_col, 
                full_outg_desc       = full_outg_desc, 
                #-----
                ami_ci_col           = ami_ci_col, 
                ami_cmi_col          = ami_cmi_col, 
                ami_t_off_col        = ami_t_off_col, 
                ami_t_on_col         = ami_t_on_col, 
                #-----
                dovs_ci_col          = dovs_ci_col, 
                dovs_cmi_col         = dovs_cmi_col, 
                dovs_t_off_col       = dovs_t_off_col, 
                dovs_t_on_col        = dovs_t_on_col, 
                dovs_outg_rec_nb_col = dovs_outg_rec_nb_col, 
                dovs_outage_nb_col   = dovs_outage_nb_col
            )
        )
        #-------------------------
        return sorted_dfs 
    

    @staticmethod
    def sort_detailed_summary_df(
        detailed_summary_df, 
        how                  = 'abs_delta_ci', 
        #-----
        outg_subset_col      = 'Outage Subset', 
        full_outg_desc       = 'Full Outage', 
        #-----
        ami_ci_col           = 'CI_i', 
        ami_cmi_col          = 'CMI_i', 
        ami_t_off_col        = 'outg_i_beg', 
        ami_t_on_col         = 'outg_i_end', 
        #-----
        dovs_ci_col          = 'CI_NB', 
        dovs_cmi_col         = 'CMI_NB', 
        dovs_t_off_col       = 'DT_OFF_TS_FULL', 
        dovs_t_on_col        = 'DT_ON_TS', 
        dovs_outg_rec_nb_col = 'OUTG_REC_NB', 
        dovs_outage_nb_col   = 'OUTAGE_NB'
    ):
        r"""
        Apparently there is no simple way to perform a groupby operation and sort the groups according to some prescription.
        For that reason, this is a bit clunkier than one would expect.
    
        Procedure is simplest when all fields of interest (see agruments to function) are found in the columns of the pd.DataFrame objects
        HOWEVER, in many cases, OUTAGE_NB and OUTG_REC_NB will be found in the index.
        To avoid all headeaches, I will call .reset_index() at the beginning and recover it at the end.
        """
        #--------------------------------------------------
        assert(isinstance(detailed_summary_df, pd.DataFrame))
        assert(outg_subset_col in detailed_summary_df.columns.tolist())
        assert(full_outg_desc in detailed_summary_df[outg_subset_col].unique().tolist())
        #-------------------------
        accptbl_hows = [
            'abs_delta_ci', 
            'abs_delta_cmi', 
            'abs_delta_ci_cmi', 
            'abs_delta_cmi_ci'
        ]
        #-----
        assert(how in accptbl_hows)
        #--------------------------------------------------
        # As mentioned in the above documentation, life is easier if I call .reset_index and restore the index later
        (detailed_summary_df, idx_names_new, idx_names_org) = Utilities_df.name_all_index_levels(df = detailed_summary_df)
        detailed_summary_df = detailed_summary_df.reset_index(drop=False)
        #-------------------------
        nec_cols = [
            outg_subset_col, 
            #-----
            ami_ci_col, 
            ami_cmi_col, 
            ami_t_off_col, 
            ami_t_on_col, 
            #-----
            dovs_ci_col, 
            dovs_cmi_col, 
            dovs_t_off_col, 
            dovs_t_on_col, 
            dovs_outg_rec_nb_col, 
            dovs_outage_nb_col
        ]
        assert(set(nec_cols+idx_names_new).difference(set(detailed_summary_df.columns.tolist()))==set())
        #--------------------------------------------------
        # The sorting proceeds according to the overall outage information (i.e., any sub-outages are ignored)
        full_outgs_df = detailed_summary_df[detailed_summary_df[outg_subset_col]==full_outg_desc].copy()
        #-----
        # Make sure there is only a single row per outage
        assert(full_outgs_df.groupby([dovs_outg_rec_nb_col, dovs_outage_nb_col]).ngroups==full_outgs_df.shape[0])
        #-------------------------
        # Generate some random column names to be used in the sorting operation
        tmp_delta_ci_col  = Utilities.generate_random_string(str_len=10, letters='letters_only')
        tmp_delta_cmi_col = Utilities.generate_random_string(str_len=10, letters='letters_only')
        #-----
        tmp_ordr_col = Utilities.generate_random_string(str_len=10, letters='letters_only')
        #-------------------------
        full_outgs_df[tmp_delta_ci_col]  = abs(full_outgs_df[dovs_ci_col]  - full_outgs_df[ami_ci_col])
        full_outgs_df[tmp_delta_cmi_col] = abs(full_outgs_df[dovs_cmi_col] - full_outgs_df[ami_cmi_col])
        #--------------------------------------------------
        if how == 'abs_delta_ci':
            full_outgs_df = full_outgs_df.sort_values(by=[tmp_delta_ci_col, dovs_ci_col, dovs_outg_rec_nb_col], ascending=False)
        #-------------------------
        elif how == 'abs_delta_cmi':
            full_outgs_df = full_outgs_df.sort_values(by=[tmp_delta_cmi_col, dovs_cmi_col, dovs_outg_rec_nb_col], ascending=False)
        #-------------------------
        elif how == 'abs_delta_ci_cmi':
            full_outgs_df = full_outgs_df.sort_values(by=[tmp_delta_ci_col, tmp_delta_cmi_col, dovs_ci_col, dovs_cmi_col, dovs_outg_rec_nb_col], ascending=False)
        #-------------------------
        elif how == 'abs_delta_cmi_ci':
            full_outgs_df = full_outgs_df.sort_values(by=[tmp_delta_cmi_col, tmp_delta_ci_col, dovs_cmi_col, dovs_ci_col, dovs_outg_rec_nb_col], ascending=False)
        #-------------------------
        else:
            assert(0)
        #--------------------------------------------------
        full_outgs_df[tmp_ordr_col] = range(0, full_outgs_df.shape[0])
        full_outgs_df = full_outgs_df[[dovs_outg_rec_nb_col, dovs_outage_nb_col, tmp_ordr_col]]
        #--------------------------------------------------
        srtd_df = pd.merge(
            detailed_summary_df, 
            full_outgs_df, 
            how      = 'inner', 
            left_on  = [dovs_outg_rec_nb_col, dovs_outage_nb_col], 
            right_on = [dovs_outg_rec_nb_col, dovs_outage_nb_col], 
        )
        assert(srtd_df.shape[0]==detailed_summary_df.shape[0])
        #-----
        srtd_df = srtd_df.sort_values(by=[tmp_ordr_col, outg_subset_col], ignore_index=True, ascending=True, key=natsort_keygen())
        srtd_df = srtd_df.drop(columns=[tmp_ordr_col])
        #--------------------------------------------------
        # Restore the original index
        srtd_df = srtd_df.set_index(idx_names_new)
        srtd_df.index.names = idx_names_org
        #--------------------------------------------------
        return srtd_df
    

    #--------------------------------------------------
    @staticmethod
    def finalize_summary_dfs(
        detailed_summary_df          , 
        detailed_summary_df_dovs_beg , 
        ci_cmi_summary_df            , 
        sort_how                     = 'abs_delta_ci_cmi'
    ):
        r"""
        """
        #--------------------------------------------------
        detailed_summary_df = DABatchOPCO.sort_detailed_summary_df(
            detailed_summary_df = detailed_summary_df, 
            how                 = sort_how, 
        )
        #-----
        detailed_summary_df_dovs_beg = DABatchOPCO.sort_detailed_summary_df(
            detailed_summary_df = detailed_summary_df_dovs_beg, 
            how                 = sort_how, 
        )
        #--------------------------------------------------
        # Mico asked for flag column to indicate if DOVS CI number (CI_NB column) above 15 
        detailed_summary_df['CI_NB > 15']          = detailed_summary_df['CI_NB']>15
        detailed_summary_df_dovs_beg['CI_NB > 15'] = detailed_summary_df_dovs_beg['CI_NB']>15
        #--------------------------------------------------
        ci_cmi_summary_df['ci_dovs']                     = ci_cmi_summary_df['ci_dovs'].astype(float)
        ci_cmi_summary_df['ci_ami']                      = ci_cmi_summary_df['ci_ami'].astype(float)
        ci_cmi_summary_df['ci_ami_dovs_beg']             = ci_cmi_summary_df['ci_ami_dovs_beg'].astype(float)
        #-----
        ci_cmi_summary_df['delta_ci_dovs_ami']           = ci_cmi_summary_df['ci_dovs']-ci_cmi_summary_df['ci_ami']
        ci_cmi_summary_df['delta_cmi_dovs_ami']          = ci_cmi_summary_df['cmi_dovs']-ci_cmi_summary_df['cmi_ami']
        #-----
        ci_cmi_summary_df['delta_ci_dovs_ami_dovs_beg']  = ci_cmi_summary_df['ci_dovs']-ci_cmi_summary_df['ci_ami_dovs_beg']
        ci_cmi_summary_df['delta_cmi_dovs_ami_dovs_beg'] = ci_cmi_summary_df['cmi_dovs']-ci_cmi_summary_df['cmi_ami_dovs_beg']
        #-----
        # For plotting purposes, make a outg_rec_in column which is simply 0 to delta_df.shape[0]-1
        ci_cmi_summary_df['outg_rec_int']                = range(ci_cmi_summary_df.shape[0])
        #--------------------------------------------------
        return (
            detailed_summary_df, 
            detailed_summary_df_dovs_beg, 
            ci_cmi_summary_df
        )
    
    
    #--------------------------------------------------
    @staticmethod
    def save_summary_dfs(
        save_dir                     , 
        detailed_summary_df          , 
        detailed_summary_df_dovs_beg , 
        ci_cmi_summary_df            , 
        detailed_path                = None, 
        detailed_dovs_beg_path       = None, 
        ci_cmi_path                  = None, 
    ):
        r"""
        NOTE: _path parameters should NOT include extensions, as these will be added!!!!!
        """
        #--------------------------------------------------
        assert(os.path.isdir(save_dir))
        #-------------------------
        if detailed_path is None:
            detailed_path          = os.path.join(save_dir, r'detailed_summary')
        if detailed_dovs_beg_path is None:
            detailed_dovs_beg_path = os.path.join(save_dir, r'detailed_summary_dovs_beg')
        if ci_cmi_path is None:
            ci_cmi_path            = os.path.join(save_dir, r'ci_cmi_summary')
        #-------------------------
        detailed_summary_df.to_pickle(         detailed_path          + '.pkl')
        detailed_summary_df_dovs_beg.to_pickle(detailed_dovs_beg_path + '.pkl')
        ci_cmi_summary_df.to_pickle(           ci_cmi_path            + '.pkl')
        #-----
        detailed_summary_df.to_csv(         detailed_path          + '.csv')
        detailed_summary_df_dovs_beg.to_csv(detailed_dovs_beg_path + '.csv')
        ci_cmi_summary_df.to_csv(           ci_cmi_path            + '.csv')

    #--------------------------------------------------
    @staticmethod
    def merge_all_pdf_tmp_subdirs(
        save_dir           , 
        pdf_paths          , 
        tmp_pdf_subdirs    , 
        delete_tmp_subdirs = True
    ):
        r"""
        """
        #--------------------------------------------------
        assert(os.path.isdir(save_dir))
        assert(pdf_paths.keys()==tmp_pdf_subdirs.keys())
        #-------------------------
        for k in pdf_paths.keys():
            PDFMerger.merge_all_pdfs_in_dir(
                dir_to_merge = os.path.join(save_dir, tmp_pdf_subdirs[k]), 
                output_path  = pdf_paths[k], 
            )
        #-------------------------
        if delete_tmp_subdirs:
            Utilities.del_tmp_save_dir(
                base_dir_path = save_dir,
                tmp_dir_name  = list(tmp_pdf_subdirs.values())
            )


    #--------------------------------------------------
    @staticmethod
    def merge_all_summary_tmp_subdirs(
        save_dir            , 
        summary_paths       , 
        tmp_summary_subdirs , 
        sort_how            = 'abs_delta_ci_cmi', 
        index_col           = None, 
        delete_tmp_subdirs  = True
    ):
        r"""
        """
        #--------------------------------------------------
        assert(os.path.isdir(save_dir))
        assert(summary_paths.keys()==tmp_summary_subdirs.keys())
        #-----
        nec_keys = ['ci_cmi', 'detailed', 'detailed_dovs_beg']
        assert(set(nec_keys).difference(set(summary_paths.keys()))==set())
        #-------------------------
        final_dfs = dict()
        for k in summary_paths.keys():
            assert(k not in final_dfs.keys())
            df_i = Utilities_df.concat_dfs_in_dir(
                dir_path             = os.path.join(save_dir, tmp_summary_subdirs[k]), 
                regex_pattern        = None, 
                ignore_case          = False, 
                ext                  = '.pkl', 
                make_col_types_equal = False, 
                index_col            = index_col, 
                return_paths         = False
            )
            #-----
            df_i = Utilities_df.drop_unnamed_columns(
                df            = df_i, 
                regex_pattern = r'Unnamed.*', 
                ignore_case   = True
            )
            #-----
            final_dfs[k] = df_i
        #-------------------------
        (
            detailed_summary_df, 
            detailed_summary_df_dovs_beg, 
            ci_cmi_summary_df
        ) = DABatchOPCO.finalize_summary_dfs(
            detailed_summary_df          = final_dfs['detailed'], 
            detailed_summary_df_dovs_beg = final_dfs['detailed_dovs_beg'], 
            ci_cmi_summary_df            = final_dfs['ci_cmi'], 
            sort_how                     = sort_how
        )
        #-------------------------
        DABatchOPCO.save_summary_dfs(
            save_dir                     = save_dir, 
            detailed_summary_df          = detailed_summary_df, 
            detailed_summary_df_dovs_beg = detailed_summary_df_dovs_beg, 
            ci_cmi_summary_df            = ci_cmi_summary_df, 
            detailed_path                = summary_paths['detailed'], 
            detailed_dovs_beg_path       = summary_paths['detailed_dovs_beg'], 
            ci_cmi_path                  = summary_paths['ci_cmi'], 
        )
        #-------------------------
        if delete_tmp_subdirs:
            Utilities.del_tmp_save_dir(
                base_dir_path = save_dir,
                tmp_dir_name  = list(tmp_summary_subdirs.values())
            )


    #--------------------------------------------------
    @staticmethod
    def merge_all_warnings(
        save_dir            , 
        warning_path        , 
        tmp_warnings_subdir , 
        delete_tmp_subdir   = True
    ):
        r"""
        warning_path:
            Where output will be written
            NOTE: warnings_path should NOT include extensions, as this will be added!!!!!
    
        tmp_warnings_subdir:
            Directory where individual warning files are contained
        """
        #--------------------------------------------------
        assert(os.path.isdir(save_dir))
        #-------------------------
        if warning_path is None:
            warning_path = os.path.join(save_dir, r'warnings')
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir      = os.path.join(save_dir, tmp_warnings_subdir), 
            glob_pattern  = r'*.txt', 
            regex_pattern = None
        )
        #-------------------------
        if len(paths)>0:
            with open(warning_path + '.txt', 'w') as outfile:
                for fname in paths:
                    with open(fname) as infile:
                        outfile.write(infile.read())
        #-------------------------
        if delete_tmp_subdir:
            Utilities.del_tmp_save_dir(
                base_dir_path = save_dir,
                tmp_dir_name  = tmp_warnings_subdir
            )


    #--------------------------------------------------
    def analyze_audits_for_opco(
        self, 
        load_prereqs_if_exist         = True, 
        reanalyze_preex_results       = False, 
        #-----
        perform_plotting              = True, 
        build_summary_dfs             = True, 
        #-----
        run_outg_inclusion_assessment = True, 
        max_pct_PNs_missing_allowed   = 0, 
        n_PNs_w_power_threshold       = 95, 
        include_suboutg_endpt_plots   = True, 
        fig_num                       = 0, 
        #-----
        verbose                       = True, 
        debug                         = False
    ):
        r"""
        For this method of running the analysis, the key member needed is self.outages_df_ami
        """
        #--------------------------------------------------
        if verbose:
            print('-----'*15)
            print(f"OPCO = {self.opco}")
            print('-----'*15)
        #--------------------------------------------------
        self.base_dirs = DABatchOPCO.build_base_dirs(
            date_0              = self.date_0, 
            date_1              = self.date_1, 
            opco                = self.opco, 
            save_dir_base       = self.save_dir_base, 
            dates_subdir_appndx = self.dates_subdir_appndx, 
            assert_base_exists  = True, 
            assert_amiede_exist = False, 
        )
        #-----
        tmp_warnings_subdir = 'TMP_warnings'
        Utilities.make_tmp_save_dir(
            base_dir_path           = self.base_dirs['save_dir'],
            tmp_dir_name            = tmp_warnings_subdir, 
            assert_dir_dne_or_empty = False, 
            return_path             = False
        )
        #-----
        summary_paths, tmp_summary_subdirs = None, None
        if build_summary_dfs:
            summary_paths, tmp_summary_subdirs = DABatchOPCO.build_summary_paths_and_tmp_subdirs(
                save_dir                 = self.base_dirs['save_dir'], 
                assert_dirs_dne_or_empty = False
            )
        #-----
        pdf_paths, tmp_pdf_subdirs = None, None
        if perform_plotting:
            pdf_paths, tmp_pdf_subdirs = DABatchOPCO.build_pdf_paths_and_tmp_subdirs(
                save_dir                    = self.base_dirs['save_dir'], 
                include_suboutg_endpt_plots = include_suboutg_endpt_plots, 
                assert_dirs_dne_or_empty    = False
            )
            
        #--------------------------------------------------
        if load_prereqs_if_exist:
            self.load_daq_prereqs(
                how        = 'full', 
                assert_min = False, 
                verbose    = verbose
            )
        #--------------------------------------------------
        # Build self.outages_df_ami if needed (needed because either load_prereqs_if_exist==False OR
        #   load_prereqs_if_exist==True but outages_df_ami not found)
        if self.outages_df_ami is None or self.outages_df_ami.shape[0]==0:
            self.generate_batch_daq_prereqs(
                load_prereqs_if_exist = False, 
                verbose               = verbose
            )
        #-------------------------
        dovs_df = DOVSOutages.consolidate_df_outage(
            df_outage                    = self.outages_df_ami, 
            outg_rec_nb_col              = 'OUTG_REC_NB', 
            addtnl_grpby_cols            = None, 
            cols_shared_by_group         = None, 
            cols_to_collect_in_lists     = None, 
            allow_duplicates_in_lists    = False, 
            allow_NaNs_in_lists          = False, 
            recover_uniqueness_violators = True, 
            gpby_dropna                  = False, 
            rename_cols                  = None,     
            premise_nb_col               = 'PREMISE_NB', 
            premise_nbs_col              = 'premise_nbs', 
            cols_to_drop                 = ['OFF_TM', 'REST_TM'], 
            sort_PNs                     = True, 
            drop_null_premise_nbs        = True, 
            set_outg_rec_nb_as_index     = True,
            drop_outg_rec_nb_if_index    = True, 
            verbose                      = True
        )
        outg_rec_nbs = dovs_df.index.unique().tolist()
        #--------------------------------------------------
        if reanalyze_preex_results:
            outg_rec_nbs = {k:False for k in outg_rec_nbs}
        else:
            outg_rec_nbs = DABatchOPCO.identify_preexisting_results(
                all_outg_rec_nbs    = outg_rec_nbs, 
                dovs_audits_dir     = self.base_dirs['dovs_audits_dir'], 
                results_dir         = self.base_dirs['save_dir'], 
                tmp_summary_subdirs = tmp_summary_subdirs,
                tmp_pdf_subdirs     = tmp_pdf_subdirs, 
                tmp_warnings_subdir = tmp_warnings_subdir, 
                build_summary_dfs   = build_summary_dfs, 
                perform_plotting    = perform_plotting, 
            )    
    
        #--------------------------------------------------
        outgs_pass = []
        outgs_fail = []
        #-------------------------
        all_detailed_summary_dfs          = []
        all_detailed_summary_dfs_dovs_beg = []
        ci_cmi_summary_dfs                = []
        warnings_text                     = ''
        
        if len(outg_rec_nbs)==0:
            print('No outages found (len(outg_rec_nbs)==0)!')
            return
        #----------------------------------------------------------------------------------------------------
        # Iterate through all outages
        for i_outg, (outg_rec_nb_i, preex_i) in enumerate(outg_rec_nbs.items()):
            if verbose:
                print(f'\n\ti_outg: {i_outg}/{len(outg_rec_nbs)-1}')
                print(f'\toutg_rec_nb_i = {outg_rec_nb_i}')
            #--------------------------------------------------
            try:
                res_dict_i = DABatchOPCO.analyze_audit_i_for_batch(
                    outg_rec_nb_i                 = outg_rec_nb_i, 
                    preex_i                       = preex_i, 
                    opco                          = self.opco, 
                    dovs_df                       = dovs_df, 
                    base_dirs                     = self.base_dirs, 
                    tmp_summary_subdirs           = tmp_summary_subdirs, 
                    tmp_pdf_subdirs               = tmp_pdf_subdirs, 
                    tmp_warnings_subdir           = tmp_warnings_subdir, 
                    build_summary_dfs             = build_summary_dfs, 
                    perform_plotting              = perform_plotting, 
                    #-----
                    run_outg_inclusion_assessment = run_outg_inclusion_assessment, 
                    max_pct_PNs_missing_allowed   = max_pct_PNs_missing_allowed, 
                    n_PNs_w_power_threshold       = n_PNs_w_power_threshold, 
                    include_suboutg_endpt_plots   = include_suboutg_endpt_plots, 
                    fig_num                       = fig_num, 
                    #-----
                    **self.dovs_audit_args
                )
                #-------------------------
                audit_i = res_dict_i['audit_i']
                #-----
                if preex_i != 'tmp_result':
                    outg_rec_nb_i = audit_i.outg_rec_nb
                else:
                    outg_rec_nb_i = audit_i
                #-----
                if res_dict_i['run_result'] != 'Pass':
                    outgs_fail.append((outg_rec_nb_i, res_dict_i['run_result']))
                else:
                    #-------------------------
                    if build_summary_dfs:
                        if res_dict_i['detailed_summary_df_i'].shape[0]>0:
                            all_detailed_summary_dfs.append(res_dict_i['detailed_summary_df_i'])
                        #-----
                        if res_dict_i['detailed_summary_df_dovs_beg_i'].shape[0]>0:
                            all_detailed_summary_dfs_dovs_beg.append(res_dict_i['detailed_summary_df_dovs_beg_i'])
                        #-------------------------
                        warnings_text += res_dict_i['warnings_text_i']
                        #-------------------------
                        ci_cmi_summary_df_i = res_dict_i['ci_cmi_summary_df_i']
                        ci_cmi_summary_df_i.index = [len(ci_cmi_summary_dfs)]
                        ci_cmi_summary_dfs.append(ci_cmi_summary_df_i)
                    #-------------------------
                    outgs_pass.append(outg_rec_nb_i)
                fig_num = res_dict_i['fig_num']
    
            except:
                outgs_fail.append((outg_rec_nb_i, "Unknown"))
                if debug:
                    raise
    
    
        #----------------------------------------------------------------------------------------------------
        # Below, delete_tmp_subdir(s) set to False in functions because, instead, all will be deleted once all successfully complete
        #-------------------------
        DABatchOPCO.merge_all_warnings(
            save_dir            = self.base_dirs['save_dir'], 
            warning_path        = None, 
            tmp_warnings_subdir = tmp_warnings_subdir, 
            delete_tmp_subdir   = False
        )
        #--------------------------------------------------
        if build_summary_dfs:
            DABatchOPCO.merge_all_summary_tmp_subdirs(
                save_dir            = self.base_dirs['save_dir'], 
                summary_paths       = summary_paths, 
                tmp_summary_subdirs = tmp_summary_subdirs, 
                sort_how            = 'abs_delta_ci_cmi', 
                index_col           = ['OUTAGE_NB', 'OUTG_REC_NB'], 
                delete_tmp_subdirs  = False
            )
        
        #--------------------------------------------------
        if perform_plotting:
            DABatchOPCO.merge_all_pdf_tmp_subdirs(
                save_dir           = self.base_dirs['save_dir'], 
                pdf_paths          = pdf_paths, 
                tmp_pdf_subdirs    = tmp_pdf_subdirs, 
                delete_tmp_subdirs = False
            )

        #--------------------------------------------------
        # Delete temporary subdirs
        Utilities.del_tmp_save_dir(
            base_dir_path = self.base_dirs['save_dir'],
            tmp_dir_name  = list(tmp_pdf_subdirs.values())
        )
        #-----
        Utilities.del_tmp_save_dir(
            base_dir_path = self.base_dirs['save_dir'],
            tmp_dir_name  = list(tmp_summary_subdirs.values())
        )
        #-----
        Utilities.del_tmp_save_dir(
            base_dir_path = self.base_dirs['save_dir'],
            tmp_dir_name  = tmp_warnings_subdir
        )
        
        #--------------------------------------------------
        if verbose:
            print(f"#OUTG_REC_NBs = {len(outg_rec_nbs)}")
            print(f"\tpass: {len(outgs_pass)}")
            print(f"\tfail: {len(outgs_fail)}")
        #--------------------------------------------------
        return dict(
            outg_rec_nbs = list(outg_rec_nbs.keys()), 
            outgs_pass   = outgs_pass, 
            outgs_fail   = outgs_fail, 
            fig_num      = fig_num
        )


    #--------------------------------------------------
    def analyze_audits_for_opco_from_csvs(
        self                               , 
        reanalyze_preex_results            = False, 
        perform_plotting                   = True, 
        build_summary_dfs                  = True, 
        #-----
        rebuild_outg_rec_nb_to_files_dicts = False, 
        #-----
        run_outg_inclusion_assessment      = True, 
        max_pct_PNs_missing_allowed        = 0, 
        n_PNs_w_power_threshold            = 95, 
        include_suboutg_endpt_plots        = True, 
        fig_num                            = 0, 
        #-----
        debug                              = False, 
        verbose                            = True, 
    ):
        r"""
        save_dir_base:
            The directory where all DOVSAudit results are housed.
            For me, this directory is r'C:\Users\s346557\Documents\LocalData\dovs_check'
            If save_dir_base is None:
                ==> os.path.join(Utilities.get_local_data_dir(), 'dovs_check') will be used

        debug:
            Using the try/except block is great for smoothing out potential unexpected issues.
            HOWEVER, imbedding the code in a try/except block makes it difficult to track down a bug, 
              as the except clause will be thrown UNLESS debug==True
            So, if the results aren't as expected (e.g., the program seems to run fine, but generates no or not many results), 
              the user should set debug = True
        """
        #--------------------------------------------------
        if verbose:
            print('-----'*15)
            print(f"OPCO = {self.opco}")
            print('-----'*15)
        #--------------------------------------------------
        assert(self.save_results)
        assert(os.path.isdir(self.save_dir_base))
        
        #--------------------------------------------------
        self.base_dirs = DABatchOPCO.build_base_dirs(
            date_0              = self.date_0, 
            date_1              = self.date_1, 
            opco                = self.opco, 
            save_dir_base       = self.save_dir_base, 
            dates_subdir_appndx = self.dates_subdir_appndx, 
            assert_base_exists  = True, 
            assert_amiede_exist = True, 
        )
        #-----
        tmp_warnings_subdir = 'TMP_warnings'
        Utilities.make_tmp_save_dir(
            base_dir_path           = self.base_dirs['save_dir'],
            tmp_dir_name            = tmp_warnings_subdir, 
            assert_dir_dne_or_empty = False, 
            return_path             = False
        )
        #-----
        summary_paths, tmp_summary_subdirs = None, None
        if build_summary_dfs:
            summary_paths, tmp_summary_subdirs = DABatchOPCO.build_summary_paths_and_tmp_subdirs(
                save_dir                 = self.base_dirs['save_dir'], 
                assert_dirs_dne_or_empty = False
            )
        #-----
        pdf_paths, tmp_pdf_subdirs = None, None
        if perform_plotting:
            pdf_paths, tmp_pdf_subdirs = DABatchOPCO.build_pdf_paths_and_tmp_subdirs(
                save_dir                    = self.base_dirs['save_dir'], 
                include_suboutg_endpt_plots = include_suboutg_endpt_plots, 
                assert_dirs_dne_or_empty    = False
            )
            
        #--------------------------------------------------
        outg_rec_nb_to_files_dict = DOVSAudit.get_outg_rec_nb_to_files_dict_ami(
            base_dir_dict   = self.base_dirs['base_dir'],
            base_dir_data   = self.base_dirs['base_dir_ami'], 
            rebuild         = rebuild_outg_rec_nb_to_files_dicts, 
            save_dict       = True
        )
        outg_rec_nbs = list(outg_rec_nb_to_files_dict.keys())
        #-------------------------
        outg_rec_nb_to_files_ede_dict = DOVSAudit.get_outg_rec_nb_to_files_dict_ede(
            base_dir_dict   = self.base_dirs['base_dir'],
            base_dir_data   = self.base_dirs['base_dir_ede'], 
            rebuild         = rebuild_outg_rec_nb_to_files_dicts, 
            save_dict       = True
        )
        
        #--------------------------------------------------
        if reanalyze_preex_results:
            outg_rec_nbs = {k:False for k in outg_rec_nbs}
        else:
            outg_rec_nbs = DABatchOPCO.identify_preexisting_results(
                all_outg_rec_nbs    = outg_rec_nbs, 
                dovs_audits_dir     = self.base_dirs['dovs_audits_dir'], 
                results_dir         = self.base_dirs['save_dir'], 
                tmp_summary_subdirs = tmp_summary_subdirs,
                tmp_pdf_subdirs     = tmp_pdf_subdirs, 
                tmp_warnings_subdir = tmp_warnings_subdir, 
                build_summary_dfs   = build_summary_dfs, 
                perform_plotting    = perform_plotting, 
            )    
    
        #--------------------------------------------------
        outgs_pass = []
        outgs_fail = []
        #-------------------------
        all_detailed_summary_dfs          = []
        all_detailed_summary_dfs_dovs_beg = []
        ci_cmi_summary_dfs                = []
        warnings_text                     = ''
        
        if len(outg_rec_nbs)==0:
            print('No outages found (len(outg_rec_nbs)==0)!')
            return
        #--------------------------------------------------
        # Build dovs_df
        dovs_df        = None
        outgs_to_build = [k for k,v in outg_rec_nbs.items() if v==False]
        if len(outgs_to_build)>0:
            if self.use_sql_std_outage:
                dovs = DOVSOutages(
                    df_construct_type         = DFConstructType.kRunSqlQuery, 
                    contstruct_df_args        = None, 
                    init_df_in_constructor    = True,
                    build_sql_function        = DOVSOutages_SQL.build_sql_std_outage, 
                    build_sql_function_kwargs = dict(
                        outg_rec_nbs    = outgs_to_build, 
                        field_to_split  = 'outg_rec_nbs', 
                        include_premise = True, 
                        opco            = self.opco, 
                        **self.addtnl_outg_sql_kwargs
                    ), 
                    build_consolidated        = True
                )
            else:
                dovs = DOVSOutages(
                    df_construct_type         = DFConstructType.kRunSqlQuery, 
                    contstruct_df_args        = None, 
                    init_df_in_constructor    = True,
                    build_sql_function        = DOVSOutages_SQL.build_sql_outage, 
                    build_sql_function_kwargs = dict(
                        outg_rec_nbs             = outgs_to_build, 
                        field_to_split           = 'outg_rec_nbs', 
                        opco                     = self.opco, 
                        **self.addtnl_outg_sql_kwargs
                    ), 
                    build_consolidated        = True
                )                
            dovs_df = dovs.df.copy()
        
        #----------------------------------------------------------------------------------------------------
        # Iterate through all outages
        for i_outg, (outg_rec_nb_i, preex_i) in enumerate(outg_rec_nbs.items()):
            if verbose:
                print(f'\n\ti_outg: {i_outg}/{len(outg_rec_nbs)-1}')
                print(f'\toutg_rec_nb_i = {outg_rec_nb_i}')
            #--------------------------------------------------
            try:
                res_dict_i = DABatchOPCO.analyze_audit_i_from_csvs_for_batch(
                    outg_rec_nb_i                 = outg_rec_nb_i, 
                    preex_i                       = preex_i, 
                    opco                          = self.opco, 
                    outg_rec_nb_to_files_dict     = outg_rec_nb_to_files_dict, 
                    outg_rec_nb_to_files_ede_dict = outg_rec_nb_to_files_ede_dict, 
                    dovs_df                       = dovs_df, 
                    base_dirs                     = self.base_dirs, 
                    tmp_summary_subdirs           = tmp_summary_subdirs, 
                    tmp_pdf_subdirs               = tmp_pdf_subdirs, 
                    tmp_warnings_subdir           = tmp_warnings_subdir, 
                    build_summary_dfs             = build_summary_dfs, 
                    perform_plotting              = perform_plotting, 
                    #-----
                    run_outg_inclusion_assessment = run_outg_inclusion_assessment, 
                    max_pct_PNs_missing_allowed   = max_pct_PNs_missing_allowed, 
                    n_PNs_w_power_threshold       = n_PNs_w_power_threshold, 
                    include_suboutg_endpt_plots   = include_suboutg_endpt_plots, 
                    fig_num                       = fig_num, 
                    #-----
                    **self.dovs_audit_args
                )
                #-------------------------
                audit_i = res_dict_i['audit_i']
                #-----
                if preex_i != 'tmp_result':
                    outg_rec_nb_i = audit_i.outg_rec_nb
                else:
                    outg_rec_nb_i = audit_i
                #-----
                if res_dict_i['run_result'] != 'Pass':
                    outgs_fail.append((outg_rec_nb_i, res_dict_i['run_result']))
                else:
                    #-------------------------
                    if build_summary_dfs:
                        if res_dict_i['detailed_summary_df_i'].shape[0]>0:
                            all_detailed_summary_dfs.append(res_dict_i['detailed_summary_df_i'])
                        #-----
                        if res_dict_i['detailed_summary_df_dovs_beg_i'].shape[0]>0:
                            all_detailed_summary_dfs_dovs_beg.append(res_dict_i['detailed_summary_df_dovs_beg_i'])
                        #-------------------------
                        warnings_text += res_dict_i['warnings_text_i']
                        #-------------------------
                        ci_cmi_summary_df_i = res_dict_i['ci_cmi_summary_df_i']
                        ci_cmi_summary_df_i.index = [len(ci_cmi_summary_dfs)]
                        ci_cmi_summary_dfs.append(ci_cmi_summary_df_i)
                    #-------------------------
                    outgs_pass.append(outg_rec_nb_i)
                fig_num = res_dict_i['fig_num']
        
            except:
                outgs_fail.append((outg_rec_nb_i, "Unknown"))
                if debug:
                    raise
    
    
        #----------------------------------------------------------------------------------------------------
        # Below, delete_tmp_subdir(s) set to False in functions because, instead, all will be deleted once all successfully complete
        #-------------------------
        DABatchOPCO.merge_all_warnings(
            save_dir            = self.base_dirs['save_dir'], 
            warning_path        = None, 
            tmp_warnings_subdir = tmp_warnings_subdir, 
            delete_tmp_subdir   = False
        )
        #--------------------------------------------------
        if build_summary_dfs:
            DABatchOPCO.merge_all_summary_tmp_subdirs(
                save_dir            = self.base_dirs['save_dir'], 
                summary_paths       = summary_paths, 
                tmp_summary_subdirs = tmp_summary_subdirs, 
                sort_how            = 'abs_delta_ci_cmi', 
                index_col           = ['OUTAGE_NB', 'OUTG_REC_NB'], 
                delete_tmp_subdirs  = False
            )
        
        #--------------------------------------------------
        if perform_plotting:
            DABatchOPCO.merge_all_pdf_tmp_subdirs(
                save_dir           = self.base_dirs['save_dir'], 
                pdf_paths          = pdf_paths, 
                tmp_pdf_subdirs    = tmp_pdf_subdirs, 
                delete_tmp_subdirs = False
            )

        #--------------------------------------------------
        # Delete temporary subdirs
        Utilities.del_tmp_save_dir(
            base_dir_path = self.base_dirs['save_dir'],
            tmp_dir_name  = list(tmp_pdf_subdirs.values())
        )
        #-----
        Utilities.del_tmp_save_dir(
            base_dir_path = self.base_dirs['save_dir'],
            tmp_dir_name  = list(tmp_summary_subdirs.values())
        )
        #-----
        Utilities.del_tmp_save_dir(
            base_dir_path = self.base_dirs['save_dir'],
            tmp_dir_name  = tmp_warnings_subdir
        )
        
        #--------------------------------------------------
        if verbose:
            print(f"#OUTG_REC_NBs = {len(outg_rec_nbs)}")
            print(f"\tpass: {len(outgs_pass)}")
            print(f"\tfail: {len(outgs_fail)}")
        #--------------------------------------------------
        return dict(
            outg_rec_nbs = list(outg_rec_nbs.keys()), 
            outgs_pass   = outgs_pass, 
            outgs_fail   = outgs_fail, 
            fig_num      = fig_num
        )


#-----------------------------------------------------------------------------------------------------------------------------
#*****************************************************************************************************************************
#-----------------------------------------------------------------------------------------------------------------------------
class DABatch:
    r"""
    Class to adjust DOVS outage times and CI/CMI
    """
    def __init__(
        self                   , 
        date_0                 , 
        date_1                 , 
        opcos                  , # e.g., ['oh'] or {'oh':[]}
        #--------------------
        save_dir_base          , 
        dates_subdir_appndx    = None, 
        #--------------------
        CI_NB_min              = None, # e.g., 15
        mjr_mnr_cause          = None, 
        use_sql_std_outage     = True, 
        addtnl_outg_sql_kwargs = None, 
        #--------------------
        daq_search_time_window = pd.Timedelta('24 hours'), 
        outg_rec_nbs           = None, 
        #--------------------
        # DOVSAudit arguments (if not None, it should be a dict with possible keys contained in dflt_dovsaudit_args below)
        dovs_audit_args        = None, 
        #--------------------
        init_dfs_to            = pd.DataFrame(),  # Should be pd.DataFrame() or None
        verbose_init           = True, 
    ):
        r"""
        date_0/date_1:
            The date range (inclusive) for which results will be compiled.
            Since the range is inclusive, date_1-date_0=6 for a single week.
            e.g., date_0='2025-01-12' and date_1='2025-01-18' is for the week of Sunday 12 January 2025
                through Saturday 18 January 2025

        opcos:
            A list or a dict object
            IF LIST:
                Simple list of opcos to be included (see accptbl_opcos in __init__ below for acceptable inputs)
                e.g., opcos = ['ap', 'im', 'oh']
            IF DICT:
                This method should be used if the user wants only specific states from the various opcos
                keys:   acceptable opcos (see accptbl_opcos in __init__ below for acceptable inputs)
                values: states to be included for each opco, or None to include all.
                e.g., opcos = {'ap':['TN', 'VA'], 'im':None, 'oh':['OH']}
            self.opcos will be standardized to be a dict object in either case

        save_dir_base:
            The directory where all DOVSAudit results are housed.
            If save_dir_base is None ==> os.path.join(Utilities.get_local_data_dir(), 'dovs_check') will be used
                For me, this directory is r'C:\Users\s346557\Documents\LocalData\dovs_check'
            If, e.g., date_0='2025-01-12', date_1='2025-01-18', dates_subdir_appndx=None the results for opco='ap' will be found in
                r'C:\Users\s346557\Documents\LocalData\dovs_check\20250112_20250118\ap\'

        dates_subdir_appndx:
            This can alter the dates subdirectory where the results will be saved.
            If, e.g., date_0='2025-01-12', date_1='2025-01-18', dates_subdir_appndx=None the results for opco='ap' will be found in
                r'C:\Users\s346557\Documents\LocalData\dovs_check\20250112_20250118\ap\'
            If, e.g., dates_subdir_appndx = '_blah', this will be altered to 
                r'C:\Users\s346557\Documents\LocalData\dovs_check\20250112_20250118_blah\ap\'

        dovs_audit_args:
            If None, dflt_dovsaudit_args (defined below) will be used.
            Otherwise, must be a dict object with keys taken from dflt_dovsaudit_args.
            e.g., if one wants combine_by_PN_likeness_thresh = pd.Timedelta('5 minutes') but all other default values,
              simple input dovs_audit_args = dict(combine_by_PN_likeness_thresh = pd.Timedelta('5 minutes'))
            NOTE: If the user inputs daq_search_time_window into dovs_audit_args, this will not throw any error (because 
                  this is an acceptable argument for DOVSAudit).  HOWEVER, the value will be replaced by the daq_search_time_window
                  value explicitly input into the constructor.
        """
        #----------------------------------------------------------------------------------------------------
        # When using self.init_dfs_to ALWAYS use copy.deepcopy(self.init_dfs_to),
        # otherwise, unintended consequences could return (self.init_dfs_to is None it
        # doesn't really matter, but when it is pd.DataFrame it does!)
        self.init_dfs_to = init_dfs_to

        #----------------------------------------------------------------------------------------------------
        self.date_0 = date_0
        self.date_1 = date_1
        #-------------------------
        assert(Utilities.is_object_one_of_types(opcos, [list, dict]))
        accptbl_opcos = ['ap', 'ky', 'oh', 'im', 'pso', 'swp', 'tx']
        self.opcos = DABatch.init_opcos(
            opcos         = opcos, 
            accptbl_opcos = accptbl_opcos
        )
        #-------------------------
        if save_dir_base is None:
            self.save_dir_base = os.path.join(Utilities.get_local_data_dir(), 'dovs_check')
        else:
            self.save_dir_base = save_dir_base
        #-----
        if not os.path.isdir(self.save_dir_base):
            os.makedirs(self.save_dir_base)
        #-----
        self.dates_subdir_appndx         = dates_subdir_appndx
        #-------------------------
        self.CI_NB_min                   = CI_NB_min
        self.mjr_mnr_cause               = mjr_mnr_cause
        self.use_sql_std_outage          = use_sql_std_outage
        #--------------------
        self.addtnl_outg_sql_kwargs      = addtnl_outg_sql_kwargs
        #-----
        if self.addtnl_outg_sql_kwargs is None:
            self.addtnl_outg_sql_kwargs = dict()
        #-----
        if not self.use_sql_std_outage:
            self.addtnl_outg_sql_kwargs['include_DOVS_PREMISE_DIM']     = True
            self.addtnl_outg_sql_kwargs['select_cols_DOVS_PREMISE_DIM'] = list(set(self.addtnl_outg_sql_kwargs.get('select_cols_DOVS_PREMISE_DIM', [])).union(set(['OFF_TM', 'REST_TM', 'PREMISE_NB'])))
        #--------------------
        self.daq_search_time_window      = daq_search_time_window
        self.outg_rec_nbs                = outg_rec_nbs
        #-------------------------
        # DOVSAudit arguments
        dflt_dovsaudit_args = dict(
            calculate_by_PN                   = True, 
            combine_by_PN_likeness_thresh     = pd.Timedelta('15 minutes'), 
            expand_outg_search_time_tight     = pd.Timedelta('1 hours'), 
            expand_outg_search_time_loose     = pd.Timedelta('12 hours'), 
            use_est_outg_times                = False, 
            use_full_ede_outgs                = False, 
            daq_search_time_window            = self.daq_search_time_window, 
            overlaps_addtnl_dovs_sql_kwargs   = dict(
                CI_NB_min  = 0, 
                CMI_NB_min = 0
            ), 
        )
        #-----
        if dovs_audit_args is None:
            dovs_audit_args = dflt_dovsaudit_args
        assert(
            isinstance(dovs_audit_args, dict) and
            set(dovs_audit_args.keys()).difference(set(dflt_dovsaudit_args.keys())) == set()
        )
        #-----
        dovs_audit_args = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = dovs_audit_args, 
            default_values_dict = dflt_dovsaudit_args, 
            extend_any_lists    = True, 
            inplace             = False
        )
        #-----
        # Set daq_search_time_window attribute to agree with that input into DABatchOPCO init
        dovs_audit_args['daq_search_time_window'] = self.daq_search_time_window
        self.dovs_audit_args                      = copy.deepcopy(dovs_audit_args)

        #-------------------------
        if verbose_init:
            print('Initializing DABatch object')
            print(f'date_0 = {self.date_0}')
            print(f'date_1 = {self.date_1}')
            print('opcos/states (state=None means include all in opco)')
            Utilities.print_dict_align_keys(
                dct        = self.opcos, 
                left_align = True
            )
            tmp_dct = dict(
                save_dir_base               = self.save_dir_base, 
                dates_subdir_appndx         = self.dates_subdir_appndx, 
                CI_NB_min                   = self.CI_NB_min, 
                mjr_mnr_cause               = self.mjr_mnr_cause, 
                daq_search_time_window      = self.daq_search_time_window, 
                outg_rec_nbs                = self.outg_rec_nbs
            )
            Utilities.print_dict_align_keys(
                dct        = tmp_dct, 
                left_align = True
            )
            Utilities.print_dict_align_keys(
                dct        = self.dovs_audit_args, 
                left_align = True
            )


    #--------------------------------------------------
    @staticmethod
    def init_opcos(
        opcos         , 
        accptbl_opcos = ['ap', 'ky', 'oh', 'im', 'pso', 'swp', 'tx']
    ):
        r"""
        As described in the DABatch class constructor, opcos input can be a list or a dict object
        IF LIST:
            Simple list of opcos to be included (see accptbl_opcos in __init__ below for acceptable inputs)
            e.g., opcos = ['ap', 'im', 'oh']
        IF DICT:
            This method should be used if the user wants only specific states from the various opcos
            keys:   acceptable opcos (see accptbl_opcos in __init__ below for acceptable inputs)
            values: states to be included for each opco, or None to include all.
            e.g., opcos = {'ap':['TN', 'VA'], 'im':None, 'oh':'OH'}
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(opcos, [list, dict]))
        #-------------------------
        if isinstance(opcos, list):
            opcos = {k:None for k in opcos}
        #-------------------------
        # Make sure all keys of opcos look good
        assert(set(opcos.keys()).difference(accptbl_opcos)==set())
        #-------------------------
        # Make sure all values of opcos look good
        for opco_i,states_i in opcos.items():
            assert(states_i is None or Utilities.is_object_one_of_types(states_i, [list, str]))
            #-----
            if isinstance(states_i, list):
                assert(Utilities.are_all_list_elements_of_type(lst=states_i, typ=str))
        #-------------------------
        return opcos


    #--------------------------------------------------
    def run_batch_daq(
        self, 
        load_prereqs_if_exist = True, 
        batch_size_ami        = 25, 
        n_update_ami          = 1, 
        batch_size_ede        = 25, 
        n_update_ede          = 1, 
        pdpu_only_ede         = True, 
        verbose               = True
    ):
        r"""
        """
        #--------------------------------------------------
        assert(
            self.save_dir_base is not None and 
            os.path.isdir(self.save_dir_base)
        )
        #-------------------------
        conn_aws  = Utilities.get_athena_prod_aws_connection()
        conn_dovs = Utilities.get_utldb01p_oracle_connection()
        #-------------------------
        for opco_i,states_i in self.opcos.items():
            if verbose:
                print('-----'*15)
                print(f'OPCO = {opco_i}')
                print('-----'*5)
            #-------------------------
            da_batch_i = DABatchOPCO(
                date_0                      = self.date_0, 
                date_1                      = self.date_1, 
                opco                        = opco_i,
                save_dir_base               = self.save_dir_base, 
                dates_subdir_appndx         = self.dates_subdir_appndx, 
                save_results                = True, 
                states                      = states_i, 
                CI_NB_min                   = self.CI_NB_min, 
                mjr_mnr_cause               = self.mjr_mnr_cause, 
                use_sql_std_outage          = self.use_sql_std_outage, 
                addtnl_outg_sql_kwargs      = self.addtnl_outg_sql_kwargs, 
                daq_search_time_window      = self.daq_search_time_window, 
                outg_rec_nbs                = self.outg_rec_nbs, 
                dovs_audit_args             = self.dovs_audit_args, 
                init_dfs_to                 = self.init_dfs_to, 
                conn_aws                    = conn_aws,
                conn_dovs                   = conn_dovs
            )
            #-------------------------
            da_batch_i.run_batch_daq(
                load_prereqs_if_exist = load_prereqs_if_exist, 
                batch_size_ami        = batch_size_ami, 
                n_update_ami          = n_update_ami, 
                batch_size_ede        = batch_size_ede, 
                n_update_ede          = n_update_ede, 
                pdpu_only_ede         = pdpu_only_ede, 
                verbose               = verbose
            )


    #----------------------------------------------------------------------------------------------------
    #--------------------------------------------------
    @staticmethod
    def print_opco_merge_info(
        date_0          , 
        date_1          , 
        fnames_to_merge , 
        base_dir        , 
        output_path     = None, 
        opcos_w_results = None, 
        pretext         = None, 
    ):
        r"""
        """
        #--------------------------------------------------
        print('-----'*15)
        if pretext is not None:
            print(pretext)
        print(f'date_0          = {date_0}')
        print(f'date_1          = {date_1}')
        print(f'fnames_to_merge = {fnames_to_merge}')
        print(f'base_dir        = {base_dir}')
        if output_path is not None:
            print(f'output_path     = {output_path}')
        if opcos_w_results is not None:
            print(f'opcos_w_results = {opcos_w_results}')
        print('\n')


    #--------------------------------------------------
    @staticmethod
    def get_opcos_with_results(
        base_dir       , 
        opcos          = None, 
        results_subdir = 'Results', 
        fname          = None, 
    ):
        r"""
        This is a helper function used to identify subdirectories containing results
        
        base_dir:
            The directory where the results subdirectories for the various OPCOs are found
    
        opcos:
            The list of OPCO results to be included in the final xlsx file.
            The OPCO names should coincide with directories found in base_dir
            Only those with results_subdir subdirectories will be included.
            If opcos is None, simply use all subdirectories found in base_dir
              WITH results_subdir subdirectories will be used
    
        results_subdir:
            The subdirectory name, for each OPCO, where the results live.
            e.g., for Ohio, os.path.join(base_dir, 'oh', 'Results') should house the results.
        """
        #--------------------------------------------------
        assert(os.path.isdir(base_dir))
        #-------------------------
        if opcos is None:
            opcos = Utilities.get_immediate_sub_dir_names(base_dir = base_dir)
        # Make sure all OPCO subdirectories are found
        assert(np.all([os.path.isdir(os.path.join(base_dir, x)) for x in opcos]))
        #-----
        # Include only those with results_subdir subdirectory
        opcos = [x for x in opcos if os.path.isdir(os.path.join(base_dir, x, results_subdir))]
        #-----
        # If fname is not None, include only those with fname file in results_subdir subdirectory
        if fname is not None:
            opcos = [x for x in opcos if os.path.exists(os.path.join(base_dir, x, results_subdir, fname))]
        #-------------------------
        return opcos
    
    #--------------------------------------------------
    @staticmethod
    def merge_opco_summary_files(
        date_0               , 
        date_1               , 
        fnames_to_merge      , 
        dovs_check_base_dir  = None, 
        dates_subdir_appndx  = None, 
        opcos                = None, 
        results_subdir       = 'Results', 
        reset_indices        = True, 
        output_fname         = None, 
        output_dates_in_name = True, 
        output_subdir        = 'AllOPCOs', 
        verbose              = True
    ):
        r"""
        Data are typically collected/analyzed by OPCO.  However, the final deliverable should contain all.
        The purpose of this function is to combine the different OPCO results into a single xlsx file.
    
        date_0 & date_1:
            These define the period over which the analysis is run.
            Typically, the analysis is run for a single week (with a lag of two weeks)
                e.g, on 2025-02-04, I run the analysis for 
                    date_0 = 2025-01-19
                    date_1 = 2025-01-25
    
        fnames_to_merge:
            The file names to merge.  These names must include the extension, and must be of type pkl or csv.
            Typically, these will be 'detailed_summary.pkl' or 'detailed_summary_dovs_beg.pkl'
        
        dovs_check_base_dir:
            The directory where all DOVSAudit results are housed.
            For me, this directory is r'C:\Users\s346557\Documents\LocalData\dovs_check'
            If dovs_check_base_dir is None:
                ==> os.path.join(Utilities.get_local_data_dir(), 'dovs_check') will be used
    
        opcos:
            The list of OPCO results to be included in the final xlsx file.
            The OPCO names should coincide with directories found in os.path.join(dovs_check_base_dir, dates_subdir)
            Only those with results_subdir subdirectories containing fnames files will be included.
            If opcos is None, simply use all subdirectories found in os.path.join(dovs_check_base_dir, dates_subdir) 
              WITH results_subdir subdirectories containing fnames files will be used
    
        results_subdir:
            The subdirectory name, for each OPCO, where the results live.
            e.g., for Ohio, os.path.join(dovs_check_base_dir, dates_subdir, 'oh', 'Results') should house the results.
    
        reset_indices:
            If true, .reset_index() will be called in DFs before writing to output file
    
        output_fname:
            If None, fnames_to_merge (with appropriate xlsx file extension) will be used
            If not None, should either not contain a file extension or contain the appropriate .xlsx file extension
            
        output_dates_in_name:
            If True, date0_date1 will be tacked on to the output file name
    
        output_subdir:
            The subdirectory where the final results will be output.
            This will be at the same level as the OPCOs
        """
        #--------------------------------------------------
        # Find results to merge into final file
        #-------------------------
        if dovs_check_base_dir is None:
            dovs_check_base_dir = os.path.join(Utilities.get_local_data_dir(), 'dovs_check')
        assert(os.path.isdir(dovs_check_base_dir))
        #-------------------------
        dates_subdir = pd.to_datetime(date_0).strftime('%Y%m%d') + '_' + pd.to_datetime(date_1).strftime('%Y%m%d')
        if dates_subdir_appndx is not None:
            dates_subdir += dates_subdir_appndx
        #-----
        base_dir     = os.path.join(dovs_check_base_dir, dates_subdir)
        #-------------------------
        # Get fnames_to_merge extension and ensure it is of correct type
        f_ext = Utilities.find_file_extension(
            file_path_or_name = fnames_to_merge, 
            include_pd        = False, 
            assert_found      = True
        )
        assert(f_ext in ['pkl', 'csv'])
        #-------------------------
        opcos_w_results = DABatch.get_opcos_with_results(
            base_dir       = base_dir, 
            opcos          = opcos, 
            results_subdir = results_subdir, 
            fname          = fnames_to_merge
        )
        # Make sure output_subdir never accidentally gets included (e.g., if results present there already)
        opcos_w_results = [x for x in opcos_w_results if x != output_subdir]
        #--------------------------------------------------
        if len(opcos_w_results)==0:
            if verbose:
                pretext = "In DABatch.merge_opco_summary_files"
                pretext += '\nNO ACCEPTABLE FILES FOUND, NOT OUTPUT CREATED'
                #-----
                DABatch.print_opco_merge_info(
                    date_0          = date_0, 
                    date_1          = date_1, 
                    fnames_to_merge = fnames_to_merge, 
                    base_dir        = base_dir, 
                    output_path     = None, 
                    opcos_w_results = None, 
                    pretext         = pretext, 
                )
            return
        
        #--------------------------------------------------
        # Define output location and make sure it exists
        #-------------------------
        output_dir = os.path.join(base_dir, output_subdir, results_subdir)
        #-------------------------
        if output_fname is None:
            output_fname = fnames_to_merge.replace(f'.{f_ext}', '.xlsx')
        else:
            out_f_ext = Utilities.find_file_extension(
                file_path_or_name = output_fname, 
                include_pd        = False, 
                assert_found      = False
            )
            #-----
            if out_f_ext is not None:
                assert(out_f_ext=='xlsx')
            else:
                output_fname = output_fname+".xlsx"
        #-------------------------
        if output_dates_in_name:
            output_fname = Utilities.append_to_path(
                save_path                     = output_fname, 
                appendix                      = '_'+dates_subdir, 
                ext_to_find                   = '.xlsx', 
                append_to_end_if_ext_no_found = False
            )
        #-------------------------
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_fname)
        
        #--------------------------------------------------
        # Write opcos to final file
        #-------------------------
        writer = pd.ExcelWriter(output_path)
        #-------------------------
        for opco_i in opcos_w_results:
            path_i = os.path.join(base_dir, opco_i, results_subdir, fnames_to_merge)
            #-------------------------
            if f_ext == 'pkl':
                df_i = pd.read_pickle(path_i)
            elif f_ext == 'csv':
                df_i = pd.read_csv(path_i)
            else:
                assert(0)
            #-------------------------
            if reset_indices:
                df_i = df_i.reset_index()
            #-------------------------
            # Mico's team prefers outg_i_beg/_end columns to have "%m/%d/%Y %H:%M" format instead of "%Y-%m-%d %H:%M:%S"
            if 'outg_i_beg' in df_i.columns:
                df_i['outg_i_beg'] = pd.to_datetime(df_i['outg_i_beg']).dt.strftime("%m/%d/%Y %H:%M")
            if 'outg_i_end' in df_i.columns:
                df_i['outg_i_end'] = pd.to_datetime(df_i['outg_i_end']).dt.strftime("%m/%d/%Y %H:%M")
            #-------------------------
            df_i.to_excel(writer, sheet_name = opco_i)
        #-------------------------
        writer.close()
        #-------------------------
        if verbose:
            pretext = "In DABatch.merge_opco_summary_files"
            DABatch.print_opco_merge_info(
                date_0          = date_0, 
                date_1          = date_1, 
                fnames_to_merge = fnames_to_merge, 
                base_dir        = base_dir, 
                output_path     = output_path, 
                opcos_w_results = opcos_w_results, 
                pretext         = pretext, 
            )


    #--------------------------------------------------
    @staticmethod  
    def merge_opco_results_pdfs(
        date_0               , 
        date_1               , 
        fnames_to_merge      , 
        dovs_check_base_dir  = None, 
        dates_subdir_appndx  = None, 
        opcos                = None, 
        results_subdir       = 'Results', 
        output_fname         = None, 
        output_dates_in_name = True, 
        output_subdir        = 'AllOPCOs', 
        verbose              = True
    ):
        r"""
        Data are typically collected/analyzed by OPCO.  However, the final deliverable should contain all.
        The purpose of this function is to combine the different OPCO results into a single xlsx file.
    
        date_0 & date_1:
            These define the period over which the analysis is run.
            Typically, the analysis is run for a single week (with a lag of two weeks)
                e.g, on 2025-02-04, I run the analysis for 
                    date_0 = 2025-01-19
                    date_1 = 2025-01-25
    
        fnames_to_merge:
            The file names to merge.  These names should either include no file extension or the appropriate .pdf exetension
        
        dovs_check_base_dir:
            The directory where all DOVSAudit results are housed.
            For me, this directory is r'C:\Users\s346557\Documents\LocalData\dovs_check'
            If dovs_check_base_dir is None:
                ==> os.path.join(Utilities.get_local_data_dir(), 'dovs_check') will be used
    
        opcos:
            The list of OPCO results to be included in the final xlsx file.
            The OPCO names should coincide with directories found in os.path.join(dovs_check_base_dir, dates_subdir)
            Only those with results_subdir subdirectories containing fnames files will be included.
            If opcos is None, simply use all subdirectories found in os.path.join(dovs_check_base_dir, dates_subdir) 
              WITH results_subdir subdirectories containing fnames files will be used
    
        results_subdir:
            The subdirectory name, for each OPCO, where the results live.
            e.g., for Ohio, os.path.join(dovs_check_base_dir, dates_subdir, 'oh', 'Results') should house the results.
    
        output_fname:
            If None, fnames_to_merge (with appropriate xlsx file extension) will be used
            If not None, should either not contain a file extension or contain the appropriate .pdf file extension
            
        output_dates_in_name:
            If True, date0_date1 will be tacked on to the output file name
    
        output_subdir:
            The subdirectory where the final results will be output.
            This will be at the same level as the OPCOs
        """
        #--------------------------------------------------
        # Find results to merge into final file
        #-------------------------
        if dovs_check_base_dir is None:
            dovs_check_base_dir = os.path.join(Utilities.get_local_data_dir(), 'dovs_check')
        assert(os.path.isdir(dovs_check_base_dir))
        #-------------------------
        dates_subdir = pd.to_datetime(date_0).strftime('%Y%m%d') + '_' + pd.to_datetime(date_1).strftime('%Y%m%d')
        if dates_subdir_appndx is not None:
            dates_subdir += dates_subdir_appndx
        #-----
        base_dir     = os.path.join(dovs_check_base_dir, dates_subdir)
        #-------------------------
        # Get fnames_to_merge extension and ensure it is of correct type
        f_ext = Utilities.find_file_extension(
            file_path_or_name = fnames_to_merge, 
            include_pd        = False, 
            assert_found      = False
        )
        if f_ext is not None:
            assert(f_ext=='pdf')
        else:
            fnames_to_merge = fnames_to_merge+".pdf"
    
        #-------------------------
        opcos_w_results = DABatch.get_opcos_with_results(
            base_dir       = base_dir, 
            opcos          = opcos, 
            results_subdir = results_subdir, 
            fname          = fnames_to_merge
        )
        # Make sure output_subdir never accidentally gets included (e.g., if results present there already)
        opcos_w_results = [x for x in opcos_w_results if x != output_subdir]
        #--------------------------------------------------
        if len(opcos_w_results)==0:
            if verbose:
                pretext = "In DABatch.merge_opco_results_pdfs"
                pretext += '\nNO ACCEPTABLE FILES FOUND, NOT OUTPUT CREATED'
                #-----
                DABatch.print_opco_merge_info(
                    date_0          = date_0, 
                    date_1          = date_1, 
                    fnames_to_merge = fnames_to_merge, 
                    base_dir        = base_dir, 
                    output_path     = None, 
                    opcos_w_results = None, 
                    pretext         = pretext, 
                )
            return
        
        #--------------------------------------------------
        # Define output location and make sure it exists
        #-------------------------
        output_dir = os.path.join(base_dir, output_subdir, results_subdir)
        #-------------------------
        if output_fname is None:
            output_fname = fnames_to_merge
        else:
            out_f_ext = Utilities.find_file_extension(
                file_path_or_name = output_fname, 
                include_pd        = False, 
                assert_found      = False
            )
            #-----
            if out_f_ext is not None:
                assert(out_f_ext=='pdf')
            else:
                output_fname = output_fname+".pdf"
        #-------------------------
        if output_dates_in_name:
            output_fname = Utilities.append_to_path(
                save_path                     = output_fname, 
                appendix                      = '_'+dates_subdir, 
                ext_to_find                   = '.pdf', 
                append_to_end_if_ext_no_found = False
            )
        #-------------------------
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_fname)
        
        #--------------------------------------------------
        # Write opcos to final file
        #-------------------------
        files_to_merge = []
        #-------------------------
        for opco_i in opcos_w_results:
            path_i = os.path.join(base_dir, opco_i, results_subdir, fnames_to_merge)
            assert(os.path.exists(path_i)) # redundant
            files_to_merge.append(path_i)
        #-------------------------
        PDFMerger.merge_list(
            files_to_merge = files_to_merge, 
            output_path    = output_path
        )
        #-------------------------
        if verbose:
            pretext = "In DABatch.merge_opco_results_pdfs"
            DABatch.print_opco_merge_info(
                date_0          = date_0, 
                date_1          = date_1, 
                fnames_to_merge = fnames_to_merge, 
                base_dir        = base_dir, 
                output_path     = output_path, 
                opcos_w_results = opcos_w_results, 
                pretext         = pretext, 
            )


    #--------------------------------------------------
    @staticmethod
    def merge_opco_results(
        date_0               , 
        date_1               , 
        fnames_summary_files = [
            'detailed_summary.pkl', 
            'detailed_summary_dovs_beg.pkl', 
            'ci_cmi_summary.pkl', 
        ], 
        fnames_pdf           = [
            'Results.pdf', 
            'Results_dovs_beg.pdf', 
            'Results_w_suboutg_endpt_plots.pdf'
        ], 
        dovs_check_base_dir  = None, 
        dates_subdir_appndx  = None, 
        opcos                = None, 
        results_subdir       = 'Results', 
        reset_indices        = True, 
        output_dates_in_name = True, 
        output_subdir        = 'AllOPCOs', 
        verbose              = True
    ):
        r"""
        """
        #--------------------------------------------------
        for fname_i in fnames_summary_files:
            DABatch.merge_opco_summary_files(
                date_0               = date_0, 
                date_1               = date_1, 
                fnames_to_merge      = fname_i, 
                dovs_check_base_dir  = dovs_check_base_dir, 
                dates_subdir_appndx  = dates_subdir_appndx, 
                opcos                = opcos, 
                results_subdir       = results_subdir, 
                reset_indices        = reset_indices, 
                output_fname         = None, 
                output_dates_in_name = output_dates_in_name, 
                output_subdir        = output_subdir, 
                verbose              = verbose
            )
        #-------------------------
        for fname_i in fnames_pdf:
            DABatch.merge_opco_results_pdfs(
                date_0               = date_0, 
                date_1               = date_1, 
                fnames_to_merge      = fname_i, 
                dovs_check_base_dir  = dovs_check_base_dir, 
                dates_subdir_appndx  = dates_subdir_appndx, 
                opcos                = opcos, 
                results_subdir       = results_subdir, 
                output_fname         = None, 
                output_dates_in_name = output_dates_in_name, 
                output_subdir        = output_subdir, 
                verbose              = verbose
            )

    #--------------------------------------------------
    @staticmethod
    def found_final_results_for_opco(
        opco                , 
        date_0              , 
        date_1              , 
        save_dir_base       = None, 
        dates_subdir_appndx = None, 
        build_summary_dfs   = True, 
        perform_plotting    = True, 
        return_found_dict   = False, 
    ):
        r"""
        """
        #--------------------------------------------------
        found_dict = {}
        #--------------------------------------------------
        base_dirs = DABatchOPCO.build_base_dirs(
            date_0              = date_0, 
            date_1              = date_1, 
            opco                = opco, 
            save_dir_base       = save_dir_base, 
            dates_subdir_appndx = dates_subdir_appndx, 
            assert_base_exists  = False, 
            assert_amiede_exist = False, 
        )
        results_dir = base_dirs['save_dir']
        #-----
        if os.path.isdir(results_dir):
            found_dict['results_dir'] = results_dir
        else:
            found_dict['results_dir'] = None
        #--------------------------------------------------
        summaries = [
            'ci_cmi_summary', 
            'detailed_summary', 
            'detailed_summary_dovs_beg', 
        ]
        pdfs = [
            'Results', 
            'Results_dovs_beg', 
            'Results_w_suboutg_endpt_plots', 
        ]
        txts = [
            'warnings'
        ]
        #--------------------------------------------------
        if build_summary_dfs:
            for name in summaries:
                assert(f'{name}.csv' not in found_dict.keys())
                assert(f'{name}.pkl' not in found_dict.keys())
                #-----
                if os.path.exists(os.path.join(results_dir, f'{name}.csv')):
                    found_dict[f'{name}.csv'] = os.path.join(results_dir, f'{name}.csv')
                else:
                    found_dict[f'{name}.csv'] = None
                #-----
                if os.path.exists(os.path.join(results_dir, f'{name}.pkl')):
                    found_dict[f'{name}.pkl'] = os.path.join(results_dir, f'{name}.pkl')
                else:
                    found_dict[f'{name}.pkl'] = None
        #-------------------------
        if perform_plotting:
            for name in pdfs:
                assert(f'{name}.pdf' not in found_dict.keys())
                if os.path.exists(os.path.join(results_dir, f'{name}.pdf')):
                    found_dict[f'{name}.pdf'] = os.path.join(results_dir, f'{name}.pdf')
                else:
                    found_dict[f'{name}.pdf'] = None
        #-------------------------
        for name in txts:
            assert(f'{name}.txt' not in found_dict.keys())
            if os.path.exists(os.path.join(results_dir, f'{name}.txt')):
                found_dict[f'{name}.txt'] = os.path.join(results_dir, f'{name}.txt')
            else:
                found_dict[f'{name}.txt'] = None
        #--------------------------------------------------
        all_found = np.all([x is not None for x in found_dict.values()])
        if return_found_dict:
            return all_found, found_dict
        else:
            return all_found

 
    #--------------------------------------------------
    def analyze_audits_for_opcos(
        self                               , 
        load_prereqs_if_exist              = True, 
        reanalyze_preex_results            = False, 
        perform_plotting                   = True, 
        build_summary_dfs                  = True, 
        #-----
        merge_outputs                      = True, 
        output_subdir                      = 'AllOPCOs', 
        #-----
        run_outg_inclusion_assessment      = True, 
        max_pct_PNs_missing_allowed        = 0, 
        n_PNs_w_power_threshold            = 95, 
        include_suboutg_endpt_plots        = True, 
        fig_num                            = 0, 
        #-----
        debug                              = False, 
        verbose                            = True, 
    ):
        r"""
        debug:
            Using the try/except block is great for smoothing out potential unexpected issues.
            HOWEVER, imbedding the code in a try/except block makes it difficult to track down a bug, 
              as the except clause will be thrown UNLESS debug==True
            So, if the results aren't as expected (e.g., the program seems to run fine, but generates no or not many results), 
              the user should set debug = True
        """
        #--------------------------------------------------
        assert(
            self.save_dir_base is not None and 
            os.path.isdir(self.save_dir_base)
        )
        #-------------------------
        conn_aws  = Utilities.get_athena_prod_aws_connection()
        conn_dovs = Utilities.get_utldb01p_oracle_connection()
        #-------------------------
        results = dict()
        for opco_i,states_i in self.opcos.items():
            if verbose:
                print('-----'*15)
                print(f'OPCO = {opco_i}')
                print('-----'*5)    
            #-------------------------
            all_found, found_dict = DABatch.found_final_results_for_opco(
                opco                = opco_i, 
                date_0              = self.date_0, 
                date_1              = self.date_1, 
                save_dir_base       = self.save_dir_base, 
                dates_subdir_appndx = self.dates_subdir_appndx, 
                build_summary_dfs   = build_summary_dfs, 
                perform_plotting    = perform_plotting, 
                return_found_dict   = True, 
            )
            if all_found and not reanalyze_preex_results:
                if verbose:
                    print("Skipping analysis (but results will still be merged into final if merge_outputs==True)")
                    print("\tas reanalyze_preex_results==True and previous results found in")
                    print(f"\tdirectory: {found_dict['results_dir']}\n\n")
                continue
            #-------------------------
            dabatch_opco_i = DABatchOPCO(
                date_0                      = self.date_0, 
                date_1                      = self.date_1, 
                opco                        = opco_i, 
                save_dir_base               = self.save_dir_base, 
                dates_subdir_appndx         = self.dates_subdir_appndx, 
                save_results                = True, 
                states                      = states_i, 
                CI_NB_min                   = self.CI_NB_min, 
                mjr_mnr_cause               = self.mjr_mnr_cause, 
                use_sql_std_outage          = self.use_sql_std_outage, 
                addtnl_outg_sql_kwargs      = self.addtnl_outg_sql_kwargs, 
                daq_search_time_window      = self.daq_search_time_window, 
                outg_rec_nbs                = self.outg_rec_nbs, 
                dovs_audit_args             = self.dovs_audit_args, 
                init_dfs_to                 = pd.DataFrame(),  
                conn_aws                    = conn_aws,
                conn_dovs                   = conn_dovs
            )
            #-------------------------
            results_i = dabatch_opco_i.analyze_audits_for_opco(
                load_prereqs_if_exist         = load_prereqs_if_exist, 
                reanalyze_preex_results       = reanalyze_preex_results, 
                #-----
                perform_plotting              = perform_plotting, 
                build_summary_dfs             = build_summary_dfs, 
                #-----
                run_outg_inclusion_assessment = run_outg_inclusion_assessment, 
                max_pct_PNs_missing_allowed   = max_pct_PNs_missing_allowed, 
                n_PNs_w_power_threshold       = n_PNs_w_power_threshold, 
                include_suboutg_endpt_plots   = include_suboutg_endpt_plots, 
                fig_num                       = fig_num, 
                #-----
                verbose                       = verbose, 
                debug                         = debug
            )
            fig_num = results_i['fig_num']
            assert(opco_i not in results.keys())
            results[opco_i] = results_i
            if verbose:
                print('\n\n')
        #-------------------------
        if merge_outputs:
            DABatch.merge_opco_results(
                date_0               = self.date_0, 
                date_1               = self.date_1, 
                fnames_summary_files = [
                    'detailed_summary.pkl', 
                    'detailed_summary_dovs_beg.pkl', 
                    'ci_cmi_summary.pkl', 
                ], 
                fnames_pdf           = [
                    'Results.pdf', 
                    'Results_dovs_beg.pdf', 
                    'Results_w_suboutg_endpt_plots.pdf'
                ], 
                dovs_check_base_dir  = self.save_dir_base, 
                dates_subdir_appndx  = self.dates_subdir_appndx, 
                opcos                = self.opcos.keys(), 
                results_subdir       = 'Results', 
                reset_indices        = True, 
                output_dates_in_name = True, 
                output_subdir        = output_subdir, 
                verbose              = verbose
            )
        #-------------------------
        return results
    

    #--------------------------------------------------
    def analyze_audits_for_opcos_from_csvs(
        self                               , 
        reanalyze_preex_results            = False, 
        perform_plotting                   = True, 
        build_summary_dfs                  = True, 
        #-----
        merge_outputs                      = True, 
        output_subdir                      = 'AllOPCOs', 
        #-----
        rebuild_outg_rec_nb_to_files_dicts = False, 
        #-----
        run_outg_inclusion_assessment      = True, 
        max_pct_PNs_missing_allowed        = 0, 
        n_PNs_w_power_threshold            = 95, 
        include_suboutg_endpt_plots        = True, 
        fig_num                            = 0, 
        #-----
        debug                              = False, 
        verbose                            = True, 
    ):
        r"""
        debug:
            Using the try/except block is great for smoothing out potential unexpected issues.
            HOWEVER, imbedding the code in a try/except block makes it difficult to track down a bug, 
              as the except clause will be thrown UNLESS debug==True
            So, if the results aren't as expected (e.g., the program seems to run fine, but generates no or not many results), 
              the user should set debug = True
        """
        #--------------------------------------------------
        assert(
            self.save_dir_base is not None and 
            os.path.isdir(self.save_dir_base)
        )
        #-------------------------
        conn_aws  = Utilities.get_athena_prod_aws_connection()
        conn_dovs = Utilities.get_utldb01p_oracle_connection()
        #-------------------------
        results = dict()
        for opco_i,states_i in self.opcos.items():
            if verbose:
                print('-----'*15)
                print(f'OPCO = {opco_i}')
                print('-----'*5)    
            #-------------------------
            all_found, found_dict = DABatch.found_final_results_for_opco(
                opco                = opco_i, 
                date_0              = self.date_0, 
                date_1              = self.date_1, 
                save_dir_base       = self.save_dir_base, 
                dates_subdir_appndx = self.dates_subdir_appndx, 
                build_summary_dfs   = build_summary_dfs, 
                perform_plotting    = perform_plotting, 
                return_found_dict   = True, 
            )
            if all_found and not reanalyze_preex_results:
                if verbose:
                    print("Skipping analysis (but results will still be merged into final if merge_outputs==True)")
                    print("\tas reanalyze_preex_results==True and previous results found in")
                    print(f"\tdirectory: {found_dict['results_dir']}\n\n")
                continue
            #-------------------------
            dabatch_opco_i = DABatchOPCO(
                date_0                      = self.date_0, 
                date_1                      = self.date_1, 
                opco                        = opco_i, 
                save_dir_base               = self.save_dir_base, 
                dates_subdir_appndx         = self.dates_subdir_appndx, 
                save_results                = True, 
                states                      = states_i, 
                CI_NB_min                   = None, # Not needed, only for DAQ
                mjr_mnr_cause               = None, # Not needed, only for DAQ
                use_sql_std_outage          = self.use_sql_std_outage, 
                addtnl_outg_sql_kwargs      = self.addtnl_outg_sql_kwargs, 
                daq_search_time_window      = None, # Not needed, only for DAQ
                outg_rec_nbs                = None, # Not needed, only for DAQ 
                dovs_audit_args             = self.dovs_audit_args, 
                init_dfs_to                 = pd.DataFrame(),  
                conn_aws                    = conn_aws,
                conn_dovs                   = conn_dovs
            )
            #-------------------------
            results_i = dabatch_opco_i.analyze_audits_for_opco_from_csvs(
                reanalyze_preex_results            = reanalyze_preex_results, 
                perform_plotting                   = perform_plotting, 
                build_summary_dfs                  = build_summary_dfs, 
                #-----
                rebuild_outg_rec_nb_to_files_dicts = rebuild_outg_rec_nb_to_files_dicts, 
                #-----
                run_outg_inclusion_assessment      = run_outg_inclusion_assessment, 
                max_pct_PNs_missing_allowed        = max_pct_PNs_missing_allowed, 
                n_PNs_w_power_threshold            = n_PNs_w_power_threshold, 
                include_suboutg_endpt_plots        = include_suboutg_endpt_plots, 
                fig_num                            = fig_num, 
                #-----
                verbose                            = verbose, 
                debug                              = debug
            )
            fig_num = results_i['fig_num']
            assert(opco_i not in results.keys())
            results[opco_i] = results_i
            if verbose:
                print('\n\n')
        #-------------------------
        if merge_outputs:
            DABatch.merge_opco_results(
                date_0               = self.date_0, 
                date_1               = self.date_1, 
                fnames_summary_files = [
                    'detailed_summary.pkl', 
                    'detailed_summary_dovs_beg.pkl', 
                    'ci_cmi_summary.pkl', 
                ], 
                fnames_pdf           = [
                    'Results.pdf', 
                    'Results_dovs_beg.pdf', 
                    'Results_w_suboutg_endpt_plots.pdf'
                ], 
                dovs_check_base_dir  = self.save_dir_base, 
                dates_subdir_appndx  = self.dates_subdir_appndx, 
                opcos                = self.opcos.keys(), 
                results_subdir       = 'Results', 
                reset_indices        = True, 
                output_dates_in_name = True, 
                output_subdir        = output_subdir, 
                verbose              = verbose
            )
        #-------------------------
        return results
    
    #****************************************************************************************************
    # Summary Plotting
    #****************************************************************************************************
    @staticmethod
    def build_ci_cmi_summary_df_subset_slicer(
        col     ,
        slicer  = None, 
        min_val = None,
        max_val = None    
    ):
        r"""
        Purpose of this function is essentially to make it easier to input a list of
        columns to get_ci_cmi_summary_df_subset
        """
        #-------------------------
        if slicer is None:
            slicer = DFSlicer()
        else:
            assert(isinstance(slicer, DFSlicer))
        #-------------------------
        if min_val is None and max_val is None:
            return slicer
        #-------------------------
        if min_val:
            slicer.add_single_slicer(        
                dict(
                    column              = col, 
                    value               = min_val, 
                    comparison_operator = '>='
                )
            )
        if max_val:
            slicer.add_single_slicer(        
                dict(
                    column              = col, 
                    value               = max_val, 
                    comparison_operator = '<='
                )
            )
        #-------------------------
        return slicer
    
    @staticmethod
    def get_ci_cmi_summary_df_subset(
        df            ,
        cols          ,
        min_val       = None,
        max_val       = None,
        label_int_col = None
    ):
        r"""
        cols should either be a single column in df or a list of columns in df
        """
        #-------------------------
        if min_val is None and max_val is None:
            return df
        #-------------------------
        if not isinstance(cols, list):
            assert(cols in df.columns.tolist())
            cols = [cols]
        #-----
        for col in cols:
            assert(col in df.columns.tolist())
        #-------------------------
        slicer = DFSlicer()
        for col in cols:
            slicer = DABatch.build_ci_cmi_summary_df_subset_slicer(
                col     = col,
                slicer  = slicer, 
                min_val = min_val,
                max_val = max_val    
            )
        #-------------------------
        return_df = slicer.perform_slicing(df.copy())
        #-------------------------
        if label_int_col is not None:
            return_df[label_int_col] = range(return_df.shape[0])
        #-------------------------
        return return_df
    

    @staticmethod
    def build_pct_fields(
        ci_cmi_summary_df , 
        fields_dict       = {
            'delta_ci_dovs_ami'           : 'ci_dovs', 
            'delta_ci_dovs_ami_dovs_beg'  : 'ci_dovs', 
            'delta_cmi_dovs_ami'          : 'cmi_dovs', 
            'delta_cmi_dovs_ami_dovs_beg' : 'cmi_dovs', 
        }
    ):
        r"""
        Make new fields by dividing keys by values in fields_dict
        """
        #--------------------------------------------------
        assert(set(list(fields_dict.keys())+list(fields_dict.values())).difference(set(ci_cmi_summary_df.columns.tolist()))==set())
        #-------------------------
        for num_i, den_i in fields_dict.items():
            ci_cmi_summary_df[f"pct_{num_i}"] = 100.*ci_cmi_summary_df[num_i]/ci_cmi_summary_df[den_i]
        #-------------------------
        return ci_cmi_summary_df
    

    @staticmethod
    def make_ci_cmi_pretty(
        ci_cmi_col  ,
        rename_cols = None
    ):
        r"""
        If rename_cols is None, use the standard rename_cols, std_rename_cols
        """
        #-------------------------
        std_rename_cols = {
            'ci_dovs'                         : 'CI DOVS',
            'ci_ami'                          : 'CI AMI',
            'ci_ami_dovs_beg'                 : 'CI AMI w/ DOVS Beg.', 
            #-----
            'cmi_dovs'                        : 'CMI DOVS',
            'cmi_ami'                         : 'CMI AMI',
            'ci_ami_dovs_beg'                 : 'CMI AMI w/ DOVS Beg.', 
            #-----
            'delta_ci_dovs_ami'               : 'Delta CI DOVS vs AMI', 
            'delta_ci_dovs_ami_dovs_beg'      : 'Delta CI DOVS vs AMI w/ DOVS Beg.', 
            #-----
            'delta_cmi_dovs_ami'              : 'Delta CMI DOVS vs AMI', 
            'delta_cmi_dovs_ami_dovs_beg'     : 'Delta CMI DOVS vs AMI w/ DOVS Beg.',
            #-----
            'pct_delta_ci_dovs_ami'           : 'Pct. Delta CI DOVS vs AMI', 
            'pct_delta_ci_dovs_ami_dovs_beg'  : 'Pct. Delta CI DOVS vs AMI w/ DOVS Beg.', 
            #-----
            'pct_delta_cmi_dovs_ami'          : 'Pct. Delta CMI DOVS vs AMI', 
            'pct_delta_cmi_dovs_ami_dovs_beg' : 'Pct. Delta CMI DOVS vs AMI w/ DOVS Beg.',
        }    
        #-------------------------
        if rename_cols is None:
            rename_cols = std_rename_cols
        rename_cols = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = rename_cols,
            default_values_dict = std_rename_cols,
            extend_any_lists    = False,
            inplace             = False,
        )
        #-------------------------
        if ci_cmi_col not in rename_cols.keys():
            return ci_cmi_col
        return rename_cols[ci_cmi_col]
    

    @staticmethod
    def make_ci_cmi_summary_df_cols_pretty(
        ci_cmi_summary_df ,
        rename_cols       = None
    ):
        r"""
        If rename_cols is None, use the standard rename_cols, std_rename_cols
        """
        #-------------------------
        std_rename_cols = {
            'ci_dovs'                         : 'CI DOVS',
            'ci_ami'                          : 'CI AMI',
            'ci_ami_dovs_beg'                 : 'CI AMI w/ DOVS Beg.', 
            #-----
            'cmi_dovs'                        : 'CMI DOVS',
            'cmi_ami'                         : 'CMI AMI',
            'ci_ami_dovs_beg'                 : 'CMI AMI w/ DOVS Beg.', 
            #-----
            'delta_ci_dovs_ami'               : 'Delta CI DOVS vs AMI', 
            'delta_ci_dovs_ami_dovs_beg'      : 'Delta CI DOVS vs AMI w/ DOVS Beg.', 
            #-----
            'delta_cmi_dovs_ami'              : 'Delta CMI DOVS vs AMI', 
            'delta_cmi_dovs_ami_dovs_beg'     : 'Delta CMI DOVS vs AMI w/ DOVS Beg.',
            #-----
            'pct_delta_ci_dovs_ami'           : 'Pct. Delta CI DOVS vs AMI', 
            'pct_delta_ci_dovs_ami_dovs_beg'  : 'Pct. Delta CI DOVS vs AMI w/ DOVS Beg.', 
            #-----
            'pct_delta_cmi_dovs_ami'          : 'Pct. Delta CMI DOVS vs AMI', 
            'pct_delta_cmi_dovs_ami_dovs_beg' : 'Pct. Delta CMI DOVS vs AMI w/ DOVS Beg.',
        }    
        #-------------------------
        if rename_cols is None:
            rename_cols = std_rename_cols
        rename_cols = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = rename_cols,
            default_values_dict = std_rename_cols,
            extend_any_lists    = False,
            inplace             = False,
        )
        #-------------------------
        rename_cols = {k:v for k,v in rename_cols.items() if k in ci_cmi_summary_df.columns.tolist()}
        ci_cmi_summary_df = ci_cmi_summary_df.rename(columns=rename_cols)
        return ci_cmi_summary_df
    

    @staticmethod
    def identify_delta_cols_in_ci_cmi_summary_df(df):
        r"""
        Basically, just return columns containing 'delta' (case insensitive).  
        """
        #-------------------------
        delta_cols = [x for x in df.columns.tolist() if 'delta' in x.lower()]
        return delta_cols
    
    @staticmethod
    def get_delta_cols_from_ci_cmi_summary_df(df):
        r"""
        """
        #-------------------------
        return df[DABatch.identify_delta_cols_in_ci_cmi_summary_df(df)]
    

    @staticmethod
    def identify_cmi_cols_in_ci_cmi_summary_df(df):
        r"""
        Basically, just return columns containing 'cmi' (case insensitive).  
        """
        #-------------------------
        cmi_cols = [x for x in df.columns.tolist() if 'cmi' in x.lower()]
        return cmi_cols
    
    @staticmethod
    def get_cmi_cols_from_ci_cmi_summary_df(df):
        r"""
        """
        #-------------------------
        return df[DABatch.identify_cmi_cols_in_ci_cmi_summary_df(df)]
    

    @staticmethod
    def identify_ci_cols_in_ci_cmi_summary_df(df):
        r"""
        Basically, just return columns containing 'ci' (case insensitive).  
        """
        #-------------------------
        ci_cols = [x for x in df.columns.tolist() if 'ci' in x.lower()]
        return ci_cols
    
    @staticmethod
    def get_ci_cols_from_ci_cmi_summary_df(df):
        r"""
        """
        #-------------------------
        return df[DABatch.identify_ci_cols_in_ci_cmi_summary_df(df)]
    

    @staticmethod
    def try_to_determine_delta_metric(
        metric      , 
        make_pretty = True, 
        strip_delta = False, 
    ):
        r"""
        Basically, given metric, this tries to extract whether Detla CI or Delta CMI
    
        e.g. 1
            metric = 'delta_cmi_dovs_ami_dovs_beg'
            -----
            make_pretty = False
            strip_delta = False
            ==> returns 'delta_cmi'
    
            make_pretty = True
            strip_delta = False
            ==> returns 'Delta CMI'
    
            make_pretty = False
            strip_delta = True
            ==> returns 'cmi'
    
            make_pretty = True
            strip_delta = True
            ==> returns 'CMI'
    
        e.g. 2
            metric = 'Delta CI DOVS vs AMI'
            -----
            make_pretty = False
            strip_delta = False
            ==> returns 'Delta CI'
    
            make_pretty = True
            strip_delta = False
            ==> returns 'Delta CI'
    
            make_pretty = False
            strip_delta = True
            ==> returns 'CI'
    
            make_pretty = True
            strip_delta = True
            ==> returns 'CI'
        """
        #--------------------------------------------------
        pattern = r"((?:delta_c(?:m)?i)|(?:Delta C(?:M)?I))"
        flags   = re.IGNORECASE
        #-------------------------
        found = re.findall(pattern, metric, flags)
        assert(len(found)<=1)
        if len(found)==0:
            return metric
        #-------------------------
        found = found[0]
        #-------------------------
        if make_pretty:
            found = re.sub('delta_', 'Delta ', found, count=0, flags=flags)
            found = re.sub('cmi', 'CMI', found, count=0, flags=flags)
            found = re.sub('ci', 'CI', found, count=0, flags=flags)
        #-------------------------
        if strip_delta:
            found = re.sub('delta_|Delta\s', '', found, count=0, flags=flags)
        #-------------------------
        return found
    

    @staticmethod
    def build_summary_table_df(
        ci_cmi_summary_df       , 
        delta_col               , 
        #-----
        slice_col               , 
        limits_coll             , 
        are_limits_pct          , 
        #-----
        dovs_col                , 
        #-----
        metric                  = None, 
    ):
        r"""
        Creates a subset of ci_cmi_summary_df (using slice_col and limits) and returns a summary pd.DataFrame
          intended to be output into a table (using pd.plotting.table)
        -------------------------
        e.g.:
                final_df_cmi = DABatch.build_summary_table_df(
                    ci_cmi_summary_df = ci_cmi_summary_df, 
                    delta_col         = 'Delta CMI DOVS vs AMI', 
                    slice_col         = None, 
                    limits_coll       = [
                        [None, None], 
                        [-10000, 10000], 
                        [-1000, 1000], 
                        [-100, 100], 
                        [-10, 10], 
                    ], 
                    are_limits_pct    = False
                    dovs_col          = 'CMI DOVS', 
                    metric            = 'CMI', 
                )
        -------------------------
        ci_cmi_summary_df:
            Summary pd.DataFrame object, typically built using DABatchOPCO/DABatch methods

        outg_rec_int_col:
            The x-column used in lineplot

        delta_col:
            The column in ci_cmi_summary_df to plot

        -------------------------
        slice_col, limits, are_limits_pct

        slice_col:
            If user wants to visualize a subset of the data (selected using limits), the slice_col will be used to create the slice.
            If slice_col is None, it will be set to delta_col

        limits_coll:
            A collection of limits to impose to create slice.  This should be a list of list/tuple elements of length 2, with element-0 representing to minimum of
              the slice and element-1 the max of the slice.

        are_limits_pct:
            Tells the function whether the slice and limits are percentages (True) or raw values (False)
        -------------------------
        dovs_col:
            If included, this will create some additional text next to the plot showing the net DOVS value and the percentage difference between 
              our AMI method and DOVS.
            If None, this additional text will not be included.
            This column should be appropriate for use with the input delta_col!
            e.g., 
                delta_col      = 'Delta CI DOVS vs AMI'
                dovs_col       = 'CI DOVS'

            e.g., 
                delta_col      = 'Delta CMI DOVS vs AMI w/ DOVS Beg.'
                dovs_col       = 'CMI DOVS'
        -------------------------
        metric:
            Should be None or either 'CI' or 'CMI', corresponding to delta_col.
            If None, DABatch.try_to_determine_delta_metric will be utilized to try to determine
            
        """
        #-------------------------
        if metric is None:
            metric = DABatch.try_to_determine_delta_metric(
                metric      = delta_col, 
                make_pretty = False, 
                strip_delta = True, 
            )
        #-------------------------
        if slice_col is None:
            slice_col = delta_col  
        #-------------------------
        min_delta_col = 'Min Delta'
        max_delta_col = 'Max Delta'
        fnl_delta_col = f'Delta {metric}'
        if are_limits_pct:
            min_delta_col = 'Min Delta (%)'
            max_delta_col = 'Max Delta (%)'
        #-----
        return_df = pd.DataFrame(columns=[
            min_delta_col, 
            max_delta_col, 
            dovs_col,  
            fnl_delta_col
        ])
        #-------------------------
        for i,limits_i in enumerate(limits_coll):
            ci_cmi_summary_df_i = DABatch.get_ci_cmi_summary_df_subset(
                df      = ci_cmi_summary_df,
                cols    = slice_col,
                min_val = limits_i[0],
                max_val = limits_i[1]
            )
            #------------------------------------
            return_df = pd.concat([
                return_df, 
                pd.DataFrame(
                    {
                        min_delta_col: f"{limits_i[0]}", 
                        max_delta_col: f"{limits_i[1]}", 
                        dovs_col:      f"{ci_cmi_summary_df_i[dovs_col].sum()}",  
                        fnl_delta_col: f"{np.round(ci_cmi_summary_df_i[delta_col].sum(), decimals=2)} ({np.round(100*ci_cmi_summary_df_i[delta_col].sum()/ci_cmi_summary_df_i[dovs_col].sum(), decimals=2)}%)",  
                    }, 
                    index=[return_df.shape[0]]
                )
            ])
        #-------------------------
        return return_df
    

    @staticmethod
    def replace_delta_with_greek(
        txt              , 
        case_insensitive = False
    ):
        r"""
        For plotting purposes, this function will replace 'Delta' with '$\Delta$' (and 'delta' with '$\delta$').
        If case_insensitive==True, any Delta ('Delta', 'DELta', 'delta', etc.) will be replaced with '$\Delta$'
        """
        if not case_insensitive:
            txt = txt.replace('Delta', '$\Delta$')
            txt = txt.replace('delta', '$\delta$')
        else:
            # Cannot simply replace in-place in a while loop as I am replacing 'Delta', with '$\Delta$', so the 
            #   while loop will be infinite!!!!
            # Therefore, first find all occurrenced of 'Delta', then replace
            found_itr = re.finditer(r'Delta', txt, flags=re.IGNORECASE)
            #-----
            found_idxs=[]
            for found in found_itr:
                found_idxs.append([found.start(), found.end()])
            #-----
            # NOTE: Must replace from the right, so the indices of those to-be-replaced remain correct after
            #       others are replaced
            found_idxs = natsorted(found_idxs, reverse=True)
            #-----
            if len(found_idxs)>0:
                for found_idxs_i in found_idxs:
                    txt_0 = txt[:found_idxs_i[0]]
                    txt_1 = '$\Delta$'
                    txt_2 = txt[found_idxs_i[1]:]
                    #-----
                    txt=txt_0+txt_1+txt_2
        return txt
    

    @staticmethod
    def draw_delta_vs_outg_rec_int(
        ci_cmi_summary_df , 
        delta_col         , 
        outg_rec_int_col  = 'outg_rec_int', 
        #-----
        slice_col         = None, 
        limits            = [None, None], 
        are_limits_pct    = False, 
        #-----
        dovs_col          = None, 
        #-----
        fig_ax            = None, 
        save_dir          = None
    ):
        r"""
        Draws a lineplot showing the delta values for each outage.
        The outages are arbitrarily ordered (using the outg_rec_int_col)

        ci_cmi_summary_df:
            Summary pd.DataFrame object, typically built using DABatchOPCO/DABatch methods

        outg_rec_int_col:
            The x-column used in lineplot

        delta_col:
            The column in ci_cmi_summary_df to plot

        -------------------------
        slice_col, limits, are_limits_pct

        slice_col:
            If user wants to visualize a subset of the data (selected using limits), the slice_col will be used to create the slice.
            If slice_col is None, it will be set to delta_col

        limits:
            The limits to impose to create slice.  This should be a list/tuple of length 2, with element-0 representing to minimum of
              the slice and element-1 the max of the slice.

        are_limits_pct:
            Tells the function whether the slice and limits are percentages (True) or raw values (False)

        e.g. 1
            delta_col      = 'Delta CMI DOVS vs AMI'
            slice_col      = None (==> slice_col = 'Delta CMI DOVS vs AMI')
            limits         = [-10000, 10000]
            are_limits_pct = False

        e.g. 2
            delta_col      = 'Delta CMI DOVS vs AMI'
            slice_col      = 'Pct. Delta CMI DOVS vs AMI'
            limits         = [-50, 50]
            are_limits_pct = True
        -------------------------
        dovs_col:
            If included, this will create some additional text next to the plot showing the net DOVS value and the percentage difference between 
              our AMI method and DOVS.
            If None, this additional text will not be included.
            This column should be appropriate for use with the input delta_col!
            e.g., 
                delta_col      = 'Delta CI DOVS vs AMI'
                dovs_col       = 'CI DOVS'

            e.g., 
                delta_col      = 'Delta CMI DOVS vs AMI w/ DOVS Beg.'
                dovs_col       = 'CMI DOVS'

        fig_ax:
            A tuple containing a figure (matplotlib.figure.Figure) and axis (matplotlib.axes._axes.Axes) object
              where the figure will be plotted.
            If fig_ax is None, it is created using Plot_General.default_subplots()

        save_dir:
            If None, do not save figure
            If not None, must be a valid directory where the plot will be saved.
        """
        #--------------------------------------------------
        if fig_ax is None:
            fig_ax = Plot_General.default_subplots()
        assert(
            Utilities.is_object_one_of_types(fig_ax, [list, tuple]) and 
            len(fig_ax) == 2
        )
        assert(isinstance(fig_ax[0], mpl.figure.Figure))
        assert(isinstance(fig_ax[1], mpl.axes._axes.Axes))
        #-----
        fig = fig_ax[0]
        ax  = fig_ax[1]
        #-------------------------
        if slice_col is None:
            slice_col = delta_col
        #-------------------------
        nec_cols = [delta_col, slice_col]
        if dovs_col is not None:
            nec_cols.append(dovs_col)
        assert(set(nec_cols).difference(set(ci_cmi_summary_df.columns.tolist()))==set())
        #--------------------------------------------------
        if limits is None:
            limits = [None, None]
        ci_cmi_summary_df_i = DABatch.get_ci_cmi_summary_df_subset(
            df            = ci_cmi_summary_df,
            cols          = slice_col,
            min_val       = limits[0],
            max_val       = limits[1], 
            label_int_col = outg_rec_int_col
        )
        sns.lineplot(ax=ax, x=outg_rec_int_col, y=delta_col, data=ci_cmi_summary_df_i)
        if are_limits_pct:
            ax.set_title(f"Min/Max $\Delta$ (%) = [{limits[0]}, {limits[1]}]", fontsize=20)
        else:
            ax.set_title(f"Min/Max $\Delta$ = [{limits[0]}, {limits[1]}]", fontsize=20)
        
        Plot_General.set_general_plotting_args(
            ax          = ax, 
            tick_args   = [
                dict(axis='x', labelrotation=0, labelsize=14.0, direction='out'), 
                dict(axis='y', labelrotation=0, labelsize=14.0, direction='out')
            ], 
            xlabel_args = dict(xlabel='Outage', fontsize=16), 
            ylabel_args = dict(ylabel=DABatch.replace_delta_with_greek(ax.get_ylabel()), fontsize=16)
        )
        #--------------------------------------------------
        text_x       = 1.05
        text_y0      = 0.70
        text_delta_y = 0.10
        text_shared_kwargs = dict(
            ha        = 'left', 
            va        = 'center', 
            transform = ax.transAxes, 
            fontsize  = 'xx-large', 
            fontdict  = {'family':'monospace'}
        )
        #-------------------------
        if are_limits_pct:
            ax.text(text_x, text_y0, f"Min/Max $\Delta$ (%) = [{limits[0]}, {limits[1]}]", **text_shared_kwargs)
        else:
            ax.text(text_x, text_y0, f"Min/Max $\Delta$ = [{limits[0]}, {limits[1]}]", **text_shared_kwargs)
        #-----
        ax.text(text_x, text_y0-2*text_delta_y, f"Sums", **text_shared_kwargs)
        #-----
        if dovs_col is None:
            ax.text(
                text_x, text_y0-3*text_delta_y, 
                f"$\Delta$AMI = {np.round(ci_cmi_summary_df_i[delta_col].sum(), decimals=2)}", 
                **text_shared_kwargs
            )
        else:
            ax.text(text_x, text_y0-3*text_delta_y, f"{dovs_col} = {np.round(ci_cmi_summary_df_i[dovs_col].sum(), decimals=2)}", **text_shared_kwargs)
            ax.text(
                text_x, text_y0-4*text_delta_y, 
                f"$\Delta$AMI = {np.round(ci_cmi_summary_df_i[delta_col].sum(), decimals=2)} ({np.round(100*ci_cmi_summary_df_i[delta_col].sum()/ci_cmi_summary_df_i[dovs_col].sum(), decimals=2)}%)", 
                **text_shared_kwargs
            )
        #-------------------------
        if save_dir is not None:
            assert(os.path.isdir(save_dir))
            #-----
            Plot_General.save_fig(
                fig         = fig, 
                save_dir    = save_dir, 
                save_name   = 'delta_vs_outg_'+delta_col.replace(' ', '')+'.png', 
                bbox_inches = 'tight'
            )
        #-------------------------
        return fig,ax
    

    @staticmethod
    def draw_delta_vs_outg_w_summary_table(
        ci_cmi_summary_df , 
        delta_col         , 
        outg_rec_int_col  = 'outg_rec_int', 
        #-----
        slice_col         = None, 
        limits_coll       = None, 
        are_limits_pct    = False, 
        #-----
        dovs_col          = None, 
        #-----
        save_dir          = None
    ):
        r"""
        Draws a lineplot showing the delta values for each outage.
        The outages are arbitrarily ordered (using the outg_rec_int_col)
    
        ci_cmi_summary_df:
            Summary pd.DataFrame object, typically built using DABatchOPCO/DABatch methods
    
        outg_rec_int_col:
            The x-column used in lineplot
    
        delta_col:
            The column in ci_cmi_summary_df to plot
    
        -------------------------
        slice_col, limits, are_limits_pct
    
        slice_col:
            If user wants to visualize a subset of the data (selected using limits), the slice_col will be used to create the slice.
            If slice_col is None, it will be set to delta_col
    
        limits_coll:
            A collection of limits to impose to create slice.  
            This should be a list of list/tuple elements of length 2, with element-0 representing to minimum of
              the slice and element-1 the max of the slice.
            Can also be None
    
        are_limits_pct:
            Tells the function whether the slice and limits are percentages (True) or raw values (False)
        -------------------------
        dovs_col:
            If included, this will create some additional text next to the plot showing the net DOVS value and the percentage difference between 
              our AMI method and DOVS.
            If None, this additional text will not be included.
            This column should be appropriate for use with the input delta_col!
            e.g., 
                delta_col      = 'Delta CI DOVS vs AMI'
                dovs_col       = 'CI DOVS'
    
            e.g., 
                delta_col      = 'Delta CMI DOVS vs AMI w/ DOVS Beg.'
                dovs_col       = 'CMI DOVS'
    
        fig_ax:
            A tuple containing a figure (matplotlib.figure.Figure) and axis (matplotlib.axes._axes.Axes) object
              where the figure will be plotted.
            If fig_ax is None, it is created using Plot_General.default_subplots()
    
        save_dir:
            If None, do not save figure
            If not None, must be a valid directory where the plot will be saved.
        """
        #--------------------------------------------------
        if slice_col is None:
            slice_col = delta_col
        #-------------------------
        nec_cols = [delta_col, slice_col]
        if dovs_col is not None:
            nec_cols.append(dovs_col)
        assert(set(nec_cols).difference(set(ci_cmi_summary_df.columns.tolist()))==set())
        #-------------------------
        if limits_coll is None:
            limits_coll = [[None, None]]
        #--------------------------------------------------
        summary_table_df = DABatch.build_summary_table_df(
            ci_cmi_summary_df = ci_cmi_summary_df, 
            limits_coll       = limits_coll, 
            delta_col         = delta_col, 
            dovs_col          = dovs_col, 
            slice_col         = slice_col, 
            metric            = None, 
            are_limits_pct    = are_limits_pct
        )
        #--------------------------------------------------
        fig, axs = Plot_General.default_subplots(
            n_x = 1, 
            n_y = len(limits_coll)+1
        )
        Plot_General.adjust_subplots_args(fig, **Plot_General.get_subplots_adjust_args(hspace=0.30))
        #-------------------------
        axs[0].axis('off')
        tbl = pd.plotting.table(
            axs[0], 
            summary_table_df, 
            cellLoc    = 'center', 
            rowLoc     = 'center', 
            bbox       = [0,0,1,1], 
            colColours = ['white', 'white', 'white', 'green']
        )
        tbl.set_fontsize(16)
        #-------------------------
        for i,ci_limits in enumerate(limits_coll):
            DABatch.draw_delta_vs_outg_rec_int(
                ci_cmi_summary_df = ci_cmi_summary_df, 
                delta_col         = delta_col, 
                outg_rec_int_col  = outg_rec_int_col, 
                slice_col         = slice_col, 
                limits            = ci_limits, 
                are_limits_pct    = are_limits_pct, 
                dovs_col          = dovs_col, 
                fig_ax            = (fig, axs[i+1]), 
                save_dir          = None
            )
        #-------------------------
        if are_limits_pct:
            fig.suptitle('Subsets extracted using relative $\Delta$s', y=0.90, fontsize='xx-large')
        else:
            fig.suptitle('Subsets extracted using raw $\Delta$s', y=0.90, fontsize='xx-large')
        #-------------------------
        if save_dir is not None:
            assert(os.path.isdir(save_dir))
            #-----
            Plot_General.save_fig(
                fig         = fig, 
                save_dir    = save_dir, 
                save_name   = 'delta_vs_outg_w_summary_table'+delta_col.replace(' ', '')+'.png', 
                bbox_inches = 'tight'
            )    
        #-------------------------
        return fig,axs
    

    @staticmethod
    def plot_deltas_histogram(
        ci_cmi_summary_df      , 
        delta_col              , 
        min_max_and_bin_size   , 
        include_over_underflow , 
        stat                   = 'count', 
        fig_ax                 = None, 
        save_dir               = None
    ):
        r"""
        ci_cmi_summary_df:
            Summary pd.DataFrame object, typically built using DABatchOPCO/DABatch methods
    
        delta_col:
            The column in ci_cmi_summary_df to plot
    
        min_max_and_bin_size:
          If min_max_and_bin_size is None, use default number of bins in pandas.DataFrame.hist, which is 10
          If min_max_and_bin_size is an integer, take that the be the number of bins
          Otherwise, min_max_and_bin_size must be a list/tuple of length 3 containing 
            min_in_plot, max_in_plot, and bin_size
        
        include_over_underflow:
          This can be a single boolean, or a list/tuple of two booleans
          If single boolean: include_overflow = include_underflow = include_over_underflow
          If pair of booleans: include_overflow = include_over_underflow[0], 
                               include_underflow = include_over_underflow[1]
    
        stat:
          Must be either 'count' or 'density'
              count: show the number of observations in each bin
              density: normalize such that the total area of the histogram equals 1
    
        fig_ax:
            A tuple containing a figure (matplotlib.figure.Figure) and axis (matplotlib.axes._axes.Axes) object
              where the figure will be plotted.
            If fig_ax is None, it is created using Plot_General.default_subplots()
    
        save_dir:
            If None, do not save figure
            If not None, must be a valid directory where the plot will be saved.
        """
        #--------------------------------------------------
        if fig_ax is None:
            fig_ax = Plot_General.default_subplots()
        assert(
            Utilities.is_object_one_of_types(fig_ax, [list, tuple]) and 
            len(fig_ax) == 2
        )
        assert(isinstance(fig_ax[0], mpl.figure.Figure))
        assert(isinstance(fig_ax[1], mpl.axes._axes.Axes))
        #-----
        fig = fig_ax[0]
        ax  = fig_ax[1]
        #-------------------------
        assert(delta_col in ci_cmi_summary_df.columns.tolist())
        #-------------------------
        Plot_Hist.plot_hist(
            ax                     = ax, 
            df                     = ci_cmi_summary_df, 
            x_col                  = delta_col, 
            min_max_and_bin_size   = min_max_and_bin_size, 
            include_over_underflow = include_over_underflow, 
            stat                   = stat
        )
        #-------------------------
        label = DABatch.replace_delta_with_greek(
            txt              = DABatch.make_ci_cmi_pretty(delta_col), 
            case_insensitive = True
        )
        #-----
        xlabel = DABatch.replace_delta_with_greek(
            txt = DABatch.try_to_determine_delta_metric(
                metric      = delta_col, 
                make_pretty = True, 
                strip_delta = False, 
            ), 
            case_insensitive = True
        )
        #-------------------------
        Plot_General.set_general_plotting_args(
            ax,
            title_args  = dict(label=label,     fontsize=18),
            xlabel_args = dict(xlabel=xlabel,   fontsize=14), 
            ylabel_args = dict(ylabel='Counts', fontsize=14)
        );
        #-------------------------
        if save_dir is not None:
            assert(os.path.isdir(save_dir))
            #-----
            Plot_General.save_fig(
                fig         = fig, 
                save_dir    = save_dir, 
                save_name   = 'deltas_hist_'+delta_col.replace(' ', '')+'.png', 
                bbox_inches = 'tight'
            )
        #-------------------------
        return fig,ax