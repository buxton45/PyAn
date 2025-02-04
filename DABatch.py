#!/usr/bin/env python

r"""
Holds DABatch class.  See DABatch.DABatch for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import pandas as pd
import numpy as np
import time
import copy

#---------------------------------------------------------------------
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from MeterPremise import MeterPremise
#-----
from AMI_SQL import AMI_SQL, DfToSqlMap
from AMINonVee_SQL import AMINonVee_SQL
from AMIEndEvents_SQL import AMIEndEvents_SQL
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from AMINonVee import AMINonVee
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
from Utilities_df import DFConstructType
#---------------------------------------------------------------------

class DABatch:
    r"""
    Class to adjust DOVS outage times and CI/CMI
    """
    def __init__(
        self, 
        date_0        , 
        date_1        , 
        states        = None, # e.g., ['OH']
        opcos         = None, # e.g., ['oh']
        CI_NB_min     = None, # e.g., 15
        mjr_mnr_cause = None, 
        
        init_dfs_to     = pd.DataFrame(),  # Should be pd.DataFrame() or None
    ):
        r"""

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
        self.date_0        = date_0
        self.date_1        = date_1
        self.states        = states
        self.opcos         = opcos
        self.CI_NB_min     = CI_NB_min
        self.mjr_mnr_cause = mjr_mnr_cause
        #-------------------------
        self.outages_sql = None
        self.outages_df  = copy.deepcopy(self.init_dfs_to)
        #-----
        self.outages_mp_df_OG  = copy.deepcopy(self.init_dfs_to)
        self.outages_mp_df     = copy.deepcopy(self.init_dfs_to)
        #-----
        self.outages_mp_df_ami = copy.deepcopy(self.init_dfs_to)
        self.outages_df_ami    = copy.deepcopy(self.init_dfs_to)
        #-----

        #-------------------------

        #---------------------------------------------------------------------------
        # Grabbing connection can take time (seconds, not hours).
        # Keep set to None, only creating if needed (see conn_aws/conn_dovs property below)
        self.__conn_aws   = None
        self.__conn_dovs  = None
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

    #----------------------------------------------------------------------------------------------------
    def load_daq_prereqs(
        self     , 
        base_dir , 
        how      = 'min', 
        verbose  = True
    ):
        r"""
        Load all of the required attributes needed for batch data acquistion.
        The purpose here is to allow one to easily re-start a DAQ run that may have failed part way through.
    
        how:
            Should be either 'min' or 'full'
            For DAQ, (aside from any kwargs, e.g., opco) the only data really needed is audit_candidates_df.
            When 'min' is selected, only audit_candidates_df is loaded
        """
        #--------------------------------------------------
        assert(os.path.isdir(base_dir))
        assert(how in ['min', 'full'])
        #-------------------------
        # In all cases, audit_candidates_df must be loaded
        assert(os.path.exists(os.path.join(base_dir, 'audit_candidates_df.pkl')))
        ac_df = pd.read_pickle(os.path.join(base_dir, 'audit_candidates_df.pkl'))
        self.set_audit_candidates_df(audit_candidates_df = ac_df)
        #-------------------------
        if how == 'min':
            if verbose:
                print(f"Loaded minimum DAQ prereqs from base_dir = {base_dir}")
            return
        #-------------------------
        other_dfs = {
            'outages_df'        : 'outages_df', 
            'outages_mp_df_OG'  : 'outages_mp_df_b4_dupl_rmvl', 
            'outages_mp_df'     : 'outages_mp_df', 
            'outages_mp_df_ami' : 'outages_mp_df_ami'
        }
        success = {'audit_candidates_df': 'True'}
        #-----
        for attr_name_i, file_name_i in other_dfs.items():
            path_i = os.path.join(base_dir, f'{file_name_i}.pkl')
            if os.path.exists(path_i):
                try:
                    df_i = pd.read_pickle(path_i)
                    setattr(self, attr_name_i, df_i)
                    success[attr_name_i] = 'True'
                except:
                    success[attr_name_i] = 'False'
            else:
                success[attr_name_i] = 'False'
        #-------------------------
        if verbose:
            print(f"Loaded full DAQ prereqs from base_dir = {base_dir}")
            for k,v in success.items():
                print(f"{k} : {v}")
    

    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    def compile_outages_sql(
        self
    ):
        r"""
        """
        #-------------------------
        outages_sql = DOVSOutages_SQL.build_sql_std_outage(
            mjr_mnr_cause   = self.mjr_mnr_cause, 
            include_premise = True, 
            date_range      = [self.date_0, self.date_1], 
            states          = self.states, 
            opcos           = self.opcos, 
            CI_NB_min       = self.CI_NB_min
        )
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
        self          , 
        save_dir_base = None, 
        verbose       = True
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
            self.conn_dovs, 
            dtype={
                'CI_NB'       : np.int32, 
                'CMI_NB'      : np.float64, 
                'OUTG_REC_NB' : np.int32
            }
        )
        #--------------------------------------------------
        if save_dir_base is not None:
            assert(os.path.isdir(save_dir_base))
            outages_df.to_pickle(os.path.join(save_dir_base, 'outages_df.pkl'))
        #--------------------------------------------------
        if verbose:
            print(f'outages_sql_stmnt:\n{outages_sql_stmnt}\n\n')
            print(f"outages_df.shape = {outages_df.shape}")
            print(f"# OUTG_REC_NBs   = {outages_df['OUTG_REC_NB'].nunique()}")
        #--------------------------------------------------
        self.outages_sql = outages_sql
        self.outages_df  = outages_df
        

    #----------------------------------------------------------------------------------------------------
    def build_outages_mp_df(
        self, 
        save_dir_base = None, 
        verbose       = True
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
        if save_dir_base is not None:
            assert(os.path.isdir(save_dir_base))
            outages_mp_df_OG.to_pickle(os.path.join(save_dir_base, 'outages_mp_df_b4_dupl_rmvl.pkl'))
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
        if save_dir_base is not None:
            assert(os.path.isdir(save_dir_base))
            outages_mp_df.to_pickle(os.path.join(save_dir_base, 'outages_mp_df.pkl'))
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
        self, 
        save_dir_base = None, 
        verbose       = True
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
        #--------------------------------------------------
        if save_dir_base is not None:
            assert(os.path.isdir(save_dir_base))
            outages_mp_df_ami.to_pickle(os.path.join(save_dir_base, 'outages_mp_df_ami.pkl'))
        #--------------------------------------------------
        if verbose:
            print(f"# Outages in outages_mp_df:     {self.outages_mp_df['OUTG_REC_NB'].nunique()}")
            print(f"# Outages in outages_mp_df_ami: {self.outages_mp_df_ami['OUTG_REC_NB'].nunique()}")


    #--------------------------------------------------
    def finalize_audit_candidates(
        self, 
        save_dir_base           = None, 
        search_time_half_window = pd.Timedelta('24 hours'), 
        verbose                 = True
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
            df_outage               = ac_df, 
            search_time_half_window = search_time_half_window
        )
        #--------------------------------------------------
        self.__audit_candidates_df = ac_df
        #--------------------------------------------------
        if save_dir_base is not None:
            assert(os.path.isdir(save_dir_base))
            ac_df.to_pickle(os.path.join(save_dir_base, 'audit_candidates_df.pkl'))

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
        save_dir_base , 
        batch_size    = 25, 
        verbose       = True, 
        n_update      = 1, 
    ):
        r"""
        """
        #--------------------------------------------------
        assert(save_dir_base is not None)
        if not os.path.exists(save_dir_base):
            os.makedirs(save_dir_base)
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
                opcos  = self.opcos, 
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
            save_dir     = os.path.join(save_dir_base, 'AMINonVee'), 
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
        save_dir_base , 
        pdpu_only     = True, 
        batch_size    = 25, 
        verbose       = True, 
        n_update      = 1, 
    ):
        r"""
        """
        #--------------------------------------------------
        assert(save_dir_base is not None)
        if not os.path.exists(save_dir_base):
            os.makedirs(save_dir_base)
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
                opcos                 = self.opcos, 
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
            save_dir     = os.path.join(save_dir_base, 'EndEvents'), 
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