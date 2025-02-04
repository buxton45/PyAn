#!/usr/bin/env python

r"""
Holds OutagePredictor class.  See OutagePredictor.OutagePredictor for more information.
"""

__author__ = "Jesse Buxton"
__email__ = "jbuxton@aep.com"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import re
from pathlib import Path
import json
import pickle
import joblib

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
#---------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib import dates
import matplotlib.colors as mcolors
import matplotlib.cm as cm #e.g. for cmap=cm.jet
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from MeterPremise import MeterPremise
from EEMSP import EEMSP
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
from AMIEDE_DEV import AMIEDE_DEV
from MECPODf import MECPODf
from MECPOAn import MECPOAn
from AMIUsgInst import AMIUsgInst
from DOVSOutages import DOVSOutages
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from TableInfos import TableInfo
from SQLElement import SQLElement
from SQLElementsCollection import SQLElementsCollection
from SQLSelect import SQLSelectElement, SQLSelect
from SQLFrom import SQLFrom
from SQLWhere import SQLWhereElement, SQLWhere
from SQLJoin import SQLJoin, SQLJoinCollection
from SQLGroupBy import SQLGroupByElement, SQLGroupBy
from SQLHaving import SQLHaving
from SQLOrderBy import SQLOrderByElement, SQLOrderBy
from SQLQuery import SQLQuery
from SQLQueryGeneric import SQLQueryGeneric
#---------------------------------------------------------------------
#sys.path.insert(0, os.path.join(os.path.realpath('..'), 'Utilities'))
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Utilities_dt
import Plot_General
import Plot_Box_sns
import Plot_Hist
import Plot_Bar
import GrubbsTest
import DataFrameSubsetSlicer
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
#---------------------------------------------------------------------

class OutagePredictor:
    r"""
    Class to make outage predictions
    """
    def __init__(
        self, 
        prediction_date=None, 
        trsf_pole_nbs = None, 
        idk_name_1 = pd.Timedelta('31D'), 
        idk_name_2 = pd.Timedelta('1D')
    ):
        r"""
        """
        #-------------------------
        # Grabbing connection can take time (seconds, not hours).
        # Keep set to None, only creating if needed (see conn_aws property below)
        self.__conn_aws  = None
        #-------------------------
        self.__trsf_pole_nbs = None
        self.trsf_pole_nbs_sql = '' # If SQL run to obtain trsf_pole_nbs, this will be set
        if trsf_pole_nbs is not None:
            self.set_trsf_pole_nbs(trsf_pole_nbs)
        #-------------------------
        if prediction_date is None:
            self.prediction_date = pd.to_datetime(datetime.date.today())
        else:
            self.prediction_date = prediction_date
        #-----
        self.idk_name_1 = idk_name_1
        self.idk_name_2 = idk_name_2
        #-----
        self.date_range = None
        self.set_date_range()
        #-------------------------
        # Although I will use AMIEndEvents methods, the object which is built will not
        #   contain end_device_event entries (ede), but rather events_summary_vw (evsSum)
        self.__evsSum_sql_fcn = None
        self.__evsSum_sql_kwargs = None
        self.evsSum = None
        #self.evsSum_df = None ############# Should only return evsSum.df, should not be separate object
        #-------------------------
        self.__regex_setup_df = None
        self.__cr_trans_dict  = None
        #-----
        self.rcpx_df = None
        self.is_norm = False
        #-------------------------
        self.merge_eemsp = False
        self.eemsp_mult_strategy='agg'
        self.eemsp_df = None
        #-------------------------
        self.__model_dir = None
        self.model_summary_dict = None
        self.data_structure_df = None
        self.model_clf = None
        self.scale_data = False
        self.scaler = None
        self.eemsp_enc = None
        self.include_month = False
        #-----
        self.X_test = None
        self.y_pred = None

    
    @property
    def evsSum_df(self):
        return self.evsSum.df.copy()
    
    @property
    def conn_aws(self):
        if self.__conn_aws is None:
            self.__conn_aws  = Utilities.get_athena_prod_aws_connection()
        return self.__conn_aws
    
    @property
    def trsf_pole_nbs(self):
        return self.__trsf_pole_nbs
    
    @property
    def regex_setup_df(self):
        if self.__regex_setup_df is None:
            self.build_regex_setup()
        return self.__regex_setup_df
    
    @property
    def cr_trans_dict(self):
        if self.__cr_trans_dict is None:
            self.build_regex_setup()
        return self.__cr_trans_dict
    
    @property
    def evsSum_sql_fcn(self):
        return self.__evsSum_sql_fcn
    
    @property
    def evsSum_sql_kwargs(self):
        return self.__evsSum_sql_kwargs
    
    @property
    def model_dir(self):
        return self.__model_dir
    
    
    def set_trsf_pole_nbs(
        self, 
        trsf_pole_nbs
    ):
        assert(isinstance(trsf_pole_nbs, list))
        self.__trsf_pole_nbs = trsf_pole_nbs
        
    def set_trsf_pole_nbs_from_sql(
        self, 
        n_trsf_pole_nbs=None, 
        **kwargs
    ):
        r"""
        Run SQL query using kwargs to gather trsf_pole_nbs and set memeber attribute.

        n_trsf_pole_nbs:
            If set, this will randomly return n_trsf_pole_nbs instead of the full set
        """
        #-------------------------
        trsf_pole_nbs_df, trsf_sql = OutagePredictor.get_distinct_trsf_pole_nbs_df(
            conn_aws=self.conn_aws, 
            n_trsf_pole_nbs=n_trsf_pole_nbs, 
            return_sql=True, 
            **kwargs
        )
        trsf_pole_nbs = trsf_pole_nbs_df['trsf_pole_nb'].tolist()
        self.set_trsf_pole_nbs(trsf_pole_nbs)
        self.trsf_pole_nbs_sql = trsf_sql
        
    def set_date_range(
        self
    ):
        r"""
        """
        #-------------------------
        if not isinstance(self.prediction_date, datetime.datetime):
            self.prediction_date = pd.to_datetime(self.prediction_date)
        if not isinstance(self.idk_name_1, pd.Timedelta):
            self.idk_name_1 = pd.Timedelta(self.idk_name_1)
        if not isinstance(self.idk_name_2, pd.Timedelta):
            self.idk_name_2 = pd.Timedelta(self.idk_name_2)
        #-------------------------
        self.date_range = [
            (self.prediction_date-self.idk_name_1).date(), 
            (self.prediction_date-self.idk_name_2).date()
        ]

        
    def set_model_dir(
        self, 
        model_dir, 
        **kwargs
    ):
        r"""
        Sets the model_dir and extracts the needed contents:
            - model_clf
            - data_structure_df
            - scaler
            - eemsp_encoder
        """
        #-------------------------
        assert(os.path.exists(model_dir))
        self.__model_dir = model_dir
        #-------------------------
        model_summary_dict_fname = kwargs.get('model_summary_dict_fname', 'summary_dict.json')
        data_structure_fname     = kwargs.get('data_structure_fname', 'data_structure_df.pkl')
        model_fname              = kwargs.get('model_fname', 'forest_clf.joblib')
        scaler_fname             = kwargs.get('scaler_fname', 'scaler.joblib')
        eemsp_encoder_fname      = kwargs.get('eemsp_encoder_fname', 'eemsp_encoder.joblib')
        #-------------------------
        # model_summary_dict, data_structure_df, model_clf MUST be present
        assert(os.path.exists(os.path.join(model_dir, model_summary_dict_fname)))
        tmp_f = open(os.path.join(model_dir, model_summary_dict_fname))
        self.model_summary_dict = json.load(tmp_f)
        tmp_f.close()
        #-----
        assert(os.path.exists(os.path.join(model_dir, data_structure_fname)))
        self.data_structure_df = pd.read_pickle(os.path.join(model_dir, data_structure_fname))
        #-----
        assert(os.path.exists(os.path.join(model_dir, model_fname)))
        self.model_clf = joblib.load(os.path.join(model_dir, model_fname))
        #-------------------------
        self.scale_data = self.model_summary_dict['run_scaler']
        if self.scale_data:
            assert(os.path.exists(os.path.join(model_dir, scaler_fname)))
            self.scaler = joblib.load(os.path.join(model_dir, scaler_fname))
        #-----
        self.merge_eemsp = self.model_summary_dict['merge_eemsp']
        if self.merge_eemsp:
            assert(os.path.exists(os.path.join(model_dir, eemsp_encoder_fname)))
            self.eemsp_enc = joblib.load(os.path.join(model_dir, eemsp_encoder_fname))
            self.eemsp_mult_strategy = self.model_summary_dict['eemsp_mult_strategy']
        #-----
        self.include_month = self.model_summary_dict['include_month']
            
    def build_events_summary(
        self, 
        evsSum_sql_fcn=AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs=None, 
        init_df_in_constructor=True, 
        save_args=False
    ):
        r"""
        If user supplies any evsSum_sql_kwargs, user SHOULD NOT include:
            - date_range
            - trsf_pole_nbs
        as these will be supplied from self
        """
        #-------------------------
        dflt_evsSum_sql_kwargs = dict(
            schema_name='meter_events', 
            table_name='events_summary_vw', 
            cols_of_interest=['*'], 
            date_range=self.date_range, 
            trsf_pole_nbs=self.trsf_pole_nbs
        )
        #-------------------------
        if evsSum_sql_kwargs is None:
            evsSum_sql_kwargs = dict()
        assert(isinstance(evsSum_sql_kwargs, dict))
        #-------------------------
        evsSum_sql_kwargs = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict=evsSum_sql_kwargs, 
            default_values_dict=dflt_evsSum_sql_kwargs, 
            extend_any_lists=False, 
            inplace=False
        )
        #-----
        # Make 100% sure date_range and trsf_pole_nbs are properly set
        evsSum_sql_kwargs['date_range']    = self.date_range
        evsSum_sql_kwargs['trsf_pole_nbs'] = self.trsf_pole_nbs
        #-------------------------
        self.__evsSum_sql_fcn    = evsSum_sql_fcn
        self.__evsSum_sql_kwargs = copy.deepcopy(evsSum_sql_kwargs)
        #-------------------------
        self.evsSum = AMIEndEvents(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = dict(conn_db=self.conn_aws), 
            build_sql_function        = self.evsSum_sql_fcn, 
            build_sql_function_kwargs = self.evsSum_sql_kwargs, 
            init_df_in_constructor    = init_df_in_constructor, 
            save_args                 = save_args
        )
        
        
    def build_regex_setup(
        self
    ):
        r"""
        Builds regex_setup_df and cr_trans_dict, where cr_trans_dict = curated reasons translation dictionary
        """
        #-------------------------
        sql = """
        SELECT * FROM meter_events.event_summ_regex_setup
        """
        self.__regex_setup_df = pd.read_sql(sql, self.conn_aws, dtype=str)
        #-------------------------
        self.__cr_trans_dict = {x[0]:x[1] for x in self.__regex_setup_df[['pivot_id', 'regex_report_title']].values.tolist()}
            
    @staticmethod
    def get_distinct_trsf_pole_nbs_df(
        conn_aws, 
        n_trsf_pole_nbs=None, 
        return_sql=False, 
        **kwargs
    ):
        return MeterPremise.get_distinct_trsf_pole_nbs(
            n_trsf_pole_nbs=n_trsf_pole_nbs, 
            conn_aws=conn_aws, 
            return_sql=return_sql, 
            **kwargs
        )
    
    @staticmethod
    def assert_timedelta_is_days(td):
        r"""
        The analysis typically expects the frequency to be in days.
        This function checks the attributes of td to ensure this is true
        """
        assert(isinstance(td, pd.Timedelta))
        td_comps = td.components
        #-----
        assert(td_comps.days>0)
        #-----
        assert(td_comps.hours==0)
        assert(td_comps.minutes==0)
        assert(td_comps.seconds==0)
        assert(td_comps.milliseconds==0)
        assert(td_comps.microseconds==0)
        assert(td_comps.nanoseconds==0)

    @staticmethod
    def get_time_pds_rename(
        curr_time_pds, 
        td_min=pd.Timedelta('1D'), 
        td_max=pd.Timedelta('31D'), 
        freq=pd.Timedelta('5D')
    ):
        r"""
        Need to convert the time periods, which are currently housed in the date_col index of 
          rcpx_0 from their specific dates to the names expected by the model.
        In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
          dates of the given interval.
        These will be converted to the titles contained in final_time_pds below
         e.g.:
               curr_time_pds = [Timestamp('2023-05-02'), Timestamp('2023-05-07'), 
                                Timestamp('2023-05-12'), Timestamp('2023-05-17'), 
                                Timestamp('2023-05-22'), Timestamp('2023-05-27')]
           ==> final_time_pds = ['01-06 Days', '06-11 Days',
                                 '11-16 Days', '16-21 Days',
                                 '21-26 Days', '26-31 Days']

         Returns: A dict object with keys equal to curr_time_pds and values equal to final_time_pds
         
        !!! NOTE !!!: This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
        """
        #-------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        freq   = pd.Timedelta(freq)
        #-------------------------
        OutagePredictor.assert_timedelta_is_days(td_min)
        OutagePredictor.assert_timedelta_is_days(td_max)
        OutagePredictor.assert_timedelta_is_days(freq)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        days_freq = freq.days
        #-----
        # Make sure (days_max-days_min) evenly divisible by days_freq
        assert((days_max-days_min) % days_freq==0)
        #-------------------------
        curr_time_pds = natsorted(curr_time_pds)
        # Each time period should have a width equal to freq
        for i in range(len(curr_time_pds)):
            if i==0:
                continue
            assert(curr_time_pds[i]-curr_time_pds[i-1]==freq)
        #-------------------------
        final_time_pds_0 = np.arange(start=days_min, stop=days_max+1, step=days_freq).tolist()
        #----------
        # Use padded zeros
        #   i.e., instead of '1-6 Days', use '01-06 Days'
        #   Will need to know longest string length for this
        str_len = np.max([len(str(x)) for x in final_time_pds_0])
        # Looks like a ton of curly braces below, but need double {{ and double }} to escape
        # ==> if str_len=2, then str_fmt = '{:02d}'
        str_fmt = f'{{:0{str_len}d}}'
        #----------
        final_time_pds = []
        for i in range(len(final_time_pds_0)):
            if i==0:
                continue
            fnl_pd_im1 = "{}-{} Days".format(
                str_fmt.format(final_time_pds_0[i-1]), 
                str_fmt.format(final_time_pds_0[i])
            )
            final_time_pds.append(fnl_pd_im1)
        #-------------------------
        assert(len(final_time_pds)==len(curr_time_pds))
        #-------------------------
        # final_time_pds is arranged from closest to prediction date to furthest
        #   e.g., final_time_pds = ['01-06 Days', '06-11 Days', ..., '26-31 Days']
        # curr_time_pds is arranged from furthesto to closest
        #   e.g., curr_time_pds = [2023-06-26, 2023-07-01, ..., 2023-07-21]
        # To make the two agree, re-sort curr_time_pds with reverse=True
        curr_time_pds = natsorted(curr_time_pds, reverse=True)
        #-------------------------
        time_pds_rename = dict(zip(curr_time_pds, final_time_pds))
        #-------------------------
        return time_pds_rename
    
    @staticmethod
    def project_time_pd_from_rcpx_0_and_prepare_OLD(
        rcpx_0, 
        date_pd_i, 
        final_time_pd_i, 
        data_structure_df, 
        meter_cnt_per_gp_srs, 
        all_groups, 
        cr_trans_dict, 
        non_reason_cols, 
        other_reasons_col, 
        group_cols=['trsf_pole_nb'], 
        date_col='aep_event_dt', 
        regex_patterns_to_remove=['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons=True, 
        include_power_down_minus_up=False, 
        total_counts_col = 'total_counts', 
        nSNs_col = 'nSNs', 
    ):
        r"""
        """
        #-------------------------
        # Get the expected columns for this time period from data_structure_df
        final_reason_cols_i = data_structure_df[final_time_pd_i].columns.tolist()
        final_reason_cols_i = [x for x in final_reason_cols_i if x not in non_reason_cols+[other_reasons_col]]
        #-------------------------
        # Project out the current time period (date_pd_i) from rcpx_0 by selecting the appropriate
        #   values from the date_col index
        rcpx_0_pd_i = rcpx_0[rcpx_0.index.get_level_values(date_col)==date_pd_i].copy()
        rcpx_0_pd_i = rcpx_0_pd_i.droplevel(date_col, axis=0)
        #-------------------------
        # Make sure all groups (typically trsf_pole_nbs) have an entry in rcpx_0_pd_i:
        #   If a group didn't register any events in a given time period, it will not be included in the projection.
        #   However, the final format requires each group have entries for each time period
        #   Therefore, we identify the groups missing from rcpx_0_pd_i (no_events_pd_i) and add approriate rows
        #     containing all 0 values for the counts
        # NOTE: If group_cols contains more than one element, the index will be a MultiIndex (equal to the group_cols)
        #         and will need to be treated slightly different
        no_events_pd_i = list(set(all_groups).difference(set(rcpx_0_pd_i.index.unique())))
        if len(group_cols)==1:
            no_ev_idx = no_events_pd_i
        else:
            no_ev_idx = pd.MultiIndex.from_tuples(no_events_pd_i)
        no_events_pd_i_df = pd.DataFrame(
            columns=rcpx_0.columns, 
            index=no_ev_idx, 
            data=np.zeros((len(no_events_pd_i), rcpx_0.shape[1]))
        )
        no_events_pd_i_df.index.names = rcpx_0_pd_i.index.names
        #-----
        # Use meter_cnt_per_gp_srs to fill the nSNs_col column in no_events_pd_i_df (since simply full of 0s)
        # NOTE: This is probably not strictly necessary, as the nSNs_col column won't be used here,
        #         since the data are not normalized.
        no_events_pd_i_df = no_events_pd_i_df.drop(columns=[nSNs_col]).merge(
            meter_cnt_per_gp_srs, 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        # Sanity check on the merge
        assert(no_events_pd_i_df[nSNs_col].notna().all())
        #-----
        # Combine rcpx_0_pd_i and no_events_pd_i_df
        assert(len(set(rcpx_0_pd_i.columns).symmetric_difference(set(no_events_pd_i_df.columns)))==0)
        no_events_pd_i_df = no_events_pd_i_df[rcpx_0_pd_i.columns]
        rcpx_0_pd_i = pd.concat([rcpx_0_pd_i, no_events_pd_i_df])
        #-------------------------
        # Rename the cr# columns to their full curated reasons
        rcpx_0_pd_i=rcpx_0_pd_i.rename(columns=cr_trans_dict)
        #--------------------------------------------------    
        #--------------------------------------------------
        # Any columns without a curated reason (i.e., those with column name = ''), have not been observed
        #   yet in the data, and therefore the sume of the counts should be 0.
        # These empty columns are not needed, so drop
        assert(rcpx_0_pd_i[''].sum().sum()==0)
        rcpx_0_pd_i=rcpx_0_pd_i.drop(columns=[''])  
        #-------------------------
        # Remove any undesired curated reasons (e.g., ['.*cleared.*', '.*Test Mode.*'])
        if regex_patterns_to_remove is not None:
            rcpx_0_pd_i = MECPODf.remove_reasons_from_rcpo_df(
                rcpo_df=rcpx_0_pd_i, 
                regex_patterns_to_remove=regex_patterns_to_remove, 
                ignore_case=True
            )
        #-----
        # After irrelevant cleared and test columns removed, need to recalculate events_tot to accurately
        #   reflect the total number of relevant events
        # Safe(r) to do this calculation in any case, so moved outside of the if block above
        assert(total_counts_col in non_reason_cols)
        rcpx_0_pd_i[total_counts_col] = rcpx_0_pd_i.drop(columns=non_reason_cols).sum(axis=1) 
        #-------------------------
        # Combine similar reasons (e.g., all 'Tamper' type reasons are combined into 1)
        # See MECPODf.combine_cpo_df_reasons for more information
        if combine_cpo_df_reasons:
            rcpx_0_pd_i = MECPODf.combine_cpo_df_reasons(rcpo_df=rcpx_0_pd_i)
        #-------------------------
        # Include the difference in power-up and power-down, if desired (typically turned off) 
        if include_power_down_minus_up:
            rcpx_0_pd_i = MECPODf.delta_cpo_df_reasons(
                rcpo_df=rcpx_0_pd_i, 
                reasons_1='Primary Power Down',
                reasons_2='Primary Power Up',
                delta_reason_name='Power Down Minus Up'
            )
        #-------------------------
        # Make sure rcpx_0_pd_i contains the expected final reason columns.
        # Once this is assured, project out these reasons and combine all other reasons into
        #   the other_reasons_col columns
        # See MECPODf.get_reasons_subset_from_cpo_df for more info
        assert(len(set(final_reason_cols_i).difference(set(rcpx_0_pd_i.columns.tolist())))==0)
        rcpx_0_pd_i = MECPODf.get_reasons_subset_from_cpo_df(
            cpo_df=rcpx_0_pd_i, 
            reasons_to_include=final_reason_cols_i, 
            combine_others=True, 
            output_combine_others_col=other_reasons_col, 
            SNs_tags=None, 
            is_norm=False, 
            counts_col=nSNs_col, 
            normalize_by_nSNs_included=False, 
            level_0_raw_col = 'counts', 
            level_0_nrm_col = 'counts_norm', 
            cols_to_ignore = [total_counts_col], 
            include_counts_col_in_output=True
        )    
        #--------------------------------------------------
        #--------------------------------------------------
        # Don't want nSNs in each pd individually
        rcpx_0_pd_i = rcpx_0_pd_i.drop(columns=[nSNs_col])
        #-------------------------
        # Add the correct time period name as level 0 of the columns
        rcpx_0_pd_i = Utilities_df.prepend_level_to_MultiIndex(
            df=rcpx_0_pd_i, 
            level_val=final_time_pd_i, 
            level_name=None, 
            axis=1
        )
        #-------------------------
        return rcpx_0_pd_i
    
    @staticmethod
    def build_rcpx_from_evsSum_df_OLD(
        evsSum_df, 
        data_structure_df, 
        prediction_date, 
        td_min, 
        td_max, 
        cr_trans_dict, 
        freq='5D', 
        group_cols=['trsf_pole_nb'], 
        date_col='aep_event_dt', 
        normalize_by_SNs=True, 
        normalize_by_time=True, 
        include_power_down_minus_up=False, 
        regex_patterns_to_remove=['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons=True, 
        xf_meter_cnt_col = 'xf_meter_cnt', 
        events_tot_col = 'events_tot', 
        trsf_pole_nb_col = 'trsf_pole_nb', 
        other_reasons_col = 'Other Reasons',  # From data_structure_df
        total_counts_col = 'total_counts', 
        nSNs_col         = 'nSNs', 
    ):
        r"""
        This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
          
        NOTE: td_min, td_max, and freq must all be in DAYS
        """
        #--------------------------------------------------
        # 0. Need data_structure_df
        #     In general, not all curated reasons will be included in the model.
        #     Typically, 10 commong curated reasons will be included, and all others will be grouped together in "Other Reasons".
        #     Furthermore, some reasons may be combined together, others may be completely removed.
        #     For these reasons, it is beneficial to have some sample data (taken from when the model was created) to utilize 
        #       in structuring the new data in the same fashion.
        #     Additionally, the data will be used to ensure the ordering of columns is correct before the data are fed into 
        #       the model.
        assert(isinstance(data_structure_df, pd.DataFrame) and data_structure_df.shape[0]>0)

        #--------------------------------------------------
        #--------------------------------------------------
        # 1. Build rcpo_0
        #     Construct rcpx_0 by aggregating evsSum_df by group_cols and by freq on date_col
        #--------------------------------------------------
        evsSum_df = evsSum_df.copy()
        #-------------------------
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        assert(len(set(group_cols).difference(set(evsSum_df.columns.tolist())))==0)
        #-------------------------
        # Need to set origin in pd.Grouper to ensure proper grouping
        freq = pd.Timedelta(freq)
        assert((td_max-td_min) % freq==pd.Timedelta(0))
        #-----
        time_grps = pd.date_range(
            start = prediction_date-td_max, 
            end   = prediction_date-td_min, 
            freq  = freq
        )
        time_grps = [(time_grps[i], time_grps[i+1]) for i in range(len(time_grps)-1)]
        assert(len(time_grps) == (td_max-td_min)/pd.Timedelta(freq))
        #-------------------------
        group_freq=pd.Grouper(freq=freq, key=date_col, origin=time_grps[0][0])
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst=evsSum_df.columns.tolist(), 
            regex_pattern=r'cr\d*', 
            ignore_case=False
        )
        #-----
        cols_to_drop = set(evsSum_df.columns.tolist()).difference(
            set(cr_cols+group_cols+[date_col, xf_meter_cnt_col, events_tot_col])
        )
        cols_to_drop = list(cols_to_drop)
        #-------------------------
        # Make sure date_col is datetime object
        evsSum_df[date_col] = pd.to_datetime(evsSum_df[date_col])
        
        #-------------------------
        # No need in wasting time grouping data we won't use
        # So, reduce evsSum_df to only the dates we're interested in 
        evsSum_df = evsSum_df[
            (evsSum_df[date_col] >= prediction_date-td_max) & 
            (evsSum_df[date_col] <= prediction_date-td_min)
        ]


        # All of the cr# columns will be aggregated with np.sum, as will events_tot_col
        # xf_meter_cnt_col will be aggregated using np.max
        agg_dict = {col:np.sum for col in cr_cols+[events_tot_col, xf_meter_cnt_col]}
        agg_dict[xf_meter_cnt_col] = np.max
        #-------------------------
        rcpx_0 = evsSum_df.drop(columns=cols_to_drop).groupby(group_cols+[group_freq]).agg(agg_dict)

        #--------------------------------------------------
        # 2. Grab meter_cnt_per_gp_srs and all_groups
        #--------------------------------------------------
        # Project out the meter count per group, as it will be used later
        #   This information will be stored in the pd.Series object meter_cnt_per_gp_srs, where the index will
        #   contain the group_cols
        meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[xf_meter_cnt_col]].drop_duplicates().set_index(group_cols).squeeze()
        assert(meter_cnt_per_gp_srs.shape[0]==meter_cnt_per_gp_srs.index.nunique())
        meter_cnt_per_gp_srs.name = nSNs_col

        # Will also need the unique groups in rcpx_0
        #   This will be used later (see no_events_pd_i below)
        #   These can be grabbed from the index of rcpx_0 (excluding the date_col level)
        all_groups = rcpx_0.droplevel(date_col, axis=0).index.unique().tolist()

        #--------------------------------------------------
        # 3. Transform rcpx_0 to the form expected by the model
        #     i.e., similar to data_structure_df.
        #     This is essentially just changing rcpo_0 from long form to wide form
        #--------------------------------------------------
        #-------------------------
        # 3a. Build time_pds_rename
        #      Need to convert the time periods, which are currently housed in the date_col index of 
        #        rcpx_0 from their specific dates to the names expected by the model.
        #      In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
        #        dates of the given interval.
        #      These will be converted to the titles contained in final_time_pds below
        #      NOTE: This is probably not 100% necessary, but is useful nonetheless
        #-------------------------
        curr_time_pds = [x[0] for x in time_grps]
        time_pds_rename = OutagePredictor.get_time_pds_rename(
            curr_time_pds=curr_time_pds, 
            td_min=td_min, 
            td_max=td_max, 
            freq=freq
        )
        final_time_pds = list(time_pds_rename.values())
        # final_time_pds should all be found in data_structure_df to help
        #   ensure the alignment between the current data and data used when modelling
        assert(set(final_time_pds).difference(data_structure_df.columns.get_level_values(0).unique())==set())
        #-------------------------
        # Overkill here (since all time windows are of length freq), but something similar will 
        #   be needed if I want to move to non-uniform period lengths
        time_grps_dict = dict()
        assert(len(curr_time_pds) == len(time_grps))
        # Each element in curr_time_pds should match exactly one of the 0th elements 
        #   in time_grps (which is a list of length-2 tuples)
        # Make sure this is so while building time_grps_dict
        for curr_time_pd_i in curr_time_pds:
            time_grp_i = [x for x in time_grps if x[0]==curr_time_pd_i]
            assert(len(time_grp_i)==1)
            assert(curr_time_pd_i not in time_grps_dict.keys())
            time_grps_dict[curr_time_pd_i] = time_grp_i[0]

        #-------------------------
        # 3b. Transform rcpx_0 to the form expected by the model
        #      As stated above, this is essentially just changing rcpo_0 from long form to wide form
        #      This will probably be formalized further in the future (i.e., function(s) developed to handle)
        rename_cols = {
            events_tot_col:total_counts_col, 
            xf_meter_cnt_col:nSNs_col
        }
        rcpx_0=rcpx_0.rename(columns=rename_cols)
        #-------------------------
        total_counts_col = total_counts_col
        nSNs_col         = nSNs_col
        non_reason_cols = [nSNs_col, total_counts_col]
        #------------------------- 
        pd_dfs = []
        for date_pd_i in curr_time_pds:
            # Grab the proper time period name from final_time_pd_i
            final_time_pd_i = time_pds_rename[date_pd_i]
            #-------------------------
            rcpx_0_pd_i = OutagePredictor.project_time_pd_from_rcpx_0_and_prepare(
                rcpx_0                      = rcpx_0, 
                date_pd_i                   = date_pd_i, 
                final_time_pd_i             = final_time_pd_i, 
                data_structure_df           = data_structure_df, 
                meter_cnt_per_gp_srs        = meter_cnt_per_gp_srs, 
                all_groups                  = all_groups, 
                cr_trans_dict               = cr_trans_dict, 
                non_reason_cols             = non_reason_cols, 
                other_reasons_col           = other_reasons_col, 
                group_cols                  = group_cols, 
                date_col                    = date_col, 
                regex_patterns_to_remove    = regex_patterns_to_remove, 
                combine_cpo_df_reasons      = combine_cpo_df_reasons, 
                include_power_down_minus_up = include_power_down_minus_up, 
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col
            )
            #-------------------------
            # Overkill here (since all time windows are of length freq), but something similar will 
            #   be needed if I want to move to non-uniform period lengths
            # One could, e.g., simply divide by length of freq in days
            if normalize_by_time:
                time_grp_i = time_grps_dict[date_pd_i]
                #-----
                days_min_outg_td_window_i = prediction_date - time_grp_i[1]
                days_max_outg_td_window_i = prediction_date - time_grp_i[0]
                #-----
                OutagePredictor.assert_timedelta_is_days(days_min_outg_td_window_i)
                OutagePredictor.assert_timedelta_is_days(days_max_outg_td_window_i)
                #-----
                days_min_outg_td_window_i = days_min_outg_td_window_i.days
                days_max_outg_td_window_i = days_max_outg_td_window_i.days
                #-------------------------
                rcpx_0_pd_i = MECPODf.normalize_rcpo_df_by_time_interval(
                    rcpo_df                 = rcpx_0_pd_i, 
                    days_min_outg_td_window = days_min_outg_td_window_i, 
                    days_max_outg_td_window = days_max_outg_td_window_i, 
                    cols_to_adjust          = None, 
                    SNs_tags                = None, 
                    inplace                 = True
                )
            #-------------------------
            pd_dfs.append(rcpx_0_pd_i)

        # Make sure all dfs in pd_dfs look correct
        shape_0 = pd_dfs[0].shape
        index_0 = pd_dfs[0].index
        for i in range(len(pd_dfs)):
            if i==0:
                continue
            assert(pd_dfs[i].shape==shape_0)
            assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
            #-----
            # Aligning the indices is not strictly necessary, as pd.concat should handle that
            # But, it's best to be safe
            pd_dfs[i] = pd_dfs[i].loc[index_0]

        # Build rcpx_final by combining all dfs in pd_dfs
        rcpx_final = pd.concat(pd_dfs, axis=1)

        # Include back in the number of SNs per group (from meter_cnt_per_gp_srs)
        rcpx_final=rcpx_final.merge(
            meter_cnt_per_gp_srs.to_frame(name=(nSNs_col, nSNs_col)), 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        # Sanity check on the merge
        assert(rcpx_final[nSNs_col].notna().all().all())

        #--------------------------------------------------
        # 4. Normalize by nSNs
        #--------------------------------------------------
        if normalize_by_SNs:
            # Kind of silly, but below I cannot simply use 'rcpx_final[final_time_pds] = ...'
            #   This will result in: "ValueError: Columns must be same length as key", because final_time_pds
            #   has only, e.g., 6 elements but rcpx_final[final_time_pds] contains, e.g., 72 columns
            # Instead, must use 'rcpx_final[rcpx_final[final_time_pds].columns] = ..'
            rcpx_final[rcpx_final[final_time_pds].columns] = rcpx_final[final_time_pds].divide(rcpx_final[(nSNs_col, nSNs_col)], axis=0)

        #--------------------------------------------------
        return rcpx_final
    

    @staticmethod
    def project_time_pd_from_rcpx_0_and_prepare(
        rcpx_0, 
        date_pd_i, 
        final_time_pd_i, 
        data_structure_df, 
        meter_cnt_per_gp_srs, 
        all_groups, 
        cr_trans_dict, 
        non_reason_cols, 
        other_reasons_col, 
        group_cols=['trsf_pole_nb'], 
        date_col='aep_event_dt', 
        regex_patterns_to_remove=['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons=True, 
        include_power_down_minus_up=False, 
        total_counts_col = 'total_counts', 
        nSNs_col = 'nSNs', 
    ):
        r"""
        """
        #-------------------------
        # Get the expected columns for this time period from data_structure_df
        final_reason_cols_i = data_structure_df[final_time_pd_i].columns.tolist()
        final_reason_cols_i = [x for x in final_reason_cols_i if x not in non_reason_cols+[other_reasons_col]]
        #-------------------------
        # Project out the current time period (date_pd_i) from rcpx_0 by selecting the appropriate
        #   values from the date_col index
        rcpx_0_pd_i = rcpx_0[rcpx_0.index.get_level_values(date_col)==date_pd_i].copy()
        rcpx_0_pd_i = rcpx_0_pd_i.droplevel(date_col, axis=0)
        #-------------------------
        # Make sure all groups (typically trsf_pole_nbs) have an entry in rcpx_0_pd_i:
        #   If a group didn't register any events in a given time period, it will not be included in the projection.
        #   However, the final format requires each group have entries for each time period
        #   Therefore, we identify the groups missing from rcpx_0_pd_i (no_events_pd_i) and add approriate rows
        #     containing all 0 values for the counts
        # NOTE: If group_cols contains more than one element, the index will be a MultiIndex (equal to the group_cols)
        #         and will need to be treated slightly different
        no_events_pd_i = list(set(all_groups).difference(set(rcpx_0_pd_i.index.unique())))
        if len(group_cols)==1:
            no_ev_idx = no_events_pd_i
        else:
            no_ev_idx = pd.MultiIndex.from_tuples(no_events_pd_i)
        no_events_pd_i_df = pd.DataFrame(
            columns=rcpx_0.columns, 
            index=no_ev_idx, 
            data=np.zeros((len(no_events_pd_i), rcpx_0.shape[1]))
        )
        no_events_pd_i_df.index.names = rcpx_0_pd_i.index.names
        #-------------------------
        # Use meter_cnt_per_gp_srs to fill the nSNs_col column in no_events_pd_i_df (since simply full of 0s)
        # NOTE: This is probably not strictly necessary, as the nSNs_col column won't be used here,
        #         since the data are not normalized.
        meter_cnt_per_gp_srs_pd_i = meter_cnt_per_gp_srs[meter_cnt_per_gp_srs.index.get_level_values(date_col)==date_pd_i].droplevel(level=date_col, axis=0)
        #-----
        # Use meter_cnt_per_gp_srs to fill the nSNs_col column in no_events_pd_i_df (since simply full of 0s)
        # NOTE: This is probably not strictly necessary, as the nSNs_col column won't be used here,
        #         since the data are not normalized.
        meter_cnt_per_gp_srs_pd_i = meter_cnt_per_gp_srs[meter_cnt_per_gp_srs.index.get_level_values(date_col)==date_pd_i].droplevel(level=date_col, axis=0)
        #-----
        # If a group did not register any events for date_pd_i, then it will be absent from meter_cnt_per_gp_srs_pd_i
        # In such cases, use the average number of meters per group from all other available date periods
        gps_w_missing_meter_cnts = list(set(all_groups).difference(set(meter_cnt_per_gp_srs_pd_i.index.tolist())))
        meter_cnt_per_gp_srs_pd_i = pd.concat([
            meter_cnt_per_gp_srs_pd_i, 
            meter_cnt_per_gp_srs[meter_cnt_per_gp_srs.index.get_level_values(0).isin(gps_w_missing_meter_cnts)].groupby(group_cols).mean()
        ])
        assert(set(meter_cnt_per_gp_srs_pd_i.index.get_level_values(0).tolist()).symmetric_difference(set(all_groups))==set())
        assert(meter_cnt_per_gp_srs_pd_i.shape[0]==meter_cnt_per_gp_srs_pd_i.index.nunique())
        #-------------------------
        no_events_pd_i_df = no_events_pd_i_df.drop(columns=[nSNs_col]).merge(
            meter_cnt_per_gp_srs_pd_i, 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        # Sanity check on the merge
        assert(no_events_pd_i_df[nSNs_col].notna().all())
        #-----
        # Combine rcpx_0_pd_i and no_events_pd_i_df
        assert(len(set(rcpx_0_pd_i.columns).symmetric_difference(set(no_events_pd_i_df.columns)))==0)
        no_events_pd_i_df = no_events_pd_i_df[rcpx_0_pd_i.columns]
        rcpx_0_pd_i = pd.concat([rcpx_0_pd_i, no_events_pd_i_df])
        #-------------------------
        # Rename the cr# columns to their full curated reasons
        rcpx_0_pd_i=rcpx_0_pd_i.rename(columns=cr_trans_dict)
        #--------------------------------------------------    
        #--------------------------------------------------
        # Any columns without a curated reason (i.e., those with column name = ''), have not been observed
        #   yet in the data, and therefore the sume of the counts should be 0.
        # These empty columns are not needed, so drop
        assert(rcpx_0_pd_i[''].sum().sum()==0)
        rcpx_0_pd_i=rcpx_0_pd_i.drop(columns=[''])  
        #-------------------------
        # Remove any undesired curated reasons (e.g., ['.*cleared.*', '.*Test Mode.*'])
        if regex_patterns_to_remove is not None:
            rcpx_0_pd_i = MECPODf.remove_reasons_from_rcpo_df(
                rcpo_df=rcpx_0_pd_i, 
                regex_patterns_to_remove=regex_patterns_to_remove, 
                ignore_case=True
            )
        #-----
        # After irrelevant cleared and test columns removed, need to recalculate events_tot to accurately
        #   reflect the total number of relevant events
        # Safe(r) to do this calculation in any case, so moved outside of the if block above
        assert(total_counts_col in non_reason_cols)
        rcpx_0_pd_i[total_counts_col] = rcpx_0_pd_i.drop(columns=non_reason_cols).sum(axis=1) 
        #-------------------------
        # Combine similar reasons (e.g., all 'Tamper' type reasons are combined into 1)
        # See MECPODf.combine_cpo_df_reasons for more information
        if combine_cpo_df_reasons:
            rcpx_0_pd_i = MECPODf.combine_cpo_df_reasons(rcpo_df=rcpx_0_pd_i)
        #-------------------------
        # Include the difference in power-up and power-down, if desired (typically turned off) 
        if include_power_down_minus_up:
            rcpx_0_pd_i = MECPODf.delta_cpo_df_reasons(
                rcpo_df=rcpx_0_pd_i, 
                reasons_1='Primary Power Down',
                reasons_2='Primary Power Up',
                delta_reason_name='Power Down Minus Up'
            )
        #-------------------------
        # Make sure rcpx_0_pd_i contains the expected final reason columns.
        # Once this is assured, project out these reasons and combine all other reasons into
        #   the other_reasons_col columns
        # See MECPODf.get_reasons_subset_from_cpo_df for more info
        assert(len(set(final_reason_cols_i).difference(set(rcpx_0_pd_i.columns.tolist())))==0)
        rcpx_0_pd_i = MECPODf.get_reasons_subset_from_cpo_df(
            cpo_df=rcpx_0_pd_i, 
            reasons_to_include=final_reason_cols_i, 
            combine_others=True, 
            output_combine_others_col=other_reasons_col, 
            SNs_tags=None, 
            is_norm=False, 
            counts_col=nSNs_col, 
            normalize_by_nSNs_included=False, 
            level_0_raw_col = 'counts', 
            level_0_nrm_col = 'counts_norm', 
            cols_to_ignore = [total_counts_col], 
            include_counts_col_in_output=True
        )    
        #--------------------------------------------------
        #--------------------------------------------------
        # Add the correct time period name as level 0 of the columns
        rcpx_0_pd_i = Utilities_df.prepend_level_to_MultiIndex(
            df=rcpx_0_pd_i, 
            level_val=final_time_pd_i, 
            level_name=None, 
            axis=1
        )
        #-------------------------
        return rcpx_0_pd_i
    

    @staticmethod
    def build_rcpx_from_evsSum_df(
        evsSum_df, 
        data_structure_df, 
        prediction_date, 
        td_min, 
        td_max, 
        cr_trans_dict, 
        freq='5D', 
        group_cols=['trsf_pole_nb'], 
        date_col='aep_event_dt', 
        normalize_by_SNs=True, 
        normalize_by_time=True, 
        include_power_down_minus_up=False, 
        regex_patterns_to_remove=['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons=True, 
        xf_meter_cnt_col = 'xf_meter_cnt', 
        events_tot_col = 'events_tot', 
        trsf_pole_nb_col = 'trsf_pole_nb', 
        other_reasons_col = 'Other Reasons',  # From data_structure_df
        total_counts_col = 'total_counts', 
        nSNs_col         = 'nSNs', 
    ):
        r"""
        This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
          
        NOTE: td_min, td_max, and freq must all be in DAYS
        """
        #--------------------------------------------------
        # 0. Need data_structure_df
        #     In general, not all curated reasons will be included in the model.
        #     Typically, 10 commong curated reasons will be included, and all others will be grouped together in "Other Reasons".
        #     Furthermore, some reasons may be combined together, others may be completely removed.
        #     For these reasons, it is beneficial to have some sample data (taken from when the model was created) to utilize 
        #       in structuring the new data in the same fashion.
        #     Additionally, the data will be used to ensure the ordering of columns is correct before the data are fed into 
        #       the model.
        assert(isinstance(data_structure_df, pd.DataFrame) and data_structure_df.shape[0]>0)

        #--------------------------------------------------
        #--------------------------------------------------
        # 1. Build rcpo_0
        #     Construct rcpx_0 by aggregating evsSum_df by group_cols and by freq on date_col
        #--------------------------------------------------
        evsSum_df = evsSum_df.copy()
        #-------------------------
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        assert(len(set(group_cols).difference(set(evsSum_df.columns.tolist())))==0)
        #-------------------------
        # Need to set origin in pd.Grouper to ensure proper grouping
        freq = pd.Timedelta(freq)
        assert((td_max-td_min) % freq==pd.Timedelta(0))
        #-----
        time_grps = pd.date_range(
            start = prediction_date-td_max, 
            end   = prediction_date-td_min, 
            freq  = freq
        )
        time_grps = [(time_grps[i], time_grps[i+1]) for i in range(len(time_grps)-1)]
        assert(len(time_grps) == (td_max-td_min)/pd.Timedelta(freq))
        #-------------------------
        group_freq=pd.Grouper(freq=freq, key=date_col, origin=time_grps[0][0])
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst=evsSum_df.columns.tolist(), 
            regex_pattern=r'cr\d*', 
            ignore_case=False
        )
        #-----
        cols_to_drop = set(evsSum_df.columns.tolist()).difference(
            set(cr_cols+group_cols+[date_col, xf_meter_cnt_col, events_tot_col])
        )
        cols_to_drop = list(cols_to_drop)
        #-------------------------
        # Make sure date_col is datetime object
        evsSum_df[date_col] = pd.to_datetime(evsSum_df[date_col])
        
        #-------------------------
        # No need in wasting time grouping data we won't use
        # So, reduce evsSum_df to only the dates we're interested in 
        evsSum_df = evsSum_df[
            (evsSum_df[date_col] >= prediction_date-td_max) & 
            (evsSum_df[date_col] <= prediction_date-td_min)
        ]


        # All of the cr# columns will be aggregated with np.sum, as will events_tot_col
        # xf_meter_cnt_col will be aggregated using np.max
        agg_dict = {col:np.sum for col in cr_cols+[events_tot_col, xf_meter_cnt_col]}
        agg_dict[xf_meter_cnt_col] = np.max
        #-------------------------
        rcpx_0 = evsSum_df.drop(columns=cols_to_drop).groupby(group_cols+[group_freq]).agg(agg_dict)

        #--------------------------------------------------
        # 2. Grab meter_cnt_per_gp_srs and all_groups
        #--------------------------------------------------
        # Project out the meter count per group, as it will be used later
        #   This information will be stored in the pd.Series object meter_cnt_per_gp_srs, where the index will
        #   contain the group_cols
        meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[date_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[date_col]).squeeze()
        assert(meter_cnt_per_gp_srs.shape[0]==meter_cnt_per_gp_srs.index.nunique())
        meter_cnt_per_gp_srs.name = nSNs_col

        # Will also need the unique groups in rcpx_0
        #   This will be used later (see no_events_pd_i below)
        #   These can be grabbed from the index of rcpx_0 (excluding the date_col level)
        all_groups = rcpx_0.droplevel(date_col, axis=0).index.unique().tolist()

        #--------------------------------------------------
        # 3. Transform rcpx_0 to the form expected by the model
        #     i.e., similar to data_structure_df.
        #     This is essentially just changing rcpo_0 from long form to wide form
        #--------------------------------------------------
        #-------------------------
        # 3a. Build time_pds_rename
        #      Need to convert the time periods, which are currently housed in the date_col index of 
        #        rcpx_0 from their specific dates to the names expected by the model.
        #      In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
        #        dates of the given interval.
        #      These will be converted to the titles contained in final_time_pds below
        #      NOTE: This is probably not 100% necessary, but is useful nonetheless
        #-------------------------
        curr_time_pds = [x[0] for x in time_grps]
        time_pds_rename = OutagePredictor.get_time_pds_rename(
            curr_time_pds=curr_time_pds, 
            td_min=td_min, 
            td_max=td_max, 
            freq=freq
        )
        final_time_pds = list(time_pds_rename.values())
        # final_time_pds should all be found in data_structure_df to help
        #   ensure the alignment between the current data and data used when modelling
        assert(set(final_time_pds).difference(data_structure_df.columns.get_level_values(0).unique())==set())
        #-------------------------
        # Overkill here (since all time windows are of length freq), but something similar will 
        #   be needed if I want to move to non-uniform period lengths
        time_grps_dict = dict()
        assert(len(curr_time_pds) == len(time_grps))
        # Each element in curr_time_pds should match exactly one of the 0th elements 
        #   in time_grps (which is a list of length-2 tuples)
        # Make sure this is so while building time_grps_dict
        for curr_time_pd_i in curr_time_pds:
            time_grp_i = [x for x in time_grps if x[0]==curr_time_pd_i]
            assert(len(time_grp_i)==1)
            assert(curr_time_pd_i not in time_grps_dict.keys())
            time_grps_dict[curr_time_pd_i] = time_grp_i[0]

        #-------------------------
        # 3b. Transform rcpx_0 to the form expected by the model
        #      As stated above, this is essentially just changing rcpo_0 from long form to wide form
        #      This will probably be formalized further in the future (i.e., function(s) developed to handle)
        rename_cols = {
            events_tot_col:total_counts_col, 
            xf_meter_cnt_col:nSNs_col
        }
        rcpx_0=rcpx_0.rename(columns=rename_cols)
        #-------------------------
        total_counts_col = total_counts_col
        nSNs_col         = nSNs_col
        non_reason_cols = [nSNs_col, total_counts_col]
        #------------------------- 
        pd_dfs = []
        for date_pd_i in curr_time_pds:
            # Grab the proper time period name from final_time_pd_i
            final_time_pd_i = time_pds_rename[date_pd_i]
            #-------------------------
            rcpx_0_pd_i = OutagePredictor.project_time_pd_from_rcpx_0_and_prepare(
                rcpx_0                      = rcpx_0, 
                date_pd_i                   = date_pd_i, 
                final_time_pd_i             = final_time_pd_i, 
                data_structure_df           = data_structure_df, 
                meter_cnt_per_gp_srs        = meter_cnt_per_gp_srs, 
                all_groups                  = all_groups, 
                cr_trans_dict               = cr_trans_dict, 
                non_reason_cols             = non_reason_cols, 
                other_reasons_col           = other_reasons_col, 
                group_cols                  = group_cols, 
                date_col                    = date_col, 
                regex_patterns_to_remove    = regex_patterns_to_remove, 
                combine_cpo_df_reasons      = combine_cpo_df_reasons, 
                include_power_down_minus_up = include_power_down_minus_up, 
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col
            )
            #-------------------------
            # Overkill here (since all time windows are of length freq), but something similar will 
            #   be needed if I want to move to non-uniform period lengths
            # One could, e.g., simply divide by length of freq in days
            if normalize_by_time:
                time_grp_i = time_grps_dict[date_pd_i]
                #-----
                days_min_outg_td_window_i = prediction_date - time_grp_i[1]
                days_max_outg_td_window_i = prediction_date - time_grp_i[0]
                #-----
                OutagePredictor.assert_timedelta_is_days(days_min_outg_td_window_i)
                OutagePredictor.assert_timedelta_is_days(days_max_outg_td_window_i)
                #-----
                days_min_outg_td_window_i = days_min_outg_td_window_i.days
                days_max_outg_td_window_i = days_max_outg_td_window_i.days
                #-------------------------
                nSNs_b4 = rcpx_0_pd_i[(final_time_pd_i, nSNs_col)].copy()
                #-----
                rcpx_0_pd_i = MECPODf.normalize_rcpo_df_by_time_interval(
                    rcpo_df                 = rcpx_0_pd_i, 
                    days_min_outg_td_window = days_min_outg_td_window_i, 
                    days_max_outg_td_window = days_max_outg_td_window_i, 
                    cols_to_adjust          = None, 
                    SNs_tags                = None, 
                    inplace                 = True
                )
                #-----
                assert(rcpx_0_pd_i[(final_time_pd_i, nSNs_col)].equals(nSNs_b4))
            #-------------------------
            if normalize_by_SNs:
                cols_to_norm = [x for x in rcpx_0_pd_i.columns if x[1]!=nSNs_col]
                rcpx_0_pd_i[cols_to_norm] = rcpx_0_pd_i[cols_to_norm].divide(rcpx_0_pd_i[(final_time_pd_i, nSNs_col)], axis=0)
            #-------------------------
            pd_dfs.append(rcpx_0_pd_i)

        # Make sure all dfs in pd_dfs look correct
        shape_0 = pd_dfs[0].shape
        index_0 = pd_dfs[0].index
        for i in range(len(pd_dfs)):
            if i==0:
                continue
            assert(pd_dfs[i].shape==shape_0)
            assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
            #-----
            # Aligning the indices is not strictly necessary, as pd.concat should handle that
            # But, it's best to be safe
            pd_dfs[i] = pd_dfs[i].loc[index_0]

        # Build rcpx_final by combining all dfs in pd_dfs
        rcpx_final = pd.concat(pd_dfs, axis=1)

        #-------------------------
        # For the number of SNs per group, use the average number of 
        #   SNs for each group across all time periods
        nSNs_cols = [x for x in rcpx_final.columns if x[1]==nSNs_col]
        #-----
        # There should be one nSNs col for each time period
        assert(len(nSNs_cols)==len(final_time_pds))
        #-----
        rcpx_final[(nSNs_col, nSNs_col)] = rcpx_final[nSNs_cols].mean(axis=1)
        #-----
        # Sanity check on the merge
        assert(rcpx_final[nSNs_col].notna().all().all())
        #-----
        # Drop the time-period-specific nSNs columns
        rcpx_final = rcpx_final.drop(columns=nSNs_cols)
        #--------------------------------------------------
        return rcpx_final
    
    
    @staticmethod
    def build_eemsp_df(
        trsf_pole_nbs, 
        date_range, 
        conn_aws=None,  
        mult_strategy='agg', 
        include_n_eemsp=True, 
        cols_of_interest_eemsp=None, 
        numeric_cols = ['kva_size'], 
        dt_cols = ['install_dt', 'removal_dt'], 
        ignore_cols = ['serial_nb'], 
        batch_size=10000, 
        verbose=True, 
        n_update=10, 
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(date_range, list) and len(date_range)==2)
        #-------------------------
        if conn_aws is None:
            conn_aws = Utilities.get_athena_prod_aws_connection()
        #-------------------------
        #--------------------------------------------------
        # 1. Grab eemsp_df for trsf_pole_nbs in date_range
        #--------------------------------------------------
        dflt_cols_of_interest_eemsp = [
            'location_nb', 
            'mfgr_nm', 
            'install_dt', 
            'last_trans_desc', 
            'eqtype_id', 
            'coolant', 
            'info', 
            'kva_size',
            'phase_cnt', 
            'prim_voltage', 
            'protection', 
            'pru_number', 
            'sec_voltage', 
            'special_char', 
            'taps', 
            'xftype'
        ]
        if cols_of_interest_eemsp is None:
            cols_of_interest_eemsp = dflt_cols_of_interest_eemsp
        if 'location_nb' not in cols_of_interest_eemsp:
            cols_of_interest_eemsp.append('location_nb')
        #-----
        cols_of_interest_eemsp_full = cols_of_interest_eemsp + ['latest_status', 'removal_dt', 'serial_nb']
        cols_of_interest_eemsp_full = list(set(cols_of_interest_eemsp_full))
        #-------------------------
        #-------------------------
        batch_idxs = Utilities.get_batch_idx_pairs(
            n_total=len(trsf_pole_nbs), 
            batch_size=batch_size, 
            absorb_last_pair_pct=None
        )
        n_batches = len(batch_idxs)
        eemsp_df = pd.DataFrame()
        #-----
        if verbose:
            print(f'n_coll = {len(trsf_pole_nbs)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches = {n_batches}')
        #-----
        for i, batch_i in enumerate(batch_idxs):
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #----- 
            sql_EEMSP_i = """
            SELECT {} 
            FROM meter_events.eems_transformer_nameplate
            WHERE location_nb IN ({})
            AND install_dt <= '{}'
            AND (removal_dt IS NULL OR removal_dt > '{}')
            """.format(
                Utilities_sql.join_list(cols_of_interest_eemsp_full, quotes_needed=False), 
                Utilities_sql.join_list(trsf_pole_nbs[i_beg:i_end], quotes_needed=True), 
                date_range[0], 
                date_range[1]
            )
            eemsp_df_i = pd.read_sql_query(sql_EEMSP_i, conn_aws)
            #-----
            if eemsp_df.shape[0]>0:
                assert(all(eemsp_df_i.columns==eemsp_df.columns))
            eemsp_df = pd.concat([eemsp_df, eemsp_df_i], axis=0, ignore_index=True)    

        #-------------------------
        eemsp_df = eemsp_df.reset_index(drop=True)
        #--------------------------------------------------
        # 2. Reduce down eemsp_df so there is a single entry for each transformer
        #     EEMSP.reduce1_eemsp_for_outg_trsf reduces eemsp_df down to contain only entries for transformers which were active 
        #       during the date(s) in question.
        #     No need to run EEMSP.reduce1_eemsp_for_outg_trsf for this case, as all share the same date restrictions which were 
        #       already imposed in sql_EEMSP.
        #     (For model development/training, this step would be necessary, as the data utilized there have many different 
        #       date restrictions, and eemsp_df cannot simply be built with the date restrictions)
        #     
        #     EEMSP.reduce2_eemsp_for_outg_trsf futher reduces eemsp_df down so there is a single entry for each transformer.
        #     How exactly this is achieved is dictated mainly by the "mult_strategy" parameter
        #--------------------------------------------------
        numeric_cols = [x for x in numeric_cols if x in eemsp_df.columns.tolist()]
        dt_cols      = [x for x in dt_cols if x in eemsp_df.columns.tolist()]
        ignore_cols  = [x for x in ignore_cols if x in eemsp_df.columns.tolist()]
        #-----
        eemsp_df_reduce2 = EEMSP.reduce2_eemsp_for_outg_trsf(
            df_eemsp=eemsp_df, 
            mult_strategy=mult_strategy, 
            include_n_eemsp=include_n_eemsp,  
            grp_by_cols=['location_nb'], 
            numeric_cols = numeric_cols, 
            dt_cols = dt_cols, 
            ignore_cols = ignore_cols, 
            cat_cols_as_strings=True
        )
        #-------------------------
        # No matter of the mult_strategy used, at this point eemsp_df_reduce2 should only have a single
        #   entry for each location_nb
        assert(all(eemsp_df_reduce2[['location_nb']].value_counts()==1))

        #----------------------------------------------------------------------------------------------------
        # Clean up eemsp_df_reduce2 and merge with rcpx_final
        #--------------------------------------------------
        # cols_to_drop was important in the past, as there was a necessary column which was added (outg_rec_nb), and
        #   simply taking eemsp_df_reduce2[cols_of_interest_eemsp] would have excluded new column(s)
        # Keeping the code in place is safe, and it will protect in the future if any new columns are added
        cols_to_drop = list(set(cols_of_interest_eemsp_full).difference(set(cols_of_interest_eemsp)))
        cols_to_drop = [x for x in cols_to_drop if x in eemsp_df_reduce2.columns]
        print(len(cols_to_drop))
        if len(cols_to_drop)>0:
            eemsp_df_reduce2 = eemsp_df_reduce2.drop(columns=cols_to_drop)
        #-------------------------
        assert(eemsp_df_reduce2.shape[0]==eemsp_df_reduce2.groupby(['location_nb']).ngroups)
        #-------------------------
        # Make all EEMSP columns (except n_eemsp) uppercase to match what was done in model development (where EEMSP)
        #   data were grabbed from the Oracle database, and columns were all uppercase)
        eemsp_df_reduce2 = Utilities_df.make_all_column_names_uppercase(eemsp_df_reduce2, cols_to_exclude=['n_eemsp'])
        #-------------------------
        return eemsp_df_reduce2
    
    @staticmethod
    def build_eemsp_df_and_merge_rcpx( 
        rcpx_df, 
        trsf_pole_nbs, 
        date_range, 
        merge_on_rcpx=['index_0'], 
        merge_on_eems=['LOCATION_NB'], 
        conn_aws=None, 
        mult_strategy='agg', 
        include_n_eemsp=True, 
        cols_of_interest_eemsp=None, 
        numeric_cols = ['kva_size'], 
        dt_cols = ['install_dt', 'removal_dt'], 
        ignore_cols = ['serial_nb'], 
        batch_size=10000, 
        verbose=True, 
        n_update=10, 
    ):
        r"""
        """
        #-------------------------
        eemsp_df = OutagePredictor.build_eemsp_df( 
            trsf_pole_nbs          = trsf_pole_nbs, 
            date_range             = date_range, 
            conn_aws               = conn_aws, 
            mult_strategy          = mult_strategy, 
            include_n_eemsp        = include_n_eemsp, 
            cols_of_interest_eemsp = cols_of_interest_eemsp, 
            numeric_cols           = numeric_cols, 
            dt_cols                = dt_cols, 
            ignore_cols            = ignore_cols, 
            batch_size             = batch_size, 
            verbose                = verbose, 
            n_update               = n_update 
        )
        #-------------------------
        #-------------------------
        og_n_rows = rcpx_df.shape[0]
        rcpx_df = EEMSP.merge_rcpx_with_eemsp(
            df_rcpx=rcpx_df, 
            df_eemsp=eemsp_df, 
            merge_on_rcpx=merge_on_rcpx, 
            merge_on_eems=merge_on_eems, 
            set_index=True, 
            drop_eemsp_merge_cols=True
        )
        #-------------------------
        # eemsp_df should only have one entry per transformer, and due to inner merge, rcpx_df
        #   the merge can only decrease the number of entries in rcpx_df
        assert(rcpx_df.shape[0]<=og_n_rows)
        #-------------------------
        return rcpx_df, eemsp_df
    
    
    @staticmethod
    def convert_install_dt_to_years(
        rcpx_df, 
        prediction_date, 
        install_dt_col=('EEMSP_0', 'INSTALL_DT'), 
        assert_col_found=False
    ):
        r"""
        Convert the install_dt_col from date to age relative to prediction_date
        """
        #-------------------------
        if assert_col_found:
            assert(install_dt_col in rcpx_df.columns.tolist())
        #-----
        if install_dt_col in rcpx_df.columns.tolist():
            rcpx_df[install_dt_col] = (prediction_date-rcpx_df[install_dt_col]).dt.total_seconds()/(60*60*24*365)
        #-----
        return rcpx_df

    @staticmethod
    def add_predict_month_to_rcpx_df(
        rcpx_df, 
        prediction_date, 
        month_col=('dummy_lvl_0', 'outg_month'), 
        dummy_col_levels_prefix='dummy'
    ):
        r"""
        month_col:
            Should either be a string or a tuple
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(month_col, [str, tuple]))
        if isinstance(month_col, str):
            if rcpx_df.columns.nlevels>1:
                n_levels_to_add = rcpx_df.columns.nlevels - 1
                month_col = [month_col]
                # With each iteration, prepending a new level from n_levels_to_add-1 to 0
                for i_new_lvl in range(n_levels_to_add):
                    month_col.insert(0, f'{dummy_col_levels_prefix}{(n_levels_to_add-1)-i_new_lvl}')
                assert(len(month_col)==rcpx_df.columns.nlevels)
                month_col = tuple(month_col)
        else:
            assert(len(month_col)==rcpx_df.columns.nlevels)
        #-------------------------
        rcpx_df[month_col] = prediction_date.month
        #-------------------------
        return rcpx_df

    @staticmethod
    def assert_rcpx_has_correct_form(
        rcpx_df, 
        data_structure_df
    ):
        r"""
        Make sure rcpx_df has the correct columns in the correct order
        """
        #-------------------------
        assert(len(set(data_structure_df.columns).symmetric_difference(set(rcpx_df.columns)))==0)
        rcpx_df=rcpx_df[data_structure_df.columns]
        #-------------------------
        return rcpx_df
    
    @staticmethod
    def build_X_test(
        rcpx_df, 
        data_structure_df, 
        eemsp_args=True, 
        scaler=None
    ):
        r"""
        Construct X_test from rcpx_df

        eemsp_args:
            Can be False or dict
            If False or empty dict:
                Do not include eemsp
            If non-empty dict:
                Include eemsp with arguments given in eemsp_args
                AT A MINIMUM, MUST INCLUDE key 'eemsp_enc'
                Possible keys:
                    eemsp_enc
                    numeric_cols:
                        default = ['KVA_SIZE', 'INSTALL_DT']
                    EEMSP_col = 'EEMSP_0'
        """
        #--------------------------------------------------
        X_test = rcpx_df.copy()
        # Make sure everything looks good
        X_test = OutagePredictor.assert_rcpx_has_correct_form(
            rcpx_df=X_test, 
            data_structure_df=data_structure_df
        )
        #--------------------------------------------------
        # 1. If EEMSP included, run EEMSP encoder (eemsp_enc)
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(eemsp_args, [bool, dict]))
        if isinstance(eemsp_args, bool):
            assert(eemsp_args==False)
        if eemsp_args and eemsp_args['eemsp_enc'] is not None:
            # Non-empty dict, MUST INCLUDE at a minimum eemsp_enc
            assert('eemsp_enc' in eemsp_args.keys())
            eemsp_enc = eemsp_args['eemsp_enc']
            #-----
            EEMSP_col = eemsp_args.get('EEMSP_col', 'EEMSP_0')
            assert(EEMSP_col in data_structure_df.columns)
            assert(EEMSP_col in X_test.columns)
            cols_to_encode = data_structure_df[EEMSP_col].columns
            #-----
            numeric_cols = eemsp_args.get('numeric_cols', ['KVA_SIZE', 'INSTALL_DT'])
            numeric_cols = [x for x in numeric_cols if x in cols_to_encode]
            #-----
            cols_to_encode = [x for x in cols_to_encode if x not in numeric_cols]
            #-------------------------
            assert(len(set(eemsp_enc.feature_names_in_).symmetric_difference(cols_to_encode))==0)
            assert(set(X_test[EEMSP_col].columns).difference(eemsp_enc.feature_names_in_)==set(numeric_cols))
            #-----
            # Make sure cols_to_encode found in X_test, and convert to MultiIndex versions if needed
            cols_to_encode = [Utilities_df.find_single_col_in_multiindex_cols(df=X_test, col=x) for x in cols_to_encode] 
            X_test[cols_to_encode] = X_test[cols_to_encode].astype(str)
            X_test[cols_to_encode] = eemsp_enc.transform(X_test[cols_to_encode].droplevel(0, axis=1))
            
        #--------------------------------------------------
        # 2. Make sure everything still looks good
        #     NOTE: Perform before scaler run because scaler strips the columns off the data
        #--------------------------------------------------
        X_test = OutagePredictor.assert_rcpx_has_correct_form(
            rcpx_df=X_test, 
            data_structure_df=data_structure_df
        )

        #--------------------------------------------------
        # 3. If scaler included, run
        #--------------------------------------------------
        if scaler is not None:
            X_test = scaler.transform(X_test)

        #--------------------------------------------------
        return X_test
    
    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def initialize_data(
        self, 
        evsSum_sql_fcn=AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs=None, 

        # THESE SHOULD BE ADDED TO MODEL_SUMMARY_DICT AND AUTOMATICALLY EXTRACTED!!!!!
        freq                        = '5D', 
        group_cols                  = ['trsf_pole_nb'], 
        date_col                    = 'aep_event_dt', 
        normalize_by_SNs            = True, 
        include_power_down_minus_up = False, 
        regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons      = True, 
        include_n_eemsp             = True
    ):
        r"""
        Build/initialize all data needed for predictions.
        NOTE: trsf_pole_nbs and model_dir must be set prior to this operation!

        What does this function do?
            - Build events summary (self.evsSum)
            - Build reason counts per x df (self.rcpx_df)
            - If including EEMSP, build and merge with rcpx (self.eemsp_df)
            - If needed, include month
            - Make sure final form of rcpx_df agrees with self.data_structure_df
            - Build/set self.X_test
        """
        #-------------------------
        # 1A. Make sure the transformer pole numbers have been set (whether done explicitly with
        #      set_trsf_pole_nbs or via sql with set_trsf_pole_nbs_from_sql)
        assert(self.trsf_pole_nbs is not None and len(self.trsf_pole_nbs)>0)

        # 1B. Make sure the model_dir has been set (and, through set_model_dir, all needed components
        #       have been extracted)
        assert(self.model_dir is not None and os.path.exists(self.model_dir))

        #-------------------------
        # 2. Build events summary (self.evsSum)
        self.build_events_summary(
                evsSum_sql_fcn=evsSum_sql_fcn,  
                evsSum_sql_kwargs=evsSum_sql_kwargs, 
                init_df_in_constructor=True, 
                save_args=False
            )

        #-------------------------
        # 3. Build reason counts per x df (self.rcpx_df)
        self.rcpx_df = OutagePredictor.build_rcpx_from_evsSum_df(
            evsSum_df                   = self.evsSum_df, 
            data_structure_df           = self.data_structure_df, 
            prediction_date             = self.prediction_date, 
            td_min                      = self.idk_name_2, 
            td_max                      = self.idk_name_1, 
            cr_trans_dict               = self.cr_trans_dict, 
            freq                        = freq, 
            group_cols                  = group_cols, 
            date_col                    = date_col, 
            normalize_by_SNs            = normalize_by_SNs, 
            normalize_by_time           = True, 
            include_power_down_minus_up = include_power_down_minus_up, 
            regex_patterns_to_remove    = regex_patterns_to_remove, 
            combine_cpo_df_reasons      = combine_cpo_df_reasons, 
            xf_meter_cnt_col            = 'xf_meter_cnt', 
            events_tot_col              = 'events_tot', 
            trsf_pole_nb_col            = 'trsf_pole_nb', 
            other_reasons_col           = 'Other Reasons', 
            total_counts_col            = 'total_counts', 
            nSNs_col                    = 'nSNs'
        )

        #-------------------------
        # 4. If including EEMSP, build and merge with rcpx
        if self.merge_eemsp:
            self.rcpx_df, self.eemsp_df = OutagePredictor.build_eemsp_df_and_merge_rcpx( 
                    rcpx_df                = self.rcpx_df, 
                    trsf_pole_nbs          = self.trsf_pole_nbs, 
                    date_range             = self.date_range, 
                    merge_on_rcpx          = ['index_0'], 
                    merge_on_eems          = ['LOCATION_NB'], 
                    conn_aws               = self.conn_aws, 
                    mult_strategy          = self.eemsp_mult_strategy, 
                    include_n_eemsp        = include_n_eemsp, 
                    cols_of_interest_eemsp = None, 
                    numeric_cols           = ['kva_size'], 
                    dt_cols                = ['install_dt', 'removal_dt'], 
                    ignore_cols            = ['serial_nb'], 
                    batch_size             = 10000, 
                    verbose                = True, 
                    n_update               = 10
                )
            #-----
            self.rcpx_df = OutagePredictor.convert_install_dt_to_years(
                rcpx_df          = self.rcpx_df, 
                prediction_date  = self.prediction_date, 
                install_dt_col   = ('EEMSP_0', 'INSTALL_DT'), 
                assert_col_found = False
            )

        #-------------------------
        # 5. Include month?
        if self.include_month:
            self.rcpx_df = OutagePredictor.add_predict_month_to_rcpx_df(
                rcpx_df                 = self.rcpx_df, 
                prediction_date         = self.prediction_date, 
                month_col               = ('dummy_lvl_0', 'outg_month'), 
                dummy_col_levels_prefix = 'dummy'
            )

        #-------------------------
        # 6. Make sure final form of self.rcpx_df agrees with self.data_structure_df
        self.rcpx_df = OutagePredictor.assert_rcpx_has_correct_form(
            rcpx_df           = self.rcpx_df, 
            data_structure_df = self.data_structure_df
        )

        #-------------------------
        # 7. Build X_test
        self.X_test = OutagePredictor.build_X_test(
            rcpx_df           = self.rcpx_df, 
            data_structure_df = self.data_structure_df, 
            eemsp_args        = dict(eemsp_enc=self.eemsp_enc), 
            scaler            = self.scaler
        )
        
    def change_prediction_date(
        self, 
        prediction_date
    ):
        r"""
        """
        #-------------------------
        self.prediction_date = pd.to_datetime(prediction_date)

        # Update self.date_range
        self.set_date_range()

        # Run initialize_data to update all data
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.initialize_data(
            evsSum_sql_fcn=self.evsSum_sql_fcn,  
            evsSum_sql_kwargs=self.evsSum_sql_kwargs, 

            # THESE SHOULD BE ADDED TO MODEL_SUMMARY_DICT AND AUTOMATICALLY EXTRACTED!!!!!
            freq                        = '5D', 
            group_cols                  = ['trsf_pole_nb'], 
            date_col                    = 'aep_event_dt', 
            normalize_by_SNs            = True, 
            include_power_down_minus_up = False, 
            regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
            combine_cpo_df_reasons      = True, 
            include_n_eemsp             = True
        )