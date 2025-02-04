#!/usr/bin/env python

r"""
Holds DOVSAudit class.  See DOVSAudit.DOVSAudit for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import re
from pathlib import Path
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

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
import matplotlib.patheffects as pe

#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from DOVSOutages_SQL import DOVSOutages_SQL
from DOVSOutages import DOVSOutages
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from TableInfos import TableInfo
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Plot_General
#---------------------------------------------------------------------

class DOVSRepeatFinder:
    r"""
    Class to find premises who outage times are classified in multiple DOVS events.
    If desired, can also be built out to identify premises suffering frequent outages.
    """
    def __init__(
        self, 
        mjr_mnr_cause   = None, 
        date_range      = None, 
        states          = None, 
        opcos           = None, 
        CI_NB_min       = None,
        premise_nbs     = None, 
    ):
        r"""
        Initializer
        """
        #--------------------------------------------------
        # Nice to explicitly declare class members first (C++ style)
        self.init_dfs_to = pd.DataFrame(),  # Should be pd.DataFrame() or None
        #-----
        self.dovs_OG     = None
        self.dovs_df     = copy.deepcopy(self.init_dfs_to)
        

        #--------------------------------------------------
        self.dovs_OG = DOVSRepeatFinder.build_dovs_OG(
            mjr_mnr_cause   = mjr_mnr_cause, 
            date_range      = date_range, 
            states          = states, 
            opcos           = opcos, 
            CI_NB_min       = CI_NB_min,
            premise_nbs     = premise_nbs, 
        )
        #-------------------------
        self.premise_cols = list(set(self.dovs_OG.df.columns.tolist()).intersection(TableInfos.DOVS_PREMISE_DIM_TI.columns_full))
        self.premise_cols.remove('OUTG_REC_NB')
        #-----
        self.dovs_gen_cols = [x for x in self.dovs_OG.df.columns.tolist() if x not in self.premise_cols]
        #-------------------------
        dovs_df = DOVSRepeatFinder.build_dovs_df_from_dovs_OG(
            dovs_OG              = self.dovs_OG, 
            xN_col               = 'PREMISE_NB', 
            intrvl_beg_col       = 'DT_OFF_TS_FULL', 
            intrvl_end_col       = 'DT_ON_TS', 
            outg_rec_nb_col      = 'OUTG_REC_NB', 
            overlaps_col         = 'overlaps', 
            overlap_grp_col      = 'overlap_grp', 
            group_index_col      = 'group_index'
        )
        #-----
        self.dovs_df = dovs_df

        #-------------------------
        self.dovs_gen_df = self.build_dovs_gen_df(
                group_index_col  = 'group_index', 
                overlap_grp_col  = 'overlap_grp', 
                addtnl_shift_col = 'addtnl_shift', 
                t_beg_col        = 'DT_OFF_TS_FULL', 
                t_end_col        = 'DT_ON_TS', 
                outg_rec_nb_col  = 'OUTG_REC_NB', 
                xN_col           = 'PREMISE_NB', 
                y_col            = 'y'
            )

    
    @staticmethod
    def build_dovs_OG(
        mjr_mnr_cause   = None, 
        date_range      = None, 
        states          = None, 
        opcos           = None, 
        CI_NB_min       = None,
        premise_nbs     = None, 
    ):
        r"""
        Build outage_df_OG
    
        If one wanted, one could allow build_sql_function and build_sql_function_kwargs to be the input arguments here, allowing
          the function to be much more general.
        One could also allow user to set, e.g., field_to_split, batch_size, etc.
        However, I have opted for ease of use, and have essentially explicitly included a restricted set of kwargs for DOVSOutages_SQL.build_sql_std_outage
        """
        #--------------------------------------------------
        if date_range is None and premise_nbs is None:
            print('!!!!! WARNING !!!!!\n\tWe suggest setting, at a minimum, date_range and/or premise_nbs')
        #--------------------------------------------------
        field_to_split  = 'premise_nbs'
        batch_size      = 1000
        verbose         = True
        n_update        = 10
        if premise_nbs is None:
            field_to_split = None
        #-------------------------    
        build_sql_function = DOVSOutages_SQL.build_sql_std_outage
        build_sql_function_kwargs = dict(
            mjr_mnr_cause              = mjr_mnr_cause, 
            include_premise            = True, 
            date_range                 = date_range, 
            states                     = states, 
            opcos                      = opcos, 
            CI_NB_min                  = CI_NB_min, 
            premise_nbs                = premise_nbs, 
            field_to_split             = field_to_split, 
            batch_size                 = batch_size, 
            verbose                    = verbose, 
            n_update                   = n_update, 
            join_type_DOVS_PREMISE_DIM = 'INNER'
        )
        #--------------------------------------------------
        dovs = DOVSOutages(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = None, 
            init_df_in_constructor    = True, 
            build_sql_function        = build_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs, 
            save_args                 = False, 
            build_consolidated        = False, 
        )
        #--------------------------------------------------
        return dovs


    @staticmethod
    def append_exists_overlapping_events_for_xN_i(
        df_i, 
        intrvl_beg_col       = 'DT_OFF_TS_FULL', 
        intrvl_end_col       = 'DT_ON_TS', 
        overlaps_col         = 'overlaps', 
        overlap_grp_col      = 'overlap_grp', 
        return_overlaps_only = False, 
        no_overlap_grp_val   = -1 # sensible values = -1 or np.nan
    ):
        r"""
        HELPER FUNCTION FOR append_exists_overlapping_events_for_xN_in_df, not intended to be used on own
        """
        #--------------------------------------------------
        df_i = Utilities_df.find_and_append_overlapping_blocks_ids_in_df(
            df                   = df_i, 
            intrvl_beg_col       = intrvl_beg_col, 
            intrvl_end_col       = intrvl_end_col, 
            overlaps_col         = overlaps_col, 
            overlap_grp_col      = overlap_grp_col, 
            return_overlaps_only = return_overlaps_only, 
            no_overlap_grp_val   = no_overlap_grp_val
        )
        return df_i
        

    @staticmethod
    def append_exists_overlapping_events_for_xN_in_df(
        df, 
        xN_col               = 'PREMISE_NB', 
        intrvl_beg_col       = 'DT_OFF_TS_FULL', 
        intrvl_end_col       = 'DT_ON_TS', 
        overlaps_col         = 'overlaps', 
        overlap_grp_col      = 'overlap_grp', 
        return_overlaps_only = False, 
        no_overlap_grp_val    = -1 # sensible values = -1 or np.nan
    ):
        r"""
        Group df by xN_col and apply DOVSRepeatFinder.append_exists_overlapping_events_for_xN_i to each group.
        xN should be, e.g., premise number, serial number, etc.
        """
        #--------------------------------------------------
        # Make sure necessary cols are present
        assert(set([xN_col, intrvl_beg_col, intrvl_end_col]).difference(set(df.columns.tolist()))==set())
    
        #--------------------------------------------------
        # Premises with only one entry cannoy possibly have overlaps, 
        #   so no need to waste resources on checking them
        xN_vcs = df[xN_col].value_counts()
        #-----
        xNs_single = xN_vcs[xN_vcs==1].index.tolist()
        xNs_multpl = xN_vcs[xN_vcs >1].index.tolist()
        #-----
        df_single = df[df[xN_col].isin(xNs_single)].copy()
        df_multpl = df[df[xN_col].isin(xNs_multpl)].copy()
        
        #--------------------------------------------------
        # Set overlaps(_grp)_col for df_single
        df_single[overlaps_col]    = False
        df_single[overlap_grp_col] = no_overlap_grp_val
        
        #--------------------------------------------------
        # Set overlaps(_grp)_col for df_multpl
        df_multpl = df_multpl.groupby(xN_col, group_keys=False, as_index=False)[df_multpl.columns].apply(
            lambda x: DOVSRepeatFinder.append_exists_overlapping_events_for_xN_i(
                df_i                 = x, 
                intrvl_beg_col       = intrvl_beg_col, 
                intrvl_end_col       = intrvl_end_col, 
                overlaps_col         = overlaps_col, 
                overlap_grp_col      = overlap_grp_col, 
                return_overlaps_only = return_overlaps_only, 
                no_overlap_grp_val   = no_overlap_grp_val
            )
        )
    
        #--------------------------------------------------
        if return_overlaps_only:
            return df_multpl
        else:
            assert((df_multpl.columns==df_single.columns).all())
            return_df = pd.concat([df_multpl, df_single])
            return return_df

    

    @staticmethod
    def build_dovs_df_from_dovs_OG(
        dovs_OG, 
        xN_col               = 'PREMISE_NB', 
        intrvl_beg_col       = 'DT_OFF_TS_FULL', 
        intrvl_end_col       = 'DT_ON_TS', 
        outg_rec_nb_col      = 'OUTG_REC_NB', 
        overlaps_col         = 'overlaps', 
        overlap_grp_col      = 'overlap_grp', 
        group_index_col      = 'group_index', 
        set_index            = True
    ):
        r"""
        """
        #--------------------------------------------------
        if dovs_OG.df.shape[0]==0:
            return dovs_OG.df
        #--------------------------------------------------
        dovs_df = DOVSRepeatFinder.append_exists_overlapping_events_for_xN_in_df(
            df                   = dovs_OG.df.copy(), 
            xN_col               = xN_col, 
            intrvl_beg_col       = intrvl_beg_col, 
            intrvl_end_col       = intrvl_end_col, 
            overlaps_col         = overlaps_col, 
            overlap_grp_col      = overlap_grp_col, 
            return_overlaps_only = True
        )
        #-------------------------
        if dovs_df.shape[0]==0:
            return dovs_df
        #--------------------------------------------------
        dovs_df = dovs_df.sort_values(by=[xN_col, intrvl_beg_col, intrvl_end_col, outg_rec_nb_col], ignore_index=True)
        #-----
        # group_index is used for plotting purposes (allowing separation in y for overlapping events)
        dovs_df[group_index_col] = dovs_df.groupby([xN_col, overlap_grp_col]).cumcount()
        #-------------------------
        dovs_df = Utilities_df.move_cols_to_front(df=dovs_df, cols_to_move=[xN_col, overlaps_col, overlap_grp_col])
        #--------------------------------------------------
        return dovs_df


    @staticmethod
    def get_break_pt_idx_groups_OLD(
        df, 
        max_delta_t, 
        pd_beg_col,
        pd_end_col=None
    ):
        r"""
        If one wants to plot df using horizontal breaks, this function will identify the groupings.
        For design case (with DOVS data for individual premise):
            pd_beg_col = 'DT_OFF_TS_FULL'
            pd_end_col = 'DT_ON_TS'
    
        Returns dict object with keys:
            df: the input df, which will now be sorted
            idx_grps: the index groups
    
        pd_beg_col/pd_end_col:
            These essentially define the ordering of the data.
            If pd_end_col is None:
                the differences between pd_beg_col of the current row and that of the previous are evaluated
                    i.e., df[pd_beg_col].diff(periods=1)
            If pd_end_col is not None:
                the differences between pd_beg_col of the current row and pd_end_col of the previous are evaluated
                    i.e., df[pd_beg_col]-df[pd_end_col].shift(periods=1)
                NOTE: For the case pd_beg_col=pd_beg_col, df[pd_beg_col]-df[pd_beg_col].shift(periods=1) reduces to 
                      df[pd_beg_col].diff(periods=1)
        """
        #-------------------------
        # Data must be sorted for this procedure to work
        if pd_end_col is None:
            df = df.sort_values(by=[pd_beg_col])
        else:
            df = df.sort_values(by=[pd_beg_col, pd_end_col])
    
        #--------------------------------------------------
        # break_pt_bool_srs will be used to find where break points need to be placed.
        #-----
        # The .diff(periods=1) operation calculates the difference of a DataFrame element compared with the element in the previous row.
        #   ==> The 0th element is always NaN/NaT
        #   ==> The 0th element of the df['DT_OFF_TS_FULL'].diff(periods=1)>max_delta_t operation will always be False
        # The .diff(periods=1)>max_delta_t operation therefore determines whether the row is outside max_delta_t (==> in different subplot)
        #   of the previous row (False if inside threshold/subplot, True if outside threshold/subplot)
        #   ==> True values of break_pt_bool_srs identify the LEFT point of the groups/subplots
        # NOTE: Given the above definitions/procedure, break_pt_bool_srs.iloc[0] should always be set to True.
        if pd_end_col is None:
            break_pt_bool_srs = df.reset_index()[pd_beg_col].diff(periods=1)>max_delta_t
        else:
            break_pt_bool_srs = (df.reset_index()[pd_beg_col]-df.reset_index().shift(1)[pd_end_col])>max_delta_t
        #-----
        break_pt_bool_srs.iloc[0]=True
    
        #-------------------------
        # Now, form the groups of indices which will go in each subplot
        idx_grps = []
        grp_i    = []
        for idx_i, bool_i in break_pt_bool_srs.items():
            if bool_i==True:
                if len(grp_i)>0:
                    idx_grps.append(grp_i)
                grp_i = []
            grp_i.append(idx_i)
        # Last group needs to be explicitly added
        assert(grp_i not in idx_grps) # Sanity check
        idx_grps.append(grp_i)
    
        #-------------------------
        return_dict = dict(
            df       = df, 
            idx_grps = idx_grps
        )
        return return_dict

    @staticmethod
    def get_break_pt_idx_groups(
        df, 
        max_delta_t, 
        pd_beg_col,
        pd_end_col=None, 
        grp_by_col='overlap_grp'
    ):
        r"""
        If one wants to plot df using horizontal breaks, this function will identify the groupings.
        For design case (with DOVS data for individual premise):
            pd_beg_col = 'DT_OFF_TS_FULL'
            pd_end_col = 'DT_ON_TS'
    
        Returns dict object with keys:
            df: the input df, which will now be sorted
            idx_grps: the index groups
    
        pd_beg_col/pd_end_col:
            These essentially define the ordering of the data.
            If pd_end_col is None:
                the differences between pd_beg_col of the current row and that of the previous are evaluated
                    i.e., df[pd_beg_col].diff(periods=1)
            If pd_end_col is not None:
                the differences between pd_beg_col of the current row and pd_end_col of the previous are evaluated
                    i.e., df[pd_beg_col]-df[pd_end_col].shift(periods=1)
                NOTE: For the case pd_beg_col=pd_beg_col, df[pd_beg_col]-df[pd_beg_col].shift(periods=1) reduces to 
                      df[pd_beg_col].diff(periods=1)
        """
        #-------------------------
        # Data must be sorted for this procedure to work
        if pd_end_col is None:
            df = df.sort_values(by=[pd_beg_col])
        else:
            df = df.sort_values(by=[pd_beg_col, pd_end_col])
    
        #--------------------------------------------------
        # break_pt_bool_srs will be used to find where break points need to be placed.
        #-----
        # The .diff(periods=1) operation calculates the difference of a DataFrame element compared with the element in the previous row.
        #   ==> The 0th element is always NaN/NaT
        #   ==> The 0th element of the df['DT_OFF_TS_FULL'].diff(periods=1)>max_delta_t operation will always be False
        # The .diff(periods=1)>max_delta_t operation therefore determines whether the row is outside max_delta_t (==> in different subplot)
        #   of the previous row (False if inside threshold/subplot, True if outside threshold/subplot)
        #   ==> True values of break_pt_bool_srs identify the LEFT point of the groups/subplots
        # NOTE: Given the above definitions/procedure, break_pt_bool_srs.iloc[0] should always be set to True.
        #--------------------------------------------------
        if grp_by_col is None:
            if pd_end_col is None:
                break_pt_bool_srs = df.reset_index()[pd_beg_col].diff(periods=1)>max_delta_t
            else:
                break_pt_bool_srs = (df.reset_index()[pd_beg_col]-df.reset_index().shift(1)[pd_end_col])>max_delta_t
            #-----
            break_pt_bool_srs.iloc[0]=True
        
            #-------------------------
            # Now, form the groups of indices which will go in each subplot
            idx_grps = []
            grp_i    = []
            for idx_i, bool_i in break_pt_bool_srs.items():
                if bool_i==True:
                    if len(grp_i)>0:
                        idx_grps.append(grp_i)
                    grp_i = []
                grp_i.append(idx_i)
            # Last group needs to be explicitly added
            assert(grp_i not in idx_grps) # Sanity check
            idx_grps.append(grp_i)
        #--------------------------------------------------
        else:
            # Create a temporary iloc column to store index location of rows
            tmp_iloc_col = Utilities.generate_random_string()
            assert(tmp_iloc_col not in df.columns.tolist())
            df[tmp_iloc_col] = range(df.shape[0])
            #-------------------------
            reduced_df = df.groupby([grp_by_col])[df.columns].apply(
                lambda x: pd.Series(
                    data  = [x[pd_beg_col].min(), x[pd_end_col].max(), list(x[tmp_iloc_col])], 
                    index = [pd_beg_col, pd_end_col, 'idxs_in_grp']
                )
            )
            reduced_df = reduced_df.sort_values(by=[pd_beg_col, pd_end_col])
            #-------------------------
            if pd_end_col is None:
                reduced_df['break_pt_bool'] = reduced_df[pd_beg_col].diff(periods=1)>max_delta_t
            else:
                reduced_df['break_pt_bool'] = (reduced_df[pd_beg_col]-reduced_df.shift(1)[pd_end_col])>max_delta_t
            #-----
            break_pt_bool_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(df=reduced_df, col='break_pt_bool', exact_match=True, assert_single=True)
            reduced_df.iloc[0, break_pt_bool_col_idx] = True
            #-------------------------
            # Now, form the groups of indices which will go in each subplot
            idx_grps = []
            grp_i    = []
            for idx_i, row_i in reduced_df.iterrows():
                bool_i = row_i['break_pt_bool']
                idxs_i = row_i['idxs_in_grp']
                if bool_i==True:
                    if len(grp_i)>0:
                        idx_grps.append(grp_i)
                    grp_i = []
                grp_i.extend(idxs_i)
            # Last group needs to be explicitly added
            assert(grp_i not in idx_grps) # Sanity check
            idx_grps.append(grp_i)        
        #-------------------------
        return_dict = dict(
            df       = df, 
            idx_grps = idx_grps
        )
        return return_dict    
    
    
    @staticmethod
    def split_df_at_break_pts(
        df, 
        max_delta_t, 
        pd_beg_col,
        pd_end_col=None
    ):
        r"""
        If one wants to plot df using horizontal breaks, this function will identify the groupings.
        For design case (with DOVS data for individual premise):
            pd_beg_col = 'DT_OFF_TS_FULL'
            pd_end_col = 'DT_ON_TS'
    
        Returns list of pd.DataFrame objects
    
        pd_beg_col/pd_end_col:
            These essentially define the ordering of the data.
            If pd_end_col is None:
                the differences between pd_beg_col of the current row and that of the previous are evaluated
                    i.e., df[pd_beg_col].diff(periods=1)
            If pd_end_col is not None:
                the differences between pd_beg_col of the current row and pd_end_col of the previous are evaluated
                    i.e., df[pd_beg_col]-df[pd_end_col].shift(periods=1)
                NOTE: For the case pd_beg_col=pd_beg_col, df[pd_beg_col]-df[pd_beg_col].shift(periods=1) reduces to 
                      df[pd_beg_col].diff(periods=1)
        """
        #--------------------------------------------------
        tmp_dct = DOVSRepeatFinder.get_break_pt_idx_groups(
            df          = df.copy(), 
            max_delta_t = max_delta_t, 
            pd_beg_col  = pd_beg_col,
            pd_end_col  = pd_end_col
        )
        #-------------------------
        df_to_split = tmp_dct['df']
        idx_grps    = tmp_dct['idx_grps']
        #--------------------------------------------------
        if len(idx_grps)==1:
            return [df]
        #--------------------------------------------------
        dfs = []
        for idx_grps_i in idx_grps:
            df_i = df_to_split.iloc[idx_grps_i].copy()
            dfs.append(df_i)
        #-------------------------
        assert(len(dfs)==len(idx_grps))
        return dfs
    
    
    @staticmethod
    def try_to_determine_mdates_loc(
        df, 
        max_ticks = 20, 
        t_beg_col = 'DT_OFF_TS_FULL', 
        t_end_col = 'DT_ON_TS', 
    ):
        r"""
        For purposes here, logically the increments should be between hours and months (but we allow down to mins and up to years)
    
        df:
            Should be a pd.DataFrame OR list of such
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(df, [pd.DataFrame, list]))
        if isinstance(df, list):
            dfs = df
        else:
            dfs = [df]
        #-------------------------
        ranges = [x[t_end_col].max()-x[t_beg_col].min() for x in dfs]
        ranges_max = np.max(ranges)
        #-------------------------
        n_mins = ranges_max/pd.Timedelta('1minute')
        n_hrs  = ranges_max/pd.Timedelta('1hour')
        n_days = ranges_max/pd.Timedelta('1day')
        n_months = ranges_max/pd.Timedelta('30.5days')
        n_years = ranges_max/pd.Timedelta('365.25days')
        #-------------------------
        if n_mins <= max_ticks:
            loc = mdates.MinuteLocator()
        elif n_hrs <=  max_ticks:
            loc = mdates.HourLocator()
        elif n_days <=  max_ticks:
            loc = mdates.DayLocator()
        elif n_months <=  max_ticks:
            loc = mdates.MonthLocator()
        elif n_years <=  max_ticks:
            loc = mdates.YearLocator()
        else:
            loc = None
        #-------------------------
        return loc


    @staticmethod
    def reduce_overlapping_outages_helper(
        input_lot
    ):
        r"""
        input_lot:
            Must be a list of tuples!
        """
        #--------------------------------------------------
        assert(
            isinstance(input_lot, list) and 
            Utilities.are_all_list_elements_of_type(lst=input_lot, typ=tuple)
        )
        #--------------------------------------------------
        output_lot   = []
        changes_made = False
        idxs_handled  = []
        #-------------------------
        for i in range(len(input_lot)):
            #----------
            if i in idxs_handled:
                continue
            #----------
            output_lot_i = list(input_lot[i])
            for j in range(i+1, len(input_lot)):
                #----------
                if j in idxs_handled:
                    continue
                #----------
                if set(output_lot_i).intersection(input_lot[j])==set():
                    continue
                else:
                    output_lot_i = list(set(output_lot_i + list(input_lot[j])))
                    changes_made = True
                    idxs_handled.append(j)
            idxs_handled.append(i)
            output_lot.append(output_lot_i)
        #-------------------------
        output_lot = [tuple(natsorted(x)) for x in output_lot]
        output_lot = list(set(output_lot))
        #-------------------------
        return output_lot, changes_made

    @staticmethod
    def reduce_overlapping_outages(
        input_lot, 
        max_itr = 100
    ):
        r"""
        """
        #-------------------------
        output_lot = copy.deepcopy(input_lot)
        #-------------------------
        changes_made = True
        cntr = 0
        #-----
        while(changes_made==True and cntr<=max_itr):
            output_lot, changes_made = DOVSRepeatFinder.reduce_overlapping_outages_helper(input_lot=output_lot)
            cntr += 1
        if cntr==max_itr:
            print('Failure!')
            assert(0)
        return output_lot


    @staticmethod
    def addtnl_shift_helper(
        df_i, 
        idxs_in_grp_col = 'idxs_in_grp'
    ):
        r"""
        !!!!! WARNING !!!!! df_i MUST BE PROPERLY SORTED BEFORE INPUT!!!!!
        
        When dealing with dovs_gen and when trying to shift and entire group, instead of simply shifting by cumcount(), one
          must also take into consideration the number of elements in the groups being shifted past
        """
        #-------------------------
        return_srs = df_i[idxs_in_grp_col].shift(periods=1).apply(lambda x: 0 if x is None else len(x)).cumsum()
        return return_srs


    def build_dovs_gen_df(
        self, 
        group_index_col  = 'group_index', 
        overlap_grp_col  = 'overlap_grp', 
        addtnl_shift_col = 'addtnl_shift', 
        t_beg_col        = 'DT_OFF_TS_FULL', 
        t_end_col        = 'DT_ON_TS', 
        outg_rec_nb_col  = 'OUTG_REC_NB', 
        xN_col           = 'PREMISE_NB', 
        y_col            = 'y'
    ):
        r"""
        Build general DOVS df, which is essentially self.dovs_df with any premise-specific information stripped away.
        These are also checked for overlaps to help with plotting, and those data are appended to the returned DF
        """
        #--------------------------------------------------
        #--------------------------------------------------
        dovs_gen = self.dovs_df[self.dovs_gen_cols].drop_duplicates().copy()
        
        #--------------------------------------------------
        assert(dovs_gen[outg_rec_nb_col].nunique()==dovs_gen.shape[0])
        assert(set([t_beg_col, t_end_col, outg_rec_nb_col]).difference(set(dovs_gen.columns.tolist()))==set())
        #-------------------------
        # dovs_gen must be properly sorted for procedure
        dovs_gen = dovs_gen.sort_values(by=[t_beg_col, t_end_col, outg_rec_nb_col])
        #-------------------------
        # Create a temporary iloc column to store index location of rows
        tmp_iloc_col = Utilities.generate_random_string()
        assert(tmp_iloc_col not in dovs_gen.columns.tolist())
        dovs_gen[tmp_iloc_col] = range(dovs_gen.shape[0])
        #----------------------------------------------------------------------------------------------------
        # First, using self.dovs_df, find groups of outages which do, in reality, overlap
        #-----
        # The code below combines chains of outages sharing premises
        # NOTE: Each outage will share at least one premise with at least one other outage in the group
        #       BUT, this means two groups do not necessarily share any premises
        # So, here, by overlap in reality we mean the outage overlaps in time and shares at least one premise 
        #   with at least one other outage in the group
        #----------------------------------------------------------------------------------------------------
        overlap_outgs = self.dovs_df.groupby([xN_col, overlap_grp_col])[outg_rec_nb_col].apply(lambda x: tuple(natsorted(x))).values.tolist()
        overlap_outgs = list(set(overlap_outgs))
        #-----
        overlap_outgs_fnl = DOVSRepeatFinder.reduce_overlapping_outages(
            input_lot = overlap_outgs, 
            max_itr = 100
        )
        #-------------------------
        assert(len(set(itertools.chain.from_iterable(overlap_outgs_fnl)))==dovs_gen.shape[0])
        #-----
        dovs_gen[overlap_grp_col] = -1
        for i,outgs_i in enumerate(overlap_outgs_fnl):
            dovs_gen.loc[dovs_gen[outg_rec_nb_col].isin(outgs_i), overlap_grp_col] = i
        #-----
        dovs_gen[group_index_col] = dovs_gen.groupby([overlap_grp_col]).cumcount()
        dovs_gen = dovs_gen.sort_values(by=[t_beg_col, t_end_col])
        #-----
        dovs_gen[addtnl_shift_col] = 0
        
        #----------------------------------------------------------------------------------------------------
        # Now, find and adjust any groups which will overlap in the final plot
        #----------------------------------------------------------------------------------------------------
        addtnl_shift_df = dovs_gen.groupby([overlap_grp_col])[dovs_gen.columns].apply(
            lambda x: pd.Series(
                data  = [x[t_beg_col].min(), x[t_end_col].max(), list(x[tmp_iloc_col])], 
                index = [t_beg_col, t_end_col, 'idxs_in_grp']
            )
        )
        # Done simply so the while loop will kick off!
        addtnl_shift_df['suprficl_ovrlp']=True
        
        #--------------------------------------------------
        cntr = 0
        max_itr = 100
        #-----
        while(addtnl_shift_df['suprficl_ovrlp'].any() and cntr<=max_itr):
            #-------------------------
            # addtnl_shift_df built above for 0th case, but needs to be updated for others
            if cntr>0:
                # At this point, members of addtnl_shift_df with suprficl_ovrlp_grp=-1 can safely be removed, they don't need any further adjustment
                addtnl_shift_df = addtnl_shift_df[addtnl_shift_df['suprficl_ovrlp_grp']!=-1].drop(columns=['addtnl_shift_i'])
                # Call .explode simply so resultant idxs_in_grp is a list, instead of nested list
                addtnl_shift_df = addtnl_shift_df.explode('idxs_in_grp').groupby(['suprficl_ovrlp_grp'])[addtnl_shift_df.columns].apply(
                    lambda x: pd.Series(
                        data  = [x[t_beg_col].min(), x[t_end_col].max(), list(x['idxs_in_grp'])], 
                        index = [t_beg_col, t_end_col, 'idxs_in_grp']
                    )
                )
            #-------------------------
            addtnl_shift_df = addtnl_shift_df.sort_values(by=[t_beg_col, t_end_col])
            addtnl_shift_df.index.name = None
            #-------------------------
            addtnl_shift_df = Utilities_df.find_and_append_overlapping_blocks_ids_in_df(
                df                   = addtnl_shift_df, 
                intrvl_beg_col       = t_beg_col, 
                intrvl_end_col       = t_end_col, 
                overlaps_col         = 'suprficl_ovrlp', 
                overlap_grp_col      = 'suprficl_ovrlp_grp', 
                return_overlaps_only = False, 
                no_overlap_grp_val   = -1 # sensible values = -1 or np.nan
            )
            #--------------------------------------------------
            if not addtnl_shift_df['suprficl_ovrlp'].any():
                continue # This effectively leaves the while loop
            #--------------------------------------------------
            addtnl_shift_i_srs = addtnl_shift_df.groupby(['suprficl_ovrlp_grp'], as_index=True, group_keys=False)[['idxs_in_grp']].apply(
                lambda x: DOVSRepeatFinder.addtnl_shift_helper(
                    df_i            = x, 
                    idxs_in_grp_col = 'idxs_in_grp'
                )
            )
            #-----
            assert(Utilities.is_object_one_of_types(addtnl_shift_i_srs, [pd.Series, pd.DataFrame]))
            if isinstance(addtnl_shift_i_srs, pd.DataFrame):
                assert(addtnl_shift_i_srs.shape[0]==1)
                addtnl_shift_i_srs = addtnl_shift_i_srs.squeeze()
            #-----
            addtnl_shift_df['addtnl_shift_i'] = addtnl_shift_i_srs
            addtnl_shift_df.loc[addtnl_shift_df['suprficl_ovrlp_grp']==-1, 'addtnl_shift_i'] = 0
            #-------------------------
            dovs_gen = pd.merge(
                dovs_gen, 
                addtnl_shift_df.explode('idxs_in_grp')[['idxs_in_grp', 'addtnl_shift_i']], 
                how      = 'left', 
                left_on  = tmp_iloc_col, 
                right_on = 'idxs_in_grp'
            ).drop(columns=['idxs_in_grp'])
            dovs_gen['addtnl_shift_i'] = dovs_gen['addtnl_shift_i'].fillna(0)
            #-----
            dovs_gen[addtnl_shift_col] = dovs_gen[addtnl_shift_col] + dovs_gen['addtnl_shift_i']
            dovs_gen = dovs_gen.drop(columns=['addtnl_shift_i'])
            #-------------------------
            cntr += 1
        
        #--------------------------------------------------
        dovs_gen[y_col] = dovs_gen[group_index_col] + dovs_gen[addtnl_shift_col]
        #-----
        dovs_gen = dovs_gen.drop(columns=[tmp_iloc_col])
        return dovs_gen


    @staticmethod
    def visualize_outages_for_PN_i_helper(
        ax_i, 
        df_i, 
        colors_dict, 
        mdates_loc       = None,  # e.g., mdates.DayLocator()
        outg_rec_nb_txt  = True, 
        y_col            = 'group_index', 
        overlap_grp_col  = 'overlap_grp', 
        t_beg_col        = 'DT_OFF_TS_FULL', 
        t_end_col        = 'DT_ON_TS', 
        outg_rec_nb_col  = 'OUTG_REC_NB', 
        x_tick_rot       = 90
    ):
        r"""
        Intended for use with df_i containing outages only for single premise (but can be used otherwise as well)
        """
        #-------------------------
        verts  = []
        colors = []
        texts  = []
        #-------------------------
        for idx, row in df_i.iterrows():
            x_0 = row[t_beg_col]
            x_1 = row[t_end_col]
            #-----
            y_0 = row[y_col]-0.4
            y_1 = row[y_col]+0.4
            #-------------------------
            v = [
                (mdates.date2num(x_0), y_0), 
                (mdates.date2num(x_0), y_1), 
                (mdates.date2num(x_1), y_1), 
                (mdates.date2num(x_1), y_0), 
                (mdates.date2num(x_0), y_0)
            ]
            #-----
            verts.append(v)
            colors.append(colors_dict[row[overlap_grp_col]])
            # NOTE: Since x are timestamps, can't use simple average operation (like can be done for y's)
            #       Mathematically, 0.5*(x_0+x_1) = x_0 + 0.5(x_1-x_0)
            texts.append(dict(
                x            = x_0 + 0.5*(x_1-x_0), 
                y            = 0.5*(y_0+y_1), 
                s            = row[outg_rec_nb_col], 
                color        = 'black', 
                fontweight   = 'bold', 
                ha           = 'center', 
                va           = 'center', 
                path_effects = [pe.withStroke(linewidth=1.5, foreground=colors_dict[row[overlap_grp_col]])]
            ))
        
        #-------------------------
        bars = PolyCollection(verts, facecolors=colors)
        ax_i.add_collection(bars)
        ax_i.autoscale()
        #-----    
        # ax_i.set_yticks(list(range(n_levels_y)))
        ax_i.set_yticks(natsorted(df_i[y_col].unique()))
        #-------------------------
        if mdates_loc is not None:
            # Each axis needs its own unique mdates_loc, so safest to call copy.deepcopy first
            mdates_loc = copy.deepcopy(mdates_loc)
            ax_i.xaxis.set_major_locator(mdates_loc)
            ax_i.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates_loc))
        #-------------------------
        if x_tick_rot!=0:
            Plot_General.rotate_ticklabels(
                ax       = ax_i, 
                rotation = x_tick_rot, 
                axis     = 'x', 
                ha       = None, 
                va       = None
            )
        #-------------------------
        if outg_rec_nb_txt:
            for text_i in texts:
                ax_i.text(**text_i)
        #-------------------------
        return ax_i
    
    
    @staticmethod
    def visualize_outages_for_PN_i( 
        df_i, 
        max_delta_t          = pd.Timedelta('1days'),  # or None
        mdates_loc           = None,  # e.g., mdates.DayLocator()
        determine_mdates_loc = True, 
        outg_rec_nb_txt      = True, 
        y_col                = 'group_index', 
        overlap_grp_col      = 'overlap_grp', 
        t_beg_col            = 'DT_OFF_TS_FULL', 
        t_end_col            = 'DT_ON_TS', 
        outg_rec_nb_col      = 'OUTG_REC_NB', 
        x_tick_rot           = 90, 
        fig_num              = 0, 
        save_path            = None
    ):
        r"""
        Intended for use with df_i containing outages only for single premise (but can be used otherwise as well)
    
        determine_mdates_loc:
            If mdates_loc is None and determine_mdates_loc is True, run DOVSRepeatFinder.try_to_determine_mdates_loc to find an appropriate value
        """
        #--------------------------------------------------
        colors_dict = Plot_General.get_standard_colors_dict(
            keys         = df_i[overlap_grp_col].unique().tolist(), 
            palette      = None, 
            random_order = True
        )
    
        #--------------------------------------------------
        if max_delta_t is None:
            dfs_i = [df_i]
        else:
            dfs_i = DOVSRepeatFinder.split_df_at_break_pts(
                df          = df_i, 
                max_delta_t = max_delta_t, 
                pd_beg_col  = t_beg_col,
                pd_end_col  = t_end_col
            )
    
        #--------------------------------------------------
        n_subplots = len(dfs_i)
        fig,axs = Plot_General.default_subplots(
            n_x                   = n_subplots,
            n_y                   = 1, 
            fig_num               = fig_num, 
            return_flattened_axes = True
        )
        assert(len(dfs_i)==len(axs))
        fig = Plot_General.adjust_subplots_args(fig=fig, wspace=0.025)
        #-------------------------
        d = 2  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        #-----
        if len(axs)>1:
            for j,ax_j in enumerate(axs):
                if j==0:
                    ax_j.spines['right'].set_visible(False)
                    #-----
                    ax_j.plot([1, 1], [0, 1], transform=ax_j.transAxes, **kwargs)
                elif j==len(axs)-1:
                    ax_j.yaxis.set_visible(False)
                    #-----
                    ax_j.spines['left'].set_visible(False)
                    #-----
                    ax_j.plot([0, 0], [0, 1], transform=ax_j.transAxes, **kwargs)
                else:
                    ax_j.yaxis.set_visible(False)
                    #-----
                    ax_j.spines['left'].set_visible(False)
                    ax_j.spines['right'].set_visible(False)
                    #-----
                    ax_j.plot([0, 0], [0, 1], transform=ax_j.transAxes, **kwargs)
                    ax_j.plot([1, 1], [0, 1], transform=ax_j.transAxes, **kwargs)
        #--------------------------------------------------
        if determine_mdates_loc and mdates_loc is None:
            mdates_loc = DOVSRepeatFinder.try_to_determine_mdates_loc(
                df        = dfs_i, 
                max_ticks = 20, 
                t_beg_col = t_beg_col, 
                t_end_col = t_end_col, 
            )
        #-----
        for j,df_j in enumerate(dfs_i):
            ax_j      = axs[j]    
            ax_j = DOVSRepeatFinder.visualize_outages_for_PN_i_helper(
                ax_i             = ax_j, 
                df_i             = df_j, 
                colors_dict      = colors_dict, 
                mdates_loc       = mdates_loc, 
                outg_rec_nb_txt  = outg_rec_nb_txt, 
                y_col            = y_col, 
                overlap_grp_col  = overlap_grp_col, 
                t_beg_col        = t_beg_col, 
                t_end_col        = t_end_col, 
                outg_rec_nb_col  = outg_rec_nb_col, 
                x_tick_rot       = x_tick_rot
            )
        #--------------------------------------------------
        if save_path is not None:
            Plot_General.save_fig_to_path(
                fig         = fig, 
                save_path   = save_path, 
                bbox_inches = 'tight'
            )
        #--------------------------------------------------
        return fig,axs


    def visualize_outages_for_PN( 
        self, 
        PN, 
        max_delta_t          = pd.Timedelta('1days'), 
        mdates_loc           = None,  # e.g., mdates.DayLocator()
        determine_mdates_loc = True, 
        outg_rec_nb_txt      = True, 
        y_col                = 'group_index', 
        overlap_grp_col      = 'overlap_grp', 
        t_beg_col            = 'DT_OFF_TS_FULL', 
        t_end_col            = 'DT_ON_TS', 
        outg_rec_nb_col      = 'OUTG_REC_NB', 
        xN_col               = 'PREMISE_NB', 
        x_tick_rot           = 90, 
        fig_num              = 0, 
        save_path            = None
    ):
        r"""
        determine_mdates_loc:
            If mdates_loc is None and determine_mdates_loc is True, run DOVSRepeatFinder.try_to_determine_mdates_loc to find an appropriate value
        """
        #--------------------------------------------------
        assert(PN in self.dovs_df[xN_col].unique())
        #-----
        df_i = self.dovs_df[self.dovs_df[xN_col]==PN]
        #-----
        fig,axs =  DOVSRepeatFinder.visualize_outages_for_PN_i( 
            df_i                 = df_i, 
            max_delta_t          = max_delta_t,  # e.g., pd.Timedelta('28days')
            mdates_loc           = mdates_loc,  # e.g., mdates.DayLocator()
            determine_mdates_loc = determine_mdates_loc, 
            outg_rec_nb_txt      = outg_rec_nb_txt, 
            y_col                = y_col, 
            overlap_grp_col      = overlap_grp_col, 
            t_beg_col            = t_beg_col, 
            t_end_col            = t_end_col, 
            outg_rec_nb_col      = outg_rec_nb_col, 
            x_tick_rot           = x_tick_rot, 
            fig_num              = fig_num, 
            save_path            = None
        )
        #-----
        fig.suptitle(f'PN = {PN}', fontsize='xx-large')
        #--------------------------------------------------
        if save_path is not None:
            Plot_General.save_fig_to_path(
                fig         = fig, 
                save_path   = save_path, 
                bbox_inches = 'tight'
            )
        #--------------------------------------------------
        return fig,axs
        
    
    def visualize_outages_general_OLD( 
        self, 
        max_delta_t          = pd.Timedelta('1days'), 
        mdates_loc           = None,  # e.g., mdates.DayLocator()
        determine_mdates_loc = True, 
        outg_rec_nb_txt      = True, 
        y_col                = 'y', 
        overlap_grp_col      = 'overlap_grp', 
        t_beg_col            = 'DT_OFF_TS_FULL', 
        t_end_col            = 'DT_ON_TS', 
        outg_rec_nb_col      = 'OUTG_REC_NB', 
        x_tick_rot           = 90, 
        fig_num              = 0, 
        save_path            = None
    ):
        r"""
        Intended for use with df_i containing outages only for single premise (but can be used otherwise as well)
    
        determine_mdates_loc:
            If mdates_loc is None and determine_mdates_loc is True, run DOVSRepeatFinder.try_to_determine_mdates_loc to find an appropriate value
        """
        #--------------------------------------------------
        assert(self.dovs_gen_df.shape[0]>0)
        #-------------------------
        fig,axs = DOVSRepeatFinder.visualize_outages_for_PN_i( 
            df_i                 = self.dovs_gen_df, 
            max_delta_t          = max_delta_t,
            mdates_loc           = mdates_loc, 
            determine_mdates_loc = determine_mdates_loc, 
            outg_rec_nb_txt      = outg_rec_nb_txt, 
            y_col                = y_col, 
            overlap_grp_col      = overlap_grp_col, 
            t_beg_col            = t_beg_col, 
            t_end_col            = t_end_col, 
            outg_rec_nb_col      = outg_rec_nb_col, 
            x_tick_rot           = x_tick_rot, 
            fig_num              = fig_num, 
            save_path            = save_path
        )
        return fig,axs


    def visualize_outages_general( 
        self, 
        max_delta_t          = pd.Timedelta('1days'), 
        mdates_loc           = None,  # e.g., mdates.DayLocator()
        determine_mdates_loc = True, 
        outg_rec_nb_txt      = True, 
        y_col                = 'y', 
        overlap_grp_col      = 'overlap_grp', 
        t_beg_col            = 'DT_OFF_TS_FULL', 
        t_end_col            = 'DT_ON_TS', 
        outg_rec_nb_col      = 'OUTG_REC_NB', 
        x_tick_rot           = 90, 
        fig_num              = 0, 
        max_y_per_plot       = None, 
        hard_max_y           = True, 
        save_path            = None, 
        close_all_figures    = False
    ):
        r"""
        Intended for use with df_i containing outages only for single premise (but can be used otherwise as well)
    
        determine_mdates_loc:
            If mdates_loc is None and determine_mdates_loc is True, run DOVSRepeatFinder.try_to_determine_mdates_loc to find an appropriate value
        """
        #--------------------------------------------------
        assert(self.dovs_gen_df.shape[0]>0)
        #-------------------------
        if max_y_per_plot is None:
            #-------------------------
            fig,axs = DOVSRepeatFinder.visualize_outages_for_PN_i( 
                df_i                 = self.dovs_gen_df, 
                max_delta_t          = max_delta_t,
                mdates_loc           = mdates_loc, 
                determine_mdates_loc = determine_mdates_loc, 
                outg_rec_nb_txt      = outg_rec_nb_txt, 
                y_col                = y_col, 
                overlap_grp_col      = overlap_grp_col, 
                t_beg_col            = t_beg_col, 
                t_end_col            = t_end_col, 
                outg_rec_nb_col      = outg_rec_nb_col, 
                x_tick_rot           = x_tick_rot, 
                fig_num              = fig_num, 
                save_path            = save_path
            )
            return fig,axs
        else:
            # Trying to combine all plots into one via subplots is complicated by the fact that visualize_outages_for_PN_i creates
            #   a collection of subplots when break points are needed
            # To avoid complexities, in this case, just return a list of (fig,axs) tuples
            dovs_gen_df = self.dovs_gen_df.copy()
            #-------------------------
            # Idea here, I guess, is to cannabalize dovs_gen_df group-by-group until nothing remains
            # The reason for this method (vs simple slicing) is that the groups are not well-defined by a single y-value, but a range.
            #-----
            tmp_idfr_col = Utilities.generate_random_string()
            dovs_gen_df[tmp_idfr_col] = range(dovs_gen_df.shape[0])
            #-------------------------
            if save_path is not None:
                pdf = PdfPages(save_path)
            #-------------------------
            cntr = 0
            max_plots = plt.rcParams['figure.max_open_warning']
            fig_axs_tups = []
            while dovs_gen_df.shape[0]>0 and cntr<1000:
                if cntr>=max_plots and not close_all_figures:
                    print('Too many figures open, must close one')
                    plt.close(fig_axs_tups[len(fig_axs_tups)-max_plots][0])
                #-------------------------
                # Find all entries with appropriate y-values
                tmp_df_0 = dovs_gen_df[(dovs_gen_df[y_col]>-1) & (dovs_gen_df[y_col]<=max_y_per_plot)]
                # Find complete groups containing at least one member with y value in appropriate values
                tmp_df_1 = dovs_gen_df[dovs_gen_df[overlap_grp_col].isin(tmp_df_0[overlap_grp_col].unique())]
                #-------------------------
                if tmp_df_0.shape[0]==tmp_df_1.shape[0]:
                    df_i_to_plot = tmp_df_0.copy() # could set equal to tmp_df_1, doesn't matter
                elif not hard_max_y:
                    df_i_to_plot = tmp_df_1.copy()
                else:
                    # For operations below, I believe you must use tmp_df_1
                    ovrlp_grps_to_rmv = tmp_df_1[tmp_df_1[y_col] == tmp_df_1[y_col].max()][overlap_grp_col].unique().tolist()
                    df_i_to_plot = tmp_df_1[~tmp_df_1[overlap_grp_col].isin(ovrlp_grps_to_rmv)].copy()
                    #-----
                    if df_i_to_plot.shape[0]==0:
                        # In this case, the group(s) themselves are larger than max_y_per_plot, so they cannot be reduced any further!
                        ovrlp_grps_to_rmv = ovrlp_grps_to_rmv[1:]
                        df_i_to_plot = tmp_df_1[~tmp_df_1[overlap_grp_col].isin(ovrlp_grps_to_rmv)].copy()
                #-------------------------
                fig_i,axs_i = DOVSRepeatFinder.visualize_outages_for_PN_i( 
                    df_i                 = df_i_to_plot, 
                    max_delta_t          = max_delta_t, 
                    mdates_loc           = mdates_loc, 
                    determine_mdates_loc = determine_mdates_loc, 
                    outg_rec_nb_txt      = outg_rec_nb_txt, 
                    y_col                = y_col, 
                    overlap_grp_col      = overlap_grp_col, 
                    t_beg_col            = t_beg_col, 
                    t_end_col            = t_end_col, 
                    outg_rec_nb_col      = outg_rec_nb_col, 
                    x_tick_rot           = x_tick_rot, 
                    fig_num              = fig_num, 
                    save_path            = None
                )
                fig_axs_tups.append((fig_i,axs_i))
                if save_path is not None:
                    pdf.savefig(fig_i, bbox_inches='tight')
                #-------------------------
                # Remove those plotted
                dovs_gen_df = dovs_gen_df[~dovs_gen_df[tmp_idfr_col].isin(df_i_to_plot[tmp_idfr_col].unique())].copy()
                # Shift all y-values so min value at 0
                dovs_gen_df[y_col] = dovs_gen_df[y_col] - dovs_gen_df[y_col].min()
                #-------------------------
                fig_num += 1
                cntr += 1
                #------------------------
                if close_all_figures:
                    plt.close(fig_i)
            #-------------------------
            if save_path is not None:
                pdf.close()
            #-------------------------
            return fig_axs_tups

    def output_results(
        self, 
        save_dir, 
        append_tag=None #e.g., append_tag = '20240101_20240106'
    ):
        r"""
        """
        #--------------------------------------------------
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        #--------------------------------------------------
        output_dovs_df = self.dovs_df.drop(columns = ['overlaps', 'group_index']).copy()
        output_dovs_df['overlap_grp'] = output_dovs_df['PREMISE_NB'].astype(str)+'_'+output_dovs_df['overlap_grp'].astype(str)
        #-------------------------
        output_dovs_df = output_dovs_df.set_index(['PREMISE_NB', 'overlap_grp']).sort_index()
        output_dovs_df = Utilities_df.move_cols_to_front(df=output_dovs_df, cols_to_move=['OUTAGE_NB', 'OUTG_REC_NB'])
        
        #--------------------------------------------------
        output_dovs_gen_df = self.dovs_gen_df.set_index('overlap_grp').sort_index()
        output_dovs_gen_df = output_dovs_gen_df.drop(columns=['group_index', 'addtnl_shift', 'y'])
        output_dovs_gen_df = Utilities_df.move_cols_to_front(df=output_dovs_gen_df, cols_to_move=['OUTAGE_NB', 'OUTG_REC_NB'])
        
        
        #--------------------------------------------------
        dovs_df_name      = 'DOVSRepeats'
        dovs_df_name_dtld = f'{dovs_df_name}_detailed'
        dovs_gen_df_name  = f'{dovs_df_name}_general'
        if append_tag is not None:
            dovs_df_name      = f'{dovs_df_name}_{append_tag}'
            dovs_df_name_dtld = f'{dovs_df_name_dtld}_{append_tag}'
            dovs_gen_df_name  = f'{dovs_gen_df_name}_{append_tag}'
            
        #-------------------------
        output_dovs_df.to_pickle(    os.path.join(save_dir, f'{dovs_df_name_dtld}.pkl'))
        output_dovs_gen_df.to_pickle(os.path.join(save_dir, f'{dovs_gen_df_name}.pkl'))
        #-----
        with pd.ExcelWriter(os.path.join(save_dir, f'{dovs_df_name}.xlsx')) as writer:  
            output_dovs_gen_df.to_excel(writer, sheet_name='General')
            output_dovs_df.to_excel(writer, sheet_name='Detailed')
        
        #--------------------------------------------------
        max_y_per_plot = 20
        save_path = os.path.join(save_dir, f'{dovs_gen_df_name}.pdf')
        
        figaxs = self.visualize_outages_general(
            max_delta_t          = pd.Timedelta('1days'), 
            mdates_loc           = None,  
            determine_mdates_loc = True, 
            outg_rec_nb_txt      = True, 
            y_col                = 'y', 
            overlap_grp_col      = 'overlap_grp', 
            t_beg_col            = 'DT_OFF_TS_FULL', 
            t_end_col            = 'DT_ON_TS', 
            outg_rec_nb_col      = 'OUTG_REC_NB', 
            x_tick_rot           = 90, 
            fig_num              = 0, 
            max_y_per_plot       = max_y_per_plot, 
            hard_max_y           = True, 
            save_path            = save_path, 
            close_all_figures    = True
        )