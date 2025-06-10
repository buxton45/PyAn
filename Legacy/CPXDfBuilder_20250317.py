#!/usr/bin/env python

r"""
Holds CPXDfBuilder class.  See CPXDfBuilder.CPXDfBuilder for more information.
This is intended to replace both the legacy MECPODf and MECPOAn objects
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
import datetime
from enum import IntEnum
import copy
from natsort import natsorted
from functools import reduce
import re
import warnings
#--------------------------------------------------
from GenAn import GenAn
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
from MeterPremise import MeterPremise
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
import Utilities_dt
import DataFrameSubsetSlicer
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
#--------------------------------------------------


#----------------------------------------------------------------------------------------------------
class CPXInputDType(IntEnum):
    r"""
    Enum class for CPXDfBuilder input data types, which can be:
        evs_sums   = Events summary data (from meter_events.events_summary_vw)
        end_events = End events (from meter_events.end_device_event)
    """
    evs_sums    = 0
    end_events  = 1
    unset       = 2

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
class CPXDfBuilder:
    r"""
    Methods to build basic CPXDf from either end events (from meter_events.end_device_event) of end events summaries (from meter_events.events_summary_vw).
    CPXDf = Counts Per X DataFrame (where, X could represent outage, transformer, etc.)
    Typically, this Meter Events Counts Per X (MECPX), but could also be Id (enddeviceeventtypeid) Counts Per X (ICPX)
    """
    
    def __init__(
        self, 
        input_data_type
    ):
        r"""
        input_data_type:
            Should be 'evs_sums' or 'end_events'.
            If lazy, one can also use integer values:
                0 ==> 'evs_sums'
                1 ==> 'end_events'
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(input_data_type, [str, int]))
        if isinstance(input_data_type, int):
            input_data_type = CPXInputDType(input_data_type).name
        #-----
        assert(CPXDfBuilder.is_acceptable_input_data_type(input_data_type))
        self.input_data_type = input_data_type


    #--------------------------------------------------
    @staticmethod
    def is_acceptable_input_data_type(
        input_data_type
    ):
        r"""
        """
        #--------------------
        accptbl_dtypes = ['evs_sums', 'end_events']
        if input_data_type in accptbl_dtypes:
            return True
        return False

    #--------------------------------------------------
    @staticmethod
    def std_XNs_cols():
        std_cols = ['SNs', '_SNs', '_SNs_wEvs', '_prem_nbs', '_outg_SNs', '_outg_prem_nbs', '_prim_SNs', '_xfmr_SNs', '_xfmr_PNs']
        return std_cols
        
    #--------------------------------------------------
    @staticmethod
    def std_nXNs_cols():
        std_cols = ['nSNs', '_nSNs', '_nSNs_wEvs', '_nprem_nbs', '_outg_nSNs', '_outg_nprem_nbs', '_prim_nSNs', '_xfmr_nSNs', '_xfmr_nPNs']
        return std_cols
    

    #--------------------------------------------------
    @staticmethod
    def determine_XNs_cols(
        cpx_df, 
        XNs_cols = None
    ):
        r"""
        Determine which columns in cpx_df are XNs cols.
        If XNs_cols is None, it is set to CPXDfBuilder.std_XNs_cols().
        Then, the XNs cols returned are simply those XNs_cols found in cpx_df.columns
        """
        #-------------------------
        if XNs_cols is None:
            XNs_cols = CPXDfBuilder.std_XNs_cols()
        assert(isinstance(XNs_cols, list))
        found_cols = list(set(cpx_df.columns.tolist()).intersection(set(XNs_cols)))
        return found_cols
    
    #--------------------------------------------------
    @staticmethod
    def determine_nXNs_cols(
        cpx_df, 
        nXNs_cols = None
    ):
        r"""
        Determine which columns in cpx_df are nXNs cols.
        If nXNs_cols is None, it is set to CPXDfBuilder.std_nXNs_cols().
        Then, the nXNs cols returned are simply those nXNs_cols found in cpx_df.columns
        """
        #-------------------------
        if nXNs_cols is None:
            nXNs_cols = CPXDfBuilder.std_nXNs_cols()
        assert(isinstance(nXNs_cols, list))
        found_cols = list(set(cpx_df.columns.tolist()).intersection(set(nXNs_cols)))
        return found_cols
    
    #--------------------------------------------------
    @staticmethod
    def remove_XNs_cols_from_rcpx_df(
        rcpx_df  , 
        XNs_tags = None, 
    ):
        r"""
        XNs_tags:
          Defaults to CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols() when XNs_tags is None
        NOTE: XNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if XNs_tags is None:
            XNs_tags=CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        #-------------------------
        assert(rcpx_df.columns.nlevels<=2)
        if rcpx_df.columns.nlevels==1:
            level_idx=0
        else:
            level_idx=1
        #-------------------------
        untagged_idxs = Utilities.find_untagged_idxs_in_list(
            lst  = rcpx_df.columns.get_level_values(level_idx).tolist(), 
            tags = XNs_tags)
        cols_to_keep = rcpx_df.columns[untagged_idxs]
        #-------------------------
        return rcpx_df[cols_to_keep]
    
    #--------------------------------------------------
    @staticmethod
    def find_XNs_cols_idxs_from_cpx_df(
        cpx_df   , 
        XNs_tags = None, 
    ):
        r"""
        Returns the index positions of those found within the list of columns (if cpx_df is wide) 
          or index leve_1 values (if cpx_df is long). 
        As opposed to returning the actual columns found, this makes it easier to exclude (or select) the 
          columns/indices of choice, especially when dealing with the case of cpx_df being long, or cpx_df 
          containing raw and normalized values (in which case, e.g., there will likely be two columns for each
          XNs_tag, one for raw and one for normalized)
    
        XNs_tags:
          Defaults to CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols() when XNs_tags is None
        NOTE: XNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if XNs_tags is None:
            XNs_tags = CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        #-------------------------
        assert(cpx_df.columns.nlevels<=2)
        if cpx_df.columns.nlevels==1:
            level_idx = 0
        else:
            level_idx = 1
        #-------------------------
        tagged_idxs = Utilities.find_tagged_idxs_in_list(
            lst  = cpx_df.columns.get_level_values(level_idx).tolist(), 
            tags = XNs_tags)
        #-------------------------
        return tagged_idxs
    

    #--------------------------------------------------
    @staticmethod
    def get_non_XNs_cols_from_cpx_df(
        cpx_df   , 
        XNs_tags = None
    ):
        r"""
        Only for wide-form cpx_dfs.
        Returns a list of columns which are not XNs cols (as found using XNs_tags)
        
        XNs_tags:
          Defaults to CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols() when XNs_tags is None
        """
        #-------------------------
        if XNs_tags is None:
            XNs_tags=CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        #-------------------------
        XNs_cols_idxs = CPXDfBuilder.find_XNs_cols_idxs_from_cpx_df(
            cpx_df   = cpx_df, 
            XNs_tags = XNs_tags
        )
        XNs_cols     = cpx_df.columns[XNs_cols_idxs].tolist()
        non_XNs_cols = [x for x in cpx_df.columns if x not in XNs_cols]
        assert(len(set(non_XNs_cols+XNs_cols).symmetric_difference(set(cpx_df.columns)))==0)
        #-------------------------
        return non_XNs_cols
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_time_pds_rename(
        curr_time_pds    , 
        td_min           = pd.Timedelta('1D'), 
        td_max           = pd.Timedelta('31D'), 
        freq             = pd.Timedelta('5D'), 
        return_time_grps = False
    ):
        r"""
        Need to convert the time periods, which are currently housed in the date_col index of 
          rcpx_0 from their specific dates to the names expected by the model.
        In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
          dates of the given interval.
        These will be converted to the titles contained in final_time_pds below

        curr_time_pds:
            When originally developed, intended to be a list.
            However, now can also be a list of tuples/lists (each element of length 2)

            curr_time_pds as list:
                e.g.:
                   curr_time_pds = [Timestamp('2023-05-02'), Timestamp('2023-05-07'), 
                                    Timestamp('2023-05-12'), Timestamp('2023-05-17'), 
                                    Timestamp('2023-05-22'), Timestamp('2023-05-27')]
                ==> final_time_pds = ['01-06 Days', '06-11 Days',
                                     '11-16 Days', '16-21 Days',
                                     '21-26 Days', '26-31 Days']
                
                Returns: A dict object with keys equal to curr_time_pds and values equal to final_time_pds

            curr_time_pds as list of tuples/lists (each element of length 2):
                e.g.:
                   curr_time_pds = [(Timestamp('2023-05-02'), Timestamp('2023-05-07')), 
                                    (Timestamp('2023-05-07'), Timestamp('2023-05-12')), 
                                    (Timestamp('2023-05-12'), Timestamp('2023-05-17')), 
                                    (Timestamp('2023-05-17'), Timestamp('2023-05-22')), 
                                    (Timestamp('2023-05-22'), Timestamp('2023-05-27')), 
                                    (Timestamp('2023-05-27'), Timestamp('2023-06-01'))]
                ==> final_time_pds = ['01-06 Days', '06-11 Days',
                                     '11-16 Days', '16-21 Days',
                                     '21-26 Days', '26-31 Days']
                
                Returns: A dict object with keys equal to [x[0] for x in curr_time_pds] and values equal to final_time_pds
         
        !!! NOTE !!!: This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
        """
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        freq   = pd.Timedelta(freq)
        #-------------------------
        Utilities_dt.assert_timedelta_is_days(td_min)
        Utilities_dt.assert_timedelta_is_days(td_max)
        Utilities_dt.assert_timedelta_is_days(freq)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        days_freq = freq.days
        #-----
        # Make sure (days_max-days_min) evenly divisible by days_freq
        assert((days_max-days_min) % days_freq==0)
        #----------------------------------------------------------------------------------------------------
        if Utilities.is_object_one_of_types(curr_time_pds[0], [tuple, list]):
            assert(Utilities.are_all_list_elements_one_of_types(curr_time_pds, [tuple, list]))
            assert(Utilities.are_list_elements_lengths_homogeneous(curr_time_pds, length=2))
            for time_grp_i in curr_time_pds:
                assert(time_grp_i[1]-time_grp_i[0]==freq)
            #-------------------------
            curr_time_pds_0 = [x[0] for x in natsorted(curr_time_pds, key=lambda x:x[0])]
            #-------------------------
            return CPXDfBuilder.get_time_pds_rename(
                curr_time_pds = curr_time_pds_0, 
                td_min        = td_min, 
                td_max        = td_max, 
                freq          = freq
            )
        #----------------------------------------------------------------------------------------------------
        curr_time_pds = natsorted(curr_time_pds)
        time_grps     = []
        # Each time period should have a width equal to freq
        for i in range(len(curr_time_pds)):
            if i==0:
                continue
            assert(curr_time_pds[i]-curr_time_pds[i-1]==freq)
            time_grps.append((curr_time_pds[i-1], curr_time_pds[i]))
        time_grps.append((time_grps[-1][-1], time_grps[-1][-1]+freq))
        #-------------------------
        n_pds_needed    = len(curr_time_pds)
        days_max_needed = days_min + days_freq*n_pds_needed
        if days_max < days_max_needed:
            days_max = days_max_needed
        #-------------------------
        final_time_pds_0 = np.arange(start=days_min, stop=days_max+1, step=days_freq).tolist()
        assert(len(final_time_pds_0) > len(curr_time_pds)) # len(final_time_pds_0) should equal len(curr_time_pds)+1
        final_time_pds_0 = final_time_pds_0[:len(curr_time_pds)+1]
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
        time_grps     = natsorted(time_grps, reverse=True)
        #-------------------------
        time_pds_rename = dict(zip(curr_time_pds, final_time_pds))
        #-------------------------
        if return_time_grps:
            return time_pds_rename, time_grps
        return time_pds_rename
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def perform_std_col_type_conversions(
        df          , 
        dtypes_dict ,
    ):
        r"""
        """
        #-------------------------
        if dtypes_dict is None:
            return df
        #-------------------------
        assert(isinstance(dtypes_dict, dict))
        dtypes_dict = {k:v for k,v in dtypes_dict.items() if k in df.columns}
        #-----
        if len(dtypes_dict) > 0:
            df = Utilities_df.convert_col_types(
                df                  = df,
                cols_and_types_dict = dtypes_dict,
                to_numeric_errors   = 'coerce',
                inplace             = True,
            )
        #-------------------------
        return df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def append_to_df(
        df                , 
        addtnl_df         , 
        sort_by           = None, 
        make_col_types_eq = False
    ):
        r"""
        addtnl_df:
            Expected to be a pd.DataFrame object, but can also be a list of such
            IF A LIST:
                All elements must be pd.DataFrame objects
                append_to_df will be called iteratively with each element

        make_col_types_eq:
            Can be time consuming if joining many DFs, so I suggest keeping False
        """
        #--------------------------------------------------
        if isinstance(addtnl_df, list):
            assert(Utilities.are_all_list_elements_of_type(lst=addtnl_df, typ=pd.DataFrame))
            for addtnl_df_i in addtnl_df:
                df = CPXDfBuilder.append_to_df(
                    df                = df, 
                    addtnl_df         = addtnl_df_i, 
                    sort_by           = sort_by, 
                    make_col_types_eq = make_col_types_eq
                )
            return df
        #--------------------------------------------------
        if addtnl_df is None or addtnl_df.shape[0]==0:
            return df
        #-------------------------
        if df is None or df.shape[0]==0:
            assert(addtnl_df.shape[0]>0)
            return addtnl_df
        #-------------------------
        assert(df.columns.equals(addtnl_df.columns))
        #-------------------------
        if make_col_types_eq:
            df, addtnl_df = Utilities_df.make_df_col_dtypes_equal(
                df_1              = df, 
                col_1             = df.columns.tolist(), 
                df_2              = addtnl_df, 
                col_2             = addtnl_df.columns.tolist(), 
                allow_reverse_set = True, 
                assert_success    = True, 
                inplace           = True
            )

        #-------------------------
        # Make sure df and addtnl_df share the same columns and 
        #   do not have any overlapping data
        assert(
            not Utilities_df.do_dfs_overlap(
                df_1            = df, 
                df_2            = addtnl_df, 
                enforce_eq_cols = True, 
                include_index   = True
            )
        )

        #-------------------------
        df = pd.concat([df, addtnl_df])
        #-------------------------
        if sort_by is not None:
            df = df.sort_values(by=sort_by)
        #-------------------------
        return df
    
    #---------------------------------------------------------------------------
    @staticmethod
    def concat_dfs(
        dfs               , 
        make_col_types_eq = False
    ):
        r"""
        """
        #-------------------------
        assert(Utilities.are_all_list_elements_of_type(lst=dfs, typ=pd.DataFrame))
        #-------------------------
        if len(dfs)==1:
            return dfs[0]
        #-------------------------
        df         = dfs[0]
        addtnl_dfs = dfs[1:]
        sort_by    = None
        #-----
        df = CPXDfBuilder.append_to_df(
            df                = df, 
            addtnl_df         = addtnl_dfs, 
            sort_by           = sort_by, 
            make_col_types_eq = make_col_types_eq
        )
        #-------------------------
        return df
    

    #---------------------------------------------------------------------------
    @staticmethod  
    def normalize_rcpx_df_by_time_interval(
        rcpx_df                 , 
        days_min_outg_td_window , 
        days_max_outg_td_window , 
        cols_to_adjust          = None, 
        XNs_tags                = None, 
        inplace                 = False
    ):
        r"""
        Normalize a Reason Counts Per X (RCPX, where X could be, e.g., outage & transformer) pd.DataFrame by the time width around the outage used to construct it.
        It is assumed that the limits on the time window are INCLUSIVE on the front and EXCLUSIVE on the back, i.e., 
          [outage_date-days_max_outg_td_window, outage_date-days_min_outg_td_window).
        Thus, the width of the window should be calculated as:
          window_widths_days = days_max_outg_td_window-days_min_outg_td_window
          e.g.:  Assume the window goes from 1 day before to 2 days before:
                   ==> window_widths_days = 1 days = 2-1
          e.g.:  Assume the window goes from 5 days before to 10 days before:
                   ==> window_widths_days = 5 days = 10-5

        cols_to_adjust:
          if cols_to_adjust is None (default), adjust all columns EXCEPT for those containing SN info (as dictated by
            XNs_tags via CPXDfBuilder.remove_XNs_cols_from_rcpc_df)

        XNs_tags:
          Defaults to CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols() when XNs_tags is None
          Used only if cols_to_adjust is None (as it is by default)
          NOTE: XNs_tags should contain strings, not tuples.
                If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if XNs_tags is None:
            XNs_tags=CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        #-------------------------
        if not inplace:
            rcpx_df = rcpx_df.copy()
        #-------------------------
        window_widths_days = (days_max_outg_td_window-days_min_outg_td_window)
        #-------------------------
        # if cols_to_adjust is None, adjust all columns EXCEPT for those containing SN info
        if cols_to_adjust is None:
            cols_to_adjust = CPXDfBuilder.remove_XNs_cols_from_rcpx_df(rcpx_df, XNs_tags=XNs_tags).columns.tolist()
        #-------------------------
        rcpx_df[cols_to_adjust] = rcpx_df[cols_to_adjust]/window_widths_days
        #-------------------------
        return rcpx_df
    

   #---------------------------------------------------------------------------------------------------- 
    @staticmethod
    def get_total_event_counts(
        cpx_df                , 
        output_col            = 'total_counts', 
        sort_output           = False, 
        XNs_tags              = None, 
        non_reason_lvl_0_vals = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        Basically just sums up the number of events for each index.
        The nXNs (and XNs) columns should not be included in the count, hence the need for the XNs tags argument.
          
        XNs_tags:
          Defaults to CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols() when XNs_tags is None
        """
        #----------------------------------------------------------------------------------------------------
        assert(cpx_df.columns.nlevels<=2)
        assert(cpx_df.index.nunique()==cpx_df.shape[0]) # Assumes a unique index
        if(
            cpx_df.columns.nlevels == 2 and 
            cpx_df.columns.get_level_values(0).nunique() > 1
        ):
            #--------------------------------------------------
            col_lvl_0_vals = natsorted(cpx_df.columns.get_level_values(0).unique())
            #-------------------------
            non_reason_lvl_0_vals = non_reason_lvl_0_vals + CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
            #--------------------------------------------------
            counts_dfs = []
            for col_lvl_0_val_i in col_lvl_0_vals:
                if col_lvl_0_val_i in non_reason_lvl_0_vals:
                    continue
                cpx_pd_i  = cpx_df[[col_lvl_0_val_i]].copy()
                cpx_pd_i = CPXDfBuilder.get_total_event_counts(
                    cpx_df                = cpx_pd_i, 
                    output_col            = output_col, 
                    sort_output           = False, 
                    XNs_tags              = XNs_tags, 
                    non_reason_lvl_0_vals = non_reason_lvl_0_vals, 
                )
                if cpx_pd_i is not None and cpx_pd_i.shape[0]>0:
                    counts_dfs.append(cpx_pd_i)
            #--------------------------------------------------
            total_df = Utilities_df.merge_list_of_dfs(
                dfs         = counts_dfs, 
                how         = 'inner', 
                left_index  = True, 
                right_index = True
            )
            total_df = total_df[total_df.columns.sort_values()]
            #--------------------------------------------------
            return total_df

        #----------------------------------------------------------------------------------------------------
        if XNs_tags is None:
            XNs_tags = CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        #-------------------------
        cols_to_sum = CPXDfBuilder.get_non_XNs_cols_from_cpx_df(
            cpx_df   = cpx_df, 
            XNs_tags = XNs_tags
        )
        #-------------------------
        if len(cols_to_sum)==0:
            return None
        #-------------------------
        if cpx_df.columns.nlevels == 2:
            assert(cpx_df.columns.get_level_values(0).nunique() == 1)
            if not isinstance(output_col, tuple):
                output_col = (cpx_df.columns.get_level_values(0).unique().tolist()[0], output_col)
        #-------------------------
        # If output_col (i.e., a totals column) is already contained in cpx_df, we don't want to double count!
        # Without this exclusion, one would see double the total_counts in reality!!!!!
        if output_col in cpx_df.columns:
            cols_to_sum.remove(output_col)
        #-------------------------
        total_df = cpx_df[cols_to_sum]
        #-------------------------
        total_df = total_df.sum(axis=1).to_frame(name=output_col)
        if sort_output:
            total_df = total_df.sort_values(by=output_col, ascending=False)
        return total_df


    #----------------------------------------------------------------------------------------------------
    #TODO Probably rename this, as it's for combining different time_grps into a single merged_df
    #     Not merging in the typical sense
    @staticmethod
    def get_merged_cpx_df_subset_below_max_total_counts(
        merged_cpx_df         , 
        max_total_counts      ,
        how                   = 'any', 
        XNs_tags              = None, 
        non_reason_lvl_0_vals = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        Similar to CPXDf.get_cpx_df_subset_below_max_total_counts, but for merged cpx DFs.

        NOT INTENDED FOR case where merged_cpx_df built from rcpx_df_OGs.  In such a case, merged_cpx_df.columns.nlevels
          will equal 3, and the assertion below will fail.
          It would not be terribly difficult to expand the functionality to include this case, but I don't see this really
            being needed in the future.
            If one did want to expand the functionality, the main issue would be how to treat each of the total_event_counts,
              as they will each have two columns (one for raw and one for norm) instead of one.
              The easiest method would be to use just one of the columns, but one could definitely develop something more complicated
              involving both.

        max_total_counts:
          This can either be a int/float or a dict
          (1). If this is a single int/float, it be the max_total_counts argument for all mecpx_an_keys in the merged_cpx_df.
          (2). If this is a dict, then unique max_total_counts values can be set for each mecpx_an_key.
               In this case, each key in max_total_counts must be contained in mecpx_an_keys.  However, not all mecpx_an_keys must
                 be included in max_total_counts.  If a mecpx_an_key is excluded from max_total_counts, its value will be set to None
               In the more general case, mecpx_an_keys are actually the merged_cpx_df.columns.get_level_values(0).unique() values.

        how:
          Should be equal to 'any' or 'all'.
          Determine if row is removed from DataFrame, when we have at least one value exceeding the max total counts 
            or all exceeding the max total counts
            'any' : If any exceeding the max total counts are present, drop that row.
            'all' : If all values are exceeding the max total counts, drop that row.

        XNs_tags:
          Only used if max_total_counts is not None
          If None: XNs_tags = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()
        """
        #-------------------------
        if XNs_tags is None:
            XNs_tags = CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        #-------------------------
        assert(merged_cpx_df.columns.nlevels==2)
        assert(how=='any' or how=='all')
        mecpx_an_keys = merged_cpx_df.columns.get_level_values(0).unique().tolist()
        #-------------------------
        # First, determine whether we're dealing with scenario 1 or scenario 2 for max_total_counts (described above)
        assert(Utilities.is_object_one_of_types(max_total_counts, [int, float, dict]))
        if not isinstance(max_total_counts, dict):
            # Scenario 1
            # Build max_total_counts_dict as a dict with one key for each mecpx_an_keys
            max_total_counts_dict = {mecpx_an_key:max_total_counts for mecpx_an_key in mecpx_an_keys}
        else:
            # Scenario 2
            max_total_counts_dict = max_total_counts
            # Assert that all max_total_counts_dict keys are in mecpx_an_keys
            assert(len(set(max_total_counts_dict.keys()).difference(mecpx_an_keys))==0)

            # If any of mecpx_an_keys are not included in max_total_counts_dict, set their value equal to None
            # Thus ensuring each of mecpx_an_keys is contained in max_total_counts_dict
            for mecpx_an_key in mecpx_an_keys:
                if mecpx_an_key not in max_total_counts_dict.keys():
                    max_total_counts_dict[mecpx_an_key] = None
        #-------------------------
        # At this point, regardless of whether scario 1 or 2, max_total_counts_dict should be a dict with keys equal
        #   to mecpx_an_keys
        assert(isinstance(max_total_counts_dict, dict))
        assert(len(set(max_total_counts_dict.keys()).symmetric_difference(mecpx_an_keys))==0)
        #-------------------------
        # Build the truth series (all_truths) and use it to project out the desired rows
        truth_series = []
        for mecpx_an_key in mecpx_an_keys:
            if max_total_counts_dict[mecpx_an_key] is None:
                continue
            total_event_counts_i = CPXDfBuilder.get_total_event_counts(
                cpx_df                = merged_cpx_df[mecpx_an_key], 
                output_col            = 'total_counts', 
                sort_output           = False, 
                XNs_tags              = XNs_tags, 
                non_reason_lvl_0_vals = non_reason_lvl_0_vals
            )
            assert(total_event_counts_i.shape[1]==1)
            truth_series_i = total_event_counts_i.squeeze()<max_total_counts_dict[mecpx_an_key]
            # Make sure truth_series_i index matches that of merged_cpx_df
            # I don't think this is absolutely necessary, as pandas should align the indices when these truth
            #   series are actually used.  However, there is no harm in performing the check, and having everything
            #   in order already should speed up the implementation of the truth series
            assert(all(truth_series_i.index==merged_cpx_df.index))
            truth_series.append(truth_series_i)
        # Combine all truth_series into all_truths DF
        all_truths = pd.concat(truth_series, axis=1, ignore_index=True)
        #-------------------------
        if how=='any':
            # If any exceed the max total counts, drop the row, otherwise keep
            # ==> Only keep if all pass the cut, i.e., only if all in row are True
            # ==> use all
            all_truths = all_truths.all(axis=1)
        elif how=='all':
            # If all exceed the max total counts, drop the row, otherwise keep
            # ==> Keep if even a single column passes the cut
            # ==> use any
            all_truths = all_truths.any(axis=1)
        else:
            assert(0)
        #-------------------------
        # At this pount, all_truths is a pd.Series object
        # Double check indices align
        assert(all(all_truths.index==merged_cpx_df.index))
        merged_cpx_df = merged_cpx_df[all_truths]
        #-------------------------
        return merged_cpx_df
    

    #----------------------------------------------------------------------------------------------------
    #TODO Probably rename this, as it's for combining different time_grps into a single merged_df
    #     Not merging in the typical sense
    @staticmethod 
    def merge_cpx_dfs(
        dfs_coll                      , 
        col_level                     = -1,
        max_total_counts              = None, 
        how_max_total_counts          = 'any', 
        XNs_tags                      = None, 
        cols_to_init_with_empty_lists = None, 
        make_cols_equal               = True, 
        sort_cols                     = True, 
    ):
        r"""
        Returns a single DF consisting of the DFs from dfs merged.

        dfs_coll:
            This may be a list of pd.DataFrame objects or a dictionary with pd.DataFrame values.
            The function was originally developed for the latter, hence why dfs_coll is immediately converted to dfs_dict in the code
            type(dfs_coll)==dict:
                In order to merge all (since all should have the same columns) the mecpx_an_key for each is added as a level 0
                  value for MultiIndex columns.
                The keys of dfs_coll will be used as the mecpx_an_keys
                NOTE: This is true even if dfs contained in values already have MultiIndex columns!
            type(dfs_coll)==list:
                The dfs contained must have MultiIndex columns, and the level-0 for each must contain a single value which is unique from the others.            

        In order to merge all (since all should have the same columns) the mecpx_an_key for each is added as a level 0
          value for MultiIndex columns
    
        max_total_counts:
          This can either be a int/float or a dict (or None!)
          (0). If max_total_counts is None, no cuts imposed
          (1). If this is a single int/float, it be the max_total_counts argument for all mecpx_an_keys in the merged_cpx_df.
          (2). If this is a dict, then unique max_total_counts values can be set for each mecpx_an_key.
               In this case, each key in max_total_counts must be contained in mecpx_an_keys.  However, not all mecpx_an_keys must
                 be included in max_total_counts.  If a mecpx_an_key is excluded from max_total_counts, its value will be set to None
               In the more general case, mecpx_an_keys are actually the merged_cpx_df.columns.get_level_values(0).unique() values.
          !!!!! NOTE !!!!!
            max_total_counts criteria/on are/is implemented AT THE END.  Using the max_total_counts_args argument in get_cpx_dfs_dict would be incorrect.
            This is due to the merge method by which all DFs are joined.  When an index is present in one of the DFs (for a given mecpx_an_key)
              to be merged, but not in another, the merge process fills in the values for the missing DF with NaNs, which are then converted to 0.0.
            Therefore, if an index were excluded from a DF in dfs_dict due to failing the max_total_counts_args criterion, in the final merged
              DF it would appear as if this index simply had no events for that mecpx_an_key (when, in reality, it has many!)
              
        how_max_total_counts:
          Should be equal to 'any' or 'all'.  Determines how max_total_counts are imposed.
          Determine if row is removed from DataFrame, when we have at least one value exceeding the max total counts 
            or all exceeding the max total counts
            'any' : If any exceeding the max total counts are present, drop that row.
            'all' : If all values are exceeding the max total counts, drop that row.
            
        XNs_tags:
          Only used if max_total_counts is not None
          If None: XNs_tags = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()

        cols_to_init_with_empty_lists:
            If None: cols_to_init_with_empty_lists = CPXDf.std_XNs_cols()
        """
        #--------------------------------------------------
        if XNs_tags is None:
            XNs_tags = CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        if cols_to_init_with_empty_lists is None:
            cols_to_init_with_empty_lists = CPXDfBuilder.std_XNs_cols()

        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(dfs_coll, [dict, list]))
        if isinstance(dfs_coll, dict):
            prepend_keys = True
            dfs_dict     = dfs_coll
        else:
            prepend_keys = False
            dfs_dict     = {}
            # Make sure all DFs have same number of levels in columns (>1)
            cols_n_levels = dfs_coll[0].columns.nlevels
            assert(cols_n_levels>1)
            assert(np.all([x.columns.nlevels==cols_n_levels for x in dfs_coll]))
            for df_i in dfs_coll:
                assert(df_i.columns.get_level_values(0).nunique()==1)
                unq_val_0_i = df_i.columns.get_level_values(0).unique().tolist()[0]
                assert(unq_val_0_i not in dfs_dict.keys())
                dfs_dict[unq_val_0_i] = df_i
        #--------------------------------------------------

        #--------------------------------------------------
        # #-------------------------
        # # Make sure DFs only have single level in columns so that make_reason_counts_per_outg_columns_equal_dfs_list can
        # #       be used out of the box.  The level 0 values will be added later
        # assert(np.all([x.columns.nlevels==1 for x in dfs_dict.values()]))
        # #-------------------------
        # # Make sure all have same columns and order
        # AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list(
        #     rcpx_dfs                      = list(dfs_dict.values()), 
        #     same_order                    = True,
        #     inplace                       = True, 
        #     cols_to_init_with_empty_lists = cols_to_init_with_empty_lists
        # )
        #-------------------------
        # Make sure all DFs have same number of levels in columns
        cols_n_levels = list(dfs_dict.values())[0].columns.nlevels
        assert(np.all([x.columns.nlevels==cols_n_levels for x in dfs_dict.values()]))
        #-------------------------
        if make_cols_equal:
            # Make sure all have same columns and order
            AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list(
                rcpo_dfs                      = list(dfs_dict.values()), 
                col_level                     = col_level,
                same_order                    = True,
                inplace                       = True, 
                cols_to_init_with_empty_lists = cols_to_init_with_empty_lists
            )
        #-------------------------
        # Add mecpx_an_key as level 0 value for MultiIndex columns
        # Build dfs list
        dfs = []
        expected_final_idxs = [] # Built so number of columns can be checked at end
        for mecpx_an_key_i, df_i in dfs_dict.items():
            if prepend_keys:
                df_i = Utilities_df.prepend_level_to_MultiIndex(
                    df         = df_i, 
                    level_val  = mecpx_an_key_i, 
                    level_name = 'mecpx_an_key', 
                    axis       = 1
                )
            dfs.append(df_i)
            expected_final_idxs.extend(df_i.index.tolist())
        #-------------------------
        # Merge all DFs together    
        merged_df = reduce(lambda left,right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
    
        # Make sure merged_df has expected number of columns
        # assert(merged_df.shape[1]==dfs[0].shape[1]*len(dfs))
    
        # Make sure merged_df has expected number of rows
        expected_final_idxs = list(set(expected_final_idxs))
        assert(merged_df.shape[0]==len(expected_final_idxs))
        #-------------------------
        # Replace all NaN values with 0.0
        # NOTE: NaN values result because the indices (outg_rec_nb) of the DFs are in general different.
        #       This happens when, e.g., the meters for index_1 (outg_rec_nb_1) have events 0-5 days out, but
        #         have no events 6-10 days out.  In such a case, the columns for 6-10 days out would all be NaN.
        #         Thus, it makes sense to replace these with 0.0
        merged_df = merged_df.fillna(0.0)
        #-------------------------
        if max_total_counts is not None:
            merged_df = CPXDfBuilder.get_merged_cpx_df_subset_below_max_total_counts(
                merged_cpx_df    = merged_df, 
                max_total_counts = max_total_counts, 
                how              = how_max_total_counts, 
                XNs_tags         = XNs_tags
            )
        #-------------------------
        if sort_cols:
            merged_df = merged_df.sort_index(axis=1)
        #-------------------------
        return merged_df

    #----------------------------------------------------------------------------------------------------
    #TODO Probably rename this, as it's for combining different time_grps into a single merged_df
    #     Not merging in the typical sense
    def get_merged_cpx_dfs(
        self, 
        cpx_df_name                         , 
        cpx_df_subset_by_mjr_mnr_cause_args = None, 
        max_total_counts                    = None, 
        how_max_total_counts                = 'any', 
        XNs_tags                            = None, 
        cols_to_init_with_empty_lists       = None, 
        make_cols_equal                     = True, 
        sort_cols                           = True, 
    ):
        r"""
        Like get_cpx_dfs, but instead of returning a list of DFs, this returns a single DF consisting of
          the DFs from get_cpx_dfs merged.
        In order to merge all (since all should have the same columns) the mecpx_an_key for each is added as a level 0
          value for MultiIndex columns

        max_total_counts:
          This can either be a int/float or a dict (or None!)
          (0). If max_total_counts is None, no cuts imposed
          (1). If this is a single int/float, it be the max_total_counts argument for all mecpx_an_keys in the merged_cpx_df.
          (2). If this is a dict, then unique max_total_counts values can be set for each mecpx_an_key.
               In this case, each key in max_total_counts must be contained in mecpx_an_keys.  However, not all mecpx_an_keys must
                 be included in max_total_counts.  If a mecpx_an_key is excluded from max_total_counts, its value will be set to None
               In the more general case, mecpx_an_keys are actually the merged_cpx_df.columns.get_level_values(0).unique() values.
          !!!!! NOTE !!!!!
            max_total_counts criteria/on are/is implemented AT THE END.  Using the max_total_counts_args argument in get_cpx_dfs_dict would be incorrect.
            This is due to the merge method by which all DFs are joined.  When an index is present in one of the DFs (for a given mecpx_an_key)
              to be merged, but not in another, the merge process fills in the values for the missing DF with NaNs, which are then converted to 0.0.
            Therefore, if an index were excluded from a DF in dfs_dict due to failing the max_total_counts_args criterion, in the final merged
              DF it would appear as if this index simply had no events for that mecpx_an_key (when, in reality, it has many!)
              
        how_max_total_counts:
          Should be equal to 'any' or 'all'.  Determines how max_total_counts are imposed.
          Determine if row is removed from DataFrame, when we have at least one value exceeding the max total counts 
            or all exceeding the max total counts
            'any' : If any exceeding the max total counts are present, drop that row.
            'all' : If all values are exceeding the max total counts, drop that row.
            
        XNs_tags:
          Only used if max_total_counts is not None
          If None: XNs_tags = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()

        cols_to_init_with_empty_lists:
            If None: cols_to_init_with_empty_lists = CPXDf.std_XNs_cols()
        """
        #--------------------------------------------------
        if XNs_tags is None:
            XNs_tags = CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
        if cols_to_init_with_empty_lists is None:
            cols_to_init_with_empty_lists = CPXDfBuilder.std_XNs_cols()
        #-------------------------
        # Grab DFs in dict object, where key is mecpx_an_key
        # NOTE: add_an_key_as_level_0_col=False below so that make_reason_counts_per_outg_columns_equal_dfs_list can
        #       be used out of the box.  The level 0 values will be added later
        dfs_dict = self.get_cpx_dfs_dict(
            cpx_df_name                         = cpx_df_name, 
            cpx_df_subset_by_mjr_mnr_cause_args = cpx_df_subset_by_mjr_mnr_cause_args, 
            max_total_counts_args               = None, 
            add_an_key_as_level_0_col           = False
        )
        #-------------------------
        merged_df = CPXDfBuilder.merge_cpx_dfs(
            dfs_coll                      = dfs_dict, 
            col_level                     = -1,
            max_total_counts              = max_total_counts, 
            how_max_total_counts          = how_max_total_counts, 
            XNs_tags                      = XNs_tags, 
            cols_to_init_with_empty_lists = cols_to_init_with_empty_lists, 
            make_cols_equal               = make_cols_equal,
            sort_cols                     = sort_cols,
        )
        #-------------------------
        return merged_df

    

    #------------------------------------------------------------------------------------------------------------------------------------------
    # Methods for building from end events (from meter_events.end_device_event)
    #------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def perform_end_events_std_col_renames_and_drops(
        end_events_df              , 
        cols_to_drop               = None, 
        rename_cols_dict           = None, 
        make_all_columns_lowercase = True
    ):
        r"""
        """
        #-------------------------
        end_events_df = Utilities_df.drop_unnamed_columns(
            df            = end_events_df, 
            regex_pattern = r'Unnamed.*', 
            ignore_case   = True, 
            inplace       = True, 
        )
        #-------------------------
        if make_all_columns_lowercase:
            end_events_df = Utilities_df.make_all_column_names_lowercase(end_events_df)
        #-------------------------
        # Drop any columns from cols_to_drop
        # IMPORTANT: This should come BEFORE rename
        if cols_to_drop is not None:
            end_events_df = end_events_df.drop(columns=cols_to_drop)
        #-------------------------
        # Rename any columns from rename_cols_dict
        # IMPORTANT: This should come AFTER drop
        if rename_cols_dict is not None:
            end_events_df = end_events_df.rename(columns=rename_cols_dict)
        #-------------------------
        return end_events_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def check_end_events_df_merge_with_mp(
        end_events_df              ,
        mp_df                      , 
        threshold_pct              = 1.0, 
        outg_rec_nb_col            = 'outg_rec_nb',
        trsf_pole_nb_col           = 'trsf_pole_nb', 
        prem_nb_col                = 'aep_premise_nb', 
        serial_number_col          = 'serialnumber',
        mp_df_cols                 = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        cols_to_drop               = None, 
        rename_cols_dict           = None, 
        make_all_columns_lowercase = True, 
    ):
        r"""
        When building the RCPX dfs, oftentimes the end events data need to be joined with meter premise data (e.g., when one
          wants to group by transformer, this information typically needs to be brought in from MP).
        The meter premise data are supplied by the user.  Mistakes can be made, so this function serves to check whether the 
          alignment between end events and meter premise is as expected.
          
        NOTE: This also runs some other checks on the end events collection as a whole (e.g., if assert_all_cols_equal, it
                checks the assertion).
              Thus, some of the operations can be replaced from the for loop in build_reason_counts_per_outage_from_csv
                (This will help performance, but I don't think gains will be significant)
                
        In order to reduce some of the load on build_reason_counts_per_outage_from_csv, this function returns merge_on_ede and merge_on_mp
          
        End events and meter premise are merged on serial number and premise number.
          If data are for outages, also merged on outg_rec_nb.
          If trsf_pole_nb is present, also merged on it.
          
        NOTE:  I have found there are meters which are 'missing' from meter premise but are present in end events.
               These seem to be caused by meters which were removed or installed close to (a few days/weeks before) an outage.
               Therefore, although the meter may not have been present at the time of the outage (and therefore was exluced from
                 meter premise), it could have registered events leading up to/following the outage.
               e.g., if a meter was removed in the days before an outage, end events are still found for this meter in the days leading up to the outage
               e.g., if a meter was installed in the days after an outage, end events are still found for this meter in the days following an outage.
               How should these be handled?
               The simplest method, which I will implement for now, is to simply ONLY consider those meters which were present
                 at the time of the outage.  THEREFORE, the two DFs should be joined with an inner merge! 
                 
        NOTE: Utilities_df.make_all_column_names_lowercase utilized when reading in end events dfs because...
          EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
          Athena maintains the original case, and does not conver to lower case,
            so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
                 
        threshold_pct:
            If the percentage of entries present in end events but absent in meter premise exceeds this value, the program will crash.
            This is to safeguard the user from inputting incorrect meter premise data.
        """
        #-------------------------
        merge_on_mp  = [mp_df_cols['serial_number_col'], mp_df_cols['prem_nb_col']]
        merge_on_ede = [serial_number_col, prem_nb_col]
        if(
            mp_df_cols['outg_rec_nb_col'] in mp_df.columns.tolist() and 
            outg_rec_nb_col               in end_events_df.columns.tolist()
        ):
            merge_on_mp.append(mp_df_cols['outg_rec_nb_col'])
            merge_on_ede.append(outg_rec_nb_col)
        #-------------------------
        end_events_df = CPXDfBuilder.perform_end_events_std_col_renames_and_drops(
            end_events_df              = end_events_df, 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-------------------------
        if trsf_pole_nb_col in end_events_df.columns:
            assert(mp_df_cols['trsf_pole_nb_col'] in mp_df.columns)
            merge_on_mp.append(mp_df_cols['trsf_pole_nb_col'])
            merge_on_ede.append(trsf_pole_nb_col)
        #-------------------------
        # Below ensures there is only one entry per 'meter' (meter here is defined by a unique grouping of merge_on_mp)
        assert(not any(mp_df.groupby(merge_on_mp).size()>1))         
        #-------------------------
        # Alter dtypes in mp_df if needed for proper merging
        end_events_df, mp_df = Utilities_df.make_df_col_dtypes_equal(
            df_1              = end_events_df, 
            col_1             = merge_on_ede, 
            df_2              = mp_df, 
            col_2             = merge_on_mp, 
            allow_reverse_set = False, 
            assert_success    = True
        )
        #-------------------------
        # Now, compare the unique combinations of merge_on_ede in end_events_df to those of merge_on_mp in mp_df
        gps_ede = list(end_events_df.groupby(merge_on_ede).groups.keys())
        gps_mp  = list(mp_df.groupby(merge_on_mp).groups.keys())
        # Finally, find the percent of meters in end events missing from meter premise (where a meter is defined as a unique
        #   combination of merge_on_ede values (e.g., serial number and premise number))
        n_missing   = len(set(gps_ede).difference(set(gps_mp)))
        pct_missing = 100*(n_missing/len(gps_ede))
        print(f'% meters in end events missing from mp_df: {pct_missing}')
        if pct_missing>threshold_pct:
            assert(0)
        #-------------------------
        return mp_df, merge_on_ede, merge_on_mp


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def check_end_events_from_CSVs_merge_with_mp(
        ede_file_paths             ,
        mp_df                      , 
        threshold_pct              = 1.0, 
        outg_rec_nb_col            = 'outg_rec_nb',
        trsf_pole_nb_col           = 'trsf_pole_nb', 
        prem_nb_col                = 'aep_premise_nb', 
        serial_number_col          = 'serialnumber',
        mp_df_cols                 = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        assert_all_cols_equal      = True, 
        cols_to_drop               = None, 
        rename_cols_dict           = None, 
        make_all_columns_lowercase = True, 
    ):
        r"""
        When building the RCPX dfs, oftentimes the end events data need to be joined with meter premise data (e.g., when one
          wants to group by transformer, this information typically needs to be brought in from MP).
        The meter premise data are supplied by the user.  Mistakes can be made, so this function serves to check whether the 
          alignment between end events and meter premise is as expected.
          
        NOTE: This also runs some other checks on the end events collection as a whole (e.g., if assert_all_cols_equal, it
                checks the assertion).
              Thus, some of the operations can be replaced from the for loop in build_reason_counts_per_outage_from_csv
                (This will help performance, but I don't think gains will be significant)
                
        In order to reduce some of the load on build_reason_counts_per_outage_from_csv, this function returns merge_on_ede and merge_on_mp
          
        End events and meter premise are merged on serial number and premise number.
          If data are for outages, also merged on outg_rec_nb.
          If trsf_pole_nb is present, also merged on it.
          
        NOTE:  I have found there are meters which are 'missing' from meter premise but are present in end events.
               These seem to be caused by meters which were removed or installed close to (a few days/weeks before) an outage.
               Therefore, although the meter may not have been present at the time of the outage (and therefore was exluced from
                 meter premise), it could have registered events leading up to/following the outage.
               e.g., if a meter was removed in the days before an outage, end events are still found for this meter in the days leading up to the outage
               e.g., if a meter was installed in the days after an outage, end events are still found for this meter in the days following an outage.
               How should these be handled?
               The simplest method, which I will implement for now, is to simply ONLY consider those meters which were present
                 at the time of the outage.  THEREFORE, the two DFs should be joined with an inner merge! 
                 
        NOTE: Utilities_df.make_all_column_names_lowercase utilized when reading in end events dfs because...
          EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
          Athena maintains the original case, and does not conver to lower case,
            so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
                 
        threshold_pct:
            If the percentage of entries present in end events but absent in meter premise exceeds this value, the program will crash.
            This is to safeguard the user from inputting incorrect meter premise data.
        """
        #-------------------------
        merge_on_mp  = [mp_df_cols['serial_number_col'], mp_df_cols['prem_nb_col']]
        merge_on_ede = [serial_number_col, prem_nb_col]
        #-------------------------
        # Grab the first end events df and check for trsf_pole_nb
        #  Actually, just need to grab first entry of first df
        end_events_df_0 = GenAn.read_df_from_csv(
            read_path                      = ede_file_paths[0], 
            cols_and_types_to_convert_dict = None, 
            to_numeric_errors              = 'coerce', 
            drop_na_rows_when_exception    = True, 
            drop_unnamed0_col              = True, 
            pd_read_csv_kwargs             = dict(nrows=1)
        )
        #-------------------------
        end_events_df_0 = CPXDfBuilder.perform_end_events_std_col_renames_and_drops(
            end_events_df              = end_events_df_0, 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-------------------------
        if(
            mp_df_cols['outg_rec_nb_col'] in mp_df.columns.tolist() and 
            outg_rec_nb_col               in end_events_df_0.columns.tolist()
        ):
            merge_on_mp.append(mp_df_cols['outg_rec_nb_col'])
            merge_on_ede.append(outg_rec_nb_col)
        #-------------------------
        if trsf_pole_nb_col in end_events_df_0.columns:
            assert(mp_df_cols['trsf_pole_nb_col'] in mp_df.columns)
            merge_on_mp.append(mp_df_cols['trsf_pole_nb_col'])
            merge_on_ede.append(trsf_pole_nb_col)
        #-------------------------
        # Below ensures there is only one entry per 'meter' (meter here is defined by a unique grouping of merge_on_mp)
        assert(not any(mp_df.groupby(merge_on_mp).size()>1))
        #--------------------------------------------------
        # Build DF with all end events to be checked with mp_df
        # The only columns which matter for the check are those in merge_on_ede, so those are the only kept.
        #   This keeps the overall size of the end_events_df smaller, allowing all to be loaded at once in most cases.
        dfs = []
        for path in ede_file_paths:
            df_i = GenAn.read_df_from_csv(
                read_path                      = path, 
                cols_and_types_to_convert_dict = None, 
                to_numeric_errors              = 'coerce', 
                drop_na_rows_when_exception    = True, 
                drop_unnamed0_col              = True
            )
            if df_i.shape[0]==0:
                continue
            #-------------------------
            df_i = CPXDfBuilder.perform_end_events_std_col_renames_and_drops(
                end_events_df              = df_i, 
                cols_to_drop               = cols_to_drop, 
                rename_cols_dict           = rename_cols_dict, 
                make_all_columns_lowercase = make_all_columns_lowercase
            )
            #-------------------------
            dfs.append(df_i[merge_on_ede])
        #-------------------------                
        df_cols = Utilities_df.get_shared_columns(dfs, maintain_df0_order=True)
        for i in range(len(dfs)):
            if assert_all_cols_equal:
                # In order to account for case where columns are the same but in different order
                # one must compare the length of dfs[i].columns to that of df_cols (found by utilizing
                # the Utilities_df.get_shared_columns(dfs) functionality)
                assert(dfs[i].shape[1]==len(df_cols))
            dfs[i] = dfs[i][df_cols]
        end_events_df = pd.concat(dfs)
        #-------------------------
        # Alter dtypes in mp_df if needed for proper merging
        end_events_df, mp_df = Utilities_df.make_df_col_dtypes_equal(
            df_1              = end_events_df, 
            col_1             = merge_on_ede, 
            df_2              = mp_df, 
            col_2             = merge_on_mp, 
            allow_reverse_set = False, 
            assert_success    = True
        )
        #-------------------------
        # Now, compare the unique combinations of merge_on_ede in end_events_df to those of merge_on_mp in mp_df
        gps_ede = list(end_events_df.groupby(merge_on_ede).groups.keys())
        gps_mp  = list(mp_df.groupby(merge_on_mp).groups.keys())
        # Finally, find the percent of meters in end events missing from meter premise (where a meter is defined as a unique
        #   combination of merge_on_ede values (e.g., serial number and premise number))
        n_missing = len(set(gps_ede).difference(set(gps_mp)))
        pct_missing = 100*(n_missing/len(gps_ede))
        print(f'% meters in end events missing from mp_df: {pct_missing}')
        if pct_missing>threshold_pct:
            assert(0)
        #-------------------------
        return mp_df, merge_on_ede, merge_on_mp

    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_rename_dict_for_gpd_for_sql_cols(df, col_level=-1):
        r"""
        The _gpd_for_sql appendix is added in the end events data acquisition stage, but is typically no longer needed after.
        NOTE: Method is case insensitive!! (i.e., col_gpd_for_sql --> col, col_GPD_FOR_SQL --> col, etc.)
        Returns a dict with key values equal to the found columns containing _gpd_for_sql and values w/o the _gpd_for_sql appendix.
        The dict can be used, e.g., for a rename call.
        """
        #-------------------------
        found_cols_dict = {}
        for col in df.columns.get_level_values(col_level).tolist():
            if col.lower().endswith('_gpd_for_sql'):
                assert(col not in found_cols_dict.keys())
                found_cols_dict[col] = col[:-12]
        return found_cols_dict
    
    #---------------------------------------------------------------------------
    @staticmethod
    def find_gpd_for_sql_cols(df, col_level=-1):
        r"""
        The _gpd_for_sql appendix is added in the end events data acquisition stage, but is typically no longer needed after.
        NOTE: Method is case insensitive!! (i.e., col_gpd_for_sql --> col, col_GPD_FOR_SQL --> col, etc.)
        Returns a list of the columns containing _gpd_for_sql
        """
        #-------------------------
        found_cols_dict = CPXDfBuilder.get_rename_dict_for_gpd_for_sql_cols(df=df, col_level=col_level)
        found_cols      = list(found_cols_dict.keys())
        return found_cols
    
    #---------------------------------------------------------------------------
    @staticmethod
    def rename_all_cols_with_gpd_for_sql_appendix(df):
        r"""
        """
        #-------------------------
        found_cols_dict = CPXDfBuilder.get_rename_dict_for_gpd_for_sql_cols(df=df)
        df = df.rename(columns=found_cols_dict)
        return df
    
    #---------------------------------------------------------------------------
    @staticmethod
    def rename_all_index_names_with_gpd_for_sql_appendix(df):
        r"""
        The _gpd_for_sql appendix is added in the end events data acquisition stage, but is typically no longer needed after.
        This function removes the appendix.
        
        NOTE: Method is case insensitive!! (i.e., col_gpd_for_sql --> col, col_GPD_FOR_SQL --> col, etc.)
        """
        #-------------------------
        idx_names_new = []
        for name in df.index.names:
            if name.lower().endswith('_gpd_for_sql'):
                idx_names_new.append(name[:-12])
            else:
                idx_names_new.append(name)
        #-----
        assert(len(df.index.names)==len(idx_names_new))
        df.index.names = idx_names_new
        return df


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def make_values_lowercase(input_dict):
        r"""
        Only works for dicts with string or list values!
        """
        #-------------------------
        assert(isinstance(input_dict, dict))
        #-------------------------
        # Remove any elements with None values
        input_dict = {k:v for k,v in input_dict.items() if v is not None}
        #-------------------------
        # This only makes sense if all values are strings!
        if not Utilities.are_all_list_elements_one_of_types(lst=list(input_dict.values()), types=[str, list, tuple]):
            print(f"CPXDfBuilder.make_values_lowercase, cannot continue, all values not of type string or list!\ninput_dict = {input_dict}")
            return input_dict
        #-------------------------
        output_dict = {}
        for k,v in input_dict.items():
            assert(k not in output_dict.keys())
            if isinstance(v, str):
                output_dict[k] = v.lower()
            else:
                assert(Utilities.is_object_one_of_types(v, [list,tuple]))
                output_dict[k] = [v_i.lower() for v_i in v]
        #-------------------------
        return output_dict
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def identify_ede_cols_of_interest_to_update_andor_drop(
        end_events_df,  
        grp_by_cols                = 'outg_rec_nb', 
        outg_rec_nb_col            = 'outg_rec_nb',
        trsf_pole_nb_col           = 'trsf_pole_nb', 
        prem_nb_col                = 'aep_premise_nb', 
        serial_number_col          = 'serialnumber',
        trust_sql_grouping         = True, 
        drop_gpd_for_sql_appendix  = True, 
        make_all_columns_lowercase = True
    ):
        r"""
        Find any entries in ede_cols_of_interest (see below) which contain _gpd_for_sql appendix and update accordingly.
          The columns in end events may sometimes be appended with _gpd_for_sql (e.g., typically one will see outg_rec_nb_gpd_for_sql 
            not outg_rec_nb in the raw CSV files).
          The _gpd_for_sql was appended during data acquisition.
          However, the user typically does not remember this, and will usually input, e.g., outg_rec_nb_col='outg_rec_nb'
            Such a scenario will obviously lead to an error, as 'outg_rec_nb' is not found in the columns
          The below methods serve to remedy that issue.
          
        ede_cols_of_interest consist of grp_by_cols+[outg_rec_nb_col, trsf_pole_nb_col, prem_nb_col, serial_number_col]
          
        NOTE: In the case that column_i and column_i_gpd_for_sql are for whatever reason BOTH found in the DF, the parameter
            trust_sql_grouping directs the code on which to use (the other will be dropped)

        -------------------------
        TYPICAL EXAMPLE:
            Input:
                end_events_df.columns = [
                    'issuertracking_id', 'serialnumber', 'enddeviceeventtypeid', 'valuesinterval', 'aep_premise_nb', 'reason', 'event_type', 'aep_opco', 
                    'aep_event_dt', 'trsf_pole_nb', 'trsf_pole_nb_gpd_for_sql', 'no_outg_rec_nb_gpd_for_sql', 'is_first_after_outg_gpd_for_sql'
                ]
                grp_by_cols        = ['trsf_pole_nb', 'no_outg_rec_nb']
                outg_rec_nb_col    = 'no_outg_rec_nb'
                trsf_pole_nb_col   = 'trsf_pole_nb'
                prem_nb_col        = 'aep_premise_nb'
                serial_number_col  = 'serialnumber'
                trust_sql_grouping = True

            Output:
                ede_gpd_coi_dict_updates = {
                    'grp_by_cols'      : ['trsf_pole_nb_gpd_for_sql', 'no_outg_rec_nb_gpd_for_sql'], 
                    'outg_rec_nb_col'  : 'outg_rec_nb', 
                    'trsf_pole_nb_col' : 'trsf_pole_nb_gpd_for_sql', 
                    'prem_nb_col'      : 'aep_premise_nb', 
                    'serial_number_col': 'serialnumber'
                }
                cols_to_drop = ['trsf_pole_nb']
        -------------------------

        drop_gpd_for_sql_appendix:
            If True, any remaining cols of interest/columns with gpd_for_sql appendix will be removed (cols of interest) or instructed
              to have their names changed (columns)
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]    
        #-------------------------
        # Below called ede_gpd_coi_dict because the columns of interest are grouped by their input parameter in build_rcpx_df_from_EndEvents_in_csvs.
        #   e.g., key value grp_by_cols should contain a list of columns (strings)
        #         key value outg_rec_nb_col contains a single string for the outg_rec_nb_col
        # The values, coi, can stand for column of interest (e.g., outg_rec_nb_col) or columns of interest (e.g., grp_by_cols)
        ede_gpd_coi_dict = dict(
            grp_by_cols       = grp_by_cols, 
            outg_rec_nb_col   = outg_rec_nb_col, 
            trsf_pole_nb_col  = trsf_pole_nb_col, 
            prem_nb_col       = prem_nb_col, 
            serial_number_col = serial_number_col
        )
        #-------------------------
        if make_all_columns_lowercase: 
            end_events_df    = Utilities_df.make_all_column_names_lowercase(end_events_df) 
            ede_gpd_coi_dict = CPXDfBuilder.make_values_lowercase(input_dict = ede_gpd_coi_dict)
        #-------------------------
        # Below, ede_cols_of_interest is essentially the flattened values from ede_gpd_coi_dict with duplicates removed
        ede_cols_of_interest = []
        for ede_coi_grp, coi in ede_gpd_coi_dict.items():
            assert(Utilities.is_object_one_of_types(coi, [str,list,tuple]))
            if isinstance(coi, str):
                ede_cols_of_interest.append(coi)
            else:
                ede_cols_of_interest.extend(coi)
        # NOTE: grp_by_cols can contain the others (e.g., outg_rec_nb_col), so set operation needed to removed duplicates
        ede_cols_of_interest = list(set(ede_cols_of_interest))
        #--------------------------------------------------
        # Below, found_cols_w_gpd_for_sql_appendix has keys equal to any columns found containing _gpd_for_sql appendix and
        #   value equal to the column without the appendix.
        # found_cols_w_gpd_for_sql_appendix_inv is the inverse (i.e., keys and values switched)
        found_cols_w_gpd_for_sql_appendix     = CPXDfBuilder.get_rename_dict_for_gpd_for_sql_cols(end_events_df)
        found_cols_w_gpd_for_sql_appendix_inv = Utilities.invert_dict(found_cols_w_gpd_for_sql_appendix)
        #--------------------------------------------------
        # NOTE: No harm below if ede_coi not actually found in end_events_df (e.g., when running over baseline data, outg_rec_nb_col 
        #       will not be found), as updates are only made to those whose values change (and can only change if found)
        ede_cols_of_interest_updates = dict()
        cols_to_drop = []
        #-----
        for ede_coi in ede_cols_of_interest:
            assert(ede_coi not in ede_cols_of_interest_updates)
            #-----
            if ede_coi in found_cols_w_gpd_for_sql_appendix_inv.keys():
                # f'{ede_coi}_gpd_for_sql' found in end_events_df, so an updadte to ede_coi in ede_cols_of_interest 
                #   and/or column drop will be needed
                if ede_coi in end_events_df.columns:
                    # Both f'{ede_coi}_gpd_for_sql' and ede_coi were found in end_events_df!
                    # One must be kept (by inserting into grp_by_cols_final), and one dropped (by inserting into cols_to_drop)
                    if trust_sql_grouping:
                        # Keep/update f'{ede_coi}_gpd_for_sql', and drop ede_coi
                        ede_cols_of_interest_updates[ede_coi] = found_cols_w_gpd_for_sql_appendix_inv[ede_coi]
                        cols_to_drop.append(ede_coi)
                    else:
                        # Keep ede_coi, drop f'{ede_coi}_gpd_for_sql'
                        ede_cols_of_interest_updates[ede_coi] = ede_coi
                        cols_to_drop.append(found_cols_w_gpd_for_sql_appendix_inv[ede_coi])
                else:
                    # Only f'{ede_coi}_gpd_for_sql' found in end_events_df, not grp_by_col, so no need to drop anything
                    ede_cols_of_interest_updates[ede_coi] = found_cols_w_gpd_for_sql_appendix_inv[ede_coi]
            else:
                ede_cols_of_interest_updates[ede_coi] = ede_coi
        #-----
        # Make sure no repeats in cols_to_drop
        cols_to_drop = list(set(cols_to_drop))
        #--------------------------------------------------
        # Build ede_gpd_coi_dict_updates, which has the same structure as ede_gpd_coi_dict (see above)
        ede_gpd_coi_dict_updates = dict()
        for ede_coi_grp, coi in ede_gpd_coi_dict.items():
            assert(ede_coi_grp not in ede_gpd_coi_dict_updates.keys())
            assert(Utilities.is_object_one_of_types(coi, [str,list,tuple]))
            if isinstance(coi, str):
                assert(coi in ede_cols_of_interest_updates.keys())
                ede_gpd_coi_dict_updates[ede_coi_grp] = ede_cols_of_interest_updates[coi]
            else:
                ede_gpd_coi_dict_updates[ede_coi_grp] = []
                for coi_i in coi:
                    assert(coi_i in ede_cols_of_interest_updates.keys())
                    ede_gpd_coi_dict_updates[ede_coi_grp].append(ede_cols_of_interest_updates[coi_i])
        assert(len(set(ede_gpd_coi_dict.keys()).symmetric_difference(set(ede_gpd_coi_dict_updates.keys())))==0)
        #-----
        # Sanity check
        assert(len(set(ede_gpd_coi_dict.keys()).symmetric_difference(set(ede_gpd_coi_dict_updates.keys())))==0)
        for ede_coi_grp, coi in ede_gpd_coi_dict.items():
            if Utilities.is_object_one_of_types(coi, [list,tuple]):
                assert(len(coi)==len(ede_gpd_coi_dict_updates[ede_coi_grp]))
        #--------------------------------------------------
        rename_dict = {}
        if drop_gpd_for_sql_appendix:
            rename_dict = CPXDfBuilder.get_rename_dict_for_gpd_for_sql_cols(df = end_events_df)
            # Don't need to rename any columns which will be dropped!
            rename_dict = {k:v for k,v in rename_dict.items() if k not in cols_to_drop}
            if len(rename_dict)>0:
                # Update ede_gpd_coi_dict_updates:
                ede_gpd_coi_dict_updates_fnl = {}
                for param_grp, cols_of_interest in ede_gpd_coi_dict_updates.items():
                    assert(param_grp not in ede_gpd_coi_dict_updates_fnl.keys())
                    assert(Utilities.is_object_one_of_types(cols_of_interest, [str,list,tuple]))
                    if isinstance(cols_of_interest, str):
                        ede_gpd_coi_dict_updates_fnl[param_grp] = rename_dict.get(cols_of_interest, cols_of_interest)
                    else:
                        cols_of_interest_new = []
                        for coi in cols_of_interest:
                            cols_of_interest_new.append(rename_dict.get(coi, coi))
                        ede_gpd_coi_dict_updates_fnl[param_grp] = cols_of_interest_new
                #-------------------------
                # Sanity checks
                #-----
                # Make sure all of rename_dict keys found in columns!
                assert(set(rename_dict.keys()).difference(set(end_events_df.columns))==set())
                #-----
                # Don't want any of the renames to already be included in the pd.DataFrame!
                assert(set(end_events_df.columns.drop(cols_to_drop)).intersection(set(rename_dict.values()))==set())
                #-----
                # Make sure all ede_cols_of_interest are found
                test_df = end_events_df.head().drop(columns=cols_to_drop).rename(columns=rename_dict)
                for param_grp, cols_of_interest in ede_gpd_coi_dict_updates_fnl.items():
                    assert(Utilities.is_object_one_of_types(cols_of_interest, [str,list,tuple]))
                    if isinstance(cols_of_interest, str):
                        assert(cols_of_interest in test_df.columns.tolist())
                    else:
                        for coi in cols_of_interest:
                            assert(coi in test_df.columns.tolist())
                #-------------------------
                ede_gpd_coi_dict_updates = ede_gpd_coi_dict_updates_fnl
            else:
                rename_dict = {}
        #--------------------------------------------------
        return ede_gpd_coi_dict_updates, cols_to_drop, rename_dict


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def perform_build_rcpx_from_end_events_df_prereqs(
        end_events_df                 ,
        mp_df                         = None, 
        ede_mp_mismatch_threshold_pct = 1.0, 
        grp_by_cols                   = 'outg_rec_nb', 
        outg_rec_nb_col               = 'outg_rec_nb',
        trsf_pole_nb_col              = 'trsf_pole_nb', 
        prem_nb_col                   = 'aep_premise_nb', 
        serial_number_col             = 'serialnumber',
        mp_df_cols                    = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        trust_sql_grouping            = True, 
        drop_gpd_for_sql_appendix     = True, 
        make_all_columns_lowercase    = True, 
    ):
        r"""
        Prepares for running build_rcpx_from_end_events_df.
        
        1. Adjust grp_by_cols and outg_rec_nb_col to handle cases where _gpd_for_sql appendix was added during data acquisition.
           If there is an instance where, e.g., outg_rec_nb_col and f'{outg_rec_nb_col}_gpd_for_sql' are both present, it settles
             the discrepancy (according to the trust_sql_grouping parameter) and compiles a list of columns which will need 
             to be dropped.
        2. Determine merge_on_ede and merge_on_mp columns (done within CPXDfBuilder.check_end_events_from_CSVs_merge_with_mp).
        3. Checks that the user supplied mp_df aligns well with the end events data.
        4. If assert_all_cols_equal==True, enforce the assertion.
    
        NOTE: make_all_columns_lowercase because...
          EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
          Athena maintains the original case, and does not conver to lower case,
            so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]  
        #--------------------------------------------------
        ede_cols_of_interest_updates, cols_to_drop, rename_cols_dict = CPXDfBuilder.identify_ede_cols_of_interest_to_update_andor_drop(
            end_events_df              = end_events_df,  
            grp_by_cols                = grp_by_cols, 
            outg_rec_nb_col            = outg_rec_nb_col,
            trsf_pole_nb_col           = trsf_pole_nb_col, 
            prem_nb_col                = prem_nb_col, 
            serial_number_col          = serial_number_col,
            trust_sql_grouping         = trust_sql_grouping, 
            drop_gpd_for_sql_appendix  = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        grp_by_cols       = ede_cols_of_interest_updates['grp_by_cols']
        outg_rec_nb_col   = ede_cols_of_interest_updates.get('outg_rec_nb_col', None)  # Not always needed, hence the get call
        trsf_pole_nb_col  = ede_cols_of_interest_updates.get('trsf_pole_nb_col', None) # Not always needed, hence the get call
        prem_nb_col       = ede_cols_of_interest_updates['prem_nb_col']
        serial_number_col = ede_cols_of_interest_updates['serial_number_col']
        #--------------------------------------------------
        #--------------------------------------------------
        if mp_df is not None:
            mp_df, merge_on_ede, merge_on_mp = CPXDfBuilder.check_end_events_df_merge_with_mp(
                end_events_df              = end_events_df,
                mp_df                      = mp_df, 
                threshold_pct              = ede_mp_mismatch_threshold_pct, 
                outg_rec_nb_col            = outg_rec_nb_col,
                trsf_pole_nb_col           = trsf_pole_nb_col, 
                prem_nb_col                = prem_nb_col, 
                serial_number_col          = serial_number_col,
                mp_df_cols                 = mp_df_cols, 
                cols_to_drop               = cols_to_drop, 
                rename_cols_dict           = rename_cols_dict, 
                make_all_columns_lowercase = make_all_columns_lowercase
            )
        else:
            merge_on_ede, merge_on_mp = None, None
        #--------------------------------------------------
        #--------------------------------------------------
        return_dict = dict(
            grp_by_cols       = grp_by_cols, 
            outg_rec_nb_col   = outg_rec_nb_col, 
            trsf_pole_nb_col  = trsf_pole_nb_col, 
            prem_nb_col       = prem_nb_col, 
            serial_number_col = serial_number_col,
            cols_to_drop      = cols_to_drop, 
            rename_cols_dict  = rename_cols_dict, 
            mp_df             = mp_df, 
            merge_on_ede      = merge_on_ede, 
            merge_on_mp       = merge_on_mp
        )
        return return_dict
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def perform_build_rcpx_from_end_events_in_csvs_prereqs(
        files_dir, 
        file_path_glob, 
        file_path_regex,
        mp_df                         = None, 
        ede_mp_mismatch_threshold_pct = 1.0, 
        grp_by_cols                   = 'outg_rec_nb', 
        outg_rec_nb_col               = 'outg_rec_nb',
        trsf_pole_nb_col              = 'trsf_pole_nb', 
        prem_nb_col                   = 'aep_premise_nb', 
        serial_number_col             = 'serialnumber',
        mp_df_cols                    = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        assert_all_cols_equal         = True, 
        trust_sql_grouping            = True, 
        drop_gpd_for_sql_appendix     = True, 
        make_all_columns_lowercase    = True, 
    ):
        r"""
        Prepares for running build_rcpx_df_from_end_events_in_csvs.
        Many of these items were done for each iteration of the main for loop in build_rcpx_df_from_EndEvents_in_csvs, which
          was really unnecessary.  
          This will improve performance (although, likely not much as none of the operations are super heavy) and simplify the code.
        
        1. Adjust grp_by_cols and outg_rec_nb_col to handle cases where _gpd_for_sql appendix was added during data acquisition.
           If there is an instance where, e.g., outg_rec_nb_col and f'{outg_rec_nb_col}_gpd_for_sql' are both present, it settles
             the discrepancy (according to the trust_sql_grouping parameter) and compiles a list of columns which will need 
             to be dropped.
        2. Determine merge_on_ede and merge_on_mp columns (done within CPXDfBuilder.check_end_events_from_CSVs_merge_with_mp).
        3. Checks that the user supplied mp_df aligns well with the end events data.
        4. If assert_all_cols_equal==True, enforce the assertion.

        NOTE: make_all_columns_lowercase because...
          EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
          Athena maintains the original case, and does not conver to lower case,
            so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]    
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir      = files_dir, 
            glob_pattern  = file_path_glob, 
            regex_pattern = file_path_regex
        )
        if len(paths)==0:
            print(f'No paths found in files_dir = {files_dir}')
            return None
        paths=natsorted(paths)
        #-------------------------
        #--------------------------------------------------
        # Grab first row from each CSV file to be used to check columns
        dfs = []
        for path in paths:
            df_i = GenAn.read_df_from_csv(
                read_path                      = path, 
                cols_and_types_to_convert_dict = None, 
                to_numeric_errors              = 'coerce', 
                drop_na_rows_when_exception    = True, 
                drop_unnamed0_col              = True, 
                pd_read_csv_kwargs             = dict(nrows=1)
            )
            if df_i.shape[0]==0:
                continue
            #-----
            dfs.append(df_i)
        #-------------------------
        if len(dfs)==0:
            print(f'No DFs found in files_dir={files_dir}')
            assert(0)
        #-------------------------                
        df_cols = Utilities_df.get_shared_columns(dfs, maintain_df0_order=True)
        for i in range(len(dfs)):
            if assert_all_cols_equal:
                # In order to account for case where columns are the same but in different order
                # one must compare the length of dfs[i].columns to that of df_cols (found by utilizing
                # the Utilities_df.get_shared_columns(dfs) functionality)
                assert(dfs[i].shape[1]==len(df_cols))
            dfs[i] = dfs[i][df_cols]
        end_events_df = pd.concat(dfs)
        #--------------------------------------------------
        #--------------------------------------------------
        ede_cols_of_interest_updates, cols_to_drop, rename_cols_dict = CPXDfBuilder.identify_ede_cols_of_interest_to_update_andor_drop(
            end_events_df              = end_events_df,  
            grp_by_cols                = grp_by_cols, 
            outg_rec_nb_col            = outg_rec_nb_col,
            trsf_pole_nb_col           = trsf_pole_nb_col, 
            prem_nb_col                = prem_nb_col, 
            serial_number_col          = serial_number_col,
            trust_sql_grouping         = trust_sql_grouping, 
            drop_gpd_for_sql_appendix  = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        grp_by_cols       = ede_cols_of_interest_updates['grp_by_cols']
        outg_rec_nb_col   = ede_cols_of_interest_updates['outg_rec_nb_col']
        trsf_pole_nb_col  = ede_cols_of_interest_updates['trsf_pole_nb_col']
        prem_nb_col       = ede_cols_of_interest_updates['prem_nb_col']
        serial_number_col = ede_cols_of_interest_updates['serial_number_col']
        #--------------------------------------------------
        #--------------------------------------------------
        if mp_df is not None:
            mp_df, merge_on_ede, merge_on_mp = CPXDfBuilder.check_end_events_from_CSVs_merge_with_mp(
                ede_file_paths             = paths,
                mp_df                      = mp_df, 
                threshold_pct              = ede_mp_mismatch_threshold_pct, 
                outg_rec_nb_col            = outg_rec_nb_col,
                trsf_pole_nb_col           = trsf_pole_nb_col, 
                prem_nb_col                = prem_nb_col, 
                serial_number_col          = serial_number_col,
                mp_df_cols                 = mp_df_cols, 
                assert_all_cols_equal      = assert_all_cols_equal, 
                cols_to_drop               = cols_to_drop, 
                rename_cols_dict           = rename_cols_dict, 
                make_all_columns_lowercase = make_all_columns_lowercase
            )
        else:
            merge_on_ede, merge_on_mp = None, None
        #--------------------------------------------------
        #--------------------------------------------------
        return_dict = dict(
            paths             = paths, 
            grp_by_cols       = grp_by_cols, 
            outg_rec_nb_col   = outg_rec_nb_col, 
            trsf_pole_nb_col  = trsf_pole_nb_col, 
            prem_nb_col       = prem_nb_col, 
            serial_number_col = serial_number_col,
            cols_to_drop      = cols_to_drop, 
            rename_cols_dict  = rename_cols_dict, 
            mp_df             = mp_df, 
            merge_on_ede      = merge_on_ede, 
            merge_on_mp       = merge_on_mp
        )
        return return_dict
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def set_faulty_mp_vals_to_nan_in_end_events_df(
        end_events_df, 
        prem_nb_col_ede='aep_premise_nb', 
        prem_nb_col_mp='prem_nb'
    ):
        r"""
        NOT REALLY NEEDED ANYMORE!
            As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
            If this was not true, then this function may actually be needed
            -----
            CPXDfBuilder.perform_build_rcpx_from_end_events_in_csvs_prereqs all but ensures a merge with MeterPremise will include, at minimum,
              premise number and serial number.
            Therefore, this is rarely needed.
        -------------------------

        Initially assumed that joining meter_events.end_device_event with default.meter_premise soley on serial number was fine.
        However, have since found that the serial number in meter_events.end_device_event does not always appear to be correct.
        This leads to a situation where the wrong MP is matched, which throws off the trsf_pole_nb (and any other attributes
        taken from default.meter_premise).
        In the future, the two will be merged on both serial number and premise number.  The join will be left, so any entries
        which don't match simply will have NaN for the MP cols.
        The purpose of this function is to reproduce this effect for the erroneous data.
        """
        #-------------------------
        assert(prem_nb_col_ede in end_events_df.columns)
        assert(prem_nb_col_mp  in end_events_df.columns)
        #-------------------------
        # NOTE: Should have been left joined.  Therefore, it is possible and acceptable for prem_nb_col_mp to be NaN when 
        #       prem_nb_col_ede is not.  Thus, the need for '& (end_events_df[prem_nb_col_mp].notna())'
        faulty_mask = (end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp]) & (end_events_df[prem_nb_col_mp].notna())
        if faulty_mask.sum()==0:
            return end_events_df
        #-------------------------
        # Find the columns from default.meter_premise which were merged into end_events_df
        mp_merged_cols = [x for x in TableInfos.MeterPremise_TI.columns_full if x in end_events_df.columns.tolist()]
        # Maintain original order
        mp_merged_cols = [x for x in end_events_df.columns.tolist() if x in mp_merged_cols]
        #-------------------------
        # Set the mp_merged_cols to NaN for faulty_mask.
        end_events_df.loc[faulty_mask,mp_merged_cols]=np.nan
        #-------------------------
        # Make sure the procedure worked
        assert(((end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp]) & (end_events_df[prem_nb_col_mp].notna())).sum()==0)
        #-------------------------
        return end_events_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def correct_or_add_active_mp_cols(
        end_events_df         , 
        df_time_col_0         ,
        df_time_col_1         = None,
        df_mp_curr            = None, 
        df_mp_hist            = None, 
        df_and_mp_merge_pairs = [
            ['serialnumber',   'mfr_devc_ser_nbr'], 
            ['aep_premise_nb', 'prem_nb']
        ], 
        assert_all_PNs_found  = True, 
        df_prem_nb_col        = 'aep_premise_nb', 
        df_mp_prem_nb_col     = 'prem_nb'
    ):
        r"""
        NOT REALLY NEEDED ANYMORE!
            As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
            If this was not true, then this function may actually be needed
            -----
            CPXDfBuilder.perform_build_rcpx_from_end_events_in_csvs_prereqs all but ensures a merge with MeterPremise will include, at minimum,
              premise number and serial number.
            Therefore, this is rarely needed.
        -------------------------

        Built to correct the MP entries in end_events_df.
        When building EndEvents data for outages, the data we joined with MeterPremise at the SQL leve.
          However, they really need to be joined with whatever meters were active at the time of the event.
          This function replaces the faulty with good

        Any columns from MP (meaning, from default.meter_premise or default.meter_premise_hist) are first identified, as
          these will need to be replaced.
        i.  In the case where df_mp_curr/hist are supplied, whatever columns are contained in those DFs will be merged with 
              end_events_df (in which case, it is necessary for any found MP cols to be dropped)
        ii. In the case where df_mp_curr/hist are not supplied, they will be built and the MP columns found in end_events_df
              will be included as addtnl_mp_df_curr/hist_cols arguments.
            If no MP cols are found in end_events_df, then addtnl_mp_df_curr/hist_cols arguments will be empty, so the MP 
              columns to be added will be the bare minimum (i.e., 'mfr_devc_ser_nbr', 'prem_nb', 'trsf_pole_nb')
        """
        #-------------------------
        # Find the columns from default.meter_premise and default.meter_premise_hist which were merged into end_events_df
        #   as these will need to be replaced
        mp_curr_merged_cols = [x for x in TableInfos.MeterPremise_TI.columns_full if x in end_events_df.columns.tolist()]
        mp_hist_merged_cols = [x for x in TableInfos.MeterPremiseHist_TI.columns_full if x in end_events_df.columns.tolist()]
        # Maintain original order
        mp_curr_merged_cols = [x for x in end_events_df.columns.tolist() if x in mp_curr_merged_cols]
        mp_hist_merged_cols = [x for x in end_events_df.columns.tolist() if x in mp_hist_merged_cols]
        #-------------------------
        # Drop the MP columns from end_events_df (only truly necessary when df_mp_curr/hist are supplied, as when they are
        #   not, they will be built with the necessary columns which will then replace those in end_events_df.  However,
        #   for either case, dropping them at this point is safe)
        end_events_df = end_events_df.drop(columns=mp_curr_merged_cols+mp_hist_merged_cols)
        #-------------------------
        # Merge end_events_df with the active MP entries
        end_events_df = MeterPremise.merge_df_with_active_mp(
            df                              = end_events_df, 
            df_time_col_0                   = df_time_col_0, 
            df_time_col_1                   = df_time_col_1, 
            df_mp_curr                      = df_mp_curr, 
            df_mp_hist                      = df_mp_hist, 
            df_and_mp_merge_pairs           = df_and_mp_merge_pairs, 
            keep_overlap                    = 'right', 
            drop_inst_rmvl_cols_after_merge = True, 
            addtnl_mp_df_curr_cols          = mp_curr_merged_cols, 
            addtnl_mp_df_hist_cols          = mp_hist_merged_cols, 
            assume_one_xfmr_per_PN          = True, 
            assert_all_PNs_found            = assert_all_PNs_found, 
            df_prem_nb_col                  = df_prem_nb_col, 
            df_mp_serial_number_col         = 'mfr_devc_ser_nbr', 
            df_mp_prem_nb_col               = df_mp_prem_nb_col, 
            df_mp_install_time_col          = 'inst_ts', 
            df_mp_removal_time_col          = 'rmvl_ts', 
            df_mp_trsf_pole_nb_col          = 'trsf_pole_nb'        
        )
        #-------------------------
        return end_events_df


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def correct_faulty_mp_vals_in_end_events_df(
        end_events_df         , 
        df_time_col_0         ,
        df_time_col_1         = None,
        df_mp_curr            = None, 
        df_mp_hist            = None, 
        prem_nb_col_ede       = 'aep_premise_nb', 
        prem_nb_col_mp        = 'prem_nb', 
        df_and_mp_merge_pairs = [
            ['serialnumber',   'mfr_devc_ser_nbr'], 
            ['aep_premise_nb', 'prem_nb']
        ], 
        assert_all_PNs_found  = True    
    ):
        r"""
        NOT REALLY NEEDED ANYMORE!
            As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
            If this was not true, then this function may actually be needed
            -----
            CPXDfBuilder.perform_build_rcpx_from_end_events_in_csvs_prereqs all but ensures a merge with MeterPremise will include, at minimum,
              premise number and serial number.
            Therefore, this is rarely needed.
        -------------------------

        Find faulty Meter Premise values in end_events_df and correct them.  See note below regarding initial assumption for why
          these values are faulty.
        The entries are determined to be faulty if the premise number from the meter_events.end_device_event database (contained
          in prem_nb_col_ede) does not match that from default.meter_premise (contained in prem_nb_col_mp).
        The faulty entries are corrected using the full power of default.meter_premise and default.meter_premise_hist combined,
          allowing the correct meter at the time of the event to be located.

        This is basically a blending of CPXDfBuilder.set_faulty_mp_vals_to_nan_in_end_events_df and 
        CPXDfBuilder.correct_or_add_active_mp_cols

        Initially assumed that joining meter_events.end_device_event with default.meter_premise soley on serial number was fine.
        However, have since found that the serial number in meter_events.end_device_event does not always appear to be correct.
        This leads to a situation where the wrong MP is matched, which throws off the trsf_pole_nb (and any other attributed
        taken from default.meter_premise).
        In the future, the two will be merged on both serial number and premise number.  The join will be left, so any entries
        which don't match simply will have NaN for the MP cols.
        The purpose of this function is to reproduce this effect for the erroneous data.
        """
        #-------------------------
        # First, identify the faulty entries and grab the subset
        #-----
        # NOTE: Also want to fix entries where prem_nb is NaN, hence why '& (end_events_df[prem_nb_col_mp].notna())' is absent here
        #       as compared to faulty_mask in CPXDfBuilder.set_faulty_mp_vals_to_nan_in_end_events_df
        faulty_mask = (end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp])
        if faulty_mask.sum()==0:
            return end_events_df
        flty_end_events_df = end_events_df[faulty_mask]
        #-------------------------
        # Reduce end_events_df to only non-faulty entries
        end_events_df = end_events_df[~faulty_mask]
        # Should no longer be any faulty idxs in end_events_df
        assert((end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp]).sum()==0)
        #-------------------------
        # Correct the faulty entries
        flty_end_events_df = CPXDfBuilder.correct_or_add_active_mp_cols(
            end_events_df         = flty_end_events_df, 
            df_time_col_0         = df_time_col_0,
            df_time_col_1         = df_time_col_1,
            df_mp_curr            = df_mp_curr, 
            df_mp_hist            = df_mp_hist, 
            df_and_mp_merge_pairs = df_and_mp_merge_pairs, 
            assert_all_PNs_found  = assert_all_PNs_found, 
            df_prem_nb_col        = prem_nb_col_ede, 
            df_mp_prem_nb_col     = prem_nb_col_mp
        )
        #-------------------------
        # Now, add the corrected flty_end_events_df back into end_events_df
        #-----
        # I suppose it's fine for flty_end_events_df to have additional columns, but it MUST contain
        #   at least those in end_events_df
        assert(len(set(end_events_df.columns).difference(set(flty_end_events_df.columns)))==0)
        flty_end_events_df = flty_end_events_df[end_events_df.columns]
        #-----
        end_events_df = pd.concat([end_events_df, flty_end_events_df])
        #-------------------------
        return end_events_df
    
    
    #-----------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------      
    #--------------------------------------------------
    @staticmethod
    def build_rcpx_from_end_events_df_helper(
        end_events_df_i    , 
        group_cols         = 'outg_rec_nb', 
        SN_col             = 'serialnumber', 
        reason_col         = 'reason', 
        inclue_zero_counts = False,
        possible_reasons   = None,
        include_nSNs       = True, 
        PN_col             = 'aep_premise_nb', 
        include_nPNs       = False,
    ):
        r"""
        Helper function for build_rcpx_from_EndEvents_df.
        Will return a pd.DataFrame object with columns=['counts'] and index equal to the contained end event reasons
    
        inclue_zero_counts/possible_reasons:
            If inclue_zero_counts is True, possible_reasons MUST be supplied also
            ==> any possible_reasons not found in end_events_df_i will be included in output with couunts = 0
    
        include_nSNs:
            If True, index=_nSNs_wEvs will be included in output with counts of number of serial numbers registering an event
    
        include_nSNs/PN_col:
            If include_nSNs==True, PN_col must be supplied
            If True, index=_nPNs_wEvs will be included in output with counts of number of premiess registering an event
        
        !!!!! If used by itself !!!!!
            end_events_df_i:
                MUST be a pd.DataFrame for a single group (meaning, each column in group_cols should have one unique value)
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(group_cols, [str, list, tuple]))
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        # NOTE: Below <=1 instead of ==1 because can equal 0 when gpby_dropna==False in get_reason_counts_per_group
        assert(all(end_events_df_i[group_cols].nunique()<=1))
        #-------------------------
        counts_i      = end_events_df_i.groupby([SN_col, reason_col]).size().groupby(reason_col).sum()
        counts_i.name = 'counts'
        #-------------------------
        if inclue_zero_counts:
            assert(possible_reasons is not None)
            for possible_reason in possible_reasons:
                if possible_reason not in counts_i:
                    counts_i[possible_reason]=0
        #-------------------------
        if include_nSNs:
            nSNs                   = end_events_df_i[SN_col].nunique()
            counts_i['_nSNs_wEvs'] = nSNs
        if include_nPNs:
            nPNs                   = end_events_df_i[PN_col].nunique()
            counts_i['_nPNs_wEvs'] = nPNs
        #-------------------------
        counts_i = counts_i.sort_index()
        counts_i = counts_i.to_frame()
        #-------------------------
        return counts_i
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_end_events_df(
        end_events_df                 , 
        prediction_date               , 
        td_min                        , 
        td_max                        , 
        group_cols                    = ['outg_rec_nb', 'trsf_pole_nb'], 
        freq                          = '5D', 
        normalize_by_time_interval    = True, 
        build_ede_typeid_to_reason_df = True, 
        inclue_zero_counts            = False, 
        possible_reasons              = None, 
        include_nSNs                  = True, 
        include_nPNs                  = False, 
        reason_col                    = 'reason', 
        valuesinterval_col            = 'valuesinterval', 
        edetypeid_col                 = 'enddeviceeventtypeid', 
        SN_col                        = 'serialnumber', 
        PN_col                        = 'aep_premise_nb', 
        outg_rec_nb_col               = None,  # only needed when mp_df is included!
        trsf_pole_nb_col              = None , # only needed when mp_df is included!
        total_counts_col              = 'total_counts', 
        addtnl_dropna_subset_cols     = None, 
        set_faulty_mp_vals_to_nan     = False, 
        correct_faulty_mp_vals        = False, 
        XNs_tags                      = None, 
        trust_sql_grouping            = True, 
        drop_gpd_for_sql_appendix     = True, 
        mp_df_cols                    = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        make_all_columns_lowercase    = True, 
        date_only                     = False, 
    ):
        r"""
        end_events_df should be a pd.DataFrame with data from multiple outages
    
        If possible_reasons is None, they will be inferred from end_events_df
        """
        #----------------------------------------------------------------------------------------------------
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(group_cols, [str, list, tuple]))
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        #-------------------------
        if inclue_zero_counts and possible_reasons is None:
            possible_reasons = end_events_df[reason_col].unique().tolist()
    
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min          = pd.Timedelta(td_min)
        td_max          = pd.Timedelta(td_max)
        prediction_date = pd.to_datetime(prediction_date)
        #-------------------------
        Utilities_dt.assert_timedelta_is_days(td_min)
        Utilities_dt.assert_timedelta_is_days(td_max)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        if freq is not None:
            freq   = pd.Timedelta(freq)
            Utilities_dt.assert_timedelta_is_days(freq)
    
        #--------------------------------------------------
        prereq_dict = CPXDfBuilder.perform_build_rcpx_from_end_events_df_prereqs(
            end_events_df                 = end_events_df, 
            mp_df                         = None, 
            ede_mp_mismatch_threshold_pct = 1.0, 
            grp_by_cols                   = group_cols, 
            outg_rec_nb_col               = outg_rec_nb_col,
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            prem_nb_col                   = PN_col, 
            serial_number_col             = SN_col,
            mp_df_cols                    = mp_df_cols,  # Only needed if mp_df is not None
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase    = make_all_columns_lowercase
        )
        group_cols        = prereq_dict['grp_by_cols']
        outg_rec_nb_col   = prereq_dict['outg_rec_nb_col']
        trsf_pole_nb_col  = prereq_dict['trsf_pole_nb_col']
        PN_col            = prereq_dict['prem_nb_col']
        SN_col            = prereq_dict['serial_number_col']
        cols_to_drop      = prereq_dict['cols_to_drop']
        rename_cols_dict  = prereq_dict['rename_cols_dict']
        mp_df             = prereq_dict['mp_df']
        merge_on_ede      = prereq_dict['merge_on_ede']
        merge_on_mp       = prereq_dict['merge_on_mp']
        #--------------------------------------------------
        #-------------------------
        if make_all_columns_lowercase:
            valuesinterval_col = valuesinterval_col.lower()
            total_counts_col   = total_counts_col.lower()
    
        #--------------------------------------------------
        end_events_df = AMIEndEvents.perform_std_initiation_and_cleaning(
            df             = end_events_df, 
            drop_na_values = False, 
            inplace        = True
        )
        #-------------------------
        end_events_df = CPXDfBuilder.perform_std_col_renames_and_drops(
            end_events_df              = end_events_df, 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
    
        #--------------------------------------------------
        # FROM WHAT I CAN TELL, the meters which are 'missing' from meter premise but are present in end events
        #   are caused by meters which were removed or installed close to (a few days/weeks before) an outage.
        # Therefore, although the meter may not have been present at the time of the outage (and therefore was exluced from
        #   meter premise), it could have registered events leading up to/following the outage.
        # e.g., if a meter was removed in the days before an outage, end events are still found for this meter in the days leading up to the outage
        # e.g., if a meter was installed in the days after an outage, end events are still found for this meter in the days following an outage.
        # How should these be handled?
        # The simplest method, which I will implement for now, is to simply ONLY consider those meters which were present
        #   at the time of the outage.  THEREFORE, the two DFs should be joined with an inner merge!
        if mp_df is not None:
            end_events_df = AMIEndEvents.merge_end_events_df_with_mp(
                end_events_df      = end_events_df, 
                df_mp              = mp_df, 
                merge_on_ede       = merge_on_ede, 
                merge_on_mp        = merge_on_mp, 
                cols_to_include_mp = None, 
                drop_cols          = None, 
                rename_cols        = None, 
                how                = 'inner', 
                inplace            = True
            )
    
        #--------------------------------------------------
        for grp_by_col in group_cols:
            assert(grp_by_col in end_events_df.columns)
        #-------------------------
        dropna_subset_cols = group_cols
        if addtnl_dropna_subset_cols is not None:
            dropna_subset_cols.extend(addtnl_dropna_subset_cols)
        end_events_df = end_events_df.dropna(subset=dropna_subset_cols)
        #-------------------------
        if set_faulty_mp_vals_to_nan and mp_df is not None:
            end_events_df = CPXDfBuilder.set_faulty_mp_vals_to_nan_in_end_events_df(
                end_events_df   = end_events_df, 
                prem_nb_col_ede = PN_col, 
                prem_nb_col_mp  = mp_df_cols['prem_nb_col']
            )
        #-------------------------
        end_events_df = Utilities_dt.strip_tz_info_and_convert_to_dt(
            df            = end_events_df, 
            time_col      = valuesinterval_col, 
            placement_col = f'{valuesinterval_col}_local', 
            run_quick     = True, 
            n_strip       = 6, 
            inplace       = False
        )
        valuesinterval_col = f'{valuesinterval_col}_local'
    
        #--------------------------------------------------
        if correct_faulty_mp_vals and mp_df is not None:
            end_events_df = CPXDfBuilder.correct_faulty_mp_vals_in_end_events_df(
                end_events_df         = end_events_df, 
                df_time_col_0         = valuesinterval_col,
                df_time_col_1         = None,
                df_mp_curr            = None, 
                df_mp_hist            = None, 
                prem_nb_col_ede       = PN_col, 
                prem_nb_col_mp        = mp_df_cols['prem_nb_col'], 
                df_and_mp_merge_pairs = [
                    [SN_col , mp_df_cols['serial_number_col']], 
                    [PN_col , mp_df_cols['prem_nb_col']]
                ], 
                assert_all_PNs_found  = False
            )
        
        #--------------------------------------------------
        end_events_df = AMIEndEvents.reduce_end_event_reasons_in_df(
            df                            = end_events_df, 
            reason_col                    = reason_col, 
            edetypeid_col                 = edetypeid_col, 
            patterns_to_replace_by_typeid = None, 
            addtnl_patterns_to_replace    = None, 
            placement_col                 = None, 
            count                         = 0, 
            flags                         = re.IGNORECASE,  
            inplace                       = True
        )
        
        #--------------------------------------------------
        if build_ede_typeid_to_reason_df:
            ede_typeid_to_reason_df = AMIEndEvents.build_ede_typeid_to_reason_df(
                end_events_df  = end_events_df, 
                reason_col     = reason_col, 
                ede_typeid_col = edetypeid_col
            )
    
        #--------------------------------------------------
        # Look for any columns ending in _GPD_FOR_SQL and if not included in grpyby cols, then print warning!
        found_gpd_for_sql_cols = CPXDfBuilder.find_gpd_for_sql_cols(
            df        = end_events_df, 
            col_level = -1
        )
        if len(found_gpd_for_sql_cols)>0:
            # make loweercase, if needed
            if make_all_columns_lowercase:
                found_gpd_for_sql_cols = [x.lower() for x in found_gpd_for_sql_cols]
            # change names, if needed
            found_gpd_for_sql_cols = [rename_cols_dict.get(x, x) for x in found_gpd_for_sql_cols]
            #-----
            not_included = list(set(found_gpd_for_sql_cols).difference(set(group_cols)))
            if len(not_included)>0:
                print('\n!!!!! WARNING !!!!!\nCPXDfBuilder.build_rcpx_from_evsSum_df\nFOUND POSSIBLE GROUPBY COLUMNS NOT INCLUDED IN group_cols argument')
                print(f"\tnot_included           = {not_included}\n\tfound_gpd_for_sql_cols = {found_gpd_for_sql_cols}\n\tgroup_cols             = {group_cols}")   
                print('!!!!!!!!!!!!!!!!!!!\n')
    
        #--------------------------------------------------
        nec_cols = group_cols + [valuesinterval_col, reason_col]
        if include_nSNs:
            nec_cols.append(SN_col)
        if include_nPNs:
            nec_cols.append(PN_col)
        assert(set(nec_cols).difference(set(end_events_df.columns.tolist()))==set()) 
        #-------------------------
        cols_to_drop = set(end_events_df.columns.tolist()).difference(set(nec_cols))
        cols_to_drop = list(cols_to_drop)
        
        #-------------------------
        # Make sure valuesinterval_col is datetime object
        end_events_df[valuesinterval_col] = pd.to_datetime(end_events_df[valuesinterval_col])
        if date_only:
            end_events_df[valuesinterval_col] = pd.to_datetime(end_events_df[valuesinterval_col].dt.strftime('%Y-%m-%d'))
        
        #-------------------------
        # No need in wasting time grouping data we won't use
        # So, reduce evsSum_df to only the dates we're interested in 
        # Note: Open left (does not include endpoint), closed right (does include endpoint) below
        #       This seems opposite to what one would intuitively think, but one must remeber that the
        #         events are all prior to, and with respect to, some prediction_date.
        #       Therefore, e.g., for a '01-06 Days' bin, one typically wants [1 Day, 6 Days)
        #       In reality, [1 Day, 6 Days) is (-6 Days, -1 Day], hence the open left and closed right below.
        end_events_df = end_events_df[
            (end_events_df[valuesinterval_col] >  prediction_date-td_max) & 
            (end_events_df[valuesinterval_col] <= prediction_date-td_min)
        ]
    
        if freq is None:
            time_grps = [(prediction_date-td_max, prediction_date-td_min)]
            #--------------------------------------------------
            rcpx_df_long = end_events_df.drop(columns=cols_to_drop).groupby(group_cols)[nec_cols].apply(
                lambda x: CPXDfBuilder.build_rcpx_from_end_events_df_helper(
                    end_events_df_i    = x, 
                    group_cols         = group_cols, 
                    SN_col             = SN_col, 
                    reason_col         = reason_col, 
                    inclue_zero_counts = False,
                    possible_reasons   = None,
                    include_nSNs       = True, 
                    PN_col             = PN_col, 
                    include_nPNs       = False,
                )
            )
            #--------------------------------------------------
            str_len = np.max([len(str(x)) for x in [days_min, days_max]])
            str_fmt = f'{{:0{str_len}d}}'
            time_pds_rename = {time_grps[0][0] : "{}-{} Days".format(str_fmt.format(days_min), str_fmt.format(days_max))}
            #-------------------------
            # For functionality, need valuesinterval_col (this function was originally designed for use with non-None freq!)
            rcpx_df_long[valuesinterval_col] = time_grps[0][0]
            rcpx_df_long = rcpx_df_long.reset_index(drop=False).set_index(group_cols+[valuesinterval_col])
        else:
            #--------------------------------------------------
            # Need to set origin in pd.Grouper to ensure proper grouping
            freq = pd.Timedelta(freq)
            assert((td_max-td_min) % freq==pd.Timedelta(0))
            #-----
            time_grps = pd.date_range(
                start = prediction_date-td_max, 
                end   = prediction_date-td_min, 
                freq  = freq
            )
            assert(len(time_grps)>1)
            time_grps = [(time_grps[i], time_grps[i+1]) for i in range(len(time_grps)-1)]
            assert(len(time_grps) == (td_max-td_min)/pd.Timedelta(freq))
            #-------------------------
            group_freq = pd.Grouper(freq=freq, key=valuesinterval_col, origin=time_grps[0][0], closed='right')
            #--------------------------------------------------
            rcpx_df_long = end_events_df.drop(columns=cols_to_drop).groupby(group_cols+[group_freq])[nec_cols].apply(
                lambda x: CPXDfBuilder.build_rcpx_from_end_events_df_helper(
                    end_events_df_i    = x, 
                    group_cols         = group_cols, 
                    SN_col             = SN_col, 
                    reason_col         = reason_col, 
                    inclue_zero_counts = False,
                    possible_reasons   = None,
                    include_nSNs       = True, 
                    PN_col             = PN_col, 
                    include_nPNs       = False,
                )
            )
            #--------------------------------------------------
            time_pds_rename = CPXDfBuilder.get_time_pds_rename(
                curr_time_pds = time_grps, 
                td_min        = td_min, 
                td_max        = td_max, 
                freq          = freq
            )
        final_time_pds = list(time_pds_rename.values())
    
        #--------------------------------------------------
        rcpx_df = AMIEndEvents.convert_reason_counts_per_outage_df_long_to_wide(
            rcpo_df_long      = rcpx_df_long, 
            reason_col        = reason_col, 
            higher_order_cols = [valuesinterval_col], 
            fillna_w_0        = True, 
            drop_lvl_0_col    = True
        )
        
        #--------------------------------------------------
        rcpx_df = rcpx_df.rename(columns=time_pds_rename, level=0)
        rcpx_df = rcpx_df[rcpx_df.columns.sort_values()]
        
        #--------------------------------------------------
        # Overkill here (since all time windows are of length freq), but something similar will 
        #   be needed if I want to move to non-uniform period lengths
        # One could, e.g., simply divide by length of freq in days
        if normalize_by_time_interval:
            normalizations_dict = dict()
            for time_grp_i in time_grps:
                delta_i        = (time_grp_i[1] - time_grp_i[0]).days
                fnl_time_grp_i = time_pds_rename[time_grp_i[0]]
                assert(fnl_time_grp_i not in normalizations_dict)
                normalizations_dict[fnl_time_grp_i] = delta_i
            assert(set(normalizations_dict.keys()).symmetric_difference(set(rcpx_df.columns.get_level_values(0).unique()))==set())
            #-------------------------
            if XNs_tags is None:
                XNs_tags = CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols()
            #-------------------------
            for fnl_time_grp_i, window_width_days_i in normalizations_dict.items():
                cols_to_norm_i = Utilities.find_untagged_in_list(
                    lst  = rcpx_df[fnl_time_grp_i].columns.tolist(), 
                    tags = XNs_tags
                )
                cols_to_norm_i = [(fnl_time_grp_i, x) for x in cols_to_norm_i]
                #-------------------------
                rcpx_df[cols_to_norm_i] = rcpx_df[cols_to_norm_i]/window_width_days_i
        #--------------------------------------------------
        if build_ede_typeid_to_reason_df:
            return rcpx_df, ede_typeid_to_reason_df
        else:
            return rcpx_df
        

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_end_events_df_wpreddates(
        end_events_df                 , 
        pred_date_col                 , 
        td_min                        , 
        td_max                        , 
        group_cols                    = ['outg_rec_nb', 'trsf_pole_nb'], 
        freq                          = '5D', 
        normalize_by_time_interval    = True, 
        build_ede_typeid_to_reason_df = True, 
        inclue_zero_counts            = False, 
        possible_reasons              = None, 
        include_nSNs                  = True, 
        include_nPNs                  = False, 
        reason_col                    = 'reason', 
        valuesinterval_col            = 'valuesinterval', 
        edetypeid_col                 = 'enddeviceeventtypeid', 
        SN_col                        = 'serialnumber', 
        PN_col                        = 'aep_premise_nb', 
        outg_rec_nb_col               = None,  # only needed when mp_df is included!
        trsf_pole_nb_col              = None , # only needed when mp_df is included!
        total_counts_col              = 'total_counts', 
        addtnl_dropna_subset_cols     = None, 
        set_faulty_mp_vals_to_nan     = False, 
        correct_faulty_mp_vals        = False, 
        XNs_tags                      = None, 
        trust_sql_grouping            = True, 
        drop_gpd_for_sql_appendix     = True, 
        mp_df_cols                    = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        make_all_columns_lowercase    = True, 
        date_only                     = False, 
    ):
        r"""
        end_events_df should be a pd.DataFrame with data from multiple outages
    
        If possible_reasons is None, they will be inferred from end_events_df

        Modeled after build_rcpx_from_evsSum_df_wpreddates_simple.
            If this function does not work for some instance, one could try building something similar to build_rcpx_from_evsSum_df_wpreddates_full
              and combining the two as is done in build_rcpx_from_evsSum_df_wpreddates
        """
        #----------------------------------------------------------------------------------------------------
        #--------------------------------------------------
        # We will alter end_events_df.  To keep consistent outside of function, create copy
        end_events_df = end_events_df.copy()
        #--------------------------------------------------
        # Build relative date column, which will be used for grouping
        # rel_date_col will essentially replace valuesinterval_col in build_rcpx_from_end_events_df
        #-----
        rel_date_col = Utilities.generate_random_string(letters='letters_only')
        end_events_df[rel_date_col] = end_events_df[pred_date_col]-end_events_df[valuesinterval_col]
        #-----
        # Shift everything relative to a common dummy prediction date
        dummy_date     = pd.to_datetime('2 Nov. 1987')
        dummy_date_col = Utilities.generate_random_string(letters='letters_only')
        end_events_df[dummy_date_col] = dummy_date - end_events_df[rel_date_col]
        end_events_df = end_events_df.drop(columns=[valuesinterval_col, rel_date_col, pred_date_col])
    
        #--------------------------------------------------
        # Now that everything is relative to common date, simply use build_rcpx_from_end_events_df
        #   with prediction_date=dummy_date and valuesinterval_col=dummy_date_col
        rcpx_df = CPXDfBuilder.build_rcpx_from_end_events_df_wpreddates(
            end_events_df                 = end_events_df, 
            pred_date_col                 = dummy_date, 
            td_min                        = td_min, 
            td_max                        = td_max, 
            group_cols                    = group_cols, 
            freq                          = freq, 
            normalize_by_time_interval    = normalize_by_time_interval, 
            build_ede_typeid_to_reason_df = build_ede_typeid_to_reason_df, 
            inclue_zero_counts            = inclue_zero_counts, 
            possible_reasons              = possible_reasons, 
            include_nSNs                  = include_nSNs, 
            include_nPNs                  = include_nPNs, 
            reason_col                    = reason_col, 
            valuesinterval_col            = dummy_date_col, 
            edetypeid_col                 = edetypeid_col, 
            SN_col                        = SN_col, 
            PN_col                        = PN_col, 
            outg_rec_nb_col               = outg_rec_nb_col,  
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            total_counts_col              = total_counts_col, 
            addtnl_dropna_subset_cols     = addtnl_dropna_subset_cols, 
            set_faulty_mp_vals_to_nan     = set_faulty_mp_vals_to_nan, 
            correct_faulty_mp_vals        = correct_faulty_mp_vals, 
            XNs_tags                      = XNs_tags, 
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            mp_df_cols                    = mp_df_cols, 
            make_all_columns_lowercase    = make_all_columns_lowercase, 
            date_only                     = date_only, 
        )
    
        #--------------------------------------------------
        return rcpx_df
    

    #------------------------------------------------------------------------------------------------------------------------------------------
    # Methods for building from events summaries (from meter_events.events_summary_vw)
    #------------------------------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    @staticmethod    
    def build_regex_setup(
        conn_aws = None
    ):
        r"""
        Builds regex_setup_df and cr_trans_dict, where cr_trans_dict = curated reasons translation dictionary
        """
        #-------------------------
        if conn_aws is None:
            conn_aws = Utilities.get_athena_prod_aws_connection()
        #-------------------------
        sql = """
        SELECT * FROM meter_events.event_summ_regex_setup
        """
        #-------------------------
        # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
        # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regex_setup_df = pd.read_sql(sql, conn_aws, dtype=str)
        #-------------------------
        cr_trans_dict = {x[0]:x[1] for x in regex_setup_df[['pivot_id', 'regex_report_title']].values.tolist()}
        return regex_setup_df, cr_trans_dict

    #---------------------------------------------------------------------------
    @staticmethod    
    def get_regex_setup(
        conn_aws = None
    ):
        r"""
        Returns regex_setup_df and cr_trans_dict, where cr_trans_dict = curated reasons translation dictionary
        """
        #-------------------------
        return CPXDfBuilder.build_regex_setup(conn_aws = conn_aws)
    

    #---------------------------------------------------------------------------
    @staticmethod
    def get_cr_trans_dict(
        conn_aws = None, 
        to       = 'reason'
    ):
        r"""
        """
        #-------------------------
        assert(to in ['reason', 'typeid'])
        #-------------------------
        regex_setup_df, cr_trans_dict = CPXDfBuilder.get_regex_setup(conn_aws = conn_aws)
        # if to == 'reason', cr_trans_dict is fine as is.
        # Otherwise, need to build
        if to != 'reason':
            cr_trans_dict = {x[0]:x[1] for x in regex_setup_df[['pivot_id', 'enddeviceeventtypeid']].values.tolist()}
        #-------------------------
        return cr_trans_dict


    #---------------------------------------------------------------------------
    @staticmethod
    def find_cr_cols(
        df
    ):
        r"""
        """
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        return cr_cols
    
    #---------------------------------------------------------------------------
    @staticmethod
    def get_evsSum_df_std_dtypes_dict(
        df=None
    ):
        r"""
        If df not included, return only non_cr_cols
        If df included:
            Look for cr_cols in df, and return in addition to non_cr_cols
        """
        #-------------------------
        non_cr_cols = {
            'serialnumber'                   : str,
            'aep_premise_nb'                 : str,
            'trsf_pole_nb'                   : str,
            'xf_meter_cnt'                   : np.int64,
            'events_tot'                     : np.int64,
            'aep_opco'                       : str,
            'aep_event_dt'                   : datetime.datetime,
            #----------
            'trsf_pole_nb_GPD_FOR_SQL'       : str,
            'outg_rec_nb_GPD_FOR_SQL'        : str,
            'no_outg_rec_nb_GPD_FOR_SQL'     : str,
            'is_first_after_outg_GPD_FOR_SQL': bool,  
            #-----
            'trsf_pole_nb'                   : str,
            'outg_rec_nb'                    : str,
            'no_outg_rec_nb'                 : str,
            'is_first_after_outg'            : bool,  
        }
        #-------------------------
        if df is None:
            return non_cr_cols
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        cr_cols = {x:np.int64 for x in cr_cols}
        #-------------------------
        return non_cr_cols|cr_cols
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_evsSum_df_from_csvs(
        files_dir                      , 
        file_path_glob                 = r'events_summary_[0-9]*.csv', 
        file_path_regex                = None, 
        batch_size                     = 50, 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        make_all_columns_lowercase     = True, 
        assert_all_cols_equal          = True,  
        n_update                       = 1, 
        verbose                        = True
    ):
        r"""
        """
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir      = files_dir, 
            glob_pattern  = file_path_glob, 
            regex_pattern = file_path_regex
        )
        if len(paths)==0:
            print(f'No paths found in files_dir = {files_dir}')
            return None
        paths=natsorted(paths)
        #-------------------------
        batch_idxs = Utilities.get_batch_idx_pairs(len(paths), batch_size)
        n_batches = len(batch_idxs)    
        if verbose:
            print(f'n_paths    = {len(paths)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches  = {n_batches}')    
        #-------------------------
        evsSum_dfs = []
        for i, batch_i in enumerate(batch_idxs):
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-----
            evsSum_df_i = GenAn.read_df_from_csv_batch(
                paths                          = paths[i_beg:i_end], 
                cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
                to_numeric_errors              = to_numeric_errors, 
                make_all_columns_lowercase     = make_all_columns_lowercase, 
                assert_all_cols_equal          = assert_all_cols_equal
            )
            #-------------------------
            if evsSum_df_i.shape[0]==0:
                continue
            evsSum_dfs.append(evsSum_df_i)
        #-------------------------
        evsSum_df_fnl = CPXDfBuilder.concat_dfs(dfs = evsSum_dfs)
        #-----
        dtypes_dict   = CPXDfBuilder.get_evsSum_df_std_dtypes_dict(df = evsSum_df_fnl)
        evsSum_df_fnl = CPXDfBuilder.perform_std_col_type_conversions(
            evsSum_df   = evsSum_df_fnl, 
            dtypes_dict = dtypes_dict
        )
        return evsSum_df_fnl
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def combine_degenerate_columns(
        rcpx_df
    ):
        r"""
        Combine any identically-named (i.e., degenerate) columns in rcpx_df
        Good for single or MultiIndex columns
    
        The original method of using groupby with rcpx_df.columns is good for a clean one-liner
        HOWEVER, for large DFs I find it can take significantly more time, as the aggregation is
          done on all columns, regardless of whether or not they are degenerate
        THEREFORE, operate only on the degenerate columns!
        Original method:
            rcpx_df = rcpx_df.groupby(rcpx_df.columns, axis=1).sum() #Original, one-liner method
        """
        #-------------------------
        col_counts = rcpx_df.columns.value_counts()
        degen_cols = col_counts[col_counts>1].index.tolist()
        #-------------------------
        if len(degen_cols)==0:
            return rcpx_df
        #-------------------------
        for degen_col_i in degen_cols:
            # Build aggregate of degen_col_i
            agg_col_i = rcpx_df[degen_col_i].sum(axis=1)
            # Drop old degenerate columns
            rcpx_df = rcpx_df.drop(columns=degen_col_i)
            # Replace with aggregate column
            rcpx_df[degen_col_i] = agg_col_i
        #-------------------------
        return rcpx_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def convert_cr_cols_to_reasons(
        rcpx_df                  , 
        cr_trans_dict            = None, 
        total_counts_col         = 'total_counts', 
    ):
        r"""
        Rename the cr# columns to their full curated reasons.
        Combine any degenerates
        """
        #--------------------------------------------------
        if cr_trans_dict is None:
            _, cr_trans_dict = CPXDfBuilder.get_regex_setup()
        #-------------------------
        rcpx_df = rcpx_df.rename(columns=cr_trans_dict)
        #--------------------------------------------------
        # Any columns without a curated reason (i.e., those with column name = ''), have not been observed
        #   yet in the data, and therefore the sume of the counts should be 0.
        # These empty columns are not needed, so drop
        if '' in rcpx_df.columns.tolist():
            assert(rcpx_df[''].sum().sum()==0)
            rcpx_df = rcpx_df.drop(columns=[''])
        #--------------------------------------------------
        # Combine any degenerate columns
        rcpx_df = CPXDfBuilder.combine_degenerate_columns(rcpx_df = rcpx_df)
        #--------------------------------------------------
        # After irrelevant cleared and test columns removed, need to (re)calculate events_tot to accurately
        #   reflect the total number of relevant events
        # Safe(r) to do this calculation in any case, so moved outside of the if block above
        if total_counts_col is not None:
            reason_cols = list(set(rcpx_df.columns).intersection(set(cr_trans_dict.values())))
            non_reason_cols = list(set(rcpx_df.columns).difference(set(cr_trans_dict.values())))
            #-----
            rcpx_df[total_counts_col] = rcpx_df[reason_cols].sum(axis=1)
        #--------------------------------------------------
        return rcpx_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def project_time_pd_from_index_of_rcpx_0(
        rcpx_0                      , 
        time_pd_i                   , 
        meter_cnt_per_gp_srs        , 
        all_groups                  ,  
        cr_trans_dict               = None, 
        group_cols                  = ['trsf_pole_nb'], 
        time_pd_grp_idx             = 'aep_event_dt', 
        time_pd_i_rename            = None, 
        nSNs_col                    = '_nSNs', 
    ):
        r"""
        During the building process for rcpx_df from evsSum_df, rcpx_0 will be in a type of long form and need to be converted to wide form.
        It will be long with respect to the time groupings.
        Essentially, the rcpx_0 pd.DataFrame object will have a form where the indices include the group_cols together with the column used for the time grouping.
        The rows will be the normal, expected cr#s (with possibly 'total_counts', '_nSNs', etc.).
        Each time grouping needs to be grabbed, and, instead of stacked vertically, these will be stacked horizontally.

        e.g., at this stage, rcpx_0 will look like:
                        cr1	cr2	cr3	cr4	cr5	cr6	cr7	cr8	cr9	cr10	...	cr208	cr209	cr210	cr211	cr212	cr213	cr214	cr215	total_counts	nSNs
            outg_rec_nb	trsf_pole_nb	aep_event_dt																					
            13273622	1838236711629	2023-04-11	0	0	0	0	1	0	0	0	0	0	...	0	0	0	0	0	0	0	0	1	9
            13273623	1837931711315	2023-04-01	0	0	0	0	7	0	0	0	0	0	...	0	0	0	0	0	0	0	0	7	12
            13273623	1837931711315	2023-04-06	0	0	0	0	3	0	0	0	0	0	...	0	0	0	0	0	0	0	0	3	12
            13273623	1837931711315	2023-04-11	0	0	0	0	7	0	0	0	0	0	...	0	0	0	0	0	0	0	0	7	12
            13273623	1837931711315	2023-04-16	0	0	0	0	5	0	0	0	0	0	...	0	0	0	0	0	0	0	0	5	12
            13274933	1832586706337	2023-04-06	0	0	0	0	15	0	0	0	0	0	...	0	0	0	0	0	0	0	0	15	24
            13274933	1832586706337	2023-04-11	0	0	0	0	6	0	0	0	0	0	...	0	0	0	0	0	0	0	0	6	24
            13274933	1832586706337	2023-04-16	0	0	0	0	4	0	0	0	0	0	...	0	0	0	0	0	0	0	0	4	24
        and we want to project out each time grouping individually (here, 2023-04-06, 2023-04-11, 2023-04-16), and (outside of this function) stack together
          horizontally instead of vertically.
        Notice, e.g., how the group (outg_rec_nb, trsf_pole_nb) = (13273622, 1838236711629) has only the time grouping 2023-04-11.
          This means, for all other time periods, no events were registered.
          This is exactly why we must find and supplement with no_events_pd_i in the code below

        cr_trans_dict:
            If not supplied, will be grabbed via the CPXDfBuilder.get_regex_setup() method
        """
        #--------------------------------------------------
        if time_pd_i_rename is None:
            time_pd_i_rename = time_pd_i
        #--------------------------------------------------
        # Project out the current time period (time_pd_i) from rcpx_0 by selecting the appropriate
        #   values from the time_pd_grp_idx index
        rcpx_0_pd_i = rcpx_0[rcpx_0.index.get_level_values(time_pd_grp_idx)==time_pd_i].copy()
        rcpx_0_pd_i = rcpx_0_pd_i.droplevel(time_pd_grp_idx, axis=0)
        #-------------------------
        # Make sure all groups (typically trsf_pole_nbs) have an entry in rcpx_0_pd_i:
        #   If a group didn't register any events in a given time period, it will not be included in the projection.
        #   However, the final format requires each group have entries for each time period
        #   Therefore, we identify the groups missing from rcpx_0_pd_i (no_events_pd_i) and add approriate rows
        #     containing all 0 values for the counts
        # NOTE: If group_cols contains more than one element, the index will be a MultiIndex (equal to the group_cols)
        #         and will need to be treated slightly different
        no_events_pd_i = list(set(all_groups).difference(set(rcpx_0_pd_i.index.unique())))
        if len(no_events_pd_i)>0:
            if len(group_cols)==1:
                no_ev_idx = no_events_pd_i
            else:
                no_ev_idx = pd.MultiIndex.from_tuples(no_events_pd_i)
            #-------------------------
            no_events_pd_i_df = pd.DataFrame(
                columns = rcpx_0.columns, 
                index   = no_ev_idx, 
                data    = np.zeros((len(no_events_pd_i), rcpx_0.shape[1]))
            )
            no_events_pd_i_df.index.names = rcpx_0_pd_i.index.names
            #-------------------------
            # Use meter_cnt_per_gp_srs to fill the nSNs_col column in no_events_pd_i_df (since simply full of 0s)
            # NOTE: This is probably not strictly necessary, as the nSNs_col column won't be used here,
            #         since the data are not normalized.
            meter_cnt_per_gp_srs_pd_i = meter_cnt_per_gp_srs[meter_cnt_per_gp_srs.index.get_level_values(time_pd_grp_idx)==time_pd_i].droplevel(level=time_pd_grp_idx, axis=0)
            #-----
            # If a group did not register any events for time_pd_i, then it will be absent from meter_cnt_per_gp_srs_pd_i
            # In such cases, use the average number of meters per group from all other available date periods
            gps_w_missing_meter_cnts = list(set(all_groups).difference(set(meter_cnt_per_gp_srs_pd_i.index.tolist())))
            meter_cnt_per_gp_srs_pd_i = pd.concat([
                meter_cnt_per_gp_srs_pd_i, 
                meter_cnt_per_gp_srs[meter_cnt_per_gp_srs.index.droplevel(level=time_pd_grp_idx).isin(gps_w_missing_meter_cnts)].groupby(group_cols).mean()
            ])
            assert(set(meter_cnt_per_gp_srs_pd_i.index.tolist()).symmetric_difference(set(all_groups))==set())
            assert(meter_cnt_per_gp_srs_pd_i.shape[0]==meter_cnt_per_gp_srs_pd_i.index.nunique())
            #-------------------------
            no_events_pd_i_df = no_events_pd_i_df.drop(columns=[nSNs_col]).merge(
                meter_cnt_per_gp_srs_pd_i, 
                left_index  = True, 
                right_index = True, 
                how         = 'left'
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
        if cr_trans_dict is None:
            _, cr_trans_dict = CPXDfBuilder.get_regex_setup()
        rcpx_0_pd_i=rcpx_0_pd_i.rename(columns=cr_trans_dict)
        #--------------------------------------------------    
        #--------------------------------------------------
        # Any columns without a curated reason (i.e., those with column name = ''), have not been observed
        #   yet in the data, and therefore the sume of the counts should be 0.
        # These empty columns are not needed, so drop
        assert(rcpx_0_pd_i[''].sum().sum()==0)
        rcpx_0_pd_i=rcpx_0_pd_i.drop(columns=[''])  
        #-------------------------
        #-------------------------
        # Combine any degenerate columns
        rcpx_0_pd_i = CPXDfBuilder.combine_degenerate_columns(rcpx_df = rcpx_0_pd_i)
            
        #--------------------------------------------------
        #--------------------------------------------------
        # Add the correct time period name as level 0 of the columns
        rcpx_0_pd_i = Utilities_df.prepend_level_to_MultiIndex(
            df         = rcpx_0_pd_i, 
            level_val  = time_pd_i_rename, 
            level_name = None, 
            axis       = 1
        )
        #-------------------------
        return rcpx_0_pd_i


    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df(
        evsSum_df                   , 
        prediction_date             , 
        td_min                      , 
        td_max                      , 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        freq                        = '5D', 
        cr_trans_dict               = None, 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        date_col                    = 'aep_event_dt', 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        total_counts_col            = 'total_counts', 
        nSNs_col                    = '_nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.

        This function assumes a single prediction_date for all.  If groups have unique prediction dates, use 
          CPXDfBuilder.build_rcpx_from_evsSum_df_wpreddates instead.
          
        NOTE: td_min, td_max, and freq must all be in DAYS
    
        cr_trans_dict:
            If not supplied, will be grabbed via the CPXDfBuilder.get_regex_setup() method
        """
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        #-------------------------
        Utilities_dt.assert_timedelta_is_days(td_min)
        Utilities_dt.assert_timedelta_is_days(td_max)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        if freq is not None:
            freq   = pd.Timedelta(freq)
            Utilities_dt.assert_timedelta_is_days(freq)
        #--------------------------------------------------
        #-------------------------
        prereq_dict = CPXDfBuilder.perform_build_rcpx_from_end_events_df_prereqs(
            end_events_df                 = evsSum_df, 
            mp_df                         = None, 
            ede_mp_mismatch_threshold_pct = 1.0, 
            grp_by_cols                   = group_cols, 
            outg_rec_nb_col               = outg_rec_nb_col,
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            prem_nb_col                   = prem_nb_col, 
            serial_number_col             = serial_number_col,
            mp_df_cols                    = None,  # Only needed if mp_df is not None
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase    = make_all_columns_lowercase
        )
        group_cols        = prereq_dict['grp_by_cols']
        outg_rec_nb_col   = prereq_dict['outg_rec_nb_col']
        trsf_pole_nb_col  = prereq_dict['trsf_pole_nb_col']
        prem_nb_col       = prereq_dict['prem_nb_col']
        serial_number_col = prereq_dict['serial_number_col']
        cols_to_drop      = prereq_dict['cols_to_drop']
        rename_cols_dict  = prereq_dict['rename_cols_dict']
        mp_df             = prereq_dict['mp_df']
        merge_on_ede      = prereq_dict['merge_on_ede']
        merge_on_mp       = prereq_dict['merge_on_mp']
        #--------------------------------------------------
        #-------------------------
        if make_all_columns_lowercase:
            date_col          = date_col.lower()
            xf_meter_cnt_col  = xf_meter_cnt_col.lower()
            events_tot_col    = events_tot_col.lower()
    
    
        #--------------------------------------------------
        # Look for any columns ending in _GPD_FOR_SQL and if not included in grpyby cols, then print warning!
        found_gpd_for_sql_cols = CPXDfBuilder.find_gpd_for_sql_cols(
            df        = evsSum_df, 
            col_level = -1
        )
        if len(found_gpd_for_sql_cols)>0:
            # make loweercase, if needed
            if make_all_columns_lowercase:
                found_gpd_for_sql_cols = [x.lower() for x in found_gpd_for_sql_cols]
            # change names, if needed
            found_gpd_for_sql_cols = [rename_cols_dict.get(x, x) for x in found_gpd_for_sql_cols]
            #-----
            not_included = list(set(found_gpd_for_sql_cols).difference(set(group_cols)))
            if len(not_included)>0:
                print('\n!!!!! WARNING !!!!!\nCPXDfBuilder.build_rcpx_from_evsSum_df\nFOUND POSSIBLE GROUPBY COLUMNS NOT INCLUDED IN group_cols argument')
                print(f"\tnot_included           = {not_included}\n\tfound_gpd_for_sql_cols = {found_gpd_for_sql_cols}\n\tgroup_cols             = {group_cols}")   
                print('!!!!!!!!!!!!!!!!!!!\n')
    
        #-------------------------
        evsSum_df = CPXDfBuilder.perform_end_events_std_col_renames_and_drops(
            end_events_df              = evsSum_df.copy(), 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-----
        nec_cols = group_cols + [date_col, xf_meter_cnt_col, events_tot_col, trsf_pole_nb_col]
        assert(set(nec_cols).difference(set(evsSum_df.columns.tolist()))==set()) 
    
        #--------------------------------------------------
        # 1. Build rcpx_0
        #     Construct rcpx_0 by aggregating evsSum_df by group_cols and by freq on date_col
        #--------------------------------------------------
        dtypes_dict = CPXDfBuilder.get_evsSum_df_std_dtypes_dict(df = evsSum_df)
        evsSum_df   = CPXDfBuilder.perform_std_col_type_conversions(
            evsSum_df   = evsSum_df, 
            dtypes_dict = dtypes_dict
        )
        #-------------------------
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        assert(len(set(group_cols).difference(set(evsSum_df.columns.tolist())))==0)
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = evsSum_df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        #-----
        # cols_to_drop below are different from cols_to_drop = prereq_dict['cols_to_drop'] above.
        #   That defined above is used to remove cases when col_x and f"{col_x}_GPD_FOR_SQL" both exist in the data.
        #     For such cases, the trust_sql_grouping parameter settles which is kept.
        #   For that defined below, cols_to_drop is used to remove all the columns no longer needed for the analysis
        #       Typically, e.g., group_cols = ['outg_rec_nb', 'trsf_pole_nb']
        #       ==> cols_to_drop = ['is_first_after_outg', 'aep_opco', 'serialnumber', 'aep_premise_nb']
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
        # Note: Open left (does not include endpoint), closed right (does include endpoint) below
        #       This seems opposite to what one would intuitively think, but one must remeber that the
        #         events are all prior to, and with respect to, some prediction_date.
        #       Therefore, e.g., for a '01-06 Days' bin, one typically wants [1 Day, 6 Days)
        #       In reality, [1 Day, 6 Days) is (-6 Days, -1 Day], hence the open left and closed right below.
        evsSum_df = evsSum_df[
            (evsSum_df[date_col] >  prediction_date-td_max) & 
            (evsSum_df[date_col] <= prediction_date-td_min)
        ]
    
        #-------------------------
        # All of the cr# columns will be aggregated with np.sum, as will events_tot_col
        # xf_meter_cnt_col will be aggregated using np.max
        agg_dict = {col:'sum' for col in cr_cols+[events_tot_col, xf_meter_cnt_col]}
        agg_dict[xf_meter_cnt_col] = 'max'
    
        if freq is None:
            time_grps = [(prediction_date-td_max, prediction_date-td_min)]
            #--------------------------------------------------
            rcpx_0 = evsSum_df.drop(columns=cols_to_drop).groupby(group_cols).agg(agg_dict)
            #--------------------------------------------------
            str_len = np.max([len(str(x)) for x in [days_min, days_max]])
            str_fmt = f'{{:0{str_len}d}}'
            time_pds_rename = {time_grps[0][0] : "{}-{} Days".format(str_fmt.format(days_min), str_fmt.format(days_max))}
            #-------------------------
            # For functionality, need date_col (this function was originally designed for use with non-None freq!)
            rcpx_0[date_col] = time_grps[0][0]
            rcpx_0 = rcpx_0.reset_index(drop=False).set_index(group_cols+[date_col])
        else:
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
            assert(len(time_grps)>1)
            time_grps = [(time_grps[i], time_grps[i+1]) for i in range(len(time_grps)-1)]
            assert(len(time_grps) == (td_max-td_min)/pd.Timedelta(freq))
            #-------------------------
            group_freq = pd.Grouper(freq=freq, key=date_col, origin=time_grps[0][0], closed='right')
            #--------------------------------------------------
            rcpx_0 = evsSum_df.drop(columns=cols_to_drop).groupby(group_cols+[group_freq]).agg(agg_dict)
            #--------------------------------------------------
            time_pds_rename = CPXDfBuilder.get_time_pds_rename(
                curr_time_pds = time_grps, 
                td_min        = td_min, 
                td_max        = td_max, 
                freq          = freq
            )
        final_time_pds = list(time_pds_rename.values())
    
        #--------------------------------------------------
        # 2. Grab meter_cnt_per_gp_srs and all_groups
        #--------------------------------------------------
        # Project out the meter count per group, as it will be used later
        #   This information will be stored in the pd.Series object meter_cnt_per_gp_srs, where the index will
        #   contain the group_cols
        #-----
        # OLD METHOD
        # meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[date_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[date_col]).squeeze()
        #-------------------------
        meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[date_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[date_col])
        # Generally, calling .squeeze() in this case works fine, UNLESS meter_cnt_per_gp_srs (which is a pd.DataFrame object as this point)
        #   has a single row, in which case a scalar is returned instead of a pd.Series object.
        # From the pd.DataFrame.squeeze documentation:
        #   Series or DataFrames with a single element are squeezed to a scalar.
        # To overcome the issue here, I specifically call .squeeze(axis=1)
        # For the more general case, where meter_cnt_per_gp_srs has multiple rows, calling .squeeze(axis=1) should deliver the same results 
        #   as calling .squeeze()
        # NOTE ALSO: .squeeze(axis=1) could have been tacked on to the above command.  I have broken it out into two steps for illustrative purposes, 
        #              so the explanation above makes sense!
        meter_cnt_per_gp_srs = meter_cnt_per_gp_srs.squeeze(axis=1)
        #-------------------------
        assert(meter_cnt_per_gp_srs.shape[0]==meter_cnt_per_gp_srs.index.nunique())
        meter_cnt_per_gp_srs.name = nSNs_col
    
        # Will also need the unique groups in rcpx_0
        #   This will be used later (see no_events_pd_i below)
        #   These can be grabbed from the index of rcpx_0 (excluding the date_col level)
        all_groups = rcpx_0.droplevel(date_col, axis=0).index.unique().tolist()
    
        #--------------------------------------------------
        # 3. Transform rcpx_0 to the form expected by the model
        #     This is essentially just changing rcpx_0 from long form to wide form
        #--------------------------------------------------
        #-------------------------
        #    time_pds_rename
        #      Need to convert the time periods, which are currently housed in the date_col index of 
        #        rcpx_0 from their specific dates to the names expected by the model.
        #      In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
        #        dates of the given interval.
        #      These will be converted to the titles contained in final_time_pds below
        #      NOTE: This is probably not 100% necessary, but is useful nonetheless
        #--------------------------------------------------    
        rename_cols = {
            events_tot_col   : total_counts_col, 
            xf_meter_cnt_col : nSNs_col
        }
        rcpx_0=rcpx_0.rename(columns=rename_cols)
        #-------------------------
        total_counts_col = total_counts_col
        nSNs_col         = nSNs_col
        #------------------------- 
        pd_dfs = []
        for time_grp_i in time_grps:
            time_pd_i = time_grp_i[0]
            # Grab the proper time period name from time_pds_rename
            time_pd_i_rename = time_pds_rename[time_pd_i]
            #-------------------------
            rcpx_0_pd_i = CPXDfBuilder.project_time_pd_from_index_of_rcpx_0(
                rcpx_0                      = rcpx_0, 
                time_pd_i                   = time_pd_i, 
                meter_cnt_per_gp_srs        = meter_cnt_per_gp_srs, 
                all_groups                  = all_groups, 
                cr_trans_dict               = cr_trans_dict, 
                group_cols                  = group_cols, 
                time_pd_grp_idx             = date_col, 
                time_pd_i_rename            = time_pd_i_rename, 
                nSNs_col                    = nSNs_col
            )
            #-------------------------
            # Overkill here (since all time windows are of length freq), but something similar will 
            #   be needed if I want to move to non-uniform period lengths
            # One could, e.g., simply divide by length of freq in days
            if normalize_by_time:
                days_min_outg_td_window_i = prediction_date - time_grp_i[1]
                days_max_outg_td_window_i = prediction_date - time_grp_i[0]
                #-----
                Utilities_dt.assert_timedelta_is_days(days_min_outg_td_window_i)
                Utilities_dt.assert_timedelta_is_days(days_max_outg_td_window_i)
                #-----
                days_min_outg_td_window_i = days_min_outg_td_window_i.days
                days_max_outg_td_window_i = days_max_outg_td_window_i.days
                #-------------------------
                nSNs_b4 = rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].copy()
                #-----
                rcpx_0_pd_i = CPXDfBuilder.normalize_rcpx_df_by_time_interval(
                    rcpx_df                 = rcpx_0_pd_i, 
                    days_min_outg_td_window = days_min_outg_td_window_i, 
                    days_max_outg_td_window = days_max_outg_td_window_i, 
                    cols_to_adjust          = None, 
                    SNs_tags                = None, 
                    inplace                 = True
                )
                #-----
                assert(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].equals(nSNs_b4))
            #-------------------------
            if normalize_by_SNs:
                cols_to_norm              = [x for x in rcpx_0_pd_i.columns if x[1]!=nSNs_col]
                rcpx_0_pd_i[cols_to_norm] = rcpx_0_pd_i[cols_to_norm].divide(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)], axis=0)
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
        # rcpx_final = pd.concat(pd_dfs, axis=1)
        rcpx_final = CPXDfBuilder.merge_cpx_dfs(
            dfs_coll                      = pd_dfs, 
            col_level                     = -1,
            max_total_counts              = None, 
            how_max_total_counts          = 'any', 
            XNs_tags                      = None, 
            cols_to_init_with_empty_lists = None, 
            make_cols_equal               = False, 
            sort_cols                     = False, 
        )
    
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
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_dfs_in_dir(
        files_dir                      , 
        prediction_date                , 
        td_min                         , 
        td_max                         , 
        return_evsSum_df               = True, 
        #-----
        file_path_glob                 = r'events_summary_[0-9]*.csv', 
        file_path_regex                = None, 
        batch_size                     = 50, 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        make_all_columns_lowercase     = True, 
        assert_all_cols_equal          = True,  
        n_update                       = 1, 
        #-----
        cr_trans_dict                  = None, 
        freq                           = '5D', 
        group_cols                     = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        date_col                       = 'aep_event_dt', 
        normalize_by_SNs               = True, 
        normalize_by_time              = True, 
        xf_meter_cnt_col               = 'xf_meter_cnt', 
        events_tot_col                 = 'events_tot', 
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        outg_rec_nb_col                = 'outg_rec_nb',
        prem_nb_col                    = 'aep_premise_nb', 
        serial_number_col              = 'serialnumber',
        total_counts_col               = 'total_counts', 
        nSNs_col                       = '_nSNs', 
        trust_sql_grouping             = True, 
        drop_gpd_for_sql_appendix      = True, 
        verbose                        = True, 
    ):
        r"""
        This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
          
        NOTE: td_min, td_max, and freq must all be in DAYS
    
        data_structure_df:
            If supplied, it will be used to determine the set of final columns desired in rcpx_0_pd_i, and the method
              MECPODf.get_reasons_subset_from_cpo_df will be used to adjust rcpx_0_pd_i accordingly.
            If not supplied, MECPODf.get_reasons_subset_from_cpo_df is not calleds
    
        cr_trans_dict:
            If not supplied, will be grabbed via the CPXDfBuilder.get_regex_setup() method
        """
        #--------------------------------------------------
        evsSum_df = CPXDfBuilder.build_evsSum_df_from_csvs(
            files_dir                      = files_dir, 
            file_path_glob                 = file_path_glob, 
            file_path_regex                = file_path_regex, 
            batch_size                     = batch_size, 
            cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors              = to_numeric_errors, 
            make_all_columns_lowercase     = make_all_columns_lowercase, 
            assert_all_cols_equal          = assert_all_cols_equal,  
            n_update                       = n_update, 
            verbose                        = verbose
        )
        #--------------------------------------------------
        if evsSum_df is None or evsSum_df.shape[0]==0:
            return None
        #--------------------------------------------------
        rcpx_df = CPXDfBuilder.build_rcpx_from_evsSum_df(
            evsSum_df                   = evsSum_df, 
            prediction_date             = prediction_date, 
            td_min                      = td_min, 
            td_max                      = td_max, 
            cr_trans_dict               = cr_trans_dict, 
            freq                        = freq, 
            group_cols                  = group_cols, 
            date_col                    = date_col, 
            normalize_by_SNs            = normalize_by_SNs, 
            normalize_by_time           = normalize_by_time, 
            xf_meter_cnt_col            = xf_meter_cnt_col, 
            events_tot_col              = events_tot_col, 
            outg_rec_nb_col             = outg_rec_nb_col,
            trsf_pole_nb_col            = trsf_pole_nb_col, 
            prem_nb_col                 = prem_nb_col, 
            serial_number_col           = serial_number_col, 
            total_counts_col            = total_counts_col, 
            nSNs_col                    = nSNs_col, 
            trust_sql_grouping          = trust_sql_grouping, 
            drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase  = make_all_columns_lowercase
        )
        #--------------------------------------------------
        if return_evsSum_df:
            return rcpx_df, evsSum_df
        return rcpx_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df_wpreddates_simple(
        evsSum_df                   , 
        pred_date_col               , 
        td_min                      , 
        td_max                      , 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        freq                        = '5D', 
        cr_trans_dict               = None, 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        date_col                    = 'aep_event_dt', 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        total_counts_col            = 'total_counts', 
        nSNs_col                    = '_nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        """
        #--------------------------------------------------
        # We will alter evsSum_df.  To keep consistent outside of function, create copy
        evsSum_df = evsSum_df.copy()
        #--------------------------------------------------
        # Build relative date column, which will be used for grouping
        # rel_date_col will essentially replace date_col in build_rcpx_from_evsSum_df
        #-----
        rel_date_col = Utilities.generate_random_string(letters='letters_only')
        evsSum_df[rel_date_col] = evsSum_df[pred_date_col]-evsSum_df[date_col]
        #-----
        # Shift everything relative to a common dummy prediction date
        dummy_date     = pd.to_datetime('2 Nov. 1987')
        dummy_date_col = Utilities.generate_random_string(letters='letters_only')
        evsSum_df[dummy_date_col] = dummy_date - evsSum_df[rel_date_col]
        evsSum_df = evsSum_df.drop(columns=[date_col, rel_date_col, pred_date_col])
    
        #--------------------------------------------------
        # Now that everything is relative to common date, simply use CPXDfBuilder.build_rcpx_from_evsSum_df 
        #   with prediction_date=dummy_date and date_col=dummy_date_col
        rcpx_df = CPXDfBuilder.build_rcpx_from_evsSum_df(
            evsSum_df                   = evsSum_df, 
            prediction_date             = dummy_date, 
            td_min                      = td_min, 
            td_max                      = td_max,
            group_cols                  = group_cols,  
            freq                        = freq, 
            cr_trans_dict               = cr_trans_dict, 
            normalize_by_SNs            = normalize_by_SNs, 
            normalize_by_time           = normalize_by_time, 
            date_col                    = dummy_date_col, 
            xf_meter_cnt_col            = xf_meter_cnt_col, 
            events_tot_col              = events_tot_col, 
            outg_rec_nb_col             = outg_rec_nb_col,
            trsf_pole_nb_col            = trsf_pole_nb_col, 
            prem_nb_col                 = prem_nb_col, 
            serial_number_col           = serial_number_col,
            total_counts_col            = total_counts_col, 
            nSNs_col                    = nSNs_col, 
            trust_sql_grouping          = trust_sql_grouping, 
            drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase  = make_all_columns_lowercase, 
        )
    
        #--------------------------------------------------
        return rcpx_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df_wpreddates_full(
        evsSum_df                   , 
        pred_date_col               , 
        td_min                      , 
        td_max                      , 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        freq                        = '5D', 
        cr_trans_dict               = None, 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        date_col                    = 'aep_event_dt', 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        total_counts_col            = 'total_counts', 
        nSNs_col                    = '_nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        This was the original idea for the function; HOWEVER, after partial completion, the _SIMPLE method became apparent.
        I suggest using the _SIMPLE version, because it is simpler and relies on CPXDfBuilder.build_rcpx_from_evsSum_df.
        THIS METHOD IS KEPT BECAUSE:
            - It provides a slightly different way of solving the problem, and a similar solution may be desired elsewhere later.
            - In case the _SIMPLE method does not work in some specific case, it will be interesting to investigate whether or not
                this method worked, and why.
            
        """
        #--------------------------------------------------
        # We will alter evsSum_df.  To keep consistent outside of function, create copy
        evsSum_df = evsSum_df.copy()
        #--------------------------------------------------
        # Build relative date column, which will be used for grouping
        # rel_date_col will essentially replace date_col in build_rcpx_from_evsSum_df
        #-----
        rel_date_col = Utilities.generate_random_string(letters='letters_only')
        evsSum_df[rel_date_col] = evsSum_df[pred_date_col]-evsSum_df[date_col]
        
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        #-------------------------
        Utilities_dt.assert_timedelta_is_days(td_min)
        Utilities_dt.assert_timedelta_is_days(td_max)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        if freq is not None:
            freq   = pd.Timedelta(freq)
            Utilities_dt.assert_timedelta_is_days(freq)
            days_freq = freq.days
        #--------------------------------------------------
        #-------------------------
        prereq_dict = CPXDfBuilder.perform_build_rcpx_from_end_events_df_prereqs(
            end_events_df                 = evsSum_df, 
            mp_df                         = None, 
            ede_mp_mismatch_threshold_pct = 1.0, 
            grp_by_cols                   = group_cols, 
            outg_rec_nb_col               = outg_rec_nb_col,
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            prem_nb_col                   = prem_nb_col, 
            serial_number_col             = serial_number_col,
            mp_df_cols                    = None,  # Only needed if mp_df is not None
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase    = make_all_columns_lowercase
        )
        group_cols        = prereq_dict['grp_by_cols']
        outg_rec_nb_col   = prereq_dict['outg_rec_nb_col']
        trsf_pole_nb_col  = prereq_dict['trsf_pole_nb_col']
        prem_nb_col       = prereq_dict['prem_nb_col']
        serial_number_col = prereq_dict['serial_number_col']
        cols_to_drop      = prereq_dict['cols_to_drop']
        rename_cols_dict  = prereq_dict['rename_cols_dict']
        mp_df             = prereq_dict['mp_df']
        merge_on_ede      = prereq_dict['merge_on_ede']
        merge_on_mp       = prereq_dict['merge_on_mp']
        #--------------------------------------------------
        #-------------------------
        if make_all_columns_lowercase:
            date_col          = date_col.lower()
            xf_meter_cnt_col  = xf_meter_cnt_col.lower()
            events_tot_col    = events_tot_col.lower()
            rel_date_col      = rel_date_col.lower()
        
        #--------------------------------------------------
        # Look for any columns ending in _GPD_FOR_SQL and if not included in grpyby cols, then print warning!
        found_gpd_for_sql_cols = CPXDfBuilder.find_gpd_for_sql_cols(
            df        = evsSum_df, 
            col_level = -1
        )
        if len(found_gpd_for_sql_cols)>0:
            # make loweercase, if needed
            if make_all_columns_lowercase:
                found_gpd_for_sql_cols = [x.lower() for x in found_gpd_for_sql_cols]
            # change names, if needed
            found_gpd_for_sql_cols = [rename_cols_dict.get(x, x) for x in found_gpd_for_sql_cols]
            #-----
            not_included = list(set(found_gpd_for_sql_cols).difference(set(group_cols)))
            if len(not_included)>0:
                print('\n!!!!! WARNING !!!!!\nCPXDfBuilder.build_rcpx_from_evsSum_df\nFOUND POSSIBLE GROUPBY COLUMNS NOT INCLUDED IN group_cols argument')
                print(f"\tnot_included           = {not_included}\n\tfound_gpd_for_sql_cols = {found_gpd_for_sql_cols}\n\tgroup_cols             = {group_cols}")   
                print('!!!!!!!!!!!!!!!!!!!\n')
        
        #-------------------------
        evsSum_df = CPXDfBuilder.perform_end_events_std_col_renames_and_drops(
            end_events_df              = evsSum_df.copy(), 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-----
        nec_cols = group_cols + [rel_date_col, xf_meter_cnt_col, events_tot_col, trsf_pole_nb_col]
        assert(set(nec_cols).difference(set(evsSum_df.columns.tolist()))==set()) 
        
        #--------------------------------------------------
        # 1. Build rcpx_0
        #     Construct rcpx_0 by aggregating evsSum_df by group_cols and by freq on rel_date_col
        #--------------------------------------------------
        dtypes_dict = CPXDfBuilder.get_evsSum_df_std_dtypes_dict(df = evsSum_df)
        evsSum_df   = CPXDfBuilder.perform_std_col_type_conversions(
            evsSum_df   = evsSum_df, 
            dtypes_dict = dtypes_dict
        )
        #-------------------------
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        assert(len(set(group_cols).difference(set(evsSum_df.columns.tolist())))==0)
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = evsSum_df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        #-----
        # cols_to_drop below are different from cols_to_drop = prereq_dict['cols_to_drop'] above.
        #   That defined above is used to remove cases when col_x and f"{col_x}_GPD_FOR_SQL" both exist in the data.
        #     For such cases, the trust_sql_grouping parameter settles which is kept.
        #   For that defined below, cols_to_drop is used to remove all the columns no longer needed for the analysis
        #       Typically, e.g., group_cols = ['outg_rec_nb', 'trsf_pole_nb']
        #       ==> cols_to_drop = ['is_first_after_outg', 'aep_opco', 'serialnumber', 'aep_premise_nb']
        cols_to_drop = set(evsSum_df.columns.tolist()).difference(
            set(cr_cols+group_cols+[rel_date_col, xf_meter_cnt_col, events_tot_col])
        )
        cols_to_drop = list(cols_to_drop)
        #-------------------------
        # Make sure rel_date_col is timedelta object
        assert(is_timedelta64_dtype(evsSum_df[rel_date_col].dtype))
        
        #-------------------------
        # No need in wasting time grouping data we won't use
        # So, reduce evsSum_df to only the dates we're interested in 
        # Note: Closed left (does include endpoint), open right (does not include endpoint) below
        #       e.g., '01-06 Days' = [1 Day, 6 Days)
        evsSum_df = evsSum_df[
            (evsSum_df[rel_date_col] >= td_min) &
            (evsSum_df[rel_date_col] <  td_max)
        ].copy()
        
        #-------------------------
        # All of the cr# columns will be aggregated with np.sum, as will events_tot_col
        # xf_meter_cnt_col will be aggregated using np.max
        agg_dict = {col:'sum' for col in cr_cols+[events_tot_col, xf_meter_cnt_col]}
        agg_dict[xf_meter_cnt_col] = 'max'
    
        #--------------------------------------------------
        time_pd_grp_col       = Utilities.generate_random_string(letters='letters_only')
        time_pd_grp_width_col = Utilities.generate_random_string(letters='letters_only')
        #-------------------------
        if freq is None:
            evsSum_df[time_pd_grp_col]       = 1
            evsSum_df[time_pd_grp_width_col] = days_max-days_min
            grps_w_widths                    = [(1, days_max-days_min)]
        else:
            evsSum_df[time_pd_grp_col]       = np.digitize(evsSum_df[rel_date_col].dt.days, np.arange(days_min, days_max+1, days_freq))
            evsSum_df[time_pd_grp_width_col] = days_freq 
            
            grps_w_widths = list(evsSum_df.groupby([time_pd_grp_col, time_pd_grp_width_col]).groups.keys())
            # Make sure grps_w_widths is properly sorted
            grps_w_widths = natsorted(grps_w_widths, key=lambda x:x[0])
            # Also note, for np.digitize, the indices apparently begin at 1, not 0
            # assertion below ensures this is true
            assert(grps_w_widths[0][0]==1)
        # Since we are using custom agg_dict, there is no need to call evsSum_df.drop(columns=cols_to_drop) before grouping/aggregation
        rcpx_0 = evsSum_df.groupby(group_cols+[time_pd_grp_col]).agg(agg_dict)
        
        #--------------------------------------------------
        # Using grps_w_widths is a bit overkill currently, since freq is constant.
        # However, if freq is allowed to be irregular in the future, the overkill included
        #   will be necessary
        #-------------------------
        time_pds_rename = {}
        #-------------------------
        # Use padded zeros
        #   i.e., instead of '1-6 Days', use '01-06 Days'
        #   Will need to know longest string length for this
        #-----
        # Largest number, therefore largest str, will be days_min + all widths
        str_len = len(str(days_min + np.sum([x[1] for x in grps_w_widths])))
        #-----
        # Looks like a ton of curly braces below, but need double {{ and double }} to escape
        # ==> if str_len=2, then str_fmt = '{:02d}'
        str_fmt = f'{{:0{str_len}d}}'
        #-------------------------
        t_rght_i = days_min
        # Remember, since np.digitize was used, the indices apparently begin at 1, not 0
        for grp_i, width_i in grps_w_widths:
            t_left_i = t_rght_i
            t_rght_i = t_left_i + width_i
            #-----
            rename_i = "{}-{} Days".format(
                str_fmt.format(t_left_i), 
                str_fmt.format(t_rght_i)
            )
            #-----
            assert(grp_i not in time_pds_rename.keys())
            time_pds_rename[grp_i] = rename_i
        
        #--------------------------------------------------
        # 2. Grab meter_cnt_per_gp_srs and all_groups
        #--------------------------------------------------
        # Project out the meter count per group, as it will be used later
        #   This information will be stored in the pd.Series object meter_cnt_per_gp_srs, where the index will
        #   contain the group_cols
        #-----
        # OLD METHOD
        # meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[time_pd_grp_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[time_pd_grp_col]).squeeze()
        #-------------------------
        meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[time_pd_grp_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[time_pd_grp_col])
        # Generally, calling .squeeze() in this case works fine, UNLESS meter_cnt_per_gp_srs (which is a pd.DataFrame object as this point)
        #   has a single row, in which case a scalar is returned instead of a pd.Series object.
        # From the pd.DataFrame.squeeze documentation:
        #   Series or DataFrames with a single element are squeezed to a scalar.
        # To overcome the issue here, I specifically call .squeeze(axis=1)
        # For the more general case, where meter_cnt_per_gp_srs has multiple rows, calling .squeeze(axis=1) should deliver the same results 
        #   as calling .squeeze()
        # NOTE ALSO: .squeeze(axis=1) could have been tacked on to the above command.  I have broken it out into two steps for illustrative purposes, 
        #              so the explanation above makes sense!
        meter_cnt_per_gp_srs = meter_cnt_per_gp_srs.squeeze(axis=1)
        #-------------------------
        assert(meter_cnt_per_gp_srs.shape[0]==meter_cnt_per_gp_srs.index.nunique())
        meter_cnt_per_gp_srs.name = nSNs_col
        
        # Will also need the unique groups in rcpx_0
        #   This will be used later (see no_events_pd_i below)
        #   These can be grabbed from the index of rcpx_0 (excluding the time_pd_grp_col level)
        all_groups = rcpx_0.droplevel(time_pd_grp_col, axis=0).index.unique().tolist()
        
        #--------------------------------------------------
        # 3. Transform rcpx_0 to the form expected by the model
        #     i.e., similar to data_structure_df (if supplied).
        #     This is essentially just changing rcpx_0 from long form to wide form
        #--------------------------------------------------
        #-------------------------
        # 3a. Build time_pds_rename
        #      Need to convert the time periods, which are currently housed in the time_pd_grp_col index of 
        #        rcpx_0 from their specific dates to the names expected by the model.
        #      In rcpx_0, after grouping by the freq intervals, the values of time_pd_grp_col are equal to the beginning
        #        dates of the given interval.
        #      These will be converted to the titles contained in final_time_pds below
        #      NOTE: This is probably not 100% necessary, but is useful nonetheless
        #-------------------------
        #--------------------------------------------------
        #    Need for data_structure_df?
        #     In general, not all curated reasons will be included in the model.
        #     Typically, 10 commong curated reasons will be included, and all others will be grouped together in "Other Reasons".
        #     Furthermore, some reasons may be combined together, others may be completely removed.
        #     For these reasons, it is beneficial to have some sample data (taken from when the model was created) to utilize 
        #       in structuring the new data in the same fashion.
        #     Additionally, the data will be used to ensure the ordering of columns is correct before the data are fed into 
        #       the model.
        #--------------------------------------------------
        # if data_structure_df is not None:
        #     assert(isinstance(data_structure_df, pd.DataFrame) and data_structure_df.shape[0]>0)
        #     # final_time_pds should all be found in data_structure_df to help
        #     #   ensure the alignment between the current data and data used when modelling
        #     assert(set(final_time_pds).difference(data_structure_df.columns.get_level_values(0).unique())==set())
        
        #-------------------------
        # 3b. Transform rcpx_0 to the form expected by the model
        #      As stated above, this is essentially just changing rcpx_0 from long form to wide form
        #      This will probably be formalized further in the future (i.e., function(s) developed to handle)
        rename_cols = {
            events_tot_col   : total_counts_col, 
            xf_meter_cnt_col : nSNs_col
        }
        rcpx_0=rcpx_0.rename(columns=rename_cols)
        #-------------------------
        total_counts_col = total_counts_col
        nSNs_col         = nSNs_col
        non_reason_cols  = [nSNs_col, total_counts_col]
        #------------------------- 
        pd_dfs = []
        for grp_i, width_i in grps_w_widths:
            time_pd_i = grp_i
            # Grab the proper time period name from final_time_pd_i
            time_pd_i_rename = time_pds_rename[time_pd_i]
            #-------------------------
            rcpx_0_pd_i = CPXDfBuilder.project_time_pd_from_index_of_rcpx_0(
                rcpx_0                      = rcpx_0, 
                time_pd_i                   = time_pd_i, 
                meter_cnt_per_gp_srs        = meter_cnt_per_gp_srs, 
                all_groups                  = all_groups, 
                cr_trans_dict               = cr_trans_dict, 
                group_cols                  = group_cols, 
                time_pd_grp_idx             = time_pd_grp_col, 
                time_pd_i_rename            = time_pd_i_rename, 
                nSNs_col                    = nSNs_col
            )
            #-------------------------
            # Overkill here (since all time windows are of length freq), but something similar will 
            #   be needed if I want to move to non-uniform period lengths
            # One could, e.g., simply divide by length of freq in days
            if normalize_by_time:
                #-------------------------
                nSNs_b4 = rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].copy()
                #-----
                rcpx_0_pd_i = CPXDfBuilder.normalize_rcpx_df_by_time_interval(
                    rcpx_df                 = rcpx_0_pd_i, 
                    days_min_outg_td_window = 0, 
                    days_max_outg_td_window = width_i, 
                    cols_to_adjust          = None, 
                    SNs_tags                = None, 
                    inplace                 = True
                )
                #-----
                assert(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].equals(nSNs_b4))
            #-------------------------
            if normalize_by_SNs:
                cols_to_norm = [x for x in rcpx_0_pd_i.columns if x[1]!=nSNs_col]
                rcpx_0_pd_i[cols_to_norm] = rcpx_0_pd_i[cols_to_norm].divide(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)], axis=0)
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
        # rcpx_final = pd.concat(pd_dfs, axis=1)
        rcpx_final = CPXDfBuilder.merge_cpx_dfs(
            dfs_coll                      = pd_dfs, 
            col_level                     = -1,
            max_total_counts              = None, 
            how_max_total_counts          = 'any', 
            XNs_tags                      = CPXDfBuilder.std_XNs_cols() + CPXDfBuilder.std_nXNs_cols(), 
            cols_to_init_with_empty_lists = CPXDfBuilder.std_XNs_cols(), 
            make_cols_equal               = False,  # b/c if data_structure_df provided, pd_dfs built accordingly
            sort_cols                     = False, 
        )
        
        #-------------------------
        # For the number of SNs per group, use the average number of 
        #   SNs for each group across all time periods
        nSNs_cols = [x for x in rcpx_final.columns if x[1]==nSNs_col]
        #-----
        # # There should be one nSNs col for each time period
        # assert(len(nSNs_cols)==len(final_time_pds))
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
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df_wpreddates(
        evsSum_df                   , 
        pred_date_col               , 
        td_min                      , 
        td_max                      , 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        freq                        = '5D', 
        cr_trans_dict               = None, 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        date_col                    = 'aep_event_dt', 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        total_counts_col            = 'total_counts', 
        nSNs_col                    = '_nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        Try _simple method first, if that doesn't work, try _full method
        """
        #--------------------------------------------------
        try:
            rcpx_df = CPXDfBuilder.build_rcpx_from_evsSum_df_wpreddates_simple(
                evsSum_df                   = evsSum_df, 
                pred_date_col               = pred_date_col, 
                td_min                      = td_min, 
                td_max                      = td_max, 
                group_cols                  = group_cols, 
                freq                        = freq, 
                cr_trans_dict               = cr_trans_dict, 
                normalize_by_SNs            = normalize_by_SNs, 
                normalize_by_time           = normalize_by_time, 
                date_col                    = date_col, 
                xf_meter_cnt_col            = xf_meter_cnt_col, 
                events_tot_col              = events_tot_col, 
                outg_rec_nb_col             = outg_rec_nb_col,
                trsf_pole_nb_col            = trsf_pole_nb_col, 
                prem_nb_col                 = prem_nb_col, 
                serial_number_col           = serial_number_col,
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col, 
                trust_sql_grouping          = trust_sql_grouping, 
                drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
                make_all_columns_lowercase  = make_all_columns_lowercase, 
            )
            return rcpx_df
        except:
            print('CPXDfBuilder.build_rcpx_from_evsSum_df_wpreddates: _simple method failed, trying _full')
            pass
        #--------------------------------------------------
        try:
            rcpx_df = CPXDfBuilder.build_rcpx_from_evsSum_df_wpreddates_full(
                evsSum_df                   = evsSum_df, 
                pred_date_col               = pred_date_col, 
                td_min                      = td_min, 
                td_max                      = td_max,
                group_cols                  = group_cols,  
                freq                        = freq, 
                cr_trans_dict               = cr_trans_dict, 
                normalize_by_SNs            = normalize_by_SNs, 
                normalize_by_time           = normalize_by_time, 
                date_col                    = date_col, 
                xf_meter_cnt_col            = xf_meter_cnt_col, 
                events_tot_col              = events_tot_col, 
                outg_rec_nb_col             = outg_rec_nb_col,
                trsf_pole_nb_col            = trsf_pole_nb_col, 
                prem_nb_col                 = prem_nb_col, 
                serial_number_col           = serial_number_col,
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col, 
                trust_sql_grouping          = trust_sql_grouping, 
                drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
                make_all_columns_lowercase  = make_all_columns_lowercase, 
            )
            return rcpx_df
        except:
            print('CPXDfBuilder.build_rcpx_from_evsSum_df_wpreddates: _full method failed, CRASH IMMINENT!')
            assert(0)
