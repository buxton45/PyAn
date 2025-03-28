#!/usr/bin/env python

r"""
Holds CPXDf class.  See CPXDf.CPXDf for more information.
This is intended to replace both the legacy MECPODf and MECPOAn objects
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys

import pandas as pd
import numpy as np
from enum import IntEnum
import copy
from natsort import natsorted
from functools import reduce
import re
#--------------------------------------------------
from GenAn import GenAn
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
from MeterPremise import MeterPremise
from CPXDfBuilder import CPXDfBuilder
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
#----------------------------------------------------------------------------------------------------
class CPXDf:
    r"""
    CPXDf = Counts Per X DataFrame (where, X could represent outage, transformer, etc.)
    Typically, this Meter Events Counts Per X (MECPX), but could also be Id (enddeviceeventtypeid) Counts Per X (ICPX)
    """
    
    def __init__(
        self, 
        data_type
    ):
        r"""
        """
        self.__mecpx_df_raw = pd.DataFrame()
        self.cpx_df         = pd.DataFrame()
        self.cpx_df_name    = None
        self.XNs_cols       = None
        self.nXNs_cols      = None


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_acceptable_cpx_df_names():
        r"""
        Returns a list of all acceptable cpx_df_names
        """
        #-------------------------
        accptbl_names = [
            'rcpo_df_raw'               , 
            'rcpo_df_norm'              , 
            'rcpo_df_norm_by_xfmr_nSNs' , 
            'rcpo_df_norm_by_outg_nSNs' , 
            'rcpo_df_norm_by_prim_nSNs' , 

            'icpo_df_raw'               , 
            'icpo_df_norm'              , 
            'icpo_df_norm_by_xfmr_nSNs' , 
            'icpo_df_norm_by_outg_nSNs' , 
            'icpo_df_norm_by_prim_nSNs' ,  
        ]
        return accptbl_names
    #--------------------------------------------------
    @staticmethod
    def assert_acceptable_cpx_df_name(name):
        r"""
        Asserts that input name is acceptable
        """
        #-------------------------
        accptbl_names = CPXDf.get_acceptable_cpx_df_names()
        assert(name in accptbl_names)

    #--------------------------------------------------
    @staticmethod
    def std_XNs_cols():
        return CPXDfBuilder.std_XNs_cols()
        
    #--------------------------------------------------
    @staticmethod
    def std_nXNs_cols():
        return CPXDfBuilder.std_nXNs_cols()
    

    #--------------------------------------------------
    @staticmethod
    def determine_XNs_cols(
        cpx_df   , 
        XNs_cols = None
    ):
        r"""
        Determine which columns in cpx_df are XNs cols.
        If XNs_cols is None, it is set to CPXDf.std_XNs_cols().
        Then, the XNs cols returned are simply those XNs_cols found in cpx_df.columns
        """
        #-------------------------
        found_cols = CPXDfBuilder.determine_XNs_cols(
            cpx_df   = cpx_df, 
            XNs_cols = XNs_cols
        )
        return found_cols
    
    #--------------------------------------------------
    @staticmethod
    def determine_nXNs_cols(
        cpx_df    , 
        nXNs_cols = None
    ):
        r"""
        Determine which columns in cpx_df are nXNs cols.
        If nXNs_cols is None, it is set to CPXDf.std_nXNs_cols().
        Then, the nXNs cols returned are simply those nXNs_cols found in cpx_df.columns
        """
        #-------------------------
        found_cols = CPXDfBuilder.determine_nXNs_cols(
            cpx_df    = cpx_df, 
            nXNs_cols = nXNs_cols
        )
        return found_cols


    #-----------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------      

    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def add_total_event_counts_to_cpx_df(
        cpx_df                , 
        output_col            = 'total_counts', 
        XNs_tags              = None,
        non_reason_lvl_0_vals = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        assert(cpx_df.columns.nlevels<=2)
        assert(cpx_df.index.nunique()==cpx_df.shape[0]) # Assumes a unique index
        #-------------------------
        total_counts_df = CPXDfBuilder.get_total_event_counts(
            cpx_df                = cpx_df, 
            output_col            = output_col, 
            sort_output           = False, 
            XNs_tags              = XNs_tags, 
            non_reason_lvl_0_vals = non_reason_lvl_0_vals
        )
        #-------------------------
        # If the output_cols are already in cpx_df, we want to replace them.
        # If so, don't want annoying _x tags, want true replacement, so must first remove
        overlap_cols = list(set(total_counts_df.columns).intersection(set(cpx_df.columns)))
        if len(overlap_cols) > 0:
            cpx_df = cpx_df.drop(columns = overlap_cols)
        #-------------------------
        og_shape_0 = cpx_df.shape[0]
        cpx_df = pd.merge(
            cpx_df, 
            total_counts_df, 
            how         = 'inner', 
            left_index  = True, 
            right_index = True
        )
        assert(cpx_df.shape[0]==og_shape_0)
        #-------------------------
        cpx_df = cpx_df[cpx_df.columns.sort_values()]
        return cpx_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def add_delta_cpx_df_reasons(
        rcpx_df               , 
        reasons_1             ,
        reasons_2             ,
        delta_reason_name     , 
        is_norm               = False, 
        counts_col            = '_nSNs', 
        non_reason_lvl_0_vals = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        Find difference between two reasons.
          e.g., power downs minus power ups.
        If reasons_1 or reasons_2 not found in rcpx_df, they will be input with value 0
    
        NOTE: Since we're combining columns within each row, there is no need to combine list elements such as _SNs etc. as
              these are shared by all columns in the row.
    
        reasons_1, reasons_2:
          Intended to be a strings (i.e., the function will take care of level_0 values if the columns in rcpx_df
            are MultiIndex).
            However, tuples will work as well when rcpx_df has MultiIndex columns, assuming the 0th element
              of the tuples aligns with the level 0 values in rcpx_df columns
          EASIEST TO SUPPLY LIST OF STRINGS!!!!!
    
        delta_reason_name:
          Name for output column made from difference of two columns
    
        is_norm:
          Set to True of rcpx_df is normalized
          In this case, counts_col will be used to un-normalize the data before perfmoring the conversion, and then
            again the re-normalize the final data.
    
        counts_col:
          Should be a string.  If rcpx_df has MultiIndex columns, the level_0 value will be handled.
          Will still function properly if appropriate tuple/list is supplied instead of string
            e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming rcpx_df columns
                  have level_0 values equal to 'counts')
          EASIEST TO SUPPLY STRING!!!!!
          NOTE: Only really needed if is_norm==True
        """
        #----------------------------------------------------------------------------------------------------
        assert(rcpx_df.columns.nlevels<=2)
        assert(rcpx_df.index.nunique()==rcpx_df.shape[0]) # Assumes a unique index
        if(
            rcpx_df.columns.nlevels == 2 and 
            rcpx_df.columns.get_level_values(0).nunique() > 1
        ):
            # Need to break out each column level-0 value and handle separately.
            # Typically, this means there are multiple time groups contained in the rcpx_df (e.g., '01-06 Days', '06-11 Days', etc)
            #--------------------------------------------------
            col_lvl_0_vals = natsorted(rcpx_df.columns.get_level_values(0).unique())
            #-------------------------
            non_reason_lvl_0_vals = non_reason_lvl_0_vals + CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()
            #--------------------------------------------------
            pd_dfs = []
            for col_lvl_0_val_i in col_lvl_0_vals:
                rcpx_pd_i = rcpx_df[[col_lvl_0_val_i]].copy()
                #-----
                if col_lvl_0_val_i in non_reason_lvl_0_vals:
                    pd_dfs.append(rcpx_pd_i)
                    continue
                #-----
                rcpx_pd_i = CPXDf.add_delta_cpx_df_reasons(
                    rcpx_df               = rcpx_pd_i, 
                    reasons_1             = reasons_1,
                    reasons_2             = reasons_2,
                    delta_reason_name     = delta_reason_name, 
                    is_norm               = is_norm, 
                    counts_col            = counts_col, 
                    non_reason_lvl_0_vals = non_reason_lvl_0_vals, 
                )
                pd_dfs.append(rcpx_pd_i)
        
            #--------------------------------------------------
            # Make sure all dfs in pd_dfs look correct
            shape_0 = pd_dfs[0].shape[0]
            index_0 = pd_dfs[0].index
            for i in range(len(pd_dfs)):
                if i==0:
                    continue
                assert(pd_dfs[i].shape[0]==shape_0)
                assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
                #-----
                # Aligning the indices is not strictly necessary, as pd.concat should handle that
                # But, it's best to be safe
                pd_dfs[i] = pd_dfs[i].loc[index_0]
            #--------------------------------------------------
            # Build rcpx_final by combining all dfs in pd_dfs
            # rcpx_final = pd.concat(pd_dfs, axis=1)
            rcpx_final = CPXDfBuilder.merge_cpx_dfs(
                dfs_coll                      = pd_dfs, 
                max_total_counts              = None, 
                how_max_total_counts          = 'any', 
                XNs_tags                      = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols(), 
                cols_to_init_with_empty_lists = CPXDf.std_XNs_cols(), 
                make_cols_equal               = False, 
                sort_cols                     = True, 
            )
            #--------------------------------------------------
            return rcpx_final
        #----------------------------------------------------------------------------------------------------
        rcpx_df = rcpx_df.copy()
        #-------------------------
        # The code below is designed to work with ONLY single time group (handles multiple above)
        assert(rcpx_df.columns.nlevels<=2)
        # One needs to ensure that reasons_1, reasons_2, counts_col all have the correct dimensionality/type, 
        #   e.g., each should have length 2 if nlevel==2). 
        are_multiindex_cols = False
        if rcpx_df.columns.nlevels==2:
            # Again, below only for single time group
            are_multiindex_cols = True
            assert(rcpx_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpx_df.columns.get_level_values(0).unique().tolist()[0]
            #-------------------------
            assert(Utilities.is_object_one_of_types(reasons_1, [str, tuple, list]))
            if isinstance(reasons_1, str):
                reasons_1 = (level_0_val, reasons_1)
            else:
                assert(len(reasons_1)==2)
            #-------------------------
            assert(Utilities.is_object_one_of_types(reasons_2, [str, tuple, list]))
            if isinstance(reasons_2, str):
                reasons_2 = (level_0_val, reasons_2)
            else:
                assert(len(reasons_2)==2)
            #-------------------------
            assert(Utilities.is_object_one_of_types(delta_reason_name, [str, tuple, list]))
            if isinstance(delta_reason_name, str):
                delta_reason_name = (level_0_val, delta_reason_name)
            else:
                assert(len(delta_reason_name)==2)
            #-------------------------
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
        #-------------------------
        if reasons_1 not in rcpx_df.columns:
            rcpx_df[reasons_1] = 0
        if reasons_2 not in rcpx_df.columns:
            rcpx_df[reasons_2] = 0
        #-------------------------
        # Make sure reasons_1,reasons_2 are in rcpx_df
        assert(reasons_1 in rcpx_df.columns)
        assert(reasons_2 in rcpx_df.columns)
        #-------------------------
        # If rcpx_df was normalized, it must first be un-normalized so the like-columns can be combined below
        if is_norm:
            assert(counts_col in rcpx_df)
            rcpx_df[[reasons_1, reasons_2]] = rcpx_df[[reasons_1, reasons_2]].multiply(rcpx_df[counts_col], axis='index')
        #-------------------------
        rcpx_df[delta_reason_name] = rcpx_df[reasons_1]-rcpx_df[reasons_2]
        #-------------------------
        # If rcpx_df was normalized, rcpx_df must now be re-normalized
        if is_norm:
            rcpx_df[[reasons_1, reasons_2, delta_reason_name]] = rcpx_df[[reasons_1, reasons_2, delta_reason_name]].divide(rcpx_df[counts_col], axis='index')
        #-------------------------
        return rcpx_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod    
    def remove_reasons_explicit_from_rcpx_df(
        rcpx_df           , 
        reasons_to_remove , 
    ):
        r"""
        Called _explicit because exact columns to removed must be explicitly given.  
        For a more flexible version, see remove_reasons_from_rcpx_df
        
        reasons_to_remove:
          Should be a list of strings.  If a given df has MultiIndex columns, this will be handled.
          Reasons are only removed if they exist (obviously)
        """
        #-------------------------
        assert(rcpx_df.columns.nlevels<=2)
        if rcpx_df.columns.nlevels==1:
            reasons_to_remove = [x for x in reasons_to_remove if x in rcpx_df.columns]
        else:
            level_0_vals = rcpx_df.columns.get_level_values(0).unique()
            reasons_to_remove = [(level_0_val, reason) for level_0_val in level_0_vals for reason in reasons_to_remove]
            reasons_to_remove  = [x for x in reasons_to_remove if x in rcpx_df.columns]
        rcpx_df = rcpx_df.drop(columns=reasons_to_remove)
        return rcpx_df


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def remove_reasons_from_rcpx_df(
        rcpx_df                  , 
        regex_patterns_to_remove , 
        ignore_case              = True
    ):
        r"""
        Remove any columns from rcpx_df where any of the patterns in regex_patterns_to_remove are found

        regex_patterns_to_remove:
          Should be a list of regex patterns (i.e., strings)
        """
        #-------------------------
        assert(rcpx_df.columns.nlevels<=2)
        if rcpx_df.columns.nlevels==1:
            reasons = rcpx_df.columns
        else:
            reasons = rcpx_df.columns.get_level_values(1)
        reasons = reasons.tolist() #Utilities.find_in_list_with_regex wants a list input
        #-------------------------
        col_idxs_to_remove = Utilities.find_idxs_in_list_with_regex(
            lst=reasons, 
            regex_pattern=regex_patterns_to_remove, 
            ignore_case=ignore_case
        )
        cols_to_remove = rcpx_df.columns[col_idxs_to_remove]
        #-------------------------
        rcpx_df = rcpx_df.drop(columns=cols_to_remove)
        return rcpx_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def project_level_0_columns_from_rcpx_wide(
        rcpx_df     , 
        level_0_val , 
        droplevel   = False
    ):
        r"""
        This is kind of a pain to write all the time, hence this function.
        Intended to be used with a reason_counts_per_outage (rcpx) DataFrame which has both raw and normalized values.
        In this case, the columns of the DF are MultiIndex, with the level 0 values raw or normalized (typically 'counts' or
        'counts_norm'), and the level 1 values are the reasons.
        """
        #-------------------------
        assert(rcpx_df.columns.nlevels==2)
        assert(level_0_val in rcpx_df.columns.get_level_values(0).unique())
        #-------------------------
        cols_to_project = rcpx_df.columns[rcpx_df.columns.get_level_values(0)==level_0_val]
        return_df = rcpx_df[cols_to_project]
        if droplevel:
            return_df = return_df.droplevel(0, axis=1)
        return return_df
    

    @staticmethod
    def distribute_multi_column_argument(
        rcpx_df               , 
        multi_column_argument , 
    ):
        r"""
        Will be used by at least CPXDf.combine_cpx_df_reasons_explicit and CPXDf.get_reasons_subset_from_cpx_df.
        This function will be invoked when rcpx_df has multiple time groups.
        Methodology originally designed for CPXDf.combine_cpx_df_reasons_explicit, hence the change of:
            multi_column_argument   --> reasons_to_combine and 
            single_column_argument  --> combined_reason_name
    
        THIS FUNCTION HANDLES multi_column_argument only (too lazy to alter documentation)
    
        This function splits out reasons_to_X and output_cols into dict object which they can be used when running over all
          time groups in rcpx_df
    
        -------------------------
        ----- From the original CPXDf.combine_cpx_df_reasons_explicit documentation -----
        ----- To interpret for this function, sub reasons_to_combine-->reasons_to_X and combined_reason_name-->output_cols
        MORE ON reasons_to_combine/combined_reason_name for the case of rcxp_df having multiple time groups:
            Simple strings will be applied to all time groups!
            Tuples will only be applied to those specific time groups!
            e.g., suppose there are 6 time groups (e.g., '01-06 Days', '06-11 Days', '11-16 Days', '16-21 Days', '21-26 Days', '26-31 Days')
                reasons_to_combine   = [reason_1, reason_2, (time_grp_1, reason_3)]
                combined_reason_name = [combined_reason_name, (time_grp_1, combined_reason_name_1), (time_grp_6, combined_reason_name_6)] 
    
                ==> For time_grp_1: 
                    reasons_to_combine   = [reason_1, reason_2, reason_3]
                    combined_reason_name = combined_reason_name_1
                ==> For time_grp_2-5:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name     
                ==> For time_grp_6:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name_6    
        """
        #----------------------------------------------------------------------------------------------------
        # Only intended for rcpx with multiple time groups!
        assert(
            rcpx_df.columns.nlevels == 2 and 
            rcpx_df.columns.get_level_values(0).nunique() > 1
        )
        #--------------------------------------------------
        # Too lazy to replace multi_column_argument --> reasons_to_combine and output_cols  --> single_column_argument
        reasons_to_combine   = multi_column_argument
        #---------------------------------------------------------------------------
        col_lvl_0_vals = natsorted(rcpx_df.columns.get_level_values(0).unique())
        #---------------------------------------------------------------------------
        # Break apart/massage reasons_to_combine into reasons_to_combine_dict
        assert(isinstance(reasons_to_combine, list))
        assert(Utilities.are_all_list_elements_one_of_types(reasons_to_combine, [str, tuple]))
        #-------------------------
        str_reasons_to_combine  = [x for x in reasons_to_combine if isinstance(x, str)] 
        tup_reasons_to_combine  = [x for x in reasons_to_combine if isinstance(x, tuple)]
        if len(tup_reasons_to_combine) > 0:
            assert(Utilities.are_list_elements_lengths_homogeneous(tup_reasons_to_combine, length=2))
            assert(set([x[0] for x in tup_reasons_to_combine]).difference(set(col_lvl_0_vals))==set())
        #-----
        # NOTE: Without the deep copy below, the appendments following will be applied to all instead of the specific
        #       desired time group!
        reasons_to_combine_dict = {
            col_lvl_0_val_i : copy.deepcopy(str_reasons_to_combine) 
            for col_lvl_0_val_i in col_lvl_0_vals
        }
        #-----
        for col_lvl_0_val_i, reason_i in tup_reasons_to_combine:
            reasons_to_combine_dict[col_lvl_0_val_i].append(reason_i)
        #---------------------------------------------------------------------------
        return reasons_to_combine_dict
    

    #----------------------------------------------------------------------------------------------------    
    @staticmethod
    def distribute_single_column_argument(
        rcpx_df                , 
        single_column_argument , 
    ):
        r"""
        Will be used by at least CPXDf.combine_cpx_df_reasons_explicit and CPXDf.get_reasons_subset_from_cpx_df.
        This function will be invoked when rcpx_df has multiple time groups.
        Methodology originally designed for CPXDf.combine_cpx_df_reasons_explicit, hence the change of:
            multi_column_argument   --> reasons_to_combine and 
            single_column_argument  --> combined_reason_name
    
        THIS FUNCTION HANDLES multi_column_argument only (too lazy to alter documentation)
    
        This function splits out reasons_to_X and output_cols into dict object which they can be used when running over all
          time groups in rcpx_df
    
        -------------------------
        ----- From the original CPXDf.combine_cpx_df_reasons_explicit documentation -----
        ----- To interpret for this function, sub reasons_to_combine-->reasons_to_X and combined_reason_name-->output_cols
        MORE ON reasons_to_combine/combined_reason_name for the case of rcxp_df having multiple time groups:
            Simple strings will be applied to all time groups!
            Tuples will only be applied to those specific time groups!
            e.g., suppose there are 6 time groups (e.g., '01-06 Days', '06-11 Days', '11-16 Days', '16-21 Days', '21-26 Days', '26-31 Days')
                reasons_to_combine   = [reason_1, reason_2, (time_grp_1, reason_3)]
                combined_reason_name = [combined_reason_name, (time_grp_1, combined_reason_name_1), (time_grp_6, combined_reason_name_6)] 
    
                ==> For time_grp_1: 
                    reasons_to_combine   = [reason_1, reason_2, reason_3]
                    combined_reason_name = combined_reason_name_1
                ==> For time_grp_2-5:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name     
                ==> For time_grp_6:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name_6    
        """
        #----------------------------------------------------------------------------------------------------
        # Only intended for rcpx with multiple time groups!
        assert(
            rcpx_df.columns.nlevels == 2 and 
            rcpx_df.columns.get_level_values(0).nunique() > 1
        )
        #--------------------------------------------------
        # Too lazy to replace output_cols  --> combined_reason_name
        combined_reason_name = single_column_argument
        #---------------------------------------------------------------------------
        col_lvl_0_vals = natsorted(rcpx_df.columns.get_level_values(0).unique())
        #---------------------------------------------------------------------------
        # Break apart/massage combined_reason_name into combined_reason_name_dict
        assert(Utilities.is_object_one_of_types(combined_reason_name, [str, list]))
        if isinstance(combined_reason_name, list):
            assert(Utilities.are_all_list_elements_one_of_types(combined_reason_name, [str, tuple]))
        #-------------------------
        if isinstance(combined_reason_name, str):
            combined_reason_name_dict = {
                col_lvl_0_val_i : combined_reason_name 
                for col_lvl_0_val_i in col_lvl_0_vals
            }
        #-------------------------
        else:
            str_combined_reason_name  = [x for x in combined_reason_name if isinstance(x, str)] 
            # Can only be one, so if multiple found, take first
            str_combined_reason_name = str_combined_reason_name[0]
            #-----
            tup_combined_reason_name  = [x for x in combined_reason_name if isinstance(x, tuple)]
            if len(tup_combined_reason_name)>0:
                assert(Utilities.are_list_elements_lengths_homogeneous(tup_combined_reason_name, length=2))
                assert(set([x[0] for x in tup_combined_reason_name]).difference(set(col_lvl_0_vals))==set())
                combined_reason_name_dict = {
                    col_lvl_0_val_i : combined_reason_name_i for 
                    col_lvl_0_val_i , combined_reason_name_i in tup_combined_reason_name
                }
            else:
                combined_reason_name_dict = dict()
            for col_lvl_0_val_i in col_lvl_0_vals:
                if col_lvl_0_val_i not in combined_reason_name_dict.keys():
                    combined_reason_name_dict[col_lvl_0_val_i] = str_combined_reason_name
        #---------------------------------------------------------------------------
        return combined_reason_name_dict
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def distribute_reasons_to_X_and_outputs(
        rcpx_df      , 
        reasons_to_X , 
        output_cols  ,
    ):
        r"""
        Will be used by at least CPXDf.combine_cpx_df_reasons_explicit and CPXDf.get_reasons_subset_from_cpx_df.
        This function will be invoked when rcpx_df has multiple time groups.
        Methodology originally designed for CPXDf.combine_cpx_df_reasons_explicit, hence the change of:
            reasons_to_X --> reasons_to_combine and 
            output_cols  --> combined_reason_name
    
        This function splits out reasons_to_X and output_cols into dict object which they can be used when running over all
          time groups in rcpx_df
    
        -------------------------
        ----- From the original CPXDf.combine_cpx_df_reasons_explicit documentation -----
        ----- To interpret for this function, sub reasons_to_combine-->reasons_to_X and combined_reason_name-->output_cols
        MORE ON reasons_to_combine/combined_reason_name for the case of rcxp_df having multiple time groups:
            Simple strings will be applied to all time groups!
            Tuples will only be applied to those specific time groups!
            e.g., suppose there are 6 time groups (e.g., '01-06 Days', '06-11 Days', '11-16 Days', '16-21 Days', '21-26 Days', '26-31 Days')
                reasons_to_combine   = [reason_1, reason_2, (time_grp_1, reason_3)]
                combined_reason_name = [combined_reason_name, (time_grp_1, combined_reason_name_1), (time_grp_6, combined_reason_name_6)] 
    
                ==> For time_grp_1: 
                    reasons_to_combine   = [reason_1, reason_2, reason_3]
                    combined_reason_name = combined_reason_name_1
                ==> For time_grp_2-5:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name     
                ==> For time_grp_6:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name_6    
        """
        #----------------------------------------------------------------------------------------------------
        # Only intended for rcpx with multiple time groups!
        assert(
            rcpx_df.columns.nlevels == 2 and 
            rcpx_df.columns.get_level_values(0).nunique() > 1
        )
        #--------------------------------------------------
        reasons_to_combine_dict = CPXDf.distribute_multi_column_argument(
            rcpx_df               = rcpx_df, 
            multi_column_argument = reasons_to_X
        )
        #-----
        combined_reason_name_dict = CPXDf.distribute_single_column_argument(
            rcpx_df                = rcpx_df, 
            single_column_argument = output_cols
        )
        #---------------------------------------------------------------------------
        return reasons_to_combine_dict, combined_reason_name_dict
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod    
    def combine_cpx_df_reasons_explicit(
        rcpx_df               , 
        reasons_to_combine    ,
        combined_reason_name  , 
        is_norm               = False, 
        counts_col            = '_nSNs', 
        non_reason_lvl_0_vals = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        Combine multiple reason columns into a single reason.  Called _explicit because exact columns to combine must be explicitly
        given.  For a more flexible version, see combine_cpx_df_reasons
          e.g., combine reasons_to_combine = [
                    'Under Voltage (CA000400) occurred for meter', 
                    'Under Voltage (CA000400) for meter Voltage out of tolerance', 
                    'Under Voltage (Diagnostic 6) occurred for meter', 
                    'Diag6: Under Voltage, Element A occurred for meter', 
                    'Under Voltage (CA000400) occurred for meter:and C'
                ]
          into a single 'Under Voltage'
        This is intended for rcpx DFs, as icpx enddeviceeventtypeids already combine multiple reasons in general.
          However, this is no reason this can't also be used for icpx DF
        AS WITH MANY OF THE FUNCTIONS IN THIS CLASS, this was originally designed for use with s rcpx_df for a SINGLE time group, 
          but has been expanded to accomodate those with multiple time groups

        NOTE: Since we're combining columns within each row, there is no need to combine list elements such as _SNs etc. as
              these are shared by all columns in the row.

        reasons_to_combine:
          Intended to be a list of strings (i.e., the function will take care of level_0 values if the columns in rcpx_df
            are MultiIndex).
            However, a list of tuples will work as well when rcpx_df has MultiIndex columns, assuming the 0th element
              of the tuples aligns with the level 0 values in rcpx_df columns
          EASIEST TO SUPPLY LIST OF STRINGS!!!!!

        combined_reason_name:
          Name for output column made from combined columns

        MORE ON reasons_to_combine/combined_reason_name for the case of rcxp_df having multiple time groups:
            Simple strings will be applied to all time groups!
            Tuples will only be applied to those specific time groups!
            e.g., suppose there are 6 time groups (e.g., '01-06 Days', '06-11 Days', '11-16 Days', '16-21 Days', '21-26 Days', '26-31 Days')
                reasons_to_combine   = [reason_1, reason_2, (time_grp_1, reason_3)]
                combined_reason_name = [combined_reason_name, (time_grp_1, combined_reason_name_1), (time_grp_6, combined_reason_name_6)] 

                ==> For time_grp_1: 
                    reasons_to_combine   = [reason_1, reason_2, reason_3]
                    combined_reason_name = combined_reason_name_1
                ==> For time_grp_2-5:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name     
                ==> For time_grp_6:
                    reasons_to_combine   = [reason_1, reason_2]
                    combined_reason_name = combined_reason_name_6    

        is_norm:
          Set to True of rcpx_df is normalized
          In this case, counts_col will be used to un-normalize the data before perfmoring the conversion, and then
            again the re-normalize the final data.

        counts_col:
          Should be a string.  If rcpx_df has MultiIndex columns, the level_0 value will be handled.
          Will still function properly if appropriate tuple/list is supplied instead of string
            e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming rcpx_df columns
                  have level_0 values equal to 'counts')
          EASIEST TO SUPPLY STRING!!!!!
          NOTE: Only really needed if is_norm==True
        """
        #----------------------------------------------------------------------------------------------------
        assert(rcpx_df.columns.nlevels<=2)
        if(
            rcpx_df.columns.nlevels == 2 and 
            rcpx_df.columns.get_level_values(0).nunique() > 1
        ):
            # Need to break out each column level-0 value and handle separately.
            # Typically, this means there are multiple time groups contained in the rcpx_df (e.g., '01-06 Days', '06-11 Days', etc)
            #---------------------------------------------------------------------------
            col_lvl_0_vals = natsorted(rcpx_df.columns.get_level_values(0).unique())
            #-------------------------
            non_reason_lvl_0_vals = non_reason_lvl_0_vals + CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()
            #---------------------------------------------------------------------------
            # Break apart/massage reasons_to_combine and combined_reason_name into reasons_to_combine_dict and combined_reason_name_dict
            reasons_to_combine_dict, combined_reason_name_dict = CPXDf.distribute_reasons_to_X_and_outputs(
                rcpx_df      = rcpx_df, 
                reasons_to_X = reasons_to_combine, 
                output_cols  = combined_reason_name
            )
            #--------------------------------------------------
            pd_dfs = []
            for col_lvl_0_val_i in col_lvl_0_vals:
                rcpx_pd_i = rcpx_df[[col_lvl_0_val_i]].copy()
                #-----
                if col_lvl_0_val_i in non_reason_lvl_0_vals:
                    pd_dfs.append(rcpx_pd_i)
                    continue
                #-----
                reasons_to_combine_i = reasons_to_combine_dict[col_lvl_0_val_i]
                reasons_to_combine_i = list(set(reasons_to_combine_i).intersection(set(rcpx_pd_i.columns.get_level_values(1))))
                #-----
                if len(reasons_to_combine_i) >= 2:
                    rcpx_pd_i = CPXDf.combine_cpx_df_reasons_explicit(
                        rcpx_df               = rcpx_pd_i, 
                        reasons_to_combine    = reasons_to_combine_i,
                        combined_reason_name  = combined_reason_name_dict[col_lvl_0_val_i], 
                        is_norm               = is_norm, 
                        counts_col            = counts_col, 
                        non_reason_lvl_0_vals = non_reason_lvl_0_vals, 
                    )
                pd_dfs.append(rcpx_pd_i)
        
            #--------------------------------------------------
            # Make sure all dfs in pd_dfs look correct
            shape_0 = pd_dfs[0].shape[0]
            index_0 = pd_dfs[0].index
            for i in range(len(pd_dfs)):
                if i==0:
                    continue
                assert(pd_dfs[i].shape[0]==shape_0)
                assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
                #-----
                # Aligning the indices is not strictly necessary, as pd.concat should handle that
                # But, it's best to be safe
                pd_dfs[i] = pd_dfs[i].loc[index_0]
            #--------------------------------------------------
            # Build rcpx_final by combining all dfs in pd_dfs
            # rcpx_final = pd.concat(pd_dfs, axis=1)
            rcpx_final = CPXDfBuilder.merge_cpx_dfs(
                dfs_coll                      = pd_dfs, 
                max_total_counts              = None, 
                how_max_total_counts          = 'any', 
                XNs_tags                      = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols(), 
                cols_to_init_with_empty_lists = CPXDf.std_XNs_cols(), 
                make_cols_equal               = False, 
                sort_cols                     = True, 
            )
            #--------------------------------------------------
            return rcpx_final

        #----------------------------------------------------------------------------------------------------
        # Grab the original number of columns for comparison later
        n_cols_OG = rcpx_df.shape[1]
        #-------------------------
        rcpx_df = rcpx_df.copy()
        #-------------------------
        assert(rcpx_df.columns.nlevels<=2)
        # One needs to ensure that reasons_to_combine has the correct dimensionality/type, 
        #   e.g., each element should have length 2 if nlevel==2). 
        #   The same is true for counts_col.
        are_multiindex_cols = False
        if rcpx_df.columns.nlevels==2:
            are_multiindex_cols = True
            assert(rcpx_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpx_df.columns.get_level_values(0).unique().tolist()[0]
            #-------------------------
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2 and counts_col[0]==level_0_val)
            #-------------------------
            assert(Utilities.is_object_one_of_types(combined_reason_name, [str, tuple, list]))
            if isinstance(combined_reason_name, str):
                combined_reason_name = (level_0_val, combined_reason_name)
            else:
                assert(len(combined_reason_name)==2 and combined_reason_name[0]==level_0_val)
            #-------------------------
            # reasons_to_combine should contain elements which are strings, tuples, or lists.
            # If strings, since rcpx_df.columns.nlevels==2, these need to be converted to a lists of tuples
            # If tuples/lists, need to make sure all have length==2
            assert(Utilities.are_all_list_elements_one_of_types_and_homogeneous(reasons_to_combine, [str, tuple, list]))
            if isinstance(reasons_to_combine[0], str):
                reasons_to_combine = pd.MultiIndex.from_product([[level_0_val], reasons_to_combine])
            else:
                assert(Utilities.are_list_elements_lengths_homogeneous(reasons_to_combine, length=2))
                assert(np.all([x[0]==level_0_val for x in reasons_to_combine]))

        #-------------------------
        # Make sure all reasons_to_combine are in rcpx_df
        assert(all([x in rcpx_df.columns for x in reasons_to_combine]))
        #-------------------------
        # If rcpx_df was normalized, it must first be un-normalized so the like-columns can be combined below
        if is_norm:
            assert(counts_col in rcpx_df)
            rcpx_df[reasons_to_combine] = rcpx_df[reasons_to_combine].multiply(rcpx_df[counts_col], axis='index')
        #-------------------------
        # Rename all columns in reasons_to_combine as combined_reason_name to create degenerate column names 
        #   (i.e., multiple columns of same name).  This is by design, and will be useful in the combination step
        rename_cols_dict = {x:combined_reason_name for x in reasons_to_combine}
        # Apparently, rename cannot be used to rename multiple levels at the same time
        if are_multiindex_cols:
            rcpx_df=rcpx_df.rename(columns={k[1]:v[1] for k,v in rename_cols_dict.items()}, level=1)
        else:
            rcpx_df=rcpx_df.rename(columns=rename_cols_dict)
        #-------------------------
        # Finally, combine like columns
        #   OLD VERSION OF CODE WILL NO LONGER BE SUPPORTED
        #       FutureWarning: DataFrame.groupby with axis=1 is deprecated. Do `frame.T.groupby(...)` without axis instead.
        #   OLD VERSION: rcpx_df = rcpx_df.groupby(rcpx_df.columns, axis=1).sum()
        #   NEW VERSION: rcpx_df = rcpx_df.T.groupby(rcpx_df.columns).sum().T
        rcpx_df = rcpx_df.T.groupby(rcpx_df.columns).sum().T

        # If the columns were MultiIndex, the groupby.sum procedure collapses these back down to a
        #   single dimension of tuples.  Therefore, expand back out
        if are_multiindex_cols:
            rcpx_df.columns=pd.MultiIndex.from_tuples(rcpx_df.columns)
        #-------------------------
        # If rcpx_df was normalized, rcpx_df must now be re-normalized
        if is_norm:
            adjusted_cols = list(set(rename_cols_dict.values()))
            if are_multiindex_cols:
                adjusted_cols = [(level_0_val,x) for x in adjusted_cols]
            # Maintain order in df, I suppose...
            adjusted_cols = [x for x in rcpx_df.columns if x in adjusted_cols]
            #-----
            rcpx_df[adjusted_cols] = rcpx_df[adjusted_cols].divide(rcpx_df[counts_col], axis='index')
        #-------------------------
        # Make sure the final number of columns makes sense.
        # len(reasons_to_combine) columns are replaced by 1, for a net of len(reasons_to_combine)-1
        assert(n_cols_OG-rcpx_df.shape[1]==len(reasons_to_combine)-1)
        #-------------------------
        return rcpx_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def combine_cpx_df_reasons(
        rcpx_df                     , 
        patterns_and_replace        = None, 
        addtnl_patterns_and_replace = None, 
        is_norm                     = False, 
        counts_col                  = '_nSNs', 
        initial_strip               = True,
        initial_punctuation_removal = True, 
        return_red_to_org_cols_dict = False, 
        non_reason_lvl_0_vals       = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        Combine groups of reasons according to patterns_and_replace.
    
        NOTE!!!!!!!!!!!!!:
          Typically, one should keep patterns_and_replace=None.  When this is the case, dflt_patterns_and_replace
            will be used.
          If one wants to add to dflt_patterns_and_replace, use the addtnl_patterns_and_replace argument.
    
        patterns_and_replace/addtnl_patterns_and_replace:
          A list of tuples (or lists) of length 2.
          Typical value = ['.*cleared.*', '.*Test Mode.*']
          For each item in the list:
            first element should be a regex pattern for which to search 
            second element is replacement
    
            DON'T FORGET ABOUT BACKREFERENCING!!!!!
              e.g.
                reason_i = 'I am a string named Jesse Thomas Buxton'
                patterns_and_replace_i = (r'(Jesse) (Thomas) (Buxton)', r'\1 \3')
                  reason_i ===> 'I am a string named Jesse Buxton'
    
        is_norm:
          Set to True of rcpx_df is normalized
          In this case, counts_col will be used to un-normalize the data before perfmoring the conversion, and then
            again the re-normalize the final data.
    
        counts_col:
          Should be a string.  If rcpx_df has MultiIndex columns, the level_0 value will be handled.
          NOTE: Only really needed if is_norm==True
        """
        #----------------------------------------------------------------------------------------------------
        assert(rcpx_df.columns.nlevels<=2)
        if(
            rcpx_df.columns.nlevels == 2 and 
            rcpx_df.columns.get_level_values(0).nunique() > 1
        ):
            # Need to break out each column level-0 value and handle separately.
            # Typically, this means there are multiple time groups contained in the rcpx_df (e.g., '01-06 Days', '06-11 Days', etc)
            #--------------------------------------------------
            col_lvl_0_vals = natsorted(rcpx_df.columns.get_level_values(0).unique())
            #-------------------------
            non_reason_lvl_0_vals = non_reason_lvl_0_vals + CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()
            #--------------------------------------------------
            pd_dfs = []
            for col_lvl_0_val_i in col_lvl_0_vals:
                rcpx_pd_i = rcpx_df[[col_lvl_0_val_i]].copy()
                #-----
                if col_lvl_0_val_i in non_reason_lvl_0_vals:
                    pd_dfs.append(rcpx_pd_i)
                    continue
                #-----
                rcpx_pd_i = CPXDf.combine_cpx_df_reasons(
                    rcpx_df                     = rcpx_pd_i, 
                    patterns_and_replace        = patterns_and_replace, 
                    addtnl_patterns_and_replace = addtnl_patterns_and_replace, 
                    is_norm                     = is_norm, 
                    counts_col                  = counts_col, 
                    initial_strip               = initial_strip,
                    initial_punctuation_removal = initial_punctuation_removal, 
                    return_red_to_org_cols_dict = return_red_to_org_cols_dict, 
                    non_reason_lvl_0_vals       = non_reason_lvl_0_vals
                )
                pd_dfs.append(rcpx_pd_i)
        
            #--------------------------------------------------
            # Make sure all dfs in pd_dfs look correct
            shape_0 = pd_dfs[0].shape[0]
            index_0 = pd_dfs[0].index
            for i in range(len(pd_dfs)):
                if i==0:
                    continue
                assert(pd_dfs[i].shape[0]==shape_0)
                assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
                #-----
                # Aligning the indices is not strictly necessary, as pd.concat should handle that
                # But, it's best to be safe
                pd_dfs[i] = pd_dfs[i].loc[index_0]
            #--------------------------------------------------
            # Build rcpx_final by combining all dfs in pd_dfs
            # rcpx_final = pd.concat(pd_dfs, axis=1)
            rcpx_final = CPXDfBuilder.merge_cpx_dfs(
                dfs_coll                      = pd_dfs, 
                max_total_counts              = None, 
                how_max_total_counts          = 'any', 
                XNs_tags                      = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols(), 
                cols_to_init_with_empty_lists = CPXDf.std_XNs_cols(), 
                make_cols_equal               = False, 
                sort_cols                     = True, 
            )
            #--------------------------------------------------
            return rcpx_final
    
    
        #----------------------------------------------------------------------------------------------------
        dflt_patterns_and_replace = [
            (r'(?:A\s*)?(NET_MGMT command) was sent with a key that (has insufficient privileges).*', r'\1 \2'), 
            (r'(Ignoring).*(Read) data for device as it has (time in the future)', r'\1 \2: \3'), 
            (r'(Device Failed).*', r'\1'), 
            (r'(Last Gasp - NIC power lost for device).*', 'Last Gasp'), 
            (r'(Last Gasp State).*', 'Last Gasp'), 
            (r'(Meter needs explicit time sync).*', r'\1'), 
            (r'(NET_MGMT command failed consecutively).*', r'\1'), 
            (r'(NIC Link Layer Handshake Failed).*', r'\1'), 
            (r'(NVRAM Error).*', r'\1'), 
            (r'(Requested operation).*(could not be applied).*', r'\1 \2'), 
            (r'(Secure association operation failed consecutively).*', r'\1'), 
            (r'Low Potential.*occurred for meter', 'Under Voltage for meter'),
            (r'.*(Over Voltage).*', r'\1'), 
            (r'.*(Under Voltage).*', r'\1'), 
            (
                (
                    r'((Meter detected tampering\s*(?:\(.*\))?\s*).*)|'\
                    r'((Tamper)\s*(?:\(.*\))?\s*detected).*|'\
                    r'(Meter event Tamper Attempt Suspected).*|'\
                    r'Meter detected a Tamper Attempt'
                ), 
                'Tamper Detected'
            ), 
            (r'Low\s*(?:Battery|Potential)\s*(?:\(.*\))?\s*.*', 'Low Battery')
        ]
        #-------------------------
        if patterns_and_replace is None:
            patterns_and_replace = dflt_patterns_and_replace
        #-----
        if addtnl_patterns_and_replace is not None:
            patterns_and_replace.extend(addtnl_patterns_and_replace)
        #-------------------------
        rcpx_df = rcpx_df.copy()
        #-------------------------
        assert(rcpx_df.columns.nlevels<=2)
        are_multiindex_cols = False
        # If rcpx_df has MultiIndex columns, there must only be a single unique value for level 0
        # In such a case, it is easiest for me to simply strip the level 0 value, perform my operations,
        #   and add it back at the end
        if rcpx_df.columns.nlevels==2:
            are_multiindex_cols = True
            # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
            assert(rcpx_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpx_df.columns.get_level_values(0).unique().tolist()[0]
            rcpx_df = CPXDf.project_level_0_columns_from_rcpx_wide(
                rcpx_df     = rcpx_df, 
                level_0_val = level_0_val, 
                droplevel   = True
            )
        #-------------------------
        # The operation: df[some_cols] = df[some_cols].multiple(df[a_col], axis='index')
        #   cannot be performed when there is degeneracy in some_cols (i.e., there are repeated names in some_cols)
        #   This is a necessary step when is_norm=True, as the values must be un-normalized, combined, and then re-normalized
        #   Therefore, one cannot simply change column names on-the-fly, as was done in initial development.
        #     Thus, the code is a bit more cumbersome than seems necessary
        #-------------------------
        columns_org = rcpx_df.columns.tolist()
        #-------------------------
        if initial_punctuation_removal:
            reduce_cols_dict_0 = {x:AMIEndEvents.remove_trailing_punctuation_from_reason(reason=x, include_normal_strip=initial_strip) 
                                  for x in columns_org}
            reduce_cols_dict = {org_col:AMIEndEvents.reduce_end_event_reason(reason=red_col_0, patterns=patterns_and_replace, verbose=False) 
                                for org_col,red_col_0 in reduce_cols_dict_0.items()}
        else:
            reduce_cols_dict = {x:AMIEndEvents.reduce_end_event_reason(reason=x, patterns=patterns_and_replace, verbose=False) for x in columns_org}
    
        # Only keep columns in reduce_cols_dict which will be changed
        reduce_cols_dict = {k:v for k,v in reduce_cols_dict.items() if k!=v}
        #-------------------------
        # If rcpx_df was normalized, it must first be un-normalized so the like-columns can be combined below
        if is_norm:
            assert(counts_col in rcpx_df)
            cols_to_adjust = list(reduce_cols_dict.keys())
            rcpx_df[cols_to_adjust] = rcpx_df[cols_to_adjust].multiply(rcpx_df[counts_col], axis='index')
        #-------------------------
        # Rename columns according to reduce_cols_dict
        # This will create degenerate column names (i.e., multiple columns of same name)
        # This is by design, and will be useful in the combination step
        rcpx_df = rcpx_df.rename(columns=reduce_cols_dict)
        #-------------------------
        # Finally, combine like columns
        rcpx_df = CPXDfBuilder.combine_degenerate_columns(rcpx_df = rcpx_df)
        #-------------------------
        # If rcpx_df was normalized originally, it must now be re-normalized
        if is_norm:
            adjusted_cols = list(set(reduce_cols_dict.values()))
            rcpx_df[adjusted_cols] = rcpx_df[adjusted_cols].divide(rcpx_df[counts_col], axis='index')
        #-------------------------
        # If the columns of rcpx_df were originally MultiIndex, make them so again
        if are_multiindex_cols:
            rcpx_df = Utilities_df.prepend_level_to_MultiIndex(
                df=rcpx_df, 
                level_val=level_0_val, 
                level_name=None, 
                axis=1
            )
        #-------------------------
        if return_red_to_org_cols_dict:
            # Want a collection of all org_cols which were grouped into each red_col
            # So, need something similar to an inverse of reduce_cols_dict (not exactly the inverse)
            red_to_org_cols_dict = {}
            for org_col, red_col in reduce_cols_dict.items():
                if red_col in red_to_org_cols_dict.keys():
                    red_to_org_cols_dict[red_col].append(org_col)
                else:
                    red_to_org_cols_dict[red_col] = [org_col]
            return rcpx_df, red_to_org_cols_dict
        #-------------------------
        return rcpx_df


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def refine_rcpx_df(
        rcpx_df                       , 
        #-----
        regex_patterns_to_remove      = ['.*cleared.*', '.*Test Mode.*'], 
        ignore_case                   = True, 
        #-----
        add_total_counts              = True, 
        total_counts_col              = 'total_counts', 
        XNs_tags                      = None, 
        #-----
        combine_cpx_df_reasons        = True, 
        addtnl_patterns_and_replace   = None, 
        counts_col                    = '_nSNs', 
        is_norm                       = False, 
        #-----
        include_power_down_minus_up   = False, 
        non_reason_lvl_0_vals         = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        """
        #--------------------------------------------------
        # Remove any undesired curated reasons (e.g., ['.*cleared.*', '.*Test Mode.*'])
        if regex_patterns_to_remove is not None:
            rcpx_df = CPXDf.remove_reasons_from_rcpx_df(
                rcpx_df                  = rcpx_df, 
                regex_patterns_to_remove = regex_patterns_to_remove, 
                ignore_case              = ignore_case
            )
        
        #--------------------------------------------------
        # Add total counts columns
        if add_total_counts:
            rcpx_df = CPXDf.add_total_event_counts_to_cpx_df(
                cpx_df      = rcpx_df, 
                output_col  = total_counts_col, 
                XNs_tags    = XNs_tags,
            )
        
        #--------------------------------------------------
        # Combine similar reasons (e.g., all 'Tamper' type reasons are combined into 1)
        # See CPXDf.combine_cpx_df_reasons for more information
        if combine_cpx_df_reasons:
            rcpx_df = CPXDf.combine_cpx_df_reasons(
                rcpx_df                     = rcpx_df, 
                patterns_and_replace        = None, 
                addtnl_patterns_and_replace = addtnl_patterns_and_replace, 
                is_norm                     = is_norm, 
                counts_col                  = counts_col, 
                initial_strip               = True,
                initial_punctuation_removal = True, 
                return_red_to_org_cols_dict = False, 
                non_reason_lvl_0_vals       = non_reason_lvl_0_vals
            )
        
        #--------------------------------------------------
        # Include the difference in power-up and power-down, if desired (typically turned off) 
        if include_power_down_minus_up:
            rcpx_df = CPXDf.add_delta_cpx_df_reasons(
                rcpx_df               = rcpx_df, 
                reasons_1             = 'Primary Power Down',
                reasons_2             = 'Primary Power Up',
                delta_reason_name     = 'Power Down Minus Up', 
                is_norm               = is_norm, 
                counts_col            = counts_col, 
                non_reason_lvl_0_vals = non_reason_lvl_0_vals, 
            )
    
        #--------------------------------------------------
        return rcpx_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def apply_lower_to_tuples(
        tpl_or_lot
    ):
        r"""
        Apply the .lower() function to each element in a tuple of strings (or a list of tuples containing strings)
        Specialty function to help out with CPXDf.get_reasons_subset_from_cpx_df
        
        tpl_or_lot:
            A single tuple or a list of tuples
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(tpl_or_lot, [tuple, list]))
        #-------------------------
        if isinstance(tpl_or_lot, list):
            return_list = [CPXDf.apply_lower_to_tuples(x) for x in tpl_or_lot]
            return return_list
        #-------------------------
        assert(isinstance(tpl_or_lot, tuple))
        assert(Utilities.are_all_list_elements_of_type(lst=list(tpl_or_lot), typ=str))
        return tuple([x.lower() for x in tpl_or_lot])


    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_reasons_subset_from_cpx_df(
        cpx_df                       , 
        reasons_to_include           ,
        combine_others               = True, 
        output_combine_others_col    = 'Other Reasons', 
        XNs_tags                     = None, 
        is_norm                      = False, 
        counts_col                   = '_nSNs', 
        cols_to_ignore               = None, 
        include_counts_col_in_output = False, 
        non_reason_lvl_0_vals        = ['EEMSP_0', 'time_info_0'], 
        verbose                      = True
    ):
        r"""
        Project out the top reasons_to_include reasons from cpx_df.
        The order is taken to be:
          reason_order = cpx_df.mean().sort_values(ascending=False).index.tolist()

        reasons_to_include:
          A list of strings representing the columns to be included.  If cpx_df has MultiIndex columns, the 
            level_0 value will be handled.
            Will still function properly if appropriate tuples/lists are supplied instead of strings

        output_combine_others_col:
          The output column name for the combined others. 
          This should be a string, even if cpx_df has MultiIndex columns (such a case will be handled below!)

        MORE ON reasons_to_include/output_combine_others_col/cols_to_ignore for the case of rcxp_df having multiple time groups:
            Simple strings will be applied to all time groups!
            Tuples will only be applied to those specific time groups!
            NOTE: cols_to_ignore will be treated like reasons_to_include
            e.g., suppose there are 6 time groups (e.g., '01-06 Days', '06-11 Days', '11-16 Days', '16-21 Days', '21-26 Days', '26-31 Days')
                reasons_to_include        = [reason_1, reason_2, (time_grp_1, reason_3)]
                output_combine_others_col = [output_combine_others_col, (time_grp_1, output_combine_others_col_1), (time_grp_6, output_combine_others_col_6)] 
    
                ==> For time_grp_1: 
                    reasons_to_include        = [reason_1, reason_2, reason_3]
                    output_combine_others_col = output_combine_others_col_1
                ==> For time_grp_2-5:
                    reasons_to_include        = [reason_1, reason_2]
                    output_combine_others_col = output_combine_others_col     
                ==> For time_grp_6:
                    reasons_to_include        = [reason_1, reason_2]
                    output_combine_others_col = output_combine_others_col_6    

        XNs_tags:
          Defaults to CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols() when XNs_tags is None
        NOTE: XNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled in CPXDfBuilder.find_XNs_cols_idxs_from_cpx_df.
              
        cols_to_ignore:
            Any columns to ignore.
            These will be left out of subset operations, but will be included in the output

        --------------------
        THE FOLLOWING ARE ONLY NEEDED WHEN combine_others==True
            is_norm
            counts_col

            counts_col:
              Should be a string.  If cpx_df has MultiIndex columns, the level_0 value will be handled.
              Will still function properly if appropriate tuple/list is supplied instead of string
                e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming cpx_df columns
                      have level_0 values equal to 'counts')
              EASIEST TO SUPPLY STRING!!!!!
              NOTE: Only really needed if is_norm==True
        """
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        assert(cpx_df.columns.nlevels<=2)
        if(
            cpx_df.columns.nlevels == 2 and 
            cpx_df.columns.get_level_values(0).nunique() > 1
        ):
            # Need to break out each column level-0 value and handle separately.
            # Typically, this means there are multiple time groups contained in the cpx_df (e.g., '01-06 Days', '06-11 Days', etc)
            #---------------------------------------------------------------------------
            col_lvl_0_vals = natsorted(cpx_df.columns.get_level_values(0).unique())
            #-------------------------
            non_reason_lvl_0_vals = non_reason_lvl_0_vals + CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()
            #---------------------------------------------------------------------------
            # Break apart/massage reasons_to_include and output_combine_others_col into reasons_to_include_dict and output_combine_others_col_dict
            reasons_to_include_dict, output_combine_others_col_dict = CPXDf.distribute_reasons_to_X_and_outputs(
                rcpx_df      = cpx_df, 
                reasons_to_X = reasons_to_include, 
                output_cols  = output_combine_others_col
            )
            cols_to_ignore_dict = CPXDf.distribute_multi_column_argument(
                rcpx_df               = cpx_df, 
                multi_column_argument = cols_to_ignore
            )
            #--------------------------------------------------
            pd_dfs = []
            for col_lvl_0_val_i in col_lvl_0_vals:
                rcpx_pd_i = cpx_df[[col_lvl_0_val_i]].copy()
                #-----
                if col_lvl_0_val_i in non_reason_lvl_0_vals:
                    pd_dfs.append(rcpx_pd_i)
                    continue
                #-----
                reasons_to_include_i = reasons_to_include_dict[col_lvl_0_val_i]
                reasons_to_include_i = list(set(reasons_to_include_i).intersection(set(rcpx_pd_i.columns.get_level_values(1))))
                #-----
                if len(reasons_to_include_i) >= 0:
                    rcpx_pd_i = CPXDf.get_reasons_subset_from_cpx_df(
                        cpx_df                       = rcpx_pd_i, 
                        reasons_to_include           = reasons_to_include_i,
                        combine_others               = combine_others, 
                        output_combine_others_col    = output_combine_others_col_dict[col_lvl_0_val_i], 
                        XNs_tags                     = XNs_tags, 
                        is_norm                      = is_norm, 
                        counts_col                   = counts_col, 
                        cols_to_ignore               = cols_to_ignore_dict[col_lvl_0_val_i], 
                        include_counts_col_in_output = include_counts_col_in_output, 
                        non_reason_lvl_0_vals        = non_reason_lvl_0_vals, 
                        verbose                      = verbose
                    )
                pd_dfs.append(rcpx_pd_i)
        
            #--------------------------------------------------
            # Make sure all dfs in pd_dfs look correct
            shape_0 = pd_dfs[0].shape[0]
            index_0 = pd_dfs[0].index
            for i in range(len(pd_dfs)):
                if i==0:
                    continue
                assert(pd_dfs[i].shape[0]==shape_0)
                assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
                #-----
                # Aligning the indices is not strictly necessary, as pd.concat should handle that
                # But, it's best to be safe
                pd_dfs[i] = pd_dfs[i].loc[index_0]
            #--------------------------------------------------
            # Build rcpx_final by combining all dfs in pd_dfs
            # rcpx_final = pd.concat(pd_dfs, axis=1)
            rcpx_final = CPXDfBuilder.merge_cpx_dfs(
                dfs_coll                      = pd_dfs, 
                max_total_counts              = None, 
                how_max_total_counts          = 'any', 
                XNs_tags                      = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols(), 
                cols_to_init_with_empty_lists = CPXDf.std_XNs_cols(), 
                make_cols_equal               = False, 
                sort_cols                     = True, 
            )
            #--------------------------------------------------
            return rcpx_final
        
        #----------------------------------------------------------------------------------------------------
        # NOTE: reasons_to_include is potentially changed in this function.  Solution is to copy it first
        reasons_to_include = copy.deepcopy(reasons_to_include)
        #-----
        if cols_to_ignore is not None:
            assert(isinstance(cols_to_ignore, list))
            cols_to_ignore = copy.deepcopy(cols_to_ignore)
        #-------------------------
        # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
        assert(cpx_df.columns.nlevels<=2)
        if cpx_df.columns.nlevels==2:
            #-----
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                assert(cpx_df.columns.get_level_values(0).nunique()==1)
                level_0_val = cpx_df.columns.get_level_values(0).unique().tolist()[0]
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
            #-----
            assert(Utilities.is_object_one_of_types(output_combine_others_col, [str, tuple, list]))
            if isinstance(output_combine_others_col, str):
                #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                assert(cpx_df.columns.get_level_values(0).nunique()==1)
                level_0_val = cpx_df.columns.get_level_values(0).unique().tolist()[0]
                output_combine_others_col = (level_0_val, output_combine_others_col)
            else:
                assert(len(output_combine_others_col)==2)
            #-----
            for i_reason in range(len(reasons_to_include)):
                reason_i = reasons_to_include[i_reason]
                assert(Utilities.is_object_one_of_types(reason_i, [str, tuple, list]))
                if isinstance(reason_i, str):
                    #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                    assert(cpx_df.columns.get_level_values(0).nunique()==1)
                    level_0_val = cpx_df.columns.get_level_values(0).unique().tolist()[0]
                    reasons_to_include[i_reason] = (level_0_val, reason_i)
                else:
                    assert(len(reason_i)==2)
                assert(reasons_to_include[i_reason] in cpx_df.columns)
            #-----
            if cols_to_ignore is not None:
                for i_col in range(len(cols_to_ignore)):
                    col_i = cols_to_ignore[i_col]
                    assert(Utilities.is_object_one_of_types(col_i, [str, tuple, list]))
                    if isinstance(col_i, str):
                        #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                        assert(cpx_df.columns.get_level_values(0).nunique()==1)
                        level_0_val = cpx_df.columns.get_level_values(0).unique().tolist()[0]
                        cols_to_ignore[i_col] = (level_0_val, col_i)
                    else:
                        assert(len(col_i)==2)
                    assert(cols_to_ignore[i_col] in cpx_df.columns)            
        #-------------------------
        if cols_to_ignore is not None:
            # Make sure some reasons_to_include aren't accidentally included in cols_to_ignore!
            cols_to_ignore = list(set(cols_to_ignore).difference(set(reasons_to_include)))
            if len(cols_to_ignore)>0:
                cols_to_ignore_df = cpx_df[cols_to_ignore].copy()
                cpx_df = cpx_df.drop(columns=cols_to_ignore)
        
        #-------------------------
        if combine_others and is_norm:
            # In this case, counts_col will be needed for combine
            assert(counts_col in cpx_df.columns)
        #-------------------------
        non_SNs_cols = CPXDfBuilder.get_non_XNs_cols_from_cpx_df(
            cpx_df   = cpx_df, 
            XNs_tags = XNs_tags
        )
        # The counts_col is typically not included in non_SNs_cols above.  However, this is not always the case (as, 
        #   e.g., when sent from MECPOCollection.get_reasons_subset_from_merged_cpx_df it will have a random name which
        #   will not be caught by CPXDfBuilder.get_non_XNs_cols_from_cpx_df).
        # It is important that it not be included
        if counts_col in non_SNs_cols:
            non_SNs_cols.remove(counts_col)

        # Sometimes, output_combine_others_col can sneak into reasons_to_include (e.g., when a reasons_to_include is taken from a 
        #   data_structure_df, which is routinely done in modeling/predicting)
        # This is almost certainly a mistake, so remove it from reasons_to_include
        #-----
        # If output_combine_others_col found in reasons_to_include, remove it!
        # NOTE: This is done on a CASE-INSENSITIVE basis!
        if CPXDf.apply_lower_to_tuples(output_combine_others_col) in CPXDf.apply_lower_to_tuples(reasons_to_include):
            if verbose:
                print(f"In CPXDf.get_reasons_subset_from_cpx_df, found \n\toutput_combine_others_col = {output_combine_others_col} in \n\treasons_to_include = {reasons_to_include} ")
            reasons_to_include = [x for x in reasons_to_include if CPXDf.apply_lower_to_tuples(x) != CPXDf.apply_lower_to_tuples(output_combine_others_col)]

        # Sanity checks
        assert(set(reasons_to_include).difference(set(cpx_df.columns.tolist()))==set())
        assert(set(reasons_to_include).difference(set(non_SNs_cols))==set())

        # Determine the other_reasons
        other_reasons = [x for x in non_SNs_cols if x not in reasons_to_include]
        #-------------------------
        output_cols = reasons_to_include
        #-----
        if combine_others:
            cpx_df = CPXDf.combine_cpx_df_reasons_explicit(
                rcpx_df                    = cpx_df, 
                reasons_to_combine         = other_reasons,
                combined_reason_name       = output_combine_others_col, 
                is_norm                    = is_norm, 
                counts_col                 = counts_col,  
                non_reason_lvl_0_vals      = non_reason_lvl_0_vals, 
            )
            #-------------------------
            if cpx_df.columns.nlevels==2 and isinstance(output_combine_others_col, str):
                combine_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(
                    df=cpx_df, 
                    col=output_combine_others_col
                )
                assert(len(combine_col_idx)==1)
                combine_col_idx = combine_col_idx[0]
                output_combine_others_col = cpx_df.columns[combine_col_idx]
            output_cols.append(output_combine_others_col)
        #-----
        if include_counts_col_in_output:
            if counts_col in cpx_df.columns:
                output_cols.append(counts_col)
            else:
                print(f'WARNING: include_counts_col_in_output==True but counts_col={counts_col} not found in cpx_df!')
        #-----
        cpx_df = cpx_df[output_cols]
        #-------------------------
        if cols_to_ignore is not None and len(cols_to_ignore)>0:
            assert(cpx_df.shape[0]==cols_to_ignore_df.shape[0])
            assert(cpx_df.index.equals(cols_to_ignore_df.index))
            cpx_df = pd.concat([cpx_df, cols_to_ignore_df], axis=1)
        #-------------------------
        return cpx_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_top_reasons_subset_from_cpx_df(
        cpx_df                    , 
        n_reasons_to_include      = 10,
        combine_others            = True, 
        output_combine_others_col = 'Other Reasons', 
        XNs_tags                  = None, 
        is_norm                   = False, 
        counts_col                = '_nSNs', 
        total_counts_col          = 'total_counts', 
        non_reason_lvl_0_vals     = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        Project out the top n_reasons_to_include reasons from cpx_df.
        The order is taken to be:
          reason_order = cpx_df.mean().sort_values(ascending=False).index.tolist()

        output_combine_others_col:
          The output column name for the combined others. 
          This should be a string, even if cpx_df has MultiIndex columns (such a case will be handled below!)

        XNs_tags:
          Defaults to CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols() when XNs_tags is None
        NOTE: XNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled in CPXDfBuilder.find_XNs_cols_idxs_from_cpx_df.

        --------------------
        THE FOLLOWING ARE ONLY NEEDED WHEN combine_others==True
            is_norm
            counts_col

            counts_col:
              Should be a string.  If cpx_df has MultiIndex columns, the level_0 value will be handled.
              Will still function properly if appropriate tuple/list is supplied instead of string
                e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming cpx_df columns
                      have level_0 values equal to 'counts')
              EASIEST TO SUPPLY STRING!!!!!
              NOTE: Only really needed if is_norm==True
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
            non_reason_lvl_0_vals = non_reason_lvl_0_vals + CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols()
            #--------------------------------------------------
            pd_dfs = []
            for col_lvl_0_val_i in col_lvl_0_vals:
                rcpx_pd_i  = cpx_df[[col_lvl_0_val_i]].copy()
                #-----
                if col_lvl_0_val_i in non_reason_lvl_0_vals:
                    pd_dfs.append(rcpx_pd_i)
                    continue
                #-----
                rcpx_pd_i = CPXDf.get_top_reasons_subset_from_cpx_df(
                    cpx_df                    = rcpx_pd_i, 
                    n_reasons_to_include      = n_reasons_to_include,
                    combine_others            = combine_others, 
                    output_combine_others_col = output_combine_others_col, 
                    XNs_tags                  = XNs_tags, 
                    is_norm                   = is_norm, 
                    counts_col                = counts_col, 
                    total_counts_col          = total_counts_col, 
                    non_reason_lvl_0_vals     = non_reason_lvl_0_vals
                )
                pd_dfs.append(rcpx_pd_i)

            #--------------------------------------------------
            # Make sure all dfs in pd_dfs look correct
            shape_0 = pd_dfs[0].shape[0]
            index_0 = pd_dfs[0].index
            for i in range(len(pd_dfs)):
                if i==0:
                    continue
                assert(pd_dfs[i].shape[0]==shape_0)
                assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
                #-----
                # Aligning the indices is not strictly necessary, as pd.concat should handle that
                # But, it's best to be safe
                pd_dfs[i] = pd_dfs[i].loc[index_0]
            #--------------------------------------------------
            # Build rcpx_final by combining all dfs in pd_dfs
            # rcpx_final = pd.concat(pd_dfs, axis=1)
            rcpx_final = CPXDfBuilder.merge_cpx_dfs(
                dfs_coll                      = pd_dfs, 
                max_total_counts              = None, 
                how_max_total_counts          = 'any', 
                XNs_tags                      = CPXDf.std_XNs_cols() + CPXDf.std_nXNs_cols(), 
                cols_to_init_with_empty_lists = CPXDf.std_XNs_cols(), 
                make_cols_equal               = False, 
                sort_cols                     = False, 
            )
            #--------------------------------------------------
            return rcpx_final

        #----------------------------------------------------------------------------------------------------
        # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
        assert(cpx_df.columns.nlevels<=2)
        if cpx_df.columns.nlevels==2:
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                assert(cpx_df.columns.get_level_values(0).nunique()==1)
                level_0_val = cpx_df.columns.get_level_values(0).unique().tolist()[0]
                counts_col  = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
        #-------------------------
        if combine_others and is_norm:
            # In this case, counts_col will be needed for combine
            # So, grab the values to be used later
            assert(counts_col in cpx_df.columns)
            counts_col_vals = cpx_df[counts_col]
        #-------------------------
        non_SNs_cols = CPXDfBuilder.get_non_XNs_cols_from_cpx_df(
            cpx_df   = cpx_df, 
            XNs_tags = XNs_tags
        )
        cpx_df = cpx_df[non_SNs_cols]
        
        # The counts_col is typically removed above with get_non_XNs_cols_from_cpx_df.
        # However, it is not always removed (as, e.g., when sent from MECPOCollection.get_top_reasons_subset_from_merged_cpo_df
        #   it will have a random name which will not be caught by CPXDfBuilder.get_non_XNs_cols_from_cpx_df
        # It is important that it be removed, so it does not affect the ordering of reasons
        if counts_col in cpx_df.columns:
            cpx_df = cpx_df.drop(columns=[counts_col])

        # Also important for total_counts to be removed!
        if total_counts_col in cpx_df.columns.get_level_values(-1):
            cpx_df = cpx_df.drop(columns=[total_counts_col], level=-1)
        
        # Make sure all remaining columns are numeric
        assert(cpx_df.shape[1]==cpx_df.select_dtypes('number').shape[1])
        #-------------------------
        reason_order       = cpx_df.mean().sort_values(ascending=False).index.tolist()
        reasons_to_include = reason_order[:n_reasons_to_include]
        other_reasons      = reason_order[n_reasons_to_include:]
        #-------------------------
        if combine_others:
            if is_norm:
                # counts_col is needed for combine_cpx_df_reasons_explicit, but should
                #   have been removed above
                assert(counts_col not in cpx_df.columns)
                cpx_df = pd.merge(cpx_df, counts_col_vals, left_index=True, right_index=True)
            cpx_df = CPXDf.combine_cpx_df_reasons_explicit(
                rcpx_df               = cpx_df, 
                reasons_to_combine    = other_reasons,
                combined_reason_name  = output_combine_others_col, 
                is_norm               = is_norm, 
                counts_col            = counts_col,  
                non_reason_lvl_0_vals = non_reason_lvl_0_vals, 
            )
            if is_norm:
                cpx_df = cpx_df.drop(columns=[counts_col])
            # NOTE: If cpx_df has MultiIndex columns, output_combine_others_col will have been correctly converted
            #       to a tuple in CPXDf.combine_cpx_df_reasons_explicit.  Therefore, to select it below,
            #       it must be converted to tuple
            if cpx_df.columns.nlevels==2:
                assert(isinstance(output_combine_others_col, str))
                combine_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(
                    df  = cpx_df, 
                    col = output_combine_others_col
                )
                assert(len(combine_col_idx)==1)
                combine_col_idx           = combine_col_idx[0]
                output_combine_others_col = cpx_df.columns[combine_col_idx]
            cpx_df = cpx_df[reasons_to_include+[output_combine_others_col]]
        else:
            cpx_df = cpx_df[reasons_to_include]
        #-------------------------
        return cpx_df
    

    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def get_overall_top_reasons_subset_from_merged_cpx_df(
        cpx_df                    ,
        n_reasons_to_include      = 10,
        combine_others            = True,
        output_combine_others_col = 'Other Reasons',
        XNs_tags                  = None, 
        is_norm                   = False, 
        counts_series             = None, 
        total_counts_col          = 'total_counts', 
        non_reason_lvl_0_vals     = ['EEMSP_0', 'time_info_0']
    ):
        r"""
        Similar to CPXDf.get_top_reasons_subset_from_cpx_df, but for entire data overall, not for each time group individually
        """
        #-------------------------
        assert(cpx_df.columns.nlevels<=2)
        #-------------------------
        need_counts_col = False
        if combine_others and is_norm:
            need_counts_col = True
            assert(counts_series is not None and isinstance(counts_series, pd.Series))
            # Make sure all needed values are found in counter_series
            assert(len(set(cpx_df.index).difference(counts_series.index))==0)
            tmp_col       = (Utilities.generate_random_string(), Utilities.generate_random_string())
            counts_series = counts_series.to_frame(name=tmp_col)
            cpx_df = pd.merge(cpx_df, counts_series, left_index=True, right_index=True, how='inner')
        #-------------------------
        # Important for total_counts to be removed!
        if total_counts_col in cpx_df.columns.get_level_values(-1):
            cpx_df = cpx_df.drop(columns=[total_counts_col], level=-1)
        #-------------------------
        # Life is much easier if I flatten the columns here
        org_cols = cpx_df.columns
        join_str = ' '
        cpx_df = Utilities_df.flatten_multiindex_columns(df=cpx_df, join_str = join_str, reverse_order=False)
        rename_cols_dict = dict(zip(cpx_df.columns, org_cols))
        if need_counts_col:
            counts_col = join_str.join(tmp_col)
            # Remove counts_col from rename_cols_dict, as CPXDf.get_top_reasons_subset_from_cpx_df will remove it
            #   from cpx_df
            del rename_cols_dict[counts_col]
        else:
            counts_col = None
        #-------------------------
        # NOTE: If combine_others==True, output_combine_others_col won't be in rename_cols_dict
        #       Add by hand, in case combine_others==True and needed
        rename_cols_dict[output_combine_others_col] = (output_combine_others_col, output_combine_others_col)
        #-------------------------
        cpx_df = CPXDf.get_top_reasons_subset_from_cpx_df(
            cpx_df                     = cpx_df, 
            n_reasons_to_include       = n_reasons_to_include,
            combine_others             = combine_others, 
            output_combine_others_col  = output_combine_others_col, 
            XNs_tags                   = XNs_tags, 
            is_norm                    = is_norm, 
            counts_col                 = counts_col, 
            total_counts_col           = total_counts_col, 
            non_reason_lvl_0_vals      = non_reason_lvl_0_vals
        )
        #-----
        # Unflatten the columns
        # Make sure column ordering is same
        new_cols = [rename_cols_dict[x] for x in cpx_df.columns]
        new_cols = pd.MultiIndex.from_tuples(new_cols)
        cpx_df.columns = new_cols
        #-------------------------
        return cpx_df