#!/usr/bin/env python

"""
Utilities specifically for DataFrames
"""

__author__ = "Jesse Buxton"
__email__  = "buxton.45.jb@gmail.com"
__status__ = "Personal"

# List of functions:

#----------------------------------------------------------------------------------------------------
import os
import sys
import glob
import re

import copy
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_object_dtype
from pandas.testing import assert_frame_equal
from scipy import stats
import statistics
from natsort import natsorted, ns, natsort_keygen
from enum import IntEnum
import datetime
from sklearn.model_selection import GroupShuffleSplit
from ast import literal_eval

import Utilities
#----------------------------------------------------------------------------------------------------
class DFConstructType(IntEnum):
    kReadCsv = 0
    kRunSqlQuery = 1
    kImportPickle = 2
    kUnset = 3


#----------------------------------------------------------------------------------------------------
def get_numeric_columns(
        df
    ):
    is_numeric_col = [is_numeric_dtype(df[col]) for col in df.columns]
    numeric_cols   = df.columns[is_numeric_col].tolist()
    #-----
    return numeric_cols

#--------------------------------------------------
def append_to_column_names(
        df, 
        append
    ):
    df.columns = [str(col)+f'_{append}' for col in df.columns]
    #-----
    return df

#--------------------------------------------------
def prepend_to_column_names(
        df, 
        prepend
    ):
    df.columns = [f'{prepend}_'+str(col) for col in df.columns]
    #-----
    return df

#--------------------------------------------------
def append_to_series_indices(
        series, 
        append
    ):
    series.index = [str(idx)+f'_{append}' for idx in series.index]
    #-----
    return series

#--------------------------------------------------
def prepend_to_series_indices(
        series, 
        prepend
    ):
    series.index = [f'{prepend}_'+str(idx) for idx in series.index]
    #-----
    return series

#----------------------------------------------------------------------------------------------------
def get_shared_columns(
        dfs                , 
        maintain_df0_order = True
    ):
    r"""
    The method here, using intersect1d, will typically sort the results
    If this is not desired, and the original order maintained, set 
    maintain_df0_order=True.  
    Note, in general the dfs may have different column orders, and
    the first df in the list will be used for the ordering
    """
    assert(Utilities.is_object_one_of_types(dfs, [list, tuple]))
    cols_shared = dfs[0].columns.tolist()
    for df in dfs:
        cols_shared = np.intersect1d(cols_shared, df.columns.tolist())
    cols_shared = cols_shared.tolist()
    if maintain_df0_order:
        cols_shared_ordered = [x for x in dfs[0].columns.tolist() if x in cols_shared]
        assert(len(cols_shared_ordered)==len(cols_shared))
        return cols_shared_ordered
    return cols_shared

#----------------------------------------------------------------------------------------------------
def move_cols_to_either_end(
        df           , 
        cols_to_move , 
        to_front     = True
    ):
    r"""
    Moves the columns in cols_to_move to the front or back of the df
    The order of the other columns is maintained
      if to_front==True -----> moved to front of df
      if to_front==False ----> moved to back of df
    NOTE: cols_to_move should be a list.
          However, feeding in e.g. df2.columns should work as well
          due to the lines below
    """
    #--------------------------------------------------
    if isinstance(cols_to_move, pd.core.indexes.base.Index):
        cols_to_move = cols_to_move.tolist()
    #-----
    assert(x in df.columns for x in cols_to_move)
    #-----
    remaining_cols = [x for x in df.columns if x not in cols_to_move]
    #-------------------------
    if to_front:
        return df[cols_to_move + remaining_cols]
    else:
        return df[remaining_cols + cols_to_move]
        
#--------------------------------------------------
def move_cols_to_front(
        df, 
        cols_to_move
    ):
    return move_cols_to_either_end(
        df           = df, 
        cols_to_move = cols_to_move, 
        to_front     = True
    )

#--------------------------------------------------
def move_cols_to_back(
        df, 
        cols_to_move
    ):
    return move_cols_to_either_end(
        df           = df, 
        cols_to_move = cols_to_move, 
        to_front     = False
    )

#--------------------------------------------------
def make_all_column_names_lowercase(
        df
    ):
    rename_dict = {x:x.lower() for x in df.columns}
    df          = df.rename(columns=rename_dict)
    return df

#--------------------------------------------------
def make_all_column_names_uppercase(
        df, 
        cols_to_exclude = None
    ):
    rename_dict = {x:x.upper() for x in df.columns}
    if cols_to_exclude is not None:
        assert(isinstance(cols_to_exclude, list))
        for exclude_i in cols_to_exclude:
            del rename_dict[exclude_i]
    df = df.rename(columns=rename_dict)
    return df

#--------------------------------------------------
def drop_col_case_insensitive(
        df        , 
        col       , 
        inplace   = True, 
        drop_dups = True
    ):
    r"""
    In a case-insensitive manner, look for col in df.  If found, drop.
    """
    #-------------------------
    if not inplace:
        df = df.copy()
    #-------------------------
    if col.lower() in [x.lower() for x in df.columns]:
        tmp_idx = [x.lower() for x in df.columns].index(col.lower())
        col     = df.columns.tolist()[tmp_idx]
        if drop_dups:
            df = df.drop(columns=[col]).drop_duplicates()
    return df

#----------------------------------------------------------------------------------------------------
def first_valid_index_for_col(
    df, 
    col, 
    return_iloc=False
):
    r"""
    Just like pd.first_valid_index, but explicitly for a single column
    
    return_iloc:
        If True, return the integer index location, between 0 and df.shape[0]-1
    """
    #-------------------------
    if return_iloc:
        return df.reset_index()[col].first_valid_index()
    else:
        return df[col].first_valid_index()
        

#----------------------------------------------------------------------------------------------------
def make_df_col_dtypes_equal(
    df_1              , 
    col_1             , 
    df_2              , 
    col_2             , 
    allow_reverse_set = False, 
    assert_success    = True, 
    inplace           = False
):
    r"""
    Try to make df_2[col_2].dtype equal df_1[col_1].dtype.
    The function attempts to alter the dtype of df_2 to match that of df_1.
    If the operation fails and allow_reverse_set==True, the function will attempt to alter
      the dtype of df_1 to match that of df_2
      
    col_1, col_2 can be single columns, or a list of columns
    
    inplace:
        Default to False because we typically do not want this operation done in place.
        If done in place and if operation fails (i.e., final assertion fails), then df_1, df_2 would still
          be altered outside of the function, which is not desired.
    """
    #--------------------------------------------------
    # First, check to make sure anything actually needs done.  If not, simply return DFs
    if isinstance(col_1, list) or isinstance(col_2, list):
        assert(isinstance(col_1, list) and isinstance(col_2, list))
        assert(len(col_1) == len(col_2))
        if df_1[col_1].dtypes.tolist() == df_2[col_2].dtypes.tolist():
            return df_1, df_2
    else:
        if df_1[col_1].dtype == df_2[col_2].dtype:
            return df_1, df_2
    
    #--------------------------------------------------
    if not inplace:
        df_1 = df_1.copy()
        df_2 = df_2.copy()
    #--------------------------------------------------
    if isinstance(col_1, list) or isinstance(col_2, list):
        assert(isinstance(col_1, list) and isinstance(col_2, list))
        assert(len(col_1)==len(col_2))
        # NOTE: inplace already taken care of above, so set to True in all iterations below
        for i_col in range(len(col_1)):
            df_1, df_2 = make_df_col_dtypes_equal(
                df_1              = df_1, 
                col_1             = col_1[i_col], 
                df_2              = df_2, 
                col_2             = col_2[i_col], 
                allow_reverse_set = allow_reverse_set, 
                assert_success    = assert_success, 
                inplace           = True
            )
        return df_1, df_2
    #--------------------------------------------------
    assert(col_1 in df_1.columns.tolist())
    assert(col_2 in df_2.columns.tolist())
    #-------------------------
    dtype_1 = df_1[col_1].dtype
    dtype_2 = df_2[col_2].dtype
    #-------------------------
    if df_1.shape[0]>0 and df_2.shape[0]>0:
        frst_valid_iloc_1 = first_valid_index_for_col(
            df          = df_1, 
            col         = col_1, 
            return_iloc = True
        )
        if frst_valid_iloc_1 is None:
            frst_valid_iloc_1 = 0
        #-----
        frst_valid_iloc_2 = first_valid_index_for_col(
            df          = df_2, 
            col         = col_2, 
            return_iloc = True
        )
        if frst_valid_iloc_2 is None:
            frst_valid_iloc_2 = 0
        #-------------------------
        # If dtypes are the same, then simply return
        # This is slightly more involved, as both could be of type object but not equal (e.g.,
        #   one could be a list and the other a string)
        if is_object_dtype(dtype_1) and is_object_dtype(dtype_2):
            if type(df_1.iloc[frst_valid_iloc_1][col_1])==type(df_2.iloc[frst_valid_iloc_2][col_2]):
                return df_1, df_2
        else:
            if dtype_1 == dtype_2:
                return df_1, df_2
        #-------------------------
        # If the dtype is object, it is uncertain exactly how it should be treated, UNLESS
        #   it is a string or a list
        if is_object_dtype(dtype_1):
            if df_1.iloc[frst_valid_iloc_1][col_1] is None:
                assert(df_2.iloc[frst_valid_iloc_2][col_2] is not None)
                df_1 = convert_col_type(
                    df                = df_1, 
                    column            = col_1, 
                    to_type           = dtype_2, 
                    to_numeric_errors = 'coerce', 
                    inplace           = False
                )                
            elif isinstance(df_1.iloc[frst_valid_iloc_1][col_1], str):
                dtype_1 = str
            elif isinstance(df_1.iloc[frst_valid_iloc_1][col_1], list):
                dtype_1 = list
                df_2[col_2] = df_2[col_2].apply(lambda x: [x])
                return df_1, df_2
            else:
                print(f'col_1={col_1} in df_1 is object type, but not string or list')
                print('NOT SURE HOW TO HANDLE!!!!!')
                if assert_success:
                    assert(0)
                else:
                    return df_1, df_2
    #-------------------------
    # At this point, if dtype_1 was an object, it must be a string
    # If it was a list, it would have been handled in elif, and if it was anything else
    #   it would have been handled in else
    # One other (much less common) possibility: df_1 and/or df_2 are empty
    #-----
    try:
        df_2 = convert_col_type(
            df                = df_2, 
            column            = col_2, 
            to_type           = dtype_1, 
            to_numeric_errors = 'coerce', 
            inplace           = False
        )
    except:
        if allow_reverse_set:
            try:
                # NOTE: Import allow_reverse_set=False below to avoid infinite loop!
                df_2, df_1 = make_df_col_dtypes_equal(
                    df_1              = df_2, 
                    col_1             = col_2, 
                    df_2              = df_1, 
                    col_2             = col_1, 
                    allow_reverse_set = False, 
                    assert_success    = assert_success
                )
            except:
                # if assert_success below will handle failure
                pass
        else:
            # if assert_success below will handle failure
            pass
    #-------------------------
    if assert_success:
        assert(df_1[col_1].dtype == df_2[col_2].dtype)
    #-------------------------
    return df_1, df_2
    

#----------------------------------------------------------------------------------------------------
# !!!!!!!!!!!!!! Probably want to look into pd.convert_dtypes for use in (or possibly to replace) the following functions!
#----------------------------------------------------------------------------------------------------
def convert_col_type_with_to_numeric(
        df      , 
        column  , 
        to_type , 
        errors  = 'coerce', 
        inplace = True
    ):
    r"""
    Documentation
    """
    #-------------------------
    if not inplace:
        df = df.copy()
    #-------------------------
    # assert(isinstance(to_type, type))
    assert(is_numeric_dtype(to_type))
    df[column] = pd.to_numeric(df[column], errors=errors)
    df[column] = df[column].astype(to_type)
    #-------------------------
    return df

#--------------------------------------------------
def convert_col_type_with_astype(
    df      , 
    column  , 
    to_type , 
    inplace = True
):
    #-------------------------
    r"""
    Sometimes, conversions are annoying, and an intermediate conversion is needed
      e.g., when I used to convert 'OUTG_REC_NB' with similar methods from string to int (I use
            convert_col_type_with_to_numeric now), I actually had to do it in two steps:
                df_outage['OUTG_REC_NB'] = df_outage['OUTG_REC_NB'].astype(np.float).astype(np.int32)
    I allow for that in the for loop below (if to_type is a list or tuple)
    """
    #-------------------------
    # NOTE: inplace handled for entire df here, so in for loop below
    #       should be set to True
    if not inplace:
        df = df.copy()
    #-------------------------
    if Utilities.is_object_one_of_types(to_type, [list, tuple]):
        for tp_i in to_type:
            df = convert_col_type_with_astype(
                df      = df, 
                column  = column, 
                to_type = tp_i, 
                inplace = True # inplace already handled for entire df at top
            )
        return df
    #-------------------------
    assert(isinstance(to_type, type))
    df[column] = df[column].astype(to_type)
    #-------------------------
    return df

#--------------------------------------------------
def convert_col_type_w_pd_to_datetime(
    df          , 
    column      , 
    format      = None, 
    output_strf = None, 
    inplace     = True
):
    #-------------------------
    r"""
    Only converts a single column at a time.  For multiple columns, see convert_col_types.
    """
    #-------------------------
    if not inplace:
        df = df.copy()
    #-------------------------
    df[column] = pd.to_datetime(
        df[column], 
        format = format
    )
    #-------------------------
    if output_strf is not None:
        df[column] = df[column].dt.strftime(output_strf)
    #-------------------------
    return df

#--------------------------------------------------
def change_empty_entries_in_col_and_convert_to_type(
        df           , 
        column       , 
        empty_entry  = '', 
        replace_with = np.nan, 
        col_to_type  = float, 
        inplace      = True
    ):
    r"""
    NOTE: The method convert_col_type_with_to_numeric now handles the original intention of this function.
          However, this function is kept as it still may be useful in the future.
          
    In df[column], this finds all empty entries (equal to empty_entry argument) 
      and sets them equal to replace_with
    Furthermore, it converts the dtype of the column to col_to_type
    NOTE: This was built to work with a column of string entries, in which case
            an empty element is equal to '' and replaced with np.nan.
          More specifically, this was built to work with the annual_kwh column
            of meter_premise dataset
    NOTE: empty_entry can be a single string or a list of strings
    """
    #-------------------------
    if not inplace:
        df = df.copy()
    #-------------------------
    assert(isinstance(column, str))
    if Utilities.is_object_one_of_types(empty_entry, [list, tuple]):
        if '' in empty_entry:
            df[column] = df[column].str.strip()
        df.loc[df[column].isin(empty_entry), column]=replace_with
    else:
        if empty_entry=='':
            df[column] = df[column].str.strip()
        df.loc[df[column]==empty_entry, column]=replace_with
    #-------------------------
    df[column] = df[column].astype(col_to_type)
    #-------------------------
    return df

#--------------------------------------------------
def convert_col_type(
        df                ,  
        column            , 
        to_type           , 
        to_numeric_errors = 'coerce', 
        inplace           = True
    ):
    r"""
    Convert the type of column in df to to_type
    If:
        - to_type is numeric  ==> convert_col_type_with_to_numeric
        - to_type is datetime ==> convert_col_type_w_pd_to_datetime
        - to_type is dict     ==> change_empty_entries_in_col_and_convert_to_type
        - else                ==> convert_col_type_with_astype
    
    Sometimes, conversions are annoying, and an intermediate conversion is needed
      e.g., when I convert 'OUTG_REC_NB' from string to int, I actually have to do it in two steps:
            df_outage['OUTG_REC_NB'] = df_outage['OUTG_REC_NB'].astype(np.float).astype(np.int32)
    I allow for that in the for loop below (if to_type is a list or tuple)
    
    NEW: to_type can also be a dict, in which case it will be unpackaged and sent
         to Utilities_df.change_empty_entries_in_col_and_convert_to_type
    """
    #-------------------------
    # NOTE: inplace handled for entire df here, so in for loop below
    #       should be set to True
    if not inplace:
        df = df.copy()
    #-------------------------
    if Utilities.is_object_one_of_types(to_type, [list, tuple]):
        # This handles the case of intermediate conversions, as described above
        #   NOT THE CASE WHERE THE DTYPE IS A LIST OR TUPLE!
        for tp_i in to_type:
            df = convert_col_type(
                df=df, 
                column=column, 
                to_type=tp_i, 
                to_numeric_errors=to_numeric_errors, 
                inplace=True # inplace already handled for entire df at top
            )
        return df
    #-------------------------
    if is_numeric_dtype(to_type):
        df = convert_col_type_with_to_numeric(
            df=df, 
            column=column, 
            to_type=to_type, 
            errors=to_numeric_errors, 
            inplace=True
        )
    elif to_type is datetime.datetime or is_datetime64_dtype(to_type):
        df = convert_col_type_w_pd_to_datetime(
            df          = df, 
            column      = column, 
            format      = None, 
            output_strf = None, 
            inplace     = True 
        )
    elif isinstance(to_type, dict):
        df = change_empty_entries_in_col_and_convert_to_type(
            df=df, 
            column=col_i, 
            inplace=True, 
            **type_i
        )
    else:
        df = convert_col_type_with_astype(
            df=df, 
            column=column, 
            to_type=to_type, 
            inplace=True)
    #-------------------------
    return df

#--------------------------------------------------
def convert_col_types(df, cols_and_types_dict, to_numeric_errors='coerce', inplace=True):
    r"""
    Documentation
    """
    #-------------------------
    if cols_and_types_dict is None:
        return df
    #-------------------------
    # NOTE: inplace handled for entire df here, so in for loop below
    #       should be set to True
    if not inplace:
        df = df.copy()
    #-------------------------
    for col_i,type_i in cols_and_types_dict.items():
        df = convert_col_type(
            df=df, 
            column=col_i, 
            to_type=type_i, 
            to_numeric_errors=to_numeric_errors, 
            inplace=True
        )
    return df


#--------------------------------------------------
def ensure_dt_cols(
    df      ,
    dt_cols 
):
    r"""
    Checks if dt_cols are datetime objects, and, if not, converts them

    df:
        pd.DataFrame or dict with pd.DataFrame values
    dt_cols:
        list of columns to check/convert
    """
    #--------------------------------------------------
    assert(isinstance(dt_cols, list))
    assert(Utilities.is_object_one_of_types(df, [pd.DataFrame, dict]))
    #--------------------------------------------------
    if isinstance(df, dict):
        for key_i in df.keys():
            df[key_i] = ensure_dt_cols(
                df      = df[key_i], 
                dt_cols = dt_cols
            )
        return df
    #--------------------------------------------------
    assert(isinstance(df, pd.DataFrame))
    assert(set(dt_cols).difference(set(df.columns.tolist()))==set())
    for col_i in dt_cols:
        if not is_datetime64_dtype(df[col_i]):
            df = convert_col_type_w_pd_to_datetime(
                df          = df, 
                column      = col_i, 
                format      = None, 
                output_strf = None, 
                inplace     = True
            )
        assert(is_datetime64_dtype(df[col_i]))
    #--------------------------------------------------
    return df

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
def prepend_level_to_MultiIndex(df, level_val, level_name=None, axis=0):
    r"""
    Prepend a level to a MultiIndex; i.e., add a new lowest level.
    axis=0:
      Prepend level to index
    axis=1:
      Prepend level to columns
    """
    #-------------------------
    # The commented line below should give the exact same result
    # df = pd.concat([df], keys=[level_val], names=[level_name], axis=axis)
    df = pd.concat({level_val: df}, names=[level_name], axis=axis)
    return df

#--------------------------------------------------
def prepend_levels_to_MultiIndex(
    df, 
    n_levels_to_add, 
    dummy_col_levels_prefix='dummy', 
    axis=1
):
    r"""
    """
    #-------------------------
    for i_new_lvl in range(n_levels_to_add):
        # With each iteration, prepending a new level from n_levels_to_add-1 to 0
        i_level_val = f'{dummy_col_levels_prefix}{(n_levels_to_add-1)-i_new_lvl}'
        df = prepend_level_to_MultiIndex(
            df=df, 
            level_val=i_level_val, 
            level_name=None, 
            axis=axis
        )
    #-------------------------
    return df

#--------------------------------------------------
def flatten_multiindex_index(df, inplace=True):
    # This basically just calls reset_index on all levels of df except for the 0th level
    # The motivation for this development was for use after creating an aggregate df with 
    #   df.groupby(...).agg(...)
    #-----
    # If indices are not a MultiIndex, then simply return the DataFrame.
    # This prevents unwanted effects if one accidentally e.g. feeds a df through twice,
    #   or attempts to feed through a df with single column index.
    #-----
    if not isinstance(df.index, pd.MultiIndex):
        return df
    #-----
    if not inplace:
        df = df.copy()
    #-----
    n_levels = df.index.nlevels
    if inplace:
        df.reset_index(level=list(range(1, n_levels)), inplace=True)
    else:
        df = df.reset_index(level=list(range(1, n_levels)))   
    return df

#--------------------------------------------------
def get_flattened_multiindex_columns(df_columns, join_str = ' ', reverse_order=True, to_ignore=['first']):
    # The level=0 value is always kept, as this is typically e.g. the measurement
    # For all other levels, if any element in to_ignore is found, that level is ignored
    #   e.g., col_1 = ('prem_nb',  'first_mtrs',  'first_TRS') and to_ignore=['first']
    #         ==> new_col = 'prem_nb'
    #   e.g., col_1 = ('value',  'first_mtrs',  'mean_TRS') and to_ignore=['first']
    #         ==> new_col = 'mean_TRS value'
    #-----
    assert(isinstance(df_columns, pd.MultiIndex) or 
           isinstance(df_columns, list) or 
           isinstance(df_columns, tuple))   
    #-----
    rename_dict = {}
    for multi_col in df_columns:
        assert(isinstance(multi_col, tuple) or isinstance(multi_col, list))
        orig_col = multi_col
        assert(multi_col not in rename_dict)
        multi_col = [multi_col[0]] + Utilities.remove_tagged_from_list(multi_col[1:], to_ignore)
        multi_col = tuple(multi_col)
        if reverse_order:
            new_col = join_str.join(reversed(multi_col))
        else:
            new_col = join_str.join(multi_col)
        rename_dict[orig_col] = new_col
    return rename_dict

#--------------------------------------------------
def flatten_multiindex_columns(df, join_str = ' ', reverse_order=True, to_ignore=['first'], inplace=True):
    # The level=0 value is always kept, as this is typically e.g. the measurement
    # For all other levels, if any element in to_ignore is found, that level is ignored
    #   e.g., col_1 = ('prem_nb',  'first_mtrs',  'first_TRS') and to_ignore=['first']
    #         ==> new_col = 'prem_nb'
    #   e.g., col_1 = ('value',  'first_mtrs',  'mean_TRS') and to_ignore=['first']
    #         ==> new_col = 'mean_TRS value'
    #-----
    # If columns are not a MultiIndex, then simply return the DataFrame.
    # This prevents unwanted effects if one accidentally e.g. feeds a df through twice,
    #   or attempts to feed through a df with single column index.
    #-----
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    #-----
    if not inplace:
        df = df.copy()    
    #-----
    rename_dict = get_flattened_multiindex_columns(df_columns=df.columns, join_str=join_str, 
                                                   reverse_order=reverse_order, to_ignore=to_ignore)
    # Have to flatten columns before renaming
    df.columns = df.columns.to_flat_index()
    if inplace:
        df.rename(columns=rename_dict, inplace=True)
    else:
        df = df.rename(columns=rename_dict)
    return df

#--------------------------------------------------
def flatten_multiindex(df, flatten_index=False, flatten_columns=False, inplace=True, 
                       index_args={}, column_args={}):
    # column_args can be any arguments in flatten_multiindex_columns except for df and inplace (as those are handled explicitly)
    # Same goes for index_args an flatten_multiindex_index
    # e.g.
    #     column_args = dict(join_str = ' ', reverse_order=True, to_ignore=['first'])
    #     column_args = {'join_str':' ', 'reverse_order':True, 'to_ignore':['first']}
    if flatten_columns+flatten_index==0:
        return df
    if flatten_index:
        df = flatten_multiindex_index(df, inplace=inplace, **index_args)
    if flatten_columns:
        df = flatten_multiindex_columns(df, inplace=inplace, **column_args)
    return df

#----------------------------------------------------------------------------------------------------
def change_idx_names_andor_order(
    df, 
    rename_idxs_dict=None, 
    final_idx_order=None, 
    inplace=True
):
    r"""
    rename_idxs_dict:
        If supplied, this must be a dictionary mapping the original index names to the new names.
          i.e., the key values must equal the df.index.names
    final_idx_order:
        If supplied, the output DF will have it's index re-ordered to match final_idx_order
        **** IMPORTANT *** If some of the df.index.names are not contained in final_idx_order, THE CORRESPONDING LEVELS WILL BE DROPPED FROM THE RETURNED DF
        If rename_idxs_dict is supplied:
            ==> final_idx_order and rename_idxs_dict.values() must contain the same elements (not necessarily in the same order though)
        If rename_idxs_dict is not supplied:
            ==> final_idx_order and df.index.names loaded in function must contain the same elements (not necessarily in the same order though)
    """
    #--------------------------------------------------
    if rename_idxs_dict is None and final_idx_order is None:
        return df
    #--------------------------------------------------
    # This method really requires the index levels to be uniquely named
    assert(all([x is not None for x in df.index.names]))
    assert(len(set(df.index.names))==df.index.nlevels)
    #--------------------------------------------------
    if not inplace:
        df = df.copy()
    #--------------------------------------------------
    if rename_idxs_dict is not None:
        assert(isinstance(rename_idxs_dict, dict))
        #-------------------------
        # Any df.index.names not included in rename_idxs_dict are injected as rename_idxs_dict[x] = x (i.e., they will be left unchanged)
        if set(df.index.names).difference(set(rename_idxs_dict.keys()))!=set():
            rename_idxs_dict = rename_idxs_dict | {x:x for x in df.index.names if x not in rename_idxs_dict.keys()}
        #-------------------------
        assert(set(df.index.names).difference(set(rename_idxs_dict.keys()))==set())
        df.index.names = [rename_idxs_dict[x] for x in df.index.names]
    #--------------------------------------------------
    if final_idx_order is not None:
        assert(isinstance(final_idx_order, list))
        #-------------------------
        idx_levels_to_drop = list(set(df.index.names).difference(set(final_idx_order)))
        if len(idx_levels_to_drop) > 0:
            for idx_level_i in idx_levels_to_drop:
                df = df.droplevel(level=idx_level_i, axis=0)
        #-------------------------
        assert(set(df.index.names).symmetric_difference(set(final_idx_order))==set())
        df = df.reorder_levels(order=final_idx_order, axis=0)
    #--------------------------------------------------
    return df
    
  
#----------------------------------------------------------------------------------------------------
def find_idxs_in_highest_order_of_columns(df, col, exact_match=True, assert_single=False):
    r"""
    Find where col occurs in highest order of MultiIndex columns
    This will work with normal columns as well
    """
    if exact_match and not col.endswith('_EXACTMATCH'):
        col += '_EXACTMATCH'
    tagged_idxs = Utilities.find_tagged_idxs_in_list(
        lst=df.columns.get_level_values(-1).tolist(), 
        tags=[col])
    if assert_single:
        assert(len(tagged_idxs)==1)
        return tagged_idxs[0]
    else:
        return tagged_idxs
    
#--------------------------------------------------
def find_single_col_in_multiindex_cols(df, col):
    r"""
    Find where col occurs in highest order of MultiIndex columns, assert only one instance
      and return the full column (instead of index, as is done in find_idxs_in_highest_order_of_columns).
    This will work with normal columns as well
    """
    #-------------------------
    # Handle case where col equals exactly a column in df
    # True whether simple string or MultiIndex (i.e., tuple)
    if col in df.columns.tolist():
        return col
        
    #-------------------------
    # Handle case where col in top level, and only occurs once
    # (Motivation for functionality was common case where e.g., full column = ('trsf_pole_nb', '') 
    #    and user inputs col='trsf_pole_nb' to this function
    if col in df.columns:
        assert(len(df[[col]].columns.tolist())==1)
        return df[[col]].columns.tolist()[0]

    #-------------------------
    # Otherwise, look in highest order
    col_idx = find_idxs_in_highest_order_of_columns(
        df=df, 
        col=col, 
        exact_match=True
    )
    assert(len(col_idx)==1)
    col_idx = col_idx[0]
    return_col = df.columns.tolist()[col_idx]
    return return_col
    
#--------------------------------------------------
def find_col_idxs_with_regex(
    df, 
    regex_pattern, 
    ignore_case=False
):
    r"""
    Identify the location of columns using the regular expression regex_pattern.
    NOTE: This only works for pd.DataFrames with single-level columns (i.e., no multiindex)
    """
    #-------------------------
    assert(df.columns.nlevels==1)
    #-------------------------
    found_idxs = Utilities.find_idxs_in_list_with_regex(
        lst           = df.columns.tolist(), 
        regex_pattern = regex_pattern, 
        ignore_case   = ignore_case
    )
    return found_idxs

#--------------------------------------------------
def find_cols_with_regex(
    df, 
    regex_pattern, 
    ignore_case=False
):
    r"""
    Identify columns using the regular expression regex_pattern.
    NOTE: This only works for pd.DataFrames with single-level columns (i.e., no multiindex)
    """
    #-------------------------
    assert(df.columns.nlevels==1)
    #-------------------------
    found_idxs = find_col_idxs_with_regex(
        df            = df, 
        regex_pattern = regex_pattern, 
        ignore_case   = ignore_case
    )
    return df.columns[found_idxs].tolist()
    

#--------------------------------------------------
def find_idxs_in_lowest_order_of_columns(
    df, 
    regex_pattern, 
    ignore_case=False
):
    r"""
    Find columns with regex_pattern in lowest order of MultiIndex columns
    This will work with normal columns as well

    Returns a list of found indices (list of integers) and the column level where the regex_pattern was matched.
      i.e., returns (found_idxs, i_lvl) = (list, int)
    If regex_pattern not matched, empty list and -1 returned
    """
    #-------------------------
    for i_lvl in range(df.columns.nlevels):
        level_vals = df.columns.get_level_values(i_lvl).tolist()
        found_idxs = Utilities.find_idxs_in_list_with_regex(
            lst           = level_vals, 
            regex_pattern = regex_pattern, 
            ignore_case   = ignore_case
        )
        if len(found_idxs)>0:
            break
    #-------------------------
    if len(found_idxs)==0:
        return found_idxs, -1
    else:
        return found_idxs, i_lvl
        

#--------------------------------------------------        
def build_column_name_of_n_levels(
    nlevels, 
    level_0_val         = None,
    level_nm1_val       = None, 
    dummy_lvl_base_name = 'dummy_lvl_', 
    level_vals_dict     = None
):
    r"""
    This essentially just returns a tuple of length nlevels which can be used to name a MultiIndex column.
    By default, each column level (i.e., each element of the tuple) will be named f'{dummy_lvl_base_name}_{0}' to
      f'{dummy_lvl_base_name}_{nlevels}'.
    If level_0_val/level_nm1_val/level_vals_dict are supplied, the names are changed accordingly (see more info below).

    level_0_val:
        The name to use for the 0th level value.
        Overrides any value supplied in level_vals_dict
    level_nm1_val:
        The name to use for the last level's value (the n minus 1, or nm1, value).
        Overrides any value supplied in level_vals_dict
    level_vals_dict:
        A dictionary object supplying names to be used for the various levels.
        The dictionary must have keys of integer value between 0 and nlevels-1
        If level_0_val or level_nm1_val are supplied, those override the values supplied by level_vals_dict
    
    """
    #--------------------------------------------------
    if level_vals_dict is not None:
        assert(isinstance(level_vals_dict, dict))
    else:
        level_vals_dict = {}
    #-------------------------
    if dummy_lvl_base_name is None:
        dummy_lvl_base_name = ''
    #--------------------------------------------------
    return_names_dict = {}
    #-----
    for i_lvl in range(nlevels):
        assert(i_lvl not in return_names_dict.keys())
        return_names_dict[i_lvl] = level_vals_dict.get(
            i_lvl, 
            f'{dummy_lvl_base_name}{i_lvl}'
        )
    #--------------------------------------------------
    if level_0_val is not None:
        return_names_dict[0] = level_0_val
    if level_nm1_val is not None and nlevels>1:
        return_names_dict[len(return_names_dict)-1] = level_nm1_val
    #--------------------------------------------------
    # Could simply use return_names_dict.values(), but this makes 100% sure the order is correct
    assert(len(return_names_dict)==nlevels)
    return_col = []
    for i_lvl in range(nlevels):
        return_col.append(return_names_dict[i_lvl])
    #--------------------------------------------------
    if len(return_col)==1:
        return return_col[0]
    else:
        return tuple(return_col)

#--------------------------------------------------
def build_column_for_df(
    df, 
    input_col             = None, 
    level_0_val           = None,
    level_nm1_val         = None, 
    dummy_lvl_base_name   = 'dummy_lvl_', 
    level_vals_dict       = None,
    insert_input_at_front = True
):
    r"""
    Build a column name with the correct number of levels for df (e.g., when df has MultiIndex columns).
    If input_col is supplied, this method essentially modifies it to have the appropriate number of levels for df.
    This utilizes Utilities_df.build_column_name_of_n_levels (see documentation for more info).
    
    input_col:
        If supplied, must be a string or a tuple, and will be used to build the final output column
    """
    #--------------------------------------------------
    nlevels = df.columns.nlevels
    col_name_0 = build_column_name_of_n_levels(
        nlevels             = nlevels, 
        level_0_val         = level_0_val,
        level_nm1_val       = level_nm1_val, 
        dummy_lvl_base_name = dummy_lvl_base_name, 
        level_vals_dict     = level_vals_dict
    )
    #--------------------------------------------------
    if input_col is None:
        return col_name_0
    #--------------------------------------------------
    assert(Utilities.is_object_one_of_types(input_col, [str, tuple]))
    if nlevels==1:
        if isinstance(input_col, str):
            return input_col
        else:
            assert(isinstance(input_col, tuple))
            return input_col[0]
    else:
        # Need to adjust col_name_0, so must be list instead of tuple!
        col_name_0 = list(col_name_0)
        if isinstance(input_col, str):
            if insert_input_at_front:
                col_name_0[0] = input_col
            else:
                col_name_0[-1] = input_col
        else:
            assert(len(input_col) <= nlevels)
            if insert_input_at_front:
                col_name_0 = list(input_col) + col_name_0[len(input_col):]
            else:
                col_name_0 = col_name_0[:-len(input_col)] + list(input_col)
        #-------------------------
        # Convert back to tuple
        col_name_0 = tuple(col_name_0)
        #-------------------------
        return col_name_0
    
#----------------------------------------------------------------------------------------------------
def name_all_index_levels(
    df
):
    r"""
    Returns (df, idx_names_new, idx_names_org) where
        df:
            The input df with all index levels named
        idx_names_new:
            The new index level names (consisting of any already present in df and those needed (re-)named)
        idx_names_org:
            The original index level names
    """
    #--------------------------------------------------
    idx_names_org = list(df.index.names)
    idx_names_new = [
        x if (x is not None and x not in df.columns.tolist()) else Utilities.generate_random_string(str_len=10, letters='letters_only') 
        for x in df.index.names
    ]
    #-------------------------
    df.index.names = idx_names_new
    #-------------------------
    return (df, idx_names_new, idx_names_org)
    
    
#----------------------------------------------------------------------------------------------------
def get_idfr_loc(
    df, 
    idfr
):
    r"""
    Returns the identifier and whether or not it was found in index in a tuple of length 2
    If idfr found in columns, essentially just reutnrs idfr
    If idfr found in index, returns the index level where it was found
    
    idfr:
        This should be a string, list, or tuple.
        If column, idfr should simply be the column
            - Single index columns --> simple string
            - MultiIndex columns   --> appropriate tuple to identify column
        If in the index:
            - Single level index --> simple string 'index' or 'index_0'
            - MultiIndex index:  --> 
                - string f'index_{level}', where level is the index level containing the outg_rec_nbs
                - tuple of length 2, with 0th element ='index' and 1st element = idx_level_name where
                    idx_level_name is the name of the index level containing the outg_rec_nbs 
    """
    #-------------------------
    assert(Utilities.is_object_one_of_types(idfr, [str, list, tuple]))
    # NOTE: pd doesn't like checking for idfr in df.columns if idfr is a list.  It is fine checking when
    #       it is a tuple, as tuples can represent columns.  Therefore, if idfr is a list, convert to tuple
    #       as this will fix the issue below and have no effect elsewhere.
    if isinstance(idfr, list):
        idfr = tuple(idfr)
    if idfr in df.columns:
        return idfr, False
    #-------------------------
    # If not in the columns (because return from function not executed above), outg_rec_nbs must be in the indices!
    # The if/else block below determines idfr_idx_lvl
    if isinstance(idfr, str):
        assert(idfr.startswith('index'))
        if idfr=='index':
            idfr_idx_lvl=0
        else:
            idfr_idx_lvl = re.findall(r'index_(\d*)', idfr)
            assert(len(idfr_idx_lvl)==1)
            idfr_idx_lvl=idfr_idx_lvl[0]
            idfr_idx_lvl=int(idfr_idx_lvl)
    else:
        assert(len(idfr)==2)
        assert(idfr[0]=='index')
        idx_level_name = idfr[1]
        assert(idx_level_name in df.index.names)
        # Need to also make sure idx_level_name only occurs once, so no ambiguity!
        assert(df.index.names.count(idx_level_name)==1)
        idfr_idx_lvl = df.index.names.index(idx_level_name)
    #-------------------------
    assert(idfr_idx_lvl < df.index.nlevels)
    return (idfr_idx_lvl, True) 

#--------------------------------------------------    
def get_vals_from_df(df, idfr, unique=False):
    r"""
    Extract the values from a df.
      
    Return Value:
        If outg_rec_nbs are stored in a column of df, the returned object will be a pd.Series
        If outg_rec_nbs are stored in the index, the returned object will be a pd index.
        If one wants the returned object to be a list, use get_outg_rec_nbs_list_from_df

    idfr:
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
    assert(Utilities.is_object_one_of_types(idfr, [str, list, tuple]))
    # NOTE: pd doesn't like checking for idfr in df.columns if idfr is a list.  It is fine checking when
    #       it is a tuple, as tuples can represent columns.  Therefore, if idfr is a list, convert to tuple
    #       as this will fix the issue below and have no effect elsewhere.
    if isinstance(idfr, list):
        idfr = tuple(idfr)
    if idfr in df.columns:
        if unique:
            return df[idfr].unique().tolist()
        else:
            return df[idfr]
    #-------------------------
    idfr_idx_lvl = get_idfr_loc(df=df, idfr=idfr)
    assert(idfr_idx_lvl[1])
    idfr_idx_lvl = idfr_idx_lvl[0]
    if unique:
        return df.index.get_level_values(idfr_idx_lvl).unique().tolist()
    else:
        return df.index.get_level_values(idfr_idx_lvl)
    

#--------------------------------------------------    
def prep_df_for_merge(
    df, 
    merge_on, 
    inplace=False, 
    return_reset_idx_called=False
):
    r"""
    Helper function for new version of merge_rcpx_with_eemsp.
    This will locate where the merge_on are in df, i.e., if in columns of indices.
    If any are in index, .reset_index() will be called on df.
        All index levels are ensured to have an identifiable name before reset_index is called
        
    Returns: df, merge_on
        Where df may possibly be modified (reset_index called) and merge_on will content the full and 
          correct columns to merge on.
    """
    #-------------------------
    if not inplace:
        df = df.copy()
    #-------------------------
    assert(isinstance(merge_on, list))
    #-------------------------
    merge_on_locs = [get_idfr_loc(df, x) for x in merge_on]
    #-------------------------
    # Determine whether any of the data to be used in the merge come from the index
    # Reason: When merging, it is easier to have all the merging fields in the index of both
    #           dfs or in the columns of both dfs, but not a mixture.
    #         e.g., if left_index=True and right_on=[col_1, ..., col_m], the merged df will have
    #           index equal to [0, n-1], i.e., the index of df_1 will not be included in final result    
    merge_needs_idx = any([True if x[1]==True else False for x in merge_on_locs])
    
    #-------------------------
    # If any merge_on are found in index, .reset_index() is going to be called.
    # To make life easier (by making the indices traceable/identifiable), make sure all index levels have a name
    # If an index level does not have a name, it will be named f'idx_{level}'
    # Before calling reset_index, if any of the index level names is already found in the columns (probably shouldn't 
    #   happen, but can) all of the index names will be given an random suffix
    reset_idx_called = False
    if merge_needs_idx:
        # Name any unnamed
        df.index.names = [name_i if name_i else f'idx_{i}' for i,name_i in enumerate(df.index.names)]
        # Add random suffix if needed
        if any([name_i in df.columns for name_i in df.index.names]):
            rand_pfx = Utilities.generate_random_string(str_len=4)
            df.index.names = [f"{name_i}_{rand_pfx}" for name_i in df.index.names]
        #-------------------------
        # Update merge_on using merge_on_locs:
        #   NOTE: Those with merge_on_locs[i][1]==True are found in index, others are not
        #   For those found in columns, the column (i.e., merge_on_locs[i][0]) is simply used
        #   For those found in index, the index level (=merge_on_locs[i][0]) name is used
        merge_on = [df.index.names[x[0]] if x[1]==True else x[0] 
                    for x in merge_on_locs]
        #-------------------------
        # As promised, reset the index
        df = df.reset_index()
        reset_idx_called = True
    else:
        merge_on = [x[0] for x in merge_on_locs]

    # If df has MultiIndex columns, make sure merge_on contains the full MutliIndex column name
    #     When df has MultiIndex columns and merge_needs_idx was run above, this procedure is
    #       explicitly necessary (e.g., when n_levels=2, after reset_index, idx_1 ==> (idx_1, '') column)
    #     This will also be necessary if user lazilly inputs merge_col_i when, e.g., in reality the 
    #       full columns is (merge_needs_idx, '')
    merge_on = [find_single_col_in_multiindex_cols(df=df, col=x) for x in merge_on]

    # Make sure all columns are found
    assert(all([x in df.columns.tolist() for x in merge_on]))
    
    #-------------------------
    if return_reset_idx_called:
        return df, merge_on, reset_idx_called
    else:
        return df, merge_on

#--------------------------------------------------
def is_df_index_simple(
    df, 
    assert_normal=False
):
    r"""
    A simple index is single-leveled with values between 0 and df.shape[0]-1 AND no repeat indices
    If assert_normal==True, then index must be 0, 1, 2, 3, ..., df.shape[0]-2, df.shape[0]-1 or else False will be returned
    """
    #-------------------------
    if df.index.nlevels>1:
        return False
    #-------------------------
    if set(df.index).symmetric_difference(set(range(df.shape[0]))) != set():
        return False
    #-------------------------
    if(
        assert_normal and 
        not pd.Index(range(df.shape[0])).equals(df.index)
    ):
        return False
    #-------------------------
    return True

#--------------------------------------------------    
def merge_dfs(
    df_1, 
    df_2, 
    merge_on_1, 
    merge_on_2, 
    how='inner', 
    final_index=None, 
    dummy_col_levels_prefix = 'dummy_lvl'
):
    r"""
    A tweak to pd.merge.
    This makes is easier to merge on any mixture of indices and/or columns
    
    merge_on_1/_2:
        See get_idfr_loc for explanation!

    final_index:
        Note: for this more flexible merge to work, df_1 and df_2 may have .reset_index called on them. 
        This determines what the output index in the returned pd.DataFrame object will be.
        Acceptable values = [None, '1', 1, '2', 2, 'merged']
        None:
            Results returned as they are without any modification
        1/'1':
            The returned pd.DataFrame will have the same index as the input df_1
        2/'2':
            The returned pd.DataFrame will have the same index as the input df_2
        'merged':
            The index in the returned pd.DataFrame will be equal to the merged columns/indices
    """
    #----------------------------------------------------------------------------------------------------
    assert(final_index in [None, '1', 1, '2', 2, 'merged'])
    #-------------------------
    df_1 = df_1.copy()
    df_2 = df_2.copy()
    #-------------------------
    if isinstance(merge_on_1, list) or isinstance(merge_on_2, list):
        assert(isinstance(merge_on_1, list) and isinstance(merge_on_2, list))
        assert(len(merge_on_1)==len(merge_on_2))
    else:
        # prep_df_for_merge wants list inputs
        merge_on_1 = [merge_on_1]
        merge_on_2 = [merge_on_2]
    #-------------------------
    # Grab original index names, in case need restored at end
    idx_names_1_og = df_1.index.names
    idx_names_2_og = df_2.index.names
    #-------------------------
    # Life is much easier when all index levels have names, so name any unnamed
    df_1.index.names = [name_i if name_i else f'idx_{i}' for i,name_i in enumerate(df_1.index.names)]
    df_2.index.names = [name_i if name_i else f'idx_{i}' for i,name_i in enumerate(df_2.index.names)]
    #-----
    idx_names_1 = df_1.index.names
    idx_names_2 = df_2.index.names
    
    #----------------------------------------------------------------------------------------------------
    # Prep the dfs (1)
    df_1, merge_on_1, reset_idx_called_1 = prep_df_for_merge(
        df                      = df_1, 
        merge_on                = merge_on_1,
        inplace                 = True, 
        return_reset_idx_called = True
    )
    if reset_idx_called_1:
        idx_names_1 = [find_single_col_in_multiindex_cols(df=df_1, col=x) for x in idx_names_1]
    #-----
    df_2, merge_on_2, reset_idx_called_2 = prep_df_for_merge(
        df       = df_2, 
        merge_on = merge_on_2,
        inplace  = True, 
        return_reset_idx_called=True
    )
    if reset_idx_called_2:
        idx_names_2 = [find_single_col_in_multiindex_cols(df=df_2, col=x) for x in idx_names_2]    

    #----------------------------------------------------------------------------------------------------
    # Prep the dfs (2)
    # In order to merge, df_1 and df_2 must have the same number of levels of columns
    if dummy_col_levels_prefix is None:
        dummy_col_levels_prefix = Utilities.generate_random_string(str_len=4, letters='letter_only')
    #-----
    if df_1.columns.nlevels != df_2.columns.nlevels:
        if df_1.columns.nlevels > df_2.columns.nlevels:
            n_levels_to_add = df_1.columns.nlevels - df_2.columns.nlevels
            #-----
            df_2 = prepend_levels_to_MultiIndex(
                df=df_2, 
                n_levels_to_add=n_levels_to_add, 
                dummy_col_levels_prefix=dummy_col_levels_prefix
            )
            #-----
            # Get new MultiIndex versions of merge_on_2
            merge_on_2 = [find_single_col_in_multiindex_cols(df=df_2, col=x) for x in merge_on_2]
        elif df_1.columns.nlevels < df_2.columns.nlevels:
            n_levels_to_add = df_2.columns.nlevels - df_1.columns.nlevels
            #-----
            df_1 = prepend_levels_to_MultiIndex(
                df=df_1, 
                n_levels_to_add=n_levels_to_add, 
                dummy_col_levels_prefix=dummy_col_levels_prefix
            )
            #-----
            # Get new MultiIndex versions of merge_on_1
            merge_on_1 = [find_single_col_in_multiindex_cols(df=df_1, col=x) for x in merge_on_1]        
        else:
            assert(0)
        
    #----------------------------------------------------------------------------------------------------
    # Perform the merge
    df_1, df_2 = make_df_col_dtypes_equal(
        df_1              = df_1, 
        col_1             = merge_on_1, 
        df_2              = df_2, 
        col_2             = merge_on_2, 
        allow_reverse_set = True, 
        assert_success    = True, 
        inplace           = True
    )
    assert(
        df_1[merge_on_1].dtypes.tolist() ==
        df_2[merge_on_2].dtypes.tolist()
    )
    #-----
    return_df = pd.merge(
        df_1, 
        df_2, 
        left_on  = merge_on_1, 
        right_on = merge_on_2, 
        how      = how
    )
    
    #----------------------------------------------------------------------------------------------------
    if final_index==1 or final_index=='1':
        if return_df.index.names != idx_names_1:
            if set(return_df.index.names) == set(idx_names_1):
                # Same names, different order
                return_df = return_df.reorder_levels(idx_names_1)
            else:
                assert(set(idx_names_1).difference(set(return_df.columns.tolist()))==set())
                drop_idx = is_df_index_simple(df=return_df, assert_normal=False)
                return_df = return_df.reset_index(
                    drop      = drop_idx, 
                    col_level = -1
                ).set_index(idx_names_1)
        assert(return_df.index.names == idx_names_1)
        return_df.index.names = idx_names_1_og
    #--------------------------------------------------
    elif final_index==2 or final_index=='2':
        if return_df.index.names != idx_names_2:
            if set(return_df.index.names) == set(idx_names_2):
                # Same names, different order
                return_df = return_df.reorder_levels(idx_names_2)
            else:
                assert(set(idx_names_2).difference(set(return_df.columns.tolist()))==set())
                drop_idx = is_df_index_simple(df=return_df, assert_normal=False)
                return_df = return_df.reset_index(
                    drop      = drop_idx, 
                    col_level = -1
                ).set_index(idx_names_2)
        assert(return_df.index.names == idx_names_2)
        return_df.index.names = idx_names_2_og
    #--------------------------------------------------
    elif final_index=='merged':
        return_df = return_df.set_index(merge_on_1)
    #--------------------------------------------------
    else:
        assert(final_index is None)
    
    #----------------------------------------------------------------------------------------------------
    # Resolve any ugly resulting index names, e.g., [('outg_rec_nb', ''), ('trsf_pole_nb', '')]
    #   i.e., if name_i is a tuple where 0th element is not empty but all other are, then change
    #         name to 0th element
    fnl_idx_names = []
    for idx_name_i in return_df.index.names:
        if(
            isinstance(idx_name_i, tuple) and 
            idx_name_i[0] and
            not any([True if idx_name_i[i] else False for i in range(1, len(idx_name_i))])
        ):
            fnl_idx_names.append(idx_name_i[0])
        else:
            fnl_idx_names.append(idx_name_i)
    return_df.index.names = fnl_idx_names

    #----------------------------------------------------------------------------------------------------
    # NOTE: If merge_on_1 and merge_on_2 columns are the same, after merge only one will remain, and 
    #       do not want to drop in such a case!
    # NOTE: It would be mostly safe to call: return_df.drop(columns=list(set(merge_on_2).difference(merge_on_1)))
    #       However, this method is safest
    cols_to_drop=[]
    for i_col in range(len(merge_on_1)):
        if merge_on_2[i_col] != merge_on_1[i_col]:
            cols_to_drop.append(merge_on_2[i_col])
    #-----
    # Depending on how index was set above, some of cols_to_drop may no tbe present anymore
    cols_to_drop = [x for x in cols_to_drop if x in return_df.columns.tolist()]
    if cols_to_drop:
        return_df = return_df.drop(columns=cols_to_drop)
    
    #----------------------------------------------------------------------------------------------------
    return return_df
    
    
#----------------------------------------------------------------------------------------------------
def concat_dfs(
    dfs                  , 
    axis                 = 0, 
    make_col_types_equal = False
):
    r"""
    Previously, one could simply call pd.concat(dfs).
    However, Pandas now generates the below FutureWarning when an element of list fed to pd.concat is empty.
        FutureWarning: The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty 
          items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.
    In most all of our applications, this should not have any consequential effect.
    However, this function is now supplied to eliminate the annoying warning message from popping up everywhere.
    I have tried many solutions, the one that seems to work most times is trying to ensure all dfs have the same data types!

    make_col_types_equal:
        If there are many pd.DataFrame objects in dfs, this should be set to False!
        This will make the concatenation MUCH slower for long lists!!!!!
    """
    #-------------------------
    dfs_to_concat = [
        x for x in dfs 
        if x is not None and x.shape[0]>0
    ]
    #-------------------------
    if make_col_types_equal:
        # I don't think this necessarily guarantees all will have the same dtypes,
        #   but in most all cases this will do the trick
        for i in range(len(dfs_to_concat)):
            for j in range(i+1, len(dfs_to_concat)):
                ovrlp_cols = list(set(dfs_to_concat[i].columns.tolist()).intersection(set(dfs_to_concat[j].columns.tolist())))
                try:
                    df_i, df_j = make_df_col_dtypes_equal(
                        df_1              = dfs_to_concat[i], 
                        col_1             = ovrlp_cols, 
                        df_2              = dfs_to_concat[j], 
                        col_2             = ovrlp_cols, 
                        allow_reverse_set = True, 
                        assert_success    = True
                    )
                except:
                    df_i, df_j = dfs_to_concat[i], dfs_to_concat[j]
                dfs_to_concat[i] = df_i
                dfs_to_concat[j] = df_j
    #-------------------------
    return_df = pd.concat(
        dfs_to_concat, 
        axis = axis
    )
    return return_df


#----------------------------------------------------------------------------------------------------
def concat_dfs_in_dir(
    dir_path             , 
    regex_pattern        = None, 
    ignore_case          = False, 
    ext                  = '.pkl', 
    make_col_types_equal = False, 
    return_paths         = False
):
    r"""
    """
    #--------------------------------------------------
    assert(os.path.isdir(dir_path))
    #-------------------------
    if ext[0] != '.':
        ext = '.' + ext
    #-----
    accptbl_exts  = ['.pkl', '.csv']
    assert(ext in accptbl_exts)
    #--------------------------------------------------
    files_in_dir = [
        os.path.join(dir_path, x) for x in os.listdir(dir_path) 
        if (
            os.path.isfile(os.path.join(dir_path, x)) and
            os.path.splitext(x)[1] == ext
        )
    ]
    assert(len(files_in_dir) > 0)
    #-------------------------
    if regex_pattern is not None:
        files_in_dir = Utilities.find_in_list_with_regex(
            lst           = files_in_dir, 
            regex_pattern = regex_pattern, 
            ignore_case   = ignore_case    
        )
        assert(len(files_in_dir) > 0)
    #--------------------------------------------------
    dfs = []
    for path_i in files_in_dir:
        if ext=='.pkl':
            df_i = pd.read_pickle(path_i)
        elif ext=='.csv':
            df_i = pd.read_csv(path_i)
        else:
            assert(0)
        #-------------------------
        dfs.append(df_i)
    #-------------------------
    return_df = concat_dfs(
        dfs                  = dfs, 
        axis                 = 0, 
        make_col_types_equal = make_col_types_equal
    )
    #-------------------------
    if return_paths:
        return return_df, files_in_dir
    return return_df
    
    
#----------------------------------------------------------------------------------------------------
def get_true_block_begend_idxs_in_srs(
    srs, 
    return_expanded=False
):
    r"""
    Identify continuous blocks of true in pd.Series object srs.
    Returns a list of lists, block_begend_idxs = [block_begend_idxs_0, block_begend_idxs_1, ..., block_begend_idxs_n]
        where, e.g., block_begend_idxs_0 contains the index locations (think iloc) of the beginning and ending locations of that block.
        THIS IS INCLUSIVE, so (4,6) means elements 4, 5, and 6 all belong to the block

    return_expanded:
        If True, expand out the block intervals
        ==> each element of the returned list of lists (block_begend_idxs) will contain all indices which belong to that particular block 
              (instead of the starting and ending points)

    ----- EXAMPLE -----
        srs = pd.Series([True, False, False, True, True, True, True, True, False, False, True, True])
        -----
        return_expanded==False:    Return block_begend_idxs = [(0, 0), (3, 7), (10, 11)]
        return_expanded==True:     Return block_idxs        = [[0], [3, 4, 5, 6, 7], [10, 11]]
          
    """
    #--------------------------------------------------
    assert(
        isinstance(srs, pd.Series) and 
        srs.dtype==bool
    )
    #-------------------------
    # Build tmp_df from srs, and populate new tmp_diff_col, which will be used to identify blocks
    tmp_df = srs.to_frame().copy()
    col    = tmp_df.columns[0]
    #-----
    tmp_diff_col = Utilities.generate_random_string()
    tmp_df[tmp_diff_col] = tmp_df[col].astype(int).diff()
    #-----
    tmp_diff_col_idx = find_idxs_in_highest_order_of_columns(df=tmp_df, col=tmp_diff_col, exact_match=True, assert_single=True)
    
    #-------------------------
    # The .diff() operation always leaves the first element as a NaN
    # However, if the first element value is True, then it represents the beginning
    #   of a block and the diff should be +1
    if tmp_df.iloc[0][col]==True:
        tmp_df.iloc[0, tmp_diff_col_idx] = 1
    
    #--------------------------------------------------
    # Continuous blocks of True begin with diff = +1 and end at the element preceding diff = -1
    block_beg_idxs = tmp_df.reset_index().index[tmp_df[tmp_diff_col]==1].tolist()
    #-----
    block_end_idxs = tmp_df.reset_index().index[tmp_df[tmp_diff_col]==-1].tolist()
    block_end_idxs = [x-1 for x in block_end_idxs]
    # If the last entry is True, the procedure above will miss the last end idx
    #   If single point above threshold at end of data ==> tmp_diff_col = +1
    #   If multiple points above threshold at the end of the data, then tmp_diff_col for the last value will be 0
    #     In this case, there does not exist a tmp_diff_col=-1 to signal the end of the block, so must add by hand
    if tmp_df.iloc[-1][col]==1:
        block_end_idxs.append(tmp_df.shape[0]-1)
    #--------------------------------------------------
    # periods of length one should have idx in both block_beg_idxs and block_end_idxs
    len_1_pd_idxs = natsorted(set(block_beg_idxs).intersection(set(block_end_idxs)))
    #-------------------------
    # Remove the len_1 idxs so the remainders can be matched
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
    
    #--------------------------------------------------
    if return_expanded:
        block_idxs = [list(range(x[0], x[1]+1)) for x in block_begend_idxs]
        return block_idxs
    else:
        return block_begend_idxs
    
#----------------------------------------------------------------------------------------------------
def get_continuous_blocks_in_df(
    df, 
    col_idfr, 
    data_sep=pd.Timedelta('15min'), 
    return_endpoints=True
):
    r"""
    Given some data, find blocks which are sequential.
    The data are contained in df, and identifed by col_idfr (whether the data are in a column or the index.  See
      Utilities_df.get_idfr_loc for more information).
    The data are assumed to be separated by data_sep.
    The function finds continuous blocks by identifying locations where the difference between two datapoints is
      not equal to data_sep
      
    df:
        Input pd.DataFrame
    col_idfr:
        Location of data in df to use for block identification.
        See Utilities_df.get_idfr_loc for more information
    data_sep:
        expected separation of sequential events in data
        e.g., for 15-minute AMI data, data_sep=set pd.Timedelta('15min')
        e.g., when looking for sequential data in integer index values, set data_sep=1
    
    return_endpoints:
        True:   return only the endpoints of the blocks.
                These endpoints are INCLUSIVE!
        False:  return all indices in the blocks
        EXAMPLE:
            block_begend_idxs = [(0, 3), (4,5), ]
    """
    #--------------------------------------------------
    # In order for method to work, col_idfr information must be in column, not index.
    # This is not absolutely necessary, but it makes life easier
    # If found in index, call reset_index
    col_idfr_loc = get_idfr_loc(
        df=df, 
        idfr=col_idfr
    )
    #-------------------------
    # If found in the index, call reset_index (making sure needed level has a name so it can be identified easily
    #   in the resultant pd.DataFrame)
    if col_idfr_loc[1]:
        if df.index.names[col_idfr_loc[0]]:
            sequence_col = df.index.names[col_idfr_loc[0]]
        else:
            sequence_col = 'sequence_col_'+Utilities.generate_random_string(str_len=4)
            assert(sequence_col not in df.columns.tolist())
            assert(sequence_col not in list(df.index.names))
            df.index = df.index.set_names(sequence_col, level=col_idfr_loc[0])
        #-----
        idx_cols = list(df.index.names)
        df = df.reset_index()
    else:
        sequence_col = col_idfr_loc[0]
        idx_cols=None
    #-------------------------
    assert(sequence_col in df.columns.tolist())
    #--------------------------------------------------
    # This is intended for use when sequence_col contains only unique values!
    assert(df[sequence_col].nunique()==df.shape[0])
    # Also, the data should be sorted!
    df = df.sort_values(by=sequence_col, key=natsort_keygen(), ascending=True)
    #--------------------------------------------------
    # Generate difference column, then alter column to contain whether or not difference
    #   is equal to data_sep (expected separation of sequence events in data)
    tmp_diff_col = Utilities.generate_random_string()
    df[tmp_diff_col] = df[sequence_col].diff()
    #-------------------------
    # Overly cautious finding tmp_diff_col_idx instead of just setting equal to df.shape[1]-1
    #   but, safest I suppose
    tmp_diff_col_idx = find_idxs_in_highest_order_of_columns(
            df            = df, 
            col           = tmp_diff_col, 
            exact_match   = True, 
            assert_single = True
        )
    # Replace NaN in 0th element with data_sep because always beginning of first block
    df.iloc[0, tmp_diff_col_idx] = data_sep
    #-------------------------
    # Don't care about the explicit values in tmp_diff_col, just whether or not they equal data_sep
    df[tmp_diff_col] = (df[tmp_diff_col]==data_sep)
    #-------------------------
    # New blocks begin when df[tmp_diff_col]==False (and, the first block, which begins at 0)
    # NOTE: Want index locations, not values (i.e., want values beteween 0 and df.shape[0]-1)
    #       As such, .reset_index is utilized below
    block_beg_idxs = [0]+df.reset_index().index[df[tmp_diff_col]==False].tolist()
    #-------------------------
    # The ending of block i equals the beginning of block (i+1) minus 1
    # The ending of the last block is simply df.shape[0]-1
    block_end_idxs = [x-1 for x in block_beg_idxs][1:] + [df.shape[0]-1]
    #-------------------------
    assert(len(block_beg_idxs)==len(block_end_idxs))
    block_begend_idxs = list(zip(block_beg_idxs, block_end_idxs))
    block_idxs = [list(range(block_beg_i, block_end_i+1)) for block_beg_i, block_end_i in block_begend_idxs]
    #--------------------------------------------------
    # If .reset_index was called earlier, set index back to original values
    if idx_cols is not None:
        df = df.set_index(idx_cols)
    #--------------------------------------------------
    if return_endpoints:
        return block_begend_idxs
    else:
        return block_idxs
        
        

#----------------------------------------------------------------------------------------------------
def get_overlapping_blocks_in_df(
    df, 
    intrvl_beg_col       = 'DT_OFF_TS_FULL', 
    intrvl_end_col       = 'DT_ON_TS', 
    return_endpoints     = True
):
    r"""
    Given some pd.DataFrame, find blocks of overlapping data.
    Returns:  tuple = (df, return_list)
        where df is the input df sorted by [intrvl_beg_col, intrvl_end_col], as required by the procedure.
              return_list contains the found blocks, and its form is described below under "return_endpoints"
        
    
    return_endpoints:
        True:   return_list contains only the endpoints of the blocks.
                These endpoints are INCLUSIVE!
        False:  return_list contains all indices in the blocks
    """
    #--------------------------------------------------
    # If only one entry, cannot possibly be ovelaps!
    if df.shape[0]<=1:
        return []
    #--------------------------------------------------
    #--------------------------------------------------
    outg_intrvl_i_col     = Utilities.generate_random_string()
    outg_intrvl_im1_col   = Utilities.generate_random_string()
    outg_intrvl_ip1_col   = Utilities.generate_random_string()
    #-----
    overlaps_im1_col      = Utilities.generate_random_string()
    overlaps_ip1_col      = Utilities.generate_random_string()
    overlaps_col          = Utilities.generate_random_string()
    #-----
    idx_col               = Utilities.generate_random_string()
    #-----
    tmp_addtnl_cols = [outg_intrvl_i_col, outg_intrvl_im1_col, outg_intrvl_ip1_col, overlaps_im1_col, overlaps_ip1_col, overlaps_col, idx_col]
    assert(set(df.columns.tolist()).intersection(set(tmp_addtnl_cols))==set())
    
    #--------------------------------------------------
    # df must be properly sorted for the functionality to behave as expected
    df = df.sort_values(by=[intrvl_beg_col, intrvl_end_col])
    #-----
    df[idx_col] = range(df.shape[0])
    
    #-------------------------
    # Set outg_intrvl_i_col, outg_intrvl_im1_col, outg_intrvl_ip1_col
    df[outg_intrvl_i_col]   = df.apply(lambda x: pd.Interval(x[intrvl_beg_col], x[intrvl_end_col]), axis=1)
    df[outg_intrvl_im1_col] = df[outg_intrvl_i_col].shift(1)
    df[outg_intrvl_ip1_col] = df[outg_intrvl_i_col].shift(-1)
    
    #-------------------------
    # Set dummy values for first and last rows simply so df.apply operation below can run without issue
    #   first row --> dummy outg_intrvl_im1
    #   last row  --> dummy outg_intrvl_ip1
    #-----
    outg_intrvl_im1_col_idx = find_idxs_in_highest_order_of_columns(df=df, col=outg_intrvl_im1_col, exact_match=True, assert_single=True)
    outg_intrvl_ip1_col_idx = find_idxs_in_highest_order_of_columns(df=df, col=outg_intrvl_ip1_col, exact_match=True, assert_single=True)
    #-----
    df.iloc[0,  outg_intrvl_im1_col_idx] = pd.Interval(pd.Timestamp.min, pd.Timestamp.min)
    df.iloc[-1, outg_intrvl_ip1_col_idx] = pd.Interval(pd.Timestamp.max, pd.Timestamp.max)
    
    #-------------------------
    # Set overlaps_im1_col and overlaps_ip1_col, containing:
    #   overlaps_im1_col: whether or not a row overlaps with the previous neighbor
    #   overlaps_ip1_col: whether or not a row overlaps with the following neighbor
    df[overlaps_im1_col] = df.apply(lambda x: x[outg_intrvl_i_col].overlaps(x[outg_intrvl_im1_col]), axis=1)
    df[overlaps_ip1_col] = df.apply(lambda x: x[outg_intrvl_i_col].overlaps(x[outg_intrvl_ip1_col]), axis=1)
    #-----
    df[overlaps_col]     = df[overlaps_im1_col]|df[overlaps_ip1_col]
    
    #--------------------------------------------------
    # Get overlap groups
    #-----
    # With the overlaps_im1_col, overlaps_ip1_col, and overlaps_col columns:
    #     overlaps_col==True   ==>  belongs to block
    #     overlaps_col==False  ==>  does not belong to any block
    #     
    #     Beginning of blocks:
    #         overlaps_im1_col = False
    #         overlaps_ip1_col = True
    #         (overlaps_col    = True)
    #
    #     End of blocks:
    #         overlaps_im1_col = True
    #         overlaps_ip1_col = False
    #         (overlaps_col    = True)
    #-----
    block_beg_idxs = df[
        (df[overlaps_im1_col] == False) & 
        (df[overlaps_ip1_col] == True) & 
        (df[overlaps_col]     == True)
    ][idx_col].values.tolist()
    #-------------------------
    block_end_idxs = df[
        (df[overlaps_im1_col] == True) & 
        (df[overlaps_ip1_col] == False) & 
        (df[overlaps_col]     == True)
    ][idx_col].values.tolist()
    #-------------------------
    assert(len(block_beg_idxs)==len(block_end_idxs))
    #-----
    block_begend_idxs = list(zip(block_beg_idxs, block_end_idxs))
    block_idxs        = [list(range(block_beg_i, block_end_i+1)) for block_beg_i, block_end_i in block_begend_idxs]
    #--------------------------------------------------
    if return_endpoints:
        return_list = block_begend_idxs
    else:
        return_list = block_idxs
    #--------------------------------------------------
    df = df.drop(columns=tmp_addtnl_cols)
    #--------------------------------------------------
    return df, return_list


#--------------------------------------------------
def find_and_append_overlapping_blocks_ids_in_df(
    df, 
    intrvl_beg_col       = 'DT_OFF_TS_FULL', 
    intrvl_end_col       = 'DT_ON_TS', 
    overlaps_col         = 'overlaps', 
    overlap_grp_col      = 'overlap_grp', 
    return_overlaps_only = False, 
    no_overlap_grp_val   = -1 # sensible values = -1 or np.nan
):
    r"""
    Given some pd.DataFrame, find blocks of overlapping data.
    Returns:  tuple = (df, return_list)
        where df is the input df sorted by [intrvl_beg_col, intrvl_end_col], as required by the procedure.
              return_list contains the found blocks, and its form is described below under "return_endpoints"
        
    
    return_endpoints:
        True:   return_list contains only the endpoints of the blocks.
                These endpoints are INCLUSIVE!
        False:  return_list contains all indices in the blocks
    """
    #--------------------------------------------------
    # If only one entry, cannot possibly be ovelaps!
    if df.shape[0]<=1:
        df[overlaps_col]    = False
        df[overlap_grp_col] = no_overlap_grp_val
        if return_overlaps_only:
            return None
        else:
            return df
    #--------------------------------------------------
    df, block_idxs = get_overlapping_blocks_in_df(
        df                   = df, 
        intrvl_beg_col       = intrvl_beg_col, 
        intrvl_end_col       = intrvl_end_col, 
        return_endpoints     = False
    )
    
    #--------------------------------------------------
    # If no overlaps, return
    if len(block_idxs)==0:
        df[overlaps_col]    = False
        df[overlap_grp_col] = no_overlap_grp_val
        if return_overlaps_only:
            return None
        else:
            return df
    
    #--------------------------------------------------
    df[overlaps_col]    = False
    df[overlap_grp_col] = no_overlap_grp_val
    #-----
    overlaps_col_idx      = find_idxs_in_highest_order_of_columns(df=df, col=overlaps_col,     exact_match=True, assert_single=True)
    overlap_grp_col_idx   = find_idxs_in_highest_order_of_columns(df=df, col=overlap_grp_col, exact_match=True, assert_single=True)
    #-----
    for i_grp, grp_idxs in enumerate(block_idxs):
        df.iloc[grp_idxs, overlaps_col_idx]    = True
        df.iloc[grp_idxs, overlap_grp_col_idx] = int(i_grp)
    
    #--------------------------------------------------
    # Sanity checks
    if np.isnan(no_overlap_grp_val):
        assert(df[df[overlaps_col]==False][overlap_grp_col].isna().all())
        assert(df[df[overlaps_col]==True][overlap_grp_col].notna().all())
    else:
        assert((df[df[overlaps_col]==False][overlap_grp_col]==no_overlap_grp_val).all())
        assert((df[df[overlaps_col]==True][overlap_grp_col] !=no_overlap_grp_val).all())

        #--------------------------------------------------
        if return_overlaps_only:
            return df[df[overlaps_col]==True]
        else:
            return df

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
def determine_freq_in_df_col(
    df, 
    groupby_SN=True, 
    return_val_if_not_found='error', 
    assert_single_freq=True, 
    assert_expected_freq_found=False, 
    expected_freq=pd.Timedelta('15 minutes'), 
    time_col='starttimeperiod_local', 
    SN_col='serialnumber'
):
    r"""
    Determine the frequency of the data in df[time_col]
    If the df contains multiple serial numbers, IT IS HIGHLY recommended that one groups by serialnumber,
      finding the frequency of each subset, and inferring the overall frequency of the data from that collection.
        In most cases, not grouping the data for, e.g., and outage by serial number will be fine.
        BUT, there are some instances where this will give unexpected results (more info below).
        THUS, to be safe, follow the recommendation above.
        
    return_val_if_not_found:
        This directs what occurs if no frequency is determined.
        Possible values (case insensitive):
            NaN:      return np.nan (which is converted by Python/pandas to NaT)
            expected: return input argument expected_freq
            error:    throw an error
        This is especially useful when finding the frequency for a group of SNs. 
            For this case, return_val_if_not_found is set to NaN within the lambda function call to 
              determine_freq_in_df_col, allowing some flexibility of not every single SN needing to return 
              a frequency for the overall frequency to be determined.
            However, after the groupby method, the overall returned result behaves according to the value of
              return_val_if_not_found, as stated above.
        
    assert_single_freq:
        NOTE: This is only enforced if frequencies are found; i.e., if no frequency is determined, the function
              progressed according to the return_val_if_not_found argument.
        -----
        For a single SN level or groupby_SN==False:
            Self-explanatory; if True, assert only a single frequency is found.
            If False (not recommended), and multiple frequencies are found, take the first
        -----
        For multiple SNs with groupby_SN==True:
            True:
                Enforce a single frequency is found for each SN
                Enforce all SNs share the same frequency
            False:
                Do not enforce single frequency found for each SN (i.e., if multiple are found for a single SN, 
                  the first is returned).
                If the collection of frequencies found from SNs contains multiple values, take the mode (if multiple
                  modes found, take the first one)
                  
    assert_expected_freq_found:
        Assert the frequency found equals that which is expected.
        NOTE: This is only enforced if a frequency is found; i.e., if no frequency is determined, the function
              progressed according to the return_val_if_not_found argument.        
        
    -----
    Potential for unexpected results:
        Consider the following simple example of a pd.DataFrame (ex_df) containing two serial numbers whose times are offset 
          relative to each other by one-minute (BTW, this is not some made-up scenario, this is observed in the data)
                -------------------------
                    serialnumber starttimeperiod_local
                0               1   2023-01-14 20:45:00
                1               2   2023-01-14 20:46:00
                2               1   2023-01-14 21:00:00
                3               2   2023-01-14 21:01:00
                4               1   2023-01-14 21:15:00
                5               2   2023-01-14 21:16:00
                6               1   2023-01-14 21:30:00
                7               2   2023-01-14 21:31:00
                8               1   2023-01-14 21:45:00
                9               2   2023-01-14 21:46:00
                10              1   2023-01-14 22:00:00
                -------------------------                
        Pandas has no idea what the frequency of the data is, and the call 
          freq = pd.infer_freq(ex_df['starttimeperiod_local'].unique()) will return a value of None
        Therefore, one falls back to the secondary method of looking for the most frequent difference in the data (i.e.,
          the method freq=stats.mode(np.diff(natsorted(df[time_col].unique()))).mode[0])
        However, the call np.diff(natsorted(ex_df['starttimeperiod_local'].unique())) will return:
          (NOTE: Actually, to make numbers easy to interpret, the call was: 
                 [pd.Timedelta(x) for x in np.diff(natsorted(ex_df['starttimeperiod_local'].unique()))])
            -------------------------                  
            [Timedelta('0 days 00:01:00'),
             Timedelta('0 days 00:14:00'),
             Timedelta('0 days 00:01:00'),
             Timedelta('0 days 00:14:00'),
             Timedelta('0 days 00:01:00'),
             Timedelta('0 days 00:14:00'),
             Timedelta('0 days 00:01:00'),
             Timedelta('0 days 00:14:00'),
             Timedelta('0 days 00:01:00'),
             Timedelta('0 days 00:14:00')]
            -------------------------
        The example I have chosen actually has two modes (1 minutes and 14 minutes), both of which are incorrect!
        Using scipy.stats.mode, only a single value is returned, and this appears to be the first, i.e. 1 minute
            ==> freq = stats.mode(np.diff(natsorted(ex_df['starttimeperiod_local'].unique()))).mode[0]
                freq = pd.Timedelta(freq)
                GIVES THE RESULT: freq = 1 minute!
        We only expect there to be one frequency of the data, but one could also use statistics.multimode to obtain:
            ==> freq = statistics.multimode(np.diff(natsorted(ex_df['starttimeperiod_local'].unique())))
                freq = [pd.Timedelta(x) for x in freq]
                GIVES THE RESULT: freq = [1 minute, 14 minutes]
                
        ----------
        SOLUTION: Group the data by serial number, and find the frequency wrt each SN
            ------------------------- 
            SN==1
            -----
                serialnumber starttimeperiod_local
            0              1   2023-01-14 20:45:00
            2              1   2023-01-14 21:00:00
            4              1   2023-01-14 21:15:00
            6              1   2023-01-14 21:30:00
            8              1   2023-01-14 21:45:00
            10             1   2023-01-14 22:00:00
            ------------------------- 
            ------------------------- 
            SN==2
            -----
               serialnumber starttimeperiod_local
            1             2   2023-01-14 20:46:00
            3             2   2023-01-14 21:01:00
            5             2   2023-01-14 21:16:00
            7             2   2023-01-14 21:31:00
            9             2   2023-01-14 21:46:00
            ------------------------- 
        ==> freq_1 == freq_2 == 15 minutes! (with pd.infer_freq or secondary method!)
        ----------
    """
    #-------------------------
    if not groupby_SN and df[SN_col].nunique()>1:
        print('Multiple SNs exist in df.\nGrouping by SN is HIGHLY RECOMMENDED!')
    #-------------------------
    return_val_if_not_found = return_val_if_not_found.lower()
    assert(return_val_if_not_found in ['nan', 'expected', 'error'])
    #-------------------------
    # Strategy below will be to develop the code as if being written for case where df has a single SN (or, the case
    #   where groupby_SN==False).
    # Then, if groupby_SN is True, one can call the function from inside a groupby lambda function (with groupby_SN
    #   set to False within the lambda call)
    #----------------------------------------------------------------------------------------------------
    if not groupby_SN:
        # First, try pd.infer_freq method
        # NOTE: infer_freq returns None if no frequency could be determined and raises a TypeError 
        #         if data are not datetime-like and ValueError is few than 3 values
        #       try/except below eliminates errors from being raise, setting freq to None in such cases instead.
        try:
            freq=pd.infer_freq(natsorted(df[time_col].unique()))
            # If freq is, e.g., 1 hour, this will return 'H', whereas pd.Timedelta wants
            #   its input to be '1H' and will complain if fed 'H'
            # Thus, check if string starts with digit.  It it doesn't, prepend '1' to it
            if not freq[0].isdigit():
                freq = '1'+freq
        except:
            freq=None
        #-------------------------
        # If pd.infer_freq was unsuccessful, use the secondary method of looking for the most frequent 
        #   difference in the data
        # NOTE!: Cannot enforce assert_single_freq in the try block below, as if the assertion is tripped this
        #        will simply switch the code to the except block!!!!!
        if freq is None:
            try:
                # NOTE: Below, freq will always be a list
                freq=statistics.multimode(np.diff(natsorted(df[time_col].unique())))
            except:
                freq=[]
            #-----
            if len(freq)==0:
                if return_val_if_not_found=='nan':
                    return np.nan
                elif return_val_if_not_found=='expected':
                    return pd.Timedelta(expected_freq)
                elif return_val_if_not_found=='error':
                    assert(0)
                else:
                    assert(0)
            else:
                # Enforce assertion and/or set freq equal to first value (as 
                #   freq returned from statistics.multimode is a list)
                if assert_single_freq:
                    assert(len(freq)==1)
                freq=freq[0]
        #-------------------------
        freq=pd.Timedelta(freq)
        if assert_expected_freq_found:
            assert(freq==pd.Timedelta(expected_freq))
        #-------------------------
        return freq
    #----------------------------------------------------------------------------------------------------
    # Now, for the more typical case when groupby_SN is True
    # In this case, we essentially call the above methods from within a groupby lambda function
    #-------------------------
    # Get the frequency for each serial number, which will be a pd.Series object
    # NOTE: groupby_SN MUST BE FALSE in the lambda function below, otherwise infinite loop!!!!!
    freqs_srs = df.groupby([SN_col]).apply(
        lambda x: determine_freq_in_df_col(
            df=x, 
            groupby_SN=False, 
            assert_single_freq=assert_single_freq, 
            expected_freq=expected_freq, 
            return_val_if_not_found='nan', 
            assert_expected_freq_found=assert_expected_freq_found, 
            time_col=time_col, 
            SN_col=SN_col
        )
    )
    #-------------------------
    freqs_unq_vals = freqs_srs.dropna().unique()
    freqs_modes = statistics.multimode(freqs_unq_vals)
    #-----        
    if len(freqs_modes)==0:
        if return_val_if_not_found=='nan':
            return np.nan
        elif return_val_if_not_found=='expected':
            return pd.Timedelta(expected_freq)
        elif return_val_if_not_found=='error':
            assert(0)
        else:
            assert(0)
    #-----
    if assert_single_freq:
        assert(len(freqs_modes)==1)
    freq=freqs_modes[0]
    freq=pd.Timedelta(freq)
    #-------------------------
    return freq
    
    
#----------------------------------------------------------------------------------------------------
def build_avgs_series(df, cols_of_interest=None, avg_type='normal'):
    # Intended for use on nrc_df from a single NISTResultContainer
    assert(avg_type=='normal' or avg_type=='trim')
    if cols_of_interest is None:
        cols_of_interest = get_numeric_columns(df)
    df = df[cols_of_interest]
    assert(all([is_numeric_dtype(df[col]) for col in df.columns]))
    
    if avg_type=='normal':
        return df.mean(axis=0)
    elif avg_type=='trim':
        return df.apply(lambda x: stats.trim_mean(x, 0.1), axis=0)
    else:
        assert(0)
    return None

#--------------------------------------------------
def build_avgs_df(agg_df, groupby_col, cols_of_interest=None, avg_type='normal'):
    # Intended for use on a aggregate of nrc_dfs from multiple NISTResultContainers
    assert(avg_type=='normal' or avg_type=='trim')
    if cols_of_interest is None:
        cols_of_interest = get_numeric_columns(agg_df)
    assert(all([is_numeric_dtype(agg_df[col]) for col in cols_of_interest]))
    assert(groupby_col not in cols_of_interest)
    agg_df = agg_df[cols_of_interest+[groupby_col]]
    
    avgs_df = pd.DataFrame(dtype=float)
    for group, frame in agg_df.groupby(groupby_col):
        avgs_series = build_avgs_series(frame, cols_of_interest, avg_type)
        # Include the group so rows can be identified
        avgs_series[groupby_col] = group
        avgs_df = avgs_df.append(avgs_series.to_frame().T, ignore_index=True)
    return avgs_df

#----------------------------------------------------------------------------------------------------
def get_default_sort_by_cols_for_comparison(full_default_sort_by_for_comparison, 
                                            df_1=None, df_2=None):
    if df_1 is None and df_2 is None:
        return full_default_sort_by_for_comparison
    #-------------------------
    if df_1 is None:
        shared_cols = df_2.columns.tolist()
    elif df_2 is None:
        shared_cols = df_1.columns.tolist()
    else:
        shared_cols = list(set(df_1.columns).intersection(set(df_2.columns)))
    #-------------------------
    return [x for x in full_default_sort_by_for_comparison 
            if x in shared_cols]

#----------------------------------------------------------------------------------------------------
def get_dfs_overlap(
    df_1, 
    df_2, 
    enforce_eq_cols = False, 
    include_index   = False
):
    r"""
    Return rows shared between df_1 and df_2.
    Only the columns shared between df_1 and df_2 are evaluated and returned.
    ==> If no columns are shared, return pd.DataFrame()
    
    enforce_eq_cols:
        If True, df_1 and df_2 must have the same columns.
        If the columns are not equal, assertion error is thrown.
    """
    #-------------------------
    cols_1 = df_1.columns.tolist()
    cols_2 = df_2.columns.tolist()
    
    #-------------------------
    # df_1 and df_2 must share at least one column!
    if len(set(cols_1).intersection(set(cols_2)))==0:
        return pd.DataFrame()
    
    #-------------------------
    # If enforce_eq_cols is True, df_1 and df_2 must have the same columns
    if(
        enforce_eq_cols and 
        len(set(cols_1).symmetric_difference(set(cols_2)))!=0
    ):
        assert(0)
    
    #-------------------------
    # If include_index, reset the indices so they are included in the comparison
    if include_index:
        assert(df_1.index.nlevels==df_2.index.nlevels)
        #-------------------------
        # Indices must have same names for comparison, if not so, make so
        if df_1.index.names != df_2.index.names:
            df_2.index.names = df_1.index.names
        #-------------------------
        # Make sure none of the index names are found in columns, as this would cause .reset_index(drop=False) to fail
        df_1.index.names = [x if x not in df_1.columns.tolist() else x + '_' + Utilities.generate_random_string(str_len=4, letters='letters_only') for x in df_1.index.names]
        df_2.index.names = [x if x not in df_2.columns.tolist() else x + '_' + Utilities.generate_random_string(str_len=4, letters='letters_only') for x in df_2.index.names]
        #-------------------------
        df_1 = df_1.reset_index(drop=False)
        df_2 = df_2.reset_index(drop=False)

    #-------------------------
    shared_df = pd.merge(
        df_1, 
        df_2, 
        left_on=None, 
        right_on=None, 
        left_index=False, 
        right_index=False, 
        how='inner', 
        indicator=False
    )
    return shared_df

#----------------------------------------------------------------------------------------------------
def get_dfs_overlap_outer(
    df_1, 
    df_2, 
    enforce_eq_cols = False, 
    include_index   = False
):
    r"""
    Return rows shared between df_1 and df_2.
    Only the columns shared between df_1 and df_2 are evaluated and returned.
    ==> If no columns are shared, return pd.DataFrame()
    
    enforce_eq_cols:
        If True, df_1 and df_2 must have the same columns.
        If the columns are not equal, assertion error is thrown.
    """
    #-------------------------
    cols_1 = df_1.columns.tolist()
    cols_2 = df_2.columns.tolist()
    
    #-------------------------
    # df_1 and df_2 must share at least one column!
    if len(set(cols_1).intersection(set(cols_2)))==0:
        return pd.DataFrame()
    
    #-------------------------
    # If enforce_eq_cols is True, df_1 and df_2 must have the same columns
    if(
        enforce_eq_cols and 
        len(set(cols_1).symmetric_difference(set(cols_2)))!=0
    ):
        assert(0)

    #-------------------------
    # If include_index, reset the indices so they are included in the comparison
    if include_index:
        assert(df_1.index.nlevels==df_2.index.nlevels)
        #-------------------------
        # Indices must have same names for comparison, if not so, make so
        if df_1.index.names != df_2.index.names:
            df_2.index.names = df_1.index.names
        #-------------------------
        # Make sure none of the index names are found in columns, as this would cause .reset_index(drop=False) to fail
        df_1.index.names = [x if x not in df_1.columns.tolist() else x + '_' + Utilities.generate_random_string(str_len=4, letters='letters_only') for x in df_1.index.names]
        df_2.index.names = [x if x not in df_2.columns.tolist() else x + '_' + Utilities.generate_random_string(str_len=4, letters='letters_only') for x in df_2.index.names]
        #-------------------------
        df_1 = df_1.reset_index(drop=False)
        df_2 = df_2.reset_index(drop=False)
    
    #-------------------------
    shared_df = pd.merge(
        df_1, 
        df_2, 
        left_on=None, 
        right_on=None, 
        left_index=False, 
        right_index=False, 
        how='outer', 
        indicator=True
    )
    return shared_df
    
#----------------------------------------------------------------------------------------------------
def do_dfs_overlap(
    df_1, 
    df_2, 
    enforce_eq_cols = True, 
    include_index   = False
):
    r"""
    Checks to see whether df_1 and df_2 share any rows.
    If rows shared, return True.
    If no rows shared, return False.
    -----
    Only the columns shared between df_1 and df_2 are evaluated.
    ==> If no columns are shared, return False
    
    enforce_eq_cols:
        If True, df_1 and df_2 must have the same columns.
        If the columns are not equal, False is returned
    """
    #-------------------------
    cols_1 = df_1.columns.tolist()
    cols_2 = df_2.columns.tolist()
    
    #-------------------------
    # If enforce_eq_cols is True, df_1 and df_2 must have the same columns
    if(
        enforce_eq_cols and 
        len(set(cols_1).symmetric_difference(set(cols_2)))!=0
    ):
        #return False
        assert(0)
    
    #-------------------------
    # NOTE: enforce_eq_cols enforced above (possibly in way unique from that in get_dfs_overlap), 
    #       so set to False in get_dfs_overlap below
    shared_df = get_dfs_overlap(
        df_1            = df_1, 
        df_2            = df_2, 
        enforce_eq_cols = False, 
        include_index   = include_index
    )
    if shared_df.shape[0]==0:
        return False
    else:
        return True


#----------------------------------------------------------------------------------------------------
def simple_sort_df(df, sort_by, ignore_index=True, inplace=False, use_natsort=True):
    # If df's index is named:
    #   one can include the index in the sorting simply by including
    #     the index name in sort_by.
    # If df's index is NOT named:
    #   one can include the index in the sorting by including
    #     'index' in sort_by
    #   !!! UNLESS 'index' is for some reason the name of a column,
    #         in which case the column named 'index' will be used for sorting
    #
    # NOTE: If index is included in sorting, ignore_index will be set to False
    #------------------------------------------------------------------------
    if not inplace:
        df = df.copy()    
    #------------------------------------------------------------------------
    include_idx_in_sort = False
    if 'index' in sort_by and 'index' in df.columns:
        print(
        """
        !!!!!!!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!!!!!!!!!!
        sort_by includes 'index', but 'index' is also a name of a column in df.
        The column named 'index' will be used in sorting, not the index of the DataFrame.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        )
    #------------------------------------------------------------------------
    if 'index' in sort_by and 'index' not in df.columns:
        include_idx_in_sort = True
        ignore_index = False
        #-----
        # This is overkill, but ensures the random temporary column name doesn't 
        # match any of the column names already present in df
        tmp_idx_col_for_srt = None
        while tmp_idx_col_for_srt is None:
            rndm_str = generate_random_string()
            if rndm_str not in df.columns:
                tmp_idx_col_for_srt = rndm_str
        #-----
        # Change 'index' in sort_by to tmp_idx_col_for_srt
        sort_by = [tmp_idx_col_for_srt if x=='index' else x for x in sort_by]
        #-----
        # Create tmp_idx_col_for_srt
        df[tmp_idx_col_for_srt] = df.index
        #-----
    #------------------------------------------------------------------------
    if use_natsort:
        df.sort_values(by=sort_by, ignore_index=ignore_index, inplace=True, key=natsort_keygen())
    else:
        df.sort_values(by=sort_by, ignore_index=ignore_index, inplace=True)
    #------------------------------------------------------------------------
    if include_idx_in_sort:
        df.drop(columns=[tmp_idx_col_for_srt], inplace=True)
    return df    
    
def are_sorted_dfs_equal(df1, df2, sort_by, cols_to_compare=None):
    # Just as in simple_sort_dfs...
    #   If df's index is named:
    #     one can include the index in the sorting simply by including
    #       the index name in sort_by.
    #   If df's index is NOT named:
    #     one can include the index in the sorting by including
    #       'index' in sort_by
    #     !!! UNLESS 'index' is for some reason the name of a column,
    #           in which case the column named 'index' will be used for sorting
    #------------------------------------------------------------------------
    assert(all(x in df1.columns for x in sort_by))
    assert(all(x in df2.columns for x in sort_by))
    #------------------------------------------------------------------------
    df1_srtd = simple_sort_df(df1, sort_by, ignore_index=True, inplace=False)
    df2_srtd = simple_sort_df(df2, sort_by, ignore_index=True, inplace=False)
    #------------------------------------------------------------------------
    if cols_to_compare is None:
        cols_to_compare = list(set(df1_srtd.columns).intersection(set(df2_srtd.columns)))
    assert(all(x in df1_srtd.columns for x in cols_to_compare))
    assert(all(x in df2_srtd.columns for x in cols_to_compare))
    result = df1_srtd[cols_to_compare].equals(df2_srtd[cols_to_compare])
    return result
    
#----------------------------------------------------------------------------------------------------
def assert_dfs_equal(df1, df2):
    assert_frame_equal(df1, df2) #from pandas.util.testing

#--------------------------------------------------
#TODO get_dfs_diff methods don't work with MultiIndex (indices or columns, probably)
def get_dfs_diff(df1, df2):
    #df1 and df2 must be identically labelled!
    #  i.e. must have same shape, indices, and columns!
    
    # Outputs diff_df highlighting differences between df1 and df2
    #  indices = MultiIndex = ('row', 'col')
    #  columns = df1_values, df2_values
    
    diff_at = df1!=df2
    # diff_at will be boolean DataFrame indicating whether or not two are equal
    # Note: One property of NaN is NaN != NaN.
    #   So if df1 and df2 both are NaN for a given element,
    #   that element will still be found as unequal.
    #   These will be removed later

    # stack essentially flattens 2D DataFrame to Series with multi-level index
    #   Basically, each row_i is transposed to a Series (i.e. a column vector)
    #   with indices (i, metric_0) to (i, metric_n)
    #   These Series are then stacked on top of each other
    diff_at_1d = diff_at.stack()
    
    # Grab only those where there is a difference (i.e. where diff_at_1d is True)
    changed = diff_at_1d[diff_at_1d]
    changed.index.names = ['row', 'col']

    # Note: In documentation, recommend using DataFrame.to_numpy() over DataFrame.values
    changed_from = df1.to_numpy()[diff_at]
    changed_to = df2.to_numpy()[diff_at]

    diff_df = pd.DataFrame({'df1_values': changed_from, 'df2_values': changed_to}, index=changed.index)

    # Now, to drop rows where df1_values=df1_values=np.nan
    # This will correctly not drop rows if only one is nan
    diff_df = diff_df.dropna(subset=['df1_values', 'df2_values'], how='all')
    
    return diff_df
    
#--------------------------------------------------    
def get_dfs_diff_WEIRD(df1, df2, stack_level=-1):
    # LOOK INTO THIS!!!!!
    #
    #
    #df1 and df2 must be identically labelled!
    #  i.e. must have same shape, indices, and columns!
    
    # TODO LOOK INTO WHY
    # For whatever reason, when time index intervals I had to
    # stack_level=0 (e.g. for df_H_agg)
    
    # Outputs diff_df highlighting differences between df1 and df2
    #  indices = MultiIndex = ('row', 'col')
    #  columns = df1_values, df2_values
    
    diff_at = df1!=df2
    # diff_at will be boolean DataFrame indicating whether or not two are equal
    # Note: One property of NaN is NaN != NaN.
    #   So if df1 and df2 both are NaN for a given element,
    #   that element will still be found as unequal.
    #   These will be removed later

    # stack essentially flattens 2D DataFrame to Series with multi-level index
    #   Basically, each row_i is transposed to a Series (i.e. a column vector)
    #   with indices (i, metric_0) to (i, metric_n)
    #   These Series are then stacked on top of each other
    diff_at_1d = diff_at.stack(level=stack_level)
    #TODO WHY WAS THIS A DATAFRAME INSTEAD OF A SERIES?????
    diff_at_1d = diff_at_1d.squeeze()
    
    # Grab only those where there is a difference (i.e. where diff_at_1d is True)
    changed = diff_at_1d[diff_at_1d]
    changed.index.names = ['row', 'col']

    # Note: In documentation, recommend using DataFrame.to_numpy() over DataFrame.values
    changed_from = df1.to_numpy()[diff_at]
    changed_to = df2.to_numpy()[diff_at]

    diff_df = pd.DataFrame({'df1_values': changed_from, 'df2_values': changed_to}, index=changed.index)

    # Now, to drop rows where df1_values=df1_values=np.nan
    # This will correctly not drop rows if only one is nan
    diff_df = diff_df.dropna(subset=['df1_values', 'df2_values'], how='all')
    
    return diff_df
    
#--------------------------------------------------    
def get_dfs_diff_approx_ok_OLD(df_1, df_2, precision=0.00001, cols_to_compare=None):
    # WARNING: for large dfs this can take a decent amount of time.
    # Get the difference between two DataFrames, 
    # but if values are approximately equal (as decided by precision)
    # then the values are considered equal
    #
    # If cols_to_compare is None, all numeric columns are compared
    # Otherwise, just columns in cols_to_compare are compares
    # If cols_to_compare are supplied, they must all be numeric!
    #
    # As with Utilities_df.get_dfs_diff, two NaN values are considered equal
    #-------------------------------------------------------
    if cols_to_compare is None:
        numeric_cols_1 = get_numeric_columns(df_1)
        numeric_cols_2 = get_numeric_columns(df_2)
        # The columns need not be in the same order, but must be the same
        assert(len(set(numeric_cols_1).symmetric_difference(set(numeric_cols_2)))==0)
        cols_to_compare = numeric_cols_1
    else:
        # Make sure supplied columns are all numeric
        assert(all(is_numeric_dtype(df_1[col]) for col in cols_to_compare))
        assert(all(is_numeric_dtype(df_2[col]) for col in cols_to_compare))
    #-------------------------------------------------------    
    df_1 = df_1[cols_to_compare].copy()
    df_2 = df_2[cols_to_compare].copy()
    n_rows, n_cols = df_1.shape
        
    differences = []
    for i_row in range(n_rows):
        for i_col in range(n_cols):
            val_1 = df_1.iloc[i_row, i_col]
            val_2 = df_2.iloc[i_row, i_col]

            if not Utilities.are_approx_equal_nans_ok(val_1, val_2, precision):
                differences.append((i_row, i_col)) 
    #------------------------------------------------------- 
    if len(differences)==0:
        diff_df = pd.DataFrame(columns=['row', 'col', 'df1_values', 'df2_values'])
        diff_df.set_index(['row', 'col'], inplace=True)
        return diff_df
    #------------------------------------------------------- 
    index = [(list(df_1.index)[row_col[0]], list(df_1.columns)[row_col[1]]) for row_col in differences]
    index = pd.MultiIndex.from_tuples(index) #Just to match output of Utilities_df.get_dfs_diff
    #-----
    values_1 = [df_1.iloc[row_col[0], row_col[1]] for row_col in differences]
    values_2 = [df_2.iloc[row_col[0], row_col[1]] for row_col in differences]
    #-----
    diff_df = pd.DataFrame(data = {'df1_values':values_1, 'df2_values':values_2}, index=index)
    diff_df.index.names = ['row', 'col'] #Again, just to match output of Utilities_df.get_dfs_diff
    #------------------------------------------------------- 
    return diff_df

#--------------------------------------------------    
def get_dfs_diff_approx_ok_numeric(df_1, df_2, precision=0.00001, cols_to_compare=None, sort_by=None):
    # Get the difference between two DataFrames, 
    # but if values are approximately equal (as decided by precision)
    # then the values are considered equal
    #
    # If cols_to_compare is None, all numeric columns are compared
    # Otherwise, just columns in cols_to_compare are compared
    # If cols_to_compare are supplied, they must all be numeric!
    #
    # As with Utilities_df.get_dfs_diff, two NaN values are considered equal
    #
    # See simple_sort_df for explanation of sort_by
    #-------------------------------------------------------
    if cols_to_compare is None:
        numeric_cols_1 = get_numeric_columns(df_1)
        numeric_cols_2 = get_numeric_columns(df_2)
        cols_to_compare = list(set(numeric_cols_1).intersection(set(numeric_cols_2)))
    else:
        # Make sure supplied columns are all numeric
        assert(all(is_numeric_dtype(df_1[col]) for col in cols_to_compare))
        assert(all(is_numeric_dtype(df_2[col]) for col in cols_to_compare))
    #-------------------------------------------------------
    if sort_by is not None:
        df_1 = simple_sort_df(df_1, sort_by, ignore_index=True, inplace=False)
        df_2 = simple_sort_df(df_2, sort_by, ignore_index=True, inplace=False)    
    #-------------------------------------------------------
    df_1 = df_1[cols_to_compare]
    df_2 = df_2[cols_to_compare]
    #-------------------------------------------------------
    diff_df = get_dfs_diff(df_1, df_2)
    diff_df['rel_delta'] = np.abs(diff_df[['df1_values', 'df2_values']].pct_change(axis=1)['df2_values'])
    diff_df = diff_df[diff_df['rel_delta'] > precision]
    return diff_df

#--------------------------------------------------
#TODO probably use get_shared_columns
def get_dfs_diff_approx_ok(df_1, df_2, precision=0.00001, cols_to_compare=None, sort_by=None, return_df_only=False, inplace=False):
    # Get the difference between two DataFrames, 
    # but if values are approximately equal (as decided by precision)
    # then the values are considered equal
    #
    # If cols_to_compare is None, all columns shared by both dfs are compared.
    # Otherwise, just columns in cols_to_compare are compared
    # If cols_to_compare are supplied, they must all be numeric!
    #
    # As with Utilities_df.get_dfs_diff, two NaN values are considered equal
    #
    # See simple_sort_df for explanation of sort_by
    #-------------------------------------------------------
    if sort_by is not None:
        df_1 = simple_sort_df(df_1, sort_by, ignore_index=True, inplace=inplace)
        df_2 = simple_sort_df(df_2, sort_by, ignore_index=True, inplace=inplace)
    #-----
    overlap_cols = list(set(df_1.columns).intersection(set(df_2.columns)))
    #-----
    numeric_cols_1 = get_numeric_columns(df_1)
    numeric_cols_2 = get_numeric_columns(df_2)
    numeric_cols_to_comp = list(set(numeric_cols_1).intersection(set(numeric_cols_2)))
    other_cols_to_comp   = [x for x in overlap_cols if x not in numeric_cols_to_comp]
    #-----
    if cols_to_compare is not None:
        numeric_cols_to_comp = [x for x in numeric_cols_to_comp if x in cols_to_compare]
        other_cols_to_comp   = [x for x in other_cols_to_comp if x in cols_to_compare]
    #-----
    leftout_cols_1 = list(set(df_1.columns).difference(set(numeric_cols_to_comp+other_cols_to_comp)))
    leftout_cols_2 = list(set(df_2.columns).difference(set(numeric_cols_to_comp+other_cols_to_comp)))
    #-----
    numeric_diffs = get_dfs_diff_approx_ok_numeric(df_1, df_2, 
                                                   precision=precision, 
                                                   cols_to_compare=numeric_cols_to_comp, 
                                                   sort_by=None)
    numeric_diffs['is_numeric'] = True
    #-----
    if len(other_cols_to_comp)>0:
        other_diffs = get_dfs_diff(df_1[other_cols_to_comp], df_2[other_cols_to_comp])
        other_diffs['rel_delta'] = '' # To match columns in numeric_diffs
        #-----
        other_diffs['is_numeric']   = False
        assert(all(numeric_diffs.columns==other_diffs.columns))
        return_df = pd.concat([numeric_diffs, other_diffs])
    else:
        return_df = numeric_diffs
    return_df = return_df.sort_index()
    if return_df_only:
        return return_df
    else:
        return_dict = {
            'diffs_df' : return_df, 
            'cols_compared' : numeric_cols_to_comp+other_cols_to_comp,
            'cols_compared_numeric' : numeric_cols_to_comp, 
            'cols_compared_other' : other_cols_to_comp, 
            'leftout_cols_1' : leftout_cols_1, 
            'leftout_cols_2' : leftout_cols_2
        }
        return return_dict
    

#--------------------------------------------------    
def get_dfs_diff_approx_ok_WEIRD(df_1, df_2, precision=0.00001, cols_to_compare=None, sort_by=None, stack_level=-1):
    # TODO ELIMINATE THIS ONCE get_dfs_diff_WEIRD is resolved
    #
    #
    # Get the difference between two DataFrames, 
    # but if values are approximately equal (as decided by precision)
    # then the values are considered equal
    #
    # If cols_to_compare is None, all numeric columns are compared
    # Otherwise, just columns in cols_to_compare are compares
    # If cols_to_compare are supplied, they must all be numeric!
    #
    # As with Utilities_df.get_dfs_diff, two NaN values are considered equal
    #
    # See simple_sort_df for explanation of sort_by
    #-------------------------------------------------------
    if cols_to_compare is None:
        numeric_cols_1 = get_numeric_columns(df_1)
        numeric_cols_2 = get_numeric_columns(df_2)
        # The columns need not be in the same order, but must be the same
        assert(len(set(numeric_cols_1).symmetric_difference(set(numeric_cols_2)))==0)
        cols_to_compare = numeric_cols_1
    else:
        # Make sure supplied columns are all numeric
        assert(all(is_numeric_dtype(df_1[col]) for col in cols_to_compare))
        assert(all(is_numeric_dtype(df_2[col]) for col in cols_to_compare))
    #-------------------------------------------------------
    if sort_by is not None:
        df_1 = simple_sort_df(df_1, sort_by, ignore_index=True, inplace=False)
        df_2 = simple_sort_df(df_2, sort_by, ignore_index=True, inplace=False)    
    #-------------------------------------------------------
    df_1 = df_1[cols_to_compare]
    df_2 = df_2[cols_to_compare]
    #-------------------------------------------------------
    diff_df = get_dfs_diff_WEIRD(df_1, df_2, stack_level=stack_level)
    diff_df['rel_delta'] = np.abs(diff_df[['df1_values', 'df2_values']].pct_change(axis=1)['df2_values'])
    diff_df = diff_df[diff_df['rel_delta'] > precision]
    return diff_df
    
#--------------------------------------------------
def get_nan_rows_and_columns(df, metrics_of_interest):
    # returns a list of dicts 
    #   where each dict has keys   = 'index' (row index in df)
    #                       values = 'metrics' (list of nan metrics)
    nrc_df = df[metrics_of_interest].copy()
    nrc_df['row_has_nan'] = nrc_df.isna().sum(axis=1)>0
    rows_w_nans = nrc_df[nrc_df['row_has_nan']==True]
    #-----
    nans_info = []
    for index, row in rows_w_nans.iterrows():
        #row is a series, hence why we grab the indices here
        #index is the metric name and data is metric value
        nan_row_index = index
        nan_metrics_in_row = row[row.isna()==True].index.tolist()
        nans_info.append({'index':nan_row_index, 'metrics':nan_metrics_in_row})
        
    return nans_info

#--------------------------------------------------
def print_nan_rows_and_columns(df, metrics_of_interest):
    nans_info = get_nan_rows_and_columns(df, metrics_of_interest)
    for nan_row in nans_info:
        print('-'*50)
        print(f'row index = {nan_row["index"]}')
        print('-'*10)
        print('nan metrics = ')
        print(*nan_row["metrics"], sep='\n')
        print('-'*50)
        print()
        
#----------------------------------------------------------------------------------------------------
def w_avg_df_col(df, w_col, x_col):
    r"""
    Compute weighted average of x_col weighted by values in w_col
    NOTE: Parentheses are important here.  Without them, df[x_col] returned trivially
    """
    #-------------------------
    # If there is only one row in df, then no averaging to be done, in which
    # case, simply return the value of df[x_col] value
    if df.shape[0]==1:
        return df[x_col].values[0]
    #-------------------------
    return (df[x_col]*df[w_col]).sum()/df[w_col].sum()

#--------------------------------------------------
def w_avg_df_cols(df, w_col, x_cols=None, include_sum_of_weights=True):
    r"""
    Returns a series.
    Compute weighted average of x_cols (multiple columns) weighted by values in w_col.
    If x_cols is None, all columns except for w_col are used.
    NOTE: Parentheses are important here.  Without them, df[x_col] returned trivially
    
    Built to be used with groupby
      e.g., agg_df = ungrouped_df.groupby(ungrouped_df.index).apply(w_avg_df_cols, w_col)
    """
    #-------------------------
    # If there is only one row in df, then no averaging to be done, in which
    # case, simply return series version of df
    if df.shape[0]==1:
        return df.squeeze()
    #-------------------------
    if x_cols is None:
        x_cols = [x for x in df.columns.tolist() if x!=w_col]
    return_series = df[x_cols].multiply(df[w_col], axis='index').sum().divide(df[w_col].sum())
    if include_sum_of_weights:
        return_series[w_col] = df[w_col].sum()
    return return_series

#--------------------------------------------------
def w_sum_df_cols(df, w_col, x_cols=None, include_sum_of_weights=True):
    r"""
    Returns a series.
    Compute weighted sum of x_cols (multiple columns) weighted by values in w_col.
    If x_cols is None, all columns except for w_col are used.
    
    Built to be used with groupby
      e.g., agg_df = ungrouped_df.groupby(ungrouped_df.index).apply(w_avg_df_cols, w_col)
    """
    #-------------------------
    # If there is only one row in df, then no averaging to be done, in which
    # case, simply return series version of df
    if df.shape[0]==1:
        return df.squeeze()
    #-------------------------
    if x_cols is None:
        x_cols = [x for x in df.columns.tolist() if x!=w_col]
    return_series = df[x_cols].multiply(df[w_col], axis='index').sum()
    if include_sum_of_weights:
        return_series[w_col] = df[w_col].sum()
    return return_series
    

#--------------------------------------------------    
def sum_and_weighted_average_of_df_cols(
    df, 
    sum_x_cols, sum_w_col, 
    wght_x_cols, wght_w_col, 
    include_sum_of_weights=True
):
    r"""
    Returns a series.
    Compute the sum of sum_x_cols (multiple columns), and
    compute the weighted average of wght_x_cols (multiple columns), where wght stands for weighted,
    weighted by values in wght_w_col.
    
    Admittedly a little confusing that there exists a sum weights column.  This is for use when
      include_sum_of_weights is True.  The idea is this will keep track of the number of counts, typically
      serial numbers.  sum_w_col is not actually used to calculate anything, but instead stores the sum of
      weights value calculated for the weighted portion of the DF
      
    There is not intended to be any overlap in sum_x_cols and wght_x_cols.  
    This means, for a given column, only the average or weighted average may be calculated here, not both.  
    If both are desired, call this function twice.
      
    In a typical use case, for which this code was intended:
    sum_x_cols = df.columns[df.columns.get_level_values(0)=='counts']
    sum_w_col = ('counts', '_nSNs')
    
    wght_x_cols = df.columns[df.columns.get_level_values(0)=='counts_norm']
    wght_w_col = ('counts_norm', '_nSNs')
    """
    #-------------------------
    # If there is only one row in df, then no summing or averaging to be done, in which
    # case, simply return series version of df
    if df.shape[0]==1:
        return df.squeeze()
    #-------------------------
    assert(df[sum_w_col].astype(float).equals(df[wght_w_col].astype(float)))
    
    # As mentioned in description, there is not intended to be any overlap in sum_x_cols and wght_x_cols
    # Unfortunately using set operations don't give the expected result when using a list of tuples.
    # Therefore, for MultiIndex column case for which this was originally intended, simply calling 
    # assert(len(set(sum_x_cols).union(set(wght_x_cols)))==0) won't work.  Need to write out long method.
    assert(len([x for x in sum_x_cols if x in wght_x_cols])==0)
    assert(len([x for x in wght_x_cols if x in sum_x_cols])==0)
    #-------------------------
    # The example I gave above is slightly lazy 
    #   (e.g., sum_x_cols = df.columns[df.columns.get_level_values(0)=='counts'], 
    #   in that sum_w_col will be contained in sum_x_cols (and wght_w_col in wght_x_cols)
    # The lines below protect against this laziness/overlooking
    sum_x_cols = [x for x in sum_x_cols if x!=sum_w_col]
    wght_x_cols = [x for x in wght_x_cols if x!=wght_w_col]
    #-------------------------
    wght_series = w_avg_df_cols(
        df=df, 
        w_col=wght_w_col, 
        x_cols=wght_x_cols, 
        include_sum_of_weights=include_sum_of_weights
    )
    #-------------------------
    sum_series = df[sum_x_cols].sum()
    # If include_sum_of_weights, set sum_series[sum_w_col] equal to wght_series[wght_w_col]
    if include_sum_of_weights:
        sum_series[sum_w_col] = wght_series[wght_w_col]
    #-------------------------
    # Combine wght_series and sum_series
    return_series = pd.concat([sum_series, wght_series])
    
    # Order return_series as df is ordered
    order = [x for x in df.columns if x in list(sum_series.index) + list(wght_series.index)]
    return_series = return_series[order]
    #-------------------------
    return return_series


#--------------------------------------------------
def sum_and_weighted_sum_of_df_cols(
    df, 
    sum_x_cols, sum_w_col, 
    wght_x_cols, wght_w_col, 
    include_sum_of_weights=True
):
    r"""
    Returns a series.
    Compute the sum of sum_x_cols (multiple columns), and
    compute the weighted average of wght_x_cols (multiple columns), where wght stands for weighted,
    weighted by values in wght_w_col.
    
    Admittedly a little confusing that there exists a sum weights column.  This is for use when
      include_sum_of_weights is True.  The idea is this will keep track of the number of counts, typically
      serial numbers.  sum_w_col is not actually used to calculate anything, but instead stores the sum of
      weights value calculated for the weighted portion of the DF
      
    THIS IS INTENDED TO BE USED on a df with MultiIndex columns (see 'In a typical use...' case below).
    HOWEVER, one may use it on a DF wit single dimensional columns, BUT, in such a case, one should set
      sum_w_col to None, otherwise, the returned Series will have two idential entries sum_w_col and wght_w_col.
      
    There is not intended to be any overlap in sum_x_cols and wght_x_cols.  
    This means, for a given column, only the average or weighted average may be calculated here, not both.  
    If both are desired, call this function twice.
      
    In a typical use case, for which this code was intended:
    sum_x_cols = df.columns[df.columns.get_level_values(0)=='counts']
    sum_w_col = ('counts', '_nSNs')
    
    wght_x_cols = df.columns[df.columns.get_level_values(0)=='counts_norm']
    wght_w_col = ('counts_norm', '_nSNs')
    """
    #-------------------------
    # If there is only one row in df, then no summing or averaging to be done, in which
    # case, simply return series version of df
    if df.shape[0]==1:
        return df.squeeze()
    #-------------------------
    if sum_w_col:
        assert(df[sum_w_col].astype(float).equals(df[wght_w_col].astype(float)))
    
    # As mentioned in description, there is not intended to be any overlap in sum_x_cols and wght_x_cols
    # Unfortunately using set operations don't give the expected result when using a list of tuples.
    # Therefore, for MultiIndex column case for which this was originally intended, simply calling 
    # assert(len(set(sum_x_cols).union(set(wght_x_cols)))==0) won't work.  Need to write out long method.
    assert(len([x for x in sum_x_cols if x in wght_x_cols])==0)
    assert(len([x for x in wght_x_cols if x in sum_x_cols])==0)
    #-------------------------
    # The example I gave above is slightly lazy 
    #   (e.g., sum_x_cols = df.columns[df.columns.get_level_values(0)=='counts'], 
    #   in that sum_w_col will be contained in sum_x_cols (and wght_w_col in wght_x_cols)
    # The lines below protect against this laziness/overlooking
    if sum_w_col:
        sum_x_cols = [x for x in sum_x_cols if x!=sum_w_col]
    else:
        sum_x_cols = [x for x in sum_x_cols if x!=wght_w_col]
    wght_x_cols = [x for x in wght_x_cols if x!=wght_w_col]
    #-------------------------
    wght_series = w_sum_df_cols(
        df=df, 
        w_col=wght_w_col, 
        x_cols=wght_x_cols, 
        include_sum_of_weights=include_sum_of_weights
    )
    #-------------------------
    sum_series = df[sum_x_cols].sum()
    # If include_sum_of_weights, set sum_series[sum_w_col] equal to wght_series[wght_w_col]
    if include_sum_of_weights and sum_w_col:
        sum_series[sum_w_col] = wght_series[wght_w_col]
    #-------------------------
    # Combine wght_series and sum_series
    return_series = pd.concat([sum_series, wght_series])
    
    # Order return_series as df is ordered
    order = [x for x in df.columns if x in return_series.index]
    return_series = return_series[order]
    #-------------------------
    return return_series
    
    

#----------------------------------------------------------------------------------------------------
def consolidate_column_of_lists(df, col, sort=False, include_None=True, batch_size=None, verbose=False):
    r"""
    Purpose is to reduce the serial numbers column (typically '_SNs') down to the unique SNs.
    Each element in the serial numbers column contains a list of SNs.
    By default, calling .sum() on a column of lists concatenates the lists together.
    The additional set() call eliminates duplicates.
    
    batch_size:
      This allows the process to be performed in batches.  This is very useful if df[col] is
        large and/or the number of elements in each list are long.
      Without much testing, 1000 seems like a decent number when dealing with outages.
        - Take with a grain of salt, as optimal number likely depends on the average number of elements
          per list more than anything.
    
    NOTE: Only works if values are lists, not numpy ndarrays!!!!!
    """
    #-------------------------
    if batch_size is not None:
        batch_idxs = Utilities.get_batch_idx_pairs(df.shape[0], batch_size)
        n_batches = len(batch_idxs)
        partial_SNs=[]
        for i, batch_i in enumerate(batch_idxs):
            if verbose:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            SNs_i = consolidate_column_of_lists(df=df.iloc[i_beg:i_end], col=col, sort=sort, include_None=include_None, batch_size=None)
            partial_SNs.append(SNs_i)
        #-----
        assert(len(partial_SNs)==n_batches)
        return_list = []
        for i in range(n_batches):
            return_list = list(set(return_list+partial_SNs[i]))
    #-------------------------
    else:
        if not isinstance(df.iloc[0][col], list):
            df[col] = df[col].apply(lambda x: list(x))
        #-----
        return_list = list(set(df[col].sum()))
    #-------------------------
    if not sort:
        return return_list
    else:
        # Note: If any elements are None, sort will Fail
        # Therefore, sort only those which are not None, and add None at end
        none_els = [x for x in return_list if x is None]
        return_list = sorted([x for x in return_list if x is not None])
        if include_None:
            return_list = none_els + return_list
        return return_list
  

#--------------------------------------------------  
def consolidate_df_OLD(
    df, 
    groupby_col, 
    cols_shared_by_group, 
    cols_to_collect_in_lists, 
    include_groupby_col_in_output_cols=False, 
    allow_duplicates_in_lists=False, 
    recover_uniqueness_violators=True, 
    rename_cols=None, 
    verbose=False
):
    r"""
    This function consolidates a DF which has a lot of repetitive information.
    More specifically, this is designed for a DF in which groups of the data share a single 
      unique value for many of their columns.
    This function will consolidate the DF into a form where each group is represented by a single row,
      columns for which the group shares a single value will be reproduced as a single value, and element 
      of columns where the group has multiple values will become a list.
    
    groupby_col:
      Designed to work with a single groupby column, but I believe using multiple columns should work as well
      
    include_groupby_col_in_output_cols:
      Default to False.
      By default, when grouping, the groupby column becomes the index for the resultant DF.
      If this is set to True, the index of the resultant DF will be named 'idx' and a column names groupby_col
        will be included in the DF.
        NOTE: This was designed when working with build_end_events_for_no_outages.  When using AMI_SQL it is easiest 
              to have the groupby value reproduced in a column (instead of just as the index) so the value can be 
              included in the SQL query and resultant DF.
      
    allow_duplicates_in_lists:
      Default to False, so lists will only have unique elements.
      e.g., when dealing with Meter Premise data, a single premise may have multiple serial numbers.
        This situation would lead to the meter premise being repeated, which is typically not desired
    """
    #-------------------------
    if cols_shared_by_group is None or len(cols_shared_by_group)==0:
        cols_shared_by_group = groupby_col
    assert(Utilities.is_object_one_of_types(cols_shared_by_group, [list, str, int]))
    if not isinstance(cols_shared_by_group, list):
        cols_shared_by_group=[cols_shared_by_group]
    #-----
    assert(Utilities.is_object_one_of_types(cols_to_collect_in_lists, [list, str, int]))
    if not isinstance(cols_to_collect_in_lists, list):
        cols_to_collect_in_lists=[cols_to_collect_in_lists]
    #-------------------------
    # Only include groups which have one unique entry for each of cols_shared_by_group
    # The below series has a True/False value for each group stating whether (True)
    #   or not (False) all cols in cols_shared_by_group have one unique value
    group_has_unq_val_where_expected = (df.groupby(groupby_col)[cols_shared_by_group].nunique()<=1).all(axis=1)

    # Those not satisfying the uniqueness criterion are collected below
    groups_violating_uniqueness = group_has_unq_val_where_expected[~group_has_unq_val_where_expected.values].index.tolist()
    #-------------------------
    # Reduce down df, keeping only those satisfying uniqueness
    red_df = df[~df[groupby_col].isin(groups_violating_uniqueness)]
    #-------------------------
    # If recover_uniqueness_violators, try to recover those in groups_violating_uniqueness if possible
    #   If empty entries (equal to '' or None) are what is causing the violation of uniquness, then these missing
    #   values will be assumed to be equal to (and alterted to be equal to) the others in the group.
    if recover_uniqueness_violators:
        for violator_group_i in groups_violating_uniqueness:
            violator_df_i = df[df[groupby_col]==violator_group_i].copy()
            # Find the columns in violator_df_i causing the violations
            violating_cols_i = violator_df_i[cols_shared_by_group].columns[violator_df_i[cols_shared_by_group].nunique()>1]

            # In the violating columns, replace any empty values with np.nan to be more easily identified
            violator_df_i[violating_cols_i] = violator_df_i[violating_cols_i].replace(['', None], np.nan)

            # Note: Below, the default behavior of nunique is to ignore NaN values, but I have dropna=True to make
            #       this explicitly clear
            # After replacing missing values with np.nans, if there is no longer a violation, the empty values
            #   are replaced by the unique value for the group's column
            if all(violator_df_i[violating_cols_i].nunique(dropna=True)==1):
                for col in violating_cols_i:
                    violator_df_i[col] = violator_df_i[col].dropna().unique()[0]
                # Make sure the replacement worked, and the uniqueness criterion is no longer violated by this group
                assert(all(violator_df_i[cols_shared_by_group].nunique()==1))
                # Remove it from list of violators
                groups_violating_uniqueness.remove(violator_group_i)

                # Add the violator back into the sample
                assert(violator_df_i.columns.tolist()==red_df.columns.tolist())
                red_df = pd.concat([red_df, violator_df_i])
    #-------------------------
    # NOTE: Below could really be completed in a single step using pd.Series.unique for both, I suppose.
    #       HOWEVER, pd.Series.unique returns a single value when one unique is found, and a list-like (series)
    #         object when multiple are found.  As I am used to having this always be a list, I split the procedure in two
    #       Separating the two also allows me to incorporate the allow_duplicates_in_lists switch
    # NOTE: A NaN (or NaT) will be ignored by nunique (i.e., if 1 unique and 1 NaN values, nunique will return a value of 1)
    #       but will not be ignored by unique (i.e., if 1 unique and 1 NaN values, unique will return (NaN, unique_val_i)
    #       Therefore, alter code from .agg(pd.Series.unique) to .agg(lambda x: x.dropna().unique())
    #return_df_a = red_df.groupby(groupby_col)[cols_shared_by_group].agg(pd.Series.unique)
    return_df_a = red_df.groupby(groupby_col)[cols_shared_by_group].agg(lambda x: x.dropna().unique())
    if allow_duplicates_in_lists:
        return_df_b = red_df.groupby(groupby_col)[cols_to_collect_in_lists].agg(list)
    else:
        return_df_b = red_df.groupby(groupby_col)[cols_to_collect_in_lists].agg(lambda x: list((set(x))))
    assert(return_df_a.shape[0]==return_df_b.shape[0])
    return_df = pd.merge(return_df_a, return_df_b, left_index=True, right_index=True)
    #-------------------------
    if verbose:
        print(f'groups_violating_uniqueness = {groups_violating_uniqueness}')
    #-------------------------
    if include_groupby_col_in_output_cols:
        return_df.index.name = 'idx'
        return_df[groupby_col] = return_df.index
        return_df =  move_cols_to_either_end(df=return_df, cols_to_move=[groupby_col], to_front=True)
    else:
        # In some instances, groupby_col could slip into the output columns, which would be unwanted if  
        #   include_groupby_col_in_output_cols==False.  This would happen if the user included groupby_col in 
        #   cols_shared_by_group explicitly, or if cols_shared_by_group is set to None, in which case it is set equal
        #   to cols_shared_by_group=[groupby_col] to allow proper functioning.
        if groupby_col in return_df.columns:
            return_df = return_df.drop(columns=[groupby_col])
    #-------------------------
    if rename_cols is not None:
        return_df = return_df.rename(columns=rename_cols)
    #-------------------------
    return return_df
    

#--------------------------------------------------    
def resolve_uniqueness_violators(
    df, 
    groupby_cols, 
    gpby_dropna=True,
    run_nan_groups_separate=True
):
    r"""
    NOTE: This method should be faster than that originally used in Utilities_df.consolidate_df
      Original Method:
        Used nunique to determine group_has_unq_val_where_expected and groups_violating_uniqueness.
        However, nunique does not count any NaNs (unless dropna explicitly set to False).
        Therefore, in the original method, group_has_unq_val_where_expected would contain duplicate entries where
          the offending values are NaNs
        Then, these are reduced away with the call:
          red_df.groupby(groupby_cols, dropna=gpby_dropna).agg(lambda x: x.dropna().unique())
        However, the majority of red_df is fine (without the duplicates), and doesn't really need to be run through the above 
          line of code.
        Thus, the new method will speed things up by NOT INCLUDING such 'acceptable duplicate' entries in 
          group_has_unq_val_where_expected.
      New Method:
        Uses size to determine group_has_unq_val_where_expected and groups_violating_uniqueness.
        NOTE: One could also use nunique(dropna=False), but is seems using size() is faster
        Using the new method, group_has_unq_val_where_expected will contain only entries for one there is one unique
          entry, regardless of any NaNs.
        Then, the 'acceptable duplicate' will be contained in groups_violating_uniqueness
    """
   #----------------------------------------------------------------------------------------------------
    assert(Utilities.is_object_one_of_types(groupby_cols, [str, list, tuple]))
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]
    #-------------------------
    #----------------------------------------------------------------------------------------------------
    # Running the case where gpby_dropna==False is much more time consuming mainly due to the use of 
    #   .apply with custom lambda function instead of simpler .filter call
    # With this in mind, let's separate out any groups containing NaN values.
    #   Then, the groups with no NaNs will be run with gpby_dropna==True, the groups with NaNs will be run
    #   separately with gpby_dropna==False, and the two will be combined.
    # NOTE: This is different than the consideration of NaNs elsewhere (where dealing with approximately duplicate values)
    #       This has to do with the groups themselves containing NaN values as keys.
    #         e.g., if grouping by mfr_devc_ser_nbr and prem_nb, if some values of prem_nb are NaNs, then there will be groups
    #           such as ('mfr_devc_ser_nbr_x', np.nan)
    # NOTE: run_nan_groups_separate was needed to avoid an infinite loop below!
    #-----
    if not gpby_dropna and run_nan_groups_separate:
        # Separate out from df any groups which contain NaN values
        do_groups_contain_nans_srs = df[groupby_cols].isna().sum(axis=1)>0
        df_w_nan_gps = df[do_groups_contain_nans_srs]
        # And drop those from df
        df = df.drop(df[df[groupby_cols].isna().sum(axis=1)>0].index)

        # Run df_w_nan_gps through resolve_uniqueness_violators with gpby_dropna=False
        #   and set gpby_dropna=True for the rest of this function
        # !!!!! IMPORTANT !!!!!
        #   Below, run_nan_groups_separate MUST BE SET TO FALSE, otherwise the function will always
        #     enter this loop, and an infinite loop is formed!
        df_w_nan_gps = resolve_uniqueness_violators(
            df=df_w_nan_gps, 
            groupby_cols=groupby_cols,  
            gpby_dropna=False, 
            run_nan_groups_separate=False
        )
        gpby_dropna=True
    else:
        df_w_nan_gps = pd.DataFrame()    
    #----------------------------------------------------------------------------------------------------
    # Only include groups which have one unique entry for each column.
    # The below series has a True/False value for each group stating whether (True)
    #   or not (False) all columns in df have one unique value
    # NOTE: Groups which have multiple values for any of the columns, but the multiple values consist of a unique value
    #         together with a NaN, will ultimately not be considered to be violating uniqueness.
    #         These are currently included in groups_violating_uniqueness, but will be weeded out into fake_violators below.
    #       The procedure below will also allow the case where groups which have multiple values for any of the columns, 
    #         but the multiple values consist of a unique value together with '' or None.
    #         These are recovered at the end of the function, after the "# Try to recover those in true_violators_df if possible" comment
    #           i.e., the procedure at the end of the function is in place to handle cases where groups have multiple values for any of 
    #             the columns but the multiple values consist of a unique value together with '' or None!
    #-----
    grp_sizes = df.groupby(groupby_cols, dropna=gpby_dropna).size()
    group_has_unq_val_where_expected = grp_sizes[grp_sizes==1].index
    groups_violating_uniqueness = grp_sizes[grp_sizes>1].index
    #-----
    # # NOTE: The procedure below would work equally as well for finding group_has_unq_val_where_expected and 
    # #         groups_violating_uniqueness, but takes longer
    # is_grp_unq_srs = (df.groupby(groupby_cols, dropna=gpby_dropna).nunique(dropna=False)<=1).all(axis=1)
    # group_has_unq_val_where_expected = is_grp_unq_srs[is_grp_unq_srs].index
    # groups_violating_uniqueness = is_grp_unq_srs[~is_grp_unq_srs].index
    #*************************-------------------------
    # At this point, if there are no groups violating uniqueness, the function can simply return
    if len(groups_violating_uniqueness)==0:
        if df_w_nan_gps.shape[0]>0:
            assert(len(set(df.columns).symmetric_difference(set(df_w_nan_gps.columns)))==0)
            df = pd.concat([df, df_w_nan_gps[df.columns]])
        return df
    #*************************-------------------------
    #-------------------------
    # Separate out non-violators (to go into reduced DF, red_df) and violators
    # NOTE: May be quicker to use only groups_violating_uniqueness below in the 'if' portion of the block,
    #       as is done in the 'else' portion
    if gpby_dropna:
        red_df       = df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False).filter(lambda x: x.name in group_has_unq_val_where_expected)
        violators_df = df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False).filter(lambda x: x.name in groups_violating_uniqueness)
    else:
        # Separate out non-violators (to go into reduced DF, red_df) and violators
        # NOTE: groups_violating_uniqueness is typically much smaller than group_has_unq_val_where_expected.
        #       Thus, here I use groups_violating_uniqueness to determine both red_dc and violators_df as it
        #         is much faster
        red_df       = df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(
            lambda x: 
            x if not any([Utilities.are_equal_nans_ok(x.name,group_i) 
                          for group_i in groups_violating_uniqueness]) 
            else pd.DataFrame()
        )
        #-----
        violators_df = df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(
            lambda x: 
            x if any([Utilities.are_equal_nans_ok(x.name,group_i) 
                      for group_i in groups_violating_uniqueness]) 
            else pd.DataFrame()
        )
    #-------------------------
    # violator_nunq_df will be used to separate true and fake violators
    violator_nunq_df = violators_df.groupby(groupby_cols, dropna=gpby_dropna).nunique()
    #-----
    # Separate out the fake violators and true violators using nunique.
    #   The fake violators will have all nunique values==1, meaning the degeneracy causing its classification in 
    #     violators_df (and, before that, groups_violating_uniqueness) was due to NaN values
    # NOTE: After this step, there still may be some fake violators hidden in true_violators, as we have not
    #       yet accounted for '' and None being treated as NaN
    true_violators = violator_nunq_df[(violator_nunq_df>1).any(axis=1)].index.tolist()
    fake_violators = violator_nunq_df[(violator_nunq_df<=1).all(axis=1)].index.tolist()
    #-----
    if gpby_dropna:
        true_violators_df = violators_df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False).filter(lambda x: x.name in true_violators)
        fake_violators_df = violators_df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False).filter(lambda x: x.name in fake_violators)
    else:
        true_violators_df = violators_df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(
            lambda x: 
            x if any([Utilities.are_equal_nans_ok(x.name,group_i) 
                      for group_i in true_violators]) 
            else pd.DataFrame()
        )
        #-----
        fake_violators_df = violators_df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(
            lambda x: 
            x if any([Utilities.are_equal_nans_ok(x.name,group_i) 
                      for group_i in fake_violators]) 
            else pd.DataFrame()
        )    
    #-------------------------
    # At this point, fake_violators_df can be reduced down using dropna().unique(), and then added to red_df
    if fake_violators_df.shape[0]>0:
        fake_violators_df = fake_violators_df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False).agg(lambda x: x.dropna().unique())
        assert(len(set(red_df.columns).symmetric_difference(set(fake_violators_df.columns)))==0)
        red_df = pd.concat([red_df, fake_violators_df[red_df.columns]])
    #--------------------------------------------------
    # Finally, try to recover any fake violators still in true_violators_df if possible
    #   If empty entries (equal to '' or None) are what is causing the violation of uniquness, then these missing
    #   values will be assumed to be equal to (and alterted to be equal to) the others in the group.
    if true_violators_df.shape[0]>0:
        for violator_group_i, violator_df_i in true_violators_df.groupby(groupby_cols, dropna=gpby_dropna):
            # Find the columns in violator_df_i causing the violations
            violating_cols_i = violator_df_i.columns[violator_df_i.nunique()>1]

            # In the violating columns, replace any empty values with np.nan to be more easily identified
            violator_df_i[violating_cols_i] = violator_df_i[violating_cols_i].replace(['', None], np.nan)

            # Note: Below, the default behavior of nunique is to ignore NaN values, but I have dropna=True to make
            #       this explicitly clear
            # After replacing missing values with np.nans, if there is no longer a violation, the empty values
            #   are replaced by the unique value for the group's column
            if all(violator_df_i.nunique(dropna=True)==1):
                violator_df_i_to_append = violator_df_i.iloc[[0]].copy()
                for col in violator_df_i.columns:
                    violator_df_i_to_append[col] = violator_df_i[col].dropna().unique()[0]

                # Add the violator back into the sample
                assert(violator_df_i_to_append.columns.tolist()==red_df.columns.tolist())
                red_df = pd.concat([red_df, violator_df_i_to_append])
    #--------------------------------------------------  
    # If nan groups were run separately, and df_w_nan_gps is not empty, add to red_df
    if df_w_nan_gps.shape[0]>0:
        assert(len(set(red_df.columns).symmetric_difference(set(df_w_nan_gps.columns)))==0)
        red_df = pd.concat([red_df, df_w_nan_gps[red_df.columns]])
    #--------------------------------------------------
    return red_df
    

#--------------------------------------------------    
def agg_func_list(x):
    return list(x)
#--------------------------------------------------
def agg_func_unq_list(x):
    return list((set(x)))
#--------------------------------------------------
def agg_func_list_dropna(x):
    return [x for x in list(x) if pd.notna(x)]
#--------------------------------------------------
def agg_func_unq_list_dropna(x):
    return  [x for x in set(x) if pd.notna(x)]



#--------------------------------------------------
def read_consolidated_df_from_csv(
    file_path, 
    list_cols, 
    set_col_to_index=None,
    drop_index_col=False, 
    convert_cols_and_types_dict=None, 
    to_numeric_errors='coerce'
):
    r"""
    When stored as a CSV, any list columns are retrieved as a strings, instead of as a lists.
    Using ast.literal_eval fixes that
    """
    #-------------------------
    df = pd.read_csv(file_path, dtype=str)
    #-------------------------
    for col in list_cols:
        df[col] = df[col].apply(lambda x: literal_eval(x))
    #-------------------------
    if set_col_to_index is not None:
        df.set_index(set_col_to_index, drop=drop_index_col, inplace=True)
        df.index.name='idx'
    #-------------------------
    if convert_cols_and_types_dict is not None:
        df = Utilities_df.convert_col_types(
            df=df, 
            cols_and_types_dict=convert_cols_and_types_dict, 
            to_numeric_errors=to_numeric_errors, 
            inplace=True
        )
    #-------------------------
    return df



#--------------------------------------------------
def consolidate_df(
    df                                  , 
    groupby_cols                        , 
    cols_shared_by_group                , 
    cols_to_collect_in_lists            , 
    cols_to_drop                        = None, 
    as_index                            = True, 
    include_groupby_cols_in_output_cols = False, 
    allow_duplicates_in_lists           = False, 
    allow_NaNs_in_lists                 = False, 
    recover_uniqueness_violators        = True, 
    gpby_dropna                         = True, 
    rename_cols                         = None, 
    custom_aggs_for_list_cols           = None, 
    verbose                             = True
):
    r"""
    TODO: Either use resolve_uniqueness_violators or implement similar methods here!
    
    This function consolidates a DF which has a lot of repetitive information.
    More specifically, this is designed for a DF in which groups of the data share a single 
      unique value for many of their columns.
    This function will consolidate the DF into a form where each group is represented by a single row,
      columns for which the group shares a single value will be reproduced as a single value, and element 
      of columns where the group has multiple values will become a list.
    
    groupby_cols:
      Designed to work with multiple groupby columns, but using a single column should work as well
      
    include_groupby_cols_in_output_cols:
      Default to False.
      By default, when grouping, the groupby columns become the indices for the resultant DF (this is due to the .agg calls on red_df,
        not to be confused with the .apply calls for which as_index is explicitly set to False).
      If this is set to True, the indices of the resultant DF will be named f'idx_lvl_{i}', and a columns for each of groupby_cols
        will be included in the DF.
        NOTE: This was designed when working with build_end_events_for_no_outages.  When using AMI_SQL it is easiest 
              to have the groupby value reproduced in a column (instead of just as the index) so the value can be 
              included in the SQL query and resultant DF.
      
    allow_duplicates_in_lists:
      Default to False, so lists will only have unique elements.
      e.g., when dealing with Meter Premise data, a single premise may have multiple serial numbers.
        This situation would lead to the meter premise being repeated, which is typically not desired
        
    gpby_dropna:
        This determines whether or not groups with NaN values are permitted.
        
    custom_aggs_for_list_cols:
        Allows the user more control over what functions are used in the aggregation of the list columns (those in cols_to_collect_in_lists).
        e.g., if a column, col, in cols_to_collect_in_lists already contains list elements, the default agg functions will not work.
              Instead, one could do, e.g., custom_aggs_for_list_cols=lambda x: list(set(itertools.chain(*x)))
        Can be:
          single function:
            ==> to be used on all list columns
          dict with keys equal to columns and values equal to functions
            ==> In this case, any list cols not explicitly contained in custom_aggs_for_list_cols will have
                  their functions set equal to the agg_func from the logic in the function (determined by 
                  allow_NaNs_in_lists and allow_duplicates_in_lists)        
        
        
    NOTE: Somewhat strange split in a couple of places below triggered by if gpby_dropna.
          Looking at the if/else statements, it appears they should achieve the same result 
              -One has .filter(lambda x: statement(x)) vs .apply(lambda x: x if statement(x) else None)
          This split is due to the strange behavior of .filter() when dropna=False in groupby.  Basically, if a 
            group has a NaN value, regardless of the returned values of the lambda function inside of .filter(), 
            that group will not be returned!  This is strange behavior, and using .apply() instead (with slightly
            different lambda function) solves the issue.
            A minimum working example can be reproduced using the following:
                d = {'col1': [1, 1], 'col2': [np.nan, np.nan], 'col3': [5,6]}
                df = pd.DataFrame(data=d)
                
                # The following line unexpectedly returns and empty DF
                df.groupby(['col1', 'col2'], dropna=False).filter(lambda x: True)
                
                # The following line returns the expected results
                df.groupby(['col1', 'col2'], dropna=False).apply(lambda x:x if True else None)
                
                # Note also that aggregation seems to work as expected, like apply.  So, filter is hopefully
                #   the only odd one out
                df.groupby(['col1', 'col2'], dropna=False)['col3'].agg(lambda x: Utilities_df.agg_func_list(x))
                
        Actually, it appears things are even stranger than the situation above.  Apparently, the behavior is different depending on the 
          data type of the column containing NaN.  The DataType is typically inferred from non-NaN values, so the different behavior is not exactly predictable.
          Below, 1A and 1B give differnt results!  1A returns an empty DF (as described above), whereas 1B returns df.  The only difference
            is that in 1B col2 has been cast as a string!
          1A:
                d = {'col1': [1, 1], 'col2': [np.nan, np.nan], 'col3': [5,6]}
                df = pd.DataFrame(data=d)
                df.groupby(['col1', 'col2'], dropna=False).filter(lambda x: True)
          
          1B: 
                d = {'col1': [1, 1], 'col2': [np.nan, np.nan], 'col3': [5,6]}
                df = pd.DataFrame(data=d)
                df['col2']=df['col2'].astype(str)
                df.groupby(['col1', 'col2'], dropna=False).filter(lambda x: True)
                
          Using apply instead of filter gives equal results in 2A and 2B
          2A:
                d = {'col1': [1, 1], 'col2': [np.nan, np.nan], 'col3': [5,6]}
                df = pd.DataFrame(data=d)
                df.groupby(['col1', 'col2'], dropna=False).apply(lambda x: x if True else None)
          
          2B: 
                d = {'col1': [1, 1], 'col2': [np.nan, np.nan], 'col3': [5,6]}
                df = pd.DataFrame(data=d)
                df['col2']=df['col2'].astype(str)
                df.groupby(['col1', 'col2'], dropna=False).apply(lambda x: x if True else None)
                
    NOTE:
        THERE APPEARS TO HAVE BEEN SOME SORT OF BUG IN THE OLD VERSION OF PANDAS (1.3.4)
        When I used groupby.apply(lambda x:...), the resultant DF NEVER had the group labels as the index (i.e., the
          original index was always maintained).
        It was as if as_index was set to False (even though the default value was True; furthermore, the value of as_index
          seemed to have no affect on the result).
        TO OBTAIN THE FORM UNDER WHICH THE METHOD WAS CONSTRUCTED, one must set as_index=False AND group_keys=False as arguments
          to groupby (as the apply operations here are essentially just filter calls)!
    """
    #-------------------------
    if cols_to_drop is not None:
        df = df.drop(columns=cols_to_drop)
    #-------------------------
    assert(Utilities.is_object_one_of_types(groupby_cols, [str, list, tuple]))
    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]
    #-------------------------
    if cols_shared_by_group is None or len(cols_shared_by_group)==0:
        cols_shared_by_group = groupby_cols
    assert(Utilities.is_object_one_of_types(cols_shared_by_group, [list, str, int]))
    if not isinstance(cols_shared_by_group, list):
        cols_shared_by_group=[cols_shared_by_group]
    #-----
    if cols_to_collect_in_lists is None:
        cols_to_collect_in_lists = [x for x in df.columns.tolist() if x not in groupby_cols+cols_shared_by_group]
    assert(Utilities.is_object_one_of_types(cols_to_collect_in_lists, [list, str, int]))
    if not isinstance(cols_to_collect_in_lists, list):
        cols_to_collect_in_lists=[cols_to_collect_in_lists]
    #-------------------------
    # Only include groups which have one unique entry for each of cols_shared_by_group
    # The below series has a True/False value for each group stating whether (True)
    #   or not (False) all cols in cols_shared_by_group have one unique value
    # NOTE: Groups which have multiple values for any of cols_shared_by_group, but the multiple values consist of a unique value
    #         together with a NaN, are not considered to be violating uniqueness.
    #       Thus, before (and immediately after) the 'if recover_uniqueness_violators' block, red_df may very well have groups
    #         with more than a single entry, AND THIS IS OK!  Those cases will be reduced down below through the call:
    #           return_df_a = red_df.groupby(groupby_cols, dropna=gpby_dropna)[cols_shared_by_group].agg(lambda x: x.dropna().unique())
    #       The recover_uniqueness_violators procedure is in place to handle cases where groups have multiple values for any of 
    #         cols_shared_by_group but the multiple values consist of a unique value together with '' or None!
    group_has_unq_val_where_expected = (df.groupby(groupby_cols, dropna=gpby_dropna)[cols_shared_by_group].nunique()<=1).all(axis=1)

    # Those not satisfying the uniqueness criterion are collected below
    groups_violating_uniqueness = group_has_unq_val_where_expected[~group_has_unq_val_where_expected.values].index.tolist()
    #-------------------------
    # Reduce down df, keeping only those satisfying uniqueness
    if gpby_dropna:
        red_df = df.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: x.name not in groups_violating_uniqueness)
    else:
        # Keep only groups (x.name) which are not found in groups_violating_uniqueness
        #   Due to the nature of NaNs (nan!=nan), and the fact that gpby_dropna=False here meaning that the groups can be equal to NaN
        #     or be equal to lists containing NaNs, one cannot simply call  .apply(lambda x: x if x.name not in groups_violating_uniqueness else None)
        #   Furthermore, np.array_equal with equal_nan=True seems like it should have worked, but it didn't
        #   Therefore, I had to create and use a custom Utilities.are_equal_nans_ok
        #   So, below, for each group, run Utilities.are_equal_nans_ok with each element of groups_violating_uniqueness.
        #     If group matches (i.e., returns True) any element in groups_violating_uniqueness, it should be excluded.
        red_df = df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(
            lambda x: 
            x if not any([Utilities.are_equal_nans_ok(x.name,group_i) 
                          for group_i in groups_violating_uniqueness]) 
            else None
        )
    #-------------------------
    # If recover_uniqueness_violators, try to recover those in groups_violating_uniqueness if possible
    #   If empty entries (equal to '' or None) are what is causing the violation of uniquness, then these missing
    #   values will be assumed to be equal to (and alterted to be equal to) the others in the group.
    if recover_uniqueness_violators:
        for violator_group_i in groups_violating_uniqueness:
            if gpby_dropna:
                violator_df_i=df.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: x.name==violator_group_i).copy()
            else:
                # Related issue as above having to do with nature of NaN.  In this case, this means I cannot simply call:
                #   violator_df_i=df.groupby(groupby_cols, dropna=gpby_dropna).apply(lambda x: x if x.name==violator_group_i else None).copy()
                # Instead, Utilities.are_equal_nans_ok must be used (not sure why else pd.DataFrame() is needed here but else None didn't complain above?)
                violator_df_i=df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(
                    lambda x: 
                    x if Utilities.are_equal_nans_ok(x.name,violator_group_i) 
                    else pd.DataFrame()
                ).copy()
            # Find the columns in violator_df_i causing the violations
            violating_cols_i = violator_df_i[cols_shared_by_group].columns[violator_df_i[cols_shared_by_group].nunique()>1]

            # In the violating columns, replace any empty values with np.nan to be more easily identified
            violator_df_i[violating_cols_i] = violator_df_i[violating_cols_i].replace(['', None], np.nan)

            # Note: Below, the default behavior of nunique is to ignore NaN values, but I have dropna=True to make
            #       this explicitly clear
            # After replacing missing values with np.nans, if there is no longer a violation, the empty values
            #   are replaced by the unique value for the group's column
            if all(violator_df_i[violating_cols_i].nunique(dropna=True)==1):
                for col in violating_cols_i:
                    violator_df_i[col] = violator_df_i[col].dropna().unique()[0]
                # Make sure the replacement worked, and the uniqueness criterion is no longer violated by this group
                assert(all(violator_df_i[cols_shared_by_group].nunique()==1))
                # Remove it from list of violators
                groups_violating_uniqueness.remove(violator_group_i)

                # Add the violator back into the sample
                assert(violator_df_i.columns.tolist()==red_df.columns.tolist())
                red_df = pd.concat([red_df, violator_df_i])
    #-------------------------
    # NOTE: Below could really be completed in a single step using pd.Series.unique for both, I suppose.
    #       HOWEVER, pd.Series.unique returns a single value when one unique is found, and a list-like (series)
    #         object when multiple are found.  As I am used to having this always be a list, I split the procedure in two
    #       Separating the two also allows me to incorporate the allow_duplicates_in_lists switch
    # NOTE: A NaN (or NaT) will be ignored by nunique (i.e., if 1 unique and 1 NaN values, nunique will return a value of 1)
    #       but will not be ignored by unique (i.e., if 1 unique and 1 NaN values, unique will return (NaN, unique_val_i)
    #       Therefore, alter code from .agg(pd.Series.unique) to .agg(lambda x: x.dropna().unique())
    #return_df_a = red_df.groupby(groupby_cols)[cols_shared_by_group].agg(pd.Series.unique)
    if cols_shared_by_group == groupby_cols:
        return_df_a = None
    else:
        # Methods above already ensure there is a single, non-NaN value for each column in cols_shared_by_group, so .agg('first') may be used.
        #   If there are NaN values and a single non-NaN value, using 'first' will select the fist non-NaN value
        #   If all values are NaN, 'first' will return NaN
        return_df_a = red_df.groupby(groupby_cols, dropna=gpby_dropna)[cols_shared_by_group].agg('first')
    #-------------------------
    if allow_NaNs_in_lists:
        if allow_duplicates_in_lists:
            agg_func = agg_func_list
        else:
            agg_func = agg_func_unq_list
    else:
        if allow_duplicates_in_lists:
            agg_func = agg_func_list_dropna
        else:
            agg_func = agg_func_unq_list_dropna
    if custom_aggs_for_list_cols is None:
        return_df_b = red_df.groupby(groupby_cols, dropna=gpby_dropna)[cols_to_collect_in_lists].agg(lambda x: agg_func(x))
    else:
        assert(isinstance(custom_aggs_for_list_cols, dict) or callable(custom_aggs_for_list_cols))
        if callable(custom_aggs_for_list_cols):
            return_df_b = red_df.groupby(groupby_cols, dropna=gpby_dropna)[cols_to_collect_in_lists].agg(
                lambda x: custom_aggs_for_list_cols(x)
            )
        else:
            # Make sure all values in dict are functions
            for agg_func_i in custom_aggs_for_list_cols.values():
                assert(callable(agg_func_i))
            # Any columns not explicitly contained in custom_aggs_for_list_cols will have 
            #   their functions set equal to agg_func
            dflt_agg_dict = {
                col:lambda x: agg_func(x) 
                for col in list(set(cols_to_collect_in_lists).difference(set(custom_aggs_for_list_cols.keys())))
            }
            # Build final agg_dict
            agg_dict = custom_aggs_for_list_cols | dflt_agg_dict
            assert(len(set(agg_dict.keys()).symmetric_difference(set(cols_to_collect_in_lists)))==0)
            #-------------------------
            return_df_b = red_df.groupby(groupby_cols, dropna=gpby_dropna)[cols_to_collect_in_lists].agg(agg_dict)
    #-------------------------
    if return_df_a is not None:
        assert(return_df_a.shape[0]==return_df_b.shape[0])
        return_df = pd.merge(return_df_a, return_df_b, left_index=True, right_index=True)
    else:
        return_df = return_df_b
    #-------------------------
    if verbose:
        print(f'groups_violating_uniqueness = {groups_violating_uniqueness}')
    #-------------------------
    if not as_index:
        include_groupby_cols_in_output_cols=True
    #-------------------------
    if include_groupby_cols_in_output_cols:
        assert(list(return_df.index.names)==groupby_cols)
        return_idx_names = [f'idx_lvl_{i}' for i in range(return_df.index.nlevels)]
        return_df.index.names = return_idx_names
        for i_lvl in range(len(groupby_cols)):
            return_df[groupby_cols[i_lvl]] = return_df.index.get_level_values(i_lvl)
        return_df =  move_cols_to_either_end(df=return_df, cols_to_move=groupby_cols, to_front=True)
    else:
        # In some instances, groupby_cols could slip into the output columns, which would be unwanted if  
        #   include_groupby_cols_in_output_cols==False.  This would happen if the user included groupby_cols in 
        #   cols_shared_by_group explicitly, or if cols_shared_by_group is set to None, in which case it is set equal
        #   to cols_shared_by_group=[groupby_cols] to allow proper functioning.
        for grp_by_col in groupby_cols:
            if grp_by_col in return_df.columns:
                return_df = return_df.drop(columns=[grp_by_col])
    #-------------------------
    if not as_index:
        return_df = return_df.reset_index(drop=True)
    #-------------------------
    if rename_cols is not None:
        return_df = return_df.rename(columns=rename_cols)
    #-------------------------
    return return_df
    
#--------------------------------------------------    
def consolidate_df_according_to_fuzzy_overlap_intervals_OLD(
    df, 
    ovrlp_intrvl_0_col, 
    ovrlp_intrvl_1_col, 
    fuzziness, 
    groupby_cols, 
    assert_single_overlap=True, 
    cols_to_collect_in_lists=None, 
    recover_uniqueness_violators=False, 
    gpby_dropna=True, 
    drop_idx_cols=False, 
    maintain_original_cols=False, 
    enforce_list_cols_for_1d_df=True
):
    r"""
    Given a DataFrame with two columns representing the start and end of some interval, find the common overlap intervals 
    and reduce the DF down to one row per overlap interval.
    
    !!!!! NOTE !!!!!: This was designed specifically for use in MeterPremise.drop_approx_mp_duplicates, in which it is utilized inside
                      of a groupby.apply(lambda x:) function (see description below)
    
    Basically, the function finds all of the (fuzzy) overlap intervals from list(zip(df['inst_ts'], df['rmvl_ts'])).
    The rows of df contained in each overlap group are recorded.
    For each overlap group, the collection of rows is consolidated down into a single row using consolidate_df, 
      according to:
          groupby_cols             = ovrlp_groupby_cols
          cols_shared_by_group     = ovrlp_groupby_cols
          cols_to_collect_in_lists = ovrlp_cols_to_collect_in_lists
    So, there should only be a single group for the overlap represented by groupby_cols.  For use with MeterPremise.drop_approx_mp_duplicates,
      a typical value for groupby_cols = ['mfr_devc_ser_nbr', 'prem_nb'].
    If groupby_cols is None, a random temporary column will be used (set to a constant value for entire DF).
    Essentially, for the row representing each overlap:
        Columns in groupby_cols will be single values.
        All other columns will have entries of type list containing all unique values from the overlap group.
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    THIS IS INTENDED FOR USE IN MeterPremise.drop_approx_mp_duplicates.
    This was built for use inside a groupby.apply(lambda x:) function, not really designed to be run on its own.
    However, there's no reason it can't be run on it's own as long as one understands what is being done.
    
    For use in MeterPremise.drop_approx_mp_duplicates, df should be a collection of entries for one single meter.
      More specifically, by one single meter I mean one distinct combination of 'mfr_devc_ser_nbs' and 'prem_nb'.
    In meter_premise_hist, there can be multiple entries for a given meter, which are all identical except for 
      except for e.g., inst_kind_cd, mtr_kind_cd, and even inst_ts and rmvl_ts.
    In some instances, it seems like these different entries represent when a meter was upgraded, but I'm not certain.
    In any case, at the end of the day, for a single meter I want only a single row with one inst_ts and one rmvl_ts
      This is why assert_single_overlap is set to True by default.
      
    For use in MeterPremise.drop_approx_mp_duplicates, typical arguments would be:
        ovrlp_intrvl_0_col = 'inst_ts'
        ovrlp_intrvl_1_col = 'rmvl_ts'
        fuzziness=pd.Timedelta('1 hour')
        groupby_cols=['mfr_devc_ser_nbr', 'prem_nb']
        assert_single_overlap=True
        cols_to_collect_in_lists=None
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    df:
        DataFrame object with two columns representing the start (ovrlp_intrvl_0_col) and end 
          (ovrlp_intrvl_1_col) of some interval.
          
    ovrlp_intrvl_0_col/ovrlp_intrvl_1_col:
        Columns containing the start/end of the interval for each entry.
        
    fuzziness:
        Sets how close two intervals must be to be considered overlapping.
        See Utilities.get_fuzzy_overlap_intervals for more information.
        MUST BE COMPATIBLE WITH + operation with data in ranges.
        ==> i.e., e.g., ranges[0][0] + fuzziness must run successfully!
        
    groupby_cols:
        Columns used to group each overlap subset of df.
        These should essentially 
        
        
    drop_idx_cols:
        This operation needs to call reset_index to function properly.  When this is done, the indices are added as columns (singular for normal index, 
        plural for MultiIndex) to the DF.  If drop_idx_cols==True, these columns will be dropped from the final DF
        
    maintain_original_cols:
        If True, the returned DF will have the same columns as the input df
        
    enforce_list_cols_for_1d_df:
        This dictates the behavior when the input df has only a single row (and maintain_original_cols is set to True).
        When True, the columns in cols_to_collect_in_lists will be returned as lists.
        When False, they will be returned as they are in df.
        REASON:
            When using in lambda function of groupby (e.g., in MeterPremise.drop_approx_mp_duplicates), it turns out to be faster (about 2x faster with limited testing)
            to have the set to False and to adjust all rows of the grouped DF afterwards via, e.g.,
                for lst_col in cols_to_collect_in_lists:
                    return_df[lst_col] = return_df[lst_col].apply(lambda x: x if isinstance(x, list) else [x])
        
    """
    #-------------------------
    # Don't want to alter df itself
    df = df.copy()
    og_cols = df.columns
    #-------------------------
    if groupby_cols is None:
        tmp_col = Utilities.generate_random_string()
        df[tmp_col] = 1
        groupby_cols = [tmp_col]
    else:
        tmp_col = None
    #-------------------------
    if cols_to_collect_in_lists is None:
        cols_to_collect_in_lists=[x for x in df.columns if x not in groupby_cols]
    else:
        # Don't want cols_to_collect_in_lists to be altered outside the function (due to .extend() call)
        #   ESPECIALLY if being used in .groupy lambda function.
        cols_to_collect_in_lists = copy.deepcopy(cols_to_collect_in_lists)
    #-------------------------
    if df.shape[0]<=1 and maintain_original_cols: 
        # Reason for maintain_original_cols:
        #   If this is used in .groupby, it is important for all to have same shape/labelling.
        #   So, if maintain_original_cols is False, in order for a df with df.shape[0]==1 to fit
        #     into the .groupby procedure, it must go through the steps below to ensure it has the same
        #     form as the other groups.
        if enforce_list_cols_for_1d_df:
        # Make any cols_to_collect_in_lists into lists EXCEPT ovrlp_intrvl_0_col and ovrlp_intrvl_1_col...
            lst_cols_1d = [x for x in cols_to_collect_in_lists if x not in [ovrlp_intrvl_0_col, ovrlp_intrvl_1_col]] #1d for 1-dimensional DF
            df[lst_cols_1d] = df[lst_cols_1d].apply(lambda x: [x.tolist()])
        return df
    #-------------------------
    # Make sure fuzziness is compatible with ovrlp_intrvl_0(1)_col
    try:
        test_0 = df.iloc[0][ovrlp_intrvl_0_col]+fuzziness
        test_1 = df.iloc[0][ovrlp_intrvl_1_col]+fuzziness
    except:
        print(f'''
        In consolidate_df_according_to_fuzzy_overlap_intervals: Incompatible fuzziness type
            type(fuzziness) = {type(fuzziness)}
            df[ovrlp_intrvl_0_col].dtype = {df[ovrlp_intrvl_0_col].dtype}
            df[ovrlp_intrvl_1_col].dtype = {df[ovrlp_intrvl_1_col].dtype}
        CRASH IMMINENT!
        ''')
    #-------------------------
    idx_level_names = df.index.names
    # When calling reset_index(drop=False), if the level has a name, it will be used as the column, 
    #   otherwise f'level_{idx_level}' is used
    # NOTE: If only a single, un-named index level, the column will be named 'index', not 'level_0'
    idx_col_names = [x if x is not None else f'level_{i}' for i,x in enumerate(idx_level_names)]
    if len(idx_col_names)==1 and idx_col_names[0]=='level_0':
        idx_col_names = ['index']
    #-------------------------
    df = df.reset_index(drop=False)
    # Make sure the idx_col_names are contained in the columns
    assert(len(set(idx_col_names).difference(set(df.columns)))==0)
    # Add the idx_col_names to cols_to_collect_in_lists
    cols_to_collect_in_lists.extend(idx_col_names)
    #-------------------------
    
    # First, make sure the second element in each tuple should be greater than the first
    # NOTE: The second element can be NaN (either NaT for times, or NaN otherwise)
    #         Apparently, NaN evaluates as False when compared in any manner to anything else
    #         (i.e., anything>NaN = False, anything<NaN = False, anything==NaN = False)
    #       Thus, df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col] will result in False whenever
    #         df[ovrlp_intrvl_1_col] is NaN, and therefore the assertion:
    #           assert(all(df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]))
    #         would fail in in unwanted circumstances, since NaN value means and open ended interval
    #         and therefore any beginning value should be considered less than a NaN value.
    #       Therefore, instead of the single-line assertion above, I will include two assertions,
    #         one to ensure df[ovrlp_intrvl_0_col] doesn't contain any NaNs, and one to ensure
    #         either df[ovrlp_intrvl_1_col] is a Nan or df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]
    # NOTE: I actually have found at least one case where ovrlp_intrvl_1_col<ovrlp_intrvl_0_col (by one second)
    #       Instead of throwing an assertion error, I will instead print a warning and simply remove the offending entry(ies)
    if any(df[ovrlp_intrvl_0_col].isna()):
        print(f'''
            !!!!! WARNING !!!!! In consolidate_df_according_to_fuzzy_overlap_intervals:
            df[ovrlp_intrvl_0_col] has NaN values!
            df[ovrlp_intrvl_0_col].isna().sum() = {df[ovrlp_intrvl_0_col].isna().sum()}
            df.shape[0]                         = {df.shape[0]}
            Row containing these NaNs will be omitted!
        ''')
        df = df[df[ovrlp_intrvl_0_col].notna()]
    #-----
    # NOTE: In printing the output below, the parentheses in (~scnd_gt_frst_srs).sum() are important!
    #       (~scnd_gt_frst_srs).sum() != ~scnd_gt_frst_srs.sum() (the latter essentially equals -1*scnd_gt_frst_srs.sum())
    scnd_gt_frst_srs = ((df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]) | (df[ovrlp_intrvl_1_col].isna()))
    if any(~scnd_gt_frst_srs):
        print(f'''
            !!!!! WARNING !!!!! In consolidate_df_according_to_fuzzy_overlap_intervals:
            df has values for which df[ovrlp_intrvl_0_col]>=df[ovrlp_intrvl_1_col]!
            Number of violators = {(~scnd_gt_frst_srs).sum()}
            df.shape[0]         = {df.shape[0]}
            Rows containing these violators will be omitted!
        ''')
        df = df[scnd_gt_frst_srs]
    # Now, at this stage assertions should both pass    
    assert(all(df[ovrlp_intrvl_0_col].notna()))
    assert(all((df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]) | (df[ovrlp_intrvl_1_col].isna())))
    # Also, sort ranges, as will be necessary for this procedure
    df = df.sort_values(by=[ovrlp_intrvl_0_col, ovrlp_intrvl_1_col])
    #-------------------------
    # Set the first range in overlaps simply as from the first entry in df
    overlaps = []
    current_beg, current_end = df.iloc[0][[ovrlp_intrvl_0_col, ovrlp_intrvl_1_col]]
    overlaps.append(
        dict(min_val=current_beg, max_val=current_end, idxs=[df.index[0]])
    )

    # Iterate through and create the overlaps items, each of which will be a dict with key
    #   equal to min_val, max_val, and idxs (where idxs identifies which indices from df
    #   belong to each overlap group)
    for i, (idx, row) in enumerate(df.iterrows()):
        if i==0:
            continue
        #---------------
        beg = row[ovrlp_intrvl_0_col]
        end = row[ovrlp_intrvl_1_col]
        if beg > current_end+fuzziness:
            # beg after current end (with fuzziness buffer), so new interval needed
            # NOTE: beg > current_end+fuzziness will evaluate to False whenever current_end
            #       is NaN, which is the desired functionality.
            overlaps.append(dict(min_val=beg, max_val=end, idxs=[idx]))
            current_beg, current_end = beg, end
        else:
            # beg <= current_end+fuzziness, so overlap
            # The beg of overlaps[-1] remains the same, but the end of overlaps[-1] should be changed to
            #   the max of current_end and end.
            # Also, idx needs to be added to the overlap
            # NOTE: max(any_non_NaT, NaT) = any_non_NaT (remember, NaN evaluates as False when compared in any manner to anything else), 
            #       which is not the funcionality I want, as an end of NaT essentially means no end (e.g., a meter which is still in service), 
            #       and should therefore be treated as Inf.  Thus, cannot simply use the one-liner 'current_end = max(current_end, end)'
            # NOTE 2: Cannot simply do 'if pd.isna(current_end) or pd.isna(end)' in single line because this function is
            #         designed to work with various data types, so in such a scenario it would be unclear what to set
            #         current_end to (e.g., should it be pd.NaT, pd.NaN, etc?).
            #         Thus, instead of if-else, need if-elif-else
            #current_end = max(current_end, end)
            if pd.isna(current_end):
                current_end=current_end
            elif pd.isna(end):
                current_end = end
            else:
                current_end = max(current_end, end)
            overlaps[-1]['max_val'] = current_end
            overlaps[-1]['idxs'].append(idx)
    #--------------------------------------------------
    if assert_single_overlap and len(overlaps)!=1:
        print(f'assert_single_overlap and len(overlaps)={len(overlaps)}')
        print(overlaps)
        print(f'df.head():\n{df.head()}')
        assert(0)
    # Now, iterate through each overlap group and combine all members of group into single row
    dfs_list = []
    for overlap_dict_i in overlaps:
        df_i = consolidate_df(
            df=df.loc[overlap_dict_i['idxs']], 
            groupby_cols=groupby_cols, 
            cols_shared_by_group=groupby_cols, 
            cols_to_collect_in_lists=cols_to_collect_in_lists, 
            recover_uniqueness_violators=recover_uniqueness_violators, 
            gpby_dropna=gpby_dropna
        )
        # Make sure the operation successfully reduced df_i down to a single row, as expected
        assert(df_i.shape[0]==1)

        # Determine the index positions of ovrlp_intrvl_0_col and ovrlp_intrvl_1_col, as these will be needed
        #   below to set to values in the consolidated DF using iloc
        # NOTE: df_i doesn't necessarily have the same columns in the same order as df, hence why this
        #       is done here and not above.  Probably, one could do this for the first iteration and use it for all,
        #       but finding the index is not very stressful, so might as well be safe and do it here every time
        #-----
        ovrlp_intrvl_0_col_idx = find_idxs_in_highest_order_of_columns(df_i, ovrlp_intrvl_0_col)
        assert(len(ovrlp_intrvl_0_col_idx)==1)
        ovrlp_intrvl_0_col_idx=ovrlp_intrvl_0_col_idx[0]
        #-----
        ovrlp_intrvl_1_col_idx = find_idxs_in_highest_order_of_columns(df_i, ovrlp_intrvl_1_col)
        assert(len(ovrlp_intrvl_1_col_idx)==1)
        ovrlp_intrvl_1_col_idx=ovrlp_intrvl_1_col_idx[0]
        #-------------------------
        # Set the values for ovrlp_intrvl_0_col and ovrlp_intrvl_1_col
        # NOTE: For whatever dumb reason, pandas now behaves in a new manner.
        #       Basically, df_i.iloc[0, ovrlp_intrvl_0_col_idx/ovrlp_intrvl_1_col_idx] will typically contain lists of datetime objects. 
        #       Although it wasn't like this in the past (v 1.3.4), now, when trying to set this equal to a single datetime value 
        #         (e.g., df_i.iloc[0, ovrlp_intrvl_0_col_idx] = overlap_dict_i['min_val']), instead of setting it equal to the single value
        #         IT REPLACES ALL ELEMENTS OF THE LIST WITH THE SINGLE VALUE!   
        #       If, instead, I wanted to set the value equal to a string or int, it would work as expected (i.e., the row would take the value 
        #         of the single string/int, not a list of those)
        #       Super dumb functionality, I have to believe this is a bug/mistake in pandas.
        #       In any case, the workaround is to first set the value equal to NaN, then set equal to overlap_dict_i['min_val'] 
        df_i.iloc[0, ovrlp_intrvl_0_col_idx] = np.nan
        df_i.iloc[0, ovrlp_intrvl_1_col_idx] = np.nan
        #-----
        df_i.iloc[0, ovrlp_intrvl_0_col_idx] = overlap_dict_i['min_val']
        df_i.iloc[0, ovrlp_intrvl_1_col_idx] = overlap_dict_i['max_val']
        #-------------------------
        # Reset the index and append to dfs_list
        df_i=df_i.reset_index()
        dfs_list.append(df_i)
    #-------------------------
    # Finally, concatenate dfs_list and return
    return_df = pd.concat(dfs_list)
    #-------------------------
    if tmp_col is not None:
        return_df = return_df.drop(columns=tmp_col)
    #-------------------------
    if drop_idx_cols:
        return_df = return_df.drop(columns=idx_col_names)
    #-------------------------
    if maintain_original_cols:
        if not drop_idx_cols:
            return_df = return_df.set_index(idx_col_names)
        assert(len(set(og_cols).difference(set(return_df.columns)))==0)
        return_df = return_df[og_cols]
    #-------------------------
    return return_df
    
    
#--------------------------------------------------    
def find_overlap_intervals_in_df(
    df, 
    ovrlp_intrvl_0_col, 
    ovrlp_intrvl_1_col, 
    fuzziness, 
    int_idxs=True
):
    r"""
    Given a pd.DataFrame df with intervals defined by starting values and ending values in columns ovrlp_intrvl_0_col
      and ovrlp_intrvl_1_col, respectively, find the reduced set of overlap intervals.
    The fuzziness argument sets how close two intervals must be to be considered overlapping (see Utilities.get_fuzzy_overlap_intervals 
      for more information.)
    Returns a list of dict objects, one for each overlap interval.
        The keys for each dict object are 'min_val', 'max_val', and 'idxs',  
        By default (when int_idxs==True) 'idxs' correspond to the integer index locations of the rows included in the overlap.
        If int_idxs==False, the index labels are instead used.
          
    NOTE: The function/operation calling this method should ensure all values are appropriate, meaning ensure that for each row
            row_i[ovrlp_intrvl_1_col] > row_i[ovrlp_intrvl_0_col]
          This method will not attempt to remedy any incorrect values, but will simply assert this is true
    """
    #-------------------------
    # Make sure fuzziness is compatible with ovrlp_intrvl_0(1)_col
    try:
        test_0 = df.iloc[0][ovrlp_intrvl_0_col]+fuzziness
        test_1 = df.iloc[0][ovrlp_intrvl_1_col]+fuzziness
    except:
        print(f'''
        In consolidate_df_according_to_fuzzy_overlap_intervals: Incompatible fuzziness type
            type(fuzziness) = {type(fuzziness)}
            df[ovrlp_intrvl_0_col].dtype = {df[ovrlp_intrvl_0_col].dtype}
            df[ovrlp_intrvl_1_col].dtype = {df[ovrlp_intrvl_1_col].dtype}
        CRASH IMMINENT!
        ''')
    #-------------------------
    # First, make sure the second element in each tuple should be greater than the first
    # NOTE: The second element can be NaN (either NaT for times, or NaN otherwise)
    #         Apparently, NaN evaluates as False when compared in any manner to anything else
    #         (i.e., anything>NaN = False, anything<NaN = False, anything==NaN = False)
    #       Thus, df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col] will result in False whenever
    #         df[ovrlp_intrvl_1_col] is NaN, and therefore the assertion:
    #           assert(all(df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]))
    #         would fail in in unwanted circumstances, since NaN value means and open ended interval
    #         and therefore any beginning value should be considered less than a NaN value.
    #       Therefore, instead of the single-line assertion above, I will include two assertions,
    #         one to ensure df[ovrlp_intrvl_0_col] doesn't contain any NaNs, and one to ensure
    #         either df[ovrlp_intrvl_1_col] is a Nan or df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]
    assert(all(df[ovrlp_intrvl_0_col].notna()))
    assert(all((df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]) | (df[ovrlp_intrvl_1_col].isna())))
    # For above assertion, probably could have isntead used: 
    #   assert(all(df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col].fillna(pd.Timestamp.max)))
    #-------------------------
    # Sort ranges, as will be necessary for this procedure
    df = df.sort_values(by=[ovrlp_intrvl_0_col, ovrlp_intrvl_1_col])
    #-------------------------
    # Set the first range in overlaps simply as from the first entry in df
    overlaps = []
    current_beg, current_end = df.iloc[0][[ovrlp_intrvl_0_col, ovrlp_intrvl_1_col]]
    overlaps.append(
        dict(min_val=current_beg, max_val=current_end)
    )
    if int_idxs:
        overlaps[0]['idxs'] = [0]
    else:
        overlaps[0]['idxs'] = [df.index[0]]
    #-------------------------
    # Iterate through and create the overlaps items, each of which will be a dict with key
    #   equal to min_val, max_val, and idxs (where idxs identifies which indices from df
    #   belong to each overlap group)
    if int_idxs:
        df_to_itr = df.reset_index()
    else:
        df_to_itr = df
    #-----
    for i, (idx, row) in enumerate(df_to_itr.iterrows()):
        if i==0:
            continue
        #---------------
        beg = row[ovrlp_intrvl_0_col]
        end = row[ovrlp_intrvl_1_col]
        if beg > current_end+fuzziness:
            # beg after current end (with fuzziness buffer), so new interval needed
            # NOTE: beg > current_end+fuzziness will evaluate to False whenever current_end
            #       is NaN, which is the desired functionality.
            overlaps.append(dict(min_val=beg, max_val=end, idxs=[idx]))
            current_beg, current_end = beg, end
        else:
            # beg <= current_end+fuzziness, so overlap
            # The beg of overlaps[-1] remains the same, but the end of overlaps[-1] should be changed to
            #   the max of current_end and end.
            # Also, idx needs to be added to the overlap
            # NOTE: max(any_non_NaT, NaT) = any_non_NaT (remember, NaN evaluates as False when compared in any manner to anything else), 
            #       which is not the funcionality I want, as an end of NaT essentially means no end (e.g., a meter which is still in service), 
            #       and should therefore be treated as Inf.  Thus, cannot simply use the one-liner 'current_end = max(current_end, end)'
            # NOTE 2: Cannot simply do 'if pd.isna(current_end) or pd.isna(end)' in single line because this function is
            #         designed to work with various data types, so in such a scenario it would be unclear what to set
            #         current_end to (e.g., should it be pd.NaT, pd.NaN, etc?).
            #         Thus, instead of if-else, need if-elif-else
            #current_end = max(current_end, end)
            if pd.isna(current_end):
                current_end=current_end
            elif pd.isna(end):
                current_end = end
            else:
                current_end = max(current_end, end)
            overlaps[-1]['max_val'] = current_end
            overlaps[-1]['idxs'].append(idx)
    #-------------------------
    return overlaps

#--------------------------------------------------
def consolidate_df_group_according_to_fuzzy_overlap_intervals(
    df_i, 
    ovrlp_intrvl_0_col, 
    ovrlp_intrvl_1_col, 
    gpd_cols, 
    fuzziness, 
    assert_single_overlap=False, 
    maintain_original_cols=False, 
    enforce_list_cols_for_1d_df=True, 
    allow_duplicates_in_lists=False, 
    allow_NaNs_in_lists=False
):
    r"""
    This is for the specific case of a df group.
    It is expected, and enforced, that there exists a single unique value for each column in df_i outside of
      ovrlp_intrvl_0_col and ovrlp_intrvl_1_col
      
    gpd_cols:
        For the typical case where this function is used inside of a groupby().apply(lambda x:) function, the
          gpd_cols should match those input into groupby (for a typical use case, see 
          Utilities_df.consolidate_df_according_to_fuzzy_overlap_intervals)
        Each of these columns should contain a single unique value.
        This input is needed so the function knows which columns to collect in lists (those outside of gpd_cols+
          [ovrlp_intrvl_0_col, ovrlp_intrvl_1_col])
    """
    #-------------------------
    if not isinstance(gpd_cols, list):
        gpd_cols = [gpd_cols]
    #-------------------------
    assert(len(set(gpd_cols).intersection(set([ovrlp_intrvl_0_col, ovrlp_intrvl_1_col])))==0)
    assert((df_i[gpd_cols].nunique()<=1).all())
    cols_to_collect_in_lists = [x for x in df_i.columns.tolist() if x not in gpd_cols+[ovrlp_intrvl_0_col, ovrlp_intrvl_1_col]]
    #-------------------------
    og_cols = df_i.columns
    #-------------------------
    if df_i.shape[0]<=1 and maintain_original_cols and len(cols_to_collect_in_lists)>0: 
        # Reason for maintain_original_cols:
        #   If this is used in .groupby, it is important for all to have same shape/labelling.
        #   So, if maintain_original_cols is False, in order for a df_i with df_i.shape[0]==1 to fit
        #     into the .groupby procedure, it must go through the steps below to ensure it has the same
        #     form as the other groups.
        if enforce_list_cols_for_1d_df:
        # Make any cols_to_collect_in_lists into lists EXCEPT ovrlp_intrvl_0_col and ovrlp_intrvl_1_col...
            lst_cols_1d = [x for x in cols_to_collect_in_lists if x not in [ovrlp_intrvl_0_col, ovrlp_intrvl_1_col]] #1d for 1-dimensional DF
            df_i[lst_cols_1d] = df_i[lst_cols_1d].apply(lambda x: [x.tolist()])
        return df_i
    #-------------------------
    if len(cols_to_collect_in_lists)>0:
        # Collect the values from cols_to_collect_in_lists, and remove from df_i
        if allow_NaNs_in_lists:
            if allow_duplicates_in_lists:
                agg_func = agg_func_list
            else:
                agg_func = agg_func_unq_list
        else:
            if allow_duplicates_in_lists:
                agg_func = agg_func_list_dropna
            else:
                agg_func = agg_func_unq_list_dropna
        list_cols_df = df_i.groupby(gpd_cols, as_index=False, group_keys=False)[cols_to_collect_in_lists].agg(lambda x: agg_func(x))
        assert(list_cols_df.shape[0]==1)
        #-----
        df_i = df_i.drop(columns=cols_to_collect_in_lists)
    #-------------------------
    overlaps = find_overlap_intervals_in_df(
        df=df_i, 
        ovrlp_intrvl_0_col=ovrlp_intrvl_0_col, 
        ovrlp_intrvl_1_col=ovrlp_intrvl_1_col, 
        fuzziness=fuzziness, 
        int_idxs=True
    )
    #-------------------------
    if assert_single_overlap and len(overlaps)!=1:
        print(f'assert_single_overlap and len(overlaps)={len(overlaps)}')
        print(overlaps)
        print(f'df_i.head():\n{df_i.head()}')
        assert(0)
    #-------------------------
    # The number of overlaps must be less than or equal to the shape of df_i
    assert(len(overlaps) <= df_i.shape[0])

    # Already know (assertion at top of function) that all columns are uniform except for [ovrlp_intrvl_0_col, ovrlp_intrvl_1_col]
    # Therefore, for return df, just grab as many rows as needed from df_i, and replace the [ovrlp_intrvl_0_col, ovrlp_intrvl_1_col]
    #   values with those from overlaps
    return_df = df_i.iloc[0:len(overlaps)].copy()
    assert(return_df.shape[0]==len(overlaps))
    #-------------------------
    # Determine the index positions of ovrlp_intrvl_0_col and ovrlp_intrvl_1_col, as these will be needed
    #   below to set to values in the consolidated DF using iloc
    ovrlp_intrvl_0_col_idx = find_idxs_in_highest_order_of_columns(return_df, ovrlp_intrvl_0_col)
    assert(len(ovrlp_intrvl_0_col_idx)==1)
    ovrlp_intrvl_0_col_idx=ovrlp_intrvl_0_col_idx[0]
    #-----
    ovrlp_intrvl_1_col_idx = find_idxs_in_highest_order_of_columns(return_df, ovrlp_intrvl_1_col)
    assert(len(ovrlp_intrvl_1_col_idx)==1)
    ovrlp_intrvl_1_col_idx=ovrlp_intrvl_1_col_idx[0]    
    #----------
    for i_row in range(return_df.shape[0]):
        return_df.iloc[i_row, ovrlp_intrvl_0_col_idx] = overlaps[i_row]['min_val']
        return_df.iloc[i_row, ovrlp_intrvl_1_col_idx] = overlaps[i_row]['max_val']
    #-------------------------
    if len(cols_to_collect_in_lists)>0:
        # Add back on the list_cols_df
        return_df = return_df.merge(
            pd.concat([list_cols_df]*return_df.shape[0]), 
            left_on=gpd_cols, right_on=gpd_cols, how='left'
        )
    #-------------------------
    if maintain_original_cols:
        assert(len(set(og_cols).difference(set(return_df.columns)))==0)
        return_df = return_df[og_cols]
    #-------------------------
    return return_df


#--------------------------------------------------
def consolidate_df_according_to_fuzzy_overlap_intervals(
    df, 
    ovrlp_intrvl_0_col, 
    ovrlp_intrvl_1_col, 
    fuzziness, 
    groupby_cols, 
    assert_single_overlap=True, 
    cols_to_collect_in_lists=None, 
    recover_uniqueness_violators=False, 
    gpby_dropna=True, 
    allow_duplicates_in_lists=False, 
    allow_NaNs_in_lists=False
):
    r"""
    """
    #-------------------------
    # Don't want to alter df itself
    df = df.copy()
    og_cols = df.columns
    #-------------------------
    if groupby_cols is None:
        tmp_col = Utilities.generate_random_string()
        df[tmp_col] = 1
        groupby_cols = [tmp_col]
    else:
        tmp_col = None
    #-------------------------
    if cols_to_collect_in_lists is None:
        cols_to_collect_in_lists=[x for x in df.columns if x not in groupby_cols]
    #-------------------------
    assert(len(set(groupby_cols).intersection(set([ovrlp_intrvl_0_col, ovrlp_intrvl_1_col])))==0)
    #-------------------------
    # Make sure fuzziness is compatible with ovrlp_intrvl_0(1)_col
    try:
        test_0 = df.iloc[0][ovrlp_intrvl_0_col]+fuzziness
        test_1 = df.iloc[0][ovrlp_intrvl_1_col]+fuzziness
    except:
        print(f'''
        In consolidate_df_according_to_fuzzy_overlap_intervals: Incompatible fuzziness type
            type(fuzziness) = {type(fuzziness)}
            df[ovrlp_intrvl_0_col].dtype = {df[ovrlp_intrvl_0_col].dtype}
            df[ovrlp_intrvl_1_col].dtype = {df[ovrlp_intrvl_1_col].dtype}
        CRASH IMMINENT!
        ''')
        
    #-------------------------
    # First, make sure the second element in each tuple should be greater than the first
    # NOTE: The second element can be NaN (either NaT for times, or NaN otherwise)
    #         Apparently, NaN evaluates as False when compared in any manner to anything else
    #         (i.e., anything>NaN = False, anything<NaN = False, anything==NaN = False)
    #       Thus, df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col] will result in False whenever
    #         df[ovrlp_intrvl_1_col] is NaN, and therefore the assertion:
    #           assert(all(df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]))
    #         would fail in in unwanted circumstances, since NaN value means and open ended interval
    #         and therefore any beginning value should be considered less than a NaN value.
    #       Therefore, instead of the single-line assertion above, I will include two assertions,
    #         one to ensure df[ovrlp_intrvl_0_col] doesn't contain any NaNs, and one to ensure
    #         either df[ovrlp_intrvl_1_col] is a Nan or df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]
    # NOTE: I actually have found at least one case where ovrlp_intrvl_1_col<ovrlp_intrvl_0_col (by one second)
    #       Instead of throwing an assertion error, I will instead print a warning and simply remove the offending entry(ies)
    if any(df[ovrlp_intrvl_0_col].isna()):
        print(f'''
            !!!!! WARNING !!!!! In consolidate_df_according_to_fuzzy_overlap_intervals:
            df[ovrlp_intrvl_0_col] has NaN values!
            df[ovrlp_intrvl_0_col].isna().sum() = {df[ovrlp_intrvl_0_col].isna().sum()}
            df.shape[0]                         = {df.shape[0]}
            Row containing these NaNs will be omitted!
        ''')
        df = df[df[ovrlp_intrvl_0_col].notna()]
    #-----
    # NOTE: In printing the output below, the parentheses in (~scnd_gt_frst_srs).sum() are important!
    #       (~scnd_gt_frst_srs).sum() != ~scnd_gt_frst_srs.sum() (the latter essentially equals -1*scnd_gt_frst_srs.sum())
    scnd_gt_frst_srs = ((df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]) | (df[ovrlp_intrvl_1_col].isna()))
    if any(~scnd_gt_frst_srs):
        print(f'''
            !!!!! WARNING !!!!! In consolidate_df_according_to_fuzzy_overlap_intervals:
            df has values for which df[ovrlp_intrvl_0_col]>=df[ovrlp_intrvl_1_col]!
            Number of violators = {(~scnd_gt_frst_srs).sum()}
            df.shape[0]         = {df.shape[0]}
            Rows containing these violators will be omitted!
        ''')
        df = df[scnd_gt_frst_srs]
    # Now, at this stage assertions should both pass    
    assert(all(df[ovrlp_intrvl_0_col].notna()))
    assert(all((df[ovrlp_intrvl_0_col]<df[ovrlp_intrvl_1_col]) | (df[ovrlp_intrvl_1_col].isna())))
    # Also, sort ranges, as will be necessary for this procedure
    df = df.sort_values(by=[ovrlp_intrvl_0_col, ovrlp_intrvl_1_col])
    #-------------------------
    return_df = df.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(
        lambda x: consolidate_df_group_according_to_fuzzy_overlap_intervals(
            df_i=x, 
            ovrlp_intrvl_0_col=ovrlp_intrvl_0_col, 
            ovrlp_intrvl_1_col=ovrlp_intrvl_1_col, 
            gpd_cols=groupby_cols, 
            fuzziness=fuzziness, 
            assert_single_overlap=assert_single_overlap, 
            maintain_original_cols=True, 
            enforce_list_cols_for_1d_df=True, 
            allow_duplicates_in_lists=allow_duplicates_in_lists,
            allow_NaNs_in_lists=allow_NaNs_in_lists
        )
    )
    return return_df


#----------------------------------------------------------------------------------------------------
def are_all_series_elements_one_of_types(srs, types):
    r"""
    Checks if all list elements are one of the types listed in types.
    NOTE: This will return True even when there are multiple types of elements in srs, so long
          as all are of one of the types in types.
    If one also wants all elements in srs to be of a single type, use are_all_series_elements_one_of_types_and_homogeneous
      instead.
    """
    for i,typ in enumerate(types):
        if i==0:
            bool_mask = (srs.apply(type)==typ)
        else:
            bool_mask = bool_mask|(srs.apply(type)==typ)
    return all(bool_mask)

#--------------------------------------------------
def are_all_series_elements_one_of_types_and_homogeneous(lst, types):
    r"""
    Checks if all elements in lst are of a single type, and that single type is one of those found in types.
    """
    for typ in types:
        bool_mask = (srs.apply(type)==typ)
        if all(bool_mask):
            return True
    return False

#--------------------------------------------------
def get_list_elements_mask_for_series(srs):
    r"""
    Returns a series (mask) with boolean values identifying which elements are lists.
    Note, as columns are DFs are simply series, this function applies for those as well (e.g., one may pass srs = df[col])
    """
    #-------------------------
    assert(isinstance(srs, pd.Series))
    list_elements_mask = ((srs.apply(type)==np.ndarray)|(srs.apply(type)==list)|(srs.apply(type)==tuple))
    return list_elements_mask
    

#--------------------------------------------------    
def get_df_cols_with_list_elements(
    df
):
    r"""
    If any element in a column contains a list object, then the column is defined as containing list elements.
    Also, only object type columns can possibly contain lists, so they are the only ones needing inspected.
    """
    #-------------------------
    is_obj_dtype_srs = df.dtypes.apply(lambda x: is_object_dtype(x))
    cols_to_inspect = is_obj_dtype_srs[is_obj_dtype_srs==True].index.tolist()
    #-------------------------
    cols_w_lists = []
    for col_i in cols_to_inspect:
        if get_list_elements_mask_for_series(df[col_i]).any():
            cols_w_lists.append(col_i)
    #-------------------------
    return cols_w_lists
    
    
#----------------------------------------------------------------------------------------------------
def train_test_split_df_group(
    X,
    y, 
    groups, 
    test_size, 
    random_state=None
    
):
    r"""
    This is simply a train-test split according to the outage groups.
    i.e., this enforces that all entries for a given group remain together (either all in train or all in test)
          and never split across train/test
          
    NOTE: If input is list, return value will be np.ndarray
          If input is np.ndarray, output np.ndarray
          If inputs pd.DataFrame/pd.Series, output pd.DataFrame/pd.Series
    """
    #-------------------------
    # The methods expect X and y to be either np.ndarrays, lists or pd.DataFrame/pd.Series (respectively)
    assert(Utilities.is_object_one_of_types(X, [np.ndarray, list, pd.DataFrame]))
    assert(Utilities.is_object_one_of_types(y, [np.ndarray, list, pd.Series]))
    assert(len(X)==len(y)) # next(split) would have failed if this wasn't true, but having it here 
    #                      makes it easier to locate and debug if it ever happens
    #-------------------------
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split = gss.split(X, y, groups=groups)
    train_idxs, test_idxs = next(split)
    #-------------------------
    # In order to grab elements simply using list of indices (instead of looping through or whatever)
    #   X and y must be np.ndarrays
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)
    #-------------------------
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idxs]
        X_test  = X.iloc[test_idxs]
    else:
        X_train = X[train_idxs]
        X_test  = X[test_idxs]
    #-----
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idxs]
        y_test  = y.iloc[test_idxs]
    else:
        y_train = y[train_idxs]
        y_test  = y[test_idxs]
    #-------------------------
    return X_train, X_test, y_train, y_test


#----------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------
def set_kit_ids(df, col_for_extraction, 
                col_for_placement='kit_id', gold_standard='S002'):
    # col_for_extraction is which column to be used to extract the kit id.
    #   This should typically be the nist or spca xml path
    # col_for_placement will be the new column created housing the kit id
    df[col_for_placement] = df[col_for_extraction].apply(lambda x: Utilities.get_kit_id(x, gold_standard=gold_standard)[0])
    return df

#--------------------------------------------------
def set_kit_ids_parent_ids_dates(df, col_for_extraction, 
                                 cols_for_placement=['kit_id', 'parent_kit_id', 'date'], 
                                 gold_standard='S002'):
    # see set_kit_ids for info
    assert(len(cols_for_placement)==3)
    df[cols_for_placement[0]], df[cols_for_placement[1]], df[cols_for_placement[2]] = zip(*df[col_for_extraction].apply(lambda x: Utilities.get_kit_id(x, gold_standard=gold_standard)))
    return df
    
