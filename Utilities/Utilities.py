#!/usr/bin/env python

"""
General utilities
"""

__author__ = "Jesse Buxton"
__email__  = "buxton.45.jb@gmail.com"
__status__ = "Personal"

import os
import sys
import glob
import re
from pathlib import Path
from difflib import SequenceMatcher
import shutil
import errno


from enum import Enum

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import pyodbc

import random
import string
import copy
import typing
from natsort import natsorted, ns
import datetime
import time
from numbers import Number

from functools import reduce
import operator
import bisect
import itertools

import Utilities_config

from importlib import reload
# NOTE: To reload a class imported as, e.g., 
# from module import class
# One must call:
#   1. import module
#   2. reload module
#   3. from module import class

#--------------------------------------------------------------------
def is_numeric(a):
    return isinstance(a, Number)
#--------------------------------------------------------------------
def get_analysis_dir():
    return Utilities_config.get_analysis_dir()
    
def get_utilities_dir():
    return Utilities_config.get_utilities_dir()
    
def get_sql_aids_dir():
    return Utilities_config.get_sql_aids_dir()

def get_local_data_dir():
    return Utilities_config.get_local_data_dir()
#--------------------------------------------------------------------
def get_pwd(pwd_type='Main', pwd_file_path=None):
    # Currently, Oracle pwd different from others
    # So, choices for pwd_type are e.g. 'Main' or 'Oracle'
    #-----
    #NOTE: If file is large, and memory problems are possible,
    # use mmap.mmap instead!
    found_line, found_idx = None, None
    if pwd_file_path is None:
        pwd_file_path = Utilities_config.get_pwd_file_path()
    with open(pwd_file_path, 'r') as f:
        for line in f:
            idx = line.find(pwd_type)
            if idx>-1:
                found_line, found_idx = line, idx    
    pwd = found_line[found_idx+len(pwd_type)+len(':'):]
    pwd = pwd.strip()
    return pwd
    
def get_prod_llap_hive_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    dsn = f'{db_user}-Hive-Prod-LLAP'
    pwd_hive = get_pwd()
    conn_str = 'DSN=' + dsn + ';Uid=' + db_user + ';PWD=' + pwd_hive + ';'
    conn_hive = pyodbc.connect(conn_str, autocommit=True)
    
def get_emr_prod_aws_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    dsn = 'EMR Prod'
    pwd_aws = get_pwd()
    conn_str = 'DSN=' + dsn + ';Uid=' + db_user + ';PWD=' + pwd_aws + ';'
    conn_aws = pyodbc.connect(conn_str, autocommit=True)
    return conn_aws
    
def get_athena_prod_aws_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    db_user = f'{db_user}@CORP.AEPSC.COM'
    dsn = Utilities_config.get_athena_prod_dsn()
    pwd_aws = get_pwd()
    conn_str = 'DSN=' + dsn + ';Uid=' + db_user + ';PWD=' + pwd_aws + ';'
    conn_aws = pyodbc.connect(conn_str, autocommit=True)
    return conn_aws
    
def get_athena_prod_aws_connection2(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    db_user = f'{db_user}@CORP.AEPSC.COM'
    dsn = 'SomeName'
    pwd_aws = get_pwd()
    conn_str = 'DSN=' + dsn + ';Uid=' + db_user + ';PWD=' + pwd_aws + ';'
    conn_aws = pyodbc.connect(conn_str, autocommit=True)
    return conn_aws
    
def get_athena_dev_aws_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    db_user = f'{db_user}@CORP.AEPSC.COM'
    dsn = Utilities_config.get_athena_dev_dsn()
    pwd_aws = get_pwd()
    conn_str = 'DSN=' + dsn + ';Uid=' + db_user + ';PWD=' + pwd_aws + ';'
    conn_aws = pyodbc.connect(conn_str, autocommit=True)
    return conn_aws
    
def get_athena_qa_aws_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    db_user = f'{db_user}@CORP.AEPSC.COM'
    dsn = Utilities_config.get_athena_qa_dsn()
    pwd_aws = get_pwd()
    conn_str = 'DSN=' + dsn + ';Uid=' + db_user + ';PWD=' + pwd_aws + ';'
    conn_aws = pyodbc.connect(conn_str, autocommit=True)
    return conn_aws

def get_utldb01p_oracle_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    pwd_oracle = get_pwd('Oracle')
    # dsn = Utilities_config.get_utldb01p_dsn()
    # conn_outages = pyodbc.connect(f'Driver={{Oracle in InstantClient_12_64}};DBQ= aep01dbadm01/{dsn};Uid=' + db_user + ';Pwd=' + pwd_oracle)
    conn_outages = pyodbc.connect(f'DSN=UTLDB01P;Uid={db_user};Pwd={pwd_oracle}')
    return conn_outages
    
def get_eemsp_oracle_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    pwd_oracle = get_pwd('Oracle')
    dsn = Utilities_config.get_eemsp_dsn()
    conn_outages = pyodbc.connect(f'Driver={{Oracle in InstantClient_12_64}};DBQ= {dsn};Uid=' + db_user + ';Pwd=' + pwd_oracle)
    return conn_outages

def get_eddsp_oracle_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    pwd_oracle = get_pwd('Oracle')
    conn_eddsp = pyodbc.connect(f'DSN=EDDSP;Uid={db_user};Pwd={pwd_oracle}')
    return conn_eddsp

#--------------------------------------------------------------------
def is_table_info_df_row_empty(table_info_df, empty_row_defn, idx):
    #NOTE: This will not work if an empty value = e.g., np.nan
    #      This is because np.nan != np.nan
    #      Such functionality can be included if desired, but left simple for now.
    #NOTE: Should work if an empty_value is None because None==None
    assert(all(x in table_info_df.columns for x in empty_row_defn.keys()))

    is_empty = True
    for col_name, empty_val in empty_row_defn.items():
        if table_info_df.loc[idx][col_name] != empty_val:
            is_empty = False
            break
    return is_empty

def clean_aws_table_info(table_info_df, 
                         end_row_id_col='col_name', end_row_id_val='# Partition Information', 
                         empty_row_defn = {'col_name':'', 'data_type':None, 'comment':None}):
    end_rows_id = table_info_df.index[table_info_df[end_row_id_col]==end_row_id_val]
    assert(len(end_rows_id)==1)
    end_rows_id = end_rows_id[0]

    # Row before '# Partition Information' should be empty, wbere by empty I mean specifically
    #   col_name=='' and data_type=comment=None
    if (empty_row_defn is not None and 
        is_table_info_df_row_empty(table_info_df, empty_row_defn, end_rows_id-1)):
        end_rows_id -= 1
    table_info_df = table_info_df.iloc[:end_rows_id]
    return table_info_df
    
def get_aws_table_info(conn_aws, schema_name, table_name, run_clean=False, **kwargs):
    # e.g., schema_name = 'default' and table_name = 'meter_premise'
    sql = f"desc extended {schema_name}.{table_name}"
    df = pd.read_sql(sql, conn_aws)
    #-------------------
    if run_clean:
        empty_row_defn = kwargs.get('empty_row_defn', None)
        if empty_row_defn is None:
            df = clean_aws_table_info(df)
        else:
            df = clean_aws_table_info(df, empty_row_defn=empty_row_defn)
    #-------------------
    return df
    
def get_athena_table_info(conn_athena, schema_name, table_name, run_clean=False, **kwargs):
    # e.g., schema_name = 'default' and table_name = 'meter_premise'
    sql = f"DESCRIBE EXTENDED {schema_name}.{table_name}"
    df = pd.read_sql(sql, conn_athena)
    #-------------------
    if run_clean:
        empty_row_defn = kwargs.get('empty_row_defn', None)
        if empty_row_defn is None:
            df = clean_aws_table_info(df)
        else:
            df = clean_aws_table_info(df, empty_row_defn=empty_row_defn)
    #-------------------
    return df

def get_outages_table_info(conn_outages, schema_name, table_name):
    # e.g., schema_name = 'DOVSADM' and table_name = 'DOVS_OUTAGE_FACT'
    sql = (
    f"""
    SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, COLUMN_ID
    FROM all_tab_cols
    WHERE OWNER = '{schema_name}' AND TABLE_NAME = '{table_name}'
    """
    )
    df = pd.read_sql(sql, conn_outages)
    
    # There is a row with COLUMN_NAME = 'SYS_...' and COLUMN_ID = NaN
    # Remove this row.
    df = df.dropna(subset=['COLUMN_ID'])        
    return df
    
#--------------------------------------------------------------------
def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

#--------------------------------------------------------------------
def is_hashable(input_obj):
    r"""
    Returns true of input_obj is hashable, else False
    """
    #-------------------------
    return isinstance(input_obj, typing.Hashable)

#--------------------------------------------------------------------
def are_all_list_elements_of_type(lst, typ):
    assert(isinstance(lst, list) or isinstance(lst, tuple))
    for el in lst:
        if not isinstance(el, typ):
            return False
    return True

def are_all_list_elements_one_of_types(lst, types):
    r"""
    Checks if all list elements are one of the types listed in types.
    NOTE: This will return True even when there are multiple types of elements in lst, so long
          as all are of one of the types in types.
    If one also wants all elements in lst to be of a single type, use are_all_list_elements_one_of_types_and_homogeneous
      instead.
    """
    assert(isinstance(lst, list) or isinstance(lst, tuple))
    # For whatever reason, if there is only a single element,
    #   using a tuple does not work, as, 
    #   e.g., type((str)) returns type not tuple
    #Therefore, types must be a list.
    assert(isinstance(types, list))
    for el in lst:
        is_el_good=False
        for typ in types:
            if isinstance(el, typ):
                is_el_good=True
                break
        if not is_el_good:
            return False
    return True
    
def are_all_list_elements_one_of_types_and_homogeneous(lst, types):
    r"""
    Checks if all elements in lst are of a single type, and that single type is one of those found in types.
    """
    for typ in types:
        if are_all_list_elements_of_type(lst, typ):
            return True
    return False

def is_object_one_of_types(obj, types):
    return are_all_list_elements_one_of_types([obj], types)
    
def are_list_elements_lengths_homogeneous(lst, length=None):
    len_0 = len(lst[0])
    if length is not None and len_0!=length:
        return False
    for el in lst:
        if len(el) != len_0:
            return False
    return True
    
#--------------------------------------------------------------------
def are_all_lists_eq(list_of_lists):
    r"""
    Checks whether all elements (lists) in a list of lists are equal.
    This was designed specifically for use in the function combine_PNs_in_best_ests_df from the check_DOVS
      project, but could possibly be useful elsewhere.
    """
    #-------------------------
    assert(isinstance(list_of_lists, list))
    assert(are_all_list_elements_one_of_types_and_homogeneous(list_of_lists, [list, tuple, set]))
    #-----
    if not are_list_elements_lengths_homogeneous(list_of_lists):
        return False
    #-----
    lst_0 = natsorted(list_of_lists[0])
    for lst_i in list_of_lists:
        if natsorted(lst_i) != lst_0:
            return False
    #-----
    return True    

#--------------------------------------------------------------------
def melt_list_of_lists(lol):
    r"""
    """
    #-------------------------
    # Make sure lol is, in fact, a list of lists (or a list of tuples)
    assert(isinstance(lol, list))
    assert(are_all_list_elements_one_of_types(lol, [list, tuple]))
    #-------------------------
    melted = list(itertools.chain.from_iterable(lol))
    return melted

def melt_list_of_lists_2(lol):
    r"""
    This allows non-list/tuple members of lol.
    Any element of lol which is not a list/tuple will be converted to a single element list
    """
    #-------------------------
    # Make sure lol is, in fact, a list 
    assert(isinstance(lol, list))
    #-----
    lol = [x if is_object_one_of_types(x, [list,tuple]) else [x] for x in lol]
    return melt_list_of_lists(lol)
    
#--------------------------------------------------------------------
def is_list_nested(
    lst, 
    enforce_if_one_all=True
):
    r"""
    enforce_if_one_all:
        If one element of lst is a list, all must be
    """
    #-------------------------
    is_nested = False
    if any(is_object_one_of_types(x, [list, tuple]) for x in lst):
        is_nested = True
        if enforce_if_one_all:
            assert(all(is_object_one_of_types(x, [list, tuple]) for x in lst))
    return is_nested

#--------------------------------------------------------------------
def supplement_dict_with_default_values(
    to_supplmnt_dict    , 
    default_values_dict , 
    extend_any_lists    = False, 
    inplace             = False
):
    r"""
    Adds key/values from default_values_dict to to_supplmnt_dict.
    WARNING: Suggested to keep inplace=False to avoid weird complications (see IMPORTANT NOTE below)
    If a key IS NOT contained in to_supplmnt_dict
      ==> a new key/value pair is simply created.
    If a key IS already contained in to_supplmnt_dict
      ==> value (to_supplmnt_dict[key]) is kept 
          UNLESS extend_any_lists==True and to_supplmnt_dict[key] and/or default_values_dict[key] is a list,
            in which case the two lists are combined
    NOTE: New functionality allows for joining of (single level) nested dicts when extend_any_lists==True.
          (There is probably a clever recursive way to allow nested dicts of any level, but this suits my needs for now)
          In such a case, to_supplmnt_dict[key] and default_values_dict[key] must both be dict objects, otherwise
            no joining will occur.
          The two nested dicts with be joined as (default_values_dict[key] | to_supplmnt_dict[key]), meaning that if there
            are any shared keys, the value from to_supplmnt_dict[key] will be kept.
            
    IMPORTANT NOTE:
        Be careful when using inplace=True.
        The safest bet is to keep it set to False.
        -----
        In Python, a default variable for a function is only evaluated and set once.
        Python makes a copy of the reference and from then on it always passes that reference as the default value.
        No re-evaluation is done.
        I saw very strange behavior when developing assess_outage_inclusion_requirements, which has a parameter check_found_ami_for_all_SNs_kwargs which
          was initially set with a default value = {}
        In the code body, I joined check_found_ami_for_all_SNs_kwargs with dflt_check_found_ami_for_all_SNs_kwargs using this function and inplace=True
        Running the function just once was fine.
        HOWEVER, upon running the code a second time (without explicitly setting a value for check_found_ami_for_all_SNs_kwargs, so the default argument was used)
          I found that the default (starting) value for check_found_ami_for_all_SNs_kwargs was not {}, but was instead the result of the 
          supplement_dict_with_default_values from the previous run!
        This error can be avoided by setting the default value to None instead of {} and/or using inplace=False.
        I now do both to be safe!
        THIS IS THE REASON THAT THE DEFAULT inplace VALUE IN THIS FUNCTION WAS SWITCHED FROM TRUE TO FALSE!
    """
    #-------------------------
    if to_supplmnt_dict is None:
        to_supplmnt_dict = {}
    #---------------
    if not inplace:
        to_supplmnt_dict = copy.deepcopy(to_supplmnt_dict)
    #---------------
    for key in default_values_dict:
        if key not in to_supplmnt_dict:
            to_supplmnt_dict[key] = default_values_dict[key]
            continue
        #---------------
        if not extend_any_lists:
            continue
        #---------------
        if (
            isinstance(default_values_dict[key], list) or 
            isinstance(to_supplmnt_dict[key], list)
        ):
            if not(isinstance(to_supplmnt_dict[key], list)):
                to_supplmnt_dict[key] = [to_supplmnt_dict[key]]
            if not(isinstance(default_values_dict[key], list)):
                default_values_dict[key] = [default_values_dict[key]]
            to_supplmnt_dict[key].extend([x for x in default_values_dict[key] 
                                          if x not in to_supplmnt_dict[key]])
        #---------------
        # New functionality allows for nested dict (only one level, though)
        if(
            isinstance(default_values_dict[key], dict) and 
            isinstance(to_supplmnt_dict[key], dict)
        ):
            to_supplmnt_dict[key] = (default_values_dict[key]|to_supplmnt_dict[key])
    #-------------------------
    return to_supplmnt_dict
    
    
#--------------------------------------------------------------------
def get_from_nested_dict(nested_dict, keys_path_list):
    r"""
    Access a nested object in nested_dict by keys_path_list sequence.
    Works equally well for lists or a mix of dictionaries and lists, so the names should really be get_by_path()
    --------------------------------------------------------------
    EXAMPLE
    Instead of calling nested_dict['a']['r'] in the example below, one
    may instead call get_from_nested_dict(nested_dict, ['a','r'])
    >>> get_from_nested_dict(
    ...                      nested_dict={
    ...                          "a":{
    ...                              "r": 'Correct Result',
    ...                              "s": 2,
    ...                              "t": 3
    ...                              },
    ...                          "b":{
    ...                              "u": 1,
    ...                              "v": {
    ...                                  "x": 1,
    ...                                  "y": 2,
    ...                                  "z": 3
    ...                              },
    ...                              "w": 3
    ...                              }
    ...                          }, 
    ...                      keys_path_list=['a','r'])
    'Correct Result'
    
    --------------------------------------------------------------
    """
    return reduce(operator.getitem, keys_path_list, nested_dict)

def pop_from_nested_dict(nested_dict, keys_path_list):
    to_return = get_from_nested_dict(nested_dict, keys_path_list)
    del get_from_nested_dict(nested_dict, keys_path_list[:-1])[keys_path_list[-1]]
    return to_return

def set_in_nested_dict(nested_dict, keys_path_list, value, inplace=False):
    if not inplace:
        nested_dict = copy.deepcopy(nested_dict)
    get_from_nested_dict(nested_dict, keys_path_list[:-1])[keys_path_list[-1]]=value
    return nested_dict
    
def invert_dict(dct):
    r"""
    Swap the keys and values in dct.
    This only works if there is a 1-1 mapping of keys to values (i.e., cannot be duplicate values!)
    """
    #-------------------------
    if len(dct.values())!=len(set(dct.values())):
        print("In invert_dict, inversion impossible as there is not a 1-1 mapping of keys to values (i.e., duplicate values exist!)")
        assert(0)
    #-------------------------
    inv_dct = {v:k for k,v in dct.items()}
    return inv_dct

#--------------------------------------------------------------------
def prepend_tabs_to_each_line(input_statement, n_tabs_to_prepend=1):
    if n_tabs_to_prepend > 0:
        join_str = '\n' + n_tabs_to_prepend*'\t'
        input_statement = join_str.join(input_statement.splitlines())
        # Need to also prepend first line...
        input_statement = n_tabs_to_prepend*'\t' + input_statement
    return input_statement
#--------------------------------------------------------------------
def generate_random_string(str_len=10, letters=string.ascii_letters + string.digits):
    # Choices for letters include e.g., 
    #   string.ascii_lowercase, string.ascii_uppercase, 
    #   string.ascii_letters, string.digits, string.punctuation, etc.
    if letters=='letters_only':
        letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(str_len))
#--------------------------------------------------------------------
class PhantomType(Enum):
    kA=0 
    kB=1 
    kFAT=2 
    kSAT=3 
    kORT=4 
    kUnsetPhantomType=5

cPhantomTypeNames = {PhantomType.kA:"A", 
                     PhantomType.kB:"B", 
                     PhantomType.kFAT:"FAT", 
                     PhantomType.kSAT:"SAT", 
                     PhantomType.kORT:"ORT", 
                     PhantomType.kUnsetPhantomType:"UnsetPhantomType"}
                     
def get_phantom_type_str(phantom_type):
    return cPhantomTypeNames[phantom_type]
                     
#--------------------------------------------------------------------                     
class TTestResultColorType(Enum):
    kGreen=0 
    kYellow=1 
    kRed=2 
    kUnsetTTestResultColorType=3

cTTestResultColorTypeNames = {TTestResultColorType.kGreen:"green", 
                              TTestResultColorType.kYellow:"yellow", 
                              TTestResultColorType.kRed:"red", 
                              TTestResultColorType.kUnsetTTestResultColorType:"UnsetTTestResultColorType"}
def get_color_type_str(t_test_result_color_type):
    return cTTestResultColorTypeNames[t_test_result_color_type]
    
def str_to_color_type(color_str, accept_unset=False):
    if CompStrings_CaseInsensitive(color_str, "green"):
        return TTestResultColorType.kGreen;
    elif CompStrings_CaseInsensitive(color_str, "yellow"):
        return TTestResultColorType.kYellow;
    elif CompStrings_CaseInsensitive(color_str, "red"):
        return TTestResultColorType.kRed;
    else:
        if(accept_unset):
            return TTestResultColorType.kUnsetTTestResultColorType
        else:
            assert(0)
            
#--------------------------------------------------------------------
class EDSType(Enum):
    k6000 = 0
    k6040CTiX = 1
    k6700 = 2
    k6700ES = 3
    k9800_DCMS = 4
    k9800_EDM = 5
    k9800_SCMS = 6
    kCT_80 = 7
    kUnsetEDSType = 8
    
cEDSTypeNames = {EDSType.k6000        : '6000',
                 EDSType.k6040CTiX    : '6040CTiX',
                 EDSType.k6700        : '6700', 
                 EDSType.k6700ES      : '6700ES', 
                 EDSType.k9800_DCMS   : '9800_DCMS', 
                 EDSType.k9800_EDM    : '9800_EDM', 
                 EDSType.k9800_SCMS   : '9800_SCMS', 
                 EDSType.kCT_80       : 'CT-80', 
                 EDSType.kUnsetEDSType: 'UnsetEDSType'}

def get_eds_type_str(eds_type):
    return cEDSTypeNames[eds_type]

#--------------------------------------------------------------------
def convert_passfail_string_to_bool(pass_fail):
    if CompStrings_CaseInsensitive(pass_fail, 'Pass'):
        return True
    elif CompStrings_CaseInsensitive(pass_fail, 'Fail'):
        return False
    else:
        assert(0)
#--------------------------------------------------------------------
def CompStrings_CaseInsensitive(aStr1, aStr2):
    return aStr1.strip().casefold()==aStr2.strip().casefold()

def StringToPhantomType(aPhantomString, aAcceptUnset=True):
    if aPhantomString is None:
        if aAcceptUnset:
            return PhantomType.kUnsetPhantomType
        else:
            assert(0)
    if CompStrings_CaseInsensitive(aPhantomString, "FAT"):
        return PhantomType.kFAT;
    elif CompStrings_CaseInsensitive(aPhantomString, "SAT"):
        return PhantomType.kSAT;
    elif CompStrings_CaseInsensitive(aPhantomString, "ORT"):
        return PhantomType.kORT;
    elif CompStrings_CaseInsensitive(aPhantomString, "A"):
        return PhantomType.kA;
    elif CompStrings_CaseInsensitive(aPhantomString, "B"):
        return PhantomType.kB;
    else:
        if(aAcceptUnset):
            return PhantomType.kUnsetPhantomType
        print(f"In StringToPhantomType\nCannot find appropriate PhantomType for input aPhantomString = {aPhantomString}\n");
        print("Kill program or return kUnsetPhantomType?\n\t0=Kill\n\t1=return kUnsetPhantomType\n");
        tResponse = None
        while tResponse is None:
            try:
                tResponse = int(input())
            except ValueError:
                print('Not a number')
                continue
                
            if(tResponse!=0 and tResponse!=1):
                print('Please enter 0 or 1')
                tResponse=None
        if(tResponse):
            return PhantomType.kUnsetPhantomType
        else:
            assert(0);
    return PhantomType.kUnsetPhantomType

def StringFind_CaseInsensitive(aStrA, aStrB):
    tStrA = aStrA.strip().casefold()
    tStrB = aStrB.strip().casefold()
    if(len(tStrA) > len(tStrB)):
        return tStrB in tStrA
    else:
        return tStrA in tStrB
        
def are_strings_equal(str_1, str_2, none_eq_empty_str=True, 
                      case_insensitive=False):
    r"""
    Compares two strings, expanding the definition of equal if desired (by setting none_eq_empty_str=True)
      to include equality when one string is None and the other is empty
    none_eq_empty_str==False:
      - Simply return str_1==str_2
      
    none_eq_empty_str==True:
      - If both are None or empty, return True
      - Otherwise, simply return str_1==str_2
      
    --------------------------------------------------------------
    >>> are_strings_equal('ant', 'ant')
    True
    
    >>> are_strings_equal('ant', 'ANT')
    False
    
    >>> are_strings_equal('ant', 'ANT', case_insensitive=True)
    True
    
    >>> are_strings_equal('', None)
    True
    
    >>> are_strings_equal('', None, none_eq_empty_str=False)
    False
    """
    if (none_eq_empty_str and 
        not str_1 and 
        not str_2):
        return True
    else:
        if case_insensitive:
            return str_1.lower()==str_2.lower()
        else:
            return str_1==str_2
            
            
def find_and_remove_from_string(strng, pattern, count=0, flags=0):
    return re.sub(pattern, '', strng, count=count, flags=flags)
    
def find_and_replace_in_string(strng, pattern, replace, count=0, flags=0):
    r"""
    DON'T FORGET ABOUT BACKREFERENCING!!!!!
      e.g.
        strng = 'I am a string named Jesse Thomas Buxton'
        pattern = r'(Jesse) (Thomas) (Buxton)'
        replace = r'\1 \3'
        re.sub(pattern, replace, strng)
        ===> 'I am a string named Jesse Buxton'
    """
    # DON'T FORGET ABOUT BACKREFERENCING!!!!!
    try:
        re.sub(pattern, replace, strng, count=count, flags=flags)
    except:
        print(strng)
    return re.sub(pattern, replace, strng, count=count, flags=flags)
    
def are_approx_equal(val_1, val_2, precision=0.00001):
    if(val_1==0. and val_2==0.):
        return True
    elif(val_1==0. or val_2==0.):
        rel = abs(val_1-val_2)
    else:
        rel = abs(val_1-val_2)/abs(val_1)
    
    are_same = True if (rel <= precision) else False
    return are_same
    
def AreApproxEqual(aVal1, aVal2, aPrecision=0.00001):
    # OUTDATED NAMING!!!!!!
    # Replaced by are_approx_equal
    return are_approx_equal(aVal1, aVal2, aPrecision)
    
def are_approx_equal_nans_ok(val_1, val_2, precision=0.00001):
    #If both are NaNs, they are considered equal
    isnan_1 = np.isnan(val_1)
    isnan_2 = np.isnan(val_2)
    if isnan_1 and isnan_2:
        return True
    elif isnan_1 or isnan_2:
        # If one is NaN, they both should be NaN, otherwise they are unequal
        # Originally had: elif (isnan_1 or isnan_2) and (isnan_1 != isnan_2)
        # but (isnan_1 != isnan_2) is trivially true because of first if statement
        # together with elif (isnan_1 or isnan_2)
        return False
    else:
        # At this point, we know both val_1 and val_2 are not NaNs
        if are_approx_equal(val_1, val_2, precision):
            return True
        else:
            return False
    
def are_equal_nans_ok(a,b):
    r"""
    Checks if a and b are equal, taking NaN==NaN.
    a,b can both be single objects (e.g., strings, ints, whatever), or they can be lists of such.
    The numpy function, np.array_equal with the parameter equal_nan=False seems like a good choice in most instances,
      but it wasn't functioning as expected when used in Utilities_df.consolidate_df, hence the construction of this function.
    """
    #-------------------------
    if pd.isna(a) and pd.isna(b):
        return True
    elif is_object_one_of_types(a, [list, tuple, np.ndarray]):
        assert(is_object_one_of_types(b, [list, tuple, np.ndarray]))
        assert(len(a)==len(b))
        elements_equal = [are_equal_nans_ok(a[i], b[i]) for i in range(len(a))]
        return all(elements_equal)
    else:
        return a==b    
    
#--------------------------------------------------------------------
# Following methods essentially taken from bisect documentation (https://docs.python.org/3/library/bisect.html)
def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError
#-------------------------
def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_lt_idx(a, x):
    'Find rightmost index less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return i-1
    raise ValueError
#-------------------------
def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError
    
def find_le_idx(a, x):
    'Find rightmost index less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return i-1
    raise ValueError

#-------------------------
def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError
    
def find_gt_idx(a, x):
    'Find leftmost index greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return i
    raise ValueError

#-------------------------
def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError
    
def find_ge_idx(a, x):
    'Find leftmost index greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return i
    raise ValueError


#--------------------------------------------------------------------
def find_longest_shared_substring(string1, string2):
    match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
    assert(string1[match.a: match.a + match.size] == string2[match.b: match.b + match.size])
    return string1[match.a: match.a + match.size]

def get_length_of_longest_shared_substring(string1, string2):
    return len(find_longest_shared_substring(string1, string2))
    
#--------------------------------------------------------------------
def find_idxs_in_list_with_regex(lst, regex_pattern, ignore_case=False):
    r"""
    Search all elements of lst for regex_pattern, return the idxs of those found.
    regex_pattern:
      Can be a single regex pattern, or a list of patterns
    """
    #-------------------------
    assert(isinstance(lst, list)) #Should I allow it to also be an np.ndarray?
    assert(is_object_one_of_types(regex_pattern, [str, list]))
    #--------------------------------------------------
    #--------------------------------------------------
    if isinstance(regex_pattern, list):
        found_idxs = []
        for re_patt in regex_pattern:
            found_idxs.extend(
                find_idxs_in_list_with_regex(
                    lst=lst, 
                    regex_pattern=re_patt, 
                    ignore_case=ignore_case
                )
            )
        found_idxs = sorted(list(set(found_idxs)))
        return found_idxs
    #--------------------------------------------------
    #--------------------------------------------------
    if ignore_case:
        flags = re.IGNORECASE
    else:
        flags = 0
    #-------------------------
    found_idxs = [i for i,x in enumerate(lst) if re.search(regex_pattern, x, flags=flags)]
    #-------------------------
    return found_idxs

def find_in_list_with_regex(
        lst           , 
        regex_pattern , 
        ignore_case   = False
    ):
    r"""
    Search all elements of lst for regex_pattern, return found.
    regex_pattern:
      Can be a single regex pattern, or a list of patterns
    """
    #-------------------------
    found_idxs = find_idxs_in_list_with_regex(lst=lst, regex_pattern=regex_pattern, ignore_case=ignore_case)
    found = np.array(lst)[found_idxs]
    found = found.tolist()
    #-------------------------
    return found

#--------------------------------------------------------------------
def remove_multiple_items_from_list_by_idx(list_1, idxs_1):
    # Need to first sort the indices in descending order
    # If instead the indices are in ascending order, e.g.
    #    del somelist[0] followed by del somelist[2] will actually
    #    delete somelist[3]
    sorted_idxs_1 = sorted(idxs_1, reverse=True)
    assert(len(sorted_idxs_1)==len(np.unique(sorted_idxs_1)))
    for idx in sorted_idxs_1:
        del list_1[idx]
    return list_1
    
def remove_tagged_from_list(lst, tags_to_ignore):
    # Only return elements where none of the tags_to_ignore are found
    # Case insensitive!
    # 20210914: 
    #  Added functionality where if a tag within tags_to_ignore ends with '_EXACTMATCH'
    #  then only elements exactly matching that tag will be removed (instead of the normal case where
    #  an element is removed if it contains the tag.
    #
    #  This new functionality was added when working with AT Tray Phantom 11, where I did not want to keep
    #  any of the 'Freq_x' columns or 'MTF_1' (as MTF_1=1 trivially for all).  
    #  For this example, 
    #  lst = ['Freq_1', 'MTF_1', 'Freq_2', 'MTF_2', 'Freq_3', 'MTF_3', 'Freq_4', 'MTF_4', 'Freq_5', 'MTF_5', 'Freq_6', 'MTF_6', 
    #         'Freq_7', 'MTF_7', 'Freq_8', 'MTF_8', 'Freq_9', 'MTF_9', 'Freq_10', 'MTF_10', 'Freq_11', 'MTF_11']
    #  Using the old method with tags_to_ignore = ['MTF_1', 'Freq_'], all 'Freq_x' and 'MTF_1' would be dropped, but
    #  'MTF_10' and 'MTF_11' would be dropped as well!
    #  Thus, for the desired effect with the new functionality, the input should be:
    #  tags_to_ignore = ['MTF_1_EXACTMATCH', 'Freq_']
    assert(isinstance(lst, list) or isinstance(lst, tuple))
    assert(isinstance(tags_to_ignore, list) or isinstance(tags_to_ignore, tuple))
    exact_matches = [x for x in tags_to_ignore if x.endswith('_EXACTMATCH')]
    reg_tags_to_ignore = [x for x in tags_to_ignore if not x.endswith('_EXACTMATCH')]
    assert(len(tags_to_ignore)==len(exact_matches)+len(reg_tags_to_ignore))

    # First, remove any exact matches specified by '_EXACTMATCH'
    if len(exact_matches)>0:
        exact_matches = [x[:x.find('_EXACTMATCH')] for x in exact_matches] #strip off '_EXACTMATCH'
        sub_list = [x for x in lst if all(x!=tag for tag in exact_matches)]
    else:
        sub_list = lst
        
    # Now, run remove_tagged_from_list as originall written
    sub_list = [x for x in sub_list if all(not StringFind_CaseInsensitive(x, tag) for tag in reg_tags_to_ignore)]
    return sub_list
    
def remove_tagged_from_list_of_paths(list_of_paths, tags_to_ignore, base_dir=None):
    # This is similar to remove_tagged_from_list
    # Case insensitive!
    # However, if base_dir is not None, this will only search for the tags in in the text of
    #   the path after base_dir
    # E.g. when looking for all NIST XMLs on BACO, I typically ignore anything with "Image"
    #   This is problematic when searching in e.g. r'\\milky-way\projects\BACOArchive\IQS-IQP_Image-Archive\Validated Baseline\IQ-ORT\RIT\CT80'
    #   because every single possible path within this directory contains "Image"! (...\IQS-IQP_Image-Archive\...)
    #   Thus, using remove_tagged_from_list would return an empty list!
    #
    # If base_dir is None, this will function exactly as remove_tagged_from_list
    # If base_dir is not None, all paths in list_of_paths should contain base_dir (this will be checked and asserted)
    if base_dir is None:
        return remove_tagged_from_list(list_of_paths, tags_to_ignore)

    return_paths = []
    for path in list_of_paths:
        assert(find_longest_shared_substring(path, base_dir)==base_dir) # the assertion promised in the description
        path_to_search = path[len(base_dir+os.sep):] #ignore base_dir portion of path
        # if none of the tags in tags_to_ignore are found in path_to_search, add path to return paths
        #  (could also implement as: if any tags found then do not add)
        if all([not StringFind_CaseInsensitive(path_to_search, tag) for tag in tags_to_ignore]): #the [] are not necessary here, but are good for clarity I suppose
            return_paths.append(path)

    return return_paths  


def find_tagged_idxs_in_list(lst, tags):
    r"""
    Similar to remove_tagged_from_list, but instead of removing the elements containing one of the tags,
    this will return the idxs of those elements within the list.
    -----
    Case insensitive!
    -----
    See remove_tagged_from_list for description of full functionality (e.g., possibility of using _EXACTMATCH)
    
    The method here is somewhat lazy, in that is used the reduced list returned by remove_tagged_from_list
    to determine which elements contain one of the tags.
    
    NOTE: In order to use the list of indices returned from this function, one must utilizie a np.array.
          e.g., assume tagged_idxs = [2,4,5]
                Calling lst[tagged_idxs] will give the error TypeError: list indices must be integers or slices, not list
                Therefore, one must use np.array and convert back to a regular list at end.
                  ==> list(np.array(lst)[tagged_idxs])
    """
    #-------------------------
    sub_list = remove_tagged_from_list(lst=lst, tags_to_ignore=tags)
    tagged_idxs = [i for i,x in enumerate(lst) if x not in sub_list]
    return tagged_idxs


def find_untagged_idxs_in_list(lst, tags):
    r"""
    Similar to remove_tagged_from_list, but instead of returning of list of elements which did not contain
    any of the tags, this will return the idxs of those elements within the list.
    -----
    Case insensitive!
    -----
    See remove_tagged_from_list for description of full functionality (e.g., possibility of using _EXACTMATCH)
    
    The method here is somewhat lazy, in that is used the reduced list returned by remove_tagged_from_list
    to determine which elements do not contain any of the tags.
    
    NOTE: In order to use the list of indices returned from this function, one must utilizie a np.array.
          e.g., assume untagged_idxs = [2,4,5]
                Calling lst[untagged_idxs] will give the error TypeError: list indices must be integers or slices, not list
                Therefore, one must use np.array and convert back to a regular list at end.
                  ==> list(np.array(lst)[untagged_idxs])
    """
    #-------------------------
    sub_list = remove_tagged_from_list(lst=lst, tags_to_ignore=tags)
    untagged_idxs = [i for i,x in enumerate(lst) if x in sub_list]
    return untagged_idxs
    
    
def get_two_lists_diff(li1, li2):
    li_diff = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_diff


def include_at_front_and_exclude_from_list(a_list, include_at_front=None, exclude_from_list=None, 
                                           inplace=True):
    r"""
    include_at_front and exclude_from_list are intended to be lists of string.
      If just a single item for either list, a string can be passed instead.
    Only elements of a_list found in include_at_front+exclude_from_list will be affected.    
    """
    #-------------------------
    if not inplace:
        a_list = copy.deepcopy(a_list)
    #-------------------------
    if not include_at_front:
        include_at_front = []
    if not exclude_from_list:
        exclude_from_list = []
    #-------------------------
    if isinstance(include_at_front, str):
        include_at_front = [include_at_front]
    if isinstance(exclude_from_list, str):
        exclude_from_list = [exclude_from_list]
    #-------------------------
    assert(isinstance(include_at_front, list) or isinstance(include_at_front, tuple))
    assert(isinstance(exclude_from_list, list) or isinstance(exclude_from_list, tuple))
    #-------------------------
    # First, remove all from include_at_front and exclude_from_list
    #   Those in include_at_front will be put back after
    idxs_to_remove = []
    for element in include_at_front+exclude_from_list:
        if element in a_list:
            idxs_to_remove.append(a_list.index(element))
    # Order matters when removing elements!!!!!
    idxs_to_remove = sorted(idxs_to_remove, reverse=True)
    for idx in idxs_to_remove:
        del a_list[idx]
    #-------------------------
    # Prepend include_at_front and return
    # Since I want to be able to use inplace, I have to
    #   do the iteration below.
    #   I cannot simply call: a_list = include_at_front + a_list
    #     as this actually creates a new list, thus negating the inplace
    #     and causing some strange behavior (input a_list would not equal 
    #     output a_list when inplace=True)
    for i,to_include in enumerate(include_at_front):
        a_list.insert(i, to_include)
    return a_list
    
    
def get_batch_idx_pairs(
    n_total, 
    batch_size, 
    absorb_last_pair_pct=None
):
    r"""
    Find the beginning and ending indices for groups when splitting collection into batches.
    Returns a list containing 2-element lists, where the two elements are the beginning and
      ending index for the batch.
    e.g., if one wants to split up a list of 84 elements into batches of size 20:
        get_batch_idx_pairs(84, 20, None) = [[0, 20], [20, 40], [40, 60], [60, 80], [80, 84]]
    
    absorb_last_pair_pct:
        If the last batch is short, this directs the batch to be absorbed into the second-to-last batch.
        The value, between 0 and 1, dictates the definition of a 'short' batch.
        Default value is None, in which case the behavior is as described above.
        e.g., if one wants to split up a list of 84 elements into batches of size 20:
            get_batch_idx_pairs(84, 20, 0.25) = [[0, 20], [20, 40], [40, 60], [60, 84]]
            get_batch_idx_pairs(84, 20, 0.20) = [[0, 20], [20, 40], [40, 60], [60, 80], [80, 84]]
        
    """
    batch_idxs = []
    n_batches = int(np.ceil(n_total/batch_size))
    for i in range(n_batches):
        i_beg = i*batch_size
        i_end = (i+1)*batch_size
        if i==n_batches-1:
            i_end = n_total
        batch_idxs.append([i_beg, i_end])
    if absorb_last_pair_pct is None:
        return batch_idxs
    else:
        if len(batch_idxs)==1:
            return batch_idxs
        assert(absorb_last_pair_pct>0 and absorb_last_pair_pct<1)
        last_group_len = batch_idxs[-1][1]-batch_idxs[-1][0]
        if last_group_len/batch_size < absorb_last_pair_pct:
            batch_idxs.pop()
            batch_idxs[-1][-1]=n_total
        return batch_idxs


def get_smallest_file_size_MB(
    paths, 
    return_min_file=False
):
    r"""
    """
    #-------------------------
    paths_w_sizes = [(path_i, 1e-6*os.path.getsize(path_i)) for path_i in paths]
    min_idx = np.argmin([x[1] for x in paths_w_sizes])
    min_size = paths_w_sizes[min_idx][1]
    min_file = paths_w_sizes[min_idx][0]
    if return_min_file:
        return min_size, min_file
    return min_size
    
def get_largest_file_size_MB(
    paths, 
    return_max_file=False
):
    r"""
    """
    #-------------------------
    paths_w_sizes = [(path_i, 1e-6*os.path.getsize(path_i)) for path_i in paths]
    max_idx = np.argmax([x[1] for x in paths_w_sizes])
    max_size = paths_w_sizes[max_idx][1]
    max_file = paths_w_sizes[max_idx][0]
    if return_max_file:
        return max_size, max_file
    return max_size
        
def get_files_split_locations(
    paths, 
    batch_size_MB, 
    tolerance_pct=0.01, 
    absorb_last_pair_pct=None
):
    r"""
    Given the location of files in paths, split the files into batches of (approximate) size batch_size_MB
      in megabytes.
    Returns a list of two-element tuples.
        Each tuple contains the beginning and ending index location (integer) for a given batch.
        Each tuple is inclusive on the front and exclusive at the end, i.e., [a,b).
        Therefore, each tuple can be fed directly into df.iloc[a:b] to obtain the desired batch
    -----
    The simplest method is to find where the cumulative sum equals increments of batch_size_MB (i.e., 
      where cumsum==i*batch_size_MB)
    However, with this approach, each block is NOT guaranteed to be smaller than batch_size_MB
    In most all cases, if a block is larger than batch_size_MB, it will only be slightly larger.
    There do exist conditions for which a block could be substantially larger than batch_size_MB.
    -----
    If all found blocks have sizes no greater than tolerance_pct, the simple method is used.
    Otherwise, the precise method is used
    
    batch_size_MB:
        The maximum batch size in megabytes
        NOTE: This must be at least as large as the largest file in paths!
        
    tolerance_pct:
        Tolerance percent for blocks to be larger than batch_size_MB.
        SHOULD BE BETWEEN 0 AND 1
        If any block exceeds this value, the precise method is used, otherwise the simple method is used (described above)
        
    absorb_last_pair_pct:
        If the last batch is small, this directs the batch to be absorbed into the second-to-last batch.
        The value (SHOULD BE BETWEEN 0 AND 1) dictates the definition of a 'small' batch.
        Default value is None, in which case the behavior is as described above.
    """
    #-------------------------
    assert(tolerance_pct>=0 and tolerance_pct<1)
    #-------------------------
    paths_w_sizes = [(path_i, 1e-6*os.path.getsize(path_i)) for path_i in paths]
    paths_w_sizes = natsorted(paths_w_sizes, key=lambda x: x[0])
    #-----
    # batch_size_MB must be at least as large as the largest file in paths
    if batch_size_MB < np.max([x[1] for x in paths_w_sizes]):
        print('batch_size_MB must be at least as large as the largest file in paths!\nCRASH IMMINENT!')
        print(f'batch_size_MB = {batch_size_MB}\nlargest fsize = {np.max([x[1] for x in paths_w_sizes])}')
        assert(0)
    #-------------------------
    paths_df = pd.DataFrame(paths_w_sizes, columns=['path', 'fsize'])
    paths_df['fname']  = paths_df['path'].apply(lambda x: os.path.basename(x))
    paths_df['cumsum'] = paths_df['fsize'].cumsum()
    paths_df = paths_df[['fname', 'path', 'fsize', 'cumsum']]
    #--------------------------------------------------
    # Simple method
    n_batches = np.ceil(paths_df.iloc[-1]['cumsum']/batch_size_MB).astype(int)
    #-------------------------
    block_end_idxs = [paths_df[paths_df['cumsum']<=i*batch_size_MB].index[-1] for i in range(1, n_batches+1)]
    assert(block_end_idxs[-1]==paths_df.shape[0]-1)
    block_beg_idxs = [0] + [x+1 for x in block_end_idxs][:-1]
    assert(len(block_beg_idxs)==len(block_end_idxs))
    block_begend_idxs = list(zip(block_beg_idxs, block_end_idxs))
    #--------------------------------------------------
    # Precise method
    #-----
    # If any of the found blocks are larger than batch_size_MB by more than tolerance_pct (i.e., 
    #   if any have sizes larger than (1+tolerance_pct)*batch_size_MB), the precise method must be used
    # Otherwise, the simple results may be returned
    if any(
        [paths_df[block_beg_i:block_end_i+1]['fsize'].sum()>(1+tolerance_pct)*batch_size_MB 
         for block_beg_i, block_end_i in block_begend_idxs]
    ):
        # Precise method
        tmp_fsize_col = generate_random_string()
        paths_df[tmp_fsize_col] = paths_df['fsize']
        tmp_fsize_col_idx = paths_df.columns.tolist().index(tmp_fsize_col)
        #-------------------------
        block_end_i = -1
        block_begend_idxs = []
        while block_end_i < paths_df.shape[0]-1:
            block_beg_i = block_end_i+1
            block_end_i = paths_df[paths_df['cumsum']<=batch_size_MB].index[-1]
            block_begend_idxs.append((block_beg_i, block_end_i))
            # Zero out the tmp_fsize for these entries so cumsum can be calculated for next block
            paths_df.iloc[block_beg_i:block_end_i+1, tmp_fsize_col_idx]=0
            paths_df['cumsum'] = paths_df[tmp_fsize_col].cumsum()
        assert(all([paths_df[block_beg_i:block_end_i+1]['fsize'].sum()<=(1+tolerance_pct)*batch_size_MB 
                    for block_beg_i, block_end_i in block_begend_idxs]))
        paths_df = paths_df.drop(columns=[tmp_fsize_col])
    #--------------------------------------------------
    # To be consistent with Utilities.get_batch_idx_pairs, the returned pairs should be
    #   inclusive on the front and exclusive at the end, i.e., [a,b)
    # As it stands right now, the entries in block_begend_idxs are inclusive on both ends.
    # ==> Increase end point of each interval by 1
    block_begend_idxs = [[x[0], x[1]+1] for x in block_begend_idxs]
    #-------------------------
    if absorb_last_pair_pct is None:
        return block_begend_idxs
    #-------------------------
    if len(block_begend_idxs)==1:
        return block_begend_idxs
    assert(absorb_last_pair_pct>0 and absorb_last_pair_pct<1)
    last_group_size = paths_df[block_begend_idxs[-1][0]:block_begend_idxs[-1][1]]['fsize'].sum()
    if last_group_size/batch_size_MB < absorb_last_pair_pct:
        end_idx = block_begend_idxs[-1][1]
        block_begend_idxs.pop()
        block_begend_idxs[-1][-1] = end_idx
    #-----
    return block_begend_idxs
    
    
#--------------------------------------------------------------------
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
    ovrlp = get_overlap_interval(
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

#--------------------------------------------------------------------
def get_overlap_intervals(ranges):
    r"""
    Returns a consolidated list of unique ranges from an input of overlapping ranges.
    For approximately overlapping ranges, see get_fuzzy_overlap_intervals
    This expected ranges to be a list of tuples, where each tuple is an individual range
    Should work if elements are ints, float, timestamps, etc.
    e.g.
        if ranges = [(1,4),(1,9),(3,7),(100,200)]
         overlaps = [(1, 9), (100, 200)]
    """
    #-------------------------
    if len(ranges)==0:
        return ranges
    #-------------------------
    # First, make sure the second element in each tuple should be greater than the first 
    # This also creates a copy so the original list is not altered
    # Also, sort ranges, as will be necessary for this procedure
    ranges = sorted([(min(x), max(x)) for x in ranges])
    
    # Set the first range in overlaps simply as the first range in the list
    overlaps = []
    current_beg, current_end = ranges.pop(0)
    overlaps.append((current_beg, current_end))
    
    for beg,end in ranges:
        if beg > current_end:
            # beg after current end, so new interval needed
            overlaps.append((beg, end))
            current_beg, current_end = beg, end
        else:
            # beg <= current_end, so overlap
            # The beg of overlaps[-1] remains the same, 
            #   but the end of overlaps[-1] should be changed to
            #   the max of current_end and end
            current_end = max(current_end, end)
            overlaps[-1] = (current_beg, current_end)
            
    return overlaps


def get_fuzzy_overlap_intervals(ranges, fuzziness): 
    r"""
    Returns a consolidated list of unique ranges from an input of approximately overlapping ranges.
    The fuzziness parameters sets how close two ranges must be to be considered overlapping.
    This function is essentially get_overlap_intervals with this additional fuzziness parameter.
        I could have simply added a default fuzziness parameter equal to None to the get_overlap_intervals
          function.  However, I chose not to, and chose to make this a unique function, because the form/type of
          fuzziness depends on the types of data in ranges.
        e.g., if ranges are ints
            ==> to recover the functionality of get_overlap_intervals, fuzziness should be 0
        e.g., if ranges are datetimes
            ==> to recover the functionality of get_overlap_intervals, fuzziness should be pd.Timedelta(0)
        To avoid the headache in get_overlap_intervals of converting fuzziness=None to the apropriate value,
          I decided to make this function separate.
    
    ranges:
        This expected ranges to be a list of tuples, where each tuple is an individual range
        Should work if elements are ints, float, timestamps, etc.
        e.g., consider the case where: ranges = [(1,4),(6,9),(3,5),(100,200)]
            - With fuzziness=0, one recovers the results of get_overlap_intervals, namely
                overlaps = [(1,5), (6,9), (100,200)]
            - With fuzziness=1, one recovers:
                overlaps = [(1,9), (100,200)]
             
    fuzziness:
        MUST BE COMPATIBLE WITH + operation with data in ranges.
        ==> i.e., e.g., ranges[0][0] + fuzziness must run successfully!
    """
    #-------------------------
    # First, make sure the second element in each tuple should be greater than the first 
    # This also creates a copy so the original list is not altered
    # Also, sort ranges, as will be necessary for this procedure
    ranges = sorted([(min(x), max(x)) for x in ranges])
    
    # Set the first range in overlaps simply as the first range in the list
    overlaps = []
    current_beg, current_end = ranges.pop(0)
    overlaps.append((current_beg, current_end))
    
    for beg,end in ranges:
        if beg > current_end+fuzziness:
            # beg after current end (with fuzziness buffer), so new interval needed
            overlaps.append((beg, end))
            current_beg, current_end = beg, end
        else:
            # beg <= current_end+fuzziness, so overlap
            # The beg of overlaps[-1] remains the same, 
            #   but the end of overlaps[-1] should be changed to
            #   the max of current_end and end
            current_end = max(current_end, end)
            overlaps[-1] = (current_beg, current_end)
            
    return overlaps
    
    
def get_fuzzy_overlap(lst, fuzziness):
    r"""
    Similar to get_fuzzy_overlap_intervals, but instead of overlapping ranges (pairs of numbers) this finds overlapping
      intervals for a list of single numbers.
      
    EX: lst = [1, 3, 4, 5, 11, 12, 15]
        fuzziness = 1
            ==> intervals = [[1, 1], [3, 5], [11, 12], [15, 15]]
            
    EX: lst = [1, 3, 4, 5, 11, 12, 15]
        fuzziness = 2
            ==> intervals = [[1, 5], [11, 12], [15, 15]]
            
    EX: lst = [1, 3, 4, 5, 11, 12, 15]
        fuzziness = 3
            ==> intervals = [[1, 5], [11, 15]]
    """
    #-------------------------
    lst = sorted(lst)
    #-----
    intervals = []
    current_beg = lst.pop(0)
    current_end = current_beg
    intervals.append([current_beg, current_end])
    #-------------------------
    for el in lst:
        if el > current_end+fuzziness:
            # el after current end (with fuzziness buffer), so new interval needed
            intervals[-1][1] = current_end
            intervals.append([el, el])
            current_end = el
        else:
            # el <= current_end+fuzziness, so overlap.
            # The beg of intervals[-1] remains the same, but the end of intervals[-1] 
            #   should be changed to the max of current_end and el
            current_end = max(current_end, el)
            intervals[-1] = [intervals[-1][0], current_end]
    #-------------------------
    return intervals
    
#--------------------------------------------------------------------
def remove_overlap_from_interval(
    intrvl_1, 
    intrvl_2, 
    closed = False
):
    r"""
    Removes from intrvl_1 any overlap with intrvl_2
    If the intervals overlap, remove the overlap region from intrvl_1 and return.
    If the intervals do not overlap, return intrvl_1.
    NOTE: If intrvl_1 completely engulfs intrvl_2, this procedure will result in intrvl_1 being split into two intervals
          As such, to be uniform, this function will always return a list of pd.Interval objects (or, an empty list)
    NOTE: Although the default value of closed is False, in many cases the users will want to set this equal to True or
            to a custom quantum value (see description below)
          It is set to False by default because I want the user to read this documentation, and understand the implications of
            setting closed=True before doing so.
          
    intrvl_1/_2:
        The interval, which must be either a list of length 2 or a pd.Interval object.
          
    closed:
        Indicates whether or not the endpoints are closed.
            open (a,b): Open interval uses parentheses. This means that the range contains all real numbers x that is 
                        precisely between the numbers a and b, i.e., the set of a < x < b.
            closed [a,b]: For closed ranges, square brackets indicate that the endpoints lie within the range. 
                          Therefore, closed intervals can be annotated as a set of a ≤ x ≤ b.
        NOTE: For current, simple, functionality, both endpoints of both intervals must have uniform type, i.e., all must
              be open or all must be closed.
        Acceptable values:
            False/None:
                The intervals are open at both endpoints.
                This essentially means the data are continuous.         
            True: 
                Sets the quantum value (see description below) to 1
            Quantum value:
                Where quatum defined as in physics as the minimum amount of a physical entity.
                In this case, this represents the discrete width of a single entry in the intervals.
                E.g., if intervals contain integers, the quantum is 1.
                E.g., if intervals contain days, the quantum is 1 day
        Behavior by simple examples:
            In the following, assume intrvl_1=[0, 10] and intrvl_2=[4, 6]
            closed = False/None:
                return_intrvls = [[0,4], [6,10]]
            closed = True:
                return_intrvls = [[0,3], [7,10]]
            closed = 2
                return_intrvls = [[0,2], [8,10]]
    """
    #---------------------------------------------------------------------------
    assert(is_object_one_of_types(intrvl_1, [list, pd.Interval]))
    assert(is_object_one_of_types(intrvl_2, [list, pd.Interval]))
    #-----
    if isinstance(intrvl_1, pd.Interval):
        intrvl_1 = [intrvl_1.left, intrvl_1.right]
    if isinstance(intrvl_2, pd.Interval):
        intrvl_2 = [intrvl_2.left, intrvl_2.right]
    #-----
    assert(len(intrvl_1)==len(intrvl_2)==2)
    #-------------------------
    if closed==True:
        closed = 1
    #-------------------------
    # Get value of closed to feed into pd.Interval
    if not closed:
        pd_closed = 'neither'
    else:
        pd_closed = 'both'
    #-------------------------
    if not pd.Interval(*intrvl_1, closed=pd_closed).overlaps(pd.Interval(*intrvl_2, closed=pd_closed)):
        return [pd.Interval(*intrvl_1, closed=pd_closed)]
    #-------------------------
    if intrvl_1 == intrvl_2:
        return []
    #-------------------------
    # Overlap region will range from max of mins to min of maxs
    ovrlp = [np.max([intrvl_1[0], intrvl_2[0]]), np.min([intrvl_1[1], intrvl_2[1]])]
    #---------------------------------------------------------------------------
    # Four situations can result
    # 1. intrvl_2 completely engulfs intrvl_1, resulting in ovrlp being equal to intrvl_1
    #      ==> Return empty
    # 2. intrvl_1 completely engulfs intrvl_2, resulting in ovrlp being equal to intrvl_2
    #      ==> Return two intervals, [intrvl_1[0], ovrlp[0]] and [ovrlp[1], intrvl_1[1]]
    # 3. Ending of intrvl_1 overlaps with beginning of intrvl_2, resulting in ovrlp[0]==intrvl_2[0] and ovrlp[1]==intrvl_1[1]
    #      ==> Return [intrvl_1[0], ovrlp[0]]
    # 4. Beginning of intrvl_1 overlaps with ending of intrvl_2, resulting in ovrlp[0]==intrvl_1[0] and ovrlp[1]==intrvl_2[1]
    #      ==> Return [ovrlp[1], intrvl_1[1]]
    #-------------------------
    return_intrvls = None
    #-----
    # 1. intrvl_2 completely engulfs intrvl_1 ==> Return empty
    if ovrlp==intrvl_1:
        return_intrvls = []
    #-------------------------
    # 2. intrvl_1 completely engulfs intrvl_2 ==> Return two intervals, [intrvl_1[0], ovrlp[0]] and [ovrlp[1], intrvl_1[1]]
    #    EXCEPT if intrvl_1[0]==intrvl_2[0] or intrvl_1[1]==intrvl_2[1], in which case only one interval is needed to be returned
    #      intrvl_1[0]==intrvl_2[0] ==> return [ovrlp[1], intrvl_1[1]]
    #      intrvl_1[1]==intrvl_2[1] ==> return [intrvl_1[0], ovrlp[0]]
    elif ovrlp==intrvl_2:
        if not closed:
            if intrvl_1[0]==intrvl_2[0]:
                return_intrvls = [pd.Interval(ovrlp[1],    intrvl_1[1], closed=pd_closed)]
            elif intrvl_1[1]==intrvl_2[1]:
                return_intrvls = [pd.Interval(intrvl_1[0], ovrlp[0],    closed=pd_closed)]
            else:
                return_intrvls = [
                    pd.Interval(intrvl_1[0], ovrlp[0],    closed=pd_closed), 
                    pd.Interval(ovrlp[1],    intrvl_1[1], closed=pd_closed)
                ]
        else:
            if intrvl_1[0]==intrvl_2[0]:
                return_intrvls = [pd.Interval(ovrlp[1]+closed, intrvl_1[1],     closed=pd_closed)]
            elif intrvl_1[1]==intrvl_2[1]:
                return_intrvls = [pd.Interval(intrvl_1[0],     ovrlp[0]-closed, closed=pd_closed)]
            else:
                return_intrvls = [
                    pd.Interval(intrvl_1[0],     ovrlp[0]-closed, closed=pd_closed), 
                    pd.Interval(ovrlp[1]+closed, intrvl_1[1],     closed=pd_closed)
                ]
    #-------------------------
    # 3. Ending of intrvl_1 overlaps with beginning of intrvl_2 ==> Return [intrvl_1[0], ovrlp[0]]
    elif intrvl_1[0] < intrvl_2[0]:
        assert(ovrlp[0]==intrvl_2[0])
        assert(ovrlp[1]==intrvl_1[1])
        if not closed:
            return_intrvls = [pd.Interval(intrvl_1[0], ovrlp[0], closed=pd_closed)]
        else:
            return_intrvls = [pd.Interval(intrvl_1[0], ovrlp[0]-closed, closed=pd_closed)]
    #-------------------------
    # 4. Beginning of intrvl_1 overlaps with ending of intrvl_2 ==> Return [ovrlp[1], intrvl_1[1]]
    elif intrvl_1[0] > intrvl_2[0]:
        assert(ovrlp[0]==intrvl_1[0])
        assert(ovrlp[1]==intrvl_2[1])
        if not closed:
            return_intrvls = [pd.Interval(ovrlp[1], intrvl_1[1], closed=pd_closed)]
        else:
            return_intrvls = [pd.Interval(ovrlp[1]+closed, intrvl_1[1], closed=pd_closed)]
    #-------------------------
    else:
        assert(0)
    #---------------------------------------------------------------------------
    # Sanity
    if len(return_intrvls)>0:
        for intrvl_i in return_intrvls:
            assert(intrvl_i.left <= intrvl_i.right)
    #---------------------------------------------------------------------------
    return return_intrvls
    
    
def remove_overlaps_from_interval(
    intrvl_1, 
    intrvls_2, 
    closed = False
):
    r"""
    Removes from intrvl_1 any overlaps with the intervals contained in intrvls_2.
    See remove_overlap_from_interval for more information.
    NOTE: If intrvl_1 completely engulfs any of intrvls_2, this procedure will result in intrvl_1 being split into two intervals
          As such, to be uniform, this function will always return a list of pd.Interval objects (or, an empty list)
    NOTE: Although the default value of closed is False, in many cases the users will want to set this equal to True or
            to a custom quantum value (see description below)
          It is set to False by default because I want the user to read this documentation, and understand the implications of
            setting closed=True before doing so.
    
    intrvl_1:
        The interval, which must be either a list of length 2 or a pd.Interval object.
        
    intrvls_2:
        A list of intervals, i.e., it must be a list where each element is either a list
        of length 2 or a pd.Interval object.
    """
    #---------------------------------------------------------------------------
    # Make sure intrvls_2 is a list.
    # NOTE: intrvl_1 and the elements of intrvls_2 will be checked for proper type in remove_overlap_from_interval
    assert(isinstance(intrvls_2, list))
    if len(intrvls_2)==0:
        return [intrvl_1]
    #-------------------------
    # Since remove_overlap_from_interval returns a list of pd.Interval objects (can see reason why in documentation for function)
    #   must convert intrvl_1 to list for simpler code below
    # Note: deepcopy probably not needed below, but, safety first kids!
    return_intrvls = [copy.deepcopy(intrvl_1)]
    #-------------------------
    for intrvl_2_i in intrvls_2:
        return_intrvls_i = []
        for return_intrvl_j in return_intrvls:
            return_intrvls_i.extend(
                remove_overlap_from_interval(
                    intrvl_1 = return_intrvl_j, 
                    intrvl_2 = intrvl_2_i, 
                    closed   = closed
                )
            )
        return_intrvls = return_intrvls_i
    #-------------------------
    return return_intrvls
    
def remove_overlaps_from_date_interval(
    intrvl_1, 
    intrvls_2, 
    closed = True
):
    r"""
    Only numeric, Timestamp and Timedelta endpoints are allowed when constructing an Interval.
    ==> Need for remove_overlaps_from_date_interval (instead of simply using remove_overlaps_from_interval)
    
    NOTE: If intrvl_1 completely engulfs any of intrvls_2, this procedure will result in intrvl_1 being split into two intervals
          As such, to be uniform, this function will always return a list of pd.Interval objects (or, an empty list)
    
    closed:
        Must be boolean.
        See remove_overlap_from_interval documentation for refresher on open/closed sets
    """
    #-------------------------
    assert(isinstance(closed, bool))
    if closed:
        closed = pd.Timedelta('1D')
    #-------------------------
    assert(isinstance(intrvls_2, list))
    if len(intrvls_2)==0:
        return [intrvl_1]
    #-------------------------
    assert(is_object_one_of_types(intrvl_1, [list, tuple]))
    assert(len(intrvl_1)==2)
    #-----
    for intrvl_2_i in intrvls_2:
        assert(is_object_one_of_types(intrvl_2_i, [list, tuple]))
        assert(len(intrvl_2_i)==2)
    #-------------------------
    # Make sure the base elements of intrvl_1 and intrvls_2 are datetime.date objects
    assert(
        are_all_list_elements_of_type(
            lst = melt_list_of_lists([intrvl_1]+intrvls_2), 
            typ = datetime.date
        )
    )
    
    #-------------------------
    # Only numeric, Timestamp and Timedelta endpoints are allowed when constructing an Interval.
    #   ==> All elements need converted from datetime.date to Timestamp
    #     intrvl_1  ==> [pd.to_datetime(x) for x in intrvl_1]
    #     intrvls_2 ==> [[pd.to_datetime(x[0]), pd.to_datetime(x[1])] for x in intrvls_2]
    return_intrvls = remove_overlaps_from_interval(
        intrvl_1  = [pd.to_datetime(x) for x in intrvl_1], 
        intrvls_2 = [[pd.to_datetime(x[0]), pd.to_datetime(x[1])] for x in intrvls_2], 
        closed    = closed
    )

    #-------------------------
    # Convert elements back to datetime.date objects
    return_intrvls = [[x.left.date(), x.right.date()] for x in return_intrvls]
    return return_intrvls
    
#--------------------------------------------------------------------
def get_kit_id(path, gold_standard='S002'):
    """
    parent_kit_id - the test candidate being analyzed when test candidate and
                        gold standard were run through the EDS.
                    This should never be the gold standard
    kit_id        - the kit whose results are contained in path.
                    This can be test candidate or gold standard
                    
    e.g. path = ...\\S001_07202020_F\\6040CTiX\\S001\\Results - FTG\\NISTResults\\Candidate_S001_20200726_030001.xml
                parent_kit_id = 'S001'
                kit_id        = 'S001'
    
    e.g. path = ...\\S001_07202020_F\\6040CTiX\\S002\\Results - FTG\\NISTResults\\GoldStandard_S002_20200726_031739.xml
                parent_kit_id = 'S001'
                kit_id        = 'S002'
    """

    if gold_standard[0]=='S':
        identifier = 'S0'
    elif gold_standard[0]=='F':
        identifier = 'F0'
    else:
        assert(0)

    i_found = [i.start() for i in re.finditer(identifier, path)]

    # Take out any gold_standard found
    parent_kit_id = [path[i:i+4] for i in i_found if path[i:i+4] != gold_standard]
    if len(parent_kit_id)==0:
        parent_kit_id = None
    else:
        # Make sure only one parent kit id found
        # Multiple occurrences of parent_kit_id are fine
        # Using set() counts the number of unique parent_kit_ids,
        # which must be one
        assert(len(set(parent_kit_id))==1)
        assert(parent_kit_id != gold_standard)
        parent_kit_id = parent_kit_id[0]

    # Look for any occurrences of gold_standard
    # If any are found, then kit_id = gold_standard, 
    # otherwise kit_id = parent_kit_id
    kit_id = [path[i:i+4] for i in i_found if path[i:i+4] == gold_standard]
    if len(kit_id)>0:
        assert(len(set(kit_id))==1)
        kit_id = gold_standard
    else:      
        kit_id = parent_kit_id  

    #Find date
    date = None
    if parent_kit_id is not None:
        idx = path.find(parent_kit_id)
        path = path[idx:]
        idx1 = path.find('_')

        idx2a = path.find('_', idx1+1)     # e.g. ...\S001_07202020_F\...
        idx2b = path.find(os.sep, idx1+1)  # e.g. ...\S001_07202020\...
        idx2 = idx2a if idx2a<idx2b else idx2b

        diff = idx2-(idx1+1)
        #diff should be exactly 8, but sometimes YYYYMMDD format not followed
        #assert(diff<15) 
        if diff<15: 
            date = path[idx1+1:idx2]
        
    return kit_id, parent_kit_id, date
    
#--------------------------------------------------------------------
# PROBABLY CREATE UTILITIES_FILES OR SIMILAR AND PUT FOLLOWING THERE
#--------------------------------------------------------------------
#NOTE: when using glob
#  if recursive is true, the pattern "**" will match any files and zero or 
#    more directories, subdirectories, and symbolic links to directories.
#    If the pattern is followed by os.sep or os.altsep then files will not match
#
#  if recursive is false, it seems that "**" is treated as "*"
#    More precisely, without recursive=True, ** would be interpreted in the normal way, 
#      with each * meaning to match any number of characters EXCLUDING \, 
#      and ** would therefore just be equivalent to *
#    In other words glob(folder+'\\**\\*.ext',recursive=False) would search for any *.ext files 
#      that are exactly one level of subdirectory down from the starting folder, but not in the 
#      starting folder itself or in deeper subdirectories.
#
#  In the past, I have frequently used *\**\ (e.g. base_dir\*\**\file_name_or_whatever*.csv)
#    However, it seems this will give the same result when recursive is true or false
#      when recursive = true:  *\**\ is used, and search is recursive
#      when recursive = false: *\*\ is used, which is still essentially recursive!
#
#    Therefore, I should instead be using **\ (e.g. base_dir\**\file_name_or_whatever*.csv)
#    This will give more control when using recursive = true or false
#      when recursive = true:  **\ is used, and search is recursive
#      when recursive = false: *\ is used, 
#--------------------------------------------------------------------

def get_immediate_sub_dirs(base_dir):
    immediate_sub_dirs = [os.path.join(base_dir, sub) for sub in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, sub))]
    return immediate_sub_dirs

def get_immediate_sub_dir_names(base_dir):
    immediate_sub_dirs = [sub for sub in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, sub))]
    return immediate_sub_dirs
    
#--------------------------------------------------------------------
def glob_find_files_with_pattern(base, pattern, recursive=True):
    search_path = os.path.join(base, pattern) 
    results = glob.glob(search_path, recursive=recursive) 
    return results


def find_all_paths(base_dir, glob_pattern, regex_pattern=None, regex_ignore_case=False, recursive=True):
    # Can only use very simple patterns with glob
    # Also, Windows doesn't seem to care about character case (i.e. case insensitive)
    paths = glob_find_files_with_pattern(base_dir, pattern=glob_pattern, recursive=recursive)
    if regex_pattern is not None:
        if regex_ignore_case:
            paths = [x for x in paths if re.search(regex_pattern, x, flags=re.IGNORECASE)]
        else:
            paths = [x for x in paths if re.search(regex_pattern, x)]
    return paths
#--------------------------------------------------------------------

def is_dir_empty(dir_path):
    assert(os.path.exists(dir_path) and os.path.isdir(dir_path))
    contents = os.listdir(dir_path)
    return True if len(contents)==0 else False

#--------------------------------------------------------------------
def delete_dir(dir_path):
    # os.rmdir only works for empty directories
    # This will work for all
    shutil.rmtree(dir_path)
    
def clear_all_dir_contents(dir_path):
    # clears everything in a directory while keeping the directory
    # NOTE: DELETES FILES AND SUB-DIRECTORIES!
    delete_dir(dir_path)
    os.mkdir(dir_path)

#--------------------------------------------------------------------
def make_tmp_save_dir(
    base_dir_path,
    tmp_dir_name, 
    return_path   = False
):
    r"""
    base_dir_path must already exist!
    os.path.join(base_dir_path, tmp_dir_name) must not exist (or, must be empty if it does exist)!

    tmp_dir_name:
        May be a string or a list of strings
    """
    #--------------------------------------------------
    assert(is_object_one_of_types(tmp_dir_name, [str, list]))
    if isinstance(tmp_dir_name, list):
        tmp_dir_paths = []
        for tmp_dir_name_i in tmp_dir_name:
            tmp_dir_path_i = make_tmp_save_dir(
                base_dir_path = base_dir_path, 
                tmp_dir_name  = tmp_dir_name_i, 
                return_path   = True
            )
            tmp_dir_paths.append(tmp_dir_path_i)
        if return_path:
            return tmp_dir_paths
        else:
            return
    #--------------------------------------------------
    assert(isinstance(tmp_dir_name, str))
    assert(os.path.isdir(base_dir_path))
    #-----
    tmp_dir_path = os.path.join(base_dir_path, tmp_dir_name)
    #-----
    if os.path.exists(tmp_dir_path):
        assert(is_dir_empty(tmp_dir_path))
    else:
        os.makedirs(tmp_dir_path)
    #-----
    if return_path:
        return tmp_dir_path


def del_tmp_save_dir(
    base_dir_path,
    tmp_dir_name
):
    r"""
    os.path.join(base_dir_path, tmp_dir_name) must exist!

    tmp_dir_name:
        May be a string or a list of strings
    """
    #--------------------------------------------------
    assert(is_object_one_of_types(tmp_dir_name, [str, list]))
    if isinstance(tmp_dir_name, list):
        for tmp_dir_name_i in tmp_dir_name:
            del_tmp_save_dir(
                base_dir_path = base_dir_path, 
                tmp_dir_name  = tmp_dir_name_i
            )
        return
    #--------------------------------------------------
    assert(isinstance(tmp_dir_name, str))
    tmp_dir_path = os.path.join(base_dir_path, tmp_dir_name)
    assert(os.path.isdir(tmp_dir_path))
    delete_dir(tmp_dir_path)

#--------------------------------------------------------------------
def do_paths_share_ancestor(path_1, path_2, ancestor_level=1, use_max_in_comparison=False):
    # ancestor_level is furthest back level at which paths should share ancestor
    # e.g. ancestor_level=1 --> paths share parent directory
    # Typically, to reach the common substring, the levels back from path_1 and 
    #   path_2 will be unequal.
    # E.g. 
    #    path_1 = ...\whatever_dir\Results\NISTResults\133026_20200623_061910.xml
    #    path_2 = ...\whatever_dir\Results\SPCAResults\temp_result_dir\133026_20200623182602_FAIL\CTIX6040_sn_133026_trnid_83444_FAIL.xml
    #
    #    level_1 = 2
    #    level_2 = 4
    
    # To judge whether or not path_1 and path_2 share ancestor, the
    # minimum of level_1 and level_2 is compared to ancestor_level
    # But this can be changed by setting use_max_in_comparison=True
    
    common_substr = find_longest_shared_substring(path_1, path_2)
    common_substr = Path(common_substr)

    level_1 = -1
    counter = 0
    while level_1 < 0:
        counter += 1
        path_1 = Path(path_1).parent
        if path_1 == common_substr:
            level_1 = counter
        if counter > 20:
            assert(0) # To prevent accidental infinite loop

    level_2 = -1
    counter = 0
    while level_2 < 0:
        counter += 1
        path_2 = Path(path_2).parent
        if path_2 == common_substr:
            level_2 = counter
        if counter > 20:
            assert(0) # To prevent accidental infinite loop
            
    if use_max_in_comparison:
        level = max(level_1, level_2)
    else:
        level = min(level_1, level_2)
        
    if level <= ancestor_level:
        return True
    else:
        return False
        

def assert_paths_share_ancestor(path_1, path_2, ancestor_level, use_max_in_comparison=False):
    assert(do_paths_share_ancestor(path_1, path_2, ancestor_level, use_max_in_comparison))


#--------------------------------------------------------------------

def search_backwards_for_file(starting_path, file_name_to_find, 
                              max_levels_to_search, search_subdirs_also, 
                              **kwargs):
    # This function will always return a list, even if only returning one path
                              
    # Can also set return_first_found, find_all, and assert_found in kwargs
    # Defaults:
    #   return_first_found = False
    #   find_all           = False
    #   assert_found       = True
    
    # For file_name_to_find, give just the file name and matching patter assuming
    # you are searching the directory where the file lives
    # e.g. Right: 'FullContributors*.csv'
    #      Wrong: '**\FullContributors*.csv'
    # This function will decide what kind of special characters to prepend
    
    # search_subdirs_also set to false will function must faster.
    #   However, this may cause you to miss the file you're looking for
    #   e.g., assume 
    #     starting_path = ...6040CTiX\\SPCAwGSBaseline\\temp_result_dir\\128107_FAIL\\CTIX6040_FAIL.xml
    #   further assume the following files all exist:
    #     ...\6040CTiX\SPCAwGSBaseline\FullContributors.csv
    #     ...\6040CTiX\FullContributors.csv
    #     ...\6040CTiX\GrubbsTest\FullContributors.csv
    #     ...\6040CTiX\SPCAwGSBaseline\OtherDir\FullContributors.csv 
    #   further assume the following
    #     file_name_to_find = 'FullContributors*.csv'
    #     search_subdirs_also = False
    #     return_first_found = False
    #     find_all = True
    #
    #   This configuration will find the following:
    #     ...\6040CTiX\SPCAwGSBaseline\FullContributors.csv
    #     ...\6040CTiX\FullContributors.csv    
    #   but will miss the following:
    #     ...\6040CTiX\GrubbsTest\FullContributors.csv
    #     ...\6040CTiX\SPCAwGSBaseline\OtherDir\FullContributors.csv    
    
    # This function will always return a list, even if only returning one path    
    # In general, if find_all=False, the best match path will be returned 
    #   best match: the path sharing the longest common substring with starting_path
    #               If multiple paths exists with same length of longest common substring,
    #                 the path with the shortest length is returned
    #               If multiple paths still exist at this point, the first is returned
    # If return_first_found=True, the first path(s) found will be returned
    #    If find_all=True, multiple paths will be returned if all found in same iteration
    #    If find_all=False, if multiple paths found from given iteration, the best match
    #      will be returned (see best match defintion above)
    
    return_first_found = kwargs.get('return_first_found', False)
    find_all = kwargs.get('find_all', False)
    assert_found = kwargs.get('assert_found', True)
    
    if search_subdirs_also:
        glob_pattern = os.path.join('**', file_name_to_find) #e.g. 'FullContributors*.csv' --> '**\FullContributors*.csv'
        glob_recursive = True
    else:
        glob_pattern = file_name_to_find
        glob_recursive = False
    
    cur_dir = starting_path
    paths_found = []
    for _ in range(max_levels_to_search):
        cur_dir = Path(cur_dir).parent
        path_found = glob_find_files_with_pattern(cur_dir, pattern=glob_pattern, recursive=glob_recursive)
        paths_found.extend(path_found)
        if return_first_found and len(paths_found)>0:
            break
        
    paths_found = list(set(paths_found)) # remove duplicates
    if not find_all and len(paths_found)>0:        
        closest_path = max(paths_found, key = lambda x:get_length_of_longest_shared_substring(x, starting_path))
        # closest_path will always be single element, even if multiple with same shared length exist
        # So, need to look for others manually
        len_longest_common = get_length_of_longest_shared_substring(closest_path, starting_path)
        closest_paths = [path for path in paths_found 
                         if get_length_of_longest_shared_substring(path, starting_path)==len_longest_common]
        assert(len(closest_paths)>0)
        if len(closest_paths)==1:
            paths_found = [closest_paths[0]]
        else:
            paths_found = [min(closest_paths, key = len)]
    if assert_found:
        assert(len(paths_found)>0)
    return paths_found


#--------------------------------------------------------------------
def find_file_extension(
    file_path_or_name, 
    include_pd=False
):
    r"""
    Find the file extension given a file path or name.
    Very simply minded, this essentially looks for the right-most period and returns
      everything to the right of it.
      If include_pd==True, the period is also returned.
    """
    pd_idx = file_path_or_name.rfind('.')
    assert(pd_idx>-1)
    if include_pd:
        return file_path_or_name[pd_idx:]
    else:
        return file_path_or_name[pd_idx+1:]


#--------------------------------------------------------------------
def append_to_path(save_path, appendix, ext_to_find='.pdf', append_to_end_if_ext_no_found=False):
    # save_path can be single path or list of paths
    # No '_' included by default, so appendix should include it if desired
    # If ext_to_find=None, appendix added to end of save_path
    # e.g. if save_path = .../Path.pdf, use ext_to_find='.pdf'
    #      if save_path = .../Path, use ext_to_find=None
    #
    #----------------------------------
    if ext_to_find is not None and ext_to_find[0] != '.':
        ext_to_find = '.'+ext_to_find
    #----------------------------------    
    if isinstance(save_path, list):
        return_paths = []
        for path in save_path:
            return_paths.append(append_to_path(path, appendix, ext_to_find, append_to_end_if_ext_no_found))
        return return_paths
    #----------------------------------
    assert(isinstance(save_path, str))
    # If appendix is None or '', simply return save_path
    if(not bool(appendix)):
        return save_path
    if ext_to_find is None:
        return_save_path = save_path+appendix
        return return_save_path
    else:
        idxs = [(i.start(),i.end()) for i in re.finditer(ext_to_find, save_path)]
        if len(idxs)<1 and append_to_end_if_ext_no_found:
            return_save_path = save_path+appendix
            return return_save_path
        # Should only find one extention, and it should be at the end of save_path
        assert(len(idxs)==1)
        assert(idxs[0][1]-idxs[0][0] == len(ext_to_find))
        assert(idxs[0][1]==len(save_path))
        #---------------------
        return_save_path = save_path.replace(ext_to_find, f'{appendix}{ext_to_find}')
        return return_save_path
        
        
#--------------------------------------------------------------------
def determine_wait_time(
    run_times_dict, 
    max_calls_per_sec, 
    lookback_period_sec
):
    r"""
    Built specifically for use in Utilities.run_tryexceptwhile_process
    """
    #-------------------------
    calls_ts = pd.Series(run_times_dict)
    calls_ts = calls_ts-calls_ts.max()
    #-------------------------
    calls_subset = calls_ts[calls_ts>-lookback_period_sec]
    if calls_subset.shape[0]<=1:
        return 0
    #-------------------------
    # The average number of seconds between calls is given by: calls_subset.diff().mean()
    # The corresponding frequency is simply the inverse
    calls_freq = 1.0/calls_subset.diff().mean()
    #-------------------------
    print(calls_freq)
    print(max_calls_per_sec)
    print()
    if calls_freq < max_calls_per_sec:
        return 0
    else:
        # The number of entries in the .diff() DF is 1 less than calls_subset, due to NaN first entry
        # Thus, although n_0+1 is desired in the numerator of the first time, it is n_0+1 FOR THE DIFF DF
        #   which is simply calls_subset.shape[0] (NOT calls_subset.shape[0]+1)
        d_ip1 = (calls_subset.shape[0])/max_calls_per_sec - calls_subset.diff().sum()
        # d_ip1 is the needed (approximate) wait time for the next call if the desired max_calls_per_sec is to be recovered
        return d_ip1
        
        
def run_tryexceptwhile_process(
    func,
    func_args_dict, 
    max_calls_per_min   = 1, 
    lookback_period_min = 15, 
    max_calls_absolute  = 1000, 
    verbose             = True
):
    r"""
    Intended for DAQ processes which take significant amounts of time, and might randomly die part way through.
    The function dying is not an issue with the underlying code, but typically an issue with connecting to the database
      or something else on the server side.
    To prevent myself from having to continually check and relaunch processes, I used to run the try/except block within a 
      while loop without any sort of check.
    There was an issue in one instance, and I was locked out by Cyber Security.
    To try to prevent this from happening again, this function was built, which intends to keep the code from contacting
      the database server in rapid succession, and therefore hopefully keeping me from being flagged by Cyber again.
    """
    #-------------------------
    max_calls_per_sec   = max_calls_per_min/60
    lookback_period_sec = lookback_period_min*60
    #-------------------------
    counter = 0
    time_0 = time.time()
    run_times = dict()
    while counter < max_calls_absolute:
        try:
            assert(counter not in run_times.keys())
            #-------------------------
            wait_time_i = determine_wait_time(
                run_times_dict      = run_times, 
                max_calls_per_sec   = max_calls_per_sec,  
                lookback_period_sec = lookback_period_sec
            )
            #-----
            if verbose:
                print(f'counter = {counter}\n\twait_time_i={wait_time_i}\n\n')
            #-----
            time.sleep(wait_time_i)
            #-------------------------
            _ = func(**func_args_dict)
            #-------------------------
            # Stop the loop if the function completes sucessfully
            # Before this function was developed, and this code lived in notebooks, I used a break statement here.
            # Now, instead, I return True
            return True
        except Exception as e:
            print("Function errored out!", e)
            print("Retrying ... ")
        counter += 1
    return False
        
#--------------------------------------------------------------------
def find_spca_contributors_csv(spca_xml, phantom_type, 
                               find_top_5=False, max_levels_to_search = 5, nr_attribute=None, assert_found=True):
    # If find_top_5==False, find FullContributors...csv
    # If find_top_5==True, find Top5Contributors...csv
    # set nr_attribute = e.g. 'CT-80 High Energy Row #0' for CT-80 
    #-------------------------
    if find_top_5:
        file_name_to_find = 'Top5Contributors'
    else:
        file_name_to_find = 'FullContributors'
    #-------------------------
    if phantom_type == PhantomType.kSAT:
        gold_standard = 'S002'
        file_name_to_find += '*.csv'
    elif phantom_type == PhantomType.kA or phantom_type == PhantomType.kB:
        gold_standard = 'F004'
        if phantom_type == PhantomType.kA:
            file_name_to_find += '_PhantomA*.csv'
        else:
            file_name_to_find += '_PhantomB*.csv'
    else:
        assert(0)
        
    if nr_attribute is not None:
        file_name_to_find = append_to_path(save_path = file_name_to_find, 
                                           appendix = nr_attribute.replace(" ", ""), 
                                           ext_to_find = '.csv')
    #---------------------------------------------------------------------------
    # First, search for the file in its typical location
    # If found, return
    temp_result_dir_idx = spca_xml.find(os.sep+'temp_result_dir')
    typical_dir = spca_xml[:temp_result_dir_idx]
    return_path = glob_find_files_with_pattern(typical_dir, pattern=file_name_to_find, recursive=False)
    assert(len(return_path)<2)
    if len(return_path)==1:
        assert(get_kit_id(return_path[0], gold_standard) == get_kit_id(spca_xml, gold_standard))
        return return_path[0]
    #---------------------------------------------------------------------------
    # If path not found in typical location, search backwards for it
    return_first_found = False
    find_all = False
    # First, try quick method not looking in subdirs on way back
    search_subdirs_also = False
    return_path = search_backwards_for_file(spca_xml, file_name_to_find, 
                                            max_levels_to_search, search_subdirs_also=search_subdirs_also, 
                                            return_first_found=return_first_found, find_all=find_all, 
                                            assert_found=False)
    assert(len(return_path)<2)
    if len(return_path)==1:
        assert(get_kit_id(return_path[0], gold_standard) == get_kit_id(spca_xml, gold_standard))
        return return_path[0]
    #---------------------------------------------------------------------------
    # If not found, use longer method of searching subdirs also
    search_subdirs_also = True
    return_path = search_backwards_for_file(spca_xml, file_name_to_find, 
                                            max_levels_to_search, search_subdirs_also=search_subdirs_also, 
                                            return_first_found=return_first_found, find_all=find_all, 
                                            assert_found=False)
    assert(len(return_path)<2)
    if len(return_path)==1:
        assert(get_kit_id(return_path[0], gold_standard) == get_kit_id(spca_xml, gold_standard))
        return return_path[0]
    #---------------------------------------------------------------------------
    # If still not found, either enforce assert_found or return empty string
    if assert_found:
        assert(0)
    else:
        return None
    



    #----------------------------------------------------------------------------------------------------------------------------------------
    def offset_int_tagged_files_in_dir(
        files_dir, 
        file_name_regex, 
        offset_int            = None, 
        new_0_int             = None, 
        new_dir               = None, 
        file_name_glob        = None, 
        copy_and_rename       = False, 
        return_rename_summary = False
    ):
        r"""
        Offset all of the files in files_dir by offset_int.
        The directory files_dir is expected to contain files of the form [file_idx_0, file_idx_1, ..., file_idx_n] where
            idx_0, idx_1, ..., idx_n are integers.
        The files can either be renamed using the offset_int argument OR the new_0_int argument, BUT NOT BOTH.
            For the case of offset_int:
                The files in this directory will be renamed [file_{idx_0+offset_int}, file_{idx_1+offset_int}, ..., file_{idx_n+offset_int}]
            For the case of new_0_int:
                The files in this directory will be renamed [file_{new_0_int}, file_{new_0_int+1}, ..., file_{new_0_int+n_files-1}]
        The files can simply be moved/renamed (copy_and_rename==False), or copied to the new directory (copy_and_rename==True and 
            new_dir not None)
        -------------------------
        files_dir:
            The directory housing the files to be renamed
            
        file_name_regex:
            A regex patten used to both identify the files to be renamed and to find the integer tag for each file.
            NOTE: file_name_regex MUST have some sort of digit capture (e.g., contain '(\d*)')
                  e.g., for the case of end events, one would use file_name_regex = r'end_events_(\d*).csv'
                  
        offset_int/new_0_int:
            These direct how the files will be renamed.  
            ONLY ONE OF THESE SHOULD BE USED ==> one should always be set to None and the other should be set to some int value
            offset_int:
                Take the identifier/tag ints, and simply shift them by offset_int.
            new_0_int:
                Start the identifier tags at new_0_int, and label from new_0_int to new_0_int + len(files in files_dir)-1
                
        new_dir:
            The directory to which the renamed files will be saved.
            Default value is None, which means the files will be saved in the input files_dir
                  
        file_name_glob:
            Used in Utilities.find_all_paths to help find the paths to be renamed.  By default this is set to None, which is then
                changed to = '*', meaning the glob portion doesn't trim down the list of files at all, but returns all contained
                in the directory.  Therefore, file_name_regex does all of the work, which is fine and really as designed
                
        copy_and_rename:
            Directs whether to call rename or copy.
            copy_and_rename=False:
                Default behavior, which means the files will be renamed and replaced.
            copy_and_rename=False:
                The files will be copied and renamed, with the originals kept intact.  This is only possible if new_dir is not None
                and new_dir != files_dir
            
        """
        #-------------------------
        # Exclusive or for offset_int and new_0_int (meaning, one but not both must not be None)
        assert(    offset_int is not None or new_0_int is not None)
        assert(not(offset_int is not None and new_0_int is not None))
        #-------------------------
        assert(os.path.isdir(files_dir))
        if file_name_glob is None:
            file_name_glob = '*'
        if new_dir is None:
            new_dir = files_dir
        #-------------------------
        if new_dir==files_dir:
            copy_and_rename = False
        #-------------------------
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        #-------------------------
        paths = find_all_paths(
            base_dir      = files_dir, 
            glob_pattern  = file_name_glob, 
            regex_pattern = file_name_regex
        )
        #-------------------------
        paths_w_tags = []
        for path in paths:
            tag = re.findall(file_name_regex, path)
            print(path)
            print(file_name_regex)
            print(tag)
            print()
            # Should have only been one tag found per path
            if len(tag)>1:
                print(tag)
            assert(len(tag)==1)
            tag = int(tag[0])
            paths_w_tags.append((path, tag))
        # NOTE: Want to sort in reverse so that highest is first.  This is so there are no naming issues when the rename occurs.  
        #       E.g., imaging there are 10 members in paths, tagged 0 through nine, and they are to be offset by 1
        #         If we started with the lowest tag, file_0, shifting it by 1 would make it file_1, which already exists!
        #         If, instead, we start with the highest tag, file_9, it shifts by 1 to file_10, which is not an issue.
        paths_w_tags = natsorted(paths_w_tags, key=lambda x: x[1], reverse=True)
        #-------------------------
        rename_summary = {}
        for i,path_w_tag_i in enumerate(paths_w_tags):
            path_i      = path_w_tag_i[0]
            tag_i       = path_w_tag_i[1]
            file_name_i = os.path.basename(path_i)
            #-----
            assert(str(tag_i) in file_name_i)
            if offset_int is not None:
                repl_int_i = str(tag_i+offset_int)
            elif new_0_int is not None:
                # Remember, sorted in descending order
                repl_int_i = str(len(paths_w_tags)+new_0_int-(i+1))
            else:
                assert(0)
            new_file_name_i = file_name_i.replace(str(tag_i), str(repl_int_i))
            #-----
            assert(new_file_name_i not in os.listdir(new_dir))
            new_path_i = os.path.join(new_dir, new_file_name_i)
            #-----
            assert(path_i not in rename_summary.keys())
            rename_summary[path_i] = new_path_i
            #-----
            if copy_and_rename:
                shutil.copy(src=path_i, dst=new_path_i)
            else:
                os.rename(path_i, new_path_i)
        #-------------------------
        if return_rename_summary:
            return rename_summary
        
        
    #---------------------------------------------------------------------------    
    def offset_int_tagged_files_w_summaries_in_dir(
        files_dir, 
        file_name_regex, 
        offset_in               = None, 
        new_0_int               = None, 
        new_dir                 = None, 
        file_name_glob          = None, 
        #-----
        summary_files_dir       = None,
        summary_file_name_regex = None,
        summary_file_name_glob  = None, 
        summary_new_dir         = None, 
        #-----
        copy_and_rename         = False, 
        return_rename_summary   = False
    ):
        r"""
        """
        #-------------------------
        if summary_files_dir is None:
            summary_files_dir          = os.path.join(files_dir, 'summary_files')
        if summary_file_name_regex is None:
            summary_file_name_regex    = file_name_regex.replace('_(\d*).csv', '_([0-9]*)_summary.json')
        if summary_file_name_glob is None:
            if file_name_glob is None:
                summary_file_name_glob = '*'
            else:
                summary_file_name_glob = file_name_glob.replace('_*.csv', '*_summary.json')
        if summary_new_dir is None:
            if new_dir is None:
                summary_new_dir = os.path.join(files_dir, 'summary_files')
            else:
                summary_new_dir = os.path.join(new_dir, 'summary_files')
        #-------------------------
        files_rename_summary = offset_int_tagged_files_in_dir(
            files_dir             = files_dir, 
            file_name_regex       = file_name_regex, 
            offset_int            = offset_int, 
            new_0_int             = new_0_int, 
            new_dir               = new_dir, 
            file_name_glob        = file_name_glob, 
            copy_and_rename       = copy_and_rename, 
            return_rename_summary = True
        )
        #-------------------------
        summaries_rename_summary = offset_int_tagged_files_in_dir(
            files_dir             = summary_files_dir, 
            file_name_regex       = summary_file_name_regex, 
            offset_int            = offset_int, 
            new_0_int             = new_0_int, 
            new_dir               = summary_new_dir, 
            file_name_glob        = summary_file_name_glob, 
            copy_and_rename       = copy_and_rename, 
            return_rename_summary = True
        )
        #-------------------------
        assert(len(files_rename_summary)==len(summaries_rename_summary))
        if return_rename_summary:
            return (files_rename_summary, summaries_rename_summary)






    #----------------------------------------------------------------------------------------------------------------------------------------