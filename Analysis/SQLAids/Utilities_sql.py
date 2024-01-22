#!/usr/bin/env python

"""
Utilities specifically for DataFrames
"""
# List of functions:

#---------------------------------------------------------------------
import os, sys
import re
import pyodbc

import Utilities_config

# IMPORTANT!!!!!: get_pwd below, get_prod_llap_hive_connection, ..., get_eemsp_oracle_connection
#                 all match functions in  Utilities
#                 They were reproduced here so SQLAids can be a standalone package.
#                 However, this may need to be revisited in the future.
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
    dsn = 'Athena Prod'
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
    conn_outages = pyodbc.connect('Driver={Oracle in InstantClient_12_64};DBQ= aep01dbadm01/UTLDB01P;Uid=' + db_user + ';Pwd=' + pwd_oracle)
    return conn_outages
    
def get_eemsp_oracle_connection(aep_user_id=None):
    #-------------------------
    if aep_user_id is None:
        db_user = Utilities_config.get_aep_user_id()
    else:
        db_user = aep_user_id
    #-------------------------
    pwd_oracle = get_pwd('Oracle')
    conn_outages = pyodbc.connect('Driver={Oracle in InstantClient_12_64};DBQ= EEMSP;Uid=' + db_user + ';Pwd=' + pwd_oracle)
    return conn_outages

#---------------------------------------------------------------------
def join_list(list_to_join, quotes_needed=True, join_str=','):
    # Default quotes_needed=True because if quotes are not needed I will
    #   probably simply use ','.join(list_to_join)
    if quotes_needed:
        return_str = join_str.join(["'{}'".format(x) for x in list_to_join])
    else:
        # The join method apparently only works if the iterable contains strings...
        # Therefore, first convert all elements to strings. If the elements are alraedy
        #   strings this has no effect
        return_str = join_str.join([str(x) for x in list_to_join])
    return return_str

def join_list_w_quotes(list_to_join, join_str=','):
    return join_list(list_to_join=list_to_join, quotes_needed=True, join_str=join_str)
    
# IMPORTANT!!!!!: are_strings_equal below matches Utilities.are_strings_equal
#                 It was reproduced here so SQLAids can be a standalone package.
#                 However, this may need to be revisited in the future.
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
            
            
# IMPORTANT!!!!!: are_all_list_elements_of_type, are_all_list_elements_one_of_types, and is_object_one_of_types 
#                 below match those in Utilities
#                 These were reproduced here so SQLAids can be a standalone package.
#                 However, this may need to be revisited in the future.            
#--------------------------------------------------------------------
def are_all_list_elements_of_type(lst, typ):
    assert(isinstance(lst, list) or isinstance(lst, tuple))
    for el in lst:
        if not isinstance(el, typ):
            return False
    return True

def are_all_list_elements_one_of_types(lst, types):
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

def is_object_one_of_types(obj, types):
    return are_all_list_elements_one_of_types([obj], types)
    
  
# IMPORTANT!!!!!: supplement_dict_with_default_values below matchesthat in Utilities
#                 It was reproduced here so SQLAids can be a standalone package.
#                 However, this may need to be revisited in the future.  
#--------------------------------------------------------------------
def supplement_dict_with_default_values(to_supplmnt_dict, default_values_dict, extend_any_lists=False, inplace=True):
    r"""
    Adds key/values from default_values_dict to to_supplmnt_dict.
    If a key IS NOT contained in to_supplmnt_dict
      ==> a new key/value pair is simply created.
    If a key IS already contained in to_supplmnt_dict
      ==> value (to_supplmnt_dict[key]) replaced with that from default_values_dict (default_values_dict[key])  
    """
    if not inplace:
        to_supplmnt_dict = copy.deepcopy(to_supplmnt_dict)
    #---------------
    for key in default_values_dict:
        to_supplmnt_dict[key] = to_supplmnt_dict.get(key, default_values_dict[key])
        #---------------
        if extend_any_lists and isinstance(default_values_dict[key], list):
            if not(isinstance(to_supplmnt_dict[key], list)):
                to_supplmnt_dict[key] = [to_supplmnt_dict[key]]
            to_supplmnt_dict[key].extend([x for x in default_values_dict[key] 
                                          if x not in to_supplmnt_dict[key]])
    return to_supplmnt_dict

# IMPORTANT!!!!!: prepend_tabs_to_each_line below matchesthat in Utilities
#                 It was reproduced here so SQLAids can be a standalone package.
#                 However, this may need to be revisited in the future.  
#--------------------------------------------------------------------
def prepend_tabs_to_each_line(input_statement, n_tabs_to_prepend=1):
    if n_tabs_to_prepend > 0:
        join_str = '\n' + n_tabs_to_prepend*'\t'
        input_statement = join_str.join(input_statement.splitlines())
        # Need to also prepend first line...
        input_statement = n_tabs_to_prepend*'\t' + input_statement
    return input_statement