#!/usr/bin/env python

r"""
Holds AMIUsgInst class.  See AMIUsgInst.AMIUsgInst for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
from scipy import stats
import datetime
import time
from natsort import natsorted, ns
#--------------------------------------------------
import CommonLearningMethods as clm
from AMIUsgInst_SQL import AMIUsgInst_SQL
from GenAn import GenAn
from AMINonVee import AMINonVee
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Utilities_dt
#--------------------------------------------------

class AMIUsgInst(GenAn):
    r"""
    class AMIUsgInst documentation
    """
    def __init__(self, 
                 df_construct_type=None, 
                 contstruct_df_args=None, 
                 init_df_in_constructor=True, 
                 build_sql_function=None, 
                 build_sql_function_kwargs=None, 
                 **kwargs):
        r"""
        if df_construct_type==DFConstructType.kReadCsv or DFConstructType.kReadCsv:
          contstruct_df_args needs to have at least 'file_path'
        if df_construct_type==DFConstructType.kRunSqlQuery:
          contstruct_df_args needs at least 'conn_db'      
        """
        #--------------------------------------------------
        # First, set self.build_sql_function and self.build_sql_function_kwargs
        # and call base class's __init__ method
        #---------------
        self.build_sql_function = (build_sql_function if build_sql_function is not None 
                                   else AMIUsgInst_SQL.build_sql_usg_inst)
        #---------------
        self.build_sql_function_kwargs = (build_sql_function_kwargs if build_sql_function_kwargs is not None 
                                          else {})
        #--------------------------------------------------
        super().__init__(
            df_construct_type=df_construct_type, 
            contstruct_df_args=contstruct_df_args, 
            init_df_in_constructor=init_df_in_constructor, 
            build_sql_function=self.build_sql_function, 
            build_sql_function_kwargs=self.build_sql_function_kwargs, 
            **kwargs
        )
        
    #****************************************************************************************************
    def get_conn_db(self):
        return Utilities.get_athena_prod_aws_connection()
    
    def get_default_cols_and_types_to_convert_dict(self):
        cols_and_types_to_convert_dict = {
            'measurement_value':np.float, 
            'aep_readtime':datetime.datetime
        }
        return cols_and_types_to_convert_dict
    
    def get_full_default_sort_by(self):
        full_default_sort_by = ['aep_readtime', 'measurement_type']
        return full_default_sort_by
        
        
    #*****************************************************************************************************************
    @staticmethod
    def perform_std_initiation_and_cleaning(df, 
                                            drop_na_values=True, 
                                            inplace=True, 
                                            **kwargs):
        kwargs['timestamp_col']      = kwargs.get('timestamp_col', 'aep_readtime_utc')
        kwargs['start_time_col']     = kwargs.get('start_time_col', None)
        kwargs['end_time_col']       = kwargs.get('end_time_col', None)
        kwargs['timezoneoffset_col'] = kwargs.get('timezoneoffset_col', 'timezoneoffset')
        kwargs['value_cols']         = kwargs.get('value_cols', ['measurement_value'])
        #-------------------------
        df = AMINonVee.perform_std_initiation_and_cleaning(
            df=df, 
            drop_na_values=drop_na_values, 
            inplace=inplace,
            **kwargs
        )
        #-------------------------
        return df