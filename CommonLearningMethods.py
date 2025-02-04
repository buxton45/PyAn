#!/usr/bin/env python

r"""
A collection of methods built to help initial learning and development efforts.
Will likely be dispersed to various other modules/classes/etc at some point.

Included functions:

get_utc_datetime_from_timestamp(timestamp)

get_timedelta_from_timezoneoffset_OLD(timezoneoffset, 
                                      pattern=r'([+-]{0,1})(\d{2}):(\d{2})', 
                                      expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2})
                                      
extract_datetime_elements_from_string(datetime_str, 
                                          pattern=r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-]{0,1})(\d{2}):(\d{2})', 
                                          expected_match_dict={'year':0, 'month':1, 'day':2, 
                                                               'hours':3, 'minutes':4, 'seconds':5, 
                                                               'tz_sign':6, 'tz_hours':7, 'tz_minutes':8})
                                                               
get_timedelta_from_timezoneoffset(timezoneoffset, 
                                      pattern=r'([\s+-]{0,1})(\d{2}):(\d{2})', 
                                      expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2})                                 
                                      
clean_timeperiod_entry(timeperiod)

extract_tz_from_tz_aware_dt_str(timeperiod, 
                                    pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})')
                                    
extract_tz_parts_from_tz_aware_dt_str(timeperiod, 
                                          pattern_full=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})', 
                                          pattern_tz=r'([+-])(\d{2}):(\d{2})')
                                          
substring_in_all(substr, arr)

find_longest_substring_shared_by_all(arr)

remove_longest_substring_shared_by_all(arr)

remove_longest_substring_shared_by_all_columns_in_df(df, inplace=True)

remove_prepend_from_columns_in_df(df, end_of_prepend_indicator='.', inplace=True)

match_events_in_df_to_outages(df_events, df_outages, 
                                  return_n_events_with_mult_matches=False, 
                                  events_placement_cols=None, events_cols=None, outages_cols=None, 
                                  lowercase_cols_default=False, verbose=True)
                                  
set_all_outages_info_in_events_df_original(df_events, df_outage, 
                                               outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                               outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                               cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'], 
                                               placement_cols = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                                              )

get_outage_info(outg_rec_nb, df_outage, 
                    outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                    cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                   )
                   
get_outage_info_for_events_df_w_single_outage(df_events, df_outage, 
                                                  outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                                  outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                                  cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                                                 )
                                                 
set_outage_info_in_events_df_w_single_outage(df_events, df_outage, 
                                                 outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                                 outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                                 cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'], 
                                                 placement_cols = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                                                )
                                                
set_all_outages_info_in_events_df(df_events, df_outage, 
                                      outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                      outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                      cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'], 
                                      placement_cols =                ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'],
                                      dropna=False, 
                                      lowercase_cols_default=False
                                     )

"""
__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy import stats
import datetime
from packaging import version
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_dt
#--------------------------------------------------

#**********************************************************************************************************************************************
# def get_utc_datetime_from_timestamp(timestamp):
#     return datetime.datetime.utcfromtimestamp(timestamp)
    
# #**********************************************************************************************************************************************    
# def get_timedelta_from_timezoneoffset_OLD(timezoneoffset, 
#                                       pattern=r'([+-]{0,1})(\d{2}):(\d{2})', 
#                                       expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2}):
#     r"""
#     OLD VERSION.  Only kept temporarily in case the new version has issues.  New version implemented 28 Jan 2022.  DELETE ME BY 1 March 2022
#     Returns datetime.timedelta object given a timezoneoffset string (e.g., '-04:00') down to microseconds if desired.
    
#     expected_match_dict is used to interpret the returned tuple of elements.
#       The value of a given key corresponds to its index location in the returned match results tuple
#       len(expected_match_dict) is the number of matched items to be returned when pattern is matched
#       The matches to be returned are specified by being enclosed in ()
    
#     Below, re.findall(pattern, timezoneoffset) should find only one instance of pattern
#       with len(expected_match_dict) elements
    
#     For the typical case with the default arguments above:
#       e.g., if timezoneoffset=='-04:00' --> re.findall(pattern, timezoneoffset) = [('-', '04', '00')]
#       e.g., if timezoneoffset== '04:00' --> re.findall(pattern, timezoneoffset) = [('', '04', '00')]
    
#     As another example, consider the following:
#       timezoneoffset = '12T00:15:00-04:00'
#       pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}'
#       expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}
#       --> re.findall(pattern, timezoneoffset) = [('12', '00', '15', '00')]
    
#     --------------------------------------------------------------
#     EXAMPLE 1
#     Simplest case and how this will likely be used 99% of the time
#     >>> get_timedelta_from_timezoneoffset('-05:00')
#     datetime.timedelta(days=-1, seconds=68400)
    
#     --------------------------------------------------------------
#     EXAMPLE 2
#     If timezoneoffset from the previous example also included seconds and fractions of a second, 
#     but the pattern (and expected_match_dict) remain the same as in the previous example.
#     -----
#     >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
#     ...                                   pattern=r'([+-]{0,1})(\d{2}):(\d{2})', 
#     ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2})
#     datetime.timedelta(days=-1, seconds=68400)
    
#     --------------------------------------------------------------
#     EXAMPLE 3
#     To extract the full value from EXAMPLE 2, one would need to adjust pattern and expected_match_dict
#     -----
#     >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
#     ...                                   pattern=r'([+-]{0,1})(\d{2}):(\d{2}):(\d{2}).(\d{3})(\d{3})', 
#     ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2, 
#     ...                                                        'seconds':3, 'milliseconds':4, 'microseconds':5})
#     datetime.timedelta(days=-1, seconds=68340, microseconds=876544)
    
#     --------------------------------------------------------------
#     EXAMPLES 4 and 5
#     This function can also simply ignore portions of timezoneoffset if desired.
#     Imagine, e.g., timezoneoffset = '12T00:15:00-04:00', but one wants to ignore the tz
#     information (i.e., ignore -04:00) and convert the datetime to a timedelta.
#     In such a case, one could use:
#       expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}   
#       pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}' OR, EVEN BETTER
#       pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*'
#     >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
#     ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}', 
#     ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
#     datetime.timedelta(days=12, seconds=900)
    
#     >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
#     ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*', 
#     ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
#     datetime.timedelta(days=12, seconds=900)
#     """
#     #!!!!!!!!!!!!!!!!!!! DELETE ME BY 1 March 2022 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     time_delta = re.findall(pattern, timezoneoffset)
#     assert(len(time_delta)==1)
#     if len(time_delta)>1:
#         time_delta = time_delta[0]
#     assert(len(time_delta)==len(expected_match_dict))

#     # ----- Unpack time_delta using expected_match_dict -----
#     # -- First, get indices if exist in expected_match_dict
#     # Most common
#     multiplier_idx   = expected_match_dict.get('multiplier', -1)
#     hours_idx        = expected_match_dict.get('hours', -1)
#     minutes_idx      = expected_match_dict.get('minutes', -1)
#     # Less common
#     days_idx         = expected_match_dict.get('days', -1)
#     weeks_idx        = expected_match_dict.get('weeks', -1)
#     seconds_idx      = expected_match_dict.get('seconds', -1)
#     milliseconds_idx = expected_match_dict.get('milliseconds', -1)
#     microseconds_idx = expected_match_dict.get('microseconds', -1)

#     # -- Use the indices found above to grab the correct entries from time_delta
#     #    If an index==-1, set it to default value ('' for multiplier_idx, 0 for all else)
#     #    NOTE: The order DOES matter below.  If time_delta[idx] is called first, it will
#     #    cause the program to crash when idx<0.  If it is called second, everything is fine
#     multiplier   = '' if multiplier_idx<0   else time_delta[multiplier_idx]
#     hours        = 0 if hours_idx<0        else time_delta[hours_idx]
#     minutes      = 0 if minutes_idx<0      else time_delta[minutes_idx]
#     # ---
#     days         = 0 if days_idx<0         else time_delta[days_idx]
#     weeks        = 0 if weeks_idx<0        else time_delta[weeks_idx]
#     seconds      = 0 if seconds_idx<0      else time_delta[seconds_idx]
#     milliseconds = 0 if milliseconds_idx<0 else time_delta[milliseconds_idx]
#     microseconds = 0 if microseconds_idx<0 else time_delta[microseconds_idx]
#     # --------------- Done unpacking ------------------------

#     # multiplier should either be: 
#     #    if timezoneoffset is positive --> multiplier = '' (i.e., an empty string)
#     #    if timezoneoffset is negative --> multiplier = '-'
#     #    if multiplier not included in search (nor in expected_match_dict)) --> multiplier=''
#     assert(len(multiplier)==0 or multiplier=='-')
#     multiplier = 1 if len(multiplier)==0 else -1

#     time_delta = datetime.timedelta(hours=multiplier*float(hours), minutes=multiplier*float(minutes), 
#                                     days=multiplier*float(days), weeks=multiplier*float(weeks), 
#                                     seconds=multiplier*float(seconds), milliseconds=multiplier*float(milliseconds), 
#                                     microseconds=multiplier*float(microseconds))
#     return time_delta
    
# #**********************************************************************************************************************************************    
# def extract_datetime_elements_from_string(datetime_str, 
#                                           pattern=r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-]{0,1})(\d{2}):(\d{2})', 
#                                           expected_match_dict={'year':0, 'month':1, 'day':2, 
#                                                                'hours':3, 'minutes':4, 'seconds':5, 
#                                                                'tz_sign':6, 'tz_hours':7, 'tz_minutes':8}):
#     r"""
#     Returns a dict of the datetime elements defined by pattern and expected_match_dict found in datetime_str.
#     (datetime_str : str, pattern : str (regex), expected_match_dict: dict(k=str, v=int)) -> dict(k=str, v=str)
#     The default values are intended for a typical AEP datetime format of e.g., datetime_str = '2021-10-12T00:15:00-04:00'
    
#     expected_match_dict is used to interpret the returned tuple of elements.
#       The value of a given key corresponds to its index location in the returned match results tuple
#       len(expected_match_dict) is the number of matched items to be returned when pattern is matched
#       The matches to be returned are specified by being enclosed in ()
    
#     For a typical case of datetime_str = '2021-10-12T00:15:00-04:00' with the default pattern and 
#     expected_match_dict arguments, the function should return:
#     return_dict = {'year': '2021', 'month': '10', 'day': '12', 
#                    'hours': '00', 'minutes': '15', 'seconds': '00', 
#                    'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
#     For the typical case with the default arguments above:
#       e.g., if timezoneoffset=='-04:00' --> re.findall(pattern, timezoneoffset) = [('-', '04', '00')]
#       e.g., if timezoneoffset== '04:00' --> re.findall(pattern, timezoneoffset) = [('', '04', '00')]
    
#     As another example, consider the following:
#       timezoneoffset = '12T00:15:00-04:00'
#       pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}'
#       expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}
#       --> re.findall(pattern, timezoneoffset) = [('12', '00', '15', '00')]
    
#     --------------------------------------------------------------
#     EXAMPLE 1
#       datetime_str = '2021-10-12T00:15:00-04:00'
#       default pattern and expected_match_dict
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00')
#     {'year': '2021', 'month': '10', 'day': '12', 'hours': '00', 'minutes': '15', 'seconds': '00', 'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
#     --------------------------------------------------------------
#     EXAMPLES 2 and 3
#     Same datetime_str as EXAMPLE 1, but suppose one wants the entire date, entire time, and entire timezone information
#       datetime_str = '2021-10-12T00:15:00-04:00'
#       expected_match_dict = {'date':0, 'time':1, 'timezone':2}
#       pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]{0,1}\d{2}:\d{2})' <-- careful
#       pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(.*)'                   <-- careless
#     -----
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
#     ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]{0,1}\d{2}:\d{2})', 
#     ...                                       expected_match_dict={'date':0, 'time':1, 'timezone':2})
#     {'date': '2021-10-12', 'time': '00:15:00', 'timezone': '-04:00'}
    
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
#     ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(.*)', 
#     ...                                       expected_match_dict={'date':0, 'time':1, 'timezone':2})
#     {'date': '2021-10-12', 'time': '00:15:00', 'timezone': '-04:00'}
    
#     --------------------------------------------------------------
#     EXAMPLE 4
#     Same as EXAMPLES 2 and 3, but suppose one does not care about the timezone information
#       datetime_str = '2021-10-12T00:15:00-04:00'
#       expected_match_dict = {'date':0, 'time':1}
#       pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})'
#     -----
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
#     ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})', 
#     ...                                       expected_match_dict={'date':0, 'time':1})
#     {'date': '2021-10-12', 'time': '00:15:00'}
    
#     --------------------------------------------------------------
#     EXAMPLES 5, 6, 7
#     Suppose one ONLY wants to timezone information
#       EXAMPLE 5
#         datetime_str = '2021-10-12T00:15:00-04:00'
#         expected_match_dict = {'tz_sign':0, 'tz_hours':1, 'tz_minutes':2}
#         pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1})(\d{2}):(\d{2})'

#       EXAMPLE 6  
#         !!!!! CAREFUL !!!!! 
#           Using pattern=r'([+-]{0,1})(\d{2}):(\d{2})' would cause found_datetime_elements = re.findall(pattern, datetime_str) 
#           to return [('', '00', '15'), ('-', '04', '00')] which would cause assert(len(found_datetime_elements)==1) to fail!
#         !!!!! HOWEVER !!!!! 
#           if one is certain the timezone will begin with + or - (and not simple be blank for the case of +),
#           one could use pattern=r'([+-]{1})(\d{2}):(\d{2})'
#         -----
#         datetime_str = '2021-10-12T00:15:00-04:00'
#         expected_match_dict = {'tz_sign':0, 'tz_hours':1, 'tz_minutes':2}
#         pattern=r'([+-]{1})(\d{2}):(\d{2})'
        
#       EXAMPLE 7
#         datetime_str = '2021-10-12T00:15:00-04:00'
#         expected_match_dict = {'timezone':0}
#         pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1}\d{2}:\d{2})' (or pattern=r'([+-]{1}\d{2}:\d{2})')
#     -----
#     EX 5
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
#     ...                                       pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1})(\d{2}):(\d{2})', 
#     ...                                       expected_match_dict={'tz_sign':0, 'tz_hours':1, 'tz_minutes':2})
#     {'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
#     EX 6
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
#     ...                                       pattern=r'([+-]{1})(\d{2}):(\d{2})', 
#     ...                                       expected_match_dict={'tz_sign':0, 'tz_hours':1, 'tz_minutes':2})
#     {'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
#     EX 7
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
#     ...                                       pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1}\d{2}:\d{2})', 
#     ...                                       expected_match_dict={'timezone':0})
#     {'timezone': '-04:00'}
    
#     --------------------------------------------------------------
#     EXAMPLE 8
#     The purpose here is to show how to include whitespace using \s (see [\s+-]{0,1}\d{2}:\d{2} in pattern below)
#     Suppose the datetime_str does not have + or - between the time and timezone, but instead just has a space
#       datetime_str = '2021-10-12T00:15:00 04:00'
#       expected_match_dict = {'date':0, 'time':1, 'timezone':2}
#       pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([\s+-]{0,1}\d{2}:\d{2})'
#     -----
#     >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00 04:00', 
#     ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([\s+-]{0,1}\d{2}:\d{2})', 
#     ...                                       expected_match_dict={'date':0, 'time':1, 'timezone':2})
#     {'date': '2021-10-12', 'time': '00:15:00', 'timezone': ' 04:00'}
 
#     """
#     #-------------------------------------------------------------------------------------------
#     # Below, re.findall(pattern, datetime_str) should find only one instance of overall pattern
#     #   with len(expected_match_dict) elements
#     found_datetime_elements = re.findall(pattern, datetime_str)
#     assert(len(found_datetime_elements)==1) #only intended to find one instance of overall pattern
#     if len(expected_match_dict)>1:
#         found_datetime_elements = found_datetime_elements[0]
#     assert(len(found_datetime_elements)==len(expected_match_dict))
#     #------------------
#     return_dict = {}
#     for element, idx in expected_match_dict.items():
#         assert(element not in return_dict)
#         return_dict[element] = found_datetime_elements[idx]
#     return return_dict
    
# #**********************************************************************************************************************************************    
# def get_timedelta_from_timezoneoffset(timezoneoffset, 
#                                       pattern=r'([\s+-]{0,1})(\d{2}):(\d{2})', 
#                                       expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2}):
#     r"""
#     Returns datetime.timedelta object given a timezoneoffset string (e.g., '-04:00') down to microseconds if desired.
#     (timezoneoffset : str, pattern : str (regex), expected_match_dict: dict(k=str, v=int)) -> datetime.timedelta
    
#     NOTE: expected_match_dict.keys() must all be found in 
#       expected_elements = ['multiplier', 'hours', 'minutes', 
#                            'days', 'weeks', 'seconds', 'milliseconds', 'microseconds']
    
#     expected_match_dict is used to interpret the returned tuple of elements.
#       The value of a given key corresponds to its index location in the returned match results tuple
#       len(expected_match_dict) is the number of matched items to be returned when pattern is matched
#       The matches to be returned are specified by being enclosed in ()
    
#     Below, re.findall(pattern, timezoneoffset) should find only one instance of pattern
#       with len(expected_match_dict) elements
    
#     For the typical case with the default arguments above:
#       e.g., if timezoneoffset=='-04:00' --> re.findall(pattern, timezoneoffset) = [('-', '04', '00')]
#       e.g., if timezoneoffset== '04:00' --> re.findall(pattern, timezoneoffset) = [('', '04', '00')]
    
#     As another example, consider the following:
#       timezoneoffset = '12T00:15:00-04:00'
#       pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}'
#       expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}
#       --> re.findall(pattern, timezoneoffset) = [('12', '00', '15', '00')]
    
#     --------------------------------------------------------------
#     EXAMPLE 1
#     Simplest case and how this will likely be used 99% of the time
#     >>> get_timedelta_from_timezoneoffset('-05:00')
#     datetime.timedelta(days=-1, seconds=68400)
    
#     --------------------------------------------------------------
#     EXAMPLE 2
#     If timezoneoffset from the previous example also included seconds and fractions of a second, 
#     but the pattern (and expected_match_dict) remain the same as in the previous example.
#     -----
#     >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
#     ...                                   pattern=r'([\s+-]{0,1})(\d{2}):(\d{2})', 
#     ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2})
#     datetime.timedelta(days=-1, seconds=68400)
    
#     --------------------------------------------------------------
#     EXAMPLE 3
#     To extract the full value from EXAMPLE 2, one would need to adjust pattern and expected_match_dict
#     -----
#     >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
#     ...                                   pattern=r'([\s+-]{0,1})(\d{2}):(\d{2}):(\d{2}).(\d{3})(\d{3})', 
#     ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2, 
#     ...                                                        'seconds':3, 'milliseconds':4, 'microseconds':5})
#     datetime.timedelta(days=-1, seconds=68340, microseconds=876544)
    
#     --------------------------------------------------------------
#     EXAMPLES 4 and 5
#     This function can also simply ignore portions of timezoneoffset if desired.
#     Imagine, e.g., timezoneoffset = '12T00:15:00-04:00', but one wants to ignore the tz
#     information (i.e., ignore -04:00) and convert the datetime to a timedelta.
#     In such a case, one could use:
#       expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}   
#       pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}' OR, EVEN BETTER
#       pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*'
#     >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
#     ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}', 
#     ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
#     datetime.timedelta(days=12, seconds=900)
    
#     >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
#     ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*', 
#     ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
#     datetime.timedelta(days=12, seconds=900)
 
#     """
#     #-------------------------
#     expected_elements = ['multiplier', 'hours', 'minutes', 
#                          'days', 'weeks', 'seconds', 'milliseconds', 'microseconds']
#     assert(all(x in expected_elements for x in expected_match_dict.keys()))
#     #-------------------------
#     found_match_dict = extract_datetime_elements_from_string(datetime_str=timezoneoffset, 
#                                                              pattern=pattern, 
#                                                              expected_match_dict=expected_match_dict)
#     #-------------------------
#     #    If an element is not found, set it to default value ('+' for multiplier_idx, 0 for all else)
#     # Most common
#     multiplier   = found_match_dict.get('multiplier', '+')
#     hours        = found_match_dict.get('hours', 0)
#     minutes      = found_match_dict.get('minutes', 0)
#     # Less common
#     days         = found_match_dict.get('days', 0)
#     weeks        = found_match_dict.get('weeks', 0)
#     seconds      = found_match_dict.get('seconds', 0)
#     milliseconds = found_match_dict.get('milliseconds', 0)
#     microseconds = found_match_dict.get('microseconds', 0)

#     # multiplier should be '+', '-', '', or ' '
#     #   multiplier == '-'              --> negative
#     #   multiplier == '+', '', or ' ' ---> positive
#     assert(multiplier in ['+', '-', '', ' '])
#     multiplier = multiplier.strip()
#     if not multiplier or multiplier=='+':
#         multiplier = 1
#     elif multiplier=='-':
#         multiplier = -1
#     else:
#         assert(0)
#     #-------------------------
#     time_delta = datetime.timedelta(hours=multiplier*float(hours), minutes=multiplier*float(minutes), 
#                                     days=multiplier*float(days), weeks=multiplier*float(weeks), 
#                                     seconds=multiplier*float(seconds), milliseconds=multiplier*float(milliseconds), 
#                                     microseconds=multiplier*float(microseconds))
#     return time_delta

    
# #**********************************************************************************************************************************************    
# def clean_timeperiod_entry(timeperiod):
#     # e.g., timeperiod = '2021-10-12T00:15:00-04:00'
#     pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})-\d{2}:\d{2}'
#     date_and_time = re.findall(pattern, timeperiod)
#     assert(len(date_and_time)==1)
#     assert(len(date_and_time[0])==2)
#     date_and_time = date_and_time[0]
#     dt = ','.join(date_and_time)    

#     fmt = '%Y-%m-%d,%H:%M:%S'
#     dt = datetime.datetime.strptime(dt, fmt)
#     return dt
    
# #**********************************************************************************************************************************************    
# def extract_tz_from_tz_aware_dt_str(timeperiod, 
#                                     pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})'):
#     # e.g., timeperiod = '2021-10-12T00:15:00-04:00'
#     date_and_time_w_tz = re.findall(pattern, timeperiod)
#     assert(len(date_and_time_w_tz)==1)
#     assert(len(date_and_time_w_tz[0])==3)
#     tz = date_and_time_w_tz[0][2]
#     return tz    

# #**********************************************************************************************************************************************    
# def extract_tz_parts_from_tz_aware_dt_str(timeperiod, 
#                                           pattern_full=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})', 
#                                           pattern_tz=r'([+-])(\d{2}):(\d{2})'):
#     # Get sign (+-), hour, and minute of timezone
#     tz_str = extract_tz_from_tz_aware_dt_str(timeperiod, pattern=pattern_full)
#     pm_h_m = re.findall(pattern_tz, tz_str)
#     assert(len(pm_h_m)==1)
#     assert(len(pm_h_m[0])==3)
#     sign = pm_h_m[0][0]
#     hour = int(pm_h_m[0][1])
#     minute = int(pm_h_m[0][2])
#     return {'sign':sign, 'hour':hour, 'minute':minute}
# #--------------------------------------------------------------------

#**********************************************************************************************************************************************
def substring_in_all(substr, arr):
    for a in arr:
        if a.find(substr)<0:
            return False
    return True

#**********************************************************************************************************************************************
def find_longest_substring_shared_by_all(arr):
    # Designed for case where table name prepended to each column name
    # when calling e.g. pd.run_sql
    # e.g., columns = ['reading_ivl_nonvee.serialnumber', 'reading_ivl_nonvee.starttimeperiod', ...
    #                  'reading_ivl_nonvee.value']
    #
    # With this in mind, let's try to simplest case first before running into full blown method.
    #   By simplest case, I mean find the longest common string between the first each of the others,
    #   and if that common string is found in all, return it
    for i in range(1, len(arr)):
        longest_common = Utilities.find_longest_shared_substring(arr[0], arr[i])
        if substring_in_all(longest_common, arr):
            return longest_common

    # Simple method did not work, so run full blown
    str0 = arr[0]
    res = ''
    for i in range(len(str0)+1):
        for j in range(i+1, len(str0)+1):
            if substring_in_all(str0[i:j], arr) and len(str0[i:j])>len(res):
                res = str0[i:j]
    return res

#**********************************************************************************************************************************************
def remove_longest_substring_shared_by_all(arr):
    common_subst = find_longest_substring_shared_by_all(arr)
    if len(common_subst)==0:
        return arr
    return_arr = []
    for a in arr:
        idx = a.find(common_subst)
        return_arr.append(a[:idx]+a[idx+len(common_subst):])
    return return_arr

#**********************************************************************************************************************************************
def remove_longest_substring_shared_by_all_columns_in_df(df, inplace=True):
    columns_org = df.columns.tolist()
    columns_new = remove_longest_substring_shared_by_all(columns_org)
    if columns_org==columns_new:
        return df
    assert(len(columns_org)==len(columns_new))
    cols_rename_dict = dict(zip(columns_org, columns_new))
    if inplace:
        df.rename(columns=cols_rename_dict, inplace=True)
    else:
        df = df.rename(columns=cols_rename_dict, inplace=False)
    return df

#**********************************************************************************************************************************************    
def remove_prepend_from_columns_in_df(df, end_of_prepend_indicator='.', inplace=True):
    # Intended to change e.g. a column named inst_msr_consume.read_type to read_type
    # Somewhat similar to remove_longest_substring_shared_by_all_columns_in_df, but more simple minded.
    # However, this works better for case where e.g. joins have occurred and therefore not all columns share
    # the same prepdended value
    # This simply finds the first occurrence of end_of_prepend_indicator and chops off
    # end_of_prepend_indicator and preceding it.
    columns_org = df.columns.tolist()
    columns_new = []
    for col in columns_org:
        found_idx = col.find(end_of_prepend_indicator)
        if found_idx > -1:
            columns_new.append(col[found_idx+len(end_of_prepend_indicator):])
        else:
            columns_new.append(col)
    cols_rename_dict = dict(zip(columns_org, columns_new))
    if inplace:
        df.rename(columns=cols_rename_dict, inplace=True)
    else:
        df = df.rename(columns=cols_rename_dict, inplace=False)
    return df
    
def remove_table_aliases(df, end_of_prepend_indicator='.', inplace=True):
    #I kept forgetting the name of remove_table_aliases
    # So, this is simply an easier name to remember
    return remove_prepend_from_columns_in_df(df=df, end_of_prepend_indicator=end_of_prepend_indicator, inplace=inplace)



#**********************************************************************************************************************************************
#**********************************************************************************************************************************************
#**********************************************************************************************************************************************

#-----------------------------------------------------------------------------------------------------
# Methods for joining outages with meter events
#-----------------------------------------------------------------------------------------------------

#**********************************************************************************************************************************************
def match_events_in_df_to_outages(df_events, df_outages, 
                                  return_n_events_with_mult_matches=False, 
                                  events_placement_cols=None, events_cols=None, outages_cols=None, 
                                  lowercase_cols_default=False, verbose=True):
    # TODO what to do if an end event is close to two outages?
    #TODO solution to above found using column of lists together with explode method
    # For now, keep OUTG_REC_NB_OLD to make sure everything works alright    
    #-------------------------
    events_placement_cols_default = {'outg_rec_nb':'OUTG_REC_NB', 
                                     'outg_rec_nb_all':'OUTG_REC_NB_ALL', 
                                     'outg_rec_nb_OLD':'OUTG_REC_NB_OLD', 
                                     'n_outages_found':'n_outages_found'}

    events_cols_default = {'premise_nb':'aep_premise_nb', 
                           'event_time':'valuesinterval_local'}

    outages_cols_default = {'outg_rec_nb':'OUTG_REC_NB', 
                            'premise_nb':'PREMISE_NB', 
                            't_search_min':'t_search_min', 
                            't_search_max':'t_search_max'}
    #-----                        
    if lowercase_cols_default:
        events_placement_cols_default = {k:v.lower() for k,v in events_placement_cols_default.items()}
        events_cols_default = {k:v.lower() for k,v in events_cols_default.items()}
        outages_cols_default = {k:v.lower() for k,v in outages_cols_default.items()}    
    #-----
    if events_placement_cols is None: 
        events_placement_cols = events_placement_cols_default
    if events_cols is None:
        events_cols = events_cols_default
    if outages_cols is None:
        outages_cols = outages_cols_default
    #-----
    assert(events_placement_cols.keys()==events_placement_cols_default.keys())
    assert(events_cols.keys()==events_cols_default.keys())
    assert(outages_cols.keys()==outages_cols_default.keys())
    #-------------------------
    df_events[events_placement_cols['outg_rec_nb_OLD']] = None
    df_events[events_placement_cols['outg_rec_nb']] = None
    df_events[events_placement_cols['outg_rec_nb_all']] = np.empty((len(df_events), 0)).tolist()
    df_events[events_placement_cols['n_outages_found']] = 0
    #-------------------------
    for outg_rec_nb, sub_df_outage in df_outages.groupby(outages_cols['outg_rec_nb']):
        #print(outg_rec_nb)    
        assert(len(sub_df_outage[outages_cols['t_search_min']].unique())==1)
        assert(len(sub_df_outage[outages_cols['t_search_max']].unique())==1)

        premises_i = sub_df_outage[outages_cols['premise_nb']]
        t_beg = sub_df_outage.iloc[0][outages_cols['t_search_min']]
        t_end = sub_df_outage.iloc[0][outages_cols['t_search_max']]

        # NOTE: both beginning and ending times are INCLUSIVE to match what is 
        #       done in the SQL query (i.e., using BETWEEN t_beg AND t_end)
        bool_mask = (df_events[events_cols['premise_nb']].isin(premises_i) & 
                     (df_events[events_cols['event_time']] >= t_beg) & 
                     (df_events[events_cols['event_time']] <= t_end)).tolist()

        df_events.loc[bool_mask, events_placement_cols['outg_rec_nb_OLD']]  = int(outg_rec_nb)
        df_events.loc[bool_mask, events_placement_cols['outg_rec_nb_all']]  = df_events.loc[bool_mask, events_placement_cols['outg_rec_nb_all']].apply(lambda x: x+[int(outg_rec_nb)])
        df_events.loc[bool_mask, events_placement_cols['n_outages_found']] += 1
    #-------------------------
    # Make sure n_events_with_mult_matches agrees with the number of rows having len('OUTG_REC_NB_ALL')>1
    n_events_with_mult_matches = df_events[df_events[events_placement_cols['n_outages_found']] > 1].shape[0]
    assert(n_events_with_mult_matches==sum(df_events[events_placement_cols['outg_rec_nb_all']].apply(len)>1))
    #-----
    # Copy outg_rec_nb from outg_rec_nb_all and explode to handle cases where
    # more than one outage found for a given event
    df_events[events_placement_cols['outg_rec_nb']] = df_events[events_placement_cols['outg_rec_nb_all']]
    n_rows_before_explode = df_events.shape[0]
    df_events = df_events.explode(events_placement_cols['outg_rec_nb'])
    n_rows_after_explode = df_events.shape[0]
    assert(n_events_with_mult_matches==n_rows_after_explode-n_rows_before_explode)
    if verbose:
        print(f"Number of events matched with multiple outages: {n_events_with_mult_matches}")
        print(f'BEFORE EXPLODE: n_rows = {n_rows_before_explode}')
        print(f'AFTER EXPLODE: n_rows = {n_rows_after_explode}')
        print(f'delta = {n_rows_after_explode-n_rows_before_explode}')  
    if return_n_events_with_mult_matches:
        return df_events, n_events_with_mult_matches
    else:
        return df_events

#**********************************************************************************************************************************************
# This only exists because it is easier to understand than set_all_outages_info_in_events_df and can be
# used if I am seeing strange results.  I also want to test the performance difference between the two
def set_all_outages_info_in_events_df_original(df_events, df_outage, 
                                               outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                               outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                               cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'], 
                                               placement_cols = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                                              ):
    assert(len(cols_to_return_from_df_outage)==len(placement_cols))
    for col in placement_cols:
        df_events[col] = None
    for out_rec_nb, df in df_events.groupby(outg_rec_nb_col_in_df_events):
        #print(out_rec_nb)
        sub_df_outage = df_outage[df_outage[outg_rec_nb_col_in_df_outage]==out_rec_nb]

        for col in cols_to_return_from_df_outage:
            assert(len(sub_df_outage[col].unique())==1)

        bool_mask = df_events[outg_rec_nb_col_in_df_events]==out_rec_nb
        for i in range(len(cols_to_return_from_df_outage)):
            df_events.loc[bool_mask, placement_cols[i]]   = sub_df_outage.iloc[0][cols_to_return_from_df_outage[i]]

    return df_events

#**********************************************************************************************************************************************
def get_outage_info(outg_rec_nb, df_outage, 
                    outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                    cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                   ):
    if pd.isnull(outg_rec_nb) or outg_rec_nb not in df_outage[outg_rec_nb_col_in_df_outage].unique():
        return [np.nan for _ in range(len(cols_to_return_from_df_outage))]
    sub_df_outage = df_outage[df_outage[outg_rec_nb_col_in_df_outage]==outg_rec_nb]
    for col in cols_to_return_from_df_outage:
        assert(len(sub_df_outage[col].unique())==1)
    return [sub_df_outage.iloc[0][x] for x in cols_to_return_from_df_outage]

#**********************************************************************************************************************************************
def get_outage_info_for_events_df_w_single_outage(df_events, df_outage, 
                                                  outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                                  outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                                  cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                                                 ):
    # Not intended to be called by user, but rather is for set_all_outages_info_in_events_df.
    # This can only be used when df_events contains only a single outage.
    assert(len(df_events[outg_rec_nb_col_in_df_events].unique())==1)
    outg_rec_nb = df_events.iloc[0][outg_rec_nb_col_in_df_events]
    return get_outage_info(outg_rec_nb, df_outage, outg_rec_nb_col_in_df_outage, cols_to_return_from_df_outage)

#**********************************************************************************************************************************************
def set_outage_info_in_events_df_w_single_outage(df_events, df_outage, 
                                                 outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                                 outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                                 cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'], 
                                                 placement_cols = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window']
                                                ):
    # Not intended to be called by user, but rather is for set_all_outages_info_in_events_df.
    # This can only be used when df_events contains only a single outage.
    assert(len(cols_to_return_from_df_outage)==len(placement_cols))
    to_be_placed = get_outage_info_for_events_df_w_single_outage(df_events, df_outage, 
                                                                 outg_rec_nb_col_in_df_events, 
                                                                 outg_rec_nb_col_in_df_outage, 
                                                                 cols_to_return_from_df_outage)
    assert(len(placement_cols)==len(to_be_placed))
    for i,col in enumerate(placement_cols):
        df_events[col] = to_be_placed[i]
    return df_events

#**********************************************************************************************************************************************
#TODO can I use transform somehow instead of apply?
# Apparently transform is supposed to be faster
def set_all_outages_info_in_events_df(df_events, df_outage, 
                                      outg_rec_nb_col_in_df_events='OUTG_REC_NB', 
                                      outg_rec_nb_col_in_df_outage='OUTG_REC_NB', 
                                      cols_to_return_from_df_outage = ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'], 
                                      placement_cols =                ['DT_OFF_TS_FULL', 'DT_ON_TS', 'STEP_DRTN_NB', 'search_time_half_window'],
                                      dropna=False, 
                                      lowercase_cols_default=False
                                     ):
    # NOTE: default behaviour of groupby is to exclude NaN values
    # HOWEVER, prior to pandas version 1.3.4 (fix might have occured sooner, but 1.3.4 for sure)
    #   NaN values were ALWAYS dropped when using .apply with .groupby.
    #   So, even if dropna=False, when using pd.__version__ < 1.3.4, the expected behaviour would not be achieved
    # This code will work regardless of pandas version, as I have implemented a workaround for the case when pd.__version__ < 1.3.4
    #   The workaround involves setting all outg_rec_nb_col_in_df_events=NaN --> -1, performing groupby and apply operation
    #   and then setting back to NaN
    #--------------------------
    if lowercase_cols_default:
        outg_rec_nb_col_in_df_events = outg_rec_nb_col_in_df_events.lower()
        outg_rec_nb_col_in_df_outage = outg_rec_nb_col_in_df_outage.lower()
        cols_to_return_from_df_outage = [x.lower() for x in cols_to_return_from_df_outage]
        placement_cols = [x.lower() for x in placement_cols]
    #--------------------------
    if version.parse(pd.__version__) < version.parse('1.3.4') and dropna==False:
        df_events.loc[pd.isnull(df_events[outg_rec_nb_col_in_df_events]), outg_rec_nb_col_in_df_events] = -1
    df_events = df_events.groupby(outg_rec_nb_col_in_df_events, dropna=dropna).apply(
        lambda x: set_outage_info_in_events_df_w_single_outage(df_events=x,  
                                                               df_outage=df_outage, 
                                                               outg_rec_nb_col_in_df_events=outg_rec_nb_col_in_df_events, 
                                                               outg_rec_nb_col_in_df_outage=outg_rec_nb_col_in_df_outage, 
                                                               cols_to_return_from_df_outage=cols_to_return_from_df_outage, 
                                                               placement_cols=placement_cols))
    if version.parse(pd.__version__) < version.parse('1.3.4') and dropna==False:
        df_events.loc[df_events[outg_rec_nb_col_in_df_events]==-1, outg_rec_nb_col_in_df_events] = np.nan
    return df_events
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()