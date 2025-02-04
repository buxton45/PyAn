#!/usr/bin/env python


r"""
Utilities specifically for date/time operations

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

"""
__author__ = "Jesse Buxton"
__email__  = "buxton.45.jb@gmail.com"
__status__ = "Personal"

#--------------------------------------------------------------------

#---------------------------------------------------------------------
import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
from pandas.testing import assert_frame_equal
from scipy import stats
from natsort import natsort_keygen

import datetime
#import time
import random

import Utilities
import Utilities_df
#---------------------------------------------------------------------
#**********************************************************************************************************************************************
def calc_dt_mean(
    dt_list
):
    r"""
    Return the mean of the list of datetime objects, dt_list
    -----
    Calling np.mean on a list of Datetime objects fails with: 
        TypeError: unsupported operand type(s) for +: 'Timestamp' and 'Timestamp'
    This seems to be the simplest method, as there is already a pandas.DatetimeIndex.mean method in place.
    Therefore, this simply converts the dt_list to a pandas.DatetimeIndex object and calls mean on it
    -----
    If this method does not perform as expected, one could develop of method by:
        1. Finding the minimum value in the list, subtracting the minimum from all values in dt_list,
           find the mean of the list of differences (as calculating the mean of Timedelta objects is fine)
           and then adding that mean back to the minimum value
        2. Convert to float values using .timestamp(), calculate the mean, and convert back via datetime.fromtimestamp
           However, one should be careful here as weird clandar effects due to lapse seconds, days, etc can result.
    """
    #-------------------------
    dt_idx_obj = pd.DatetimeIndex(dt_list)
    mean_val = dt_idx_obj.mean()
    return mean_val

#**********************************************************************************************************************************************
def append_random_time_to_date(date, rand_seed=None):
    random.seed(rand_seed)
    time = random.random() * datetime.timedelta(days=1)
    # Strip off time if present in date (resetting to 00:00:00)
    date = pd.to_datetime(date.date())
    # Add date to time, and keep only up to seconds
    dt = (date+time).replace(microsecond=0)
    return dt

def get_random_date_between(date_0, date_1, rand_seed=None):
    r"""
    Generate a random date between date_0 and date_1
    """
    random.seed(rand_seed)
    #-------------------------
    if not is_datetime64_dtype(date_0):
        date_0 = pd.to_datetime(date_0)
    if not is_datetime64_dtype(date_1):
        date_1 = pd.to_datetime(date_1)
    #-------------------------
    selection_range_days = (date_1 - date_0).days
    if selection_range_days==0:
        return pd.to_datetime(date_0.date())
    random_offset = random.randrange(selection_range_days)
    random_date = pd.to_datetime(date_0.date()) + datetime.timedelta(days=random_offset)
    #-------------------------
    return random_date
    
def get_random_datetime_interval_between(date_0, date_1, window_width, rand_seed=None):
    r"""
    Generate a random datetime interval of width=window_width between date_0 and date_1
    """
    #-------------------------
    if not is_datetime64_dtype(date_0):
        date_0 = pd.to_datetime(date_0)
    if not is_datetime64_dtype(date_1):
        date_1 = pd.to_datetime(date_1)
    assert(Utilities.is_object_one_of_types(window_width, [datetime.timedelta, pd.Timedelta]))
    #-------------------------
    # Since the window must be window_width wide, the endpoint for start_date selection
    # is date_1 - window_width
    start_date = get_random_date_between(
        date_0=date_0, 
        date_1=(date_1-window_width), 
        rand_seed=rand_seed
    )
    start_date = append_random_time_to_date(date=start_date, rand_seed=rand_seed)
    end_date = start_date + window_width
    #-------------------------
    return [start_date, end_date]

def get_random_date_interval_between(date_0, date_1, window_width, rand_seed=None):
    r"""
    Generate a random date interval of width=window_width between date_0 and date_1
    """
    #-------------------------
    dt_interval = get_random_datetime_interval_between(
        date_0=date_0, 
        date_1=date_1, 
        window_width=window_width, 
        rand_seed=rand_seed
    )
    return [pd.to_datetime(dt_interval[0].date()), pd.to_datetime(dt_interval[1].date())]


#**********************************************************************************************************************************************
def get_utc_datetime_from_timestamp(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp)
    
#**********************************************************************************************************************************************    
def get_timedelta_from_timezoneoffset_OLD(timezoneoffset, 
                                      pattern=r'([+-]{0,1})(\d{2}):(\d{2})', 
                                      expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2}):
    r"""
    OLD VERSION.  Only kept temporarily in case the new version has issues.  New version implemented 28 Jan 2022.  DELETE ME BY 1 March 2022
    Returns datetime.timedelta object given a timezoneoffset string (e.g., '-04:00') down to microseconds if desired.
    
    expected_match_dict is used to interpret the returned tuple of elements.
      The value of a given key corresponds to its index location in the returned match results tuple
      len(expected_match_dict) is the number of matched items to be returned when pattern is matched
      The matches to be returned are specified by being enclosed in ()
    
    Below, re.findall(pattern, timezoneoffset) should find only one instance of pattern
      with len(expected_match_dict) elements
    
    For the typical case with the default arguments above:
      e.g., if timezoneoffset=='-04:00' --> re.findall(pattern, timezoneoffset) = [('-', '04', '00')]
      e.g., if timezoneoffset== '04:00' --> re.findall(pattern, timezoneoffset) = [('', '04', '00')]
    
    As another example, consider the following:
      timezoneoffset = '12T00:15:00-04:00'
      pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}'
      expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}
      --> re.findall(pattern, timezoneoffset) = [('12', '00', '15', '00')]
    
    --------------------------------------------------------------
    EXAMPLE 1
    Simplest case and how this will likely be used 99% of the time
    >>> get_timedelta_from_timezoneoffset('-05:00')
    datetime.timedelta(days=-1, seconds=68400)
    
    --------------------------------------------------------------
    EXAMPLE 2
    If timezoneoffset from the previous example also included seconds and fractions of a second, 
    but the pattern (and expected_match_dict) remain the same as in the previous example.
    -----
    >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
    ...                                   pattern=r'([+-]{0,1})(\d{2}):(\d{2})', 
    ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2})
    datetime.timedelta(days=-1, seconds=68400)
    
    --------------------------------------------------------------
    EXAMPLE 3
    To extract the full value from EXAMPLE 2, one would need to adjust pattern and expected_match_dict
    -----
    >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
    ...                                   pattern=r'([+-]{0,1})(\d{2}):(\d{2}):(\d{2}).(\d{3})(\d{3})', 
    ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2, 
    ...                                                        'seconds':3, 'milliseconds':4, 'microseconds':5})
    datetime.timedelta(days=-1, seconds=68340, microseconds=876544)
    
    --------------------------------------------------------------
    EXAMPLES 4 and 5
    This function can also simply ignore portions of timezoneoffset if desired.
    Imagine, e.g., timezoneoffset = '12T00:15:00-04:00', but one wants to ignore the tz
    information (i.e., ignore -04:00) and convert the datetime to a timedelta.
    In such a case, one could use:
      expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}   
      pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}' OR, EVEN BETTER
      pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*'
    >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
    ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}', 
    ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
    datetime.timedelta(days=12, seconds=900)
    
    >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
    ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*', 
    ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
    datetime.timedelta(days=12, seconds=900)
    """
    #!!!!!!!!!!!!!!!!!!! DELETE ME BY 1 March 2022 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    time_delta = re.findall(pattern, timezoneoffset)
    assert(len(time_delta)==1)
    if len(time_delta)>1:
        time_delta = time_delta[0]
    assert(len(time_delta)==len(expected_match_dict))

    # ----- Unpack time_delta using expected_match_dict -----
    # -- First, get indices if exist in expected_match_dict
    # Most common
    multiplier_idx   = expected_match_dict.get('multiplier', -1)
    hours_idx        = expected_match_dict.get('hours', -1)
    minutes_idx      = expected_match_dict.get('minutes', -1)
    # Less common
    days_idx         = expected_match_dict.get('days', -1)
    weeks_idx        = expected_match_dict.get('weeks', -1)
    seconds_idx      = expected_match_dict.get('seconds', -1)
    milliseconds_idx = expected_match_dict.get('milliseconds', -1)
    microseconds_idx = expected_match_dict.get('microseconds', -1)

    # -- Use the indices found above to grab the correct entries from time_delta
    #    If an index==-1, set it to default value ('' for multiplier_idx, 0 for all else)
    #    NOTE: The order DOES matter below.  If time_delta[idx] is called first, it will
    #    cause the program to crash when idx<0.  If it is called second, everything is fine
    multiplier   = '' if multiplier_idx<0   else time_delta[multiplier_idx]
    hours        = 0 if hours_idx<0        else time_delta[hours_idx]
    minutes      = 0 if minutes_idx<0      else time_delta[minutes_idx]
    # ---
    days         = 0 if days_idx<0         else time_delta[days_idx]
    weeks        = 0 if weeks_idx<0        else time_delta[weeks_idx]
    seconds      = 0 if seconds_idx<0      else time_delta[seconds_idx]
    milliseconds = 0 if milliseconds_idx<0 else time_delta[milliseconds_idx]
    microseconds = 0 if microseconds_idx<0 else time_delta[microseconds_idx]
    # --------------- Done unpacking ------------------------

    # multiplier should either be: 
    #    if timezoneoffset is positive --> multiplier = '' (i.e., an empty string)
    #    if timezoneoffset is negative --> multiplier = '-'
    #    if multiplier not included in search (nor in expected_match_dict)) --> multiplier=''
    assert(len(multiplier)==0 or multiplier=='-')
    multiplier = 1 if len(multiplier)==0 else -1

    time_delta = datetime.timedelta(hours=multiplier*float(hours), minutes=multiplier*float(minutes), 
                                    days=multiplier*float(days), weeks=multiplier*float(weeks), 
                                    seconds=multiplier*float(seconds), milliseconds=multiplier*float(milliseconds), 
                                    microseconds=multiplier*float(microseconds))
    return time_delta
    
#**********************************************************************************************************************************************    
def extract_datetime_elements_from_string(datetime_str, 
                                          pattern=r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})([+-]{0,1})(\d{2}):(\d{2})', 
                                          expected_match_dict={'year':0, 'month':1, 'day':2, 
                                                               'hours':3, 'minutes':4, 'seconds':5, 
                                                               'tz_sign':6, 'tz_hours':7, 'tz_minutes':8}):
    r"""
    Returns a dict of the datetime elements defined by pattern and expected_match_dict found in datetime_str.
    (datetime_str : str, pattern : str (regex), expected_match_dict: dict(k=str, v=int)) -> dict(k=str, v=str)
    The default values are intended for a typical AEP datetime format of e.g., datetime_str = '2021-10-12T00:15:00-04:00'
    
    expected_match_dict is used to interpret the returned tuple of elements.
      The value of a given key corresponds to its index location in the returned match results tuple
      len(expected_match_dict) is the number of matched items to be returned when pattern is matched
      The matches to be returned are specified by being enclosed in ()
    
    For a typical case of datetime_str = '2021-10-12T00:15:00-04:00' with the default pattern and 
    expected_match_dict arguments, the function should return:
    return_dict = {'year': '2021', 'month': '10', 'day': '12', 
                   'hours': '00', 'minutes': '15', 'seconds': '00', 
                   'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
    For the typical case with the default arguments above:
      e.g., if timezoneoffset=='-04:00' --> re.findall(pattern, timezoneoffset) = [('-', '04', '00')]
      e.g., if timezoneoffset== '04:00' --> re.findall(pattern, timezoneoffset) = [('', '04', '00')]
    
    As another example, consider the following:
      timezoneoffset = '12T00:15:00-04:00'
      pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}'
      expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}
      --> re.findall(pattern, timezoneoffset) = [('12', '00', '15', '00')]
    
    --------------------------------------------------------------
    EXAMPLE 1
      datetime_str = '2021-10-12T00:15:00-04:00'
      default pattern and expected_match_dict
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00')
    {'year': '2021', 'month': '10', 'day': '12', 'hours': '00', 'minutes': '15', 'seconds': '00', 'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
    --------------------------------------------------------------
    EXAMPLES 2 and 3
    Same datetime_str as EXAMPLE 1, but suppose one wants the entire date, entire time, and entire timezone information
      datetime_str = '2021-10-12T00:15:00-04:00'
      expected_match_dict = {'date':0, 'time':1, 'timezone':2}
      pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]{0,1}\d{2}:\d{2})' <-- careful
      pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(.*)'                   <-- careless
    -----
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
    ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]{0,1}\d{2}:\d{2})', 
    ...                                       expected_match_dict={'date':0, 'time':1, 'timezone':2})
    {'date': '2021-10-12', 'time': '00:15:00', 'timezone': '-04:00'}
    
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
    ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(.*)', 
    ...                                       expected_match_dict={'date':0, 'time':1, 'timezone':2})
    {'date': '2021-10-12', 'time': '00:15:00', 'timezone': '-04:00'}
    
    --------------------------------------------------------------
    EXAMPLE 4
    Same as EXAMPLES 2 and 3, but suppose one does not care about the timezone information
      datetime_str = '2021-10-12T00:15:00-04:00'
      expected_match_dict = {'date':0, 'time':1}
      pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})'
    -----
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
    ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})', 
    ...                                       expected_match_dict={'date':0, 'time':1})
    {'date': '2021-10-12', 'time': '00:15:00'}
    
    --------------------------------------------------------------
    EXAMPLES 5, 6, 7
    Suppose one ONLY wants to timezone information
      EXAMPLE 5
        datetime_str = '2021-10-12T00:15:00-04:00'
        expected_match_dict = {'tz_sign':0, 'tz_hours':1, 'tz_minutes':2}
        pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1})(\d{2}):(\d{2})'

      EXAMPLE 6  
        !!!!! CAREFUL !!!!! 
          Using pattern=r'([+-]{0,1})(\d{2}):(\d{2})' would cause found_datetime_elements = re.findall(pattern, datetime_str) 
          to return [('', '00', '15'), ('-', '04', '00')] which would cause assert(len(found_datetime_elements)==1) to fail!
        !!!!! HOWEVER !!!!! 
          if one is certain the timezone will begin with + or - (and not simple be blank for the case of +),
          one could use pattern=r'([+-]{1})(\d{2}):(\d{2})'
        -----
        datetime_str = '2021-10-12T00:15:00-04:00'
        expected_match_dict = {'tz_sign':0, 'tz_hours':1, 'tz_minutes':2}
        pattern=r'([+-]{1})(\d{2}):(\d{2})'
        
      EXAMPLE 7
        datetime_str = '2021-10-12T00:15:00-04:00'
        expected_match_dict = {'timezone':0}
        pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1}\d{2}:\d{2})' (or pattern=r'([+-]{1}\d{2}:\d{2})')
    -----
    EX 5
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
    ...                                       pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1})(\d{2}):(\d{2})', 
    ...                                       expected_match_dict={'tz_sign':0, 'tz_hours':1, 'tz_minutes':2})
    {'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
    EX 6
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
    ...                                       pattern=r'([+-]{1})(\d{2}):(\d{2})', 
    ...                                       expected_match_dict={'tz_sign':0, 'tz_hours':1, 'tz_minutes':2})
    {'tz_sign': '-', 'tz_hours': '04', 'tz_minutes': '00'}
    
    EX 7
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00-04:00', 
    ...                                       pattern=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]{0,1}\d{2}:\d{2})', 
    ...                                       expected_match_dict={'timezone':0})
    {'timezone': '-04:00'}
    
    --------------------------------------------------------------
    EXAMPLE 8
    The purpose here is to show how to include whitespace using \s (see [\s+-]{0,1}\d{2}:\d{2} in pattern below)
    Suppose the datetime_str does not have + or - between the time and timezone, but instead just has a space
      datetime_str = '2021-10-12T00:15:00 04:00'
      expected_match_dict = {'date':0, 'time':1, 'timezone':2}
      pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([\s+-]{0,1}\d{2}:\d{2})'
    -----
    >>> extract_datetime_elements_from_string(datetime_str='2021-10-12T00:15:00 04:00', 
    ...                                       pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([\s+-]{0,1}\d{2}:\d{2})', 
    ...                                       expected_match_dict={'date':0, 'time':1, 'timezone':2})
    {'date': '2021-10-12', 'time': '00:15:00', 'timezone': ' 04:00'}
 
    """
    #-------------------------------------------------------------------------------------------
    # Below, re.findall(pattern, datetime_str) should find only one instance of overall pattern
    #   with len(expected_match_dict) elements
    found_datetime_elements = re.findall(pattern, datetime_str)
    assert(len(found_datetime_elements)==1) #only intended to find one instance of overall pattern
    if len(expected_match_dict)>1:
        found_datetime_elements = found_datetime_elements[0]
    assert(len(found_datetime_elements)==len(expected_match_dict))
    #------------------
    return_dict = {}
    for element, idx in expected_match_dict.items():
        assert(element not in return_dict)
        return_dict[element] = found_datetime_elements[idx]
    return return_dict
    
#**********************************************************************************************************************************************    
def get_timedelta_from_timezoneoffset(timezoneoffset, 
                                      pattern=r'([\s+-]{0,1})(\d{2}):(\d{2})', 
                                      expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2}):
    r"""
    Returns datetime.timedelta object given a timezoneoffset string (e.g., '-04:00') down to microseconds if desired.
    (timezoneoffset : str, pattern : str (regex), expected_match_dict: dict(k=str, v=int)) -> datetime.timedelta
    
    NOTE: expected_match_dict.keys() must all be found in 
      expected_elements = ['multiplier', 'hours', 'minutes', 
                           'days', 'weeks', 'seconds', 'milliseconds', 'microseconds']
    
    expected_match_dict is used to interpret the returned tuple of elements.
      The value of a given key corresponds to its index location in the returned match results tuple
      len(expected_match_dict) is the number of matched items to be returned when pattern is matched
      The matches to be returned are specified by being enclosed in ()
    
    Below, re.findall(pattern, timezoneoffset) should find only one instance of pattern
      with len(expected_match_dict) elements
    
    For the typical case with the default arguments above:
      e.g., if timezoneoffset=='-04:00' --> re.findall(pattern, timezoneoffset) = [('-', '04', '00')]
      e.g., if timezoneoffset== '04:00' --> re.findall(pattern, timezoneoffset) = [('', '04', '00')]
    
    As another example, consider the following:
      timezoneoffset = '12T00:15:00-04:00'
      pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}'
      expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}
      --> re.findall(pattern, timezoneoffset) = [('12', '00', '15', '00')]
    
    --------------------------------------------------------------
    EXAMPLE 1
    Simplest case and how this will likely be used 99% of the time
    >>> get_timedelta_from_timezoneoffset('-05:00')
    datetime.timedelta(days=-1, seconds=68400)
    
    --------------------------------------------------------------
    EXAMPLE 2
    If timezoneoffset from the previous example also included seconds and fractions of a second, 
    but the pattern (and expected_match_dict) remain the same as in the previous example.
    -----
    >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
    ...                                   pattern=r'([\s+-]{0,1})(\d{2}):(\d{2})', 
    ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2})
    datetime.timedelta(days=-1, seconds=68400)
    
    --------------------------------------------------------------
    EXAMPLE 3
    To extract the full value from EXAMPLE 2, one would need to adjust pattern and expected_match_dict
    -----
    >>> get_timedelta_from_timezoneoffset('-05:00:59.123456', 
    ...                                   pattern=r'([\s+-]{0,1})(\d{2}):(\d{2}):(\d{2}).(\d{3})(\d{3})', 
    ...                                   expected_match_dict={'multiplier':0, 'hours':1, 'minutes':2, 
    ...                                                        'seconds':3, 'milliseconds':4, 'microseconds':5})
    datetime.timedelta(days=-1, seconds=68340, microseconds=876544)
    
    --------------------------------------------------------------
    EXAMPLES 4 and 5
    This function can also simply ignore portions of timezoneoffset if desired.
    Imagine, e.g., timezoneoffset = '12T00:15:00-04:00', but one wants to ignore the tz
    information (i.e., ignore -04:00) and convert the datetime to a timedelta.
    In such a case, one could use:
      expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3}   
      pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}' OR, EVEN BETTER
      pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*'
    >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
    ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})-\d{2}:\d{2}', 
    ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
    datetime.timedelta(days=12, seconds=900)
    
    >>> get_timedelta_from_timezoneoffset('12T00:15:00-04:00', 
    ...                                   pattern = r'(\d{2})T(\d{2}):(\d{2}):(\d{2})*', 
    ...                                   expected_match_dict = {'days':0, 'hours':1, 'minutes':2, 'seconds':3})
    datetime.timedelta(days=12, seconds=900)
 
    """
    #-------------------------
    expected_elements = ['multiplier', 'hours', 'minutes', 
                         'days', 'weeks', 'seconds', 'milliseconds', 'microseconds']
    assert(all(x in expected_elements for x in expected_match_dict.keys()))
    #-------------------------
    found_match_dict = extract_datetime_elements_from_string(datetime_str=timezoneoffset, 
                                                             pattern=pattern, 
                                                             expected_match_dict=expected_match_dict)
    #-------------------------
    #    If an element is not found, set it to default value ('+' for multiplier_idx, 0 for all else)
    # Most common
    multiplier   = found_match_dict.get('multiplier', '+')
    hours        = found_match_dict.get('hours', 0)
    minutes      = found_match_dict.get('minutes', 0)
    # Less common
    days         = found_match_dict.get('days', 0)
    weeks        = found_match_dict.get('weeks', 0)
    seconds      = found_match_dict.get('seconds', 0)
    milliseconds = found_match_dict.get('milliseconds', 0)
    microseconds = found_match_dict.get('microseconds', 0)

    # multiplier should be '+', '-', '', or ' '
    #   multiplier == '-'              --> negative
    #   multiplier == '+', '', or ' ' ---> positive
    assert(multiplier in ['+', '-', '', ' '])
    multiplier = multiplier.strip()
    if not multiplier or multiplier=='+':
        multiplier = 1
    elif multiplier=='-':
        multiplier = -1
    else:
        assert(0)
    #-------------------------
    time_delta = datetime.timedelta(hours=multiplier*float(hours), minutes=multiplier*float(minutes), 
                                    days=multiplier*float(days), weeks=multiplier*float(weeks), 
                                    seconds=multiplier*float(seconds), milliseconds=multiplier*float(milliseconds), 
                                    microseconds=multiplier*float(microseconds))
    return time_delta

    
#**********************************************************************************************************************************************    
def clean_timeperiod_entry(timeperiod):
    r"""
    This captures only the datetime from timeperiod, and returns a datetime object.
    Essentially, any fractional seconds (if present) are ignored (using (?:.\d*)?), and 
      any timezone information (if present) are ignored (using (?:-\d{2}:\d{2})?)
      
    e.g., timeperiod = '2021-10-12T00:15:00-04:00'
    e.g., timeperiod = '2023-01-31T02:12:19.000-05:00'
    """
    #-------------------------
    pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:.\d*)?(?:-\d{2}:\d{2})?'
    date_and_time = re.findall(pattern, timeperiod)
    assert(len(date_and_time)==1)
    assert(len(date_and_time[0])==2)
    date_and_time = date_and_time[0]
    dt = ','.join(date_and_time)    

    fmt = '%Y-%m-%d,%H:%M:%S'
    dt = datetime.datetime.strptime(dt, fmt)
    return dt
    
#**********************************************************************************************************************************************    
def extract_tz_from_tz_aware_dt_str(timeperiod, 
                                    pattern=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})'):
    # e.g., timeperiod = '2021-10-12T00:15:00-04:00'
    date_and_time_w_tz = re.findall(pattern, timeperiod)
    assert(len(date_and_time_w_tz)==1)
    assert(len(date_and_time_w_tz[0])==3)
    tz = date_and_time_w_tz[0][2]
    return tz    

#**********************************************************************************************************************************************    
def extract_tz_parts_from_tz_aware_dt_str(timeperiod, 
                                          pattern_full=r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})', 
                                          pattern_tz=r'([+-])(\d{2}):(\d{2})'):
    # Get sign (+-), hour, and minute of timezone
    tz_str = extract_tz_from_tz_aware_dt_str(timeperiod, pattern=pattern_full)
    pm_h_m = re.findall(pattern_tz, tz_str)
    assert(len(pm_h_m)==1)
    assert(len(pm_h_m[0])==3)
    sign = pm_h_m[0][0]
    hour = int(pm_h_m[0][1])
    minute = int(pm_h_m[0][2])
    return {'sign':sign, 'hour':hour, 'minute':minute}
#--------------------------------------------------------------------

#**********************************************************************************************************************************************    
def convert_timestamp_to_utc_in_df(df, timestamp_col, placement_col=None, inplace=False):
    if not inplace:
        df = df.copy()
    if placement_col is None:
        placement_col = f'{timestamp_col}_from_timestamp'
    df[placement_col] = df[timestamp_col].apply(datetime.datetime.utcfromtimestamp)
    return df

def build_utc_time_column(df, time_col, placement_col=None, naive=True, inplace=False):
    # If naive=True, the timezone information is dropped from each entrty
    # If naive=False, the timezone information is kept, but will always be 
    #   equal to +00:00 as all returned times will be in UTC
    #
    # The datetime in time_col should be timezone aware! (e.g., 2020-06-28 14:45:00-04:00)
    #
    # NOTE: time_col can be a single column or a list of columns
    # NOTE: If time_col is None, the df is simply returned
    #-----------------------------------
    if time_col is None:
        return df
    #-----------------------------------
    if not inplace:
        df = df.copy()    
    #-----------------------------------
    assert(isinstance(time_col, str) or isinstance(time_col, list))
    if isinstance(time_col, list):
        # Note: inplace already taken care of above, so don't need to waste memory
        #       copying df for each iteration, which is why inplace=True below
        assert((isinstance(placement_col, list) and len(placement_col)==len(time_col)) 
               or placement_col is None)
        for i,col in enumerate(time_col):
            df = build_utc_time_column(df, time_col=col, 
                                       placement_col=None if placement_col is None else placement_col[i], 
                                       naive=naive, inplace=True)
        return df
    #-----------------------------------
    # All entries should be timezone aware.
    # However, calling pd.to_datetime(df[time_col]).dt.tz will only work if all entries
    #   have the same timezone offset (otherwise, it will crash)
    # To be most careful, one would therefore need to check each element individually.
    # I'm going to favor speed here, and only check the first element
    if pd.to_datetime(df.iloc[0][time_col]).tz is None:
        print(f"In Utilities_dt.build_utc_time_column, time_col={time_col} must contain timezone aware datetimes!")
    assert(pd.to_datetime(df.iloc[0][time_col]).tz is not None)
    #----------------------------------- 
    if placement_col is None:
        placement_col = f'{time_col}_utc'
    # NOTE: dt.tz_localize(None) drops the timezone information, creating timezone naive entries
    #       Calling simply .tz_localize(None) doesn't seem to do anything
    df[placement_col] = pd.to_datetime(df[time_col], utc=True)
    if naive:
        df[placement_col] = df[placement_col].dt.tz_localize(None)
    return df

def convert_timezoneoffset_col_to_timedelta(df, timezoneoffset_col, inplace=False):
    if not inplace:
        df = df.copy()
    if not is_timedelta64_dtype(df[timezoneoffset_col]):
        df[timezoneoffset_col] = df[timezoneoffset_col].apply(lambda x: get_timedelta_from_timezoneoffset(x))    
    return df

def strip_tz_info_and_convert_to_dt(
    df, 
    time_col, 
    placement_col=None, 
    run_quick=True, 
    n_strip=6, 
    inplace=True
):
    r"""
    Entries in time_col should be strings of format e.g., '2020-01-01T00:00:00-05:00'
    If run_quick==True, but the quick operation was unsuccessful, the slow operation is run instead
      (see more info in NOTE below)
    NOTE: After updating the pandas version, run_quick is not as reliable as it previously was!
          Previously, the method (which, as described below, simply strips off characters and then runs
            pd.to_datetime) used to handle cases where the data were of slightly different format
          e.g., it could handle when one row had a value '2021-10-12T00:15:00-04:00' and another
            had a value '2023-01-31T02:12:19.000-05:00'
          It appears pd.to_datetime no longer likes this, as I receive the following error:
              ValueError: unconverted data remains when parsing with format "%Y-%m-%dT%H:%M:%S": ".000", 
              at position 7.
          Therefore, the method has been changed such that if run_quick==True, the quick method is run within
            a try/except block, with the except method falling back to the slow method
    
    run_quick:
        If False, Utilities_dt.clean_timeperiod_entry is used for conversion
        If True, n_strip characters are stripped off the end of the time string, and then
          pd.to_datetime is run.
          
    n_strip is the number of elements to strip off the back of the time
    For, e.g., '2020-01-01T00:00:00-05:00', n_strip=6 as len('-05:00')==6
    
    NOTE: time_col can be a single column or a list of columns
    """
    #-----------------------------------
    if not inplace:
        df = df.copy()    
    #-----------------------------------
    assert(isinstance(time_col, str) or isinstance(time_col, list))
    if isinstance(time_col, list):
        # Note: inplace already taken care of above, so don't need to waste memory
        #       copying df for each iteration, which is why inplace=True below
        assert((isinstance(placement_col, list) and len(placement_col)==len(time_col)) 
               or placement_col is None)
        for i,col in enumerate(time_col):
            df = strip_tz_info_and_convert_to_dt(
                df, 
                time_col=col, 
                placement_col=None if placement_col is None else placement_col[i], 
                run_quick=run_quick, 
                n_strip=n_strip, 
                inplace=True
            )
        return df
    #-----------------------------------  
    if placement_col is None:
        placement_col = time_col
    if run_quick:
        try:
            df[placement_col] = df[time_col].str[:-n_strip]
            df[placement_col] = pd.to_datetime(df[placement_col])
        except:
            print('Quick method in Utilities_dt.strip_tz_info_and_convert_to_dt failed, using slow method instead!')
            df[placement_col] = df[time_col].apply(clean_timeperiod_entry)
            df[placement_col] = pd.to_datetime(df[placement_col])
    else:
        df[placement_col] = df[time_col].apply(clean_timeperiod_entry)
        df[placement_col] = pd.to_datetime(df[placement_col])
    return df
    
    
def determine_us_timezone(shifts, assert_found=False):
    # shifts should either be a single (negative) integer or a list/tuple of two (negative) integers
    #   Note: A list/tuple of one (negative) integer will work too
    # If only a single shift is given, this function can at best return two possible timezones
    utc_shifts = {
        'US/Eastern':  [-5, -4], 
        'US/Central':  [-6, -5], 
        'US/Mountain': [-7, -6], 
        'US/Pacific':  [-8, -7], 
        'US/Alaska':   [-9, -8], 
        'US/Hawaii':   [-10, -10]
    }
    #-------------------------------
    found_tz = None
    if isinstance(shifts,int) or len(shifts)==1:
        if not isinstance(shifts,int):
            shifts = shifts[0]
        found_tz = [tz for tz,tz_shifts in utc_shifts.items() if shifts>=tz_shifts[0] and shifts<=tz_shifts[1]]
    else:
        assert(len(shifts)==2)
        shifts=sorted(shifts)
        found_tz = [tz for tz,tz_shifts in utc_shifts.items() if shifts==tz_shifts]
    #-------------------------------
    if len(found_tz)==0:
        found_tz=None
    if found_tz is None:
        if assert_found:
            assert(0)
        else:
            return found_tz
    #-----
    if len(found_tz)==1:
        found_tz=found_tz[0]
    return found_tz
    
    
def convert_local_to_utc_time(t_local, timezone):
    # timezone should be e.g., 'US/Eastern'
    #-----------------------------------
    if isinstance(t_local, list) or isinstance(t_local, tuple):
        return_list = []
        for t in t_local:
            return_list.append(convert_local_to_utc_time(t, timezone))
        return return_list
    #-----------------------------------    
    t_utc = pd.to_datetime(t_local).tz_localize(timezone).tz_convert(None)
    return t_utc

def determine_timezone_and_convert_local_to_utc_time(t_local, unique_tz_offsets, 
                                                     timezone_aware_times=None, **kwargs):
    # Should be complete hour 
    #  (there do exist half-hour and 45-minute time zones in t
    #   the world, but there shouldn't be any in AEP data)
    #
    # As convert_local_to_utc_time works whether t_local is single time or a list
    # of times, this function does as well.
    #--------------------
    # Preferable to use unique_tz_offsets, 
    #   which is a list of timezone offsets
    #     can be strings (e.g., '-05:00'), datetime.timedeltas, or pd./np.Timedeltas
    # HOWEVER, can set unique_tz_offsets to None and use timezone_aware_times instead
    #--------------------
    if unique_tz_offsets is None:
        assert(timezone_aware_times is not None)
        dflt_pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})([+-]\d{2}:\d{2})'
        pattern = kwargs.get('pattern', dflt_pattern)
        unique_tz_offsets = [extract_tz_from_tz_aware_dt_str(x, pattern=pattern) for x in timezone_aware_times]
        unique_tz_offsets = list(set(unique_tz_offsets))        
    #--------------------
    # Make sure elements in unique_tz_offsets are of proper type, and all are unique
    unq_tz_offsets = []
    for x in unique_tz_offsets:
        if is_timedelta64_dtype(x):
            unq_tz_offsets.append(x)
        elif isinstance(x, datetime.timedelta):
            unq_tz_offsets.append(pd.to_timedelta(x))
        else:
            assert(isinstance(x, str))
            unq_tz_offsets.append(pd.to_timedelta(get_timedelta_from_timezoneoffset(x)))
    # Make sure unq_tz_offsets truly is unique
    unq_tz_offsets = list(set(unq_tz_offsets))    
    #--------------------
    assert(all(x.total_seconds()%3600==0 for x in unq_tz_offsets))
    unq_tz_offsets = [round(x.total_seconds()/3600) for x in unq_tz_offsets]
    found_tz = determine_us_timezone(unq_tz_offsets, assert_found=True)
    #--------------------
    return convert_local_to_utc_time(t_local, found_tz)


def get_date_ranges(
    date_0            , 
    date_1            , 
    freq              , 
    include_endpoints = True
):
    r"""
    Essentially takes the output of pd.date_range(date_0, date_1, freq), which is a list of dates,
      and turns it into a list of ranges (i.e., length-2 lists/tuples)

    freq:
        Must be at least 1D in duration (e.g, W, MS, 31D, etc. all work)
    
    NOTE: Any time information in date_0/date_1 IS IGNORED!
          Only the date is used
    """
    #--------------------------------------------------
    # Make sure freq is at least 1D in duration
    dummy_dt = pd.to_datetime('1987-11-02')
    assert(dummy_dt+pd.tseries.frequencies.to_offset(freq) > dummy_dt+pd.tseries.frequencies.to_offset('1D'))
    #-------------------------
    date_0 = pd.to_datetime(date_0).date()
    date_1 = pd.to_datetime(date_1).date()
    #-------------------------
    dates = pd.date_range(
        start = date_0, 
        end   = date_1, 
        freq  = freq
    )
    #-------------------------
    # For the intervals, i.e. the two-element objects, formed by [element_i, element_i+1 - 1Day]
    dates = [
        [dates[i].date(), (dates[i+1]-pd.Timedelta('1D')).date()] 
        for i in range(len(dates)-1)
    ]

    #-------------------------
    # If, e.g., MS is used, then date_0 generally won't be included
    if include_endpoints and dates[0][0]!=date_0:
        dates.insert(0, [date_0, dates[0][0]-pd.Timedelta('1D')])
        assert(dates[0][0]==date_0)
    
    #-------------------------
    # In general, date_1 won't be included (unless freq perfectly splits the [date_0,date_1] interval)
    if include_endpoints and dates[-1][-1]!=date_1:
        dates.append([dates[-1][-1]+pd.Timedelta('1D'), date_1])
        assert(dates[-1][-1]==date_1)

    #-------------------------
    # Make elements tuples, instead of lists (since tuples are immutable)
    dates = [tuple(i) for i in dates]
    
    #-------------------------
    # SANITY CHECK: Make sure the end of each range differs from the beginning of the next by 1 day
    for i in range(len(dates)):
        if i==0:
            continue
        range_i   = dates[i]
        range_im1 = dates[i-1]
        assert(range_i[0]-range_im1[1] == pd.Timedelta('1D'))

    #-------------------------
    return dates
    
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()