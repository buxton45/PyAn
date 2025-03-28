#!/usr/bin/env python

r"""
Holds AMIEDE_DEV class.  See AMIEDE_DEV.AMIEDE_DEV for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys
import re

import pandas as pd

#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
#--------------------------------------------------


class AMIEDE_DEV:
    r"""
    """
    
    def __init__(
        self
    ):
        self.cpo_df = pd.DataFrame()
        
    #****************************************************************************************************
    @staticmethod        
    def extract_last_gasp_reason_components(reason):
        timestamp_pattern = r'('\
                            r'(?:\d{4}-\d{2}-\d{2}T.\d*-?)'\
                            r'|'\
                            r'(?:[0-9a-zA-Z]{3,9} \d{1,2}, \d{4} -\s*(?:\d{2}:\d{2}:\d{2})?\s*[A|P]M [0-9a-zA-Z]{3})'\
                            r'|'\
                            r'(?:\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}-\d{2}:\d{2})'\
                            r')'
        #-------------------------
        pattern = r'(Last Gasp\s*-\s*[0-9a-zA-Z\s]*)[\s\:,.]*'\
                  r'(?:(?:[0-9a-zA-Z]{1,2})(?:\:[0-9a-zA-Z]{1,2})+)?,?\s*'\
                  r'Reboot Count: (\d*),?\s*'
        pattern += r'NIC timestamp\: {}[,\s]*'.format(timestamp_pattern)
        pattern += r'Received timestamp\: {}[,\s]*'.format(timestamp_pattern)
        pattern += r'(Fail Reason: .*)$'
        #-------------------------
        matches = re.findall(pattern, reason)
        if len(matches)==0:
            return dict()
        assert(len(matches)==1)
        #-----
        matches=matches[0]
        assert(len(matches)==5)
        #-------------------------
        return_dict = {
            'general_reason':     matches[0], 
            'reboot_count':       matches[1], 
            'nic_timestamp':      matches[2], 
            'received_timestamp': matches[3], 
            'fail_reason':        matches[4]
        }
        #-------------------------
        return return_dict

    #****************************************************************************************************
    @staticmethod
    def last_gasp_reduce_func_2(reason, include_fail_reason=True):
        reason_components_dict = AMIEDE_DEV.extract_last_gasp_reason_components(reason=reason)
        if len(reason_components_dict)==0:
            return reason
        #-------------------------
        if include_fail_reason:
            assert('fail_reason' in reason_components_dict)
            assert('general_reason' in reason_components_dict)
            return f"{reason_components_dict['general_reason']}, {reason_components_dict['fail_reason']}"
        else:
            assert('general_reason' in reason_components_dict)
            return reason_components_dict['general_reason']