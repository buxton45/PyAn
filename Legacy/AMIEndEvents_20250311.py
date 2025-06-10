#!/usr/bin/env python

r"""
Holds AMIEndEvents class.  See AMIEndEvents.AMIEndEvents for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re
from string import punctuation
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
import warnings
#--------------------------------------------------
from AMIEndEvents_SQL import AMIEndEvents_SQL
from GenAn import GenAn
from AMINonVee import AMINonVee
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
#--------------------------------------------------

class AMIEndEvents(GenAn):
    r"""
    class AMIEndEvents documentation.
    See MECPOAn class if dealing with multiple Reason Counts Per Outage (RCPO) and/or Id (enddeviceeventtypeid) Counts Per Outage (ICPO) DFs
    """
    def __init__(
        self, 
        df_construct_type         = DFConstructType.kRunSqlQuery, 
        contstruct_df_args        = None, 
        init_df_in_constructor    = True, 
        build_sql_function        = None, 
        build_sql_function_kwargs = None, 
        save_args                 = False, 
        **kwargs
    ):
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
                                   else AMIEndEvents_SQL.build_sql_end_events)
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
            save_args=save_args, 
            **kwargs
        )
        #--------------------------------------------------
    
    # NOTE:  Below, std_SNs_cols and std_nSNs_cols will likely be eliminated once all RCPO methods are moved over to MECPODf
    @staticmethod
    def std_SNs_cols():
        std_cols = ['_SNs', '_prem_nbs', '_outg_SNs', '_outg_prem_nbs', '_prim_SNs', '_xfmr_SNs', '_xfmr_PNs']
        return std_cols
        
    @staticmethod
    def std_nSNs_cols():
        std_cols = ['_nSNs', '_nprem_nbs', '_outg_nSNs', '_outg_nprem_nbs', '_prim_nSNs', '_xfmr_nSNs', '_xfmr_nPNs']
        return std_cols


        
    #****************************************************************************************************
    def get_conn_db(self):
        return Utilities.get_athena_prod_aws_connection()
    
    def get_default_cols_and_types_to_convert_dict(self):
        cols_and_types_to_convert_dict = None
        return cols_and_types_to_convert_dict
    
    def get_full_default_sort_by(self):
        full_default_sort_by = ['issuertracking_id', 'aep_event_dt', 'valuesinterval', 'serialnumber']
        return full_default_sort_by
        
    #****************************************************************************************************
    @staticmethod
    def get_end_event_distinct_fields(
        date_0                           , 
        date_1                           , 
        fields                           = ['serialnumber'], 
        are_datetime                     = False, 
        addtnl_build_sql_function_kwargs = {}, 
        cols_and_types_to_convert_dict   = None, 
        to_numeric_errors                = 'coerce', 
        save_args                        = False, 
        return_sql                       = False, 
        **kwargs
    ):
        conn_db = Utilities.get_athena_prod_aws_connection()
        build_sql_function = AMIEndEvents_SQL.build_sql_end_event_distinct_fields
        build_sql_function_kwargs = dict(
            date_0       = date_0, 
            date_1       = date_1, 
            fields       = fields, 
            are_datetime = are_datetime, 
            **addtnl_build_sql_function_kwargs
        )
        df = GenAn.build_df_general(
            conn_db                        = conn_db, 
            build_sql_function             = build_sql_function, 
            build_sql_function_kwargs      = build_sql_function_kwargs, 
            cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors              = to_numeric_errors, 
            save_args                      = save_args, 
            return_sql                     = return_sql, 
            **kwargs
        )
        return df    
    
    #****************************************************************************************************
    @staticmethod
    def extract_reboot_counts_from_reason(reason_str, pattern=r'Reboot Count: (\d*)', flags=re.IGNORECASE):
        # NOTE: Seems to be present only for Last Gasp reasons
        rbt_cnt = re.findall(pattern, reason_str, flags=flags)
        if rbt_cnt:
            assert(len(rbt_cnt)==1)
            rbt_cnt = rbt_cnt[0]
        else:
            rbt_cnt=''
        return rbt_cnt

    @staticmethod
    def extract_reboot_counts_from_reasons_in_df(
        df, 
        reason_col='reason', 
        placement_col='reboot_counts', 
        pattern=r'Reboot Count: (\d*)', 
        flags=re.IGNORECASE, 
        inplace=True
    ):
        if not inplace:
            df = df.copy()
        df[placement_col] = df[reason_col].apply(
            lambda x: AMIEndEvents.extract_reboot_counts_from_reason(
                reason_str=x, 
                pattern=pattern, 
                flags=flags
            )
        )
        return df
        
    @staticmethod
    def extract_fail_reason_from_reason(reason_str, pattern=r'Fail Reason: (.*)$', flags=re.IGNORECASE):
        # NOTE: Fail reason appears to be last item in string, hence the use of $
        #       Also, seems to be present only for Last Gasp reasons
        fail_reason = re.findall(pattern, reason_str, flags=flags)
        if fail_reason:
            assert(len(fail_reason)==1)
            fail_reason = fail_reason[0]
        else:
            fail_reason=''
        return fail_reason

    @staticmethod
    def extract_fail_reasons_from_reasons_in_df(
        df, 
        reason_col='reason', 
        placement_col='fail_reason', 
        pattern=r'Fail Reason: (.*)$', 
        flags=re.IGNORECASE, 
        inplace=True
    ):
        if not inplace:
            df = df.copy()
        df[placement_col] = df[reason_col].apply(
            lambda x: AMIEndEvents.extract_fail_reason_from_reason(
                reason_str=x, 
                pattern=pattern, 
                flags=flags
            )
        )
        return df
 
        
    @staticmethod
    def reduce_end_event_reason(
        reason, 
        patterns, 
        count=0, 
        #flags=re.IGNORECASE, 
        flags=0, 
        verbose=True
    ):
        r"""
        Searches for pattern in reason, if found, replaced in string.
          NOTE: Each pattern can be a single string, or a pair of strings.
          If a single string:
            the string represents the pattern and if found it is replaced by ''
          If a pair of strings:
            the first element is the pattern and the second element is the replacement
            NOTE: The replacement can be a string, a pattern, or a function!
        Once regex is matched, the function performs the substitution and exists.
            i.e., multiple regexs are not applied!
            This is in contrast the the OLD version
        """
        for item in patterns:
            if Utilities.is_object_one_of_types(item, [list, tuple]):
                assert(len(item)==2)
                pattern = item[0]
                replace = item[1]
            else:
                assert(isinstance(item, str))
                pattern = item
                replace = ''
            if re.search(pattern=pattern, string=reason, flags=flags):
                reason = Utilities.find_and_replace_in_string(strng=reason, pattern=pattern, replace=replace, count=count, flags=flags)
                return reason 
        if verbose:
            print(f'WARNING: reason="{reason}" not matched!')
        return reason
        
    @staticmethod
    def remove_trailing_punctuation_from_reason(
        reason, 
        include_normal_strip=True, 
        include_closing_brackets=True
    ):
        r"""
        NOTE: Below, punctuation from string library.
              punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
              
        As include_closing_brackets==True by default, the default punctuation to be removed from
        the end is: !"#$%&'(*+,-./:;<=>?@[\^_`{|~
              
        If include_closing_brackets==True, )]} will not be removed.
        NOTE: rstrip expects punct_to_remove to be a string, hence the need for
              translate with ord
              translate: replaces each character in the string using the given mapping table or dictionary.
              ord: function returns an integer representing the Unicode character.
        """
        #-------------------------
        if include_normal_strip:
            reason = reason.strip()
        #-------------------------
        punct_to_remove = punctuation
        if include_closing_brackets:
            punct_to_remove = punct_to_remove.translate({ord(i): None for i in ')]}'})
        #-----
        # Add ' ' to punct_to_remove to handle case where space separates characters to be removed,
        #   e.g., 'Under Voltage (Diagnostic 6) occurred for meter: .'
        punct_to_remove += ' '
        #-------------------------
        reason = reason.rstrip(punct_to_remove)
        return reason
        
        
    @staticmethod
    def under_voltage_match_func(match_obj):
        r"""
        The under voltage match pattern is: 
            r'(Under Voltage)\s*'\
            r'([0-9a-zA-Z]*)?\s*'\
            r'(\([0-9a-zA-Z\s]*\))\s*'\
            r'([0-9a-zA-Z]*)?\s?'\
            r'(for meter\:?\s*)'\
            r'(?:(?:[0-9a-zA-Z]{1,2})(?:\:[0-9a-zA-Z]{1,2})+)?[\s:,.]*'\
            r'(?:Phase\s{1,2}[ABC](?:(?:\s*and\s*[ABC])|(?:,\s*[ABC])*))?\s*'\
            r'(Voltage out of tolerance)?'
        This function basically normalizes things such that the code always follows "Under Voltage".
        This is needed as I have seen, for example, the following:
          Under Voltage cleared (CA000400) for meter XXXX.
          Under Voltage (CA000400) cleared for meter XXXX.
        For either of these, the output will be: 
          Under Voltage (CA000400) cleared for meter XXXX.
        NOTE: This assumes any MAC-esque ID has already been removed!

        NOTE: In initial version, before capturing and replacing groups, I simply used (, [ABC])*
              However, here I am using (?:, [ABC])*, as I don't want this to be captured as a separate group!
              Update: Added ?: in front of Phase because I don't want that group captured either

        The pattern to capture the code , \([0-9a-zA-Z\s]*\), includes a possible space (\s) to handle
          codes such as 'Diagnostic 6'

        """
        # When grabbing from match_obj.groups(), index from 0
        # When using match_obj.group(idx), index from 1 as index 0 is full match
        matches = match_obj.groups()
        #-------------------------
        uv_code_idx = 2    
        # Idea here is that descriptive word (i.e., cleared, occurred, etc.) should be directly before or after
        # uv_code (e.g., (CA000400)), or not present at all, but certainly not both before and after
        assert(uv_code_idx>0) # Simply because I'll be looking for descriptive word right before and right after
        assert(not(matches[uv_code_idx-1] and matches[uv_code_idx+1]))
        str_middle = matches[uv_code_idx]
        if matches[uv_code_idx-1]:
            str_middle += f' {matches[uv_code_idx-1]}'
        elif matches[uv_code_idx+1]:
            str_middle += f' {matches[uv_code_idx+1]}'
        else:
            str_middle += ''
        #-------------------------
        str_front = ''
        for i in range(uv_code_idx-1):
            if matches[i] and not re.search('Phase', matches[i], flags=re.IGNORECASE):
                str_front += f' {matches[i]}'
        #-------------------------
        str_end = ''
        for i in range(uv_code_idx+2, len(matches)):
            if matches[i] and not re.search('Phase', matches[i], flags=re.IGNORECASE):
                str_end += f' {matches[i]}'
        #-------------------------
        str_front  = str_front.strip()
        str_middle = str_middle.strip()
        str_end    = str_end.strip()
        #-------------------------
        return_str = f'{str_front} {str_middle} {str_end}'
        return return_str
        
    @staticmethod
    def last_gasp_reduce_func(match_obj, include_fail_reason=True):
        r"""
        """
        # When grabbing from match_obj.groups(), index from 0
        # When using match_obj.group(idx), index from 1 as index 0 is full match
        matches = match_obj.groups()
        #-------------------------
        if include_fail_reason:
            return f'{matches[0]}, {matches[-1]}'
        else:
            return matches[0]
        
        
    @staticmethod
    def get_std_patterns_to_replace_by_typeid_EXPLICIT():
        r"""
        Returns the standard patterns_to_replace_by_typeid.
        Function exists essentially to keep reduce_end_event_reasons_in_df from being too cluttered

        NOTE: Multi-line strings have more parentheses than absolutely necessary for clarity.
        NOTE: In (r'\s{2,}', ' '), the second element ' ' has a space between the quotes! So, multiple white-space characters
                are to be replaced with a single-space.
              This is different from many other cases, where ther is no space between the quotes, and in which case the match
                is to be removed (by replacing it with an empty character, e.g., (r'\s+([0-9a-zA-Z]+)(\:[0-9a-zA-Z]+)+', ''))  
        """
        #-------------------------
        patterns_to_replace_by_typeid = {
            #-------------------------
            # Examples
            # 'Power Loss cleared for meter 00:13:50:05:ff:0d:dd:17'
            '3.2.0.28':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],  

            #-------------------------
            # Examples
            # 'Power loss detected on meter 00:13:50:05:ff:0d:dd:17'
            '3.2.0.85':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Low Battery cleared for meter 00:13:50:02:00:0a:9e:1e.'
            '3.2.22.28':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Low Battery (C1219 Table 3) occurred for meter 00:13:50:02:00:0a:9e:1e.'
            '3.2.22.150':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Test Mode Started occurred for meter 00:13:50:05:ff:1d:2d:4a.'
            # 'Meter event Test Mode Started  Time event occurred on meter = 01/20/2020 12:56:50  Sequence number = 56  User id = 1  Event argument = 00-00'
            # SEEMS RARE, BUT HAVE SEEN:
            # 'Meter is currently in test mode.'
            '3.7.19.242':    [
                #-----
                (r'(Test Mode Started)(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', r'\1'), 
                #-----
                (r'(Meter event Test Mode Started)\s*Time event occurred on meter = .* Sequence number = .* User id = .*  Event argument = .*', r'\1'), 
                #-----
                (r'(Meter event Test Mode Started).*', 'Test Mode Started'), 
                #-----
                (r'(Meter is currently in test mode)[\s\.]*', r'\1')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Test Mode Stopped occurred for meter 00:13:50:03:ff:06:20:96.'
            # 'Meter event Test Mode Stopped  Time event occurred on meter = 01/02/2020 08:25:04  Sequence number = 573  User id = 1  Event argument = 00-00'
            '3.7.19.243':    [
                #-----
                (r'(Test Mode Stopped)(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', r'\1'), 
                #-----
                (r'(Meter event Test Mode Stopped)\s*Time event occurred on meter = .* Sequence number = .* User id = .*  Event argument = .*', r'\1'), 
                #-----
                (r'(Meter event Test Mode Stopped).*', 'Test Mode Stopped')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'NIC operating on backup battery: 00:13:50:ff:fe:70:96:ac, Reboot Count: 6, NIC timestamp: 2021-01-14T10:17:54.000-05:00, Received timestamp: 2021-01-14T10:17:55.728-05:00'
            '3.7.22.4':    [
                #-----
                (r'(NIC operating on backup battery)\:(\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?), Reboot Count: .*, NIC timestamp: .*, Received timestamp: .*', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples:
            # All seem to be: 'NIC backup battery inactive'
            # So, don't need to run any regex, but should still so no flags raised
            '3.7.22.19':    [
                #-----
                (r'(NIC backup battery inactive).*', r'\1')
                #-----
            ],      

            #-------------------------
            # Examples
            # 'Demand Reset occurred for meter 00:13:50:05:ff:18:b3:c1.'
            '3.8.0.215':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter Program Seal mismatch for Device [Device ID, MAC Id] = [1ND785727916NMD06, 00:13:50:05:ff:1d:16:96] - program seal = 0bda0569c8e6ad375375eaa177034d3f61a80478 vs. UIQ program seal = e2dbe1ccbeef1f2df38adffeeafdb86fbd8fb006 at 2021-01-08T06:16:49.116-05:00.'
            '3.9.83.159':    [
                #-----
                (r'(Meter Program Seal mismatch for Device)\s*\[Device ID, MAC Id\]\s*=\s*.*', r'\1')
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'Requested operation JOB_OP_TYPE_ARB_METER_COMMAND could not be applied to the given device type and firmware version. Device: 00:13:50:05:ff:2d:ff:34, DeviceType: DID_SUBTYPE_I210CRD_HAN, Firmware Version: 3.12.5c'
            '3.11.63.161':  [
                #-----
                # Below, Device sometimes blank, sometime MAC-esque
                (r'(Requested operation)\s*(.*?)\s*(could not be applied) to the given device type and firmware version.\s*Device\:?(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)?,\s*DeviceType\:?.*,\s*Firmware Version\:?.*', r'\1 \3: \2')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Tamper Attempt Suspected  Time event occurred on meter = 01/10/2020 10:03:37  Sequence number = 1189  User id = 0  Event argument = 00-00'
            # 'Meter detected a Tamper Attempt.'
            # 'Tamper attempt detected.'
            '3.12.0.257':   [
                #-----
                (r'(Meter event Tamper Attempt Suspected)\s*Time event occurred on meter = .* Sequence number = .* User id = .*  Event argument = .*', r'\1'), 
                #-----
                (r'(Meter event Tamper Attempt Suspected).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
                (r'(Meter detected a Tamper Attempt|Tamper attempt detected).*', 'Meter detected a Tamper Attempt'), 
                #-----
            ],    

            #-------------------------
            # Examples:
            # 'Tamper (Meter Inversion) detected on meter 00:13:50:05:ff:27:3f:a0.'
            '3.12.17.257':  [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)\.?', '') 
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'Diag1 Condition cleared for meter 00:13:50:05:ff:04:55:a1.'
            # SEEMS RARE, BUT HAVE SEEN:
            # Reverse energy cleared for meter 00:13:50:05:ff:0d:dd:17.
            '3.12.48.28':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'Diag1: Polarity, Cross Phase, Reverse Energy Flow occurred for meter 00:13:50:03:00:34:b7:e9. Angle out of tolerance [Voltage - Phase  B].'
            # 'KV2c meter event Polarity, Cross Phase, Reverse Energy Flow Diagnostic flags:Phase A Current'
            # 'Service disconnect operation occurred with status: Close operation started'
            # 'Reverse energy (C1219 Table 3) occurred for meter 00:13:50:05:ff:0d:dd:17.'
            '3.12.48.219':  [
                #-----
                (r'(Diag1: Polarity, Cross Phase, Reverse Energy Flow)(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)\s*(Angle out of tolerance) \[.*\]', r'\1: \2'), 
                #-----
                (r'(KV2c meter event Polarity, Cross Phase, Reverse Energy Flow) Diagnostic flags:.*', r'\1'), 
                #-----
                (r'(Service disconnect operation occurred with status:) (Open|Close) operation (started|succeeded)\s*', r'\1 \2 \3'), 
                #-----
                (r'((?:Diag1 Condition|Reverse energy)\s*(?:\(.*\))?)\s*(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', r'\1'), 
                #-----
            ],    

            #-------------------------
            # Examples
            # 'Cleared: Meter00:13:50:05:ff:07:88:4b, cleared tamper detection (C1219 Table 3)'
            '3.12.93.28': [
                #-----
                (r'(Cleared: Meter)(?:\s*(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?), (cleared tamper detection.*)', r'\1 \2'), 
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'Meter event Reverse Rotation Detected  Time event occurred on meter = 01/28/2020 15:31:34  Sequence number = 854  User id = 0  Event argument = 00-00'
            # 'Meter detected a Reverse Rotation.'
            # 'Meter 00:13:50:05:ff:0b:ae:84, detected tampering (C1219 Table 3)'
            # SEEMS RARE, BUT HAVE SEEN:
            # Meter event Demand Reset Occurred  Time event occurred on meter = 10/25/2020 10:07:33  Sequence number = 1256  User id = 1  Event argument = 00-00
            '3.12.93.219':    [
                #-----
                (r'(Meter)(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\.,]?)\s*(detected tampering\s*(?:\(.*\))?)\s*', r'\1 \2'), 
                #-----
                (r'(Meter event (?:Reverse Rotation Detected|Demand Reset Occurred))\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'),
                #-----
                (r'(Meter event Reverse Rotation Detected).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
                (r'(Meter detected a Reverse Rotation).*', 'Meter event Reverse Rotation Detected'), 
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'A NET_MGMT command was sent from fdc9:ccbe:52c0:cbc0:250:56ff:feb5:91c7 with a key that has insufficient privileges to execute it. ID: 53 READ SUBID: 65535 ASSOC_ID: 8448'
            # 'A NET_MGMT command was sent from fdc9:ccbe:52c0:cbc0:a6ba:99ff:fe12:1b0e with a key that has insufficient privileges to execute it. ID: 207 WRITE SUBID: 65535 ASSOC_ID: 768'
            '3.12.136.38':  [
                #-----
                # BELOW, ID seemed meter-specific (or, at least, not general), as READ/WRITE SUBID and ASSOC_ID seem to only take on a few different values
                # Therefore, group 3, for ID, is not included in output
                (r'A?\s+?(.*command was sent) from(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)\s*(with a key that has insufficient privileges) to execute it.*(ID:\s*[0-9]+)\s*((?:READ|WRITE)\s*SUBID:\s*[0-9]+)\s*(ASSOC_ID:\s*[0-9]+).*', r'\1 \2: \4 \5'), 
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'NET_MGMT command failed consecutively for 1 times for fd37:ec90:20c2:5c58:250:56ff:feb5:1010. WRITE'
            # 'Secure association operation failed consecutively for 1 times for fdc9:ccbe:52c0:d3a0:250:56ff:feb5:ec3c. 16270'
            # 'failed consecutively for 1 times for 0:0:0:0:0:0:0:0.'
            # 'N/A failed consecutively for 1 times for 0:0:0:0:0:0:0:0. N/A'
            '3.12.136.85':  [
                #-----
                (r'(.*) failed consecutively for [0-9]+ times?(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*', r'\1 failed consecutively for 1 or more times'), 
                #-----
                (r'^failed consecutively for [0-9]+ times? for 0:0:0:0:0:0:0:0.$', 'N/A failed consecutively for 1 or more times'), 
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'Meter detected a RAM failure.'
            # So, don't need to run any regex, but should still so no flags raised
            '3.18.1.199':    [
                #-----
                (r'(Meter detected a RAM failure).*', r'\1')
                #-----
            ],         

            #-------------------------
            # Examples:
            # All seem to be: 'Meter detected a ROM failure.'
            # So, don't need to run any regex, but should still so no flags raised
            '3.18.1.220':    [
                #-----
                (r'(Meter detected a ROM failure).*', r'\1')
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'NVRAM Error cleared for meter 00:13:50:05:ff:0d:7e:64.'
            '3.18.72.28':   [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'NVRAM Error (C1219 Table 3) occurred for meter 00:13:50:05:ff:20:91:6c.'
            '3.18.72.79':   [
                #-----
                (r'(NVRAM Error\s*(?:\(.*\))?)\s*(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'Meter event Nonvolatile Memory Failure Detected  Time event occurred on meter = 01/21/2020 09:45:01  Sequence number = 3335  User id = 0  Event argument = 00-00'
            # 'null'????????????
            # SEEMS RARE, BUT HAVE SEEN:
            # 'Meter event Demand Reset Occurred  Time event occurred on meter = 01/29/2023  Sequence number = 711  User id = 1  Event argument = 00-00 '
            # 'Meter event Reset List Pointers  Time event occurred on meter = 09/07/2021  Sequence number = 227  User id = 0  Event argument = 03-00'
            # 'KV2c meter event Received kWh Caution Diagnostic flags:Phase A Voltage'
            # 'Universal event Unsupported Event(41) with priority Alarm. '
            '3.18.72.85':   [
                #-----
    #             (r'(Meter event (?:Nonvolatile Memory Failure Detected|Demand Reset Occurred|Reset List Pointers))\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'),
                (r'(Meter event.*?)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                # FAILSAFE in case Sequence number etc. not found above
                # NOTE: Cannot do (Meter event.*).*, as that will reduce everything altered above donw to simply 'Meter event'
                #       Also, if any additional info (e.g, Sequence number) is to be extracted using above, this must be removed
                (r'(Meter event (?:Nonvolatile Memory Failure Detected|Demand Reset Occurred|Reset List Pointers)).*', r'\1'),
                #-----
                (r'(Meter detected a nonvolatile memory failure).*', 'Meter event Nonvolatile Memory Failure Detected'), 
                #-----
                (r'(KV2c meter event .*?)\s*(Diagnostic flags)\s*\:\s*(.*?)\s*$', r'\1 (\2 = \3)'),
                #-----
                (r'(Universal event Unsupported Event\s*(?:\(.*\))?\s*with priority Alarm)[\s*\.]*', r'\1'), 
                #-----
                ('null', 'Reason is null, ID=3.18.72.85')
                #-----
            ],         

            #-------------------------
            # Examples:
            # 'RAM Error cleared for meter 00:13:50:05:ff:0d:dd:17.'
            '3.18.85.28':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'RAM Error (C1219 Table 3) occurred for meter 00:13:50:05:ff:0d:dd:17.'
            '3.18.85.79':   [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter detected a RAM failure.'
            # 'Meter event Ram Failure Detected  Time event occurred on meter = 7/27/2021 12:45:53 PM  Sequence number = 89  User id = 34891  Event argument = 7D-75'
            '3.18.85.85':   [
                #-----
                (r'(Meter event Ram Failure Detected)\s*Time event occurred on meter\s*=.*\s*Sequence number=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Ram Failure Detected).*', r'\1'),
                #-----
                (r'(Meter detected a RAM failure).*', 'Meter event Ram Failure Detected'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'ROM Error (C1219 Table 3) occurred for meter 00:13:50:05:ff:1f:ba:d0.'
            '3.18.92.28':   [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'ROM Error (C1219 Table 3) occurred for meter 00:13:50:05:ff:1f:ba:d0.'
            '3.18.92.79':   [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Rom Failure Detected  Time event occurred on meter = 03/16/2020 07:00:51  Sequence number = 1983  User id = 1024  Event argument = 00-00'
            # 'Meter detected a ROM failure.'
            # 'ROM failure detected.'
            '3.18.92.85':    [
                #-----
                (r'(Meter event Rom Failure Detected)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Rom Failure Detected).*', r'\1'),
                #-----
                (r'^(?:Meter detected a ROM failure|ROM failure detected).*$', 'Meter event Rom Failure Detected'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Measurement Error Detected  Time event occurred on meter = 08/26/2021 08:27:25  Sequence number = 307  User id = 0  Event argument = 00-00 '
            '3.21.1.79':    [
                #-----
                (r'(Meter event Measurement Error Detected)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Measurement Error Detected).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
                (r'(Meter detected a measurement error).*', 'Meter event Measurement Error Detected')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Nonvolatile Memory Failure Detected  Time event occurred on meter = 08/30/2021 04:00:11  Sequence number = 3256  User id = 0  Event argument = 00-00 '
            '3.21.1.173':    [
                #-----
                (r'(Meter event Nonvolatile Memory Failure Detected)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Nonvolatile Memory Failure Detected).*', r'\1'),
                #-----
                (r'(Meter detected a nonvolatile memory failure).*', 'Meter event Nonvolatile Memory Failure Detected'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'System Error cleared for meter 00:13:50:05:ff:11:2d:23.'
            '3.21.3.28':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'System Error (C1219 Table 3: Er000020) occurred for meter 00:13:50:05:ff:0f:17:ba.'
            '3.21.3.79':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter detected a self check error.'
            '3.21.17.28':    [
                #-----
                (r'(Meter detected a self check error).', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter detected a self check error.'
            '3.21.18.79':    [
                #-----
                (r'(Meter detected a self check error).*', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Detected end of voltage sag on meter 00:13:50:05:ff:32:66:22 on one or several phases. Duration: 21 cycles (less than a second), Min RMS Voltage: Phase A 118.1 V, Phase B 95.4 V, Phase C 120.3 V, RMS Current (at min voltage): Phase A 1.0 A, Phase B 0.2 A, Phase C 1.2 A'
            '3.21.38.223':  [
                #-----
                (r'(Detected end of voltage sag)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*Duration:.*, Min RMS Voltage: Phase A .*, Phase B .*, Phase C .*, RMS Current \(at min voltage\): Phase A .*, Phase B .*, Phase C .*', r'\1'), 
                #-----
                # Failsafe
                (r'(Detected end of voltage sag)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Detected end of voltage swell on meter 00:13:50:05:ff:1b:49:b0 on one or several phases. Duration: 5 cycles (less than a second), Max RMS Voltage: Phase A 111.2 V, Phase B 133.2 V, Phase C 216.1 V, RMS Current (at max voltage): Phase A 2.8 A, Phase B 1.3 A, Phase C 0.8 A'
            '3.21.38.248':  [
                #-----
                (r'(Detected end of voltage swell)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*Duration: .*, Min RMS Voltage: .*, RMS Current \(at min voltage\): .*', r'\1'), 
                #-----
                # Failsafe
                (r'(Detected end of voltage swell)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*', r'\1')  
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Detected end of voltage sag on meter 00:13:50:03:ff:01:a6:36. Duration: 2 seconds, Min RMS Voltage: Phase A 227.9 V, RMS Current (at min voltage): Phase A 0.0 A'
            '3.21.43.223':  [
                #-----
                (r'(Detected end of voltage sag)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*Duration: .*, Min RMS Voltage: .*, RMS Current \(at min voltage\): .*', r'\1'), 
                #-----
                (r'(Detected end of voltage sag)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*', r'\1'), # failsafe  
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Detected end of voltage swell on meter 00:13:50:05:ff:2d:2c:25. Duration: 195 seconds, Max RMS Voltage: Phase A 265.7 V, RMS Current (at max voltage): Phase A 0.0 A'
            '3.21.43.248':  [
                #-----
                (r'(Detected end of voltage swell)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*Duration: .*, Min RMS Voltage: .*, RMS Current \(at min voltage\): .*', r'\1'), 
                #-----
                (r'(Detected end of voltage swell)(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?).*', r'\1'), # failsafe  
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Measurement Error cleared for meter 00:13:50:05:ff:07:c0:55.'
            '3.21.67.28':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Measurement Error Detected  Time event occurred on meter = 01/26/2020 19:47:00  Sequence number = 284  User id = 0  Event argument = 00-00'
            # 'Measurement Error (C1219 Table 3) occurred for meter 00:13:50:03:ff:06:d6:d5.'
            '3.21.67.79':    [
                #-----
                (r'(Measurement Error\s*(?:\(.*\))?)\s*(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', r'\1'), 
                #-----
                (r'(Meter event Measurement Error Detected)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Measurement Error Detected).*', r'\1'),
                #-----
                (r'(Meter detected a measurement error).*', 'Meter event Measurement Error Detected')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'DSP Error cleared for meter 00:13:50:05:ff:0b:87:fc.'
            '3.21.82.28':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'DSP Error (C1219 Table 3: Er200000) occurred for meter 00:13:50:05:ff:2c:f3:e2.'
            '3.21.82.79':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Test Mode Stopped  Time event occurred on meter = 08/27/2021 10:04:04  Sequence number = 23834  User id = 1  Event argument = 00-00 '
            '3.22.12.243':   [
                #-----
                (r'(Meter event Test Mode Stopped)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Test Mode Stopped).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Test Mode Started  Time event occurred on meter = 08/27/2021 07:48:19  Sequence number = 765  User id = 1  Event argument = 00-00 '
            '3.22.19.242':   [
                #-----
                (r'(Meter event Test Mode Started)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Test Mode Started).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Device 00:13:50:05:ff:2e:17:7e has been determined to have exceeded the max allowable trap threshold, 20, within a certain time limit, 3600 seconds.'
            '3.23.17.139':   [
                #-----
                (r'(Device)(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+).*(exceeded the max allowable trap threshold).*', r'\1 \2'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Access Point 00:13:50:08:ff:00:06:8e has lost connectivity with FHSS 900 MHz band.'
            '3.23.136.47':  [
                #-----
                (r'(Access Point)(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)\s*(has lost connectivity with FHSS 900 MHz band.)', r'\1 \2') 
                #-----
            ],         

            #-------------------------
            # Examples
            # Device: 00:13:50:05:ff:20:f5:ef Time: 2021-01-20T06:03:47.000-05:00 Failed Device: 00:13:50:05:ff:20:f5:ef Reason: Security public key mismatch Reboot Counter: 44 Refresh Counter: 0 Seconds since last reboot 6872803
            # 'NIC Link Layer Handshake Failed: Device: 00:13:50:05:ff:19:4e:6b, Rejected neighbor Mac ID: 00:13:50:05:ff:18:a3:52, Rejection Cause: invalid eblob signature'
            '3.23.136.85':  [
                #-----
                # Below, Device and Failed Device sometimes blank, sometime MAC-esque
                (r'Device\:?(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)? Time: .* Failed Device\:?(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)? (Reason: .*) Reboot Counter: .* Refresh Counter: .*', r'Device Failed: \1'), 
                #-----
                (r'(NIC Link Layer Handshake Failed): Device:(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+), Rejected neighbor Mac ID:(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+), (Rejection Cause: .*)', r'\1: \2')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter generated an energy polarity gyrbox call in event.'
            '3.25.17.3':   [
                #-----
                (r'(Meter generated an energy polarity gyrbox call in event).*', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Primary Power Down occurred for meter 00:13:50:05:ff:16:5f:61.'
            # 'Meter had a power outage. Unsafe power fail count = 32714'
            # 'Meter event Primary Power Down  Time event occurred on meter = 08/05/2021 16:59:27  Sequence number = 691  User id = 0  Event argument = 00-00'
            '3.26.0.47': [
                #-----
                (r'(Primary Power Down)(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', r'\1'), 
                #-----
                (r'(Meter had a power outage)\.?\s*(Unsafe power fail)\s*count\s*=\s*[0-9]+', r'\1 (\2)'), 
                #-----
                (r'^(Meter had a power outage).*$', r'\1'), #Fail proof, in case Unsafe power fail, count, etc., not found 
                #-----
                (r'(Meter event Primary Power Down)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Primary Power Down).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
            ],         

            #-------------------------
            # Examples
            # 'Primary Power Up occurred for meter 00:13:50:05:ff:22:2a:0c.'
            # 'Meter event Primary Power Up Time event occurred on meter = 08/14/2020 Sequence number = 343 User id = 0 Event argument = 00-00'
            '3.26.0.216':   [
                #-----
                (r'(Primary Power Up)(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', r'\1'), 
                #-----
                (r'Meter event (Primary Power Up)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'Meter event (Primary Power Up).*', r'\1'),  #Fail proof, in case time, sequence, etc., not found 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter had a power outage. Unsafe power fail count = 13050'
            # Meter event Primary Power Down  Time event occurred on meter = 08/31/2021 20:07:18  Sequence number = 103  User id = 0  Event argument = 00-00 
            '3.26.17.185':  [
                #-----
                (r'(Meter had a power outage).\s*(Unsafe power fail) count = [0-9]+', r'\1 (\2)'), 
                #-----
                (r'(Meter event Primary Power Down)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Primary Power Down).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Primary Power Up  Time event occurred on meter = 08/31/2021 20:07:44  Sequence number = 104  User id = 0  Event argument = 00-00 '
            '3.26.17.216':  [
                #-----
                (r'Meter event (Primary Power Up)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'Meter event (Primary Power Up).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
            ], 

            #-------------------------
            # Examples
            # 'Under Voltage cleared (CA000400) for meter 00:13:50:05:ff:0b:88:ec.'
            # 'Under Voltage (CA000400) cleared for meter 00:13:50:05:ff:15:e8:3b.'
            # 'Under Voltage (Diagnostic 6) cleared for meter 00:13:50:05:ff:3f:87:07N/A.'
            '3.26.38.37':   [
                #-----
                # NOTE: Due to annoying N/A at end of some MAC-esque IDs, I have to specify length of numerical entries
                #       as [0-9a-zA-Z]{1,2}, instead of the more general [0-9a-zA-Z]+ found elsewhere!
                (r'(Under Voltage|Low Potential|Diag6 Condition)\s*(\scleared)?\s*(\s\(.*\))?\s*(\scleared)?\s*(?:(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]{1,2})(?:\:[0-9a-zA-Z]{1,2})+(?:N/A)?\.?)', r'\1\3\2\4')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Low Potential (C1219 Table 3) occurred for meter 00:13:50:05:ff:1b:67:3b.'
            '3.26.38.47': [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Diag7 Condition cleared for meter 00:13:50:03:00:4d:41:fa.'
            # 'Over Voltage (Diagnostic 7) cleared for meter 00:13:50:05:ff:37:87:3dN/A.'
            '3.26.38.73':   [
                #-----
                # NOTE: Due to annoying N/A at end of some MAC-esque IDs, I have to specify length of numerical entries
                #       as [0-9a-zA-Z]{1,2}, instead of the more general [0-9a-zA-Z]+ found elsewhere!
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]{1,2})(?:\:[0-9a-zA-Z]{1,2})+(?:N/A)?\.?)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Diag7: Over Voltage, Element A occurred for meter 03.34.0.400:13:50:02:00:0a:d7:98.'
            # 'Over Voltage (Diagnostic 7) occurred for meter 00:13:50:05:ff:41:8a:8f: Phase A.'
            # SEEMS RARE, BUT HAVE SEEN:
            # 'KV2c meter event Over Voltage Diagnostic flags:Phase A Voltage'
            # 'KV2c meter event Under Voltage Caution Diagnostic flags:Phase B Voltage'
            # 'KV2c meter event Received kWh Caution Diagnostic flags:Phase A Voltage '
            # 'KV2c meter event Received kWh Caution Cleared '
            '3.26.38.93':   [
                #-----
                (r'(KV2c meter event (?:(?:Over|Under) Voltage)|(?:Voltage Out of Tolerance)|(?:(?:Received|Delivered) kWh))(?:\s*Caution)?\s*(Diagnostic flags)\s*\:\s*(.*?)\s*$', r'\1 (\2 = \3)'), 
                #-----
                (r'(KV2c meter event (?:(?:Over|Under) Voltage)|(?:Voltage Out of Tolerance)|(?:(?:Received|Delivered) kWh))(?:\s*Caution)?\s*Cleared\s*$', r'\1 Cleared'), 
                #-----
                (r'(.*Over Voltage.*?)(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)(.*?)\s*$', r'\1\2') 
                #-----
            ], 

            #-------------------------
            # Examples
            # 'Under Voltage (CA000400) for meter 00:13:50:05:ff:0b:88:ec. Phase  C Voltage out of tolerance.'
            # 'Under Voltage (CA000400) occurred for meter 00:13:50:05:ff:15:e8:3b: Phase A.'
            # 'KV2c meter event Under Voltage Diagnostic flags:Phase A Voltage'
            # 'Diag6: Under Voltage, Element A occurred for meter 00:13:50:05:ff:0b:88:ec.'
            '3.26.38.150':   [
                #-----
                # NOTE: Due to annoying N/A at end of some MAC-esque IDs, I have to specify length of numerical entries
                #       as [0-9a-zA-Z]{1,2}, instead of the more general [0-9a-zA-Z]+ found elsewhere!
                # NOTE: Also, the use of non-greedy (.*?) at beginning.
                #       If (.*) were used instead, results would be, e.g.
                #          'Diag6: Under Voltage, Element A occurred for meter'
                #       instead of 
                #          'Diag6: Under Voltage, Element A'
                (r'(Under Voltage\s*(?:\(.*\))?)\s*(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\.\:]?)(\s?.*?)', r'\1\2'), 
                #-----
                (r'(Diag6.*?)(?:(?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\.\:]?)(\s?.*?)', r'\1\2'), 
                #-----
                (r'(KV2c meter event .*?)\s*(Diagnostic flags)\s*\:\s*(.*?)\s*$', r'\1 (\2 = \3)')
                #-----
            ], 

            #-------------------------
            # Examples
            # 'Last Gasp - NIC power lost for device: 00:13:50:05:ff:0b:83:e4, Reboot Count: 12264, NIC timestamp: 2020-01-19T22:21:02.000-05:00, Received timestamp: 2020-01-19T22:21:04.353-05:00, Fail Reason: [0x49] LG_ZERO_X_DETECTOR ,LG_DIRECT_NOTIFICATION'
            '3.26.136.47':  [
                #-----
                (r'(Last Gasp - NIC power lost for device):(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+).*(Fail Reason: .*)$', r'\1, \2')
                #-----
            ], 

            #-------------------------
            # Examples
            # 'Device 00:13:50:05:ff:0b:8a:18, Last Gasp State: EL_EVENT_POWER_FAIL_DETECT_LG_DISABLED, Detector State: EL_EVENT_POWER_FAIL_DETECT_NIC_ZX_DISABLED, Reboot Count: 108'
            '3.26.136.66':  [
                #-----
                (r'Device(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+), (Last Gasp State: .*), (Detector State: .*), Reboot Count: \d*', r'\1, \2'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'NIC Power Restore Trap Received from device: 00:13:50:05:ff:0c:7e:93, Reboot Count: 61, NIC timestamp: 2020-01-27T09:19:56.000-05:00, Received Timestamp: 2020-01-27T09:19:57.204-05:00, Power Restoration Timestamp: 2020-01-27T09:19:42.000-05:00, State: 4,p'
            '3.26.136.216': [
                #-----
                (r'(NIC Power Restore Trap Received from device):(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+).*', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter assumed to be disconnected has reported Load side voltage indicating a potential case of tamper.'
            '3.31.1.143':   [
                #-----
                (r'(Meter assumed to be disconnected has reported Load side voltage indicating a potential case of tamper).*', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter event Reverse Rotation Detected  Time event occurred on meter = 08/30/2021 04:25:38  Sequence number = 13124  User id = 0  Event argument = 00-00 '
            '3.33.1.219':   [
                #-----
                (r'(Meter event Reverse Rotation Detected)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Reverse Rotation Detected).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
                (r'(Meter detected a Reverse Rotation).*', 'Meter event Reverse Rotation Detected'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter detected a Tamper Attempt.'
            # 'Meter event Tamper Attempt Suspected  Time event occurred on meter = 08/28/2021 08:45:05  Sequence number = 72  User id = 0  Event argument = 00-00 '
            '3.33.1.257':   [
                #-----
                (r'(Meter event Tamper Attempt Suspected)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Tamper Attempt Suspected).*', r'\1'), #Fail proof, in case time, sequence, etc., not found 
                #-----
                (r'(Meter detected a Tamper Attempt).*', 'Meter event Tamper Attempt Suspected'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Cleared: Meter 00:13:50:05:ff:11:69:bb detected a high temperature condition. (C1219 Table 3)'
            '3.35.0.28': [
                #-----
                # NOTE: Here, the MAC-esque code occurs in middle of string, not the end as is common elsewhere.
                #       This is why the pattern here excludes [\s*\.]* at the end
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+\.?)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter 00:13:50:05:ff:11:69:bb detected a high temperature condition. (C1219 Table 3)'
            # 'Meter's temperature threshold exceeded.'
            # 'Meter event AX Temp Threshold Exceeded  Time event occurred on meter = 02/19/2021 13:33:50  Sequence number = 5115  User id = 0  Event argument = 00-00 '
            # 'Meter event S4TemperatureThreshold  Time event occurred on meter = 05/25/2023 17:12:50  Sequence number = 45  User id = 0  Event argument = 00-00-00-00-00-00 '
            '3.35.0.40': [
                #-----
                (r'(Meter)(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)\s*(detected a high temperature condition)\.\s*(.*)', r'\1 \2 \3'), 
                #-----
                (r"(Meter's temperature threshold exceeded).", r'\1'), 
                #-----
                (r'(Meter event .*?)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event (?:AX Temp Threshold Exceeded|S4TemperatureThreshold)).*', r'\1'),
            ],  

            #-------------------------
            # Examples
            # 'Meter detected a clock error.'
            # 'Meter event Clock Error Detected  Time event occurred on meter = 04/18/2020 10:00:10  Sequence number = 332  User id = 0  Event argument = 00-00'
            '3.36.0.79': [
                #-----
                (r'(Meter event Clock Error Detected)\s*Time event occurred on meter\s*=.*\s*Sequence number\s*=.*\s*User id\s*=.*\s*Event argument\s*=.*', r'\1'), 
                #-----
                (r'(Meter event Clock Error Detected).*', r'\1'),
                #-----
                (r'(Meter detected a clock error).*', 'Meter event Clock Error Detected'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter detected a clock error.'
            '3.36.1.29': [
                #-----
                ('(Meter detected a clock error).', r'\1'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter 00:13:50:05:ff:2e:13:de, detected loss of time (C1219 Table 3)'
            '3.36.114.73':  [
                #-----
                (r'(Meter)(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+),\s*(detected loss of time .*)', r'\1 \2'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Ignoring Interval Read data for device 00:13:50:05:ff:1e:fa:a4 as it has time in the future 2069-05-02 18:51:00.0'
            '3.36.114.159': [
                #-----
                (r'(Ignoring (?:Interval|Register) Read data for device)(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)\s*(as it has time in the future).*', r'\1 \2'), 
                #-----
            ],         

            #-------------------------
            # Examples
            # 'Meter 00:13:50:05:ff:14:c9:61 needs explicit time sync. Drift: 14391 s, Encountered Problems:  TS_ERR_LP_BX, TS_ERR_BIG_DRIFT [0x44], Meter_Time: 06:33:41'
            '3.36.136.73':  [
                #-----
                (r'(Meter)(?:\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+)\s*(needs explicit time sync.) Drift: -?\d* s, (Encountered Problems:\s*.*), Meter_Time.*', r'\1 \2 \3'), 
                #-----
            ], 

            #-------------------------
            # Examples
            # 'Error occurred when attempting to synch meter time with NIC time for device 00:13:50:05:ff:29:e1:20'
            # So, standard beg/end patterns should suffice
            '3.36.136.79':    [
                #-----
                (r'((?:\s*occurred\s*)?(?:\s*(?:for|on)?\s*(?:meter)?)?\s+(?:[0-9a-zA-Z]+)(?:\:[0-9a-zA-Z]+)+[\s*\.]*)', '')
                #-----
            ],         

            #-------------------------
            # Examples
            # 'KV2c meter event Polarity, Cross Phase, Reverse Energy Flow Diagnostic flags:Phase B Voltage '
            # 'KV2c meter event Polarity, Cross Phase, Reverse Energy Flow Diagnostic flags:Phase B Voltage, Phase C Voltage '
            '3.38.1.139':    [
                #-----
                (r'(KV2c meter event Polarity, Cross Phase, Reverse Energy Flow) Diagnostic flags:.*', r'\1'), 
                #-----
            ]
        }
        #-------------------------
        return patterns_to_replace_by_typeid
        
        
        
    @staticmethod
    def reduce_end_event_reasons_in_df(
        df                            , 
        reason_col                    = 'reason', 
        edetypeid_col                 = 'enddeviceeventtypeid', 
        patterns_to_replace_by_typeid = None, 
        addtnl_patterns_to_replace    = None, 
        placement_col                 = None, 
        count                         = 0, 
        flags                         = re.IGNORECASE,  
        inplace                       = True
    ):
        r"""
        Searches for each of patterns (= patterns_to_replace + addtnl_patterns_to_replace) in reason, if found, replaced in string.
          NOTE: Each pattern can be a single string, or a pair of strings.
          If a single string:
            the string represents the pattern and if found it is replaced by ''
          If a pair of strings:
            the first element is the pattern and the second element is the replacement
            NOTE: The replacement can be a string, a pattern, or a function!

        e.g., with default patterns:
            i.  
              'Under Voltage (CA000400) cleared for meter 00:13:50:01:00:a7:bc:52.'
        ==>   'Under Voltage (CA000400) cleared'
            ii.  
              'Last Gasp - NIC power lost for device: 00:13:50:02:00:9c:8f:2b, Reboot Count: 194, NIC timestamp: Jan 1, 2018 - 
               02:14:19 AM EST, Received timestamp: Jan 1, 2018 - 02:14:27 AM EST, Fail Reason: NIC power fail'
        ==>   'Last Gasp - NIC power lost'
            iii.  
              'Under Voltage (CA000400) cleared for meter 00:13:50:01:00:a7:bc:52. for device 00:13:50:01:00:a7:bc:52.'
        ==>   'Under Voltage (CA000400) cleared'

        ---------------------------------------------
        Description/motivation for default patterns:
        ---------------------------------------------
        - r'\:?\s?([0-9a-zA-Z]{1,2})(\:[0-9a-zA-Z]{1,2})+'
          Matches all MAC-esque meter/device IDs
          NOTE: + used at end because want ONE OR MORE repetitions of (\:[0-9a-zA-Z]{1,2})
                * matches ZERO OR MORE 
                EXAMPLE: Assume strng = 'Demand Reset occurred for meter 00:13:50:01:00:17:c6:41.'
                  If * used: match = 'De'
                  If + used: match = ' 00:13:50:01:00:17:c6:41'
          EX:
                'Under Voltage (CA000400) cleared for meter 00:13:50:01:00:a7:bc:52.'
            ==> 'Under Voltage (CA000400) cleared for meter.'
        """
        #-------------------------
        if not inplace:
            df = df.copy()
        #-------------------------
        if patterns_to_replace_by_typeid is None:
            patterns_to_replace_by_typeid = AMIEndEvents.get_std_patterns_to_replace_by_typeid_EXPLICIT()
        #-------------------------
        if addtnl_patterns_to_replace is not None:
            assert(isinstance(addtnl_patterns_to_replace, dict))
            patterns_to_replace_by_typeid = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict=patterns_to_replace_by_typeid, 
                default_values_dict=addtnl_patterns_to_replace,
                extend_any_lists=True
            )
        #-------------------------
        if placement_col is None:
            placement_col = reason_col
        #-------------------------
        # If 'all_beg' in patterns_to_replace_by_typeid, prepend contents to all
        # If 'all_end' in patterns_to_replace_by_typeid, append contents to all
        keys_to_skip_add = ['all_beg', 'all_end']
        #-----
        if 'all_beg' in patterns_to_replace_by_typeid.keys():
            patterns_for_all = patterns_to_replace_by_typeid.pop('all_beg')
            for edetypeid, patterns_to_replace in patterns_to_replace_by_typeid.items():
                if edetypeid in keys_to_skip_add:
                    continue
                patterns_to_replace_by_typeid[edetypeid] = patterns_for_all + patterns_to_replace
        #-----
        if 'all_end' in patterns_to_replace_by_typeid.keys():
            patterns_for_all = patterns_to_replace_by_typeid.pop('all_end')
            for edetypeid, patterns_to_replace in patterns_to_replace_by_typeid.items():
                if edetypeid in keys_to_skip_add:
                    continue
                patterns_to_replace_by_typeid[edetypeid] = patterns_to_replace + patterns_for_all
        #-------------------------
        # This would be handled in reduce_end_event_reason if not here, but doing it once here
        # should save time, especially for large DFs, I believe.
        for edetypeid, patterns_to_replace in patterns_to_replace_by_typeid.items():
            for i in range(len(patterns_to_replace)):
                pattern = patterns_to_replace[i]
                if Utilities.is_object_one_of_types(pattern, [list, tuple]):
                    assert(len(pattern)==2)
                    continue
                else:
                    assert(isinstance(pattern, str))
                    patterns_to_replace_by_typeid[edetypeid][i] = (pattern, '')
        #-------------------------
        df[placement_col] = df.apply(
            lambda x: AMIEndEvents.reduce_end_event_reason(
                reason=x[reason_col], 
                patterns=patterns_to_replace_by_typeid[x[edetypeid_col]], 
                count=count, 
                flags=flags
            ), 
            axis=1
        )
        #-------------------------
        return df        
        
    
    @staticmethod
    def perform_std_initiation_and_cleaning(
        df, 
        drop_na_values=True, 
        inplace=True, 
        **kwargs
    ):
        kwargs['timestamp_col']      = kwargs.get('timestamp_col', None)
        kwargs['start_time_col']     = kwargs.get('start_time_col', 'valuesinterval')
        kwargs['start_time_utc_col'] = kwargs.get('start_time_utc_col', 'valuesinterval_utc')
        kwargs['end_time_col']       = kwargs.get('end_time_col', None)
        kwargs['timezoneoffset_col'] = kwargs.get('timezoneoffset_col', None)
        kwargs['value_cols']         = kwargs.get('value_cols', None)
        kwargs['set_index_col']      = kwargs.get('start_time_utc_col', 'valuesinterval_utc')
        #-------------------------
        df = AMINonVee.perform_std_initiation_and_cleaning(
            df=df, 
            drop_na_values=drop_na_values, 
            inplace=inplace,
            **kwargs
        )
        #-------------------------
        return df
        
        
    @staticmethod
    def enforce_end_events_i_within_interval_of_outage(
        end_events_df_i           , 
        outg_times_series         , 
        min_timedelta             , 
        max_timedelta             , 
        outg_rec_nb_col           = 'outg_rec_nb', 
        datetime_col              = 'valuesinterval_local', 
        assert_one_time_per_group = True
    ):
        r"""
        This is really intended to be used in a lambda function of a groupby operation.

        end_events_df_i should be an end events pd.DataFrame for a single outage

        outg_times_series should be a series with indices equal to the outage number and values equal
          to the time of the outage (typically 'DT_OFF_TS_FULL')
        NOTE: Initial functionality required outg_times_series to have a single level index.
              Functionality has been expanded to allow outg_times_series to have MultiIndex indices.
              If outg_times_series has MultiIndex index, then one index must align with outg_rec_nb and
                whose name (as contained in outg_times_series.index.names) must equal outg_rec_nb_col.

        min_timedelta is the minimum time BEFORE the outage to allow end events
        max_timedelta is the maximum time BEFORE the outage to allow end events
          e.g., for beteween 1 and 30 days before the outage, set:
                min_timedelta = datetime.timedelta(days=1)
                max_timedelta = datetime.timedelta(days=30)
          I suppose if you wanted to allow end events AFTER the outage you could set min_timedelta to a negative value?
            (although I HAVE NOT TESTED THIS!)
        min_ or max_timedelta can be set to None to have one end open

        assert_one_time_per_group:
            Dictates whether or not multiple time values are allowed for each 'group'.
            When running with outage data, assert_one_time_per_group should be set to True, as each outage group
              should only have a single time.
            However, when running e.g. with the new baseline method (in which the same premises from the outage sample
              are used, but events are collected from clean periods of time during which no outage occurred), a 'group'
              will oftentimes be a group of meters connected to a transformer (or, meters connected to a premise).
            In this case, the group may have multiple clean periods, in which case there will be multiple times 
              associated with the group.
            Here, one would want assert_one_time_per_group=False.
            When assert_one_time_per_group is False, as long as an event falls within min_timedelta/max_timedelta of
              ONE of the associated times it will be accepted.
        """
        #-------------------------
        if min_timedelta is None and max_timedelta is None:
            return end_events_df_i
        #-------------------------
        assert(end_events_df_i[outg_rec_nb_col].nunique()==1)
        outg_rec_nb = end_events_df_i[outg_rec_nb_col].unique().tolist()[0] 
        #-------------------------
        if outg_times_series.index.nlevels==1:
            if outg_rec_nb not in outg_times_series.index:
                print(f'In enforce_end_events_i_within_interval_of_outage, '\
                      f'outg_rec_nb={outg_rec_nb} not in outg_times_series!!!!!'\
                      f'\nCRASH IMMINENT!')
            #-------------------------
            # NOTE: Double braces below ensures outg_times_i will be a pd.Series object
            outg_times_i = outg_times_series.loc[[outg_rec_nb]]
        else:
            assert(outg_rec_nb_col in outg_times_series.index.names)
            idx_level = list(outg_times_series.index.names).index(outg_rec_nb_col)
            if outg_rec_nb not in outg_times_series.index.get_level_values(idx_level):
                print(f'In enforce_end_events_i_within_interval_of_outage, '\
                      f'outg_rec_nb={outg_rec_nb} not in outg_times_series!!!!!'\
                      f'\nCRASH IMMINENT!')
            #-------------------------
            outg_times_i = outg_times_series.loc[outg_times_series.index.get_level_values(idx_level)==outg_rec_nb]
        #-------------------------
        assert(isinstance(outg_times_i, pd.Series))
        if assert_one_time_per_group:
            assert(outg_times_i.shape[0]==1)
        #-------------------------
        # min_timedelta/max_timedelta can either be None or datetime.timedelta objects
        # Furthermore, at least one must NOT be None
        assert(min_timedelta is not None or max_timedelta is not None)
        #-----
        assert(min_timedelta is None or isinstance(min_timedelta, datetime.timedelta))
        assert(max_timedelta is None or isinstance(max_timedelta, datetime.timedelta))
        #-------------------------
        # As stated above, when assert_one_time_per_group is False, as long as an event falls within 
        # min_timedelta/max_timedelta of ONE of the associated times it will be accepted.
        # ==> Initiate truth_series to all False values, and use OR operator
        truth_series = pd.Series(index=end_events_df_i.index, data=False)
        for _,outg_time_i in outg_times_i.items():       
            if min_timedelta is None:
                truth_series_i = outg_time_i-end_events_df_i[datetime_col]  < max_timedelta
            elif max_timedelta is None:
                truth_series_i = outg_time_i-end_events_df_i[datetime_col] >= min_timedelta
            else:
                truth_series_i = (
                    (outg_time_i-end_events_df_i[datetime_col] >= min_timedelta) & 
                    (outg_time_i-end_events_df_i[datetime_col]  < max_timedelta)
                )
            #----------
            truth_series = truth_series|truth_series_i
        return end_events_df_i.loc[truth_series]
        
    @staticmethod
    def enforce_end_events_within_interval_of_outage(
        end_events_df             , 
        outg_times_series         , 
        min_timedelta             , 
        max_timedelta             , 
        outg_rec_nb_col           = 'outg_rec_nb', 
        datetime_col              = 'valuesinterval_local', 
        assert_one_time_per_group = True
    ):
        r"""    
        end_events_df should be an end events pd.DataFrame containing multiple outages (although, 
          this will work also for a single outage)

        outg_times_series should be a series with indices equal to the outage number and values equal
          to the time of the outage (typically 'DT_OFF_TS_FULL')
          outg_times_series needs to include all outages found in end_events_df, and the dtype should be the same
            e.g., if outages are strings in end_events_df, the index of outg_times_series needs to be strings

        min_timedelta is the minimum time BEFORE the outage to allow end events
        max_timedelta is the maximum time BEFORE the outage to allow end events
          e.g., for beteween 1 and 30 days before the outage, set:
                min_timedelta = datetime.timedelta(days=1)
                max_timedelta = datetime.timedelta(days=30)
          I suppose if you wanted to allow end events AFTER the outage you could set min_timedelta to a negative value?
            (although I HAVE NOT TESTED THIS!)
        min_ or max_timedelta can be set to None to have one end open
        
        assert_one_time_per_group:
            Dictates whether or not multiple time values are allowed for each 'group'.
            When running with outage data, assert_one_time_per_group should be set to True, as each outage group
              should only have a single time.
            However, when running e.g. with the new baseline method (in which the same premises from the outage sample
              are used, but events are collected from clean periods of time during which no outage occurred), a 'group'
              will oftentimes be a group of meters connected to a transformer (or, meters connected to a premise).
            In this case, the group may have multiple clean periods, in which case there will be multiple times 
              associated with the group.
            Here, one would want assert_one_time_per_group=False.
            When assert_one_time_per_group is False, as long as an event falls within min_timedelta/max_timedelta of
              ONE of the associated times it will be accepted.
        """
        #-------------------------
        if min_timedelta is None and max_timedelta is None:
            return end_events_df
        #-------------------------
        reduced_end_events_df = end_events_df.groupby(outg_rec_nb_col).apply(
            lambda x: AMIEndEvents.enforce_end_events_i_within_interval_of_outage(
                end_events_df_i           = x, 
                outg_times_series         = outg_times_series, 
                min_timedelta             = min_timedelta, 
                max_timedelta             = max_timedelta, 
                outg_rec_nb_col           = outg_rec_nb_col, 
                datetime_col              = datetime_col, 
                assert_one_time_per_group = assert_one_time_per_group
            )
        )
        #-------------------------
        # Reset the index, so reduced_end_events_df has the same form as end_events_df
        #   Otherwise, outg_rec_nb_col would be the new index
        if outg_rec_nb_col in reduced_end_events_df: #should always be true
            drop_idx_in_reset = True
        else:
            drop_idx_in_reset = False
        reduced_end_events_df = reduced_end_events_df.reset_index(drop=drop_idx_in_reset)
        #-------------------------
        return reduced_end_events_df
        
        
    #****************************************************************************************************
    @staticmethod
    def merge_end_events_df_with_mp(
        end_events_df      , 
        df_mp              , 
        merge_on_ede       = ['serialnumber', 'aep_premise_nb'], 
        merge_on_mp        = ['mfr_devc_ser_nbr', 'prem_nb'], 
        cols_to_include_mp = None, 
        drop_cols          = ['mfr_devc_ser_nbr', 'prem_nb'], 
        rename_cols        = None, 
        how                = 'left', 
        indicator          = False, 
        inplace            = True
    ):
        r"""
        NOTE: One may use the non-static version merge_df_with_mp

        If cols_to_include_mp is None, all columns in df_mp are included in the merge
        """
        #-------------------------
        if not inplace:
            end_events_df = end_events_df.copy()
        #-------------------------
        if cols_to_include_mp is None:
            df_mp_to_merge = df_mp
        else:
            assert(Utilities.is_object_one_of_types(cols_to_include_mp, [list, tuple]))
            df_mp_to_merge = df_mp[cols_to_include_mp]
        #-------------------------
        # For proper merge to take place, the data types for the columns being merged should align
        #   e.g., for each, should both be strings, or should both be ints, etc.
        if isinstance(merge_on_ede, list):
            assert(isinstance(merge_on_mp, list) and len(merge_on_ede)==len(merge_on_mp))
            for i_merge in range(len(merge_on_ede)):
                if end_events_df[merge_on_ede[i_merge]].dtype!=df_mp_to_merge[merge_on_mp[i_merge]].dtype:
                    df_mp_to_merge = Utilities_df.convert_col_type(
                        df                = df_mp_to_merge, 
                        column            = merge_on_mp[i_merge], 
                        to_type           = end_events_df[merge_on_ede[i_merge]].dtype, 
                        to_numeric_errors = 'coerce', 
                        inplace           = True
                    )
                assert(end_events_df[merge_on_ede[i_merge]].dtype==df_mp_to_merge[merge_on_mp[i_merge]].dtype)
        else:
            if end_events_df[merge_on_ede].dtype!=df_mp_to_merge[merge_on_mp].dtype:
                df_mp_to_merge = Utilities_df.convert_col_type(
                    df                = df_mp_to_merge, 
                    column            = merge_on_mp, 
                    to_type           = end_events_df[merge_on_ede].dtype, 
                    to_numeric_errors = 'coerce', 
                    inplace           = True
                )
            assert(end_events_df[merge_on_ede].dtype==df_mp_to_merge[merge_on_mp].dtype)
        #-----
        end_events_df = end_events_df.merge(df_mp_to_merge, how=how, left_on=merge_on_ede, right_on=merge_on_mp, indicator=indicator)
        if drop_cols is not None:
            end_events_df = end_events_df.drop(columns=drop_cols)
        if rename_cols is not None:
            end_events_df = end_events_df.rename(columns=rename_cols)
        #-------------------------
        return end_events_df    
    
    #****************************************************************************************************
    @staticmethod
    def get_reason_counts_for_group(
        end_events_df_i           , 
        group_cols                = 'outg_rec_nb', 
        serial_number_col         = 'serialnumber', 
        reason_col                = 'reason', 
        include_normalize_by_nSNs = False, 
        inclue_zero_counts        = False,
        possible_reasons          = None,
        include_nSNs              = True, 
        include_SNs               = True, 
        prem_nb_col               = 'aep_premise_nb', 
        include_nprem_nbs         = False,
        include_prem_nbs          = False
    ):
        r"""
        end_events_df_i should be a pd.DataFrame for a single group
            Meaning, each column in group_cols should have one unique value

        Returns pd.DataFrame object
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(group_cols, [str, list, tuple]))
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        # NOTE: Below <=1 instead of ==1 because can equal 0 when gpby_dropna==False in get_reason_counts_per_group
        assert(all(end_events_df_i[group_cols].nunique()<=1))
        #-------------------------
        n_sns = end_events_df_i[serial_number_col].nunique()
        if include_nprem_nbs:
            n_prem_nbs = end_events_df_i[prem_nb_col].nunique()
        # TODO WHICH METHOD for calculating counts_i below is most correct?
        counts_i = end_events_df_i.groupby([serial_number_col, reason_col]).size().groupby(reason_col).sum()
        #counts_i = end_events_df_i.groupby([serial_number_col, reason_col])[reason_col].count().groupby(reason_col).sum()
        counts_i.name = 'counts'
        #---------------
        if inclue_zero_counts:
            assert(possible_reasons is not None)
            for possible_reason in possible_reasons:
                if possible_reason not in counts_i:
                    counts_i[possible_reason]=0
        #---------------
        if include_nSNs:
            counts_i['_nSNs']=n_sns
        if include_nprem_nbs:
            counts_i['_nprem_nbs']=n_prem_nbs
        #---------------
        counts_i=counts_i.sort_index()
        #---------------
        if include_normalize_by_nSNs:
            counts_i_norm= counts_i / n_sns
            # Don't want the number of SNs to be normalized, as this would trivially be 1
            if include_nSNs:
                counts_i_norm['_nSNs']      = n_sns
            if include_nprem_nbs:
                counts_i_norm['_nprem_nbs'] = n_prem_nbs
            if include_SNs:
                counts_i['_SNs']            = sorted(end_events_df_i[serial_number_col].unique().tolist())
                counts_i_norm['_SNs']       = sorted(end_events_df_i[serial_number_col].unique().tolist())
            if include_prem_nbs:
                counts_i['_prem_nbs']       = sorted(end_events_df_i[prem_nb_col].unique().tolist())
                counts_i_norm['_prem_nbs']  = sorted(end_events_df_i[prem_nb_col].unique().tolist())
            counts_i_norm.name = 'counts_norm'
            counts_i = pd.merge(counts_i, counts_i_norm, right_index=True, left_index=True)
        else:
            if include_SNs:
                counts_i['_SNs']      = sorted(end_events_df_i[serial_number_col].unique().tolist())
            if include_prem_nbs:
                counts_i['_prem_nbs'] = sorted(end_events_df_i[prem_nb_col].unique().tolist())
            counts_i = counts_i.to_frame()
        #---------------
        return counts_i
    
    @staticmethod
    def get_reason_counts_for_outage(
        end_events_df_i           , 
        outg_rec_nb_col           = 'outg_rec_nb', 
        serial_number_col         = 'serialnumber', 
        reason_col                = 'reason', 
        include_normalize_by_nSNs = False, 
        inclue_zero_counts        = False,
        possible_reasons          = None,
        include_nSNs              = True, 
        include_SNs               = True, 
        prem_nb_col               = 'aep_premise_nb', 
        include_nprem_nbs         = False,
        include_prem_nbs          = False
    ):
        r"""
        end_events_df_i should be a pd.DataFrame for a single outage
        
        Returns pd.DataFrame object
        """
        assert(end_events_df_i[outg_rec_nb_col].nunique()==1)
        return AMIEndEvents.get_reason_counts_for_group(
            end_events_df_i           = end_events_df_i, 
            group_cols                = outg_rec_nb_col, 
            serial_number_col         = serial_number_col, 
            reason_col                = reason_col, 
            include_normalize_by_nSNs = include_normalize_by_nSNs, 
            inclue_zero_counts        = inclue_zero_counts,
            possible_reasons          = possible_reasons,
            include_nSNs              = include_nSNs, 
            include_SNs               = include_SNs, 
            prem_nb_col               = prem_nb_col, 
            include_nprem_nbs         = include_nprem_nbs,
            include_prem_nbs          = include_prem_nbs
        )
        
    @staticmethod
    def convert_reason_counts_per_outage_df_long_to_wide(
        rcpo_df_long      , 
        reason_col        = 'reason', 
        higher_order_cols = None, 
        fillna_w_0        = True, 
        drop_lvl_0_col    = False
    ):
        r"""
        Convert a long format reason_counts_per_outage_df to a wide format.

        First, need to flatten index of rcpo_df_long, making reason_col a column instead of level of the index
        Then, simply pivot about reason_col

        NOTE: Each entry in long does not necessarily have the same collection of reasons as the others.
              Missing reasons are filled with NaN.  Setting fillna_w_0=True replaces these with 0 (as these signify
              0 counts for that reason)
        """
        #--------------------------------------------------
        warnings.filterwarnings('ignore', '.*DataFrame is highly fragmented.*', )
        #--------------------------------------------------
        # Need reason_col (and any higher_order_cols) to be contained in columns
        if higher_order_cols is not None:
            assert(isinstance(higher_order_cols, list))
            final_cols = higher_order_cols + [reason_col]
        else:
            final_cols = [reason_col]
        #-------------------------
        # If not in columns, need to find which index level contains final_cols and reset those indices 
        #  (effectively turning those indices into columns)
        for final_col_i in final_cols:
            if final_col_i not in rcpo_df_long.columns:
                assert(final_col_i in rcpo_df_long.index.names)
                final_col_i_idx_level = rcpo_df_long.index.names.index(final_col_i)
                rcpo_df_long          = rcpo_df_long.reset_index(level=final_col_i_idx_level)
        #-------------------------
        assert(len(set(rcpo_df_long.columns).difference(set(final_cols)))==1)
        rcpo_df_wide = rcpo_df_long.pivot(columns=final_cols)
        #-------------------------
        if fillna_w_0:
            rcpo_df_wide = rcpo_df_wide.fillna(0)
        #-------------------------
        rcpo_df_wide = rcpo_df_wide[rcpo_df_wide.columns.sort_values()]
        if drop_lvl_0_col and rcpo_df_wide.columns.get_level_values(0).nunique()==1:
            rcpo_df_wide = rcpo_df_wide.droplevel(level=0, axis=1)
        #--------------------------------------------------
        warnings.filterwarnings('default', '.*DataFrame is highly fragmented.*', )
        #--------------------------------------------------
        return rcpo_df_wide

    @staticmethod
    def convert_reason_counts_per_outage_df_wide_to_long(
        rcpo_df_wide               , 
        normalize_by_nSNs_included , 
        new_counts_col             = 'counts', 
        new_counts_norm_col        = None, 
        org_counts_col             = 'counts', 
        org_counts_norm_col        = 'counts_norm', 
        idx_0_name                 = 'outg_rec_nb'
    ):
        r"""
        Convert a wide format reason_counts_per_outage_df to a long format.
        
        If new_counts_norm_col is None, new_counts_norm_col = f'{new_counts_col}_norm'
        
        normalize_by_nSNs_included:
          If True, the columns in rcpo_df_wide should be MultiIndex, with
            level 0: type of counts, raw or normalized (typically values are 'counts' and 'counts_norm', 
                     and typical name=None)
            level 1: reason (e.g., values are 'Last Gasp...', etc., and typical name='reason')
        ----------------------------------------------------------------------------------------------------
        NOTE: When using pd.melt without specifying var_name (as I WAS doing initially, see why I no longer am below):
         For case of normal, single indexed columns (i.e., when normalize_by_nSNs_included is False): 
           If the columns name is None, 'variable' is used.
           Typically, when when normalize_by_nSNs_included is False, rcpo_df_wide.columns.name is 'reason'
             (or whatever reason_col was in the original end_events_df) due to pivot about reason_col in 
             get_reason_counts_per_outage.
           However, one must be careful and check if rcpo_df_wide.columns.name (or rcpo_df_wide.columns.names[0]) 
             is None, in which case new_reason_col should be set to 'variable'
         For case of MultiIndexed columns (i.e., when normalize_by_nSNs_included is True): 
           A new variable column is created for each level in rcpo_df_wide.columns.
           By default, the column names are used as the var_names
             Side Note: The number of variable columns created will equal the number of names supplied in var_name
                        So, even if df.columns has two levels, if var_name='whatever', only level 0 will be included
                        For both to be included, var_name must be a tuple/list with two elements
           If one of the column level names is empty, the corresponding column name is simply None (and printed as NaN when viewing)
             Calling all_cols_melted_df[None] will correctly retrieve that resulting column.
           However, if BOTH of the column level names are empty, the default behavior is to set var_name=['variable_0', 'variable_1'], 
             in which case, below, one needs to set:
             new_counts_type_col = 'variable_0' (instead of col_level_names[0])
             new_reason_col      = 'variable_1' (instead of col_level_names[1])
             
         Why am I now specifying var_name, instead of leaving it blank?
           The main reason for this switch due to the case when new_reason_column is None.  As described above, this would only occur
             when normalize_by_nSNs_included is True, rcpo_df_wide.columns has a level 0 name, but not a level 1 name.  In reality,
             when I'm using the code, this situation will likely never occur, but it is good to plan for all foreseeable cases.
           This is an issue simply due to the fact that reason_col is dropped at some point in the code.  Even if only one column with name None
             exists, pandas still fails to do this.  
           I could circumvent this by, e.g., all_cols_melted_df = all_cols_melted_df.drop(columns=[all_cols_melted_df.columns[1]]), but I would
             have to put in another if normalize_by_nSNs_included statement (as, when normalize_by_nSNs_included is False, the column to be dropped
             would be all_cols_melted_df.columns[0], not all_cols_melted_df.columns[1]).
           The simpler solution is to include a var_name when calling pd.melt!
        ----------------------------------------------------------------------------------------------------
        """
        #-------------------------
        if new_counts_norm_col is None:
            new_counts_norm_col = f'{new_counts_col}_norm'
        #-------------------------
        col_level_names = list(rcpo_df_wide.columns.names)
        # If normalized values included, there should two levels
        #   The first level containing the type info (counts or counts_norm)
        #   The second level containing the reason
        # If normalized values not included, typically only one level
        #   But allow for the case of two levels, if, e.g., a single value for level 0 (e.g., counts)
        #   was projected out of a df containing normalized values
        # Either way, len(col_level_names) should be less than or equal to 2
        assert(len(col_level_names)<=2)
        if normalize_by_nSNs_included:
            assert(len(col_level_names)==2)
        #-------------------------
        if len(col_level_names)==2:
            if col_level_names[0] is None and col_level_names[1] is None:
                new_counts_type_col = 'variable_0'
                new_reason_col      = 'variable_1'
            else:
                new_counts_type_col = col_level_names[0]
                new_reason_col      = col_level_names[1]
                if new_reason_col is None:
                    new_reason_col = 'tmp_reason_col'

            # If I use value_name = 'counts' when normalize_by_nSNs_included is True, I get the warning:
            #   FutureWarning: This dataframe has a column name that matches the 'value_name' column name of the 
            #   resulting Dataframe. In the future this will raise an error, please set the 'value_name' parameter 
            #   of DataFrame.melt to a unique name.
            # This is because level 0 of the MutliIndex columns contains 'counts' for half the columns
            # To be safe, since new_counts_col is oftentimes set to 'counts' (as this is the default behavior)
            #   set value_name to a random temporary value, instead of to new_counts_col as is done when 
            #   normalize_by_nSNs_included is False below
            value_name = Utilities.generate_random_string()
            var_name   = [new_counts_type_col, new_reason_col]
        else:
            assert(len(col_level_names)==1)
            new_reason_col = col_level_names[0]
            if new_reason_col is None:
                new_reason_col = 'variable' # default value given to var_name when rcpo_df_wide.columns.name is None
            value_name = new_counts_col
            var_name   = new_reason_col
        #--------------------------------------------------
        # Below, ignore_index is False so that outg_rec_nb index is maintained
        all_cols_melted_df = pd.melt(rcpo_df_wide, ignore_index=False, value_name=value_name, var_name=var_name)
        #--------------------------------------------------
        # Grab the reason series, containing the outg_rec_nb and reason, make it a list of tuples, and use that to 
        # build a MultiIndex for the dataframe
        all_cols_melted_df = all_cols_melted_df.set_index(pd.MultiIndex.from_tuples(all_cols_melted_df[new_reason_col].items()))
        all_cols_melted_df.index.set_names(names=[idx_0_name, new_reason_col], level=[0,1], inplace=True)
        # Sort the index
        all_cols_melted_df = all_cols_melted_df.sort_index()
        # Drop the reason column, as it is now included in the index and therefore redundant
        all_cols_melted_df = all_cols_melted_df.drop(columns=new_reason_col)
        #--------------------------------------------------
        # If normalize_by_nSNs_included, the counts and counts_norm values are intermixed in a single column
        # It is possible to determine which is which through the new_counts_type_col (typically = None, and printed as NaN)
        # Therefore, these two sets must be separated out and then merged back together so the counts and counts_norm
        # are in different columns
        if normalize_by_nSNs_included:
            # First, project out raw and norm
            all_cols_melted_df_raw = all_cols_melted_df[all_cols_melted_df[new_counts_type_col]==org_counts_col]
            all_cols_melted_df_nrm = all_cols_melted_df[all_cols_melted_df[new_counts_type_col]==org_counts_norm_col]

            # Now, new_counts_type_col no longer needed, so drop
            all_cols_melted_df_raw = all_cols_melted_df_raw.drop(columns=[new_counts_type_col])
            all_cols_melted_df_nrm = all_cols_melted_df_nrm.drop(columns=[new_counts_type_col])

            # Rename the counts columns
            # Remember, above, due to the FutureWarning, I had to give value_name a random name
            all_cols_melted_df_raw = all_cols_melted_df_raw.rename(columns={value_name:new_counts_col})
            all_cols_melted_df_nrm = all_cols_melted_df_nrm.rename(columns={value_name:new_counts_norm_col})

            # Finally, merge the two
            assert(all_cols_melted_df_raw.shape==all_cols_melted_df_nrm.shape)
            all_cols_melted_df = all_cols_melted_df_raw.merge(all_cols_melted_df_nrm, how='inner', left_index=True, right_index=True)
        else:
            # For the case where normalize_by_nSNs_included is False, but DF still has MultiIndex columns, need to drop new_counts_type_col
            #   and rename columns back from value_name=Utilities.generate_random_string()
            if len(col_level_names)==2:
                assert(rcpo_df_wide.columns.get_level_values(0).nunique()==1)
                all_cols_melted_df = all_cols_melted_df.drop(columns=[new_counts_type_col])
                all_cols_melted_df = all_cols_melted_df.rename(columns={value_name:new_counts_col})
        #--------------------------------------------------
        return all_cols_melted_df 
        
        
    @staticmethod
    def get_reason_counts_per_group(
        end_events_df             , 
        group_cols                = 'outg_rec_nb', 
        group_freq                = None, 
        gpby_dropna               = True, 
        serial_number_col         = 'serialnumber', 
        reason_col                = 'reason', 
        include_normalize_by_nSNs = False, 
        inclue_zero_counts        = False,
        possible_reasons          = None, 
        include_nSNs              = True, 
        include_SNs               = True, 
        prem_nb_col               = 'aep_premise_nb', 
        include_nprem_nbs         = False,
        include_prem_nbs          = False, 
        return_form               = dict(
            return_multiindex_outg_reason = True, 
            return_normalized_separately  = False
        )    
    ):
        r"""
        end_events_df should be a pd.DataFrame with data from multiple outages

        If possible_reasons is None, they will be inferred from end_events_df

        Returns:
          return_form['return_normalized_separately']==False:
            a pd.DataFrame object
          return_form['return_normalized_separately']==True
            a dict object with keys 'counts' and 'counts_norm' and values which are pd.DataFrame objects


        return_form['return_multiindex_outg_reason']:
          If return_form['return_multiindex_outg_reason']==True:
            The returned pd.DataFrame object(s) has(have) MultiIndex indices with df.index.names  = ['outg_rec_nb', 'reason']
              e.g., (00000001, 'Diag1 Condition cleared')
            This is a LONG FORMAT dataframe
            The returned pd.DataFrame object(s) will have a form similar to:
                _______________________________________________________________________________________
                                                                                counts    counts_norm
                outg_rec_nb reason                                                       
                00000001    Diag1 Condition cleared                                 29    1.035714
                            ...
                            Under Voltage cleared (CA000400)                       112    4.0   
                00000002    Diag1 Condition cleared                                 26    0.742857
                            ...
                            Under Voltage cleared (CA000400)                        35    1.0   
                ...
                _______________________________________________________________________________________

          If return_form['return_multiindex_outg_reason']==False:
            The returned pd.DataFrame object(s) has(have) indices equal to the outg_rec_nb, and columns equal to the
              various reasons.  Basically, the above pd.DataFrame pivoted with columns='reason' (after flattening the MultiIndex)
            This is a WIDE FORMAT dataframe  
            The returned pd.DataFrame object(s) will have a form similar to:
                _______________________________________________________________________________________________________________________________________________        
                            counts                                                             counts_norm
                reason      Diag1 Condition cleared   ...   Under Voltage cleared (CA000400)   Diag1 Condition cleared   ...   Under Voltage cleared (CA000400)
                outg_rec_nb                           
                00000001                         29         112                                1.035714                        4.0
                00000002                         26         35                                 0.742857                        1.0
                ...
                _______________________________________________________________________________________________________________________________________________
                *** NOTE: In this case, with both counts and counts_norm, the columns are MultiIndex (e.g., ('counts', 'Diag1 Condition cleared'))
                          df.index.name    = 'outg_rec_nb'
                          df.columns.names = [None, 'reason']


            In the case where return_form['return_normalized_separately']==True (or, include_normalize_by_nSNs==False), to MultiIndex columns shown above will
              be flattened, i.e., will have the form
                ____________________________________________________________________________  
                reason      Diag1 Condition cleared   ...   Under Voltage cleared (CA000400)
                outg_rec_nb                           
                00000001                         29         112                             
                00000002                         26         35                              
                ...
                ____________________________________________________________________________

        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(group_cols, [str, list, tuple]))
        if isinstance(group_cols, str):
            group_cols = [group_cols]
        #-------------------------
        dflt_return_form = dict(
            return_multiindex_outg_reason = True, 
            return_normalized_separately  = False
        )
        return_form = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = return_form, 
            default_values_dict = dflt_return_form, 
            extend_any_lists    = False, 
            inplace             = False            
        )
        #-------------------------
        # NOTE: If returning a wide-format DF, i.e., when return_form['return_multiindex_outg_reason']==False, the conversion process (via 
        #         AMIEndEvents.convert_reason_counts_per_outage_df_long_to_wide) will naturally expand the set of reasons for each group
        #         to match the full set of reasons observed in all groups.
        #       IN OTHER WORDS, the conversion process will essentially cause inclue_zero_counts to be True (see documentation in 
        #         AMIEndEvents.convert_reason_counts_per_outage_df_long_to_wide for more details).
        #       It turns out that convert_reason_counts_per_outage_df_long_to_wide achieves the result more efficiently (in a shorter time)
        #         than using inclue_zero_counts=True for each group via  AMIEndEvents.get_reason_counts_for_group
        #       THUS, for efficiency, if return_form['return_multiindex_outg_reason']==False, inclue_zero_counts will be set to False,
        #         as the final result will be the same regardless, and setting to False runs faster.
        if return_form['return_multiindex_outg_reason']==False:
            inclue_zero_counts=False
        #-------------------------
        if inclue_zero_counts and possible_reasons is None:
            possible_reasons = end_events_df[reason_col].unique().tolist()
        #-------------------------
        if group_freq is not None:
            assert(Utilities.is_object_one_of_types(group_freq, [str, pd.core.resample.TimeGrouper]))
            if isinstance(group_freq, str):
                # Grouping by time on index!  If one wants to group by time on column, need to supply full
                #   pd.Grouper object with key argument equal to column
                group_freq = pd.Grouper(freq=group_freq)
            #-----
            grp_by_full = group_cols+[group_freq]
        else:
            grp_by_full = group_cols
        reason_counts_per_outg_df = end_events_df.groupby(grp_by_full, dropna=gpby_dropna).apply(
            lambda x: AMIEndEvents.get_reason_counts_for_group(
                end_events_df_i           = x, 
                group_cols                = group_cols, 
                serial_number_col         = serial_number_col, 
                reason_col                = reason_col, 
                include_normalize_by_nSNs = include_normalize_by_nSNs, 
                inclue_zero_counts        = inclue_zero_counts, 
                possible_reasons          = possible_reasons, 
                include_nSNs              = include_nSNs, 
                include_SNs               = include_SNs, 
                prem_nb_col               = prem_nb_col, 
                include_nprem_nbs         = include_nprem_nbs,
                include_prem_nbs          = include_prem_nbs
            )
        )
        #-------------------------
        if reason_counts_per_outg_df.shape[0]==0:
            if return_form['return_normalized_separately']:
                return {'counts':pd.DataFrame(), 'counts_norm':pd.DataFrame()}
            else:
                return pd.DataFrame()
        #-------------------------
        # NOTE: The assertions below are too strong.  If groups are empty for some reason (e.g., if grouping by two columns,
        #         and all the values for the second column of a group are NaN) then the assertion will fail.
        #       This shouldn't happen all that often, and the values should be approximately equal, but there are instances
        #         when they will be different, therefore the assertions must be removed.
        #for group_col in group_cols:
        #    idx_level = reason_counts_per_outg_df.index.names.index(group_col)
        #    assert(reason_counts_per_outg_df.index.get_level_values(idx_level).nunique()==end_events_df[group_col].nunique())
        #-------------------------
        if return_form['return_multiindex_outg_reason']:
            if return_form['return_normalized_separately'] and include_normalize_by_nSNs:
                return {
                    'counts':      reason_counts_per_outg_df[['counts']], 
                    'counts_norm': reason_counts_per_outg_df[['counts_norm']]
                }
            else:
                return reason_counts_per_outg_df
        else:
            reason_counts_per_outg_df = AMIEndEvents.convert_reason_counts_per_outage_df_long_to_wide(
                rcpo_df_long = reason_counts_per_outg_df, 
                reason_col   = reason_col, 
                fillna_w_0   = True
            )
            if not include_normalize_by_nSNs:
                # Below simply to get rid of annoying MultiIndex columns, which are not necessary here
                # since all would have level(0) values equal to 'counts'
                return reason_counts_per_outg_df['counts']
            else:
                if return_form['return_normalized_separately']:
                    return {
                        'counts':      reason_counts_per_outg_df['counts'], 
                        'counts_norm': reason_counts_per_outg_df['counts_norm']
                    }
                else:
                    return reason_counts_per_outg_df
                    
                    
    def get_reason_counts_per_group_QUICK(
            end_events_df         , 
            group_cols            = ['serialnumber', 'aep_premise_nb'], 
            group_freq            = None, 
            gpby_dropna           = True, 
            reason_col            = 'reason', 
            set_group_cols_as_idx = True
    ):
        r"""
        """
        #-------------------------
        # String and tuple (for MultiIndex columns) used for single group_col, whereas
        #   list used for multiple
        assert(Utilities.is_object_one_of_types(group_cols, [str, list, tuple]))
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        #-------------------------
        tmp_idx_col=None #Only set if group_freq is str, in which case grouping index
        if group_freq is not None:
            assert(Utilities.is_object_one_of_types(group_freq, [str, pd.core.resample.TimeGrouper]))
            if isinstance(group_freq, str):
                # Grouping by time on index!  
                # If one wants to group by time on column, need to supply full pd.Grouper object with 
                #   key argument equal to column
                # HOWEVER, functionality is easier here when grouping by column instead of index, so reset_index
                #   making it a column
                tmp_idx_col   = Utilities.generate_random_string()
                assert(end_events_df.index.nlevels==1)
                end_events_df = end_events_df.reset_index(drop=False, names=tmp_idx_col)
                group_freq    = pd.Grouper(freq=group_freq, key=tmp_idx_col)
            #-----
            grp_by_full = group_cols+[group_freq]
        else:
            grp_by_full = group_cols
        #-------------------------
        # First, get the number of counts in each group for each reason
        # Must add reason to grp_by_full, if not already (set operation to ensure not added twice)
        grp_by_full = list(set(grp_by_full+[reason_col]))

        # The result of this operation will be a DF with columns equal to grp_by_full+'size'
        return_df = end_events_df.groupby(grp_by_full, dropna=gpby_dropna, as_index=False).size()

        #-------------------------
        # Second, pivot about the reason column (with indices equal to all others in grp_by_full)
        # So, reason_col must be removed form grp_by_full
        # Also, if group_freq included, the item in grp_by_full needs to be changed from the pd.Grouper object
        #   to the column used for grouping (i.e., to group_freq.key)
        # NOTE: This will make the columns MultiIndex, with level_0 values equal to 'size' and level_1
        #       values equal to the reasons.
        # Any groups missing counts for various reasons will be filled with nan, hence the need for fillna(0)
        grp_by_full.remove(reason_col)
        if group_freq is not None:
            grp_by_full.remove(group_freq)
            grp_by_full.append(group_freq.key)
        return_df = return_df.pivot(index=grp_by_full, columns=reason_col).fillna(0)
        #-----
        # Drop level 0 from columns (all values = 'size', as stated above)
        # This will leave the reason columns.  However, this will give return_df.columns.name = reason_col, which
        #   will be confusing after we reset the index to include the grp_by_full values.
        # So, also rename the columns to None
        return_df = return_df.droplevel(0, axis=1)
        return_df.columns.name = None
        return_df = return_df.reset_index()
        
        #-------------------------
        if set_group_cols_as_idx:
            return_df = return_df.set_index(group_cols)

        #-------------------------
        return return_df
        
    @staticmethod    
    def get_reason_counts_per_outage(
        end_events_df             , 
        outg_rec_nb_col           = 'outg_rec_nb', 
        group_freq                = None, 
        serial_number_col         = 'serialnumber', 
        reason_col                = 'reason', 
        include_normalize_by_nSNs = False, 
        inclue_zero_counts        = False,
        possible_reasons          = None, 
        include_nSNs              = True, 
        include_SNs               = True, 
        prem_nb_col               = 'aep_premise_nb', 
        include_nprem_nbs         = False,
        include_prem_nbs          = False, 
        return_form               = dict(
            return_multiindex_outg_reason = True, 
            return_normalized_separately  = False
        )
    ):
        r"""
        end_events_df should be a pd.DataFrame with data from multiple outages
        
        If possible_reasons is None, they will be inferred from end_events_df
        
        Returns:
          return_form['return_normalized_separately']==False:
            a pd.DataFrame object
          return_form['return_normalized_separately']==True
            a dict object with keys 'counts' and 'counts_norm' and values which are pd.DataFrame objects
                    
                    
        return_form['return_multiindex_outg_reason']:
          If return_form['return_multiindex_outg_reason']==True:
            The returned pd.DataFrame object(s) has(have) MultiIndex indices with df.index.names  = ['outg_rec_nb', 'reason']
              e.g., (00000001, 'Diag1 Condition cleared')
            This is a LONG FORMAT dataframe
            The returned pd.DataFrame object(s) will have a form similar to:
                _______________________________________________________________________________________
                                                                                counts    counts_norm
                outg_rec_nb reason                                                       
                00000001    Diag1 Condition cleared                                 29    1.035714
                            ...
                            Under Voltage cleared (CA000400)                       112    4.0   
                00000002    Diag1 Condition cleared                                 26    0.742857
                            ...
                            Under Voltage cleared (CA000400)                        35    1.0   
                ...
                _______________________________________________________________________________________
                
          If return_form['return_multiindex_outg_reason']==False:
            The returned pd.DataFrame object(s) has(have) indices equal to the outg_rec_nb, and columns equal to the
              various reasons.  Basically, the above pd.DataFrame pivoted with columns='reason' (after flattening the MultiIndex)
            This is a WIDE FORMAT dataframe  
            The returned pd.DataFrame object(s) will have a form similar to:
                _______________________________________________________________________________________________________________________________________________        
                            counts                                                             counts_norm
                reason      Diag1 Condition cleared   ...   Under Voltage cleared (CA000400)   Diag1 Condition cleared   ...   Under Voltage cleared (CA000400)
                outg_rec_nb                           
                00000001                         29         112                                1.035714                        4.0
                00000002                         26         35                                 0.742857                        1.0
                ...
                _______________________________________________________________________________________________________________________________________________
                *** NOTE: In this case, with both counts and counts_norm, the columns are MultiIndex (e.g., ('counts', 'Diag1 Condition cleared'))
                          df.index.name    = 'outg_rec_nb'
                          df.columns.names = [None, 'reason']

                
            In the case where return_form['return_normalized_separately']==True (or, include_normalize_by_nSNs==False), to MultiIndex columns shown above will
              be flattened, i.e., will have the form
                ____________________________________________________________________________  
                reason      Diag1 Condition cleared   ...   Under Voltage cleared (CA000400)
                outg_rec_nb                           
                00000001                         29         112                             
                00000002                         26         35                              
                ...
                ____________________________________________________________________________
            
        """
        #-------------------------
        return AMIEndEvents.get_reason_counts_per_group(
            end_events_df             = end_events_df, 
            group_cols                = outg_rec_nb_col, 
            group_freq                = group_freq, 
            serial_number_col         = serial_number_col, 
            reason_col                = reason_col, 
            include_normalize_by_nSNs = include_normalize_by_nSNs, 
            inclue_zero_counts        = inclue_zero_counts,
            possible_reasons          = possible_reasons, 
            include_nSNs              = include_nSNs, 
            include_SNs               = include_SNs, 
            prem_nb_col               = prem_nb_col, 
            include_nprem_nbs         = include_nprem_nbs,
            include_prem_nbs          = include_prem_nbs, 
            return_form               = return_form
        )
                    
    #****************************************************************************************************
    @staticmethod                
    def make_reason_counts_per_outg_columns_equal(
        rcpo_df_1                     , 
        rcpo_df_2                     , 
        same_order                    = True, 
        inplace                       = True, 
        cols_to_init_with_empty_lists = None
    ):
        r"""
        Make two reason_counts_per_outage_dfs have the same set of columns, i.e., the same set of reasons.
        The missing columns in each are added as columns of zeros, which is appropriate as no instances of
          the missing reason were found.
          EXCEPT any missing column in cols_to_init_with_empty_lists is added as a column of empty lists!
          
        This is important when joining two pd.DataFrames, or when plotting two together  
        
        If inplace is True, nothing is returned
        If inplace is False, the updated input DFs are returned
        
        See make_reason_counts_per_outg_columns_equal_dfs_list if functionality desired for more than two DFs
          This can probably be replaced by make_reason_counts_per_outg_columns_equal_dfs_list, or at a minimum
          make this simply return make_reason_counts_per_outg_columns_equal_dfs_list([rcpo_df_1, rcpo_df_2])
          
        cols_to_init_with_empty_lists:
          Defaults to AMIEndEvents.std_SNs_cols() when cols_to_init_with_empty_lists is None
        """
        #-------------------------
        if cols_to_init_with_empty_lists is None:
            cols_to_init_with_empty_lists=AMIEndEvents.std_SNs_cols()
        #-------------------------
        warnings.filterwarnings('ignore', '.*DataFrame is highly fragmented.*', )
        #-------------------------
        if not inplace:
            rcpo_df_1 = rcpo_df_1.copy()
            rcpo_df_2 = rcpo_df_2.copy()
        #-------------------------
        symm_diff_cols = set(rcpo_df_1.columns).symmetric_difference(set(rcpo_df_2.columns))
        n_rows_1 = rcpo_df_1.shape[0]
        n_rows_2 = rcpo_df_2.shape[0]
        for col in symm_diff_cols:
            if col not in rcpo_df_1.columns:
                if col in cols_to_init_with_empty_lists:
                    rcpo_df_1[col] = [[]]*n_rows_1
                else:
                    rcpo_df_1[col] = [0.0]*n_rows_1
            if col not in rcpo_df_2.columns:
                if col in cols_to_init_with_empty_lists:
                    rcpo_df_2[col] = [[]]*n_rows_2
                else:
                    rcpo_df_2[col] = [0.0]*n_rows_2
        # Make sure operation worked as expected
        assert(len(set(rcpo_df_1.columns).symmetric_difference(set(rcpo_df_2.columns)))==0)
        if same_order:
            rcpo_df_1.sort_index(axis=1, inplace=True)
            rcpo_df_2.sort_index(axis=1, inplace=True)
            # Make sure operation worked as expected
            assert(rcpo_df_1.columns.tolist()==rcpo_df_2.columns.tolist())
        #-------------------------
        warnings.filterwarnings('default', '.*DataFrame is highly fragmented.*', )
        #-------------------------
        if not inplace:
            return (rcpo_df_1, rcpo_df_2)
        else:
            return
        
    @staticmethod
    def make_reason_counts_per_outg_columns_equal_dfs_list(
        rcpo_dfs                      , 
        col_level                     = -1, 
        same_order                    = True, 
        inplace                       = True, 
        cols_to_init_with_empty_lists = None
    ):
        r"""
        Just like make_reason_counts_per_outg_columns_equal, but for a list of pd.DataFrames, not just two.
        Probably make_reason_counts_per_outg_columns_equal can be replaced in favor of this more general function.
        
        The missing columns in each are added as columns of zeros, which is appropriate as no instances of
          the missing reason were found.
          EXCEPT any missing column in cols_to_init_with_empty_lists is added as a column of empty lists!
    
        This is important when joining two pd.DataFrames, or when plotting two together  
        
        If inplace is True, nothing is returned
        If inplace is False, the updated input DFs are returned
        
        NOTE: Occasionally pandas will output a warning due to the code below stating, "PerformanceWarning: DataFrame is highly fragmented...."
              I suggest you just ignore this.  There is no option for inplace with pd.concat, so when inplace=True the method suggested by Python
              will not work.
              
        cols_to_init_with_empty_lists:
          Defaults to AMIEndEvents.std_SNs_cols() when cols_to_init_with_empty_lists is None
        """
        #--------------------------------------------------
        if cols_to_init_with_empty_lists is None:
            cols_to_init_with_empty_lists=AMIEndEvents.std_SNs_cols()
        #-------------------------
        warnings.filterwarnings('ignore', '.*DataFrame is highly fragmented.*', )
        #--------------------------------------------------
        # Make sure all DFs have same number of levels in columns
        cols_n_levels = rcpo_dfs[0].columns.nlevels
        assert(cols_n_levels<=2)
        assert(np.all([x.columns.nlevels==cols_n_levels for x in rcpo_dfs]))
        #--------------------------------------------------
        if cols_n_levels==2:
            # The rcpo_dfs can have distinct, unique, lvl_0 values.
            #   e.g., = ['01-06 Days'], ['06-11 Days'], ['11-16 Days'], ['16-21 Days'], ['21-26 Days'], ['26-30 Days']
            # HOWEVER, if they have more than one, they must share the same unique sets
            #   e.g., = [['counts', 'counts_norm]], [['counts', 'counts_norm]]
            n_unq_lvl_0_vals = rcpo_dfs[0].columns.get_level_values(0).nunique()
            assert(np.all([x.columns.get_level_values(0).nunique()==n_unq_lvl_0_vals for x in rcpo_dfs]))
            if n_unq_lvl_0_vals>1:
                unq_lvl_0_vals = rcpo_dfs[0].columns.get_level_values(0).unique().tolist()
                assert(np.all([set(x.columns.get_level_values(0).unique().tolist()).symmetric_difference(set(unq_lvl_0_vals))==set() for x in rcpo_dfs]))
                dfs_by_lvl_0_val = []
                for unq_lvl_0_val_i in unq_lvl_0_vals:
                    rcpo_dfs_i = [df_i[unq_lvl_0_val_i].copy() for df_i in rcpo_dfs]
                    AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list(
                        rcpo_dfs                      = rcpo_dfs_i, 
                        col_level                     = col_level, 
                        same_order                    = same_order, 
                        inplace                       = True, 
                        cols_to_init_with_empty_lists = cols_to_init_with_empty_lists
                    )
                    dfs_by_lvl_0_val.append(rcpo_dfs_i)
                #-------------------------
                # Need to combine back together the various lvl_0_vals for each df
                assert(len(dfs_by_lvl_0_val)==len(unq_lvl_0_vals))
                assert(Utilities.are_list_elements_lengths_homogeneous(dfs_by_lvl_0_val, len(rcpo_dfs)))
                return_dfs_2d = [[x] for x in dfs_by_lvl_0_val[0]]
                for i,dfs_by_lvl_0_val_i in enumerate(dfs_by_lvl_0_val):
                    if i==0:
                        continue
                    #-------------------------
                    assert(len(return_dfs_2d)==len(dfs_by_lvl_0_val_i))
                    for j in range(len(return_dfs_2d)):
                        return_dfs_2d[j].append(dfs_by_lvl_0_val_i[j])
                #-------------------------
                return_dfs = [pd.concat(x, axis=1) for x in return_dfs_2d]
                #-------------------------
                return return_dfs
        #--------------------------------------------------
        if col_level == -1:
            col_level = cols_n_levels-1
        #-----
        # col_lvl_const_vals will be used later if rcpo_dfs have MultiIndex columns!
        # It will allow use to:
        #   Set the values of this column level all equal to each other.
        #   If df has a single value for this column level, use it.
        #   Otherwise, generate a random string
        col_lvl_const_vals = [Utilities.generate_random_string(str_len=5, letters='letters_only') for _ in range(cols_n_levels)]
        #--------------------------------------------------
        all_cols = []
        for df in rcpo_dfs:
            all_cols.extend(df.columns.get_level_values(col_level).tolist())
        all_cols=list(set(all_cols))
        #-------------------------
        return_dfs = []
        for i,df in enumerate(rcpo_dfs):
            n_rows = df.shape[0]
            new_cols_dict={}
            for col in all_cols:
                if col not in df.columns.get_level_values(col_level).tolist():
                    if col in cols_to_init_with_empty_lists:
                        new_col_vals = [[]]*n_rows
                    else:
                        new_col_vals = [0.0]*n_rows
                    #---------------
                    assert(col not in new_cols_dict)
                    new_cols_dict[col] = new_col_vals
                else:
                    continue
            # END for col in all_cols
            if len(new_cols_dict)>0:
                new_cols_df = pd.DataFrame(data=new_cols_dict, index=df.index)
                #--------------------------------------------------
                # If df has MultiIndex columns, insert levels into new_cols_df
                if df.columns.nlevels > 1:
                    #-------------------------
                    multi_idx_cols = []
                    #-----
                    for i_lvl in range(df.columns.nlevels):
                        if i_lvl == col_level:
                            multi_idx_cols.append(new_cols_df.columns.tolist())
                            continue
                        # If df has a single value for this column level, use it.
                        # Otherwise, use col_lvl_const_vals
                        lvl_i_vals = col_lvl_const_vals[i_lvl]
                        if df.columns.get_level_values(i_lvl).nunique()==1:
                            lvl_i_vals = df.columns.get_level_values(i_lvl).unique().tolist()[0]
                        multi_idx_cols.append([lvl_i_vals]*new_cols_df.shape[1])
                    #-------------------------
                    assert(len(multi_idx_cols)==df.columns.nlevels)
                    multi_idx_cols = pd.MultiIndex.from_arrays(multi_idx_cols, names=df.columns.names)
                    #-----
                    assert(multi_idx_cols.get_level_values(col_level).equals(new_cols_df.columns))
                    new_cols_df.columns = multi_idx_cols
                #--------------------------------------------------
                assert(df.columns.nlevels==new_cols_df.columns.nlevels)
                if not inplace:
                    df = pd.concat([df, new_cols_df], axis=1)
                else:
                    # See note above concerning warning
                    df[new_cols_df.columns] = new_cols_df.values
            if same_order:
                # df.sort_index(axis=1, level=col_level, inplace=True, key=natsort_keygen())
                df.sort_index(axis=1, inplace=True)
            return_dfs.append(df)
        # END for i,df in enumerate(rcpo_dfs)
        #-------------------------
        # Make sure operation worked as expected
        for df in return_dfs:
            assert(len(set(df.columns.get_level_values(col_level).tolist()).symmetric_difference(set(all_cols)))==0)
        #-------------------------
        if same_order:
            # Not sure why df.sort_index doesn't always handle things exactly right...
            col_level_order = return_dfs[0].columns.get_level_values(col_level).tolist()
            for i in range(len(return_dfs)):
                assert(set(return_dfs[i].columns.get_level_values(col_level).tolist()).symmetric_difference(set(col_level_order))==set())
                if return_dfs[i].columns.get_level_values(col_level).tolist() != col_level_order:
                    return_dfs[i] = return_dfs[i].reindex(col_level_order, level=col_level, axis=1)


            for df in return_dfs:
                # Make sure operation worked as expected
                assert(df.columns.get_level_values(col_level).tolist()==return_dfs[0].columns.get_level_values(col_level).tolist())
        #-------------------------
        warnings.filterwarnings('default', '.*DataFrame is highly fragmented.*', )
        #-------------------------
        if not inplace:
            return return_dfs
        else:
            return
            
    @staticmethod
    def make_reason_counts_per_outg_columns_equal_list_of_dicts_of_dfs(
        rcpo_dfs_dicts, 
        df_key_tags_to_ignore = ['ede_typeid_to_reason', 'reason_to_ede_typeid'], 
        same_order=True, 
        cols_to_init_with_empty_lists=None
    ):
        r"""
        rcpo_dfs_dicts should be a list with elements equal to dicts with pd.DataFrame values.
        This function will make all DFs with matching keys have the same columns.
        
        df_key_tags_to_ignore:
          Utilities.remove_tagged_from_list is used to remove these from all df_keys found
          
        cols_to_init_with_empty_lists:
          Defaults to AMIEndEvents.std_SNs_cols() when cols_to_init_with_empty_lists is None

        NOTE: The changes are made IN PLACE!
              I suppose making an option with inplace=False is possible, but more of a 
                hassle than it's worth.

        e.g.
            Suppose rcpo_dfs_dicts = [dfs_dict_0, dfs_dict_1, dfs_dict_2]
            where:
                   dfs_dict_0 = {'df_a':df_0_a, 'df_b':df_0_b, 'df_c':df_0_c}
                   dfs_dict_1 = {'df_a':df_1_a, 'df_b':df_1_b               }
                   dfs_dict_2 = {               'df_b':df_2_b, 'df_c':df_2_c}
            This function will make the columns equal for:
                df_0_a.columns==df_1_a.columns
                df_0_b.columns==df_1_b.columns==df_2_b.columns
                df_0_c.columns==df_2_c.columns
        """
        #-------------------------
        if cols_to_init_with_empty_lists is None:
            cols_to_init_with_empty_lists=AMIEndEvents.std_SNs_cols()
        #-------------------------
        # Make sure rcpo_dfs_dicts is a list of dicts
        assert(isinstance(rcpo_dfs_dicts, list))
        assert(all([isinstance(x, dict) for x in rcpo_dfs_dicts]))

        #-------------------------
        # Get unique keys from all dicts
        unique_df_keys = []
        for x in rcpo_dfs_dicts:
            unique_df_keys.extend(list(x.keys()))
        unique_df_keys = list(set(unique_df_keys))
        
        #-------------------------
        # Remove any df_key_tags_to_ignore from unique_df_keys
        if df_key_tags_to_ignore is not None:
            unique_df_keys = Utilities.remove_tagged_from_list(lst=unique_df_keys, tags_to_ignore=df_key_tags_to_ignore)

        #-------------------------
        # For each key, grab all entries from each dict having that key,
        #   and make the columns equal
        for df_key_i in unique_df_keys:
            rcpo_dfs_i = [dfs_dict[df_key_i] for dfs_dict in rcpo_dfs_dicts 
                          if df_key_i in dfs_dict]
            assert(len(rcpo_dfs_i)>0)
            #-------------------------
            # If only one DF, no need to call make_reason_counts_per_outg_columns_equal_dfs_list
            if len(rcpo_dfs_i)==0:
                continue
            #-------------------------
            AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list(
                rcpo_dfs = rcpo_dfs_i, 
                same_order=same_order, 
                inplace=True, 
                cols_to_init_with_empty_lists=cols_to_init_with_empty_lists
            )
        #-------------------------
        return
        
    @staticmethod
    def make_multiindex_reason_counts_per_outg_reasons_uniform(
        mi_rcpo_df, 
        same_order=True, 
        sort_index=True,
        cols_to_init_with_empty_lists=None
    ):
        r"""
        This is similar in spirit to make_reason_counts_per_outg_columns_equal.
        
        Here, the input pd.DataFrame (mi_rcpo_df, where mi stands for multiindex) is indexed by the outage number (level 0)
        and the reason (level 1).  The columns will either by 'counts', 'counts_norm', or both.
        This ensure that each outg_rec_nb index has the same reasons but inserting 0 values for missing reasons.
        
        cols_to_init_with_empty_lists:
          Defaults to AMIEndEvents.std_SNs_cols() when cols_to_init_with_empty_lists is None
        """
        #-------------------------
        if cols_to_init_with_empty_lists is None:
            cols_to_init_with_empty_lists=AMIEndEvents.std_SNs_cols()
        #-------------------------
        assert(mi_rcpo_df.index.nlevels==2)
        all_reasons = mi_rcpo_df.index.get_level_values(1).unique().tolist()
        return_df = pd.DataFrame()
        for idx, df_i in mi_rcpo_df.groupby(mi_rcpo_df.index.get_level_values(0)):
            reasons_needed = set(all_reasons).difference(set(df_i.index.get_level_values(1)))
            #-----
            multi_idx_tuples = [(idx, x) for x in reasons_needed]
            multi_idx = pd.MultiIndex.from_tuples(multi_idx_tuples, names=df_i.index.names)
            #-----
            # If none of cols_to_init_with_empty_lists found in reasons_needed, can proceed with 
            # original, simple method.
            # Otherwise, need to be more careful
            if len(set(cols_to_init_with_empty_lists).intersection(reasons_needed))==0:
                vals_to_append = np.zeros(shape=(len(reasons_needed), df_i.shape[1]))
            else:
                vals_to_append = []
                for reason in reasons_needed:
                    if reason in cols_to_init_with_empty_lists:
                        vals_to_append.append([])
                    else:
                        vals_to_append.append(0.0)
            vals_df_to_append = pd.DataFrame(vals_to_append, index=multi_idx, columns=df_i.columns)
            #-----
            return_df_i = pd.concat([df_i, vals_df_to_append])
            if sort_index:
                return_df_i = return_df_i.sort_index()
            #-------------------------
            return_df = pd.concat([return_df, return_df_i])
        #-------------------------
        return return_df

    @staticmethod
    def sum_numeric_cols_and_join_list_cols(
        df, 
        list_cols=['_SNs'], 
        list_counts_cols=['_nSNs'], 
        numeric_cols=None,
        update_list_counts_from_lists=True
    ):
        r"""
        The numeric columns in df are summed, whereas the columns containing lists (e.g., lists of serial numbers)
        are joined together

        list_cols:
          Identify all the columns whose elements are lists

        list_counts_cols:
          Columns containing the counts of each list columns.
          There must be the same number of list_cols as list_counts_cols, and the ordering should be the same.

        numeric_cols:
          Identify all the numeric columns to be summed.  If this is left as None, it is taken to be
          all columns not in list_cols

        update_list_counts_from_lists:
          If True (default), update the columns in list_counts_cols with the lengths of the lists in lists_cols.
          If False, list_counts_cols will simply be summed just like numeric_cols.
          NOTE:  It is typically a good idea to have update_list_counts_from_lists=True.  Typically, one wants, e.g.,
                 the number of unique serial numbers.  If a serial number exists in two (or more) rows of the DF, it will
                 be double (or more) counted when update_list_counts_from_lists==False, and thus the counts will not reflect
                 the number of unique entries!
        """
        #-------------------------
        # If there is only one row in df, then no summing/averaging/joining to be done, in which
        # case, simply return series version of df
        if df.shape[0]==1:
            return df.squeeze()
        #-------------------------
        # There must be the same number of list_cols as list_counts_cols, and the ordering should be the same.
        assert(len(list_cols)==len(list_counts_cols))
        #-------------------------
        if numeric_cols is None:
            numeric_cols = [x for x in df.columns if x not in list_cols]
        #-------------------------
        return_series = df[numeric_cols].sum()
        #-------------------------
        for i, list_col_i in enumerate(list_cols):
            if list_col_i in df.columns:
                joined_list_i = Utilities_df.consolidate_column_of_lists(df, list_col_i)
                return_series[list_col_i] = joined_list_i
                if update_list_counts_from_lists:
                    return_series[list_counts_cols[i]] = len(joined_list_i)

        # Order return_series as df is ordered
        order = [x for x in df.columns if x in return_series.index]
        return_series = return_series[order]
        #-------------------------
        return return_series

    @staticmethod
    def w_avg_df_numeric_cols_and_join_list_cols(
        df, 
        w_col, 
        list_cols=['_SNs'], 
        list_counts_cols=['_nSNs'], 
        numeric_cols=None, 
        update_list_counts_from_lists=True
    ):
        r"""
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        NEEDED FOR ASYNCHRONOUS CASE!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Suppose the general case, where outg_rec_nb split over multiple rows, and some SNs are repeated/split
          across multiple rows.
           e.g., df = 
           outg_rec_nb    reason_0    reason_1    ...    _SNs        _nSNs
           00000001        #_0_1       #_1_1            [1,2,3]         3
           00000001        #_0_2       #_1_2            [3,4,5,6]       4
           00000001        #_0_3       #_1_3            [4,5,6,7,8]     5

        Using the normal weighted average method, for reason_0 this would give
          agg_reason_0 = ((#_0_1*3) + (#_0_2*4) + (#_0_3*5))/(3+4+5)
        The point is to normalize by the number of serial numbers.  The above prescription would give the incorrect
          value, as there are not (3+4+5)=12 serial numbers, but only 8!
        The situation above is exactly why this function is needed.  The numerator of the above prescription is still correct,
          as this essentially calculates the raw values, given the norm values and the number of counts.
        However, the denominator will need to be calculated at a later point, after the number of unique SNs are tallied!

        This should work properly regardless of whether or not w_col is in list_counts_cols.
        -------------------------
        w_col:
          Column containing the weights.  Typically, this is one of the columns in list_counts_cols (e.g., '_nSNs')

        list_cols:
          Identify all the columns whose elements are lists

        list_counts_cols:
          Columns containing the counts of each list columns.
          There must be the same number of list_cols as list_counts_cols, and the ordering should be the same.

        numeric_cols:
          Identify all the numeric columns to be summed.  If this is left as None, it is taken to be
          all columns not in [w_col]+list_cols+list_counts_cols

        update_list_counts_from_lists:
          If True (default), update the columns in list_counts_cols with the lengths of the lists in lists_cols.
          If False, list_counts_cols will simply be summed just like numeric_cols.
          NOTE:  It is typically a good idea to have update_list_counts_from_lists=True.  Typically, one wants, e.g.,
                 the number of unique serial numbers.  If a serial number exists in two (or more) rows of the DF, it will
                 be double (or more) counted when update_list_counts_from_lists==False, and thus the counts will not reflect
                 the number of unique entries!
        """
        #-------------------------
        # If there is only one row in df, then no averaging/joining to be done, in which
        # case, simply return series version of df
        if df.shape[0]==1:
            return df.squeeze()
        #-------------------------
        # There must be the same number of list_cols as list_counts_cols, and the ordering should be the same.
        assert(len(list_cols)==len(list_counts_cols))
        #-------------------------
        if numeric_cols is None:
            numeric_cols = [x for x in df.columns.tolist() if x not in [w_col]+list_cols+list_counts_cols]
        #-------------------------
        # The weighted sum can be calculated like normal (see discussion in documentation above regarding the numerator 
        # still being correct)
        # However, the overall weighted average will need to be handled more carefully.
        # NOTE: Never want list_counts_cols to be weighted, hence the use of sum_and_weighted_sum_of_df_cols below instead
        #       of w_sum_df_cols
        return_series = Utilities_df.sum_and_weighted_sum_of_df_cols(
            df=df, 
            sum_x_cols=list_counts_cols, sum_w_col=None,
            wght_x_cols=numeric_cols, wght_w_col=w_col, 
            include_sum_of_weights=True
        )
        #-------------------------
        for i, list_col_i in enumerate(list_cols):
            if list_col_i in df.columns:
                joined_list_i = Utilities_df.consolidate_column_of_lists(df, list_col_i)
                return_series[list_col_i] = joined_list_i
                if update_list_counts_from_lists:
                    return_series[list_counts_cols[i]] = len(joined_list_i)
        #-------------------------
        return_series[numeric_cols] /= return_series[w_col]
        #-------------------------
        # Order return_series as df is ordered
        order = [x for x in df.columns if x in return_series.index]
        return_series = return_series[order]
        #-------------------------
        return return_series

    @staticmethod
    def sum_and_weighted_average_of_df_numeric_cols_and_join_list_cols(
        df, 
        sum_numeric_cols,  sum_w_col,  sum_list_cols,  sum_list_counts_cols, 
        wght_numeric_cols, wght_w_col, wght_list_cols, wght_list_counts_cols, 
        update_list_counts_from_lists=True
    ): 
        r"""
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        NEEDED FOR ASYNCHRONOUS CASE!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Suppose the general case, where outg_rec_nb split over multiple rows, and some SNs are repeated/split
          across multiple rows.
           e.g., df = 
           outg_rec_nb    reason_0    reason_1    ...    _SNs        _nSNs
           00000001        #_0_1       #_1_1            [1,2,3]         3
           00000001        #_0_2       #_1_2            [3,4,5,6]       4
           00000001        #_0_3       #_1_3            [4,5,6,7,8]     5

        Using the normal weighted average method, for reason_0 this would give
          agg_reason_0 = ((#_0_1*3) + (#_0_2*4) + (#_0_3*5))/(3+4+5)
        The point is to normalize by the number of serial numbers.  The above prescription would give the incorrect
          value, as there are not (3+4+5)=12 serial numbers, but only 8!
        The situation above is exactly why this function is needed.  The numerator of the above prescription is still correct,
          as this essentially calculates the raw values, given the norm values and the number of counts.
        However, the denominator will need to be calculated at a later point, after the number of unique SNs are tallied!
        """
        #-------------------------
        # If there is only one row in df, then no summing/averaging/joining to be done, in which
        # case, simply return series version of df
        if df.shape[0]==1:
            return df.squeeze()
        #-------------------------
        # There must be the same number of list_cols as list_counts_cols, and the ordering should be the same.
        assert(len(sum_list_cols) ==len(sum_list_counts_cols))
        assert(len(wght_list_cols)==len(wght_list_counts_cols))
        #-------------------------
        # Allow myself to be a little lazy by permitting sum_list_cols to be in sum_numeric_cols
        # (and, similarly, wght_SNs_col in wght_x_cols)
        # NOTE: The same is true for sum_w_col and wght_w_col
        sum_numeric_cols =  [x for x in sum_numeric_cols  if x not in [sum_w_col] +sum_list_cols +sum_list_counts_cols]
        wght_numeric_cols = [x for x in wght_numeric_cols if x not in [wght_w_col]+wght_list_cols+wght_list_counts_cols]
        #-------------------------
        wght_series = AMIEndEvents.w_avg_df_numeric_cols_and_join_list_cols(
            df=df, 
            w_col=wght_w_col, 
            list_cols=wght_list_cols, 
            list_counts_cols=wght_list_counts_cols, 
            numeric_cols=wght_numeric_cols, 
            update_list_counts_from_lists=update_list_counts_from_lists
        )    
        #-------------------------
        sum_series = AMIEndEvents.sum_numeric_cols_and_join_list_cols(
            df=df, 
            list_cols=sum_list_cols, 
            list_counts_cols=sum_list_counts_cols, 
            numeric_cols=sum_numeric_cols,
            update_list_counts_from_lists=update_list_counts_from_lists
        )
        sum_series[sum_w_col] = wght_series[wght_w_col]
        #-------------------------
        return_series = pd.concat([sum_series, wght_series])
        #-------------------------
        # Order return_series as df is ordered
        order = [x for x in df.columns if x in return_series.index]
        return_series = return_series[order]
        #-------------------------
        return return_series
                    
    @staticmethod                
    def combine_two_reason_counts_per_outage_dfs(
        rcpo_df_1, 
        rcpo_df_2, 
        are_dfs_wide_form, 
        normalize_by_nSNs_included, 
        is_norm, 
        list_cols=['_SNs'], 
        list_counts_cols=['_nSNs'],
        w_col = None, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm', 
        convert_rcpo_wide_to_long_col_args=None
    ):
        r"""
        Combine two reason_counts_per_outage_dfs.
        This is necessary for batch reading/building.
        
        NOTE: The resultant DF will contain the full union set of reasons observed in both, regardless of whether
              the dfs are wide or long form.
        
        The normalization status of both rcpo_df_1 and rcpo_df_2 should be the same!
        i.e., normalize_by_nSNs_included/is_norm should be the same between the two
        
        If normalize_by_nSNs_included is True, is_norm is ignored
              
        are_dfs_wide_form:
          The case where are_dfs_wide_form=True is pretty simple and straighforward.
          The case where are_dfs_wide_form=False and is_norm=False is also pretty simple and straighforward.
          However, the case where are_dfs_wide_form=False and is_norm=True is more involved.
          Therefore, the simplest solution for the more involved case is to first convert the long form DFs to wide form,
            perform the combination, and the convert back to long form.
          I believe I will be using the case are_dfs_wide_form in the vast majority of cases.
          
        Note: Elements of list_cols and list_counts_cols should be strings, even in the case of MultiIndex columns.
              If MultiIndex columns, the full column names will be compiled in the function
              
        w_col:
          The column to use as a weight when combining normalized rcpo_dfs (typically, '_nSNs').
          If not specified, it will be taken as list_counts_cols[0] (again, typicall '_nSNs')
        """
        #----------------------------------------------------------------------------------------------------
        # There must be the same number of list_cols as list_counts_cols, and the ordering should be the same
        assert(len(list_cols)==len(list_counts_cols))
        if w_col is None:
            w_col = list_counts_cols[0]
        #----------------------------------------------------------------------------------------------------
        if not are_dfs_wide_form:
            # More invovled case.  Solution is to (1) convert to wide form, (2) run through this function,
            #   and then (3) convert back to long form.
            #-------------------------
            # (1) convert to wide form
            rcpo_df_1 = AMIEndEvents.convert_reason_counts_per_outage_df_long_to_wide(rcpo_df_1)
            rcpo_df_2 = AMIEndEvents.convert_reason_counts_per_outage_df_long_to_wide(rcpo_df_2)
            #-------------------------
            # (2) run wide form through this function
            rcpo_full_wide = AMIEndEvents.combine_two_reason_counts_per_outage_dfs(
                rcpo_df_1=rcpo_df_1, 
                rcpo_df_2=rcpo_df_2, 
                are_dfs_wide_form=True, 
                normalize_by_nSNs_included=normalize_by_nSNs_included, 
                is_norm=is_norm, 
                list_cols=list_cols, 
                list_counts_cols=list_counts_cols,
                w_col = w_col, 
                level_0_raw_col=level_0_raw_col, 
                level_0_nrm_col=level_0_nrm_col, 
                convert_rcpo_wide_to_long_col_args=convert_rcpo_wide_to_long_col_args
            )
            #-------------------------
            # (3) convert back to long form
            dflt_convert_rcpo_wide_to_long_col_args = dict(        
                new_counts_col='counts', 
                new_counts_norm_col='counts_norm', 
                org_counts_col='counts', 
                org_counts_norm_col='counts_norm', 
                idx_0_name = 'outg_rec_nb'
            )
            if convert_rcpo_wide_to_long_col_args is None:
                convert_rcpo_wide_to_long_col_args = {}
            convert_rcpo_wide_to_long_col_args = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict=convert_rcpo_wide_to_long_col_args, 
                default_values_dict=dflt_convert_rcpo_wide_to_long_col_args, 
                extend_any_lists=False, inplace=True
            )
            if not normalize_by_nSNs_included and is_norm:
                convert_rcpo_wide_to_long_col_args['new_counts_col'] = convert_rcpo_wide_to_long_col_args['new_counts_norm_col']
                convert_rcpo_wide_to_long_col_args['org_counts_col'] = convert_rcpo_wide_to_long_col_args['org_counts_norm_col']
            rcpo_full = AMIEndEvents.convert_reason_counts_per_outage_df_wide_to_long(
                rcpo_df_wide=rcpo_full_wide, 
                normalize_by_nSNs_included=normalize_by_nSNs_included, 
                **convert_rcpo_wide_to_long_col_args
            )
            return rcpo_full
        #----------------------------------------------------------------------------------------------------
        #--------------------------------------------------
        AMIEndEvents.make_reason_counts_per_outg_columns_equal(rcpo_df_1, rcpo_df_2, same_order=True, inplace=True)
        #-------------------------
        rcpo_full = pd.concat([rcpo_df_1, rcpo_df_2])
        rcpo_full = rcpo_full[rcpo_full.columns.sort_values()]
        #-------------------------
        # NOTE: If rcpo_dfs have MultiIndex indices, the aggregation methods below (e.g., sum_numeric_cols_and_join_list_cols, 
        #         w_avg_df_numeric_cols_and_join_list_cols, etc.) will collapse these down to a single dimension.
        #       Therefore, keep track of index levels to restore later 
        assert(rcpo_df_1.index.nlevels==rcpo_df_2.index.nlevels)
        if rcpo_df_1.index.names==rcpo_df_2.index.names:
            idx_level_names = list(rcpo_df_1.index.names)
        else:
            idx_level_names = [f'idx_lvl_{i}' for i in range(rcpo_df_1.index.nlevels)]
        #-------------------------
        #TODO  w_col='_nSNs' need to be in parameters
        #TODO  sum_w_col and wght_w_col should be in parameters or made more general
        # Other uses of 'counts' and 'count_norm' as well
        if normalize_by_nSNs_included:
            # NOTE: Using sum_and_weighted_average_of_df_cols MUCH MUCH faster than 
            #       1. splitting up into raw and normal
            #       2. using rcpo_full_raw = rcpo_full_raw.groupby(rcpo_full_raw.index).sum() and
            #                rcpo_full_nrm = rcpo_full_nrm.groupby(rcpo_full_nrm.index).apply(Utilities_df.w_avg_df_cols, w_col=nSNs_col)
            #       3. Recombining into rcpo_full
            assert(rcpo_full.columns.nlevels==2)
            sum_numeric_cols = rcpo_full.columns[rcpo_full.columns.get_level_values(0)==level_0_raw_col]
            sum_w_col = (level_0_raw_col, w_col)
            sum_list_cols        = [(level_0_raw_col, x) for x in list_cols]
            sum_list_counts_cols = [(level_0_raw_col, x) for x in list_counts_cols]

            wght_numeric_cols = rcpo_full.columns[rcpo_full.columns.get_level_values(0)==level_0_nrm_col]
            wght_w_col = (level_0_nrm_col, w_col)
            wght_list_cols        = [(level_0_nrm_col, x) for x in list_cols]
            wght_list_counts_cols = [(level_0_nrm_col, x) for x in list_counts_cols]

            rcpo_full = rcpo_full.groupby(rcpo_full.index).apply(
                lambda x: AMIEndEvents.sum_and_weighted_average_of_df_numeric_cols_and_join_list_cols(
                    df=x, 
                    sum_numeric_cols=sum_numeric_cols,   sum_w_col=sum_w_col,   sum_list_cols=sum_list_cols,   sum_list_counts_cols=sum_list_counts_cols, 
                    wght_numeric_cols=wght_numeric_cols, wght_w_col=wght_w_col, wght_list_cols=wght_list_cols, wght_list_counts_cols=wght_list_counts_cols, 
                    update_list_counts_from_lists=True                    
                )
            )
        else:
            if rcpo_full.columns.nlevels>1:
                assert(rcpo_full.columns.nlevels==2)
                assert(rcpo_full.columns.get_level_values(0).nunique()==1)
                list_cols =        [(rcpo_full.columns.get_level_values(0).unique().tolist()[0], x) for x in list_cols]
                list_counts_cols = [(rcpo_full.columns.get_level_values(0).unique().tolist()[0], x) for x in list_counts_cols]
                w_col =             (rcpo_full.columns.get_level_values(0).unique().tolist()[0], w_col)
                
                nSNs_col = (rcpo_full.columns.get_level_values(0).unique().tolist()[0], nSNs_col)
                SNs_col = (rcpo_full.columns.get_level_values(0).unique().tolist()[0], SNs_col)
            #NOTE: No assert for list_cols (SNs_col) because this function is intended for use both with and without it
            assert(all([x in rcpo_full for x in list_counts_cols]))
            #-------------------------
            if not is_norm:
                rcpo_full = rcpo_full.groupby(rcpo_full.index).apply(
                    lambda x: AMIEndEvents.sum_numeric_cols_and_join_list_cols(
                        df=x, 
                        list_cols=list_cols, 
                        list_counts_cols=list_counts_cols, 
                        numeric_cols=None,
                        update_list_counts_from_lists=True
                    )
                )
            #-------------------------
            else:
                rcpo_full = rcpo_full.groupby(rcpo_full.index).apply(
                    lambda x: AMIEndEvents.w_avg_df_numeric_cols_and_join_list_cols(
                        df=x, 
                        w_col=w_col, 
                        list_cols=list_cols, 
                        list_counts_cols=list_counts_cols, 
                        numeric_cols=None, 
                        update_list_counts_from_lists=True
                    )
                )
        #--------------------------------------------------
        # If rcpo_dfs originally had MultiIndex indices, the procedure above flattened these out
        #   and they need to be restored
        if len(idx_level_names)>1:
            assert(len(rcpo_full.index[0])==len(idx_level_names))
            assert(Utilities.are_list_elements_lengths_homogeneous(rcpo_full.index.tolist(), len(idx_level_names)))
            rcpo_full = rcpo_full.set_index(pd.MultiIndex.from_tuples(rcpo_full.index, names=idx_level_names))        
        #--------------------------------------------------
        return rcpo_full
                    
    #****************************************************************************************************
    @staticmethod
    def build_ede_typeid_to_reason_df(
        end_events_df, 
        reason_col='reason', 
        ede_typeid_col='enddeviceeventtypeid'
    ):
        r"""
        Build a pd.DataFrame to allow the mapping of enddeviceeventtypeid to reason
        """
        ede_typeid_to_reason_df = end_events_df.groupby(ede_typeid_col)[reason_col].unique().to_frame()
        ede_typeid_to_reason_df[reason_col] = ede_typeid_to_reason_df[reason_col].apply(lambda x: list(x))
        return ede_typeid_to_reason_df

    @staticmethod
    def combine_two_ede_typeid_to_reason_dfs(
        ede_typeid_to_reason_df1, 
        ede_typeid_to_reason_df2,
        sort=False
    ):
        r"""
        """
        #-------------------------
        assert(ede_typeid_to_reason_df1.shape[1]==ede_typeid_to_reason_df2.shape[1]==1)
        assert(ede_typeid_to_reason_df1.columns[0]==ede_typeid_to_reason_df2.columns[0])
        # After calling Utilities_df.consolidate_column_of_lists, the column name is forgotten, so save here
        reason_col = ede_typeid_to_reason_df1.columns[0]
        #-------------------------
        ede_typeid_to_reason_df_full = pd.concat([ede_typeid_to_reason_df1, ede_typeid_to_reason_df2])
        ede_typeid_to_reason_df_full=ede_typeid_to_reason_df_full.groupby(ede_typeid_to_reason_df_full.index).apply(
            lambda x: Utilities_df.consolidate_column_of_lists(
                df=x, 
                col=reason_col, 
                sort=sort
            )
        )
        ede_typeid_to_reason_df_full=ede_typeid_to_reason_df_full.to_frame(name=reason_col)
        #-------------------------
        assert(len(set(ede_typeid_to_reason_df1.index.tolist()+ede_typeid_to_reason_df2.index.tolist()))==
               ede_typeid_to_reason_df_full.shape[0])
        #-------------------------
        return ede_typeid_to_reason_df_full 

    @staticmethod
    def invert_ede_typeid_to_reason_df(ede_typeid_to_reason_df):
        r"""
        Invert ede_typeid_to_reason_df to create a pd.DataFrame to allow the mapping of reason to enddeviceeventtypeid
        """
        #-------------------------
        assert(ede_typeid_to_reason_df.shape[1]==1)
        reason_col = ede_typeid_to_reason_df.columns[0]
        #-------------------------
        reason_to_ede_typeid_df = ede_typeid_to_reason_df.explode(column=reason_col)
        # Make a new column to house the ede_typeid (previously the index)
        reason_to_ede_typeid_df[reason_to_ede_typeid_df.index.name]=reason_to_ede_typeid_df.index
        reason_to_ede_typeid_df=reason_to_ede_typeid_df.set_index(reason_col)
        #-------------------------
        return reason_to_ede_typeid_df    
    
    #****************************************************************************************************
    @staticmethod
    def get_net_reason_counts_for_outages_from_rcpo_df(
        rcpo_df, 
        reason_col='reason', 
        normalize_by_nSNs_included=False,
        agg_types = [np.sum, np.mean], 
        SNs_reasons_contain='_SNs', 
        convert_rcpo_wide_to_long_col_args=None
    ):
        r"""
        This function is designed to work with long-format rcpo df WITHOUT list of SNs.
        If it is found to be wide format (judged by the number of levels in the index)
          rcpo_df will be converted to long form.
        If rcpo is found to contain _SNs, they will be removed.
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(rcpo_df, [pd.DataFrame, dict]))
        if isinstance(rcpo_df, dict):
            return_dict = {}
            if convert_rcpo_wide_to_long_col_args is None:
                counts_col = 'counts'
            else:
                counts_col = convert_rcpo_wide_to_long_col_args.get(new_counts_col, 'counts')
            for key,df in rcpo_df.items():
                assert(key not in return_dict)
                net_reason_counts_df_i = AMIEndEvents.get_net_reason_counts_for_outages_from_rcpo_df(
                    rcpo_df=df, 
                    reason_col=reason_col, 
                    normalize_by_nSNs_included=False,
                    agg_types=agg_types, 
                    SNs_reasons_contain=SNs_reasons_contain, 
                    convert_rcpo_wide_to_long_col_args=convert_rcpo_wide_to_long_col_args
                )
                # Below sets column equal to key (this is needed so counts_norm does not come back simply as counts)
                assert(net_reason_counts_df_i.columns.nlevels<=2)
                if net_reason_counts_df_i.columns.nlevels==1:
                    assert(len(net_reason_counts_df_i.columns)==1)
                    net_reason_counts_df_i.columns=key
                else:
                    assert(net_reason_counts_df_i.columns.get_level_values(0).nunique()==1)
                    net_reason_counts_df_i.columns = net_reason_counts_df_i.columns.set_levels([key], level=0)
                return_dict[key] = net_reason_counts_df_i
            return return_dict
        #-------------------------
        rcpo_df = rcpo_df.copy()
        # This function is designed to work with long-format rcpo df
        # If the index only has a single level, it is assumed to be wide-format,
        # and will be converted.
        if rcpo_df.index.nlevels==1:
            #----------
            dflt_convert_rcpo_wide_to_long_col_args = dict(        
                new_counts_col='counts', 
                new_counts_norm_col='counts_norm', 
                org_counts_col='counts', 
                org_counts_norm_col='counts_norm', 
                idx_0_name = 'outg_rec_nb'
            )
            if convert_rcpo_wide_to_long_col_args is None:
                convert_rcpo_wide_to_long_col_args = {}
            convert_rcpo_wide_to_long_col_args = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict=convert_rcpo_wide_to_long_col_args, 
                default_values_dict=dflt_convert_rcpo_wide_to_long_col_args, 
                extend_any_lists=False, inplace=True
            )
            #----------
            rcpo_df = AMIEndEvents.convert_reason_counts_per_outage_df_wide_to_long(
                rcpo_df_wide=rcpo_df, 
                normalize_by_nSNs_included=normalize_by_nSNs_included, 
                **convert_rcpo_wide_to_long_col_args
            )
        #-------------------------
        agg_dict = {'counts':agg_types}
        if normalize_by_nSNs_included:
            agg_dict['counts_norm'] = agg_types
        #-------------------------
        # The aggregation will not work if the list of SNs is included, so any such entries must be removed.
        rcpo_df = rcpo_df.iloc[~rcpo_df.index.get_level_values(1).str.contains(SNs_reasons_contain)]
        #-------------------------
        reason_counts_df = rcpo_df.groupby(reason_col).agg(agg_dict)
        #-------------------------
        return reason_counts_df
    
    @staticmethod                
    def get_net_reason_counts_for_outages_from_end_events_df(
        end_events_df, 
        outg_rec_nb_col='outg_rec_nb', 
        serial_number_col='serialnumber', 
        reason_col='reason', 
        include_normalize_by_nSNs=False,
        include_nSNs=True,
        agg_types = [np.sum, np.mean]
    ):
        # inclue_zero_counts must be True to get correct values for mean etc
        #   - sum values would be same either way
        return_form = dict(return_multiindex_outg_reason = True, 
                           return_normalized_separately  = False)
        reason_counts_per_outg_df = AMIEndEvents.get_reason_counts_per_outage(
            end_events_df = end_events_df, 
            outg_rec_nb_col=outg_rec_nb_col, 
            serial_number_col=serial_number_col, 
            reason_col=reason_col, 
            include_normalize_by_nSNs=include_normalize_by_nSNs, 
            inclue_zero_counts=True,
            possible_reasons=None, 
            include_nSNs=include_nSNs, 
            include_SNs=False, 
            prem_nb_col='aep_premise_nb', 
            include_nprem_nbs=False,
            include_prem_nbs=False, 
            return_form = return_form
        )
        #-------------------------
        if include_normalize_by_nSNs and not return_form['return_normalized_separately']:
            normalize_by_nSNs_included=True
        else:
            normalize_by_nSNs_included=False
        return AMIEndEvents.get_net_reason_counts_for_outages_from_rcpo_df(
            rcpo_df=reason_counts_per_outg_df, 
            reason_col=reason_col, 
            normalize_by_nSNs_included=normalize_by_nSNs_included,
            agg_types = agg_types            
        )
    

    @staticmethod
    def find_summary_file_from_csv(csv_path):
        r"""
        Simple function to find summary file from csv file.
        e.g., from 'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data\EndEvents_NoOutg\end_events_0.csv'
              find 'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data\EndEvents_NoOutg\summary_files\end_events_0_summary.json'
              
        If more flexbile function is needed, one should re-develop using e.g., Utilities.find_all_paths
        """
        #-------------------------
        parent_dir = Path(csv_path).parent
        file_name = os.path.basename(csv_path)
        assert(os.path.join(parent_dir, file_name)==csv_path)
        #-------------------------
        summary_file_name = Utilities.append_to_path(save_path=file_name, appendix='_summary', ext_to_find='.csv')
        summary_file_name =  summary_file_name.replace('.csv', '.json')
        #-----
        summary_path = os.path.join(parent_dir, 'summary_files', summary_file_name)
        assert(os.path.exists(summary_path))
        return summary_path

    @staticmethod
    def get_summary_paths_for_data_in_dir(
        data_dir, 
        file_path_glob=r'end_events_[0-9]*.csv', 
        return_dict=False
    ):
        r"""
        data_dir should point to the directory containing the actual data CSV files.
        It is expected that the summary files live in os.path.join(data_dir, 'summary_files')
        """
        #-------------------------
        assert(os.path.isdir(data_dir))
        data_paths = Utilities.find_all_paths(base_dir=data_dir, glob_pattern=file_path_glob)
        assert(len(data_paths)>0)
        #-----
        summary_paths = [AMIEndEvents.find_summary_file_from_csv(x) for x in data_paths]
        assert(len(summary_paths)>0)
        assert(len(summary_paths)==len(data_paths))
        #-------------------------
        if return_dict:
            return dict(zip(data_paths, summary_paths))
        return summary_paths