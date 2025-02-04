#!/usr/bin/env python

r"""
Holds MECPOCollection class.  See MECPOCollection.MECPOCollection for more information.
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
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
from scipy import stats
import datetime
import time
from natsort import natsorted, ns
import warnings
import copy

from functools import reduce
#--------------------------------------------------
import CommonLearningMethods as clm
from AMIEndEvents_SQL import AMIEndEvents_SQL
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
from MECPODf import MECPODf, OutageDType
from MECPOAn import MECPOAn
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df


class MECPOCollection:
    r"""
    A collection of MECPOAn objects.
    Originally built mainly to help facilitate the plotting process.
    However, will likely be useful in creating datasets for modelling.
    """
    
    def __init__(
        self, 
        data_type, 
        mecpo_coll, 
        coll_label, 
        barplot_kwargs_shared=None, 
        read_and_load_all_pickles=False, 
        pkls_base_dir=None,
        days_min_max_outg_td_windows=None, 
        pkls_sub_dirs=None,
        naming_tag=None,
        normalize_by_time_interval=False, 
        are_no_outg=False
    ):
        r"""
        data_type:
            self.data_type will eventually be of type OutageDType.
            However, the input input_data_type may be any of the following:
              OutageDType ==> self.data_type set directly
              int         ==> value must be within [e.value for e in OutageDType], but not equal to OutageDType['unset'].value
                              Converted via self.data_type = OutageDType(input_data_type)
              str         ==> value must be within [e.name for e in OutageDType], but not equal to 'unset'
                              Converted via self.data_type = OutageDType[input_data_type]
        
        mecpo_coll:
          A collection of MECPOAn objects.
          The intent was for this collection to be a dictionary.
            In this case, the keys should be analysis identifiers, and the values should be MECPOAn object.
            These keys will be used as title for subplot (e.g., see draw_cpo_dfs_full_vs_direct_3x2)
          However, it can also be passed as a list, in which case integer keys will be assigned.
          
        coll_label:
          Label to be used for the collection when plotting.  This will override any label set in barplot_kwargs_shared
          
        barplot_kwargs_shared:
          Barplot kwargs which will be used for all members of collection.  Individual members can add to or alter these
        ------------------------- 
        IF mecpo_coll IS NOT SUPPLIED (i.e., if mecpo_coll is None):
        read_and_load_all_pickles:
          If True, the MECPOAn object will be built using the parameters:
            pkls_base_dir:
                The base directory in which the file directories reside.
            days_min_max_outg_td_windows/pkls_sub_dirs:
                These identify the subdirectories within which the pickle files should reside.
                Only one or the other should be set, NOT BOTH.
                #-----
                days_min_max_outg_td_windows:
                    Should be a tuples each with two int values, 0th element for the min_outg_td_window, and the 1st element 
                    for max_outg_td_window
                    Each sub-directory in which the pickle files should reside will be:
                        f'outg_td_window_{days_min_max_outg_td_windows[i][0]}_to_{days_min_max_outg_td_windows[i][1]}_days'
                pkls_sub_dirs:
                    Should be a list of strings each representing the name of the sub-directory in which the files reside.
                
            naming_tag:
              Common tag identifier for all files of this type.
                e.g., for entire outage collection available, naming_tag is None
                e.g., for direct meters, naming_tag = '_prim_strict'
                e.g., for no outages sample, naming_tag = '_no_outg'
                e.g., rcpo_df_OG will be expected to be located at the following path:
                  os.path.join(pkls_dir, f'rcpo{naming_tag}_df_OG.pkl')
            normalize_by_time_interval:
              If True, the DFs are normalized by the time interval (taken from days_min_max_outg_td_window)
              If days_min_max_outg_td_window is None, this is set to False
            are_no_outg:
                Direct which DFs to load
        ------------------------- 
        """
        #-------------------------
        self.data_type             = None
        self.mecpo_coll_dict       = None
        self.cpx_pkls_dirs         = None
        self.grp_by                = None
        self.red_to_org_cols_dicts = None
        self.time_infos_df         = None
        self.coll_label            = None
        self.barplot_kwargs_shared = None
        
        #-------------------------
        self.set_data_type(input_data_type = data_type)
        #-------------------------
        if mecpo_coll is None and read_and_load_all_pickles:
            mecpo_coll = MECPOCollection.build_mecpo_coll_dict_from_pkls(
            data_type                        = self.data_type, 
                pkls_base_dir                = pkls_base_dir, 
                days_min_max_outg_td_windows = days_min_max_outg_td_windows, 
                pkls_sub_dirs                = pkls_sub_dirs,
                naming_tag                   = naming_tag,
                normalize_by_time_interval   = normalize_by_time_interval, 
                are_no_outg                  = are_no_outg, 
                use_subdirs_as_mecpo_an_keys = True
            )
            self.cpx_pkls_dirs = {}
            for mecpo_an_key_i, mecpo_an_i in mecpo_coll.items():
                assert(mecpo_an_key_i not in self.cpx_pkls_dirs.keys())
                self.cpx_pkls_dirs[mecpo_an_key_i] = mecpo_an_i.cpx_pkls_dir
        #-------------------------
        assert(Utilities.is_object_one_of_types(mecpo_coll, [list, dict]))
        if isinstance(mecpo_coll, list):
            self.mecpo_coll_dict = {i:mecpo for i,mecpo in enumerate(mecpo_coll)}
        else:
            self.mecpo_coll_dict = copy.deepcopy(mecpo_coll)
        #-----
        self.grp_by = self.get_grp_by()
        #-------------------------
        self.coll_label=coll_label
        if barplot_kwargs_shared is None:
            self.barplot_kwargs_shared = {}
        else:
            self.barplot_kwargs_shared=barplot_kwargs_shared
        self.barplot_kwargs_shared['label'] = self.coll_label
        #-------------------------

        
    def set_data_type(
        self, 
        input_data_type
    ):
        r"""
        self.data_type will eventually be of type OutageDType.
        However, the input input_data_type may be any of the following:
          OutageDType ==> self.data_type set directly
          int         ==> value must be within [e.value for e in OutageDType], but not equal to OutageDType['unset'].value
                          Converted via self.data_type = OutageDType(input_data_type)
          str         ==> value must be within [e.name for e in OutageDType], but not equal to 'unset'
                          Converted via self.data_type = OutageDType[input_data_type]
        """
        #-------------------------
        self.data_type = MECPOAn.get_data_type(input_data_type=input_data_type)
        
    def get_grp_by(self):
        r"""
        All MECPOAn objects should have the same grp_by value.
        Make sure this is so, and return it
        """
        #-------------------------
        grp_by_vals = []
        for an_key_i, mecpo_an_i in self.mecpo_coll_dict.items():
            grp_by_vals.append(mecpo_an_i.grp_by)
        assert(len(set(grp_by_vals))==1)
        return grp_by_vals[0]
        
    def get_cpx_pkls_base_dir(self):
        r"""
        Expected that all MECPOAn objects share the same base directory
        """
        #-------------------------
        if self.cpx_pkls_dirs is None:
            return None
        #-------------------------
        cpx_pkls_dirs = list(self.cpx_pkls_dirs.values())
        #-----
        pkls_base_dir = list(set([Path(x).parent for x in cpx_pkls_dirs]))
        assert(len(pkls_base_dir)==1)
        return str(pkls_base_dir[0])
        
    def get_base_dir(self):
        r"""
        """
        #-------------------------
        pkls_base_dir = self.get_cpx_pkls_base_dir()
        base_dir = Path(pkls_base_dir).parent
        assert(os.path.isdir(base_dir))
        return str(base_dir)
        
    def get_end_events_dir(self):
        r"""
        """
        #-------------------------
        base_dir = self.get_base_dir()
        end_events_dir = os.path.join(base_dir, 'EndEvents')
        assert(os.path.isdir(end_events_dir))
        return end_events_dir
        
    @staticmethod
    def build_mecpo_coll_dict_from_pkls(
        data_type, 
        pkls_base_dir, 
        days_min_max_outg_td_windows=None, 
        pkls_sub_dirs=None,
        naming_tag=None,
        normalize_by_time_interval=False, 
        are_no_outg=False, 
        use_subdirs_as_mecpo_an_keys=True
    ):
        r"""
        data_type:
            However, the input data_type may be any of the following:
              OutageDType ==> data_type set directly
              int         ==> value must be within [e.value for e in OutageDType], but not equal to OutageDType['unset'].value
                              Converted via self.data_type = OutageDType(data_type)
              str         ==> value must be within [e.name for e in OutageDType], but not equal to 'unset'
                              Converted via self.data_type = OutageDType[data_type]
        
        pkls_base_dir:
            The base directory in which the file directories reside.
        days_min_max_outg_td_windows/pkls_sub_dirs:
            Should be a list of tuples (days_min_max_outg_td_windows) or a list of strings (pkls_sub_dirs)
            Each element in the list identifies the subdirectory within which the pickle files should reside.
            If both are not None, the subdirectories to search is taken from pkls_sub_dirs, and days_min_max_outg_td_windows will
              only be used if normalize_by_time_interval==True
            If one is None and the other is not, the latter will be used to identify the subdirectories.
            #-----
            days_min_max_outg_td_windows:
                Should be a tuples each with two int values, 0th element for the min_outg_td_window, and the 1st element 
                for max_outg_td_window
                Each sub-directory in which the pickle files should reside will be:
                    f'outg_td_window_{days_min_max_outg_td_windows[i][0]}_to_{days_min_max_outg_td_windows[i][1]}_days'
            pkls_sub_dirs:
                Should be a list of strings each representing the name of the sub-directory in which the files reside.

        naming_tag:
          Common tag identifier for all files of this type.
            e.g., for entire outage collection available, naming_tag is None
            e.g., for direct meters, naming_tag = '_prim_strict'
            e.g., for no outages sample, naming_tag = '_no_outg'
            e.g., rcpo_df_OG will be expected to be located at the following path:
              os.path.join(pkls_dir, f'rcpo{naming_tag}_df_OG.pkl')
        normalize_by_time_interval:
          If True, the DFs are normalized by the time interval (taken from days_min_max_outg_td_window)
        are_no_outg:
            Direct which DFs to load

        use_subdirs_as_mecpo_an_keys:
            If True, the mecpo_an_keys will be the subdirs
            If False, the mecpo_an_keys will be ints
        """
        #-------------------------
        assert(days_min_max_outg_td_windows is not None or pkls_sub_dirs is not None)
        #-----
        if naming_tag is None:
            naming_tag=''
        #-----    
        if normalize_by_time_interval:
            assert(days_min_max_outg_td_windows is not None)
        #-------------------------
        if pkls_sub_dirs is None:
            assert(days_min_max_outg_td_windows is not None)
            pkls_sub_dirs = []
            for days_min_max_outg_td_window in days_min_max_outg_td_windows:
                pkls_subdir = f'outg_td_window_{days_min_max_outg_td_window[0]}_to_{days_min_max_outg_td_window[1]}_days'
                pkls_sub_dirs.append(pkls_subdir)
        #-------------------------
        mecpo_coll = {}
        # NOTE: For the case where pkls_sub_dirs input was None (so days_min_max_outg_td_windows is used solely), even though 
        #         pkls_sub_dirs was built above, it still needs to be supplied below in case normalize_by_time_interval==True
        #       For this to work, at this poing, days_min_max_outg_td_windows must always be a list with length equal to 
        #         that of pkls_sub_dirs
        if days_min_max_outg_td_windows is None:
            days_min_max_outg_td_windows = [None]*len(pkls_sub_dirs)
        assert(len(pkls_sub_dirs)==len(days_min_max_outg_td_windows))
        for i, (pkls_subdir, days_min_max_outg_td_window) in enumerate(zip(pkls_sub_dirs, days_min_max_outg_td_windows)):
            mecpo_an_i = MECPOAn(
                data_type                   = data_type, 
                pkls_base_dir               = pkls_base_dir, 
                days_min_max_outg_td_window = days_min_max_outg_td_window, 
                pkls_sub_dir                = pkls_subdir,
                naming_tag                  = naming_tag,
                normalize_by_time_interval  = normalize_by_time_interval, 
                is_no_outg                  = are_no_outg, 
                outg_rec_nb_col             = 'index', 
                read_and_load_all_pickles   = True, 
                init_cpo_dfs_to             = pd.DataFrame() # Should be pd.DataFrame() or None
            )
            #-----
            if use_subdirs_as_mecpo_an_keys:
                mecpo_an_key_i = pkls_subdir
            else:
                mecpo_an_key_i = i
            #-----
            assert(mecpo_an_key_i not in mecpo_coll.keys())
            mecpo_coll[mecpo_an_key_i] = mecpo_an_i
        #-------------------------
        return mecpo_coll
            
            
    @property
    def n_mecpo_ans(self):
        return len(self.mecpo_coll_dict)
    
    @property
    def mecpo_an_keys(self):
        return list(self.mecpo_coll_dict.keys())
    
    @property
    def all_cpo_df_names(self):
        return self.get_all_cpo_df_names(shared_by_all=False)
        
    @property
    def cpo_df_names(self):
        return self.all_cpo_df_names
    
    @property
    def shared_cpo_df_names(self):
        return self.get_all_cpo_df_names(shared_by_all=True)
        
    def change_mecpo_an_keys(
        self, 
        old_to_new_keys_dict
    ):
        r"""
        Change the mecpo_an_keys using old_to_new_keys_dict
        """
        #-------------------------
        for old_key, new_key in old_to_new_keys_dict.items():
            assert(old_key in self.mecpo_coll_dict.keys())
            self.mecpo_coll_dict[new_key] = self.mecpo_coll_dict.pop(old_key)
            
    def get_mecpo_an(self, mecpo_an_key):
        r"""
        Get MECPOAn object by mecpo_an_key
        """
        #-------------------------
        assert(mecpo_an_key in self.mecpo_coll_dict.keys())
        return self.mecpo_coll_dict[mecpo_an_key]
        
    def get_mecpo_ans(self, order=None):
        r"""
        Return a list of the MECPOAn objects (stored in self.mecpo_coll_dict) in the order specified by order
        order:
          Should be a list of valid mecpo_an_keys
          If order is None, the order will be taken as self.mecpo_coll_dict.keys()
        """
        #-------------------------
        if order is None:
            order = self.mecpo_an_keys
        #-------------------------
        # Make sure all mecpo_an_keys in order are valid
        assert(all([x in self.mecpo_an_keys for x in order]))
        #-------------------------
        return_ans = []
        for mecpo_an_key in order:
            return_ans.append(self.mecpo_coll_dict[mecpo_an_key])
        #-------------------------
        return return_ans
    
    def get_all_cpo_df_names(self, shared_by_all=True):
        r"""
        Get all cpo_df names contained in all MECPOAn objects of the collection.
        If shared_by_all==True, only those names shared by all MECPOAn objects are returned.
        """
        #-------------------------
        cpo_df_names_by_an = []
        for an_id, mecpo_an in self.mecpo_coll_dict.items():
            cpo_df_names_by_an.append(list(mecpo_an.cpo_dfs.keys()))
        #-------------------------
        all_cpo_df_names = []
        if shared_by_all:
            for i,names in enumerate(cpo_df_names_by_an):
                if i==0:
                    all_cpo_df_names = names
                else:
                    all_cpo_df_names = set(all_cpo_df_names).intersection(names)
            all_cpo_df_names = list(all_cpo_df_names)
        else:
            for names in cpo_df_names_by_an:
                all_cpo_df_names.extend(names)
            all_cpo_df_names = list(set(all_cpo_df_names))
        #-------------------------
        return all_cpo_df_names
    
    def get_cpo_df(
        self,
        mecpo_an_key,
        cpo_df_name, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        max_total_counts_args=None, 
        add_an_key_as_level_0_col=False
    ):
        r"""
        mecpo_an_key:
          The MECPOAn key, chooses from which MECPOAn in self.mecpo_coll_dict to grab the DF
        
        cpo_df_name:
          The name of the cpo pd.DataFrames, used to retrieve the correct DF from self.mecpo_coll_dict[mecpo_an_key]
          
        cpo_df_subset_by_mjr_mnr_cause_args:
          Allows one to select subsets of the pd.DataFrames by the outage type.
          This should be a dict with arguments appropriate for the MECPOAn.get_cpo_df_subset_by_mjr_mnr_cause
            function (except for the cpo_df_name argument, which will be set to cpo_df_name)
            
        max_total_counts_args:
          Allows one to further select a subset by the maximum number of total counts.
          This should be a dict with arguments appropriate for the MECPODf.get_cpo_df_subset_below_max_total_counts
            function (except for the cpo_df argument, which will be set to return_df
            *** Actually, can also be a simple int/float, which is interpreted as the max_total_counts argument for
                MECPODf.get_cpo_df_subset_below_max_total_counts (all other arguments are set to defaults)
            
        add_an_key_as_level_0_col:
          If True, cpo_df will always be returned with MultiIndex columns where the level 0
            value is equal to mecpo_an_key.
            NOTE: If columns previously normal Index, returned will be MultiIndex with 2 level
                  If columns previously MultiIndex, one additional level will be preprended.
        """
        #-------------------------
        assert(mecpo_an_key in self.mecpo_coll_dict)
        assert(cpo_df_name in self.mecpo_coll_dict[mecpo_an_key].cpo_dfs)
        #-------------------------
        cpo_df = self.mecpo_coll_dict[mecpo_an_key].get_cpo_df(
            cpo_df_name=cpo_df_name, 
            cpo_df_subset_by_mjr_mnr_cause_args=cpo_df_subset_by_mjr_mnr_cause_args, 
            max_total_counts_args=max_total_counts_args
        )
        #-------------------------
        if add_an_key_as_level_0_col:
            cpo_df = Utilities_df.prepend_level_to_MultiIndex(
                df=cpo_df, 
                level_val=mecpo_an_key, 
                level_name='mecpo_an_key', 
                axis=1
            )
        #-------------------------
        return cpo_df
    
    def get_cpo_dfs_dict(
        self, 
        cpo_df_name, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        max_total_counts_args=None, 
        add_an_key_as_level_0_col=False
    ):
        r"""
        Returns a dict object where the keys are the mecpo_an_key and the values are the 
          corresponding cpo_df
          
        cpo_df_name:
          The name of the cpo pd.DataFrames, used to retrieve the correct item from self.mecpo_coll_dict
          
        cpo_df_subset_by_mjr_mnr_cause_args:
          Allows one to select subsets of the pd.DataFrames by the outage type.
          This should be a dict with arguments appropriate for the MECPOAn.get_cpo_df_subset_by_mjr_mnr_cause
            function (except for the cpo_df_name argument, which will be set to cpo_df_name)
            
        max_total_counts_args:
          Allows one to further select a subset by the maximum number of total counts.
          This should be a dict with arguments appropriate for the MECPODf.get_cpo_df_subset_below_max_total_counts
            function (except for the cpo_df argument, which will be set to return_df
            *** Actually, can also be a simple int/float, which is interpreted as the max_total_counts argument for
                MECPODf.get_cpo_df_subset_below_max_total_counts (all other arguments are set to defaults)
        """
        #-------------------------
        cpo_dfs_dict = {}
        for an_key in list(self.mecpo_coll_dict.keys()):
            assert(an_key not in cpo_dfs_dict)
            cpo_dfs_dict[an_key] = self.get_cpo_df(
                mecpo_an_key=an_key,
                cpo_df_name=cpo_df_name, 
                cpo_df_subset_by_mjr_mnr_cause_args=cpo_df_subset_by_mjr_mnr_cause_args, 
                max_total_counts_args=max_total_counts_args, 
                add_an_key_as_level_0_col=add_an_key_as_level_0_col
            )
        #-------------------------
        return cpo_dfs_dict
    
    def get_cpo_dfs(
        self, 
        cpo_df_name, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        max_total_counts_args=None, 
        mecpo_an_order=None, 
        add_an_key_as_level_0_col=False
    ):
        r"""
        cpo_df_name:
          The name of the cpo pd.DataFrames, used to retrieve the correct item from self.mecpo_coll_dict
          
        cpo_df_subset_by_mjr_mnr_cause_args:
          Allows one to select subsets of the pd.DataFrames by the outage type.
          This should be a dict with arguments appropriate for the MECPOAn.get_cpo_df_subset_by_mjr_mnr_cause
            function (except for the cpo_df_name argument, which will be set to cpo_df_name)
            
        max_total_counts_args:
          Allows one to further select a subset by the maximum number of total counts.
          This should be a dict with arguments appropriate for the MECPODf.get_cpo_df_subset_below_max_total_counts
            function (except for the cpo_df argument, which will be set to return_df
            *** Actually, can also be a simple int/float, which is interpreted as the max_total_counts argument for
                MECPODf.get_cpo_df_subset_below_max_total_counts (all other arguments are set to defaults)
          
        mecpo_an_order:
          The ordering of the analyses before returning the DFs.  
          Should be a list containing the keys in self.mecpo_coll_dict
          If None, the order is simply what is returned from self.mecpo_coll_dict.keys()
        """
        #-------------------------
        cpo_dfs_dict = self.get_cpo_dfs_dict(
            cpo_df_name=cpo_df_name, 
            cpo_df_subset_by_mjr_mnr_cause_args=cpo_df_subset_by_mjr_mnr_cause_args, 
            max_total_counts_args=max_total_counts_args, 
            add_an_key_as_level_0_col=add_an_key_as_level_0_col
        )
        #-------------------------
        if mecpo_an_order is None:
            mecpo_an_order = list(self.mecpo_coll_dict.keys())
        #-------------------------
        cpo_dfs = []
        for an_key in mecpo_an_order:
            cpo_dfs.append(cpo_dfs_dict[an_key])
        #-------------------------
        return cpo_dfs


    #-----------------------------------------------------------------------------------------------------------------------------
    def build_time_infos_df(
        self, 
        save_to_pkl=False, 
        build_all=False
    ):
        r"""
        In most cases, one should use MECPOCollection.build_or_load_time_infos_df
    
        build_all:
            If True, the time_infos_df will be built for each MECPOAn, regardless of whether it's needed.
            I'm not sure when one would want this set to True?
        """
        #--------------------------------------------------
        # FOR BASELINE DATA
        #   If all of the MECPOAn objects have the same end_events_dir (which is often the case) then their time_infos_df objects
        #     will be the same, so only one needs to be grabbed.
        #   In general, the number of time_infos_dfs to be built/grabbed will be equal to the number of unique ede_data_dirs_dicts
        # FOR OUTAGE DATA
        #   If being built using DOVS (which it currently is), one should always build for all, as some outages may not have data for certain
        #     time periods, resulting in the different analyses having non-identical time_infos_dfs (although, they should be very similar!)
        #   If these were built using the summary file method, then this wouldn't be necessary.
        #-------------------------
        assert(isinstance(self.data_type, OutageDType) and self.data_type!=OutageDType.unset)
        #-------------------------
        ede_data_dirs_dict = {an_key_i:mecpo_an_i.end_events_dir for an_key_i,mecpo_an_i in self.mecpo_coll_dict.items()}
        unq_dirs = list(set(ede_data_dirs_dict.values()))
        #-------------------------
        if self.data_type==OutageDType.otbl or self.data_type==OutageDType.prbl:
            if build_all:
                mecpo_ans_to_build = list(ede_data_dirs_dict.keys())
            else:
                # Below, e.g., if ede_data_dirs_dict = dict(an_1='dir_1', an_2='dir_2', an_3='dir_1', an_4='dir_1', an_5='dir_2', an_6='dir_3')
                #   ==> mecpo_ans_to_build = ['an_1', 'an_2', 'an_6']
                mecpo_ans_to_build = [list(ede_data_dirs_dict.keys())[list(ede_data_dirs_dict.values()).index(x)] for x in unq_dirs]
                #-----
                # Sanity checks
                assert(len(mecpo_ans_to_build) == len(unq_dirs))
                assert(set([ede_data_dirs_dict[x] for x in mecpo_ans_to_build]) == set(unq_dirs))
        else:
            mecpo_ans_to_build = list(ede_data_dirs_dict.keys())
        #-------------------------
        time_infos_df = []
        for an_key_i in mecpo_ans_to_build:
            self.mecpo_coll_dict[an_key_i].build_time_infos_df(save_to_pkl=save_to_pkl)
            time_infos_df.append(self.mecpo_coll_dict[an_key_i].time_infos_df)
        time_infos_df = pd.concat(time_infos_df, axis=0)
        #-------------------------
        # Want to drop duplicate rows, but the index values must also be considered in comparisons, hence the need for
        #   reset_index() and then set_index() below
        # NOTE: Cannot call drop_duplicates() on any column with list elements, so these must be excluded!
        # NOTE ALSO: The reset_index() and set_index() method below relies on the index levels being named, so ensure this is so!
        time_infos_df.index.names = [name_i if name_i else f'idx_{i}' for i,name_i in enumerate(time_infos_df.index.names)]
        cols_w_lists = Utilities_df.get_df_cols_with_list_elements(df = time_infos_df)
        if len(cols_w_lists)==0:
            subset=None
        else:
            subset = [x for x in time_infos_df.columns.tolist() if x not in cols_w_lists]
        #-----
        # Need to include the index level names in subset, since .reset_index() is called before drop_duplicates
        if subset is not None:
            subset.extend(list(time_infos_df.index.names))
        #-----
        time_infos_df = time_infos_df.reset_index().drop_duplicates(subset=subset).set_index(time_infos_df.index.names)
        self.time_infos_df = time_infos_df


    def build_or_load_time_infos_df(
        self, 
        save_to_pkl=False, 
        verbose=True
    ):
        r"""
        """
        #--------------------------------------------------
        # FOR BASELINE DATA
        #   If all of the MECPOAn objects have the same end_events_dir (which is often the case) then their time_infos_df objects
        #     will be the same, so only one needs to be grabbed.
        #   In general, the number of time_infos_dfs to be built/grabbed will be equal to the number of unique ede_data_dirs_dicts
        # FOR OUTAGE DATA
        #   If being built using DOVS (which it currently is), one should always build for all, as some outages may not have data for certain
        #     time periods, resulting in the different analyses having non-identical time_infos_dfs (although, they should be very similar!)
        #   If these were built using the summary file method, then this wouldn't be necessary.
        #-------------------------
        assert(isinstance(self.data_type, OutageDType) and self.data_type!=OutageDType.unset)
        #-------------------------
        ede_data_dirs_dict = {an_key_i:mecpo_an_i.end_events_dir for an_key_i,mecpo_an_i in self.mecpo_coll_dict.items()}
        unq_dirs = list(set(ede_data_dirs_dict.values()))
        #-------------------------
        if self.data_type==OutageDType.otbl or self.data_type==OutageDType.prbl:
            # Below, e.g., if ede_data_dirs_dict = dict(an_1='dir_1', an_2='dir_2', an_3='dir_1', an_4='dir_1', an_5='dir_2', an_6='dir_3')
            #   ==> mecpo_ans_to_build = ['an_1', 'an_2', 'an_6']
            mecpo_ans_to_build = [list(ede_data_dirs_dict.keys())[list(ede_data_dirs_dict.values()).index(x)] for x in unq_dirs]
            #-----
            # Sanity checks
            assert(len(mecpo_ans_to_build) == len(unq_dirs))
            assert(set([ede_data_dirs_dict[x] for x in mecpo_ans_to_build]) == set(unq_dirs))
        else:
            mecpo_ans_to_build = list(ede_data_dirs_dict.keys())
        #-------------------------
        time_infos_df = []
        for an_key_i in mecpo_ans_to_build:
            self.mecpo_coll_dict[an_key_i].build_or_load_time_infos_df(save_to_pkl=save_to_pkl, verbose=verbose)
            time_infos_df.append(self.mecpo_coll_dict[an_key_i].time_infos_df)
        time_infos_df = pd.concat(time_infos_df, axis=0)
        #-------------------------
        # Want to drop duplicate rows, but the index values must also be considered in comparisons, hence the need for
        #   reset_index() and then set_index() below
        # NOTE: Cannot call drop_duplicates() on any column with list elements, so these must be excluded!
        # NOTE ALSO: The reset_index() and set_index() method below relies on the index levels being named, so ensure this is so!
        time_infos_df.index.names = [name_i if name_i else f'idx_{i}' for i,name_i in enumerate(time_infos_df.index.names)]
        cols_w_lists = Utilities_df.get_df_cols_with_list_elements(df = time_infos_df)
        if len(cols_w_lists)==0:
            subset=None
        else:
            subset = [x for x in time_infos_df.columns.tolist() if x not in cols_w_lists]
        #-----
        # Need to include the index level names in subset, since .reset_index() is called before drop_duplicates
        if subset is not None:
            subset.extend(list(time_infos_df.index.names))
        #-----
        time_infos_df = time_infos_df.reset_index().drop_duplicates(subset=subset).set_index(time_infos_df.index.names)
        self.time_infos_df = time_infos_df
        #-------------------------
        # MECPOAn objects are not allowed to save time_infos_df if of type OutageDType.outg (See MECPOAn.build_time_infos_df for more info).
        # Therefore, if OutageDType.outg, we must save here
        if self.data_type==OutageDType.outg and save_to_pkl:
            save_base = self.get_base_dir()
            pkl_path = os.path.join(save_base, 'time_infos_df.pkl')
            self.time_infos_df.to_pickle(pkl_path)
    
    
        
    #-----------------------------------------------------------------------------------------------------------------------------    
    def build_and_set_norm_counts_df_for_all(
        self, 
        remove_cols_from_dfs=True, 
        cpo_df_name_w_counts='rcpo_df_raw', 
        include_norm_lists=True, 
        include_all_available=True, 
        look_in_all_dfs_for_missing=True
    ):
        r"""
        For all analyses, call build_and_set_norm_counts_df.

        Thus, for each analysis, this will build a DF containing all of the normalization counts (and possibly lists of SNs).
        Instead of keeping these stored in the DFs themselves (or in rcpo_df_raw), it makes more sense
          to store them separately.  This way, they can be used by all, and any un-needed DFs can be dropped
          to save space without having and unintended consequences.
        """
        #-------------------------
        for mecpo_an in self.mecpo_coll_dict.values():
            mecpo_an.build_and_set_norm_counts_df(
                remove_cols_from_dfs=remove_cols_from_dfs, 
                cpo_df_name_w_counts=cpo_df_name_w_counts, 
                include_norm_lists=include_norm_lists, 
                include_all_available=include_all_available, 
                look_in_all_dfs_for_missing=look_in_all_dfs_for_missing, 
                replace_if_present=False
            )        
        
        
        
    def remove_SNs_cols_from_all_cpo_dfs(
        self, 
        SNs_tags=None, 
        include_cpo_df_OG=False,
        include_cpo_df_raw=False,
        cpo_dfs_to_ignore=None, 
        mecpo_an_keys_to_ignore=None, 
        is_long=False
    ):
        r"""
        Remove the SNs cols from all cpo_dfs in each MECPOAn in self.mecpo_coll_dict.
        By default, the SNs cols are NOT REMOVED FROM: rcpo_df_OG, rcpo_df_raw, and icpo_df_raw
          unless include_cpo_df_OG or include_cpo_df_OG is True

        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        NOTE: SNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled in MECPODf.remove_SNs_cols_from_rcpo_df.

        include_cpo_df_OG:
          If True, also remove SNs from rcpo_df_OG

        include_cpo_df_raw:
          If True, also remove SNs from rcpo_df_raw and icpo_df_raw

        cpo_dfs_to_ignore:
          List of cpo_df names.  The SNs cols will not be removed from any of these DFs

        mecpo_an_keys_to_ignore:
          List of mecpo_an_keys.  The SNs cols will not be removed from any of MECPOAn objects
        """
        #-------------------------
        if mecpo_an_keys_to_ignore is None:
            mecpo_an_keys_to_ignore=[]
        assert(Utilities.is_object_one_of_types(mecpo_an_keys_to_ignore, [list, tuple]))
        #-------------------------
        for mecpo_an_key in self.mecpo_an_keys:
            if mecpo_an_key in mecpo_an_keys_to_ignore:
                continue
            self.mecpo_coll_dict[mecpo_an_key].remove_SNs_cols_from_all_cpo_dfs(
                SNs_tags=SNs_tags, 
                include_cpo_df_OG=include_cpo_df_OG,
                include_cpo_df_raw=include_cpo_df_raw,
                cpo_dfs_to_ignore=cpo_dfs_to_ignore, 
                is_long=is_long
            )
    
    def make_cpo_columns_equal(
        self,
        same_order=True, 
        cols_to_init_with_empty_lists=MECPODf.std_SNs_cols(), 
        drop_empty_cpo_dfs=False
    ):
        r"""
        This makes the cpo columns equal between MECPOAn objects within self.mecpo_coll_dict
        """
        #-------------------------
        MECPOAn.make_cpo_columns_equal(
            mecpo_coll = list(self.mecpo_coll_dict.values()), 
            same_order=same_order, 
            cols_to_init_with_empty_lists=cols_to_init_with_empty_lists, 
            drop_empty_cpo_dfs=drop_empty_cpo_dfs
        )
        
    @staticmethod
    def make_cpo_columns_equal_between_mecpo_colls(
        mecpo_colls, 
        same_order=True, 
        cols_to_init_with_empty_lists=MECPODf.std_SNs_cols(), 
        drop_empty_cpo_dfs=False
    ):
        r"""
        This makes cpo columns equal between the MECPOCollection objects within mecpo_colls.
        The MECPOAn objects within each MECPOCollection are matched according to mecpo_an_keys

        mecpo_colls:
          A collection of MECPOCollection objects.
          This can be a list or dict.
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(mecpo_colls, [list, dict]))
        if isinstance(mecpo_colls, dict):
            mecpo_colls = list(mecpo_colls.values())
        #-------------------------
        all_mecpo_an_keys = set()
        for mecpo_coll_i in mecpo_colls:
            all_mecpo_an_keys = all_mecpo_an_keys.union(set(mecpo_coll_i.mecpo_an_keys))
        #-------------------------
        for mecpo_an_key_i in all_mecpo_an_keys:
            mecpo_an_list_i = []
            for mecpo_coll_i in mecpo_colls:
                if mecpo_an_key_i in mecpo_coll_i.mecpo_an_keys:
                    mecpo_an_list_i.append(mecpo_coll_i.mecpo_coll_dict[mecpo_an_key_i])
            if len(mecpo_an_list_i)>1:
                MECPOAn.make_cpo_columns_equal(
                    mecpo_coll=mecpo_an_list_i, 
                    same_order=same_order, 
                    cols_to_init_with_empty_lists=cols_to_init_with_empty_lists, 
                    drop_empty_cpo_dfs=drop_empty_cpo_dfs
                )
        #-------------------------
        return
        
    @staticmethod
    def make_mixed_cpo_columns_equal_between_mecpo_colls(
        mecpo_colls_with_cpo_df_names, 
        segregate_by_mecpo_an_keys=False, 
        same_order=True, 
        cols_to_init_with_empty_lists=MECPODf.std_SNs_cols(), 
        drop_empty_cpo_dfs=False
    ):
        r"""
        This allows the columns of DFs with different names to be made equal.
        Different from make_cpo_columns_equal_between_mecpo_colls, as that makes columns equal between DFs
          with the same name.  

        mecpo_colls_with_cpo_df_names:
          This needs to be a list of tuples.
          For each tuple, the first item (0th index) is a MECPOCollection object and the second item (1st index)
            is a list of cpo_df_names to include (actually, second item can also be single string if only on
            cpo df name to be included).

        segregate_by_mecpo_an_keys:
          If True, only DFs from MECPOAn objects having the same mecpo_an_key will have columns made equal.
          If False, all DFs with appropriate cpo_df_names will have columns made equal.
        """
        #-------------------------
        assert(isinstance(mecpo_colls_with_cpo_df_names, list))
        for coll_w_names in mecpo_colls_with_cpo_df_names:
            assert(Utilities.is_object_one_of_types(coll_w_names, [list, tuple]))
            assert(len(coll_w_names)==2)
            assert(isinstance(coll_w_names[0], MECPOCollection))
            assert(Utilities.is_object_one_of_types(coll_w_names[1], [str, list, tuple]))
            if isinstance(coll_w_names[1], str):
                coll_w_names[1] = [coll_w_names[1]]
        #-------------------------
        if segregate_by_mecpo_an_keys:
            all_mecpo_an_keys = set()
            for coll_w_names in mecpo_colls_with_cpo_df_names:
                all_mecpo_an_keys = all_mecpo_an_keys.union(set(coll_w_names[0].mecpo_an_keys))
            #----------
            for mecpo_an_key_i in all_mecpo_an_keys:
                dfs_list = []
                for coll_w_names in mecpo_colls_with_cpo_df_names:
                    coll         = coll_w_names[0]
                    cpo_df_names = coll_w_names[1]
                    if mecpo_an_key_i in coll.mecpo_an_keys:
                        for cpo_df_name in cpo_df_names:
                            if cpo_df_name in coll.mecpo_coll_dict[mecpo_an_key_i].cpo_dfs.keys():
                                dfs_list.append(coll.mecpo_coll_dict[mecpo_an_key_i].cpo_dfs[cpo_df_name])
                if len(dfs_list)>1:
                    AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list(
                        rcpo_dfs=dfs_list, 
                        same_order=same_order, 
                        inplace=True, 
                        cols_to_init_with_empty_lists=cols_to_init_with_empty_lists 
                    )
        else:
            dfs_list = []
            for coll_w_names in mecpo_colls_with_cpo_df_names:
                coll         = coll_w_names[0]
                cpo_df_names = coll_w_names[1]
                for cpo_df_name in cpo_df_names:
                    dfs_list.extend(coll.get_cpo_dfs(cpo_df_name))
            AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list(
                rcpo_dfs=dfs_list, 
                same_order=same_order, 
                inplace=True, 
                cols_to_init_with_empty_lists=cols_to_init_with_empty_lists            
            )
        
    def get_rough_reason_ordering(
        self,
        cpo_df_name, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        max_total_counts_args=None
    ):
        r"""
        Basically, find the ordering for each member of the collection, and take the overall ordering to be the
          average of those.  There is no weighting or anything complicated involved here, so if the DFs are of
          very different size, this will not fairly reflect that.

        In future, may want to implement a more involved method, e.g., one could use
          AMIEndEvents.combine_two_reason_counts_per_outage_dfs to combine all DFs, and then
          sort by the mean values.
          This would amount to essentially trying to build e.g., df_01_30 from the collection 
            of [df_01_05, df_06_10, df_11_15, df_16_20, df_21_25, df_26_30]
          The issue here is that one needs the counts columns (e.g., '_nSNs') in order to combine the
            normalized DFs, and these are typically dropped.  However, I could possibly choose to keep them
            in the future, and simply drop them whenever the DFs are grabbed for plotting?
        """
        #-------------------------
        cpo_dfs = self.get_cpo_dfs(
            cpo_df_name=cpo_df_name, 
            cpo_df_subset_by_mjr_mnr_cause_args=cpo_df_subset_by_mjr_mnr_cause_args, 
            max_total_counts_args=max_total_counts_args, 
            mecpo_an_order=None
        )
        cpo_dfs = AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list(rcpo_dfs=cpo_dfs, inplace=False)
        #-------------------------
        orders = [x.mean().sort_values(ascending=False) for x in cpo_dfs]
        orders_df = pd.concat(orders, axis=1)
        return_order = orders_df.mean(axis=1).sort_values(ascending=False).index.tolist()
        #-------------------------
        return return_order
        
    def add_counts_col_to_all_cpo_dfs_of_name(
        self, 
        cpo_df_name, 
        include_list_col=False
    ):
        r"""
        For each df of name cpo_df_name in each MECPOAn, add counts col (and possibly list col), as determined by 
          MECPOAn.cpo_df_name_to_norm_counts_col_dict
        """
        #-------------------------
        for mecpo_an in self.mecpo_coll_dict.values():
            mecpo_an.add_counts_col_to_df(
                cpo_df_name=cpo_df_name, 
                include_list_col=include_list_col
            )
        
    def add_counts_col_to_all_rcpo_dfs(
        self, 
        include_list_col=False
    ):
        r"""
        For each rcpo_df in each MECPOAn, add counts col (and possibly list col), as determined by 
          MECPOAn.cpo_df_name_to_norm_counts_col_dict
        """
        #-------------------------
        for mecpo_an in self.mecpo_coll_dict.values():
            mecpo_an.add_counts_col_to_all_rcpo_dfs(
                include_list_col=include_list_col
            )
            
    def get_counts_series(
        self, 
        cpo_df_name, 
        include_list_col=False
    ):
        r"""
        Grab counts series for DF from name cpo_df_name from each MECPOAn in collection.
        Combine all count series into a single pd.Series object with identifier indicies (typically, e.g., 
          outg_rec_nb or xfmr_nb) and values equal to counts

        NOTE: If include_list_col==True, the returned object will be a pd.DataFrame, not a pd.Series
        """
        #-------------------------
        all_count_series_list = []
        for mecpo_an in self.mecpo_coll_dict.values():
            count_series_i = mecpo_an.get_counts_series(
                cpo_df_name=cpo_df_name, 
                include_list_col=include_list_col           
            )
            all_count_series_list.append(count_series_i)
        all_count_series = pd.concat(all_count_series_list)

        # Need to drop any duplicate entries.  Simply using all_count_series.drop_duplicates() would delete duplicate values.
        #   However, what I want is to drop and duplicate index/value combinations.
        #   Therefore, convert to DF, drop duplicates, and convert back
        assert(Utilities.is_object_one_of_types(all_count_series, [pd.Series, pd.DataFrame]))
        if isinstance(all_count_series, pd.Series):
            all_count_series=all_count_series.to_frame()
        tmp_cols = [Utilities.generate_random_string() for _ in range(all_count_series.index.nlevels)]
        for i,tmp_col in enumerate(tmp_cols):
            all_count_series[tmp_col] = all_count_series.index.get_level_values(i)
        # If include_list_col, then the list col must be excluded from
        #   drop_duplicates, as it throws an error (TypeError: unhashable type: 'list')
        if include_list_col:
            drop_subset = MECPODf.get_non_SNs_cols_from_cpo_df(
                cpo_df=all_count_series, 
                SNs_tags=MECPODf.std_SNs_cols()
            )
        else:
            drop_subset=None
        all_count_series=all_count_series.drop_duplicates(subset=drop_subset)
        all_count_series=all_count_series.drop(columns=tmp_cols)

        # If all_count_series has only 1 column at this point (i.e., was originally a Series)
        # squeeze down to a pd.Series
        if all_count_series.shape[1]==1:
            all_count_series = all_count_series.squeeze()

        ## Make sure 1-1 relationship between identifier and counts
        #assert(all_count_series.index.nunique()==all_count_series.shape[0])
        if all_count_series.index.nunique()!=all_count_series.shape[0]:
            print('In MECPOCollection.get_counts_series, found indices with multiple values')
            n_unq_idx      = all_count_series.index.nunique()
            idx_val_counts = all_count_series.index.value_counts()
            all_count_series_w_dups  = all_count_series.loc[idx_val_counts[idx_val_counts>1].index]
            all_count_series_wo_dups = all_count_series.loc[idx_val_counts[idx_val_counts==1].index]
            assert(all_count_series_w_dups.shape[0]+all_count_series_wo_dups.shape[0]==all_count_series.shape[0])
            #-------------------------
            all_count_series_w_dups = all_count_series_w_dups.groupby(all_count_series_w_dups.index).apply(max)
            #-------------------------
            all_count_series = pd.concat([all_count_series_wo_dups, all_count_series_w_dups])
            assert(all_count_series.index.nunique()==n_unq_idx)
        assert(all_count_series.index.nunique()==all_count_series.shape[0])
        #-------------------------
        return all_count_series            

            
    @staticmethod
    def get_merged_cpo_df_subset_below_max_total_counts(
        merged_cpo_df, 
        max_total_counts,
        how='any', 
        SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
    ):
        r"""
        Similar to MECPODf.get_cpo_df_subset_below_max_total_counts, but for merged cpo DFs.

        NOT INTENDED FOR case where merged_cpo_df built from rcpo_df_OGs.  In such a case, merged_cpo_df.columns.nlevels
          will equal 3, and the assertion below will fail.
          It would not be terribly difficult to expand the functionality to include this case, but I don't see this really
            being needed in the future.
            If one did want to expand the functionality, the main issue would be how to treat each of the total_event_counts,
              as they will each have two columns (one for raw and one for norm) instead of one.
              The easiest method would be to use just one of the columns, but one could definitely develop something more complicated
              involving both.

        max_total_counts:
          This can either be a int/float or a dict
          (1). If this is a single int/float, it be the max_total_counts argument for all mecpo_an_keys in the merged_cpo_df.
          (2). If this is a dict, then unique max_total_counts values can be set for each mecpo_an_key.
               In this case, each key in max_total_counts must be contained in mecpo_an_keys.  However, not all mecpo_an_keys must
                 be included in max_total_counts.  If a mecpo_an_key is excluded from max_total_counts, its value will be set to None
               In the more general case, mecpo_an_keys are actually the merged_cpo_df.columns.get_level_values(0).unique() values.

        how:
          Should be equal to 'any' or 'all'.
          Determine if row is removed from DataFrame, when we have at least one value exceeding the max total counts 
            or all exceeding the max total counts
            'any' : If any exceeding the max total counts are present, drop that row.
            'all' : If all values are exceeding the max total counts, drop that row.
        """
        #-------------------------
        assert(merged_cpo_df.columns.nlevels==2)
        assert(how=='any' or how=='all')
        mecpo_an_keys = merged_cpo_df.columns.get_level_values(0).unique().tolist()
        #-------------------------
        # First, determine whether we're dealing with scenario 1 or scenario 2 for max_total_counts (described above)
        assert(Utilities.is_object_one_of_types(max_total_counts, [int, float, dict]))
        if not isinstance(max_total_counts, dict):
            # Scenario 1
            # Build max_total_counts_dict as a dict with one key for each mecpo_an_keys
            max_total_counts_dict = {mecpo_an_key:max_total_counts for mecpo_an_key in mecpo_an_keys}
        else:
            # Scenario 2
            max_total_counts_dict = max_total_counts
            # Assert that all max_total_counts_dict keys are in mecpo_an_keys
            assert(len(set(max_total_counts_dict.keys()).difference(mecpo_an_keys))==0)

            # If any of mecpo_an_keys are not included in max_total_counts_dict, set their value equal to None
            # Thus ensuring each of mecpo_an_keys is contained in max_total_counts_dict
            for mecpo_an_key in mecpo_an_keys:
                if mecpo_an_key not in max_total_counts_dict.keys():
                    max_total_counts_dict[mecpo_an_key] = None
        #-------------------------
        # At this point, regardless of whether scario 1 or 2, max_total_counts_dict should be a dict with keys equal
        #   to mecpo_an_keys
        assert(isinstance(max_total_counts_dict, dict))
        assert(len(set(max_total_counts_dict.keys()).symmetric_difference(mecpo_an_keys))==0)
        #-------------------------
        # Build the truth series (all_truths) and use it to project out the desired rows
        truth_series = []
        for mecpo_an_key in mecpo_an_keys:
            if max_total_counts_dict[mecpo_an_key] is None:
                continue
            total_event_counts_i = MECPODf.get_total_event_counts(
                merged_cpo_df[mecpo_an_key], 
                output_col='total_counts', 
                sort_output=False, 
                SNs_tags=SNs_tags
            )
            assert(total_event_counts_i.shape[1]==1)
            truth_series_i = total_event_counts_i.squeeze()<max_total_counts_dict[mecpo_an_key]
            # Make sure truth_series_i index matches that of merged_cpo_df
            # I don't think this is absolutely necessary, as pandas should align the indices when these truth
            #   series are actually used.  However, there is no harm in performing the check, and having everything
            #   in order already should speed up the implementation of the truth series
            assert(all(truth_series_i.index==merged_cpo_df.index))
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
        assert(all(all_truths.index==merged_cpo_df.index))
        merged_cpo_df = merged_cpo_df[all_truths]
        #-------------------------
        return merged_cpo_df
    

    @staticmethod
    def merge_cpo_dfs(
        dfs_coll                      , 
        col_level                     = -1,
        max_total_counts              = None, 
        how_max_total_counts          = 'any', 
        SNs_tags                      = MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols(), 
        cols_to_init_with_empty_lists = MECPODf.std_SNs_cols(), 
        make_cols_equal               = True
    ):
        r"""
        Returns a single DF consisting of the DFs from dfs merged.

        dfs_coll:
            This may be a list of pd.DataFrame objects or a dictionary with pd.DataFrame values.
            The function was originally developed for the latter, hence why dfs_coll is immediately converted to dfs_dict in the code
            type(dfs_coll)==dict:
                In order to merge all (since all should have the same columns) the mecpo_an_key for each is added as a level 0
                  value for MultiIndex columns.
                The keys of dfs_coll will be used as the mecpo_an_keys
                NOTE: This is true even if dfs contained in values already have MultiIndex columns!
            type(dfs_coll)==list:
                The dfs contained must have MultiIndex columns, and the level-0 for each must contain a single value which is unique from the others.            

        In order to merge all (since all should have the same columns) the mecpo_an_key for each is added as a level 0
          value for MultiIndex columns
    
        max_total_counts:
          This can either be a int/float or a dict (or None!)
          (0). If max_total_counts is None, no cuts imposed
          (1). If this is a single int/float, it be the max_total_counts argument for all mecpo_an_keys in the merged_cpo_df.
          (2). If this is a dict, then unique max_total_counts values can be set for each mecpo_an_key.
               In this case, each key in max_total_counts must be contained in mecpo_an_keys.  However, not all mecpo_an_keys must
                 be included in max_total_counts.  If a mecpo_an_key is excluded from max_total_counts, its value will be set to None
               In the more general case, mecpo_an_keys are actually the merged_cpo_df.columns.get_level_values(0).unique() values.
          !!!!! NOTE !!!!!
            max_total_counts criteria/on are/is implemented AT THE END.  Using the max_total_counts_args argument in get_cpo_dfs_dict would be incorrect.
            This is due to the merge method by which all DFs are joined.  When an index is present in one of the DFs (for a given mecpo_an_key)
              to be merged, but not in another, the merge process fills in the values for the missing DF with NaNs, which are then converted to 0.0.
            Therefore, if an index were excluded from a DF in dfs_dict due to failing the max_total_counts_args criterion, in the final merged
              DF it would appear as if this index simply had no events for that mecpo_an_key (when, in reality, it has many!)
              
        how_max_total_counts:
          Should be equal to 'any' or 'all'.  Determines how max_total_counts are imposed.
          Determine if row is removed from DataFrame, when we have at least one value exceeding the max total counts 
            or all exceeding the max total counts
            'any' : If any exceeding the max total counts are present, drop that row.
            'all' : If all values are exceeding the max total counts, drop that row.
            
        SNs_tags:
          Only used if max_total_counts is not None
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(dfs_coll, [dict, list]))
        if isinstance(dfs_coll, dict):
            prepend_keys = True
            dfs_dict    = dfs_coll
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
        # AMIEndEvents.make_reason_counts_per_outg_columns_equal_dfs_list_OLD(
        #     rcpo_dfs                      = list(dfs_dict.values()), 
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
        # Add mecpo_an_key as level 0 value for MultiIndex columns
        # Build dfs list
        dfs = []
        expected_final_idxs = [] # Built so number of columns can be checked at end
        for mecpo_an_key_i, df_i in dfs_dict.items():
            if prepend_keys:
                df_i = Utilities_df.prepend_level_to_MultiIndex(
                    df         = df_i, 
                    level_val  = mecpo_an_key_i, 
                    level_name = 'mecpo_an_key', 
                    axis       = 1
                )
            dfs.append(df_i)
            expected_final_idxs.extend(df_i.index.tolist())
        #-------------------------
        # Merge all DFs together    
        merged_df = reduce(lambda left,right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
    
        # Make sure merged_df has expected number of columns
        assert(merged_df.shape[1]==dfs[0].shape[1]*len(dfs))
    
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
            merged_df = MECPOCollection.get_merged_cpo_df_subset_below_max_total_counts(
                merged_cpo_df    = merged_df, 
                max_total_counts = max_total_counts, 
                how              = how_max_total_counts, 
                SNs_tags         = SNs_tags
            )
        #-------------------------
        merged_df = merged_df.sort_index(axis=1)
        #-------------------------
        return merged_df
        
    def get_merged_cpo_dfs(
        self, 
        cpo_df_name                         , 
        cpo_df_subset_by_mjr_mnr_cause_args = None, 
        max_total_counts                    = None, 
        how_max_total_counts                = 'any', 
        SNs_tags                            = MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols(), 
        cols_to_init_with_empty_lists       = MECPODf.std_SNs_cols(), 
        make_cols_equal                     = True
    ):
        r"""
        Like get_cpo_dfs, but instead of returning a list of DFs, this returns a single DF consisting of
          the DFs from get_cpo_dfs merged.
        In order to merge all (since all should have the same columns) the mecpo_an_key for each is added as a level 0
          value for MultiIndex columns

        max_total_counts:
          This can either be a int/float or a dict (or None!)
          (0). If max_total_counts is None, no cuts imposed
          (1). If this is a single int/float, it be the max_total_counts argument for all mecpo_an_keys in the merged_cpo_df.
          (2). If this is a dict, then unique max_total_counts values can be set for each mecpo_an_key.
               In this case, each key in max_total_counts must be contained in mecpo_an_keys.  However, not all mecpo_an_keys must
                 be included in max_total_counts.  If a mecpo_an_key is excluded from max_total_counts, its value will be set to None
               In the more general case, mecpo_an_keys are actually the merged_cpo_df.columns.get_level_values(0).unique() values.
          !!!!! NOTE !!!!!
            max_total_counts criteria/on are/is implemented AT THE END.  Using the max_total_counts_args argument in get_cpo_dfs_dict would be incorrect.
            This is due to the merge method by which all DFs are joined.  When an index is present in one of the DFs (for a given mecpo_an_key)
              to be merged, but not in another, the merge process fills in the values for the missing DF with NaNs, which are then converted to 0.0.
            Therefore, if an index were excluded from a DF in dfs_dict due to failing the max_total_counts_args criterion, in the final merged
              DF it would appear as if this index simply had no events for that mecpo_an_key (when, in reality, it has many!)
              
        how_max_total_counts:
          Should be equal to 'any' or 'all'.  Determines how max_total_counts are imposed.
          Determine if row is removed from DataFrame, when we have at least one value exceeding the max total counts 
            or all exceeding the max total counts
            'any' : If any exceeding the max total counts are present, drop that row.
            'all' : If all values are exceeding the max total counts, drop that row.
            
        SNs_tags:
          Only used if max_total_counts is not None
        """
        #-------------------------
        # Grab DFs in dict object, where key is mecpo_an_key
        # NOTE: add_an_key_as_level_0_col=False below so that make_reason_counts_per_outg_columns_equal_dfs_list can
        #       be used out of the box.  The level 0 values will be added later
        dfs_dict = self.get_cpo_dfs_dict(
            cpo_df_name                         = cpo_df_name, 
            cpo_df_subset_by_mjr_mnr_cause_args = cpo_df_subset_by_mjr_mnr_cause_args, 
            max_total_counts_args               = None, 
            add_an_key_as_level_0_col           = False
        )
        #-------------------------
        merged_df = MECPOCollection.merge_cpo_dfs(
            dfs_coll                      = dfs_dict, 
            col_level                     = -1,
            max_total_counts              = max_total_counts, 
            how_max_total_counts          = how_max_total_counts, 
            SNs_tags                      = SNs_tags, 
            cols_to_init_with_empty_lists = cols_to_init_with_empty_lists, 
            make_cols_equal               = make_cols_equal
        )
        #-------------------------
        return merged_df
        
    @staticmethod
    def get_top_reasons_subset_from_merged_cpo_df(
        merged_cpo_df,
        how='per_mecpo_an', 
        n_reasons_to_include=10,
        combine_others=True,
        output_combine_others_col='Other Reasons',
        SNs_tags=None, 
        is_norm=False, 
        counts_series=None
    ):
        r"""
        Similar to MECPODf.get_top_reasons_subset_from_cpo_df, but for merged cpo DFs.

        NOT INTENDED FOR case where merged_cpo_df built from rcpo_df_OGs.  In such a case, merged_cpo_df.columns.nlevels
          will equal 3, and the assertion below will fail.
          It would not be terribly difficult to expand the functionality to include this case, but I don't see this really
            being needed in the future.
            If one did want to expand the functionality, the main issue would be how to treat each of the total_event_counts,
              as they will each have two columns (one for raw and one for norm) instead of one.
              The easiest method would be to use just one of the columns, but one could definitely develop something more complicated
              involving both.

        how:
          Should equal 'per_mecpo_an' or 'overall'.
          'per_mecpo_an':
            The top n_reasons_to_include will be included from each mecpo_an_key (i.e., from each or the 
              merged_cpo_df.columns.get_level_values(0).unique() values.)
            In this case, the output should have n_reasons_to_include*merged_cpo_df.columns.get_level_values(0).nunique()
              columns (if combine_others==True, additional merged_cpo_df.columns.get_level_values(0).nunique() columns)
          'overall':
            The top overall n_reasons_to_include will be included, regardless of mecpo_an_key.
            In this case, the output should have n_reasons_to_include columns (if combine_others==True, additional 1 column)
        """
        #-------------------------
        assert(merged_cpo_df.columns.nlevels==2)
        assert(how=='per_mecpo_an' or how=='overall')
        mecpo_an_keys = merged_cpo_df.columns.get_level_values(0).unique().tolist()
        #-------------------------
        need_counts_col=False
        if combine_others and is_norm:
            need_counts_col=True
            assert(counts_series is not None and isinstance(counts_series, pd.Series))
            # Make sure all needed values are found in counter_series
            assert(len(set(merged_cpo_df.index).difference(counts_series.index))==0)
            tmp_col = (Utilities.generate_random_string(), Utilities.generate_random_string())
            counts_series = counts_series.to_frame(name=tmp_col)
            merged_cpo_df = pd.merge(merged_cpo_df, counts_series, left_index=True, right_index=True, how='inner')
        #-------------------------
        raw_dfs_w_idntfrs = []
        if how=='per_mecpo_an':
            for mecpo_an_key in mecpo_an_keys:
                if need_counts_col:
                    raw_df = merged_cpo_df[[mecpo_an_key, tmp_col[0]]].copy()
                else:
                    # NOTE: Need double [[]] below because want to keep both levels of MultiIndex columns
                    #       If only single [] were used, raw_df would not have MultiIndex columns
                    raw_df = merged_cpo_df[[mecpo_an_key]].copy()
                raw_dfs_w_idntfrs.append((raw_df, mecpo_an_key))
        elif how=='overall':
            raw_df = merged_cpo_df
            raw_dfs_w_idntfrs.append((raw_df, 'Overall'))
        else:
            assert(0)
        #-------------------------        
        dfs=[]    
        for merged_cpo_df_i_w_idntfr in raw_dfs_w_idntfrs:
            merged_cpo_df_i = merged_cpo_df_i_w_idntfr[0]
            idntfr          = merged_cpo_df_i_w_idntfr[1]
            # Life is much easier if I flatten the columns here
            org_cols = merged_cpo_df_i.columns
            join_str = ' '
            merged_cpo_df_i = Utilities_df.flatten_multiindex_columns(df=merged_cpo_df_i, join_str = join_str, reverse_order=False)
            rename_cols_dict = dict(zip(merged_cpo_df_i.columns, org_cols))
            if need_counts_col:
                counts_col = join_str.join(tmp_col)
                # Remove counts_col from rename_cols_dict, as MECPODf.get_top_reasons_subset_from_cpo_df will remove it
                #   from merged_cpo_df_i
                del rename_cols_dict[counts_col]
            else:
                counts_col = None
            #-----
            merged_cpo_df_i = MECPODf.get_top_reasons_subset_from_cpo_df(
                cpo_df=merged_cpo_df_i, 
                n_reasons_to_include=n_reasons_to_include,
                combine_others=combine_others, 
                output_combine_others_col=output_combine_others_col, 
                SNs_tags=SNs_tags, 
                is_norm=is_norm, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False
            )
            #-----
            # Unflatten the columns
            # Make sure column ordering is same
            # NOTE: If combine_others==True, output_combine_others_col won't be in rename_cols_dict,
            #       hence the need for using .get below instead of rename_cols_dict[x]
            #       Also, output_combine_others_col must be MultiIndex, hence (idntfr, x) below
            new_cols = [rename_cols_dict.get(x, (idntfr, x)) for x in merged_cpo_df_i.columns]
            new_cols = pd.MultiIndex.from_tuples(new_cols)
            merged_cpo_df_i.columns = new_cols
            dfs.append(merged_cpo_df_i)
        #-------------------------
        if len(dfs) > 1:
            assert(how=='per_mecpo_an')
            dfs_shape = dfs[0].shape
            for df in dfs:
                assert(df.shape==dfs_shape)
            merged_cpo_df = reduce(lambda left,right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), dfs)
            assert(merged_cpo_df.shape[0]==dfs_shape[0])
        else:
            assert(len(dfs)==1 and how=='overall')
            merged_cpo_df = dfs[0]
        #-------------------------
        return merged_cpo_df
        
        
    @staticmethod
    def get_reasons_subset_from_merged_cpo_df(
        merged_cpo_df,
        reasons_to_include,
        combine_others=True,
        output_combine_others_col='Other Reasons',
        SNs_tags=None, 
        is_norm=False, 
        counts_series=None
    ):
        r"""
        Similar to MECPODf.get_reasons_subset_from_cpo_df, but for merged cpo DFs.

        NOT INTENDED FOR case where merged_cpo_df built from rcpo_df_OGs.  In such a case, merged_cpo_df.columns.nlevels
          will equal 3, and the assertion below will fail.
          It would not be terribly difficult to expand the functionality to include this case, but I don't see this really
            being needed in the future.

        reasons_to_include:
            Can be a list or a dict.
            if isinstance(reasons_to_include, dict):
                Each key should be contained in merged_cpo_df.columns.get_level_values(0).unique().
                    NOTE: Do not need to include a key for each of merged_cpo_df.columns.get_level_values(0).unique().
                          Any which are not included will simply not be included in the output
                Each value should be a list of columns to include for that key (i.e., a list of strings)
            if isinstance(reasons_to_include, list):
                Each value should be a list of columns to include.  This list will be used for each of the
                subset DFs identified by merged_cpo_df.columns.get_level_values(0).unique()
        """
        #-------------------------
        assert(merged_cpo_df.columns.nlevels==2)
        mecpo_an_keys = merged_cpo_df.columns.get_level_values(0).unique().tolist()
        #-------------------------
        need_counts_col=False
        if combine_others and is_norm:
            need_counts_col=True
            assert(counts_series is not None and isinstance(counts_series, pd.Series))
            # Make sure all needed values are found in counter_series
            assert(len(set(merged_cpo_df.index).difference(counts_series.index))==0)
            tmp_col = (Utilities.generate_random_string(), Utilities.generate_random_string())
            counts_series = counts_series.to_frame(name=tmp_col)
            merged_cpo_df = pd.merge(merged_cpo_df, counts_series, left_index=True, right_index=True, how='inner')
        #-------------------------
        assert(Utilities.is_object_one_of_types(reasons_to_include, [list, dict]))
        if isinstance(reasons_to_include, list):
            reasons_to_include = {mecpo_an_key:reasons_to_include for mecpo_an_key in mecpo_an_keys}
        assert(isinstance(reasons_to_include, dict))

        # Each key in reasons_to_include must be contained in mecpo_an_keys (but not vice versa)
        assert(len(set(reasons_to_include.keys()).difference(set(mecpo_an_keys)))==0)

        # Each value should be a list whose elements are strings
        for an_key in reasons_to_include.keys():
            assert(isinstance(reasons_to_include[an_key], list))
            assert(Utilities.are_all_list_elements_of_type(reasons_to_include[an_key], str))
        #-------------------------
        dfs = []
        for mecpo_an_key in mecpo_an_keys:
            #-------------------------
            if mecpo_an_key not in reasons_to_include:
                continue
            #-------------------------
            if need_counts_col:
                merged_cpo_df_i = merged_cpo_df[[mecpo_an_key, tmp_col[0]]].copy()
            else:
                # NOTE: Need double [[]] below because want to keep both levels of MultiIndex columns
                #       If only single [] were used, raw_df would not have MultiIndex columns
                merged_cpo_df_i = merged_cpo_df[[mecpo_an_key]].copy()
            idntfr = mecpo_an_key
            #-----
            # Life is much easier if I flatten the columns here
            org_cols = merged_cpo_df_i.columns
            join_str = ' '
            merged_cpo_df_i = Utilities_df.flatten_multiindex_columns(df=merged_cpo_df_i, join_str = join_str, reverse_order=False)
            rename_cols_dict = dict(zip(merged_cpo_df_i.columns, org_cols))
            if need_counts_col:
                counts_col = join_str.join(tmp_col)
                # Remove counts_col from rename_cols_dict, as MECPODf.get_reasons_subset_from_cpo_df will remove it
                #   from merged_cpo_df_i
                del rename_cols_dict[counts_col]
            else:
                counts_col = None
            #-----
            reasons_to_include_i = reasons_to_include[mecpo_an_key]
            # Since columns of DF were flattened, same needs to happen with reasons_to_include_i
            reasons_to_include_i = [join_str.join([mecpo_an_key, x]) for x in reasons_to_include_i]
            #-----
            merged_cpo_df_i = MECPODf.get_reasons_subset_from_cpo_df(
                cpo_df=merged_cpo_df_i, 
                reasons_to_include=reasons_to_include_i, 
                combine_others=combine_others, 
                output_combine_others_col=output_combine_others_col, 
                SNs_tags=SNs_tags, 
                is_norm=is_norm, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False
            )
            #-----
            # Unflatten the columns
            # Make sure column ordering is same
            # NOTE: If combine_others==True, output_combine_others_col won't be in rename_cols_dict,
            #       hence the need for using .get below instead of rename_cols_dict[x]
            #       Also, output_combine_others_col must be MultiIndex, hence (idntfr, x) below
            new_cols = [rename_cols_dict.get(x, (idntfr, x)) for x in merged_cpo_df_i.columns]
            new_cols = pd.MultiIndex.from_tuples(new_cols)
            merged_cpo_df_i.columns = new_cols
            dfs.append(merged_cpo_df_i)
        #-------------------------
        if len(dfs) > 1:
            # In general, dfs don't need to have same number of columns
            # So, assertion is looser here than in get_top_reasons_subset_from_merged_cpo_df
            dfs_shape = dfs[0].shape
            dfs_index = dfs[0].index
            for df in dfs:
                assert(df.shape[0]==dfs_shape[0])
                assert(all(df.index==dfs_index))
            # Using the original method (commented out below) with reduce and merge is good in theory, but not great on memory
            # consumption when the dfs are large.
            # Therefore, I instead ensure all indices are equal in dfs, then use pd.concat, which appears to use much less memory
            #merged_cpo_df = reduce(lambda left,right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), dfs)
            merged_cpo_df = pd.concat(dfs, axis=1)
            assert(merged_cpo_df.shape[0]==dfs_shape[0])
        else:
            merged_cpo_df = dfs[0]
        #-------------------------
        return merged_cpo_df
        
    @staticmethod
    def get_top_reasons_subset_from_merged_cpo_df_and_project_from_others(
        merged_cpo_df,
        other_dfs_w_counts_series, 
        how='per_mecpo_an', 
        n_reasons_to_include=10,
        combine_others=True,
        output_combine_others_col='Other Reasons',
        SNs_tags=None, 
        is_norm=False, 
        counts_series=None
    ):
        r"""
        Get the top reasons subset from merged_cpo_df and project out those reasons from merged_cpo_df and the DFs found
        in other_dfs_w_counts_series

        other_dfs_w_counts_series:
          A list of tuples, where 0th element is merged_cpo_df_i and 1st element is accompanying counts_series_i
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(other_dfs_w_counts_series, [list, tuple]))
        assert(Utilities.are_list_elements_lengths_homogeneous(other_dfs_w_counts_series, 2))
        #-------------------------
        merged_cpo_df = MECPOCollection.get_top_reasons_subset_from_merged_cpo_df(
            merged_cpo_df=merged_cpo_df,
            how=how, 
            n_reasons_to_include=n_reasons_to_include,
            combine_others=combine_others,
            output_combine_others_col=output_combine_others_col,
            SNs_tags=SNs_tags, 
            is_norm=is_norm, 
            counts_series=counts_series
        )
        #-------------------------
        reasons_to_include = {}
        for mecpo_an_key in merged_cpo_df.columns.get_level_values(0).unique():
            assert(mecpo_an_key not in reasons_to_include)
            if combine_others:
                # In this case, don't include output_combine_others_col
                reasons_to_include_i = [x for x in merged_cpo_df[mecpo_an_key].columns if x!=output_combine_others_col]
            else:
                reasons_to_include_i = merged_cpo_df[mecpo_an_key].columns.tolist()
            reasons_to_include[mecpo_an_key] = reasons_to_include_i

        #-------------------------
        return_dfs = []
        for i, (df_i, counts_series_i) in enumerate(other_dfs_w_counts_series):
            if df_i is None or df_i.shape[0]==0:
                return_dfs.append(df_i)
                continue
            #-----
            df_i = MECPOCollection.get_reasons_subset_from_merged_cpo_df(
                merged_cpo_df = df_i,
                reasons_to_include=reasons_to_include,
                combine_others=combine_others,
                output_combine_others_col=output_combine_others_col,
                SNs_tags=SNs_tags, 
                is_norm=is_norm, 
                counts_series=counts_series_i
            )
            return_dfs.append(df_i)
        #-------------------------
        return merged_cpo_df, return_dfs
        
        
    def remove_reasons_from_all_rcpo_dfs(
        self, 
        regex_patterns_to_remove, 
        ignore_case=True    
    ):
        r"""
        For each rcpo_df in each MECPOAn, remove any columns from rcpo_df where any of the patterns in 
          regex_patterns_to_remove are found
        """
        #-------------------------
        for mecpo_an in self.mecpo_coll_dict.values():
            mecpo_an.remove_reasons_from_all_rcpo_dfs(
                regex_patterns_to_remove=regex_patterns_to_remove, 
                ignore_case=ignore_case
            )
            
    def combine_reasons_in_all_rcpo_dfs(
        self, 
        patterns_and_replace=None, 
        addtnl_patterns_and_replace=None, 
        initial_strip=True,
        initial_punctuation_removal=True, 
        level_0_raw_col='counts', 
        level_0_nrm_col='counts_norm', 
        return_red_to_org_cols_dict=False
    ):
        r"""
        For each rcpo_df in each MECPOAn, combine groups of reasons in according to patterns_and_replace.
        NOTE: Specifically for rcpo_dfs, as icpo_dfs will have different columns
              (rcpo_dfs have reasons, icpo_dfs have enddeviceeventtypeids)

        NOTE!!!!!!!!!!!!!:
          Typically, one should keep patterns_and_replace=None.  When this is the case, dflt_patterns_and_replace
            will be used (see see dflt_patterns_and_replace in MECPODf.combine_cpo_df_reasons)
          If one wants to add to dflt_patterns_and_replace, use the addtnl_patterns_and_replace argument.

        patterns_and_replace/addtnl_patterns_and_replace:
          A list of tuples (or lists) of length 2.
          For each item in the list:
            first element should be a regex pattern for which to search 
            second element is replacement

            DON'T FORGET ABOUT BACKREFERENCING!!!!!
              e.g.
                reason_i = 'I am a string named Jesse Thomas Buxton'
                patterns_and_replace_i = (r'(Jesse) (Thomas) (Buxton)', r'\1 \3')
                  reason_i ===> 'I am a string named Jesse Buxton'              
        """
        #-------------------------
        # NOTE: If return_red_to_org_cols_dict==False, then self.combine_reasons_in_all_rcpo_dfs will return None,
        #       and red_to_org_cols_dicts will be a dict with all values equal to None (so, no harm in building regardless)
        red_to_org_cols_dicts={}
        for mecpo_an_key, mecpo_an in self.mecpo_coll_dict.items():
            red_to_org_cols_dicts_i = mecpo_an.combine_reasons_in_all_rcpo_dfs(
                patterns_and_replace        = patterns_and_replace, 
                addtnl_patterns_and_replace = addtnl_patterns_and_replace, 
                initial_strip               = initial_strip,
                initial_punctuation_removal = initial_punctuation_removal, 
                level_0_raw_col             = level_0_raw_col, 
                level_0_nrm_col             = level_0_nrm_col, 
                return_red_to_org_cols_dict = return_red_to_org_cols_dict
            )
            red_to_org_cols_dicts[mecpo_an_key] = red_to_org_cols_dicts_i
        self.red_to_org_cols_dicts = red_to_org_cols_dicts
        if return_red_to_org_cols_dict:
            return red_to_org_cols_dicts
            
    def delta_cpo_df_reasons_in_all_rcpo_dfs(
        self,  
        reasons_1,
        reasons_2,
        delta_reason_name, 
        level_0_raw_col='counts', 
        level_0_nrm_col='counts_norm'
    ):
        r"""
        For each rcpo_df in each MECPOAn, combine groups of reasons according to patterns_and_replace.
        """
        #-------------------------
        for mecpo_an in self.mecpo_coll_dict.values():
            mecpo_an.delta_cpo_df_reasons_in_all_rcpo_dfs(
                reasons_1=reasons_1,
                reasons_2=reasons_2,
                delta_reason_name=delta_reason_name, 
                level_0_raw_col=level_0_raw_col, 
                level_0_nrm_col=level_0_nrm_col
            )
            
            
    def get_total_event_counts(
        self, 
        mecpo_an_key,
        cpo_df_name, 
        cpo_df_subset_by_mjr_mnr_cause_args=None, 
        output_col='total_counts', 
        SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols(), 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm', 
        add_an_key_as_level_0_col=False
    ):
        r"""
        mecpo_an_key:
          The MECPOAn key, chooses from which MECPOAn in self.mecpo_coll_dict to grab the DF

        cpo_df_name:
          The name of the cpo pd.DataFrames, used to retrieve the correct DF from self.mecpo_coll_dict[mecpo_an_key]

        cpo_df_subset_by_mjr_mnr_cause_args:
          Allows one to select subsets of the pd.DataFrames by the outage type.
          This should be a dict with arguments appropriate for the MECPOAn.get_cpo_df_subset_by_mjr_mnr_cause
            function (except for the cpo_df_name argument, which will be set to cpo_df_name)
            
        add_an_key_as_level_0_col:
          If True, cpo_df will always be returned with MultiIndex columns where the level 0
            value is equal to mecpo_an_key.
            NOTE: If columns previously normal Index, returned will be MultiIndex with 2 level
                  If columns previously MultiIndex, one additional level will be preprended.
                  
        The nSNs (and SNs) columns should not be included in the count, hence the need for the SNs tags argument.
        If normalize_by_nSNs_included==True, the return DF with have MultiIndex level_0 values matching those of cpo_df
          (i.e., with level_0_raw_col and level_0_nrm_col)
        """
        #-------------------------
        assert(mecpo_an_key in self.mecpo_coll_dict.keys())
        assert(cpo_df_name in self.mecpo_coll_dict[mecpo_an_key].cpo_dfs.keys())
        #-------------------------
        total_df = self.mecpo_coll_dict[mecpo_an_key].get_total_event_counts(
            cpo_df_name=cpo_df_name, 
            cpo_df_subset_by_mjr_mnr_cause_args=cpo_df_subset_by_mjr_mnr_cause_args, 
            output_col=output_col, 
            SNs_tags=SNs_tags, 
            normalize_by_nSNs_included=normalize_by_nSNs_included, 
            level_0_raw_col = level_0_raw_col, 
            level_0_nrm_col = level_0_nrm_col
        )
        #-------------------------
        if add_an_key_as_level_0_col:
            total_df = Utilities_df.prepend_level_to_MultiIndex(
                df=total_df, 
                level_val=mecpo_an_key, 
                level_name='mecpo_an_key', 
                axis=1
            )
        #-------------------------
        return total_df
        
        
    @staticmethod
    def get_total_event_counts_for_merged_cpo_df(
        merged_df, 
        output_col='total_counts', 
        SNs_tags=None
    ):
        r"""
        Get total event counts for a merged cpo df
        """
        #-------------------------
        assert(merged_df.columns.nlevels==2)
        mecpo_an_keys = merged_df.columns.get_level_values(0).unique().tolist()
        #-------------------------
        total_counts_dfs = []
        final_col_order = [] # To maintain original order after inserting new cols
        for mecpo_an_key in mecpo_an_keys:
            cpo_df_i = merged_df[[mecpo_an_key]]
            total_counts_i = MECPODf.get_total_event_counts(
                cpo_df=cpo_df_i, 
                output_col = (mecpo_an_key, output_col), 
                sort_output=False, 
                SNs_tags=SNs_tags, 
                normalize_by_nSNs_included=False
            )
            total_counts_dfs.append(total_counts_i)
            #-----
            final_col_order.extend(merged_df[[mecpo_an_key]].columns.tolist())
            final_col_order.append((mecpo_an_key, output_col))
        #-------------------------
        # Just as in get_reasons_subset_from_merged_cpo_df above, the merge methods are terrible on memory consumption with large DFs.
        # Therefore, I will instead the DFs share the same indices, and then do a pd.concat along axis=1
        #total_counts_df = reduce(lambda left,right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), total_counts_dfs)
        dfs_shape = total_counts_dfs[0].shape
        dfs_index = total_counts_dfs[0].index
        for i in range(len(total_counts_dfs)):
            assert(total_counts_dfs[i].shape[0]==dfs_shape[0])
            if not all(total_counts_dfs[i].index==dfs_index):
                total_counts_dfs[i] = total_counts_dfs[i].loc[dfs_index]
        total_counts_df = pd.concat(total_counts_dfs, axis=1)
        assert(merged_df.shape[0]==total_counts_df.shape[0])
        #-------------------------
        len_og = merged_df.shape[0] 
        merged_df = pd.merge(merged_df, total_counts_df, left_index=True, right_index=True, how='inner')
        assert(len_og==merged_df.shape[0])
        assert(len(set(final_col_order).symmetric_difference(set(merged_df.columns)))==0)
        merged_df = merged_df[final_col_order]
        #-------------------------
        return merged_df
        
        
    def add_unique_idfr_to_idx_in_all_dfs(self):
        r"""
        THIS ONLY WORKS FOR THE SPECIAL CASE WHERE THE INDEX ONLY HAS ONE LEVEL!
        After this operation, the DFs will have indices with two levels, the first equals
          the original index and the second contains the unique identifiers.
        The unique identifiers are simply a random prefix with the iloc value of the row in rcpo_df_OG

        Also, the DFs should have unique indices.

        Purpose/intent: This was built to handle the original baseline, where a single clean period was queried for
                          each transformer, and thus the results were only grouped by trsf_pole_nb.
                        The issue occurs when combining the baseline results from different years, for each year will
                          contain many of the same trsf_pole_nbs as the others, and after combining there is no way to
                          distinguish them from each other.
                        The different entries should be kept separate, not aggregated together, thus the ability to
                          distinguish them is important.
        """
        #-------------------------
        for mecpo_an in self.mecpo_coll_dict.values():
            mecpo_an.add_unique_idfr_to_idx_in_all_dfs()
            
            
            
    def remove_all_cpo_dfs_except(
        self, 
        to_keep, 
        first_build_and_set_norm_counts_df_for_all=True, 
        build_and_set_norm_counts_df_for_all_kwargs=None, 
        keep_rcpo_df_OG=True
    ):
        r"""
        By remove, I mean set to self.init_cpo_dfs_to (which is typically an empty DF)

        to_keep:
            Should be the name of a DF to keep, or a list of names of DFs to keep.
            
        first_build_and_set_norm_counts_df_for_all:
            If true, before removing DFs, call build_and_set_norm_counts_df_for_all.
            This is important as it creates the norm_counts_df which is then used whenever combining DFs,
              normalizing, etc.
        """
        #-------------------------
        if first_build_and_set_norm_counts_df_for_all:
            if build_and_set_norm_counts_df_for_all_kwargs is None:
                build_and_set_norm_counts_df_for_all_kwargs = {}
            self.build_and_set_norm_counts_df_for_all(**build_and_set_norm_counts_df_for_all_kwargs)
        #-------------------------
        if not isinstance(to_keep, list):
            to_keep = [to_keep]
        #-------------------------
        # Typically want to keep rcpo_df_OG!
        if keep_rcpo_df_OG and 'rcpo_df_OG' not in to_keep:
            to_keep = ['rcpo_df_OG'] + to_keep
        #-------------------------
        for mecpo_an in self.mecpo_coll_dict.values():
            mecpo_an.remove_all_cpo_dfs_except(
                to_keep=to_keep, 
                keep_rcpo_df_OG=keep_rcpo_df_OG
            )
            
            
    @staticmethod
    def combine_two_mecpo_colls(
        mecpo_coll_1, 
        mecpo_coll_2, 
        append_only=True
    ):
        r"""
        NOTE: Case where append_only==False should work, but has not been super scrutinized and is somewhat hacky

        If the datasets are mutually exclusive, append_only should be set to True.
            In this case, having append_only=True or False should give the same result, but having it set to
            True will run much faster
        """
        #-------------------------
        # Get all mecpo_an_keys as the union of those found in mecpo_coll_1 and mecpo_coll_2
        all_mecpo_an_keys = list(set(mecpo_coll_1.mecpo_an_keys).union(set(mecpo_coll_2.mecpo_an_keys)))
        #-------------------------
        mecpo_coll_12_dict = {}
        for mecpo_an_key_i in all_mecpo_an_keys:
            assert(mecpo_an_key_i not in mecpo_coll_12_dict.keys())
            an_key_i_in_1 = mecpo_an_key_i in mecpo_coll_1.mecpo_coll_dict.keys()
            an_key_i_in_2 = mecpo_an_key_i in mecpo_coll_2.mecpo_coll_dict.keys()
            assert(an_key_i_in_1 or an_key_i_in_2)
            #-------------------------
            # If mecpo_an_key_i only found in one of mecpo_coll_1 or mecpo_coll_2, simply add
            #   the present one to mecpo_coll_12_dict
            if not(an_key_i_in_1 and an_key_i_in_2):
                if an_key_i_in_1:
                    mecpo_coll_12_dict[mecpo_an_key_i] = copy.deepcopy(mecpo_coll_1)
                else:
                    mecpo_coll_12_dict[mecpo_an_key_i] = copy.deepcopy(mecpo_coll_2)
                continue
            #-------------------------
            # At this point, mecpo_an_key_i present in both mecpo_coll_1 and mecpo_coll_2
            mecpo_an_i_1 = mecpo_coll_1.get_mecpo_an(mecpo_an_key_i)
            mecpo_an_i_2 = mecpo_coll_2.get_mecpo_an(mecpo_an_key_i)

            # Combine mecpo_an_i_1 and mecpo_an_i_2
            mecpo_an_i_12 = MECPOAn.combine_two_mecpo_ans(
                mecpo_an_i_1, 
                mecpo_an_i_2, 
                append_only=append_only
            )
            mecpo_coll_12_dict[mecpo_an_key_i] = mecpo_an_i_12
        #-------------------------
        # Determine what coll_label should be used for mecpo_coll_12
        # If the coll_label is the same for mecpo_coll_1 and mecpo_coll_2, simply use it
        # Otherwise, use coll_label_1 + coll_label_2
        coll_label_1 = mecpo_coll_1.coll_label
        coll_label_2 = mecpo_coll_2.coll_label
        if coll_label_1==coll_label_2:
            coll_label_12 = coll_label_1
        else:
            coll_label_12 = f'{coll_label_1} + {coll_label_2}'
        #-------------------------
        # Use mecpo_coll_12_dict and coll_label_12 to build mecpo_coll_12
        mecpo_coll_12 = MECPOCollection(
            mecpo_coll=mecpo_coll_12_dict, 
            coll_label=coll_label_12
        )
        #-------------------------
        return mecpo_coll_12

    @staticmethod
    def combine_mecpo_colls(
        mecpo_colls_list, 
        append_only=True
    ):
        r"""
        NOTE: Case where append_only==False should work, but has not been super scrutinized and is somewhat hacky

        If the datasets are mutually exclusive, append_only should be set to True.
            In this case, having append_only=True or False should give the same result, but having it set to
            True will run much faster
        """
        #-------------------------
        assert(isinstance(mecpo_colls_list, list) and len(mecpo_colls_list)>1)
        #-------------------------
        mecpo_coll_comb = copy.deepcopy(mecpo_colls_list[0])
        for i_coll in range(1, len(mecpo_colls_list)):
            mecpo_coll_comb = MECPOCollection.combine_two_mecpo_colls(
                mecpo_coll_1=mecpo_coll_comb, 
                mecpo_coll_2=mecpo_colls_list[i_coll], 
                append_only=append_only
            )    
        #-------------------------
        return mecpo_coll_comb