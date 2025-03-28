#!/usr/bin/env python

r"""
Holds OMInput class.  See OMInput.OMInput for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import re
import json
import pickle

import pandas as pd

import copy

#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from MECPODf import MECPODf
from MECPOAn import MECPOAn
from MECPOCollection import MECPOCollection
from OutageDAQ import OutageDataInfo as ODI
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
#---------------------------------------------------------------------
#sys.path.insert(0, os.path.join(os.path.realpath('..'), 'Utilities'))
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
from CustomJSON import CustomWriter


#------------------------------------------------------------------------------------------------------------------------------------------
class OMInput:
    r"""
    Class to build/store the input datasets needed to construct the OutageModeler.
    At the time of writing, there are three distinct types of data that can be used to build the model:
        outg  = OUTaGe == signal data / outages dataset (always needed)
        otbl  = Outage Transformer BaseLine (always needed)
        prbl  = PRistine BaseLine (optionals)
    """
    #---------------------------------------------------------------------------
    def __init__(
        self                       , 
        dataset                    , 
        acq_run_date               , 
        data_date_ranges           , 
        grp_by_cols                , 
        cpx_dfs_name               = 'rcpo_df_norm_by_xfmr_nSNs', 
        acq_run_date_subdir_appndx = None, 
        data_dir_base              = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data', 
        init_dfs_to                = pd.DataFrame(),  # Should be pd.DataFrame() or None
        om_input                   = None, # if copy constructor functionality is desired
        **mecpx_build_kwargs
    ):
        r"""
        dataset:
            Must be one of ['outg', 'otbl', 'prbl']
            
        acq_run_date:
            The acquisition date of the data (i.e., the date the acquisition program was run to save the needed 
              data locally)
            
        data_date_ranges:
            The date ranges to be included in the MECPOCollection.
            This should be a list of two-element.
            
        grp_by_cols:
            The columns which were grouped in the packaging of the data.
            Typically, this is:
                ['outg_rec_nb', 'trsf_pole_nb'] for outage data
                ['trsf_pole_nb', 'no_outg_rec_nb'] for baseline data
                
        data_dir_base:
            Tbe base directory housing the data.
            Typically this should be 
                r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data' or
                r'U:\CloudData\dovs_and_end_events_data'

        mecpx_build_kwargs:
            For list of acceptable keyword arguments, see OMInput.get_default_mecpx_build_info_dict.
            NOTE: Those keys explicitly handled (i.e., those which are input parameters to the __init__ constructor)
                  will be ignored.  See handled_kwargs for those which will be ignored.
        """
        #---------------------------------------------------------------------------
        if om_input is not None:
            assert(isinstance(om_input, OMInput))
            self.copy_constructor(om_input)
            return
        #--------------------------------------------------
        self.init_dfs_to = init_dfs_to
        #--------------------------------------------------
        ODI.assert_dataset(dataset)
        self.__data_info            = ODI(dataset)
        #--------------------------------------------------
        self.__mecpx_build_info_dict  = OMInput.get_default_mecpx_build_info_dict(dataset = self.dataset)
        #-------------------------
        assert(Utilities.is_object_one_of_types(data_date_ranges, [list, tuple]))
        assert(Utilities.are_list_elements_lengths_homogeneous(data_date_ranges, length=2))
        #-------------------------
        self.set_mecpx_build_args(
            extend_any_lists           = False, 
            acq_run_date               = acq_run_date, 
            data_date_ranges           = data_date_ranges, 
            grp_by_cols                = grp_by_cols, 
            acq_run_date_subdir_appndx = acq_run_date_subdir_appndx, 
            data_dir_base              = data_dir_base,
            cpx_dfs_name               = cpx_dfs_name, 
        )
        #-------------------------
        handled_kwargs = [
            'acq_run_date', 
            'data_date_ranges',
            'grp_by_cols', 
            'acq_run_date_subdir_appndx', 
            'data_dir_base', 
            'cpx_dfs_name'
        ]
        mecpx_build_kwargs = {k:v for k,v in mecpx_build_kwargs if k not in handled_kwargs}
        if len(mecpx_build_kwargs)>0:
            self.set_mecpx_build_args(
                extend_any_listsb = False, 
                **mecpx_build_kwargs
            )
        #--------------------------------------------------
        # MECPOCollection object
        #     These are needed to compile the finals DFs which will be used for modelling.
        #     HOWEVER, they are EXTREMELY heavy.
        #     Therefore, they should be closed out after being built and saved, and only used when needed.
        #     E.g., after building DFs, one should save to disk so they can be loaded directly when running model
        #       instead of needing to build from MECPOCollection objects each time.
        #-------------------------
        # mecpx = Meter Event Counts Per X (e.g., per transformer)
        self.__mecpx_coll    = None
        self.__merged_df     = Utilities.general_copy(self.init_dfs_to)
        self.__counts_series = None

    #---------------------------------------------------------------------------
    @property
    def data_info(self):
        return self.__data_info
    #-----
    @property
    def dataset(self):
        return self.__data_info.dataset
    @property
    def subdir(self):
        return self.__data_info.subdir
    @property
    def naming_tag(self):
        return self.__data_info.naming_tag
    @property
    def is_no_outage(self):
        return self.__data_info.is_no_outage
    #-------------------------
    @property
    def mecpx_build_info_dict(self):
        return self.__mecpx_build_info_dict
    #-----
    @property
    def acq_run_date(self):
        return self.__mecpx_build_info_dict['acq_run_date']
    @property
    def data_date_ranges(self):
        return self.__mecpx_build_info_dict['data_date_ranges']
    @property
    def grp_by_cols(self):
        return self.__mecpx_build_info_dict['grp_by_cols']
    @property
    def acq_run_date_subdir_appndx(self):
        return self.__mecpx_build_info_dict['acq_run_date_subdir_appndx']
    @property
    def data_dir_base(self):
        return self.__mecpx_build_info_dict['data_dir_base']
    @property
    def cpx_dfs_name(self):
        return self.__mecpx_build_info_dict['cpx_dfs_name']
    @property
    def std_dict_grp_by_cols(self):
        return self.__mecpx_build_info_dict['std_dict_grp_by_cols']
    @property
    def coll_label(self):
        return self.__mecpx_build_info_dict['coll_label']
    @property
    def barplot_kwargs_shared(self):
        return self.__mecpx_build_info_dict['barplot_kwargs_shared']
    @property
    def rec_nb_idfr(self):
        return self.__mecpx_build_info_dict['rec_nb_idfr']
    @property
    def trsf_pole_nb_idfr(self):
        return self.__mecpx_build_info_dict['trsf_pole_nb_idfr']
    @property
    def normalize_by_time_interval(self):
        return self.__mecpx_build_info_dict['normalize_by_time_interval']
    @property
    def include_power_down_minus_up(self):
        return self.__mecpx_build_info_dict['include_power_down_minus_up']
    @property
    def pd_col(self):
        return self.__mecpx_build_info_dict['pd_col']
    @property
    def pu_col(self):
        return self.__mecpx_build_info_dict['pu_col']
    @property
    def pd_m_pu_col(self):
        return self.__mecpx_build_info_dict['pd_m_pu_col']
    @property
    def regex_to_remove_patterns(self):
        return self.__mecpx_build_info_dict['regex_to_remove_patterns']
    @property
    def regex_to_remove_ignore_case(self):
        return self.__mecpx_build_info_dict['regex_to_remove_ignore_case']
    @property
    def max_total_counts(self):
        return self.__mecpx_build_info_dict['max_total_counts']
    @property
    def how_max_total_counts(self):
        return self.__mecpx_build_info_dict['how_max_total_counts']
    @property
    def mecpo_idx_for_ordering(self):
        return self.__mecpx_build_info_dict['mecpo_idx_for_ordering']
    @property
    def freq(self):
        return self.__mecpx_build_info_dict['freq']
    @property
    def days_min_max_outg_td_windows(self):
        return self.__mecpx_build_info_dict['days_min_max_outg_td_windows']
    @property
    def mecpx_an_keys(self):
        return self.__mecpx_build_info_dict['mecpx_an_keys']
    @property
    def old_to_new_keys_dict(self):
        return self.__mecpx_build_info_dict['old_to_new_keys_dict']
    #-------------------------
    @property
    def mecpx_coll(self):
        return self.__mecpx_coll
    @property
    def merged_df(self):
        return self.__merged_df
    @property
    def counts_series(self):
        return self.__counts_series
    #--------------------------------------------------
    def set_mecpx_coll(
        self, 
        mecpx_coll
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(mecpx_coll, MECPOCollection))
        self.__mecpx_coll = mecpx_coll


    #---------------------------------------------------------------------------
    def copy_constructor(
        self, 
        om_input
    ):
        r"""
        Annoyingly, the follow simple solution does not work:
            self = copy.deepcopy(om_input)
          neither does:
            self = OMInput()
            self.__dict__ = copy.deepcopy(om_input.__dict__)
    
        So, I guess it's back to headache C++ style...
        """
        #--------------------------------------------------
        assert(isinstance(om_input, OMInput))
        #--------------------------------------------------
        self.init_dfs_to             = Utilities.general_copy(om_input.init_dfs_to)
        self.__data_info             = Utilities.general_copy(om_input.__data_info)
        self.__mecpx_build_info_dict = Utilities.general_copy(om_input.__mecpx_build_info_dict)
        self.__mecpx_coll            = Utilities.general_copy(om_input.__mecpx_coll)
        self.__merged_df             = Utilities.general_copy(om_input.__merged_df)
        self.__counts_series         = Utilities.general_copy(om_input.__counts_series)

    #---------------------------------------------------------------------------
    def copy(self):
        r"""
        """
        #-------------------------
        return_om_input = OMInput(om_input=self)
        return return_om_input


    #---------------------------------------------------------------------------
    def are_data_normalized(self):
        r"""
        """
        #-------------------------
        is_norm = MECPOAn.is_df_normalized(cpx_df_name = self.cpx_dfs_name)
        return is_norm

    #---------------------------------------------------------------------------
    @staticmethod
    def check_mecpx_build_info_dict(mecpx_build_info_dict):
        r"""
        """
        exit_value = 0
        try:
            #-------------------------
            # exit_value = 0
            if mecpx_build_info_dict['old_to_new_keys_dict'] is not None:
                assert(len(mecpx_build_info_dict['old_to_new_keys_dict']) == len(mecpx_build_info_dict['days_min_max_outg_td_windows']))
            exit_value+=1
            #-------------------------
            # exit_value = 1
            # Sanity check
            for window_i in mecpx_build_info_dict['days_min_max_outg_td_windows']:
                assert(pd.Timedelta(f'{window_i[1]}D')-pd.Timedelta(f'{window_i[0]}D')==pd.Timedelta(mecpx_build_info_dict['freq']))
            exit_value+=1
            #-------------------------
            # exit_value = 2
            assert(set(mecpx_build_info_dict['grp_by_cols']).symmetric_difference(set(mecpx_build_info_dict['std_dict_grp_by_cols'].keys()))==set())
            exit_value+=1
            #-------------------------
            # Make sure rec_nb_idfr and trsf_pole_nb_idfr contained in final indices
            # exit_value = 3
            assert(mecpx_build_info_dict['rec_nb_idfr'][1]       in mecpx_build_info_dict['std_dict_grp_by_cols'].values())
            assert(mecpx_build_info_dict['trsf_pole_nb_idfr'][1] in mecpx_build_info_dict['std_dict_grp_by_cols'].values())
            exit_value+=1
            #-------------------------
            # exit_value = 4
            return True
        except:
            print(f"FAILED in OMInput.check_mecpx_build_info_dict\n\texit_value = {exit_value}")
            return False

    #---------------------------------------------------------------------------
    def self_check_mecpx_build_info_dict(self):
        r"""
        """
        passed = OMInput.check_mecpx_build_info_dict(self.mecpx_build_info_dict)
        if not passed:
            print(f'dataset = {self.dataset}')
        return passed


    #---------------------------------------------------------------------------
    @staticmethod
    def build_old_to_new_an_keys(
        old_an_keys
    ):
        r"""
        Given a list of old analysis keys (i.e., those assigned during the data acquisition and packagaing), convert these
          to their simpler form.
                e.g., 'outg_td_window_1_to_6_days' ==> '01-06 Days'
        Returns a dictionary whose keys are the old values and values are new keys
        """
        #-------------------------
        assert(isinstance(old_an_keys, list))
        regex = r'.*_(\d*)_to_(\d*)_days'
        return_dict = {}
        for old_key_i in old_an_keys:
            assert(old_key_i not in return_dict.keys())
            #-----
            new_key_i = re.findall(regex, old_key_i)
            assert(len(new_key_i)==1 and len(new_key_i[0])==2)
            #-----
            new_key_i = "{:02d}-{:02d} Days".format(int(new_key_i[0][0]), int(new_key_i[0][1]))
            #-----
            return_dict[old_key_i] = new_key_i
        #-------------------------
        return return_dict
        
    #---------------------------------------------------------------------------
    @staticmethod
    def get_default_mecpx_build_info_dict(
        dataset=None
    ):
        r"""
        Mainly so user doesn't have to enter all values each time.
        This will probably be used with Utilities.supplement_dict_with_default_values
        """
        #-------------------------
        if dataset is None:
            dataset = 'outg'
        ODI.assert_dataset(dataset)
        #-------------------------
        mecpx_build_info_dict                                 = dict()
        #-------------------------
        mecpx_build_info_dict['acq_run_date']                 = '20240101'
        mecpx_build_info_dict['acq_run_date_subdir_appndx']   = None
        mecpx_build_info_dict['data_date_ranges']             = [
            ['2023-01-01', '2023-12-31']
        ]
        mecpx_build_info_dict['data_dir_base']                = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
        #-------------------------
        if dataset   == 'outg':
            mecpx_build_info_dict['grp_by_cols']              = ['outg_rec_nb', 'trsf_pole_nb']
        elif dataset == 'otbl':
            mecpx_build_info_dict['grp_by_cols']              = ['trsf_pole_nb', 'no_outg_rec_nb']
        elif dataset == 'prbl':
            mecpx_build_info_dict['grp_by_cols']              = ['trsf_pole_nb', 'no_outg_rec_nb']
        else:
            assert(0)
        #-------------------------
        if dataset   == 'outg':
            mecpx_build_info_dict['std_dict_grp_by_cols']     = {'outg_rec_nb':'outg_rec_nb',    'trsf_pole_nb':'trsf_pole_nb'}
        elif dataset == 'otbl':
            mecpx_build_info_dict['std_dict_grp_by_cols']     = {'no_outg_rec_nb':'outg_rec_nb', 'trsf_pole_nb':'trsf_pole_nb'}
        elif dataset == 'prbl':
            mecpx_build_info_dict['std_dict_grp_by_cols']     = {'no_outg_rec_nb':'outg_rec_nb', 'trsf_pole_nb':'trsf_pole_nb'}
        else:
            assert(0)
        #-------------------------
        mecpx_build_info_dict['coll_label']                   = 'coll_label'
        mecpx_build_info_dict['barplot_kwargs_shared']        = dict(facecolor='red')
        #-------------------------
        mecpx_build_info_dict['rec_nb_idfr']                  = ('index', 'outg_rec_nb')
        mecpx_build_info_dict['trsf_pole_nb_idfr']            = ('index', 'trsf_pole_nb')        
        #-------------------------
        mecpx_build_info_dict['normalize_by_time_interval']   = True
        #-------------------------
        mecpx_build_info_dict['include_power_down_minus_up']  = False
        mecpx_build_info_dict['pd_col']                       = 'Primary Power Down'
        mecpx_build_info_dict['pu_col']                       = 'Primary Power Up'
        mecpx_build_info_dict['pd_m_pu_col']                  = 'Power Down Minus Up'
        #-------------------------
        mecpx_build_info_dict['regex_to_remove_patterns']     = [
            '.*cleared.*', 
            '.*Test Mode.*'
        ]
        mecpx_build_info_dict['regex_to_remove_ignore_case']  = True
        #-------------------------
        mecpx_build_info_dict['max_total_counts']             = None
        mecpx_build_info_dict['how_max_total_counts']         = 'any'
        #-------------------------
        mecpx_build_info_dict['mecpo_idx_for_ordering']       = 0
        #-------------------------
        mecpx_build_info_dict['cpx_dfs_name']                 = 'rcpo_df_norm_by_xfmr_nSNs'
        #-------------------------
        mecpx_build_info_dict['freq']                         = '5D'
        #-----
        mecpx_build_info_dict['days_min_max_outg_td_windows'] = [
            [1,6], [6,11], [11,16], [16,21], [21,26], [26,31]
        ]
        #-----
        mecpx_build_info_dict['mecpx_an_keys']                = None
        #-----
        mecpx_build_info_dict['old_to_new_keys_dict']         = {
            'outg_td_window_1_to_6_days'  :'01-06 Days',
            'outg_td_window_6_to_11_days' :'06-11 Days',
            'outg_td_window_11_to_16_days':'11-16 Days',
            'outg_td_window_16_to_21_days':'16-21 Days',
            'outg_td_window_21_to_26_days':'21-26 Days',
            'outg_td_window_26_to_31_days':'26-31 Days'
        }
        #-------------------------
        assert(OMInput.check_mecpx_build_info_dict(mecpx_build_info_dict))
        return mecpx_build_info_dict
    

    #---------------------------------------------------------------------------
    def set_mecpx_build_args(
        self, 
        extend_any_lists = False, 
        **kwargs
    ):
        r"""
        Set any of the arguments used to build the MECPO collections.
            e.g., outg_mdlr.set_mecpx_build_args(max_total_counts=1000)
        extend_any_lists:
            If True, any list values in self.mecpx_build_info_dict will be supplemented with the values from kwargs
              instead of being replace by them.
        -----
        Note, for the purposes here the input to Utilities.supplement_dict_with_default_values may seem backwards, but
         they are, in fact, correct.
        Namely:
            to_supplmnt_dict    = kwargs
            default_values_dict = self.mecpx_build_info_dict
        From Utilities.supplement_dict_with_default_values documentation:
                If a key IS NOT contained in to_supplmnt_dict
                  ==> a new key/value pair is simply created.
                If a key IS already contained in to_supplmnt_dict
                  ==> value (to_supplmnt_dict[key]) is kept 
                      UNLESS extend_any_lists==True and to_supplmnt_dict[key] and/or default_values_dict[key] is a list,
                        in which case the two lists are combined
        ==> Since to_supplmnt_dict=kwargs below, the values from kwargs are kept over those in self.mecpx_build_info_dict
        """
        #----------------------------------------------------------------------------------------------------
        if len(kwargs)==0:
            return
        #----------------------------------------------------------------------------------------------------
        acceptable_kwargs         = list(OMInput.get_default_mecpx_build_info_dict(dataset = self.dataset).keys())
        unacceptable_kwargs_found = list(set(kwargs.keys()).difference(set(acceptable_kwargs)))
        if len(unacceptable_kwargs_found)>0:
            print(f"Unacceptable kwargs found in OMInput.set_mecpx_build_args for dataset = {self.dataset}")
            print(unacceptable_kwargs_found)
            assert(0)

        #----------------------------------------------------------------------------------------------------
        # NOTE: For our purposes here, self.mecpx_build_info_dict will actually be the default_values_dict parameter below!
        #       This is because we want to keep values in kwargs over those in self.mecpx_build_info_dict
        self.__mecpx_build_info_dict = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = kwargs, 
            default_values_dict = self.__mecpx_build_info_dict, 
            extend_any_lists    = extend_any_lists, 
            inplace             = False
        )
        #-------------------------
        assert(self.self_check_mecpx_build_info_dict())


    #---------------------------------------------------------------------------    
    @staticmethod
    def static_get_mecpx_base_dirs(
        dataset                    , 
        acq_run_date               , 
        data_date_ranges           , 
        acq_run_date_subdir_appndx = None, 
        data_dir_base              = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data', 
        assert_found               = True
    ):
        r"""
        Find the directories housing the data to be used for OMInput.static_build_and_combine_mecpo_colls_for_dates.
        A directory for each date range in data_date_ranges will be found as:
            rcpo_pkl_dir_i = os.path.join(
                data_dir_base, 
                acq_run_date_subdir, 
                date_pd_subdir, 
                ODI.get_subdir(dataset)
            )
        where acq_run_date_subdir = acq_run_date + acq_run_date_subdir_appndx (or, = acq_run_date if acq_run_date_subdir_appndx is None)
        and   date_pd_subdir corresponds to the particular date range, dataset_subdirs can be found in OutageDataInfo class (in OutageDAQ module)
        -------------------------
        dataset:
            Must be one of ['outg', 'otbl', 'prbl']
            
        acq_run_date:
            The acquisition date of the data (i.e., the date the acquisition program was run to save the needed 
              data locally)
            
        data_date_ranges:
            The date ranges to be included in the MECPOCollection.
            This should be a list of two-element.
                
        data_dir_base:
            Tbe base directory housing the data.
            Typically this should be 
                r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data' or
                r'U:\CloudData\dovs_and_end_events_data'
        """
        #--------------------------------------------------
        ODI.assert_dataset(dataset)
        #-----
        assert(Utilities.is_object_one_of_types(data_date_ranges, [list, tuple]))
        assert(Utilities.are_list_elements_lengths_homogeneous(data_date_ranges, length=2))
        #-------------------------
        acq_run_date_subdir = acq_run_date
        if acq_run_date_subdir_appndx is not None:
            acq_run_date_subdir += acq_run_date_subdir_appndx
        #-------------------------
        data_dirs_dict = {}
        for date_0, date_1 in data_date_ranges:
            date_pd_subdir = f"{date_0.replace('-','')}_{date_1.replace('-','')}"
            data_dir_i = os.path.join(
                data_dir_base, 
                acq_run_date_subdir, 
                date_pd_subdir, 
                ODI.get_subdir(dataset)
            )
            if assert_found and not os.path.isdir(data_dir_i):
                print(f'Directory DNE!\n\t{data_dir_i}\nCRASH IMMINENT!!!!!!')
                assert(0)
            assert(date_pd_subdir not in data_dirs_dict.keys())
            data_dirs_dict[date_pd_subdir] = data_dir_i
        #--------------------------------------------------
        return data_dirs_dict


    #---------------------------------------------------------------------------
    def get_mecpx_base_dirs(
        self         , 
        assert_found = True
    ):
        r"""
        Find the directories housing the data to be used for OMInput.build_and_combine_mecpo_colls_for_dates.
        A directory for each date range in data_date_ranges will be found as:
            rcpo_pkl_dir_i = os.path.join(
                data_dir_base, 
                acq_run_date_subdir, 
                date_pd_subdir, 
                self.subdir
            )
        where acq_run_date_subdir = acq_run_date + acq_run_date_subdir_appndx (or, = acq_run_date if acq_run_date_subdir_appndx is None)
        and   date_pd_subdir corresponds to the particular date range, dataset_subdirs can be found in OutageDataInfo class (in OutageDAQ module)

        See OMInput.static_get_mecpx_base_dirs for more information.
        """
        #--------------------------------------------------
        data_dirs_dict = OMInput.static_get_mecpx_base_dirs(
            dataset                    = self.dataset, 
            acq_run_date               = self.acq_run_date, 
            data_date_ranges           = self.data_date_ranges, 
            acq_run_date_subdir_appndx = self.acq_run_date_subdir_appndx, 
            data_dir_base              = self.data_dir_base, 
            assert_found               = assert_found
        )
        #--------------------------------------------------
        return data_dirs_dict
    

    #---------------------------------------------------------------------------
    @staticmethod
    def static_get_rcpo_pkl_base_dirs(
        dataset                    , 
        acq_run_date               , 
        data_date_ranges           , 
        grp_by_cols                , 
        acq_run_date_subdir_appndx = None, 
        data_dir_base              = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
    ):
        r"""
        Find the base directories housing the data (pickle files) to be used for OMInput.static_build_and_combine_mecpo_colls_for_dates.
        A directory for each date range in data_date_ranges will be found as:
            rcpo_pkl_dir_i = os.path.join(
                data_dir_base, 
                acq_run_date_subdir, 
                date_pd_subdir, 
                ODI.get_subdir(dataset), 
                data_subdir
            )
        where acq_run_date_subdir = acq_run_date + acq_run_date_subdir_appndx (or, = acq_run_date if acq_run_date_subdir_appndx is None)
        and   date_pd_subdir corresponds to the particular date range in data_date_ranges, 
        and   dataset_subdirs can be found in OutageDataInfo class (in OutageDAQ module), 
        and   data_subdir is determined from grp_by_cols
        -------------------------
        e.g., suppose data were acquired on '20250101' for the years of 2023 and 2024, split into two data acquistion runs (i.e., data_date_ranges = 
                [['2023-01-01', '2023-12-31'], ['2024-01-01', '2024-12-31']]).
              Furthermore, suppose we are building the data for dataset='outg' and grp_by_cols   == ['outg_rec_nb', 'trsf_pole_nb']
        ==> rcpo_pkl_dirs = {
                '20230401_20240831': 'data_dir_base\\20250101\\20230101_20231231\\Outages\\rcpo_dfs_GRP_BY_OUTG_AND_XFMR', 
                '20230401_20240831': 'data_dir_base\\20250101\\20240101_20241231\\Outages\\rcpo_dfs_GRP_BY_OUTG_AND_XFMR'
            }
        -------------------------
        dataset:
            Must be one of ['outg', 'otbl', 'prbl']
            
        acq_run_date:
            The acquisition date of the data (i.e., the date the acquisition program was run to save the needed 
              data locally)
            
        data_date_ranges:
            The date ranges to be included in the MECPOCollection.
            This should be a list of two-element.
            
        grp_by_cols:
            The columns which were grouped in the packaging of the data.
            Typically, this is:
                ['outg_rec_nb', 'trsf_pole_nb'] for outage data
                ['trsf_pole_nb', 'no_outg_rec_nb'] for baseline data
                
        data_dir_base:
            Tbe base directory housing the data.
            Typically this should be 
                r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data' or
                r'U:\CloudData\dovs_and_end_events_data'
        """
        #--------------------------------------------------
        data_subdir = 'rcpo_dfs'
        if grp_by_cols   == ['outg_rec_nb', 'trsf_pole_nb']:
            data_subdir += '_GRP_BY_OUTG_AND_XFMR'
        elif grp_by_cols == 'trsf_pole_nb':
            data_subdir += '_GRP_BY_XFMR'
        elif grp_by_cols == 'outg_rec_nb':
            data_subdir += '_GRP_BY_OUTG'
        elif grp_by_cols == ['trsf_pole_nb', 'no_outg_rec_nb']:
            data_subdir += '_GRP_BY_NO_OUTG_AND_XFMR'
        else:
            assert(0)
        #--------------------------------------------------
        data_dirs_dict = OMInput.static_get_mecpx_base_dirs(
            dataset                    = dataset, 
            acq_run_date               = acq_run_date, 
            data_date_ranges           = data_date_ranges, 
            acq_run_date_subdir_appndx = acq_run_date_subdir_appndx, 
            data_dir_base              = data_dir_base, 
            assert_found               = True
        )
        rcpo_pkl_dirs = {}
        for date_range_i, data_dir_i in data_dirs_dict.items():
            rcpo_pkl_dir_i = os.path.join(
                data_dir_i, 
                data_subdir
            )
            if not os.path.isdir(rcpo_pkl_dir_i):
                print(f'Directory DNE!\n\t{rcpo_pkl_dir_i}\nCRASH IMMINENT!!!!!!')
                assert(0)
            assert(date_range_i not in rcpo_pkl_dirs.keys())
            rcpo_pkl_dirs[date_range_i] = rcpo_pkl_dir_i
        #--------------------------------------------------
        return rcpo_pkl_dirs
    

    #---------------------------------------------------------------------------
    def get_rcpo_pkl_base_dirs(
        self
    ):
        r"""
        Find the base directories housing the data (pickle files) to be used for OMInput.build_and_combine_mecpo_colls_for_dates.
        A directory for each date range in data_date_ranges will be found as:
            rcpo_pkl_dir_i = os.path.join(
                self.data_dir_base, 
                acq_run_date_subdir, 
                date_pd_subdir, 
                self.subdir, 
                data_subdir
            )
        where acq_run_date_subdir = acq_run_date + acq_run_date_subdir_appndx (or, = acq_run_date if acq_run_date_subdir_appndx is None)
        and   date_pd_subdir corresponds to the particular date range in self.data_date_ranges, 
        and   dataset_subdirs can be found in OutageDataInfo class (in OutageDAQ module), 
        and   data_subdir is determined from self.grp_by_cols

        See OMInput.static_get_rcpo_pkl_base_dirs for more information.
        -------------------------
        """
        #--------------------------------------------------
        rcpo_pkl_dirs = OMInput.static_get_rcpo_pkl_base_dirs(
            dataset                    = self.dataset, 
            acq_run_date               = self.acq_run_date, 
            data_date_ranges           = self.data_date_ranges, 
            grp_by_cols                = self.grp_by_cols, 
            acq_run_date_subdir_appndx = self.acq_run_date_subdir_appndx,  
            data_dir_base              = self.data_dir_base
        )
        #--------------------------------------------------
        return rcpo_pkl_dirs


    #---------------------------------------------------------------------------
    @staticmethod
    def static_build_and_combine_mecpo_colls_for_dates(
        dataset                      , 
        acq_run_date                 , 
        data_date_ranges             , 
        grp_by_cols                  , 
        days_min_max_outg_td_windows , 
        old_to_new_keys_dict         , 
        acq_run_date_subdir_appndx   = None, 
        coll_label                   = None, 
        barplot_kwargs_shared        = None, 
        normalize_by_time_interval   = True, 
        data_dir_base                = r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data'
    ):
        r"""
        Build a MECPOCollection object for data which are split over multiple date ranges (e.g., in a typical case,
          data are acquired for each calendar year).
          
        NOTE: One could build all mecpo_coll objects first, and then use combine_mecpo_colls to combine them.
              However, this method saves on memory consumption by building one at a time and combining with
                the growing aggregate.
          
        See OMInput.static_get_rcpo_pkl_base_dirs for explanation of
            dataset, 
            acq_run_date, 
            data_date_ranges, 
            acq_run_date_subdir_appndx, 
            grp_by_cols, 
            data_dir_base
        """
        #--------------------------------------------------
        if old_to_new_keys_dict is not None:
            assert(len(old_to_new_keys_dict)==len(days_min_max_outg_td_windows))
        #-------------------------
        rcpo_pkl_dirs = OMInput.static_get_rcpo_pkl_base_dirs(
            dataset                    = dataset, 
            acq_run_date               = acq_run_date, 
            data_date_ranges           = data_date_ranges, 
            grp_by_cols                = grp_by_cols, 
            acq_run_date_subdir_appndx = acq_run_date_subdir_appndx, 
            data_dir_base              = data_dir_base
        )
        #-------------------------
        if coll_label is None:
            coll_label = ODI.get_subdir(dataset) + Utilities.generate_random_string(str_len=4)
        if barplot_kwargs_shared is None:
            barplot_kwargs_shared = dict(facecolor='tab:blue')
        #-------------------------
        mecpo_coll = None
        for i, (date_range_i, rcpo_pkl_dir_i) in enumerate(rcpo_pkl_dirs.items()):
            mecpo_coll_i = MECPOCollection(
                data_type                    = dataset, 
                mecpo_coll                   = None, 
                coll_label                   = coll_label, 
                barplot_kwargs_shared        = barplot_kwargs_shared, 
                read_and_load_all_pickles    = True, 
                pkls_base_dir                = rcpo_pkl_dir_i,
                days_min_max_outg_td_windows = days_min_max_outg_td_windows, 
                pkls_sub_dirs                = None,
                naming_tag                   = ODI.get_naming_tag(dataset),
                normalize_by_time_interval   = normalize_by_time_interval, 
                are_no_outg                  = ODI.get_is_no_outage(dataset) 
            )
            if old_to_new_keys_dict is not None:
                mecpo_coll_i.change_mecpo_an_keys(old_to_new_keys_dict)
    
            if i==0:
                mecpo_coll = copy.deepcopy(mecpo_coll_i)
            else:
                mecpo_coll = MECPOCollection.combine_two_mecpo_colls(
                    mecpo_coll_1 = mecpo_coll, 
                    mecpo_coll_2 = mecpo_coll_i, 
                    append_only  = True
                )
            del mecpo_coll_i
        #-------------------------
        return mecpo_coll
    
    
    #---------------------------------------------------------------------------
    def build_and_combine_mecpo_colls_for_dates(
        self
    ):
        r"""
        Build a MECPOCollection object for data which are split over multiple date ranges (e.g., in a typical case,
          data are acquired for each calendar year).
          
        NOTE: One could build all mecpo_coll objects first, and then use combine_mecpo_colls to combine them.
              However, this method saves on memory consumption by building one at a time and combining with
                the growing aggregate.
        """
        #--------------------------------------------------
        if self.coll_label is None:
            self.__mecpx_build_info_dict['coll_label'] = self.subdir + Utilities.generate_random_string(str_len=4)
        if self.barplot_kwargs_shared is None:
            self.__mecpx_build_info_dict['barplot_kwargs_shared'] = dict(facecolor='tab:blue')
        #-------------------------
        self.__mecpx_coll = OMInput.static_build_and_combine_mecpo_colls_for_dates(
            dataset                      = self.dataset, 
            acq_run_date                 = self.acq_run_date, 
            data_date_ranges             = self.data_date_ranges, 
            grp_by_cols                  = self.grp_by_cols, 
            days_min_max_outg_td_windows = self.days_min_max_outg_td_windows, 
            old_to_new_keys_dict         = self.old_to_new_keys_dict, 
            acq_run_date_subdir_appndx   = self.acq_run_date_subdir_appndx, 
            coll_label                   = self.coll_label, 
            barplot_kwargs_shared        = self.barplot_kwargs_shared, 
            normalize_by_time_interval   = self.normalize_by_time_interval, 
            data_dir_base                = self.data_dir_base
        )
    

    #---------------------------------------------------------------------------
    def save_mecpx_build_info_dict(
        self, 
        save_base_dir
    ):
        r"""
        """
        #-------------------------
        assert(os.path.isdir(save_base_dir))
        CustomWriter.output_dict_to_json(
            os.path.join(save_base_dir, f'mecpx_build_info_dict{self.naming_tag}.json'), 
            self.mecpx_build_info_dict
        )

    #---------------------------------------------------------------------------
    def save_mecpx_colls(
        self, 
        save_base_dir
    ):
        r"""
        """
        #-------------------------
        assert(self.mecpx_coll is not None)
        if not os.path.isdir(save_base_dir):
            print(f'OMInput.save_mecpx_colls - Directory DNE!: {save_base_dir}\nCRASH IMMINENT!')
        assert(os.path.isdir(save_base_dir))
        #-----
        with open(os.path.join(save_base_dir, f'mecpo_coll{self.naming_tag}.pkl'), 'wb') as handle:
            pickle.dump(self.mecpx_coll, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #-------------------------
        self.save_mecpx_build_info_dict(save_base_dir=save_base_dir)


    #---------------------------------------------------------------------------
    def save_merged_dfs(
        self          , 
        save_base_dir , 
        tag           = None
    ):
        r"""
        tag:
            If None,   save_name = 'merged_df_xxxx'
            Otherwise, save_name = f'merged_df_xxxx_{tag}'
        """
        #-------------------------
        assert(self.merged_df is not None)
        assert(os.path.isdir(save_base_dir))
        #-----
        if tag is None:
            appndx = ''
        else:
            appndx = f'_{tag}'
        #-----
        self.merged_df.to_pickle(os.path.join(save_base_dir, f'merged_df{self.naming_tag}{appndx}.pkl'))


    #---------------------------------------------------------------------------
    def save_counts_series(
        self          , 
        save_base_dir , 
        tag           = None
    ):
        r"""
        tag:
            If None,   save_name = 'counts_series_xxxx'
            Otherwise, save_name = f'counts_series_xxxx_{tag}'
        """
        #-------------------------
        assert(self.counts_series is not None)
        assert(os.path.isdir(save_base_dir))
        #-------------------------
        if tag is None:
            appndx = ''
        else:
            appndx = f'_{tag}'
        #-------------------------
        with open(os.path.join(save_base_dir, f'counts_series{self.naming_tag}{appndx}.pkl'), 'wb') as handle:
            pickle.dump(self.counts_series_outg, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #---------------------------------------------------------------------------
    def save_all_mecpx(
        self          , 
        save_base_dir , 
        tag           = None
    ):
        r"""
        Save all MECPOCollection objects and all associated objects (i.e., merged_dfs, counts_series, etc.)
        """
        #-------------------------
        self.save_mecpx_colls(
            save_base_dir = save_base_dir
        )
        self.save_merged_dfs(
            save_base_dir = save_base_dir, 
            tag           = tag
        )
        self.save_counts_series(
            save_base_dir = save_base_dir, 
            tag           = tag
        )    


    #---------------------------------------------------------------------------
    def load_mecpx_build_info_dict(
        self          , 
        save_base_dir , 
        verbose       = True
    ):
        r"""
        """
        #-------------------------
        tpath = os.path.join(save_base_dir, f'mecpx_build_info_dict{self.naming_tag}.json')
        assert(os.path.exists(tpath))
        #-----
        with open(tpath, 'rb') as handle:
            self.__mecpx_build_info_dict = json.load(handle)
        #-------------------------
        # Any tuples are saved/retrieved from json files as lists.
        # In most cases, this distinction does not matter, but it does for rec_nb_idfr and trsf_pole_nb_idfr
        if isinstance(self.mecpx_build_info_dict['rec_nb_idfr'], list):
            self.__mecpx_build_info_dict['rec_nb_idfr'] = tuple(self.mecpx_build_info_dict['rec_nb_idfr'])
        #-----
        if isinstance(self.mecpx_build_info_dict['trsf_pole_nb_idfr'], list):
            self.__mecpx_build_info_dict['trsf_pole_nb_idfr'] = tuple(self.mecpx_build_info_dict['trsf_pole_nb_idfr'])
        #-------------------------
        if verbose:
            print('Successfully loaded mecpx_build_info_dict')
        

    #---------------------------------------------------------------------------
    def load_mecpx_colls_from_pkls(
        self          , 
        save_base_dir , 
        verbose       = True
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        assert(os.path.isdir(save_base_dir))
        assert(os.path.exists(os.path.join(save_base_dir, f'mecpo_coll{self.naming_tag}.pkl')))
        #----------------------------------------------------------------------------------------------------
        with open(os.path.join(save_base_dir, f'mecpo_coll{self.naming_tag}.pkl'), 'rb') as handle:
            self.__mecpx_coll = pickle.load(handle)
        an_keys = self.mecpx_coll.mecpo_an_keys
        #-------------------------
        loaded = [
            f"mecpo_coll{self.naming_tag}: {os.path.join(save_base_dir, f'mecpo_coll{self.naming_tag}.pkl')}", 
        ]
        #--------------------------------------------------
        self.load_mecpx_build_info_dict(
            save_base_dir = save_base_dir, 
            verbose       = verbose
        )
        #-------------------------
        if self.mecpx_build_info_dict['mecpx_an_keys'] is None:
            self.mecpx_build_info_dict['mecpx_an_keys'] = an_keys
        else:
            assert(set(self.mecpx_build_info_dict['mecpx_an_keys']).symmetric_difference(set(an_keys))==set())
        #----------------------------------------------------------------------------------------------------
        if verbose:
            print('Successfully loaded MECPOCollection objects:')
            print(*loaded, sep='\n')


    #---------------------------------------------------------------------------
    def make_cpx_columns_equal(
        self                          ,
        same_order                    = True, 
        cols_to_init_with_empty_lists = MECPODf.std_SNs_cols(), 
        drop_empty_cpo_dfs            = False
    ):
        r"""
        """
        #-------------------------
        self.__mecpx_coll.make_cpo_columns_equal(
            same_order                    = same_order, 
            cols_to_init_with_empty_lists = cols_to_init_with_empty_lists, 
            drop_empty_cpo_dfs            = drop_empty_cpo_dfs
        )

    #---------------------------------------------------------------------------
    @staticmethod
    def generate_OMIs_dict(
        omis
    ):
        r"""
        Takes a list of OMI objects (omis) and returns a dict whose keys are unique identifiers and values are the OMI objects

        omis:
            A list of OMI objects
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(omis, [list, tuple]))
        assert(Utilities.are_all_list_elements_of_type(omis, OMInput))
        #-------------------------
        # The chances of generating the same random string for two elements is so slim that I am not
        #   going to put in place methods to deal with such a situation, I'll simply enfore that it's true
        return_dict = {Utilities.generate_random_string(str_len=10) : x for x in omis}
        assert(len(return_dict)==len(omis))
        return return_dict
    
    #---------------------------------------------------------------------------
    @staticmethod
    def make_cpx_columns_equal_between_OMIs(
        omis                          , 
        same_order                    = True, 
        cols_to_init_with_empty_lists = MECPODf.std_SNs_cols(), 
        drop_empty_cpo_dfs            = False
    ):
        r"""
        Make columns equal between the MECPOCollections in the OMIs
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(omis, [list, tuple]))
        assert(Utilities.are_all_list_elements_of_type(omis, OMInput))
        #-------------------------