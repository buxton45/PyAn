#!/usr/bin/env python

r"""
Holds OutageDAQ class.  See OutageDAQ.OutageDAQ for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import string
from enum import IntEnum
import json
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_dtype
import datetime
import time
import copy
#---------------------------------------------------------------------
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from MeterPremise import MeterPremise
#-----
from AMI_SQL import AMI_SQL, DfToSqlMap
from AMIEndEvents_SQL import AMIEndEvents_SQL
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from GenAn import GenAn
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
from SQLQuery import SQLQuery
#---------------------------------------------------------------------
#sys.path.insert(0, os.path.join(os.path.realpath('..'), 'Utilities'))
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Utilities_dt
from CustomJSON import CustomWriter

#------------------------------------------------------------------------------------------------------------------------------------------
class OutageDType(IntEnum):
    r"""
    Enum class for outage data types, which can be:
        outg = OUTaGe data
        otbl = Outage Transformers BaseLine
        prbl = PRistine BaseLine
    """
    outg  = 0
    otbl  = 1
    prbl  = 2
    unset = 3

#------------------------------------------------------------------------------------------------------------------------------------------
class OutageDataInfo:
    r"""
    Small class/container to hold dataset information.
    Originally designed as a general clas to hold information on all datasets.
    Functionality expanded to hold a specific instance as well, if dataset is supplied to constructor.
    This development is why the form is a little strange, but it is fully functional.
    """
    #--------------------------------------------------
    def __init__(
            self, 
            dataset = None
        ):
        r"""
        """
        #--------------------------------------------------
        # General stuff
        #--------------------------------------------------
        datasets = ['outg', 'otbl', 'prbl']
        dataset_subdirs = {
            'outg': r'Outages', 
            'otbl': r'OutgXfmrBaseline', 
            'prbl': r'PristineBaseline'
        }
        dataset_naming_tags = {
            'outg': '_outg', 
            'otbl': '_otbl', 
            'prbl': '_prbl'
        }
        dataset_is_no_outage = {
            'outg': False, 
            'otbl': True, 
            'prbl': True
        }
        assert(len(set(datasets).symmetric_difference(set(dataset_subdirs.keys())))==0)
        assert(len(set(datasets).symmetric_difference(set(dataset_naming_tags.keys())))==0)
        assert(len(set(datasets).symmetric_difference(set(dataset_is_no_outage.keys())))==0)
        #-------------------------
        self.datasets             = datasets
        self.dataset_subdirs      = dataset_subdirs
        self.dataset_naming_tags  = dataset_naming_tags
        self.dataset_is_no_outage = dataset_is_no_outage

        #--------------------------------------------------
        # Specific stuff, if dataset supplied
        #--------------------------------------------------
        self.__dataset      = dataset
        self.__subdir       = None
        self.__naming_tag   = None
        self.__is_no_outage = None
        if self.__dataset is not None:
            OutageDataInfo.assert_dataset(self.__dataset)
            #-----
            self.__subdir       = OutageDataInfo.get_subdir(self.__dataset)
            self.__naming_tag   = OutageDataInfo.get_naming_tag(self.__dataset)
            self.__is_no_outage = OutageDataInfo.get_is_no_outage(self.__dataset)

    #---------------------------------------------------------------------------
    def copy_constructor(
        self, 
        odi
    ):
        r"""
        Annoyingly, the follow simple solution does not work:
            self = copy.deepcopy(odi)
          neither does:
            self = OutageDataInput()
            self.__dict__ = copy.deepcopy(odi.__dict__)
    
        So, I guess it's back to headache C++ style...
        """
        #--------------------------------------------------
        assert(isinstance(odi, OutageDataInfo))
        #--------------------------------------------------
        self.datasets             = Utilities.general_copy(odi.datasets)
        self.dataset_subdirs      = Utilities.general_copy(odi.dataset_subdirs)
        self.dataset_naming_tags  = Utilities.general_copy(odi.dataset_naming_tags)
        self.dataset_is_no_outage = Utilities.general_copy(odi.dataset_is_no_outage)
        #-----
        self.__dataset            = Utilities.general_copy(odi.__dataset)
        self.__subdir             = Utilities.general_copy(odi.__subdir)
        self.__naming_tag         = Utilities.general_copy(odi.__naming_tag)
        self.__is_no_outage       = Utilities.general_copy(odi.__is_no_outage)

    #--------------------------------------------------
    def copy(self):
        r"""
        """
        #-------------------------
        return_odi = OutageDataInfo(odi=self)
        return return_odi

    #--------------------------------------------------
    @property
    def dataset(self):
        return self.__dataset
    @property
    def subdir(self):
        return self.__subdir
    @property
    def naming_tag(self):
        return self.__naming_tag
    @property
    def is_no_outage(self):
        return self.__is_no_outage

    #--------------------------------------------------
    @staticmethod
    def datasets():
        odi = OutageDataInfo()
        return odi.datasets
    @staticmethod
    def dataset_subdirs():
        odi = OutageDataInfo()
        return odi.dataset_subdirs
    @staticmethod
    def dataset_naming_tags():
        odi = OutageDataInfo()
        return odi.dataset_naming_tags
    @staticmethod
    def dataset_is_no_outage():
        odi = OutageDataInfo()
        return odi.dataset_is_no_outage
    
    #--------------------------------------------------
    @staticmethod
    def accpt_dataset(dataset):
        return dataset in OutageDataInfo.datasets()
    @staticmethod
    def assert_dataset(dataset):
        assert(OutageDataInfo.accpt_dataset(dataset=dataset))

    @staticmethod
    def get_subdir(dataset):
        OutageDataInfo.assert_dataset(dataset=dataset)
        return OutageDataInfo.dataset_subdirs()[dataset]
    
    @staticmethod
    def get_naming_tag(dataset):
        OutageDataInfo.assert_dataset(dataset=dataset)
        return OutageDataInfo.dataset_naming_tags()[dataset]
    
    @staticmethod
    def get_is_no_outage(dataset):
        OutageDataInfo.assert_dataset(dataset=dataset)
        return OutageDataInfo.dataset_is_no_outage()[dataset]



#------------------------------------------------------------------------------------------------------------------------------------------
class OutageDAQ:
    def __init__(
        self, 
        run_date, 
        date_0, 
        date_1, 
        collect_evs_sum_vw,     # boolean
        save_sub_dir, 
        td_left                 = pd.Timedelta('-31D'), 
        td_right                = pd.Timedelta('-1D'),      
        states                  = None, 
        opcos                   = None, 
        cities                  = None, 
        single_zip_xfmrs_only   = False, 
        save_end_events         = False, 
        save_dfs_to_file        = False, 
        read_dfs_from_file      = True, 
        base_dir                = os.path.join(
            Utilities.get_local_data_dir(), 
            r'dovs_and_end_events_data'
        ), 
        dates_subdir_appndx     = None, 
        run_using_slim          = False
    ):
        r"""
        date_0/date_1:
            Expecting YYYY-mm-dd or YYYYmmdd
            NOTE: An exception will be thrown if you lazily give it, e.g., '2024-06-31' since June only has 30 days
        """
        #--------------------------------------------------
        assert(save_dfs_to_file+read_dfs_from_file <=1) # Should never both read and write!
        #--------------------------------------------------
        self.run_date              = run_date
        #--------------------------------------------------
        if not Utilities.is_datetime(obj=date_0, strict=False):
            date_0 = pd.to_datetime(date_0, yearfirst=True)
        if not Utilities.is_datetime(obj=date_1, strict=False):
            date_1 = pd.to_datetime(date_1, yearfirst=True)
        assert(date_1 > date_0)
        #-----
        self.date_0                = date_0
        self.date_1                = date_1
        #--------------------------------------------------
        self.collect_evs_sum_vw    = collect_evs_sum_vw
        #--------------------------------------------------
        assert(td_left <= td_right)
        # For no great reason, I require td_left and td_right to either both be positive or both be negative.
        # If this becomes too restrictive for whatever reason, it can probably be removed.
        # All methods were designed with the idea of td_left and td_right both being negative (i.e., the windows
        #   being defined prior to the date of interest)
        assert(
            (td_left >= pd.Timedelta(0) and td_right >= pd.Timedelta(0)) or 
            (td_left <= pd.Timedelta(0) and td_right <= pd.Timedelta(0))
        )
        #-------------------------
        if not Utilities.is_timedelta(td_left):
            td_left  = pd.to_timedelta(td_left)
        if not Utilities.is_timedelta(td_right):
            td_right = pd.to_timedelta(td_right)
        #-----
        self.td_left               = td_left
        self.td_right              = td_right
        #--------------------------------------------------
        assert(self.date_1-self.date_0 > self.window_width)
        #--------------------------------------------------
        self.states                = states
        self.opcos                 = opcos
        self.cities                = cities
        #-------------------------
        self.single_zip_xfmrs_only = single_zip_xfmrs_only
        #-------------------------
        self.dates_subdir_appndx   = dates_subdir_appndx
        self.save_end_events       = save_end_events
        self.save_dfs_to_file      = save_dfs_to_file
        self.read_dfs_from_file    = read_dfs_from_file
        self.run_using_slim        = run_using_slim
        #-------------------------
        self.prep_save_locations( 
            save_sub_dir = save_sub_dir,  
            base_dir     = base_dir
        )
        #--------------------------------------------------
        self.conn_outages = None
        self.conn_aws     = None
        if not read_dfs_from_file:
            self.conn_outages = Utilities.get_utldb01p_oracle_connection()
            self.conn_aws     = Utilities.get_athena_prod_aws_connection()

        #--------------------------------------------------
        self.df_outage_OG          = None
        self.sql_outage_full       = None



    def get_summary_dict(
        self
    ):
        r"""
        DAQ settings with adjustments to a few entries (so they can be easily output into the summary JSON file)
        e.g., Timestamp is not JSON serializable, hence the need for strftime below
        e.g., Timedelta is not JSON serializable, hence the conversion to total seconds
        """
        #--------------------------------------------------
        summary_dict = dict()
        #-------------------------
        summary_dict['run_date']              = self.run_date
        #-------------------------
        summary_dict['date_0']                = self.date_0.strftime('%Y-%m-%d %H:%M:%S')
        summary_dict['date_1']                = self.date_1.strftime('%Y-%m-%d %H:%M:%S')
        #-------------------------
        summary_dict['collect_evs_sum_vw']    = self.collect_evs_sum_vw
        #-------------------------
        summary_dict['td_left']               = self.td_left.total_seconds()
        summary_dict['td_right']              = self.td_right.total_seconds()
        #-------------------------
        summary_dict['states']                = self.states
        summary_dict['opcos']                 = self.opcos
        summary_dict['cities']                = self.cities
        #-------------------------
        summary_dict['single_zip_xfmrs_only'] = self.single_zip_xfmrs_only
        #-------------------------
        summary_dict['dates_subdir_appndx']   = self.dates_subdir_appndx
        summary_dict['save_end_events']       = self.save_end_events
        summary_dict['save_dfs_to_file']      = self.save_dfs_to_file
        summary_dict['read_dfs_from_file']    = self.read_dfs_from_file
        summary_dict['run_using_slim']        = self.run_using_slim
        #-------------------------
        summary_dict['save_dir_base']         = self.save_dir_base
        summary_dict['end_events_save_args']  = self.end_events_save_args
        #--------------------------------------------------
        return summary_dict
    
    @staticmethod
    def read_summary_dict(
        save_dir_base      , 
        summary_dict_fname = 'summary_dict.json'
    ):
        r"""
        """
        #-------------------------
        assert(os.path.exists(os.path.join(save_dir_base, summary_dict_fname)))
        tmp_f = open(os.path.join(save_dir_base, summary_dict_fname))
        summary_dict = json.load(tmp_f)
        tmp_f.close()
        return summary_dict

    @property
    def window_width(
        self
    ):
        r"""
        It is assumed that the limits on the time window are EXCLUSIVE on the left and INCLUSIVE on the right, i.e., 
          (pred_date+td_left, pred_date+td_right].
            e.g., for most normal cases (where td_left and td_right are negative)
              (-6 Days, -1 Day], (-11 Days, -6 Days], etc.
            HOWEVER, note that when td_left and td_right are positive, this implies, e.g., 
              (1 Day, 6 Days], (6 Days, 11 Days], etc.
        Thus, the width of the window should be calculated as:
          window_widths_days = td_right-td_left
          e.g.:  Assume the window goes from 1 day before (td_right=-1) to 2 days before (td_left=-2):
                   ==> window_widths_days = 1 days = td_right-td_left =  -1-(-2) = 1
          e.g.:  Assume the window goes from 1 day before (td_right=-1) to 6 days before (td_left=-6):
                   ==> window_widths_days = 5 days = td_right-td_left =  -1-(-6) = 5
          e.g.:  Assume the window goes from 1 day after (td_left=+1) to 6 days after (td_right=+6):
                   ==> window_widths_days = 5 days = td_right-td_left =  6-1 = 5
        """
        #-------------------------
        window_width = self.td_right - self.td_left
        return window_width
    
    
    @property
    def window_width_days(
        self
    ):
        r"""
        """
        #-------------------------
        window_width_days = self.window_width
        if not isinstance(window_width_days, int):
            assert(Utilities.is_timedelta(window_width_days))
            window_width_days = window_width_days.days
        return window_width_days
    

    def prep_save_locations( 
        self         , 
        save_sub_dir , 
        base_dir     = os.path.join(
            Utilities.get_local_data_dir(), 
            r'dovs_and_end_events_data'
        )
    ):
        r"""
        This function sets the following attributues:
            save_dir_base, end_events_save_args
        If the needed save directories do not exist, they will be created
        """
        #----------------------------------------------------------------------------------------------------
        # DFs will be saved in save_dir_base
        # Collection of end events files will be saved in os.path.join(save_dir_base, 'EndEvents')
        #-------------------------
        dates_subdir  = self.date_0.strftime('%Y%m%d') + '_' + self.date_1.strftime('%Y%m%d')
        if self.dates_subdir_appndx is not None:
            dates_subdir += self.dates_subdir_appndx
        #-------------------------
        self.save_dir_base = os.path.join(
            base_dir, 
            self.run_date, 
            dates_subdir, 
            save_sub_dir
        )
        #-------------------------
        if self.collect_evs_sum_vw:
            self.end_events_save_args = dict(
                save_to_file = self.save_end_events, 
                save_dir     = os.path.join(self.save_dir_base, 'EvsSums'), 
                save_name    = r'events_summary.csv', 
                index        = True
            )
        else:
            self.end_events_save_args = dict(
                save_to_file = self.save_end_events, 
                save_dir     = os.path.join(self.save_dir_base, 'EndEvents'), 
                save_name    = r'end_events.csv', 
                index        = True
            )
        #-------------------------
        print(f"self.save_dir_base = {self.save_dir_base}")
        print('self.end_events_save_args')
        for k,v in self.end_events_save_args.items():
            print(f"\t{k} : {v}")
        #-------------------------
        if self.save_dfs_to_file or self.save_end_events:
            if not os.path.exists(self.save_dir_base):
                os.makedirs(self.save_dir_base)
            #-----
            if self.save_end_events and not os.path.exists(self.end_events_save_args['save_dir']):
                os.makedirs(self.end_events_save_args['save_dir'])


    def build_or_load_df_outage_OG(
        self, 
        verbose = True
    ):
        r"""
        This either loads or builds (depending on the value of self.read_dfs_from_file) df_outage_OG.
            If built, self.sql_outage_full will also be set
        df_outage_OG contains outages between date_0 and date_1 for states
        """
        #----------------------------------------------------------------------------------------------------
        csv_cols_and_types_to_convert_dict = {'CI_NB':np.int32, 'CMI_NB':np.float64, 'OUTG_REC_NB':[np.float64, np.int32]}
        start=time.time()
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_outage_OG from file: {os.path.join(self.save_dir_base, 'df_outage_OG.pkl')}")
            #-------------------------
            df_outage_OG = pd.read_pickle(os.path.join(self.save_dir_base, 'df_outage_OG.pkl'))
            df_outage_OG = Utilities_df.convert_col_types(df_outage_OG, csv_cols_and_types_to_convert_dict)
        #----------------------------------------------------------------------------------------------------
        else:
            # Find outages between date_0 and date_1 for states
            if verbose:
                print('-----'*20+f'\nFinding outages between {self.date_0} and {self.date_1} for states={self.states}\n'+'-----'*10)
            self.sql_outage_full = DOVSOutages_SQL.build_sql_std_outage(
                mjr_mnr_cause   = None, 
                include_premise = True, 
                date_range      = [self.date_0, self.date_1], 
                states          = self.states, 
                opcos           = self.opcos, 
                cities          = self.cities
            ).get_sql_statement()
            #-----
            if verbose:
                print(f'self.sql_outage_full:\n{self.sql_outage_full}\n\n')
            #-------------------------
            df_outage_OG = pd.read_sql_query(
                self.sql_outage_full, 
                self.conn_outages, 
                dtype = {
                    'CI_NB'       : np.int32, 
                    'CMI_NB'      : np.float64, 
                    'OUTG_REC_NB' : np.int32
                }
            )
            df_outage_OG = Utilities_df.convert_col_types(df_outage_OG, csv_cols_and_types_to_convert_dict)
            #-------------------------
            if self.save_dfs_to_file:
                df_outage_OG.to_pickle(os.path.join(self.save_dir_base, 'df_outage_OG.pkl'))
        #-------------------------
        self.df_outage_OG = df_outage_OG
        #-----
        if verbose:
            print(f"self.df_outage_OG.shape = {self.df_outage_OG.shape}")
            print(f"# OUTG_REC_NBs     = {self.df_outage_OG['OUTG_REC_NB'].nunique()}")
            print(f'\ntime = {time.time()-start}\n'+'-----'*20)


    @staticmethod
    def build_trsf_pole_zips_df(
        field_to_split_and_val, 
        states                 = None, 
        opcos                  = None, 
        cities                 = None
    ):
        r"""
        This will build and return trsf_pole_zips_df, mp_for_zips, and trsf_pole_df_full
        field_to_split_and_val:
            MUST be a tuple of length 2
            e.g., ('premise_nbs', PNs)
            e.g., ('premise_nbs', df_mp_outg['prem_nb'].unique().tolist())
            e.g., ('trsf_pole_nbs', trsf_pole_nbs)
                  
        """
        #--------------------------------------------------
        assert(
            isinstance(field_to_split_and_val, tuple) and 
            len(field_to_split_and_val)==2 and 
            isinstance(field_to_split_and_val[0], str)
        )
        #--------------------------------------------------
        build_sql_function_kwargs = dict(
            cols_of_interest = ['trsf_pole_nb', 'prem_nb', 'mfr_devc_ser_nbr', 'inst_ts', 'rmvl_ts', 'state_cd', 'srvc_addr_4_nm', 'county_nm'], 
            states           = states, 
            opcos            = opcos, 
            cities           = cities
        )
        #-----
        build_sql_function_kwargs[field_to_split_and_val[0]] = field_to_split_and_val[1]
        build_sql_function_kwargs['field_to_split']          = field_to_split_and_val[0]
        #--------------------------------------------------
        mp_for_zips = MeterPremise(
            df_construct_type         = DFConstructType.kRunSqlQuery,
            contstruct_df_args        = None, 
            init_df_in_constructor    = True, 
            build_sql_function        = None, 
            build_sql_function_kwargs = build_sql_function_kwargs, 
            max_n_prem_nbs            = 10000
        )
        mp_for_zips_df = mp_for_zips.df
        #--------------------------------------------------
        trsf_pole_df_full = MeterPremise.get_trsf_location_info_from_mp_df(
            mp_df            = mp_for_zips_df.copy(), 
            trsf_pole_nb_col = 'trsf_pole_nb', 
            srvc_addr_4_col  = 'srvc_addr_4_nm', 
            county_col       = 'county_nm'
        )
        #--------------------------------------------------
        start = time.time()
        trsf_pole_zips_df = mp_for_zips_df.groupby(['trsf_pole_nb', 'srvc_addr_4_nm'], as_index=False, group_keys=False).apply(
            lambda x: MeterPremise.extract_zipcode(
                input_str  = x['srvc_addr_4_nm'].values[0], 
                return_srs = True
            )
        )
        print(f"trsf_pole_zips_df build time: {time.time()-start}")
        #--------------------------------------------------
        trsf_pole_zips_df = trsf_pole_zips_df.drop(columns=['srvc_addr_4_nm']).drop_duplicates()
        #--------------------------------------------------
        return dict(
            trsf_pole_zips_df = trsf_pole_zips_df, 
            mp_for_zips_df    = mp_for_zips_df, 
            trsf_pole_df_full = trsf_pole_df_full
        )


    @staticmethod
    def set_random_date_interval_for_each_entry_in_df(
        df, 
        date_0, 
        date_1, 
        window_width, 
        rand_seed      = None, 
        placement_cols = ['t_search_min', 't_search_max'], 
        inplace        = True
    ):
        assert(len(placement_cols)==2)
        if not inplace:
            df = df.copy()
        df[placement_cols] = df.apply(
            lambda _: Utilities_dt.get_random_datetime_interval_between(
                date_0       = date_0, 
                date_1       = date_1, 
                window_width = window_width, 
                rand_seed    = rand_seed
            ), 
            axis        = 1, 
            result_type = 'expand'
        )
        return df
    
    
    @staticmethod
    def set_random_date_interval_in_df(
        df             , 
        date_0         , 
        date_1         , 
        window_width   , 
        groupby        = None, 
        rand_seed      = None, 
        placement_cols = ['t_search_min', 't_search_max'], 
        inplace        = True
    ):
        r"""
        Sets single interval for entire df UNLESS groupby is not None
        If groupby is not None, then a single interval is set for EACH GROUP
          - NOTE: In this case, rand_seed must be an int (or other object to which 
          adding an int makes sense)
        
        It is not suggested, but if one wanted a unique interval for each entry in df,
        one could use this function after grouping df by unique indentifier (row index, etc)
          - Or, one could use OutageDAQ.set_random_date_interval_for_each_entry_in_df
        """
        #-------------------------
        assert(len(placement_cols)==2)
        if not inplace:
            df = df.copy()
        #-------------------------
        if groupby is not None:
            # Below, groupby should be set to None in OutageDAQ.set_random_date_interval_in_df, 
            #   as the df.groupby(groupby) essentially sends each group's DataFrame into
            #   this function (and setting groupby=None ensures the code doesn't end up right back here!)
            # Note: inplace already handled above, so inplace=True below
            # Note: A single rand_seed value would make each group have the exact same interval!
            #       Therefore, if rand_seed is set, somehow each call to this function must have a unique
            #       rand_seed value!.  The simplest way to achieve this is to use rand_seed = rand_seed+x.shape[0]
            #       where x is the df for a particular group.  Note that groups with the same number of entries will
            #       have the same seed, and therefore the same time iterval.
            #       This functionality is intended for debugging/development, therefore rand_seed should be set
            #       to done when run for production.
            if rand_seed is not None:
                try:
                    rand_seed+1
                except:
                    assert(0)
            df=df.groupby(groupby).apply(
                lambda x: OutageDAQ.set_random_date_interval_in_df(
                    df             = x, 
                    date_0         = date_0, 
                    date_1         = date_1, 
                    window_width   = window_width, 
                    groupby        = None, 
                    rand_seed      = rand_seed if rand_seed is None else x.shape[0]+rand_seed, 
                    placement_cols = placement_cols,
                    inplace        = True
                )
            )
            return df
        #-------------------------
        interval = Utilities_dt.get_random_datetime_interval_between(
            date_0       = date_0, 
            date_1       = date_1, 
            window_width = window_width, 
            rand_seed    = rand_seed
        )
        df[placement_cols] = interval
        return df
    
    
    @staticmethod
    def set_random_date_interval_in_df_by_xfmr(
        df             , 
        date_0         , 
        date_1         , 
        window_width   , 
        xfmr_col       = 'trsf_pole_nb', 
        rand_seed      = None, 
        placement_cols = ['t_search_min', 't_search_max'], 
        inplace        = True
    ):
        r"""
        Sets a single interval for each unique value in xfmr_col.
        The functionality here should be the same as OutageDAQ.set_random_date_interval_in_df with groupby=xfmr_col, 
          BUT THIS METHOD IS MUCH MUCH FASTER!
        """
        #-------------------------
        assert(len(placement_cols)==2)
        if not inplace:
            df = df.copy()
        #-------------------------
        trsf_pole_nbs = df[xfmr_col].unique().tolist()
        #-------------------------
        trsf_pole_nbs_w_intervals = {}
        for trsf_pole_nb in trsf_pole_nbs:
            interval = Utilities_dt.get_random_datetime_interval_between(
                date_0       = date_0, 
                date_1       = date_1, 
                window_width = window_width, 
                rand_seed    = rand_seed
            )
            assert(trsf_pole_nb not in trsf_pole_nbs_w_intervals)
            trsf_pole_nbs_w_intervals[trsf_pole_nb] = interval
        assert(len(trsf_pole_nbs)==len(trsf_pole_nbs_w_intervals))
        #-------------------------
        trsf_pole_nbs_w_intervals_df = pd.DataFrame.from_dict(trsf_pole_nbs_w_intervals, orient='index', columns=placement_cols)
        og_shape = df.shape
        df = pd.merge(
            df, 
            trsf_pole_nbs_w_intervals_df, 
            how         = 'left', 
            left_on     = xfmr_col, 
            right_index = True
        )
        assert(df.shape[0]==og_shape[0])
        assert(df.shape[1]==og_shape[1]+2)
        #-------------------------
        return df
    

    @staticmethod
    def find_clean_subwindows_for_group(
        final_df_i                      , 
        min_window_width                , 
        include_is_first_after_outg_col = True, 
        t_clean_min_col                 = 't_clean_min', 
        t_clean_max_col                 = 't_clean_max', 
        return_t_search_min_col         = 't_search_min', 
        return_t_search_max_col         = 't_search_max'
    ):
        r"""
        Designed for use in OutageDAQOtBL.find_clean_window_for_group when search_window_strategy=='all_subwindows'.
        For the clean windows in final_df_i, this will find all acceptable subwindows.
          So, e.g., if a clean window is of length 171 days and min_window_width=30 days, this will find 5 acceptable subwindows.
          
        It is expected that final_df_i contains data for a single group (typically, a single transformer).
        The DF will have as many rows as clean periods found (see OutageDAQOtBL.find_clean_window_for_group for more information).
        It is expected that the buffer times have already been taken care of inf final_df_i when defining t_clean_min and max (as
          is the case when this function is used within OutageDAQOtBL.find_clean_window_for_group)
        """
        #-------------------------
        # Generate random string to be safe when dong all the index re-naming below
        idx_rndm = Utilities.generate_random_string()
    
        # Grab final_df_i index name for later
        final_df_i_idx_nm = final_df_i.index.name
    
        # Iterate over each row in final_df_i, find acceptable subwindows, and add to return_dfs collection.
        # As noted in the above documentation, final_df_i should contain one row for each clean period
        return_dfs = []
        for idx, row_i in final_df_i.iterrows():
            t_clean_min_i  = row_i[t_clean_min_col]
            t_clean_max_i  = row_i[t_clean_max_col]
            n_subwindows_i = np.floor((t_clean_max_i-t_clean_min_i)/min_window_width).astype(int)
            #-------------------------
            windows_i = []
            for i_window in range(n_subwindows_i):
                window_i_min = t_clean_min_i + i_window*min_window_width
                window_i_max = t_clean_min_i + (i_window+1)*min_window_width
                windows_i.append([window_i_min, window_i_max])
            #-------------------------
            # Sanity check
            assert(windows_i[-1][1] <= t_clean_max_i)
            #-------------------------
            # Create len(windows_i) copies of row_i and merge (concat with axis=1) with windows_i
            #-----
            # Need to call reset_index on rows_i for merge to work, but want to use original index later,
            #   so rename original to more easily grab later
            # NOTE: In newer versions of pandas (>=1.5) one can use the names argument of .reset_index,
            #       allowing the merge to happen in a single line
            rows_i            = pd.concat([pd.DataFrame(row_i).T]*len(windows_i))
            assert(rows_i.index.nlevels==1)
            idx_nm_og         = 'index_og'+idx_rndm
            rows_i.index.name = idx_nm_og
            rows_i            =  rows_i.reset_index()
            #-----
            return_df_i = pd.concat([
                rows_i, 
                pd.DataFrame(windows_i, columns=[return_t_search_min_col, return_t_search_max_col])
            ], axis=1)
            #-----
            if include_is_first_after_outg_col:
                return_df_i['is_first_after_outg']        = 0
                return_df_i.loc[0, 'is_first_after_outg'] = 1
            #-----
            return_dfs.append(return_df_i)
        #-------------------------
        # Combine all return_dfs into return_df
        return_df = pd.concat(return_dfs)
        #-------------------------
        # Join together the original index and the new index (new index should be 0-len(subwindows_i)-1)
        # This will allow one to track where subwindows came from, in case one needs to debug or whatever
        # As noted above, in newer versions of pandas (>=1.5) one can use the names argument of .reset_index
        assert(return_df.index.nlevels==1)
        idx_nm_new           = 'index_new'+idx_rndm
        return_df.index.name = idx_nm_new
        return_df            = return_df.reset_index()
        #-----
        idx_nm_final            = 'index_final'+idx_rndm
        return_df[idx_nm_final] = return_df[[idx_nm_og, idx_nm_new]].astype(str).agg('_'.join, axis=1)
        # Set index to combination of og and new, rename the index to match that of final_df_i, 
        #   and drop idx_nm_og and idx_nm_new columns as they are no longer needed
        return_df            = return_df.set_index(idx_nm_final)
        return_df.index.name = final_df_i_idx_nm
        return_df            = return_df.drop(columns=[idx_nm_og, idx_nm_new])
        #-------------------------
        return return_df



    @staticmethod
    def standardize_bsln_time_interval_infos_df(
        df               , 
        PN_regex         = r"prem(?:ise)?_nbs?", 
        t_min_regex      = r"t(?:_search)?_min", 
        t_max_regex      = r"t(?:_search)?_max", 
        drop_gpd_for_sql = True, 
        return_gpd_cols  = False
    ):
        r"""
        Standardized output columns for premise, t_min, t_max:
            PN           : premise number
            t_search_min : t_min
            t_search_max : t_max
    
        This function will cause a crash if it fails!
        If one wants a softer version, use OutageDAQ.try_to_standardize_bsln_time_interval_infos_df
    
        return_gpd_cols:
            If True, returns (df, gpd_cols)
            For DAQ, the data are always grouped by t_min/t_max.
                Additional grouped columns are tagged with _GPD_FOR_SQL
        """
        #--------------------------------------------------
        PN_col = Utilities_df.find_cols_with_regex(
            df            = df, 
            regex_pattern = PN_regex, 
            ignore_case   = True
        )
        #-----
        t_min_col = Utilities_df.find_cols_with_regex(
            df            = df, 
            regex_pattern = t_min_regex, 
            ignore_case   = True
        )
        #-----
        t_max_col = Utilities_df.find_cols_with_regex(
            df            = df, 
            regex_pattern = t_max_regex, 
            ignore_case   = True
        )
        #--------------------------------------------------
        assert(len(PN_col)==len(t_min_col)==len(t_max_col)==1)
        #--------------------------------------------------
        PN_col    = PN_col[0]
        t_min_col = t_min_col[0]
        t_max_col = t_max_col[0]
        #-------------------------
        rename_dict = {
            PN_col    : 'PN', 
            t_min_col : 't_search_min', 
            t_max_col : 't_search_max', 
        }
        #-------------------------
        df = df.rename(columns = rename_dict)
        #--------------------------------------------------
        if return_gpd_cols:
            gpd_cols              = ['t_search_min', 't_search_max']
            gpd_for_sql_cols_dict = AMI_SQL.get_rename_dict_for_gpd_for_sql_cols(df=df, col_level=-1)
            if len(gpd_for_sql_cols_dict) > 0:
                if drop_gpd_for_sql:
                    gpd_cols.extend(gpd_for_sql_cols_dict.values())
                else:
                    gpd_cols.extend(gpd_for_sql_cols_dict.keys())
        #--------------------------------------------------
        if drop_gpd_for_sql:
            df = AMI_SQL.rename_all_cols_with_gpd_for_sql_appendix(df=df)
        #--------------------------------------------------
        df['t_search_min'] = pd.to_datetime(df['t_search_min'])
        df['t_search_max'] = pd.to_datetime(df['t_search_max'])
        #-----
        if 'doi' in df.columns:
            df['doi'] = pd.to_datetime(df['doi'])
        #-----
        if 'doi_GPD_FOR_SQL' in df.columns:
            df['doi_GPD_FOR_SQL'] = pd.to_datetime(df['doi_GPD_FOR_SQL'])

        #--------------------------------------------------
        if return_gpd_cols:
            return df, gpd_cols
        return df
    
    
    @staticmethod
    def try_to_standardize_bsln_time_interval_infos_df(
        df               , 
        PN_regex         = r"prem(?:ise)?_nbs?", 
        t_min_regex      = r"t(?:_search)?_min", 
        t_max_regex      = r"t(?:_search)?_max", 
        drop_gpd_for_sql = True, 
        return_gpd_cols  = False
    ):
        r"""
        Standardized output columns for premise, t_min, t_max:
            PN           : premise number
            t_search_min : t_min
            t_search_max : t_max
    
        This function will cause a crash if it fails!
        If one wants a softer version, use OutageDAQ.try_to_standardize_bsln_time_interval_infos_df
    
        return_gpd_cols:
            If True, returns (df, gpd_cols)
            For DAQ, the data are always grouped by t_min/t_max.
                Additional grouped columns are tagged with _GPD_FOR_SQL
        """
        #--------------------------------------------------
        try:
            df, gpd_cols = OutageDAQ.standardize_bsln_time_interval_infos_df(
                df               = df, 
                PN_regex         = PN_regex, 
                t_min_regex      = t_min_regex, 
                t_max_regex      = t_max_regex, 
                drop_gpd_for_sql = drop_gpd_for_sql, 
                return_gpd_cols  = True
            )
            if return_gpd_cols:
                return df, gpd_cols
            return df
        except:
            print('!!!!! OutageDAQ.try_to_standardize_bsln_time_interval_infos_df FAILED !!!!!')
            if return_gpd_cols:
                return df, []
            return df


    @staticmethod
    def get_bsln_time_interval_infos_df_from_summary_file(
        summary_path         , 
        alias                = 'mapping_table', 
        include_summary_path = False, 
        consolidate          = False, 
        PN_regex             = r"prem(?:ise)?_nbs?", 
        t_min_regex          = r"t(?:_search)?_min", 
        t_max_regex          = r"t(?:_search)?_max", 
        drop_gpd_for_sql     = True, 
        return_gpd_cols      = False, 
        verbose              = True
    ):
        r"""
        Returns a pd.DataFrame version of MECPOAn.get_bsln_time_interval_infos_from_summary_file
        """
        #--------------------------------------------------
        assert(os.path.exists(summary_path))
        #-------------------------
        f = open(summary_path)
        summary_json_data = json.load(f)
        assert('sql_statement' in summary_json_data)
        sql_statement = summary_json_data['sql_statement']
        #-------------------------
        f.close()
    
        #--------------------------------------------------
        return_df = AMI_SQL.extract_mapping_taple_from_sql_statement(
            sql_statement = sql_statement, 
            alias         = alias
        )
        #-------------------------
        if include_summary_path:
            return_df['summary_path'] = summary_path
        #--------------------------------------------------
        return_df, gpd_cols = OutageDAQ.standardize_bsln_time_interval_infos_df(
            df               = return_df, 
            PN_regex         = PN_regex, 
            t_min_regex      = t_min_regex, 
            t_max_regex      = t_max_regex, 
            drop_gpd_for_sql = drop_gpd_for_sql, 
            return_gpd_cols  = True
        )
        return_df = return_df.drop_duplicates()
        #--------------------------------------------------
        if consolidate:
            return_df = Utilities_df.consolidate_df(
                df                                  = return_df, 
                groupby_cols                        = gpd_cols, 
                cols_shared_by_group                = None, 
                cols_to_collect_in_lists            = None, 
                cols_to_drop                        = None, 
                as_index                            = True, 
                include_groupby_cols_in_output_cols = False, 
                allow_duplicates_in_lists           = False, 
                allow_NaNs_in_lists                 = False, 
                recover_uniqueness_violators        = True, 
                gpby_dropna                         = True, 
                rename_cols                         = None, 
                custom_aggs_for_list_cols           = None, 
                verbose                             = verbose
            )
        #--------------------------------------------------
        if return_gpd_cols:
            return return_df, gpd_cols
        return return_df 
    

    @staticmethod
    def get_bsln_time_interval_infos_df_from_summary_files(
        summary_paths         , 
        alias                 = 'mapping_table', 
        include_summary_paths = False, 
        consolidate           = False, 
        PN_regex              = r"prem(?:ise)?_nbs?", 
        t_min_regex           = r"t(?:_search)?_min", 
        t_max_regex           = r"t(?:_search)?_max", 
        drop_gpd_for_sql      = True, 
        return_gpd_cols       = False, 
        verbose               = True
    ):
        r"""
        Handles multiple summary files
    
        NOTE: Drop duplicates description below mainly pertains to old MECPOAn method.  Keep here for reminder though
            Note: drop_duplicates will remove rows if indices are different (but all columns equal)
                  Therefore, if consolidate==True, this should only be done AFTER drop duplicates
                  This explains why consolidate=False in the call to OutageDAQ.get_bsln_time_interval_infos_df_from_summary_file
            Note: The reason for drop duplicates is for the case where a collection is split over mulitple
                  files/runs (i.e., the asynchronous case)
        """
        dfs       = []
        gpd_cols  = None
        for i,summary_path_i in enumerate(summary_paths):
            df_i, gpd_cols_i = OutageDAQ.get_bsln_time_interval_infos_df_from_summary_file(
                summary_path         = summary_path_i, 
                alias                = alias, 
                include_summary_path = include_summary_paths, 
                consolidate          = False, 
                PN_regex             = PN_regex, 
                t_min_regex          = t_min_regex, 
                t_max_regex          = t_max_regex, 
                drop_gpd_for_sql     = drop_gpd_for_sql, 
                return_gpd_cols      = True, 
                verbose              = verbose
            )
            #-------------------------
            if i==0:
                gpd_cols = gpd_cols_i
            #-------------------------
            assert(set(gpd_cols).symmetric_difference(set(gpd_cols_i))==set())
            dfs.append(df_i)
        #-------------------------
        return_df = Utilities_df.concat_dfs(
            dfs                  = dfs, 
            axis                 = 0, 
            make_col_types_equal = False
        )
        return_df = return_df.drop_duplicates()
        #--------------------------------------------------
        # It is possible that a group was split over multiple files/runs
        #   e.g., if not run using slim, then a particular outage/transformer group might have premises split
        #     across neighboring files
        # Consolidation method below will combine such entries
        #-------------------------
        if consolidate:
            return_df = Utilities_df.consolidate_df(
                df                                  = return_df, 
                groupby_cols                        = gpd_cols, 
                cols_shared_by_group                = None, 
                cols_to_collect_in_lists            = None, 
                cols_to_drop                        = None, 
                as_index                            = True, 
                include_groupby_cols_in_output_cols = False, 
                allow_duplicates_in_lists           = False, 
                allow_NaNs_in_lists                 = False, 
                recover_uniqueness_violators        = True, 
                gpby_dropna                         = True, 
                rename_cols                         = None, 
                custom_aggs_for_list_cols           = None, 
                verbose                             = verbose
            )
        #--------------------------------------------------
        if return_gpd_cols:
            return return_df, gpd_cols
        return return_df 
    

    @staticmethod
    def get_bsln_time_interval_infos_df_for_data_in_dir(
        data_dir                , 
        file_path_glob          = r'events_summary_[0-9]*.csv', 
        alias                   = 'mapping_table', 
        include_summary_paths   = False, 
        consolidate             = False, 
        PN_regex                = r"prem(?:ise)?_nbs?", 
        t_min_regex             = r"t(?:_search)?_min", 
        t_max_regex             = r"t(?:_search)?_max", 
        drop_gpd_for_sql        = True, 
        return_gpd_cols         = False, 
        verbose                 = True
    ):
        r"""
        data_dir should point to the directory containing the actual data CSV files.
            e.g., data_dir = r'...\LocalData\dovs_and_end_events_data\20250318\20240401_20240630\Outages\EvsSums'
            It is expected that the summary files live in os.path.join(data_dir, 'summary_files')

        If using raw meter events, instead of events summary data, one should likely change
        file_path_glob = r'events_summary_[0-9]*.csv' ==> r'end_events_[0-9]*.csv'
        """
        #-------------------------
        assert(os.path.isdir(data_dir))
        data_paths = Utilities.find_all_paths(base_dir=data_dir, glob_pattern=file_path_glob)
        if len(data_paths)==0:
            print(f'No files found in base_dir={data_dir} with glob_pattern={file_path_glob}')
        assert(len(data_paths)>0)
        #-----
        summary_paths = [AMIEndEvents.find_summary_file_from_csv(x) for x in data_paths]
        assert(len(summary_paths)>0)
        assert(len(summary_paths)==len(data_paths))
        #-------------------------
        bsln_time_infos_df, gpd_cols = OutageDAQ.get_bsln_time_interval_infos_df_from_summary_files(
            summary_paths         = summary_paths, 
            alias                 = alias, 
            include_summary_paths = include_summary_paths, 
            consolidate           = consolidate, 
            PN_regex              = PN_regex, 
            t_min_regex           = t_min_regex, 
            t_max_regex           = t_max_regex, 
            drop_gpd_for_sql      = drop_gpd_for_sql, 
            return_gpd_cols       = True, 
            verbose               = verbose
        )
        #-------------------------
        if return_gpd_cols:
            return bsln_time_infos_df, gpd_cols
        return bsln_time_infos_df
    

    @staticmethod
    def build_baseline_time_infos_df_simple(
        data_dir                , 
        min_req                 = False, 
        file_path_glob          = r'events_summary_[0-9]*.csv', 
        alias                   = 'mapping_table', 
        include_summary_paths   = False, 
        consolidate             = False, 
        PN_regex                = r"prem(?:ise)?_nbs?", 
        t_min_regex             = r"t(?:_search)?_min", 
        t_max_regex             = r"t(?:_search)?_max", 
        drop_gpd_for_sql        = True, 
        return_gpd_cols         = False, 
        verbose                 = True
    ):
        r"""
        !!!!! Probably use OutageDAQ.build_baseline_time_infos_df method instead !!!!!
        -----
        data_dir should point to the directory containing the actual data CSV files.
            e.g., data_dir = r'...\LocalData\dovs_and_end_events_data\20250318\20240401_20240630\Outages\EvsSums'
            It is expected that the summary files live in os.path.join(data_dir, 'summary_files')
    
        If using raw meter events, instead of events summary data, one should likely change
        file_path_glob = r'events_summary_[0-9]*.csv' ==> r'end_events_[0-9]*.csv'
    
        min_req:
            If True, the only columns which are returned will be gpd_cols, which should include min_req_cols
        """
        #--------------------------------------------------
        min_req_cols = ['t_search_min', 't_search_max']
        #--------------------------------------------------
        time_infos_df, gpd_cols = OutageDAQ.get_bsln_time_interval_infos_df_for_data_in_dir(
            data_dir                = data_dir, 
            file_path_glob          = file_path_glob, 
            alias                   = alias, 
            include_summary_paths   = include_summary_paths, 
            consolidate             = consolidate, 
            PN_regex                = PN_regex, 
            t_min_regex             = t_min_regex, 
            t_max_regex             = t_max_regex, 
            drop_gpd_for_sql        = drop_gpd_for_sql, 
            return_gpd_cols         = True, 
            verbose                 = verbose
        )
        #-------------------------
        if consolidate:
            time_infos_df = time_infos_df.reset_index()
        #-------------------------
        assert(set(min_req_cols).difference(set(time_infos_df.columns.tolist()))==set())
        #-------------------------
        if min_req:
            assert(set(min_req_cols).difference(set(gpd_cols))==set())
            time_infos_df = time_infos_df[gpd_cols].copy()
            time_infos_df = Utilities_df.move_cols_to_back(
                df           = time_infos_df, 
                cols_to_move = min_req_cols
            )
        #-------------------------
        # .drop_dupicates is especially needed if min_req==True, but is safe to run in any case
        time_infos_df = time_infos_df.drop_duplicates()
        #-------------------------
        if return_gpd_cols:
            return time_infos_df, gpd_cols
        return time_infos_df
    

    @staticmethod
    def build_baseline_time_infos_df(
        data_dir_base           , 
        min_req                 = False, 
        summary_dict_fname      = 'summary_dict.json', 
        alias                   = 'mapping_table', 
        include_summary_paths   = False, 
        consolidate             = False, 
        PN_regex                = r"prem(?:ise)?_nbs?", 
        t_min_regex             = r"t(?:_search)?_min", 
        t_max_regex             = r"t(?:_search)?_max", 
        drop_gpd_for_sql        = True, 
        return_gpd_cols         = False, 
        verbose                 = True
    ):
        r"""
        data_dir_base should point one directory above that containing the actual data CSV files.
            i.e., should be the save_dir_base attribute of OutageDAQ class
            e.g., data_dir = r'...\LocalData\dovs_and_end_events_data\20250318\20240401_20240630\Outages'
            It is expected that the summary files live in os.path.join(data_dir, 'summary_files')
    
        summary_dict_fname:
            Must exist in os.path.join(data_dir_base, summary_dict_fname) and must be a json file (output by OutageDAQ
              methods, typically at end of collect_events call)
    
        min_req:
            If True, the only columns which are returned will be gpd_cols, which should include min_req_cols
        """
        #--------------------------------------------------
        assert(os.path.exists(os.path.join(data_dir_base, summary_dict_fname)))
        summary_dict = OutageDAQ.read_summary_dict(        
            save_dir_base      = data_dir_base, 
            summary_dict_fname = summary_dict_fname)
        
        #-------------------------
        dataset  = summary_dict['dataset']
        save_dir = summary_dict['end_events_save_args']['save_dir']
        
        #-------------------------
        td_left  = pd.to_timedelta(f"{summary_dict['td_left']}seconds")
        td_right = pd.to_timedelta(f"{summary_dict['td_right']}seconds")
        
        #-------------------------
        save_name       = summary_dict['end_events_save_args']['save_name']
        save_name_regex = Utilities.append_to_path(
            save_path                     = save_name, 
            appendix                      = r'_[0-9]*', 
            ext_to_find                   = '.csv', 
            append_to_end_if_ext_no_found = False
        )
        
        #--------------------------------------------------
        time_infos_df, gpd_cols = OutageDAQ.build_baseline_time_infos_df_simple(
            data_dir                = save_dir, 
            min_req                 = min_req, 
            file_path_glob          = save_name_regex, 
            alias                   = alias, 
            include_summary_paths   = include_summary_paths, 
            consolidate             = consolidate, 
            PN_regex                = PN_regex, 
            t_min_regex             = t_min_regex, 
            t_max_regex             = t_max_regex, 
            drop_gpd_for_sql        = drop_gpd_for_sql, 
            return_gpd_cols         = True, 
            verbose                 = verbose
        )
        
        #--------------------------------------------------
        if dataset == 'prbl':
            return_window_strategy = summary_dict['return_window_strategy']
            if return_window_strategy == 'entire':
                #-------------------------
                gpd_cols_time     = ['t_search_min', 't_search_max', 'doi']
                assert(set(gpd_cols_time).difference(set(time_infos_df.columns))==set())
                #-------------------------
                assert(
                    time_infos_df['t_search_min'].nunique()==1 and
                    time_infos_df['t_search_max'].nunique()==1
                )
                #-------------------------
                time_infos_df = time_infos_df.rename(columns = {
                    't_search_min' : 't_clean_min', 
                    't_search_max' : 't_clean_max'
                })
                #-------------------------
                window_width = td_right - td_left
                
                #--------------------------------------------------
                time_infos_df = OutageDAQ.find_clean_subwindows_for_group(
                    final_df_i                      = time_infos_df, 
                    min_window_width                = window_width, 
                    include_is_first_after_outg_col = False, 
                    t_clean_min_col                 = 't_clean_min', 
                    t_clean_max_col                 = 't_clean_max', 
                    return_t_search_min_col         = 't_search_min', 
                    return_t_search_max_col         = 't_search_max'
                )
                
                #-------------------------
                time_infos_df['doi']   = time_infos_df['t_search_min'] - td_left
                time_infos_df['doi_r'] = time_infos_df['t_search_max'] - td_right
                if time_infos_df['doi'].equals(time_infos_df['doi_r']):
                    time_infos_df = time_infos_df.drop(columns=['doi_r'])
                else:
                    if verbose:
                        print("!!!!! WARNING: OutageDAQ.build_baseline_time_infos_df!!!!!\n\tdoi and doi_r columns disagree, so both will be kept!")
                #-------------------------
                time_infos_df                     = time_infos_df.drop(columns=['t_clean_min', 't_clean_max'])
                time_infos_df                     = time_infos_df.reset_index(drop=True)
                #-------------------------
                rec_nb_col                       = summary_dict['rec_nb_col']
                time_infos_df[f'{rec_nb_col}_0'] = time_infos_df[rec_nb_col].copy()
                time_infos_df[rec_nb_col]        = time_infos_df[rec_nb_col] + '_' + time_infos_df.index.astype(str)
        
        #--------------------------------------------------
        if return_gpd_cols:
            return time_infos_df, gpd_cols
        return time_infos_df

#------------------------------------------------------------------------------------------------------------------------------------------
class OutageDAQOutg(OutageDAQ):
    r"""
    Class to hold methods specific to Outage (i.e., signal) data acquisition
    """
    def __init__(
        self                    , 
        run_date                , 
        date_0                  , 
        date_1                  , 
        collect_evs_sum_vw      ,     # boolean
        save_sub_dir            , 
        td_left                 = pd.Timedelta('-31D'), 
        td_right                = pd.Timedelta('-1D'),     
        states                  = None, 
        opcos                   = None, 
        cities                  = None, 
        single_zip_xfmrs_only   = False, 
        save_end_events         = False, 
        save_dfs_to_file        = False, 
        read_dfs_from_file      = True, 
        base_dir                = os.path.join(
            Utilities.get_local_data_dir(), 
            r'dovs_and_end_events_data'
        ), 
        dates_subdir_appndx     = None, 
    ):
        r"""
        """
        #--------------------------------------------------
        self.df_mp_outg     = None
        self.df_outage      = None
        self.df_outage_slim = None

        #--------------------------------------------------
        super().__init__(
            run_date                = run_date, 
            date_0                  = date_0, 
            date_1                  = date_1, 
            collect_evs_sum_vw      = collect_evs_sum_vw, 
            save_sub_dir            = save_sub_dir, 
            td_left                 = td_left, 
            td_right                = td_right, 
            states                  = states, 
            opcos                   = opcos, 
            cities                  = cities, 
            single_zip_xfmrs_only   = single_zip_xfmrs_only, 
            save_end_events         = save_end_events, 
            save_dfs_to_file        = save_dfs_to_file, 
            read_dfs_from_file      = read_dfs_from_file, 
            base_dir                = base_dir, 
            dates_subdir_appndx     = dates_subdir_appndx, 
            run_using_slim          = False, 
        )

    def get_summary_dict(
        self
    ):
        r"""
        DAQ settings with adjustments to a few entries (so they can be easily output into the summary JSON file)
        e.g., Timestamp is not JSON serializable, hence the need for strftime below
        e.g., Timedelta is not JSON serializable, hence the conversion to total seconds
        """
        #--------------------------------------------------
        summary_dict = super().get_summary_dict()
        #--------------------------------------------------
        summary_dict['dataset']    = 'outg'
        summary_dict['rec_nb_col'] = 'outg_rec_nb'
        #--------------------------------------------------
        return summary_dict
    
    def save_summary_dict(
        self
    ):
        r"""
        """
        #-------------------------
        summary_dict = self.get_summary_dict()
        #-----
        assert(os.path.isdir(self.save_dir_base))
        #-----
        CustomWriter.output_dict_to_json(
            os.path.join(self.save_dir_base, 'summary_dict.json'), 
            summary_dict
        )
    

    @staticmethod
    def build_active_MP_for_outages_df(
        df_outage, 
        prem_nb_col, 
        df_mp_curr                 = None, 
        df_mp_hist                 = None, 
        assert_all_PNs_found       = True, 
        drop_inst_rmvl_cols        = False, 
        outg_rec_nb_col            = 'OUTG_REC_NB',  #TODO!!!!!!!!!!!!!!!!!!!!!!! what if index?!
        is_slim                    = False, 
        dt_on_ts_col               = 'DT_ON_TS', 
        df_off_ts_full_col         = 'DT_OFF_TS_FULL', 
        consolidate_PNs_batch_size = 1000, 
        df_mp_serial_number_col    = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col          = 'prem_nb', 
        df_mp_install_time_col     = 'inst_ts', 
        df_mp_removal_time_col     = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col     = 'trsf_pole_nb', 
        addtnl_mp_cols             = None
    ):
        r"""
        Similar to build_active_MP_for_outages
        """
        #-------------------------
        assert(
            prem_nb_col        in df_outage.columns and 
            dt_on_ts_col       in df_outage.columns and 
            df_off_ts_full_col in df_outage.columns
        )
        #-------------------------
        if not is_slim:
            PNs = df_outage[prem_nb_col].unique().tolist()
        else:
            PNs = Utilities_df.consolidate_column_of_lists(
                df           = df_outage, 
                col          = prem_nb_col, 
                sort         = True,
                include_None = False,
                batch_size   = consolidate_PNs_batch_size, 
                verbose      = False
            )
        #-----
        PNs = [x for x in PNs if pd.notna(x)]
        #-------------------------
        mp_df_curr_hist_dict = MeterPremise.build_mp_df_curr_hist_for_PNs(
            PNs                    = PNs, 
            mp_df_curr             = df_mp_curr,
            mp_df_hist             = df_mp_hist, 
            join_curr_hist         = False, 
            addtnl_mp_df_curr_cols = addtnl_mp_cols, 
            addtnl_mp_df_hist_cols = addtnl_mp_cols, 
            assert_all_PNs_found   = assert_all_PNs_found, 
            assume_one_xfmr_per_PN = True, 
            drop_approx_duplicates = True
        )
        df_mp_curr = mp_df_curr_hist_dict['mp_df_curr']
        df_mp_hist = mp_df_curr_hist_dict['mp_df_hist']
        #-------------------------
        # Only reason for making dict is to ensure outg_rec_nbs are not repeated 
        active_SNs_in_outgs_dfs_dict = {}
    
        if not is_slim:
            for outg_rec_nb_i, df_i in df_outage.groupby(outg_rec_nb_col):
                # Don't want to include outg_rec_nb_i=-2147483648
                if int(outg_rec_nb_i) < 0:
                    continue
                # There should only be a single unique dt_on_ts and dt_off_ts_full for each outage
                if(df_i[dt_on_ts_col].nunique()!=1 or 
                   df_i[df_off_ts_full_col].nunique()!=1):
                    print(f'outg_rec_nb_i = {outg_rec_nb_i}')
                    print(f'df_i[dt_on_ts_col].nunique()       = {df_i[dt_on_ts_col].nunique()}')
                    print(f'df_i[df_off_ts_full_col].nunique() = {df_i[df_off_ts_full_col].nunique()}')
                    print('CRASH IMMINENT!')
                    assert(0)
                # Grab power out/on time and PNs from df_i
                dt_on_ts_i       = df_i[dt_on_ts_col].unique()[0]
                df_off_ts_full_i = df_i[df_off_ts_full_col].unique()[0]
                PNs_i            = df_i[prem_nb_col].unique().tolist()
    
                # Just as was done above for PNs, NaN values must be removed from PNs_i
                #   The main purpose here is to remove instances where PNs_i = [nan]
                #   NOTE: For case of slim df, the NaNs should already be removed
                # After removal, if len(PNs_i)==0, contine
                PNs_i = [x for x in PNs_i if pd.notna(x)]
                if len(PNs_i)==0:
                    continue
                
                # Build active_SNs_df_i and add it to active_SNs_in_outgs_dfs_dict
                # NOTE: assume_one_xfmr_per_PN=True above in MeterPremise.build_mp_df_curr_hist_for_PNs,
                #       so does not need to be set again (i.e., assume_one_xfmr_per_PN=False below)
                active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                    PNs                    = PNs_i,
                    df_mp_curr             = df_mp_curr, 
                    df_mp_hist             = df_mp_hist, 
                    dt_0                   = df_off_ts_full_i,
                    dt_1                   = dt_on_ts_i,
                    assume_one_xfmr_per_PN = False, 
                    output_index           = None,
                    output_groupby         = None, 
                    assert_all_PNs_found   = False
                )
                active_SNs_df_i[outg_rec_nb_col] = outg_rec_nb_i
                assert(outg_rec_nb_i not in active_SNs_in_outgs_dfs_dict)
                active_SNs_in_outgs_dfs_dict[outg_rec_nb_i] = active_SNs_df_i
        else:
            for outg_rec_nb_i, row_i in df_outage.iterrows():
                # NOTE: assume_one_xfmr_per_PN=True above in MeterPremise.build_mp_df_curr_hist_for_PNs,
                #       so does not need to be set again (i.e., assume_one_xfmr_per_PN=False below)
                active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                    PNs                    = row_i[prem_nb_col],
                    df_mp_curr             = df_mp_curr, 
                    df_mp_hist             = df_mp_hist, 
                    dt_0                   = row_i[df_off_ts_full_col],
                    dt_1                   = row_i[dt_on_ts_col],
                    assume_one_xfmr_per_PN = False, 
                    output_index           = None,
                    output_groupby         = None, 
                    assert_all_PNs_found   = False
                )
                active_SNs_df_i[outg_rec_nb_col] = outg_rec_nb_i
                assert(outg_rec_nb_i not in active_SNs_in_outgs_dfs_dict)
                active_SNs_in_outgs_dfs_dict[outg_rec_nb_i] = active_SNs_df_i
        #-------------------------
        active_SNs_df = pd.concat(list(active_SNs_in_outgs_dfs_dict.values()))
        #-------------------------
        if drop_inst_rmvl_cols:
            active_SNs_df = active_SNs_df.drop(columns=[df_mp_install_time_col, df_mp_removal_time_col])
        #-------------------------
        return active_SNs_df
    

    def build_or_load_df_mp_outg(
        self, 
        verbose = True
    ):
        r"""
        This either loads or builds (depending on the value of self.read_dfs_from_file) df_mp_outg.
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            # No real reason to read in df_mp_outg_OG, as it's not used after df_mp_outg is built
            # df_mp_outg_OG = pd.read_pickle(os.path.join(self.save_dir_base, 'df_mp_outg_b4_dupl_rmvl.pkl'))
            self.df_mp_outg = pd.read_pickle(os.path.join(self.save_dir_base, 'df_mp_outg_full.pkl'))
        else:
            df_mp_outg_OG = OutageDAQOutg.build_active_MP_for_outages_df(
                df_outage            = self.df_outage_OG, 
                prem_nb_col          = 'PREMISE_NB', 
                is_slim              = False, 
                assert_all_PNs_found = False
            )
            #-----
            df_mp_outg_OG['inst_ts'] = pd.to_datetime(df_mp_outg_OG['inst_ts'])
            df_mp_outg_OG['rmvl_ts'] = pd.to_datetime(df_mp_outg_OG['rmvl_ts'])
            #-------------------------
            if self.save_dfs_to_file:
                df_mp_outg_OG.to_pickle(os.path.join(self.save_dir_base, 'df_mp_outg_b4_dupl_rmvl.pkl'))
            #-------------------------
            self.df_mp_outg = MeterPremise.drop_approx_mp_duplicates(
                mp_df                 = df_mp_outg_OG, 
                fuzziness             = pd.Timedelta('1 hour'), 
                assert_single_overlap = True, 
                addtnl_groupby_cols   = ['OUTG_REC_NB'], 
                gpby_dropna           = False
            )
            #-------------------------
            if self.save_dfs_to_file:
                self.df_mp_outg.to_pickle(os.path.join(self.save_dir_base, 'df_mp_outg_full.pkl'))


    def build_or_load_df_outage_slim(
        self
    ):
        r"""
        This either loads or builds (depending on the value of self.read_dfs_from_file) df_outage_slim.
        Note: Along with df_outage_slim, df_outage will be built, which is essentially df_outage_OG merged with df_mp_outg
        """
        #----------------------------------------------------------------------------------------------------  
        if self.read_dfs_from_file:
            self.df_outage      = pd.read_pickle(os.path.join(self.save_dir_base, 'df_outage.pkl'))
            self.df_outage_slim = pd.read_pickle(os.path.join(self.save_dir_base, 'df_outage_slim.pkl'))
        else:
            self.df_outage = DOVSOutages.merge_df_outage_with_mp(
                df_outage          = self.df_outage_OG.copy(), 
                df_mp              = self.df_mp_outg.copy(), 
                merge_on_outg      = ['OUTG_REC_NB', 'PREMISE_NB'], 
                merge_on_mp        = ['OUTG_REC_NB', 'prem_nb'], 
                cols_to_include_mp = None, 
                drop_cols          = None, 
                rename_cols        = None, 
                inplace            = True
            )
            #-------------------------
            self.df_outage_slim    = DOVSOutages.consolidate_df_outage(
                df_outage                = self.df_outage, 
                addtnl_grpby_cols        = ['trsf_pole_nb'], 
                set_outg_rec_nb_as_index = False
            )

            #-------------------------
            # To be certain we obtain the range we need, should I expand the search window +- 1 day on each end?
            # The only concern (mainly for baseline) is this would mess up the windows grabbed using, e.g., 
            #  get_bsln_time_interval_infos_df_from_summary_files
            # search_time_window = [
            #     self.td_left  - pd.Timedelta('1Day'), 
            #     self.td_right + pd.Timedelta('1Day')
            # ]
            search_time_window = [self.td_left, self.td_right]            
            #-----
            self.df_outage_slim = DOVSOutages.set_search_time_in_outage_df(
                df_outage                   = self.df_outage_slim, 
                search_time_window          = search_time_window, 
                power_out_col               = 'DT_OFF_TS_FULL',
                power_on_col                = 'DT_ON_TS',
                t_search_min_col            = 't_search_min', 
                t_search_max_col            = 't_search_max', 
                wrt_out_only                = True
            )

            #-------------------------
            if self.save_dfs_to_file:
                self.df_outage.to_pickle(     os.path.join(self.save_dir_base, 'df_outage.pkl'))
                self.df_outage_slim.to_pickle(os.path.join(self.save_dir_base, 'df_outage_slim.pkl'))


    def build_or_load_trsf_pole_zips_info(
        self, 
        verbose=False
    ):
        r"""
        This either loads or builds (depending on the value of self.read_dfs_from_file) mp_for_zips_df, trsf_pole_df_full, trsf_pole_zips_df.
        FURTHERMORE, depending on the value of self.single_zip_xfmrs_only, self.df_outage and self.df_outage_slim may be altered
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading mp_for_zips_df from file:      {os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl')}")
                print(f"Reading trsf_pole_df_full from file:   {os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl')}")
                print(f"Reading trsf_pole_zips_df from file:   {os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl')}")
            #-------------------------
            self.mp_for_zips_df    = pd.read_pickle(os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl'))
            self.trsf_pole_df_full = pd.read_pickle(os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl'))
            self.trsf_pole_zips_df = pd.read_pickle(os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl'))
        else:
            zips_dict = OutageDAQ.build_trsf_pole_zips_df(
                field_to_split_and_val = ('premise_nbs', self.df_mp_outg['prem_nb'].unique().tolist()), 
                states                 = self.states, 
                opcos                  = self.opcos, 
                cities                 = self.cities
            )
            #----------
            self.trsf_pole_zips_df = zips_dict['trsf_pole_zips_df']
            self.trsf_pole_df_full = zips_dict['trsf_pole_df_full']
            self.mp_for_zips_df    = zips_dict['mp_for_zips_df']
            #--------------------------------------------------
            if self.save_dfs_to_file:
                self.mp_for_zips_df.to_pickle(   os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl'))
                self.trsf_pole_df_full.to_pickle(os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl'))
                self.trsf_pole_zips_df.to_pickle(os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl'))
        #----------------------------------------------------------------------------------------------------
        if self.single_zip_xfmrs_only:
            trsf_pole_nzips   = self.trsf_pole_zips_df.drop(columns=['zip+4']).drop_duplicates()['trsf_pole_nb'].value_counts()
            single_zip_poles  = trsf_pole_nzips[trsf_pole_nzips==1].index.tolist()
            #-----
            self.df_outage      = self.df_outage[self.df_outage['trsf_pole_nb'].isin(single_zip_poles)].copy()
            self.df_outage_slim = self.df_outage_slim[self.df_outage_slim['trsf_pole_nb'].isin(single_zip_poles)].copy()


    def collect_events(
        self, 
        batch_size         = None, 
        verbose            = True, 
        n_update           = 1, 
        delete_all_others  = True
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        if delete_all_others:
            try:
                del df_outage
            except:
                pass
            #----------
            try:
                del df_mp_outg
            except:
                pass
            #----------
        #----------------------------------------------------------------------------------------------------
        if batch_size is None:
            if self.collect_evs_sum_vw:
                batch_size = 50 
            else:
                batch_size = 30
        #----------------------------------------------------------------------------------------------------
        # A little confusing below........
        #   AMIEndEvents_SQL.build_sql_end_events_for_outages is the method used to collect the events.
        #   The confusion enters because the aforementioned method accepts build_sql_function and build_sql_function_kwargs arguments.
        #   So, this introduces a sort of nested structure
        #   AMIEndEvents:
        #       build_sql_function        = AMIEndEvents_SQL.build_sql_end_events_for_outages
        #       build_sql_function_kwargs = dict(
        #           cols_of_interest   = ..., 
        #           df_outage          = ..., 
        #           build_sql_function = AMIEndEvents_SQL.build_sql_end_events, 
        #           build_sql_function_kwargs = dict(
        #             opcos       = ..., 
        #             date_range  = ..., 
        #             premise_nbs = ..., 
        #             etc.
        #             
        #           )
        #       )
        #----------------------------------------------------------------------------------------------------
        df_outage_slim = self.df_outage_slim.copy()
        #-----
        # Below, doi = Date Of Interest
        df_outage_slim = df_outage_slim.rename(columns={'DT_OFF_TS_FULL' : 'doi'})
        #--------------------------------------------------
        df_construct_type              = DFConstructType.kRunSqlQuery
        contstruct_df_args_end_events  = None
        addtnl_groupby_cols            = ['OUTG_REC_NB', 'trsf_pole_nb', 'doi']
        #-----
        if self.collect_evs_sum_vw:
            cols_of_interest_end_dev_event = ['*']
            date_only                      = True
            build_sql_function_kwargs = dict(
                schema_name = 'meter_events', 
                table_name  = 'events_summary_vw', 
                opco        = self.opcos
            )
        else:
            cols_of_interest_end_dev_event = TableInfos.AMIEndEvents_TI.std_columns_of_interest
            date_only                      = False
            build_sql_function_kwargs = dict(
                states = self.states, 
                opcos  = self.opcos, 
                cities = self.cities
            )
        #--------------------------------------------------
        end_events_sql_function_kwargs = dict(
            cols_of_interest                  = cols_of_interest_end_dev_event, 
            df_outage                         = df_outage_slim, 
            build_sql_function                = AMIEndEvents_SQL.build_sql_end_events, 
            build_sql_function_kwargs         = build_sql_function_kwargs, 
            join_mp_args                      = False, 
            date_only                         = date_only, 
            output_t_minmax                   = True, 
            df_args                           = dict(
                addtnl_groupby_cols = addtnl_groupby_cols, 
                mapping_to_ami      = DfToSqlMap(df_col='PREMISE_NBS', kwarg='premise_nbs', sql_col='aep_premise_nb'), 
                is_df_consolidated  = True
            ),
            # GenAn - keys_to_pop in GenAn.build_sql_general 
            field_to_split                    = 'df_outage', 
            field_to_split_location_in_kwargs = ['df_outage'], 
            save_and_dump                     = True, 
            sort_coll_to_split                = True,
            batch_size                        = batch_size, 
            verbose                           = verbose, 
            n_update                          = n_update
        )
        #----------------------------------------------------------------------------------------------------
        start=time.time()
        exit_status = Utilities.run_tryexceptwhile_process(
            func                = AMIEndEvents,
            func_args_dict      = dict(
                df_construct_type         = df_construct_type, 
                contstruct_df_args        = contstruct_df_args_end_events, 
                build_sql_function        = AMIEndEvents_SQL.build_sql_end_events_for_outages, 
                build_sql_function_kwargs = end_events_sql_function_kwargs, 
                init_df_in_constructor    = True, 
                save_args                 = self.end_events_save_args
            ), 
            max_calls_per_min   = 1, 
            lookback_period_min = 15, 
            max_calls_absolute  = 1000, 
            verbose             = verbose
        )
        if verbose:
            print(f'exit_status = {exit_status}')
            print(f'Build time  = {time.time()-start}')
        #----------------------------------------------------------------------------------------------------
        if exit_status:
            self.save_summary_dict()
            #-------------------------
            time_infos_df = OutageDAQ.build_baseline_time_infos_df(
                data_dir_base           = self.save_dir_base, 
                min_req                 = True, 
                summary_dict_fname      = 'summary_dict.json', 
                alias                   = 'mapping_table', 
                include_summary_paths   = False, 
                consolidate             = False, 
                PN_regex                = r"prem(?:ise)?_nbs?", 
                t_min_regex             = r"t(?:_search)?_min", 
                t_max_regex             = r"t(?:_search)?_max", 
                drop_gpd_for_sql        = True, 
                return_gpd_cols         = False, 
                verbose                 = verbose
            )
            time_infos_df.to_pickle(os.path.join(self.save_dir_base, 'time_infos_df.pkl'))

    
#------------------------------------------------------------------------------------------------------------------------------------------
class OutageDAQOtBL(OutageDAQ):
    r"""
    Class to hold methods specific to Outage BaseLine (OtBL) data acquisition
    """
    def __init__(
        self, 
        run_date, 
        date_0, 
        date_1, 
        collect_evs_sum_vw,     # boolean
        save_sub_dir, 
        groupby_col             = 'trsf_pole_nb', 
        td_left                 = pd.Timedelta('-31D'), 
        td_right                = pd.Timedelta('-1D'),   
        buffer_time_left        = pd.Timedelta('1 days'), 
        buffer_time_rght        = pd.Timedelta('31 days'), 
        pd_selection_stategy    = 'all', 
        search_window_strategy  = 'all_subwindows', 
        states                  = None, 
        opcos                   = None, 
        cities                  = None, 
        single_zip_xfmrs_only   = False, 
        save_end_events         = False, 
        save_dfs_to_file        = False, 
        read_dfs_from_file      = True, 
        base_dir                = os.path.join(
            Utilities.get_local_data_dir(), 
            r'dovs_and_end_events_data'
        ), 
        dates_subdir_appndx     = None, 
        run_using_slim          = False 
    ):
        r"""
        """
        #--------------------------------------------------
        # Not sure if fully built out for 'PREMISE_NB' since making updates
        # assert(groupby_col in ['trsf_pole_nb', 'PREMISE_NB'])
        assert(groupby_col in ['trsf_pole_nb'])
        self.groupby_col            = groupby_col
        #-------------------------
        if not Utilities.is_timedelta(buffer_time_left):
            buffer_time_left  = pd.to_timedelta(buffer_time_left)
        if not Utilities.is_timedelta(buffer_time_rght):
            buffer_time_rght = pd.to_timedelta(buffer_time_rght)
        #-----
        assert(
            buffer_time_left >= pd.Timedelta(0) and
            buffer_time_rght >= pd.Timedelta(0)
        )
        #-----
        self.buffer_time_left       = buffer_time_left
        self.buffer_time_rght       = buffer_time_rght
        #-------------------------
        assert(pd_selection_stategy   in ['max', 'min', 'rand', 'all'])
        assert(search_window_strategy in ['centered', 'rand', 'all_subwindows'] or isinstance(search_window_strategy, datetime.timedelta))
        #-----
        self.pd_selection_stategy   = pd_selection_stategy
        self.search_window_strategy = search_window_strategy
        #--------------------------------------------------
        self.PNs                    = None
        #-----
        self.mp_df_PNs              = None
        self.mp_df_sql_stmnts       = None
        #-----
        self.mp_df_xfmrs            = None
        #-------------------------
        self.all_outages_df         = None
        self.df_mp                  = None
        self.df_no_outage           = None
        self.df_no_outage_slim      = None
        #--------------------------------------------------
        super().__init__(
            run_date                = run_date, 
            date_0                  = date_0, 
            date_1                  = date_1, 
            collect_evs_sum_vw      = collect_evs_sum_vw, 
            save_sub_dir            = save_sub_dir, 
            td_left                 = td_left, 
            td_right                = td_right, 
            states                  = states, 
            opcos                   = opcos, 
            cities                  = cities, 
            single_zip_xfmrs_only   = single_zip_xfmrs_only, 
            save_end_events         = save_end_events, 
            save_dfs_to_file        = save_dfs_to_file, 
            read_dfs_from_file      = read_dfs_from_file, 
            base_dir                = base_dir, 
            dates_subdir_appndx     = dates_subdir_appndx, 
            run_using_slim          = run_using_slim
        )
        self.min_window_width       = self.window_width


    def get_summary_dict(
        self
    ):
        r"""
        DAQ settings with adjustments to a few entries (so they can be easily output into the summary JSON file)
        e.g., Timestamp is not JSON serializable, hence the need for strftime below
        e.g., Timedelta is not JSON serializable, hence the conversion to total seconds
        """
        #--------------------------------------------------
        summary_dict = super().get_summary_dict()
        #--------------------------------------------------
        summary_dict['dataset']                = 'otbl'
        summary_dict['rec_nb_col']             = 'no_outg_rec_nb'
        #-------------------------
        summary_dict['groupby_col']            = self.groupby_col
        #-------------------------
        summary_dict['buffer_time_left']       = self.buffer_time_left.total_seconds()
        summary_dict['buffer_time_rght']       = self.buffer_time_rght.total_seconds()
        #-------------------------
        summary_dict['pd_selection_stategy']   = self.pd_selection_stategy
        summary_dict['search_window_strategy'] = self.search_window_strategy
        #-------------------------
        summary_dict['min_window_width']       = self.min_window_width.total_seconds()
        #--------------------------------------------------
        return summary_dict
    
    def save_summary_dict(
        self
    ):
        r"""
        """
        #-------------------------
        summary_dict = self.get_summary_dict()
        #-----
        assert(os.path.isdir(self.save_dir_base))
        #-----
        CustomWriter.output_dict_to_json(
            os.path.join(self.save_dir_base, 'summary_dict.json'), 
            summary_dict
        )
    

    def build_or_load_mp_df_xfmrs(
        self, 
        verbose
    ):
        r"""
        This build/sets mp_df_xfmrs and mp_df_PNs 
        -----
        Need to find all PNs, which consist not only of those directly from df_outage, but also those not in df_outage who were connected to transformers having entries in df_outage.
        This function build mp_df_xfmrs, which contains the latter (in reality, I think mp_df_xfmrs should contain all, but the purpose for it being built is the inclusion of the 
          latter described PNs)
        """
        #-------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading mp_df_PNs from file:   {os.path.join(self.save_dir_base, 'mp_df_PNs_no_outg.pkl')}")
                print(f"Reading mp_df_xfmrs from file: {os.path.join(self.save_dir_base, 'mp_df_xfmrs_no_outg.pkl')}")
            #-------------------------
            self.mp_df_PNs   = pd.read_pickle(os.path.join(self.save_dir_base, 'mp_df_PNs_no_outg.pkl'))
            self.mp_df_xfmrs = pd.read_pickle(os.path.join(self.save_dir_base, 'mp_df_xfmrs_no_outg.pkl'))
        else:
            #--------------------------------------------------
            PNs = self.df_outage_OG['PREMISE_NB'].unique().tolist()
            #-------------------------
            start=time.time()
            mp_df_PNs, sql_stmnts = MeterPremise.get_distinct_trsf_pole_nbs_for_PNs(
                PNs        = PNs, 
                batch_size = 10000, 
                conn_aws   = self.conn_aws, 
                return_sql = True, 
                states     = self.states, 
                opcos      = self.opcos, 
                cities     = self.cities
            )
            if verbose:
                print(f"mp_df_PNs: {time.time()-start}")
            #-------------------------
            start=time.time()
            mp_xfmrs = GenAn(
                df_construct_type         = DFConstructType.kRunSqlQuery, 
                contstruct_df_args        = dict(conn_db=self.conn_aws), 
                init_df_in_constructor    = True, 
                build_sql_function        = MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs = dict(
                    cols_of_interest = ['trsf_pole_nb', 'prem_nb', 'mfr_devc_ser_nbr', 'inst_ts', 'rmvl_ts', 'state_cd'], 
                    trsf_pole_nb     = [x for x in mp_df_PNs['trsf_pole_nb'].unique() if x!='TRANSMISSION'], 
                    field_to_split   = 'trsf_pole_nb', 
                    states           = self.states, 
                    opcos            = self.opcos, 
                    cities           = self.cities
                )
            )
            if verbose:
                print(f"mp_df_xfmrs: {time.time()-start}")
            mp_df_xfmrs = mp_xfmrs.df
            #--------------------------------------------------
            self.mp_df_PNs        = mp_df_PNs
            self.mp_df_sql_stmnts = sql_stmnts
            self.mp_df_xfmrs      = mp_df_xfmrs
            #-------------------------
            self.PNs = list(set(self.df_outage_OG['PREMISE_NB'].unique().tolist() + self.mp_df_xfmrs['prem_nb'].unique().tolist()))
            #--------------------------------------------------
            if self.save_dfs_to_file:
                self.mp_df_PNs.to_pickle(  os.path.join(self.save_dir_base, 'mp_df_PNs_no_outg.pkl'))
                self.mp_df_xfmrs.to_pickle(os.path.join(self.save_dir_base, 'mp_df_xfmrs_no_outg.pkl'))


    def build_or_load_trsf_pole_zips_info(
        self, 
        verbose=False
    ):
        r"""
        This either loads or builds (depending on the value of self.read_dfs_from_file) mp_for_zips_df, trsf_pole_df_full, trsf_pole_zips_df.
        FURTHERMORE, depending on the value of self.single_zip_xfmrs_only, self.PNs may be altered
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading mp_for_zips_df from file:      {os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl')}")
                print(f"Reading trsf_pole_df_full from file:   {os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl')}")
                print(f"Reading trsf_pole_zips_df from file:   {os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl')}")
            #-------------------------
            self.mp_for_zips_df    = pd.read_pickle(os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl'))
            self.trsf_pole_df_full = pd.read_pickle(os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl'))
            self.trsf_pole_zips_df = pd.read_pickle(os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl'))
        else:
            zips_dict = OutageDAQ.build_trsf_pole_zips_df(
                field_to_split_and_val = ('premise_nbs', self.PNs), 
                states                 = self.states, 
                opcos                  = self.opcos, 
                cities                 = self.cities
            )
            #----------
            self.trsf_pole_zips_df = zips_dict['trsf_pole_zips_df']
            self.trsf_pole_df_full = zips_dict['trsf_pole_df_full']
            self.mp_for_zips_df    = zips_dict['mp_for_zips_df']
            #--------------------------------------------------
            if self.save_dfs_to_file:
                self.mp_for_zips_df.to_pickle(   os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl'))
                self.trsf_pole_df_full.to_pickle(os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl'))
                self.trsf_pole_zips_df.to_pickle(os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl'))
        #----------------------------------------------------------------------------------------------------
        if self.single_zip_xfmrs_only:
            trsf_pole_nzips   = self.trsf_pole_zips_df.drop(columns=['zip+4']).drop_duplicates()['trsf_pole_nb'].value_counts()
            single_zip_poles  = trsf_pole_nzips[trsf_pole_nzips==1].index.tolist()
            #-----
            self.PNs = self.mp_for_zips_df[self.mp_for_zips_df['trsf_pole_nb'].isin(single_zip_poles)]['prem_nb'].unique().tolist()
    

    @staticmethod
    def find_all_outages_for_pns(
        PNs, 
        date_0, 
        date_1, 
        cols_of_interest                   = None, 
        mjr_mnr_cause                      = None, 
        method                             = 'decide_at_runtime', 
        addtnl_build_sql_std_outage_kwargs = None, 
        conn_outages                       = None, 
        verbose                            = True, 
        n_update                           = 10, 
        batch_size                         = 1000
    ):
        r"""
        By default, the returned columns are [DT_ON_TS, DT_OFF_TS_FULL, PREMISE_NB].
            The first two are explicitly added below under 'if cols_of_interest is None:'
            The last is more subtly added via the 'select_cols_DOVS_PREMISE_DIM=['PREMISE_NB']' input parameter
              to DOVSOutages_SQL.build_sql_std_outage
          
        method:
            Possible values: 'query_pns_only', 'query_all', 'decide_at_runtime'
            'query_pns_only':  Build the SQL queries using the premise numbers in PNs.
                               NOTE: With this method, likely the premise numbers will need to be split into multiple queries, 
                                     hence the need for full-blown DOVSOutages/GenAn.build_df_general as opposed to use of 
                                     DOVSOutages.build_sql_std_outage in 'query_all' method
                               Pro: Less memory, as only PNs we're interested in are grabbed in SQL queries
                               Con: Takes significantly more time to run when number of PNs is large.
                               NOTE: If len(PNs) > 100000, then method 'query_all' will typically be faster.
                                     Take this with a grain on salt, as the number 100000 was found for just one particular
                                     collection of PNs for specific date_0 and date_1, so others may differ.
    
            'query_all':       Build the SQL query using only date_0 and date_1 (together with mjr_mnr_cause, etc.), 
                                 i.e., data for ALL premise numbers are grabbed
                               After the SQL query returns, slim the data down to include only the PNs of interest.
                               Pro: Takes significantly less time to run when number of PNs is large
                               Con: Consumes more memory, as we're grabbing everything then slimming down
                               
            'decide_at_runtime': Decide between methods 'query_pns_only' and 'query_all' at runtime.
                                 If len(PNs) > 100000, use 'query_all', else use 'query_pns_all'
                                 As mentioned above, the number 100000 was found for just one particular
                                 collection of PNs for specific date_0 and date_1, so others may differ.
        
        NOTE: This uses DOVSOutages_SQL.build_sql_std_outage, so the standard DOVS cuts (listed below) are included:
                DOV.MJR_CAUSE_CD <> 'NI'
                DOV.DEVICE_CD <> 85
                DOV2.INTRPTN_TYP_CD = 'S'
                DOV2.CURR_REC_STAT_CD = 'A'
        
        """
        #-------------------------
        if conn_outages is None:
            conn_outages = Utilities.get_utldb01p_oracle_connection()
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = [
                'DT_ON_TS', 
                {'field_desc': 'DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)',
                 'alias': 'DT_OFF_TS_FULL',
                 'table_alias_prefix': None}
            ]
        #-------------------------
        # Make sure only unique values in PNs
        PNs = list(set(PNs))
        #-------------------------
        assert(method in ['query_pns_only', 'query_all', 'decide_at_runtime'])
        if method=='decide_at_runtime':
            if len(PNs) > 100000:
                method='query_all'
            else:
                method='query_pns_only'
        #-------------------------
        if method=='query_pns_only':
            build_sql_std_outage_kwargs = dict(
                mjr_mnr_cause                       = mjr_mnr_cause, 
                include_premise                     = True, 
                cols_of_interest                    = cols_of_interest, 
                select_cols_DOVS_PREMISE_DIM        = ['PREMISE_NB'], 
                alias_DOVS_PREMISE_DIM              = 'PRIM', 
                date_range                          = [date_0, date_1], 
                premise_nbs                         = PNs, 
                include_DOVS_MASTER_GEO_DIM         = False, 
                include_DOVS_OUTAGE_ATTRIBUTES_DIM  = False, 
                include_DOVS_CLEARING_DEVICE_DIM    = False, 
                include_DOVS_EQUIPMENT_TYPES_DIM    = False, 
                include_DOVS_OUTAGE_CAUSE_TYPES_DIM = False, 
                field_to_split                      = 'premise_nbs', 
                batch_size                          = batch_size, 
                n_update                            = n_update, 
                verbose                             = verbose
            )
            if addtnl_build_sql_std_outage_kwargs is not None:
                build_sql_std_outage_kwargs = Utilities.supplement_dict_with_default_values(
                    to_supplmnt_dict    = build_sql_std_outage_kwargs, 
                    default_values_dict = addtnl_build_sql_std_outage_kwargs, 
                    extend_any_lists    = True, 
                    inplace             = True
                )
            return_df = GenAn.build_df_general(
                conn_db                   = conn_outages, 
                build_sql_function        = DOVSOutages_SQL.build_sql_std_outage, 
                build_sql_function_kwargs = build_sql_std_outage_kwargs
            )
        elif method=='query_all':
            build_sql_std_outage_kwargs = dict(
                mjr_mnr_cause                       = mjr_mnr_cause, 
                include_premise                     = True, 
                cols_of_interest                    = cols_of_interest, 
                select_cols_DOVS_PREMISE_DIM        = ['PREMISE_NB'], 
                alias_DOVS_PREMISE_DIM              = 'PRIM', 
                date_range                          = [date_0, date_1], 
                include_DOVS_MASTER_GEO_DIM         = False, 
                include_DOVS_OUTAGE_ATTRIBUTES_DIM  = False, 
                include_DOVS_CLEARING_DEVICE_DIM    = False, 
                include_DOVS_EQUIPMENT_TYPES_DIM    = False, 
                include_DOVS_OUTAGE_CAUSE_TYPES_DIM = False
            )
            if addtnl_build_sql_std_outage_kwargs is not None:
                build_sql_std_outage_kwargs = Utilities.supplement_dict_with_default_values(
                    to_supplmnt_dict    = build_sql_std_outage_kwargs, 
                    default_values_dict = addtnl_build_sql_std_outage_kwargs, 
                    extend_any_lists    = True, 
                    inplace             = True
                )
            sql_outages_for_PNs = DOVSOutages_SQL.build_sql_std_outage(**build_sql_std_outage_kwargs)
            sql_outages_for_PNs = sql_outages_for_PNs.get_sql_statement()
            return_df           = pd.read_sql_query(sql_outages_for_PNs, conn_outages)
            return_df           = return_df[return_df['PREMISE_NB'].isin(PNs)]
        else:
            assert(0)
        #-------------------------
        return return_df
    
    
    @staticmethod
    def find_clean_window_for_group(
        df_i                       , 
        min_window_width           , 
        buffer_time_left           , 
        buffer_time_rght           ,
        date_1                     ,  
        set_search_window          = True, 
        pd_selection_stategy       = 'max', 
        search_window_strategy     = 'centered', 
        needs_sorted               = True, 
        outg_beg_col               = 'DT_OFF_TS_FULL', 
        outg_end_col               = 'DT_ON_TS', 
        record_clean_window_usable = True
    ):
        r"""
        INTENDED FOR USE IN .groupby().apply(lambda x) function.
        This can still be used on its own, but the user should be aware of the functionality and intent
        
        FASTEST RUN TIME SUGGESTIONS:
            - Sort the DF prior, and set needs_sorted=False
            - Use search_window_strategy = 'centered'
            
        NOTE: If search_window_strategy is a timedelta object, the search period will begin search_window_strategy
                after the buffer_time_left (not after the end of the previous outage)
        
        
        needs_sorted:
            IF YOU ARE NOT SURE, KEEP needs_sorted=True, as the proper sorting of the DF is vital for the functionality.
            When running this within a .groupby().apply(lambda x) function, a little bit of time can be saved by 
              sorting the overall DataFrame first before the groupby call.
            Regardless of needs_sorted, sorting first will save time.
            If sorting already done, no need to re-sort here, so a little more time can be saved by setting needs_sorted=False

        date_1:
            To find the amount of clean time after the last outage, use date_1 as an endpoint
        """
        #-------------------------
        assert(pd_selection_stategy   in ['max', 'min', 'rand', 'all'])
        assert(search_window_strategy in ['centered', 'rand', 'all_subwindows'] or isinstance(search_window_strategy, datetime.timedelta))
        #-------------------------
        # For this to function properly, df_i must be sorted according to time
        if needs_sorted:
            df_i = df_i.sort_values(by=[outg_beg_col, outg_end_col], ascending=True).copy()
    
        #-------------------------
        # Find the clean periods of time following each outage by subtracting the beginning time
        # of the next outage from the end of the current outage.
        clean_windows_after = df_i[outg_beg_col].shift(-1)-df_i[outg_end_col]
    
        # To find the amount of clean time after the last outage, use date_1 as an endpoint
        #   i.e., subtract the end time of the current outage from the end of the overall interval, date_1
        clean_windows_after.iloc[-1] = pd.to_datetime(date_1) - df_i.iloc[-1][outg_end_col]
    
        #-------------------------
        # Find the acceptable periods for which the clean time is greater than the desired length
        # NOTE: The buffer_time_left/_rght arguments allow one to ensure the period of time is not 
        #       immediately proceeding or preceding an outage event
        # NOTE: good_clean_windows_after must have a name in order to merge with df_i
        good_clean_windows_after      = clean_windows_after[clean_windows_after > min_window_width+buffer_time_left+buffer_time_rght]
        good_clean_windows_after.name = 'clean_window_full'
        if len(good_clean_windows_after)==0:
            return pd.DataFrame()
    
        #-------------------------
        # Construct good_df_i using the entries from good_clean_windows_after
        # Merge this with good_clean_windows_after to include clean_window information
        good_df_i = df_i.loc[good_clean_windows_after.index]
        good_df_i = pd.merge(good_df_i, good_clean_windows_after, left_index=True, right_index=True, how='inner')
        if record_clean_window_usable:
            good_df_i['clean_window_usable'] = good_df_i['clean_window_full'] - (buffer_time_left+buffer_time_rght)
        #-------------------------
        # Select subset of good_df_i according to pd_selection_stategy
        if pd_selection_stategy   == 'max':
            final_df_i = good_df_i.iloc[[good_df_i['clean_window_full'].argmax()]].copy()
        elif pd_selection_stategy == 'min':
            final_df_i = good_df_i.iloc[[good_df_i['clean_window_full'].argmin()]].copy()
        elif pd_selection_stategy == 'rand':
            final_df_i = good_df_i.sample().copy()
        elif pd_selection_stategy == 'all':
            final_df_i = good_df_i.copy()
        else:
            assert(0)
    
        #-------------------------
        # Create columns to hold the min and max clean times
        #   The clean time begins (min) buffer_time_left after the outage ends
        #   The clean time ends (max) buffer_time_rght before the next outage
        #     (which is equal to the time the current outage ends, plus the clean window, 
        #      minus the buffer_time_rght)
        final_df_i['t_clean_min'] = final_df_i[outg_end_col] + buffer_time_left
        final_df_i['t_clean_max'] = final_df_i[outg_end_col] + final_df_i['clean_window_full'] - buffer_time_rght
    
        #-------------------------
        if set_search_window:
            if search_window_strategy == 'centered':
                # Mid point of clean time interval = final_df_i[['t_clean_min', 't_clean_max']].mean(numeric_only=False, axis=1)
                # ==> Left  point = (final_df_i[['t_clean_min', 't_clean_max']].mean(numeric_only=False, axis=1)) - min_window_width/2
                # ==> Right point = (final_df_i[['t_clean_min', 't_clean_max']].mean(numeric_only=False, axis=1)) + min_window_width/2
                final_df_i['t_search_min'] = (final_df_i[['t_clean_min', 't_clean_max']].mean(numeric_only=False, axis=1)) - min_window_width/2
                final_df_i['t_search_max'] = (final_df_i[['t_clean_min', 't_clean_max']].mean(numeric_only=False, axis=1)) + min_window_width/2
            elif search_window_strategy == 'rand':
                final_df_i['t_search_min'] = pd.NaT
                final_df_i['t_search_max'] = pd.NaT
                #-----
                for idx, row_i in final_df_i.iterrows():
                    rnd_intrvl_i = Utilities_dt.get_random_datetime_interval_between(
                        date_0       = row_i['t_clean_min'], 
                        date_1       = row_i['t_clean_max'], 
                        window_width = min_window_width, 
                        rand_seed    = None        
                    )
                    final_df_i.loc[idx, ['t_search_min', 't_search_max']] = rnd_intrvl_i
            elif search_window_strategy == 'all_subwindows':
                final_df_i = OutageDAQ.find_clean_subwindows_for_group(
                    final_df_i                      = final_df_i, 
                    min_window_width                = min_window_width, 
                    include_is_first_after_outg_col = True, 
                    t_clean_min_col                 = 't_clean_min', 
                    t_clean_max_col                 = 't_clean_max', 
                    return_t_search_min_col         = 't_search_min', 
                    return_t_search_max_col         = 't_search_max'
                )
            elif isinstance(search_window_strategy, datetime.timedelta):
                final_df_i['t_search_min'] = final_df_i['t_clean_min'] + search_window_strategy
                final_df_i['t_search_max'] = final_df_i['t_clean_min'] + search_window_strategy + min_window_width
            else:
                assert(0)
        #-------------------------
        # Don't need outg_beg_col or outg_end_col anymore.
        # These columns contain information about the outage(s) after which the clean period(s) was(were) selected.
        # If pd_selection_stategy=='all', I suppose this information would make sense.  But, in any other case, 
        #   the information isn't really useful, as only one clean period is returned following 
        #   one of the (possibly randomly) selected outages
        final_df_i = final_df_i.drop(columns=[outg_beg_col, outg_end_col])
        #-------------------------
        return final_df_i
    
    @staticmethod
    def find_clean_windows(
        df, 
        groupby_col, 
        min_window_width, 
        buffer_time_left, 
        buffer_time_rght,  
        date_1, 
        set_search_window          = True, 
        pd_selection_stategy       = 'max', 
        search_window_strategy     = 'centered', 
        outg_beg_col               = 'DT_OFF_TS_FULL', 
        outg_end_col               = 'DT_ON_TS', 
        record_clean_window_usable = True
    ):
        r"""
        FASTEST RUN TIME SUGGESTIONS:
            - Use search_window_strategy = 'centered' 
        """
        #-------------------------
        # Only really need the three columns [groupby_col, outg_beg_col, outg_end_col]
        df = df[[groupby_col, outg_beg_col, outg_end_col]].copy()
        #-------------------------
        # Drop any duplicates
        df = df.drop_duplicates()
        #-------------------------
        # Make sure outg_beg_col/outg_end_col are datetime
        if not is_datetime64_dtype(df[outg_beg_col]):
            df = Utilities_df.convert_col_type(df=df, column=outg_beg_col, to_type=datetime.datetime)
        if not is_datetime64_dtype(df[outg_end_col]):
            df = Utilities_df.convert_col_type(df=df, column=outg_end_col, to_type=datetime.datetime)
        #-------------------------
        # To speed things up, first sort df
        df = df.sort_values(by=[groupby_col, outg_beg_col, outg_end_col]).copy()
        #-----
        return_df = df.groupby(groupby_col, as_index=False, group_keys=False).apply(
            lambda x: OutageDAQOtBL.find_clean_window_for_group(
                df_i = x, 
                min_window_width           = min_window_width, 
                buffer_time_left           = buffer_time_left, 
                buffer_time_rght           = buffer_time_rght, 
                date_1                     = date_1, 
                set_search_window          = set_search_window, 
                pd_selection_stategy       = pd_selection_stategy, 
                search_window_strategy     = search_window_strategy, 
                outg_beg_col               = outg_beg_col, 
                outg_end_col               = outg_end_col, 
                record_clean_window_usable = record_clean_window_usable, 
                needs_sorted               = False
            )
        )
        #-------------------------
        return return_df
    
    def build_or_load_all_outages_df(
        self, 
        verbose               = True, 
        n_update              = 10, 
        batch_size            = 1000, 
        method                = 'decide_at_runtime'
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_mp from file:                             {os.path.join(self.save_dir_base, 'df_mp_no_outg.pkl')}")
                print(f"Reading all_outages_df from file:                    {os.path.join(self.save_dir_base, 'all_outages_df.pkl')}")
                print(f"Reading all_outages_df_for_non_active_pns from file: {os.path.join(self.save_dir_base, 'all_outages_df_for_non_active_pns.pkl')}")
            #-------------------------
            self.df_mp                             = pd.read_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg.pkl'))
            self.all_outages_df                    = pd.read_pickle(os.path.join(self.save_dir_base, 'all_outages_df.pkl'))
            self.all_outages_df_for_non_active_pns = pd.read_pickle(os.path.join(self.save_dir_base, 'all_outages_df_for_non_active_pns.pkl'))
            return
        #----------------------------------------------------------------------------------------------------
        addtnl_build_sql_std_outage_kwargs=dict(
            states = self.states, 
            opcos  = self.opcos, 
            cities = self.cities
        )
        #-------------------------
        start = time.time()
        all_outages_df                    = None
        df_mp                             = None
        all_outages_df_for_non_active_pns = None
        #-----
        all_outages_df = OutageDAQOtBL.find_all_outages_for_pns(
            PNs                                = self.PNs, 
            date_0                             = self.date_0, 
            date_1                             = self.date_1, 
            cols_of_interest                   = None, 
            mjr_mnr_cause                      = None, 
            method                             = method, 
            conn_outages                       = self.conn_outages, 
            addtnl_build_sql_std_outage_kwargs = addtnl_build_sql_std_outage_kwargs, 
            verbose                            = verbose, 
            n_update                           = n_update, 
            batch_size                         = batch_size
        )
        if verbose:
            print(f"Time to run OutageDAQOtBL.find_all_outages_for_pns: {time.time()-start}")
            print(f"# Unique PNs in df_outage:      {self.df_outage_OG['PREMISE_NB'].nunique()}")
            print(f"# Unique PNs in all_outages_df: {all_outages_df['PREMISE_NB'].nunique()}")
            print(time.time()-start)
    
        #---------------------------------------------------------------------------
        # If grouping by transformer, the trsf_pole_nb from MeterPremise must be merged with all_outages_df
        # Also, the active meters at the time of outage must be selected by comparing inst_ts,rmvl_ts to 
        #   DT_OFF_TS_FULL,DT_ON_TS.
        # This is documented in the code below
        #---------------------------------------------------------------------------
        if self.groupby_col=='trsf_pole_nb':
            df_mp = MeterPremise.build_mp_df_curr_hist_for_PNs(
                PNs                    =  self.PNs, 
                mp_df_curr             = None,
                mp_df_hist             = None, 
                join_curr_hist         = True, 
                addtnl_mp_df_curr_cols = None, 
                addtnl_mp_df_hist_cols = None, 
                assert_all_PNs_found   = False, 
                assume_one_xfmr_per_PN = True, 
                drop_approx_duplicates = True
            )
            #--------------------------------------------------
            # Some premise numbers from DOVS are missing from df_mp.
            # This is not an issue with the code, I checked. 
            # This means DOVS says a premise was affected by an outage, but at the time of the outage there were 
            #   no active meters on the premise.
            # My question is: How did DOVS therefore know the premise was affected?
            # How are premise numbers in DOVS determined?
            #-------------------------
            # I want to at least get a count to quantify the situation described above, i.e., how many premises from DOVS
            #   did not have any active meters at the time of the outage.
            # Note, for this, I cannot simply do, e.g., 
            #     set(all_outages_df['PREMISE_NB'].unique()).difference(set(df_mp['prem_nb'].unique()))
            #   as this might reflect a smaller number of missing PNs than in reality, as df_mp has not yet been chopped down
            #   to only those present at time of outage (which is done below comparing 'inst_ts' to 'DT_OFF_TS_FULL' and 
            #   'rmvl_ts' to 'DT_ON_TS')
            #-------------------------
            # The meters present at the time of the outages can only be select after all_outages_df and df_mp are merged.
            #-------------------------
            # Note: A left merge is used below instead of an inner to protect against the case of a df_mp (being read in from a CSV 
            #       file) which contains extra entries than in all_outages_df
            #-------------------------
            all_outages_df = DOVSOutages.merge_df_outage_with_mp(
                df_outage          = all_outages_df, 
                df_mp              = df_mp,  
                merge_on_outg      = ['PREMISE_NB'], 
                merge_on_mp        = ['prem_nb'], 
                cols_to_include_mp = None, 
                drop_cols          = None, 
                rename_cols        = None, 
                how                = 'left', 
                inplace            = True
            )
        
            #-------------------------
            # Only include serial numbers which were present at the time of the outage.
            #-----
            # NOTE the use of .fillna(pd.Timestamp.min) (YES, MIN) below, as this is different from MeterPremise.get_active_SNs_for_PNs_at_datetime_interval
            #   and MeterPremise.merge_df_with_active_mp.
            # This is needed so the premises missing from df_mp are not removed at this stage (yes, they will ultimately be removed, 
            #   but I don't want them removed yet because I want to track them!)
            # Without this, any entry with 'inst_ts'=NaT would be removed, as a comparison of NaT to anything returns False
            #-----
            all_outages_df = Utilities_df.convert_col_types(
                df                  = all_outages_df, 
                cols_and_types_dict = {
                    'inst_ts' : datetime.datetime, 
                    'rmvl_ts' : datetime.datetime
                }, 
                to_numeric_errors   = 'coerce', 
                inplace             = True
            )
            #-----
            all_outages_df = all_outages_df[
                (all_outages_df['inst_ts'].fillna(pd.Timestamp.min) <= all_outages_df['DT_OFF_TS_FULL']) & 
                (all_outages_df['rmvl_ts'].fillna(pd.Timestamp.max) >  all_outages_df['DT_ON_TS'])
            ]
        
            #-------------------------
            # Find the entries with missing df_mp data, i.e., find the entries where DOVS says a premise was affected by an outage, 
            #   but at the time of the outage there were no active meters on the premise.
            all_outages_df_for_non_active_pns = all_outages_df[all_outages_df[df_mp.columns].isna().all(axis=1)].copy()
            non_active_pns_from_DOVS          = all_outages_df_for_non_active_pns['PREMISE_NB'].unique().tolist()
        
            # And remove the entries with missing df_mp data from all_outages_df
            all_outages_df = all_outages_df.dropna(subset=df_mp.columns, how='all')
            #-------------------------
            if verbose:
                print("""
                Some premise numbers from DOVS are missing from df_mp
                This is not an issue with the code, I checked.  
                This means DOVS says a premise was affected by an outage, but at the time of the outage there were no active meters on the premise.
                My question is: How did DOVS therefore know the premise was affected?
                How are premise numbers in DOVS determined?
                """)
                print(f"Number of premise numbers from DOVS without an active meter at outage time: {len(non_active_pns_from_DOVS)}") 
            #-------------------------
            # At this point, any trsf_pole_nbs to be excluded can be removed
            # Remove 'TRANSMISSION', 'PRIMARY', and 'NETWORK' transformers
            all_outages_df = all_outages_df[~all_outages_df['trsf_pole_nb'].isin(['TRANSMISSION', 'NETWORK', 'PRIMARY'])] 
        #---------------------------------------------------------------------------
        else:
            print("At this point, self.groupby_col must equal 'trsf_pole_nb'")
            assert(0)
            # all_outages_df = something
            # df_mp          = something
            # all_outages_df_for_non_active_pns = something
        #---------------------------------------------------------------------------
        self.df_mp                             = df_mp
        self.all_outages_df                    = all_outages_df
        self.all_outages_df_for_non_active_pns = all_outages_df_for_non_active_pns
        #--------------------------------------------------
        if self.save_dfs_to_file:
            self.all_outages_df.to_pickle(                   os.path.join(self.save_dir_base, 'all_outages_df.pkl'))
            self.df_mp.to_pickle(                            os.path.join(self.save_dir_base, 'df_mp_no_outg.pkl'))
            self.all_outages_df_for_non_active_pns.to_pickle(os.path.join(self.save_dir_base, 'all_outages_df_for_non_active_pns.pkl'))


    def build_or_load_df_no_outage(
        self, 
        verbose=True
    ):
        r"""
        Build/loads self.df_no_outage
        If building and saving, df_no_outage_FINAL.pkl will be output
        If building and saving, clean_windows_by_grp.pkl and clean_windows_by_grp_mrg_mp.pkl will also be output
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_no_outage from file: {os.path.join(self.save_dir_base, 'df_no_outage_FINAL.pkl')}")
            #-------------------------
            self.df_no_outage = pd.read_pickle(os.path.join(self.save_dir_base, 'df_no_outage_FINAL.pkl'))
            return
        #----------------------------------------------------------------------------------------------------
        # Find the clean windows for each group and build df_no_outage
        #-------------------------
        start = time.time()
        clean_windows_by_grp = OutageDAQOtBL.find_clean_windows(
            df                     = self.all_outages_df, 
            groupby_col            = self.groupby_col, 
            min_window_width       = self.min_window_width, 
            buffer_time_left       = self.buffer_time_left, 
            buffer_time_rght       = self.buffer_time_rght, 
            date_1                 = self.date_1, 
            set_search_window      = True, 
            pd_selection_stategy   = self.pd_selection_stategy, 
            search_window_strategy = self.search_window_strategy, 
            outg_beg_col           = 'DT_OFF_TS_FULL', 
            outg_end_col           = 'DT_ON_TS'
        )
        if verbose:
            print(f"find_clean_windows run time in OutageDAQOtBL.build_or_load_df_no_outage: {time.time()-start}")
        #-------------------------
        # All groups (trsf_pole_nbs, PREMISE_NBs, etc.) in clean_windows_by_grp should also be found in self.all_outages_df, 
        #   but the reverse is not true
        assert(len(set(clean_windows_by_grp[self.groupby_col].unique()).difference(set(self.all_outages_df[self.groupby_col].unique())))==0)
        
        # Groups where no clean time period was found
        grps_with_no_clean = self.all_outages_df[~self.all_outages_df[self.groupby_col].isin(clean_windows_by_grp[self.groupby_col].unique())]
        
        print(f"groupby_col = {self.groupby_col}")
        print(f"a. # Groups:                      {self.all_outages_df[self.groupby_col].nunique()}")
        print(f"b. # Groups with clean period:    {clean_windows_by_grp[self.groupby_col].nunique()}")
        print(f"c. # Groups without clean period: {len(set(self.all_outages_df[self.groupby_col].unique()).difference(set(clean_windows_by_grp[self.groupby_col].unique())))}")
        print("NOTE: There may be a difference of 1 between a and b+c due to fact that nunique() does not including NaNs but unique does")
        #----------------------------------------------------------------------------------------------------
        # Merge clean_windows_by_grp with df_mp
        #-------------------------
        clean_windows_by_grp_mrg_mp = pd.merge(
            clean_windows_by_grp, 
            self.df_mp, 
            left_on  = 'trsf_pole_nb', 
            right_on = 'trsf_pole_nb', 
            how      = 'left'
        )
        clean_windows_by_grp_mrg_mp = Utilities_df.convert_col_types(
            df                  = clean_windows_by_grp_mrg_mp, 
            cols_and_types_dict = {
                'inst_ts' : datetime.datetime, 
                'rmvl_ts' : datetime.datetime
            }, 
            to_numeric_errors   = 'coerce', 
            inplace             = True
        )
        clean_windows_by_grp_mrg_mp = clean_windows_by_grp_mrg_mp[
            (clean_windows_by_grp_mrg_mp['inst_ts'].fillna(pd.Timestamp.min)<=clean_windows_by_grp_mrg_mp['t_search_min']) & 
            (clean_windows_by_grp_mrg_mp['rmvl_ts'].fillna(pd.Timestamp.max)>clean_windows_by_grp_mrg_mp['t_search_max'])
        ]
        #--------------------------------------------------
        df_no_outage = clean_windows_by_grp_mrg_mp.copy()
        df_no_outage = df_no_outage.sort_values(by=[self.groupby_col, 'prem_nb', 't_search_min'], ignore_index=True)
        #--------------------------------------------------
        # Add no_outg_rec_nb column to allow easier grouping when building rcpo_dfs
        rand_pfx                       = Utilities.generate_random_string(str_len=5, letters=string.ascii_letters + string.digits)
        df_no_outage['no_outg_rec_nb'] = df_no_outage.groupby(['trsf_pole_nb', 't_search_min', 't_search_max']).ngroup()
        df_no_outage['no_outg_rec_nb'] = rand_pfx + df_no_outage['no_outg_rec_nb'].astype(str)
        #--------------------------------------------------
        # Set the date of interest (doi) column, which is akin to the DT_OFF_TS_FULL column for the outg case
        #   The doi is the reference point from which the various time groupings will be formed.
        # Reconstruction of prediction_date/doi
        #   range_left  = prediction_date + td_left  ==> prediction_date = range_left - td_left
        #   range_right = prediction_date + td_right ==> prediction_date = range_right - td_right
        # As shown above, reconstruction using td_left or td_right should yield the same result.
        # IF FOR SOME REASON THEY YIELD DIFFERENT RESULTS, BOTH WILL BE KEPT (columns 'doi', from left, and 'doi_r' from right)
        df_no_outage['doi']   = df_no_outage['t_search_min'] - self.td_left
        df_no_outage['doi_r'] = df_no_outage['t_search_max'] - self.td_right
        if df_no_outage['doi'].equals(df_no_outage['doi_r']):
            df_no_outage = df_no_outage.drop(columns=['doi_r'])
        else:
            if verbose:
                print("!!!!! WARNING: OutageDAQOtBL.build_or_load_df_no_outage !!!!!\n\tdoi and doi_r columns disagree, so both will be kept!")
        #----------------------------------------------------------------------------------------------------
        self.df_no_outage = df_no_outage
        #----------------------------------------------------------------------------------------------------
        if self.save_dfs_to_file:
            clean_windows_by_grp.to_pickle(       os.path.join(self.save_dir_base, 'clean_windows_by_grp.pkl'))
            clean_windows_by_grp_mrg_mp.to_pickle(os.path.join(self.save_dir_base, 'clean_windows_by_grp_mrg_mp.pkl'))
            self.df_no_outage.to_pickle(          os.path.join(self.save_dir_base, 'df_no_outage_FINAL.pkl'))


    def build_or_load_df_no_outage_slim(
        self, 
        verbose=True
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_no_outage_slim from file: {os.path.join(self.save_dir_base, 'df_no_outage_slim.pkl')}")
            #-------------------------
            self.df_no_outage_slim = pd.read_pickle(os.path.join(self.save_dir_base, 'df_no_outage_slim.pkl'))
            return
        #----------------------------------------------------------------------------------------------------
        #--------------------------------------------------
        # Convert to slim 
        cols_shared_by_group     = None
        cols_to_collect_in_lists = ['prem_nb']
        rename_cols              = {'prem_nb':'premise_nbs'}
        if self.groupby_col=='trsf_pole_nb':
            cols_to_collect_in_lists.append('mfr_devc_ser_nbr')
            rename_cols['mfr_devc_ser_nbr'] = 'serial_numbers'
        #-------------------------
        if self.groupby_col=='trsf_pole_nb':
            consol_groupby_cols = ['no_outg_rec_nb', self.groupby_col, 't_search_min', 't_search_max', 'doi']
        elif self.groupby_col=='PREMISE_NB':
            consol_groupby_cols =[ 'no_outg_rec_nb', 't_search_min', 't_search_max', 'doi']
        else:
            assert(0)
        #-------------------------
        if self.search_window_strategy=='all_subwindows':
            consol_groupby_cols.append('is_first_after_outg')
        #-------------------------
        df_no_outage_slim = Utilities_df.consolidate_df(
            df                       = self.df_no_outage, 
            groupby_cols             = consol_groupby_cols, 
            cols_shared_by_group     = cols_shared_by_group, 
            cols_to_collect_in_lists = cols_to_collect_in_lists, 
            rename_cols              = rename_cols, 
            verbose                  = True
        )
        #-------------------------
        df_no_outage_slim=df_no_outage_slim.reset_index()
        if self.groupby_col=='trsf_pole_nb':
            df_no_outage_slim = df_no_outage_slim.set_index(['no_outg_rec_nb', 'trsf_pole_nb'], drop=False)
            df_no_outage_slim.index.names=['idx_no_outg_rec_nb', 'idx_trsf_pole_nb']
        #-------------------------
        df_no_outage_slim['premise_nbs'] = df_no_outage_slim['premise_nbs'].apply(sorted)
        if self.groupby_col=='trsf_pole_nb':
            df_no_outage_slim['serial_numbers'] = df_no_outage_slim['serial_numbers'].apply(sorted)
        #----------------------------------------------------------------------------------------------------
        self.df_no_outage_slim = df_no_outage_slim
        #----------------------------------------------------------------------------------------------------
        if self.save_dfs_to_file:
            self.df_no_outage_slim.to_pickle(os.path.join(self.save_dir_base, 'df_no_outage_slim.pkl'))


    def collect_events(
        self, 
        batch_size = None, 
        verbose    = True, 
        n_update   = 1
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        # if batch_size is None:
        #     if self.collect_evs_sum_vw:
        #         batch_size = 200
        #     else:
        #         if self.run_using_slim:
        #             batch_size = 10
        #         else:
        #             batch_size = 200

        if batch_size is None:
            if self.run_using_slim:
                batch_size = 10
            else:
                batch_size = 200
        #----------------------------------------------------------------------------------------------------
        # A little confusing below........
        #   AMIEndEvents_SQL.build_sql_end_events_for_no_outages is the method used to collect the events.
        #   The confusion enters because the aforementioned method accepts build_sql_function and build_sql_function_kwargs arguments.
        #   So, this introduces a sort of nested structure
        #   AMIEndEvents:
        #       build_sql_function        = AMIEndEvents_SQL.build_sql_end_events_for_outages
        #       build_sql_function_kwargs = dict(
        #           cols_of_interest   = ..., 
        #           df_outage          = ..., 
        #           build_sql_function = AMIEndEvents_SQL.build_sql_end_events, 
        #           build_sql_function_kwargs = dict(
        #             opcos       = ..., 
        #             date_range  = ..., 
        #             premise_nbs = ..., 
        #             etc.
        #             
        #           )
        #       )
        #----------------------------------------------------------------------------------------------------
        df_construct_type             = DFConstructType.kRunSqlQuery
        contstruct_df_args_end_events = None
        #-------------------------
        if self.groupby_col=='trsf_pole_nb':
            addtnl_groupby_cols = ['trsf_pole_nb', 'no_outg_rec_nb']
        if self.groupby_col=='PREMISE_NB':
            addtnl_groupby_cols = ['no_outg_rec_nb']
        #-----
        addtnl_groupby_cols.append('doi')
        #-----
        if self.search_window_strategy=='all_subwindows':
            addtnl_groupby_cols.append('is_first_after_outg')
        #-------------------------
        if self.collect_evs_sum_vw:
            cols_of_interest_end_dev_event = ['*']
            date_only                      = True
            build_sql_function_kwargs = dict(
                schema_name = 'meter_events', 
                table_name  = 'events_summary_vw', 
                opco        = self.opcos
            )
        else:
            cols_of_interest_end_dev_event = TableInfos.AMIEndEvents_TI.std_columns_of_interest
            date_only                      = False
            build_sql_function_kwargs = dict(
                states = self.states, 
                opcos  = self.opcos, 
                cities = self.cities
            )
        #--------------------------------------------------
        end_events_sql_function_kwargs = dict(
            cols_of_interest                  = cols_of_interest_end_dev_event, 
            build_sql_function                = AMIEndEvents_SQL.build_sql_end_events, 
            build_sql_function_kwargs         = build_sql_function_kwargs, 
            join_mp_args                      = False, 
            date_only                         = date_only, 
            output_t_minmax                   = True, 
            df_args                           = dict(
                addtnl_groupby_cols = addtnl_groupby_cols, 
                t_search_min_col    = 't_search_min', 
                t_search_max_col    = 't_search_max'
            ), 
            # GenAn - keys_to_pop in GenAn.build_sql_general 
            field_to_split                    = 'df_mp_no_outg', 
            field_to_split_location_in_kwargs = ['df_mp_no_outg'], 
            save_and_dump                     = True, 
            sort_coll_to_split                = False,
            batch_size                        = batch_size, 
            verbose                           = verbose, 
            n_update                          = n_update
        )
        #----------------------------------------------------------------------------------------------------
        if self.run_using_slim:
            end_events_sql_function_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = end_events_sql_function_kwargs, 
                default_values_dict = dict(
                    df_mp_no_outg = self.df_no_outage_slim, 
                    df_args       = dict(
                        mapping_to_ami     = DfToSqlMap(df_col='premise_nbs', kwarg='premise_nbs', sql_col='aep_premise_nb'), 
                        is_df_consolidated = True
                    )
                ), 
                extend_any_lists    = True,
                inplace             = True
            )
        #-------------------------
        else:
            df_mp_no_outg = self.df_no_outage.copy()
            if self.groupby_col=='trsf_pole_nb':
                df_mp_no_outg = df_mp_no_outg.sort_values(by=['no_outg_rec_nb', 'trsf_pole_nb', 'prem_nb', 't_search_min'], ignore_index=True)
            if self.groupby_col=='PREMISE_NB':
                df_mp_no_outg = df_mp_no_outg.sort_values(by=['no_outg_rec_nb', 'PREMISE_NB', 't_search_min'], ignore_index=True)
            #----------
            end_events_sql_function_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict = end_events_sql_function_kwargs, 
                default_values_dict = dict(
                    df_mp_no_outg = df_mp_no_outg, 
                    df_args       = dict(
                        mapping_to_ami     = DfToSqlMap(df_col='prem_nb', kwarg='premise_nbs', sql_col='aep_premise_nb'), 
                        is_df_consolidated = False
                    )
                ), 
                extend_any_lists = True,
                inplace          = True
            )
        #----------------------------------------------------------------------------------------------------
        start=time.time()
        exit_status = Utilities.run_tryexceptwhile_process(
            func                = AMIEndEvents,
            func_args_dict      = dict(
                df_construct_type         = df_construct_type, 
                contstruct_df_args        = contstruct_df_args_end_events, 
                build_sql_function        = AMIEndEvents_SQL.build_sql_end_events_for_no_outages, 
                build_sql_function_kwargs = end_events_sql_function_kwargs, 
                init_df_in_constructor    = True, 
                save_args                 = self.end_events_save_args
            ), 
            max_calls_per_min   = 1, 
            lookback_period_min = 15, 
            max_calls_absolute  = 1000, 
            verbose             = verbose
        )
        if verbose:
            print(f'exit_status = {exit_status}')
            print(f'Build time  = {time.time()-start}')
        #----------------------------------------------------------------------------------------------------
        if exit_status:
            self.save_summary_dict()
            #-------------------------
            time_infos_df = OutageDAQ.build_baseline_time_infos_df(
                data_dir_base           = self.save_dir_base, 
                min_req                 = True, 
                summary_dict_fname      = 'summary_dict.json', 
                alias                   = 'mapping_table', 
                include_summary_paths   = False, 
                consolidate             = False, 
                PN_regex                = r"prem(?:ise)?_nbs?", 
                t_min_regex             = r"t(?:_search)?_min", 
                t_max_regex             = r"t(?:_search)?_max", 
                drop_gpd_for_sql        = True, 
                return_gpd_cols         = False, 
                verbose                 = verbose
            )
            time_infos_df.to_pickle(os.path.join(self.save_dir_base, 'time_infos_df.pkl'))



    
#------------------------------------------------------------------------------------------------------------------------------------------
class OutageDAQPrBL(OutageDAQ):
    r"""
    Class to hold methods specific to Pristine BaseLine (PrBL) data acquisition
    """
    def __init__(
        self                    , 
        run_date                , 
        date_0                  , 
        date_1                  , 
        collect_evs_sum_vw      ,     # boolean
        save_sub_dir            , 
        td_left                 = pd.Timedelta('-31D'), 
        td_right                = pd.Timedelta('-1D'), 
        return_window_strategy  = 'entire', 
        states                  = None, 
        opcos                   = None, 
        cities                  = None, 
        single_zip_xfmrs_only   = False, 
        trsf_pole_nbs_to_ignore = [' ', 'TRANSMISSION', 'PRIMARY', 'NETWORK'], 
        save_end_events         = False, 
        save_dfs_to_file        = False, 
        read_dfs_from_file      = True, 
        base_dir                = os.path.join(
            Utilities.get_local_data_dir(), 
            r'dovs_and_end_events_data'
        ), 
        dates_subdir_appndx     = None, 
        run_using_slim          = True
    ):
        r"""
        """
        #--------------------------------------------------
        assert(return_window_strategy in ['all_subwindows', 'entire', 'rand'])
        self.return_window_strategy      = return_window_strategy
        #--------------------------------------------------
        self.df_outage_location_ids      = None
        self.sql_outage_location_ids     = None
        #--------------------------------------------------
        self.trsf_pole_nbs_to_ignore     = trsf_pole_nbs_to_ignore
        self.df_xfmrs_no_outg            = None
        self.trsf_pole_nbs               = None
        #-------------------------
        self.df_mp_no_outg               = None
        self.pns_with_end_events         = None
        self.df_mp_no_outg_w_events      = None 
        self.df_mp_no_outg_w_events_slim = None
        #--------------------------------------------------
        super().__init__(
            run_date                = run_date, 
            date_0                  = date_0, 
            date_1                  = date_1, 
            collect_evs_sum_vw      = collect_evs_sum_vw, 
            save_sub_dir            = save_sub_dir, 
            td_left                 = td_left, 
            td_right                = td_right, 
            states                  = states, 
            opcos                   = opcos, 
            cities                  = cities, 
            single_zip_xfmrs_only   = single_zip_xfmrs_only, 
            save_end_events         = save_end_events, 
            save_dfs_to_file        = save_dfs_to_file, 
            read_dfs_from_file      = read_dfs_from_file, 
            base_dir                = base_dir, 
            dates_subdir_appndx     = dates_subdir_appndx, 
            run_using_slim          = run_using_slim
        )


    def get_summary_dict(
        self
    ):
        r"""
        DAQ settings with adjustments to a few entries (so they can be easily output into the summary JSON file)
        e.g., Timestamp is not JSON serializable, hence the need for strftime below
        e.g., Timedelta is not JSON serializable, hence the conversion to total seconds
        """
        #--------------------------------------------------
        summary_dict = super().get_summary_dict()
        #--------------------------------------------------
        summary_dict['dataset']                 = 'prbl'
        summary_dict['rec_nb_col']              = 'no_outg_rec_nb'
        summary_dict['return_window_strategy']  = self.return_window_strategy
        summary_dict['trsf_pole_nbs_to_ignore'] = self.trsf_pole_nbs_to_ignore
        #--------------------------------------------------
        return summary_dict
    
    def save_summary_dict(
        self
    ):
        r"""
        """
        #-------------------------
        summary_dict = self.get_summary_dict()
        #-----
        assert(os.path.isdir(self.save_dir_base))
        #-----
        CustomWriter.output_dict_to_json(
            os.path.join(self.save_dir_base, 'summary_dict.json'), 
            summary_dict
        )



    def build_or_load_df_outage_location_ids(
        self, 
        verbose=True
    ):
        r"""
        Builds or sets df_outage_location_ids
        """
        #-------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_outage_location_ids from file:   {os.path.join(self.save_dir_base, 'df_outage_location_ids.pkl')}")
            #-------------------------
            self.df_outage_location_ids = pd.read_pickle(os.path.join(self.save_dir_base, 'df_outage_location_ids.pkl'))
        else:
            #--------------------------------------------------
            # First, find all transformers which HAVE experienced an outage
            start=time.time()
            sql_outage_location_ids = DOVSOutages_SQL.build_sql_find_outage_xfmrs(
                mjr_mnr_cause   = None, 
                include_premise = False, 
                date_range      = [self.date_0, self.date_1], 
                states          = self.states, 
                opcos           = self.opcos, 
                cities          = self.cities
            )
            sql_stmnt_outage_location_ids = sql_outage_location_ids.get_sql_statement()
            #-------------------------
            if verbose:
                print('-----'*20+'\nFinding all transformers which HAVE experienced an outage\n'+'-----'*10)
                print(f'sql_stmnt_outage_location_ids:\n{sql_stmnt_outage_location_ids}\n\n')
            #-------------------------
            df_outage_location_ids = pd.read_sql_query(
                sql_stmnt_outage_location_ids, 
                self.conn_outages
            )
            #-------------------------
            if verbose:
                print(f'time = {time.time()-start}\n'+'-----'*20)
            #-------------------------
            self.df_outage_location_ids  = df_outage_location_ids
            self.sql_outage_location_ids = sql_stmnt_outage_location_ids
            #-------------------------
            if self.save_dfs_to_file:
                self.df_outage_location_ids.to_pickle(os.path.join(self.save_dir_base, 'df_outage_location_ids.pkl'))


    def build_or_load_df_xfmrs_all_outg(
        self, 
        verbose=True
    ):
        r"""
        Builds or sets df_xfmrs_all, df_xfmrs_outg, and df_xfmrs_no_outg.
        -----
        Get the set of all transformers.
        Then, we will be able to find the set which did not suffer an outage 
          as set(df_xfmrs_all).difference(set(self.df_outage_location_ids))
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_xfmrs_no_outg from file:  {os.path.join(self.save_dir_base, 'df_xfmrs_no_outg.pkl')}")
            #-------------------------
            self.df_xfmrs_no_outg = pd.read_pickle(os.path.join(self.save_dir_base, 'df_xfmrs_no_outg.pkl'))
        else:
            #----------------------------------------------------------------------------------------------------
            # Next, get the set of all transformers.
            # Then, we will be able to find the set which did not suffer an outage 
            #   as set(df_xfmrs_all).difference(set(self.df_outage_location_ids))
            #----------------------------------------------------------------------------------------------------
            start=time.time()
            #--------------------------------------------------
            sql_xfmrs_all = MeterPremise.build_sql_meter_premise(
                cols_of_interest = ['DISTINCT(trsf_pole_nb)'], 
                states           = self.states, 
                opcos            = self.opcos, 
                cities           = self.cities, 
                from_table_alias = None
            )
            # If OPCOs is None/not supplied, then SQLQuery object will be returned (original, default behavior)
            # If OPCOs is supplied, then string will be returned.
            # In either case, we need sql_xfmrs_all to be a str before inputting to pd.read_sql
            assert(Utilities.is_object_one_of_types(sql_xfmrs_all, [SQLQuery, str]))
            if isinstance(sql_xfmrs_all, SQLQuery):
                sql_xfmrs_all = sql_xfmrs_all.get_sql_statement()
            #-------------------------
            df_xfmrs_all = pd.read_sql(sql_xfmrs_all, self.conn_aws) 
            #--------------------------------------------------
            # Get no outage collection as those in df_xfmrs_all without trsf_pole_nb matching those in self.df_outage_location_ids
            #   i.e., set(df_xfmrs_all).difference(set(self.df_outage_location_ids))
            #-------------------------
            df_xfmrs_no_outg = df_xfmrs_all[~df_xfmrs_all['trsf_pole_nb'].isin(self.df_outage_location_ids['LOCATION_ID'].tolist())]
            assert(df_xfmrs_no_outg['trsf_pole_nb'].shape[0]==df_xfmrs_no_outg['trsf_pole_nb'].nunique())
            #-------------------------
            if verbose:
                print('-----'*20+'\nFinding set of all transformers\n'+'-----'*10)
                print(f'sql_xfmrs_all:\n{sql_xfmrs_all}\n\n')
                print(f"# trsf_pole_nbs in no-outage collection: {df_xfmrs_no_outg['trsf_pole_nb'].nunique()}")
            #-------------------------
            #--------------------------------------------------
            if self.trsf_pole_nbs_to_ignore is not None:
                shape_b4         = df_xfmrs_no_outg.shape
                df_xfmrs_no_outg = df_xfmrs_no_outg[~df_xfmrs_no_outg['trsf_pole_nb'].isin(self.trsf_pole_nbs_to_ignore)]
                #-------------------------
                if verbose:
                    print('-----'*20+'\nRemoving unwanted trsf_pole_nbs from no-outage collection\n')
                    print(f"BEFORE: {shape_b4}")
                    print(f"AFTER : {df_xfmrs_no_outg.shape}")
                    print('-----'*20)
                #-------------------------
            #--------------------------------------------------
            # Get outage collection as those in df_xfmrs_all with trsf_pole_nb matching those in self.df_outage_location_ids
            df_xfmrs_outg    = df_xfmrs_all[df_xfmrs_all['trsf_pole_nb'].isin(self.df_outage_location_ids['LOCATION_ID'].tolist())]
            #--------------------------------------------------
            self.df_xfmrs_no_outg = df_xfmrs_no_outg
            #--------------------------------------------------
            if verbose:
                print(f'\n\ntime = {time.time()-start}\n'+'-----'*20)
            #--------------------------------------------------
            if self.save_dfs_to_file:
                df_xfmrs_all.to_pickle(         os.path.join(self.save_dir_base, 'df_xfmrs_all.pkl'))
                df_xfmrs_outg.to_pickle(        os.path.join(self.save_dir_base, 'df_xfmrs_outg.pkl'))
                self.df_xfmrs_no_outg.to_pickle(os.path.join(self.save_dir_base, 'df_xfmrs_no_outg.pkl'))
        #----------------------------------------------------------------------------------------------------
        # Should have been handled, but to be certain when DFs are read in, reduce again
        if self.trsf_pole_nbs_to_ignore is not None:
            self.df_xfmrs_no_outg = self.df_xfmrs_no_outg[~self.df_xfmrs_no_outg['trsf_pole_nb'].isin(self.trsf_pole_nbs_to_ignore)]
        #--------------------------------------------------
        self.trsf_pole_nbs = self.df_xfmrs_no_outg['trsf_pole_nb'].unique().tolist()


    def build_or_load_trsf_pole_zips_info(
        self, 
        verbose=False
    ):
        r"""
        This either loads or builds (depending on the value of self.read_dfs_from_file) mp_for_zips_df, trsf_pole_df_full, trsf_pole_zips_df.
        FURTHERMORE, depending on the value of self.single_zip_xfmrs_only, self.trsf_pole_nbs may be altered
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading mp_for_zips_df from file:      {os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl')}")
                print(f"Reading trsf_pole_df_full from file:   {os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl')}")
                print(f"Reading trsf_pole_zips_df from file:   {os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl')}")
            #-------------------------
            self.mp_for_zips_df    = pd.read_pickle(os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl'))
            self.trsf_pole_df_full = pd.read_pickle(os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl'))
            self.trsf_pole_zips_df = pd.read_pickle(os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl'))
        else:
            zips_dict = OutageDAQ.build_trsf_pole_zips_df(
                field_to_split_and_val = ('trsf_pole_nbs', self.trsf_pole_nbs), 
                states                 = self.states, 
                opcos                  = self.opcos, 
                cities                 = self.cities
            )
            #----------
            self.trsf_pole_zips_df = zips_dict['trsf_pole_zips_df']
            self.trsf_pole_df_full = zips_dict['trsf_pole_df_full']
            self.mp_for_zips_df    = zips_dict['mp_for_zips_df']
            #--------------------------------------------------
            if self.save_dfs_to_file:
                self.mp_for_zips_df.to_pickle(   os.path.join(self.save_dir_base, 'mp_for_zips_df.pkl'))
                self.trsf_pole_df_full.to_pickle(os.path.join(self.save_dir_base, 'trsf_location_info_df.pkl'))
                self.trsf_pole_zips_df.to_pickle(os.path.join(self.save_dir_base, 'trsf_pole_zips_df.pkl'))
        #----------------------------------------------------------------------------------------------------
        if self.single_zip_xfmrs_only:
            trsf_pole_nzips   = self.trsf_pole_zips_df.drop(columns=['zip+4']).drop_duplicates()['trsf_pole_nb'].value_counts()
            single_zip_poles  = trsf_pole_nzips[trsf_pole_nzips==1].index.tolist()
            #-----
            self.trsf_pole_nbs = self.mp_for_zips_df[self.mp_for_zips_df['trsf_pole_nb'].isin(single_zip_poles)]['trsf_pole_nb'].unique().tolist()

    
    def build_or_load_df_mp_no_outg(
        self            , 
        verbose         = True, 
        ensure_no_lists = False
    ):
        r"""
        Reads df_mp_no_outg_OG from file df_mp_no_outg_full.pkl
        If building from scratch, results output to df_mp_no_outg_curr_hist.pkl then df_mp_no_outg_full.pkl

        ensure_no_lists:
            Suggest keeping False, as it takes time and shouldn't really be necessary anymore, post dev
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_mp_no_outg from file:      {os.path.join(self.save_dir_base, 'df_mp_no_outg_full.pkl')}")
            #-------------------------
            df_mp_no_outg_OG = pd.read_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg_full.pkl'))
        #----------------------------------------------------------------------------------------------------
        else:
            #----------------------------------------------------------------------------------------------------
            # Build joined current and historical MeterPremise collections for all transformers
            #   found in df_xfmrs_no_outg.
            # Need to use both current and historical because each transformer will be assigned a random
            #   datetime (between some date_0 and date_1 limites) below around which to collect events.
            #----------------------------------------------------------------------------------------------------
            start = time.time()
            #-----
            addtnl_mp_df_cols = ['state_cd', 'circuit_nb', 'circuit_nm', 'station_nb', 'station_nm']
            df_mp_no_outg_OG = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
                trsf_pole_nbs          = self.trsf_pole_nbs, 
                join_curr_hist         = True, 
                drop_approx_duplicates = False, 
                addtnl_mp_df_curr_cols = addtnl_mp_df_cols, 
                addtnl_mp_df_hist_cols = addtnl_mp_df_cols
            )
            if verbose:
                print(
                    '-----'*20+'\nBuilding joined current and historical MeterPremise collections '+
                    'for all transformers in no-outage collection\n'+'-----'*10
                )
                print(f"\nFound MeterPremise data for {df_mp_no_outg_OG['trsf_pole_nb'].nunique()} transformers")
                print(f'\ntime = {time.time()-start}\n'+'-----'*20)
            #****************************************************************************************************
            # Since df_mp_no_outg_OG takes such a significant time to build, save at this point, so all
            #   is not lost in case of crash
            if self.save_dfs_to_file:
                df_mp_no_outg_OG.to_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg_curr_hist.pkl'))
            #****************************************************************************************************
            #----------------------------------------------------------------------------------------------------
            # Set t_search_min/t_search_max columns according to the strategy set as self.return_window_strategy
            #----------------------------------------------------------------------------------------------------
            #--------------------------------------------------
            if self.return_window_strategy == 'rand':
                # For each transformer, set a random date interval between date_0 and date_1 around which
                #   results will be acquired.
                # This is analogous to the outage event for that dataset.
                #--------------------------------------------------
                start = time.time()
                df_mp_no_outg_OG = OutageDAQ.set_random_date_interval_in_df_by_xfmr(
                    df             = df_mp_no_outg_OG, 
                    date_0         = self.date_0, 
                    date_1         = self.date_1, 
                    window_width   = self.window_width, 
                    xfmr_col       = 'trsf_pole_nb', 
                    rand_seed      = None, 
                    placement_cols = ['t_search_min', 't_search_max'], 
                    inplace        = True    
                )
                # If I'm looking at a single year, with a window_width of 30 days, there are only
                # 365-30=335 possible unique start dates, regardless of number of groups.
                # So, there are bound to be repeats!
                # HOWEVER, I am now using also a random time for each day, so the likeliness of repeats
                #   is significantly reduced!
                if verbose:
                    print('-----'*20+f'\nSetting random date interval of width={self.window_width} between date_0={self.date_0} and date_1={self.date_1}'+
                        '\nfor all remaining transformers in no-outage collection\n')
                    print(f"# trsf_pole_nbs: {df_mp_no_outg_OG['trsf_pole_nb'].nunique()}")
                    print(f"# t_search_mins: {df_mp_no_outg_OG['t_search_min'].nunique()}")
                    print(f"# t_search_maxs: {df_mp_no_outg_OG['t_search_max'].nunique()}")
                    print(f'\ntime = {time.time()-start}\n'+'-----'*20)
            #--------------------------------------------------
            elif self.return_window_strategy == 'all_subwindows':
                # ALL PRISTINES WILL HAVE THE SAME WINDOWS, which should be fine.
                # If one did want differnt transformers to have different windows, one would need to set
                #   unique t_clean_min and t_clean_max for each transformer, but I don't see the reason for doing so at this point
                #-----
                df_mp_no_outg_OG['t_clean_min'] = self.date_0
                df_mp_no_outg_OG['t_clean_max'] = self.date_1
                #-----
                df_mp_no_outg_OG = OutageDAQ.find_clean_subwindows_for_group(
                    final_df_i                      = df_mp_no_outg_OG, 
                    min_window_width                = self.window_width, 
                    include_is_first_after_outg_col = False, 
                    t_clean_min_col                 = 't_clean_min', 
                    t_clean_max_col                 = 't_clean_max', 
                    return_t_search_min_col         = 't_search_min', 
                    return_t_search_max_col         = 't_search_max'
                )
            #--------------------------------------------------
            elif self.return_window_strategy == 'entire':
                df_mp_no_outg_OG['t_search_min'] = self.date_0
                df_mp_no_outg_OG['t_search_max'] = self.date_1
            else:
                assert(0)
            #----------------------------------------------------------------------------------------------------
            # Currently, df_mp_no_outg_OG contains all current and historical MeterPremise data.
            # Keep only the data for the meters active during the relevant time periods, as
            #   dictated by the t_search_min and t_search_max fields.
            #----------------------------------------------------------------------------------------------------
            shape_b4 = df_mp_no_outg_OG.shape
            df_mp_no_outg_OG = df_mp_no_outg_OG[
                (df_mp_no_outg_OG['inst_ts']                          <= df_mp_no_outg_OG['t_search_min']) & 
                (df_mp_no_outg_OG['rmvl_ts'].fillna(pd.Timestamp.max) >  df_mp_no_outg_OG['t_search_max'])
            ]
            if verbose:
                print('-----'*20+f'\nKeep only meters active for the relevant times\n')
                print(f"BEFORE: df_mp_no_outg_OG.shape = {shape_b4}")
                print(f"AFTER : df_mp_no_outg_OG.shape = {df_mp_no_outg_OG.shape}")
                print('-----'*20)
    
            #----------------------------------------------------------------------------------------------------
            # Drop approximate duplicates from df_mp_no_outg_OG so there is a single entry for each 
            #   serial number/premise
            #----------------------------------------------------------------------------------------------------
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # PROBLEM:
            # drop_approx_mp_duplicates will turn some column elements into lists, which is an issue 
            #   when this is plugged into consolidate_df.
            # The current solution is to have the addtnl_groupby_cols below, but a more general solution 
            #   should be implemented, e.g., if lists, then join the lists or whatever
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            shape_b4            = df_mp_no_outg_OG.shape
            addtnl_groupby_cols = addtnl_mp_df_cols+['t_search_min', 't_search_max']
            if self.return_window_strategy == 'all_subwindows':
                addtnl_groupby_cols.extend(['t_clean_min', 't_clean_max'])
            df_mp_no_outg_OG = MeterPremise.drop_approx_mp_duplicates(
                mp_df               = df_mp_no_outg_OG, 
                fuzziness           = pd.Timedelta('1 hour'), 
                addtnl_groupby_cols = addtnl_groupby_cols, 
            )
            #-------------------------
            if verbose:
                print('-----'*20+f'\nDrop approximate duplicates\n')
                print(f"BEFORE: df_mp_no_outg_OG.shape = {shape_b4}")
                print(f"AFTER : df_mp_no_outg_OG.shape = {df_mp_no_outg_OG.shape}")
                print(f'\ntime = {time.time()-start}\n'+'-----'*20)
            #-------------------------
            if ensure_no_lists:
                # Make sure no list entries! (PROBABLY NOT NEEDED ANYMORE AFTER DEV STAGE!)
                # See also Utilities_df.find_columns_with_list_element/does_df_contain_any_column_elements
                if df_mp_no_outg_OG.map(lambda x: isinstance(x, list)).any().any():
                    print('In OutageDAQPrBL.build_or_load_df_mp_no_outg: LISTS FOUND IN df_mp_no_outg_OG!!!!!')
                    print(df_mp_no_outg_OG.map(lambda x: isinstance(x, list)).any())
                    assert(0)
    
            #----------------------------------------------------------------------------------------------------
            # Add no_outg_rec_nb column to allow easier grouping when building rcpo_dfs, and for identification later
            #----------------------------------------------------------------------------------------------------
            start=time.time()
            rand_pfx = Utilities.generate_random_string(str_len=5, letters=string.ascii_letters + string.digits)
            df_mp_no_outg_OG['no_outg_rec_nb'] = df_mp_no_outg_OG.groupby(['trsf_pole_nb', 't_search_min', 't_search_max']).ngroup()
            df_mp_no_outg_OG['no_outg_rec_nb'] = rand_pfx + df_mp_no_outg_OG['no_outg_rec_nb'].astype(str)
            if verbose:
                print('-----'*20+f'\nCreating no_outg_rec_nb column\n')
                print(f'\ntime = {time.time()-start}\n'+'-----'*20)
            
            #----------------------------------------------------------------------------------------------------
            #--------------------------------------------------
            # Set the date of interest (doi) column, which is akin to the DT_OFF_TS_FULL column for the outg case
            #   The doi is the reference point from which the various time groupings will be formed.
            # Reconstruction of prediction_date/doi
            #   range_left  = prediction_date + td_left  ==> prediction_date = range_left - td_left
            #   range_right = prediction_date + td_right ==> prediction_date = range_right - td_right
            # As shown above, reconstruction using td_left or td_right should yield the same result.
            # IF FOR SOME REASON THEY YIELD DIFFERENT RESULTS, BOTH WILL BE KEPT (columns 'doi', from left, and 'doi_r' from right)
            df_mp_no_outg_OG['doi']   = df_mp_no_outg_OG['t_search_min'] - self.td_left
            df_mp_no_outg_OG['doi_r'] = df_mp_no_outg_OG['t_search_max'] - self.td_right
            if df_mp_no_outg_OG['doi'].equals(df_mp_no_outg_OG['doi_r']):
                df_mp_no_outg_OG = df_mp_no_outg_OG.drop(columns=['doi_r'])
            else:
                if verbose:
                    print("!!!!! WARNING: OutageDAQPrBL.build_or_load_df_mp_no_outg !!!!!\n\tdoi and doi_r columns disagree, so both will be kept!")

            #****************************************************************************************************
            if self.save_dfs_to_file:
                df_mp_no_outg_OG.to_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg_full.pkl'))
            #****************************************************************************************************
        #----------------------------------------------------------------------------------------------------
        # Install and removal timestamps no longer needed, so remove.
        cols_to_drop = ['inst_ts', 'rmvl_ts']
        cols_to_drop = [x for x in cols_to_drop if x in df_mp_no_outg_OG.columns.tolist()]
        if len(cols_to_drop)>0:
            df_mp_no_outg_OG = df_mp_no_outg_OG.drop(columns=cols_to_drop)
        # Sort by no_outg_rec_nb, so all like 'no outage events' are sequential in DF
        df_mp_no_outg_OG = df_mp_no_outg_OG.sort_values(by=['no_outg_rec_nb', 'trsf_pole_nb', 'prem_nb', 'mfr_devc_ser_nbr'], ignore_index=True)
        #-------------------------
        self.df_mp_no_outg = df_mp_no_outg_OG

    def build_or_load_pns_with_end_events(
        self, 
        verbose=True
    ):
        r"""
        Find the premise numbers which recorded an event between date_0 and date_1.
        Premises without any events during the period will be removed from df_mp_no_outg_OG.
        This is relatively quick, and saves much time in the data acquisition step as we do not
          have to waste time searching for empty results (which is suprisingly sluggish)
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading pns_with_end_events from file:    {os.path.join(self.save_dir_base, 'pns_with_end_events.pkl')}")
                print(f"Reading df_mp_no_outg_w_events from file: {os.path.join(self.save_dir_base, 'df_mp_no_outg_PNs_w_events.pkl')}")
            #-------------------------
            self.pns_with_end_events    = pd.read_pickle(os.path.join(self.save_dir_base, 'pns_with_end_events.pkl'))
            self.df_mp_no_outg_w_events = pd.read_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg_PNs_w_events.pkl'))
        #----------------------------------------------------------------------------------------------------
        else:
            #----------------------------------------------------------------------------------------------------
            # Find the premise numbers which recorded an event between date_0 and date_1.
            # Premises without any events during the period will be removed from df_mp_no_outg_OG.
            # This is relatively quick, and saves much time in the data acquisition step as we do not
            #   have to waste time searching for empty results (which is suprisingly sluggish)
            #----------------------------------------------------------------------------------------------------
            start=time.time()
            pns_with_end_events = AMIEndEvents.get_end_event_distinct_fields(
                date_0                           = self.date_0, 
                date_1                           = self.date_1, 
                fields                           = ['aep_premise_nb'], 
                are_datetime                     = False, 
                addtnl_build_sql_function_kwargs = dict(
                    states = self.states, 
                    opcos  = self.opcos, 
                    cities = self.cities
                )
            )
            #-------------------------
            df_mp_no_outg_w_events = self.df_mp_no_outg[self.df_mp_no_outg['prem_nb'].isin(pns_with_end_events['aep_premise_nb'].tolist())].copy()
            #-------------------------
            self.pns_with_end_events    = pns_with_end_events
            self.df_mp_no_outg_w_events = df_mp_no_outg_w_events
            #-------------------------
            if verbose:
                print('-----'*20+f'\nFind PNs with recorded events between date_0={self.date_0} and date_1={self.date_1}\n'+'-----'*10)
                print('Removing PNs without recorded events to speed up data acquisition\n'+'-----'*10)
                print(f'BEFORE: df_mp_no_outg.shape = {self.df_mp_no_outg.shape}')
                print(f"        df_mp_no_outg['trsf_pole_nb'].nunique() = {self.df_mp_no_outg['trsf_pole_nb'].nunique()}")
                print(f'AFTER:  df_mp_no_outg_w_events.shape = {df_mp_no_outg_w_events.shape}')
                print(f"        df_mp_no_outg_w_events['trsf_pole_nb'].nunique() = {df_mp_no_outg_w_events['trsf_pole_nb'].nunique()}")
                #----------
                print(f'\n\ntime = {time.time()-start}\n'+'-----'*20)
            #-------------------------
            if self.save_dfs_to_file:
                self.pns_with_end_events.to_pickle(   os.path.join(self.save_dir_base, 'pns_with_end_events.pkl'))
                self.df_mp_no_outg_w_events.to_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg_PNs_w_events.pkl'))


    def build_or_load_df_mp_no_outg_w_events_slim(
        self, 
        verbose=True
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        if self.read_dfs_from_file:
            if verbose:
                print(f"Reading df_mp_no_outg_w_events_slim from file: {os.path.join(self.save_dir_base, 'df_mp_no_outg_PNs_w_events_slim.pkl')}")
            #-------------------------
            self.df_mp_no_outg_w_events_slim    = pd.read_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg_PNs_w_events_slim.pkl'))
        #----------------------------------------------------------------------------------------------------
        else:
            #----------------------------------------------------------------------------------------------------
            # Build consolidated (slim) version of df_mp_no_outage
            #----------------------------------------------------------------------------------------------------
            addtnl_mp_df_cols    = ['state_cd', 'circuit_nb', 'circuit_nm', 'station_nb', 'station_nm']
            cols_shared_by_group = addtnl_mp_df_cols
            cols_to_collect_in_lists = ['mfr_devc_ser_nbr', 'prem_nb']
            rename_cols = {'mfr_devc_ser_nbr':'serial_numbers', 'prem_nb':'premise_nbs'}
            #-------------------------
            start=time.time()
            df_mp_no_outg_slim_OG = Utilities_df.consolidate_df(
                df                                  = self.df_mp_no_outg_w_events, 
                groupby_cols                        = ['no_outg_rec_nb', 'trsf_pole_nb', 't_search_min', 't_search_max', 'doi'], 
                cols_shared_by_group                = cols_shared_by_group, 
                cols_to_collect_in_lists            = cols_to_collect_in_lists, 
                include_groupby_cols_in_output_cols = False, 
                allow_duplicates_in_lists           = False, 
                recover_uniqueness_violators        = True, 
                rename_cols                         = rename_cols, 
                verbose                             = True
            )
            #-------------------------
            df_mp_no_outg_slim_OG=df_mp_no_outg_slim_OG.reset_index().set_index(['no_outg_rec_nb', 'trsf_pole_nb'], drop=False)
            df_mp_no_outg_slim_OG.index.names=['idx_no_outg_rec_nb', 'idx_trsf_pole_nb']
            #-----
            df_mp_no_outg_slim_OG['premise_nbs']    = df_mp_no_outg_slim_OG['premise_nbs'].apply(sorted)
            df_mp_no_outg_slim_OG['serial_numbers'] = df_mp_no_outg_slim_OG['serial_numbers'].apply(sorted)
            #-------------------------
            self.df_mp_no_outg_w_events_slim = df_mp_no_outg_slim_OG
            #-------------------------
            if verbose:
                print('-----'*20+f'\nBuilding consolidated version of df_mp_no_outg\n'+'-----'*10)
                print(f'\ntime = {time.time()-start}\n'+'-----'*20)
            #-------------------------
            if self.save_dfs_to_file:
                self.df_mp_no_outg_w_events_slim.to_pickle(os.path.join(self.save_dir_base, 'df_mp_no_outg_PNs_w_events_slim.pkl'))


    def collect_events(
        self, 
        batch_size = None, 
        verbose    = True, 
        n_update   = 1
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        # if batch_size is None:
        #     if self.collect_evs_sum_vw:
        #         batch_size = 200
        #     else:
        #         if self.run_using_slim:
        #             batch_size = 10
        #         else:
        #             batch_size = 200

        if batch_size is None:
            if self.run_using_slim:
                batch_size = 10
            else:
                batch_size = 200
        #----------------------------------------------------------------------------------------------------
        # A little confusing below........
        #   AMIEndEvents_SQL.build_sql_end_events_for_no_outages is the method used to collect the events.
        #   The confusion enters because the aforementioned method accepts build_sql_function and build_sql_function_kwargs arguments.
        #   So, this introduces a sort of nested structure
        #   AMIEndEvents:
        #       build_sql_function        = AMIEndEvents_SQL.build_sql_end_events_for_outages
        #       build_sql_function_kwargs = dict(
        #           cols_of_interest   = ..., 
        #           df_outage          = ..., 
        #           build_sql_function = AMIEndEvents_SQL.build_sql_end_events, 
        #           build_sql_function_kwargs = dict(
        #             opcos       = ..., 
        #             date_range  = ..., 
        #             premise_nbs = ..., 
        #             etc.
        #             
        #           )
        #       )
        #----------------------------------------------------------------------------------------------------
        df_construct_type              = DFConstructType.kRunSqlQuery
        contstruct_df_args_end_events  = None
        addtnl_groupby_cols            = ['trsf_pole_nb', 'no_outg_rec_nb', 'doi']
        #-------------------------
        if self.collect_evs_sum_vw:
            cols_of_interest_end_dev_event = ['*']
            date_only                      = True
            build_sql_function_kwargs = dict(
                schema_name = 'meter_events', 
                table_name  = 'events_summary_vw', 
                opcos       = self.opcos
            )
        else:
            cols_of_interest_end_dev_event = TableInfos.AMIEndEvents_TI.std_columns_of_interest
            date_only                      = False
            build_sql_function_kwargs = dict(
                states = self.states, 
                opcos  = self.opcos, 
                cities = self.cities
            )
        #--------------------------------------------------
        end_events_sql_function_kwargs = dict(
            cols_of_interest                  = cols_of_interest_end_dev_event, 
            build_sql_function                = AMIEndEvents_SQL.build_sql_end_events, 
            build_sql_function_kwargs         = build_sql_function_kwargs, 
            join_mp_args                      = False, 
            date_only                         = date_only, 
            output_t_minmax                   = True, 
            df_args                           = dict(
                addtnl_groupby_cols = addtnl_groupby_cols, 
                t_search_min_col    = 't_search_min', 
                t_search_max_col    = 't_search_max'
            ), 
            # GenAn - keys_to_pop in GenAn.build_sql_general 
            field_to_split                    = 'df_mp_no_outg', 
            field_to_split_location_in_kwargs = ['df_mp_no_outg'], 
            save_and_dump                     = True, 
            sort_coll_to_split                = False,
            batch_size                        = batch_size, 
            verbose                           = verbose, 
            n_update                          = n_update
        )
        #----------------------------------------------------------------------------------------------------
        if self.run_using_slim:
            end_events_sql_function_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = end_events_sql_function_kwargs, 
                default_values_dict = dict(
                    df_mp_no_outg   = self.df_mp_no_outg_w_events_slim, 
                    df_args         = dict(
                        mapping_to_ami     = DfToSqlMap(df_col='premise_nbs', kwarg='premise_nbs', sql_col='aep_premise_nb'), 
                        is_df_consolidated = True
                    )
                ), 
                extend_any_lists    = True,
                inplace             = True
            )
        #-------------------------
        else:
            df_mp_no_outg = self.df_mp_no_outg_w_events.copy()
            df_mp_no_outg = df_mp_no_outg.sort_values(by=['no_outg_rec_nb', 'trsf_pole_nb', 'prem_nb', 't_search_min'], ignore_index=True)
            #----------
            end_events_sql_function_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict = end_events_sql_function_kwargs, 
                default_values_dict = dict(
                    df_mp_no_outg   = df_mp_no_outg, 
                    df_args         = dict(
                        mapping_to_ami     = DfToSqlMap(df_col='prem_nb', kwarg='premise_nbs', sql_col='aep_premise_nb'), 
                        is_df_consolidated = False
                    )
                ), 
                extend_any_lists    = True,
                inplace             = True
            )
        #----------------------------------------------------------------------------------------------------
        start=time.time()
        exit_status = Utilities.run_tryexceptwhile_process(
            func                = AMIEndEvents,
            func_args_dict      = dict(
                df_construct_type         = df_construct_type, 
                contstruct_df_args        = contstruct_df_args_end_events, 
                build_sql_function        = AMIEndEvents_SQL.build_sql_end_events_for_no_outages, 
                build_sql_function_kwargs = end_events_sql_function_kwargs, 
                init_df_in_constructor    = True, 
                save_args                 = self.end_events_save_args
            ), 
            max_calls_per_min   = 1, 
            lookback_period_min = 15, 
            max_calls_absolute  = 1000, 
            verbose             = verbose
        )
        if verbose:
            print(f'exit_status = {exit_status}')
            print(f'Build time  = {time.time()-start}')
        #----------------------------------------------------------------------------------------------------
        if exit_status:
            self.save_summary_dict()
            #-------------------------
            time_infos_df = OutageDAQ.build_baseline_time_infos_df(
                data_dir_base           = self.save_dir_base, 
                min_req                 = True, 
                summary_dict_fname      = 'summary_dict.json', 
                alias                   = 'mapping_table', 
                include_summary_paths   = False, 
                consolidate             = False, 
                PN_regex                = r"prem(?:ise)?_nbs?", 
                t_min_regex             = r"t(?:_search)?_min", 
                t_max_regex             = r"t(?:_search)?_max", 
                drop_gpd_for_sql        = True, 
                return_gpd_cols         = False, 
                verbose                 = verbose
            )
            time_infos_df.to_pickle(os.path.join(self.save_dir_base, 'time_infos_df.pkl'))
