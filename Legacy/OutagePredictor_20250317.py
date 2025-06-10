#!/usr/bin/env python

r"""
Holds OutagePredictor class.  See OutagePredictor.OutagePredictor for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import json
import pickle
import _pickle as cPickle
import joblib

import pandas as pd
import numpy as np
import datetime
from natsort import natsorted

import copy
import warnings
#---------------------------------------------------------------------
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from AMI_SQL import AMI_SQL
from AMIEndEvents_SQL import AMIEndEvents_SQL
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from MeterPremise import MeterPremise
from EEMSP import EEMSP
from AMIEndEvents import AMIEndEvents
from MECPODf import MECPODf
from DOVSOutages import DOVSOutages
from OutageDAQ import OutageDAQ
from OutageMdlrPrep import OutageMdlrPrep
from CPXDfBuilder import CPXDfBuilder
from CPXDf import CPXDf
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
from SQLSelect import SQLSelect
from SQLFrom import SQLFrom
from SQLWhere import SQLWhere
from SQLQuery import SQLQuery
#---------------------------------------------------------------------
#sys.path.insert(0, os.path.join(os.path.realpath('..'), 'Utilities'))
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
import Utilities_dt
from Utilities_df import DFConstructType
#---------------------------------------------------------------------

class OutagePredictor:
    r"""
    Class to make outage predictions
    """
    def __init__(
        self, 
        prediction_date = None, 
        trsf_pole_nbs   = None, 
        idk_name_1      = pd.Timedelta('31D'), 
        idk_name_2      = pd.Timedelta('1D'), 
        outg_pred       = None, # if copy constructor functionality is desired
        min_mem_load    = False
    ):
        r"""
        !!!!! IMPORTANT !!!!!
        If you add attributes to the class, don't forget to add to copy_constructor method (and possibly
          the load_min_memory and save_min_memory memories as well)
        """
        #---------------------------------------------------------------------------
        if outg_pred is not None:
            assert(
                isinstance(outg_pred, OutagePredictor) or 
                (isinstance(outg_pred, str) and os.path.exists(outg_pred))
            )
            #-------------------------
            if isinstance(outg_pred, OutagePredictor):
                self.copy_constructor(outg_pred)
            else:
                if min_mem_load:
                    self.load_min_memory(
                        file_path  = outg_pred, 
                        keep_attrs = None

                    )
                else:
                    self.load(file_path = outg_pred)
            return
        #---------------------------------------------------------------------------
        # Grabbing connection can take time (seconds, not hours).
        # Keep set to None, only creating if needed (see conn_aws property below)
        self.__conn_aws              = None
        self.__conn_dovs             = None
        self.__conn_eems             = None  #NOTE: Typically using the EEMS data which has been moved to Athena, so usually don't need this connection
        #-------------------------
        self.__trsf_pole_nbs         = None
        self.trsf_pole_nbs_sql       = '' # If SQL run to obtain trsf_pole_nbs, this will be set
        if trsf_pole_nbs is not None:
            self.set_trsf_pole_nbs(trsf_pole_nbs)
        #-----
        self.__trsf_pole_nbs_found   = None # Will be set when self.rcpx_df is built
        self.__trsf_pole_zips_df     = None # Will be built if needed
        #-------------------------
        if prediction_date is None:
            self.prediction_date     = pd.to_datetime(datetime.date.today())
        else:
            self.prediction_date     = prediction_date
        #-----
        self.idk_name_1              = idk_name_1
        self.idk_name_2              = idk_name_2
        #-----
        self.date_range              = None
        self.set_date_range()
        #-------------------------
        self.daq_date_ranges         = [] # Tracks the date ranges of the acquired data
        #-------------------------
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.opco                    = None
        self.states                  = None
        self.cities                  = None
        #-------------------------
        # Although I will use AMIEndEvents methods, the object which is built will not
        #   contain end_device_event entries (ede), but rather events_summary_vw (evsSum)
        self.__evsSum_sql_fcn        = None
        self.__evsSum_sql_kwargs     = None
        self.__evsSum_sql_stmnts     = []     # If prediction date changed or whatever, evsSum_df will be agg. of multiple DAQ runs
        self.__evsSum_df             = None 
        self.__zero_counts_evsSum_df = None   # IMPORTANT: Remember, the 'serialnumber' and 'aep_premise_nb' are essentially meaningless in this DF
        self.__zc_xfmrs_by_date      = dict() # This may be a temporary member, keeping because seeing some (possibly) strange things during dev
        #-------------------------
        self.__regex_setup_df        = None
        self.__cr_trans_dict         = None
        #-----
        self.rcpx_df                 = None
        self.is_norm                 = False
        #-------------------------
        self.merge_eemsp             = False
        self.eemsp_mult_strategy     = 'agg'
        self.eemsp_df                = None
        #-------------------------
        self.__model_dir             = None
        self.model_summary_dict      = None
        self.data_structure_df       = None
        self.model_clf               = None
        self.scale_data              = False
        self.scaler                  = None
        self.eemsp_enc               = None
        self.include_month           = False
        #-----
        self.model_by_zip_clusters   = False
        self.zip_clstrs_ledger       = None
        self.xfmrs_ledger            = None 
        #-----
        self.__X_test                = None
        self.__y_pred                = None
        self.__y_pred_df             = None

    
    @property
    def evsSum_df(self):
        if self.__evsSum_df is None:
            return None
        return self.__evsSum_df.copy()
        
    @property
    def zero_counts_evsSum_df(self):
        if self.__zero_counts_evsSum_df is None:
            return None
        return self.__zero_counts_evsSum_df.copy()
    
    @property
    def zc_xfmrs_by_date(self):
        return copy.deepcopy(self.__zc_xfmrs_by_date)
        
    @property
    def evsSum_sql_stmnts(self):
        return self.__evsSum_sql_stmnts.copy()
    
    @property
    def conn_aws(self):
        if self.__conn_aws is None:
            self.__conn_aws  = Utilities.get_athena_prod_aws_connection()
        return self.__conn_aws
    
    @property
    def conn_dovs(self):
        if self.__conn_dovs is None:
            self.__conn_dovs  = Utilities.get_utldb01p_oracle_connection()
        return self.__conn_dovs
    
    @property
    def conn_eems(self):
        if self.__conn_eems is None:
            self.__conn_eems  = Utilities.get_eemsp_oracle_connection()
        return self.__conn_eems
    
    @property
    def trsf_pole_nbs(self):
        return self.__trsf_pole_nbs
    @property
    def trsf_pole_nbs_found(self):
        return self.__trsf_pole_nbs_found
    def set_trsf_pole_nbs_found(self, trsf_pole_nbs_found):
        self.__trsf_pole_nbs_found = trsf_pole_nbs_found

    @property
    def trsf_pole_zips_df(self):
        return OutagePredictor.general_copy(self.__trsf_pole_zips_df)
    
    @property
    def regex_setup_df(self):
        if self.__regex_setup_df is None:
            self.build_and_load_regex_setup()
        return self.__regex_setup_df
    
    @property
    def cr_trans_dict(self):
        if self.__cr_trans_dict is None:
            self.build_and_load_regex_setup()
        return self.__cr_trans_dict
    
    @property
    def evsSum_sql_fcn(self):
        return self.__evsSum_sql_fcn
    
    @property
    def evsSum_sql_kwargs(self):
        return self.__evsSum_sql_kwargs
    
    @property
    def model_dir(self):
        return self.__model_dir
    
    @property
    def X_test(self):
        if self.__X_test is None:
            return None
        return self.__X_test.copy()
    def set_X_test(self, X_test):
        self.__X_test = X_test

    @property
    def y_pred(self):
        if self.__y_pred is None:
            return None
        return self.__y_pred.copy()
    @property
    def y_pred_df(self):
        if self.__y_pred_df is None:
            return None
        return self.__y_pred_df.copy()
    
    
    #----------------------------------------------------------------------------------------------------
    @staticmethod
    def general_copy(
        attr
    ):
        r"""
        """
        #-------------------------
        if attr is None:
            return None
        #-------------------------
        # pd.DataFrame, pd.Series, and others have their own, built-in, .copy methods
        try:
            cpy = attr.copy()
            return cpy
        except:
            pass
        #-------------------------
        cpy = copy.deepcopy(attr)
        return cpy
        
    def copy_constructor(
        self, 
        outg_pred
    ):
        r"""
        Annoyingly, the follow simple solution does not work:
            self = copy.deepcopy(outg_pred)
          neither does:
            self = OutagePredictor()
            self.__dict__ = copy.deepcopy(outg_pred.__dict__)
    
        So, I guess it's back to headache C++ style...
        """
        #--------------------------------------------------
        assert(isinstance(outg_pred, OutagePredictor))
        #--------------------------------------------------
        self.__conn_aws                = None
        self.__conn_dovs               = None
        self.__conn_eems               = None
        #--------------------------------------------------
        self.__trsf_pole_nbs           = OutagePredictor.general_copy(outg_pred.__trsf_pole_nbs)
        self.trsf_pole_nbs_sql         = OutagePredictor.general_copy(outg_pred.trsf_pole_nbs_sql)
        self.__trsf_pole_nbs_found     = OutagePredictor.general_copy(outg_pred.__trsf_pole_nbs_found)
        self.__trsf_pole_zips_df     = OutagePredictor.general_copy(outg_pred.__trsf_pole_zips_df)
        #--------------------------------------------------
        self.prediction_date           = OutagePredictor.general_copy(outg_pred.prediction_date)
        self.idk_name_1                = OutagePredictor.general_copy(outg_pred.idk_name_1)
        self.idk_name_2                = OutagePredictor.general_copy(outg_pred.idk_name_2)
        self.set_date_range()
        #--------------------------------------------------
        self.daq_date_ranges           = OutagePredictor.general_copy(outg_pred.daq_date_ranges)
        #--------------------------------------------------
        self.opco                      = OutagePredictor.general_copy(outg_pred.opco)
        self.states                    = OutagePredictor.general_copy(outg_pred.states)
        self.cities                    = OutagePredictor.general_copy(outg_pred.cities)
        #--------------------------------------------------
        self.__evsSum_sql_fcn          = OutagePredictor.general_copy(outg_pred.__evsSum_sql_fcn)
        self.__evsSum_sql_kwargs       = OutagePredictor.general_copy(outg_pred.__evsSum_sql_kwargs)
        self.__evsSum_sql_stmnts       = OutagePredictor.general_copy(outg_pred.__evsSum_sql_stmnts)
        self.__evsSum_df               = OutagePredictor.general_copy(outg_pred.__evsSum_df)
        self.__zero_counts_evsSum_df   = OutagePredictor.general_copy(outg_pred.__zero_counts_evsSum_df)
        self.__zc_xfmrs_by_date        = OutagePredictor.general_copy(outg_pred.__zc_xfmrs_by_date)
        #--------------------------------------------------
        self.__regex_setup_df          = OutagePredictor.general_copy(outg_pred.__regex_setup_df)
        self.__cr_trans_dict           = OutagePredictor.general_copy(outg_pred.__cr_trans_dict)
        #-------------------------
        self.rcpx_df                   = OutagePredictor.general_copy(outg_pred.rcpx_df)
        self.is_norm                   = OutagePredictor.general_copy(outg_pred.is_norm)
        #--------------------------------------------------
        self.merge_eemsp               = OutagePredictor.general_copy(outg_pred.merge_eemsp)
        self.eemsp_mult_strategy       = OutagePredictor.general_copy(outg_pred.eemsp_mult_strategy)
        self.eemsp_df                  = OutagePredictor.general_copy(outg_pred.eemsp_df)
        #--------------------------------------------------
        self.__model_dir               = OutagePredictor.general_copy(outg_pred.__model_dir)
        self.model_summary_dict        = OutagePredictor.general_copy(outg_pred.model_summary_dict)
        self.data_structure_df         = OutagePredictor.general_copy(outg_pred.data_structure_df)
        self.model_clf                 = OutagePredictor.general_copy(outg_pred.model_clf)
        self.scale_data                = OutagePredictor.general_copy(outg_pred.scale_data)
        self.scaler                    = OutagePredictor.general_copy(outg_pred.scaler)
        self.eemsp_enc                 = OutagePredictor.general_copy(outg_pred.eemsp_enc)
        self.include_month             = OutagePredictor.general_copy(outg_pred.include_month)
        #-----
        self.model_by_zip_clusters     = OutagePredictor.general_copy(outg_pred.model_by_zip_clusters)
        self.zip_clstrs_ledger         = OutagePredictor.general_copy(outg_pred.zip_clstrs_ledger)
        self.xfmrs_ledger              = OutagePredictor.general_copy(outg_pred.xfmrs_ledger)
        #--------------------------------------------------
        self.__X_test                  = OutagePredictor.general_copy(outg_pred.__X_test)
        self.__y_pred                  = OutagePredictor.general_copy(outg_pred.__y_pred)
        self.__y_pred_df               = OutagePredictor.general_copy(outg_pred.__y_pred_df)


    def copy(self):
        r"""
        """
        #-------------------------
        return_outg_pred = OutagePredictor(outg_pred=self)
        return return_outg_pred

    #----------------------------------------------------------------------------------------------------
    def load(
        self, 
        file_path
    ):
        r"""
        """
        #-------------------------
        f = open(file_path, 'rb')
        tmp_dict = cPickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 

    
    def load_min_memory(
        self, 
        file_path  , 
        keep_attrs = None
    ):
        r"""
        """
        #--------------------------------------------------
        dflt_keep_attrs = [
            # '_OutagePredictor__conn_aws',
            # '_OutagePredictor__conn_dovs',
            # '_OutagePredictor__conn_eems',
            '_OutagePredictor__trsf_pole_nbs',
            'trsf_pole_nbs_sql',
            '_OutagePredictor__trsf_pole_nbs_found',
            '_OutagePredictor__trsf_pole_zips_df',
            'prediction_date',
            'idk_name_1',
            'idk_name_2',
            'date_range',
            'daq_date_ranges',
            'opco', 
            'states', 
            'cities', 
            '_OutagePredictor__evsSum_sql_fcn',
            '_OutagePredictor__evsSum_sql_kwargs',
            '_OutagePredictor__evsSum_sql_stmnts',
            # '_OutagePredictor__evsSum_df',
            # '_OutagePredictor__zero_counts_evsSum_df',
            # '_OutagePredictor__zc_xfmrs_by_date',
            # '_OutagePredictor__regex_setup_df',
            # '_OutagePredictor__cr_trans_dict',
            'rcpx_df',
            'is_norm',
            'merge_eemsp',
            'eemsp_mult_strategy',
            'eemsp_df',
            '_OutagePredictor__model_dir',
            'model_summary_dict',
            'data_structure_df',
            'model_clf',
            'scale_data',
            'scaler',
            'eemsp_enc',
            'include_month',
            'model_by_zip_clusters', 
            'zip_clstrs_ledger', 
            'xfmrs_ledger', 
            '_OutagePredictor__X_test',
            '_OutagePredictor__y_pred',
            '_OutagePredictor__y_pred_df'
        ]
        #--------------------------------------------------
        if keep_attrs is None:
            keep_attrs = dflt_keep_attrs
        assert(isinstance(keep_attrs, list))
        #--------------------------------------------------
        f = open(file_path, 'rb')
        tmp_dict = cPickle.load(f)
        f.close()
        #-----
        # Simply make all non-kept attributes None (as opposed to excluding)
        tmp_dict = {k:v if k in keep_attrs else None for k,v in tmp_dict.items()}
        #--------------------------------------------------
        self.__dict__.update(tmp_dict)


    def save_min_memory(
        self, 
        file_path  , 
        keep_attrs = None
    ):
        r"""
        """
        #--------------------------------------------------
        dflt_keep_attrs = [
            # '_OutagePredictor__conn_aws',
            # '_OutagePredictor__conn_dovs',
            # '_OutagePredictor__conn_eems',
            '_OutagePredictor__trsf_pole_nbs',
            'trsf_pole_nbs_sql',
            '_OutagePredictor__trsf_pole_nbs_found',
            '_OutagePredictor__trsf_pole_zips_df',
            'prediction_date',
            'idk_name_1',
            'idk_name_2',
            'date_range',
            'daq_date_ranges',
            'opco', 
            'states', 
            'cities', 
            '_OutagePredictor__evsSum_sql_fcn',
            '_OutagePredictor__evsSum_sql_kwargs',
            '_OutagePredictor__evsSum_sql_stmnts',
            # '_OutagePredictor__evsSum_df',
            # '_OutagePredictor__zero_counts_evsSum_df',
            # '_OutagePredictor__zc_xfmrs_by_date',
            # '_OutagePredictor__regex_setup_df',
            # '_OutagePredictor__cr_trans_dict',
            'rcpx_df',
            'is_norm',
            'merge_eemsp',
            'eemsp_mult_strategy',
            'eemsp_df',
            '_OutagePredictor__model_dir',
            'model_summary_dict',
            'data_structure_df',
            'model_clf',
            'scale_data',
            'scaler',
            'eemsp_enc',
            'include_month',
            'model_by_zip_clusters', 
            'zip_clstrs_ledger', 
            'xfmrs_ledger', 
            '_OutagePredictor__X_test',
            '_OutagePredictor__y_pred',
            '_OutagePredictor__y_pred_df'
        ]
        #--------------------------------------------------
        if keep_attrs is None:
            keep_attrs = dflt_keep_attrs
        assert(isinstance(keep_attrs, list))
        assert('_OutagePredictor__conn_aws'  not in keep_attrs) # cannot pickle 'pyodbc.Connection'!
        assert('_OutagePredictor__conn_dovs' not in keep_attrs) # cannot pickle 'pyodbc.Connection'!
        assert('_OutagePredictor__conn_eems' not in keep_attrs) # cannot pickle 'pyodbc.Connection'!
        #--------------------------------------------------
        dump_dict = {k:v if k in keep_attrs else None for k,v in self.__dict__.items()}
        #-------------------------
        f = open(file_path, 'wb')
        cPickle.dump(dump_dict, f)
        f.close()


    def save(
        self, 
        file_path    , 
        min_mem_save = False, 
        keep_attrs   = None
    ):
        r"""
        cannot pickle 'pyodbc.Connection' object ==> must set self.__conn_aws to None
        """
        #-------------------------
        if min_mem_save:
            self.save_min_memory(
                file_path  = file_path, 
                keep_attrs = keep_attrs
            )
            return
        #-------------------------
        self.__conn_aws  = None
        self.__conn_dovs = None
        self.__conn_eems = None
        #-------------------------
        f = open(file_path, 'wb')
        cPickle.dump(self.__dict__, f)
        f.close()



    #----------------------------------------------------------------------------------------------------
    
    
    def set_trsf_pole_nbs(
        self, 
        trsf_pole_nbs
    ):
        assert(isinstance(trsf_pole_nbs, list))
        self.__trsf_pole_nbs = trsf_pole_nbs
        
    def set_trsf_pole_nbs_from_sql(
        self, 
        n_trsf_pole_nbs = None, 
        **kwargs
    ):
        r"""
        Run SQL query using kwargs to gather trsf_pole_nbs and set memeber attribute.

        n_trsf_pole_nbs:
            If set, this will randomly return n_trsf_pole_nbs instead of the full set
        """
        #-------------------------
        trsf_pole_nbs_df, trsf_sql = OutagePredictor.get_distinct_trsf_pole_nbs_df(
            conn_aws=self.conn_aws, 
            n_trsf_pole_nbs=n_trsf_pole_nbs, 
            return_sql=True, 
            **kwargs
        )
        trsf_pole_nbs = trsf_pole_nbs_df['trsf_pole_nb'].tolist()
        self.set_trsf_pole_nbs(trsf_pole_nbs)
        self.trsf_pole_nbs_sql = trsf_sql
        
    def set_date_range(
        self
    ):
        r"""
        """
        #-------------------------
        if not isinstance(self.prediction_date, datetime.datetime):
            self.prediction_date = pd.to_datetime(self.prediction_date)
        if not isinstance(self.idk_name_1, pd.Timedelta):
            self.idk_name_1 = pd.Timedelta(self.idk_name_1)
        if not isinstance(self.idk_name_2, pd.Timedelta):
            self.idk_name_2 = pd.Timedelta(self.idk_name_2)
        #-------------------------
        self.date_range = [
            (self.prediction_date-self.idk_name_1).date(), 
            (self.prediction_date-self.idk_name_2).date()
        ]

        
    def set_model_dir(
        self                  , 
        model_dir             , 
        model_by_zip_clusters = False, 
        **kwargs
    ):
        r"""
        Sets the model_dir and extracts the needed contents:
            - model_clf
            - data_structure_df
            - scaler
            - eemsp_encoder
        """
        #-------------------------
        assert(os.path.exists(model_dir))
        self.__model_dir           = model_dir
        #-------------------------
        self.model_by_zip_clusters = model_by_zip_clusters
        #-----
        if self.model_by_zip_clusters:
            assert(os.path.exists(os.path.join(self.model_dir, 'Slices', 'ledger.json')))
        #-------------------------
        model_summary_dict_fname   = kwargs.get('model_summary_dict_fname', 'summary_dict.json')
        data_structure_fname       = kwargs.get('data_structure_fname', 'data_structure_df.pkl')
        model_fname                = kwargs.get('model_fname', 'model_clf.joblib')
        scaler_fname               = kwargs.get('scaler_fname', 'scaler.joblib')
        eemsp_encoder_fname        = kwargs.get('eemsp_encoder_fname', 'eemsp_encoder.joblib')
        #-------------------------
        # model_summary_dict, data_structure_df, model_clf MUST be present
        assert(os.path.exists(os.path.join(model_dir, model_summary_dict_fname)))
        tmp_f = open(os.path.join(model_dir, model_summary_dict_fname))
        self.model_summary_dict = json.load(tmp_f)
        tmp_f.close()
        #-----
        assert(os.path.exists(os.path.join(model_dir, data_structure_fname)))
        self.data_structure_df = pd.read_pickle(os.path.join(model_dir, data_structure_fname))
        #-----
        assert(os.path.exists(os.path.join(model_dir, model_fname)))
        self.model_clf = joblib.load(os.path.join(model_dir, model_fname))
        #-------------------------
        self.scale_data = self.model_summary_dict['run_scaler']
        if self.scale_data:
            assert(os.path.exists(os.path.join(model_dir, scaler_fname)))
            self.scaler = joblib.load(os.path.join(model_dir, scaler_fname))
        #-----
        self.merge_eemsp = self.model_summary_dict['merge_eemsp']
        if self.merge_eemsp:
            assert(os.path.exists(os.path.join(model_dir, eemsp_encoder_fname)))
            self.eemsp_enc = joblib.load(os.path.join(model_dir, eemsp_encoder_fname))
            self.eemsp_mult_strategy = self.model_summary_dict['eemsp_mult_strategy']
        #-----
        self.include_month = self.model_summary_dict['include_month']

    @staticmethod
    def get_cr_cols(
        df, 
        regex_pattern = r'cr\d*'
    ):
        r"""
        """
        #-------------------------
        #-----
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = df.columns.tolist(), 
            regex_pattern = regex_pattern, 
            ignore_case   = False
        )
        #-------------------------
        return cr_cols
    

    @staticmethod
    def build_ledger_df(
        ledger         , 
        model_idfr_col = 'model_idfr', 
        zip_col        = 'zip'
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(ledger, dict))
        ledger_df = pd.Series(ledger).to_frame(name=model_idfr_col).reset_index(names=zip_col)
        return ledger_df
    

    @staticmethod
    def build_ledger_inv(
        ledger        , 
        return_type   = pd.Series
    ):
        r"""
        Input:
            keys   = zip_codes
            values = model identifiers
    
        Output:
            return_type == pd.Series
                a pd.Series object with index equal to model identifier and values equal to lists of zip_codes
            return_type == dict
                keys   = model identifiers
                values = lists of zip_codes covered by model
        """
        #-------------------------
        assert(return_type in [pd.Series, dict])
        #-------------------------
        ledger_srs = pd.Series(ledger)
        #-------------------------
        # Invert the series, so indices are model ids and values are zip_codes
        # AND group by model id and collect zip_codes in lists
        ledger_inv = pd.Series(ledger_srs.index.values, index=ledger_srs).groupby(level=0).apply(list)
        #-------------------------
        if return_type==dict:
            ledger_inv = ledger_inv.to_dict()
        #-------------------------
        return ledger_inv
    

    @staticmethod
    def build_ledger_inv_with_trsf_pole_nbs(
        ledger            , 
        trsf_pole_zips_df , 
        trsf_pole_nb_col  = 'trsf_pole_nb', 
        zip_col           = 'zip', 
        assert_all_found  = True, 
        return_type       = pd.Series
    ):
        r"""
        assert_all_found:
            If True, asserts that all zip codes found in trsf_pole_zips_df are also contained in ledger
        """
        #--------------------------------------------------
        assert(return_type in [pd.Series, dict])
        #-------------------------
        nec_cols = [trsf_pole_nb_col, zip_col]
        assert(set(nec_cols).difference(set(trsf_pole_zips_df.columns.tolist()))==set())
        #-------------------------
        ledger_df = OutagePredictor.build_ledger_df(
            ledger = ledger, 
            model_idfr_col = 'model_idfr', 
            zip_col        = zip_col
        )
        #-----
        if assert_all_found:
            assert(set(trsf_pole_zips_df[zip_col].unique()).difference(set(ledger_df[zip_col].unique()))==set())
        #-------------------------
        ledger_inv = pd.merge(
            trsf_pole_zips_df[[trsf_pole_nb_col, zip_col]], 
            ledger_df, 
            how      = 'left', 
            left_on  = zip_col, 
            right_on = zip_col
        )
        #-----
        ledger_inv = ledger_inv.groupby('model_idfr')[trsf_pole_nb_col].apply(lambda x: list(np.unique(x)))
        #-------------------------
        if return_type==dict:
            ledger_inv = ledger_inv.to_dict()
        #-------------------------
        return ledger_inv
        
        
    @staticmethod
    def build_sql_xfmr_meter_cnt_closest_to_date(
        trsf_pole_nbs      , 
        date               , 
        opco               = None, 
        sql_kwargs         = None, 
        include_meter_cols = False
    ):
        r"""
        This function was built specifically to be used with the OutagePredictor class (and also to be used in the data acquisition
          portion of the baseline formation).
        For these data grabs, any transformer whose meters did not register any events will not be retrieved by the query.
        However, these zero event transformers should really be included in the baseline, and must be included in the OutagePredictor
          process otherwise no predictions can be made for these (the expectation is that the model should return False, i.e., predict no outage
          for transformers suffering zero events in a given time period, but this will be interesting, and possibly enlightening, to check)

        include_meter_cols:
            If True, ['serialnumber', 'aep_premise_nb'] will be included.
            However, these don't have much meaning, as the query finds the entry for each trsf_pole_nb with entry closest to date
            So, ['serialnumber', 'aep_premise_nb'] represent the entry from which this info was grabbed, but to returned info is for the trsf_pole_nb as a wholes
    
        The SQL statement returned from this function will be similar to the following:
            WITH cte AS
            	(
            	SELECT trsf_pole_nb, xf_meter_cnt, aep_opco, aep_event_dt, ABS(DATE_DIFF('day', CAST('2023-07-26' AS TIMESTAMP), CAST(aep_event_dt AS TIMESTAMP))) diff, 
            		ROW_NUMBER() OVER (PARTITION BY trsf_pole_nb ORDER BY ABS(DATE_DIFF('day', CAST('2023-07-26' AS TIMESTAMP), CAST(aep_event_dt AS TIMESTAMP)))) AS SEQUENCE
            	FROM meter_events.events_summary_vw EDE
            	WHERE EDE.aep_opco = 'oh'
            	AND   EDE.trsf_pole_nb IN ('40810546000051','40810546A30011', .....)
            	)
            
            SELECT *
            FROM cte
            WHERE SEQUENCE = 1
        """
        #--------------------------------------------------
        dflt_sql_kwargs = dict(
            cols_of_interest = ['trsf_pole_nb', 'xf_meter_cnt', 'aep_opco', 'aep_event_dt'], 
            trsf_pole_nbs    = trsf_pole_nbs, 
            opco             = opco, 
        )
        #-------------------------
        if sql_kwargs is None:
            sql_kwargs = dict()
        assert(isinstance(sql_kwargs, dict))
        #-------------------------
        sql_kwargs = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = sql_kwargs, 
            default_values_dict = dflt_sql_kwargs, 
            extend_any_lists    = False, 
            inplace             = False
        )
        #-----
        # Make 100% sure trsf_pole_nbs properly set
        sql_kwargs['trsf_pole_nbs'] = trsf_pole_nbs
        sql_kwargs['opco']          = opco
        #--------------------------------------------------
        cols_of_interest = sql_kwargs['cols_of_interest']
        if include_meter_cols:
            meter_cols = ['serialnumber', 'aep_premise_nb']
            cols_of_interest = cols_of_interest + list(set(meter_cols).difference(set(cols_of_interest)))
        #-----
        cols_of_interest = cols_of_interest + [
            dict(
                field_desc = f"ABS(DATE_DIFF('day', CAST('{date}' AS TIMESTAMP), CAST(aep_event_dt AS TIMESTAMP)))", 
                alias      = 'diff', 
                table_alias_prefix = None
            ), 
            dict(
                field_desc = f"ROW_NUMBER() OVER (PARTITION BY trsf_pole_nb ORDER BY ABS(DATE_DIFF('day', CAST('{date}' AS TIMESTAMP), CAST(aep_event_dt AS TIMESTAMP))))", 
                alias      = 'SEQUENCE', 
                table_alias_prefix = None
            ), 
        ]
        #-------------------------
        sql_select = SQLSelect(cols_of_interest, global_table_alias_prefix=None)
        #--------------------------------------------------
        sql_from = SQLFrom('meter_events', 'events_summary_vw', alias=None)
        #--------------------------------------------------
        sql_where = SQLWhere()
        sql_where = AMI_SQL.add_ami_where_statements(sql_where, **sql_kwargs)
        #--------------------------------------------------
        sql = SQLQuery(
            sql_select  = sql_select, 
            sql_from    = sql_from, 
            sql_where   = sql_where, 
            sql_groupby = None, 
            alias       = None
        )
        #--------------------------------------------------
        sql_stmnt = f"WITH cte as(\n{sql.get_sql_statement(insert_n_tabs_to_each_line=1, include_alias=False)}\n)\nSELECT *\nFROM cte\nWHERE SEQUENCE=1"
        #--------------------------------------------------
        return sql_stmnt
    
    @staticmethod
    def get_xfmr_meter_cnt_closest_to_date(
        trsf_pole_nbs, 
        date, 
        opco               = None, 
        sql_kwargs         = None, 
        include_meter_cols = False, 
        batch_size         = 1000, 
        verbose            = True, 
        n_update           = 10, 
        conn_aws           = None, 
        return_sql         = False,
        keep_extra_cols    = False
    ):
        r"""
        !!!!! I suggest you use MeterPremise.get_xfmr_meter_cnt instead !!!!!
        However, as meter_events.events_summary_vw continues to grow, maybe this will be better?
        NOT USED ANYWHERE CURRENTLY

        This function was built specifically to be used with the OutagePredictor class (and also to be used in the data acquisition
          portion of the baseline formation).
        For these data grabs, any transformer whose meters did not register any events will not be retrieved by the query.
        However, these zero event transformers should really be included in the baseline, and must be included in the OutagePredictor
          process otherwise no predictions can be made for these (the expectation is that the model should return False, i.e., predict no outage
          for transformers suffering zero events in a given time period, but this will be interesting, and possibly enlightening, to check)

        include_meter_cols:
            If True, ['serialnumber', 'aep_premise_nb'] will be included.
            However, these don't have much meaning, as the query finds the entry for each trsf_pole_nb with entry closest to date
            So, ['serialnumber', 'aep_premise_nb'] represent the entry from which this info was grabbed, but to returned info is for the trsf_pole_nb as a wholes

        Returns pd.DataFrame (and possibly sql_stmnts) with following columns:
            ['trsf_pole_nb', 'xf_meter_cnt', 'aep_opco', 'aep_event_dt']
        If include_meter_cols==True, ['serialnumber', 'aep_premise_nb'] are also returned
        """
        #--------------------------------------------------
        if conn_aws is None:
            conn_aws = Utilities.get_athena_prod_aws_connection()
        #--------------------------------------------------
        batch_idxs = Utilities.get_batch_idx_pairs(
            n_total              = len(trsf_pole_nbs), 
            batch_size           = batch_size, 
            absorb_last_pair_pct = None
        )
        n_batches = len(batch_idxs)
        #--------------------------------------------------
        return_df = pd.DataFrame()
        #--------------------------------------------------
        if verbose:
            print(f'n_coll = {len(trsf_pole_nbs)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches = {n_batches}')
        #--------------------------------------------------
        sql_stmnts = []
        for i, batch_i in enumerate(batch_idxs):
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-----
            sql_i = OutagePredictor.build_sql_xfmr_meter_cnt_closest_to_date(
                trsf_pole_nbs      = trsf_pole_nbs[i_beg:i_end], 
                date               = date, 
                opco               = opco, 
                sql_kwargs         = sql_kwargs, 
                include_meter_cols = include_meter_cols
            )
            assert(isinstance(sql_i, str))
            #-------------------------
            # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
            # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_i = pd.read_sql_query(sql_i, conn_aws)
            #-------------------------
            sql_stmnts.append(sql_i)
            #-----
            if return_df.shape[0]>0:
                assert(all(df_i.columns==return_df.columns))
            return_df = pd.concat([return_df, df_i], axis=0, ignore_index=False)
        #--------------------------------------------------
        if not keep_extra_cols:
            return_df = return_df.drop(columns=['diff', "SEQUENCE"])
        #--------------------------------------------------
        if return_sql:
            return return_df, sql_stmnts
        else:
            return return_df
        

    @staticmethod
    def build_zero_counts_evsSum_df_for_date_range(
        trsf_pole_nbs           , 
        evsSum_df               , 
        date_range              , 
        opco                    = None, 
        missing_mtr_cnt_strtgy  = 'mean', 
        sql_kwargs              = None, 
        date_col                = 'aep_event_dt',
        trsf_pole_nb_col        = 'trsf_pole_nb', 
        SN_col                  = 'serialnumber', 
        PN_col                  = 'aep_premise_nb', 
        xf_meter_cnt_col        = 'xf_meter_cnt', 
        events_tot_col          = 'events_tot', 
        opco_col                = 'aep_opco', 
        conn_aws                = None, 
        return_zc_xfmrs_by_date = False, 
        verbose                 = True    
    ):
        r"""
        Build zero_counts_evsSums_df for a range of dates.
        Quick version because for any transformers needing to be included in zero counts:
            1. Try to grab xf_meter_cnt from evsSum_df
            2. Query for rest using MeterPremise.get_xfmr_meter_cnt
        """
        #----------------------------------------------------------------------------------------------------
        assert(missing_mtr_cnt_strtgy is None or missing_mtr_cnt_strtgy in ['mean', 'median', 'bfill', 'ffill'])
        #--------------------------------------------------
        # Make sure date_col is datetime object
        evsSum_df[date_col] = pd.to_datetime(evsSum_df[date_col])
        
        #---------------------------------------------------------------------------
        # First, form zc_xfmrs_by_date, which is a dictionary containing the transformers which need to be included
        #   in zero-counts for each date.
        # keys are dates, values are lists of trsf_pole_nbs
        #-----
        zc_xfmrs_by_date = dict()
        dates = pd.date_range(start=date_range[0], end=date_range[1])
        for date_i in dates:
            evsSum_df_i         = evsSum_df[evsSum_df[date_col]==date_i]
            zero_counts_xfmrs_i = list(set(trsf_pole_nbs).difference(set(evsSum_df_i[trsf_pole_nb_col].unique())))
            if len(zero_counts_xfmrs_i) > 0:
                assert(date_i not in zc_xfmrs_by_date.keys())
                zc_xfmrs_by_date[date_i] = zero_counts_xfmrs_i
        
        #--------------------------------------------------
        # Melt down zc_xfmrs_by_date.values() to grab a list of all trsf_pole_nbs which will need entries (we'll handle dates later)
        # Also, figure out which xfmrs we can use evsSums_df to grab meter counts (xfmr_meter_cnts_from_evsSum_df) 
        #   and for which we will need to query (zc_xfmrs_to_query)
        #-------------------------
        zc_xfmrs_needed = Utilities.melt_list_of_lists_and_return_unique(
            lol  = list(zc_xfmrs_by_date.values()), 
            sort = None
        )
        if len(zc_xfmrs_needed)==0:
            return pd.DataFrame()
        #-------------------------
        # If we can find meter counts in evsSum_df, simply use those instead of querying to save time
        xfmr_meter_cnts_from_evsSum_df = evsSum_df.groupby(trsf_pole_nb_col)[xf_meter_cnt_col].agg('max')
        xfmr_meter_cnts_from_evsSum_df = xfmr_meter_cnts_from_evsSum_df.dropna()
        #-------------------------
        # The following are needed, but weren't found in evsSum_df
        zc_xfmrs_to_query = list(set(zc_xfmrs_needed).difference(set(xfmr_meter_cnts_from_evsSum_df.index)))
        
        #---------------------------------------------------------------------------
        # Use MeterPremise.get_xfmr_meter_cnt to try to recover meter counts for zc_xfmrs_to_query
        #-------------------------
        if len(zc_xfmrs_to_query) > 0:
            if sql_kwargs is None:
                sql_kwargs = dict()
            assert(isinstance(sql_kwargs, dict))
            sql_kwargs = copy.deepcopy(sql_kwargs) # Possibly popping some things off, so don't want to alter original
            #-----
            # keys_to_ignore are handled by MeterPremise.get_xfmr_meter_cnt_sql_stmnt.
            # They would be ignored over there as well, but I keep them here so the user is explicitly aware
            keys_to_ignore = [
                'cols_of_interest', 
                'trsf_pole_nb',  'trsf_pole_nbs', 
                'serial_number', 'serial_numbers', 'mfr_devc_ser_nbr', 'mfr_devc_ser_nbrs', 
                'premise_nb',    'premise_nbs',    'aep_premise_nb',   'aep_premise_nbs', 
                'groupby_cols'
            ]
            sql_kwargs         = {k:v for k,v in sql_kwargs.items() if k not in keys_to_ignore}
            sql_kwargs['opco'] = opco
            #-------------------------
            # Note: When used by class, this is fed self.evsSum_sql_kwargs, which may contain, e.g., batch_size
            batch_size = sql_kwargs.pop('batch_size', 1000)
            n_update   = sql_kwargs.pop('n_update', 10)
            #-------------------------
            xfmr_meter_cnts_df = MeterPremise.get_xfmr_meter_cnt(
                trsf_pole_nbs = zc_xfmrs_to_query, 
                return_PN_cnt = False, 
                batch_size    = batch_size, 
                verbose       = verbose, 
                n_update      = n_update, 
                conn_aws      = conn_aws, 
                return_sql    = False, 
                **sql_kwargs
            )
            xfmr_meter_cnts_df = xfmr_meter_cnts_df.rename(columns={'xfmr_SN_cnt':xf_meter_cnt_col})
        else:
            xfmr_meter_cnts_df = pd.DataFrame(columns=[trsf_pole_nb_col, xf_meter_cnt_col])

        #---------------------------------------------------------------------------
        # Sometimes cannot find all needed transformers in data.  Not sure why.
        # In any case, still want to include these data, we'll just need to set xf_meter_cnt to NaN
        # This should make it easy to exclude/identify later if wanted (if missing_mtr_cnt_strtgy is None as well!)
        #-------------------------
        missing_xfmrs = list(set(zc_xfmrs_to_query).difference(set(xfmr_meter_cnts_df[trsf_pole_nb_col].unique())))
        #-----
        missing_meter_cnts_df = pd.DataFrame(
            data    = missing_xfmrs, 
            columns = [trsf_pole_nb_col]
        )
        missing_meter_cnts_df[xf_meter_cnt_col] = np.nan
        
        #--------------------------------------------------
        # Convert xfmr_meter_cnts_from_evsSum_df to the form needed.
        xfmr_meter_cnts_from_evsSum_df = xfmr_meter_cnts_from_evsSum_df.to_frame().reset_index()
        
        #--------------------------------------------------
        # Ensure there is no overlap in trsf_pole_nb, and expected columns
        assert(set(xfmr_meter_cnts_df[trsf_pole_nb_col]).intersection(            set(xfmr_meter_cnts_from_evsSum_df[trsf_pole_nb_col]))==set())
        assert(set(xfmr_meter_cnts_df[trsf_pole_nb_col]).intersection(            set(missing_meter_cnts_df[trsf_pole_nb_col]))==set())
        assert(set(xfmr_meter_cnts_from_evsSum_df[trsf_pole_nb_col]).intersection(set(missing_meter_cnts_df[trsf_pole_nb_col]))==set())
        #-----
        assert(set(xfmr_meter_cnts_df.columns).difference(set())==set(xfmr_meter_cnts_from_evsSum_df.columns))
        assert(set(xfmr_meter_cnts_df.columns).difference(set())==set(missing_meter_cnts_df.columns))
        
        #--------------------------------------------------
        # Concat the three dfs, which will be used to populate the full zero_counts_df
        xfmr_meter_cnts_df = Utilities_df.concat_dfs(
            dfs                  = [
                xfmr_meter_cnts_df, 
                xfmr_meter_cnts_from_evsSum_df, 
                missing_meter_cnts_df
            ], 
            axis                 = 0, 
            make_col_types_equal = False
        )
        #-------------------------
        if evsSum_df[opco_col].nunique()==1 and opco is None:
            xfmr_meter_cnts_df[opco_col] = evsSum_df[opco_col].unique().tolist()[0]
        else:
            xfmr_meter_cnts_df[opco_col] = opco
        #-------------------------
        if missing_mtr_cnt_strtgy is not None:
            if missing_mtr_cnt_strtgy=='mean':
                xfmr_meter_cnts_df[xf_meter_cnt_col] = xfmr_meter_cnts_df[xf_meter_cnt_col].fillna(np.round(xfmr_meter_cnts_df[xf_meter_cnt_col].mean()).astype(int))
            elif missing_mtr_cnt_strtgy=='median':
                xfmr_meter_cnts_df[xf_meter_cnt_col] = xfmr_meter_cnts_df[xf_meter_cnt_col].fillna(np.round(xfmr_meter_cnts_df[xf_meter_cnt_col].median()).astype(int))
            else:
                xfmr_meter_cnts_df[xf_meter_cnt_col] = xfmr_meter_cnts_df[xf_meter_cnt_col].fillna(method = missing_mtr_cnt_strtgy)
        #---------------------------------------------------------------------------
        # Iterate through zc_xfmrs_by_date, grab the entries needed from xfmr_meter_cnts_df, set appropriate date, and add to collection (zc_dfs)
        #-------------------------
        # Set index to trsf_pole_nb so I can use .loc below
        xfmr_meter_cnts_df = xfmr_meter_cnts_df.set_index(trsf_pole_nb_col)
        #-------------------------
        zc_dfs = []
        for date_i, zc_xfmrs_i in zc_xfmrs_by_date.items():
            zc_df_date_i           = xfmr_meter_cnts_df.loc[zc_xfmrs_i].reset_index()
            zc_df_date_i[date_col] = date_i
            zc_dfs.append(zc_df_date_i)
        #-------------------------
        zero_counts_df = Utilities_df.concat_dfs(
            dfs                  = zc_dfs, 
            axis                 = 0, 
            make_col_types_equal = False
        )
        zero_counts_df = zero_counts_df.reset_index(drop=True)
        #--------------------------------------------------
        # Set remaining columns
        #-------------------------
        zero_counts_df[SN_col] = 'SN_for_'+zero_counts_df[trsf_pole_nb_col]
        zero_counts_df[PN_col] = 'PN_for_'+zero_counts_df[trsf_pole_nb_col]
        #-----
        zero_counts_df[events_tot_col]   = 0
        #-------------------------
        cr_cols    = OutagePredictor.get_cr_cols(df=evsSum_df)
        og_shape_0 = zero_counts_df.shape[0]
        addtln_zero_counts_df_cols = pd.DataFrame(
            data    = 0, 
            index   = zero_counts_df.index, 
            columns = cr_cols
        )
        #-----
        zero_counts_df = pd.merge(
            zero_counts_df, 
            addtln_zero_counts_df_cols, 
            left_index  = True, 
            right_index = True, 
            how         = 'inner'
        )
        assert(zero_counts_df.shape[0]==og_shape_0)
        
        #--------------------------------------------------
        # Finish
        #-------------------------
        assert(set(zero_counts_df.columns).symmetric_difference(set(evsSum_df.columns))==set())
        zero_counts_df = zero_counts_df[evsSum_df.columns]
        if Utilities_df.is_df_index_simple_v2(df=evsSum_df):
            zero_counts_df = zero_counts_df.set_index(pd.Index(range(evsSum_df.index[-1]+1, evsSum_df.index[-1]+1+zero_counts_df.shape[0])))
        #--------------------------------------------------
        if return_zc_xfmrs_by_date:
            return zero_counts_df, zc_xfmrs_by_date
        else:
            return zero_counts_df
        

    @staticmethod
    def build_zero_counts_evsSum_df_for_date_ranges(
        trsf_pole_nbs           , 
        evsSum_df               , 
        daq_date_ranges         , 
        opco                    = None, 
        sql_kwargs              = None, 
        date_col                = 'aep_event_dt',
        trsf_pole_nb_col        = 'trsf_pole_nb', 
        SN_col                  = 'serialnumber', 
        PN_col                  = 'aep_premise_nb', 
        xf_meter_cnt_col        = 'xf_meter_cnt', 
        events_tot_col          = 'events_tot', 
        opco_col                = 'aep_opco', 
        conn_aws                = None, 
        return_zc_xfmrs_by_date = False, 
        verbose                 = True    
    ):
        r"""
        Build zero_counts_evsSums_df for a list of date ranges.
        Quick version because for any transformers needing to be included in zero counts:
            1. Try to grab xf_meter_cnt from evsSum_df
            2. Query for rest using MeterPremise.get_xfmr_meter_cnt
        """
        #----------------------------------------------------------------------------------------------------
        assert(Utilities.is_object_one_of_types(daq_date_ranges, [list, tuple]))
        assert(Utilities.is_list_nested(lst=daq_date_ranges, enforce_if_one_all=True))
        assert(Utilities.are_list_elements_lengths_homogeneous(lst=daq_date_ranges, length=2))
        #--------------------------------------------------
        zc_xfmrs_by_date = dict()
        zero_counts_df = pd.DataFrame()
        for date_range_i in daq_date_ranges:
            zero_counts_evsSum_df_i, zc_xfmrs_by_date_i = OutagePredictor.build_zero_counts_evsSum_df_for_date_range(
                trsf_pole_nbs           = trsf_pole_nbs, 
                evsSum_df               = evsSum_df, 
                date_range              = date_range_i, 
                opco                    = opco, 
                sql_kwargs              = sql_kwargs, 
                date_col                = date_col,
                trsf_pole_nb_col        = trsf_pole_nb_col, 
                SN_col                  = SN_col, 
                PN_col                  = PN_col, 
                xf_meter_cnt_col        = xf_meter_cnt_col, 
                events_tot_col          = events_tot_col, 
                opco_col                = opco_col, 
                conn_aws                = conn_aws, 
                return_zc_xfmrs_by_date = True, 
                verbose                 = verbose    
            )
            #-------------------------
            if zero_counts_df.shape[0]>0:
                assert(
                    set(zero_counts_evsSum_df_i[['trsf_pole_nb', 'aep_event_dt']].value_counts().index.tolist()).intersection(
                    set(zero_counts_df[['trsf_pole_nb', 'aep_event_dt']].value_counts().index.tolist())
                    )==set()
                )
            assert(set(zc_xfmrs_by_date_i.keys()).intersection(zc_xfmrs_by_date.keys())==set())
            #-------------------------
            zero_counts_df = Utilities_df.concat_dfs(
                dfs                  = [zero_counts_df, zero_counts_evsSum_df_i], 
                axis                 = 0, 
                make_col_types_equal = False
            )
            zc_xfmrs_by_date = zc_xfmrs_by_date|zc_xfmrs_by_date_i
        #--------------------------------------------------
        if return_zc_xfmrs_by_date:
            return zero_counts_df, zc_xfmrs_by_date
        else:
            return zero_counts_df



    @staticmethod
    def determine_unique_date_ranges_needed_for_daq(
        new_date_ranges      , 
        existing_date_ranges , 
    ):
        r"""
        First, determine which unique date ranges, if any, we need to acquire/build data for
          1. Consolidate new_date_ranges using Utilities.get_overlap_intervals (building date_ranges_0)
          2. For each unique date range in date_ranges_0, remove any intervals already contained in existing_date_ranges, 
               leaving only the ranges needed for acquisition
          3. Note: date_ranges_fnl.extend is utilized instead of .append because Utilities.remove_overlaps_from_date_interval was 
               built such that a list of pd.Interval objects is always returned.
        """
        #--------------------------------------------------
        if new_date_ranges is None or len(new_date_ranges)==0:
            return []
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(new_date_ranges, [list, tuple]))
        date_ranges_0 = []
        for date_range_i in new_date_ranges:
            assert(Utilities.is_object_one_of_types(date_range_i, [list, tuple, pd.Interval]))
            if isinstance(date_range_i, pd.Interval):
                date_ranges_0.append([date_range_i.left, date_range_i.right])
            elif Utilities.is_object_one_of_types(date_range_i, [list, tuple]):
                assert(len(date_range_i)==2)
                date_ranges_0.append(list(date_range_i))
            else:
                assert(0)    
        #--------------------------------------------------
        date_ranges_0 = Utilities.get_overlap_intervals(ranges=date_ranges_0)
        #-------------------------
        date_ranges_fnl = []
        for date_range_i in date_ranges_0:
            date_ranges_fnl_i = Utilities.remove_overlaps_from_date_interval(
                intrvl_1  = date_range_i, 
                intrvls_2 = existing_date_ranges, 
                closed    = True
            )
            date_ranges_fnl.extend(date_ranges_fnl_i)
        #--------------------------------------------------
        # If all dates in date_ranges are already accounted for in self.daq_date_ranges, simply return
        if len(date_ranges_fnl)==0:
            return []
        #--------------------------------------------------
        date_ranges_fnl = Utilities.get_overlap_intervals(ranges=date_ranges_fnl)
        #--------------------------------------------------
        return date_ranges_fnl


    def buildextend_eemsp_df(
        self                   , 
        daq_date_ranges        , 
        addtnl_kwargs          = None, 
        include_n_eemsp        = True, 
        cols_of_interest_eemsp = None, 
        numeric_cols           = ['kva_size'], 
        dt_cols                = ['install_dt', 'removal_dt'], 
        ignore_cols            = ['serial_nb'], 
        eemsp_cols_to_drop     = ['latest_status', 'serial_nb'], 
        batch_size             = 10000, 
        verbose                = True, 
        n_update               = 10, 
    ):
        r"""
        Build and/or extend eemsp_df
        If new DAQ is needed, grab the data and append to self.eemsp_df
        """
        #----------------------------------------------------------------------------------------------------
        # If all dates in date_ranges are already accounted for, simply return
        if len(daq_date_ranges)==0:
            return
        #----------------------------------------------------------------------------------------------------
        # DAQ and append new results to self.eemsp_df
        #--------------------------------------------------
        eemsp_df = OutagePredictor.build_eemsp_df(
            trsf_pole_nbs          = self.trsf_pole_nbs, 
            date_range             = daq_date_ranges, 
            addtnl_kwargs          = addtnl_kwargs, 
            conn_aws               = self.conn_aws, 
            mult_strategy          = self.eemsp_mult_strategy, 
            include_n_eemsp        = include_n_eemsp, 
            cols_of_interest_eemsp = cols_of_interest_eemsp, 
            numeric_cols           = numeric_cols, 
            dt_cols                = dt_cols, 
            ignore_cols            = ignore_cols, 
            eemsp_cols_to_drop     = eemsp_cols_to_drop, 
            batch_size             = batch_size, 
            verbose                = verbose, 
            n_update               = n_update, 
        )
        #--------------------------------------------------
        if self.eemsp_df is None or self.eemsp_df.shape[0]==0:
            self.eemsp_df = eemsp_df
        else:
            # Append eemsp_df to self.eemsp_df
            #   But make temporary eemsp_df, instead of setting directly to self.eemsp_df so if
            #   any error occurs in concat then original data are not lost
            eemsp_df = OutageMdlrPrep.append_to_evsSum_df(
                evsSum_df        = self.eemsp_df, 
                addtnl_evsSum_df = eemsp_df, 
                sort_by          = None
            )
            self.eemsp_df = eemsp_df.copy()


    @staticmethod
    def get_date_range_for_prediction(
        prediction_date , 
        idk_name_1      , 
        idk_name_2      , 
    ):
        r"""
        """
        #-------------------------
        if not isinstance(prediction_date, datetime.datetime):
            prediction_date = pd.to_datetime(prediction_date)
        if not isinstance(idk_name_1, pd.Timedelta):
            idk_name_1 = pd.Timedelta(idk_name_1)
        if not isinstance(idk_name_2, pd.Timedelta):
            idk_name_2 = pd.Timedelta(idk_name_2)
        #-------------------------
        date_range = [
            (prediction_date-idk_name_1).date(), 
            (prediction_date-idk_name_2).date()
        ]
        #-------------------------
        return date_range
        
    @staticmethod
    def get_date_ranges_for_predictions(
        prediction_dates  , 
        predictions_range = None, 
        idk_name_1        = pd.Timedelta('31D'), 
        idk_name_2        = pd.Timedelta('1D')
    ):
        r"""
        Given a list of prediction dates, this will return the date ranges which are needed for DAQ
        Typically, this is the period 1-31 days before a prediction date.
        The returned list will be consolidated.

        prediction_dates:
            A list of prediction dates (or a pd.DatetimeIndex object).
            If a list, the elements should be one of the following types: 
                datetime.date, datetime.datetime, pd.Timestamp, or string

        predictions_range:
            A tuple of length 2, giving the start and end dates of the range.
            The prediction dates will then be taken from pd.date_range with freq=pd.Timedelta('1D')
        """
        #--------------------------------------------------
        assert(prediction_dates is not None or predictions_range is not None)
        #-------------------------
        pred_dates_fnl = []
        #--------------------------------------------------
        if prediction_dates is not None:
            assert(Utilities.is_object_one_of_types(prediction_dates, [list, tuple, pd.DatetimeIndex]))
            if isinstance(prediction_dates, pd.DatetimeIndex):
                prediction_dates = prediction_dates.tolist()
            assert(
                Utilities.are_all_list_elements_one_of_types(
                    prediction_dates, 
                    [datetime.date, datetime.datetime, pd.Timestamp, str]
                )
            )
            #-------------------------
            # NOTE: isinstance(pd.to_datetime('2023-05-02'), datetime.date) = True AND!!!!!!
            #       isinstance(pd.to_datetime('2023-05-02'), datetime.datetime) = True
            # ==> Cannot split datetime.date and [datetime.datetime, pd.Timestamp] below
            # ==> I guess I'll use try/except, because a true datetime.date object will throw an error
            #     if .date() called on it
            for dt_i in prediction_dates:
                if isinstance(dt_i, str):
                    pred_dates_fnl.append(pd.to_datetime(dt_i).date())
                elif Utilities.is_object_one_of_types(dt_i, [datetime.datetime, pd.Timestamp, datetime.date]):
                    try:
                        to_appnd = dt_i.date()
                    except:
                        to_appnd = dt_i
                    pred_dates_fnl.append(to_appnd)
                else:
                    assert(0)
        #--------------------------------------------------
        if predictions_range is not None:
            assert(
                Utilities.is_object_one_of_types(predictions_range, [list, tuple]) and
                len(predictions_range)==2
            )
            #-------------------------
            pred_dates_2 = pd.date_range(
                start = predictions_range[0], 
                end   = predictions_range[1], 
                freq  = pd.Timedelta('1D')
            )
            pred_dates_2 = [x.date() for x in pred_dates_2]
            #-------------------------
            pred_dates_fnl.extend(pred_dates_2)
        #--------------------------------------------------
        assert(len(pred_dates_fnl) > 0)
        # Make sure unique values only:
        pred_dates_fnl = natsorted(list(set(pred_dates_fnl)))
        #--------------------------------------------------
        # Get the date range needed for each prediction date, then consolidate
        #   using Utilities.get_overlap_intervals
        date_ranges = [
            OutagePredictor.get_date_range_for_prediction(
                prediction_date = x, 
                idk_name_1      = idk_name_1, 
                idk_name_2      = idk_name_2      
            ) 
            for x in pred_dates_fnl
        ]
        date_ranges = Utilities.get_overlap_intervals(
            ranges = date_ranges
        )
        date_ranges = natsorted(date_ranges)
        #--------------------------------------------------
        return date_ranges


    def build_raw_data_general(
        self              , 
        prediction_dates  , 
        predictions_range = None, 
        evsSum_sql_fcn    = AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs = None, 
        save_args         = False, 
        verbose           = False
    ):
        r"""
        Builds the raw data, which are 
            1. the needed events summary data (to be added to__evsSum_df)
            2. Data for transformers not registering an event during the period(s) of interest, 
               (to be added to zero_counts_evsSum_df)
            3. The eemsp data, if needed

        prediction_dates:
            A list of prediction dates (or a pd.DatetimeIndex object).
            If a list, the elements should be one of the following types: 
                datetime.date, datetime.datetime, pd.Timestamp, or string
    
        predictions_range:
            A tuple of length 2, giving the start and end dates of the range.
            The prediction dates will then be taken from pd.date_range with freq=pd.Timedelta('1D')
    
        If user supplies any evsSum_sql_kwargs, user SHOULD NOT include:
            - date_range
            - trsf_pole_nbs
        as these will be supplied from self
    
        Function tagged as _general because it will be used by:
            OutagePredictor.build_raw_data (pre-dates this function, and is essentially the guts of this function)
            OutagePredictor.prep_multiple_prediction_dates
        """
        #----------------------------------------------------------------------------------------------------
        assert(prediction_dates is not None or predictions_range is not None)
        #-------------------------
        date_ranges = OutagePredictor.get_date_ranges_for_predictions(
            prediction_dates  = prediction_dates, 
            predictions_range = predictions_range, 
            idk_name_1        = self.idk_name_1, 
            idk_name_2        = self.idk_name_2
        )
        #-----
        if len(date_ranges)==0:
            return
        #----------------------------------------------------------------------------------------------------
        # First, determine which unique date ranges, if any, we need to acquire/build data for
        #--------------------------------------------------
        date_ranges_fnl = OutagePredictor.determine_unique_date_ranges_needed_for_daq(
            new_date_ranges      = date_ranges, 
            existing_date_ranges = self.daq_date_ranges, 
        )
        #--------------------------------------------------
        # If all dates in date_ranges are already accounted for in self.daq_date_ranges, simply return
        if len(date_ranges_fnl)==0:
            return
        #----------------------------------------------------------------------------------------------------
        # DAQ and append new results to self.__evsSum_df
        #--------------------------------------------------
        dflt_evsSum_sql_kwargs = dict(
            schema_name      = 'meter_events', 
            table_name       = 'events_summary_vw', 
            cols_of_interest = ['*'], 
            date_range       = date_ranges_fnl, 
            trsf_pole_nbs    = self.trsf_pole_nbs, 
            field_to_split   = 'trsf_pole_nbs', 
            batch_size       = 1000
        )
        #-------------------------
        if evsSum_sql_kwargs is None:
            evsSum_sql_kwargs = dict()
        assert(isinstance(evsSum_sql_kwargs, dict))
        #-------------------------
        evsSum_sql_kwargs = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict    = evsSum_sql_kwargs, 
            default_values_dict = dflt_evsSum_sql_kwargs, 
            extend_any_lists    = False, 
            inplace             = False
        )
        #-----
        # Make 100% sure date_range and trsf_pole_nbs are properly set
        evsSum_sql_kwargs['date_range']    = date_ranges_fnl
        evsSum_sql_kwargs['trsf_pole_nbs'] = self.trsf_pole_nbs
        #-------------------------
        self.__evsSum_sql_fcn    = evsSum_sql_fcn
        self.__evsSum_sql_kwargs = copy.deepcopy(evsSum_sql_kwargs)
        #-------------------------
        evsSum = AMIEndEvents(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = dict(conn_db=self.conn_aws), 
            build_sql_function        = self.evsSum_sql_fcn, 
            build_sql_function_kwargs = self.evsSum_sql_kwargs, 
            init_df_in_constructor    = True, 
            save_args                 = save_args
        )
        #-------------------------
        if self.__evsSum_df is None or self.__evsSum_df.shape[0]==0:
            self.__evsSum_df = evsSum.df
        else:
            # Append evsSum.df to self.__evsSum_df
            #   But make temporary evsSum_df_new, instead of setting directly to self.__evsSum_df so if
            #   any error occurs in concat then original data are not lost
            evsSum_df = OutageMdlrPrep.append_to_evsSum_df(
                evsSum_df        = self.evsSum_df, 
                addtnl_evsSum_df = evsSum.df, 
                sort_by          = None
            )
            self.__evsSum_df = evsSum_df.copy()
        #-------------------------
        self.__evsSum_sql_stmnts.append(evsSum.get_sql_statement())
    
        #----------------------------------------------------------------------------------------------------
        # Handle __zero_counts_evsSum_df
        #     Any transformer whose meters did not register events will not be retrieved by the query utilized in OutagePredictor.build_raw_data
        #     Therefore, they must be added by hand!
        #     To avoid any more confusion than necessary, self.__evsSum_df will contain only meters registering events
        #       This is natural, as the SQL query will only return such elements.
        #     self.__zero_counts_evsSum_df will hold any transformers not registering events
        keys_to_ignore_for_zc = ['schema_name', 'table_name', 'cols_of_interest', 'date_range', 'field_to_split']
        sql_kwargs_zc         = {k:v for k,v in self.evsSum_sql_kwargs.items() if k not in keys_to_ignore_for_zc}
        #-----
        zero_counts_evsSum_df, zc_xfmrs_by_date = OutagePredictor.build_zero_counts_evsSum_df_for_date_ranges(
            trsf_pole_nbs           = self.trsf_pole_nbs, 
            evsSum_df               = self.evsSum_df, 
            daq_date_ranges         = date_ranges_fnl, 
            opco                    = self.opco, 
            sql_kwargs              = sql_kwargs_zc, 
            date_col                = 'aep_event_dt',
            trsf_pole_nb_col        = 'trsf_pole_nb', 
            SN_col                  = 'serialnumber', 
            PN_col                  = 'aep_premise_nb', 
            xf_meter_cnt_col        = 'xf_meter_cnt', 
            events_tot_col          = 'events_tot', 
            opco_col                = 'aep_opco', 
            conn_aws                = self.conn_aws, 
            return_zc_xfmrs_by_date = True, 
            verbose                 = verbose    
        )
        #-------------------------
        if self.__zero_counts_evsSum_df is None or self.__zero_counts_evsSum_df.shape[0]==0:
            self.__zero_counts_evsSum_df = zero_counts_evsSum_df
        else:
            # Append zero_counts_evsSum_df to self.__zero_counts_evsSum_df
            zero_counts_evsSum_df = OutageMdlrPrep.append_to_evsSum_df(
                evsSum_df        = self.zero_counts_evsSum_df, 
                addtnl_evsSum_df = zero_counts_evsSum_df, 
                sort_by          = None
            )
            self.__zero_counts_evsSum_df = zero_counts_evsSum_df
        #-------------------------
        assert(set(self.__zc_xfmrs_by_date.keys()).intersection(set(zc_xfmrs_by_date.keys()))==set())
        self.__zc_xfmrs_by_date = self.__zc_xfmrs_by_date | zc_xfmrs_by_date
    
        #----------------------------------------------------------------------------------------------------
        if self.__evsSum_df.shape[0]>0 and self.__zero_counts_evsSum_df.shape[0]>0:
            assert(all(self.__evsSum_df.columns==self.__zero_counts_evsSum_df.columns))

        #----------------------------------------------------------------------------------------------------
        # EEMSP DAQ and append new results to self.eemsp_df
        #--------------------------------------------------
        if self.merge_eemsp:
            self.buildextend_eemsp_df(
                daq_date_ranges        = date_ranges_fnl, 
                addtnl_kwargs          = None, 
                include_n_eemsp        = True, 
                cols_of_interest_eemsp = None, 
                numeric_cols           = ['kva_size'], 
                dt_cols                = ['install_dt', 'removal_dt'], 
                ignore_cols            = ['serial_nb'], 
                eemsp_cols_to_drop     = ['latest_status', 'serial_nb'], 
                batch_size             = 10000, 
                verbose                = verbose, 
                n_update               = 10, 
            )


        #----------------------------------------------------------------------------------------------------
        # Update daq_date_ranges to reflect newly acquired data run
        self.daq_date_ranges.extend(date_ranges_fnl)
        self.daq_date_ranges = Utilities.get_overlap_intervals(self.daq_date_ranges)


    def build_raw_data(
        self              , 
        evsSum_sql_fcn    = AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs = None, 
        save_args         = False
    ):
        r"""
        If user supplies any evsSum_sql_kwargs, user SHOULD NOT include:
            - date_range
            - trsf_pole_nbs
        as these will be supplied from self
        """
        #-------------------------
        self.build_raw_data_general( 
            prediction_dates  = [self.prediction_date], 
            predictions_range = None, 
            evsSum_sql_fcn    = evsSum_sql_fcn,  
            evsSum_sql_kwargs = evsSum_sql_kwargs, 
            save_args         = save_args
        )
        
        
        
    def prep_multiple_prediction_dates(
        self              , 
        prediction_dates  , 
        predictions_range = None, 
        evsSum_sql_fcn    = AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs = None, 
        save_args         = False, 
        verbose           = False
    ):
        r"""
        """
        #-------------------------
        self.build_raw_data_general(
            prediction_dates  = prediction_dates, 
            predictions_range = predictions_range, 
            evsSum_sql_fcn    = evsSum_sql_fcn,  
            evsSum_sql_kwargs = evsSum_sql_kwargs, 
            save_args         = save_args, 
            verbose           = verbose
        )
        #-------------------------
        pred_dates = set(prediction_dates).union(
            set(pd.date_range(
                start = predictions_range[0], 
                end   = predictions_range[1], 
                freq  = pd.Timedelta('1D')
            ))
        )
        pred_dates = natsorted(pred_dates)
        self.change_prediction_date(prediction_date = pred_dates[0])

        
        
    def build_and_load_regex_setup(
        self
    ):
        r"""
        Builds regex_setup_df and cr_trans_dict, where cr_trans_dict = curated reasons translation dictionary
        """
        #-------------------------
        sql = """
        SELECT * FROM meter_events.event_summ_regex_setup
        """
        # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
        # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.__regex_setup_df = pd.read_sql(sql, self.conn_aws, dtype=str)
        #-------------------------
        self.__cr_trans_dict = {x[0]:x[1] for x in self.__regex_setup_df[['pivot_id', 'regex_report_title']].values.tolist()}
            
    @staticmethod
    def get_distinct_trsf_pole_nbs_df(
        conn_aws        , 
        n_trsf_pole_nbs = None, 
        return_sql      = False, 
        **kwargs
    ):
        return MeterPremise.get_distinct_trsf_pole_nbs(
            n_trsf_pole_nbs = n_trsf_pole_nbs, 
            conn_aws        = conn_aws, 
            return_sql      = return_sql, 
            **kwargs
        )
    
        
    @staticmethod
    def build_eemsp_df(
        trsf_pole_nbs          , 
        date_range             , 
        addtnl_kwargs          = None, 
        conn_aws               = None, 
        mult_strategy          = 'agg', 
        include_n_eemsp        = True, 
        cols_of_interest_eemsp = None, 
        numeric_cols           = ['kva_size'], 
        dt_cols                = ['install_dt', 'removal_dt'], 
        ignore_cols            = ['serial_nb'], 
        eemsp_cols_to_drop     = ['latest_status', 'removal_dt', 'serial_nb'], 
        batch_size             = 10000, 
        verbose                = True, 
        n_update               = 10, 
    ):
        r"""
        date_range:
            Intended to be a list/tuple of length-2 defining the range.
            HOWEVER, functionality expanded to also accept a list/tuple of such objects
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(date_range, [list, tuple]))
        if Utilities.is_list_nested(lst=date_range, enforce_if_one_all=True):
            assert(Utilities.are_list_elements_lengths_homogeneous(lst=date_range, length=2))
        else:
            assert(isinstance(date_range, list) and len(date_range)==2)
        #-------------------------
        #--------------------------------------------------
        # 1. Grab eemsp_df for trsf_pole_nbs in date_range
        #--------------------------------------------------
        dflt_cols_of_interest_eemsp = TableInfos.EEMSP_ME_TI.std_columns_of_interest
        if cols_of_interest_eemsp is None:
            cols_of_interest_eemsp = dflt_cols_of_interest_eemsp
        if 'location_nb' not in cols_of_interest_eemsp:
            cols_of_interest_eemsp.append('location_nb')
        #-----
        #Important note: addtnl_cols below will be dropped later, so one would need to adjust the code if these were 
        #  desired in the output.  They are undesired for me, and this is the simplest solution.
        addtnl_cols = ['latest_status', 'removal_dt', 'serial_nb'] 
        cols_of_interest_eemsp_full = cols_of_interest_eemsp + addtnl_cols
        cols_of_interest_eemsp_full = list(set(cols_of_interest_eemsp_full))
        #-------------------------
        #-------------------------
        build_sql_function_kwargs = {}
        if addtnl_kwargs is not None:
            assert(isinstance(addtnl_kwargs, dict))
            build_sql_function_kwargs = addtnl_kwargs
        build_sql_function_kwargs['cols_of_interest'] = cols_of_interest_eemsp_full
        build_sql_function_kwargs['trsf_pole_nbs']    = trsf_pole_nbs
        build_sql_function_kwargs['datetime_range']   = date_range
        build_sql_function_kwargs['field_to_split']   = 'trsf_pole_nbs'
        build_sql_function_kwargs['batch_size']       = batch_size
        build_sql_function_kwargs['verbose']          = verbose
        build_sql_function_kwargs['n_update']         = n_update
        #-------------------------
        eemsp = EEMSP(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = dict(conn_db=conn_aws), 
            init_df_in_constructor    = True, 
            build_sql_function        = EEMSP.build_sql_eemsp, 
            build_sql_function_kwargs = build_sql_function_kwargs,
        )
        eemsp_df = eemsp.df
        #-------------------------
        eemsp_df = eemsp_df.reset_index(drop=True)
        #--------------------------------------------------
        # 2. Reduce down eemsp_df so there is a single entry for each transformer
        #     EEMSP.reduce1_eemsp_for_outg_trsf reduces eemsp_df down to contain only entries for transformers which were active 
        #       during the date(s) in question.
        #     No need to run EEMSP.reduce1_eemsp_for_outg_trsf for this case, as all share the same date restrictions which were 
        #       already imposed in sql_EEMSP.
        #     (For model development/training, this step would be necessary, as the data utilized there have many different 
        #       date restrictions, and eemsp_df cannot simply be built with the date restrictions)
        #     
        #     EEMSP.reduce2_eemsp_for_outg_trsf futher reduces eemsp_df down so there is a single entry for each transformer.
        #     How exactly this is achieved is dictated mainly by the "mult_strategy" parameter
        #--------------------------------------------------
        numeric_cols = [x for x in numeric_cols if x in eemsp_df.columns.tolist()]
        dt_cols      = [x for x in dt_cols if x in eemsp_df.columns.tolist()]
        ignore_cols  = [x for x in ignore_cols if x in eemsp_df.columns.tolist()]
        #-----
        eemsp_df_reduce2 = EEMSP.reduce2_eemsp_for_outg_trsf(
            df_eemsp            = eemsp_df, 
            mult_strategy       = mult_strategy, 
            include_n_eemsp     = include_n_eemsp,  
            grp_by_cols         = ['location_nb'], 
            numeric_cols        = numeric_cols, 
            dt_cols             = dt_cols, 
            ignore_cols         = ignore_cols, 
            cat_cols_as_strings = True
        )
        #-------------------------
        # No matter of the mult_strategy used, at this point eemsp_df_reduce2 should only have a single
        #   entry for each location_nb
        assert(all(eemsp_df_reduce2[['location_nb']].value_counts()==1))
    
        #----------------------------------------------------------------------------------------------------
        # Clean up eemsp_df_reduce2 and merge with rcpx_final
        #--------------------------------------------------
        cols_to_drop = [x for x in eemsp_cols_to_drop if x in eemsp_df_reduce2.columns]
        if len(cols_to_drop)>0:
            eemsp_df_reduce2 = eemsp_df_reduce2.drop(columns=cols_to_drop)
        #-------------------------
        assert(eemsp_df_reduce2.shape[0]==eemsp_df_reduce2.groupby(['location_nb']).ngroups)
        #-------------------------
        # Make all EEMSP columns (except n_eemsp) uppercase to match what was done in model development (where EEMSP)
        #   data were grabbed from the Oracle database, and columns were all uppercase)
        #eemsp_df_reduce2 = Utilities_df.make_all_column_names_uppercase(eemsp_df_reduce2, cols_to_exclude=['n_eemsp'])
        #-------------------------
        return eemsp_df_reduce2
    

    @staticmethod
    def merge_rcpx_with_eemsp(
        rcpx_df              , 
        eemsp_df             , 
        prediction_date      , 
        td_min               , 
        td_max               , 
        merge_on_rcpx        = ['index_0'], 
        merge_on_eems        = ['location_nb'],     
        eemsp_install_dt_col = 'install_dt', 
        eemsp_removal_dt_col = 'removal_dt', 
        drop_removal_dt_col  = True
    ):
        r"""
        """
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        #-------------------------
        Utilities_dt.assert_timedelta_is_days(td_min)
        Utilities_dt.assert_timedelta_is_days(td_max)
    
        #--------------------------------------------------
        # Grab only the time-appropriate entries from eemsp_df (meaning, installed before earliest date we need
        #   and removed after the latest date we need, if at all)
        eemsp_df_i = eemsp_df[
            (eemsp_df[eemsp_install_dt_col]                         <= prediction_date-td_max) & 
            (eemsp_df[eemsp_removal_dt_col].fillna(pd.Timestamp.max) > prediction_date-td_min)
        ]
        if drop_removal_dt_col:
            eemsp_df_i = eemsp_df_i.drop(columns=[eemsp_removal_dt_col])
        #--------------------------------------------------
        og_n_rows = rcpx_df.shape[0]
        rcpx_df = EEMSP.merge_rcpx_with_eemsp(
            df_rcpx               = rcpx_df, 
            df_eemsp              = eemsp_df_i, 
            merge_on_rcpx         = merge_on_rcpx, 
            merge_on_eems         = merge_on_eems, 
            set_index             = True, 
            drop_eemsp_merge_cols = True
        )
        #-------------------------
        # eemsp_df_i should only have one entry per transformer, and due to inner merge, rcpx_df
        #   the merge can only decrease the number of entries in rcpx_df
        assert(rcpx_df.shape[0]<=og_n_rows)
        #-------------------------
        return rcpx_df
    
    
    @staticmethod
    def convert_install_dt_to_years(
        rcpx_df          , 
        prediction_date  , 
        install_dt_col   = ('EEMSP_0', 'install_dt'), 
        assert_col_found = False
    ):
        r"""
        Convert the install_dt_col from date to age relative to prediction_date
        """
        #-------------------------
        if assert_col_found:
            assert(install_dt_col in rcpx_df.columns.tolist())
        #-----
        if install_dt_col in rcpx_df.columns.tolist():
            rcpx_df[install_dt_col] = (prediction_date-rcpx_df[install_dt_col]).dt.total_seconds()/(60*60*24*365)
        #-----
        return rcpx_df

    @staticmethod
    def add_predict_month_to_rcpx_df(
        rcpx_df                 , 
        prediction_date         , 
        month_col               = ('time_info_0', 'outg_month'), 
        dummy_col_levels_prefix = 'dummy'
    ):
        r"""
        month_col:
            Should either be a string or a tuple
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(month_col, [str, tuple]))
        if isinstance(month_col, str):
            if rcpx_df.columns.nlevels>1:
                n_levels_to_add = rcpx_df.columns.nlevels - 1
                month_col = [month_col]
                # With each iteration, prepending a new level from n_levels_to_add-1 to 0
                for i_new_lvl in range(n_levels_to_add):
                    month_col.insert(0, f'{dummy_col_levels_prefix}{(n_levels_to_add-1)-i_new_lvl}')
                assert(len(month_col)==rcpx_df.columns.nlevels)
                month_col = tuple(month_col)
        else:
            assert(len(month_col)==rcpx_df.columns.nlevels)
        #-------------------------
        rcpx_df[month_col] = prediction_date.month
        #-------------------------
        return rcpx_df

    @staticmethod
    def assert_rcpx_has_correct_form(
        rcpx_df           , 
        data_structure_df , 
    ):
        r"""
        Make sure rcpx_df has the correct columns in the correct order
        """
        #-------------------------
        if not len(set(data_structure_df.columns).symmetric_difference(set(rcpx_df.columns)))==0:
            print('rcpx_df does not have correct form!!!!!')
            #-----
            print('data_structure_df.columns = ')
            print(*data_structure_df.columns.tolist(), sep='\n')
            #-----
            print('rcpx_df.columns           = ')
            print(*rcpx_df.columns.tolist(), sep='\n')
            #-----
            print('Symmetric Difference      = ')
            print(*set(data_structure_df.columns).symmetric_difference(set(rcpx_df.columns)), sep='\n')
            #-----
            assert(0)
        rcpx_df=rcpx_df[data_structure_df.columns]
        #-------------------------
        return rcpx_df
    
    @staticmethod
    def build_X_test(
        rcpx_df           , 
        data_structure_df , 
        eemsp_args        = True, 
        scaler            = None
    ):
        r"""
        Construct X_test from rcpx_df

        eemsp_args:
            Can be False or dict
            If False or empty dict:
                Do not include eemsp
            If non-empty dict:
                Include eemsp with arguments given in eemsp_args
                AT A MINIMUM, MUST INCLUDE key 'eemsp_enc'
                Possible keys:
                    eemsp_enc
                    numeric_cols:
                        default = ['kva_size', 'install_dt']
                    EEMSP_col = 'EEMSP_0'
        """
        #--------------------------------------------------
        X_test = rcpx_df.copy()
        # Make sure everything looks good
        X_test = OutagePredictor.assert_rcpx_has_correct_form(
            rcpx_df=X_test, 
            data_structure_df=data_structure_df
        )
        #--------------------------------------------------
        # 1. If EEMSP included, run EEMSP encoder (eemsp_enc)
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(eemsp_args, [bool, dict]))
        if isinstance(eemsp_args, bool):
            assert(eemsp_args==False)
        if eemsp_args and eemsp_args['eemsp_enc'] is not None:
            # Non-empty dict, MUST INCLUDE at a minimum eemsp_enc
            assert('eemsp_enc' in eemsp_args.keys())
            eemsp_enc = eemsp_args['eemsp_enc']
            #-----
            EEMSP_col = eemsp_args.get('EEMSP_col', 'EEMSP_0')
            assert(EEMSP_col in data_structure_df.columns)
            assert(EEMSP_col in X_test.columns)
            cols_to_encode = data_structure_df[EEMSP_col].columns
            #-----
            numeric_cols = eemsp_args.get('numeric_cols', ['kva_size', 'install_dt'])
            numeric_cols = [x for x in numeric_cols if x in cols_to_encode]
            #-----
            cols_to_encode = [x for x in cols_to_encode if x not in numeric_cols]
            #-------------------------
            assert(len(set(eemsp_enc.feature_names_in_).symmetric_difference(cols_to_encode))==0)
            assert(set(X_test[EEMSP_col].columns).difference(eemsp_enc.feature_names_in_)==set(numeric_cols))
            #-----
            # Make sure cols_to_encode found in X_test, and convert to MultiIndex versions if needed
            cols_to_encode = [Utilities_df.find_single_col_in_multiindex_cols(df=X_test, col=x) for x in cols_to_encode] 
            X_test[cols_to_encode] = X_test[cols_to_encode].astype(str)
            X_test[cols_to_encode] = eemsp_enc.transform(X_test[cols_to_encode].droplevel(0, axis=1))
            
        #--------------------------------------------------
        # 2. Make sure everything still looks good
        #     NOTE: Perform before scaler run because scaler strips the columns off the data
        #--------------------------------------------------
        X_test = OutagePredictor.assert_rcpx_has_correct_form(
            rcpx_df           = X_test, 
            data_structure_df = data_structure_df
        )

        #--------------------------------------------------
        # 3. If scaler included, run
        #--------------------------------------------------
        if scaler is not None:
            X_test = scaler.transform(X_test)

        #--------------------------------------------------
        return X_test
    
    
    def build_xfmrs_ledger(
        self
    ):
        r"""
        If self.model_by_zip_clusters==True, then ledger must be loaded.
        This will build:
            1. self.zip_clstrs_ledger
            2. zip_clstrs_ledger
            3. self.__trsf_pole_zips_df (if needed)
    
        The general ledger, which is a simple dict object with keys equal to zip codes and values equal to model identifiers, 
          is expected to be located at os.path.join(self.model_dir, 'Slices', 'ledger.json')
    
        self.zip_clstrs_ledger:
            This is essentially a simplified inverse of ledger.
            The index will be the model identifiers, and the values will be lists containing the associated zip codes.
    
        self.zip_clstrs_ledger
            This is similar to self.zip_clstrs_ledger, except the values are lists containing the associated transformer pole numbers.
            In order for this to be built, the zip codes of the various transformers is needed.
            This information is supplied by self.__trsf_pole_zips_df.
            If self.__trsf_pole_zips_df does not exist, it will be built using OutageDAQ.build_trsf_pole_zips_df
        
        """
        #--------------------------------------------------
        assert(os.path.isdir(os.path.join(self.model_dir, 'Slices')))
        assert(os.path.exists(os.path.join(self.model_dir, 'Slices', 'ledger.json')))
        #-----
        tmp_f = open(os.path.join(self.model_dir, 'Slices', 'ledger.json'))
        ledger = json.load(tmp_f)
        tmp_f.close()
        #-----
        assert(isinstance(ledger, dict))
        #-------------------------
        if(
            self.__trsf_pole_zips_df is None or 
            self.__trsf_pole_zips_df.shape[0]==0
        ):
            zips_dict = OutageDAQ.build_trsf_pole_zips_df(
                field_to_split_and_val = ('trsf_pole_nbs', self.trsf_pole_nbs), 
                states                 = self.states, 
                opco                   = self.opco, 
                cities                 = self.cities
            )
            #-----
            self.__trsf_pole_zips_df = zips_dict['trsf_pole_zips_df']
        #-------------------------
        self.zip_clstrs_ledger = OutagePredictor.build_ledger_inv(
            ledger        = ledger, 
            return_type   = pd.Series
        )
        #-------------------------
        self.xfmrs_ledger = OutagePredictor.build_ledger_inv_with_trsf_pole_nbs(
            ledger            = ledger, 
            trsf_pole_zips_df = self.__trsf_pole_zips_df, 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            zip_col           = 'zip', 
            return_type       = pd.Series
        ) 

    
    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def initialize_data(
        self              , 
        evsSum_sql_fcn    = AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs = None, 

        # THESE SHOULD BE ADDED TO MODEL_SUMMARY_DICT AND AUTOMATICALLY EXTRACTED!!!!!
        freq                        = '5D', 
        group_cols                  = ['trsf_pole_nb'], 
        date_col                    = 'aep_event_dt', 
        normalize_by_SNs            = True, 
        include_power_down_minus_up = False, 
        regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons      = True, 
        include_n_eemsp             = True
    ):
        r"""
        Build/initialize all data needed for predictions.
        NOTE: trsf_pole_nbs and model_dir must be set prior to this operation!

        What does this function do?
            - Build events summary (self.__evsSum_df)
            - Build reason counts per x df (self.rcpx_df)
            - If including EEMSP, build and merge with rcpx (self.eemsp_df)
            - If needed, include month
            - Make sure final form of rcpx_df agrees with self.data_structure_df
            - Build/set self.X_test
        """
        #---------------------------------------------------------------------------
        # 1A. Make sure the transformer pole numbers have been set (whether done explicitly with
        #      set_trsf_pole_nbs or via sql with set_trsf_pole_nbs_from_sql)
        assert(self.trsf_pole_nbs is not None and len(self.trsf_pole_nbs)>0)

        # 1B. Make sure the model_dir has been set (and, through set_model_dir, all needed components
        #       have been extracted)
        assert(self.model_dir is not None and os.path.exists(self.model_dir))

        # 1C. Build zip_clstrs_ledger/xfmrs_ledger if needed
        if(
            self.model_by_zip_clusters and 
            (self.zip_clstrs_ledger is None or self.xfmrs_ledger is None)
        ):
            self.build_xfmrs_ledger()

        #---------------------------------------------------------------------------
        # 2. Build events summary (self.__evsSum_df and self.__zero_counts_evsSum_df)
        self.build_raw_data(
            evsSum_sql_fcn    = evsSum_sql_fcn,  
            evsSum_sql_kwargs = evsSum_sql_kwargs, 
            save_args         = False
        )

        #---------------------------------------------------------------------------
        # 3. Build EEMSP if needed

        #---------------------------------------------------------------------------
        # 3. Build reason counts per x df (self.rcpx_df) 
        evsSum_df = self.evsSum_df
        if self.__zero_counts_evsSum_df.shape[0]>0:
            evsSum_df = OutageMdlrPrep.append_to_evsSum_df(
                evsSum_df         = evsSum_df, 
                addtnl_evsSum_df  = self.__zero_counts_evsSum_df, 
                sort_by           = None, 
                make_col_types_eq = True
            )
        #-----
        self.rcpx_df = CPXDfBuilder.build_rcpx_from_evsSum_df(
            evsSum_df                   = evsSum_df,  
            data_structure_df           = self.data_structure_df, 
            prediction_date             = self.prediction_date, 
            td_min                      = self.idk_name_2, 
            td_max                      = self.idk_name_1, 
            cr_trans_dict               = self.cr_trans_dict, 
            freq                        = freq, 
            group_cols                  = group_cols, 
            date_col                    = date_col, 
            normalize_by_SNs            = normalize_by_SNs, 
            normalize_by_time           = True, 
            include_power_down_minus_up = include_power_down_minus_up, 
            regex_patterns_to_remove    = regex_patterns_to_remove, 
            combine_cpo_df_reasons      = combine_cpo_df_reasons, 
            xf_meter_cnt_col            = 'xf_meter_cnt', 
            events_tot_col              = 'events_tot', 
            trsf_pole_nb_col            = 'trsf_pole_nb', 
            other_reasons_col           = 'Other Reasons', 
            total_counts_col            = 'total_counts', 
            nSNs_col                    = 'nSNs'
        )

        #---------------------------------------------------------------------------
        # 4. If including EEMSP, merge with rcpx
        if self.merge_eemsp:
            self.rcpx_df = OutagePredictor.merge_rcpx_with_eemsp(
                rcpx_df              = self.rcpx_df, 
                eemsp_df             = self.eemsp_df, 
                prediction_date      = self.prediction_date, 
                td_min               = self.idk_name_2, 
                td_max               = self.idk_name_1, 
                merge_on_rcpx        = ['index_0'], 
                merge_on_eems        = ['location_nb'],     
                eemsp_install_dt_col = 'install_dt', 
                eemsp_removal_dt_col = 'removal_dt', 
                drop_removal_dt_col  = True
            )
            #-----
            self.rcpx_df = OutagePredictor.convert_install_dt_to_years(
                rcpx_df          = self.rcpx_df, 
                prediction_date  = self.prediction_date, 
                install_dt_col   = ('EEMSP_0', 'install_dt'), 
                assert_col_found = False
            )

        #---------------------------------------------------------------------------
        # 5. Include month?
        if self.include_month:
            self.rcpx_df = OutagePredictor.add_predict_month_to_rcpx_df(
                rcpx_df                 = self.rcpx_df, 
                prediction_date         = self.prediction_date, 
                month_col               = ('time_info_0', 'outg_month'), 
                dummy_col_levels_prefix = 'dummy'
            )

        #---------------------------------------------------------------------------
        # 6. Make sure final form of self.rcpx_df agrees with self.data_structure_df
        self.rcpx_df = OutagePredictor.assert_rcpx_has_correct_form(
            rcpx_df           = self.rcpx_df, 
            data_structure_df = self.data_structure_df
        )

        #---------------------------------------------------------------------------
        # 7. Set self.__trsf_pole_nbs_found
        self.__trsf_pole_nbs_found = self.rcpx_df.index.unique().tolist()

        #---------------------------------------------------------------------------
        # 8. Build X_test
        if self.rcpx_df.shape[0]>0:
            self.__X_test = OutagePredictor.build_X_test(
                rcpx_df           = self.rcpx_df, 
                data_structure_df = self.data_structure_df, 
                eemsp_args        = dict(eemsp_enc=self.eemsp_enc), 
                scaler            = self.scaler
            )
        else:
            self.__X_test = None
        
    def change_prediction_date(
        self, 
        prediction_date
    ):
        r"""
        """
        #-------------------------
        self.prediction_date = pd.to_datetime(prediction_date)

        # Update self.date_range
        self.set_date_range()

        # Run initialize_data to update all data
        self.initialize_data(
            evsSum_sql_fcn             = self.evsSum_sql_fcn,  
            evsSum_sql_kwargs          = self.evsSum_sql_kwargs, 

            # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # THESE SHOULD BE ADDED TO MODEL_SUMMARY_DICT AND AUTOMATICALLY EXTRACTED!!!!!
            freq                        = '5D', 
            group_cols                  = ['trsf_pole_nb'], 
            date_col                    = 'aep_event_dt', 
            normalize_by_SNs            = True, 
            include_power_down_minus_up = False, 
            regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
            combine_cpo_df_reasons      = True, 
            include_n_eemsp             = True
        )

    def build_y_pred_df(
        self, 
        threshold = 0.5
    ):
        r"""
        Build and set self.__y_pred_df
        """
        #--------------------------------------------------
        if self.X_test is None:
            print('Cannot build y_pred_df, X_test is None!')
            return
        #--------------------------------------------------
        y_scores = self.model_clf.predict_proba(self.X_test)
        #-------------------------
        assert(len(y_scores)==self.rcpx_df.shape[0])
        y_scores_df = pd.DataFrame(
            data    = y_scores, 
            columns = ['prob_0','prob_1'], 
            index   = self.rcpx_df.index
        )
        y_scores_df['pred'] = (y_scores_df['prob_1']>threshold).astype(int)
        #--------------------------------------------------
        self.__y_pred_df = y_scores_df
    
    def predict(
        self
    ):
        r"""
        Build and set self.__y_pred and self.__y_pred_df
        """
        #--------------------------------------------------
        if self.X_test is None:
            print('Cannot make prediction, X_test is None!')
            return
        #--------------------------------------------------
        y_pred        = self.model_clf.predict(self.X_test)
        self.__y_pred = y_pred
        #-------------------------
        self.build_y_pred_df(threshold = 0.5)


    def get_merged_rcpx_and_preds(
        self, 
        preds_lvl_0_val = 'pred', 
        sort_by_pred    = True
    ):
        r"""
        """
        #-------------------------
        assert(self.__y_pred_df.index.equals(self.rcpx_df.index))
        #-----
        rcpx_df = pd.merge(
            self.rcpx_df.copy(), 
            Utilities_df.prepend_level_to_MultiIndex(
                df         = self.y_pred_df, 
                level_val  = preds_lvl_0_val, 
                level_name = None, 
                axis       = 1
            ), 
            how         = 'inner', 
            left_index  = True, 
            right_index = True
        )
        assert(self.rcpx_df.shape[0]==rcpx_df.shape[0])
        #-------------------------
        if sort_by_pred:
            rcpx_df = rcpx_df.sort_values(by=[(preds_lvl_0_val, 'prob_1')], ascending=False)
        #-------------------------
        return rcpx_df
    

    def get_model_feature_importances(
        self, 
        return_sorted = True
    ):
        r"""
        """
        #-------------------------
        assert(self.rcpx_df.shape[1]==len(self.model_clf.feature_importances_))
        importances = list(zip(self.rcpx_df.columns, self.model_clf.feature_importances_))
        #-----
        if return_sorted:
            importances = natsorted(importances, key=lambda x: x[1], reverse=True)
        #-------------------------
        return importances
    
    def get_rcpx_for_features_of_top_importance(
        self, 
        n_features = 10
    ):
        r"""
        """
        #-------------------------
        importances = self.get_model_feature_importances(return_sorted=True)
        rcpx_df = self.rcpx_df[[x[0] for x in importances[:n_features]]].copy()
        return rcpx_df
    

    def internal_slice(
        self, 
        trsf_pole_nbs     , 
        trsf_pole_nb_idfr = 'trsf_pole_nb'
    ):
        r"""
        Currently, only intended to slice off specific subset of trsf_pole_nbs.
        """
        #--------------------------------------------------
        assert(isinstance(trsf_pole_nbs, list))
        #--------------------------------------------------
        if self.__evsSum_df is not None and self.__evsSum_df.shape[0]>0:
            assert(trsf_pole_nb_idfr in self.__evsSum_df.columns.tolist())
            self.__evsSum_df = self.__evsSum_df[self.__evsSum_df[trsf_pole_nb_idfr].isin(trsf_pole_nbs)].copy()
        #-------------------------
        if self.__zero_counts_evsSum_df is not None and self.__zero_counts_evsSum_df.shape[0]>0:
            assert(trsf_pole_nb_idfr in self.__zero_counts_evsSum_df.columns.tolist())
            self.__zero_counts_evsSum_df = self.__zero_counts_evsSum_df[self.__zero_counts_evsSum_df[trsf_pole_nb_idfr].isin(trsf_pole_nbs)].copy()
        #-------------------------
        assert(self.rcpx_df.index.name == trsf_pole_nb_idfr)
        self.rcpx_df = self.rcpx_df.loc[list(set(trsf_pole_nbs).intersection(set(self.rcpx_df.index.tolist())))].copy()
        #-------------------------
        if self.merge_eemsp:
            assert('location_nb' in self.eemsp_df.columns.tolist())
            self.eemsp_df = self.eemsp_df[self.eemsp_df['location_nb'].isin(trsf_pole_nbs)].copy()
    
    def slice_off_predictor(
        self, 
        trsf_pole_nbs , 
        model_dir     = None, 
        **model_kwargs
    ):
        r"""
        Currently, only intended to slice off specific subset of trsf_pole_nbs.
        This could be expanded later to be more general (like OutageModeler.slice_off_model is, I believe)
        """
        #--------------------------------------------------
        return_outg_pred = self.copy()
        #-------------------------
        return_outg_pred.set_trsf_pole_nbs(trsf_pole_nbs)
        #-------------------------
        return_outg_pred.internal_slice(
            trsf_pole_nbs     = trsf_pole_nbs, 
            trsf_pole_nb_idfr = 'trsf_pole_nb'
        )
        #-------------------------
        if model_dir is not None:
            return_outg_pred.set_model_dir(
                model_dir = model_dir, 
                **model_kwargs
            )
        #--------------------------------------------------
        assert(return_outg_pred.trsf_pole_nbs is not None and len(return_outg_pred.trsf_pole_nbs)>0)
        assert(return_outg_pred.model_dir is not None and os.path.exists(return_outg_pred.model_dir))
        #--------------------------------------------------
        # Make sure final form of return_outg_pred.rcpx_df agrees with return_outg_pred.data_structure_df
        return_outg_pred.rcpx_df = OutagePredictor.assert_rcpx_has_correct_form(
            rcpx_df           = return_outg_pred.rcpx_df, 
            data_structure_df = return_outg_pred.data_structure_df
        )
        #--------------------------------------------------
        # Set return_outg_pred.__trsf_pole_nbs_found
        return_outg_pred.set_trsf_pole_nbs_found(trsf_pole_nbs_found = return_outg_pred.rcpx_df.index.unique().tolist())
        #--------------------------------------------------
        # Build X_test
        if return_outg_pred.rcpx_df.shape[0]>0:
            X_test =  OutagePredictor.build_X_test(
                rcpx_df           = return_outg_pred.rcpx_df, 
                data_structure_df = return_outg_pred.data_structure_df, 
                eemsp_args        = dict(eemsp_enc=return_outg_pred.eemsp_enc), 
                scaler            = return_outg_pred.scaler
            )
        else:
            X_test = None
        return_outg_pred.set_X_test(X_test=X_test)
        #--------------------------------------------------
        return return_outg_pred
    

    #----------------------------------------------------------------------------------------------------
    def build_active_mp_df(
        self, 
        dt_0              = None, 
        dt_1              = None, 
        enforce_dt        = True
    ):
        r"""
        dt_0/_1:
            By default, self.date_range[0]/[1] are used.
            If user inputs values, the resulting range must encompass [self.date_range[0], self.date_range[1]]
                i.e., dt_0 <= self.date_range[0] and dt_1 >= self.date_range[1]
        """
        #--------------------------------------------------
        if dt_0 is None:
            dt_0 = self.date_range[0]
        if dt_1 is None:
            dt_1 = self.date_range[1]
        #-----
        if enforce_dt:
            assert(
                dt_0 <= self.date_range[0] and 
                dt_1 >= self.date_range[1]
            )
        
        #--------------------------------------------------
        df_mp_serial_number_col     = 'mfr_devc_ser_nbr'
        df_mp_prem_nb_col           = 'prem_nb'
        df_mp_install_time_col      = 'inst_ts'
        df_mp_removal_time_col      = 'rmvl_ts'
        df_mp_trsf_pole_nb_col      = 'trsf_pole_nb'
        #-----
        mp_df = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
            trsf_pole_nbs               = self.trsf_pole_nbs, 
            join_curr_hist              = True, 
            addtnl_mp_df_curr_cols      = None, 
            addtnl_mp_df_hist_cols      = None, 
            assume_one_xfmr_per_PN      = True, 
            drop_approx_duplicates      = True, 
            drop_approx_duplicates_args = None, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col      = df_mp_trsf_pole_nb_col
        )
        # Only want meters active at the relevant time period
        mp_df = mp_df[
            (mp_df[df_mp_install_time_col] <= pd.to_datetime(dt_0)) & 
            (mp_df[df_mp_removal_time_col].fillna(pd.Timestamp.max) > pd.to_datetime(dt_1))
        ].copy()
    
        #--------------------------------------------------
        return mp_df
    
    
    def build_dovs_df(
        self, 
        mp_df                   = None, 
        dt_0                    = None, 
        dt_1                    = None, 
        dovs_sql_function       = DOVSOutages_SQL.build_sql_std_outage,
        addtnl_dovs_sql_kwargs  = None, 
        enforce_dt              = True, 
        mp_overlap_assertion    = 0.95, 
        return_mp_df            = False, 
        df_mp_prem_nb_col       = 'prem_nb', 
        df_mp_install_time_col  = 'inst_ts', 
        df_mp_removal_time_col  = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col  = 'trsf_pole_nb'
    ):
        r"""
        mp_df:
            Essentially just need a mapping of premise to transformer
            If not supplied, it will be built using MeterPremise.build_mp_df_curr_hist_for_xfmrs.
    
        dt_0/_1:
            By default, np.min(self.daq_date_ranges) and np.max(self.daq_date_ranges) are used (which, if 
                only single data prediction, this will be self.date_range[0]/[1])
            If user inputs values, the resulting range must encompass [self.date_range[0], self.date_range[1]]
                i.e., dt_0 <= self.date_range[0] and dt_1 >= self.date_range[1]

        addtnl_dovs_sql_kwargs:
            The following are handled by the function: [premise_nbs, date_range, field_to_split, include_premise, include_DOVS_PREMISE_DIM]
            Any other appropriate kwargs can be implemented here
    
        mp_overlap_assertion:
            Must be 'full', 'any', or a float between 0 and 1
            Enforces the percentage of self.trsf_pole_nbs that must be found in mp_df
        """
        #--------------------------------------------------
        assert(
            isinstance(mp_overlap_assertion, float) or 
            mp_overlap_assertion in ['full', 'any']
        )
        
        #--------------------------------------------------
        if dt_0 is None:
            dt_0 = np.min(self.daq_date_ranges)
        if dt_1 is None:
            dt_1 = np.max(self.daq_date_ranges)
        if enforce_dt:
            assert(
                dt_0 <= self.date_range[0] and 
                dt_1 >= self.date_range[1]
            )
        
        #--------------------------------------------------
        if mp_df is None:
            mp_df = self.build_active_mp_df(
                dt_0       = dt_0, 
                dt_1       = dt_1, 
                enforce_dt = enforce_dt
            )
        # Only want meters active at the relevant time period
        mp_df = mp_df[
            (mp_df[df_mp_install_time_col] <= pd.to_datetime(dt_0)) & 
            (mp_df[df_mp_removal_time_col].fillna(pd.Timestamp.max) > pd.to_datetime(dt_1))
        ].copy()
            
        #--------------------------------------------------
        if isinstance(mp_overlap_assertion, float):
            assert(0 <= mp_overlap_assertion <= 1)
            overlap_pct = len(set(self.trsf_pole_nbs).intersection(set(mp_df[df_mp_trsf_pole_nb_col].unique())))/len(self.trsf_pole_nbs)
            assert(overlap_pct >= mp_overlap_assertion)
        else:
            if mp_overlap_assertion == 'full':
                assert(len(set(self.trsf_pole_nbs).difference(set(mp_df[df_mp_trsf_pole_nb_col].unique())))==0)
            elif mp_overlap_assertion == 'any':
                assert(len(set(self.trsf_pole_nbs).intersection(set(mp_df[df_mp_trsf_pole_nb_col].unique())))>0)
            else:
                assert(0)
                
        #--------------------------------------------------
        # Build dovs_df
        #-------------------------
        # There are instances where a user may actually prefer dt_0 to be greater than dt_1 for the mp_df functionality
        #   above (to be fully inclusive).
        # In such cases, dt_0,dt_1 need to be switched for dovs below.
        date_range = [np.min((dt_0,dt_1)), np.max((dt_0,dt_1))]
        #-----
        # NOTE: include_DOVS_PREMISE_DIM is included below (together with include_premise) to ensure premise is included
        #         regardless of dovs_sql_function used
        build_sql_function_kwargs = dict(
            premise_nbs              = mp_df[df_mp_prem_nb_col].unique().tolist(), 
            date_range               = date_range, 
            field_to_split           = 'premise_nbs', 
            include_premise          = True, 
            include_DOVS_PREMISE_DIM = True
        )
        #-------------------------
        if addtnl_dovs_sql_kwargs is not None:
            assert(isinstance(addtnl_dovs_sql_kwargs, dict))
            build_sql_function_kwargs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = build_sql_function_kwargs, 
                default_values_dict = addtnl_dovs_sql_kwargs, 
                extend_any_lists    = False, 
                inplace             = False
            )
        #-------------------------
        dovs = DOVSOutages(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = dict(conn_db=self.conn_dovs), 
            init_df_in_constructor    = True,
            build_sql_function        = dovs_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs, 
            build_consolidated        = False
        )
        dovs_df = dovs.df
    
        #-------------------------
        # Add trsf_pole_nb info to dovs_df
        dovs_df = pd.merge(
            dovs_df, 
            mp_df[[df_mp_prem_nb_col, df_mp_trsf_pole_nb_col]].drop_duplicates(), 
            how      = 'left', 
            left_on  = 'PREMISE_NB', 
            right_on = df_mp_prem_nb_col, 
        )
    
        #--------------------------------------------------
        if return_mp_df:
            return dovs_df, mp_df
        return dovs_df
    

