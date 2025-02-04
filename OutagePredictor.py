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
from OutageMdlrPrep import OutageMdlrPrep
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
        self.__conn_aws  = None
        #-------------------------
        self.__trsf_pole_nbs = None
        self.trsf_pole_nbs_sql = '' # If SQL run to obtain trsf_pole_nbs, this will be set
        if trsf_pole_nbs is not None:
            self.set_trsf_pole_nbs(trsf_pole_nbs)
        #-----
        self.__trsf_pole_nbs_found = None # Will be set when self.rcpx_df is built
        #-------------------------
        if prediction_date is None:
            self.prediction_date = pd.to_datetime(datetime.date.today())
        else:
            self.prediction_date = prediction_date
        #-----
        self.idk_name_1 = idk_name_1
        self.idk_name_2 = idk_name_2
        #-----
        self.date_range = None
        self.set_date_range()
        #-------------------------
        self.daq_date_ranges = [] # Tracks the date ranges of the acquired data
        #-------------------------
        # Although I will use AMIEndEvents methods, the object which is built will not
        #   contain end_device_event entries (ede), but rather events_summary_vw (evsSum)
        self.__evsSum_sql_fcn        = None
        self.__evsSum_sql_kwargs     = None
        self.__evsSum_sql_stmnts     = []     # If prediction date changed or whatever, evsSum_df will be agg. of multiple DAQ runs
        self.__evsSum_df             = None 
        self.__zero_counts_evsSum_df = None   # IMPORTANT: Remember, the 'serialnumber' and 'aep_premise_nb' are essentially meaningless in this DF
        self.__zc_xfmrs_by_date      = dict() # This may be a temporary member, keeping because seeing some (possibly) strange things during dev
        self.zero_counts_evsSum_df_TMP = None
        #-------------------------
        self.__regex_setup_df = None
        self.__cr_trans_dict  = None
        #-----
        self.rcpx_df = None
        self.is_norm = False
        #-------------------------
        self.merge_eemsp = False
        self.eemsp_mult_strategy='agg'
        self.eemsp_df = None
        #-------------------------
        self.__model_dir        = None
        self.model_summary_dict = None
        self.data_structure_df  = None
        self.model_clf          = None
        self.scale_data         = False
        self.scaler             = None
        self.eemsp_enc          = None
        self.include_month      = False
        #-----
        self.__X_test    = None
        self.__y_pred    = None
        self.__y_pred_df = None

    
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
    def trsf_pole_nbs(self):
        return self.__trsf_pole_nbs
    @property
    def trsf_pole_nbs_found(self):
        return self.__trsf_pole_nbs_found
    def set_trsf_pole_nbs_found(self, trsf_pole_nbs_found):
        self.__trsf_pole_nbs_found = trsf_pole_nbs_found
    
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
        #--------------------------------------------------
        self.__trsf_pole_nbs           = OutagePredictor.general_copy(outg_pred.__trsf_pole_nbs)
        self.trsf_pole_nbs_sql         = OutagePredictor.general_copy(outg_pred.trsf_pole_nbs_sql)
        self.__trsf_pole_nbs_found     = OutagePredictor.general_copy(outg_pred.__trsf_pole_nbs_found)
        #--------------------------------------------------
        self.prediction_date           = OutagePredictor.general_copy(outg_pred.prediction_date)
        self.idk_name_1                = OutagePredictor.general_copy(outg_pred.idk_name_1)
        self.idk_name_2                = OutagePredictor.general_copy(outg_pred.idk_name_2)
        self.set_date_range()
        #--------------------------------------------------
        self.daq_date_ranges           = OutagePredictor.general_copy(outg_pred.daq_date_ranges)
        #--------------------------------------------------
        self.__evsSum_sql_fcn          = OutagePredictor.general_copy(outg_pred.__evsSum_sql_fcn)
        self.__evsSum_sql_kwargs       = OutagePredictor.general_copy(outg_pred.__evsSum_sql_kwargs)
        self.__evsSum_sql_stmnts       = OutagePredictor.general_copy(outg_pred.__evsSum_sql_stmnts)
        self.__evsSum_df               = OutagePredictor.general_copy(outg_pred.__evsSum_df)
        self.__zero_counts_evsSum_df   = OutagePredictor.general_copy(outg_pred.__zero_counts_evsSum_df)
        self.__zc_xfmrs_by_date        = OutagePredictor.general_copy(outg_pred.__zc_xfmrs_by_date)
        self.zero_counts_evsSum_df_TMP = OutagePredictor.general_copy(outg_pred.zero_counts_evsSum_df_TMP)
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
            '_OutagePredictor__trsf_pole_nbs',
            'trsf_pole_nbs_sql',
            '_OutagePredictor__trsf_pole_nbs_found',
            'prediction_date',
            'idk_name_1',
            'idk_name_2',
            'date_range',
            'daq_date_ranges',
            '_OutagePredictor__evsSum_sql_fcn',
            '_OutagePredictor__evsSum_sql_kwargs',
            '_OutagePredictor__evsSum_sql_stmnts',
            # '_OutagePredictor__evsSum_df',
            # '_OutagePredictor__zero_counts_evsSum_df',
            # '_OutagePredictor__zc_xfmrs_by_date',
            # 'zero_counts_evsSum_df_TMP',
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
            '_OutagePredictor__trsf_pole_nbs',
            'trsf_pole_nbs_sql',
            '_OutagePredictor__trsf_pole_nbs_found',
            'prediction_date',
            'idk_name_1',
            'idk_name_2',
            'date_range',
            'daq_date_ranges',
            '_OutagePredictor__evsSum_sql_fcn',
            '_OutagePredictor__evsSum_sql_kwargs',
            '_OutagePredictor__evsSum_sql_stmnts',
            # '_OutagePredictor__evsSum_df',
            # '_OutagePredictor__zero_counts_evsSum_df',
            # '_OutagePredictor__zc_xfmrs_by_date',
            # 'zero_counts_evsSum_df_TMP',
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
            '_OutagePredictor__X_test',
            '_OutagePredictor__y_pred',
            '_OutagePredictor__y_pred_df'
        ]
        #--------------------------------------------------
        if keep_attrs is None:
            keep_attrs = dflt_keep_attrs
        assert(isinstance(keep_attrs, list))
        assert('_OutagePredictor__conn_aws' not in keep_attrs) # cannot pickle 'pyodbc.Connection'!
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
        self.__conn_aws = None
        #-------------------------
        f = open(file_path, 'wb')
        cPickle.dump(self.__dict__, f)
        f.close()

    # def save_to_pkl(
    #     self, 
    #     save_path
    # ):
    #     r"""
    #     cannot pickle 'pyodbc.Connection' object ==> must set self.__conn_aws to None
    #     """
    #     #-------------------------
    #     self.__conn_aws = None
    #     #-------------------------
    #     with open(save_path, 'wb') as handle:
    #         pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)



    #----------------------------------------------------------------------------------------------------
    
    
    def set_trsf_pole_nbs(
        self, 
        trsf_pole_nbs
    ):
        assert(isinstance(trsf_pole_nbs, list))
        self.__trsf_pole_nbs = trsf_pole_nbs
        
    def set_trsf_pole_nbs_from_sql(
        self, 
        n_trsf_pole_nbs=None, 
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
        self, 
        model_dir, 
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
        self.__model_dir = model_dir
        #-------------------------
        model_summary_dict_fname = kwargs.get('model_summary_dict_fname', 'summary_dict.json')
        data_structure_fname     = kwargs.get('data_structure_fname', 'data_structure_df.pkl')
        model_fname              = kwargs.get('model_fname', 'model_clf.joblib')
        scaler_fname             = kwargs.get('scaler_fname', 'scaler.joblib')
        eemsp_encoder_fname      = kwargs.get('eemsp_encoder_fname', 'eemsp_encoder.joblib')
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
        df
    ):
        r"""
        """
        #-------------------------
        regex_pattern=r'cr\d*'
        #-----
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        #-------------------------
        return cr_cols
    
    
    @staticmethod
    def build_ledger_inv(
        ledger, 
        return_type = pd.Series, 
        trsf_pole_nbs = None
    ):
        r"""
        Input:
            keys   = trsf_pole_nbs
            values = model identifiers
    
        Output:
            return_type == pd.Series
                a pd.Series object with index equal to model identifier and values equal to lists of trsf_pole_nbs
            return_type == dict
                keys   = model identifiers
                values = lists of trsf_pole_nbs covered by model
    
        trsf_pole_nbs:
            If None, simply inver ledger
            If supplied, ensure they are contained in ledger, and then return inverted version of ledger only containing the 
              poles contained in trsf_pole_nbs
        """
        #-------------------------
        assert(return_type in [pd.Series, dict])
        #-------------------------
        ledger_srs = pd.Series(ledger)
        #-------------------------
        if trsf_pole_nbs is not None:
            assert(set(trsf_pole_nbs).difference(set(ledger.keys()))==set())
            ledger_srs = ledger_srs.loc[trsf_pole_nbs]
        #-------------------------
        # Invert the series, so indices are model ids and values are trsf_pole_nbs
        # AND group by model id and collect trsf_pole_nbs in lists
        ledger_inv = pd.Series(ledger_srs.index.values, index=ledger_srs).groupby(level=0).apply(list)
        #-------------------------
        if return_type==dict:
            ledger_inv = ledger_inv.to_dict()
        #-------------------------
        return ledger_inv
    
    @staticmethod
    def group_trsf_pole_nbs_by_model(
        trsf_pole_nbs, 
        ledger
    ):
        r"""
        Returns a pd.Series object with index equal to model identifier and values equal to lists of trsf_pole_nbs
        """
        #-------------------------
        ledger_inv = OutagePredictor.build_ledger_inv(
            ledger        = ledger, 
            return_type   = pd.Series, 
            trsf_pole_nbs = trsf_pole_nbs        
        )
        #-------------------------
        return ledger_inv
        
        
    @staticmethod
    def build_sql_xfmr_meter_cnt_closest_to_date(
        trsf_pole_nbs, 
        date, 
        evsSum_sql_kwargs=None, 
        include_meter_cols=False
    ):
        r"""
        This function was built specifically to be used with the OutagePredictor class (and also to be used in the data acquisition
          portion of the baseline formation).
        For these data grabs, any transformer whose meters did not register any events will not be retrieved by the query.
        However, these zero event transformers should really be included in the baseline, and must be included in the OutagePredictor
          process otherwise no predictions can be made for these (the expectation is that the model should return False, i.e., predict no outage
          for transformers suffering zero events in a given time period, but this will be interesting, and possibly enlightening, to check)
    
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
        dflt_evsSum_sql_kwargs = dict(
            cols_of_interest = ['trsf_pole_nb', 'xf_meter_cnt', 'aep_opco', 'aep_event_dt'], 
            trsf_pole_nbs    = trsf_pole_nbs
        )
        #-------------------------
        if evsSum_sql_kwargs is None:
            evsSum_sql_kwargs = dict()
        assert(isinstance(evsSum_sql_kwargs, dict))
        #-------------------------
        evsSum_sql_kwargs = Utilities.supplement_dict_with_default_values(
            to_supplmnt_dict=evsSum_sql_kwargs, 
            default_values_dict=dflt_evsSum_sql_kwargs, 
            extend_any_lists=False, 
            inplace=False
        )
        #-----
        # Make 100% sure trsf_pole_nbs properly set
        evsSum_sql_kwargs['trsf_pole_nbs'] = trsf_pole_nbs
        #--------------------------------------------------
        cols_of_interest = evsSum_sql_kwargs['cols_of_interest']
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
        sql_where = AMI_SQL.add_ami_where_statements(sql_where, **evsSum_sql_kwargs)
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
        evsSum_sql_kwargs  = None, 
        include_meter_cols = False, 
        batch_size         = 1000, 
        verbose            = True, 
        n_update           = 10, 
        conn_aws           = None, 
        return_sql         = False,
        keep_extra_cols    = False
    ):
        r"""
        This function was built specifically to be used with the OutagePredictor class (and also to be used in the data acquisition
          portion of the baseline formation).
        For these data grabs, any transformer whose meters did not register any events will not be retrieved by the query.
        However, these zero event transformers should really be included in the baseline, and must be included in the OutagePredictor
          process otherwise no predictions can be made for these (the expectation is that the model should return False, i.e., predict no outage
          for transformers suffering zero events in a given time period, but this will be interesting, and possibly enlightening, to check)
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
                evsSum_sql_kwargs  = evsSum_sql_kwargs, 
                include_meter_cols = include_meter_cols
            )
            assert(isinstance(sql_i, str))
            #-----
            df_i = pd.read_sql_query(sql_i, conn_aws)
            #-----
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
    def build_zero_counts_evsSum_df(
        trsf_pole_nbs, 
        evsSum_df, 
        date, 
        evsSum_sql_kwargs = None, 
        date_col          = 'aep_event_dt',
        trsf_pole_nb_col  = 'trsf_pole_nb', 
        conn_aws          = None, 
        verbose           = True
    ):
        r"""
        Any transformer whose meters did not register events will not be retrieved by the query.
            Therefore, they must be added by hand.
            That is the purpose of this function.
    
        Find any trsf_pole_nbs missing from evsSum_df, grab their meter counts info from events_summary_vw, and then populate
          a pd.DataFrame of same structure as evsSum_df with all zeros for counts on date input  (intended to be self.prediction_date - self.idk_name_2).
        NOTE: This originally was not a static function, hence the somewhat strange structure and documentation.
        NOTE: keys_to_ignore (defined below) will be ignored if input in evsSum_sql_kwargs
    
        To avoid any more confusion than necessary, self.__evsSum_df will contain only meters registering events
            This is natural, as the SQL query will only return such elements.
            Therefore, the zero_counts_df built here will not be appended to self.__evsSum_df.
    
        This function will produce a pd.DataFrame object of the same form as self.__evsSum_df for the transformers whose meters
          did not register an event.
        Only one single date is needed for each transformer, the function OutageMdlrPrep.build_rcpx_from_evsSum_df will handle the rest, as usual.
        IMPORTANT NOTE: It doesn't really matter what this single date is, AS LONG AS IT FALLS WITHIN THE WINDOW OF INTEREST.
                          i.e., the date must be between start=prediction_date-td_max and end=prediction_date-td_min.
                        However, it does slightly matter, as the function OutagePredictor.get_xfmr_meter_cnt_closest_to_date will be used to find the meter counts for
                          the transformers, and therefore using the date closest to the prediction date is likely to best choice (although, the meter count
                          is likely stable during the window of interest, which is why it doesn't really matter much)
        """
        #--------------------------------------------------
        zero_counts_xfmrs = list(set(trsf_pole_nbs).difference(set(evsSum_df[trsf_pole_nb_col].unique())))
        #-------------------------
        if len(zero_counts_xfmrs)==0:
            return pd.DataFrame()
            # return None
        #-------------------------
        if evsSum_sql_kwargs is None:
            evsSum_sql_kwargs = dict()
        assert(isinstance(evsSum_sql_kwargs, dict))
        #-----
        # Let trsf_pole_nbs, cols_of_interest, and date (and, therefore, date_range) be handled by OutagePredictor.get_xfmr_meter_cnt_closest_to_date
        keys_to_ignore = ['trsf_pole_nbs', 'trsf_pole_nb', 'cols_of_interest', 'date', 'date_range', 'datetime_range']
        evsSum_sql_kwargs = {k:v for k,v in evsSum_sql_kwargs.items() if k not in keys_to_ignore}
        #----------
        zero_counts_df = OutagePredictor.get_xfmr_meter_cnt_closest_to_date(
            trsf_pole_nbs      = zero_counts_xfmrs, 
            date               = date, 
            evsSum_sql_kwargs  = evsSum_sql_kwargs, 
            include_meter_cols = True, 
            batch_size         = 1000, 
            verbose            = verbose, 
            n_update           = 10, 
            conn_aws           = conn_aws, 
            return_sql         = False,
            keep_extra_cols    = False
        )
        #-------------------------
        tmp_idx_col = Utilities.generate_random_string()
        zero_counts_df[tmp_idx_col] = list(range(evsSum_df.shape[0], evsSum_df.shape[0]+zero_counts_df.shape[0]))
        zero_counts_df = zero_counts_df.set_index(tmp_idx_col, drop=True)
        zero_counts_df.index.name = None
        #--------------------------------------------------
        addtln_zero_counts_df_cols = pd.DataFrame(
            data    = 0, 
            index   = list(range(evsSum_df.shape[0], evsSum_df.shape[0]+zero_counts_df.shape[0])), 
            columns = list(set(evsSum_df.columns).difference(set(zero_counts_df.columns)))
        )
        assert(all(zero_counts_df.index==addtln_zero_counts_df_cols.index))
        zero_counts_df = pd.concat([zero_counts_df, addtln_zero_counts_df_cols], axis=1, ignore_index=False)
        #-------------------------
        # As described above, only need each trsf_pole_nb to have a single entry within the window of interest, the fuction
        # OutagePredictor.project_time_pd_from_rcpx_0_and_prepare will handle zeroing everything else out
        zero_counts_df[date_col] = date
        #-----
        assert(set(evsSum_df.columns).symmetric_difference(set(zero_counts_df.columns))==set())
        zero_counts_df = zero_counts_df[evsSum_df.columns]
        #--------------------------------------------------
        return zero_counts_df
    
    @staticmethod
    def build_zero_counts_evsSum_df_multiple_pred_dates(
        trsf_pole_nbs, 
        evsSum_df, 
        prediction_dates, 
        idk_name_1, 
        idk_name_2, 
        evsSum_sql_kwargs       = None, 
        date_col                = 'aep_event_dt',
        trsf_pole_nb_col        = 'trsf_pole_nb', 
        conn_aws                = None, 
        return_zc_xfmrs_by_date = False, 
        verbose                 = True    
    ):
        r"""
        """
        #--------------------------------------------------
        # Make sure date_col is datetime object
        evsSum_df[date_col] = pd.to_datetime(evsSum_df[date_col])
        
        zc_xfmrs_by_date = dict()
        zero_counts_df = pd.DataFrame()
        for pred_date_i in prediction_dates:
            evsSum_df_i = evsSum_df[
                (evsSum_df[date_col] >= pred_date_i - idk_name_1) & 
                (evsSum_df[date_col] <= pred_date_i - idk_name_2)
            ]
            zero_counts_xfmrs_i = list(set(trsf_pole_nbs).difference(set(evsSum_df_i[trsf_pole_nb_col].unique())))
            if len(zero_counts_xfmrs_i)>0:
                assert(pred_date_i not in zc_xfmrs_by_date.keys())
                zc_xfmrs_by_date[pred_date_i] = zero_counts_xfmrs_i
                #-----
                zero_counts_df_i = OutagePredictor.build_zero_counts_evsSum_df(
                    trsf_pole_nbs     = zero_counts_xfmrs_i, 
                    evsSum_df         = evsSum_df_i, 
                    date              = pred_date_i - idk_name_2, 
                    evsSum_sql_kwargs = evsSum_sql_kwargs, 
                    date_col          = date_col,
                    trsf_pole_nb_col  = trsf_pole_nb_col, 
                    conn_aws          = conn_aws, 
                    verbose           = verbose
                )
                #-------------------------
                zero_counts_df = OutageMdlrPrep.append_to_evsSum_df(
                    evsSum_df        = zero_counts_df, 
                    addtnl_evsSum_df = zero_counts_df_i, 
                    sort_by          = None
                )
        #--------------------------------------------------
        if return_zc_xfmrs_by_date:
            return zero_counts_df, zc_xfmrs_by_date
        else:
            return zero_counts_df
            
        
    # def build_events_summary_general(
    #     self, 
    #     date_ranges, 
    #     evsSum_sql_fcn=AMIEndEvents_SQL.build_sql_end_events,  
    #     evsSum_sql_kwargs=None, 
    #     save_args=False
    # ):
    #     r"""
    #     If user supplies any evsSum_sql_kwargs, user SHOULD NOT include:
    #         - date_range
    #         - trsf_pole_nbs
    #     as these will be supplied from self

    #     Function tagged as _general because it will be used by:
    #         OutagePredictor.build_events_summary (pre-dates this function, and is essentially the guts of this function)
    #         OutagePredictor.prep_multiple_prediction_dates
    #     """
    #     #----------------------------------------------------------------------------------------------------
    #     # First, determine which unique date ranges, if any, we need to acquire/build data for
    #     #   1. Consolidate date_ranges using Utilities.get_overlap_intervals (building date_ranges_0)
    #     #   2. For each unique date range in date_ranges_0, remove any intervals already contained in self.daq_date_ranges, 
    #     #        leaving only the ranges needed for acquisition
    #     #   3. Note: date_ranges_fnl.extend is utilized instead of .append because Utilities.remove_overlaps_from_date_interval was 
    #     #        built such that a list of pd.Interval objects is always returned.
    #     #--------------------------------------------------
    #     assert(Utilities.is_object_one_of_types(date_ranges, [list, tuple]))
    #     date_ranges_0 = []
    #     for date_range_i in date_ranges:
    #         assert(Utilities.is_object_one_of_types(date_range_i, [list, tuple, pd.Interval]))
    #         if isinstance(date_range_i, pd.Interval):
    #             date_ranges_0.append([date_range_i.left, date_range_i.right])
    #         elif Utilities.is_object_one_of_types(date_range_i, [list, tuple]):
    #             date_ranges_0.append(list(date_range_i))
    #         else:
    #             assert(0)    
    #     #--------------------------------------------------
    #     date_ranges_0 = Utilities.get_overlap_intervals(ranges=date_ranges_0)
    #     #-------------------------
    #     date_ranges_fnl = []
    #     for date_range_i in date_ranges_0:
    #         date_ranges_fnl_i = Utilities.remove_overlaps_from_date_interval(
    #             intrvl_1  = date_range_i, 
    #             intrvls_2 = self.daq_date_ranges, 
    #             closed    = True
    #         )
    #         date_ranges_fnl.extend(date_ranges_fnl_i)
    #     #--------------------------------------------------
    #     # If all dates in date_ranges are already accounted for in self.daq_date_ranges, simply return
    #     if len(date_ranges_fnl)==0:
    #         return    
    #     #----------------------------------------------------------------------------------------------------
    #     # DAQ and append new results to self.__evsSum_df
    #     #--------------------------------------------------
    #     date_ranges_fnl = Utilities.get_overlap_intervals(ranges=date_ranges_fnl)
    #     #--------------------------------------------------
    #     dflt_evsSum_sql_kwargs = dict(
    #         schema_name='meter_events', 
    #         table_name='events_summary_vw', 
    #         cols_of_interest=['*'], 
    #         date_range=date_ranges_fnl, 
    #         trsf_pole_nbs=self.trsf_pole_nbs
    #     )
    #     #-------------------------
    #     if evsSum_sql_kwargs is None:
    #         evsSum_sql_kwargs = dict()
    #     assert(isinstance(evsSum_sql_kwargs, dict))
    #     #-------------------------
    #     evsSum_sql_kwargs = Utilities.supplement_dict_with_default_values(
    #         to_supplmnt_dict=evsSum_sql_kwargs, 
    #         default_values_dict=dflt_evsSum_sql_kwargs, 
    #         extend_any_lists=False, 
    #         inplace=False
    #     )
    #     #-----
    #     # Make 100% sure date_range and trsf_pole_nbs are properly set
    #     evsSum_sql_kwargs['date_range']    = date_ranges_fnl
    #     evsSum_sql_kwargs['trsf_pole_nbs'] = self.trsf_pole_nbs
    #     #-------------------------
    #     self.__evsSum_sql_fcn    = evsSum_sql_fcn
    #     self.__evsSum_sql_kwargs = copy.deepcopy(evsSum_sql_kwargs)
    #     #-------------------------
    #     evsSum = AMIEndEvents(
    #         df_construct_type         = DFConstructType.kRunSqlQuery, 
    #         contstruct_df_args        = dict(conn_db=self.conn_aws), 
    #         build_sql_function        = self.evsSum_sql_fcn, 
    #         build_sql_function_kwargs = self.evsSum_sql_kwargs, 
    #         init_df_in_constructor    = True, 
    #         save_args                 = save_args
    #     )
    #     #-------------------------
    #     if self.__evsSum_df is None or self.__evsSum_df.shape[0]==0:
    #         self.__evsSum_df = evsSum.df.copy()
    #     else:
    #         # Append evsSum.df to self.__evsSum_df
    #         #   But make temporary evsSum_df_new, instead of setting directly to self.__evsSum_df so if
    #         #   any error occurs in concat then original data are not lost
    #         evsSum_df = OutageMdlrPrep.append_to_evsSum_df(
    #             evsSum_df        = self.evsSum_df, 
    #             addtnl_evsSum_df = evsSum.df, 
    #             sort_by          = None
    #         )
    #         self.__evsSum_df = evsSum_df.copy()
    #     #-------------------------
    #     self.__evsSum_sql_stmnts.append(evsSum.get_sql_statement())
    #     self.daq_date_ranges.extend(date_ranges_fnl)
    #     self.daq_date_ranges = Utilities.get_overlap_intervals(self.daq_date_ranges)


    def build_events_summary_general(
        self, 
        prediction_dates, 
        predictions_range=None, 
        evsSum_sql_fcn=AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs=None, 
        save_args=False, 
        verbose=False
    ):
        r"""
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
            OutagePredictor.build_events_summary (pre-dates this function, and is essentially the guts of this function)
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
        #----------------------------------------------------------------------------------------------------
        # First, determine which unique date ranges, if any, we need to acquire/build data for
        #   1. Consolidate date_ranges using Utilities.get_overlap_intervals (building date_ranges_0)
        #   2. For each unique date range in date_ranges_0, remove any intervals already contained in self.daq_date_ranges, 
        #        leaving only the ranges needed for acquisition
        #   3. Note: date_ranges_fnl.extend is utilized instead of .append because Utilities.remove_overlaps_from_date_interval was 
        #        built such that a list of pd.Interval objects is always returned.
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(date_ranges, [list, tuple]))
        date_ranges_0 = []
        for date_range_i in date_ranges:
            assert(Utilities.is_object_one_of_types(date_range_i, [list, tuple, pd.Interval]))
            if isinstance(date_range_i, pd.Interval):
                date_ranges_0.append([date_range_i.left, date_range_i.right])
            elif Utilities.is_object_one_of_types(date_range_i, [list, tuple]):
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
                intrvls_2 = self.daq_date_ranges, 
                closed    = True
            )
            date_ranges_fnl.extend(date_ranges_fnl_i)
        #--------------------------------------------------
        # If all dates in date_ranges are already accounted for in self.daq_date_ranges, simply return
        if len(date_ranges_fnl)==0:
            return    
        #----------------------------------------------------------------------------------------------------
        # DAQ and append new results to self.__evsSum_df
        #--------------------------------------------------
        date_ranges_fnl = Utilities.get_overlap_intervals(ranges=date_ranges_fnl)
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
        self.daq_date_ranges.extend(date_ranges_fnl)
        self.daq_date_ranges = Utilities.get_overlap_intervals(self.daq_date_ranges)
    
        #----------------------------------------------------------------------------------------------------
        # Handle __zero_counts_evsSum_df
        #     Any transformer whose meters did not register events will not be retrieved by the query utilized in OutagePredictor.build_events_summary
        #     Therefore, they must be added by hand!
        #     To avoid any more confusion than necessary, self.__evsSum_df will contain only meters registering events
        #       This is natural, as the SQL query will only return such elements.
        #     self.__zero_counts_evsSum_df will hold any transformers not registering events
        zero_counts_evsSum_df, zc_xfmrs_by_date = OutagePredictor.build_zero_counts_evsSum_df_multiple_pred_dates(
            trsf_pole_nbs           = self.trsf_pole_nbs, 
            evsSum_df               = self.evsSum_df, 
            prediction_dates        = prediction_dates, 
            idk_name_1              = self.idk_name_1, 
            idk_name_2              = self.idk_name_2, 
            evsSum_sql_kwargs       = self.evsSum_sql_kwargs, 
            date_col                = 'aep_event_dt',
            trsf_pole_nb_col        = 'trsf_pole_nb', 
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


    def build_events_summary(
        self, 
        evsSum_sql_fcn=AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs=None, 
        save_args=False
    ):
        r"""
        If user supplies any evsSum_sql_kwargs, user SHOULD NOT include:
            - date_range
            - trsf_pole_nbs
        as these will be supplied from self
        """
        #-------------------------
        self.build_events_summary_general( 
            prediction_dates  = [self.prediction_date], 
            predictions_range = None, 
            evsSum_sql_fcn    = evsSum_sql_fcn,  
            evsSum_sql_kwargs = evsSum_sql_kwargs, 
            save_args         = save_args
        )
        

    @staticmethod
    def get_date_range_for_prediction(
        prediction_date, 
        idk_name_1, 
        idk_name_2
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
        prediction_dates, 
        predictions_range=None, 
        idk_name_1 = pd.Timedelta('31D'), 
        idk_name_2 = pd.Timedelta('1D')
    ):
        r"""
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
        
        
    def prep_multiple_prediction_dates(
        self, 
        prediction_dates, 
        predictions_range=None, 
        evsSum_sql_fcn=AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs=None, 
        save_args=False, 
        verbose=False
    ):
        r"""
        """
        #-------------------------
        self.build_events_summary_general(
            prediction_dates  = prediction_dates, 
            predictions_range = predictions_range, 
            evsSum_sql_fcn    = evsSum_sql_fcn,  
            evsSum_sql_kwargs = evsSum_sql_kwargs, 
            save_args         = save_args, 
            verbose           = verbose
        )

        
        
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
        self.__regex_setup_df = pd.read_sql(sql, self.conn_aws, dtype=str)
        #-------------------------
        self.__cr_trans_dict = {x[0]:x[1] for x in self.__regex_setup_df[['pivot_id', 'regex_report_title']].values.tolist()}
            
    @staticmethod
    def get_distinct_trsf_pole_nbs_df(
        conn_aws, 
        n_trsf_pole_nbs=None, 
        return_sql=False, 
        **kwargs
    ):
        return MeterPremise.get_distinct_trsf_pole_nbs(
            n_trsf_pole_nbs=n_trsf_pole_nbs, 
            conn_aws=conn_aws, 
            return_sql=return_sql, 
            **kwargs
        )
    
    
    @staticmethod
    def project_time_pd_from_rcpx_0_and_prepare_OLD(
        rcpx_0, 
        date_pd_i, 
        final_time_pd_i, 
        data_structure_df, 
        meter_cnt_per_gp_srs, 
        all_groups, 
        cr_trans_dict, 
        non_reason_cols, 
        other_reasons_col, 
        group_cols=['trsf_pole_nb'], 
        date_col='aep_event_dt', 
        regex_patterns_to_remove=['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons=True, 
        include_power_down_minus_up=False, 
        total_counts_col = 'total_counts', 
        nSNs_col = 'nSNs', 
    ):
        r"""
        """
        #-------------------------
        # Get the expected columns for this time period from data_structure_df
        final_reason_cols_i = data_structure_df[final_time_pd_i].columns.tolist()
        final_reason_cols_i = [x for x in final_reason_cols_i if x not in non_reason_cols+[other_reasons_col]]
        #-------------------------
        # Project out the current time period (date_pd_i) from rcpx_0 by selecting the appropriate
        #   values from the date_col index
        rcpx_0_pd_i = rcpx_0[rcpx_0.index.get_level_values(date_col)==date_pd_i].copy()
        rcpx_0_pd_i = rcpx_0_pd_i.droplevel(date_col, axis=0)
        #-------------------------
        # Make sure all groups (typically trsf_pole_nbs) have an entry in rcpx_0_pd_i:
        #   If a group didn't register any events in a given time period, it will not be included in the projection.
        #   However, the final format requires each group have entries for each time period
        #   Therefore, we identify the groups missing from rcpx_0_pd_i (no_events_pd_i) and add approriate rows
        #     containing all 0 values for the counts
        # NOTE: If group_cols contains more than one element, the index will be a MultiIndex (equal to the group_cols)
        #         and will need to be treated slightly different
        no_events_pd_i = list(set(all_groups).difference(set(rcpx_0_pd_i.index.unique())))
        if len(group_cols)==1:
            no_ev_idx = no_events_pd_i
        else:
            no_ev_idx = pd.MultiIndex.from_tuples(no_events_pd_i)
        no_events_pd_i_df = pd.DataFrame(
            columns=rcpx_0.columns, 
            index=no_ev_idx, 
            data=np.zeros((len(no_events_pd_i), rcpx_0.shape[1]))
        )
        no_events_pd_i_df.index.names = rcpx_0_pd_i.index.names
        #-----
        # Use meter_cnt_per_gp_srs to fill the nSNs_col column in no_events_pd_i_df (since simply full of 0s)
        # NOTE: This is probably not strictly necessary, as the nSNs_col column won't be used here,
        #         since the data are not normalized.
        no_events_pd_i_df = no_events_pd_i_df.drop(columns=[nSNs_col]).merge(
            meter_cnt_per_gp_srs, 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        # Sanity check on the merge
        assert(no_events_pd_i_df[nSNs_col].notna().all())
        #-----
        # Combine rcpx_0_pd_i and no_events_pd_i_df
        assert(len(set(rcpx_0_pd_i.columns).symmetric_difference(set(no_events_pd_i_df.columns)))==0)
        no_events_pd_i_df = no_events_pd_i_df[rcpx_0_pd_i.columns]
        rcpx_0_pd_i = pd.concat([rcpx_0_pd_i, no_events_pd_i_df])
        #-------------------------
        # Rename the cr# columns to their full curated reasons
        rcpx_0_pd_i=rcpx_0_pd_i.rename(columns=cr_trans_dict)
        #--------------------------------------------------    
        #--------------------------------------------------
        # Any columns without a curated reason (i.e., those with column name = ''), have not been observed
        #   yet in the data, and therefore the sume of the counts should be 0.
        # These empty columns are not needed, so drop
        assert(rcpx_0_pd_i[''].sum().sum()==0)
        rcpx_0_pd_i=rcpx_0_pd_i.drop(columns=[''])  
        #-------------------------
        # Remove any undesired curated reasons (e.g., ['.*cleared.*', '.*Test Mode.*'])
        if regex_patterns_to_remove is not None:
            rcpx_0_pd_i = MECPODf.remove_reasons_from_rcpo_df(
                rcpo_df=rcpx_0_pd_i, 
                regex_patterns_to_remove=regex_patterns_to_remove, 
                ignore_case=True
            )
        #-----
        # After irrelevant cleared and test columns removed, need to recalculate events_tot to accurately
        #   reflect the total number of relevant events
        # Safe(r) to do this calculation in any case, so moved outside of the if block above
        assert(total_counts_col in non_reason_cols)
        rcpx_0_pd_i[total_counts_col] = rcpx_0_pd_i.drop(columns=non_reason_cols).sum(axis=1) 
        #-------------------------
        # Combine similar reasons (e.g., all 'Tamper' type reasons are combined into 1)
        # See MECPODf.combine_cpo_df_reasons for more information
        if combine_cpo_df_reasons:
            rcpx_0_pd_i = MECPODf.combine_cpo_df_reasons(rcpo_df=rcpx_0_pd_i)
        #-------------------------
        # Include the difference in power-up and power-down, if desired (typically turned off) 
        if include_power_down_minus_up:
            rcpx_0_pd_i = MECPODf.delta_cpo_df_reasons(
                rcpo_df=rcpx_0_pd_i, 
                reasons_1='Primary Power Down',
                reasons_2='Primary Power Up',
                delta_reason_name='Power Down Minus Up'
            )
        #-------------------------
        # Make sure rcpx_0_pd_i contains the expected final reason columns.
        # Once this is assured, project out these reasons and combine all other reasons into
        #   the other_reasons_col columns
        # See MECPODf.get_reasons_subset_from_cpo_df for more info
        assert(len(set(final_reason_cols_i).difference(set(rcpx_0_pd_i.columns.tolist())))==0)
        rcpx_0_pd_i = MECPODf.get_reasons_subset_from_cpo_df(
            cpo_df=rcpx_0_pd_i, 
            reasons_to_include=final_reason_cols_i, 
            combine_others=True, 
            output_combine_others_col=other_reasons_col, 
            SNs_tags=None, 
            is_norm=False, 
            counts_col=nSNs_col, 
            normalize_by_nSNs_included=False, 
            level_0_raw_col = 'counts', 
            level_0_nrm_col = 'counts_norm', 
            cols_to_ignore = [total_counts_col], 
            include_counts_col_in_output=True
        )    
        #--------------------------------------------------
        #--------------------------------------------------
        # Don't want nSNs in each pd individually
        rcpx_0_pd_i = rcpx_0_pd_i.drop(columns=[nSNs_col])
        #-------------------------
        # Add the correct time period name as level 0 of the columns
        rcpx_0_pd_i = Utilities_df.prepend_level_to_MultiIndex(
            df=rcpx_0_pd_i, 
            level_val=final_time_pd_i, 
            level_name=None, 
            axis=1
        )
        #-------------------------
        return rcpx_0_pd_i
    
    @staticmethod
    def build_rcpx_from_evsSum_df_OLD(
        evsSum_df, 
        data_structure_df, 
        prediction_date, 
        td_min, 
        td_max, 
        cr_trans_dict, 
        freq='5D', 
        group_cols=['trsf_pole_nb'], 
        date_col='aep_event_dt', 
        normalize_by_SNs=True, 
        normalize_by_time=True, 
        include_power_down_minus_up=False, 
        regex_patterns_to_remove=['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons=True, 
        xf_meter_cnt_col = 'xf_meter_cnt', 
        events_tot_col = 'events_tot', 
        trsf_pole_nb_col = 'trsf_pole_nb', 
        other_reasons_col = 'Other Reasons',  # From data_structure_df
        total_counts_col = 'total_counts', 
        nSNs_col         = 'nSNs', 
    ):
        r"""
        This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
          
        NOTE: td_min, td_max, and freq must all be in DAYS
        """
        #--------------------------------------------------
        # 0. Need data_structure_df
        #     In general, not all curated reasons will be included in the model.
        #     Typically, 10 commong curated reasons will be included, and all others will be grouped together in "Other Reasons".
        #     Furthermore, some reasons may be combined together, others may be completely removed.
        #     For these reasons, it is beneficial to have some sample data (taken from when the model was created) to utilize 
        #       in structuring the new data in the same fashion.
        #     Additionally, the data will be used to ensure the ordering of columns is correct before the data are fed into 
        #       the model.
        assert(isinstance(data_structure_df, pd.DataFrame) and data_structure_df.shape[0]>0)

        #--------------------------------------------------
        #--------------------------------------------------
        # 1. Build rcpo_0
        #     Construct rcpx_0 by aggregating evsSum_df by group_cols and by freq on date_col
        #--------------------------------------------------
        evsSum_df = evsSum_df.copy()
        #-------------------------
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        assert(len(set(group_cols).difference(set(evsSum_df.columns.tolist())))==0)
        #-------------------------
        # Need to set origin in pd.Grouper to ensure proper grouping
        freq = pd.Timedelta(freq)
        assert((td_max-td_min) % freq==pd.Timedelta(0))
        #-----
        time_grps = pd.date_range(
            start = prediction_date-td_max, 
            end   = prediction_date-td_min, 
            freq  = freq
        )
        time_grps = [(time_grps[i], time_grps[i+1]) for i in range(len(time_grps)-1)]
        assert(len(time_grps) == (td_max-td_min)/pd.Timedelta(freq))
        #-------------------------
        group_freq=pd.Grouper(freq=freq, key=date_col, origin=time_grps[0][0])
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst=evsSum_df.columns.tolist(), 
            regex_pattern=r'cr\d*', 
            ignore_case=False
        )
        #-----
        cols_to_drop = set(evsSum_df.columns.tolist()).difference(
            set(cr_cols+group_cols+[date_col, xf_meter_cnt_col, events_tot_col])
        )
        cols_to_drop = list(cols_to_drop)
        #-------------------------
        # Make sure date_col is datetime object
        evsSum_df[date_col] = pd.to_datetime(evsSum_df[date_col])
        
        #-------------------------
        # No need in wasting time grouping data we won't use
        # So, reduce evsSum_df to only the dates we're interested in 
        evsSum_df = evsSum_df[
            (evsSum_df[date_col] >= prediction_date-td_max) & 
            (evsSum_df[date_col] <= prediction_date-td_min)
        ]


        # All of the cr# columns will be aggregated with np.sum, as will events_tot_col
        # xf_meter_cnt_col will be aggregated using np.max
        agg_dict = {col:np.sum for col in cr_cols+[events_tot_col, xf_meter_cnt_col]}
        agg_dict[xf_meter_cnt_col] = np.max
        #-------------------------
        rcpx_0 = evsSum_df.drop(columns=cols_to_drop).groupby(group_cols+[group_freq]).agg(agg_dict)

        #--------------------------------------------------
        # 2. Grab meter_cnt_per_gp_srs and all_groups
        #--------------------------------------------------
        # Project out the meter count per group, as it will be used later
        #   This information will be stored in the pd.Series object meter_cnt_per_gp_srs, where the index will
        #   contain the group_cols
        meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[xf_meter_cnt_col]].drop_duplicates().set_index(group_cols).squeeze()
        assert(meter_cnt_per_gp_srs.shape[0]==meter_cnt_per_gp_srs.index.nunique())
        meter_cnt_per_gp_srs.name = nSNs_col

        # Will also need the unique groups in rcpx_0
        #   This will be used later (see no_events_pd_i below)
        #   These can be grabbed from the index of rcpx_0 (excluding the date_col level)
        all_groups = rcpx_0.droplevel(date_col, axis=0).index.unique().tolist()

        #--------------------------------------------------
        # 3. Transform rcpx_0 to the form expected by the model
        #     i.e., similar to data_structure_df.
        #     This is essentially just changing rcpo_0 from long form to wide form
        #--------------------------------------------------
        #-------------------------
        # 3a. Build time_pds_rename
        #      Need to convert the time periods, which are currently housed in the date_col index of 
        #        rcpx_0 from their specific dates to the names expected by the model.
        #      In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
        #        dates of the given interval.
        #      These will be converted to the titles contained in final_time_pds below
        #      NOTE: This is probably not 100% necessary, but is useful nonetheless
        #-------------------------
        curr_time_pds = [x[0] for x in time_grps]
        time_pds_rename = OutageMdlrPrep.get_time_pds_rename(
            curr_time_pds=curr_time_pds, 
            td_min=td_min, 
            td_max=td_max, 
            freq=freq
        )
        final_time_pds = list(time_pds_rename.values())
        # final_time_pds should all be found in data_structure_df to help
        #   ensure the alignment between the current data and data used when modelling
        assert(set(final_time_pds).difference(data_structure_df.columns.get_level_values(0).unique())==set())
        #-------------------------
        # Overkill here (since all time windows are of length freq), but something similar will 
        #   be needed if I want to move to non-uniform period lengths
        time_grps_dict = dict()
        assert(len(curr_time_pds) == len(time_grps))
        # Each element in curr_time_pds should match exactly one of the 0th elements 
        #   in time_grps (which is a list of length-2 tuples)
        # Make sure this is so while building time_grps_dict
        for curr_time_pd_i in curr_time_pds:
            time_grp_i = [x for x in time_grps if x[0]==curr_time_pd_i]
            assert(len(time_grp_i)==1)
            assert(curr_time_pd_i not in time_grps_dict.keys())
            time_grps_dict[curr_time_pd_i] = time_grp_i[0]

        #-------------------------
        # 3b. Transform rcpx_0 to the form expected by the model
        #      As stated above, this is essentially just changing rcpo_0 from long form to wide form
        #      This will probably be formalized further in the future (i.e., function(s) developed to handle)
        rename_cols = {
            events_tot_col:total_counts_col, 
            xf_meter_cnt_col:nSNs_col
        }
        rcpx_0=rcpx_0.rename(columns=rename_cols)
        #-------------------------
        total_counts_col = total_counts_col
        nSNs_col         = nSNs_col
        non_reason_cols = [nSNs_col, total_counts_col]
        #------------------------- 
        pd_dfs = []
        for date_pd_i in curr_time_pds:
            # Grab the proper time period name from final_time_pd_i
            final_time_pd_i = time_pds_rename[date_pd_i]
            #-------------------------
            rcpx_0_pd_i = OutagePredictor.project_time_pd_from_rcpx_0_and_prepare_OLD(
                rcpx_0                      = rcpx_0, 
                date_pd_i                   = date_pd_i, 
                final_time_pd_i             = final_time_pd_i, 
                data_structure_df           = data_structure_df, 
                meter_cnt_per_gp_srs        = meter_cnt_per_gp_srs, 
                all_groups                  = all_groups, 
                cr_trans_dict               = cr_trans_dict, 
                non_reason_cols             = non_reason_cols, 
                other_reasons_col           = other_reasons_col, 
                group_cols                  = group_cols, 
                date_col                    = date_col, 
                regex_patterns_to_remove    = regex_patterns_to_remove, 
                combine_cpo_df_reasons      = combine_cpo_df_reasons, 
                include_power_down_minus_up = include_power_down_minus_up, 
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col
            )
            #-------------------------
            # Overkill here (since all time windows are of length freq), but something similar will 
            #   be needed if I want to move to non-uniform period lengths
            # One could, e.g., simply divide by length of freq in days
            if normalize_by_time:
                time_grp_i = time_grps_dict[date_pd_i]
                #-----
                days_min_outg_td_window_i = prediction_date - time_grp_i[1]
                days_max_outg_td_window_i = prediction_date - time_grp_i[0]
                #-----
                OutageMdlrPrep.assert_timedelta_is_days(days_min_outg_td_window_i)
                OutageMdlrPrep.assert_timedelta_is_days(days_max_outg_td_window_i)
                #-----
                days_min_outg_td_window_i = days_min_outg_td_window_i.days
                days_max_outg_td_window_i = days_max_outg_td_window_i.days
                #-------------------------
                rcpx_0_pd_i = MECPODf.normalize_rcpo_df_by_time_interval(
                    rcpo_df                 = rcpx_0_pd_i, 
                    days_min_outg_td_window = days_min_outg_td_window_i, 
                    days_max_outg_td_window = days_max_outg_td_window_i, 
                    cols_to_adjust          = None, 
                    SNs_tags                = None, 
                    inplace                 = True
                )
            #-------------------------
            pd_dfs.append(rcpx_0_pd_i)

        # Make sure all dfs in pd_dfs look correct
        shape_0 = pd_dfs[0].shape
        index_0 = pd_dfs[0].index
        for i in range(len(pd_dfs)):
            if i==0:
                continue
            assert(pd_dfs[i].shape==shape_0)
            assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
            #-----
            # Aligning the indices is not strictly necessary, as pd.concat should handle that
            # But, it's best to be safe
            pd_dfs[i] = pd_dfs[i].loc[index_0]

        # Build rcpx_final by combining all dfs in pd_dfs
        rcpx_final = pd.concat(pd_dfs, axis=1)

        # Include back in the number of SNs per group (from meter_cnt_per_gp_srs)
        rcpx_final=rcpx_final.merge(
            meter_cnt_per_gp_srs.to_frame(name=(nSNs_col, nSNs_col)), 
            left_index=True, 
            right_index=True, 
            how='left'
        )
        # Sanity check on the merge
        assert(rcpx_final[nSNs_col].notna().all().all())

        #--------------------------------------------------
        # 4. Normalize by nSNs
        #--------------------------------------------------
        if normalize_by_SNs:
            # Kind of silly, but below I cannot simply use 'rcpx_final[final_time_pds] = ...'
            #   This will result in: "ValueError: Columns must be same length as key", because final_time_pds
            #   has only, e.g., 6 elements but rcpx_final[final_time_pds] contains, e.g., 72 columns
            # Instead, must use 'rcpx_final[rcpx_final[final_time_pds].columns] = ..'
            rcpx_final[rcpx_final[final_time_pds].columns] = rcpx_final[final_time_pds].divide(rcpx_final[(nSNs_col, nSNs_col)], axis=0)

        #--------------------------------------------------
        return rcpx_final
    
        
    @staticmethod
    def build_eemsp_df(
        trsf_pole_nbs, 
        date_range, 
        addtnl_kwargs=None, 
        mult_strategy='agg', 
        include_n_eemsp=True, 
        cols_of_interest_eemsp=None, 
        numeric_cols = ['kva_size'], 
        dt_cols = ['install_dt', 'removal_dt'], 
        ignore_cols = ['serial_nb'], 
        batch_size=10000, 
        verbose=True, 
        n_update=10, 
    ):
        r"""
        """
        #-------------------------
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
            contstruct_df_args        = None,
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
            df_eemsp=eemsp_df, 
            mult_strategy=mult_strategy, 
            include_n_eemsp=include_n_eemsp,  
            grp_by_cols=['location_nb'], 
            numeric_cols = numeric_cols, 
            dt_cols = dt_cols, 
            ignore_cols = ignore_cols, 
            cat_cols_as_strings=True
        )
        #-------------------------
        # No matter of the mult_strategy used, at this point eemsp_df_reduce2 should only have a single
        #   entry for each location_nb
        assert(all(eemsp_df_reduce2[['location_nb']].value_counts()==1))
    
        #----------------------------------------------------------------------------------------------------
        # Clean up eemsp_df_reduce2 and merge with rcpx_final
        #--------------------------------------------------
        cols_to_drop = [x for x in addtnl_cols if x in eemsp_df_reduce2.columns]
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
    def build_eemsp_df_and_merge_rcpx( 
        rcpx_df, 
        trsf_pole_nbs, 
        date_range, 
        merge_on_rcpx=['index_0'], 
        merge_on_eems=['location_nb'], 
        conn_aws=None, 
        mult_strategy='agg', 
        include_n_eemsp=True, 
        cols_of_interest_eemsp=None, 
        numeric_cols = ['kva_size'], 
        dt_cols = ['install_dt', 'removal_dt'], 
        ignore_cols = ['serial_nb'], 
        batch_size=10000, 
        verbose=True, 
        n_update=10, 
    ):
        r"""
        """
        #-------------------------
        eemsp_df = OutagePredictor.build_eemsp_df( 
            trsf_pole_nbs          = trsf_pole_nbs, 
            date_range             = date_range, 
            mult_strategy          = mult_strategy, 
            include_n_eemsp        = include_n_eemsp, 
            cols_of_interest_eemsp = cols_of_interest_eemsp, 
            numeric_cols           = numeric_cols, 
            dt_cols                = dt_cols, 
            ignore_cols            = ignore_cols, 
            batch_size             = batch_size, 
            verbose                = verbose, 
            n_update               = n_update 
        )
        #-------------------------
        #-------------------------
        og_n_rows = rcpx_df.shape[0]
        rcpx_df = EEMSP.merge_rcpx_with_eemsp(
            df_rcpx=rcpx_df, 
            df_eemsp=eemsp_df, 
            merge_on_rcpx=merge_on_rcpx, 
            merge_on_eems=merge_on_eems, 
            set_index=True, 
            drop_eemsp_merge_cols=True
        )
        #-------------------------
        # eemsp_df should only have one entry per transformer, and due to inner merge, rcpx_df
        #   the merge can only decrease the number of entries in rcpx_df
        assert(rcpx_df.shape[0]<=og_n_rows)
        #-------------------------
        return rcpx_df, eemsp_df
    
    
    @staticmethod
    def convert_install_dt_to_years(
        rcpx_df, 
        prediction_date, 
        install_dt_col=('EEMSP_0', 'install_dt'), 
        assert_col_found=False
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
        rcpx_df, 
        prediction_date, 
        month_col=('time_info_0', 'outg_month'), 
        dummy_col_levels_prefix='dummy'
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
        rcpx_df, 
        data_structure_df
    ):
        r"""
        Make sure rcpx_df has the correct columns in the correct order
        """
        #-------------------------
        assert(len(set(data_structure_df.columns).symmetric_difference(set(rcpx_df.columns)))==0)
        rcpx_df=rcpx_df[data_structure_df.columns]
        #-------------------------
        return rcpx_df
    
    @staticmethod
    def build_X_test(
        rcpx_df, 
        data_structure_df, 
        eemsp_args=True, 
        scaler=None
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
            rcpx_df=X_test, 
            data_structure_df=data_structure_df
        )

        #--------------------------------------------------
        # 3. If scaler included, run
        #--------------------------------------------------
        if scaler is not None:
            X_test = scaler.transform(X_test)

        #--------------------------------------------------
        return X_test
    
    # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def initialize_data(
        self, 
        evsSum_sql_fcn=AMIEndEvents_SQL.build_sql_end_events,  
        evsSum_sql_kwargs=None, 

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

        #---------------------------------------------------------------------------
        # 2. Build events summary (self.__evsSum_df and self.__zero_counts_evsSum_df)
        self.build_events_summary(
                evsSum_sql_fcn    = evsSum_sql_fcn,  
                evsSum_sql_kwargs = evsSum_sql_kwargs, 
                save_args         = False
            )

        # #---------------------------------------------------------------------------
        # # 3. Build reason counts per x df (self.rcpx_df)
        # self.rcpx_df = OutageMdlrPrep.build_rcpx_from_evsSum_df(
        #     evsSum_df                   = self.evsSum_df,  
        #     data_structure_df           = self.data_structure_df, 
        #     prediction_date             = self.prediction_date, 
        #     td_min                      = self.idk_name_2, 
        #     td_max                      = self.idk_name_1, 
        #     cr_trans_dict               = self.cr_trans_dict, 
        #     freq                        = freq, 
        #     group_cols                  = group_cols, 
        #     date_col                    = date_col, 
        #     normalize_by_SNs            = normalize_by_SNs, 
        #     normalize_by_time           = True, 
        #     include_power_down_minus_up = include_power_down_minus_up, 
        #     regex_patterns_to_remove    = regex_patterns_to_remove, 
        #     combine_cpo_df_reasons      = combine_cpo_df_reasons, 
        #     xf_meter_cnt_col            = 'xf_meter_cnt', 
        #     events_tot_col              = 'events_tot', 
        #     trsf_pole_nb_col            = 'trsf_pole_nb', 
        #     other_reasons_col           = 'Other Reasons', 
        #     total_counts_col            = 'total_counts', 
        #     nSNs_col                    = 'nSNs'
        # )

        # #-------------------------
        # # 3b. Build reason counts per x df for self.__zero_counts_evsSum_df and append to self.rcpx_df
        # if self.__zero_counts_evsSum_df.shape[0]>0:
        #     rcpx_df_zc = OutageMdlrPrep.build_rcpx_from_evsSum_df(
        #         evsSum_df                   = self.__zero_counts_evsSum_df, 
        #         data_structure_df           = self.data_structure_df, 
        #         prediction_date             = self.prediction_date, 
        #         td_min                      = self.idk_name_2, 
        #         td_max                      = self.idk_name_1, 
        #         cr_trans_dict               = self.cr_trans_dict, 
        #         freq                        = freq, 
        #         group_cols                  = group_cols, 
        #         date_col                    = date_col, 
        #         normalize_by_SNs            = normalize_by_SNs, 
        #         normalize_by_time           = True, 
        #         include_power_down_minus_up = include_power_down_minus_up, 
        #         regex_patterns_to_remove    = regex_patterns_to_remove, 
        #         combine_cpo_df_reasons      = combine_cpo_df_reasons, 
        #         xf_meter_cnt_col            = 'xf_meter_cnt', 
        #         events_tot_col              = 'events_tot', 
        #         trsf_pole_nb_col            = 'trsf_pole_nb', 
        #         other_reasons_col           = 'Other Reasons', 
        #         total_counts_col            = 'total_counts', 
        #         nSNs_col                    = 'nSNs'
        #     )
        #     #-----
        #     self.rcpx_df = OutageMdlrPrep.append_to_evsSum_df(
        #         evsSum_df        = self.rcpx_df, 
        #         addtnl_evsSum_df = rcpx_df_zc, 
        #         sort_by          = None
        #     )

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
        self.rcpx_df = OutageMdlrPrep.build_rcpx_from_evsSum_df(
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
        # 4. If including EEMSP, build and merge with rcpx
        if self.merge_eemsp:
            self.rcpx_df, self.eemsp_df = OutagePredictor.build_eemsp_df_and_merge_rcpx( 
                    rcpx_df                = self.rcpx_df, 
                    trsf_pole_nbs          = self.trsf_pole_nbs, 
                    date_range             = self.date_range, 
                    merge_on_rcpx          = ['index_0'], 
                    merge_on_eems          = ['location_nb'], 
                    conn_aws               = self.conn_aws, 
                    mult_strategy          = self.eemsp_mult_strategy, 
                    include_n_eemsp        = include_n_eemsp, 
                    cols_of_interest_eemsp = None, 
                    numeric_cols           = ['kva_size'], 
                    dt_cols                = ['install_dt', 'removal_dt'], 
                    ignore_cols            = ['serial_nb'], 
                    batch_size             = 10000, 
                    verbose                = True, 
                    n_update               = 10
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
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.initialize_data(
            evsSum_sql_fcn=self.evsSum_sql_fcn,  
            evsSum_sql_kwargs=self.evsSum_sql_kwargs, 

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
            columns = [0,1], 
            index   = self.rcpx_df.index
        )
        y_scores_df['pred'] = (y_scores_df[1]>threshold).astype(int)
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
            rcpx_df = rcpx_df.sort_values(by=[(preds_lvl_0_val, 1)], ascending=False)
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
        mp_df = mp_df[(mp_df[df_mp_install_time_col] <= pd.to_datetime(dt_0)) & 
                      (mp_df[df_mp_removal_time_col].fillna(pd.Timestamp.max) > pd.to_datetime(dt_1))].copy()
    
        #--------------------------------------------------
        return mp_df
    
    
    def build_dovs_df(
        self, 
        mp_df                   = None, 
        dt_0                    = None, 
        dt_1                    = None, 
        enforce_dt              = True, 
        overlap_assertion       = 0.95, 
        df_mp_serial_number_col = 'mfr_devc_ser_nbr', 
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
            By default, self.date_range[0]/[1] are used.
            If user inputs values, the resulting range must encompass [self.date_range[0], self.date_range[1]]
                i.e., dt_0 <= self.date_range[0] and dt_1 >= self.date_range[1]
    
        overlap_assertion:
            Must be 'full', 'any', or a float between 0 and 1
        """
        #--------------------------------------------------
        assert(
            isinstance(overlap_assertion, float) or 
            overlap_assertion in ['full', 'any']
        )
        
        #--------------------------------------------------
        if dt_0 is None:
            dt_0 = self.date_range[0]
        if dt_1 is None:
            dt_1 = self.date_range[1]
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
        mp_df = mp_df[(mp_df[df_mp_install_time_col] <= pd.to_datetime(dt_0)) & 
                      (mp_df[df_mp_removal_time_col].fillna(pd.Timestamp.max) > pd.to_datetime(dt_1))].copy()
            
        #--------------------------------------------------
        if isinstance(overlap_assertion, float):
            assert(0 <= overlap_assertion <= 1)
            overlap_pct = len(set(self.trsf_pole_nbs).intersection(set(mp_df['trsf_pole_nb'].unique())))/len(self.trsf_pole_nbs)
            assert(overlap_pct >= overlap_assertion)
        else:
            if overlap_assertion == 'full':
                assert(len(set(self.trsf_pole_nbs).difference(set(mp_df['trsf_pole_nb'].unique())))==0)
            elif overlap_assertion == 'any':
                assert(len(set(self.trsf_pole_nbs).intersection(set(mp_df['trsf_pole_nb'].unique())))>0)
            else:
                assert(0)
                
        #--------------------------------------------------
        # Build dovs_df
        #-------------------------
        # There are instances where a user may actually prefer dt_0 to be greater than dt_1 for the mp_df functionality
        #   above (to be fully inclusive).
        # In such cases, dt_0,dt_1 need to be switched for dovs below.
        date_range = [np.min((dt_0,dt_1)), np.max((dt_0,dt_1))]
        #-------------------------
        dovs = DOVSOutages(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = None, 
            init_df_in_constructor    = True,
            build_sql_function        = DOVSOutages_SQL.build_sql_std_outage, 
            build_sql_function_kwargs = dict(
                premise_nbs     = mp_df['prem_nb'].unique().tolist(), 
                date_range      = date_range, 
                field_to_split  = 'premise_nbs', 
                include_premise = True
            ), 
            build_consolidated        = False
        )
        dovs_df = dovs.df
    
        #-------------------------
        # Add trsf_pole_nb info to dovs_df
        dovs_df = pd.merge(
            dovs_df, 
            mp_df[['prem_nb', 'trsf_pole_nb']].drop_duplicates(), 
            how      = 'left', 
            left_on  = 'PREMISE_NB', 
            right_on = 'prem_nb', 
        )
    
        #--------------------------------------------------
        return dovs_df
    

