#!/usr/bin/env python

r"""
Holds DOVSOutages class.  See DOVSOutages.DOVSOutages for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_dtype
import datetime
from natsort import natsorted
from ast import literal_eval
#--------------------------------------------------
from MeterPremise import MeterPremise
from DOVSOutages_SQL import DOVSOutages_SQL
from GenAn import GenAn
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
#--------------------------------------------------

class DOVSOutages(GenAn):
    r"""
    DOVSOutages class is intended to build and handle pd.DataFrame objects derived from the tables within DOVSADM.
    The pd.DataFrame object may have one row per outage event, or there may be multiple rows per outage event
      where the different rows for a given event contain the various premise numbers affected by the outage.
    """
    def __init__(
        self, 
        df_construct_type         = DFConstructType.kRunSqlQuery, 
        contstruct_df_args        = None, 
        init_df_in_constructor    = True, 
        build_sql_function        = None, 
        build_sql_function_kwargs = None, 
        save_args                 = False, 
        build_consolidated        = False, 
        **kwargs
    ):
        r"""
        if df_construct_type==DFConstructType.kReadCsv or DFConstructType.kReadCsv:
          contstruct_df_args needs to have at least 'file_path'
        if df_construct_type==DFConstructType.kRunSqlQuery:
          contstruct_df_args needs at least 'conn_db'   

        NOTE: build_consolidated essentially means build the dovs_df using build_sql_function and build_sql_function_kwargs, 
                then append on a column containing the premise numbers for each outage
        """
        #--------------------------------------------------
        # First, set self.build_sql_function and self.build_sql_function_kwargs
        # and call base class's __init__ method
        #---------------
        self.build_sql_function = (build_sql_function if build_sql_function is not None 
                                   else DOVSOutages_SQL.build_sql_std_outage)
        #---------------
        self.build_sql_function_kwargs = (build_sql_function_kwargs if build_sql_function_kwargs is not None 
                                          else {})
        #--------------------------------------------------
        if contstruct_df_args is None:
            contstruct_df_args = {}
        outg_rec_nb_col = self.build_sql_function_kwargs.get('outg_rec_nb_col', 'OUTG_REC_NB')
        premise_nb_col  = self.build_sql_function_kwargs.get('premise_nb_col', 'PREMISE_NB')
        contstruct_df_args['read_sql_args'] = contstruct_df_args.get(
            'read_sql_args', 
            dict(dtype={outg_rec_nb_col:np.int64, premise_nb_col:str})
        )
        #--------------------------------------------------
        if not build_consolidated:
            super().__init__(
                df_construct_type=df_construct_type, 
                contstruct_df_args=contstruct_df_args, 
                init_df_in_constructor=init_df_in_constructor, 
                build_sql_function=self.build_sql_function, 
                build_sql_function_kwargs=self.build_sql_function_kwargs, 
                save_args=save_args, 
                **kwargs
            )        
        else:
            # FOR NOW: This only works when running an SQL Query, so df_construct_type must be DFConstructType.kRunSqlQuery
            # Also, the DataFrame will be initialized via the build_consolidated_outage_df method, so init_in_df should be set
            # to False when evoking the super constructor.
            # The purpse of calling the super constructor at all in this case is simply to set attributes such as
            # self.read_sql_args, self.save_args, etc.
            if df_construct_type is None:
                df_construct_type = DFConstructType.kRunSqlQuery
            else:
                assert(df_construct_type == DFConstructType.kRunSqlQuery)
            #----------
            # If build_consolidated is True, it is expected that the DF will be built
            # But, again, it should be set to False in the super constructor.
            assert(init_df_in_constructor)
            #----------
            super().__init__(
                df_construct_type=df_construct_type, 
                contstruct_df_args=contstruct_df_args, 
                init_df_in_constructor=False, 
                build_sql_function=self.build_sql_function, 
                build_sql_function_kwargs=self.build_sql_function_kwargs, 
                save_args=save_args, 
                **kwargs
            )
            #----------
            self.df = DOVSOutages.build_consolidated_outage_df(
                contstruct_df_args=contstruct_df_args, 
                build_sql_function=self.build_sql_function, 
                build_sql_function_kwargs=self.build_sql_function_kwargs, 
                **kwargs
            )

        #--------------------------------------------------
        #TODO
        self.outg_mp_dict = {}

    #****************************************************************************************************
    def get_conn_db(self):
        return Utilities.get_utldb01p_oracle_connection()
    
    def set_df(
        self, 
        df
    ):
        r"""
        """
        #-------------------------
        self.df = df

        
    @staticmethod
    def get_dovs_conn_db():
        return Utilities.get_utldb01p_oracle_connection()
    
    def get_default_cols_and_types_to_convert_dict(self):
        cols_and_types_to_convert_dict = {
            'CI_NB':np.int64, 
            'CMI_NB':np.float64, 
            'OUTG_REC_NB':[np.int64, str], 
            'START_YEAR':np.int64, 
            'DT_ON_TS':datetime.datetime,
            'DT_OFF_TS_FULL':datetime.datetime
        }
        return cols_and_types_to_convert_dict
    
    def get_full_default_sort_by(self):
        full_default_sort_by = ['OUTG_REC_NB', 'PREMISE_NB']
        return full_default_sort_by
        
    #****************************************************************************************************
    @staticmethod
    def build_mjr_cause_cd_to_nm_dict():
        r"""
        ***** Likely one wants to use get_mjr_cause_cd_to_nm_dict instead of this function *****
        This function builds the mjr_cause_cd_to_nm_dict using the SQL database.
        This only really needs to be run if the codes have been changed for some reason.
        """
        #-------------------------
        sql = DOVSOutages_SQL.build_sql_DOVS_OUTAGE_CAUSE_TYPES_DIM()
        conn_db = DOVSOutages.get_dovs_conn_db()
        #-------------------------
        cause_types_df = pd.read_sql_query(sql.get_sql_statement(), conn_db) 
        mjr_cause_cd_to_nm_dict = list(cause_types_df.groupby(by=['MJR_CAUSE_CD', 'MJR_CAUSE_NM']).groups.keys())
        mjr_cause_cd_to_nm_dict = {cd:nm for cd,nm in mjr_cause_cd_to_nm_dict}
        #-------------------------
        return mjr_cause_cd_to_nm_dict

    @staticmethod
    def get_mjr_cause_cd_to_nm_dict():
        r"""
        This function simply grabs the default mjr_cause_cd_to_nm_dict stored below.
        If one suspects this has been updated/expanded in Athena/SQL, use build_mjr_cause_cd_to_nm_dict instead
        """
        mjr_cause_cd_to_nm_dict = {
            'DIS': 'DISTRIBUTION SOURCE',
            'DL': 'DISTRIBUTION LINE',
            'DS': 'DISTRIBUTION STATION',
            'G': 'GENERATION',
            'NI': 'NO INTERRUPTION',
            'PP': 'PARTIAL POWER',
            'ST': 'SUBTRANSMISSION LINE',
            'TL': 'TRANSMISSION LINE',
            'TS': 'TRANSMISSION STATION'
        }
        return mjr_cause_cd_to_nm_dict


    @staticmethod
    def build_mnr_cause_cd_to_nm_dict():
        r"""
        ***** Likely one wants to use get_mnr_cause_cd_to_nm_dict instead of this function *****
        This function builds the mnr_cause_cd_to_nm_dict using the SQL database.
        This only really needs to be run if the codes have been changed for some reason.
        """
        #-------------------------
        sql = DOVSOutages_SQL.build_sql_DOVS_OUTAGE_CAUSE_TYPES_DIM()
        conn_db = DOVSOutages.get_dovs_conn_db()
        #-------------------------
        cause_types_df = pd.read_sql_query(sql.get_sql_statement(), conn_db) 
        mnr_cause_cd_to_nm_dict = list(cause_types_df.groupby(by=['MNR_CAUSE_CD', 'MNR_CAUSE_NM']).groups.keys())
        mnr_cause_cd_to_nm_dict = {cd:nm for cd,nm in mnr_cause_cd_to_nm_dict}
        #-------------------------
        return mnr_cause_cd_to_nm_dict


    @staticmethod
    def get_mnr_cause_cd_to_nm_dict():
        r"""
        This function simply grabs the default mnr_cause_cd_to_nm_dict stored below.
        If one suspects this has been updated/expanded in Athena/SQL, use build_mnr_cause_cd_to_nm_dict instead
        """
        mnr_cause_cd_to_nm_dict = {
            'A': 'ANIMAL - NON BIRD',
            'AB': 'ANIMAL BUS',
            'ABI': 'ANIMAL - BIRD',
            'ABX': 'ANIMAL BUSHING XFMR',
            'AE': 'AEP EQUIPMENT - NO CUST OUT',
            'AF': 'ABNORMAL FEED',
            'AMI': 'No Cust Out - AMI Operation',
            'AO': 'ANIMAL - OTHER',
            'BE': 'BLAST/EXPLOSION(NON-AEP)',
            'C': 'NO CUST OUT - AEP CONDUCTOR',
            'CE': 'CUSTOMER EQUIPMENT, 1 CUSTOMER OUT',
            'CF': 'CONTAMINATION/FLASHOVER',
            'CO': 'CORROSION',
            'CP': 'NO CUST OUT - CATV OR PHONE CONDUCTOR',
            'CTC': 'TRIP CHARGE',
            'DNP': 'DNP',
            'DOT': 'DUPLICATE OUTAGE TICKET',
            'DR': 'DISCONNECT/RECONNECT',
            'EQF': 'EQUIPMENT FAILURE',
            'ERF': 'ERROR-FIELD',
            'ERO': 'ERROR-OPERATIONS',
            'F': 'FIRE-AEP, OR AFFECTING > 1 CUSTOMER',
            'FC': 'FIRE - CUSTOMER, 1 CUSTOMER OUT',
            'FO': 'FOREIGN OBJECT (NON ANIMAL)',
            'FW': 'FACILITATION OF WORK',
            'G': 'GENERATION',
            'GC': 'GALLOPING CONDUCTOR',
            'L': 'AEP - OUTDOOR/STREET LIGHTS',
            'LS': 'LOAD SHED',
            'MM': 'MAP MANIPULATION',
            'O': 'OTHER',
            'OL': 'OVERLOAD',
            'OU': 'OTHER UTILITY CUSTOMER OUT',
            'OV': 'OVERVOLTAGE',
            'PQ': 'POWER QUALITY (FLICKERING, DIM, BRIGHT LIGHTS ETC>)',
            'R': 'RELAY MIS-OPERATION',
            'SCO': 'SCHEDULED COMPANY',
            'SO': 'SCHEDULED OUTSIDE REQUEST > 1 CUSTOMER',
            'SS': 'SWITCHING SURGE',
            'TC': 'NO CUST OUT - TREE CONDITION',
            'TIN': 'TRANSMISSION INFORMATION NEEDED',
            'TIR': 'TREE INSIDE ROW',
            'TOR': 'TREE OUT OF ROW',
            'TR': 'TREE REMOVAL (NON AEP)',
            'U': 'UNKNOWN (NON WEATHER)',
            'UB': 'UNBALANCE',
            'UG': 'UG CONST. /DIG-INS (NON AEP)',
            'UL': 'UG LOCATE',
            'UT': 'UNNECESSARY TRIP',
            'V': 'VANDALISM',
            'VA': 'VEHICLE ACCIDENT (NON AEP)',
            'VIN': 'VINE',
            'WFS': 'WEATHER-FLOOD/SLIDE',
            'WH': 'WEATHER - HURRICANE',
            'WI': 'WEATHER - ICE (1/2 INCH OR > 6 " SNOW)',
            'WL': 'WEATHER - LIGHTNING',
            'WT': 'WEATHER - TORNADO',
            'WTI': 'WEATHER TREE INSIDE ROW',
            'WTO': 'WEATHER TREE OUTSIDE ROW',
            'WU': 'WEATHER - UNKNOWN',
            'WW': 'WEATHER - HIGH WINDS (EXCEEDING 60 MPH)'
        }
        return mnr_cause_cd_to_nm_dict
        
        
    @staticmethod
    def set_mjr_mnr_cause_nm_col(
        df, 
        mjr_mnr_cause_nm_col='MJR_MNR_CAUSE_NM', 
        set_null_to_NA=True, 
        mjr_cause_nm_col='MJR_CAUSE_NM', 
        mnr_cause_nm_col='MNR_CAUSE_NM', 
        mjr_cause_nm_abbr_dict=None
    ):
        r"""
        Combine the major and minor cause columns into a single major minor column

        Null values cause .agg('-'.join, axis=1) to crash, so set any null values equal to 'NA'
          If set_null_to_NA is False, these 'NA' values are only temporary, and will be converted
          back to null at the end of the function.
          Otherwise, the null values are permanently changed to 'NA'
        """
        #-------------------------
        df.loc[pd.isnull(df[mjr_cause_nm_col]), mjr_cause_nm_col] = 'NA'
        df.loc[pd.isnull(df[mnr_cause_nm_col]), mnr_cause_nm_col] = 'NA'
        df[mjr_mnr_cause_nm_col] = df[[mjr_cause_nm_col, mnr_cause_nm_col]].agg('-'.join, axis=1)

        # Substitute in any abbreviations for the major cause names
        #  e.g. 'DISTRIBUTION LINE' --> 'DL' using mjr_cause_nm_abbr_dict = {'DISTRIBUTION LINE':'DL'}
        if mjr_cause_nm_abbr_dict is not None:
            for mjr_cause in mjr_cause_nm_abbr_dict:
                df[mjr_mnr_cause_nm_col] = df[mjr_mnr_cause_nm_col].str.replace(mjr_cause, mjr_cause_nm_abbr_dict[mjr_cause])
        # If set_null_to_NA==False, set NA back to null values
        if not set_null_to_NA:    
            df.loc[df[mjr_cause_nm_col]=='NA', mjr_cause_nm_col] = np.nan
            df.loc[df[mnr_cause_nm_col]=='NA', mnr_cause_nm_col] = np.nan

        return df
    
    #****************************************************************************************************
    @staticmethod
    def build_consolidated_outage(
        contstruct_df_args=None, 
        build_sql_function=None, 
        build_sql_function_kwargs=None, 
        return_premise_nbs_col='premise_nbs', 
        addtnl_get_premise_nbs_for_outages_kwargs=None, 
        **kwargs
    ):
        r"""
        FOR NOW: This only works when running an SQL Query
        The consolidated DF contains a single row for each outage, but still contains all premise information
          in a column of the DF containing lists of premise numbers.

        It turns out that (i) is quicker than (ii)
          (i)  running SQL query without premises included, grabbing premises after, and then joining the two
          (ii) (a) running SQL query with premises and then (b) running consolidate_df_outage
        For method (ii), the main time-consuming process is (iia), i.e. letting SQL join the premise dimension.
          It probably doesn't help that the returned DF is still large, as it is not yet grouped by outage (hence
          the need for running consolidate_df_outage)
        """
        #-------------------------
        df_construct_type=DFConstructType.kRunSqlQuery
        init_df_in_constructor=True
        #-------------------------
        if build_sql_function is None:
            build_sql_function = DOVSOutages_SQL.build_sql_std_outage
        #-----
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs={}
        #-----
        if build_sql_function == DOVSOutages_SQL.build_sql_std_outage:
            build_sql_function_kwargs['include_premise']=False
            build_sql_function_kwargs['include_DOVS_PREMISE_DIM']=False
        else:
            build_sql_function_kwargs['include_DOVS_PREMISE_DIM']=False
        #-------------------------
        # Turn off verbose by default
        build_sql_function_kwargs['verbose'] = build_sql_function_kwargs.get('verbose', False)
        #-------------------------
        # !!!!!!!!!!!!!!! IMPORTANT !!!!!!!!!!!!!!!
        # Below, it is important (although counterintuitive) for build_consolidated to be set to False
        # The purpose of build_consolidated in the constructor is to direct the code here, not to direct the
        # code within here to do anything.  If build_consolidated were set to True here, THE CODE WOULD
        # ENTER INTO AN INFINITE LOOP!!!!!!!!!!
        dovs_outgs = DOVSOutages(
            df_construct_type=df_construct_type, 
            contstruct_df_args=contstruct_df_args, 
            init_df_in_constructor=init_df_in_constructor,
            build_sql_function=build_sql_function, 
            build_sql_function_kwargs=build_sql_function_kwargs, 
            build_consolidated=False, 
            **kwargs
        )
        dovs_outgs_df = dovs_outgs.get_df()
        outg_rec_nb_col = build_sql_function_kwargs.get('outg_rec_nb_col', 'OUTG_REC_NB')
        #-------------------------
        get_premise_nbs_for_outages_kwargs = dict(
            outg_rec_nbs=dovs_outgs_df[outg_rec_nb_col].tolist(), 
            return_type=pd.Series, 
            verbose=build_sql_function_kwargs['verbose']
        )
        #-----
        if addtnl_get_premise_nbs_for_outages_kwargs is not None:
            get_premise_nbs_for_outages_kwargs = {**addtnl_get_premise_nbs_for_outages_kwargs, 
                                                  **get_premise_nbs_for_outages_kwargs}
        #-----
        premise_nbs_series = DOVSOutages.get_premise_nbs_for_outages(
            **get_premise_nbs_for_outages_kwargs
        )
        #-------------------------
        # For proper merge to take place, the data types should align for the outg_rec_nb
        #   e.g., should both be strings, or should both be ints, etc.
        assert(dovs_outgs_df[outg_rec_nb_col].dtype==premise_nbs_series.index.dtype)
        #-------------------------
        dovs_outgs_df=dovs_outgs_df.set_index(outg_rec_nb_col)
        dovs_outgs_df = dovs_outgs_df.merge(premise_nbs_series, left_index=True, right_index=True)
        #-------------------------
        dovs_outgs.df = dovs_outgs_df
        #-------------------------
        return dovs_outgs

    @staticmethod
    def build_consolidated_outage_df(
        contstruct_df_args=None, 
        build_sql_function=None, 
        build_sql_function_kwargs=None, 
        **kwargs
    ):
        r"""
        See build_consolidated_outage for more information
        """
        dovs_outgs = DOVSOutages.build_consolidated_outage(
            contstruct_df_args=contstruct_df_args, 
            build_sql_function=build_sql_function, 
            build_sql_function_kwargs=build_sql_function_kwargs, 
            **kwargs
        )
        return dovs_outgs.get_df()

    
    #****************************************************************************************************
    @staticmethod
    def set_search_time_in_outage_df(
        df_outage          , 
        search_time_window , 
        power_out_col      = 'DT_OFF_TS_FULL',          #input column
        power_on_col       = 'DT_ON_TS',                #input column
        t_search_min_col   = 't_search_min',            #output column
        t_search_max_col   = 't_search_max',            #output column
        wrt_out_only       = False
    ):
        r"""
        The function creates the search time intervals which will be used to retrieve the usage (15-minute interval 
          and instantaneous), end events, etc. for the meters affected by the outage.
        The expectation is that the search window will be centered around the outage; i.e., the window creation is as follows:
            df_outage[t_search_min_col] = df_outage[power_out_col] - search_time_window
            df_outage[t_search_max_col] = df_outage[power_on_col]  + search_time_window

        df_outage:
          - pd.DataFrame object housing the outage data

        search_time_window:
          - the amount of time for which to retrieve data before and after the outage.
          - Can be 
              - an int representing the number of MINUTES
              - a datetime.timedelta object
            -OR- (new development)
              - a list of length-2 representing the left and right Timedeltas (or the columns in which 
                  the Timedeltas live) to be used in defining the window.
                IF THIS IS THE CASE, the user needs to be careful to define td_left and td_right
                  with the appropriate signs (+/-) for the desired output.
                See Utilities_df.build_range_from_doi_left_right/Utilities_dt.build_range_from_doi_left_right for more information.

        power_out_col:
          - the column in df_outage housing the time the power outage began
          - default = 'DT_OFF_TS_FULL'

        power_on_col:
          - the column in df_outage housing the time the power outage ended
          - default = 'DT_ON_TS'

        t_search_min_col:
          - the column to be created in df_outage to house the minimum of the search time interval
          - default = 't_search_min'

        t_search_max_col:
          - the column to be created in df_outage to house the maximum of the search time interval
          - default = 't_search_max'

        wrt_out_only:
            If True, the search times are taken with respect to (wrt) the power_out time only 
        """
        #--------------------------------------------------
        if wrt_out_only:
            power_on_col = power_out_col
        #--------------------------------------------------
        assert(power_out_col in df_outage.columns)
        assert(power_on_col in df_outage.columns)
        #--------------------------------------------------
        if not is_datetime64_dtype(df_outage[power_out_col]):
            df_outage = Utilities_df.convert_col_type(df_outage, power_out_col, datetime.datetime)
        if not is_datetime64_dtype(df_outage[power_on_col]):
            df_outage = Utilities_df.convert_col_type(df_outage, power_on_col, datetime.datetime)
        #--------------------------------------------------
        if isinstance(search_time_window, list):
            assert(len(search_time_window)==2)
            assert(Utilities.are_all_list_elements_one_of_types_and_homogeneous(search_time_window, [int, datetime.timedelta, str]))
            if Utilities.are_all_list_elements_of_type(search_time_window, str):
                assert(set(search_time_window).difference(set(df_outage.columns.tolist()))==set())
            else:
                search_time_window = [
                    x if isinstance(x, datetime.timedelta) else datetime.timedelta(minutes=x) 
                    for x in search_time_window
                ]
                assert(Utilities.are_all_list_elements_of_type(search_time_window, datetime.timedelta))
            #--------------------------------------------------
            df_outage = Utilities_df.build_range_from_doi_left_right(
                df                 = df_outage, 
                doi_col            = None,
                td_left            = search_time_window[0],
                td_right           = search_time_window[1], 
                placement_cols     = [t_search_min_col, t_search_max_col], 
                doi_col_left       = power_out_col, 
                doi_col_right      = power_on_col, 
                assert_makes_sense = True , 
            )
        else:
            assert(Utilities.is_object_one_of_types(search_time_window, [int, datetime.timedelta]))
            if isinstance(search_time_window, int):
                search_time_window = datetime.timedelta(minutes=search_time_window)
            #--------------------------------------------------
            df_outage[t_search_min_col] = df_outage[power_out_col] - search_time_window
            df_outage[t_search_max_col] = df_outage[power_on_col]  + search_time_window
        #--------------------------------------------------
        return df_outage
        
        
    def set_search_time(
        self               , 
        search_time_window , 
        power_out_col      = 'DT_OFF_TS_FULL', 
        power_on_col       = 'DT_ON_TS', 
        t_search_min_col   = 't_search_min', 
        t_search_max_col   = 't_search_max', 
        wrt_out_only       = False
    ):
        self.df = DOVSOutages.set_search_time_in_outage_df(
            df_outage          = self.df, 
            search_time_window = search_time_window, 
            power_out_col      = power_out_col, 
            power_on_col       = power_on_col, 
            t_search_min_col   = t_search_min_col, 
            t_search_max_col   = t_search_max_col, 
            wrt_out_only       = wrt_out_only
        )
    
    #****************************************************************************************************
    @staticmethod
    def consolidate_df_outage_OLD(
        df_outage, 
        outg_rec_nb_col='OUTG_REC_NB', 
        premise_nb_col='PREMISE_NB', 
        premise_nbs_col='PREMISE_NBS', 
        cols_to_drop=['OFF_TM', 'REST_TM'], 
        drop_null_premise_nbs=True, 
        set_outg_rec_nb_as_index=True,
        drop_outg_rec_nb_if_index=False
    ):
        # A given outage in df_outage contains multiple rows
        # These rows only differ by the premise number.
        # Here, we consolidate each outage into a single row, where premise_nb_col
        #   now holds a list of all premise numbers
        #
        # Only duplicates come from DOVS_PREMISE_DIM
        # NOTE: 'OFF_TM', 'REST_TM' could possibly be useful, but they aren't always constant for all
        #       meters in an outage (mainly, I believe, some entries have e.g. REST_TM= NaT as no restored time
        #       was input)
        #
        # I oftentimes refer to the df returned by this function as df_slim
        if cols_to_drop is not None:
            df_outage = df_outage.drop(columns=cols_to_drop)

        if drop_null_premise_nbs:
            df_outage = df_outage.dropna(subset=[premise_nb_col])
            
        return_df = df_outage.groupby(outg_rec_nb_col, as_index=False).agg('first')

        list_of_prem_nbs_df = df_outage.groupby(outg_rec_nb_col).apply(lambda x: x[premise_nb_col].tolist())
        list_of_prem_nbs_df = list_of_prem_nbs_df.to_frame(name=premise_nbs_col).reset_index()

        assert(return_df.shape[0]==list_of_prem_nbs_df.shape[0])
        assert(return_df.shape[0]==return_df[outg_rec_nb_col].nunique())

        return_df = return_df.merge(list_of_prem_nbs_df, how='inner', left_on=outg_rec_nb_col, right_on=outg_rec_nb_col)
        return_df = return_df.drop(columns=[premise_nb_col])
        
        if set_outg_rec_nb_as_index:
            return_df.set_index(outg_rec_nb_col, drop=False, inplace=True)
            return_df.index.name='idx'
            if drop_outg_rec_nb_if_index:
                return_df=return_df.drop(columns=[outg_rec_nb_col])
                return_df.index.name=outg_rec_nb_col
        
        return return_df
        
    @staticmethod
    def find_dovs_non_prem_cols_in_df(df):
        r"""
        Find the columns in df which come from DOVS, but not the premise dimension.
        This is useful in consolidate_df_outage when the user does not supply a cols_shared_by_group argument.
        NOTE: df can be the DataFrame itself, or a list of the columns
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(df, [pd.DataFrame, list]))
        #-------------------------
        if isinstance(df, pd.DataFrame):
            df_cols = df.columns.tolist()
        else:
            df_cols = df
        #-------------------------
        dovs_non_prem_cols = list(set(
            TableInfos.DOVS_CLEARING_DEVICE_DIM_TI.columns_full    +
            TableInfos.DOVS_EQUIPMENT_TYPES_DIM_TI.columns_full    +
            TableInfos.DOVS_MASTER_GEO_DIM_TI.columns_full         +
            TableInfos.DOVS_OUTAGE_ATTRIBUTES_DIM_TI.columns_full  +
            TableInfos.DOVS_OUTAGE_CAUSE_TYPES_DIM_TI.columns_full + 
            TableInfos.DOVS_OUTAGE_FACT_TI.columns_full
        ))
        # There are a few typical aliases I use for DOVS columns (SHORT_NM_CLR_DEV, SHORT_NM_EQP_TYP) and 
        #   standard additional column (DT_OFF_TS_FULL, START_YEAR)
        #   which must be added by hand to the list above
        dovs_non_prem_cols.extend(['DT_OFF_TS_FULL', 'START_YEAR', 'SHORT_NM_CLR_DEV', 'SHORT_NM_EQP_TYP'])
        #-------------------------
        dovs_non_prem_cols_in_df = list(set(df_cols).intersection(set(dovs_non_prem_cols)))
        return dovs_non_prem_cols_in_df
        
        
    @staticmethod
    def consolidate_df_outage(
        df_outage                    , 
        outg_rec_nb_col              = 'OUTG_REC_NB', 
        addtnl_grpby_cols            = None, 
        cols_shared_by_group         = None, 
        cols_to_collect_in_lists     = None, 
        allow_duplicates_in_lists    = False, 
        allow_NaNs_in_lists          = False, 
        recover_uniqueness_violators = True, 
        gpby_dropna                  = False, 
        rename_cols                  = None,     
        premise_nb_col               = 'PREMISE_NB', 
        premise_nbs_col              = 'PREMISE_NBS', 
        cols_to_drop                 = ['OFF_TM', 'REST_TM'], 
        sort_PNs                     = True, 
        drop_null_premise_nbs        = True, 
        set_outg_rec_nb_as_index     = True,
        drop_outg_rec_nb_if_index    = False, 
        verbose                      = True
    ):
        r"""
        This function consolidates an outage DF which has a lot of repetitive information.
        I oftentimes refer to the df returned by this function as df_slim
        -----
        A given outage in df_outage contains multiple rows.
        In the simplest case, where data are only from DOVS, these rows only differ by the premise number (and other
          premise specific fields, e.g., 'OFF_TM' and 'REST_TM').
        In this case, the desire is to consolidate each outage into a single row, where premise_nb_col
          now holds a list of all premise numbers
        -----
        Since the original development of the method, more complicated DF structures are common, which require a complete
          re-working of the method.
        The method now relies mainly on the Utilities_df.consolidate_df method (the original consolidate_df_outage method
          preceded, and inspired, the Utilities_df.consolidate_df method).
        Instead of only grouping by outg_rec_nb_col, the user may implement additional columns by which to group.
            This is useful, e.g., when one wants to group by outages and trsf_pole_nbs.
        -----
        cols_shared_by_group:
            Columns which are shared by each group.
            If cols_shared_by_group is None, it will be set using find_dovs_non_prem_cols_in_df
            It is expected, and will be enforced, that, for each group, each of these columns has a single unique value.
            If multiple unique values exist for a group, the group will be ignored in the final result.

        cols_to_collect_in_lists:
            Columns which are not shared by each group, and whose values will be collected in list elements.
            premise_nb_col is always included in cols_to_collect_in_lists.
            If cols_to_collect_in_lists is None, it will be set to the collection of columns 
              in df_outage excluding outg_rec_nb_col + addtnl_grpby_cols + cols_shared_by_group + cols_to_drop

        """
        #--------------------------------------------------
        if cols_to_drop is not None:
            return_df = df_outage.drop(columns=cols_to_drop)
        else:
            return_df = df_outage.copy()
        #-------------------------
        if drop_null_premise_nbs:
            return_df = return_df.dropna(subset=[premise_nb_col])
        #--------------------------------------------------
        groupby_cols = [outg_rec_nb_col]
        if addtnl_grpby_cols is not None:
            if not isinstance(addtnl_grpby_cols, list):
                addtnl_grpby_cols = [addtnl_grpby_cols]
            groupby_cols.extend(addtnl_grpby_cols)
        #-------------------------
        if cols_shared_by_group is None or len(cols_shared_by_group)==0:
            cols_shared_by_group = DOVSOutages.find_dovs_non_prem_cols_in_df(return_df)
        assert(Utilities.is_object_one_of_types(cols_shared_by_group, [list, str, int]))
        if not isinstance(cols_shared_by_group, list):
            cols_shared_by_group=[cols_shared_by_group]
        #-------------------------
        if cols_to_collect_in_lists is None:
            cols_to_collect_in_lists = [x for x in return_df.columns.tolist() if x not in groupby_cols+cols_shared_by_group]
        assert(Utilities.is_object_one_of_types(cols_to_collect_in_lists, [list, str, int]))
        if not isinstance(cols_to_collect_in_lists, list):
            cols_to_collect_in_lists=[cols_to_collect_in_lists]
        if premise_nb_col not in cols_to_collect_in_lists:
            cols_to_collect_in_lists.append(premise_nb_col)
        #-------------------------
        if rename_cols is None:
            rename_cols = {premise_nb_col : premise_nbs_col}
        else:
            assert(isinstance(rename_cols, dict))
            rename_cols[premise_nb_col] = premise_nbs_col
        #--------------------------------------------------
        return_df = Utilities_df.consolidate_df(
            df=return_df, 
            groupby_cols=groupby_cols, 
            cols_shared_by_group=cols_shared_by_group, 
            cols_to_collect_in_lists=cols_to_collect_in_lists, 
            as_index=False, 
            include_groupby_cols_in_output_cols=False, 
            allow_duplicates_in_lists=allow_duplicates_in_lists, 
            allow_NaNs_in_lists=allow_NaNs_in_lists, 
            recover_uniqueness_violators=recover_uniqueness_violators, 
            gpby_dropna=gpby_dropna, 
            rename_cols=rename_cols, 
            verbose=verbose
        )
        #-------------------------
        if sort_PNs:
            return_df[premise_nbs_col] = return_df[premise_nbs_col].apply(lambda x: natsorted(x)) 
        #-------------------------
        if set_outg_rec_nb_as_index:
            return_df.set_index(outg_rec_nb_col, drop=False, inplace=True)
            return_df.index.name='idx'
            if drop_outg_rec_nb_if_index:
                return_df=return_df.drop(columns=[outg_rec_nb_col])
                return_df.index.name=outg_rec_nb_col    
        #-------------------------
        return return_df        
        
        
    @staticmethod
    def get_prem_nbs_from_consolidated_df_outage(df_outage_slim, premise_nbs_col='PREMISE_NBS', unique=True):
        #TODO PROBABLY NEED TO IMPLEMENT A MORE CAREFUL PROCEDURE HERE
        # If, for instance, there are NaNs in the lists of premise numbers, these are not so easily removed
        #   as the concatenate procedure makes them 'nan' strings (at least, when the premise numbers are also
        #   strings, I'm not sure what happens when the premise numbers are floats or ints)
        # However, the default in consolidate_df_outage is to drop null premise numbers, so this
        #   isn't a huge issue and can be handled more carefully at a later date.
        prem_nbs = np.concatenate(df_outage_slim[premise_nbs_col].values).tolist()
        if unique:
            prem_nbs = list(set(prem_nbs))
        return prem_nbs
        
    @staticmethod
    def read_df_outage_slim_from_csv(
        file_path, 
        premise_nbs_col='PREMISE_NBS', 
        dt_off_ts_col='DT_OFF_TS', 
        cols_and_types_to_convert_dict = {
            'CI_NB':np.int64, 
            'CMI_NB':np.float64, 
            'OUTG_REC_NB':np.int64, 
            'annual_kwh':float, 
            'annual_max_dmnd':float
        }, 
        outg_rec_nb_col='OUTG_REC_NB', 
        set_outg_rec_nb_as_index=True        
    ):
        r"""
        When stored as a CSV, PREMISE_NBS are retrieved as a string, instead of as a list
        i.e.,      "['078595203', '070885203', '074585203', '077685203']" 
         instead of ['078595203', '070885203', '074585203', '077685203']
        Using ast.literal_eval fixes that

        DT_OFF_TS is typically just a date, not a full datetime.
        Reading from CSV brings back full datetime (i.e., '2021-01-01 00:00:00' instead of '2021-01-01')
        Calling pd.to_datetime fixes that
        """
        #-------------------------
        df_outage_slim = pd.read_csv(file_path, dtype=str)
        df_outage_slim[premise_nbs_col] = df_outage_slim[premise_nbs_col].apply(lambda x: literal_eval(x))
        if dt_off_ts_col:
            df_outage_slim[dt_off_ts_col]=pd.to_datetime(df_outage_slim[dt_off_ts_col])
        if cols_and_types_to_convert_dict:
            # In cols_and_types_to_convert_dict, keep only entries contained in df_outage_slim to prevent error
            #   when more conversions present in cols_and_types_to_convert_dict than in DF
            cols_and_types_to_convert_dict = {k:v for k,v in cols_and_types_to_convert_dict.items() 
                                              if k in df_outage_slim.columns.tolist()}
            df_outage_slim = Utilities_df.convert_col_types(df_outage_slim, cols_and_types_to_convert_dict)
            
        if set_outg_rec_nb_as_index:
            df_outage_slim.set_index(outg_rec_nb_col, drop=False, inplace=True)
            df_outage_slim.index.name='idx'
            
        return df_outage_slim
        
    #****************************************************************************************************
    @staticmethod
    def build_mp_for_outgs(
        df_outage, 
        cols_of_interest, 
        premise_nb_col='PREMISE_NB', 
        df_construct_type=DFConstructType.kRunSqlQuery, 
        build_sql_function=MeterPremise.build_sql_meter_premise, 
        addtnl_build_sql_function_kwargs={}, 
        max_n_prem_nbs=10000
    ):
        r"""
        max_n_prem_nbs
            If the number of premise numbers is very large, it is actually faster to grab the entire
            MeterPremise database and then select the desired subset using the pd.DataFrame object.
            The purpose of the max_n_prem_nbs variable is to set the threshold above which the entire database 
            will be grabbed.
        """
        #-------------------------
        assert(premise_nb_col in df_outage)
        #-------------------------
        premise_nbs=df_outage['PREMISE_NB'].unique().tolist()
        build_sql_function_kwargs=dict(
            cols_of_interest=cols_of_interest, 
            premise_nbs=natsorted(premise_nbs)
        )
        build_sql_function_kwargs = {**build_sql_function_kwargs, 
                                     **addtnl_build_sql_function_kwargs}
        #-------------------------
        mp_for_outgs = MeterPremise(
            df_construct_type=df_construct_type, 
            build_sql_function=build_sql_function, 
            build_sql_function_kwargs=build_sql_function_kwargs, 
            init_df_in_constructor=True, 
            max_n_prem_nbs=max_n_prem_nbs
        )
        #-------------------------
        return mp_for_outgs

    @staticmethod
    def build_mp_df_for_outgs(
        df_outage, 
        cols_of_interest, 
        premise_nb_col='PREMISE_NB', 
        df_construct_type=DFConstructType.kRunSqlQuery, 
        build_sql_function=MeterPremise.build_sql_meter_premise, 
        addtnl_build_sql_function_kwargs={}, 
        max_n_prem_nbs=10000
    ):
        r"""
        NOTE: One may use the non-static version build_mp_df
        """
        #-------------------------
        mp_for_outgs = DOVSOutages.build_mp_for_outgs(
            df_outage=df_outage, 
            cols_of_interest=cols_of_interest, 
            premise_nb_col=premise_nb_col, 
            df_construct_type=df_construct_type, 
            build_sql_function=build_sql_function, 
            addtnl_build_sql_function_kwargs=addtnl_build_sql_function_kwargs, 
            max_n_prem_nbs=max_n_prem_nbs
        )
        #-------------------------
        return mp_for_outgs.get_df()
        
    def build_mp_df(
        self, 
        cols_of_interest, 
        premise_nb_col='PREMISE_NB', 
        df_construct_type=DFConstructType.kRunSqlQuery, 
        build_sql_function=MeterPremise.build_sql_meter_premise, 
        addtnl_build_sql_function_kwargs={}
    ):
        #-------------------------
        return DOVSOutages.build_mp_df_for_outgs(
            df_outage=self.df, 
            cols_of_interest=cols_of_interest, 
            premise_nb_col=premise_nb_col, 
            df_construct_type=df_construct_type, 
            build_sql_function=build_sql_function, 
            addtnl_build_sql_function_kwargs=addtnl_build_sql_function_kwargs    
        )
                
    @staticmethod
    def merge_df_outage_with_mp(
        df_outage, 
        df_mp, 
        merge_on_outg='PREMISE_NB', 
        merge_on_mp='prem_nb', 
        cols_to_include_mp=None, 
        drop_cols = ['prem_nb'], 
        rename_cols={'mfr_devc_ser_nbr':'serial_number'}, 
        how='left', 
        inplace=True
    ):
        r"""
        NOTE: One may use the non-static version merge_df_with_mp
        
        If cols_to_include_mp is None, all columns in df_mp are included in the merge
        """
        #-------------------------
        if not inplace:
            df_outage = df_outage.copy()
        #-------------------------
        if cols_to_include_mp is None:
            df_mp_to_merge = df_mp
        else:
            assert(Utilities.is_object_one_of_types(cols_to_include_mp, [list, tuple]))
            df_mp_to_merge = df_mp[cols_to_include_mp]
        #-------------------------
        # For proper merge to take place, the data types should align for the outg_rec_nb
        #   e.g., should both be strings, or should both be ints, etc.
        if isinstance(merge_on_outg, list):
            assert(isinstance(merge_on_mp, list) and len(merge_on_outg)==len(merge_on_mp))
            for i_merge in range(len(merge_on_outg)):
                if df_outage[merge_on_outg[i_merge]].dtype!=df_mp_to_merge[merge_on_mp[i_merge]].dtype:
                    df_mp_to_merge = Utilities_df.convert_col_type(
                        df=df_mp_to_merge, 
                        column=merge_on_mp[i_merge], 
                        to_type=df_outage[merge_on_outg[i_merge]].dtype, 
                        to_numeric_errors='coerce', 
                        inplace=True
                    )
                assert(df_outage[merge_on_outg[i_merge]].dtype==df_mp_to_merge[merge_on_mp[i_merge]].dtype)
        else:
            if df_outage[merge_on_outg].dtype!=df_mp_to_merge[merge_on_mp].dtype:
                df_mp_to_merge = Utilities_df.convert_col_type(
                    df=df_mp_to_merge, 
                    column=merge_on_mp, 
                    to_type=df_outage[merge_on_outg].dtype, 
                    to_numeric_errors='coerce', 
                    inplace=True
                )
            assert(df_outage[merge_on_outg].dtype==df_mp_to_merge[merge_on_mp].dtype)
        #-----
        df_outage = df_outage.merge(df_mp_to_merge, how=how, left_on=merge_on_outg, right_on=merge_on_mp)
        if drop_cols is not None:
            df_outage = df_outage.drop(columns=drop_cols)
        if rename_cols is not None:
            df_outage = df_outage.rename(columns=rename_cols)
        #-------------------------
        return df_outage
        
    @staticmethod
    def build_mp_df_and_merge_with_df_outage(
        df_outage, 
        cols_of_interest_met_prem, 
        build_mp_df_args = dict(
            premise_nb_col='PREMISE_NB', 
            df_construct_type=DFConstructType.kRunSqlQuery, 
            build_sql_function=MeterPremise.build_sql_meter_premise, 
            addtnl_build_sql_function_kwargs={}
        ), 
        merge_on_outg='PREMISE_NB', 
        merge_on_mp='prem_nb', 
        cols_to_include_mp=None, 
        drop_cols = ['prem_nb'], 
        rename_cols={'mfr_devc_ser_nbr':'serial_number'}, 
        how='left', 
        inplace=True
    ):
        r"""
        NOTE: One may use the non-static version merge_df_with_mp
        
        If cols_to_include_mp is None, it is set to cols_of_interest_met_prem
        """
        #-------------------------
        df_mp = DOVSOutages.build_mp_df_for_outgs(
            df_outage=df_outage, 
            cols_of_interest=cols_of_interest_met_prem, 
            **build_mp_df_args
        )
        if cols_to_include_mp is None:
            cols_to_include_mp = cols_of_interest_met_prem
        return DOVSOutages.merge_df_outage_with_mp(
            df_outage=df_outage, 
            df_mp=df_mp, 
            merge_on_outg=merge_on_outg, 
            merge_on_mp=merge_on_mp, 
            cols_to_include_mp=cols_to_include_mp, 
            drop_cols = drop_cols, 
            rename_cols=rename_cols, 
            how=how, 
            inplace=inplace
        )

    def merge_df_with_mp(
        self, 
        cols_of_interest_met_prem, 
        build_mp_df_args = dict(
            premise_nb_col='PREMISE_NB', 
            df_construct_type=DFConstructType.kRunSqlQuery, 
            build_sql_function=MeterPremise.build_sql_meter_premise, 
            addtnl_build_sql_function_kwargs={}
        ), 
        merge_on_outg='PREMISE_NB', 
        merge_on_mp='prem_nb', 
        cols_to_include_mp=['mfr_devc_ser_nbr', 'prem_nb'], 
        drop_cols = ['prem_nb'], 
        rename_cols={'mfr_devc_ser_nbr':'serial_number'}, 
        how='left', 
    ):
        r"""        
        If cols_to_include_mp is None, it is set to cols_of_interest_met_prem
        """
        #-------------------------
        df_mp = self.build_mp_df(
            cols_of_interest=cols_of_interest_met_prem, 
            **build_mp_df_args
        )
        if cols_to_include_mp is None:
            cols_to_include_mp = cols_of_interest_met_prem
        self.df = DOVSOutages.merge_df_outage_with_mp(
            df_outage=self.df, 
            df_mp=df_mp, 
            merge_on_outg=merge_on_outg, 
            merge_on_mp=merge_on_mp, 
            cols_to_include_mp=cols_to_include_mp, 
            drop_cols = drop_cols, 
            rename_cols=rename_cols, 
            how=how, 
            inplace=True
        )
    
    #****************************************************************************************************
    @staticmethod
    def get_top_n_outage_mjr_mnr_causes_dfs(df_outage, cols_for_drop_duplicates, 
                                            top_outage_metrics='cmi', top_n=None, 
                                            mjr_cause_col='MJR_CAUSE_CD', mnr_cause_col='MNR_CAUSE_CD', 
                                            cmi_nb_col='CMI_NB', ci_nb_col='CI_NB'):
        # metrics included in top_outage_metrics must equal 'cmi', 'ci', or 'n_outages'
        # If a single metric is included in top_outage_metrics, a single df will be returned
        # If more than one metric included in top_outage_metrics, a dict of dfs will be returned with key
        #   values equal to the metrics in top_outage_metrics. 
        # If top_n is None, all will be returned
        #   Otherwise, only top_5 causes will be returned
        #-------------------------
        df_outage_slim = df_outage.drop_duplicates(subset=cols_for_drop_duplicates)
        #-------------------------    
        if isinstance(top_outage_metrics, str):
            top_outage_metrics = [top_outage_metrics]
        expected_top_outage_metrics = ['cmi', 'ci', 'n_outages']
        assert(all(x in expected_top_outage_metrics for x in top_outage_metrics))  
        top_outage_metrics = top_outage_metrics.copy()
        #-------------------------
        metric_to_cols_dict = {'cmi':cmi_nb_col, 'ci':ci_nb_col}
        #-------------------------
        include_n_outages = False
        if 'n_outages' in top_outage_metrics:
            include_n_outages = True
            top_outage_metrics.pop(top_outage_metrics.index('n_outages'))
        #-------------------------
        agg_dict = {}
        for i,top_outage_metric in enumerate(top_outage_metrics):
            assert(top_outage_metric not in agg_dict)
            agg_dict[metric_to_cols_dict[top_outage_metric]] = ['sum']
        if include_n_outages:
            if len(agg_dict)>0:
                # To calculate n_outages, doesn't matter if 'cmi' or 'ci' used
                agg_dict[list(agg_dict.keys())[0]].append('count')
            else:
                # If neither of these are already included in agg_dict 
                #   (i.e., only calculating n_outages), use 'cmi'
                agg_dict[metric_to_cols_dict['cmi']] = ['count']

        #-------------------------
        outg_by_cause_counts_df = df_outage_slim.groupby([mjr_cause_col, mnr_cause_col]).agg(agg_dict)
        #-------------------------
        outg_by_cause_counts_dfs_by_metric = {}
        for top_outage_metric in top_outage_metrics:
            assert(top_outage_metric not in outg_by_cause_counts_dfs_by_metric)
            df_i = outg_by_cause_counts_df[(metric_to_cols_dict[top_outage_metric], 'sum')].sort_values(ascending=False).to_frame()
            df_i.columns = df_i.columns.to_flat_index()
            df_i.index = df_i.index.to_flat_index()
            if top_n is not None:
                assert(top_n < df_i.shape[0])
                df_i = df_i.iloc[:top_n]
            outg_by_cause_counts_dfs_by_metric[top_outage_metric] = df_i
        if include_n_outages:
            n_outages_col = [x for x in outg_by_cause_counts_df.columns if x[1]=='count']
            assert(len(n_outages_col)==1)
            n_outages_col = n_outages_col[0]
            df_i = outg_by_cause_counts_df[n_outages_col].sort_values(ascending=False).to_frame()
            df_i.columns = df_i.columns.to_flat_index()
            df_i.index = df_i.index.to_flat_index()
            if top_n is not None:
                assert(top_n < df_i.shape[0])
                df_i = df_i.iloc[:top_n]
            outg_by_cause_counts_dfs_by_metric['n_outages'] = df_i
        #-------------------------
        if len(outg_by_cause_counts_dfs_by_metric)==1:
            return outg_by_cause_counts_dfs_by_metric[list(outg_by_cause_counts_dfs_by_metric.keys())[0]]
        else:
            return outg_by_cause_counts_dfs_by_metric

    @staticmethod    
    def get_top_n_outage_mjr_mnr_causes(df_outage, cols_for_drop_duplicates, 
                                        top_outage_metrics='cmi', top_n=None, 
                                        mjr_cause_col='MJR_CAUSE_CD', mnr_cause_col='MNR_CAUSE_CD', 
                                        cmi_nb_col='CMI_NB', ci_nb_col='CI_NB'):
        outg_by_cause_counts_dfs_by_metric = DOVSOutages.get_top_n_outage_mjr_mnr_causes_dfs(df_outage=df_outage, 
                                                                                 cols_for_drop_duplicates=cols_for_drop_duplicates, 
                                                                                 top_outage_metrics=top_outage_metrics, top_n=top_n, 
                                                                                 mjr_cause_col=mjr_cause_col, mnr_cause_col=mnr_cause_col, 
                                                                                 cmi_nb_col=cmi_nb_col, ci_nb_col=ci_nb_col)
        if isinstance(outg_by_cause_counts_dfs_by_metric, pd.DataFrame):
            return outg_by_cause_counts_dfs_by_metric.index.tolist()
        else:
            assert(isinstance(outg_by_cause_counts_dfs_by_metric, dict))
            return_dict = {}
            for metric, df in outg_by_cause_counts_dfs_by_metric.items():
                assert(metric not in return_dict)
                return_dict[metric] = df.index.tolist()
            return return_dict

    @staticmethod
    def build_top_5_df_outages_subsets(top5_mjr_mnr_causes, df_outage):
        top5_df_outages = {}
        top5_df_outages_by_xfmr = {}
        for mjr_mnr_cause in top5_mjr_mnr_causes:
            assert(mjr_mnr_cause not in top5_df_outages)
            assert(mjr_mnr_cause not in top5_df_outages_by_xfmr)
            #-----
            top5_df_outages[mjr_mnr_cause] = df_outage[(df_outage['MJR_CAUSE_CD']==mjr_mnr_cause[0]) & 
                                                       (df_outage['MNR_CAUSE_CD']==mjr_mnr_cause[1])].copy()
            top5_df_outages_by_xfmr[mjr_mnr_cause] = top5_df_outages[mjr_mnr_cause].groupby('trsf_pole_nb')[['annual_kwh', 'annual_max_dmnd']].sum()
        return (top5_df_outages, top5_df_outages_by_xfmr)

    
    @staticmethod    
    def remove_trsf_pole_nb_without_numeric_digit(df, trsf_pole_nb_col='trsf_pole_nb'):
        # eliminates, e.g. ['TRANSMISSION', 'NETWORK', ' ', 'PRIMARY', 'IDONOTKNOWXX','HJKHKJHKJ']
        # Works for a list of dfs too
        #---------------------------------------------------
        if isinstance(df, list) or isinstance(df, tuple):
            dfs = []
            for df_i in df:
                df_i = DOVSOutages.remove_trsf_pole_nb_without_numeric_digit(df_i, trsf_pole_nb_col=trsf_pole_nb_col)
                dfs.append(df_i)
            return dfs
        #---------------------------------------------------
        assert(isinstance(df, pd.DataFrame))
        df = df[(df[trsf_pole_nb_col].str.contains(r'\d')) | (df[trsf_pole_nb_col].isnull())]
        return df


    @staticmethod
    def get_mjr_mnr_cause_df_outage_subset(df_outage, mjr_cause, mnr_cause, 
                                           mjr_cause_col='MJR_CAUSE_CD', mnr_cause_col='MNR_CAUSE_CD', 
                                           return_copy=True):
        return_df = df_outage[(df_outage[mjr_cause_col]==mjr_cause) & 
                              (df_outage[mnr_cause_col]==mnr_cause)]
        if return_copy:
            return_df = return_df.copy()
        return return_df



    #TODO MIGHT WANT TO DO E.G. df_outage.drop_duplicates(subset=[x for x in df_outage.columns if x in cols_of_interest_met_prem]).dropna(how='all')
    # TO KEEP MORE COLUMNS OF INTEREST (AT LEAST FOR BY METER?).  MAYBE NOT?
    @staticmethod
    def build_df_outage_by_meter_xfmr_outage(df_outage, cols_for_drop_duplicates, 
                                             xfmr_gpby_cols=['trsf_pole_nb'], 
                                             xfmr_gpby_agg_dict={'annual_kwh':'sum', 'annual_max_dmnd':'sum'}, 
                                             outg_gpby_cols=['OUTG_REC_NB'], 
                                             outg_gpby_agg_dict={'annual_kwh':'sum', 'annual_max_dmnd':'sum'}):
        df_outage_by_meter = df_outage[cols_for_drop_duplicates].drop_duplicates().dropna(how='all')
        df_outage_by_xfmr  = df_outage_by_meter.groupby(xfmr_gpby_cols).agg(xfmr_gpby_agg_dict)
        df_outage_by_outg  = df_outage.groupby(outg_gpby_cols).agg(outg_gpby_agg_dict)
        return {
            'by_meter':df_outage_by_meter, 
            'by_xfmr':df_outage_by_xfmr, 
            'by_outg':df_outage_by_outg
        }
        
    #****************************************************************************************************
    @staticmethod
    def get_premise_nbs_for_outages(
        outg_rec_nbs, 
        return_type=dict, 
        col_type_outg_rec_nb=str, 
        col_type_premise_nb=None, 
        to_numeric_errors='coerce', 
        verbose=False,
        **kwargs
    ):
        r"""
        Acceptable return types:
            dict (default)
            pd.Series
            pd.DataFrame
            
        !!!!!!!!!!!!!!!!!!!!!!!!!    
        If other outage information (e.g., DT_ON_TS, DT_OFF_TS_FULL, etc.) are desired, see get_premise_nbs_and_others_for_outages
        Don't try to re-invent the wheel by developing a new method.  get_premise_nbs_and_others_for_outages utilizes
          DOVSOutages.build_consolidated_outage_df, where it is noted that (i) is quicker than (ii)
            (i)  running SQL query without premises included, grabbing premises after, and then joining the two
            (ii) (a) running SQL query with premises and then (b) running consolidate_df_outage
        !!!!!!!!!!!!!!!!!!!!!!!!!
        """
        #-------------------------
        assert(return_type==dict or 
               return_type==pd.Series or 
               return_type==pd.DataFrame)
        #-------------------------
        outg_rec_nb_col = kwargs.get('outg_rec_nb_col', 'OUTG_REC_NB')
        premise_nb_col  = kwargs.get('premise_nb_col', 'PREMISE_NB')
        #-----
        return_premise_nbs_col = kwargs.get('return_premise_nbs_col', 'premise_nbs' if return_type!=pd.DataFrame else 'premise_nb')
        #-------------------------
        # If the number of outg_rec_nbs is greater than 1000, the system crashes, as apparently
        # '[HY000] [Oracle][ODBC][Ora]ORA-01795: maximum number of expressions in a list is 1000'
        # So, if len(outg_rec_nbs)>1000, use batches!!!
        dovs_outgs = DOVSOutages(                 
            df_construct_type=DFConstructType.kRunSqlQuery, 
            contstruct_df_args=dict(read_sql_args=dict(dtype={outg_rec_nb_col:np.int64, premise_nb_col:str})), 
            init_df_in_constructor=True, 
            build_sql_function=DOVSOutages_SQL.build_sql_outage_premises, 
            build_sql_function_kwargs=dict(        
                outg_rec_nbs=outg_rec_nbs, 
                cols_of_interest=[outg_rec_nb_col, premise_nb_col], 
                field_to_split='outg_rec_nbs', 
                batch_size=1000, 
                verbose=verbose
            ),
            save_args=False, 
            build_consolidated=False
        )
        df = dovs_outgs.df
        #-------------------------
        if col_type_outg_rec_nb is None:
            col_type_outg_rec_nb = type(outg_rec_nbs[0])
        if col_type_premise_nb is None:
            col_type_premise_nb = col_type_outg_rec_nb
        df = Utilities_df.convert_col_types(
            df=df, 
            cols_and_types_dict={outg_rec_nb_col:col_type_outg_rec_nb, premise_nb_col:col_type_premise_nb}, 
            to_numeric_errors=to_numeric_errors, 
            inplace=True
        )
        df = df.rename(columns={premise_nb_col:return_premise_nbs_col})
        #-------------------------
        if return_type==pd.DataFrame:
            return df
        else:
            return_series = df.groupby(outg_rec_nb_col)[return_premise_nbs_col].unique()
            return_series=return_series.apply(lambda x: sorted(list(x)))
            if return_type==pd.Series:
                return return_series
            else:
                return return_series.to_dict()
                
    @staticmethod
    def get_premise_nbs_and_others_for_outages(
        outg_rec_nbs, 
        other_outg_cols=None, 
        return_premise_nbs_col='premise_nbs', 
        addtnl_build_sql_function_kwargs=None, 
        col_type_outg_rec_nb=str, 
        col_type_premise_nb=None, 
        to_numeric_errors='coerce', 
        dovs_outg_rec_nb_col='OUTG_REC_NB', 
        dovs_premise_nb_col='PREMISE_NB', 
        verbose=False
    ):
        r"""
        other_outg_cols:
          Other columns from DOVS_OUTAGE_FACT (or other tables, assuming they are included properly (see note below)), to be included
            in the output.  Typical values are, e.g., other_outg_cols=['DT_ON_TS', 'DT_OFF_TS_FULL']
          There should only be ONE VALUE PER OUTAGE for the members in other_outg_cols.
            e.g., adding 'ACCOUNT_NB' would be inappropriate

        NOTE: If any of other_outg_cols are from tables other than DOVS_OUTAGE_FACT, one needs to explicitly include those tables
          e.g., if one wanted to include 'STATE_ABBR_TX', one would need to set include_DOVS_MASTER_GEO_DIM to True in 
                addtnl_build_sql_function_kwargs
        """
        #----------------------------------------------------------------------------------------------------
        # If other_outg_cols is None, run get_premise_nbs_for_outages like usual (with return_type=pd.Series)
        if other_outg_cols is None:
            return_type=pd.Series
            df = DOVSOutages.get_premise_nbs_for_outages(
                outg_rec_nbs=outg_rec_nbs, 
                return_type=return_type, 
                col_type_outg_rec_nb=col_type_outg_rec_nb, 
                col_type_premise_nb=col_type_premise_nb, 
                to_numeric_errors=to_numeric_errors, 
                verbose=verbose,
                return_premise_nbs_col=return_premise_nbs_col, 
                outg_rec_nb_col=dovs_outg_rec_nb_col, 
                premise_nb_col=dovs_premise_nb_col
            )
            return df
        #----------------------------------------------------------------------------------------------------
        #-------------------------
        assert(Utilities.is_object_one_of_types(other_outg_cols, [list, str]))
        if isinstance(other_outg_cols, str):
            other_outg_cols = [other_outg_cols]
        #-------------------------
        # NOTE: dovs_outg_rec_nb_col is not included in final_cols_of_interest because it will be made the 
        #       index of the returned df (by build_consolidated_outage_df)
        final_cols_of_interest = other_outg_cols + [return_premise_nbs_col]
        #-------------------------
        # NOTE: Setting cols_of_interest=None in build_sql_function_kwargs below means the standard columns of interest 
        #       are used (see DOVSOutages_SQL.get_std_cols_of_interest).  This is the easiest method to ensure what is
        #       desired is returned, since adding things like DT_OFF_TS_FULL are somewhat difficult/cumbersome otherwise.
        # The actual returned columns will be chopped down later
        std_cols_of_interest = DOVSOutages_SQL.get_std_cols_of_interest()
        addtnl_cols_of_interest = [x for x in other_outg_cols 
                                   if not DOVSOutages_SQL.alias_found_in_cols_of_interest(x, std_cols_of_interest)]
        #-----
        # If the number of outg_rec_nbs is greater than 1000, the system crashes, as apparently
        # '[HY000] [Oracle][ODBC][Ora]ORA-01795: maximum number of expressions in a list is 1000'
        # So, if len(outg_rec_nbs)>1000, use batches!!!
        build_sql_function_kwargs=dict(
            cols_of_interest=None, 
            addtnl_cols_of_interest=addtnl_cols_of_interest, 
            outg_rec_nbs=outg_rec_nbs, 
            field_to_split='outg_rec_nbs', 
            batch_size=1000, 
            verbose=verbose
        )
        if addtnl_build_sql_function_kwargs is not None:
            build_sql_function_kwargs = {**build_sql_function_kwargs, **addtnl_build_sql_function_kwargs}
        #-------------------------
        df = DOVSOutages.build_consolidated_outage_df(
            contstruct_df_args=dict(read_sql_args=dict(dtype={dovs_outg_rec_nb_col:np.int64, dovs_premise_nb_col:str})), 
            build_sql_function=DOVSOutages_SQL.build_sql_outage, 
            build_sql_function_kwargs=build_sql_function_kwargs, 
            return_premise_nbs_col=return_premise_nbs_col, 
            addtnl_get_premise_nbs_for_outages_kwargs=dict(
                col_type_outg_rec_nb=col_type_outg_rec_nb, 
                col_type_premise_nb=col_type_premise_nb, 
                to_numeric_errors=to_numeric_errors, 
                outg_rec_nb_col=dovs_outg_rec_nb_col, 
                premise_nb_col=dovs_premise_nb_col, 
                return_premise_nbs_col=return_premise_nbs_col
            )
        )
        #-------------------------
        df = df[final_cols_of_interest]
        #-------------------------
        return df
                
                
    @staticmethod
    def get_SNs_from_mp_df_gpdby_prem_nb(mp_df, prem_nb):
        try:
            sn = mp_df[prem_nb].tolist()
        except:
            sn = []
        return sn

    @staticmethod
    def get_serial_numbers_for_outages(
        outg_rec_nbs, 
        return_type=dict, 
        col_type_outg_rec_nb=str, 
        col_type_premise_nb=None, 
        col_type_serial_nb=None, 
        to_numeric_errors='coerce', 
        return_premise_nbs_for_outages=False, 
        mp_df=None, 
        **kwargs
    ):
        r"""
        return_premise_nbs_for_outages: 
          If return_premise_nbs_for_outages is True 
            The premise numbers for outages are returned as well.
            These need to be built regardless, so setting to True will not take any additional time.
            If return_type==dict:
              a dict will be returned, with keys= ['serial_nbs', 'premise_nbs']
            If return_type==pd.Series:
              the serial_nbs series will be joined with the premise_nbs series to form a pd.DataFrame.
              Note: The outg_rec_nbs will be grouped, so there will be a single entry for each outage
            If return_type==pd.DataFrame:
              the serial_nbs dataframe will be joined with the premise_nbs dataframe to form a pd.DataFrame
              Note: The outg_rec_nbs will not be grouped, so there will be multiple entries for each outage

        mp_df:
          Meter premise pd.DataFrame.  This DOES NOT need to be supplied.  If it is not supplied, it will be built.
            Buidling mp_df can take a few minutes, so the purpose of allowing mp_df as an input is to save time if
            the user already has mp_df to use.
          HOWEVER!!!!! If a very large mp_df (e.g., the full dataset) is supplied, this will actually TAKE LONGER
            to run than not supplying mp_df and having the function build it.  The only time when supplying the full
            mp_df dataset would save time is when the full (or nearly full) dataset it needed (which is unlikely)
          If mp_df is supplied, one must be careful to ensure all premise numbers in outg_rec_nbs (most specifically, 
            all premise numbers in outg_rec_nbs found in the MeterPremise database) are included in mp_df.

        Acceptable return types:
            dict (default)
            pd.Series
            pd.DataFrame
        """
        #-------------------------
        assert(return_type==dict or 
               return_type==pd.Series or 
               return_type==pd.DataFrame)
        #-------------------------
        # NOTE: outg_rec_nb_col and conn_db not explicitly used here, but are used in 
        #       DOVSOutages.get_premise_nbs_for_outages.  Kept here to remind me of their existence.
        outg_rec_nb_col           = kwargs.get('outg_rec_nb_col', 'OUTG_REC_NB')
        premise_nb_col            = kwargs.get('premise_nb_col', 'PREMISE_NB')
        conn_db                   = kwargs.get('conn_db', Utilities.get_utldb01p_oracle_connection())
        #-----
        return_serial_nbs_col     = kwargs.get('return_serial_nbs_col', 'serial_numbers')
        return_premise_nbs_col    = kwargs.get('return_premise_nbs_col', 'premise_nbs' if return_type!=pd.DataFrame else 'premise_nb')
        #-----
        mp_serial_number_col      = kwargs.get('mp_serial_number_col', 'mfr_devc_ser_nbr')   
        mp_premise_nb_col         = kwargs.get('mp_premise_nb_col', 'prem_nb')
        cols_of_interest_met_prem = kwargs.get('cols_of_interest_met_prem', [mp_serial_number_col, mp_premise_nb_col])
        #-------------------------
        if col_type_outg_rec_nb is None:
            col_type_outg_rec_nb = type(outg_rec_nbs[0])
        if col_type_premise_nb is None:
            col_type_premise_nb = col_type_outg_rec_nb
        if col_type_serial_nb is None:
            col_type_serial_nb = col_type_outg_rec_nb
        #-------------------------
        prem_nbs_in_outgs = DOVSOutages.get_premise_nbs_for_outages(
            outg_rec_nbs=outg_rec_nbs, 
            return_type=return_type, 
            col_type_outg_rec_nb=col_type_outg_rec_nb, 
            col_type_premise_nb=col_type_premise_nb, 
            to_numeric_errors=to_numeric_errors, 
            **kwargs        
        )
        #-------------------------
        if isinstance(prem_nbs_in_outgs, dict):
            all_prem_nbs = []
            for outg_rec_nb, prem_nbs in prem_nbs_in_outgs.items():
                all_prem_nbs.extend(prem_nbs)
            all_prem_nbs = list(set(all_prem_nbs))
        elif isinstance(prem_nbs_in_outgs, pd.Series):
            # NOTE: The premise numbers are actually stored in np.ndarrys here, for which calling .sum() 
            #       does not work.  This is why the .apply(lambda x: list(x)) must be implemented
            all_prem_nbs = list(set(prem_nbs_in_outgs.apply(lambda x: list(x)).sum()))
        elif isinstance(prem_nbs_in_outgs, pd.DataFrame):
            # NOTE: When pd.DataFrame is returned, there is no grouping, so the PREMISE_NB column contains only single
            #       values (and values in OUTG_REC_NB column are repeated for as many prem nbs as exist)
            all_prem_nbs = prem_nbs_in_outgs[return_premise_nbs_col].tolist()
        else:
            assert(0)
        #-------------------------
        if mp_df is None:
            mp = MeterPremise(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                init_df_in_constructor=True, 
                build_sql_function=MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs=dict(
                    cols_of_interest=cols_of_interest_met_prem, 
                    premise_nbs=all_prem_nbs
                )
            )
            #----------
            mp_df = mp.df
        #----------
        mp_df = Utilities_df.convert_col_types(
            df=mp_df, 
            cols_and_types_dict={mp_serial_number_col:col_type_serial_nb, mp_premise_nb_col:col_type_premise_nb}, 
            to_numeric_errors=to_numeric_errors, 
            inplace=True    
        )
        #----------
        mp_df = mp_df.groupby(mp_premise_nb_col)[mp_serial_number_col].unique()
        #-------------------------
        if Utilities.is_object_one_of_types(prem_nbs_in_outgs, [dict, pd.Series]):
            if isinstance(prem_nbs_in_outgs, dict):
                SNs_in_outgs = dict()
            else:
                SNs_in_outgs = pd.Series(dtype='object')
                SNs_in_outgs.index.name = prem_nbs_in_outgs.index.name
                SNs_in_outgs.name = return_serial_nbs_col
            #-------------------------
            for outg_rec_nb, prem_nbs in prem_nbs_in_outgs.items():
                SNs_i = []
                for prem_nb in prem_nbs:
                    # Not all premise numbers from outages found in MeterPremise database?!
                    # This is why the try/except block is needed below
                    try:
                        SNs_ij = mp_df[prem_nb].tolist()
                    except:
                        continue
                    SNs_i.extend(SNs_ij)
                assert(outg_rec_nb not in SNs_in_outgs)
                SNs_in_outgs[outg_rec_nb] = SNs_i
            #-------------------------
            if return_premise_nbs_for_outages:
                if isinstance(prem_nbs_in_outgs, dict):
                    SNs_in_outgs = {return_serial_nbs_col:SNs_in_outgs, 
                                    return_premise_nbs_col:prem_nbs_in_outgs}
                else:
                    SNs_in_outgs = pd.merge(prem_nbs_in_outgs, SNs_in_outgs, left_index=True, right_index=True)
        elif isinstance(prem_nbs_in_outgs, pd.DataFrame):
            SNs_in_outgs = prem_nbs_in_outgs.copy()
            SNs_in_outgs[return_serial_nbs_col] = SNs_in_outgs[return_premise_nbs_col].apply(lambda x: DOVSOutages.get_SNs_from_mp_df_gpdby_prem_nb(mp_df, x))
            if not return_premise_nbs_for_outages:
                SNs_in_outgs = SNs_in_outgs.drop(columns=[return_premise_nbs_col])
        else:
            assert(0)
        #-------------------------
        return SNs_in_outgs
        
    @staticmethod
    def get_active_SNs_and_others_for_outages(
        outg_rec_nbs, 
        df_mp_curr, 
        df_mp_hist, 
        addtnl_other_outg_cols=None, 
        return_premise_nbs_col='premise_nbs', 
        return_premise_nbs_from_MP_col='premise_nbs_from_MP',
        return_SNs_col='SNs', 
        addtnl_build_sql_function_kwargs=None, 
        col_type_outg_rec_nb=str, 
        col_type_premise_nb=None, 
        to_numeric_errors='coerce', 
        verbose=False,
        consolidate_PNs_batch_size=1000, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb', 
        dovs_outg_rec_nb_col='OUTG_REC_NB', 
        dovs_premise_nb_col='PREMISE_NB', 
        dovs_t_min_col='DT_OFF_TS_FULL',
        dovs_t_max_col='DT_ON_TS'
    ):
        r"""
        NOTE: If one takes outg_rec_nbs from a general list of outages (e.g., performing a query for all outages between 
              some date range), typically DOVSOutages.get_premise_nbs_and_others_for_outages will not return an entry for
              every outage.  The outages which are excluded may have 0 CI_NB/CMI_NB, the DT_ON_TS and/or STEP_DRTN_NB may
              be null (in either case, DT_OFF_TS_FULL will be null as well), or the outg_rec_nb may be equal to 
              -2147483648, which I'm still not full sure what that negative outage number means.
        """
        #-------------------------
        # Build PNs and others from outages
        other_outg_cols=[dovs_t_min_col, dovs_t_max_col]
        if addtnl_other_outg_cols is not None:
            other_outg_cols = list(set(other_outg_cols+addtnl_other_outg_cols))
        PNs_for_outgs = DOVSOutages.get_premise_nbs_and_others_for_outages(
            outg_rec_nbs=outg_rec_nbs, 
            other_outg_cols=other_outg_cols, 
            return_premise_nbs_col=return_premise_nbs_col, 
            addtnl_build_sql_function_kwargs=addtnl_build_sql_function_kwargs, 
            col_type_outg_rec_nb=col_type_outg_rec_nb, 
            col_type_premise_nb=col_type_premise_nb, 
            to_numeric_errors=to_numeric_errors, 
            dovs_outg_rec_nb_col=dovs_outg_rec_nb_col, 
            dovs_premise_nb_col=dovs_premise_nb_col, 
            verbose=verbose
        )
        #-------------------------
        # From PNs_for_outgs, grab the list of PNs
        # In very limited testing, a batch size of 1000 seemed to work well here
        PNs = Utilities_df.consolidate_column_of_lists(
            df=PNs_for_outgs, 
            col=return_premise_nbs_col, 
            sort=True,
            include_None=False,
            batch_size=consolidate_PNs_batch_size, 
            verbose=False
        )
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        # If df_mp_curr or df_mp_hist are not supplied, they will be built    
        if df_mp_hist is None:
            mp_hist = MeterPremise(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                init_df_in_constructor=True, 
                build_sql_function=MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs=dict(
                    cols_of_interest=necessary_mp_cols, 
                    premise_nbs=PNs, 
                    table_name='meter_premise_hist'
                )
            )
            df_mp_hist = mp_hist.df
        #-----
        if df_mp_curr is None:
            mp_curr = MeterPremise(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                init_df_in_constructor=True, 
                build_sql_function=MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs=dict(
                    cols_of_interest=necessary_mp_cols+[df_mp_trsf_pole_nb_col], 
                    premise_nbs=PNs, 
                    table_name='meter_premise'
                )
            )
            df_mp_curr = mp_curr.df
        #-------------------------
        # At a bare minimum, df_mp_curr and df_mp_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in df_mp_curr.columns for x in necessary_mp_cols+[df_mp_trsf_pole_nb_col]]))
        assert(all([x in df_mp_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        # NOT ALL PREMISE NUMBERS FROM OUTAGES ARE FOUND IN METER PREMISE!!!!!!!!!!!!!!
        # TAKE prem_nb = 'NC0526903' from outg_rec_nb='11411965'

        # Below, a simpler method was first applied, in which output_groupby was None in the call
        #   to get_active_SNs_for_PNs_at_datetime_interval.  This avoided the extra complication of
        #   resetting the index and filling empty prem_nb entries with np.nan and empty mfr_devc_ser_nbr with []
        #   The empty outages could still be recovered by finding the empty DFs in active_SNs_in_outgs_dfs_dict after
        #   and appending them to the final DF.
        # The method implemented is superior in that outage numbers are included by default even if no serial numbers
        #   are found.  Furthermore, this can handle the case where prem numbers are found but no serial numbers are,
        #   in which case the prem numbers will still be recovered (with the first method, recovering these was impossible)

        # NOTE: Some are empty because the DOVS database has null value(s) for DT_ON_TS and/or STEP_DRTN_NB
        #   e.g., outg_rec_nb='11411975' where premise_nbs = ['NC0526900']
        #      NC0526900 is not found in meter_premise nor meter_premise_hist, therefore both mfr_devc_ser_nbr and 
        #      prem_nb are equal to [nan]
        #      NOTE: There is also no time info for this outage!
        #   e.g., outg_rec_nb='11412036' where premise_nbs = ['070613522', '074313522', '075223522', '077413522', '078323522'] 
        #      These premise numbers do exist in meter_premise, but the DT_ON_TS and STEP_DRTN_NB fields are null (thus, also
        #      making DT_OFF_TS_FULL null as well).  Therefore, these have prem_nbs equal to those found, but mfr_devc_ser_nbr
        #      equal to [nan]. 
        #      Using the other method, this information about the prem_nbs would not be retrieved

        # Only reason for making dict is to ensure outg_rec_nbs are not repeated 
        active_SNs_in_outgs_dfs_dict = {}

        for outg_rec_nb_i, row_i in PNs_for_outgs.iterrows():
            # active_SNs_df_i will have indices equal to premise numbers and value equal to lists
            #   of active SNs for each PN
            active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                PNs=row_i[return_premise_nbs_col],
                df_mp_curr=df_mp_curr, 
                df_mp_hist=df_mp_hist, 
                dt_0=row_i[dovs_t_min_col],
                dt_1=row_i[dovs_t_max_col],
                output_index=None,
                output_groupby=[df_mp_prem_nb_col], 
                include_prems_wo_active_SNs_when_groupby=True, 
                assert_all_PNs_found=False
            )
            active_SNs_df_i = active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col] = np.nan
                active_SNs_df_i[df_mp_serial_number_col] = [[]]    
            active_SNs_df_i[dovs_outg_rec_nb_col] = outg_rec_nb_i
            active_SNs_df_i = active_SNs_df_i.explode(df_mp_serial_number_col)
            assert(outg_rec_nb_i not in active_SNs_in_outgs_dfs_dict)
            active_SNs_in_outgs_dfs_dict[outg_rec_nb_i] = active_SNs_df_i
        #-------------------------
        active_SNs_df = pd.concat(list(active_SNs_in_outgs_dfs_dict.values()))
        #-------------------------
        active_SNs_df = Utilities_df.consolidate_df(
            df=active_SNs_df, 
            groupby_cols=dovs_outg_rec_nb_col, 
            cols_shared_by_group=None, 
            cols_to_collect_in_lists=[df_mp_serial_number_col, df_mp_prem_nb_col], 
            include_groupby_cols_in_output_cols=False, 
            allow_duplicates_in_lists=False, 
            recover_uniqueness_violators=True, 
            rename_cols=None, 
            verbose=True
        )
        #-------------------------
        # Change [nan] entries to []
        #----------
        # Old method of doing everything in a single step and setting with = [[]] (commented out below) gives the error:
        #   ValueError: Must have equal len keys and value when setting with an ndarray
        # Must split into two steps, and set using = [[] for _ in len]
        #----------
        # Old method
        # active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0, df_mp_serial_number_col] = [[]]
        # active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0, df_mp_prem_nb_col] = [[]]
        #----------
        empty_mask = active_SNs_df[df_mp_serial_number_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        active_SNs_df.loc[empty_mask, df_mp_serial_number_col] = [[] for _ in range(empty_mask.sum())]
        #-----
        empty_mask = active_SNs_df[df_mp_prem_nb_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        active_SNs_df.loc[empty_mask, df_mp_prem_nb_col] = [[] for _ in range(empty_mask.sum())]
        #-------------------------
        active_SNs_df = active_SNs_df.rename(columns={
            df_mp_prem_nb_col:return_premise_nbs_from_MP_col, 
            df_mp_serial_number_col:return_SNs_col
        })
        active_SNs_and_others_df = pd.merge(PNs_for_outgs, active_SNs_df, left_index=True, right_index=True, how='left')
        #-------------------------
        return active_SNs_and_others_df
        
        
    @staticmethod
    def get_mjr_mnr_causes_for_outages(
        outg_rec_nbs, 
        include_equip_type=True, 
        set_outg_rec_nb_as_index=True
    ):
        r"""
        """
        #-------------------------
        cols_of_interest = ['OUTG_REC_NB', 'MJR_CAUSE_CD', 'MNR_CAUSE_CD']
        #-------------------------
        if include_equip_type:
            cols_of_interest.append('EQUIPMENT_CD')
            include_DOVS_EQUIPMENT_TYPES_DIM=True
        else:
            include_DOVS_EQUIPMENT_TYPES_DIM=False
        #-------------------------
        dovs_outgs = DOVSOutages(                 
            df_construct_type=DFConstructType.kRunSqlQuery, 
            contstruct_df_args=None, 
            init_df_in_constructor=True, 
            build_sql_function=DOVSOutages_SQL.build_sql_outage, 
            build_sql_function_kwargs=dict(        
                outg_rec_nbs = outg_rec_nbs, 
                cols_of_interest=cols_of_interest, 
                include_DOVS_OUTAGE_CAUSE_TYPES_DIM=True, 
                include_DOVS_EQUIPMENT_TYPES_DIM=include_DOVS_EQUIPMENT_TYPES_DIM, 
                field_to_split='outg_rec_nbs'
            ),
        )
        #-------------------------
        mjr_mnr_causes_df = dovs_outgs.df
        #-------------------------
        # The length of mjr_mnr_causes_df should match that of outg_rec_nbs
        # However, I suppose it is possible that not all outage numbers are found?
        # For now, keep assertion strict, but maybe loosen to 
        #   mjr_mnr_causes_df.shape[0] <= len(outg_rec_nbs)
        # In any case, the length of mjr_mnr_causes_df should always match the number
        #   of unique outg_rec_nbs
        assert(mjr_mnr_causes_df.shape[0] == len(set(outg_rec_nbs)))
        assert(mjr_mnr_causes_df.shape[0] == mjr_mnr_causes_df['OUTG_REC_NB'].nunique())
        #-------------------------
        if set_outg_rec_nb_as_index:
            mjr_mnr_causes_df = mjr_mnr_causes_df.set_index('OUTG_REC_NB').sort_index()
        #-------------------------
        return mjr_mnr_causes_df
        
        
    @staticmethod
    def get_outg_rec_nbs_from_df(df, idfr):
        r"""
        Extract the outg_rec_nbs from a df.
          
        Return Value:
            If outg_rec_nbs are stored in a column of df, the returned object will be a pd.Series
            If outg_rec_nbs are stored in the index, the returned object will be a pd index.
            If one wants the returned object to be a list, use get_outg_rec_nbs_list_from_df

        idfr:
            This directs from where the outg_rec_nbs will be retrieved.
            This should be a string, list, or tuple.
            If the outg_rec_nbs are located in a column, idfr should simply be the column
                - Single index columns --> simple string
                - MultiIndex columns   --> appropriate tuple to identify column
            If the outg_rec_nbs are located in the index:
                - Single level index --> simple string 'index' or 'index_0'
                - MultiIndex index:  --> 
                    - string f'index_{level}', where level is the index level containing the outg_rec_nbs
                    - tuple of length 2, with 0th element ='index' and 1st element = idx_level_name where
                        idx_level_name is the name of the index level containing the outg_rec_nbs 
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(idfr, [str, list, tuple]))
        # NOTE: pd doesn't like checking for idfr in df.columns if idfr is a list.  It is fine checking when
        #       it is a tuple, as tuples can represent columns.  Therefore, if idfr is a list, convert to tuple
        #       as this will fix the issue below and have no effect elsewhere.
        if isinstance(idfr, list):
            idfr = tuple(idfr)
        if idfr in df.columns:
            return df[idfr]
        #-------------------------
        # If not in the columns (because return from function not executed above), outg_rec_nbs must be in the indices!
        # The if/else block below determines idfr_idx_lvl
        if isinstance(idfr, str):
            assert(idfr.startswith('index'))
            if idfr=='index':
                idfr_idx_lvl=0
            else:
                idfr_idx_lvl = re.findall(r'index_(\d*)', idfr)
                assert(len(idfr_idx_lvl)==1)
                idfr_idx_lvl=idfr_idx_lvl[0]
                idfr_idx_lvl=int(idfr_idx_lvl)
        else:
            assert(len(idfr)==2)
            assert(idfr[0]=='index')
            idx_level_name = idfr[1]
            assert(idx_level_name in df.index.names)
            # Need to also make sure idx_level_name only occurs once, so no ambiguity!
            assert(df.index.names.count(idx_level_name)==1)
            idfr_idx_lvl = df.index.names.index(idx_level_name)
        #-------------------------
        assert(idfr_idx_lvl < df.index.nlevels)
        return df.index.get_level_values(idfr_idx_lvl)
    
    
    @staticmethod
    def get_outg_rec_nbs_list_from_df(df, idfr, unique_only=True):
        r"""
        See get_outg_rec_nbs_from_df for details.
        This returns a list version of get_outg_rec_nbs_from_df
        """
        #-------------------------
        out_rec_nbs = DOVSOutages.get_outg_rec_nbs_from_df(df=df, idfr=idfr)
        if unique_only:
            return out_rec_nbs.unique().tolist()
        else:
            return out_rec_nbs.tolist()
            
            
    @staticmethod
    def get_outg_info_for_df(
        df, 
        outg_rec_nb_idfr, 
        contstruct_df_args=None, 
        build_sql_function=None, 
        build_sql_function_kwargs=None, 
        set_outg_rec_nb_as_index=True
    ):
        r"""
        To a DF containing outage data, get additional information of the outage from DOVS.
        Intended for use with MECPODf (or similar) to append e.g., the date of the outage for filtering dataset into various years
          (or whatever time periods).
        Also, intended for simple fields, e.g., those from DOVS_OUTAGE_FACT, not lists of premise numbers in outage etc.
          If one did want to include the premise numbers, one could utilize build_consolidated_outage or similar in this function.

        Note: Leaving build_sql_function==build_sql_function_kwargs==None will use DOVSOutages_SQL.build_sql_std_outage for the
              outg_rec_nbs in DF (e.g., DOVSOutages_SQL.get_std_cols_of_interest will be used to get columns of interest, etc.)
        """
        #--------------------------------------------------
        # Get outg_rec_nbs (series) and outg_rec_nbs_unq (list) from df
        outg_rec_nbs = DOVSOutages.get_outg_rec_nbs_from_df(df=df, idfr=outg_rec_nb_idfr)
        assert(len(df)==len(outg_rec_nbs)) # Important in ensuring proper merge at end
        #outg_rec_nbs_unq = outg_rec_nbs.unique().tolist()
        # NOTE: Cannot include NaN outg_rec_nbs in SQL query!
        outg_rec_nbs_unq = outg_rec_nbs.dropna().unique().tolist()
        #--------------------------------------------------
        # Build dovs_outgs_df
        df_construct_type=DFConstructType.kRunSqlQuery
        init_df_in_constructor=True
        #-------------------------
        if build_sql_function is None:
            build_sql_function = DOVSOutages_SQL.build_sql_std_outage
        #-----
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs={}
        #-----
        if build_sql_function == DOVSOutages_SQL.build_sql_std_outage:
            build_sql_function_kwargs['include_premise']=False
            build_sql_function_kwargs['include_DOVS_PREMISE_DIM']=False
        else:
            build_sql_function_kwargs['include_DOVS_PREMISE_DIM']=False
        #-------------------------
        # Turn off verbose by default
        build_sql_function_kwargs['verbose'] = build_sql_function_kwargs.get('verbose', False)
        #-------------------------
        build_sql_function_kwargs['outg_rec_nbs'] = outg_rec_nbs_unq
        build_sql_function_kwargs['field_to_split'] = 'outg_rec_nbs'
        #-------------------------   
        dovs_outgs = DOVSOutages(
            df_construct_type=df_construct_type, 
            contstruct_df_args=contstruct_df_args, 
            init_df_in_constructor=init_df_in_constructor,
            build_sql_function=build_sql_function, 
            build_sql_function_kwargs=build_sql_function_kwargs, 
            build_consolidated=False
        )
        dovs_outgs_df = dovs_outgs.get_df()
        if set_outg_rec_nb_as_index and dovs_outgs_df.shape[0]>0:
            outg_rec_nb_col = build_sql_function_kwargs.get('outg_rec_nb_col', 'OUTG_REC_NB')
            dovs_outgs_df = dovs_outgs_df.set_index(outg_rec_nb_col).sort_index()
        #--------------------------------------------------
        return dovs_outgs_df
    
    
    @staticmethod
    def append_outg_info_to_df(
        df, 
        outg_rec_nb_idfr, 
        contstruct_df_args=None, 
        build_sql_function=None, 
        build_sql_function_kwargs=None, 
        dummy_col_levels_prefix='outg_dummy_lvl_'
    ):
        r"""
        To a DF containing outage data, append additional information of the outage from DOVS.
        Intended for use with MECPODf (or similar) to append e.g., the date of the outage for filtering dataset into various years
          (or whatever time periods).
        Also, intended for simple fields, e.g., those from DOVS_OUTAGE_FACT, not lists of premise numbers in outage etc.
          If one did want to include the premise numbers, one could utilize build_consolidated_outage or similar in this function.
    
        Note: Leaving build_sql_function==build_sql_function_kwargs==None will use DOVSOutages_SQL.build_sql_std_outage for the
              outg_rec_nbs in DF (e.g., DOVSOutages_SQL.get_std_cols_of_interest will be used to get columns of interest, etc.)
              
        dummy_col_levels_prefix:
            In order to merge, df and dovs_outgs_df must have same number of levels in columns.
            If dovs_outgs_df needs additional levels, dummy levels will be added with values equal to 
              dummy_col_levels_prefix_i (i from 0 to however many dummy levels are needed)
        """
        #--------------------------------------------------
        # Note: Setting the index of dovs_outgs_df to outg_rec_nb_col and using right_index=True in merge below
        #         (instead of right_on=outg_rec_nb_col) causes return_df to have the same index as df, as desired,
        #         instead of resetting the index (as would be done otherwise)
        #       Hence, set_outg_rec_nb_as_index=True in get_outg_info_for_df below
        dovs_outgs_df = DOVSOutages.get_outg_info_for_df(
            df                        = df, 
            outg_rec_nb_idfr          = outg_rec_nb_idfr, 
            contstruct_df_args        = contstruct_df_args, 
            build_sql_function        = build_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs, 
            set_outg_rec_nb_as_index  = True
        )
        #--------------------------------------------------
        # In order to merge, df and dovs_outgs_df must have same number of levels in columns
        if df.columns.nlevels>1:
            n_levels_to_add = df.columns.nlevels - dovs_outgs_df.columns.nlevels
            #-----
            for i_new_lvl in range(n_levels_to_add):
                # With each iteration, prepending a new level from n_levels_to_add-1 to 0
                i_level_val = f'{dummy_col_levels_prefix}{(n_levels_to_add-1)-i_new_lvl}'
                dovs_outgs_df = Utilities_df.prepend_level_to_MultiIndex(
                    df         = dovs_outgs_df, 
                    level_val  = i_level_val, 
                    level_name = None, 
                    axis       = 1
                )
        assert(df.columns.nlevels==dovs_outgs_df.columns.nlevels)        
        #--------------------------------------------------
        # Merge dovs_outgs_df with df
        #-----
        # Pandas functionality has apparently changed, and old method leads to undesirable result.
        # OLD METHOD:
        #     outg_rec_nbs = DOVSOutages.get_outg_rec_nbs_from_df(df=df, idfr=outg_rec_nb_idfr)
        #     return_df = pd.merge(df, dovs_outgs_df, how='left', left_on=outg_rec_nbs, right_index=True)
        # UNDESIRABLE RESULT:
        #     For whatever reason, return_df now contains an additional column housing the contents of outg_rec_nbs.
        #     For the normal case of MultiIndex columns, this additional column is ('key_0', '')
        # SOLUTION:
        #     Use Utilities_df.merge_dfs (general function, but specifically built for this purpose)
        #-------------------------
        og_shape = df.shape
        #-----
        idfr_loc = Utilities_df.get_idfr_loc(
            df   = df, 
            idfr = outg_rec_nb_idfr
        )
        if idfr_loc[1]:
            final_index = 1
        else:
            final_index = None
        #-----
        return_df = Utilities_df.merge_dfs(
            df_1        = df, 
            df_2        = dovs_outgs_df, 
            merge_on_1  = outg_rec_nb_idfr, 
            merge_on_2  = 'index', 
            how         = 'left', 
            final_index = final_index
        )
        assert(og_shape[0]==return_df.shape[0])
        #--------------------------------------------------
        return return_df
        
    @staticmethod
    def append_outg_dt_off_ts_full_to_df(
        df, 
        outg_rec_nb_idfr, 
        dummy_col_levels_prefix='outg_dummy_lvl_', 
        include_dt_on_ts=False
    ):
        r"""
        To a DF containing outage data, append DT_OFF_TS_FULL from DOVS.
        Intended for use with MECPODf (or similar) to append e.g., the date of the outage for filtering dataset into various years
          (or whatever time periods).
          
        dummy_col_levels_prefix:
            In order to merge, df and dovs_outgs_df must have same number of levels in columns.
            If dovs_outgs_df needs additional levels, dummy levels will be added with values equal to 
              dummy_col_levels_prefix_i (i from 0 to however many dummy levels are needed)
        """
        #-------------------------
        cols_of_interest=[
            'OUTG_REC_NB', 
            dict(field_desc=f"DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)", 
                 alias='DT_OFF_TS_FULL', table_alias_prefix=None)
        ]
        if include_dt_on_ts:
            cols_of_interest.append('DT_ON_TS')
        #-------------------------
        contstruct_df_args=None
        build_sql_function = DOVSOutages_SQL.build_sql_outage
        build_sql_function_kwargs=dict(
            datetime_col='DT_OFF_TS_FULL', 
            cols_of_interest=cols_of_interest
        )
        #-------------------------
        return DOVSOutages.append_outg_info_to_df(
            df                        = df, 
            outg_rec_nb_idfr          = outg_rec_nb_idfr, 
            contstruct_df_args        = contstruct_df_args, 
            build_sql_function        = build_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs, 
            dummy_col_levels_prefix   = dummy_col_levels_prefix
        )
            
            
            
    @staticmethod
    def append_to_df_mjr_mnr_causes_for_outages(
        df, 
        outg_rec_nb_idfr, 
        mjr_mnr_causes_df=None, 
        include_equip_type=True, 
        mjr_cause_col='MJR_CAUSE_CD', 
        mnr_cause_col='MNR_CAUSE_CD'
    ):
        r"""
        Appends the major and minor causes to the DF containing outages.
        outg_rec_nb_idfr:
            Directs where to find the outg_rec_nbs, see DOVSOutages.get_outg_rec_nbs_from_df
        """
        #-------------------------
        # Get outg_rec_nbs (series) and outg_rec_nbs_unq (list) from df
        outg_rec_nbs = DOVSOutages.get_outg_rec_nbs_from_df(df=df, idfr=outg_rec_nb_idfr)
        assert(len(df)==len(outg_rec_nbs)) # Important in ensuring proper merge at end
        outg_rec_nbs_unq = outg_rec_nbs.unique().tolist()
        #-------------------------
        # Build mjr_mnr_causes_df if not supplied, and ensure all outg_rec_nbs found in mjr_mnr_causes_df.
        if mjr_mnr_causes_df is None:
            mjr_mnr_causes_df = DOVSOutages.get_mjr_mnr_causes_for_outages(
                outg_rec_nbs=outg_rec_nbs_unq, 
                include_equip_type=include_equip_type, 
                set_outg_rec_nb_as_index=True
            )
            mjr_cause_col='MJR_CAUSE_CD'
            mnr_cause_col='MNR_CAUSE_CD'
        # Make sure all outg_rec_nbs_unq in mjr_mnr_causes_df
        assert(len(set(outg_rec_nbs_unq).difference(set(mjr_mnr_causes_df.index.tolist())))==0)
        #-------------------------
        # Merge mjr_mnr_causes_df with df
        return_df = pd.merge(df, mjr_mnr_causes_df, how='left', left_on=outg_rec_nbs, right_index=True)
        return return_df
        
        
    @staticmethod
    def get_df_subset_excluding_mjr_mnr_causes(
        df, 
        mjr_mnr_causes_to_exclude, 
        mjr_causes_to_exclude=None,
        mnr_causes_to_exclude=None, 
        mjr_cause_col='MJR_CAUSE_CD', 
        mnr_cause_col='MNR_CAUSE_CD'
    ):
        r"""
        Return the subset of df excluding any major and/or minor causes in mjr_mnr_causes_to_exclude, mjr_causes_to_exclude, mnr_causes_to_exclude.
        To ignore specific mjr/mnr cause pairs, use mjr_mnr_causes_to_exclude.
        To ignore mjr causes, regardless of mnr cause, use mjr_causes_to_exclude.
        To ignore mnr causes, regardless of mjr cause, use mnr_causes_to_exclude.

        mjr_mnr_causes_to_exclude:
            This is designed to be a list of dict objects (but it can also be a single dict object instead of a list of length 1
            if only a single needed).
            Each dict object, mjr_mnr_cause, should have keys 'mjr_cause', 'mnr_cause', and optionally 'addtnl_slicers'.
            mjr_mnr_cause['mjr_cause']:
                Should be a string specifying the major cause to be excluded.
            mjr_mnr_cause['mnr_cause']:
                Should be a string specifying the minor cause to be excluded.
            mjr_mnr_cause['addtnl_slicers']:
                Should be a list of dict/DataFrameSubsetSingleSlicer objects to be input into add_single_slicer (however, can be
                  a single dict/DataFrameSubsetSingleSlicer object if only one needed).
                If element is a dict, it should have key/value pairs to build a DataFrameSubsetSingleSlicer object
                  ==> keys = 'column', 'value', and optionally 'comparison_operator'

        mjr_causes_to_exclude:
            A list of strings specifying the major causes to be excluded (or, a single string if only one to be excluded).

        mjr_causes_to_exclude:
            A list of strings specifying the minor causes to be excluded (or, a single string if only one to be excluded)
        """
        #-------------------------
        return_df = df.copy()
        #-------------------------
        if mjr_causes_to_exclude is not None:
            assert(Utilities.is_object_one_of_types(mjr_causes_to_exclude, [str, list]))
            if isinstance(mjr_causes_to_exclude, str):
                mjr_causes_to_exclude = [mjr_causes_to_exclude]
            #----------
            slicer = DFSlicer(
                single_slicers=[
                    dict(column=mjr_cause_col, value=mjr_causes_to_exclude, comparison_operator='notin'), 
                ]
            )
            return_df = slicer.perform_slicing(return_df)
        #-------------------------
        if mnr_causes_to_exclude is not None:
            assert(Utilities.is_object_one_of_types(mnr_causes_to_exclude, [str, list]))
            if isinstance(mnr_causes_to_exclude, str):
                mnr_causes_to_exclude = [mnr_causes_to_exclude]
            #----------
            slicer = DFSlicer(
                single_slicers=[
                    dict(column=mnr_cause_col, value=mnr_causes_to_exclude, comparison_operator='notin'), 
                ]
            )
            return_df = slicer.perform_slicing(return_df)
        #-------------------------
        if mjr_mnr_causes_to_exclude is not None:
            assert(Utilities.is_object_one_of_types(mjr_mnr_causes_to_exclude, [dict, list]))
            if not isinstance(mjr_mnr_causes_to_exclude, list):
                mjr_mnr_causes_to_exclude=[mjr_mnr_causes_to_exclude]
            for mjr_mnr_cause in mjr_mnr_causes_to_exclude:
                assert(isinstance(mjr_mnr_cause, dict))
                assert('mjr_cause' in mjr_mnr_cause.keys() and 'mnr_cause' in mjr_mnr_cause.keys())
                mjr_mnr_cause['addtnl_slicers'] = mjr_mnr_cause.get('addtnl_slicers', None)
                slicer = DFSlicer(
                    single_slicers=[
                        dict(column=mjr_cause_col, value=mjr_mnr_cause['mjr_cause']), 
                        dict(column=mnr_cause_col, value=mjr_mnr_cause['mnr_cause'])
                    ], 
                    apply_not=True
                )
                if mjr_mnr_cause['addtnl_slicers'] is not None:
                    slicer.add_slicers(mjr_mnr_cause['addtnl_slicers'])
                return_df = slicer.perform_slicing(return_df)
        #-------------------------
        return return_df
        
        
    @staticmethod
    def retrieve_outage_from_dovs_df(
        dovs_df                  , 
        outg_rec_nb              , 
        outg_rec_nb_idfr         , 
        assert_outg_rec_nb_found = True
    ):
        r"""
        Retrieve the rows in dovs_df corresponding to outg_rec_nb

        outg_rec_nb_idfr:
            default 'index'
            This directs where the outg_rec_nbs are stored in dovs_df, which can be a column or the index.
            This should be a string, list, or tuple.
            If the outg_rec_nbs are located in a column, idfr should simply be the column
                - Single index columns --> simple string
                - MultiIndex columns   --> appropriate tuple to identify column
            If the outg_rec_nbs are located in the index:
                - Single level index --> simple string 'index' or 'index_0'
                - MultiIndex index:  --> 
                    - string f'index_{level}', where level is the index level containing the outg_rec_nbs
                    - tuple of length 2, with 0th element ='index' and 1st element = idx_level_name where
                        idx_level_name is the name of the index level containing the outg_rec_nbs
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(outg_rec_nb_idfr, [str, list, tuple]))
        #-------------------------
        # Check for outg_rec_nb_idfr in columns
        #-----
        # NOTE: pd doesn't like checking for idfr in df.columns if idfr is a list.  It is fine checking when
        #       it is a tuple, as tuples can represent columns.  Therefore, if idfr is a list, convert to tuple
        #       as this will fix the issue below and have no effect elsewhere.
        if isinstance(outg_rec_nb_idfr, list):
            outg_rec_nb_idfr = tuple(outg_rec_nb_idfr)
        if outg_rec_nb_idfr in dovs_df.columns:
            if assert_outg_rec_nb_found:
                assert(outg_rec_nb in dovs_df[outg_rec_nb_idfr].unique().tolist())
            return_df = dovs_df[dovs_df[outg_rec_nb_idfr]==outg_rec_nb]
            return return_df

        #-------------------------
        # If not in the columns (because return from function not executed above), outg_rec_nbs must be in the indices!
        # The if/else block below determines idfr_idx_lvl
        if isinstance(outg_rec_nb_idfr, str):
            assert(outg_rec_nb_idfr.startswith('index'))
            if outg_rec_nb_idfr=='index':
                idfr_idx_lvl=0
            else:
                idfr_idx_lvl = re.findall(r'index_(\d*)', outg_rec_nb_idfr)
                assert(len(idfr_idx_lvl)==1)
                idfr_idx_lvl=idfr_idx_lvl[0]
                idfr_idx_lvl=int(idfr_idx_lvl)
        else:
            assert(len(outg_rec_nb_idfr)==2)
            assert(outg_rec_nb_idfr[0]=='index')
            idx_level_name = outg_rec_nb_idfr[1]
            assert(idx_level_name in dovs_df.index.names)
            idfr_idx_lvl = dovs_df.index.names.index(idx_level_name)
        #-------------------------
        assert(idfr_idx_lvl < dovs_df.index.nlevels)
        if assert_outg_rec_nb_found:
            assert(outg_rec_nb in dovs_df.index.get_level_values(idfr_idx_lvl).unique().tolist())
        return_df = dovs_df[dovs_df.index.get_level_values(idfr_idx_lvl)==outg_rec_nb]
        return return_df
        
        
    @staticmethod
    def get_outg_rec_nb_col_from_idfr(
        dovs_df, 
        outg_rec_nb_idfr
    ):
        r"""
        Find the outg_rec_nb column given the identifier outg_rec_nb_idfr.
            For more information, see Utilities_df.get_idfr_loc.
        Returns:
            (dovs_df, outg_rec_nb_col)
            NOTE: If outg_rec_nb information is in the index, then .reset_index will be called on dovs_df, 
                    and the returned DF will NOT equal the input DF (see description below for more info)
        If the outg_rec_nb information is found in the columns of dovs_df, then simply return the column.
        If the outg_rec_nb information is found in the index:
            - Call .reset_index() on the index level in which the outg_rec_nb information is stored.
            - If the index level did not have a name, give it a name so the outg_rec_nb column is identifiable
                after .reset_index() is called
        """
        #-------------------------
        dovs_df = dovs_df.copy()
        #-------------------------
        outg_rec_nb_idfr_loc = Utilities_df.get_idfr_loc(
            df   = dovs_df, 
            idfr = outg_rec_nb_idfr

        )
        # If the outg_rec_nbs are in the index, then reset_index must be called for
        #   many functions to run properly
        if outg_rec_nb_idfr_loc[1]:
            outg_rec_nb_idx_lvl = outg_rec_nb_idfr_loc[0]
            #-----
            if dovs_df.index.names[outg_rec_nb_idx_lvl]:
                outg_rec_nb_col = dovs_df.index.names[outg_rec_nb_idx_lvl]
            else:
                outg_rec_nb_col = 'OUTG_REC_NB_'+Utilities.generate_random_string(str_len=4)
                assert(outg_rec_nb_col not in dovs_df.columns.tolist())
                assert(outg_rec_nb_col not in list(dovs_df.index.names))
                dovs_df.index = dovs_df.index.set_names(outg_rec_nb_col, level=outg_rec_nb_idx_lvl)
            #-----
            # Set the outg_rec_nb_col and drop index
            dovs_df[outg_rec_nb_col] = dovs_df.index.get_level_values(outg_rec_nb_idx_lvl)
            if dovs_df.index.nlevels==1:
                # NOTE: Values already placed in outg_rec_nb_col above, hence why drop=True below
                dovs_df = dovs_df.reset_index(drop=True)
            else:
                dovs_df = dovs_df.droplevel(outg_rec_nb_idx_lvl, axis=0)
        else:
            outg_rec_nb_col = outg_rec_nb_idfr_loc[0]
        #-------------------------
        assert(outg_rec_nb_col in dovs_df.columns.tolist())
        #-------------------------
        return dovs_df, outg_rec_nb_col
        
        
    @staticmethod
    def build_direct_SNs_in_outgs_df(
        outg_rec_nbs, 
        use_exploded_method=True, 
        equip_typ_nms_of_interest=None
    ):
        r"""
        Can't use slim, because need to join with MeterPremise and need trsf_pole_nb for each premise in an
        outage to judge whether or not it is direct.  Therefore, consolidating at this point wouldn't work.
        HOWEVER, what would work though, is running consolidated (because it's faster) and then
        exploding it out (~6 times faster, but this should be further tested/verified, which is main
        reason for use_exploded_method is still available as switch)

        equip_typ_nms_of_interest:
          Allows user to select specific equipment types of interest.
          Typical case: equip_typ_nms_of_interest = ['TRANSFORMER, OH', 'TRANSFORMER, UG']
        """
        #-------------------------
        if not use_exploded_method:
            dovs_outgs = DOVSOutages(                 
                df_construct_type=DFConstructType.kRunSqlQuery, 
                contstruct_df_args=None, 
                init_df_in_constructor=True, 
                build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                build_sql_function_kwargs=dict(        
                    outg_rec_nbs = outg_rec_nbs, 
                    include_DOVS_EQUIPMENT_TYPES_DIM=True, 
                    include_DOVS_PREMISE_DIM=True,
                    field_to_split='outg_rec_nbs'
                ),
            )
            dovs_outgs_df = dovs_outgs.df
        else:
            dovs_outgs_slim = DOVSOutages.build_consolidated_outage(
                contstruct_df_args=None, 
                build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                build_sql_function_kwargs=dict(        
                    outg_rec_nbs = outg_rec_nbs, 
                    include_DOVS_EQUIPMENT_TYPES_DIM=True, 
                    field_to_split='outg_rec_nbs'
                ),
            )
            dovs_outgs_df = dovs_outgs_slim.df
            # Make slim df un-slim
            dovs_outgs_df=dovs_outgs_df.explode(column='premise_nbs')
            dovs_outgs_df=dovs_outgs_df.rename(columns={'premise_nbs':'PREMISE_NB'})
            dovs_outgs_df=dovs_outgs_df.reset_index()
        #-------------------------
        assert(all([x in dovs_outgs_df.columns 
                    for x in ['OUTG_REC_NB', 'LOCATION_ID', 'EQUIP_TYP_NM', 'PREMISE_NB']]))
        #-------------------------
        direct_SNs_in_outgs = DOVSOutages.build_mp_df_and_merge_with_df_outage(
            df_outage=dovs_outgs_df[['OUTG_REC_NB', 'LOCATION_ID', 'EQUIP_TYP_NM', 'PREMISE_NB']], 
            cols_of_interest_met_prem=['mfr_devc_ser_nbr', 'prem_nb', 'trsf_pole_nb'], 
            drop_cols = None, 
            rename_cols={'mfr_devc_ser_nbr':'serial_number'}, 
            inplace=False    
        )
        direct_SNs_in_outgs=direct_SNs_in_outgs[direct_SNs_in_outgs['trsf_pole_nb']==direct_SNs_in_outgs['LOCATION_ID']]
        direct_SNs_in_outgs = DOVSOutages.consolidate_df_outage(
            df_outage=direct_SNs_in_outgs, 
            outg_rec_nb_col='OUTG_REC_NB', 
            addtnl_grpby_cols=None, 
            cols_shared_by_group=['LOCATION_ID', 'EQUIP_TYP_NM'], 
            cols_to_collect_in_lists=['PREMISE_NB', 'prem_nb', 'serial_number', 'trsf_pole_nb'], 
            allow_duplicates_in_lists=False, 
            allow_NaNs_in_lists=False, 
            recover_uniqueness_violators=True, 
            gpby_dropna=True, 
            rename_cols=None,     
            premise_nb_col='serial_number', 
            premise_nbs_col='direct_serial_numbers', 
            cols_to_drop=None, 
            sort_PNs=True, 
            drop_null_premise_nbs=True, 
            set_outg_rec_nb_as_index=True,
            drop_outg_rec_nb_if_index=True, 
            verbose=False
        )
        #-------------------------
        if equip_typ_nms_of_interest is not None:
            if isinstance(equip_typ_nms_of_interest, str):
                equip_typ_nms_of_interest = [equip_typ_nms_of_interest]
            assert(Utilities.is_object_one_of_types(equip_typ_nms_of_interest, [list, tuple]))
            direct_SNs_in_outgs=direct_SNs_in_outgs[direct_SNs_in_outgs['EQUIP_TYP_NM'].isin(equip_typ_nms_of_interest)]
        #-------------------------
        return direct_SNs_in_outgs
        
        
    #TODO: Utilize this throughout MeterPremise 
    @staticmethod
    def build_mp_df_curr_hist_for_outgs(
        outg_rec_nbs, 
        join_curr_hist=False, 
        addtnl_mp_df_curr_cols=None, 
        addtnl_mp_df_hist_cols=None, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb'
    ):
        r"""
        By default, necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
          are included in both mp_df_curr and mp_df_hist.
          mp_df_curr additionally includes df_mp_trsf_pole_nb_col by default.
        Additional columns can be included using addtnl_mp_df_curr_cols/addtnl_mp_df_hist_cols
        """
        #-------------------------
        # When return_type=pd.DataFrame in DOVSOutages.get_premise_nbs_for_outages, the returned object is a DF
        #   with one premise number per row (and multiple rows for a given outage).  This is different from when
        #   return_type is dict of pd.Series, in which case the DF is essentially grouped by outage number, and the values
        #   are lists of premise numbers
        #  Using return_type=pd.DataFrame allows me to simply grab the unique premise columns using .unique().tolist(), and this
        #    method is slightly faster than the alternative (e.g., using return_type=pd.Series and calling 
        #    Utilities_df.consolidate_column_of_lists on PNs_for_outgs.to_frame())
        PNs_for_outgs = DOVSOutages.get_premise_nbs_for_outages(
            outg_rec_nbs=outg_rec_nbs, 
            return_type=pd.DataFrame, 
            col_type_outg_rec_nb=str, 
            col_type_premise_nb=None, 
            to_numeric_errors='coerce', 
            verbose=False, 
            return_premise_nbs_col='premise_nb'
        ) 
        PNs = PNs_for_outgs['premise_nb'].unique().tolist()
        #-------------------------
        return MeterPremise.build_mp_df_curr_hist_for_PNs(
            PNs=PNs, 
            join_curr_hist=join_curr_hist, 
            addtnl_mp_df_curr_cols=addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols=addtnl_mp_df_hist_cols, 
			assert_all_PNs_found=False, 
            df_mp_serial_number_col=df_mp_serial_number_col, 
            df_mp_prem_nb_col=df_mp_prem_nb_col, 
            df_mp_install_time_col=df_mp_install_time_col, 
            df_mp_removal_time_col=df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
        )
        
        
    @staticmethod
    def build_active_MP_for_outages(
        outg_rec_nbs, 
        df_mp_curr, 
        df_mp_hist, 
        addtnl_other_outg_cols=None, 
        drop_inst_rmvl_cols=False, 
        return_premise_nbs_col='premise_nbs', 
        addtnl_build_sql_function_kwargs=None, 
        col_type_outg_rec_nb=str, 
        col_type_premise_nb=None, 
        to_numeric_errors='coerce', 
        verbose=False,
        consolidate_PNs_batch_size=1000, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb', 
        dovs_outg_rec_nb_col='OUTG_REC_NB', 
        dovs_premise_nb_col='PREMISE_NB'
    ):
        r"""
        NOTE: If one takes outg_rec_nbs from a general list of outages (e.g., performing a query for all outages between 
              some date range), typically DOVSOutages.get_premise_nbs_and_others_for_outages will not return an entry for
              every outage.  The outages which are excluded may have 0 CI_NB/CMI_NB, the DT_ON_TS and/or STEP_DRTN_NB may
              be null (in either case, DT_OFF_TS_FULL will be null as well), or the outg_rec_nb may be equal to 
              -2147483648, which I'm still not full sure what that negative outage number means.
        """
        #-------------------------
        # Build PNs and others from outages
        other_outg_cols=['DT_ON_TS', 'DT_OFF_TS_FULL']
        if addtnl_other_outg_cols is not None:
            other_outg_cols = list(set(other_outg_cols+addtnl_other_outg_cols))
        PNs_for_outgs = DOVSOutages.get_premise_nbs_and_others_for_outages(
            outg_rec_nbs=outg_rec_nbs, 
            other_outg_cols=other_outg_cols, 
            return_premise_nbs_col=return_premise_nbs_col, 
            addtnl_build_sql_function_kwargs=addtnl_build_sql_function_kwargs, 
            col_type_outg_rec_nb=col_type_outg_rec_nb, 
            col_type_premise_nb=col_type_premise_nb, 
            to_numeric_errors=to_numeric_errors, 
            dovs_outg_rec_nb_col=dovs_outg_rec_nb_col, 
            dovs_premise_nb_col=dovs_premise_nb_col, 
            verbose=verbose
        )
        #-------------------------
        # From PNs_for_outgs, grab the list of PNs
        # In very limited testing, a batch size of 1000 seemed to work well here
        PNs = Utilities_df.consolidate_column_of_lists(
            df=PNs_for_outgs, 
            col=return_premise_nbs_col, 
            sort=True,
            include_None=False,
            batch_size=consolidate_PNs_batch_size, 
            verbose=False
        )
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        # If df_mp_curr or df_mp_hist are not supplied, they will be built    
        if df_mp_hist is None:
            mp_hist = MeterPremise(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                init_df_in_constructor=True, 
                build_sql_function=MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs=dict(
                    cols_of_interest=necessary_mp_cols, 
                    premise_nbs=PNs, 
                    table_name='meter_premise_hist'
                )
            )
            df_mp_hist = mp_hist.df
        #-----
        if df_mp_curr is None:
            mp_curr = MeterPremise(
                df_construct_type=DFConstructType.kRunSqlQuery, 
                init_df_in_constructor=True, 
                build_sql_function=MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs=dict(
                    cols_of_interest=necessary_mp_cols+[df_mp_trsf_pole_nb_col], 
                    premise_nbs=PNs, 
                    table_name='meter_premise'
                )
            )
            df_mp_curr = mp_curr.df
        #-------------------------
        # At a bare minimum, df_mp_curr and df_mp_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in df_mp_curr.columns for x in necessary_mp_cols+[df_mp_trsf_pole_nb_col]]))
        assert(all([x in df_mp_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        # Only reason for making dict is to ensure outg_rec_nbs are not repeated 
        active_SNs_in_outgs_dfs_dict = {}

        for outg_rec_nb_i, row_i in PNs_for_outgs.iterrows():
            active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                PNs=row_i[return_premise_nbs_col],
                df_mp_curr=df_mp_curr, 
                df_mp_hist=df_mp_hist, 
                dt_0=row_i['DT_OFF_TS_FULL'],
                dt_1=row_i['DT_ON_TS'],
                output_index=None,
                output_groupby=None, 
                assert_all_PNs_found=False
            )
            active_SNs_df_i[dovs_outg_rec_nb_col] = outg_rec_nb_i
            assert(outg_rec_nb_i not in active_SNs_in_outgs_dfs_dict)
            active_SNs_in_outgs_dfs_dict[outg_rec_nb_i] = active_SNs_df_i
        #-------------------------
        active_SNs_df = pd.concat(list(active_SNs_in_outgs_dfs_dict.values()))
        #-------------------------
        if drop_inst_rmvl_cols:
            active_SNs_df = active_SNs_df.drop(columns=[df_mp_install_time_col, df_mp_removal_time_col])
        #-------------------------
        return active_SNs_df 
        
        
    @staticmethod
    def build_active_direct_SNs_in_outgs_df(
        outg_rec_nbs, 
        use_exploded_method=True, 
        equip_typ_nms_of_interest=None, 
        mp_df_curr=None,
        mp_df_hist=None
    ):
        r"""
        Can't use slim, because need to join with MeterPremise and need trsf_pole_nb for each premise in an
        outage to judge whether or not it is direct.  Therefore, consolidating at this point wouldn't work.
        HOWEVER, what would work though, is running consolidated (because it's faster) and then
        exploding it out (~6 times faster, but this should be further tested/verified, which is main
        reason for use_exploded_method is still available as switch)

        equip_typ_nms_of_interest:
          Allows user to select specific equipment types of interest.
          Typical case: equip_typ_nms_of_interest = ['TRANSFORMER, OH', 'TRANSFORMER, UG']
        """
        #-------------------------
        if not use_exploded_method:
            dovs_outgs = DOVSOutages(                 
                df_construct_type=DFConstructType.kRunSqlQuery, 
                contstruct_df_args=None, 
                init_df_in_constructor=True, 
                build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                build_sql_function_kwargs=dict(        
                    outg_rec_nbs = outg_rec_nbs, 
                    include_DOVS_EQUIPMENT_TYPES_DIM=True, 
                    include_DOVS_PREMISE_DIM=True,
                    field_to_split='outg_rec_nbs'
                ),
            )
            dovs_outgs_df = dovs_outgs.df
        else:
            dovs_outgs_slim = DOVSOutages.build_consolidated_outage(
                contstruct_df_args=None, 
                build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                build_sql_function_kwargs=dict(        
                    outg_rec_nbs = outg_rec_nbs, 
                    include_DOVS_EQUIPMENT_TYPES_DIM=True, 
                    field_to_split='outg_rec_nbs'
                ),
            )
            dovs_outgs_df = dovs_outgs_slim.df
            # Make slim df un-slim
            dovs_outgs_df=dovs_outgs_df.explode(column='premise_nbs')
            dovs_outgs_df=dovs_outgs_df.rename(columns={'premise_nbs':'PREMISE_NB'})
            dovs_outgs_df=dovs_outgs_df.reset_index()
        #-------------------------
        assert(all([x in dovs_outgs_df.columns 
                    for x in ['OUTG_REC_NB', 'LOCATION_ID', 'EQUIP_TYP_NM', 'PREMISE_NB']]))
        #-------------------------
        active_MP = DOVSOutages.build_active_MP_for_outages(
            outg_rec_nbs = dovs_outgs_df['OUTG_REC_NB'].unique().tolist(), 
            df_mp_curr=mp_df_curr, 
            df_mp_hist=mp_df_hist, 
            drop_inst_rmvl_cols=True
        )
        direct_SNs_in_outgs = pd.merge(
            dovs_outgs_df[['OUTG_REC_NB', 'LOCATION_ID', 'EQUIP_TYP_NM', 'PREMISE_NB']], 
            active_MP, 
            how='left', 
            left_on=['OUTG_REC_NB', 'PREMISE_NB'], 
            right_on=['OUTG_REC_NB', 'prem_nb'])
        direct_SNs_in_outgs=direct_SNs_in_outgs[direct_SNs_in_outgs['trsf_pole_nb']==direct_SNs_in_outgs['LOCATION_ID']]    
        direct_SNs_in_outgs = DOVSOutages.consolidate_df_outage(
            df_outage=direct_SNs_in_outgs, 
            outg_rec_nb_col='OUTG_REC_NB', 
            addtnl_grpby_cols=None, 
            cols_shared_by_group=['LOCATION_ID', 'EQUIP_TYP_NM'], 
            cols_to_collect_in_lists=['PREMISE_NB']+active_MP.columns.tolist(), 
            allow_duplicates_in_lists=False, 
            allow_NaNs_in_lists=False, 
            recover_uniqueness_violators=True, 
            gpby_dropna=True, 
            rename_cols=None,     
            premise_nb_col='mfr_devc_ser_nbr', 
            premise_nbs_col='direct_serial_numbers', 
            cols_to_drop=None, 
            sort_PNs=True, 
            drop_null_premise_nbs=True, 
            set_outg_rec_nb_as_index=True,
            drop_outg_rec_nb_if_index=True, 
            verbose=False
        )
        #-------------------------
        if equip_typ_nms_of_interest is not None:
            if isinstance(equip_typ_nms_of_interest, str):
                equip_typ_nms_of_interest = [equip_typ_nms_of_interest]
            assert(Utilities.is_object_one_of_types(equip_typ_nms_of_interest, [list, tuple]))
            direct_SNs_in_outgs=direct_SNs_in_outgs[direct_SNs_in_outgs['EQUIP_TYP_NM'].isin(equip_typ_nms_of_interest)]
        #-------------------------
        return direct_SNs_in_outgs
        
        
    #-----------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def reduce_active_MP_for_outages(
        df_outage, 
        prem_nb_col, 
        df_mp_curr, 
        df_mp_hist, 
        drop_inst_rmvl_cols=False, 
        outg_rec_nb_idfr='OUTG_REC_NB', 
        is_slim=False, 
        drop_approx_duplicates=True, 
        drop_approx_duplicates_args=None, 
        dt_on_ts_col='DT_ON_TS', 
        df_off_ts_full_col='DT_OFF_TS_FULL', 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb'
    ):
        r"""
        Intended to be used within build_active_MP_for_outages_df and build_active_MP_for_xfmrs_in_outages_df.
        This does not mean it cannot be used elsewhere, but use with caution (understand functionality before use!)
        """
        #-------------------------
        # Life is easier if outg_rec_nb information is in columns instead of index
        # Calling DOVSOutages.get_outg_rec_nb_col_from_idfr will accomplish this
        df_outage, outg_rec_nb_col = DOVSOutages.get_outg_rec_nb_col_from_idfr(
            dovs_df          = df_outage.copy(), 
            outg_rec_nb_idfr = outg_rec_nb_idfr
        )

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
                    PNs=PNs_i,
                    df_mp_curr=df_mp_curr, 
                    df_mp_hist=df_mp_hist, 
                    dt_0=df_off_ts_full_i,
                    dt_1=dt_on_ts_i,
                    addtnl_mp_df_curr_cols=None, 
                    addtnl_mp_df_hist_cols=None,
                    assume_one_xfmr_per_PN=False, 
                    output_index=None,
                    output_groupby=None, 
                    include_prems_wo_active_SNs_when_groupby=True, 
                    assert_all_PNs_found=False, 
                    drop_approx_duplicates=drop_approx_duplicates, 
                    drop_approx_duplicates_args=drop_approx_duplicates_args, 
                    df_mp_serial_number_col=df_mp_serial_number_col, 
                    df_mp_prem_nb_col=df_mp_prem_nb_col, 
                    df_mp_install_time_col=df_mp_install_time_col, 
                    df_mp_removal_time_col=df_mp_removal_time_col, 
                    df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
                )
                active_SNs_df_i[outg_rec_nb_col] = outg_rec_nb_i
                assert(outg_rec_nb_i not in active_SNs_in_outgs_dfs_dict)
                active_SNs_in_outgs_dfs_dict[outg_rec_nb_i] = active_SNs_df_i
        else:
            for idx_i, row_i in df_outage.iterrows():
                # NOTE: assume_one_xfmr_per_PN=True above in MeterPremise.build_mp_df_curr_hist_for_PNs,
                #       so does not need to be set again (i.e., assume_one_xfmr_per_PN=False below)
                outg_rec_nb_i = row_i[outg_rec_nb_col]
                active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                    PNs=row_i[prem_nb_col],
                    df_mp_curr=df_mp_curr, 
                    df_mp_hist=df_mp_hist, 
                    dt_0=row_i[df_off_ts_full_col],
                    dt_1=row_i[dt_on_ts_col],
                    addtnl_mp_df_curr_cols=None, 
                    addtnl_mp_df_hist_cols=None,
                    assume_one_xfmr_per_PN=False, 
                    output_index=None,
                    output_groupby=None, 
                    include_prems_wo_active_SNs_when_groupby=True, 
                    assert_all_PNs_found=False, 
                    drop_approx_duplicates=drop_approx_duplicates, 
                    drop_approx_duplicates_args=drop_approx_duplicates_args, 
                    df_mp_serial_number_col=df_mp_serial_number_col, 
                    df_mp_prem_nb_col=df_mp_prem_nb_col, 
                    df_mp_install_time_col=df_mp_install_time_col, 
                    df_mp_removal_time_col=df_mp_removal_time_col, 
                    df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col                
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
        
        
    @staticmethod
    def build_active_MP_for_outages_df(
        df_outage, 
        prem_nb_col, 
        df_mp_curr=None, 
        df_mp_hist=None, 
        assert_all_PNs_found=True, 
        drop_inst_rmvl_cols=False, 
        outg_rec_nb_idfr='OUTG_REC_NB', 
        is_slim=False, 
        addtnl_mp_df_curr_cols=None, 
        addtnl_mp_df_hist_cols=None, 
        dt_on_ts_col='DT_ON_TS', 
        df_off_ts_full_col='DT_OFF_TS_FULL', 
        consolidate_PNs_batch_size=1000, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb', 
        early_return=False
    ):
        r"""
        Similar to build_active_MP_for_outages

        If addtnl_mp_df_curr_cols and addtnl_mp_df_hist_cols are included (i.e., are not None), their intersection will be added
          as addtnl_groupby_cols input argument to drop_approx_duplicates (otherwise, they would be returned in the DF as list
          objects, and would likely need to be exploded)
        """
        #-------------------------
        assert(prem_nb_col in df_outage.columns and 
               dt_on_ts_col in df_outage.columns and 
               df_off_ts_full_col in df_outage.columns)
        #-------------------------
        if not is_slim:
            PNs = df_outage[prem_nb_col].unique().tolist()
        else:
            PNs = Utilities_df.consolidate_column_of_lists(
                df=df_outage, 
                col=prem_nb_col, 
                sort=True,
                include_None=False,
                batch_size=consolidate_PNs_batch_size, 
                verbose=False
            )
        #-----
        PNs = [x for x in PNs if pd.notna(x)]
        PNs_type = type(PNs[0])
        assert(Utilities.are_all_list_elements_of_type(PNs, PNs_type))
        #-------------------------
        if addtnl_mp_df_curr_cols is not None and addtnl_mp_df_hist_cols is not None:
            assert(Utilities.is_object_one_of_types(addtnl_mp_df_curr_cols, [list, tuple]))
            assert(Utilities.is_object_one_of_types(addtnl_mp_df_hist_cols, [list, tuple]))
            drop_approx_duplicates_args = dict(
                addtnl_groupby_cols=list(set(addtnl_mp_df_curr_cols).intersection(set(addtnl_mp_df_hist_cols)))
            )
        else:
            drop_approx_duplicates_args=None
        #-----
        mp_df_curr_hist_dict = MeterPremise.build_mp_df_curr_hist_for_PNs(
            PNs=PNs, 
            mp_df_curr=df_mp_curr,
            mp_df_hist=df_mp_hist, 
            join_curr_hist=False, 
            addtnl_mp_df_curr_cols=addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols=addtnl_mp_df_hist_cols, 
            assert_all_PNs_found=assert_all_PNs_found, 
            assume_one_xfmr_per_PN=True, 
            drop_approx_duplicates=True, 
            drop_approx_duplicates_args=drop_approx_duplicates_args, 
            df_mp_serial_number_col=df_mp_serial_number_col, 
            df_mp_prem_nb_col=df_mp_prem_nb_col, 
            df_mp_install_time_col=df_mp_install_time_col, 
            df_mp_removal_time_col=df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
        )
        df_mp_curr = mp_df_curr_hist_dict['mp_df_curr']
        df_mp_hist = mp_df_curr_hist_dict['mp_df_hist']
        if early_return:
            return df_mp_curr, df_mp_hist
        #-------------------------
        active_SNs_df = DOVSOutages.reduce_active_MP_for_outages(
            df_outage=df_outage, 
            prem_nb_col=prem_nb_col, 
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist, 
            drop_inst_rmvl_cols=drop_inst_rmvl_cols, 
            outg_rec_nb_idfr=outg_rec_nb_idfr, 
            is_slim=is_slim, 
            drop_approx_duplicates=True, 
            drop_approx_duplicates_args=drop_approx_duplicates_args, 
            dt_on_ts_col=dt_on_ts_col, 
            df_off_ts_full_col=df_off_ts_full_col, 
            df_mp_serial_number_col=df_mp_serial_number_col, 
            df_mp_prem_nb_col=df_mp_prem_nb_col, 
            df_mp_install_time_col=df_mp_install_time_col, 
            df_mp_removal_time_col=df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
        )
        #-------------------------
        return active_SNs_df
    


    #-----------------------------------------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def append_zero_counts_to_dovs_outgs(
        dovs_outgs, 
        build_sql_outage_kwargs, 
        PN_col                      = 'PREMISE_NB', 
        count_col                   = 'COUNT(*)', 
        possible_premise_nbs_kwargs = ['premise_nbs', 'premise_nb', 'aep_premise_nbs', 'aep_premise_nb']
    ):
        r"""
        dovs_outgs:
            Must be either a DOVSOutages object or a pd.DataFrame.
            In either case, the underlying data (either dovs_outgs.df or dovs_outgs, depending on type)
              must have two columns, PN_col and count_col
        """
        #----------------------------------------------------------------------------------------------------
        if build_sql_outage_kwargs is None:
            return dovs_outgs
        assert(isinstance(build_sql_outage_kwargs, dict))
        #-------------------------
        if dovs_outgs is None:
            dovs_outgs = pd.DataFrame(columns=[PN_col, count_col])
        assert(Utilities.is_object_one_of_types(dovs_outgs, [DOVSOutages, pd.DataFrame]))
        #-------------------------
        found_premise_nbs_kwargs = [x for x in build_sql_outage_kwargs if x in possible_premise_nbs_kwargs]
        assert(len(found_premise_nbs_kwargs)<=1)
        if found_premise_nbs_kwargs:
            input_PNs = build_sql_outage_kwargs[found_premise_nbs_kwargs[0]]
        else:
            input_PNs = None
        #--------------------------------------------------
        if input_PNs is None:
            return dovs_outgs
        #----------------------------------------------------------------------------------------------------
        if isinstance(dovs_outgs, DOVSOutages):
            df = dovs_outgs.df.copy()
        elif isinstance(dovs_outgs, pd.DataFrame):
            df = dovs_outgs.copy()
        else:
            assert(0)
        #--------------------------------------------------
        if df is None or df.shape[0]==0:
            zero_cnt_PNs = input_PNs
            index        = range(0, len(zero_cnt_PNs))
        else:
            assert(set([PN_col, count_col]).difference(set(df.columns.tolist()))==set())
            #-----
            zero_cnt_PNs = list(set(input_PNs).difference(set(df[PN_col].unique().tolist())))
            index        = range(df.shape[0], df.shape[0]+len(zero_cnt_PNs))
        #--------------------------------------------------
        if len(zero_cnt_PNs)==0:
            return dovs_outgs
        #--------------------------------------------------
        zero_cnt_PNs = pd.DataFrame(
            data  = {
                PN_col    : zero_cnt_PNs, 
                count_col : [0]*len(zero_cnt_PNs)
            }, 
            index = index
        )
        #-----
        other_cols = list(set(df.columns.tolist()).difference(set([PN_col, count_col])))
        if len(other_cols)>0:
            zero_cnt_PNs[other_cols] = np.nan
        #--------------------------------------------------
        if df is None or df.shape[0]==0:
            df = zero_cnt_PNs
        else:
            assert(df.index[-1]+1 == zero_cnt_PNs.index[0])
            df = pd.concat([df, zero_cnt_PNs], axis=0)
        #----------------------------------------------------------------------------------------------------
        if isinstance(dovs_outgs, DOVSOutages):
            dovs_outgs.set_df(df=df)
        elif isinstance(dovs_outgs, pd.DataFrame):
            dovs_outgs = df
        else:
            assert(0)
        #-------------------------
        return dovs_outgs
    

    @staticmethod
    def append_zero_counts_to_n_outgs_per_PN_df(
        n_outgs_per_PN_df, 
        search_times_df, 
        cols_n_outgs  = None, 
        cols_srch_tms = None, 
        zero_val      = 0
    ):
        r"""
        This is a somewhat specific function.  Although it is not currently general, if needed, it could be made general.
        The basic idea is to first identify any groups which ARE contained in search_times_df but ARE NOT in n_outgs_per_PN_df.
        The groups will be formed by nt_grp_by_cols + [t_min_col, t_max_col] (nt stands for "non-time" in nt_grp_by_cols)
        These identified missing groups will then be assigned a value of zero_val (default 0, but np.nan is useful in some situations)
          and appended to n_outgs_per_PN_df
        For these, any additional columns in n_outgs_per_PN_df not found in nt_grp_by_cols + [t_min_col, t_max_col, n_outgs_col] will have
          their values set to np.nan
    
        n_outgs_per_PN_df:
            A pd.DataFrame object with the following columns:
                nt_grp_by_cols
                t_min_col
                t_max_col
                n_outgs_col
    
        search_times_df:
            A pd.DataFrame object with the following columns:
                nt_grp_by_cols
                t_min_col
                t_max_col
        
        cols_n_outgs/cols_srch_tms:
            dict objects standardizing the expected column locations
            See dflt_cols_n_outgs/dflt_cols_srch_tms for example
        """
        #----------------------------------------------------------------------------------------------------
        dflt_cols_n_outgs = dict(
            nt_grp_by_cols   = ['TRSF_POLE_NB'], 
            t_min_col        = 'T_SEARCH_MIN', 
            t_max_col        = 'T_SEARCH_MAX', 
            n_outgs_col      = 'COUNT(*)'
        )
        #-------------------------
        if cols_n_outgs is None:
            cols_n_outgs = dflt_cols_n_outgs
        else:
            assert(isinstance(cols_n_outgs, dict))
            cols_n_outgs = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = cols_n_outgs, 
                default_values_dict = dflt_cols_n_outgs, 
                extend_any_lists    = False, 
                inplace             = False
            )
        #-------------------------
        # Make sure all needed columns are found
        cols_n_outgs['needed_cols'] = Utilities.melt_list_of_lists_2(list(cols_n_outgs.values()))
        assert(set(cols_n_outgs['needed_cols']).difference(set(n_outgs_per_PN_df.columns.tolist())) == set())
        cols_n_outgs['other_cols']  = list(set(n_outgs_per_PN_df.columns.tolist()).difference(set(cols_n_outgs['needed_cols'])))
        
        #--------------------------------------------------
        dflt_cols_srch_tms = dict(
            nt_grp_by_cols   = ['trsf_pole_nb'], 
            t_min_col        = 't_min', 
            t_max_col        = 't_max'
        )
        #-----
        if cols_srch_tms is None:
            cols_srch_tms = dflt_cols_srch_tms
        else:
            assert(isinstance(cols_srch_tms, dict))
            cols_srch_tms = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = cols_srch_tms, 
                default_values_dict = dflt_cols_srch_tms, 
                extend_any_lists    = False, 
                inplace             = False
            )
        #-----
        # Make sure all needed columns are found
        cols_srch_tms['needed_cols'] = Utilities.melt_list_of_lists_2(list(cols_srch_tms.values()))
        assert(set(cols_srch_tms['needed_cols']).difference(set(search_times_df.columns.tolist())) == set())
        
        
        #----------------------------------------------------------------------------------------------------
        # Make sure the time columns in n_outgs_per_PN_df are actually of time type
        #   NOTE: This will also ensure the same for search_times_df through the Utilities_df.make_df_col_dtypes_equal call below
        n_outgs_per_PN_df = Utilities_df.convert_col_types(
            df                  = n_outgs_per_PN_df, 
            cols_and_types_dict = {
                cols_n_outgs['t_min_col'] : datetime.datetime, 
                cols_n_outgs['t_max_col'] : datetime.datetime
            },
            to_numeric_errors   = 'coerce',
            inplace             = True,
        )
        #--------------------------------------------------
        assert(isinstance(cols_n_outgs['nt_grp_by_cols'], list))
        grp_by_cols_n_outgs = cols_n_outgs['nt_grp_by_cols'] + [cols_n_outgs['t_min_col'], cols_n_outgs['t_max_col']]
        #-----
        assert(isinstance(cols_srch_tms['nt_grp_by_cols'], list))
        grp_by_cols_srch_tms = cols_srch_tms['nt_grp_by_cols'] + [cols_srch_tms['t_min_col'], cols_srch_tms['t_max_col']]
        
        #--------------------------------------------------
        n_outgs_per_PN_df, search_times_df = Utilities_df.make_df_col_dtypes_equal(
            df_1              = n_outgs_per_PN_df, 
            col_1             = grp_by_cols_n_outgs, 
            df_2              = search_times_df, 
            col_2             = grp_by_cols_srch_tms, 
            allow_reverse_set = True, 
            assert_success    = True, 
            inplace           = False
        )
        assert(n_outgs_per_PN_df[grp_by_cols_n_outgs].dtypes.values.tolist()==search_times_df[grp_by_cols_srch_tms].dtypes.values.tolist())
        #--------------------------------------------------
        zero_cnt_grps = list(
            set(list(search_times_df.groupby(grp_by_cols_srch_tms).groups.keys())).difference(
                set(list(n_outgs_per_PN_df.groupby(grp_by_cols_n_outgs).groups.keys()))
            )
        )
        if len(zero_cnt_grps)>0:
            zero_cnt_df = pd.DataFrame(
                data    = zero_cnt_grps, 
                columns = grp_by_cols_n_outgs, 
                index   = range(n_outgs_per_PN_df.shape[0], n_outgs_per_PN_df.shape[0]+len(zero_cnt_grps))
            )
            #-----
            if len(cols_n_outgs['other_cols'])>0:
                zero_cnt_df[cols_n_outgs['other_cols']] = np.nan
            zero_cnt_df[cols_n_outgs['n_outgs_col']] = zero_val
            #-------------------------
            assert(set(n_outgs_per_PN_df.columns.tolist()).symmetric_difference(set(zero_cnt_df.columns.tolist()))==set())
            n_outgs_per_PN_df = Utilities_df.concat_dfs(
                dfs                  = [n_outgs_per_PN_df, zero_cnt_df], 
                axis                 = 0, 
                make_col_types_equal = True
            )
        #----------------------------------------------------------------------------------------------------
        return n_outgs_per_PN_df


    @staticmethod
    def get_n_outgs_per_PN(
        build_sql_outage_kwargs = None, 
        include_zero_counts     = True, 
        save_args               = False, 
        verbose                 = False, 
        return_full_obj         = False, 
    ):
        r"""
        !!!!! IMPORTANT !!!!!
            If, e.g., the data acquistion needs to proceed in batches, one may include the field_to_split parameter
            inside of build_sql_outage_kwargs to instruct the DOVSOutages/GenAn object to to operate as desired.
                e.g., build_sql_outage_kwargs = dict(field_to_split = 'outg_rec_nbs')
        !!!!!!!!!!!!!!!!!!!!!

        include_zero_counts:
            Only affects functionality when premise numbers are found in build_sql_outage_kwargs.
            When this is true, 
                include_zero_counts==True : assign a value of 0 to any premise numbers not returned in dovs_outgs
        """
        #--------------------------------------------------
        if build_sql_outage_kwargs is None:
            build_sql_outage_kwargs = {}
        assert(isinstance(build_sql_outage_kwargs, dict))
        build_sql_outage_kwargs['ignore_index'] = build_sql_outage_kwargs.get('ignore_index', True)
        #-------------------------
        possible_premise_nbs_kwargs = ['premise_nbs', 'premise_nb', 'aep_premise_nbs', 'aep_premise_nb']
        found_premise_nbs_kwargs = [x for x in build_sql_outage_kwargs if x in possible_premise_nbs_kwargs]
        assert(len(found_premise_nbs_kwargs)<=1)
        if found_premise_nbs_kwargs:
            build_sql_outage_kwargs['field_to_split'] = found_premise_nbs_kwargs[0]
            build_sql_outage_kwargs['batch_size']     = 1000
        #--------------------------------------------------
        dovs_outgs = DOVSOutages(                 
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = dict(read_sql_args=None), 
            init_df_in_constructor    = True, 
            build_sql_function        = DOVSOutages_SQL.build_sql_n_outgs_per_PN, 
            build_sql_function_kwargs = build_sql_outage_kwargs,
            save_args                 = save_args, 
            build_consolidated        = False
        )
        #-------------------------
        if include_zero_counts:
            dovs_outgs = DOVSOutages.append_zero_counts_to_dovs_outgs(
                dovs_outgs                  = dovs_outgs, 
                build_sql_outage_kwargs     = build_sql_outage_kwargs, 
                PN_col                      = 'PREMISE_NB', 
                count_col                   = 'COUNT(*)', 
                possible_premise_nbs_kwargs = possible_premise_nbs_kwargs
            )
        #-------------------------
        if return_full_obj:
            return dovs_outgs
        else:
            return dovs_outgs.df
        

    @staticmethod
    def get_n_outgs_per_xfmr_OLD(
        date_0, 
        date_1, 
        build_sql_outage_kwargs = None, 
        save_args               = False, 
        verbose                 = False
    ):
        r"""
        One may:
            1. explicitly supply the trsf_pole_nbs desired, OR 
            2. Find premises suffering outages between [date_0, date_1] and satisfying any other restrictions contained in build_sql_outage_kwargs.
               Then, working back to find the corresponding trsf_pole_nbs for the premises, and finally returning the outage counts.

        date_0/_1:
            These will be input into build_sql_outage_kwargs (and will replace the corresponding keys, if they exist)
            The MeterPremise.get_active_SNs_for_PNs_at_datetime_interval specifically requires them, which is why they 
              are explicitly required here.
        
        !!!!! IMPORTANT !!!!!
            If, e.g., the data acquistion needs to proceed in batches, one may include the field_to_split parameter
            inside of build_sql_outage_kwargs to instruct the DOVSOutages/GenAn object to to operate as desired.
                e.g., build_sql_outage_kwargs = dict(field_to_split = 'outg_rec_nbs')
        !!!!!!!!!!!!!!!!!!!!!
    
        Function description:
            1. DOVSOutages.get_n_outgs_per_PN is utilized to find the number of outages suffered by each premise during
                 the time interval [date_0, date_1]
               The output of this step is n_outgs_per_PN_df
            2. Input the premises in n_outgs_per_PN_df to the function MeterPremise.get_active_SNs_for_PNs_at_datetime_interval.
               Obtain mp_df, which contains the mapping of PN to trsf_pole_nb
            3. Merge n_outgs_per_PN_df and mp_df to obtain fnl_df
            4. Build fnl_srs by grouping n_outgs_per_PN_df by 'trsf_pole_nb' and returning the maximum value of 'n_outgs' for each group
        """
        #--------------------------------------------------
        if build_sql_outage_kwargs is None:
            build_sql_outage_kwargs = {}
        assert(isinstance(build_sql_outage_kwargs, dict))
        build_sql_outage_kwargs['date_range'] = [date_0, date_1]
        #--------------------------------------------------
        # 1. 
        #-----
        dovs_outgs = DOVSOutages.get_n_outgs_per_PN(
            build_sql_outage_kwargs = build_sql_outage_kwargs, 
            save_args               = save_args, 
            verbose                 = verbose, 
            return_full_obj         = True
        )
        #-----
        n_outgs_per_PN_df = dovs_outgs.df
        n_outgs_per_PN_df = n_outgs_per_PN_df.rename(columns={'COUNT(*)':'n_outgs'})
    
        #--------------------------------------------------
        # 2. 
        #-----
        mp_df = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
            PNs                                      = n_outgs_per_PN_df['PREMISE_NB'].unique().tolist(), 
            df_mp_curr                               = None, 
            df_mp_hist                               = None, 
            dt_0                                     = date_0, 
            dt_1                                     = date_1, 
            addtnl_mp_df_curr_cols                   = None, 
            addtnl_mp_df_hist_cols                   = None,
            assume_one_xfmr_per_PN                   = True, 
            output_index                             = ['trsf_pole_nb', 'prem_nb'],
            output_groupby                           = None, 
            include_prems_wo_active_SNs_when_groupby = True, 
            assert_all_PNs_found                     = False, 
            drop_approx_duplicates                   = True, 
            drop_approx_duplicates_args              = None, 
            df_mp_serial_number_col                  = 'mfr_devc_ser_nbr', 
            df_mp_prem_nb_col                        = 'prem_nb', 
            df_mp_install_time_col                   = 'inst_ts', 
            df_mp_removal_time_col                   = 'rmvl_ts', 
            df_mp_trsf_pole_nb_col                   = 'trsf_pole_nb'
        )
        mp_df = mp_df.reset_index()
        
        #--------------------------------------------------
        # 3. 
        #-----
        fnl_df = pd.merge(
            n_outgs_per_PN_df, 
            mp_df, 
            how      = 'left', 
            left_on  = 'PREMISE_NB', 
            right_on = 'prem_nb'
        )
    
        #--------------------------------------------------
        # 4. 
        #-----
        fnl_srs = fnl_df.groupby(['trsf_pole_nb'])['n_outgs'].max()
        fnl_srs = fnl_srs.sort_index()
        #--------------------------------------------------
        #--------------------------------------------------
        return fnl_srs
    

    @staticmethod
    def get_n_outgs_per_xfmr(
        date_0, 
        date_1, 
        trsf_pole_nbs           = None, 
        trsf_pole_nbs_to_ignore = [' ', 'TRANSMISSION', 'PRIMARY', 'NETWORK'], 
        include_zero_counts     = True, 
        build_sql_outage_kwargs = None, 
        save_args               = False, 
        verbose                 = False, 
        return_df               = False
    ):
        r"""
        One may:
            1. explicitly supply the trsf_pole_nbs desired, OR 
            2. Find premises suffering outages between [date_0, date_1] and satisfying any other restrictions contained in build_sql_outage_kwargs.
               Then, working back to find the corresponding trsf_pole_nbs for the premises, and finally returning the outage counts.
    
        date_0/_1:
            These will be input into build_sql_outage_kwargs (and will replace the corresponding keys, if they exist)
            The MeterPremise.get_active_SNs_for_PNs_at_datetime_interval specifically requires them, which is why they 
              are explicitly required here.

        include_zero_counts:
            Only affects functionality when trsf_pole_nbs is not None.
            When this is true and include_zero_counts==True : 
                1. dovs_outgs = DOVSOutages.get_n_outgs_per_PN(...) will contain all active PNs found, with some likely
                     containing n_outages=0 counts.
                2. After fnl_df is built by merging n_outgs_per_PN_df and active_SNs_df any trsf_pole_nbs (from input argument) missing from
                     fnl_df are appended with n_outgs = np.nan
                   These are assigned np.nan values because no active premise numbers were found for the transformers
        
        !!!!! IMPORTANT !!!!!
            If, e.g., the data acquistion needs to proceed in batches, one may include the field_to_split parameter
            inside of build_sql_outage_kwargs to instruct the DOVSOutages/GenAn object to to operate as desired.
                e.g., build_sql_outage_kwargs = dict(field_to_split = 'outg_rec_nbs')
        !!!!!!!!!!!!!!!!!!!!!
    
        Function description:
            1. DOVSOutages.get_n_outgs_per_PN is utilized to find the number of outages suffered by each premise during
                 the time interval [date_0, date_1]
               The output of this step is n_outgs_per_PN_df
            2. Input the premises in n_outgs_per_PN_df to the function MeterPremise.get_active_SNs_for_PNs_at_datetime_interval.
               Obtain mp_df, which contains the mapping of PN to trsf_pole_nb
            3. Merge n_outgs_per_PN_df and mp_df to obtain fnl_df
            4. Build fnl_srs by grouping n_outgs_per_PN_df by 'trsf_pole_nb' and returning the maximum value of 'n_outgs' for each group
        """
        #--------------------------------------------------
        if build_sql_outage_kwargs is None:
            build_sql_outage_kwargs = {}
        assert(isinstance(build_sql_outage_kwargs, dict))
        build_sql_outage_kwargs['date_range'] = [date_0, date_1]
        #--------------------------------------------------
        if trsf_pole_nbs_to_ignore is not None:
            assert(isinstance(trsf_pole_nbs_to_ignore, list))
        else:
            trsf_pole_nbs_to_ignore = []
        #--------------------------------------------------
        if trsf_pole_nbs is not None:
            # Find the corresponding PNs, and insert into build_sql_outage_kwargs
            assert(isinstance(trsf_pole_nbs, list))
            trsf_pole_nbs = [x for x in trsf_pole_nbs if x not in trsf_pole_nbs_to_ignore]
            #-------------------------
            active_SNs_df = MeterPremise.get_active_SNs_and_PNs_for_xfmrs_at_datetime_interval(
                trsf_pole_nbs                            = trsf_pole_nbs,
                dt_0                                     = date_0, 
                dt_1                                     = date_1, 
                addtnl_mp_df_curr_cols                   = None, 
                addtnl_mp_df_hist_cols                   = None,
                assume_one_xfmr_per_PN                   = True, 
                assert_all_trsf_pole_nbs_found           = False, 
                drop_approx_duplicates                   = True, 
                drop_approx_duplicates_args              = None, 
                df_mp_serial_number_col                  = 'mfr_devc_ser_nbr', 
                df_mp_prem_nb_col                        = 'prem_nb', 
                df_mp_install_time_col                   = 'inst_ts', 
                df_mp_removal_time_col                   = 'rmvl_ts', 
                df_mp_trsf_pole_nb_col                   = 'trsf_pole_nb', 
                return_PNs_only                          = False
            )
            #-----
            active_PNs = active_SNs_df['prem_nb'].unique().tolist()
            #-----
            build_sql_outage_kwargs['premise_nbs']    = active_PNs
            build_sql_outage_kwargs['field_to_split'] = 'premise_nbs'
        #--------------------------------------------------
        # 1. 
        #-----
        dovs_outgs = DOVSOutages.get_n_outgs_per_PN(
            build_sql_outage_kwargs = build_sql_outage_kwargs, 
            include_zero_counts     = include_zero_counts, 
            save_args               = save_args, 
            verbose                 = verbose, 
            return_full_obj         = True
        )
        #-----
        n_outgs_per_PN_df = dovs_outgs.df
        n_outgs_per_PN_df = n_outgs_per_PN_df.rename(columns={'COUNT(*)':'n_outgs'})
        #-----
        n_outgs_per_PN_df['t_min'] = pd.to_datetime(date_0)
        n_outgs_per_PN_df['t_max'] = pd.to_datetime(date_1)
        #--------------------------------------------------
        # 2/3. 
        #-----
        if trsf_pole_nbs is not None:
            fnl_df = MeterPremise.simple_merge_df_with_active_mp(
                df                              = n_outgs_per_PN_df, 
                df_mp                           = active_SNs_df, 
                df_time_col_0                   = 't_min',
                df_time_col_1                   = 't_max',
                df_and_mp_merge_pairs           = [
                    ['PREMISE_NB', 'prem_nb']
                ], 
                keep_overlap                    = 'right', 
                drop_inst_rmvl_cols_after_merge = True, 
                df_mp_install_time_col          = 'inst_ts', 
                df_mp_removal_time_col          = 'rmvl_ts', 
                assert_max_one_to_one           = False
            )
            if include_zero_counts:
                no_cnt_trsf_pole_nbs = list(set(trsf_pole_nbs).difference(set(fnl_df['trsf_pole_nb'].unique().tolist())))
                if len(no_cnt_trsf_pole_nbs)>0:
                    index = range(fnl_df.shape[0], fnl_df.shape[0]+len(no_cnt_trsf_pole_nbs))
                    #-----
                    no_cnt_df = pd.DataFrame(
                        data    = np.nan, 
                        columns = fnl_df.columns.tolist(), 
                        index   = index
                    )
                    no_cnt_df['trsf_pole_nb'] = no_cnt_trsf_pole_nbs
                    #-----
                    fnl_df = Utilities_df.concat_dfs(
                        dfs                  = [fnl_df, no_cnt_df], 
                        axis                 = 0, 
                        make_col_types_equal = True
                    )

        else:
            fnl_df = MeterPremise.merge_df_with_active_mp(
                df                              = n_outgs_per_PN_df, 
                df_time_col_0                   = 't_min',
                df_time_col_1                   = 't_max',
                df_mp_curr                      = None, 
                df_mp_hist                      = None, 
                df_and_mp_merge_pairs           = [
                    ['PREMISE_NB', 'prem_nb']
                ], 
                keep_overlap                    = 'right', 
                drop_inst_rmvl_cols_after_merge = True, 
                addtnl_mp_df_curr_cols          = None, 
                addtnl_mp_df_hist_cols          = None,
                assume_one_xfmr_per_PN          = True, 
                assert_all_PNs_found            = False, 
                drop_approx_duplicates          = True, 
                drop_approx_duplicates_args     = None, 
                df_prem_nb_col                  = 'PREMISE_NB', 
                df_mp_serial_number_col         = 'mfr_devc_ser_nbr', 
                df_mp_prem_nb_col               = 'prem_nb', 
                df_mp_install_time_col          = 'inst_ts', 
                df_mp_removal_time_col          = 'rmvl_ts', 
                df_mp_trsf_pole_nb_col          = 'trsf_pole_nb', 
                assert_max_one_to_one           = False
            )
        #--------------------------------------------------
        # 4. 
        #-----
        fnl_srs = fnl_df.groupby(['trsf_pole_nb'])['n_outgs'].max().sort_index()
        #--------------------------------------------------
        if trsf_pole_nbs_to_ignore is not None:
            fnl_srs = fnl_srs[~fnl_srs.index.isin(trsf_pole_nbs_to_ignore)].copy()

        #--------------------------------------------------
        if return_df:
            return fnl_srs, fnl_df
        return fnl_srs
    

    @staticmethod
    def get_n_outgs_per_xfmr_in_df_OLD(
        search_times_df, 
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        t_min_col                      = 't_search_min', 
        t_max_col                      = 't_search_max', 
        trsf_pole_nbs_to_ignore        = [' ', 'TRANSMISSION', 'PRIMARY', 'NETWORK'], 
        include_zero_counts            = True, 
        build_sql_outage_kwargs        = None, 
        assert_all_trsf_pole_nbs_found = False, 
        save_args                      = False, 
        batch_size                     = 30, 
        n_update                       = 1, 
        verbose                        = False, 
        return_df                      = False
    ):
        r"""    
        THIS VERSION IS VERY INEFFECIENT COMPARED TO THE NEW VERSION!
        THE NEW METHOD IS ALSO SUPERIOR IN TERMS OF THE CREATION OF n_outgs_per_PN_df (which is also 
          returned when return_df=True)

        df:
            Should be a pd.DataFrame object with a single row for each (trsf_pole_nb, t_min, t_max) combination.
        
        !!!!! IMPORTANT !!!!!
            If, e.g., the data acquistion needs to proceed in batches, one may include the field_to_split parameter
            inside of build_sql_outage_kwargs to instruct the DOVSOutages/GenAn object to to operate as desired.
                e.g., build_sql_outage_kwargs = dict(field_to_split = 'outg_rec_nbs')
        !!!!!!!!!!!!!!!!!!!!!
        """
        #--------------------------------------------------
        nec_cols = [trsf_pole_nb_col, t_min_col, t_max_col]
        assert(set(nec_cols).difference(set(search_times_df.columns.tolist()))==set())
        #--------------------------------------------------
        if build_sql_outage_kwargs is None:
            build_sql_outage_kwargs = {}
        assert(isinstance(build_sql_outage_kwargs, dict))
        #--------------------------------------------------
        if trsf_pole_nbs_to_ignore is not None:
            assert(isinstance(trsf_pole_nbs_to_ignore, list))
            search_times_df = search_times_df[~search_times_df[trsf_pole_nb_col].isin(trsf_pole_nbs_to_ignore)]
        #----------------------------------------------------------------------------------------------------
        # Find the active PNs
        #-------------------------
        active_PNs_df = MeterPremise.get_active_SNs_and_PNs_for_xfmrs_in_df(
            df                              = search_times_df,
            trsf_pole_nb_col                = trsf_pole_nb_col, 
            t_min_col                       = t_min_col, 
            t_max_col                       = t_max_col, 
            addtnl_mp_df_curr_cols          = None, 
            addtnl_mp_df_hist_cols          = None,
            assume_one_xfmr_per_PN          = True, 
            assert_all_trsf_pole_nbs_found  = assert_all_trsf_pole_nbs_found, 
            drop_approx_duplicates          = True, 
            drop_approx_duplicates_args     = None, 
            df_mp_serial_number_col         = 'mfr_devc_ser_nbr', 
            df_mp_prem_nb_col               = 'prem_nb', 
            df_mp_install_time_col          = 'inst_ts', 
            df_mp_removal_time_col          = 'rmvl_ts', 
            df_mp_trsf_pole_nb_col          = 'trsf_pole_nb', 
            verbose                         = verbose
        )
        #-------------------------
        # Consolidate active_PNs_df so it may be used with 
        #   DOVSOutages_SQL.build_sql_n_outgs_per_PN_for_xfmrs_in_consolidated_df
        active_PNs_df = Utilities_df.consolidate_df(
            df                                  = active_PNs_df, 
            groupby_cols                        = [trsf_pole_nb_col, t_min_col, t_max_col], 
            cols_shared_by_group                = None, 
            cols_to_collect_in_lists            = ['mfr_devc_ser_nbr', 'prem_nb'], 
            as_index                            = False, 
            include_groupby_cols_in_output_cols = False, 
            allow_duplicates_in_lists           = False, 
            allow_NaNs_in_lists                 = False, 
            recover_uniqueness_violators        = True, 
            gpby_dropna                         = True, 
            rename_cols                         = None, 
            custom_aggs_for_list_cols           = None, 
            verbose                             = verbose
        )
        
        #----------------------------------------------------------------------------------------------------
        # Utilize DOVSOutages_SQL.build_sql_n_outgs_per_PN_for_xfmrs_in_consolidated_df
        #   to find the number of outages per (active) PN
        #-------------------------
        build_sql_for_xfmrs_in_df_kwargs = dict(
            df_cnsldtd                        = active_PNs_df, 
            trsf_pole_nb_col                  = trsf_pole_nb_col, 
            t_min_col                         = t_min_col, 
            t_max_col                         = t_max_col, 
            PNs_col                           = 'prem_nb', 
            trsf_pole_nbs_to_ignore           = trsf_pole_nbs_to_ignore, 
            build_sql_outage_kwargs           = build_sql_outage_kwargs, 
            # GenAn - keys_to_pop in GenAn.build_sql_general
            field_to_split                    = 'df_cnsldtd', 
            field_to_split_location_in_kwargs = ['df_cnsldtd'], 
            # save_and_dump                     = True, 
            save_and_dump                     = False, 
            sort_coll_to_split                = True,
            batch_size                        = batch_size, 
            verbose                           = verbose, 
            n_update                          = n_update, 
            ignore_index                      = True
        )
        #-------------------------
        dovs_obj = DOVSOutages(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = dict(read_sql_args=None), 
            build_sql_function        = DOVSOutages_SQL.build_sql_n_outgs_per_PN_for_xfmrs_in_consolidated_df, 
            build_sql_function_kwargs = build_sql_for_xfmrs_in_df_kwargs, 
            init_df_in_constructor    = True, 
            save_args                 = save_args
        )
        
        #----------------------------------------------------------------------------------------------------
        # Grab n_outgs_per_PN_df and rename columns
        #-----
        n_outgs_per_PN_df = dovs_obj.df
        #-----
        # Apparently Oracle always puts things in all caps...
        n_outgs_per_PN_df = n_outgs_per_PN_df.rename(columns={
            'COUNT(*)'               : 'n_outgs', 
            trsf_pole_nb_col.upper() : trsf_pole_nb_col, 
            t_min_col.upper()        : t_min_col, 
            t_max_col.upper()        : t_max_col, 
        })
        #-----
        n_outgs_per_PN_df = Utilities_df.convert_col_types(
            df                  = n_outgs_per_PN_df, 
            cols_and_types_dict = {
                t_min_col : datetime.datetime, 
                t_max_col : datetime.datetime
            },
            to_numeric_errors   = 'coerce',
            inplace             = True,
        )
    
        #----------------------------------------------------------------------------------------------------
        # Append zero and np.nan counts if include_zero_counts==True
        #-----
        if include_zero_counts:
            cols_n_outgs = dict(
                nt_grp_by_cols = [trsf_pole_nb_col], 
                t_min_col      = t_min_col, 
                t_max_col      = t_max_col, 
                n_outgs_col    = 'n_outgs'
            )
            cols_srch_tms = dict(
                nt_grp_by_cols = [trsf_pole_nb_col], 
                t_min_col      = t_min_col, 
                t_max_col      = t_max_col
            )
            #----------
            # Include zero counts
            n_outgs_per_PN_df = DOVSOutages.append_zero_counts_to_n_outgs_per_PN_df(
                n_outgs_per_PN_df = n_outgs_per_PN_df, 
                search_times_df   = active_PNs_df, 
                cols_n_outgs      = cols_n_outgs, 
                cols_srch_tms     = cols_srch_tms, 
                zero_val          = 0
            )
            #----------
            # Include np.nan counts
            n_outgs_per_PN_df = DOVSOutages.append_zero_counts_to_n_outgs_per_PN_df(
                n_outgs_per_PN_df = n_outgs_per_PN_df, 
                search_times_df   = search_times_df, 
                cols_n_outgs      = cols_n_outgs, 
                cols_srch_tms     = cols_srch_tms, 
                zero_val          = np.nan
        )
        
        #----------------------------------------------------------------------------------------------------
        # Finalize!
        #-----
        fnl_srs = n_outgs_per_PN_df.groupby([trsf_pole_nb_col, t_min_col, t_max_col])['n_outgs'].max().sort_index()
        #-----
        if trsf_pole_nbs_to_ignore is not None:
            fnl_srs = fnl_srs[~fnl_srs.index.get_level_values(trsf_pole_nb_col).isin(trsf_pole_nbs_to_ignore)].copy()
    
        #--------------------------------------------------
        if return_df:
            return fnl_srs, n_outgs_per_PN_df
        return fnl_srs
    

    @staticmethod
    def get_n_outgs_per_xfmr_in_df_helper(
        active_PNs_df_i, 
        search_times_df, 
        include_zero_counts = True, 
        t_min_col           = 't_min', 
        t_max_col           = 't_max', 
        PN_col              = 'prem_nb', 
        trsf_pole_nb_col    = 'trsf_pole_nb'
    ):
        r"""
        """
        #--------------------------------------------------
        assert(active_PNs_df_i[t_min_col].nunique()==1)
        assert(active_PNs_df_i[t_max_col].nunique()==1)
        t_min = active_PNs_df_i[t_min_col].unique().tolist()[0]
        t_max = active_PNs_df_i[t_max_col].unique().tolist()[0]
        #-------------------------
        build_sql_outage_kwargs_i = dict(
            premise_nbs = active_PNs_df_i[PN_col].unique().tolist(), 
            date_range  = [t_min, t_max], 
            addtnl_const_selects = None
        )
        #--------------------------------------------------
        n_outgs_per_PN_i = DOVSOutages.get_n_outgs_per_PN(
            build_sql_outage_kwargs = build_sql_outage_kwargs_i, 
            include_zero_counts     = include_zero_counts, 
            save_args               = False, 
            verbose                 = False, 
            return_full_obj         = False, 
        )
        #--------------------------------------------------
        n_outgs_per_PN_i = n_outgs_per_PN_i.rename(columns={
            'COUNT(*)'   : 'n_outgs', 
            'PREMISE_NB' : PN_col, 
        })
        #-------------------------
        n_outgs_per_PN_i = pd.merge(
            active_PNs_df_i, 
            n_outgs_per_PN_i,
            how              = 'left', 
            left_on          = [PN_col], 
            right_on         = [PN_col]
        )
        if include_zero_counts:
            n_outgs_per_PN_i['n_outgs'] = n_outgs_per_PN_i['n_outgs'].fillna(0)
        else:
            n_outgs_per_PN_i = n_outgs_per_PN_i.dropna(subset=['n_outgs'])
        #--------------------------------------------------
        if include_zero_counts:
            # The above procedure handled PNs which were found but had no outages (n_outgs=0)
            # Below will handle the case for trsf_pole_nbs for which no premises were found, and which
            #   will be filled with n_outgs = np.nan
            expctd_xfmrs = search_times_df[
                (search_times_df[t_min_col] >= t_min) & 
                (search_times_df[t_max_col] <= t_max)
            ][trsf_pole_nb_col].unique().tolist()
            #-----
            found_xfmrs  = n_outgs_per_PN_i[trsf_pole_nb_col].unique().tolist()
            #-----
            no_cnt_trsf_pole_nbs = list(set(expctd_xfmrs).difference(set(found_xfmrs)))
            if len(no_cnt_trsf_pole_nbs)>0:
                index = range(n_outgs_per_PN_i.shape[0], n_outgs_per_PN_i.shape[0]+len(no_cnt_trsf_pole_nbs))
                #-----
                no_cnt_df = pd.DataFrame(
                    data    = np.nan, 
                    columns = n_outgs_per_PN_i.columns.tolist(), 
                    index   = index
                )
                no_cnt_df[trsf_pole_nb_col] = no_cnt_trsf_pole_nbs
                no_cnt_df[t_min_col]        = t_min
                no_cnt_df[t_max_col]        = t_max
                #-----
                n_outgs_per_PN_i = Utilities_df.concat_dfs(
                    dfs                  = [n_outgs_per_PN_i, no_cnt_df], 
                    axis                 = 0, 
                    make_col_types_equal = True
                )
        #--------------------------------------------------
        return n_outgs_per_PN_i
    

    @staticmethod
    def get_n_outgs_per_xfmr_in_df(
        search_times_df, 
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        t_min_col                      = 't_search_min', 
        t_max_col                      = 't_search_max', 
        trsf_pole_nbs_to_ignore        = [' ', 'TRANSMISSION', 'PRIMARY', 'NETWORK'], 
        include_zero_counts            = True, 
        build_sql_outage_kwargs        = None, 
        assert_all_trsf_pole_nbs_found = False, 
        save_args                      = False, 
        batch_size                     = 30, 
        n_update                       = 1, 
        verbose                        = False, 
        return_df                      = False
    ):
        r"""    
        df:
            Should be a pd.DataFrame object with a single row for each (trsf_pole_nb, t_min, t_max) combination.

        NOTE: Only the date (not full datetime) of tmin/tmax is ultimately used!
              The main purpose is to speed up the DAQ
        
        !!!!! IMPORTANT !!!!!
            If, e.g., the data acquistion needs to proceed in batches, one may include the field_to_split parameter
            inside of build_sql_outage_kwargs to instruct the DOVSOutages/GenAn object to to operate as desired.
                e.g., build_sql_outage_kwargs = dict(field_to_split = 'outg_rec_nbs')
        !!!!!!!!!!!!!!!!!!!!!
        """
        #--------------------------------------------------
        nec_cols = [trsf_pole_nb_col, t_min_col, t_max_col]
        assert(set(nec_cols).difference(set(search_times_df.columns.tolist()))==set())
        #--------------------------------------------------
        if build_sql_outage_kwargs is None:
            build_sql_outage_kwargs = {}
        assert(isinstance(build_sql_outage_kwargs, dict))
        #--------------------------------------------------
        if trsf_pole_nbs_to_ignore is not None:
            assert(isinstance(trsf_pole_nbs_to_ignore, list))
            search_times_df = search_times_df[~search_times_df[trsf_pole_nb_col].isin(trsf_pole_nbs_to_ignore)]
        #----------------------------------------------------------------------------------------------------
        # Find the active PNs
        #-------------------------
        search_times_df = Utilities_df.convert_col_types(
            df                  = search_times_df, 
            cols_and_types_dict = {
                t_min_col : datetime.datetime, 
                t_min_col : datetime.datetime
            },
            to_numeric_errors   = 'coerce',
            inplace             = True,
        )
        #-------------------------
        active_PNs_df = MeterPremise.get_active_SNs_and_PNs_for_xfmrs_in_df(
            df                              = search_times_df,
            trsf_pole_nb_col                = trsf_pole_nb_col, 
            t_min_col                       = t_min_col, 
            t_max_col                       = t_max_col, 
            addtnl_mp_df_curr_cols          = None, 
            addtnl_mp_df_hist_cols          = None,
            assume_one_xfmr_per_PN          = True, 
            assert_all_trsf_pole_nbs_found  = assert_all_trsf_pole_nbs_found, 
            drop_approx_duplicates          = True, 
            drop_approx_duplicates_args     = None, 
            df_mp_serial_number_col         = 'mfr_devc_ser_nbr', 
            df_mp_prem_nb_col               = 'prem_nb', 
            df_mp_install_time_col          = 'inst_ts', 
            df_mp_removal_time_col          = 'rmvl_ts', 
            df_mp_trsf_pole_nb_col          = 'trsf_pole_nb', 
            verbose                         = verbose
        )
    
        #----------------------------------------------------------------------------------------------------
        # Group by [t_min_col, t_max_col] and use DOVSOutages.get_n_outgs_per_xfmr_in_df_helper
        #-------------------------
        active_PNs_df = Utilities_df.convert_col_types(
            df                  = active_PNs_df, 
            cols_and_types_dict = {
                t_min_col : datetime.datetime, 
                t_min_col : datetime.datetime
            },
            to_numeric_errors   = 'coerce',
            inplace             = True,
        )
        # Use only the date, not the full datetime!
        active_PNs_df[t_min_col] = pd.to_datetime(active_PNs_df[t_min_col].dt.date)
        active_PNs_df[t_max_col] = pd.to_datetime(active_PNs_df[t_max_col].dt.date)
        #-------------------------
        n_outgs_per_PN_df = active_PNs_df.groupby([t_min_col, t_max_col], as_index=False, group_keys=False)[active_PNs_df.columns].apply(
            lambda x: DOVSOutages.get_n_outgs_per_xfmr_in_df_helper(
                active_PNs_df_i     = x, 
                search_times_df     = search_times_df, 
                include_zero_counts = include_zero_counts, 
                t_min_col           = t_min_col, 
                t_max_col           = t_max_col, 
                PN_col              = 'prem_nb', 
            )
        )
    
        #----------------------------------------------------------------------------------------------------
        # Finalize!
        #-----
        fnl_srs = n_outgs_per_PN_df.groupby([trsf_pole_nb_col, t_min_col, t_max_col])['n_outgs'].max().sort_index()
        #-----
        if trsf_pole_nbs_to_ignore is not None:
            fnl_srs = fnl_srs[~fnl_srs.index.get_level_values(trsf_pole_nb_col).isin(trsf_pole_nbs_to_ignore)].copy()
    
        #--------------------------------------------------
        if return_df:
            return fnl_srs, n_outgs_per_PN_df
        return fnl_srs