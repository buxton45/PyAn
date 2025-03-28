#!/usr/bin/env python

r"""
Holds EEMSP class.  See EEMSP.EEMSP for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
from scipy import stats
import datetime
import time
from natsort import natsorted, ns, natsort_keygen
import copy
#--------------------------------------------------
from MeterPremise import MeterPremise
from AMI_SQL import AMI_SQL
from GenAn import GenAn
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from TableInfos import TableInfo
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Utilities_dt
from CustomJSON import CustomWriter
#--------------------------------------------------

class EEMSP(GenAn):
    r"""
    class EEMSP documentation
    """
    def __init__(
        self, 
        df_construct_type=DFConstructType.kRunSqlQuery, 
        contstruct_df_args=None,
        init_df_in_constructor=True, 
        build_sql_function=None, 
        build_sql_function_kwargs=None,
        **kwargs
    ):
        r"""
        NOTE: IT moved the EEMSP data I need over to meter_events.eems_transformer_nameplate (AWS).
              So, this table may be utilized instead of the Oracle ones.
        
        if df_construct_type==DFConstructType.kReadCsv or DFConstructType.kReadCsv:
          contstruct_df_args needs to have at least 'file_path'
        if df_construct_type==DFConstructType.kRunSqlQuery:
          contstruct_df_args needs at least 'conn_db' AND 'trsf_pole_nbs'        
        """
        #--------------------------------------------------
        # First, set self.build_sql_function and self.build_sql_function_kwargs
        # and call base class's __init__ method
        #---------------
        self.build_sql_function = (build_sql_function if build_sql_function is not None 
                                   else EEMSP.build_sql_eemsp_oracle)
        self.use_aws = True
        if self.build_sql_function == EEMSP.build_sql_eemsp_oracle or self.build_sql_function == EEMSP.build_sql_eemsp_active_oracle:
            self.use_aws = False
        #---------------
        self.build_sql_function_kwargs = (build_sql_function_kwargs if build_sql_function_kwargs is not None 
                                          else {})
        assert('trsf_pole_nbs' in self.build_sql_function_kwargs)
        self.build_sql_function_kwargs['field_to_split'] = self.build_sql_function_kwargs.get('field_to_split', 'trsf_pole_nbs')
        self.build_sql_function_kwargs['batch_size'] = self.build_sql_function_kwargs.get('batch_size', 1000)
        self.build_sql_function_kwargs['verbose'] = self.build_sql_function_kwargs.get('verbose', True)
        self.build_sql_function_kwargs['n_update'] = self.build_sql_function_kwargs.get('n_update', 10)
        #--------------------------------------------------
        super().__init__(
            df_construct_type=df_construct_type, 
            contstruct_df_args=contstruct_df_args, 
            init_df_in_constructor=init_df_in_constructor, 
            build_sql_function=self.build_sql_function, 
            build_sql_function_kwargs=self.build_sql_function_kwargs, 
            **kwargs
        )
        

    #****************************************************************************************************
    def get_conn_db(self):
        if self.use_aws:
            return Utilities.get_athena_prod_aws_connection()
        else:
            return Utilities.get_eemsp_oracle_connection()
    
    def get_default_cols_and_types_to_convert_dict(self):
        cols_to_convert = ['EQSEQ_ID', 'TRANSACTION_ID', 'LOCATION_ID', 
                           'EQTYPE_ID', 'XCOORD_NB', 'YCOORD_NB', 'ROWNUMBER']
        cols_and_types_to_convert_dict = {x:np.float64 for x in cols_to_convert}
        cols_and_types_to_convert_dict['INSTALL_DT'] = datetime.datetime
        return cols_and_types_to_convert_dict

    def get_full_default_sort_by(self):
        full_default_sort_by = None
        return full_default_sort_by
    
        
    #****************************************************************************************************    
    @staticmethod
    def build_sql_eemsp_active_oracle(trsf_pole_nbs):
        sql_eemsp = (
        """
        WITH Transformer_Attributes AS 
        (
            SELECT   
                pvt.EQTYPE_ID, 
                pvt.KIND_CD, 
                pvt.COOLANT, 
                pvt.INFO, 
                pvt.KVA_SIZE, 
                pvt.PHASE_CNT, 
                pvt.PRIM_VOLTAGE,
                pvt.PROTECTION, 
                pvt.PRU_NUMBER, 
                pvt.SEC_VOLTAGE, 
                pvt.SPECIAL_CHAR, 
                pvt.TAPS, 
                pvt.XFTYPE
            FROM 
            (
                SELECT *
                FROM EEMS.EQUIP_TYPE_CHAR x
                WHERE KIND_CD = 'T'
            ) a
            PIVOT 
            (
                MAX(a.CHAR_VL) 
                FOR CHAR_NM IN 
                (
                    'COOLANT' AS coolant, 
                    'INFO' AS info, 
                    'KVA_SIZE' AS kva_size, 
                    'PHASE_CNT' AS phase_cnt,
                    'PRIM_VOLTAGE' AS prim_voltage, 
                    'PROTECTION' AS protection, 
                    'PRU_NUMBER' AS pru_number, 
                    'SEC_VOLTAGE' AS sec_voltage, 
                    'SPECIAL_CHAR' AS special_char, 
                    'TAPS' AS taps, 
                    'TYPE' AS xftype
                )
            ) pvt
        ), 
        Equip_Hist AS 
        (
            SELECT   
                a.EQSEQ_ID, 
                a.EQUIP_ID, 
                a.SERIAL_NB, 
                b.KIND_CD, 
                b.KIND_NM, 
                c.TRANSACTION_ID, 
                f.STATUS_DS AS LATEST_STATUS,
                e.MFGR_NM, 
                c.OPEN_DT AS INSTALL_DT,
                CASE 
                    WHEN c.REMOVE_DT IS NOT NULL AND CLOSE_DT IS NULL THEN c.REMOVE_DT 
                    ELSE c.CLOSE_DT 
                END AS REMOVAL_DT,
                d.REASON_DS AS LAST_TRANS_DESC, 
                c.LOCATION_NB, 
                c.LOCATION_ID, 
                a.EQTYPE_ID, 
                h.EQTYPE_DS, 
                a.COMPANY_NB,
                COALESCE(j.COMPANY_NM, k.COMPANY_NM) AS COMPANY_NM, 
                k.OPER_COMPANY_NB,
                COALESCE(j.SUBAREA_NB, k.SUBAREA_NB) AS SUBAREA_NB,
                COALESCE(j.SUBAREA_NM, k.SUBAREA_NM) AS SUBAREA_NM,
                COALESCE(j.REGION_NB, k.REGION_NB) AS REGION_NB,
                COALESCE(j.REGION_NM, k.REGION_NM) AS REGION_NM,
                COALESCE(j.DISTRICT_NB, k.DISTRICT_NB) AS DISTRICT_NB,
                COALESCE(j.DISTRICT_NM, k.DISTRICT_NM) AS DISTRICT_NM,
                COALESCE(j.STATE_ABBR, k.STATE_ABBR) AS STATE,
                c.STATION_NB, 
                k.STATION_NM, 
                c.CIRCUIT_NB, 
                k.CIRCUIT_NM, 
                k.GIS_CIRCUIT_NB, 
                i.XCOORD_NB, 
                i.YCOORD_NB,
                g.COOLANT, 
                g.INFO, 
                g.KVA_SIZE, 
                g.PHASE_CNT, 
                g.PRIM_VOLTAGE, 
                g.PROTECTION, 
                g.PRU_NUMBER, 
                g.SEC_VOLTAGE, 
                g.SPECIAL_CHAR, 
                g.TAPS, 
                g.XFTYPE,
                ROW_NUMBER() OVER (PARTITION BY c.LOCATION_NB ORDER BY c.OPEN_DT DESC) AS RowNumber
            FROM EEMS.EQUIP a
            INNER JOIN EEMS.KIND b ON b.KIND_CD = a.KIND_CD
            INNER JOIN EEMS.LOC_HIST c ON c.EQSEQ_ID = a.EQSEQ_ID AND c.KIND_CD = a.KIND_CD
            LEFT OUTER JOIN EEMS.REASON d ON d.REASON_CD = c.REASON_CD
            INNER JOIN EEMS.MFGR e ON e.MFGR_CD = a.MFGR_CD
            INNER JOIN EEMS.EQUIP_STATUS f ON f.STATUS_CD = a.STATUS_CD
            LEFT OUTER JOIN Transformer_Attributes g ON g.EQTYPE_ID = a.EQTYPE_ID AND g.KIND_CD = a.KIND_CD
            LEFT OUTER JOIN EEMS.EQUIP_TYPE h ON h.EQTYPE_ID = a.EQTYPE_ID AND h.KIND_CD = a.KIND_CD
            LEFT OUTER JOIN EEMS.LOCATION_MAST i ON i.LOCATION_ID = c.LOCATION_ID
            LEFT OUTER JOIN EEMS.EEMS_SUBAREAS j ON j.SUBAREA_NB = i.SUBAREA_NB
            LEFT OUTER JOIN EEMS.LOCAL_COM_STATION_CIRCUIT_VIEW k ON k.STATION_NB = c.STATION_NB AND k.CIRCUIT_NB = c.CIRCUIT_NB AND k.COMPANY_NB = a.COMPANY_NB
            WHERE UPPER(b.KIND_NM) = 'TRANSFORMER'
        )
        SELECT *
        FROM Equip_Hist a
        WHERE a.RowNumber = 1
        AND REMOVAL_DT IS NULL --Current
        AND LOCATION_NB IN ({})
        """
        ).format(Utilities_sql.join_list_w_quotes(trsf_pole_nbs))
        return sql_eemsp
        
        
    @staticmethod
    def build_sql_eemsp_oracle(trsf_pole_nbs):
        sql_eemsp = (
        """
        WITH Transformer_Attributes AS 
        (
            SELECT   
                pvt.EQTYPE_ID, 
                pvt.KIND_CD, 
                pvt.COOLANT, 
                pvt.INFO, 
                pvt.KVA_SIZE, 
                pvt.PHASE_CNT, 
                pvt.PRIM_VOLTAGE,
                pvt.PROTECTION, 
                pvt.PRU_NUMBER, 
                pvt.SEC_VOLTAGE, 
                pvt.SPECIAL_CHAR, 
                pvt.TAPS, 
                pvt.XFTYPE
            FROM 
            (
                SELECT *
                FROM EEMS.EQUIP_TYPE_CHAR x
                WHERE KIND_CD = 'T'
            ) a
            PIVOT 
            (
                MAX(a.CHAR_VL) 
                FOR CHAR_NM IN 
                (
                    'COOLANT' AS coolant, 
                    'INFO' AS info, 
                    'KVA_SIZE' AS kva_size, 
                    'PHASE_CNT' AS phase_cnt,
                    'PRIM_VOLTAGE' AS prim_voltage, 
                    'PROTECTION' AS protection, 
                    'PRU_NUMBER' AS pru_number, 
                    'SEC_VOLTAGE' AS sec_voltage, 
                    'SPECIAL_CHAR' AS special_char, 
                    'TAPS' AS taps, 
                    'TYPE' AS xftype
                )
            ) pvt
        ), 
        Equip_Hist AS 
        (
            SELECT   
                a.EQSEQ_ID, 
                a.EQUIP_ID, 
                a.SERIAL_NB, 
                b.KIND_CD, 
                b.KIND_NM, 
                c.TRANSACTION_ID, 
                f.STATUS_DS AS LATEST_STATUS,
                e.MFGR_NM, 
                c.OPEN_DT AS INSTALL_DT,
                CASE 
                    WHEN c.REMOVE_DT IS NOT NULL AND CLOSE_DT IS NULL THEN c.REMOVE_DT 
                    ELSE c.CLOSE_DT 
                END AS REMOVAL_DT,
                d.REASON_DS AS LAST_TRANS_DESC, 
                c.LOCATION_NB, 
                c.LOCATION_ID, 
                a.EQTYPE_ID, 
                h.EQTYPE_DS, 
                a.COMPANY_NB,
                COALESCE(j.COMPANY_NM, k.COMPANY_NM) AS COMPANY_NM, 
                k.OPER_COMPANY_NB,
                COALESCE(j.SUBAREA_NB, k.SUBAREA_NB) AS SUBAREA_NB,
                COALESCE(j.SUBAREA_NM, k.SUBAREA_NM) AS SUBAREA_NM,
                COALESCE(j.REGION_NB, k.REGION_NB) AS REGION_NB,
                COALESCE(j.REGION_NM, k.REGION_NM) AS REGION_NM,
                COALESCE(j.DISTRICT_NB, k.DISTRICT_NB) AS DISTRICT_NB,
                COALESCE(j.DISTRICT_NM, k.DISTRICT_NM) AS DISTRICT_NM,
                COALESCE(j.STATE_ABBR, k.STATE_ABBR) AS STATE,
                c.STATION_NB, 
                k.STATION_NM, 
                c.CIRCUIT_NB, 
                k.CIRCUIT_NM, 
                k.GIS_CIRCUIT_NB, 
                i.XCOORD_NB, 
                i.YCOORD_NB,
                g.COOLANT, 
                g.INFO, 
                g.KVA_SIZE, 
                g.PHASE_CNT, 
                g.PRIM_VOLTAGE, 
                g.PROTECTION, 
                g.PRU_NUMBER, 
                g.SEC_VOLTAGE, 
                g.SPECIAL_CHAR, 
                g.TAPS, 
                g.XFTYPE
            FROM EEMS.EQUIP a
            INNER JOIN EEMS.KIND b ON b.KIND_CD = a.KIND_CD
            INNER JOIN EEMS.LOC_HIST c ON c.EQSEQ_ID = a.EQSEQ_ID AND c.KIND_CD = a.KIND_CD
            LEFT OUTER JOIN EEMS.REASON d ON d.REASON_CD = c.REASON_CD
            INNER JOIN EEMS.MFGR e ON e.MFGR_CD = a.MFGR_CD
            INNER JOIN EEMS.EQUIP_STATUS f ON f.STATUS_CD = a.STATUS_CD
            LEFT OUTER JOIN Transformer_Attributes g ON g.EQTYPE_ID = a.EQTYPE_ID AND g.KIND_CD = a.KIND_CD
            LEFT OUTER JOIN EEMS.EQUIP_TYPE h ON h.EQTYPE_ID = a.EQTYPE_ID AND h.KIND_CD = a.KIND_CD
            LEFT OUTER JOIN EEMS.LOCATION_MAST i ON i.LOCATION_ID = c.LOCATION_ID
            LEFT OUTER JOIN EEMS.EEMS_SUBAREAS j ON j.SUBAREA_NB = i.SUBAREA_NB
            LEFT OUTER JOIN EEMS.LOCAL_COM_STATION_CIRCUIT_VIEW k ON k.STATION_NB = c.STATION_NB AND k.CIRCUIT_NB = c.CIRCUIT_NB AND k.COMPANY_NB = a.COMPANY_NB
            WHERE UPPER(b.KIND_NM) = 'TRANSFORMER'
        )
        SELECT *
        FROM Equip_Hist a
        WHERE LOCATION_NB IN ({})
        """
        ).format(Utilities_sql.join_list_w_quotes(trsf_pole_nbs))
        return sql_eemsp
        

    @staticmethod
    def build_sql_eemsp(
        cols_of_interest=None, 
        **kwargs
    ):
        r"""
        IT moved the EEMSP data I need over to meter_events.eems_transformer_nameplate (AWS).
        This method utilizes that table (instead of the EEMS table in Oracle).
    
        Acceptable kwargs:
    
            - serial_number(s)
            - serial_number_col
                - default: serial_nb
    
            - trsf_pole_nb(s)
            - trsf_pole_nb_col
                - default: location_nb
    
            - opco(s) (or, aep_opco(s) will work also)
            - opco_col
                - default: aep_opco
    
            *** NOTE: datetime_range here (and in MeterPremise) is conceptually different than in AMINonVee/AMIEndEvents/etc.
                      Here, the datetime_range essentially, this enforces that a transformer was installed before datetime_range[0] 
                        and removed after datetime_range[1].
                      Since transformers which are currently still active have removal_dt values which are empty, the logic is slightly more involved.
                      See MeterPremise.add_inst_rmvl_ts_where_statement for more information.
            - datetime_range
            - inst_ts_col
                - default: install_dt
            - rmvl_ts_col
                - default: removal_dt
            - datetime_pattern
                - default: None (because already properly formatted.  I suppose, if one really wanted to, one could use
                                 datetime_pattern = r"([0-9]{4})-([0-9]{2})-([0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
                                 and datetime_replace = r"$1-$2-$3 $4")
            - datetime_replace
                - default: None (because already properly formatted.  See datetime_pattern description above)
    
            - state(s)
            - state_col
                - default: state
                
            ***** For more info on datetime/inst_ts/rmvl_ts, see MeterPremise.add_inst_rmvl_ts_where_statement *****
        """
        #-------------------------
        acceptable_kwargs = [
            'serial_number', 'serial_numbers', 'serial_number_col', 
            'trsf_pole_nb', 'trsf_pole_nbs', 'trsf_pole_nb_col', 
            'opco', 'opcos', 'aep_opco', 'aep_opcos', 'opco_col', 
            'datetime_range', 'inst_ts_col', 'rmvl_ts_col', 'datetime_pattern', 'datetime_replace', 
            'state', 'states', 'state_col'
        ]
        assert(set(kwargs.keys()).difference(set(acceptable_kwargs))==set())
        #-------------------------
        # First, make any necessary adjustments to kwargs
        kwargs['schema_name']       = kwargs.get('schema_name', 'meter_events')
        kwargs['table_name']        = kwargs.get('table_name', 'eems_transformer_nameplate')
        kwargs['from_table_alias']  = kwargs.get('from_table_alias', 'EEMSP')
        #-----
        kwargs['serial_number_col'] = kwargs.get('serial_number_col', 'serial_nb')
        kwargs['trsf_pole_nb_col']  = kwargs.get('trsf_pole_nb_col', 'location_nb')
        kwargs['inst_ts_col']       = kwargs.get('inst_ts_col', 'install_dt')
        kwargs['rmvl_ts_col']       = kwargs.get('rmvl_ts_col', 'removal_dt')
        kwargs['datetime_pattern']  = kwargs.get('datetime_pattern', None)
        kwargs['datetime_replace']  = kwargs.get('datetime_replace', None)
        kwargs['state_col']         = kwargs.get('state_col', 'state')
        #--------------------------------------------------
        # field_descs_dict needed if any where elements are to be combined
        field_descs_dict = dict()
        #-------------------------
        # If any where statements are to be combined, they must be combined here instead of within AMI_SQL.build_sql_ami,
        #   as the latter doesn't have all the necessary info (i.e., doesn't have all possible field descriptions)
        # Thus, wheres_to_combine must be popped off of kwargs before being used in AMI_SQL.build_sql_ami
        wheres_to_combine = kwargs.pop('wheres_to_combine', None)
        #-------------------------
        # datetime_range will be handled by MeterPremise.add_inst_rmvl_ts_where_statement, so we want to remove it
        #   from kwargs so AMI_SQL.build_sql_ami doesn't use it
        datetime_range   = kwargs.pop('datetime_range', None)
        #-----
        inst_ts_col      = kwargs.get('inst_ts_col', 'inst_ts')
        rmvl_ts_col      = kwargs.get('rmvl_ts_col', 'rmvl_ts')
        datetime_pattern = kwargs.get('datetime_pattern', r"([0-9]{2})/([0-9]{2})/([0-9]{4}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        datetime_replace = kwargs.get('datetime_replace', r"$3-$1-$2 $4")
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = TableInfos.EEMSP_ME_TI.std_columns_of_interest
        sql = AMI_SQL.build_sql_ami(
            cols_of_interest = cols_of_interest, 
            **kwargs
        )
        #--------------------------------------------------
        # Other WHERE statements not handled by AMI_SQL.build_sql_ami
        if datetime_range is not None:
            sql.sql_where = MeterPremise.add_inst_rmvl_ts_where_statement(
                sql_where        = sql.sql_where, 
                datetime_range   = datetime_range, 
                from_table_alias = kwargs['from_table_alias'], 
                inst_ts_col      = inst_ts_col, 
                rmvl_ts_col      = rmvl_ts_col, 
                datetime_pattern = datetime_pattern, 
                datetime_replace = datetime_replace
            )
        #--------------------------------------------------
        return sql
        
        
    @staticmethod
    def reduce1_eemsp_for_outg_trsf_i(
        time_infos_df_i, 
        df_eemsp, 
        outg_rec_nb_idfr, 
        trsf_pole_nb_idfr, 
        dt_min_col, 
        dt_max_col, 
        eemsp_location_nb_col = 'LOCATION_NB', 
        eemsp_install_dt_col  = 'INSTALL_DT', 
        eemsp_removal_dt_col  = 'REMOVAL_DT', 
        return_eemsp_outg_rec_nb_col = 'OUTG_REC_NB_TO_MERGE'
    ):
        r"""
        For a particular (outg_rec_nb, trsf_pole_nb) group, find the corresponding EEMSP entries.
        Typically, EEMSP will have multiple entries for each transformer, so the point of this function
          is to find the correct entries at the time of the outage.
        After this procedure, all entries will be time-appropriate, but there may still be multiple entries
          for a particular (outg_rec_nb, trsf_pole_nb) group.
        Multiple entries will occur, e.g., if there are multiple transformers on the particular trsf_pole_nb
          (at this point, I do not know how to determine to which transformer a meter is connected, only the
          trsf_pole_nb).
        For reductions down to one entry, see EEMSP.reduce2_eemsp_for_outg_trsf_i

        NOTE: For this case, the outg_rec_nb_idfr/trsf_pole_nb_idfr should be equal to a column.
              One cannot use, e.g., 'index_0'
              This function is not really meant to be used on its own, and the correct formatting of time_infos_df_i
                is taken care of by the calling function
        """
        #-------------------------
        assert(isinstance(time_infos_df_i, pd.Series))
        #-------------------------
        outg_rec_nb  = time_infos_df_i[outg_rec_nb_idfr]
        trsf_pole_nb = time_infos_df_i[trsf_pole_nb_idfr]
        dt_min       = time_infos_df_i[dt_min_col]
        if dt_max_col is not None:
            dt_max = time_infos_df_i[dt_max_col]
        else:
            dt_max = dt_min
        #-------------------------
        df_eemsp_i = df_eemsp[df_eemsp[eemsp_location_nb_col]==trsf_pole_nb]
        #-----
        df_eemsp_i = df_eemsp_i[
            (df_eemsp_i[eemsp_install_dt_col] <= dt_min) & 
            (df_eemsp_i[eemsp_removal_dt_col].fillna(pd.Timestamp.max) > dt_max)
        ]
        df_eemsp_i = df_eemsp_i.drop_duplicates()
        #-------------------------    
        df_eemsp_i[return_eemsp_outg_rec_nb_col] = outg_rec_nb
        #-------------------------
        return df_eemsp_i


    @staticmethod
    def reduce1_eemsp_for_outg_trsf(
        time_infos_df, 
        df_eemsp, 
        outg_rec_nb_idfr  = 'index_0', 
        trsf_pole_nb_idfr = 'index_1', 

        dt_min_col = ('dummy_lvl_0', 'DT_OFF_TS_FULL'), 
        dt_max_col = None, 

        eemsp_location_nb_col = 'LOCATION_NB', 
        eemsp_install_dt_col  = 'INSTALL_DT', 
        eemsp_removal_dt_col  = 'REMOVAL_DT', 
        return_eemsp_outg_rec_nb_col = 'OUTG_REC_NB_TO_MERGE', 
        verbose=True,
        n_update=1000
    ):
        r"""
        For each (outg_rec_nb, trsf_pole_nb) group, find the corresponding EEMSP entries.
        Typically, EEMSP will have multiple entries for each transformer/pole, so the point of this function
          is to find the correct entries at the time of the outage.
        After this procedure, all entries will be time-appropriate, but there may still be multiple entries
          for a particular (outg_rec_nb, trsf_pole_nb) group.
        Multiple entries will occur, e.g., if there are multiple transformers on the particular trsf_pole_nb
          (at this point, I do not know how to determine to which transformer a meter is connected, only the
          trsf_pole_nb).
        For reductions down to one entry, see EEMSP.reduce2_eemsp_for_outg_trsf_i

        If the outg_rec_nbs/trsf_pole_nbs are stored in the indices, and the indices are named,
          one can simply supply the corresponding names.
        Otherwise, one can always supply, e.g., index_0 and index_1

        If only a single time is to be used (e.g., the outage starting time), set only dt_min_col
        If two times (e.g., outage starting and stopping) set both dt_min_col and dt_max_col
        """
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        # 1. First, determine where exactly outg_rec_nb and trsf_pole_nb are located in time_infos_df.
        # 2. If in indices, call reset_index()
        # 3. Set grp_by_cols, and make sure each combination of outg_rec_nb, trsf_pole_nb 
        #    has a single datetime entry.
        #-------------------------
        outg_rec_nb_idfr_loc  = Utilities_df.get_idfr_loc(time_infos_df, outg_rec_nb_idfr)
        trsf_pole_nb_idfr_loc = Utilities_df.get_idfr_loc(time_infos_df, trsf_pole_nb_idfr)
        #--------------------------------------------------
        # If outg_rec_nbs in index (i.e., outg_rec_nb_idfr_loc[1]==True), grab the index name and set 
        #   outg_rec_nb_idfr equal to it.
        #   If no index name, give it a random name
        # If outg_rec_nbs in a column (i.e., outg_rec_nb_idfr_loc[1]==False), simply set outg_rec_nb_idfr
        #   equal to it (i.e., set equal to outg_rec_nb_idfr_loc[0])
        if outg_rec_nb_idfr_loc[1]:
            if time_infos_df.index.names[outg_rec_nb_idfr_loc[0]]:
                outg_rec_nb_idfr = time_infos_df.index.names[outg_rec_nb_idfr_loc[0]]
            else:
                tmp_outg_rec_nb_name = 'outg_rec_nb_'+Utilities.generate_random_string(str_len=4)
                time_infos_df.index  = time_infos_df.index.set_names(tmp_outg_rec_nb_name, level=outg_rec_nb_idfr_loc[0])
                outg_rec_nb_idfr     = tmp_outg_rec_nb_name
        else:
            outg_rec_nb_idfr = outg_rec_nb_idfr_loc[0]
        #-------------------------
        # Do the same thing for trsf_pole_nbs
        if trsf_pole_nb_idfr_loc[1]:
            if time_infos_df.index.names[trsf_pole_nb_idfr_loc[0]]:
                trsf_pole_nb_idfr = time_infos_df.index.names[trsf_pole_nb_idfr_loc[0]]
            else:
                tmp_trsf_pole_nb_name = 'trsf_pole_nb_'+Utilities.generate_random_string(str_len=4)
                time_infos_df.index   = time_infos_df.index.set_names(tmp_trsf_pole_nb_name, level=trsf_pole_nb_idfr_loc[0])
                trsf_pole_nb_idfr     = tmp_trsf_pole_nb_name
        else:
            trsf_pole_nb_idfr = trsf_pole_nb_idfr_loc[0]
        #--------------------------------------------------
        grp_by_cols = [outg_rec_nb_idfr, trsf_pole_nb_idfr, dt_min_col]
        if dt_max_col is None:
            dt_max_col = dt_min_col
        else:
            grp_by_cols.append(dt_max_col)
        #-------------------------
        # Even if outg_rec_nbs and/or trsf_pole_nbs are in the index, the above methods ensure the indices will be 
        #   named in such a case.
        # Therefore, one can use .groupby with the index names or columns
        # NOTE: Each combination of outg_rec_nb, trsf_pole_nb should only have a single datetime entry!
        #       The assertion below enforces that
        n_groups = time_infos_df.groupby(grp_by_cols).ngroups
        assert(
            n_groups == time_infos_df.groupby([outg_rec_nb_idfr, trsf_pole_nb_idfr]).ngroups
        )    

        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        # Form groups_df, which will have columns housing the outg_rec_nb, trsf_pole_nb, and the time info (one
        #   additional column if dt_max_col is None, two additional columns if not None)
        #-------------------------
        #--------------------------------------------------
        # Using the groupby method is much easier, but also much slower!
        #   group by method: groups = list(time_infos_df.groupby(grp_by_cols).groups.keys())
        # So, instead, I will use .value_counts method after calling reset_index()
        # This is annoying because if outg_rec_nb or trsf_pole_nb in index, I must call reset_index
        # Furthermore, if time_infos_df has MultiIndex columns, the identifiers will change from strings to tuples
        # In such a case, pandas still is able to grab time_infos_df[[outg_rec_nb_idfr, trsf_pole_nb_idfr]]
        #   i.e., is smart enough to find, e.g., ['outg_rec_nb', 'trsf_pole_nb'], even though the columns
        #   are technically [('outg_rec_nb', ''), ('trsf_pole_nb', '')]
        # But, pandas is NOT smart enough to grab time_infos_df[grp_by_cols] 
        #   i.e., too dumb to find, e.g., ['outg_rec_nb', 'trsf_pole_nb', ('dummy_lvl_0', 'DT_OFF_TS_FULL')] when
        #   the columns are technically  [('outg_rec_nb', ''), ('trsf_pole_nb', ''), ('dummy_lvl_0', 'DT_OFF_TS_FULL')]
        #-------------------------
        if outg_rec_nb_idfr_loc[1] and time_infos_df.columns.nlevels>1:
            outg_rec_nb_idfr = tuple([outg_rec_nb_idfr] + ['' for _ in range(time_infos_df.columns.nlevels-1)])
            grp_by_cols[0]   = outg_rec_nb_idfr
        #-----
        if trsf_pole_nb_idfr_loc[1] and time_infos_df.columns.nlevels>1:
            trsf_pole_nb_idfr = tuple([trsf_pole_nb_idfr] + ['' for _ in range(time_infos_df.columns.nlevels-1)])
            grp_by_cols[1]    = trsf_pole_nb_idfr
        #-------------------------
        groups_df = time_infos_df.reset_index()[grp_by_cols].value_counts()
        groups_df = groups_df.reset_index().drop(columns='count')
        assert(groups_df.shape[0]==n_groups)

        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        # Chop down size of df_eemsp first
        # Get rid of any with install date after last time or removal date before first time
        df_eemsp=df_eemsp[~(
            (df_eemsp[eemsp_install_dt_col] > time_infos_df[dt_max_col].max()) |
            (df_eemsp[eemsp_removal_dt_col] < time_infos_df[dt_min_col].min())
        )]
        df_eemsp = df_eemsp.drop_duplicates()
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        # Iterate through groups_df, and grab the appropriate entries from df_eemsp
        if verbose:
            print('\nEEMSP.reduce1_eemsp_for_outg_trsf')
        eemsp_dfs = []
        for i,(idx_i, df_i) in enumerate(groups_df.iterrows()):
            if verbose and i%n_update==0:
                print(f"\t{i} of {groups_df.shape[0]}")
            eemsp_df_i = EEMSP.reduce1_eemsp_for_outg_trsf_i(
                time_infos_df_i              = df_i, 
                df_eemsp                     = df_eemsp, 
                outg_rec_nb_idfr             = outg_rec_nb_idfr, 
                trsf_pole_nb_idfr            = trsf_pole_nb_idfr, 
                dt_min_col                   = dt_min_col, 
                dt_max_col                   = dt_max_col, 
                eemsp_location_nb_col        = eemsp_location_nb_col, 
                eemsp_install_dt_col         = eemsp_install_dt_col, 
                eemsp_removal_dt_col         = eemsp_removal_dt_col, 
                return_eemsp_outg_rec_nb_col = return_eemsp_outg_rec_nb_col
            )
            eemsp_dfs.append(eemsp_df_i)
        return_df_eemsp = pd.concat(eemsp_dfs)
        return return_df_eemsp
        
        
    @staticmethod
    def remove_ambiguous_from_df_eemsp(
        df_eemsp, 
        grp_by_cols=['OUTG_REC_NB_TO_MERGE', 'LOCATION_NB']
    ):
        r"""
        Any group with multiple entries will be removed
        """
        #-------------------------
        # Make sure grp_by_cols are present in df_eemsp
        # NOTE: If simple string given for columns whereas df_eemsp has MultiIndex columns, the
        #       following should remdy such a situation by changing string to appropriate tuple
        grp_by_cols = [Utilities_df.find_single_col_in_multiindex_cols(df=df_eemsp, col=x) for x in grp_by_cols]
        #-------------------------
        no_ambiguity = (df_eemsp[grp_by_cols].value_counts()==1)
        no_ambiguity = no_ambiguity[no_ambiguity==True].index
        #-------------------------
        df_eemsp_singles = df_eemsp.set_index(grp_by_cols).loc[no_ambiguity].reset_index()
        #-------------------------
        return df_eemsp_singles
        
        
    @staticmethod
    def reduce2_eemsp_for_outg_trsf_i(
        df_eemsp_i, 
        agg_dict, 
        return_idx_order, 
        mult_strategy='agg', 
        include_n_eemsp=True
    ):
        r"""
        Helper function for EEMSP.reduce2_eemsp_for_outg_trsf
        In reality, only mult_strategy used here should be 'agg' ('first' is handled in 
          EEMSP.reduce2_eemsp_for_outg_trsf with the nth() function)
        """
        #-------------------------
        # NOTE: If mult_strategy=='exclude' in EEMSP.reduce2_eemsp_for_outg_trsf, the workflow
        #         should not utilize this function!
        assert(mult_strategy in ['agg', 'first'])
        #-------------------------
        if df_eemsp_i.shape[0]==1:
            srs_i = df_eemsp_i.iloc[0]
        else:
            if mult_strategy=='agg':
                srs_i = df_eemsp_i.agg(agg_dict)
            else:
                assert(mult_strategy=='first')
                srs_i = df_eemsp_i.iloc[0]
        #-------------------------
        srs_i=srs_i.reindex(index=return_idx_order)
        #-------------------------
        if include_n_eemsp:
            srs_i['n_eemsp'] = df_eemsp_i.shape[0]
        #-------------------------
        return srs_i
        
        
    @staticmethod
    def reduce2_eemsp_for_outg_trsf(
        df_eemsp, 
        mult_strategy='agg', 
        include_n_eemsp=True, 
        grp_by_cols=['OUTG_REC_NB_TO_MERGE', 'LOCATION_NB'], 
        numeric_cols = ['KVA_SIZE'], 
        dt_cols = ['INSTALL_DT', 'REMOVAL_DT'], 
        ignore_cols = ['SERIAL_NB'], 
        cat_cols_as_strings=True
    ):
        r"""
        To be run after EEMSP.reduce1_eemsp_for_outg_trsf_i!!!!!
        The intent of this function is to reduce df_eemsp down to one row per group (typically location_nb_col
          and/or outg_rec_nb_col)

        mult_strategy:
            Dictates how (outg_rec_nb, trsf_pole_nb) groups with multiple EEMSP entries will be handled.
            Can be 'agg', 'first', or 'exclude'
            'agg':
                For (outg_rec_nb, trsf_pole_nb) groups with multiple entries, aggregate
            'first':
                For (outg_rec_nb, trsf_pole_nb) groups with multiple entries, take the first
            'exclude':
                Exclude (outg_rec_nb, trsf_pole_nb) groups with multiple entries

        cat_cols_as_strings:
            Categorical columns can either be aggregated as (sorted) lists of unique values (cat_cols_as_strings==False),
              or as strings (cat_cols_as_strings==True)
            If cat_cols_as_strings==True, a given string is simply the (sorted) lists of unique values joined by commas


        """
        #-------------------------
        assert(mult_strategy in ['agg', 'first', 'exclude'])
        #-------------------------
        # Copy below probably not necessary
        df_eemsp = df_eemsp.copy()
        #-------------------------
        # Make sure grp_by_cols and columns in numeric_cols/dt_cols are present in df_eemsp
        # NOTE: If simple string given for columns whereas df_eemsp has MultiIndex columns, the
        #       following should remdy such a situation by changing string to appropriate tuple
        grp_by_cols  = [Utilities_df.find_single_col_in_multiindex_cols(df=df_eemsp, col=x) for x in grp_by_cols]
        numeric_cols = [Utilities_df.find_single_col_in_multiindex_cols(df=df_eemsp, col=x) for x in numeric_cols]
        dt_cols      = [Utilities_df.find_single_col_in_multiindex_cols(df=df_eemsp, col=x) for x in dt_cols]
        #-------------------------
        # Make sure numeric_cols are numeric types and dt_cols are datetime
        for i,col in enumerate(numeric_cols):
            if not is_numeric_dtype(df_eemsp[numeric_cols[i]].dtype):
                df_eemsp = Utilities_df.convert_col_type(
                    df=df_eemsp, 
                    column=numeric_cols[i], 
                    to_type=float
                )
        #-----
        for i,col in enumerate(dt_cols):
            if not is_datetime64_dtype(df_eemsp[dt_cols[i]].dtype):
                df_eemsp = Utilities_df.convert_col_type(
                    df=df_eemsp, 
                    column=dt_cols[i], 
                    to_type=datetime.datetime
                )
        #-------------------------
        if ignore_cols:
            # ignore_cols may or may not actually be contained in df_eemsp, hence the need for try/except
            ignore_cols_OG = copy.deepcopy(ignore_cols)
            ignore_cols = []
            for i,col in enumerate(ignore_cols_OG):
                try:
                    col = Utilities_df.find_single_col_in_multiindex_cols(df=df_eemsp, col=col)
                    ignore_cols.append(col)
                except:
                    pass
        else:
            ignore_cols=[]
        #-------------------------
        # All columns not in numeric_cols + dt_cols will be considered categorical
        cat_cols = [
            x for x in df_eemsp.columns 
            if x not in numeric_cols+dt_cols+ignore_cols+grp_by_cols
        ]
        #--------------------------------------------------
        # Build the aggregation dictionary and aggregate
        numeric_dict = {k:'mean' for k in numeric_cols}
        #-----
        dt_dict      = {k:'mean' for k in dt_cols}
        #-----
        # For whatever reason, using list(set(x)) in this case (using dictionary
        #   with column keys and function values) returns a list of all the unique
        #   characters in the column.
        # Instead, I must use list(set(x.tolist())) 
        cat_dict     = {k:lambda x: natsorted(list(set(x.tolist()))) for k in cat_cols}
        if cat_cols_as_strings:
            # NOTE: Need .astype(str) below because Python apparently only likes joining lists of strings!
            cat_dict = {k:lambda x: ', '.join(natsorted(list(set(x.astype(str).tolist())))) for k in cat_cols}
        #-------------------------
        agg_dict = (numeric_dict | dt_dict | cat_dict)
        #-------------------------
        for col_i in grp_by_cols:
            agg_dict[col_i] = lambda x: x.tolist()[0]
        #-------------------------
        return_idx_order = (
            grp_by_cols + 
            natsorted(numeric_dict.keys()) + 
            natsorted(dt_dict.keys()) + 
            natsorted(cat_dict.keys())
        )
        assert(len(set(agg_dict.keys()).symmetric_difference(set(return_idx_order)))==0)
        #-------------------------
        if mult_strategy=='exclude':
            if ignore_cols:
                df_eemsp = df_eemsp.drop(columns=ignore_cols)
            return_df = EEMSP.remove_ambiguous_from_df_eemsp(
                df_eemsp=df_eemsp, 
                grp_by_cols=grp_by_cols
            )
        else:
            if mult_strategy=='first':
                if ignore_cols:
                    df_eemsp = df_eemsp.drop(columns=ignore_cols)
                if dt_cols:
                    df_eemsp = df_eemsp.sort_values(by=dt_cols, ascending=False)
                # NOTE: I want to use nth() below, NOT first()
                #       Apparently, first doesn't grab the first row of each group, 
                #         it returns the first non-null entry of each column (so, if null values
                #         exist, this will be a mixture of 2 or more rows)
                return_df = df_eemsp.groupby(
                    grp_by_cols, 
                    as_index=False, 
                    group_keys=False, 
                    dropna=False
                ).nth(0)
                #----------
                if include_n_eemsp:
                    n_eemsp_df = df_eemsp.groupby(
                        grp_by_cols, 
                        as_index=False, 
                        group_keys=False, 
                        dropna=False
                    ).size()
                    assert('size' in n_eemsp_df.columns)
                    n_eemsp_df = n_eemsp_df.rename(columns={'size':'n_eemsp'})
                    #-----
                    tmp_shape = return_df.shape
                    return_df = pd.merge(
                        return_df, 
                        n_eemsp_df, 
                        left_on= grp_by_cols, 
                        right_on=grp_by_cols, 
                        how='inner'
                    )
                    #-----
                    assert(return_df.shape[0]==tmp_shape[0])
                    assert(return_df.shape[1]==tmp_shape[1]+1)

            else:
                assert(mult_strategy=='agg')
                return_df = df_eemsp.groupby(
                    grp_by_cols, 
                    as_index=False, 
                    group_keys=False, 
                    dropna=False
                ).apply(
                    lambda x: EEMSP.reduce2_eemsp_for_outg_trsf_i(
                        df_eemsp_i=x, 
                        agg_dict=agg_dict, 
                        return_idx_order=return_idx_order, 
                        mult_strategy=mult_strategy, 
                        include_n_eemsp=include_n_eemsp
                    )
                )
        #-------------------------
        return return_df
        
        
    @staticmethod
    def merge_rcpx_with_eemsp(
        df_rcpx, 
        df_eemsp, 
        merge_on_rcpx, 
        merge_on_eems,
        set_index=True, 
        drop_eemsp_merge_cols=True
    ):
        r"""
        merge_on_rcpx/merge_on_eems:
            These should both be lists of equal length.
            Pairs to be merged should have the same index between the two lists

        set_index:
            If True, the index of return_df will be set to merge_on_rcpx
        """
        #-------------------------
        # I will likely be manipulating df_rcpx and df_eemsp.
        # I probably don't want to change the DFs outside of this function, so copy
        df_rcpx = df_rcpx.copy()
        df_eemsp = df_eemsp.copy()

        #-------------------------
        # merge_on_rcpx and merge_on_eems must be lists of the same length
        assert(isinstance(merge_on_rcpx, list) and isinstance(merge_on_eems, list))
        assert(len(merge_on_rcpx)==len(merge_on_eems))

        #-------------------------
        # Call reset_index if needed and identify full/true merge_on values
        df_rcpx, merge_on_rcpx = Utilities_df.prep_df_for_merge(
            df=df_rcpx, 
            merge_on=merge_on_rcpx, 
            inplace=True
        )
        #-----
        df_eemsp, merge_on_eems = Utilities_df.prep_df_for_merge(
            df=df_eemsp, 
            merge_on=merge_on_eems, 
            inplace=True
        )
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        # In order to merge, df_rcpx and df_eemsp must have the same number of levels of columns
        if df_rcpx.columns.nlevels != df_eemsp.columns.nlevels:
            if df_rcpx.columns.nlevels > df_eemsp.columns.nlevels:
                n_levels_to_add = df_rcpx.columns.nlevels - df_eemsp.columns.nlevels
                #-----
                df_eemsp = Utilities_df.prepend_levels_to_MultiIndex(
                    df=df_eemsp, 
                    n_levels_to_add=n_levels_to_add, 
                    dummy_col_levels_prefix='EEMSP_'
                )
                #-----
                # Get new MultiIndex versions of merge_on_eems
                merge_on_eems = [Utilities_df.find_single_col_in_multiindex_cols(df=df_eemsp, col=x) for x in merge_on_eems]
            elif df_rcpx.columns.nlevels < df_eemsp.columns.nlevels:
                n_levels_to_add = df_eemsp.columns.nlevels - df_rcpx.columns.nlevels
                #-----
                df_rcpx = Utilities_df.prepend_levels_to_MultiIndex(
                    df=df_rcpx, 
                    n_levels_to_add=n_levels_to_add, 
                    dummy_col_levels_prefix='RCPX_'
                )
                #-----
                # Get new MultiIndex versions of merge_on_rcpx
                merge_on_rcpx = [Utilities_df.find_single_col_in_multiindex_cols(df=df_rcpx, col=x) for x in merge_on_rcpx]        
            else:
                assert(0)
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        # Merge
        assert(
            df_rcpx[merge_on_rcpx].dtypes.tolist() ==
            df_eemsp[merge_on_eems].dtypes.tolist()
        )
        #-----
        return_df = pd.merge(
            df_rcpx, 
            df_eemsp, 
            left_on=merge_on_rcpx, 
            right_on=merge_on_eems, 
            how='inner'
        )
        #-------------------------
        if drop_eemsp_merge_cols:
            # NOTE: If rcpx and eemsp columns are the same, after merge only one will remain, and 
            #       do not want to drop in such a case!
            cols_to_drop=[]
            for i_col in range(len(merge_on_rcpx)):
                if merge_on_eems[i_col] != merge_on_rcpx[i_col]:
                    cols_to_drop.append(merge_on_eems[i_col])
            #-----
            if cols_to_drop:
                return_df = return_df.drop(columns=cols_to_drop)
        #-------------------------
        if set_index:
            return_df = return_df.set_index(merge_on_rcpx)
            #-----
            # Resolve any ugly resulting index names, e.g., [('outg_rec_nb', ''), ('trsf_pole_nb', '')]
            #   i.e., if name_i is a tuple where 0th element is not empty but all other are, then change
            #         name to 0th element
            fnl_idx_names = []
            for idx_name_i in return_df.index.names:
                if(
                    isinstance(idx_name_i, tuple) and 
                    idx_name_i[0] and
                    not any([True if idx_name_i[i] else False for i in range(1, len(idx_name_i))])
                ):
                    fnl_idx_names.append(idx_name_i[0])
                else:
                    fnl_idx_names.append(idx_name_i)
            return_df.index.names = fnl_idx_names
        #-------------------------
        return return_df
    

    @staticmethod
    def build_eemsp_df(
        build_sql_function        = None, 
        build_sql_function_kwargs = None
    ):
        r"""
        NOTE: By default, this method uses the AWS table meter_events.eems_transformer_nameplate instead of the Oracle tables
        To build eemsp_df for modeler, use EEMSP.build_eemsp_df_for_modeler!!!!!
        """
        #-------------------------
        if build_sql_function is None:
            build_sql_function = EEMSP.build_sql_eemsp
        #-------------------------
        eemsp = EEMSP(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = None,
            init_df_in_constructor    = True, 
            build_sql_function        = build_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs,
        )
        return eemsp.df
        
        
    @staticmethod
    def build_eemsp_df_for_xfmrs(
        trsf_pole_nbs , 
        batch_size    = 1000, 
        verbose       = True, 
        n_update      = 10, 
        addtnl_kwargs = None
    ):
        r"""
        NOTE: This method uses the AWS table meter_events.eems_transformer_nameplate instead of the Oracle tables
        """
        #-------------------------
        build_sql_function_kwargs = {}
        if addtnl_kwargs is not None:
            assert(isinstance(addtnl_kwargs, dict))
            build_sql_function_kwargs = addtnl_kwargs
        build_sql_function_kwargs['trsf_pole_nbs']  = trsf_pole_nbs
        build_sql_function_kwargs['field_to_split'] = 'trsf_pole_nbs'
        build_sql_function_kwargs['batch_size']     = batch_size
        build_sql_function_kwargs['verbose']        = verbose
        build_sql_function_kwargs['n_update']       = n_update
        #-------------------------
        if verbose:
            print('\nEEMSP.build_eemsp_df_for_xfmrs\nBuilding EEMSP object')
        eemsp = EEMSP(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = None,
            init_df_in_constructor    = True, 
            build_sql_function        = EEMSP.build_sql_eemsp, 
            build_sql_function_kwargs = build_sql_function_kwargs,
        )
        return eemsp.df
        
        
    @staticmethod
    def build_eemsp_df_for_xfmrs_w_time_info(
        trsf_pole_nbs, 
        time_infos_df, 
        mult_strategy               = 'agg', 
        batch_size                  = 1000, 
        verbose                     = True, 
        n_update                    = 10, 
        addtnl_kwargs               = None, 
        time_infos_df_info_dict     = None, 
        eemsp_df_info_dict          = None, 
        min_pct_found_in_time_infos = 0.75, 
        save_dfs                    = False, 
        save_dir                    = None, 
        return_3_dfs                = False, 
        return_eemsp_df_info_dict   = False
    ):
        r"""
        Build EEMSP data for time periods specified in time_infos_df.
        This allows each transformer to utilize a different time window (as defined in time_infos_df), and this function is generally
          used when building the model (for which each transformer/outage generally has a unique time window).
        
        NOTE: This method uses the AWS table meter_events.eems_transformer_nameplate instead of the Oracle tables
        
        return_eemsp_df_info_dict:
            If True, dictionary object always returned
            If False:
                If return_3_dfs==True, (df_eemsp, df_eemsp_reduce1, df_eemsp_reduce2) is returned
                If return_3_dfs==False, df_eemsp_reduce2 is returned
        """
        #----------------------------------------------------------------------------------------------------
        if save_dfs:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            assert(os.path.isdir(save_dir))
        #-----
        assert(time_infos_df.shape[0]==time_infos_df.reset_index().drop_duplicates().shape[0])
        #----------------------------------------------------------------------------------------------------
        #--------------------------------------------------
        dflt_time_infos_df_info_dict = dict(
            outg_rec_nb_idfr  = 'index_0', 
            trsf_pole_nb_idfr = 'index_1', 
            dt_min_col        = 't_min', 
            dt_max_col        = 't_max'
        )
        #-------------------------
        if time_infos_df_info_dict is None:
            time_infos_df_info_dict = dflt_time_infos_df_info_dict
        else:
            assert(isinstance(time_infos_df_info_dict, dict))
            time_infos_df_info_dict = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = time_infos_df_info_dict, 
                default_values_dict = dflt_time_infos_df_info_dict, 
                extend_any_lists    = False, 
                inplace             = False
            )
        #-------------------------
        outg_rec_nb_idfr  = time_infos_df_info_dict['outg_rec_nb_idfr']
        trsf_pole_nb_idfr = time_infos_df_info_dict['trsf_pole_nb_idfr']
        dt_min_col        = time_infos_df_info_dict['dt_min_col']
        dt_max_col        = time_infos_df_info_dict['dt_max_col']
        #--------------------------------------------------
        # NOTE: removal_dt needed for algorithm, but typically is not desired in output, so in eemsp_cols_to_drop by default
        dflt_eemsp_df_info_dict = dict(
            eemsp_location_nb_col        = 'location_nb', 
            eemsp_install_dt_col         = 'install_dt', 
            eemsp_removal_dt_col         = 'removal_dt', 
            eemsp_numeric_cols           = ['kva_size'], 
            eemsp_ignore_cols            = ['serial_nb'], 
            rec_nb_to_merge_col          = 'rec_nb_to_merge', 
            eemsp_cols_of_interest       = TableInfos.EEMSP_ME_TI.std_columns_of_interest, 
            eemsp_cols_to_drop           = ['latest_status', 'removal_dt', 'serial_nb'] # See note above
        )
        #-------------------------
        if eemsp_df_info_dict is None:
            eemsp_df_info_dict = dflt_eemsp_df_info_dict
        else:
            assert(isinstance(eemsp_df_info_dict, dict))
            eemsp_df_info_dict = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = eemsp_df_info_dict, 
                default_values_dict = dflt_eemsp_df_info_dict, 
                extend_any_lists    = False, 
                inplace             = False
            ) 
        #-------------------------
        eemsp_location_nb_col        = eemsp_df_info_dict['eemsp_location_nb_col']
        eemsp_install_dt_col         = eemsp_df_info_dict['eemsp_install_dt_col']
        eemsp_removal_dt_col         = eemsp_df_info_dict['eemsp_removal_dt_col']
        eemsp_numeric_cols           = eemsp_df_info_dict['eemsp_numeric_cols']
        eemsp_ignore_cols            = eemsp_df_info_dict['eemsp_ignore_cols']
        rec_nb_to_merge_col          = eemsp_df_info_dict['rec_nb_to_merge_col']
        eemsp_cols_of_interest       = eemsp_df_info_dict['eemsp_cols_of_interest']
        eemsp_cols_to_drop           = eemsp_df_info_dict['eemsp_cols_to_drop']
        
        #----------------------------------------------------------------------------------------------------
        assert(0 < min_pct_found_in_time_infos <= 1)
        n_xfmrs_matched = len(set(trsf_pole_nbs).intersection(Utilities_df.get_vals_from_df(df=time_infos_df, idfr=trsf_pole_nb_idfr, unique=True)))
        pct_found = len(set(trsf_pole_nbs))/n_xfmrs_matched
        if verbose:
            print(f'# trsf_pole_nbs supplied = {len(set(trsf_pole_nbs))}')
            print(f'% found in time_infos_df = {100*pct_found}')
        assert(pct_found > min_pct_found_in_time_infos)
        #----------------------------------------------------------------------------------------------------
        if addtnl_kwargs is None:
            addtnl_kwargs = {}
        if 'cols_of_interest' not in addtnl_kwargs.keys():
            addtnl_kwargs['cols_of_interest'] = eemsp_cols_of_interest
        #-----
        df_eemsp = EEMSP.build_eemsp_df_for_xfmrs(
            trsf_pole_nbs = trsf_pole_nbs, 
            batch_size    = batch_size, 
            verbose       = verbose, 
            n_update      = n_update, 
            addtnl_kwargs = addtnl_kwargs
        )
        #-----
        nec_cols_eemps = [eemsp_location_nb_col, eemsp_install_dt_col, eemsp_removal_dt_col]
        assert(set(nec_cols_eemps).difference(set(df_eemsp.columns.tolist()))==set())
        #-----
        df_eemsp = Utilities_df.convert_col_types(
            df                  = df_eemsp, 
            cols_and_types_dict = {
                eemsp_install_dt_col : datetime.datetime, 
                eemsp_removal_dt_col : datetime.datetime
            }, 
            to_numeric_errors  = 'coerce', 
            inplace            = True
        )
        if save_dfs:
            df_eemsp.to_pickle(os.path.join(save_dir, 'df_eemsp_OG.pkl'))
        
        #----------------------------------------------------------------------------------------------------
        # Run reduce1, keeping only entries from df_eemsp active at the correct time periods
        #--------------------------------------------------
        df_eemsp_reduce1 = EEMSP.reduce1_eemsp_for_outg_trsf(
            time_infos_df                = time_infos_df, 
            df_eemsp                     = df_eemsp, 
            outg_rec_nb_idfr             = outg_rec_nb_idfr, 
            trsf_pole_nb_idfr            = trsf_pole_nb_idfr, 
            dt_min_col                   = dt_min_col, 
            dt_max_col                   = dt_max_col, 
            eemsp_location_nb_col        = eemsp_location_nb_col, 
            eemsp_install_dt_col         = eemsp_install_dt_col, 
            eemsp_removal_dt_col         = eemsp_removal_dt_col, 
            return_eemsp_outg_rec_nb_col = rec_nb_to_merge_col, 
            verbose                      = verbose, 
            n_update                     = 1000
        )
        if save_dfs:
            df_eemsp_reduce1.to_pickle(os.path.join(save_dir, 'df_eemsp_reduce1.pkl'))
            
        #----------------------------------------------------------------------------------------------------
        # Run reduce2, keeping only one entry per outg_rec_nb, location_nb (trsf_pole_nb) group
        #--------------------------------------------------
        eemsp_numeric_cols = [x for x in eemsp_numeric_cols if x in df_eemsp_reduce1.columns.tolist()]
        eemsp_ignore_cols  = [x for x in eemsp_ignore_cols if x in df_eemsp_reduce1.columns.tolist()]
        #-----
        df_eemsp_reduce2 = EEMSP.reduce2_eemsp_for_outg_trsf(
            df_eemsp            = df_eemsp_reduce1, 
            mult_strategy       = mult_strategy, 
            include_n_eemsp     = True, 
            grp_by_cols         = [rec_nb_to_merge_col, eemsp_location_nb_col], 
            numeric_cols        = eemsp_numeric_cols, 
            dt_cols             = [eemsp_install_dt_col, eemsp_removal_dt_col], 
            ignore_cols         = eemsp_ignore_cols, 
            cat_cols_as_strings = True
        )
        #-------------------------
        # No matter of the mult_strategy used, at this point df_eemsp_reduce2 should only have a single
        #   entry for each outg_rec_nb, location_nb pair
        assert(all(df_eemsp_reduce2[[rec_nb_to_merge_col, eemsp_location_nb_col]].value_counts()==1))
        #----------------------------------------------------------------------------------------------------
        # Clean up df_eemsp_reduce2 and merge with merged_df_outg, merged_df_otbl, and merged_df_prbl
        #--------------------------------------------------
        # Can't simply take df_eemsp_reduce2[cols_of_interest_eemsp] because we need also the new column
        #   rec_nb_to_merge_col (and any others which may be added in the future)
        cols_to_drop = [x for x in eemsp_cols_to_drop if x in df_eemsp_reduce2.columns]
        if len(cols_to_drop)>0:
            df_eemsp_reduce2 = df_eemsp_reduce2.drop(columns=cols_to_drop)
        #-------------------------
        assert(df_eemsp_reduce2.shape[0]==df_eemsp_reduce2.groupby([rec_nb_to_merge_col, eemsp_location_nb_col]).ngroups)
        #----------------------------------------------------------------------------------------------------
        if save_dfs:
            df_eemsp_reduce2.to_pickle(os.path.join(save_dir, f'df_eemsp_reduce2_{mult_strategy}.pkl'))
            #-----
            CustomWriter.output_dict_to_json(
                os.path.join(save_dir, 'eemsp_df_info_dict.json'), 
                eemsp_df_info_dict
            )
        #----------------------------------------------------------------------------------------------------
        if return_eemsp_df_info_dict:
            return_dict = dict(
                df_eemsp_reduce2   = df_eemsp_reduce2, 
                eemsp_df_info_dict = eemsp_df_info_dict
            )
            if return_3_dfs:
                return_dict = return_dict | dict(
                    df_eemsp_OG      = df_eemsp, 
                    df_eemsp_reduce1 = df_eemsp_reduce1
                )
            return return_dict
        else:
            if return_3_dfs:
                return df_eemsp, df_eemsp_reduce1, df_eemsp_reduce2
            else:
                return df_eemsp_reduce2