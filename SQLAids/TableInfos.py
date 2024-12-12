#!/usr/bin/env python

import sys, os
import Utilities_sql
import numpy as np
import copy

class TableInfo:
    def __init__(self, 
                 schema_name, 
                 table_name, 
                 conn_fcn=None, 
                 columns_full=None, 
                 std_columns_of_interest=None, 
                 columns_and_data_types=None):
        r"""
        NOTE: Decided to use conn_fcn instead of conn (e.g., conn_fcn = Utilities_sql.get_athena_prod_aws_connection
              instead of conn = Utilities_sql.get_athena_prod_aws_connection().
              The main reason being that the old method will fail if there is any sort of connection error with the server.
                So, it would essentially be impossible to run without internet I suppose.
              The second reason is that it takes time to connect to the server, and most times I use a TableInfo instance
                I don't use the connection (e.g., I grab the standard cols of interest, etc.), so this time to connect is unnecessary.
        """
        self.__schema_name = schema_name
        self.__table_name = table_name
        self.__conn_fcn = conn_fcn
        self.__columns_full = columns_full
        self.__std_columns_of_interest = std_columns_of_interest
        self.__columns_and_data_types = columns_and_data_types
        
        if self.__std_columns_of_interest is None:
            self.__std_columns_of_interest = self.__columns_full
            
    @property
    def schema_name(self):
        return copy.deepcopy(self.__schema_name)
        
    @property
    def table_name(self):
        return copy.deepcopy(self.__table_name)
        
    @property
    def conn_fcn(self):
        return copy.deepcopy(self.__conn_fcn)
        
    @property
    def columns_full(self):
        return copy.deepcopy(self.__columns_full)

    @property
    def std_columns_of_interest(self):
        return copy.deepcopy(self.__std_columns_of_interest)
        
    @property
    def columns_and_data_types(self):
        return copy.deepcopy(self.__columns_and_data_types)
        
#****************************************************************************************************
# AMINonVee
#****************************************************************************************************
AMINonVee_TI = TableInfo(
    schema_name='usage_nonvee', 
    table_name='reading_ivl_nonvee', 
    conn_fcn=Utilities_sql.get_athena_prod_aws_connection, 
    columns_full = [
        'aep_acct_cls_cd',
        'aep_acct_type_cd',
        'aep_billable_ind',
        'aep_channel_id',
        'aep_city',
        'aep_comp_mtr_mltplr',
        'aep_data_quality_cd',
        'aep_data_validation',
        'aep_derived_uom',
        'aep_devicecode',
        'aep_endtime_utc',
        'aep_meter_alias',
        'aep_meter_program',
        'aep_mtr_install_ts',
        'aep_mtr_pnt_nb',
        'aep_mtr_removal_ts',
        'aep_opco',
        'aep_premise_nb',
        'aep_raw_uom',
        'aep_raw_value',
        'aep_sec_per_intrvl',
        'aep_service_point',
        'aep_srvc_dlvry_id',
        'aep_srvc_qlty_idntfr',
        'aep_state',
        'aep_tarf_pnt_nb',
        'aep_timezone_cd',
        'aep_usage_dt',
        'aep_usage_type',
        'aep_zip',
        'authority',
        'endtimeperiod',
        'hdp_insert_dttm',
        'hdp_update_dttm',
        'hdp_update_user',
        'isvirtual_meter',
        'isvirtual_register',
        'name_register',
        'scalarfloat',
        'serialnumber',
        'source',
        'starttimeperiod',
        'timezoneoffset',
        'toutier',
        'toutiername',
        'value'
    ], 
    std_columns_of_interest = [
        'serialnumber', 
        'aep_premise_nb', 
        'starttimeperiod', 
        'endtimeperiod', 
        'aep_endtime_utc', 
        'timezoneoffset', 
        'aep_derived_uom', 
        'aep_srvc_qlty_idntfr', 
        'value', 
        'aep_opco', 
        'aep_usage_dt'        
    ]
)

#****************************************************************************************************
# AMIUsgInst
#****************************************************************************************************
AMIUsgInst_TI = TableInfo(
    schema_name='usage_instantaneous', 
    table_name='inst_msr_consume', 
    conn_fcn=Utilities_sql.get_athena_prod_aws_connection, 
    columns_full = [
        'aep_acct_cls_cd',
        'aep_acct_type_cd',
        'aep_area_cd',
        'aep_city',
        'aep_devicecode',
        'aep_meter_program',
        'aep_mtr_install_ts',
        'aep_mtr_pnt_nb',
        'aep_mtr_removal_ts',
        'aep_opco',
        'aep_premise_nb',
        'aep_read_dt',
        'aep_readtime',
        'aep_readtime_utc',
        'aep_service_point',
        'aep_srvc_dlvry_id',
        'aep_state',
        'aep_sub_area_cd',
        'aep_tarf_pnt_nb',
        'aep_timezone_cd',
        'aep_zip',
        'authority',
        'devc_status',
        'hdp_insert_dttm',
        'hdp_update_dttm',
        'hdp_update_user',
        'latitude',
        'longitude',
        'measurement_type',
        'measurement_value',
        'read_type',
        'serialnumber',
        'timezoneoffset'
    ], 
    std_columns_of_interest=[
        'read_type',
        'serialnumber',
        'aep_premise_nb',
        'timezoneoffset',
        'aep_readtime',
        'aep_readtime_utc',
        'measurement_type',
        'measurement_value',
        'longitude',
        'latitude',
        'aep_opco',
        'aep_read_dt'        
    ]
)

#****************************************************************************************************
# AMIEndEvents
#****************************************************************************************************
AMIEndEvents_TI = TableInfo(
    schema_name='meter_events', 
    table_name='end_device_event', 
    conn_fcn=Utilities_sql.get_athena_prod_aws_connection, 
    columns_full = [
        'aep_area_cd',
        'aep_bill_account_nb',
        'aep_city',
        'aep_devicecode',
        'aep_event_dt',
        'aep_mtr_pnt_nb',
        'aep_opco',
        'aep_premise_nb',
        'aep_service_point',
        'aep_state',
        'aep_sub_area_cd',
        'aep_tarf_pnt_nb',
        'aep_timezone_cd',
        'aep_zip',
        'domain',
        'enddeviceeventtypeid',
        'event_type',
        'eventoraction',
        'hdp_insert_dttm',
        'hdp_update_dttm',
        'hdp_update_user',
        'issuer_id',
        'issuertracking_id',
        'latitude',
        'longitude',
        'manufacturer_id',
        'reason',
        'serialnumber',
        'sub_domain',
        'user_id',
        'valuesinterval'
    ], 
    std_columns_of_interest=[
        'issuertracking_id',    
        'serialnumber',
        'enddeviceeventtypeid',
        'valuesinterval',
        'aep_premise_nb',
        'reason',
        'event_type',
        #'longitude',
        #'latitude',
        'aep_opco',
        'aep_event_dt'           
    ]
)

#****************************************************************************************************
# MeterPremise
#****************************************************************************************************
MeterPremise_TI = TableInfo(
    schema_name='default', 
    table_name='meter_premise', 
    conn_fcn=Utilities_sql.get_athena_prod_aws_connection, 
    columns_full = [
        'addl_srv_data_ad',
        'ami_ftprnt_cd',
        'annual_kwh',
        'annual_max_dmnd',
        'area_cd',
        'area_cd_desc',
        'bill_cnst',
        'bill_fl',
        'building_type',
        'building_type_desc',
        'circuit_nb',
        'circuit_nm',
        'cmsg_mtr_mult_cd',
        'co_cd_ownr',
        'co_cd_ownr_desc',
        'co_gen_fl',
        'comm_cd',
        'comm_desc',
        'county_cd',
        'county_nm',
        'cumu_cd',
        'cumu_cd_desc',
        'curr_acct_cls_cd',
        'curr_bill_acct_id',
        'curr_bill_acct_nb',
        'curr_cust_nm',
        'curr_enrgy_efncy_pgm_cd',
        'curr_enrgy_efncy_pgm_dt',
        'curr_enrgy_efncy_prtcpnt_fl',
        'curr_rvn_cls_cd',
        'curr_tarf_cd',
        'curr_tarf_cd_desc',
        'cycl_nb',
        'delv_pt_cd',
        'devc_cd',
        'devc_cd_desc',
        'devc_stat_cd',
        'devc_stat_cd_desc',
        'dial_cnst',
        'directions',
        'district_nb',
        'district_nm',
        'dstrbd_gen_capcty_nb',
        'dstrbd_gen_ind_cd',
        'dstrbd_gen_instl_dt',
        'dstrbd_gen_typ_cd',
        'dvsn_cd',
        'dvsn_cd_desc',
        'emrgncy_gen_fl',
        'enrgy_dvrn_cd',
        'enrgy_dvrn_dt',
        'enrgy_dvrn_fl',
        'esi_id',
        'first_in_srvc_dt',
        'frst_turn_on_dt',
        'heat_src_fuel_typ_cd',
        'heat_typ_cd',
        'heat_typ_cd_desc',
        'hsng_ctgy_cd',
        'hsng_ctgy_cd_desc',
        'inst_tod_cd',
        'inst_ts',
        'interval_data',
        'intrvl_data_use_cd',
        'intrvl_data_use_cd_desc',
        'jrsd_cd',
        'jrsd_cd_descr',
        'last_fld_test_date',
        'last_turn_off_dt',
        'latitude',
        'latitude_nb',
        'load_area_cd',
        'longitude',
        'longitude_nb',
        'mfr_cd',
        'mfr_cd_desc',
        'mfr_devc_ser_nbr',
        'mtr_kind_cds',
        'mtr_pnt_nb',
        'mtr_point_location',
        'mtr_stat_cd',
        'mtr_stat_cd_desc',
        'naics_cd',
        'oms_area',
        'owns_home_cd',
        'owns_home_cd_desc',
        'pgm_id_nm',
        'phys_inst_dt',
        'power_pool_cd',
        'prem_nb',
        'prem_stat_cd',
        'prem_stat_cd_desc',
        'premise_id',
        'profile_id',
        'rmvl_ts',
        'route_nb_ad',
        'rurl_rte_type_cd',
        'rurl_rte_type_cd_desc',
        'rvn_cls_cd_desc',
        'seasonal_fl',
        'ser_half_ind_ad',
        'serv_city_ad',
        'serv_hous_nbr_ad',
        'serv_prdr_ad',
        'serv_ptdr_ad',
        'serv_st_dsgt_ad',
        'serv_st_name_ad',
        'serv_unit_dsgt_ad',
        'serv_unit_nbr_ad',
        'serv_zip_ad',
        'squr_feet_mkt_qy',
        'srvc_addr_1_nm',
        'srvc_addr_2_nm',
        'srvc_addr_3_nm',
        'srvc_addr_4_nm',
        'srvc_entn_cd',
        'srvc_pnt_nm',
        'srvc_pole_nb',
        'st_cd_ad',
        'state_cd',
        'state_cd_desc',
        'station_nb',
        'station_nm',
        'sub_area_cd',
        'sub_area_cd_desc',
        'tarf_pnt_nb',
        'tarf_pt_stat_cd',
        'tarf_pt_stat_cd_desc',
        'tax_dstc_cd',
        'tax_dstc_cd_desc',
        'technology_desc',
        'technology_tx',
        'tod_mtr_fl',
        'trsf_pole_nb',
        'type_of_srvc_cd',
        'type_of_srvc_cd_desc',
        'type_srvc_cd',
        'type_srvc_cd_desc',
        'vintage_year',
        'wthr_stn_cd',
        'xfmr_name',
        'xfmr_nb',
        'xfmr_type',
        'year_strc_cmpl_dt'
    ], 
    # std_columns_of_interest = [
        # 'mfr_devc_ser_nbr',
        # 'longitude',
        # 'latitude',
        # 'state_cd', 
        # 'prem_nb',
        # 'srvc_pole_nb',
        # 'trsf_pole_nb',
        # 'latitude_nb',
        # 'longitude_nb',
        # 'xfmr_nb'
    # ], 
    std_columns_of_interest = [
        'mfr_devc_ser_nbr',
        'longitude',
        'latitude',
        'state_cd', 
        'prem_nb',
        'srvc_pole_nb',
        'trsf_pole_nb',
        'latitude_nb',
        'longitude_nb',
        'circuit_nb', 
        'circuit_nm', 
        'station_nb', 
        'station_nm', 
        'xfmr_nb', 
        'annual_kwh',
        'annual_max_dmnd', 
        'mtr_stat_cd',
        'mtr_stat_cd_desc', 
        'devc_stat_cd', 
        'devc_stat_cd_desc'
    ]
)



MeterPremiseHist_TI = TableInfo(
    schema_name='default', 
    table_name='meter_premise_hist', 
    conn_fcn=Utilities_sql.get_athena_prod_aws_connection, 
    columns_full = [
        'bill_cnst',
         'circuit_nb',
         'circuit_nm',
         'co_cd_desc',
         'co_cd_ownr',
         'co_gen_fl',
         'comm_cd',
         'comm_desc',
         'cur_acct_id',
         'cur_bill_acct_nb',
         'cur_tarf_cd',
         'cur_tarf_cd_desc',
         'curr_acct_cls_cd',
         'curr_rvn_cls_cd',
         'devc_cd',
         'devc_cd_desc',
         'devc_stat_cd',
         'devc_stat_cd_desc',
         'dial_cnst',
         'distributed_generation_flag',
         'first_in_srvc_dt',
         'inst_kind_cd',
         'inst_ts',
         'interval_data',
         'intrvl_data_use_cd',
         'intrvl_data_use_cd_desc',
         'last_fld_test_date',
         'latitude',
         'longitude',
         'mfr_cd',
         'mfr_cd_desc',
         'mfr_devc_ser_nbr',
         'mtr_kind_cd',
         'mtr_kind_cd_desc',
         'mtr_pnt_nb',
         'mtr_point_location',
         'mtr_stat_cd',
         'mtr_stat_cd_desc',
         'mtr_use_cd',
         'mtr_use_cd_desc',
         'pgm_id_nm',
         'phys_inst_dt',
         'prem_nb',
         'premise_id',
         'rmvl_ts',
         'rvn_cls_cd_desc',
         'srvc_pnt_nm',
         'state_cd',
         'station_nb',
         'station_nm',
         'tarf_pnt_nb',
         'tarf_pt_stat_cd',
         'tarf_pt_stat_cd_desc',
         'tarf_type_servc_cd',
         'tarf_type_servc_cd_desc',
         'technology_desc',
         'technology_tx',
         'time_use_mtr_fl',
         'type_of_srvc_cd',
         'type_of_srvc_cd_desc',
         'vintage_year'
    ], 

    std_columns_of_interest = [
        'circuit_nb',
        'circuit_nm',
        'devc_stat_cd',
        'devc_stat_cd_desc',
        'latitude',
        'longitude',
        'mfr_devc_ser_nbr',
        'mtr_stat_cd',
        'mtr_stat_cd_desc',
        'prem_nb',
        'state_cd',
        'station_nb',
        'station_nm'
    ]
)


#****************************************************************************************************
# DOVS_OUTAGE_FACT
#****************************************************************************************************
DOVS_OUTAGE_FACT_TI = TableInfo(
    schema_name='DOVSADM', 
    table_name='DOVS_OUTAGE_FACT', 
    conn_fcn=Utilities_sql.get_utldb01p_oracle_connection, 
    columns_full = [
        'AEP_WEST_LOC_ID',
        'AMED_FL',
        'AMI_CALL_TYPE',
        'ANLST_CHCKED_FL',
        'ANLST_CRRCTD_FL',
        'ATMPT_TO_DSPTCH_DRTN_NB',
        'CAUSE_CD',
        'CAUSE_CD_SHRT_DESCN_TX',
        'CI_NB',
        'CMI_NB',
        'CMM_NB',
        'CM_NB',
        'COMMUNITY_ID',
        'COUNTR_1_NB',
        'COUNTR_2_NB',
        'CREW_CMPLTD_TO_RSTRN_DRTN_NB',
        'CUSTS_RSTRD_BY_STEPS_QY',
        'DEVICE_CD',
        'DISTRICT_NB',
        'DOR_VOLTG_CD',
        'DSPTCH_DRTN_NB',
        'DSPTCH_PRFMC_DRTN_NB',
        'DT_OFF_TS',
        'DT_ON_TS',
        'EDW_LAST_UPDT_DT',
        'EQUIPMENT_CD',
        'FDR_NB',
        'FIELD_ETR_SET_IND_CD',
        'GIS_CRCT_NB',
        'GNRTN_CAUSE_IND',
        'JMED_FL',
        'LAST_EDIT_USER_ID',
        'LOCATION_ID',
        'MAX_CREW_TM_NB',
        'MGMT_CHCKED_FL',
        'MGMT_CRRCTD_FL',
        'MIN_DSPTCH_TM_NB',
        'MJR_CAUSE_CD',
        'MNR_CAUSE_CD',
        'NMRC_DT_NM',
        'NOT_VRFYD_BY_ANY_DISP_FL',
        'NUMBER_OPRTN_NB',
        'OMED_FL',
        'OPCO_NBR',
        'OPERATING_UNIT_ID',
        'OPMED_FL',
        'ORIGINATOR_ID',
        'OUTAGE_CD',
        'OUTAGE_NB',
        'OUTG_REC_NB',
        'PMED_FL',
        'PO_ORDR_FAULT_LOC_NB',
        'REGION_NB',
        'ROW_LAST_EDIT_DT',
        'SCADA_IND',
        'SOURCE_NM',
        'SRVC_CNTR_NB',
        'STATE_ABBR_TX',
        'STEP_DRTN_NB',
        'SUBAREA_NB',
        'SUB_CAUSE_CD',
        'SUB_NB',
        'SURR_KEY_NB',
        'TARGETS',
        'TOWN_NB',
        'TRBL_TKT_NB',
        'VRFYD_BY_DISP_INV_FL',
        'WEATHER_CD',
        'WTHR_DESCN_TX'
    ]
)

#****************************************************************************************************
# DOVS_MASTER_GEO_DIM
#****************************************************************************************************
DOVS_MASTER_GEO_DIM_TI = TableInfo(
    schema_name='DOVSADM', 
    table_name='DOVS_MASTER_GEO_DIM', 
    conn_fcn=Utilities_sql.get_utldb01p_oracle_connection, 
    columns_full = [
        'AREA_ID',
        'AREA_NM',
        'CIRCUIT_NM',
        'DISTRICT_ID',
        'DISTRICT_NAME',
        'EDW_LAST_UPDT_DT',
        'GIS_CIRCUIT_ID',
        'OPCO_ABBRV_NM',
        'OPCO_ID',
        'OPCO_NM',
        'OPRTG_UNT_ABBRV_NM',
        'OPRTG_UNT_ID',
        'OPRTG_UNT_NM',
        'STATE_ID',
        'STATE_NM',
        'SUBSTATION_NB',
        'SUBSTATION_NM',
        'YEAR_NB'
    ]
)

#****************************************************************************************************
# DOVS_OUTAGE_ATTRIBUTES_DIM
#****************************************************************************************************
DOVS_OUTAGE_ATTRIBUTES_DIM_TI = TableInfo(
    schema_name='DOVSADM', 
    table_name='DOVS_OUTAGE_ATTRIBUTES_DIM', 
    conn_fcn=Utilities_sql.get_utldb01p_oracle_connection, 
    columns_full = [
        'ACBS_FL',
        'AEP_WEST_LOC_ID',
        'APPROVER_ID',
        'APRVD_DT_TS',
        'ATMPT_TO_DSPTCH_DT',
        'BRKR_ACTVY_CD',
        'CREW_WRK_RQRD_TX',
        'CURR_INTRPT_STAT_CD',
        'CURR_REC_STAT_CD',
        'CUST_OVRD_CD',
        'DEVICE_CD',
        'DISTRICT_NB',
        'DOR_VOLTG_CD',
        'DT_OFF_TS',
        'DT_ON_TS',
        'EDW_LAST_UPDT_DT',
        'EQUIP_CD',
        'FDR_NB',
        'FIELD_ETR_SET_IND_CD',
        'FRST_UNQ_DISPTR_ID',
        'FRST_UNQ_DISPTR_NM',
        'INTRPTN_EXTNT_TX',
        'INTRPTN_TYP_CD',
        'LAST_EDIT_DT_TS',
        'LOCATION_ID',
        'MAX_CREW_TM',
        'MIN_DSPTCH_TM',
        'MJR_CAUSE_CD',
        'MJR_EVNT_CD',
        'MSTR_OUTG_NB',
        'MSTR_OUTG_TYP_CD',
        'OMS_CLOSE_TIME',
        'OMS_OWNR_USER_NM',
        'OPCO_NB',
        'ORGTR_DT_TS',
        'ORGTR_RPTS_TO_ID',
        'ORGTR_RPTS_TO_NM',
        'ORIGINATOR_ID',
        'ORIGINATOR_NM',
        'OUTAGE_CD',
        'OUTAGE_NB',
        'OUTG_REC_NB',
        'PO_ORDR_CIRCT_REF_TX',
        'PO_ORDR_CREW_AREA_NM',
        'PO_ORDR_DISP_AREA_NM',
        'PO_ORDR_EXTRNL_TYP_TX',
        'PO_ORDR_FAULT_LOC_NB',
        'PO_ORDR_TYP_TX',
        'REGION_NB',
        'RL1_CD',
        'RL2_CD',
        'RL3_CD',
        'RL4_CD',
        'RL5_CD',
        'SOURCE_NM',
        'SRVC_CNTR_NB',
        'SUBAREA_NB',
        'SUB_CAUSE_CD',
        'TOWN_NB',
        'TRBL_ADDR_TX',
        'TRBL_TKT_NB',
        'UPDTD_BY_NM',
        'UPDTD_DT_TS'
    ]
)

#****************************************************************************************************
# DOVS_CLEARING_DEVICE_DIM
#****************************************************************************************************
DOVS_CLEARING_DEVICE_DIM_TI = TableInfo(
    schema_name='DOVSADM', 
    table_name='DOVS_CLEARING_DEVICE_DIM', 
    conn_fcn=Utilities_sql.get_utldb01p_oracle_connection, 
    columns_full = [
        'DEVICE_CD',
        'DVC_DESCN_TX',
        'DVC_TYP_NM',
        'EDW_LAST_UPDT_DT',
        'SHORT_NM',
        'SOURCE_NM'
    ]
)

#****************************************************************************************************
# DOVS_EQUIPMENT_TYPES_DIM
#****************************************************************************************************
DOVS_EQUIPMENT_TYPES_DIM_TI = TableInfo(
    schema_name='DOVSADM', 
    table_name='DOVS_EQUIPMENT_TYPES_DIM', 
    conn_fcn=Utilities_sql.get_utldb01p_oracle_connection, 
    columns_full = [
        'EDW_LAST_UPDT_DT',
        'EQUIPMENT_CD',
        'EQUIP_DESCN_TX',
        'EQUIP_TYP_NM',
        'SHORT_NM',
        'SOURCE_NM'
    ]
)

#****************************************************************************************************
# DOVS_OUTAGE_CAUSE_TYPES_DIM
#****************************************************************************************************
DOVS_OUTAGE_CAUSE_TYPES_DIM_TI = TableInfo(
    schema_name='DOVSADM', 
    table_name='DOVS_OUTAGE_CAUSE_TYPES_DIM', 
    conn_fcn=Utilities_sql.get_utldb01p_oracle_connection, 
    columns_full = [
        'EDW_LAST_UPDT_DT_TS',
        'MJR_CAUSE_CD',
        'MJR_CAUSE_DESCN_TX',
        'MJR_CAUSE_NM',
        'MNR_CAUSE_CD',
        'MNR_CAUSE_DESCN_TX',
        'MNR_CAUSE_NM',
        'SOURCE_NM'
    ]
)

#****************************************************************************************************
# DOVS_PREMISE_DIM
#****************************************************************************************************
DOVS_PREMISE_DIM_TI = TableInfo(
    schema_name='DOVSADM', 
    table_name='DOVS_PREMISE_DIM', 
    conn_fcn=Utilities_sql.get_utldb01p_oracle_connection, 
    columns_full = [
        'ACCOUNT_NB',
        'ADDRESS_TX',
        'ADTNL_INFO_TX',
        'AMI_FL',
        'CALLR_NM',
        'CALL_CMNTS_TX',
        'CALL_MTR_CYC_TX',
        'CALL_MTR_ID',
        'CALL_MTR_RTE_TX',
        'CALL_MTR_SEQ_TX',
        'CALL_SVC_POLE_NM',
        'CALL_TM',
        'CALL_TYP_TX',
        'CALL_UNLCTD_KEY',
        'CALL_XFRM_POLE_NM',
        'CIRCT_NM',
        'CITY_NM',
        'CUSTOMER_NM',
        'CUST_PHN_NB',
        'CUST_TICK_CD',
        'CUST_TYP_CD',
        'EDW_LAST_UPDT_DT',
        'HOUSE_NB',
        'LATITUDE_NB',
        'LONGITUDE_NB',
        'NRML_CIRCT_NB',
        'NRML_PRMSE_PHS_CD',
        'NUM_CALLS_QY',
        'OFF_TM',
        'OMS_AREA_CD',
        'OUTG_REC_NB',
        'POSTAL_CD',
        'PREMISE_NB',
        'PRIORY_CD',
        'PRMSE_RGN_CD',
        'PWR_CNDTN_TX',
        'REST_TM',
        'SOURCE_NM',
        'STATE_CD',
        'STATION_NB',
        'STATION_NM',
        'STREET_NM',
        'TM_OFF_TM',
        'TM_RSTRD_TM',
        'TRBL_TKT_NB',
        'UPDTD_BY_NM',
        'UPDTD_TS_TM'
    ]
)

#****************************************************************************************************
# EEMSP
#****************************************************************************************************
EEMSP_ME_TI = TableInfo(
    schema_name='meter_events', 
    table_name='eems_transformer_nameplate', 
    conn_fcn=Utilities_sql.get_athena_prod_aws_connection, 
    columns_full = [
        'aep_opco', 
        'aws_update_dttm', 
        'circuit_nb', 
        'circuit_nm', 
        'company_nb', 
        'company_nm', 
        'coolant', 
        'district_nb', 
        'district_nm', 
        'eqseq_id', 
        'eqtype_ds', 
        'eqtype_id', 
        'equip_id', 
        'gis_circuit_nb', 
        'info', 
        'install_dt', 
        'kind_cd', 
        'kind_nm', 
        'kva_size', 
        'last_trans_desc', 
        'latest_status', 
        'location_id', 
        'location_nb', 
        'mfgr_nm', 
        'oper_company_nb', 
        'phase_cnt', 
        'prim_voltage', 
        'protection', 
        'pru_number', 
        'region_nb', 
        'region_nm', 
        'removal_dt', 
        'rownumber', 
        'sec_voltage', 
        'serial_nb', 
        'special_char', 
        'state', 
        'station_nb', 
        'station_nm', 
        'subarea_nb', 
        'subarea_nm', 
        'taps', 
        'transaction_id', 
        'xcoord_nb', 
        'xftype', 
        'ycoord_nb'
    ], 

    std_columns_of_interest = [
        'location_nb', 
        'mfgr_nm', 
        'install_dt', 
        'removal_dt', 
        'last_trans_desc', 
        'eqtype_id', 
        'coolant', 
        'info', 
        'kva_size',
        'phase_cnt', 
        'prim_voltage', 
        'protection', 
        'pru_number', 
        'sec_voltage', 
        'special_char', 
        'taps', 
        'xftype', 
        'latest_status', 
        'serial_nb'
    ]
)


