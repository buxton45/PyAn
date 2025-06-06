#!/usr/bin/env python

import sys, os
import datetime
import string
import pandas as pd
import numpy as np
from natsort import natsorted
import time
from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
import win32com.client as win32
import warnings

#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
import Plot_General


#----------------------------------------------------------------------------------------------------
def build_pdpu_matcher_sql(
    date_0                 , 
    date_1                 , 
    opco                   , 
    td_sec_seqntl_pu_or_pd = 5, 
    addtnl_group_by        = ['station_nb', 'circuit_nb'], 
    pd_ids                 = ['3.26.0.47', '3.26.136.47', '3.26.136.66'], 
    pu_ids                 = ['3.26.0.216', '3.26.136.216'], 
    include_final_select   = True, 
):
    r"""
    NOTE: In order to be returned, the meter must have a listed phase_val from cds_ds_db.com_ccp_dgis_extract 
    """
    #--------------------------------------------------
    # BELOW LINES ARE TEMPORARY UNTIL FULL cds_ds_db.com_ccp_dgis_extract table set up!!!!!
    assert(opco in ['pso', 'im'])
    if opco=='pso':
        dgis_query = 'SELECT prem_nb, " phase_val" AS phase_val FROM cds_ds_db.com_ccp_dgis_extract'
    else:
        dgis_query = 'SELECT prem_nb, phase_val FROM cds_ds_db.com_ccp_dgis_extract_im'
    #--------------------------------------------------
    if addtnl_group_by is None:
        group_by = []
    else:
        group_by = addtnl_group_by
    #--------------------------------------------------
    accptbl_group_by = ['station_nb', 'station_nm', 'circuit_nb', 'circuit_nm', 'trsf_pole_nb']
    assert(set(group_by).difference(set(accptbl_group_by))==set())
    #--------------------------------------------------
    datetime_pattern = r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
    group_by_str = Utilities.join_list(
        list_to_join  = group_by, 
        quotes_needed = False, 
        join_str      = ', '
    )
    #--------------------------------------------------
    if len(group_by)>0:
        group_by_comma = ', '
    else:
        group_by_comma = ''
    #--------------------------------------------------
    sql = """
    WITH MP AS (
        SELECT
            MP.mfr_devc_ser_nbr,
            MP.prem_nb, \n{}
            COMP.opco_nm, 
            DGIS.phase_val
    	FROM 
            default.meter_premise MP
    	INNER JOIN (SELECT opco_nb,opco_nm FROM default.company) COMP 
            ON  MP.co_cd_ownr=COMP.opco_nb
    	INNER JOIN ({}) DGIS 
            ON  MP.prem_nb=DGIS.prem_nb""".format(
        Utilities.join_list(
            list_to_join  = [f'            MP.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )+group_by_comma, 
        dgis_query
    )
    if opco is not None:
        sql += """
        WHERE 
            COMP.opco_nm = '{}'""".format(opco)
    sql += "\n    ),"
    #--------------------------------------------------
    sql += """
    filtered_events AS (
        SELECT
            EDE.serialnumber,
            EDE.aep_premise_nb, 
            CAST(regexp_replace(EDE.valuesinterval, '{}', '$1 $2') AS TIMESTAMP) AS valuesinterval,
            CASE 
                WHEN EDE.enddeviceeventtypeid IN ({}) THEN 1 -- Power-up
                WHEN EDE.enddeviceeventtypeid IN ({}) THEN 0 -- Power-down
                ELSE NULL
            END AS power_status, 
            MP.*
        FROM 
            meter_events.end_device_event EDE
        INNER JOIN MP 
            ON  EDE.serialnumber=MP.mfr_devc_ser_nbr 
            AND EDE.aep_premise_nb=MP.prem_nb
        WHERE 
            EDE.aep_event_dt BETWEEN '{}' AND '{}'
            AND   EDE.enddeviceeventtypeid IN ({}) \n{}""".format(
        datetime_pattern, 
        Utilities.join_list(
            list_to_join  = pu_ids, 
            quotes_needed = True, 
            join_str      = ', '
        ), 
        Utilities.join_list(
            list_to_join  = pd_ids, 
            quotes_needed = True, 
            join_str      = ', '
        ), 
        date_0, 
        date_1, 
        Utilities.join_list(
            list_to_join  = pd_ids+pu_ids, 
            quotes_needed = True, 
            join_str      = ', '
        ), 
        Utilities.join_list(
            list_to_join  = [f"            AND NULLIF(TRIM(MP.{x}), '') IS NOT NULL" for x in group_by], 
            quotes_needed = False, 
            join_str      = ' \n'
        )
    )
    if opco is not None:
        sql += "\n            AND   EDE.aep_opco = '{}'".format(opco)
    sql += "\n    ),"
    #--------------------------------------------------
    sql += """
    marked_events_by_SN AS (
        SELECT 
            *,
            CASE
                WHEN ROW_NUMBER() OVER (PARTITION BY {}aep_premise_nb, serialnumber ORDER BY valuesinterval, power_status) = 1 AND power_status = 0 THEN -1
                WHEN ROW_NUMBER() OVER (PARTITION BY {}aep_premise_nb, serialnumber ORDER BY valuesinterval, power_status) = 1 AND power_status = 1 THEN  0
                ELSE power_status - LAG(power_status) OVER (PARTITION BY {}aep_premise_nb, serialnumber ORDER BY valuesinterval, power_status) 
            END AS diff_power_status, 
            DATE_DIFF('second', LAG(valuesinterval) OVER (PARTITION BY {}aep_premise_nb, serialnumber ORDER BY valuesinterval, power_status), valuesinterval) AS diff_time
        FROM 
            filtered_events
    ), """.format(
        group_by_str+group_by_comma, 
        group_by_str+group_by_comma, 
        group_by_str+group_by_comma, 
        group_by_str+group_by_comma
    )
    #--------------------------------------------------
    sql += """
    id_outage_groups_by_SN AS (
        SELECT 
            *,
            SUM(CASE
                    WHEN power_status = 0 THEN -- diff_power_status can only equal 0 or -1 in this case
                        CASE
                            WHEN diff_power_status = -1 THEN 1                   -- always start new group on transition
                            WHEN diff_power_status = 0 AND diff_time > {} THEN 1 -- if previous PD was more than 1 minute prior, define new
                            ELSE 0                                               -- if previous PD was within past minute, keep in this group
                        END
                    WHEN power_status = 1 THEN -- diff_power_status can only equal 0 or +1 in this case
                        CASE
                            WHEN diff_power_status = 1 THEN 0                    -- If PD immediately preceding, always keep together
                            WHEN diff_power_status = 0 AND diff_time > {} THEN 1 -- if previous PU was more than 1 minute prior, define new
                            ELSE 0                                               -- if previous PU was within past minute, keep in this group
                        END
                    ELSE NULL -- Should never happen 
                END) OVER (PARTITION BY {}aep_premise_nb, serialnumber ORDER BY valuesinterval) AS outage_group
        FROM 
            marked_events_by_SN
    ), """.format(
        td_sec_seqntl_pu_or_pd, 
        td_sec_seqntl_pu_or_pd, 
        group_by_str+group_by_comma
    )
    #--------------------------------------------------
    sql += """
    agg_outage_groups_by_SN AS (
        SELECT \n{}
            aep_premise_nb, 
            serialnumber,
            phase_val, 
            CASE 
                WHEN COUNT(DISTINCT power_status) > 1 THEN MIN(CASE WHEN power_status = 0 THEN valuesinterval END) --could also use MAX, depending on preferences
                ELSE
                    CASE
                        WHEN MIN(power_status) = 0 THEN MIN(valuesinterval)
                        ELSE NULL
                    END
            END AS outage_start, 
            CASE 
                WHEN COUNT(DISTINCT power_status) > 1 THEN MAX(CASE WHEN power_status = 1 THEN valuesinterval END) --could also use MIN, depending on preferences
                ELSE
                    CASE
                        WHEN MIN(power_status) = 0 THEN NULL
                        ELSE MAX(valuesinterval) 
                    END
            END AS outage_end, 
            SUM(CASE WHEN power_status = 0 THEN 1 END) AS n_pd, 
            SUM(CASE WHEN power_status = 1 THEN 1 END) AS n_pu
        FROM 
            id_outage_groups_by_SN
        GROUP BY \n{}
            aep_premise_nb, 
            serialnumber, 
            phase_val, 
            outage_group
        ORDER BY \n{}
            aep_premise_nb, 
            serialnumber, 
            phase_val, 
            outage_start
    )""".format(
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )+group_by_comma, 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )+group_by_comma, 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )+group_by_comma, 
    )

    #--------------------------------------------------
    if include_final_select:
        sql += """
        SELECT 
            *, 
            DATE_DIFF('second', outage_start, outage_end) as duration
        FROM 
            agg_outage_groups_by_SN"""
    #--------------------------------------------------
    return sql


#----------------------------------------------------------------------------------------------------
def build_CRED_sql(
    date_0                 , 
    date_1                 , 
    opco                   , 
    min_pct_SN             = None, 
    td_sec_group_daisy     = 5, 
    td_sec_seqntl_pu_or_pd = 5, 
    td_sec_final_daisy     = 5, 
    group_by               = ['station_nb', 'circuit_nb'], 
    pd_ids                 = ['3.26.0.47', '3.26.136.47', '3.26.136.66'], 
    pu_ids                 = ['3.26.0.216', '3.26.136.216']
):
    r"""
    """
    #--------------------------------------------------
    # BELOW LINES ARE TEMPORARY UNTIL FULL cds_ds_db.com_ccp_dgis_extract table set up!!!!!
    assert(opco in ['pso', 'im'])
    if opco=='pso':
        dgis_query = 'SELECT prem_nb, " phase_val" AS phase_val FROM cds_ds_db.com_ccp_dgis_extract'
    else:
        dgis_query = 'SELECT prem_nb, phase_val FROM cds_ds_db.com_ccp_dgis_extract_im'
    #--------------------------------------------------
    accptbl_group_by = ['station_nb', 'station_nm', 'circuit_nb', 'circuit_nm', 'trsf_pole_nb']
    assert(set(group_by).difference(set(accptbl_group_by))==set())
    #--------------------------------------------------
    datetime_pattern = r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
    group_by_str = Utilities.join_list(
        list_to_join  = group_by, 
        quotes_needed = False, 
        join_str      = ', '
    )
    #--------------------------------------------------
    sql = build_pdpu_matcher_sql(
        date_0                 = date_0, 
        date_1                 = date_1, 
        opco                   = opco, 
        td_sec_seqntl_pu_or_pd = td_sec_seqntl_pu_or_pd, 
        addtnl_group_by        = group_by, 
        pd_ids                 = pd_ids, 
        pu_ids                 = pu_ids, 
        include_final_select   = False
    )+','

    #--------------------------------------------------
    sql += """
    id_outage_groups_0 AS (
        SELECT 
            *, 
            DATE_DIFF('second', outage_start, outage_end) as duration, 
            -- Below, if first entry, set difference to 0 so always evaluates to True
            -- Additional ordering by aep_premise_nb, serialnumber to make sure first entry always consistent/reproducible
            CASE
                WHEN ROW_NUMBER() OVER (PARTITION BY {} ORDER BY outage_start, aep_premise_nb, serialnumber) = 1 THEN 0
                ELSE 
                    CASE
                        WHEN DATE_DIFF('second', LAG(outage_start) OVER (PARTITION BY {} ORDER BY outage_start, aep_premise_nb, serialnumber), outage_start) <= {} THEN 0
                        ELSE 1
                    END
            END AS grp_incrementor
        FROM 
            agg_outage_groups_by_SN
        WHERE 
            outage_start IS NOT NULL
    ), """.format(
        group_by_str, 
        group_by_str, 
        td_sec_group_daisy
    )
    #--------------------------------------------------
    sql += """
    id_outage_groups_1 AS (
        SELECT 
            *, 
            CASE
                WHEN duration < 8 THEN 0
                WHEN duration >= 8 and duration < 300 THEN 1
                WHEN duration >= 300 THEN 2
                ELSE 3
            END as duration_grp, 
            SUM(grp_incrementor) OVER (PARTITION BY {} ORDER BY outage_start, aep_premise_nb, serialnumber) as group_by_grp_i
        FROM 
            id_outage_groups_0
    ), """.format(
        group_by_str
    )
    #--------------------------------------------------
    sql += """
    agg_outage_groups_0_total AS (
        SELECT \n{}, 
            group_by_grp_i,
            COUNT(serialnumber)            AS total_SN_events,
            COUNT(DISTINCT serialnumber)   AS unique_SNs,
            COUNT(aep_premise_nb)          AS total_PN_events,
            COUNT(DISTINCT aep_premise_nb) AS unique_PNs, 
            MIN(outage_start)              AS min_outage_start, 
            MAX(outage_start)              AS max_outage_start, 
            MIN(duration)                  AS min_duration, 
            MAX(duration)                  AS max_duration, 
            AVG(duration)                  AS avg_duration, 
            AVG(duration*duration)         AS avg_duration_sq, 
            STDDEV(duration)               AS std_duration, 
            STDDEV_POP(duration)           AS std_pop_duration
        FROM 
            id_outage_groups_1
        GROUP BY \n{}, 
            group_by_grp_i
        ORDER BY \n{}, 
            group_by_grp_i
    ),  """.format(
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    )
    #--------------------------------------------------
    sql += """
    agg_outage_groups_0_phase_dur AS (
        SELECT \n{}, 
            phase_val, 
            group_by_grp_i,
            duration_grp, 
            COUNT(serialnumber)            AS total_SN_events,
            COUNT(DISTINCT serialnumber)   AS unique_SNs,
            COUNT(aep_premise_nb)          AS total_PN_events,
            COUNT(DISTINCT aep_premise_nb) AS unique_PNs, 
            MIN(outage_start)              AS min_outage_start, 
            MAX(outage_start)              AS max_outage_start, 
            MIN(duration)                  AS min_duration, 
            MAX(duration)                  AS max_duration, 
            AVG(duration)                  AS avg_duration, 
            AVG(duration*duration)         AS avg_duration_sq, 
            STDDEV(duration)               AS std_duration, 
            STDDEV_POP(duration)           AS std_pop_duration
        FROM 
            id_outage_groups_1
        GROUP BY \n{}, 
            phase_val, 
            group_by_grp_i, 
            duration_grp
        ORDER BY \n{}, 
            phase_val, 
            group_by_grp_i, 
            duration_grp
    ),  """.format(
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    )
    #--------------------------------------------------
    sql += """
    agg_outage_groups_0_phase_dur_wide AS (
        SELECT \n{}, 
            group_by_grp_i,
            -- Below, there shuold only be one entry for each duration_grp (for each group in GROUP BY clause)
            -- Therefore, the SUM/MIN/MAX/w.e. are not really necessary
            ----------------------------------------------------------------------------------------------------
            -- Net momentaries, requested by client
            SUM(
                CASE
                    WHEN duration_grp = 1 THEN total_SN_events ELSE 0
                END
            ) AS total_SN_events_1, 
            SUM(
                CASE
                    WHEN duration_grp = 1 THEN total_PN_events ELSE 0
                END
            ) AS total_PN_events_1, 
            ----------------------------------------------------------------------------------------------------
            -- A_0
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 0 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_A_0, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 0 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_A_0, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 0 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_A_0, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 0 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_A_0, 
            --------------------------------------------------
            -- B_0
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_B_0, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_B_0, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_B_0, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_B_0, 
            --------------------------------------------------
            -- C_0
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_C_0, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_C_0, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_C_0, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 0 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_C_0, 
            ----------------------------------------------------------------------------------------------------
            -- A_1
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 1 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_A_1, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 1 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_A_1, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 1 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_A_1, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 1 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_A_1, 
            --------------------------------------------------
            -- B_1
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_B_1, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_B_1, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_B_1, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_B_1, 
            --------------------------------------------------
            -- C_1
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_C_1, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_C_1, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_C_1, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 1 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_C_1, 
            ----------------------------------------------------------------------------------------------------
            -- A_2
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 2 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_A_2, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 2 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_A_2, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 2 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_A_2, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 2 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_A_2, 
            --------------------------------------------------
            -- B_2
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_B_2, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_B_2, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_B_2, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_B_2, 
            --------------------------------------------------
            -- C_2
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_C_2, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_C_2, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_C_2, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 2 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_C_2, 
            ----------------------------------------------------------------------------------------------------
            -- A_3
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 3 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_A_3, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 3 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_A_3, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 3 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_A_3, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') AND duration_grp = 3 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_A_3, 
            --------------------------------------------------
            -- B_3
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_B_3, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_B_3, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_B_3, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_B_3, 
            --------------------------------------------------
            -- C_3
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN total_SN_events
                    ELSE 0
                END
            ) AS total_SN_events_C_3, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN unique_SNs
                    ELSE 0
                END
            ) AS unique_SNs_C_3, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN total_PN_events
                    ELSE 0
                END
            ) AS total_PN_events_C_3, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') AND duration_grp = 3 THEN unique_PNs
                    ELSE 0
                END
            ) AS unique_PNs_C_3
        FROM 
            agg_outage_groups_0_phase_dur
        GROUP BY \n{}, 
            group_by_grp_i
        ORDER BY \n{}, 
            group_by_grp_i
    ),  """.format(
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    )
    #--------------------------------------------------
    sql += """
    group_infos AS (
        SELECT \n{},
            COUNT(DISTINCT MP.mfr_devc_ser_nbr) AS group_SN_cnt,
            COUNT(DISTINCT MP.prem_nb)          AS group_PN_cnt, 
            AVG(CAST(COALESCE(NULLIF(TRIM(MP.latitude), ''),  NULLIF(TRIM(MP.latitude_nb), '')) AS DOUBLE))  AS avg_latitude, 
            AVG(CAST(COALESCE(NULLIF(TRIM(MP.longitude), ''), NULLIF(TRIM(MP.longitude_nb), '')) AS DOUBLE)) AS avg_longitude
        FROM 
            default.meter_premise MP
        INNER JOIN (SELECT opco_nb,opco_nm FROM default.company) COMP 
            ON  MP.co_cd_ownr=COMP.opco_nb
        INNER JOIN ({}) DGIS
            ON  MP.prem_nb=DGIS.prem_nb
        WHERE 
            COMP.opco_nm = '{}'
        GROUP BY \n{}
    ),""".format(
        Utilities.join_list(
            list_to_join  = [f'            MP.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        dgis_query, 
        opco, 
        Utilities.join_list(
            list_to_join  = [f'            MP.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    )
    #--------------------------------------------------
    sql += """
    group_infos_by_phase_0 AS (
        SELECT \n{},
            DGIS.phase_val, 
            COUNT(DISTINCT MP.mfr_devc_ser_nbr) AS group_SN_cnt,
            COUNT(DISTINCT MP.prem_nb)          AS group_PN_cnt
        FROM 
            default.meter_premise MP
        INNER JOIN (SELECT opco_nb,opco_nm FROM default.company) COMP 
            ON  MP.co_cd_ownr=COMP.opco_nb
        INNER JOIN ({}) DGIS
            ON  MP.prem_nb=DGIS.prem_nb
        WHERE 
            COMP.opco_nm = '{}'
        GROUP BY \n{}, 
            DGIS.phase_val
    ),""".format(
        Utilities.join_list(
            list_to_join  = [f'            MP.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        dgis_query, 
        opco, 
        Utilities.join_list(
            list_to_join  = [f'            MP.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    ) 
    #--------------------------------------------------
    sql += """
    group_infos_by_phase AS (
        SELECT \n{},
            ----------------------------------------------------------------------------------------------------
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') THEN group_SN_cnt
                    ELSE 0
                END
            ) AS group_SN_cnt_A, 
            SUM(
                CASE
                    WHEN (phase_val='A' or phase_val='AB' or phase_val='AC' or phase_val='ABC') THEN group_PN_cnt
                    ELSE 0
                END
            ) AS group_PN_cnt_A, 
            ----------------------------------------------------------------------------------------------------
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') THEN group_SN_cnt
                    ELSE 0
                END
            ) AS group_SN_cnt_B, 
            SUM(
                CASE
                    WHEN (phase_val='B' or phase_val='AB' or phase_val='BC' or phase_val='ABC') THEN group_PN_cnt
                    ELSE 0
                END
            ) AS group_PN_cnt_B, 
            ----------------------------------------------------------------------------------------------------
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') THEN group_SN_cnt
                    ELSE 0
                END
            ) AS group_SN_cnt_C, 
            SUM(
                CASE
                    WHEN (phase_val='C' or phase_val='AC' or phase_val='BC' or phase_val='ABC') THEN group_PN_cnt
                    ELSE 0
                END
            ) AS group_PN_cnt_C
        FROM 
            group_infos_by_phase_0
        GROUP BY \n{}
    ),""".format(
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    )
    #--------------------------------------------------
    sql += """
    agg_outage_groups AS (
        SELECT 
            aog_0_tot.*, 
            group_infos.group_SN_cnt, 
            group_infos.group_PN_cnt, 
            group_infos.avg_latitude, 
            group_infos.avg_longitude, 
            aog_0_tot.total_SN_events*1.0 / group_infos.group_SN_cnt*1.0 AS norm_total_SN_events, -- *1.0 to make double instead of int!
            aog_0_tot.unique_SNs*1.0      / group_infos.group_SN_cnt*1.0 AS norm_unique_SNs, 
            aog_0_tot.total_PN_events*1.0 / group_infos.group_PN_cnt*1.0 AS norm_total_PN_events, 
            aog_0_tot.unique_PNs*1.0      / group_infos.group_PN_cnt*1.0 AS norm_unique_PNs
        FROM 
            agg_outage_groups_0_total aog_0_tot
        LEFT JOIN group_infos 
            ON  {}   
    ), """.format(
        Utilities.join_list(
            list_to_join  = [f"aog_0_tot.{x}=group_infos.{x}" for x in group_by], 
            quotes_needed = False, 
            join_str      = '\n            AND '
        )
    )
    #--------------------------------------------------
    sql += """
    id_super_time_grps_0 AS (
        SELECT
            *, 
            CASE
                WHEN ROW_NUMBER() OVER (ORDER BY min_outage_start) = 1 THEN 0
                ELSE 
                    CASE
                        WHEN DATE_DIFF('second', LAG(max_outage_start) OVER (ORDER BY min_outage_start), min_outage_start) <= {} THEN 0
                        ELSE 1
                    END
            END AS super_time_grp_incrmntr
        FROM
            agg_outage_groups""".format(
        td_sec_final_daisy
    )
    if min_pct_SN is not None:
        sql += """
        WHERE
            norm_unique_SNs > {}""".format(min_pct_SN)
    sql += """
        ORDER BY 
            min_outage_start
    ), """
    #--------------------------------------------------
    sql += """
    id_super_time_grps_1 AS (
        SELECT 
            *, 
            SUM(super_time_grp_incrmntr) OVER (ORDER BY min_outage_start) as super_time_grp
        FROM 
            id_super_time_grps_0
    ) """
    #--------------------------------------------------
    sql += """
    SELECT 
        fnl.super_time_grp, 
        super_times.super_min_outage_start, \n{}, 
        fnl.group_by_grp_i, 
        100.0*fnl.norm_unique_SNs      AS norm_unique_SNs, 
        100.0*fnl.norm_unique_PNs      AS norm_unique_PNs, 
        100.0*fnl.norm_total_SN_events AS norm_total_SN_events, 
        100.0*fnl.norm_total_PN_events AS norm_total_PN_events, 
        --------------------------------------------------
        100.0*aog_phs_dur.unique_SNs_A_0*1.0      / group_infos_bp.group_SN_cnt_A*1.0 AS norm_unique_SNs_A_0,
        100.0*aog_phs_dur.unique_PNs_A_0*1.0      / group_infos_bp.group_PN_cnt_A*1.0 AS norm_unique_PNs_A_0,
        100.0*aog_phs_dur.total_SN_events_A_0*1.0 / group_infos_bp.group_SN_cnt_A*1.0 AS norm_total_SN_events_A_0,
        100.0*aog_phs_dur.total_PN_events_A_0*1.0 / group_infos_bp.group_PN_cnt_A*1.0 AS norm_total_PN_events_A_0,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_B_0*1.0      / group_infos_bp.group_SN_cnt_B*1.0 AS norm_unique_SNs_B_0,
        100.0*aog_phs_dur.unique_PNs_B_0*1.0      / group_infos_bp.group_PN_cnt_B*1.0 AS norm_unique_PNs_B_0,
        100.0*aog_phs_dur.total_SN_events_B_0*1.0 / group_infos_bp.group_SN_cnt_B*1.0 AS norm_total_SN_events_B_0,
        100.0*aog_phs_dur.total_PN_events_B_0*1.0 / group_infos_bp.group_PN_cnt_B*1.0 AS norm_total_PN_events_B_0,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_C_0*1.0      / group_infos_bp.group_SN_cnt_C*1.0 AS norm_unique_SNs_C_0,
        100.0*aog_phs_dur.unique_PNs_C_0*1.0      / group_infos_bp.group_PN_cnt_C*1.0 AS norm_unique_PNs_C_0,
        100.0*aog_phs_dur.total_SN_events_C_0*1.0 / group_infos_bp.group_SN_cnt_C*1.0 AS norm_total_SN_events_C_0,
        100.0*aog_phs_dur.total_PN_events_C_0*1.0 / group_infos_bp.group_PN_cnt_C*1.0 AS norm_total_PN_events_C_0,
        --------------------------------------------------
        100.0*aog_phs_dur.unique_SNs_A_1*1.0      / group_infos_bp.group_SN_cnt_A*1.0 AS norm_unique_SNs_A_1,
        100.0*aog_phs_dur.unique_PNs_A_1*1.0      / group_infos_bp.group_PN_cnt_A*1.0 AS norm_unique_PNs_A_1,
        100.0*aog_phs_dur.total_SN_events_A_1*1.0 / group_infos_bp.group_SN_cnt_A*1.0 AS norm_total_SN_events_A_1,
        100.0*aog_phs_dur.total_PN_events_A_1*1.0 / group_infos_bp.group_PN_cnt_A*1.0 AS norm_total_PN_events_A_1,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_B_1*1.0      / group_infos_bp.group_SN_cnt_B*1.0 AS norm_unique_SNs_B_1,
        100.0*aog_phs_dur.unique_PNs_B_1*1.0      / group_infos_bp.group_PN_cnt_B*1.0 AS norm_unique_PNs_B_1,
        100.0*aog_phs_dur.total_SN_events_B_1*1.0 / group_infos_bp.group_SN_cnt_B*1.0 AS norm_total_SN_events_B_1,
        100.0*aog_phs_dur.total_PN_events_B_1*1.0 / group_infos_bp.group_PN_cnt_B*1.0 AS norm_total_PN_events_B_1,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_C_1*1.0      / group_infos_bp.group_SN_cnt_C*1.0 AS norm_unique_SNs_C_1,
        100.0*aog_phs_dur.unique_PNs_C_1*1.0      / group_infos_bp.group_PN_cnt_C*1.0 AS norm_unique_PNs_C_1,
        100.0*aog_phs_dur.total_SN_events_C_1*1.0 / group_infos_bp.group_SN_cnt_C*1.0 AS norm_total_SN_events_C_1,
        100.0*aog_phs_dur.total_PN_events_C_1*1.0 / group_infos_bp.group_PN_cnt_C*1.0 AS norm_total_PN_events_C_1,
        --------------------------------------------------
        100.0*aog_phs_dur.unique_SNs_A_2*1.0      / group_infos_bp.group_SN_cnt_A*1.0 AS norm_unique_SNs_A_2,
        100.0*aog_phs_dur.unique_PNs_A_2*1.0      / group_infos_bp.group_PN_cnt_A*1.0 AS norm_unique_PNs_A_2,
        100.0*aog_phs_dur.total_SN_events_A_2*1.0 / group_infos_bp.group_SN_cnt_A*1.0 AS norm_total_SN_events_A_2,
        100.0*aog_phs_dur.total_PN_events_A_2*1.0 / group_infos_bp.group_PN_cnt_A*1.0 AS norm_total_PN_events_A_2,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_B_2*1.0      / group_infos_bp.group_SN_cnt_B*1.0 AS norm_unique_SNs_B_2,
        100.0*aog_phs_dur.unique_PNs_B_2*1.0      / group_infos_bp.group_PN_cnt_B*1.0 AS norm_unique_PNs_B_2,
        100.0*aog_phs_dur.total_SN_events_B_2*1.0 / group_infos_bp.group_SN_cnt_B*1.0 AS norm_total_SN_events_B_2,
        100.0*aog_phs_dur.total_PN_events_B_2*1.0 / group_infos_bp.group_PN_cnt_B*1.0 AS norm_total_PN_events_B_2,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_C_2*1.0      / group_infos_bp.group_SN_cnt_C*1.0 AS norm_unique_SNs_C_2,
        100.0*aog_phs_dur.unique_PNs_C_2*1.0      / group_infos_bp.group_PN_cnt_C*1.0 AS norm_unique_PNs_C_2,
        100.0*aog_phs_dur.total_SN_events_C_2*1.0 / group_infos_bp.group_SN_cnt_C*1.0 AS norm_total_SN_events_C_2,
        100.0*aog_phs_dur.total_PN_events_C_2*1.0 / group_infos_bp.group_PN_cnt_C*1.0 AS norm_total_PN_events_C_2,
        --------------------------------------------------
        100.0*aog_phs_dur.unique_SNs_A_3*1.0      / group_infos_bp.group_SN_cnt_A*1.0 AS norm_unique_SNs_A_3,
        100.0*aog_phs_dur.unique_PNs_A_3*1.0      / group_infos_bp.group_PN_cnt_A*1.0 AS norm_unique_PNs_A_3,
        100.0*aog_phs_dur.total_SN_events_A_3*1.0 / group_infos_bp.group_SN_cnt_A*1.0 AS norm_total_SN_events_A_3,
        100.0*aog_phs_dur.total_PN_events_A_3*1.0 / group_infos_bp.group_PN_cnt_A*1.0 AS norm_total_PN_events_A_3,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_B_3*1.0      / group_infos_bp.group_SN_cnt_B*1.0 AS norm_unique_SNs_B_3,
        100.0*aog_phs_dur.unique_PNs_B_3*1.0      / group_infos_bp.group_PN_cnt_B*1.0 AS norm_unique_PNs_B_3,
        100.0*aog_phs_dur.total_SN_events_B_3*1.0 / group_infos_bp.group_SN_cnt_B*1.0 AS norm_total_SN_events_B_3,
        100.0*aog_phs_dur.total_PN_events_B_3*1.0 / group_infos_bp.group_PN_cnt_B*1.0 AS norm_total_PN_events_B_3,
        -------------------------
        100.0*aog_phs_dur.unique_SNs_C_3*1.0      / group_infos_bp.group_SN_cnt_C*1.0 AS norm_unique_SNs_C_3,
        100.0*aog_phs_dur.unique_PNs_C_3*1.0      / group_infos_bp.group_PN_cnt_C*1.0 AS norm_unique_PNs_C_3,
        100.0*aog_phs_dur.total_SN_events_C_3*1.0 / group_infos_bp.group_SN_cnt_C*1.0 AS norm_total_SN_events_C_3,
        100.0*aog_phs_dur.total_PN_events_C_3*1.0 / group_infos_bp.group_PN_cnt_C*1.0 AS norm_total_PN_events_C_3,
        --------------------------------------------------
        fnl.min_outage_start, 
        fnl.max_outage_start, 
        fnl.min_duration, 
        fnl.max_duration, 
        fnl.avg_duration, 
        fnl.avg_duration_sq, 
        fnl.std_duration, 
        fnl.std_pop_duration, 
        fnl.unique_SNs, 
        fnl.unique_PNs, 
        fnl.total_SN_events, 
        fnl.total_PN_events, 
        --------------------------------------------------
        aog_phs_dur.total_SN_events_1, 
        aog_phs_dur.total_PN_events_1, 
        --------------------------------------------------
        aog_phs_dur.unique_SNs_A_0,
        aog_phs_dur.unique_PNs_A_0,
        aog_phs_dur.total_SN_events_A_0,
        aog_phs_dur.total_PN_events_A_0,
        -------------------------
        aog_phs_dur.unique_SNs_B_0,
        aog_phs_dur.unique_PNs_B_0,
        aog_phs_dur.total_SN_events_B_0,
        aog_phs_dur.total_PN_events_B_0,
        -------------------------
        aog_phs_dur.unique_SNs_C_0,
        aog_phs_dur.unique_PNs_C_0,
        aog_phs_dur.total_SN_events_C_0,
        aog_phs_dur.total_PN_events_C_0,
        --------------------------------------------------
        aog_phs_dur.unique_SNs_A_1,
        aog_phs_dur.unique_PNs_A_1,
        aog_phs_dur.total_SN_events_A_1,
        aog_phs_dur.total_PN_events_A_1,
        -------------------------
        aog_phs_dur.unique_SNs_B_1,
        aog_phs_dur.unique_PNs_B_1,
        aog_phs_dur.total_SN_events_B_1,
        aog_phs_dur.total_PN_events_B_1,
        -------------------------
        aog_phs_dur.unique_SNs_C_1,
        aog_phs_dur.unique_PNs_C_1,
        aog_phs_dur.total_SN_events_C_1,
        aog_phs_dur.total_PN_events_C_1,
        --------------------------------------------------
        aog_phs_dur.unique_SNs_A_2,
        aog_phs_dur.unique_PNs_A_2,
        aog_phs_dur.total_SN_events_A_2,
        aog_phs_dur.total_PN_events_A_2,
        -------------------------
        aog_phs_dur.unique_SNs_B_2,
        aog_phs_dur.unique_PNs_B_2,
        aog_phs_dur.total_SN_events_B_2,
        aog_phs_dur.total_PN_events_B_2,
        -------------------------
        aog_phs_dur.unique_SNs_C_2,
        aog_phs_dur.unique_PNs_C_2,
        aog_phs_dur.total_SN_events_C_2,
        aog_phs_dur.total_PN_events_C_2,
        --------------------------------------------------
        aog_phs_dur.unique_SNs_A_3,
        aog_phs_dur.unique_PNs_A_3,
        aog_phs_dur.total_SN_events_A_3,
        aog_phs_dur.total_PN_events_A_3,
        -------------------------
        aog_phs_dur.unique_SNs_B_3,
        aog_phs_dur.unique_PNs_B_3,
        aog_phs_dur.total_SN_events_B_3,
        aog_phs_dur.total_PN_events_B_3,
        -------------------------
        aog_phs_dur.unique_SNs_C_3,
        aog_phs_dur.unique_PNs_C_3,
        aog_phs_dur.total_SN_events_C_3,
        aog_phs_dur.total_PN_events_C_3,
        --------------------------------------------------
        fnl.group_SN_cnt, 
        fnl.group_PN_cnt, 
        group_infos_bp.group_SN_cnt_A, 
        group_infos_bp.group_PN_cnt_A, 
        group_infos_bp.group_SN_cnt_B, 
        group_infos_bp.group_PN_cnt_B, 
        group_infos_bp.group_SN_cnt_C, 
        group_infos_bp.group_PN_cnt_C, 
        fnl.avg_latitude, 
        fnl.avg_longitude
    FROM
        id_super_time_grps_1 fnl
    LEFT JOIN agg_outage_groups_0_phase_dur_wide aog_phs_dur 
        ON  {}
        AND fnl.group_by_grp_i=aog_phs_dur.group_by_grp_i
    LEFT JOIN group_infos_by_phase group_infos_bp
        ON  {}
    LEFT JOIN (
        SELECT 
            super_time_grp,
            MIN(min_outage_start) AS super_min_outage_start 
        FROM 
            id_super_time_grps_1
        GROUP BY
            super_time_grp
    ) super_times
        ON  fnl.super_time_grp=super_times.super_time_grp
    ORDER BY 
        fnl.super_time_grp, \n{}, 
        fnl.group_by_grp_i""".format(
        Utilities.join_list(
            list_to_join  = [f'        fnl.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f"fnl.{x}=aog_phs_dur.{x}" for x in group_by], 
            quotes_needed = False, 
            join_str      = '\n        AND '
        ), 
        Utilities.join_list(
            list_to_join  = [f"fnl.{x}=group_infos_bp.{x}" for x in group_by], 
            quotes_needed = False, 
            join_str      = '\n        AND '
        ), 
        Utilities.join_list(
            list_to_join  = [f'        fnl.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    )
    #--------------------------------------------------
    return sql


#----------------------------------------------------------------------------------------------------
def build_avg_momentaries_sql(
    date_0                 , 
    date_1                 , 
    opco                   , 
    td_sec_seqntl_pu_or_pd = 5, 
    group_by               = ['station_nb', 'circuit_nb'], 
    pd_ids                 = ['3.26.0.47', '3.26.136.47', '3.26.136.66'], 
    pu_ids                 = ['3.26.0.216', '3.26.136.216']
):
    r"""
    """
    #--------------------------------------------------
    accptbl_group_by = ['station_nb', 'station_nm', 'circuit_nb', 'circuit_nm', 'trsf_pole_nb']
    assert(set(group_by).difference(set(accptbl_group_by))==set())
    #--------------------------------------------------
    # NOTE: In SQL queries, the dates in the range are INCLUSIVE, hence to addition of 1 day below
    pd_n_days = (pd.to_datetime(date_1)-pd.to_datetime(date_0)+pd.Timedelta('1D')).days
    #--------------------------------------------------
    sql = build_pdpu_matcher_sql(
        date_0                 = date_0, 
        date_1                 = date_1, 
        opco                   = opco, 
        td_sec_seqntl_pu_or_pd = td_sec_seqntl_pu_or_pd, 
        addtnl_group_by        = group_by, 
        pd_ids                 = pd_ids, 
        pu_ids                 = pu_ids, 
        include_final_select   = False
    )+','
    #--------------------------------------------------
    sql += """
    agg_outage_groups_by_SN_2 AS (
        SELECT 
            *, 
            DATE_DIFF('second', outage_start, outage_end) as duration
        FROM 
            agg_outage_groups_by_SN
        WHERE 
            outage_start IS NOT NULL
            AND DATE_DIFF('second', outage_start, outage_end) >= 8
            AND DATE_DIFF('second', outage_start, outage_end) < 300
    ), """
    #--------------------------------------------------
    sql += """
    n_momentaries_by_SN AS (
        SELECT \n{}, 
            aep_premise_nb, 
            serialnumber,
            phase_val, 
            COUNT(*) AS n_momentaries
        FROM
            agg_outage_groups_by_SN_2
        GROUP BY \n{},  
            aep_premise_nb, 
            serialnumber, 
            phase_val
        ORDER BY \n{},  
            aep_premise_nb, 
            serialnumber, 
            phase_val
    ), """.format(
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
    )
    #--------------------------------------------------
    sql += """
    group_infos AS (
        SELECT \n{},
            COUNT(DISTINCT MP.mfr_devc_ser_nbr) AS group_SN_cnt,
            COUNT(DISTINCT MP.prem_nb)          AS group_PN_cnt
        FROM 
            default.meter_premise MP
        INNER JOIN (SELECT opco_nb,opco_nm FROM default.company) COMP 
            ON  MP.co_cd_ownr=COMP.opco_nb
        INNER JOIN (SELECT prem_nb, " phase_val" AS phase_val FROM cds_ds_db.com_ccp_dgis_extract) DGIS
            ON  MP.prem_nb=DGIS.prem_nb
        WHERE 
            COMP.opco_nm = '{}'
        GROUP BY \n{}
    ),""".format(
        Utilities.join_list(
            list_to_join  = [f'            MP.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        opco, 
        Utilities.join_list(
            list_to_join  = [f'            MP.{x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        )
    )
    #--------------------------------------------------
    sql += """
    agg_momentaries AS (
        SELECT \n{}, 
            COUNT(DISTINCT CASE WHEN n_momentaries > 13 THEN serialnumber END)   AS nSNs_w_gt13_mom,
            COUNT(DISTINCT CASE WHEN n_momentaries > 13 THEN aep_premise_nb END) AS nPNs_w_gt13_mom,
            SUM(n_momentaries) AS sum_n_momentaries
        FROM
            n_momentaries_by_SN
        GROUP BY \n{} 
    ) """.format(
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        Utilities.join_list(
            list_to_join  = [f'            {x}' for x in group_by], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
    )
    #--------------------------------------------------
    # NOTE: Below, for average, cannot simply call AVG, as this would be the average of meters registering an outage
    #       We want average of all meters in group, so we must normalize using group_infos.group_SN_cnt
    #-------------------------
    # Include the nSNs_w_gt13_mom/nPNs_w_gt13_mom columns ONLY if pd_n_days==365
    n_w_gt13_mom_cols = []
    if pd_n_days==365:
        n_w_gt13_mom_cols = ['nSNs_w_gt13_mom', 'nPNs_w_gt13_mom']
    #-------------------------
    sql += """
    SELECT \n{}, 
        agg_mom.sum_n_momentaries*1.0/(group_infos.group_SN_cnt*1.0*{}) AS avg_daily_n_mom_per_SN, 
        agg_mom.sum_n_momentaries*1.0/(group_infos.group_PN_cnt*1.0*{}) AS avg_daily_n_mom_per_PN
    FROM
        agg_momentaries agg_mom
    LEFT JOIN group_infos 
        ON  {} """.format(
        Utilities.join_list(
            list_to_join  = [f'        agg_mom.{x}' for x in group_by+n_w_gt13_mom_cols], 
            quotes_needed = False, 
            join_str      = ', \n'
        ), 
        pd_n_days, 
        pd_n_days, 
        Utilities.join_list(
            list_to_join  = [f"agg_mom.{x}=group_infos.{x}" for x in group_by], 
            quotes_needed = False, 
            join_str      = '\n        AND '
        )
    )

    #--------------------------------------------------
    return sql


#----------------------------------------------------------------------------------------------------
def set_spatial_groups_for_time_group_i(
    df_i           , 
    eps_km         = 25, 
    min_samples    = 1, 
    lat_col        = 'avg_latitude', 
    lon_col        = 'avg_longitude', 
    spat_group_col = 'spatial_grp'
):
    r"""
    Intended for use with df_i containing only a single time group!
    Also, lat_col/lon_col must NOT contain any NaNs (i.e., parent df should be cleaned before using this)
    """
    #--------------------------------------------------
    kms_per_radian = 6371.0088
    eps            = eps_km/kms_per_radian
    #-------------------------
    db = DBSCAN(
        eps         = eps, 
        min_samples = min_samples, 
        algorithm   = 'ball_tree', 
        metric      = 'haversine'
    ).fit(np.radians(df_i[[lat_col, lon_col]]))
    #-------------------------
    df_i[spat_group_col] = db.labels_
    #-------------------------
    return df_i

#----------------------------------------------------------------------------------------------------
def set_spatial_groups(
    df             , 
    time_group_col = 'super_time_grp', 
    eps_km         = 25, 
    min_samples    = 1, 
    lat_col        = 'avg_latitude', 
    lon_col        = 'avg_longitude', 
    spat_group_col = 'spatial_grp', 
    fnl_group_col  = 'super_time_spatial_grp'
):
    r"""
    Intended for use with df_i containing only a single time group!
    Also, lat_col/lon_col must NOT contain any NaNs (i.e., parent df should be cleaned before using this)
    """
    #--------------------------------------------------
    df[spat_group_col] = -1
    #-------------------------
    # DBSCAN doesn't like NaN values, so drop those
    # We will add back at end
    df_na     = df[(df[lat_col].isna()) | (df[lon_col].isna())].copy()
    df_non_na = df.dropna(subset=[lat_col, lon_col]).copy()
    #-------------------------
    df_non_na = df_non_na.groupby(time_group_col, as_index=False, group_keys=False)[df_non_na.columns].apply(
        lambda x: set_spatial_groups_for_time_group_i(
            df_i           = x, 
            eps_km         = eps_km, 
            min_samples    = min_samples, 
            lat_col        = lat_col, 
            lon_col        = lon_col, 
            spat_group_col = spat_group_col
        )
    )
    #-------------------------
    df_non_na[fnl_group_col] = df_non_na[time_group_col].astype(str) + '_' + df_non_na[spat_group_col].astype(str)
    df_na[fnl_group_col]     = df_na[time_group_col].astype(str)     + '_NaN'
    #-------------------------
    df = pd.concat([df_non_na, df_na], axis=0)
    #-------------------------
    # Move [fnl_group_col, time_group_col, spat_group_col] to front
    front_cols     = [fnl_group_col, time_group_col, spat_group_col]
    remaining_cols = [x for x in df.columns if x not in front_cols]
    df             =  df[front_cols + remaining_cols]
    #-------------------------
    return df


#----------------------------------------------------------------------------------------------------
def plot_spatial_groups_for_time_group_i(
    df_i, 
    us_ok, 
    spat_group_col = 'spatial_grp', 
    lat_col        = 'avg_latitude', 
    lon_col        = 'avg_longitude',     
):
    r"""
    Intended for use with df_i containing only a single time group!
    Also, lat_col/lon_col must NOT contain any NaNs (i.e., parent df should be cleaned before using this)
    """
    #--------------------------------------------------
    if us_ok is None:
        us_shp_path = r'C:\Users\s346557\Downloads\tl_2023_us_state\tl_2023_us_state.shp'
        us_df = gpd.read_file(us_shp_path)
        us_df = us_df.to_crs("EPSG:4326")
        #-----
        us_ok = us_df[us_df['STUSPS'].isin(['OK'])].copy()
    #--------------------------------------------------
    geometry = [Point(xy) for xy in zip(df_i[lon_col], df_i[lat_col])]
    geo_df_i = gpd.GeoDataFrame(
        df_i, 
        crs = 'EPSG:4326', 
        geometry = geometry
    )
    geo_df_i = geo_df_i.dropna(subset=[lon_col, lat_col])
    #--------------------------------------------------
    colors      = Plot_General.get_standard_colors(n_colors = geo_df_i[spat_group_col].nunique())
    edge_colors = Plot_General.get_standard_colors(n_colors = 10, palette='colorblind')
    #--------------------------------------------------
    tmp_size_col           = Utilities.generate_random_string(letters='letters_only')
    geo_df_i[tmp_size_col] = MinMaxScaler(feature_range=(10, 100)).fit_transform(geo_df_i['norm_unique_SNs'].values.reshape(-1,1))    
    #-------------------------
    fig,ax = Plot_General.default_subplots(
        n_x                 = 1,
        n_y                 = 1,
        fig_num             = 0,
        unit_figsize_width  = 14,
        unit_figsize_height = 6
    )
    #-------------------------
    us_ok.plot(ax=ax, color='lightgrey')
    #-------------------------
    for j,spat_group_j in enumerate(natsorted(geo_df_i[spat_group_col].unique())):
        df_ij    = geo_df_i[geo_df_i[spat_group_col]==spat_group_j]
        label_ij = "{} - {}".format(
            df_ij['min_outage_start'].min().strftime('%H:%M:%S'), 
            df_ij['min_outage_start'].max().strftime('%H:%M:%S')
        )
        df_ij.plot(
            ax=ax, 
            color=colors[j], 
            alpha=0.5, 
            edgecolors=edge_colors[j%10], 
            label=label_ij, 
            markersize = tmp_size_col
        )
    #-------------------------
    label = "{} - {}".format(
        df_i['min_outage_start'].min().strftime('%Y-%m-%d %H:%M:%S'), 
        df_i['min_outage_start'].max().strftime('%Y-%m-%d %H:%M:%S')
    )
    ax.set_title(
        label = label,
        fontdict = dict(
            fontsize = 'x-large'
        )
    )
    ax.legend()
    #--------------------------------------------------
    return fig,ax


#----------------------------------------------------------------------------------------------------
def write_CRED_worksheet(
    writer          , 
    df_fnl          , 
    reset_index     , 
    format1         , 
    format2         , 
    format3         , 
    format4         , 
    idx_format      , 
    normalize       = True, 
    n_avg_moms_cols = 0, 
):
    r"""
    """
    #--------------------------------------------------
    if reset_index:
        addtnl_col_offset = df_fnl.index.nlevels #should be 1
        #-----
        # df_fnl.columns = [','.join(x) for x in df_fnl.columns]
        df_fnl.columns = [x[1] for x in df_fnl.columns] 
        df_fnl         = df_fnl.reset_index().copy()
        #-----
        df_fnl.to_excel(writer, sheet_name='noIdxGroups', header=True, index=True)
        worksheet = writer.sheets['noIdxGroups']
    else:
        addtnl_col_offset = 0
        #-----
        df_fnl.to_excel(writer, sheet_name='IdxGroups', header=True, index=True)
        worksheet = writer.sheets['IdxGroups']
    #-------------------------
    if normalize:
        addtnl_col_offset += 1
    #-------------------------
    worksheet.autofit()
    #-------------------------
    color_mapping = {
        2+addtnl_col_offset+df_fnl.index.nlevels  : format1, 
        3+addtnl_col_offset+df_fnl.index.nlevels  : format1, 
        4+addtnl_col_offset+df_fnl.index.nlevels  : format1, 
        #-----
        5+addtnl_col_offset+df_fnl.index.nlevels  : format2, 
        6+addtnl_col_offset+df_fnl.index.nlevels  : format2, 
        7+addtnl_col_offset+df_fnl.index.nlevels  : format2, 
        #-----
        8+addtnl_col_offset+df_fnl.index.nlevels  : format3, 
        9+addtnl_col_offset+df_fnl.index.nlevels  : format3, 
        10+addtnl_col_offset+df_fnl.index.nlevels : format3, 
        #-----
        11+addtnl_col_offset+df_fnl.index.nlevels : format4, 
        12+addtnl_col_offset+df_fnl.index.nlevels : format4, 
        13+addtnl_col_offset+df_fnl.index.nlevels : format4, 
        #-----
    }
    #-----
    width = 9
    for col,fmt in color_mapping.items():
        # worksheet.set_column(col, col, width, cell_format=fmt)
        worksheet.set_column(col, col, width)
        worksheet.conditional_format(0, col, df_fnl.shape[0]+df_fnl.columns.nlevels, col, options={'type':'no_blanks', 'format':fmt})
        worksheet.conditional_format(0, col, df_fnl.shape[0]+df_fnl.columns.nlevels, col, options={'type':'blanks', 'format':fmt})
    #-------------------------
    # For whatever reason, autofit slightly undersizes min(max)_outage_start columns
    worksheet.set_column(
        14 + df_fnl.index.nlevels + addtnl_col_offset + n_avg_moms_cols, 
        15 + df_fnl.index.nlevels + addtnl_col_offset + n_avg_moms_cols, 
        18
    )
    #-------------------------
    if not reset_index:
        for idx_i in range(df_fnl.index.nlevels):
            column_i = list(string.ascii_uppercase)[idx_i]
            worksheet.merge_range(f'{column_i}1:{column_i}{df_fnl.columns.nlevels+1}', df_fnl.index.names[idx_i], idx_format)
    #-------------------------
    worksheet.write(df_fnl.shape[0]+df_fnl.columns.nlevels+5, 0, '*Note: % values between 0 and 1 set to 1')


#----------------------------------------------------------------------------------------------------
def save_CRED_xlsx(
    df_fnl          , 
    save_dir        , 
    opco            , 
    date_0          , 
    date_1          , 
    normalize       = True, 
    n_avg_moms_cols = 0, 
    reset_index     = 'both'
):
    r"""
    """
    #--------------------------------------------------
    assert(reset_index=='both' or isinstance(reset_index, bool))
    #-------------------------
    assert(isinstance(df_fnl, pd.DataFrame) or isinstance(df_fnl, list))
    #-------------------------
    if date_0==date_1:
        save_name = "CRED_{}_{}".format(
            opco, 
            pd.to_datetime(date_0).strftime('%Y%m%d')
        )
    else:
        save_name = "CRED_{}_{}_{}".format(
            opco, 
            pd.to_datetime(date_0).strftime('%Y%m%d'), 
            pd.to_datetime(date_1).strftime('%Y%m%d')
        )
    #-----
    if reset_index==False:
        save_name += '_noIdxGroups'
    save_name += '.xlsx'
    #-----
    save_path = os.path.join(save_dir, save_name)
    #-------------------------
    writer = pd.ExcelWriter(
        path   = save_path, 
        engine = 'xlsxwriter'
    )
    #-------------------------
    workbook  = writer.book
    #-------------------------
    format1 = workbook.add_format({'bg_color': '#a1c9f4', 'border' : 1, 'border_color' : '#000000'})
    format2 = workbook.add_format({'bg_color': '#ffb482', 'border' : 1, 'border_color' : '#000000'})
    format3 = workbook.add_format({'bg_color': '#8de5a1', 'border' : 1, 'border_color' : '#000000'})
    format4 = workbook.add_format({'bg_color': '#ff9f9b', 'border' : 1, 'border_color' : '#000000'})
    #-------------------------
    bold_centered_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
    #-------------------------
    if reset_index=='both':
        if isinstance(df_fnl, list):
            assert(len(df_fnl)==2)
            df_1 = df_fnl[0]
            df_2 = df_fnl[1]
        else:
            df_1 = df_fnl
            df_2 = df_fnl
        #-----
        write_CRED_worksheet(
            writer          = writer, 
            df_fnl          = df_1, 
            reset_index     = False, 
            format1         = format1, 
            format2         = format2, 
            format3         = format3, 
            format4         = format4, 
            idx_format      = bold_centered_format, 
            normalize       = normalize, 
            n_avg_moms_cols = n_avg_moms_cols, 
        )
        #-----
        write_CRED_worksheet(
            writer          = writer, 
            df_fnl          = df_2, 
            reset_index     = True, 
            format1         = format1, 
            format2         = format2, 
            format3         = format3, 
            format4         = format4, 
            idx_format      = bold_centered_format, 
            normalize       = normalize, 
            n_avg_moms_cols = n_avg_moms_cols, 
        )
    else:
        assert(isinstance(df_fnl, pd.DataFrame))
        write_CRED_worksheet(
            writer          = writer, 
            df_fnl          = df_fnl, 
            reset_index     = reset_index, 
            format1         = format1, 
            format2         = format2, 
            format3         = format3, 
            format4         = format4, 
            idx_format      = bold_centered_format, 
            normalize       = normalize, 
            n_avg_moms_cols = n_avg_moms_cols, 
        )
    writer.close()
    return save_name, save_path

#----------------------------------------------------------------------------------------------------
def convert_xlsx_to_xlsm(xlsx_file):
    # Initialize the Excel application
    excel = win32.Dispatch("Excel.Application")
    excel.Visible = False  # Set to True if you want to see the process

    # Open the .xlsx file
    workbook = excel.Workbooks.Open(xlsx_file)

    # Save as .xlsm
    xlsm_file = xlsx_file.replace('.xlsx', '.xlsm')
    workbook.SaveAs(xlsm_file, FileFormat=52)  # 52 is the file format for .xlsm

    # Close the workbook and quit Excel
    workbook.Close(SaveChanges=True)
    excel.Quit()

    return xlsm_file

#----------------------------------------------------------------------------------------------------
def convert_xlsm_to_xlsx(xlsm_file):
    # Initialize the Excel application
    excel = win32.Dispatch("Excel.Application")
    excel.Visible = False  # Set to True if you want to see the process

    # Open the .xlsm file
    workbook = excel.Workbooks.Open(xlsm_file)

    # Suppress alerts
    # Specifically because if I don't, a window pops up saying some things can't be saved in macro-free
    #   workbook and asks if I want to continue (thankfully, default option is Yes, continue).
    excel.DisplayAlerts = False

    # Save as .xlsx
    xlsx_file = xlsm_file.replace('.xlsm', '.xlsx')
    # If path exists, window pops up and asks if I want to replace.  Sadly, default is no, so must remove myself.
    if os.path.exists(xlsx_file):
        os.remove(xlsx_file)
    workbook.SaveAs(xlsx_file, FileFormat=51)  # 51 is the file format for .xlsx

    # Close the workbook and quit Excel
    workbook.Close(SaveChanges=True)
    excel.Quit()

    return xlsx_file

#----------------------------------------------------------------------------------------------------
def import_and_run_vba_macro(
    excel_file_path , 
    vba_file_path   , 
    vba_macro_name  ,
    pretty_group_by , 
    addtnl_cols     = None
):
    r"""
    So annoying.  As long as the VBA macro does not accept any parameters, a .xlsx file is fine.
    BUT, if it does accept parameters, then .xlsm must be used!

    It seems like the multiple win32.Dispatch objects that are utilized here (one explicitly, two behind the
      scenes in convert_xlsx_to_xlsm/convert_xlsm_to_xlsx) can overload Excel, I guess.
    Inserting a second of wait time seems to fix things....
    """
    #-------------------------
    # Convert .xlsx to .xlsm
    xlsm_file = convert_xlsx_to_xlsm(excel_file_path)
    time.sleep(1)

    #-------------------------
    # Create an instance of Excel
    excel = win32.Dispatch("Excel.Application")
    excel.Visible = False  # Optional: Set to True to see Excel

    # Open the workbook
    workbook = excel.Workbooks.Open(xlsm_file)

    try:
        # Import the VBA module
        vb_component = workbook.VBProject.VBComponents.Add(1)  # 1 corresponds to a module
        with open(vba_file_path, 'r') as f:
            vba_code = f.read()
        vb_component.CodeModule.AddFromString(vba_code)
    
        # Run the macro
        # It was a headache to try to feed a list to the macro, so convert list to CSV
        # But, to avoid confusion, the elements of pretty_group_by should therefore not contain commas!
        #-------------------------
        assert(np.all([x.find(',')==-1 for x in group_by]))
        group_by_string = ','.join(pretty_group_by)
        #-------------------------
        if addtnl_cols is None:
            addtnl_cols_string = ''
        else:
            assert(np.all([x.find(',')==-1 for x in addtnl_cols]))
            addtnl_cols_string = ','.join(addtnl_cols)            
        excel.Application.Run(vba_macro_name, group_by_string, addtnl_cols_string)
    
        # Optional: Save and close the workbook
        workbook.Save()
        workbook.Close()
    
        # Quit Excel application
        excel.Quit()
        time.sleep(1)

        # Convert back to .xlsx
        output_xlsx = convert_xlsm_to_xlsx(xlsm_file)
        time.sleep(1)

        # Clean up
        os.remove(xlsm_file)
    except:
        workbook.Close()
        excel.Quit()
        os.remove(xlsm_file)


#----------------------------------------------------------------------------------------------------
def get_CRED_sort_order(
    df            , 
    group_by      , 
    date_idx      = 'date', 
    event_tag_idx = 'super_time_spatial_grp_fnl', 
    n_moms_col    = 'total_PN_events_1', 
):
    r"""
    Sort by
    1. Date
    2. Number of momentaries in overall event tag
    3. Number of momentaries in group_by columns
    """
    #--------------------------------------------------
    assert(set(df.index.names).symmetric_difference(set([date_idx, event_tag_idx]+group_by))==set())
    #-------------------------
    sort_df = df.groupby([date_idx, event_tag_idx])[n_moms_col].sum().sort_values(ascending=False).reset_index(name='sort_lvl_0')
    #-------------------------
    for i in range(len(group_by)):
        sort_df_i = df.groupby([date_idx, event_tag_idx]+group_by[:i+1])[n_moms_col].sum().sort_values(ascending=False).reset_index(name=f'sort_lvl_{i+1}')
        #-----
        sort_df = pd.merge(
            sort_df, 
            sort_df_i, 
            how      = 'inner', 
            left_on  = [date_idx, event_tag_idx]+group_by[:i], 
            right_on = [date_idx, event_tag_idx]+group_by[:i], 
        )
    #-------------------------
    srt_by = [date_idx] + [f'sort_lvl_{i}' for i in range(len(group_by)+1)]
    asc    = [True]     + [False for _ in range(len(group_by)+1)]
    #-------------------------
    idx_order = sort_df.sort_values(by=srt_by, ascending=asc).set_index([date_idx, event_tag_idx]+group_by).index
    assert(idx_order.nunique()==len(idx_order))
    #-------------------------
    return idx_order

#----------------------------------------------------------------------------------------------------
def sort_CRED_df(
    df            , 
    group_by      , 
    date_idx      = 'date', 
    event_tag_idx = 'super_time_spatial_grp_fnl', 
    n_moms_col    = 'total_PN_events_1', 
):
    r"""
    """
    #--------------------------------------------------
    idx_order = get_CRED_sort_order(
        df            = df, 
        group_by      = group_by, 
        date_idx      = date_idx, 
        event_tag_idx = event_tag_idx, 
        n_moms_col    = n_moms_col, 
    )
    #-------------------------
    df = df.loc[idx_order]
    return df


#----------------------------------------------------------------------------------------------------
def run_cred(
    date_0                 , 
    date_1                 , 
    opco                   , 
    min_pct_SN             = 0.10, 
    by_premise             = True, 
    normalize              = True, 
    td_sec_group_daisy     = 5, 
    td_sec_seqntl_pu_or_pd = 5, 
    td_sec_final_daisy     = 5, 
    eps_km                 = 30, 
    group_by               = ['station_nb', 'circuit_nb'], 
    pd_ids                 = ['3.26.0.47'], 
    pu_ids                 = ['3.26.0.216'], 
    conn_aws               = None
):
    r"""
    """
    #----------------------------------------------------------------------------------------------------
    accptbl_group_by = ['station_nb', 'station_nm', 'circuit_nb', 'circuit_nm', 'trsf_pole_nb']
    assert(set(group_by).difference(set(accptbl_group_by))==set())
    #--------------------------------------------------
    # BELOW LINES ARE TEMPORARY UNTIL FULL cds_ds_db.com_ccp_dgis_extract table set up!!!!!
    assert(opco in ['pso', 'im'])
    if opco=='pso':
        dgis_query = 'SELECT prem_nb, " phase_val" AS phase_val FROM cds_ds_db.com_ccp_dgis_extract'
    else:
        dgis_query = 'SELECT prem_nb, phase_val FROM cds_ds_db.com_ccp_dgis_extract_im'
    #--------------------------------------------------
    if conn_aws is None:
        conn_aws = Utilities.get_athena_prod_aws_connection()
    #-------------------------
    CRED_sql = build_CRED_sql(
        date_0                 = date_0, 
        date_1                 = date_1, 
        opco                   = opco, 
        min_pct_SN             = min_pct_SN, 
        td_sec_group_daisy     = td_sec_group_daisy, 
        td_sec_seqntl_pu_or_pd = td_sec_seqntl_pu_or_pd, 
        td_sec_final_daisy     = td_sec_final_daisy, 
        group_by               = group_by, 
        pd_ids                 = pd_ids, 
        pu_ids                 = pu_ids
    )
    #-------------------------
    #-------------------------
    # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
    # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #-----
        df = pd.read_sql_query(
            sql = CRED_sql, 
            con = conn_aws
        )
    #--------------------------------------------------
    time_group_col = 'super_time_grp'
    min_samples    = 1
    lat_col        = 'avg_latitude'
    lon_col        = 'avg_longitude'
    spat_group_col = 'spatial_grp'
    fnl_group_col  = 'super_time_spatial_grp'
    
    df = set_spatial_groups(
        df             = df, 
        time_group_col = time_group_col, 
        eps_km         = eps_km, 
        min_samples    = min_samples, 
        lat_col        = lat_col, 
        lon_col        = lon_col, 
        spat_group_col = spat_group_col, 
        fnl_group_col  = fnl_group_col
    )
    #--------------------------------------------------
    # Current process only really makes sense if date_0==date_1
    #--------------------------------------------------
    # NOTE: In SQL queries, the dates in the range are INCLUSIVE
    #       ==> for 48 hours, use pd.Timedelta('1D')
    #       ==> for 1 week,   use pd.Timedelta('6D')
    #       ==> for 30 days,  use pd.Timedelta('29D')
    #       ==> for 180 days, use pd.Timedelta('179D')
    #       ==> for 1 year,   use pd.Timedelta('364D')
    #--------------------------------------------------
    avg_moms_cols    = None
    include_avg_moms = date_0==date_1
    if include_avg_moms:
        avg_moms_cols = []
        time_deltas = ['1D', '6D', '29D', '179D', '364D']
        #-------------------------
        avg_moms_date_1 = date_1
        for td_i in time_deltas:
            avg_moms_date_0 = (pd.to_datetime(date_1)-pd.Timedelta(td_i)).strftime('%Y-%m-%d')
            #-------------------------
            sql_avg_moms_i = build_avg_momentaries_sql(
                date_0                 = avg_moms_date_0, 
                date_1                 = avg_moms_date_1, 
                opco                   = opco, 
                td_sec_seqntl_pu_or_pd = td_sec_seqntl_pu_or_pd, 
                group_by               = group_by, 
                pd_ids                 = pd_ids, 
                pu_ids                 = pu_ids
            )
            #-----
            #-------------------------
            # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
            # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #-----
                df_avg_moms_i = pd.read_sql_query(sql_avg_moms_i, conn_aws)
            #-------------------------
            pd_n_days_i   = (pd.to_datetime(avg_moms_date_1)-pd.to_datetime(avg_moms_date_0)+pd.Timedelta('1D')).days
            avg_PN_col_i  = f'{pd_n_days_i}d'
            if pd_n_days_i==365:
                df_avg_moms_i = df_avg_moms_i.drop(columns=['avg_daily_n_mom_per_SN', 'nSNs_w_gt13_mom']).rename(columns={'avg_daily_n_mom_per_PN':avg_PN_col_i})
                df_avg_moms_i = df_avg_moms_i[group_by + [avg_PN_col_i, 'nPNs_w_gt13_mom']]
                #-----
                avg_moms_cols.extend([avg_PN_col_i, 'nPNs_w_gt13_mom'])
            else:
                df_avg_moms_i = df_avg_moms_i.drop(columns=['avg_daily_n_mom_per_SN']).rename(columns={'avg_daily_n_mom_per_PN':avg_PN_col_i})
                #-----
                avg_moms_cols.append(avg_PN_col_i)
            #-------------------------
            df = pd.merge(
                df, 
                df_avg_moms_i, 
                how      = 'left', 
                left_on  = group_by, 
                right_on = group_by
            )
            df[avg_PN_col_i] = df[avg_PN_col_i].fillna(0).round(4)
    #--------------------------------------------------
    df['date'] = df['min_outage_start'].dt.date
    df['super_time_spatial_grp_fnl'] = df['super_min_outage_start'].dt.strftime("%Y%m%d_%H:%M:%S") + '_' + df['spatial_grp'].astype(str)
    df = df.set_index(['date', 'super_time_spatial_grp_fnl']+group_by)
    #-------------------------
    if by_premise:
        fnl_cols_general = 'unique_PNs'
    else:
        fnl_cols_general = 'unique_SNs'
    #-------------------------
    if normalize:
        fnl_cols_general = 'norm_'+fnl_cols_general
    #-------------------------
    pct_title_appdx  = f"({'%' if normalize else '#'} {'Premises' if by_premise else 'Serial Numbers'})"
    #-------------------------
    fnl_cols = [
        '', 
        #-----
        '_A_1', 
        '_B_1', 
        '_C_1', 
        #-----
        '_A_2', 
        '_B_2', 
        '_C_2', 
        #-----
        '_A_0', 
        '_B_0', 
        '_C_0', 
        #-----
        '_A_3', 
        '_B_3', 
        '_C_3', 
    ]
    #-------------------------
    fnl_cols = [fnl_cols_general+x for x in fnl_cols]
    #-------------------------
    # Include total number of momentary events
    fnl_cols = [fnl_cols[0]] + ['total_PN_events_1'] + fnl_cols[1:]
    #-------------------------
    if include_avg_moms:
        fnl_cols.extend(avg_moms_cols)
    #-------------------------
    fnl_cols.extend([
        'min_outage_start',
        'max_outage_start',
        'min_duration',
        'max_duration',
        'avg_duration',
        'std_duration',
    ])
    #-------------------------
    if normalize:
        fnl_cols = [f"group_{'PN' if by_premise else 'SN'}_cnt"] + fnl_cols
    #-------------------------
    df_fnl = df[fnl_cols].copy()

    #--------------------------------------------------
    # Round percentage columns to nearest int
    # NOTE: We want anything between 0 and 1 to be set as 1 (to avoid confusion where large max_duration but seemingly no sustained events)
    non_round_cols = [
        f"group_{'PN' if by_premise else 'SN'}_cnt", 
        'total_PN_events_1', 
        'min_outage_start',
        'max_outage_start',
        'min_duration',
        'max_duration',
        'avg_duration',
        'std_duration',    
    ]
    if include_avg_moms:
        non_round_cols.extend(avg_moms_cols)
    round_cols = list(set(df_fnl.columns).difference(non_round_cols))
    # df_fnl[round_cols] = df_fnl[round_cols].fillna(0).round(0).astype(int)
    #-------------------------
    # We want anything between 0 and 1 to be set as 1 (to avoid confusion where large max_duration but seemingly no sustained events)
    # 1. Identify values to be set as 1 by setting to -1 (everything else should be >=0)
    for col_i in round_cols:
        df_fnl.loc[(df_fnl[col_i]>0) & (df_fnl[col_i]<1), col_i] = -1
    # 2. Round like usual
    df_fnl[round_cols] = df_fnl[round_cols].fillna(0).round(0).astype(int)
    # 3. Set values for identified to 1
    for col_i in round_cols:
        df_fnl.loc[df_fnl[col_i]==-1, col_i] = 1

    #--------------------------------------------------
    # Sort by date and total_PN_events_1 in groups
    df_fnl = sort_CRED_df(
        df            = df_fnl, 
        group_by      = group_by, 
        date_idx      = 'date', 
        event_tag_idx = 'super_time_spatial_grp_fnl', 
        n_moms_col    = 'total_PN_events_1', 
    )

    #--------------------------------------------------
    df_fnl_1 = df_fnl.copy()
    #-----
    fnl_cols_MI_1 = [
        (f"{'Premises' if by_premise else 'Serial Numbers'}", f"Effected ({'%' if normalize else '#'})"), 
        #-----
        ('Mom.', 'Events'), 
        #-----
        ('Mom. By Phase ' + pct_title_appdx, 'Am'), 
        ('Mom. By Phase ' + pct_title_appdx, 'Bm'), 
        ('Mom. By Phase ' + pct_title_appdx, 'Cm'), 
        #-----
        ('Sust. By Phase ' + pct_title_appdx, 'As'), 
        ('Sust. By Phase ' + pct_title_appdx, 'Bs'), 
        ('Sust. By Phase ' + pct_title_appdx, 'Cs'), 
        #-----
        ('<8s By Phase ' + pct_title_appdx, 'A-'), 
        ('<8s By Phase ' + pct_title_appdx, 'B-'), 
        ('<8s By Phase ' + pct_title_appdx, 'C-'), 
        #-----
        ('Unrslvd By Phase ' + pct_title_appdx, 'Ax'), 
        ('Unrslvd By Phase ' + pct_title_appdx, 'Bx'), 
        ('Unrslvd By Phase ' + pct_title_appdx, 'Cx')
    ]
    #-----
    if include_avg_moms:
        fnl_cols_MI_1.extend([('Past Momentaries', x) for x in avg_moms_cols])
    #-----
    fnl_cols_MI_1.extend([
        (' ', 'min_outage_start'),
        (' ', 'max_outage_start'),
        (' ', 'min_duration'),
        (' ', 'max_duration'),
        (' ', 'avg_duration'),
        (' ', 'std_duration')
    ])
    #-----
    if normalize:
        fnl_cols_MI_1 = [(f"{'Premises' if by_premise else 'Serial Numbers'}", 'Total')] + fnl_cols_MI_1
    #-----
    df_fnl_1.columns = pd.MultiIndex.from_tuples(fnl_cols_MI_1)
    
    #--------------------------------------------------
    df_fnl_2 = df_fnl.copy()
    #-----
    fnl_cols_MI_2 = [
        (' ', f"{'PNs ' if by_premise else 'SNs '}{'(%)' if normalize else '(#)'}"), 
        #-----
        (' ', 'Mom. Evs'), 
        #-----
        ('Mom. By Phase ' + pct_title_appdx, 'Am'), 
        ('Mom. By Phase ' + pct_title_appdx, 'Bm'), 
        ('Mom. By Phase ' + pct_title_appdx, 'Cm'), 
        #-----
        ('Sust. By Phase ' + pct_title_appdx, 'As'), 
        ('Sust. By Phase ' + pct_title_appdx, 'Bs'), 
        ('Sust. By Phase ' + pct_title_appdx, 'Cs'), 
        #-----
        ('<8s By Phase ' + pct_title_appdx, 'A-'), 
        ('<8s By Phase ' + pct_title_appdx, 'B-'), 
        ('<8s By Phase ' + pct_title_appdx, 'C-'), 
        #-----
        ('Unrslvd By Phase ' + pct_title_appdx, 'Ax'), 
        ('Unrslvd By Phase ' + pct_title_appdx, 'Bx'), 
        ('Unrslvd By Phase ' + pct_title_appdx, 'Cx')
    ]
    #-----
    if include_avg_moms:
        fnl_cols_MI_2.extend([(' ', x) for x in avg_moms_cols])
    #-----
    fnl_cols_MI_2.extend([
        (' ', 'min_outage_start'),
        (' ', 'max_outage_start'),
        (' ', 'min_duration'),
        (' ', 'max_duration'),
        (' ', 'avg_duration'),
        (' ', 'std_duration')
    ])
    #-----
    if normalize:
        fnl_cols_MI_2 = [(' ', f"{'PNs (#)' if by_premise else 'SNs (#)'}")] + fnl_cols_MI_2
    #-----
    df_fnl_2.columns = pd.MultiIndex.from_tuples(fnl_cols_MI_2)

    #--------------------------------------------------
    pretty_idxs = {
        'date'                       : 'Date', 
        'super_time_spatial_grp_fnl' : 'Event Tag', 
        'station_nb'                 : 'Station Number', 
        'station_nm'                 : 'Station Name', 
        'circuit_nb'                 : 'Circuit Number', 
        'circuit_nm'                 : 'Circuit Name', 
        'trsf_pole_nb'               : 'Transformer Pole'
    }
    #-----
    assert(set(df_fnl_1.index.names).difference(set(pretty_idxs.keys()))==set())
    assert(set(df_fnl_2.index.names).difference(set(pretty_idxs.keys()))==set())
    pretty_idx_names = pd.Series(df_fnl_1.index.names).map(pretty_idxs).values.tolist()
    assert(df_fnl_1.index.nlevels==df_fnl_2.index.nlevels==len(pretty_idx_names))
    #-----
    df_fnl_1.index.names = pretty_idx_names
    df_fnl_2.index.names = pretty_idx_names

    #--------------------------------------------------
    return df_fnl, df_fnl_1, df_fnl_2, avg_moms_cols


#----------------------------------------------------------------------------------------------------
def save_cred(
    cred_df_v1     , 
    cred_df_v2     , 
    opco           , 
    date_0         , 
    date_1         , 
    avg_moms_cols  , 
    normalize      = True, 
    save_dir_base  = r'C:\Users\s346557\Documents\LocalData\CRED', 
    vba_file_path  = r'C:\Users\s346557\Documents\Analysis\CRED_PrettyPivot.bas', 
    vba_macro_name = "RunAll",     
    verbose        = True
):
    r"""
    """
    #----------------------------------------------------------------------------------------------------
    #--------------------------------------------------
    if save_dir_base is None:
        print("No save_dir_base supplied!  Failure immenent!")
        assert(0)
    #-----
    if not os.path.exists(save_dir_base):
        if verbose:
            print(f'save_dir_base={save_dir_base} does not exists, so creating')
        os.makedirs(save_dir_base)
    #-----
    save_dir = os.path.join(save_dir_base, opco)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    #--------------------------------------------------
    # If vba_file_path is supplied, make sure the file exists
    # If it doesn't exist, ignore it by setting equal to None
    if vba_file_path is not None:
        if not os.path.exists(vba_file_path):
            if verbose:
                print(f'vba_file_path={vba_file_path} supplied but does not exists, so ignoring')
            vba_file_path = None
    #-------------------------
    # If vba_file_path is not None at this point, the file exists
    # Check if vba_macro_name found in file.
    # If it's not found, set vba_macro_name equal to None
    if vba_file_path is not None:
        macro_name_found = False
        with open(vba_file_path) as f:
            if f'Sub {vba_macro_name}' in f.read():
                macro_name_found = True
        if not macro_name_found:
            if verbose:
                print(f'vba_macro_name={vba_macro_name} not found in vba_file_path={vba_file_path}, so ignoring')
            vba_macro_name = None

    #--------------------------------------------------
    save_name, save_path = save_CRED_xlsx(
        df_fnl          = [cred_df_v1, cred_df_v2], 
        save_dir        = save_dir, 
        opco            = opco, 
        date_0          = date_0, 
        date_1          = date_1, 
        normalize       = normalize, 
        n_avg_moms_cols = 0 if avg_moms_cols is None else len(avg_moms_cols), 
        reset_index     = 'both'
    )
    #--------------------------------------------------
    if (
        vba_file_path  is not None and 
        vba_macro_name is not None
    ):
        assert(cred_df_v1.index.names==cred_df_v2.index.names)
        #-----
        start = time.time()
        #-----
        import_and_run_vba_macro(
            excel_file_path = save_path, 
            vba_file_path   = vba_file_path, 
            vba_macro_name  = vba_macro_name, 
            pretty_group_by = list(cred_df_v1.index.names)[2:], 
            addtnl_cols     = avg_moms_cols
        )
        #-----
        vba_time = time.time()-start
        if verbose:
            print(f'vba_time = {vba_time}')
    #--------------------------------------------------
    return save_name, save_path

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    opco          = 'pso'
    min_pct_SN    = 0.10
    eps_km        = 30
    save_results  = True
    reset_index   = 'both'
    save_dir_base = r'C:\Users\s346557\Documents\LocalData\CRED'
    #--------------------------------------------------
    if len(sys.argv) > 1:
        opco = sys.argv[1]
    if len(sys.argv) > 2:
        min_pct_SN = sys.argv[2]
    if len(sys.argv) > 3:
        eps_km = sys.argv[3]
    if len(sys.argv) > 4:
        save_results = sys.argv[4]
    if len(sys.argv) > 5:
        reset_index = sys.argv[5]
    if len(sys.argv) > 6:
        save_dir_base = sys.argv[6]
    #--------------------------------------------------
    # print(f'opco          = {opco}')
    # print(f'min_pct_SN    = {min_pct_SN}')
    # print(f'eps_km        = {eps_km}')
    # print(f'save_results  = {save_results}')
    # print(f'reset_index   = {reset_index}')
    # print(f'save_dir_base = {save_dir_base}')
    #--------------------------------------------------
    by_premise = True
    normalize  = True
    #-----
    vba_file_path  = r'C:\Users\s346557\Documents\Analysis\CRED_PrettyPivot.bas'
    vba_macro_name = "RunAll"
    #--------------------------------------------------
    date_0                 = (datetime.datetime.now()-datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    date_1                 = date_0
    td_sec_group_daisy     = 5
    td_sec_seqntl_pu_or_pd = 5
    td_sec_final_daisy     = 5
    group_by               = ['station_nm', 'circuit_nm']
    pd_ids                 = ['3.26.0.47']
    pu_ids                 = ['3.26.0.216']
    #--------------------------------------------------
    cred_df_raw, cred_df_v1, cred_df_v2, avg_moms_cols = run_cred(
        date_0                 = date_0, 
        date_1                 = date_1, 
        opco                   = opco, 
        min_pct_SN             = min_pct_SN, 
        by_premise             = by_premise, 
        normalize              = normalize, 
        td_sec_group_daisy     = td_sec_group_daisy, 
        td_sec_seqntl_pu_or_pd = td_sec_seqntl_pu_or_pd, 
        td_sec_final_daisy     = td_sec_final_daisy, 
        eps_km                 = eps_km, 
        group_by               = group_by, 
        pd_ids                 = pd_ids, 
        pu_ids                 = pu_ids, 
        conn_aws               = None
    )

    #--------------------------------------------------
    save_name, save_path = save_cred(
        cred_df_v1     = cred_df_v1, 
        cred_df_v2     = cred_df_v2, 
        opco           = opco, 
        date_0         = date_0, 
        date_1         = date_1, 
        avg_moms_cols  = avg_moms_cols, 
        normalize      = normalize, 
        save_dir_base  = save_dir_base, 
        vba_file_path  = vba_file_path, 
        vba_macro_name = vba_macro_name,     
        verbose        = True
    )

    #--------------------------------------------------
    outlook = win32.Dispatch('outlook.application')
    #-----
    mail = outlook.CreateItem(0)
    mail.To = 'basirman@aep.com; zmread@aep.com; septomey@aep.com'
    mail.CC = 'jbuxton@aep.com'
    mail.Subject = save_name
    mail.Body = """The CRED results for the date {} are attached to this email.  Please let me know if you have any questions.

    Cheers,
    Jesse
    """.format(date_0)
    #-----
    # To attach a file to the email (optional):
    attachment  = save_path
    mail.Attachments.Add(attachment)
    #-----
    mail.Send()
    #--------------------------------------------------
    if not save_results:
        os.remove(save_path)