#!/usr/bin/env python

r"""
Holds DOVSOutages_SQL class.  See DOVSOutages_SQL.DOVSOutages_SQL for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys
import re
import copy

from natsort import natsorted
#--------------------------------------------------
from AMI_SQL import AMI_SQL
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from SQLSelect import SQLSelectElement, SQLSelect
from SQLFrom import SQLFrom
from SQLWhere import SQLWhereElement, SQLWhere
from SQLJoin import SQLJoin, SQLJoinCollection
from SQLGroupBy import SQLGroupBy
from SQLQuery import SQLQuery
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
#--------------------------------------------------

class DOVSOutages_SQL:
    r"""
    This class is intended to build SQL queries to fetch data from the tables within DOVSADM.
    """
    def __init__(self):
        self.sql_query=None 
        #NOTE: These are only the columns for DOVSADM.DOVS_OUTAGE_FACT
    
    #****************************************************************************************************
    @staticmethod
    def alias_found_in_cols_of_interest(
        alias, 
        cols_of_interest
    ):
        r"""
        Search for alias in cols_of_interest.  Return True if found, False if not.
        For simple string columns in cols_of_interest, the alias is assumed to be to str
          i.e., assume 
                cols_of_interest = [
                    'CI_NB',
                    {'field_desc': 'DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)',
                     'alias': 'DT_OFF_TS_FULL',
                     'table_alias_prefix': None}
                ]
            the alias of the [0] element is 'CI_NB', whereas the alias of the [1] element is 'DT_OFF_TS_FULL'
        """
        return AMI_SQL.alias_found_in_cols_of_interest(alias=alias, cols_of_interest=cols_of_interest)
     
    #--------------------------------------------------
    @staticmethod
    def get_std_cols_of_interest(
        **kwargs
    ):
        r"""
        """
        #-------------------------
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'DOV')
        kwargs['datetime_col']     = kwargs.get('datetime_col', 'DT_OFF_TS_FULL')
        #-------------------------
        cols_of_interest=[
            'CI_NB', 'CMI_NB', 'OUTG_REC_NB', 'OUTAGE_NB', 'DT_ON_TS', 'DT_OFF_TS', 
            dict(field_desc=('DT_ON_TS - STEP_DRTN_NB/(60*24)' if kwargs['from_table_alias'] is None else 
                             f"{kwargs['from_table_alias']}.DT_ON_TS - {kwargs['from_table_alias']}.STEP_DRTN_NB/(60*24)"), 
                 alias=kwargs['datetime_col'], table_alias_prefix=None), 
            'STEP_DRTN_NB', 
            dict(field_desc=('EXTRACT(YEAR FROM DT_OFF_TS)' if kwargs['from_table_alias'] is None else 
                             f"EXTRACT(YEAR FROM {kwargs['from_table_alias']}.DT_OFF_TS)"), 
                 alias='START_YEAR', table_alias_prefix=None), 
            'OPERATING_UNIT_ID', 'STATE_ABBR_TX', 'MJR_CAUSE_CD', 'MNR_CAUSE_CD', 'LOCATION_ID', 'GIS_CRCT_NB'
        ]
        #-------------------------
        return cols_of_interest
        
    @staticmethod
    def add_dt_off_ts_full_to_cols_of_interest(
        cols_of_interest, 
        alias              = 'DT_OFF_TS_FULL', 
        from_table_alias   = None, 
        table_alias_prefix = None, 
        inplace            = False
    ):
        r"""
        """
        #-------------------------
        if not inplace:
            cols_of_interest = copy.deepcopy(cols_of_interest)
        #-------------------------
        cols_of_interest.append(
            dict(
                field_desc=('DT_ON_TS - STEP_DRTN_NB/(60*24)' if from_table_alias is None else 
                             f"{from_table_alias}.DT_ON_TS - {from_table_alias}.STEP_DRTN_NB/(60*24)"), 
                alias=alias, 
                table_alias_prefix=table_alias_prefix
            )
        )
        #-------------------------
        return cols_of_interest
        
        
    @staticmethod
    def build_sql_outage(
        cols_of_interest        = None, 
        addtnl_cols_of_interest = None, 
        **kwargs
    ):
        r"""
        Typically, one should leave cols_of_interest set as None.
        If one would like to add to the default columns of interest (see declaration in code below), one should
          use addtnl_cols_of_interest.
        
        Acceptable kwargs:
          --------------------------------------------------
          FOR USE WITH AMI_SQL.add_ami_where_statements
          --------------------------------------------------
          See AMI_SQL.add_ami_where_statements for my updated list of acceptable kwargs with respect to the where statement.
          NOTE: Not all kwargs possible for AMI_SQL.add_ami_where_statements make sense here. For example, serialnumbers won't do
                anything here, as it is not possible to search serial number in DOVS (although, premise numbers are possible)
        
          - date_range
            - tuple with two string elements, e.g., ['2021-01-01', '2021-04-01']
          - date_col
            - default: DT_OFF_TS
            
          - datetime_range
            - tuple with two string elements, e.g., ['2021-01-01 00:00:00', '2021-04-01 12:00:00']
          - datetime_col
            - default: DT_OFF_TS_FULL
            - NOTE: DT_OFF_TS_FULL is built from DT_ON_TS and STEP_DRTN_NB, but IS NOT a default column in DOVS_OUTAGE_FACT.
                    DT_OFF_TS_FULL is built by default in the code below.  
                    However, if user supplies own cols_of_interest and does not include DT_OFF_TS_FULL, probably DT_ON_TS
                    should be used as datetime_col instead.
          - datetime_pattern
            - Regex pattern used to convert string to a form that SQL/Athena/Oracle can convert to a TIMESTAMP
            - default: None

          - premise_nb(s) (or, aep_premise_nb(s) will work also)
          - premise_nb_col
            - default: PREMISE_NB
            - NOTE: If premise_nbs are included, kwargs['include_DOVS_PREMISE_DIM'] will be set to True

          - opco(s) (or, aep_opco(s) will work also)
            - NOT REALLY SUGGESTED TO USE, as DOVS does not contain such a field.
              However, OPERATING_UNIT_ID(s) or opco_id(s) can be used.
              If opco are supplied by user, they are converted to opco_ids using opco_to_id dictionary in the code
              
          - OPERATING_UNIT_ID(s) (or, opco_id(s) will work also)
          - opco_id_col
            - default: OPERATING_UNIT_ID

          - state(s)
          - state_col
            - default: STATE_ABBR_TX
            - NOTE: If state(s) are included, kwargs['include_DOVS_MASTER_GEO_DIM'] will be set to True

          - city(ies)
          - city_col
            - default: CITY_NM

          - location_ids
          - location_id_col
            - default: LOCATION_ID

          - circuit_nbs
          - circuit_nb_col
            - default: GIS_CRCT_NB


          - groupby_cols
          - agg_cols_and_types
          - include_counts_including_null

          - schema_name
            - default: DOVSADM
          - table_name
            - default: DOVS_OUTAGE_FACT
          - from_table_alias
            - default: DOV

          - alias (query alias)

          - return_args

          - join_mp_args
      
          --------------------------------------------------
          OTHER KWARGS
          --------------------------------------------------
          - INTRPTN_TYP_CD
            - default: = 'S'
            - NOTE: If INTRPTN_TYP_CD is included, kwargs['include_DOVS_OUTAGE_ATTRIBUTES_DIM'] will be set to True
          - INTRPTN_TYP_CD_col: 
            - default: INTRPTN_TYP_CD
            
          - CURR_REC_STAT_CD
            - default: = 'A'
            - NOTE: If CURR_REC_STAT_CD is included, kwargs['include_DOVS_OUTAGE_ATTRIBUTES_DIM'] will be set to True
          - CURR_REC_STAT_CD_col: 
            - default: CURR_REC_STAT_CD
            
          - MJR_CAUSE_CD
            - default: <> 'NI'
          - MJR_CAUSE_CD_col: 
            - default: MJR_CAUSE_CD
            
          - DEVICE_CD
            - default: <> 85
          - DEVICE_CD_col: 
            - default: DEVICE_CD
            
          - STATE_ABBR_TX
            - default: None
            - NOTE: One should only set state(s) OR STATE_ABBR_TX, NOT BOTH!
          - STATE_ABBR_TX_col: 
            - default: kwargs['state_col']
            
          - CI_NB_min
            - default: None
          - CI_NB_max
            - default: None
          - CI_NB_col
            - default: CI_NB
            
          - CMI_NB_min
            - default: None
          - CMI_NB_max
            - default: None
          - CMI_NB_col
            - default: CMI_NB
            
          - CI_NB_min
          
          ----------
          NOTE: dt_off_ts_full and dt_on_ts are a little different, as one typically does not set these using
                  equality, but instead >(=) or <(=)
                The functionality here is so one can specifically pick outages which began before/after some datetime
                  and ended before/after some other datetime.
                If one only need, e.g., dt_off_ts_full within some range, then simply use the datetime_range argument!
          ----------
          - dt_off_ts_full
            - default: None
            - If not None, must be a dict with keys=['value', 'comparison_operator']
          - dt_off_ts_full_col
            - default: DT_OFF_TS_FULL
            
          - dt_on_ts
            - default: None
            - If not None, must be a dict with keys=['value', 'comparison_operator']
          - dt_on_ts_col
            - default: DT_ON_TS
          
          --------------------------------------------------
          FOR JOINING WITH OTHER DOVS TABLES
          --------------------------------------------------
          - As of now, there are six possible other tables which can be joined to DOVS_OUTAGE_FACT table.
            These are TABLE_NAMEs=:
              - DOVS_MASTER_GEO_DIM
              - DOVS_OUTAGE_ATTRIBUTES_DIM
              - DOVS_CLEARING_DEVICE_DIM
              - DOVS_EQUIPMENT_TYPES_DIM
              - DOVS_OUTAGE_CAUSE_TYPES_DIM
              - DOVS_PREMISE_DIM
              
          - For each TABLE_NAME above, there are four possible kwargs:
            - include_TABLE_NAME 
              - .e.g, include_DOVS_MASTER_GEO_DIM
              - Boolean, whether or not to join to table in the SQL query
            - alias_TABLE_NAME 
              - e.g., alias_DOVS_MASTER_GEO_DIM
              - string, an alias for TABLE_NAME
            - cols_to_join_TABLE_NAME 
              - e.g., cols_to_join_DOVS_MASTER_GEO_DIM
              - a list which dictates which columns to be joined between DOVS_OUTAGE_FACT and TABLE_NAME
              - See SQLJoin class for more information
            - select_cols_TABLE_NAME 
              - e.g., select_cols_DOVS_MASTER_GEO_DIM
              - Columns from TABLE_NAME which the user would like to also be added to the SELECT statement.
            
        """
        #--------------------------------------------------
        # Set up initial SQL Query using AMI_SQL.build_sql_ami
        #--------------------------------------------------
        # Some normal kwargs for AMI_SQL.build_sql_ami
        kwargs['schema_name']      = kwargs.get('schema_name', 'DOVSADM')
        kwargs['table_name']       = kwargs.get('table_name', 'DOVS_OUTAGE_FACT')
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'DOV')

        kwargs['date_col']         = kwargs.get('date_col', 'DT_OFF_TS')
        kwargs['datetime_col']     = kwargs.get('datetime_col', 'DT_OFF_TS_FULL')
        kwargs['datetime_pattern'] = kwargs.get('datetime_pattern', None)
        
        kwargs['premise_nb_col']   = kwargs.get('premise_nb_col', 'PREMISE_NB')
        kwargs['state_col']        = kwargs.get('state_col', 'STATE_ABBR_TX')
        
        # Make sure only one of states, state, STATE_ABBR_TX found in kwargs
        assert(sum([x in kwargs for x in ['states', 'state', 'STATE_ABBR_TX']])<=1)
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest=DOVSOutages_SQL.get_std_cols_of_interest(**kwargs)
            # NOTE: Cannot use alias names in WHERE called in SQL (e.g., cannot call, 'WHERE DOV_OFF_TS_FULL = ...')
            #         Therefore, must instead call, e.g., 'WHERE DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24) = ...'
            #       Also, AMI_SQL.add_ami_where_statements will add the from_table_alias to the front (if it is not None), therefore
            #         here, one wants DT_ON_TS - DOV.STEP_DRTN_NB/(60*24) (as, if instead one had DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24), 
            #         the functionality in AMI_SQL.add_ami_where_statements would change this to DOV.DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)
            kwargs['datetime_col'] = ('DT_ON_TS - STEP_DRTN_NB/(60*24)' if kwargs['from_table_alias'] is None else 
                                     f"{kwargs['from_table_alias']}.DT_ON_TS - {kwargs['from_table_alias']}.STEP_DRTN_NB/(60*24)")
        if addtnl_cols_of_interest is not None:
            cols_of_interest.extend(addtnl_cols_of_interest)
        #--------------------------------------------------
        # Need to handle premise numbers separately, since these won't come from DOVSADM.DOVS_OUTAGE_FACT (DOV), but will instead come from
        # DOVSADM.DOVS_PREMISE_DIM (PRIM)
        #--------------------------------------------------
        # If (non-None) premise_nbs found in kwargs, kwargs['include_DOVS_PREMISE_DIM'] will be set to True
        kwargs['premise_nb_col']    = kwargs.get('premise_nb_col', 'PREMISE_NB')
        possible_premise_nbs_kwargs = ['premise_nbs', 'premise_nb', 'aep_premise_nbs', 'aep_premise_nb']
        found_premise_nbs_kwargs    = [x for x in kwargs if x in possible_premise_nbs_kwargs]
        assert(len(found_premise_nbs_kwargs)<=1)
        if found_premise_nbs_kwargs:
            # Need to remove premise_nbs from kwargs, otherwise AMI_SQL.build_sql_ami will incorrectly
            #   add premise numbers to SQL query with DOV alias (i.e., it will input 'DOV.PREMISE_NB =...'
            #   when it should actually be 'PRIM.PREMISE_NB =...')
            # These will be added explicitly after AMI_SQL.build_sql_ami
            premise_nbs = kwargs.pop(found_premise_nbs_kwargs[0])
        else:
            premise_nbs = None
        #-------------------------
        if premise_nbs is not None:
            kwargs['include_DOVS_PREMISE_DIM']     = True
            kwargs['select_cols_DOVS_PREMISE_DIM'] = kwargs.get('select_cols_DOVS_PREMISE_DIM', [])
            if kwargs['premise_nb_col'] not in kwargs['select_cols_DOVS_PREMISE_DIM']:
                kwargs['select_cols_DOVS_PREMISE_DIM'].append(kwargs['premise_nb_col']) 

        #-------------------------
        # Note: Need to pop cities/city so not fed into AMI_SQL.build_sql_ami
        kwargs['city_col'] = kwargs.get('city_col', 'CITY_NM')
        cities             = kwargs.pop('cities', kwargs.pop('city', None))
        if cities is not None:
            kwargs['include_DOVS_PREMISE_DIM']     = True
            kwargs['select_cols_DOVS_PREMISE_DIM'] = kwargs.get('select_cols_DOVS_PREMISE_DIM', [])
            if kwargs['city_col'] not in kwargs['select_cols_DOVS_PREMISE_DIM']:
                kwargs['select_cols_DOVS_PREMISE_DIM'].append(kwargs['city_col']) 
            
        #--------------------------------------------------
        # opco name (e.g., 'oh', 'pso', etc.) cannot be used in DOVS, as that field does not exist.
        # However, OPERATING_UNIT_ID can be used.
        # If opco(s) found in kwargs, convert to OPERATING_UNIT_ID and insert into where statements later
        possible_opco_kwargs = ['opco', 'aep_opco', 'opcos', 'aep_opcos']
        found_opco_kwargs    = [x for x in kwargs if x in possible_opco_kwargs]
        assert(len(found_opco_kwargs)<=1)
        #-----
        opco_id_col             = kwargs.get('opco_id_col', 'OPERATING_UNIT_ID')
        possible_opco_id_kwargs = ['opco_id', 'opco_ids', 'OPERATING_UNIT_ID', 'OPERATING_UNIT_IDs']
        found_opco_id_kwargs    = [x for x in kwargs if x in possible_opco_id_kwargs]
        assert(len(found_opco_id_kwargs)<=1)
        if found_opco_id_kwargs:
            opco_ids = kwargs[found_opco_id_kwargs[0]]
        else:
            opco_ids = None        
        #-----
        if found_opco_kwargs and kwargs[found_opco_kwargs[0]] is not None:
            print('\nDOVSOutages_SQL: Found opco in kwargs\nConverting to OPERATING_UNIT_ID\n')
            # Note: Need to pop opco so it is not fed into AMI_SQL.build_sql_ami
            opcos = kwargs.pop(found_opco_kwargs[0])
            opco_to_id = dict(
                ap  = 1, 
                ky  = 2, 
                oh  = 3, 
                im  = 4, 
                pso = 5, 
                swp = 6, 
                tx  = 7
            )
            assert(Utilities.is_object_one_of_types(opcos, [str, list]))
            if isinstance(opcos, str):
                opcos=[opcos]
            opco_ids_cnvrtd = []
            for opco_i in opcos:
                opco_id_i = opco_to_id[opco_i.lower()]
                opco_ids_cnvrtd.append(opco_id_i)
            print(f'Input opcos = {opcos}')
            print(f'OPERATING_UNIT_IDs used = {opco_ids_cnvrtd}')
            kwargs['opcos_handled'] = True
            if opco_ids is not None:
                assert(Utilities.is_object_one_of_types(opco_ids, [int, str, list]))
                if Utilities.is_object_one_of_types(opco_ids, [int, str]):
                    opco_ids = [int(opco_ids)]
                opco_ids = [int(x) for x in opco_ids]
                #-----
                if natsorted(opco_ids)!=natsorted(opco_ids_cnvrtd):
                    print(f"User input opcos={opcos} does not agree with user input OPERATING_UNIT_ID={opco_ids}")
                    print('CRASH IMMINENT!!!!!')
                assert(natsorted(opco_ids)==natsorted(opco_ids_cnvrtd))
            else:
                opco_ids=opco_ids_cnvrtd
            
        #**************************************************
        sql = AMI_SQL.build_sql_ami(cols_of_interest=cols_of_interest, **kwargs)
        #**************************************************
        
        #--------------------------------------------------
        # Other WHERE statements not handled by AMI_SQL.build_sql_ami
        #--------------------------------------------------
        #-----
        # Cotinuation of opco/opco_ids
        if opco_ids is not None:
            sql.sql_where = SQLWhere.add_where_statement_equality_or_in(
                sql_where          = sql.sql_where, 
                field_desc         = opco_id_col, 
                value              = opco_ids, 
                needs_quotes       = False, 
                table_alias_prefix = kwargs['from_table_alias'], 
                idx                = None
            ) 
        #-----
        if premise_nbs is not None:
            sql.sql_where = SQLWhere.add_where_statement_equality_or_in(
                sql_where          = sql.sql_where, 
                field_desc         = kwargs['premise_nb_col'], 
                value              = premise_nbs, 
                table_alias_prefix = kwargs.get('alias_DOVS_PREMISE_DIM', 'PRIM')
            )
        #-----
        if cities is not None:
            sql.sql_where = SQLWhere.add_where_statement_equality_or_in(
                sql_where          = sql.sql_where, 
                field_desc         = kwargs['city_col'], 
                value              = cities, 
                table_alias_prefix = kwargs.get('alias_DOVS_PREMISE_DIM', 'PRIM')
            )
        #-----
        specific_dates = kwargs.get('specific_dates', None)
        if specific_dates is not None:
            sql.sql_where.add_where_statement(
                field_desc          = kwargs['date_col'], 
                table_alias_prefix  = kwargs['from_table_alias'], 
                comparison_operator = 'IN', 
                value               = f'({Utilities_sql.join_list_w_quotes(specific_dates)})', 
                needs_quotes        = False, 
                idx                 = None
            )
        #-----
        outg_rec_nbs    = kwargs.get('outg_rec_nbs', None)
        outg_rec_nb_col = kwargs.get('outg_rec_nb_col', 'OUTG_REC_NB')
        if outg_rec_nbs is not None:
            if isinstance(outg_rec_nbs, SQLWhereElement):
                sql.sql_where.add_where_statement(outg_rec_nbs)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = outg_rec_nb_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = 'IN', 
                    value               = f'({Utilities_sql.join_list_w_quotes(outg_rec_nbs)})', 
                    needs_quotes        = False
                )
        #-----
        outage_nbs    = kwargs.get('outage_nbs', None)
        outage_nb_col = kwargs.get('outage_nb_col', 'OUTAGE_NB')
        if outage_nbs is not None:
            if isinstance(outage_nbs, SQLWhereElement):
                sql.sql_where.add_where_statement(outage_nbs)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = outage_nb_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = 'IN', 
                    value               = f'({Utilities_sql.join_list_w_quotes(outage_nbs)})', 
                    needs_quotes        = False
                )
        #-----
        location_ids    = kwargs.get('location_ids', None)
        location_id_col = kwargs.get('location_id_col', 'LOCATION_ID')
        if location_ids is not None:
            if isinstance(location_ids, SQLWhereElement):
                sql.sql_where.add_where_statement(location_ids)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = location_id_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = 'IN', 
                    value               = f'({Utilities_sql.join_list_w_quotes(location_ids)})', 
                    needs_quotes        = False
                )
        #-----
        circuit_nbs    = kwargs.get('circuit_nbs', None)
        circuit_nb_col = kwargs.get('circuit_nb_col', 'GIS_CRCT_NB')
        if circuit_nbs is not None:
            if isinstance(circuit_nbs, SQLWhereElement):
                sql.sql_where.add_where_statement(circuit_nbs)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = circuit_nb_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = 'IN', 
                    value               = f'({Utilities_sql.join_list_w_quotes(circuit_nbs)})', 
                    needs_quotes        = False
                )
        #-----
        MJR_CAUSE_CD_col = kwargs.get('MJR_CAUSE_CD_col', 'MJR_CAUSE_CD')
        MNR_CAUSE_CD_col = kwargs.get('MNR_CAUSE_CD_col', 'MNR_CAUSE_CD')
        MJR_CAUSE_CD     = kwargs.get('MJR_CAUSE_CD', None)
        MNR_CAUSE_CD     = kwargs.get('MNR_CAUSE_CD', None)
        mjr_mnr_cause = kwargs.get('mjr_mnr_cause', None)
        if mjr_mnr_cause is not None:
            sql.sql_where.add_where_statements([dict(field_desc=MJR_CAUSE_CD_col, table_alias_prefix=kwargs['from_table_alias'], comparison_operator='=', value=mjr_mnr_cause[0]), 
                                                dict(field_desc=MNR_CAUSE_CD_col, table_alias_prefix=kwargs['from_table_alias'], comparison_operator='=', value=mjr_mnr_cause[1])], 
                                               idxs=None, run_check=True)
        if MJR_CAUSE_CD is not None:
            assert(Utilities.is_object_one_of_types(MJR_CAUSE_CD, [dict, SQLWhere]))
            if isinstance(MJR_CAUSE_CD, SQLWhere):
                sql.sql_where.add_where_statement(MJR_CAUSE_CD, idx=None, run_check=True)
            else:
                sql.sql_where.add_where_statement(**MJR_CAUSE_CD, idx=None, run_check=True)
        if MNR_CAUSE_CD is not None:
            assert(Utilities.is_object_one_of_types(MNR_CAUSE_CD, [dict, SQLWhere]))
            if isinstance(MNR_CAUSE_CD, SQLWhere):
                sql.sql_where.add_where_statement(MNR_CAUSE_CD, idx=None, run_check=True)
            else:
                sql.sql_where.add_where_statement(**MNR_CAUSE_CD, idx=None, run_check=True)                                                     
        #-----                                          
        DEVICE_CD_col = kwargs.get('DEVICE_CD_col', 'DEVICE_CD')
        DEVICE_CD     = kwargs.get('DEVICE_CD', None)
        if DEVICE_CD is not None:
            assert(Utilities.is_object_one_of_types(DEVICE_CD, [dict, SQLWhere]))
            if isinstance(DEVICE_CD, SQLWhere):
                sql.sql_where.add_where_statement(DEVICE_CD, idx=None, run_check=True)
            else:
                sql.sql_where.add_where_statement(**DEVICE_CD, idx=None, run_check=True)
        #-----
        CI_NB_col = kwargs.get('CI_NB_col', 'CI_NB')
        CI_NB_min = kwargs.get('CI_NB_min', None)
        CI_NB_max = kwargs.get('CI_NB_max', None)
        if CI_NB_min is not None:
            if isinstance(CI_NB_min, SQLWhereElement):
                sql.sql_where.add_where_statement(CI_NB_min)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = CI_NB_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = '>', 
                    value               = CI_NB_min, 
                    needs_quotes        = False
                )
        if CI_NB_max is not None:
            if isinstance(CI_NB_max, SQLWhereElement):
                sql.sql_where.add_where_statement(CI_NB_max)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = CI_NB_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = '<', 
                    value               = CI_NB_max, 
                    needs_quotes        = False
                )
        #-----
        CMI_NB_col = kwargs.get('CMI_NB_col', 'CMI_NB')
        CMI_NB_min = kwargs.get('CMI_NB_min', None)
        CMI_NB_max = kwargs.get('CMI_NB_max', None)
        if CMI_NB_min is not None:
            if isinstance(CMI_NB_min, SQLWhereElement):
                sql.sql_where.add_where_statement(CMI_NB_min)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = CMI_NB_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = '>', 
                    value               = CMI_NB_min, 
                    needs_quotes        = False
                )
        if CMI_NB_max is not None:
            if isinstance(CMI_NB_max, SQLWhereElement):
                sql.sql_where.add_where_statement(CMI_NB_max)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = CMI_NB_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = '<', 
                    value               = CMI_NB_max, 
                    needs_quotes        = False
                )
        #----------
        # dt_off_ts_full and dt_on_ts
        #-----
        # dt_off_ts_full
        dt_off_ts_full     = kwargs.get('dt_off_ts_full', None)
        dt_off_ts_full_col = kwargs.get('dt_off_ts_full_col', None)
        if dt_off_ts_full is not None:
            #-----
            assert(isinstance(dt_off_ts_full, dict))
            assert(set(dt_off_ts_full.keys()).difference(set(['value', 'comparison_operator']))==set())
            #-----
            if dt_off_ts_full_col is None:
                if kwargs['from_table_alias'] is None:
                    dt_off_ts_full_col = 'DT_ON_TS - STEP_DRTN_NB/(60*24)'
                else:
                    dt_off_ts_full_col = f"{kwargs['from_table_alias']}.DT_ON_TS - {kwargs['from_table_alias']}.STEP_DRTN_NB/(60*24)"
            #-----
            if kwargs['datetime_pattern'] is None:
                dt_field_desc = f"CAST({dt_off_ts_full_col} AS TIMESTAMP)"
            else:
                dt_field_desc = r"CAST(regexp_replace({}, ".format(dt_off_ts_full_col) + r"'{}', '$1 $2') AS TIMESTAMP)".format(kwargs['datetime_pattern'])
            #-----
            # field_descs_dict['dt_off_ts_full'] = dt_field_desc
            sql.sql_where.add_where_statement(
                field_desc          = dt_field_desc, 
                comparison_operator = dt_off_ts_full['comparison_operator'], 
                value               = dt_off_ts_full['value'], 
                needs_quotes        = True, 
                table_alias_prefix  = None, 
                is_timestamp        = True, 
                idx                 = None
            )
        #-----
        # dt_on_ts
        dt_on_ts     = kwargs.get('dt_on_ts', None)
        dt_on_ts_col = kwargs.get('dt_on_ts_col', 'DT_ON_TS')
        if dt_on_ts is not None:
            assert(isinstance(dt_on_ts, dict))
            assert(set(dt_on_ts.keys()).difference(set(['value', 'comparison_operator']))==set())
            if kwargs['from_table_alias']:
                dt_on_ts_col = f"{kwargs['from_table_alias']}.{dt_on_ts_col}"
            #-----
            if kwargs['datetime_pattern'] is None:
                dt_field_desc = f"CAST({dt_on_ts_col} AS TIMESTAMP)"
            else:
                dt_field_desc = r"CAST(regexp_replace({}, ".format(dt_on_ts_col) + r"'{}', '$1 $2') AS TIMESTAMP)".format(kwargs['datetime_pattern'])
            #-----
            # field_descs_dict['dt_on_ts'] = dt_field_desc
            sql.sql_where.add_where_statement(
                field_desc          = dt_field_desc, 
                comparison_operator = dt_on_ts['comparison_operator'], 
                value               = dt_on_ts['value'], 
                needs_quotes        = True, 
                table_alias_prefix  = None, 
                is_timestamp        = True, 
                idx                 = None
            )        
        
        #--------------------------------------------------
        # Joins with other DOVS databases
        #--------------------------------------------------
        #-------------------------
        # DOVS_MASTER_GEO_DIM kwargs
        #-------------------------
        kwargs['include_DOVS_MASTER_GEO_DIM'] = kwargs.get('include_DOVS_MASTER_GEO_DIM', False)
        kwargs['alias_DOVS_MASTER_GEO_DIM']   = kwargs.get('alias_DOVS_MASTER_GEO_DIM', 'DOV1')
        kwargs['cols_to_join_DOVS_MASTER_GEO_DIM']   = kwargs.get('cols_to_join_DOVS_MASTER_GEO_DIM', 
                                                                  [
                                                                      ['OPERATING_UNIT_ID', 'OPRTG_UNT_ID'], 
                                                                      ['STATE_ABBR_TX', 'STATE_ID'], 
                                                                      ['OPCO_NBR', 'OPCO_ID'], 
                                                                      ['DISTRICT_NB', 'DISTRICT_ID'], 
                                                                      ['SRVC_CNTR_NB', 'AREA_ID'], 
                                                                      ['GIS_CRCT_NB', 'GIS_CIRCUIT_ID']
                                                                  ]
                                                                 )
        kwargs['select_cols_DOVS_MASTER_GEO_DIM'] = kwargs.get('select_cols_DOVS_MASTER_GEO_DIM', ['OPRTG_UNT_NM'])
        #-----
        # Inclusion of DOVS_MASTER_GEO_DIM
        if kwargs['include_DOVS_MASTER_GEO_DIM']:
            sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_MASTER_GEO_DIM(
                alias_DOVS_OUTAGE_FACT    = kwargs['from_table_alias'], 
                alias_DOVS_MASTER_GEO_DIM = kwargs['alias_DOVS_MASTER_GEO_DIM'], 
                list_of_columns_to_join   = kwargs['cols_to_join_DOVS_MASTER_GEO_DIM']
            )
            sql.add_join(
                sql_join                   = sql_join, 
                idx                        = None, 
                run_check                  = False, 
                join_cols_to_add_to_select = kwargs['select_cols_DOVS_MASTER_GEO_DIM']
            )

        #-------------------------
        # DOVS_OUTAGE_ATTRIBUTES_DIM kwargs
        #-------------------------
        kwargs['include_DOVS_OUTAGE_ATTRIBUTES_DIM']      = kwargs.get('include_DOVS_OUTAGE_ATTRIBUTES_DIM', False)
        kwargs['alias_DOVS_OUTAGE_ATTRIBUTES_DIM']        = kwargs.get('alias_DOVS_OUTAGE_ATTRIBUTES_DIM', 'DOV2')
        kwargs['cols_to_join_DOVS_OUTAGE_ATTRIBUTES_DIM'] = kwargs.get('cols_to_join_DOVS_OUTAGE_ATTRIBUTES_DIM', ['OUTG_REC_NB'])
        kwargs['select_cols_DOVS_OUTAGE_ATTRIBUTES_DIM']  = kwargs.get('select_cols_DOVS_OUTAGE_ATTRIBUTES_DIM', None)
        #----------
        INTRPTN_TYP_CD_col = kwargs.get('INTRPTN_TYP_CD_col', 'INTRPTN_TYP_CD')
        INTRPTN_TYP_CD     = kwargs.get('INTRPTN_TYP_CD', None)
        if INTRPTN_TYP_CD is not None:
            if isinstance(INTRPTN_TYP_CD, SQLWhereElement):
                sql.sql_where.add_where_statement(INTRPTN_TYP_CD)
            else:
                sql.sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql.sql_where, 
                    field_desc         = INTRPTN_TYP_CD_col, 
                    value              = INTRPTN_TYP_CD, 
                    needs_quotes       = True, 
                    table_alias_prefix = kwargs['alias_DOVS_OUTAGE_ATTRIBUTES_DIM'], 
                    idx                = None
                )
        #-----
        CURR_REC_STAT_CD_col = kwargs.get('CURR_REC_STAT_CD_col', 'CURR_REC_STAT_CD')
        CURR_REC_STAT_CD = kwargs.get('CURR_REC_STAT_CD', None)
        if CURR_REC_STAT_CD is not None:
            if isinstance(CURR_REC_STAT_CD, SQLWhereElement):
                sql.sql_where.add_where_statement(CURR_REC_STAT_CD)
            else:
                sql.sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql.sql_where, 
                    field_desc         = CURR_REC_STAT_CD_col, 
                    value              = CURR_REC_STAT_CD, 
                    needs_quotes       = True, 
                    table_alias_prefix = kwargs['alias_DOVS_OUTAGE_ATTRIBUTES_DIM'], 
                    idx                = None
                )    
        
        #-----
        # If INTRPTN_TYP_CD or CURR_REC_STAT_CD found in kwargs, kwargs['include_DOVS_OUTAGE_ATTRIBUTES_DIM'] will be set to True
        if any([x is not None for x in [INTRPTN_TYP_CD, CURR_REC_STAT_CD]]):
            kwargs['include_DOVS_OUTAGE_ATTRIBUTES_DIM'] = True    
        #----------
        # Inclusion of DOVS_OUTAGE_ATTRIBUTES_DIM
        if kwargs['include_DOVS_OUTAGE_ATTRIBUTES_DIM']:
            sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_ATTRIBUTES_DIM(
                alias_DOVS_OUTAGE_FACT           = kwargs['from_table_alias'], 
                alias_DOVS_OUTAGE_ATTRIBUTES_DIM = kwargs['alias_DOVS_OUTAGE_ATTRIBUTES_DIM'], 
                list_of_columns_to_join          = kwargs['cols_to_join_DOVS_OUTAGE_ATTRIBUTES_DIM']
            )
            sql.add_join(
                sql_join                   = sql_join, 
                idx                        = None, 
                run_check                  = False, 
                join_cols_to_add_to_select = kwargs['select_cols_DOVS_OUTAGE_ATTRIBUTES_DIM']
            )

        #-------------------------
        # DOVS_CLEARING_DEVICE_DIM kwargs
        #-------------------------
        kwargs['include_DOVS_CLEARING_DEVICE_DIM']      = kwargs.get('include_DOVS_CLEARING_DEVICE_DIM', False)
        kwargs['alias_DOVS_CLEARING_DEVICE_DIM']        = kwargs.get('alias_DOVS_CLEARING_DEVICE_DIM', 'DOV3')
        kwargs['cols_to_join_DOVS_CLEARING_DEVICE_DIM'] = kwargs.get('cols_to_join_DOVS_CLEARING_DEVICE_DIM', ['DEVICE_CD'])
        kwargs['select_cols_DOVS_CLEARING_DEVICE_DIM']  = kwargs.get('select_cols_DOVS_CLEARING_DEVICE_DIM', 
                                                                    [
                                                                        'DVC_TYP_NM', 
                                                                        dict(
                                                                            field_desc='SHORT_NM', 
                                                                            alias='SHORT_NM_CLR_DEV', 
                                                                            table_alias_prefix=kwargs['alias_DOVS_CLEARING_DEVICE_DIM']
                                                                        ), 
                                                                    ]
                                                                   )
        #-----
        # Inclusion of DOVS_CLEARING_DEVICE_DIM
        if kwargs['include_DOVS_CLEARING_DEVICE_DIM']:
            sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_CLEARING_DEVICE_DIM(
                alias_DOVS_OUTAGE_FACT         = kwargs['from_table_alias'], 
                alias_DOVS_CLEARING_DEVICE_DIM = kwargs['alias_DOVS_CLEARING_DEVICE_DIM'], 
                list_of_columns_to_join        = kwargs['cols_to_join_DOVS_CLEARING_DEVICE_DIM']
            )
            sql.add_join(
                sql_join                   = sql_join, 
                idx                        = None, 
                run_check                  = False, 
                join_cols_to_add_to_select = kwargs['select_cols_DOVS_CLEARING_DEVICE_DIM']
            )

        #-------------------------
        # DOVS_EQUIPMENT_TYPES_DIM kwargs
        #-------------------------
        kwargs['include_DOVS_EQUIPMENT_TYPES_DIM']      = kwargs.get('include_DOVS_EQUIPMENT_TYPES_DIM', False)
        kwargs['alias_DOVS_EQUIPMENT_TYPES_DIM']        = kwargs.get('alias_DOVS_EQUIPMENT_TYPES_DIM', 'DOV4')
        kwargs['cols_to_join_DOVS_EQUIPMENT_TYPES_DIM'] = kwargs.get('cols_to_join_DOVS_EQUIPMENT_TYPES_DIM', ['EQUIPMENT_CD'])
        kwargs['select_cols_DOVS_EQUIPMENT_TYPES_DIM']  = kwargs.get('select_cols_DOVS_EQUIPMENT_TYPES_DIM', 
                                                                    [
                                                                        'EQUIP_TYP_NM', 
                                                                        dict(
                                                                            field_desc='SHORT_NM', 
                                                                            alias='SHORT_NM_EQP_TYP', 
                                                                            table_alias_prefix=kwargs['alias_DOVS_EQUIPMENT_TYPES_DIM']
                                                                        )
                                                                    ]
                                                                   )
        #-----
        # Inclusion of DOVS_EQUIPMENT_TYPES_DIM
        if kwargs['include_DOVS_EQUIPMENT_TYPES_DIM']:
            sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_EQUIPMENT_TYPES_DIM(
                alias_DOVS_OUTAGE_FACT         = kwargs['from_table_alias'], 
                alias_DOVS_EQUIPMENT_TYPES_DIM = kwargs['alias_DOVS_EQUIPMENT_TYPES_DIM'], 
                list_of_columns_to_join        = kwargs['cols_to_join_DOVS_EQUIPMENT_TYPES_DIM']
            )
            sql.add_join(
                sql_join                   = sql_join, 
                idx                        = None, 
                run_check                  = False, 
                join_cols_to_add_to_select = kwargs['select_cols_DOVS_EQUIPMENT_TYPES_DIM']
            )

        #-------------------------
        # DOVS_OUTAGE_CAUSE_TYPES_DIM kwargs
        #-------------------------
        kwargs['include_DOVS_OUTAGE_CAUSE_TYPES_DIM']      = kwargs.get('include_DOVS_OUTAGE_CAUSE_TYPES_DIM', False)
        kwargs['alias_DOVS_OUTAGE_CAUSE_TYPES_DIM']        = kwargs.get('alias_DOVS_OUTAGE_CAUSE_TYPES_DIM', 'DOV5')
        kwargs['cols_to_join_DOVS_OUTAGE_CAUSE_TYPES_DIM'] = kwargs.get('cols_to_join_DOVS_OUTAGE_CAUSE_TYPES_DIM', ['MJR_CAUSE_CD', 'MNR_CAUSE_CD'])
        kwargs['select_cols_DOVS_OUTAGE_CAUSE_TYPES_DIM']  = kwargs.get('select_cols_DOVS_OUTAGE_CAUSE_TYPES_DIM', ['MJR_CAUSE_NM', 'MNR_CAUSE_NM'])
        #-----
        # Inclusion of DOVS_OUTAGE_CAUSE_TYPES_DIM
        if kwargs['include_DOVS_OUTAGE_CAUSE_TYPES_DIM']:
            sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_CAUSE_TYPES_DIM(
                alias_DOVS_OUTAGE_FACT            = kwargs['from_table_alias'], 
                alias_DOVS_OUTAGE_CAUSE_TYPES_DIM = kwargs['alias_DOVS_OUTAGE_CAUSE_TYPES_DIM'], 
                list_of_columns_to_join           = kwargs['cols_to_join_DOVS_OUTAGE_CAUSE_TYPES_DIM']
            )
            sql.add_join(
                sql_join                   = sql_join, 
                idx                        = None, 
                run_check                  = False, 
                join_cols_to_add_to_select = kwargs['select_cols_DOVS_OUTAGE_CAUSE_TYPES_DIM']
            )

        #-------------------------
        # DOVS_PREMISE_DIM kwargs
        #-------------------------
        kwargs['include_DOVS_PREMISE_DIM']      = kwargs.get('include_DOVS_PREMISE_DIM', False)
        kwargs['alias_DOVS_PREMISE_DIM']        = kwargs.get('alias_DOVS_PREMISE_DIM', 'PRIM')
        kwargs['cols_to_join_DOVS_PREMISE_DIM'] = kwargs.get('cols_to_join_DOVS_PREMISE_DIM', ['OUTG_REC_NB'])
        kwargs['join_type_DOVS_PREMISE_DIM']    = kwargs.get('join_type_DOVS_PREMISE_DIM', 'LEFT OUTER')
        kwargs['select_cols_DOVS_PREMISE_DIM']  = kwargs.get('select_cols_DOVS_PREMISE_DIM', ['OFF_TM', 'REST_TM', 'PREMISE_NB'])
        #-----
        # Inclusion of DOVS_PREMISE_DIM
        if kwargs['include_DOVS_PREMISE_DIM']:
            sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_PREMISE_DIM(
                alias_DOVS_OUTAGE_FACT  = kwargs['from_table_alias'], 
                alias_DOVS_PREMISE_DIM  = kwargs['alias_DOVS_PREMISE_DIM'], 
                list_of_columns_to_join = kwargs['cols_to_join_DOVS_PREMISE_DIM'], 
                join_type               = kwargs['join_type_DOVS_PREMISE_DIM']
            )
            sql.add_join(
                sql_join                   = sql_join, 
                idx                        = None, 
                run_check                  = False, 
                join_cols_to_add_to_select = kwargs['select_cols_DOVS_PREMISE_DIM']
                )    
        #-------------------------
        #--------------------------------------------------
        return sql
        
        
    #--------------------------------------------------
    @staticmethod
    def build_sql_std_outage(
        mjr_mnr_cause   = None, 
        include_premise = True, 
        **kwargs
    ):
        r"""
        include_premise
            If True, the columns ['OFF_TM', 'REST_TM', 'PREMISE_NB'] will be included from DOVS_PREMISE_DIM
        
        Acceptable kwargs:
          - date_range
            - tuple with two string elements, e.g., ['2021-01-01', '2021-04-01']
            - makes, e.g., DOV.DT_OFF_TS BETWEEN '2020-01-01' AND '2022-01-01'
          - specific_dates
            - list of string elements, e.g., ['2020-10-12', '2017-09-16', '2020-06-14']
            - makes, e.g., DOV.DT_OFF_TS IN ('2020-10-12', '2017-09-16', '2020-06-14')
          - dt_off_ts_col
          ***** Most likely, don't want both date_range and specific_dates set together!

          - outg_rec_nbs
          - outg_rec_nb_col

        """
        #-------------------------
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'DOV')
        #-------------------------
        possible_state_kwargs = ['states', 'state', 'STATE_ABBR_TX']
        found_states_kwargs = [x for x in kwargs if x in possible_state_kwargs]
        assert(len(found_states_kwargs)<=1)
        if len(found_states_kwargs)==0:
            kwargs['states'] = None
        #-------------------------
        kwargs['mjr_mnr_cause'] = mjr_mnr_cause
        #-------------------------
        kwargs['MJR_CAUSE_CD_col'] = kwargs.get('MJR_CAUSE_CD_col', 'MJR_CAUSE_CD')
        kwargs['MJR_CAUSE_CD']     = kwargs.get(
            'MJR_CAUSE_CD', 
            dict(field_desc=kwargs['MJR_CAUSE_CD_col'], 
                 table_alias_prefix=kwargs['from_table_alias'], 
                 comparison_operator='<>', 
                 value='NI', 
                 needs_quotes=True)
        )
        #-------------------------
        kwargs['DEVICE_CD_col'] = kwargs.get('DEVICE_CD_col', 'DEVICE_CD')
        kwargs['DEVICE_CD']     = kwargs.get(
            'DEVICE_CD', 
            dict(field_desc=kwargs['DEVICE_CD_col'], 
                 table_alias_prefix=kwargs['from_table_alias'], 
                 comparison_operator='<>', 
                 value=85, 
                 needs_quotes=False)
        )
        #-------------------------
        kwargs['INTRPTN_TYP_CD_col'] = kwargs.get('INTRPTN_TYP_CD_col', 'INTRPTN_TYP_CD')
        kwargs['INTRPTN_TYP_CD']     = kwargs.get('INTRPTN_TYP_CD', 'S')
        #-------------------------
        kwargs['CURR_REC_STAT_CD_col'] = kwargs.get('CURR_REC_STAT_CD_col', 'CURR_REC_STAT_CD')
        kwargs['CURR_REC_STAT_CD']     = kwargs.get('CURR_REC_STAT_CD', 'A')
        #-------------------------
        kwargs['include_DOVS_MASTER_GEO_DIM']         = kwargs.get('include_DOVS_MASTER_GEO_DIM',         True)
        kwargs['include_DOVS_OUTAGE_ATTRIBUTES_DIM']  = kwargs.get('include_DOVS_OUTAGE_ATTRIBUTES_DIM',  True)
        kwargs['include_DOVS_CLEARING_DEVICE_DIM']    = kwargs.get('include_DOVS_CLEARING_DEVICE_DIM',    True)
        kwargs['include_DOVS_EQUIPMENT_TYPES_DIM']    = kwargs.get('include_DOVS_EQUIPMENT_TYPES_DIM',    True)
        kwargs['include_DOVS_OUTAGE_CAUSE_TYPES_DIM'] = kwargs.get('include_DOVS_OUTAGE_CAUSE_TYPES_DIM', True)
        # Need to protect again case where user sets include_DOVS_PREMISE_DIM to one value and
        # include_premise to a different value.
        if kwargs.get('include_DOVS_PREMISE_DIM', include_premise) != include_premise:
            print(f"ERROR IN DOVSOutages_SQL.build_sql_std_outage, include_DOVS_PREMISE_DIM={kwargs['include_DOVS_PREMISE_DIM']} and include_premise={include_premise} disagree!")
            print("Program will crash, please resolve (either set equal, or simply remove include_DOVS_PREMISE_DIM from arguments")
        kwargs['include_DOVS_PREMISE_DIM'] = include_premise
        
        # If the user set kwargs['select_cols_DOVS_PREMISE_DIM'], but didn't include PREMISE_NB, then the premise numbers won't actually be included in the query.
        #   e.g., suppose the user set include_premise=True and select_cols_DOVS_PREMISE_DIM=['CIRCT_NM'], then the only column taken from DOVS_PREMISE_DIM 
        #         will be CIRCT_NM
        # The purpose of the include_premise boolean was to control whether or not the premise numbers were included, not just the DOVS_PREMISE_DIM table! (to that point,
        #   the parameter should have probably been named 'include_PNs' instead of 'include_premise'
        # The lines below ensure that ['OFF_TM', 'REST_TM', 'PREMISE_NB'] will always be selected (in addition to whatever is input by the user) when include_premise==True 
        #   (now clearly stated in documentation as well)
        if include_premise:
            kwargs['select_cols_DOVS_PREMISE_DIM'] = list(set(kwargs.get('select_cols_DOVS_PREMISE_DIM', [])).union(set(['OFF_TM', 'REST_TM', 'PREMISE_NB'])))
        #-------------------------
        return DOVSOutages_SQL.build_sql_outage(
            cols_of_interest=kwargs.pop('cols_of_interest', None), 
            addtnl_cols_of_interest=kwargs.pop('addtnl_cols_of_interest', None), 
            **kwargs
        )
    
    #--------------------------------------------------
    @staticmethod
    def build_sql_distinct_outage_field(
        distinct_field, 
        mjr_mnr_cause   = None, 
        include_premise = True, 
        date_range      = None, 
        **kwargs
    ):
        assert(Utilities_sql.is_object_one_of_types(distinct_field, [SQLSelectElement, str, dict]))
        #-------------------------
        sql_query = DOVSOutages_SQL.build_sql_std_outage(
            mjr_mnr_cause=mjr_mnr_cause, 
            include_premise=include_premise, 
            date_range=date_range, 
            **kwargs
        )
        #-------------------------
        if isinstance(distinct_field, SQLSelectElement):
            sql_el = distinct_field
        elif isinstance(distinct_field, str):
            sql_el = SQLSelectElement(field_desc=distinct_field)
        elif isinstance(distinct_field, dict):
            sql_el = SQLSelectElement(**distinct_field)
        else:
            assert(0)
        #-------------------------
        # The final field_desc should be of the form DISTINCT(...)
        # If not in this form because, e.g., a normal str given (which will oftentimes be the case)
        #   then put it in such a form
        pattern = r'DISTINCT\(.*\).*'
        found = re.findall(pattern, sql_el.field_desc)
        assert(len(found)==0 or len(found)==1)
        if len(found)==0:
            # Use get_field_desc(include_alias=False, include_table_alias_prefix=True) for new field_desc
            #   and set sql_el.table_alias_prefix=None afterwards because:
            #     WANT: table.field AS alias --> DISTINCT(table.field) AS alias
            #     NOT:  table.field AS alias --> DISTINCT(table.field AS alias)
            sql_el.field_desc = f"DISTINCT({sql_el.get_field_desc(include_alias=False, include_table_alias_prefix=True)})"
            sql_el.table_alias_prefix = None
        sql_select = SQLSelect(field_descs=[sql_el])
        sql_query.set_sql_select(sql_select)
        #-------------------------
        return sql_query
        
    
    @staticmethod
    def build_sql_find_outage_xfmrs(
        mjr_mnr_cause   = None, 
        include_premise = True, 
        date_range      = None, 
        **kwargs
    ):
        return DOVSOutages_SQL.build_sql_distinct_outage_field(
            distinct_field='DISTINCT(DOV.LOCATION_ID)', 
            mjr_mnr_cause=mjr_mnr_cause, 
            include_premise=include_premise, 
            date_range=date_range, 
            **kwargs
        )


    @staticmethod
    def build_sql_outage_premises(
        outg_rec_nbs, 
        cols_of_interest = None, 
        **kwargs
    ):
        r"""
        """
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = TableInfos.DOVS_PREMISE_DIM_TI.std_columns_of_interest
        #-------------------------
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'PRIM')
        sql_select = SQLSelect(field_descs=cols_of_interest, global_table_alias_prefix=kwargs['from_table_alias'])
        #-------------------------
        sql_from = SQLFrom(
            schema_name='DOVSADM', 
            table_name='DOVS_PREMISE_DIM', 
            alias=kwargs['from_table_alias'], 
            sql_join_coll=None
        )
        #-------------------------
        outg_rec_nb_col = kwargs.get('outg_rec_nb_col', 'OUTG_REC_NB')
        sql_where = SQLWhere()
        sql_where.add_where_statement(field_desc=outg_rec_nb_col, table_alias_prefix=kwargs['from_table_alias'], comparison_operator='IN', 
                                      value=f'({Utilities_sql.join_list_w_quotes(outg_rec_nbs)})', needs_quotes=False)
        #-------------------------
        return_sql = SQLQuery(sql_select, sql_from, sql_where)
        #-------------------------
        return return_sql
    

    @staticmethod
    def build_sql_n_outgs_per_PN(
        alias_0              = 'sql_0', 
        return_separate      = False, 
        addtnl_const_selects = None, 
        **build_sql_outage_kwargs, 
    ):
        r"""
        addtnl_const_selects:
            If not None, this must be a dict object
            This can be used if one wants to have some constant values returned in the table being built.
                e.g., if collecting for a specific trsf_pole_nb, one may want 'trsf_pole_nb' = 123456 in the table for easier use later.
                      In such a case, one would want addtnl_const_selects = dict(trsf_pole_nb=123456)
        """
        #-------------------------
        build_sql_outage_kwargs = {k:v for k,v in build_sql_outage_kwargs.items() if k not in ['cols_of_interest', 'include_DOVS_PREMISE_DIM', 'select_cols_DOVS_PREMISE_DIM', 'alias']}
        #-------------------------
        sql_0 = DOVSOutages_SQL.build_sql_outage(
            cols_of_interest             = ['OUTG_REC_NB'], 
            include_DOVS_PREMISE_DIM     = True, 
            select_cols_DOVS_PREMISE_DIM = ['PREMISE_NB'], 
            alias                        = alias_0, 
            **build_sql_outage_kwargs
        )
        #-----
        if addtnl_const_selects is not None:
            assert(isinstance(addtnl_const_selects, dict))
            for k,v in addtnl_const_selects.items():
                sql_0.sql_select.add_select_element(field_desc=f"'{v}'", alias=f'{k}')
        #-----
        include_with=True
        if return_separate:
            include_with = False
        sql_stmnt_0 = sql_0.get_sql_statement(
            insert_n_tabs_to_each_line = 0, 
            include_alias              = True, 
            include_with               = include_with
            )
        #-------------------------
        sql_select = SQLSelect(['PREMISE_NB', 'COUNT(*)'], global_table_alias_prefix=None)
        #--------------------------------------------------
        sql_from = SQLFrom(schema_name = None, table_name = alias_0, alias=None)
        #--------------------------------------------------
        grp_by = ['PREMISE_NB']
        #-----
        if addtnl_const_selects is not None:
            sql_select.add_select_elements(field_descs=list(addtnl_const_selects.keys()))
            grp_by.extend(list(addtnl_const_selects.keys()))
        #-----
        sql_groupby = SQLGroupBy(
            field_descs               = grp_by, 
            global_table_alias_prefix = None, 
            idxs                      = None, 
            run_check                 = True
        )
        #--------------------------------------------------
        sql = SQLQuery(
            sql_select  = sql_select, 
            sql_from    = sql_from, 
            sql_where   = None, 
            sql_groupby = sql_groupby, 
            alias       = None
        )
        #--------------------------------------------------
        sql_stmnt = f"{sql_stmnt_0}\n{sql.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=False, include_with=False)}"
        #-------------------------
        if return_separate:
            sql_stmnt_gen = sql.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=False, include_with=False)
            return sql_stmnt_0, sql_stmnt_gen
        else:
            return sql_stmnt
        

    @staticmethod
    def build_sql_n_outgs_per_PN_for_xfmrs_in_consolidated_df(
        df_cnsldtd, 
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        t_min_col                      = 't_search_min', 
        t_max_col                      = 't_search_max', 
        PNs_col                        = 'prem_nb', 
        trsf_pole_nbs_to_ignore        = [' ', 'TRANSMISSION', 'PRIMARY', 'NETWORK'], 
        build_sql_outage_kwargs        = None, 
    ):
        r"""
        df_cnsldtd:
            Should be a pd.DataFrame object with a single row for each (trsf_pole_nb, t_min, t_max) combination.
            The PNs_col column contains a list of premise numbers for each transformer at that specific t_min,t_max period
        """
        #--------------------------------------------------
        if build_sql_outage_kwargs is None:
            build_sql_outage_kwargs = {}
        assert(isinstance(build_sql_outage_kwargs, dict))
        #--------------------------------------------------
        nec_cols = [trsf_pole_nb_col, t_min_col, t_max_col, PNs_col]
        assert(set(nec_cols).difference(set(df_cnsldtd.columns.tolist()))==set())
        #-----
        df = df_cnsldtd[nec_cols].copy()
        #--------------------------------------------------
        if trsf_pole_nbs_to_ignore is not None:
            assert(isinstance(trsf_pole_nbs_to_ignore, list))
            df = df[~df[trsf_pole_nb_col].isin(trsf_pole_nbs_to_ignore)]
        #--------------------------------------------------
        rndm_pfx   = Utilities.generate_random_string(str_len=4, letters='letters_only')
        sql_stmnts     = []
        sql_stmnts_gen = []
        for idx_i, (idx, row) in enumerate(df.iterrows()):
            # Set premise_nbs and date_range in build_sql_outage_kwargs_i
            build_sql_outage_kwargs_i = copy.deepcopy(build_sql_outage_kwargs)
            build_sql_outage_kwargs_i['premise_nbs'] = row[PNs_col]
            build_sql_outage_kwargs_i['date_range']  = [row[t_min_col], row[t_max_col]]
            #-----
            sql_i_0, sql_i_gen = DOVSOutages_SQL.build_sql_n_outgs_per_PN(
                alias_0              = f"{rndm_pfx}_{idx_i}", 
                return_separate      = True, 
                addtnl_const_selects = {
                    trsf_pole_nb_col : row[trsf_pole_nb_col], 
                    t_min_col        : row[t_min_col], 
                    t_max_col        : row[t_max_col]
                }, 
                **build_sql_outage_kwargs_i
            )
            sql_stmnts.append(sql_i_0)
            sql_stmnts_gen.append(sql_i_gen)
        
        #--------------------------------------------------
        fnl_stmnt = ''
        #-------------------------
        for idx_i, stmnt_i in enumerate(sql_stmnts):
            if idx_i==0:
                fnl_stmnt += 'WITH '
            #-----
            fnl_stmnt += stmnt_i
            #-----
            if idx_i != len(sql_stmnts)-1:
                fnl_stmnt += ','
            fnl_stmnt += '\n'
        #-------------------------
        assert(len(sql_stmnts)==len(sql_stmnts_gen))
        for idx_i, sql_i_gen in enumerate(sql_stmnts_gen):
            fnl_stmnt += sql_i_gen+'\n'
            if idx_i != len(sql_stmnts)-1:
                fnl_stmnt += 'UNION\n'
        #--------------------------------------------------
        return fnl_stmnt
    
    #-----------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_std_join_DOVS_OUTAGE_FACT_w_DOVS_MASTER_GEO_DIM(
        alias_DOVS_OUTAGE_FACT    = 'DOV', 
        alias_DOVS_MASTER_GEO_DIM = 'DOV1', 
        list_of_columns_to_join   = None
    ):
        join_type = 'LEFT OUTER'
        join_table = 'DOVSADM.DOVS_MASTER_GEO_DIM'
        join_table_alias = alias_DOVS_MASTER_GEO_DIM
        orig_table_alias = alias_DOVS_OUTAGE_FACT
        if list_of_columns_to_join is None:
            list_of_columns_to_join = [
                ['OPERATING_UNIT_ID', 'OPRTG_UNT_ID'], 
                ['STATE_ABBR_TX', 'STATE_ID'], 
                ['OPCO_NBR', 'OPCO_ID'], 
                ['DISTRICT_NB', 'DISTRICT_ID'], 
                ['SRVC_CNTR_NB', 'AREA_ID'], 
                ['GIS_CRCT_NB', 'GIS_CIRCUIT_ID']
            ]
        sql_join = SQLJoin(join_type=join_type, join_table=join_table, 
                           join_table_alias=join_table_alias, orig_table_alias=orig_table_alias, 
                           list_of_columns_to_join=list_of_columns_to_join)
        return sql_join

    #--------------------------------------------------
    @staticmethod
    def get_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_ATTRIBUTES_DIM(
        alias_DOVS_OUTAGE_FACT           = 'DOV', 
        alias_DOVS_OUTAGE_ATTRIBUTES_DIM = 'DOV2', 
        list_of_columns_to_join          = None
    ):
        r"""
        """
        #-------------------------
        join_type = 'LEFT OUTER'
        join_table = 'DOVSADM.DOVS_OUTAGE_ATTRIBUTES_DIM'
        join_table_alias = alias_DOVS_OUTAGE_ATTRIBUTES_DIM
        orig_table_alias = alias_DOVS_OUTAGE_FACT
        if list_of_columns_to_join is None:
            list_of_columns_to_join = ['OUTG_REC_NB']
        sql_join = SQLJoin(join_type=join_type, join_table=join_table, 
                           join_table_alias=join_table_alias, orig_table_alias=orig_table_alias, 
                           list_of_columns_to_join=list_of_columns_to_join)
        return sql_join

    #--------------------------------------------------
    @staticmethod
    def get_std_join_DOVS_OUTAGE_FACT_w_DOVS_CLEARING_DEVICE_DIM(
        alias_DOVS_OUTAGE_FACT         = 'DOV', 
        alias_DOVS_CLEARING_DEVICE_DIM = 'DOV3', 
        list_of_columns_to_join        = None
    ):
        r"""
        """
        #-------------------------
        join_type = 'LEFT OUTER'
        join_table = 'DOVSADM.DOVS_CLEARING_DEVICE_DIM'
        join_table_alias = alias_DOVS_CLEARING_DEVICE_DIM
        orig_table_alias = alias_DOVS_OUTAGE_FACT
        if list_of_columns_to_join is None:
            list_of_columns_to_join = ['DEVICE_CD']
        sql_join = SQLJoin(join_type=join_type, join_table=join_table, 
                           join_table_alias=join_table_alias, orig_table_alias=orig_table_alias, 
                           list_of_columns_to_join=list_of_columns_to_join)
        return sql_join

    #--------------------------------------------------
    @staticmethod
    def get_std_join_DOVS_OUTAGE_FACT_w_DOVS_EQUIPMENT_TYPES_DIM(
        alias_DOVS_OUTAGE_FACT         = 'DOV', 
        alias_DOVS_EQUIPMENT_TYPES_DIM = 'DOV4', 
        list_of_columns_to_join        = None
    ):
        r"""
        """
        #-------------------------
        join_type = 'LEFT OUTER'
        join_table = 'DOVSADM.DOVS_EQUIPMENT_TYPES_DIM'
        join_table_alias = alias_DOVS_EQUIPMENT_TYPES_DIM
        orig_table_alias = alias_DOVS_OUTAGE_FACT
        if list_of_columns_to_join is None:
            list_of_columns_to_join = ['EQUIPMENT_CD']
        sql_join = SQLJoin(join_type=join_type, join_table=join_table, 
                           join_table_alias=join_table_alias, orig_table_alias=orig_table_alias, 
                           list_of_columns_to_join=list_of_columns_to_join)
        return sql_join

    #--------------------------------------------------
    @staticmethod
    def get_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_CAUSE_TYPES_DIM(
        alias_DOVS_OUTAGE_FACT            = 'DOV', 
        alias_DOVS_OUTAGE_CAUSE_TYPES_DIM = 'DOV5', 
        list_of_columns_to_join           = None
    ):
        r"""
        """
        #-------------------------
        join_type = 'LEFT OUTER'
        join_table = 'DOVSADM.DOVS_OUTAGE_CAUSE_TYPES_DIM'
        join_table_alias = alias_DOVS_OUTAGE_CAUSE_TYPES_DIM
        orig_table_alias = alias_DOVS_OUTAGE_FACT
        if list_of_columns_to_join is None:
            list_of_columns_to_join = ['MJR_CAUSE_CD', 'MNR_CAUSE_CD']
        sql_join = SQLJoin(join_type=join_type, join_table=join_table, 
                           join_table_alias=join_table_alias, orig_table_alias=orig_table_alias, 
                           list_of_columns_to_join=list_of_columns_to_join)
        return sql_join

    #--------------------------------------------------
    @staticmethod
    def get_std_join_DOVS_OUTAGE_FACT_w_DOVS_PREMISE_DIM(
        alias_DOVS_OUTAGE_FACT  = 'DOV', 
        alias_DOVS_PREMISE_DIM  = 'PRIM', 
        list_of_columns_to_join = None, 
        join_type               = 'LEFT OUTER'
    ):
        r"""
        """
        #-------------------------
        join_table = 'DOVSADM.DOVS_PREMISE_DIM'
        join_table_alias = alias_DOVS_PREMISE_DIM
        orig_table_alias = alias_DOVS_OUTAGE_FACT
        if list_of_columns_to_join is None:
            list_of_columns_to_join = ['OUTG_REC_NB']
        sql_join = SQLJoin(join_type=join_type, join_table=join_table, 
                           join_table_alias=join_table_alias, orig_table_alias=orig_table_alias, 
                           list_of_columns_to_join=list_of_columns_to_join)
        return sql_join
    #-----------------------------------------------------------------------------------------------------------
    #--------------------------------------------------
    @staticmethod
    def add_join(
        add_to_me, 
        sql_join, 
        idx       = None, 
        run_check = False
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        #-------------------------
        if isinstance(add_to_me, SQLJoinCollection) or isinstance(add_to_me, SQLFrom):
            add_to_me.add_join_to_coll(sql_join=sql_join, idx=idx, run_check=run_check)
        elif isinstance(add_to_me, SQLQuery):
            add_to_me.add_join(sql_join=sql_join, idx=idx, run_check=run_check)
        else:
            assert(0)
        return add_to_me

    #--------------------------------------------------
    @staticmethod
    def add_std_join_DOVS_OUTAGE_FACT_w_DOVS_MASTER_GEO_DIM(
        add_to_me, 
        idx                       = None, 
        run_check                 = False, 
        alias_DOVS_OUTAGE_FACT    = 'DOV', 
        alias_DOVS_MASTER_GEO_DIM = 'DOV1'
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_MASTER_GEO_DIM(alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
                                                                                       alias_DOVS_MASTER_GEO_DIM=alias_DOVS_MASTER_GEO_DIM)
        add_to_me = DOVSOutages_SQL.add_join(add_to_me=add_to_me, sql_join=sql_join, 
                                             idx=idx, run_check=run_check)
        return add_to_me

    #--------------------------------------------------
    @staticmethod
    def add_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_ATTRIBUTES_DIM(
        add_to_me, 
        idx                              = None, 
        run_check                        = False, 
        alias_DOVS_OUTAGE_FACT           = 'DOV', 
        alias_DOVS_OUTAGE_ATTRIBUTES_DIM = 'DOV2'
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_ATTRIBUTES_DIM(alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
                                                                                              alias_DOVS_OUTAGE_ATTRIBUTES_DIM=alias_DOVS_OUTAGE_ATTRIBUTES_DIM)
        add_to_me = DOVSOutages_SQL.add_join(add_to_me=add_to_me, sql_join=sql_join, 
                                             idx=idx, run_check=run_check)
        return add_to_me

    #--------------------------------------------------
    @staticmethod
    def add_std_join_DOVS_OUTAGE_FACT_w_DOVS_CLEARING_DEVICE_DIM(
        add_to_me, 
        idx                            = None, 
        run_check                      = False, 
        alias_DOVS_OUTAGE_FACT         = 'DOV', 
        alias_DOVS_CLEARING_DEVICE_DIM = 'DOV3'
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_CLEARING_DEVICE_DIM(alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
                                                                                            alias_DOVS_CLEARING_DEVICE_DIM=alias_DOVS_CLEARING_DEVICE_DIM)
        add_to_me = DOVSOutages_SQL.add_join(add_to_me=add_to_me, sql_join=sql_join, 
                                             idx=idx, run_check=run_check)
        return add_to_me

    #--------------------------------------------------
    @staticmethod
    def add_std_join_DOVS_OUTAGE_FACT_w_DOVS_EQUIPMENT_TYPES_DIM(
        add_to_me, 
        idx                            = None, 
        run_check                      = False, 
        alias_DOVS_OUTAGE_FACT         = 'DOV', 
        alias_DOVS_EQUIPMENT_TYPES_DIM = 'DOV4'
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_EQUIPMENT_TYPES_DIM(alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
                                                                                            alias_DOVS_EQUIPMENT_TYPES_DIM=alias_DOVS_EQUIPMENT_TYPES_DIM)
        add_to_me = DOVSOutages_SQL.add_join(add_to_me=add_to_me, sql_join=sql_join, 
                                             idx=idx, run_check=run_check)
        return add_to_me

    #--------------------------------------------------
    @staticmethod
    def add_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_CAUSE_TYPES_DIM(
        add_to_me, 
        idx                               = None, 
        run_check                         = False, 
        alias_DOVS_OUTAGE_FACT            = 'DOV', 
        alias_DOVS_OUTAGE_CAUSE_TYPES_DIM = 'DOV5'
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_CAUSE_TYPES_DIM(alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
                                                                                               alias_DOVS_OUTAGE_CAUSE_TYPES_DIM=alias_DOVS_OUTAGE_CAUSE_TYPES_DIM)
        add_to_me = DOVSOutages_SQL.add_join(add_to_me=add_to_me, sql_join=sql_join, 
                                             idx=idx, run_check=run_check)
        return add_to_me

    #--------------------------------------------------
    @staticmethod
    def add_std_join_DOVS_OUTAGE_FACT_w_DOVS_PREMISE_DIM(
        add_to_me, 
        idx                    = None, 
        run_check              = False, 
        alias_DOVS_OUTAGE_FACT = 'DOV', 
        alias_DOVS_PREMISE_DIM = 'PRIM'
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        sql_join = DOVSOutages_SQL.get_std_join_DOVS_OUTAGE_FACT_w_DOVS_PREMISE_DIM(alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
                                                                                    alias_DOVS_PREMISE_DIM=alias_DOVS_PREMISE_DIM)
        add_to_me = DOVSOutages_SQL.add_join(add_to_me=add_to_me, sql_join=sql_join, 
                                             idx=idx, run_check=run_check)
        return add_to_me
    
    #--------------------------------------------------
    @staticmethod
    def add_all_std_joins(
        add_to_me, 
        idx                               = None, 
        run_check                         = False, 
        include_premise_dim               = True, 
        alias_DOVS_OUTAGE_FACT            = 'DOV', 
        alias_DOVS_MASTER_GEO_DIM         = 'DOV1', 
        alias_DOVS_OUTAGE_ATTRIBUTES_DIM  = 'DOV2', 
        alias_DOVS_CLEARING_DEVICE_DIM    = 'DOV3', 
        alias_DOVS_EQUIPMENT_TYPES_DIM    = 'DOV4', 
        alias_DOVS_OUTAGE_CAUSE_TYPES_DIM = 'DOV5', 
        alias_DOVS_PREMISE_DIM            = 'PRIM'
    ):
        r"""
        add_to_me can be type SQLJoinCollection, SQLFrom, or SQLQuery
        """
        #-------------------------
        assert(Utilities_sql.is_object_one_of_types(add_to_me, [SQLJoinCollection, SQLFrom, SQLQuery]))
        #-----
        add_to_me = DOVSOutages_SQL.add_std_join_DOVS_OUTAGE_FACT_w_DOVS_MASTER_GEO_DIM(
            add_to_me=add_to_me, idx=idx, run_check=run_check, 
            alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
            alias_DOVS_MASTER_GEO_DIM=alias_DOVS_MASTER_GEO_DIM
        )
        if idx is not None:
            idx+=1
        #-----
        add_to_me = DOVSOutages_SQL.add_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_ATTRIBUTES_DIM(
            add_to_me=add_to_me, idx=idx, run_check=run_check, 
            alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
            alias_DOVS_OUTAGE_ATTRIBUTES_DIM=alias_DOVS_OUTAGE_ATTRIBUTES_DIM
        )
        if idx is not None:
            idx+=1
        #-----
        add_to_me = DOVSOutages_SQL.add_std_join_DOVS_OUTAGE_FACT_w_DOVS_CLEARING_DEVICE_DIM(
            add_to_me=add_to_me, idx=idx, run_check=run_check, 
            alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
            alias_DOVS_CLEARING_DEVICE_DIM=alias_DOVS_CLEARING_DEVICE_DIM
        )
        if idx is not None:
            idx+=1
        #-----
        add_to_me = DOVSOutages_SQL.add_std_join_DOVS_OUTAGE_FACT_w_DOVS_EQUIPMENT_TYPES_DIM(
            add_to_me=add_to_me, idx=idx, run_check=run_check, 
            alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
            alias_DOVS_EQUIPMENT_TYPES_DIM=alias_DOVS_EQUIPMENT_TYPES_DIM
        )
        if idx is not None:
            idx+=1
        #-----
        add_to_me = DOVSOutages_SQL.add_std_join_DOVS_OUTAGE_FACT_w_DOVS_OUTAGE_CAUSE_TYPES_DIM(
            add_to_me=add_to_me, idx=idx, run_check=run_check, 
            alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
            alias_DOVS_OUTAGE_CAUSE_TYPES_DIM=alias_DOVS_OUTAGE_CAUSE_TYPES_DIM
        )
        if idx is not None:
            idx+=1
        #-----
        if include_premise_dim:
            add_to_me = DOVSOutages_SQL.add_std_join_DOVS_OUTAGE_FACT_w_DOVS_PREMISE_DIM(
                add_to_me=add_to_me, idx=idx, run_check=run_check, 
                alias_DOVS_OUTAGE_FACT=alias_DOVS_OUTAGE_FACT, 
                alias_DOVS_PREMISE_DIM=alias_DOVS_PREMISE_DIM
            )
            if idx is not None:
                idx+=1
        return add_to_me
        
    #****************************************************************************************************
    @staticmethod
    def build_sql_DOVS_OUTAGE_CAUSE_TYPES_DIM():
        r"""
        Builds a SQLQuery object to grab entire DOVS_OUTAGE_CAUSE_TYPES_DIM table (not very big, ~300 entries)
        """
        cols_of_interest = TableInfos.DOVS_OUTAGE_CAUSE_TYPES_DIM_TI.columns_full
        schema_name = TableInfos.DOVS_OUTAGE_CAUSE_TYPES_DIM_TI.schema_name
        table_name  = TableInfos.DOVS_OUTAGE_CAUSE_TYPES_DIM_TI.table_name
        conn_db_fcn = TableInfos.DOVS_OUTAGE_CAUSE_TYPES_DIM_TI.conn_fcn
        #-------------------------
        sql_select = SQLSelect(cols_of_interest)
        sql_from = SQLFrom(schema_name, table_name)
        sql = SQLQuery(sql_select = sql_select, 
                       sql_from = sql_from, 
                       sql_where = None, 
                       sql_groupby = None, 
                       alias = None
                      )
        #-------------------------
        return sql    