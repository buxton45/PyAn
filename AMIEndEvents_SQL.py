#!/usr/bin/env python

r"""
Holds AMIEndEvents_SQL class.  See AMIEndEvents_SQL.AMIEndEvents_SQL for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys

#--------------------------------------------------
from AMI_SQL import AMI_SQL, DfToSqlMap
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from SQLWhere import SQLWhereElement, SQLWhere
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
#--------------------------------------------------

class AMIEndEvents_SQL(AMI_SQL):
    def __init__(self):
        self.sql_query = None
        #self.table_info = 
        
    #****************************************************************************************************
    @staticmethod
    def build_sql_end_events(
        cols_of_interest=None, 
        **kwargs
    ):
        r"""
        See AMI_SQL.add_ami_where_statements for my updated list of acceptable kwargs with respect to the where statement.

        Acceptable kwargs:
          - date_range
            - tuple with two string elements, e.g., ['2021-01-01', '2021-04-01']
          - date_col
            - default: aep_event_dt (NOTE: not aep_usage_dt, as in AMI_SQL.add_ami_where_statements

          - datetime_range
            - tuple with two string elements, e.g., ['2021-01-01 00:00:00', '2021-04-01 12:00:00']
          - datetime_col
            - default: valuesinterval (NOTE: not starttimeperiod, as in AMI_SQL.add_ami_where_statements)
          - datetime_pattern
            - Regex pattern used to convert string to a form that SQL/Athena/Oracle can convert to a TIMESTAMP
            - default: r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"

          - serial_number(s)
          - serial_number_col
            - default: serialnumber

          - premise_nb(s) (or, aep_premise_nb(s) will work also)
          - premise_nb_col
            - default: aep_premise_nb

          - opco(s) (or, aep_opco(s) will work also)
          - opco_col
            - default: aep_opco

          - state(s)
          - state_col
            - default: aep_state

          - schema_name
            - default: meter_events
          - table_name
            - default: end_device_event
            
          - issuertracking_ids
          - issuertracking_id_col
          
          - enddeviceeventtypeid(s)
          - enddeviceeventtypeid_col
          
          - wheres_to_combine
            - default: None
            - Allows user to combine where elements with e.g. 'OR' or 'AND' operators
            - This needs to be a dict or list of dicts (a list can be utilized when multiple groups are to be combined).
            - The keys in the dict(s) must be:
                to_combine: REQUIRED
                join_operator: optional, set to 'OR' if no argument supplied.
            - Within to_combine, one should use the same nomenclature as used in kwargs.
                - e.g., if one uses serial_number in kwargs, one must also use serial_number (NOT serial_numbers) in to_combine
            - EX. 1: 
                wheres_to_combine = dict(to_combine=['date_range', 'serial_numbers'], join_operator='OR')
            - EX. 2:
                wheres_to_combine = [
                    dict(to_combine=['date_range', 'serial_numbers'], join_operator='OR'), 
                    dict(to_combine=['datetime_range', 'premise_nbs'], join_operator='OR')
                ]   
        """
        #-------------------------
        # First, make any necessary adjustments to kwargs
        kwargs['schema_name']      = kwargs.get('schema_name', 'meter_events')
        kwargs['table_name']       = kwargs.get('table_name', 'end_device_event')
        #kwargs['from_table_alias'] = kwargs.get('from_table_alias', None)
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'EDE')       
        kwargs['date_col']         = kwargs.get('date_col', 'aep_event_dt')
        kwargs['datetime_col']     = kwargs.get('datetime_col', 'valuesinterval')
        kwargs['datetime_pattern'] = kwargs.get('datetime_pattern', 
                                                 r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        #--------------------------------------------------
        # field_descs_dict needed if any where elements are to be combined
        field_descs_dict = dict()
        #-------------------------
        # If any where statements are to be combined, they must be combined here instead of within AMI_SQL.build_sql_ami,
        #   as the latter doesn't have all the necessary info (i.e., doesn't have all possible field descriptions)
        # Thus, wheres_to_combine must be popped off of kwargs before being used in AMI_SQL.build_sql_ami
        wheres_to_combine = kwargs.pop('wheres_to_combine', None)
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = TableInfos.AMIEndEvents_TI.std_columns_of_interest
        sql = AMI_SQL.build_sql_ami(cols_of_interest=cols_of_interest, **kwargs)
        #--------------------------------------------------
        # Other WHERE statements not handled by AMI_SQL.build_sql_ami
        issuertracking_ids     = kwargs.get('issuertracking_ids', None)
        issuertracking_id_col  = kwargs.get('issuertracking_id_col', 'issuertracking_id')
        if issuertracking_ids is not None:
            field_descs_dict['issuertracking_ids'] = issuertracking_id_col
            if isinstance(issuertracking_ids, SQLWhereElement):
                sql.sql_where.add_where_statement(issuertracking_ids)
            else:
                sql.sql_where.add_where_statement(
                    field_desc          = issuertracking_id_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = 'IN', 
                    value               = f'({Utilities_sql.join_list_w_quotes(issuertracking_ids)})', 
                    needs_quotes        = False
                )
        #--------------------------------------------------
        enddeviceeventtypeid_col = kwargs.get('enddeviceeventtypeid_col', 'enddeviceeventtypeid')
        assert(not('enddeviceeventtypeids' in kwargs and 'enddeviceeventtypeid' in kwargs))
        enddeviceeventtypeids    = kwargs.get('enddeviceeventtypeids', kwargs.get('enddeviceeventtypeid', None))
        if enddeviceeventtypeids is not None:
            field_descs_dict['enddeviceeventtypeids' if 'enddeviceeventtypeids' in kwargs else 'enddeviceeventtypeid'] = enddeviceeventtypeid_col
            if isinstance(enddeviceeventtypeids, SQLWhereElement):
                sql.sql_where.add_where_statement(enddeviceeventtypeids)
            else:
                sql.sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql.sql_where, 
                    field_desc         = enddeviceeventtypeid_col, 
                    value              = enddeviceeventtypeids, 
                    table_alias_prefix = kwargs['from_table_alias']
                )
        #--------------------------------------------------
        if wheres_to_combine is not None:
            # To get full field_descs_dict, combine the one built above with that from sql.sql_where.addtnl_info['field_descs_dict']
            assert('field_descs_dict' in sql.sql_where.addtnl_info.keys())
            field_descs_dict = {**sql.sql_where.addtnl_info['field_descs_dict'], **field_descs_dict}
            sql.sql_where = AMI_SQL.combine_where_elements(
                sql_where         = sql.sql_where, 
                wheres_to_combine = wheres_to_combine, 
                field_descs_dict  = field_descs_dict
            )
            sql.sql_where.addtnl_info['field_descs_dict'] = field_descs_dict
        #--------------------------------------------------
        return sql
        
        
    #****************************************************************************************************
    @staticmethod            
    def build_sql_end_events_for_df_with_search_time_window(
        cols_of_interest           , 
        df_with_search_time_window , 
        build_sql_function         = None, 
        build_sql_function_kwargs  = {},  
        sql_alias_base             = 'USG_', 
        max_n_prem_per_outg        = None, 
        join_mp_args               = False, 
        date_only                  = False, 
        output_t_minmax            = False, 
        df_args                    = {}
    ):
        r"""
        NOTE: python won't let me set the default value to build_sql_function=AMIEndEvents_SQL.build_sql_end_events.
              Thus, the workaround is to set the default to None, then set it to 
              AMIEndEvents_SQL.build_sql_end_events if it is None

        TODO: Investigate what an appropriate default value for max_n_prem_per_outg would be....
              Probably something large, like 1000.  But, should find outages with huge numbers of customers
              affected to find the right number.
        """
        #-------------------------------------------------- 
        if build_sql_function is None:
            build_sql_function=AMIEndEvents_SQL.build_sql_end_events
        #--------------------------------------------------
        # Different default values for some fields
        #--------------------------------------------------
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = {}
        build_sql_function_kwargs['serialnumber_col'] = build_sql_function_kwargs.get('serialnumber_col', 'serialnumber')
        build_sql_function_kwargs['from_table_alias'] = build_sql_function_kwargs.get('from_table_alias', 'un_rin')
        build_sql_function_kwargs['datetime_col']     = build_sql_function_kwargs.get('datetime_col', 'valuesinterval')
        build_sql_function_kwargs['datetime_pattern'] = build_sql_function_kwargs.get('datetime_pattern', 
                                                                                       r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        build_sql_function_kwargs['date_col']         = build_sql_function_kwargs.get('date_col', 'aep_event_dt')
        #--------------------------------------------------
        return AMI_SQL.build_sql_ami_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_with_search_time_window, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            output_t_minmax            = output_t_minmax, 
            df_args                    = df_args
        )
        
        
    #****************************************************************************************************
    @staticmethod
    # COMMENTS FROM OLDER CODE, BUT STILL APPLICABLE HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #NEWEST VERSION
    # Can occur that len(df_outage['PREMISE_NB'].unique()) > len(df_end_dev_event['aep_premise_nb'].unique())
    # e.g. For OUTG_REC_NB 10143524.0 (which occured on 2017-09-16, PREMISE_NB 102920620 is inclued in df_outage.
    #      However, no end_device_events are found for aep_premise_nb = 102920620 on 2017-09-16
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def build_sql_end_events_for_outages(
        cols_of_interest          , 
        df_outage                 , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        date_only                 = False, 
        output_t_minmax           = False, 
        df_args                   = {}
    ):
        return AMIEndEvents_SQL.build_sql_end_events_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_outage, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            output_t_minmax            = output_t_minmax, 
            df_args                    = df_args
        )
   
    
    #****************************************************************************************************
    @staticmethod
    def build_sql_end_events_for_no_outages(
        cols_of_interest          , 
        df_mp_no_outg             , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        date_only                 = False,
        output_t_minmax           = False, 
        df_args                   = {}
    ):
        r"""
        """
        #--------------------------------------------------
        if df_args is None:
            df_args = {}
        df_args['t_search_min_col']    = df_args.get('t_search_min_col', 'start_date')
        df_args['t_search_max_col']    = df_args.get('t_search_max_col', 'end_date')
        df_args['addtnl_groupby_cols'] = df_args.get('addtnl_groupby_cols', None)
        df_args['mapping_to_ami']      = df_args.get('mapping_to_ami', DfToSqlMap(df_col='prem_nb', kwarg='premise_nbs', sql_col='aep_premise_nb'))
        df_args['mapping_to_mp']       = df_args.get('mapping_to_mp', {'state_cd':'states'})
        #-----

        return AMIEndEvents_SQL.build_sql_end_events_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_mp_no_outg, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            output_t_minmax            = output_t_minmax, 
            df_args                    = df_args
        )
        
    #****************************************************************************************************
    @staticmethod
    def build_sql_end_event_distinct_fields(
        date_0       , 
        date_1       , 
        fields       = ['serialnumber'], 
        are_datetime = False, 
        **kwargs
    ):
        r"""
        Intended use: find unique serial numbers recording some sort of end event between date_0 and date_1
        Default fields=['serialnumber'], but could also use, e.g., fields=['serialnumber', 'aep_premise_nb']
        """
        #-------------------------
        # First, make any necessary adjustments to kwargs
        kwargs['schema_name']      = kwargs.get('schema_name', 'meter_events')
        kwargs['table_name']       = kwargs.get('table_name', 'end_device_event')
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'EDE')       
        kwargs['date_col']         = kwargs.get('date_col', 'aep_event_dt')
        kwargs['datetime_col']     = kwargs.get('datetime_col', 'valuesinterval')
        kwargs['datetime_pattern'] = kwargs.get('datetime_pattern', 
                                                 r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        #-------------------------
        if are_datetime:
            kwargs['datetime_range'] = [date_0, date_1]
        else:
            kwargs['date_range'] = [date_0, date_1]
        cols_of_interest = fields
        #-------------------------
        sql = AMI_SQL.build_sql_ami(cols_of_interest=cols_of_interest, **kwargs)
        sql.sql_select.select_distinct=True
        #-------------------------
        return sql