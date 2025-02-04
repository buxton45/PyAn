#!/usr/bin/env python

r"""
Holds AMIUsgInst_SQL class.  See AMIUsgInst_SQL.AMIUsgInst_SQL for more information.
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
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
#--------------------------------------------------

class AMIUsgInst_SQL(AMI_SQL):
    def __init__(self):
        self.sql_query = None
        
    #****************************************************************************************************
    @staticmethod
    def build_sql_usg_inst(
        cols_of_interest, 
        **kwargs
    ):
        r"""
        See AMI_SQL.add_ami_where_statements for my updated list of acceptable kwargs with respect to the where statement.

        Acceptable kwargs:
          - date_range
            - tuple with two string elements, e.g., ['2021-01-01', '2021-04-01']
          - date_col
            - default: aep_read_dt (NOTE: not aep_usage_dt, as in AMI_SQL.add_ami_where_statements)

          - datetime_range
            - tuple with two string elements, e.g., ['2021-01-01 00:00:00', '2021-04-01 12:00:00']
          - datetime_col
            - default: aep_readtime (NOTE: not starttimeperiod, as in AMI_SQL.add_ami_where_statements)
          - datetime_pattern
            - Regex pattern used to convert string to a form that SQL/Athena/Oracle can convert to a TIMESTAMP
            - default: r"([0-9]{4}-[0-9]{2}-[0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*"

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
            - default: usage_instantaneous
          - table_name
            - default: inst_msr_consume
        """
        #-------------------------
        # First, make any necessary adjustments to kwargs
        kwargs['schema_name']      = kwargs.get('schema_name', 'usage_instantaneous')
        kwargs['table_name']       = kwargs.get('table_name', 'inst_msr_consume')
        #kwargs['from_table_alias'] = kwargs.get('from_table_alias', None)
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'USG_INST')       
        kwargs['date_col']         = kwargs.get('date_col', 'aep_read_dt')
        kwargs['datetime_col']     = kwargs.get('datetime_col', 'aep_readtime')
        kwargs['datetime_pattern'] = kwargs.get('datetime_pattern', 
                                                 r"([0-9]{4}-[0-9]{2}-[0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        #-------------------------
        sql = AMI_SQL.build_sql_ami(cols_of_interest=cols_of_interest, **kwargs)
        return sql


    #****************************************************************************************************
    @staticmethod
    def build_sql_usg_inst_for_df_with_search_time_window(
        cols_of_interest           , 
        df_with_search_time_window , 
        build_sql_function         = None, 
        build_sql_function_kwargs  = {}, 
        sql_alias_base             = 'USG_', 
        max_n_prem_per_outg        = None, 
        join_mp_args               = False, 
        df_args                    = {}
    ):
        r"""
        NOTE: python won't let me set the default value to build_sql_function=AMIUsgInst.build_sql_usg_inst.
              Thus, the workaround is to set the default to None, then set it to 
              AMIUsgInst.build_sql_usg_inst if it is None

        TODO: Investigate what an appropriate default value for max_n_prem_per_outg would be....
              Probably something large, like 1000.  But, should find outages with huge numbers of customers
              affected to find the right number.
        """
        #--------------------------------------------------
        if build_sql_function is None:
            build_sql_function=AMIUsgInst_SQL.build_sql_usg_inst
        #--------------------------------------------------
        # Different default values for some fields (specifically for kwargs)
        #--------------------------------------------------
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = {}
        build_sql_function_kwargs['serialnumber_col'] = build_sql_function_kwargs.get('serialnumber_col', 'serialnumber')
        build_sql_function_kwargs['from_table_alias'] = build_sql_function_kwargs.get('from_table_alias', 'un_rin') 
        build_sql_function_kwargs['datetime_col']     = build_sql_function_kwargs.get('datetime_col', 'aep_readtime')
        build_sql_function_kwargs['datetime_pattern'] = build_sql_function_kwargs.get(
          'datetime_pattern', 
          r"([0-9]{4}-[0-9]{2}-[0-9]{2}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
        )
        build_sql_function_kwargs['date_col']         = build_sql_function_kwargs.get('date_col', 'aep_read_dt')
        #--------------------------------------------------
        return AMI_SQL.build_sql_ami_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_with_search_time_window, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            df_args                    = df_args
        )
    
    
    #****************************************************************************************************
    @staticmethod            
    def build_sql_usg_inst_for_outages(
        cols_of_interest          , 
        df_outage                 , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        df_args                   = {}
    ):
        return AMIUsgInst_SQL.build_sql_usg_inst_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_outage, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            df_args                    = df_args
        )
        
        
    #****************************************************************************************************
    @staticmethod
    def build_sql_usg_inst_for_no_outages(
        cols_of_interest          , 
        df_mp_no_outg             , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        date_only                 = False, 
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

        return AMIUsgInst_SQL.build_sql_usg_inst_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_mp_no_outg, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            df_args                    = df_args
        )