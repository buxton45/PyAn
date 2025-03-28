#!/usr/bin/env python

r"""
Holds MeterPremise class.  See MeterPremise.MeterPremise for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys
import re
import copy
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_dtype
import datetime
import warnings
#--------------------------------------------------
from GenAn import GenAn
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from SQLSelect import SQLSelect
from SQLFrom import SQLFrom
from SQLWhere import SQLWhereElement, SQLWhere
from SQLGroupBy import SQLGroupBy
from SQLQuery import SQLQuery
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
from DataFrameSubsetSlicer import DataFrameSubsetSingleSlicer as DFSingleSlicer
#--------------------------------------------------

class MeterPremise(GenAn):
    r"""
    class MeterPremise documentation
    """
    def __init__(
        self, 
        df_construct_type         = DFConstructType.kRunSqlQuery,
        contstruct_df_args        = None, 
        init_df_in_constructor    = True, 
        build_sql_function        = None, 
        build_sql_function_kwargs = None, 
        max_n_prem_nbs            = 10000, 
        **kwargs
    ):
        r"""
        if df_construct_type==DFConstructType.kReadCsv or DFConstructType.kReadCsv:
          contstruct_df_args needs to have at least 'file_path'
        if df_construct_type==DFConstructType.kRunSqlQuery:
          contstruct_df_args needs at least 'conn_db'     
          
        max_n_prem_nbs
            If the number of premises is greater than max_n_prem_nbs, the building of the DF will be
              instructed to be done in batches, if not already set.
        """
        #--------------------------------------------------
        # First, set self.build_sql_function and self.build_sql_function_kwargs
        # and call base class's __init__ method
        #---------------
        self.build_sql_function = (build_sql_function if build_sql_function is not None 
                                   else MeterPremise.build_sql_meter_premise)
        #---------------
        self.build_sql_function_kwargs = (build_sql_function_kwargs if build_sql_function_kwargs is not None 
                                          else {})
        #--------------------------------------------------
        # TODO may want to offload this (and subsequent reduction of self.df at end of constructor) to a separate 
        # function, mainly for case where DF is not initialized in the constructor
        if max_n_prem_nbs:
            # See MeterPremise.build_sql_meter_premise for possible_premise_nbs_kwargs 
            possible_premise_nbs_kwargs = ['premise_nbs', 'premise_nb', 'aep_premise_nbs', 'aep_premise_nb']
            found_premise_nbs_kwargs    = [x for x in self.build_sql_function_kwargs if x in possible_premise_nbs_kwargs]
            assert(len(found_premise_nbs_kwargs)<=1)
            if found_premise_nbs_kwargs:
                self.build_sql_function_kwargs['field_to_split'] = self.build_sql_function_kwargs.get('field_to_split', found_premise_nbs_kwargs[0])
                self.build_sql_function_kwargs['batch_size']     = self.build_sql_function_kwargs.get('batch_size', max_n_prem_nbs)
        #--------------------------------------------------
        super().__init__(
            df_construct_type         = df_construct_type, 
            contstruct_df_args        = contstruct_df_args, 
            init_df_in_constructor    = init_df_in_constructor, 
            build_sql_function        = self.build_sql_function, 
            build_sql_function_kwargs = self.build_sql_function_kwargs, 
            **kwargs
        )
        
    #****************************************************************************************************
    def get_conn_db(self):
        return Utilities.get_athena_prod_aws_connection()
    
    def get_default_cols_and_types_to_convert_dict(self):
        cols_and_types_to_convert_dict = None
        return cols_and_types_to_convert_dict
    
    def get_full_default_sort_by(self):
        full_default_sort_by = None
        return full_default_sort_by


    #****************************************************************************************************  
    @staticmethod
    def combine_where_elements(
        sql_where, 
        wheres_to_combine, 
        field_descs_dict
    ):
        r"""
        Designed for internal use by MeterPremise.build_sql_meter_premise

        sql_where:
            SQLWhere object to be altered

        wheres_to_combine:
            Directs which where elements to combine.
            This needs to be a dict or list of dicts.
            A list can be utilized when multiple groups are to be combined.
            The keys in the dict(s) must be:
                to_combine: REQUIRED
                join_operator: optional, set to 'OR' if no argument supplied.

            EX. 1
                wheres_to_combine = dict(to_combine=['date_range', 'serial_numbers'], join_operator='OR')
            EX. 2
                wheres_to_combine = [
                    dict(to_combine=['date_range', 'serial_numbers'], join_operator='OR'), 
                    dict(to_combine=['datetime_range', 'premise_nbs'], join_operator='OR')
                ]

        field_descs_dict:
            A dict whose keys are the keyword arguments utilized in add_ami_where_statements(sql_where, **kwargs)
            and whose values are the corresponding field_desc
        """
        #-------------------------
        assert(isinstance(field_descs_dict, dict))
        assert(Utilities.is_object_one_of_types(wheres_to_combine, [dict, list]))
        #--------------------------------------------------
        if isinstance(wheres_to_combine, list):
            for wheres_to_combine_i in wheres_to_combine:
                sql_where = MeterPremise.combine_where_elements(
                    sql_where         = sql_where, 
                    wheres_to_combine = wheres_to_combine_i, 
                    field_descs_dict  = field_descs_dict
                )
            return sql_where
        #--------------------------------------------------
        assert(isinstance(wheres_to_combine, dict))
        assert('to_combine' in wheres_to_combine.keys())
        wheres_to_combine['join_operator'] = wheres_to_combine.get('join_operator', 'OR')
        #-------------------------
        idxs_to_combine = []
        for to_comb_i in wheres_to_combine['to_combine']:
            fd_i  = field_descs_dict[to_comb_i]
            idx_i = sql_where.find_idx_of_approx_element_in_collection_dict(fd_i)
            assert(idx_i>-1)
            idxs_to_combine.append(idx_i)
        #-----
        sql_where.combine_where_elements(
            idxs_to_combine    = idxs_to_combine, 
            join_operator      = wheres_to_combine['join_operator'], 
            close_gaps_in_keys = True
        )
        #-------------------------
        return sql_where  
    

    @staticmethod
    def add_inst_rmvl_ts_where_statements(
        sql_where        , 
        datetime_ranges  , 
        from_table_alias = None, 
        inst_ts_col      = 'inst_ts', 
        rmvl_ts_col      = 'rmvl_ts', 
        datetime_pattern = r"([0-9]{2})/([0-9]{2})/([0-9]{4}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*", 
        datetime_replace = r"$3-$1-$2 $4"
    ):
        r"""
        Just add_inst_rmvl_ts_where_statement expanded to accept multiple datetime ranges!
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(datetime_ranges, [list, tuple]))
        assert(Utilities.is_list_nested(lst=datetime_ranges, enforce_if_one_all=True))
        assert(Utilities.are_list_elements_lengths_homogeneous(lst=datetime_ranges, length=2))
        #--------------------------------------------------
        for datetime_range_i in datetime_ranges:
            n_where = len(sql_where.collection_dict)
            #-----
            sql_where = MeterPremise.add_inst_rmvl_ts_where_statement(
                sql_where        = sql_where, 
                datetime_range   = datetime_range_i, 
                from_table_alias = from_table_alias, 
                inst_ts_col      = inst_ts_col, 
                rmvl_ts_col      = rmvl_ts_col, 
                datetime_pattern = datetime_pattern, 
                datetime_replace = datetime_replace
            )
            #-----
            # Two elements should have been added to sql_where at each iteration
            assert(len(sql_where.collection_dict) - n_where == 2)
            #-----
            sql_where.combine_last_n_where_elements(
                last_n             = 2, 
                join_operator      = 'AND', 
                close_gaps_in_keys = True
            )
        
        #--------------------------------------------------
        sql_where.combine_last_n_where_elements(
            last_n             = len(datetime_ranges), 
            join_operator      = 'OR', 
            close_gaps_in_keys = True
        )
        #--------------------------------------------------
        return sql_where
        
        
    @staticmethod
    def add_inst_rmvl_ts_where_statement(
        sql_where        , 
        datetime_range   , 
        from_table_alias = None, 
        inst_ts_col      = 'inst_ts', 
        rmvl_ts_col      = 'rmvl_ts', 
        datetime_pattern = r"([0-9]{2})/([0-9]{2})/([0-9]{4}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*", 
        datetime_replace = r"$3-$1-$2 $4"
    ):
        r"""
        Add a where statement to enforce that a meter was installed during the time period defined in datetime_range.
        Essentially, this enforces that a SN was installed before datetime_range[0] and removed after datetime_range[1].
        Since meters which are currently still active have rmvl_ts values which are empty, the logic is slightly more involved.

        This will insert a WHERE statement similar to the following:
            WHERE CAST(regexp_replace(MP.inst_ts, '([0-9]{2})/([0-9]{2})/([0-9]{4}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*', '$3-$1-$2 $4') AS TIMESTAMP) <= TIMESTAMP '2023-01-01 12:00:00'
            AND   (
                (
                    MP.rmvl_ts IS NULL OR 
                    MP.rmvl_ts = ''
                ) OR 
                CAST(regexp_replace(MP.rmvl_ts, '([0-9]{2})/([0-9]{2})/([0-9]{4}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*', '$3-$1-$2 $4') AS TIMESTAMP) >= TIMESTAMP '2023-05-01 12:00:00'
            )
            
        !!! UNLESS !!! datetime_pattern is None or datetime_replace is None, in which case the WHERE statement will be similar to the following:
            WHERE MP.inst_ts <= '2023-01-01 12:00:00'
            AND   (
                (
                    MP.rmvl_ts IS NULL OR 
                    MP.rmvl_ts = ''
                ) OR 
                MP.rmvl_ts >= '2023-05-01 12:00:00'
            )

        This is essentially the SQL implementation of:
            active_SNs_at_time = df_mp_hist[(df_mp_hist[df_mp_install_time_col]<=pd.to_datetime(dt_0)) & 
                                            (df_mp_hist[df_mp_removal_time_col].fillna(pd.Timestamp.max)>pd.to_datetime(dt_1))]
          OR
            active_SNs_at_time = df_mp_hist[(df_mp_hist[df_mp_install_time_col]<=pd.to_datetime(dt_0)) & 
                                            ((df_mp_hist[df_mp_removal_time_col].isna()) | (df_mp_hist[df_mp_removal_time_col]>pd.to_datetime(dt_1)))]

        datetime_range:
            This should typically be a list/tuple of length-2 defining the range.
            HOWEVER, functionality has been expanded to accept a list of such ranges. 
        """
        #--------------------------------------------------
        # If datetime_range is actually a list of such object, ship off to add_inst_rmvl_ts_where_statements
        assert(Utilities.is_object_one_of_types(datetime_range, [list, tuple]))
        if Utilities.is_list_nested(lst=datetime_range, enforce_if_one_all=True) and len(datetime_range)>1:
            sql_where = MeterPremise.add_inst_rmvl_ts_where_statements(
                sql_where        = sql_where, 
                datetime_ranges  = datetime_range, 
                from_table_alias = from_table_alias, 
                inst_ts_col      = inst_ts_col, 
                rmvl_ts_col      = rmvl_ts_col, 
                datetime_pattern = datetime_pattern, 
                datetime_replace = datetime_replace
            )
            return sql_where

        #--------------------------------------------------
        if Utilities.is_list_nested(lst=datetime_range, enforce_if_one_all=True):
            assert(len(datetime_range)==1)
            datetime_range = datetime_range[0]
        assert(len(datetime_range)==2)
        #--------------------------------------------------
        if from_table_alias:
            inst_ts_col = f'{from_table_alias}.{inst_ts_col}'
            rmvl_ts_col = f'{from_table_alias}.{rmvl_ts_col}'
        #-------------------------
        if datetime_pattern is None or datetime_replace is None:
            inst_ts_field_desc   = r"CAST({} AS TIMESTAMP)".format(inst_ts_col)
            rmvl_ts_field_desc_2 = r"CAST({} AS TIMESTAMP)".format(rmvl_ts_col)
        else:
            inst_ts_field_desc   = r"CAST(regexp_replace({}, '{}', '{}') AS TIMESTAMP)".format(inst_ts_col, datetime_pattern, datetime_replace)
            rmvl_ts_field_desc_2 = r"CAST(regexp_replace({}, '{}', '{}') AS TIMESTAMP)".format(rmvl_ts_col, datetime_pattern, datetime_replace)
        #-------------------------
        # inst_ts
        sql_where.add_where_statement(
            field_desc          = inst_ts_field_desc, 
            comparison_operator = '<=', 
            value               = datetime_range[0], 
            needs_quotes        = True, 
            table_alias_prefix  = None, 
            is_timestamp        = True, 
            idx                 = None
        )
        #-------------------------
        # rmvl_ts 1
        sql_where.add_where_statement(
            field_desc          = rmvl_ts_col, 
            comparison_operator = 'IS', 
            value               = 'NULL', 
            needs_quotes        = False, 
            table_alias_prefix  = None, 
            is_timestamp        = False, 
            idx                 = None
        )
        #-----
        sql_where.add_where_statement(
            field_desc          = rmvl_ts_col, 
            comparison_operator = '=', 
            value               = "''", 
            needs_quotes        = False, 
            table_alias_prefix  = None, 
            is_timestamp        = False, 
            idx                 = None
        )
        #-------------------------
        # rmvl_ts 2
        sql_where.add_where_statement(
            field_desc          = rmvl_ts_field_desc_2, 
            comparison_operator = '>=', 
            value               = datetime_range[1], 
            needs_quotes        = True, 
            table_alias_prefix  = None, 
            is_timestamp        = True, 
            idx                 = None
        )
        #-----
        sql_where.combine_last_n_where_elements(
            last_n             = 3, 
            join_operator      = 'OR', 
            close_gaps_in_keys = True
        )
        #-------------------------
        return sql_where
        
        
    @staticmethod
    def add_opco_nm_to_mp_sql(
        mp_sql                 , 
        opcos                  = None, 
        comp_cols              = ['opco_nm', 'opco_nb'], 
        comp_alias             = 'COMP', 
        join_type              = 'LEFT', 
        include_comp_in_select = True, 
        return_alias           = None
    ):
        r"""
        In order to add opco_nm to default.meter_premise, we must join with default.company.
        mp_sql must be a SQLQuery object.
        -------
        WARNING: This procedure will collapse mp_sql from SQLQuery object down to a simple string!
        -------
        opcos:
            If not None, should be a list of opcos, (e.g., ['ap','ky','oh','im','pso','swp','tx']
        
        """
        #-------------------------
        if opcos is None and not include_comp_in_select:
            return_stmnt = mp_sql.get_sql_statement()
        else:
            if comp_cols is None:
                comp_cols = ['opco_nm', 'opco_nb']
            assert(isinstance(comp_cols, list))
            # Make sure 'opco_nm' and 'opco_nb' are in select for default.company
            # NOTE: e.g., if one is only using opco_nm in the where statement (via opcos argument) then
            #         don't necessarily need opco_nm/opco_nb in final output
            comp_cols_full = list(set(comp_cols+['opco_nm', 'opco_nb']))
            #-----
            sql_opco = "SELECT {}\nFROM default.company".format(
                Utilities_sql.join_list(comp_cols_full, quotes_needed=False)
            )
            #-----
            sql_opco = Utilities_sql.prepend_tabs_to_each_line(
                input_statement   = sql_opco, 
                n_tabs_to_prepend = 1
            )
            #-------------------------
            if not mp_sql.sql_from.alias:
                mp_sql.sql_from.alias = Utilities.generate_random_string(str_len=5, letters='letter_only')    
            #-------------------------
            if include_comp_in_select:
                join_cols_to_add_to_select = comp_cols
            else:
                join_cols_to_add_to_select = None
            #-----
            join_company_args = dict(
                join_type                  = join_type, 
                join_table                 = None, 
                join_table_alias           = comp_alias, 
                orig_table_alias           = mp_sql.sql_from.alias, 
                list_of_columns_to_join    = [['co_cd_ownr', 'opco_nb']], 
                idx                        = None, 
                run_check                  = False, 
                join_cols_to_add_to_select = join_cols_to_add_to_select
            )
            mp_sql.build_and_add_join(**join_company_args)
            #-------------------------
            if opcos is not None:
                assert(Utilities.is_object_one_of_types(opcos, [list, str]))
                if isinstance(opcos, str):
                    opcos = [opcos]
                mp_sql.sql_where.addtnl_info['field_descs_dict']['opcos'] = 'opco_nm'
                mp_sql.sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = mp_sql.sql_where, 
                    field_desc         = 'opco_nm', 
                    value              = opcos, 
                    needs_quotes       = True, 
                    table_alias_prefix = comp_alias, 
                    idx                = None
                )        
            #-------------------------
            return_stmnt = "WITH {} AS (\n{}\n)\n".format(comp_alias, sql_opco)
            return_stmnt = return_stmnt + mp_sql.get_sql_statement()
        #-------------------------
        if return_alias is not None:
            return_stmnt = Utilities_sql.prepend_tabs_to_each_line(return_stmnt, n_tabs_to_prepend=1)
            return_stmnt = f"{return_alias} AS (\n{return_stmnt}\n)"
        #-------------------------
        return return_stmnt
    
    
    @staticmethod
    def build_sql_meter_premise(
        cols_of_interest=None, 
        **kwargs
    ):
        r"""
        IMPORTANT: If this ever gets converted to inherit from GenAn and AMI_SQL, then datetime_range will need to be popped off
                     kwargs before being fed into AMI_SQL.build_sql_ami because datetime range is handled different here, and taken care
                     of by MeterPremise.add_inst_rmvl_ts_where_statement.
                   See EEMSP.build_sql_eemsp_ME
        
        Acceptable kwargs:
          - trsf_pole_nb(s)
          - trsf_pole_nb_col
            - default: trsf_pole_nb
        
          - premise_nb(s) (or, aep_premise_nb(s) will work also)
          - premise_nb_col
            - default: prem_nb
        
          NOTE: Use of serial_numbers instead of mfr_devc_ser_nbr to be consistent
                with other similar build methods (e.g., build_sql_usg, etc.)
          - serial_number(s) (or, mfr_devc_ser_nbr(s) will work also)
          - serial_number_col
            - default: mfr_devc_ser_nbr
          
          - state(s)
          - state_col
            - default: state_cd

          - city(ies)
          - city_col
            - default: serv_city_ad
            
          - srvc_addr_2_nm(s)
          - srvc_addr_2_nm_col
            - default: srvc_addr_2_nm
            
          - technology_tx(s)
          - technology_tx_col
            - default: technology_tx
            
          - curr_cust_nm(s)
          - curr_cust_nm_col
            - default: curr_cust_nm
            
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
          
          - datetime_range
          - inst_ts_col
            - default: inst_ts
          - rmvl_ts_col
            - default: rmvl_ts
          - datetime_pattern
            - default: r"([0-9]{2})/([0-9]{2})/([0-9]{4}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
          - datetime_replace
            - default: r"$3-$1-$2 $4"
          ***** For more info on datetime/inst_ts/rmvl_ts, see MeterPremise.add_inst_rmvl_ts_where_statement *****
          
          - opcos
            - IMPORTANT: If setting OPCOs, then add_opco_nm_to_mp_sql will be utilized, and the returned object will be a string,
                           NOT a SQLQuery object!
          
          - schema_name
            - default: default
          - table_name
            - default: meter_premise
            
          - alias (query alias)
          
          TODO: How can I adjust this function to allow e.g., trsf_pole_nbs NOT IN...
                 - For now, easiest method will be to use standard build_sql_meter_premise then
                   the utilize the change_comparison_operator_of_element/change_comparison_operator_of_element_at_idx
                   method of SQLWhere to change mp_sql.sql_where after mp_sql is returned
                 - The second option would be to input trsf_pole_nbs of type SQLWhereElement with all correct
                   attributes set before inputting
                   
          - addtnl_cols_of_interest
        """
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = TableInfos.MeterPremise_TI.std_columns_of_interest
        addtnl_cols_of_interest = kwargs.pop('addtnl_cols_of_interest', None)
        if addtnl_cols_of_interest is not None:
            assert(isinstance(addtnl_cols_of_interest, list))
            # Only want to include those not already included
            addtnl_cols_of_interest = list(set(addtnl_cols_of_interest).difference(set(cols_of_interest)))
            if addtnl_cols_of_interest:
                cols_of_interest.extend(addtnl_cols_of_interest)
        #**************************************************
        # SELECT STATEMENT
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'MP')
        from_table_alias           = kwargs['from_table_alias']
        #-----
        sql_select = SQLSelect(cols_of_interest, global_table_alias_prefix=from_table_alias)
        #**************************************************
        # FROM STATEMENT
        schema_name = kwargs.get('schema_name', 'default')
        table_name  = kwargs.get('table_name', 'meter_premise')
        sql_from    = SQLFrom(schema_name, table_name, alias=from_table_alias)
        #**************************************************
        # WHERE STATEMENT
        sql_where = SQLWhere()
        
        #*******************************************************
        # field_descs_dict needed if any where elements are to be combined
        field_descs_dict = dict()
        
        #***** STATES **************************************************
        state_col = kwargs.get('state_col', 'state_cd')
        assert(not('states' in kwargs and 'state' in kwargs))
        states = kwargs.get('states', kwargs.get('state', None))
        if states is not None:
            field_descs_dict['states' if 'states' in kwargs else 'state'] = state_col
            if isinstance(states, SQLWhereElement):
                sql_where.add_where_statement(states)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = state_col, 
                    value              = states, 
                    table_alias_prefix = from_table_alias
                )
        #***** CITIES **************************************************
        city_col = kwargs.get('city_col', 'serv_city_ad')
        assert(not('cities' in kwargs and 'city' in kwargs))
        cities = kwargs.get('cities', kwargs.get('city', None))
        if cities is not None:
            field_descs_dict['cities' if 'cities' in kwargs else 'city'] = city_col
            if isinstance(cities, SQLWhereElement):
                sql_where.add_where_statement(cities)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = city_col, 
                    value              = cities, 
                    table_alias_prefix = from_table_alias
                )
        #***** TRANSFORMER POLE NUMBERS **************************************************
        trsf_pole_nb_col = kwargs.get('trsf_pole_nb_col', 'trsf_pole_nb')
        assert(not('trsf_pole_nbs' in kwargs and 'trsf_pole_nb' in kwargs))
        trsf_pole_nbs = kwargs.get('trsf_pole_nbs', kwargs.get('trsf_pole_nb', None))
        if trsf_pole_nbs is not None:
            field_descs_dict['trsf_pole_nbs' if 'trsf_pole_nbs' in kwargs else 'trsf_pole_nb'] = trsf_pole_nb_col
            if isinstance(trsf_pole_nbs, SQLWhereElement):
                sql_where.add_where_statement(trsf_pole_nbs)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = trsf_pole_nb_col, 
                    value              = trsf_pole_nbs, 
                    table_alias_prefix = from_table_alias
                )
        #***** PREMISE NUMBERS **************************************************
        premise_nb_col              = kwargs.get('premise_nb_col', 'prem_nb')
        possible_premise_nbs_kwargs = ['premise_nbs', 'premise_nb', 'aep_premise_nbs', 'aep_premise_nb']
        found_premise_nbs_kwargs    = [x for x in kwargs if x in possible_premise_nbs_kwargs]
        assert(len(found_premise_nbs_kwargs)<=1)
        if found_premise_nbs_kwargs:
            premise_nbs = kwargs[found_premise_nbs_kwargs[0]]
        else:
            premise_nbs = None
        if premise_nbs is not None:
            field_descs_dict[found_premise_nbs_kwargs[0]] = premise_nb_col
            if isinstance(premise_nbs, SQLWhereElement):
                sql_where.add_where_statement(premise_nbs)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = premise_nb_col, 
                    value              = premise_nbs, 
                    table_alias_prefix = from_table_alias
                )
        #***** SERIAL NUMBERS **************************************************
        serial_number_col              = kwargs.get('serial_number_col', 'mfr_devc_ser_nbr')      
        possible_serial_numbers_kwargs = ['serial_numbers', 'serial_number', 'mfr_devc_ser_nbrs', 'mfr_devc_ser_nbr']
        found_serial_numbers_kwargs    = [x for x in kwargs if x in possible_serial_numbers_kwargs]
        assert(len(found_serial_numbers_kwargs)<=1)
        if found_serial_numbers_kwargs:
            serial_numbers = kwargs[found_serial_numbers_kwargs[0]]
        else:
            serial_numbers = None        
        if serial_numbers is not None:
            field_descs_dict[found_serial_numbers_kwargs[0]] = serial_number_col
            if isinstance(serial_numbers, SQLWhereElement):
                sql_where.add_where_statement(serial_numbers)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = serial_number_col, 
                    value              = serial_numbers, 
                    table_alias_prefix = from_table_alias
                )
                
        #***** SERVICE ADDRESS 2 NAMES **************************************************
        srvc_addr_2_nm_col = kwargs.get('srvc_addr_2_nm_col', 'srvc_addr_2_nm')
        assert(not('srvc_addr_2_nms' in kwargs and 'srvc_addr_2_nm' in kwargs))
        srvc_addr_2_nms = kwargs.get('srvc_addr_2_nms', kwargs.get('srvc_addr_2_nm', None))
        if srvc_addr_2_nms is not None:
            field_descs_dict['srvc_addr_2_nms' if 'srvc_addr_2_nms' in kwargs else 'srvc_addr_2_nm'] = srvc_addr_2_nm_col
            if isinstance(srvc_addr_2_nms, SQLWhereElement):
                sql_where.add_where_statement(srvc_addr_2_nms)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = srvc_addr_2_nm_col, 
                    value              = srvc_addr_2_nms, 
                    table_alias_prefix = from_table_alias
                )
                
        #***** technology_tx **************************************************
        technology_tx_col = kwargs.get('technology_tx_col', 'technology_tx')
        assert(not('technology_txs' in kwargs and 'technology_tx' in kwargs))
        technology_txs = kwargs.get('technology_txs', kwargs.get('technology_tx', None))
        if technology_txs is not None:
            field_descs_dict['technology_txs' if 'technology_txs' in kwargs else 'technology_tx'] = technology_tx_col
            if isinstance(technology_txs, SQLWhereElement):
                sql_where.add_where_statement(technology_txs)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = technology_tx_col, 
                    value              = technology_txs, 
                    table_alias_prefix = from_table_alias
                )
                
        #***** curr_cust_nm **************************************************
        curr_cust_nm_col = kwargs.get('curr_cust_nm_col', 'curr_cust_nm')
        assert(not('curr_cust_nms' in kwargs and 'curr_cust_nm' in kwargs))
        curr_cust_nms = kwargs.get('curr_cust_nms', kwargs.get('curr_cust_nm', None))
        if curr_cust_nms is not None:
            field_descs_dict['curr_cust_nms' if 'curr_cust_nms' in kwargs else 'curr_cust_nm'] = curr_cust_nm_col
            if isinstance(curr_cust_nms, SQLWhereElement):
                sql_where.add_where_statement(curr_cust_nms)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = curr_cust_nm_col, 
                    value              = curr_cust_nms, 
                    table_alias_prefix = from_table_alias
                )


        #**************************************************
        wheres_to_combine = kwargs.get('wheres_to_combine', None)
        if wheres_to_combine is not None:
            sql_where = MeterPremise.combine_where_elements(
                sql_where         = sql_where, 
                wheres_to_combine = wheres_to_combine, 
                field_descs_dict  = field_descs_dict
            )        
        sql_where.addtnl_info['field_descs_dict'] = field_descs_dict
        
        #***** DATETIME **************************************************
        inst_ts_col      = kwargs.get('inst_ts_col', 'inst_ts')
        rmvl_ts_col      = kwargs.get('rmvl_ts_col', 'rmvl_ts')
        datetime_range   = kwargs.get('datetime_range', None)
        datetime_pattern = kwargs.get('datetime_pattern', r"([0-9]{2})/([0-9]{2})/([0-9]{4}) ([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        datetime_replace = kwargs.get('datetime_replace', r"$3-$1-$2 $4")
        if datetime_range is not None:
            sql_where = MeterPremise.add_inst_rmvl_ts_where_statement(
                sql_where        = sql_where, 
                datetime_range   = datetime_range, 
                from_table_alias = from_table_alias, 
                inst_ts_col      = inst_ts_col, 
                rmvl_ts_col      = rmvl_ts_col, 
                datetime_pattern = datetime_pattern, 
                datetime_replace = datetime_replace
            )
        
        #**************************************************
        #**************************************************
        mp_sql = SQLQuery(
            sql_select = sql_select, 
            sql_from   = sql_from, 
            sql_where  = sql_where, 
            alias      = kwargs.get('alias', None)
        )
                         
        #**************************************************
        #**************************************************
        assert(not('opcos' in kwargs and 'opco' in kwargs))
        opcos = kwargs.get('opcos', kwargs.get('opco', None))
        if opcos is not None:
            sql_stmnt = MeterPremise.add_opco_nm_to_mp_sql(
                mp_sql     = mp_sql, 
                opcos      = opcos, 
                comp_cols  = ['opco_nm'], 
                comp_alias = 'COMP', 
                join_type  = 'LEFT', 
            )
            return sql_stmnt
        else:
            return mp_sql
        

    #****************************************************************************************************
    @staticmethod
    def combine_rmvl_ts_nat_entries(
        df_mp                   , 
        df_mp_serial_number_col = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col       = 'prem_nb', 
        df_mp_install_time_col  = 'inst_ts', 
        df_mp_removal_time_col  = 'rmvl_ts', 
    ):
        r"""
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        THIS IS BASICALLY OBSOLETE.  USE THE MORE GENERAL MeterPremise.drop_approx_mp_duplicates
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        combine_rmvl_ts_nat_entries = Combine rmvl_ts NaT entries.
        I have found instances where entries having rmvl_ts=NaT are exactly the same (except for e.g., inst_kind_cd, mtr_kind_cd) 
          for my intents.
        If one merges such a df_mp with another df (e.g., see merge_df_with_active_mp), this will cause undesired duplicates.
        This function solves such an issue by eliminating the duplicates in df_mp.
          For each nearly duplicate group, that with the smallest inst_ts is kept.
        """
        #-------------------------
        # NOTE: Originally included trsf_pole_nb (if found in df_mp) in necessary_mp_cols below.
        #       However, I believe now that I don't actually want trsf_pole_nb included in necessary_mp_cols, as this would cause and 
        #         entries with trsf_pole_nb=NaN to be excluded (because groupby doesn't capture NaNs), which we don't want, as the historical
        #         default.meter_premise_hist does not have trsf_pole_nbs.
        #         So, if dp_mp were a concatenation of df_mp_curr and df_mp_hist (or just df_mp_hist), all entries from df_mp_hist would
        #         be excluded if trsf_pole_nb were in necessary_mp_cols
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]

        #-------------------------
        # Only need to run the selection procedure on entries with rmvl_ts=NaT
        rmvl_nat_mask = df_mp['rmvl_ts'].isna()
        df_mp_notna   = df_mp[~rmvl_nat_mask]
        df_mp_na      = df_mp[rmvl_nat_mask]
        #-------------------------
        # Want to group by all df_mp_na columns in necessary_mp_cols EXCEPT inst_ts and rmvl_ts
        gpby_cols = [x for x in necessary_mp_cols 
                     if x not in [df_mp_install_time_col, df_mp_removal_time_col]]
        #-------------------------
        # For each group in gpd_na, keep only the entry with the minimum inst_ts
        # NOTE: New method using idxmin() MUCH faster than old method (commented out below).
        #       Plus, old method would still have degeneracy is multiple group members all had
        #         df_mp_install_time_col equal to the group min!
        # NOTE: To be sure the new method functions correctly, it is important for the index of
        #       df_mp_na to be unique.  So, call reset_index(drop=False)
        #-----
        #gpd_na = df_mp_na.groupby(gpby_cols)
        #df_mp_na = gpd_na.apply(lambda x: x[x[df_mp_install_time_col]==x[df_mp_install_time_col].min()])
        #df_mp_na = df_mp_na.reset_index(drop=True)
        #-------------------------
        cols_org        = df_mp_na.columns.tolist()
        idx_level_names = df_mp_na.index.names
        # When calling reset_index(drop=False), if the level has a name, it will be used as the column, 
        #   otherwise f'level_{idx_level}' is used
        # NOTE: If only a single, un-named index level, the column will be named 'index', not 'level_0'
        idx_col_names = [x if x is not None else f'level_{i}' for i,x in enumerate(idx_level_names)]
        if len(idx_col_names)==1 and idx_col_names[0]=='level_0':
            idx_col_names = ['index']
        #-----
        df_mp_na = df_mp_na.reset_index(drop=False)
        min_idxs = df_mp_na.groupby(gpby_cols)[df_mp_install_time_col].idxmin()
        df_mp_na = df_mp_na.loc[min_idxs]
        #-----
        # Make sure the idx_col_names are contained in the columns
        assert(len(set(idx_col_names).difference(set(df_mp_na.columns)))==0)
        # Set the index values (and level names) equal to what they originally were
        df_mp_na = df_mp_na.set_index(idx_col_names)
        df_mp_na.index.names = idx_level_names
        #-----
        # Make sure the columns are as expected
        assert(len(set(df_mp_na.columns).symmetric_difference(set(cols_org)))==0)
        #--------------------------------------------------
        # Join back together df_mp_notna and df_mp_na
        assert(len(set(df_mp_na.columns).symmetric_difference(set(df_mp_notna.columns)))==0)
        df_mp = pd.concat([df_mp_notna, df_mp_na])
        #-------------------------
        # Sort by necessary_mp_cols just to give it some sort of order
        df_mp = df_mp.sort_values(by=necessary_mp_cols)
        #-------------------------
        return df_mp
        
    #****************************************************************************************************
    @staticmethod
    def get_dflt_args_drop_approx_mp_duplicates(**kwargs):
        return dict(
            fuzziness                = kwargs.get('fuzziness', pd.Timedelta('1 hour')), 
            assert_single_overlap    = kwargs.get('assert_single_overlap', False), 
            addtnl_groupby_cols      = kwargs.get('addtnl_groupby_cols', None), 
            gpby_dropna              = kwargs.get('gpby_dropna', False), 
            df_mp_serial_number_col  = kwargs.get('df_mp_serial_number_col', 'mfr_devc_ser_nbr'), 
            df_mp_prem_nb_col        = kwargs.get('df_mp_prem_nb_col', 'prem_nb'), 
            df_mp_install_time_col   = kwargs.get('df_mp_install_time_col', 'inst_ts'), 
            df_mp_removal_time_col   = kwargs.get('df_mp_removal_time_col', 'rmvl_ts'), 
            df_mp_trsf_pole_nb_col   = kwargs.get('df_mp_trsf_pole_nb_col', 'trsf_pole_nb')
        )

    @staticmethod
    def agg_func_set_drop_dup_flag(x, flag_col):
        if x.shape[0]==1:
            x[flag_col]=False
        else:
            x[flag_col]=True
        return x
        
    @staticmethod
    def drop_approx_mp_duplicates_OLD(
        mp_df                   , 
        fuzziness               , 
        assert_single_overlap   = True, 
        addtnl_groupby_cols     = None, 
        gpby_dropna             = False, 
        df_mp_serial_number_col = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col       = 'prem_nb', 
        df_mp_install_time_col  = 'inst_ts', 
        df_mp_removal_time_col  = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col  = 'trsf_pole_nb'
    ):
        r"""
        In meter_premise_hist, there can be multiple entries for a given meter, which are all identical except for 
          except for e.g., inst_kind_cd, mtr_kind_cd, and even inst_ts and rmvl_ts (see Analysis Note for more information).
        In some instances, it seems like these different entries represent when a meter was upgraded, but I'm not certain.
        In any case, at the end of the day, for a single meter I want only a single row with one inst_ts and one rmvl_ts

        fuzziness:
            Sets how close two intervals must be to be considered overlapping.
            See Utilities_df.consolidate_df_according_to_fuzzy_overlap_intervals and Utilities.get_fuzzy_overlap_intervals for more information.
            Typical value: fuzziness = pd.Timedelta('1 hour')

        assert_single_overlap:
            Set to True to enforce only one overlap per group.

        addtnl_groupby_cols:
            Any additional columns to group by.  
            The default grouping columns are df_mp_serial_number_col, df_mp_prem_nb_col, and df_mp_trsf_pole_nb_col (if present).
            Example:
                After running build_active_MP_for_outages_df, which has an additional 'OUTG_REC_NB' column, one would likely want to
                have addtnl_groupby_cols=['OUTG_REC_NB'] 
                    This would preserve the 'OUTG_REC_NB', which can then be used to join with e.g. and end events for outages DF.
                    Also, this prevents unwanted tripping of assert_single_overlap=True, as all the really matters in this case is that there
                      is a single overlap for each meter/premise for each outage!
                    Example: addtnl_groupby_cols=None
                        !!!!! Now that exact duplicates are also removed (.drop_duplicates call below) !!!!!
                            mfr_devc_ser_nbr	prem_nb	inst_ts	rmvl_ts	trsf_pole_nb	OUTG_REC_NB
                            0	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	[12200360]
                            1	876723448	101152780	2021-11-05 11:27:41	2022-06-21 12:00:00	NaN	[12549182]

                        !!!!! Below would be results without exact duplicate removal (i.e., if .drop_duplicates were commented out below) !!!!!
                            mfr_devc_ser_nbr	prem_nb	inst_ts	rmvl_ts	trsf_pole_nb	OUTG_REC_NB
                            0	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	[12361925, 12200360, 12256712, 12287375, 12238...
                            1	876723448	101152780	2021-11-05 11:27:41	2022-06-21 12:00:00	NaN	[12549182]
                        Without grouping additionally by 'OUTG_REC_NB', the example below would be returned and if assert_single_overlap==True, would
                          throw the assertion error.  However, for this case, the assertion should not be thrown, because there exists one overlap
                          for each meter/premise for each outage.
                        The example above also highlights a reason to additionally group by 'OUTG_REC_NB'.  Without grouping with 'OUTG_REC_NB', the outage numbers
                          would be consolidated and collected into a column of lists, as above.
                    Example: addtnl_groupby_cols=['OUTG_REC_NB']
                            mfr_devc_ser_nbr	prem_nb	inst_ts	rmvl_ts	trsf_pole_nb	OUTG_REC_NB
                            0	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12200360
                            1	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12238642
                            2	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12256712
                            3	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12287375
                            4	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12361925
                            5	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12393114
                            6	876723448	101152780	2021-11-05 11:27:41	2022-06-21 12:00:00	NaN	12549182

        NOTE: gpby_dropna defaults to False so that, e.g., meters without trsf_pole_nbs are still included
        
        NOTE:
            THERE APPEARS TO HAVE BEEN SOME SORT OF BUG IN THE OLD VERSION OF PANDAS (1.3.4)
            When I used groupby.apply(lambda x:...), the resultant DF NEVER had the group labels as the index (i.e., the
              original index was always maintained).
            It was as if as_index was set to False (even though the default value was True; furthermore, the value of as_index
              seemed to have no affect on the result).
            TO OBTAIN THE FORM UNDER WHICH THE METHOD WAS CONSTRUCTED, one must set as_index=False AND group_keys=False as arguments
              to groupby (as the apply operations here are essentially just filter calls)!
        """
        #-------------------------
        if mp_df.shape[0]==0:
            return mp_df        
        #-------------------------
        og_cols = mp_df.columns
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        groupby_cols = [df_mp_serial_number_col, df_mp_prem_nb_col]
        if df_mp_trsf_pole_nb_col in mp_df.columns:
            groupby_cols.append(     df_mp_trsf_pole_nb_col)
            necessary_mp_cols.append(df_mp_trsf_pole_nb_col)
        if addtnl_groupby_cols is not None:
            if not isinstance(addtnl_groupby_cols, list):
                addtnl_groupby_cols=[addtnl_groupby_cols]
            groupby_cols.extend([x for x in addtnl_groupby_cols if x not in groupby_cols])
            necessary_mp_cols.extend([x for x in addtnl_groupby_cols if x not in necessary_mp_cols])
        #-------------------------
        # Idea is first sorting by groupby_cols will speed things up
        return_df = mp_df.sort_values(by=groupby_cols).copy()
        
        #-------------------------
        # First, run drop_duplicates to remove any exact duplicates before removing fuzzing duplicates
        return_df = return_df.drop_duplicates(subset=necessary_mp_cols)
        #-------------------------
        # Make sure all dates are datetime objects, not e.g., strings
        if(not is_datetime64_dtype(return_df[df_mp_install_time_col]) or 
           not is_datetime64_dtype(return_df[df_mp_removal_time_col])):
            # If one isn't, chances are both are not (and no harm in converting both either way)
            return_df = Utilities_df.convert_col_types(
                df                  = return_df, 
                cols_and_types_dict = {
                    df_mp_install_time_col : datetime.datetime, 
                    df_mp_removal_time_col : datetime.datetime
                }
            )
        #-------------------------
        # Split return_df into a subset needing approx duplicates dropped (return_df_w_dups) and one not (return_df_wo_dups).
        #   Then, to save time, only return_df_w_dups needs to be run through the duplicate removal process
        # NOTE: See Utilities_df.consolidate_df for discusssion regarding the different treatment below for the case where
        #       gpby_dropna==True vs gpby_dropna==False
        if gpby_dropna:
            return_df_w_dups  = return_df.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)>1)
            return_df_wo_dups = return_df.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)==1)
            assert(return_df_w_dups.shape[0]+return_df_wo_dups.shape[0]<=return_df.shape[0])
        else:
            # Using the methods for gpby_dropna==False (i.e., creating a new column and using .apply instead of simply using .filter)
            #   appear to be much more costly from both a time and memory standpoint.
            # Therefore, to reduce the impact, split the data into return_df_w_nans and return_df_wo_nans, and only run the
            #   methods for gpby_dropna==False on return_df_w_nans
            # To be certain, those needing to be run through the gpby_dropna==False methodology are those having at least one NaN
            #   value in any of the groupby_cols
            #-------------------------
            rows_w_nan_grp_by_cols = return_df[groupby_cols].isna().sum(axis=1)>0
            return_df_w_nans       = return_df[rows_w_nan_grp_by_cols].copy()
            return_df_wo_nans      = return_df[~rows_w_nan_grp_by_cols].copy()
            #-------------------------
            # First, handle return_df_wo_nans, just as in "if gpby_dropna(==True )" above
            return_df_wo_nans_w_dups  = return_df_wo_nans.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)>1)
            return_df_wo_nans_wo_dups = return_df_wo_nans.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)==1)
            assert(return_df_wo_nans_w_dups.shape[0]+return_df_wo_nans_wo_dups.shape[0]<=return_df_wo_nans.shape[0])
            # return_df_wo_nans no longer needed
            del return_df_wo_nans
            #-------------------------
            # Second, handle return_df_w_nans as originally done for gpby_dropna==False
            tmp_col                   = Utilities.generate_random_string()
            return_df_w_nans[tmp_col] = np.nan
            return_df_w_nans          = return_df_w_nans.groupby(
                groupby_cols, 
                dropna     = gpby_dropna, 
                as_index   = False, 
                group_keys = False
            ).apply(
                lambda x: MeterPremise.agg_func_set_drop_dup_flag(
                    x, 
                    flag_col = tmp_col
                )
            )
            #-----
            # NOTE: Using return_df_w_nans[tmp_col]==True and return_df_w_nans[tmp_col]==False instead of return_df_w_nans[tmp_col] and ~return_df_w_nans[tmp_col] protects
            #         against case where return_df_w_nans is empty.  
            #       Using the new method (==True, ==False) will always return a pd.DataFrame with the expected columns, so no error thrown after .drop(columns=[tmp_col]) call below.
            #       Using the old method, when return_df_w_nans is empty, return_df_w_nans_w_dups and return_df_w_nans_wo_dups would be empty (as expected) BUT would not have any
            #         columns, thus resulting in an error when .drop(columns=[tmp_col]) is called!
            return_df_w_nans_w_dups  = return_df_w_nans[return_df_w_nans[tmp_col]==True].copy()
            return_df_w_nans_wo_dups = return_df_w_nans[return_df_w_nans[tmp_col]==False].copy()
            #-----
            return_df_w_nans_w_dups  = return_df_w_nans_w_dups.drop(columns=[tmp_col])
            return_df_w_nans_wo_dups = return_df_w_nans_wo_dups.drop(columns=[tmp_col])
            assert(return_df_w_nans_w_dups.shape[0]+return_df_w_nans_wo_dups.shape[0]==return_df_w_nans.shape[0])
            # return_df_w_nans no longer needed
            del return_df_w_nans
            #-------------------------
            # Finally, combine to obtain return_df_w_dups and return_df_wo_dups
            assert(len(set(return_df_wo_nans_w_dups.columns).symmetric_difference(set(return_df_w_nans_w_dups.columns)))==0)
            assert(len(set(return_df_wo_nans_wo_dups.columns).symmetric_difference(set(return_df_w_nans_wo_dups.columns)))==0)
            if return_df_wo_nans_w_dups.shape[0]>0 and return_df_w_nans_w_dups.shape[0]>0:
                assert(return_df_wo_nans_w_dups.index.names==return_df_w_nans_w_dups.index.names)
            if return_df_wo_nans_wo_dups.shape[0]>0 and return_df_w_nans_wo_dups.shape[0]>0:
                assert(return_df_wo_nans_wo_dups.index.names==return_df_w_nans_wo_dups.index.names)
            return_df_w_dups  = pd.concat([return_df_wo_nans_w_dups,  return_df_w_nans_w_dups])
            return_df_wo_dups = pd.concat([return_df_wo_nans_wo_dups, return_df_w_nans_wo_dups])
            # No longer need: return_df_wo_nans_w_dups, return_df_w_nans_w_dups, return_df_wo_nans_wo_dups, return_df_w_nans_wo_dups
            del return_df_wo_nans_w_dups
            del return_df_w_nans_w_dups
            del return_df_wo_nans_wo_dups
            del return_df_w_nans_wo_dups
        #-------------------------
        # In return_df_wo_dups need to match any columns collected in lists in return_df_w_dups (through consolidate_df_according_to_fuzzy_overlap_intervals)
        cols_to_collect_in_lists=[x for x in return_df_wo_dups.columns if x not in groupby_cols+[df_mp_install_time_col, df_mp_removal_time_col]]
        for lst_col in cols_to_collect_in_lists:
            return_df_wo_dups[lst_col] = return_df_wo_dups[lst_col].apply(lambda x: x if isinstance(x, list) else [x])
        #-----
        # If there aren't any to run through duplicate removal process (either due to none existing at all, or due to already
        #   being weeded out about by drop_duplicates), return return_df_wo_dups
        if return_df_w_dups.shape[0]==0:
            return return_df_wo_dups
        #-------------------------
        return_df_w_dups = return_df_w_dups.groupby(groupby_cols, dropna=gpby_dropna).apply(
            lambda x: Utilities_df.consolidate_df_according_to_fuzzy_overlap_intervals_OLD(
                df                           = x, 
                ovrlp_intrvl_0_col           = df_mp_install_time_col, 
                ovrlp_intrvl_1_col           = df_mp_removal_time_col, 
                fuzziness                    = fuzziness, 
                groupby_cols                 = groupby_cols, 
                assert_single_overlap        = assert_single_overlap, 
                cols_to_collect_in_lists     = None, 
                recover_uniqueness_violators = False, 
                gpby_dropna                  = gpby_dropna, 
                drop_idx_cols                = True, 
                maintain_original_cols       = True, 
                enforce_list_cols_for_1d_df  = True
            )
        )
        #-------------------------
        # By default, groupby will set the index of the returned df equal to the groupby cols (together with an unnamed
        #   index representing the index number of the sub-df, due to the fact that .reset_index() is called towards the end of 
        #   Utilities_df.consolidate_df_according_to_fuzzy_overlap_intervals for each df_i)
        # These cols are also reproduced in the columns of the resultant DF.  Therefore, it is safe to reset the index
        #   with drop=True
        assert(len(set(groupby_cols).difference(set(return_df_w_dups.index.names)))==0)
        assert(len(groupby_cols)+1==len(return_df_w_dups.index.names))
        return_df_w_dups = return_df_w_dups.reset_index(drop=True)
        #-------------------------
        assert(len(set(return_df_wo_dups.columns).symmetric_difference(set(return_df_w_dups.columns)))==0)
        if return_df_wo_dups.shape[0]>0 and return_df_w_dups.shape[0]>0:
            assert(return_df_wo_dups.index.names==return_df_w_dups.index.names)
        return_df = pd.concat([return_df_wo_dups, return_df_w_dups])
        #-------------------------
        return return_df
        
        
    @staticmethod
    def drop_approx_mp_duplicates(
        mp_df                   , 
        fuzziness               , 
        assert_single_overlap   = True, 
        addtnl_groupby_cols     = None, 
        gpby_dropna             = False, 
        df_mp_serial_number_col = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col       = 'prem_nb', 
        df_mp_install_time_col  = 'inst_ts', 
        df_mp_removal_time_col  = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col  = 'trsf_pole_nb'
    ):
        r"""
        In meter_premise_hist, there can be multiple entries for a given meter, which are all identical except for 
          except for e.g., inst_kind_cd, mtr_kind_cd, and even inst_ts and rmvl_ts (see Analysis Note for more information).
        In some instances, it seems like these different entries represent when a meter was upgraded, but I'm not certain.
        In any case, at the end of the day, for a single meter I want only a single row with one inst_ts and one rmvl_ts

        fuzziness:
            Sets how close two intervals must be to be considered overlapping.
            See Utilities_df.consolidate_df_according_to_fuzzy_overlap_intervals and Utilities.get_fuzzy_overlap_intervals for more information.
            Typical value: fuzziness = pd.Timedelta('1 hour')

        assert_single_overlap:
            Set to True to enforce only one overlap per group.

        addtnl_groupby_cols:
            Any additional columns to group by.  
            The default grouping columns are df_mp_serial_number_col, df_mp_prem_nb_col, and df_mp_trsf_pole_nb_col (if present).
            Example:
                After running build_active_MP_for_outages_df, which has an additional 'OUTG_REC_NB' column, one would likely want to
                have addtnl_groupby_cols=['OUTG_REC_NB'] 
                    This would preserve the 'OUTG_REC_NB', which can then be used to join with e.g. and end events for outages DF.
                    Also, this prevents unwanted tripping of assert_single_overlap=True, as all the really matters in this case is that there
                      is a single overlap for each meter/premise for each outage!
                    Example: addtnl_groupby_cols=None
                        !!!!! Now that exact duplicates are also removed (.drop_duplicates call below) !!!!!
                            mfr_devc_ser_nbr	prem_nb	inst_ts	rmvl_ts	trsf_pole_nb	OUTG_REC_NB
                            0	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	[12200360]
                            1	876723448	101152780	2021-11-05 11:27:41	2022-06-21 12:00:00	NaN	[12549182]

                        !!!!! Below would be results without exact duplicate removal (i.e., if .drop_duplicates were commented out below) !!!!!
                            mfr_devc_ser_nbr	prem_nb	inst_ts	rmvl_ts	trsf_pole_nb	OUTG_REC_NB
                            0	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	[12361925, 12200360, 12256712, 12287375, 12238...
                            1	876723448	101152780	2021-11-05 11:27:41	2022-06-21 12:00:00	NaN	[12549182]
                        Without grouping additionally by 'OUTG_REC_NB', the example below would be returned and if assert_single_overlap==True, would
                          throw the assertion error.  However, for this case, the assertion should not be thrown, because there exists one overlap
                          for each meter/premise for each outage.
                        The example above also highlights a reason to additionally group by 'OUTG_REC_NB'.  Without grouping with 'OUTG_REC_NB', the outage numbers
                          would be consolidated and collected into a column of lists, as above.
                    Example: addtnl_groupby_cols=['OUTG_REC_NB']
                            mfr_devc_ser_nbr	prem_nb	inst_ts	rmvl_ts	trsf_pole_nb	OUTG_REC_NB
                            0	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12200360
                            1	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12238642
                            2	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12256712
                            3	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12287375
                            4	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12361925
                            5	876723448	101152780	2018-04-16 12:00:20	2021-10-22 12:00:00	NaN	12393114
                            6	876723448	101152780	2021-11-05 11:27:41	2022-06-21 12:00:00	NaN	12549182

        NOTE: gpby_dropna defaults to False so that, e.g., meters without trsf_pole_nbs are still included
        
        NOTE:
            THERE APPEARS TO HAVE BEEN SOME SORT OF BUG IN THE OLD VERSION OF PANDAS (1.3.4)
            When I used groupby.apply(lambda x:...), the resultant DF NEVER had the group labels as the index (i.e., the
              original index was always maintained).
            It was as if as_index was set to False (even though the default value was True; furthermore, the value of as_index
              seemed to have no affect on the result).
            TO OBTAIN THE FORM UNDER WHICH THE METHOD WAS CONSTRUCTED, one must set as_index=False AND group_keys=False as arguments
              to groupby (as the apply operations here are essentially just filter calls)!
        """
        #-------------------------
        if mp_df.shape[0]==0:
            return mp_df        
        #-------------------------
        og_cols = mp_df.columns
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        groupby_cols      = [df_mp_serial_number_col, df_mp_prem_nb_col]
        if df_mp_trsf_pole_nb_col in mp_df.columns:
            groupby_cols.append(     df_mp_trsf_pole_nb_col)
            necessary_mp_cols.append(df_mp_trsf_pole_nb_col)
        if addtnl_groupby_cols is not None:
            if not isinstance(addtnl_groupby_cols, list):
                addtnl_groupby_cols=[addtnl_groupby_cols]
            groupby_cols.extend(     [x for x in addtnl_groupby_cols if x not in groupby_cols])
            necessary_mp_cols.extend([x for x in addtnl_groupby_cols if x not in necessary_mp_cols])
        #-------------------------
        # Idea is first sorting by groupby_cols will speed things up
        return_df = mp_df.sort_values(by=groupby_cols).copy()
        
        #-------------------------
        # First, run drop_duplicates to remove any exact duplicates before removing fuzzing duplicates
        return_df = return_df.drop_duplicates(subset=necessary_mp_cols)
        #-------------------------
        # Make sure all dates are datetime objects, not e.g., strings
        if(not is_datetime64_dtype(return_df[df_mp_install_time_col]) or 
           not is_datetime64_dtype(return_df[df_mp_removal_time_col])):
            # If one isn't, chances are both are not (and no harm in converting both either way)
            return_df = Utilities_df.convert_col_types(
                df                  = return_df, 
                cols_and_types_dict = {
                    df_mp_install_time_col : datetime.datetime, 
                    df_mp_removal_time_col : datetime.datetime
                }
            )
        #-------------------------
        # Split return_df into a subset needing approx duplicates dropped (return_df_w_dups) and one not (return_df_wo_dups).
        #   Then, to save time, only return_df_w_dups needs to be run through the duplicate removal process
        # NOTE: See Utilities_df.consolidate_df for discusssion regarding the different treatment below for the case where
        #       gpby_dropna==True vs gpby_dropna==False
        if gpby_dropna:
            return_df_w_dups  = return_df.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)>1)
            return_df_wo_dups = return_df.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)==1)
            assert(return_df_w_dups.shape[0]+return_df_wo_dups.shape[0]<=return_df.shape[0])
        else:
            # Using the methods for gpby_dropna==False (i.e., creating a new column and using .apply instead of simply using .filter)
            #   appear to be much more costly from both a time and memory standpoint.
            # Therefore, to reduce the impact, split the data into return_df_w_nans and return_df_wo_nans, and only run the
            #   methods for gpby_dropna==False on return_df_w_nans
            # To be certain, those needing to be run through the gpby_dropna==False methodology are those having at least one NaN
            #   value in any of the groupby_cols
            #-------------------------
            rows_w_nan_grp_by_cols = return_df[groupby_cols].isna().sum(axis=1)>0
            return_df_w_nans       = return_df[rows_w_nan_grp_by_cols].copy()
            return_df_wo_nans      = return_df[~rows_w_nan_grp_by_cols].copy()
            #-------------------------
            # First, handle return_df_wo_nans, just as in "if gpby_dropna(==True )" above
            return_df_wo_nans_w_dups  = return_df_wo_nans.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)>1)
            return_df_wo_nans_wo_dups = return_df_wo_nans.groupby(groupby_cols, dropna=gpby_dropna).filter(lambda x: len(x)==1)
            assert(return_df_wo_nans_w_dups.shape[0]+return_df_wo_nans_wo_dups.shape[0]<=return_df_wo_nans.shape[0])
            # return_df_wo_nans no longer needed
            del return_df_wo_nans
            #-------------------------
            # Second, handle return_df_w_nans as originally done for gpby_dropna==False
            tmp_col                   = Utilities.generate_random_string()
            return_df_w_nans[tmp_col] = np.nan
            return_df_w_nans          = return_df_w_nans.groupby(groupby_cols, dropna=gpby_dropna, as_index=False, group_keys=False).apply(lambda x: MeterPremise.agg_func_set_drop_dup_flag(x, flag_col=tmp_col))
            #-----
            # NOTE: Using return_df_w_nans[tmp_col]==True and return_df_w_nans[tmp_col]==False instead of return_df_w_nans[tmp_col] and ~return_df_w_nans[tmp_col] protects
            #         against case where return_df_w_nans is empty.  
            #       Using the new method (==True, ==False) will always return a pd.DataFrame with the expected columns, so no error thrown after .drop(columns=[tmp_col]) call below.
            #       Using the old method, when return_df_w_nans is empty, return_df_w_nans_w_dups and return_df_w_nans_wo_dups would be empty (as expected) BUT would not have any
            #         columns, thus resulting in an error when .drop(columns=[tmp_col]) is called!
            return_df_w_nans_w_dups  = return_df_w_nans[return_df_w_nans[tmp_col]==True].copy()
            return_df_w_nans_wo_dups = return_df_w_nans[return_df_w_nans[tmp_col]==False].copy()
            #-----
            return_df_w_nans_w_dups  = return_df_w_nans_w_dups.drop(columns=[tmp_col])
            return_df_w_nans_wo_dups = return_df_w_nans_wo_dups.drop(columns=[tmp_col])
            assert(return_df_w_nans_w_dups.shape[0]+return_df_w_nans_wo_dups.shape[0]==return_df_w_nans.shape[0])
            # return_df_w_nans no longer needed
            del return_df_w_nans
            #--------------------------------------------------
            # Finally, combine to obtain return_df_w_dups and return_df_wo_dups
            # NOTE: Pandas now generates the below FutureWarning when an element of list fed to pd.concat is empty.
            #       I don't think this has any consequential effect for us, but I have altered the code (included annoying if/else statements below)
            #         to avoid the situation leading to the warning
            #     FutureWarning: The behavior of array concatenation with empty entries is deprecated. In a future version, this will no longer exclude empty 
            #       items when determining the result dtype. To retain the old behavior, exclude the empty entries before the concat operation.
            #-------------------------
            assert(len(set(return_df_wo_nans_w_dups.columns).symmetric_difference(set(return_df_w_nans_w_dups.columns)))==0)
            if return_df_wo_nans_w_dups.shape[0]>0 and return_df_w_nans_w_dups.shape[0]>0:
                assert(return_df_wo_nans_w_dups.index.names==return_df_w_nans_w_dups.index.names)
                return_df_w_dups  = pd.concat([return_df_wo_nans_w_dups,  return_df_w_nans_w_dups])
            else:
                if return_df_wo_nans_w_dups.shape[0]>0:
                    return_df_w_dups = return_df_wo_nans_w_dups.copy()
                else:
                    # NOTE: This handles case where both are empty as well, in which case return_df_w_dups will be empty
                    return_df_w_dups = return_df_w_nans_w_dups.copy()
            
            #-------------------------
            assert(len(set(return_df_wo_nans_wo_dups.columns).symmetric_difference(set(return_df_w_nans_wo_dups.columns)))==0)
            if return_df_wo_nans_wo_dups.shape[0]>0 and return_df_w_nans_wo_dups.shape[0]>0:
                assert(return_df_wo_nans_wo_dups.index.names==return_df_w_nans_wo_dups.index.names)
                return_df_wo_dups = pd.concat([return_df_wo_nans_wo_dups, return_df_w_nans_wo_dups])
            else:
                if return_df_wo_nans_wo_dups.shape[0]>0:
                    return_df_wo_dups = return_df_wo_nans_wo_dups.copy()
                else:
                    # NOTE: This handles case where both are empty as well, in which case return_df_wo_dups will be empty
                    return_df_wo_dups = return_df_w_nans_wo_dups.copy()
            #--------------------------------------------------
            # No longer need: return_df_wo_nans_w_dups, return_df_w_nans_w_dups, return_df_wo_nans_wo_dups, return_df_w_nans_wo_dups
            del return_df_wo_nans_w_dups
            del return_df_w_nans_w_dups
            del return_df_wo_nans_wo_dups
            del return_df_w_nans_wo_dups
        #-------------------------
        # In return_df_wo_dups need to match any columns collected in lists in return_df_w_dups (through consolidate_df_according_to_fuzzy_overlap_intervals)
        cols_to_collect_in_lists=[x for x in return_df_wo_dups.columns if x not in groupby_cols+[df_mp_install_time_col, df_mp_removal_time_col]]
        for lst_col in cols_to_collect_in_lists:
            return_df_wo_dups[lst_col] = return_df_wo_dups[lst_col].apply(lambda x: x if isinstance(x, list) else [x])
        #-----
        # If there aren't any to run through duplicate removal process (either due to none existing at all, or due to already
        #   being weeded out about by drop_duplicates), return return_df_wo_dups
        if return_df_w_dups.shape[0]==0:
            return return_df_wo_dups
        #-------------------------
        return_df_w_dups = Utilities_df.consolidate_df_according_to_fuzzy_overlap_intervals(
            df                           = return_df_w_dups, 
            ovrlp_intrvl_0_col           = df_mp_install_time_col, 
            ovrlp_intrvl_1_col           = df_mp_removal_time_col, 
            fuzziness                    = fuzziness, 
            groupby_cols                 = groupby_cols, 
            assert_single_overlap        = assert_single_overlap, 
            cols_to_collect_in_lists     = cols_to_collect_in_lists, 
            recover_uniqueness_violators = False, 
            gpby_dropna                  = gpby_dropna, 
            allow_duplicates_in_lists    = False, 
            allow_NaNs_in_lists          = False
        )
        #-------------------------
        assert(len(set(return_df_wo_dups.columns).symmetric_difference(set(return_df_w_dups.columns)))==0)
        if return_df_wo_dups.shape[0]>0 and return_df_w_dups.shape[0]>0:
            assert(return_df_wo_dups.index.names==return_df_w_dups.index.names)
        return_df = pd.concat([return_df_wo_dups, return_df_w_dups])
        #-------------------------
        return return_df
        
    #****************************************************************************************************
    @staticmethod
    def handle_PNs_with_ambiguous_xfmrs(
        PN_to_xfmr      , 
        how             = 'first', 
        drop_nans_first = True
    ):
        r"""
        PN_to_xfmr should be a series whose indices are premise numbers and values are transformer pole numbers.
        It is expected that each PN should have one xfmr, however, unexpected instances arise in data.
        If a PN has multiple xfrms, this function determines what to do.
        NOTE: I considered making a hard assertion that each PN have one trsf_pole_nb, but using this function
              allows the program to continue running in the rare instance that a PN has more than one trsf_pole_nbs.

        First, if drop_nans_first==True and if a PN has two xfmr values and one is NaN, keep that which is not NaN.
        Otherwise, the handling is determing by how.

        how:
            Must be equal to 'first' or 'exclude'
            'first':
                The first value in the collection of xfmrs is kept.
            'exclude':
                The entire entry is dropped.
        """
        #-------------------------
        assert(isinstance(PN_to_xfmr, pd.Series))
        how = how.lower()
        assert(how=='first' or how=='exclude')
        #-------------------------
        list_elements_mask = Utilities_df.get_list_elements_mask_for_series(PN_to_xfmr)
        # If no list elements found, simply return
        if list_elements_mask.sum()==0:
            return PN_to_xfmr
        #-------------------------
        # Remove any NaNs
        if drop_nans_first:
            PN_to_xfmr[list_elements_mask]=PN_to_xfmr[list_elements_mask].apply(lambda x: pd.Series(x).dropna().unique())
        #-------------------------
        # Next, resolve any cases where the value is a list with a single element
        PN_to_xfmr[list_elements_mask]=PN_to_xfmr[list_elements_mask].apply(lambda x: x[0] if len(x)==1 else x)
        #-------------------------
        # Now, as the above steps may have resolved some issues in PN_to_xfmr, regenerate list_elements_mask
        list_elements_mask = Utilities_df.get_list_elements_mask_for_series(PN_to_xfmr)
        # If no list elements found, simply return
        if list_elements_mask.sum()==0:
            return PN_to_xfmr
        #-------------------------
        # Finally, finish according to the value of how
        if how=='first':
            PN_to_xfmr[list_elements_mask]=PN_to_xfmr[list_elements_mask].apply(lambda x: x[0])
        elif how=='exclude':
            PN_to_xfmr = PN_to_xfmr[~list_elements_mask]
        else:
            assert(0)
        #-------------------------
        return PN_to_xfmr

    @staticmethod
    def fill_trsf_pole_nbs_in_mp_df_hist_from_curr(
        mp_df_hist                , 
        mp_df_curr                , 
        replace_if_present        = True, 
        prem_nb_col               = 'prem_nb', 
        trsf_pole_nb_col          = 'trsf_pole_nb', 
        how_to_handle_ambiguities = 'first'
    ):
        r"""
        The historical MeterPremise (default.meter_premise_hist) does not have trsf_pole_nb information.
        This function used the premise number (PN) and transformer pole number (trsf_pole_nb) information from
          mp_df_curr to fill in the values for mp_df_hist
        ASSUMPTION!!!!!:  The assumption here is that the transformer number for a premise has not changed over time.
        
        NOTE: mp_df_hist is expected to not already contain trsf_pole_nb_col, as the purpose of this is to append such a column.
              However, within a chain of function calls, it is quite possible that mp_df_hist containing a trsf_pole_nb_col will 
                be passed.  In such a case, the default behavior will be for the trsf_pole_nb_col in mp_df_hist to be replaced
                by the results of this function.  In such a situation, and without such a replacement, the returned mp_df_hist
                would have columns f'{trsf_pole_nb_col}_x' and f'{trsf_pole_nb_col}_y', which is not desired.
              The parameter replace_if_present can alter this functionality.  If True, the functionality is as described above.
                If False and trsf_pole_nb_col is already present in mp_df_hist, then mp_df_hist will simply be returned (UNLESS
                all of the values in mp_df_hist[trsf_pole_nb_col] are NaNs, in which case the False value will be overriden and
                the function will proceed).
                
        replace_if_present:
            As described in the NOTE above, this controls the functionality in the case where trsf_pole_nb_col is already present
              in mp_df_hist.  If trsf_pole_nb_col is not in mp_df_hist, this parameter will have no effect and the results of the
              function will be appended to mp_df_hist as usual.
            If trsf_pole_nb_col is in mp_df_hist:
                replace_if_present==True:
                    Replace the contents with the results of the function.
                replace_if_present==False:
                    Simply return mp_df_hist without performing any action !!!!!UNLESS!!!!! all of the values in mp_df_hist[trsf_pole_nb_col]
                      are NaNs, in which case the False value will be overriden and replace_if_present will essentially be set to True.

        how_to_handle_ambiguities:
            Should be 'first' or 'exclude'.
            See MeterPremise.handle_PNs_with_ambiguous_xfmrs for more information
        """
        #-------------------------
        # As described in the documentation above, the default behavior for the case where trsf_pole_nb_col is already in mp_df_hist
        # is to replace the contents with the result of this function.  The only scenario where the values will not be replaced is when
        # replace_if_present==False AND not all of the values in mp_df_hist[trsf_pole_nb_col] are NaNs (i.e., at least one value is not NaN)
        if not replace_if_present:
            # Being 'present' essentially means trsf_pole_nb_col exists in mp_df_hist AND not all NaNs
            if(trsf_pole_nb_col in mp_df_hist.columns and mp_df_hist[trsf_pole_nb_col].notna().sum()>0):
                return mp_df_hist
        #-------------------------
        # Build a pd.Series object with indices equal to the premise numbers and values equal to the transformer pole numbers
        #   NOTE: The use of MeterPremise.handle_PNs_with_ambiguous_xfmrs ensures each PN has at most one associated trsf_pole_nb
        #   NOTE: A NaN (or NaT) will be ignored by nunique (i.e., if 1 unique and 1 NaN values, nunique will return a value of 1)
        #         but will not be ignored by unique (i.e., if 1 unique and 1 NaN values, unique will return (NaN, unique_val_i)
        #         Therefore, alter code from .agg(pd.Series.unique) to .agg(lambda x: x.dropna().unique())
        prem_nb_to_trsf_pole = mp_df_curr.groupby(prem_nb_col)[trsf_pole_nb_col].agg(lambda x: x.dropna().unique())
        prem_nb_to_trsf_pole = MeterPremise.handle_PNs_with_ambiguous_xfmrs(
            PN_to_xfmr      = prem_nb_to_trsf_pole, 
            how             = how_to_handle_ambiguities, 
            drop_nans_first = True
        )
        assert(Utilities_df.are_all_series_elements_one_of_types(prem_nb_to_trsf_pole, [str, int]))
        #-------------------------
        # Merge prem_nb_to_trsf_pole with mp_df_hist
        assert(mp_df_hist[prem_nb_col].dtype==prem_nb_to_trsf_pole.index.dtype)
        # If trsf_pole_nb_col already in mp_df_hist, remove it so the returned DF doesn't have columns 
        #   f'{trsf_pole_nb_col}_x' and f'{trsf_pole_nb_col}_y'
        if trsf_pole_nb_col in mp_df_hist.columns:
            mp_df_hist = mp_df_hist.drop(columns=[trsf_pole_nb_col])
        # After the merge, mp_df_hist should have one more column and the same number of rows as initially
        og_shape   = mp_df_hist.shape
        mp_df_hist = pd.merge(mp_df_hist, prem_nb_to_trsf_pole, how='left', left_on=prem_nb_col, right_index=True)
        assert(mp_df_hist.shape[0]==og_shape[0])
        assert(mp_df_hist.shape[1]==og_shape[1]+1)
        #-------------------------
        return mp_df_hist

    @staticmethod
    def fill_trsf_pole_nbs_in_mp_df_infer_within(
        mp_df                     , 
        prem_nb_col               = 'prem_nb', 
        trsf_pole_nb_col          = 'trsf_pole_nb', 
        how_to_handle_ambiguities = 'first'    
    ):
        r"""
        The historical MeterPremise (default.meter_premise_hist) does not have trsf_pole_nb information.
        This function locates all entries in mp_df with NaN trsf_pole_nbs and tries to infer their value by looking
          for the appropriate PNs with trsf_pole_nbs elsewhere in the DF.
        This was built for use in updating the trsf_pole_nb in mp_df resulting from combining mp_df_curr and mp_df_hist.

        ASSUMPTION!!!!!:  The assumption here is that the transformer number for a premise has not changed over time.

        how_to_handle_ambiguities:
            Should be 'first' or 'exclude'.
            See MeterPremise.handle_PNs_with_ambiguous_xfmrs for more information
        """
        #-------------------------
        # Basically, the strategy here is to utilize MeterPremise.fill_trsf_pole_nbs_in_mp_df_hist_from_curr.  In order to do so, 
        #   I will treat the portion of mp_df with NaN trsf_pole_nbs as mp_df_hist, and the rest as mp_df_curr
        og_shape      = mp_df.shape
        nan_trsf_mask = mp_df[trsf_pole_nb_col].isna()
        # Note: Don't want trsf_pole_nb_x and trsf_pole_nb_y to be returned from fill_trsf_pole_nbs_in_mp_df_hist_from_curr, 
        #       so need to exclude trsf_pole_nb from mp_df[nan_trsf_mask] in input
        mp_df[nan_trsf_mask] = MeterPremise.fill_trsf_pole_nbs_in_mp_df_hist_from_curr(
            mp_df_hist                = mp_df[nan_trsf_mask][[x for x in mp_df if x!=trsf_pole_nb_col]], 
            mp_df_curr                = mp_df[~nan_trsf_mask], 
            prem_nb_col               = prem_nb_col, 
            trsf_pole_nb_col          = trsf_pole_nb_col, 
            how_to_handle_ambiguities = how_to_handle_ambiguities
        )
        assert(mp_df.shape==og_shape)
        #-------------------------
        return mp_df

    #TODO: Utilize this throughout MeterPremise (e.g., in get_historic_SNs_for_PNs)
    @staticmethod
    def build_andor_join_mp_curr_hist(
        mp_df_curr                     ,
        mp_df_hist                     ,
        build_sql_function_kwargs_curr ,
        build_sql_function_kwargs_hist , 
        assume_one_xfmr_per_PN         = True, 
        drop_approx_duplicates         = True, 
        drop_approx_duplicates_args    = None, 
        df_mp_serial_number_col        = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col              = 'prem_nb', 
        df_mp_install_time_col         = 'inst_ts', 
        df_mp_removal_time_col         = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col         = 'trsf_pole_nb'
    ):
        r"""
        Mainly used for joining current and historic Meter Premise DFs.  
        However, can also be used to first build mp_df_curr and/or mp_df_hist and then join them.

        If mp_df_curr or mp_df_hist are not supplied, they will be built using MeterPremise.build_sql_meter_premise
          together with build_sql_function_kwargs_curr/hist

        mp_df_curr and mp_df_hist will then be joined together using concat.
          NOTE: Join can done either with concat or with a merge.  I'm not sure if one is advantageous over the other,
                  but in my limited testing, it appears the concat method is faster (and, in my testing, the results were 
                  consistent after the drop_duplicates call)
                I'll leave the merge method commented below, in case further testing is needed/desired
          NOTE: a single serial number can have multiple inst_ts and rmvl_ts entries, therefore these must be included in
                  drop_duplicates below (excluding them would elmininate all but one of the service time periods)
                  
        drop_approx_duplicates_args:
            Arguments already handled inside this function, so SHOULD NOT BE INCLUDED IN drop_approx_duplicates_args:
                mp_df
                df_mp_serial_number_col
                df_mp_prem_nb_col
                df_mp_install_time_col
                df_mp_removal_time_col
                df_mp_trsf_pole_nb_col
            Possible arguments to supply to drop_approx_duplicates_args (at time of writing, check drop_approx_mp_duplicates for
              most up-to-date info).  Anything else in drop_approx_duplicates_args will be ignored
                fuzziness
                assert_single_overlap
                addtnl_groupby_cols
                gpby_dropna
        """
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        #--------------------------------------------------
        # If mp_df_curr or mp_df_hist are not supplied, they will be built
        #-------------------------
        if build_sql_function_kwargs_hist is None:
            build_sql_function_kwargs_hist=copy.deepcopy(build_sql_function_kwargs_curr)
        #-------------------------
        if mp_df_curr is None:
            if build_sql_function_kwargs_curr is None:
                build_sql_function_kwargs_curr = {}
            #-----
            if 'cols_of_interest' not in build_sql_function_kwargs_curr.keys():
                build_sql_function_kwargs_curr['cols_of_interest'] = []
            # Instead of for loop below, could use:
            #     i. build_sql_function_kwargs_curr['cols_of_interest'].extend(necessary_mp_cols)
            #    ii. build_sql_function_kwargs_curr['cols_of_interest'] = list(set(build_sql_function_kwargs_curr['cols_of_interest']))
            #  but this would alter the order of the original cols_of_interest
            for necessary_col in necessary_mp_cols:
                if necessary_col not in build_sql_function_kwargs_curr['cols_of_interest']:
                    build_sql_function_kwargs_curr['cols_of_interest'].append(necessary_col)
            if assume_one_xfmr_per_PN and df_mp_trsf_pole_nb_col not in build_sql_function_kwargs_curr['cols_of_interest']:
                build_sql_function_kwargs_curr['cols_of_interest'].append(df_mp_trsf_pole_nb_col)
            #-----
            build_sql_function_kwargs_curr['table_name']='meter_premise'
            #-----
            mp_curr = MeterPremise(
                df_construct_type         = DFConstructType.kRunSqlQuery, 
                init_df_in_constructor    = True, 
                build_sql_function        = MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs = build_sql_function_kwargs_curr
            )
            mp_df_curr = mp_curr.df    
        #-------------------------
        if mp_df_hist is None:
            if build_sql_function_kwargs_hist is None:
                build_sql_function_kwargs_hist = {}
            #-----
            if 'cols_of_interest' not in build_sql_function_kwargs_hist.keys():
                build_sql_function_kwargs_hist['cols_of_interest'] = []
            # Instead of for loop below, could use:
            #     i. build_sql_function_kwargs_hist['cols_of_interest'].extend(necessary_mp_cols)
            #    ii. build_sql_function_kwargs_hist['cols_of_interest'] = list(set(build_sql_function_kwargs_hist['cols_of_interest']))
            #  but this would alter the order of the original cols_of_interest
            for necessary_col in necessary_mp_cols:
                if necessary_col not in build_sql_function_kwargs_hist['cols_of_interest']:
                    build_sql_function_kwargs_hist['cols_of_interest'].append(necessary_col)
            # Keep in mind, meter_premise_hist has no trsf_pole_nb field!
            if df_mp_trsf_pole_nb_col in build_sql_function_kwargs_hist['cols_of_interest']:
                build_sql_function_kwargs_hist['cols_of_interest'].remove(df_mp_trsf_pole_nb_col)
            #-----
            build_sql_function_kwargs_hist['table_name']='meter_premise_hist'
            #-----
            mp_hist = MeterPremise(
                df_construct_type         = DFConstructType.kRunSqlQuery, 
                init_df_in_constructor    = True, 
                build_sql_function        = MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs = build_sql_function_kwargs_hist
            )
            mp_df_hist = mp_hist.df
        #-------------------------
        # Handle case of one or both mp_df_curr/hist being empty
        if mp_df_curr.shape[0]==0 and mp_df_hist.shape[0]==0:
            return pd.DataFrame()
        if mp_df_curr.shape[0]==0:
            mp_df_curr = pd.DataFrame(columns=mp_df_hist.columns)
        if mp_df_hist.shape[0]==0:
            mp_df_hist = pd.DataFrame(columns=mp_df_curr.columns)
        #-------------------------
        # At a bare minimum, mp_df_curr and mp_df_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in mp_df_curr.columns for x in necessary_mp_cols]))
        assert(all([x in mp_df_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        # Make sure all dates are datetime objects, not e.g., strings
        if(not is_datetime64_dtype(mp_df_hist[df_mp_install_time_col]) or 
           not is_datetime64_dtype(mp_df_hist[df_mp_removal_time_col])):
            # If one isn't, chances are both are not (and no harm in converting both either way)
            mp_df_hist = Utilities_df.convert_col_types(
                df                  = mp_df_hist, 
                cols_and_types_dict = {
                    df_mp_install_time_col : datetime.datetime, 
                    df_mp_removal_time_col : datetime.datetime
                }
            )
        #-----
        if(not is_datetime64_dtype(mp_df_curr[df_mp_install_time_col]) 
           or not is_datetime64_dtype(mp_df_curr[df_mp_removal_time_col])):
            # If one isn't, chances are both are not (and no harm in converting both either way)
            mp_df_curr = Utilities_df.convert_col_types(
                df                  = mp_df_curr, 
                cols_and_types_dict = {
                    df_mp_install_time_col : datetime.datetime, 
                    df_mp_removal_time_col : datetime.datetime
                }
            )    
        #-------------------------
        # If assume_one_xfmr_per_PN==True, fill the trsf_pole_nbs in mp_df_hist using mp_df_curr
        if assume_one_xfmr_per_PN:
            mp_df_hist = MeterPremise.fill_trsf_pole_nbs_in_mp_df_hist_from_curr(
                mp_df_hist                = mp_df_hist, 
                mp_df_curr                = mp_df_curr, 
                prem_nb_col               = df_mp_prem_nb_col, 
                trsf_pole_nb_col          = df_mp_trsf_pole_nb_col, 
                how_to_handle_ambiguities = 'first'
            )
        #-------------------------
        # Combine together mp_df_hist and mp_df_curr.
        # This can be done either with concat or with a merge.  I'm not sure if one is advantageous over the other,
        #   but in my limited testing, it appears the concat method is faster (and, in my testing, the results were 
        #   consistent after the drop_duplicates call)
        #   I'll leave the merge method commented below, in case further testing is needed/desired
        # Note: a single serial number can have multiple inst_ts and rmvl_ts entries, therefore these must be included in
        #       drop_duplicates below (excluding them would elmininate all but one of the service time periods)
        # Concat method
        mp_df_hist = pd.concat([mp_df_curr, mp_df_hist])
        mp_df_hist = mp_df_hist.drop_duplicates(subset=necessary_mp_cols)
        # Merge method
    #     mp_df_hist = pd.merge(
    #         mp_df_curr, 
    #         mp_df_hist, 
    #         how='outer', 
    #         left_on=necessary_mp_cols, 
    #         right_on=necessary_mp_cols
    #     )
    #     mp_df_hist = mp_df_hist.drop_duplicates(subset=necessary_mp_cols)
        #-------------------------
        if drop_approx_duplicates:
            dflt_args_drop_approx_mp_duplicates = MeterPremise.get_dflt_args_drop_approx_mp_duplicates(
                df_mp_serial_number_col = df_mp_serial_number_col, 
                df_mp_prem_nb_col       = df_mp_prem_nb_col, 
                df_mp_install_time_col  = df_mp_install_time_col, 
                df_mp_removal_time_col  = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col  = df_mp_trsf_pole_nb_col
            )
            drop_approx_duplicates_args = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = drop_approx_duplicates_args, 
                default_values_dict = dflt_args_drop_approx_mp_duplicates, 
                extend_any_lists    = False, 
                inplace             = True
            )
            mp_df_hist = MeterPremise.drop_approx_mp_duplicates(
                mp_df = mp_df_hist, 
                **drop_approx_duplicates_args
            )
        #-------------------------
        return mp_df_hist


    #TODO: Utilize this throughout MeterPremise 
    @staticmethod
    def build_mp_df_curr_hist_for_PNs(
        PNs                         , 
        mp_df_curr                  = None,
        mp_df_hist                  = None, 
        join_curr_hist              = False, 
        addtnl_mp_df_curr_cols      = None, 
        addtnl_mp_df_hist_cols      = None, 
        assert_all_PNs_found        = True, 
        assume_one_xfmr_per_PN      = True, 
        drop_approx_duplicates      = True, 
        drop_approx_duplicates_args = None, 
        df_mp_serial_number_col     = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col           = 'prem_nb', 
        df_mp_install_time_col      = 'inst_ts', 
        df_mp_removal_time_col      = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col      = 'trsf_pole_nb'
    ):
        r"""
        By default, necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
          are included in both mp_df_curr and mp_df_hist.
          mp_df_curr additionally includes df_mp_trsf_pole_nb_col by default.
        Additional columns can be included using addtnl_mp_df_curr_cols/addtnl_mp_df_hist_cols
        
        drop_approx_duplicates_args:
            Arguments already handled inside this function, so SHOULD NOT BE INCLUDED IN drop_approx_duplicates_args:
                mp_df
                df_mp_serial_number_col
                df_mp_prem_nb_col
                df_mp_install_time_col
                df_mp_removal_time_col
                df_mp_trsf_pole_nb_col
            Possible arguments to supply to drop_approx_duplicates_args (at time of writing, check drop_approx_mp_duplicates for
              most up-to-date info).  Anything else in drop_approx_duplicates_args will be ignored
                fuzziness
                assert_single_overlap
                addtnl_groupby_cols
                gpby_dropna
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(PNs, [list, str, int]))
        if not isinstance(PNs, list):
            PNs=[PNs]
        #-------------------------
        # Remove any NaN entries from PNs.
        # Also, make sure all entries in PNs are of the same type
        # NOTE: This check was originally performed after mp_df_curr/_hist was built.
        #       Moved up here to avoid wasting time when the check fails
        PNs      = [x for x in PNs if pd.notna(x)]
        PNs_type = type(PNs[0])
        assert(Utilities.are_all_list_elements_of_type(PNs, PNs_type))
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        # If mp_df_curr or mp_df_hist are not supplied, they will be built
        #-----
        if mp_df_curr is None:
            cols_of_interest_curr=necessary_mp_cols+[df_mp_trsf_pole_nb_col]
            if addtnl_mp_df_curr_cols is not None:
                cols_of_interest_curr.extend([x for x in addtnl_mp_df_curr_cols if x not in cols_of_interest_curr])
            mp_curr = MeterPremise(
                df_construct_type         = DFConstructType.kRunSqlQuery, 
                init_df_in_constructor    = True, 
                build_sql_function        = MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs = dict(
                    cols_of_interest=cols_of_interest_curr, 
                    premise_nbs = PNs, 
                    table_name  = 'meter_premise'
                )
            )
            mp_df_curr = mp_curr.df
        #-----
        if mp_df_hist is None:
            cols_of_interest_hist=necessary_mp_cols
            if addtnl_mp_df_hist_cols is not None:
                cols_of_interest_hist.extend([x for x in addtnl_mp_df_hist_cols if x not in cols_of_interest_hist])
            mp_hist = MeterPremise(
                df_construct_type         = DFConstructType.kRunSqlQuery, 
                init_df_in_constructor    = True, 
                build_sql_function        = MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs = dict(
                    cols_of_interest=cols_of_interest_hist, 
                    premise_nbs = PNs, 
                    table_name  = 'meter_premise_hist'
                )
            )
            mp_df_hist = mp_hist.df
        #-------------------------
        # At a bare minimum, mp_df_curr and mp_df_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in mp_df_curr.columns for x in necessary_mp_cols]))
        assert(all([x in mp_df_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        if assert_all_PNs_found:
            assert(len(set(PNs).difference(
                set(mp_df_curr[df_mp_prem_nb_col]).union(set(mp_df_hist[df_mp_prem_nb_col]))
            ))==0)
        #-------------------------
        # Make sure all dates are datetime objects, not e.g., strings
        if(not is_datetime64_dtype(mp_df_hist[df_mp_install_time_col]) or 
           not is_datetime64_dtype(mp_df_hist[df_mp_removal_time_col])):
            # If one isn't, chances are both are not (and no harm in converting both either way)
            mp_df_hist = Utilities_df.convert_col_types(
                df                  = mp_df_hist, 
                cols_and_types_dict = {
                    df_mp_install_time_col : datetime.datetime, 
                    df_mp_removal_time_col : datetime.datetime
                }
            )
        #-----
        if(not is_datetime64_dtype(mp_df_curr[df_mp_install_time_col]) 
           or not is_datetime64_dtype(mp_df_curr[df_mp_removal_time_col])):
            # If one isn't, chances are both are not (and no harm in converting both either way)
            mp_df_curr = Utilities_df.convert_col_types(
                df                  = mp_df_curr, 
                cols_and_types_dict = {
                    df_mp_install_time_col : datetime.datetime, 
                    df_mp_removal_time_col : datetime.datetime
                }
            )    
        #-------------------------
        # Make sure the premise numbers in PNs, mp_df_hist, and mp_df_curr are all of the same type
        #  NOTE: Only checking the first element of each to save time, so not a thorough check but should
        #        still perform as intended.  Also note, cannot do, e.g., mp_df_hist[df_mp_prem_nb_col].dtype,
        #        because when elements are strings this will come back as dtype('O'), not str!
        PNs_type = type(PNs[0])
        if mp_df_hist.shape[0]>0:
            assert(type(mp_df_hist.iloc[0][df_mp_prem_nb_col])==PNs_type)
        if mp_df_curr.shape[0]>0:
            assert(type(mp_df_curr.iloc[0][df_mp_prem_nb_col])==PNs_type)
        #-------------------------
        # From mp_df_curr/mp_df_hist, grab only the PNs needed
        mp_df_hist = mp_df_hist[mp_df_hist[df_mp_prem_nb_col].isin(PNs)]
        mp_df_curr = mp_df_curr[mp_df_curr[df_mp_prem_nb_col].isin(PNs)]
        #-------------------------
        # If assume_one_xfmr_per_PN==True, fill the trsf_pole_nbs in mp_df_hist using mp_df_curr
        if assume_one_xfmr_per_PN:
            mp_df_hist = MeterPremise.fill_trsf_pole_nbs_in_mp_df_hist_from_curr(
                mp_df_hist                = mp_df_hist, 
                mp_df_curr                = mp_df_curr, 
                prem_nb_col               = df_mp_prem_nb_col, 
                trsf_pole_nb_col          = df_mp_trsf_pole_nb_col, 
                how_to_handle_ambiguities = 'first'
            )
        #-------------------------
        if drop_approx_duplicates:
            dflt_args_drop_approx_mp_duplicates = MeterPremise.get_dflt_args_drop_approx_mp_duplicates(
                df_mp_serial_number_col = df_mp_serial_number_col, 
                df_mp_prem_nb_col       = df_mp_prem_nb_col, 
                df_mp_install_time_col  = df_mp_install_time_col, 
                df_mp_removal_time_col  = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col  = df_mp_trsf_pole_nb_col
            )
            drop_approx_duplicates_args = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = drop_approx_duplicates_args, 
                default_values_dict = dflt_args_drop_approx_mp_duplicates, 
                extend_any_lists    = False, 
                inplace             = True
            )
            #-----
            mp_df_hist = MeterPremise.drop_approx_mp_duplicates(
                mp_df = mp_df_hist, 
                **drop_approx_duplicates_args
            )
            #-----
            mp_df_curr = MeterPremise.drop_approx_mp_duplicates(
                mp_df = mp_df_curr, 
                **drop_approx_duplicates_args
            )
        #-------------------------
        if not join_curr_hist:
            return {'mp_df_curr':mp_df_curr, 'mp_df_hist':mp_df_hist}
        else:
            # NOTE: assume_one_xfmr_per_PN already handled above, so it would be wasteful to call it again in build_andor_join_mp_curr_hist
            #         Thus, assume_one_xfmr_per_PN=False below.
            #       Not same situation for drop_approx_duplicates, because although entries may be removed from mp_df_curr and mp_df_hist
            #         separately, once they are combined there could be other instances which need removed!
            mp_df_currhist = MeterPremise.build_andor_join_mp_curr_hist(
                mp_df_curr=mp_df_curr          , 
                mp_df_hist=mp_df_hist          , 
                build_sql_function_kwargs_curr = None,
                build_sql_function_kwargs_hist = None, 
                assume_one_xfmr_per_PN         = False, 
                drop_approx_duplicates         = drop_approx_duplicates, 
                drop_approx_duplicates_args    = drop_approx_duplicates_args, 
                df_mp_serial_number_col        = df_mp_serial_number_col, 
                df_mp_prem_nb_col              = df_mp_prem_nb_col, 
                df_mp_install_time_col         = df_mp_install_time_col, 
                df_mp_removal_time_col         = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col         = df_mp_trsf_pole_nb_col            
            )
            return mp_df_currhist


    #TODO: Utilize this throughout MeterPremise 
    @staticmethod
    def build_mp_df_curr_hist_for_xfmrs(
        trsf_pole_nbs               , 
        join_curr_hist              = False, 
        addtnl_mp_df_curr_cols      = None, 
        addtnl_mp_df_hist_cols      = None, 
        assume_one_xfmr_per_PN      = True, 
        drop_approx_duplicates      = True, 
        drop_approx_duplicates_args = None, 
        df_mp_serial_number_col     = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col           = 'prem_nb', 
        df_mp_install_time_col      = 'inst_ts', 
        df_mp_removal_time_col      = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col      = 'trsf_pole_nb'
    ):
        r"""
        Difficulty is that default.meter_premise_hist does not have trsf_pole_nb field.
        Therefore, one must use default.meter_premise to find the premise numbers for xfrms in trsf_pole_nbs,
          then use those PNs to select the correct entries from default.meter_premise_hist.
        """
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        #-----
        cols_of_interest_curr = necessary_mp_cols+[df_mp_trsf_pole_nb_col]
        if addtnl_mp_df_curr_cols is not None:
            cols_of_interest_curr.extend([x for x in addtnl_mp_df_curr_cols if x not in cols_of_interest_curr])
        mp_curr = MeterPremise(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            init_df_in_constructor    = True, 
            build_sql_function        = MeterPremise.build_sql_meter_premise, 
            build_sql_function_kwargs = dict(
                cols_of_interest = cols_of_interest_curr, 
                trsf_pole_nbs    = trsf_pole_nbs, 
                field_to_split   = 'trsf_pole_nbs', 
                batch_size       = 1000, 
                n_update         = 10, 
                table_name       = 'meter_premise'
            )
        )
        mp_df_curr = mp_curr.df
        #-------------------------
        PNs = mp_df_curr[df_mp_prem_nb_col].unique().tolist()
        #-------------------------
        cols_of_interest_hist = necessary_mp_cols
        if addtnl_mp_df_hist_cols is not None:
            cols_of_interest_hist.extend([x for x in addtnl_mp_df_hist_cols if x not in cols_of_interest_hist])
        mp_hist = MeterPremise(
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            init_df_in_constructor    = True, 
            build_sql_function        = MeterPremise.build_sql_meter_premise, 
            build_sql_function_kwargs = dict(
                cols_of_interest = cols_of_interest_hist, 
                premise_nbs      = PNs, 
                field_to_split   = 'premise_nbs', 
                batch_size       = 1000, 
                n_update         = 10, 
                table_name       = 'meter_premise_hist'
            )
        )
        mp_df_hist = mp_hist.df
        #-------------------------
        # If assume_one_xfmr_per_PN==True, fill the trsf_pole_nbs in mp_df_hist using mp_df_curr
        if assume_one_xfmr_per_PN:
            mp_df_hist = MeterPremise.fill_trsf_pole_nbs_in_mp_df_hist_from_curr(
                mp_df_hist                = mp_df_hist, 
                mp_df_curr                = mp_df_curr, 
                prem_nb_col               = df_mp_prem_nb_col, 
                trsf_pole_nb_col          = df_mp_trsf_pole_nb_col, 
                how_to_handle_ambiguities = 'first'
            )
        #-------------------------
        if drop_approx_duplicates:
            dflt_args_drop_approx_mp_duplicates = MeterPremise.get_dflt_args_drop_approx_mp_duplicates(
                df_mp_serial_number_col = df_mp_serial_number_col, 
                df_mp_prem_nb_col       = df_mp_prem_nb_col, 
                df_mp_install_time_col  = df_mp_install_time_col, 
                df_mp_removal_time_col  = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col  = df_mp_trsf_pole_nb_col
            )
            drop_approx_duplicates_args = Utilities.supplement_dict_with_default_values(
                to_supplmnt_dict    = drop_approx_duplicates_args, 
                default_values_dict = dflt_args_drop_approx_mp_duplicates, 
                extend_any_lists    = False, 
                inplace             = True
            )
            #-----
            mp_df_hist = MeterPremise.drop_approx_mp_duplicates(
                mp_df = mp_df_hist, 
                **drop_approx_duplicates_args
            )
            #-----
            mp_df_curr = MeterPremise.drop_approx_mp_duplicates(
                mp_df = mp_df_curr, 
                **drop_approx_duplicates_args
            )
        #-------------------------
        if not join_curr_hist:
            return {'mp_df_curr':mp_df_curr, 'mp_df_hist':mp_df_hist}
        else:
            # NOTE: assume_one_xfmr_per_PN already handled above, so it would be wasteful to call it again in build_andor_join_mp_curr_hist
            #         Thus, assume_one_xfmr_per_PN=False below.
            #       Not same situation for drop_approx_duplicates, because although entries may be removed from mp_df_curr and mp_df_hist
            #         separately, once they are combined there could be other instances which need removed!
            mp_df_currhist = MeterPremise.build_andor_join_mp_curr_hist(
                mp_df_curr=mp_df_curr          , 
                mp_df_hist=mp_df_hist          , 
                build_sql_function_kwargs_curr = None,
                build_sql_function_kwargs_hist = None, 
                assume_one_xfmr_per_PN         = False, 
                drop_approx_duplicates         = drop_approx_duplicates, 
                drop_approx_duplicates_args    = drop_approx_duplicates_args, 
                df_mp_serial_number_col        = df_mp_serial_number_col, 
                df_mp_prem_nb_col              = df_mp_prem_nb_col, 
                df_mp_install_time_col         = df_mp_install_time_col, 
                df_mp_removal_time_col         = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col         = df_mp_trsf_pole_nb_col            
            )
            return mp_df_currhist
        
        
        
    #****************************************************************************************************
    @staticmethod
    def get_historic_SNs_for_PNs(
        PNs                         , 
        df_mp_curr                  , 
        df_mp_hist                  , 
        addtnl_mp_df_curr_cols      = None, 
        addtnl_mp_df_hist_cols      = None, 
        assume_one_xfmr_per_PN      = True, 
        output_index                = ['trsf_pole_nb', 'prem_nb'],
        output_groupby              = None, 
        assert_all_PNs_found        = True, 
        drop_approx_duplicates      = True, 
        drop_approx_duplicates_args = None, 
        df_mp_serial_number_col     = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col           = 'prem_nb', 
        df_mp_install_time_col      = 'inst_ts', 
        df_mp_removal_time_col      = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col      = 'trsf_pole_nb'
    ):
        r"""
        NOTE: Both df_mp_curr and df_mp_hist are needed because meter_premise_hist does not seem to be completely up-to-date.
              Therefore, one should actually use the current meter_premise together with the historical

        PNs:
          Designed to be a list of premise numbers, but a single premise number will work as well

        df_mp_curr:
          Current meter premise DF, taken from default.meter_premise
          IF NOT SUPPLIED (i.e., if is None), WILL BE BUILT

        df_mp_hist:
          Historical meter premise DF, taken from default.meter_premise_hist
          IF NOT SUPPLIED (i.e., if is None), WILL BE BUILT
          
        addtnl_mp_df_curr_cols/addtnl_mp_df_hist_cols:
          Only utilized when df_mp_curr/df_mp_hist is None, in which case can be used to add columns of interest
            to the default grabbed.

        output_index:
          Columns to use as index for the output.  If None, the DF will be returned just as essentially df_mp_hist joined
            with df_mp_curr
            
        output_groupby:
          If None, the DF will be returned just as essentially df_mp_hist joined with df_mp_curr.
          If not None, this should be a column or list of columns.  In this case, the DF which is returned will only include
            the output_groupby as index (or indices, if multiple), and single column of all serial numbers for each group
            collected in a list.  The return object will be a pd.Series.
            i.e., it will return df_mp_hist.groupby(output_groupby)[df_mp_serial_number_col].apply(lambda x: list(set(x)))

        assume_one_xfmr_per_PN:
          The historical data are lacking the transformer number information.  If assume_one_xfmr_per_PN is set to True,
            any missing transformer numbers for historical entries will be filled with the transformer number inferred from
            the current meter premise data.
        """
        #-------------------------
        mp_df_currhist = MeterPremise.build_mp_df_curr_hist_for_PNs(
            PNs                         = PNs, 
            mp_df_curr                  = df_mp_curr,
            mp_df_hist                  = df_mp_hist, 
            join_curr_hist              = True, 
            addtnl_mp_df_curr_cols      = addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols      = addtnl_mp_df_hist_cols, 
            assert_all_PNs_found        = assert_all_PNs_found, 
            assume_one_xfmr_per_PN      = assume_one_xfmr_per_PN, 
            drop_approx_duplicates      = drop_approx_duplicates, 
            drop_approx_duplicates_args = drop_approx_duplicates_args, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col      = df_mp_trsf_pole_nb_col
        )
        #-------------------------
        # Set index, if specified
        if output_index:
            mp_df_currhist = mp_df_currhist.set_index(output_index).sort_index()
        #-------------------------
        # Groupby, if specified
        # NOTE: In this case, the DF which is returned will only include the output_groupby as index (or indices, if multiple), 
        #       and single column of all serial numbers for each group collected in a list.
        #       The return object will be a pd.Series
        if output_groupby:
            mp_df_currhist = mp_df_currhist.groupby(output_groupby)[df_mp_serial_number_col].apply(lambda x: sorted(list(set(x))))
        #-------------------------
        return mp_df_currhist



    #TODO!!!!!!!!! : use build_andor_join_mp_curr_hist here!
    @staticmethod
    def get_historic_SNs_for_PNs_in_df(
        df                          , 
        df_mp_curr                  , 
        df_mp_hist                  , 
        addtnl_mp_df_curr_cols      = None, 
        addtnl_mp_df_hist_cols      = None,
        assume_one_xfmr_per_PN      = True, 
        output_index                = ['trsf_pole_nb', 'prem_nb'],
        output_groupby              = None, 
        assert_all_PNs_found        = True, 
        drop_approx_duplicates      = True, 
        drop_approx_duplicates_args = None, 
        df_prem_nb_col              = 'prem_nb', 
        df_mp_serial_number_col     = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col           = 'prem_nb', 
        df_mp_install_time_col      = 'inst_ts', 
        df_mp_removal_time_col      = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col      = 'trsf_pole_nb'
    ):
        r"""
        NOTE: Both df_mp_curr and df_mp_hist are needed because meter_premise_hist does not seem to be completely up-to-date.
              Therefore, one should actually use the current meter_premise together with the historical

        df_mp_curr:
          Current meter premise DF, taken from default.meter_premise

        df_mp_hist:
          Historical meter premise DF, taken from default.meter_premise_hist

        output_index:
          Columns to use as index for the output.  If None, the DF will be returned just as essentially df_mp_hist joined
            with df_mp_curr

        assume_one_xfmr_per_PN:
          The historical data are lacking the transformer number information.  If assume_one_xfmr_per_PN is set to True,
            any missing transformer numbers for historical entries will be filled with the transformer number inferred from
            the current meter premise data.
        """
        #-------------------------
        # Make sure the premise numbers in df, df_mp_hist, and df_mp_curr are all of the same type (if the latter two are not None)
        if df_mp_curr is not None:
            assert(df[df_prem_nb_col].dtype==df_mp_curr[df_mp_prem_nb_col].dtype)
        if df_mp_hist is not None:
            assert(df[df_prem_nb_col].dtype==df_mp_hist[df_mp_prem_nb_col].dtype)
        #-------------------------
        # Grab PNs from df and feed everything into get_historic_SNs_for_PNs
        PNs = df[df_prem_nb_col].unique().tolist()
        #-------------------------
        SNs_for_PNs = MeterPremise.get_historic_SNs_for_PNs(
            PNs                         = PNs, 
            df_mp_curr                  = df_mp_curr, 
            df_mp_hist                  = df_mp_hist, 
            addtnl_mp_df_curr_cols      = addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols      = addtnl_mp_df_hist_cols, 
            assume_one_xfmr_per_PN      = assume_one_xfmr_per_PN, 
            output_index                = output_index,
            output_groupby              = output_groupby, 
            assert_all_PNs_found        = assert_all_PNs_found, 
            drop_approx_duplicates      = drop_approx_duplicates, 
            drop_approx_duplicates_args = drop_approx_duplicates_args, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col      = df_mp_trsf_pole_nb_col
        )
        return SNs_for_PNs


    @staticmethod
    def get_active_SNs_for_PNs_at_datetime_interval(
        PNs                                      , 
        df_mp_curr                               , 
        df_mp_hist                               , 
        dt_0                                     , 
        dt_1                                     , 
        addtnl_mp_df_curr_cols                   = None, 
        addtnl_mp_df_hist_cols                   = None,
        assume_one_xfmr_per_PN                   = True, 
        output_index                             = ['trsf_pole_nb', 'prem_nb'],
        output_groupby                           = None, 
        include_prems_wo_active_SNs_when_groupby = True, 
        assert_all_PNs_found                     = True, 
        drop_approx_duplicates                   = True, 
        drop_approx_duplicates_args              = None, 
        df_mp_serial_number_col                  = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col                        = 'prem_nb', 
        df_mp_install_time_col                   = 'inst_ts', 
        df_mp_removal_time_col                   = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col                   = 'trsf_pole_nb'
    ):
        r"""
        NOTE: Both df_mp_curr and df_mp_hist are needed because meter_premise_hist does not seem to be completely up-to-date.
              Therefore, one should actually use the current meter_premise together with the historical

        df_mp_curr:
          Current meter premise DF, taken from default.meter_premise

        df_mp_hist:
          Historical meter premise DF, taken from default.meter_premise_hist

        dt_0:
          Lower limit datetime
        dt_1:
          Upper limit datetime

        output_index:
          Columns to use as index for the output.  If None, the DF will be returned just as essentially df_mp_hist joined
            with df_mp_curr

        output_groupby:
          If None, the DF will be returned just as essentially df_mp_hist joined with df_mp_curr.
          If not None, this should be a column or list of columns.  In this case, the DF which is returned will only include
            the output_groupby as index (or indices, if multiple), and single column of all serial numbers for each group
            collected in a list.  The return object will be a pd.Series.
            i.e., it will return active_SNs_at_time.groupby(output_groupby)[df_mp_serial_number_col].apply(lambda x: list(set(x)))

        include_prems_wo_active_SNs_when_groupby:
          Only has an effect when output_groupby is not None.  So, for description below, assume that to be true.
          If False:
            Premise numbers without any active SNs at the time will be left out of returned series
          If True:
            Premise numbers without any active SNs at the time will be included, and their value will be empty lists.

        assume_one_xfmr_per_PN:
          The historical data are lacking the transformer number information.  If assume_one_xfmr_per_PN is set to True,
            any missing transformer numbers for historical entries will be filled with the transformer number inferred from
            the current meter premise data.
        """
        #-------------------------
        # Make sure all dates are datetime object, not e.g., strings
        dt_0 = pd.to_datetime(dt_0)
        dt_1 = pd.to_datetime(dt_1)
        #-------------------------
        # Build df_mp_hist
        df_mp_hist = MeterPremise.get_historic_SNs_for_PNs( 
            PNs                         = PNs, 
            df_mp_curr                  = df_mp_curr, 
            df_mp_hist                  = df_mp_hist, 
            addtnl_mp_df_curr_cols      = addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols      = addtnl_mp_df_hist_cols, 
            assume_one_xfmr_per_PN      = assume_one_xfmr_per_PN, 
            output_index                = None,
            output_groupby              = None, 
            assert_all_PNs_found        = assert_all_PNs_found, 
            drop_approx_duplicates      = drop_approx_duplicates, 
            drop_approx_duplicates_args = drop_approx_duplicates_args, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col      = df_mp_trsf_pole_nb_col
        )
        #-------------------------
        if df_mp_hist.shape[0]==0:
            return pd.DataFrame()
        # Enforce the time criteria
        active_SNs_at_time = df_mp_hist[(df_mp_hist[df_mp_install_time_col]                         <= pd.to_datetime(dt_0)) & 
                                        (df_mp_hist[df_mp_removal_time_col].fillna(pd.Timestamp.max) > pd.to_datetime(dt_1))]
        #-------------------------
        if include_prems_wo_active_SNs_when_groupby and output_groupby is not None:
            prems_wo_active_SNs = list(set(PNs).difference(set(active_SNs_at_time[df_mp_prem_nb_col])))
            df_mp_hist_wo       = df_mp_hist[df_mp_hist[df_mp_prem_nb_col].isin(prems_wo_active_SNs)]
            # Below line simply makes a pd.Series object with index (indices) equal to output_groupby values and values
            #   equal to empty lists. This is much quicker than my old method.
            prems_wo_active_SNs = df_mp_hist_wo.groupby(output_groupby)[df_mp_serial_number_col].agg(lambda x: [])
        #-------------------------
        # Set index, if specified
        if output_index:
            active_SNs_at_time = active_SNs_at_time.set_index(output_index).sort_index()
        #-------------------------
        # Groupby, if specified
        # NOTE: In this case, the DF which is returned will only include the output_groupby as index (or indices, if multiple), 
        #       and single column of all serial numbers for each group collected in a list.
        #       The return object will be a pd.Series
        if output_groupby:
            active_SNs_at_time = active_SNs_at_time.groupby(output_groupby)[df_mp_serial_number_col].apply(lambda x: sorted(list(set(x))))
            if include_prems_wo_active_SNs_when_groupby:
                # Depending on what columns are used for output_groupby, it is possible that an index is found in both
                #   active_SNs_at_time and prems_wo_active_SNs.  For instance, if grouping by only trsf_pole_nb, if some of
                #   the SNs were active, but others were not, then the index would be in both.  In this case, in the final
                #   pd.Series, we don't want then entry from prems_wo_active_SNs (if included, there would be an entry for
                #   the index with the SNs included, and an entry with an empty list)
                prems_wo_active_SNs = prems_wo_active_SNs.loc[~prems_wo_active_SNs.index.isin(active_SNs_at_time.index)] 
                active_SNs_at_time  = pd.concat([active_SNs_at_time, prems_wo_active_SNs])
                active_SNs_at_time  = active_SNs_at_time.sort_index()
        #-------------------------
        return active_SNs_at_time


    @staticmethod
    def get_active_SNs_for_PNs_in_df_at_datetime_interval(
        df                                       , 
        df_mp_curr                               , 
        df_mp_hist                               , 
        dt_0                                     , 
        dt_1                                     , 
        assume_one_xfmr_per_PN                   = True, 
        output_index                             = ['trsf_pole_nb', 'prem_nb'],
        output_groupby                           = None, 
        include_prems_wo_active_SNs_when_groupby = True, 
        assert_all_PNs_found                     = True, 
        drop_approx_duplicates                   = True, 
        drop_approx_duplicates_args              = None, 
        df_prem_nb_col                           = 'prem_nb', 
        df_mp_serial_number_col                  = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col                        = 'prem_nb', 
        df_mp_install_time_col                   = 'inst_ts', 
        df_mp_removal_time_col                   = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col                   = 'trsf_pole_nb'
    ):
        r"""
        NOTE: Both df_mp_curr and df_mp_hist are needed because meter_premise_hist does not seem to be completely up-to-date.
              Therefore, one should actually use the current meter_premise together with the historical

        df_mp_curr:
          Current meter premise DF, taken from default.meter_premise

        df_mp_hist:
          Historical meter premise DF, taken from default.meter_premise_hist

        dt_0:
          Lower limit datetime
        dt_1:
          Upper limit datetime

        output_index:
          Columns to use as index for the output.  If None, the DF will be returned just as essentially df_mp_hist joined
            with df_mp_curr

        output_groupby:
          If None, the DF will be returned just as essentially df_mp_hist joined with df_mp_curr.
          If not None, this should be a column or list of columns.  In this case, the DF which is returned will only include
            the output_groupby as index (or indices, if multiple), and single column of all serial numbers for each group
            collected in a list.  The return object will be a pd.Series.
            i.e., it will return active_SNs_at_time.groupby(output_groupby)[df_mp_serial_number_col].apply(lambda x: list(set(x)))

        include_prems_wo_active_SNs_when_groupby:
          Only has an effect when output_groupby is not None.  So, for description below, assume that to be true.
          If False:
            Premise numbers without any active SNs at the time will be left out of returned series
          If True:
            Premise numbers without any active SNs at the time will be included, and their value will be empty lists.

        assume_one_xfmr_per_PN:
          The historical data are lacking the transformer number information.  If assume_one_xfmr_per_PN is set to True,
            any missing transformer numbers for historical entries will be filled with the transformer number inferred from
            the current meter premise data.
        """
        #-------------------------
        # Grab PNs from df and feed everything into get_active_SNs_for_PNs_at_datetime_interval
        PNs = df[df_prem_nb_col].unique().tolist()
        #-------------------------
        active_SNs_for_PNs_at_dt = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
            PNs                                      = PNs, 
            df_mp_curr                               = df_mp_curr, 
            df_mp_hist                               = df_mp_hist, 
            dt_0                                     = dt_0, 
            dt_1                                     = dt_1, 
            assume_one_xfmr_per_PN                   = assume_one_xfmr_per_PN, 
            output_index                             = output_index,
            output_groupby                           = output_groupby, 
            include_prems_wo_active_SNs_when_groupby = include_prems_wo_active_SNs_when_groupby, 
            assert_all_PNs_found                     = assert_all_PNs_found, 
            drop_approx_duplicates                   = drop_approx_duplicates, 
            drop_approx_duplicates_args              = drop_approx_duplicates_args, 
            df_mp_serial_number_col                  = df_mp_serial_number_col, 
            df_mp_prem_nb_col                        = df_mp_prem_nb_col, 
            df_mp_install_time_col                   = df_mp_install_time_col, 
            df_mp_removal_time_col                   = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col                   = df_mp_trsf_pole_nb_col
        )
        return active_SNs_for_PNs_at_dt


    @staticmethod
    def get_active_SNs_for_PN_at_datetime_interval(
        PN                          , 
        df_mp_curr                  , 
        df_mp_hist                  , 
        dt_0                        , 
        dt_1                        , 
        assume_one_xfmr_per_PN      = True, 
        drop_approx_duplicates      = True, 
        drop_approx_duplicates_args = None, 
        df_mp_serial_number_col     = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col           = 'prem_nb', 
        df_mp_install_time_col      = 'inst_ts', 
        df_mp_removal_time_col      = 'rmvl_ts'
    ):
        r"""
        PN:
          Premise number
        df_mp_hist:
          meter_premise_hist DF
        dt_0:
          Lower limit datetime
        dt_1:
          Upper limit datetime
        """
        #-------------------------
        SNs_for_PN_at_dt = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
            PNs                                      = PN, 
            df_mp_curr                               = df_mp_curr, 
            df_mp_hist                               = df_mp_hist, 
            dt_0                                     = dt_0, 
            dt_1                                     = dt_1, 
            assume_one_xfmr_per_PN                   = assume_one_xfmr_per_PN, 
            output_index                             = None,
            output_groupby                           = None, 
            include_prems_wo_active_SNs_when_groupby = False, 
            assert_all_PNs_found                     = True, 
            drop_approx_duplicates                   = drop_approx_duplicates, 
            drop_approx_duplicates_args              = drop_approx_duplicates_args, 
            df_mp_serial_number_col                  = df_mp_serial_number_col, 
            df_mp_prem_nb_col                        = df_mp_prem_nb_col, 
            df_mp_install_time_col                   = df_mp_install_time_col, 
            df_mp_removal_time_col                   = df_mp_removal_time_col
        )
        #-------------------------
        SNs_for_PN_at_dt = SNs_for_PN_at_dt[df_mp_serial_number_col].unique().tolist()
        return SNs_for_PN_at_dt


    @staticmethod
    def get_active_SNs_for_PN_at_datetime(
        PN                          , 
        df_mp_curr                  , 
        df_mp_hist                  , 
        dt_0                        , 
        assume_one_xfmr_per_PN      = True, 
        drop_approx_duplicates      = True, 
        drop_approx_duplicates_args = None, 
        df_mp_serial_number_col     = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col           = 'prem_nb', 
        df_mp_install_time_col      = 'inst_ts', 
        df_mp_removal_time_col      = 'rmvl_ts'
    ):
        r"""
        prem_nb:
          Premise number
        df_mp_hist:
          meter_premise_hist DF
        dt_0:
          Datetime
        """
        #-------------------------
        return MeterPremise.get_active_SNs_for_PN_at_datetime_interval(
            PN                          = PN, 
            df_mp_curr                  = df_mp_curr, 
            df_mp_hist                  = df_mp_hist, 
            dt_0                        = dt_0, 
            dt_1                        = dt_0, 
            assume_one_xfmr_per_PN      = assume_one_xfmr_per_PN, 
            drop_approx_duplicates      = drop_approx_duplicates, 
            drop_approx_duplicates_args = drop_approx_duplicates_args, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col
        )
        
        
    @staticmethod
    def get_SNs_andor_PNs_for_xfmrs(
        trsf_pole_nbs                  , 
        include_SNs                    = True,
        include_PNs                    = True,
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        serial_number_col              = 'mfr_devc_ser_nbr', 
        prem_nb_col                    = 'prem_nb', 
        return_SNs_col                 = 'SNs',
        return_PNs_col                 = 'prem_nbs', 
        assert_all_trsf_pole_nbs_found = True, 
        mp_df                          = None, 
        return_mp_df_also              = False
    ):
        r"""
        Oddly enough, in testing, the largest batch_size seemed to be 16,126
          Set it equal to 15,000 for rounder number
          
        NOTE: One should probably actually use get_active_SNs_and_PNs_for_xfmrs_at_datetime_interval instead!  
          
        mp_df:
          Meter Premise DF can be supplied by used to speed things up.
          If supplied, it is important for the user to ensure the DF contains all of the 
            needed trsf_pole_nbs
        """
        #-------------------------
        assert(include_SNs or include_PNs) #Need to include at least one of them!!!!!
        #-------------------------
        cols_of_interest = [trsf_pole_nb_col]
        if include_SNs:
            cols_of_interest.append(serial_number_col)
        if include_PNs:
            cols_of_interest.append(prem_nb_col)
        #-------------------------
        if mp_df is None:
            mp = MeterPremise(
                df_construct_type         = DFConstructType.kRunSqlQuery, 
                contstruct_df_args        = None, 
                init_df_in_constructor    = True, 
                build_sql_function        = MeterPremise.build_sql_meter_premise, 
                build_sql_function_kwargs = dict(
                    cols_of_interest = cols_of_interest,  
                    trsf_pole_nbs    = trsf_pole_nbs, 
                    field_to_split   = 'trsf_pole_nbs', 
                    batch_size       = 15000,
                    n_update         = 1000
                )
            )
            mp_df = mp.df
        else:
            mp_df = mp_df[mp_df[trsf_pole_nb_col].isin(trsf_pole_nbs)]
        #-------------------------
        # Make sure all trsf_pole_nbs found in mp_df
        if assert_all_trsf_pole_nbs_found:
            not_found   = set(trsf_pole_nbs).difference(set(mp_df[trsf_pole_nb_col]))
            n_not_found = len(not_found)
            if n_not_found!=0:
                print(f'n_not_found = {n_not_found}')
                print(*not_found, sep='\n')
            assert(n_not_found==0)
        #-------------------------
        # Remove trsf_pole_nb_col from cols_of_interest so cols_of_interest can then be used in selection
        # after groupby (NOTE: cols_of_interest.remove(trsf_pole_nb_col) removes the element and returns None,
        # so one could not simply put it in the groupby line)
        cols_of_interest.remove(trsf_pole_nb_col)
        return_df = mp_df.groupby(trsf_pole_nb_col)[cols_of_interest].agg(lambda x: list(set(x)))
        return_df = return_df.rename(columns={
            serial_number_col : return_SNs_col, 
            prem_nb_col       : return_PNs_col
        })
        #-------------------------
        if return_mp_df_also:
            return return_df, mp_df
        else:
            return return_df
        
        
    @staticmethod
    def get_SNs_for_xfmrs(
        trsf_pole_nbs                  ,
        include_prem_nbs               = True,
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        serial_number_col              = 'mfr_devc_ser_nbr', 
        prem_nb_col                    = 'prem_nb', 
        return_SNs_col                 = 'SNs',
        return_prem_nbs_col            = 'prem_nbs', 
        assert_all_trsf_pole_nbs_found = True, 
        mp_df                          = None, 
        return_mp_df_also              = False
    ):
        r"""
        Oddly enough, in testing, the largest batch_size seemed to be 16,126
          Set it equal to 15,000 for rounder number
          
        NOTE: One should probably actually use get_active_SNs_and_PNs_for_xfmrs_at_datetime_interval instead!  
          
        mp_df:
          Meter Premise DF can be supplied by used to speed things up.
          If supplied, it is important for the user to ensure the DF contains all of the 
            needed trsf_pole_nbs
        """
        #-------------------------
        return MeterPremise.get_SNs_andor_PNs_for_xfmrs(
            trsf_pole_nbs                  = trsf_pole_nbs,
            include_SNs                    = True,
            include_PNs                    = include_prem_nbs,
            trsf_pole_nb_col               = trsf_pole_nb_col, 
            serial_number_col              = serial_number_col, 
            prem_nb_col                    = prem_nb_col, 
            return_SNs_col                 = return_SNs_col,
            return_PNs_col                 = return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found = assert_all_trsf_pole_nbs_found, 
            mp_df                          = mp_df, 
            return_mp_df_also              = return_mp_df_also
        )
    
    @staticmethod
    def get_active_SNs_and_PNs_for_xfmrs_at_datetime_interval(
        trsf_pole_nbs,
        dt_0, 
        dt_1, 
        addtnl_mp_df_curr_cols                   = None, 
        addtnl_mp_df_hist_cols                   = None,
        assume_one_xfmr_per_PN                   = True, 
        assert_all_trsf_pole_nbs_found           = True, 
        drop_approx_duplicates                   = True, 
        drop_approx_duplicates_args              = None, 
        df_mp_serial_number_col                  = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col                        = 'prem_nb', 
        df_mp_install_time_col                   = 'inst_ts', 
        df_mp_removal_time_col                   = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col                   = 'trsf_pole_nb', 
        return_PNs_only                          = False
    ):
        r"""
        """
        #-------------------------
        # Make sure all dates are datetime object, not e.g., strings
        dt_0 = pd.to_datetime(dt_0)
        dt_1 = pd.to_datetime(dt_1)
        #-------------------------
        mp_df_currhist = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
            trsf_pole_nbs               = trsf_pole_nbs, 
            join_curr_hist              = True, 
            addtnl_mp_df_curr_cols      = addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols      = addtnl_mp_df_hist_cols, 
            assume_one_xfmr_per_PN      = assume_one_xfmr_per_PN, 
            drop_approx_duplicates      = drop_approx_duplicates, 
            drop_approx_duplicates_args = drop_approx_duplicates_args, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col      = df_mp_trsf_pole_nb_col
        )
        #-------------------------
        if assert_all_trsf_pole_nbs_found:
            not_found = set(trsf_pole_nbs).difference(set(mp_df_currhist[df_mp_trsf_pole_nb_col]))
            n_not_found = len(not_found)
            if n_not_found!=0:
                print(f'n_not_found = {n_not_found}')
                print(*not_found, sep='\n')
            assert(n_not_found==0)
        #-------------------------
        if mp_df_currhist.shape[0]==0:
            return pd.DataFrame()
        #-------------------------
        # Enforce the time criteria
        active_SNs_at_time = mp_df_currhist[
            (mp_df_currhist[df_mp_install_time_col]<=pd.to_datetime(dt_0)) & 
            (mp_df_currhist[df_mp_removal_time_col].fillna(pd.Timestamp.max)>pd.to_datetime(dt_1))
        ]
        #-------------------------
        if return_PNs_only:
            return active_SNs_at_time[df_mp_prem_nb_col].unique().tolist()
        else:
            return active_SNs_at_time

        
    @staticmethod
    def get_active_SNs_and_PNs_for_xfmrs_in_df(
        df, 
        trsf_pole_nb_col                         = 'trsf_pole_nb', 
        t_min_col                                = 't_search_min', 
        t_max_col                                = 't_search_max', 
        addtnl_mp_df_curr_cols                   = None, 
        addtnl_mp_df_hist_cols                   = None,
        assume_one_xfmr_per_PN                   = True, 
        assert_all_trsf_pole_nbs_found           = True, 
        drop_approx_duplicates                   = True, 
        drop_approx_duplicates_args              = None, 
        df_mp_serial_number_col                  = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col                        = 'prem_nb', 
        df_mp_install_time_col                   = 'inst_ts', 
        df_mp_removal_time_col                   = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col                   = 'trsf_pole_nb', 
        verbose                                  = False
    ):
        r"""
        """
        #--------------------------------------------------
        nec_cols = [trsf_pole_nb_col, t_min_col, t_max_col]
        assert(set(nec_cols).difference(set(df.columns.tolist()))==set())
        #-----
        df = df[nec_cols].copy()
        #-------------------------
        # Make sure all dates are datetime object, not e.g., strings
        df = Utilities_df.convert_col_types(
            df                  = df,
            cols_and_types_dict = {
                t_min_col : datetime.datetime, 
                t_max_col : datetime.datetime
            },
            to_numeric_errors   = 'coerce',
            inplace             = False
        )
        #-----
        date_0 = df[t_min_col].min()
        date_1 = df[t_max_col].max()
        if verbose:
            print("Date range collected for ALL trsf_pole_nbs:")
            print(f"\t{date_0} to\n\t{date_1}")
        #-------------------------
        trsf_pole_nbs = df[trsf_pole_nb_col].unique().tolist()
        #-------------------------
        mp_df_currhist = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
            trsf_pole_nbs               = trsf_pole_nbs, 
            join_curr_hist              = True, 
            addtnl_mp_df_curr_cols      = addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols      = addtnl_mp_df_hist_cols, 
            assume_one_xfmr_per_PN      = assume_one_xfmr_per_PN, 
            drop_approx_duplicates      = drop_approx_duplicates, 
            drop_approx_duplicates_args = drop_approx_duplicates_args, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col      = df_mp_trsf_pole_nb_col
        )
        #-----
        mp_df_currhist = mp_df_currhist.sort_values(by=[df_mp_trsf_pole_nb_col], ignore_index=True)
        #-------------------------
        if assert_all_trsf_pole_nbs_found:
            not_found = set(trsf_pole_nbs).difference(set(mp_df_currhist[df_mp_trsf_pole_nb_col]))
            n_not_found = len(not_found)
            if n_not_found!=0:
                print(f'n_not_found = {n_not_found}')
                print(*not_found, sep='\n')
            assert(n_not_found==0)
        #-------------------------
        if mp_df_currhist.shape[0]==0:
            return pd.DataFrame()
        #-------------------------
        df = pd.merge(
            df, 
            mp_df_currhist, 
            how      = 'left', 
            left_on  = trsf_pole_nb_col, 
            right_on = df_mp_trsf_pole_nb_col
        )
        
        #-------------------------
        # Enforce the time criteria
        df = df[
            (df[df_mp_install_time_col]                          <= df[t_min_col]) & 
            (df[df_mp_removal_time_col].fillna(pd.Timestamp.max) >  df[t_max_col])
        ]
    
        #-------------------------
        return df

    @staticmethod
    def simple_merge_df_with_active_mp(
        df, 
        df_mp, 
        df_time_col_0,
        df_time_col_1                   = None,
        df_and_mp_merge_pairs           = [
            ['serialnumber', 'mfr_devc_ser_nbr'], 
            ['aep_premise_nb', 'prem_nb']
        ], 
        keep_overlap                    = 'right', 
        drop_inst_rmvl_cols_after_merge = True, 
        df_mp_install_time_col          = 'inst_ts', 
        df_mp_removal_time_col          = 'rmvl_ts', 
        assert_max_one_to_one           = False
    ):
        r"""
        Get the active SNs for the PNs in df at the time given in the df_time_col_0 column, or between the times given
          in df_time_col_0 and df_time_col_1 if df_time_col_1 is not None.

        This was originally built so that the correct trsf_pole_nbs could be added to AMIEndEvents for the case of outages.  
          Doing this at collection time with SQL is typically impossible, as one must find the meter which was active at the
          time of the event.

        NOTE: Both df_mp_curr and df_mp_hist are needed because meter_premise_hist does not seem to be completely up-to-date.
              Therefore, one should actually use the current meter_premise together with the historical

        df:
          DF with which to join the active MP data

        df_time_col_0/_1:
          The column in df containing time information to use to determine which meter(s) were active.
          This column must contain datetime data!

        df_and_mp_merge_pairs:
          This defines how df and MeterPremise DF, df_mp, will be merged.
          This should be a list of pairs, where the first member of the pair is a column from df to use in the merge
            and the second member of the pair is the associated column from df_mp to use.

        keep_overlap:
          Must be equal to 'right', 'left' or 'both'.
          When there are overlapping columns in df and df_mp, this determined which should be kept in the finally returned DF
          'right' (default): Keep overlap from df_mp, get rid of those from df
          'left': Keep overlap from df, get rid of those from df_mp
          'both': Keep overlap from both.  Those from df will have suffix '_x', and those from df_mp will have suffix '_y'
        """
        #-------------------------
        keep_overlap = keep_overlap.lower()
        #-------------------------
        if df_time_col_1 is None:
            df_time_col_1=df_time_col_0
        # The data in df[df_time_col_0/_1] must be datetime
        assert(is_datetime64_dtype(df[df_time_col_0]))
        assert(is_datetime64_dtype(df[df_time_col_1]))
        #-------------------------
        # Compile left_on and right_on from df_and_mp_merge_pairs
        #----------
        assert(Utilities.is_object_one_of_types(df_and_mp_merge_pairs, [list, tuple]) and 
               Utilities.are_list_elements_lengths_homogeneous(df_and_mp_merge_pairs, 2))
        left_on  = [x[0] for x in df_and_mp_merge_pairs]
        right_on = [x[1] for x in df_and_mp_merge_pairs]
        assert(len(set(left_on).difference(set(df.columns)))==0)
        assert(len(set(right_on).difference(set(df_mp.columns)))==0)
        #-------------------------  
        # Merge df and df_mp
        #----------
        df_cols_og = df.columns
        df_shape_og = df.shape
        # Note: Below, any overlapping columns from df will have '_x' suffix
        #       and those from df_mp will have '_y' suffix
        df = pd.merge(
            df, 
            df_mp, 
            left_on  = left_on, 
            right_on = right_on, 
            how      = 'left', 
            suffixes = ['_x', '_y']
        )    
        #-------------------------
        # Impose the time restrictions
        df = df[(df[df_mp_install_time_col] <= df[df_time_col_0]) & 
                (df[df_mp_removal_time_col].fillna(pd.Timestamp.max) > df[df_time_col_1])]
                
        if drop_inst_rmvl_cols_after_merge:
            df = df.drop(columns=[df_mp_install_time_col, df_mp_removal_time_col])
        #-------------------------
        # Handle any overlap columns
        overlap_cols_left = [x for x in df.columns if 
                             x.endswith('_x') and x[:-2] in df_cols_og]
        overlap_cols_right = [x for x in df.columns if 
                              x.endswith('_y') and x[:-2] in df_mp.columns]
        assert(len(overlap_cols_left)==len(overlap_cols_right))
        if len(overlap_cols_left)>0 and keep_overlap!='both':
            if keep_overlap=='left':
                df = df.drop(columns=overlap_cols_right)
                df = df.rename(columns={x:x[:-2] for x in overlap_cols_left})
            elif keep_overlap=='right':
                df = df.drop(columns=overlap_cols_left)
                df = df.rename(columns={x:x[:-2] for x in overlap_cols_right})
            else:
                assert(0)
        #-------------------------
        if assert_max_one_to_one:
            if df.shape[0]>df_shape_og[0]:
                print(f'df.shape[0] = {df.shape[0]}')
                print(f'df_shape_og[0] = {df_shape_og[0]}')
            assert(df.shape[0]<=df_shape_og[0])
        #-------------------------
        return df
        
        
    @staticmethod
    def merge_df_with_active_mp(
        df, 
        df_time_col_0,
        df_time_col_1                   = None,
        df_mp_curr                      = None, 
        df_mp_hist                      = None, 
        df_and_mp_merge_pairs           = [
            ['serialnumber', 'mfr_devc_ser_nbr'], 
            ['aep_premise_nb', 'prem_nb']
        ], 
        keep_overlap                    = 'right', 
        drop_inst_rmvl_cols_after_merge = True, 
        addtnl_mp_df_curr_cols          = None, 
        addtnl_mp_df_hist_cols          = None,
        assume_one_xfmr_per_PN          = True, 
        assert_all_PNs_found            = True, 
        drop_approx_duplicates          = True, 
        drop_approx_duplicates_args     = None, 
        df_prem_nb_col                  = 'aep_premise_nb', 
        df_mp_serial_number_col         = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col               = 'prem_nb', 
        df_mp_install_time_col          = 'inst_ts', 
        df_mp_removal_time_col          = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col          = 'trsf_pole_nb', 
        assert_max_one_to_one           = False
    ):
        r"""
        Get the active SNs for the PNs in df at the time given in the df_time_col_0 column, or between the times given
          in df_time_col_0 and df_time_col_1 if df_time_col_1 is not None.

        This was originally built so that the correct trsf_pole_nbs could be added to AMIEndEvents for the case of outages.  
          Doing this at collection time with SQL is typically impossible, as one must find the meter which was active at the
          time of the event.

        NOTE: Both df_mp_curr and df_mp_hist are needed because meter_premise_hist does not seem to be completely up-to-date.
              Therefore, one should actually use the current meter_premise together with the historical

        df:
          DF with which to join the active MP data

        df_time_col_0/_1:
          The column in df containing time information to use to determine which meter(s) were active.
          This column must contain datetime data!

        df_mp_curr:
          Current meter premise DF, taken from default.meter_premise
          WILL BE BUILT IF NOT SUPPLIED

        df_mp_hist:
          Historical meter premise DF, taken from default.meter_premise_hist
          WILL BE BUILT IF NOT SUPPLIED

        df_and_mp_merge_pairs:
          This defines how df and MeterPremise DF, df_mp, will be merged.
          This should be a list of pairs, where the first member of the pair is a column from df to use in the merge
            and the second member of the pair is the associated column from df_mp to use.

        keep_overlap:
          Must be equal to 'right', 'left' or 'both'.
          When there are overlapping columns in df and df_mp, this determined which should be kept in the finally returned DF
          'right' (default): Keep overlap from df_mp, get rid of those from df
          'left': Keep overlap from df, get rid of those from df_mp
          'both': Keep overlap from both.  Those from df will have suffix '_x', and those from df_mp will have suffix '_y'

        assume_one_xfmr_per_PN:
          The historical data are lacking the transformer number information.  If assume_one_xfmr_per_PN is set to True,
            any missing transformer numbers for historical entries will be filled with the transformer number inferred from
            the current meter premise data.
        """
        #-------------------------
        keep_overlap = keep_overlap.lower()
        #-------------------------
        if df_time_col_1 is None:
            df_time_col_1=df_time_col_0
        # The data in df[df_time_col_0/_1] must be datetime
        assert(is_datetime64_dtype(df[df_time_col_0]))
        assert(is_datetime64_dtype(df[df_time_col_1]))
        #-------------------------
        # Build/get df_mp
        df_mp = MeterPremise.get_historic_SNs_for_PNs_in_df(
            df                          = df, 
            df_mp_curr                  = df_mp_curr, 
            df_mp_hist                  = df_mp_hist, 
            addtnl_mp_df_curr_cols      = addtnl_mp_df_curr_cols, 
            addtnl_mp_df_hist_cols      = addtnl_mp_df_hist_cols, 
            assume_one_xfmr_per_PN      = assume_one_xfmr_per_PN, 
            output_index                = None,
            output_groupby              = None, 
            assert_all_PNs_found        = assert_all_PNs_found, 
            drop_approx_duplicates      = drop_approx_duplicates, 
            drop_approx_duplicates_args = drop_approx_duplicates_args, 
            df_prem_nb_col              = df_prem_nb_col, 
            df_mp_serial_number_col     = df_mp_serial_number_col, 
            df_mp_prem_nb_col           = df_mp_prem_nb_col, 
            df_mp_install_time_col      = df_mp_install_time_col, 
            df_mp_removal_time_col      = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col      = df_mp_trsf_pole_nb_col
        )
        #-------------------------
        # Compile left_on and right_on from df_and_mp_merge_pairs
        #----------
        assert(Utilities.is_object_one_of_types(df_and_mp_merge_pairs, [list, tuple]) and 
               Utilities.are_list_elements_lengths_homogeneous(df_and_mp_merge_pairs, 2))
        left_on  = [x[0] for x in df_and_mp_merge_pairs]
        right_on = [x[1] for x in df_and_mp_merge_pairs]
        assert(len(set(left_on).difference(set(df.columns)))==0)
        assert(len(set(right_on).difference(set(df_mp.columns)))==0)
        #-------------------------  
        # Merge df and df_mp
        #----------
        df_cols_og  = df.columns
        df_shape_og = df.shape
        # Note: Below, any overlapping columns from df will have '_x' suffix
        #       and those from df_mp will have '_y' suffix
        df = pd.merge(
            df, 
            df_mp, 
            left_on  = left_on, 
            right_on = right_on, 
            how      = 'left', 
            suffixes = ['_x', '_y']
        )    
        #-------------------------
        # Impose the time restrictions
        df = df[(df[df_mp_install_time_col]                         <= df[df_time_col_0]) & 
                (df[df_mp_removal_time_col].fillna(pd.Timestamp.max) > df[df_time_col_1])]
                
        if drop_inst_rmvl_cols_after_merge:
            df = df.drop(columns=[df_mp_install_time_col, df_mp_removal_time_col])
        #-------------------------
        # Handle any overlap columns
        overlap_cols_left = [x for x in df.columns if 
                             x.endswith('_x') and x[:-2] in df_cols_og]
        overlap_cols_right = [x for x in df.columns if 
                              x.endswith('_y') and x[:-2] in df_mp.columns]
        assert(len(overlap_cols_left)==len(overlap_cols_right))
        if len(overlap_cols_left)>0 and keep_overlap!='both':
            if keep_overlap=='left':
                df = df.drop(columns=overlap_cols_right)
                df = df.rename(columns={x:x[:-2] for x in overlap_cols_left})
            elif keep_overlap=='right':
                df = df.drop(columns=overlap_cols_left)
                df = df.rename(columns={x:x[:-2] for x in overlap_cols_right})
            else:
                assert(0)
        #-------------------------
        if assert_max_one_to_one:
            if df.shape[0]>df_shape_og[0]:
                print(f'df.shape[0] = {df.shape[0]}')
                print(f'df_shape_og[0] = {df_shape_og[0]}')
            assert(df.shape[0]<=df_shape_og[0])
        #-------------------------
        return df
        

    @staticmethod
    def build_sql_distinct_trsf_pole_nbs_OLD(
        n_trsf_pole_nbs=None, 
        **kwargs
    ):
        r"""
        n_trsf_pole_nbs:
            If set, this will randomly return n_trsf_pole_nbs instead of the full set
        
        kwargs can house any build_sql_meter_premise arguments (e.g., premise_nbs, states, etc.)

        NOTE: default.meter_premise_hist does not have trsf_pole_nb information.
              Therefore, trsf_pole_nbs can only be selected from those currently active.
        """
        #-------------------------
        sql = MeterPremise.build_sql_meter_premise(
            cols_of_interest=['trsf_pole_nb'], 
            **kwargs
        )
        sql.sql_select.select_distinct=True
        if n_trsf_pole_nbs is not None:
            # Obnoxious that I have to overcomplicate things and use a CTE because the following won't work
            #     SELECT DISTINCT
            #         MP.trsf_pole_nb
            #     FROM default.meter_premise MP
            #     ORDER BY RAND()
            #     LIMIT 10
            #  (gives error message, "... EXPRESSION_NOT_IN_DISTINCT ...")
            # So one cannot simply call the following two lines:
            #     sql.set_sql_orderby(SQLOrderBy(field_descs=['RAND()']))
            #     sql.set_limit(n_trsf_pole_nbs)
            #-------------------------
            alias = Utilities.generate_random_string(str_len=3, letters='letters_only')
            sql.alias = alias
            sql_stmnt = "WITH " + sql.get_sql_statement(include_alias=True)
            sql_2 = "\nSELECT *\nFROM {}\nORDER BY RAND()\nLIMIT {}".format(alias, n_trsf_pole_nbs)
            sql_stmnt = sql_stmnt+sql_2
        else:
            sql_stmnt = sql.get_sql_statement()
        #-------------------------
        return sql_stmnt
        
        
    @staticmethod
    def build_sql_distinct_trsf_pole_nbs(
        n_trsf_pole_nbs = None, 
        **kwargs
    ):
        r"""
        n_trsf_pole_nbs:
            If set, this will randomly return n_trsf_pole_nbs instead of the full set
        
        kwargs can house any build_sql_meter_premise arguments (e.g., premise_nbs, states, etc.)
    
        NOTE: default.meter_premise_hist does not have trsf_pole_nb information.
              Therefore, trsf_pole_nbs can only be selected from those currently active.
        """
        #--------------------------------------------------
        # At the time of updating this code (20240405), including opcos in build_sql_meter_premise will cause the returned object to 
        #   collapse down from a SQLQuery object to a string (through the use of MeterPremise.add_opco_nm_to_mp_sql)
        # At this point in the code, we need sql to remain a SQLQuery object, so we pop off opcos and include it manually later
        assert(not('opcos' in kwargs and 'opco' in kwargs))
        opcos = kwargs.pop('opcos', kwargs.pop('opco', None))
        
        #--------------------------------------------------
        sql = MeterPremise.build_sql_meter_premise(
            cols_of_interest = ['trsf_pole_nb'], 
            **kwargs
        )
        assert(isinstance(sql, SQLQuery))
        #-----
        sql.sql_select.select_distinct=True
        #--------------------------------------------------
        rndm_alias = None
        if n_trsf_pole_nbs is not None:
            rndm_alias = Utilities.generate_random_string(str_len=3, letters='letters_only')
        #-------------------------
        # Note: Below, if opcos is None then MeterPremise.add_opco_nm_to_mp_sql will simply
        #         return sql.get_sql_statement() (since include_comp_in_select==False)
        sql_stmnt = MeterPremise.add_opco_nm_to_mp_sql(
            mp_sql                 = sql, 
            opcos                  = opcos, 
            comp_cols              = ['opco_nm', 'opco_nb'], 
            comp_alias             = 'COMP', 
            join_type              = 'LEFT', 
            include_comp_in_select = False, 
            return_alias           = rndm_alias
        )
        #--------------------------------------------------
        if n_trsf_pole_nbs is not None:
            # Obnoxious that I have to overcomplicate things and use a CTE because the following won't work
            #     SELECT DISTINCT
            #         MP.trsf_pole_nb
            #     FROM default.meter_premise MP
            #     ORDER BY RAND()
            #     LIMIT 10
            #  (gives error message, "... EXPRESSION_NOT_IN_DISTINCT ...")
            # So one cannot simply call the following two lines:
            #     sql.set_sql_orderby(SQLOrderBy(field_descs=['RAND()']))
            #     sql.set_limit(n_trsf_pole_nbs)
            #-------------------------
            sql_stmnt = "WITH " + sql_stmnt
            #-----
            sql_2 = "\nSELECT *\nFROM {}\nORDER BY RAND()\nLIMIT {}".format(rndm_alias, n_trsf_pole_nbs)
            sql_stmnt = sql_stmnt+sql_2
        #--------------------------------------------------
        return sql_stmnt
        
    @staticmethod
    def get_distinct_trsf_pole_nbs(
        n_trsf_pole_nbs = None, 
        conn_aws        = None, 
        return_sql      = False, 
        **kwargs
    ):
        r"""
        kwargs can house any build_sql_meter_premise arguments (e.g., premise_nbs, states, etc.)
        
        NOTE: max_n_prem_nbs/batch_size and field_to_split are not included by default here, as in the MeterPremise constructor.
              So, e.g., if feeding in a long list of premise_nbs, one would want field_to_split='premise_nbs' and batch_size=10000 (or whatever)

        NOTE: default.meter_premise_hist does not have trsf_pole_nb information.
              Therefore, trsf_pole_nbs can only be selected from those currently active.
        """
        #-------------------------
        if conn_aws is None:
            conn_aws = Utilities.get_athena_prod_aws_connection()
        #-------------------------
        sql = MeterPremise.build_sql_distinct_trsf_pole_nbs(
            n_trsf_pole_nbs = n_trsf_pole_nbs, 
            **kwargs
        )
        assert(isinstance(sql, str))
        #-------------------------
        # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
        # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql_query(sql, conn_aws)
        #-------------------------
        if return_sql:
            return df, sql
        else:
            return df
        
    @staticmethod
    def get_distinct_trsf_pole_nbs_for_PNs(
        PNs, 
        batch_size = 10000, 
        verbose    = True, 
        n_update   = 10, 
        conn_aws   = None, 
        return_sql = False,
        **kwargs
    ):
        r"""
        NOTE: default.meter_premise_hist does not have trsf_pole_nb information.
              Therefore, trsf_pole_nbs can only be selected from those currently active.
        """
        #-------------------------
        if conn_aws is None:
            conn_aws = Utilities.get_athena_prod_aws_connection()
        #-------------------------
        batch_idxs = Utilities.get_batch_idx_pairs(
            n_total              = len(PNs), 
            batch_size           = batch_size, 
            absorb_last_pair_pct = None
        )
        n_batches = len(batch_idxs)
        return_df = pd.DataFrame()
        #-----
        if verbose:
            print(f'n_coll     = {len(PNs)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches  = {n_batches}')
        #-----
        sql_stmnts = []
        for i, batch_i in enumerate(batch_idxs):
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-----
            df_i, sql_i = MeterPremise.get_distinct_trsf_pole_nbs(
                n_trsf_pole_nbs = None, 
                conn_aws        = conn_aws, 
                return_sql      = True, 
                premise_nbs     = PNs[i_beg:i_end], 
                **kwargs
            )
            sql_stmnts.append(sql_i)
            #-----
            if return_df.shape[0]>0:
                assert(all(df_i.columns==return_df.columns))
            return_df = pd.concat([return_df, df_i], axis=0, ignore_index=False)
        #-------------------------
        # Due to the batch procedure, there can be repeat trsf_pole_nbs in return df (e.g., if PN_1_1 and PN_1_2 are
        #    in separate batches, trsf_pole_nb_1 will be in return_df twice)
        # ==> Remove duplicates
        return_df = return_df.drop_duplicates()
        return_df = return_df.reset_index(drop=True)
        #-------------------------
        if return_sql:
            return return_df, sql_stmnts
        else:
            return return_df
        
        
    @staticmethod
    def extract_zipcode(
        input_str, 
        return_srs = False
    ):
        r"""
        Designed to work with srvc_addr_4_nm field in meter premise
        ==> expecting, e.g., 
                input_str = 'LIMA, OH, 45806-1114'
                input_str = 'Columbus, OH, 43214'
            but, should work with any input string for which there is a single block of
            5 continuous numbers possibly followed by a dash and 4 additional continuous numbers
        Always returns a tuple of length 2 consisting of:
            (zip, zip+4)
        e.g., ('45086', '1114')
              ('43214', '')
        As hinted above, if zip+4 not found, the second element of the tuple will be an empty string
        """
        #-------------------------
        pattern = r'.*([0-9]{5}(?:-[0-9]{4})?).*$'
        #-------------------------
        found = re.findall(pattern, input_str)
        assert(len(found)==1)
        found = found[0]
        #-------------------------
        # Now, check to see whether zip+4 found
        # If found, two elements returned from split operation
        # If not found, one element returned
        zipcode = found.split('-')
        if len(zipcode)==1:
            zipcode = (zipcode[0], '')
        else:
            assert(len(zipcode)==2)
            zipcode = tuple(zipcode)
        #-------------------------
        if return_srs:
            return pd.Series(data=zipcode, index=['zip', 'zip+4'])
        else:
            return zipcode
            
    @staticmethod
    def extract_city_state_zipcode(
        input_str, 
        return_srs = False
    ):
        r"""
        Designed to work with srvc_addr_4_nm field in meter premise
        ==> expecting, e.g., 
                input_str = 'LIMA, OH, 45806-1114'
                input_str = 'Columbus, OH, 43214'
        Always returns a tuple of length 2 consisting of:
            (city, state, zip, zip+4)
        e.g., ('MOUNT ORAB', 'OH', 45154', '9615')
              ('COLUMBUS', 'OH', 43214', '')
        As hinted above, if zip+4 not found, the second element of the tuple will be an empty string
        """
        #-------------------------
        pattern = r'(.*)\s*,\s*(.*)\s*,\s*([0-9]{5}(?:-[0-9]{4})?).*$'
        #-------------------------
        found = re.findall(pattern, input_str)
        assert(len(found)==1)
        assert(len(found[0])==3)
        found = found[0]
        #-------------------------
        city, state, zipcode = found
        #-------------------------
        # Now, check to see whether zip+4 found
        # If found, two elements returned from split operation
        # If not found, one element returned
        zipcode = zipcode.split('-')
        if len(zipcode)==1:
            zipcode = [zipcode[0], '']
        else:
            assert(len(zipcode)==2)
            zipcode = list(zipcode)
        #-------------------------
        return_vals = tuple([city, state]+zipcode)
        #-------------------------
        if return_srs:
            return pd.Series(data=return_vals, index=['city', 'state', 'zip', 'zip+4'])
        else:
            return return_vals
            
    @staticmethod
    def get_trsf_location_info_from_mp_df(
        mp_df, 
        trsf_pole_nb_col = 'trsf_pole_nb', 
        srvc_addr_4_col  = 'srvc_addr_4_nm', 
        county_col       = 'county_nm', 
        SN_col           = 'mfr_devc_ser_nbr', 
        PN_col           = 'prem_nb'
    ):
        r"""
        From mp_df, find all transformer pole numbers in the given location(s) defined by city, state, zipcode, zipp4, county
        NOTE: mp_df MUST contain trsf_pole_nb_col and srvc_addr_4_col
        -----
        This function was developed specifically to extract transformers from a particular zip code.
        It has been expanded to include city, state, and county.
        If more options are desired, you may want to re-write a more general solution using e.g. DFSlicer
        -----
        Input values for city, state, zipcode, zipp4, and county:
            Each may be a single value (string in all cases should work, I suppose an int would work for zipcode/zipp4), a list
              of such values, or a DataFrameSubsetSingleSlicer object.
            single value:
                slicer added with '==' comparison operator
            list of values:
                slicer added with 'isin' comparison operator
            DataFrameSubsetSingleSlicer:
                input slicer added
        """
        #--------------------------------------------------
        # Make sure the necessary columns are contained
        assert(set([trsf_pole_nb_col, srvc_addr_4_col]).difference(set(mp_df.columns.tolist()))==set())
        #-----
        grp_by_cols = [trsf_pole_nb_col, srvc_addr_4_col]
        #-----
        if county_col is not None:
            assert(county_col in mp_df.columns.tolist())
            grp_by_cols.append(county_col)
        #-------------------------
        # Since transformer location is determined by the locations of its meters, a transformer can span, e.g., multiple zip codes
        # It will be beneficial to keep track of the number of meters in each grouping if, e.g., one wants to assign a single zip code
        #   to a transformer as determined by the zip containing the most meters.
        #-----
        # This number represents the number of premises/serial numbers having the particular value of srvc_addr_4_nm for the trsf_pole_nb
        #   So, if a trsf_pole_nb has meters split across two different srvc_addr_4_nms (e.g., different zips), mp_df_fnl will have two entries
        #     for trsf_pole_nb, one for srvc_addr_4_nm_1 (with associated meter counts) and one for srvc_addr_4_nm_2 (with associated meter counts)
        #-----
        # The line of code:
        #     mp_df_fnl = mp_df.groupby(grp_by_cols, as_index=False).size()
        # confused me for a second earlier, so I'm going to add the information below to help speed up the recovery from my confusion next time....
        # If one wanted the number of premises and meters for each trsf_pole_nb/srvc_addr_4_name, one would do the following:
        #   mp_df.groupby(['trsf_pole_nb', 'srvc_addr_4_nm'], as_index=False)[['prem_nb', 'mfr_devc_ser_nbr']].nunique(dropna=False)
        #-----
        # The original method was simply:
        #   mp_df_fnl = mp_df.groupby(grp_by_cols, as_index=False).size()
        # In most cases, this is fine and can be interpreted as the number of meters having the particular value of srvc_addr_4_nm for the trsf_pole_nb.
        # HOWEVER, if there are, e.g., duplicat entries, then this number will be incorrect!
        # Therefore, using nunique is preferred!
        #-------------------------
        #--------------------------------------------------
        mp_df_fnl = mp_df.groupby(grp_by_cols, as_index=False).size().rename(columns={'size':'nIDK_xfmri_sa4i'})
        #-------------------------
        # If available, include the SN/PN counts
        xN_cols_in_df = [x for x in [SN_col, PN_col] if x in mp_df.columns.tolist()]
        if len(xN_cols_in_df)>0:
            rename_xN_cols = {SN_col:'nSNs_xfmri_sa4i', PN_col:'nPNs_xfmri_sa4i'}
            #-----
            mp_df_fnl_xN_info = mp_df.groupby(grp_by_cols, as_index=False)[xN_cols_in_df].nunique(dropna=False).rename(columns=rename_xN_cols)
            #-----
            assert(mp_df_fnl.set_index(grp_by_cols).index.equals(mp_df_fnl_xN_info.set_index(grp_by_cols).index))
            mp_df_fnl = pd.concat([mp_df_fnl, mp_df_fnl_xN_info.drop(columns=grp_by_cols)], axis=1)
        #-------------------------
        trsf_pole_df = mp_df_fnl.apply(
            lambda x: MeterPremise.extract_city_state_zipcode(input_str=x[srvc_addr_4_col], return_srs=True), 
            axis=1
        )
        #-------------------------
        # This will ensure county_col (and any other future additions) are included in output
        trsf_pole_df = pd.concat([mp_df_fnl, trsf_pole_df], axis=1)
        #-------------------------
        trsf_pole_df = trsf_pole_df.drop(columns=[srvc_addr_4_col]).drop_duplicates()
        trsf_pole_df = trsf_pole_df.sort_values(by=[trsf_pole_nb_col], ignore_index=True)
        return trsf_pole_df
        
        
    @staticmethod
    def get_trsf_pole_df_in_location_helper(
        trsf_pole_df_i, 
        slicer, 
        strategy
    ):
        r"""
        """
        #-------------------------
        assert(strategy in ['any', 'all'])
        #-------------------------
        if(
            strategy=='all' and
            slicer.perform_slicing(trsf_pole_df_i).shape==trsf_pole_df_i.shape
        ):
            return True
        elif(
            strategy=='any' and
            slicer.perform_slicing(trsf_pole_df_i).shape[0]>0
        ):
            return True
        else:
            return False
    
    
    @staticmethod
    def get_trsf_pole_df_in_location(
        trsf_pole_df_full, 
        city                = None, 
        state               = None, 
        zipcode             = None, 
        zipp4               = None,
        county              = None, 
        join_single_slicers = 'and', 
        strategy            = 'simple', 
        trsf_pole_nb_col    = 'trsf_pole_nb', 
        srvc_addr_4_col     = 'srvc_addr_4_nm', 
        county_col          = 'county_nm'
    ):
        r"""
        From mp_df, find all transformer pole numbers in the given location(s) defined by city, state, zipcode, zipp4, county
        NOTE: mp_df MUST contain trsf_pole_nb_col and srvc_addr_4_col
        -----
        This function was developed specifically to extract transformers from a particular zip code.
        It has been expanded to include city, state, and county.
        If more options are desired, you may want to re-write a more general solution using e.g. DFSlicer
        -----
        Input values for city, state, zipcode, zipp4, and county:
            Each may be a single value (string in all cases should work, I suppose an int would work for zipcode/zipp4), a list
              of such values, or a DataFrameSubsetSingleSlicer object.
            single value:
                slicer added with '==' comparison operator
            list of values:
                slicer added with 'isin' comparison operator
            DataFrameSubsetSingleSlicer:
                input slicer added
        ----------------------------------------------------------------------------------------------------
        strategy:
            Here, we determine the location of a transformer through the locations of the connected meters.
            This means that a transformer can span multiple boundaries, where a boundary is a zip code, city, etc.
            -------------------------
            In some cases, I believe a transformer probably does span a couple zip codes.
            -----
            In other cases, I think this is largely a result of user input error in our data.
              e.g., trsf_pole_nb = 1818591704125 is connected to meters in two different cities (Columbus and Galloway)
                and in three different zip codes (43119, 42301, and 43228).
                After looking at a few of the meter addresses on Google maps, I believe all meters shuold be listed in 
                  Galloway with zip 43119
            -------------------------
            At the end of the day, it doesn't matter if the result is due to reality or mistaken input, at this point
              the data are what they are.
            What does matter is how the situation will be treated.
            -------------------------
            This is where the strict parameter comes into play
            This allows the user to return a trsf_pole_nb if any of its meter fulfill the search criteria, or
              only if all of its meters fulfill the criteria
            ---------------------------------------------------------------------------
            For all, a DFSlicer object is first built using city, state, zipcode, zipp4, and county.
            The DFSlicer object is constructed with join_single_slicers, meaning the individual slicers are joined together
              with join_single_slicers (either 'and' or 'or')
            After the DFSlicer object is built, it is applied to trsf_pole_df_full before any strategy logic is involved
            -----
            Definition: complete transformer (in returned trsf_pole_df)
                If a transformer, trsf_pole_nb_i, is included in the returned trsf_pole_df, then all entries from trsf_pole_df_full with
                  trsf_pole_nb==trsf_pole_nb_i are included.
            Strategy method 'simple'         DOES NOT guarantee completeness of transformers in the returned pd.DataFrame
            Strategy methods 'all' and 'any' DO guarantee completeness of transformers in the returned pd.DataFrame
            -------------------------
            'simple':
                Simply return the result of applying slicer to trsf_pole_df_full (i.e., return trsf_pole_df)
                Note, this case DOES NOT GUARANTEE that all returned transformer data are complete
    
            'all':
                In order for a transformer to be included in the returned pd.DataFrame, all of the entries in trsf_pole_df_full for that
                  transformer (i.e., all meters connected to the transformer) must pass the DFSlicer described above.
    
            'any':
                In order for a transformer to be included in the returned pd.DataFrame, at least one of the entries in trsf_pole_df_full for that
                  transformer (i.e., all meters connected to the transformer) must pass the DFSlicer described above.
    
        -----------------------------------------------------------------------------------------------------------------------------
        To be as clear as possible regarding the functionality, consider the following example: 
            city    = 'MOUNT ORAB'
            state   = None
            zipcode = 45154
            
            with the following cases:
                A
                    join_single_slicers = 'and', 
                    strategy            = 'simple',  
                B:
                    join_single_slicers = 'and', 
                    strategy            = 'all',            
                C:
                    join_single_slicers = 'and', 
                    strategy            = 'any',   
                D:
                    join_single_slicers = 'or', 
                    strategy            = 'simple', 
                E:
                    join_single_slicers = 'or', 
                    strategy            = 'all',         
                F:
                    join_single_slicers = 'or', 
                    strategy            = 'any',
                    
            The four methods give the same results EXCEPT for trsf_pole_nbs 1617155393730 and 1618262395121.
            The relevant entries are shown below, together with whether or not they are contained in the results
              for cases A, B, C, and D described above
        
                        trsf_pole_nb      city        state    zip    A     B    C    D     E    F     
                    0    1617155393730    BUFORD        OH    45154              1    1     1    1
                    1    1617155393730    MOUNT ORAB    OH    45154   1          1    1     1    1
                    2    1618262395121    BUFORD        OH    45171              1               1
                    3    1618262395121    MOUNT ORAB    OH    45154   1          1    1          1
                    
            -------------------------        
            A:
                join_single_slicers = 'and', 
                strategy            = 'simple',
                -----
                df.shape[0]            = 20
                # unique trsf_pole_nbs = 20
        
            Returned trsf_pole_nbs: recovers both trsf_pole_nbs 1617155393730 and 1618262395121
            Returned trsf_pole_df:  recovers only (2) entries with city=MOUNT ORAB and zip=45154
        
            Comments: 
                Each row's logic: (city='MOUNT ORAB' AND zipcode=45154)
                Each transformer has one entry satisfying the above logic.
                Since strategy=='simple', return only the passing entries
            
            -------------------------        
            B:
                join_single_slicers = 'and', 
                strategy            = 'all',
                -----
                df.shape[0]            = 18
                # unique trsf_pole_nbs = 18
        
            Returned trsf_pole_nbs: recovers neither
            Returned trsf_pole_df:  recovers neither
        
            Comments: 
                Each row's logic: (city='MOUNT ORAB' AND zipcode=45154)
                Each transformer has one entry satisfying the above logic, BUT NOT ALL!
                Since strategy=='all', the transformers are rejected in entirety
        
            -------------------------
            C:
                join_single_slicers = 'and', 
                strategy            = 'any',
                -----
                df.shape[0]            = 22
                # unique trsf_pole_nbs = 20
                
            Returned trsf_pole_nbs: recovers both trsf_pole_nbs 1617155393730 and 1618262395121
            Returned trsf_pole_df:  recovers all entries
        
            Comments: 
                Each row's logic: (city='MOUNT ORAB' AND zipcode=45154)
                Each transformer has one entry satisfying the above logic
                Since strategy=='any', both transformers are accepted in entirety
        
            -------------------------
            D:
                join_single_slicers = 'or', 
                strategy            = 'simple', 
                -----
                df.shape[0]            = 21
                # unique trsf_pole_nbs = 20
                
            Returned trsf_pole_nbs: recovers both trsf_pole_nbs 1617155393730 and 1618262395121
            Returned trsf_pole_df:  recovers both entries for trsf_pole_nb 1617155393730 but only recover one entry for trsf_pole_nb 1618262395121
        
            Comments: 
                Each row's logic: (city='MOUNT ORAB' OR zipcode=45154)
                Recovers both entries for trsf_pole_nb 1617155393730, as both entries have city=MOUNT ORAB OR zip=45154
                Only recovers one entry for trsf_pole_nb 1618262395121, as only one entry satisfies city=MOUNT ORAB OR zip=45154
                Since strategy=='simple', return only the passing entries
            
            -------------------------
            E:
                join_single_slicers = 'or', 
                strategy            = 'all', 
                -----
                df.shape[0]            = 20
                # unique trsf_pole_nbs = 19
                
            Returned trsf_pole_nbs: recovers only trsf_pole_nb 1617155393730
            Returned trsf_pole_df:  recovers both entries for trsf_pole_nb 1617155393730
        
            Comments: 
                Each row's logic: (city='MOUNT ORAB' OR zipcode=45154)
                Both entries for trsf_pole_nb 1617155393730 pass the logic.
                Onle one entry for trsf_pole_nb 1618262395121 passes the logic.
                Since strategy=='all', trsf_pole_nb 1617155393730 is accepted while 1618262395121 is rejected
        
            -------------------------
            F:
                join_single_slicers = 'or', 
                strategy            = 'any',
                -----
                df.shape[0]            = 22
                # unique trsf_pole_nbs = 20
                
            Returned trsf_pole_nbs: recovers both trsf_pole_nbs 1617155393730 and 1618262395121
            Returned trsf_pole_df:  recovers all entries
        
            Comments: 
                Each row's logic: (city='MOUNT ORAB' OR zipcode=45154)
                Both entries for trsf_pole_nb 1617155393730 pass the logic.
                Onle one entry for trsf_pole_nb 1618262395121 passes the logic.
                Since strategy=='any', both transformers are accepted in entirety
    
        
        """
        #--------------------------------------------------
        cuts_dict = {
            'city'    : city, 
            'state'   : state, 
            'zip'     : zipcode, 
            'zip+4'   : zipp4
        }
        if county is not None:
            cuts_dict[county_col] = county
        #--------------------------------------------------
        assert(strategy in ['simple', 'any', 'all'])
        #--------------------------------------------------
        slicer = DFSlicer(
            single_slicers      = None, 
            name                = None, 
            apply_not           = False, 
            join_single_slicers = join_single_slicers
        )
        for col, cut_val in cuts_dict.items():
            if cut_val is None:
                continue
            #-----
            assert(Utilities.is_object_one_of_types(cut_val, [str, int, list, tuple, DFSingleSlicer]))
            if Utilities.is_object_one_of_types(cut_val, [str, int]):
                slicer.add_single_slicer(input_arg=dict(
                    column              = col, 
                    value               = cut_val, 
                    comparison_operator = '=='
                ))
            elif Utilities.is_object_one_of_types(cut_val, [list, tuple]):
                slicer.add_single_slicer(input_arg=dict(
                    column              = col, 
                    value               = list(cut_val), 
                    comparison_operator = 'isin'
                ))
            elif isinstance(cut_val, DFSingleSlicer):
                slicer.add_single_slicer(input_arg=cut_val)
            else:
                assert(0)
        #-------------------------
        # Grab 'simple' result
        trsf_pole_df = slicer.perform_slicing(trsf_pole_df_full).copy()
        if strategy=='simple':
            return trsf_pole_df
        #--------------------------------------------------
        # If we want to apply the 'any' or 'all' strategy, we first need to grab the candidate trsf_pole_nbs (in trsf_pole_df) from the full, original, 
        #   trsf_pole_df_full and then check that any(all) meter(s) pass for each trsf_pole_nb
        trsf_pole_df_tmp = trsf_pole_df_full[trsf_pole_df_full[trsf_pole_nb_col].isin(trsf_pole_df[trsf_pole_nb_col].unique())].copy()
        #-------------------------
        # NOTE: Functionality of groupby.apply has changed (sometime been pd versions 2.0.2 and 2.2.0)
        #   Error Message:  DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will 
        #                   be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
        # So, if the columns in grp_by_cols are to be included in the apply function, one must make the following adjustment, e.g.,
        #     WRONG/DEPRICATED: df.groupby(grp_by_cols).apply(...)
        #     CORRECT         : df.groupby(grp_by_cols)[df.columns].apply(...)
        passing_bool = trsf_pole_df_tmp.groupby([trsf_pole_nb_col])[trsf_pole_df_tmp.columns].apply(
            lambda x: MeterPremise.get_trsf_pole_df_in_location_helper(
                trsf_pole_df_i = x, 
                slicer         = slicer, 
                strategy       = strategy
            )
        )
        passing_trsf_pole_nbs = passing_bool[passing_bool==True].index.tolist()
        #-----
        trsf_pole_df = trsf_pole_df_tmp[trsf_pole_df_tmp[trsf_pole_nb_col].isin(passing_trsf_pole_nbs)].copy()
        trsf_pole_df = trsf_pole_df.drop_duplicates()
        return trsf_pole_df
    
    
    @staticmethod
    def get_trsf_pole_nbs_in_location_from_mp_df(
        mp_df, 
        city                = None, 
        state               = None, 
        zipcode             = None, 
        zipp4               = None,
        county              = None, 
        join_single_slicers = 'and', 
        strategy            = 'simple', 
        return_df           = False, 
        trsf_pole_nb_col    = 'trsf_pole_nb', 
        srvc_addr_4_col     = 'srvc_addr_4_nm', 
        county_col          = 'county_nm'
    ):
        r"""
        From mp_df, find all transformer pole numbers in the given location(s) defined by city, state, zipcode, zipp4, county
        NOTE: mp_df MUST contain trsf_pole_nb_col and srvc_addr_4_col
        SEE MeterPremise.get_trsf_pole_df_in_location for (much) more information!
        -----
        By default this returns a list of the trsf_pole_nbs, UNLESS return_df==True
        -----
        This function was developed specifically to extract transformers from a particular zip code.
        It has been expanded to include city, state, and county.
        If more options are desired, you may want to re-write a more general solution using e.g. DFSlicer
        -----
        Input values for city, state, zipcode, zipp4, and county:
            Each may be a single value (string in all cases should work, I suppose an int would work for zipcode/zipp4), a list
              of such values, or a DataFrameSubsetSingleSlicer object.
            single value:
                slicer added with '==' comparison operator
            list of values:
                slicer added with 'isin' comparison operator
            DataFrameSubsetSingleSlicer:
                input slicer added
        """
        #--------------------------------------------------
        cuts_dict = {
            'city'    : city, 
            'state'   : state, 
            'zip'     : zipcode, 
            'zip+4'   : zipp4
        }
        if county is not None:
            cuts_dict[county_col] = county
        #-------------------------
        if all([x is None for x in cuts_dict.values()]) and not return_df:
            return mp_df[trsf_pole_nb_col].unique().tolist()
        #--------------------------------------------------
        assert(strategy in ['simple', 'any', 'all'])
        #--------------------------------------------------
        trsf_pole_df_full = MeterPremise.get_trsf_location_info_from_mp_df(
            mp_df            = mp_df, 
            trsf_pole_nb_col = trsf_pole_nb_col, 
            srvc_addr_4_col  = srvc_addr_4_col, 
            county_col       = None if county is None else county_col
        )
        #--------------------------------------------------
        trsf_pole_df = MeterPremise.get_trsf_pole_df_in_location(
            trsf_pole_df_full   = trsf_pole_df_full, 
            city                = city, 
            state               = state, 
            zipcode             = zipcode, 
            zipp4               = zipp4,
            county              = county, 
            join_single_slicers = join_single_slicers, 
            strategy            = strategy, 
            trsf_pole_nb_col    = trsf_pole_nb_col, 
            srvc_addr_4_col     = srvc_addr_4_col, 
            county_col          = county_col
        )
        if return_df:
            return trsf_pole_df
        else:
            return trsf_pole_df[trsf_pole_nb_col].unique().tolist()
        

    @staticmethod
    def get_xfmr_meter_cnt_sql_stmnt(
        trsf_pole_nbs , 
        return_PN_cnt = False, 
        **kwargs
    ):
        r"""
        Returns a string containing the SQL statement to be used by MeterPremise.get_xfmr_meter_cnt
    
        kwargs:
            If user supplies the following keyword arguments, they will be ignore:
                [cols_of_interest, trsf_pole_nb(s), serial_number(s)\mfr_devc_ser_nbr(s), premise_nb(s)\aep_premise_nb(s), 'groupby_cols']
        """
        #--------------------------------------------------
        # opcos will be handled after, since including in call to MeterPremise.build_sql_meter_premise
        #   could collapse it down to a string instead of SQLQuery object
        opcos = kwargs.pop('opcos', None)
        #--------------------------------------------------
        kwargs['from_table_alias']  = kwargs.get('from_table_alias', None)
        kwargs['trsf_pole_nb_col']  = kwargs.get('trsf_pole_nb_col', 'trsf_pole_nb')
        kwargs['serial_number_col'] = kwargs.get('serial_number_col', 'mfr_devc_ser_nbr')
        kwargs['premise_nb_col']    = kwargs.get('premise_nb_col', 'prem_nb')
        #-------------------------
        keys_to_ignore = [
            'cols_of_interest', 
            'trsf_pole_nb',  'trsf_pole_nbs', 
            'serial_number', 'serial_numbers', 'mfr_devc_ser_nbr', 'mfr_devc_ser_nbrs', 
            'premise_nb',    'premise_nbs',    'aep_premise_nb',   'aep_premise_nbs', 
            'groupby_cols'
        ]
        sql_kwargs =  {k:v for k,v in kwargs.items() if k not in keys_to_ignore}
        #-------------------------
        sql_kwargs['trsf_pole_nbs'] = trsf_pole_nbs
        
        #--------------------------------------------------
        cols_of_interest = [
            kwargs['trsf_pole_nb_col'], 
            dict(
                field_desc         = (f"COUNT(DISTINCT {kwargs['serial_number_col']})" if kwargs['from_table_alias'] is None else
                                      f"COUNT(DISTINCT {kwargs['from_table_alias']}.{kwargs['serial_number_col']})"), 
                alias              = 'xfmr_SN_cnt', 
                table_alias_prefix = None
            )
        ]
        #-----
        if return_PN_cnt:
            cols_of_interest.append(
                dict(
                    field_desc         = (f"COUNT(DISTINCT {kwargs['premise_nb_col']})" if kwargs['from_table_alias'] is None else
                                          f"COUNT(DISTINCT {kwargs['from_table_alias']}.{kwargs['premise_nb_col']})"), 
                    alias              = 'xfmr_PN_cnt', 
                    table_alias_prefix = None
                )
            )
        
        #--------------------------------------------------
        mp_sql = MeterPremise.build_sql_meter_premise(
            cols_of_interest = cols_of_interest, 
            **sql_kwargs
        )
        
        #--------------------------------------------------
        sql_groupby = SQLGroupBy(
            field_descs               = [kwargs['trsf_pole_nb_col']], 
            global_table_alias_prefix = None, 
            idxs                      = None, 
            run_check                 = True
        )
        mp_sql.set_sql_groupby(sql_groupby)
        
        #--------------------------------------------------
        if opcos is not None:
            sql_stmnt = MeterPremise.add_opco_nm_to_mp_sql(
                mp_sql     = mp_sql, 
                opcos      = opcos, 
                comp_cols  = ['opco_nm'], 
                comp_alias = 'COMP', 
                join_type  = 'LEFT', 
            )
        else:
            sql_stmnt = mp_sql.get_sql_statement()
        #-------------------------
        return sql_stmnt
    
    
    @staticmethod
    def get_xfmr_meter_cnt(
        trsf_pole_nbs , 
        return_PN_cnt = False, 
        batch_size    = 1000, 
        verbose       = True, 
        n_update      = 10, 
        conn_aws      = None, 
        return_sql    = False, 
        **kwargs
    ):
        r"""
        Returns a pd.DataFrame containing trsf_pole_nb and the number of meters connected to it (xfmr_SN_cnt)
        If return_PN_cnt == True, xfmr_PN_cnt will also be included.
    
        kwargs:
            If user supplies the following keyword arguments, they will be ignore:
                [cols_of_interest, trsf_pole_nb(s), serial_number(s)\mfr_devc_ser_nbr(s), premise_nb(s)\aep_premise_nb(s), 'groupby_cols']
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
            print(f'n_coll     = {len(trsf_pole_nbs)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches  = {n_batches}')
        #--------------------------------------------------
        sql_stmnts = []
        for i, batch_i in enumerate(batch_idxs):
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-----
            sql_stmnt_i = MeterPremise.get_xfmr_meter_cnt_sql_stmnt(
                trsf_pole_nbs = trsf_pole_nbs[i_beg:i_end], 
                return_PN_cnt = return_PN_cnt, 
                **kwargs
            )
            assert(isinstance(sql_stmnt_i, str))
            #-------------------------
            # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
            # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df_i = pd.read_sql_query(sql_stmnt_i, conn_aws)
            #-------------------------
            sql_stmnts.append(sql_stmnt_i)
            #-----
            if return_df.shape[0]>0:
                assert(all(df_i.columns==return_df.columns))
            return_df = pd.concat([return_df, df_i], axis=0, ignore_index=False)
        #--------------------------------------------------
        if return_sql:
            return return_df, sql_stmnts
        else:
            return return_df