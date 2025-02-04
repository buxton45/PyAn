#!/usr/bin/env python

r"""
Holds AMI_SQL class.  See AMI_SQL.AMI_SQL for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys

import pandas as pd
import datetime
import copy
import textwrap
#--------------------------------------------------
from MeterPremise import MeterPremise
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from SQLElement import SQLElement
from SQLSelect import SQLSelectElement, SQLSelect
from SQLFrom import SQLFrom
from SQLWhere import SQLWhereElement, CombinedSQLWhereElements, SQLWhere
from SQLJoin import SQLJoin
from SQLGroupBy import SQLGroupBy
from SQLQuery import SQLQuery
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
#--------------------------------------------------

#****************************************************************************************************
class DfToSqlMap:
    def __init__(
        self    , 
        df_col  , 
        kwarg   , 
        sql_col , 
    ):
        r"""
        Super simple class for use in AMI_SQL.build_sql_ami_for_df_with_search_time_window

        Example:
            Suppose you have a pd.DataFrame object containing DOVS outage data, including the outage times and premises.
            We want to use this pd.DataFrame to search for meter events occuring around the time of the outages (i.e., we are
              ultimately querying data from the meter_events.end_device_event table)
            These premises are stored in the "PREMISE_NB" field of the pd.DataFrame.
                ==> df_col = "PREMISE_NB"
            The keyword argument uses to select premise numbers within my AMI_SQL framework is "premise_nbs" (actually, the software is 
              set up to accept any of ['premise_nbs', 'premise_nb', 'aep_premise_nbs', 'aep_premise_nb'], but let's assume "premise_nbs" is 
              the only option, for simplicity here)
                ==> kwarg = "premise_nbs"
            Finally, the data we will query is contained in meter_events.end_device_event table, which stores the premise number data in 
              the "aep_premise_nb" field.
                ==> sql_col = "aep_premise_nb"

            So, for this example, we have 
                df_col  = "PREMISE_NB", 
                kwarg   = "premise_nbs", 
                sql_col = "aep_premise_nb", 
        """
        #--------------------------------------------------
        assert(isinstance(df_col,  str))
        assert(isinstance(kwarg,   str))
        assert(isinstance(sql_col, str))
        #-------------------------
        self.__df_col  = df_col
        self.__kwarg   = kwarg
        self.__sql_col = sql_col

    @property
    def df_col(self):
        return self.__df_col

    @property
    def kwarg(self):
        return self.__kwarg

    @property
    def sql_col(self):
        return self.__sql_col
    
    # Methods below allow DfToSqlMap to be written JSON using CustomJSON module
    def get_dict(self):
        return {
            'df_col'  : self.__df_col, 
            'kwarg'   : self.__kwarg, 
            'sql_col' : self.__sql_col
        }
        
    def to_json_dict_key(self):
        desc_dict = self.get_dict()
        return_key = ''
        for k,v in desc_dict.items():
            return_key += f'{k}:{v}; '
        return return_key
            
    def to_json_value(self):
        return self.get_dict()

#****************************************************************************************************
class AMI_SQL:
    def __init__(self):
        self.sql_query = None

    @staticmethod
    def alias_found_in_cols_of_interest(alias, cols_of_interest):
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
        #-------------------------
        # First, build the collection of aliases from cols_of_interest
        aliases = []
        for col in cols_of_interest:
            assert(Utilities.is_object_one_of_types(col, [str, dict, SQLElement]))
            if isinstance(col, str):
                aliases.append(col)
            elif isinstance(col, dict):
                assert('alias' in col.keys())
                aliases.append(col['alias'])
            elif isinstance(col, SQLElement):
                aliases.append(col.alias)
            else:
                assert(0)
        #-------------------------
        return alias in aliases
        
    @staticmethod
    def combine_where_elements(
        sql_where, 
        wheres_to_combine, 
        field_descs_dict
    ):
        r"""
        Designed for internal use by AMI_SQL.add_ami_where_statements

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
                sql_where = AMI_SQL.combine_where_elements(
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
    def add_date_where_statement(
        sql_where        , 
        date_range       = None, 
        date_col         = 'aep_usage_dt', 
        from_table_alias = None, 
        idx              = None, 
        return_idx       = False
    ):
        r"""
        """
        #---------------------------------------------------------------------------
        if date_range is None:
            if return_idx:
                return sql_where, -1
            else:
                return sql_where
        #-------------------------
        assert(
            Utilities.is_object_one_of_types(date_range, [list, tuple]) and 
            len(date_range)==2
        )
        #-------------------------
        if idx is None:
            idx = len(sql_where.collection_dict)
        #-------------------------
        # If any elements in date_range are of type pd.Timestamp or datetime.datetime, convert to date
        # This is not explicitly necessary, but this will make it easier to remove any duplicates later
        date_range_fnl = [
            x if not Utilities.is_object_one_of_types(x, [datetime.datetime, pd.Timestamp]) else x.date() 
            for x in date_range
        ]
        assert(len(date_range_fnl)==2)
        #-------------------------
        sql_where.add_where_statement(
            field_desc          = date_col, 
            comparison_operator = 'BETWEEN', 
            value               = [f'{date_range_fnl[0]}', f'{date_range_fnl[1]}'], 
            needs_quotes        = True, 
            table_alias_prefix  = from_table_alias, 
            idx                 = idx
        )
        #-------------------------
        if return_idx:
            return sql_where, idx
        else:
            return sql_where
            
            
    @staticmethod
    def add_datetime_where_statement(
        sql_where        , 
        datetime_range   = None, 
        datetime_col     = 'starttimeperiod', 
        datetime_pattern = r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*", 
        date_col         = 'aep_usage_dt', 
        from_table_alias = None, 
        idx              = None, 
        return_idxs      = False
    ):
        r"""
        NOTE: The datetime operation in the code, which typically involves regex expressions, is a heavy lift.
              The query runs much smoother if date_range is used together with datetime_range (I have seen instances where
                running with only datetime_range returns an error from Athena).
              My guess is the date_range (the way I have it written) is executed first, so the set upon which the regex operations
                need to be run is largely reduced.
              THEREFORE, if datetime_range is set, set date_range as well.
              NOTE: This can only be done if the elements of datetime_range are timestamp objects, so I can extract the dates
        """
        #---------------------------------------------------------------------------
        if datetime_range is None:
            if return_idxs:
                return sql_where, -1, -1
            else:
                return sql_where
        #-------------------------
        assert(
            Utilities.is_object_one_of_types(datetime_range, [list, tuple]) and 
            len(datetime_range)==2
        )
        #-------------------------
        if idx is None:
            idx = len(sql_where.collection_dict)
        #---------------------------------------------------------------------------
        # As described above, first include the date where statement to make things run smoother
        # NOTE: This can only be done if the elements of datetime_range are timestamp objects, so I can extract the dates
        if all([isinstance(x, datetime.datetime) for x in datetime_range]):
            date_range = [x.date() for x in datetime_range]
            sql_where = AMI_SQL.add_date_where_statement(
                sql_where        = sql_where, 
                date_range       = date_range, 
                date_col         = date_col, 
                from_table_alias = from_table_alias, 
                idx              = idx
            )
            if idx is not None:
                idx += 1
        #---------------------------------------------------------------------------
        if from_table_alias:
            # If from_table_alias has already been added, don't add again!!!
            #   This is typical for, e.g., dt_off_ts_full/datetime_col in DOVSOutages_SQL, where, due to it's complicated
            #     structure, the from_table_alias is added before reaching this point
            if not datetime_col.startswith(f'{from_table_alias}.'):
                datetime_col = f'{from_table_alias}.{datetime_col}'
        #-----
        if datetime_pattern is None:
            dt_field_desc = f"CAST({datetime_col} AS TIMESTAMP)"
        elif datetime_pattern.lower()=='no_cast':
            dt_field_desc = datetime_col
        else:
            dt_field_desc = r"CAST(regexp_replace({}, ".format(datetime_col) + r"'{}', '$1 $2') AS TIMESTAMP)".format(datetime_pattern)
        sql_where.add_where_statement(
            field_desc          = dt_field_desc, 
            comparison_operator = 'BETWEEN', 
            value               = [f'{datetime_range[0]}', f'{datetime_range[1]}'], 
            needs_quotes        = True, 
            table_alias_prefix  = None, 
            is_timestamp        = True, 
            idx                 = idx
        )
        #---------------------------------------------------------------------------
        if return_idxs:
            date_idx     = idx-1
            datetime_idx = idx
            return sql_where, datetime_idx, date_idx
        else:
            return sql_where
            
    @staticmethod
    def add_date_where_statements(
        sql_where        , 
        date_ranges      = None, 
        date_col         = 'aep_usage_dt', 
        from_table_alias = None, 
        idx              = None, 
        return_idx       = False
    ):
        r"""
        """
        #---------------------------------------------------------------------------
        if date_ranges is None:
            if return_idx:
                return sql_where, -1
            else:
                return sql_where
        #-------------------------
        assert(Utilities.is_object_one_of_types(date_ranges, [list, tuple]))
        for date_range_i in date_ranges:
            assert(
                Utilities.is_object_one_of_types(date_range_i, [list, tuple]) and 
                len(date_range_i)==2
            )
        #-------------------------
        if idx is None:
            idx = len(sql_where.collection_dict)
        #---------------------------------------------------------------------------
        date_idxs = []
        for date_range_i in date_ranges:
            n_where = len(sql_where.collection_dict)
            #-----
            sql_where = AMI_SQL.add_date_where_statement(
                sql_where        = sql_where, 
                date_range       = date_range_i, 
                date_col         = date_col, 
                from_table_alias = from_table_alias, 
                idx              = idx
            )
            #-----
            # One element should have been added to sql_where at each iteration
            assert(len(sql_where.collection_dict) - n_where == 1)
            date_idxs.append(idx)
            idx += 1
        #-------------------------
        assert(len(date_idxs)>0)
        if len(date_idxs)>1:
            comb_date_idx = sql_where.combine_where_elements(
                idxs_to_combine    = date_idxs, 
                join_operator      = 'OR', 
                close_gaps_in_keys = True, 
                return_idx         = True
            )
        else:
            comb_date_idx = idx-1 # -1 because idx += 1 at the end of for loop above
        #-------------------------
        if return_idx:
            return sql_where, comb_date_idx
        else:
            return sql_where
            
    @staticmethod
    def add_datetime_where_statements(
        sql_where        , 
        datetime_ranges  = None, 
        datetime_col     = 'starttimeperiod', 
        datetime_pattern = r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*", 
        date_col         = 'aep_usage_dt', 
        from_table_alias = None, 
        idx              = None, 
        return_idxs      = False
    ):
        r"""
        """
        #---------------------------------------------------------------------------
        if datetime_ranges is None:
            if return_idxs:
                return sql_where, -1, -1
            else:
                return sql_where
        #-------------------------
        assert(Utilities.is_object_one_of_types(datetime_ranges, [list, tuple]))
        for datetime_range_i in datetime_ranges:
            assert(
                Utilities.is_object_one_of_types(datetime_range_i, [list, tuple]) and 
                len(datetime_range_i)==2
            )
        #-------------------------
        if idx is None:
            idx = len(sql_where.collection_dict)
        #---------------------------------------------------------------------------
        date_idxs     = []
        datetime_idxs = []
        #-----
        for datetime_range_i in datetime_ranges:
            n_where = len(sql_where.collection_dict)
            #-----
            sql_where = AMI_SQL.add_datetime_where_statement(
                sql_where        = sql_where, 
                datetime_range   = datetime_range_i, 
                datetime_col     = datetime_col, 
                datetime_pattern = datetime_pattern, 
                date_col         = date_col, 
                from_table_alias = from_table_alias, 
                idx              = idx    
            )
            # When possible (when .date() can be called on datetime_range_i contents), the function AMI_SQL.add_datetime_where_statement
            #  will add two where statements, one for the date portion of datetime_range_i and one for the full datetime (see documentation in 
            #  AMI_SQL.add_datetime_where_statement for explanation of why).
            # Thus, the size of sql_where.collection_dict right before the AMI_SQL.add_datetime_where_statement call should be 1 or 2 elements
            #   smaller than after the call.
            #   If difference = 1: ==> only datetime where statement added
            #   If difference = 2: ==> both date and datetime where statements added
            delta_n = len(sql_where.collection_dict) - n_where
            assert(delta_n<=2 and delta_n>0)
            if delta_n==1:
                datetime_idxs.append(idx)
            elif delta_n==2:
                date_idxs.append(idx)
                datetime_idxs.append(idx+1)
            else:
                assert(0)
            idx += delta_n
        #-------------------------
        if len(date_idxs)==0:
            comb_date_idx = -1
        elif len(date_idxs)==1:
            comb_date_idx = date_idxs[0]
        else:
            comb_date_idx = sql_where.combine_where_elements(
                idxs_to_combine    = date_idxs, 
                join_operator      = 'OR', 
                close_gaps_in_keys = False, 
                return_idx         = True
            )
        #-------------------------
        assert(len(datetime_idxs)>0)
        if len(datetime_idxs)==1:
            comb_datetime_idx = datetime_idxs[0]
        else:
            comb_datetime_idx = sql_where.combine_where_elements(
                idxs_to_combine    = datetime_idxs, 
                join_operator      = 'OR', 
                close_gaps_in_keys = True, 
                return_idx         = True
            )
        #-------------------------
        if comb_date_idx > comb_datetime_idx:
            sql_where.swap_idxs(
                idx_1 = comb_date_idx, 
                idx_2 = comb_datetime_idx
            )
            comb_date_idx, comb_datetime_idx = comb_datetime_idx, comb_date_idx
        #---------------------------------------------------------------------------
        if return_idxs:
            return sql_where, comb_datetime_idx, comb_date_idx
        else:
            return sql_where
            
    @staticmethod
    def add_date_and_datetime_where_statements(
        sql_where        , 
        date_ranges      = None, 
        datetime_ranges  = None, 
        field_descs_dict = None, 
        date_col         = 'aep_usage_dt', 
        datetime_col     = 'starttimeperiod', 
        datetime_pattern = r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*", 
        from_table_alias = None, 
        idx              = None
    ):
        r"""
        field_descs_dict:
            If supplied, will be returned together with sql_where
        """
        #---------------------------------------------------------------------------
        if field_descs_dict is not None:
            assert(isinstance(field_descs_dict, dict))
        #-------------------------
        if date_ranges is None and datetime_ranges is None:
            if field_descs_dict is not None:
                return sql_where, field_descs_dict
            else:
                return sql_where
        #---------------------------------------------------------------------------
        assert(date_ranges     is None or Utilities.is_object_one_of_types(date_ranges,     [list, tuple]))
        assert(datetime_ranges is None or Utilities.is_object_one_of_types(datetime_ranges, [list, tuple]))
        #---------------------------------------------------------------------------
        #***** DATE ****************************************************************
        if date_ranges is not None and len(date_ranges)>0:
            if field_descs_dict is not None:
                field_descs_dict['date_range'] = date_col
            #-------------------------
            # To make life easier (and code more compact), convert date_ranges to list of lists, if not already
            #   i.e., even if a single date_range, make date_ranges = [[date_range[0], date_range[1]]]
            assert(Utilities.is_object_one_of_types(date_ranges, [list, tuple]))
            if not Utilities.is_list_nested(lst=date_ranges, enforce_if_one_all=True):
                date_ranges = [date_ranges]
            assert(Utilities.is_list_nested(lst=date_ranges, enforce_if_one_all=True))
            assert(Utilities.are_list_elements_lengths_homogeneous(lst=date_ranges, length=2))
            #-------------------------
            sql_where, date_idx_0 = AMI_SQL.add_date_where_statements(
                sql_where        = sql_where, 
                date_ranges      = date_ranges, 
                date_col         = date_col, 
                from_table_alias = from_table_alias, 
                idx              = idx, 
                return_idx       = True
            )
        else:
            date_idx_0 = -1

        #***** DATETIME ************************************************************
        if datetime_ranges is not None and len(datetime_ranges)>0:
            if field_descs_dict is not None:
                if datetime_pattern is None:
                    dt_field_desc = f"CAST({datetime_col} AS TIMESTAMP)"
                else:
                    dt_field_desc = r"CAST(regexp_replace({}, ".format(datetime_col) + r"'{}', '$1 $2') AS TIMESTAMP)".format(datetime_pattern)
                field_descs_dict['datetime_range'] = dt_field_desc
            #-------------------------
            # To make life easier (and code more compact), convert datetime_ranges to list of lists, if not already
            #   i.e., even if a single datetime_range, make datetime_ranges = [[datetime_range[0], datetime_range[1]]]
            assert(Utilities.is_object_one_of_types(datetime_ranges, [list, tuple]))
            if not Utilities.is_list_nested(lst=datetime_ranges, enforce_if_one_all=True):
                datetime_ranges = [datetime_ranges]
            assert(Utilities.is_list_nested(lst=datetime_ranges, enforce_if_one_all=True))
            assert(Utilities.are_list_elements_lengths_homogeneous(lst=datetime_ranges, length=2))
            #-------------------------
            sql_where, datetime_idx, date_idx_1 = AMI_SQL.add_datetime_where_statements(
                sql_where        = sql_where, 
                datetime_ranges  = datetime_ranges, 
                datetime_col     = datetime_col, 
                datetime_pattern = datetime_pattern, 
                date_col         = date_col, 
                from_table_alias = from_table_alias, 
                idx              = date_idx_0+1 if date_idx_0>=0 else idx, 
                return_idxs      = True
            )
        else:
            datetime_idx, date_idx_1 = -1, -1

        #---------------------------------------------------------------------------
        # If date ranges added by both date_ranges and datetime_ranges, then the two need to be combined with an OR statement
        if date_idx_0>=0 and date_idx_1>=0:
            date_idx = sql_where.combine_where_elements_smart(
                idxs_to_combine    = [date_idx_0, date_idx_1], 
                join_operator      = 'OR', 
                close_gaps_in_keys = True, 
                return_idx         = True
            )
        else:
            if date_idx_0<0 and date_idx_1<0:
                date_idx = -1
            elif date_idx_0>=0:
                assert(date_idx_1<0)
                date_idx = date_idx_0
            elif date_idx_1>=0:
                assert(date_idx_0<0)
                date_idx = date_idx_1
            else:
                assert(0)

        #---------------------------------------------------------------------------
        # Make sure date comes before datetime
        if date_idx>=0 and datetime_idx>=0 and date_idx > datetime_idx:
            sql_where.swap_idxs(
                idx_1 = date_idx, 
                idx_2 = datetime_idx
            )
            date_idx, datetime_idx = datetime_idx, date_idx

        #---------------------------------------------------------------------------
        if field_descs_dict is not None:
            return sql_where, field_descs_dict
        else:
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
        See MeterPremise.add_inst_rmvl_ts_where_statement
        """
        #-------------------------
        return MeterPremise.add_inst_rmvl_ts_where_statement(
            sql_where        = sql_where, 
            datetime_range   = datetime_range, 
            from_table_alias = from_table_alias, 
            inst_ts_col      = inst_ts_col, 
            rmvl_ts_col      = rmvl_ts_col, 
            datetime_pattern = datetime_pattern, 
            datetime_replace = datetime_replace
        )


    @staticmethod
    def add_ami_where_statements_OLD(sql_where, **kwargs):
        r"""
        NEW METHOD CREATED 2024-01-30
        THIS METHOD SHOULD BE DELETED AFTER DOUBLE CHECKING FUNCTIONALITY OF NEW METHOD
        
        Method for adding general where statements which are used by multiple of the AMI type databases.
        e.g., usage_nonvee.reading_ivl_nonvee, usage_instantaneous.inst_msr_consume, and others have very
              similar fields.  This saves time/energy from having to write multiple versions of this codeblock
        NOTE: The _col arguments might need to be adjusted for the different dataset.
              e.g., usage_nonvee.reading_ivl_nonvee has an aep_usage_dt field for the date,
                    whereas usage_instantaneous.inst_msr_consume has aep_read_dt.
              e.g., usage_nonvee.reading_ivl_nonvee has an aep_premise_nb field for the premise numbers,
                    whereas default.meter_premise has prem_nb
        Acceptable kwargs:
          - from_table_alias
            - default: None
            
          - date_range
            - tuple with two string elements, e.g., ['2021-01-01', '2021-04-01']
          - date_col
            - default: aep_usage_dt
            
          - datetime_range
            - tuple with two string elements, e.g., ['2021-01-01 00:00:00', '2021-04-01 12:00:00']
          - datetime_col
            - default: starttimeperiod
          - datetime_pattern
            - Regex pattern used to convert string to a form that SQL/Athena/Oracle can convert to a TIMESTAMP
            - default: r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"

          - serial_number(s)
          - serial_number_col
            - default: serialnumber

          - premise_nb(s) (or, aep_premise_nb(s) will work also)
          - premise_nb_col
            - default: aep_premise_nb
            
          - trsf_pole_nb(s)
          - trsf_pole_nb_col
            - default: trsf_pole_nb

          - opco(s) (or, aep_opco(s) will work also)
          - opco_col
            - default: aep_opco

          - state(s)
          - state_col
            - default: aep_state
            
          - exclude_null_val_cols
            - default: None
            
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
        #*******************************************************
        from_table_alias = kwargs.get('from_table_alias', None)
        
        #*******************************************************
        # field_descs_dict needed if any where elements are to be combined
        field_descs_dict = dict()
        
        #***** OPCO **************************************************
        #-----
        opco_col             = kwargs.get('opco_col', 'aep_opco')
        possible_opco_kwargs = ['opco', 'aep_opco', 'opcos', 'aep_opcos']
        found_opco_kwargs    = [x for x in kwargs if x in possible_opco_kwargs]
        assert(len(found_opco_kwargs)<=1)
        if found_opco_kwargs:
            opco = kwargs[found_opco_kwargs[0]]
        else:
            opco = None
        #-----
        if opco is not None:
            field_descs_dict[found_opco_kwargs[0]] = opco_col
            sql_where = SQLWhere.add_where_statement_equality_or_in(
                sql_where          = sql_where, 
                field_desc         = opco_col, 
                value              = opco, 
                needs_quotes       = True, 
                table_alias_prefix = from_table_alias, 
                idx                = 0
            )  
        else:
            print("!!!!! WARNING !!!!! NO OPCOs SELECTED!")

        #***** DATE **************************************************
        #-----
        # NOTE: The datetime operation in the next block, which typically involves regex expressions, is a heavy lift.
        #       The query runs much smoother if date_range is used together with datetime_range (I have seen instances where
        #         running with only datetime_range returns an error from Athena).
        #       My guess is the date_range (the way I have it written) is executed first, so the set upon which the regex operations
        #         need to be run is largely reduced.
        #       THEREFORE, if datetime_range is set but date_range is not, set it as well.
        #       NOTE: This can only be done if the elements of datetime_range are timestamp objects, so I can extract the dates
        #-----
        date_col         = kwargs.get('date_col', 'aep_usage_dt')
        date_range       = kwargs.get('date_range', None)
        datetime_range   = kwargs.get('datetime_range', None)
        #-----
        if date_range is None and datetime_range is not None:
            if all([isinstance(x, datetime.datetime) for x in datetime_range]):
                assert(len(datetime_range)==2)
                date_range = [x.date() for x in datetime_range]
        #-----
        if date_range is not None:
            field_descs_dict['date_range'] = date_col
            assert((isinstance(date_range, list) or isinstance(date_range, tuple)) and 
                   len(date_range)==2)
            sql_where.add_where_statement(
                field_desc          = date_col, 
                comparison_operator = 'BETWEEN', 
                value               = [f'{date_range[0]}', f'{date_range[1]}'], 
                needs_quotes        = True, 
                table_alias_prefix  = from_table_alias, 
                idx                 = None
            )
        #***** DATETIME **************************************************
        datetime_col     = kwargs.get('datetime_col', 'starttimeperiod')
        datetime_range   = kwargs.get('datetime_range', None)
        datetime_pattern = kwargs.get('datetime_pattern', r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        if datetime_range is not None:
            assert((isinstance(datetime_range, list) or isinstance(datetime_range, tuple)) and 
                   len(datetime_range)==2)
            if from_table_alias:
                # If from_table_alias has already been added, don't add again!!!
                #   This is typical for, e.g., dt_off_ts_full/datetime_col in DOVSOutages_SQL, where, due to it's complicated
                #     structure, the from_table_alias is added before reaching this point
                if not datetime_col.startswith(f'{from_table_alias}.'):
                    datetime_col = f'{from_table_alias}.{datetime_col}'
            #-----
            if datetime_pattern is None:
                dt_field_desc = f"CAST({datetime_col} AS TIMESTAMP)"
            else:
                dt_field_desc = r"CAST(regexp_replace({}, ".format(datetime_col) + r"'{}', '$1 $2') AS TIMESTAMP)".format(datetime_pattern)
            field_descs_dict['datetime_range'] = dt_field_desc
            sql_where.add_where_statement(
                field_desc          = dt_field_desc, 
                comparison_operator = 'BETWEEN', 
                value               = [f'{datetime_range[0]}', f'{datetime_range[1]}'], 
                needs_quotes        = True, 
                table_alias_prefix  = None, 
                is_timestamp        = True, 
                idx                 = None
            )
        #***** SERIAL NUMBERS **************************************************
        serial_number_col = kwargs.get('serial_number_col', 'serialnumber')
        assert(not('serial_numbers' in kwargs and 'serial_number' in kwargs))
        serial_numbers = kwargs.get('serial_numbers', kwargs.get('serial_number', None))
        if serial_numbers is not None:
            field_descs_dict['serial_numbers' if 'serial_numbers' in kwargs else 'serial_number'] = serial_number_col
            if isinstance(serial_numbers, SQLWhereElement):
                sql_where.add_where_statement(serial_numbers)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = serial_number_col, 
                    value              = serial_numbers, 
                    table_alias_prefix = from_table_alias
                )
        #***** PREMISE NUMBERS **************************************************
        premise_nb_col              = kwargs.get('premise_nb_col', 'aep_premise_nb')
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
        #***** STATES **************************************************
        state_col = kwargs.get('state_col', 'aep_state')
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
        #***** EXCLUDE NULL VALUES ************************
        exclude_null_val_cols = kwargs.get('exclude_null_val_cols', None) 
        if exclude_null_val_cols is not None:
            for exclude_col in exclude_null_val_cols:
                sql_where.add_where_statement(
                    field_desc          = exclude_col, 
                    comparison_operator = 'IS NOT', 
                    value               = 'NULL', 
                    needs_quotes        = False, 
                    table_alias_prefix  = from_table_alias, 
                    idx                 = None
                )
        #**************************************************
        wheres_to_combine = kwargs.get('wheres_to_combine', None)
        if wheres_to_combine is not None:
            sql_where = AMI_SQL.combine_where_elements(
                sql_where         = sql_where, 
                wheres_to_combine = wheres_to_combine, 
                field_descs_dict  = field_descs_dict
            )
        #----------
        if 'field_descs_dict' not in sql_where.addtnl_info.keys():
            sql_where.addtnl_info['field_descs_dict'] = field_descs_dict
        else:
            sql_where.addtnl_info['field_descs_dict'] = {**sql_where.addtnl_info['field_descs_dict'], **field_descs_dict}
        #**************************************************
        return sql_where
        
    @staticmethod
    def add_ami_where_statements(
        sql_where, 
        **kwargs
    ):
        r"""
        Method for adding general where statements which are used by multiple of the AMI type databases.
        e.g., usage_nonvee.reading_ivl_nonvee, usage_instantaneous.inst_msr_consume, and others have very
              similar fields.  This saves time/energy from having to write multiple versions of this codeblock
        NOTE: The _col arguments might need to be adjusted for the different dataset.
              e.g., usage_nonvee.reading_ivl_nonvee has an aep_usage_dt field for the date,
                    whereas usage_instantaneous.inst_msr_consume has aep_read_dt.
              e.g., usage_nonvee.reading_ivl_nonvee has an aep_premise_nb field for the premise numbers,
                    whereas default.meter_premise has prem_nb
        Acceptable kwargs:
          - from_table_alias
            - default: None
            
          - date_range
            - tuple/list with two string elements, e.g., ['2021-01-01', '2021-04-01']
            - Can also be a list of such objects, in which case the date ranges will be combined with OR operators
          - date_col
            - default: aep_usage_dt
            
          - datetime_range
            - tuple/list with two string elements, e.g., ['2021-01-01 00:00:00', '2021-04-01 12:00:00']
            - Can also be a list of such objects, in which case the datetime ranges will be combined with OR operators
          - datetime_col
            - default: starttimeperiod
          - datetime_pattern
            - Regex pattern used to convert string to a form that SQL/Athena/Oracle can convert to a TIMESTAMP
            - default: r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"

          - serial_number(s)
          - serial_number_col
            - default: serialnumber

          - premise_nb(s) (or, aep_premise_nb(s) will work also)
          - premise_nb_col
            - default: aep_premise_nb
            
          - trsf_pole_nb(s)
          - trsf_pole_nb_col
            - default: trsf_pole_nb

          - opco(s) (or, aep_opco(s) will work also)
          - opco_col
            - default: aep_opco

          - state(s)
          - state_col
            - default: aep_state

          - city(ies)
          - city_col
            - default: aep_city
            
          - exclude_null_val_cols
            - default: None
            
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
        #*******************************************************
        from_table_alias = kwargs.get('from_table_alias', None)
        verbose          = kwargs.get('verbose', True)
        
        #*******************************************************
        # field_descs_dict needed if any where elements are to be combined
        field_descs_dict = dict()
        
        #***** OPCO **************************************************
        #-----
        opco_col             = kwargs.get('opco_col', 'aep_opco')
        possible_opco_kwargs = ['opco', 'aep_opco', 'opcos', 'aep_opcos']
        found_opco_kwargs    = [x for x in kwargs if x in possible_opco_kwargs]
        opcos_handled        = kwargs.get('opcos_handled', False)
        assert(len(found_opco_kwargs)<=1)
        if found_opco_kwargs:
            opco = kwargs[found_opco_kwargs[0]]
        else:
            opco = None
        #-----
        if opco is not None:
            field_descs_dict[found_opco_kwargs[0]] = opco_col
            sql_where = SQLWhere.add_where_statement_equality_or_in(
                sql_where          = sql_where, 
                field_desc         = opco_col, 
                value              = opco, 
                needs_quotes       = True, 
                table_alias_prefix = from_table_alias, 
                idx                = 0
            )  
        else:
            if verbose and not opcos_handled:
                print("!!!!! WARNING !!!!! NO OPCOs SELECTED!")

        #***** DATE and DATETIME *************************************
        #-----
        # NOTE: With datetime, the query runs smoother when a date cut is first applied, so datetime will actually
        #       add two WHERE statements.  See AMI_SQL.add_datetime_where_statement for more information.
        #-------------------------
        date_col         = kwargs.get('date_col', 'aep_usage_dt')
        datetime_col     = kwargs.get('datetime_col', 'starttimeperiod')
        #-----
        date_range       = kwargs.get('date_range', None)
        datetime_range   = kwargs.get('datetime_range', None)
        datetime_pattern = kwargs.get('datetime_pattern', r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        #-------------------------
        sql_where, field_descs_dict = AMI_SQL.add_date_and_datetime_where_statements(
            sql_where        = sql_where, 
            date_ranges      = date_range, 
            datetime_ranges  = datetime_range, 
            field_descs_dict = field_descs_dict, 
            date_col         = date_col, 
            datetime_col     = datetime_col, 
            datetime_pattern = datetime_pattern, 
            from_table_alias = from_table_alias, 
            idx              = None
        )

        #***** SERIAL NUMBERS **************************************************
        serial_number_col = kwargs.get('serial_number_col', 'serialnumber')
        assert(not('serial_numbers' in kwargs and 'serial_number' in kwargs))
        serial_numbers = kwargs.get('serial_numbers', kwargs.get('serial_number', None))
        if serial_numbers is not None:
            field_descs_dict['serial_numbers' if 'serial_numbers' in kwargs else 'serial_number'] = serial_number_col
            if isinstance(serial_numbers, SQLWhereElement):
                sql_where.add_where_statement(serial_numbers)
            else:
                sql_where = SQLWhere.add_where_statement_equality_or_in(
                    sql_where          = sql_where, 
                    field_desc         = serial_number_col, 
                    value              = serial_numbers, 
                    table_alias_prefix = from_table_alias
                )
        #***** PREMISE NUMBERS **************************************************
        premise_nb_col              = kwargs.get('premise_nb_col', 'aep_premise_nb')
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
        #***** STATES **************************************************
        state_col = kwargs.get('state_col', 'aep_state')
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
        city_col = kwargs.get('city_col', 'aep_city')
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
        #***** EXCLUDE NULL VALUES ************************
        exclude_null_val_cols = kwargs.get('exclude_null_val_cols', None) 
        if exclude_null_val_cols is not None:
            for exclude_col in exclude_null_val_cols:
                sql_where.add_where_statement(
                    field_desc          = exclude_col, 
                    comparison_operator = 'IS NOT', 
                    value               = 'NULL', 
                    needs_quotes        = False, 
                    table_alias_prefix  = from_table_alias, 
                    idx                 = None
                )
        #**************************************************
        wheres_to_combine = kwargs.get('wheres_to_combine', None)
        if wheres_to_combine is not None:
            sql_where = AMI_SQL.combine_where_elements(
                sql_where         = sql_where, 
                wheres_to_combine = wheres_to_combine, 
                field_descs_dict  = field_descs_dict
            )
        #----------
        if 'field_descs_dict' not in sql_where.addtnl_info.keys():
            sql_where.addtnl_info['field_descs_dict'] = field_descs_dict
        else:
            sql_where.addtnl_info['field_descs_dict'] = {**sql_where.addtnl_info['field_descs_dict'], **field_descs_dict}
        #**************************************************
        return sql_where

        
    #****************************************************************************************************
    @staticmethod
    def compile_join_mp_args_for_SQLQuery(
        join_mp_args     , 
        build_usg_kwargs = None, 
        build_mp_kwargs  = None
    ):
        r"""
        """
        #-------------------------
        if join_mp_args is None:
            join_mp_args = {}
        #-------------------------
        if 'build_mp_kwargs' in join_mp_args:
            join_table_alias = join_mp_args['build_mp_kwargs'].get('alias', 'MP')
            join_col_mp      = join_mp_args['build_mp_kwargs'].get('serialnumber_col', 'mfr_devc_ser_nbr')
        else:
            join_table_alias = 'MP'
            join_col_mp      = 'mfr_devc_ser_nbr'            
        #-------------------------
        if build_usg_kwargs is None:
            orig_table_alias = 'USG'
            join_col_usg     = 'serialnumber'
        else:
            orig_table_alias = build_usg_kwargs['from_table_alias']
            join_col_usg     = build_usg_kwargs['serialnumber_col']
        #-------------------------
        dflt_join_mp_args = dict(
            join_type                  = 'INNER', 
            join_table                 = None, 
            join_table_alias           = join_table_alias, 
            orig_table_alias           = orig_table_alias, 
            list_of_columns_to_join    = [[join_col_usg, join_col_mp]], 
            idx                        = None, 
            run_check                  = False, 
            join_cols_to_add_to_select = ['*']
        )
        #-------------------------
        # If statement below prevents, e.g., MP.mfr_devc_ser_nbr, MP.longitude, ..., xfmr_nb, MP.* in the SELECT statement,
        #   in favor of MP.mfr_devc_ser_nbr, MP.longitude, ..., xfmr_nb
        if join_mp_args.get('join_cols_to_add_to_select', None) is not None:
            dflt_join_mp_args['join_cols_to_add_to_select'] = []
        #-------------------------
        join_mp_args_for_SQLQuery = {k:v for k,v in join_mp_args.items() if k in dflt_join_mp_args}
        join_mp_args_for_SQLQuery = Utilities_sql.supplement_dict_with_default_values(join_mp_args_for_SQLQuery, dflt_join_mp_args, extend_any_lists=True)
        #-------------------------
        return join_mp_args_for_SQLQuery
        
    #****************************************************************************************************
    @staticmethod
    def asseble_sql_ami_join_mp_statement(
        usg_join_mp_dict           , 
        prepend_with_to_stmnt      = False, 
        insert_n_tabs_to_each_line = 0
    ):
        #-------------------------------------
        assert('mp_sql'  in usg_join_mp_dict)
        assert('usg_sql' in usg_join_mp_dict)
        #-----
        mp_sql    = usg_join_mp_dict['mp_sql']
        usg_sql   = usg_join_mp_dict['usg_sql']
        #-------------------------------------
        mp_sql_stmnt  = mp_sql.get_sql_statement(
            insert_n_tabs_to_each_line=0, 
            include_alias=True
        )
        usg_sql_stmnt = usg_sql.get_sql_statement(
            insert_n_tabs_to_each_line = 0, 
            include_alias              = True if usg_sql.alias is not None else False
        )
        #-------------------------------------
        if prepend_with_to_stmnt:
            return_stmnt = "WITH "
        else:
            return_stmnt = ""
            
        return_stmnt += f"{mp_sql_stmnt}"
        if usg_sql.alias is not None:
            return_stmnt += ','  
        return_stmnt += f" \n{usg_sql_stmnt}"
        #-------------------------------------
        if insert_n_tabs_to_each_line>0:
            return_stmnt = Utilities_sql.prepend_tabs_to_each_line(
                return_stmnt, 
                n_tabs_to_prepend = insert_n_tabs_to_each_line
            )
        #-------------------------------------
        return return_stmnt
    
    
    #****************************************************************************************************
    #****************************************************************************************************
    #TODO do I need to add add_aggregate_elements kwargs to allow one to tweak e.g.,  comp_table_alias_prefix etc.?
    @staticmethod
    def build_sql_ami(
        cols_of_interest, 
        **kwargs
    ):
        r"""
        See AMI_SQL.add_ami_where_statements for my updated list of acceptable kwargs with respect to the where statement.
        
        Acceptable kwargs:
          - date_range
            - tuple with two string elements, e.g., ['2021-01-01', '2021-04-01']
          - date_col
            - default: aep_usage_dt

          - serial_number(s)
          - serial_number_col
            - default: serialnumber

          - premise_nb(s) (or, aep_premise_nb(s) will work also)
          - premise_nb_col
            - default: aep_premise_nb
            
          - trsf_pole_nb(s)
          - trsf_pole_nb_col
            - default: trsf_pole_nb

          - opco(s) (or, aep_opco(s) will work also)
          - opco_col
            - default: aep_opco

          - state(s)
          - state_col
            - default: aep_state

          - city(ies)
          - city_col
            - default: aep_city
            
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


          - groupby_cols
          - agg_cols_and_types
          - include_counts_including_null

          - schema_name
          - table_name
          
          - alias (query alias)
          
          - return_args
          
          - join_mp_args
          
          - addtnl_select_elements
              - A list of additional select elements to be added to the SQL select statement
              - See SQLSelect.add_select_element for the form that each element should take
              - Built so I could more easily add, e.g., '10248878' AS OUTG_REC_NB_GPD_FOR_SQL when 
                running cleanup for first end events for outages data acquisition.
        
        TODO: How can I adjust this function to allow e.g., serial_numbers NOT IN...
                 - For now, easiest method will be to use standard build_sql_meter_premise then
                   the utilize the change_comparison_operator_of_element/change_comparison_operator_of_element_at_idx
                   method of SQLWhere to change mp_sql.sql_where after mp_sql is returned
                 - The second option would be to input serial_numbers of type SQLWhereElement with all correct
                   attributes set before inputting
        """
        # SELECT STATEMENT
        if cols_of_interest is None:
            cols_of_interest = ['*']
        #kwargs['from_table_alias'] = kwargs.get('from_table_alias', None)
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'AMI')
        sql_select                 = SQLSelect(cols_of_interest, global_table_alias_prefix=kwargs['from_table_alias'])
        #**************************************************
        # FROM STATEMENT
        schema_name = kwargs.get('schema_name', 'usage_nonvee')
        table_name  = kwargs.get('table_name', 'reading_ivl_nonvee')
        sql_from    = SQLFrom(schema_name, table_name, alias=kwargs['from_table_alias'])
        #**************************************************
        # WHERE STATEMENT
        usg_sql_where = SQLWhere()
        usg_sql_where = AMI_SQL.add_ami_where_statements(usg_sql_where, **kwargs)

        #**************************************************
        # AGGREGATE AND GROUP BY
        agg_cols_and_types            = kwargs.get('agg_cols_and_types', None)
        groupby_cols                  = kwargs.get('groupby_cols', None)
        include_counts_including_null = kwargs.get('include_counts_including_null', True)
        if agg_cols_and_types is not None:
            assert(groupby_cols is not None)
        #-----
        if agg_cols_and_types is not None:
            sql_select.add_aggregate_elements(
                agg_cols_and_types            = agg_cols_and_types, 
                include_counts_including_null = include_counts_including_null
            )
        #-----
        #TODO what should global_table_alias_prefix be called in kwargs?
        if groupby_cols is not None:
            sql_groupby = SQLGroupBy(
                field_descs               = groupby_cols, 
                global_table_alias_prefix = None, 
                idxs                      = None, 
                run_check                 = True
            )
        else:
            sql_groupby = None
        #-------------------------
        addtnl_select_elements = kwargs.get('addtnl_select_elements', None)
        if addtnl_select_elements is not None:
            assert(isinstance(addtnl_select_elements, list))
            for select_el in addtnl_select_elements:
                assert(Utilities.is_object_one_of_types(select_el, [dict, SQLSelectElement]))
                if isinstance(select_el, dict):
                    sql_select.add_select_element(**select_el)
                else:
                    sql_select.add_select_element(select_el)
        #-------------------------
        usg_sql = SQLQuery(
            sql_select  = sql_select, 
            sql_from    = sql_from, 
            sql_where   = usg_sql_where, 
            sql_groupby = sql_groupby, 
            alias       = kwargs.get('alias', None)
        )
        #-------------------------
        join_mp_args = kwargs.get('join_mp_args', False)
        # If join_mp_args is False or an empty dict (or None), no join will occur
        if not join_mp_args:
            return_args                = kwargs.get('return_args', {})
            return_statement           = return_args.get('return_statement', False)
            insert_n_tabs_to_each_line = return_args.get('insert_n_tabs_to_each_line', 0)
            include_alias              = return_args.get('include_alias', False)
            if return_statement:
                return usg_sql.get_sql_statement(
                    insert_n_tabs_to_each_line = insert_n_tabs_to_each_line, 
                    include_alias              = include_alias
                )
            else:
                return usg_sql
        #--------------------------------------------------
        # JOINING WITH MP
        #--------------------------------------------------
        return AMI_SQL.join_sql_ami_with_mp(usg_sql, kwargs, join_mp_args)
            
    #****************************************************************************************************
    @staticmethod
    def join_sql_ami_with_mp(
        usg_sql, 
        build_sql_ami_kwargs, 
        join_mp_args
    ):
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(join_mp_args, [bool, dict]))
        if isinstance(join_mp_args, bool):
            assert(join_mp_args)
            join_mp_args = {}
        #--------------------------------------------------
        # Serial numbers columns must be included in usg and mp as this is how the two are joined
        build_sql_ami_kwargs['serialnumber_col'] = build_sql_ami_kwargs.get('serialnumber_col', 'serialnumber')
        build_sql_ami_kwargs['from_table_alias'] = build_sql_ami_kwargs.get('from_table_alias', 'AMI')
        build_sql_ami_kwargs['alias']            = build_sql_ami_kwargs.get('alias', 'AMI_join_MP')
        
        if 'build_mp_kwargs' not in join_mp_args:
            join_mp_args['build_mp_kwargs'] = {}
        join_mp_args['build_mp_kwargs']['cols_of_interest'] = join_mp_args['build_mp_kwargs'].get('cols_of_interest', 
                                                                                                  TableInfos.MeterPremise_TI.std_columns_of_interest)
        join_mp_args['build_mp_kwargs']['serialnumber_col'] = join_mp_args['build_mp_kwargs'].get('serialnumber_col', 'mfr_devc_ser_nbr')
        join_mp_args['build_mp_kwargs']['alias']            = join_mp_args['build_mp_kwargs'].get('alias', 'MP')
        #-------------------------------------
        join_mp_args_for_SQLQuery = AMI_SQL.compile_join_mp_args_for_SQLQuery(
            join_mp_args     = join_mp_args, 
            build_usg_kwargs = build_sql_ami_kwargs
        )
        #-------------------------------------
        # Note: MeterPremise.build_sql_meter_premise will handle the case where a huge number of premise_nbs are fed
        # as input (see the max_n_prem_nbs, default value = 10000, argument in the aforementioned function)
        mp_sql = MeterPremise.build_sql_meter_premise(**join_mp_args['build_mp_kwargs'])
        #-------------------------------------
        usg_sql.build_and_add_join(**join_mp_args_for_SQLQuery)
        #-------------------------------------
        return_args                = build_sql_ami_kwargs.get('return_args', {})
        return_statement           = return_args.get('return_statement', True)
        insert_n_tabs_to_each_line = return_args.get('insert_n_tabs_to_each_line', 0)
        include_alias              = return_args.get('include_alias', False)
        prepend_with_to_stmnt      = return_args.get('prepend_with_to_stmnt', True)
        #-------------------------------------
        join_with_CTE = join_mp_args.get('join_with_CTE', False)
        if not join_with_CTE:
            if return_statement:
                return usg_sql.get_sql_statement(
                    insert_n_tabs_to_each_line = insert_n_tabs_to_each_line, 
                    include_alias              = include_alias
                )
            else:
                return usg_sql
        else:
            #-------------------------------------
            usg_join_mp_dict = {
                'mp_sql'  : mp_sql, 
                'usg_sql' : usg_sql
            }
            #-------------------------------------        
            if return_statement:
                return AMI_SQL.asseble_sql_ami_join_mp_statement(
                    usg_join_mp_dict, 
                    prepend_with_to_stmnt      = prepend_with_to_stmnt, 
                    insert_n_tabs_to_each_line = insert_n_tabs_to_each_line
                )
            else:
                return usg_join_mp_dict        

    #****************************************************************************************************
    @staticmethod
    def combine_sql_gnrl_and_i_coll_where_stmnts(
        sql_gnrl_and_i_coll_dict
    ):
        assert('sql_gnrl'   in sql_gnrl_and_i_coll_dict)
        assert('sql_i_coll' in sql_gnrl_and_i_coll_dict)
        sql_gnrl   = sql_gnrl_and_i_coll_dict['sql_gnrl']
        sql_i_coll = sql_gnrl_and_i_coll_dict['sql_i_coll']
        #-------------------------
        comb_where = SQLWhere()
        #-------------------------
        # First, add where statement from sql_gnrl
        n_where_elms = len(sql_gnrl.sql_where.collection_dict)
        comb_where.add_where_statements(list(sql_gnrl.sql_where.collection_dict.values()))
        comb_where.combine_last_n_where_elements(last_n=n_where_elms, join_operator='AND') # Probably not really needed...but safe to keep
        #-------------------------
        # Then, iterate through sql_i_coll, and add each
        n_sql_i = len(sql_i_coll)
        for sql_i in sql_i_coll:
            n_where_elms = len(sql_i.sql_where.collection_dict)
            comb_where.add_where_statements(list(sql_i.sql_where.collection_dict.values()))
            comb_where.combine_last_n_where_elements(last_n=n_where_elms, join_operator='AND')
        #-------------------------
        # Finally, combine the various where statements from sql_i_coll with OR operators
        comb_where.combine_last_n_where_elements(last_n=n_sql_i, join_operator='OR')
        #-------------------------
        return comb_where    
    #****************************************************************************************************
    @staticmethod
    def combine_sql_gnrl_and_i_coll(
        sql_gnrl_and_i_coll_dict
    ):
        assert('sql_gnrl'   in sql_gnrl_and_i_coll_dict)
        assert('sql_i_coll' in sql_gnrl_and_i_coll_dict)
        sql_gnrl   = sql_gnrl_and_i_coll_dict['sql_gnrl']
        sql_i_coll = sql_gnrl_and_i_coll_dict['sql_i_coll']
        #-------------------------
        return_sql = copy.deepcopy(sql_gnrl)
        comb_where = AMI_SQL.combine_sql_gnrl_and_i_coll_where_stmnts(sql_gnrl_and_i_coll_dict)
        return_sql.sql_where = comb_where
        return return_sql    
    #****************************************************************************************************
    @staticmethod
    def assemble_sql_ami_for_outages_statement(
        sql_gnrl_and_i_coll_dict , 
        final_table_alias        = None, 
        prepend_with_to_stmnt    = False, 
        split_to_CTEs            = True
    ):
        r"""
        split_to_CTEs - splits the query into a set of Common Table Expressions, one general 
                        broad cut shared by all, then a CTE for each in sql_i_coll
        Note: When split_to_CTEs is False, the ability to include the outg_rec_nb is no longer possible.
              Therefore, one would need to use, e.g., clm.match_events_in_df_to_outages
        """
        assert('sql_gnrl'   in sql_gnrl_and_i_coll_dict)
        assert('sql_i_coll' in sql_gnrl_and_i_coll_dict)
        sql_gnrl   = sql_gnrl_and_i_coll_dict['sql_gnrl']
        sql_i_coll = sql_gnrl_and_i_coll_dict['sql_i_coll']
        sql_mp     = sql_gnrl_and_i_coll_dict.get('sql_mp', None)
        #-------------------------
        if prepend_with_to_stmnt:
            return_stmnt = "WITH "
        else:
            return_stmnt = ""
        #-------------------------
        if not split_to_CTEs:
            if final_table_alias is None:
                return_stmnt          = ""
                prepend_with_to_stmnt = False # Otherwise, would be e.g. WITH SELECT ..., which doesn't make sense
                include_alias         = False
            else:
                include_alias = True
            comb_sql = AMI_SQL.combine_sql_gnrl_and_i_coll(sql_gnrl_and_i_coll_dict)
            comb_sql.alias = final_table_alias
            if sql_mp is None:
                return_stmnt += comb_sql.get_sql_statement(include_alias=include_alias)
            else:
                if return_stmnt.find('WITH')>-1:
                    prepend_with_to_stmnt=False
                else:
                    prepend_with_to_stmnt=True
                return_stmnt += AMI_SQL.asseble_sql_ami_join_mp_statement(
                    usg_join_mp_dict           = dict(
                        mp_sql  = sql_mp, 
                        usg_sql = comb_sql
                    ),
                    prepend_with_to_stmnt      = prepend_with_to_stmnt, 
                    insert_n_tabs_to_each_line = 0
                )
            return return_stmnt
        #-------------------------
        if sql_mp is None:
            sql_gnrl_stmnt = sql_gnrl.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=True)
        else:
            sql_gnrl_stmnt = AMI_SQL.asseble_sql_ami_join_mp_statement(
                usg_join_mp_dict           = dict(
                    mp_sql  = sql_mp, 
                    usg_sql = sql_gnrl
                ), 
                prepend_with_to_stmnt      = False, 
                insert_n_tabs_to_each_line = 0
            )
        return_stmnt += f"{sql_gnrl_stmnt}, \n"
        for i in range(len(sql_i_coll)):
            sql_i_stmnt   = sql_i_coll[i].get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=True)
            return_stmnt += sql_i_stmnt
            if i<len(sql_i_coll)-1 or final_table_alias is not None:
                return_stmnt += ', '
            return_stmnt += '\n'    
        return_stmnt += '\n'
        #-------------------------
        if final_table_alias is not None:
            final_union_stmnt = f"{final_table_alias} AS (\n"
            n_tabs            = 1
        else:
            final_union_stmnt = ''
            n_tabs            = 0
        for i in range(len(sql_i_coll)):
            final_union_stmnt += n_tabs*'\t'+f"SELECT * FROM {sql_i_coll[i].alias}"
            if i<len(sql_i_coll)-1:
                final_union_stmnt += "\n{}UNION\n".format(n_tabs*'\t')
        if final_table_alias is not None:
            final_union_stmnt += "\n)"
        #-------------------------
        return_stmnt += final_union_stmnt
        #-------------------------
        return return_stmnt

        
    #****************************************************************************************************
    @staticmethod
    # More general version of build_sql_ami_for_outages
    def build_sql_ami_for_df_with_search_time_window_v0(
        cols_of_interest           , 
        df_with_search_time_window , 
        build_sql_function         = None, 
        build_sql_function_kwargs  = {}, 
        sql_alias_base             = 'USG_', 
        return_args                = dict(
            return_statement           = True, 
            insert_n_tabs_to_each_line = 0
        ), 
        final_table_alias          = None, 
        prepend_with_to_stmnt      = True, 
        split_to_CTEs              = True, 
        max_n_prem_per_outg        = None, 
        join_mp_args               = False, 
        date_only                  = False, 
        df_args                    = {}
    ):
        r"""
        FOR NOW, only single mapping allowed between df and ami (due to desire for batch functionality)

        The datetime arguments are taken from df_with_search_time_window
        Therefore, no datetime arguments should be found in build_sql_function_kwargs

        Also, df_args['mapping_to_ami'] will be taken care of for the various collections.
        Therefore, the value in df_args['mapping_to_ami'] should not be found in build_sql_function_kwargs
        
        Input arguments:
            cols_of_interest:
                Columns of interest to be extracted from query.
                Fed as argument to build_sql_function
            
            df_with_search_time_window:
                A datetime containing the search time window information.
            
            build_sql_function:
                Default: AMI_SQL.build_sql_ami
                NOTE: Must return a SQLQuery object (cannot simply return a string)
                
            build_sql_function_kwargs:
                Default: Listed below
                Keyword arguments for build_sql_function.
                The exact acceptable arguments depend on the function used for build_sql_function.
                Default key/value pairs:
                    serialnumber_col: 'serialnumber'
                    from_table_alias: 'un_rin'
                    date_col:         'aep_usage_dt'
                    datetime_col:     'starttimeperiod'
                    datetime_pattern: r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
                
            sql_alias_base:
                Default: 'USG_'
                
            return_args: 
                Default: dict(return_statement=True, insert_n_tabs_to_each_line=0)
                
            final_table_alias:
                Default: None
                
            prepend_with_to_stmnt:
                Default: True
                
            split_to_CTEs:
                Default: True
                
            max_n_prem_per_outg:
                Default: None
                
            join_mp_args:
                Default: False
            
            date_only
                Default: False
                If True, only the date portion of the datetime supplied by df_with_search_time_window
                    i.e., the hours, minutes, seconds, are ignored
                
            df_args:
                Default: Listed below
                Contains arguments which are used to extract the correct information from df_with_search_time_window.
                The acceptable keys for df_args are:
                    t_search_min_col:
                        Default: 't_search_min'
                        
                    t_search_max_col:
                        Default: 't_search_max'
                        
                    addtnl_groupby_cols:
                        Default: ['OUTG_REC_NB']
                        Any additional columns in df_with_search_time_window to group by.
                        The DF is always grouped by t_search_min_col and t_search_max_col.
                        When additional columns are added, these are grouped first, i.e. the time columns are always
                          grouped last.
                        These groups basically dictate how the SQL query is divided into sub-queries.  The DF 
                          df_with_search_time_window is first grouped by addtnl_groupby_cols+[t_search_min_col, t_search_max_col].
                          The groups are then iterated over, and a subquery is built using each group.
                          For each subquery, the values of addtnl_groupby_cols are included as constant values, and can be
                            identified in the ultimate resulting DF as they are tagged with '_GPD_FOR_SQL'
                        
                    is_df_consolidated:
                        Default: False
                        
                    dropna:
                        Default: False
                        
                    mapping_to_ami:
                        Default: {'PREMISE_NB':'premise_nbs'}
                        This basically dictates what will be taken from df_with_search_time_window and used as inputs
                          to the queries run by build_sql_function.
                        The keys give the column in df_with_search_time_window from which to extract the information.
                        The values contain the parameter to be set in build_sql_function with the information extracted
                          from df_with_search_time_window
                        e.g., mapping_to_ami = {'PREMISE_NB':'premise_nbs'}
                            NOTE: OVERSIMPLIFIED, NOT TO BE TAKEN AS EXACT
                            NOTE: key_0                 = 'PREMISE_NB' 
                                  mapping_to_ami[key_0] = 'premise_nbs'
                            vals_from_df = df_with_search_time_window[key_0].unique()
                            --> sql = build_sql_function(... mapping_to_ami[key_0] = vals_from_df, ...)
                            
                    mapping_to_mp:
                        Default: {'STATE_ABBR_TX':'states'}
                        Only used when joining with MeterPremise (i.e., only when join_mp_args)
                        Similar in spirit to mapping_to_ami, except the extracted information is used in the MeterPremise
                          query (through join_mp_args['build_mp_kwargs'])
                          
                        This basically dictates what will be taken from df_with_search_time_window and used as inputs
                          to the queries run by AMI.join_sql_ami_with_mp.
                        The keys give the column in df_with_search_time_window from which to extract the information.
                        The values contain the parameter to be set in join_mp_args['build_mp_kwargs'] (and ultimately used
                          in AMI.join_sql_ami_with_mp) with the information extracted from df_with_search_time_window

                        

                
        """
        #--------------------------------------------------
        # The whole purpose of this functionality is to take the datetime parameters from df_with_search_time_window
        # Therefore, no datetime arguments should be found in build_sql_function_kwargs
        datetime_kwargs = ['date_range', 'datetime_range']
        if len(set(datetime_kwargs).intersection(set(build_sql_function_kwargs.keys()))) != 0:
            print(f'ERROR: In build_sql_ami_for_df_with_search_time_window_v0, datetime argument(s)={datetime_kwargs}'\
                  ' found in build_sql_function_kwargs\nCRASH IMMINENT')
            assert(0)
        #--------------------------------------------------
        # build_sql_function default = AMI_SQL.build_sql_ami 
        if build_sql_function is None:
            build_sql_function=AMI_SQL.build_sql_ami
        #--------------------------------------------------
        # Set up default arguments in build_sql_function_kwargs
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = {}
        build_sql_function_kwargs['serialnumber_col'] = build_sql_function_kwargs.get('serialnumber_col', 'serialnumber')
        build_sql_function_kwargs['from_table_alias'] = build_sql_function_kwargs.get('from_table_alias', 'un_rin')
        build_sql_function_kwargs['date_col']         = build_sql_function_kwargs.get('date_col', 'aep_usage_dt')
        build_sql_function_kwargs['datetime_col']     = build_sql_function_kwargs.get('datetime_col', 'starttimeperiod')
        # NOTE: Below, for datetime_pattern, using [0-9] instead of \\d (or \d) seems more robust.
        #       EMR accepts \\d but not \d, whereas Athena accepts \d but not \\d.  Both accept [0-9]
        build_sql_function_kwargs['datetime_pattern'] = build_sql_function_kwargs.get(
            'datetime_pattern', 
            r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
        )
        #--------------------------------------------------
        # Set up default arguments in df_args.  These are used to essentially translate the information in 
        # df_with_search_time_window (i.e., used to extract the correct information from DF)
        df_args['t_search_min_col']    = df_args.get('t_search_min_col', 't_search_min')
        df_args['t_search_max_col']    = df_args.get('t_search_max_col', 't_search_max')
        df_args['addtnl_groupby_cols'] = df_args.get('addtnl_groupby_cols', ['OUTG_REC_NB'])
        df_args['is_df_consolidated']  = df_args.get('is_df_consolidated', False)
        df_args['dropna']              = df_args.get('dropna', False)
        df_args['mapping_to_ami']      = df_args.get('mapping_to_ami', {'PREMISE_NB':'premise_nbs'})
        df_args['mapping_to_mp']       = df_args.get('mapping_to_mp', {'STATE_ABBR_TX':'states'})
        #--------------------------------------------------
        # For now, only single mapping allowed between df and ami (due to desire for batch functionality)
        #   Therefore, enforce len(df_args['mapping_to_ami'])==1
        #   When transition to multiple, below should be changed to for loop over keys in df_args['mapping_to_ami'],
        #     making sure each corresponding ami_field not in build_sql_function_kwargs
        #   i.e., for df_col, ami_field in df_args['mapping_to_ami'].items() assert(ami_field not in build_sql_function_kwargs)
        # Below, df_col is the column in df_with_search_time_window from which to extract the information
        #        ami_field is the parameter to be set in build_sql_function with the information extracted
        # NOTE: ami_field handled for each sql_i in collection.  Therefore, should be absent from
        #         sql_gnrl, and thus absent from build_sql_function_kwargs
        assert(len(df_args['mapping_to_ami'])==1)
        df_col    = list(df_args['mapping_to_ami'].keys())[0]
        ami_field = df_args['mapping_to_ami'][df_col]
        assert(ami_field not in build_sql_function_kwargs)
        #--------------------------------------------------
        #--------------------------------------------------
        # Determine the columns by which df_with_search_time_window will be grouped.
        # The DF is always grouped by t_search_min_col and t_search_max_col.
        #   df_args['addtnl_groupby_cols'] prepends columns to the collection
        # These groups basically dictate how the SQL query is divided into sub-queries.
        #   The groups are iterated over, and a subquery is built using each group.
        #   NOTE: For each subquery, the values of addtnl_groupby_cols are included as constant values, 
        #         and can be identified in the ultimate resulting DF as they are tagged with '_GPD_FOR_SQL'
        if df_args['addtnl_groupby_cols'] is None:
            df_args['addtnl_groupby_cols'] = []
            groupby_cols                   = []
        elif Utilities.is_object_one_of_types(df_args['addtnl_groupby_cols'], [list, tuple]):
            # NOTE: Need to make copy of df_args['addtnl_groupby_cols'] below.
            #       Cannot simply call, e.g., groupby_cols = df_args['addtnl_groupby_cols'], as this would
            #         cause t_search_min/max to also be appended to df_args['addtnl_groupby_cols'] when they 
            #         are appended to groupby_cols, which is not desired
            #       Not sure why I was so careful here?  Doesn't really matter, as this should be harmless performance-wise
            groupby_cols = copy.deepcopy(df_args['addtnl_groupby_cols'])
        else:
            assert(isinstance(df_args['addtnl_groupby_cols'], str))
            groupby_cols = [df_args['addtnl_groupby_cols']]
        #----------
        groupby_cols.extend([df_args['t_search_min_col'], df_args['t_search_max_col']])
        #--------------------------------------------------
        # Build the grouped version of df_with_search_time_window
        # Grab both the group keys and the number of groups
        df_gpd     = df_with_search_time_window.groupby(groupby_cols, dropna=df_args['dropna'])
        group_keys = list(df_gpd.groups.keys())
        n_groups   = df_gpd.ngroups
        #assert(len(group_keys)==n_groups)
        #--------------------------------------------------
        # group_keys contains the list of groups in df_with_search_time_window.groupby(groupby_cols)
        #   The length of group_keys is the number of groups.
        #   The length of each group in group_keys should equal the length of groupby_cols, which is equal
        #     to the length of df_args['addtnl_groupby_cols'] plus 2 (for t_min and t_max)
        #   In each group, t_min and t_max are the last two elements.  Using the t_min,t_max values from all
        #     groups, build the unique date ranges.
        #   unique_date_ranges will be used in the general SQL statement, before the sub-queries.  Therefore, since
        #     there can be duplicate date ranges, and since many date ranges likely overlap, get only the unique
        #     overlap intervals for efficiency
        unique_date_ranges = [
            (pd.to_datetime(x[-2]).date(), pd.to_datetime(x[-1]).date()) 
            for x in group_keys
        ]
        unique_date_ranges = Utilities.get_overlap_intervals(unique_date_ranges)
        #--------------------------------------------------
        #--------------------------------------------------
        # Build general SQL statement, sql_gnrl
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = ['*']
        #-------------------------
        # In general, we want to cast the datetime column as a timestamp in SQL general so it won't need to be done in following CTEs/where statements/whatever
        # As such, build_sql_function_kwargs['datetime_col'] should be included in the cols_of_interest (except if date_only==True)
        new_dt_alias = build_sql_function_kwargs['datetime_col']
        if not date_only and build_sql_function_kwargs['datetime_col'] not in cols_of_interest:
            # Special case: If cols_of_intererst==['*'], then datetime column will essentially be selected twice, so that
            #   cast as a timestamp must be given a unique alias (which will then be used by following CTEs instead of original datetime column)
            if cols_of_interest == ['*']:
                new_dt_alias = build_sql_function_kwargs['datetime_col']+'_ts'
            #-----
            cols_of_interest.append(build_sql_function_kwargs['datetime_col'])
        #-------------------------
        # Need build_sql_function to return a SQLQuery object, not a string or anything else (-->return_statement=False)
        build_sql_function_kwargs['return_args'] = dict(return_statement=False)
        #-------------------------
        # If join_mp_args, use df_args['mapping_to_mp'] to fully set up join_mp_args['build_mp_kwargs']
        # NOTE: If join_mp_args is False or an empty dict, no join will occur
        # NOTE: In df_args['mapping_to_mp'], the keys give the column in df_with_search_time_window from which to 
        #       extract the information, and the values contain the parameter to be set in join_mp_args['build_mp_kwargs'] 
        #       with the information extracted
        assert(Utilities.is_object_one_of_types(join_mp_args, [bool, dict]))
        if join_mp_args:
            if isinstance(join_mp_args, bool):
                join_mp_args = {}
            if 'build_mp_kwargs' not in join_mp_args:
                join_mp_args['build_mp_kwargs'] = {}
            #-----
            for df_col_mp, mp_field in df_args['mapping_to_mp'].items():
                join_mp_args['build_mp_kwargs'][mp_field] = df_with_search_time_window[df_col_mp].unique().tolist()
        else:
            join_mp_args = False
        #-------------------------
        # Build sql_gnrl
        sql_gnrl = build_sql_function(
            cols_of_interest = cols_of_interest, 
            join_mp_args     = join_mp_args, 
            **build_sql_function_kwargs
        )
        #-------------------------
        # As mentioned above, we want to cast the datetime column as a timestamp in SQL general
        if not date_only:
            sql_gnrl.cast_sql_select_element_as_timestamp(
                field_desc       = build_sql_function_kwargs['datetime_col'], 
                datetime_pattern = build_sql_function_kwargs['datetime_pattern'], 
                new_alias        = new_dt_alias
            )
        #-------------------------
        # If joined with MP, grab sql_mp and sql_gnrl
        if isinstance(sql_gnrl, dict):
            assert(join_mp_args) # Should only happen when joining with MP
            assert('usg_sql' in sql_gnrl)
            sql_mp   = sql_gnrl['mp_sql']
            sql_gnrl = sql_gnrl['usg_sql']
        else: 
            sql_mp = None
        #-------------------------
        # Add where statements for all the date ranges in unique_date_ranges
        sql_gnrl_where = sql_gnrl.sql_where
        if sql_gnrl_where is None:
            sql_gnrl_where = SQLWhere()
        for unq_date_range in unique_date_ranges:
            sql_gnrl_where.add_where_statement(
                field_desc          = build_sql_function_kwargs['date_col'] , 
                comparison_operator = 'BETWEEN', 
                value               = [f'{unq_date_range[0]}', f'{unq_date_range[1]}'], 
                needs_quotes        = True, 
                table_alias_prefix  = build_sql_function_kwargs['from_table_alias']
            )
        sql_gnrl_where.combine_last_n_where_elements(last_n=len(unique_date_ranges), join_operator = 'OR')
        sql_gnrl.sql_where = sql_gnrl_where # Probably not necessary, but to be safe
        sql_gnrl.alias     = f'{sql_alias_base}gnrl'
        #--------------------------------------------------

        #--------------------------------------------------
        # Build the collection of SQL sub-queries for each group in df_gpd
        #-------------------------
        sql_i_coll = []
        count      = 0 # Count is used as appendix to give each sub-query a unique alias (=f'{sql_alias_base}{count}')
        for addtnl_groupby_vals_w_t_search_min_max, sub_df in df_gpd:
            addtnl_groupby_vals = addtnl_groupby_vals_w_t_search_min_max[:-2]
            t_search_min        = addtnl_groupby_vals_w_t_search_min_max[-2]
            t_search_max        = addtnl_groupby_vals_w_t_search_min_max[-1]
            #----------        
    #         # If df_args['mapping_to_ami'] is allowed to have multiple mappings
    #         # code similar to that commented out below will be needed
    #         ami_input = {}
    #         for df_col, ami_field in df_args['mapping_to_ami'].items():
    #             assert(ami_field not in ami_input)
    #             if df_args['is_df_consolidated']:
    #                 assert(sub_df.shape[0]==1)
    #                 ami_field_values = sub_df.iloc[0][df_col]
    #                 assert(Utilities.is_object_one_of_types(ami_field_values, [list, tuple]))
    #                 ami_field_values = [x for x in ami_field_values if pd.notna(x)] # In test case, first entry had all NaN prem nbs
    #             else:
    #                 ami_field_values = sub_df[df_col].unique().tolist()
    #             #-----
    #             if len(ami_field_values)==0:
    #                 continue
    #             ami_input[ami_field] = ami_field_values
    #         #----------
    #         if not ami_input:
    #             continue
    #         #----------
            # For now, only single mapping allowed between df and ami (due to desire for batch functionality)
            #   Therefore, enforce len(df_args['mapping_to_ami'])==1
            # Below, df_col is the column in df_with_search_time_window from which to extract the information
            #        ami_field is the parameter to be set in build_sql_function with the information extracted
            assert(len(df_args['mapping_to_ami'])==1)
            ami_input = {}
            df_col    = list(df_args['mapping_to_ami'].keys())[0]
            ami_field = df_args['mapping_to_ami'][df_col]
            # Using df_col, extract the infrmation (i.e., values) from sub_df
            if df_args['is_df_consolidated']:
                assert(sub_df.shape[0]==1)
                ami_field_values = sub_df.iloc[0][df_col]
                assert(Utilities.is_object_one_of_types(ami_field_values, [list, tuple]))
                ami_field_values = [x for x in ami_field_values if pd.notna(x)] # In test case, first entry had all NaN prem nbs
                if len(ami_field_values)==0:
                    continue
            else:
                ami_field_values = sub_df[df_col].unique().tolist()
            assert(ami_field not in ami_input)
            ami_input[ami_field] = ami_field_values
            #--------------------------------------------------
            # Enforce max_n_prem_per_outg, running by batches if the collection size exceeds max_n
            # TODO To allows for more mappings, maybe have max_n be a dict?
            if max_n_prem_per_outg is not None and len(ami_field_values)>max_n_prem_per_outg:
                batch_idxs = Utilities.get_batch_idx_pairs(len(ami_field_values), max_n_prem_per_outg)
            else:
                batch_idxs = [[0, len(ami_field_values)]]
            #-------------------------
            # Generate a sub-query for each batch in batch_idxs (or, I guess one could think of it as a sub-sub-query as
            #   the query for this group is split into multiple when len(batch_idxs)>1)
            for batch_i in batch_idxs:
                i_beg       = batch_i[0]
                i_end       = batch_i[1]
                ami_input_i = {ami_field:ami_field_values[i_beg:i_end]}
                # TODO COULD DO: ami_input_i = {k:v[i_beg:i_end] for k,v in ami_input.items()}
                #-----
                # NOTE: the datetime casting was taken care of in sql_gnrl, so datetime_pattern should be set to 'no_cast' and datetime_col should
                #   be set to the casted column
                # NOTE: After sql_general built, turn off verbose (otherwise, e.g., "No OPCOs SELECTED" warning would hit for each sub-query)
                kwargs_i = dict(
                    cols_of_interest = [f"*"], 
                    datetime_range   = [t_search_min, t_search_max], 
                    datetime_pattern = 'no_cast', 
                    datetime_col     = build_sql_function_kwargs['datetime_col'] if new_dt_alias is None else new_dt_alias, 
                    table_name       = sql_gnrl.alias, 
                    schema_name      = None, 
                    from_table_alias = build_sql_function_kwargs['from_table_alias'], 
                    verbose          = False
                )
                if date_only:
                    del kwargs_i['datetime_range']
                    kwargs_i['date_range'] = [
                        pd.to_datetime(t_search_min).date(), 
                        pd.to_datetime(t_search_max).date()
                    ]
                kwargs_i = kwargs_i|ami_input_i
                sql_i    = build_sql_function(**kwargs_i)
                #----------
                #sql_i.sql_from = SQLFrom(table_name=sql_gnrl.alias)
                #----------
                # For each subquery, the values of addtnl_groupby_cols are included as constant values, and can be
                #   identified in the ultimate resulting DF as they are tagged with '_GPD_FOR_SQL'
                assert(len(df_args['addtnl_groupby_cols'])==len(addtnl_groupby_vals)) # Sanity check
                for i_gpby_col in range(len(df_args['addtnl_groupby_cols'])):
                    gpby_col     = df_args['addtnl_groupby_cols'][i_gpby_col]
                    gpby_col_val = addtnl_groupby_vals[i_gpby_col]
                    #TODO In build_sql_ami_for_outages, had field_desc=f"'{int(outg_rec_nb)}'"
                    sql_i.sql_select.add_select_element(
                        field_desc = f"'{gpby_col_val}'", 
                        alias      = f'{gpby_col}_GPD_FOR_SQL'
                    )
                #-----
                sql_i.alias = f'{sql_alias_base}{count}'
                #-----
                sql_i_coll.append(sql_i)
                count += 1
        #--------------------------------------------------
        return_dict = {
            'sql_gnrl'   : sql_gnrl, 
            'sql_i_coll' : sql_i_coll, 
            'sql_mp'     : sql_mp
        }
        #--------------------------------------------------
        return_statement           = return_args.get('return_statement', False)
        insert_n_tabs_to_each_line = return_args.get('insert_n_tabs_to_each_line', 0)
        prepend_with_to_stmnt      = return_args.get('prepend_with_to_stmnt', True) 
        if return_statement:
            return AMI_SQL.assemble_sql_ami_for_outages_statement(
                return_dict, 
                final_table_alias     = final_table_alias, 
                prepend_with_to_stmnt = prepend_with_to_stmnt, 
                split_to_CTEs         = split_to_CTEs
            )
        else:
            return return_dict 
            
    #****************************************************************************************************
    @staticmethod
    def build_sql_ami_for_outages_v0(
        cols_of_interest          , 
        df_outage                 , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        return_args               = dict(
            return_statement           = True, 
            insert_n_tabs_to_each_line = 0
        ), 
        final_table_alias         = None, 
        prepend_with_to_stmnt     = True, 
        split_to_CTEs             = True, 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        date_only                 = False, 
        df_args                   = {}
    ):
        return AMI_SQL.build_sql_ami_for_df_with_search_time_window_v0(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_outage, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            return_args                = return_args, 
            final_table_alias          = final_table_alias, 
            prepend_with_to_stmnt      = prepend_with_to_stmnt, 
            split_to_CTEs              = split_to_CTEs, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            df_args                    = df_args
        )
    
    #****************************************************************************************************
    @staticmethod
    def build_sql_ami_for_no_outages_v0(
        cols_of_interest          , 
        df_mp_no_outg             , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {},  
        sql_alias_base            = 'USG_', 
        return_args               = dict(
            return_statement           = True, 
            insert_n_tabs_to_each_line = 0
        ), 
        final_table_alias         = None, 
        prepend_with_to_stmnt     = True, 
        split_to_CTEs             = True, 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        date_only                 = False, 
        df_args                   = {}
    ):
        r"""
        build_sql_function allows this to be used for e.g. build_sql_usg_for_outages, build_sql_end_events_for_outages,
                           build_sql_usg_inst_for_outages, etc.
                           NOTE: python won't let me set the default value to build_sql_function=AMI_SQL.build_sql_ami.
                                 Thus, the workaround is to set the default to None, then set it to 
                                 AMINonVeeSQL.build_sql_usg if it is None

        TODO: Investigate what an appropriate default value for max_n_prem_per_outg would be....
              Probably something large, like 1000.  But, should find outages with huge numbers of customers
              affected to find the right number.
        """
        #--------------------------------------------------
        if df_args is None:
            df_args = {}
        df_args['t_search_min_col']    = df_args.get('t_search_min_col', 'start_date')
        df_args['t_search_max_col']    = df_args.get('t_search_max_col', 'end_date')
        df_args['addtnl_groupby_cols'] = df_args.get('addtnl_groupby_cols', None)
        df_args['mapping_to_ami']      = df_args.get('mapping_to_ami', {'prem_nb':'premise_nbs'})
        df_args['mapping_to_mp']       = df_args.get('mapping_to_mp', {'state_cd':'states'})
        #-----

        return AMI_SQL.build_sql_ami_for_df_with_search_time_window_v0(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_mp_no_outg, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            return_args                = return_args, 
            final_table_alias          = final_table_alias, 
            prepend_with_to_stmnt      = prepend_with_to_stmnt, 
            split_to_CTEs              = split_to_CTEs, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            df_args                    = df_args
        )
    

    #****************************************************************************************************
    def create_mapping_table_rows(
        df                  , 
        join_cols           , 
        mapped_cols         ,
        timestamp_cols      = None, 
        consolidated_col    = None, 
        include_final_comma = False
    ):
        r"""
        Basically, takes df[join_cols+mapped_cols] and makes a string that can be fed to a SQL method to create a temporary table

        e.g.,
            df = 
                    prem_nb	OUTG_REC_NB	trsf_pole_nb
                0	107807398	14103171	1866459723680
                0	100402938	14103171	1866459723680
                0	101445052	14103171	1866459723680
                0	101608997	14103171	1866459723680
                0	103635052	14103171	1866459723680

            return_str = "ROW('107807398','14103171','1866459723680'), \nROW('100402938','14103171','1866459723680'), \nROW('101445052','14103171','1866459723680'), \nROW('101608997','14103171','1866459723680'), \nROW('103635052','14103171','1866459723680')"

            print(return_str) = 
                ROW('107807398','14103171','1866459723680'), 
                ROW('100402938','14103171','1866459723680'), 
                ROW('101445052','14103171','1866459723680'), 
                ROW('101608997','14103171','1866459723680'), 
                ROW('103635052','14103171','1866459723680')
        """
        #--------------------------------------------------
        assert(isinstance(join_cols, list))
        assert(isinstance(mapped_cols, list))
        assert(timestamp_cols is None or isinstance(timestamp_cols, list))
        assert(set(join_cols).intersection(set(mapped_cols))==set())
        #-----
        nec_cols = join_cols+mapped_cols
        #-----
        assert(set(nec_cols).difference(df.columns.tolist())==set())
        #-------------------------
        if consolidated_col is not None:
            assert(consolidated_col in nec_cols)
            df = df[nec_cols].explode(consolidated_col)
        else:
            df = df[nec_cols]
        #-------------------------
        if timestamp_cols is not None:
            df = df.copy()
            assert(set(timestamp_cols).difference(set(nec_cols))==set())
            for col_i in nec_cols:
                df[col_i] = df[col_i].apply(lambda x: f"'{x}'")
            #-----
            for ts_col_i in timestamp_cols:
                df[ts_col_i] = df[ts_col_i].apply(lambda x: f"CAST({x} as TIMESTAMP)")
            #-----
            return_srs = df.apply(
                lambda x: f"ROW({Utilities.join_list(list_to_join = x.values.tolist(), quotes_needed=False, join_str = ',')}), ", 
                axis=1
            )
        else:
            return_srs = df.apply(
                lambda x: f"ROW({Utilities.join_list_w_quotes(list_to_join = x.values.tolist(), join_str = ',')}), ", 
                axis=1
            )
        #-------------------------
        return_str = Utilities.join_list(
            list_to_join  = return_srs.values.tolist(), 
            quotes_needed = False, 
            join_str      = '\n'
        )
        #-------------------------
        if not include_final_comma:
            # Above methodology joins all with a comma.  However, we do not want this for the final row
            # The lines below remove the comma (and any other junk at the end)
            # It really shouldn't take more the 2-3 iterations, max.  The purpose of n_max_itr is to prevent
            #   possibility of any runaway while loops (which should never occur, but safest to ensure they can't occur)
            n_max_itr = 5
            counts = 0
            while not return_str.endswith(')') and counts < n_max_itr:
                return_str = return_str[:-1]
                counts += 1
            #-------------------------
            if counts>=n_max_itr:
                print('WARNING: AMI_SQL.create_mapping_table_rows FAILED!')
        #-------------------------
        return return_str
    
    #****************************************************************************************************
    def finalize_mapping_table(
        table_rows   , 
        join_cols    , 
        mapped_cols  , 
        alias        = None, 
        include_with = False, 
        mapped_tag   = '_GPD_FOR_SQL'
    ):
        r"""
        table_rows:
            Should be a string created by AMI_SQL.create_mapping_table_rows or a list of such strings
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(table_rows, [str, list]))
        #-----
        if isinstance(table_rows, str):
            table_rows = [table_rows]
        #-----
        assert(
            Utilities.are_all_list_elements_of_type(
                lst = table_rows, 
                typ = str
            )
        )
        #-------------------------
        table_rows_str = Utilities.join_list(
            list_to_join  = table_rows, 
            quotes_needed = False, 
            join_str      = '\n'
        )
        table_rows_str = Utilities.prepend_tabs_to_each_line(
            input_statement   = table_rows_str, 
            n_tabs_to_prepend = 1
        )
        #-------------------------
        # table_rows_str should end with ")", not, e.g., ","
        # The lines below remove the comma (and any other junk at the end)
        # It really shouldn't take more the 2-3 iterations, max.  The purpose of n_max_itr is to prevent
        #   possibility of any runaway while loops (which should never occur, but safest to ensure they can't occur)
        n_max_itr = 5
        counts = 0
        while not table_rows_str.endswith(')') and counts < n_max_itr:
            table_rows_str = table_rows_str[:-1]
            counts += 1
        #-----
        if counts>=n_max_itr:
            print('WARNING: AMI_SQL.finalize_mapping_table FAILED!')
        #-------------------------
        return_stmnt = """
        SELECT *
        FROM (
            VALUES
            {}
        )
        AS tmp_table({})
        """
        #-----
        # Silly formatting using """ """ causes leading \n (without putting SELECT on next line, though
        #   the textwrap.dedent would not function as desired)
        if return_stmnt.startswith('\n'):
            return_stmnt = return_stmnt[1:]
        #-----
        return_stmnt = textwrap.dedent(return_stmnt)
        #-----
        if mapped_tag is not None:
            mapped_cols = [f"{x}{mapped_tag}" for x in mapped_cols]
        #-----
        return_stmnt = return_stmnt.format(
            table_rows_str, 
            Utilities.join_list(
                list_to_join  = join_cols+mapped_cols, 
                quotes_needed = False, 
                join_str      = ','
            )
        )
        #-------------------------
        if alias is None:
            include_with = False
        #-----
        if alias is not None:
            return_stmnt = Utilities_sql.prepend_tabs_to_each_line(return_stmnt, n_tabs_to_prepend=1)
            return_stmnt = f"{alias} AS (\n{return_stmnt}\n)"
        #-----
        if include_with:
            return_stmnt = f"WITH {return_stmnt}"
        #-------------------------
        return return_stmnt
    
    #****************************************************************************************************
    @staticmethod
    # More general version of build_sql_ami_for_outages
    def build_sql_ami_for_df_with_search_time_window(
        cols_of_interest           , 
        df_with_search_time_window , 
        build_sql_function         = None, 
        build_sql_function_kwargs  = {}, 
        sql_alias_base             = 'USG_', 
        max_n_prem_per_outg        = None, 
        join_mp_args               = False, 
        date_only                  = False, 
        df_args                    = {}
    ):
        r"""
        FOR NOW, only single mapping allowed between df and ami (due to desire for batch functionality)

        The datetime arguments are taken from df_with_search_time_window
        Therefore, no datetime arguments should be found in build_sql_function_kwargs

        Also, df_args['mapping_to_ami'] will be taken care of for the various collections.
        Therefore, the value in df_args['mapping_to_ami'] should not be found in build_sql_function_kwargs
        
        Input arguments:
            cols_of_interest:
                Columns of interest to be extracted from query.
                Fed as argument to build_sql_function
            
            df_with_search_time_window:
                A datetime containing the search time window information.
            
            build_sql_function:
                Default: AMI_SQL.build_sql_ami
                NOTE: Must return a SQLQuery object (cannot simply return a string)
                
            build_sql_function_kwargs:
                Default: Listed below
                Keyword arguments for build_sql_function.
                The exact acceptable arguments depend on the function used for build_sql_function.
                Default key/value pairs:
                    serialnumber_col: 'serialnumber'
                    from_table_alias: 'un_rin'
                    date_col:         'aep_usage_dt'
                    datetime_col:     'starttimeperiod'
                    datetime_pattern: r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
                
            sql_alias_base:
                Default: 'USG_'
                
            max_n_prem_per_outg:
                Default: None
                
            join_mp_args:
                Default: False
            
            date_only
                Default: False
                If True, only the date portion of the datetime supplied by df_with_search_time_window
                    i.e., the hours, minutes, seconds, are ignored
                
            df_args:
                Default: Listed below
                Contains arguments which are used to extract the correct information from df_with_search_time_window.
                The acceptable keys for df_args are:
                    t_search_min_col:
                        Default: 't_search_min'
                        
                    t_search_max_col:
                        Default: 't_search_max'
                        
                    addtnl_groupby_cols:
                        Default: ['OUTG_REC_NB']
                        Any additional columns in df_with_search_time_window to group by.
                        The DF is always grouped by t_search_min_col and t_search_max_col.
                        When additional columns are added, these are grouped first, i.e. the time columns are always
                          grouped last.
                        These groups basically dictate how the SQL query is divided into sub-queries.  The DF 
                          df_with_search_time_window is first grouped by addtnl_groupby_cols+[t_search_min_col, t_search_max_col].
                          The groups are then iterated over, and a subquery is built using each group.
                          For each subquery, the values of addtnl_groupby_cols are included as constant values, and can be
                            identified in the ultimate resulting DF as they are tagged with '_GPD_FOR_SQL'
                        
                    is_df_consolidated:
                        Default: False
                        
                    dropna:
                        Default: False
                        
                    mapping_to_ami:
                        Default: DfToSqlMap(df_col='PREMISE_NB', kwarg='premise_nbs', sql_col='aep_premise_nb')
                        This basically dictates what will be taken from df_with_search_time_window and used as inputs
                          to the queries run by build_sql_function.
                        The keys give the column in df_with_search_time_window from which to extract the information.
                        The values contain the parameter to be set in build_sql_function with the information extracted
                          from df_with_search_time_window
                        e.g., mapping_to_ami = DfToSqlMap(df_col='PREMISE_NB', kwarg='premise_nbs', sql_col='aep_premise_nb')
                            NOTE: OVERSIMPLIFIED, NOT TO BE TAKEN AS EXACT
                            NOTE: df_col    = 'PREMISE_NB' 
                                  ami_field = 'premise_nbs'
                            vals_from_df = df_with_search_time_window[df_col].unique()
                            --> sql = build_sql_function(... ami_field = vals_from_df, ...)
                            
                    mapping_to_mp:
                        Default: {'STATE_ABBR_TX':'states'}
                        Only used when joining with MeterPremise (i.e., only when join_mp_args)
                        Similar in spirit to mapping_to_ami, except the extracted information is used in the MeterPremise
                          query (through join_mp_args['build_mp_kwargs'])
                          
                        This basically dictates what will be taken from df_with_search_time_window and used as inputs
                          to the queries run by AMI.join_sql_ami_with_mp.
                        The keys give the column in df_with_search_time_window from which to extract the information.
                        The values contain the parameter to be set in join_mp_args['build_mp_kwargs'] (and ultimately used
                          in AMI.join_sql_ami_with_mp) with the information extracted from df_with_search_time_window

                        

                
        """
        #--------------------------------------------------
        # The whole purpose of this functionality is to take the datetime parameters from df_with_search_time_window
        # Therefore, no datetime arguments should be found in build_sql_function_kwargs
        datetime_kwargs = ['date_range', 'datetime_range']
        if len(set(datetime_kwargs).intersection(set(build_sql_function_kwargs.keys()))) != 0:
            print(f'ERROR: In build_sql_ami_for_df_with_search_time_window, datetime argument(s)={datetime_kwargs}'\
                  ' found in build_sql_function_kwargs\nCRASH IMMINENT')
            assert(0)
        #--------------------------------------------------
        # build_sql_function default = AMI_SQL.build_sql_ami 
        if build_sql_function is None:
            build_sql_function=AMI_SQL.build_sql_ami
        #--------------------------------------------------
        # Set up default arguments in build_sql_function_kwargs
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = {}
        build_sql_function_kwargs['serialnumber_col'] = build_sql_function_kwargs.get('serialnumber_col', 'serialnumber')
        build_sql_function_kwargs['from_table_alias'] = build_sql_function_kwargs.get('from_table_alias', 'un_rin')
        build_sql_function_kwargs['date_col']         = build_sql_function_kwargs.get('date_col', 'aep_usage_dt')
        build_sql_function_kwargs['datetime_col']     = build_sql_function_kwargs.get('datetime_col', 'starttimeperiod')
        # NOTE: Below, for datetime_pattern, using [0-9] instead of \\d (or \d) seems more robust.
        #       EMR accepts \\d but not \d, whereas Athena accepts \d but not \\d.  Both accept [0-9]
        build_sql_function_kwargs['datetime_pattern'] = build_sql_function_kwargs.get(
            'datetime_pattern', 
            r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*"
        )
        #--------------------------------------------------
        # Set up default arguments in df_args.  These are used to essentially translate the information in 
        # df_with_search_time_window (i.e., used to extract the correct information from DF)
        df_args['t_search_min_col']    = df_args.get('t_search_min_col', 't_search_min')
        df_args['t_search_max_col']    = df_args.get('t_search_max_col', 't_search_max')
        df_args['addtnl_groupby_cols'] = df_args.get('addtnl_groupby_cols', ['OUTG_REC_NB'])
        df_args['is_df_consolidated']  = df_args.get('is_df_consolidated', False)
        df_args['dropna']              = df_args.get('dropna', False)
        df_args['mapping_to_ami']      = df_args.get('mapping_to_ami', DfToSqlMap(df_col='PREMISE_NB', kwarg='premise_nbs', sql_col='aep_premise_nb'))
        df_args['mapping_to_mp']       = df_args.get('mapping_to_mp', {'STATE_ABBR_TX':'states'})
        #--------------------------------------------------
        # For now, only single mapping allowed between df and ami (due to desire for batch functionality)
        #   Therefore, enforce len(df_args['mapping_to_ami'])==1
        #   When transition to multiple, below should be changed to for loop over keys in df_args['mapping_to_ami'],
        #     making sure each corresponding ami_field not in build_sql_function_kwargs
        #   i.e., for df_col, ami_field in df_args['mapping_to_ami'].items() assert(ami_field not in build_sql_function_kwargs)
        # Below, df_col is the column in df_with_search_time_window from which to extract the information
        #        ami_field is the parameter to be set in build_sql_function with the information extracted
        # NOTE: ami_field handled for each sql_i in collection.  Therefore, should be absent from
        #         sql_gnrl, and thus absent from build_sql_function_kwargs
        #-----
        assert(Utilities.is_object_one_of_types(df_args['mapping_to_ami'], [list, DfToSqlMap]))
        if isinstance(df_args['mapping_to_ami'], DfToSqlMap):
            df_args['mapping_to_ami'] = [df_args['mapping_to_ami']]
        #-----
        assert(len(df_args['mapping_to_ami'])==1)
        mapping_to_ami_i = df_args['mapping_to_ami'][0]
        df_col    = mapping_to_ami_i.df_col
        ami_field = mapping_to_ami_i.kwarg
        assert(ami_field not in build_sql_function_kwargs)
        #--------------------------------------------------
        #--------------------------------------------------
        # Determine the columns by which df_with_search_time_window will be grouped.
        # The DF is always grouped by t_search_min_col and t_search_max_col.
        #   df_args['addtnl_groupby_cols'] prepends columns to the collection
        # These groups basically dictate how the SQL query is divided into sub-queries.
        #   The groups are iterated over, and a subquery is built using each group.
        #   NOTE: For each subquery, the values of addtnl_groupby_cols are included as constant values, 
        #         and can be identified in the ultimate resulting DF as they are tagged with '_GPD_FOR_SQL'
        if df_args['addtnl_groupby_cols'] is None:
            df_args['addtnl_groupby_cols'] = []
            groupby_cols                   = []
        elif Utilities.is_object_one_of_types(df_args['addtnl_groupby_cols'], [list, tuple]):
            # NOTE: Need to make copy of df_args['addtnl_groupby_cols'] below.
            #       Cannot simply call, e.g., groupby_cols = df_args['addtnl_groupby_cols'], as this would
            #         cause t_search_min/max to also be appended to df_args['addtnl_groupby_cols'] when they 
            #         are appended to groupby_cols, which is not desired
            #       Not sure why I was so careful here?  Doesn't really matter, as this should be harmless performance-wise
            groupby_cols = copy.deepcopy(df_args['addtnl_groupby_cols'])
        else:
            assert(isinstance(df_args['addtnl_groupby_cols'], str))
            groupby_cols = [df_args['addtnl_groupby_cols']]
        #----------
        groupby_cols.extend([df_args['t_search_min_col'], df_args['t_search_max_col']])
        #--------------------------------------------------
        # Build the grouped version of df_with_search_time_window
        # Grab both the group keys and the number of groups
        df_gpd     = df_with_search_time_window.groupby(groupby_cols, dropna=df_args['dropna'])
        group_keys = list(df_gpd.groups.keys())
        n_groups   = df_gpd.ngroups
        #assert(len(group_keys)==n_groups)
        #--------------------------------------------------
        # group_keys contains the list of groups in df_with_search_time_window.groupby(groupby_cols)
        #   The length of group_keys is the number of groups.
        #   The length of each group in group_keys should equal the length of groupby_cols, which is equal
        #     to the length of df_args['addtnl_groupby_cols'] plus 2 (for t_min and t_max)
        #   In each group, t_min and t_max are the last two elements.  Using the t_min,t_max values from all
        #     groups, build the unique date ranges.
        #   unique_date_ranges will be used in the general SQL statement, before the sub-queries.  Therefore, since
        #     there can be duplicate date ranges, and since many date ranges likely overlap, get only the unique
        #     overlap intervals for efficiency
        unique_date_ranges = [
            (pd.to_datetime(x[-2]).date(), pd.to_datetime(x[-1]).date()) 
            for x in group_keys
        ]
        unique_date_ranges = Utilities.get_overlap_intervals(unique_date_ranges)
        #--------------------------------------------------
        #--------------------------------------------------
        # Build general SQL statement, sql_gnrl
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = ['*']
        #-------------------------
        # In general, we want to cast the datetime column as a timestamp in SQL general so it won't need to be done in following CTEs/where statements/whatever
        # As such, build_sql_function_kwargs['datetime_col'] should be included in the cols_of_interest (except if date_only==True)
        new_dt_alias = build_sql_function_kwargs['datetime_col']
        if not date_only:
            new_dt_alias += '_ts'
            if build_sql_function_kwargs['datetime_col'] not in cols_of_interest:
                cols_of_interest.append(build_sql_function_kwargs['datetime_col'])
        #-------------------------
        # If join_mp_args, use df_args['mapping_to_mp'] to fully set up join_mp_args['build_mp_kwargs']
        # NOTE: If join_mp_args is False or an empty dict, no join will occur
        # NOTE: In df_args['mapping_to_mp'], the keys give the column in df_with_search_time_window from which to 
        #       extract the information, and the values contain the parameter to be set in join_mp_args['build_mp_kwargs'] 
        #       with the information extracted
        assert(Utilities.is_object_one_of_types(join_mp_args, [bool, dict]))
        if join_mp_args:
            if isinstance(join_mp_args, bool):
                join_mp_args = {}
            if 'build_mp_kwargs' not in join_mp_args:
                join_mp_args['build_mp_kwargs'] = {}
            #-----
            for df_col_mp, mp_field in df_args['mapping_to_mp'].items():
                join_mp_args['build_mp_kwargs'][mp_field] = df_with_search_time_window[df_col_mp].unique().tolist()
        else:
            join_mp_args = False
        #-------------------------
        # Build sql_gnrl
        sql_gnrl = build_sql_function(
            cols_of_interest = cols_of_interest, 
            join_mp_args     = join_mp_args, 
            **build_sql_function_kwargs
        )
        #-------------------------
        # As mentioned above, we want to cast the datetime column as a timestamp in SQL general
        if not date_only:
            # Could add simply orignal to end (in if cols_of_interest != ['*'] below), but I like things together, hence the find_idx.. call below
            found_idx = sql_gnrl.sql_select.find_idx_of_approx_element_in_collection_dict(
                element                  = build_sql_function_kwargs['datetime_col'], 
                assert_max_one           = True, 
                comp_alias               = False, 
                comp_table_alias_prefix  = False, 
                comp_comparison_operator = False, 
                comp_value               = False
            )
            assert(found_idx >= 0)
            #-------------------------
            sql_gnrl.cast_sql_select_element_as_timestamp(
                field_desc       = build_sql_function_kwargs['datetime_col'], 
                datetime_pattern = build_sql_function_kwargs['datetime_pattern'], 
                new_alias        = new_dt_alias
            )
            # Also want to include original datetime_col, as this is generally expected (in original form) in downstream methods
            if cols_of_interest != ['*']:
                sql_gnrl.sql_select.add_select_element(
                    field_desc         = build_sql_function_kwargs['datetime_col'], 
                    alias              = None, 
                    table_alias_prefix = None, 
                    idx                = found_idx+1, 
                    run_check          = False
                )
        #-------------------------
        # If joined with MP, grab sql_mp and sql_gnrl
        if isinstance(sql_gnrl, dict):
            assert(join_mp_args) # Should only happen when joining with MP
            assert('usg_sql' in sql_gnrl)
            sql_mp   = sql_gnrl['mp_sql']
            sql_gnrl = sql_gnrl['usg_sql']
        else: 
            sql_mp = None
        #-------------------------
        # Add where statements for all the date ranges in unique_date_ranges
        sql_gnrl_where = sql_gnrl.sql_where
        if sql_gnrl_where is None:
            sql_gnrl_where = SQLWhere()
        for unq_date_range in unique_date_ranges:
            sql_gnrl_where.add_where_statement(
                field_desc          = build_sql_function_kwargs['date_col'] , 
                comparison_operator = 'BETWEEN', 
                value               = [f'{unq_date_range[0]}', f'{unq_date_range[1]}'], 
                needs_quotes        = True, 
                table_alias_prefix  = build_sql_function_kwargs['from_table_alias']
            )
        sql_gnrl_where.combine_last_n_where_elements(last_n=len(unique_date_ranges), join_operator = 'OR')
        sql_gnrl.sql_where = sql_gnrl_where # Probably not necessary, but to be safe
        sql_gnrl.alias     = f'{sql_alias_base}gnrl'
        #--------------------------------------------------

        #--------------------------------------------------
        # Build the collection of SQL sub-queries for each group in df_gpd
        #-------------------------
        sql_i_coll = []
        count      = 0 # Count is used as appendix to give each sub-query a unique alias (=f'{sql_alias_base}{count}')
        map_strs   = []
        for addtnl_groupby_vals_w_t_search_min_max, sub_df in df_gpd:
            addtnl_groupby_vals = addtnl_groupby_vals_w_t_search_min_max[:-2]
            t_search_min        = addtnl_groupby_vals_w_t_search_min_max[-2]
            t_search_max        = addtnl_groupby_vals_w_t_search_min_max[-1]
            #----------        
            # # If df_args['mapping_to_ami'] is allowed to have multiple mappings
            # # code similar to that commented out below will be needed
            # ami_input = {}
            # for df_col, ami_field in df_args['mapping_to_ami'].items():
            #     assert(ami_field not in ami_input)
            #     if df_args['is_df_consolidated']:
            #         assert(sub_df.shape[0]==1)
            #         ami_field_values = sub_df.iloc[0][df_col]
            #         assert(Utilities.is_object_one_of_types(ami_field_values, [list, tuple]))
            #         ami_field_values = [x for x in ami_field_values if pd.notna(x)] # In test case, first entry had all NaN prem nbs
            #     else:
            #         ami_field_values = sub_df[df_col].unique().tolist()
            #     #-----
            #     if len(ami_field_values)==0:
            #         continue
            #     ami_input[ami_field] = ami_field_values
            # #----------
            # if not ami_input:
            #     continue
            # #----------
            # For now, only single mapping allowed between df and ami (due to desire for batch functionality)
            #   Therefore, enforce len(df_args['mapping_to_ami'])==1
            # Below, df_col is the column in df_with_search_time_window from which to extract the information
            #        ami_field is the parameter to be set in build_sql_function with the information extracted
            ami_input = {}
            # Using df_col, extract the infrmation (i.e., values) from sub_df
            if df_args['is_df_consolidated']:
                assert(sub_df.shape[0]==1)
                ami_field_values = sub_df.iloc[0][df_col]
                #-----
                assert(Utilities.is_object_one_of_types(ami_field_values, [list, tuple]))
                ami_field_values = [x for x in ami_field_values if pd.notna(x)] # In test case, first entry had all NaN prem nbs
                #-----
                consolidated_col = df_col
                #-----
                if len(ami_field_values)==0:
                    continue
            else:
                ami_field_values = sub_df[df_col].unique().tolist()
                consolidated_col = None
            assert(ami_field not in ami_input)
            ami_input[ami_field] = ami_field_values
            #-------------------------
            if len(df_args['addtnl_groupby_cols']) > 0:
                map_str_gp_i = AMI_SQL.create_mapping_table_rows(
                    df                  = sub_df, 
                    join_cols           = [
                        mapping_to_ami_i.df_col, 
                        df_args['t_search_min_col'], 
                        df_args['t_search_max_col']
                    ], 
                    mapped_cols         = df_args['addtnl_groupby_cols'],
                    timestamp_cols      = [
                        df_args['t_search_min_col'], 
                        df_args['t_search_max_col']
                    ], 
                    consolidated_col    = consolidated_col, 
                    include_final_comma = True
                )
                map_strs.append(map_str_gp_i)
            #--------------------------------------------------
            # Enforce max_n_prem_per_outg, running by batches if the collection size exceeds max_n
            # TODO To allows for more mappings, maybe have max_n be a dict?
            if max_n_prem_per_outg is not None and len(ami_field_values)>max_n_prem_per_outg:
                batch_idxs = Utilities.get_batch_idx_pairs(len(ami_field_values), max_n_prem_per_outg)
            else:
                batch_idxs = [[0, len(ami_field_values)]]
            #-------------------------
            # Generate a sub-query for each batch in batch_idxs (or, I guess one could think of it as a sub-sub-query as
            #   the query for this group is split into multiple when len(batch_idxs)>1)
            for batch_i in batch_idxs:
                i_beg       = batch_i[0]
                i_end       = batch_i[1]
                ami_input_i = {ami_field:ami_field_values[i_beg:i_end]}
                # TODO COULD DO: ami_input_i = {k:v[i_beg:i_end] for k,v in ami_input.items()}
                #-----
                # NOTE: the datetime casting was taken care of in sql_gnrl, so datetime_pattern should be set to 'no_cast' and datetime_col should
                #   be set to the casted column
                # NOTE: After sql_general built, turn off verbose (otherwise, e.g., "No OPCOs SELECTED" warning would hit for each sub-query)
                kwargs_i = dict(
                    cols_of_interest = [f"*"], 
                    datetime_range   = [t_search_min, t_search_max], 
                    datetime_pattern = 'no_cast', 
                    datetime_col     = new_dt_alias, 
                    table_name       = sql_gnrl.alias, 
                    schema_name      = None, 
                    from_table_alias = 'gnrl', 
                    verbose          = False
                )
                if date_only:
                    del kwargs_i['datetime_range']
                    kwargs_i['date_range'] = [
                        pd.to_datetime(t_search_min).date(), 
                        pd.to_datetime(t_search_max).date()
                    ]
                kwargs_i = kwargs_i|ami_input_i
                sql_i    = build_sql_function(**kwargs_i)
                #----------
                sql_i_coll.append(sql_i)
                count += 1

        #--------------------------------------------------
        # Grab the WHERE statements from sql_i_coll and combine with OR statements
        combined_where_stmnts = [
            CombinedSQLWhereElements(
                collection_dict = sql_i.sql_where.collection_dict, 
                idxs_to_combine = list(range(len(sql_i.sql_where.collection_dict))), 
                join_operator   = 'AND'
            ).get_combined_where_elements_string()
            for sql_i in sql_i_coll
        ]
        #-----
        combined_where_stmnts = Utilities.join_list(
            list_to_join  = combined_where_stmnts, 
            quotes_needed = False, 
            join_str      = '\nOR\n'
        )
        #-----
        combined_where_stmnts = Utilities.prepend_tabs_to_each_line(
            input_statement   = combined_where_stmnts, 
            n_tabs_to_prepend = 1
        )
        #-----
        combined_where_stmnts = 'WHERE\n' + combined_where_stmnts

        #--------------------------------------------------
        # Build the final SQL query elements
        sql_from_fnl   = SQLFrom(
            schema_name = None, 
            table_name  = sql_gnrl.alias, 
            alias       = 'gnrl'
        )
        #-------------------------
        # NOTE: The original method effective removed duplicate rows through the use of UNION (instead of UNION ALL) to join the CTEs
        #       For this method to remove duplicates, I utilize the select_distinct argument
        # NOTE: These duplicates seem to not result from the code or any type of joining etc. on my end, but seem to reflect actual
        #         duplicates within the data themselves.
        if len(df_args['addtnl_groupby_cols'])==0:
            sql_select_fnl = SQLSelect(
                field_descs     = ['gnrl.*'], 
                select_distinct = True
            )
        else:
            mapped_tag = '_GPD_FOR_SQL'
            #-----
            sql_select_fnl = SQLSelect(
                field_descs     = ['gnrl.*'] + [f"map.{x}{mapped_tag}" for x in df_args['addtnl_groupby_cols']], 
                select_distinct = True
            )
            #-------------------------
            # Finalize the mapping table which allows us to include addtnl_groupby_cols in the final output
            mapping_table = AMI_SQL.finalize_mapping_table(
                table_rows  = map_strs, 
                join_cols   = [
                    mapping_to_ami_i.df_col, 
                    df_args['t_search_min_col'], 
                    df_args['t_search_max_col']
                ], 
                mapped_cols  = df_args['addtnl_groupby_cols'],
                alias        = 'mapping_table', 
                include_with = False, 
                mapped_tag   = mapped_tag
            )
            #-------------------------
            sql_join_fnl = SQLJoin(
                join_type               = 'LEFT', 
                join_table              = 'mapping_table', 
                join_table_alias        = 'map', 
                orig_table_alias        = 'gnrl', 
                list_of_columns_to_join = [
                    [mapping_to_ami_i.sql_col, mapping_to_ami_i.df_col]
                ]
            )
            #-----
            sql_join_fnl_stmnt = sql_join_fnl.get_statement_string()
            #-----
            sql_join_fnl_stmnt += "\nAND gnrl.{} BETWEEN map.{} and map.{}".format(
                build_sql_function_kwargs['date_col'] if date_only else new_dt_alias, 
                df_args['t_search_min_col'], 
                df_args['t_search_max_col']
            )
        #-------------------------
        sql_gnrl_stmnt = sql_gnrl.get_sql_statement(
            insert_n_tabs_to_each_line = 0,
            include_alias              = True,
            include_with               = True,
        )
        #-------------------------
        sql_final = SQLQuery(
            sql_select    = sql_select_fnl, 
            sql_from      = sql_from_fnl, 
            sql_where     = None, 
        )
        #-------------------------
        if len(df_args['addtnl_groupby_cols'])==0:
            sql_final_stmnt = (
                sql_gnrl_stmnt +
                '\n'          +
                sql_final.get_sql_statement() +
                '\n' +
                combined_where_stmnts 
            )
        else:
            sql_final_stmnt = (
                sql_gnrl_stmnt +
                ',\n'          +
                mapping_table  +
                '\n'           +
                sql_final.get_sql_statement() +
                '\n' +
                sql_join_fnl_stmnt + 
                '\n' +
                combined_where_stmnts 
            )
        #-------------------------
        return sql_final_stmnt
    
    
    #****************************************************************************************************
    @staticmethod
    def build_sql_ami_for_outages(
        cols_of_interest          , 
        df_outage                 , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        date_only                 = False, 
        df_args                   = {}
    ):
        return AMI_SQL.build_sql_ami_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_outage, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            df_args                    = df_args
        )

    #****************************************************************************************************
    @staticmethod
    def build_sql_ami_for_no_outages(
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
        build_sql_function allows this to be used for e.g. build_sql_usg_for_outages, build_sql_end_events_for_outages,
                           build_sql_usg_inst_for_outages, etc.
                           NOTE: python won't let me set the default value to build_sql_function=AMI_SQL.build_sql_ami.
                                 Thus, the workaround is to set the default to None, then set it to 
                                 AMINonVeeSQL.build_sql_usg if it is None

        TODO: Investigate what an appropriate default value for max_n_prem_per_outg would be....
              Probably something large, like 1000.  But, should find outages with huge numbers of customers
              affected to find the right number.
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

        return AMI_SQL.build_sql_ami_for_df_with_search_time_window(
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