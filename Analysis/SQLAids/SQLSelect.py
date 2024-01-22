#!/usr/bin/env python

import sys, os
from SQLElement import SQLElement
from SQLElementsCollection import SQLElementsCollection

class SQLSelectElement(SQLElement):
    def __init__(self, field_desc, 
                 alias=None, table_alias_prefix=None, is_agg=False):
        # TODO for now, dict_init not set up
        assert(isinstance(field_desc,str))
        # First, call base class's __init__ method
        super().__init__(field_desc=field_desc, 
                         alias=alias, table_alias_prefix=table_alias_prefix, 
                         comparison_operator=None, value=None)
        self.is_agg = is_agg
						 
						 
class SQLSelect(SQLElementsCollection):
    def __init__(self, 
                 field_descs=None, 
                 global_table_alias_prefix=None, 
                 idxs=None, run_check=False, 
                 select_distinct=False):
        # First, call base class's __init__ method
        super().__init__(field_descs=field_descs, 
                         global_table_alias_prefix=global_table_alias_prefix, 
                         idxs=idxs, run_check=run_check, SQLElementType=SQLSelectElement)
        self.select_distinct=select_distinct
        
    def add_select_element(self, field_desc, alias=None, table_alias_prefix=None, 
                           idx=None, run_check=False):
        if isinstance(field_desc, SQLSelectElement):
            element = field_desc
        else:
            element = SQLSelectElement(field_desc=field_desc, alias=alias, table_alias_prefix=table_alias_prefix)
        self.insert_single_element_to_collection_at_idx(element=element, idx=idx, run_check=run_check)
        
    def add_select_elements(self, field_descs, 
                            global_table_alias_prefix=None, 
                            idxs=None, run_check=False):
        # field_descs should be a list with elements of type SQLElement, dict, or string.
        # See SQLElementsCollection.insert_to_collection_at_idx for more information
        self.insert_to_collection_at_idx(field_descs=field_descs, 
                                         global_table_alias_prefix=global_table_alias_prefix, 
                                         idxs=idxs, run_check=run_check)
        
    def get_statement_string(self, include_alias=True, include_table_alias_prefix=True):
        # If not last line ==> end with comma (',')
        # else             ==> end with nothing
        #---------------
        _ = self.check_collection_dict(enforce_pass=True)
        #---------------
        sql = "SELECT"
        if self.select_distinct:
            sql += ' DISTINCT'
        self.build_collection_list()
        for idx,element in enumerate(self.collection):
            line = f"\n\t{element.get_field_desc(include_alias=include_alias, include_table_alias_prefix=include_table_alias_prefix)}"
            if idx<len(self.collection)-1:
                line += ','
            sql += line
        return sql
        
    def print_statement_string(self, include_alias=True, include_table_alias_prefix=True):
        stmnt_str = self.get_statement_string(include_alias=include_alias, 
                                              include_table_alias_prefix=include_table_alias_prefix)
        print(stmnt_str)
        
    def print(self, include_alias=True, include_table_alias_prefix=True):
        self.print_statement_string(include_alias=include_alias, 
                                    include_table_alias_prefix=include_table_alias_prefix)
        
    def get_agg_element_ids(self):
        agg_elements = []
        for idx,sql_el in self.collection_dict.items():
            if sql_el.is_agg:
                agg_elements.append(idx)
        agg_elements = sorted(agg_elements)   
        return agg_elements
        
        
    @staticmethod
    def refine_agg_cols_and_types(agg_cols_and_types, try_to_split_col_strs=True):
        r"""
        Changes any string keys to SQLElement keys in agg_cols_and_types and returns.
        
        - This should modify the agg_cols_and_types dict in place, but still is probably safest to
            call agg_cols_and_types  = refine_agg_cols_and_types(agg_cols_and_types, try_to_split_col_strs)
        - The input keys of agg_cols_and_types may be of type SQLElement or str (when simple column names).    
        - agg_cols_and_types is a dictionary with:
           keys:
               equal to column names OR SQLElement objects (representing column names) to be aggregated
           values:
               each value should be equal to a list of aggregations to perform on column
               At this time, the available aggregate functions are:
                 'sum', 'sq_sum', 'mean', 'std', 'count'
               NOTE: If more aggregate functions are added, to_SQL_dict must be updated appropriately
        
        
        - try_to_split_col_strs:
            - when True and a key of type str is found, this will attempt to split the column name into 
                field_desc and table_alias_prefix components, which will then be used when creating the 
                SQLElement replacement key.
            - e.g. 'U.value' --> field_desc='value' and table_alias_prefix='U'
        
        """
        # NOTE: Cannot alter dict when iterating over dict
        #       Therefore, one must iterate over a list of the keys
        #         i.e. for col in list(agg_cols_and_types.keys())
        #       Even if one defines keys = agg_cols_and_types.keys() beforehand,
        #         and tries for col in keys, this will not work!
        for col in list(agg_cols_and_types.keys()):
            assert(isinstance(col, str) or isinstance(col, SQLElement))
            if isinstance(col, str):
                if try_to_split_col_strs:
                    components_dict = SQLElement.split_field_desc(col)
                    field_desc         = components_dict['field_desc']
                    table_alias_prefix = components_dict['table_alias_prefix']
                else:
                    field_desc=col
                    table_alias_prefix=None
                sql_el = SQLElement(field_desc=field_desc, 
                                    table_alias_prefix=table_alias_prefix)
                assert(sql_el not in agg_cols_and_types)
                agg_cols_and_types[sql_el] = agg_cols_and_types[col]
                del agg_cols_and_types[col]
        return agg_cols_and_types   
    
    
    @staticmethod
    def add_aggregate_elements_to_sql_select(sql_select, agg_cols_and_types, 
                                             try_to_split_col_strs=True, 
                                             include_counts_including_null=True, **kwargs):
        r"""
        NOTE: Non-static verson = add_aggregate_elements
        
        - agg_cols_and_types_dict:
           keys:
               equal to column names OR SQLElement objects (representing column names) to be aggregated
           values:
               each value should be equal to a list of aggregations to perform on column
               At this time, the available aggregate functions are:
                 'sum', 'sq_sum', 'mean', 'std', 'count'
               NOTE: If more aggregate functions are added, to_SQL_dict must be updated appropriately
        
        - try_to_split_col_strs:
            - when True and a key of type str is found, this will attempt to split the column name into 
                field_desc and table_alias_prefix components, which will then be used when creating the 
                SQLElement replacement key.
            - e.g. 'U.value' --> field_desc='value' and table_alias_prefix='U'
        
        """
        #---------------------
        to_SQL_dict = {'sum':'SUM({})', 
                       'sq_sum':'SUM(POWER({}, 2))', 
                       'mean':'AVG({})', 
                       'std':'STDDEV_SAMP({})', 
                       'count':'COUNT({})', 
                       'min':'MIN({})', 
                       'max':'MIN({})'}
        #---------------------
        # Make all keys in agg_cols_and_types type SQLElement
        agg_cols_and_types = SQLSelect.refine_agg_cols_and_types(agg_cols_and_types, try_to_split_col_strs)
        #---------------------
        # If any of the aggregate columns (which, at this point, are all SQLElement objects) are found
        # in the sql_select, remove them.
        comp_alias = kwargs.get('comp_alias', False)
        comp_table_alias_prefix = kwargs.get('comp_table_alias_prefix', False)
        for sql_elm in agg_cols_and_types.keys():
            found_idx = sql_select.find_idx_of_approx_element_in_collection_dict(sql_elm, 
                                                                                 comp_alias=comp_alias, 
                                                                                 comp_table_alias_prefix=comp_table_alias_prefix, 
                                                                                 assert_max_one=True)
            if found_idx > -1:
                sql_select.remove_single_element_from_collection_at_idx(found_idx)
        #---------------------
        # Get new agg_cols with aliases
        agg_sql_elements = [] 
        for col_el,agg_types in agg_cols_and_types.items():
            for agg_type in agg_types:
                field_desc_i = to_SQL_dict[agg_type].format(col_el.get_field_desc(include_table_alias_prefix=True))
                #alias_i = f"{col_el.get_field_desc(include_table_alias_prefix=False)}_{agg_type}"
                alias_i = f"{agg_type}_{col_el.get_field_desc(include_table_alias_prefix=False)}"
                sql_el_i = SQLSelectElement(field_desc=field_desc_i, alias=alias_i, is_agg=True)
                assert(sql_el_i not in agg_sql_elements)
                agg_sql_elements.append(sql_el_i)
        if include_counts_including_null:
            agg_sql_elements.append(SQLSelectElement(field_desc='COUNT(*)', alias='counts_including_null', is_agg=True))
        #---------------------
        # Add new agg_cols with aliases (stored now in SQLSelectElement objects) to sql_select
        sql_select.add_select_elements(agg_sql_elements, run_check=True)
        #---------------------
        return sql_select    
        

    def add_aggregate_elements(self, agg_cols_and_types, 
                               try_to_split_col_strs=True, 
                               include_counts_including_null=True, **kwargs):
        SQLSelect.add_aggregate_elements_to_sql_select(self, 
                                                       agg_cols_and_types=agg_cols_and_types, 
                                                       try_to_split_col_strs=try_to_split_col_strs, 
                                                       include_counts_including_null=include_counts_including_null, 
                                                       **kwargs)
    
    @staticmethod
    def build_aggregate_sql_select(field_descs, agg_cols_and_types, 
                                   try_to_split_col_strs=True, 
                                   global_table_alias_prefix=None, idxs=None, run_check=False, 
                                   include_counts_including_null=True, 
                                   **kwargs):
        r"""
        - field_descs, global_table_alias_prefix, idxs, run_check:
            - These are just as should be input into SQLSelect            
            - field_descs should be a list with elements of type dict or string.
                - If element is a dict, the keys should contain at a minimum 'field_desc'
                    - Full possible set of keys = ['field_desc', 'alias', 'table_alias_prefix', 
                                                   'comparison_operator', 'value']
                - If element is a string, the string should be a field_desc.
                    - All elements of type string will have:
                        alias=comparison_operator=value=None
                        table_alias_prefix = global_table_alias_prefix
            - global_table_alias_prefix
                - If global_table_alias_prefix is not None, all elements without an included table_alias_prefix with have their
                    value set equal to global_table_alias_prefix.
                - NOTE: If one explicitly sets table_alias_prefix to None, global_table_alias_prefix will not change this!
                    
        - agg_cols_and_types_dict:
           keys:
               equal to column names OR SQLElement objects (representing column names) to be aggregated
           values:
               each value should be equal to a list of aggregations to perform on column
               At this time, the available aggregate functions are:
                 'sum', 'sq_sum', 'mean', 'std', 'count'
               NOTE: If more aggregate functions are added, to_SQL_dict must be updated appropriately
        
        - try_to_split_col_strs:
            - when True and a key of type str is found, this will attempt to split the column name into 
                field_desc and table_alias_prefix components, which will then be used when creating the 
                SQLElement replacement key.
            - e.g. 'U.value' --> field_desc='value' and table_alias_prefix='U'
        """

        #---------------------
        sql_select = SQLSelect(field_descs=field_descs, 
                               global_table_alias_prefix=global_table_alias_prefix, 
                               idxs=idxs, run_check=run_check)
        sql_select = SQLSelect.add_aggregate_elements_to_sql_select(sql_select=sql_select, 
                                                                    agg_cols_and_types=agg_cols_and_types, 
                                                                    try_to_split_col_strs=try_to_split_col_strs, 
                                                                    include_counts_including_null=include_counts_including_null, 
                                                                    **kwargs) 
        return sql_select