#!/usr/bin/env python

import sys, os
from SQLElement import SQLElement
from SQLElementsCollection import SQLElementsCollection

class SQLOrderByElement(SQLElement):
    def __init__(self, field_desc, 
                 table_alias_prefix=None, descending=False):      
        # TODO for now, dict_init not set up
        assert(isinstance(field_desc,str))
        # First, call base class's __init__ method
        super().__init__(field_desc=field_desc, 
                 alias=None, table_alias_prefix=table_alias_prefix, 
                 comparison_operator=None, value=None)
        self.descending = descending
         
    def get_orderby_element_string(self, include_table_alias_prefix=True):
        field_desc = self.get_field_desc(include_alias=False, include_table_alias_prefix=include_table_alias_prefix)
        descending = self.descending
        asc_desc_strings = ['ASC', 'DESC']
        if descending:
            elmnt_strng = f"{field_desc} {asc_desc_strings[descending]}"
        else:
            elmnt_strng = f"{field_desc}"
        return elmnt_strng
		
		
		
class SQLOrderBy(SQLElementsCollection):
    def __init__(self, 
                 field_descs=None, 
                 global_table_alias_prefix=None, 
                 idxs=None, run_check=False):
        # First, call base class's __init__ method
        super().__init__(field_descs=field_descs, 
                         global_table_alias_prefix=global_table_alias_prefix, 
                         idxs=idxs, run_check=run_check, SQLElementType=SQLOrderByElement)
        
    def add_orderby_statement(self, field_desc, table_alias_prefix=None, 
                              idx=None, run_check=False):
        element = SQLOrderByElement(field_desc=field_desc, alias=None, table_alias_prefix=table_alias_prefix)
        self.insert_single_element_to_collection_at_idx(element=element, idx=idx, run_check=run_check)
        
    def add_orderby_statements(self, field_descs, 
                               global_table_alias_prefix=None, 
                               idxs=None, run_check=False):
        # field_descs should be a list with elements of type SQLElement, dict, or string.
        # See SQLElementsCollection.insert_to_collection_at_idx for more information
        self.insert_to_collection_at_idx(field_descs=field_descs, 
                                         global_table_alias_prefix=global_table_alias_prefix, 
                                         idxs=idxs, run_check=run_check)
        
        
    def get_statement_string(self, include_table_alias_prefix=True):
        # If not last line ==> end with comma (',')
        # else             ==> end with nothing
        #---------------
        if len(self.collection_dict)==0:
            return ''
        #---------------
        _ = self.check_collection_dict(enforce_pass=True)
        #---------------
        sql = "ORDER BY"
        self.build_collection_list()
        for idx,element in enumerate(self.collection):
            line = f"\n\t{element.get_orderby_element_string(include_table_alias_prefix=include_table_alias_prefix)}"
            #line = f"\n\t{element.get_field_desc(include_alias=False, include_table_alias_prefix=include_table_alias_prefix)}"
            if idx<len(self.collection)-1:
                line += ','
            sql += line
        return sql
        
    def print_statement_string(self, include_table_alias_prefix=True):
        stmnt_str = self.get_statement_string(include_table_alias_prefix=include_table_alias_prefix)
        print(stmnt_str)
        
    def print(self, include_table_alias_prefix=True):
        self.print_statement_string(include_table_alias_prefix=include_table_alias_prefix)