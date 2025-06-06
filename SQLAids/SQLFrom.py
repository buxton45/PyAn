#!/usr/bin/env python

import sys, os
from SQLJoin import SQLJoin, SQLJoinCollection

class SQLFrom:
    def __init__(self, schema_name=None, table_name=None, alias=None, sql_join_coll=None):
        self.schema_name = schema_name
        self.table_name = table_name
        self.alias = alias
        self.from_statement = SQLFrom.build_from_statement(self.schema_name, self.table_name, self.alias)
        if sql_join_coll is None:
            self.sql_join_coll = SQLJoinCollection()
        else:
            self.set_sql_join_coll(sql_join_coll)
            
    @staticmethod
    def build_from_statement(schema_name, table_name, alias=None):
        if alias is None:
            alias = ''
        else:
            alias = f" {alias}"
        if table_name is None:
            from_stmnt = f"FROM {schema_name}{alias}"
        elif schema_name is None:
            from_stmnt = f"FROM {table_name}{alias}"
        else:
            from_stmnt = f"FROM {schema_name}.{table_name}{alias}"
        return from_stmnt  
        
    def get_from_statement(self):
        return SQLFrom.build_from_statement(self.schema_name, self.table_name, self.alias)
            
    def set_sql_join_coll(self, sql_join_coll):
        if(isinstance(sql_join_coll, SQLJoin)):
            self.sql_join_coll = SQLJoinCollection()
            SQLJoinCollection.insert_to_joins_dict_at_idx(sql_join_coll, self.sql_join_coll.joins_dict, idx=None, run_check=False)            
        else:
            assert(isinstance(sql_join_coll, SQLJoinCollection))
            self.sql_join_coll = sql_join_coll
        
    def add_join_to_coll(self, sql_join, 
                         idx=None, run_check=False):
        assert(isinstance(sql_join, SQLJoin))
        if self.sql_join_coll is None:
            self.sql_join_coll = SQLJoinCollection()
        self.sql_join_coll.add_join_to_coll(sql_join=sql_join, idx=idx, run_check=run_check)
        
    def build_and_add_join_to_coll(self, join_type, join_table, join_table_alias, orig_table_alias, list_of_columns_to_join, 
                                   idx=None, run_check=False):
        if self.sql_join_coll is None:
            self.sql_join_coll = SQLJoinCollection()
        self.sql_join_coll.build_and_add_join_to_coll(join_type=join_type, join_table=join_table, 
                                                      join_table_alias=join_table_alias, orig_table_alias=orig_table_alias, 
                                                      list_of_columns_to_join=list_of_columns_to_join, 
                                                      idx=idx, run_check=run_check)

    def get_join_statement(self, include_leading_whitespace=None):
        return self.sql_join_coll.get_statement_string(include_leading_whitespace=include_leading_whitespace)
                   
        
    @staticmethod
    def combine_from_and_join_statement_strings(sql_from, sql_join_coll=None, include_leading_join_whitespace=None):
        # If sql_join_coll is None, use sql_join_coll from sql_from object
        if sql_join_coll is None:
            sql_join_coll = sql_from.sql_join_coll
        if not sql_join_coll.is_empty():
            return f"{sql_from.get_from_statement()}\n{sql_join_coll.get_statement_string(include_leading_whitespace=include_leading_join_whitespace)}"
        else:
            return sql_from.get_from_statement()
        
    def get_statement_string(self, include_leading_join_whitespace=None):
        return SQLFrom.combine_from_and_join_statement_strings(self, include_leading_join_whitespace=include_leading_join_whitespace)
        
    def print_statement_string(self, include_leading_join_whitespace=None):
        stmnt_str = self.get_statement_string(include_leading_join_whitespace=include_leading_join_whitespace)
        print(stmnt_str)
        
    def print(self, include_leading_join_whitespace=None):
        self.print_statement_string(include_leading_join_whitespace=include_leading_join_whitespace)
        
