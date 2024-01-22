#!/usr/bin/env python

import sys, os
import Utilities_sql
from SQLElement import SQLElement
from SQLElementsCollection import SQLElementsCollection
from SQLSelect import SQLSelectElement, SQLSelect
from SQLFrom import SQLFrom
from SQLWhere import SQLWhereElement, SQLWhere
from SQLJoin import SQLJoin, SQLJoinCollection
from SQLGroupBy import SQLGroupByElement, SQLGroupBy
from SQLHaving import SQLHaving
from SQLOrderBy import SQLOrderByElement, SQLOrderBy

        
class SQLQuery:
    def __init__(self, sql_select, sql_from, sql_where, 
                 sql_groupby=None, sql_having=None, sql_orderby=None, sql_join_coll=None, alias=None):
        self.set_sql_select(sql_select)
        self.set_sql_from(sql_from, sql_join_coll)
        self.set_sql_where(sql_where)
        self.set_sql_groupby(sql_groupby)
        self.set_sql_having(sql_having)
        self.set_sql_orderby(sql_orderby)
        self.alias = alias
        self.common_table_expressions = None
        self.limit = None
        
    def set_limit(self, limit):
        self.limit = limit
        
    def set_sql_select(self, sql_select):
        assert(isinstance(sql_select, SQLSelect))
        self.sql_select = sql_select
        
    def set_sql_from(self, sql_from, sql_join_coll=None):
        assert(isinstance(sql_from, SQLFrom))
        self.sql_from = sql_from
        if sql_join_coll is not None:
            self.sql_from.set_sql_join_coll(sql_join_coll)
            
    def add_join(self, sql_join, 
                 idx=None, run_check=False, join_cols_to_add_to_select=None):
        assert(isinstance(sql_join, SQLJoin))
        self.sql_from.add_join_to_coll(sql_join=sql_join, idx=idx, run_check=run_check)
        #-------------------------
        if join_cols_to_add_to_select is not None:
            self.sql_select.add_select_elements(field_descs=join_cols_to_add_to_select, 
                                                global_table_alias_prefix=sql_join.join_table_alias, 
                                                idxs=None, 
                                                run_check=run_check)
        
    def build_and_add_join(self, join_type, join_table, join_table_alias, orig_table_alias, 
                           list_of_columns_to_join, 
                           idx=None, run_check=False, join_cols_to_add_to_select=None):
        self.sql_from.build_and_add_join_to_coll(join_type=join_type, 
                                                 join_table=join_table, 
                                                 join_table_alias=join_table_alias, 
                                                 orig_table_alias=orig_table_alias, 
                                                 list_of_columns_to_join=list_of_columns_to_join, 
                                                 idx=idx, 
                                                 run_check=run_check)
        #-------------------------
        if join_cols_to_add_to_select is not None:
            self.sql_select.add_select_elements(field_descs=join_cols_to_add_to_select, 
                                                global_table_alias_prefix=join_table_alias, 
                                                idxs=None, 
                                                run_check=run_check)
        
    def set_sql_where(self, sql_where):
        if sql_where is None:
            self.sql_where = SQLWhere()
        else:
            assert(isinstance(sql_where, SQLWhere))
            self.sql_where = sql_where
        
    def set_sql_groupby(self, sql_groupby):
        if sql_groupby is None:
            self.sql_groupby = SQLGroupBy()
        else:
            assert(isinstance(sql_groupby, SQLGroupBy))
            self.sql_groupby = sql_groupby
            
    def set_sql_having(self, sql_having):
        if sql_having is None:
            self.sql_having = SQLHaving()
        else:
            assert(isinstance(sql_having, SQLHaving))
            self.sql_having = sql_having
        
    def set_sql_orderby(self, sql_orderby):
        if sql_orderby is None:
            self.sql_orderby = SQLOrderBy()
        else:
            assert(isinstance(sql_orderby, SQLOrderBy))
            self.sql_orderby = sql_orderby
        
    def get_sql_statement(self, insert_n_tabs_to_each_line=0, include_alias=False):
        r"""
        - If include_alias==True, the alias attribute will be used to return a common table expression of
          the form: 
            f"{alias} AS (\n{***SQL STATEMENT***}\n)"
        - If include_alias==False, the sql statement is returned as normal
        """
        select_stmnt = self.sql_select.get_statement_string(include_alias=True, include_table_alias_prefix=True)
        from_stmnt = self.sql_from.get_statement_string(include_leading_join_whitespace='\t')
        where_stmnt = self.sql_where.get_statement_string(include_table_alias_prefix=True)
        groupby_stmnt = self.sql_groupby.get_statement_string(include_table_alias_prefix=True)
        having_stmnt = self.sql_having.get_statement_string(include_table_alias_prefix=True)
        orderby_stmnt = self.sql_orderby.get_statement_string(include_table_alias_prefix=True)
        #-----
        return_stmnt =  SQLQuery.combine_sql_components(select_stmnt=select_stmnt, 
                                                        from_and_join_stmnts=from_stmnt, 
                                                        where_stmnt=where_stmnt, 
                                                        groupby_stmnt=groupby_stmnt, 
                                                        having_stmnt=having_stmnt, 
                                                        orderby_stmnt=orderby_stmnt, 
                                                        limit=self.limit)
        #-----
        if include_alias:
            assert(self.alias)
            return_stmnt = Utilities_sql.prepend_tabs_to_each_line(return_stmnt, n_tabs_to_prepend=1)
            return_stmnt = f"{self.alias} AS (\n{return_stmnt}\n)"
        #-----
        if insert_n_tabs_to_each_line > 0:
            return_stmnt = Utilities_sql.prepend_tabs_to_each_line(return_stmnt, 
                                                                   n_tabs_to_prepend=insert_n_tabs_to_each_line)
        #-----
        return return_stmnt
        
    def print_sql_statement(self, insert_n_tabs_to_each_line=0, include_alias=False):
        sql_statement = self.get_sql_statement(insert_n_tabs_to_each_line=insert_n_tabs_to_each_line, 
                                               include_alias=include_alias)
        print(sql_statement)
        
    def print(self, insert_n_tabs_to_each_line=0, include_alias=False):
        self.print_sql_statement(insert_n_tabs_to_each_line=insert_n_tabs_to_each_line, 
                                 include_alias=include_alias)
        
    @staticmethod
    def combine_sql_components(select_stmnt, from_and_join_stmnts, where_stmnt, 
                               groupby_stmnt='', having_stmnt='', orderby_stmnt='', limit=None):
        sql = f"{select_stmnt}\n{from_and_join_stmnts}"
        if where_stmnt:
            sql = f"{sql}\n{where_stmnt}"
        if groupby_stmnt:
            sql = f"{sql}\n{groupby_stmnt}"
        if having_stmnt:
            assert(groupby_stmnt)
            sql = f"{sql}\n{having_stmnt}"
        if orderby_stmnt:
            sql = f"{sql}\n{orderby_stmnt}"
        if limit:
            sql = f"{sql}\nLIMIT {limit}"
        return sql