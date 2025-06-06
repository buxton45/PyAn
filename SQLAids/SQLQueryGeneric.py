#!/usr/bin/env python

import sys, os
import Utilities_sql

        
class SQLQueryGeneric:
    def __init__(
        self          , 
        sql_statement , 
        alias         = None
    ):
        self.sql_statement = sql_statement
        self.alias         = alias

        
    def get_sql_statement(
        self                       , 
        insert_n_tabs_to_each_line = 0, 
        include_alias              = False, 
        include_with               = False
    ):
        #-------------------------
        if not include_alias:
            include_with = False
        #-------------------------
        return_stmnt = self.sql_statement
        #-----
        if include_alias:
            assert(self.alias)
            return_stmnt = Utilities_sql.prepend_tabs_to_each_line(return_stmnt, n_tabs_to_prepend=1)
            return_stmnt = f"{self.alias} AS (\n{return_stmnt}\n)"
        #-----
        if include_with:
            return_stmnt = f"WITH {return_stmnt}"
        #-----
        if insert_n_tabs_to_each_line > 0:
            join_str = '\n' + insert_n_tabs_to_each_line*'\t'
            return_stmnt = join_str.join(return_stmnt.splitlines())
            return_stmnt = '\t' + return_stmnt
        #-----
        return return_stmnt
        
    def print_sql_statement(
        self                       , 
        insert_n_tabs_to_each_line = 0
    ):
        sql_statement = self.get_sql_statement(insert_n_tabs_to_each_line=insert_n_tabs_to_each_line)
        print(sql_statement)