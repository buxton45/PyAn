#!/usr/bin/env python

import sys, os
from SQLElement import SQLElement
from SQLElementsCollection import SQLElementsCollection
from SQLWhere import SQLWhereElement, SQLWhere

#****************************************************************************************************************************************************        
class SQLHaving(SQLWhere):
    def __init__(self, 
                 field_descs=None, 
                 idxs=None, run_check=False):
        # First, call base class's __init__ method
        super().__init__(field_descs=field_descs, 
                         idxs=idxs, run_check=run_check)

    #------------------------------------------------------------------------------------------------
    def get_statement_string(self, include_table_alias_prefix=True):
        # If first line ==> begin with HAVING
        # else          ==> being with AND
        #
        # If not last line ==> end with comma (',')
        # else             ==> end with nothing
        #---------------
        _ = self.check_collection_dict(enforce_pass=True)
        #---------------
        sql = ""
        self.build_collection_list()
        for idx,element in enumerate(self.collection):
            assert(isinstance(element, SQLWhereElement) or isinstance(element, CombinedSQLWhereElements))
            line = ""
            if idx==0:
                line = 'HAVING '
            else:
                line = '\nAND   '
            if isinstance(element, SQLWhereElement):
                line += f"{element.get_where_element_string(include_table_alias_prefix=include_table_alias_prefix)}"
            else:
                line += element.get_combined_where_elements_string(include_table_alias_prefix=include_table_alias_prefix) 
            sql += line
        return sql