#!/usr/bin/env python

import sys, os
import numpy as np

class SQLJoin:
    def __init__(
        self                    , 
        join_type               , 
        join_table              , 
        join_table_alias        , 
        orig_table_alias        , 
        list_of_columns_to_join ,
    ):
        self.join_type               = join_type
        self.join_table              = join_table
        self.join_table_alias        = join_table_alias
        self.orig_table_alias        = orig_table_alias
        self.list_of_columns_to_join = list_of_columns_to_join
        #-----
        self.join_statement = self.get_statement_string()
    
    
    @staticmethod 
    def build_list_of_columns_to_join_dicts(list_of_columns_to_join):
        r"""
        Converts list_of_columns_to_join to list_of_columns_to_join_dicts intended to be used in build_join_statement.
        Returned list_of_columns_to_join_dicts is a list of dicts where each dict item has keys = ['orig', 'join', 'needs_prefixes'].
        This list is for joining one 'join' table to one 'orig' table

        list_of_columns_to_join must always be a list (or tuple).
        However, the elements of the list can take many different forms.  Consider list_of_columns_to_join[i]=cols_to_join.
          cols_to_join can be of the following forms:
            cols_to_join = str (where str must be the name of a column found in both tables to be joined)
              ==> cols_to_join = {'orig':cols_to_join, 'join':cols_to_join, 'needs_prefixes':True}
            cols_to_join = list
              cols_to_join = [str]            ==> cols_to_join = {'orig':cols_to_join[0], 'join':cols_to_join[0], 'needs_prefixes':True}
              cols_to_join = [str, str]       ==> cols_to_join = {'orig':cols_to_join[0], 'join':cols_to_join[1], 'needs_prefixes':True}
              cols_to_join = [str, bool]      ==> cols_to_join = {'orig':cols_to_join[0], 'join':cols_to_join[0], 'needs_prefixes':join':cols_to_join[1]}
              cols_to_join = [bool, str]      ==> cols_to_join = {'orig':cols_to_join[1], 'join':cols_to_join[1], 'needs_prefixes':join':cols_to_join[0]}
              cols_to_join = [str, str, bool] ==> cols_to_join = {'orig':cols_to_join[0], 'join':cols_to_join[1], 'needs_prefixes':join':cols_to_join[2]}
            cols_to_join = list = dict
              cols_to_join.keys() = ['orig']                           ==> cols_to_join = {'orig':cols_to_join['orig'], 'join':cols_to_join['orig'], 
                                                                                           'needs_prefixes':True}
              cols_to_join.keys() = ['join']                           ==> cols_to_join = {'orig':cols_to_join['join'], 'join':cols_to_join['join'], 
                                                                                           'needs_prefixes':True}
              cols_to_join.keys() = ['orig', 'needs_prefixes']         ==> cols_to_join = {'orig':cols_to_join['orig'], 'join':cols_to_join['orig'], 
                                                                                           'needs_prefixes':cols_to_join['needs_prefixes']}
              (^similar for cols_to_join.keys() = ['join', 'needs_prefixes']  )                                                                           
              cols_to_join.keys() = ['orig', 'join', 'needs_prefixes'] ==> cols_to_join = {'orig':cols_to_join['orig'], 'join':cols_to_join['join'], 
                                                                                           'needs_prefixes':cols_to_join['needs_prefixes']}
        --------------------------------------------------------------
        EXAMPLES
        list_of_columns_to_join = [
            'col_12', 
            ['col_12'], 
            ['col_12', False], 
            [False, 'col_12'], 
            ['col_1', 'col_2'], 
            ['col_1', 'col_2', True], 
            ['col_1', 'col_2', False], 
            {'orig':'col_1'},
            {'join':'col_2'},
            {'orig':'col_1', 'needs_prefixes':False},
            {'join':'col_2', 'needs_prefixes':False},
            {'orig':'col_1', 'join':'col_2'}, 
            {'orig':'col_1', 'join':'col_2', 'needs_prefixes':True}, 
            {'orig':'col_1', 'join':'col_2', 'needs_prefixes':False}
        ]
        But these will be split up into smaller groups for space reasons (can't figure out how to split expected outcome over multiple lines!)

        >>> build_list_of_columns_to_join_dicts(['col_12', ['col_12'], ['col_12', False]])
        [{'orig': 'col_12', 'join': 'col_12', 'needs_prefixes': True}, {'orig': 'col_12', 'join': 'col_12', 'needs_prefixes': True}, {'orig': 'col_12', 'join': 'col_12', 'needs_prefixes': False}]

        >>> build_list_of_columns_to_join_dicts([[False, 'col_12'], ['col_1', 'col_2'], ['col_1', 'col_2', True]])
        [{'orig': 'col_12', 'join': 'col_12', 'needs_prefixes': False}, {'orig': 'col_1', 'join': 'col_2', 'needs_prefixes': True}, {'orig': 'col_1', 'join': 'col_2', 'needs_prefixes': True}]

        >>> build_list_of_columns_to_join_dicts([['col_1', 'col_2', False], {'orig':'col_1'}, {'join':'col_2'}])
        [{'orig': 'col_1', 'join': 'col_2', 'needs_prefixes': False}, {'orig': 'col_1', 'join': 'col_1', 'needs_prefixes': True}, {'join': 'col_2', 'orig': 'col_2', 'needs_prefixes': True}]

        >>> build_list_of_columns_to_join_dicts([{'orig':'col_1', 'needs_prefixes':False}, {'join':'col_2', 'needs_prefixes':False}, {'orig':'col_1', 'join':'col_2'}])
        [{'orig': 'col_1', 'needs_prefixes': False, 'join': 'col_1'}, {'join': 'col_2', 'needs_prefixes': False, 'orig': 'col_2'}, {'orig': 'col_1', 'join': 'col_2', 'needs_prefixes': True}]

        >>> build_list_of_columns_to_join_dicts([{'orig':'col_1', 'join':'col_2', 'needs_prefixes':True}, {'orig':'col_1', 'join':'col_2', 'needs_prefixes':False}])
        [{'orig': 'col_1', 'join': 'col_2', 'needs_prefixes': True}, {'orig': 'col_1', 'join': 'col_2', 'needs_prefixes': False}]

        """
        # For build_join_statement, in the end, list_of_columns_to_join should be a list of dict items,
        #   where each dict item has keys = ['orig', 'join', 'needs_prefixes']
        # Thus, list_of_columns_to_join needs to be converted to list_of_columns_to_join_dicts.
        # Default will be for list_of_columns_to_join_dicts[idx]['needs_prefixes']=True, but one would want it 
        #  equal to False for e.g., EXTRACT(YEAR FROM DOV.DT_OFF_TS)=DOV_X.year
        assert(
            isinstance(list_of_columns_to_join, list) or 
            isinstance(list_of_columns_to_join, tuple)
        )
        list_of_columns_to_join_dicts = []
        for cols_to_join in list_of_columns_to_join:
            if isinstance(cols_to_join, dict):
                assert(
                    'orig' in cols_to_join or 
                    'join' in cols_to_join
                )
                if not ('orig' in cols_to_join and 'join' in cols_to_join):
                    if 'orig' in cols_to_join:
                        cols_to_join['join'] = cols_to_join['orig']
                    else:
                        cols_to_join['orig'] = cols_to_join['join']
                cols_to_join['needs_prefixes'] = cols_to_join.get('needs_prefixes', True)
            elif isinstance(cols_to_join, list) or isinstance(cols_to_join, tuple):
                assert(len(cols_to_join)<=3)
                if len(cols_to_join)==1:
                    cols_to_join = {
                        'orig'           : cols_to_join[0], 
                        'join'           : cols_to_join[0], 
                        'needs_prefixes' : True
                    }
                elif len(cols_to_join)==2:
                    if isinstance(cols_to_join[0], bool):
                        assert(isinstance(cols_to_join[1], str))
                        cols_to_join = {
                            'orig'           : cols_to_join[1], 
                            'join'           : cols_to_join[1], 
                            'needs_prefixes' : cols_to_join[0]
                        }
                    elif isinstance(cols_to_join[1], bool):
                        assert(isinstance(cols_to_join[0], str))
                        cols_to_join = {
                            'orig'           : cols_to_join[0], 
                            'join'           : cols_to_join[0], 
                            'needs_prefixes' : cols_to_join[1]
                        }
                    else:
                        assert(isinstance(cols_to_join[0], str) and isinstance(cols_to_join[1], str))
                        cols_to_join = {
                            'orig'           : cols_to_join[0], 
                            'join'           : cols_to_join[1], 
                            'needs_prefixes' : True
                        }
                else:
                    cols_to_join = {
                        'orig'           : cols_to_join[0], 
                        'join'           : cols_to_join[1], 
                        'needs_prefixes' : cols_to_join[2]
                    }
            else:
                assert(isinstance(cols_to_join, str))
                cols_to_join = {
                    'orig'           : cols_to_join, 
                    'join'           : cols_to_join, 
                    'needs_prefixes' : True
                }
            list_of_columns_to_join_dicts.append(cols_to_join)
        return list_of_columns_to_join_dicts
    
    @staticmethod
    def convert_column_to_join_dict_to_str(
        col_to_join_dict , 
        join_table_alias , 
        orig_table_alias ,
    ):
        # This is to build a single equality string which will go within a single join statement
        # Typically, there will be more than one.  In general, this will be called as many times
        #   as there are columns being joined between the two tables.
        # This list is for a single join statement (joining one 'join' table to one 'orig' table)
        assert(
            'orig'           in col_to_join_dict and 
            'join'           in col_to_join_dict and 
            'needs_prefixes' in col_to_join_dict
        )
        prefix_orig = f"{orig_table_alias}." if (col_to_join_dict['needs_prefixes'] and orig_table_alias) else ''
        prefix_join = f"{join_table_alias}." if (col_to_join_dict['needs_prefixes'] and join_table_alias) else ''
        return_str  =  f"{prefix_orig}{col_to_join_dict['orig']}={prefix_join}{col_to_join_dict['join']}"
        return return_str

    @staticmethod
    def build_join_statement(
        join_type               , 
        join_table              , 
        join_table_alias        , 
        orig_table_alias        , 
        list_of_columns_to_join ,
    ):
        # This is for a single join statement (joining one 'join' table to one 'orig' table)
        # join_type should be e.g. 'INNER', 'LEFT OUTER', etc.
        # See build_list_of_columns_to_join_dicts documentation for allowed values of list_of_columns_to_join
        if join_table:
            join_stmnt = f"{join_type} JOIN {join_table} {join_table_alias} ON"
        else:
            join_stmnt = f"{join_type} JOIN {join_table_alias} ON"
        list_of_columns_to_join_dicts = SQLJoin.build_list_of_columns_to_join_dicts(list_of_columns_to_join)
        for i,col_to_join_dict in enumerate(list_of_columns_to_join_dicts):
            stmnt_i = ' '
            if i>0:
                stmnt_i = ' AND '
            stmnt_i += SQLJoin.convert_column_to_join_dict_to_str(col_to_join_dict, join_table_alias, orig_table_alias)
            join_stmnt += stmnt_i
        return join_stmnt
    
    def get_statement_string(self):
        join_stmnt = SQLJoin.build_join_statement(
            join_type               = self.join_type, 
            join_table              = self.join_table, 
            join_table_alias        = self.join_table_alias, 
            orig_table_alias        = self.orig_table_alias, 
            list_of_columns_to_join = self.list_of_columns_to_join
        )
        return join_stmnt
    
  
class SQLJoinCollection():
    def __init__(self):
        self.joins_dict={}
        
    def is_empty(self):
        n_joins = len(self.joins_dict)
        if n_joins == 0:
            return True
        else:
            assert(n_joins>0)
            return False
        
        
    @staticmethod    
    def check_joins_dict(
        joins_dict   , 
        enforce_pass = True
    ):
        # Statements should be labelled 0 to n_statements-1
        #     ==> number of statements should equal the unique number of keys
        # AND ==> number of statements should equal the maximum key value +1
        # If joins_dict is empty (e.g. when beginning to be built), return true
        if len(joins_dict)==0:
            return True
        pass_check = (
            len(joins_dict)==len(np.unique(list(joins_dict.keys()))) and 
            len(joins_dict)==max(joins_dict.keys())+1
        )
        if enforce_pass:
            assert(pass_check)
        return pass_check

    @staticmethod
    def insert_to_joins_dict_at_idx(
        statement  , 
        joins_dict , 
        idx        = None, 
        run_check  = False
    ):
        # If idx is None, put new element at end.
        # Otherwise, insert statement at idx
        #   Which involves first shifting all with keys >= idx forward one
        if run_check:
            _ = SQLJoinCollection.check_joins_dict(joins_dict, enforce_pass=True)
        if idx is None:
            idx = len(joins_dict)
        else:
            # Need to shift all entries with key>=idx up one, then insert statement at idx
            # Python doesn't like me doing this in one step with curly brackets {}.
            #   So instead I first make a list up tuples then conver to dict.
            joins_dict = dict([(k,v) if k<idx else ((k+1),v)
                                    for k,v in joins_dict.items()])
        assert(idx not in joins_dict)
        joins_dict[idx] = statement
        if run_check:
            _ = SQLJoinCollection.check_joins_dict(joins_dict, enforce_pass=True)
        return joins_dict
    

    def add_join_to_coll(
        self      , 
        sql_join  , 
        idx       = None, 
        run_check = False
    ):
        assert(isinstance(sql_join, SQLJoin))
        self.joins_dict = SQLJoinCollection.insert_to_joins_dict_at_idx(
            sql_join, 
            self.joins_dict, 
            idx       = idx, 
            run_check = run_check
        )
                                                                        
    def build_and_add_join_to_coll(
        self                    , 
        join_type               , 
        join_table              , 
        join_table_alias        , 
        orig_table_alias        , 
        list_of_columns_to_join , 
        idx                     = None, 
        run_check               = False
    ):
        sql_join = SQLJoin(
            join_type               = join_type, 
            join_table              = join_table, 
            join_table_alias        = join_table_alias, 
            orig_table_alias        = orig_table_alias, 
            list_of_columns_to_join = list_of_columns_to_join
        )
        self.add_join_to_coll(sql_join=sql_join, idx=idx, run_check=run_check)

        
    def get_statement_string(
        self                       , 
        include_leading_whitespace = '\t'
    ):
        #---------------
        if len(self.joins_dict)==0:
            return ''
        #---------------
        _ = SQLJoinCollection.check_joins_dict(self.joins_dict, enforce_pass=True)
        #---------------
        sql = ""
        for idx in range(len(self.joins_dict)):
            stmnt_i = self.joins_dict[idx].get_statement_string()
            if include_leading_whitespace is not None:
                stmnt_i = f"\t{stmnt_i}"
            if idx>0:
                stmnt_i = f"\n{stmnt_i}"
            sql += stmnt_i
        return sql
    
		
