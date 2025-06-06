#!/usr/bin/env python

r"""
Holds GenAn class.  See GenAn.GenAn for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from natsort import natsorted, natsort_keygen
import copy
import json
import warnings
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
from SQLQuery import SQLQuery
from SQLQueryGeneric import SQLQueryGeneric
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
#--------------------------------------------------
from CustomJSON import CustomWriter

class GenAn:
    r"""
    class GenAn documentation
    GenAn stands for GeneralAnalysis
    """
    def __init__(
        self                      , 
        df_construct_type         = None, 
        contstruct_df_args        = None, 
        init_df_in_constructor    = True, 
        build_sql_function        = None, 
        build_sql_function_kwargs = None, 
        save_args                 = False, 
    ):
        r"""
        if df_construct_type==DFConstructType.kReadCsv or DFConstructType.kReadCsv:
          contstruct_df_args needs to have at least 'file_path'
        if df_construct_type==DFConstructType.kRunSqlQuery:
          contstruct_df_args needs at least 'conn_db'      
        """
        #--------------------------------------------------
        self.build_sql_function             = build_sql_function
        self.build_sql_function_kwargs      = build_sql_function_kwargs
        self.save_args                      = save_args
        #--------------------------------------------------
        self.df                             = pd.DataFrame()
        
        self.df_construct_type              = df_construct_type
        self.contstruct_df_args             = contstruct_df_args
        if self.contstruct_df_args is None:
            self.contstruct_df_args = {}
        #--------------------------------------------------
        self.cols_and_types_to_convert_dict = self.contstruct_df_args.pop(
            'cols_and_types_to_convert_dict', 
            self.get_default_cols_and_types_to_convert_dict()
        ) 
        self.to_numeric_errors              = self.contstruct_df_args.pop('to_numeric_errors', 'coerce')
        self.read_sql_args                  = self.contstruct_df_args.pop('read_sql_args', {})
        #--------------------------------------------------
        if init_df_in_constructor:
            assert(self.build_sql_function is not None)
            self.init_df()

    #****************************************************************************************************
    # NOTE: The following non-static methods should likely be re-defined for any inhertied classes
    def get_conn_db(self):
        return None
    
    def get_default_cols_and_types_to_convert_dict(self):
        cols_and_types_to_convert_dict = None
        return cols_and_types_to_convert_dict
    
    def get_full_default_sort_by(self):
        full_default_sort_by = None
        return full_default_sort_by
    
    #****************************************************************************************************
    def get_df(self):
        return self.df

    def write_df_to_csv(
        self, 
        save_path, 
        index=False
    ):
        self.df.to_csv(save_path, index=index)
    
    def df_equals(
        self            , 
        df_2            , 
        sort_by         = None, 
        cols_to_compare = None
    ):
        if isinstance(df_2, self.__class__):
            df_2 = df_2.get_df()
        #-------------------------
        if sort_by is None:
            full_default_sort_by = self.get_full_default_sort_by()
            sort_by = Utilities_df.get_default_sort_by_cols_for_comparison(
                full_default_sort_by_for_comparison = full_default_sort_by, 
                df_1                                = self.get_df(), 
                df_2                                = df_2
            )
        return Utilities_df.are_sorted_dfs_equal(
            df1             = self.get_df(), 
            df2             = df_2, 
            sort_by         = sort_by, 
            cols_to_compare = cols_to_compare
        )
    
    #****************************************************************************************************
    @staticmethod
    def read_df_from_csv(
        read_path                      , 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        drop_na_rows_when_exception    = True, 
        drop_unnamed0_col              = True, 
        pd_read_csv_kwargs             = None
    ):
        #-------------------------
        if pd_read_csv_kwargs is None:
            pd_read_csv_kwargs = {}
        #-------------------------
        try:
            df = pd.read_csv(
                read_path , 
                dtype     = str, 
                **pd_read_csv_kwargs
            )
        except:
            print(f"Error reading: {read_path}\nTrying again with encoding_errors='ignore' and on_bad_lines='skip'\n")
            df = pd.read_csv(
                read_path, 
                dtype           = str, 
                encoding_errors = 'replace', 
                on_bad_lines    = 'skip', 
                **pd_read_csv_kwargs
            )
            # Using encoding_errors='replace' above causes any errors to be replaced by U+FFFD, the official REPLACEMENT CHARACTER
            #  (this is essentially a question mark within a black diamond, print u'\uFFFD' if interested)
            # When these errors occur, it is best to simply remove the entire rows containing errors, which the following lines achieve
            #   Below, bad_rows will be a pd.Series object with True/False (boolean) values denoting whether or not a row is bad_rows
            #   We want only good lines, hence we keep df.loc[~bad_lines]
            bad_rows = df.apply(lambda row: row.astype(str).str.contains(u'\uFFFD').any(), axis=1)
            df = df.loc[~bad_rows]
            
            # Sometimes, errors above cause surrounding rows to be filled with all NaNs (except possible the 'Unnamed: 0' column)
            # If drop_na_rows_when_exception==True, drop any such rows.
            if drop_na_rows_when_exception:
                df = df.dropna(subset=[x for x in df.columns if x!='Unnamed: 0'], how='all')
        #-------------------------    
        df = Utilities_df.remove_table_aliases(df)
        #-------------------------
        if drop_unnamed0_col:
            df = Utilities_df.drop_unnamed_columns(
                df            = df, 
                regex_pattern = r'Unnamed.*', 
                ignore_case   = True, 
                inplace       = True, 
            )
        #-------------------------
        if cols_and_types_to_convert_dict is not None:
            cols_and_types_to_convert_dict = {
                k:v for k,v in cols_and_types_to_convert_dict.items() 
                if k in df.columns
            }
        df = Utilities_df.convert_col_types(
            df                  = df, 
            cols_and_types_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors   = to_numeric_errors, 
            inplace             = True
        )
        if 'outg_rec_nb' in df.columns:
            df = Utilities_df.convert_col_type(
                df                = df, 
                column            = 'outg_rec_nb', 
                to_type           = np.int32, 
                to_numeric_errors = to_numeric_errors, 
                inplace           = True
            )
        return df
        
    @staticmethod
    def read_df_from_csv_batch(
        paths                          , 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        drop_na_rows_when_exception    = True, 
        drop_unnamed0_col              = True, 
        pd_read_csv_kwargs             = None, 
        make_all_columns_lowercase     = False, 
        assert_all_cols_equal          = True, 
        min_fsize_MB                   = None
    ):
        r"""
        assert_all_cols_equal:
          - If True, then all DataFrames found must have the same columns
          - If False, then the subset of columns shared by all DataFrames will be used
          
        min_fsize_MB:
            If set (i.e., if not None): 
                skip any file in paths whose size in megabytes (MB) is less than OR EQUAL TO min_fsize_MB
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(paths, [list, str]))
        if isinstance(paths, str):
            paths = [paths]
        #-------------------------
        dfs = []
        for path in paths:
            assert(os.path.exists(path))
            if min_fsize_MB is not None:
                fsize = 1e-6*os.path.getsize(path)
                if fsize <= min_fsize_MB:
                    continue
            #-----
            df_i = GenAn.read_df_from_csv(
                read_path                      = path, 
                cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
                to_numeric_errors              = to_numeric_errors, 
                drop_na_rows_when_exception    = drop_na_rows_when_exception, 
                drop_unnamed0_col              = drop_unnamed0_col, 
                pd_read_csv_kwargs             = pd_read_csv_kwargs
            )
            if df_i.shape[0]==0:
                continue
            if make_all_columns_lowercase:
                df_i = Utilities_df.make_all_column_names_lowercase(df_i) 
            dfs.append(df_i)
        #-------------------------
        if len(dfs)==0:
            return pd.DataFrame()
        #-------------------------                
        df_cols = Utilities_df.get_shared_columns(dfs, maintain_df0_order=True)
        for i in range(len(dfs)):
            if assert_all_cols_equal:
                # In order to account for case where columns are the same but in different order
                # one must compare the length of dfs[i].columns to that of df_cols (found by utilizing
                # the Utilities_df.get_shared_columns(dfs) functionality)
                assert(dfs[i].shape[1]==len(df_cols))
            dfs[i] = dfs[i][df_cols]
        
        # for df_i in dfs:
            # assert(all(df_i.columns==df_cols))
        #-------------------------
        return_df = pd.concat(dfs)
        return return_df
        
    @staticmethod
    def read_df_from_csv_dir_batches(
        files_dir                      , 
        file_path_glob                 , 
        file_path_regex                = None, 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        drop_unnamed0_col              = True, 
        pd_read_csv_kwargs             = None, 
        assert_all_cols_equal          = True
    ):
        r"""
        assert_all_cols_equal:
          - If True, then all DataFrames found must have the same columns
          - If False, then the subset of columns shared by all DataFrames will be used
        """
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir      = files_dir, 
            glob_pattern  = file_path_glob, 
            regex_pattern = file_path_regex
        )
        paths=natsorted(paths)
        #-------------------------
        return GenAn.read_df_from_csv_batch(
            paths                          = paths, 
            cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors              = to_numeric_errors, 
            drop_unnamed0_col              = drop_unnamed0_col, 
            pd_read_csv_kwargs             = pd_read_csv_kwargs, 
            assert_all_cols_equal          = assert_all_cols_equal
        )
    
    #****************************************************************************************************
    @staticmethod
    def build_sql_general(
        build_sql_function        , 
        build_sql_function_kwargs = None
    ):
        #-------------------------
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = dict()
        #-------------------------
        # Some special kwargs are popped off before feeding into sql function.
        # Replicate that here
        keys_to_pop = [
            'field_to_split', 
            'field_to_split_location_in_kwargs', 
            'save_and_dump', 
            'sort_coll_to_split', 
            'batch_size', 
            'verbose', 
            'n_update', 
            'ignore_index', 
        ]
        kwargs_fnl = {
            k:v for k,v in build_sql_function_kwargs.items() 
            if k not in keys_to_pop
        }
        #-------------------------
        sql = build_sql_function(**kwargs_fnl)
        return sql
        
    @staticmethod
    def reduce_dtypes_in_read_sql_args(
        sql           , 
        read_sql_args , 
    ):
        r"""
        When running pd.read_sql_query with dtype argument equal to a dict, the method fails if columns are included
        in dtypes which are not found in the query.  The error code is:
          KeyError: 'Only a column name can be used for the key in a dtype mappings argument.'
        This function attempts to remove such columns from read_sql_args.
        The purpose is because I like to set read_sql_args for some classes, but I don't always grab all columns when running
          queries.  For example, in DOVSOutages, I have read_sql_args = {'OUTG_REC_NB':np.int32, 'PREMISE_NB':np.int32}.
                      When running a query without grabbing premise numbers, read_sql_query would fail without this
                      reduce_dtypes_in_read_sql_args functionality.
        When sql is a SQLQuery object, I look for the columns in sql.sql_select.
        When sql is a string, I simply look for the columns somewhere in the string (inbetween where 'SELECT' is found and 
          where '\nFrom' is found)
            - NOTE: '\n' needed in '\nFrom' to weed out, e.g., the FROM in 'EXTRACT(YEAR FROM DOV.DT_OFF_TS) AS START_YEAR', 
                    in which case the search would not occur over the full select statement!
            - not the best method or most refined method, but should be good enough for now!)
        """
        #-------------------------
        if 'dtype' not in read_sql_args:
            return read_sql_args
        if isinstance(read_sql_args['dtype'], str):
            return read_sql_args
        assert(isinstance(read_sql_args['dtype'], dict))
        dtype_args = read_sql_args['dtype']
        #-------------------------
        if isinstance(sql, SQLQuery):
            dtype_args = {k:v for k,v in dtype_args.items() 
                          if sql.sql_select.find_idx_of_approx_element_in_collection_dict(k)>-1}
        elif isinstance(sql, SQLQueryGeneric):
            sql = sql.get_sql_statement()
            return GenAn.reduce_dtypes_in_read_sql_args(
                sql           = sql, 
                read_sql_args = read_sql_args
            )
        else:
            assert(isinstance(sql, str))
            #----------
            beg_search = sql.find('SELECT')
            assert(beg_search>-1)
            beg_search += len('SELECT')
            #----------
            end_search = sql.find('\nFROM')
            assert(end_search>-1 and end_search>beg_search)
            end_search += len('\nFROM')
            #----------
            dtype_args = {
                k:v for k,v in dtype_args.items() 
                if sql.find(k, beg_search, end_search)>-1
            }
        #-------------------------
        read_sql_args['dtype'] = dtype_args
        return read_sql_args

    @staticmethod
    def build_sql_statement_general(
        build_sql_function        , 
        build_sql_function_kwargs = None, 
        read_sql_args             = None
    ):
        sql = GenAn.build_sql_general(
            build_sql_function        = build_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs
        )
        #-----
        if read_sql_args:
            read_sql_args = GenAn.reduce_dtypes_in_read_sql_args(
                sql           = sql, 
                read_sql_args = read_sql_args
            )
        else:
            # In case read_sql_args was None, set to {}
            read_sql_args = {}
        #-----
        if Utilities.is_object_one_of_types(sql, [SQLQuery, SQLQueryGeneric]):
            sql = sql.get_sql_statement()
        #----------
        return sql, read_sql_args

    #****************************************************************************************************        
    @staticmethod
    def prepare_save_args(
        save_args            , 
        make_save_dir_if_dne = True
    ):
        r"""
        Prepare save_args to be used in build_df_general_batches
        
        To not save, save_args may be None, False, or empty dict
          In such a case, this returns a dict with save_to_file=save_individual_batches=save_full_final=False
          
        To save, save_args must be a dict with AT LEAST save_dir and save_name keys
          Full set of acceptable keys include:
            - save_dir
            - save_name
            - save_ext
                - default to extension in save_name (if exists) or '.csv'
            - save_to_file
                - default True
            - save_individual_batches
                - default True
            - save_full_final
                - default True
            - index
                - default False
            - offset_int
                - default None
            
        NOTE: If save_ext is supplied, it trumps all others (even if save_args['save_name'] contains an extension!)
              Then, the extension of save_args['save_name'], if it exists.
              Finally, '.csv' as default
        """
        # If save_args is None, False, or empty dict, etc.
        # return dict with save_to_file=save_individual_batches=save_full_final=False
        if not save_args:
            save_args = dict(
                save_to_file            = False, 
                save_individual_batches = False, 
                save_full_final         = False
            )
            return save_args
        else:
            assert(isinstance(save_args, dict))
            #----------
            save_args['save_to_file']            = save_args.get('save_to_file', True)
            save_args['save_individual_batches'] = save_args.get('save_individual_batches', True)
            save_args['save_full_final']         = save_args.get('save_full_final', True)
            save_args['index']                   = save_args.get('index', False)
            save_args['save_summary']            = save_args.get('save_summary', True)
            save_args['offset_int']              = save_args.get('offset_int', None)
            #----------
            if save_args['save_to_file']:
                assert('save_dir' in save_args)
                assert('save_name' in save_args)
                name,ext = os.path.splitext(save_args['save_name'])
                save_args['save_name'] = name
                #----------
                # If save_ext is supplied, it trumps all others (even if save_args['save_name'] contains an extension!)
                # Then, the extension of save_args['save_name'], if it exists.
                # Finally, '.csv' as default
                save_args['save_ext'] = save_args.get('save_ext', ext if ext else '.csv')
                if save_args['save_ext'][0] != '.':
                    save_args['save_ext'] = '.'+save_args['save_ext']
                #----------
                save_args['save_path']         = os.path.join(save_args['save_dir'], save_args['save_name']+save_args['save_ext'])
                save_args['save_summary_path'] = os.path.join(save_args['save_dir'], 'summary_files', save_args['save_name']+'_summary.json')
                #----------
                if not os.path.exists(save_args['save_dir']) and make_save_dir_if_dne:
                    os.makedirs(save_args['save_dir'])
                if not os.path.exists(os.path.join(save_args['save_dir'], 'summary_files')) and save_args['save_summary'] and make_save_dir_if_dne:
                    os.makedirs(os.path.join(save_args['save_dir'], 'summary_files'))
                #----------
            return save_args
            
    @staticmethod
    def prepare_summary(        
        build_sql_function                , 
        sql_statement                     , 
        build_sql_function_kwargs         = None, 
        cols_and_types_to_convert_dict    = None, 
        to_numeric_errors                 = 'coerce',        
        field_to_split                    = None, 
        field_to_split_location_in_kwargs = None, 
        collection_i                      = None
    ):
        r"""
        """
        #-------------------------
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = dict()
        #-------------------------
        if field_to_split is not None:
            assert(field_to_split_location_in_kwargs is not None)
            assert(collection_i is not None)
        #----------
        return_dict = {
            'build_sql_function'             : build_sql_function, 
            'sql_statement'                  : sql_statement, 
            'build_sql_function_kwargs'      : build_sql_function_kwargs, 
            'cols_and_types_to_convert_dict' : cols_and_types_to_convert_dict, 
            'to_numeric_errors'              : to_numeric_errors
        }
        if field_to_split is not None:
            return_dict = {
                **return_dict, 
                **{
                    'field_to_split':field_to_split, 
                    'field_to_split_location_in_kwargs':field_to_split_location_in_kwargs,
                    'collection_i':collection_i
                }
            }
        return return_dict

    @staticmethod
    def output_summary(
        output_path                       , 
        build_sql_function                , 
        sql_statement                     , 
        build_sql_function_kwargs         = None, 
        cols_and_types_to_convert_dict    = None, 
        to_numeric_errors                 = 'coerce',        
        field_to_split                    = None, 
        field_to_split_location_in_kwargs = None, 
        collection_i                      = None
    ):
        summary_dict = GenAn.prepare_summary(    
            build_sql_function                = build_sql_function, 
            sql_statement                     = sql_statement, 
            build_sql_function_kwargs         = build_sql_function_kwargs, 
            cols_and_types_to_convert_dict    = cols_and_types_to_convert_dict, 
            to_numeric_errors                 = to_numeric_errors,        
            field_to_split                    = field_to_split, 
            field_to_split_location_in_kwargs = field_to_split_location_in_kwargs, 
            collection_i                      = collection_i
        )
        CustomWriter.output_dict_to_json(output_path, summary_dict)
        
        
    @staticmethod
    def find_summary_files(save_args):
        r"""
        The following procedure will find save_dir+'summary_files'+save_name_summary.json as well as save_dir+save_name_#_summary.json
          e.g., 
                files_dir       = r'C:\Users\s346557\Documents\TMP\summary_files'
                file_path_glob  = r'Test*_summary.json'
                file_path_regex = r'Test(_[0-9]*)|()_summary.json'
             Will return
                ['C:\\Users\\s346557\\Documents\\TMP\\Test_0_summary.json',
                 'C:\\Users\\s346557\\Documents\\TMP\\Test_1_summary.json',
                 'C:\\Users\\s346557\\Documents\\TMP\\Test_2_summary.json',
                 'C:\\Users\\s346557\\Documents\\TMP\\Test_3_summary.json',
                 'C:\\Users\\s346557\\Documents\\TMP\\Test_4_summary.json',
                 'C:\\Users\\s346557\\Documents\\TMP\\Test_summary.json']

         If only save_dir+save_name_#_summary.json are wanted, one could use
                files_dir       = r'C:\Users\s346557\Documents\TMP'
                file_path_glob  = r'Test_[0-9]*_summary.json'
                file_path_regex = None
        """
        save_args = copy.deepcopy(save_args)
        save_args = GenAn.prepare_save_args(
            save_args            = save_args, 
            make_save_dir_if_dne = False
        )
        #-----
        files_dir       = Path(save_args['save_summary_path']).parent
        file_path_glob  = save_args['save_name']+'*_summary.json'
        file_path_regex = save_args['save_name']+'(_[0-9]*)|()_summary.json'
        #-----
        paths = Utilities.find_all_paths(
            base_dir      = files_dir, 
            glob_pattern  = file_path_glob, 
            regex_pattern = file_path_regex
        )
        #-----
        return paths

    @staticmethod
    def get_split_collection_from_summary(summary_path):
        with open(summary_path, 'r') as f:
            summary_dict = json.load(f)
        #----------
        if (
            'build_sql_function_kwargs' not in summary_dict or 
            'field_to_split_location_in_kwargs' not in summary_dict
        ):
            return []
        #----------
        assert('build_sql_function_kwargs'         in summary_dict)
        assert('field_to_split_location_in_kwargs' in summary_dict)
        #----------
        coll_i = Utilities.get_from_nested_dict(
            nested_dict    = summary_dict['build_sql_function_kwargs'], 
            keys_path_list = summary_dict['field_to_split_location_in_kwargs']
        )
        #----------
        return coll_i

    @staticmethod
    def get_split_collection_from_all_summary_files(save_args):
        summary_paths = GenAn.find_summary_files(save_args=save_args)
        return_coll = []
        for summary_path in summary_paths:
            coll_i = GenAn.get_split_collection_from_summary(summary_path)
            return_coll.extend(coll_i)
        # NOTE: The list(set()) method only works with elements are hashable.
        # If events were collected using a consolidated DF (i.e., slim) with MultiIndex
        #   index, then the members will be lists, which are not hashable!
        # The fix is to convert the elements from lists to tuples, which are hashable
        # NOTE: However, if no previous files, return_coll will be empty list, in which case
        #       return_coll[0] will throw an error, hence the need for 'if return_coll and...'
        if return_coll and isinstance(return_coll[0], list):
            return_coll = [tuple(x) for x in return_coll]
            assert(Utilities.is_hashable(return_coll[0]))
        return_coll = list(set(return_coll))
        return return_coll
        
    @staticmethod
    def get_next_summary_file_tag_int(save_args):
        summary_paths = GenAn.find_summary_files(save_args=save_args)
        #----------
        save_args = copy.deepcopy(save_args)
        save_args = GenAn.prepare_save_args(
            save_args            = save_args, 
            make_save_dir_if_dne = False
        )
        #-----
        file_path_regex = save_args['save_name']+r'_(\d*)_summary.json'
        #---------- 
        tags = []
        for path in summary_paths:
            tags.append(re.findall(file_path_regex, path))
        # Should have only been one tag found per path
        assert(all([len(x)==1 for x in tags]))
        tags = [x[0] for x in tags]
        tags = natsorted(tags)
        #----------
        if len(tags)==0:
            return 0
        else:
            return int(tags[-1])+1

    #****************************************************************************************************
    @staticmethod
    def build_df_general(
        conn_db                        , 
        build_sql_function             , 
        build_sql_function_kwargs      = None, 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        save_args                      = False, 
        return_sql                     = False, 
        read_sql_args                  = None, 
    ):
        r"""
        NOTE: Presence of (non-None valued) 'field_to_split' in build_sql_function_kwargs determines
              whether to run build_df_general or build_df_general_batches
        """
        #-------------------------
        # Function will alter build_sql_function_kwargs by popping various elements off.
        # In order to maintain the original build_sql_function_kwargs (which likely came from some instance
        # of GenAn or classes derived from it), copy it.
        build_sql_function_kwargs = copy.deepcopy(build_sql_function_kwargs)
        #-------------------------
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = dict()
        if read_sql_args is None:
            read_sql_args = dict()
        #-------------------------
        # Run in batches???
        # If field_to_split is not None, yes.
        field_to_split                    = build_sql_function_kwargs.pop('field_to_split', None)
        field_to_split_location_in_kwargs = build_sql_function_kwargs.pop('field_to_split_location_in_kwargs', None)
        save_and_dump                     = build_sql_function_kwargs.pop('save_and_dump', False)
        sort_coll_to_split                = build_sql_function_kwargs.pop('sort_coll_to_split', False)
        batch_size                        = build_sql_function_kwargs.pop('batch_size', 1000)
        verbose                           = build_sql_function_kwargs.pop('verbose', True)
        n_update                          = build_sql_function_kwargs.pop('n_update', 10)
        ignore_index                      = build_sql_function_kwargs.pop('ignore_index', False)
        save_args                         = GenAn.prepare_save_args(
            save_args            = save_args, 
            make_save_dir_if_dne = True
        )
        if field_to_split is not None:
            # GenAn.build_df_general_batches essentially splits up the batches and sends them
            # right back to GenAn.build_df_general
            return GenAn.build_df_general_batches(
                conn_db                           = conn_db, 
                build_sql_function                = build_sql_function, 
                build_sql_function_kwargs         = build_sql_function_kwargs, 
                field_to_split                    = field_to_split, 
                field_to_split_location_in_kwargs = field_to_split_location_in_kwargs, 
                save_and_dump                     = save_and_dump, 
                sort_coll_to_split                = sort_coll_to_split, 
                batch_size                        = batch_size, 
                verbose                           = verbose, 
                n_update                          = n_update, 
                cols_and_types_to_convert_dict    = cols_and_types_to_convert_dict, 
                to_numeric_errors                 = to_numeric_errors, 
                save_args                         = save_args, 
                return_sql                        = return_sql, 
                read_sql_args                     = read_sql_args, 
                exclude_previously_recorded       = True, 
                ignore_index                      = ignore_index, 
            )
        #-------------------------
        sql, read_sql_args = GenAn.build_sql_statement_general(
            build_sql_function        = build_sql_function, 
            build_sql_function_kwargs = build_sql_function_kwargs, 
            read_sql_args             = read_sql_args
        )
        assert(isinstance(sql, str))
        #-------------------------
        # filterwarnings call to eliminate annoying "UserWarning: pandas only supports SQLAlchemy connectable..."
        # If one wants to get rid of this functionality, remove "with warnings.catch_warnings()" and  "warnings.filterwarnings" call
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql_query(sql, conn_db, **read_sql_args) 
        #-------------------------
        df = Utilities_df.remove_table_aliases(df)
        if cols_and_types_to_convert_dict is not None:
            cols_and_types_to_convert_dict = {
                k:v for k,v in cols_and_types_to_convert_dict.items() 
                if k in df.columns
            }
        df = Utilities_df.convert_col_types(
            df                  = df, 
            cols_and_types_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors   = to_numeric_errors, 
            inplace             = True
        )
        if 'outg_rec_nb' in df.columns:
            df = Utilities_df.convert_col_type(
                df                = df, 
                column            = 'outg_rec_nb', 
                to_type           = np.int32, 
                to_numeric_errors = to_numeric_errors, 
                inplace           = True
            )
        #-------------------------
        if save_args['save_to_file']:
            if not os.path.exists(save_args['save_dir']):
                os.makedirs(save_args['save_dir'])
            df.to_csv(save_args['save_path'], index=save_args['index'])
            if save_args['save_summary']:
                GenAn.output_summary(
                    output_path                       = save_args['save_summary_path'], 
                    build_sql_function                = build_sql_function, 
                    sql_statement                     = sql, 
                    build_sql_function_kwargs         = build_sql_function_kwargs, 
                    cols_and_types_to_convert_dict    = cols_and_types_to_convert_dict, 
                    to_numeric_errors                 = to_numeric_errors,        
                    field_to_split                    = None, 
                    field_to_split_location_in_kwargs = None, 
                    collection_i                      = None
                )
        #-------------------------
        if not return_sql:
            return df
        else:
            return df, sql
    
    @staticmethod
    def build_df_general_batches(
        conn_db                           , 
        build_sql_function                , 
        build_sql_function_kwargs         , 
        field_to_split                    , 
        field_to_split_location_in_kwargs = None, 
        save_and_dump                     = False, 
        sort_coll_to_split                = False, 
        batch_size                        = 1000, 
        verbose                           = True, 
        n_update                          = 10, 
        cols_and_types_to_convert_dict    = None, 
        to_numeric_errors                 = 'coerce', 
        save_args                         = False, 
        return_sql                        = False, 
        read_sql_args                     = None, 
        exclude_previously_recorded       = True, 
        ignore_index                      = False, 
    ):
        r"""
        This essentially splits up the batches and sends them back to GenAn.build_df_general
        -----
        field_to_split_location_in_kwargs:
          This allows for the possibility of field_to_split being buried somewhere in a nested build_sql_function_kwargs.
            e.g., build_sql_function_kwargs may have build_sql_usg_args dict within, in which field_to_split is located
                  In such a case, one would want field_to_split_location_in_kwargs=['build_sql_usg_args', 'field_to_split']
            If field_to_split_location_in_kwargs is None (default), then it is assumed that field_to_split is a (first level) 
              key in build_sql_function_kwargs.
              
        save_and_dump:
            If True, the full return_df will not be built.  Each of the individual df_i's will be built and saved, but not combined
              in the end the form return_df.  An empty pd.DataFrame will be returned in this case.
            The main purpopse is for use when making large data acquisition runs, for which the purpose is simply to save all of the results
              to local/cloud CSVs, pickles, etc.
              Without building the large return_df, this allows the data acquisiton to occur while utilizing much less memory.
        """
        #-------------------------
        # Function will alter build_sql_function_kwargs by popping various elements off.
        # In order to maintain the original build_sql_function_kwargs (which likely came from some instance
        # of GenAn or classes derived from it), copy it.
        build_sql_function_kwargs = copy.deepcopy(build_sql_function_kwargs)
        #-------------------------
        save_args = GenAn.prepare_save_args(
            save_args            = save_args, 
            make_save_dir_if_dne = True
        )
        #-------------------------
        # If save_and_dump==True, the DFs must be set to save!
        if save_and_dump:
            assert(
                save_args['save_to_file'] and 
                save_args['save_individual_batches']
            )
            save_args['save_full_final'] = False
        #-------------------------
        if field_to_split_location_in_kwargs is None:
            field_to_split_location_in_kwargs = [field_to_split]
        if field_to_split != field_to_split_location_in_kwargs[-1]:
            field_to_split_location_in_kwargs.append(field_to_split)
        coll_to_split = Utilities.pop_from_nested_dict(
            nested_dict    = build_sql_function_kwargs, 
            keys_path_list = field_to_split_location_in_kwargs
        )
        if sort_coll_to_split:
            if isinstance(coll_to_split, pd.DataFrame):
                coll_to_split.sort_index(inplace=True, key=natsort_keygen())
            else:
                coll_to_split = natsorted(coll_to_split)
        #-------------------------
        if exclude_previously_recorded and save_args['save_to_file']:
            # If coll_to_split is pd.DataFrame, prev_rec_coll will be list of indices (see CustomJSON.CustomEncoder)
            prev_rec_coll           = GenAn.get_split_collection_from_all_summary_files(save_args)
            save_args['offset_int'] = GenAn.get_next_summary_file_tag_int(save_args)
            if verbose:
                print(f'BEFORE EXCLUDE PREVIOUSLY RECORDED: n_coll = {len(coll_to_split)}')
            if isinstance(coll_to_split, pd.DataFrame):
                coll_to_split = coll_to_split[~coll_to_split.index.isin(prev_rec_coll)]
            else:
                coll_to_split = [
                    x for x in coll_to_split 
                    if x not in prev_rec_coll
                ]
        #-------------------------
        return_df        = pd.DataFrame()
        return_sql_stmnt = ''
        batch_idxs       = Utilities.get_batch_idx_pairs(len(coll_to_split), batch_size)
        n_batches        = len(batch_idxs)
        if verbose:
            print(f'n_coll     = {len(coll_to_split)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches  = {n_batches}')
            
        for i, batch_i in enumerate(batch_idxs):
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-------------------------
            save_args_i = copy.deepcopy(save_args)
            if save_args['save_to_file'] and save_args['save_individual_batches']:
                batch_int = i
                if save_args_i['offset_int'] is not None:
                    assert(isinstance(save_args_i['offset_int'], int))
                    batch_int += save_args_i['offset_int']
                save_args_i['save_name'] = Utilities.append_to_path(
                    save_path                     = save_args_i['save_name'], 
                    appendix                      = f'_{batch_int}', 
                    ext_to_find                   = save_args_i['save_ext'], 
                    append_to_end_if_ext_no_found = True
                )
                # Call prepare_save_args again to compile save_path, save_summary_path, etc.                                          
                save_args_i = GenAn.prepare_save_args(
                    save_args            = save_args_i, 
                    make_save_dir_if_dne = True
                )
            else:
                save_args_i['save_to_file'] = False
            #-----
            save_args_i['save_summary'] = False # Never want build_df_general to handle output of summary for batches
                                                # as the batch collection needs to be included
            #-------------------------
            if isinstance(coll_to_split, pd.DataFrame):
                value = coll_to_split.iloc[i_beg:i_end]
            else:
                value = coll_to_split[i_beg:i_end]
            #-----
            df_i, sql_i = GenAn.build_df_general(
                conn_db                        = conn_db, 
                build_sql_function             = build_sql_function, 
                build_sql_function_kwargs      = Utilities.set_in_nested_dict(
                    nested_dict    = build_sql_function_kwargs, 
                    keys_path_list = field_to_split_location_in_kwargs, 
                    value          = value, 
                    inplace        = False
                ), 
                cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
                to_numeric_errors              = to_numeric_errors, 
                save_args                      = save_args_i, 
                return_sql                     = True, 
                read_sql_args                  = read_sql_args, 
            )
            if(
                save_args['save_to_file'] and 
                save_args['save_individual_batches'] and 
                save_args['save_summary']
            ):
                GenAn.output_summary(
                    output_path                       = save_args_i['save_summary_path'], 
                    build_sql_function                = build_sql_function, 
                    sql_statement                     = sql_i, 
                    build_sql_function_kwargs         = Utilities.set_in_nested_dict(
                        nested_dict    = build_sql_function_kwargs, 
                        keys_path_list = field_to_split_location_in_kwargs, 
                        value          = value, 
                        inplace        = False
                    ), 
                    cols_and_types_to_convert_dict    = cols_and_types_to_convert_dict, 
                    to_numeric_errors                 = to_numeric_errors,        
                    field_to_split                    = field_to_split, 
                    field_to_split_location_in_kwargs = field_to_split_location_in_kwargs, 
                    collection_i                      = value
                )
            #-----
            return_sql_stmnt = return_sql_stmnt + sql_i + '\n' + '-----'*10 + '\n'
            #-----
            if not save_and_dump:
                if return_df.shape[0]>0:
                    assert(all(df_i.columns==return_df.columns))
                return_df = pd.concat([return_df, df_i], axis=0, ignore_index=ignore_index)
        #-------------------------
        if save_args['save_to_file'] and save_args['save_full_final']:
            if not os.path.exists(save_args['save_dir']):
                os.makedirs(save_args['save_dir'])
            return_df.to_csv(save_args['save_path'], index=save_args['index'])   
        #-------------------------
        if not return_sql:
            return return_df
        else:
            return return_df, return_sql_stmnt
    
    #****************************************************************************************************
    def init_df(self):
        assert(
            self.df_construct_type is not None and 
            self.df_construct_type < DFConstructType.kUnset and 
            self.df_construct_type > -1 and 
            self.contstruct_df_args is not None
        )
        if self.df_construct_type == DFConstructType.kReadCsv:
            assert('file_path' in self.contstruct_df_args)
            file_path = self.contstruct_df_args['file_path']
            #-----
            self.df = GenAn.read_df_from_csv( 
                read_path                      = file_path, 
                cols_and_types_to_convert_dict = self.cols_and_types_to_convert_dict, 
                to_numeric_errors              = self.to_numeric_errors, 
                drop_unnamed0_col              = True, 
                pd_read_csv_kwargs             = None
            )
        elif self.df_construct_type == DFConstructType.kRunSqlQuery:
            # contstruct_df_args SHOULD HAVE build_sql_function and build_sql_function_kwargs, 
            #   but doesn't necessarily need to
            conn_db = self.contstruct_df_args.pop('conn_db', self.get_conn_db())
            #-----
            self.df = GenAn.build_df_general(
                conn_db                        = conn_db, 
                build_sql_function             = self.build_sql_function, 
                build_sql_function_kwargs      = self.build_sql_function_kwargs, 
                cols_and_types_to_convert_dict = self.cols_and_types_to_convert_dict, 
                to_numeric_errors              = self.to_numeric_errors, 
                save_args                      = self.save_args, 
                read_sql_args                  = self.read_sql_args
            )
        elif self.df_construct_type==DFConstructType.kImportPickle:
            # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            print('Imprt pickle not yet set up!')
            assert(0)
        else:
            assert(0)

    #****************************************************************************************************            
    def get_sql(self):
        if self.build_sql_function:
            return GenAn.build_sql_general(
                build_sql_function        = self.build_sql_function, 
                build_sql_function_kwargs = self.build_sql_function_kwargs
            )
        else:
            return None

    def get_sql_statement(self):
        if self.build_sql_function:
            sql_statement, _ = GenAn.build_sql_statement_general(
                build_sql_function        = self.build_sql_function, 
                build_sql_function_kwargs = self.build_sql_function_kwargs, 
                read_sql_args             = None
            )
            return sql_statement
        else:
            return None