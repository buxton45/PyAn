#!/usr/bin/env python

r"""
Holds AMINonVee class.  See AMINonVee.AMINonVee for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re
import copy

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
from scipy import stats
import datetime
import time
from natsort import natsorted, ns
#---------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib import dates
#--------------------------------------------------
import CommonLearningMethods as clm
from AMINonVee_SQL import AMINonVee_SQL
from GenAn import GenAn
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Utilities_dt
import Plot_General
#--------------------------------------------------

class AMINonVee(GenAn):
    r"""
    class AMINonVee documentation
    """
    def __init__(
        self, 
        df_construct_type         = DFConstructType.kRunSqlQuery,
        contstruct_df_args        = None, 
        init_df_in_constructor    = True, 
        build_sql_function        = None, 
        build_sql_function_kwargs = None, 
        **kwargs
    ):
        r"""
        if df_construct_type==DFConstructType.kReadCsv or DFConstructType.kReadCsv:
          contstruct_df_args needs to have at least 'file_path'
        if df_construct_type==DFConstructType.kRunSqlQuery:
          contstruct_df_args needs at least 'conn_db'     
        """
        #--------------------------------------------------
        # First, set self.build_sql_function and self.build_sql_function_kwargs
        # and call base class's __init__ method
        #---------------
        self.build_sql_function = (build_sql_function if build_sql_function is not None 
                                   else AMINonVee_SQL.build_sql_usg)
        #---------------
        self.build_sql_function_kwargs = (build_sql_function_kwargs if build_sql_function_kwargs is not None 
                                          else {})
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
        full_default_sort_by = ['aep_endtime_utc', 'serialnumber', 'aep_srvc_qlty_idntfr']
        return full_default_sort_by
        
    #****************************************************************************************************
    @staticmethod
    def get_usg_distinct_fields(
        date_0                           , 
        date_1                           , 
        fields                           = ['serialnumber'], 
        are_datetime                     = False, 
        addtnl_build_sql_function_kwargs = {}, 
        cols_and_types_to_convert_dict   = None, 
        to_numeric_errors                = 'coerce', 
        save_args                        = False, 
        conn_aws                         = None, 
        return_sql                       = False, 
        **kwargs
    ):
        #-------------------------
        if conn_aws is None:
            conn_aws = Utilities.get_athena_prod_aws_connection()
        #-------------------------
        build_sql_function = AMINonVee_SQL.build_sql_usg_distinct_fields
        #-----
        build_sql_function_kwargs = dict(
            date_0       = date_0, 
            date_1       = date_1, 
            fields       = fields, 
            are_datetime = are_datetime, 
            **addtnl_build_sql_function_kwargs
        )
        #-------------------------
        df = GenAn.build_df_general(
            conn_db                        = conn_aws, 
            build_sql_function             = build_sql_function, 
            build_sql_function_kwargs      = build_sql_function_kwargs, 
            cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors              = to_numeric_errors, 
            save_args                      = save_args, 
            return_sql                     = return_sql, 
            **kwargs
        )
        #-------------------------
        return df    


            
    
    #*****************************************************************************************************************
    @staticmethod
    def perform_std_initiation_and_cleaning(
        df             , 
        drop_na_values = True, 
        inplace        = True, 
        **kwargs
    ):
        r"""
        Performs standard cleaning and initialization operations.
        These include:
          1. From the epoch UTC time in timestamp_col, create a human-readable datetime column
          2. Build UTC versions of start_time_col and end_time_col
          3. Change timezoneoffset_col from string to datetime.timedelta
          4. Drop any na values in the value_cols columns (if drop_na_values==True)
          5. Set the index equal to the column created in Utilities_dt.convert_timestamp_to_utc_in_df
             (as this seems to be the most reliable)
        -------------------------
        NOTE: Any of the functionality associated with a given kwarg can be turned off my setting
              the value equal to None
        Acceptable kwargs:
          - timestamp_col 
            - default: aep_endtime_utc
          - utc_from_timestamp_col
            - default: f'{timestamp_col}_from_timestamp'

          - start_time_col
            - default: starttimeperiod
          - start_time_utc_col
            - default: f'{start_time_col}_utc'

          - end_time_col
            - default: endtimeperiod
          - end_time_utc_col
            - default: f'{end_time_col}_utc'

          - timezoneoffset_col
            - default: timezoneoffset

          - value_cols
            - default: ['value']

          - set_index_col
            -  default: utc_from_timestamp_col
            -  Setting equal to None will cause no index to be set
          - index_name
            - default: 'time_idx'
        """
        #-------------------------
        if not inplace:
            df = df.copy()
        #**************************************************
        # Unpack kwargs
        timestamp_col = kwargs.get('timestamp_col', 'aep_endtime_utc')
        if timestamp_col is not None:
            utc_from_timestamp_col = kwargs.get('utc_from_timestamp_col', f'{timestamp_col}_from_timestamp')
        else:
            utc_from_timestamp_col = None
        #----------
        start_time_col     = kwargs.get('start_time_col', 'starttimeperiod')
        start_time_utc_col = kwargs.get('start_time_utc_col', f'{start_time_col}_utc')
        #----------
        end_time_col       = kwargs.get('end_time_col', 'endtimeperiod')
        end_time_utc_col   = kwargs.get('end_time_utc_col', f'{end_time_col}_utc')
        #----------
        timezoneoffset_col = kwargs.get('timezoneoffset_col', 'timezoneoffset')
        #----------
        value_cols         = kwargs.get('value_cols', ['value'])
        #----------
        set_index_col      = kwargs.get('set_index_col', utc_from_timestamp_col)
        index_name         = kwargs.get('index_name', 'time_idx')
        #**************************************************
        df = clm.remove_table_aliases(df)
        #-------------------------
        # Make sure all value_cols are floats
        if value_cols is not None:
            cols_and_types_dict = {x:np.float64 for x in value_cols}
            df = Utilities_df.convert_col_types(
                df                  = df, 
                cols_and_types_dict = cols_and_types_dict, 
                to_numeric_errors   = 'coerce', 
                inplace             = True
            )
        #-------------------------
        # From the epoch UTC time in aep_endtime_utc, create a human-readable datetime column
        # Need to first convert column to int
        if timestamp_col is not None:
            df = Utilities_df.convert_col_type(
                df      = df, 
                column  = timestamp_col, 
                to_type = int
        )
            df = Utilities_dt.convert_timestamp_to_utc_in_df(
                df            = df, 
                timestamp_col = timestamp_col, 
                placement_col = utc_from_timestamp_col
            )
        # Build UTC versions of start_time_col (starttimeperiod) and end_time_col (endtimeperiod)
        if (start_time_col is not None or 
            end_time_col is not None):
            df = Utilities_dt.build_utc_time_column(
                df            = df, 
                time_col      = [start_time_col, end_time_col], 
                placement_col = [start_time_utc_col, end_time_utc_col])
        # Change timezoneoffset_col from string to datetime.timedelta
        #   e.g., '-04:00' ==> datetime.timedelta object = -1 days +20:00:00
        if timezoneoffset_col is not None:
            df = Utilities_dt.convert_timezoneoffset_col_to_timedelta(
                df                 = df,
                timezoneoffset_col = timezoneoffset_col
            ) 
        # #----------------------------
        # Drop any na values in the value_cols columns
        if drop_na_values and value_cols is not None:
            df = df.dropna(subset=value_cols)
        # #----------------------------
        # Set the index equal to the column created in Utilities_dt.convert_timestamp_to_utc_in_df,
        # as this seems to be the most reliable
        if set_index_col is not None:
            df = df.set_index(set_index_col, drop=False).sort_index()
            df.index.name=index_name
        # #----------------------------
        return df    
    
            
    #*****************************************************************************************************************        
    @staticmethod
    def combine_kwh_delivered_and_received_values(
        df, 
        aep_derived_uom_col      = 'aep_derived_uom', 
        aep_srvc_qlty_idntfr_col = 'aep_srvc_qlty_idntfr', 
        value_cols               = ['value'], 
        merge_and_groupby_cols   = ['serialnumber', 'aep_endtime_utc'], 
        keep_rec_and_del_cols    = False, 
        sort_index               = True
    ):
        # df should either be a pd.DataFrame with only aep_derived_uom=='KWH'
        # or else it needs to contain aep_derived_uom_col so that case of =='KWH'
        # can be differentiated from others.
        if aep_derived_uom_col in df.columns:
            df_kwh        = df[df[aep_derived_uom_col]=='KWH']
            df_other_uoms = df[df[aep_derived_uom_col]!='KWH']
        else:
            df_kwh = df
            df_other_uoms = None
        #-------------------------
        # Should only find aep_srvc_qlty_idntfr_col values 'DELIVERED', 'RECEIVED', 
        # and 'TOTAL' in df_kwh (maybe not all 3, hence difference instead of symmetric_difference)
        kwh_aep_srvc_qlty_idntfrs = set(['DELIVERED', 'RECEIVED', 'TOTAL'])
        assert(len(set(df_kwh[aep_srvc_qlty_idntfr_col].unique()).difference(kwh_aep_srvc_qlty_idntfrs))==0)
        
        # First, drop any duplicate entries
        df_kwh = df_kwh.drop_duplicates()

        # Match all 'DELIVERED' with 'RECEIVED' to form TOTAL
        # NOTE: Some entries may already have 'aep_srvc_qlty_idntfr' = 'TOTAL'
        #       Store these in df_tot_0, these will be added to combined results at end
        df_del   = df_kwh[df_kwh[aep_srvc_qlty_idntfr_col]=='DELIVERED']
        df_rec   = df_kwh[df_kwh[aep_srvc_qlty_idntfr_col]=='RECEIVED']
        df_tot_0 = df_kwh[df_kwh[aep_srvc_qlty_idntfr_col]=='TOTAL']

        df_del   = df_del.rename(columns={x:f'{x}_delivered' for x in value_cols})
        df_rec   = df_rec.rename(columns={x:f'{x}_received' for x in value_cols})
        df_tot_0 = df_tot_0.rename(columns={x:f'{x}_total' for x in value_cols})

        # Merge the delivered and received DataFrames.  This will allow me to easily
        # calculate the difference in delivered and received for each
        # NOTE: Have to use the index treament below because the merge operation
        #         does not maintain the index, so this is the workaround.
        #       Cannot simply use df_del.reset_index().merge(...).set_index('index') because the
        #         name of the index is typically not None, so the name generated by reset_index will 
        #         not be 'index', but rather the original name
        og_idx_name = df_del.index.name
        if og_idx_name is None:
            og_idx_name = 'index'
        df_tot_1 = df_del.reset_index().merge(
            df_rec[merge_and_groupby_cols + [f'{x}_received' for x in value_cols]], 
            left_on  = merge_and_groupby_cols, 
            right_on = merge_and_groupby_cols
        ).set_index(og_idx_name)
        df_tot_1[aep_srvc_qlty_idntfr_col] = 'TOTAL'
        for value_col in value_cols:
            df_tot_1[f'{value_col}_total'] = df_tot_1[f'{value_col}_delivered']-df_tot_1[f'{value_col}_received']

        # Combine newly combined results (df_tot_1) with pre-existing TOTAL results (df+_tot_0)
        assert(all(df_tot_0.columns==df_tot_0.columns))
        df_tot = pd.concat([df_tot_0, df_tot_1])

        # There should only be a single value column for each unique combination of merge_and_groupby_cols
        if df_tot.groupby(merge_and_groupby_cols).ngroups != df_tot.shape[0]:
            print(f'df_tot.groupby(merge_and_groupby_cols).ngroups = {df_tot.groupby(merge_and_groupby_cols).ngroups}')
            print(f'df_tot.shape[0] = {df_tot.shape[0]}')
        assert(df_tot.groupby(merge_and_groupby_cols).ngroups== df_tot.shape[0])

        df_tot   = df_tot.rename(columns={f'{x}_total':x for x in value_cols})
        if not keep_rec_and_del_cols:
            df_tot = df_tot.drop(columns=[f'{x}_delivered' for x in value_cols] + [f'{x}_received' for x in value_cols])
        #-------------------------
        # Combine back in all entries from df_other_uoms
        if df_other_uoms is not None:
            # In order to combine, if keep_rec_and_del_cols then df_other_uoms
            # will need the columns [f'{x}_delivered' for x in value_cols] + [f'{x}_received' for x in value_cols]
            if keep_rec_and_del_cols:
                df_other_uoms = df_other_uoms.copy() # If not copy, get warning about value trying to be set on slice
                df_other_uoms[[f'{x}_delivered' for x in value_cols] + 
                              [f'{x}_received' for x in value_cols]] = np.nan
            assert(len(set(df_tot.columns).symmetric_difference(set(df_other_uoms.columns)))==0)
            df_tot = pd.concat([df_tot, df_other_uoms[df_tot.columns]])
        #-------------------------
        # Combination operations throw off sorting
        if sort_index:
            df_tot = df_tot.sort_index()
        return df_tot
    
    
    #*****************************************************************************************************************
    @staticmethod
    def assemble_kwh_vlt_dfs_from_saved_csvs(
        file_dir                           , 
        glob_pattern                       , 
        cols_of_interest                   , 
        drop_na_values                     = True, 
        keep_rec_and_del_cols              = False, 
        combine_kwh_delivered_and_received = True, 
        verbose                            = False, 
        **kwargs
    ):
        r"""
        Performs standard cleaning and initialization operations.
        These include:
          1. From the epoch UTC time in timestamp_col, create a human-readable datetime column
          2. Build UTC versions of start_time_col and end_time_col
          3. Change timezoneoffset_col from string to datetime.timedelta
          4. Drop any na values in the value_cols columns (if drop_na_values==True)
          5. Set the index equal to the column created in Utilities_dt.convert_timestamp_to_utc_in_df
             (as this seems to be the most reliable)
        -------------------------
        Acceptable kwargs:
          ********************
          *** AMINonVee.perform_std_initiation_and_cleaning kwargs (see function for most up to date list)
          - timestamp_col 
            - default: aep_endtime_utc
          - utc_from_timestamp_col
            - default: f'{timestamp_col}_from_timestamp'

          - start_time_col
            - default: starttimeperiod
          - start_time_utc_col
            - default: f'{start_time_col}_utc'

          - end_time_col
            - default: endtimeperiod
          - end_time_utc_col
            - default: f'{end_time_col}_utc'

          - timezoneoffset_col
            - default: timezoneoffset

          - value_cols
            - default: ['value']

          - set_index_col
            -  default: utc_from_timestamp_col
            -  Setting equal to None will cause no index to be set
          - index_name
            - default: 'time_idx'

          ********************
          *** Additional kwargs
            - aep_derived_uom_col
              - default: aep_derived_uom

            - aep_srvc_qlty_idntfr_col
              - default: aep_srvc_qlty_idntfr

            - merge_and_groupby_cols
              - default: ['serialnumber', 'aep_endtime_utc']
        """
        #-------------------------
        # Unpack/set default kwargs
        kwargs['timestamp_col']          = kwargs.get('timestamp_col', 'aep_endtime_utc')
        kwargs['utc_from_timestamp_col'] = kwargs.get('utc_from_timestamp_col', f"{kwargs['timestamp_col']}_from_timestamp")

        kwargs['start_time_col']         = kwargs.get('start_time_col', 'starttimeperiod')
        kwargs['start_time_utc_col']     = kwargs.get('start_time_utc_col', f"{kwargs['start_time_col']}_utc")

        kwargs['end_time_col']           = kwargs.get('end_time_col', 'endtimeperiod')
        kwargs['end_time_utc_col']       = kwargs.get('end_time_utc_col', f"{kwargs['end_time_col']}_utc")

        kwargs['timezoneoffset_col']     = kwargs.get('timezoneoffset_col', 'timezoneoffset')

        kwargs['value_cols']             = kwargs.get('value_cols', ['value'])

        kwargs['set_index_col']          = kwargs.get('set_index_col', kwargs['utc_from_timestamp_col'])
        kwargs['index_name']             = kwargs.get('index_name', 'time_idx')
        ##-----
        aep_derived_uom_col              = kwargs.get('aep_derived_uom_col', 'aep_derived_uom')
        aep_srvc_qlty_idntfr_col         = kwargs.get('aep_srvc_qlty_idntfr_col', 'aep_srvc_qlty_idntfr')
        merge_and_groupby_cols           = kwargs.get('merge_and_groupby_cols', ['serialnumber', 'aep_endtime_utc'])
        #-------------------------
        aep_derived_uom_col              = kwargs.get('aep_derived_uom_col', 'aep_derived_uom') 
        aep_srvc_qlty_idntfr_col         = kwargs.get('aep_srvc_qlty_idntfr_col', 'aep_srvc_qlty_idntfr')
        merge_and_groupby_cols           = kwargs.get('merge_and_groupby_cols', ['serialnumber', 'aep_endtime_utc'])
        #----------------------------
        assert(kwargs['timestamp_col'] in cols_of_interest)
        assert(kwargs['start_time_col'] in cols_of_interest)
        assert(kwargs['end_time_col'] in cols_of_interest)
        assert(kwargs['timezoneoffset_col'] in cols_of_interest)
        for col in kwargs['value_cols']:
            assert(col in cols_of_interest)
        assert(aep_derived_uom_col in cols_of_interest)
        assert(aep_srvc_qlty_idntfr_col in cols_of_interest)
        for col in merge_and_groupby_cols:
            assert(col in cols_of_interest)
        #----------------------------
        csvs = Utilities.find_all_paths(base_dir=file_dir, glob_pattern=glob_pattern)    
        #----------------------------
        dfs_full = []
        for csv in csvs:
            if verbose:
                print('Reading file: ', csv)
            df = pd.read_csv(csv)
            df = clm.remove_prepend_from_columns_in_df(df)
            df = df[cols_of_interest]
            if df.shape[0]==0:
                continue
            dfs_full.append(df)

        df_full_15T = pd.concat(dfs_full)
        df_full_15T = AMINonVee.perform_std_initiation_and_cleaning(
            df             = df_full_15T, 
            drop_na_values = drop_na_values, 
            inplace        = True, 
            **kwargs
        )
        df_vlt_15T = df_full_15T[df_full_15T[aep_derived_uom_col]=='VOLT'].copy()
        df_kwh_15T = df_full_15T[df_full_15T[aep_derived_uom_col]=='KWH'].copy()
        if combine_kwh_delivered_and_received:
            df_kwh_15T = AMINonVee.combine_kwh_delivered_and_received_values(
                df                       = df_kwh_15T, 
                aep_srvc_qlty_idntfr_col = aep_srvc_qlty_idntfr_col, 
                value_cols               = kwargs['value_cols'], 
                merge_and_groupby_cols   = merge_and_groupby_cols, 
                keep_rec_and_del_cols    = keep_rec_and_del_cols
            )
        #----------------------------
        return {'kwh':df_kwh_15T, 'vlt':df_vlt_15T, 'full':df_full_15T}
    
    
    #*****************************************************************************************************************
    # TODO: Do I need to update this function to use t_int_beg_col, t_int_end_col instead of time_col?
    @staticmethod
    def group_ami_df_by_PN(
        ami_df                       , 
        SN_col                       = 'serialnumber', 
        PN_col                       = 'aep_premise_nb', 
        time_col                     = 'starttimeperiod_local', 
        value_col                    = 'value', 
        include_index_in_shared_cols = True, 
        gpby_dropna                  = True
    ):
        r"""
        Relatively simple-minded method to group ami_df by premise number (i.e., instead of the possibility of
          multiple entries for a premise at a given datetime corresponding the multiple meters on the premise, the
          DF will be collapsed down to a single value for the premise).
        The DF will be grouped by PN_col and time_col.
        All columns outside of SN_col, PN_col, time_col, and value_col must have a single value per group (i.e., per
          unique combination of PN_col and time_col).
        The value_col will be averaged.
        The SN_col will be dropped

        include_index_in_shared_cols:
            As various elements will be grouped/rows collapsed/however you want to think about it, the index will
              necessarily be lost, 
            In many cases (e.g., when the indices are time stamps) the indices for those being combined should be 
              shared (but, pandas doesn't know that).
            If this is the case, set include_index_in_shared_cols==True
            NOTE: For the foreseeable applications, this only really makes sense if the indices are datetime objects,
                  so this will be enforced.
        """
        #-------------------------
        # Make sure all of the needed columns are present
        assert(len(set([SN_col, PN_col, time_col, value_col]).difference(set(ami_df.columns.tolist())))==0)

        #-------------------------
        grp_by_cols = [PN_col, time_col]
        cols_shared_by_groups = [x for x in ami_df.columns.tolist() 
                                 if x not in grp_by_cols+[SN_col, value_col]]
        return_col_order = [x for x in ami_df.columns.tolist() if x != SN_col]

        #-------------------------
        # Really, the aggregation operation only needs to be performed on PNs with more than one SN
        # To save resources and time, split apart those needing reduced and those not
        #-----
        # To be most correct, for this application one should include time_col, so this is slightly different to what is 
        #   done in combine_PNs_in_best_ests_df
        SNs_per_PN_and_time = ami_df[[SN_col, PN_col, time_col]].drop_duplicates()[[PN_col, time_col]].value_counts()
        PNs_w_mult_SNs      = SNs_per_PN_and_time[SNs_per_PN_and_time>1].index.get_level_values(0).unique().tolist()

        # If there are no PNs with multiple SNs, return ami_df (without SN_col, as the purpose of this
        #   is to eliminate SN_col by grouping by premise)
        if len(PNs_w_mult_SNs)==0:
            return ami_df.drop(columns=[SN_col]).sort_values(by=[time_col, PN_col])

        ami_df_w_mult  = ami_df[ami_df[PN_col].isin(PNs_w_mult_SNs)]
        ami_df_wo_mult = ami_df[~ami_df[PN_col].isin(PNs_w_mult_SNs)]
        assert(ami_df_w_mult.shape[0]+ami_df_wo_mult.shape[0]==ami_df.shape[0])

        #-------------------------
        if include_index_in_shared_cols:
            # As various elements will be grouped/rows collapsed/however you want to think about it, the index will
            #   necessarily be lost, even though the indices for those being combined should be shared (but, pandas
            #   doesn't know that).
            # In order to retain the index, stored it in a temporary column and re-set it later
            # NOTE: This assumes there is a single index level.  If MultiIndex, method will need re-worked
            assert(ami_df.index.nlevels==1)
            assert(is_datetime64_dtype(ami_df.index.dtype))
            if ami_df.index.name is not None and ami_df.index.name not in ami_df.columns.tolist():
                tmp_idx_col = ami_df.index.name
            else:
                tmp_idx_col = Utilities.generate_random_string()
            ami_df_w_mult = ami_df_w_mult.reset_index(drop=False, names=tmp_idx_col)
            cols_shared_by_groups.append(tmp_idx_col)

        #-------------------------
        # Make sure all columns in cols_shared_by_groups have a single value per group
        assert((ami_df_w_mult.groupby(grp_by_cols, dropna=gpby_dropna)[cols_shared_by_groups].nunique()<=1).all().all())

        #-------------------------
        # Perform aggregation
        return_df = ami_df_w_mult.groupby(
            grp_by_cols, dropna=gpby_dropna, as_index=False, group_keys=False
        ).agg(
            {x:'first' for x in cols_shared_by_groups}|
            {value_col:'mean'}
        )

        #-------------------------
        if include_index_in_shared_cols:
            # Set index back to original values
            return_df = return_df.set_index(tmp_idx_col, drop=True)

        #-------------------------
        # Make sure columns are as expected
        assert(len(set(return_df.columns.tolist()).symmetric_difference(set(return_col_order)))==0)
        return_df = return_df[return_col_order]

        #-------------------------
        # Combine together with collection of PNs having a single SN
        # NOTE: The SN_col should not be present in the final output
        ami_df_wo_mult = ami_df_wo_mult.drop(columns=[SN_col])
        assert(return_df.columns.tolist()==ami_df_wo_mult.columns.tolist())
        return_df = pd.concat([return_df, ami_df_wo_mult])

        #-------------------------
        return_df = return_df.sort_values(by=[time_col, PN_col])
        return return_df

    #*****************************************************************************************************************
    @staticmethod
    def get_n_unique_per_column(
        df
    ):
        return df.nunique().to_dict()

    @staticmethod
    def get_cols_with_single_unique_value(
        df                         , 
        include_cols_with_all_nans = False
    ):
        if include_cols_with_all_nans:
            return df.columns[df.nunique()<=1].tolist()
        else:
            return df.columns[df.nunique()==1].tolist()

    @staticmethod
    def are_other_cols_to_keep_appropriate(
        df, 
        group_by, 
        other_cols_to_keep
    ):
        for idx, grp_df in df.groupby(group_by):
            col_n_unique_dict = AMINonVee.get_n_unique_per_column(df = grp_df)
            for col in other_cols_to_keep:
                if col_n_unique_dict[col]>1:
                    return False
        return True

    def decide_which_other_cols_to_keep(
        df                         , 
        group_by                   , 
        include_cols_with_all_nans = False, 
        check_first_n              = 5
    ):
        # If check_first_n=None, all groups will be checked
        # This can take a long time, so it is not suggested
        # ------------------------------------------
        # As an initial value, set other_cols_to_keep equal to all columns in df EXCEPT any
        # in group_by.  The reason for including all first is because I will use a set intersection
        # method below.
        # Note: group_by should either be the name of a column, or a list of column names
        if isinstance(group_by, str):
            group_by = [group_by]
        assert(isinstance(group_by, list) or isinstance(group_by, tuple))
        other_cols_to_keep = set(df.columns) - set(group_by)
        for i,(idx, grp_df) in enumerate(df.groupby(group_by)):
            if check_first_n is not None and i>=check_first_n:
                break
            #print(i)
            cols_with_single_unique_value = AMINonVee.get_cols_with_single_unique_value(
                df                         = df, 
                include_cols_with_all_nans = include_cols_with_all_nans
            )
            other_cols_to_keep = other_cols_to_keep.intersection(set(cols_with_single_unique_value))
        # Using set will not maintain the original order of the columns, which is likely desired
        # Therefore, convert other_cols_to_keep to list and sort as in df
        other_cols_to_keep = sorted(other_cols_to_keep, key = lambda x: list(df.columns).index(x))
        return other_cols_to_keep
    
    
    #*****************************************************************************************************************
    @staticmethod
    #TODO Probably don't use ensure_other_cols_to_keep_are_appropriate or decide_other_cols_to_keep
    #     Either improve methods or remove functionality
    def build_aggregate_of_meters_df(
        df_15T, time_col_for_agg                  , 
        agg_cols                                  = None, 
        agg_types                                 = None, 
        agg_dict                                  = None, 
        identifier                                = '_mtrs', 
        other_cols_to_keep                        = [], 
        flatten_columns                           = True, 
        flatten_index                             = True, 
        ensure_other_cols_to_keep_are_appropriate = False, 
        decide_other_cols_to_keep                 = False
    ):
        # Previously was build_df_aggregated_for_each_time_index
        # TODO should I implement methods allowing one to group by index?  e.g. through  df.groupby(level=0)
        #---------------------------------
        # This groups the DataFrame by the time (in time_col_for_agg) and outputs a DataFrame built by aggregating 
        #   each group by the functions specified in agg_types.
        # Typically, this corresponds an average over all meters in the collection for each time index.
        # -----
        # The time column by which to group is specified in time_col_for_agg.
        #   Note: If the timestamp is in the index of df_15T, the easiest method will be to give the index
        #           a name (e.g. df_15T.index.name = 'time_idx'), and then feed the index name into this 
        #           function via time_col_for_agg.
        # -----
        # The columns to be aggregated are specified in agg_cols.
        # -----
        # The aggregate functions to use are specified in agg_types
        #   e.g. agg_types = ['mean', 'sum', 'std']
        # -----
        # If other_cols_to_keep=[] or other_cols_to_keep=None, only time_col_for_agg (as the index) and agg_cols 
        #   will be contained in df_15T_agg.
        # If other_cols_to_keep are included, these columns should only have a single unique value for each 
        #   group, as the value from the first row in the group will be used
        # The user can ensure other_cols_to_keep are appropriate by setting 
        #   ensure_other_cols_to_keep_are_appropriate=True
        # The user can let the program decide which columns to keep by setting 
        #   decide_other_cols_to_keep=True
        #   Note: if decide_other_cols_to_keep=True, anything provided in other_cols_to_keep will 
        #         be ignored
        #-------------------------------------------------------------------------
        # For the code below, assume the following to understand explanations:
        # time_col_for_agg='endtimeperiod_utc'
        # agg_cols = ['value']
        # agg_types=['mean', 'sum']
        # other_cols_to_keep = ['srvc_pole_nb', 'aep_derived_uom', 'aep_srvc_qlty_idntfr']
        #-------------------------------------------------------------------------
        # agg_dict can be supplied for greater control.  If agg_dict is not None,
        #   agg_cols and agg_types will be ignored
        #
        # Need either agg_cols and agg_types OR agg_dict
        assert((agg_cols and agg_types) or agg_dict)
        #-------------------------------------------------------------------------
        if other_cols_to_keep is None:
            other_cols_to_keep=[]
        if decide_other_cols_to_keep:
            other_cols_to_keep = AMINonVee.decide_which_other_cols_to_keep(df_15T, time_col_for_agg)
        if ensure_other_cols_to_keep_are_appropriate and not decide_other_cols_to_keep:
            # Don't need to check if AMINonVee.decide_which_other_cols_to_keep run already
            assert(AMINonVee.are_other_cols_to_keep_appropriate(df_15T, time_col_for_agg, other_cols_to_keep))
        #---------------------------        
        # Create initial agg_dict, which will cause agg_types to be applied to all columns in agg_cols
        # and the 'first' aggregate function to be used on all columns in other_cols_to_keep
        # --- The assumption is each grouping has only a single unique value for each
        #     column in other_cols_to_keep, therefore using 'first' is appropriate
        if agg_dict is None:
            if isinstance(agg_types, str):
                agg_types = [agg_types]
            # There should be no overlap between the columns to be aggregated and other_cols_to_keep
            assert(len(set(agg_cols).intersection(set(other_cols_to_keep)))==0)
            agg_dict = {**{x:agg_types for x in agg_cols}, **{x:'first' for x in other_cols_to_keep}}
        else:
            # There should be no overlap between the columns to be aggregated and other_cols_to_keep and other_grouper_cols
            assert(len(set(agg_dict.keys()).intersection(set(other_cols_to_keep)))==0)
            agg_dict = {**agg_dict, **{x:'first' for x in other_cols_to_keep}}
        df_15T_agg = df_15T.groupby(time_col_for_agg).agg(agg_dict)
        #--------------------------------------------------------------
        # If the identifier exists, append to measurement name, which is in the newly created and lowest
        # level of the multiindex columns
        df_15T_agg.columns = df_15T_agg.columns.set_levels([f'{x}{identifier}' for x in df_15T_agg.columns.levels[-1]], level=-1)

        # Flatten down the columns
        # After aggregating, the columns of df_15T_agg become multi-levelled (i.e. becomes a multiindex)
        # level 0: the original name of the column
        # level 1: the corresponding aggregate function applied
        # ===> column = 'value' --> columns [('value', 'mean'), ('value', 'sum')]
        # The code below flattens down the columns and names them 'orignal name' + ' agg function'
        # ===> ('value', 'mean') --> mean value
        # ===> ('value', 'sum') --> sum value
        # For columns not included in aggregation (i.e. those not in agg_cols, for which the 'first'
        #   aggregation funciton was used), the original column name is retained
        #TODO DELETE OLD FLATTEN SEPARATE METHODS
    #     if flatten_columns:
    #         df_15T_agg = Utilities_df.flatten_multiindex_columns(df_15T_agg, join_str = ' ', reverse_order=True, to_ignore=['first'])
    #     if flatten_index:
    #         df_15T_agg = Utilities_df.flatten_multiindex_index(df_15T_agg)        
        df_15T_agg = Utilities_df.flatten_multiindex(
            df_15T_agg, 
            flatten_columns = flatten_columns, 
            flatten_index   = flatten_index, 
            inplace         = True, 
            index_args      = {}, 
            column_args     = dict(join_str=' ', reverse_order=True, to_ignore=['first'])
        )

        return df_15T_agg
    
    
    
    #*****************************************************************************************************************
    @staticmethod
    def build_time_resampled_df(
        df_15T             , 
        freq               , 
        other_grouper_cols , 
        agg_cols           = None, 
        agg_types          = None, 
        agg_dict           = None, 
        other_cols_to_keep = [], 
        flatten_columns    = True, 
        flatten_index      = True, 
        identifier         = '_TRS', 
        base_freq          = '15min'
    ):
        # Previously was build_resampled_df
        #-----------------------------------
        # NOTE: freq can be a single frequency (of type str, e.g. 'H')
        #         In this case, a single pd.DataFrame is returned
        #       OR freq can be a list of frequencies (e.g., ['H', '4H', 'D', 'MS'])
        #         In this case, a dict is returned whose keys are the frequencies
        #         and whose values are the corresponding pd.DataFrames
        #
        # agg_dict can be supplied for greater control.  If agg_dict is not None,
        #   agg_cols and agg_types will be ignored
        #
        # Need either agg_cols and agg_types OR agg_dict
        assert((agg_cols and agg_types) or agg_dict)

        assert(isinstance(freq, str) or isinstance(freq, list))
        if isinstance(freq, list):
            return_dict = {}
            # If building multiple resampled dfs, is oftentimes convenient to also have orignal freq
            # Therefore, it is included by default
            if base_freq not in freq:
                freq = [base_freq] + freq
                # Note: Previously had: freq.insert(0,base_freq)
                #       However, this alters freq outside of this function, which is probably not desired
            for f in freq:
                assert(f not in return_dict)
                if f==base_freq:
                    # base_freq df (df_15T) needs to be flattened to be consistent with how others are formatted            
                    return_dict[base_freq] = Utilities_df.flatten_multiindex(
                        df_15T, 
                        flatten_columns = flatten_columns, 
                        flatten_index   = flatten_index, 
                        inplace         = False, 
                        index_args      = {}, 
                        column_args     = dict(join_str=' ', reverse_order=True, to_ignore=['first'])
                    )
                    # If flatten_index==False, then the dfs in return_dict will in general have a MultiIndex index.
                    # However, the base_freq df in return_dict will not, as it is not put through the same
                    #   groupby aggregation as all others.
                    # Therefore, to make the format of base_freq df consistent with the others, one must use set_index
                    #   method below (append=True so that orignal level 0 time_index is not removed)
                    #
                    # NOTE: A similar thing can not be done for columns as base_freq df is not aggregated, so columns 
                    #       such as ('mean', 'value') don't make sense (base_freq df will only have e.g., a column 'value')
                    if not flatten_index:
                        return_dict[base_freq] = return_dict[base_freq].set_index(other_grouper_cols, append=True)
                else:
                    return_dict[f] = AMINonVee.build_time_resampled_df(
                        df_15T             = df_15T, 
                        freq               = f, 
                        other_grouper_cols = other_grouper_cols, 
                        agg_cols           = agg_cols, 
                        agg_types          = agg_types, 
                        agg_dict           = agg_dict, 
                        other_cols_to_keep = other_cols_to_keep, 
                        flatten_columns    = flatten_columns, 
                        flatten_index      = flatten_index, 
                        identifier         = identifier
                    )
            return return_dict
        #-----------------------------------
        #-----------------------------------
        # Create initial agg_dict, which will cause agg_types to be applied to all columns in agg_cols
        # and the 'first' aggregate function to be used on all columns in other_cols_to_keep
        # --- The assumption is each grouping has only a single unique value for each
        #     column in other_cols_to_keep, therefore using 'first' is appropriate
        if agg_dict is None:
            if isinstance(agg_types, str):
                agg_types = [agg_types]
            # There should be no overlap between the columns to be aggregated and other_cols_to_keep and other_grouper_cols
            assert(len(set(agg_cols).intersection(set(other_grouper_cols)))==0)
            assert(len(set(agg_cols).intersection(set(other_cols_to_keep)))==0)
            agg_dict = {**{x:agg_types for x in agg_cols}, **{x:'first' for x in other_cols_to_keep}}
        else:
            # There should be no overlap between the columns to be aggregated and other_cols_to_keep and other_grouper_cols
            assert(len(set(agg_dict.keys()).intersection(set(other_grouper_cols)))==0)
            assert(len(set(agg_dict.keys()).intersection(set(other_cols_to_keep)))==0)
            agg_dict = {**agg_dict, **{x:'first' for x in other_cols_to_keep}}
        df_resampled = df_15T.groupby([pd.Grouper(freq=freq)] + other_grouper_cols).agg(agg_dict)

        #--------------------------------------------------------------
        # If the identifier exists, append to measurement name, which is in the newly created and lowest
        # level of the multiindex columns
        df_resampled.columns = df_resampled.columns.set_levels([f'{x}{identifier}' for x in df_resampled.columns.levels[-1]], level=-1)

        #--------------------------------------------------------------
        # Flatten down the columns
        # After aggregating, the columns of df_resampled become multi-levelled (i.e. becomes a multiindex)
        # level 0: the original name of the column
        # level 1: the corresponding aggregate function applied
        # ===> column = 'value' --> columns [('value', 'mean'), ('value', 'sum')]
        # The code below flattens down the columns and names them 'orignal name' + ' agg function'
        # ===> ('value', 'mean') --> mean value
        # ===> ('value', 'sum') --> sum value
        # For columns not included in aggregation (i.e. those not in agg_cols, for which the 'first'
        #   aggregation funciton was used), the original column name is retained
        #TODO DELETE OLD FLATTEN SEPARATE METHODS
    #     if flatten_columns:
    #         df_resampled = Utilities_df.flatten_multiindex_columns(df_resampled, join_str = ' ', reverse_order=True, to_ignore=['first'])
    #     if flatten_index:
    #         df_resampled = Utilities_df.flatten_multiindex_index(df_resampled)        
        df_resampled = Utilities_df.flatten_multiindex(
            df_resampled, 
            flatten_columns = flatten_columns, 
            flatten_index   = flatten_index, 
            inplace         = True, 
            index_args      = {}, 
            column_args     = dict(join_str=' ', reverse_order=True, to_ignore=['first'])
        )

        return df_resampled
    
    
    #*****************************************************************************************************************
    @staticmethod
    # THIS IS REALLY UGLY AND HARD TO FOLLOW, BUT IT WORKS EXACTLY AS INTENDED
    # SHOULD GO BACK AND ADD NOTES AND DESCRIPTION
    def get_agg_and_rename_dicts_for_agg_rounds(
        agg_cols, 
        agg_types, 
        mix_agg_functions , 
        n_rounds          = 2, 
        identifiers       = None, 
        join_str          = ' ', 
        reverse_order     = True, 
        to_ignore         = ['first']
    ):
        # all_rounds_agg_dicts[i] is the agg dictionary needed to run round_i
        # all_rounds_rename_dicts[i] is the resulting rename which occurs after round one if flattened
        #
        # Note: Keys in all_rounds from 1 to n_rounds (not 0 to n_rounds-1)
        # identifiers are tags that will be added to the aggregate names in each round
        #   If identifiers is not None, it must be a list whose length equals at least n_rounds
        #   Typical case: First, aggregate of meters for each time.  Second, time resampling
        #               ---> identifiers = ['_mtrs', '_TRS']
        #                                  '_mtrs' represents aggregate of meters
        #                                  '_TRS' stands for Time ReSampled
        #--------------------------------------------------
        if identifiers is not None:
            assert(len(identifiers)>=n_rounds)
        all_rounds_rename_dicts = {}
        all_rounds_agg_dicts    = {}
        #--------------------------------------------------
        rd_1_rename_dict = []
        rd_1_agg_dict    = {}
        idfr=''
        if identifiers is not None:
            idfr = identifiers[1-1]
        for agg_col in agg_cols:
            assert(agg_col not in rd_1_agg_dict)
            rd_1_agg_dict[agg_col] = agg_types
            for agg_type in agg_types:
                agg_type  = f'{agg_type}{idfr}'
                curr_mult = (agg_col, agg_type)
                rd_1_rename_dict.append(curr_mult)
        rd_1_rename_dict = Utilities_df.get_flattened_multiindex_columns(
            rd_1_rename_dict, 
            join_str      = join_str, 
            reverse_order = reverse_order, 
            to_ignore     = to_ignore
        )
        #-----
        all_rounds_rename_dicts[1] = rd_1_rename_dict
        all_rounds_agg_dicts[1]    = rd_1_agg_dict
        #--------------------------------------------------
        for i_round in list(range(2,n_rounds+1)):
            rd_i_rename_dict   = []
            rd_i_agg_dict      = {}
            rd_im1_rename_dict = all_rounds_rename_dicts[i_round-1]
            idfr=''
            if identifiers is not None:
                idfr = identifiers[i_round-1]
            prev_idfr = identifiers[i_round-2]
            for prev_col_mult, prev_col_flat in rd_im1_rename_dict.items():
                agg_types_i = []
                for agg_type in agg_types:
                    prev_agg_type = prev_col_mult[1]
                    if not mix_agg_functions:                       
                        if prev_idfr:
                            found_prev_idfr = prev_agg_type.find(prev_idfr)
                            assert(found_prev_idfr>-1)
                            to_comp = prev_agg_type[:found_prev_idfr]
                        else:
                            to_comp = prev_agg_type
                        if to_comp!=agg_type:
                            continue
                    #-----
                    agg_types_i.append(agg_type)
                    curr_agg_type = f'{agg_type}{idfr}'
                    curr_mult     = (prev_col_flat, curr_agg_type)
                    rd_i_rename_dict.append(curr_mult)
                assert(prev_col_flat not in rd_i_agg_dict)
                rd_i_agg_dict[prev_col_flat] = agg_types_i
            rd_i_rename_dict = Utilities_df.get_flattened_multiindex_columns(
                rd_i_rename_dict, 
                join_str      = join_str, 
                reverse_order = reverse_order, 
                to_ignore     = to_ignore
            )
            #-----
            all_rounds_rename_dicts[i_round] = rd_i_rename_dict
            all_rounds_agg_dicts[i_round]    = rd_i_agg_dict
        return all_rounds_agg_dicts, all_rounds_rename_dicts
    
    
    #*****************************************************************************************************************
    @staticmethod
    def build_time_resampled_aggregate_of_meters_df(
        df_15T                 , 
        time_col_for_meter_agg , 
        freq                   = 'H', 
        agg_cols               = None, 
        agg_types              = None, 
        mix_agg_functions      = True, 
        agg_dict_meter_agg     = None, 
        agg_dict_TRS           = None, 
        other_cols_to_keep     = [], 
        base_freq              = '15min', 
        flatten_columns        = True, 
        flatten_index          = True, 
        identifiers            = ['_mtrs', '_TRS'], 
        **kwargs
    ):
        # Previously was build_resampled_df_aggregated_for_each_time_index
        # First, perform aggregation of meters
        # Then, time resample
        #-----------------------------------
        # Suggest using agg_cols and agg_types over agg_dict_meter_agg and agg_dict_TRS
        #-----------------------------------
        # Can supply own df_15T_agg if desired.  This functionality was added for the case of freq as list
        #   where multiple dfs are built.  For this case, df_15T_agg doesn't need to be built each time, 
        #   it only needs built once and used by all!
        #-----------------------------------
        # Time resample df_15T_agg
        # The intent of this function is to be used on a DataFrame built with build_aggregate_of_meters_df.
        #   - df_15T contains multiple meters for multiple time period
        #   - df_15T_agg = build_aggregate_of_meters_df(df_15T, ...) then contains aggregate meter values
        #     for each time period
        #   - This function will resample df_15T_agg to, e.g. df_H_agg (hourly)
        #-----------------------------------
        # NOTE: freq can be a single frequency (of type str, e.g. 'H')
        #         In this case, a single pd.DataFrame is returned
        #       OR freq can be a list of frequencies (e.g., ['H', '4H', 'D', 'MS'])
        #         In this case, a dict is returned whose keys are the frequencies
        #         and whose values are the corresponding pd.DataFrames
        #-----------------------------------
        # agg_dict_meter_agg and agg_dict_TRS can be supplied for greater control.  If both are not None,
        #   agg_cols and agg_types will be ignored
        #-----------------------------------
        # Need either agg_cols and agg_types OR agg_dict_meter_agg and agg_dict_TRS
        assert((agg_cols and agg_types) or (agg_dict_meter_agg and agg_dict_TRS))
        #-----------------------------------
        df_15T_agg = kwargs.get("df_15T_agg", None)
        if df_15T_agg is None:
            df_15T_agg = AMINonVee.build_aggregate_of_meters_df(
                df_15T                                    = df_15T, 
                time_col_for_agg                          = time_col_for_meter_agg, 
                agg_cols                                  = agg_cols, 
                agg_types                                 = agg_types, 
                agg_dict                                  = agg_dict_meter_agg, 
                identifier                                = identifiers[0], 
                other_cols_to_keep                        = other_cols_to_keep, 
                flatten_columns                           = flatten_columns, 
                flatten_index                             = flatten_index, 
                ensure_other_cols_to_keep_are_appropriate = False, 
                decide_other_cols_to_keep                 = False
            )    
        #-----------------------------------
        assert(isinstance(freq, str) or isinstance(freq, list))
        if isinstance(freq, list):
            return_dict = {}
            # If building multiple resampled dfs, is oftentimes convenient to also have orignal freq.
            # In fact, this is likely desired more times than not.
            # Therefore, if base_freq is not in freq, include it by default
            if base_freq not in freq:
                freq = [base_freq] + freq
                # Note: Previously had: freq.insert(0,base_freq)
                #       However, this alters freq outside of this function, which is probably not desired
            for f in freq:
                assert(f not in return_dict)
                if f==base_freq:
                    return_dict[f] = df_15T_agg
                else:
                    return_dict[f] = AMINonVee.build_time_resampled_aggregate_of_meters_df(
                        df_15T                 = df_15T, 
                        time_col_for_meter_agg = time_col_for_meter_agg, 
                        agg_cols               = agg_cols, 
                        agg_types              = agg_types, 
                        agg_dict_meter_agg     = agg_dict_meter_agg, 
                        agg_dict_TRS           = agg_dict_TRS, 
                        other_cols_to_keep     = other_cols_to_keep, 
                        freq                   = f, 
                        mix_agg_functions      = mix_agg_functions, 
                        df_15T_agg             = df_15T_agg
                    )
            return return_dict
        #-----------------------------------
        # For the code below, assume the following to understand explanations:
        # agg_cols = ['value']
        # agg_types=['mean', 'sum']
        # other_cols_to_keep = ['srvc_pole_nb', 'aep_derived_uom', 'aep_srvc_qlty_idntfr']
        #-----------------------------------
        # The same agg_types will be used to build df_15T_agg as df_resampled_agg.
        # mix_agg_functions allows the user to set how these two aggregations will work.
        # If mix_agg_functions=True:
        #   Final aggregate columns will be: 'mean mean value', 'sum mean value', 'mean sum value', 'sum sum value'
        # If mix_agg_functions=False:
        #   Final aggregate columns will be: 'mean mean value' and 'sum sum value'
        #-----------------------------------------------------------------------------

        if agg_dict_TRS is None:
            if isinstance(agg_types, str): 
                agg_types = [agg_types]
            all_rounds_agg_dicts, _ = AMINonVee.get_agg_and_rename_dicts_for_agg_rounds(
                agg_cols          = agg_cols, 
                agg_types         = agg_types, 
                mix_agg_functions = mix_agg_functions, 
                n_rounds          = 2, 
                identifiers       = identifiers, 
                join_str          = ' ', 
                reverse_order     = True, 
                to_ignore         = ['first']
            )    
            agg_dict_TRS = all_rounds_agg_dicts[2]
        df_resampled_agg = AMINonVee.build_time_resampled_df(
            df_15T_agg, 
            freq, 
            other_grouper_cols = [], 
            agg_dict           = agg_dict_TRS, 
            other_cols_to_keep = other_cols_to_keep, 
            flatten_columns    = flatten_columns, 
            flatten_index      = flatten_index, 
            identifier         = identifiers[1]
        )
        return df_resampled_agg
    
    
    #*****************************************************************************************************************
    @staticmethod
    def build_time_resampled_dfs(
        df_15T                 , 
        base_freq              = '15min', 
        freqs                  = ['H', '4H', 'D', 'MS'], 
        other_grouper_cols     = ['serialnumber'], 
        other_cols_to_keep     = [], 
        flatten_index          = True, 
        flatten_columns        = True, 
        build_agg_dfs          = True, 
        time_col_for_agg       = 'endtimeperiod_utc', 
        agg_cols               = ['value'], 
        agg_types              = ['mean'], 
        other_cols_to_keep_agg = [], 
        mix_agg_functions      = True, 
        df_key                 = 'df', 
        df_agg_key             = 'df_agg'
    ):
        # Was previously get_resampled_dfs
        #
        # If build_agg_dfs is False, this is essentially just a call to 
        # build_time_resampled_df with a list of frequencies,
        #---------------------
        # By default, when grouping the grouped columns become indices
        # If flatten_index = True, the indices will be flattened back out
        # to one-dimensional
        #   Note: As df_15T is not grouped, it does not need to be flattened
        #-------------------------------------------------------------
        return_dict = {}
        #------------------
        # Build resampled dfs and add to return_dict
        resampled_dfs_dict = AMINonVee.build_time_resampled_df(
            df_15T             = df_15T, 
            freq               = freqs, 
            other_grouper_cols = other_grouper_cols, 
            agg_cols           = agg_cols, 
            agg_types          = agg_types, 
            agg_dict           = None, 
            other_cols_to_keep = other_cols_to_keep, 
            flatten_columns    = flatten_columns, 
            flatten_index      = flatten_index, 
            identifier         = '_TRS', 
            base_freq          = base_freq
        )
        for freq in resampled_dfs_dict.keys():
            assert(freq not in return_dict and freq in [base_freq]+freqs)
            df_i = resampled_dfs_dict[freq]
            # Add 'date' column to each df, as that was done in first version of code
            # TODO: WHY?
            df_i['date'] = df_i.index.get_level_values(0)
            return_dict[freq] = {df_key:df_i}
        #------------------
        # Build resampled aggregate dfs if build_agg_dfs==True
        if build_agg_dfs:
            resampled_agg_dfs_dict = AMINonVee.build_time_resampled_aggregate_of_meters_df(
                df_15T                 = df_15T, 
                time_col_for_meter_agg = time_col_for_agg, 
                freq                   = freqs, 
                agg_cols               = agg_cols, 
                agg_types              = agg_types, 
                mix_agg_functions      = mix_agg_functions, 
                agg_dict_meter_agg     = None, 
                agg_dict_TRS           = None, 
                other_cols_to_keep     = other_cols_to_keep_agg, 
                base_freq              = base_freq, 
                flatten_columns        = flatten_columns, 
                flatten_index          = flatten_index
            )
            # Add resampled agg dfs to return_dict
            for freq in resampled_agg_dfs_dict.keys():
                assert(freq in return_dict)
                df_agg_i = resampled_agg_dfs_dict[freq]
                # Add index name and create time_col_for_agg for each df_agg, 
                # as that was done in first version of code
                df_agg_i.index.name='time_idx'
                df_agg_i[time_col_for_agg] = df_agg_i.index
                #-----
                return_dict[freq][df_agg_key] = df_agg_i
            #------------------        
        return return_dict
    

    #*****************************************************************************************************************
    @staticmethod
    def plot_continuous_usage(
        ax, 
        data, 
        x, 
        y, 
        hue, 
        data_sep        = pd.Timedelta('15min'), 
        palette         = 'colorblind', 
        data_label      = '',
        lineplot_kwargs = None
    ):
        r"""
        Only continuous portions of data[x] are connected with lines.
        Intended for use in plot_usage_around_outage

        Typical case:
            x        = 'starttimeperiod_local'
            y        = 'value'
            hue      = 'serialnumber'
            data_sep = pd.Timedelta('15min')


        palette:
            Must be either an acceptable argument to sns.color_palette OR a dict with keys equal to
              data[hue].unique().tolist() and values equal to colors
            In either case, after intial prep, palette will be a dict with the aforementioned keys.
            This is necessary to maintain color consistency between the different portions of the plot
            
        IMPORTANT: data_label is currently ignored!
                   There seems to be a bug in the current version of seaborn (0.13.2) where sns.lineplot cannot
                     accept a label argument.
                   If one is provided, an error is thrown, ending with:
                        TypeError: functools.partial(<class 'matplotlib.lines.Line2D'>, xdata=[], ydata=[]) got multiple values for keyword argument 'label'
                   I think/wonder if when hue is set, the values are automatically used as labels; therefore, supplying 
                     label additionally results in the error message above.
                   This is poor coding, and I expect it to be resolved in future releases.  But, for now, we muse adjust.
                   So, if hue is not None, then label cannot be supplied as an argument to sns.lineplot (even setting label=None causes error)
        """
        #-------------------------
        if lineplot_kwargs is None:
            lineplot_kwargs = {}
        #-------------------------
        if not isinstance(palette, dict):
            palette = Plot_General.get_standard_colors_dict(
                keys=data[hue].unique().tolist(), 
                palette=palette
            )
        assert(set(data[hue].unique().tolist()).difference(set(palette.keys()))==set())
        #-------------------------
        lineplot_kwargs_fnl = (
            lineplot_kwargs | 
            dict(
                x       = x, 
                y       = y, 
                hue     = hue, 
                palette = palette
            )
        )
        if hue is None:
            lineplot_kwargs_fnl['label'] = data_label
        #-------------------------
        for hue_i in data[hue].unique().tolist():
            data_i = data[data[hue]==hue_i].copy()
            #-----
            cont_blocks_i = Utilities_df.get_continuous_blocks_in_df(
                df               = data_i, 
                col_idfr         = x, 
                data_sep         = data_sep, 
                return_endpoints = True
            )
            #-----
            for block_beg_j, block_end_j in cont_blocks_i:
                sns.lineplot(
                    ax   = ax, 
                    data = data_i.iloc[block_beg_j:block_end_j+1], 
                    **lineplot_kwargs_fnl
                )
        #-------------------------
        Plot_General.remove_duplicates_from_legend(ax, ax_props=dict(title=hue));
    
    @staticmethod
    def plot_usage(
        fig, 
        ax, 
        data, 
        x, 
        y, 
        hue, 
        data_label            = '', 
        title_args            = None, 
        ax_args               = None, 
        xlabel_args           = None, 
        ylabel_args           = None, 
        df_mean               = None, 
        df_mean_col           = None, 
        mean_args             = None, 
        draw_without_hue_also = False, 
        seg_line_freq         = None, 
        palette               = 'colorblind'
    ):
        r"""
        Setting hue=None will aggregate over repeated values to show the mean and 95% confidence interval
        seg_line_freq can be set to e.g., seg_line_freq='D'
    
        IMPORTANT: data_label is currently ignored!
                   There seems to be a bug in the current version of seaborn (0.13.2) where sns.lineplot cannot
                     accept a label argument.
                   If one is provided, an error is thrown, ending with:
                        TypeError: functools.partial(<class 'matplotlib.lines.Line2D'>, xdata=[], ydata=[]) got multiple values for keyword argument 'label'
                   I think/wonder if when hue is set, the values are automatically used as labels; therefore, supplying 
                     label additionally results in the error message above.
                   This is poor coding, and I expect it to be resolved in future releases.  But, for now, we muse adjust.
                   So, if hue is not None, then label cannot be supplied as an argument to sns.lineplot (even setting label=None causes error)
        """
        #-------------------------
        if df_mean_col is None:
            df_mean_col = y
        #-------------------------
        lineplot_kwargs = dict(
            data    = data, 
            x       = x,
            y       = y, 
            hue     = hue, 
            palette = palette
        )
        if hue is None:
            lineplot_kwargs['label'] = data_label
        sns.lineplot(
            ax = ax, 
            **lineplot_kwargs
        )
        if draw_without_hue_also and hue is not None:
            sns.lineplot(
                ax        = ax, 
                data      = data, 
                x         = x, 
                y         = y, 
                hue       = None, 
                color     = 'deeppink', 
                linestyle = '--', 
                label     = 'AVG'
            )
        #----------------------------
        # Note: if hue=None is drawn, then average will already be drawn!
        if (df_mean is not None 
            and hue is not None 
            and not draw_without_hue_also):
            if mean_args is None:
                avg_label = 'AVG'
                if data_label:
                    avg_label = f'{data_label} AVG'
                mean_args = dict(style='--', linewidth=3, alpha=0.50, color='deeppink', label=avg_label, legend=True)
            df_mean[df_mean_col].plot(ax=ax, **mean_args)
        #----------------------------
        # if seg_line_freq is not None:
            # seg_min = pd.to_datetime(out_t_beg-expand_time).round(seg_line_freq)
            # seg_max = pd.to_datetime(out_t_end+expand_time).round(seg_line_freq)
            # all_segs = pd.date_range(seg_min, seg_max, freq=seg_line_freq)
            # for seg in all_segs:
                # ax.axvline(seg, color='black', linestyle='--');
        #----------------------------
        if isinstance(title_args, str):
            title_args = dict(label=title_args)
        if title_args is not None:
            ax.set_title(**title_args)
        #----------------------------
        if ax_args is not None:
            ax.set(**ax_args)
        if xlabel_args is not None:
            ax.set_xlabel(**xlabel_args)
        if ylabel_args is not None:
            ax.set_ylabel(**ylabel_args)

        return fig,ax
        
        
    @staticmethod
    def draw_outage_limits_on_ax(
        ax, 
        out_t_beg, 
        out_t_end, 
        plot_t_beg                 = None, 
        plot_t_end                 = None, 
        draw_outage_limits_kwargs  = None, 
        include_outage_limits_text = False, 
        out_t_beg_line_color       = 'red', 
        out_t_end_line_color       = 'green', 
        text_only                  = False
    ):
        r"""
        plot_t_beg/plot_t_end:
            If supplied, out_t_beg and out_t_end will only be drawn if within plot_t_beg/plot_t_end interval
        """
        #----------------------------------------------------------------------------------------------------
        if draw_outage_limits_kwargs is None:
            draw_outage_limits_kwargs = {}
        if text_only:
            assert(include_outage_limits_text)
        #--------------------------------------------------
        if plot_t_beg is None:
            plot_t_beg = pd.Timestamp.min
        if plot_t_end is None:
            plot_t_end = pd.Timestamp.max
        #--------------------------------------------------
        # Draw the lines
        if not text_only:
            if out_t_beg>=plot_t_beg and out_t_beg<=plot_t_end:
                ax.axvline(out_t_beg, color=out_t_beg_line_color, **draw_outage_limits_kwargs)
            if out_t_end>=plot_t_beg and out_t_end<=plot_t_end:
                ax.axvline(out_t_end, color=out_t_end_line_color, **draw_outage_limits_kwargs)
        #----------------------------------------------------------------------------------------------------
        dflt_include_outage_limits_text_dict = dict(
            include_out_t_beg_text = True, 
            out_t_beg_text         = 'Outage Beg.', 
            out_t_beg_ypos         = ax.get_ylim()[0], 
            out_t_beg_rot          = 90, 
            out_t_beg_va           = 'bottom', 
            out_t_beg_ha           = 'right', 
            out_t_beg_color        = 'black', 
            #----------
            include_out_t_end_text = True, 
            out_t_end_text         = 'Outage End ', 
            out_t_end_ypos         = ax.get_ylim()[0], 
            out_t_end_rot          = 90, 
            out_t_end_va           = 'bottom', 
            out_t_end_ha           = 'left', 
            out_t_end_color        = 'black', 
        )
        #-------------------------
        if include_outage_limits_text:
            if isinstance(include_outage_limits_text, dict):
                include_outage_limits_text = Utilities.supplement_dict_with_default_values(
                    to_supplmnt_dict    = include_outage_limits_text,
                    default_values_dict = dflt_include_outage_limits_text_dict,
                    extend_any_lists    = False,
                    inplace             = False,
                )
            else:
                include_outage_limits_text = dflt_include_outage_limits_text_dict
            #-------------------------
            include_out_t_beg_text = include_outage_limits_text['include_out_t_beg_text']
            out_t_beg_text  = include_outage_limits_text['out_t_beg_text']
            out_t_beg_ypos  = include_outage_limits_text['out_t_beg_ypos']
            out_t_beg_rot   = include_outage_limits_text['out_t_beg_rot']
            out_t_beg_va    = include_outage_limits_text['out_t_beg_va']
            out_t_beg_ha    = include_outage_limits_text['out_t_beg_ha']
            out_t_beg_color = include_outage_limits_text['out_t_beg_color']
            #----------
            include_out_t_end_text = include_outage_limits_text['include_out_t_end_text']
            out_t_end_text  = include_outage_limits_text['out_t_end_text']
            out_t_end_ypos  = include_outage_limits_text['out_t_end_ypos']
            out_t_end_rot   = include_outage_limits_text['out_t_end_rot']
            out_t_end_va    = include_outage_limits_text['out_t_end_va']
            out_t_end_ha    = include_outage_limits_text['out_t_end_ha']
            out_t_end_color = include_outage_limits_text['out_t_end_color']
            #-------------------------
            # If out_t_beg/out_t_end outside of plot_t_beg/plot_t_end, do not include the text, regardless
            #   of the values of include_out_t_beg_text/include_out_t_end_text
            if out_t_beg<plot_t_beg or out_t_beg>plot_t_end:
                include_out_t_beg_text=False
            if out_t_end<plot_t_beg or out_t_end>plot_t_end:
                include_out_t_end_text=False
            #-------------------------
            if Utilities.is_object_one_of_types(out_t_beg_ypos, [list, tuple]) and len(out_t_beg_ypos)==2:
                assert(out_t_beg_ypos[1] in ['eval_at_rt', 'ax_coord'])
                if out_t_beg_ypos[1]=='eval_at_rt':
                    out_t_beg_ypos = eval(out_t_beg_ypos[0])
                else:
                    out_t_beg_ypos = ax.transLimits.inverted().transform((0,out_t_beg_ypos[0]))[1]
            #-----
            if Utilities.is_object_one_of_types(out_t_end_ypos, [list, tuple]) and len(out_t_end_ypos)==2:
                assert(out_t_end_ypos[1] in ['eval_at_rt', 'ax_coord'])
                if out_t_end_ypos[1]=='eval_at_rt':
                    out_t_end_ypos = eval(out_t_end_ypos[0])
                else:
                    out_t_end_ypos = ax.transLimits.inverted().transform((0,out_t_end_ypos[0]))[1]            
            #-------------------------
            if include_out_t_beg_text:
                ax.text(
                    x                   = out_t_beg, 
                    y                   = out_t_beg_ypos, 
                    s                   = out_t_beg_text, 
                    rotation            = out_t_beg_rot, 
                    verticalalignment   = out_t_beg_va, 
                    horizontalalignment = out_t_beg_ha, 
                    color               = out_t_beg_color
                )
            if include_out_t_end_text:
                ax.text(
                    x                   = out_t_end, 
                    y                   = out_t_end_ypos, 
                    s                   = out_t_end_text, 
                    rotation            = out_t_end_rot, 
                    verticalalignment   = out_t_end_va, 
                    horizontalalignment = out_t_end_ha, 
                    color               = out_t_end_color
                )
        #-------------------------
        return ax
        
    
    @staticmethod
    def plot_usage_around_outage(
        fig, 
        ax, 
        data, 
        x, 
        y, 
        hue, 
        out_t_beg, 
        out_t_end, 
        expand_time, 
        plot_time_beg_end          = None, 
        only_connect_continuous    = False, 
        data_freq                  = pd.Timedelta('15min'), 
        data_label                 = '', 
        title_args                 = None, 
        ax_args                    = None, 
        xlabel_args                = None, 
        ylabel_args                = None, 
        df_mean                    = None, 
        df_mean_col                = None, 
        mean_args                  = None, 
        draw_outage_limits         = True, 
        draw_outage_limits_kwargs  = None, 
        include_outage_limits_text = False, 
        draw_without_hue_also      = False, 
        seg_line_freq              = None, 
        palette                    = 'colorblind',
        lineplot_kwargs            = None
    ):
        r"""
        NOTE: By default, out_t_beg-expand_time to out_t_end+expand_time will be plotted.
              If plot_time_beg_end is set, then plot_time_beg_end[0]-expand_time to plot_time_beg_end[1]+expand_time will be plotted.
        
        Seaborn must have implemented some dumb update that caused lineplot and others to not function properly with
          time series data with repeated indiced.
        Now, when there are duplicate time indices, the plot crashes with "ValueError: cannot reindex from a duplicate axis".
        It doesn't matter if hue is set, and even when each hue subgroup has unique indices, it still crashes.
        In the past, one could use duplicate indices, and Seaborn would aggregate over repeated values to show the mean and 95% 
          confidence interval.
        Due to these issues, I must call reset_index, and can no longer use the simpler:
            data.loc[out_t_beg-expand_time:out_t_end+expand_time]
          but must instead use:
            data[(data[x]>=out_t_beg-expand_time) & (data[x]<out_t_end+expand_time)]

        draw_outage_limits:
            Outage limits (and outage limits text) will only be drawn if outage is within the original/desired x-limits defined by 
              plot_t_beg/plot_t_end
              i.e., the x-limits will not be expanded to accomodate plotting the outage limits!
            
        include_outage_limits_text:
            Can be a boolean or a dict containing information on how to print text
            
        NOTE on out_t_beg_ypos/out_t_end_ypos:
            In many cases, it is advantageous to set this equal to a number which is normalized to the axes, i.e., 
              0 for bottom of the axis, and 1 for the top.
            If one wanted to use a transform on both the x and y values of ax.text, one could simply add the argument: transform=ax.transAxes
              or similar to the ax.text call.
              The problem here is that we want the x-value to be set on the coordinate of the data, but the y-value to be set on the coordinate
                of the axes, so one cannot simply use the transform argument in ax.text
            Typically, this can be done through an inverted transform; e.g., to set the y-position to -0.1 in the normalized
              coordinate system, one could use:
                ypos=ax.transLimits.inverted().transform((0,-0.1))[1]
            THE PROBLEM is that is one inputs the above line as an argument to this function, at the time of the input the axes are
              not typically formed, and are therefore only defined on a range (0,1), instead of the actual final range after the data are plotted.
            Thus, the transformed value must be grabbed after the data are plotted, which, of course, is impossible to do with an input argument at the
              time this function is called.
            THE SOLUTION:
                For out_t_beg_ypos/out_t_end_ypos, one can input a tuple of two elements.
                1.
                    The first element is a string representing the expression to be evaluated at run-time.
                    The second element is simply the string 'eval_at_rt' to ensure this is actually what the user wants done.
                    ==> e.g., 
                                include_outage_limits_text = dict(
                                    ..., 
                                    out_t_beg_ypos=('ax.transLimits.inverted().transform((0,-0.1))[1]', 'eval_at_rt'), 
                                    ...
                                )
                2.
                    The first elemet in a number (e.g., float)
                    The second element is the string 'ax_coord', which directs the code to interpret the value in the axis coordinate system
                      instead of the data coordinate system
                    ==> e.g., 
                                include_outage_limits_text = dict(
                                    ..., 
                                    out_t_beg_ypos=(0.1, 'ax_coord'), 
                                    ...
                                )
                                
        IMPORTANT: data_label is currently ignored!
                   There seems to be a bug in the current version of seaborn (0.13.2) where sns.lineplot cannot
                     accept a label argument.
                   If one is provided, an error is thrown, ending with:
                        TypeError: functools.partial(<class 'matplotlib.lines.Line2D'>, xdata=[], ydata=[]) got multiple values for keyword argument 'label'
                   I think/wonder if when hue is set, the values are automatically used as labels; therefore, supplying 
                     label additionally results in the error message above.
                   This is poor coding, and I expect it to be resolved in future releases.  But, for now, we muse adjust.
                   So, if hue is not None, then label cannot be supplied as an argument to sns.lineplot (even setting label=None causes error)

        """
        #-------------------------
        if lineplot_kwargs is None:
            lineplot_kwargs = {}
        if draw_outage_limits_kwargs is None:
            draw_outage_limits_kwargs = {}
        #-------------------------
        plot_t_beg = out_t_beg-expand_time
        plot_t_end = out_t_end+expand_time
        if plot_time_beg_end is not None:
            assert(Utilities.is_object_one_of_types(plot_time_beg_end, [list, tuple]))
            assert(len(plot_time_beg_end)==2)
            plot_t_beg = plot_time_beg_end[0]-expand_time
            plot_t_end = plot_time_beg_end[1]+expand_time        
        #-------------------------
        draw_data = data[(data[x]>=plot_t_beg) & (data[x]<=plot_t_end)]
        draw_data = draw_data.reset_index()
        #-------------------------
        # Setting hue=None will aggregate over repeated values to show the mean and 95% confidence interval
        # seg_line_freq can be set to e.g., seg_line_freq='D'
        if df_mean_col is None:
            df_mean_col = y
        if only_connect_continuous:
            AMINonVee.plot_continuous_usage(
                ax              = ax, 
                data            = draw_data, 
                x               = x, 
                y               = y, 
                hue             = hue, 
                data_sep        = data_freq, 
                palette         = palette, 
                data_label      = data_label, 
                lineplot_kwargs = lineplot_kwargs
            )
        else:
            lineplot_kwargs_fnl = (
                lineplot_kwargs | 
                dict(
                    x       = x, 
                    y       = y, 
                    hue     = hue, 
                    palette = palette
                )
            )
            if hue is None:
                lineplot_kwargs_fnl['label'] = data_label
            #-------------------------
            sns.lineplot(
                ax=ax, 
                data=draw_data, 
                **lineplot_kwargs_fnl
            )
        if draw_without_hue_also and hue is not None:
            sns.lineplot(
                ax        = ax, 
                data      = draw_data, 
                x         = x, 
                y         = y, 
                hue       = None, 
                color     = 'deeppink', 
                linestyle = '--', 
                label     = 'AVG'
            )
        #----------------------------
        # Note: if hue=None is drawn, then average will already be drawn!
        if (df_mean is not None 
            and hue is not None 
            and not draw_without_hue_also):
            if mean_args is None:
                avg_label = 'AVG'
                if data_label:
                    avg_label = f'{data_label} AVG'
                mean_args = dict(style='--', linewidth=3, alpha=0.50, color='deeppink', label=avg_label, legend=True)
            df_mean.loc[plot_t_beg:plot_t_end][df_mean_col].plot(ax=ax, **mean_args)
        #----------------------------
        if draw_outage_limits:
            ax = AMINonVee.draw_outage_limits_on_ax(
                ax                         = ax, 
                out_t_beg                  = out_t_beg, 
                out_t_end                  = out_t_end, 
                plot_t_beg                 = plot_t_beg, 
                plot_t_end                 = plot_t_end, 
                draw_outage_limits_kwargs  = draw_outage_limits_kwargs, 
                include_outage_limits_text = include_outage_limits_text, 
                out_t_beg_line_color       = 'red', 
                out_t_end_line_color       = 'green',                 
            )
        #----------------------------
        if seg_line_freq is not None:
            seg_min = pd.to_datetime(plot_t_beg).round(seg_line_freq)
            seg_max = pd.to_datetime(plot_t_end).round(seg_line_freq)
            all_segs = pd.date_range(seg_min, seg_max, freq=seg_line_freq)
            for seg in all_segs:
                ax.axvline(seg, color='black', linestyle='--');
        #----------------------------
        if isinstance(title_args, str):
            title_args = dict(label=title_args)
        if title_args is not None:
            ax.set_title(**title_args)
        #----------------------------
        if ax_args is not None:
            ax.set(**ax_args)
        if xlabel_args is not None:
            ax.set_xlabel(**xlabel_args)
        if ylabel_args is not None:
            ax.set_ylabel(**ylabel_args)

        return fig,ax