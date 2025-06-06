#!/usr/bin/env python

r"""
Holds AbsPerfectPower class.  See AbsPerfectPower.AbsPerfectPower for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import time

import pandas as pd
import numpy as np


import matplotlib as mpl

#---------------------------------------------------------------------
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from AMINonVee_SQL import AMINonVee_SQL
#-----
from AMINonVee import AMINonVee
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
#---------------------------------------------------------------------
#sys.path.insert(0, os.path.join(os.path.realpath('..'), 'Utilities'))
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
import Utilities_dt
import Plot_General
import Plot_Hist
import Plot_Bar
#---------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------
class AbsPerfectPower:
    r"""
    Class to make outage predictions
    """
    def __init__(
        self, 
    ):
        r"""
        """
        #---------------------------------------------------------------------------
        # Grabbing connection can take time (seconds, not hours).
        # Keep set to None, only creating if needed (see conn_aws property below)
        self.__conn_aws  = None
    
    @property
    def conn_aws(self):
        if self.__conn_aws is None:
            self.__conn_aws  = Utilities.get_athena_prod_aws_connection()
        return self.__conn_aws
    
    #-----------------------------------------------------------------------------------------------------------------------------
    # METHOD: Minimum Bias (MB)
    # The intent here is, for each meter, to essentially count how many zero-voltage timestamps it registers vs the 
    #   total number of timestamps registering any measurement
    #---------------------------------------------------------------------
    @staticmethod
    def get_nzeros_ntotal_sql_v1(
        date_0  , 
        date_1  , 
        measure = 'n_zeros', 
        **sql_kwargs
    ):
        r"""
        """
        #--------------------------------------------------
        assert(measure in ['n_zeros', 'n_total'])
        #-------------------------
        sql_0_alias = 'sql_0'
        #-------------------------
        if measure == 'n_zeros':
            value = 0
        else:
            value = None
        #-------------------------
        # The following kwargs are handled by function, and therefore should not be found in sql_kwargs
        handled_kwargs = [
            'cols_of_interest', 
            'aep_derived_uoms_and_idntfrs', 
            'value', 
            'date_range', 
            'groupby_cols', 
            'alias'
        ]
        sql_kwargs = {k:v for k,v in sql_kwargs.items() if k not in handled_kwargs}
        #-------------------------
        sql_0 = AMINonVee_SQL.build_sql_usg(
            cols_of_interest             = ['aep_premise_nb', 'serialnumber', 'starttimeperiod', 'COUNT(*)'], 
            aep_derived_uoms_and_idntfrs = ['VOLT'], 
            value                        = value, 
            date_range                   = [date_0, date_1], 
            groupby_cols                 = ['aep_premise_nb', 'serialnumber', 'starttimeperiod'], 
            alias                        = sql_0_alias, 
            **sql_kwargs
        )
        #-----
        sql_0_stmnt = sql_0.get_sql_statement(    
            insert_n_tabs_to_each_line = 0,
            include_alias              = True,
            include_with               = True,
        )
        
        #--------------------------------------------------
        sql_1 = AMINonVee_SQL.build_sql_usg(
            cols_of_interest = ['aep_premise_nb', 'serialnumber', 'COUNT(*)'], 
            groupby_cols     = ['aep_premise_nb', 'serialnumber'], 
            schema_name      = None, 
            table_name       = sql_0_alias
        )
        #-------------------------
        sql_1_stmnt = sql_1.get_sql_statement(    
            insert_n_tabs_to_each_line = 0,
            include_alias              = False,
            include_with               = False,
        )
        
        #--------------------------------------------------
        sql_stmnt = sql_0_stmnt + '\n' + sql_1_stmnt
        #-----
        return sql_stmnt
    
    #---------------------------------------------------------------------
    @staticmethod
    def get_nzeros_ntotal_sql_v2(
        date_0       , 
        date_1       , 
        measure      = 'n_zeros', 
        return_stmnt = True, 
        **sql_kwargs
    ):
        r"""
        """
        #--------------------------------------------------
        assert(measure in ['n_zeros', 'n_total'])
        #-------------------------
        if measure == 'n_zeros':
            value = 0
        else:
            value = None
        #-------------------------
        # The following kwargs are handled by function, and therefore should not be found in sql_kwargs
        handled_kwargs = [
            'cols_of_interest', 
            'aep_derived_uoms_and_idntfrs', 
            'value', 
            'date_range', 
            'groupby_cols'
        ]
        sql_kwargs = {k:v for k,v in sql_kwargs.items() if k not in handled_kwargs}
        #-------------------------
        sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest             = ['aep_premise_nb', 'serialnumber', 'COUNT(DISTINCT starttimeperiod)'], 
            aep_derived_uoms_and_idntfrs = ['VOLT'], 
            value                        = value, 
            date_range                   = [date_0, date_1], 
            groupby_cols                 = ['aep_premise_nb', 'serialnumber'], 
            **sql_kwargs
        )
        #-------------------------
        if return_stmnt:
            return sql.get_sql_statement()
        else:
            return sql
        

    #---------------------------------------------------------------------
    @staticmethod
    def get_nz_nt_ratio_partial_df(
        date_0       , 
        date_1       , 
        measure      = 'n_zeros', 
        method       = 'v1', 
        conn_aws     = None, 
        **sql_kwargs
    ):
        r"""
        """
        #--------------------------------------------------
        assert(measure in ['n_zeros', 'n_total'])
        assert(method in ['v1', 'v2'])
        #-------------------------
        if method == 'v1':
            sql_stmnt = AbsPerfectPower.get_nzeros_ntotal_sql_v1(
                date_0       = date_0, 
                date_1       = date_1, 
                measure      = measure,  
                **sql_kwargs
            )
        else:
            sql_stmnt = AbsPerfectPower.get_nzeros_ntotal_sql_v2(
                date_0       = date_0, 
                date_1       = date_1, 
                measure      = measure, 
                return_stmnt = True, 
                **sql_kwargs
            )
        #--------------------------------------------------
        if conn_aws is None:
            conn_aws  = Utilities.get_athena_prod_aws_connection()
        #-------------------------
        df = pd.read_sql_query(
            sql_stmnt, 
            conn_aws
        )
        #-------------------------
        if measure == 'n_zeros':
            df = df.rename(columns={'_col2':'n_zeros'})
        else:
            df = df.rename(columns={'_col2':'n_total'})
        #-------------------------
        return df
    

    #---------------------------------------------------------------------
    @staticmethod    
    def compile_pct_0_not_cols(
        df            , 
        n_zeros_col   = 'n_zeros', 
        n_total_col   = 'n_total', 
        pct_0_col     = 'pct_0', 
        pct_not_0_col = 'pct_~0',  
    ):
        r"""
        """
        #--------------------------------------------------
        df[n_zeros_col] = df[n_zeros_col].fillna(0)
        #-------------------------
        df[pct_0_col]   = df[n_zeros_col].div(df[n_total_col])
        #-----
        df[pct_not_0_col]  = 1.0-df[pct_0_col]
        #-----
        df[pct_0_col]   = 100*df[pct_0_col]
        df[pct_not_0_col]  = 100*df[pct_not_0_col]
        #--------------------------------------------------
        return df
    
    
    #---------------------------------------------------------------------
    @staticmethod
    def build_nzeros_ntotal_ratio_df(
        date_0       , 
        date_1       , 
        method       = 'v1', 
        conn_aws     = None, 
        **sql_kwargs
    ):
        r"""
        """
        #--------------------------------------------------
        assert(method in ['v1', 'v2'])
        #-------------------------
        df_zeros = AbsPerfectPower.get_nz_nt_ratio_partial_df(
            date_0       = date_0, 
            date_1       = date_1, 
            measure      = 'n_zeros', 
            method       = method, 
            conn_aws     = conn_aws, 
            **sql_kwargs
        )
        #-----
        df_total = AbsPerfectPower.get_nz_nt_ratio_partial_df(
            date_0       = date_0, 
            date_1       = date_1, 
            measure      = 'n_total', 
            method       = method, 
            conn_aws     = conn_aws, 
            **sql_kwargs
        )
        #--------------------------------------------------
        df = pd.merge(
            df_total.dropna(), 
            df_zeros.dropna(), 
            how      = 'left', 
            left_on  = ['aep_premise_nb', 'serialnumber'], 
            right_on = ['aep_premise_nb', 'serialnumber']
        )
        #-------------------------
        df['n_zeros'] = df['n_zeros'].fillna(0)
        #-------------------------
        df = AbsPerfectPower.compile_pct_0_not_cols(
            df            = df, 
            n_zeros_col   = 'n_zeros', 
            n_total_col   = 'n_total', 
            pct_0_col     = 'pct_0', 
            pct_not_0_col = 'pct_~0',  
        )
        #--------------------------------------------------
        return df
    

    #---------------------------------------------------------------------
    @staticmethod
    def concat_MB_ratio_dfs(
        ratio_dfs            , 
        make_col_types_equal = False, 
        SN_col               = 'serialnumber', 
        PN_col               = 'aep_premise_nb', 
        n_zeros_col          = 'n_zeros', 
        n_total_col          = 'n_total', 
        pct_0_col            = 'pct_0', 
        pct_not_0_col        = 'pct_~0'
    ):
        r"""
        """
        #--------------------------------------------------
        return_df = Utilities_df.concat_dfs(
            dfs                  = ratio_dfs, 
            axis                 = 0, 
            make_col_types_equal = make_col_types_equal
        )
        #-------------------------
        return_df = return_df.groupby([PN_col, SN_col], as_index=False).sum()
        #-------------------------
        return_df = AbsPerfectPower.compile_pct_0_not_cols(
            df            = return_df, 
            n_zeros_col   = n_zeros_col, 
            n_total_col   = n_total_col, 
            pct_0_col     = pct_0_col, 
            pct_not_0_col = pct_not_0_col,  
        )
        #-------------------------
        return return_df
    
    
    #---------------------------------------------------------------------
    @staticmethod
    def concat_MB_ratio_dfs_in_dir(
        dir_path             , 
        regex_pattern        = r'\d{8}_\d{8}.*', 
        ignore_case          = False, 
        ext                  = '.pkl', 
        make_col_types_equal = False, 
        return_paths         = False, 
        SN_col               = 'serialnumber', 
        PN_col               = 'aep_premise_nb', 
        n_zeros_col          = 'n_zeros', 
        n_total_col          = 'n_total', 
        pct_0_col            = 'pct_0', 
        pct_not_0_col        = 'pct_~0'
    ):
        r"""
        """
        #--------------------------------------------------
        return_df, files_in_dir = Utilities_df.concat_dfs_in_dir(
            dir_path             = dir_path, 
            regex_pattern        = regex_pattern, 
            ignore_case          = ignore_case, 
            ext                  = ext, 
            make_col_types_equal = make_col_types_equal, 
            return_paths         = True
        )
        #-------------------------
        return_df = return_df.groupby([PN_col, SN_col], as_index=False).sum()
        #-------------------------
        return_df = AbsPerfectPower.compile_pct_0_not_cols(
            df            = return_df, 
            n_zeros_col   = n_zeros_col, 
            n_total_col   = n_total_col, 
            pct_0_col     = pct_0_col, 
            pct_not_0_col = pct_not_0_col,  
        )
        #-------------------------
        if return_paths:
            return return_df, files_in_dir
        return return_df
    
    #---------------------------------------------------------------------
    @staticmethod
    def run_daq_MB(
        save_dir , 
        date_0   , 
        date_1   , 
        opcos    , 
        daq_freq = 'W', 
        conn_aws = None, 
        method   = 'v2'
    ):
        r"""
        daq_freq:
            Determines how the data acquisition will be split over time.
            If None, then data acquistion attempted in single pass (probably won't work if, e.g., 
              date_0 and date_1 differ by 1 year))
            Otherwise, should be accpetable str/Timedelta/whatever 
              e.g., 'W' for weekly
        """
        #---------------------------------------------------------------------------
        # Final save destination = os.path.join(save_dir, opco_i, save_subdir)
        #                        = os.path.join(save_dir, opco_i, date_0_date_1/METHOD_1)
        #-----
        v1_subdir   = r'METHOD_1'
        save_subdir = os.path.join(f"{date_0.strftime('%Y%m%d')}_{date_1.strftime('%Y%m%d')}", v1_subdir)
        #-------------------------
        if conn_aws is None:
            conn_aws  = Utilities.get_athena_prod_aws_connection()
        #---------------------------------------------------------------------------
        if daq_freq is None:
            dates = [(date_0, date_1)]
        else:
            dates = pd.date_range(
                start = date_0, 
                end   = date_1, 
                freq  = daq_freq
            )
            dates = [(dates[i], dates[i+1]-pd.Timedelta('1D')) for i in range(len(dates)-1)]
            #-----
            if date_1 != dates[-1][-1]:
                partial_range = (dates[-1][-1]+pd.Timedelta('1D'), date_1)
                dates.append(partial_range)
        #---------------------------------------------------------------------------
        for opco_i in opcos:
            #--------------------------------------------------
            saved_paths_i = []
            save_dir_i    = os.path.join(save_dir, opco_i, save_subdir)
            if not os.path.exists(save_dir_i):
                os.makedirs(save_dir_i)
            #--------------------------------------------------
            for dates_j in dates:
                date_j_0 = dates_j[0].strftime('%Y-%m-%d')
                date_j_1 = dates_j[1].strftime('%Y-%m-%d')
                #-------------------------
                print(f"\n\n\nopco = {opco_i}\ndate_0 = {date_j_0}\ndate_1 = {date_j_1}")
                #-------------------------
                start = time.time()
                app_df_ij = AbsPerfectPower.build_nzeros_ntotal_ratio_df(
                    date_0       = date_j_0, 
                    date_1       = date_j_1, 
                    method       = method, 
                    conn_aws     = conn_aws, 
                    opcos        = [opco_i]
                )
                print(time.time()-start)
                #-------------------------
                if daq_freq is None:
                    save_dir_ij  = save_dir_i
                    save_name_ij = f"app_df_{opco_i}.pkl"
                else:
                    save_dir_ij  = os.path.join(save_dir_i, 'partials')
                    if not os.path.exists(save_dir_ij):
                        os.makedirs(save_dir_ij)
                    #-----
                    save_name_ij = date_j_0.replace('-','') + '_' + date_j_1.replace('-','') + '.pkl'
                #-------------------------
                save_path_ij = os.path.join(save_dir_ij, save_name_ij)
                app_df_ij.to_pickle(save_path_ij)
                saved_paths_i.append(save_path_ij)
        
            #--------------------------------------------------
            # Compile final pd.DataFrame, if acquistion was split
            if daq_freq is not None:
                #-------------------------
                save_name_i = f"app_df_agg_{opco_i}.pkl"
                save_path_i = os.path.join(save_dir_i, save_name_i)
                #-------------------------
                app_df_i, files_in_dir_i = AbsPerfectPower.concat_MB_ratio_dfs_in_dir(
                    dir_path             = os.path.join(save_dir_i, 'partials'), 
                    regex_pattern        = r'\d{8}_\d{8}.*', 
                    ignore_case          = False, 
                    ext                  = '.pkl', 
                    make_col_types_equal = False, 
                    return_paths         = True, 
                    SN_col               = 'serialnumber', 
                    PN_col               = 'aep_premise_nb', 
                    n_zeros_col          = 'n_zeros', 
                    n_total_col          = 'n_total', 
                    pct_0_col            = 'pct_0', 
                    pct_not_0_col        = 'pct_~0'
                )
                assert(set(files_in_dir_i).symmetric_difference(saved_paths_i)==set())
                #-------------------------
                app_df_i.to_pickle(save_path_i)


    #---------------------------------------------------------------------
    @staticmethod
    def plot_hist_MB_ratios_df(
        ratios_df     , 
        figax         = None, 
        fig_num       = 0, 
        set_logy      = True, 
        title         = None, 
        include_text  = False, 
        #-----
        SN_col        = 'serialnumber', 
        PN_col        = 'aep_premise_nb', 
        pct_0_col     = 'pct_0', 
        pct_not_0_col = 'pct_~0'
    ):
        r"""
        set_logy:
            Can be True, False, or 'both'
        """
        #----------------------------------------------------------------------------------------------------
        assert(Utilities.is_object_one_of_types(set_logy, [bool, str]))
        #----------------------------------------------------------------------------------------------------
        if isinstance(set_logy, str):
            assert(set_logy=='both')
            #-------------------------
            if figax is None:
                figax = Plot_General.default_subplots(
                    fig_num = fig_num, 
                    n_x     = 1, 
                    n_y     = 2, 
                    return_flattened_axes = True
                )
            #-----
            assert(Utilities.is_object_one_of_types(figax, [list, tuple]))
            assert(len(figax)==2)
            assert(isinstance(figax[0], mpl.figure.Figure))
            # There must be two axes objects since plotting both
            assert(len(figax[1])==2)
            #-------------------------
            fig  = figax[0]
            ax_0 = figax[1][0]
            ax_1 = figax[1][1]
            #-------------------------
            fig, ax_0 = AbsPerfectPower.plot_hist_MB_ratios_df(
                ratios_df     = ratios_df, 
                figax         = (fig, ax_0), 
                fig_num       = fig_num, 
                set_logy      = False, 
                title         = title, 
                include_text  = include_text, 
                SN_col        = SN_col, 
                PN_col        = PN_col, 
                pct_0_col     = pct_0_col, 
                pct_not_0_col = pct_not_0_col
            )
            #-----
            fig, ax_1 = AbsPerfectPower.plot_hist_MB_ratios_df(
                ratios_df     = ratios_df, 
                figax         = (fig, ax_1), 
                fig_num       = fig_num, 
                set_logy      = True, 
                title         = title, 
                include_text  = include_text, 
                SN_col        = SN_col, 
                PN_col        = PN_col, 
                pct_0_col     = pct_0_col, 
                pct_not_0_col = pct_not_0_col
            )
            #-------------------------
            return fig, [ax_0,ax_1]
        #----------------------------------------------------------------------------------------------------
        assert(isinstance(set_logy, bool))
        #-----
        if figax is None: 
            figax = Plot_General.default_subplots(
                fig_num = fig_num, 
                n_x     = 1, 
                n_y     = 1, 
                return_flattened_axes = False
            )
        #-------------------------
        assert(Utilities.is_object_one_of_types(figax, [list, tuple]))
        assert(len(figax)==2)
        assert(isinstance(figax[0], mpl.figure.Figure))
        assert(isinstance(figax[1], mpl.axes.Axes))
        fig,ax = figax
        #--------------------------------------------------
        ax = Plot_Hist.plot_hist(
            ax                            = ax, 
            df                            = ratios_df, 
            x_col                         = pct_not_0_col, 
            min_max_and_bin_size          = [0, 101, 1], 
            include_over_underflow        = True, 
            stat                          = 'count', 
            plot_sns                      = False, 
            hist_plot_kwargs              = None, 
            keep_edges_opaque             = True, 
            div_drawn_width_by            = None, 
            relative_position_idx         = None, 
            run_set_general_plotting_args = True, 
            orient                        = 'v'
        )
        #--------------------------------------------------
        if include_text:
            # Found cases where a single serial number corresponds to two different premises.
            # My guess is that the meter was moved and re-used elsewhere.
            # In any case, this situation means I cannot simply call, e.g., 
            #   n_perfect = ratios_df[ratios_df['pct_0']==0]['serialnumber'].nunique()
            # but must instead use
            #   n_perfect = ratios_df[ratios_df['pct_0']==0].groupby(['aep_premise_nb', 'serialnumber']).ngroups
            #-----
            n_SNs = ratios_df.groupby([PN_col, SN_col]).ngroups
            n_perfect = ratios_df[ratios_df[pct_0_col]==0].groupby([PN_col, SN_col]).ngroups
            n_imprfct = ratios_df[ratios_df[pct_0_col]!=0].groupby([PN_col, SN_col]).ngroups
            assert(n_perfect+n_imprfct==n_SNs)
            #-----
            n_geq_99 = ratios_df[ratios_df[pct_not_0_col]>=99].groupby([PN_col, SN_col]).ngroups
            #-----
            ax.text(0.25, 0.900, f"n_SNs     = {np.round(n_SNs, decimals=2)}", ha='left', va='center', transform=ax.transAxes, fontsize='x-large', fontdict={'family':'monospace'})
            ax.text(0.25, 0.825, f"n_perfect = {np.round(n_perfect, decimals=2)} ({np.round(100*n_perfect/n_SNs, decimals=2)}%)", ha='left', va='center', transform=ax.transAxes, fontsize='x-large', fontdict={'family':'monospace'})
            ax.text(0.25, 0.750, f"n_imprfct = {np.round(n_imprfct, decimals=2)} ({np.round(100*n_imprfct/n_SNs, decimals=2)}%)", ha='left', va='center', transform=ax.transAxes, fontsize='x-large', fontdict={'family':'monospace'})
            ax.text(0.25, 0.675, f"n_geq_99  = {np.round(n_geq_99, decimals=2)} ({np.round(100*n_geq_99/n_SNs, decimals=2)}%)", ha='left', va='center', transform=ax.transAxes, fontsize='x-large', fontdict={'family':'monospace'})
        #--------------------------------------------------
        if title is None:
            title = ''
        #-----
        if set_logy:
            ax.set_yscale('log')
            title = f"{title} (log-y)"
        ax.set_title(title, fontsize='xx-large')
        ax.set_xlabel('% Power', fontsize='xx-large', loc='right')
        #-------------------------
        return fig,ax



    #-----------------------------------------------------------------------------------------------------------------------------
    # METHOD 2
    # The intent here to reproduce a table given to Jon summarizing the accumulated service interruptions and accumulated
    #   outage durations.
    #
    #   TABLE 1
    #               Acc. Service Interrptions TTM | Customers (Meters) | Percentage of Total
    #                                             |                    |
    #                                0            |      450,405       |      24.4%
    #                                1 TO 6       |      926,814       |      50.2%
    #                                7 TO 12      |      241,447       |      13.1%
    #                                13 TO 24     |      211,159       |      11.4%
    #                                25 OR MORE   |      18,190        |      1%
    #                                Total        |      1,848,045     |      100%
    #
    #   TABLE 2
    #               Acc. Outage Duration TTM      | Customers (Meters) | Percentage of Total
    #                                             |                    |
    #                                <6           |      1,597,787     |      86.46%
    #                                >6-24        |      219,926       |      11.90%
    #                                >24-48       |      22,074        |      1.19%
    #                                >48-96       |      6,811         |      0.37%
    #                                >96          |      1,447         |      0.08%
    #                                Total        |      1,848,045     |      100%
    #
    #---------------------------------------------------------------------
    @staticmethod
    def build_zero_times_sql_v1_OLD(
        date_0           ,
        date_1           ,
        split_if_missing = False, 
        **kwargs
    ):
        r"""
        !!!!! IMPORTANT !!!!!
        1.  This method is slower than v2, as we retain time/is_zero status instead of immediately grabbing entries with value=0.
            However, the benefit is some added flexibility through the split_if_missing parameter.
                - split_if_missing parameter allows one to choose how to handle, e.g.,  situations where we have a bunch of sequential 
                    zero-value measurements but a single measurement is missing.
                    - split_if_missing = False ==> Treat as a single interruption
                    - split_if_missing = True  ==> Treat as two interruptions
                - e.g., (12:00, 0), (12:15, 0), (12:30, 0), (13:00, 0), (13:15, 0)
                    - split_if_missing = False ==> #1[(12:00, 0), (12:15, 0), (12:30, 0), (13:00, 0), (13:15, 0)]
                    - split_if_missing = True  ==> #1[(12:00, 0), (12:15, 0), (12:30, 0)] and #2[(13:00, 0), (13:15, 0)]
            The case split_if_missing=True should return the same results as build_zero_times_sql_v2
        """
        #---------------------------------------------------------------------------
        # 1. Build ZeroTimestamps CTE
        #---------------------------------------------------------------------------
        cols_of_interest = [
            'serialnumber', 
            "CAST(regexp_replace(starttimeperiod, '([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*', '$1 $2') AS TIMESTAMP) as starttimeperiod", 
            'MAX(CASE WHEN value = 0 THEN 1 ELSE 0 END) AS is_zero'
        ]
        #-----
        kwargs['date_range']   = [date_0, date_1]
        kwargs['groupby_cols'] = ['serialnumber', 'starttimeperiod']
        kwargs['alias']        = 'ZeroTimestamps'
        #**************************************************
        sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest             = cols_of_interest, 
            aep_derived_uoms_and_idntfrs = ['VOLT'],  
            kwh_and_vlt_only             = False, 
            **kwargs
        )
    
        #---------------------------------------------------------------------------
        # 2. Rest of the CTEs and final query
        #---------------------------------------------------------------------------
        cte_laglead = """
        CTE_DECIMAL_DATA AS (
            SELECT
                serialnumber,
                starttimeperiod,
                is_zero, 
                LAG(starttimeperiod)  OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS starttimeperiod_prev, 
                LAG(is_zero)          OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS is_zero_prev, 
                LEAD(starttimeperiod) OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS starttimeperiod_next, 
                LEAD(is_zero)         OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS is_zero_next, 
                ROW_NUMBER() OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS island_location 
            FROM 
                 ZeroTimestamps
        )"""
        #-------------------------
        cte_island_start = """
        CTE_ISLAND_START AS (
            SELECT
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_start_starttimeperiod,
                island_location AS island_start_location
            FROM CTE_DECIMAL_DATA
            WHERE
                (
                    is_zero = 1 AND
                    (
                        is_zero_prev = 0 OR 
                        is_zero_prev IS NULL"""
        #-----
        if split_if_missing:
            cte_island_start += """ OR
                        DATE_DIFF('minute', starttimeperiod_prev, starttimeperiod) > 15
                    ) 
                )
        )"""
        else:
            cte_island_start += """
                    ) 
                )
        )"""
        #-------------------------
        cte_island_end = """
        CTE_ISLAND_END AS (
            SELECT
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_end_starttimeperiod,
                island_location AS island_end_location
            FROM CTE_DECIMAL_DATA
            WHERE
                (
                    is_zero = 1 AND
                    (
                        is_zero_next = 0 OR 
                        is_zero_next IS NULL"""
        #-----
        if split_if_missing:
            cte_island_end += """ OR
                        DATE_DIFF('minute', starttimeperiod, starttimeperiod_next) > 15
                    ) 
                )
        )"""
        else:
            cte_island_end += """
                    ) 
                )
        )"""
        #-------------------------
        sql_fnl = """
        SELECT
            I_START.*, 
            I_END.island_end_starttimeperiod, 
            I_END.island_end_location
        FROM 
            CTE_ISLAND_START I_START
        INNER JOIN 
            CTE_ISLAND_END I_END
            ON  I_END.island_number = I_START.island_number 
            AND I_END.serialnumber  = I_START.serialnumber
        """
    
        #---------------------------------------------------------------------------
        # 3. Put it all together
        #---------------------------------------------------------------------------
        sql_stmnt = sql.get_sql_statement(
            insert_n_tabs_to_each_line = 0,
            include_alias              = True,
            include_with               = True,
        )
        sql_stmnt = (
            sql_stmnt        + ', ' + 
            cte_laglead      + ', ' + 
            cte_island_start + ', ' + 
            cte_island_end   + 
            sql_fnl
        )
        #-------------------------
        return sql_stmnt


    #---------------------------------------------------------------------
    @staticmethod
    def build_zero_times_sql_v1(
        date_0           ,
        date_1           ,
        allow_null_SNs   = False, 
        allow_null_PNs   = False, 
        split_if_missing = False, 
        **kwargs
    ):
        r"""
        !!!!! IMPORTANT !!!!!
        1.  This method is slower than v2, as we retain time/is_zero status instead of immediately grabbing entries with value=0.
            However, the benefit is some added flexibility through the split_if_missing parameter.
                - split_if_missing parameter allows one to choose how to handle, e.g.,  situations where we have a bunch of sequential 
                    zero-value measurements but a single measurement is missing.
                    - split_if_missing = False ==> Treat as a single interruption
                    - split_if_missing = True  ==> Treat as two interruptions
                - e.g., (12:00, 0), (12:15, 0), (12:30, 0), (13:00, 0), (13:15, 0)
                    - split_if_missing = False ==> #1[(12:00, 0), (12:15, 0), (12:30, 0), (13:00, 0), (13:15, 0)]
                    - split_if_missing = True  ==> #1[(12:00, 0), (12:15, 0), (12:30, 0)] and #2[(13:00, 0), (13:15, 0)]
            The case split_if_missing=True should return the same results as build_zero_times_sql_v2
        """
        #---------------------------------------------------------------------------
        # 1. Build ZeroTimestamps CTE
        #---------------------------------------------------------------------------
        cols_of_interest = [
            'aep_premise_nb', 
            'serialnumber', 
            "CAST(regexp_replace(starttimeperiod, '([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*', '$1 $2') AS TIMESTAMP) as starttimeperiod", 
            'MAX(CASE WHEN value = 0 THEN 1 ELSE 0 END) AS is_zero'
        ]
        #-----
        kwargs['date_range']   = [date_0, date_1]
        kwargs['groupby_cols'] = ['aep_premise_nb', 'serialnumber', 'starttimeperiod']
        kwargs['alias']        = 'ZeroTimestamps'
        #**************************************************
        sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest             = cols_of_interest, 
            aep_derived_uoms_and_idntfrs = ['VOLT'],  
            kwh_and_vlt_only             = False, 
            **kwargs
        )
    
        #---------------------------------------------------------------------------
        # 2. Rest of the CTEs and final query
        #---------------------------------------------------------------------------
        cte_laglead = """
        CTE_DECIMAL_DATA AS (
            SELECT
                aep_premise_nb, 
                serialnumber,
                starttimeperiod,
                is_zero, 
                LAG(starttimeperiod)  OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS starttimeperiod_prev, 
                LAG(is_zero)          OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS is_zero_prev, 
                LEAD(starttimeperiod) OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS starttimeperiod_next, 
                LEAD(is_zero)         OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS is_zero_next, 
                ROW_NUMBER()          OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS island_location 
            FROM 
                 ZeroTimestamps
        )"""
        #-------------------------
        cte_island_start = """
        CTE_ISLAND_START AS (
            SELECT
                aep_premise_nb, 
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_start_starttimeperiod,
                island_location AS island_start_location
            FROM CTE_DECIMAL_DATA
            WHERE
                (
                    is_zero = 1 AND
                    (
                        is_zero_prev = 0 OR 
                        is_zero_prev IS NULL"""
        #-----
        if split_if_missing:
            cte_island_start += """ OR
                        DATE_DIFF('minute', starttimeperiod_prev, starttimeperiod) > 15
                    ) 
                )
        )"""
        else:
            cte_island_start += """
                    ) 
                )
        )"""
        #-------------------------
        cte_island_end = """
        CTE_ISLAND_END AS (
            SELECT
                aep_premise_nb, 
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_end_starttimeperiod,
                island_location AS island_end_location
            FROM CTE_DECIMAL_DATA
            WHERE
                (
                    is_zero = 1 AND
                    (
                        is_zero_next = 0 OR 
                        is_zero_next IS NULL"""
        #-----
        if split_if_missing:
            cte_island_end += """ OR
                        DATE_DIFF('minute', starttimeperiod, starttimeperiod_next) > 15
                    ) 
                )
        )"""
        else:
            cte_island_end += """
                    ) 
                )
        )"""
        #--------------------------------------------------
        sql_fnl = """
        SELECT
            I_START.*, 
            I_END.island_end_starttimeperiod, 
            I_END.island_end_location
        FROM 
            CTE_ISLAND_START I_START
        INNER JOIN 
            CTE_ISLAND_END I_END
            ON   I_END.island_number   = I_START.island_number"""
        #-------------------------
        if allow_null_SNs:
            sql_fnl += """
            AND (I_END.serialnumber    = I_START.serialnumber OR (I_END.serialnumber IS NULL AND I_START.serialnumber IS NULL))"""
        else:
            sql_fnl += """
            AND I_END.serialnumber     = I_START.serialnumber """
        #-------------------------
        if allow_null_PNs:
            sql_fnl += """
            AND (I_END.aep_premise_nb  = I_START.aep_premise_nb OR (I_END.aep_premise_nb IS NULL AND I_START.aep_premise_nb IS NULL))"""
        else:
            sql_fnl += """
            AND I_END.aep_premise_nb   = I_START.aep_premise_nb"""
    
        #---------------------------------------------------------------------------
        # 3. Put it all together
        #---------------------------------------------------------------------------
        sql_stmnt = sql.get_sql_statement(
            insert_n_tabs_to_each_line = 0,
            include_alias              = True,
            include_with               = True,
        )
        sql_stmnt = (
            sql_stmnt        + ', ' + 
            cte_laglead      + ', ' + 
            cte_island_start + ', ' + 
            cte_island_end   + 
            sql_fnl
        )
        #-------------------------
        return sql_stmnt
    
    
    #---------------------------------------------------------------------
    @staticmethod
    def build_zero_times_sql_v2_OLD(
        date_0,
        date_1,
        **kwargs
    ):
        r"""
        !!!!! IMPORTANT !!!!!
        1.  This method is faster than v1, as we grab entries with value=0 immediately.
            However, with this speed increase comes a minor restriction:
                - Any missing data will cause the termination of a zeros-island.
                    - Meaning, in situations where we have a bunch of sequential zero-value measurements but a single measurement is missing, 
                        instead of (possibly, if desired) being considered a single interruption period, this will be considered two separate
                        interruptions
                    - e.g., (12:00, 0), (12:15, 0), (12:30, 0), (13:00, 0), (13:15, 0)
                      ==> #1[(12:00, 0), (12:15, 0), (12:30, 0)] and #2[(13:00, 0), (13:15, 0)]
                      due to the missing 12:45 measurement.
                - This is due to the fact that we only have time information of zeros, instead of time/is_zero information in v1.
                - With both time/is_zero information, we can allow periods of outage to be defined by only the is_zero status of the
                    data point and those of its two neighbors.
        2.  The GROUP BY operation in ZeroTimestamps serves to collapse the (likely) multiple voltage measurements
        (e.g., 'AVG', 'INSTV(ABC)1') for a given (serialnumber,starttimeperiond) into a single value.
        Implicitly, the collapse to a single value occurs via an OR operation.
        Meaning, if any one of the voltage measurements is zero, the starttimeperiod is kept for the serial number as a zero-value measurement
        """
        #---------------------------------------------------------------------------
        # 1. Build ZeroTimestamps CTE
        #---------------------------------------------------------------------------
        cols_of_interest = [
            'serialnumber', 
            "CAST(regexp_replace(starttimeperiod, '([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*', '$1 $2') AS TIMESTAMP) as starttimeperiod"
        ]
        #-----
        kwargs['date_range']   = [date_0, date_1]
        kwargs['value']        = 0
        kwargs['groupby_cols'] = ['serialnumber', 'starttimeperiod']
        kwargs['alias']        = 'ZeroTimestamps'
        #**************************************************
        sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest             = cols_of_interest, 
            aep_derived_uoms_and_idntfrs = ['VOLT'],  
            kwh_and_vlt_only             = False, 
            **kwargs
        )
    
        #---------------------------------------------------------------------------
        # 2. Rest of the CTEs and final query
        #---------------------------------------------------------------------------
        cte_laglead = """
        CTE_DECIMAL_DATA AS (
            SELECT
                serialnumber,
                starttimeperiod,
                LAG(starttimeperiod)  OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS starttimeperiod_prev, 
                LEAD(starttimeperiod) OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS starttimeperiod_next, 
                ROW_NUMBER()          OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS island_location 
            FROM 
                 ZeroTimestamps
        )"""
        #-------------------------
        cte_island_start = """
        CTE_ISLAND_START AS (
            SELECT
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_start_starttimeperiod,
                island_location AS island_start_location
            FROM CTE_DECIMAL_DATA
            WHERE
                DATE_DIFF('minute', starttimeperiod_prev, starttimeperiod) > 15 OR
                starttimeperiod_prev IS NULL
        )"""
        #-------------------------
        cte_island_end = """
        CTE_ISLAND_END AS (
            SELECT
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_end_starttimeperiod,
                island_location AS island_end_location
            FROM CTE_DECIMAL_DATA
            WHERE
                DATE_DIFF('minute', starttimeperiod, starttimeperiod_next) > 15 OR
                starttimeperiod_next IS NULL
        )"""
        #-------------------------
        sql_fnl = """
        SELECT
            I_START.*, 
            I_END.island_end_starttimeperiod, 
            I_END.island_end_location
        FROM 
            CTE_ISLAND_START I_START
        INNER JOIN 
            CTE_ISLAND_END I_END
            ON  I_END.island_number = I_START.island_number 
            AND I_END.serialnumber  = I_START.serialnumber    
        """
    
        #---------------------------------------------------------------------------
        # 3. Put it all together
        #---------------------------------------------------------------------------
        sql_stmnt = sql.get_sql_statement(
            insert_n_tabs_to_each_line = 0,
            include_alias              = True,
            include_with               = True,
        )
        sql_stmnt = (
            sql_stmnt        + ', ' + 
            cte_laglead      + ', ' + 
            cte_island_start + ', ' + 
            cte_island_end   + 
            sql_fnl
        )
        #-------------------------
        return sql_stmnt
    

    #---------------------------------------------------------------------
    @staticmethod
    def build_zero_times_sql_v2(
        date_0,
        date_1,
        allow_null_SNs   = False, 
        allow_null_PNs   = False, 
        **kwargs
    ):
        r"""
        !!!!! IMPORTANT !!!!!
        1.  This method is faster than v1, as we grab entries with value=0 immediately.
            However, with this speed increase comes a minor restriction:
                - Any missing data will cause the termination of a zeros-island.
                    - Meaning, in situations where we have a bunch of sequential zero-value measurements but a single measurement is missing, 
                        instead of (possibly, if desired) being considered a single interruption period, this will be considered two separate
                        interruptions
                    - e.g., (12:00, 0), (12:15, 0), (12:30, 0), (13:00, 0), (13:15, 0)
                      ==> #1[(12:00, 0), (12:15, 0), (12:30, 0)] and #2[(13:00, 0), (13:15, 0)]
                      due to the missing 12:45 measurement.
                - This is due to the fact that we only have time information of zeros, instead of time/is_zero information in v1.
                - With both time/is_zero information, we can allow periods of outage to be defined by only the is_zero status of the
                    data point and those of its two neighbors.
        2.  The GROUP BY operation in ZeroTimestamps serves to collapse the (likely) multiple voltage measurements
        (e.g., 'AVG', 'INSTV(ABC)1') for a given (aep_premise_nb,serialnumber,starttimeperiond) into a single value.
        Implicitly, the collapse to a single value occurs via an OR operation.
        Meaning, if any one of the voltage measurements is zero, the starttimeperiod is kept for the serial number as a zero-value measurement
        """
        #---------------------------------------------------------------------------
        # 1. Build ZeroTimestamps CTE
        #---------------------------------------------------------------------------
        cols_of_interest = [
            'aep_premise_nb', 
            'serialnumber', 
            "CAST(regexp_replace(starttimeperiod, '([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*', '$1 $2') AS TIMESTAMP) as starttimeperiod"
        ]
        #-----
        kwargs['date_range']   = [date_0, date_1]
        kwargs['value']        = 0
        kwargs['groupby_cols'] = ['aep_premise_nb', 'serialnumber', 'starttimeperiod']
        kwargs['alias']        = 'ZeroTimestamps'
        #**************************************************
        sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest             = cols_of_interest, 
            aep_derived_uoms_and_idntfrs = ['VOLT'],  
            kwh_and_vlt_only             = False, 
            **kwargs
        )
    
        #---------------------------------------------------------------------------
        # 2. Rest of the CTEs and final query
        #---------------------------------------------------------------------------
        cte_laglead = """
        CTE_DECIMAL_DATA AS (
            SELECT
                aep_premise_nb, 
                serialnumber,
                starttimeperiod,
                LAG(starttimeperiod)  OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS starttimeperiod_prev, 
                LEAD(starttimeperiod) OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS starttimeperiod_next, 
                ROW_NUMBER()          OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS island_location 
            FROM 
                 ZeroTimestamps
        )"""
        #-------------------------
        cte_island_start = """
        CTE_ISLAND_START AS (
            SELECT
                aep_premise_nb, 
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_start_starttimeperiod,
                island_location AS island_start_location
            FROM CTE_DECIMAL_DATA
            WHERE
                DATE_DIFF('minute', starttimeperiod_prev, starttimeperiod) > 15 OR
                starttimeperiod_prev IS NULL
        )"""
        #-------------------------
        cte_island_end = """
        CTE_ISLAND_END AS (
            SELECT
                aep_premise_nb, 
                serialnumber, 
                ROW_NUMBER() OVER (PARTITION BY aep_premise_nb,serialnumber ORDER BY starttimeperiod) AS island_number,
                starttimeperiod AS island_end_starttimeperiod,
                island_location AS island_end_location
            FROM CTE_DECIMAL_DATA
            WHERE
                DATE_DIFF('minute', starttimeperiod, starttimeperiod_next) > 15 OR
                starttimeperiod_next IS NULL
        )"""
        #--------------------------------------------------
        sql_fnl = """
        SELECT
            I_START.*, 
            I_END.island_end_starttimeperiod, 
            I_END.island_end_location
        FROM 
            CTE_ISLAND_START I_START
        INNER JOIN 
            CTE_ISLAND_END I_END
            ON   I_END.island_number   = I_START.island_number"""
        #-------------------------
        if allow_null_SNs:
            sql_fnl += """
            AND (I_END.serialnumber    = I_START.serialnumber OR (I_END.serialnumber IS NULL AND I_START.serialnumber IS NULL))"""
        else:
            sql_fnl += """
            AND I_END.serialnumber     = I_START.serialnumber """
        #-------------------------
        if allow_null_PNs:
            sql_fnl += """
            AND (I_END.aep_premise_nb  = I_START.aep_premise_nb OR (I_END.aep_premise_nb IS NULL AND I_START.aep_premise_nb IS NULL))"""
        else:
            sql_fnl += """
            AND I_END.aep_premise_nb   = I_START.aep_premise_nb"""
    
        #---------------------------------------------------------------------------
        # 3. Put it all together
        #---------------------------------------------------------------------------
        sql_stmnt = sql.get_sql_statement(
            insert_n_tabs_to_each_line = 0,
            include_alias              = True,
            include_with               = True,
        )
        sql_stmnt = (
            sql_stmnt        + ', ' + 
            cte_laglead      + ', ' + 
            cte_island_start + ', ' + 
            cte_island_end   + 
            sql_fnl
        )
        #-------------------------
        return sql_stmnt

    
    #---------------------------------------------------------------------
    @staticmethod
    def get_zero_times_df(
        date_0           , 
        date_1           , 
        allow_null_SNs   = False, 
        allow_null_PNs   = False, 
        split_if_missing = True, 
        method           = 'v2', 
        conn_aws         = None, 
        **sql_kwargs
    ):
        r"""
        split_if_missing:
            Can only be set to False if method=='v1'.
            Using split_if_missing=True with v1 should return the same results as v2
        """
        #--------------------------------------------------
        assert(method in ['v1', 'v2'])
        #-------------------------
        if method=='v2' and not split_if_missing:
            print(f'WARNING: split_if_missing = {split_if_missing}, therefore v1 must be used!  The runtime may be longer than expected')
            method = 'v1'
        #--------------------------------------------------
        if method=='v1':
            sql_i = AbsPerfectPower.build_zero_times_sql_v1(
                date_0           = date_0,
                date_1           = date_1,
                allow_null_SNs   = allow_null_SNs, 
                allow_null_PNs   = allow_null_PNs, 
                split_if_missing = split_if_missing, 
                **sql_kwargs
            )
        elif method=='v2':
            sql_i = AbsPerfectPower.build_zero_times_sql_v2(
                date_0           = date_0,
                date_1           = date_1,
                allow_null_SNs   = allow_null_SNs, 
                allow_null_PNs   = allow_null_PNs, 
                **sql_kwargs
            )
        else:
            assert(0)
        #--------------------------------------------------
        if conn_aws is None:
            conn_aws  = Utilities.get_athena_prod_aws_connection()
        #-----
        df_i = pd.read_sql_query(
            sql_i, 
            conn_aws
        )
        #--------------------------------------------------
        return df_i


    #---------------------------------------------------------------------
    @staticmethod
    def combine_zero_times_by_PN_helper(
        df_i         , 
        t_min_col    = 'island_start_starttimeperiod', 
        t_max_col    = 'island_end_starttimeperiod', 
        PN_col       = 'aep_premise_nb', 
    ):
        r"""
        Intended for use with a single PN
        """
        #-------------------------
        PN_i = df_i[PN_col].unique().tolist()
        assert(len(PN_i)==1)
        PN_i = PN_i[0]
        #-------------------------
        ranges_i   = df_i[[t_min_col, t_max_col]].values
        overlaps_i = Utilities.get_overlap_intervals(ranges = ranges_i)
        #-------------------------
        if len(ranges_i) == len(overlaps_i):
            return_df = df_i[[PN_col, t_min_col, t_max_col]]
        else:
            return_df = pd.DataFrame(
                data    = overlaps_i, 
                columns = [t_min_col, t_max_col], 
                index   = [PN_i]*len(overlaps_i)
            ).reset_index(names = PN_col)
        #-------------------------
        return return_df
    
    #---------------------------------------------------------------------
    @staticmethod
    def combine_zero_times_by_PN(
        zero_times_df , 
        t_min_col     = 'island_start_starttimeperiod', 
        t_max_col     = 'island_end_starttimeperiod', 
        PN_col        = 'aep_premise_nb',    
    ):
        r"""
        The resultant return_df will contain only nec_cols = [t_min_col, t_max_col, PN_col]
        """
        #-------------------------
        nec_cols = [t_min_col, t_max_col, PN_col]
        assert(set(nec_cols).difference(set(zero_times_df.columns))==set())
        #-------------------------
        return_df = zero_times_df.groupby(PN_col, dropna=True, as_index=False, group_keys=False).apply(
            lambda x: AbsPerfectPower.combine_zero_times_by_PN_helper(
                df_i         = x, 
                t_min_col    = t_min_col, 
                t_max_col    = t_max_col, 
                PN_col       = PN_col, 
            )
        ).reset_index(drop=True)
        #-------------------------
        return return_df
    

    #---------------------------------------------------------------------
    @staticmethod
    def concat_zero_times_dfs(
        app_dfs              , 
        make_col_types_equal = False, 
        cols_to_drop         = [
            'island_number', 
            'island_start_location', 
            'island_end_location'
        ], 
    ):
        r"""
        IMPORTANT!
        After concatenation, the following columns become meaningless:
            island_number
            island_start_location
            island_end_location
        """
        #--------------------------------------------------
        return_df = Utilities_df.concat_dfs(
            dfs                  = app_dfs, 
            axis                 = 0, 
            make_col_types_equal = make_col_types_equal
        )
        #-------------------------
        if cols_to_drop is not None:
            cols_to_drop = list(set(cols_to_drop).intersection(set(return_df.columns)))
            return_df = return_df.drop(columns=cols_to_drop)
        #-------------------------
        return return_df
    
    
    #---------------------------------------------------------------------
    @staticmethod
    def concat_zero_times_dfs_in_dir(
        dir_path             , 
        regex_pattern        = r'\d{8}_\d{8}.*', 
        ignore_case          = False, 
        ext                  = '.pkl', 
        make_col_types_equal = False, 
        cols_to_drop         = [
            'island_number', 
            'island_start_location', 
            'island_end_location'
        ], 
        return_paths         = False, 
    ):
        r"""
        IMPORTANT!
        After concatenation, the following columns become meaningless:
            island_number
            island_start_location
            island_end_location
        """
        #--------------------------------------------------
        return_df, files_in_dir = Utilities_df.concat_dfs_in_dir(
            dir_path             = dir_path, 
            regex_pattern        = regex_pattern, 
            ignore_case          = ignore_case, 
            ext                  = ext, 
            make_col_types_equal = make_col_types_equal, 
            return_paths         = True
        )
        #-------------------------
        if cols_to_drop is not None:
            cols_to_drop = list(set(cols_to_drop).intersection(set(return_df.columns)))
            return_df = return_df.drop(columns=cols_to_drop)
        #-------------------------
        if return_paths:
            return return_df, files_in_dir
        return return_df
    

    #---------------------------------------------------------------------
    @staticmethod
    def get_all_xNs(
        date_0   ,
        date_1   ,
        conn_aws = None, 
        **kwargs
    ):
        r"""
        Get all distinct serialnumber/aep_premise_nb
        """
        #---------------------------------------------------------------------------
        kwargs['aep_derived_uoms_and_idntfrs'] = ['VOLT']
        #--------------------------------------------------
        df = AMINonVee.get_usg_distinct_fields(
            date_0                           = date_0, 
            date_1                           = date_1, 
            fields                           = ['serialnumber', 'aep_premise_nb'], 
            are_datetime                     = False, 
            addtnl_build_sql_function_kwargs = kwargs, 
            cols_and_types_to_convert_dict   = None, 
            to_numeric_errors                = 'coerce', 
            save_args                        = False, 
            conn_aws                         = conn_aws, 
            return_sql                       = False
        )
        return df
    

    #---------------------------------------------------------------------
    @staticmethod
    def run_daq(
        save_dir         , 
        date_0           , 
        date_1           , 
        opcos            , 
        daq_freq         = 'W', 
        conn_aws         = None, 
        allow_null_SNs   = False, 
        allow_null_PNs   = False, 
        split_if_missing = True, 
        method           = 'v2'
    ):
        r"""
        daq_freq:
            Determined how the data acquisition will be split over time.
            If None, then data acquistion attempted in single pass (probably won't work if, e.g., 
              date_0 and date_1 differ by 1 year))
            Otherwise, should be accpetable str/Timedelta/whatever 
              e.g., 'W' for weekly
        """
        #---------------------------------------------------------------------------
        # Final save destination = os.path.join(save_dir, opco_i, save_subdir)
        #                        = os.path.join(save_dir, opco_i, date_0_date_1/METHOD_2)
        #-----
        v1_subdir   = r'METHOD_2'
        save_subdir = os.path.join(f"{date_0.strftime('%Y%m%d')}_{date_1.strftime('%Y%m%d')}", v1_subdir)
        #-------------------------
        if conn_aws is None:
            conn_aws  = Utilities.get_athena_prod_aws_connection()
        #---------------------------------------------------------------------------
        if daq_freq is None:
            dates = [(date_0, date_1)]
        else:
            dates = Utilities_dt.get_date_ranges(
                date_0            = date_0, 
                date_1            = date_1, 
                freq              = daq_freq, 
                include_endpoints = True
            )
        #---------------------------------------------------------------------------
        for opco_i in opcos:
            #--------------------------------------------------
            zts_saved_paths_i = []
            xNs_saved_paths_i = []
            save_dir_i    = os.path.join(save_dir, opco_i, save_subdir)
            if not os.path.exists(save_dir_i):
                os.makedirs(save_dir_i)
            #--------------------------------------------------
            for dates_j in dates:
                date_j_0 = dates_j[0].strftime('%Y-%m-%d')
                date_j_1 = dates_j[1].strftime('%Y-%m-%d')
                #-------------------------
                print(f"\n\n\nopco = {opco_i}\ndate_0 = {date_j_0}\ndate_1 = {date_j_1}")
                #-------------------------
                start = time.time()
                zero_times_df_ij = AbsPerfectPower.get_zero_times_df(
                    date_0           = date_j_0, 
                    date_1           = date_j_1, 
                    allow_null_SNs   = allow_null_SNs, 
                    allow_null_PNs   = allow_null_PNs, 
                    split_if_missing = split_if_missing, 
                    method           = method, 
                    conn_aws         = conn_aws, 
                    opcos            = [opco_i]
                )
                print(f"zero_times_df_ij: {time.time()-start}")
                #-------------------------
                start = time.time()
                all_xNs_ij = AbsPerfectPower.get_all_xNs(
                    date_0           = date_j_0, 
                    date_1           = date_j_1, 
                    conn_aws         = conn_aws, 
                    opcos            = [opco_i]
                )
                print(f"all_xNs_ij: {time.time()-start}")
                #-------------------------
                if daq_freq is None:
                    save_dir_ij  = save_dir_i
                    zts_save_name_ij = f"zero_times_df_{opco_i}.pkl"
                    xNs_save_name_ij =    f"all_xNs_df_{opco_i}.pkl"
                else:
                    save_dir_ij  = os.path.join(save_dir_i, 'partials')
                    if not os.path.exists(save_dir_ij):
                        os.makedirs(save_dir_ij)
                    #-----
                    zts_save_name_ij = "zero_times_df_" + date_j_0.replace('-','') + '_' + date_j_1.replace('-','') + '.pkl'
                    xNs_save_name_ij =    "all_xNs_df_" + date_j_0.replace('-','') + '_' + date_j_1.replace('-','') + '.pkl'
                #-------------------------
                zts_save_path_ij = os.path.join(save_dir_ij, zts_save_name_ij)
                zero_times_df_ij.to_pickle(zts_save_path_ij)
                zts_saved_paths_i.append(zts_save_path_ij)
                #-----
                xNs_save_path_ij = os.path.join(save_dir_ij, xNs_save_name_ij)
                all_xNs_ij.to_pickle(xNs_save_path_ij)
                xNs_saved_paths_i.append(xNs_save_path_ij)

            #--------------------------------------------------
            # Compile final pd.DataFrame, if acquistion was split
            if daq_freq is not None:
                #--------------------------------------------------
                save_name_i = f"zero_times_df_{opco_i}.pkl"
                save_path_i = os.path.join(save_dir_i, save_name_i)
                #-------------------------
                zero_times_df_i, files_in_dir_i = AbsPerfectPower.concat_zero_times_dfs_in_dir(
                    dir_path             = os.path.join(save_dir_i, 'partials'), 
                    regex_pattern        = r'zero_times_df_\d{8}_\d{8}.*', 
                    ignore_case          = False, 
                    ext                  = '.pkl', 
                    make_col_types_equal = False, 
                    cols_to_drop         = [
                        'island_number', 
                        'island_start_location', 
                        'island_end_location'
                    ], 
                    return_paths         = True
                )
                assert(set(files_in_dir_i).symmetric_difference(zts_saved_paths_i)==set())
                #-------------------------
                zero_times_df_i.to_pickle(save_path_i) 

                #--------------------------------------------------
                save_name_i = f"all_xNs_df_{opco_i}.pkl"
                save_path_i = os.path.join(save_dir_i, save_name_i)
                #-------------------------
                all_xNs_i, files_in_dir_i = AbsPerfectPower.concat_zero_times_dfs_in_dir(
                    dir_path             = os.path.join(save_dir_i, 'partials'), 
                    regex_pattern        = r'all_xNs_df_\d{8}_\d{8}.*', 
                    ignore_case          = False, 
                    ext                  = '.pkl', 
                    make_col_types_equal = False, 
                    cols_to_drop         = None, 
                    return_paths         = True
                )
                assert(set(files_in_dir_i).symmetric_difference(xNs_saved_paths_i)==set())
                all_xNs_i = all_xNs_i.drop_duplicates()
                #-------------------------
                all_xNs_i.to_pickle(save_path_i)    
    

    #---------------------------------------------------------------------
    @staticmethod
    def check_bins(
        bins
    ):
        r"""
        Generally, bins should be a list/tuple of 2-element list/tuples representing the inclusive minimum
          and exclusive maximum describing each bin.
          i.e., the bins are [)
        -----
        Special cases for elements of bins, bin_i:
            - bin_i is a single number (int/float/whatever)
                  ==> entries must match exactly bin_i to be included
                      i.e., it's basically a bin with no width
            - bin_i has None as one of it's elements (BUT NOT BOTH!)
                  [None, a] ==> bin from [-inf, a)
                  [a, None] ==> bin from [a, inf)
        -----
        e.g., bins = [
                        0, 
                        (1,7), 
                        (7,13), 
                        (13,25), 
                        (25, None)
                     ]
        """
        #--------------------------------------------------
        try:
            assert(Utilities.is_object_one_of_types(bins, [list, tuple]))
            for i,bin_i in enumerate(bins):
                assert(
                    Utilities.is_object_one_of_types(bin_i, [list, tuple]) or
                    Utilities.is_numeric(bin_i)
                )
                if Utilities.is_object_one_of_types(bin_i, [list, tuple]):
                    assert(len(bin_i)==2)
                    assert(np.sum([Utilities.is_numeric(x) for x in bin_i])>=1)
                    if np.sum([Utilities.is_numeric(x) for x in bin_i])!=2:
                        assert(i==0 or i==len(bins)-1)
            #-------------------------
            return True
        except:
            return False
    
    #---------------------------------------------------------------------
    @staticmethod
    def build_table_df(
        n_intr_srs     , 
        bins           = None, 
        supplmntl_data = None
    ):
        r"""
        n_intr_srs:
            Should be a series with index = serialnumbers and values equal to the number of interruptions suffered by
            the given serialnumber
    
        supplmntl_data:
            Should be a dictionary with a key equal to the bin which should be supplemented and value equal to additional counts.
            This was originally created to account for the meters with zero interruptions (see explanation below) but can be used
              to supplement any bin in general
            -----
            In general, the data for meters not experiencing any interruptions is not contained in n_intr_srs (as n_intr_srs is obtained by 
              calling .value_counts() on some pd.DataFrame).
            Therefore, these must be provided as well.
            supplmntl_data should be a dictionary with a key equal to the bin which should contain these zero interruptions counts
        """
        #--------------------------------------------------
        dflt_bins = [
            0, 
            (1,7), 
            (7,13), 
            (13,25), 
            (25, None)
        ]
        if bins is None:
            bins = dflt_bins
        #--------------------------------------------------
        if supplmntl_data is None:
            supplmntl_data = dict()
        assert(isinstance(supplmntl_data, dict))
        assert(set(supplmntl_data.keys()).difference(set(bins))==set())
    
        #--------------------------------------------------
        assert(isinstance(n_intr_srs, pd.Series))
        assert(AbsPerfectPower.check_bins(bins=bins))
        #--------------------------------------------------
        df_data = []
        for bin_i in bins:
            if Utilities.is_numeric(bin_i):
                title_i  = f"{bin_i}"
                counts_i = n_intr_srs[(n_intr_srs==bin_i)].shape[0]
            else:
                assert(len(bin_i)==2)
                #-------------------------
                if bin_i[0] is None:
                    title_i  = f"[{bin_i[1]}-)"
                    counts_i = n_intr_srs[(n_intr_srs<bin_i[1])].shape[0]
                elif bin_i[1] is None:
                    title_i  = f"[{bin_i[0]}+)"
                    counts_i = n_intr_srs[(n_intr_srs>=bin_i[0])].shape[0]
                else:
                    title_i  = f"[{bin_i[0]}-{bin_i[1]})"
                    counts_i = n_intr_srs[(n_intr_srs>=bin_i[0]) & (n_intr_srs<bin_i[1])].shape[0]
            #-------------------------
            if bin_i in supplmntl_data.keys():
                assert(Utilities.is_numeric(supplmntl_data[bin_i]))
                counts_i += supplmntl_data[bin_i]
            #-------------------------
            df_data.append((title_i, counts_i))
        #--------------------------------------------------
        if n_intr_srs.index.name:
            col_name = f"{n_intr_srs.index.name}s"
        else:
            col_name = 'Meters'
        #-------------------------
        table_df = pd.DataFrame(
            data    = df_data, 
            columns = ['n_intr', col_name]
        ).set_index('n_intr')
        table_df['% of Total'] = 100*table_df[col_name]/table_df[col_name].sum()
        table_df.loc['Total'] = table_df.sum(axis=0)
        #--------------------------------------------------
        return table_df
    

    #---------------------------------------------------------------------
    @staticmethod
    def build_acc_srvc_intrrptns_table_df(
        zero_times_df  , 
        bins           = None, 
        supplmntl_data = None, 
        #-----
        SN_col         = 'serialnumber', 
    ):
        r"""
        """
        #--------------------------------------------------
        vcs_srs = zero_times_df[SN_col].value_counts()
        table_df = AbsPerfectPower.build_table_df(
            n_intr_srs     = vcs_srs, 
            bins           = bins, 
            supplmntl_data = supplmntl_data
        )
        return table_df
    

    #---------------------------------------------------------------------
    @staticmethod
    def build_acc_outg_drtn_table_df(
        zero_times_df  , 
        bins           = None, 
        supplmntl_data = None, 
        #-----
        SN_col         = 'serialnumber', 
        island_beg_col = 'island_start_starttimeperiod', 
        island_end_col = 'island_end_starttimeperiod', 
        data_freq      = pd.Timedelta('15 min')
    ):
        r"""
        """
        #--------------------------------------------------
        if not isinstance(data_freq, pd.Timedelta):
            data_freq = pd.Timedelta(data_freq)
        assert(isinstance(data_freq, pd.Timedelta))
        #-------------------------
        nec_cols = [SN_col, island_beg_col, island_end_col]
        assert(set(nec_cols).difference(set(zero_times_df.columns))==set())
        #--------------------------------------------------
        dflt_bins = [
            (None, 6), 
            (6,24), 
            (24,48), 
            (48,96), 
            (96, None)
        ]
        if bins is None:
            bins = dflt_bins
        #--------------------------------------------------
        df_i = zero_times_df.copy()
        #-------------------------
        df_i['delta'] = pd.to_datetime(df_i[island_end_col])-pd.to_datetime(df_i[island_beg_col])+data_freq
        df_i['n_15T'] = df_i['delta']/data_freq
        #-----
        df_i = df_i.groupby([SN_col])['delta'].sum()
        df_i = df_i.div(pd.Timedelta('1 hour'))
    
        df_i = AbsPerfectPower.build_table_df(
            n_intr_srs     = df_i.copy(), 
            bins           = bins, 
            supplmntl_data = supplmntl_data
        )
    
        return df_i
    
    
    #---------------------------------------------------------------------
    @staticmethod
    def histplot_acc_srvc_intrrptns(
        zero_times_df , 
        bins                  = None, 
        supplmntl_data        = None, 
        stat                  = 'count', 
        figax                 = None, 
        fig_num               = 0, 
        set_logy              = True, 
        title                 = None, 
        include_text          = True, 
        #-----
        SN_col                = 'serialnumber', 
    ):
        r"""
        """
        #--------------------------------------------------
        table_df = AbsPerfectPower.build_acc_srvc_intrrptns_table_df(
            zero_times_df         = zero_times_df, 
            bins                  = bins, 
            supplmntl_data        = supplmntl_data, 
            SN_col                = SN_col, 
        )
        #--------------------------------------------------
        assert(isinstance(set_logy, bool))
        assert(stat in ['count', 'probability'])
        #-----
        if figax is None: 
            figax = Plot_General.default_subplots(
                fig_num = fig_num, 
                n_x     = 1, 
                n_y     = 1, 
                return_flattened_axes = False
            )
        #-------------------------
        assert(Utilities.is_object_one_of_types(figax, [list, tuple]))
        assert(len(figax)==2)
        assert(isinstance(figax[0], mpl.figure.Figure))
        assert(isinstance(figax[1], mpl.axes.Axes))
        fig,ax = figax
        
        #--------------------------------------------------
        if stat == 'count':
            y = 'serialnumbers'
        else:
            y = '% of Total'
        ax = Plot_Bar.plot_barplot(
            ax = ax, 
            df = table_df.drop(index=['Total']), 
            x  = 'n_intr', 
            y  = y
        )
    
        #--------------------------------------------------
        if title is None:
            title = ''
        #-----
        if set_logy:
            ax.set_yscale('log')
            title = f"{title} (log-y)"
        ax.set_title(title, fontsize='xx-large')
        ax.set_xlabel('Acc. Service Interruptions', fontsize='xx-large', loc='right')
        ax.set_ylabel(f'Customers/Meters (stat = {stat})', fontsize='xx-large', loc='top')
        #-------------------------
        if include_text:
            if stat=='count':
                fmt = '%.0f'
            else:
                fmt = '%.2e'
            ax.bar_label(ax.containers[0], fmt=fmt);
        #--------------------------------------------------
        return fig,ax