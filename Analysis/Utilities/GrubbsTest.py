#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from pandas.api.types import is_numeric_dtype
import csv

#---------------------------------------------------------------------
#sys.path.insert(1, os.path.join(os.path.realpath('..'), 'Utilities'))
import Utilities
import Utilities_df

#-----------------------------------------------------------------
# Decided to make this a class so I can more easily store and retain information
# However, there are still many methods which can be simply called as if it were a module

def identify_outliers_in_metric_column(df, metric, p_val=0.01, include_output_info=False, inplace=True):
    # NOTE: RETURN df, output_info_dict (NOT just df)
    # Any outliers in the column are found and identified by
    # setting equal to np.nan
    
    # Returns df, output_info_dict (optional)
    #   df is the input df with any found outliers in metric column set to np.nan
    #   output_info_dict
    #     if include_output_info==False, output_info_dict=None is returned
    #     if include_output_info==True, output_info_dict is returned with keys:
    #        'metric_name', 
    #        'outlier_value', 
    #        'outlier_ids', 
    #        'n_outliers', 
    #        'ids_where_max_d', 
    #        'g_val', 
    #        'g_crit', 
    #        'is_significant'

    if not inplace:
        df = df.copy()

    tmp_col = 'd_i' # column to store d_i = |x_i - mean|

    mean = df[metric].mean()
    std = df[metric].std()
    df[tmp_col] = abs(df[metric]-mean)

    # Cannot simply use idxmax because it only returns first idx if multiple
    #   This method should allow for multiple equal max values
    # Note: using np.asarray(condition).nonzero() as opposed to np.where(condition[,x,y])
    #   In documentation, stated this is preferable when only condition given
    #   Probably .nonzero() is unnecessary, but doesn't hurt
    max_d_val = df[tmp_col].max()
    ids_max = df.index[np.asarray(df[tmp_col]==max_d_val).nonzero()]
    metric_val_where_max_d = df.loc[ids_max[0], metric]

    n_obs = df[tmp_col].count() # count non-NA cells for tmp_col
    t_val = stats.t.ppf(1.0-p_val/(2*n_obs), n_obs-2)
    g_crit = ((n_obs-1)/np.sqrt(n_obs))*np.sqrt(t_val**2/(n_obs-2+t_val**2))
    g_val = max_d_val/std

    if g_val > g_crit:
        for id_max in ids_max:
            df.at[id_max, metric] = np.nan

    # Remove tmp_col from df
    df.drop(columns=[tmp_col], inplace=True)
    
    if include_output_info:
        # Create output info dict
        output_info_dict = {}
        output_info_dict['metric_name'] = metric
        output_info_dict['outlier_value'] = metric_val_where_max_d if g_val > g_crit else np.nan
        output_info_dict['outlier_ids']   = ids_max.tolist() if g_val > g_crit else [np.nan]
        output_info_dict['n_outliers']    = len(ids_max.tolist()) if g_val > g_crit else 0
        output_info_dict['ids_where_max_d'] = ids_max.tolist()
        output_info_dict['g_val']  = g_val
        output_info_dict['g_crit'] = g_crit
        output_info_dict['is_significant'] = True if g_val > g_crit else False
    else:
        output_info_dict = None
    return df, output_info_dict

#-----------------------------------------------------------------
def identify_outliers_in_all_metric_columns(df, p_val=0.01, metrics_of_interest=None, include_output_info=False):
    # Any outliers found and identified by setting equal to np.nan
    
    # Returns df, output_info_df (optional)
    #   df is the input df with any found outliers in columns=metrics_of_interest set to np.nan
    #   output_info_df
    #     if include_output_info==False, output_info_df=None is returned
    #     if include_output_info==True, output_info_df is returned.
    #       output_info_df is a collection of output_info_dict (from identify_outliers_in_metric_column) 
    #         for each metric in metrics_of_interest.
    #       The output_info_df is for this specific iteration of identify_outliers_in_all_metric_columns (or round of Grubbs).
    #       This is basically (a transposed version of) what is printed in the output csv at the end of each round.
    #       Each row of output_info_df is for a single metric, with columns equal to the keys of output_info_dict
    #       output_info_df.shape = (len(metrics_of_interest), len(output_info_dict))

    # if no metrics_of_interest provided, assume all numeric columns included
    if metrics_of_interest is None:
        metrics_of_interest = Utilities_df.get_numeric_columns(df)
    # Make sure metrics_of_interest are all columns in df
    assert(all(elem in df.columns.tolist() for elem in metrics_of_interest))
    
    if include_output_info:
        output_info_df = pd.DataFrame(columns=['metric_name', 'outlier_value', 'outlier_ids', 'n_outliers', 'ids_where_max_d', 'g_val', 'g_crit', 'is_significant'])
    else:
        output_info_df = None
        
    for metr in metrics_of_interest:
        df, output_info_dict = identify_outliers_in_metric_column(df, metr, p_val, include_output_info)
        if include_output_info:
            assert(set(output_info_dict.keys()) == set(output_info_df.columns))
            output_info_df = output_info_df.append(output_info_dict, ignore_index=True)
    return df, output_info_df

#-----------------------------------------------------------------
def identify_outliers_in_all_metric_columns_multiple_iterations(df, n_iterations=3, p_val=0.01, metrics_of_interest=None):
    for _ in range(n_iterations):
        df, output_info_df = identify_outliers_in_all_metric_columns(df, p_val, metrics_of_interest)
    return df

#-----------------------------------------------------------------    
def identify_outliers_in_all_metric_columns_find_all(df, p_val=0.01, metrics_of_interest=None):
    if metrics_of_interest is None:
        metrics_of_interest = Utilities_df.get_numeric_columns(df)
    # Make sure metrics_of_interest are all columns in df
    assert(all(elem in df.columns.tolist() for elem in metrics_of_interest))
    # Keep running Grubbs' until no more outliers found
    n_NaNs = df[metrics_of_interest].isna().sum().sum()
    delta_NaNs = 1
    while delta_NaNs > 0:
        df, output_info_df = identify_outliers_in_all_metric_columns(df, p_val, metrics_of_interest)
        delta_NaNs = df[metrics_of_interest].isna().sum().sum() - n_NaNs
        n_NaNs = df[metrics_of_interest].isna().sum().sum()
    return df
    
#-----------------------------------------------------------------
def get_grubbs_failing_scan_ids(df, n_iterations=3, n_outliers_to_fail=11, 
                                p_val=0.01, metrics_of_interest=None, 
                                include_output_info=False):
    df_for_grubbs = df[metrics_of_interest].copy()
    if include_output_info:
        output_info_list_of_dfs = []
    else:
        output_info_list_of_dfs = None
    for _ in range(n_iterations):
        if include_output_info:
            # Want to include df_for_grubbs before test run
            df_for_grubbs_cpy = df_for_grubbs.copy()
        df_for_grubbs, output_info_df = identify_outliers_in_all_metric_columns(df_for_grubbs, p_val, metrics_of_interest, include_output_info)
        if include_output_info: 
            output_info_list_of_dfs.append((output_info_df, df_for_grubbs_cpy))
    n_NaNs_per_scan = df_for_grubbs.isnull().sum(axis=1)
    fail_ids = [idx for idx,val in n_NaNs_per_scan.items() 
                if val >= n_outliers_to_fail]
    
    if include_output_info:
        # add final result
        output_info_list_of_dfs.append((None, df_for_grubbs))
    
    return fail_ids, output_info_list_of_dfs
    
    
def get_grubbs_failing_scan_fileNames(df, image_fileName_col = 'image_fileName', 
                                      n_iterations=3, n_outliers_to_fail=11, 
                                      p_val=0.01, metrics_of_interest=None, 
                                      include_output_info=False):
    assert(image_fileName_col in df.columns)
    fail_ids, output_info_list_of_dfs = get_grubbs_failing_scan_ids(df, n_iterations, n_outliers_to_fail, 
                                                                    p_val, metrics_of_interest, 
                                                                    include_output_info=include_output_info)
    fail_fileNames = df.loc[fail_ids, image_fileName_col].tolist()
    return fail_fileNames, output_info_list_of_dfs
    
#-----------------------------------------------------------------
def run_grubbs(df, n_iterations=3, n_outliers_to_fail=11, 
               p_val=0.01, metrics_of_interest=None):
    # First, remove any scans with missing data
    df = df.dropna(subset=metrics_of_interest, how='any')
    if df.empty:
        print('After df.dropna in GrubbsTest.run_grubbs, df is empty!!')
        print('So apparently each scan is missing at least one measurement')
        return df
    
    fail_ids, _ = get_grubbs_failing_scan_ids(df, n_iterations, n_outliers_to_fail, 
                                              p_val, metrics_of_interest, 
                                              include_output_info=False)
    df = df.drop(fail_ids)
    return df
#-----------------------------------------------------------------
def run_grubbs_with_output_info(df, n_iterations=3, n_outliers_to_fail=11, 
                                p_val=0.01, metrics_of_interest=None, drop_rows_missing_values=False):
    #TODO drop_rows_missing_values is new from original development
    # The default behavior before was essentially for this to be set to true
    if metrics_of_interest is None:
        metrics_of_interest = Utilities_df.get_numeric_columns(df)
    # Make sure metrics_of_interest are all columns in df
    assert(all(elem in df.columns.tolist() for elem in metrics_of_interest))
                                
    include_output_info = True
    fails_df = pd.DataFrame()
    if drop_rows_missing_values:
        # First, remove any scans with missing data
        df_w_missing_removed = df.dropna(subset=metrics_of_interest, how='any')
        if df.shape[0] != df_w_missing_removed.shape[0]:
            idx_full = df.index.tolist()
            idx_w_missing_removed = df_w_missing_removed.index.tolist()
            idx_missing = Utilities.get_two_lists_diff(idx_full, idx_w_missing_removed)
            fails_df = df.loc[idx_missing]
        df = df_w_missing_removed
    
    n_NaNs_before = df[metrics_of_interest].isna().sum().sum()
    if drop_rows_missing_values:
        assert(n_NaNs_before==0) #in future, I may allow this
    fail_ids, output_info_dfs = get_grubbs_failing_scan_ids(df, n_iterations, n_outliers_to_fail, 
                                                            p_val, metrics_of_interest, 
                                                            include_output_info=include_output_info)
    n_NaNs_after = output_info_dfs[-1][1][metrics_of_interest].isna().sum().sum()
    n_outliers = n_NaNs_after-n_NaNs_before
    if fails_df.shape[0] > 0:
        assert(fails_df.columns.tolist()==df.loc[fail_ids].columns.tolist())
        #fails_df = fails_df.append(df.loc[fail_ids], ignore_index=True) #TODO Not sure why I had ignore_index=True?
        fails_df = fails_df.append(df.loc[fail_ids], ignore_index=False)
    else:
        fails_df = df.loc[fail_ids]
    df = df.drop(fail_ids)
    return df, (fails_df, output_info_dfs, n_outliers)
#-----------------------------------------------------------------
def get_round_summary_from_info_df(info_df):
    total_n_outliers = info_df['n_outliers'].sum()
    outlier_scans = [idx for metric_row in info_df['outlier_ids'].tolist() 
                     for idx in metric_row if not np.isnan(idx)]
    outlier_scans = set(outlier_scans)
    total_n_scans_w_outliers = len(outlier_scans)
    return_dict = {'total_n_outliers':total_n_outliers, 
                   'total_n_scans_w_outliers':total_n_scans_w_outliers, 
                   'outlier_scans':outlier_scans}
    return return_dict
#-----------------------------------------------------------------
def get_all_rounds_summaries(output_info_dfs):
    all_summaries = []
    for i,output_info in enumerate(output_info_dfs):
        is_final = True if i==len(output_info_dfs)-1 else False
        if not is_final:
            info_df = output_info[0]
            summary = get_round_summary_from_info_df(info_df)
            all_summaries.append(summary)
    return all_summaries       
#-----------------------------------------------------------------
def run_grubbs_and_write_output(df, output_path, 
                                n_iterations=3, n_outliers_to_fail=11, 
                                p_val=0.01, metrics_of_interest=None, drop_rows_missing_values=False): 
    #TODO drop_rows_missing_values is new from original development
    # The default behavior before was essentially for this to be set to true
    df, (fails_df, output_info_dfs, n_outliers) = run_grubbs_with_output_info(
                                                      df, 
                                                      n_iterations=n_iterations, 
                                                      n_outliers_to_fail=n_outliers_to_fail, 
                                                      p_val=p_val, 
                                                      metrics_of_interest=metrics_of_interest, 
                                                      drop_rows_missing_values=drop_rows_missing_values)
    fail_ids = sorted(fails_df.index.tolist())
    #newline='' needed in Windows, otherwise extra \n added after each line!
    with open(output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        all_summaries = get_all_rounds_summaries(output_info_dfs)
        csv_writer.writerow(['No. scans with > 10 outliers', len(fail_ids)])
        csv_writer.writerow(['Scan ID#s with > 10 outliers']+fail_ids)
        csv_writer.writerow(['Round', 'No. outliers', 'No. scans with at least one outlier'])
        total_n_outliers_all_rounds = 0
        for i,summary in enumerate(all_summaries):
            total_n_outliers_all_rounds += summary['total_n_outliers']
            csv_writer.writerow([i+1, summary['total_n_outliers'], summary['total_n_scans_w_outliers']])
        assert(total_n_outliers_all_rounds==n_outliers)
        csv_writer.writerow([])
        csv_writer.writerow([])
        for i,output_info in enumerate(output_info_dfs):
            is_final = True if i==len(output_info_dfs)-1 else False
            info_df = output_info[0]
            metr_df = output_info[1]
            if is_final:
                csv_writer.writerow(['FINAL COLLECTION'])
            else:
                assert(info_df.T.loc['metric_name'].tolist()==metr_df.columns.tolist())
                csv_writer.writerow(['Round', i+1])
            row = ['']+[i+1 for i in range(metr_df.shape[1])]
            csv_writer.writerow(row)
            metr_df.to_csv(csv_file)
            csv_writer.writerow([])
            if not is_final:
                info_df.T.to_csv(csv_file, header=False)
                csv_writer.writerow([])
                
                total_n_outliers = all_summaries[i]['total_n_outliers']
                total_n_scans_w_outliers = all_summaries[i]['total_n_scans_w_outliers']
                outlier_scans = all_summaries[i]['outlier_scans']
                
                csv_writer.writerow(['No. outliers found', total_n_outliers])
                csv_writer.writerow(['No. scans with at least one outlier', total_n_scans_w_outliers])
                csv_writer.writerow(['Scan ID#s with at least one outlier']+sorted(list(outlier_scans)))
                
                csv_writer.writerow([])
                csv_writer.writerow([])
    return df, (fails_df, output_info_dfs, n_outliers)