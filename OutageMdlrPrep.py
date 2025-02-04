#!/usr/bin/env python

r"""
Holds OutageModeler class.  See OutageMdlrPrep.OutageMdlrPrep for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#---------------------------------------------------------------------
import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
import datetime
import time
from natsort import natsorted


#---------------------------------------------------------------------
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
#-----
from MeterPremise import MeterPremise
#-----
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from GenAn import GenAn
from AMIEndEvents import AMIEndEvents
from MECPODf import MECPODf
from MECPOAn import MECPOAn
from MECPOCollection import MECPOCollection
from DOVSOutages import DOVSOutages
from OutageDAQ import OutageDataInfo as ODI
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import TableInfos
#---------------------------------------------------------------------
#sys.path.insert(0, os.path.join(os.path.realpath('..'), 'Utilities'))
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Utilities_dt

#---------------------------------------------------------------------

class OutageMdlrPrep:
    r"""
    Class to construct outage model
    """
    def __init__(
        self,
        dataset         , 
        run_date        , 
        date_0          ,  
        date_1          , 
        data_evs_sum_vw = False, 
        data_base_dir   = None, 
        verbose         = True


    ):
        r"""
        run_date:
            Date of data acquistion.
            Expected format: yyyymmdd

        date_0/date_1:
            Define the time period of the data acquired, i.e., data for the selected meters were 
              returned for the dates defined by [date_0, date_1]
            Expected format: yyyy-mm-dd

        data_base_dir:
            Base directory for the data.
            The EndEvents/EvsSums data must be located in:
                os.path.join(data_base_dir, ODI.get_subdir(dataset), 'EndEvents'/'EvsSums')
            If data_base_dir is None:
                data_base_dir = os.path.join(
                    Utilities.get_local_data_dir(), 
                    r'dovs_and_end_events_data', 
                    self.run_date, 
                    f"{self.date_0.replace('-','')}_{self.date_1.replace('-','')}"
                )

        data_evs_sum_vw:
            Indicates whether or not the data being prepared are from the meter_events.events_summary_vw (data_evs_sum_vw=True) table
              or the meter_events.end_device_event (data_evs_sum_vw=False) data.
            The latter case (data_evs_sum_vw=False) is what the bulk of this class is intended to handle
        """
        #---------------------------------------------------------------------------
        ODI.assert_dataset(dataset)
        self.dataset         = dataset
        self.data_evs_sum_vw = data_evs_sum_vw
        #-------------------------
        self.run_date = pd.to_datetime(run_date).strftime('%Y%m%d')
        self.date_0   = pd.to_datetime(date_0).strftime('%Y-%m-%d')
        self.date_1   = pd.to_datetime(date_1).strftime('%Y-%m-%d')
        #-----
        self.verbose  = verbose
        #-------------------------
        if data_base_dir is None:
            date_pd_subdir = f"{self.date_0.replace('-','')}_{self.date_1.replace('-','')}"
            self.data_base_dir = os.path.join(
                Utilities.get_local_data_dir(), 
                r'dovs_and_end_events_data', 
                self.run_date, 
                date_pd_subdir
            )            
        else:
            self.data_base_dir = data_base_dir
        #-----
        self.data_base_dir = os.path.join(self.data_base_dir, ODI.get_subdir(self.dataset))
        #-----
        if not os.path.isdir(self.data_base_dir):
            print(f'ERROR: data_base_dir = {self.data_base_dir} DOES NOT EXIST!')
            assert(0)
        #-------------------------
        if self.data_evs_sum_vw:
            self.files_dir = os.path.join(self.data_base_dir, 'EvsSums')
        else:
            self.files_dir = os.path.join(self.data_base_dir, 'EndEvents')
        #-----
        if not os.path.isdir(self.files_dir):
            print(f'ERROR: files_dir = {self.files_dir} DOES NOT EXIST!')
            assert(0)
        #-------------------------
        self.naming_tag   = ODI.get_naming_tag(self.dataset)
        self.is_no_outage = ODI.get_is_no_outage(self.dataset)
        #-------------------------
        if self.verbose:
            print(f'data_base_dir = {self.data_base_dir}')
            print(f'files_dir     = {self.files_dir}')
            print(f'naming_tag    = {self.naming_tag}')
            print(f'is_no_outage  = {self.is_no_outage}')



    #------------------------------------------------------------------------------------------------------------------------------------------------------
    # OLD FUNCTIONS TO BE UPDATED
    #---------------------------------------------------------------------------
    @staticmethod
    def get_active_SNs_for_xfmrs_OLD(
        trsf_pole_nbs, 
        df_mp_curr, 
        df_mp_hist,
        no_outg_time_infos_df=None, 
        addtnl_mp_df_curr_cols=None, 
        addtnl_mp_df_hist_cols=None, 
        files_dir_no_outg=r'C:\Users\s346557\Documents\LocalData\dovs_and_end_events_data\EndEvents_NoOutg', 
        file_path_glob_no_outg = r'end_events_[0-9]*.csv', 
        return_SNs_col='SNs', 
        return_prem_nbs_col='prem_nbs', 
        assert_all_trsf_pole_nbs_found=True, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb', 
        t_min_col='t_min', 
        t_max_col='t_max'
    ):
        r"""
        Difficulty is that default.meter_premise_hist does not have trsf_pole_nb field.
        Therefore, one must use default.meter_premise to find the premise numbers for xfrms in trsf_pole_nbs,
          then use those PNs to select the correct entries from default.meter_premise_hist.
        
        If df_mp_curr OR df_mp_hist is not supplied, both will be built!
        
        addtnl_mp_df_curr_cols/addtnl_mp_df_hist_cols:
          Only used when df_mp_curr/df_mp_hist not supplied and therefore need to be built
          
        If no_outg_time_infos_df is not supplied, it will be built.
          files_dir_no_outg and file_path_glob_no_outg are only used when no_outg_time_infos_df needs built
        """
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        #-------------------------
        if df_mp_curr is None or df_mp_hist is None:
            mp_df_curr_hist = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
                trsf_pole_nbs, 
                join_curr_hist=False, 
                addtnl_mp_df_curr_cols=addtnl_mp_df_curr_cols, 
                addtnl_mp_df_hist_cols=addtnl_mp_df_hist_cols, 
                df_mp_serial_number_col=df_mp_serial_number_col, 
                df_mp_prem_nb_col=df_mp_prem_nb_col, 
                df_mp_install_time_col=df_mp_install_time_col, 
                df_mp_removal_time_col=df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
            )
            df_mp_curr = mp_df_curr_hist['mp_df_curr']
            df_mp_hist = mp_df_curr_hist['mp_df_hist']
        #-------------------------
        # At a bare minimum, df_mp_curr and df_mp_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in df_mp_curr.columns for x in necessary_mp_cols+[df_mp_trsf_pole_nb_col]]))
        assert(all([x in df_mp_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        # PNs_for_xfmrs is a DF with trsf_pole_nbs indices and elements which are lists of PNs for each xfmr
        PNs_for_xfmrs = MeterPremise.get_SNs_andor_PNs_for_xfmrs(
            trsf_pole_nbs=trsf_pole_nbs, 
            include_SNs=False,
            include_PNs=True,
            trsf_pole_nb_col=df_mp_trsf_pole_nb_col, 
            serial_number_col=df_mp_serial_number_col, 
            prem_nb_col=df_mp_prem_nb_col, 
            return_SNs_col=None, #Not grabbing SNs
            return_PNs_col=return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found=assert_all_trsf_pole_nbs_found, 
            mp_df=df_mp_curr, 
            return_mp_df_also=False
        )
        #-------------------------
        # Instead of a DF with trsf_pole_nb index and prem_nb column, we want opposite
        xfmr_for_PNs_df = PNs_for_xfmrs.explode(return_prem_nbs_col)
        xfmr_for_PNs_df[xfmr_for_PNs_df.index.name] = xfmr_for_PNs_df.index
        xfmr_for_PNs_df = xfmr_for_PNs_df.set_index(return_prem_nbs_col)  
        #-------------------------
        # If no_outg_time_infos_df is None, build it.  
        #   no_outg_time_infos_df has prem_nbs indices and t_min, t_max columns
        #   This is where the time information for each premise number comes from
        if no_outg_time_infos_df is None:
            paths_no_outg = Utilities.find_all_paths(base_dir=files_dir_no_outg, glob_pattern=file_path_glob_no_outg)
            no_outg_time_infos_df = MECPOAn.get_bsln_time_interval_infos_df_from_summary_files(
                summary_paths=[AMIEndEvents.find_summary_file_from_csv(x) for x in paths_no_outg], 
                output_prem_nbs_col=return_prem_nbs_col, 
                output_t_min_col=t_min_col, 
                output_t_max_col=t_max_col, 
                make_prem_nbs_idx=True, 
                include_summary_paths=False
            )    
        #-------------------------
        # Merge xfmr_for_PNs_df with no_outg_time_infos_df to append the time data to the former
        # NOTE: It is possible for t_min/t_max to be NaT (NaN) for some entries after the merge, meaning that the 
        #       premise numbers were not found in no_outg_time_infos_df
        #       This happens because these premise numbers must not have had any meter events, and thus were not included 
        #         in the SQL query (as it takes a long time to find empty results, so I weed these out before running the 
        #         query), and therefore the premise numbers were not found in the summary files/no_outg_time_infos_df.    
        xfmr_for_PNs_df = pd.merge(xfmr_for_PNs_df, no_outg_time_infos_df, how='left', left_index=True, right_index=True)
        #-------------------------
        # Want to consolidate xfmr_for_PNs_df, grouping by trsf_pole_nb and collecting t_min,t_max, and a list
        # of the premise numbers.  Therefore, first the index must be reset to make a PNs columns
        xfmr_for_PNs_df=xfmr_for_PNs_df.reset_index()
        #-----
        # Consolidate xfmr_for_PNs_df
        # NOTE: If t_min/t_max is NaT (NaN) for all premise numbers in a given trsf_pole_nb (see NOTE above before merge
        #       with no_outg_time_infos_df), then Utilities_df.consolidate_df will return an empty list (technically, an
        #       empty np.ndarray) for that trsf_pole_nb
        xfmr_for_PNs_df=Utilities_df.consolidate_df_OLD(
            df=xfmr_for_PNs_df, 
            groupby_col=df_mp_trsf_pole_nb_col, 
            cols_shared_by_group=[t_min_col, t_max_col], 
            cols_to_collect_in_lists=[return_prem_nbs_col]
        )    
        #--------------------------------------------------
        # Only reason for making dict is to ensure trsf_pole_nbs are not repeated 
        active_SNs_in_xfmrs_dfs_dict = {}
    
        for trsf_pole_nb_i, row_i in xfmr_for_PNs_df.iterrows():
            # active_SNs_df_i will have indices equal to premise numbers and value equal to lists
            #   of active SNs for each PN
            PNs_i=row_i[return_prem_nbs_col]
            dt_0_i=row_i[t_min_col]
            dt_1_i=row_i[t_max_col]
            # See NOTEs above regarding t_min/t_max being empty
            # In such a case, it is simply impossibe (with the summary files currently generated) to access
            #   the date over which the data would have been run, if any events existed.
            #   In future versions, this information will be included in the summary files!
            # I don't want to completely exclude these (by e.g., setting dt_0_i=pd.Timestamp.min and 
            #   dt_1_i=pd.Timestamp.max), so I will simply include the meters which are active TODAY.
            # This obviously is not correct, but this occurrence is rare (only happening when every single meter
            #   on a transformer had no events during the time period) and this crude approximation will be fine.
            if Utilities.is_object_one_of_types(dt_0_i, [list, np.ndarray]):
                assert(len(dt_0_i)==0)
                # I believe if this happens for one it should happen for both...
                assert(Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]) and len(dt_1_i)==0)
                dt_0_i=pd.Timestamp.today()
            if Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]):
                assert(len(dt_1_i)==0)
                # I believe if this happens for one it should happen for both...
                # But, dt_0_i changed already above, so much check row_i[t_min_col] instead!
                assert(Utilities.is_object_one_of_types(row_i[t_min_col], [list, np.ndarray]) and len(row_i[t_min_col])==0)
                dt_1_i=pd.Timestamp.today()
            if((not isinstance(PNs_i, list) and pd.isna(PNs_i)) or 
               len(PNs_i)==0):
                active_SNs_df_i = pd.DataFrame()
            else:
                active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                    PNs=PNs_i,
                    df_mp_curr=df_mp_curr, 
                    df_mp_hist=df_mp_hist, 
                    dt_0=dt_0_i,
                    dt_1=dt_1_i,
                    output_index=None,
                    output_groupby=[df_mp_prem_nb_col], 
                    include_prems_wo_active_SNs_when_groupby=True, 
                    assert_all_PNs_found=False
                )
                active_SNs_df_i=active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col] = np.nan
                active_SNs_df_i[df_mp_serial_number_col] = [[]] 
            active_SNs_df_i[df_mp_trsf_pole_nb_col] = trsf_pole_nb_i
            active_SNs_df_i = active_SNs_df_i.explode(df_mp_serial_number_col)
            assert(trsf_pole_nb_i not in active_SNs_in_xfmrs_dfs_dict)
            active_SNs_in_xfmrs_dfs_dict[trsf_pole_nb_i] = active_SNs_df_i
        #-------------------------
        active_SNs_df = pd.concat(list(active_SNs_in_xfmrs_dfs_dict.values()))
        #-------------------------
        active_SNs_df = Utilities_df.consolidate_df_OLD(
            df=active_SNs_df, 
            groupby_col=df_mp_trsf_pole_nb_col, 
            cols_shared_by_group=None, 
            cols_to_collect_in_lists=[df_mp_serial_number_col, df_mp_prem_nb_col], 
            include_groupby_col_in_output_cols=False, 
            allow_duplicates_in_lists=False, 
            recover_uniqueness_violators=True, 
            rename_cols=None, 
            verbose=False
        )
        #-----
        # Change [nan] entries to []
        active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0, df_mp_serial_number_col] = [[]]
        active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0, df_mp_prem_nb_col] = [[]]
        #-------------------------
        active_SNs_df = active_SNs_df.rename(columns={
            df_mp_prem_nb_col:return_prem_nbs_col, 
            df_mp_serial_number_col:return_SNs_col
        })
        #-------------------------
        return active_SNs_df
    
    #---------------------------------------------------------------------------
    @staticmethod
    def add_xfmr_active_SNs_to_rcpo_df_OLD(
        rcpo_df, 
        set_xfmr_nSNs=True, 
        include_active_xfmr_PNs=False, #Should be equal to the PNs already in rcpo_df!
        df_mp_curr=None,
        df_mp_hist=None, 
        no_outg_time_infos_df=None, 
        addtnl_get_active_SNs_for_xfmrs_kwargs=None, 
        xfmr_SNs_col='_xfmr_SNs', 
        xfmr_nSNs_col='_xfmr_nSNs', 
        xfmr_PNs_col='_xfmr_PNs', 
        xfmr_nPNs_col='_xfmr_nPNs', 
    ):
        r"""
        NOTE: If include_active_xfmr_PNs is True, this column (named xfmr_PNs_col='_xfmr_SNs') should be 
              equal to the PNs already in rcpo_df!
        NOTE: xfmr_SNs_col, xfmr_nSNs_col, xfmr_PNs_col, and xfmr_nPNs_col should all be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
              
        NOTE: If any of xfmr_SNs_col, xfmr_nSNs_col, xfmr_PNs_col, and xfmr_nPNs_col are already contained in 
              rcpo_df, they will be replaced.  This is needed so that the merge operation does not come back with _x and _y
              values.  So, one should make sure this function call is truly needed, as grabbing the serial numbers for the
              outages typically takes a couple/few minutes.
              
        NOTE: To make things run faster, the user can supply df_mp_curr and df_mp_hist.  These will be included in 
              get_active_SNs_for_xfmrs_kwargs.
              NOTE: If df_mp_curr/df_mp_hist is also supplied in addtnl_get_active_SNs_for_xfmrs_kwargs,
                    that/those in addtnl_get_active_SNs_for_xfmrs_kwargs will ultimately be used (not the
                    explicity df_mp_hist/curr in the function arguments!)
              CAREFUL: If one does supple df_mp_curr/hist, one must be certain these DFs contain all necessary elements!
        """
        #-------------------------
        get_active_SNs_for_xfmrs_kwargs = dict(
            trsf_pole_nbs=rcpo_df.index.unique().tolist(), 
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist, 
            no_outg_time_infos_df=no_outg_time_infos_df, 
            return_prem_nbs_col=xfmr_PNs_col, 
            return_SNs_col=xfmr_SNs_col
        )
        if addtnl_get_active_SNs_for_xfmrs_kwargs is not None:
            get_active_SNs_for_xfmrs_kwargs = {**get_active_SNs_for_xfmrs_kwargs, 
                                               **addtnl_get_active_SNs_for_xfmrs_kwargs}
        active_SNs_df = OutageMdlrPrep.get_active_SNs_for_xfmrs_OLD(**get_active_SNs_for_xfmrs_kwargs)
        assert(isinstance(active_SNs_df, pd.DataFrame))
        #-------------------------
        # Assert below might be too strong here...
        assert(sorted(rcpo_df.index.unique().tolist())==sorted(active_SNs_df.index.unique().tolist()))
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            #----------
            # See note above about columns being replaced/dropped
            cols_to_drop = [x for x in rcpo_df.columns if x in active_SNs_df.columns]
            if len(cols_to_drop)>0:
                rcpo_df = rcpo_df.drop(columns=cols_to_drop)
            #----------
            rcpo_df = rcpo_df.merge(active_SNs_df, left_index=True, right_index=True)
            #----------
            if set_xfmr_nSNs:
                rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, xfmr_SNs_col, xfmr_nSNs_col)
                if include_active_xfmr_PNs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, xfmr_PNs_col, xfmr_nPNs_col)
        else:
            # Currently, only expecting raw and/or norm.  No problem to allow more, but for now keep this to alert 
            # of anything unexpected
            assert(rcpo_df.columns.get_level_values(0).nunique()<=2)
            for i,level_0_val in enumerate(rcpo_df.columns.get_level_values(0).unique()):
                if i==0:
                    active_SNs_df.columns = pd.MultiIndex.from_product([[level_0_val], active_SNs_df.columns])
                else:
                    active_SNs_df.columns = active_SNs_df.columns.set_levels([level_0_val], level=0)
                #----------
                # See note above about columns being replaced/dropped
                cols_to_drop = [x for x in rcpo_df.columns if x in active_SNs_df.columns]
                if len(cols_to_drop)>0:
                    rcpo_df = rcpo_df.drop(columns=cols_to_drop)
                #----------
                rcpo_df = rcpo_df.merge(active_SNs_df, left_index=True, right_index=True)
                #----------
                if set_xfmr_nSNs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, xfmr_SNs_col), (level_0_val, xfmr_nSNs_col))
                    if include_active_xfmr_PNs:
                        rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, xfmr_PNs_col), (level_0_val, xfmr_nPNs_col))
        #-------------------------
        rcpo_df = rcpo_df.sort_index(axis=1,level=0)
        #-------------------------
        return rcpo_df



    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpo_df_norm_by_xfmr_active_nSNs_OLD(
        rcpo_df_raw, 
        xfmr_nSNs_col='_xfmr_nSNs', 
        xfmr_SNs_col='_xfmr_SNs', 
        other_SNs_col_tags_to_ignore=['_SNs', '_nSNs', '_prem_nbs', '_nprem_nbs', '_xfmr_PNs', '_xfmr_nPNs'], 
        drop_xfmr_nSNs_eq_0=True, 
        new_level_0_val='counts_norm_by_xfmr_nSNs', 
        remove_SNs_cols=False, 
        df_mp_curr=None,
        df_mp_hist=None, 
        no_outg_time_infos_df=None, 
        addtnl_get_active_SNs_for_xfmrs_kwargs=None
    ):
        r"""
        Build rcpo_df normalized by the number of serial numbers in each outage
    
        drop_xfmr_nSNs_eq_0:
          It is possible for the number of serial numbers in an outage to be zero!
          Premise numbers are always found, but the meter_premise database does not always 
            contain the premise numbers.
          Dividing by zero will make all counts for such an entry equal to NaN or inf.
          When drop_xfmr_nSNs_eq_0 is True, such entries will be removed.
    
        NOTE: xfmr_SNs_col and xfmr_nSNs_col should both be strings, not tuples.
              If column is MultiIndex, the level_0 value will be handled below.
        """
        #-------------------------
        n_counts_col = xfmr_nSNs_col
        list_col = xfmr_SNs_col
        #-------------------------
        # NOTE: MECPODf.add_outage_SNs_to_rcpo_df expects xfmr_SNs_col and xfmr_nSNs_col to be strings, not tuples
        #       as it handles the level 0 values if they exist.  So, if tuples, use only highest level values (i.e., level 1)
        assert(Utilities.is_object_one_of_types(list_col, [str, tuple]))
        assert(Utilities.is_object_one_of_types(n_counts_col, [str, tuple]))
        #-----
        add_list_col_to_rcpo_df_func = OutageMdlrPrep.add_xfmr_active_SNs_to_rcpo_df_OLD
        add_list_col_to_rcpo_df_kwargs = dict(
            set_xfmr_nSNs=True, 
            include_active_xfmr_PNs=True, 
            df_mp_curr=df_mp_curr,
            df_mp_hist=df_mp_hist, 
            no_outg_time_infos_df=no_outg_time_infos_df, 
            addtnl_get_active_SNs_for_xfmrs_kwargs=addtnl_get_active_SNs_for_xfmrs_kwargs, 
            xfmr_SNs_col='_xfmr_SNs', 
            xfmr_nSNs_col='_xfmr_nSNs', 
            xfmr_PNs_col='_xfmr_PNs', 
            xfmr_nPNs_col='_xfmr_nPNs', 
        )
        #-------------------------
        other_col_tags_to_ignore = other_SNs_col_tags_to_ignore
        drop_n_counts_eq_0 = drop_xfmr_nSNs_eq_0
        new_level_0_val = new_level_0_val
        remove_ignored_cols = remove_SNs_cols
        #-------------------------
        return MECPODf.build_rcpo_df_norm_by_list_counts(
            rcpo_df_raw=rcpo_df_raw, 
            n_counts_col=n_counts_col, 
            list_col=list_col, 
            add_list_col_to_rcpo_df_func=add_list_col_to_rcpo_df_func, 
            add_list_col_to_rcpo_df_kwargs=add_list_col_to_rcpo_df_kwargs, 
            other_col_tags_to_ignore=other_col_tags_to_ignore, 
            drop_n_counts_eq_0=drop_n_counts_eq_0, 
            new_level_0_val=new_level_0_val, 
            remove_ignored_cols=remove_ignored_cols
        )


    #------------------------------------------------------------------------------------------------------------------------------------------------------
    # NEW FUNCTIONS
    #---------------------------------------------------------------------------
    @staticmethod
    def reset_index_and_identify_cols_to_merge(
        df, 
        merge_on, 
        tag_for_idx_names=None
    ):
        r"""
        Designed to work with OutageMdlrPrep.merge_rcpo_and_df (but should definitely be useful elsewhere), which 
        allows the user to join the DFs by columns, specific index levels or any mixture of the two.
        
        In order to achieve this type of general merging, it is easiest to call reset_index, making everything
          to be merged on a column.
        HOWEVER, in order to keep track of the original indices, which likely will be restored after the merge,
          it is important for all index levels to have a name.
          
        merge_on:
            This is a list that directs the columns/indices to be used in the join.
            Column identifiers:
                Single strings for normal DF, lists/tuples for MultiIndex columns
            Index identifiers:
                f'index_{idx_level}' to specify index level by number
                ('index', idx_level_name) to specify index level by name
                
        RETURNS:
            A dict object with keys = ['df', 'df_idx_names_OG', 'df_idx_names', 'reset_merge_on']
        """
        #-------------------------
        df=df.copy()
        #-------------------------
        assert(Utilities.is_object_one_of_types(merge_on, [list, tuple]))
        #-------------------------
        # Make sure all index levels have names!
        # If the level has a name, the code below will leave it unchanged in df_idx_names
        # If it does not have a name, it will be names f'index_{idx_level}', where idx_level 
        #   is the index level number
        df_idx_names_OG = list(df.index.names)
        df_idx_names = [x if x is not None else f'index_{i}' 
                        for i,x in enumerate(df_idx_names_OG)]
        if tag_for_idx_names is not None:
            df_idx_names = [f'{x}_{tag_for_idx_names}' for x in df_idx_names]
        df.index.names=df_idx_names
        #-------------------------
        reset_merge_on = []
        for idfr in merge_on:
            assert(Utilities.is_object_one_of_types(idfr, [str, list, tuple]))
            if idfr in df.columns:
                reset_merge_on.append(idfr)
            else:
                # Must be in indices!
                if isinstance(idfr, str):
                    assert(idfr.startswith('index'))
                    if idfr=='index':
                        idfr_idx_lvl=0
                    else:
                        idfr_idx_lvl = re.findall('index_(\d*)', idfr)
                        assert(len(idfr_idx_lvl)==1)
                        idfr_idx_lvl=idfr_idx_lvl[0]
                        idfr_idx_lvl=int(idfr_idx_lvl)
                else:
                    assert(len(idfr)==2)
                    assert(idfr[0]=='index')
                    idx_level_name = idfr[1]
                    # If tag_for_idx_names, df.index.names already changed, so idx_level_name must be adjusted
                    if tag_for_idx_names is not None:
                        idx_level_name = f'{idx_level_name}_{tag_for_idx_names}'
                    assert(idx_level_name in df.index.names)
                    idfr_idx_lvl = df.index.names.index(idx_level_name)
                #---------------
                assert(idfr_idx_lvl < df.index.nlevels)
                reset_merge_on_i = df.index.names[idfr_idx_lvl]
                # NOTE: If df.columns.nlevels>1, then calling df.reset_index() below
                #       will make the bottom level reset_merge_on_i and all the rest ''
                #       e.g., if nlevels=2, after df.reset_index(), reset_merge_on_i--> (reset_merge_on_i, '')
                if df.columns.nlevels>1:
                    reset_merge_on_i=tuple([reset_merge_on_i] + ['']*(df.columns.nlevels-1))
                reset_merge_on.append(reset_merge_on_i)
        #-------------------------
        # Call reset_index on df, making all indices into columns, and double check that all reset_merge_on
        #   are contained in the columns
        df = df.reset_index()
        assert(len(set(reset_merge_on).difference(set(df.columns)))==0)
        #-------------------------
        return dict(
            df=df, 
            df_idx_names_OG=df_idx_names_OG, 
            df_idx_names=df_idx_names, 
            reset_merge_on=reset_merge_on
        )
    
    #---------------------------------------------------------------------------
    @staticmethod
    def merge_rcpo_and_df(
        rcpo_df, 
        df_2, 
        rcpo_df_on,
        df_2_on, 
        how='left'
    ):
        r"""
        Merge together rcpo_df and df_2 dfs.
        Designed specifically for rcpo_df and time_infos_df.
        
        rcpo_df_on/df_2_on:
            These are lists which direct the columns/indices to be used in the join.
            Column identifiers:
                Single strings for normal DF, lists/tuples for MultiIndex columns
            Index identifiers:
                f'index_{idx_level}' to specify index level by number
                ('index', idx_level_name) to specify index level by name
                
            NOTE: Calling reset_index() on both DFs will be the easiest method for this type of general merging, in
                  which the DFs can be merged by columns, specific index levels or any mixture of the two.
                  This is done through the use of OutageMdlrPrep.reset_index_and_identify_cols_to_merge
        """
        #--------------------------------------------------
        # Only expecting at most 2 levels in columns (e.g., counts or counts_norm as level 0, and reason as level 1)
        #     - probably not a necessary assertion, if function to be expanded later
        #   Ultimately, the number of levels in df_2 will match that of rcpo_df, so making the same
        #     assertion on df_2
        assert((rcpo_df.columns.nlevels <= 2) and (df_2.columns.nlevels <= 2))
        if rcpo_df.columns.nlevels==2:
            # If rcpo_df has two column levels, df_2 must also for the proper merging to occur
            #   --Merging with unequal levels will cause all levels to be collapsed down to single dimension
            #-----
            # Expecting at most 2 unique values for level 0 (definitely not a necessary assertion)
            assert(rcpo_df.columns.get_level_values(0).nunique()<=2)
            # In this case, likely df_2 has only single level columns.
            #   For proper merge, the number of levels should match
            if df_2.columns.nlevels==1:
                level_0_vals = rcpo_df.columns.get_level_values(0).unique().tolist()
                # Add new top level to df_2 with value equal to level_0_vals[0]
                df_2=Utilities_df.prepend_level_to_MultiIndex(
                    df=df_2, level_val=level_0_vals[0], level_name=None, axis=1
                )
                if len(level_0_vals)>1:
                    # Grab df without new column level to be copied to other new column level values
                    # NOTE: If extra [] placed around level_0_vals[0] below, the new column level would be returned
                    #       (which is not desired here!)
                    df_0 = df_2[level_0_vals[0]]
                    df_0_cols = df_0.columns.tolist()
                    # Reproduce the entries of df_2 for all column level 0 values in rcpo_df
                    for i_lvl in range(1,len(level_0_vals)):
                        new_cols = pd.MultiIndex.from_product([[level_0_vals[i_lvl]], df_0_cols])
                        df_2[new_cols] = df_0.copy()
            # Now, at this stage, rcpo_df and df_2 should have the same number of column levels
            #   and should have overlapping level 0 values if nlevels>1
            assert((rcpo_df.columns.nlevels <= 2) and (df_2.columns.nlevels <= 2)) #not needed
            assert(rcpo_df.columns.nlevels == df_2.columns.nlevels)
            if rcpo_df.columns.nlevels == 2:
                # Make sure overlapping 0 values.  
                # I suppose user could supply df_2 with only a single level 0 value while rcpo_df has
                #   two values.  In this case, only the single value would be merged to rcpo_df
                assert(len(set(df_2.columns.get_level_values(0).unique().tolist()).difference(
                    set(rcpo_df.columns.get_level_values(0).unique().tolist())
                ))==0)
        else:
            assert(rcpo_df.columns.nlevels==df_2.columns.nlevels==1)
        #--------------------------------------------------
        reset_rcpo_df_dict = OutageMdlrPrep.reset_index_and_identify_cols_to_merge(
            df=rcpo_df, 
            merge_on=rcpo_df_on, 
            tag_for_idx_names='from_rcpo_df'
        )
    
        reset_df_2_dict = OutageMdlrPrep.reset_index_and_identify_cols_to_merge(
            df=df_2, 
            merge_on=df_2_on, 
            tag_for_idx_names='from_df_2'
        )
        #-------------------------
        merged_df = pd.merge(
            reset_rcpo_df_dict['df'], 
            reset_df_2_dict['df'], 
            left_on=reset_rcpo_df_dict['reset_merge_on'], 
            right_on = reset_df_2_dict['reset_merge_on'], 
            how=how
        )
        #-------------------------
        # Want to set index first before changing names back to originals, as the
        #   originals could have been None
        merged_df = merged_df.set_index(reset_rcpo_df_dict['df_idx_names'])
    
        # Two lines below (instead of calling simply df.index.names=df_idx_names_OG) 
        #   ensures the order is the same
        rcpo_df_rename_dict = dict(zip(reset_rcpo_df_dict['df_idx_names'], reset_rcpo_df_dict['df_idx_names_OG']))
        merged_df.index.names = [rcpo_df_rename_dict[x] for x in merged_df.index.names]
        #-------------------------
        # When merging two columns whose names are different, both columns are kept
        #   This is redundant, as these will have identical values, as they were merged, so get rid of
        #   Note: If the column names are the same, this is obviously not an issue (hence the need to find
        #         cols_to_drop below, instead of simply dropping all of reset_df_2_dict['reset_merge_on']
        cols_to_drop = [x for x in reset_df_2_dict['reset_merge_on'] if x in merged_df.columns]
        merged_df = merged_df.drop(columns=cols_to_drop)
    
        # Rename columns from df_2 to original values (if original values were not None!)
        df_2_rename_dict = dict(zip(reset_df_2_dict['df_idx_names'], reset_df_2_dict['df_idx_names_OG']))
        df_2_rename_dict = {k:v for k,v in df_2_rename_dict.items() if k in merged_df.columns and v is not None}
        merged_df=merged_df.rename(columns=df_2_rename_dict)
        #-------------------------
        return merged_df


    #---------------------------------------------------------------------------
    # TODO still needs work...
    # This replaces OutageMdlrPrep.get_active_SNs_for_xfmrs_OLD (but, typically use OutageMdlrPrep.get_active_SNs_for_xfmrs_in_rcpo_df)
    @staticmethod
    def get_active_SNs_for_xfmrs(
        trsf_pole_nbs,     
        df_mp_curr, 
        df_mp_hist,
        time_infos_df,     
        time_infos_to_PNs = ['index'], 
        PNs_to_time_infos = ['index'], 
        how='left',     
        output_trsf_pole_nb_col=None, 
        addtnl_mp_df_curr_cols=None, 
        addtnl_mp_df_hist_cols=None, 
        return_SNs_col='SNs', 
        return_prem_nbs_col='prem_nbs', 
        assert_all_trsf_pole_nbs_found=True, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb', 
        t_min_col='t_min', 
        t_max_col='t_max'
    ):
        r"""
        Difficulty is that default.meter_premise_hist does not have trsf_pole_nb field.
        Therefore, one must use default.meter_premise to find the premise numbers for xfrms in trsf_pole_nbs,
          then use those PNs to select the correct entries from default.meter_premise_hist.
          
        time_infos_to_PNs:
          Defines how time_infos_df and PNs_for_xfmrs will be merged.
          NOTE: PNs_for_xfmrs will have indices equal to trsf_pole_nbs and values equal to lists of associated prem_nbs
          See OutageMdlrPrep.merge_rcpo_and_df and OutageMdlrPrep.reset_index_and_identify_cols_to_merge for more information
    
        If df_mp_curr OR df_mp_hist is not supplied, both will be built!
        
        addtnl_mp_df_curr_cols/addtnl_mp_df_hist_cols:
          Only used when df_mp_curr/df_mp_hist not supplied and therefore need to be built
          
        """
        #--------------------------------------------------
        assert(t_min_col in time_infos_df.columns and 
               t_max_col in time_infos_df.columns)
        time_infos_df = time_infos_df[[t_min_col, t_max_col]]
        #-----
        # Remove any duplicates from time_infos_df
        if time_infos_df.index.nlevels==1:
            tmp_col = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=[tmp_col])
        else:
            tmp_cols = [Utilities.generate_random_string() for _ in range(time_infos_df.index.nlevels)]
            for i_col, tmp_col in enumerate(tmp_cols):
                time_infos_df[tmp_col] = time_infos_df.index.get_level_values(i_col)
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=tmp_cols)
        #--------------------------------------------------
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        #-------------------------
        if df_mp_curr is None or df_mp_hist is None:
            mp_df_curr_hist = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
                trsf_pole_nbs, 
                join_curr_hist=False, 
                addtnl_mp_df_curr_cols=addtnl_mp_df_curr_cols, 
                addtnl_mp_df_hist_cols=addtnl_mp_df_hist_cols, 
                df_mp_serial_number_col=df_mp_serial_number_col, 
                df_mp_prem_nb_col=df_mp_prem_nb_col, 
                df_mp_install_time_col=df_mp_install_time_col, 
                df_mp_removal_time_col=df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
            )
            df_mp_curr = mp_df_curr_hist['mp_df_curr']
            df_mp_hist = mp_df_curr_hist['mp_df_hist']
        #-------------------------
        # At a bare minimum, df_mp_curr and df_mp_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in df_mp_curr.columns for x in necessary_mp_cols+[df_mp_trsf_pole_nb_col]]))
        assert(all([x in df_mp_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        # PNs_for_xfmrs is a DF with trsf_pole_nbs indices and elements which are lists of PNs for each xfmr
        PNs_for_xfmrs = MeterPremise.get_SNs_andor_PNs_for_xfmrs(
            trsf_pole_nbs=trsf_pole_nbs, 
            include_SNs=False,
            include_PNs=True,
            trsf_pole_nb_col=df_mp_trsf_pole_nb_col, 
            serial_number_col=df_mp_serial_number_col, 
            prem_nb_col=df_mp_prem_nb_col, 
            return_SNs_col=None, #Not grabbing SNs
            return_PNs_col=return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found=assert_all_trsf_pole_nbs_found, 
            mp_df=df_mp_curr, 
            return_mp_df_also=False
        )
        #-------------------------
        # Join together time_infos_df and PNs_for_xfmrs
        #-----
        time_infos_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df=time_infos_df, 
            df_2=PNs_for_xfmrs, 
            rcpo_df_on=time_infos_to_PNs,
            df_2_on=PNs_to_time_infos, 
            how=how
        )
        #--------------------------------------------------
        # Only reason for making dict is to ensure trsf_pole_nbs are not repeated 
        active_SNs_in_xfmrs_dfs_dict = {}
        if output_trsf_pole_nb_col is None:
            output_trsf_pole_nb_col='trsf_pole_nb'
        for trsf_pole_nb in trsf_pole_nbs:
            # active_SNs_df_i will have indices equal to premise numbers and value equal to lists
            #   of active SNs for each PN
            PNs_i=time_infos_df.loc[trsf_pole_nb, return_prem_nbs_col]
            dt_0_i=time_infos_df.loc[trsf_pole_nb, t_min_col]
            dt_1_i=time_infos_df.loc[trsf_pole_nb, t_max_col]
            #-----
            # See NOTEs above regarding t_min/t_max being empty
            # In such a case, it is simply impossibe (with the summary files currently generated) to access
            #   the date over which the data would have been run, if any events existed.
            #   In future versions, this information will be included in the summary files!
            # I don't want to completely exclude these (by e.g., setting dt_0_i=pd.Timestamp.min and 
            #   dt_1_i=pd.Timestamp.max), so I will simply include the meters which are active TODAY.
            # This obviously is not correct, but this occurrence is rare (only happening when every single meter
            #   on a transformer had no events during the time period) and this crude approximation will be fine.
            if Utilities.is_object_one_of_types(dt_0_i, [list, np.ndarray]):
                assert(len(dt_0_i)==0)
                # I believe if this happens for one it should happen for both...
                assert(Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]) and len(dt_1_i)==0)
                dt_0_i=pd.Timestamp.today()
            if Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]):
                assert(len(dt_1_i)==0)
                # I believe if this happens for one it should happen for both...
                # But, dt_0_i changed already above, so must check time_infos_df.loc[trsf_pole_nb, t_min_col] instead!
                assert(Utilities.is_object_one_of_types(time_infos_df.loc[trsf_pole_nb, t_min_col], [list, np.ndarray]) and 
                       len(time_infos_df.loc[trsf_pole_nb, t_min_col])==0)
                dt_1_i=pd.Timestamp.today()
            if((not isinstance(PNs_i, list) and pd.isna(PNs_i)) or 
               len(PNs_i)==0):
                active_SNs_df_i = pd.DataFrame()
            else:
                active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                    PNs=PNs_i,
                    df_mp_curr=df_mp_curr, 
                    df_mp_hist=df_mp_hist, 
                    dt_0=dt_0_i,
                    dt_1=dt_1_i,
                    output_index=None,
                    output_groupby=[df_mp_prem_nb_col], 
                    include_prems_wo_active_SNs_when_groupby=True, 
                    assert_all_PNs_found=False
                )
                active_SNs_df_i=active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col] = np.nan
                active_SNs_df_i[df_mp_serial_number_col] = [[]] 
                active_SNs_df_i[output_trsf_pole_nb_col] = trsf_pole_nb
                active_SNs_df_i = active_SNs_df_i.set_index(output_trsf_pole_nb_col)
            else:
                active_SNs_df_i[output_trsf_pole_nb_col] = trsf_pole_nb
                active_SNs_df_i = active_SNs_df_i.explode(df_mp_serial_number_col)
                active_SNs_df_i = Utilities_df.consolidate_df(
                    df=active_SNs_df_i, 
                    groupby_cols=[output_trsf_pole_nb_col], 
                    cols_shared_by_group=None, 
                    cols_to_collect_in_lists=[df_mp_serial_number_col, df_mp_prem_nb_col], 
                    include_groupby_cols_in_output_cols=False, 
                    allow_duplicates_in_lists=False, 
                    recover_uniqueness_violators=True, 
                    rename_cols=None, 
                    verbose=False
                )
            assert(trsf_pole_nb not in active_SNs_in_xfmrs_dfs_dict)
            active_SNs_in_xfmrs_dfs_dict[trsf_pole_nb] = active_SNs_df_i
        #-------------------------
        active_SNs_df = pd.concat(list(active_SNs_in_xfmrs_dfs_dict.values()))
        #-------------------------
        # Change [nan] entries to []
        #-----
        # First, if any entries equal NaN, change to []
        active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].isna(), df_mp_serial_number_col] = active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].isna(), df_mp_serial_number_col].apply(lambda x: [])
        # Now, change any entries equal to [] or [NaN] to []
        found_nans_srs = active_SNs_df[df_mp_serial_number_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        if found_nans_srs.sum()>0:
            active_SNs_df.loc[found_nans_srs, df_mp_serial_number_col] = active_SNs_df.loc[found_nans_srs, df_mp_serial_number_col].apply(lambda x: [])
        #-----
        # First, if any entries equal NaN, change to []
        active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].isna(), df_mp_prem_nb_col] = active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].isna(), df_mp_prem_nb_col].apply(lambda x: [])
        # Now, change any entries equal to [] or [NaN] to []
        found_nans_srs = active_SNs_df[df_mp_prem_nb_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        if found_nans_srs.sum()>0:
            active_SNs_df.loc[found_nans_srs, df_mp_prem_nb_col] = active_SNs_df.loc[found_nans_srs, df_mp_prem_nb_col].apply(lambda x: [])
        #-------------------------
        active_SNs_df = active_SNs_df.rename(columns={
            df_mp_prem_nb_col:return_prem_nbs_col, 
            df_mp_serial_number_col:return_SNs_col
        })
        #-------------------------
        return active_SNs_df


    #---------------------------------------------------------------------------
    # Replaces OutageMdlrPrep.get_active_SNs_for_xfmrs_OLD, but should probably build OutageMdlrPrep.get_active_SNs_for_xfmrs
    #  which accepts a list of trsf_pole_nbs instead of rcpo_df, which this function can use
    @staticmethod
    def get_active_SNs_for_xfmrs_in_rcpo_df(
        rcpo_df, 
        trsf_pole_nbs_loc, 
        df_mp_curr, 
        df_mp_hist,
        time_infos_df, 
        rcpo_df_to_time_infos_on = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on = ['index'], 
        how='left', 
        rcpo_df_to_PNs_on = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on = ['index'], 
        addtnl_mp_df_curr_cols=None, 
        addtnl_mp_df_hist_cols=None, 
        return_SNs_col='SNs', 
        return_prem_nbs_col='prem_nbs', 
        assert_all_trsf_pole_nbs_found=True, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb', 
        t_min_col='t_min', 
        t_max_col='t_max'
    ):
        r"""
        Difficulty is that default.meter_premise_hist does not have trsf_pole_nb field.
        Therefore, one must use default.meter_premise to find the premise numbers for xfrms in trsf_pole_nbs,
          then use those PNs to select the correct entries from default.meter_premise_hist.
        The trsf_pole_nbs should be contained in rcpo_df, and will be found using the trsf_pole_nbs_loc
          parameter described below.
          
        trsf_pole_nbs_loc:
            Directs where the transformer pole numbers are located
            This should identify an index (w/ level)
            Set equal to 'index' for normal DFs, or when trsf_pole_nbs are in level 0 of index.
            For a DF with MultiIndex index, there are two options:
                i.  Set equal to f'index_{idx_level}' for a DF with MutliIndex index, where idx_level
                    is an int identifying the level in which the trsf_pole_nbs reside
                ii. Set equal to the tuple ('index', trsf_pole_nbs_idx_name), where trsf_pole_nbs_idx_name is
                the name of the index level in which the trsf_pole_nbs reside.
    
        If df_mp_curr OR df_mp_hist is not supplied, both will be built!
        
        addtnl_mp_df_curr_cols/addtnl_mp_df_hist_cols:
          Only used when df_mp_curr/df_mp_hist not supplied and therefore need to be built
          
        """
        #--------------------------------------------------
        assert(t_min_col in time_infos_df.columns and 
               t_max_col in time_infos_df.columns)
        time_infos_df = time_infos_df[[t_min_col, t_max_col]]
        #-----
        # Remove any duplicates from time_infos_df
        if time_infos_df.index.nlevels==1:
            tmp_col = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=[tmp_col])
        else:
            tmp_cols = [Utilities.generate_random_string() for _ in range(time_infos_df.index.nlevels)]
            for i_col, tmp_col in enumerate(tmp_cols):
                time_infos_df[tmp_col] = time_infos_df.index.get_level_values(i_col)
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=tmp_cols)
        #--------------------------------------------------
        # trsf_pole_nbs_loc can be a string or tuple/list
        # First, find trsf_pole_nbs and trsf_pole_nbs_idx_lvl
        assert(Utilities.is_object_one_of_types(trsf_pole_nbs_loc, [str, list, tuple]))
        if isinstance(trsf_pole_nbs_loc, str):
            assert(trsf_pole_nbs_loc.startswith('index'))
            if trsf_pole_nbs_loc=='index':
                trsf_pole_nbs_idx_lvl = 0
            else:
                trsf_pole_nbs_idx_lvl = re.findall('index_(\d*)', trsf_pole_nbs_loc)
                assert(len(trsf_pole_nbs_idx_lvl)==1)
                trsf_pole_nbs_idx_lvl=trsf_pole_nbs_idx_lvl[0]
                trsf_pole_nbs_idx_lvl=int(trsf_pole_nbs_idx_lvl)
        else:
            assert(len(trsf_pole_nbs_loc)==2)
            assert(trsf_pole_nbs_loc[0]=='index')
            assert(trsf_pole_nbs_loc[1] in rcpo_df.index.names)
            trsf_pole_nbs_idx_lvl = rcpo_df.index.names.index(trsf_pole_nbs_loc[1])
            #---------------
            assert(trsf_pole_nbs_idx_lvl < rcpo_df.index.nlevels)
            trsf_pole_nbs = rcpo_df.index.get_level_values(trsf_pole_nbs_idx_lvl).tolist()
        #--------------------------------------------------
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        #-------------------------
        if df_mp_curr is None or df_mp_hist is None:
            mp_df_curr_hist = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
                trsf_pole_nbs, 
                join_curr_hist=False, 
                addtnl_mp_df_curr_cols=addtnl_mp_df_curr_cols, 
                addtnl_mp_df_hist_cols=addtnl_mp_df_hist_cols, 
                df_mp_serial_number_col=df_mp_serial_number_col, 
                df_mp_prem_nb_col=df_mp_prem_nb_col, 
                df_mp_install_time_col=df_mp_install_time_col, 
                df_mp_removal_time_col=df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
            )
            df_mp_curr = mp_df_curr_hist['mp_df_curr']
            df_mp_hist = mp_df_curr_hist['mp_df_hist']
        #-------------------------
        # At a bare minimum, df_mp_curr and df_mp_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in df_mp_curr.columns for x in necessary_mp_cols+[df_mp_trsf_pole_nb_col]]))
        assert(all([x in df_mp_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        # PNs_for_xfmrs is a DF with trsf_pole_nbs indices and elements which are lists of PNs for each xfmr
        PNs_for_xfmrs = MeterPremise.get_SNs_andor_PNs_for_xfmrs(
            trsf_pole_nbs=trsf_pole_nbs, 
            include_SNs=False,
            include_PNs=True,
            trsf_pole_nb_col=df_mp_trsf_pole_nb_col, 
            serial_number_col=df_mp_serial_number_col, 
            prem_nb_col=df_mp_prem_nb_col, 
            return_SNs_col=None, #Not grabbing SNs
            return_PNs_col=return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found=assert_all_trsf_pole_nbs_found, 
            mp_df=df_mp_curr, 
            return_mp_df_also=False
        )
        #-------------------------
        # Join together rcpo_df, time_infos_df and PNs_for_xfmrs
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df=rcpo_df, 
            df_2=time_infos_df, 
            rcpo_df_on=rcpo_df_to_time_infos_on,
            df_2_on=time_infos_to_rcpo_df_on, 
            how=how
        )
        #-----
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df=rcpo_df, 
            df_2=PNs_for_xfmrs, 
            rcpo_df_on=rcpo_df_to_PNs_on,
            df_2_on=PNs_to_rcpo_df_on, 
            how=how
        )
        #--------------------------------------------------
        # Only reason for making dict is to ensure trsf_pole_nbs are not repeated 
        active_SNs_in_xfmrs_dfs_dict = {}
    
        rcpo_idx_names = list(rcpo_df.index.names)
        assert(not any([x is None for x in rcpo_idx_names]))
        for idx_i, row_i in rcpo_df.iterrows():
            # active_SNs_df_i will have indices equal to premise numbers and value equal to lists
            #   of active SNs for each PN
            # Purpose of making idx_names_w_vals a list of tuples, instead of a dict, is to ensure the correct order is maintained
            #   Dicts usually return the correct order, but this is not guaranteed
            if len(rcpo_idx_names)==1:
                assert(rcpo_df.index.nlevels==1)
                idx_names_w_vals = [(rcpo_idx_names[0], idx_i)]
            else:
                idx_names_w_vals = [((rcpo_idx_names[i] if i!=trsf_pole_nbs_idx_lvl else df_mp_trsf_pole_nb_col), idx_i[i]) 
                                    for i in range(len(idx_i))]
            PNs_i=row_i[return_prem_nbs_col]
            dt_0_i=row_i[t_min_col]
            dt_1_i=row_i[t_max_col]
            #-----
            # See NOTEs above regarding t_min/t_max being empty
            # In such a case, it is simply impossibe (with the summary files currently generated) to access
            #   the date over which the data would have been run, if any events existed.
            #   In future versions, this information will be included in the summary files!
            # I don't want to completely exclude these (by e.g., setting dt_0_i=pd.Timestamp.min and 
            #   dt_1_i=pd.Timestamp.max), so I will simply include the meters which are active TODAY.
            # This obviously is not correct, but this occurrence is rare (only happening when every single meter
            #   on a transformer had no events during the time period) and this crude approximation will be fine.
            if Utilities.is_object_one_of_types(dt_0_i, [list, np.ndarray]):
                assert(len(dt_0_i)==0)
                # I believe if this happens for one it should happen for both...
                assert(Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]) and len(dt_1_i)==0)
                dt_0_i=pd.Timestamp.today()
            if Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]):
                assert(len(dt_1_i)==0)
                # I believe if this happens for one it should happen for both...
                # But, dt_0_i changed already above, so must check row_i[t_min_col] instead!
                assert(Utilities.is_object_one_of_types(row_i[t_min_col], [list, np.ndarray]) and len(row_i[t_min_col])==0)
                dt_1_i=pd.Timestamp.today()
            if((not isinstance(PNs_i, list) and pd.isna(PNs_i)) or 
               len(PNs_i)==0):
                active_SNs_df_i = pd.DataFrame()
            else:
                active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                    PNs=PNs_i,
                    df_mp_curr=df_mp_curr, 
                    df_mp_hist=df_mp_hist, 
                    dt_0=dt_0_i,
                    dt_1=dt_1_i,
                    output_index=None,
                    output_groupby=[df_mp_prem_nb_col], 
                    include_prems_wo_active_SNs_when_groupby=True, 
                    assert_all_PNs_found=False
                )
                active_SNs_df_i=active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col] = np.nan
                active_SNs_df_i[df_mp_serial_number_col] = [[]] 
                for name,val in idx_names_w_vals:
                    active_SNs_df_i[name] = val
                active_SNs_df_i = active_SNs_df_i.set_index([x[0] for x in idx_names_w_vals])
            else:
                for name,val in idx_names_w_vals:
                    active_SNs_df_i[name] = val
                active_SNs_df_i = active_SNs_df_i.explode(df_mp_serial_number_col)
                active_SNs_df_i = Utilities_df.consolidate_df(
                    df=active_SNs_df_i, 
                    groupby_cols=[x[0] for x in idx_names_w_vals], 
                    cols_shared_by_group=None, 
                    cols_to_collect_in_lists=[df_mp_serial_number_col, df_mp_prem_nb_col], 
                    include_groupby_cols_in_output_cols=False, 
                    allow_duplicates_in_lists=False, 
                    recover_uniqueness_violators=True, 
                    rename_cols=None, 
                    verbose=False
                )
            assert(idx_i not in active_SNs_in_xfmrs_dfs_dict)
            active_SNs_in_xfmrs_dfs_dict[idx_i] = active_SNs_df_i
        #-------------------------
        active_SNs_df = pd.concat(list(active_SNs_in_xfmrs_dfs_dict.values()))
        #-------------------------
        # Change [nan] entries to []
        #-----
        # First, if any entries equal NaN, change to []
        active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].isna(), df_mp_serial_number_col] = active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].isna(), df_mp_serial_number_col].apply(lambda x: [])
        # Now, change any entries equal to [] or [NaN] to []
        found_nans_srs = active_SNs_df[df_mp_serial_number_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        if found_nans_srs.sum()>0:
            active_SNs_df.loc[found_nans_srs, df_mp_serial_number_col] = active_SNs_df.loc[found_nans_srs, df_mp_serial_number_col].apply(lambda x: [])
        #-----
        # First, if any entries equal NaN, change to []
        active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].isna(), df_mp_prem_nb_col] = active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].isna(), df_mp_prem_nb_col].apply(lambda x: [])
        # Now, change any entries equal to [] or [NaN] to []
        found_nans_srs = active_SNs_df[df_mp_prem_nb_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        if found_nans_srs.sum()>0:
            active_SNs_df.loc[found_nans_srs, df_mp_prem_nb_col] = active_SNs_df.loc[found_nans_srs, df_mp_prem_nb_col].apply(lambda x: [])
        #-------------------------
        active_SNs_df = active_SNs_df.rename(columns={
            df_mp_prem_nb_col:return_prem_nbs_col, 
            df_mp_serial_number_col:return_SNs_col
        })
        #-------------------------
        return active_SNs_df

    
    #---------------------------------------------------------------------------
    # Replaces OutageMdlrPrep.get_active_SNs_for_xfmrs_OLD, but should probably build OutageMdlrPrep.get_active_SNs_for_xfmrs
    #  which accepts a list of trsf_pole_nbs instead of rcpo_df, which this function can use
    @staticmethod
    def get_active_SNs_for_xfmrs_in_rcpo_df_v2(
        rcpo_df, 
        trsf_pole_nbs_loc, 
        df_mp_curr, 
        df_mp_hist,
        time_infos_df, 
        rcpo_df_to_time_infos_on = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on = ['index'], 
        how='left', 
        rcpo_df_to_PNs_on = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on = ['index'], 
        addtnl_mp_df_curr_cols=None, 
        addtnl_mp_df_hist_cols=None, 
        return_SNs_col='SNs', 
        return_prem_nbs_col='prem_nbs', 
        assert_all_trsf_pole_nbs_found=True, 
        df_mp_serial_number_col='mfr_devc_ser_nbr', 
        df_mp_prem_nb_col='prem_nb', 
        df_mp_install_time_col='inst_ts', 
        df_mp_removal_time_col='rmvl_ts', 
        df_mp_trsf_pole_nb_col='trsf_pole_nb', 
        t_min_col='t_min', 
        t_max_col='t_max'
    ):
        r"""
        Difficulty is that default.meter_premise_hist does not have trsf_pole_nb field.
        Therefore, one must use default.meter_premise to find the premise numbers for xfrms in trsf_pole_nbs,
          then use those PNs to select the correct entries from default.meter_premise_hist.
        The trsf_pole_nbs should be contained in rcpo_df, and will be found using the trsf_pole_nbs_loc
          parameter described below.
          
        trsf_pole_nbs_loc:
            Directs where the transformer pole numbers are located
            This should identify an index (w/ level)
            Set equal to 'index' for normal DFs, or when trsf_pole_nbs are in level 0 of index.
            For a DF with MultiIndex index, there are two options:
                i.  Set equal to f'index_{idx_level}' for a DF with MutliIndex index, where idx_level
                    is an int identifying the level in which the trsf_pole_nbs reside
                ii. Set equal to the tuple ('index', trsf_pole_nbs_idx_name), where trsf_pole_nbs_idx_name is
                the name of the index level in which the trsf_pole_nbs reside.
    
        If df_mp_curr OR df_mp_hist is not supplied, both will be built!
        
        addtnl_mp_df_curr_cols/addtnl_mp_df_hist_cols:
          Only used when df_mp_curr/df_mp_hist not supplied and therefore need to be built
          
        """
        #--------------------------------------------------
        assert(t_min_col in time_infos_df.columns and 
               t_max_col in time_infos_df.columns)
        time_infos_df = time_infos_df[[t_min_col, t_max_col]]
        #-----
        # Remove any duplicates from time_infos_df
        if time_infos_df.index.nlevels==1:
            tmp_col = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=[tmp_col])
        else:
            tmp_cols = [Utilities.generate_random_string() for _ in range(time_infos_df.index.nlevels)]
            for i_col, tmp_col in enumerate(tmp_cols):
                time_infos_df[tmp_col] = time_infos_df.index.get_level_values(i_col)
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=tmp_cols)
        #--------------------------------------------------
        # trsf_pole_nbs_loc can be a string or tuple/list
        # First, find trsf_pole_nbs and trsf_pole_nbs_idx_lvl
        assert(Utilities.is_object_one_of_types(trsf_pole_nbs_loc, [str, list, tuple]))
        if isinstance(trsf_pole_nbs_loc, str):
            assert(trsf_pole_nbs_loc.startswith('index'))
            if trsf_pole_nbs_loc=='index':
                trsf_pole_nbs_idx_lvl = 0
            else:
                trsf_pole_nbs_idx_lvl = re.findall('index_(\d*)', trsf_pole_nbs_loc)
                assert(len(trsf_pole_nbs_idx_lvl)==1)
                trsf_pole_nbs_idx_lvl=trsf_pole_nbs_idx_lvl[0]
                trsf_pole_nbs_idx_lvl=int(trsf_pole_nbs_idx_lvl)
        else:
            assert(len(trsf_pole_nbs_loc)==2)
            assert(trsf_pole_nbs_loc[0]=='index')
            assert(trsf_pole_nbs_loc[1] in rcpo_df.index.names)
            trsf_pole_nbs_idx_lvl = rcpo_df.index.names.index(trsf_pole_nbs_loc[1])
            #---------------
            assert(trsf_pole_nbs_idx_lvl < rcpo_df.index.nlevels)
            trsf_pole_nbs = rcpo_df.index.get_level_values(trsf_pole_nbs_idx_lvl).tolist()
        #--------------------------------------------------
        #-------------------------
        necessary_mp_cols = [df_mp_serial_number_col, df_mp_prem_nb_col, df_mp_install_time_col, df_mp_removal_time_col]
        #-------------------------
        if df_mp_curr is None or df_mp_hist is None:
            mp_df_curr_hist = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
                trsf_pole_nbs, 
                join_curr_hist=False, 
                addtnl_mp_df_curr_cols=addtnl_mp_df_curr_cols, 
                addtnl_mp_df_hist_cols=addtnl_mp_df_hist_cols, 
                df_mp_serial_number_col=df_mp_serial_number_col, 
                df_mp_prem_nb_col=df_mp_prem_nb_col, 
                df_mp_install_time_col=df_mp_install_time_col, 
                df_mp_removal_time_col=df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col=df_mp_trsf_pole_nb_col
            )
            df_mp_curr = mp_df_curr_hist['mp_df_curr']
            df_mp_hist = mp_df_curr_hist['mp_df_hist']
        #-------------------------
        # At a bare minimum, df_mp_curr and df_mp_hist must both have the following columns:
        #   necessary_mp_cols = ['mfr_devc_ser_nbr', 'prem_nb', 'inst_ts', 'rmvl_ts']
        assert(all([x in df_mp_curr.columns for x in necessary_mp_cols+[df_mp_trsf_pole_nb_col]]))
        assert(all([x in df_mp_hist.columns for x in necessary_mp_cols]))
        #-------------------------
        # PNs_for_xfmrs is a DF with trsf_pole_nbs indices and elements which are lists of PNs for each xfmr
        PNs_for_xfmrs = MeterPremise.get_SNs_andor_PNs_for_xfmrs(
            trsf_pole_nbs=trsf_pole_nbs, 
            include_SNs=False,
            include_PNs=True,
            trsf_pole_nb_col=df_mp_trsf_pole_nb_col, 
            serial_number_col=df_mp_serial_number_col, 
            prem_nb_col=df_mp_prem_nb_col, 
            return_SNs_col=None, #Not grabbing SNs
            return_PNs_col=return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found=assert_all_trsf_pole_nbs_found, 
            mp_df=df_mp_curr, 
            return_mp_df_also=False
        )
        #-------------------------
        # Join together rcpo_df, time_infos_df and PNs_for_xfmrs
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df=rcpo_df, 
            df_2=time_infos_df, 
            rcpo_df_on=rcpo_df_to_time_infos_on,
            df_2_on=time_infos_to_rcpo_df_on, 
            how=how
        )
        #-----
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df=rcpo_df, 
            df_2=PNs_for_xfmrs, 
            rcpo_df_on=rcpo_df_to_PNs_on,
            df_2_on=PNs_to_rcpo_df_on, 
            how=how
        )
        #--------------------------------------------------
        # Only reason for making dict is to ensure trsf_pole_nbs are not repeated 
        active_SNs_in_xfmrs_dfs_dict = {}
    
        rcpo_idx_names = list(rcpo_df.index.names)
        assert(not any([x is None for x in rcpo_idx_names]))
        for idx_i, row_i in rcpo_df.iterrows():
            # active_SNs_df_i will have indices equal to premise numbers and value equal to lists
            #   of active SNs for each PN
            # Purpose of making idx_names_w_vals a list of tuples, instead of a dict, is to ensure the correct order is maintained
            #   Dicts usually return the correct order, but this is not guaranteed
            if len(rcpo_idx_names)==1:
                assert(rcpo_df.index.nlevels==1)
                idx_names_w_vals = [(rcpo_idx_names[0], idx_i)]
            else:
                idx_names_w_vals = [((rcpo_idx_names[i] if i!=trsf_pole_nbs_idx_lvl else df_mp_trsf_pole_nb_col), idx_i[i]) 
                                    for i in range(len(idx_i))]
            PNs_i=row_i[return_prem_nbs_col]
            dt_0_i=row_i[t_min_col]
            dt_1_i=row_i[t_max_col]
            #-----
            # See NOTEs above regarding t_min/t_max being empty
            # In such a case, it is simply impossibe (with the summary files currently generated) to access
            #   the date over which the data would have been run, if any events existed.
            #   In future versions, this information will be included in the summary files!
            # I don't want to completely exclude these (by e.g., setting dt_0_i=pd.Timestamp.min and 
            #   dt_1_i=pd.Timestamp.max), so I will simply include the meters which are active TODAY.
            # This obviously is not correct, but this occurrence is rare (only happening when every single meter
            #   on a transformer had no events during the time period) and this crude approximation will be fine.
            if Utilities.is_object_one_of_types(dt_0_i, [list, np.ndarray]):
                assert(len(dt_0_i)==0)
                # I believe if this happens for one it should happen for both...
                assert(Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]) and len(dt_1_i)==0)
                dt_0_i=pd.Timestamp.today()
            if Utilities.is_object_one_of_types(dt_1_i, [list, np.ndarray]):
                assert(len(dt_1_i)==0)
                # I believe if this happens for one it should happen for both...
                # But, dt_0_i changed already above, so must check row_i[t_min_col] instead!
                assert(Utilities.is_object_one_of_types(row_i[t_min_col], [list, np.ndarray]) and len(row_i[t_min_col])==0)
                dt_1_i=pd.Timestamp.today()
            if((not isinstance(PNs_i, list) and pd.isna(PNs_i)) or 
               len(PNs_i)==0):
                active_SNs_df_i = pd.DataFrame()
            else:
                active_SNs_df_i = MeterPremise.get_active_SNs_for_PNs_at_datetime_interval(
                    PNs=PNs_i,
                    df_mp_curr=df_mp_curr, 
                    df_mp_hist=df_mp_hist, 
                    dt_0=dt_0_i,
                    dt_1=dt_1_i,
                    output_index=None,
                    output_groupby=[df_mp_prem_nb_col], 
                    include_prems_wo_active_SNs_when_groupby=True, 
                    assert_all_PNs_found=False
                )
                active_SNs_df_i=active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col] = np.nan
                active_SNs_df_i[df_mp_serial_number_col] = [[]] 
            for name,val in idx_names_w_vals:
                active_SNs_df_i[name] = val
            active_SNs_df_i = active_SNs_df_i.explode(df_mp_serial_number_col)
            assert(idx_i not in active_SNs_in_xfmrs_dfs_dict)
            active_SNs_in_xfmrs_dfs_dict[idx_i] = active_SNs_df_i
        #-------------------------
        active_SNs_df = pd.concat(list(active_SNs_in_xfmrs_dfs_dict.values()))
        #-------------------------
        active_SNs_df = Utilities_df.consolidate_df(
            df=active_SNs_df, 
            groupby_cols=[x[0] for x in idx_names_w_vals], 
            cols_shared_by_group=None, 
            cols_to_collect_in_lists=[df_mp_serial_number_col, df_mp_prem_nb_col], 
            include_groupby_cols_in_output_cols=False, 
            allow_duplicates_in_lists=False, 
            recover_uniqueness_violators=True, 
            rename_cols=None, 
            verbose=False
        )
        #-------------------------
        # Change [nan] entries to []
        #-----
        # First, if any entries equal NaN, change to []
        active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].isna(), df_mp_serial_number_col] = active_SNs_df.loc[active_SNs_df[df_mp_serial_number_col].isna(), df_mp_serial_number_col].apply(lambda x: [])
        # Now, change any entries equal to [] or [NaN] to []
        found_nans_srs = active_SNs_df[df_mp_serial_number_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        if found_nans_srs.sum()>0:
            active_SNs_df.loc[found_nans_srs, df_mp_serial_number_col] = active_SNs_df.loc[found_nans_srs, df_mp_serial_number_col].apply(lambda x: [])
        #-----
        # First, if any entries equal NaN, change to []
        active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].isna(), df_mp_prem_nb_col] = active_SNs_df.loc[active_SNs_df[df_mp_prem_nb_col].isna(), df_mp_prem_nb_col].apply(lambda x: [])
        # Now, change any entries equal to [] or [NaN] to []
        found_nans_srs = active_SNs_df[df_mp_prem_nb_col].apply(lambda x: len([ix for ix in x if not pd.isna(ix)]))==0
        if found_nans_srs.sum()>0:
            active_SNs_df.loc[found_nans_srs, df_mp_prem_nb_col] = active_SNs_df.loc[found_nans_srs, df_mp_prem_nb_col].apply(lambda x: [])
        #------------------------- 
        active_SNs_df = active_SNs_df.rename(columns={
            df_mp_prem_nb_col:return_prem_nbs_col, 
            df_mp_serial_number_col:return_SNs_col
        })
        #-------------------------
        return active_SNs_df


    #---------------------------------------------------------------------------
    # Replaces OutageMdlrPrep.add_xfmr_active_SNs_to_rcpo_df_OLD
    @staticmethod
    def add_xfmr_active_SNs_to_rcpo_df(
        rcpo_df, 
        trsf_pole_nbs_loc, 
        set_xfmr_nSNs=True, 
        include_active_xfmr_PNs=False, 
        df_mp_curr=None,
        df_mp_hist=None, 
        time_infos_df=None, 
        rcpo_df_to_time_infos_on = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on = ['index'], 
        how='left', 
        rcpo_df_to_PNs_on = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on = ['index'], 
        addtnl_get_active_SNs_for_xfmrs_kwargs=None, 
        xfmr_SNs_col='_xfmr_SNs', 
        xfmr_nSNs_col='_xfmr_nSNs', 
        xfmr_PNs_col='_xfmr_PNs', 
        xfmr_nPNs_col='_xfmr_nPNs', 
    ):
        r"""
        NOTE: If include_active_xfmr_PNs is True, this column (named xfmr_PNs_col='_xfmr_SNs') should be 
              equal to the PNs already in rcpo_df!
        NOTE: xfmr_SNs_col, xfmr_nSNs_col, xfmr_PNs_col, and xfmr_nPNs_col should all be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
              
        NOTE: If any of xfmr_SNs_col, xfmr_nSNs_col, xfmr_PNs_col, and xfmr_nPNs_col are already contained in 
              rcpo_df, they will be replaced.  This is needed so that the merge operation does not come back with _x and _y
              values.  So, one should make sure this function call is truly needed, as grabbing the serial numbers for the
              outages typically takes a couple/few minutes.
              
        NOTE: To make things run faster, the user can supply df_mp_curr and df_mp_hist.  These will be included in 
              get_active_SNs_for_xfmrs_kwargs.
              NOTE: If df_mp_curr/df_mp_hist is also supplied in addtnl_get_active_SNs_for_xfmrs_kwargs,
                    that/those in addtnl_get_active_SNs_for_xfmrs_kwargs will ultimately be used (not the
                    explicity df_mp_hist/curr in the function arguments!)
              CAREFUL: If one does supple df_mp_curr/hist, one must be certain these DFs contain all necessary elements!
        """
        #-------------------------
        get_active_SNs_for_xfmrs_kwargs = dict(
            rcpo_df=rcpo_df, 
            trsf_pole_nbs_loc=trsf_pole_nbs_loc, 
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist, 
            time_infos_df=time_infos_df, 
            rcpo_df_to_time_infos_on=rcpo_df_to_time_infos_on, 
            time_infos_to_rcpo_df_on=time_infos_to_rcpo_df_on, 
            how=how, 
            rcpo_df_to_PNs_on=rcpo_df_to_PNs_on, 
            PNs_to_rcpo_df_on=PNs_to_rcpo_df_on, 
            return_prem_nbs_col=xfmr_PNs_col, 
            return_SNs_col=xfmr_SNs_col
        )
        if addtnl_get_active_SNs_for_xfmrs_kwargs is not None:
            get_active_SNs_for_xfmrs_kwargs = {**get_active_SNs_for_xfmrs_kwargs, 
                                               **addtnl_get_active_SNs_for_xfmrs_kwargs}
        active_SNs_df = OutageMdlrPrep.get_active_SNs_for_xfmrs_in_rcpo_df(**get_active_SNs_for_xfmrs_kwargs)
        assert(isinstance(active_SNs_df, pd.DataFrame))
        #-------------------------
        # Assert below might be too strong here...
        assert(sorted(rcpo_df.index.unique().tolist())==sorted(active_SNs_df.index.unique().tolist()))
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            #----------
            # See note above about columns being replaced/dropped
            cols_to_drop = [x for x in rcpo_df.columns if x in active_SNs_df.columns]
            if len(cols_to_drop)>0:
                rcpo_df = rcpo_df.drop(columns=cols_to_drop)
            #----------
            rcpo_df = rcpo_df.merge(active_SNs_df, left_index=True, right_index=True)
            #----------
            if set_xfmr_nSNs:
                rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, xfmr_SNs_col, xfmr_nSNs_col)
                if include_active_xfmr_PNs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, xfmr_PNs_col, xfmr_nPNs_col)
        else:
            # Currently, only expecting raw and/or norm.  No problem to allow more, but for now keep this to alert 
            # of anything unexpected
            assert(rcpo_df.columns.get_level_values(0).nunique()<=2)
            for i,level_0_val in enumerate(rcpo_df.columns.get_level_values(0).unique()):
                if i==0:
                    active_SNs_df.columns = pd.MultiIndex.from_product([[level_0_val], active_SNs_df.columns])
                else:
                    active_SNs_df.columns = active_SNs_df.columns.set_levels([level_0_val], level=0)
                #----------
                # See note above about columns being replaced/dropped
                cols_to_drop = [x for x in rcpo_df.columns if x in active_SNs_df.columns]
                if len(cols_to_drop)>0:
                    rcpo_df = rcpo_df.drop(columns=cols_to_drop)
                #----------
                rcpo_df = rcpo_df.merge(active_SNs_df, left_index=True, right_index=True)
                #----------
                if set_xfmr_nSNs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, xfmr_SNs_col), (level_0_val, xfmr_nSNs_col))
                    if include_active_xfmr_PNs:
                        rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, xfmr_PNs_col), (level_0_val, xfmr_nPNs_col))
        #-------------------------
        rcpo_df = rcpo_df.sort_index(axis=1,level=0)
        #-------------------------
        return rcpo_df


    #---------------------------------------------------------------------------
    # Replaces OutageMdlrPrep.build_rcpo_df_norm_by_xfmr_active_nSNs_OLD
    @staticmethod
    def build_rcpo_df_norm_by_xfmr_active_nSNs(
        rcpo_df_raw, 
        trsf_pole_nbs_loc, 
        xfmr_nSNs_col='_xfmr_nSNs', 
        xfmr_SNs_col='_xfmr_SNs', 
        other_SNs_col_tags_to_ignore=['_SNs', '_nSNs', '_prem_nbs', '_nprem_nbs', '_xfmr_PNs', '_xfmr_nPNs'], 
        drop_xfmr_nSNs_eq_0=True, 
        new_level_0_val='counts_norm_by_xfmr_nSNs', 
        remove_SNs_cols=False, 
        df_mp_curr=None,
        df_mp_hist=None, 
        time_infos_df=None,
        rcpo_df_to_time_infos_on = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on = ['index'], 
        how='left', 
        rcpo_df_to_PNs_on = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on = ['index'], 
        addtnl_get_active_SNs_for_xfmrs_kwargs=None
    ):
        r"""
        Build rcpo_df normalized by the number of serial numbers in each outage
    
        drop_xfmr_nSNs_eq_0:
          It is possible for the number of serial numbers in an outage to be zero!
          Premise numbers are always found, but the meter_premise database does not always 
            contain the premise numbers.
          Dividing by zero will make all counts for such an entry equal to NaN or inf.
          When drop_xfmr_nSNs_eq_0 is True, such entries will be removed.
    
        NOTE: xfmr_SNs_col and xfmr_nSNs_col should both be strings, not tuples.
              If column is MultiIndex, the level_0 value will be handled below.
        """
        #-------------------------
        n_counts_col = xfmr_nSNs_col
        list_col = xfmr_SNs_col
        #-------------------------
        # NOTE: MECPODf.add_outage_SNs_to_rcpo_df expects xfmr_SNs_col and xfmr_nSNs_col to be strings, not tuples
        #       as it handles the level 0 values if they exist.  So, if tuples, use only highest level values (i.e., level 1)
        assert(Utilities.is_object_one_of_types(list_col, [str, tuple]))
        assert(Utilities.is_object_one_of_types(n_counts_col, [str, tuple]))
        #-----
        add_list_col_to_rcpo_df_func = OutageMdlrPrep.add_xfmr_active_SNs_to_rcpo_df
        add_list_col_to_rcpo_df_kwargs = dict(
            trsf_pole_nbs_loc=trsf_pole_nbs_loc, 
            set_xfmr_nSNs=True, 
            include_active_xfmr_PNs=True, 
            df_mp_curr=df_mp_curr,
            df_mp_hist=df_mp_hist, 
            time_infos_df=time_infos_df, 
            rcpo_df_to_time_infos_on=rcpo_df_to_time_infos_on, 
            time_infos_to_rcpo_df_on=time_infos_to_rcpo_df_on, 
            how=how, 
            rcpo_df_to_PNs_on=rcpo_df_to_PNs_on, 
            PNs_to_rcpo_df_on=PNs_to_rcpo_df_on, 
            addtnl_get_active_SNs_for_xfmrs_kwargs=addtnl_get_active_SNs_for_xfmrs_kwargs, 
            xfmr_SNs_col='_xfmr_SNs', 
            xfmr_nSNs_col='_xfmr_nSNs', 
            xfmr_PNs_col='_xfmr_PNs', 
            xfmr_nPNs_col='_xfmr_nPNs', 
        )
        #-------------------------
        other_col_tags_to_ignore = other_SNs_col_tags_to_ignore
        drop_n_counts_eq_0 = drop_xfmr_nSNs_eq_0
        new_level_0_val = new_level_0_val
        remove_ignored_cols = remove_SNs_cols
        #-------------------------
        return MECPODf.build_rcpo_df_norm_by_list_counts(
            rcpo_df_raw=rcpo_df_raw, 
            n_counts_col=n_counts_col, 
            list_col=list_col, 
            add_list_col_to_rcpo_df_func=add_list_col_to_rcpo_df_func, 
            add_list_col_to_rcpo_df_kwargs=add_list_col_to_rcpo_df_kwargs, 
            other_col_tags_to_ignore=other_col_tags_to_ignore, 
            drop_n_counts_eq_0=drop_n_counts_eq_0, 
            new_level_0_val=new_level_0_val, 
            remove_ignored_cols=remove_ignored_cols
        )


    #---------------------------------------------------------------------------
    @staticmethod
    def get_outg_time_infos_df(
        rcpo_df, 
        outg_rec_nb_idx_lvl, 
        times_relative_to_off_ts_only=True, 
        td_for_left=None, 
        td_for_right=None
    ):
        r"""
        Given a rcpo_df, with outg_rec_nbs stored in the index (located by outg_rec_nb_idx_lvl), construct the
        time_infos_df whose indices are outg_rec_nbs and columns, ['t_min', 't_max'], give the search time windows
        for each outage.
        
        This was designed, and is generally used, for finding the active meters for each outage.
        
        td_for_left / td_for_right: 
            datetime.timedelta objects used to define the left and right edges of the time window.
            These can be positive or negative.
            If input value is None, set equal to 0 (datetime.timedelta(days=0)), meaning the returned t_min values
              will be the time the power went out, and the returned t_max values will either be the time the power went out
              (times_relative_to_off_ts_only==True) or the time power was restored (times_relative_to_off_ts_only==False), 
              depending on the value of times_relative_to_off_ts_only.
            The return t_min column is calculated by:
                adding td_for_left to the time the outage began, 
                    time_infos_df['t_min'] = pd.to_datetime(time_infos_df['DT_OFF_TS_FULL'])+td_for_left
                Thus, for t_min to be before the outage, td_for_left must be negative.
            The return t_max column is calculated by:
                If times_relative_to_off_ts_only==True:  adding td_for_right to the time the outage began (DT_OFF_TS_FULL)
                If times_relative_to_off_ts_only==False: adding td_for_right to the time the outage ended (DT_ON_TS)
            RESTRICTIONS:    
                When times_relative_to_off_ts_only==True, td_for_left must be less than td_for_right, so that t_min is less
                  than t_max.
                However, when times_relative_to_off_ts_only==False, this is not strictly true (although, in most cases, should be true)
                At the end of the day, all that really matters is that the application of td_for_left and td_for_right to
                  the off_ts/on_ts columns results in time_infos_df['t_min'] being less than time_infos_df['t_max']
                ----------
                When can td_for_left be greater than td_for_right?
                -----
                    Take, for example, an outage that lasts an entire day.  
                      Assume the outage begins off_ts=2022-10-05 12:00:00 and ends on_ts=2022-10-06 12:00:00.
                      Further assume td_for_left = datetime.timedelta(hours=1) and td_for_right = datetime.timedelta(hours=-1), so
                        clearly td_for_left > td_for_right.
                      ==> left  = off_ts+td_for_left = 2022-10-05 13:00:00
                          right = on_ts+td_for_right = 2022-10-06 11:00:00
                      Thus, left<right, and the window makes sense!
        """
        #-------------------------
        if td_for_left is None:
            td_for_left = datetime.timedelta(days=0)
        if td_for_right is None:
            td_for_right = datetime.timedelta(days=0)
        assert(isinstance(td_for_left, datetime.timedelta))
        assert(isinstance(td_for_right, datetime.timedelta))
        #-------------------------
        dovs_outgs = DOVSOutages(                 
            df_construct_type=DFConstructType.kRunSqlQuery, 
            contstruct_df_args=None, 
            init_df_in_constructor=True, 
            build_sql_function=DOVSOutages_SQL.build_sql_outage, 
            build_sql_function_kwargs=dict(
                outg_rec_nbs=rcpo_df.index.get_level_values(outg_rec_nb_idx_lvl).tolist(), 
                from_table_alias='DOV', 
                datetime_col='DT_OFF_TS_FULL', 
                cols_of_interest=[
                    'OUTG_REC_NB', 
                    dict(field_desc=f"DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)", 
                         alias='DT_OFF_TS_FULL', table_alias_prefix=None), 
                    'DT_ON_TS'
                ], 
                field_to_split='outg_rec_nbs'
            )
        )
        #-------------------------
        time_infos_df = dovs_outgs.df
        time_infos_df = Utilities_df.convert_col_type(df=time_infos_df, column='OUTG_REC_NB', to_type=str)
        time_infos_df=time_infos_df.set_index('OUTG_REC_NB')
        #-------------------------
        time_infos_df['t_min'] = pd.to_datetime(time_infos_df['DT_OFF_TS_FULL'])+td_for_left
        if times_relative_to_off_ts_only:
            assert(td_for_left<=td_for_right)
            time_infos_df['t_max'] = pd.to_datetime(time_infos_df['DT_OFF_TS_FULL'])+td_for_right
        else:
            time_infos_df['t_max'] = pd.to_datetime(time_infos_df['DT_ON_TS'])+td_for_right
            assert(all(time_infos_df['t_min'] <= time_infos_df['t_max']))
        #-------------------------
        time_infos_df = time_infos_df.drop(columns=['DT_OFF_TS_FULL', 'DT_ON_TS'])
        #-------------------------
        # Remove any duplicates from time_infos_df
        if time_infos_df.index.nlevels==1:
            tmp_col = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=[tmp_col])
        else:
            tmp_cols = [Utilities.generate_random_string() for _ in range(time_infos_df.index.nlevels)]
            for i_col, tmp_col in enumerate(tmp_cols):
                time_infos_df[tmp_col] = time_infos_df.index.get_level_values(i_col)
            time_infos_df = time_infos_df.drop_duplicates()
            time_infos_df = time_infos_df.drop(columns=tmp_cols)
        #-------------------------
        return time_infos_df


    #---------------------------------------------------------------------------
    @staticmethod
    def build_reason_counts_per_outage_from_csvs_v0(    
        files_dir, 
        file_path_glob, 
        file_path_regex, 
        mp_df, 
        min_outg_td_window=datetime.timedelta(days=1),
        max_outg_td_window=datetime.timedelta(days=30),
        build_ede_typeid_to_reason_df=False, 
        batch_size=50, 
        cols_and_types_to_convert_dict=None, 
        to_numeric_errors='coerce', 
        assert_all_cols_equal=True, 
        include_normalize_by_nSNs=True, 
        inclue_zero_counts=True, 
        return_multiindex_outg_reason=False, 
        return_normalized_separately=False, 
        verbose=True, 
        n_update=1, 
        grp_by_cols='outg_rec_nb', 
        outg_rec_nb_col='outg_rec_nb',
        trsf_pole_nb_col = 'trsf_pole_nb', 
        addtnl_dropna_subset_cols=None, 
        is_no_outage=False, 
        prem_nb_col='aep_premise_nb', 
        serial_number_col='serialnumber', 
        include_prem_nbs=False, 
        trust_sql_grouping=True, 
        mp_df_cols = dict(
            serial_number_col='mfr_devc_ser_nbr', 
            prem_nb_col='prem_nb', 
            outg_rec_nb_col='OUTG_REC_NB'
        )
    ):
        r"""
        Note: The larger the batch_size, the more memory that will be consumed during building
    
        Any rows with NaNs in outg_rec_nb_col+addtnl_dropna_subset_cols will be dropped
    
        #NOTE: Currently, only set up for the case return_normalized_separately==False
        """
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO Allow method for reading DOVS from CSV (instead of only from SQL query)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #-------------------------
        assert(not return_normalized_separately)
        #-------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]        
        #-------------------------
        # If including prem numbers, also include n_prem numbers
        include_nprem_nbs = include_prem_nbs
        #-------------------------
        # normalize_by_nSNs_included needed when return_multiindex_outg_reason is True
        if include_normalize_by_nSNs and not return_normalized_separately:
            normalize_by_nSNs_included=True
        else:
            normalize_by_nSNs_included=False
    
        are_dfs_wide_form = not return_multiindex_outg_reason
        is_norm=False #TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir=files_dir, 
            glob_pattern=file_path_glob, 
            regex_pattern=file_path_regex
        )
        if len(paths)==0:
            print(f'No paths found in files_dir = {files_dir}')
            return None
        paths=natsorted(paths)
        #-------------------------
        rcpo_full = pd.DataFrame()
        #-------------------------
        batch_idxs = Utilities.get_batch_idx_pairs(len(paths), batch_size)
        n_batches = len(batch_idxs)    
        if verbose:
            print(f'n_paths = {len(paths)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches = {n_batches}')    
        #-------------------------
        for i, batch_i in enumerate(batch_idxs):
            start = time.time()
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-----
            # NOTE: make_all_columns_lowercase=True because...
            #   EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
            #   Athena maintains the original case, and does not conver to lower case,
            #     so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
            end_events_df_i = GenAn.read_df_from_csv_batch(
                paths                          = paths[i_beg:i_end], 
                cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
                to_numeric_errors              = to_numeric_errors, 
                make_all_columns_lowercase     = True, 
                assert_all_cols_equal          = assert_all_cols_equal
            )
            print(f'0: {time.time()-start}')
            #-------------------------
            if end_events_df_i.shape[0]==0:
                continue
            #-------------------------
            start = time.time()
            for grp_by_col in grp_by_cols:
                if f'{grp_by_col}_gpd_for_sql' in end_events_df_i.columns:
                    if grp_by_col in end_events_df_i.columns:
                        if trust_sql_grouping:
                            end_events_df_i = end_events_df_i.drop(columns=[grp_by_col])
                        else:
                            end_events_df_i = end_events_df_i.drop(columns=[f'{grp_by_col}_gpd_for_sql'])
                    end_events_df_i = end_events_df_i.rename(columns={f'{grp_by_col}_gpd_for_sql':grp_by_col})
    #             assert(grp_by_col in end_events_df_i.columns)
            #-----
            if not is_no_outage:
                if f'{outg_rec_nb_col}_gpd_for_sql' in end_events_df_i.columns:
                    if outg_rec_nb_col in end_events_df_i.columns:
                        if trust_sql_grouping:
                            end_events_df_i = end_events_df_i.drop(columns=[outg_rec_nb_col])
                        else:
                            end_events_df_i = end_events_df_i.drop(columns=[f'{outg_rec_nb_col}_gpd_for_sql'])
                    end_events_df_i = end_events_df_i.rename(columns={f'{outg_rec_nb_col}_gpd_for_sql':outg_rec_nb_col})
                assert(outg_rec_nb_col in end_events_df_i.columns)
            #-------------------------
            merge_on_mp = [mp_df_cols['serial_number_col'], mp_df_cols['prem_nb_col']]
            merge_on_ede = [serial_number_col, prem_nb_col]
            if not is_no_outage:
                merge_on_mp.append(mp_df_cols['outg_rec_nb_col'])
                merge_on_ede.append(outg_rec_nb_col)
            if trsf_pole_nb_col in end_events_df_i.columns:
                assert(trsf_pole_nb_col in mp_df.columns)
                merge_on_mp.append(trsf_pole_nb_col)
                merge_on_ede.append(trsf_pole_nb_col)
            # Below ensures there is only one entry per 'meter' (meter here is defined by a unique grouping of merge_on_mp)
            assert(not any(mp_df.groupby(merge_on_mp).size()>1))
            #-----
            # FROM WHAT I CAN TELL, the meters which are 'missing' from meter premise but are present in end events
            #   are caused by meters which were removed or installed close to (a few days/weeks before) an outage.
            # Therefore, although the meter may not have been present at the time of the outage (and therefore was exluced from
            #   meter premise), it could have registered events leading up to/following the outage.
            # e.g., if a meter was removed in the days before an outage, end events are still found for this meter in the days leading up to the outage
            # e.g., if a meter was installed in the days after an outage, end events are still found for this meter in the days following an outage.
            # How should these be handled?
            # The simplest method, which I will implement for now, is to simply ONLY consider those meters which were present
            #   at the time of the outage.  THEREFORE, the two DFs should be joined with an inner merge!
            end_events_df_i = AMIEndEvents.merge_end_events_df_with_mp(
                end_events_df=end_events_df_i, 
                df_mp=mp_df, 
                merge_on_ede=merge_on_ede, 
                merge_on_mp=merge_on_mp, 
                cols_to_include_mp=None, 
                drop_cols = None, 
                rename_cols=None, 
                how='inner', 
                inplace=True
            )
            for grp_by_col in grp_by_cols:
                assert(grp_by_col in end_events_df_i.columns)
            #-------------------------
            dropna_subset_cols = grp_by_cols
            if addtnl_dropna_subset_cols is not None:
                dropna_subset_cols.extend(addtnl_dropna_subset_cols)
            end_events_df_i = end_events_df_i.dropna(subset=dropna_subset_cols)
            #-------------------------
            end_events_df_i=Utilities_dt.strip_tz_info_and_convert_to_dt(
                df=end_events_df_i, 
                time_col='valuesinterval', 
                placement_col='valuesinterval_local', 
                run_quick=True, 
                n_strip=6, 
                inplace=False
            )
            print(f'1: {time.time()-start}')
            start = time.time()
            #---------------------------------------------------------------------------
            # If min_outg_td_window or max_outg_td_window is not None, enforce time restrictions around outages
            if min_outg_td_window is not None or max_outg_td_window is not None:
                #----------
                if not is_no_outage:
                    dovs_outgs = DOVSOutages(                 
                        df_construct_type=DFConstructType.kRunSqlQuery, 
                        contstruct_df_args=None, 
                        init_df_in_constructor=True, 
                        build_sql_function=DOVSOutages_SQL.build_sql_outage, 
                        build_sql_function_kwargs=dict(
                            outg_rec_nbs=end_events_df_i[outg_rec_nb_col].unique().tolist(), 
                            from_table_alias='DOV', 
                            datetime_col='DT_OFF_TS_FULL', 
                            cols_of_interest=[
                                'OUTG_REC_NB', 
                                dict(field_desc=f"DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)", 
                                     alias='DT_OFF_TS_FULL', table_alias_prefix=None)
                            ], 
                            field_to_split='outg_rec_nbs'
                        ),
                    )
                    outg_dt_off_df = dovs_outgs.df
                    outg_dt_off_df = Utilities_df.convert_col_type(df=outg_dt_off_df, column='OUTG_REC_NB', to_type=str)
                    outg_dt_off_df=outg_dt_off_df.set_index('OUTG_REC_NB')
                    outg_dt_off_series = outg_dt_off_df['DT_OFF_TS_FULL']
    
                    end_events_df_i = AMIEndEvents.enforce_end_events_within_interval_of_outage(
                        end_events_df=end_events_df_i, 
                        outg_times_series=outg_dt_off_series, 
                        min_timedelta=min_outg_td_window, 
                        max_timedelta=max_outg_td_window, 
                        outg_rec_nb_col = outg_rec_nb_col, 
                        datetime_col='valuesinterval_local', 
                        assert_one_time_per_group=True
                    )
                else:
                    no_outg_time_infos_df = MECPOAn.get_bsln_time_interval_infos_df_from_summary_files(
                        summary_paths=[AMIEndEvents.find_summary_file_from_csv(x) for x in paths[i_beg:i_end]], 
                        output_prem_nbs_col='prem_nbs', 
                        output_t_min_col='t_min', 
                        output_t_max_col='DT_OFF_TS_FULL', 
                        make_addtnl_groupby_idx=True, 
                        include_summary_paths=False
                    )
                    assert(no_outg_time_infos_df.index.name==trsf_pole_nb_col)
                    assert(end_events_df_i[trsf_pole_nb_col].dtype==no_outg_time_infos_df.index.dtype)
                    no_outg_time_infos_series = no_outg_time_infos_df['DT_OFF_TS_FULL']    
    
                    end_events_df_i = AMIEndEvents.enforce_end_events_within_interval_of_outage(
                        end_events_df=end_events_df_i, 
                        outg_times_series=no_outg_time_infos_series, 
                        min_timedelta=min_outg_td_window, 
                        max_timedelta=max_outg_td_window, 
                        outg_rec_nb_col = trsf_pole_nb_col, 
                        datetime_col='valuesinterval_local', 
                        assert_one_time_per_group=False
                    )
            print(f'2: {time.time()-start}')
            #---------------------------------------------------------------------------
            # After enforcing events within specific time frame, it is possible end_events_df_i is empty.
            # If so, continue
            if end_events_df_i.shape[0]==0:
                continue
            #-------------------------
    #         end_events_df_i = AMIEndEvents.extract_reboot_counts_from_reasons_in_df(end_events_df_i)
    #         end_events_df_i = AMIEndEvents.extract_fail_reasons_from_reasons_in_df(end_events_df_i)
            #-------------------------
            start = time.time()
            end_events_df_i = AMIEndEvents.reduce_end_event_reasons_in_df(
                df                            = end_events_df_i, 
                reason_col                    = 'reason', 
                edetypeid_col                 = 'enddeviceeventtypeid', 
                patterns_to_replace_by_typeid = None, 
                addtnl_patterns_to_replace    = None, 
                placement_col                 = None, 
                count                         = 0, 
                flags                         = re.IGNORECASE,  
                inplace                       = True
            )
            print(f'3: {time.time()-start}')
            #-------------------------
            start = time.time()
            if build_ede_typeid_to_reason_df:
                ede_typeid_to_reason_df_i = AMIEndEvents.build_ede_typeid_to_reason_df(
                    end_events_df=end_events_df_i, 
                    reason_col='reason', 
                    ede_typeid_col='enddeviceeventtypeid'
                )
                if i==0:
                    ede_typeid_to_reason_df = ede_typeid_to_reason_df_i.copy()
                else:
                    ede_typeid_to_reason_df = AMIEndEvents.combine_two_ede_typeid_to_reason_dfs(
                        ede_typeid_to_reason_df1=ede_typeid_to_reason_df, 
                        ede_typeid_to_reason_df2=ede_typeid_to_reason_df_i,
                        sort=True
                    )
            print(f'4: {time.time()-start}')
            #-------------------------
    #         ordinal_encoder = OrdinalEncoder()
    #         end_events_df_i['reason_enc'] = ordinal_encoder.fit_transform(end_events_df_i[['reason']])
            #-------------------------
            start = time.time()
            rcpo_i = AMIEndEvents.get_reason_counts_per_group(
                end_events_df = end_events_df_i, 
                group_cols=grp_by_cols, 
                group_freq=None, 
                serial_number_col='serialnumber', 
                reason_col='reason', 
                include_normalize_by_nSNs=include_normalize_by_nSNs, 
                inclue_zero_counts=inclue_zero_counts,
                possible_reasons=None, 
                include_nSNs=True, 
                include_SNs=True, 
                prem_nb_col=prem_nb_col, 
                include_nprem_nbs=include_nprem_nbs,
                include_prem_nbs=include_prem_nbs,   
                return_form = dict(return_multiindex_outg_reason = return_multiindex_outg_reason, 
                                   return_normalized_separately  = return_normalized_separately)
            )
            print(f'5: {time.time()-start}')
            #-------------------------
            if rcpo_i.shape[0]==0:
                continue
            #-------------------------
            start = time.time()
            # Include or below in case i=0 rcpo_i comes back empty, causing continue to be called above
            if i==0 or rcpo_full.shape[0]==0:
                rcpo_full = rcpo_i.copy()
            else:
                list_cols=['_SNs']
                list_counts_cols=['_nSNs']
                w_col = '_nSNs'
                if include_prem_nbs:
                    list_cols.append('_prem_nbs')
                    list_counts_cols.append('_nprem_nbs')
                #-----
                rcpo_full = AMIEndEvents.combine_two_reason_counts_per_outage_dfs(
                    rcpo_df_1=rcpo_full, 
                    rcpo_df_2=rcpo_i, 
                    are_dfs_wide_form=are_dfs_wide_form, 
                    normalize_by_nSNs_included=normalize_by_nSNs_included, 
                    is_norm=is_norm, 
                    list_cols=list_cols, 
                    list_counts_cols=list_counts_cols,
                    w_col = w_col, 
                    level_0_raw_col = 'counts', 
                    level_0_nrm_col = 'counts_norm', 
                    convert_rcpo_wide_to_long_col_args=None
                )
            print(f'6: {time.time()-start}')
        if not build_ede_typeid_to_reason_df:
            return rcpo_full
        else:
            return rcpo_full, ede_typeid_to_reason_df
        

    #---------------------------------------------------------------------------
    @staticmethod
    def perform_std_col_renames_and_drops(
        end_events_df              , 
        cols_to_drop               = None, 
        rename_cols_dict           = None, 
        make_all_columns_lowercase = True
    ):
        r"""
        """
        #-------------------------
        if make_all_columns_lowercase:
            end_events_df = Utilities_df.make_all_column_names_lowercase(end_events_df)
        #-------------------------
        # Drop any columns from cols_to_drop
        # IMPORTANT: This should come BEFORE rename
        if cols_to_drop is not None:
            end_events_df = end_events_df.drop(columns=cols_to_drop)
        #-------------------------
        # Rename any columns from rename_cols_dict
        # IMPORTANT: This should come AFTER drop
        if rename_cols_dict is not None:
            end_events_df = end_events_df.rename(columns=rename_cols_dict)
        #-------------------------
        return end_events_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def check_end_events_df_merge_with_mp(
        end_events_df              ,
        mp_df                      , 
        threshold_pct              = 1.0, 
        outg_rec_nb_col            = 'outg_rec_nb',
        trsf_pole_nb_col           = 'trsf_pole_nb', 
        prem_nb_col                = 'aep_premise_nb', 
        serial_number_col          = 'serialnumber',
        mp_df_cols                 = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        is_no_outage               = False, 
        cols_to_drop               = None, 
        rename_cols_dict           = None, 
        make_all_columns_lowercase = True, 
    ):
        r"""
        When building the RCPX dfs, oftentimes the end events data need to be joined with meter premise data (e.g., when one
          wants to group by transformer, this information typically needs to be brought in from MP).
        The meter premise data are supplied by the user.  Mistakes can be made, so this function serves to check whether the 
          alignment between end events and meter premise is as expected.
          
        NOTE: This also runs some other checks on the end events collection as a whole (e.g., if assert_all_cols_equal, it
                checks the assertion).
              Thus, some of the operations can be replaced from the for loop in build_reason_counts_per_outage_from_csv
                (This will help performance, but I don't think gains will be significant)
                
        In order to reduce some of the load on build_reason_counts_per_outage_from_csv, this function returns merge_on_ede and merge_on_mp
          
        End events and meter premise are merged on serial number and premise number.
          If data are for outages, also merged on outg_rec_nb.
          If trsf_pole_nb is present, also merged on it.
          
        NOTE:  I have found there are meters which are 'missing' from meter premise but are present in end events.
               These seem to be caused by meters which were removed or installed close to (a few days/weeks before) an outage.
               Therefore, although the meter may not have been present at the time of the outage (and therefore was exluced from
                 meter premise), it could have registered events leading up to/following the outage.
               e.g., if a meter was removed in the days before an outage, end events are still found for this meter in the days leading up to the outage
               e.g., if a meter was installed in the days after an outage, end events are still found for this meter in the days following an outage.
               How should these be handled?
               The simplest method, which I will implement for now, is to simply ONLY consider those meters which were present
                 at the time of the outage.  THEREFORE, the two DFs should be joined with an inner merge! 
                 
        NOTE: Utilities_df.make_all_column_names_lowercase utilized when reading in end events dfs because...
          EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
          Athena maintains the original case, and does not conver to lower case,
            so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
                 
        threshold_pct:
            If the percentage of entries present in end events but absent in meter premise exceeds this value, the program will crash.
            This is to safeguard the user from inputting incorrect meter premise data.
        """
        #-------------------------
        merge_on_mp = [mp_df_cols['serial_number_col'], mp_df_cols['prem_nb_col']]
        merge_on_ede = [serial_number_col, prem_nb_col]
        if not is_no_outage:
            merge_on_mp.append(mp_df_cols['outg_rec_nb_col'])
            merge_on_ede.append(outg_rec_nb_col)
        #-------------------------
        end_events_df = OutageMdlrPrep.perform_std_col_renames_and_drops(
            end_events_df              = end_events_df, 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-------------------------
        if trsf_pole_nb_col in end_events_df.columns:
            assert(mp_df_cols['trsf_pole_nb_col'] in mp_df.columns)
            merge_on_mp.append(mp_df_cols['trsf_pole_nb_col'])
            merge_on_ede.append(trsf_pole_nb_col)
        #-------------------------
        # Below ensures there is only one entry per 'meter' (meter here is defined by a unique grouping of merge_on_mp)
        assert(not any(mp_df.groupby(merge_on_mp).size()>1))         
        #-------------------------
        # Alter dtypes in mp_df if needed for proper merging
        end_events_df, mp_df = Utilities_df.make_df_col_dtypes_equal(
            df_1              = end_events_df, 
            col_1             = merge_on_ede, 
            df_2              = mp_df, 
            col_2             = merge_on_mp, 
            allow_reverse_set = False, 
            assert_success    = True
        )
        #-------------------------
        # Now, compare the unique combinations of merge_on_ede in end_events_df to those of merge_on_mp in mp_df
        gps_ede = list(end_events_df.groupby(merge_on_ede).groups.keys())
        gps_mp  = list(mp_df.groupby(merge_on_mp).groups.keys())
        # Finally, find the percent of meters in end events missing from meter premise (where a meter is defined as a unique
        #   combination of merge_on_ede values (e.g., serial number and premise number))
        n_missing = len(set(gps_ede).difference(set(gps_mp)))
        pct_missing = 100*(n_missing/len(gps_ede))
        print(f'% meters in end events missing from mp_df: {pct_missing}')
        if pct_missing>threshold_pct:
            assert(0)
        #-------------------------
        return mp_df, merge_on_ede, merge_on_mp


    #---------------------------------------------------------------------------
    @staticmethod
    def check_end_events_merge_with_mp(
        ede_file_paths             ,
        mp_df                      , 
        threshold_pct              = 1.0, 
        outg_rec_nb_col            = 'outg_rec_nb',
        trsf_pole_nb_col           = 'trsf_pole_nb', 
        prem_nb_col                = 'aep_premise_nb', 
        serial_number_col          = 'serialnumber',
        mp_df_cols                 = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        is_no_outage               = False, 
        assert_all_cols_equal      = True, 
        cols_to_drop               = None, 
        rename_cols_dict           = None, 
        make_all_columns_lowercase = True, 
    ):
        r"""
        When building the RCPX dfs, oftentimes the end events data need to be joined with meter premise data (e.g., when one
          wants to group by transformer, this information typically needs to be brought in from MP).
        The meter premise data are supplied by the user.  Mistakes can be made, so this function serves to check whether the 
          alignment between end events and meter premise is as expected.
          
        NOTE: This also runs some other checks on the end events collection as a whole (e.g., if assert_all_cols_equal, it
                checks the assertion).
              Thus, some of the operations can be replaced from the for loop in build_reason_counts_per_outage_from_csv
                (This will help performance, but I don't think gains will be significant)
                
        In order to reduce some of the load on build_reason_counts_per_outage_from_csv, this function returns merge_on_ede and merge_on_mp
          
        End events and meter premise are merged on serial number and premise number.
          If data are for outages, also merged on outg_rec_nb.
          If trsf_pole_nb is present, also merged on it.
          
        NOTE:  I have found there are meters which are 'missing' from meter premise but are present in end events.
               These seem to be caused by meters which were removed or installed close to (a few days/weeks before) an outage.
               Therefore, although the meter may not have been present at the time of the outage (and therefore was exluced from
                 meter premise), it could have registered events leading up to/following the outage.
               e.g., if a meter was removed in the days before an outage, end events are still found for this meter in the days leading up to the outage
               e.g., if a meter was installed in the days after an outage, end events are still found for this meter in the days following an outage.
               How should these be handled?
               The simplest method, which I will implement for now, is to simply ONLY consider those meters which were present
                 at the time of the outage.  THEREFORE, the two DFs should be joined with an inner merge! 
                 
        NOTE: Utilities_df.make_all_column_names_lowercase utilized when reading in end events dfs because...
          EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
          Athena maintains the original case, and does not conver to lower case,
            so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
                 
        threshold_pct:
            If the percentage of entries present in end events but absent in meter premise exceeds this value, the program will crash.
            This is to safeguard the user from inputting incorrect meter premise data.
        """
        #-------------------------
        merge_on_mp = [mp_df_cols['serial_number_col'], mp_df_cols['prem_nb_col']]
        merge_on_ede = [serial_number_col, prem_nb_col]
        if not is_no_outage:
            merge_on_mp.append(mp_df_cols['outg_rec_nb_col'])
            merge_on_ede.append(outg_rec_nb_col)
        #-------------------------
        # Grab the first end events df and check for trsf_pole_nb
        #  Actually, just need to grab first entry of first df
        end_events_df_0 = GenAn.read_df_from_csv(
            read_path=ede_file_paths[0], 
            cols_and_types_to_convert_dict=None, 
            to_numeric_errors='coerce', 
            drop_na_rows_when_exception=True, 
            drop_unnamed0_col=True, 
            pd_read_csv_kwargs = dict(nrows=1)
        )
        #-------------------------
        end_events_df_0 = OutageMdlrPrep.perform_std_col_renames_and_drops(
            end_events_df             = end_events_df_0, 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-------------------------
        if trsf_pole_nb_col in end_events_df_0.columns:
            assert(mp_df_cols['trsf_pole_nb_col'] in mp_df.columns)
            merge_on_mp.append(mp_df_cols['trsf_pole_nb_col'])
            merge_on_ede.append(trsf_pole_nb_col)
        #-------------------------
        # Below ensures there is only one entry per 'meter' (meter here is defined by a unique grouping of merge_on_mp)
        assert(not any(mp_df.groupby(merge_on_mp).size()>1))
        #--------------------------------------------------
        # Build DF with all end events to be checked with mp_df
        # The only columns which matter for the check are those in merge_on_ede, so those are the only kept.
        #   This keeps the overall size of the end_events_df smaller, allowing all to be loaded at once in most cases.
        dfs = []
        for path in ede_file_paths:
            df_i = GenAn.read_df_from_csv(
                read_path=path, 
                cols_and_types_to_convert_dict=None, 
                to_numeric_errors='coerce', 
                drop_na_rows_when_exception=True, 
                drop_unnamed0_col=True
            )
            if df_i.shape[0]==0:
                continue
            #-------------------------
            df_i = OutageMdlrPrep.perform_std_col_renames_and_drops(
                end_events_df              = df_i, 
                cols_to_drop               = cols_to_drop, 
                rename_cols_dict           = rename_cols_dict, 
                make_all_columns_lowercase = make_all_columns_lowercase
            )
            #-------------------------
            dfs.append(df_i[merge_on_ede])
        #-------------------------                
        df_cols = Utilities_df.get_shared_columns(dfs, maintain_df0_order=True)
        for i in range(len(dfs)):
            if assert_all_cols_equal:
                # In order to account for case where columns are the same but in different order
                # one must compare the length of dfs[i].columns to that of df_cols (found by utilizing
                # the Utilities_df.get_shared_columns(dfs) functionality)
                assert(dfs[i].shape[1]==len(df_cols))
            dfs[i] = dfs[i][df_cols]
        end_events_df = pd.concat(dfs)
        #-------------------------
        # Alter dtypes in mp_df if needed for proper merging
        end_events_df, mp_df = Utilities_df.make_df_col_dtypes_equal(
            df_1              = end_events_df, 
            col_1             = merge_on_ede, 
            df_2              = mp_df, 
            col_2             = merge_on_mp, 
            allow_reverse_set = False, 
            assert_success    = True
        )
        #-------------------------
        # Now, compare the unique combinations of merge_on_ede in end_events_df to those of merge_on_mp in mp_df
        gps_ede = list(end_events_df.groupby(merge_on_ede).groups.keys())
        gps_mp  = list(mp_df.groupby(merge_on_mp).groups.keys())
        # Finally, find the percent of meters in end events missing from meter premise (where a meter is defined as a unique
        #   combination of merge_on_ede values (e.g., serial number and premise number))
        n_missing = len(set(gps_ede).difference(set(gps_mp)))
        pct_missing = 100*(n_missing/len(gps_ede))
        print(f'% meters in end events missing from mp_df: {pct_missing}')
        if pct_missing>threshold_pct:
            assert(0)
        #-------------------------
        return mp_df, merge_on_ede, merge_on_mp


    #---------------------------------------------------------------------------
    @staticmethod
    def perform_build_RCPX_from_csvs_prereqs_v0(
        files_dir, 
        file_path_glob, 
        file_path_regex,
        mp_df, 
        ede_mp_mismatch_threshold_pct=1.0, 
        grp_by_cols='outg_rec_nb', 
        outg_rec_nb_col='outg_rec_nb',
        trsf_pole_nb_col = 'trsf_pole_nb', 
        prem_nb_col='aep_premise_nb', 
        serial_number_col='serialnumber',
        mp_df_cols = dict(
            serial_number_col='mfr_devc_ser_nbr', 
            prem_nb_col='prem_nb', 
            trsf_pole_nb_col = 'trsf_pole_nb', 
            outg_rec_nb_col='OUTG_REC_NB'
        ), 
        is_no_outage=False, 
        assert_all_cols_equal=True, 
        trust_sql_grouping=True, 
        make_all_columns_lowercase=True, 
    ):
        r"""
        Prepares for running build_reason_counts_per_outage_from_csvs.
        Many of these items were done for each iteration of the main for loop in build_reason_counts_per_outage_from_csvs, which
          was really unnecessary.  
          This will improve performance (although, likely not much as none of the operations are super heavy) and simplify the code.
        
        1. Adjust grp_by_cols and outg_rec_nb_col to handle cases where _gpd_for_sql appendix was added during data acquisition.
           If there is an instance where, e.g., outg_rec_nb_col and f'{outg_rec_nb_col}_gpd_for_sql' are both present, it settles
             the discrepancy (according to the trust_sql_grouping parameter) and compiles a list of columns which will need 
             to be dropped.
        2. Determine merge_on_ede and merge_on_mp columns (done within OutageMdlrPrep.check_end_events_merge_with_mp).
        3. Checks that the user supplied mp_df aligns well with the end events data.
        4. If assert_all_cols_equal==True, enforce the assertion.
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]    
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir=files_dir, 
            glob_pattern=file_path_glob, 
            regex_pattern=file_path_regex
        )
        if len(paths)==0:
            print(f'No paths found in files_dir = {files_dir}')
            return None
        paths=natsorted(paths)
        #-------------------------
        #--------------------------------------------------
        # Grab first row from each CSV file to be used to check columns
        dfs = []
        for path in paths:
            df_i = GenAn.read_df_from_csv(
                read_path                      = path, 
                cols_and_types_to_convert_dict = None, 
                to_numeric_errors              = 'coerce', 
                drop_na_rows_when_exception    = True, 
                drop_unnamed0_col              = True, 
                pd_read_csv_kwargs             = dict(nrows=1)
            )
            if df_i.shape[0]==0:
                continue
            #-----
            # NOTE: make_all_columns_lowercase because...
            #   EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
            #   Athena maintains the original case, and does not conver to lower case,
            #     so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
            if make_all_columns_lowercase:
                df_i = Utilities_df.make_all_column_names_lowercase(df_i) 
            #-----
            dfs.append(df_i)
        #-------------------------
        if len(dfs)==0:
            print(f'No DFs found in files_dir={files_dir}')
            assert(0)
        #-------------------------                
        df_cols = Utilities_df.get_shared_columns(dfs, maintain_df0_order=True)
        for i in range(len(dfs)):
            if assert_all_cols_equal:
                # In order to account for case where columns are the same but in different order
                # one must compare the length of dfs[i].columns to that of df_cols (found by utilizing
                # the Utilities_df.get_shared_columns(dfs) functionality)
                assert(dfs[i].shape[1]==len(df_cols))
            dfs[i] = dfs[i][df_cols]
        end_events_df = pd.concat(dfs)
        #--------------------------------------------------
        #--------------------------------------------------
        # The columns in grp_by_cols may sometimes be appended with _gpd_for_sql (e.g., typically one will see outg_rec_nb_gpd_for_sql 
        #   not outg_rec_nb in the raw CSV files).
        # The _gpd_for_sql was appended during data acquisition.
        # However, the user typically does not remember this, and will usually input, e.g., outg_rec_nb_col='outg_rec_nb'
        #   Such a scenario will obviously lead to an error, as 'outg_rec_nb' is not found in the columns
        # The below methods serve to remedy that issue.
        # NOTE: In the case that column_i and column_i_gpd_for_sql are for whatever reason BOTH found in the DF, the parameter
        #       trust_sql_grouping directs the code on which to use (the other will be dropped)
        cols_to_drop = []
        #-------------------------
        grp_by_cols_final = []
        for grp_by_col in grp_by_cols:
            if f'{grp_by_col}_gpd_for_sql' in end_events_df.columns:
                if grp_by_col in end_events_df.columns:
                    # Both f'{grp_by_col}_gpd_for_sql' and grp_by_col.
                    # One must be kept (by inserting into grp_by_cols_final), and one dropped (by inserting into cols_to_drop)
                    if trust_sql_grouping:
                        grp_by_cols_final.append(f'{grp_by_col}_gpd_for_sql')
                        cols_to_drop.append(grp_by_col)
                    else:
                        grp_by_cols_final.append(grp_by_col)
                        cols_to_drop.append(f'{grp_by_col}_gpd_for_sql')
                else:
                    # Only f'{grp_by_col}_gpd_for_sql', not grp_by_col, so no need to drop anything
                    grp_by_cols_final.append(f'{grp_by_col}_gpd_for_sql')
            else:
                grp_by_cols_final.append(grp_by_col)
        grp_by_cols = grp_by_cols_final
        #-------------------------
        if not is_no_outage:
            if f'{outg_rec_nb_col}_gpd_for_sql' in end_events_df.columns:
                if outg_rec_nb_col in end_events_df.columns:
                    # Both f'{outg_rec_nb_col}_gpd_for_sql' and outg_rec_nb_col.
                    # One must be kept (by setting equal to outg_rec_nb_col), and one dropped (by inserting into cols_to_drop)
                    if trust_sql_grouping:
                        cols_to_drop.append(outg_rec_nb_col)
                        outg_rec_nb_col = f'{outg_rec_nb_col}_gpd_for_sql'
                    else:
                        cols_to_drop.append(f'{outg_rec_nb_col}_gpd_for_sql')
                        outg_rec_nb_col = outg_rec_nb_col
                else:
                    # Only f'{outg_rec_nb_col}_gpd_for_sql', not outg_rec_nb_col, so no need to drop anything
                    outg_rec_nb_col = f'{outg_rec_nb_col}_gpd_for_sql'
            assert(outg_rec_nb_col in end_events_df.columns)
        #-------------------------
        if f'{trsf_pole_nb_col}_gpd_for_sql' in end_events_df.columns:
            if trsf_pole_nb_col in end_events_df.columns:
                # Both f'{trsf_pole_nb_col}_gpd_for_sql' and trsf_pole_nb_col.
                # One must be kept (by setting equal to trsf_pole_nb_col), and one dropped (by inserting into cols_to_drop)
                if trust_sql_grouping:
                    cols_to_drop.append(trsf_pole_nb_col)
                    trsf_pole_nb_col = f'{trsf_pole_nb_col}_gpd_for_sql'
                else:
                    cols_to_drop.append(f'{trsf_pole_nb_col}_gpd_for_sql')
                    trsf_pole_nb_col = trsf_pole_nb_col
            else:
                # Only f'{trsf_pole_nb_col}_gpd_for_sql', not trsf_pole_nb_col, so no need to drop anything
                trsf_pole_nb_col = f'{trsf_pole_nb_col}_gpd_for_sql'        
        #-------------------------
        # NOTE: Many times outg_rec_nb_col is in grp_by_cols, hence the need for the set operation (to eliminate duplicates!)
        cols_to_drop = list(set(cols_to_drop))
        #--------------------------------------------------
        #--------------------------------------------------
        mp_df, merge_on_ede, merge_on_mp = OutageMdlrPrep.check_end_events_merge_with_mp(
            ede_file_paths             = paths,
            mp_df                      = mp_df, 
            threshold_pct              = ede_mp_mismatch_threshold_pct, 
            outg_rec_nb_col            = outg_rec_nb_col,
            trsf_pole_nb_col           = trsf_pole_nb_col, 
            prem_nb_col                = prem_nb_col, 
            serial_number_col          = serial_number_col,
            mp_df_cols                 = mp_df_cols, 
            is_no_outage               = is_no_outage, 
            assert_all_cols_equal      = assert_all_cols_equal, 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = None, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #--------------------------------------------------
        #--------------------------------------------------
        return_dict = dict(
            paths            = paths, 
            grp_by_cols      = grp_by_cols, 
            outg_rec_nb_col  = outg_rec_nb_col, 
            trsf_pole_nb_col = trsf_pole_nb_col, 
            cols_to_drop     = cols_to_drop, 
            mp_df            = mp_df, 
            merge_on_ede     = merge_on_ede, 
            merge_on_mp      = merge_on_mp
        )
        return return_dict


    #---------------------------------------------------------------------------
    @staticmethod
    def find_all_cols_with_gpd_for_sql_appendix(df):
        r"""
        The _gpd_for_sql appendix is added in the end events data acquisition stage, but is typically no longer needed after.
        NOTE: Method is case insensitive!! (i.e., col_gpd_for_sql --> col, col_GPD_FOR_SQL --> col, etc.)
        Returns a dict with key values equal to the found columns containing _gpd_for_sql and values w
        """
        #-------------------------
        found_cols_dict = {}
        for col in df.columns.tolist():
            if col.lower().endswith('_gpd_for_sql'):
                assert(col not in found_cols_dict.keys())
                found_cols_dict[col] = col[:-12]
        return found_cols_dict
    
    #---------------------------------------------------------------------------
    @staticmethod
    def rename_all_cols_with_gpd_for_sql_appendix(df):
        r"""
        """
        #-------------------------
        found_cols_dict = OutageMdlrPrep.find_all_cols_with_gpd_for_sql_appendix(df=df)
        df = df.rename(columns=found_cols_dict)
        return df


    #---------------------------------------------------------------------------
    @staticmethod
    def make_values_lowercase(input_dict):
        r"""
        Only works for dicts with string or list values!
        """
        #-------------------------
        assert(isinstance(input_dict, dict))
        #-------------------------
        # This only makes sense if all values are strings!
        if not Utilities.are_all_list_elements_one_of_types(lst=list(input_dict.values()), types=[str, list, tuple]):
            print(f"OutageMdlrPrep.make_values_lowercase, cannot continue, all values not of type string or list!\ninput_dict = {input_dict}")
            return input_dict
        #-------------------------
        output_dict = {}
        for k,v in input_dict.items():
            assert(k not in output_dict.keys())
            if isinstance(v, str):
                output_dict[k] = v.lower()
            else:
                assert(Utilities.is_object_one_of_types(v, [list,tuple]))
                output_dict[k] = [v_i.lower() for v_i in v]
        #-------------------------
        return output_dict

    #---------------------------------------------------------------------------
    @staticmethod
    def identify_ede_cols_of_interest_to_update_andor_drop(
        end_events_df,  
        grp_by_cols                = 'outg_rec_nb', 
        outg_rec_nb_col            = 'outg_rec_nb',
        trsf_pole_nb_col           = 'trsf_pole_nb', 
        prem_nb_col                = 'aep_premise_nb', 
        serial_number_col          = 'serialnumber',
        trust_sql_grouping         = True, 
        drop_gpd_for_sql_appendix  = True, 
        make_all_columns_lowercase = True
    ):
        r"""
        Find any entries in ede_cols_of_interest (see below) which contain _gpd_for_sql appendix and update accordingly.
          The columns in end events may sometimes be appended with _gpd_for_sql (e.g., typically one will see outg_rec_nb_gpd_for_sql 
            not outg_rec_nb in the raw CSV files).
          The _gpd_for_sql was appended during data acquisition.
          However, the user typically does not remember this, and will usually input, e.g., outg_rec_nb_col='outg_rec_nb'
            Such a scenario will obviously lead to an error, as 'outg_rec_nb' is not found in the columns
          The below methods serve to remedy that issue.
          
        ede_cols_of_interest consist of grp_by_cols+[outg_rec_nb_col, trsf_pole_nb_col, prem_nb_col, serial_number_col]
          
        NOTE: In the case that column_i and column_i_gpd_for_sql are for whatever reason BOTH found in the DF, the parameter
            trust_sql_grouping directs the code on which to use (the other will be dropped)

        -------------------------
        TYPICAL EXAMPLE:
            Input:
                end_events_df.columns = [
                    'issuertracking_id', 'serialnumber', 'enddeviceeventtypeid', 'valuesinterval', 'aep_premise_nb', 'reason', 'event_type', 'aep_opco', 
                    'aep_event_dt', 'trsf_pole_nb', 'trsf_pole_nb_gpd_for_sql', 'no_outg_rec_nb_gpd_for_sql', 'is_first_after_outg_gpd_for_sql'
                ]
                grp_by_cols        = ['trsf_pole_nb', 'no_outg_rec_nb']
                outg_rec_nb_col    = 'no_outg_rec_nb'
                trsf_pole_nb_col   = 'trsf_pole_nb'
                prem_nb_col        = 'aep_premise_nb'
                serial_number_col  = 'serialnumber'
                trust_sql_grouping = True

            Output:
                ede_gpd_coi_dict_updates = {
                    'grp_by_cols'      : ['trsf_pole_nb_gpd_for_sql', 'no_outg_rec_nb_gpd_for_sql'], 
                    'outg_rec_nb_col'  : 'outg_rec_nb', 
                    'trsf_pole_nb_col' : 'trsf_pole_nb_gpd_for_sql', 
                    'prem_nb_col'      : 'aep_premise_nb', 
                    'serial_number_col': 'serialnumber'
                }
                cols_to_drop = ['trsf_pole_nb']
        -------------------------

        drop_gpd_for_sql_appendix:
            If True, any remaining cols of interest/columns with gpd_for_sql appendix will be removed (cols of interest) or instructed
              to have their names changed (columns)
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]    
        #-------------------------
        # Below called ede_gpd_coi_dict because the columns of interest are grouped by their input parameter in build_reason_counts_per_outage_from_csvs.
        #   e.g., key value grp_by_cols should contain a list of columns (strings)
        #         key value outg_rec_nb_col contains a single string for the outg_rec_nb_col
        # The values, coi, can stand for column of interest (e.g., outg_rec_nb_col) or columns of interest (e.g., grp_by_cols)
        ede_gpd_coi_dict = dict(
            grp_by_cols       = grp_by_cols, 
            outg_rec_nb_col   = outg_rec_nb_col, 
            trsf_pole_nb_col  = trsf_pole_nb_col, 
            prem_nb_col       = prem_nb_col, 
            serial_number_col = serial_number_col
        )
        #-------------------------
        if make_all_columns_lowercase: 
            end_events_df    = Utilities_df.make_all_column_names_lowercase(end_events_df) 
            ede_gpd_coi_dict = OutageMdlrPrep.make_values_lowercase(input_dict = ede_gpd_coi_dict)
        #-------------------------
        # Below, ede_cols_of_interest is essentially the flattened values from ede_gpd_coi_dict with duplicates removed
        ede_cols_of_interest = []
        for ede_coi_grp, coi in ede_gpd_coi_dict.items():
            assert(Utilities.is_object_one_of_types(coi, [str,list,tuple]))
            if isinstance(coi, str):
                ede_cols_of_interest.append(coi)
            else:
                ede_cols_of_interest.extend(coi)
        # NOTE: grp_by_cols can contain the others (e.g., outg_rec_nb_col), so set operation needed to removed duplicates
        ede_cols_of_interest = list(set(ede_cols_of_interest))
        #--------------------------------------------------
        # Below, found_cols_w_gpd_for_sql_appendix has keys equal to any columns found containing _gpd_for_sql appendix and
        #   value equal to the column without the appendix.
        # found_cols_w_gpd_for_sql_appendix_inv is the inverse (i.e., keys and values switched)
        found_cols_w_gpd_for_sql_appendix     = OutageMdlrPrep.find_all_cols_with_gpd_for_sql_appendix(end_events_df)
        found_cols_w_gpd_for_sql_appendix_inv = Utilities.invert_dict(found_cols_w_gpd_for_sql_appendix)
        #--------------------------------------------------
        # NOTE: No harm below if ede_coi not actually found in end_events_df (e.g., when running over baseline data, outg_rec_nb_col 
        #       will not be found), as updates are only made to those whose values change (and can only change if found)
        ede_cols_of_interest_updates = dict()
        cols_to_drop = []
        #-----
        for ede_coi in ede_cols_of_interest:
            assert(ede_coi not in ede_cols_of_interest_updates)
            #-----
            if ede_coi in found_cols_w_gpd_for_sql_appendix_inv.keys():
                # f'{ede_coi}_gpd_for_sql' found in end_events_df, so an updadte to ede_coi in ede_cols_of_interest 
                #   and/or column drop will be needed
                if ede_coi in end_events_df.columns:
                    # Both f'{ede_coi}_gpd_for_sql' and ede_coi were found in end_events_df!
                    # One must be kept (by inserting into grp_by_cols_final), and one dropped (by inserting into cols_to_drop)
                    if trust_sql_grouping:
                        # Keep/update f'{ede_coi}_gpd_for_sql', and drop ede_coi
                        ede_cols_of_interest_updates[ede_coi] = found_cols_w_gpd_for_sql_appendix_inv[ede_coi]
                        cols_to_drop.append(ede_coi)
                    else:
                        # Keep ede_coi, drop f'{ede_coi}_gpd_for_sql'
                        ede_cols_of_interest_updates[ede_coi] = ede_coi
                        cols_to_drop.append(found_cols_w_gpd_for_sql_appendix_inv[ede_coi])
                else:
                    # Only f'{ede_coi}_gpd_for_sql' found in end_events_df, not grp_by_col, so no need to drop anything
                    ede_cols_of_interest_updates[ede_coi] = found_cols_w_gpd_for_sql_appendix_inv[ede_coi]
            else:
                ede_cols_of_interest_updates[ede_coi] = ede_coi
        #-----
        # Make sure no repeats in cols_to_drop
        cols_to_drop = list(set(cols_to_drop))
        #--------------------------------------------------
        # Build ede_gpd_coi_dict_updates, which has the same structure as ede_gpd_coi_dict (see above)
        ede_gpd_coi_dict_updates = dict()
        for ede_coi_grp, coi in ede_gpd_coi_dict.items():
            assert(ede_coi_grp not in ede_gpd_coi_dict_updates.keys())
            assert(Utilities.is_object_one_of_types(coi, [str,list,tuple]))
            if isinstance(coi, str):
                assert(coi in ede_cols_of_interest_updates.keys())
                ede_gpd_coi_dict_updates[ede_coi_grp] = ede_cols_of_interest_updates[coi]
            else:
                ede_gpd_coi_dict_updates[ede_coi_grp] = []
                for coi_i in coi:
                    assert(coi_i in ede_cols_of_interest_updates.keys())
                    ede_gpd_coi_dict_updates[ede_coi_grp].append(ede_cols_of_interest_updates[coi_i])
        assert(len(set(ede_gpd_coi_dict.keys()).symmetric_difference(set(ede_gpd_coi_dict_updates.keys())))==0)
        #-----
        # Sanity check
        assert(len(set(ede_gpd_coi_dict.keys()).symmetric_difference(set(ede_gpd_coi_dict_updates.keys())))==0)
        for ede_coi_grp, coi in ede_gpd_coi_dict.items():
            if Utilities.is_object_one_of_types(coi, [list,tuple]):
                assert(len(coi)==len(ede_gpd_coi_dict_updates[ede_coi_grp]))
        #--------------------------------------------------
        rename_dict = {}
        if drop_gpd_for_sql_appendix:
            rename_dict = OutageMdlrPrep.find_all_cols_with_gpd_for_sql_appendix(df = end_events_df)
            if len(rename_dict)>0:
                # Update ede_gpd_coi_dict_updates:
                ede_gpd_coi_dict_updates_fnl = {}
                for param_grp, cols_of_interest in ede_gpd_coi_dict_updates.items():
                    assert(param_grp not in ede_gpd_coi_dict_updates_fnl.keys())
                    assert(Utilities.is_object_one_of_types(cols_of_interest, [str,list,tuple]))
                    if isinstance(cols_of_interest, str):
                        ede_gpd_coi_dict_updates_fnl[param_grp] = rename_dict.get(cols_of_interest, cols_of_interest)
                    else:
                        cols_of_interest_new = []
                        for coi in cols_of_interest:
                            cols_of_interest_new.append(rename_dict.get(coi, coi))
                        ede_gpd_coi_dict_updates_fnl[param_grp] = cols_of_interest_new
                #-------------------------
                # Sanity checks
                #-----
                # Make sure all of rename_dict keys found in columns!
                assert(set(rename_dict.keys()).difference(set(end_events_df.columns))==set())
                #-----
                # Don't want any of the renames to already be included in the pd.DataFrame!
                assert(set(end_events_df.columns.drop(cols_to_drop)).intersection(set(rename_dict.values()))==set())
                #-----
                # Make sure all ede_cols_of_interest are found
                test_df = end_events_df.head().drop(columns=cols_to_drop).rename(columns=rename_dict)
                for param_grp, cols_of_interest in ede_gpd_coi_dict_updates_fnl.items():
                    assert(Utilities.is_object_one_of_types(cols_of_interest, [str,list,tuple]))
                    if isinstance(cols_of_interest, str):
                        assert(cols_of_interest in test_df.columns.tolist())
                    else:
                        for coi in cols_of_interest:
                            assert(coi in test_df.columns.tolist())
                #-------------------------
                ede_gpd_coi_dict_updates = ede_gpd_coi_dict_updates_fnl
            else:
                rename_dict = {}
        #--------------------------------------------------
        return ede_gpd_coi_dict_updates, cols_to_drop, rename_dict
    

    #---------------------------------------------------------------------------
    @staticmethod
    def perform_build_RCPX_from_end_events_df_prereqs(
        end_events_df                 , 
        mp_df                         = None, 
        ede_mp_mismatch_threshold_pct = 1.0, 
        grp_by_cols                   = 'outg_rec_nb', 
        outg_rec_nb_col               = 'outg_rec_nb',
        trsf_pole_nb_col              = 'trsf_pole_nb', 
        prem_nb_col                   = 'aep_premise_nb', 
        serial_number_col             = 'serialnumber',
        mp_df_cols                    = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        is_no_outage                  = False, 
        trust_sql_grouping            = True, 
        drop_gpd_for_sql_appendix     = True, 
        make_all_columns_lowercase    = True
    ):
        r"""
        Prepares for running build_reason_counts_per_outage_from_csvs.
        Many of these items were done for each iteration of the main for loop in build_reason_counts_per_outage_from_csvs, which
          was really unnecessary.  
          This will improve performance (although, likely not much as none of the operations are super heavy) and simplify the code.
        
        1. Adjust grp_by_cols and outg_rec_nb_col to handle cases where _gpd_for_sql appendix was added during data acquisition.
           If there is an instance where, e.g., outg_rec_nb_col and f'{outg_rec_nb_col}_gpd_for_sql' are both present, it settles
             the discrepancy (according to the trust_sql_grouping parameter) and compiles a list of columns which will need 
             to be dropped.
        2. Determine merge_on_ede and merge_on_mp columns (done within OutageMdlrPrep.check_end_events_merge_with_mp).
        3. Checks that the user supplied mp_df aligns well with the end events data.
        4. If assert_all_cols_equal==True, enforce the assertion.
        """
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]    
        #--------------------------------------------------
        ede_cols_of_interest_updates, cols_to_drop, rename_cols_dict = OutageMdlrPrep.identify_ede_cols_of_interest_to_update_andor_drop(
            end_events_df              = end_events_df,  
            grp_by_cols                = grp_by_cols, 
            outg_rec_nb_col            = outg_rec_nb_col,
            trsf_pole_nb_col           = trsf_pole_nb_col, 
            prem_nb_col                = prem_nb_col, 
            serial_number_col          = serial_number_col,
            trust_sql_grouping         = trust_sql_grouping, 
            drop_gpd_for_sql_appendix  = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        grp_by_cols       = ede_cols_of_interest_updates['grp_by_cols']
        outg_rec_nb_col   = ede_cols_of_interest_updates['outg_rec_nb_col']
        trsf_pole_nb_col  = ede_cols_of_interest_updates['trsf_pole_nb_col']
        prem_nb_col       = ede_cols_of_interest_updates['prem_nb_col']
        serial_number_col = ede_cols_of_interest_updates['serial_number_col']
        #--------------------------------------------------
        #--------------------------------------------------
        if mp_df is not None:
            mp_df, merge_on_ede, merge_on_mp = OutageMdlrPrep.check_end_events_df_merge_with_mp(
                end_events_df                 = end_events_df,
                mp_df                         = mp_df, 
                threshold_pct                 = ede_mp_mismatch_threshold_pct, 
                outg_rec_nb_col               = outg_rec_nb_col,
                trsf_pole_nb_col              = trsf_pole_nb_col, 
                prem_nb_col                   = prem_nb_col, 
                serial_number_col             = serial_number_col,
                mp_df_cols                    = mp_df_cols, 
                is_no_outage                  = is_no_outage, 
                cols_to_drop                  = cols_to_drop, 
                rename_cols_dict              = rename_cols_dict, 
                make_all_columns_lowercase    = make_all_columns_lowercase
            )
        else:
            merge_on_ede, merge_on_mp = None, None
        #--------------------------------------------------
        #--------------------------------------------------
        return_dict = dict(
            grp_by_cols       = grp_by_cols, 
            outg_rec_nb_col   = outg_rec_nb_col, 
            trsf_pole_nb_col  = trsf_pole_nb_col, 
            prem_nb_col       = prem_nb_col, 
            serial_number_col = serial_number_col,
            cols_to_drop      = cols_to_drop, 
            rename_cols_dict  = rename_cols_dict, 
            mp_df             = mp_df, 
            merge_on_ede      = merge_on_ede, 
            merge_on_mp       = merge_on_mp
        )
        return return_dict


    #---------------------------------------------------------------------------
    @staticmethod
    def perform_build_RCPX_from_csvs_prereqs(
        files_dir, 
        file_path_glob, 
        file_path_regex,
        mp_df                         = None, 
        ede_mp_mismatch_threshold_pct = 1.0, 
        grp_by_cols                   = 'outg_rec_nb', 
        outg_rec_nb_col               = 'outg_rec_nb',
        trsf_pole_nb_col              = 'trsf_pole_nb', 
        prem_nb_col                   = 'aep_premise_nb', 
        serial_number_col             = 'serialnumber',
        mp_df_cols                    = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        is_no_outage                  = False, 
        assert_all_cols_equal         = True, 
        trust_sql_grouping            = True, 
        drop_gpd_for_sql_appendix     = True, 
        make_all_columns_lowercase    = True, 
    ):
        r"""
        Prepares for running build_reason_counts_per_outage_from_csvs.
        Many of these items were done for each iteration of the main for loop in build_reason_counts_per_outage_from_csvs, which
          was really unnecessary.  
          This will improve performance (although, likely not much as none of the operations are super heavy) and simplify the code.
        
        1. Adjust grp_by_cols and outg_rec_nb_col to handle cases where _gpd_for_sql appendix was added during data acquisition.
           If there is an instance where, e.g., outg_rec_nb_col and f'{outg_rec_nb_col}_gpd_for_sql' are both present, it settles
             the discrepancy (according to the trust_sql_grouping parameter) and compiles a list of columns which will need 
             to be dropped.
        2. Determine merge_on_ede and merge_on_mp columns (done within OutageMdlrPrep.check_end_events_merge_with_mp).
        3. Checks that the user supplied mp_df aligns well with the end events data.
        4. If assert_all_cols_equal==True, enforce the assertion.

        NOTE: make_all_columns_lowercase because...
          EMR would return lowercase outg_rec_nb or outg_rec_nb_gpd_for_sql
          Athena maintains the original case, and does not conver to lower case,
            so it returns OUTG_REC_NB or OUTG_REC_NB_GPD_FOR_SQL
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]    
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir      = files_dir, 
            glob_pattern  = file_path_glob, 
            regex_pattern = file_path_regex
        )
        if len(paths)==0:
            print(f'No paths found in files_dir = {files_dir}')
            return None
        paths=natsorted(paths)
        #-------------------------
        #--------------------------------------------------
        # Grab first row from each CSV file to be used to check columns
        dfs = []
        for path in paths:
            df_i = GenAn.read_df_from_csv(
                read_path                      = path, 
                cols_and_types_to_convert_dict = None, 
                to_numeric_errors              = 'coerce', 
                drop_na_rows_when_exception    = True, 
                drop_unnamed0_col              = True, 
                pd_read_csv_kwargs             = dict(nrows=1)
            )
            if df_i.shape[0]==0:
                continue
            #-----
            dfs.append(df_i)
        #-------------------------
        if len(dfs)==0:
            print(f'No DFs found in files_dir={files_dir}')
            assert(0)
        #-------------------------                
        df_cols = Utilities_df.get_shared_columns(dfs, maintain_df0_order=True)
        for i in range(len(dfs)):
            if assert_all_cols_equal:
                # In order to account for case where columns are the same but in different order
                # one must compare the length of dfs[i].columns to that of df_cols (found by utilizing
                # the Utilities_df.get_shared_columns(dfs) functionality)
                assert(dfs[i].shape[1]==len(df_cols))
            dfs[i] = dfs[i][df_cols]
        end_events_df = pd.concat(dfs)
        #--------------------------------------------------
        #--------------------------------------------------
        ede_cols_of_interest_updates, cols_to_drop, rename_cols_dict = OutageMdlrPrep.identify_ede_cols_of_interest_to_update_andor_drop(
            end_events_df              = end_events_df,  
            grp_by_cols                = grp_by_cols, 
            outg_rec_nb_col            = outg_rec_nb_col,
            trsf_pole_nb_col           = trsf_pole_nb_col, 
            prem_nb_col                = prem_nb_col, 
            serial_number_col          = serial_number_col,
            trust_sql_grouping         = trust_sql_grouping, 
            drop_gpd_for_sql_appendix  = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        grp_by_cols       = ede_cols_of_interest_updates['grp_by_cols']
        outg_rec_nb_col   = ede_cols_of_interest_updates['outg_rec_nb_col']
        trsf_pole_nb_col  = ede_cols_of_interest_updates['trsf_pole_nb_col']
        prem_nb_col       = ede_cols_of_interest_updates['prem_nb_col']
        serial_number_col = ede_cols_of_interest_updates['serial_number_col']
        #--------------------------------------------------
        #--------------------------------------------------
        if mp_df is not None:
            mp_df, merge_on_ede, merge_on_mp = OutageMdlrPrep.check_end_events_merge_with_mp(
                ede_file_paths             = paths,
                mp_df                      = mp_df, 
                threshold_pct              = ede_mp_mismatch_threshold_pct, 
                outg_rec_nb_col            = outg_rec_nb_col,
                trsf_pole_nb_col           = trsf_pole_nb_col, 
                prem_nb_col                = prem_nb_col, 
                serial_number_col          = serial_number_col,
                mp_df_cols                 = mp_df_cols, 
                is_no_outage               = is_no_outage, 
                assert_all_cols_equal      = assert_all_cols_equal, 
                cols_to_drop               = cols_to_drop, 
                rename_cols_dict           = rename_cols_dict, 
                make_all_columns_lowercase = make_all_columns_lowercase
            )
        else:
            merge_on_ede, merge_on_mp = None, None
        #--------------------------------------------------
        #--------------------------------------------------
        return_dict = dict(
            paths             = paths, 
            grp_by_cols       = grp_by_cols, 
            outg_rec_nb_col   = outg_rec_nb_col, 
            trsf_pole_nb_col  = trsf_pole_nb_col, 
            prem_nb_col       = prem_nb_col, 
            serial_number_col = serial_number_col,
            cols_to_drop      = cols_to_drop, 
            rename_cols_dict  = rename_cols_dict, 
            mp_df             = mp_df, 
            merge_on_ede      = merge_on_ede, 
            merge_on_mp       = merge_on_mp
        )
        return return_dict
    

    #---------------------------------------------------------------------------
    @staticmethod
    def find_gpd_for_sql_cols(df, col_level=-1):
        r"""
        The _gpd_for_sql appendix is added in the end events data acquisition stage, but is typically no longer needed after.
        This function finds any columns in df with the appendix.
        
        NOTE: Method is case insensitive!! (i.e., col_gpd_for_sql --> col, col_GPD_FOR_SQL --> col, etc.)
        """
        #-------------------------
        found_cols = []
        for col in df.columns.get_level_values(col_level).tolist():
            if col.lower().endswith('_gpd_for_sql'):
                found_cols.append(col)
        return found_cols

    #---------------------------------------------------------------------------
    @staticmethod
    def drop_gpd_for_sql_appendix_from_all_cols(df):
        r"""
        The _gpd_for_sql appendix is added in the end events data acquisition stage, but is typically no longer needed after.
        This function removes the appendix.
        
        NOTE: Method is case insensitive!! (i.e., col_gpd_for_sql --> col, col_GPD_FOR_SQL --> col, etc.)
        """
        #-------------------------
        for col in df.columns.tolist():
            if col.lower().endswith('_gpd_for_sql'):
                df = df.rename(columns={col:col[:-12]})
        return df
    
    
    #---------------------------------------------------------------------------
    @staticmethod
    def drop_gpd_for_sql_appendix_from_all_index_names(df):
        r"""
        The _gpd_for_sql appendix is added in the end events data acquisition stage, but is typically no longer needed after.
        This function removes the appendix.
        
        NOTE: Method is case insensitive!! (i.e., col_gpd_for_sql --> col, col_GPD_FOR_SQL --> col, etc.)
        """
        #-------------------------
        idx_names_new = []
        for name in df.index.names:
            if name.lower().endswith('_gpd_for_sql'):
                idx_names_new.append(name[:-12])
            else:
                idx_names_new.append(name)
        #-----
        assert(len(df.index.names)==len(idx_names_new))
        df.index.names = idx_names_new
        return df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def set_faulty_mp_vals_to_nan_in_end_events_df(
        end_events_df, 
        prem_nb_col_ede='aep_premise_nb', 
        prem_nb_col_mp='prem_nb'
    ):
        r"""
        NOT REALLY NEEDED ANYMORE!
            As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
            If this was not true, then this function may actually be needed
            -----
            OutageMdlrPrep.perform_build_RCPX_from_csvs_prereqs all but ensures a merge with MeterPremise will include, at minimum,
              premise number and serial number.
            Therefore, this is rarely needed.
        -------------------------

        Initially assumed that joining meter_events.end_device_event with default.meter_premise soley on serial number was fine.
        However, have since found that the serial number in meter_events.end_device_event does not always appear to be correct.
        This leads to a situation where the wrong MP is matched, which throws off the trsf_pole_nb (and any other attributes
        taken from default.meter_premise).
        In the future, the two will be merged on both serial number and premise number.  The join will be left, so any entries
        which don't match simply will have NaN for the MP cols.
        The purpose of this function is to reproduce this effect for the erroneous data.
        """
        #-------------------------
        assert(prem_nb_col_ede in end_events_df.columns)
        assert(prem_nb_col_mp  in end_events_df.columns)
        #-------------------------
        # NOTE: Should have been left joined.  Therefore, it is possible and acceptable for prem_nb_col_mp to be NaN when 
        #       prem_nb_col_ede is not.  Thus, the need for '& (end_events_df[prem_nb_col_mp].notna())'
        faulty_mask = (end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp]) & (end_events_df[prem_nb_col_mp].notna())
        if faulty_mask.sum()==0:
            return end_events_df
        #-------------------------
        # Find the columns from default.meter_premise which were merged into end_events_df
        mp_merged_cols = [x for x in TableInfos.MeterPremise_TI.columns_full if x in end_events_df.columns.tolist()]
        # Maintain original order
        mp_merged_cols = [x for x in end_events_df.columns.tolist() if x in mp_merged_cols]
        #-------------------------
        # Set the mp_merged_cols to NaN for faulty_mask.
        end_events_df.loc[faulty_mask,mp_merged_cols]=np.nan
        #-------------------------
        # Make sure the procedure worked
        assert(((end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp]) & (end_events_df[prem_nb_col_mp].notna())).sum()==0)
        #-------------------------
        return end_events_df
        
    #---------------------------------------------------------------------------
    @staticmethod
    def correct_or_add_active_mp_cols(
        end_events_df, 
        df_time_col_0,
        df_time_col_1=None,
        df_mp_curr=None, 
        df_mp_hist=None, 
        df_and_mp_merge_pairs=[
            ['serialnumber', 'mfr_devc_ser_nbr'], 
            ['aep_premise_nb', 'prem_nb']
        ], 
        assert_all_PNs_found=True, 
        df_prem_nb_col='aep_premise_nb', 
        df_mp_prem_nb_col='prem_nb'
    ):
        r"""
        NOT REALLY NEEDED ANYMORE!
            As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
            If this was not true, then this function may actually be needed
            -----
            OutageMdlrPrep.perform_build_RCPX_from_csvs_prereqs all but ensures a merge with MeterPremise will include, at minimum,
              premise number and serial number.
            Therefore, this is rarely needed.
        -------------------------

        Built to correct the MP entries in end_events_df.
        When building EndEvents data for outages, the data we joined with MeterPremise at the SQL leve.
          However, they really need to be joined with whatever meters were active at the time of the event.
          This function replaces the faulty with good

        Any columns from MP (meaning, from default.meter_premise or default.meter_premise_hist) are first identified, as
          these will need to be replaced.
        i.  In the case where df_mp_curr/hist are supplied, whatever columns are contained in those DFs will be merged with 
              end_events_df (in which case, it is necessary for any found MP cols to be dropped)
        ii. In the case where df_mp_curr/hist are not supplied, they will be built and the MP columns found in end_events_df
              will be included as addtnl_mp_df_curr/hist_cols arguments.
            If no MP cols are found in end_events_df, then addtnl_mp_df_curr/hist_cols arguments will be empty, so the MP 
              columns to be added will be the bare minimum (i.e., 'mfr_devc_ser_nbr', 'prem_nb', 'trsf_pole_nb')
        """
        #-------------------------
        # Find the columns from default.meter_premise and default.meter_premise_hist which were merged into end_events_df
        #   as these will need to be replaced
        mp_curr_merged_cols = [x for x in TableInfos.MeterPremise_TI.columns_full if x in end_events_df.columns.tolist()]
        mp_hist_merged_cols = [x for x in TableInfos.MeterPremiseHist_TI.columns_full if x in end_events_df.columns.tolist()]
        # Maintain original order
        mp_curr_merged_cols = [x for x in end_events_df.columns.tolist() if x in mp_curr_merged_cols]
        mp_hist_merged_cols = [x for x in end_events_df.columns.tolist() if x in mp_hist_merged_cols]
        #-------------------------
        # Drop the MP columns from end_events_df (only truly necessary when df_mp_curr/hist are supplied, as when they are
        #   not, they will be built with the necessary columns which will then replace those in end_events_df.  However,
        #   for either case, dropping them at this point is safe)
        end_events_df = end_events_df.drop(columns=mp_curr_merged_cols+mp_hist_merged_cols)
        #-------------------------
        # Merge end_events_df with the active MP entries
        end_events_df = MeterPremise.merge_df_with_active_mp(
            df=end_events_df, 
            df_time_col_0=df_time_col_0, 
            df_time_col_1=df_time_col_1, 
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist, 
            df_and_mp_merge_pairs=df_and_mp_merge_pairs, 
            keep_overlap = 'right', 
            drop_inst_rmvl_cols_after_merge=True, 
            addtnl_mp_df_curr_cols=mp_curr_merged_cols, 
            addtnl_mp_df_hist_cols=mp_hist_merged_cols, 
            assume_one_xfmr_per_PN=True, 
            assert_all_PNs_found=assert_all_PNs_found, 
            df_prem_nb_col=df_prem_nb_col, 
            df_mp_serial_number_col='mfr_devc_ser_nbr', 
            df_mp_prem_nb_col=df_mp_prem_nb_col, 
            df_mp_install_time_col='inst_ts', 
            df_mp_removal_time_col='rmvl_ts', 
            df_mp_trsf_pole_nb_col='trsf_pole_nb'        
        )
        #-------------------------
        return end_events_df
        
        
    #---------------------------------------------------------------------------
    @staticmethod
    def correct_faulty_mp_vals_in_end_events_df(
        end_events_df, 
        df_time_col_0,
        df_time_col_1=None,
        df_mp_curr=None, 
        df_mp_hist=None, 
        prem_nb_col_ede='aep_premise_nb', 
        prem_nb_col_mp='prem_nb', 
        df_and_mp_merge_pairs=[
            ['serialnumber', 'mfr_devc_ser_nbr'], 
            ['aep_premise_nb', 'prem_nb']
        ], 
        assert_all_PNs_found=True    
    ):
        r"""
        NOT REALLY NEEDED ANYMORE!
            As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
            If this was not true, then this function may actually be needed
            -----
            OutageMdlrPrep.perform_build_RCPX_from_csvs_prereqs all but ensures a merge with MeterPremise will include, at minimum,
              premise number and serial number.
            Therefore, this is rarely needed.
        -------------------------

        Find faulty Meter Premise values in end_events_df and correct them.  See note below regarding initial assumption for why
          these values are faulty.
        The entries are determined to be faulty if the premise number from the meter_events.end_device_event database (contained
          in prem_nb_col_ede) does not match that from default.meter_premise (contained in prem_nb_col_mp).
        The faulty entries are corrected using the full power of default.meter_premise and default.meter_premise_hist combined,
          allowing the correct meter at the time of the event to be located.

        This is basically a blending of OutageMdlrPrep.set_faulty_mp_vals_to_nan_in_end_events_df and 
        OutageMdlrPrep.correct_or_add_active_mp_cols

        Initially assumed that joining meter_events.end_device_event with default.meter_premise soley on serial number was fine.
        However, have since found that the serial number in meter_events.end_device_event does not always appear to be correct.
        This leads to a situation where the wrong MP is matched, which throws off the trsf_pole_nb (and any other attributed
        taken from default.meter_premise).
        In the future, the two will be merged on both serial number and premise number.  The join will be left, so any entries
        which don't match simply will have NaN for the MP cols.
        The purpose of this function is to reproduce this effect for the erroneous data.
        """
        #-------------------------
        # First, identify the faulty entries and grab the subset
        #-----
        # NOTE: Also want to fix entries where prem_nb is NaN, hence why '& (end_events_df[prem_nb_col_mp].notna())' is absent here
        #       as compared to faulty_mask in set_faulty_mp_vals_to_nan_in_end_events_df
        faulty_mask = (end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp])
        if faulty_mask.sum()==0:
            return end_events_df
        flty_end_events_df = end_events_df[faulty_mask]
        #-------------------------
        # Reduce end_events_df to only non-faulty entries
        end_events_df = end_events_df[~faulty_mask]
        # Should no longer be any faulty idxs in end_events_df
        assert((end_events_df[prem_nb_col_ede] != end_events_df[prem_nb_col_mp]).sum()==0)
        #-------------------------
        # Correct the faulty entries
        flty_end_events_df = OutageMdlrPrep.correct_or_add_active_mp_cols(
            end_events_df=flty_end_events_df, 
            df_time_col_0=df_time_col_0,
            df_time_col_1=df_time_col_1,
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist, 
            df_and_mp_merge_pairs=df_and_mp_merge_pairs, 
            assert_all_PNs_found=assert_all_PNs_found, 
            df_prem_nb_col=prem_nb_col_ede, 
            df_mp_prem_nb_col=prem_nb_col_mp
        )
        #-------------------------
        # Now, add the corrected flty_end_events_df back into end_events_df
        #-----
        # I suppose it's fine for flty_end_events_df to have additional columns, but it MUST contain
        #   at least those in end_events_df
        assert(len(set(end_events_df.columns).difference(set(flty_end_events_df.columns)))==0)
        flty_end_events_df = flty_end_events_df[end_events_df.columns]
        #-----
        end_events_df = pd.concat([end_events_df, flty_end_events_df])
        #-------------------------
        return end_events_df


    #---------------------------------------------------------------------------
    @staticmethod
    def build_reason_counts_per_outage_from_csvs(    
        files_dir                      , 
        mp_df                          , 
        file_path_glob                 = r'end_events_[0-9]*.csv', 
        file_path_regex                = None, 
        min_outg_td_window             = datetime.timedelta(days=1),
        max_outg_td_window             = datetime.timedelta(days=30),
        build_ede_typeid_to_reason_df  = False, 
        batch_size                     = 50, 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        assert_all_cols_equal          = True, 
        include_normalize_by_nSNs      = True, 
        inclue_zero_counts             = True, 
        return_multiindex_outg_reason  = False, 
        return_normalized_separately   = False, 
        verbose                        = True, 
        n_update                       = 1, 
        grp_by_cols                    = 'outg_rec_nb', 
        outg_rec_nb_col                = 'outg_rec_nb',
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        addtnl_dropna_subset_cols      = None, 
        is_no_outage                   = False, 
        prem_nb_col                    = 'aep_premise_nb', 
        serial_number_col              = 'serialnumber', 
        include_prem_nbs               = False, 
        set_faulty_mp_vals_to_nan      = False,
        correct_faulty_mp_vals         = False, 
        trust_sql_grouping             = True, 
        drop_gpd_for_sql_appendix      = True, 
        mp_df_cols                     = dict(
            serial_number_col = 'mfr_devc_ser_nbr', 
            prem_nb_col       = 'prem_nb', 
            trsf_pole_nb_col  = 'trsf_pole_nb', 
            outg_rec_nb_col   = 'OUTG_REC_NB'
        ), 
        make_all_columns_lowercase     = True, 
        date_only                      = False
    ):
        r"""
        Note: The larger the batch_size, the more memory that will be consumed during building
    
        Any rows with NaNs in outg_rec_nb_col+addtnl_dropna_subset_cols will be dropped
    
        #NOTE: Currently, only set up for the case return_normalized_separately==False

        mp_df:
            May be None.

        set_faulty_mp_vals_to_nan & correct_faulty_mp_vals:
            NOT REALLY NEEDED ANYMORE!
                I suggest keeping both of these False.
                It's fine to set True, but run time will likely be negatively affected (especially if many faulty MP values found
                  which need to be corrected)
                -----
                As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
                If this was not true, then this function may actually be needed
                -----
                OutageMdlrPrep.perform_build_RCPX_from_csvs_prereqs all but ensures a merge with MeterPremise will include, at minimum,
                premise number and serial number.
                Therefore, this is rarely needed.
        """
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO Allow method for reading DOVS from CSV (instead of only from SQL query)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #-------------------------
        assert(not return_normalized_separately)
        #-------------------------
        assert(Utilities.is_object_one_of_types(grp_by_cols, [str, list, tuple]))
        if isinstance(grp_by_cols, str):
            grp_by_cols = [grp_by_cols]        
        #-------------------------
        # If including prem numbers, also include n_prem numbers
        include_nprem_nbs = include_prem_nbs
        #-------------------------
        # normalize_by_nSNs_included needed when return_multiindex_outg_reason is True
        if include_normalize_by_nSNs and not return_normalized_separately:
            normalize_by_nSNs_included = True
        else:
            normalize_by_nSNs_included = False
    
        are_dfs_wide_form = not return_multiindex_outg_reason
        is_norm=False #TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #-------------------------
        prereq_dict = OutageMdlrPrep.perform_build_RCPX_from_csvs_prereqs(
            files_dir                     = files_dir, 
            file_path_glob                = file_path_glob, 
            file_path_regex               = file_path_regex,
            mp_df                         = mp_df, 
            ede_mp_mismatch_threshold_pct = 1.0, 
            grp_by_cols                   = grp_by_cols, 
            outg_rec_nb_col               = outg_rec_nb_col,
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            prem_nb_col                   = prem_nb_col, 
            serial_number_col             = serial_number_col,
            mp_df_cols                    = mp_df_cols, 
            is_no_outage                  = is_no_outage, 
            assert_all_cols_equal         = assert_all_cols_equal, 
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase    = make_all_columns_lowercase
        )
        paths             = prereq_dict['paths']
        grp_by_cols       = prereq_dict['grp_by_cols']
        outg_rec_nb_col   = prereq_dict['outg_rec_nb_col']
        trsf_pole_nb_col  = prereq_dict['trsf_pole_nb_col']
        prem_nb_col       = prereq_dict['prem_nb_col']
        serial_number_col = prereq_dict['serial_number_col']
        cols_to_drop      = prereq_dict['cols_to_drop']
        rename_cols_dict  = prereq_dict['rename_cols_dict']
        mp_df             = prereq_dict['mp_df']
        merge_on_ede      = prereq_dict['merge_on_ede']
        merge_on_mp       = prereq_dict['merge_on_mp']
        #-------------------------
        rcpo_full = pd.DataFrame()
        #-------------------------
        batch_idxs = Utilities.get_batch_idx_pairs(len(paths), batch_size)
        n_batches = len(batch_idxs)    
        if verbose:
            print(f'n_paths    = {len(paths)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches  = {n_batches}')    
        #-------------------------
        for i, batch_i in enumerate(batch_idxs):
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-----
            end_events_df_i = GenAn.read_df_from_csv_batch(
                paths                          = paths[i_beg:i_end], 
                cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
                to_numeric_errors              = to_numeric_errors, 
                make_all_columns_lowercase     = make_all_columns_lowercase, 
                assert_all_cols_equal          = assert_all_cols_equal
            )
            #-------------------------
            if end_events_df_i.shape[0]==0:
                continue
            #-------------------------
            end_events_df_i = OutageMdlrPrep.perform_std_col_renames_and_drops(
                end_events_df              = end_events_df_i, 
                cols_to_drop               = cols_to_drop, 
                rename_cols_dict           = rename_cols_dict, 
                make_all_columns_lowercase = make_all_columns_lowercase
            )
            #-------------------------
            # FROM WHAT I CAN TELL, the meters which are 'missing' from meter premise but are present in end events
            #   are caused by meters which were removed or installed close to (a few days/weeks before) an outage.
            # Therefore, although the meter may not have been present at the time of the outage (and therefore was exluced from
            #   meter premise), it could have registered events leading up to/following the outage.
            # e.g., if a meter was removed in the days before an outage, end events are still found for this meter in the days leading up to the outage
            # e.g., if a meter was installed in the days after an outage, end events are still found for this meter in the days following an outage.
            # How should these be handled?
            # The simplest method, which I will implement for now, is to simply ONLY consider those meters which were present
            #   at the time of the outage.  THEREFORE, the two DFs should be joined with an inner merge!
            if mp_df is not None:
                end_events_df_i = AMIEndEvents.merge_end_events_df_with_mp(
                    end_events_df=end_events_df_i, 
                    df_mp              = mp_df, 
                    merge_on_ede       = merge_on_ede, 
                    merge_on_mp        = merge_on_mp, 
                    cols_to_include_mp = None, 
                    drop_cols          = None, 
                    rename_cols        = None, 
                    how                = 'inner', 
                    inplace            = True
                )
            #-------------------------
            for grp_by_col in grp_by_cols:
                assert(grp_by_col in end_events_df_i.columns)
            #-----
            if not is_no_outage:
                assert(outg_rec_nb_col in end_events_df_i.columns)
            #-------------------------
            dropna_subset_cols = grp_by_cols
            if addtnl_dropna_subset_cols is not None:
                dropna_subset_cols.extend(addtnl_dropna_subset_cols)
            end_events_df_i = end_events_df_i.dropna(subset=dropna_subset_cols)
            #-------------------------
            if set_faulty_mp_vals_to_nan and mp_df is not None:
                end_events_df_i = OutageMdlrPrep.set_faulty_mp_vals_to_nan_in_end_events_df(
                    end_events_df   = end_events_df_i, 
                    prem_nb_col_ede = prem_nb_col, 
                    prem_nb_col_mp  = mp_df_cols['prem_nb_col']
                )
            #-------------------------
            end_events_df_i=Utilities_dt.strip_tz_info_and_convert_to_dt(
                df            = end_events_df_i, 
                time_col      = 'valuesinterval', 
                placement_col = 'valuesinterval_local', 
                run_quick     = True, 
                n_strip       = 6, 
                inplace       = False
            )
            #---------------------------------------------------------------------------
            # If min_outg_td_window or max_outg_td_window is not None, enforce time restrictions around outages
            if min_outg_td_window is not None or max_outg_td_window is not None:
                #----------
                if not is_no_outage:
                    dovs_outgs = DOVSOutages(                 
                        df_construct_type         = DFConstructType.kRunSqlQuery, 
                        contstruct_df_args        = None, 
                        init_df_in_constructor    = True, 
                        build_sql_function        = DOVSOutages_SQL.build_sql_outage, 
                        build_sql_function_kwargs = dict(
                            outg_rec_nbs     = end_events_df_i[outg_rec_nb_col].unique().tolist(), 
                            from_table_alias = 'DOV', 
                            datetime_col     = 'DT_OFF_TS_FULL', 
                            cols_of_interest = [
                                'OUTG_REC_NB', 
                                dict(
                                    field_desc         = f"DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)", 
                                    alias              = 'DT_OFF_TS_FULL', 
                                    table_alias_prefix = None
                                )
                            ], 
                            field_to_split   = 'outg_rec_nbs'
                        ),
                    )
                    outg_dt_off_df = dovs_outgs.df
                    #-----
                    outg_dt_off_df = Utilities_df.convert_col_type(
                        df      = outg_dt_off_df, 
                        column  = 'OUTG_REC_NB', 
                        to_type = str
                    ).set_index('OUTG_REC_NB')
                    #-----
                    outg_dt_off_series = outg_dt_off_df['DT_OFF_TS_FULL']
                    #-----
                    if date_only:
                        outg_dt_off_series = pd.to_datetime(outg_dt_off_series.dt.strftime('%Y-%m-%d'))
                        end_events_df_i['valuesinterval_local'] = pd.to_datetime(end_events_df_i['valuesinterval_local'].dt.strftime('%Y-%m-%d'))
                    #-----
                    end_events_df_i = AMIEndEvents.enforce_end_events_within_interval_of_outage(
                        end_events_df             = end_events_df_i, 
                        outg_times_series         = outg_dt_off_series, 
                        min_timedelta             = min_outg_td_window, 
                        max_timedelta             = max_outg_td_window, 
                        outg_rec_nb_col           = outg_rec_nb_col, 
                        datetime_col              = 'valuesinterval_local', 
                        assert_one_time_per_group = True
                    )
                else:
                    no_outg_time_infos_df = MECPOAn.get_bsln_time_interval_infos_df_from_summary_files(
                        summary_paths           = [AMIEndEvents.find_summary_file_from_csv(x) for x in paths[i_beg:i_end]], 
                        output_prem_nbs_col     = 'prem_nbs', 
                        output_t_min_col        = 't_min', 
                        output_t_max_col        = 'DT_OFF_TS_FULL', 
                        make_addtnl_groupby_idx = True, 
                        include_summary_paths   = False
                    )
                    #-----
                    if no_outg_time_infos_df.index.nlevels==1:
                        assert(no_outg_time_infos_df.index.name==trsf_pole_nb_col)
                        assert(end_events_df_i[trsf_pole_nb_col].dtype==no_outg_time_infos_df.index.dtype)
                    else:
                        assert(trsf_pole_nb_col in no_outg_time_infos_df.index.names)
                        assert(end_events_df_i[trsf_pole_nb_col].dtype==no_outg_time_infos_df.index.dtypes[trsf_pole_nb_col])
                    #-----
                    no_outg_time_infos_series = no_outg_time_infos_df['DT_OFF_TS_FULL']    
                    #-----
                    if date_only:
                        no_outg_time_infos_series = pd.to_datetime(no_outg_time_infos_series.dt.strftime('%Y-%m-%d'))
                        end_events_df_i['valuesinterval_local'] = pd.to_datetime(end_events_df_i['valuesinterval_local'].dt.strftime('%Y-%m-%d'))
                    #-----    
                    end_events_df_i = AMIEndEvents.enforce_end_events_within_interval_of_outage(
                        end_events_df             = end_events_df_i, 
                        outg_times_series         = no_outg_time_infos_series, 
                        min_timedelta             = min_outg_td_window, 
                        max_timedelta             = max_outg_td_window, 
                        outg_rec_nb_col           = trsf_pole_nb_col, 
                        datetime_col              = 'valuesinterval_local', 
                        assert_one_time_per_group = False
                    )
            #---------------------------------------------------------------------------
            # After enforcing events within specific time frame, it is possible end_events_df_i is empty.
            # If so, continue
            if end_events_df_i.shape[0]==0:
                continue
            #-------------------------
            if correct_faulty_mp_vals and mp_df is not None:
                end_events_df_i = OutageMdlrPrep.correct_faulty_mp_vals_in_end_events_df(
                    end_events_df         = end_events_df_i, 
                    df_time_col_0         = 'valuesinterval_local',
                    df_time_col_1         = None,
                    df_mp_curr            = None, 
                    df_mp_hist            = None, 
                    prem_nb_col_ede       = prem_nb_col, 
                    prem_nb_col_mp        = mp_df_cols['prem_nb_col'], 
                    df_and_mp_merge_pairs = [
                        [serial_number_col, mp_df_cols['serial_number_col']], 
                        [prem_nb_col,       mp_df_cols['prem_nb_col']]
                    ], 
                    assert_all_PNs_found  = False
                )
            #-------------------------
    #         end_events_df_i = AMIEndEvents.extract_reboot_counts_from_reasons_in_df(end_events_df_i)
    #         end_events_df_i = AMIEndEvents.extract_fail_reasons_from_reasons_in_df(end_events_df_i)
            #-------------------------
            end_events_df_i = AMIEndEvents.reduce_end_event_reasons_in_df(
                df                            = end_events_df_i, 
                reason_col                    = 'reason', 
                edetypeid_col                 = 'enddeviceeventtypeid', 
                patterns_to_replace_by_typeid = None, 
                addtnl_patterns_to_replace    = None, 
                placement_col                 = None, 
                count                         = 0, 
                flags                         = re.IGNORECASE,  
                inplace                       = True
            )
            #-------------------------
            if build_ede_typeid_to_reason_df:
                ede_typeid_to_reason_df_i = AMIEndEvents.build_ede_typeid_to_reason_df(
                    end_events_df  = end_events_df_i, 
                    reason_col     = 'reason', 
                    ede_typeid_col = 'enddeviceeventtypeid'
                )
                if i==0:
                    ede_typeid_to_reason_df = ede_typeid_to_reason_df_i.copy()
                else:
                    ede_typeid_to_reason_df = AMIEndEvents.combine_two_ede_typeid_to_reason_dfs(
                        ede_typeid_to_reason_df1 = ede_typeid_to_reason_df, 
                        ede_typeid_to_reason_df2 = ede_typeid_to_reason_df_i,
                        sort                     = True
                    )
            #-------------------------
            rcpo_i = AMIEndEvents.get_reason_counts_per_group(
                end_events_df             = end_events_df_i, 
                group_cols                = grp_by_cols, 
                group_freq                = None, 
                serial_number_col         = serial_number_col, 
                reason_col                = 'reason', 
                include_normalize_by_nSNs = include_normalize_by_nSNs, 
                inclue_zero_counts        = inclue_zero_counts,
                possible_reasons          = None, 
                include_nSNs              = True, 
                include_SNs               = True, 
                prem_nb_col               = prem_nb_col, 
                include_nprem_nbs         = include_nprem_nbs,
                include_prem_nbs          = include_prem_nbs,   
                return_form = dict(
                    return_multiindex_outg_reason = return_multiindex_outg_reason, 
                    return_normalized_separately  = return_normalized_separately
                )
            )
            #-------------------------
            if rcpo_i.shape[0]==0:
                continue
            #-------------------------
            # Include or below in case i=0 rcpo_i comes back empty, causing continue to be called above
            if i==0 or rcpo_full.shape[0]==0:
                rcpo_full = rcpo_i.copy()
            else:
                list_cols=['_SNs']
                list_counts_cols=['_nSNs']
                w_col = '_nSNs'
                if include_prem_nbs:
                    list_cols.append('_prem_nbs')
                    list_counts_cols.append('_nprem_nbs')
                #-----
                rcpo_full = AMIEndEvents.combine_two_reason_counts_per_outage_dfs(
                    rcpo_df_1                          = rcpo_full, 
                    rcpo_df_2                          = rcpo_i, 
                    are_dfs_wide_form                  = are_dfs_wide_form, 
                    normalize_by_nSNs_included         = normalize_by_nSNs_included, 
                    is_norm                            = is_norm, 
                    list_cols                          = list_cols, 
                    list_counts_cols                   = list_counts_cols,
                    w_col                              = w_col, 
                    level_0_raw_col                    = 'counts', 
                    level_0_nrm_col                    = 'counts_norm', 
                    convert_rcpo_wide_to_long_col_args = None
                )
        #-------------------------
        # Drop _gpd_for_sql appendix from any index names 
        rcpo_full = OutageMdlrPrep.drop_gpd_for_sql_appendix_from_all_index_names(rcpo_full)
        #-------------------------
        if not build_ede_typeid_to_reason_df:
            return rcpo_full
        else:
            return rcpo_full, ede_typeid_to_reason_df


    #------------------------------------------------------------------------------------------------------------------------------------------------------
    # data_evs_sum_vw = True methods!
    #---------------------------------------------------------------------------
    @staticmethod    
    def build_regex_setup(
        conn_aws = None
    ):
        r"""
        Builds regex_setup_df and cr_trans_dict, where cr_trans_dict = curated reasons translation dictionary
        """
        #-------------------------
        if conn_aws is None:
            conn_aws = Utilities.get_athena_prod_aws_connection()
        #-------------------------
        sql = """
        SELECT * FROM meter_events.event_summ_regex_setup
        """
        regex_setup_df = pd.read_sql(sql, conn_aws, dtype=str)
        cr_trans_dict = {x[0]:x[1] for x in regex_setup_df[['pivot_id', 'regex_report_title']].values.tolist()}
        return regex_setup_df, cr_trans_dict

    #---------------------------------------------------------------------------
    @staticmethod    
    def get_regex_setup(
        conn_aws = None
    ):
        r"""
        Returns regex_setup_df and cr_trans_dict, where cr_trans_dict = curated reasons translation dictionary
        """
        #-------------------------
        return OutageMdlrPrep.build_regex_setup(conn_aws = conn_aws)
    

    #---------------------------------------------------------------------------
    @staticmethod
    def get_cr_trans_dict(
        conn_aws = None, 
        to       = 'reason'
    ):
        r"""
        """
        #-------------------------
        assert(to in ['reason', 'typeid'])
        #-------------------------
        regex_setup_df, cr_trans_dict = OutageMdlrPrep.get_regex_setup(conn_aws = conn_aws)
        # if to == 'reason', cr_trans_dict is fine as is.
        # Otherwise, need to build
        if to != 'reason':
            cr_trans_dict = {x[0]:x[1] for x in regex_setup_df[['pivot_id', 'enddeviceeventtypeid']].values.tolist()}
        #-------------------------
        return cr_trans_dict
    

    #---------------------------------------------------------------------------
    @staticmethod
    def assert_timedelta_is_days(td):
        r"""
        The analysis typically expects the frequency to be in days.
        This function checks the attributes of td to ensure this is true
        """
        assert(isinstance(td, pd.Timedelta))
        td_comps = td.components
        #-----
        assert(td_comps.days>=0)
        #-----
        assert(td_comps.hours==0)
        assert(td_comps.minutes==0)
        assert(td_comps.seconds==0)
        assert(td_comps.milliseconds==0)
        assert(td_comps.microseconds==0)
        assert(td_comps.nanoseconds==0)

    #---------------------------------------------------------------------------
    @staticmethod
    def get_time_pds_rename(
        curr_time_pds    , 
        td_min           = pd.Timedelta('1D'), 
        td_max           = pd.Timedelta('31D'), 
        freq             = pd.Timedelta('5D'), 
        return_time_grps = False
    ):
        r"""
        Need to convert the time periods, which are currently housed in the date_col index of 
          rcpx_0 from their specific dates to the names expected by the model.
        In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
          dates of the given interval.
        These will be converted to the titles contained in final_time_pds below

        curr_time_pds:
            When originally developed, intended to be a list.
            However, now can also be a list of tuples/lists (each element of length 2)

            curr_time_pds as list:
                e.g.:
                   curr_time_pds = [Timestamp('2023-05-02'), Timestamp('2023-05-07'), 
                                    Timestamp('2023-05-12'), Timestamp('2023-05-17'), 
                                    Timestamp('2023-05-22'), Timestamp('2023-05-27')]
                ==> final_time_pds = ['01-06 Days', '06-11 Days',
                                     '11-16 Days', '16-21 Days',
                                     '21-26 Days', '26-31 Days']
                
                Returns: A dict object with keys equal to curr_time_pds and values equal to final_time_pds

            curr_time_pds as list of tuples/lists (each element of length 2):
                e.g.:
                   curr_time_pds = [(Timestamp('2023-05-02'), Timestamp('2023-05-07')), 
                                    (Timestamp('2023-05-07'), Timestamp('2023-05-12')), 
                                    (Timestamp('2023-05-12'), Timestamp('2023-05-17')), 
                                    (Timestamp('2023-05-17'), Timestamp('2023-05-22')), 
                                    (Timestamp('2023-05-22'), Timestamp('2023-05-27')), 
                                    (Timestamp('2023-05-27'), Timestamp('2023-06-01'))]
                ==> final_time_pds = ['01-06 Days', '06-11 Days',
                                     '11-16 Days', '16-21 Days',
                                     '21-26 Days', '26-31 Days']
                
                Returns: A dict object with keys equal to [x[0] for x in curr_time_pds] and values equal to final_time_pds
         
        !!! NOTE !!!: This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
        """
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        freq   = pd.Timedelta(freq)
        #-------------------------
        OutageMdlrPrep.assert_timedelta_is_days(td_min)
        OutageMdlrPrep.assert_timedelta_is_days(td_max)
        OutageMdlrPrep.assert_timedelta_is_days(freq)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        days_freq = freq.days
        #-----
        # Make sure (days_max-days_min) evenly divisible by days_freq
        assert((days_max-days_min) % days_freq==0)
        #----------------------------------------------------------------------------------------------------
        if Utilities.is_object_one_of_types(curr_time_pds[0], [tuple, list]):
            assert(Utilities.are_all_list_elements_one_of_types(curr_time_pds, [tuple, list]))
            assert(Utilities.are_list_elements_lengths_homogeneous(curr_time_pds, length=2))
            for time_grp_i in curr_time_pds:
                assert(time_grp_i[1]-time_grp_i[0]==freq)
            #-------------------------
            curr_time_pds_0 = [x[0] for x in natsorted(curr_time_pds, key=lambda x:x[0])]
            #-------------------------
            return OutageMdlrPrep.get_time_pds_rename(
                curr_time_pds = curr_time_pds_0, 
                td_min        = td_min, 
                td_max        = td_max, 
                freq          = freq
            )
        #----------------------------------------------------------------------------------------------------
        curr_time_pds = natsorted(curr_time_pds)
        time_grps     = []
        # Each time period should have a width equal to freq
        for i in range(len(curr_time_pds)):
            if i==0:
                continue
            assert(curr_time_pds[i]-curr_time_pds[i-1]==freq)
            time_grps.append((curr_time_pds[i-1], curr_time_pds[i]))
        time_grps.append((time_grps[-1][-1], time_grps[-1][-1]+freq))
        #-------------------------
        n_pds_needed    = len(curr_time_pds)
        days_max_needed = days_min + days_freq*n_pds_needed
        if days_max < days_max_needed:
            days_max = days_max_needed
        #-------------------------
        final_time_pds_0 = np.arange(start=days_min, stop=days_max+1, step=days_freq).tolist()
        assert(len(final_time_pds_0) > len(curr_time_pds)) # len(final_time_pds_0) should equal len(curr_time_pds)+1
        final_time_pds_0 = final_time_pds_0[:len(curr_time_pds)+1]
        #----------
        # Use padded zeros
        #   i.e., instead of '1-6 Days', use '01-06 Days'
        #   Will need to know longest string length for this
        str_len = np.max([len(str(x)) for x in final_time_pds_0])
        # Looks like a ton of curly braces below, but need double {{ and double }} to escape
        # ==> if str_len=2, then str_fmt = '{:02d}'
        str_fmt = f'{{:0{str_len}d}}'
        #----------
        final_time_pds = []
        for i in range(len(final_time_pds_0)):
            if i==0:
                continue
            fnl_pd_im1 = "{}-{} Days".format(
                str_fmt.format(final_time_pds_0[i-1]), 
                str_fmt.format(final_time_pds_0[i])
            )
            final_time_pds.append(fnl_pd_im1)
        #-------------------------
        assert(len(final_time_pds)==len(curr_time_pds))
        #-------------------------
        # final_time_pds is arranged from closest to prediction date to furthest
        #   e.g., final_time_pds = ['01-06 Days', '06-11 Days', ..., '26-31 Days']
        # curr_time_pds is arranged from furthesto to closest
        #   e.g., curr_time_pds = [2023-06-26, 2023-07-01, ..., 2023-07-21]
        # To make the two agree, re-sort curr_time_pds with reverse=True
        curr_time_pds = natsorted(curr_time_pds, reverse=True)
        time_grps     = natsorted(time_grps, reverse=True)
        #-------------------------
        time_pds_rename = dict(zip(curr_time_pds, final_time_pds))
        #-------------------------
        if return_time_grps:
            return time_pds_rename, time_grps
        return time_pds_rename

    #---------------------------------------------------------------------------
    @staticmethod
    def find_cr_cols(
        df
    ):
        r"""
        """
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        return cr_cols
    
    #---------------------------------------------------------------------------
    @staticmethod
    def get_evsSum_df_std_dtypes_dict(
        df=None
    ):
        r"""
        If df not included, return only non_cr_cols
        If df included:
            Look for cr_cols in df, and return in addition to non_cr_cols
        """
        #-------------------------
        non_cr_cols = {
            'serialnumber'                   : str,
            'aep_premise_nb'                 : str,
            'trsf_pole_nb'                   : str,
            'xf_meter_cnt'                   : np.int64,
            'events_tot'                     : np.int64,
            'aep_opco'                       : str,
            'aep_event_dt'                   : datetime.datetime,
            #----------
            'trsf_pole_nb_GPD_FOR_SQL'       : str,
            'outg_rec_nb_GPD_FOR_SQL'        : str,
            'no_outg_rec_nb_GPD_FOR_SQL'     : str,
            'is_first_after_outg_GPD_FOR_SQL': bool,  
            #-----
            'trsf_pole_nb'                   : str,
            'outg_rec_nb'                    : str,
            'no_outg_rec_nb'                 : str,
            'is_first_after_outg'            : bool,  
        }
        #-------------------------
        if df is None:
            return non_cr_cols
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        cr_cols = {x:np.int64 for x in cr_cols}
        #-------------------------
        return non_cr_cols|cr_cols
    
    #---------------------------------------------------------------------------
    @staticmethod
    def perform_std_col_type_conversions(
        evsSum_df
    ):
        r"""
        """
        #-------------------------
        dtypes_dict = OutageMdlrPrep.get_evsSum_df_std_dtypes_dict(df=evsSum_df)
        dtypes_dict = {k:v for k,v in dtypes_dict.items() if k in evsSum_df.columns}
        #-----
        if len(dtypes_dict) > 0:
            evsSum_df = Utilities_df.convert_col_types(
                df                  = evsSum_df,
                cols_and_types_dict = dtypes_dict,
                to_numeric_errors   = 'coerce',
                inplace             = True,
            )
        #-------------------------
        return evsSum_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def append_to_evsSum_df(
        evsSum_df         , 
        addtnl_evsSum_df  , 
        sort_by           = None, 
        make_col_types_eq = False
    ):
        r"""
        addtnl_evsSum_df:
            Expected to be a pd.DataFrame object, but can also be a list of such
            IF A LIST:
                All elements must be pd.DataFrame objects
                append_to_evsSum_df will be called iteratively with each element
        """
        #--------------------------------------------------
        if isinstance(addtnl_evsSum_df, list):
            assert(Utilities.are_all_list_elements_of_type(lst=addtnl_evsSum_df, typ=pd.DataFrame))
            for addtnl_evsSum_df_i in addtnl_evsSum_df:
                evsSum_df = OutageMdlrPrep.append_to_evsSum_df(
                    evsSum_df        = evsSum_df, 
                    addtnl_evsSum_df = addtnl_evsSum_df_i, 
                    sort_by          = sort_by
                )
            return evsSum_df
        #--------------------------------------------------
        if addtnl_evsSum_df is None or addtnl_evsSum_df.shape[0]==0:
            return evsSum_df
        #-------------------------
        if evsSum_df is None or evsSum_df.shape[0]==0:
            assert(addtnl_evsSum_df.shape[0]>0)
            return addtnl_evsSum_df
        #-------------------------
        assert(evsSum_df.columns.equals(addtnl_evsSum_df.columns))
        #-------------------------
        if make_col_types_eq:
            evsSum_df, addtnl_evsSum_df = Utilities_df.make_df_col_dtypes_equal(
                df_1              = evsSum_df, 
                col_1             = evsSum_df.columns.tolist(), 
                df_2              = addtnl_evsSum_df, 
                col_2             = addtnl_evsSum_df.columns.tolist(), 
                allow_reverse_set = True, 
                assert_success    = True, 
                inplace           = True
            )

        #-------------------------
        # Make sure evsSum_df and addtnl_evsSum_df share the same columns and 
        #   do not have any overlapping data
        assert(
            not Utilities_df.do_dfs_overlap(
                df_1            = evsSum_df, 
                df_2            = addtnl_evsSum_df, 
                enforce_eq_cols = True, 
                include_index   = True
            )
        )

        #-------------------------
        evsSum_df = pd.concat([evsSum_df, addtnl_evsSum_df])
        #-------------------------
        if sort_by is not None:
            evsSum_df = evsSum_df.sort_values(by=sort_by)
        #-------------------------
        return evsSum_df
    
    #---------------------------------------------------------------------------
    @staticmethod
    def concat_evsSum_dfs(
        evsSum_dfs
    ):
        r"""
        """
        #-------------------------
        assert(Utilities.are_all_list_elements_of_type(lst=evsSum_dfs, typ=pd.DataFrame))
        #-------------------------
        if len(evsSum_dfs)==1:
            return evsSum_dfs[0]
        #-------------------------
        evsSum_df         = evsSum_dfs[0]
        addtnl_evsSum_dfs = evsSum_dfs[1:]
        sort_by           = None
        #-----
        evsSum_df = OutageMdlrPrep.append_to_evsSum_df(
            evsSum_df        = evsSum_df, 
            addtnl_evsSum_df = addtnl_evsSum_dfs, 
            sort_by          = sort_by
        )
        #-------------------------
        return evsSum_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_evsSum_df_from_csvs(
        files_dir                      , 
        file_path_glob                 = r'events_summary_[0-9]*.csv', 
        file_path_regex                = None, 
        batch_size                     = 50, 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        make_all_columns_lowercase     = True, 
        assert_all_cols_equal          = True,  
        n_update                       = 1, 
        verbose                        = True
    ):
        r"""
        """
        #-------------------------
        paths = Utilities.find_all_paths(
            base_dir      = files_dir, 
            glob_pattern  = file_path_glob, 
            regex_pattern = file_path_regex
        )
        if len(paths)==0:
            print(f'No paths found in files_dir = {files_dir}')
            return None
        paths=natsorted(paths)
        #-------------------------
        batch_idxs = Utilities.get_batch_idx_pairs(len(paths), batch_size)
        n_batches = len(batch_idxs)    
        if verbose:
            print(f'n_paths    = {len(paths)}')
            print(f'batch_size = {batch_size}')
            print(f'n_batches  = {n_batches}')    
        #-------------------------
        evsSum_dfs = []
        for i, batch_i in enumerate(batch_idxs):
            start = time.time()
            if verbose and (i+1)%n_update==0:
                print(f'{i+1}/{n_batches}')
            i_beg = batch_i[0]
            i_end = batch_i[1]
            #-----
            evsSum_df_i = GenAn.read_df_from_csv_batch(
                paths                          = paths[i_beg:i_end], 
                cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
                to_numeric_errors              = to_numeric_errors, 
                make_all_columns_lowercase     = make_all_columns_lowercase, 
                assert_all_cols_equal          = assert_all_cols_equal
            )
            #-------------------------
            if evsSum_df_i.shape[0]==0:
                continue
            evsSum_dfs.append(evsSum_df_i)
        #-------------------------
        evsSum_df_fnl = OutageMdlrPrep.concat_evsSum_dfs(evsSum_dfs = evsSum_dfs)
        evsSum_df_fnl = OutageMdlrPrep.perform_std_col_type_conversions(evsSum_df = evsSum_df_fnl)
        return evsSum_df_fnl

    
    #---------------------------------------------------------------------------
    @staticmethod
    def convert_cr_cols_to_reasons(
        rcpx_df                  , 
        cr_trans_dict            = None, 
        regex_patterns_to_remove = ['.*cleared.*', '.*Test Mode.*'], 
        total_counts_col         = 'total_counts', 
    ):
        r"""
        Rename the cr# columns to their full curated reasons.
        Combine any degenerates
        """
        #--------------------------------------------------
        if cr_trans_dict is None:
            _, cr_trans_dict = OutageMdlrPrep.get_regex_setup()
        #-------------------------
        rcpx_df = rcpx_df.rename(columns=cr_trans_dict)
        #--------------------------------------------------
        # Any columns without a curated reason (i.e., those with column name = ''), have not been observed
        #   yet in the data, and therefore the sume of the counts should be 0.
        # These empty columns are not needed, so drop
        if '' in rcpx_df.columns.tolist():
            assert(rcpx_df[''].sum().sum()==0)
            rcpx_df = rcpx_df.drop(columns=[''])
        #--------------------------------------------------
        # Remove any undesired curated reasons (e.g., ['.*cleared.*', '.*Test Mode.*'])
        if regex_patterns_to_remove is not None:
            rcpx_df = MECPODf.remove_reasons_from_rcpo_df(
                rcpo_df=rcpx_df, 
                regex_patterns_to_remove=regex_patterns_to_remove, 
                ignore_case=True
            )
        #--------------------------------------------------
        # Combine any degenerate columns
        rcpx_df = MECPODf.combine_degenerate_columns(rcpo_df = rcpx_df)
        #--------------------------------------------------
        # After irrelevant cleared and test columns removed, need to (re)calculate events_tot to accurately
        #   reflect the total number of relevant events
        # Safe(r) to do this calculation in any case, so moved outside of the if block above
        if total_counts_col is not None:
            reason_cols = list(set(rcpx_df.columns).intersection(set(cr_trans_dict.values())))
            non_reason_cols = list(set(rcpx_df.columns).difference(set(cr_trans_dict.values())))
            #-----
            rcpx_df[total_counts_col] = rcpx_df[reason_cols].sum(axis=1)
        #--------------------------------------------------
        return rcpx_df


    #---------------------------------------------------------------------------
    @staticmethod
    def project_time_pd_from_rcpx_0_and_prepare(
        rcpx_0                      , 
        time_pd_i                   , 
        meter_cnt_per_gp_srs        , 
        all_groups                  ,  
        non_reason_cols             , 
        data_structure_df           = None, 
        other_reasons_col           = None, 
        cr_trans_dict               = None, 
        group_cols                  = ['trsf_pole_nb'], 
        time_pd_grp_idx             = 'aep_event_dt', 
        time_pd_i_rename            = None, 
        regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons      = True, 
        include_power_down_minus_up = False, 
        total_counts_col            = 'total_counts', 
        nSNs_col                    = 'nSNs', 
    ):
        r"""
        data_structure_df:
            If supplied, it will be used to determine the set of final columns desired in rcpx_0_pd_i, and the method
              MECPODf.get_reasons_subset_from_cpo_df will be used to adjust rcpx_0_pd_i accordingly.
            If not supplied, MECPODf.get_reasons_subset_from_cpo_df is not calleds

        cr_trans_dict:
            If not supplied, will be grabbed via the OutageMdlrPrep.get_regex_setup() method
        """
        #--------------------------------------------------
        if time_pd_i_rename is None:
            time_pd_i_rename = time_pd_i
        #--------------------------------------------------
        # Project out the current time period (time_pd_i) from rcpx_0 by selecting the appropriate
        #   values from the time_pd_grp_idx index
        rcpx_0_pd_i = rcpx_0[rcpx_0.index.get_level_values(time_pd_grp_idx)==time_pd_i].copy()
        rcpx_0_pd_i = rcpx_0_pd_i.droplevel(time_pd_grp_idx, axis=0)
        #-------------------------
        # Make sure all groups (typically trsf_pole_nbs) have an entry in rcpx_0_pd_i:
        #   If a group didn't register any events in a given time period, it will not be included in the projection.
        #   However, the final format requires each group have entries for each time period
        #   Therefore, we identify the groups missing from rcpx_0_pd_i (no_events_pd_i) and add approriate rows
        #     containing all 0 values for the counts
        # NOTE: If group_cols contains more than one element, the index will be a MultiIndex (equal to the group_cols)
        #         and will need to be treated slightly different
        no_events_pd_i = list(set(all_groups).difference(set(rcpx_0_pd_i.index.unique())))
        if len(no_events_pd_i)>0:
            if len(group_cols)==1:
                no_ev_idx = no_events_pd_i
            else:
                no_ev_idx = pd.MultiIndex.from_tuples(no_events_pd_i)
            #-------------------------
            no_events_pd_i_df = pd.DataFrame(
                columns = rcpx_0.columns, 
                index   = no_ev_idx, 
                data    = np.zeros((len(no_events_pd_i), rcpx_0.shape[1]))
            )
            no_events_pd_i_df.index.names = rcpx_0_pd_i.index.names
            #-------------------------
            # Use meter_cnt_per_gp_srs to fill the nSNs_col column in no_events_pd_i_df (since simply full of 0s)
            # NOTE: This is probably not strictly necessary, as the nSNs_col column won't be used here,
            #         since the data are not normalized.
            meter_cnt_per_gp_srs_pd_i = meter_cnt_per_gp_srs[meter_cnt_per_gp_srs.index.get_level_values(time_pd_grp_idx)==time_pd_i].droplevel(level=time_pd_grp_idx, axis=0)
            #-----
            # If a group did not register any events for time_pd_i, then it will be absent from meter_cnt_per_gp_srs_pd_i
            # In such cases, use the average number of meters per group from all other available date periods
            gps_w_missing_meter_cnts = list(set(all_groups).difference(set(meter_cnt_per_gp_srs_pd_i.index.tolist())))
            meter_cnt_per_gp_srs_pd_i = pd.concat([
                meter_cnt_per_gp_srs_pd_i, 
                meter_cnt_per_gp_srs[meter_cnt_per_gp_srs.index.droplevel(level=time_pd_grp_idx).isin(gps_w_missing_meter_cnts)].groupby(group_cols).mean()
            ])
            assert(set(meter_cnt_per_gp_srs_pd_i.index.tolist()).symmetric_difference(set(all_groups))==set())
            assert(meter_cnt_per_gp_srs_pd_i.shape[0]==meter_cnt_per_gp_srs_pd_i.index.nunique())
            #-------------------------
            no_events_pd_i_df = no_events_pd_i_df.drop(columns=[nSNs_col]).merge(
                meter_cnt_per_gp_srs_pd_i, 
                left_index  = True, 
                right_index = True, 
                how         = 'left'
            )
            # Sanity check on the merge
            assert(no_events_pd_i_df[nSNs_col].notna().all())
            #-----
            # Combine rcpx_0_pd_i and no_events_pd_i_df
            assert(len(set(rcpx_0_pd_i.columns).symmetric_difference(set(no_events_pd_i_df.columns)))==0)
            no_events_pd_i_df = no_events_pd_i_df[rcpx_0_pd_i.columns]
            rcpx_0_pd_i = pd.concat([rcpx_0_pd_i, no_events_pd_i_df])
        #-------------------------
        # Rename the cr# columns to their full curated reasons
        if cr_trans_dict is None:
            _, cr_trans_dict = OutageMdlrPrep.get_regex_setup()
        rcpx_0_pd_i=rcpx_0_pd_i.rename(columns=cr_trans_dict)
        #--------------------------------------------------    
        #--------------------------------------------------
        # Any columns without a curated reason (i.e., those with column name = ''), have not been observed
        #   yet in the data, and therefore the sume of the counts should be 0.
        # These empty columns are not needed, so drop
        assert(rcpx_0_pd_i[''].sum().sum()==0)
        rcpx_0_pd_i=rcpx_0_pd_i.drop(columns=[''])  
        #-------------------------
        # Remove any undesired curated reasons (e.g., ['.*cleared.*', '.*Test Mode.*'])
        if regex_patterns_to_remove is not None:
            rcpx_0_pd_i = MECPODf.remove_reasons_from_rcpo_df(
                rcpo_df=rcpx_0_pd_i, 
                regex_patterns_to_remove=regex_patterns_to_remove, 
                ignore_case=True
            )
        #-------------------------
        # Combine any degenerate columns
        rcpx_0_pd_i = MECPODf.combine_degenerate_columns(rcpo_df = rcpx_0_pd_i)
        #-------------------------
        # After irrelevant cleared and test columns removed, need to recalculate events_tot to accurately
        #   reflect the total number of relevant events
        # Safe(r) to do this calculation in any case, so moved outside of the if block above
        assert(total_counts_col in non_reason_cols)
        rcpx_0_pd_i[total_counts_col] = rcpx_0_pd_i.drop(columns=non_reason_cols).sum(axis=1) 
        #-------------------------
        # Combine similar reasons (e.g., all 'Tamper' type reasons are combined into 1)
        # See MECPODf.combine_cpo_df_reasons for more information
        if combine_cpo_df_reasons:
            rcpx_0_pd_i = MECPODf.combine_cpo_df_reasons(rcpo_df=rcpx_0_pd_i)
        #-------------------------
        # Include the difference in power-up and power-down, if desired (typically turned off) 
        if include_power_down_minus_up:
            rcpx_0_pd_i = MECPODf.delta_cpo_df_reasons(
                rcpo_df=rcpx_0_pd_i, 
                reasons_1='Primary Power Down',
                reasons_2='Primary Power Up',
                delta_reason_name='Power Down Minus Up'
            )
        #--------------------------------------------------
        # If data_structure_df is supplied, call MECPODf.get_reasons_subset_from_cpo_df to match the structure of data_structure_df
        if data_structure_df is not None:
            assert(isinstance(data_structure_df, pd.DataFrame))
            #-----
            # Get the expected columns for this time period from data_structure_df
            final_reason_cols_i = data_structure_df[time_pd_i_rename].columns.tolist()
            final_reason_cols_i = [x for x in final_reason_cols_i if x not in non_reason_cols+[other_reasons_col]]
            #-------------------------
            # Make sure rcpx_0_pd_i contains the expected final reason columns.
            # Once this is assured, project out these reasons and combine all other reasons into
            #   the other_reasons_col columns
            # See MECPODf.get_reasons_subset_from_cpo_df for more info
            #-----
            assert(len(set(final_reason_cols_i).difference(set(rcpx_0_pd_i.columns.tolist())))==0)
            rcpx_0_pd_i = MECPODf.get_reasons_subset_from_cpo_df(
                cpo_df                       = rcpx_0_pd_i, 
                reasons_to_include           = final_reason_cols_i, 
                combine_others               = True, 
                output_combine_others_col    = other_reasons_col, 
                SNs_tags                     = None, 
                is_norm                      = False, 
                counts_col                   = nSNs_col, 
                normalize_by_nSNs_included   = False, 
                level_0_raw_col              = 'counts', 
                level_0_nrm_col              = 'counts_norm', 
                cols_to_ignore               = [total_counts_col], 
                include_counts_col_in_output = True
            )
            assert(set(data_structure_df[time_pd_i_rename].columns.tolist()).difference(set(rcpx_0_pd_i.columns.tolist()))==set())
            # NOTE: DO NOT want to reduce rcpx_0_pd_i columns to match data_structure_df exactly at this point 
            #         (i.e., do not want to call: rcpx_0_pd_i = rcpx_0_pd_i[data_structure_df[time_pd_i_rename].columns.tolist()])
            #       This is because not all operations on rcpx_0_pd_i are done yet, and some columns may be needed for operations but 
            #         are not found in final set of features.
            #       e.g., nSNs will be needed if normalization desired, but may not be included in final set of features.
            
        #--------------------------------------------------
        #--------------------------------------------------
        # Add the correct time period name as level 0 of the columns
        rcpx_0_pd_i = Utilities_df.prepend_level_to_MultiIndex(
            df         = rcpx_0_pd_i, 
            level_val  = time_pd_i_rename, 
            level_name = None, 
            axis       = 1
        )
        #-------------------------
        return rcpx_0_pd_i
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df(
        evsSum_df                   , 
        prediction_date             , 
        td_min                      , 
        td_max                      , 
        data_structure_df           = None, 
        cr_trans_dict               = None, 
        freq                        = '5D', 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        date_col                    = 'aep_event_dt', 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        include_power_down_minus_up = False, 
        regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons      = True, 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        other_reasons_col           = 'Other Reasons',  # From data_structure_df
        total_counts_col            = 'total_counts', 
        nSNs_col                    = 'nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.

        This function assumes a single prediction_date for all.  If groups have unique prediction dates, use 
          OutageMdlrPrep.build_rcpx_from_evsSum_df_wpreddates instead.
          
        NOTE: td_min, td_max, and freq must all be in DAYS
    
        data_structure_df:
            If supplied, it will be used to determine the set of final columns desired in rcpx_0_pd_i, and the method
              MECPODf.get_reasons_subset_from_cpo_df will be used to adjust rcpx_0_pd_i accordingly.
            If not supplied, MECPODf.get_reasons_subset_from_cpo_df is not calleds
    
        cr_trans_dict:
            If not supplied, will be grabbed via the OutageMdlrPrep.get_regex_setup() method
        """
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        #-------------------------
        OutageMdlrPrep.assert_timedelta_is_days(td_min)
        OutageMdlrPrep.assert_timedelta_is_days(td_max)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        if freq is not None:
            freq   = pd.Timedelta(freq)
            OutageMdlrPrep.assert_timedelta_is_days(freq)
        #--------------------------------------------------
        #-------------------------
        prereq_dict = OutageMdlrPrep.perform_build_RCPX_from_end_events_df_prereqs(
            end_events_df                 = evsSum_df, 
            mp_df                         = None, 
            ede_mp_mismatch_threshold_pct = 1.0, 
            grp_by_cols                   = group_cols, 
            outg_rec_nb_col               = outg_rec_nb_col,
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            prem_nb_col                   = prem_nb_col, 
            serial_number_col             = serial_number_col,
            mp_df_cols                    = None,  # Only needed if mp_df is not None
            is_no_outage                  = False, # Only needed if mp_df is not None
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase    = make_all_columns_lowercase
        )
        group_cols        = prereq_dict['grp_by_cols']
        outg_rec_nb_col   = prereq_dict['outg_rec_nb_col']
        trsf_pole_nb_col  = prereq_dict['trsf_pole_nb_col']
        prem_nb_col       = prereq_dict['prem_nb_col']
        serial_number_col = prereq_dict['serial_number_col']
        cols_to_drop      = prereq_dict['cols_to_drop']
        rename_cols_dict  = prereq_dict['rename_cols_dict']
        mp_df             = prereq_dict['mp_df']
        merge_on_ede      = prereq_dict['merge_on_ede']
        merge_on_mp       = prereq_dict['merge_on_mp']
        #--------------------------------------------------
        #-------------------------
        if make_all_columns_lowercase:
            date_col          = date_col.lower()
            xf_meter_cnt_col  = xf_meter_cnt_col.lower()
            events_tot_col    = events_tot_col.lower()
    
    
        #--------------------------------------------------
        # Look for any columns ending in _GPD_FOR_SQL and if not included in grpyby cols, then print warning!
        found_gpd_for_sql_cols = OutageMdlrPrep.find_gpd_for_sql_cols(
            df        = evsSum_df, 
            col_level = -1
        )
        if len(found_gpd_for_sql_cols)>0:
            # make loweercase, if needed
            if make_all_columns_lowercase:
                found_gpd_for_sql_cols = [x.lower() for x in found_gpd_for_sql_cols]
            # change names, if needed
            found_gpd_for_sql_cols = [rename_cols_dict.get(x, x) for x in found_gpd_for_sql_cols]
            #-----
            not_included = list(set(found_gpd_for_sql_cols).difference(set(group_cols)))
            if len(not_included)>0:
                print('\n!!!!! WARNING !!!!!\nOutageMdlrPrep.build_rcpx_from_evsSum_df\nFOUND POSSIBLE GROUPBY COLUMNS NOT INCLUDED IN group_cols argument')
                print(f"\tnot_included           = {not_included}\n\tfound_gpd_for_sql_cols = {found_gpd_for_sql_cols}\n\tgroup_cols             = {group_cols}")   
                print('!!!!!!!!!!!!!!!!!!!\n')
    
        #-------------------------
        evsSum_df = OutageMdlrPrep.perform_std_col_renames_and_drops(
            end_events_df              = evsSum_df.copy(), 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-----
        nec_cols = group_cols + [date_col, xf_meter_cnt_col, events_tot_col, trsf_pole_nb_col]
        assert(set(nec_cols).difference(set(evsSum_df.columns.tolist()))==set()) 
    
        #--------------------------------------------------
        # 1. Build rcpo_0
        #     Construct rcpx_0 by aggregating evsSum_df by group_cols and by freq on date_col
        #--------------------------------------------------
        evsSum_df = OutageMdlrPrep.perform_std_col_type_conversions(evsSum_df = evsSum_df)
        #-------------------------
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        assert(len(set(group_cols).difference(set(evsSum_df.columns.tolist())))==0)
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = evsSum_df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        #-----
        # cols_to_drop below are different from cols_to_drop = prereq_dict['cols_to_drop'] above.
        #   That defined above is used to remove cases when col_x and f"{col_x}_GPD_FOR_SQL" both exist in the data.
        #     For such cases, the trust_sql_grouping parameter settles which is kept.
        #   For that defined below, cols_to_drop is used to remove all the columns no longer needed for the analysis
        #       Typically, e.g., group_cols = ['outg_rec_nb', 'trsf_pole_nb']
        #       ==> cols_to_drop = ['is_first_after_outg', 'aep_opco', 'serialnumber', 'aep_premise_nb']
        cols_to_drop = set(evsSum_df.columns.tolist()).difference(
            set(cr_cols+group_cols+[date_col, xf_meter_cnt_col, events_tot_col])
        )
        cols_to_drop = list(cols_to_drop)
        #-------------------------
        # Make sure date_col is datetime object
        evsSum_df[date_col] = pd.to_datetime(evsSum_df[date_col])
        
        #-------------------------
        # No need in wasting time grouping data we won't use
        # So, reduce evsSum_df to only the dates we're interested in 
        # Note: Open left (does not include endpoint), closed right (does include endpoint) below
        #       This seems opposite to what one would intuitively think, but one must remeber that the
        #         events are all prior to, and with respect to, some prediction_date.
        #       Therefore, e.g., for a '01-06 Days' bin, one typically wants [1 Day, 6 Days)
        #       In reality, [1 Day, 6 Days) is (-6 Days, -1 Day], hence the open left and closed right below.
        evsSum_df = evsSum_df[
            (evsSum_df[date_col] >  prediction_date-td_max) & 
            (evsSum_df[date_col] <= prediction_date-td_min)
        ]
    
        #-------------------------
        # All of the cr# columns will be aggregated with np.sum, as will events_tot_col
        # xf_meter_cnt_col will be aggregated using np.max
        agg_dict = {col:'sum' for col in cr_cols+[events_tot_col, xf_meter_cnt_col]}
        agg_dict[xf_meter_cnt_col] = 'max'
    
        if freq is None:
            time_grps = [(prediction_date-td_max, prediction_date-td_min)]
            #--------------------------------------------------
            rcpx_0 = evsSum_df.drop(columns=cols_to_drop).groupby(group_cols).agg(agg_dict)
            #--------------------------------------------------
            str_len = np.max([len(str(x)) for x in [days_min, days_max]])
            str_fmt = f'{{:0{str_len}d}}'
            time_pds_rename = {time_grps[0][0] : "{}-{} Days".format(str_fmt.format(days_min), str_fmt.format(days_max))}
            #-------------------------
            # For functionality, need date_col (this function was originally designed for use with non-None freq!)
            rcpx_0[date_col] = time_grps[0][0]
            rcpx_0 = rcpx_0.reset_index(drop=False).set_index(group_cols+[date_col])
        else:
            #-------------------------
            # Need to set origin in pd.Grouper to ensure proper grouping
            freq = pd.Timedelta(freq)
            assert((td_max-td_min) % freq==pd.Timedelta(0))
            #-----
            time_grps = pd.date_range(
                start = prediction_date-td_max, 
                end   = prediction_date-td_min, 
                freq  = freq
            )
            assert(len(time_grps)>1)
            time_grps = [(time_grps[i], time_grps[i+1]) for i in range(len(time_grps)-1)]
            assert(len(time_grps) == (td_max-td_min)/pd.Timedelta(freq))
            #-------------------------
            group_freq = pd.Grouper(freq=freq, key=date_col, origin=time_grps[0][0], closed='right')
            #--------------------------------------------------
            rcpx_0 = evsSum_df.drop(columns=cols_to_drop).groupby(group_cols+[group_freq]).agg(agg_dict)
            #--------------------------------------------------
            time_pds_rename = OutageMdlrPrep.get_time_pds_rename(
                curr_time_pds = time_grps, 
                td_min        = td_min, 
                td_max        = td_max, 
                freq          = freq
            )
        final_time_pds = list(time_pds_rename.values())
    
        #--------------------------------------------------
        # 2. Grab meter_cnt_per_gp_srs and all_groups
        #--------------------------------------------------
        # Project out the meter count per group, as it will be used later
        #   This information will be stored in the pd.Series object meter_cnt_per_gp_srs, where the index will
        #   contain the group_cols
        #-----
        # OLD METHOD
        # meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[date_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[date_col]).squeeze()
        #-------------------------
        meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[date_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[date_col])
        # Generally, calling .squeeze() in this case works fine, UNLESS meter_cnt_per_gp_srs (which is a pd.DataFrame object as this point)
        #   has a single row, in which case a scalar is returned instead of a pd.Series object.
        # From the pd.DataFrame.squeeze documentation:
        #   Series or DataFrames with a single element are squeezed to a scalar.
        # To overcome the issue here, I specifically call .squeeze(axis=1)
        # For the more general case, where meter_cnt_per_gp_srs has multiple rows, calling .squeeze(axis=1) should deliver the same results 
        #   as calling .squeeze()
        # NOTE ALSO: .squeeze(axis=1) could have been tacked on to the above command.  I have broken it out into two steps for illustrative purposes, 
        #              so the explanation above makes sense!
        meter_cnt_per_gp_srs = meter_cnt_per_gp_srs.squeeze(axis=1)
        #-------------------------
        assert(meter_cnt_per_gp_srs.shape[0]==meter_cnt_per_gp_srs.index.nunique())
        meter_cnt_per_gp_srs.name = nSNs_col
    
        # Will also need the unique groups in rcpx_0
        #   This will be used later (see no_events_pd_i below)
        #   These can be grabbed from the index of rcpx_0 (excluding the date_col level)
        all_groups = rcpx_0.droplevel(date_col, axis=0).index.unique().tolist()
    
        #--------------------------------------------------
        # 3. Transform rcpx_0 to the form expected by the model
        #     i.e., similar to data_structure_df (if supplied).
        #     This is essentially just changing rcpo_0 from long form to wide form
        #--------------------------------------------------
        #-------------------------
        # 3a. Build time_pds_rename
        #      Need to convert the time periods, which are currently housed in the date_col index of 
        #        rcpx_0 from their specific dates to the names expected by the model.
        #      In rcpx_0, after grouping by the freq intervals, the values of date_col are equal to the beginning
        #        dates of the given interval.
        #      These will be converted to the titles contained in final_time_pds below
        #      NOTE: This is probably not 100% necessary, but is useful nonetheless
        #-------------------------
        #--------------------------------------------------
        #    Need for data_structure_df?
        #     In general, not all curated reasons will be included in the model.
        #     Typically, 10 commong curated reasons will be included, and all others will be grouped together in "Other Reasons".
        #     Furthermore, some reasons may be combined together, others may be completely removed.
        #     For these reasons, it is beneficial to have some sample data (taken from when the model was created) to utilize 
        #       in structuring the new data in the same fashion.
        #     Additionally, the data will be used to ensure the ordering of columns is correct before the data are fed into 
        #       the model.
        #--------------------------------------------------
        if data_structure_df is not None:
            assert(isinstance(data_structure_df, pd.DataFrame) and data_structure_df.shape[0]>0)
            # final_time_pds should all be found in data_structure_df to help
            #   ensure the alignment between the current data and data used when modelling
            assert(set(final_time_pds).difference(data_structure_df.columns.get_level_values(0).unique())==set())
    
        #-------------------------
        # 3b. Transform rcpx_0 to the form expected by the model
        #      As stated above, this is essentially just changing rcpo_0 from long form to wide form
        #      This will probably be formalized further in the future (i.e., function(s) developed to handle)
        rename_cols = {
            events_tot_col   : total_counts_col, 
            xf_meter_cnt_col : nSNs_col
        }
        rcpx_0=rcpx_0.rename(columns=rename_cols)
        #-------------------------
        total_counts_col = total_counts_col
        nSNs_col         = nSNs_col
        non_reason_cols  = [nSNs_col, total_counts_col]
        #------------------------- 
        pd_dfs = []
        for time_grp_i in time_grps:
            time_pd_i = time_grp_i[0]
            # Grab the proper time period name from time_pds_rename
            time_pd_i_rename = time_pds_rename[time_pd_i]
            #-------------------------
            rcpx_0_pd_i = OutageMdlrPrep.project_time_pd_from_rcpx_0_and_prepare(
                rcpx_0                      = rcpx_0, 
                time_pd_i                   = time_pd_i, 
                meter_cnt_per_gp_srs        = meter_cnt_per_gp_srs, 
                all_groups                  = all_groups, 
                non_reason_cols             = non_reason_cols, 
                data_structure_df           = data_structure_df, 
                other_reasons_col           = other_reasons_col, 
                cr_trans_dict               = cr_trans_dict, 
                group_cols                  = group_cols, 
                time_pd_grp_idx             = date_col, 
                time_pd_i_rename            = time_pd_i_rename, 
                regex_patterns_to_remove    = regex_patterns_to_remove, 
                combine_cpo_df_reasons      = combine_cpo_df_reasons, 
                include_power_down_minus_up = include_power_down_minus_up, 
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col
            )
            #-------------------------
            # Overkill here (since all time windows are of length freq), but something similar will 
            #   be needed if I want to move to non-uniform period lengths
            # One could, e.g., simply divide by length of freq in days
            if normalize_by_time:
                days_min_outg_td_window_i = prediction_date - time_grp_i[1]
                days_max_outg_td_window_i = prediction_date - time_grp_i[0]
                #-----
                OutageMdlrPrep.assert_timedelta_is_days(days_min_outg_td_window_i)
                OutageMdlrPrep.assert_timedelta_is_days(days_max_outg_td_window_i)
                #-----
                days_min_outg_td_window_i = days_min_outg_td_window_i.days
                days_max_outg_td_window_i = days_max_outg_td_window_i.days
                #-------------------------
                nSNs_b4 = rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].copy()
                #-----
                rcpx_0_pd_i = MECPODf.normalize_rcpo_df_by_time_interval(
                    rcpo_df                 = rcpx_0_pd_i, 
                    days_min_outg_td_window = days_min_outg_td_window_i, 
                    days_max_outg_td_window = days_max_outg_td_window_i, 
                    cols_to_adjust          = None, 
                    SNs_tags                = None, 
                    inplace                 = True
                )
                #-----
                assert(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].equals(nSNs_b4))
            #-------------------------
            if normalize_by_SNs:
                cols_to_norm              = [x for x in rcpx_0_pd_i.columns if x[1]!=nSNs_col]
                rcpx_0_pd_i[cols_to_norm] = rcpx_0_pd_i[cols_to_norm].divide(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)], axis=0)
            #-------------------------
            pd_dfs.append(rcpx_0_pd_i)
    
        # Make sure all dfs in pd_dfs look correct
        shape_0 = pd_dfs[0].shape
        index_0 = pd_dfs[0].index
        for i in range(len(pd_dfs)):
            if i==0:
                continue
            assert(pd_dfs[i].shape==shape_0)
            assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
            #-----
            # Aligning the indices is not strictly necessary, as pd.concat should handle that
            # But, it's best to be safe
            pd_dfs[i] = pd_dfs[i].loc[index_0]
    
        # Build rcpx_final by combining all dfs in pd_dfs
        # rcpx_final = pd.concat(pd_dfs, axis=1)
        rcpx_final = MECPOCollection.merge_cpo_dfs(
            dfs_coll                      = pd_dfs, 
            max_total_counts              = None, 
            how_max_total_counts          = 'any', 
            SNs_tags                      = MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols(), 
            cols_to_init_with_empty_lists = MECPODf.std_SNs_cols(), 
            make_cols_equal               = False # b/c if data_structure_df provided, pd_dfs built accordingly
        )
    
        #-------------------------
        # For the number of SNs per group, use the average number of 
        #   SNs for each group across all time periods
        nSNs_cols = [x for x in rcpx_final.columns if x[1]==nSNs_col]
        #-----
        # There should be one nSNs col for each time period
        assert(len(nSNs_cols)==len(final_time_pds))
        #-----
        rcpx_final[(nSNs_col, nSNs_col)] = rcpx_final[nSNs_cols].mean(axis=1)
        #-----
        # Sanity check on the merge
        assert(rcpx_final[nSNs_col].notna().all().all())
        #-----
        # Drop the time-period-specific nSNs columns
        rcpx_final = rcpx_final.drop(columns=nSNs_cols)
        #--------------------------------------------------
        return rcpx_final
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_dfs_in_dir(
        files_dir                      , 
        prediction_date                , 
        td_min                         , 
        td_max                         , 
        return_evsSum_df               = True, 
        #-----
        file_path_glob                 = r'events_summary_[0-9]*.csv', 
        file_path_regex                = None, 
        batch_size                     = 50, 
        cols_and_types_to_convert_dict = None, 
        to_numeric_errors              = 'coerce', 
        make_all_columns_lowercase     = True, 
        assert_all_cols_equal          = True,  
        n_update                       = 1, 
        #-----
        data_structure_df              = None, 
        cr_trans_dict                  = None, 
        freq                           = '5D', 
        group_cols                     = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        date_col                       = 'aep_event_dt', 
        normalize_by_SNs               = True, 
        normalize_by_time              = True, 
        include_power_down_minus_up    = False, 
        regex_patterns_to_remove       = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons         = True, 
        xf_meter_cnt_col               = 'xf_meter_cnt', 
        events_tot_col                 = 'events_tot', 
        trsf_pole_nb_col               = 'trsf_pole_nb', 
        outg_rec_nb_col                = 'outg_rec_nb',
        prem_nb_col                    = 'aep_premise_nb', 
        serial_number_col              = 'serialnumber',
        other_reasons_col              = 'Other Reasons',  # From data_structure_df
        total_counts_col               = 'total_counts', 
        nSNs_col                       = 'nSNs', 
        trust_sql_grouping             = True, 
        drop_gpd_for_sql_appendix      = True, 
        verbose                        = True, 
    ):
        r"""
        This function only permits uniform time periods; i.e., all periods will have length equal to freq.
        If one wants to use variable spacing (e.g., maybe the first group is one day in width, the second is
          three days, all others equal to five days), a new function will need to be built.
          
        NOTE: td_min, td_max, and freq must all be in DAYS
    
        data_structure_df:
            If supplied, it will be used to determine the set of final columns desired in rcpx_0_pd_i, and the method
              MECPODf.get_reasons_subset_from_cpo_df will be used to adjust rcpx_0_pd_i accordingly.
            If not supplied, MECPODf.get_reasons_subset_from_cpo_df is not calleds
    
        cr_trans_dict:
            If not supplied, will be grabbed via the OutageMdlrPrep.get_regex_setup() method
        """
        #--------------------------------------------------
        evsSum_df = OutageMdlrPrep.build_evsSum_df_from_csvs(
            files_dir                      = files_dir, 
            file_path_glob                 = file_path_glob, 
            file_path_regex                = file_path_regex, 
            batch_size                     = batch_size, 
            cols_and_types_to_convert_dict = cols_and_types_to_convert_dict, 
            to_numeric_errors              = to_numeric_errors, 
            make_all_columns_lowercase     = make_all_columns_lowercase, 
            assert_all_cols_equal          = assert_all_cols_equal,  
            n_update                       = n_update, 
            verbose                        = verbose
        )
        #--------------------------------------------------
        if evsSum_df is None or evsSum_df.shape[0]==0:
            return None
        #--------------------------------------------------
        rcpx_df = OutageMdlrPrep.build_rcpx_from_evsSum_df(
            evsSum_df                   = evsSum_df, 
            prediction_date             = prediction_date, 
            td_min                      = td_min, 
            td_max                      = td_max, 
            data_structure_df           = data_structure_df, 
            cr_trans_dict               = cr_trans_dict, 
            freq                        = freq, 
            group_cols                  = group_cols, 
            date_col                    = date_col, 
            normalize_by_SNs            = normalize_by_SNs, 
            normalize_by_time           = normalize_by_time, 
            include_power_down_minus_up = include_power_down_minus_up, 
            regex_patterns_to_remove    = regex_patterns_to_remove, 
            combine_cpo_df_reasons      = combine_cpo_df_reasons, 
            xf_meter_cnt_col            = xf_meter_cnt_col, 
            events_tot_col              = events_tot_col, 
            outg_rec_nb_col             = outg_rec_nb_col,
            trsf_pole_nb_col            = trsf_pole_nb_col, 
            prem_nb_col                 = prem_nb_col, 
            serial_number_col           = serial_number_col, 
            other_reasons_col           = other_reasons_col,  
            total_counts_col            = total_counts_col, 
            nSNs_col                    = nSNs_col, 
            trust_sql_grouping          = trust_sql_grouping, 
            drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase  = make_all_columns_lowercase
        )
        #--------------------------------------------------
        if return_evsSum_df:
            return rcpx_df, evsSum_df
        return rcpx_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df_wpreddates_simple(
        evsSum_df                   , 
        pred_date_col               , 
        td_min                      , 
        td_max                      , 
        data_structure_df           = None, 
        cr_trans_dict               = None, 
        freq                        = '5D', 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        date_col                    = 'aep_event_dt', 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        include_power_down_minus_up = False, 
        regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons      = True, 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        other_reasons_col           = 'Other Reasons',  # From data_structure_df
        total_counts_col            = 'total_counts', 
        nSNs_col                    = 'nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        """
        #--------------------------------------------------
        # We will alter evsSum_df.  To keep consistent outside of function, create copy
        evsSum_df = evsSum_df.copy()
        #--------------------------------------------------
        # Build relative date column, which will be used for grouping
        # rel_date_col will essentially replace date_col in build_rcpx_from_evsSum_df
        #-----
        rel_date_col = Utilities.generate_random_string(letters='letters_only')
        evsSum_df[rel_date_col] = evsSum_df[pred_date_col]-evsSum_df[date_col]
        #-----
        # Shift everything relative to a common dummy prediction date
        dummy_date     = pd.to_datetime('2 Nov. 1987')
        dummy_date_col = Utilities.generate_random_string(letters='letters_only')
        evsSum_df[dummy_date_col] = dummy_date - evsSum_df[rel_date_col]
        evsSum_df = evsSum_df.drop(columns=[date_col, rel_date_col, pred_date_col])
    
        #--------------------------------------------------
        # Now that everything is relative to common date, simply use OutageMdlrPrep.build_rcpx_from_evsSum_df 
        #   with prediction_date=dummy_date and date_col=dummy_date_col
        rcpx_df = OutageMdlrPrep.build_rcpx_from_evsSum_df(
            evsSum_df                   = evsSum_df, 
            prediction_date             = dummy_date, 
            td_min                      = td_min, 
            td_max                      = td_max, 
            data_structure_df           = data_structure_df, 
            cr_trans_dict               = cr_trans_dict, 
            freq                        = freq, 
            group_cols                  = group_cols, 
            date_col                    = dummy_date_col, 
            normalize_by_SNs            = normalize_by_SNs, 
            normalize_by_time           = normalize_by_time, 
            include_power_down_minus_up = include_power_down_minus_up, 
            regex_patterns_to_remove    = regex_patterns_to_remove, 
            combine_cpo_df_reasons      = combine_cpo_df_reasons, 
            xf_meter_cnt_col            = xf_meter_cnt_col, 
            events_tot_col              = events_tot_col, 
            outg_rec_nb_col             = outg_rec_nb_col,
            trsf_pole_nb_col            = trsf_pole_nb_col, 
            prem_nb_col                 = prem_nb_col, 
            serial_number_col           = serial_number_col,
            other_reasons_col           = other_reasons_col, 
            total_counts_col            = total_counts_col, 
            nSNs_col                    = nSNs_col, 
            trust_sql_grouping          = trust_sql_grouping, 
            drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase  = make_all_columns_lowercase, 
        )
    
        #--------------------------------------------------
        return rcpx_df
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df_wpreddates_full(
        evsSum_df                   , 
        pred_date_col               , 
        td_min                      , 
        td_max                      , 
        data_structure_df           = None, 
        cr_trans_dict               = None, 
        freq                        = '5D', 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        date_col                    = 'aep_event_dt', 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        include_power_down_minus_up = False, 
        regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons      = True, 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        other_reasons_col           = 'Other Reasons',  # From data_structure_df
        total_counts_col            = 'total_counts', 
        nSNs_col                    = 'nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        This was the original idea for the function; HOWEVER, after partial completion, the _SIMPLE method became apparent.
        I suggest using the _SIMPLE version, because it is simpler and relies on OutageMdlrPrep.build_rcpx_from_evsSum_df.
        THIS METHOD IS KEPT BECAUSE:
            - It provides a slightly different way of solving the problem, and a similar solution may be desired elsewhere later.
            - In case the _SIMPLE method does not work in some specific case, it will be interesting to investigate whether or not
                this method worked, and why.
            
        """
        #--------------------------------------------------
        # We will alter evsSum_df.  To keep consistent outside of function, create copy
        evsSum_df = evsSum_df.copy()
        #--------------------------------------------------
        # Build relative date column, which will be used for grouping
        # rel_date_col will essentially replace date_col in build_rcpx_from_evsSum_df
        #-----
        rel_date_col = Utilities.generate_random_string(letters='letters_only')
        evsSum_df[rel_date_col] = evsSum_df[pred_date_col]-evsSum_df[date_col]
        
        #--------------------------------------------------
        # Make sure td_min, td_max, and freq are all pd.Timedelta objects
        td_min = pd.Timedelta(td_min)
        td_max = pd.Timedelta(td_max)
        #-------------------------
        OutageMdlrPrep.assert_timedelta_is_days(td_min)
        OutageMdlrPrep.assert_timedelta_is_days(td_max)
        #-----
        days_min = td_min.days
        days_max = td_max.days
        if freq is not None:
            freq   = pd.Timedelta(freq)
            OutageMdlrPrep.assert_timedelta_is_days(freq)
            days_freq = freq.days
        #--------------------------------------------------
        #-------------------------
        prereq_dict = OutageMdlrPrep.perform_build_RCPX_from_end_events_df_prereqs(
            end_events_df                 = evsSum_df, 
            mp_df                         = None, 
            ede_mp_mismatch_threshold_pct = 1.0, 
            grp_by_cols                   = group_cols, 
            outg_rec_nb_col               = outg_rec_nb_col,
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            prem_nb_col                   = prem_nb_col, 
            serial_number_col             = serial_number_col,
            mp_df_cols                    = None,  # Only needed if mp_df is not None
            is_no_outage                  = False, # Only needed if mp_df is not None
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase    = make_all_columns_lowercase
        )
        group_cols        = prereq_dict['grp_by_cols']
        outg_rec_nb_col   = prereq_dict['outg_rec_nb_col']
        trsf_pole_nb_col  = prereq_dict['trsf_pole_nb_col']
        prem_nb_col       = prereq_dict['prem_nb_col']
        serial_number_col = prereq_dict['serial_number_col']
        cols_to_drop      = prereq_dict['cols_to_drop']
        rename_cols_dict  = prereq_dict['rename_cols_dict']
        mp_df             = prereq_dict['mp_df']
        merge_on_ede      = prereq_dict['merge_on_ede']
        merge_on_mp       = prereq_dict['merge_on_mp']
        #--------------------------------------------------
        #-------------------------
        if make_all_columns_lowercase:
            date_col          = date_col.lower()
            xf_meter_cnt_col  = xf_meter_cnt_col.lower()
            events_tot_col    = events_tot_col.lower()
            rel_date_col      = rel_date_col.lower()
        
        #--------------------------------------------------
        # Look for any columns ending in _GPD_FOR_SQL and if not included in grpyby cols, then print warning!
        found_gpd_for_sql_cols = OutageMdlrPrep.find_gpd_for_sql_cols(
            df        = evsSum_df, 
            col_level = -1
        )
        if len(found_gpd_for_sql_cols)>0:
            # make loweercase, if needed
            if make_all_columns_lowercase:
                found_gpd_for_sql_cols = [x.lower() for x in found_gpd_for_sql_cols]
            # change names, if needed
            found_gpd_for_sql_cols = [rename_cols_dict.get(x, x) for x in found_gpd_for_sql_cols]
            #-----
            not_included = list(set(found_gpd_for_sql_cols).difference(set(group_cols)))
            if len(not_included)>0:
                print('\n!!!!! WARNING !!!!!\nOutageMdlrPrep.build_rcpx_from_evsSum_df\nFOUND POSSIBLE GROUPBY COLUMNS NOT INCLUDED IN group_cols argument')
                print(f"\tnot_included           = {not_included}\n\tfound_gpd_for_sql_cols = {found_gpd_for_sql_cols}\n\tgroup_cols             = {group_cols}")   
                print('!!!!!!!!!!!!!!!!!!!\n')
        
        #-------------------------
        evsSum_df = OutageMdlrPrep.perform_std_col_renames_and_drops(
            end_events_df              = evsSum_df.copy(), 
            cols_to_drop               = cols_to_drop, 
            rename_cols_dict           = rename_cols_dict, 
            make_all_columns_lowercase = make_all_columns_lowercase
        )
        #-----
        nec_cols = group_cols + [rel_date_col, xf_meter_cnt_col, events_tot_col, trsf_pole_nb_col]
        assert(set(nec_cols).difference(set(evsSum_df.columns.tolist()))==set()) 
        
        #--------------------------------------------------
        # 1. Build rcpo_0
        #     Construct rcpx_0 by aggregating evsSum_df by group_cols and by freq on rel_date_col
        #--------------------------------------------------
        evsSum_df = OutageMdlrPrep.perform_std_col_type_conversions(evsSum_df = evsSum_df)
        #-------------------------
        if not isinstance(group_cols, list):
            group_cols = [group_cols]
        assert(len(set(group_cols).difference(set(evsSum_df.columns.tolist())))==0)
        #-------------------------
        cr_cols = Utilities.find_in_list_with_regex(
            lst           = evsSum_df.columns.tolist(), 
            regex_pattern = r'cr\d*', 
            ignore_case   = False
        )
        #-----
        # cols_to_drop below are different from cols_to_drop = prereq_dict['cols_to_drop'] above.
        #   That defined above is used to remove cases when col_x and f"{col_x}_GPD_FOR_SQL" both exist in the data.
        #     For such cases, the trust_sql_grouping parameter settles which is kept.
        #   For that defined below, cols_to_drop is used to remove all the columns no longer needed for the analysis
        #       Typically, e.g., group_cols = ['outg_rec_nb', 'trsf_pole_nb']
        #       ==> cols_to_drop = ['is_first_after_outg', 'aep_opco', 'serialnumber', 'aep_premise_nb']
        cols_to_drop = set(evsSum_df.columns.tolist()).difference(
            set(cr_cols+group_cols+[rel_date_col, xf_meter_cnt_col, events_tot_col])
        )
        cols_to_drop = list(cols_to_drop)
        #-------------------------
        # Make sure rel_date_col is timedelta object
        assert(is_timedelta64_dtype(evsSum_df[rel_date_col].dtype))
        
        #-------------------------
        # No need in wasting time grouping data we won't use
        # So, reduce evsSum_df to only the dates we're interested in 
        # Note: Closed left (does include endpoint), open right (does not include endpoint) below
        #       e.g., '01-06 Days' = [1 Day, 6 Days)
        evsSum_df = evsSum_df[
            (evsSum_df[rel_date_col] >= td_min) &
            (evsSum_df[rel_date_col] <  td_max)
        ].copy()
        
        #-------------------------
        # All of the cr# columns will be aggregated with np.sum, as will events_tot_col
        # xf_meter_cnt_col will be aggregated using np.max
        agg_dict = {col:'sum' for col in cr_cols+[events_tot_col, xf_meter_cnt_col]}
        agg_dict[xf_meter_cnt_col] = 'max'
    
        #--------------------------------------------------
        time_pd_grp_col       = Utilities.generate_random_string(letters='letters_only')
        time_pd_grp_width_col = Utilities.generate_random_string(letters='letters_only')
        #-------------------------
        if freq is None:
            evsSum_df[time_pd_grp_col]       = 1
            evsSum_df[time_pd_grp_width_col] = days_max-days_min
            grps_w_widths                    = [(1, days_max-days_min)]
        else:
            evsSum_df[time_pd_grp_col]       = np.digitize(evsSum_df[rel_date_col].dt.days, np.arange(days_min, days_max+1, days_freq))
            evsSum_df[time_pd_grp_width_col] = days_freq 
            
            grps_w_widths = list(evsSum_df.groupby([time_pd_grp_col, time_pd_grp_width_col]).groups.keys())
            # Make sure grps_w_widths is properly sorted
            grps_w_widths = natsorted(grps_w_widths, key=lambda x:x[0])
            # Also note, for np.digitize, the indices apparently begin at 1, not 0
            # assertion below ensures this is true
            assert(grps_w_widths[0][0]==1)
        # Since we are using custom agg_dict, there is no need to call evsSum_df.drop(columns=cols_to_drop) before grouping/aggregation
        rcpx_0 = evsSum_df.groupby(group_cols+[time_pd_grp_col]).agg(agg_dict)
        
        #--------------------------------------------------
        # Using grps_w_widths is a bit overkill currently, since freq is constant.
        # However, if freq is allowed to be irregular in the future, the overkill included
        #   will be necessary
        #-------------------------
        time_pds_rename = {}
        #-------------------------
        # Use padded zeros
        #   i.e., instead of '1-6 Days', use '01-06 Days'
        #   Will need to know longest string length for this
        #-----
        # Largest number, therefore largest str, will be days_min + all widths
        str_len = len(str(days_min + np.sum([x[1] for x in grps_w_widths])))
        #-----
        # Looks like a ton of curly braces below, but need double {{ and double }} to escape
        # ==> if str_len=2, then str_fmt = '{:02d}'
        str_fmt = f'{{:0{str_len}d}}'
        #-------------------------
        t_rght_i = days_min
        # Remember, since np.digitize was used, the indices apparently begin at 1, not 0
        for grp_i, width_i in grps_w_widths:
            t_left_i = t_rght_i
            t_rght_i = t_left_i + width_i
            #-----
            rename_i = "{}-{} Days".format(
                str_fmt.format(t_left_i), 
                str_fmt.format(t_rght_i)
            )
            #-----
            assert(grp_i not in time_pds_rename.keys())
            time_pds_rename[grp_i] = rename_i
        
        #--------------------------------------------------
        # 2. Grab meter_cnt_per_gp_srs and all_groups
        #--------------------------------------------------
        # Project out the meter count per group, as it will be used later
        #   This information will be stored in the pd.Series object meter_cnt_per_gp_srs, where the index will
        #   contain the group_cols
        #-----
        # OLD METHOD
        # meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[time_pd_grp_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[time_pd_grp_col]).squeeze()
        #-------------------------
        meter_cnt_per_gp_srs = rcpx_0.reset_index()[group_cols+[time_pd_grp_col, xf_meter_cnt_col]].drop_duplicates().set_index(group_cols+[time_pd_grp_col])
        # Generally, calling .squeeze() in this case works fine, UNLESS meter_cnt_per_gp_srs (which is a pd.DataFrame object as this point)
        #   has a single row, in which case a scalar is returned instead of a pd.Series object.
        # From the pd.DataFrame.squeeze documentation:
        #   Series or DataFrames with a single element are squeezed to a scalar.
        # To overcome the issue here, I specifically call .squeeze(axis=1)
        # For the more general case, where meter_cnt_per_gp_srs has multiple rows, calling .squeeze(axis=1) should deliver the same results 
        #   as calling .squeeze()
        # NOTE ALSO: .squeeze(axis=1) could have been tacked on to the above command.  I have broken it out into two steps for illustrative purposes, 
        #              so the explanation above makes sense!
        meter_cnt_per_gp_srs = meter_cnt_per_gp_srs.squeeze(axis=1)
        #-------------------------
        assert(meter_cnt_per_gp_srs.shape[0]==meter_cnt_per_gp_srs.index.nunique())
        meter_cnt_per_gp_srs.name = nSNs_col
        
        # Will also need the unique groups in rcpx_0
        #   This will be used later (see no_events_pd_i below)
        #   These can be grabbed from the index of rcpx_0 (excluding the time_pd_grp_col level)
        all_groups = rcpx_0.droplevel(time_pd_grp_col, axis=0).index.unique().tolist()
        
        #--------------------------------------------------
        # 3. Transform rcpx_0 to the form expected by the model
        #     i.e., similar to data_structure_df (if supplied).
        #     This is essentially just changing rcpo_0 from long form to wide form
        #--------------------------------------------------
        #-------------------------
        # 3a. Build time_pds_rename
        #      Need to convert the time periods, which are currently housed in the time_pd_grp_col index of 
        #        rcpx_0 from their specific dates to the names expected by the model.
        #      In rcpx_0, after grouping by the freq intervals, the values of time_pd_grp_col are equal to the beginning
        #        dates of the given interval.
        #      These will be converted to the titles contained in final_time_pds below
        #      NOTE: This is probably not 100% necessary, but is useful nonetheless
        #-------------------------
        #--------------------------------------------------
        #    Need for data_structure_df?
        #     In general, not all curated reasons will be included in the model.
        #     Typically, 10 commong curated reasons will be included, and all others will be grouped together in "Other Reasons".
        #     Furthermore, some reasons may be combined together, others may be completely removed.
        #     For these reasons, it is beneficial to have some sample data (taken from when the model was created) to utilize 
        #       in structuring the new data in the same fashion.
        #     Additionally, the data will be used to ensure the ordering of columns is correct before the data are fed into 
        #       the model.
        #--------------------------------------------------
        # if data_structure_df is not None:
        #     assert(isinstance(data_structure_df, pd.DataFrame) and data_structure_df.shape[0]>0)
        #     # final_time_pds should all be found in data_structure_df to help
        #     #   ensure the alignment between the current data and data used when modelling
        #     assert(set(final_time_pds).difference(data_structure_df.columns.get_level_values(0).unique())==set())
        
        #-------------------------
        # 3b. Transform rcpx_0 to the form expected by the model
        #      As stated above, this is essentially just changing rcpo_0 from long form to wide form
        #      This will probably be formalized further in the future (i.e., function(s) developed to handle)
        rename_cols = {
            events_tot_col   : total_counts_col, 
            xf_meter_cnt_col : nSNs_col
        }
        rcpx_0=rcpx_0.rename(columns=rename_cols)
        #-------------------------
        total_counts_col = total_counts_col
        nSNs_col         = nSNs_col
        non_reason_cols  = [nSNs_col, total_counts_col]
        #------------------------- 
        pd_dfs = []
        for grp_i, width_i in grps_w_widths:
            time_pd_i = grp_i
            # Grab the proper time period name from final_time_pd_i
            time_pd_i_rename = time_pds_rename[time_pd_i]
            #-------------------------
            rcpx_0_pd_i = OutageMdlrPrep.project_time_pd_from_rcpx_0_and_prepare(
                rcpx_0                      = rcpx_0, 
                time_pd_i                   = time_pd_i, 
                meter_cnt_per_gp_srs        = meter_cnt_per_gp_srs, 
                all_groups                  = all_groups, 
                non_reason_cols             = non_reason_cols, 
                data_structure_df           = data_structure_df, 
                other_reasons_col           = other_reasons_col, 
                cr_trans_dict               = cr_trans_dict, 
                group_cols                  = group_cols, 
                time_pd_grp_idx             = time_pd_grp_col, 
                time_pd_i_rename            = time_pd_i_rename, 
                regex_patterns_to_remove    = regex_patterns_to_remove, 
                combine_cpo_df_reasons      = combine_cpo_df_reasons, 
                include_power_down_minus_up = include_power_down_minus_up, 
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col
            )
            #-------------------------
            # Overkill here (since all time windows are of length freq), but something similar will 
            #   be needed if I want to move to non-uniform period lengths
            # One could, e.g., simply divide by length of freq in days
            if normalize_by_time:
                #-------------------------
                nSNs_b4 = rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].copy()
                #-----
                rcpx_0_pd_i = MECPODf.normalize_rcpo_df_by_time_interval(
                    rcpo_df                 = rcpx_0_pd_i, 
                    days_min_outg_td_window = 0, 
                    days_max_outg_td_window = width_i, 
                    cols_to_adjust          = None, 
                    SNs_tags                = None, 
                    inplace                 = True
                )
                #-----
                assert(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)].equals(nSNs_b4))
            #-------------------------
            if normalize_by_SNs:
                cols_to_norm = [x for x in rcpx_0_pd_i.columns if x[1]!=nSNs_col]
                rcpx_0_pd_i[cols_to_norm] = rcpx_0_pd_i[cols_to_norm].divide(rcpx_0_pd_i[(time_pd_i_rename, nSNs_col)], axis=0)
            #-------------------------
            pd_dfs.append(rcpx_0_pd_i)
        
        # Make sure all dfs in pd_dfs look correct
        shape_0 = pd_dfs[0].shape
        index_0 = pd_dfs[0].index
        for i in range(len(pd_dfs)):
            if i==0:
                continue
            assert(pd_dfs[i].shape==shape_0)
            assert(len(set(index_0).symmetric_difference(set(pd_dfs[i].index)))==0)
            #-----
            # Aligning the indices is not strictly necessary, as pd.concat should handle that
            # But, it's best to be safe
            pd_dfs[i] = pd_dfs[i].loc[index_0]
        
        # Build rcpx_final by combining all dfs in pd_dfs
        # rcpx_final = pd.concat(pd_dfs, axis=1)
        rcpx_final = MECPOCollection.merge_cpo_dfs(
            dfs_coll                      = pd_dfs, 
            max_total_counts              = None, 
            how_max_total_counts          = 'any', 
            SNs_tags                      = MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols(), 
            cols_to_init_with_empty_lists = MECPODf.std_SNs_cols(), 
            make_cols_equal               = False # b/c if data_structure_df provided, pd_dfs built accordingly
        )
        
        #-------------------------
        # For the number of SNs per group, use the average number of 
        #   SNs for each group across all time periods
        nSNs_cols = [x for x in rcpx_final.columns if x[1]==nSNs_col]
        #-----
        # # There should be one nSNs col for each time period
        # assert(len(nSNs_cols)==len(final_time_pds))
        #-----
        rcpx_final[(nSNs_col, nSNs_col)] = rcpx_final[nSNs_cols].mean(axis=1)
        #-----
        # Sanity check on the merge
        assert(rcpx_final[nSNs_col].notna().all().all())
        #-----
        # Drop the time-period-specific nSNs columns
        rcpx_final = rcpx_final.drop(columns=nSNs_cols)
    
        #--------------------------------------------------
        return rcpx_final
    

    #---------------------------------------------------------------------------
    @staticmethod
    def build_rcpx_from_evsSum_df_wpreddates(
        evsSum_df                   , 
        pred_date_col               , 
        td_min                      , 
        td_max                      , 
        data_structure_df           = None, 
        cr_trans_dict               = None, 
        freq                        = '5D', 
        group_cols                  = ['OUTG_REC_NB_GPD_FOR_SQL', 'trsf_pole_nb_GPD_FOR_SQL'], 
        date_col                    = 'aep_event_dt', 
        normalize_by_SNs            = True, 
        normalize_by_time           = True, 
        include_power_down_minus_up = False, 
        regex_patterns_to_remove    = ['.*cleared.*', '.*Test Mode.*'], 
        combine_cpo_df_reasons      = True, 
        xf_meter_cnt_col            = 'xf_meter_cnt', 
        events_tot_col              = 'events_tot', 
        outg_rec_nb_col             = 'outg_rec_nb',
        trsf_pole_nb_col            = 'trsf_pole_nb', 
        prem_nb_col                 = 'aep_premise_nb', 
        serial_number_col           = 'serialnumber',
        other_reasons_col           = 'Other Reasons',  # From data_structure_df
        total_counts_col            = 'total_counts', 
        nSNs_col                    = 'nSNs', 
        trust_sql_grouping          = True, 
        drop_gpd_for_sql_appendix   = True, 
        make_all_columns_lowercase  = True, 
    ):
        r"""
        Try _simple method first, if that doesn't work, try _full method
        """
        #--------------------------------------------------
        try:
            rcpx_df = OutageMdlrPrep.build_rcpx_from_evsSum_df_wpreddates_simple(
                evsSum_df                   = evsSum_df, 
                pred_date_col               = pred_date_col, 
                td_min                      = td_min, 
                td_max                      = td_max, 
                data_structure_df           = data_structure_df, 
                cr_trans_dict               = cr_trans_dict, 
                freq                        = freq, 
                group_cols                  = group_cols, 
                date_col                    = date_col, 
                normalize_by_SNs            = normalize_by_SNs, 
                normalize_by_time           = normalize_by_time, 
                include_power_down_minus_up = include_power_down_minus_up, 
                regex_patterns_to_remove    = regex_patterns_to_remove, 
                combine_cpo_df_reasons      = combine_cpo_df_reasons, 
                xf_meter_cnt_col            = xf_meter_cnt_col, 
                events_tot_col              = events_tot_col, 
                outg_rec_nb_col             = outg_rec_nb_col,
                trsf_pole_nb_col            = trsf_pole_nb_col, 
                prem_nb_col                 = prem_nb_col, 
                serial_number_col           = serial_number_col,
                other_reasons_col           = other_reasons_col,  
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col, 
                trust_sql_grouping          = trust_sql_grouping, 
                drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
                make_all_columns_lowercase  = make_all_columns_lowercase, 
            )
            return rcpx_df
        except:
            print('OutageMdlrPrep.build_rcpx_from_evsSum_df_wpreddates: _simple method failed, trying _full')
            pass
        #--------------------------------------------------
        try:
            rcpx_df = OutageMdlrPrep.build_rcpx_from_evsSum_df_wpreddates_full(
                evsSum_df                   = evsSum_df, 
                pred_date_col               = pred_date_col, 
                td_min                      = td_min, 
                td_max                      = td_max, 
                data_structure_df           = data_structure_df, 
                cr_trans_dict               = cr_trans_dict, 
                freq                        = freq, 
                group_cols                  = group_cols, 
                date_col                    = date_col, 
                normalize_by_SNs            = normalize_by_SNs, 
                normalize_by_time           = normalize_by_time, 
                include_power_down_minus_up = include_power_down_minus_up, 
                regex_patterns_to_remove    = regex_patterns_to_remove, 
                combine_cpo_df_reasons      = combine_cpo_df_reasons, 
                xf_meter_cnt_col            = xf_meter_cnt_col, 
                events_tot_col              = events_tot_col, 
                outg_rec_nb_col             = outg_rec_nb_col,
                trsf_pole_nb_col            = trsf_pole_nb_col, 
                prem_nb_col                 = prem_nb_col, 
                serial_number_col           = serial_number_col,
                other_reasons_col           = other_reasons_col,  
                total_counts_col            = total_counts_col, 
                nSNs_col                    = nSNs_col, 
                trust_sql_grouping          = trust_sql_grouping, 
                drop_gpd_for_sql_appendix   = drop_gpd_for_sql_appendix, 
                make_all_columns_lowercase  = make_all_columns_lowercase, 
            )
            return rcpx_df
        except:
            print('OutageMdlrPrep.build_rcpx_from_evsSum_df_wpreddates: _full method failed, CRASH IMMINENT!')
            assert(0)


    #---------------------------------------------------------------------------
    @staticmethod
    def try_to_steal_mp_df_curr_hist_from_previous(
        save_dir_base_pkls      , 
        days_min_outg_td_window , 
        days_max_outg_td_window , 
        subdir_regex            = r'^outg_td_window_(\d*)_to_(\d*)_days$', 
        naming_tag              = '', 
        verbose                 = True, 
    ):
        r"""
        If appropriate results are found, a dict object is returned with keys = ['mp_df_hist', 'mp_df_curr'] and values equal
          to the corresponding MeterPremise pd.DataFrame objects.
        If appropriate results are not found, None is returned.
    
        subdir_regex:
            Default = r'^outg_td_window_(\d*)_to_(\d*)_days$' means the subdirs strictly must begin with outg.... and end with ...days
            To loosen this restriction, use subdir_regex = r'outg_td_window_(\d*)_to_(\d*)_days' instead
        """
        #--------------------------------------------------
        subdirs = [
            x for x in os.listdir(save_dir_base_pkls) 
            if os.path.isdir(os.path.join(save_dir_base_pkls, x))
        ]
        #-----
        subdirs = Utilities.find_in_list_with_regex(
            lst           = subdirs, 
            regex_pattern = subdir_regex, 
            ignore_case   = False
        )
        
        #--------------------------------------------------
        # Extract windows and see if days_min_outg_td_window/days_max_outg_td_window
        #   completely contained
        #-----
        found_contained  = False
        #-----
        for subdir_i in subdirs:
            found_i = re.findall(subdir_regex, subdir_i)
            assert(len(found_i)    == 1)
            assert(len(found_i[0]) == 2)
            #-----
            window_i_min = int(found_i[0][0])
            window_i_max = int(found_i[0][1])
            #-------------------------
            # In order to use previous MeterPremise data, current window must be completely contained by previous
            #   AND previous results must exist!
            if(
                window_i_min <= days_min_outg_td_window and 
                window_i_max >= days_max_outg_td_window and
                os.path.exists(os.path.join(save_dir_base_pkls, subdir_i, f'mp{naming_tag}_df_hist.pkl')) and 
                os.path.exists(os.path.join(save_dir_base_pkls, subdir_i, f'mp{naming_tag}_df_curr.pkl'))
            ):
                try:
                    mp_df_curr_hist = {}
                    mp_df_curr_hist['mp_df_hist'] = pd.read_pickle(os.path.join(save_dir_base_pkls, subdir_i, f'mp{naming_tag}_df_hist.pkl'))
                    mp_df_curr_hist['mp_df_curr'] = pd.read_pickle(os.path.join(save_dir_base_pkls, subdir_i, f'mp{naming_tag}_df_curr.pkl'))
                    #-------------------------
                    if verbose:
                        print('Found previous results which completely contain current, so using previous MeterPremise data')
                        print(f'\tPrevious subdirectory = {subdir_i}')
                        print(f'\tCurrent window        = {days_min_outg_td_window}:{days_max_outg_td_window}')
                    #-------------------------
                    found_contained  = True
                    #-------------------------
                    break
                except:
                    continue
        #--------------------------------------------------
        if found_contained:
            return mp_df_curr_hist
        else:
            return None

    #---------------------------------------------------------------------------
    @staticmethod
    def rough_time_slice_and_drop_dups_mp_df_curr_hist(
        mp_df_curr_hist         , 
        t_min                   , 
        t_max                   , 
        df_mp_serial_number_col = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col       = 'prem_nb', 
        df_mp_install_time_col  = 'inst_ts', 
        df_mp_removal_time_col  = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col  = 'trsf_pole_nb',
    ):
        r"""
        Specialized function which performs time slicing and drops duplicates in mp_df_curr_hist.
        1. Time slicing:
            We want to keep only meters which were active for the relevant time period.
            SEE COMMENTS WITHIN CODE AS WELL!
            -----
            I decided to not do the removal on current, only hist!.
            This is because current is used for get_SNs_andor_PNs_for_xfmrs
            e.g., I was missing some PNs because maybe a new meter was installed after rcpo_df['DT_OFF_TS_FULL'].max()
                So, in all likelihood that was an appropriate meter entry in historical, but this was excluded because
                there wasn't an entry in current that passed the cuts below
            -----
            NOTE:
            Having extra meters in mp_df_curr_hist is not harmful.
            Furthermore, doing time slicing meter-by-meter for a bunch of different relevant time periods is time and resource consuming.
            Therefore, we impose only a rough time slicing, utilizing only a single t_min and t_max for the collection (i.e., utilizing
              a single relevant time period)
            -----
            t_min/t_max typical values:
                rcpo_df[(time_cols_lvl_0_val, t_min/max_col)].min()/max() where, e.g., 
                    t_min/max_col = 't_min/max' or 
                    t_min/max_col = 'DT_OFF_TS_FULL/DT_ON_TS' 
        """
        #--------------------------------------------------
        assert(isinstance(mp_df_curr_hist, dict))
        assert(set(mp_df_curr_hist.keys()).symmetric_difference(set(['mp_df_curr', 'mp_df_hist']))==set())
        #--------------------------------------------------
        # 1. Time slicing
        # As mentioned above, I decided to not do the removal on current, only hist AND this is only a rough slicing.
        # NOTICE, this form looks a bit strange when compared to the normal/similar functionality elsewhere.
        #   e.g., in MeterPremise.get_active_SNs_for_PNs_at_datetime_interval, we have:
        #        active_SNs_at_time = df_mp_hist[(df_mp_hist[df_mp_install_time_col]                         <= pd.to_datetime(dt_0)) & 
        #                                        (df_mp_hist[df_mp_removal_time_col].fillna(pd.Timestamp.max) > pd.to_datetime(dt_1))]
        # Here, we have (... <= t_max) & (... > t_min), elsewhere uses (... <= t_min) & (... > t_max).
        # Elsewhere, we want, e.g., only meters installed sometime before an event begins and removed sometime after the event ends (or still
        #   present/not removed).
        # Here, we are dealing with many meters from many events.  
        #   To ensure we do not throw away any entries that will be needed, the restrictions must be relaxed.
        #     The MINIMUM time before which a meter must have been installed to be included is INCREASED (t_min-->t_max)
        #     The MAXIMUM time after  which a meter must have been removed   to be included is DECREASED (t_max-->t_min)
        # This flopping of (... <= t_min) & (... > t_max) ==> (... <= t_max) & (... > t_min) makes inclusion less restrictive and
        #   ensures that we do not throw out any of the baby with the bathwater.
        # Suppose the first event in the collection happens a year before the last.  In that year, some of the meters from the first event 
        # are removed.  If we incorrectly used (... <= t_min) & (... > t_max), (... > t_max) would exclude those meters which were removed 
        # during the year, but will definitely be needed when analyzing the first event.  Some of the baby has been thrown out
        #-----
        mp_df_curr_hist['mp_df_hist'] = mp_df_curr_hist['mp_df_hist'][
            (mp_df_curr_hist['mp_df_hist'][df_mp_install_time_col]                         <= pd.to_datetime(t_max)) & 
            (mp_df_curr_hist['mp_df_hist'][df_mp_removal_time_col].fillna(pd.Timestamp.max) > pd.to_datetime(t_min))
        ]
    
        #--------------------------------------------------
        # 2. Drop approx duplicates
        drop_approx_duplicates_args = MeterPremise.get_dflt_args_drop_approx_mp_duplicates(
            df_mp_serial_number_col = df_mp_serial_number_col, 
            df_mp_prem_nb_col       = df_mp_prem_nb_col, 
            df_mp_install_time_col  = df_mp_install_time_col, 
            df_mp_removal_time_col  = df_mp_removal_time_col, 
            df_mp_trsf_pole_nb_col  = df_mp_trsf_pole_nb_col
        )
        #-----
        mp_df_curr_hist['mp_df_hist'] = MeterPremise.drop_approx_mp_duplicates(
            mp_df = mp_df_curr_hist['mp_df_hist'], 
            **drop_approx_duplicates_args
        )
        #-----
        mp_df_curr_hist['mp_df_curr'] = MeterPremise.drop_approx_mp_duplicates(
            mp_df = mp_df_curr_hist['mp_df_curr'], 
            **drop_approx_duplicates_args
        )
    
        #--------------------------------------------------
        return mp_df_curr_hist


    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
