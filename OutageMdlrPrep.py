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
import datetime
import time

#---------------------------------------------------------------------
#---------------------------------------------------------------------
sys.path.insert(0, os.path.realpath('..'))
import Utilities_config
#-----
from MeterPremise import MeterPremise
#-----
from AMI_SQL import AMI_SQL
from DOVSOutages_SQL import DOVSOutages_SQL
#-----
from GenAn import GenAn
from AMIEndEvents import AMIEndEvents
from CPXDfBuilder import CPXDfBuilder
from CPXDf import CPXDf
from MECPODf import MECPODf
from MECPOAn import MECPOAn
from DOVSOutages import DOVSOutages
from OutageDAQ import OutageDataInfo as ODI
#---------------------------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
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
        dataset                , 
        run_date               , 
        date_0                 ,  
        date_1                 , 
        run_date_subdir_appndx = None, 
        data_evs_sum_vw        = False, 
        data_base_dir          = None, 
        verbose                = True, 
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
                    run_date_subdir_appndx (if not None), 
                    f"{self.date_0.replace('-','')}_{self.date_1.replace('-','')}"
                )

        data_evs_sum_vw:
            Indicates whether or not the data being prepared are from the meter_events.events_summary_vw (data_evs_sum_vw=True) table
              or the meter_events.end_device_event (data_evs_sum_vw=False) data.
            The latter case (data_evs_sum_vw=False) is what the bulk of this class is intended to handle
        """
        #---------------------------------------------------------------------------
        ODI.assert_dataset(dataset)
        #-------------------------
        self.dataset         = dataset
        self.data_evs_sum_vw = data_evs_sum_vw
        self.naming_tag      = ODI.get_naming_tag(self.dataset)
        self.is_no_outage    = ODI.get_is_no_outage(self.dataset)
        #-------------------------
        self.run_date        = pd.to_datetime(run_date).strftime('%Y%m%d')
        self.date_0          = pd.to_datetime(date_0).strftime('%Y-%m-%d')
        self.date_1          = pd.to_datetime(date_1).strftime('%Y-%m-%d')
        #-----
        self.verbose         = verbose
        #--------------------------------------------------
        if data_base_dir is None:
            run_date_subdir = self.run_date
            if run_date_subdir_appndx is not None:
                run_date_subdir += run_date_subdir_appndx
            #-----
            date_pd_subdir = f"{self.date_0.replace('-','')}_{self.date_1.replace('-','')}"
            #-----
            self.data_base_dir = os.path.join(
                Utilities.get_local_data_dir(), 
                r'dovs_and_end_events_data', 
                run_date_subdir, 
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
        #--------------------------------------------------
        if self.data_evs_sum_vw:
            self.files_dir = os.path.join(self.data_base_dir, 'EvsSums')
        else:
            self.files_dir = os.path.join(self.data_base_dir, 'EndEvents')
        #-----
        if not os.path.isdir(self.files_dir):
            print(f'ERROR: files_dir = {self.files_dir} DOES NOT EXIST!')
            assert(0)
        #-------------------------
        if self.verbose:
            print(f'data_base_dir = {self.data_base_dir}')
            print(f'files_dir     = {self.files_dir}')
            print(f'naming_tag    = {self.naming_tag}')
            print(f'is_no_outage  = {self.is_no_outage}')


    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    @staticmethod
    def reset_index_and_identify_cols_to_merge(
        df                , 
        merge_on          , 
        tag_for_idx_names = None
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
        df_idx_names = [
            x if x is not None else f'index_{i}' 
            for i,x in enumerate(df_idx_names_OG)
        ]
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
                        idfr_idx_lvl = idfr_idx_lvl[0]
                        idfr_idx_lvl = int(idfr_idx_lvl)
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
            df              = df, 
            df_idx_names_OG = df_idx_names_OG, 
            df_idx_names    = df_idx_names, 
            reset_merge_on  = reset_merge_on
        )
    
    #---------------------------------------------------------------------------
    @staticmethod
    def merge_rcpo_and_df(
        rcpo_df    , 
        df_2       , 
        rcpo_df_on ,
        df_2_on    , 
        how        = 'left'
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
            df                = rcpo_df, 
            merge_on          = rcpo_df_on, 
            tag_for_idx_names = 'from_rcpo_df'
        )
    
        reset_df_2_dict = OutageMdlrPrep.reset_index_and_identify_cols_to_merge(
            df                = df_2, 
            merge_on          = df_2_on, 
            tag_for_idx_names = 'from_df_2'
        )
        #-------------------------
        merged_df = pd.merge(
            reset_rcpo_df_dict['df'], 
            reset_df_2_dict['df'], 
            left_on  = reset_rcpo_df_dict['reset_merge_on'], 
            right_on = reset_df_2_dict['reset_merge_on'], 
            how      = how
        )
        #-------------------------
        # Want to set index first before changing names back to originals, as the
        #   originals could have been None
        merged_df = merged_df.set_index(reset_rcpo_df_dict['df_idx_names'])
    
        # Two lines below (instead of calling simply df.index.names=df_idx_names_OG) 
        #   ensures the order is the same
        rcpo_df_rename_dict   = dict(zip(reset_rcpo_df_dict['df_idx_names'], reset_rcpo_df_dict['df_idx_names_OG']))
        merged_df.index.names = [rcpo_df_rename_dict[x] for x in merged_df.index.names]
        #-------------------------
        # When merging two columns whose names are different, both columns are kept
        #   This is redundant, as these will have identical values, as they were merged, so get rid of
        #   Note: If the column names are the same, this is obviously not an issue (hence the need to find
        #         cols_to_drop below, instead of simply dropping all of reset_df_2_dict['reset_merge_on']
        cols_to_drop = [x for x in reset_df_2_dict['reset_merge_on'] if x in merged_df.columns]
        merged_df    = merged_df.drop(columns=cols_to_drop)
    
        # Rename columns from df_2 to original values (if original values were not None!)
        df_2_rename_dict = dict(zip(reset_df_2_dict['df_idx_names'], reset_df_2_dict['df_idx_names_OG']))
        df_2_rename_dict = {k:v for k,v in df_2_rename_dict.items() if k in merged_df.columns and v is not None}
        merged_df        = merged_df.rename(columns=df_2_rename_dict)
        #-------------------------
        return merged_df


    #---------------------------------------------------------------------------
    # TODO still needs work...
    # Typically use OutageMdlrPrep.get_active_SNs_for_xfmrs_in_rcpo_df
    @staticmethod
    def get_active_SNs_for_xfmrs(
        trsf_pole_nbs                  ,     
        df_mp_curr                     , 
        df_mp_hist                     ,
        time_infos_df                  ,     
        time_infos_to_PNs              = ['index'], 
        PNs_to_time_infos              = ['index'], 
        how                            = 'left',     
        output_trsf_pole_nb_col        = None, 
        addtnl_mp_df_curr_cols         = None, 
        addtnl_mp_df_hist_cols         = None, 
        return_SNs_col                 = 'SNs', 
        return_prem_nbs_col            = 'prem_nbs', 
        assert_all_trsf_pole_nbs_found = True, 
        df_mp_serial_number_col        = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col              = 'prem_nb', 
        df_mp_install_time_col         = 'inst_ts', 
        df_mp_removal_time_col         = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col         = 'trsf_pole_nb', 
        t_min_col                      = 't_min', 
        t_max_col                      = 't_max'
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
        assert(
            t_min_col in time_infos_df.columns and 
            t_max_col in time_infos_df.columns
        )
        time_infos_df = time_infos_df[[t_min_col, t_max_col]]
        #-----
        # Remove any duplicates from time_infos_df
        if time_infos_df.index.nlevels==1:
            tmp_col                = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df          = time_infos_df.drop_duplicates()
            time_infos_df          = time_infos_df.drop(columns=[tmp_col])
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
                join_curr_hist          = False, 
                addtnl_mp_df_curr_cols  = addtnl_mp_df_curr_cols, 
                addtnl_mp_df_hist_cols  = addtnl_mp_df_hist_cols, 
                df_mp_serial_number_col = df_mp_serial_number_col, 
                df_mp_prem_nb_col       = df_mp_prem_nb_col, 
                df_mp_install_time_col  = df_mp_install_time_col, 
                df_mp_removal_time_col  = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col  = df_mp_trsf_pole_nb_col
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
            trsf_pole_nbs                  = trsf_pole_nbs, 
            include_SNs                    = False,
            include_PNs                    = True,
            trsf_pole_nb_col               = df_mp_trsf_pole_nb_col, 
            serial_number_col              = df_mp_serial_number_col, 
            prem_nb_col                    = df_mp_prem_nb_col, 
            return_SNs_col                 = None, #Not grabbing SNs
            return_PNs_col                 = return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found = assert_all_trsf_pole_nbs_found, 
            mp_df                          = df_mp_curr, 
            return_mp_df_also              = False
        )
        #-------------------------
        # Join together time_infos_df and PNs_for_xfmrs
        #-----
        time_infos_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df    = time_infos_df, 
            df_2       = PNs_for_xfmrs, 
            rcpo_df_on = time_infos_to_PNs,
            df_2_on    = PNs_to_time_infos, 
            how        = how
        )
        #--------------------------------------------------
        # Only reason for making dict is to ensure trsf_pole_nbs are not repeated 
        active_SNs_in_xfmrs_dfs_dict = {}
        if output_trsf_pole_nb_col is None:
            output_trsf_pole_nb_col='trsf_pole_nb'
        for trsf_pole_nb in trsf_pole_nbs:
            # active_SNs_df_i will have indices equal to premise numbers and value equal to lists
            #   of active SNs for each PN
            PNs_i  = time_infos_df.loc[trsf_pole_nb, return_prem_nbs_col]
            dt_0_i = time_infos_df.loc[trsf_pole_nb, t_min_col]
            dt_1_i = time_infos_df.loc[trsf_pole_nb, t_max_col]
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
                    PNs                                      = PNs_i,
                    df_mp_curr                               = df_mp_curr, 
                    df_mp_hist                               = df_mp_hist, 
                    dt_0                                     = dt_0_i,
                    dt_1                                     = dt_1_i,
                    output_index                             = None,
                    output_groupby                           = [df_mp_prem_nb_col], 
                    include_prems_wo_active_SNs_when_groupby = True, 
                    assert_all_PNs_found                     = False
                )
                active_SNs_df_i = active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col]       = np.nan
                active_SNs_df_i[df_mp_serial_number_col] = [[]] 
                active_SNs_df_i[output_trsf_pole_nb_col] = trsf_pole_nb
                active_SNs_df_i                          = active_SNs_df_i.set_index(output_trsf_pole_nb_col)
            else:
                active_SNs_df_i[output_trsf_pole_nb_col] = trsf_pole_nb
                active_SNs_df_i                          = active_SNs_df_i.explode(df_mp_serial_number_col)
                active_SNs_df_i = Utilities_df.consolidate_df(
                    df                                  = active_SNs_df_i, 
                    groupby_cols                        = [output_trsf_pole_nb_col], 
                    cols_shared_by_group                = None, 
                    cols_to_collect_in_lists            = [df_mp_serial_number_col, df_mp_prem_nb_col], 
                    include_groupby_cols_in_output_cols = False, 
                    allow_duplicates_in_lists           = False, 
                    recover_uniqueness_violators        = True, 
                    rename_cols                         = None, 
                    verbose                             = False
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
            df_mp_prem_nb_col       : return_prem_nbs_col, 
            df_mp_serial_number_col : return_SNs_col
        })
        #-------------------------
        return active_SNs_df


    #---------------------------------------------------------------------------
    # Should probably build OutageMdlrPrep.get_active_SNs_for_xfmrs
    #  which accepts a list of trsf_pole_nbs instead of rcpo_df, which this function can use
    @staticmethod
    def get_active_SNs_for_xfmrs_in_rcpo_df(
        rcpo_df                        , 
        trsf_pole_nbs_loc              , 
        df_mp_curr                     , 
        df_mp_hist                     ,
        time_infos_df                  , 
        rcpo_df_to_time_infos_on       = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on       = ['index'], 
        how                            = 'left', 
        rcpo_df_to_PNs_on              = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on              = ['index'], 
        addtnl_mp_df_curr_cols         = None, 
        addtnl_mp_df_hist_cols         = None, 
        return_SNs_col                 = 'SNs', 
        return_prem_nbs_col            = 'prem_nbs', 
        assert_all_trsf_pole_nbs_found = True, 
        df_mp_serial_number_col        = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col              = 'prem_nb', 
        df_mp_install_time_col         = 'inst_ts', 
        df_mp_removal_time_col         = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col         = 'trsf_pole_nb', 
        t_min_col                      = 't_min', 
        t_max_col                      = 't_max'
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
        assert(
            t_min_col in time_infos_df.columns and 
            t_max_col in time_infos_df.columns
        )
        time_infos_df = time_infos_df[[t_min_col, t_max_col]]
        #-----
        # Remove any duplicates from time_infos_df
        if time_infos_df.index.nlevels==1:
            tmp_col                = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df          = time_infos_df.drop_duplicates()
            time_infos_df          = time_infos_df.drop(columns=[tmp_col])
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
                trsf_pole_nbs_idx_lvl = trsf_pole_nbs_idx_lvl[0]
                trsf_pole_nbs_idx_lvl = int(trsf_pole_nbs_idx_lvl)
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
                join_curr_hist          = False, 
                addtnl_mp_df_curr_cols  = addtnl_mp_df_curr_cols, 
                addtnl_mp_df_hist_cols  = addtnl_mp_df_hist_cols, 
                df_mp_serial_number_col = df_mp_serial_number_col, 
                df_mp_prem_nb_col       = df_mp_prem_nb_col, 
                df_mp_install_time_col  = df_mp_install_time_col, 
                df_mp_removal_time_col  = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col  = df_mp_trsf_pole_nb_col
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
            trsf_pole_nbs                  = trsf_pole_nbs, 
            include_SNs                    = False,
            include_PNs                    = True,
            trsf_pole_nb_col               = df_mp_trsf_pole_nb_col, 
            serial_number_col              = df_mp_serial_number_col, 
            prem_nb_col                    = df_mp_prem_nb_col, 
            return_SNs_col                 = None, #Not grabbing SNs
            return_PNs_col                 = return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found = assert_all_trsf_pole_nbs_found, 
            mp_df                          = df_mp_curr, 
            return_mp_df_also              = False
        )
        #-------------------------
        # Join together rcpo_df, time_infos_df and PNs_for_xfmrs
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df    = rcpo_df, 
            df_2       = time_infos_df, 
            rcpo_df_on = rcpo_df_to_time_infos_on,
            df_2_on    = time_infos_to_rcpo_df_on, 
            how        = how
        )
        #-----
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df    = rcpo_df, 
            df_2       = PNs_for_xfmrs, 
            rcpo_df_on = rcpo_df_to_PNs_on,
            df_2_on    = PNs_to_rcpo_df_on, 
            how        = how
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
            PNs_i  = row_i[return_prem_nbs_col]
            dt_0_i = row_i[t_min_col]
            dt_1_i = row_i[t_max_col]
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
                    PNs                                      = PNs_i,
                    df_mp_curr                               = df_mp_curr, 
                    df_mp_hist                               = df_mp_hist, 
                    dt_0                                     = dt_0_i,
                    dt_1                                     = dt_1_i,
                    output_index                             = None,
                    output_groupby                           = [df_mp_prem_nb_col], 
                    include_prems_wo_active_SNs_when_groupby = True, 
                    assert_all_PNs_found                     = False
                )
                active_SNs_df_i = active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col]       = np.nan
                active_SNs_df_i[df_mp_serial_number_col] = [[]] 
                for name,val in idx_names_w_vals:
                    active_SNs_df_i[name] = val
                active_SNs_df_i = active_SNs_df_i.set_index([x[0] for x in idx_names_w_vals])
            else:
                for name,val in idx_names_w_vals:
                    active_SNs_df_i[name] = val
                active_SNs_df_i = active_SNs_df_i.explode(df_mp_serial_number_col)
                active_SNs_df_i = Utilities_df.consolidate_df(
                    df                                  = active_SNs_df_i, 
                    groupby_cols                        = [x[0] for x in idx_names_w_vals], 
                    cols_shared_by_group                = None, 
                    cols_to_collect_in_lists            = [df_mp_serial_number_col, df_mp_prem_nb_col], 
                    include_groupby_cols_in_output_cols = False, 
                    allow_duplicates_in_lists           = False, 
                    recover_uniqueness_violators        = True, 
                    rename_cols                         = None, 
                    verbose                             = False
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
            df_mp_prem_nb_col       : return_prem_nbs_col, 
            df_mp_serial_number_col : return_SNs_col
        })
        #-------------------------
        return active_SNs_df

    
    #---------------------------------------------------------------------------
    # Should probably build OutageMdlrPrep.get_active_SNs_for_xfmrs
    #  which accepts a list of trsf_pole_nbs instead of rcpo_df, which this function can use
    @staticmethod
    def get_active_SNs_for_xfmrs_in_rcpo_df_v2(
        rcpo_df                        , 
        trsf_pole_nbs_loc              , 
        df_mp_curr                     , 
        df_mp_hist                     ,
        time_infos_df                  , 
        rcpo_df_to_time_infos_on       = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on       = ['index'], 
        how                            = 'left', 
        rcpo_df_to_PNs_on              = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on              = ['index'], 
        addtnl_mp_df_curr_cols         = None, 
        addtnl_mp_df_hist_cols         = None, 
        return_SNs_col                 = 'SNs', 
        return_prem_nbs_col            = 'prem_nbs', 
        assert_all_trsf_pole_nbs_found = True, 
        df_mp_serial_number_col        = 'mfr_devc_ser_nbr', 
        df_mp_prem_nb_col              = 'prem_nb', 
        df_mp_install_time_col         = 'inst_ts', 
        df_mp_removal_time_col         = 'rmvl_ts', 
        df_mp_trsf_pole_nb_col         = 'trsf_pole_nb', 
        t_min_col                      = 't_min', 
        t_max_col                      = 't_max'
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
        assert(
            t_min_col in time_infos_df.columns and 
            t_max_col in time_infos_df.columns
        )
        time_infos_df = time_infos_df[[t_min_col, t_max_col]]
        #-----
        # Remove any duplicates from time_infos_df
        if time_infos_df.index.nlevels==1:
            tmp_col                = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df          = time_infos_df.drop_duplicates()
            time_infos_df          = time_infos_df.drop(columns=[tmp_col])
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
                trsf_pole_nbs_idx_lvl = trsf_pole_nbs_idx_lvl[0]
                trsf_pole_nbs_idx_lvl = int(trsf_pole_nbs_idx_lvl)
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
                join_curr_hist          = False, 
                addtnl_mp_df_curr_cols  = addtnl_mp_df_curr_cols, 
                addtnl_mp_df_hist_cols  = addtnl_mp_df_hist_cols, 
                df_mp_serial_number_col = df_mp_serial_number_col, 
                df_mp_prem_nb_col       = df_mp_prem_nb_col, 
                df_mp_install_time_col  = df_mp_install_time_col, 
                df_mp_removal_time_col  = df_mp_removal_time_col, 
                df_mp_trsf_pole_nb_col  = df_mp_trsf_pole_nb_col
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
            trsf_pole_nbs                  = trsf_pole_nbs, 
            include_SNs                    = False,
            include_PNs                    = True,
            trsf_pole_nb_col               = df_mp_trsf_pole_nb_col, 
            serial_number_col              = df_mp_serial_number_col, 
            prem_nb_col                    = df_mp_prem_nb_col, 
            return_SNs_col                 = None, #Not grabbing SNs
            return_PNs_col                 = return_prem_nbs_col, 
            assert_all_trsf_pole_nbs_found = assert_all_trsf_pole_nbs_found, 
            mp_df                          = df_mp_curr, 
            return_mp_df_also              = False
        )
        #-------------------------
        # Join together rcpo_df, time_infos_df and PNs_for_xfmrs
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df    = rcpo_df, 
            df_2       = time_infos_df, 
            rcpo_df_on = rcpo_df_to_time_infos_on,
            df_2_on    = time_infos_to_rcpo_df_on, 
            how        = how
        )
        #-----
        rcpo_df = OutageMdlrPrep.merge_rcpo_and_df(
            rcpo_df    = rcpo_df, 
            df_2       = PNs_for_xfmrs, 
            rcpo_df_on = rcpo_df_to_PNs_on,
            df_2_on    = PNs_to_rcpo_df_on, 
            how        = how
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
            PNs_i  = row_i[return_prem_nbs_col]
            dt_0_i = row_i[t_min_col]
            dt_1_i = row_i[t_max_col]
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
                    PNs                                      = PNs_i,
                    df_mp_curr                               = df_mp_curr, 
                    df_mp_hist                               = df_mp_hist, 
                    dt_0                                     = dt_0_i,
                    dt_1                                     = dt_1_i,
                    output_index                             = None,
                    output_groupby                           = [df_mp_prem_nb_col], 
                    include_prems_wo_active_SNs_when_groupby = True, 
                    assert_all_PNs_found                     = False
                )
                active_SNs_df_i=active_SNs_df_i.reset_index()
            if active_SNs_df_i.shape[0]==0:
                active_SNs_df_i[df_mp_prem_nb_col]       = np.nan
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
            df                                  = active_SNs_df, 
            groupby_cols                        = [x[0] for x in idx_names_w_vals], 
            cols_shared_by_group                = None, 
            cols_to_collect_in_lists            = [df_mp_serial_number_col, df_mp_prem_nb_col], 
            include_groupby_cols_in_output_cols = False, 
            allow_duplicates_in_lists           = False, 
            recover_uniqueness_violators        = True, 
            rename_cols                         = None, 
            verbose                             = False
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
    @staticmethod
    def add_xfmr_active_SNs_to_rcpo_df(
        rcpo_df                                , 
        trsf_pole_nbs_loc                      , 
        set_xfmr_nSNs                          = True, 
        include_active_xfmr_PNs                = False, 
        df_mp_curr                             = None,
        df_mp_hist                             = None, 
        time_infos_df                          = None, 
        rcpo_df_to_time_infos_on               = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on               = ['index'], 
        how                                    = 'left', 
        rcpo_df_to_PNs_on                      = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on                      = ['index'], 
        addtnl_get_active_SNs_for_xfmrs_kwargs = None, 
        xfmr_SNs_col                           = '_xfmr_SNs', 
        xfmr_nSNs_col                          = '_xfmr_nSNs', 
        xfmr_PNs_col                           = '_xfmr_PNs', 
        xfmr_nPNs_col                          = '_xfmr_nPNs', 
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
            rcpo_df                  = rcpo_df, 
            trsf_pole_nbs_loc        = trsf_pole_nbs_loc, 
            df_mp_curr               = df_mp_curr, 
            df_mp_hist               = df_mp_hist, 
            time_infos_df            = time_infos_df, 
            rcpo_df_to_time_infos_on = rcpo_df_to_time_infos_on, 
            time_infos_to_rcpo_df_on = time_infos_to_rcpo_df_on, 
            how                      = how, 
            rcpo_df_to_PNs_on        = rcpo_df_to_PNs_on, 
            PNs_to_rcpo_df_on        = PNs_to_rcpo_df_on, 
            return_prem_nbs_col      = xfmr_PNs_col, 
            return_SNs_col           = xfmr_SNs_col
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
    @staticmethod
    def build_rcpo_df_norm_by_xfmr_active_nSNs(
        rcpo_df_raw                            , 
        trsf_pole_nbs_loc                      , 
        xfmr_nSNs_col                          = '_xfmr_nSNs', 
        xfmr_SNs_col                           = '_xfmr_SNs', 
        other_SNs_col_tags_to_ignore           = [
            '_SNs', 
            '_nSNs', 
            '_prem_nbs', 
            '_nprem_nbs', 
            '_xfmr_PNs', 
            '_xfmr_nPNs'
        ], 
        drop_xfmr_nSNs_eq_0                    = True, 
        new_level_0_val                        = 'counts_norm_by_xfmr_nSNs', 
        remove_SNs_cols                        = False, 
        df_mp_curr                             = None,
        df_mp_hist                             = None, 
        time_infos_df                          = None,
        rcpo_df_to_time_infos_on               = [('index', 'outg_rec_nb')], 
        time_infos_to_rcpo_df_on               = ['index'], 
        how                                    = 'left', 
        rcpo_df_to_PNs_on                      = [('index', 'trsf_pole_nb')], 
        PNs_to_rcpo_df_on                      = ['index'], 
        addtnl_get_active_SNs_for_xfmrs_kwargs = None
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
            trsf_pole_nbs_loc                      = trsf_pole_nbs_loc, 
            set_xfmr_nSNs                          = True, 
            include_active_xfmr_PNs                = True, 
            df_mp_curr                             = df_mp_curr,
            df_mp_hist                             = df_mp_hist, 
            time_infos_df                          = time_infos_df, 
            rcpo_df_to_time_infos_on               = rcpo_df_to_time_infos_on, 
            time_infos_to_rcpo_df_on               = time_infos_to_rcpo_df_on, 
            how                                    = how, 
            rcpo_df_to_PNs_on                      = rcpo_df_to_PNs_on, 
            PNs_to_rcpo_df_on                      = PNs_to_rcpo_df_on, 
            addtnl_get_active_SNs_for_xfmrs_kwargs = addtnl_get_active_SNs_for_xfmrs_kwargs, 
            xfmr_SNs_col                           = '_xfmr_SNs', 
            xfmr_nSNs_col                          = '_xfmr_nSNs', 
            xfmr_PNs_col                           = '_xfmr_PNs', 
            xfmr_nPNs_col                          = '_xfmr_nPNs', 
        )
        #-------------------------
        other_col_tags_to_ignore = other_SNs_col_tags_to_ignore
        drop_n_counts_eq_0       = drop_xfmr_nSNs_eq_0
        new_level_0_val          = new_level_0_val
        remove_ignored_cols      = remove_SNs_cols
        #-------------------------
        return MECPODf.build_rcpo_df_norm_by_list_counts(
            rcpo_df_raw                    = rcpo_df_raw, 
            n_counts_col                   = n_counts_col, 
            list_col                       = list_col, 
            add_list_col_to_rcpo_df_func   = add_list_col_to_rcpo_df_func, 
            add_list_col_to_rcpo_df_kwargs = add_list_col_to_rcpo_df_kwargs, 
            other_col_tags_to_ignore       = other_col_tags_to_ignore, 
            drop_n_counts_eq_0             = drop_n_counts_eq_0, 
            new_level_0_val                = new_level_0_val, 
            remove_ignored_cols            = remove_ignored_cols
        )


    #---------------------------------------------------------------------------
    @staticmethod
    def get_outg_time_infos_df(
        rcpo_df                       , 
        outg_rec_nb_idx_lvl           , 
        times_relative_to_off_ts_only = True, 
        td_for_left                   = None, 
        td_for_right                  = None
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
            df_construct_type         = DFConstructType.kRunSqlQuery, 
            contstruct_df_args        = None, 
            init_df_in_constructor    = True, 
            build_sql_function        = DOVSOutages_SQL.build_sql_outage, 
            build_sql_function_kwargs = dict(
                outg_rec_nbs     = rcpo_df.index.get_level_values(outg_rec_nb_idx_lvl).tolist(), 
                from_table_alias = 'DOV', 
                datetime_col     = 'DT_OFF_TS_FULL', 
                cols_of_interest = [
                    'OUTG_REC_NB', 
                    dict(
                        field_desc         = f"DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)", 
                        alias              = 'DT_OFF_TS_FULL', 
                        table_alias_prefix = None), 
                    'DT_ON_TS'
                ], 
                field_to_split        = 'outg_rec_nbs'
            )
        )
        #-------------------------
        time_infos_df = dovs_outgs.df
        time_infos_df = Utilities_df.convert_col_type(df=time_infos_df, column='OUTG_REC_NB', to_type=str)
        time_infos_df = time_infos_df.set_index('OUTG_REC_NB')
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
            tmp_col                = Utilities.generate_random_string()
            time_infos_df[tmp_col] = time_infos_df.index
            time_infos_df          = time_infos_df.drop_duplicates()
            time_infos_df          = time_infos_df.drop(columns=[tmp_col])
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
    def build_rcpx_df_from_EndEvents_in_csvs(    
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
            May be None (NOT REALLY NEEDED ANYMORE, see below)

        set_faulty_mp_vals_to_nan & correct_faulty_mp_vals:
            NOT REALLY NEEDED ANYMORE!
                I suggest keeping both of these False.
                It's fine to set True, but run time will likely be negatively affected (especially if many faulty MP values found
                  which need to be corrected)
                -----
                As long as end_events_df was merged with MeterPremise via serial number and premise number, there should be no issue.
                If this was not true, then this function may actually be needed
                -----
                CPXDfBuilder.perform_build_rcpx_from_end_events_in_dir_prereqs all but ensures a merge with MeterPremise will include, at minimum,
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
        prereq_dict = CPXDfBuilder.perform_build_rcpx_from_end_events_in_dir_prereqs(
            files_dir                     = files_dir, 
            file_path_glob                = file_path_glob, 
            file_path_regex               = file_path_regex,
            mp_df                         = mp_df, 
            ede_mp_mismatch_threshold_pct = 1.0, 
            grp_by_cols                   = grp_by_cols, 
            rec_nb_col                    = outg_rec_nb_col,
            trsf_pole_nb_col              = trsf_pole_nb_col, 
            prem_nb_col                   = prem_nb_col, 
            serial_number_col             = serial_number_col,
            mp_df_cols                    = mp_df_cols, 
            assert_all_cols_equal         = assert_all_cols_equal, 
            trust_sql_grouping            = trust_sql_grouping, 
            drop_gpd_for_sql_appendix     = drop_gpd_for_sql_appendix, 
            make_all_columns_lowercase    = make_all_columns_lowercase
        )
        paths             = prereq_dict['paths']
        grp_by_cols       = prereq_dict['grp_by_cols']
        outg_rec_nb_col   = prereq_dict['rec_nb_col']
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
            end_events_df_i = CPXDfBuilder.perform_std_col_renames_and_drops(
                df                         = end_events_df_i, 
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
                end_events_df_i = CPXDfBuilder.set_faulty_mp_vals_to_nan_in_end_events_df(
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
                end_events_df_i = CPXDfBuilder.correct_faulty_mp_vals_in_end_events_df(
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
        rcpo_full = AMI_SQL.rename_all_index_names_with_gpd_for_sql_appendix(rcpo_full)
        #-------------------------
        if not build_ede_typeid_to_reason_df:
            return rcpo_full
        else:
            return rcpo_full, ede_typeid_to_reason_df



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
    @staticmethod
    def build_all_rcpx_dfs_from_EndEvents_in_csvs(
        data_base_dir                        , 
        dataset                              , 
        days_min_outg_td_window              = 0, 
        days_max_outg_td_window              = 31, 
        save_dfs_to_pkl                      = True, 
        read_dfs_from_pkl                    = False, 
        file_path_glob                       = r'end_events_[0-9]*.csv', 
        file_path_regex                      = None, 
        assert_all_cols_equal                = False, 
        inclue_zero_counts                   = True, 
        xfmr_equip_typ_nms_of_interest       = ['TRANSFORMER, OH', 'TRANSFORMER, UG'], 
        mp_df_needed_to_build_rcpx_from_csvs = False, 
        borrow_mp_df_curr_hist_if_exists     = True, 
    ):
        r"""
    
        mp_df_needed_to_build_rcpx_from_csvs:
            In the past, the mp_df data (which was saved during DAQ) were needed for the trsf_pole_nb information
            However, these data are now typically saved with the end_events data themselves (in the csvs), so 
              mp_df is rarely needed anymore
        
        """
        #----------------------------------------------------------------------------------------------------
        files_dir    = os.path.join(data_base_dir, 'EndEvents')
        naming_tag   = ODI.get_naming_tag(dataset)
        is_no_outage = ODI.get_is_no_outage(dataset)
        
        #----------------------------------------------------------------------------------------------------
        # 1. Initialize some variables, e.g., batch_size, grp_by_col, outg_rec_nb_col, save_dir_base_pkls, save_dir_pkls
        #    Create save_dir_pkls if needed
        #--------------------------------------------------
        if is_no_outage:
            batch_size      = 100
            grp_by_col      = ['trsf_pole_nb', 'no_outg_rec_nb']
            outg_rec_nb_col = 'no_outg_rec_nb'
        else:
            batch_size      = 1000
            grp_by_col      = ['outg_rec_nb', 'trsf_pole_nb']
            outg_rec_nb_col = 'outg_rec_nb'
        
        #--------------------------------------------------
        # Currently, expecting grp_by_col to be 'outg_rec_nb', 'trsf_pole_nb', or ['outg_rec_nb', 'trsf_pole_nb']
        #   Actually, 'outg_rec_nb' will probably not be run again, instead will likely always be ['outg_rec_nb', 'trsf_pole_nb']
        assert(
            grp_by_col=='outg_rec_nb' or 
            grp_by_col==['outg_rec_nb', 'trsf_pole_nb'] or 
            grp_by_col=='trsf_pole_nb' or 
            grp_by_col==['trsf_pole_nb', 'no_outg_rec_nb']
        )
        #-----
        # Not possible for have outg_rec_nb for no outage case!
        if is_no_outage:
            assert(grp_by_col!='outg_rec_nb' and grp_by_col!=('outg_rec_nb', 'trsf_pole_nb'))
        #--------------------------------------------------
        save_subdir_pkls = 'rcpo_dfs'
        if   grp_by_col == ['outg_rec_nb', 'trsf_pole_nb']:
            save_subdir_pkls += '_GRP_BY_OUTG_AND_XFMR'
        elif grp_by_col == 'trsf_pole_nb':
            save_subdir_pkls += '_GRP_BY_XFMR'
        elif grp_by_col == 'outg_rec_nb':
            save_subdir_pkls += '_GRP_BY_OUTG'
        elif grp_by_col == ['trsf_pole_nb', 'no_outg_rec_nb']:
            save_subdir_pkls += '_GRP_BY_NO_OUTG_AND_XFMR'
        else:
            assert(0)
        #-----
        save_dir_base_pkls = os.path.join(data_base_dir, save_subdir_pkls)
        save_dir_pkls      = os.path.join(save_dir_base_pkls, f'outg_td_window_{days_min_outg_td_window}_to_{days_max_outg_td_window}_days')
        #-----
        if save_dfs_to_pkl and not os.path.exists(save_dir_pkls):
            os.makedirs(save_dir_pkls)
    
        #----------------------------------------------------------------------------------------------------
        # 2. Grab/build mp_df if needed
        #--------------------------------------------------
        if mp_df_needed_to_build_rcpx_from_csvs:
            mp_df_path      = os.path.join(data_base_dir, r'df_mp_outg_full.pkl')
            mp_df           = pd.read_pickle(mp_df_path)
            merge_on_mp     = ['mfr_devc_ser_nbr', 'prem_nb', 'OUTG_REC_NB']
            mp_df_cols      = dict(
                serial_number_col = 'mfr_devc_ser_nbr', 
                prem_nb_col       = 'prem_nb', 
                trsf_pole_nb_col  = 'trsf_pole_nb', 
                outg_rec_nb_col   = 'OUTG_REC_NB'
            )    
            # Below ensures there is only one entry per 'meter' (meter here is defined by a unique grouping of merge_on_mp)
            if any(mp_df.groupby(merge_on_mp).size()>1):
                print('Resolving uniqueness violators')
                mp_df = Utilities_df.resolve_uniqueness_violators(
                    df                      = mp_df, 
                    groupby_cols            = merge_on_mp, 
                    gpby_dropna             = False,
                    run_nan_groups_separate = True
                )
            assert(not any(mp_df.groupby(merge_on_mp).size()>1))
        else:
            mp_df           = None
            mp_df_cols      = None    
    
        #----------------------------------------------------------------------------------------------------
        # 3. Build rcpo_df and ede_typeid_to_reason_df_OG using OutageMdlrPrep.build_rcpx_df_from_EndEvents_in_csvs
        #--------------------------------------------------
        if not read_dfs_from_pkl:
            start = time.time()
            rcpo_df, ede_typeid_to_reason_df_OG = OutageMdlrPrep.build_rcpx_df_from_EndEvents_in_csvs(    
                files_dir                      = files_dir, 
                mp_df                          = mp_df, 
                file_path_glob                 = file_path_glob, 
                file_path_regex                = file_path_regex, 
                min_outg_td_window             = datetime.timedelta(days=days_min_outg_td_window),
                max_outg_td_window             = datetime.timedelta(days=days_max_outg_td_window),
                build_ede_typeid_to_reason_df  = True, 
                batch_size                     = batch_size, 
                cols_and_types_to_convert_dict = None, 
                to_numeric_errors              = 'coerce', 
                assert_all_cols_equal          = assert_all_cols_equal, 
                include_normalize_by_nSNs      = True, 
                inclue_zero_counts             = inclue_zero_counts, 
                return_multiindex_outg_reason  = False, 
                return_normalized_separately   = False, 
                verbose                        = True, 
                n_update                       = 1, 
                grp_by_cols                    = grp_by_col, 
                outg_rec_nb_col                = outg_rec_nb_col,
                trsf_pole_nb_col               = 'trsf_pole_nb', 
                addtnl_dropna_subset_cols      = None, 
                is_no_outage                   = is_no_outage, 
                prem_nb_col                    = 'aep_premise_nb', 
                serial_number_col              = 'serialnumber', 
                include_prem_nbs               = True, 
                set_faulty_mp_vals_to_nan      = False,
                correct_faulty_mp_vals         = False, 
                trust_sql_grouping             = True, 
                drop_gpd_for_sql_appendix      = True, 
                mp_df_cols                     = mp_df_cols, 
                make_all_columns_lowercase     = True, 
                date_only                      = False
            )
            print(f"Runtime for OutageMdlrPrep.build_rcpx_df_from_EndEvents_in_csvs: {time.time()-start}")
            #-------------------------
            if save_dfs_to_pkl:
                rcpo_df.to_pickle(                   os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_OG.pkl'))
                ede_typeid_to_reason_df_OG.to_pickle(os.path.join(save_dir_pkls, f'ede_typeid_to_reason{naming_tag}_df_OG.pkl'))        
        else:
            rcpo_df                    = pd.read_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_OG.pkl'))
            ede_typeid_to_reason_df_OG = pd.read_pickle(os.path.join(save_dir_pkls, f'ede_typeid_to_reason{naming_tag}_df_OG.pkl'))
        reason_to_ede_typeid_df = AMIEndEvents.invert_ede_typeid_to_reason_df(ede_typeid_to_reason_df_OG)
    
        #----------------------------------------------------------------------------------------------------
        # 4. Append time information to rcpo_df using either:
        #     -  Build no_outg_time_infos_df for baseline data
        #     -  Utilize DOVSOutages.append_outg_info_to_df for signal data
        #--------------------------------------------------
        time_cols_prefix = 'dummy_lvl_'
        if is_no_outage:
            # Build no_outg_time_infos_df, which has prem_nbs indices and t_min, t_max (and possible summary_path) columns
            # This is where the time information for each premise number comes from
            paths = Utilities.find_all_paths(
                base_dir     = files_dir, 
                glob_pattern = file_path_glob
            )
            #-----
            no_outg_time_infos_df = MECPOAn.get_bsln_time_interval_infos_df_from_summary_files(
                summary_paths           = [AMIEndEvents.find_summary_file_from_csv(x) for x in paths], 
                output_prem_nbs_col     = 'prem_nbs', 
                output_t_min_col        = 't_min', 
                output_t_max_col        = 't_max', 
                make_addtnl_groupby_idx = True, 
                include_summary_paths   = True
            )
            #-------------------------
            rcpo_df = Utilities_df.merge_dfs(
                df_1        = rcpo_df, 
                df_2        = no_outg_time_infos_df, 
                merge_on_1  = [
                    ('index', 'trsf_pole_nb'), 
                    ('index', 'no_outg_rec_nb')
                ], 
                merge_on_2  = [
                    ('index', 'trsf_pole_nb'), 
                    ('index', 'no_outg_rec_nb')
                ], 
                how         = 'left', 
                final_index = 1, 
                dummy_col_levels_prefix = time_cols_prefix
            )
        else:
            rcpo_df = DOVSOutages.append_outg_info_to_df(
                df                        = rcpo_df, 
                outg_rec_nb_idfr          = 'index', 
                contstruct_df_args        = None, 
                build_sql_function        = None, 
                build_sql_function_kwargs = None, 
                dummy_col_levels_prefix   = time_cols_prefix
            )
        #-------------------------    
        assert(rcpo_df.columns.nlevels==2)
        time_cols_lvl_0_val = f'{time_cols_prefix}0'
        assert(time_cols_lvl_0_val in rcpo_df.columns.get_level_values(0))
    
        #----------------------------------------------------------------------------------------------------
        # 5. Build mp_df_curr_hist, which is needed to add active SNs/PNs/etc.
        #--------------------------------------------------
        # TODO NEED TO BE AUTOMATED
        if is_no_outage:
            outg_rec_nb_idx_lvl      =-1
            trsf_pole_nbs_idx_lvl    = 0
            
            trsf_pole_nbs_loc        = ('index', 'trsf_pole_nb')
            rcpo_df_to_time_infos_on = [('index', 'trsf_pole_nb'), ('index', 'no_outg_rec_nb')]
            time_infos_to_rcpo_df_on = [('index', 'trsf_pole_nb'), ('index', 'no_outg_rec_nb')]
            
            rcpo_df_to_PNs_on        = ['index']
            PNs_to_rcpo_df_on        = ['index']
            how                      = 'left'
        else:
            outg_rec_nb_idx_lvl      = 0
            trsf_pole_nbs_idx_lvl    = 1
            
            trsf_pole_nbs_loc        = ('index', 'trsf_pole_nb')
            rcpo_df_to_time_infos_on = [('index', 'outg_rec_nb')]
            time_infos_to_rcpo_df_on = ['index']
            
            rcpo_df_to_PNs_on        = [('index', 'trsf_pole_nb')]
            PNs_to_rcpo_df_on        = ['index']
            how                      = 'left'
        
        #--------------------------------------------------
        mp_df_curr_hist = None
        if borrow_mp_df_curr_hist_if_exists:
            mp_df_curr_hist = OutageMdlrPrep.try_to_steal_mp_df_curr_hist_from_previous(
                save_dir_base_pkls      = save_dir_base_pkls, 
                days_min_outg_td_window = days_min_outg_td_window, 
                days_max_outg_td_window = days_max_outg_td_window, 
                subdir_regex            = r'^outg_td_window_(\d*)_to_(\d*)_days$', 
                naming_tag              = naming_tag, 
                verbose                 = True, 
            )
        
        #--------------------------------------------------
        # If borrow_mp_df_curr_hist_if_exists==False OR if True but no appropriate mp_df_curr_hist was found
        #   then mp_df_curr_hist must be built!
        # In either case, mp_df_curr_hist will be None
        if mp_df_curr_hist is None:
            #--------------------------------------------------
            # NOTE: Need mp_df_curr and mp_df_hist separate for functionality, so one cannot simply use mp_df loaded earlier.
            # NOTE: drop_approx_duplicates=False below. These will be dropped later
            #-----
            if grp_by_col=='outg_rec_nb':
                mp_df_curr_hist = DOVSOutages.build_mp_df_curr_hist_for_outgs(
                    outg_rec_nbs            = rcpo_df.index.get_level_values(outg_rec_nb_idx_lvl).tolist(), 
                    join_curr_hist          = False, 
                    addtnl_mp_df_curr_cols  = None, 
                    addtnl_mp_df_hist_cols  = None, 
                    df_mp_serial_number_col = 'mfr_devc_ser_nbr', 
                    df_mp_prem_nb_col       = 'prem_nb', 
                    df_mp_install_time_col  = 'inst_ts', 
                    df_mp_removal_time_col  = 'rmvl_ts', 
                    df_mp_trsf_pole_nb_col  = 'trsf_pole_nb'
                )
            else:
                mp_df_curr_hist = MeterPremise.build_mp_df_curr_hist_for_xfmrs(
                    trsf_pole_nbs               = rcpo_df.index.get_level_values(trsf_pole_nbs_idx_lvl).tolist(), 
                    join_curr_hist              = False, 
                    addtnl_mp_df_curr_cols      = None, 
                    addtnl_mp_df_hist_cols      = None, 
                    assume_one_xfmr_per_PN      = True, 
                    drop_approx_duplicates      = False, 
                    drop_approx_duplicates_args = None, 
                    df_mp_serial_number_col     = 'mfr_devc_ser_nbr', 
                    df_mp_prem_nb_col           = 'prem_nb', 
                    df_mp_install_time_col      = 'inst_ts', 
                    df_mp_removal_time_col      = 'rmvl_ts', 
                    df_mp_trsf_pole_nb_col      = 'trsf_pole_nb'
                )
        
        #--------------------------------------------------
        # Make sure all dates are datetime objects, not e.g., strings
        mp_df_curr_hist = Utilities_df.ensure_dt_cols(
            df      = mp_df_curr_hist, 
            dt_cols = ['inst_ts', 'rmvl_ts']
        )
        
        #--------------------------------------------------
        # I don't think I want to do the removal on current, only hist!
        # This is because current is used for get_SNs_andor_PNs_for_xfmrs
        # e.g., I was missing some PNs because maybe a new meter was installed after rcpo_df['DT_OFF_TS_FULL'].max()
        #   So, in all likelihood that was an appropriate meter entry in historical, but this was excluded because
        #   there wasn't an entry in current that passed the cuts below
        #-----
        if is_no_outage:
            t_min_col = 't_min'
            t_max_col = 't_max'
        else:
            t_min_col = 'DT_OFF_TS_FULL'
            t_max_col = 'DT_OFF_TS_FULL'
            # t_max_col = 'DT_ON_TS'
        #-----    
        rcpo_df_t_min = rcpo_df[(time_cols_lvl_0_val, t_min_col)].min()
        rcpo_df_t_max = rcpo_df[(time_cols_lvl_0_val, t_max_col)].max()
        
        #--------------------------------------------------
        mp_df_curr_hist = OutageMdlrPrep.rough_time_slice_and_drop_dups_mp_df_curr_hist(
            mp_df_curr_hist         = mp_df_curr_hist, 
            t_min                   = rcpo_df_t_min, 
            t_max                   = rcpo_df_t_max, 
            df_mp_serial_number_col = 'mfr_devc_ser_nbr', 
            df_mp_prem_nb_col       = 'prem_nb', 
            df_mp_install_time_col  = 'inst_ts', 
            df_mp_removal_time_col  = 'rmvl_ts', 
            df_mp_trsf_pole_nb_col  = 'trsf_pole_nb',
        )
        
        #--------------------------------------------------
        if save_dfs_to_pkl:
            mp_df_curr_hist['mp_df_hist'].to_pickle(os.path.join(save_dir_pkls, f'mp{naming_tag}_df_hist.pkl'))
            mp_df_curr_hist['mp_df_curr'].to_pickle(os.path.join(save_dir_pkls, f'mp{naming_tag}_df_curr.pkl'))
    
        #----------------------------------------------------------------------------------------------------
        # 6. Get/build time_infos_df if needed
        #    Need time_infos_df only if grp_by_col!='outg_rec_nb'
        #--------------------------------------------------
        if grp_by_col!='outg_rec_nb':
            if not read_dfs_from_pkl:
                if is_no_outage:
                    paths = Utilities.find_all_paths(
                        base_dir     = files_dir, 
                        glob_pattern = file_path_glob
                    )
                    #-----
                    time_infos_df = MECPOAn.get_bsln_time_interval_infos_df_from_summary_files(
                        summary_paths           = [AMIEndEvents.find_summary_file_from_csv(x) for x in paths], 
                        output_prem_nbs_col     = 'prem_nbs', 
                        output_t_min_col        = 't_min', 
                        output_t_max_col        = 't_max', 
                        make_addtnl_groupby_idx = True, 
                        include_summary_paths   = False
                    )
                else:
                    time_infos_df = OutageMdlrPrep.get_outg_time_infos_df(
                        rcpo_df                       = rcpo_df, 
                        outg_rec_nb_idx_lvl           = outg_rec_nb_idx_lvl, 
                        times_relative_to_off_ts_only = True, 
                        td_for_left                   = None, 
                        td_for_right                  = None
                    )
                #-------------------------
                if save_dfs_to_pkl:
                    time_infos_df.to_pickle(os.path.join(save_dir_pkls, f'time_infos{naming_tag}_df.pkl'))
            else:
                time_infos_df = pd.read_pickle(os.path.join(save_dir_pkls, f'time_infos{naming_tag}_df.pkl'))
    
        #----------------------------------------------------------------------------------------------------
        # 7. Build all other rcpo_dfs/icpo_dfs
        #--------------------------------------------------
        if not read_dfs_from_pkl:
            start=time.time()
            #--------------------------------------------------
            rcpo_df_raw = MECPODf.project_level_0_columns_from_rcpo_wide(
                rcpo_df_wide = rcpo_df, 
                level_0_val  = 'counts', 
                droplevel    = True
            )
            #-----
            if grp_by_col=='outg_rec_nb':
                rcpo_df_raw = MECPODf.add_outage_active_SNs_to_rcpo_df(
                    rcpo_df                    = rcpo_df_raw, 
                    set_outage_nSNs            = True, 
                    include_outage_premise_nbs = True, 
                    df_mp_curr                 = mp_df_curr_hist['mp_df_curr'], 
                    df_mp_hist                 = mp_df_curr_hist['mp_df_hist']
                )
                #-----
                rcpo_df_raw = MECPODf.add_active_prim_SNs_to_rcpo_df(
                    rcpo_df                             = rcpo_df_raw, 
                    direct_SNs_in_outgs_df              = None, 
                    outg_rec_nb_col                     = 'index', 
                    prim_SNs_col                        = 'direct_serial_numbers', 
                    set_prim_nSNs                       = True, 
                    sort_SNs                            = True, 
                    build_direct_SNs_in_outgs_df_kwargs = {}, 
                    mp_df_curr                          = mp_df_curr_hist['mp_df_curr'], 
                    mp_df_hist                          = mp_df_curr_hist['mp_df_hist']
                )
            else:
                rcpo_df_raw = OutageMdlrPrep.add_xfmr_active_SNs_to_rcpo_df(
                    rcpo_df                                = rcpo_df_raw, 
                    trsf_pole_nbs_loc                      = trsf_pole_nbs_loc, 
                    set_xfmr_nSNs                          = True, 
                    include_active_xfmr_PNs                = True, 
                    df_mp_curr                             = mp_df_curr_hist['mp_df_curr'],
                    df_mp_hist                             = mp_df_curr_hist['mp_df_hist'], 
                    time_infos_df                          = time_infos_df, 
                    rcpo_df_to_time_infos_on               = rcpo_df_to_time_infos_on, 
                    time_infos_to_rcpo_df_on               = time_infos_to_rcpo_df_on, 
                    how                                    = how, 
                    rcpo_df_to_PNs_on                      = rcpo_df_to_PNs_on, 
                    PNs_to_rcpo_df_on                      = PNs_to_rcpo_df_on, 
                    addtnl_get_active_SNs_for_xfmrs_kwargs = dict(
                        assert_all_trsf_pole_nbs_found=False
                    ), 
                    xfmr_SNs_col                           = '_xfmr_SNs', 
                    xfmr_nSNs_col                          = '_xfmr_nSNs', 
                    xfmr_PNs_col                           = '_xfmr_PNs', 
                    xfmr_nPNs_col                          = '_xfmr_nPNs',  
                )
            #-------------------------
            icpo_df_raw = MECPODf.convert_rcpo_to_icpo_df(
                rcpo_df                 = rcpo_df_raw, 
                reason_to_ede_typeid_df = reason_to_ede_typeid_df, 
                is_norm                 = False
            )
            #-------------------------
            if save_dfs_to_pkl:
                #-------------------------
                rcpo_df_raw.to_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_raw.pkl'))
                icpo_df_raw.to_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_raw.pkl'))
        
            
            #--------------------------------------------------
            rcpo_df_norm = MECPODf.project_level_0_columns_from_rcpo_wide(
                rcpo_df_wide = rcpo_df, 
                level_0_val  = 'counts_norm', 
                droplevel    = True
            )
            #-------------------------
            icpo_df_norm = MECPODf.convert_rcpo_to_icpo_df(
                rcpo_df                 = rcpo_df_norm, 
                reason_to_ede_typeid_df = reason_to_ede_typeid_df, 
                is_norm                 = True, 
                counts_col              = '_nSNs'
            )
            #-------------------------
            if save_dfs_to_pkl:
                #-------------------------
                rcpo_df_norm.to_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm.pkl'))
                icpo_df_norm.to_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm.pkl'))
        
            #--------------------------------------------------
            if grp_by_col=='outg_rec_nb':
                #--------------------------------------------------
                rcpo_df_norm_by_outg_nSNs = MECPODf.build_rcpo_df_norm_by_outg_active_nSNs(
                    rcpo_df_raw, 
                    df_mp_curr = mp_df_curr_hist['mp_df_curr'], 
                    df_mp_hist = mp_df_curr_hist['mp_df_hist']
                )
                #-------------------------
                icpo_df_norm_by_outg_nSNs = MECPODf.convert_rcpo_to_icpo_df(
                    rcpo_df                 = rcpo_df_norm_by_outg_nSNs, 
                    reason_to_ede_typeid_df = reason_to_ede_typeid_df, 
                    is_norm                 = True, 
                    counts_col              = '_outg_nSNs'
                )
                #-------------------------
                if save_dfs_to_pkl:
                    #-------------------------
                    rcpo_df_norm_by_outg_nSNs.to_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm_by_outg_nSNs.pkl'))
                    icpo_df_norm_by_outg_nSNs.to_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm_by_outg_nSNs.pkl'))
                
        
                #--------------------------------------------------
                rcpo_df_norm_by_prim_nSNs = MECPODf.build_rcpo_df_norm_by_prim_active_nSNs(
                    rcpo_df_raw                         = rcpo_df_raw, 
                    direct_SNs_in_outgs_df              = None, 
                    outg_rec_nb_col                     = 'index', 
                    prim_nSNs_col                       = '_prim_nSNs', 
                    prim_SNs_col                        = '_prim_SNs', 
                    other_SNs_col_tags_to_ignore        = ['_SNs', '_nSNs', '_prem_nbs', '_nprem_nbs'], 
                    drop_prim_nSNs_eq_0                 = True, 
                    new_level_0_val                     = 'counts_norm_by_prim_nSNs', 
                    remove_SNs_cols                     = False, 
                    build_direct_SNs_in_outgs_df_kwargs = dict(
                        equip_typ_nms_of_interest = xfmr_equip_typ_nms_of_interest
                    ), 
                    df_mp_curr                          = mp_df_curr_hist['mp_df_curr'], 
                    df_mp_hist                          = mp_df_curr_hist['mp_df_hist']
                )
                #-------------------------
                icpo_df_norm_by_prim_nSNs = MECPODf.convert_rcpo_to_icpo_df(
                    rcpo_df                 = rcpo_df_norm_by_prim_nSNs, 
                    reason_to_ede_typeid_df = reason_to_ede_typeid_df, 
                    is_norm                 = True, 
                    counts_col              = '_prim_nSNs'
                )
                #-------------------------
                if save_dfs_to_pkl:
                    #-------------------------
                    rcpo_df_norm_by_prim_nSNs.to_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm_by_prim_nSNs.pkl'))
                    icpo_df_norm_by_prim_nSNs.to_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm_by_prim_nSNs.pkl'))
            else:
                #--------------------------------------------------
                rcpo_df_norm_by_xfmr_nSNs = OutageMdlrPrep.build_rcpo_df_norm_by_xfmr_active_nSNs(
                    rcpo_df_raw                            = rcpo_df_raw, 
                    trsf_pole_nbs_loc                      = trsf_pole_nbs_loc, 
                    xfmr_nSNs_col                          = '_xfmr_nSNs', 
                    xfmr_SNs_col                           = '_xfmr_SNs', 
                    other_SNs_col_tags_to_ignore           = ['_SNs', '_nSNs', '_prem_nbs', '_nprem_nbs', '_xfmr_PNs', '_xfmr_nPNs'], 
                    drop_xfmr_nSNs_eq_0                    = True, 
                    new_level_0_val                        = 'counts_norm_by_xfmr_nSNs', 
                    remove_SNs_cols                        = False, 
                    df_mp_curr                             = mp_df_curr_hist['mp_df_curr'],
                    df_mp_hist                             = mp_df_curr_hist['mp_df_hist'], 
                    time_infos_df                          = time_infos_df,
                    rcpo_df_to_time_infos_on               = rcpo_df_to_time_infos_on, 
                    time_infos_to_rcpo_df_on               = time_infos_to_rcpo_df_on, 
                    how                                    = how, 
                    rcpo_df_to_PNs_on                      = rcpo_df_to_PNs_on, 
                    PNs_to_rcpo_df_on                      = PNs_to_rcpo_df_on, 
                    addtnl_get_active_SNs_for_xfmrs_kwargs = dict(
                        assert_all_trsf_pole_nbs_found=False
                    ), 
                )
                #-------------------------
                icpo_df_norm_by_xfmr_nSNs = MECPODf.convert_rcpo_to_icpo_df(
                    rcpo_df                 = rcpo_df_norm_by_xfmr_nSNs, 
                    reason_to_ede_typeid_df = reason_to_ede_typeid_df, 
                    is_norm                 = True, 
                    counts_col              = '_xfmr_nSNs'
                )
                #-------------------------
                if save_dfs_to_pkl:
                    #-------------------------
                    rcpo_df_norm_by_xfmr_nSNs.to_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm_by_xfmr_nSNs.pkl'))
                    icpo_df_norm_by_xfmr_nSNs.to_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm_by_xfmr_nSNs.pkl'))
            
            #-------------------------
            print(f"Time to build all other rcpo_dfs = {time.time()-start}")
            #-------------------------
        else:
            rcpo_df_raw               = pd.read_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_raw.pkl'))
            icpo_df_raw               = pd.read_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_raw.pkl'))
            #-----
            rcpo_df_norm              = pd.read_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm.pkl'))
            icpo_df_norm              = pd.read_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm.pkl'))
        
            if grp_by_col=='outg_rec_nb':
                rcpo_df_norm_by_outg_nSNs = pd.read_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm_by_outg_nSNs.pkl'))
                icpo_df_norm_by_outg_nSNs = pd.read_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm_by_outg_nSNs.pkl'))
                #-----
                rcpo_df_norm_by_prim_nSNs = pd.read_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm_by_prim_nSNs.pkl'))
                icpo_df_norm_by_prim_nSNs = pd.read_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm_by_prim_nSNs.pkl'))
            else:
                rcpo_df_norm_by_xfmr_nSNs = pd.read_pickle(os.path.join(save_dir_pkls, f'rcpo{naming_tag}_df_norm_by_xfmr_nSNs.pkl'))
                icpo_df_norm_by_xfmr_nSNs = pd.read_pickle(os.path.join(save_dir_pkls, f'icpo{naming_tag}_df_norm_by_xfmr_nSNs.pkl'))


    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    @staticmethod
    def check_with_and_conform_to_data_structure_df(
        rcpx_df                     , 
        data_structure_df           , 
        other_reasons_col           = 'Other Reasons', 
        total_counts_col            = 'total_counts', 
        nSNs_col                    = '_nSNs', 
        non_reason_lvl_0_vals       = ['EEMSP_0', 'time_info_0', '_nSNs'], 
        col_lvl_0_vals_to_ignore    = None, 
        enforce_exact_match         = False, 
        XNs_tags                    = None, 
        is_norm                     = False, 
        verbose                     = True, 
    ):
        r"""
        In general, not all curated reasons will be included in the model.
        Typically, 10 commong curated reasons will be included, and all others will be grouped together in "Other Reasons".
        Furthermore, some reasons may be combined together, others may be completely removed.
        For these reasons, it is beneficial to have some sample data (taken from when the model was created) to utilize 
            in structuring the new data in the same fashion.
        Additionally, the data will be used to ensure the ordering of columns is correct before the data are fed into 
            the model.

        data_structure_df:
            Used to determine the set of final columns desired in rcpx_0_pd_i, and the method
              MECPODf.get_reasons_subset_from_cpo_df will be used to adjust rcpx_0_pd_i accordingly.
        """
        #--------------------------------------------------
        assert(isinstance(data_structure_df, pd.DataFrame) and data_structure_df.shape[0]>0)
        assert(data_structure_df.columns.nlevels==2)
        #-------------------------
        if enforce_exact_match:
            col_lvl_0_vals_to_ignore = None
        #-------------------------
        if col_lvl_0_vals_to_ignore is not None:
            col_lvl_0_vals_to_ignore = list(set(col_lvl_0_vals_to_ignore).intersection(set(data_structure_df.columns.get_level_values(0))))
            data_structure_df        = data_structure_df.drop(columns=col_lvl_0_vals_to_ignore, level=0)
        #-------------------------
        # Get the expected columns for this time period from data_structure_df
        if other_reasons_col in data_structure_df.columns.get_level_values(1):
            combine_others    = True
            final_reason_cols = data_structure_df.drop(columns=[other_reasons_col], level=1).columns.tolist()
        else:
            combine_others    = False
            final_reason_cols = data_structure_df.columns.tolist()
        #-----
        reason_lvl_0_vals = list(set(data_structure_df.columns.get_level_values(0)).difference(set(non_reason_lvl_0_vals)))
        if nSNs_col in data_structure_df[reason_lvl_0_vals].columns.get_level_values(1):
            include_counts_col_in_output    = True
        else:
            include_counts_col_in_output    = False
        #-------------------------
        # Make sure rcpx_df contains the expected final reason columns.
        # Once this is assured, project out these reasons and combine all other reasons into
        #   the other_reasons_col columns
        # See MECPODf.get_reasons_subset_from_cpo_df for more info
        #-----
        assert(len(set(final_reason_cols).difference(set(rcpx_df.columns.tolist())))==0)
        rcpx_df = CPXDf.get_reasons_subset_from_cpx_df(
            cpx_df                       = rcpx_df, 
            reasons_to_include           = final_reason_cols,
            combine_others               = combine_others, 
            output_combine_others_col    = other_reasons_col, 
            XNs_tags                     = XNs_tags, 
            is_norm                      = is_norm, 
            counts_col                   = nSNs_col, 
            cols_to_ignore               = [total_counts_col], 
            include_counts_col_in_output = include_counts_col_in_output, 
            non_reason_lvl_0_vals        = non_reason_lvl_0_vals, 
            verbose                      = verbose
        )
        assert(set(data_structure_df.columns).difference(set(rcpx_df.columns))==set())
        if enforce_exact_match:
            rcpx_df = rcpx_df[data_structure_df.columns].copy()
            assert(set(data_structure_df.columns).symmetric_difference(set(rcpx_df.columns))==set())
        #--------------------------------------------------
        return rcpx_df


    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
