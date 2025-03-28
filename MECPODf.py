#!/usr/bin/env python

r"""
Holds MECPODf class.  See MECPODf.MECPODf for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys

import pandas as pd
import numpy as np
from enum import IntEnum
import copy
#--------------------------------------------------
from AMIEndEvents import AMIEndEvents
from DOVSOutages import DOVSOutages
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
import DataFrameSubsetSlicer
from DataFrameSubsetSlicer import DataFrameSubsetSlicer as DFSlicer
#--------------------------------------------------


#----------------------------------------------------------------------------------------------------
class OutageDType(IntEnum):
    r"""
    Enum class for outage data types, which can be:
        outg = OUTaGe data
        otbl = Outage Transformers BaseLine
        prbl = PRistine BaseLine
    """
    outg  = 0
    otbl  = 1
    prbl  = 2
    unset = 3

#----------------------------------------------------------------------------------------------------
class MECPODf:
    r"""
    MECPODf = Meter Event Counts Per Outage DataFrame
    Class to hold methods for building/manipulating Reason Counts Per Outage (RCPO) and Id (enddeviceeventtypeid) Counts Per Outage (ICPO) DataFrames.
    This could probably be a module instead of a class, as it will contain only static methods, but I created it as a class in case I want to add 
      instance specific methods/attributes later on.
      ALSO, creating it as a class makes it easier to develop/test/tweak any functions, as the function call syntax will always be MECPODf.function, 
        as opposed to the module case, where in the module the syntax would be function whereas outside it would be MECPODf.function
    """
    
    def __init__(
        self
    ):
        self.cpo_df = pd.DataFrame()
        
    @staticmethod
    def std_SNs_cols():
        # NOTE: Eventually all RCPO methods will be brough over from AMIEndEvents.
        #       When that occurs, used commented lines below instead
        #std_cols = ['_SNs', '_prem_nbs', '_outg_SNs', '_outg_prem_nbs', '_prim_SNs', '_xfmr_SNs', '_xfmr_PNs']
        #return std_cols
        return AMIEndEvents.std_SNs_cols()
        
    @staticmethod
    def std_nSNs_cols():
        # NOTE: Eventually all RCPO methods will be brough over from AMIEndEvents.
        #       When that occurs, used commented lines below instead
        #std_cols = ['_nSNs', '_nprem_nbs', '_outg_nSNs', '_outg_nprem_nbs', '_prim_nSNs', '_xfmr_nSNs', '_xfmr_nPNs']
        #return std_cols
        return AMIEndEvents.std_nSNs_cols()

    #****************************************************************************************************
    # MOVED TO CPXDf
    @staticmethod
    def project_level_0_columns_from_rcpo_wide(rcpo_df_wide, level_0_val, droplevel=False):
        r"""
        This is kind of a pain to write all the time, hence this function.
        Intended to be used with a reason_counts_per_outage (rcpo) DataFrame which has both raw and normalized values.
        In this case, the columns of the DF are MultiIndex, with the level 0 values raw or normalized (typically 'counts' or
        'counts_norm'), and the level 1 values are the reasons.
        """
        #-------------------------
        assert(rcpo_df_wide.columns.nlevels==2)
        assert(level_0_val in rcpo_df_wide.columns.get_level_values(0).unique())
        #-------------------------
        cols_to_project = rcpo_df_wide.columns[rcpo_df_wide.columns.get_level_values(0)==level_0_val]
        return_df = rcpo_df_wide[cols_to_project]
        if droplevel:
            return_df = return_df.droplevel(0, axis=1)
        return return_df

    @staticmethod
    def set_nSNs_from_SNs_in_rcpo_df(rcpo_df_wide, SNs_col, nSNs_col):
        r"""
        !!!!!!!!!!
        NOTE: Doing something similar for the case of long-form DFs is not so simple.  
        If needed, I suggest probably converting to wide-form, running this code, then converting back
        """
        #-------------------------
        if nSNs_col not in rcpo_df_wide.columns:
            rcpo_df_wide[nSNs_col] = 0
        #-------------------------
        assert(SNs_col in rcpo_df_wide.columns)
        rcpo_df_wide[nSNs_col] = rcpo_df_wide[SNs_col].apply(len)
        return rcpo_df_wide
        
    @staticmethod
    def sort_list_with_nans(lst, put_nan_at_front=True):
        r"""
        Built for use mainly in MECPODf.sort_SNs_in_rcpo_df_wide
        If nan is included in, e.g., SNs list, the sorting operation will fail.
        Therefore, the solution is to take out the NaNs, sort, then place NaNs back in
        """
        #-------------------------
        org_len = len(lst)
        lst = sorted([x for x in lst if pd.notna(x)])
        n_nan_to_add = org_len - len(lst)
        if put_nan_at_front:
            lst = [np.nan]*n_nan_to_add + lst
        else:
            lst = lst + [np.nan]*n_nan_to_add
        #-------------------------
        return lst

    @staticmethod
    def sort_SNs_in_rcpo_df_wide(rcpo_df_wide, SNs_col):
        try:
            rcpo_df_wide[SNs_col] = rcpo_df_wide[SNs_col].apply(sorted)
        except:
            rcpo_df_wide[SNs_col] = rcpo_df_wide[SNs_col].apply(lambda x: MECPODf.sort_list_with_nans(lst=x, put_nan_at_front=True))
        return rcpo_df_wide

    @staticmethod
    def find_SNs_cols_and_sort(
        rcpo_df, 
        SNs_tags=None
    ):
        r"""
        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() when SNs_tags is None
        NOTE: SNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols()
        #-------------------------
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            level_idx=0
        else:
            level_idx=1
        #-------------------------
        tagged_idxs = Utilities.find_tagged_idxs_in_list(
            lst=rcpo_df.columns.get_level_values(level_idx).tolist(), 
            tags=SNs_tags)
        cols_to_sort = rcpo_df.columns[tagged_idxs]
        #-------------------------
        for col in cols_to_sort:
            rcpo_df = MECPODf.sort_SNs_in_rcpo_df_wide(
                rcpo_df_wide=rcpo_df, 
                SNs_col=col
            )
        return rcpo_df
        
    @staticmethod
    def get_existing_full_rcpo_col(
        col_str, 
        rcpo_df, 
        max_nlevels=2, 
        assert_single_unique_value_for_lower_levels=True
    ):
        r"""
        Returns the full column name for col_str.
        If the columns of rcpo_df are normal Index items, then this simply returns col_str.
        If the columns of rcpo_df are MultiIndex objects, this will return (level_0, col_str)

        Designed for the case where rcpo_df has at most two levels, and level 0 has a single unique value.
        Will work with DFs having higher order column levels, and will work if lower levels have more than
          one unique value per level.
        HOWEVER!!!!!
          If the lower levels have more than one unique value per level, there can ONLY BE ONE INSTANCE of col_str
          in the highest level.  Otherwise, there would be degeneracy, and it would be uncertain which column was desired
            e.g., consider a DF with columns ('counts', 'Reason_1') and ('counts_norm', 'Reason_1').
                    If col_str='Reason_1', there would be no way of knowing which of these to return!
                  However, if instead we had a DF with columns ('counts', 'Reason_1') and ('counts_norm', 'Reason_2'), then
                    there is no ambiguity, and there is also no issue with level 0 having more than one unique value!
        """
        #-------------------------
        n_levels = rcpo_df.columns.nlevels
        assert(n_levels<=max_nlevels)
        #-------------------------
        # If only one level, simply return col_str (assuming it is found in rcpo_df.columns!)
        if n_levels==1:
            assert(col_str in rcpo_df.columns)
            return col_str
        #-------------------------
        if assert_single_unique_value_for_lower_levels:
            for i_level in range(n_levels-1):
                assert(rcpo_df.columns.get_level_values(i_level).nunique()==1)
        #-------------------------
        highest_level_col_values = rcpo_df.columns.get_level_values(n_levels-1).tolist()
        assert(col_str in highest_level_col_values)
        #-----
        found_idxs = Utilities.find_tagged_idxs_in_list(highest_level_col_values, [col_str+'_EXACTMATCH'])
        # There can only be one found, otherwise there is ambiguity!
        assert(len(found_idxs)==1)
        return_col = rcpo_df.columns[found_idxs[0]]
        #-------------------------
        return return_col

    @staticmethod
    def build_full_rcpo_col(
        col_str, 
        rcpo_df, 
        max_nlevels=2
    ):
        r"""
        If the columns of rcpo_df are MultiIndex, this will essentially build a tuple containing the lower level column values
          and ending with col_str.
        This is different from get_existing_full_rcpo_col in that col_str does not need to already exist in rcpo_df.
        However, for this to make sense, when the columns of rcpo_df are MultiIndex, all lower level values
          must contain only a single unique value (otherwise, there is ambiguity in which value should be chosen!)
        """
        #-------------------------
        n_levels = rcpo_df.columns.nlevels
        assert(n_levels<=max_nlevels)
        #-------------------------
        return_col = []
        for i_level in range(n_levels-1):
            assert(rcpo_df.columns.get_level_values(i_level).nunique()==1)
            return_col.append(rcpo_df.columns.get_level_values(i_level).unique().tolist()[0])
        return_col.append(col_str)
        #-------------------------
        assert(len(return_col)==n_levels)
        return_col = tuple(return_col)
        return return_col
        
    
    @staticmethod
    def add_outage_SNs_to_rcpo_df(
        rcpo_df, 
        set_outage_nSNs=True, 
        include_outage_premise_nbs=True, 
        **kwargs
    ):
        r"""
        NOTE: outg_SNs_col, outg_nSNs_col, outg_prem_nbs_col, and outg_nprem_nbs_col should all be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.

        NOTE: If any of outg_SNs_col, outg_nSNs_col, outg_prem_nbs_col, and outg_nprem_nbs_col are already contained in 
              rcpo_df, they will be replaced.  This is needed so that the merge operation does not come back with _x and _y
              values.  So, one should make sure this function call is truly needed, as grabbing the serial numbers for the
              outages typically takes a couple/few minutes.
        """
        #-------------------------
        outg_SNs_col       = kwargs.get('outg_SNs_col', '_outg_SNs')
        outg_nSNs_col      = kwargs.get('outg_nSNs_col', '_outg_nSNs')
        outg_prem_nbs_col  = kwargs.get('outg_prem_nbs_col', '_outg_prem_nbs')
        outg_nprem_nbs_col = kwargs.get('outg_nprem_nbs_col', '_outg_nprem_nbs')
        #-------------------------
        SNs_in_outgs = DOVSOutages.get_serial_numbers_for_outages(
            outg_rec_nbs=rcpo_df.index.unique().tolist(), 
            return_type=pd.Series, 
            col_type_outg_rec_nb=str, 
            col_type_premise_nb=None, 
            col_type_serial_nb=None, 
            return_premise_nbs_for_outages=include_outage_premise_nbs, 
            return_serial_nbs_col = outg_SNs_col, 
            return_premise_nbs_col = outg_prem_nbs_col
        )
        # If include_outage_premise_nbs is False, then SNs_in_outgs will be a series.
        # The methods below were developed assuming it would be a pd.DataFrame, so convert if needed
        if isinstance(SNs_in_outgs, pd.Series):
            SNs_in_outgs = SNs_in_outgs.to_frame()
        #-------------------------
        assert(sorted(rcpo_df.index.unique().tolist())==sorted(SNs_in_outgs.index.unique().tolist()))
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            #----------
            # See note above about columns being replaced/dropped
            cols_to_drop = [x for x in rcpo_df.columns if x in SNs_in_outgs.columns]
            if len(cols_to_drop)>0:
                rcpo_df = rcpo_df.drop(columns=cols_to_drop)
            #----------
            rcpo_df = rcpo_df.merge(SNs_in_outgs, left_index=True, right_index=True)
            #----------
            if set_outage_nSNs:
                rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, outg_SNs_col, outg_nSNs_col)
                if include_outage_premise_nbs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, outg_prem_nbs_col, outg_nprem_nbs_col)
        else:
            # Currently, only expecting raw and/or norm.  No problem to allow more, but for now keep this to alert 
            # of anything unexpected
            assert(rcpo_df.columns.get_level_values(0).nunique()<=2)
            for i,level_0_val in enumerate(rcpo_df.columns.get_level_values(0).unique()):
                if i==0:
                    SNs_in_outgs.columns = pd.MultiIndex.from_product([[level_0_val], SNs_in_outgs.columns])
                else:
                    SNs_in_outgs.columns = SNs_in_outgs.columns.set_levels([level_0_val], level=0)
                #----------
                # See note above about columns being replaced/dropped
                cols_to_drop = [x for x in rcpo_df.columns if x in SNs_in_outgs.columns]
                if len(cols_to_drop)>0:
                    rcpo_df = rcpo_df.drop(columns=cols_to_drop)
                #----------
                rcpo_df = rcpo_df.merge(SNs_in_outgs, left_index=True, right_index=True)
                #----------
                if set_outage_nSNs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, outg_SNs_col), (level_0_val, outg_nSNs_col))
                    if include_outage_premise_nbs:
                        rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, outg_prem_nbs_col), (level_0_val, outg_nprem_nbs_col))
        #-------------------------
        rcpo_df = rcpo_df.sort_index(axis=1,level=0)
        #-------------------------
        return rcpo_df
        
    @staticmethod
    def get_outg_rec_nbs_from_cpo_df(cpo_df, idfr):
        r"""
        Extract the outg_rec_nbs from a cpo_df (or, really, any DF with outg_rec_nbs).
        Previously, it was assumed that outg_rec_nbs would either be contained in the index or in a column.
        While this is obviously still the case, the original methodology did not account for the slightly more
          complicated case where cpo_df has a MultiIndex index, and the outg_rec_nbs reside in one of the levels.
          The purpose of this function is to handle that case (as well as maintaining the original functionality)
          
        Return Value:
            If outg_rec_nbs are stored in a column of cpo_df, the returned object will be a pd.Series
            If outg_rec_nbs are stored in the index, the returned object will be a pd index.
            If one wants the returned object to be a list, use get_outg_rec_nbs_list_from_cpo_df

        idfr:
            This directs from where the outg_rec_nbs will be retrieved.
            This should be a string, list, or tuple.
            If the outg_rec_nbs are located in a column, idfr should simply be the column
                - Single index columns --> simple string
                - MultiIndex columns   --> appropriate tuple to identify column
            If the outg_rec_nbs are located in the index:
                - Single level index --> simple string 'index' or 'index_0'
                - MultiIndex index:  --> 
                    - string f'index_{level}', where level is the index level containing the outg_rec_nbs
                    - tuple of length 2, with 0th element ='index' and 1st element = idx_level_name where
                        idx_level_name is the name of the index level containing the outg_rec_nbs
        """
        #-------------------------
        return DOVSOutages.get_outg_rec_nbs_from_df(df=cpo_df, idfr=idfr)


    @staticmethod
    def get_outg_rec_nbs_list_from_cpo_df(cpo_df, idfr, unique_only=True):
        r"""
        See get_outg_rec_nbs_from_cpo_df for details.
        This returns a list version of get_outg_rec_nbs_from_cpo_df
        """
        #-------------------------
        out_rec_nbs = MECPODf.get_outg_rec_nbs_from_cpo_df(cpo_df=cpo_df, idfr=idfr)
        if unique_only:
            return out_rec_nbs.unique().tolist()
        else:
            return out_rec_nbs.tolist()


    @staticmethod
    def get_prim_SNs_for_outage_i(
        rcpo_df_i, 
        direct_SNs_in_outgs_df, 
        outg_rec_nb_col='index', 
        prim_SNs_col='direct_serial_numbers'
    ):
        r"""
        rcpo_df_i should only contain a single outage.
        For functionality with multiple outages, use MECPODf.add_prim_SNs_to_rcpo_df

        outg_rec_nb_col:
          Outage record number column in rcpo_df_i.  This is set to the index by default

        prim_SNs_col:
          The primary serial numbers column in direct_SNs_in_outgs_df
        """
        #-------------------------
        if outg_rec_nb_col=='index':
            outg_rec_nb = rcpo_df_i.index.unique().tolist()
        else:
            outg_rec_nb = rcpo_df_i[outg_rec_nb_col].unique().tolist()
        # Only one outage_rec_nb should be found
        assert(len(outg_rec_nb)==1)
        outg_rec_nb=outg_rec_nb[0]
        #-------------------------
        if outg_rec_nb not in direct_SNs_in_outgs_df.index:
            return []
        direct_SNs_in_outgs_df_i = direct_SNs_in_outgs_df.loc[outg_rec_nb]
        # Only one entry should be found in direct_SNs_in_outgs_df for outg_rec_nb
        # Thus, direct_SNs_in_outgs_df_i should be a series, not a dataframe
        assert(isinstance(direct_SNs_in_outgs_df_i, pd.Series))
        direct_SNs_in_outgs_df_i=direct_SNs_in_outgs_df_i[prim_SNs_col]
        return direct_SNs_in_outgs_df_i


    @staticmethod
    def add_prim_SNs_to_rcpo_df(
        rcpo_df, 
        direct_SNs_in_outgs_df, 
        outg_rec_nb_col='index', 
        prim_SNs_col='direct_serial_numbers', 
        set_prim_nSNs=True, 
        sort_SNs=True, 
        build_direct_SNs_in_outgs_df_kwargs={}, 
        **kwargs
    ):
        r"""
        rcpo_df should contain multiple outages (although, should work with only a single outage as well).

        If direct_SNs_in_outgs_df is None, it will be built using DOVSOutages.build_direct_SNs_in_outgs_df

        outg_rec_nb_col:
          Outage record number column in rcpo_df.  This is set to the index by default

        prim_SNs_col:
          The primary serial numbers column in direct_SNs_in_outgs_df

        kwargs:
          placement_prim_SNs_col:
            New column in rcpo_df where primary serial numbers will be placed
          placement_prim_nSNs_col:
            New column in rcpo_df where number of primary serial numbers will be placed
        """
        #-------------------------
        placement_prim_SNs_col  = kwargs.get('placement_prim_SNs_col', '_prim_SNs')
        placement_prim_nSNs_col = kwargs.get('placement_prim_nSNs_col', '_prim_nSNs')
        #-------------------------
        assert(rcpo_df.columns.nlevels<=2)
        # Should only be raw df, so if n_col_levels==2, should only be one unique value in level 0
        if rcpo_df.columns.nlevels==2:
            assert(rcpo_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpo_df.columns.get_level_values(0).unique().tolist()[0]
            # In this case, all columns should be tuples, not strings
            if isinstance(placement_prim_SNs_col, str):
                placement_prim_SNs_col = (level_0_val, placement_prim_SNs_col)
            assert(isinstance(placement_prim_SNs_col,tuple) and 
                   len(placement_prim_SNs_col)==2 and 
                   placement_prim_SNs_col[0]==level_0_val)
            #-----
            if isinstance(placement_prim_nSNs_col, str):
                placement_prim_nSNs_col = (level_0_val, placement_prim_nSNs_col)
            assert(isinstance(placement_prim_nSNs_col,tuple) and 
                   len(placement_prim_nSNs_col)==2 and 
                   placement_prim_nSNs_col[0]==level_0_val)
            #-----
            if outg_rec_nb_col != 'index':
                if isinstance(outg_rec_nb_col, str):
                    outg_rec_nb_col = (level_0_val, outg_rec_nb_col)
                assert(isinstance(outg_rec_nb_col,tuple) and 
                       len(outg_rec_nb_col)==2 and 
                       outg_rec_nb_col[0]==level_0_val)
        #-------------------------
        # If direct_SNs_in_outgs_df is None, build it
        if direct_SNs_in_outgs_df is None:
            if outg_rec_nb_col=='index':
                outg_rec_nbs = rcpo_df.index.unique().tolist()
            else:
                outg_rec_nbs = rcpo_df[outg_rec_nb_col].unique().tolist()
            build_direct_SNs_in_outgs_df_kwargs['outg_rec_nbs']=outg_rec_nbs
            direct_SNs_in_outgs_df = DOVSOutages.build_direct_SNs_in_outgs_df(**build_direct_SNs_in_outgs_df_kwargs)
        #-------------------------
        if outg_rec_nb_col=='index':
            gp_by = rcpo_df.index
        else:
            gp_by = outg_rec_nb_col
        #-----
        rcpo_df[placement_prim_SNs_col] = rcpo_df.groupby(gp_by).apply(
            lambda x: MECPODf.get_prim_SNs_for_outage_i(
                rcpo_df_i=x, 
                direct_SNs_in_outgs_df=direct_SNs_in_outgs_df, 
                outg_rec_nb_col=outg_rec_nb_col, 
                prim_SNs_col=prim_SNs_col
            )
        )
        #-------------------------
        if set_prim_nSNs:
            rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(
                rcpo_df_wide=rcpo_df, 
                SNs_col=placement_prim_SNs_col, 
                nSNs_col=placement_prim_nSNs_col
            )
        #-------------------------
        if sort_SNs:
            rcpo_df=MECPODf.sort_SNs_in_rcpo_df_wide(rcpo_df, placement_prim_SNs_col)
        #-------------------------
        return rcpo_df
        
        
    #TODO: Does include_outage_premise_nbs do anything?!
    @staticmethod
    def add_outage_active_SNs_to_rcpo_df(
        rcpo_df, 
        set_outage_nSNs=True, 
        include_outage_premise_nbs=True, 
        df_mp_curr=None,
        df_mp_hist=None, 
        addtnl_get_active_SNs_and_others_for_outages_kwargs=None, 
        **kwargs
    ):
        r"""
        NOTE: outg_SNs_col, outg_nSNs_col, outg_prem_nbs_col, and outg_nprem_nbs_col should all be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.

        NOTE: If any of outg_SNs_col, outg_nSNs_col, outg_prem_nbs_col, and outg_nprem_nbs_col are already contained in 
              rcpo_df, they will be replaced.  This is needed so that the merge operation does not come back with _x and _y
              values.  So, one should make sure this function call is truly needed, as grabbing the serial numbers for the
              outages typically takes a couple/few minutes.

        NOTE: To make things run faster, the user can supply df_mp_curr and df_mp_hist.  These will be included in 
              get_active_SNs_and_others_for_outages_kwargs.
              NOTE: If df_mp_curr/df_mp_hist is also supplied in addtnl_get_active_SNs_and_others_for_outages_kwargs,
                    that/those in addtnl_get_active_SNs_and_others_for_outages_kwargs will ultimately be used (not the
                    explicity df_mp_hist/curr in the function arguments!)
              CAREFUL: If one does supple df_mp_curr/hist, one must be certain these DFs contain all necessary elements!
        """
        #-------------------------
        outg_SNs_col       = kwargs.get('outg_SNs_col', '_outg_SNs')
        outg_nSNs_col      = kwargs.get('outg_nSNs_col', '_outg_nSNs')
        outg_prem_nbs_col  = kwargs.get('outg_prem_nbs_col', '_outg_prem_nbs')
        outg_nprem_nbs_col = kwargs.get('outg_nprem_nbs_col', '_outg_nprem_nbs')
        #-------------------------
        get_active_SNs_and_others_for_outages_kwargs = dict(
            outg_rec_nbs=rcpo_df.index.unique().tolist(), 
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist, 
            return_premise_nbs_col=outg_prem_nbs_col, 
            return_SNs_col=outg_SNs_col
        )
        if addtnl_get_active_SNs_and_others_for_outages_kwargs is not None:
            get_active_SNs_and_others_for_outages_kwargs = {**get_active_SNs_and_others_for_outages_kwargs, 
                                                            **addtnl_get_active_SNs_and_others_for_outages_kwargs}
        SNs_in_outgs = DOVSOutages.get_active_SNs_and_others_for_outages(**get_active_SNs_and_others_for_outages_kwargs)
        assert(isinstance(SNs_in_outgs, pd.DataFrame))
        # Drop the columns which are no longer needed
        cols_to_drop = [
            get_active_SNs_and_others_for_outages_kwargs.get('dovs_t_min_col', 'DT_OFF_TS_FULL'), 
            get_active_SNs_and_others_for_outages_kwargs.get('dovs_t_max_col', 'DT_ON_TS'), 
            get_active_SNs_and_others_for_outages_kwargs.get('return_premise_nbs_from_MP_col', 'premise_nbs_from_MP')
        ]
        SNs_in_outgs = SNs_in_outgs.drop(columns=cols_to_drop)
        #-------------------------
        assert(sorted(rcpo_df.index.unique().tolist())==sorted(SNs_in_outgs.index.unique().tolist()))
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            #----------
            # See note above about columns being replaced/dropped
            cols_to_drop = [x for x in rcpo_df.columns if x in SNs_in_outgs.columns]
            if len(cols_to_drop)>0:
                rcpo_df = rcpo_df.drop(columns=cols_to_drop)
            #----------
            rcpo_df = rcpo_df.merge(SNs_in_outgs, left_index=True, right_index=True)
            #----------
            if set_outage_nSNs:
                rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, outg_SNs_col, outg_nSNs_col)
                if include_outage_premise_nbs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, outg_prem_nbs_col, outg_nprem_nbs_col)
        else:
            # Currently, only expecting raw and/or norm.  No problem to allow more, but for now keep this to alert 
            # of anything unexpected
            assert(rcpo_df.columns.get_level_values(0).nunique()<=2)
            for i,level_0_val in enumerate(rcpo_df.columns.get_level_values(0).unique()):
                if i==0:
                    SNs_in_outgs.columns = pd.MultiIndex.from_product([[level_0_val], SNs_in_outgs.columns])
                else:
                    SNs_in_outgs.columns = SNs_in_outgs.columns.set_levels([level_0_val], level=0)
                #----------
                # See note above about columns being replaced/dropped
                cols_to_drop = [x for x in rcpo_df.columns if x in SNs_in_outgs.columns]
                if len(cols_to_drop)>0:
                    rcpo_df = rcpo_df.drop(columns=cols_to_drop)
                #----------
                rcpo_df = rcpo_df.merge(SNs_in_outgs, left_index=True, right_index=True)
                #----------
                if set_outage_nSNs:
                    rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, outg_SNs_col), (level_0_val, outg_nSNs_col))
                    if include_outage_premise_nbs:
                        rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(rcpo_df, (level_0_val, outg_prem_nbs_col), (level_0_val, outg_nprem_nbs_col))
        #-------------------------
        rcpo_df = rcpo_df.sort_index(axis=1,level=0)
        #-------------------------
        return rcpo_df
        
        
    #TODO what about e.g., equip_typ_nms_of_interest in DOVSOutages.build_active_direct_SNs_in_outgs_df?????
    @staticmethod
    def add_active_prim_SNs_to_rcpo_df(
        rcpo_df, 
        direct_SNs_in_outgs_df, 
        mp_df_curr=None,
        mp_df_hist=None, 
        outg_rec_nb_col='index', 
        prim_SNs_col='direct_serial_numbers', 
        set_prim_nSNs=True, 
        sort_SNs=True, 
        build_direct_SNs_in_outgs_df_kwargs={}, 
        **kwargs
    ):
        r"""
        rcpo_df should contain multiple outages (although, should work with only a single outage as well).

        If direct_SNs_in_outgs_df is None, it will be built using DOVSOutages.build_active_direct_SNs_in_outgs_df

        outg_rec_nb_col:
          Outage record number column in rcpo_df.  This is set to the index by default

        prim_SNs_col:
          The primary serial numbers column in direct_SNs_in_outgs_df

        kwargs:
          placement_prim_SNs_col:
            New column in rcpo_df where primary serial numbers will be placed
          placement_prim_nSNs_col:
            New column in rcpo_df where number of primary serial numbers will be placed
        """
        #-------------------------
        placement_prim_SNs_col  = kwargs.get('placement_prim_SNs_col', '_prim_SNs')
        placement_prim_nSNs_col = kwargs.get('placement_prim_nSNs_col', '_prim_nSNs')
        #-------------------------
        assert(rcpo_df.columns.nlevels<=2)
        # Should only be raw df, so if n_col_levels==2, should only be one unique value in level 0
        if rcpo_df.columns.nlevels==2:
            assert(rcpo_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpo_df.columns.get_level_values(0).unique().tolist()[0]
            # In this case, all columns should be tuples, not strings
            if isinstance(placement_prim_SNs_col, str):
                placement_prim_SNs_col = (level_0_val, placement_prim_SNs_col)
            assert(isinstance(placement_prim_SNs_col,tuple) and 
                   len(placement_prim_SNs_col)==2 and 
                   placement_prim_SNs_col[0]==level_0_val)
            #-----
            if isinstance(placement_prim_nSNs_col, str):
                placement_prim_nSNs_col = (level_0_val, placement_prim_nSNs_col)
            assert(isinstance(placement_prim_nSNs_col,tuple) and 
                   len(placement_prim_nSNs_col)==2 and 
                   placement_prim_nSNs_col[0]==level_0_val)
            #-----
            if outg_rec_nb_col != 'index':
                if isinstance(outg_rec_nb_col, str):
                    outg_rec_nb_col = (level_0_val, outg_rec_nb_col)
                assert(isinstance(outg_rec_nb_col,tuple) and 
                       len(outg_rec_nb_col)==2 and 
                       outg_rec_nb_col[0]==level_0_val)
        #-------------------------
        # If direct_SNs_in_outgs_df is None, build it
        if direct_SNs_in_outgs_df is None:
            if outg_rec_nb_col=='index':
                outg_rec_nbs = rcpo_df.index.unique().tolist()
            else:
                outg_rec_nbs = rcpo_df[outg_rec_nb_col].unique().tolist()
            build_direct_SNs_in_outgs_df_kwargs['outg_rec_nbs']=outg_rec_nbs
            build_direct_SNs_in_outgs_df_kwargs['mp_df_curr'] = build_direct_SNs_in_outgs_df_kwargs.get('mp_df_curr', mp_df_curr)
            build_direct_SNs_in_outgs_df_kwargs['mp_df_hist'] = build_direct_SNs_in_outgs_df_kwargs.get('mp_df_hist', mp_df_curr)
            direct_SNs_in_outgs_df = DOVSOutages.build_active_direct_SNs_in_outgs_df(**build_direct_SNs_in_outgs_df_kwargs)
        #-------------------------
        if outg_rec_nb_col=='index':
            gp_by = rcpo_df.index
        else:
            gp_by = outg_rec_nb_col
        #-----
        rcpo_df[placement_prim_SNs_col] = rcpo_df.groupby(gp_by).apply(
            lambda x: MECPODf.get_prim_SNs_for_outage_i(
                rcpo_df_i=x, 
                direct_SNs_in_outgs_df=direct_SNs_in_outgs_df, 
                outg_rec_nb_col=outg_rec_nb_col, 
                prim_SNs_col=prim_SNs_col
            )
        )
        #-------------------------
        if set_prim_nSNs:
            rcpo_df = MECPODf.set_nSNs_from_SNs_in_rcpo_df(
                rcpo_df_wide=rcpo_df, 
                SNs_col=placement_prim_SNs_col, 
                nSNs_col=placement_prim_nSNs_col
            )
        #-------------------------
        if sort_SNs:
            rcpo_df=MECPODf.sort_SNs_in_rcpo_df_wide(rcpo_df, placement_prim_SNs_col)
        #-------------------------
        return rcpo_df
        
    # MOVED TO CPXDfBuilder    
    @staticmethod
    def remove_SNs_cols_from_rcpo_df(
        rcpo_df, 
        SNs_tags=None, 
        is_long=False
    ):
        r"""
        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        NOTE: SNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        if is_long:
            assert(rcpo_df.index.nlevels==2)
            untagged_idxs = Utilities.find_untagged_idxs_in_list(
                lst=rcpo_df.index.get_level_values(1).tolist(), 
                tags=SNs_tags)
            return rcpo_df.iloc[untagged_idxs]
        #-------------------------
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            level_idx=0
        else:
            level_idx=1
        #-------------------------
        untagged_idxs = Utilities.find_untagged_idxs_in_list(
            lst=rcpo_df.columns.get_level_values(level_idx).tolist(), 
            tags=SNs_tags)
        cols_to_keep = rcpo_df.columns[untagged_idxs]
        #-------------------------
        return rcpo_df[cols_to_keep]

    # MOVED TO CPXDf 
    @staticmethod
    def find_SNs_cols_idxs_from_cpo_df(
        cpo_df, 
        SNs_tags=None, 
        is_long=False
    ):
        r"""
        Returns the index positions of those found within the list of columns (if cpo_df is wide) 
          or index leve_1 values (if cpo_df is long). 
        As opposed to returning the actual columns found, this makes it easier to exclude (or select) the 
          columns/indices of choice, especially when dealing with the case of cpo_df being long, or cpo_df 
          containing raw and normalized values (in which case, e.g., there will likely be two columns for each
          SNs_tag, one for raw and one for normalized)

        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        NOTE: SNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        if is_long:
            assert(cpo_df.index.nlevels==2)
            tagged_idxs = Utilities.find_tagged_idxs_in_list(
                lst=cpo_df.index.get_level_values(1).tolist(), 
                tags=SNs_tags)
            return tagged_idxs
        #-------------------------
        assert(cpo_df.columns.nlevels<=2)
        if cpo_df.columns.nlevels==1:
            level_idx=0
        else:
            level_idx=1
        #-------------------------
        tagged_idxs = Utilities.find_tagged_idxs_in_list(
            lst=cpo_df.columns.get_level_values(level_idx).tolist(), 
            tags=SNs_tags)
        #-------------------------
        return tagged_idxs

    # MOVED TO CPXDf
    @staticmethod
    def get_non_SNs_cols_from_cpo_df(
        cpo_df, 
        SNs_tags=None
    ):
        r"""
        Only for wide-form cpo_dfs.
        Returns a list of columns which are not SNs cols (as found using SNs_tags)
        
        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        """
        #-------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        SNs_cols_idxs = MECPODf.find_SNs_cols_idxs_from_cpo_df(
            cpo_df=cpo_df, 
            SNs_tags=SNs_tags, 
            is_long=False
        )
        SNs_cols = cpo_df.columns[SNs_cols_idxs].tolist()
        non_SNs_cols = [x for x in cpo_df.columns if x not in SNs_cols]
        assert(len(set(non_SNs_cols+SNs_cols).symmetric_difference(set(cpo_df.columns)))==0)
        #-------------------------
        return non_SNs_cols
        
    @staticmethod
    def build_rcpo_df_norm_by_list_counts(
        rcpo_df_raw, 
        n_counts_col='_outg_nSNs', 
        list_col='_outg_SNs', 
        add_list_col_to_rcpo_df_func=None, 
        add_list_col_to_rcpo_df_kwargs={}, 
        other_col_tags_to_ignore=None, 
        drop_n_counts_eq_0=True, 
        new_level_0_val='counts_norm_by_outg_nSNs', 
        remove_ignored_cols=False
    ):
        r"""
        Build rcpo_df normalized by the number of elements in list_col (which are stored in n_counts_col).
        
        If n_counts_col exists in rcpo_df_raw, the normalized DF is simply formed by dividing the columns to be
          normalized (see notes below on other_col_tags_to_ignore) by n_counts_col.
        If n_counts_col does not exist:
          If list_col exists in rcpo_df_raw, then n_counts_col is built by simply taking the length of each list
            element in list_col.  The normalization then proceeds simply as described above.
          If list_col does not exist, it will be built using the function add_list_col_to_rcpo_df_func together with the
            function arguments rcpo_df=rcpo_df_norm, **add_list_col_to_rcpo_df_kwargs
        
        drop_n_counts_eq_0:
          It is possible for the number of counts to be zero.
          Dividing by zero will make all counts for such an entry equal to NaN or inf.
          When drop_n_counts_eq_0 is True, such entries will be removed.
          
        other_col_tags_to_ignore:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None

        NOTE: list_col and n_counts_col should both be strings, not tuples.
              If column is MultiIndex, the level_0 value will be handled below.
        """
        #-------------------------
        if other_col_tags_to_ignore is None:
            other_col_tags_to_ignore=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        rcpo_df_norm = rcpo_df_raw.copy()
        #-------------------------
        assert(rcpo_df_norm.columns.nlevels<=2)
        if rcpo_df_norm.columns.nlevels==1:
            reason_level_idx=0
            col_tags_to_ignore = other_col_tags_to_ignore + [n_counts_col+'_EXACTMATCH', list_col+'_EXACTMATCH']
        else:
            # Should only be raw df, so if n_col_levels==2, should only be one unique value in level 0
            assert(rcpo_df_norm.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpo_df_norm.columns.get_level_values(0).unique().tolist()[0]
            #-----
            if isinstance(list_col, str):
                list_col = MECPODf.build_full_rcpo_col(col_str=list_col, rcpo_df=rcpo_df_norm, max_nlevels=2)
            assert(isinstance(list_col,tuple) and len(list_col)==2 and list_col[0]==level_0_val)
            #-----
            if isinstance(n_counts_col, str):
                n_counts_col = MECPODf.build_full_rcpo_col(col_str=n_counts_col, rcpo_df=rcpo_df_norm, max_nlevels=2)
            assert(isinstance(n_counts_col,tuple) and len(n_counts_col)==2 and n_counts_col[0]==level_0_val)
            #---------------
            reason_level_idx=1
            col_tags_to_ignore = other_col_tags_to_ignore + [n_counts_col[1]+'_EXACTMATCH', list_col[1]+'_EXACTMATCH']
        #-------------------------
        # If n_counts_col, which will be used to normalize, is not in rcpo_df_norm, build it
        if n_counts_col not in rcpo_df_norm.columns:
            # If for some reason list_col exists (although n_counts_col doesn't), simply call 
            # MECPODf.set_nSNs_from_SNs_in_rcpo_df instead of running full-blown add_list_col_to_rcpo_df_func
            if list_col in rcpo_df_norm.columns:
                rcpo_df_norm = MECPODf.set_nSNs_from_SNs_in_rcpo_df(
                    rcpo_df_wide=rcpo_df_norm, 
                    SNs_col=list_col, 
                    nSNs_col=n_counts_col
                )
            else:
                # More typical case, where both n_counts_col and list_col are absent from rcpo_df_norm
                assert(add_list_col_to_rcpo_df_func is not None)
                rcpo_df_norm = add_list_col_to_rcpo_df_func(
                    rcpo_df=rcpo_df_norm, 
                    **add_list_col_to_rcpo_df_kwargs
                )
        #-------------------------
        # Find the columns to normalize as those not in col_tags_to_ignore
        cols_to_norm_idxs = Utilities.find_untagged_idxs_in_list(
            lst=rcpo_df_norm.columns.get_level_values(reason_level_idx).tolist(), 
            tags=col_tags_to_ignore)
        cols_to_norm = rcpo_df_norm.columns[cols_to_norm_idxs]
        norm_col = n_counts_col
        #-----
        # Casting columns as np.float64 allows the operation of dividing by zero (should result in NaN (or inf in some instances))
        # This will happen when no SNs are found for an outage 
        #   NOTE: premise numbers are always found, but the meter_premise database does not always contain the premise number
        rcpo_df_norm = Utilities_df.convert_col_types(
            df=rcpo_df_norm, 
            cols_and_types_dict = {x:np.float64 for x in cols_to_norm.tolist()+[norm_col]}, 
            to_numeric_errors='coerce', 
            inplace=True
        )
        #-----
        rcpo_df_norm[cols_to_norm] = rcpo_df_norm[cols_to_norm].divide(rcpo_df_norm[norm_col], axis=0)
        #-------------------------
        if drop_n_counts_eq_0:
            rcpo_df_norm = rcpo_df_norm.drop(index=rcpo_df_norm[rcpo_df_norm[n_counts_col]==0].index)
        #-------------------------
        if rcpo_df_norm.columns.nlevels==2:
            rcpo_df_norm.columns = rcpo_df_norm.columns.set_levels([new_level_0_val], level=0)
        #-------------------------
        if remove_ignored_cols:
            rcpo_df_norm = MECPODf.remove_SNs_cols_from_rcpo_df(
                rcpo_df=rcpo_df_norm, 
                SNs_tags=col_tags_to_ignore, 
                is_long=False
            )
        #-------------------------
        return rcpo_df_norm
        
        
        
    @staticmethod
    def build_rcpo_df_norm_by_outg_nSNs(
        rcpo_df_raw, 
        outg_nSNs_col='_outg_nSNs', 
        outg_SNs_col='_outg_SNs', 
        other_SNs_col_tags_to_ignore=None, 
        drop_outg_nSNs_eq_0=True, 
        new_level_0_val='counts_norm_by_outg_nSNs', 
        remove_SNs_cols=False
    ):
        r"""
        Build rcpo_df normalized by the number of serial numbers in each outage

        drop_outg_nSNs_eq_0:
          It is possible for the number of serial numbers in an outage to be zero!
          Premise numbers are always found, but the meter_premise database does not always 
            contain the premise numbers.
          Dividing by zero will make all counts for such an entry equal to NaN or inf.
          When drop_outg_nSNs_eq_0 is True, such entries will be removed.
          
        other_SNs_col_tags_to_ignore:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None

        NOTE: outg_SNs_col and outg_nSNs_col should both be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if other_SNs_col_tags_to_ignore is None:
            other_SNs_col_tags_to_ignore=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        n_counts_col = outg_nSNs_col
        list_col = outg_SNs_col
        #-------------------------
        # NOTE: MECPODf.add_outage_SNs_to_rcpo_df expects outg_SNs_col and outg_nSNs_col to be strings, not tuples
        #       as it handles the level 0 values if they exist.  So, if tuples, use only highest level values (i.e., level 1)
        assert(Utilities.is_object_one_of_types(list_col, [str, tuple]))
        assert(Utilities.is_object_one_of_types(n_counts_col, [str, tuple]))
        #-----
        add_list_col_to_rcpo_df_func = MECPODf.add_outage_SNs_to_rcpo_df
        add_list_col_to_rcpo_df_kwargs = dict(
            set_outage_nSNs=True, 
            include_outage_premise_nbs=False, 
            outg_SNs_col =(outg_SNs_col  if isinstance(outg_SNs_col, str)  else outg_SNs_col[1]), 
            outg_nSNs_col=(outg_nSNs_col if isinstance(outg_nSNs_col, str) else outg_nSNs_col[1])
        )
        #-------------------------
        other_col_tags_to_ignore = other_SNs_col_tags_to_ignore
        drop_n_counts_eq_0 = drop_outg_nSNs_eq_0
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
        

    @staticmethod
    def build_rcpo_df_norm_by_outg_active_nSNs(
        rcpo_df_raw, 
        outg_nSNs_col='_outg_nSNs', 
        outg_SNs_col='_outg_SNs', 
        other_SNs_col_tags_to_ignore=None, 
        drop_outg_nSNs_eq_0=True, 
        new_level_0_val='counts_norm_by_outg_nSNs', 
        remove_SNs_cols=False, 
        df_mp_curr=None,
        df_mp_hist=None
    ):
        r"""
        Build rcpo_df normalized by the number of serial numbers in each outage

        drop_outg_nSNs_eq_0:
          It is possible for the number of serial numbers in an outage to be zero!
          Premise numbers are always found, but the meter_premise database does not always 
            contain the premise numbers.
          Dividing by zero will make all counts for such an entry equal to NaN or inf.
          When drop_outg_nSNs_eq_0 is True, such entries will be removed.
          
        other_SNs_col_tags_to_ignore:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None

        NOTE: outg_SNs_col and outg_nSNs_col should both be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if other_SNs_col_tags_to_ignore is None:
            other_SNs_col_tags_to_ignore=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        n_counts_col = outg_nSNs_col
        list_col = outg_SNs_col
        #-------------------------
        # NOTE: MECPODf.add_outage_SNs_to_rcpo_df expects outg_SNs_col and outg_nSNs_col to be strings, not tuples
        #       as it handles the level 0 values if they exist.  So, if tuples, use only highest level values (i.e., level 1)
        assert(Utilities.is_object_one_of_types(list_col, [str, tuple]))
        assert(Utilities.is_object_one_of_types(n_counts_col, [str, tuple]))
        #-----
        add_list_col_to_rcpo_df_func = MECPODf.add_outage_active_SNs_to_rcpo_df
        add_list_col_to_rcpo_df_kwargs = dict(
            set_outage_nSNs=True, 
            include_outage_premise_nbs=False, 
            outg_SNs_col =(outg_SNs_col  if isinstance(outg_SNs_col, str)  else outg_SNs_col[1]), 
            outg_nSNs_col=(outg_nSNs_col if isinstance(outg_nSNs_col, str) else outg_nSNs_col[1]), 
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist
        )
        #-------------------------
        other_col_tags_to_ignore = other_SNs_col_tags_to_ignore
        drop_n_counts_eq_0 = drop_outg_nSNs_eq_0
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
        
        
    @staticmethod
    def build_rcpo_df_norm_by_prim_nSNs(
        rcpo_df_raw, 
        direct_SNs_in_outgs_df=None, 
        outg_rec_nb_col='index', 
        prim_nSNs_col='_prim_nSNs', 
        prim_SNs_col='_prim_SNs', 
        other_SNs_col_tags_to_ignore=None, 
        drop_prim_nSNs_eq_0=True, 
        new_level_0_val='counts_norm_by_prim_nSNs', 
        remove_SNs_cols=False, 
        build_direct_SNs_in_outgs_df_kwargs={}
    ):
        r"""
        Build rcpo_df normalized by the number of primary serial numbers in each outage

        drop_prim_nSNs_eq_0:
          It is possible for the number of serial numbers in an outage to be zero!
          Premise numbers are always found, but the meter_premise database does not always 
            contain the premise numbers.
          Dividing by zero will make all counts for such an entry equal to NaN or inf.
          When drop_prim_nSNs_eq_0 is True, such entries will be removed.
          
        other_SNs_col_tags_to_ignore:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None

        NOTE: prim_SNs_col and prim_nSNs_col should both be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if other_SNs_col_tags_to_ignore is None:
            other_SNs_col_tags_to_ignore=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        n_counts_col = prim_nSNs_col
        list_col = prim_SNs_col
        #-------------------------
        add_list_col_to_rcpo_df_func = MECPODf.add_prim_SNs_to_rcpo_df
        add_list_col_to_rcpo_df_kwargs = dict(
            direct_SNs_in_outgs_df=direct_SNs_in_outgs_df, 
            outg_rec_nb_col=outg_rec_nb_col, 
            prim_SNs_col='direct_serial_numbers', 
            set_prim_nSNs=True, 
            sort_SNs=True, 
            build_direct_SNs_in_outgs_df_kwargs=build_direct_SNs_in_outgs_df_kwargs, 
            placement_prim_SNs_col=prim_SNs_col, 
            placement_prim_nSNs_col=prim_nSNs_col
        )
        #-------------------------
        other_col_tags_to_ignore = other_SNs_col_tags_to_ignore
        drop_n_counts_eq_0 = drop_prim_nSNs_eq_0
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
        
        
    @staticmethod
    def build_rcpo_df_norm_by_prim_active_nSNs(
        rcpo_df_raw, 
        direct_SNs_in_outgs_df=None, 
        outg_rec_nb_col='index', 
        prim_nSNs_col='_prim_nSNs', 
        prim_SNs_col='_prim_SNs', 
        other_SNs_col_tags_to_ignore=None, 
        drop_prim_nSNs_eq_0=True, 
        new_level_0_val='counts_norm_by_prim_nSNs', 
        remove_SNs_cols=False, 
        build_direct_SNs_in_outgs_df_kwargs={}, 
        df_mp_curr=None,
        df_mp_hist=None
    ):
        r"""
        Build rcpo_df normalized by the number of primary serial numbers in each outage

        drop_prim_nSNs_eq_0:
          It is possible for the number of serial numbers in an outage to be zero!
          Premise numbers are always found, but the meter_premise database does not always 
            contain the premise numbers.
          Dividing by zero will make all counts for such an entry equal to NaN or inf.
          When drop_prim_nSNs_eq_0 is True, such entries will be removed.
          
        other_SNs_col_tags_to_ignore:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None

        NOTE: prim_SNs_col and prim_nSNs_col should both be strings, not tuples.
              If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if other_SNs_col_tags_to_ignore is None:
            other_SNs_col_tags_to_ignore=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        n_counts_col = prim_nSNs_col
        list_col = prim_SNs_col
        #-------------------------
        add_list_col_to_rcpo_df_func = MECPODf.add_active_prim_SNs_to_rcpo_df
        add_list_col_to_rcpo_df_kwargs = dict(
            direct_SNs_in_outgs_df=direct_SNs_in_outgs_df, 
            outg_rec_nb_col=outg_rec_nb_col, 
            prim_SNs_col='direct_serial_numbers', 
            set_prim_nSNs=True, 
            sort_SNs=True, 
            build_direct_SNs_in_outgs_df_kwargs=build_direct_SNs_in_outgs_df_kwargs, 
            placement_prim_SNs_col=prim_SNs_col, 
            placement_prim_nSNs_col=prim_nSNs_col, 
            df_mp_curr=df_mp_curr, 
            df_mp_hist=df_mp_hist
        )
        #-------------------------
        other_col_tags_to_ignore = other_SNs_col_tags_to_ignore
        drop_n_counts_eq_0 = drop_prim_nSNs_eq_0
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
        
        
    def get_rcpo_df_subset_by_mjr_mnr_causes( 
        rcpo_df, 
        df_subset_slicers, 
        outg_rec_nb_col='index', 
        mjr_mnr_causes_df=None
    ):
        r"""
        Returns a dictionary whose values are subset pd.DataFrame objects and whose
          keys are the corresponding name attributes for the df_subset_slicer
          UNLESS only a single df_subset_slicers is given, in which case just a pd.DataFrame is returned

        See MECPODf.get_rcpo_df_subset_by_std_mjr_mnr_causes for examples of standard df_subset_slicers

        df_subset_slicers: 
          See DataFrameSubsetSlicer for more information

        mjr_mnr_causes_df:
          If not supplied, it will be built via DOVSOutages.get_mjr_mnr_causes_for_outages
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(df_subset_slicers, [DataFrameSubsetSlicer.DataFrameSubsetSlicer, list]))
        if isinstance(df_subset_slicers, DataFrameSubsetSlicer.DataFrameSubsetSlicer):
            df_subset_slicers=[df_subset_slicers]
        #-------------------------
        outg_rec_nbs = MECPODf.get_outg_rec_nbs_from_cpo_df(cpo_df=rcpo_df, idfr=outg_rec_nb_col)
        assert(len(rcpo_df)==len(outg_rec_nbs)) # Important in ensuring proper selection towards end of function
        outg_rec_nbs_unq = outg_rec_nbs.unique().tolist()
        #-------------------------
        if mjr_mnr_causes_df is None:
            mjr_mnr_causes_df = DOVSOutages.get_mjr_mnr_causes_for_outages(
                outg_rec_nbs=outg_rec_nbs_unq, 
                include_equip_type=True, 
                set_outg_rec_nb_as_index=True
            )
        # Make sure all outg_rec_nbs_unq in mjr_mnr_causes_df
        assert(len(set(outg_rec_nbs_unq).difference(set(mjr_mnr_causes_df.index.tolist())))==0)
        #-------------------------
        subset_dfs_dict = {}
        for slicer in df_subset_slicers:
            assert(slicer.name not in subset_dfs_dict)

            # First, get subset of outages satisfying the cuts in slicer
            outgs_subset_df = slicer.perform_slicing(mjr_mnr_causes_df)

            # Use the subset of outages to select the desired entries from rcpo_df
            assert(slicer.name not in subset_dfs_dict.keys())
            subset_dfs_dict[slicer.name] = rcpo_df[outg_rec_nbs.isin(outgs_subset_df.index)]
        #-------------------------
        if len(subset_dfs_dict)==1:
            return list(subset_dfs_dict.values())[0]
        #-------------------------
        return subset_dfs_dict

    @staticmethod
    def get_rcpo_df_subset_by_std_mjr_mnr_causes(
        rcpo_df, 
        addtnl_df_subset_slicers=None, 
        outg_rec_nb_col='index', 
        mjr_mnr_causes_df=None
    ):
        r"""
        See MECPODf.get_rcpo_df_subset_by_mjr_mnr_causes for more information
        """
        #-------------------------
        slicer_dl_ol = DFSlicer(
            single_slicers=[
                dict(column='MJR_CAUSE_CD', value='DL'), 
                dict(column='MNR_CAUSE_CD', value='OL')
            ], 
            name='DL_OL')
        #-----
        slicer_dl_eqf = DFSlicer(
            single_slicers=[
                dict(column='MJR_CAUSE_CD', value='DL'), 
                dict(column='MNR_CAUSE_CD', value='EQF')
            ], 
            name='DL_EQF')
        #-----
        xfmr_equip_typ_nms_of_interest = ['TRANSFORMER, OH', 'TRANSFORMER, UG']
        slicer_dl_eqf_xfmr = DFSlicer(
            single_slicers=[
                dict(column='MJR_CAUSE_CD', value='DL'), 
                dict(column='MNR_CAUSE_CD', value='EQF'), 
                dict(column='EQUIP_TYP_NM', value=xfmr_equip_typ_nms_of_interest)
            ], 
            name='DL_EQF_XFMR')
        #-----
        df_subset_slicers = [slicer_dl_ol, slicer_dl_eqf, slicer_dl_eqf_xfmr]
        #-------------------------
        if addtnl_df_subset_slicers is not None:
            df_subset_slicers.extend(addtnl_df_subset_slicers)
        #-------------------------
        subset_dfs_dict = MECPODf.get_rcpo_df_subset_by_mjr_mnr_causes(
            rcpo_df=rcpo_df, 
            df_subset_slicers=df_subset_slicers, 
            outg_rec_nb_col=outg_rec_nb_col, 
            mjr_mnr_causes_df=mjr_mnr_causes_df
        )
        return subset_dfs_dict
        
    @staticmethod
    def get_cpo_df_subset_excluding_mjr_mnr_causes( 
        cpo_df, 
        mjr_mnr_causes_to_exclude, 
        mjr_causes_to_exclude=None,
        mnr_causes_to_exclude=None,
        outg_rec_nb_col='index', 
        mjr_mnr_causes_df=None, 
        mjr_cause_col='MJR_CAUSE_CD', 
        mnr_cause_col='MNR_CAUSE_CD'
    ):
        r"""
        Returns a subset of cpo_df excluding those with major and/or minor causes in mjr_mnr_causes_to_exclude, 
          mjr_causes_to_exclude, mnr_causes_to_exclude.
        To ignore specific mjr/mnr cause pairs, use mjr_mnr_causes_to_exclude.
        To ignore mjr causes, regardless of mnr cause, use mjr_causes_to_exclude.
        To ignore mnr causes, regardless of mjr cause, use mnr_causes_to_exclude.
        NOTE: The user can supply mjr_mnr_causes_df, but if not supplied it will be built.

        mjr_mnr_causes_to_exclude:
            This is designed to be a list of dict objects (but it can also be a single dict object instead of a list of length 1
            if only a single needed).
            Each dict object, mjr_mnr_cause, should have keys 'mjr_cause', 'mnr_cause', and optionally 'addtnl_slicers'.
            mjr_mnr_cause['mjr_cause']:
                Should be a string specifying the major cause to be excluded.
            mjr_mnr_cause['mnr_cause']:
                Should be a string specifying the minor cause to be excluded.
            mjr_mnr_cause['addtnl_slicers']:
                Should be a list of dict/DataFrameSubsetSingleSlicer objects to be input into add_single_slicer (however, can be
                  a single dict/DataFrameSubsetSingleSlicer object if only one needed).
                If element is a dict, it should have key/value pairs to build a DataFrameSubsetSingleSlicer object
                  ==> keys = 'column', 'value', and optionally 'comparison_operator'

        mjr_mnr_causes_df:
            A DF with mjr/mnr cause information for each outage (NOTE: outg_rec_nb should be index of DF!).
            If not supplied, it will be built via DOVSOutages.get_mjr_mnr_causes_for_outages
        mjr_cause_col/mnr_cause_col:
            The columns in mjr_mnr_causes_df to find major and minor causes.
        """
        #-------------------------
        # Get outg_rec_nbs (series) and outg_rec_nbs_unq (list) from cpo_df
        outg_rec_nbs = MECPODf.get_outg_rec_nbs_from_cpo_df(cpo_df=cpo_df, idfr=outg_rec_nb_col)
        assert(len(cpo_df)==len(outg_rec_nbs)) # Important in ensuring proper selection towards end of function
        outg_rec_nbs_unq = outg_rec_nbs.unique().tolist()
        #-------------------------
        # Build mjr_mnr_causes_df if not supplied, and ensure all outg_rec_nbs found in mjr_mnr_causes_df.
        if mjr_mnr_causes_df is None:
            mjr_mnr_causes_df = DOVSOutages.get_mjr_mnr_causes_for_outages(
                outg_rec_nbs=outg_rec_nbs_unq, 
                include_equip_type=True, 
                set_outg_rec_nb_as_index=True
            )
            mjr_cause_col='MJR_CAUSE_CD'
            mnr_cause_col='MNR_CAUSE_CD'
        # Make sure all outg_rec_nbs_unq in mjr_mnr_causes_df
        assert(len(set(outg_rec_nbs_unq).difference(set(mjr_mnr_causes_df.index.tolist())))==0)
        #-------------------------
        # Get subset of mjr_mnr_causes_df excluding those with major and/or minor causes in mjr_mnr_causes_to_exclude, 
        #   mjr_causes_to_exclude, mnr_causes_to_exclude.
        mjr_mnr_causes_df = DOVSOutages.get_df_subset_excluding_mjr_mnr_causes(
            df=mjr_mnr_causes_df, 
            mjr_mnr_causes_to_exclude=mjr_mnr_causes_to_exclude, 
            mjr_causes_to_exclude=mjr_causes_to_exclude,
            mnr_causes_to_exclude=mnr_causes_to_exclude, 
            mjr_cause_col=mjr_cause_col, 
            mnr_cause_col=mnr_cause_col
        )
        #-------------------------
        # Use the outg_rec_nbs remaining in mjr_mnr_causes_df to select appropriate entries from cpo_df
        # Selection method below is why assertion above is important (see # Important in ensuring proper 
        #   selection towards end of function)
        return_df = cpo_df[outg_rec_nbs.isin(mjr_mnr_causes_df.index)]
        return return_df
        
    @staticmethod    
    def convert_rcpo_to_icpo_df(
        rcpo_df, 
        reason_to_ede_typeid_df, 
        cols_to_adjust=None, 
        is_norm=False, 
        counts_col='_nSNs', 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm', 
        SNs_tags=None, 
        return_columns_name='enddeviceeventtypeid'
    ):
        r"""
        Convert a Reason Counts Per Outage (rcpo) pd.DataFrame to a Id (enddeviceeventtypeid) Counts Per Outage (icpo).

        cols_to_adjust:
          if cols_to_adjust is None (default), adjust all columns EXCEPT for those containing SN info (as dictated by
            SNs_tags via MECPODf.remove_SNs_cols_from_rcpo_df)
          Intended to be a list of strings (i.e., the function will take care of level_0 values if the columns in rcpo_df
            are MultiIndex).
            However, a list of tuples will work as well when rcpo_df has MultiIndex columns, assuming the 0th element
              of the tuples aligns with the level 0 values in rcpo_df columns
          EASIEST TO SUPPLY LIST OF STRINGS!!!!!

        is_norm:
          Set to True of rcpo_df is normalized
          In this case, counts_col will be used to un-normalize the data before perfmoring the conversion, and then
            again the re-normalize the final data.

        counts_col:
          Should be a string.  If rcpo_df has MultiIndex columns, the level_0 value will be handled.
          Will still function properly if appropriate tuple/list is supplied instead of string
            e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming rcpo_df columns
                  have level_0 values equal to 'counts')
          EASIEST TO SUPPLY STRING!!!!!
          NOTE: Only really needed if is_norm==True or normalize_by_nSNs_included==True 
                (i.e., not needed when is_norm==normalize_by_nSNs_included==False)

        normalize_by_nSNs_included:
          Set to True if rcpo_df contains both the raw and normalized data.
          In this case, since these DFs should not be large, I will use the lazy solution of separating out the raw
            and normal components, sending them through this function separately, and then recombining.
            - Note: In AMIEndEvents.combine_two_reason_counts_per_outage_dfs, it was found that this splitting and re-joining
                    was MUCH slower than handling everything at once.  However, this procedure is much different and simpler,
                    so I don't think the difference in speed will be worth the effort of developing the method to handle
                    everything at once (if things do end up very slow, this should be re-visited!)

        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
          Used only if cols_to_adjust is None (as it is by default)
          NOTE: SNs_tags should contain strings, not tuples.
                If column is multiindex, the level_0 value will be handled below.
        """
        #----------------------------------------------------------------------------------------------------
        if normalize_by_nSNs_included:
            rcpo_df_raw  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_raw_col, droplevel=False)
            rcpo_df_nrm  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_nrm_col, droplevel=False)
            #-------------------------
            # normalize_by_nSNs_included must be False below or inifite loop!!!!!
            icpo_df_raw = MECPODf.convert_rcpo_to_icpo_df(
                rcpo_df=rcpo_df_raw, 
                reason_to_ede_typeid_df=reason_to_ede_typeid_df, 
                cols_to_adjust=cols_to_adjust, 
                is_norm=False, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col, 
                SNs_tags=SNs_tags
            )
            icpo_df_nrm = MECPODf.convert_rcpo_to_icpo_df(
                rcpo_df=rcpo_df_nrm, 
                reason_to_ede_typeid_df=reason_to_ede_typeid_df, 
                cols_to_adjust=cols_to_adjust, 
                is_norm=True, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col, 
                SNs_tags=SNs_tags
            )
            #-------------------------
            icpo_df = pd.merge(icpo_df_raw, icpo_df_nrm, left_index=True, right_index=True)
            assert(icpo_df_raw.shape[0]==icpo_df_nrm.shape[0]==icpo_df.shape[0])
            assert(icpo_df_raw.shape[1]+icpo_df_nrm.shape[1]==icpo_df.shape[1])
            #-------------------------
            icpo_df.columns.name=return_columns_name
            #-------------------------
            return icpo_df
        #----------------------------------------------------------------------------------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        icpo_df = rcpo_df.copy()
        #-------------------------
        # if cols_to_adjust is None, adjust all columns EXCEPT for those containing SN info
        if cols_to_adjust is None:
            cols_to_adjust = MECPODf.remove_SNs_cols_from_rcpo_df(icpo_df, SNs_tags=SNs_tags).columns.tolist()
        #-------------------------
        # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
        assert(icpo_df.columns.nlevels<=2)
        # When MECPODf.remove_SNs_cols_from_rcpo_df is called, cols_to_adjust will have correct tuple structure if icpo_df
        #   has MultiIndex columns 
        #     NOTE: if cols_to_adjust is supplied by user, one needs to ensure it has the correct
        #           dimensionality/type, e.g. each element should have length 2 if nlevel==2). 
        # If icpo_df has MultiIndex columns,  reason_to_ede_typeid_df will need to be updated to have
        #   a MultiIndex index.  
        #   counts_col will also need to be updated to be a tuple instead of string
        are_multiindex_cols = False
        if icpo_df.columns.nlevels==2:
            # Again, below only for raw OR norm, NOT BOTH.
            are_multiindex_cols = True
            assert(icpo_df.columns.get_level_values(0).nunique()==1)
            level_0_val = icpo_df.columns.get_level_values(0).unique().tolist()[0]
            #-----
            reason_to_ede_typeid_df=reason_to_ede_typeid_df.copy() #To not alter origin
            reason_to_ede_typeid_df.index = pd.MultiIndex.from_product([[level_0_val], reason_to_ede_typeid_df.index])
            #-------------------------
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
            #-------------------------
            # cols_to_adjust should contain elements which are strings, tuples, or lists.
            # If strings, since icpo_df.columns.nlevels==2, these need to be converted to a lists of tuples
            # If tuples/lists, need to make sure all have length==2
            assert(Utilities.are_all_list_elements_one_of_types_and_homogeneous(cols_to_adjust, [str, tuple, list]))
            if isinstance(cols_to_adjust[0], str):
                cols_to_adjust = pd.MultiIndex.from_product([[level_0_val], cols_to_adjust])
            else:
                assert(Utilities.are_list_elements_lengths_homogeneous(cols_to_adjust) and len(cols_to_adjust[0])==2)

        #-------------------------
        # Make sure all cols_to_adjust are in reason_to_ede_typeid_df
        assert(all([x in reason_to_ede_typeid_df.index for x in cols_to_adjust]))
        #-------------------------
        # If rcpo_df was normalized, it must first be un-normalized so the like-columns can be combined below
        if is_norm:
            assert(counts_col in icpo_df)
            icpo_df[cols_to_adjust] = icpo_df[cols_to_adjust].multiply(icpo_df[counts_col], axis='index')
        #-------------------------
        # Rename columns from reason to enddeviceeventtypeid
        # In general, this will create degenerate column names (i.e., multiple columns of same name)
        # This is by design, and will be useful in the combination step
        #-----
        # NOTE: If you call squeeze on a df with shape=(1,1), it will not return a series but instead returns the value
        #       This is the reason for the if statement below (this only happens when running over small data)
        if reason_to_ede_typeid_df.shape[0]==1:
            rename_cols_dict = {reason_to_ede_typeid_df.index[0] : reason_to_ede_typeid_df.iloc[0,0]}
        else:
            rename_cols_dict = {x:reason_to_ede_typeid_df.squeeze().to_dict()[x] for x in cols_to_adjust}
        # Apparently, rename cannot be used to rename multiple levels at the same time
        if are_multiindex_cols:
            icpo_df=icpo_df.rename(columns={k[1]:v for k,v in rename_cols_dict.items()}, level=1)
        else:
            icpo_df=icpo_df.rename(columns=rename_cols_dict)
        #-------------------------
        # Finally, combine like columns
        #   OLD VERSION OF CODE WILL NO LONGER BE SUPPORTED
        #       FutureWarning: DataFrame.groupby with axis=1 is deprecated. Do `frame.T.groupby(...)` without axis instead.
        #   OLD VERSION: icpo_df = icpo_df.groupby(icpo_df.columns, axis=1).sum()
        #   NEW VERSION: icpo_df = icpo_df.T.groupby(icpo_df.columns).sum().T
        icpo_df = icpo_df.T.groupby(icpo_df.columns).sum().T

        # If the columns were MultiIndex, the groupby.sum procedure collapses these back down to a
        #   single dimension of tuples.  Therefore, expand back out
        if are_multiindex_cols:
            icpo_df.columns=pd.MultiIndex.from_tuples(icpo_df.columns)
        #-------------------------
        # If rcpo_df was normalized, icpo_df must now be re-normalized
        if is_norm:
            adjusted_cols = list(set(rename_cols_dict.values()))
            if are_multiindex_cols:
                adjusted_cols = [(level_0_val,x) for x in adjusted_cols]
            # Maintain order in df, I suppose...
            adjusted_cols = [x for x in icpo_df.columns if x in adjusted_cols]
            #-----
            icpo_df[adjusted_cols] = icpo_df[adjusted_cols].divide(icpo_df[counts_col], axis='index')
        #-------------------------
        icpo_df.columns.name=return_columns_name
        #-------------------------
        return icpo_df

    # MOVED TO CPXDf
    @staticmethod    
    def combine_cpo_df_reasons_explicit(
        rcpo_df, 
        reasons_to_combine,
        combined_reason_name, 
        is_norm=False, 
        counts_col='_nSNs', 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        Combine multiple reason columns into a single reason.  Called _explicit because exact columns to combine must be explicitly
        given.  For a more flexible version, see combine_cpo_df_reasons
          e.g., combine reasons_to_combine = [
                    'Under Voltage (CA000400) occurred for meter', 
                    'Under Voltage (CA000400) for meter Voltage out of tolerance', 
                    'Under Voltage (Diagnostic 6) occurred for meter', 
                    'Diag6: Under Voltage, Element A occurred for meter', 
                    'Under Voltage (CA000400) occurred for meter:and C'
                ]
          into a single 'Under Voltage'
        This is intended for rcpo DFs, as icpo enddeviceeventtypeids already combine multiple reasons in general.
          However, this is no reason this can't also be used for icpo DF

        NOTE: Since we're combining columns within each row, there is no need to combine list elements such as _SNs etc. as
              these are shared by all columns in the row.

        reasons_to_combine:
          Intended to be a list of strings (i.e., the function will take care of level_0 values if the columns in rcpo_df
            are MultiIndex).
            However, a list of tuples will work as well when rcpo_df has MultiIndex columns, assuming the 0th element
              of the tuples aligns with the level 0 values in rcpo_df columns
          EASIEST TO SUPPLY LIST OF STRINGS!!!!!

        combined_reason_name:
          Name for output column made from combined columns

        is_norm:
          Set to True of rcpo_df is normalized
          In this case, counts_col will be used to un-normalize the data before perfmoring the conversion, and then
            again the re-normalize the final data.

        counts_col:
          Should be a string.  If rcpo_df has MultiIndex columns, the level_0 value will be handled.
          Will still function properly if appropriate tuple/list is supplied instead of string
            e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming rcpo_df columns
                  have level_0 values equal to 'counts')
          EASIEST TO SUPPLY STRING!!!!!
          NOTE: Only really needed if is_norm==True or normalize_by_nSNs_included==True 
                (i.e., not needed when is_norm==normalize_by_nSNs_included==False)

        normalize_by_nSNs_included:
          Set to True if rcpo_df contains both the raw and normalized data.
          In this case, since these DFs should not be large, I will use the lazy solution of separating out the raw
            and normal components, sending them through this function separately, and then recombining.
            - Note: In AMIEndEvents.combine_two_reason_counts_per_outage_dfs, it was found that this splitting and re-joining
                    was MUCH slower than handling everything at once.  However, this procedure is much different and simpler,
                    so I don't think the difference in speed will be worth the effort of developing the method to handle
                    everything at once (if things do end up very slow, this should be re-visited!)
        """
        #-------------------------
        #----------------------------------------------------------------------------------------------------
        if normalize_by_nSNs_included:
            rcpo_df_raw  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_raw_col, droplevel=False)
            rcpo_df_nrm  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_nrm_col, droplevel=False)
            #-------------------------
            # normalize_by_nSNs_included must be False below or inifite loop!!!!!
            rcpo_df_raw = MECPODf.combine_cpo_df_reasons_explicit(
                rcpo_df=rcpo_df_raw, 
                reasons_to_combine=reasons_to_combine, 
                combined_reason_name=combined_reason_name, 
                is_norm=False, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col
            )
            rcpo_df_nrm = MECPODf.combine_cpo_df_reasons_explicit(
                rcpo_df=rcpo_df_nrm, 
                reasons_to_combine=reasons_to_combine, 
                combined_reason_name=combined_reason_name, 
                is_norm=True, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col
            )
            #-------------------------
            rcpo_df = pd.merge(rcpo_df_raw, rcpo_df_nrm, left_index=True, right_index=True)
            assert(rcpo_df_raw.shape[0]==rcpo_df_nrm.shape[0]==rcpo_df.shape[0])
            assert(rcpo_df_raw.shape[1]+rcpo_df_nrm.shape[1]==rcpo_df.shape[1])
            #-------------------------
            return rcpo_df
        #----------------------------------------------------------------------------------------------------
        # Grab the original number of columns for comparison later
        n_cols_OG = rcpo_df.shape[1]
        #-------------------------
        rcpo_df = rcpo_df.copy()
        #-------------------------
        # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
        assert(rcpo_df.columns.nlevels<=2)
        # One needs to ensure that reasons_to_combine has the correct dimensionality/type, 
        #   e.g., each element should have length 2 if nlevel==2). 
        #   The same is true for counts_col.
        are_multiindex_cols = False
        if rcpo_df.columns.nlevels==2:
            # Again, below only for raw OR norm, NOT BOTH.
            are_multiindex_cols = True
            assert(rcpo_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpo_df.columns.get_level_values(0).unique().tolist()[0]
            #-------------------------
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
            #-------------------------
            # reasons_to_combine should contain elements which are strings, tuples, or lists.
            # If strings, since rcpo_df.columns.nlevels==2, these need to be converted to a lists of tuples
            # If tuples/lists, need to make sure all have length==2
            assert(Utilities.are_all_list_elements_one_of_types_and_homogeneous(reasons_to_combine, [str, tuple, list]))
            if isinstance(reasons_to_combine[0], str):
                reasons_to_combine = pd.MultiIndex.from_product([[level_0_val], reasons_to_combine])
            else:
                assert(Utilities.are_list_elements_lengths_homogeneous(reasons_to_combine) and len(reasons_to_combine[0])==2)

        #-------------------------
        # Make sure all reasons_to_combine are in rcpo_df
        assert(all([x in rcpo_df.columns for x in reasons_to_combine]))
        #-------------------------
        # If rcpo_df was normalized, it must first be un-normalized so the like-columns can be combined below
        if is_norm:
            assert(counts_col in rcpo_df)
            rcpo_df[reasons_to_combine] = rcpo_df[reasons_to_combine].multiply(rcpo_df[counts_col], axis='index')
        #-------------------------
        # Rename all columns in reasons_to_combine as combined_reason_name to create degenerate column names 
        #   (i.e., multiple columns of same name).  This is by design, and will be useful in the combination step
        rename_cols_dict = {x:combined_reason_name for x in reasons_to_combine}
        # Apparently, rename cannot be used to rename multiple levels at the same time
        if are_multiindex_cols:
            rcpo_df=rcpo_df.rename(columns={k[1]:v for k,v in rename_cols_dict.items()}, level=1)
        else:
            rcpo_df=rcpo_df.rename(columns=rename_cols_dict)
        #-------------------------
        # Finally, combine like columns
        #   OLD VERSION OF CODE WILL NO LONGER BE SUPPORTED
        #       FutureWarning: DataFrame.groupby with axis=1 is deprecated. Do `frame.T.groupby(...)` without axis instead.
        #   OLD VERSION: rcpo_df = rcpo_df.groupby(rcpo_df.columns, axis=1).sum()
        #   NEW VERSION: rcpo_df = rcpo_df.T.groupby(rcpo_df.columns).sum().T
        rcpo_df = rcpo_df.T.groupby(rcpo_df.columns).sum().T

        # If the columns were MultiIndex, the groupby.sum procedure collapses these back down to a
        #   single dimension of tuples.  Therefore, expand back out
        if are_multiindex_cols:
            rcpo_df.columns=pd.MultiIndex.from_tuples(rcpo_df.columns)
        #-------------------------
        # If rcpo_df was normalized, rcpo_df must now be re-normalized
        if is_norm:
            adjusted_cols = list(set(rename_cols_dict.values()))
            if are_multiindex_cols:
                adjusted_cols = [(level_0_val,x) for x in adjusted_cols]
            # Maintain order in df, I suppose...
            adjusted_cols = [x for x in rcpo_df.columns if x in adjusted_cols]
            #-----
            rcpo_df[adjusted_cols] = rcpo_df[adjusted_cols].divide(rcpo_df[counts_col], axis='index')
        #-------------------------
        # Make sure the final number of columns makes sense.
        # len(reasons_to_combine) columns are replaced by 1, for a net of len(reasons_to_combine)-1
        assert(n_cols_OG-rcpo_df.shape[1]==len(reasons_to_combine)-1)
        #-------------------------
        return rcpo_df
    
    # MOVED TO CPXDf
    @staticmethod
    def combine_degenerate_columns(
        rcpo_df
    ):
        r"""
        Combine any identically-named (i.e., degenerate) columns in rcpo_df
    
        The original method of using groupby with rcpo_df.columns is good for a clean one-liner
        HOWEVER, for large DFs I find it can take significantly more time, as the aggregation is
          done on all columns, regardless of whether or not they are degenerate
        THEREFORE, operate only on the degenerate columns!
        Original method:
            rcpo_df = rcpo_df.groupby(rcpo_df.columns, axis=1).sum() #Original, one-liner method
        """
        #-------------------------
        col_counts = rcpo_df.columns.value_counts()
        degen_cols = col_counts[col_counts>1].index.tolist()
        #-------------------------
        if len(degen_cols)==0:
            return rcpo_df
        #-------------------------
        for degen_col_i in degen_cols:
            # Build aggregate of degen_col_i
            agg_col_i = rcpo_df[degen_col_i].sum(axis=1)
            # Drop old degenerate columns
            rcpo_df = rcpo_df.drop(columns=degen_col_i)
            # Replace with aggregate column
            rcpo_df[degen_col_i] = agg_col_i
        #-------------------------
        return rcpo_df
    
    # MOVED TO CPXDf    
    @staticmethod
    def combine_cpo_df_reasons(
        rcpo_df, 
        patterns_and_replace=None, 
        addtnl_patterns_and_replace=None, 
        is_norm=False, 
        counts_col='_nSNs', 
        normalize_by_nSNs_included=False, 
        initial_strip=True,
        initial_punctuation_removal=True, 
        level_0_raw_col='counts', 
        level_0_nrm_col='counts_norm',
        return_red_to_org_cols_dict=False
    ):
        r"""
        Combine groups of reasons according to patterns_and_replace.

        NOTE!!!!!!!!!!!!!:
          Typically, one should keep patterns_and_replace=None.  When this is the case, dflt_patterns_and_replace
            will be used.
          If one wants to add to dflt_patterns_and_replace, use the addtnl_patterns_and_replace argument.

        patterns_and_replace/addtnl_patterns_and_replace:
          A list of tuples (or lists) of length 2.
          Typical value = ['.*cleared.*', '.*Test Mode.*']
          For each item in the list:
            first element should be a regex pattern for which to search 
            second element is replacement

            DON'T FORGET ABOUT BACKREFERENCING!!!!!
              e.g.
                reason_i = 'I am a string named Jesse Thomas Buxton'
                patterns_and_replace_i = (r'(Jesse) (Thomas) (Buxton)', r'\1 \3')
                  reason_i ===> 'I am a string named Jesse Buxton'

        is_norm:
          Set to True of rcpo_df is normalized
          In this case, counts_col will be used to un-normalize the data before perfmoring the conversion, and then
            again the re-normalize the final data.

        counts_col:
          Should be a string.  If rcpo_df has MultiIndex columns, the level_0 value will be handled.
          NOTE: Only really needed if is_norm==True or normalize_by_nSNs_included==True 
                (i.e., not needed when is_norm==normalize_by_nSNs_included==False)

        normalize_by_nSNs_included:
          Set to True if rcpo_df contains both the raw and normalized data.
          In this case, since these DFs should not be large, I will use the lazy solution of separating out the raw
            and normal components, sending them through this function separately, and then recombining.
            - Note: In AMIEndEvents.combine_two_reason_counts_per_outage_dfs, it was found that this splitting and re-joining
                    was MUCH slower than handling everything at once.  However, this procedure is much different and simpler,
                    so I don't think the difference in speed will be worth the effort of developing the method to handle
                    everything at once (if things do end up very slow, this should be re-visited!)
        """
        #----------------------------------------------------------------------------------------------------
        if normalize_by_nSNs_included:
            rcpo_df_raw  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_raw_col, droplevel=False)
            rcpo_df_nrm  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_nrm_col, droplevel=False)
            #-------------------------
            # normalize_by_nSNs_included must be False below or inifite loop!!!!!
            rcpo_df_raw, red_to_org_cols_dict_raw = MECPODf.combine_cpo_df_reasons(
                rcpo_df=rcpo_df_raw, 
                patterns_and_replace=patterns_and_replace, 
                addtnl_patterns_and_replace=addtnl_patterns_and_replace, 
                is_norm=False, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                initial_strip=initial_strip,
                initial_punctuation_removal=initial_punctuation_removal, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col, 
                return_red_to_org_cols_dict=True
            )
            rcpo_df_nrm, red_to_org_cols_dict_nrm = MECPODf.combine_cpo_df_reasons(
                rcpo_df=rcpo_df_nrm, 
                patterns_and_replace=patterns_and_replace, 
                addtnl_patterns_and_replace=addtnl_patterns_and_replace, 
                is_norm=True, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                initial_strip=initial_strip,
                initial_punctuation_removal=initial_punctuation_removal, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col, 
                return_red_to_org_cols_dict=True
            )
            #-------------------------
            rcpo_df = pd.merge(rcpo_df_raw, rcpo_df_nrm, left_index=True, right_index=True)
            assert(rcpo_df_raw.shape[0]==rcpo_df_nrm.shape[0]==rcpo_df.shape[0])
            assert(rcpo_df_raw.shape[1]+rcpo_df_nrm.shape[1]==rcpo_df.shape[1])
            #-------------------------
            if return_red_to_org_cols_dict:
                #assert(len(set(red_to_org_cols_dict_raw.keys()).symmetric_difference(red_to_org_cols_dict_raw.keys()))==0)
                #red_to_org_cols_dict = {**red_to_org_cols_dict_raw, **red_to_org_cols_dict_nrm}
                red_to_org_cols_dict = {level_0_raw_col:red_to_org_cols_dict_raw, 
                                        level_0_nrm_col:red_to_org_cols_dict_nrm}
                return rcpo_df, red_to_org_cols_dict
            #-------------------------
            return rcpo_df
        #----------------------------------------------------------------------------------------------------
        # OLD VERSION
        # dflt_patterns_and_replace = [
            # (r'(failed consecutively).*', r'\1'), 
            # (r'A (NET_MGMT command) was sent from .* with a key that (has insufficient privileges).*', r'\1 \2'), 
            # (r'(Ignoring).*(Read) data for device as it has (time in the future)', r'\1 \2: \3'), 
            # (r'(Last Gasp - NIC power lost for device).*', r'\1'), 
            # (r'(Last Gasp State).*', r'\1'), 
            # (r'(Meter needs explicit time sync).*', r'\1'), 
            # (r'(NET_MGMT command failed consecutively).*', r'\1'), 
            # (r'(NIC Link Layer Handshake Failed).*', r'\1'), 
            # (r'(NVRAM Error).*', r'\1'), 
            # (r'(Requested operation).*(could not be applied to the given device type and firmware version)', r'\1 \2'), 
            # (r'(Secure association operation failed consecutively).*', r'\1'), 
            # (r'Low Potential.*occurred for meter', 'Under Voltage for meter'),
            # (r'.*(Under Voltage).*(for meter).*', r'\1 \2'), 
        # ]
        
        dflt_patterns_and_replace = [
            (r'(?:A\s*)?(NET_MGMT command) was sent with a key that (has insufficient privileges).*', r'\1 \2'), 
            (r'(Ignoring).*(Read) data for device as it has (time in the future)', r'\1 \2: \3'), 
            (r'(Device Failed).*', r'\1'), 
            (r'(Last Gasp - NIC power lost for device).*', 'Last Gasp'), 
            (r'(Last Gasp State).*', 'Last Gasp'), 
            (r'(Meter needs explicit time sync).*', r'\1'), 
            (r'(NET_MGMT command failed consecutively).*', r'\1'), 
            (r'(NIC Link Layer Handshake Failed).*', r'\1'), 
            (r'(NVRAM Error).*', r'\1'), 
            (r'(Requested operation).*(could not be applied).*', r'\1 \2'), 
            (r'(Secure association operation failed consecutively).*', r'\1'), 
            (r'Low Potential.*occurred for meter', 'Under Voltage for meter'),
            (r'.*(Over Voltage).*', r'\1'), 
            (r'.*(Under Voltage).*', r'\1'), 
            (
                (
                    r'((Meter detected tampering\s*(?:\(.*\))?\s*).*)|'\
                    r'((Tamper)\s*(?:\(.*\))?\s*detected).*|'\
                    r'(Meter event Tamper Attempt Suspected).*|'\
                    r'Meter detected a Tamper Attempt'
                ), 
                'Tamper Detected'
            ), 
            (r'Low\s*(?:Battery|Potential)\s*(?:\(.*\))?\s*.*', 'Low Battery')
        ]
        #-------------------------
        if patterns_and_replace is None:
            patterns_and_replace = dflt_patterns_and_replace
        #-----
        if addtnl_patterns_and_replace is not None:
            patterns_and_replace.extend(addtnl_patterns_and_replace)
        #-------------------------
        rcpo_df = rcpo_df.copy()
        #-------------------------
        assert(rcpo_df.columns.nlevels<=2)
        are_multiindex_cols = False
        # If rcpo_df has MultiIndex columns, there must only be a single unique value for level 0
        # In such a case, it is easiest for me to simply strip the level 0 value, perform my operations,
        #   and add it back at the end
        if rcpo_df.columns.nlevels==2:
            are_multiindex_cols = True
            # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
            assert(rcpo_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpo_df.columns.get_level_values(0).unique().tolist()[0]
            rcpo_df = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_val, droplevel=True)
        #-------------------------
        # The operation: df[some_cols] = df[some_cols].multiple(df[a_col], axis='index')
        #   cannot be performed when there is degeneracy in some_cols (i.e., there are repeated names in some_cols)
        #   This is a necessary step when is_norm=True, as the values must be un-normalized, combined, and then re-normalized
        #   Therefore, one cannot simply change column names on-the-fly, as was done in initial development.
        #     Thus, the code is a bit more cumbersome than seems necessary
        #-------------------------
        columns_org = rcpo_df.columns.tolist()
        #-------------------------
        if initial_punctuation_removal:
            reduce_cols_dict_0 = {x:AMIEndEvents.remove_trailing_punctuation_from_reason(reason=x, include_normal_strip=initial_strip) 
                                  for x in columns_org}
            reduce_cols_dict = {org_col:AMIEndEvents.reduce_end_event_reason(reason=red_col_0, patterns=patterns_and_replace, verbose=False) 
                                for org_col,red_col_0 in reduce_cols_dict_0.items()}
        else:
            reduce_cols_dict = {x:AMIEndEvents.reduce_end_event_reason(reason=x, patterns=patterns_and_replace, verbose=False) for x in columns_org}

        # Only keep columns in reduce_cols_dict which will be changed
        reduce_cols_dict = {k:v for k,v in reduce_cols_dict.items() if k!=v}
        #-------------------------
        # If rcpo_df was normalized, it must first be un-normalized so the like-columns can be combined below
        if is_norm:
            assert(counts_col in rcpo_df)
            cols_to_adjust = list(reduce_cols_dict.keys())
            rcpo_df[cols_to_adjust] = rcpo_df[cols_to_adjust].multiply(rcpo_df[counts_col], axis='index')
        #-------------------------
        # Rename columns according to reduce_cols_dict
        # This will create degenerate column names (i.e., multiple columns of same name)
        # This is by design, and will be useful in the combination step
        rcpo_df = rcpo_df.rename(columns=reduce_cols_dict)
        #-------------------------
        # Finally, combine like columns
        rcpo_df = MECPODf.combine_degenerate_columns(rcpo_df = rcpo_df)
        #-------------------------
        # If rcpo_df was normalized originally, it must now be re-normalized
        if is_norm:
            adjusted_cols = list(set(reduce_cols_dict.values()))
            rcpo_df[adjusted_cols] = rcpo_df[adjusted_cols].divide(rcpo_df[counts_col], axis='index')
        #-------------------------
        # If the columns of rcpo_df were originally MultiIndex, make them so again
        if are_multiindex_cols:
            rcpo_df = Utilities_df.prepend_level_to_MultiIndex(
                df=rcpo_df, 
                level_val=level_0_val, 
                level_name=None, 
                axis=1
            )
        #-------------------------
        if return_red_to_org_cols_dict:
            # Want a collection of all org_cols which were grouped into each red_col
            # So, need something similar to an inverse of reduce_cols_dict (not exactly the inverse)
            red_to_org_cols_dict = {}
            for org_col, red_col in reduce_cols_dict.items():
                if red_col in red_to_org_cols_dict.keys():
                    red_to_org_cols_dict[red_col].append(org_col)
                else:
                    red_to_org_cols_dict[red_col] = [org_col]
            return rcpo_df, red_to_org_cols_dict
        #-------------------------
        return rcpo_df
        
    # MOVED TO CPXDf    
    @staticmethod    
    def remove_reasons_explicit_from_rcpo_df(
        rcpo_df, 
        reasons_to_remove
    ):
        r"""
        Called _explicit because exact columns to removed must be explicitly given.  
        For a more flexible version, see remove_reasons_from_rcpo_df
        
        reasons_to_remove:
          Should be a list of strings.  If a given df has MultiIndex columns, this will be handled.
          Reasons are only removed if they exist (obviously)
        """
        #-------------------------
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            reasons_to_remove = [x for x in reasons_to_remove if x in rcpo_df.columns]
        else:
            level_0_vals = rcpo_df.columns.get_level_values(0).unique()
            reasons_to_remove = [(level_0_val, reason) for level_0_val in level_0_vals for reason in reasons_to_remove]
            reasons_to_remove  = [x for x in reasons_to_remove if x in rcpo_df.columns]
        rcpo_df = rcpo_df.drop(columns=reasons_to_remove)
        return rcpo_df
        
    # MOVED TO CPXDf    
    @staticmethod
    def remove_reasons_from_rcpo_df(
        rcpo_df, 
        regex_patterns_to_remove, 
        ignore_case=True
    ):
        r"""
        Remove any columns from rcpo_df where any of the patterns in regex_patterns_to_remove are found

        regex_patterns_to_remove:
          Should be a list of regex patterns (i.e., strings)
        """
        #-------------------------
        assert(rcpo_df.columns.nlevels<=2)
        if rcpo_df.columns.nlevels==1:
            reasons = rcpo_df.columns
        else:
            reasons = rcpo_df.columns.get_level_values(1)
        reasons = reasons.tolist() #Utilities.find_in_list_with_regex wants a list input
        #-------------------------
        col_idxs_to_remove = Utilities.find_idxs_in_list_with_regex(
            lst=reasons, 
            regex_pattern=regex_patterns_to_remove, 
            ignore_case=ignore_case
        )
        cols_to_remove = rcpo_df.columns[col_idxs_to_remove]
        #-------------------------
        rcpo_df = rcpo_df.drop(columns=cols_to_remove)
        return rcpo_df

        
    # Moved to CPXDf
    @staticmethod
    def delta_cpo_df_reasons(
        rcpo_df, 
        reasons_1,
        reasons_2,
        delta_reason_name, 
        is_norm=False, 
        counts_col='_nSNs', 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        Find difference between two reasons.
          e.g., power downs minus power ups

        NOTE: Since we're combining columns within each row, there is no need to combine list elements such as _SNs etc. as
              these are shared by all columns in the row.

        reasons_1, reasons_2:
          Intended to be a strings (i.e., the function will take care of level_0 values if the columns in rcpo_df
            are MultiIndex).
            However, tuples will work as well when rcpo_df has MultiIndex columns, assuming the 0th element
              of the tuples aligns with the level 0 values in rcpo_df columns
          EASIEST TO SUPPLY LIST OF STRINGS!!!!!

        delta_reason_name:
          Name for output column made from difference of two columns

        is_norm:
          Set to True of rcpo_df is normalized
          In this case, counts_col will be used to un-normalize the data before perfmoring the conversion, and then
            again the re-normalize the final data.

        counts_col:
          Should be a string.  If rcpo_df has MultiIndex columns, the level_0 value will be handled.
          Will still function properly if appropriate tuple/list is supplied instead of string
            e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming rcpo_df columns
                  have level_0 values equal to 'counts')
          EASIEST TO SUPPLY STRING!!!!!
          NOTE: Only really needed if is_norm==True or normalize_by_nSNs_included==True 
                (i.e., not needed when is_norm==normalize_by_nSNs_included==False)

        normalize_by_nSNs_included:
          Set to True if rcpo_df contains both the raw and normalized data.
          In this case, since these DFs should not be large, I will use the lazy solution of separating out the raw
            and normal components, sending them through this function separately, and then recombining.
            - Note: In AMIEndEvents.combine_two_reason_counts_per_outage_dfs, it was found that this splitting and re-joining
                    was MUCH slower than handling everything at once.  However, this procedure is much different and simpler,
                    so I don't think the difference in speed will be worth the effort of developing the method to handle
                    everything at once (if things do end up very slow, this should be re-visited!)
        """
        #-------------------------
        #----------------------------------------------------------------------------------------------------
        if normalize_by_nSNs_included:
            rcpo_df_raw  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_raw_col, droplevel=False)
            rcpo_df_nrm  = MECPODf.project_level_0_columns_from_rcpo_wide(rcpo_df, level_0_nrm_col, droplevel=False)
            #-------------------------
            # normalize_by_nSNs_included must be False below or inifite loop!!!!!
            rcpo_df_raw = MECPODf.delta_cpo_df_reasons(
                rcpo_df=rcpo_df_raw, 
                reasons_1=reasons_1, 
                reasons_2=reasons_2, 
                delta_reason_name=delta_reason_name, 
                is_norm=False, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col
            )
            rcpo_df_nrm = MECPODf.delta_cpo_df_reasons(
                rcpo_df=rcpo_df_nrm, 
                reasons_1=reasons_1, 
                reasons_2=reasons_2, 
                delta_reason_name=delta_reason_name, 
                is_norm=True, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col
            )
            #-------------------------
            rcpo_df = pd.merge(rcpo_df_raw, rcpo_df_nrm, left_index=True, right_index=True)
            assert(rcpo_df_raw.shape[0]==rcpo_df_nrm.shape[0]==rcpo_df.shape[0])
            assert(rcpo_df_raw.shape[1]+rcpo_df_nrm.shape[1]==rcpo_df.shape[1])
            #-------------------------
            return rcpo_df
        #----------------------------------------------------------------------------------------------------
        rcpo_df = rcpo_df.copy()
        #-------------------------
        # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
        assert(rcpo_df.columns.nlevels<=2)
        # One needs to ensure that reasons_1, reasons_2, counts_col all have the correct dimensionality/type, 
        #   e.g., each should have length 2 if nlevel==2). 
        are_multiindex_cols = False
        if rcpo_df.columns.nlevels==2:
            # Again, below only for raw OR norm, NOT BOTH.
            are_multiindex_cols = True
            assert(rcpo_df.columns.get_level_values(0).nunique()==1)
            level_0_val = rcpo_df.columns.get_level_values(0).unique().tolist()[0]
            #-------------------------
            assert(Utilities.is_object_one_of_types(reasons_1, [str, tuple, list]))
            if isinstance(reasons_1, str):
                reasons_1 = (level_0_val, reasons_1)
            else:
                assert(len(reasons_1)==2)
            #-------------------------
            assert(Utilities.is_object_one_of_types(reasons_2, [str, tuple, list]))
            if isinstance(reasons_2, str):
                reasons_2 = (level_0_val, reasons_2)
            else:
                assert(len(reasons_2)==2)
            #-------------------------
            assert(Utilities.is_object_one_of_types(delta_reason_name, [str, tuple, list]))
            if isinstance(delta_reason_name, str):
                delta_reason_name = (level_0_val, delta_reason_name)
            else:
                assert(len(delta_reason_name)==2)
            #-------------------------
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
        #-------------------------
        # Make sure reasons_1,reasons_2 are in rcpo_df
        assert(reasons_1 in rcpo_df)
        assert(reasons_2 in rcpo_df)
        #-------------------------
        # If rcpo_df was normalized, it must first be un-normalized so the like-columns can be combined below
        if is_norm:
            assert(counts_col in rcpo_df)
            rcpo_df[[reasons_1, reasons_2]] = rcpo_df[[reasons_1, reasons_2]].multiply(rcpo_df[counts_col], axis='index')
        #-------------------------
        rcpo_df[delta_reason_name] = rcpo_df[reasons_1]-rcpo_df[reasons_2]
        #-------------------------
        # If rcpo_df was normalized, rcpo_df must now be re-normalized
        if is_norm:
            rcpo_df[[reasons_1, reasons_2, delta_reason_name]] = rcpo_df[[reasons_1, reasons_2, delta_reason_name]].divide(rcpo_df[counts_col], axis='index')
        #-------------------------
        return rcpo_df
        
    # MOVED TO CPXDfBuilder
    @staticmethod  
    def normalize_rcpo_df_by_time_interval(
        rcpo_df, 
        days_min_outg_td_window, 
        days_max_outg_td_window, 
        cols_to_adjust=None, 
        SNs_tags=None, 
        inplace=False
    ):
        r"""
        Normalize a Reason Counts Per Outage (RCPO) pd.DataFrame by the time width around the outage used to construct it.
        It is assumed that the limits on the time window are INCLUSIVE on the front and EXCLUSIVE on the back, i.e., 
          [outage_date-days_max_outg_td_window, outage_date-days_min_outg_td_window).
        Thus, the width of the window should be calculated as:
          window_widths_days = days_max_outg_td_window-days_min_outg_td_window
          e.g.:  Assume the window goes from 1 day before to 2 days before:
                   ==> window_widths_days = 1 days = 2-1
          e.g.:  Assume the window goes from 5 days before to 10 days before:
                   ==> window_widths_days = 5 days = 10-5

        cols_to_adjust:
          if cols_to_adjust is None (default), adjust all columns EXCEPT for those containing SN info (as dictated by
            SNs_tags via MECPODf.remove_SNs_cols_from_rcpo_df)

        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
          Used only if cols_to_adjust is None (as it is by default)
          NOTE: SNs_tags should contain strings, not tuples.
                If column is multiindex, the level_0 value will be handled below.
        """
        #-------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        if not inplace:
            rcpo_df = rcpo_df.copy()
        #-------------------------
        window_widths_days = (days_max_outg_td_window-days_min_outg_td_window)
        #-------------------------
        # if cols_to_adjust is None, adjust all columns EXCEPT for those containing SN info
        if cols_to_adjust is None:
            cols_to_adjust = MECPODf.remove_SNs_cols_from_rcpo_df(rcpo_df, SNs_tags=SNs_tags).columns.tolist()
        #-------------------------
        rcpo_df[cols_to_adjust] = rcpo_df[cols_to_adjust]/window_widths_days
        #-------------------------
        return rcpo_df

    # MOVED TO CPXDf    
    @staticmethod
    def get_total_event_counts(
        cpo_df, 
        output_col='total_counts', 
        sort_output=False, 
        SNs_tags=None, 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        Basically just sums up the number of events for each index.
        The nSNs (and SNs) columns should not be included in the count, hence the need for the SNs tags argument.
        If normalize_by_nSNs_included==True, the return DF with have MultiIndex columns matching those of cpo_df
          (i.e., with level_0_raw_col and level_0_nrm_col)
          
        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        """
        #----------------------------------------------------------------------------------------------------
        if normalize_by_nSNs_included:
            assert(cpo_df.columns.nlevels==2)
            assert((level_0_raw_col in cpo_df.columns.get_level_values(0).unique() and 
                    level_0_nrm_col in cpo_df.columns.get_level_values(0).unique()))
            cpo_df_raw  = MECPODf.project_level_0_columns_from_rcpo_wide(cpo_df, level_0_raw_col, droplevel=False)
            cpo_df_nrm  = MECPODf.project_level_0_columns_from_rcpo_wide(cpo_df, level_0_nrm_col, droplevel=False)
            #-------------------------
            total_df_raw = MECPODf.get_total_event_counts(
                cpo_df=cpo_df_raw, 
                output_col=output_col, 
                sort_output=False, 
                SNs_tags=SNs_tags, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col=level_0_raw_col, 
                level_0_nrm_col=level_0_nrm_col
            )
            total_df_raw = Utilities_df.prepend_level_to_MultiIndex(
                df=total_df_raw, 
                level_val=level_0_raw_col, 
                level_name=None, 
                axis=1
            )    
            #-------------------------
            total_df_nrm = MECPODf.get_total_event_counts(
                cpo_df=cpo_df_nrm, 
                output_col=output_col, 
                sort_output=False, 
                SNs_tags=SNs_tags, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col=level_0_raw_col, 
                level_0_nrm_col=level_0_nrm_col
            )
            total_df_nrm = Utilities_df.prepend_level_to_MultiIndex(
                df=total_df_nrm, 
                level_val=level_0_nrm_col, 
                level_name=None, 
                axis=1
            )
            #-------------------------
            assert(total_df_raw.shape[0]==total_df_nrm.shape[0])
            org_len = total_df_raw.shape[0]
            total_df = pd.merge(total_df_raw, total_df_nrm, left_index=True, right_index=True, how='inner')
            assert(total_df.shape[0]==org_len)
            #-------------------------
            if sort_output:
                total_df = total_df.sort_values(by=(level_0_raw_col,output_col), ascending=False)
            #-------------------------
            return total_df
        #----------------------------------------------------------------------------------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        non_SNs_cols = MECPODf.get_non_SNs_cols_from_cpo_df(
            cpo_df=cpo_df, 
            SNs_tags=SNs_tags
        )
        total_df = cpo_df[non_SNs_cols]
        #-------------------------
        total_df = total_df.sum(axis=1).to_frame(name=output_col)
        if sort_output:
            total_df = total_df.sort_values(by=output_col, ascending=False)
        return total_df
        
    @staticmethod
    def append_total_events_column_to_df(
        cpo_df, 
        output_col='total_counts', 
        SNs_tags=None, 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        Build the total events df (which is a DF with one column, so really just a series) and append to cpo_df
        
        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        """
        #-------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        total_df = MECPODf.get_total_event_counts(
            cpo_df=cpo_df, 
            output_col=output_col, 
            SNs_tags=SNs_tags, 
            normalize_by_nSNs_included=normalize_by_nSNs_included, 
            level_0_raw_col = level_0_raw_col, 
            level_0_nrm_col = level_0_nrm_col
        )
        if normalize_by_nSNs_included:
            assert(total_df.shape[1]==2)
        else:
            assert(total_df.shape[1]==1)
        #-------------------------
        org_shape = cpo_df.shape
        assert(output_col not in cpo_df.columns)
        assert(total_df.shape[0]==cpo_df.shape[0])
        #-----
        assert(cpo_df.columns.nlevels==total_df.columns.nlevels)
        cpo_df = pd.merge(cpo_df, total_df, left_index=True, right_index=True, how='inner')
        assert(cpo_df.shape[1]==org_shape[1]+total_df.shape[1])
        assert(cpo_df.shape[0]==org_shape[0])
        #-------------------------
        # If normalize_by_nSNs_included, then new columns are both added to the end, whereas I'd rather they
        #   be within their raw/nrm subgroups.  The following line achieves this
        if normalize_by_nSNs_included:
            cpo_df.sort_index(axis=1, level=0, inplace=True)
        #-------------------------
        return cpo_df
        
    @staticmethod
    def get_cpo_df_subset_below_max_total_counts(
        cpo_df, 
        max_total_counts, 
        SNs_tags=None, 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        Calculate the total reason counts for each index of cpo_df, and return only those with 
        values less than max_total_counts.

        max_total_counts:
          The maximum number of total_counts to included in returned DF.
          For the case where normalize_by_nSNs_included==True, max_total_counts can be a list/tuple of length 2, 
              in which case the 0th element is the limit for level_0_raw_col and the 1st is the limit for level_0_nrm_col.
            If only one value is given, it will be assumed to be for level_0_raw_col
              i.e., max_total_counts --> [max_total_counts, None]
            To include limit for level_0_nrm_col but exclude for level_0_raw_col, input 
              max_total_counts = [None, max_total_counts_nrm]

        Note: If max_total_counts is None, simply return cpo_df
        
        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        """
        #-------------------------
        if max_total_counts is None:
            return cpo_df
        #-------------------------
        if SNs_tags is None:
            SNs_tags=MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols()
        #-------------------------
        total_counts_col = 'total_counts'
        org_columns = cpo_df.columns #Going to add total_counts column, so this will make it easier to drop at the end 
                                     # (if, e.g., normalize_by_nSNs_included is True, otherwsie straightforward either way)
        cpo_df = MECPODf.append_total_events_column_to_df(
            cpo_df=cpo_df, 
            output_col=total_counts_col, 
            SNs_tags=SNs_tags, 
            normalize_by_nSNs_included=normalize_by_nSNs_included, 
            level_0_raw_col = level_0_raw_col, 
            level_0_nrm_col = level_0_nrm_col       
        )
        #-------------------------
        if not normalize_by_nSNs_included:
            cpo_df = cpo_df[cpo_df[total_counts_col] < max_total_counts]
        else:
            assert(Utilities.is_object_one_of_types(max_total_counts, [int, float, list, tuple]))
            if Utilities.is_object_one_of_types(max_total_counts, [list, tuple]):
                assert(len(max_total_counts)==2)
            else:
                max_total_counts = [max_total_counts, None]
            #----------
            assert(not(max_total_counts[0] is None and max_total_counts[1] is None))
            #----------
            if max_total_counts[0] is not None and max_total_counts[1] is not None:
                cpo_df = cpo_df[(cpo_df[(level_0_raw_col, total_counts_col)] < max_total_counts[0]) & 
                                (cpo_df[(level_0_nrm_col, total_counts_col)] < max_total_counts[1])]
            elif max_total_counts[0] is not None:
                cpo_df = cpo_df[cpo_df[(level_0_raw_col, total_counts_col)] < max_total_counts[0]]
            elif max_total_counts[1] is not None:
                cpo_df = cpo_df[cpo_df[(level_0_nrm_col, total_counts_col)] < max_total_counts[1]]
            else:
                assert(0)
        #-------------------------
        cols_to_drop = [x for x in cpo_df.columns if x not in org_columns]
        cpo_df = cpo_df.drop(columns=cols_to_drop)
        #-------------------------
        return cpo_df

    # MOVED TO CPXDf       
    @staticmethod
    def get_top_reasons_subset_from_cpo_df(
        cpo_df, 
        n_reasons_to_include=10,
        combine_others=True, 
        output_combine_others_col='Other Reasons', 
        SNs_tags=None, 
        is_norm=False, 
        counts_col='_nSNs', 
        normalize_by_nSNs_included=False, 
        level_0_raw_col = 'counts', 
        level_0_nrm_col = 'counts_norm'
    ):
        r"""
        Project out the top n_reasons_to_include reasons from cpo_df.
        The order is taken to be:
          reason_order = cpo_df.mean().sort_values(ascending=False).index.tolist()

        output_combine_others_col:
          The output column name for the combined others. 
          This should be a string, even if cpo_df has MultiIndex columns (such a case will be handled below!)

        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        NOTE: SNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled in MECPODf.find_SNs_cols_idxs_from_cpo_df.

        --------------------
        THE FOLLOWING ARE ONLY NEEDED WHEN combine_others==True
            is_norm
            counts_col
            normalize_by_nSNs_included
            level_0_raw_col 
            level_0_nrm_col

            counts_col:
              Should be a string.  If cpo_df has MultiIndex columns, the level_0 value will be handled.
              Will still function properly if appropriate tuple/list is supplied instead of string
                e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming cpo_df columns
                      have level_0 values equal to 'counts')
              EASIEST TO SUPPLY STRING!!!!!
              NOTE: Only really needed if is_norm==True or normalize_by_nSNs_included==True 
                    (i.e., not needed when is_norm==normalize_by_nSNs_included==False)
        """
        #----------------------------------------------------------------------------------------------------
        if normalize_by_nSNs_included:
            cpo_df_raw  = MECPODf.project_level_0_columns_from_rcpo_wide(cpo_df, level_0_raw_col, droplevel=False)
            cpo_df_nrm  = MECPODf.project_level_0_columns_from_rcpo_wide(cpo_df, level_0_nrm_col, droplevel=False)
            #-------------------------
            # normalize_by_nSNs_included must be False below or inifite loop!!!!!
            cpo_df_raw = MECPODf.get_top_reasons_subset_from_cpo_df(
                cpo_df=cpo_df_raw, 
                n_reasons_to_include=n_reasons_to_include,
                combine_others=combine_others, 
                output_combine_others_col=output_combine_others_col, 
                SNs_tags=SNs_tags, 
                is_norm=False, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col
            )
            cpo_df_nrm = MECPODf.get_top_reasons_subset_from_cpo_df(
                cpo_df=cpo_df_nrm, 
                n_reasons_to_include=n_reasons_to_include,
                combine_others=combine_others, 
                output_combine_others_col=output_combine_others_col, 
                SNs_tags=SNs_tags, 
                is_norm=True, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=False, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col
            )
            #-------------------------
            cpo_df = pd.merge(cpo_df_raw, cpo_df_nrm, left_index=True, right_index=True)
            assert(cpo_df_raw.shape[0]==cpo_df_nrm.shape[0]==cpo_df.shape[0])
            assert(cpo_df_raw.shape[1]+cpo_df_nrm.shape[1]==cpo_df.shape[1])
            #-------------------------
            return cpo_df
        #----------------------------------------------------------------------------------------------------
        # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
        assert(cpo_df.columns.nlevels<=2)
        if cpo_df.columns.nlevels==2:
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                assert(cpo_df.columns.get_level_values(0).nunique()==1)
                level_0_val = cpo_df.columns.get_level_values(0).unique().tolist()[0]
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
        #-------------------------
        if combine_others and is_norm:
            # In this case, counts_col will be needed for combine
            # So, grab the values to be used later
            assert(counts_col in cpo_df.columns)
            counts_col_vals = cpo_df[counts_col]
        #-------------------------
        non_SNs_cols = MECPODf.get_non_SNs_cols_from_cpo_df(
            cpo_df=cpo_df, 
            SNs_tags=SNs_tags
        )
        cpo_df = cpo_df[non_SNs_cols]
        
        # The counts_col is typically removed above with get_non_SNs_cols_from_cpo_df.
        # However, it is not always removed (as, e.g., when sent from MECPOCollection.get_top_reasons_subset_from_merged_cpo_df
        #   it will have a random name which will not be caught by MECPODf.get_non_SNs_cols_from_cpo_df
        # It is important that it be removed, so it does not affect the ordering of reasons
        if counts_col in cpo_df.columns:
            cpo_df = cpo_df.drop(columns=[counts_col])
        
        # Make sure all remaining columns are numeric
        assert(cpo_df.shape[1]==cpo_df.select_dtypes('number').shape[1])
        #-------------------------
        reason_order = cpo_df.mean().sort_values(ascending=False).index.tolist()
        reasons_to_include = reason_order[:n_reasons_to_include]
        other_reasons      = reason_order[n_reasons_to_include:]
        #-------------------------
        if combine_others:
            if is_norm:
                # counts_col is needed for combine_cpo_df_reasons_explicit, but should
                #   have been removed above
                assert(counts_col not in cpo_df.columns)
                cpo_df = pd.merge(cpo_df, counts_col_vals, left_index=True, right_index=True)
            cpo_df = MECPODf.combine_cpo_df_reasons_explicit(
                rcpo_df=cpo_df, 
                reasons_to_combine=other_reasons,
                combined_reason_name=output_combine_others_col, 
                is_norm=is_norm, 
                counts_col=counts_col, 
                normalize_by_nSNs_included=normalize_by_nSNs_included, 
                level_0_raw_col = level_0_raw_col, 
                level_0_nrm_col = level_0_nrm_col      
            )
            if is_norm:
                cpo_df = cpo_df.drop(columns=[counts_col])
            # NOTE: If cpo_df has MultiIndex columns, output_combine_others_col will have been correctly converted
            #       to a tuple in MECPODf.combine_cpo_df_reasons_explicit.  Therefore, to select it below,
            #       it must be converted to tuple
            if cpo_df.columns.nlevels==2:
                assert(isinstance(output_combine_others_col, str))
                combine_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(
                    df=cpo_df, 
                    col=output_combine_others_col
                )
                assert(len(combine_col_idx)==1)
                combine_col_idx = combine_col_idx[0]
                output_combine_others_col = cpo_df.columns[combine_col_idx]
            cpo_df = cpo_df[reasons_to_include+[output_combine_others_col]]
        else:
            cpo_df = cpo_df[reasons_to_include]
        #-------------------------
        return cpo_df

    # TODO: SHOULD be able to use get_reasons_subset_from_cpo_df in get_top_reasons_subset_from_cpo_df
    #       However, I don't really trust either all that much at this point, both need cleaned
    # MOVED TO CPXDf
    @staticmethod
    def get_reasons_subset_from_cpo_df(
        cpo_df                       , 
        reasons_to_include           ,
        combine_others               = True, 
        output_combine_others_col    = 'Other Reasons', 
        SNs_tags                     = None, 
        is_norm                      = False, 
        counts_col                   = '_nSNs', 
        normalize_by_nSNs_included   = False, 
        level_0_raw_col              = 'counts', 
        level_0_nrm_col              = 'counts_norm', 
        cols_to_ignore               = None, 
        include_counts_col_in_output = False, 
        verbose                      = True
    ):
        r"""
        Project out the top reasons_to_include reasons from cpo_df.
        The order is taken to be:
          reason_order = cpo_df.mean().sort_values(ascending=False).index.tolist()

        reasons_to_include:
          A list of strings representing the columns to be included.  If cpo_df has MultiIndex columns, the 
            level_0 value will be handled.
            Will still function properly if appropriate tuples/lists are supplied instead of strings
          NOTE: If normalize_by_nSNs_included, then reasons_to_include MUST BE A DICT with keys
                'raw' and 'nrm'.  This is because in general the reasons wanted from raw and nrm will be different

        output_combine_others_col:
          The output column name for the combined others. 
          This should be a string, even if cpo_df has MultiIndex columns (such a case will be handled below!)

        SNs_tags:
          Defaults to MECPODf.std_SNs_cols() + MECPODf.std_nSNs_cols() when SNs_tags is None
        NOTE: SNs_tags should contain strings, not tuples.
              If column is multiindex, the level_0 value will be handled in MECPODf.find_SNs_cols_idxs_from_cpo_df.
              
        cols_to_ignore:
            Any columns to ignore.
            These will be left out of subset operations, but will be included in the output

        --------------------
        THE FOLLOWING ARE ONLY NEEDED WHEN combine_others==True
            is_norm
            counts_col
            normalize_by_nSNs_included
            level_0_raw_col 
            level_0_nrm_col

            counts_col:
              Should be a string.  If cpo_df has MultiIndex columns, the level_0 value will be handled.
              Will still function properly if appropriate tuple/list is supplied instead of string
                e.g., counts_col='_nSNs' will work, as will counts_col=('counts', '_nSNs') (assuming cpo_df columns
                      have level_0 values equal to 'counts')
              EASIEST TO SUPPLY STRING!!!!!
              NOTE: Only really needed if is_norm==True or normalize_by_nSNs_included==True 
                    (i.e., not needed when is_norm==normalize_by_nSNs_included==False)
        """
        #----------------------------------------------------------------------------------------------------
        if normalize_by_nSNs_included:
            cpo_df_raw  = MECPODf.project_level_0_columns_from_rcpo_wide(cpo_df, level_0_raw_col, droplevel=False)
            cpo_df_nrm  = MECPODf.project_level_0_columns_from_rcpo_wide(cpo_df, level_0_nrm_col, droplevel=False)
            assert(isinstance(reasons_to_include, dict))
            reasons_to_include_raw = reasons_to_include['raw']
            reasons_to_include_nrm = reasons_to_include['nrm']
            #-------------------------
            # normalize_by_nSNs_included must be False below or inifite loop!!!!!
            cpo_df_raw = MECPODf.get_reasons_subset_from_cpo_df(
                cpo_df                     = cpo_df_raw, 
                reasons_to_include         = reasons_to_include_raw,
                combine_others             = combine_others, 
                output_combine_others_col  = output_combine_others_col, 
                SNs_tags                   = SNs_tags, 
                is_norm                    = False, 
                counts_col                 = counts_col, 
                normalize_by_nSNs_included = False, 
                level_0_raw_col            = level_0_raw_col, 
                level_0_nrm_col            = level_0_nrm_col
            )
            cpo_df_nrm = MECPODf.get_reasons_subset_from_cpo_df(
                cpo_df                     = cpo_df_nrm, 
                reasons_to_include         = reasons_to_include_nrm,
                combine_others             = combine_others, 
                output_combine_others_col  = output_combine_others_col, 
                SNs_tags                   = SNs_tags, 
                is_norm                    = True, 
                counts_col                 = counts_col, 
                normalize_by_nSNs_included = False, 
                level_0_raw_col            = level_0_raw_col, 
                level_0_nrm_col            = level_0_nrm_col
            )
            #-------------------------
            cpo_df = pd.merge(cpo_df_raw, cpo_df_nrm, left_index=True, right_index=True)
            assert(cpo_df_raw.shape[0]==cpo_df_nrm.shape[0]==cpo_df.shape[0])
            assert(cpo_df_raw.shape[1]+cpo_df_nrm.shape[1]==cpo_df.shape[1])
            #-------------------------
            return cpo_df
        #----------------------------------------------------------------------------------------------------
        # NOTE: reasons_to_include is potentially changed in this function.  This messed up the proper functioning
        #         of the function when normalize_by_nSNs_included as reasons_to_include as first altered to suit
        #         the raw DF, but are then no longer suitable for the nrm DF.  Solution is to copy it first
        reasons_to_include = copy.deepcopy(reasons_to_include)
        #-----
        if cols_to_ignore is not None:
            assert(isinstance(cols_to_ignore, list))
            cols_to_ignore = copy.deepcopy(cols_to_ignore)
        #-------------------------
        # The code below is designed to work with ONLY raw OR normalized, NOT BOTH (both case handled at top)
        assert(cpo_df.columns.nlevels<=2)
        if cpo_df.columns.nlevels==2:
            #-----
            assert(Utilities.is_object_one_of_types(counts_col, [str, tuple, list]))
            if isinstance(counts_col, str):
                #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                assert(cpo_df.columns.get_level_values(0).nunique()==1)
                level_0_val = cpo_df.columns.get_level_values(0).unique().tolist()[0]
                counts_col = (level_0_val, counts_col)
            else:
                assert(len(counts_col)==2)
            #-----
            for i_reason in range(len(reasons_to_include)):
                reason_i = reasons_to_include[i_reason]
                assert(Utilities.is_object_one_of_types(reason_i, [str, tuple, list]))
                if isinstance(reason_i, str):
                    #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                    assert(cpo_df.columns.get_level_values(0).nunique()==1)
                    level_0_val = cpo_df.columns.get_level_values(0).unique().tolist()[0]
                    reasons_to_include[i_reason] = (level_0_val, reason_i)
                else:
                    assert(len(reason_i)==2)
                assert(reasons_to_include[i_reason] in cpo_df.columns)
            #-----
            if cols_to_ignore is not None:
                for i_col in range(len(cols_to_ignore)):
                    col_i = cols_to_ignore[i_col]
                    assert(Utilities.is_object_one_of_types(col_i, [str, tuple, list]))
                    if isinstance(col_i, str):
                        #If only a str given, must infer level 0 value, which is only feasible if only one unique value
                        assert(cpo_df.columns.get_level_values(0).nunique()==1)
                        level_0_val = cpo_df.columns.get_level_values(0).unique().tolist()[0]
                        cols_to_ignore[i_col] = (level_0_val, col_i)
                    else:
                        assert(len(col_i)==2)
                    assert(cols_to_ignore[i_col] in cpo_df.columns)            
        #-------------------------
        if cols_to_ignore is not None:
            cols_to_ignore_df = cpo_df[cols_to_ignore].copy()
            cpo_df = cpo_df.drop(columns=cols_to_ignore)
        
        #-------------------------
        if combine_others and is_norm:
            # In this case, counts_col will be needed for combine
            assert(counts_col in cpo_df.columns)
        #-------------------------
        non_SNs_cols = MECPODf.get_non_SNs_cols_from_cpo_df(
            cpo_df   = cpo_df, 
            SNs_tags = SNs_tags
        )
        # The counts_col is typically not included in non_SNs_cols above.  However, this is not always the case (as, 
        #   e.g., when sent from MECPOCollection.get_reasons_subset_from_merged_cpo_df it will have a random name which
        #   will not be caught by MECPODf.get_non_SNs_cols_from_cpo_df).
        # It is important that it not be included
        if counts_col in non_SNs_cols:
            non_SNs_cols.remove(counts_col)

        # Sometimes, output_combine_others_col can sneak into reasons_to_include (e.g., when a reasons_to_include is taken from a 
        #   data_structure_df, which is routinely done in modeling/predicting)
        # This is almost certainly a mistake, so remove it from reasons_to_include
        #-----
        # If output_combine_others_col found in reasons_to_include, remove it!
        # NOTE: This is done on a CASE-INSENSITIVE basis!
        if output_combine_others_col.lower() in [x.lower() for x in reasons_to_include]:
            if verbose:
                print(f"In MECPODf.get_reasons_subset_from_cpo_df, found \n\toutput_combine_others_col = {output_combine_others_col} in \n\treasons_to_include = {reasons_to_include} ")
            reasons_to_include = [x for x in reasons_to_include if x.lower()!=output_combine_others_col.lower()]

        # Sanity checks
        assert(set(reasons_to_include).difference(set(cpo_df.columns.tolist()))==set())
        assert(set(reasons_to_include).difference(set(non_SNs_cols))==set())

        # Determine the other_reasons
        other_reasons = [x for x in non_SNs_cols if x not in reasons_to_include]
        #-------------------------
        output_cols = reasons_to_include
        #-----
        if combine_others:
            cpo_df = MECPODf.combine_cpo_df_reasons_explicit(
                rcpo_df                    = cpo_df, 
                reasons_to_combine         = other_reasons,
                combined_reason_name       = output_combine_others_col, 
                is_norm                    = is_norm, 
                counts_col                 = counts_col, 
                normalize_by_nSNs_included = normalize_by_nSNs_included, 
                level_0_raw_col            = level_0_raw_col, 
                level_0_nrm_col            = level_0_nrm_col      
            )
            if is_norm and not include_counts_col_in_output:
                cpo_df = cpo_df.drop(columns=[counts_col])
            # NOTE: If cpo_df has MultiIndex columns, output_combine_others_col will have been correctly converted
            #       to a tuple in MECPODf.combine_cpo_df_reasons_explicit.  Therefore, to select it below,
            #       it must be converted to tuple
            if cpo_df.columns.nlevels==2:
                assert(isinstance(output_combine_others_col, str))
                combine_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(
                    df=cpo_df, 
                    col=output_combine_others_col
                )
                assert(len(combine_col_idx)==1)
                combine_col_idx = combine_col_idx[0]
                output_combine_others_col = cpo_df.columns[combine_col_idx]
            output_cols.append(output_combine_others_col)
        #-----
        if include_counts_col_in_output:
            output_cols.append(counts_col)
        #-----
        cpo_df = cpo_df[output_cols]
        #-------------------------
        if cols_to_ignore is not None:
            assert(cpo_df.shape[0]==cols_to_ignore_df.shape[0])
            assert(cpo_df.index.equals(cols_to_ignore_df.index))
            cpo_df = pd.concat([cpo_df, cols_to_ignore_df], axis=1)
        #-------------------------
        return cpo_df