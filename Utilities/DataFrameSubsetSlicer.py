#!/usr/bin/env python

import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype

import Utilities
import Utilities_df
from Utilities_df import DFConstructType
import Utilities_dt

class DataFrameSubsetSingleSlicer:
    r"""
    Very simple class to help project out desired slices of a pd.DataFrame
    The class was created specifically to handle grabbing unique major and minor causes
    in the outage analyses.
    
    Even if using only a single cut, DataFrameSubsetSlicer class can be used
    instead of this base class.
    
    Attributes:
        column:
          - The column in the pd.DataFrame on which to cut
        value:
          - The cut value (or values).
          - If value is a list/tuple, .isin(value) is used by default
            Otherwise, ==value is used by default.
        comparison_operator:
          - Operator to use in comparison for cut.
          - As mentioned above, if value is a list/tuple, .isin(value) is used by default
            Otherwise, ==value is used by default.
          - Currently supported values:
            -- value is singular:
                 - '=='
                 - '!='
            -- value is list/tuple:
                 - 'isin'
                 - 'notin'
    """
    def __init__(
        self, 
        column, 
        value, 
        comparison_operator=None
    ):
        self.column=column
        self.value=value
        self.comparison_operator=comparison_operator
        if self.comparison_operator is None:
            self.set_standard_comparison_operator()
        self.check_comparison_operator()
        
    def set_standard_comparison_operator(self):
        if Utilities.is_object_one_of_types(self.value, [list, tuple]):
            self.comparison_operator = 'isin'
        else:
            self.comparison_operator = '=='

    def check_comparison_operator(self):
        accptbl_operators = [
            '==', 
            '!=', 
            '>', 
            '<', 
            '>=', 
            '<=', 
            'isin', 
            'notin'
        ]
        assert(self.comparison_operator in accptbl_operators)
            
    def as_dict(self):
        r"""
        Package column, value, and comparison_operator in dict.
        Main purpose I so I can save slicers in JSON files when modelling
        
        NOTE: Object of type Index is not JSON serializable
              So, if value is Index, it is converted to list
        """
        #-------------------------
        output_value = self.value
        if isinstance(output_value, pd.Index):
            output_value = output_value.tolist()
        #-----
        output_dict = {
            'column'              : self.column, 
            'value'               : output_value, 
            'comparison_operator' : self.comparison_operator
        }
        return output_dict
        
            
    def get_slicing_booleans(self, df):
        #-------------------------
        # Now possible to slice using index values instead of columns!
        # Methods are slightly different there, and handled via DataFrameSubsetSingleSlicer.get_slicing_booleans_for_idx_slicer
        # See Utilities_df.get_idfs_loc for acceptable inputs when using index instead of column
        if self.column not in df.columns:
            slicing_booleans = self.get_slicing_booleans_for_idx_slicer(df=df)
            return slicing_booleans
        #-------------------------
        possible_comparison_operators = ['==', '!=', '>', '<', '>=', '<=', 'isin', 'notin']
        assert(self.comparison_operator in possible_comparison_operators)
        #-------------------------
        if self.comparison_operator=='==':
            return df[self.column]==self.value
        elif self.comparison_operator=='!=':
            return df[self.column]!=self.value
        elif self.comparison_operator=='>':
            return df[self.column]>self.value
        elif self.comparison_operator=='<':
            return df[self.column]<self.value
        elif self.comparison_operator=='>=':
            return df[self.column]>=self.value
        elif self.comparison_operator=='<=':
            return df[self.column]<=self.value
        elif self.comparison_operator=='isin':
            return df[self.column].isin(self.value)
        elif self.comparison_operator=='notin':
            return ~df[self.column].isin(self.value)
        else:
            assert(0)
            
    def get_slicing_booleans_for_idx_slicer(
        self, 
        df
    ):
        r"""
        Fun complications when slicing using index (or level of an index) instead of a column!
        NOTE: The operations here should restore df to its original form.
              I chose to not simply call df=df.copy() at the beginning of the function in a hopes of saving memory
                for cases when df is large (but this is always an option is things don't look right)
               
        """
        #--------------------------------------------------
        idfr_loc = Utilities_df.get_idfr_loc(
            df   = df, 
            idfr = self.column
        )
        #-------------------------
        # Should only find ourselves here if we are slicing with the index
        # In which case, idfr_loc[1] will be True
        assert(idfr_loc[1]==True)
        #--------------------------------------------------
        # Since column is in the index (as weird as that may sound...), .reset_index() will need to be called.
        # After slicing operation is done, I want to return DF to its original form, with original names.
        # For this to work, all index levels must have names.
        # Those without names will be given names, but the original values will be recorded so they can be recovered later
        #-------------------------
        # Record original names, to be recovered at end
        org_idx_names = df.index.names
        #-------------------------
        # Name any un-named
        random_nm = Utilities.generate_random_string()
        new_idx_names = [name_i if name_i else f'{random_nm}_{i}' for i,name_i in enumerate(df.index.names)]
        df.index.names = new_idx_names
        idfr_lvl_name = df.index.names[idfr_loc[0]]
        #-------------------------
        # Call reset_index, so now the desired values to slice on are contained in a column instead of the index.
        # idfr_lvl_name is the aforementioned column.
        df = df.reset_index()
        #-------------------------
        # Make sure only a single column is returned by df[idfr_lvl_name]
        # This may sound trivial, but if df has MutliIndex columns, then such an operation could
        #   actually return multiple columns!
        assert(isinstance(df[idfr_lvl_name], pd.Series))
        #-------------------------
        # Now, one can simply use the standard DFSingleSlicer.get_slicing_booleans method to slice
        # Note, however, that self.column must first be changed (save org in case needed later)
        org_column = self.column
        self.column = idfr_lvl_name
        slicing_booleans = self.get_slicing_booleans(df=df)
        #-------------------------
        # However, to be useful, the slicing boolean needs to share the same index as the original df!
        #-----
        # Make sure df and slicing_booleans both have the same index, which should be integers from 0 to n-1
        assert(all(slicing_booleans.index==df.index))
        assert(all(df.index==list(range(len(df)))))
        #-----
        # Restore df
        df = df.set_index(new_idx_names)
        assert(len(df.index.names)==len(org_idx_names))
        df.index.names = org_idx_names
        #-----
        # I suppose restore self.column as well
        self.column = org_column
        #-----
        # Set index of slicing_booleans equal to that of df
        slicing_booleans.index = df.index
        #-------------------------
        return slicing_booleans
            
            
    def set_simple_column_value(self, df, column, value):
        r"""
        Use the slicer to set a column value for all entries in slicing_booleans
        """
        #-------------------------
        slicing_booleans = self.get_slicing_booleans(df)
        df.loc[slicing_booleans, column]=value
        return df
            
    def perform_slicing(self, df):
        slicing_booleans = self.get_slicing_booleans(df)
        return df[slicing_booleans]
    
    
class DataFrameSubsetSlicer:
    r"""
    Very simple class to help project out desired slices of a pd.DataFrame
    The class was created specifically to handle grabbing unique major and minor causes
    in the outage analyses.
    
    This class can be used to create a single subset pd.DataFrame
    
    As of now, multiple slicers can only be joined via &.  If expanded functionality is desired,
      the best route may be to subclass from SQLAids methods.
    
    Attributes:
        single_slicers:
          A collection of DataFrameSubsetSingleSlicer objects to be used in slicing the DF
        
        name:
          Used as identifier.  Probably, when multiple slices are returned simultaneously,
            they will be stored in a dict object and identified via names 
            
        apply_not:
            Applies ~ (or, not) to the overall boolean series to be returned.
            Purpose is because sometimes it's easiest to implement a negation by finding all finding all memebers to be negated
              and negating at the very end.
            e.g., to exclude outages from df with MJR_CAUSE_CD='DL' and MNR_CAUSE_CD='OL', it is easiest to call
                df[~((df['MJR_CAUSE_CD']=='DL') & (df['MNR_CAUSE_CD']=='OL'))]
              Note, the following would not give the desired result:
                df[(df['MJR_CAUSE_CD']!='DL') & (df['MNR_CAUSE_CD']!='OL')]
              But this would (although much more confusing than using ~ above):
                df[(df['MJR_CAUSE_CD']!='DL') | (df['MNR_CAUSE_CD']!='OL')]
    """
    def __init__(
        self, 
        single_slicers=None, 
        name=None, 
        apply_not=False, 
        join_single_slicers='and'
    ):
        r"""
        single_slicers can be:
          i.   a collection (list/tuple) of DataFrameSubsetSingleSlicer objects
          ii.  a collection (list/tuple) of dict objects representing kwargs for building
               DataFrameSubsetSingleSlicer objects
          iii. a mixture of (i) and (ii)
          
        join_single_slicers:
            Must be equal to 'and' (default) or 'or'
        """
        self.single_slicers=[]
        self.name=name
        self.apply_not=apply_not
        assert(join_single_slicers in ['and', 'or'])
        self.join_single_slicers = join_single_slicers
        if single_slicers is not None:
            for slicer in single_slicers:
                self.add_single_slicer(slicer)
        
    def add_single_slicer(self, input_arg):
        r"""
        input_arg can be:
          i.   a DataFrameSubsetSingleSlicer object
          ii.  a dict object representing kwargs for building a DataFrameSubsetSingleSlicer object
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(input_arg, [dict, DataFrameSubsetSingleSlicer]))
        if isinstance(input_arg, dict):
            input_arg = DataFrameSubsetSingleSlicer(**input_arg)
        assert(isinstance(input_arg, DataFrameSubsetSingleSlicer))
        self.single_slicers.append(input_arg)
        
    def add_slicers(self, slicers):
        r"""
        slicers is designed to be a list of dict/DataFrameSubsetSingleSlicer objects to be input into add_single_slicer.
        However, slicers can also be a single dict/DataFrameSubsetSingleSlicer object
        """
        #-------------------------
        assert(Utilities.is_object_one_of_types(slicers, [dict, DataFrameSubsetSlicer.DataFrameSubsetSingleSlicer, list]))
        if not isinstance(slicers, list):
            slicers=[slicers]
        #-------------------------
        for slicer in slicers:
            self.add_single_slicer(slicer)
            
    def as_dict(self):
        r"""
        Package slicers in dict.
        Main purpose I so I can save slicers in JSON files when modelling
        """
        #-------------------------
        output_dict = {
            'name'                : self.name, 
            'apply_not'           : self.apply_not, 
            'join_single_slicers' : self.join_single_slicers
        }
        #-------------------------
        for i, slicer_i in enumerate(self.single_slicers):
            assert(f'slicer_{i}' not in output_dict)
            output_dict[f'slicer_{i}'] = slicer_i.as_dict()
        #-------------------------
        return output_dict
        
    def get_slicing_booleans(self, df):
        r"""
        Grab a pd.Series containing boolean values to use for slicing.
        
        Added functionality for null slicer, in which case all values returned are True.
          So, essentially no slicing occurs.
        """
        #-------------------------
        # Null slicer case
        if self.single_slicers is None or len(self.single_slicers)==0:
            return pd.Series(data=True, index=df.index)
        #-------------------------
        for i,slicer in enumerate(self.single_slicers):
            if i==0:
                slicing_booleans = slicer.get_slicing_booleans(df)
            else:
                if self.join_single_slicers=='and':
                    slicing_booleans = slicing_booleans & slicer.get_slicing_booleans(df)
                elif self.join_single_slicers=='or':
                    slicing_booleans = slicing_booleans | slicer.get_slicing_booleans(df)
                else:
                    assert(0)
        if self.apply_not:
            return ~slicing_booleans
        else:
            return slicing_booleans
            
    def set_simple_column_value(self, df, column, value):
        r"""
        Use the slicer to set a column value for all entries in slicing_booleans
        """
        #-------------------------
        slicing_booleans = self.get_slicing_booleans(df)
        df.loc[slicing_booleans, column]=value
        return df
        
    def perform_slicing(self, df):
        slicing_booleans = self.get_slicing_booleans(df)
        return df[slicing_booleans]
        
        
    @staticmethod
    def combine_slicers_and_get_slicing_booleans(
        df, 
        slicers, 
        join_slicers='and', 
        apply_not=False
    ):
        r"""
        Combine multiple DataFrameSubsetSlicer objects and get resultant boolean series.
        This is just like DataFrameSubsetSlicer.get_slicing_booleans, but for combining multiple
          DataFrameSubsetSlicer objects instead of DataFrameSubsetSingleSlicer objects

        slicers should be a list of DataFrameSubsetSlicer objects
        join_slicers:
            Must be equal to 'and' (default) or 'or'
        """
        #-------------------------
        assert(isinstance(slicers, list))
        # Assertion commented out below works if function defined outside of class.
        # Not sure how to make it work when defined inside of class.
        # assert(Utilities.are_all_list_elements_of_type(slicers, DataFrameSubsetSlicer.DataFrameSubsetSlicer))
        assert(join_slicers in ['and', 'or'])
        #-------------------------
        for i,slicer in enumerate(slicers):
            if i==0:
                slicing_booleans = slicer.get_slicing_booleans(df)
            else:
                if join_slicers=='and':
                    slicing_booleans = slicing_booleans & slicer.get_slicing_booleans(df)
                elif join_slicers=='or':
                    slicing_booleans = slicing_booleans | slicer.get_slicing_booleans(df)
                else:
                    assert(0)
        #-------------------------
        if apply_not:
            return ~slicing_booleans
        else:
            return slicing_booleans

    @staticmethod
    def combine_slicers_and_perform_slicing(
        df, 
        slicers, 
        join_slicers='and', 
        apply_not=False
    ):
        r"""
        Combine multiple DataFrameSubsetSlicer objects and perform slicing on df.
        This is just like DataFrameSubsetSlicer.perform_slicing, but for combining multiple
          DataFrameSubsetSlicer objects instead of DataFrameSubsetSingleSlicer objects

        slicers should be a list of DataFrameSubsetSlicer objects
        join_slicers:
            Must be equal to 'and' (default) or 'or'
        """
        #-------------------------
        slicing_srs = DataFrameSubsetSlicer.combine_slicers_and_get_slicing_booleans(
            df=df, 
            slicers=slicers, 
            join_slicers=join_slicers, 
            apply_not=apply_not
        )
        return df[slicing_srs]