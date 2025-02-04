#!/usr/bin/env python

r"""
Holds OpItem class.  See OpItem.OpItem for more information.
"""

__author__ = "Jesse Buxton"
__email__ = "jbuxton@aep.com"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import pandas as pd
import numpy as np
import numpy_financial as npf
from natsort import natsorted, ns
import copy
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
import Utilities_df
#--------------------------------------------------


class OpItem:
    def __init__(
        self, 
        val_1, 
        deltas, 
        reimb_pcts, 
        n_vals, 
        main_cash_drctn, 
        name, 
        year_0             = 1, 
        expansion_strategy ='last'
    ):
        r"""
        NOTE: If main_cash_drctn=='plus' then reimb_pcts is set to zero.  I am not sure of a case where
                a cash in-flow will be covered by tenants!
        """
        #--------------------------------------------------
        assert(main_cash_drctn in ['plus', 'minus'])
        assert(expansion_strategy=='last' or expansion_strategy=='mean')
        #-------------------------
        self.years              = list(range(year_0, year_0+n_vals))
        self.val_1              = val_1
        self.name               = name 
        self.main_cash_drctn    = main_cash_drctn
        self.expansion_strategy = expansion_strategy
        #-----
        assert(isinstance(n_vals, int))
        self.n_vals             = n_vals
        #-------------------------
        assert(isinstance(deltas, float) or isinstance(deltas, int) or isinstance(deltas, list))
        self.deltas = OpItem.get_deltas_to_set(
            deltas              = deltas, 
            n_vals              = self.n_vals, 
            expansion_strategy  = self.expansion_strategy
        )
        #----------
        assert(isinstance(reimb_pcts, float) or isinstance(reimb_pcts, int) or isinstance(reimb_pcts, list))
        if self.main_cash_drctn=='plus':
            reimb_pcts = 0
        #-----
        self.reimb_pcts = OpItem.get_reimb_pcts_to_set(
            reimb_pcts          = reimb_pcts, 
            n_vals              = self.n_vals, 
            expansion_strategy  = self.expansion_strategy
        )        
        #-------------------------
        assert(len(self.deltas)==len(self.reimb_pcts)==self.n_vals)
        #--------------------------------------------------
        self.vals = []
        self.vals.append(self.val_1)
        for i in list(range(1,self.n_vals)):
            val_i = self.vals[-1]*(1+self.deltas[i])
            self.vals.append(val_i)
        #-------------------------
        self.reimbs = list(np.multiply(self.vals, self.reimb_pcts))
        
    @staticmethod
    def get_deltas_to_set(
        deltas, 
        n_vals, 
        expansion_strategy='last'
    ):
        #--------------------------------------------------
        assert(expansion_strategy=='last' or expansion_strategy=='mean')
        assert(isinstance(n_vals, int))
        assert(isinstance(deltas, float) or isinstance(deltas, int) or isinstance(deltas, list))
        #-------------------------
        if isinstance(deltas, float) or isinstance(deltas, int):
            return_deltas = [deltas]*n_vals
        else:
            if len(deltas)==n_vals:
                return_deltas = deltas
            elif len(deltas)<n_vals:
                n_needed = n_vals-len(deltas)
                if expansion_strategy=='last':
                    val_ipj = deltas[-1]
                elif expansion_strategy=='mean':
                    val_ipj = np.mean(deltas)
                else:
                    assert(0)
                return_deltas = deltas + [val_ipj]*n_needed
            elif len(deltas)>n_vals:
                return_deltas = deltas[n_vals]
            else:
                assert(0)
        #-------------------------
        return return_deltas
    
    @staticmethod
    def expand_srs(
        srs_to_ex, 
        n_vals, 
        expansion_strategy='last'
    ):
        #--------------------------------------------------
        # Intended only for a certain type of series!
        assert(np.max(srs_to_ex.index.tolist())==srs_to_ex.index[-1])
        assert(len(srs_to_ex)==srs_to_ex.index.nunique())
        #--------------------------------------------------
        assert(expansion_strategy=='last' or expansion_strategy=='mean')
        assert(isinstance(n_vals, int))
        assert(isinstance(srs_to_ex, pd.Series))
        #-------------------------
        if len(srs_to_ex)==n_vals:
            return srs_to_ex
        elif len(srs_to_ex)<n_vals:
            n_needed = n_vals-len(srs_to_ex)
            if expansion_strategy=='last':
                val_ipj = srs_to_ex.iloc[-1]
            elif expansion_strategy=='mean':
                val_ipj = srs_to_ex.mean()
            else:
                assert(0)
            addtln_srs = pd.Series(
                data  = [val_ipj]*n_needed, 
                index = range(srs_to_ex.index[-1]+1, srs_to_ex.index[-1]+1+n_needed)
            )
            return pd.concat([srs_to_ex, addtln_srs])
        elif len(srs_to_ex)>n_vals:
            return srs_to_ex.iloc[:n_vals]
        else:
            assert(0)
        
        
        if isinstance(deltas, float) or isinstance(deltas, int):
            return_deltas = [deltas]*n_vals
        else:
            if len(deltas)==n_vals:
                return_deltas = deltas
            elif len(deltas)<n_vals:
                n_needed = n_vals-len(deltas)
                if expansion_strategy=='last':
                    val_ipj = deltas[-1]
                elif expansion_strategy=='mean':
                    val_ipj = np.mean(deltas)
                else:
                    assert(0)
                return_deltas = deltas + [val_ipj]*n_needed
            elif len(deltas)>n_vals:
                return_deltas = deltas[n_vals]
            else:
                assert(0)
        #-------------------------
        return return_deltas

    @staticmethod
    def get_reimb_pcts_to_set(
        reimb_pcts, 
        n_vals, 
        expansion_strategy='last'
    ):
        #--------------------------------------------------
        return OpItem.get_deltas_to_set(
            deltas             = reimb_pcts, 
            n_vals             = n_vals, 
            expansion_strategy = expansion_strategy
        )
    
    def get_deltas(self):
        return copy.deepcopy(self.deltas)
    
    def get_reimb_pcts(self):
        return copy.deepcopy(self.reimb_pcts)
    
    
    def get_table(
        self
    ):
        r"""
        """
        #-------------------------
        return_table = pd.DataFrame(
            data=dict(
                delta     = self.deltas, 
                val       = self.vals, 
                reimb_pct = self.reimb_pcts, 
                reimb     = self.reimbs
            ), 
            index = self.years
        ).T
        return return_table
    
    @property
    def df(self):
        r"""
        """
        #-------------------------
        return_df = self.get_table()
        assert(return_df.index.nunique()==return_df.shape[0])
        return return_df
    
    @property
    def deltas_srs(self):
        return self.df.loc['delta']

    @property
    def vals_srs(self):
        return self.df.loc['val']

    @property
    def reimb_pcts_srs(self):
        return self.df.loc['reimb_pct']

    @property
    def reimbs_srs(self):
        return self.df.loc['reimb']