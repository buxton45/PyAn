#!/usr/bin/env python

r"""
Holds general items.  See individual classes for more info
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
from OpItem import OpItem
from Depreciators import AnnualDepreciator, DepreciationTable, PropertyDepreciationTable, SimpleAmortizer, SimpleAmortTable
#--------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
class GenItem:
    r"""
    Class for a general cash flow "item"
    This item essentially contains an OpItem, a DepreciationTable item, and a SimpleAmortTable item
    """
    def __init__(
        self, 
        val_1, 
        deltas, 
        reimb_pcts, 
        n_vals, 
        main_cash_drctn, 
        name, 
        year_0, 
        hold_years          = None,         
        depr_life_years     = None, 
        depr_schdl          = None, 
        amort_life_years    = None, 
        amort_schdl         = None, 
        expansion_strategy  = 'last', 
        init_depr_tbl       = True, 
        init_simp_amort_tbl = True
    ):
        r"""
        depr_life_years/amort_life_years:
            If not set and needed, n_vals is used (i.e., the life of the item)
        """
        #-------------------------
        if hold_years is None:
            hold_years = year_0 + n_vals
        #-------------------------
        self.op_item        = OpItem(
            val_1              = val_1, 
            deltas             = deltas, 
            reimb_pcts         = reimb_pcts, 
            n_vals             = n_vals, 
            main_cash_drctn    = main_cash_drctn, 
            name               = name, 
            year_0             = year_0, 
            expansion_strategy ='last'
        )
        #-------------------------
        self.depr_tbl = None
        if init_depr_tbl:
            if depr_life_years is None:
                print(f"WARNING: depr_life_years not set, using n_vals={n_vals}")
                depr_life_years = n_vals
            ads_list = [
                AnnualDepreciator(
                    acq_cost     = val_i, 
                    life_n_years = depr_life_years, 
                    salv_val     = 0, 
                    year_0       = year_i, 
                    schedule     = depr_schdl, 
                    name         = None
                ) 
                for year_i, val_i in (self.op_item.df.loc['val']).items()
            ]
            self.depr_tbl = DepreciationTable(
                ads_list   = ads_list, 
                hold_years = hold_years, 
                name       = name
            )
        #-------------------------
        self.simp_amort_tbl = None
        if init_simp_amort_tbl:
            if amort_life_years is None:
                print(f"WARNING: amort_life_years not set, using n_vals={n_vals}")
                amort_life_years = n_vals
            sas_list = [
                SimpleAmortizer(
                    val_0        = val_i, 
                    life_n_years = amort_life_years, 
                    year_0       = year_i, 
                    schedule     = amort_schdl, 
                    name         = None
                ) 
                for year_i, val_i in (self.op_item.df.loc['val']).items()
            ]
            self.simp_amort_tbl = SimpleAmortTable(
                sas_list = sas_list, 
                name     = name
            )
            
            
    @property
    def vals_srs(self):
        return self.op_item.vals_srs
    @property
    def vals(self):
        return self.vals_srs.values.tolist()
    def get_val_i(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        vals_srs = self.vals_srs
        assert(year_i in vals_srs.index)
        return vals_srs.loc[year_i]


    @property
    def reimbs_srs(self):
        return self.op_item.reimbs_srs
    @property
    def reimbs(self):
        return self.reimbs_srs.values.tolist()
    def get_reimb_i(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        reimbs_srs = self.reimbs_srs
        assert(year_i in reimbs_srs.index)
        return reimbs_srs.loc[year_i]
    
    
    @property
    def depr_tbl_df(self):
        return self.depr_tbl.depr_tbl_df
    def get_depr_tot_i(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        depr_tbl_df = self.depr_tbl_df
        assert(year_i in depr_tbl_df.columns)
        assert('Total' in depr_tbl_df.index)
        #-----
        depr_tot_i = depr_tbl_df.loc['Total', year_i]
        return depr_tot_i
    
    
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
class CommonAreaMaintenance(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas,
        reimb_pcts, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None,         
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = False
        )
        
 
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
class PropertyTax(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas,
        reimb_pcts, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None,         
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = False
        )
        
        
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
class Insurance(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas,
        reimb_pcts, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None,         
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = False
        )
        
 
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------  
class REUtilities(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas,
        reimb_pcts, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None,         
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = False
        )
        
        
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
class TenantImprovements(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None, 
        depr_life_years     = None, 
        depr_schdl          = None, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = 0, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = depr_life_years, 
            depr_schdl          = depr_schdl, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = True, 
            init_simp_amort_tbl = False
        )

#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
class CapEx(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None, 
        depr_life_years     = None, 
        depr_schdl          = None, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = 0, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = depr_life_years, 
            depr_schdl          = depr_schdl, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = True, 
            init_simp_amort_tbl = False
        )
        
        
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
class LeaseCommissions(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None, 
        amort_life_years    = None, 
        amort_schdl         = None, 
        expansion_strategy  = 'last', 
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = 0, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = amort_life_years, 
            amort_schdl         = amort_schdl, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = True            
        )
        
    def add_unamort_row_to_simp_amort_tbl(
        self
    ):
        r"""
        """
        #-------------------------
        amort_df        = self.simp_amort_tbl.depr_tbl_df.copy()
        cum_depr_tbl_df = self.simp_amort_tbl.cum_depr_tbl_df
        #-----
        og_amort_n_rows = amort_df.shape[0]
        #-------------------------
        vals_srs = self.vals_srs
        cost_row_dict = {}
        assert(np.max(vals_srs.index)==vals_srs.index[-1])
        for year_i in amort_df.columns.tolist():
            if year_i > vals_srs.index[-1]:
                val_i = 0
            else:
                val_i = self.get_val_i(year_i)
            assert(year_i not in cost_row_dict.items())
            cost_row_dict[year_i] = val_i
        cost_row = pd.Series(cost_row_dict)
        #-------------------------
        assert(cost_row.shape[0]==cost_row.index.nunique())
        if cost_row.shape[0] != amort_df.shape[1]:
            assert(cost_row.shape[0] <= amort_df.shape[1])
            cost_row = pd.concat([cost_row, pd.Series({i:0 for i in range(cost_row.shape[0]+1, amort_df.shape[1]+1)})])
        assert(cost_row.shape[0]==amort_df.shape[1])
        #-----
        amort_df = pd.concat([amort_df, cost_row.to_frame(name='Cumulative Cost').T], axis=0)
        #-------------------------
        for year_i in amort_df.columns.tolist():
            if year_i==1:
                continue
            amort_df.loc['Cumulative Cost', year_i] = amort_df.loc['Cumulative Cost', year_i] + amort_df.loc['Cumulative Cost', year_i-1] 
        #-------------------------
        assert(cum_depr_tbl_df.columns.tolist()==amort_df.columns.tolist())
        amort_df = pd.concat([amort_df, cum_depr_tbl_df.loc['Total'].to_frame(name='Cumulative Amortization').T])
        amort_df.loc['Still Unamortized at Year-end'] = amort_df.loc['Cumulative Cost'] - amort_df.loc['Cumulative Amortization']
        #-------------------------
        assert(amort_df.iloc[:og_amort_n_rows].equals(self.simp_amort_tbl.depr_tbl_df))
        self.simp_amort_tbl.depr_tbl_df = amort_df

        
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------         
class LoanPoints(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None, 
        amort_life_years    = None, 
        amort_schdl         = None
    ):
        r"""
        """
        #-------------------------
        super().__init__(
            val_1               = val_1, 
            deltas              = [0, -1, 0], 
            reimb_pcts          = 0, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'minus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = amort_life_years, 
            amort_schdl         = amort_schdl, 
            expansion_strategy  = 'last', 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = True            
        )
        
    def add_unamort_row_to_simp_amort_tbl(
        self
    ):
        r"""
        """
        #-------------------------
        amort_df        = self.simp_amort_tbl.depr_tbl_df.copy()
        cum_depr_tbl_df = self.simp_amort_tbl.cum_depr_tbl_df
        #-----
        og_amort_n_rows = amort_df.shape[0]
        #-------------------------
        vals_srs = self.vals_srs
        cost_row_dict = {}
        assert(np.max(vals_srs.index)==vals_srs.index[-1])
        for year_i in amort_df.columns.tolist():
            if year_i > vals_srs.index[-1]:
                val_i = 0
            else:
                val_i = self.get_val_i(year_i)
            assert(year_i not in cost_row_dict.items())
            cost_row_dict[year_i] = val_i
        cost_row = pd.Series(cost_row_dict)
        #-------------------------
        assert(cost_row.shape[0]==cost_row.index.nunique())
        if cost_row.shape[0] != amort_df.shape[1]:
            assert(cost_row.shape[0] <= amort_df.shape[1])
            cost_row = pd.concat([cost_row, pd.Series({i:0 for i in range(cost_row.shape[0]+1, amort_df.shape[1]+1)})])
        assert(cost_row.shape[0]==amort_df.shape[1])
        #-----
        amort_df = pd.concat([amort_df, cost_row.to_frame(name='Cumulative Cost').T], axis=0)
        #-------------------------
        for year_i in amort_df.columns.tolist():
            if year_i==1:
                continue
            amort_df.loc['Cumulative Cost', year_i] = amort_df.loc['Cumulative Cost', year_i] + amort_df.loc['Cumulative Cost', year_i-1] 
        #-------------------------
        assert(cum_depr_tbl_df.columns.tolist()==amort_df.columns.tolist())
        amort_df = pd.concat([amort_df, cum_depr_tbl_df.loc['Total'].to_frame(name='Cumulative Amortization').T])
        amort_df.loc['Still Unamortized at Year-end'] = amort_df.loc['Cumulative Cost'] - amort_df.loc['Cumulative Amortization']
        #-------------------------
        assert(amort_df.iloc[:og_amort_n_rows].equals(self.simp_amort_tbl.depr_tbl_df))
        self.simp_amort_tbl.depr_tbl_df = amort_df
        
        
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
class BaseRentalRevenues(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1, 
        deltas, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None,  
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = 0, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'plus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = False
        )

#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------         
class PercentRent(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1, 
        deltas, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None,  
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = 0, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'plus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = False
        )

#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
class AncillaryIncome(GenItem):
    r"""
    """
    def __init__(
        self, 
        val_1_per_sqf, 
        sqf, 
        deltas, 
        n_vals, 
        name, 
        year_0, 
        hold_years          = None,  
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.val_1_per_sqf = val_1_per_sqf
        self.sqf           = sqf
        #-----
        val_1 = self.val_1_per_sqf*self.sqf
        #-----
        super().__init__(
            val_1               = val_1, 
            deltas              = deltas, 
            reimb_pcts          = 0, 
            n_vals              = n_vals, 
            main_cash_drctn     = 'plus', 
            name                = name, 
            year_0              = year_0, 
            hold_years          = hold_years, 
            depr_life_years     = None, 
            depr_schdl          = None, 
            amort_life_years    = None, 
            amort_schdl         = None, 
            expansion_strategy  = expansion_strategy, 
            init_depr_tbl       = False, 
            init_simp_amort_tbl = False
        )
        
        
        
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------ 
#-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****-----*****    
class SuspendedTaxLoss:
    def __init__(
        self, 
        hold_years, 
        year_0 = 1
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(hold_years, int))
        assert(isinstance(year_0, int))
        self.hold_years = hold_years
        self.year_0     = year_0
        #-------------------------
        self.__susp_tax_df = pd.DataFrame(
            columns=['year', 'taxable_income', 'book_beg', 'loss', 'susp_losses_appld', 'book_end'], 
            dtype=float, 
        )
        
    def check_susp_tax_df(self):
        r"""
        """
        #-------------------------
        if self.__susp_tax_df.shape[0] < self.hold_years:
            return False
        #-------------------------
        return True
        
    @property
    def susp_tax_df(self):
        if not self.check_susp_tax_df():
            print("Cannot return susp_tax_df because did not pass check_susp_tax_df!")
            assert(0)
        return self.__susp_tax_df.copy()
    @property
    def susp_tax_df_incmplt(self):
        return self.__susp_tax_df.copy()
    @property
    def suspended_tax_loss_df(self):
        if not self.check_susp_tax_df():
            print("Cannot return susp_tax_df because did not pass check_susp_tax_df!")
            assert(0)
        return self.susp_tax_df.T
        
    @staticmethod
    def add_tax_for_year_i_to_susp_tax_df(
        taxable_income_i, 
        susp_tax_df, 
        hold_years, 
        year_0 = 1, 
        return_susp_losses_appld= False
    ):
        r"""
        """
        #--------------------------------------------------
        cols_expctd = ['year', 'taxable_income', 'book_beg', 'loss', 'susp_losses_appld', 'book_end']
        assert(set(susp_tax_df.columns.tolist()).symmetric_difference(set(cols_expctd))==set())
        if susp_tax_df.shape[0]==0:
            book_beg_i = 0
            year_i     = year_0
        else:
            book_beg_i = susp_tax_df.iloc[-1]['book_end']
            year_i     = susp_tax_df.iloc[-1]['year']+1
        #-----
        assert(book_beg_i<=0)
        #--------------------------------------------------
        if taxable_income_i>0:
            ann_loss_i = 0
        else:
            ann_loss_i = taxable_income_i
        #--------------------------------------------------
        applctn_of_susp_loss_i = -1
        if year_i < hold_years:
            # We we have taxable income and a suspended loss balance, use it!
            if taxable_income_i>0 and book_beg_i<0:
                # Use as much of book_beg_i as possible, up to taxable_income_i
                applctn_of_susp_loss_i = np.min([-book_beg_i, taxable_income_i])
            else:
                # Either we don't have taxable income (taxable_income_i<1) or we don't have a suspended loss balance.
                # In either case, there is no susp. loss to apply!
                applctn_of_susp_loss_i = 0
        elif year_i == hold_years:
            # If it's the hold_years, we are selling property and should use up all of susp. loss balance
            applctn_of_susp_loss_i = -book_beg_i
        else:
            applctn_of_susp_loss_i = 0
        #-----
        assert(applctn_of_susp_loss_i>=0)
        #--------------------------------------------------
        book_end_i = book_beg_i + ann_loss_i + applctn_of_susp_loss_i
        #-----
        entry_i = pd.Series(dict(
            year              = year_i, 
            taxable_income    = taxable_income_i,
            book_beg          = book_beg_i, 
            loss              = ann_loss_i, 
            susp_losses_appld = applctn_of_susp_loss_i, 
            book_end          = book_end_i
        )).to_frame(name=susp_tax_df.shape[0]+1).T
        #-----
        susp_tax_df = pd.concat([susp_tax_df, entry_i])
        #--------------------------------------------------
        if return_susp_losses_appld:
            return susp_tax_df, susp_tax_df.iloc[-1]['susp_losses_appld']
        else:
            return susp_tax_df
    
    def add_tax_for_year_i(
        self, 
        taxable_income_i, 
        return_susp_losses_appld = False
    ):
        r"""
        """
        #--------------------------------------------------
        self.__susp_tax_df, susp_losses_appld_i = SuspendedTaxLoss.add_tax_for_year_i_to_susp_tax_df(
            taxable_income_i         = taxable_income_i, 
            susp_tax_df              = self.__susp_tax_df, 
            hold_years               = self.hold_years, 
            year_0                   = self.year_0, 
            return_susp_losses_appld = True
        )
        if return_susp_losses_appld:
            return susp_losses_appld_i
        

#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------         
class ReplacementReserve:
    def __init__(
        self, 
        capex_rsrv_at_closing, 
        hold_years, 
        year_0 = 1
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(hold_years, int))
        assert(isinstance(year_0, int))
        self.hold_years = hold_years
        self.year_0     = year_0
        #-------------------------
        self.__repl_rsrv_df = pd.Series(dict(
            year                = year_0-1, 
            book_beg            = 0, 
            contribution        = capex_rsrv_at_closing, 
            balance_after_contr = 0, 
            draw_for_capex      = 0, 
            book_end            = 0
        )).to_frame(name=0).T
        
    def check_repl_rsrv_df(self):
        r"""
        """
        #-------------------------
        if self.__repl_rsrv_df.shape[0] < self.hold_years:
            return False
        #-------------------------
        return True
        
    @property
    def repl_rsrv_df(self):
        if not self.check_repl_rsrv_df():
            print("Cannot return repl_rsrv_df because did not pass check_repl_rsrv_df!")
            assert(0)
        return self.__repl_rsrv_df.copy()
    @property
    def repl_rsrv_df_incmplt(self):
        return self.__repl_rsrv_df.copy()
    @property
    def replacement_reserve_df(self):
        if not self.check_repl_rsrv_df():
            print("Cannot return repl_rsrv_df because did not pass check_repl_rsrv_df!")
            assert(0)
        return self.repl_rsrv_df.T
        
    @staticmethod
    def add_contr_for_year_i_to_repl_rsrv_df(
        contr_i, 
        capex_i, 
        repl_rsrv_df, 
        hold_years, 
        year_0 = 1, 
        return_draw_for_capex= False
    ):
        r"""
        """
        #--------------------------------------------------
        cols_expctd = ['year', 'book_beg', 'contribution', 'balance_after_contr', 'draw_for_capex', 'book_end']
        assert(set(repl_rsrv_df.columns.tolist()).symmetric_difference(set(cols_expctd))==set())
        assert(repl_rsrv_df.shape[0]>0)
        if repl_rsrv_df.shape[0]==1:
            book_beg_i = repl_rsrv_df.iloc[-1]['contribution']
            year_i     = year_0
        else:
            book_beg_i = repl_rsrv_df.iloc[-1]['book_end']
            year_i     = repl_rsrv_df.iloc[-1]['year']+1
        #-----
        assert(book_beg_i>=0)
        #--------------------------------------------------
        balance_after_contr_i = book_beg_i + contr_i
        draw_for_capex_i = -np.min([capex_i, balance_after_contr_i])
        book_end_i = balance_after_contr_i + draw_for_capex_i
        #--------------------------------------------------
        entry_i = pd.Series(dict(
            year                = year_i, 
            book_beg            = book_beg_i,
            contribution        = contr_i, 
            balance_after_contr = balance_after_contr_i, 
            draw_for_capex      = draw_for_capex_i, 
            book_end            = book_end_i
        )).to_frame(name=repl_rsrv_df.index[-1]+1).T
        #-----
        repl_rsrv_df = pd.concat([repl_rsrv_df, entry_i])
        #--------------------------------------------------
        repl_rsrv_df[repl_rsrv_df['year']>hold_years]=0
        #--------------------------------------------------
        if return_draw_for_capex:
            return repl_rsrv_df, repl_rsrv_df.iloc[-1]['draw_for_capex']
        else:
            return repl_rsrv_df
        
    def add_contr_for_year_i(
        self, 
        contr_i, 
        capex_i, 
        return_draw_for_capex = False
    ):
        r"""
        """
        #--------------------------------------------------
        self.__repl_rsrv_df, draw_for_capex_i = ReplacementReserve.add_contr_for_year_i_to_repl_rsrv_df(
            contr_i         = contr_i, 
            capex_i         = capex_i, 
            repl_rsrv_df              = self.__repl_rsrv_df, 
            hold_years               = self.hold_years, 
            year_0                   = self.year_0, 
            return_draw_for_capex = True
        )
        if return_draw_for_capex:
            return draw_for_capex_i