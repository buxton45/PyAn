#!/usr/bin/env python

r"""
Holds REValuator class.  See REValuator.REValuator for more information.
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
#-----
from GenItems import GenItem
from GenItems import CommonAreaMaintenance, PropertyTax, Insurance, REUtilities
from GenItems import TenantImprovements, CapEx, LeaseCommissions, LoanPoints, BaseRentalRevenues, PercentRent, AncillaryIncome, SuspendedTaxLoss, ReplacementReserve
#--------------------------------------------------


class REValuator:
    def __init__(
        self, 
        leasable_sqf, 
        base_rent_per_sqf, 
        hold_years, 
        purchase_price, 
        capex_rsrv_0_pct, 
        loan_size_ltc, 
        loan_amor_term_years, 
        int_rate_ann, 
        loan_points_pct, 
        loan_points_amor_term_years, 
        tax_rate, 
        cap_rate_exit, 
        sell_costs_pct_gsp, 
        acc_depr_tax_rate, 
        cap_gains_tax_rate, 
        base_rental_rev_growth, 
        vac_pct_gr,
        opex_growth, 
        credit_loss_pct_gi, 
        mngmnt_fee_pct_reimbs, 
        repl_rsrv_pct, 
        #----------
        pct_land          = 0.20, 
        pct_strctr        = 0.50,  
        n_years_strctr    = 39, 
        x_year_items_dict = {
            7 : 0.2, 
            3 : 0.1
        }, 
        #----------
        expansion_strategy_rent_rev = 'last', 
        proj_years        = None
    ):
        r"""
        """
        #-------------------------
        self.leasable_sqf                = leasable_sqf
        self.base_rent_per_sqf           = base_rent_per_sqf
        self.hold_years                  = hold_years
        self.purchase_price              = purchase_price
        self.capex_rsrv_0_pct            = capex_rsrv_0_pct
        self.loan_size_ltc               = loan_size_ltc
        self.loan_amor_term_years        = loan_amor_term_years
        self.int_rate_ann                = int_rate_ann
        self.loan_points_pct             = loan_points_pct
        self.loan_points_amor_term_years = loan_points_amor_term_years
        self.tax_rate                    = tax_rate
        self.cap_rate_exit               = cap_rate_exit
        self.sell_costs_pct_gsp          = sell_costs_pct_gsp
        self.acc_depr_tax_rate           = acc_depr_tax_rate
        self.cap_gains_tax_rate          = cap_gains_tax_rate
        #--------------------------------------------------
        if proj_years is None:
            self.proj_years = self.hold_years+1
        else:
            self.proj_years = proj_years
        assert(self.proj_years >= self.hold_years)
        #--------------------------------------------------
        self.base_rental_rev_growth      = base_rental_rev_growth
        self.expansion_strategy_rent_rev = expansion_strategy_rent_rev
        self.gross_pot_rntl_rev          = self.base_rent_per_sqf*self.leasable_sqf
        #-----
        self.base_rental_revs            = None
        self.set_BaseRentalRevenues()
        #--------------------------------------------------
        self.vac_pct_gr_srs              = OpItem.expand_srs(
            srs_to_ex          = pd.Series(vac_pct_gr, index=range(1,len(vac_pct_gr)+1)), 
            n_vals             = self.proj_years, 
            expansion_strategy = 'last'
        )
        #-----
        self.opex_growth                 = opex_growth
        #-----
        self.credit_loss_pct_gi_srs      = OpItem.expand_srs(
            srs_to_ex          = pd.Series(credit_loss_pct_gi, index=range(1,len(credit_loss_pct_gi)+1)), 
            n_vals             = self.proj_years, 
            expansion_strategy = 'last'
        )
        #-----
        #-----
        self.mngmnt_fee_pct_reimbs_srs   = OpItem.expand_srs(
            srs_to_ex          = pd.Series(mngmnt_fee_pct_reimbs, index=range(1,len(mngmnt_fee_pct_reimbs)+1)), 
            n_vals             = self.proj_years, 
            expansion_strategy = 'last'
        )
        #-----
        self.repl_rsrv_pct               = repl_rsrv_pct
        #-------------------------
        self.pct_land                    = pct_land
        self.pct_strctr                  = pct_strctr
        self.n_years_strctr              = n_years_strctr
        self.x_year_items_dict           = x_year_items_dict 
        #-------------------------        
        self.capex_rsrv_0    = self.capex_rsrv_0_pct*self.purchase_price
        self.loan_size       = self.loan_size_ltc*(self.purchase_price+self.capex_rsrv_0)
        self.equity_0        = self.purchase_price+self.capex_rsrv_0-self.loan_size
        self.loan_points_tot = self.loan_points_pct*self.loan_size
        #-------------------------
        self.loan_amort_term_months = 12*self.loan_amor_term_years
        self.int_rate = self.int_rate_ann/12
        #-----
        self.dbt_serv_pmt     = npf.pmt(self.int_rate, self.loan_amort_term_months, -self.loan_size, fv=0, when='end')
        self.dbt_serv_pmt_ann = 12*self.dbt_serv_pmt
        #--------------------------------------------------
        # Build mort_amort_df
        self.mort_amort_df = None
        self.build_and_set_mort_amort_df()
        #--------------------------------------------------
        # Build prop_depr_tbl
        self.prop_depr_tbl = PropertyDepreciationTable(
            purchase_price    = self.purchase_price, 
            pct_land          = self.pct_land, 
            pct_strctr        = self.pct_strctr, 
            hold_years        = self.hold_years, 
            x_year_items_dict = self.x_year_items_dict, 
            n_years_strctr    = self.n_years_strctr, 
        )
        #--------------------------------------------------
        self.susp_tax_loss = SuspendedTaxLoss(
            hold_years = self.hold_years, 
            year_0     = 1
        )
        #-----
        self.repl_rsrv_tbl = ReplacementReserve(
            capex_rsrv_at_closing = self.capex_rsrv_0, 
            hold_years            = self.hold_years, 
            year_0                =  1
        )
        
        #--------------------------------------------------
        self.cam              = None
        self.prop_tax         = None
        self.insurance        = None
        self.utilities        = None
        self.percent_rent     = None
        self.anc_inc          = None
        self.tnnt_imprvs      = None
        self.capex            = None
        self.lease_comm       = None
        self.loan_points      = None
        
        
        #--------------------------------------------------
        self.cash_flows_cmpnnts_by_year = dict()
        self.cash_flows_raw_by_year     = dict()
        self.cash_flows_df              = None

        
    @staticmethod
    def build_monthly_amortization_table(
        loan_size, 
        amort_term_years, 
        interest_rate_annual, 
        start_date             = pd.to_datetime('2024-01-01'), 
        addtnl_principal       = 0, 
        return_payment_summary = False
    ):
        r"""
        """
        #--------------------------------------------------
        assert(addtnl_principal is None or addtnl_principal>=0)
        #-------------------------
        amort_term_months     = 12*amort_term_years
        interest_rate_monthly = interest_rate_annual/12
        #-------------------------
        pmt_i = npf.pmt(interest_rate_monthly, amort_term_months, -loan_size, fv=0, when='end')
        debt_service_annually = 12*pmt_i
        #-------------------------
        pmt_dates = pd.date_range(start_date, periods=amort_term_years*12, freq='MS')
        pmt_dates.name = 'pmt_date'
        #--------------------------------------------------
        amort_df = pd.DataFrame(
            index=pmt_dates, 
            columns=['pmt', 'principal', 'interest', 'addtnl_principal', 'balance_curr'], 
            dtype=float, 
        )
        amort_df = amort_df.reset_index()
        amort_df.index = list(range(1, amort_df.shape[0]+1))
        amort_df.index.name = 'pd'
        #-------------------------
        amort_df['pmt']       = pmt_i
        amort_df['principal'] = npf.ppmt(interest_rate_monthly, amort_df.index, amort_term_months, -loan_size, fv=0, when='end')
        amort_df['interest']  = npf.ipmt(interest_rate_monthly, amort_df.index, amort_term_months, -loan_size, fv=0, when='end')
        #-----
        amort_df['addtnl_principal'] = addtnl_principal
        #-------------------------
        # Store cumulative and ensure it never gets larger than the original principal
        amort_df['cumulative_principal'] = (amort_df['principal']+amort_df['addtnl_principal']).cumsum()
        amort_df['cumulative_principal'] = amort_df['cumulative_principal'].clip(upper=loan_size)
        #-------------------------
        # Calculate balance_curr
        amort_df['balance_curr'] = loan_size - amort_df['cumulative_principal']
        #-------------------------
        # Determine the last payment date
        try:
            last_payment = amort_df.query("balance_curr <= 0")["balance_curr"].idxmax(axis=1, skipna=True)
        except ValueError:
            last_payment = amort_df.last_valid_index()
        last_payment_date = "{:%m-%d-%Y}".format(amort_df.loc[last_payment, "pmt_date"])
        #-------------------------
        # Remove any extra payment periods (e.g., if addtnl principal contributed)
        amort_df = amort_df.loc[0:last_payment].copy()

        # Calculate the principal for the last row
        amort_df.loc[last_payment, "principal"] = amort_df.loc[last_payment-1, 'balance_curr']

        # Calculate the total payment for the last row
        amort_df.loc[last_payment, "pmt"] = amort_df.loc[last_payment, ["principal", "interest"]].sum()

        # Zero out the additional principal
        amort_df.loc[last_payment, "addtnl_principal"] = 0
        #-------------------------
        # Get the payment info into a DataFrame in column order
        payment_info = amort_df[["pmt", "principal", "addtnl_principal", "interest"]].sum().to_frame().T
        # Format the Date DataFrame
        payment_details = pd.DataFrame.from_dict(dict([
            ('payoff_date', [last_payment_date]),
            ('Interest Rate', [interest_rate_annual]),
            ('Number of years', [amort_term_years])
        ]))
        # Add a column showing how much we pay each period.
        # Combine addl principal with principal for total payment
        payment_details["Period_Payment"] = round(pmt_i, 2) + addtnl_principal
        #-----
        payment_summary = pd.concat([payment_details, payment_info], axis=1)
        #-------------------------
        if return_payment_summary:
            return amort_df, payment_summary
        else:
            return amort_df
        
    def build_and_set_mort_amort_df(
        self
    ):
        r"""
        """
        #--------------------------------------------------
        # Build mort_amort_df
        mort_amort_df = REValuator.build_monthly_amortization_table(
            loan_size              = self.loan_size, 
            amort_term_years       = self.loan_amor_term_years, 
            interest_rate_annual   = self.int_rate_ann, 
            start_date             = pd.to_datetime('2024-01-01'), 
            addtnl_principal       = 0, 
            return_payment_summary = False
        )
        #-------------------------
        mort_amort_df['month'] = mort_amort_df['pmt_date'].dt.month 
        #----------
        mort_amort_df['year']  = mort_amort_df['pmt_date'].dt.year 
        #-----
        curr_year_0 = mort_amort_df['year'].min()
        year_0 = 1 # Want year_0 to be 1
        mort_amort_df['year'] = mort_amort_df['year'] - (curr_year_0-year_0)
        #----------
        mort_amort_df = mort_amort_df.drop(columns=['pmt_date'])
        mort_amort_df = Utilities_df.move_cols_to_front(
            df           = mort_amort_df, 
            cols_to_move = ['month', 'year']
        )
        #-------------------------
        self.mort_amort_df = mort_amort_df
        
        
    def get_mort_amort_year_end_principal(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        mort_amort_df = self.mort_amort_df
        assert(year_i in mort_amort_df['year'].unique().tolist())
        #-----
        mort_amort_df_i = mort_amort_df[mort_amort_df['year']==year_i]
        assert(
            mort_amort_df_i.shape[0]==12 and 
            set(mort_amort_df_i['month'].unique().tolist()).symmetric_difference(set(list(range(1, 13))))==set()
        )
        #-----
        year_i_prncpl = mort_amort_df_i['principal'].sum()
        #-------------------------
        return year_i_prncpl
    
    
    def get_mort_amort_year_end_balance(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        mort_amort_df = self.mort_amort_df
        assert(year_i in mort_amort_df['year'].unique().tolist())
        #-----
        mort_amort_df_i = mort_amort_df[mort_amort_df['year']==year_i]
        assert(
            mort_amort_df_i.shape[0]==12 and 
            set(mort_amort_df_i['month'].unique().tolist()).symmetric_difference(set(list(range(1, 13))))==set()
        )
        #-------------------------
        year_end_bal_i = mort_amort_df_i.loc[mort_amort_df_i['month']==12,'balance_curr']
        #-----
        assert(year_end_bal_i.shape[0]==1)
        year_end_bal_i = year_end_bal_i.values[0]
        #-------------------------
        return year_end_bal_i
        
        
        
    def set_CAM(
        self, 
        val_1_per_sqf, 
        deltas, 
        reimb_pcts, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.cam = CommonAreaMaintenance(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = self.proj_years, 
            name                = 'Common Area Maintenance', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            expansion_strategy  = expansion_strategy   
        )

    def set_PropertyTax(
        self, 
        val_1_per_sqf, 
        deltas, 
        reimb_pcts, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.prop_tax = PropertyTax(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = self.proj_years, 
            name                = 'Property Tax', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            expansion_strategy  = expansion_strategy  
        )

    def set_Insurance(
        self, 
        val_1_per_sqf, 
        deltas, 
        reimb_pcts, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.insurance = Insurance(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = self.proj_years, 
            name                = 'Insurance', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            expansion_strategy  = expansion_strategy  
        )

    def set_Utilities(
        self, 
        val_1_per_sqf, 
        deltas, 
        reimb_pcts, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.utilities = REUtilities(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas, 
            reimb_pcts          = reimb_pcts, 
            n_vals              = self.proj_years, 
            name                = 'Utilities', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            expansion_strategy  = expansion_strategy  
        )
        
    def set_BaseRentalRevenues(
        self
    ):
        r"""
        """
        #-------------------------
        self.base_rental_revs = BaseRentalRevenues(
            val_1               = self.gross_pot_rntl_rev, 
            deltas              = self.base_rental_rev_growth,  
            n_vals              = self.proj_years, 
            name                = 'Base Rental Revenues', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            expansion_strategy  = self.expansion_strategy_rent_rev  
        )
        
    def set_PercentRent(
        self, 
        val_1, 
        deltas, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.percent_rent = PercentRent(
            val_1               = val_1, 
            deltas              = deltas,  
            n_vals              = self.proj_years, 
            name                = 'Percent Rent', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            expansion_strategy  = expansion_strategy  
        )

    def set_AncillaryIncome(
        self, 
        val_1_per_sqf, 
        deltas, 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.anc_inc = AncillaryIncome(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas,  
            n_vals              = self.proj_years, 
            name                = 'Ancillary Income', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            expansion_strategy  = expansion_strategy  
        )


    def set_TenantImprovements(
        self, 
        val_1_per_sqf, 
        deltas, 
        depr_life_years     = 7, 
        depr_schdl          = (0.5, 'year1'), 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.tnnt_imprvs = TenantImprovements(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas, 
            n_vals              = self.proj_years, 
            name                = 'Tenant Improvements', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            depr_life_years     = depr_life_years, 
            depr_schdl          = depr_schdl, 
            expansion_strategy  = expansion_strategy
        )

    def set_CapEx(
        self, 
        val_1_per_sqf, 
        deltas, 
        depr_life_years     = 7, 
        depr_schdl          = 'sl', 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.capex = CapEx(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas, 
            n_vals              = self.proj_years, 
            name                = 'Cap Ex', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            depr_life_years     = depr_life_years, 
            depr_schdl          = depr_schdl, 
            expansion_strategy  = expansion_strategy
        )

    def set_LeaseCommissions(
        self, 
        val_1_per_sqf, 
        deltas, 
        amort_life_years    = 7, 
        amort_schdl         = 'sl', 
        expansion_strategy  = 'last'
    ):
        r"""
        """
        #-------------------------
        self.lease_comm = LeaseCommissions(
            val_1_per_sqf       = val_1_per_sqf, 
            sqf                 = self.leasable_sqf, 
            deltas              = deltas, 
            n_vals              = self.proj_years, 
            name                = 'Lease Commissions', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            amort_life_years    = amort_life_years, 
            amort_schdl         = amort_schdl, 
            expansion_strategy  = expansion_strategy
        )
        self.lease_comm.add_unamort_row_to_simp_amort_tbl()

    def set_LoanPoints(
        self, 
        amort_life_years    = 7, 
        amort_schdl         = 'sl'
    ):
        r"""
        """
        #-------------------------
        self.loan_points = LoanPoints(
            val_1               = self.loan_points_tot, 
            n_vals              = self.proj_years, 
            name                = 'Loan Points', 
            year_0              = 1, 
            hold_years          = self.hold_years, 
            amort_life_years    = amort_life_years, 
            amort_schdl         = amort_schdl
        )
        self.loan_points.add_unamort_row_to_simp_amort_tbl()
        
    def add_tax_for_year_i(
        self, 
        year_i, 
        taxable_income_i, 
        return_susp_losses_appld=True
    ):
        r"""
        """
        #-------------------------
        if year_i==1:
            assert(self.susp_tax_loss.susp_tax_df_incmplt.shape[0]==0)
        else:
            assert(Utilities.are_approx_equal(year_i-1, self.susp_tax_df_incmplt.iloc[-1]['year']))
        #-------------------------
        susp_losses_appld_i = self.susp_tax_loss.add_tax_for_year_i(
            taxable_income_i         = taxable_income_i, 
            return_susp_losses_appld = True
        )
        if return_susp_losses_appld:
            return susp_losses_appld_i
        
    @staticmethod
    def sale_income_tax_accounting(
        hold_year_p1_adj_noi, 
        cap_rate_exit, 
        sell_costs_pct_gsp, 
        purchase_price, 
        net_ti_and_capex, 
        accum_depr, 
        mort_bal_outstanding, 
        acc_depr_tax_rate  = 0.25, 
        cap_gains_tax_rate = 0.15, 
        hold_years         = None, 
        return_full        = False
    ):
        r"""
        """
        #-------------------------
        gsp        = hold_year_p1_adj_noi/cap_rate_exit
        sell_costs = sell_costs_pct_gsp*gsp
        #-----
        net_sales_price = gsp-sell_costs
        #-------------------------
        adj_cost_basis = purchase_price + net_ti_and_capex - accum_depr
        #-------------------------
        gain_on_sale = net_sales_price - adj_cost_basis
        #----------
        # Components of cap. gain
        depr_recap = accum_depr
        prop_appr  = gain_on_sale-depr_recap
        #----------
        # Cap gains tax on sale
        inc_tax_liability = acc_depr_tax_rate*depr_recap + cap_gains_tax_rate*prop_appr
        #-------------------------
        # Net sales proceeds
        net_sales_proceeds = net_sales_price - inc_tax_liability - mort_bal_outstanding
        #-------------------------
        nsp_cal_srs = pd.Series(dict(
            gsp                  = gsp, 
            sell_costs           = sell_costs, 
            net_sales_price      = net_sales_price, 
            inc_tax_liability    = inc_tax_liability, 
            mort_bal_outstanding = mort_bal_outstanding, 
            net_sales_proceeds   = net_sales_proceeds
        ))
        #-------------------------
        if not return_full:
            return nsp_cal_srs
        #-------------------------
        nsp_cal_srs_full = pd.Series({
            f"Year {hold_years} Adjusted NOI"                                    :  np.round(hold_year_p1_adj_noi, 2), 
            f"Cap Rate"                                                          :  np.round(100*cap_rate_exit, 2), 
            f"Gross Sales Price = ( Year {hold_years} Adjusted NOI / Cap Rate )" :  np.round(gsp, 2),  
            f"Less Selling Costs ({np.round(100*sell_costs_pct_gsp, 2)})"        : -np.round(sell_costs, 2), 
            "Net Sales Price"                                                    :  np.round(net_sales_price, 2), 
            "Less: Adjusted Cost Basis" : "", 
            "Acquistion Cost"                                                    :  np.round(purchase_price, 2), 
            "Plus TIs and CapEx"                                                 :  np.round(net_ti_and_capex, 2), 
            "Less: Acc. Depr"                                                    : -np.round(accum_depr, 2), 
            "Adj. Cost Basis"                                                    :  np.round(adj_cost_basis, 2), 
            "":"", 
            "Gain-on-Sale (Capital Gain)"                                        :  np.round(gain_on_sale, 2), 
            " ":"", 
            "Componenets of Cap. Gain:" : "", 
            "Depr. Recapture"                                                    :  np.round(depr_recap, 2), 
            "Prop. Appreciation"                                                 :  np.round(prop_appr, 2), 
            "   "                                                                :  np.round(gain_on_sale, 2), 
            "Cap. Gains Tax on Sale": "", 
            f"On Acc. Depr. ({np.round(100*acc_depr_tax_rate, 2)})"              :  np.round(acc_depr_tax_rate*depr_recap, 2), 
            f"On Prop. Appr. ({np.round(100*cap_gains_tax_rate, 2)})"            :  np.round(cap_gains_tax_rate*prop_appr, 2), 
            "Total Sale Income Tax Liability"                                    :  np.round(inc_tax_liability, 2), 
            "    ":"", 
            "Net Sales Proceeds Calculation" : "", 
            "Gross Sales Price"                                                  :  np.round(gsp, 2), 
            "Less Selling Costs"                                                 : -np.round(sell_costs, 2), 
            "Net Sales Price"                                                    :  np.round(net_sales_price, 2), 
            "Less Sale Income Tax Liability"                                     : -np.round(inc_tax_liability, 2), 
            "Less Outstanding Mortgage Balance"                                  : -np.round(mort_bal_outstanding, 2), 
            "Net Sales Proceeds"                                                 :  np.round(net_sales_proceeds, 2)
        })
        return nsp_cal_srs_full
    
    
    @staticmethod
    def assemble_cash_flows_i_df_beg(
        base_rental_rev_i, 
        cam_reimb_i, 
        prop_tax_reimb_i, 
        insurance_reimb_i, 
        utilities_reimb_i, 
        reimb_tot_i, 
        gross_rev_i, 
        vacancies_i, 
        net_base_rental_rev_i, 
        percentage_rent_i, 
        rent_income_tot_i, 
        anc_inc_i, 
        gross_income_i, 
        credit_loss_i, 
        egi_i
    ):
        r"""
        """
        #-------------------------
        # Sanity checks
        assert(reimb_tot_i           == cam_reimb_i + prop_tax_reimb_i + insurance_reimb_i + utilities_reimb_i)
        assert(gross_rev_i           == base_rental_rev_i + reimb_tot_i)
        assert(net_base_rental_rev_i == gross_rev_i - vacancies_i)
        assert(rent_income_tot_i     == net_base_rental_rev_i + percentage_rent_i)
        assert(gross_income_i        == rent_income_tot_i + anc_inc_i)
        assert(egi_i                 == gross_income_i - credit_loss_i)
        #-------------------------
        cash_flows_i_df = pd.DataFrame.from_dict(
            orient='index', 
            columns=['desc', 'val'], 
            data={
                1:  ["Base Rental Revenues",         base_rental_rev_i], 
                2:  ["Expense Reimbursements",       ""], 
                3:  ["Plus: CAM Billings",           cam_reimb_i], 
                4:  ["Plus: Property Tax Billings",  prop_tax_reimb_i], 
                5:  ["Plus: Insurance Billings",     insurance_reimb_i], 
                6:  ["Plus: Utilities Billings",     utilities_reimb_i], 
                7:  ["Total Reimbursements",         reimb_tot_i], 
                8:  ["Gross Revenues",               gross_rev_i], 
                9:  ["Less:  Vacancies",             -vacancies_i], 
                10: ["Net Base Rental Revenue",      net_base_rental_rev_i], 
                11: ["Percentage Rents",             percentage_rent_i], 
                12: ["Total Rental Income",          rent_income_tot_i], 
                13: ["Plus: Ancillary Income",       anc_inc_i], 
                14: ["Gross Income",                 gross_income_i], 
                15: ["Less: Credit Loss",            -credit_loss_i], 
                16: ["Effective Gross Income (EGI)", egi_i]
            }
        )
        #-------------------------
        return cash_flows_i_df

    def build_cash_flows_i_df_opexs(
        self, 
        idx_0, 
        year_i
    ):
        r"""
        """
        #-------------------------
        orient='index'
        columns=['desc', 'val']
        #-----
        data = dict()
        data[idx_0]   = ["Less: Operating Expenses", ""]
        #-------------------------
        # gross_rev_i is needed for repl_rsrv_i calculation
        gross_rev_i = self.base_rental_revs.get_val_i(year_i) + (
            self.cam.get_reimb_i(year_i) + 
            self.prop_tax.get_reimb_i(year_i) + 
            self.insurance.get_reimb_i(year_i) + 
            self.utilities.get_reimb_i(year_i)
        )
        #-------------------------
        # Possibly reimbursable: cam, prop_tax, insurance, utilities
        # Not reimbursable:      mngmnt_fee_i, repl_rsrv_i
        # ----- Possibly reimbursable -----
        cam       = self.cam.get_val_i(year_i)
        prop_tax  = self.prop_tax.get_val_i(year_i)
        insurance = self.insurance.get_val_i(year_i)
        utilities = self.utilities.get_val_i(year_i)
        # ----- Not reimbursable -----
        mngmnt_fee_i = (cam + prop_tax)*self.mngmnt_fee_pct_reimbs_srs.loc[year_i]
        if year_i<=self.hold_years:
            repl_rsrv_i = self.repl_rsrv_pct*gross_rev_i
        else:
            repl_rsrv_i = 0
        #----------
        # Total expenses
        expenses_tot_i = cam + prop_tax + insurance + utilities + mngmnt_fee_i + repl_rsrv_i

        #--------------------------------------------------
        # TODO: Until more elegant approach built, putting all in both reimb and nonreimb right now
        reimb_data     = {}
        non_reimb_data = {}
        #-------------------------
        if self.cam.get_reimb_i(year_i)>0:
            reimb_data[len(reimb_data)]         = ["Common Area Maintanence", -cam]
            non_reimb_data[len(non_reimb_data)] = ["Common Area Maintanence", 0]
        else:
            assert(self.cam.get_reimb_i(year_i)==0)
            reimb_data[len(reimb_data)]         = ["Common Area Maintanence", 0]
            non_reimb_data[len(non_reimb_data)] = ["Common Area Maintanence", -cam]

        #-------------------------
        if self.prop_tax.get_reimb_i(year_i)>0:
            reimb_data[len(reimb_data)]         = ["Property Taxes", -prop_tax]
            non_reimb_data[len(non_reimb_data)] = ["Property Taxes", 0]
        else:
            assert(self.prop_tax.get_reimb_i(year_i)==0)
            reimb_data[len(reimb_data)]         = ["Property Taxes", 0]
            non_reimb_data[len(non_reimb_data)] = ["Property Taxes", -prop_tax]

        #-------------------------
        if self.insurance.get_reimb_i(year_i)>0:
            reimb_data[len(reimb_data)]         = ["Insurance", -insurance]
            non_reimb_data[len(non_reimb_data)] = ["Insurance", 0]
        else:
            assert(self.insurance.get_reimb_i(year_i)==0)
            reimb_data[len(reimb_data)]         = ["Insurance", 0]
            non_reimb_data[len(non_reimb_data)] = ["Insurance", -insurance]

        #-------------------------
        if self.utilities.get_reimb_i(year_i)>0:
            reimb_data[len(reimb_data)]         = ["Utilities", -utilities]
            non_reimb_data[len(non_reimb_data)] = ["Utilities", 0]
        else:
            assert(self.utilities.get_reimb_i(year_i)==0)
            reimb_data[len(reimb_data)]         = ["Utilities", 0]
            non_reimb_data[len(non_reimb_data)] = ["Utilities", -utilities]
        #--------------------------------------------------
        reimb_data     = {(k+idx_0+2):v                   for k,v in reimb_data.items()}
        non_reimb_data = {(k+idx_0+2+len(reimb_data)+1):v for k,v in non_reimb_data.items()} # extra +1 for "Non-Reimbursable Expenses" title row
        #-------------------------
        data[idx_0+1] = ["Reimbursable Expenses", ""]
        #-----
        if len(reimb_data)>0:
            assert(np.max(list(data.keys()))+1 == np.min(list(reimb_data.keys())))
            data = data | reimb_data
        #-------------------------
        data[np.max(list(data.keys()))+1] = ["Non-Reimbursable Expenses", ""]
        #-----
        if len(non_reimb_data)>0:
            assert(np.max(list(data.keys()))+1 == np.min(list(non_reimb_data.keys())))
            data = data | non_reimb_data

        #--------------------------------------------------
        idx_j = np.max(list(data.keys()))+1
        #-----
        data[idx_j+0] = ["Management",                -mngmnt_fee_i]
        data[idx_j+1] = ["Repleacement Reserce (RR)", -repl_rsrv_i]
        data[idx_j+2] = ["Total Expenses",            -expenses_tot_i]

        #--------------------------------------------------
        cash_flows_i_df_opexs = pd.DataFrame.from_dict(
            orient  = orient, 
            columns = columns, 
            data    = data
        )
        return cash_flows_i_df_opexs
    
    
    @staticmethod
    def assemble_cash_flows_i_df_end(
        idx_0, 
        noi_i, 
        tnnt_imprvs_i, 
        lease_comm_i, 
        capex_i, 
        draw_for_capex_i, 
        adjusted_noi_i, 
        loan_points_i, 
        debt_srvc_pmt_i, 
        lev_cash_b4_tax_i, 
        prop_depr_i, 
        tnnt_imprvs_depr_i, 
        capex_depr_i, 
        lease_comm_amort_i, 
        loan_points_amort_i, 
        cash_xfer_to_rr_i, 
        principal_amort_i, 
        taxable_income_i, 
        susp_losses_appld_i, 
        net_taxable_income_i, 
        tax_liability_i, 
        rfnd_of_rr_acct_bal, 
        cash_after_tax, 

    ):
        r"""
        """
        #-------------------------
        # Sanity checks
        assert(adjusted_noi_i       == noi_i - tnnt_imprvs_i - lease_comm_i - capex_i - draw_for_capex_i)
        assert(lev_cash_b4_tax_i    == adjusted_noi_i - loan_points_i - debt_srvc_pmt_i)
        assert(net_taxable_income_i == taxable_income_i - susp_losses_appld_i)
        assert(cash_after_tax       == lev_cash_b4_tax_i - tax_liability_i + rfnd_of_rr_acct_bal)

        #-------------------------
        cash_flows_i_df = pd.DataFrame.from_dict(
            orient='index', 
            columns=['desc', 'val'], 
            data={
                idx_0:    ["Net Operating Income (NOI)",              noi_i], 
                idx_0+1:  ["Less: TI",                               -tnnt_imprvs_i], 
                idx_0+2:  ["Less: Leasing Commissions",              -lease_comm_i], 
                idx_0+3:  ["Less: Cap Ex",                           -capex_i], 
                idx_0+4:  ["Adjusted Net Operating Income",           adjusted_noi_i], 
                idx_0+5:  ["Less: Loan Points",                      -loan_points_i], 
                idx_0+6:  ["Less: Debt Service Payment",             -debt_srvc_pmt_i], 
                idx_0+7:  ["Before-Tax Levered Cash Flow",            lev_cash_b4_tax_i], 
                idx_0+8:  ["Less: Depreciation (Purchase Price)",    -prop_depr_i], 
                idx_0+9:  ["Less: Depreciation (TIs)",               -tnnt_imprvs_depr_i], 
                idx_0+10: ["Less: Depreciation (CapEx)",             -capex_depr_i], 
                idx_0+11: ["Less: Leasing Commissions Amortization", -lease_comm_amort_i], 
                idx_0+12: ["Less: Loan Points Amortization",         -loan_points_amort_i], 
                idx_0+13: ["Plus: Cash Transfer to RR account",       cash_xfer_to_rr_i], 
                idx_0+14: ["Plus: TIs",                               tnnt_imprvs_i], 
                idx_0+15: ["Plus: Leasing Commissions",               lease_comm_i], 
                idx_0+16: ["Plus: Principal Amortization",            principal_amort_i], 
                idx_0+17: ["Taxable Income",                          taxable_income_i], 
                idx_0+18: ["Less: Application of Suspended Losses",  -susp_losses_appld_i], 
                idx_0+19: ["Net Taxable Income",                      net_taxable_income_i], 
                idx_0+20: ["Less: Tax Liability",                    -tax_liability_i], 
                idx_0+21: ["Plus: Refund at sale of RR acct. bal.",   rfnd_of_rr_acct_bal], 
                idx_0+22: ["Plus: Depreciation (Purchase Price)",     prop_depr_i], 
                idx_0+23: ["Plus: Depreciation (TIs)",                tnnt_imprvs_depr_i], 
                idx_0+24: ["Plus: Depreciation (CapEx)",              capex_depr_i], 
                idx_0+25: ["Plus: Leasing Commissions Amortization",  lease_comm_amort_i], 
                idx_0+26: ["Plus: Loan Points Amortization",          loan_points_amort_i], 
                idx_0+27: ["Less: TIs",                              -tnnt_imprvs_i], 
                idx_0+28: ["Less: Leasing Commissions",              -lease_comm_i], 
                idx_0+29: ["Less: Cap Ex",                           -cash_xfer_to_rr_i], # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                idx_0+30: ["Less: Principal Amortization",           -principal_amort_i], 
                idx_0+31: ["After-Tax Cash Flow",                     cash_after_tax]
            }
        )
        #-------------------------
        return cash_flows_i_df
    
    def build_cash_flow_year_i(
        self, 
        year_i, 
        base_rental_rev_i, 
        cam_reimb_i, 
        prop_tax_reimb_i, 
        insurance_reimb_i, 
        utilities_reimb_i, 
        reimb_tot_i, 
        gross_rev_i, 
        vacancies_i, 
        net_base_rental_rev_i, 
        percentage_rent_i, 
        rent_income_tot_i, 
        anc_inc_i, 
        gross_income_i, 
        credit_loss_i, 
        egi_i, 
        noi_i, 
        tnnt_imprvs_i, 
        lease_comm_i, 
        capex_i, 
        draw_for_capex_i, 
        adjusted_noi_i, 
        loan_points_i, 
        debt_srvc_pmt_i, 
        lev_cash_b4_tax_i, 
        prop_depr_i, 
        tnnt_imprvs_depr_i, 
        capex_depr_i, 
        lease_comm_amort_i, 
        loan_points_amort_i, 
        cash_xfer_to_rr_i, 
        principal_amort_i, 
        taxable_income_i, 
        susp_losses_appld_i, 
        net_taxable_income_i, 
        tax_liability_i, 
        rfnd_of_rr_acct_bal, 
        cash_after_tax, 
    ):
        r"""
        """
        #-------------------------
        assert(year_i>0 and year_i<=self.proj_years)
        #-------------------------
        cash_flows_i_df_beg = REValuator.assemble_cash_flows_i_df_beg(
            base_rental_rev_i     = base_rental_rev_i, 
            cam_reimb_i           = cam_reimb_i, 
            prop_tax_reimb_i      = prop_tax_reimb_i, 
            insurance_reimb_i     = insurance_reimb_i, 
            utilities_reimb_i     = utilities_reimb_i, 
            reimb_tot_i           = reimb_tot_i, 
            gross_rev_i           = gross_rev_i, 
            vacancies_i           = vacancies_i, 
            net_base_rental_rev_i = net_base_rental_rev_i, 
            percentage_rent_i     = percentage_rent_i, 
            rent_income_tot_i     = rent_income_tot_i, 
            anc_inc_i             = anc_inc_i, 
            gross_income_i        = gross_income_i, 
            credit_loss_i         = credit_loss_i, 
            egi_i                 = egi_i
        )
        #-------------------------
        assert(np.max(cash_flows_i_df_beg.index)==cash_flows_i_df_beg.index[-1])
        cash_flows_i_df_opexs = self.build_cash_flows_i_df_opexs(
            idx_0  = cash_flows_i_df_beg.index[-1]+1, 
            year_i = year_i
        )
        #-------------------------
        assert(np.max(cash_flows_i_df_opexs.index)==cash_flows_i_df_opexs.index[-1])
        cash_flows_i_df_end = REValuator.assemble_cash_flows_i_df_end(
            idx_0                = cash_flows_i_df_opexs.index[-1]+1, 
            noi_i                = noi_i, 
            tnnt_imprvs_i        = tnnt_imprvs_i, 
            lease_comm_i         = lease_comm_i, 
            capex_i              = capex_i, 
            draw_for_capex_i     = draw_for_capex_i, 
            adjusted_noi_i       = adjusted_noi_i, 
            loan_points_i        = loan_points_i, 
            debt_srvc_pmt_i      = debt_srvc_pmt_i, 
            lev_cash_b4_tax_i    = lev_cash_b4_tax_i, 
            prop_depr_i          = prop_depr_i, 
            tnnt_imprvs_depr_i   = tnnt_imprvs_depr_i, 
            capex_depr_i         = capex_depr_i, 
            lease_comm_amort_i   = lease_comm_amort_i, 
            loan_points_amort_i  = loan_points_amort_i, 
            cash_xfer_to_rr_i    = cash_xfer_to_rr_i, 
            principal_amort_i    = principal_amort_i, 
            taxable_income_i     = taxable_income_i, 
            susp_losses_appld_i  = susp_losses_appld_i, 
            net_taxable_income_i = net_taxable_income_i, 
            tax_liability_i      = tax_liability_i, 
            rfnd_of_rr_acct_bal  = rfnd_of_rr_acct_bal, 
            cash_after_tax       = cash_after_tax, 

        )
        #--------------------------------------------------
        assert(year_i not in self.cash_flows_cmpnnts_by_year.keys())
        cash_flows_i = dict(
            cash_flows_i_df_beg   = cash_flows_i_df_beg, 
            cash_flows_i_df_opexs = cash_flows_i_df_opexs, 
            cash_flows_i_df_end   = cash_flows_i_df_end
        )
        self.cash_flows_cmpnnts_by_year[year_i] = cash_flows_i
        #-----
        assert(year_i not in self.cash_flows_raw_by_year.keys())
        self.cash_flows_raw_by_year[year_i] = self.assemble_cash_flows_for_year(year_i = year_i)

    @staticmethod
    def assemble_cash_flows_i_df(
        cash_flows_i_df_beg, 
        cash_flows_i_df_opexs, 
        cash_flows_i_df_end
    ):
        r"""
        """
        #-------------------------
        assert(np.max(cash_flows_i_df_beg.index)   == cash_flows_i_df_beg.index[-1])
        assert(np.max(cash_flows_i_df_opexs.index) == cash_flows_i_df_opexs.index[-1])
        assert(np.max(cash_flows_i_df_end.index)   == cash_flows_i_df_end.index[-1])
        #-----
        assert(cash_flows_i_df_beg.index[-1]   + 1 == cash_flows_i_df_opexs.index[0])
        assert(cash_flows_i_df_opexs.index[-1] + 1 == cash_flows_i_df_end.index[0])
        #-------------------------
        cash_flows_i_df = pd.concat([cash_flows_i_df_beg, cash_flows_i_df_opexs, cash_flows_i_df_end])
        assert(cash_flows_i_df.index[0]==1)
        assert(list(range(1, cash_flows_i_df.shape[0]+1))==cash_flows_i_df.index.tolist())
        #-------------------------
        return cash_flows_i_df
    
    def assemble_cash_flows_for_year(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        assert(year_i in self.cash_flows_cmpnnts_by_year.keys())
        #-----
        cash_flows_i_df_beg   = self.cash_flows_cmpnnts_by_year[year_i]['cash_flows_i_df_beg']
        cash_flows_i_df_opexs = self.cash_flows_cmpnnts_by_year[year_i]['cash_flows_i_df_opexs']
        cash_flows_i_df_end   = self.cash_flows_cmpnnts_by_year[year_i]['cash_flows_i_df_end']
        #-------------------------
        assert(np.max(cash_flows_i_df_beg.index)   == cash_flows_i_df_beg.index[-1])
        assert(np.max(cash_flows_i_df_opexs.index) == cash_flows_i_df_opexs.index[-1])
        assert(np.max(cash_flows_i_df_end.index)   == cash_flows_i_df_end.index[-1])
        #-----
        assert(cash_flows_i_df_beg.index[-1]   + 1 == cash_flows_i_df_opexs.index[0])
        assert(cash_flows_i_df_opexs.index[-1] + 1 == cash_flows_i_df_end.index[0])
        #-------------------------
        cash_flows_i_df = pd.concat([cash_flows_i_df_beg, cash_flows_i_df_opexs, cash_flows_i_df_end])
        assert(cash_flows_i_df.index[0]==1)
        assert(list(range(1, cash_flows_i_df.shape[0]+1))==cash_flows_i_df.index.tolist())
        #-------------------------
        return cash_flows_i_df
    
    def construct_cash_flows_for_year_i(
        self,
        year_i
    ):
        r"""
        """
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        # Base rental revenue
        base_rental_rev_i = self.base_rental_revs.get_val_i(year_i)

        #-------------------------
        # Operating expense reimbursements
        #-----
        # Collect re-imbursements from:
        #    cam, prop_tax, insurance, utilities
        cam_reimb_i       = self.cam.get_reimb_i(year_i)
        prop_tax_reimb_i  = self.prop_tax.get_reimb_i(year_i)
        insurance_reimb_i = self.insurance.get_reimb_i(year_i)
        utilities_reimb_i = self.utilities.get_reimb_i(year_i)
        #-----
        reimb_tot_i = cam_reimb_i + prop_tax_reimb_i + insurance_reimb_i + utilities_reimb_i

        #-------------------------
        # Gross revenues
        gross_rev_i = base_rental_rev_i+reimb_tot_i
        # Less: vacancies
        vacancies_i = self.vac_pct_gr_srs.loc[year_i]*gross_rev_i
        #-------------------------
        # Net Base Rental Revenue
        net_base_rental_rev_i = gross_rev_i - vacancies_i
        # Plus: Percentage rents
        percentage_rent_i = self.percent_rent.get_val_i(year_i)
        #-------------------------
        # Total rental income
        rent_income_tot_i = net_base_rental_rev_i + percentage_rent_i
        # Plus: Ancillary Income
        anc_inc_i = self.anc_inc.get_val_i(year_i)
        #-------------------------
        # Gross Income
        gross_income_i = rent_income_tot_i+anc_inc_i
        # Less: Credit Loss
        credit_loss_i = gross_income_i*self.credit_loss_pct_gi_srs.loc[year_i]
        #-------------------------
        # Effective Gross Income (EGI)
        egi_i = gross_income_i - credit_loss_i
        #-------------------------
        # Less: Operating expenses
        # Operating expenses
        # ----- Possibly reimbursable -----
        cam       = self.cam.get_val_i(year_i)
        prop_tax  = self.prop_tax.get_val_i(year_i)
        insurance = self.insurance.get_val_i(year_i)
        utilities = self.utilities.get_val_i(year_i)
        # ----- Not reimbursable -----
        mngmnt_fee_i = (cam + prop_tax)*self.mngmnt_fee_pct_reimbs_srs.loc[year_i]
        if year_i<=self.hold_years:
            repl_rsrv_i = self.repl_rsrv_pct*gross_rev_i
        else:
            repl_rsrv_i = 0
        #----------
        # Total expenses
        expenses_tot_i = cam + prop_tax + insurance + utilities + mngmnt_fee_i + repl_rsrv_i
        #-------------------------
        # Net Operating Income
        noi_i = egi_i - expenses_tot_i
        # Less tennant improvements, leasing commissions, cap ex, draw_for_capex_i
        tnnt_imprvs_i = self.tnnt_imprvs.get_val_i(year_i)
        lease_comm_i  = self.lease_comm.get_val_i(year_i)
        capex_i       = self.capex.get_val_i(year_i)
        #-------------------------
        cash_xfer_to_rr_i = repl_rsrv_i
        draw_for_capex_i = self.repl_rsrv_tbl.add_contr_for_year_i(
            contr_i = cash_xfer_to_rr_i,
            capex_i = capex_i,
            return_draw_for_capex = True
        )
        #-------------------------
        # Adjusted Net Operating Income
        adjusted_noi_i = noi_i - tnnt_imprvs_i - lease_comm_i - capex_i - draw_for_capex_i
        # Less: loan points and debt service
        if year_i==0:# TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            loan_points_i = self.loan_points_tot
        else:
            loan_points_i = 0
        #-----
        debt_srvc_pmt_i = self.dbt_serv_pmt_ann
        #-------------------------
        # Before-Tax Levered Cash Flow
        lev_cash_b4_tax_i = adjusted_noi_i - loan_points_i - debt_srvc_pmt_i
        #-------------------------
        # Less: Depreciation for Purchase Price, Tennant Improvements, Cap Ex
        # Less: Leasing commissions amort, loan points amort
        # Plus: Cash xfer to RR, tennant improvement, leasing commissions, principal amort.
        #----------
        prop_depr_i        = self.prop_depr_tbl.get_prop_depr_for_year_i(year_i=year_i)
        tnnt_imprvs_depr_i = self.tnnt_imprvs.depr_tbl.get_total_depr_for_year_i(year_i=year_i)
        capex_depr_i       = self.capex.depr_tbl.get_total_depr_for_year_i(year_i=year_i)
        #----------
        if year_i<=self.hold_years:
            lease_comm_amort_i  = self.lease_comm.simp_amort_tbl.get_total_depr_for_year_i(year_i=year_i)
            if year_i==self.hold_years:
                lease_comm_amort_i += self.lease_comm.simp_amort_tbl.get_still_unamortized_for_year_i(year_i=year_i)
        else:
            lease_comm_amort_i = 0
        #----------
        if year_i<=self.hold_years:
            loan_points_amort_i = self.loan_points.simp_amort_tbl.get_total_depr_for_year_i(year_i=year_i)
            if year_i==self.hold_years:
                loan_points_amort_i += self.loan_points.simp_amort_tbl.get_still_unamortized_for_year_i(year_i=year_i)
        else:
            loan_points_amort_i = 0
        #-------------------------
        principal_amort_i = self.get_mort_amort_year_end_principal(year_i=year_i)
        #-------------------------
        taxable_income_i = (
            lev_cash_b4_tax_i 
            - (prop_depr_i + tnnt_imprvs_depr_i + capex_depr_i) 
            - (lease_comm_amort_i + loan_points_amort_i) 
            + (cash_xfer_to_rr_i + tnnt_imprvs_i + lease_comm_i + principal_amort_i)
        )
        #-------------------------
        # Application of suspended losses
        susp_losses_appld_i = self.susp_tax_loss.add_tax_for_year_i(
            taxable_income_i         = taxable_income_i, 
            return_susp_losses_appld = True
        )
        #-------------------------
        # Net taxable income
        net_taxable_income_i = taxable_income_i - susp_losses_appld_i
        #-------------------------
        if net_taxable_income_i > 0:
            tax_liability_i = self.tax_rate*net_taxable_income_i
        else:
            tax_liability_i = 0
        #-----
        if year_i != self.hold_years:
            rfnd_of_rr_acct_bal = 0
        else:
            rfnd_of_rr_acct_bal = self.repl_rsrv_tbl.repl_rsrv_df[self.repl_rsrv_tbl.repl_rsrv_df['year']==self.hold_years]['book_end']
            assert(rfnd_of_rr_acct_bal.shape[0]==1)
            rfnd_of_rr_acct_bal = rfnd_of_rr_acct_bal.values[0]            
        #-----
        cash_after_tax = lev_cash_b4_tax_i - tax_liability_i + rfnd_of_rr_acct_bal
        #----------------------------------------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------
        self.build_cash_flow_year_i(
            year_i                = year_i, 
            base_rental_rev_i     = base_rental_rev_i, 
            cam_reimb_i           = cam_reimb_i, 
            prop_tax_reimb_i      = prop_tax_reimb_i, 
            insurance_reimb_i     = insurance_reimb_i, 
            utilities_reimb_i     = utilities_reimb_i, 
            reimb_tot_i           = reimb_tot_i, 
            gross_rev_i           = gross_rev_i, 
            vacancies_i           = vacancies_i, 
            net_base_rental_rev_i = net_base_rental_rev_i, 
            percentage_rent_i     = percentage_rent_i, 
            rent_income_tot_i     = rent_income_tot_i, 
            anc_inc_i             = anc_inc_i, 
            gross_income_i        = gross_income_i, 
            credit_loss_i         = credit_loss_i, 
            egi_i                 = egi_i, 
            noi_i                 = noi_i, 
            tnnt_imprvs_i         = tnnt_imprvs_i, 
            lease_comm_i          = lease_comm_i, 
            capex_i               = capex_i, 
            draw_for_capex_i      = draw_for_capex_i, 
            adjusted_noi_i        = adjusted_noi_i, 
            loan_points_i         = loan_points_i, 
            debt_srvc_pmt_i       = debt_srvc_pmt_i, 
            lev_cash_b4_tax_i     = lev_cash_b4_tax_i, 
            prop_depr_i           = prop_depr_i, 
            tnnt_imprvs_depr_i    = tnnt_imprvs_depr_i, 
            capex_depr_i          = capex_depr_i, 
            lease_comm_amort_i    = lease_comm_amort_i, 
            loan_points_amort_i   = loan_points_amort_i, 
            cash_xfer_to_rr_i     = cash_xfer_to_rr_i, 
            principal_amort_i     = principal_amort_i, 
            taxable_income_i      = taxable_income_i, 
            susp_losses_appld_i   = susp_losses_appld_i, 
            net_taxable_income_i  = net_taxable_income_i, 
            tax_liability_i       = tax_liability_i, 
            rfnd_of_rr_acct_bal   = rfnd_of_rr_acct_bal, 
            cash_after_tax        = cash_after_tax, 
        )
        
    def combine_year_i_cash_flows(
        self
    ):
        r"""
        """
        #-------------------------
        assert(len(self.cash_flows_raw_by_year)>=self.proj_years)
        #-------------------------
        df_i_shape0 = self.cash_flows_raw_by_year[1].shape[0]
        df_i_cols = self.cash_flows_raw_by_year[1].columns.tolist()
        cash_flows_dfs = []
        for year_i in range(1, self.proj_years+1):
            df_i = self.cash_flows_raw_by_year[year_i]
            assert(df_i.shape[0]==df_i_shape0)
            assert(df_i.columns.tolist()==df_i_cols)
            if year_i==1:
                cash_flows_dfs.append(df_i)
            else:
                cash_flows_dfs.append(df_i['val'].to_frame())
        #-------------------------
        cash_flows_df = pd.concat(cash_flows_dfs, axis=1)
        #-----
        assert(cash_flows_df.columns.tolist() == ['desc']+['val']*(cash_flows_df.shape[1]-1))
        new_cols = ['desc'] + [f'Year {x}' for x in range(1, cash_flows_df.shape[1])]
        assert(len(new_cols) == cash_flows_df.shape[1])
        cash_flows_df.columns = new_cols
        #-----
        self.cash_flows_df = cash_flows_df
        

    def get_desc_iloc(
        self, 
        desc
    ):
        r"""
        """
        #-------------------------
        idx = Utilities.find_idxs_in_list_with_regex(
            lst=self.cash_flows_df['desc'].tolist(), 
            regex_pattern=r'^{}$'.format(desc), 
            ignore_case=False
        )
        assert(len(idx)==1)
        idx = idx[0]    
        return idx
        
    def assemble_time_0(
        self
    ):
        r"""
        """
        #-------------------------
        loan_pts_idx = self.get_desc_iloc('Less: Loan Points')
        init_eqt_invstmnt_idx = self.get_desc_iloc('Less: Initial Equity Investment')
        #-------------------------
        table_0 = pd.DataFrame(
            index   = self.cash_flows_df.index, 
            columns = ['desc', 'Time 0'], 
            data    = 0
        )
        table_0['desc'] = self.cash_flows_df['desc']
        #-------------------------
        table_0.iloc[loan_pts_idx, -1]          = -self.loan_points_tot
        table_0.iloc[init_eqt_invstmnt_idx, -1] = -self.equity_0-self.loan_points_tot    
        #-------------------------
        return table_0
    
    def prepend_time_0_to_cfs_df(
        self, 
        time_0_df
    ):
        r"""
        """
        #-------------------------
        assert(time_0_df.index.equals(self.cash_flows_df.index)) # don't really need bc of assert below
        assert(time_0_df['desc'].equals(self.cash_flows_df['desc']))
        #-------------------------
        og_shape = self.cash_flows_df.shape
        #-------------------------
        cash_flows_df = pd.merge(
            time_0_df, 
            self.cash_flows_df, 
            left_index            = True, 
            right_index           = True,
            how                = 'inner'
        ).drop(columns=['desc_y']).rename(columns={'desc_x':'desc'})
        #-------------------------
        assert(self.cash_flows_df.shape[0]   == cash_flows_df.shape[0])
        assert(self.cash_flows_df.shape[1]+1 == cash_flows_df.shape[1])
        #-------------------------
        self.cash_flows_df = cash_flows_df
        
    def assemble_and_prepend_time_0(
        self
    ):
        r"""
        """
        #-------------------------
        time_0_df = self.assemble_time_0()
        self.prepend_time_0_to_cfs_df(time_0_df=time_0_df)
        
        
    def append_after_tax_levered_cash_flow(
        self, 
        include_adj=True
    ):
        r"""
        """
        #--------------------------------------------------
        aftr_tax_cash_idx          = self.get_desc_iloc('After-Tax Cash Flow')
        net_sales_price_idx        = self.get_desc_iloc('Net Sales Price')
        sale_inc_tax_liability_idx = self.get_desc_iloc('Less: Sale Income Tax Liability')
        mort_bal_outstanding_idx   = self.get_desc_iloc('Less: Outstanding Mortgage Balance')
        init_eqt_invstmnt_idx      = self.get_desc_iloc('Less: Initial Equity Investment')
        #-------------------------
        assert(init_eqt_invstmnt_idx==self.cash_flows_df.shape[0]-1)
        assert(
            [aftr_tax_cash_idx, net_sales_price_idx, sale_inc_tax_liability_idx, mort_bal_outstanding_idx, init_eqt_invstmnt_idx] == 
            list(range(self.cash_flows_df.shape[0]-5, self.cash_flows_df.shape[0]))
        )

        #--------------------------------------------------
        # Annoying, but have to strip off columns that won't be used in addition/subtraction of rows 
        #   e.g., what is, 'After-Tax Cash Flow' + 'Net Sales Price'???  'After-Tax Cash FlowNet Sales Price' seems sensible
        #    but, what is, 'Net Sales Price	' - 'Less: Sale Income Tax Liability'????? I have no idea
        data_col_idxs = Utilities.find_idxs_in_list_with_regex(
            lst           = self.cash_flows_df.columns.tolist(),
            regex_pattern = r'(Time|Year) \d*',
            ignore_case   = False
        )

        #--------------------------------------------------
        # All signs should be handled previous, so all addition here!!!
        # i.e., not aftr_tax_cash_idx + net_sales_price_idx - sale_inc_tax_liability_idx - mort_bal_outstanding_idx - init_eqt_invstmnt_idx
        #       but aftr_tax_cash_idx + net_sales_price_idx + sale_inc_tax_liability_idx + mort_bal_outstanding_idx + init_eqt_invstmnt_idx
        #   as the last 3 should already be negative
        aftr_tax_lev_cf_srs = (
            self.cash_flows_df.iloc[aftr_tax_cash_idx, data_col_idxs] 
            + self.cash_flows_df.iloc[net_sales_price_idx, data_col_idxs] 
            + self.cash_flows_df.iloc[sale_inc_tax_liability_idx, data_col_idxs] 
            + self.cash_flows_df.iloc[mort_bal_outstanding_idx, data_col_idxs] 
            + self.cash_flows_df.iloc[init_eqt_invstmnt_idx, data_col_idxs]
        )
        aftr_tax_lev_cf_srs['desc'] = 'After-Tax Levered Cash Flow'
        #-----
        # Technically, this next step turns the srs into a single-row df
        aftr_tax_lev_cf_srs = aftr_tax_lev_cf_srs.to_frame(name=self.cash_flows_df.shape[0]).T
        assert(set(aftr_tax_lev_cf_srs.columns).symmetric_difference(set(self.cash_flows_df.columns))==set())
        #-----
        aftr_tax_lev_cf_srs = aftr_tax_lev_cf_srs[self.cash_flows_df.columns]
        #--------------------------------------------------
        if not include_adj:
            self.cash_flows_df = pd.concat([self.cash_flows_df, aftr_tax_lev_cf_srs])
            return
        #--------------------------------------------------
        #--------------------------------------------------
        # Also want to incorporate Ryan's NPV for property valuation
        aftr_tax_lev_cf_srs_alt = (
            self.cash_flows_df.iloc[aftr_tax_cash_idx, data_col_idxs] 
            + self.cash_flows_df.iloc[net_sales_price_idx, data_col_idxs] 
        )
        aftr_tax_lev_cf_srs_alt['desc'] = 'After-Tax Levered Cash Flow (Alt)'
        #-----
        # Technically, this next step turns the srs into a single-row df
        aftr_tax_lev_cf_srs_alt = aftr_tax_lev_cf_srs_alt.to_frame(name=self.cash_flows_df.shape[0]+1).T
        assert(set(aftr_tax_lev_cf_srs_alt.columns).symmetric_difference(set(self.cash_flows_df.columns))==set())
        #-----
        aftr_tax_lev_cf_srs_alt = aftr_tax_lev_cf_srs_alt[self.cash_flows_df.columns]
        #--------------------------------------------------
        self.cash_flows_df = pd.concat([self.cash_flows_df, aftr_tax_lev_cf_srs, aftr_tax_lev_cf_srs_alt])
        
        
    def finalize_cash_flows_df(
        self
    ):
        r"""
        """
        #-------------------------
        adj_noi_srs = self.cash_flows_df[self.cash_flows_df['desc']=='Adjusted Net Operating Income']
        assert(adj_noi_srs.shape[0]==1)
        assert(f"Year {self.proj_years}" in adj_noi_srs.columns.tolist())
        hold_year_p1_adj_noi = adj_noi_srs.iloc[0][f"Year {self.proj_years}"]
        #-----
        cap_rate_exit        = self.cap_rate_exit
        sell_costs_pct_gsp   = self.sell_costs_pct_gsp
        purchase_price       = self.purchase_price
        acc_depr_tax_rate    = self.acc_depr_tax_rate
        cap_gains_tax_rate   = self.cap_gains_tax_rate
        #-----
        net_ti    = np.sum([self.tnnt_imprvs.get_val_i(year_i) for year_i in range(1, self.hold_years+1)])
        net_capex = np.sum([self.capex.get_val_i(year_i) for year_i in range(1, self.hold_years+1)])
        net_ti_and_capex = net_ti + net_capex
        #-----
        accum_depr = (
            self.tnnt_imprvs.depr_tbl.cum_depr_tbl_df.loc['Total', self.hold_years] +
            self.capex.depr_tbl.cum_depr_tbl_df.loc['Total', self.hold_years] + 
            self.prop_depr_tbl.cum_depr_tbl_df.loc['Total', self.hold_years]
        )
        #-----
        mort_bal_outstanding = self.get_mort_amort_year_end_balance(year_i=self.hold_years)
        #-------------------------
        sale_items = []
        for year_i in range(1, 1+self.proj_years):
            if year_i==0:
                sale_items_i = pd.Series(
                    index=['net_sales_price', 'inc_tax_liability', 'mort_bal_outstanding', 'init_eqt_invstmnt'], 
                    data=[0, 0, 0, -self.equity_0-self.loan_points_tot]
                )
            elif year_i != self.hold_years:
                sale_items_i = pd.Series(
                    index=['net_sales_price', 'inc_tax_liability', 'mort_bal_outstanding', 'init_eqt_invstmnt'], 
                    data=0
                )
            else:
                nsp_cal_srs = REValuator.sale_income_tax_accounting(
                    hold_year_p1_adj_noi, 
                    cap_rate_exit, 
                    sell_costs_pct_gsp, 
                    purchase_price, 
                    net_ti_and_capex, 
                    accum_depr, 
                    mort_bal_outstanding, 
                    acc_depr_tax_rate  = acc_depr_tax_rate, 
                    cap_gains_tax_rate = cap_gains_tax_rate, 
                    hold_years         = None, 
                    return_full        = False
                )
                #-----
                sale_items_i = nsp_cal_srs.loc[['net_sales_price', 'inc_tax_liability', 'mort_bal_outstanding']]
                sale_items_i['inc_tax_liability']    = -1*sale_items_i['inc_tax_liability']
                sale_items_i['mort_bal_outstanding'] = -1*sale_items_i['mort_bal_outstanding']
                sale_items_i['init_eqt_invstmnt']    = 0
            #-------------------------
            sale_items.append(sale_items_i)
        #-------------------------
        cash_flows_df_fnl_rows = pd.concat(sale_items, axis=1)
        #-------------------------
        cash_flows_df_fnl_rows.columns = [f"Year {col_i+1}" for col_i in cash_flows_df_fnl_rows.columns.tolist()]
        #-------------------------
        idx_j = self.cash_flows_df.index[-1]+1
        #-----
        cash_flows_df_fnl_rows.index.name = 'desc'
        cash_flows_df_fnl_rows = cash_flows_df_fnl_rows.reset_index(drop=False)
        cash_flows_df_fnl_rows.index = range(idx_j, idx_j+cash_flows_df_fnl_rows.shape[0])
        #-------------------------
        assert(cash_flows_df_fnl_rows.columns.tolist()==self.cash_flows_df.columns.tolist())
        #-------------------------
        cash_flows_df_fnl_rows['desc'] = cash_flows_df_fnl_rows['desc'].replace(dict(
            net_sales_price      = 'Net Sales Price', 
            inc_tax_liability    = 'Less: Sale Income Tax Liability', 
            mort_bal_outstanding = 'Less: Outstanding Mortgage Balance', 
            init_eqt_invstmnt    = 'Less: Initial Equity Investment'
        ))
        #-------------------------
        self.cash_flows_df = pd.concat([self.cash_flows_df, cash_flows_df_fnl_rows])
        #-------------------------
        self.assemble_and_prepend_time_0()
        
        #-------------------------
        # Make sure data types are as desired
        str_cols = ['desc']
        int_cols = ['Time 0']
        float_cols = [x for x in self.cash_flows_df.columns.tolist() if (x not in str_cols and x not in int_cols)]
        #-----
        cols_and_types_dict = (
            {x:str   for x in str_cols} |
            {x:int   for x in int_cols} |
            {x:float for x in float_cols}
        )
        #-----
        self.cash_flows_df = Utilities_df.convert_col_types(
            df                  = self.cash_flows_df, 
            cols_and_types_dict = cols_and_types_dict, 
            to_numeric_errors   = 'coerce',
            inplace             = True    
        )
        #-------------------------
        self.append_after_tax_levered_cash_flow(include_adj=True)
        
        #-------------------------
        # This didn't work at the time on constructing time_0_df or whatever it's called.
        # There, I couldn't simply set = to '', or np.nan, or None!!!! Not exactly sure why...
        # Trying a workaround at this point...
        t0_col_idx = Utilities_df.find_idxs_in_highest_order_of_columns(
            df            = self.cash_flows_df, 
            col           = 'Time 0', 
            exact_match   = True, 
            assert_single = True
        )
        self.cash_flows_df.iloc[self.get_desc_iloc('Expense Reimbursements'),    t0_col_idx] = self.cash_flows_df.iloc[self.get_desc_iloc('Expense Reimbursements'),    t0_col_idx+1]
        self.cash_flows_df.iloc[self.get_desc_iloc('Less: Operating Expenses'),  t0_col_idx] = self.cash_flows_df.iloc[self.get_desc_iloc('Less: Operating Expenses'),  t0_col_idx+1]
        self.cash_flows_df.iloc[self.get_desc_iloc('Reimbursable Expenses'),     t0_col_idx] = self.cash_flows_df.iloc[self.get_desc_iloc('Reimbursable Expenses'),     t0_col_idx+1]
        self.cash_flows_df.iloc[self.get_desc_iloc('Non-Reimbursable Expenses'), t0_col_idx] = self.cash_flows_df.iloc[self.get_desc_iloc('Non-Reimbursable Expenses'), t0_col_idx+1]
        #-------------------------
        
        
        
        
    def build_cash_flows(
        self
    ):
        r"""
        """
        #-------------------------
        for year_i in range(1, 1+self.proj_years):
            self.construct_cash_flows_for_year_i(year_i = year_i)

        self.combine_year_i_cash_flows()
        self.finalize_cash_flows_df()
        
        
      
    def get_egi_idx(self):
        return self.get_desc_iloc(r'Effective Gross Income \(EGI\)')
    #-----
    def get_noi_idx(self):
        return self.get_desc_iloc(r'Net Operating Income \(NOI\)')
    #-----
    def get_adj_noi_idx(self):
        return self.get_desc_iloc(r'Adjusted Net Operating Income')
    #-----
    def get_bfr_tax_lev_cf_idx(self):
        return self.get_desc_iloc(r'Before-Tax Levered Cash Flow')
    #-----
    def get_taxable_inc_idx(self):
        return self.get_desc_iloc(r'Taxable Income')
    #-----
    def get_net_taxable_inc_idx(self):
        return self.get_desc_iloc(r'Net Taxable Income')
    #-----
    def get_aftr_tax_cf_idx(self):
        return self.get_desc_iloc(r'After-Tax Cash Flow')
    #-----
    def get_aftr_tax_lev_cf_idx(self):
        return self.get_desc_iloc(r'After-Tax Levered Cash Flow')
    #-----
    def get_aftr_tax_lev_cf_alt_idx(self):
        return self.get_desc_iloc(r'After-Tax Levered Cash Flow \(Alt\)')
    #-----
    def get_gross_revs_idx(self):
        return self.get_desc_iloc(r'Gross Revenues')
    #-----
    def get_gross_inc_idx(self):
        return self.get_desc_iloc(r'Gross Income')
    #-----
    def get_total_reimbs_idx(self):
        return self.get_desc_iloc(r'Total Reimbursements')
    #-----
    def get_total_expenses_idx(self):
        return self.get_desc_iloc(r'Total Expenses')
    #-----
    def get_base_rental_rev_idx(self):
        return self.get_desc_iloc(r'Base Rental Revenues')
    #-----
    
      
    def get_rows_to_shade_dict(
        self
    ):
        r"""
        """
        #-------------------------
        return_dict = dict(
            egi_idx                    = self.get_egi_idx(), 
            noi_idx                    = self.get_noi_idx(), 
            adj_noi_idx                = self.get_adj_noi_idx(), 
            bfr_tax_lev_cf_idx         = self.get_bfr_tax_lev_cf_idx(), 
            taxable_inc_idx            = self.get_taxable_inc_idx(), 
            net_taxable_inc_idx        = self.get_net_taxable_inc_idx(), 
            aftr_tax_cf_idx            = self.get_aftr_tax_cf_idx(),
            aftr_tax_lev_cf_idx        = self.get_aftr_tax_lev_cf_idx(), 
            gross_revs_idx             = self.get_gross_revs_idx(), 
            gross_inc_idx_idx          = self.get_gross_inc_idx(), 
            total_reimbs_idx           = self.get_total_reimbs_idx(), 
            total_expenses_idx         = self.get_total_expenses_idx(), 
            base_rental_rev_idx        = self.get_base_rental_rev_idx(), 
        )
        #-------------------------
        return return_dict
        
    def get_rows_to_shade(
        self
    ):
        r"""
        """
        #-------------------------
        return list(self.get_rows_to_shade_dict().values())
        
        
    def calc_npv(
        self, 
        dscnt_rate, 
        calc_alt=False
    ):
        r"""
        IMPORTANT NOTE: npf.npv is programmed different from Excel's NPV
                        npf.npv expects the user to input time_0 cash flow
                        Excel's NPV expects the user to add time_0 cash flow on his own
        """
        #-------------------------
        if calc_alt:
            cf_idx = self.get_aftr_tax_lev_cf_alt_idx()
        else:
            cf_idx = self.get_aftr_tax_lev_cf_idx()
        #-------------------------
        cfs    = self.cash_flows_df.iloc[cf_idx].copy()
        #-------------------------
        needed_year_pds = [f"Year {year_i}" for year_i in range(1, self.hold_years+1)]
        assert(set(needed_year_pds).difference(cfs.index)==set())
        assert('Time 0' in cfs.index.tolist())
        #-------------------------
        npv = npf.npv(dscnt_rate, cfs.loc[['Time 0']+needed_year_pds].values)
        #-------------------------
        return npv

    def calc_irr(
        self
    ):
        r"""
        IMPORTANT NOTE: npf.npv is programmed different from Excel's NPV
                        npf.npv expects the user to input time_0 cash flow
                        Excel's NPV expects the user to add time_0 cash flow on his own
        """
        #-------------------------
        cf_idx = self.get_aftr_tax_lev_cf_idx()
        #-------------------------
        cfs    = self.cash_flows_df.iloc[cf_idx].copy()
        #-------------------------
        needed_year_pds = [f"Year {year_i}" for year_i in range(1, self.hold_years+1)]
        assert(set(needed_year_pds).difference(cfs.index)==set())
        assert('Time 0' in cfs.index.tolist())
        #-------------------------
        irr = npf.irr(cfs.loc[['Time 0']+needed_year_pds].values)
        #-------------------------
        return irr

    def calc_npv_and_irr(
        self, 
        dscnt_rate
    ):
        r"""
        IMPORTANT NOTE: npf.npv is programmed different from Excel's NPV
                        npf.npv expects the user to input time_0 cash flow
                        Excel's NPV expects the user to add time_0 cash flow on his own
        """
        #-------------------------
        cf_idx = self.get_aftr_tax_lev_cf_idx()
        #-------------------------
        cfs    = self.cash_flows_df.iloc[cf_idx].copy()
        #-------------------------
        needed_year_pds = [f"Year {year_i}" for year_i in range(1, self.hold_years+1)]
        assert(set(needed_year_pds).difference(cfs.index)==set())
        assert('Time 0' in cfs.index.tolist())
        #-------------------------
        npv = npf.npv(dscnt_rate, cfs.loc[['Time 0']+needed_year_pds].values)
        irr = npf.irr(cfs.loc[['Time 0']+needed_year_pds].values)
        #-------------------------
        return npv, irr


    def get_cash_flow_srs(
        self, 
        get_alt=False, 
        return_desc=False
    ):
        r"""
        """
        #-------------------------
        if get_alt:
            cf_idx = self.get_aftr_tax_lev_cf_alt_idx()
        else:
            cf_idx = self.get_aftr_tax_lev_cf_idx()
        #-------------------------
        cfs    = self.cash_flows_df.iloc[cf_idx].copy()
        #-------------------------
        needed_year_pds = [f"Year {year_i}" for year_i in range(1, self.hold_years+1)]
        assert(set(needed_year_pds).difference(cfs.index)==set())
        assert('Time 0' in cfs.index.tolist())
        #-------------------------
        if return_desc:
            return cfs.loc[['Time 0']+needed_year_pds], cfs.loc['desc']
        else:
            return cfs.loc[['Time 0']+needed_year_pds]
        

    def calc_net_cash_flow(
        self, 
        calc_alt=False
    ):
        r"""
        IMPORTANT NOTE: npf.npv is programmed different from Excel's NPV
                        npf.npv expects the user to input time_0 cash flow
                        Excel's NPV expects the user to add time_0 cash flow on his own
        """
        #-------------------------
        cfs_srs = self.get_cash_flow_srs(
            get_alt     = calc_alt, 
            return_desc = False
        )
        #-------------------------
        net_cf = np.sum(cfs_srs.values)
        #-------------------------
        return net_cf
        
    def calc_net_after_tax_cash_flows(
        self
    ):
        r"""
        """
        #-------------------------
        cfs_srs = self.get_cash_flow_srs(
            get_alt     = False, 
            return_desc = False
        )
        rtrn_val = cfs_srs.loc['Year 1':].sum()
        return rtrn_val

    def calc_multiple(
        self
    ):
        r"""
        """
        #-------------------------
        init_eq_invst = self.equity_0+self.loan_points_tot
        net_cf = self.calc_net_cash_flow(calc_alt=False)
        #-----
        multiple = 1 + net_cf/init_eq_invst
        return multiple
        
        
    def print_results_summary(
        self, 
        dscnt_rate, 
        include_cfs_srs=True
    ):
        r"""
        """
        #-------------------------
        npv, irr = self.calc_npv_and_irr(
            dscnt_rate = dscnt_rate
        )
        #-----
        npv_alt = self.calc_npv(
            dscnt_rate = dscnt_rate, 
            calc_alt   = True
        )
        #-----
        net_cf     = self.calc_net_cash_flow(calc_alt = False)
        net_cf_alt = self.calc_net_cash_flow(calc_alt = True)
        #-----
        multpl = self.calc_multiple()
        #-------------------------
        print('-'*50)
        print('-'*25)
        print(f"npv     = {npv}")
        print('-'*10)
        print(f"irr     = {irr}")
        print('-'*10)
        print(f"net_cf  = {net_cf}")
        print(f"multpl  = {multpl}")
        print('-'*25)
        print(f"npv_alt = {npv_alt}")    
        #-------------------------
        if include_cfs_srs:
            print('\n')
            print('-'*50)
            print('-'*50)
            cfs_srs, desc = self.get_cash_flow_srs(return_desc=True)
            print(cfs_srs.to_frame(name=desc).T)
            
            
    def get_nsp_cal_srs(
        self, 
        return_full = False
    ):
        r"""
        """
        #-------------------------
        adj_noi_srs = self.cash_flows_df.iloc[self.get_adj_noi_idx()]
        assert(f"Year {self.proj_years}" in adj_noi_srs.index.tolist())
        hold_year_p1_adj_noi = adj_noi_srs[f"Year {self.proj_years}"]
        #-----
        cap_rate_exit        = self.cap_rate_exit
        sell_costs_pct_gsp   = self.sell_costs_pct_gsp
        purchase_price       = self.purchase_price
        acc_depr_tax_rate    = self.acc_depr_tax_rate
        cap_gains_tax_rate   = self.cap_gains_tax_rate
        #-----
        net_ti    = np.sum([self.tnnt_imprvs.get_val_i(year_i) for year_i in range(1, self.hold_years+1)])
        net_capex = np.sum([self.capex.get_val_i(year_i) for year_i in range(1, self.hold_years+1)])
        net_ti_and_capex = net_ti + net_capex
        #-----
        accum_depr = (
            self.tnnt_imprvs.depr_tbl.cum_depr_tbl_df.loc['Total', self.hold_years] +
            self.capex.depr_tbl.cum_depr_tbl_df.loc['Total', self.hold_years] + 
            self.prop_depr_tbl.cum_depr_tbl_df.loc['Total', self.hold_years]
        )
        #-----
        mort_bal_outstanding = self.get_mort_amort_year_end_balance(year_i=self.hold_years)

        #-----
        nsp_cal_srs = REValuator.sale_income_tax_accounting(
            hold_year_p1_adj_noi, 
            cap_rate_exit, 
            sell_costs_pct_gsp, 
            purchase_price, 
            net_ti_and_capex, 
            accum_depr, 
            mort_bal_outstanding, 
            acc_depr_tax_rate  = acc_depr_tax_rate, 
            cap_gains_tax_rate = cap_gains_tax_rate, 
            hold_years         = self.hold_years, 
            return_full        = return_full
        )

        return nsp_cal_srs
        
        
    def calc_nsp_pct_tot(
        self
    ):
        r"""
        """
        #-------------------------
        nsp_srs = self.get_nsp_cal_srs()
        #-----
        num = nsp_srs['net_sales_price'] - nsp_srs['inc_tax_liability'] - nsp_srs['mort_bal_outstanding']
        den = self.calc_net_after_tax_cash_flows()
        #-----
        rtrn_val = num/den
        #-----
        return rtrn_val
        
        
    def get_final_results_to_excel_df(
        self, 
        dscnt_rate, 
        return_tot_row = False
    ):
        r"""
        """
        #-------------------------
        df = self.get_cash_flow_srs(get_alt=False, return_desc=False).to_frame(name='').T
        #--------------------------------------------------
        # Get the dimensions of the dataframe.
        (max_row, max_col) = df.shape
        tot_row = max_row
        #--------------------------------------------------
        df = pd.concat([df, pd.DataFrame(index=[''], columns=df.columns)])
        #--------------------------------------------------
        metrics_srs = pd.Series({
            "Net Cash Flow"                          : np.round(self.calc_net_cash_flow(calc_alt=False), 2), 
            "IRR"                                    : np.round(100*self.calc_irr(), 2), 
            f"NPV at {np.round(100*dscnt_rate, 2)}%" : np.round(self.calc_npv(dscnt_rate = dscnt_rate), 2), 
            "Multiple"                               : np.round(self.calc_multiple(), 2), 
            "Net After-Tax Cash Flows"               : np.round(self.calc_net_after_tax_cash_flows(), 2), 
            "Net Sales Proceeds % Total"             : np.round(100*self.calc_nsp_pct_tot(), 2)
        })
        metrics_df = metrics_srs.to_frame(name='Time 0')
        df         = pd.concat([df, metrics_df])
        #--------------------------------------------------
        if return_tot_row:
            return df, tot_row
        else:
            return df
            
    def get_full_results_to_excel_df(
        self, 
        dscnt_rate, 
        return_tot_row = False
    ):
        r"""
        """
        #--------------------------------------------------
        df = self.cash_flows_df.copy()
        #--------------------------------------------------
        # Get the dimensions of the dataframe.
        (max_row, max_col) = df.shape
        tot_row = max_row
        #--------------------------------------------------
        df = pd.concat([df, pd.DataFrame(index=[''], columns=df.columns)])
        #--------------------------------------------------
        metrics_srs = pd.Series({
            "Net Cash Flow"                          : np.round(self.calc_net_cash_flow(calc_alt=False), 2), 
            "IRR"                                    : np.round(100*self.calc_irr(), 2), 
            f"NPV at {np.round(100*dscnt_rate, 2)}%" : np.round(self.calc_npv(dscnt_rate = dscnt_rate), 2), 
            "Multiple"                               : np.round(self.calc_multiple(), 2), 
            "Net After-Tax Cash Flows"               : np.round(self.calc_net_after_tax_cash_flows(), 2), 
            "Net Sales Proceeds % Total"             : np.round(100*self.calc_nsp_pct_tot(), 2)
        })
        metrics_df = metrics_srs.to_frame(name='Time 0')
        df         = pd.concat([df, metrics_df])
        #--------------------------------------------------
        if return_tot_row:
            return df, tot_row
        else:
            return df
            
    @staticmethod 
    def output_results_to_excel_worksheet(
        df, 
        writer, 
        sheet_name    = 'Sheet1', 
        tot_row       = None, 
        col_max_len   = None, 
        rows_to_shade = None,
        startrow      = 0, 
        startcol      = 0
    ):
        r"""
        """
        #-------------------------
        max_col = df.shape[1]
        #--------------------------------------------------
        # Convert the dataframe to an XlsxWriter Excel object.
        df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol)

        # Get the xlsxwriter workbook and worksheet objects.
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        #-------------------------
        if col_max_len is None:
            col_max_lens = []
            for idx, col in enumerate(df):  # loop through all columns
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                )) + 1  # adding a little extra space
                # worksheet.set_column(idx, idx, max_len)  # set column width
                col_max_lens.append(max_len)
            col_max_len = np.max(col_max_lens)
        #-------------------------
        # Light red fill with dark red text.
        format1 = workbook.add_format({'bg_color':   '#d9d9d9',
                                       'font_color': '#000000'})
        #-------------------------
        # Add some cell formats.
        format_num = workbook.add_format({"num_format": "#,##0.00"})
        format_pct = workbook.add_format({"num_format": "0%"})
        #-------------------------
        # _ = worksheet.set_column(1, last_col=max_col, cell_format=format_num)
        _ = worksheet.set_column(1, max_col, col_max_len, cell_format=format_num)
        #-------------------------
        if startcol==0:
            idx_width = np.max([len(str(x)) for x in df.index.tolist()])
            _ = worksheet.set_column(0, 1, idx_width)
        #-------------------------
        format_red = workbook.add_format({'font_color': '#ff0000'})        
        worksheet.conditional_format(1, 1, df.shape[0], max_col, {"type": "cell", 'criteria': 'less than', 'value': 0, 'format': format_red})
        #-------------------------
        if tot_row is not None:
            # Apply a conditional format to the required cell range.
            # worksheet.conditional_format(tot_row, 1, tot_row, max_col, {"type": "3_color_scale"})
            # worksheet.set_row(tot_row, cell_format=format1)
            worksheet.conditional_format(tot_row, 1, tot_row, max_col, {"type": "cell", 'criteria': 'greater than or equal to', 'value': df.min().min(), 'format': format1})
        #-------------------------    
        if rows_to_shade is not None:
            for row_i in rows_to_shade:
                row_i = row_i+1 # Excel numbering starts at 1, iloc starts at 0
                worksheet.conditional_format(row_i, 1, row_i, max_col, {"type": "cell", 'criteria': 'greater than or equal to', 'value': df.min().min(), 'format': format1})
            
            
    @staticmethod 
    def output_results_to_excel(
        df, 
        path, 
        tot_row       = None, 
        col_max_len   = None, 
        rows_to_shade = None,
        startrow      = 0, 
        startcol      = 0
    ):
        r"""
        """
        #--------------------------------------------------
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(path, engine="xlsxwriter")
        #-------------------------    
        REValuator.output_results_to_excel_worksheet(
                df            = df, 
                writer        = writer, 
                sheet_name    = 'Sheet1', 
                tot_row       = tot_row, 
                col_max_len   = col_max_len, 
                rows_to_shade = rows_to_shade, 
                startrow      = startrow, 
                startcol      = startcol
            )
        #-------------------------
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
        
        
    def output_final_results_to_excel(
        self, 
        dscnt_rate, 
        path, 
        col_max_len=None,
        startrow      = 0, 
        startcol      = 0
    ):
        r"""
        """
        #-------------------------
        df, tot_row = self.get_final_results_to_excel_df(
            dscnt_rate     = dscnt_rate, 
            return_tot_row = True
        )
        #-------------------------
        REValuator.output_results_to_excel(
            df            = df, 
            path          = path, 
            tot_row       = tot_row, 
            col_max_len   = col_max_len, 
            rows_to_shade = None, 
            startrow      = startrow, 
            startcol      = startcol
        )
        
        
    def output_full_results_to_excel(
        self, 
        dscnt_rate, 
        path, 
        col_max_len   = None,
        startrow      = 0, 
        startcol      = 0
    ):
        r"""
        """
        #-------------------------
        df, tot_row = self.get_full_results_to_excel_df(
            dscnt_rate     = dscnt_rate, 
            return_tot_row = True
        )
        #-------------------------
        REValuator.output_results_to_excel(
            df            = df, 
            path          = path, 
            tot_row       = tot_row, 
            col_max_len   = col_max_len, 
            rows_to_shade = self.get_rows_to_shade(), 
            startrow      = startrow, 
            startcol      = startcol
        )
        
    def output_final_and_full_results_to_excel(
        self, 
        dscnt_rate, 
        path, 
        col_max_len   = None,
        startrow_fnl  = 0, 
        startcol_fnl  = 0,
        startrow_full = 0, 
        startcol_full = 0
    ):
        r"""
        """
        #--------------------------------------------------
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(path, engine="xlsxwriter")
        
        #--------------------------------------------------
        df_fnl, tot_row_fnl = self.get_final_results_to_excel_df(
            dscnt_rate     = dscnt_rate, 
            return_tot_row = True
        )
        #-------------------------    
        REValuator.output_results_to_excel_worksheet(
                df            = df_fnl, 
                writer        = writer, 
                sheet_name    = 'Final', 
                tot_row       = tot_row_fnl, 
                col_max_len   = col_max_len, 
                rows_to_shade = None, 
                startrow      = startrow_fnl, 
                startcol      = startcol_fnl
            )
            
        #--------------------------------------------------
        df_full, tot_row_full = self.get_full_results_to_excel_df(
            dscnt_rate     = dscnt_rate, 
            return_tot_row = True
        )
        #-------------------------    
        REValuator.output_results_to_excel_worksheet(
                df            = df_full, 
                writer        = writer, 
                sheet_name    = 'Full', 
                tot_row       = tot_row_full, 
                col_max_len   = col_max_len, 
                rows_to_shade = self.get_rows_to_shade(), 
                startrow      = startrow_full, 
                startcol      = startcol_full
            )
        
        #--------------------------------------------------
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
        
        
    def output_all_depr_amort_tables_to_excel_worksheet(
        self, 
        writer, 
        sheet_name      = 'Sheet1', 
        col_max_len     = None, 
        startrow        = 0, 
        startcol        = 0, 
        return_startrow = False
    ):
        r"""
        """
        #--------------------------------------------------
        startrow = self.prop_depr_tbl.output_pct_allocs_and_depr_tbl_to_excel_worksheet(
            writer          = writer, 
            sheet_name      = sheet_name, 
            col_max_len     = col_max_len, 
            startrow        = startrow, 
            startcol        = startcol, 
            return_startrow = True
        )
        startrow += 2
        #-------------------------
        startrow = self.tnnt_imprvs.depr_tbl.output_depr_tbl_to_excel_worksheet(
            writer        = writer, 
            sheet_name    = sheet_name, 
            col_max_len   = col_max_len, 
            startrow      = startrow, 
            startcol      = startcol, 
            return_startrow = True
        )
        startrow += 2
        #-------------------------
        startrow = self.capex.depr_tbl.output_depr_tbl_to_excel_worksheet(
            writer        = writer, 
            sheet_name    = sheet_name, 
            col_max_len   = col_max_len, 
            startrow      = startrow, 
            startcol      = startcol, 
            return_startrow = True
        )
        startrow += 2
        #-------------------------
        startrow = self.lease_comm.simp_amort_tbl.output_depr_tbl_to_excel_worksheet(
            writer        = writer, 
            sheet_name    = sheet_name, 
            col_max_len   = col_max_len, 
            startrow      = startrow, 
            startcol      = startcol, 
            return_startrow = True
        )
        startrow += 2
        #-------------------------
        startrow = self.loan_points.simp_amort_tbl.output_depr_tbl_to_excel_worksheet(
            writer        = writer, 
            sheet_name    = sheet_name, 
            col_max_len   = col_max_len, 
            startrow      = startrow, 
            startcol      = startcol, 
            return_startrow = True
        )
        if return_startrow:
            return startrow


    def output_all_depr_amort_tables_to_excel(
        self, 
        path, 
        col_max_len   = None, 
        startrow      = 0, 
        startcol      = 0
    ):
        r"""
        """
        #--------------------------------------------------
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(path, engine="xlsxwriter")
        #-------------------------
        self.output_all_depr_amort_tables_to_excel_worksheet(
            writer        = writer, 
            sheet_name    = 'Sheet1', 
            col_max_len   = col_max_len, 
            startrow      = startrow, 
            startcol      = startcol
        )
        #-------------------------
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()
        
        
    def output_susp_tax_loss_to_excel_worksheet(
        self, 
        writer, 
        sheet_name      = 'Sheet1', 
        col_max_len     = None, 
        startrow        = 0, 
        startcol        = 0, 
        return_startrow = False
    ):
        r"""
        """
        #--------------------------------------------------
        df = self.susp_tax_loss.susp_tax_df.copy()
        #-------------------------
        DepreciationTable.output_results_to_excel_worksheet(
            df            = df, 
            writer        = writer, 
            sheet_name    = sheet_name, 
            tot_row       = None, 
            col_max_len   = col_max_len, 
            rows_to_shade = None,
            startrow      = startrow, 
            startcol      = startcol
        )
        startrow += df.shape[0]
        if return_startrow:
            return startrow

    def output_repl_rsrv_tbl_to_excel_worksheet(
        self, 
        writer, 
        sheet_name      = 'Sheet1', 
        col_max_len     = None, 
        startrow        = 0, 
        startcol        = 0, 
        return_startrow = False
    ):
        r"""
        """
        #--------------------------------------------------
        df = self.repl_rsrv_tbl.repl_rsrv_df.copy()
        #-------------------------
        DepreciationTable.output_results_to_excel_worksheet(
            df            = df, 
            writer        = writer, 
            sheet_name    = sheet_name, 
            tot_row       = None, 
            col_max_len   = col_max_len, 
            rows_to_shade = None,
            startrow      = startrow, 
            startcol      = startcol
        )
        startrow += df.shape[0]  
        if return_startrow:
            return startrow
            
    def output_sale_tax_and_proceeds_to_excel_worksheet(
        self, 
        writer, 
        sheet_name      = 'Sheet1', 
        col_max_len     = None, 
        startrow        = 0, 
        startcol        = 0, 
        return_startrow = False
    ):
        r"""
        """
        #--------------------------------------------------
        df = self.get_nsp_cal_srs(return_full=True).to_frame(name = 'Sale Tax and Proceeds').copy()
        #-------------------------
        DepreciationTable.output_results_to_excel_worksheet(
            df            = df, 
            writer        = writer, 
            sheet_name    = sheet_name, 
            tot_row       = None, 
            col_max_len   = col_max_len, 
            rows_to_shade = [2, 4, 9, 11, 20, 27],
            startrow      = startrow, 
            startcol      = startcol
        )
        startrow += df.shape[0]
        if return_startrow:
            return startrow
            
    def output_all_depr_amort_susptaxloss_replrsrv_sale_tables_to_excel_worksheet(
        self, 
        writer, 
        sheet_name    = 'Sheet1', 
        col_max_len   = None, 
        startrow      = 0, 
        startcol      = 1, 
        return_startrow = False
    ):
        r"""
        """
        #--------------------------------------------------
        startrow = self.output_all_depr_amort_tables_to_excel_worksheet(
            writer          = writer, 
            sheet_name      = sheet_name, 
            col_max_len     = col_max_len, 
            startrow        = startrow, 
            startcol        = startcol+1, 
            return_startrow = True
        )
        startrow += 5

        #--------------------------------------------------
        startrow = self.output_susp_tax_loss_to_excel_worksheet(
            writer          = writer, 
            sheet_name      = sheet_name, 
            col_max_len     = col_max_len, 
            startrow        = startrow, 
            startcol        = startcol+1, 
            return_startrow = True
        )
        startrow += 5    

        #--------------------------------------------------
        startrow = self.output_repl_rsrv_tbl_to_excel_worksheet(
            writer          = writer, 
            sheet_name      = sheet_name, 
            col_max_len     = col_max_len, 
            startrow        = startrow, 
            startcol        = startcol+1, 
            return_startrow = True
        )
        startrow += 5
        
        #--------------------------------------------------
        startrow = self.output_sale_tax_and_proceeds_to_excel_worksheet(
            writer          = writer, 
            sheet_name      = sheet_name, 
            col_max_len     = col_max_len, 
            startrow        = startrow, 
            startcol        = startcol, 
            return_startrow = True
        )

        if return_startrow:
            return startrow