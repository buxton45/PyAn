#!/usr/bin/env python

r"""
Holds depreciator and simple amortization classes.  See individual classes for more info
"""

__author__ = "Jesse Buxton"
__email__  = "buxton.45.jb@gmail.com"
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


#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
#Depreciation class
class AnnualDepreciator:
    def __init__(
        self, 
        acq_cost, 
        life_n_years,
        salv_val, 
        year_0   = 1, 
        schedule = None, 
        name     = None
    ):
        r"""
        
        schedule:
            Can supply a schedule, or can use keywords such as 'straight-line'
            If the schedule is supplied, the values must be between 0 and 1 and are taken with respect to the
              total depreciable value (acq_cost-salv_val), NOT the remaining value (i.e., NOT the book value)
        """
        #--------------------------------------------------
        #-------------------------
        self.acq_cost     = acq_cost
        self.life_n_years = life_n_years
        self.salv_val     = salv_val
        self.name         = name

        assert(isinstance(year_0, int))
        self.year_0   = year_0

        #--------------------------------------------------
        self.depr_df = None
        self.depr_schedule = None
        #-----
        assert(
            schedule is None or 
            Utilities.is_object_one_of_types(schedule, [str, float, list, tuple])
        )
        if isinstance(schedule, str):
            assert(schedule in ['sl', 'ddb'])
        #-------------------------
        if schedule is None or schedule == 'sl':
            # Straight-line depreciation method used
            self.rates = [1.0/self.life_n_years]*self.life_n_years
            self.depr_df = AnnualDepreciator.build_simple_depr_df(
                acq_cost     = self.acq_cost, 
                life_n_years = self.life_n_years, 
                salv_val     = self.salv_val, 
                rates        = self.rates
            )
        elif isinstance(schedule, str):
            assert(schedule=='ddb') # 'sl' case should have been caught above
            self.depr_df = AnnualDepreciator.double_declining_balance_depreciation(
                acq_cost     = self.acq_cost, 
                life_n_years = self.life_n_years, 
                salv_val     = self.salv_val
            )
            self.depr_schedule = self.depr_df['depr'].tolist()
            self.rates = list(np.divide(self.depr_schedule, self.acq_cost-self.salv_val))
        #-------------------------
        elif isinstance(schedule, float):
            assert(schedule>=0 and schedule<=1)
            self.rates = [schedule]*self.life_n_years
            self.depr_df = AnnualDepreciator.build_simple_depr_df(
                acq_cost     = self.acq_cost, 
                life_n_years = self.life_n_years, 
                salv_val     = self.salv_val, 
                rates        = self.rates
            )
        #-------------------------
        elif Utilities.is_object_one_of_types(schedule, [list, tuple]):
            self.rates = AnnualDepreciator.complete_rates(
                acq_cost     = self.acq_cost, 
                life_n_years = self.life_n_years,
                salv_val     = self.salv_val,  
                schedule     = schedule
            )
            self.depr_df = AnnualDepreciator.build_simple_depr_df(
                acq_cost     = self.acq_cost, 
                life_n_years = self.life_n_years, 
                salv_val     = self.salv_val, 
                rates        = self.rates
            )
        #-------------------------
        else:
            assert(0)
        #-------------------------
        self.depr_df['year'] = list(range(self.year_0, self.depr_df.shape[0]+self.year_0))
        self.depr_df = Utilities_df.move_cols_to_front(self.depr_df, ['year'])
            
        # If self.depr_schedule not already set, set
        self.depr_schedule = self.depr_df['depr'].tolist()
        
        if acq_cost==salv_val==0:
            return
        
        # Make sure self.rates (which should be set in all cases) is consistent with what one would
        #   expect from self.depr_schedule
        # HOWEVER, I suppose if depreciation finishes before life_n_years, then the lengths will be unequal
        rates_expctd = list(np.divide(self.depr_schedule, self.acq_cost-self.salv_val))
        assert(len(rates_expctd)>=len(self.rates))
        rates_expctd  = rates_expctd[:len(self.rates)]
        assert(np.all([Utilities.are_approx_equal(self.rates[i], rates_expctd[i]) for i in range(len(self.rates))]))
        

    @property
    def depr_srs(self):
        return self.depr_df.set_index('year')['depr'].copy()
    
    def get_srs(self, col):
        assert(col in self.depr_df.set_index('year').columns)
        return self.depr_df.set_index('year')[col].copy()
        
    @staticmethod
    def double_declining_balance_depreciation(
        acq_cost, 
        life_n_years, 
        salv_val
    ):
        r"""
        Everything I can find suggests the correct method is to NOT include the salvage value into the calculation.
        That makes this method even more advantageous, as you essentially double the rate with respect to the acquistion
          cost instead of the total depreciable amount (acq_cost-salv_val)
        """
        #-------------------------
        # This method requires a non-zero salvage value, otherwise the procedure is an infinite loop!
        assert(salv_val>0)

        #-------------------------
        # Since the salvage value is not included in the calculation, the rate is simply 1/life_n_years
        rate_sl_nosalv = 1.0/life_n_years
        rate = 2*rate_sl_nosalv

        #--------------------------------------------------
        depr_df = pd.DataFrame(
            columns=['book_beg', 'rate', 'depr', 'cum_depr', 'book_end'], 
            dtype=float, 
        )
        #-------------------------
        end_val_i = acq_cost
        count=0
        while end_val_i>salv_val:
            #-------------------------
            count+=1
            if count > 1000:
                print('ERROR in double_declining_balance_depreciation!!!!!')
                assert(0)
            #-------------------------
            beg_val_i = end_val_i
            depr_i    = beg_val_i*rate
            end_val_i = beg_val_i - depr_i
            #-----
            if depr_df.shape[0]==0:
                cum_depr_i = depr_i
            else:
                cum_depr_i = depr_df.iloc[-1]['cum_depr'] + depr_i
            entry_i = pd.Series(dict(
                book_beg = beg_val_i, 
                rate     = rate, 
                depr     = depr_i,
                cum_depr = cum_depr_i, 
                book_end = end_val_i
            )).to_frame(name=depr_df.shape[0]+1).T
            depr_df = pd.concat([depr_df, entry_i])
        #------------------------- 
        # Clean up the last entry, as we will have typically depreciated slightly past the salvage value
        extra_paid = salv_val-depr_df.iloc[-1]['book_end']
        assert(extra_paid>=0)
        depr_df.iloc[-1]['depr']    -= extra_paid
        depr_df.iloc[-1]['cum_depr'] = depr_df.iloc[-2]['cum_depr'] + depr_df.iloc[-1]['depr']
        depr_df.iloc[-1]['rate']     = depr_df.iloc[-1]['depr']/depr_df.iloc[-1]['book_beg']
        depr_df.iloc[-1]['book_end'] = depr_df.iloc[-1]['book_beg']-depr_df.iloc[-1]['depr']
        assert(Utilities.are_approx_equal(depr_df.iloc[-1]['book_end'], salv_val))
        assert(Utilities.are_approx_equal(depr_df.iloc[-1]['cum_depr'], acq_cost-salv_val))
        #------------------------- 
        return depr_df
    
    
    @staticmethod
    def complete_rates(
        acq_cost, 
        life_n_years,
        salv_val,  
        schedule
    ):
        r"""
        Only meant for use with list-type schedule objects, and designed for use
          in constructor of AnnualDepreciation class
        """
        #-------------------------
        assert(acq_cost>0)
        assert(salv_val>=0 and salv_val<acq_cost)
        #-------------------------
        approp_descs = ['constant', 'year1']
        #-------------------------
        assert(Utilities.is_object_one_of_types(schedule, [list, tuple]))
        if isinstance(schedule, tuple):
            assert(len(schedule)==2)
            val  = schedule[0]
            desc = schedule[1]
            #-----
            assert(isinstance(val, float))
            assert(val>=0 and val<=1)
            #-----
            assert(desc in approp_descs)
            #-------------------------
            if desc=='constant':
                rates = [val]*life_n_years
            #-------------------------
            elif desc=='year1':
                rates = [val]
                remaining_pct = 1.0 - val - salv_val/acq_cost
                rate_i = remaining_pct/(life_n_years-1)
                rates.extend([rate_i]*(life_n_years-1))
            #-------------------------
            else:
                assert(0)
        else:
            assert(np.all([x>=0 and x<=1 for x in schedule]))
            if len(schedule)==life_n_years:
                rates = schedule
            elif len(schedule)>life_n_years:
                print(f'Warning: In AnnualDepreciator.complete_rates: More rates supplied ({len(schedule)}) than needed ({life_n_years})')
                print(f'Only using the first {life_n_years}')
                rates = schedule[:life_n_years]
            else:
                print(f'Warning: In AnnualDepreciator.complete_rates: Less rates supplied ({len(schedule)}) than needed ({life_n_years})')
                print(f'Repeating last rate to complete the set')
                end_rates = [rates[-1]]*(life_n_years-len(schedule))
                rates.extend(end_rates)
                assert(len(rates)==life_n_years)
        #-------------------------
        return rates
    
    
    @staticmethod
    def build_simple_depr_df(
        acq_cost, 
        life_n_years, 
        salv_val, 
        rates
    ):
        r"""
        As stated in documentation, rates are always with respect to the total depreciable value (acq_cost-salv_val), 
          NOT the remaining value (i.e., NOT the book value)
        """
        #-------------------------
        assert(acq_cost>=0)
        assert(salv_val>=0 and (salv_val<acq_cost or salv_val==acq_cost==0))
        assert(len(rates)==life_n_years)
        #--------------------------------------------------
        if salv_val==acq_cost==0:
            depr_df = pd.DataFrame(
                columns = ['book_beg', 'rate', 'depr', 'cum_depr', 'book_end', 'fully_depr'], 
                index   = list(range(1, life_n_years+1)), 
                data    = 0, 
                dtype   = float, 
            )
            depr_df.iloc[-1]['fully_depr'] = True
            return depr_df
        #-------------------------
        depr_df = pd.DataFrame(
            columns=['book_beg', 'rate', 'depr', 'cum_depr', 'book_end', 'fully_depr'], 
            dtype=float, 
        )
        #-------------------------
        tot_depr_val = acq_cost - salv_val
        #-------------------------
        end_val_i = acq_cost
        for rate_i in rates:
            #----------
            if end_val_i < salv_val:
                break
            #----------
            beg_val_i = end_val_i
            depr_i    = tot_depr_val*rate_i
            end_val_i = beg_val_i - depr_i
            #-----
            if depr_df.shape[0]==0:
                cum_depr_i = depr_i
            else:
                cum_depr_i = depr_df.iloc[-1]['cum_depr'] + depr_i
            #-----
            entry_i = pd.Series(dict(
                book_beg   = beg_val_i, 
                rate       = rate_i, 
                depr       = depr_i,
                cum_depr   = cum_depr_i, 
                book_end   = end_val_i, 
                fully_depr = False
            )).to_frame(name=depr_df.shape[0]+1).T
            depr_df = pd.concat([depr_df, entry_i])
        #------------------------- 
        # Clean up the last entry, as we will have typically depreciated slightly past the salvage value
        extra_paid = salv_val-depr_df.iloc[-1]['book_end']
        if extra_paid > 0:
            depr_df.iloc[-1]['depr']      -= extra_paid
            depr_df.iloc[-1]['cum_depr']   = depr_df.iloc[-2]['cum_depr'] + depr_df.iloc[-1]['depr']
            depr_df.iloc[-1]['rate']       = depr_df.iloc[-1]['depr']/depr_df.iloc[-1]['book_beg']
            depr_df.iloc[-1]['book_end']   = depr_df.iloc[-1]['book_beg']-depr_df.iloc[-1]['depr']
            depr_df.iloc[-1]['fully_depr'] = True
            assert(Utilities.are_approx_equal(depr_df.iloc[-1]['book_end'], salv_val))
            assert(Utilities.are_approx_equal(depr_df.iloc[-1]['cum_depr'], acq_cost-salv_val))
        elif extra_paid==0:
            depr_df.iloc[-1]['fully_depr'] = True
        else:
            pass
        #------------------------- 
        return depr_df
    
    @staticmethod
    def extend_depr_df(
        depr_df, 
        n_needed
    ):
        r"""
        """
        #-------------------------
        new_entry_i = pd.DataFrame(
            data  = dict(
                year       = depr_df.iloc[-1]['year']+1, 
                book_beg   = depr_df.iloc[-1]['book_end'], 
                rate       = np.nan, 
                depr       = 0, 
                cum_depr   = depr_df.iloc[-1]['cum_depr'], 
                book_end   = depr_df.iloc[-1]['book_end'], 
                fully_depr = depr_df.iloc[-1]['fully_depr']
            ), 
            index=[depr_df.index[-1]+1], 
            dtype = float
        )
        #-------------------------
        to_appnd = pd.concat([new_entry_i]*n_needed)
        #-----
        to_appnd['year'] = range(int(to_appnd.iloc[0]['year']), int(to_appnd.iloc[0]['year']+n_needed))
        #-------------------------
        assert(depr_df.columns.tolist()   == to_appnd.columns.tolist())
        assert(depr_df.iloc[-1]['year']+1 == to_appnd.iloc[0]['year'])
        #-----
        depr_df = pd.concat([depr_df, to_appnd])
        #-------------------------
        return depr_df


    @staticmethod
    def extend_depr_df_for_years_needed(
        depr_df, 
        years_needed
    ):
        r"""
        """
        #-------------------------
        new_entry_aftr = pd.DataFrame(
            data  = dict(
                year       = depr_df.iloc[-1]['year']+1, 
                book_beg   = depr_df.iloc[-1]['book_end'], 
                rate       = np.nan, 
                depr       = 0, 
                cum_depr   = depr_df.iloc[-1]['cum_depr'], 
                book_end   = depr_df.iloc[-1]['book_end'], 
                fully_depr = depr_df.iloc[-1]['fully_depr']
            ), 
            index=[depr_df.index[-1]+1], 
            dtype = float
        )
        #-----
        new_entry_bfor = pd.DataFrame(
            data  = dict(
                year       = depr_df.iloc[0]['year']-1, 
                book_beg   = 0, 
                rate       = np.nan, 
                depr       = 0, 
                cum_depr   = 0, 
                book_end   = 0, 
                fully_depr = False
            ), 
            index=[depr_df.index[0]-1], 
            dtype = float
        )
        #-------------------------
        years_needed = natsorted(years_needed)
        years_needed_bfor = natsorted([x for x in years_needed if x < depr_df.iloc[0]['year']-1])
        years_needed_aftr = natsorted([x for x in years_needed if x > depr_df.iloc[-1]['year']+1])
        #-------------------------
        if len(years_needed_bfor)>0:
            to_appnd_bfor = pd.concat([new_entry_bfor]*len(years_needed_bfor))
            to_appnd_bfor['year'] = range(int(to_appnd_bfor.iloc[-1]['year']+1-len(years_needed_bfor)), int(to_appnd_bfor.iloc[-1]['year']+1))
            #-----
            assert(depr_df.columns.tolist() == to_appnd_bfor.columns.tolist())
            assert(to_appnd_bfor.iloc[-1]['year']+1 == depr_df.iloc[0]['year'])
            #-----
            depr_df = pd.concat([to_appnd_bfor, depr_df])
        if len(years_needed_aftr)>0:
            to_appnd_aftr = pd.concat([new_entry_aftr]*len(years_needed_aftr))
            to_appnd_aftr['year'] = range(int(to_appnd_aftr.iloc[0]['year']), int(to_appnd_aftr.iloc[0]['year']+len(years_needed_aftr)))
            #-----
            assert(depr_df.columns.tolist()    == to_appnd_aftr.columns.tolist())
            assert(depr_df.iloc[-1]['year']+1  == to_appnd_aftr.iloc[0]['year'])
            #-----
            depr_df = pd.concat([depr_df, to_appnd_aftr])
        #-------------------------
        return depr_df
    
    
    def extend_for_years_needed(
        self, 
        years_needed
    ):
        r"""
        """
        #-------------------------
        depr_df = self.depr_df.copy()
        depr_df = AnnualDepreciator.extend_depr_df_for_years_needed(
            depr_df      = depr_df, 
            years_needed = years_needed
        )
        self.depr_df = depr_df
        
        
        
#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
#Depreciation class
class DepreciationTable:
    def __init__(
        self, 
        ads_list, 
        hold_years=-1, 
        exclude_year_in_idx  = False, 
        exclude_spend_in_idx = False, 
        name                 = None
    ):
        r"""
        
        schedule:
            Can supply a schedule, or can use keywords such as 'straight-line'
            If the schedule is supplied, the values must be between 0 and 1 and are taken with respect to the
              total depreciable value (acq_cost-salv_val), NOT the remaining value (i.e., NOT the book value)
        """
        #--------------------------------------------------
        assert(isinstance(hold_years, int) and (hold_years==-1 or hold_years>0))
        #----------
        if ads_list is None:
            ads_list = []
        assert(
            Utilities.is_object_one_of_types(ads_list, [list, tuple]) and 
            Utilities.are_all_list_elements_one_of_types_and_homogeneous(ads_list, [AnnualDepreciator, SimpleAmortizer])
        )
        #--------------------------------------------------
        self.ads_list             = copy.deepcopy(ads_list)
        self.hold_years           = hold_years
        self.exclude_year_in_idx  = exclude_year_in_idx
        self.exclude_spend_in_idx = exclude_spend_in_idx
        self.name                 = name
        if len(self.ads_list)>0:
            self.depr_tbl_df = DepreciationTable.build_depr_table(
                ads_list             = self.ads_list, 
                hold_years           = self.hold_years, 
                exclude_year_in_idx  = self.exclude_year_in_idx,
                exclude_spend_in_idx = self.exclude_spend_in_idx
            )
            
    @property
    def book_end_tbl_df(self):
        return self.build_table(col='book_end')

    @property
    def cum_depr_tbl_df(self):
        return self.build_table(col='cum_depr')
        
    @staticmethod
    def build_depr_table(
        ads_list, 
        hold_years=-1, 
        exclude_year_in_idx   = False, 
        exclude_spend_in_idx  = False
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(hold_years, int) and (hold_years==-1 or hold_years>0))
        #-------------------------
        year_0_min   = np.min([x.depr_df['year'].min() for x in ads_list])
        timeline_max = np.max([x.depr_df['year'].max() for x in ads_list])
        years_needed = list(range(year_0_min, timeline_max+1))
        #-------------------------
        depr_srs_list = []
        for ad_i in ads_list:
            ad_i.extend_for_years_needed(years_needed = years_needed)
            depr_srs_i = ad_i.depr_srs
            years_needed_i = list(set(years_needed).difference(set(depr_srs_i.index)))
            #-----
            depr_srs_i = pd.concat([depr_srs_i, pd.Series([0]*len(years_needed_i), index=years_needed_i, dtype=float)]).sort_index()
            #----------
            if ad_i.name:
                depr_srs_i.name = f"{ad_i.name}"
            else:
                depr_srs_i.name = f"Year"
            #-----
            if not exclude_year_in_idx:
                depr_srs_i.name = depr_srs_i.name + f" {ad_i.year_0}"
            if not exclude_spend_in_idx:
                depr_srs_i.name = depr_srs_i.name + ": Spend"
            #----------
            depr_srs_list.append(depr_srs_i)
        #-------------------------
        tmp_idx = depr_srs_list[0].index
        for depr_srs_i in depr_srs_list:
            assert((depr_srs_i.index==tmp_idx).all())
        #-----
        depr_df_fnl = pd.concat(depr_srs_list, axis=1)
        depr_df_fnl['Total'] = depr_df_fnl.sum(axis=1)
        if hold_years>0:
            depr_df_fnl.loc[depr_df_fnl.index>hold_years, 'Total'] = 0
        #-----
        depr_df_fnl = depr_df_fnl.T
        #-------------------------
        return depr_df_fnl
    
    def build_table(
        self, 
        col
    ):
        r"""
        """
        #-------------------------
        ads_list  = self.ads_list
        hold_years = self.hold_years
        #-------------------------
        assert(isinstance(hold_years, int) and (hold_years==-1 or hold_years>0))
        #-------------------------
        year_0_min   = np.min([x.depr_df['year'].min() for x in ads_list])
        timeline_max = np.max([x.depr_df['year'].max() for x in ads_list])
        years_needed = list(range(year_0_min, timeline_max+1))
        #-------------------------
        srs_list = []
        for ad_i in ads_list:
            srs_i = ad_i.get_srs(col=col)
            years_needed_i = list(set(years_needed).difference(set(srs_i.index)))
            #-----
            srs_i = pd.concat([srs_i, pd.Series([0]*len(years_needed_i), index=years_needed_i, dtype=float)]).sort_index()
            #----------
            if ad_i.name:
                srs_i.name = f"{ad_i.name}"
            else:
                srs_i.name = f"Year"
            #-----
            if not self.exclude_year_in_idx:
                srs_i.name = srs_i.name + f" {ad_i.year_0}"
            if not self.exclude_spend_in_idx:
                srs_i.name = srs_i.name + ": Spend"
            #----------
            srs_list.append(srs_i)
        #-------------------------
        tmp_idx = srs_list[0].index
        for srs_i in srs_list:
            assert((srs_i.index==tmp_idx).all())
        #-----
        df_fnl = pd.concat(srs_list, axis=1)
        df_fnl['Total'] = df_fnl.sum(axis=1)
        if hold_years>0:
            df_fnl.loc[df_fnl.index>hold_years, 'Total'] = 0
        #-----
        df_fnl = df_fnl.T
        #-------------------------
        return df_fnl
    
    
    def add_ad(
        self, 
        ad_i
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(ad_i, AnnualDepreciator) or isinstance(ad_i, SimpleAmortizer))
        if len(self.ads_list)>0:
            assert(type(self.ads_list[0])==type(ad_i))
        self.ads_list.append(ad_i)
        #----------
        self.depr_tbl_df = DepreciationTable.build_depr_table(
            ads_list  = self.ads_list, 
            hold_years = self.hold_years
        )

    def get_total_depr_for_year_i(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        depr_tbl_T = self.depr_tbl_df.T
        assert(year_i in depr_tbl_T.index)
        assert('Total' in depr_tbl_T.columns)
        prop_depr_i = depr_tbl_T.loc[year_i, 'Total']
        #-------------------------
        return prop_depr_i
        
        
    def get_depr_tbl_df_to_output(
        self
    ):
        r"""
        """
        #-------------------------
        depr_tbl_df = self.depr_tbl_df.copy()
        depr_tbl_df.columns = [f"Year {x}" for x in depr_tbl_df.columns.tolist()]
        return depr_tbl_df
        
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
        idx_width = np.max([len(str(x)) for x in df.index.tolist()])
        _ = worksheet.set_column(0, max_col, idx_width)
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
        
    def output_depr_tbl_to_excel_worksheet(
        self, 
        writer, 
        sheet_name      = 'Sheet1', 
        col_max_len     = None, 
        rows_to_shade   = None,
        startrow        = 0, 
        startcol        = 0, 
        return_startrow = False
    ):
        r"""
        """
        #-------------------------
        df = self.get_depr_tbl_df_to_output()
        if self.name is not None:
            df.index.name = self.name
        #-------------------------
        DepreciationTable.output_results_to_excel_worksheet(
            df            = df, 
            writer        = writer, 
            sheet_name    = sheet_name, 
            tot_row       = df.shape[0]+startrow, 
            col_max_len   = col_max_len, 
            rows_to_shade = rows_to_shade,
            startrow      = startrow, 
            startcol      = startcol
        )
        startrow += df.shape[0]
        if return_startrow:
            return startrow
        
    def output_depr_tbl_to_excel(
        self, 
        path,  
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
        self.output_depr_tbl_to_excel_worksheet(
                writer        = writer, 
                sheet_name    = 'Sheet1', 
                col_max_len   = col_max_len, 
                rows_to_shade = rows_to_shade, 
                startrow      = startrow, 
                startcol      = startcol
            )
        #-------------------------
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()


#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
class PropertyDepreciationTable(DepreciationTable):
    def __init__(
        self, 
        purchase_price, 
        pct_land, 
        pct_strctr, 
        hold_years, 
        x_year_items_dict = None, 
        n_years_strctr    = 39, 
    ):
        r"""
        x_year_items_dict:
            A dict whose keys equal the number of depreciation years and values equal allocation percentages
            e.g., for 20% 7-year items and 10% 3-year items:
                x_year_items_dict = {
                    7:0.2, 
                    3:0.1
                }
        """
        #-------------------------
        if x_year_items_dict is None:
            x_year_items_dict={}
        else:
            assert(isinstance(x_year_items_dict, dict))
            assert(np.all([isinstance(x, int) for x in x_year_items_dict.keys()]))
        #-------------------------
        # All of the allocation percentages must sum to 1!
        assert(Utilities.are_approx_equal(pct_strctr + pct_land + np.sum(list(x_year_items_dict.values())), 1))
        #-------------------------
        self.purchase_price    = purchase_price
        self.pct_land          = pct_land
        self.pct_strctr        = pct_strctr
        self.hold_years        = hold_years #Not needed bc set by super, but didn't want to confuse future self with exclusion
        self.x_year_items_dict = x_year_items_dict
        self.n_years_strctr    = n_years_strctr
        #-------------------------
        ads_list = []
        ads_list.append(
            AnnualDepreciator(
                acq_cost     = 0, 
                life_n_years = hold_years, 
                salv_val     = 0, 
                year_0       = 1, 
                schedule     = None, 
                name         = 'Land'
            )
        )
        ads_list.append(
            AnnualDepreciator(
                acq_cost     = pct_strctr*purchase_price, 
                life_n_years = n_years_strctr, 
                salv_val     = 0, 
                year_0       = 1, 
                schedule     = 'sl', 
                name         = 'Structure'
            )
        )
        for life_i, pct_i in x_year_items_dict.items():
            ads_list.append(
                AnnualDepreciator(
                    acq_cost     = pct_i*purchase_price, 
                    life_n_years = life_i, 
                    salv_val     = 0, 
                    year_0       = 1, 
                    schedule     = 'sl', 
                    name         = f"{life_i}-year items"
                )
            )
        
        super().__init__(
            ads_list             = ads_list, 
            hold_years           = hold_years, 
            exclude_year_in_idx  = True, 
            exclude_spend_in_idx = True, 
            name                 = None
        )

    def get_prop_depr_for_year_i(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        depr_tbl_T = self.depr_tbl_df.T
        assert(year_i in depr_tbl_T.index)
        assert('Total' in depr_tbl_T.columns)
        prop_depr_i = depr_tbl_T.loc[year_i, 'Total']
        #-------------------------
        return prop_depr_i
        
        
    def get_pct_allocs_tbl(
        self
    ):
        r"""
        """
        #-------------------------
        pct_allocs_tbl = pd.Series(
            {
                'Purchase Price':                           self.purchase_price, 
                'Percentage Allocations':                   '', 
                'Land':                                     self.pct_land, 
                f'Structure ({self.n_years_strctr} years)': self.pct_strctr, 
            } 
            |
            {
                f"{k}-year items":v for k,v in self.x_year_items_dict.items()
            }
        ).to_frame(name='Purchase Information')
        #-------------------------
        return pct_allocs_tbl
        
    def output_pct_allocs_and_depr_tbl_to_excel_worksheet(
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
        df = self.get_pct_allocs_tbl()
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
        startrow += df.shape[0] + 2

        #--------------------------------------------------
        df = self.get_depr_tbl_df_to_output()
        #-------------------------
        DepreciationTable.output_results_to_excel_worksheet(
            df            = df, 
            writer        = writer, 
            sheet_name    = sheet_name, 
            tot_row       = df.shape[0]+startrow, 
            col_max_len   = col_max_len, 
            rows_to_shade = None,
            startrow      = startrow, 
            startcol      = startcol
        )
        startrow += df.shape[0]
        if return_startrow:
            return startrow
        
        
    def output_pct_allocs_and_depr_tbl_to_excel(
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
        self.output_pct_allocs_and_depr_tbl_to_excel_worksheet(
                writer        = writer, 
                sheet_name    = 'Sheet1', 
                col_max_len   = col_max_len, 
                startrow      = startrow, 
                startcol      = startcol
            )
        #-------------------------
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()



#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
class SimpleAmortizer(AnnualDepreciator):
    def __init__(
        self, 
        val_0, 
        life_n_years, 
        year_0   = 1, 
        schedule = None, 
        name     = None
    ):
        r"""
        """
        #-------------------------
        self.val_0        = val_0
        self.life_n_years = life_n_years 
        self.year_0       = year_0 
        self.schedule     = schedule 
        self.name         = name
        #-------------------------
        super().__init__(
            acq_cost     = self.val_0, 
            life_n_years = self.life_n_years,
            salv_val     = 0, 
            year_0       = self.year_0, 
            schedule     = self.schedule, 
            name         = self.name
        )


#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------
class SimpleAmortTable(DepreciationTable):
    def __init__(
        self, 
        sas_list, 
        name
    ):
        r"""
        """
        #--------------------------------------------------
        if sas_list is None:
            sas_list = []
        assert(
            Utilities.is_object_one_of_types(sas_list, [list, tuple]) and 
            Utilities.are_all_list_elements_of_type(sas_list, SimpleAmortizer)
        )
        #-------------------------
        super().__init__(
            ads_list   = sas_list, 
            hold_years = -1, 
            name       = name
        )
        
    @property
    def sas_list(self):
        return self.ads_list
    
    def add_sa(
        self, 
        sa_i
    ):
        r"""
        """
        #-------------------------
        assert(isinstance(sa_i, SimpleAmortizer))
        self.add_ad(
            ad_i = sa_i
        )
        
    def get_still_unamortized_for_year_i(
        self, 
        year_i
    ):
        r"""
        """
        #-------------------------
        depr_tbl_T = self.depr_tbl_df.T
        assert(year_i in depr_tbl_T.index)
        assert('Still Unamortized at Year-end' in depr_tbl_T.columns)
        prop_depr_i = depr_tbl_T.loc[year_i, 'Still Unamortized at Year-end']
        #-------------------------
        return prop_depr_i


#------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------