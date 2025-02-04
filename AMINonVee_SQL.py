#!/usr/bin/env python

r"""
Holds AMINonVee_SQL class.  See AMINonVee_SQL.AMINonVee_SQL for more information.
"""

__author__ = "Jesse Buxton"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys

#--------------------------------------------------
from AMI_SQL import AMI_SQL, DfToSqlMap
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
import Utilities_sql
import TableInfos
from SQLSelect import SQLSelectElement, SQLSelect
from SQLFrom import SQLFrom
from SQLWhere import SQLWhereElement, SQLWhere
from SQLHaving import SQLHaving
from SQLQuery import SQLQuery
from SQLQueryGeneric import SQLQueryGeneric
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_utilities_dir())
import Utilities
#--------------------------------------------------

class AMINonVee_SQL(AMI_SQL):
    def __init__(self):
        self.sql_query = None

    
    #****************************************************************************************************
    @staticmethod
    def is_appropriate_aep_derived_uom(input_str, possible_aep_derived_uoms=None):
        #--------------------------------------------------
        possible_aep_derived_uoms_and_aep_srvc_qlty_idntfrs = {
            'AMP'   : ['INSTCURA1', 'INSTCURC1'], 
            'KVAH'  : ['DELIVERED'], 
            'KVARH' : ['LAG', 'LEAD', 'NET'], 
            'KWH'   : ['DELIVERED', 'RECEIVED', 'TOTAL'], 
            'UNK'   : ['UNK'], 
            'VOLT'  : ['AVG', 'INSTVA1', 'INSTVB1', 'INSTVC1']
        }
        #--------------------------------------------------
        if possible_aep_derived_uoms is None:
            possible_aep_derived_uoms = list(possible_aep_derived_uoms_and_aep_srvc_qlty_idntfrs.keys())
        #--------------------------------------------------  
        if input_str in possible_aep_derived_uoms:
            return True
        else:
            return False
    
    #****************************************************************************************************
    @staticmethod
    def is_appropriate_uom_and_srvc_qlty_idntfr_pair(
        derived_uom         , 
        srvc_qlty_idntfr    = None, 
        possible_pairs_dict = None
    ):
        #--------------------------------------------------
        possible_aep_derived_uoms_and_aep_srvc_qlty_idntfrs = {
            'AMP'   : ['INSTCURA1', 'INSTCURC1'], 
            'KVAH'  : ['DELIVERED'], 
            'KVARH' : ['LAG', 'LEAD', 'NET'], 
            'KWH'   : ['DELIVERED', 'RECEIVED', 'TOTAL'], 
            'UNK'   : ['UNK'], 
            'VOLT'  : ['AVG', 'INSTVA1', 'INSTVB1', 'INSTVC1']
        }
        #--------------------------------------------------
        if possible_pairs_dict is None:
            possible_pairs_dict = possible_aep_derived_uoms_and_aep_srvc_qlty_idntfrs
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(derived_uom, [str, dict, SQLWhereElement]))
        if srvc_qlty_idntfr is not None:
            assert(Utilities.is_object_one_of_types(srvc_qlty_idntfr, [str, dict, SQLWhereElement]))
        #--------------------------------------------------
        derived_uom_where_el = SQLWhere.standardize_to_where_element(derived_uom, field_desc='aep_derived_uom')
        if derived_uom_where_el.value not in possible_pairs_dict:
            return False
        #----------
        if srvc_qlty_idntfr is not None:
            srvc_qlty_idntfr_where_el = SQLWhere.standardize_to_where_element(srvc_qlty_idntfr, field_desc='aep_aep_srvc_qlty_idntfr')
            if srvc_qlty_idntfr_where_el.value not in possible_pairs_dict[derived_uom_where_el.value]:
                return False
        return True
    
    #****************************************************************************************************
    @staticmethod
    def standardize_uom_and_idntfr(uom_and_maybe_idntfr):
        r"""
        Take a sloppy uom_and_maybe_idntfr (acceptable element types described below) and build a dict object
        with keys 'aep_derived_uom' and 'aep_srvc_qlty_idntfr' and values equal to SQLWhereElement objects (or, 
        possibly equal to None if aep_srvc_qlty_idntfr information not provided)

        - This function was designed for use within AMINonVee_SQL.standardize_aep_derived_uoms_and_srvc_qlty_idntfrs.
        - The purpose is to look at a given element, uom_and_maybe_idntfr, within aep_uoms_and_qlty_idntfrs
            and determine whether the object contains only aep_derived_uom, or both aep_derived_uom 
            and aep_aep_srvc_qlty_idntfr.
        - From the AMINonVee_SQL.standardize_aep_derived_uoms_and_srvc_qlty_idntfrs documentation:
            - uom_and_maybe_idntfr may therefore take on the form:
                - uom_and_maybe_idntfr = uom_i = str, dict, or SQLWhereElement
                    ==> is_idntfr_with_uom = False
                - uom_and_maybe_idntfr = [uom_i, idntfr_i]
                                       = [(str, dict, or SQLWhereElement), (str, dict, or SQLWhereElement)]
                    ==> is_idntfr_with_uom = True
                - uom_and_maybe_idntfr = dict(aep_derived_uom=uom_i, aep_srvc_qlty_idntfr=idntfr_i)
                                       = dict(aep_derived_uom      = (str, dict, or SQLWhereElement), 
                                              aep_srvc_qlty_idntfr = (str, dict, or SQLWhereElement))
                    ==> is_idntfr_with_uom = True
        - The function is designed also to fail if the contents of uom_and_maybe_idntfr don't appear to be
            aep_derived_uom/aep_aep_srvc_qlty_idntfr.    

        """
        assert(
            isinstance(uom_and_maybe_idntfr, str) or 
            isinstance(uom_and_maybe_idntfr, dict) or
            isinstance(uom_and_maybe_idntfr, SQLWhereElement) or 
            isinstance(uom_and_maybe_idntfr, list) or 
            isinstance(uom_and_maybe_idntfr, tuple)
        )
        #--------------------------------------------------
        # If uom_and_maybe_idntfr is str or SQLWhereElement, then only aep_derived_uom information input.
        # ==> is_idntfr_with_uom=False
        if isinstance(uom_and_maybe_idntfr, str) or isinstance(uom_and_maybe_idntfr, SQLWhereElement):
            is_idntfr_with_uom=False
            # If uom_and_maybe_idntfr is a str, convert it to SQLWhereElement
            if isinstance(uom_and_maybe_idntfr, str):
                if not AMINonVee_SQL.is_appropriate_aep_derived_uom(uom_and_maybe_idntfr):
                    print(f'In is_idntfr_with_uom, uom_and_maybe_idntfr={uom_and_maybe_idntfr}')
                    print('FAILED AMINonVee_SQL.is_appropriate_aep_derived_uom(uom_and_maybe_idntfr).\nCRASH IMMINENT')
                    assert(0)
                aep_derived_uom_where_el = SQLWhere.standardize_to_where_element(uom_and_maybe_idntfr, field_desc='aep_derived_uom')
                return_dict = dict(
                    aep_derived_uom      = aep_derived_uom_where_el, 
                    aep_srvc_qlty_idntfr = None
                )
            else:
                assert(
                    AMINonVee_SQL.is_appropriate_uom_and_srvc_qlty_idntfr_pair(
                        derived_uom      = uom_and_maybe_idntfr, 
                        srvc_qlty_idntfr = None
                    )
                )
                return_dict = dict(
                    aep_derived_uom      = uom_and_maybe_idntfr, 
                    aep_srvc_qlty_idntfr = None
                )
        #--------------------------------------------------
        # If uom_and_maybe_idntfr is a dict, it is most likely only aep_derived_uom information input.
        # The only way aep_derived_uom and aep_srvc_qlty_idntfr are both contained is if uom_and_maybe_idntfr
        #   has only two keys equal to 'aep_derived_uom' and 'aep_srvc_qlty_idntfr'
        elif isinstance(uom_and_maybe_idntfr, dict):
            if (len(uom_and_maybe_idntfr)==2 and 
                'aep_derived_uom' in uom_and_maybe_idntfr and 
                'aep_srvc_qlty_idntfr' in uom_and_maybe_idntfr):
                # TODO how to ensure these are appropriate?
                is_idntfr_with_uom=True
                #-----
                try:
                    aep_derived_uom_where_el = SQLWhere.standardize_to_where_element(
                        uom_and_maybe_idntfr['aep_derived_uom'], 
                        field_desc = 'aep_derived_uom'
                    )
                    if uom_and_maybe_idntfr['aep_srvc_qlty_idntfr'] is not None:
                        aep_srvc_qlty_idntfr_where_el = SQLWhere.standardize_to_where_element(
                            uom_and_maybe_idntfr['aep_srvc_qlty_idntfr'], 
                            field_desc = 'aep_srvc_qlty_idntfr'
                    )
                    else:
                        aep_srvc_qlty_idntfr_where_el = None
                except:
                    print(f'In is_idntfr_with_uom, uom_and_maybe_idntfr={uom_and_maybe_idntfr}')
                    print('FAILED SQLWhere.standardize_to_where_element on one of dict values.\nCRASH IMMINENT')
                    assert(0)
                #-----
                assert(
                    AMINonVee_SQL.is_appropriate_uom_and_srvc_qlty_idntfr_pair(
                        derived_uom      = aep_derived_uom_where_el, 
                        srvc_qlty_idntfr = aep_srvc_qlty_idntfr_where_el
                    )
                )
                return_dict = dict(
                    aep_derived_uom      = aep_derived_uom_where_el, 
                    aep_srvc_qlty_idntfr = aep_srvc_qlty_idntfr_where_el
                )
            else:
                is_idntfr_with_uom=False
                # Instead of just accepting that this is single at this point,
                #   try to build an SQLWhereElement to make sure everything is OK
                #-----
                try:
                    if 'field_desc' not in uom_and_maybe_idntfr:
                        uom_and_maybe_idntfr['field_desc'] = 'aep_derived_uom'
                    aep_derived_uom_where_el = SQLWhereElement(**uom_and_maybe_idntfr)
                except:
                    print(f'In is_idntfr_with_uom, uom_and_maybe_idntfr={uom_and_maybe_idntfr}')
                    print('FAILED SQLWhereElement(**uom_and_maybe_idntfr).\nCRASH IMMINENT')
                    assert(0)
                #-----
                assert(
                    AMINonVee_SQL.is_appropriate_uom_and_srvc_qlty_idntfr_pair(
                        derived_uom = aep_derived_uom_where_el, 
                        srvc_qlty_idntfr = None
                    )
                )
                return_dict = dict(
                    aep_derived_uom      = aep_derived_uom_where_el, 
                    aep_srvc_qlty_idntfr = None
                )
        #--------------------------------------------------
        # If uom_and_maybe_idntfr is a list or tuple (with length 2), then aep_derived_uom 
        #   and aep_srvc_qlty_idntfr are both contained
        elif isinstance(uom_and_maybe_idntfr, list) or isinstance(uom_and_maybe_idntfr, tuple):
            is_idntfr_with_uom=True
            if len(uom_and_maybe_idntfr)!=2:
                print(f'In is_idntfr_with_uom, uom_and_maybe_idntfr={uom_and_maybe_idntfr}')
                print('FAILED len(uom_and_maybe_idntfr)==2.\nCRASH IMMINENT')
                assert(0)
            #-----
            try:
                aep_derived_uom_where_el      = SQLWhere.standardize_to_where_element(
                    uom_and_maybe_idntfr[0], 
                    field_desc='aep_derived_uom'
                )
                aep_srvc_qlty_idntfr_where_el = SQLWhere.standardize_to_where_element(
                    uom_and_maybe_idntfr[1], 
                    field_desc='aep_srvc_qlty_idntfr'
                )
            except:
                print(f'In is_idntfr_with_uom, uom_and_maybe_idntfr={uom_and_maybe_idntfr}')
                print('FAILED SQLWhere.standardize_to_where_element on one of elements.\nCRASH IMMINENT')
                assert(0)
            #-----
            assert(
                AMINonVee_SQL.is_appropriate_uom_and_srvc_qlty_idntfr_pair(
                    derived_uom = aep_derived_uom_where_el, 
                    srvc_qlty_idntfr = aep_srvc_qlty_idntfr_where_el
                )
            )
            return_dict = dict(
                aep_derived_uom      = aep_derived_uom_where_el, 
                aep_srvc_qlty_idntfr = aep_srvc_qlty_idntfr_where_el
            )
        #--------------------------------------------------
        else:
            print('In is_idntfr_with_uom, uom_and_maybe_idntfr NOT OF CORRECT TYPE')
            print(f'type(uom_and_maybe_idntfr) = {type(uom_and_maybe_idntfr)}')
            assert(0)
        return return_dict
    
    #****************************************************************************************************
    @staticmethod
    def standardize_aep_derived_uoms_and_srvc_qlty_idntfrs(
        aep_uoms_and_qlty_idntfrs, 
        aep_derived_uom_fd='aep_derived_uom', 
        aep_srvc_qlty_idntfr_fd='aep_srvc_qlty_idntfr'
    ): 
        r"""
        Take a sloppy aep_uoms_and_qlty_idntfrs (acceptable element types described below) and build a 
        standardized list of dicts with keys aep_derived_uom and aep_srvc_qlty_idntfr
          - Each dict in list will have:
              keys:   aep_derived_uom and aep_srvc_qlty_idntfr
              values: equal to SQLWhereElement objects
          - ==> Returns, e.g.,: [dict(aep_derived_uom=SQLWhereElement_1_1, aep_srvc_qlty_idntfr=SQLWhereElement_1_2), 
                                 dict(aep_derived_uom=SQLWhereElement_2_1, aep_srvc_qlty_idntfr=None), 
                                 dict(aep_derived_uom=SQLWhereElement_3_1, aep_srvc_qlty_idntfr=SQLWhereElement_3_2), ...]

        - aep_uoms_and_qlty_idntfrs should be a list.  It's elements can be various types, which can be broken down
            into two two groups: uom_only and uom_w_idntfr (= [single-valued, single-valued])
                - Therefore, aep_uoms_and_qlty_idntfrs should be, e.g.,
                    =  [uom_1, uom_2, [uom_3, idntfr_3], 
                        uom_4, dict(aep_derived_uom=uom_5, aep_srvc_qlty_idntfr=idntfr_5), ...]
                    == [uom_and_maybe_idntfr_1, uom_and_maybe_idntfr_2, uom_and_maybe_idntfr_3, 
                       uom_and_maybe_idntfr_4, uom_and_maybe_idntfr_5, ...]
                - In either case, the basic element (uom_i or idntfr_i) can be of type 
                  str, dict, or SQLWhereElement.
                - uom_and_maybe_idntfr_i may therefore take on the form:
                    - uom_and_maybe_idntfr_i = uom_i = str, dict, or SQLWhereElement
                    - uom_and_maybe_idntfr_i = [uom_i, idntfr_i]
                                             = [(str, dict, or SQLWhereElement), (str, dict, or SQLWhereElement)]
                    - uom_and_maybe_idntfr_i = dict(aep_derived_uom=uom_i, aep_srvc_qlty_idntfr=idntfr_i)
                                             = dict(aep_derived_uom      = (str, dict, or SQLWhereElement), 
                                                    aep_srvc_qlty_idntfr = (str, dict, or SQLWhereElement))
        - field_desc DETERMINATION: In all cases, field_desc for the SQLWhereElement to be built is 
          determined by whether:
            - uom_and_maybe_idntfr_i = uom_i:
                ==> uom_i field_desc_determined = 'aep_derived_uom'
            - uom_and_maybe_idntfr_i = [uom_i, idntfr_i]
                ==> uom_i    field_desc_determined = 'aep_derived_uom'
                ==> idntfr_i field_desc_determined = 'aep_srvc_qlty_idntfr'
            - uom_and_maybe_idntfr_i = dict(aep_derived_uom=uom_i, aep_srvc_qlty_idntfr=idntfr_i)
                ==> uom_i    field_desc_determined = 'aep_derived_uom'
                ==> idntfr_i field_desc_determined = 'aep_srvc_qlty_idntfr'
        - See SQLWhere.standardize_to_where_element for more details, but in short...
          - uom_i/idntfr_i = str
              - This is possible (as opposed to input options in SQLWhere.standardize_to_where_element) because 
                field_desc is determined by... SEE "field_desc DETERMINATION" above
              - SQLWhereElement(field_desc=field_desc_determined, 
                                value=uom_i/idntfr_i, 
                                comparison_operator='=', 
                                needs_quotes=True, 
                                table_alias_prefix=None)
          - uom_i/idntfr_i = dict ==>
              - SQLWhereElement(field_desc=field_desc_determined, 
                                value = uom_i/idntfr_i['value'], 
                                comparison_operator = uom_i/idntfr_i.get('comparison_operator', '='), 
                                needs_quotes = uom_i/idntfr_i.get('needs_quotes', True), 
                                table_alias_prefix = uom_i/idntfr_i.get('table_alias_prefix', None))             
          - uom_i/idntfr_i = SQLWhereElement ==> uom_i/idntfr_i         


            - Single-valued element:
              - The element represent only aep_derived_uom, and aep_srvc_qlty_idntfr is taken to be None.

            - Pair-valued element (of type list/tuple):
              - The element should have length 2.
              - The first component represents aep_derived_uom, 
              - The second component represents aep_srvc_qlty_idntfr

        TODO REDO DOCUMENTATION BELOW, AS THINGS HAVE CHANGED
        aep_uoms_and_qlty_idntfrs should be a list whose elements are of type:
          i.   string, equal to a aep_derived_uom 
                   e.g. aep_uoms_and_qlty_idntfrs = ['KVARH', 'KVAH', ...]
          ii.  list/tuple, equal to [aep_derived_uom, aep_srvc_qlty_idntfr] pair, in that order
                   e.g. aep_uoms_and_qlty_idntfrs = [['VOLT', 'AVG'], ...]
          iii. dict with keys equal to aep_derived_uom, aep_srvc_qlty_idntfr
                 - The aep_derived_uom key is required, aep_srvc_qlty_idntfr is optional
                 - The values can be a single string or a list/tuple of length 2
                 - value = single string
                     This is taken as the value to be fed into SQLWhereElement
                 - value = list/tuple of length 2
                     The first element is the value to be fed into SQLWhereElement
                     The second element is the comparison method to be fed into SQLWhereElement
                  e.g. aep_uoms_and_qlty_idntfrs = [dict(aep_derived_uom='VOLT', aep_srvc_qlty_idntfr='AVG'), 
                                                    dict(aep_derived_uom=['KWH', '<>'])]
          iv.  SQLWhereElement object
          iv.  any combination of the aboe
                   e.g. aep_uoms_and_qlty_idntfrs = ['KVARH', ['VOLT', 'AVG'], 
                                                     dict(aep_derived_uom='KVAH', aep_srvc_qlty_idntfr='DELIVERED')]
        """
        assert(isinstance(aep_uoms_and_qlty_idntfrs, list) or isinstance(aep_uoms_and_qlty_idntfrs, tuple))
        std_derived_uoms_and_srvc_qlty_idntfrs = [] #a list of dicts with keys aep_derived_uom, aep_srvc_qlty_idntfr
        for x in aep_uoms_and_qlty_idntfrs:
            assert(
                isinstance(x, str) or 
                isinstance(x, dict) or 
                isinstance(x, SQLWhereElement) or 
                ((isinstance(x, list) or isinstance(x, tuple)) and len(x)==2)
            )
            std_uom_and_idntfr_i = AMINonVee_SQL.standardize_uom_and_idntfr(x)
            std_derived_uoms_and_srvc_qlty_idntfrs.append(std_uom_and_idntfr_i)
        return std_derived_uoms_and_srvc_qlty_idntfrs
    
    #****************************************************************************************************
    @staticmethod
    def add_uoms_and_idntfrs_to_sql_where(
        sql_where                    , 
        aep_derived_uoms_and_idntfrs , 
        aep_derived_uom_col          = 'aep_derived_uom', 
        aep_srvc_qlty_idntfr_col     = 'aep_srvc_qlty_idntfr'
    ):
        r"""
        This was developed assuming no aep_derived_uom where statements in original sql_where

        aep_derived_uoms_and_idntfrs should be a list whose elements are of type:
          i.   string, equal to a aep_derived_uom 
                   e.g. aep_derived_uoms_and_idntfrs = ['KVARH', 'KVAH']
          ii.  tuple, equal to [aep_derived_uom, aep_srvc_qlty_idntfr] pair, in that order
                   e.g. aep_derived_uoms_and_idntfrs = [['VOLT', 'AVG']]
          iii. dict with keys equal to aep_derived_uom, aep_srvc_qlty_idntfr
                   e.g. aep_derived_uoms_and_idntfrs = [dict(aep_derived_uom='VOLT', aep_srvc_qlty_idntfr='AVG')]
          iv.  any combination of the aboe
                   e.g. aep_derived_uoms_and_idntfrs = ['KVARH', ['VOLT', 'AVG'], 
                                                        dict(aep_derived_uom='KVAH', aep_srvc_qlty_idntfr='DELIVERED')]
        """
        if aep_derived_uoms_and_idntfrs is None or len(aep_derived_uoms_and_idntfrs)==0:
            return sql_where
        # First, convert aep_derived_uoms_and_idntfrs to standardized version derived_uoms_and_srvc_qlty_idntfrs
        std_aep_derived_uoms_and_idntfrs = AMINonVee_SQL.standardize_aep_derived_uoms_and_srvc_qlty_idntfrs(aep_derived_uoms_and_idntfrs) 
        #-----
        for uom_idfr_dict in std_aep_derived_uoms_and_idntfrs:
            sql_where.add_where_statement(
                field_desc          = aep_derived_uom_col, 
                comparison_operator = '=', 
                value               = uom_idfr_dict[aep_derived_uom_col].value, 
                needs_quotes        = True
            )
            if uom_idfr_dict[aep_srvc_qlty_idntfr_col] is not None:
                sql_where.add_where_statement(
                    field_desc          = aep_srvc_qlty_idntfr_col, 
                    comparison_operator = '=', 
                    value               = uom_idfr_dict[aep_srvc_qlty_idntfr_col].value, 
                    needs_quotes        = True
                )
                # I want to join these together, and they should be the last two in the collection
                # Purpose of joining together is to make adding additional values easier in future
                sql_where.combine_last_n_where_elements(last_n=2, join_operator='AND', close_gaps_in_keys=True)
        # Now, finally, all of the various aep_derived_uom/aep_srvc_qlty_idntfr combinations need to be joined with OR operators
        if len(std_aep_derived_uoms_and_idntfrs)>1:
            sql_where.combine_last_n_where_elements(
                last_n             = len(std_aep_derived_uoms_and_idntfrs), 
                join_operator      = 'OR', 
                close_gaps_in_keys = True
            )
        return sql_where    
    
    #****************************************************************************************************
    #****************************************************************************************************
    #TODO do I need to add add_aggregate_elements kwargs to allow one to tweak e.g.,  comp_table_alias_prefix etc.?
    @staticmethod
    def build_sql_usg(
        cols_of_interest             = None, 
        aep_derived_uoms_and_idntfrs = None,  
        kwh_and_vlt_only             = False, 
        **kwargs
    ):
        r"""
        See AMI_SQL.add_ami_where_statements for my updated list of acceptable kwargs with respect to the where statement.
        
        Acceptable kwargs:
          *** kwargs from AMI_SQL.add_ami_where_statements
          *************************
          - date_range
            - tuple with two string elements, e.g., ['2021-01-01', '2021-04-01']
          - date_col
            - default: aep_usage_dt

          - serial_number(s)
          - serial_number_col
            - default: serialnumber

          - premise_nb(s) (or, aep_premise_nb(s) will work also)
          - premise_nb_col
            - default: aep_premise_nb

          - opco(s) (or, aep_opco(s) will work also)
          - opco_col
            - default: aep_opco

          - state(s)
          - state_col
            - default: aep_state
            
          - value_min
          - value_max
          - value
          - value_col
            - default: 'value'


          - groupby_cols
          - agg_cols_and_types
          - include_counts_including_null

          - schema_name
          - table_name
          
          - alias (query alias)

          *** Additional kwargs
          *************************
          - aep_derived_uom_col
          - aep_srvc_qlty_idntfr_col
          


        If kwh_and_vlt_only==True, this overrides anything in aep_derived_uoms_and_idntfrs
        
        TODO: How can I adjust this function to allow e.g., serial_numbers NOT IN...
                 - For now, easiest method will be to use standard build_sql_meter_premise then
                   the utilize the change_comparison_operator_of_element/change_comparison_operator_of_element_at_idx
                   method of SQLWhere to change mp_sql.sql_where after mp_sql is returned
                 - The second option would be to input serial_numbers of type SQLWhereElement with all correct
                   attributes set before inputting

        aep_derived_uoms_and_idntfrs should be a list whose elements are of type:
          i.   string, equal to a aep_derived_uom 
                   e.g. aep_derived_uoms_and_idntfrs = ['KVARH', 'KVAH']
          ii.  tuple, equal to [aep_derived_uom, aep_srvc_qlty_idntfr] pair, in that order
                   e.g. aep_derived_uoms_and_idntfrs = [['VOLT', 'AVG']]
          iii. dict with keys equal to aep_derived_uom, aep_srvc_qlty_idntfr
                   e.g. aep_derived_uoms_and_idntfrs = [dict(aep_derived_uom='VOLT', aep_srvc_qlty_idntfr='AVG')]
          iv.  any combination of the aboe
                   e.g. aep_derived_uoms_and_idntfrs = ['KVARH', ['VOLT', 'AVG'], 
                                                        dict(aep_derived_uom='KVAH', aep_srvc_qlty_idntfr='DELIVERED')]
        """
        #-------------------------
        # Some preliminary unpacking...
        aep_derived_uom_col        = kwargs.get('aep_derived_uom_col', 'aep_derived_uom')
        aep_srvc_qlty_idntfr_col   = kwargs.get('aep_srvc_qlty_idntfr_col', 'aep_srvc_qlty_idntfr')
        #**************************************************
        kwargs['schema_name']      = kwargs.get('schema_name', 'usage_nonvee')
        kwargs['table_name']       = kwargs.get('table_name', 'reading_ivl_nonvee')
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', None)
        #-------------------------
        if cols_of_interest is None:
            cols_of_interest = TableInfos.AMINonVee_TI.std_columns_of_interest
        # The following ensures a str is not returned.  Therefore, usg_sql below will either be:
        #   i. a SQLQuery object, or
        #  ii. a dict with values which are SQLQuery objects (when joined with MP)
        usg_sql = AMI_SQL.build_sql_ami(
            cols_of_interest=cols_of_interest, 
            **{
                **{k:v for k,v in kwargs.items() if k!='return_args'}, 
                **{'return_args':{'return_statement':False}}
            }
        )
        #-------------------------
        value_col = kwargs.get('value_col', 'value')
        value_min = kwargs.get('value_min', None)
        value_max = kwargs.get('value_max', None)
        value     = kwargs.get('value', None)
        #-----
        if value is not None:
            assert(
                value_min is None and 
                value_max is None
            )
        #-----
        if value_min is not None:
            if isinstance(value_min, SQLWhereElement):
                usg_sql.sql_where.add_where_statement(value_min)
            else:
                usg_sql.sql_where.add_where_statement(
                    field_desc          = value_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = '>', 
                    value               = value_min, 
                    needs_quotes        = False
                )
        if value_max is not None:
            if isinstance(value_max, SQLWhereElement):
                usg_sql.sql_where.add_where_statement(value_max)
            else:
                usg_sql.sql_where.add_where_statement(
                    field_desc          = value_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = '<', 
                    value               = value_max, 
                    needs_quotes        = False
                )
        if value is not None:
            if isinstance(value, SQLWhereElement):
                usg_sql.sql_where.add_where_statement(value)
            else:
                usg_sql.sql_where.add_where_statement(
                    field_desc          = value_col, 
                    table_alias_prefix  = kwargs['from_table_alias'], 
                    comparison_operator = '=', 
                    value               = value, 
                    needs_quotes        = False
                )
        
        #-------------------------
        if kwh_and_vlt_only:
            aep_derived_uoms_and_idntfrs = [['VOLT', 'AVG'], 'KWH']
        #-------------------------
        assert(Utilities.is_object_one_of_types(usg_sql, [SQLQuery, dict]))
        if isinstance(usg_sql, SQLQuery):
            usg_sql.sql_where = AMINonVee_SQL.add_uoms_and_idntfrs_to_sql_where(
                sql_where                    = usg_sql.sql_where, 
                aep_derived_uoms_and_idntfrs = aep_derived_uoms_and_idntfrs, 
                aep_derived_uom_col          = aep_derived_uom_col, 
                aep_srvc_qlty_idntfr_col     = aep_srvc_qlty_idntfr_col
            )
            return usg_sql
        else:
            assert('mp_sql' in usg_sql)
            assert('usg_sql' in usg_sql)
            assert(isinstance(usg_sql['mp_sql'], SQLQuery))
            assert(isinstance(usg_sql['usg_sql'], SQLQuery))
            usg_sql['usg_sql'].sql_where = AMINonVee_SQL.add_uoms_and_idntfrs_to_sql_where(
                sql_where                    = usg_sql['usg_sql'].sql_where, 
                aep_derived_uoms_and_idntfrs = aep_derived_uoms_and_idntfrs, 
                aep_derived_uom_col          = aep_derived_uom_col, 
                aep_srvc_qlty_idntfr_col     = aep_srvc_qlty_idntfr_col
            )
            return usg_sql

        
    #****************************************************************************************************
    # NET KWH METHODS
    #****************************************************************************************************
    #****************************************************************************************************
    @staticmethod
    def adjust_value_to_sum_signed_val_in_sql_select(
        sql_select, 
        value_col            = 'value', 
        sum_signed_val       = 'SUM(signed_value)', 
        sum_signed_val_alias = None
    ):
        r"""
        In sql_select, changes value to SUM(signed_value) AS value.
        
        TODO: Does this make more sense living in SQLSelect?

        - The function tries to find value_col in sql_select using the most basic version of 
            SQLElementsCollection.find_idx_of_approx_element_in_collection_dict, which only chekcs field_desc
            (not, e.g., alias or table_alias_prefix, etc.)
        - If value_col not found, function crashes
        - Replace value_col with  SQLSelectElement(field_desc=sum_signed_val, 
                                                   alias=sum_signed_val_alias (if not None) or value_col, 
                                                   table_alias_prefix=None)

        """
        if not sum_signed_val_alias:
            sum_signed_val_alias = value_col
        value_idx = sql_select.find_idx_of_approx_element_in_collection_dict(SQLSelectElement(value_col))
        assert(value_idx>-1)
        sql_select.remove_single_element_from_collection_at_idx(value_idx)
        sql_select.add_select_element(field_desc=sum_signed_val, alias=sum_signed_val_alias, idx=value_idx) 
        return sql_select

    #****************************************************************************************************
    @staticmethod
    def adjust_aep_srvc_qlty_idntfr_to_const_in_sql_select(
        sql_select, 
        aep_srvc_qlty_idntfr_col = 'aep_srvc_qlty_idntfr', 
        new_const_val            = 'CALCULATED_NET'
    ):
        r"""
        In sql_select, changes aep_srvc_qlty_idntfr to 'CALCULATED_NET' AS aep_srvc_qlty_idntfr.
          - So, instead of gathering the actual aep_srvc_qlty_idntfr, all are set to 'CALCULATED_NET'
          - This was designed to work when combining delivered and received values into net 

        - The function tries to find aep_srvc_qlty_idntfr_col in sql_select using the most basic version of 
            SQLElementsCollection.find_idx_of_approx_element_in_collection_dict, which only chekcs field_desc
            (not, e.g., alias or table_alias_prefix, etc.)
        - If aep_srvc_qlty_idntfr_col not found, function crashes
        - Replace aep_srvc_qlty_idntfr_col with  SQLSelectElement(field_desc=f"'{new_const_val}'", 
                                                                  alias=aep_srvc_qlty_idntfr_col, 
                                                                  table_alias_prefix=None)
        """
        # Going to sum over aep_srvc_qlty_idntfr, (by excluding from groupby) so don't want it in selection anymore
        # However, do still want it in table, just as a specified constant value signifying that it was calculated by hand
        aep_srvc_qlty_idntfr_idx = sql_select.find_idx_of_approx_element_in_collection_dict(SQLSelectElement(aep_srvc_qlty_idntfr_col))
        assert(aep_srvc_qlty_idntfr_idx>-1)
        sql_select.remove_single_element_from_collection_at_idx(aep_srvc_qlty_idntfr_idx)
        # But, add in generic aep_srvc_qlty_idntfr with value equal to 'CALCULATED_NET'
        sql_select.add_select_element(field_desc=f"'{new_const_val}'", alias=aep_srvc_qlty_idntfr_col, idx=aep_srvc_qlty_idntfr_idx)
        return sql_select


    #****************************************************************************************************
    @staticmethod
    def build_sql_kwh_usg_delrec_w_signed_val(
        cols_of_interest, 
        **kwargs
    ):
        r"""
        Returns a SQLQuery object for aep_derived_uom = 'KWH' and aep_srvc_qlty_idntfr = 'RECEIVED' or 'DELIVERED' with an additional
          returned field signed_value.
        - See AMINonVee_SQL.build_sql_usg for possible kwargs key/value combinations
        - aep_srvc_qlty_idntfr = 'RECEIVED':
            ==> signed_value = -1*value
        - aep_srvc_qlty_idntfr = 'DELIVERED':
            ==> signed_value = value
        """
        usg_sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest, 
            aep_derived_uoms_and_idntfrs=[['KWH', 'RECEIVED'], ['KWH', 'DELIVERED']], 
            kwh_and_vlt_only=False, 
            **kwargs
        )
        usg_sql.sql_select.add_select_element(field_desc="IF(aep_srvc_qlty_idntfr='RECEIVED', -1*value, value)", alias="signed_value")    
        return usg_sql

    #****************************************************************************************************
    @staticmethod
    def build_sql_kwh_usg_total_only(
        cols_of_interest, 
        **kwargs
    ):
        r"""
        Returns a SQLQuery object for aep_derived_uom = 'KWH' and aep_srvc_qlty_idntfr = 'RECEIVED' or 'DELIVERED' with an additional
          returned field signed_value.
        - See AMINonVee_SQL.build_sql_usg for possible kwargs key/value combinations
        - aep_srvc_qlty_idntfr = 'RECEIVED':
            ==> signed_value = -1*value
        - aep_srvc_qlty_idntfr = 'DELIVERED':
            ==> signed_value = value
        """
        usg_sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest, 
            aep_derived_uoms_and_idntfrs = [['KWH', 'TOTAL']], 
            kwh_and_vlt_only             = False, 
            **kwargs
        )  
        return usg_sql


    #****************************************************************************************************
    @staticmethod
    def build_sql_usg_agg_by_srvc_qlty_idntfr(
        cols_of_interest                   , 
        from_table_info                    , 
        groupby_cols                       = None, 
        value_col                          = 'value', 
        sum_signed_val_col                 = 'signed_value', 
        sum_signed_val_alias               = None, 
        new_const_aep_srvc_qlty_idntfr_val = 'CALCULATED_NET', 
        having_value                       = 2, 
        **kwargs
    ):
        r"""
        Build and returns SQLQuery object for aggregating the value_col field of a table by aep_srvc_qlty_idntfr.
          - value_col, kwargs['serialnumber_col'] (default to 'serialnumber') and kwargs['aep_srvc_qlty_idntfr_col']
            (default to 'aep_srvc_qlty_idntfr') must all be contained in cols_of_interest.
          - By default, the table will be grouped by all fields in cols_of_interest EXCLUDING value_col
            and kwargs['aep_srvc_qlty_idntfr_col']
            - However, groupby can be set to override this functionality.
          - aep_srvc_qlty_idntfr will be set equal to the constant value new_const_aep_srvc_qlty_idntfr_val for
            all elements in the resultant table.
        ---------------
        - cols_of_interest:
            - list of strings representing table fields to select
        ---------------
        - from_table_info:
            - Can be a simple str, SQLFrom object, or dict with keys appropriate for building a SQLFrom object
            - str:
                sql_usg.sql_from = SQLFrom(table_name=from_table_info)
            - SQLFrom object:
                sql_usg.sql_from = from_table_info
            - dict:
                sql_usg.sql_from = SQLFrom(**from_table_info)

        ---------------
        - groupby_cols:
          - list of string representing table fields by which to group.
          - By default, when groupby_cols is None, the table will be grouped by all fields in cols_of_interest 
            EXCLUDING value_col and kwargs['aep_srvc_qlty_idntfr_col']
        ---------------
        - value_col, sum_signed_val_col, sum_signed_val_alias:
          - Input into AMINonVee_SQL.adjust_value_to_sum_signed_val_in_sql_select: 
              - value_col, f'SUM({sum_signed_val_col})', sum_signed_val_alias
          - In the SELECT statement, this makes the change:
              - value_col --> f'SUM({sum_signed_val_col})' AS sum_signed_val_alias
                e.g.,  value --> SUM(signed_value) as sum_signed_val_alias
          - When sum_signed_val_alias is None, it is set equal to value_col
                e.g.,  value --> SUM(signed_value) as value   
        ---------------
        - new_const_aep_srvc_qlty_idntfr_val:
          - aep_srvc_qlty_idntfr will be set equal to the constant value new_const_aep_srvc_qlty_idntfr_val for
            all elements in the resultant table.    
        ---------------
        - having_value:
          - Value to input into SQLHaving statement
            - SQLHaving([dict(field_desc=f'COUNT({sum_signed_val_col})', comparison_operator='=', 
                         value=f'{having_value}', needs_quotes=False)
                         ], idxs=None, run_check=True)
          - Main purpose is to ensure the groups found are of the size expected
            - e.g., when dealing with combining DELIVERED and RECEIVED, expect group size to be equal to 2,
                therefore, set having_value=2
            - e.g., when combining net delivered/recieved with total, expect group size to be equal to 1 (since 
                a given serial number should only have DELIVERED/RECEIVED or TOTAL/RECEIVED, but not both), 
                therefore, set having_value=1           
        ---------------
        - See AMINonVee_SQL.build_sql_usg for possible kwargs key/value pairs
        """
        # TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO IN FUTURE will allow for groupby_cols input argument, which will default to 
        # cols_of_interest - value_col - aep_srvc_qlty_idntfr_col if non input
        # Leaving out for now because I want a failure if groupby_cols is included, to ensure everything is working
        # TODO also in the future maybe(?) make serial_numbers = None by default, but again want to fail if not for now
        # TODO also, maybe want aep_opco etc in different dict than kwargs...
        #-------------------------
        kwargs['serialnumber_col']         = kwargs.get('serialnumber_col', 'serialnumber')
        kwargs['aep_srvc_qlty_idntfr_col'] = kwargs.get('aep_srvc_qlty_idntfr_col', 'aep_srvc_qlty_idntfr')
        #-----
        assert(value_col                          in cols_of_interest)
        assert(kwargs['aep_srvc_qlty_idntfr_col'] in cols_of_interest)
        assert(kwargs['serialnumber_col']         in cols_of_interest)
        #-------------------------
        if groupby_cols is None:
            groupby_cols = Utilities.include_at_front_and_exclude_from_list(
                cols_of_interest, 
                exclude_from_list = [kwargs['aep_srvc_qlty_idntfr_col'], value_col], 
                inplace           = False
            )
        #-------------------------
        sql_usg = AMINonVee_SQL.build_sql_usg(
            cols_of_interest             = cols_of_interest, 
            aep_derived_uoms_and_idntfrs = None, 
            kwh_and_vlt_only             = False, 
            groupby_cols                 = groupby_cols, 
            **kwargs
        )
        #-------------------------
        # Instead of value, want SUM(signed_value) AS value
        sql_usg.sql_select = AMINonVee_SQL.adjust_value_to_sum_signed_val_in_sql_select(
            sql_usg.sql_select, 
            value_col            = value_col, 
            sum_signed_val       = f'SUM({sum_signed_val_col})', 
            sum_signed_val_alias = sum_signed_val_alias
        )
        #-----
        # Going to sum over aep_srvc_qlty_idntfr, (by excluding from groupby) so don't want it in selection anymore
        # But, add in generic aep_srvc_qlty_idntfr with value equal to 'CALCULATED_NET'
        sql_usg.sql_select = AMINonVee_SQL.adjust_aep_srvc_qlty_idntfr_to_const_in_sql_select(
            sql_usg.sql_select, 
            aep_srvc_qlty_idntfr_col = kwargs['aep_srvc_qlty_idntfr_col'], 
            new_const_val            = new_const_aep_srvc_qlty_idntfr_val
        )
        #-------------------------
        assert(Utilities.is_object_one_of_types(from_table_info, [str, SQLFrom, dict]))
        if isinstance(from_table_info, str):
            sql_usg.sql_from = SQLFrom(table_name=from_table_info)
        elif isinstance(from_table_info, SQLFrom):
            sql_usg.sql_from = from_table_info
        elif isinstance(from_table_info, dict):
            sql_usg.sql_from = SQLFrom(**from_table_info)
        else:
            assert(0)
        #-------------------------
        if having_value>0:
            sql_usg.sql_having = SQLHaving(
                [
                    dict(
                        field_desc          = f'COUNT({sum_signed_val_col})', 
                        comparison_operator = '=', 
                        value               = f'{having_value}', 
                        needs_quotes        = False
                    )
                ], 
                idxs      = None, 
                run_check = True
            )
        return sql_usg


    #****************************************************************************************************
    @staticmethod
    def assemble_net_kwh_usage_sql_statement(usg_sql_dict, final_table_alias='USG_KWH', 
                                             insert_n_tabs_to_each_line=0, prepend_with_to_stmnt=False):
        assert('sql_kwh_usg_delrec_w_signed_val'      in usg_sql_dict)
        assert('sql_kwh_usg_delrec_net'               in usg_sql_dict)
        assert('sql_kwh_usg_total_only'               in usg_sql_dict)
        assert('sql_kwh_usg_delrec_net_union_total_0' in usg_sql_dict)
        assert('sql_kwh_usg_delrec_net_union_total'   in usg_sql_dict)
        assert('additional_sql'                       in usg_sql_dict)
        #-----
        sql_kwh_usg_delrec_w_signed_val      = usg_sql_dict['sql_kwh_usg_delrec_w_signed_val']
        sql_kwh_usg_delrec_net               = usg_sql_dict['sql_kwh_usg_delrec_net']
        sql_kwh_usg_total_only               = usg_sql_dict['sql_kwh_usg_total_only']
        sql_kwh_usg_delrec_net_union_total_0 = usg_sql_dict['sql_kwh_usg_delrec_net_union_total_0']
        sql_kwh_usg_delrec_net_union_total   = usg_sql_dict['sql_kwh_usg_delrec_net_union_total']
        additional_sql                       = usg_sql_dict['additional_sql']
        #-----
        if prepend_with_to_stmnt:
            sql_full_stmnt = "WITH "
        else:
            sql_full_stmnt = ""
        sql_full_stmnt += f"{sql_kwh_usg_delrec_w_signed_val.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=True)}, \n"\
                          f"\n{sql_kwh_usg_delrec_net.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=True)}, \n"\
                          f"\n{sql_kwh_usg_total_only.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=True)}, \n"\
                          f"\n{sql_kwh_usg_delrec_net_union_total_0.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=True)}, \n"
        if additional_sql is None:
            sql_full_stmnt += f"\n{final_table_alias} AS (\n{sql_kwh_usg_delrec_net_union_total.get_sql_statement(insert_n_tabs_to_each_line=1, include_alias=False)}\n)"
        else:
            sql_full_stmnt += f"\n{sql_kwh_usg_delrec_net_union_total.get_sql_statement(insert_n_tabs_to_each_line=0, include_alias=True)}, \n"
            usage_union_sql = SQLQuery(sql_select = SQLSelect(['*']), 
                                       sql_from = SQLFrom(table_name=sql_kwh_usg_delrec_net_union_total.alias), 
                                       sql_where = None)        
            sql_full_stmnt += f"\n{final_table_alias} AS (\n" \
                              f"\t(\n{usage_union_sql.get_sql_statement(insert_n_tabs_to_each_line=1, include_alias=False)}\n\t)" \
                              f"\n\tUNION\n" \
                              f"\t(\n{additional_sql.get_sql_statement(insert_n_tabs_to_each_line=1, include_alias=False)}\n\t)" \
                              f"\n)"
        if insert_n_tabs_to_each_line>0:
            sql_full_stmnt = Utilities_sql.prepend_tabs_to_each_line(sql_full_stmnt, n_tabs_to_prepend=insert_n_tabs_to_each_line)
        return sql_full_stmnt


    #****************************************************************************************************
    @staticmethod
    def build_net_kwh_usage_sql_statement(
        cols_of_interest                     , 
        additional_derived_uoms              = None, 
        run_careful                          = True, 
        value_col                            = 'value', 
        return_statement                     = True, 
        final_table_alias                    = 'USG_KWH', 
        insert_n_tabs_to_each_line           = 0, 
        prepend_with_to_stmnt                = False, 
        join_mp_args                         = False, 
        allow_kwh_in_additional_derived_uoms = False, 
        **kwargs
    ):
        r"""
        Build and returns an object which can be used to run an SQL query to fetch the net kWh value, which
        is calculated as DELIVERED-RECIEVED or TOTAL
          - For aep_derived_uom=KWH, a given serial number typically has aep_srvc_qlty_idntfrs = DELIVERED/RECEIVED
            or TOTAL/RECEIVED, but not both.
          - For DELIVERED/RECEIVED entries, the net value is calculated as the difference.
          - For TOTAL/RECEIVED entries, the net value is taken to be equal to TOTAL

        Return value:
          - If return_statement==False: 
              - returns a dict whose values are the various SQLQuery objects needed to compose the
                full SQL statement
          - If return_statement==True:
              - returns a string representing the full SQL statement, which can then be fed into e.g. pd.read_sql
              - Uses AMINonVee_SQL.assemble_net_kwh_usage_sql_statement to build the string

        kwargs:
          - See AMINonVee_SQL.build_sql_usg for key/value descriptions


        additional_derived_uoms:
          - additional_derived_uoms can take on any form acceptable for the aep_derived_uoms input parameter
              in AMINonVee_SQL.build_sql_usg (reproduced below).
          - ADDITIONALLY, additional_derived_uoms may equal the string 'ALL'
              If additional_derived_uoms=='ALL', then the net kWh table will be combine with
              aep_derived_uoms of all OTHER types (exclduing, of course, 'KWH', as this is handled in
              the rest of this function!)
              
        join_mp_args:
          - Adds join statements to sql_kwh_usg_delrec_w_signed_val, sql_kwh_usg_total_only, and additional_sql
          - can be bool (True or False) or dict with possible keys (w/ default values) = 
              join_type='INNER' 
              join_table=None 
              join_table_alias='MP' 
              join_table_column='mfr_devc_ser_nbr' 
              idx=None 
              run_check=False
              
        allow_kwh_in_additional_derived_uoms:
          - Typically, when building a table with net kwh, one does not want the typical individual aep_srvc_qlty_idntfrs,
            only the net kwh.
          - However, one may want new kwh to be calculated, the aep_srvc_qlty_idntfr value set to 'CALCULATED_NET', and 
            combined with all others
            - In such a case, the possible aep_srvc_qlty_idntfrs for aep_derived_uom=='KWH' would be
              RECEIVED, DELIVERED, TOTAL, CALCULATED_NET
          - If such a functionality is desired, set allow_kwh_in_additional_derived_uoms=True
            - It is False by default
          - NOTE: This switch will only have an effect if additional_derived_uoms=='ALL' or contains 'KWH'

        *** from AMINonVee_SQL.build_sql_usg ***
        aep_derived_uoms should be a list whose elements are of type:
          i.   string, equal to a aep_derived_uom 
                   e.g. aep_derived_uoms = ['KVARH', 'KVAH']
          ii.  tuple, equal to [aep_derived_uom, aep_srvc_qlty_idntfr] pair, in that order
                   e.g. aep_derived_uoms = [['VOLT', 'AVG']]
          iii. dict with keys equal to aep_derived_uom, aep_srvc_qlty_idntfr
                   e.g. aep_derived_uoms = [dict(aep_derived_uom='VOLT', aep_srvc_qlty_idntfr='AVG')]
          iv.  any combination of the aboe
                   e.g. aep_derived_uoms = ['KVARH', ['VOLT', 'AVG'], 
                                            dict(aep_derived_uom='KVAH', aep_srvc_qlty_idntfr='DELIVERED')]

        NOTE: FOR UNION TO WORK, NEED TO BE CAREFUL AND ENSURE COLUMNS ARE EXACTLY THE SAME AS ARE THEIR ORDERS!
        TODO: Should implement methods to make sure everything aligns before union

        ****************************************************************************************************      
        Flow of statement construction is as follows:
          *** Unless otherwise noted, it is assumed that aep_derived_uom=='KWH' for the below tables
          1. sql_kwh_usg_delrec_w_signed_val (KWH_USG_DELREC_W_SIGNED_VAL)
              - A table where aep_srvc_qlty_idntfr=='DELIVERED' or 'RECEIVED' with an additional returned
                field signed_value.
              - aep_srvc_qlty_idntfr = 'RECEIVED':
                  ==> signed_value = -1*value
              - aep_srvc_qlty_idntfr = 'DELIVERED':
                  ==> signed_value = value
          2. sql_kwh_usg_delrec_net (KWH_USG_DELREC_NET)
              - A table built from KWH_USG_DELREC_W_SIGNED_VAL in which the DELIVERED and RECEIVED values
                are combined to form a net (DEL_MINUS_REC) value
              - In KWH_USG_DELREC_W_SIGNED_VAL, value is summed after aggregating by all fields except
                aep_srvc_qlty_idntfr (and value, of course)
                - There should be two members in each group of the aggregation, which is why SQLHaving is
                  included with a value of 2.
              - aep_srvc_qlty_idntfr set to 'DEL_MINUS_REC' for all
          3. sql_kwh_usg_total_only (KWH_USG_TOTAL_VAL)
             - A table built where aep_srvc_qlty_idntfr=='TOTAL'
             - As entries are DELIVERED/RECEIVED or TOTAL/RECEIVED, the data contained in KWH_USG_TOTAL_VAL should
               be unique from those in KWH_USG_DELREC_NET/KWH_USG_DELREC_W_SIGNED_VAL
          4. sql_kwh_usg_delrec_net_union_total_0 (KWH_USG_DELREC_NET_UNION_TOTAL_0)
              - A union of KWH_USG_DELREC_NET and KWH_USG_TOTAL_VAL (i.e., simply stacking the two tables)
          5. sql_kwh_usg_delrec_net_union_total (KWH_USG_DELREC_NET_UNION_TOTAL/USG_KWH)
              - If additional_derived_uoms is None:
                - USG_KWH is built either normally or carefully (depending on run_careful status) 
                  from KWH_USG_DELREC_NET_UNION_TOTAL_0
              - If additional_derived_uoms is not None:
                - KWH_USG_DELREC_NET_UNION_TOTAL is built either normally or carefully (depending on run_careful status)
                  from KWH_USG_DELREC_NET_UNION_TOTAL_0
                  - KWH_USG_DELREC_NET_UNION_TOTAL equals USG_KWH from "If additional_derived_uoms is None" 
                - USG_KWH is built as the union of KWH_USG_DELREC_NET_UNION_TOTAL and a table containing the other
                  derived uoms contained in additional_derived_uoms
        """
        kwargs['serialnumber_col']         = kwargs.get('serialnumber_col', 'serialnumber')
        kwargs['aep_srvc_qlty_idntfr_col'] = kwargs.get('aep_srvc_qlty_idntfr_col', 'aep_srvc_qlty_idntfr')
        #--------------------------------------------------
        assert(Utilities.is_object_one_of_types(join_mp_args, [bool, dict]))
        dflt_join_mp_args = dict(
            join_type                  = 'INNER', 
            join_table                 = None, 
            join_table_alias           = 'MP', 
            join_table_column          = 'mfr_devc_ser_nbr', 
            idx                        = None, 
            run_check                  = False, 
            join_cols_to_add_to_select = None
        )
        # If join_mp_args is False or an empty dict, no join will occur
        if join_mp_args:
            if isinstance(join_mp_args, bool):
                join_mp_args = dflt_join_mp_args
            elif isinstance(join_mp_args, dict):
                join_mp_args = Utilities_sql.supplement_dict_with_default_values(join_mp_args, dflt_join_mp_args)
            else:
                assert(0)
            #-------------------------
            join_table_column_mp  = join_mp_args.pop('join_table_column')
            join_mp_args['list_of_columns_to_join'] = [[kwargs['serialnumber_col'], join_table_column_mp]]
            #-------------------------
            join_cols_to_add_to_select = join_mp_args.pop('join_cols_to_add_to_select')
        else:
            join_mp_args = False
        #--------------------------------------------------

        # Building table with name KWH_USG_DELREC_W_SIGNED_VAL
        #   - aep_derived_uom='KWH' and aep_srvc_qlty_idntfr=='DELIVERED' or 'RECEIVED'
        #   - signed_value field = value when aep_srvc_qlty_idntfr=='DELIVERED' 
        #       and -1*value when aep_srvc_qlty_idntfr=='RECEIVED'
        sql_kwh_usg_delrec_w_signed_val = AMINonVee_SQL.build_sql_kwh_usg_delrec_w_signed_val(
            cols_of_interest = cols_of_interest, 
            alias            = 'KWH_USG_DELREC_W_SIGNED_VAL', 
            from_table_alias = 'USG_i', 
            **{k:v for k,v in kwargs.items() if k!='from_table_alias'} #Don't repeat from_table_alias
        ) 
        if join_mp_args:
            join_mp_args['orig_table_alias'] = 'USG_i'
            sql_kwh_usg_delrec_w_signed_val.build_and_add_join(**join_mp_args)
            if join_cols_to_add_to_select is not None:
                sql_kwh_usg_delrec_w_signed_val.sql_select.add_select_elements(
                    join_cols_to_add_to_select, 
                    global_table_alias_prefix = join_mp_args['join_table_alias'], 
                    idxs                      = None, 
                    run_check                 = True
                )
        #------------------------------------------------------------
        # Building table with name KWH_USG_DELREC_NET (from KWH_USG_DELREC_W_SIGNED_VAL)
        #   - Basically, combining DELIVERED and RECEIVED values in KWH_USG_DELREC_W_SIGNED_VAL
        #   - WHERE statements already handled in sql_kwh_usg_delrec_w_signed_val.
        #       Therefore, kwargs not needed here in sql_kwh_usg_delrec_net.
        sql_kwh_usg_delrec_net = AMINonVee_SQL.build_sql_usg_agg_by_srvc_qlty_idntfr(
            cols_of_interest                   = cols_of_interest, 
            from_table_info                    = 'KWH_USG_DELREC_W_SIGNED_VAL', 
            groupby_cols                       = None, 
            value_col                          = value_col, 
            sum_signed_val_col                 = 'signed_value', 
            sum_signed_val_alias               = None, 
            new_const_aep_srvc_qlty_idntfr_val = 'DEL_MINUS_REC',  
            having_value                       = 2, 
            alias                              = 'KWH_USG_DELREC_NET'
        )
        #------------------------------------------------------------
        # Building table with name KWH_USG_TOTAL_VAL
        #   - aep_derived_uom='KWH' and aep_srvc_qlty_idntfr=='TOTAL'
        sql_kwh_usg_total_only = AMINonVee_SQL.build_sql_kwh_usg_total_only(
            cols_of_interest = cols_of_interest, 
            alias            = 'KWH_USG_TOTAL_VAL', 
            from_table_alias = 'USG_i', 
            **{k:v for k,v in kwargs.items() if k!='from_table_alias'} #Don't repeat from_table_alias
        )
        if join_mp_args:
            join_mp_args['orig_table_alias'] = 'USG_i'
            sql_kwh_usg_total_only.build_and_add_join(**join_mp_args)
            if join_cols_to_add_to_select is not None:
                sql_kwh_usg_total_only.sql_select.add_select_elements(
                    join_cols_to_add_to_select, 
                    global_table_alias_prefix = join_mp_args['join_table_alias'], 
                    idxs                      = None, 
                    run_check                 = True
                )
        #------------------------------------------------------------
        # Building table with name KWH_USG_DELREC_NET_UNION_TOTAL_0, which is
        #   a union of KWH_USG_DELREC_NET (sql_kwh_usg_delrec_net) and KWH_USG_TOTAL_VAL (sql_kwh_usg_total_only)
        #   i.e., simply stacking KWH_USG_DELREC_NET and KWH_USG_TOTAL_VAL
        #   - sql_kwh_usg_delrec_net combined any entries with 'RECEIVED' and 'DELIVERED'
        #   - sql_kwh_usg_total_only kept only 'TOTAL' entries (while discarding 'RECEIVED' for these pairs)
        #     ==> So, the procedure here allows for the case where the sample contains entires of both types
        #          'RECEIVED'/'DELIVERED' and 'RECEIVED'/'TOTAL'
        #
        # NOTE: easier to do SELECT *, but better to do it this way
        #       because it is important that columns are exactly the same in unions.
        #       Especially important if including any additional unions here
        sub_query_usg_wnv = SQLQuery(
            sql_select = SQLSelect(cols_of_interest), 
            sql_from   = SQLFrom(table_name='KWH_USG_DELREC_NET'), 
            sql_where  = None
        )

        sub_query_usg_wtv = SQLQuery(
            sql_select = SQLSelect(cols_of_interest), 
            sql_from   = SQLFrom(table_name='KWH_USG_TOTAL_VAL'), 
            sql_where  = None
        )
        #-------------------------
        sub_query_stmnt = f"(\n{sub_query_usg_wnv.get_sql_statement(insert_n_tabs_to_each_line=insert_n_tabs_to_each_line)}\n)" \
                          f"\nUNION\n" \
                          f"(\n{sub_query_usg_wtv.get_sql_statement(insert_n_tabs_to_each_line=insert_n_tabs_to_each_line)}\n)"
        #-------------------------
        sql_kwh_usg_delrec_net_union_total_0 = SQLQueryGeneric(sub_query_stmnt, alias='KWH_USG_DELREC_NET_UNION_TOTAL_0')
        #------------------------------------------------------------
        # Create final table, typically called USG_KWH from KWH_USG_DELREC_NET_UNION_TOTAL_0
        #  either as simple SELECT * FROM when run_careful=False
        #  or a simple groupby when run_carefule = True
        # Now, aggregate table from sql_kwh_usg_delrec_net_union_total_0 (similar to sql_kwh_usgbuild_net_kwh_delrec_net) to get the final net
        # value, which is the ..

        # Now that I am writing this out and thinking about it, I don't think all of this is necessary.
        # At this point, everything is still being done at the serial number level
        # Therefore, should not come across a case where all 'DELIVERED', 'RECEIVED' and 'TOTAL' are present.
        # IF such an instance exists, then TOTAL better equal DELIVERED-RECEIVED!
        # I'll finish out this effort anyway, but intend to go back to working with non-aggregate data when
        # developing this functionality instead of trying to skip ahead with agg
        if run_careful:
            # This "aggregates" again, but enforces COUNT({value_col})=1
            # Which essentially means make sure there is only one value per group.
            # The reason for this is as follows:
            #   A serial should either have 'DELIVERED'/'RECEIVED' OR 'TOTAL'/'RECEIVED'
            #   Therefore, all entries for a given serial number should be in sql_kwh_usg_delrec_net OR sql_kwh_usg_total_only
            #     but not both.  The code below essentially enforces this.
            #
            #   - WHERE statements already handled in sql_kwh_usg_delrec_net (sql_kwh_usg_delrec_w_signed_val) and sql_kwh_usg_total_only.
            #       Therefore, kwargs not needed here in sql_kwh_usg_delrec_net.
            sql_kwh_usg_delrec_net_union_total = AMINonVee_SQL.build_sql_usg_agg_by_srvc_qlty_idntfr(
                cols_of_interest                   = cols_of_interest, 
                from_table_info                    = 'KWH_USG_DELREC_NET_UNION_TOTAL_0', 
                groupby_cols                       = None, 
                value_col                          = value_col, 
                sum_signed_val_col                 = value_col, 
                sum_signed_val_alias               = None, 
                new_const_aep_srvc_qlty_idntfr_val = 'CALCULATED_NET', 
                having_value                       = 1, 
                alias                              = 'KWH_USG_DELREC_NET_UNION_TOTAL'
            )
        else:
            sql_kwh_usg_delrec_net_union_total = SQLQuery(
                sql_select = SQLSelect(cols_of_interest), 
                sql_from   = SQLFrom('KWH_USG_DELREC_NET_UNION_TOTAL_0'), 
                sql_where  = None, 
                alias      = 'KWH_USG_DELREC_NET_UNION_TOTAL'
            )
            sql_kwh_usg_delrec_net_union_total.sql_select = AMINonVee_SQL.adjust_aep_srvc_qlty_idntfr_to_const_in_sql_select(
                sql_kwh_usg_delrec_net_union_total.sql_select, 
                aep_srvc_qlty_idntfr_col = kwargs['aep_srvc_qlty_idntfr_col'], 
                new_const_val            = 'CALCULATED_NET'
            ) 



        #------------------------------------------------------------
        if additional_derived_uoms == 'ALL':
            additional_sql = AMINonVee_SQL.build_sql_usg(
                cols_of_interest             = cols_of_interest, 
                aep_derived_uoms_and_idntfrs = None, 
                kwh_and_vlt_only             = False, 
                from_table_alias             = 'USG_i', 
                **{k:v for k,v in kwargs.items() if k!='from_table_alias'} #Don't repeat from_table_alias
            )
            if not allow_kwh_in_additional_derived_uoms:
                additional_sql.sql_where.add_where_statement(
                    field_desc          = 'aep_derived_uom', 
                    comparison_operator = '<>', 
                    value               = 'KWH', 
                    needs_quotes        = True
                )        

        elif additional_derived_uoms is not None and len(additional_derived_uoms)>0:
            additional_derived_uoms_std = AMINonVee_SQL.standardize_aep_derived_uoms_and_srvc_qlty_idntfrs(additional_derived_uoms)
            if not allow_kwh_in_additional_derived_uoms:
                additional_derived_uoms_std = [x for x in additional_derived_uoms_std if x['aep_derived_uom'].value!='KWH']
            if len(additional_derived_uoms_std)>0:
                additional_sql = AMINonVee_SQL.build_sql_usg(
                    cols_of_interest             = cols_of_interest, 
                    aep_derived_uoms_and_idntfrs = additional_derived_uoms_std,  
                    kwh_and_vlt_only             = False, 
                    from_table_alias             = 'USG_i', 
                    **{k:v for k,v in kwargs.items() if k!='from_table_alias'} #Don't repeat from_table_alias
                )
            else:
                additional_sql = None
        else:
            additional_sql = None
        #-----
        if join_mp_args and isinstance(additional_sql, SQLQuery):
            join_mp_args['orig_table_alias'] = 'USG_i'
            additional_sql.build_and_add_join(**join_mp_args)
            if join_cols_to_add_to_select is not None:
                additional_sql.sql_select.add_select_elements(
                    join_cols_to_add_to_select, 
                    global_table_alias_prefix = join_mp_args['join_table_alias'], 
                    idxs                      = None, 
                    run_check                 = True
                )
        #-------------------------
        if additional_sql is not None:
            sub_query_stmnt +=  f"\nUNION\n" \
                                f"(\n{additional_sql.get_sql_statement(insert_n_tabs_to_each_line=insert_n_tabs_to_each_line)}\n)"


        usg_sql_dict = {
            'sql_kwh_usg_delrec_w_signed_val'      : sql_kwh_usg_delrec_w_signed_val, 
            'sql_kwh_usg_delrec_net'               : sql_kwh_usg_delrec_net, 
            'sql_kwh_usg_total_only'               : sql_kwh_usg_total_only, 
            'sql_kwh_usg_delrec_net_union_total_0' : sql_kwh_usg_delrec_net_union_total_0, 
            'sql_kwh_usg_delrec_net_union_total'   : sql_kwh_usg_delrec_net_union_total, 
            'additional_sql'                       : additional_sql
        }
        if return_statement:
            return AMINonVee_SQL.assemble_net_kwh_usage_sql_statement(
                usg_sql_dict               = usg_sql_dict, 
                final_table_alias          = final_table_alias, 
                insert_n_tabs_to_each_line = insert_n_tabs_to_each_line, 
                prepend_with_to_stmnt      = prepend_with_to_stmnt
            )
        else:
            return usg_sql_dict
            
            
    #****************************************************************************************************
    @staticmethod
    def build_sql_usg_for_df_with_search_time_window(
        cols_of_interest, 
        df_with_search_time_window, 
        build_sql_function=None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        df_args                   = {}
    ):
        r"""
        NOTE: python won't let me set the default value to build_sql_function=AMINonVee_SQL.build_sql_usg.
              Thus, the workaround is to set the default to None, then set it to 
              AMINonVee_SQL.build_sql_usg if it is None
        
        TODO: Investigate what an appropriate default value for max_n_prem_per_outg would be....
              Probably something large, like 1000.  But, should find outages with huge numbers of customers
              affected to find the right number.
        """
        #--------------------------------------------------
        if build_sql_function is None:
            build_sql_function=AMINonVee_SQL.build_sql_usg
        #--------------------------------------------------
        if build_sql_function_kwargs is None:
            build_sql_function_kwargs = {}
        build_sql_function_kwargs['serialnumber_col']         = build_sql_function_kwargs.get('serialnumber_col', 'serialnumber')
        build_sql_function_kwargs['aep_srvc_qlty_idntfr_col'] = build_sql_function_kwargs.get('aep_srvc_qlty_idntfr_col', 'aep_srvc_qlty_idntfr')
        build_sql_function_kwargs['from_table_alias']         = build_sql_function_kwargs.get('from_table_alias', 'un_rin')
        build_sql_function_kwargs['datetime_col']             = build_sql_function_kwargs.get('datetime_col', 'starttimeperiod')
        build_sql_function_kwargs['datetime_pattern']         = build_sql_function_kwargs.get('datetime_pattern', 
                                                                                              r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        build_sql_function_kwargs['date_col']                 = build_sql_function_kwargs.get('date_col', 'aep_usage_dt')
        #--------------------------------------------------
        return AMI_SQL.build_sql_ami_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_with_search_time_window, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            df_args                    = df_args
        )
        
    
    #****************************************************************************************************
    @staticmethod            
    def build_sql_usg_for_outages(
        cols_of_interest          , 
        df_outage                 , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        df_args                   = {}
    ):
        return AMINonVee_SQL.build_sql_usg_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_outage, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            df_args                    = df_args
        )
        
    #****************************************************************************************************
    @staticmethod
    def build_sql_usg_for_no_outages(
        cols_of_interest          , 
        df_mp_no_outg             , 
        build_sql_function        = None, 
        build_sql_function_kwargs = {}, 
        sql_alias_base            = 'USG_', 
        max_n_prem_per_outg       = None, 
        join_mp_args              = False, 
        date_only                 = False, 
        df_args                   = {}
    ):
        r"""
        """
        #--------------------------------------------------
        if df_args is None:
            df_args = {}
        df_args['t_search_min_col']    = df_args.get('t_search_min_col', 'start_date')
        df_args['t_search_max_col']    = df_args.get('t_search_max_col', 'end_date')
        df_args['addtnl_groupby_cols'] = df_args.get('addtnl_groupby_cols', None)
        df_args['mapping_to_ami']      = df_args.get('mapping_to_ami', DfToSqlMap(df_col='prem_nb', kwarg='premise_nbs', sql_col='aep_premise_nb'))
        df_args['mapping_to_mp']       = df_args.get('mapping_to_mp', {'state_cd':'states'})
        #-----

        return AMINonVee_SQL.build_sql_usg_for_df_with_search_time_window(
            cols_of_interest           = cols_of_interest, 
            df_with_search_time_window = df_mp_no_outg, 
            build_sql_function         = build_sql_function, 
            build_sql_function_kwargs  = build_sql_function_kwargs, 
            sql_alias_base             = sql_alias_base, 
            max_n_prem_per_outg        = max_n_prem_per_outg, 
            join_mp_args               = join_mp_args, 
            date_only                  = date_only, 
            df_args                    = df_args
        )
        
        
    #****************************************************************************************************
    @staticmethod
    def build_sql_usg_distinct_fields(
        date_0       , 
        date_1       , 
        fields       = ['serialnumber'], 
        are_datetime = False, 
        **kwargs
        ):
        r"""
        Intended use: find unique serial numbers recording some sort of end event between date_0 and date_1
        Default fields=['serialnumber'], but could also use, e.g., fields=['serialnumber', 'aep_premise_nb']
        """
        #-------------------------
        # First, make any necessary adjustments to kwargs
        kwargs['schema_name']      = kwargs.get('schema_name', 'usage_nonvee')
        kwargs['table_name']       = kwargs.get('table_name', 'reading_ivl_nonvee')
        kwargs['from_table_alias'] = kwargs.get('from_table_alias', 'USG')       
        kwargs['date_col']         = kwargs.get('date_col', 'aep_usage_dt')
        kwargs['datetime_col']     = kwargs.get('datetime_col', 'starttimeperiod')
        kwargs['datetime_pattern'] = kwargs.get('datetime_pattern', 
                                                 r"([0-9]{4}-[0-9]{2}-[0-9]{2})T([0-9]{2}:[0-9]{2}:[0-9]{2}).*")
        #-------------------------
        if are_datetime:
            kwargs['datetime_range'] = [date_0, date_1]
        else:
            kwargs['date_range'] = [date_0, date_1]
        cols_of_interest = fields
        #-------------------------
        sql = AMINonVee_SQL.build_sql_usg(
            cols_of_interest = cols_of_interest, 
            **kwargs
        )
        sql.sql_select.select_distinct=True
        #-------------------------
        return sql