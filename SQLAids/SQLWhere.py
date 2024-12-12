#!/usr/bin/env python

import sys, os
import copy
import numpy as np
import Utilities_sql
from SQLElement import SQLElement
from SQLElementsCollection import SQLElementsCollection

#****************************************************************************************************************************************************
class SQLWhereElement(SQLElement):
    #TODO Maybe set default comparison_operator  '='?
    def __init__(self, field_desc,  
                 value, comparison_operator, needs_quotes=True, 
                 table_alias_prefix=None, 
                 is_timestamp=False):
        r"""
        TODO for now, dict_init not set up
        
        is_timestamp:
          If the element is a timestamp, this should be set to True.
          NOTE: FOR NOW, FUNCTIONALITY ONLY REALLY AVAILABLE FOR add_where_statement
          Reason is because Athena is annoying:
            Doesn't work: CAST(regexp_replace(starttimeperiod, '(\\d{4}-\\d{2}-\\d{2})T(\\d{2}:\\d{2}:\\d{2}).*', '$1 $2') AS TIMESTAMP) BETWEEN '2021-01-01 00:00:00' AND '2021-01-01 12:00:00'
            Does work:    CAST(regexp_replace(starttimeperiod, '(\\d{4}-\\d{2}-\\d{2})T(\\d{2}:\\d{2}:\\d{2}).*', '$1 $2') AS TIMESTAMP) BETWEEN TIMESTAMP '2021-01-01 00:00:00' AND TIMESTAMP '2021-01-01 12:00:00'
        """
        assert(isinstance(field_desc,str))
        # First, call base class's __init__ method
        super().__init__(field_desc=field_desc, 
                         alias=None, table_alias_prefix=table_alias_prefix, 
                         comparison_operator=comparison_operator, value=value)
        self.needs_quotes = needs_quotes
        self.comparison_operator = comparison_operator.upper()
        self.is_timestamp = is_timestamp
        # NOTE: If comparison_operator is 'BETWEEN', then value must be a list or tuple with 2 elements
        if self.comparison_operator == 'BETWEEN':
            assert((isinstance(self.value, list) or isinstance(self.value, tuple)) and 
                   len(self.value)==2)
            
    def get_where_element_string(self, include_table_alias_prefix=True):
        # NOTE: If comparison_operator is 'BETWEEN', then value must be a list or tuple with 2 elements
        #-----
        if self.needs_quotes:
            value_fmt = "'{}'"
        else:
            value_fmt = "{}"
        #-----
        field_desc = self.get_field_desc(include_alias=False, include_table_alias_prefix=include_table_alias_prefix)
        elmnt_strng = f"{field_desc} {self.comparison_operator} "
        if self.comparison_operator == 'BETWEEN':
            assert((isinstance(self.value, list) or isinstance(self.value, tuple)) and 
                   len(self.value)==2)
            if not self.is_timestamp:
                elmnt_strng += f"{value_fmt} AND {value_fmt}".format(self.value[0], self.value[1])
            else:
                elmnt_strng += f"TIMESTAMP {value_fmt} AND TIMESTAMP {value_fmt}".format(self.value[0], self.value[1])
        else:
            if not self.is_timestamp:
                elmnt_strng += f"{value_fmt}".format(self.value)
            else:
                elmnt_strng += f"TIMESTAMP {value_fmt}".format(self.value)
        #--------------------------------------
        return elmnt_strng

#****************************************************************************************************************************************************        
class CombinedSQLWhereElements:
    def __init__(self, collection_dict, idxs_to_combine, join_operator):
        self.elements = [collection_dict[idx] for idx in idxs_to_combine]
        self.join_operator=join_operator
        
    def get_combined_where_elements_string(self, include_table_alias_prefix=True):
        return_str = "(\n"
        for i,element in enumerate(self.elements):
            assert(isinstance(element, SQLWhereElement) or isinstance(element, CombinedSQLWhereElements))
            if i>0:
                return_str += f" {self.join_operator} \n"
            if isinstance(element, SQLWhereElement):
                return_str += Utilities_sql.prepend_tabs_to_each_line(
                    element.get_where_element_string(include_table_alias_prefix=include_table_alias_prefix), 
                    n_tabs_to_prepend=1
                )
            else:
                return_str += Utilities_sql.prepend_tabs_to_each_line(
                    element.get_combined_where_elements_string(include_table_alias_prefix=include_table_alias_prefix), 
                    n_tabs_to_prepend=1
                )
        return_str += '\n)'
        return return_str
        
    def __eq__(self, other):
        if isinstance(other, CombinedSQLWhereElements):
            if(
                self.join_operator == other.join_operator and
                set(self.elements) == set(other.elements)
            ):
                return True
            else:
                return False
        else:
            return NotImplemented

    def __hash__(self):
        return hash((
            self.join_operator, 
            tuple(self.elements)
        ))

    def approx_eq(
        self, 
        other, 
        comp_table_alias_prefix=False, 
        **kwargs
    ):
        r"""
        Purpose of kwargs is simply so this can be used interchangable with SQLElement.approx_eq.
        The inclusion of kwargs allows the inputs to be the same (so, e.g., a SQLElement or 
          CombinedSQLWhereElements object can be used in 
          SQLElementsCollection.find_idx_of_approx_element_in_collection_dict)
        """
        #-------------------------
        str_1 = self.get_combined_where_elements_string(include_table_alias_prefix=comp_table_alias_prefix)
        if isinstance(other, CombinedSQLWhereElements):
            str_2 = other.get_combined_where_elements_string(include_table_alias_prefix=comp_table_alias_prefix)
        else:
            str_2 = other.field_desc
        #-------------------------
        return Utilities_sql.are_strings_equal(str_1, str_2)

#****************************************************************************************************************************************************        
class SQLWhere(SQLElementsCollection):
    def __init__(self, 
                 field_descs=None, 
                 idxs=None, run_check=False):
        necessary_keys = ['field_desc', 'comparison_operator', 'value']
        if field_descs is not None:
            assert(isinstance(field_descs, list) or isinstance(field_descs, tuple))
            for field_desc in field_descs:
                assert(isinstance(field_desc, SQLElement) or isinstance(field_desc, dict))
                if isinstance(field_desc, dict):
                    assert(all(x in field_desc for x in necessary_keys))
        # First, call base class's __init__ method
        super().__init__(field_descs=field_descs, 
                         global_table_alias_prefix=None, 
                         idxs=idxs, run_check=run_check, SQLElementType=SQLWhereElement, 
                         other_acceptable_SQLElementTypes=[CombinedSQLWhereElements])
        #-------------------------
        # NOTE: addtnl_info is just an empty dict which can be used for whatever
        #       I developed it to house field_descs_dict for use with combine_where_elements 
        #       functionality in AMI_SQL, AMIEndEvents_SQL, etc., but, as stated above, it can
        #       be used for whatever
        self.addtnl_info = dict()
        
    def add_where_statement(
            self, 
            field_desc          , 
            comparison_operator = '', 
            value               = '', 
            needs_quotes        = True, 
            table_alias_prefix  = None, 
            is_timestamp        = False, 
            idx                 = None, 
            run_check           = False
        ):
        if isinstance(field_desc, SQLWhereElement):
            element = field_desc
        else:
            element = SQLWhereElement(field_desc=field_desc, 
                                      value=value, comparison_operator=comparison_operator, needs_quotes=needs_quotes, 
                                      table_alias_prefix=table_alias_prefix, 
                                      is_timestamp=is_timestamp)
        self.insert_single_element_to_collection_at_idx(element=element, idx=idx, run_check=run_check)
        
    def add_where_statements(
            self, 
            field_descs , 
            idxs        = None, 
            run_check   = False
        ):
        # field_descs should be a list with elements of type SQLWhereElement or dict.
        # See SQLElementsCollection.insert_to_collection_at_idx for more information
        assert(isinstance(field_descs, list) or isinstance(field_descs, tuple))
        necessary_keys = ['field_desc', 'comparison_operator', 'value']
        for field_desc in field_descs:
            #assert(isinstance(field_desc, SQLWhereElement) or isinstance(field_desc, dict))
            assert(Utilities_sql.is_object_one_of_types(field_desc, self.all_acceptable_SQLElementTypes+[dict]))
            if isinstance(field_desc, dict):
                assert(all(x in field_desc for x in necessary_keys))
        self.insert_to_collection_at_idx(field_descs=field_descs, 
                                         global_table_alias_prefix=None, 
                                         idxs=idxs, run_check=run_check)
                                         
    def add_where_statement_equality_or_in(
            sql_where            , 
            field_desc           , 
            value                , 
            needs_quotes         = True, 
            table_alias_prefix   = None, 
            idx                  = None, 
            run_check            = False, 
            allowed_types_single = [str, int]
        ):
        is_input_list = Utilities_sql.is_object_one_of_types(value, [list, tuple, np.ndarray])
        if is_input_list and len(value)==1:
            is_input_list=False
            value=value[0]
        #-------------------------
        if is_input_list:
            comparison_operator='IN'
            value=f'({Utilities_sql.join_list(value, quotes_needed=needs_quotes)})'
            needs_quotes=False
        else:
            comparison_operator='='
            value=value
            needs_quotes=needs_quotes
        #-------------------------
        sql_where.add_where_statement(field_desc=field_desc, 
                                      comparison_operator=comparison_operator, 
                                      value=value, 
                                      needs_quotes=needs_quotes, 
                                      table_alias_prefix=table_alias_prefix, 
                                      idx=idx, 
                                      run_check=run_check)
        return sql_where

    def remove_where_statement_at_idx(self, idx, run_check=False):
        self.remove_single_element_from_collection_at_idx(idx=idx, run_check=run_check)
        
    #------------------------------------------------------------------------------------------------
    def standardize_to_where_element(input_args, field_desc=None):
        f"""
        Takes input_args, which can have various forms as described below, and creates and returns
        a SQLWhereElement objects.

        WARNING: Be careful when using input_args as type list

        input_args:
          - input_args = str:
              - Only works when field_desc is not None
              - Returns SQLWhereElement(field_desc=field_desc, value=input_args)

          - input_args = list/tuple:
              - if field_desc is None:
                  - Returns SQLWhereElement(field_desc = input_args[0],   
                                            value = input_args[1], 
                                            comparison_operator = input_args[2] if exists otherwise default '=', 
                                            needs_quotes        = input_args[3] if exists otherwise default True, 
                                            table_alias_prefix  = input_args[4] if exists otherwise default None)
              - if field_desc is not None:
                  - Returns SQLWhereElement(field_desc = field_desc,   
                                            value = input_args[0], 
                                            comparison_operator = input_args[1] if exists otherwise default '=', 
                                            needs_quotes        = input_args[2] if exists otherwise default True, 
                                            table_alias_prefix  = input_args[3] if exists otherwise default None)

          - input_args = dict:
              - Returns SQLWhereElement(field_desc = input_args[field_desc] if field_desc is None else field_desc,   
                                        value = input_args['value'], 
                                        comparison_operator = input_args.get('comparison_operator', '='), 
                                        needs_quotes        = input_args.get('needs_quotes', True), 
                                        table_alias_prefix  = input_args.get('table_alias_prefix', None))

        """
        return_where_elm = None
        assert(isinstance(input_args, str) or 
               isinstance(input_args, list) or isinstance(input_args, tuple) or 
               isinstance(input_args, dict) or 
               isinstance(input_args, SQLWhereElement))
        if isinstance(input_args, str):
            assert(field_desc is not None)
            return_where_elm = SQLWhereElement(field_desc=field_desc,   
                                               value=input_args, 
                                               comparison_operator='=', 
                                               needs_quotes=True, 
                                               table_alias_prefix=None)
        elif isinstance(input_args, list) or isinstance(input_args, tuple):
            if field_desc is None:
                shift = 0
                field_desc = input_args[0]
            else:
                shift = 1
            #---------------
            assert(len(input_args)>=2-shift and len(input_args)<=5-shift)
            value      = input_args[1-shift]
            comparison_operator = '='
            needs_quotes        = True
            table_alias_prefix  = None
            #-------------------------
            if len(input_args)>2-shift:
                comparison_operator = input_args[2-shift]
            if len(input_args)>3-shift:
                needs_quotes        = input_args[3-shift]
            if len(input_args)>4-shift:
                table_alias_prefix  = input_args[4-shift]           
            #-------------------------
            return_where_elm = SQLWhereElement(field_desc=field_desc,   
                                               value=value, 
                                               comparison_operator=comparison_operator, 
                                               needs_quotes=needs_quotes, 
                                               table_alias_prefix=table_alias_prefix)
        elif isinstance(input_args, dict):
            #-----
            if field_desc is None:
                assert('field_desc' in input_args)
                field_desc = input_args['field_desc']
            #-----
            assert('value' in input_args)
            value      = input_args['value']
            #-----
            comparison_operator = input_args.get('comparison_operator', '=')
            needs_quotes        = input_args.get('needs_quotes', True)
            table_alias_prefix  = input_args.get('table_alias_prefix', None)
            #-------------------------
            return_where_elm = SQLWhereElement(field_desc=field_desc,   
                                               value=value, 
                                               comparison_operator=comparison_operator, 
                                               needs_quotes=needs_quotes, 
                                               table_alias_prefix=table_alias_prefix)

        elif isinstance(input_args, SQLWhereElement):
            return_where_elm = copy.deepcopy(input_args)
            if field_desc is not None:
                return_where_elm.field_desc = field_desc
        else:
            assert(0)
        return return_where_elm    
    
    #------------------------------------------------------------------------------------------------
    def combine_where_elements_OLD(
        self, 
        idxs_to_combine, 
        join_operator, 
        close_gaps_in_keys=True, 
        return_idx=False
    ):
        r"""
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        NEW VERSION CREATED 20240201.  This old version should be deleted
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        join_operator is should be a string for joining the elements
          e.g. 'AND' or 'OR'
        Setting close_gaps_in_keys=False would be useful if e.g. one wants to first combine
          elements 2 and 3, then combine that result with 5.  One could either do
            i.  self.combine_where_elements([2,3], join_operator='AND', close_gaps_in_keys=False)
                self.combine_where_elements([2,5], join_operator='OR', close_gaps_in_keys=True)

            ii. self.combine_where_elements([2,3], join_operator='AND', close_gaps_in_keys=True)
                self.combine_where_elements([2,4], join_operator='OR', close_gaps_in_keys=True)
        """
        #-------------------------
        combined_elements = CombinedSQLWhereElements(
            collection_dict=self.collection_dict, 
            idxs_to_combine=idxs_to_combine, 
            join_operator=join_operator
        )
        idx_to_replace = min(idxs_to_combine)
        idxs_to_delete = [x for x in idxs_to_combine if x!=idx_to_replace]
        self.collection_dict[idx_to_replace] = combined_elements
        for idx in idxs_to_delete:
            del self.collection_dict[idx]
        if close_gaps_in_keys:
            SQLElementsCollection.close_gaps_in_dict_keys(self.collection_dict)
        #-------------------------
        if return_idx:
            # dict inversion possible thanks to __hash__ definintion in CombinedSQLWhereElements
            comb_idx = Utilities_sql.invert_dict(self.collection_dict)[combined_elements]
            return comb_idx
            
    def combine_where_elements(
        self, 
        idxs_to_combine, 
        join_operator, 
        close_gaps_in_keys=True, 
        return_idx=False
    ):
        r"""
        Combine multiple where elements using join_operator.
        If any of the elements to be combined are equal, duplicates are removed.
            This is possible thanks to the __hash__ definintions in SQLElement and CombinedSQLWhereElements

        join_operator:
            should be a string for joining the elements
            e.g. 'AND' or 'OR'

        close_gaps_in_keys:
            Setting close_gaps_in_keys=False would be useful if e.g. one wants to first combine
              elements 2 and 3, then combine that result with 5.  One could either do
                i.  self.combine_where_elements([2,3], join_operator='AND', close_gaps_in_keys=False)
                    self.combine_where_elements([2,5], join_operator='OR', close_gaps_in_keys=True)

                ii. self.combine_where_elements([2,3], join_operator='AND', close_gaps_in_keys=True)
                    self.combine_where_elements([2,4], join_operator='OR', close_gaps_in_keys=True)    
        """
        #-------------------------
        els_to_combine = [self.collection_dict[idx] for idx in idxs_to_combine]
        # Sanity check
        assert(Utilities_sql.are_all_list_elements_one_of_types(lst=els_to_combine, types=[SQLWhereElement, CombinedSQLWhereElements]))
        #-------------------------
        # Make sure they are no repeat entries through the set operation below.
        # NOTE: The set operation may change the order of els_to_combine, but this shouldn't matter, as any decent query optimizer
        #       will rearrange the WHERE clause anyway.
        els_to_combine = list(set(els_to_combine))
        #-------------------------
        assert(len(els_to_combine)>0)
        if len(els_to_combine)==1:
            combined_elements = els_to_combine[0]
        else:
            # Build CombinedSQLWhereElements, which accepts a dict and keys to grab from dict as inputs
            combined_elements = CombinedSQLWhereElements(
                collection_dict = {i:x for i,x in enumerate(els_to_combine)}, 
                idxs_to_combine = list(range(len(els_to_combine))), 
                join_operator   = join_operator
            )
        #-------------------------
        # Find the element to replace with combined_elements and those to remove completely
        idx_to_replace = min(idxs_to_combine)
        idxs_to_delete = [x for x in idxs_to_combine if x!=idx_to_replace]
        self.collection_dict[idx_to_replace] = combined_elements
        for idx in idxs_to_delete:
            del self.collection_dict[idx]
        if close_gaps_in_keys:
            SQLElementsCollection.close_gaps_in_dict_keys(self.collection_dict)
        #-------------------------
        if return_idx:
            # dict inversion possible thanks to __hash__ definintion in CombinedSQLWhereElements
            comb_idx = Utilities_sql.invert_dict(self.collection_dict)[combined_elements]
            return comb_idx
            
    def combine_last_n_where_elements(self, last_n, join_operator, close_gaps_in_keys=True):
        # Combine the last n where elements
        # Made for convenience because I often add things to end and then combine,
        #   so this makes that procedure easier
        n_where_elms = self.get_n_elements()
        idxs_to_combine = list(range(n_where_elms-last_n, n_where_elms))
        assert(idxs_to_combine[0]>-1 and idxs_to_combine[-1]<n_where_elms)
        self.combine_where_elements(idxs_to_combine=idxs_to_combine, 
                                    join_operator=join_operator, 
                                    close_gaps_in_keys=close_gaps_in_keys)
                                    
    #------------------------------------------------------------------------------------------------
    def combine_where_elements_smart(
        self, 
        idxs_to_combine, 
        join_operator, 
        close_gaps_in_keys=True, 
        return_idx=False
    ):
        r"""
        If the elements to combine are all simple SQLWhereElement objects, then this will behave exactly as SQLWhere.combine_where_elements.
        If any of the elements to combine are of type CombinedSQLWhereElements, the the behavior MAY be different.
        If join_operator is consistent with the elements of type CombinedSQLWhereElements, then all will be joined together in a single 
          CombinedSQLWhereElements element, removing any duplicates shared.
        If one (or more) of the elements of type CombinedSQLWhereElements has a join_operator which does not equal that supplied to this function, then
          this will again behave exactly as SQLWhere.combine_where_elements, i.e., the elements to combine will be simply combined using the
          join_operator input.
        ----------
        This may be confusing, so hopefully the following examples will illustrate the points.
        -----
        Assume the elements to combine (i.e., [collection_dict[idx] for idx in idxs_to_combine]) are, conceptually:
            SQLWhereElement: 'Where a=1'
            CombinedSQLWhereElements: '(Where a=1 AND b=2)'
            CombinedSQLWhereElements: '(Where c=3 AND d=4)'
        Assume join_operator = 'AND'
        ==> Behavior unique from SQLWhere.combine_where_elements!
        ==> Output: CombinedSQLWhereElements: '(Where a=1 AND b=2 AND c=3 AND d=4)'

        -----
        Assume the elements to combine (i.e., [collection_dict[idx] for idx in idxs_to_combine]) are, conceptually:
            SQLWhereElement: 'Where a=1'
            CombinedSQLWhereElements: '(Where a=1 AND b=2)'
            CombinedSQLWhereElements: '(Where c=3 OR d=4)'
        Assume join_operator = 'AND'
        ==> Behaves just as SQLWhere.combine_where_elements!
        ==> Output: CombinedSQLWhereElements: (
                                               'Where a=1' AND
                                               (a=1 AND b=2) AND 
                                               (c=3 OR d=4)
                                              )

        -----
        Assume the elements to combine (i.e., [collection_dict[idx] for idx in idxs_to_combine]) are, conceptually:
            SQLWhereElement: 'Where a=1'
            CombinedSQLWhereElements: '(Where a=1 AND b=2)'
            CombinedSQLWhereElements: '(Where c=3 OR d=4)'
        Assume join_operator = 'OR'
        ==> Behaves just as SQLWhere.combine_where_elements!
        ==> Output: CombinedSQLWhereElements: (
                                               'Where a=1' OR
                                               (a=1 AND b=2) OR 
                                               (c=3 OR d=4)
                                              )

        """
        #-------------------------
        els_to_combine = [self.collection_dict[idx] for idx in idxs_to_combine]
        # Sanity check
        assert(Utilities_sql.are_all_list_elements_one_of_types(lst=els_to_combine, types=[SQLWhereElement, CombinedSQLWhereElements]))
        #-------------------------
        # If the elements to combine are all simple SQLWhereElement objects, 
        #   then behave exactly as SQLWhere.combine_where_elements.
        if all([isinstance(x, SQLWhereElement) for x in els_to_combine]):
            comb_idx = self.combine_where_elements(
                idxs_to_combine    = idxs_to_combine, 
                join_operator      = join_operator, 
                close_gaps_in_keys = close_gaps_in_keys, 
                return_idx         = True
            )
        else:
            # If one (or more) of the elements of type CombinedSQLWhereElements has a join_operator which does not equal 
            #   that supplied to this function, then this will again behave exactly as SQLWhere.combine_where_elements
            join_operators = []
            for el_i in els_to_combine:
                if isinstance(el_i, CombinedSQLWhereElements):
                    join_operators.append(el_i.join_operator)
            join_operators = list(set(join_operators))
            assert(len(join_operators)>0)
            #-----
            # If join_operators has more than one element, it is impossible for all the be consistent.
            # The only way for all the be consistent is is len(join_operators)==1 and join_operators[0]==join_operators
            if len(join_operators)>1 or join_operators[0]!=join_operator:
                # No common join operator for all, so return SQLWhere.combine_where_elements
                comb_idx = self.combine_where_elements(
                    idxs_to_combine    = idxs_to_combine, 
                    join_operator      = join_operator, 
                    close_gaps_in_keys = close_gaps_in_keys, 
                    return_idx         = True
                )
            else:
                els_to_combine_fnl = []
                for el_i in els_to_combine:
                    if isinstance(el_i, SQLWhereElement):
                        els_to_combine_fnl.append(el_i)
                    else:
                        assert(isinstance(el_i, CombinedSQLWhereElements)) #unnecessary
                        els_to_combine_fnl.extend(el_i.elements)
                # Make sure they are no repeat entries through the set operation below.
                # NOTE: This is possible thanks to the definition of __hash__ in SQLElement
                els_to_combine_fnl = list(set(els_to_combine_fnl))

                # Build CombinedSQLWhereElements, which accepts a dict and keys to grab from dict as inputs
                combined_elements = CombinedSQLWhereElements(
                    collection_dict = {i:x for i,x in enumerate(els_to_combine_fnl)}, 
                    idxs_to_combine = list(range(len(els_to_combine_fnl))), 
                    join_operator   = join_operator
                )
                #-------------------------
                # Find the element to replace with combined_elements and those to remove completely
                idx_to_replace = min(idxs_to_combine)
                idxs_to_delete = [x for x in idxs_to_combine if x!=idx_to_replace]
                self.collection_dict[idx_to_replace] = combined_elements
                for idx in idxs_to_delete:
                    del self.collection_dict[idx]
                if close_gaps_in_keys:
                    SQLElementsCollection.close_gaps_in_dict_keys(self.collection_dict)
                #-------------------------
                # dict inversion possible thanks to __hash__ definintion in SQLElement and CombinedSQLWhereElements
                comb_idx = Utilities_sql.invert_dict(self.collection_dict)[combined_elements]
                #-------------------------
        if return_idx:
            return comb_idx    
    
    
    #------------------------------------------------------------------------------------------------
    def change_comparison_operator_of_element_at_idx(self, idx, new_comparison_operator):
        self.collection_dict[idx].comparison_operator = new_comparison_operator

    def change_comparison_operator_of_element(self, element_idntfr, new_comparison_operator, find_kwargs=None):
        r"""
        element_idntfr can be of type SQLWhereElement, dict, or str
          - element_idntfr of type SQLWhereElement
              - Attributes for comparison grabbed from object
          - element_idntfr of type dict:
              - It should have keys and values appropriate for building a SQLWhereElement
                - i.e., it needs keys field_desc, value, and comparison_operator (and can also accept
                  needs_quotes and table_alias_prefix)
              - SQLWhereElement built from dict and then attributes for comparison grabbed from object
          - element_idntfr of type str:
              - Input interpreted as field_desc
              - SQLWhereElement built as SQLWhereElement(field_desc=element_idntfr, 
                                                         value='', 
                                                         comparison_operator='')
              - comp_comparison_operator and comp_value set to False
              - Attributes for comparison grabbed from object
              
        find_kwargs can be any arguments (aside from element) accepted by 
          SQLElementsCollection.find_idx_of_approx_element_in_collection_dict
          - e.g., can call self.change_comparison_operator_of_element(SQLWhereElement(field_desc='trsf_pole_nb', 
                                                                                      value="('1','2','3', '4')", 
                                                                                      comparison_operator ='IN'), 
                                                                      'NOT IN', 
                                                                      find_kwargs = dict(comp_value=True))
        """
        assert(Utilities_sql.is_object_one_of_types(element_idntfr, [SQLWhereElement, dict, str]))
        #-----
        default_find_kwargs = dict(assert_max_one=True, 
                                   comp_alias=False, 
                                   comp_table_alias_prefix=False, 
                                   comp_comparison_operator=False, 
                                   comp_value=False)
        if find_kwargs is None:
            find_kwargs = default_find_kwargs
        else:
            find_kwargs = Utilities_sql.supplement_dict_with_default_values(find_kwargs, default_find_kwargs)
            
        # NOTE: Changing comparison_operator attribute, so comp_comparison_operator should ALWAYS
        #       be set to False
        find_kwargs['comp_comparison_operator'] = False
        #-----
        if isinstance(element_idntfr, SQLWhereElement):
            sql_elm = element_idntfr
        elif isinstance(element_idntfr, dict):
            sql_elm = SQLWhereElement(**element_idntfr)
        elif isinstance(element_idntfr, str):
            sql_elm = SQLWhereElement(field_desc=element_idntfr, 
                                      value='', 
                                      comparison_operator='')
            find_kwargs['comp_comparison_operator'] = False
            find_kwargs['comp_value'] = False
        else:
            assert(0)
        #-----
        found_idx = self.find_idx_of_approx_element_in_collection_dict(element=sql_elm, **find_kwargs)
        if found_idx>-1:
            self.change_comparison_operator_of_element_at_idx(idx=found_idx, 
                                                              new_comparison_operator=new_comparison_operator)        
    #------------------------------------------------------------------------------------------------
    def get_statement_string(self, include_table_alias_prefix=True):
        # If first line ==> begin with WHERE
        # else          ==> being with AND
        #
        # If not last line ==> end with comma (',')
        # else             ==> end with nothing
        #---------------
        _ = self.check_collection_dict(enforce_pass=True)
        #---------------
        sql = ""
        self.build_collection_list()
        for idx,element in enumerate(self.collection):
            assert(isinstance(element, SQLWhereElement) or isinstance(element, CombinedSQLWhereElements))
            line = ""
            if idx==0:
                line = 'WHERE '
            else:
                line = '\nAND   '
            if isinstance(element, SQLWhereElement):
                line += f"{element.get_where_element_string(include_table_alias_prefix=include_table_alias_prefix)}"
            else:
                line += element.get_combined_where_elements_string(include_table_alias_prefix=include_table_alias_prefix) 
            sql += line
        return sql
        
    def print_statement_string(self, include_table_alias_prefix=True):
        stmnt_str = self.get_statement_string(include_table_alias_prefix=include_table_alias_prefix)
        print(stmnt_str)
        
    def print(self, include_table_alias_prefix=True):
        self.print_statement_string(include_table_alias_prefix=include_table_alias_prefix)