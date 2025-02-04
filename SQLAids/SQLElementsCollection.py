#!/usr/bin/env python

import sys, os
import Utilities_sql
import numpy as np
from SQLElement import SQLElement

class SQLElementsCollection():
    def __init__(self, 
                 field_descs=None, 
                 global_table_alias_prefix=None, 
                 idxs=None, run_check=False, SQLElementType=SQLElement, 
                 other_acceptable_SQLElementTypes=None):
        self.collection_dict = {}
        self.collection = []
        self.SQLElementType = SQLElementType
        self.other_acceptable_SQLElementTypes = other_acceptable_SQLElementTypes
        if self.other_acceptable_SQLElementTypes is None:
            self.other_acceptable_SQLElementTypes=[]
        assert(Utilities_sql.is_object_one_of_types(self.other_acceptable_SQLElementTypes, [list, tuple]))
        self.all_acceptable_SQLElementTypes = [self.SQLElementType] + self.other_acceptable_SQLElementTypes
        if field_descs is not None:
            self.insert_to_collection_at_idx(field_descs=field_descs, 
                                             global_table_alias_prefix=global_table_alias_prefix, 
                                             idxs=idxs, run_check=run_check)
        
    def __iter__(self):
        self.build_collection_list()
        return iter(self.collection)
    
    def __len__(self):
        return len(self.collection_dict)
    
    def __getitem__(self, item):
        return self.collection_dict[item]
        
    def get_n_elements(self):
        return len(self.collection_dict)
    
    @staticmethod
    def build_multiple_elements(field_descs, global_table_alias_prefix=None, SQLElementType=SQLElement):
        # field_descs should be a list with elements of type dict or string.
        #   If element is a dict, the keys should contain at a minimum 'field_desc'
        #     Full possible set of keys = ['field_desc', 'alias', 'table_alias_prefix', 
        #                                  'comparison_operator', 'value']
        #   If element is a string, the string should be a field_desc.
        #   All elements of type string will have:
        #     alias=comparison_operator=value=None
        #     table_alias_prefix = global_table_alias_prefix
        # If global_table_alias_prefix is not None, all elements without an included table_alias_prefix with have their
        #   value set equal to global_table_alias_prefix.
        #   NOTE: If one explicitly sets table_alias_prefix to None, global_table_alias_prefix will not change this!
        #
        # field_descs should be a list (or tuple)
        assert(isinstance(field_descs, list) or isinstance(field_descs, tuple))
        #-----
        field_descs_final = []
        for x in field_descs:
            # Orignal elements should be of type dict or string.
            assert(isinstance(x, dict) or isinstance(x, str))            
            if isinstance(x, dict):
                field_descs_final.append(x)
            else:
                field_descs_final.append(dict(field_desc=x, table_alias_prefix=global_table_alias_prefix))
        #-------------------------
        return_elements = []
        for field_desc in field_descs_final:
            assert('field_desc' in field_desc)
            if global_table_alias_prefix is not None:
                field_desc['table_alias_prefix'] = field_desc.get('table_alias_prefix', global_table_alias_prefix)
            element = SQLElementType(**field_desc)
            return_elements.append(element)
        return return_elements
    
    def build_collection_list(self):
        self.collection = [self.collection_dict[key] for key in sorted(self.collection_dict.keys())]
        
    def check_collection_dict(self, enforce_pass=True):
        # Elements should be labelled 0 to n_elements-1
        #     ==> number of elements should equal the unique number of keys
        # AND ==> number of elements should equal the maximum key value +1
        # If self.collection_dict is empty (e.g. when beginning to be built), return true
        if len(self.collection_dict)==0:
            return True
        pass_check = (len(self.collection_dict)==len(np.unique(list(self.collection_dict.keys()))) and 
                      len(self.collection_dict)==max(self.collection_dict.keys())+1)
        if enforce_pass:
            assert(pass_check)
        return pass_check
    
    def insert_single_element_to_collection_at_idx(self, element, idx=None, run_check=False):
        r"""
        As the name suggests, the inserts and element into the collection at position idx

        - element:
          - Currently has to be of type self.SQLElementType

        - idx:
          - Should be an int or None
          - If idx is None, put new element at end.
          - Otherwise, insert element at idx (which involves first shifting all with keys >= idx forward one)
        """
        #assert(isinstance(element, self.SQLElementType))
        assert(Utilities_sql.is_object_one_of_types(element, self.all_acceptable_SQLElementTypes))
        #-------------------------
        if run_check:
            _ = self.check_collection_dict(enforce_pass=True)
        #-------------------------
        if idx is None or idx==len(self.collection_dict):
            idx = len(self.collection_dict)
        else:
            # Need to shift all entries with key>=idx up one, then insert element at idx
            # Python doesn't like me doing this in one step with curly brackets {}.
            #   So instead I first make a list up tuples then conver to dict.
            self.collection_dict = dict([(k,v) if k<idx else ((k+1),v)
                                    for k,v in self.collection_dict.items()])
        assert(idx not in self.collection_dict)
        self.collection_dict[idx] = element
        #-------------------------
        if run_check:
            _ = self.check_collection_dict(enforce_pass=True)
        #-------------------------
        self.build_collection_list()
    
    def insert_elements_to_collection_at_idx(self, elements, idxs=None, run_check=False):
        # If idx is None, put new element at end.
        # Otherwise, insert element at idx
        #   Which involves first shifting all with keys >= idx forward one
        #-------------------------
        # If single SQLElement given, call insert_single_element_to_collection_at_idx
        #if isinstance(elements, self.SQLElementType):
        if Utilities_sql.is_object_one_of_types(elements, self.all_acceptable_SQLElementTypes):
            assert(idxs is None or isinstance(idxs, int))
            self.insert_single_element_to_collection_at_idx(elements, idxs, run_check)
            return
        #-------------------------
        assert(isinstance(elements, list) or isinstance(elements, tuple))
        for element in elements:
            #assert(isinstance(element, self.SQLElementType))
            assert(Utilities_sql.is_object_one_of_types(element, self.all_acceptable_SQLElementTypes))
        #-------------------------
        if run_check:
            _ = self.check_collection_dict(enforce_pass=True)
        #-------------------------
        # If idxs is a list (or tuple), one must assume that the entries are non-sequential
        # and must each be treated carefull with the insert_single_element_to_collection_at_idx method
        if isinstance(idxs, list) or isinstance(idxs, tuple):
            assert(len(idxs)==len(elements))
            for i,idx in enumerate(idxs):
                self.insert_single_element_to_collection_at_idx(elements[i], idx, run_check=False)
        else:
            # For both cases here, idxs will be seqeuential and one does not need to be as careful as above
            if idxs is None:
                idxs = list(range(len(self.collection_dict), len(self.collection_dict)+len(elements)))
            else:
                assert(isinstance(idxs, int))
                idxs = list(range(idxs, idxs+len(elements)))
            # Need to shift all entries with key>=idx up len(elements), then insert element at idxs
            # Python doesn't like me doing this in one step with curly brackets {}.
            #   So instead I first make a list up tuples then conver to dict.                
            self.collection_dict = dict([(k,v) if k<idxs[0] else ((k+len(idxs)),v)
                                    for k,v in self.collection_dict.items()])
            for i,idx in enumerate(idxs):
                assert(idx not in self.collection_dict)
                self.collection_dict[idx] = elements[i]
        #-------------------------
        if run_check:
            _ = self.check_collection_dict(enforce_pass=True)
        #-------------------------
        self.build_collection_list()
        #return self.collection_dict
        
    def insert_to_collection_at_idx(self, field_descs, 
                                    global_table_alias_prefix=None, 
                                    idxs=None, run_check=False):
        # field_descs should be a list with elements of type SQLElement, dict, or string.
        # See build_multiple_elements for more information
        #
        if idxs is None:
            idxs = list(range(len(self.collection_dict), len(self.collection_dict)+len(field_descs)))
        elif isinstance(idxs, int):
            idxs = list(range(idxs, idxs+len(field_descs)))
        else:
            assert(len(idxs)==len(field_descs))
        #-------------------------
        # field_descs should be a list (or tuple)
        assert(isinstance(field_descs, list) or isinstance(field_descs, tuple))
        elements_dict = {}
        # Ensure proper types of all elements, and add any elements of type SQLElement to elements_dict
        # NOTE: any elements (and corresponding index) found of type SQLElement will 
        #       need to be removed from original field_descs
        to_remove = []
        for i,x in enumerate(field_descs):
            # Orignal elements should be of type SQLElement, dict, or string.
            #assert(isinstance(x, self.SQLElementType) or isinstance(x, dict) or isinstance(x, str))
            assert(Utilities_sql.is_object_one_of_types(x, self.all_acceptable_SQLElementTypes+[dict, str]))
            assert(idxs[i] not in elements_dict)
            #if isinstance(x, self.SQLElementType):
            if Utilities_sql.is_object_one_of_types(x, self.all_acceptable_SQLElementTypes):
                elements_dict[idxs[i]] = x
                to_remove.append(i)
        # Remove SQLElements included above
        if len(to_remove)>0:
            # Need to remove the highest indices first
            # If started with lowest indices instead, this would affect the 
            #   index position of the higher elements, causing a mismatch between
            #   the values in to_remove and the desired element to be removed in idxs and field_descs
            to_remove = sorted(to_remove, reverse=True)
            for idx in to_remove:
                del idxs[idx]
                del field_descs[idx]
        #-------------------------
        # Build remaining SQLElements, and add them to elements_dict
        elements_to_add = SQLElementsCollection.build_multiple_elements(field_descs, global_table_alias_prefix, 
                                                                        SQLElementType=self.SQLElementType)
        assert(len(elements_to_add)==len(idxs))
        for i in range(len(elements_to_add)):
            assert(idxs[i] not in elements_dict)
            elements_dict[idxs[i]] = elements_to_add[i]
        #-------------------------
        srtd_idxs = sorted(elements_dict.keys())
        final_elements = [elements_dict[i] for i in srtd_idxs]
        final_idxs = srtd_idxs
        self.insert_elements_to_collection_at_idx(elements=final_elements, idxs=final_idxs, run_check=run_check)
        
    def remove_single_element_from_collection_at_idx(self, idx, run_check=False):
        # In future probably want to create another method that can find the element to be removed
        # from the element's attributes, instead of simply the index in the dict
        assert(idx in self.collection_dict)
        del self.collection_dict[idx]
        
        # Need to shift all entries with key>idx down one
        self.collection_dict = dict([(k,v) if k<idx else ((k-1),v)
                                     for k,v in self.collection_dict.items()])
        #-------------------------
        if run_check:
            _ = self.check_collection_dict(enforce_pass=True)
        #-------------------------
        self.build_collection_list()
        
    def remove_elements_from_collection_at_idxs(self, idxs, run_check=False):
        assert(isinstance(idxs, list) or isisntance(idxs, tuple))
        for idx in idxs:
            assert(idx in self.collection_dict)
            del self.collection_dict[idx]
        #-------------------------
        self.collection_dict = SQLElementsCollection.close_gaps_in_dict_keys(self.collection_dict)
        #-------------------------
        if run_check:
            _ = self.check_collection_dict(enforce_pass=True)
        #-------------------------
        self.build_collection_list()
        
    @staticmethod
    def close_gaps_in_dict_keys(collection_dict):
        # For this procedure to function properly, it must proceed
        # in ascending order.
        # For example: 
        #   consider new_key,old_key pairs = [0,0], [1,3], [2,4], [3,5], [4,10], [5,12]
        #   If one proceded first with changing old_key=12 to new_key=5, this would
        #   replace the value already present at idx=5, which is not what is desired.
        for new_key,old_key in enumerate(sorted(list(collection_dict.keys()))):
            if new_key==old_key:
                continue
            assert(new_key not in collection_dict)
            collection_dict[new_key] = collection_dict[old_key]
            del collection_dict[old_key]
        return collection_dict
        
    def close_gaps_in_collection_dict_keys(self, run_check=True):
        self.collection_dict = SQLElementsCollection.close_gaps_in_dict_keys(self.collection_dict)
        #-------------------------
        if run_check:
            _ = self.check_collection_dict(enforce_pass=True)
        #-------------------------
        self.build_collection_list()
        
    def find_idx_of_element_in_collection_dict(self, element, assert_max_one=True):
        found_idxs = []
        for idx,element_i in self.collection_dict.items():
            if element_i==element:
                found_idxs.append(idx)
        #---------------
        if assert_max_one:
            assert(len(found_idxs)<2)
        #---------------
        if len(found_idxs)==0:
            found_idxs = -1
        elif len(found_idxs)==1:
            found_idxs  = found_idxs[0]
        else:
            found_idxs = found_idxs
        #---------------
        return found_idxs
        
        
    def find_idx_of_approx_element_in_collection_dict(
        self                     , 
        element                  , 
        assert_max_one           = True, 
        comp_alias               = False, 
        comp_table_alias_prefix  = False, 
        comp_comparison_operator = False, 
        comp_value               = False
    ):
        #-------------------------
        # Allow for possibility of element being a simple string
        if isinstance(element, str):
            assert(comp_alias+comp_table_alias_prefix+comp_comparison_operator+comp_value==0)
            element = SQLElement(element)
        #-------------------------
        found_idxs = []
        for idx,element_i in self.collection_dict.items():
            if element_i.approx_eq(element, 
                                   comp_alias=comp_alias, 
                                   comp_table_alias_prefix=comp_table_alias_prefix, 
                                   comp_comparison_operator=comp_comparison_operator, 
                                   comp_value=comp_value):
                found_idxs.append(idx)
        #---------------
        if assert_max_one:
            assert(len(found_idxs)<2)
        #---------------
        if len(found_idxs)==0:
            found_idxs = -1
        elif len(found_idxs)==1:
            found_idxs  = found_idxs[0]
        else:
            found_idxs = found_idxs
        #---------------
        return found_idxs
        
    def find_and_remove_element_in_collection_dict(
        self           , 
        element        , 
        assert_max_one = True, 
        run_check      = True
    ):
        found_idxs = self.find_idx_of_element_in_collection_dict(
            element        = element, 
            assert_max_one = assert_max_one
        )
        if found_idxs==-1:
            return
        if isinstance(found_idxs, int):
            found_idxs = [found_idxs]
        self.remove_elements_from_collection_at_idxs(found_idxs, run_check=run_check)

    def find_and_remove_approx_element_in_collection_dict(
        self                     , 
        element                  , 
        assert_max_one           = True, 
        comp_alias               = False, 
        comp_table_alias_prefix  = False, 
        comp_comparison_operator = False, 
        comp_value               = False, 
        run_check                = True
    ):
        found_idxs = self.find_idx_of_approx_element_in_collection_dict(
            element                  = element, 
            assert_max_one           = assert_max_one, 
            comp_alias               = comp_alias, 
            comp_table_alias_prefix  = comp_table_alias_prefix, 
            comp_comparison_operator = comp_comparison_operator, 
            comp_value               = comp_value
        )
        if found_idxs==-1:
            return
        if isinstance(found_idxs, int):
            found_idxs = [found_idxs]
        self.remove_elements_from_collection_at_idxs(found_idxs, run_check=run_check)
        
        
    def get_elements_by_table_alias_prefix(self):
        table_alias_prefixes = list(set(x.table_alias_prefix for x in self))
        return_dict = {x:[] for x in table_alias_prefixes}
        for element in self:
            assert(element.table_alias_prefix in return_dict)
            return_dict[element.table_alias_prefix].append(element)
        return return_dict
       
    @staticmethod
    def extract_field_descs_from_elements_list(elements_list):
        r"""
        Returns a list of strings of the field_desc attribute from all elements in elements_list.
        The point is to return a simplified version of elements in elements_list.
        
        - Each element in elements_list can be of type str, dict, or SQLElement.
        - Originally developed for use in build_sql_outage_others_on_circuit to convert
            the input argument field_descs to cols_of_interest_usage for use in building 
            usage table one step above first aggregation.
        """
        assert(isinstance(elements_list, list) or isinstance(elements_list, tuple))
        return_list = []
        for x in elements_list:
            if isinstance(x, str):
                return_list.append(x)
            elif isinstance(x, dict):
                return_list.append(x[field_desc])
            elif isinstance(x, SQLElement):
                return_list.append(x.field_desc)
            else:
                assert(0)
        return return_list
        
        
    def swap_idxs(
        self, 
        idx_1,
        idx_2
    ):
        r"""
        """
        #-------------------------
        assert(idx_1 < len(self.collection_dict))
        assert(idx_2 < len(self.collection_dict))
        assert(idx_1 != idx_2)
        #-------------------------
        self.collection_dict[idx_1], self.collection_dict[idx_2] = self.collection_dict[idx_2], self.collection_dict[idx_1]