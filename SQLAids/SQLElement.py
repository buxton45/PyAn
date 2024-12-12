#!/usr/bin/env python

import sys, os
import Utilities_sql

class SQLElement:
    def __init__(self, field_desc, 
                 alias=None, table_alias_prefix=None, 
                 comparison_operator=None, value=None):
        # This is intended to be a base class which can be used multiple places throughout the SQL query.
        # Simplest case: element is field from table
        # Examples
        # field_desc = 'CI_NB', alias=None, table_alias_prefix = 'DOV'
        #   intended: DOV.CI_NB
        #
        # field_desc = 'DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24)', alias='DT_OFF_TS_FULL', table_alias_prefix = None
        #   intended: DOV.DT_ON_TS - DOV.STEP_DRTN_NB/(60*24) AS DT_OFF_TS_FULL
        #
        #
        if isinstance(field_desc, dict):
            self.dict_init(field_desc)
            return
        
        self.field_desc = field_desc
        self.alias = alias
        self.table_alias_prefix = table_alias_prefix
        self.comparison_operator = comparison_operator
        self.value = value
        
    def dict_init(self, field_desc_dict):
        assert('field_desc' in field_desc_dict)
        self.field_desc          = field_desc_dict['field_desc']
        self.alias               = field_desc_dict.get('alias', None)
        self.table_alias_prefix  = field_desc_dict.get('table_alias_prefix', None)
        self.comparison_operator = field_desc_dict.get('comparison_operator', None)
        self.value               = field_desc_dict.get('value', None)
        
    def get_describe_str(self):
        return (
            f"""
            field_desc = {self.field_desc}
            alias = {self.alias}
            table_alias_prefix = {self.table_alias_prefix}
            comparison_operator = {self.comparison_operator}
            value = {self.value}
            """
        )
        
    def describe(self):
        print((
            f"""
            field_desc = {self.field_desc}
            alias = {self.alias}
            table_alias_prefix = {self.table_alias_prefix}
            comparison_operator = {self.comparison_operator}
            value = {self.value}
            """
        ))
        
    # def __eq__(self, other):
        # if(
            # self.field_desc==other.field_desc and
            # self.alias==other.alias and
            # self.table_alias_prefix==other.table_alias_prefix and
            # self.comparison_operator==other.comparison_operator and
            # self.value==other.value
        # ):
            # return True
        # return False
        
    # def __hash__(self):
        # #
        # # By default, all instances of custom classes are hashable, 
        # # and therefore can be used as dictionary keys. 
        # # However, when __eq__() method is reimplemented, instances
        # # are no longer hashable.  This can be fixed by providing a
        # # __hash__() special method.
            # return hash(id(self))
            
    def __key(self):
        # NOTE: self.value can sometimes be a list, which is not hashable (I think this is due to
        #       the fact that lists are mutable).
        # ==> if self.value is a list, return tuple(self.list) instead
        value = self.value
        if isinstance(value, list):
            value = tuple(value)
        return (
            self.field_desc, 
            self.alias, 
            self.table_alias_prefix, 
            self.comparison_operator, 
            value
        )

    def __eq__(self, other):
        if isinstance(other, SQLElement):
            return self.__key() == other.__key()
        else:
            return NotImplemented

    def __hash__(self):
        # By default, all instances of custom classes are hashable, 
        # and therefore can be used as dictionary keys. 
        # However, when __eq__() method is reimplemented, instances
        # are no longer hashable.  This can be fixed by providing a
        # __hash__() special method.
        #
        # Instead of being unique like previously (i.e., return hash(id(self))), I want
        #   the hash between two equal SQLElement objects to be equal.
        # This property will allow me to use set operations on collections of SQLElement objects
        #return hash(id(self))
        return hash(self.__key())
        
    def approx_eq(self, other, 
                  comp_alias=False, 
                  comp_table_alias_prefix=False, 
                  comp_comparison_operator=False, 
                  comp_value=False, 
                  none_eq_empty_str=True, 
                  case_insensitive=False):
        # none_eq_empty_str allows None=='' to return as True
        #   See Utilities_sql.are_strings_equal for more details
        #
        #Go through and check equality for each input set to True
        # NOTE: For field_desc, none_eq_empty_str hardcoded to False, as the field_desc
        #       should always contain a value
        #---------------
        if not Utilities_sql.are_strings_equal(self.field_desc, other.field_desc, 
                                               none_eq_empty_str=False, 
                                               case_insensitive=case_insensitive):
            return False
        #---------------
        if (comp_alias and 
            not Utilities_sql.are_strings_equal(self.alias, other.alias, 
                                                none_eq_empty_str=none_eq_empty_str, 
                                                case_insensitive=case_insensitive)):
            return False
        #---------------
        if (comp_table_alias_prefix and 
            not Utilities_sql.are_strings_equal(self.table_alias_prefix, other.table_alias_prefix, 
                                                none_eq_empty_str=none_eq_empty_str, 
                                                case_insensitive=case_insensitive)):
            return False        
        #---------------
        if (comp_comparison_operator and 
            not Utilities_sql.are_strings_equal(self.comparison_operator, other.comparison_operator, 
                                                none_eq_empty_str=none_eq_empty_str, 
                                                case_insensitive=case_insensitive)):
            return False        
        #---------------
        if (comp_value and 
            not Utilities_sql.are_strings_equal(self.value, other.value, 
                                                none_eq_empty_str=none_eq_empty_str, 
                                                case_insensitive=case_insensitive)):
            return False
        #---------------
        return True
        
        
    def get_dict(self):
        return {
            'field_desc':self.field_desc, 
            'alias':self.alias, 
            'table_alias_prefix':self.table_alias_prefix, 
            'comparison_operator':self.comparison_operator, 
            'value':self.value
        }
        
    def to_json_dict_key(self):
        desc_dict = self.get_dict()
        return_key = ''
        for k,v in desc_dict.items():
            return_key += f'{k}:{v}; '
        return return_key
            
    def to_json_value(self):
        return self.get_dict()
        
    def get_field_desc(self, include_alias=True, include_table_alias_prefix=True):
        return_name = ''
        if include_table_alias_prefix and self.table_alias_prefix:
            return_name =  f"{self.table_alias_prefix}.{self.field_desc}"
        else:
            return_name = self.field_desc
        #-----
        if include_alias and self.alias is not None:
            return_name = f"{return_name} AS {self.alias}"
        #-----        
        return return_name
    
    def get_field_alias(self):
        return self.alias
    
    @staticmethod
    def split_field_desc(input_str):
        # Try to split the input_str into field_desc, table_alias_prefix, and alias
        field_desc = table_alias_prefix = alias = ''
        #-----
        found_pd = input_str.find('.')
        if found_pd>-1:
            table_alias_prefix = input_str[:found_pd]
            field_desc = input_str[found_pd+1:]
        else:
            field_desc = input_str
        #-----
        found_AS = field_desc.lower().find(' as ')
        if found_AS>-1:
            alias = field_desc[found_AS+len(' as '):]
            field_desc = field_desc[:found_AS]
        #-----
        return {'field_desc':field_desc, 
                'table_alias_prefix':table_alias_prefix, 
                'alias':alias}