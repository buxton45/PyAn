#!/usr/bin/env python

r"""
Holds CustomEncoder class.  See CustomEncoder.CustomEncoder for more information.
"""

__author__ = "Jesse Buxton"
__email__ = "jbuxton@aep.com"
__status__ = "Personal"

#--------------------------------------------------
import Utilities_config
import sys, os
import re

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype, is_timedelta64_dtype
from scipy import stats
import datetime
import time
from natsort import natsorted, ns
import copy
import json
#import jsonpickle
#--------------------------------------------------
import Utilities
#--------------------------------------------------
sys.path.insert(0, Utilities_config.get_sql_aids_dir())
from SQLElement import SQLElement

class CustomEncoder(json.JSONEncoder):
    r"""
    class CustomEncoder documentation
    extend the json.JSONEncoder class
    """
    # overload method default
    def default(self, obj):
        # Match all the types you want to handle in your converter
        if hasattr(obj, 'to_json_value'):
            return obj.to_json_value()
        if isinstance(obj, type):
            return obj.__name__
        elif callable(obj): # detects if object is a function
            return obj.__name__
        elif isinstance(obj, pd.DataFrame):
            #return obj.to_json()
            return obj.index.tolist()
        elif isinstance(obj, SQLElement):
            return obj.get_dict()
        # Call the default method for other types
        return json.JSONEncoder.default(self, obj)
    
    
class CustomWriter:
    r"""
    class CustomWriter documentation
    """
    
    @staticmethod
    def convert_dict_keys(dct):
        # For writing of JSON, dict keys must be str, int, float, bool or None
        # If this is not the case, return to_json_dict_key method.
        # If method does not exist, return 'ERROR' string as key
        return_dict =  {}
        for k,v in dct.items():
            #---------------
            if Utilities.is_object_one_of_types(k, [str, int, float, bool]) or k is None:
                key_i = k
            else:
                if hasattr(k, 'to_json_dict_key'):
                    key_i = k.to_json_dict_key()
                else:
                    key_i = 'ERROR'
            #---------------
            if isinstance(v, dict):
                val_i = CustomWriter.convert_dict_keys(v)
            else:
                val_i = v
            #---------------
            assert(key_i not in return_dict)
            return_dict[key_i] = val_i
            #---------------
        return return_dict
        
    @staticmethod
    def output_dict_to_json(
        output_path, 
        output_dict
    ):
        output_dict = CustomWriter.convert_dict_keys(output_dict)
        with open(output_path, "w") as outfile:
            json.dump(output_dict, outfile, indent=2, cls=CustomEncoder)
    
# To read in, one can simply do the following:
#    with open(path, 'r') as f:
#        summary_dict = json.load(f)

    
# #  Opposite procedure with JSONDecoder. 
# #  It’s a little bit different here, we have to pass a method to object_hook in the constructor. 
# #  We will implement our custom deserializations conditions in this method.  
# class JSONDecoder(json.JSONDecoder):
    # def __init__(self, *args, **kwargs):
        # json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    # def object_hook(self, obj):
        # # handle your custom classes
        # if isinstance(obj, dict):
            # if "value1" in obj and "value2" in obj:
                # print(obj)
                # return MyClass(obj.get("value_1"), obj.get("value_2"))

        # # handling the resolution of nested objects
        # if isinstance(obj, dict):
            # for key in list(obj):
                # obj[key] = self.object_hook(obj[key])

            # return obj

        # if isinstance(obj, list):
            # for i in range(0, len(obj)):
                # obj[i] = self.object_hook(obj[i])

            # return obj

        # # resolving simple strings objects
        # # dates
        # if isinstance(obj, str):
            # obj = self._extract_date(obj)

        # return obj