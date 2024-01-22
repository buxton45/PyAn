#!/usr/bin/env python

import os
import sys
import glob
import re
from pathlib import Path

import xml.etree.ElementTree as ET
import xml.parsers.expat
from xml.dom import minidom

import Utilities

#--------------------------------------------------------------------
def print_sub_demo():
    sample_str = r'C:\Users\s346557\Documents\Learning\LocalData\sample_outages_full\outg_rec_nb_11738944\my_usage_2018_q4.csv'
    print(f'sample_str = {sample_str}\n')

    # ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT_2018_q4.csv
    print('-'*50)
    print(r'Desired functionality: ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT_2018_q4.csv')
    print("\tpattern = r'my_usage'")
    print("\trepl = 'NEWTEXT'")
    print("\tCommand = re.sub(pattern, repl, sample_str)")
    pattern = r'my_usage'
    repl = 'NEWTEXT'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str)}\n')

    # ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT
    print('-'*50)
    print(r'Desired functionality: ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT')
    print("\tpattern = r'my_usage_.*.csv'")
    print("\trepl = 'NEWTEXT'")
    print("\tCommand = re.sub(pattern, repl, sample_str)")
    pattern = r'my_usage_.*.csv'
    repl = 'NEWTEXT'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str)}\n')

    # ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT.csv
    print('-'*50)
    print(r'Desired functionality: ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT.csv')
    print("\tpattern = r'my_usage_.*(.csv)'")
    print("\trepl = r'NEWTEXT\1'")
    print("\tCommand = re.sub(pattern, repl, sample_str)")
    pattern = r'my_usage_.*(.csv)'
    repl = r'NEWTEXT\1'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str)}\n')

    # ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT.csv
    print('-'*50)
    print(r'Desired functionality: ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT.csv (version 2)')
    print("\tpattern = r'(my_usage_.*)(.csv)'")
    print("\trepl = r'NEWTEXT\2'")
    print("\tCommand = re.sub(pattern, repl, sample_str)")
    pattern = r'(my_usage_.*)(.csv)'
    repl = r'NEWTEXT\2'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str)}\n')

    # ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT_2018_q4.csv
    print('-'*50)
    print(r'Desired functionality: ...\my_usage_2018_q4.csv  --->  ...\NEWTEXT_2018_q4.csv (version 2)')
    print("\tpattern = r'(my_usage)(_.*)(.csv)'")
    print("\trepl = r'NEWTEXT\2'")
    print("\tCommand = re.sub(pattern, repl, sample_str)")
    pattern = r'(my_usage)(_.*)(.csv)'
    repl = r'NEWTEXT\2'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str)}\n')

    # Replace all occurrences of out
    print('-'*50)
    print(r'Desired functionality: Replace all occurrences of out')
    print("\tpattern = 'out'")
    print("\trepl = 'NEXTEXT'")
    print("\tCommand = re.sub(pattern, repl, sample_str)")
    pattern = 'out'
    repl = 'NEXTEXT'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str)}\n')

    # Replace all occurrences of out
    print('-'*50)
    print(r'Desired functionality: Replace all occurrences of out (version 2)')
    print("\tpattern = 'OUT'")
    print("\trepl = 'NEXTEXT'")
    print("\tCommand = re.sub(pattern, repl, sample_str, flags=re.IGNORECASE)")
    pattern = 'OUT'
    repl = 'NEXTEXT'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str, flags=re.IGNORECASE)}\n')

    # Replace first occurrence of out
    print('-'*50)
    print(r'Desired functionality: Replace first occurrence of out')
    print("\tpattern = 'out'")
    print("\trepl = 'NEXTEXT'")
    print("\tCommand = re.sub(pattern, repl, sample_str, count=1)")
    pattern = 'out'
    repl = 'NEXTEXT'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str, count=1)}\n')

    # Replace last occurrence out out
    print('-'*50)
    print(r'Desired functionality: Replace last occurrence out out')
    print("\tpattern = r'(.*)out([^.]*)'")
    print("\trepl = r'\1NEWTEXT\2'")
    print("\tCommand = re.sub(pattern, repl, sample_str)")
    print(
    """
    NOTE:
    \tSpecial characters lose their special meaning inside sets.
    \tSo, within the [] the . means just a dot. And the leading ^ means "anything but ...".
    \tTherefore, [^.]* matches zero or more non-dots.
    
    NOTE 2:
    \t Using a greedy quantifier and a capture group
    """
    )
    pattern = r'(.*)out([^.]*)'
    repl = r'\1NEWTEXT\2'
    print(f'Input:  {sample_str}')
    print(f'Output: {re.sub(pattern, repl, sample_str)}\n')
    
    
    print('*'*100)
    print('In a loop, it would be better to compile the regular expression first')
    
    # Replace last occurrence out out
    print('-'*50)
    print(r'Desired functionality: Replace last occurrence out out')
    print("\tpattern = r'(.*)out([^.]*)'")
    print("\trepl = r'\1NEWTEXT\2'")
    print(
    """
    \tCommand = 
    \t\tregex = re.compile(pattern)
    \t\tregex.sub(repl, sample_str)
    """
    )
    pattern = r'(.*)out([^.]*)'
    regex = re.compile(pattern)
    repl = r'\1NEWTEXT\2'
    print(f'Input:  {sample_str}')
    print(f'Output: {regex.sub(repl, sample_str)}\n')
    
    
    # Replace all occurrences of out
    print('-'*50)
    print(r'Desired functionality: Replace all occurrences of out (version 2)')
    print("\tpattern = 'OUT'")
    print("\trepl = 'NEXTEXT'")
    print(
    """
    \tCommand =
    \t\tregex = re.compile(pattern, flags=re.IGNORECASE))
    \t\tregex.sub(repl, sample_str)
    """
    )
    pattern = 'OUT'
    regex = re.compile(pattern, flags=re.IGNORECASE)
    repl = 'NEXTEXT'
    print(f'Input:  {sample_str}')
    print(f'Output: {regex.sub(repl, sample_str)}\n')