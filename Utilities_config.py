#!/usr/bin/env python

import sys, os
import yaml
import re
import pathlib
from pathlib import Path
from importlib.machinery import SourceFileLoader


#--------------------------------------------------------------------
def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
        
def determine_aep_user_id(users_dir=r'C:\Users'):
    r"""
    Try to determine the AEP User Id.
    It does this simply by looking for a s#####... directory in C:\Users.
    If multiple found, ask user to choose one.
    If none found, notify user and CRASH.
    """
    #-------------------------
    # Make sure users_dir exists!
    assert(os.path.isdir(users_dir))

    # Get the directory contents, and keep on subdirectories (i.e., don't consider 
    #   any files living in users_dir)
    dir_contents = os.listdir(users_dir)
    subdirs = [x for x in dir_contents if os.path.isdir(os.path.join(users_dir, x))]

    # Find entries in subdirs which match a normal AEP User Id, i.e. 's' followed by some
    #  numbers, e.g., s123456.
    # These are candidate User Ids.  If only one found, return.
    regex_pattern = r's\d+'
    cand_ids = [x for x in subdirs if re.search(regex_pattern, x, flags=re.IGNORECASE)]
    #-----
    if len(cand_ids)==1:
        user_id = cand_ids[0]
        return user_id
    else:
        if len(cand_ids)==0:
            print('NO POTENTIAL USER IDS FOUND!\nCRASH IMMINENT')
            assert(0)
            #-----
            return 0
        else:
            print('MULTIPLE CANDIDATE USER IDS FOUND!\nPLEASE SELECT WHICH IS CORRECT:')
            for i,cand_id in enumerate(cand_ids):
                print(f'{i}: {cand_id}')
            correct_idx = input(f'Please select the correct entry number (0 to {len(cand_ids)-1}):\n')
            #-----
            correct_idx = int(correct_idx)
            assert(correct_idx>0 and correct_idx<len(cand_ids)-1)
            #-----
            user_id = cand_ids[correct_idx]
            return user_id           
        
def generate_initial_config_file(
    aep_user_id                  = None, 
    pwd_file_path                = None, 
    local_data_dir               = None, 
    create_local_data_dir_if_dne = True, 
    athena_prod_dsn              = 'Athena Prod', 
    athena_dev_dsn               = 'Athena Dev', 
    athena_qa_dsn                = 'Athena QA', 
    utldb01p_dsn                 = 'UTLDB01P', 
    eemsp_dsn                    = 'EEMSP', 
):
    r"""
    Generate the initial configuration file, containing the locations of the various directories, as well
    as other important information such as the user's AEP Id, and the location of the password file.
    
    aep_user_id:
        The user's AEP ID, e.g., 's123456'
        It is suggested that the user supply this, but one can also leave it as None, in which case 
          Utilities_config.determine_aep_user_id will attempt to determine the AEP ID.
          
    local_data_dir:
        A local directory where the user stores data (e.g., intermediate results, results of SQL queries, etc.)
        
    pwd_file_path:
        The path to a text file containing the user's AEP password(s).
        If pwd_file_path is None, it is assumed to be located in the Analysis directory with name 'pwd_file.txt'.
        This (together with the User's ID) are needed to run any SQL queries.
        NOTE: At one point, my Athena and Oracle passwords were different, which is why there is a 'Main' and 
              'Oracle' entry in the password file.
        The password file should be a simple text file with the following form:
            Main:MyPassword_123
            Oracle:MyPassword_123
          (where MyPassword_123 should be replaced by the user's password)
    """
    #-------------------------
    config_path = os.path.join(pathlib.Path(__file__).parent.resolve(), r'config.yaml')
    config_template_path = os.path.join(pathlib.Path(__file__).parent.resolve(), r'config_template_init.yaml')
    assert(os.path.exists(config_template_path))
    config = read_yaml(config_template_path)
    #-------------------------
    config['analysis_dir'] = str(pathlib.Path(__file__).parent.resolve())
    config['utilities_dir'] = os.path.join(config['analysis_dir'], 'Utilities')
    config['sql_aids_dir'] = os.path.join(config['analysis_dir'], 'SQLAids')
    #-------------------------
    if local_data_dir is None:
        print('Warning: No local_data_dir set.\nI suggest setting aside a local data directory, but one is not strictly necessary\n-----')
        print('To Continue without local_data_dir, enter 0')
        print('To create and/or set local_data_dir to default value, enter default.')
        print(f"\tdefault location = {os.path.join(Path(config['analysis_dir']).parent.resolve(), 'LocalData')}")
        print('To create and/or set local_data_dir to custom location, enter desired path (must be complete path!)')
        #-----
        response = input().strip()
        #-----
        if response=='0':
            config['local_data_dir'] = None #i.e., =local_data_dir, but this is more explicit
        elif response=='default':
            # By default, place the local_data_dir (named 'LocalData') directory at the same level (not inside of!) the analysis_dir
            config['local_data_dir'] = os.path.join(Path(config['analysis_dir']).parent.resolve(), 'LocalData')
        else:
            config['local_data_dir'] = response
    else:
        config['local_data_dir'] = local_data_dir
    #-----
    if(
        create_local_data_dir_if_dne and 
        not os.path.exists(config['local_data_dir'])
    ):
        os.makedirs(config['local_data_dir'])
    #-------------------------
    if pwd_file_path is None:
        config['pwd_file_path'] = os.path.join(config['analysis_dir'], 'pwd_file.txt')
    else:
        config['pwd_file_path'] = pwd_file_path
    #-------------------------
    if aep_user_id is None:
        aep_user_id = determine_aep_user_id()
    config['aep_user_id'] = aep_user_id
    #-------------------------
    config['athena_prod_dsn'] = athena_prod_dsn
    config['athena_dev_dsn']  = athena_dev_dsn
    config['athena_qa_dsn']   = athena_qa_dsn
    config['utldb01p_dsn']    = utldb01p_dsn
    config['eemsp_dsn']       = eemsp_dsn
    #-------------------------
    config['config_verified'] = False
    #-------------------------
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    #-------------------------
    check_config()
        
def get_config_entry(field):
    config_path = os.path.join(pathlib.Path(__file__).parent.resolve(), r'config.yaml')
    assert(os.path.exists(config_path))
    config = read_yaml(config_path)
    #-------------------------
    if not config['config_verified']:
        check_config()
    #-------------------------
    assert(field in config)
    return config[field]
        
def check_config():
    config_path = os.path.join(pathlib.Path(__file__).parent.resolve(), r'config.yaml')
    assert(os.path.exists(config_path))
    config = read_yaml(config_path)
    #-------------------------
    analysis_dir = config['analysis_dir']
    assert(os.path.exists(analysis_dir) and os.path.isdir(analysis_dir))
    #----------
    utilities_dir     = config['utilities_dir']
    assert(os.path.exists(utilities_dir) and os.path.isdir(utilities_dir))
    #----------
    sql_aids_dir      = config['sql_aids_dir']
    assert(os.path.exists(sql_aids_dir) and os.path.isdir(sql_aids_dir))
    #----------
    local_data_dir    = config['local_data_dir']
    if local_data_dir is not None:
        assert(os.path.exists(local_data_dir) and os.path.isdir(local_data_dir))
    #----------
    pwd_file_path     = config['pwd_file_path']
    if pwd_file_path is not None:
        assert(os.path.exists(pwd_file_path))
    #-------------------------
    # At this point, config file has been verified
    # Set the config_verified field to True
    config['config_verified'] = True
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

#--------------------------------------------------------------------
def get_analysis_dir():
    return get_config_entry('analysis_dir')
    
def get_utilities_dir():
    return get_config_entry('utilities_dir')
    
def get_sql_aids_dir():
    return get_config_entry('sql_aids_dir')

def get_local_data_dir():
    return get_config_entry('local_data_dir')
    
def get_pwd_file_path():
    return get_config_entry('pwd_file_path')
    
def get_aep_user_id():
    return get_config_entry('aep_user_id')
    
#--------------------------------------------------------------------
def get_dsn(db_name):
    r"""
    """
    #-------------------------
    acceptable_names = [
        'athena_prod', 
        'athena_dev', 
        'athena_qa', 
        'utldb01p', 
        'eemsp'
    ]
    assert(db_name in acceptable_names)
    #-------------------------
    return get_config_entry(db_name+'_dsn')
    
def get_athena_prod_dsn():
    return get_dsn(db_name = 'athena_prod')

def get_athena_dev_dsn():
    return get_dsn(db_name = 'athena_dev')

def get_athena_qa_dsn():
    return get_dsn(db_name = 'athena_qa')

def get_utldb01p_dsn():
    return get_dsn(db_name = 'utldb01p')

def get_eemsp_dsn():
    return get_dsn(db_name = 'eemsp')