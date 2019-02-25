# ----------------------------------------------------------------------------
#
# TITLE - util.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
# - stround
# - 
# 
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Utility functions for the AST 1501 project.
'''
__author__ = "James Lane"

### Imports

## None

# ----------------------------------------------------------------------------

def stround(num,nplace):
    '''stround:
    
    Return a number, rounded to a certain number of decimal points, as a string
    
    Args:
        num (float) - Number to be rounded
        nplace (int) - Number of decimal places to round        
    
    Returns:
        rounded_str (string) - rounded number
    '''
    return round(num,nplace)
#def
    
# ----------------------------------------------------------------------------

def df_evaluator_write_params(logfile,params,param_names):
    '''df_evaluator_write_params:
    
    Write the parameters used in the DF evaluation to file.
    
    Args:
        logfile (open text file)
        params (N-array) A list of parameters to write to file.
        param_names (N-array) A list of parameter names to write to file.
        
    
    '''
    logfile.write('Parameters\n')
    logfile.write('==========\n\n')
    
    n_params = len(params)
    for i in range(n_params):    
        logfile.write(param_names[i]+': ')
        if isinstance( params[i], list ):
            for p in params[i]:
                logfile.write(str(p)+', ')
            ###p
        else:
            logfile.write(str(params[i])+', ')
        ##ie
        logfile.write('\n')
    logfile.write('==========\n')
    return logfile
#def
