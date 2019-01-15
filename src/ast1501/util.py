# ----------------------------------------------------------------------------
#
# TITLE - util.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
# 1 - stround
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
    
