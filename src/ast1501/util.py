# ----------------------------------------------------------------------------
#
# TITLE -
# AUTHOR - James Lane
# PROJECT -
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''

'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb
# import glob
# import subprocess

## Plotting
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib import colors
# from matplotlib import cm
# import aplpy

## Astropy
# from astropy.io import fits
# from astropy.coordinates import SkyCoord
# from astropy import table
# from astropy import units as u
# from astropy import wcs

## galpy
# from galpy import orbit
# from galpy import potential
# from galpy.util import bovy_coords as gpcoords

# ----------------------------------------------------------------------------

def stround(num,nplace):
    '''
    stround:
    
    Return a number, rounded to a certain number of decimal points, as a string
    
    Args:
        num (float) - Number to be rounded
        nplace (int) - Number of decimal places to round        
    
    Returns:
        rounded_str (string) - rounded number
    '''
    return round(num,nplace)
#def
    
