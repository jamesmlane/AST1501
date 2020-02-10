# ----------------------------------------------------------------------------
#
# TITLE - coordinates.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
# 1 - calculate_galactic_azimuth
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Functions for unit conversions for the AST1501 project
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb
# import copy
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
# from galpy.util import bovy_conversion as gpconv
# from galpy.util import bovy_plot as gpplot

# ----------------------------------------------------------------------------

def calculate_galactic_azimuth( gc_x,
                                gc_y,
                                cw=True,
                                lh=True,
                                negative_domain=True
                    ):
    '''
    calculate_galactic_azimuth:
    
    Calculate a consistent galactic azimuth using galactocentric X and Y 
    coordinates. The 0 point lies along the GC-Sun line. The domain is [-pi,pi]
    
    Args:
        gc_x (float array) - Galactocentric X
        gc_y (float array) - Galactocentric Y
        cw (bool) - Should galactocentric azimuth increase clockwise (i.e. with 
            galactic rotation) w.r.t. the galactic north pole? [True]
        lh (bool) - Are the X and Y coordinates given in a left-handed system 
            such that the sun lies towards positive X w.r.t. the galactic center
            and positive Y lies in the direction of galactic rotationself.
    
    Returns:
        gc_az (float array) - Galactocentric azimuth angle
    '''
    
    # Do the conversion assuming the system is left-handed
    if lh == False:
        gc_x *= -1
    ##fi
    
    # Make the azimuth coordinate. Account for CW or CCW coordinates
    if cw:
        gc_az = np.arctan2(gc_y,gc_x)
    else:
        gc_az = np.arctan2(-gc_y,gc_x)
    ##ie
    
    return gc_az    
#def

# ----------------------------------------------------------------------------
