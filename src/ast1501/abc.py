# ----------------------------------------------------------------------------
#
# TITLE - abc.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS: 
# LinearModel
# LinearModel2
# LinearModelSolution
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Functions for ABC
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy

## Plotting
from matplotlib import pyplot as plt

## Astropy
from astropy import units as apu

## galpy
from galpy import orbit
from galpy import potential
from galpy import df
from galpy import actionAngle
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv
from galpy.util import multi

## Scipy
from scipy.stats import binned_statistic
from scipy import stats
from scipy import interpolate

## AST 1501
from . import df as ast1501_df

# ----------------------------------------------------------------------------

def interpolate_bar_model(R,phi,bar_model_data): 
    '''interpolate_bar_model:
    
    Function to interpolate the velocity profile of a DF-evaluated bar model
    for arbitrarily grids
    
    Args:
        R - (float array) New R locations
        phi - (float array) New phi locations
        bar_model_data (float array) Bar model DF data
        
    Returns:
        vR - (float array) Interpolated vR at (R,phi)
        vT - (float array) Interpolated vT at (R,phi)
    '''
    # Load data
    X_bin_cents = bar_model_data[:,2].astype(float)
    Y_bin_cents = bar_model_data[:,3].astype(float)
    vR_values = bar_model_data[:,4].astype(float)
    vT_values = bar_model_data[:,6].astype(float)
    X_cur = R*np.cos(phi)
    Y_cur = R*np.sin(phi)
        
    # Now make the interpolation grid
    interpolation_kind = 'cubic'
    interpolation_function = 'griddata'
    
    if interpolation_function == 'interp2d':
        vR_interpolator = interpolate.interp2d(X_bin_cents, Y_bin_cents, vR_values, 
            kind=interpolation_kind)
        vT_interpolator = interpolate.interp2d(X_bin_cents, Y_bin_cents, vT_values, 
            kind=interpolation_kind)
            
        # Interpolate and return
        vR_interp = vR_interpolator(X_cur, Y_cur)
        vT_interp = vT_interpolator(X_cur, Y_cur)
        
        return vR_interp[0,:], vT_interp[0,:]
    ##fi
    
    if interpolation_function == 'griddata':
        # Try griddata
        bin_cents = np.array([X_bin_cents,Y_bin_cents]).T
        cur_cents = np.array([X_cur,Y_cur]).T
        
        vR_interp = interpolate.griddata(bin_cents, vR_values, cur_cents,
            method=interpolation_kind)
        vT_interp = interpolate.griddata(bin_cents, vT_values, cur_cents,
            method=interpolation_kind)
        return vR_interp, vT_interp
    ##fi
    
#def