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
import yaml

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

# ----------------------------------------------------------------------------

def load_abc_params(filename):
    '''load_abc_params:
    
    Will take a .yaml parameter file, load it into a dictionary, manipulate 
    the results, and then return the dictionary for loading into the 
    local namespace.
    
    Args:
        filename (string) - Path to the parameter file
    
    Returns:
        parameters (dict) - Parameter dictionary to be loaded into namespace
    '''
    
    # Load into dictionary
    with open(filename, 'r') as stream:
        try:
            parameter_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        ##te
    ##wi
    
    # Now parse the dictionary to evaluate it properly
    for key in parameter_dict:
        
        # numpy objects are converted into a string, so evaluate them
        if type(parameter_dict[key]) is str:
            if 'np.' in parameter_dict[key]:
                parameter_dict[key] = eval(parameter_dict[key])
            ##fi
        ##fi
        
        # Also check through arrays
        if type(parameter_dict[key]) is list:
            for i in range(len(parameter_dict[key])):
                if type(parameter_dict[key][i]) is str:
                    if 'np.' in parameter_dict[key][i]:
                        parameter_dict[key][i] = eval(parameter_dict[key][i])
                    ##fi
                ##fi
            ###i
        ##fi
        
    #key
    
    return parameter_dict
#def

# ----------------------------------------------------------------------------

def plot_posterior_histogram(data,bins=20,lims=None,plot_kws={},fig=None,ax=None):
    '''plot_posterior_histogram:
    
    Plot the posterior distribution in a histogram style
    
    Args:
        data (float array) - Data to plot
        lims (2-array) - Limits of the data
        fig (matplotlib figure) - Optional figure object [None]
        ax (matplotlib axis) - Optional axis object [None]
        
    Returns:
        fig (matplotlib figure) - Figure object
        ax (matplotlib axis) - Axis object    
    '''
    
    # Format for this figure is single pane
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ##fi
    
    # Make the histogram and normalize so it's a PDF
    hist, bin_edges = np.histogram( data, bins=bins, range=lims )
    hist = hist / len(data)
    
    # Make a step plot
    ax.step( hist, bin_edges, **plot_kws )
    
    return fig,ax
#def
    
# ----------------------------------------------------------------------------

def plot_posterior_discrete(data, fig=None, ax=None, plot_kws={}):
    '''plot_posterior_discrete:
    
    Plot the posterior of a discrete quantity (like bar properties) using 
    straight lines and points instead of a histogram
    
    Args:
        data (float array) - Data to plot
        fig (matplotlib figure) - Optional figure object [None]
        ax (matplotlib axis) - Optional axis object [None]
    
    Returns:
        fig (matplotlib figure) - Figure object
        ax (matplotlib axis) - Axis object    
    '''
    
    # Format for this figure is single pane
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ##fi 
    
    # Figure out unique data points and total number of data points
    n_data_points = len(data)
    unique_data_points = np.sort(np.unique(data))
    n_unique_data_points = len(unique_data_points)
    
    # Loop over the 
    unique_data_density = np.zeros_like(unique_data_points)
    for i in range( n_unique_data_points ):
        this_point = unique_data_points[i]
        unique_data_density[i] = len(np.where( data == this_point )[0]) /\
                                    n_unique_data_points
    ###i
    
    ax.plot( unique_data_points, unique_data_density, **plot_kws)
    
    return fig, ax
#def

# ----------------------------------------------------------------------------
