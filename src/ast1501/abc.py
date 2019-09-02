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
    ax.step( bin_edges[:-1], hist, **plot_kws )
    
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

# Staircase plotting function
def staircase_plot_kernel(data,
                          data_labels,
                          plot_ci = False,
                          plot_median = False,
                          fig = None,
                          ax = None
                          ):
    '''
    staircase_plot:
    
    Take in N variables in M samples and plot their correlations.
    
    Args:
        data (mxn array) - The input data. The first axis should be the sample 
            number and the second axis should be the variable
        data_labels (length n array) - The variable labels
        fig (matplotlib figure) - The input figure to plot on. If None then make 
            one [None].
        ax (matplotlib axis) - The input axis to plot on. If None then make one 
            [None].
        plot_median (boolean) - Plot the median of the sample?
        plot_ci = (boolean) - Plot the 68% confidence intervals?
    
    Returns:
        fig, ax (matplotlib figure and axis object) - The matplotlib figure and 
            axis objects.
    '''
    
    # Figure out the number of variables
    n_var = len( data[0,:] )
    
    kde_bw = 0.75
    
    # Check if the figure was provided
    if fig == None:
        fig = plt.figure( figsize=( int(n_var+4), int(n_var+4) ) )
    ##fi
    if ax == None:
        axs = fig.subplots( nrows=n_var, ncols=n_var )
    ##fi
    
    # Double loop over the number of variables
    for i in range(n_var): # Indexes along columns (down)
        for j in range(n_var): # Indexes along rows (across)
            
            # Maxima and minima
            xmin = np.min(data[:,j]) - 0.1*np.median(data[:,j])
            xmax = np.max(data[:,j]) + 0.1*np.median(data[:,j])
            ymin = np.min(data[:,i]) - 0.1*np.median(data[:,i])
            ymax = np.max(data[:,i]) + 0.1*np.median(data[:,i])
            
            # If this is an upper-right plot its a duplicate, remove it
            if j > i:
                axs[i,j].set_axis_off()
                continue
                
            # If the two indices are equal just make a histogram of the data
            if j == i: 
                
                # Make and plot the kernel
                kernel = stats.gaussian_kde( data[:,i], bw_method=kde_bw )
                kernel_grid = np.linspace( xmin, xmax, 1000 )
                kernel_evaluate = kernel.evaluate( kernel_grid )
                kernel_normalize = np.max(kernel_evaluate)
                axs[i,j].plot( kernel_grid, kernel_evaluate/kernel_normalize, color='Black' ) 
                axs[i,j].set_ylabel('KDE')
                
                if plot_median:
                    plot_median = np.median( data[:,i] )
                    axs[i,j].axvline(plot_median, linestyle='dashed', color='Black')
                if plot_ci:
                    plot_upper_68_ci = np.sort( data[:,i] )[ int((0.5+0.68/2)*len(data[:,i])) ]
                    plot_lower_68_ci = np.sort( data[:,i] )[ int((0.5-0.68/2)*len(data[:,i])) ]
                    axs[i,j].axvline(plot_upper_68_ci, linestyle='dotted', color='Black')
                    axs[i,j].axvline(plot_lower_68_ci, linestyle='dotted', color='Black')
                ##fi
                    
                # Decorate
                axs[i,j].set_xlim( xmin, xmax)
                axs[i,j].tick_params(labelleft='off', labelright='on')
                axs[i,j].yaxis.set_label_position('right')
                
            # If the two indices are not equal make a scatter plot
            if j < i:
                # axs[i,j].scatter( data[:,j], data[:,i], s=4, color='Black', 
                #     alpha=0.3 )
                
                xx, yy = np.mgrid[ xmin:xmax:100j, ymin:ymax:100j ]
                positions = np.vstack([ xx.ravel(), yy.ravel() ])
                values = np.vstack([ data[:,j], data[:,i] ])
                kernel = stats.gaussian_kde( values, bw_method=kde_bw )
                kernel_evaluate = np.reshape( kernel(positions).T, xx.shape )

                cfset = axs[i,j].contourf(xx, yy, kernel_evaluate, cmap='Blues',
                    alpha=0.75)
                cset = axs[i,j].contour(xx, yy, kernel_evaluate, colors='Black',
                    linewidths=0.5)
                
                if plot_median:
                    plot_median_x = np.median( data[:,j] )
                    axs[i,j].axvline(plot_median_x, linestyle='dashed', color='Black')
                    plot_median_y = np.median( data[:,i] )
                    axs[i,j].axhline(plot_median_y, linestyle='dashed', color='Black')
                ##fi
                
                if plot_ci:
                    plot_upper_68_ci_x = np.sort( data[:,j] )[ int((0.5+0.68/2)*len(data[:,j])) ]
                    plot_lower_68_ci_x = np.sort( data[:,j] )[ int((0.5-0.68/2)*len(data[:,j])) ]
                    axs[i,j].axvline(plot_upper_68_ci_x, linestyle='dotted', color='Black')
                    axs[i,j].axvline(plot_lower_68_ci_x, linestyle='dotted', color='Black')
                    
                    plot_upper_68_ci_y = np.sort( data[:,i] )[ int((0.5+0.68/2)*len(data[:,i])) ]
                    plot_lower_68_ci_y = np.sort( data[:,i] )[ int((0.5-0.68/2)*len(data[:,i])) ]
                    axs[i,j].axhline(plot_upper_68_ci_y, linestyle='dotted', color='Black')
                    axs[i,j].axhline(plot_lower_68_ci_y, linestyle='dotted', color='Black')
                ##fi
                
                axs[i,j].set_xlim( xmin, xmax)
                axs[i,j].set_ylim( ymin, ymax)
            
            
            # Make X axis
            if i == n_var-1:
                axs[i,j].set_xlabel( data_labels[j] )
            else:
                axs[i,j].tick_params(labelbottom='off')    
            
            # Make Y axis    
            if j == 0 and i!=0:
                axs[i,j].set_ylabel( data_labels[i] )
            else:
                axs[i,j].tick_params(labelleft='off')  
                
    return fig, axs            
#def