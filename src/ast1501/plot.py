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
Plotting functions
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb

## Plotting
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm

## Scipy
from scipy import stats

# ----------------------------------------------------------------------------

# Staircase plotting function
def staircase_plot(data,
                   data_labels,
                   fig = None,
                   ax = None):
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
    
    Returns:
        fig, ax (matplotlib figure and axis object) - The matplotlib figure and 
            axis objects.
    '''
    
    # Figure out the number of variables
    n_var = len( data[0,:] )
    
    # Check if the figure was provided
    if fig == None:
        fig = plt.figure( figsize=( int(n_var+3), int(n_var+3) ) )
    ##fi
    if ax == None:
        axs = fig.subplots( nrows=n_var, ncols=n_var )
    ##fi
    
    # Double loop over the number of variables
    for i in range(n_var): # Indexes along columns (down)
        for j in range(n_var): # Indexes along rows (across)
            
            # Maxima and minima
            xmin = np.min(data[:,j])
            xmax = np.max(data[:,j])
            ymin = np.min(data[:,i])
            ymax = np.max(data[:,i])
            
            # If this is an upper-right plot it is a duplicate, remove it
            if j > i:
                axs[i,j].set_axis_off()
                continue
            ##fi
                
            # If the two indices are equal just make a histogram of the data
            if j == i: 
                
                # Make and plot the kernel
                kernel = stats.gaussian_kde( data[:,i] )
                kernel_grid = np.linspace( np.min(data[:,i]), np.max(data[:,i]), 1000 )
                kernel_evaluate = kernel.evaluate( kernel_grid )
                axs[i,j].plot( kernel_grid, kernel_evaluate, color='Black' )
                
                # Decorate
                axs[i,j].set_xlim( np.min(data[:,i]), np.max(data[:,i]) )
                axs[i,j].tick_params(labelleft='off', labelright='on')
                axs[i,j].set_ylabel('KDE')
                axs[i,j].yaxis.set_label_position('right')
            ##fi
                
            # If the two indices are not equal make a scatter plot
            if j < i:
                
                # Make the 2D gaussian KDE
                xx, yy = np.mgrid[ xmin:xmax:100j, ymin:ymax:100j ]
                positions = np.vstack([ xx.ravel(), yy.ravel() ])
                values = np.vstack([ data[:,j], data[:,i] ])
                kernel = stats.gaussian_kde( values )
                kernel_evaluate = np.reshape( kernel(positions).T, xx.shape )
                
                # Make contours out of the KDE
                cfset = axs[i,j].contourf(xx, yy, kernel_evaluate, cmap='Blues')
                cset = axs[i,j].contour(xx, yy, kernel_evaluate, colors='Black')
                
                # Decorate
                axs[i,j].set_xlim( xmin, xmax)
                axs[i,j].set_ylim( ymin, ymax)
            ##fi
            
            # Make X axis
            if i == n_var-1:
                axs[i,j].set_xlabel( data_labels[j] )
            else:
                axs[i,j].tick_params(labelbottom='off')   
            ##ie 
            
            # Make Y axis    
            if j == 0 and i!=0:
                axs[i,j].set_ylabel( data_labels[i] )
            else:
                axs[i,j].tick_params(labelleft='off')  
            ##ie
                
    return fig, axs            
#def

# ----------------------------------------------------------------------------
