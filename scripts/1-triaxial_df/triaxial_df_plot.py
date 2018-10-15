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
from matplotlib import pyplot as plt
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

## Keywords

# Assume exists
dir='./run5'

# ----------------------------------------------------------------------------

## Read the data

# .npy file
data_in = np.load(dir+'/results.npy')

# Unpack. Note: b is 1st index, phi is 2nd index
tri_b,tri_phi,vR_range,vT_range,df0_all,dfp_all = data_in

# Number of parameters
n_b = len(tri_b)
n_phi = len(tri_phi)

# Velocity deltas
if len(np.unique(np.diff(vR_range))) != 1:
    sys.exit('Error, non-singular delta vR')
if len(np.unique(np.diff(vT_range))) != 1:
    sys.exit('Error, non-singular delta vT')
##fi
delta_vR = np.unique(np.diff(vR_range))[0]
delta_vT = np.unique(np.diff(vT_range))[0]

# ----------------------------------------------------------------------------

## Plot the distribution function

fig = plt.figure( figsize=(10,10) )
# nrows is 1st index, ncols is 2nd index
axs = fig.subplots(nrows=n_b, ncols=n_phi)

# Reshape if only one of one parameter
if n_phi == 1:
    axs = axs.reshape((n_b,1))
##fi
if n_b == 1:
    axs = axs.reshape((1,n_phi))
##fi

# Loop over the b and phi values
for i in range( n_b ):
    for j in range( n_phi ):

        # Get the distribution function data
        dfp = df0_all[i,j]

        # Calculate the density of the DF
        densp = np.sum(dfp) * delta_vR * delta_vT

        # Make the image
        intimg = np.rot90( dfp/densp )

        # pdb.set_trace()
        img = axs[i,j].imshow(np.log10(intimg), interpolation='nearest',
                extent=[np.min(vR_range), np.max(vR_range), np.min(vT_range), np.max(vT_range)],
                cmap='Blues_r')

        axs[i,j].set_xlabel(r'$V_{R}$ [km/s]')
        axs[i,j].set_ylabel(r'$V_{\phi}$ [km/s]')
        axs[i,j].tick_params(direction='in', right='on', top='on')

    ###j
###i

fig.savefig(dir+'/dfp.pdf')
plt.close('all')






# ----------------------------------------------------------------------------
