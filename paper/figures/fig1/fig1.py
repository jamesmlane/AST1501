# ----------------------------------------------------------------------------
#
# TITLE - fig1.py
# AUTHOR - James Lane
# PROJECT - AST1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Make Figure 1 for Lane+Bovy paper. Plot 2D map of Gaia DR2 + AstroNN 
stars. Make two versions: one which is only a 2D map of stellar 
number density, and one which has three panels, including panels showing 
radial and tangential velocity.
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb

## Plotting
from matplotlib import pyplot as plt
from matplotlib import colors, cm, colorbar

## Astropy
from astropy.io import fits
from astropy import table
from astropy import units as apu
from astropy.coordinates import CartesianDifferential

## Scipy
from scipy.stats import binned_statistic_2d

# ----------------------------------------------------------------------------

## matplotlibrc

plt.rc('font', family='serif', size=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('text', usetex=True)

# ----------------------------------------------------------------------------

## Load data

gaiadr2_apogee_catalog = '../../../data/generated/gaiadr2-apogee_dataset.FIT'
f = fits.open(gaiadr2_apogee_catalog)
data = f[1].data

where_low_z = np.where( np.abs(data['Z']) < 0.3 )[0]
data_low_z = data[where_low_z] 
z_select_text = r'$|$Z$_{GC}| < 0.3$ kpc'

apid = data_low_z['APOGEE_ID']
locid = data_low_z['LOCATION_ID']
vhelio = data_low_z['VHELIO']
pmll = data_low_z['PM_LL']
pmbb = data_low_z['PM_BB']
gc_x = data_low_z['X']
gc_y = data_low_z['Y']
gc_z = data_low_z['Z']
gc_vx = data_low_z['VX']
gc_vy = data_low_z['VY']
gc_vz = data_low_z['VZ']

gc_R = np.sqrt(np.square(gc_x)+np.square(gc_y))
gc_phi = np.arctan2(gc_y,gc_x)
gc_vR = np.cos(gc_phi)*gc_vx + np.sin(gc_phi)*gc_vy
gc_vT = -np.sin(gc_phi)*gc_vx + np.cos(gc_phi)*gc_vy

# ----------------------------------------------------------------------------

## Functions

# ----------------------------------------------------------------------------

## Prepare for plotting

# Bins
n_bins = 25

# Sun
x_sun = -8.125

# Ranges
x_hi = 0
x_lo = -15
y_hi = 10
y_lo = -5
hist_range = [ [x_lo,x_hi], [y_lo,y_hi] ]

# ----------------------------------------------------------------------------

## Make the first version of the figure

fig = plt.figure( figsize=(6,5) )
ax = fig.add_subplot(111)

hist, xedges, yedges = np.histogram2d(gc_x, gc_y, bins=n_bins, range=hist_range)

# Rotate to account for histogram -> plotting grid
hist = np.rot90(hist)

# Find low-N bins. As long as histogram geometry remains the same this will be 
# used for greying out values.
where_low_N = np.where( (hist < 10) & (hist > 0) )
low_N_mask = np.zeros((n_bins,n_bins,4))
low_N_mask[:,:,3] = 0.0
low_N_mask[where_low_N[0],where_low_N[1],:3] = 0.75
low_N_mask[where_low_N[0],where_low_N[1],3] = 1.0

# Find 0-N bins. This will differentiate between no data and too little data
where_no_N = np.where( hist == 0 )
no_N_mask = np.zeros((n_bins,n_bins,4))
no_N_mask[:,:,3] = 0.0
no_N_mask[where_no_N[0],where_no_N[1],:3] = 1
no_N_mask[where_no_N[0],where_no_N[1],3] = 1.0

# pdb.set_trace()

img = ax.imshow(np.log10(hist,where=hist>0), interpolation='nearest',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
               cmap='viridis', vmin=0.5, vmax=3.5)
low_N_img = ax.imshow(low_N_mask, interpolation='nearest',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
no_N_img = ax.imshow(no_N_mask, interpolation='nearest',
                     extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

cbar = plt.colorbar(img)
ax.scatter(x_sun, 0, marker=r'$\odot$', color='Black', s=256)
orbit_circle = plt.Circle((0, 0), x_sun, edgecolor='Black', facecolor='None')
ax.add_artist(orbit_circle)

ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
cbar.set_label(r'Log$_{10}$[N]')
ax.annotate(z_select_text, xy=(0.55,0.1), xycoords='axes fraction')

fig.savefig('fig1_v1.pdf')
plt.close(fig)

# ----------------------------------------------------------------------------

## Make the second version of the figure

fig = plt.figure(figsize=(6,12))
axs = fig.subplots( nrows=3, ncols=1 )

# Make the number histogram
hist, xedges, yedges = np.histogram2d(gc_x, gc_y, bins=n_bins, range=hist_range)
hist = np.rot90(hist)
img = axs[0].imshow(np.log10(hist,where=hist>0), interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', 
    vmin=0.5, vmax=3.5)
    
# Make the velocity histograms
hist_vR, _, _, _ = binned_statistic_2d(gc_x, gc_y, values=gc_vR, 
    statistic='median', bins=n_bins, range=hist_range)
hist_vT, _, _, _ = binned_statistic_2d(gc_x, gc_y, values=-gc_vT, 
    statistic='median', bins=n_bins, range=hist_range)

img_vR = axs[1].imshow(np.rot90(hist_vR), interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap='RdYlBu', vmin=-15, vmax=15)
img_vT = axs[2].imshow(np.rot90(hist_vT), interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap='RdYlBu', vmin=190, vmax=250)

# Find low-N bins. As long as histogram geometry remains the same this will be 
# used for greying out values.
where_low_N = np.where( (hist < 10) & (hist > 0) )
low_N_mask = np.zeros((n_bins,n_bins,4))
low_N_mask[:,:,3] = 0.0
low_N_mask[where_low_N[0],where_low_N[1],:3] = 0.75
low_N_mask[where_low_N[0],where_low_N[1],3] = 1.0

axs[0].imshow(low_N_mask, interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
axs[1].imshow(low_N_mask, interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
axs[2].imshow(low_N_mask, interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

# Find 0-N bins. This will differentiate between no data and too little data
where_no_N = np.where( hist == 0 )
no_N_mask = np.zeros((n_bins,n_bins,4))
no_N_mask[:,:,3] = 0.0
no_N_mask[where_no_N[0],where_no_N[1],:3] = 1
no_N_mask[where_no_N[0],where_no_N[1],3] = 1.0

axs[0].imshow(no_N_mask, interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
axs[1].imshow(no_N_mask, interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
axs[2].imshow(no_N_mask, interpolation='nearest',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

# Decorate
axs[0].set_ylabel(r'Y$_{GC}$ [kpc]')
axs[1].set_ylabel(r'Y$_{GC}$ [kpc]')
axs[2].set_xlabel(r'X$_{GC}$ [kpc]')
axs[2].set_ylabel(r'Y$_{GC}$ [kpc]')
axs[0].set_xlim(x_lo-0.5, x_hi+0.5)
axs[0].set_ylim(y_lo-0.5, y_hi+0.5)
axs[1].set_xlim(x_lo-0.5, x_hi+0.5)
axs[1].set_ylim(y_lo-0.5, y_hi+0.5)
axs[2].set_xlim(x_lo-0.5, x_hi+0.5)
axs[2].set_ylim(y_lo-0.5, y_hi+0.5)
axs[0].tick_params(direction='in', top='on', right='on', labelbottom='off')
axs[1].tick_params(direction='in', top='on', right='on', labelbottom='off')
axs[2].tick_params(direction='in', top='on', right='on')

# # Label the colorbar
cbar = plt.colorbar(img, ax=axs[0], shrink=0.9)
cbar_vR = plt.colorbar(img_vR, ax=axs[1], shrink=0.9)
cbar_vT = plt.colorbar(img_vT, ax=axs[2], shrink=0.9)
cbar.set_label(r'Log$_{10}$(N)')
cbar_vR.set_label(r'Median $V_{R}$ [km/s]')
cbar_vT.set_label(r'Median $V_{T}$ [km/s]')

# Decorate solar orbit
axs[0].scatter(x_sun, 0, marker=r'$\odot$', color='Black', s=256)
axs[1].scatter(x_sun, 0, marker=r'$\odot$', color='Black', s=256)
axs[2].scatter(x_sun, 0, marker=r'$\odot$', color='Black', s=256)
orbit_circle = plt.Circle((0, 0), x_sun, edgecolor='Black', facecolor='None')
vR_orbit_circle = plt.Circle((0, 0), x_sun, edgecolor='Black', facecolor='None')
vT_orbit_circle = plt.Circle((0, 0), x_sun, edgecolor='Black', facecolor='None')
axs[0].add_artist(orbit_circle)
axs[1].add_artist(vR_orbit_circle)
axs[2].add_artist(vT_orbit_circle)

# Annotate
z_select_text = r'$|$Z$_{GC}| < 0.3$ kpc'
axs[0].annotate(z_select_text, xy=(0.55,0.1), xycoords='axes fraction', 
    fontsize=12)

fig.subplots_adjust(hspace=0)
plt.tight_layout()
fig.savefig('fig1_v2.pdf')
plt.close(fig)

# ----------------------------------------------------------------------------
