# ----------------------------------------------------------------------------
#
# TITLE - fig2.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Make Figure 2 for Lane + Bovy. Will show bootstrap samples + errors in
radial bins.
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb

## Plotting
from matplotlib import pyplot as plt

## Astropy
from astropy.io import fits
from astropy import table
from astropy import units as apu

## Project specific
sys.path.append('../../../src/')
import ast1501.linear_model

# ----------------------------------------------------------------------------

## matplotlibrc

plt.rc('font', family='serif', size=17)
plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=13)
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
gc_vT = np.sin(gc_phi)*gc_vx - np.cos(gc_phi)*gc_vy
gc_phi = np.arctan2(gc_y,-gc_x)

# ----------------------------------------------------------------------------

## Use the linear model class to make bootstrap samples

# Radial bin range and size
R_lims = [5,15]
R_bin_size = 1.0
R_bin_cents = np.arange( R_lims[0], R_lims[1], R_bin_size ) + R_bin_size/2

# Phi bin range and size
phi_lims = [-np.pi/2, np.pi/2]
phi_bin_size = np.pi/30
phi_bin_cents = np.arange( phi_lims[0], phi_lims[1], phi_bin_size ) + phi_bin_size/2

# Phib bin range and size
phib_lims = [0, np.pi/2]
phib_bin_size = np.pi/30
phib_bin_cents = np.arange( phib_lims[0], phib_lims[1], phib_bin_size ) + phi_bin_size/2

lm = ast1501.linear_model.LinearModel(instantiate_method=1, gc_R=gc_R, 
    gc_phi=gc_phi, gc_vR=gc_vR, gc_vT=gc_vT, R_lims=R_lims, 
    R_bin_size=R_bin_size, phi_lims=phi_lims, 
    phi_bin_size=phi_bin_size, phib_lims=phib_lims, n_bs=1000,
    phib_bin_size=phib_bin_size)

bs_vR, bs_vT = lm.get_bs_samples()
n_R_bins = len(bs_vR)

# ----------------------------------------------------------------------------

## Make the figure

fig = plt.figure( figsize=(11,n_R_bins*1.5) )
axs = fig.subplots( nrows=n_R_bins, ncols=2 )

# Loop over all radii
for i in range( n_R_bins ):
    
    # Unpack the velocity sample for this radius
    bin_R_cent = bs_vR[i][0]
    bin_vR = bs_vR[i][1]
    bin_vR_err = bs_vR[i][2]
    bin_phi = bs_vR[i][3]
    bin_vT = bs_vT[i][1]
    bin_vT_err = bs_vT[i][2]
    
    # Plot
    axs[i,0].errorbar( bin_phi*180/np.pi, bin_vR, yerr=bin_vR_err, fmt='o', 
        ecolor='Black', marker='o', markerfacecolor='None', capsize=1.5,
        markeredgecolor='Black', markersize=5, elinewidth=0.75, capthick=0.75)
    axs[i,1].errorbar( bin_phi*180/np.pi, bin_vT, yerr=bin_vT_err, fmt='o', 
        ecolor='Black', marker='o', markerfacecolor='None', capsize=1.5,
        markeredgecolor='Black', markersize=5, elinewidth=0.75, capthick=0.75)

    # Add fiducials: bar, 0 line or tangential velocity curve
    axs[i,0].axvline( 25, linestyle='dotted', linewidth=0.75, 
        color='Red')
    axs[i,1].axvline( 25, linestyle='dotted', linewidth=0.75, 
        color='Red')
    axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=0.75)
    X0, _ = lm._generate_gaussian_prior_m_b('vT', bin_R_cent)
    b0 = X0[0,0]
    axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=0.75)
    axs[i,1].axhline( b0, linestyle='dashed', color='Black', linewidth=0.75)
        
    # Annotate
    axs[i,0].annotate( r'$R_{cen}=$'+str(bin_R_cent)+' kpc', 
        xy=(0.05,0.1), xycoords='axes fraction', fontsize=12 )
        
    # Set limits
    axs[i,0].set_xlim( -30, 70 )
    axs[i,1].set_xlim( -30, 70 )
    axs[i,0].set_ylim( -45, 55 )
    axs[i,1].set_ylim( 175, 245 )
    
    # Set the labels
    axs[i,0].set_ylabel(r'$v_{R}$ [km/s]')
    axs[i,1].set_ylabel(r'$v_{T}$ [km/s]')
    if i == n_R_bins-1:
        axs[i,0].set_xlabel(r'$\phi$ [deg]')
        axs[i,1].set_xlabel(r'$\phi$ [deg]')
    else:
        axs[i,0].tick_params(labelbottom='off')
        axs[i,1].tick_params(labelbottom='off')

fig.subplots_adjust(hspace=0.05)
plt.savefig('fig2.pdf')

# ----------------------------------------------------------------------------