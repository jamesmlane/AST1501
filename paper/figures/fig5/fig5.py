# ----------------------------------------------------------------------------
#
# TITLE - fig2.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Make Figure52 for Lane + Bovy. Will show bootstrap samples + errors in
radial bins for the outter disk + the best-fitting radial profiles.
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

### Cut on galactocentric absolute Z < 0.3 kpc
where_low_z = np.where( np.abs(data['Z']) < 0.3 )[0]
data_low_z = data[where_low_z] 
z_select_text = r'$|$Z$_{GC}| < 0.3$ kpc'

### Read catalog values

# ID, RA, Dec, logg, abundances, errors
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

### Convert to galactocentric radius and radial velocity
gc_R = np.sqrt(np.square(gc_x)+np.square(gc_y))
gc_phi = np.arctan2(gc_y,gc_x)
gc_vR = np.cos(gc_phi)*gc_vx + np.sin(gc_phi)*gc_vy
gc_vT = np.sin(gc_phi)*gc_vx - np.cos(gc_phi)*gc_vy
gc_phi = np.arctan2(gc_y,-gc_x)

# ----------------------------------------------------------------------------

## Use the linear model class to make bootstrap samples

# Radial bin range and size
R_lim = [12,15]
R_bin_size = 0.75
R_bin_cents = np.arange( R_lim[0], R_lim[1], R_bin_size ) + R_bin_size/2

# Phi bin range and size
phi_lim = [-np.pi/2, np.pi/2]
phi_bin_size = np.pi/30
phi_bin_cents = np.arange( phi_lim[0], phi_lim[1], phi_bin_size ) + phi_bin_size/2

# Phib bin range and size
phib_lim = [-np.pi/4, np.pi/4]
phib_bin_size = np.pi/60
phib_bin_cents = np.arange( phib_lim[0], phib_lim[1], phib_bin_size ) + phib_bin_size/2

# ----------------------------------------------------------------------------

lm = ast1501.linear_model.LinearModel(instantiate_method=1, gc_R=gc_R, 
    gc_phi=gc_phi, gc_vR=gc_vR, gc_vT=gc_vT, R_lims=R_lim, 
    R_bin_size=R_bin_size, phi_lims=phi_lim, 
    phi_bin_size=phi_bin_size, phib_lims=phib_lim, n_bs=100,
    phib_bin_size=phib_bin_size, force_yint_zero_vR=False, use_velocities=['vR','vT'])

# ----------------------------------------------------------------------------


fig = plt.figure( figsize=(10,lm.n_R_bins*2) )
axs = fig.subplots( nrows=lm.n_R_bins, ncols=2 )

# Loop over all radii
for i in range( lm.n_R_bins ):
    
    # Unpack the velocity sample for this radius
    bin_R_cent = lm.bs_sample_vR[i][0]
    bin_vR = lm.bs_sample_vR[i][1]
    bin_vR_err = lm.bs_sample_vR[i][2]
    bin_phi = lm.bs_sample_vR[i][3]
    bin_vT = lm.bs_sample_vT[i][1]
    bin_vT_err = lm.bs_sample_vT[i][2]
    
    # Plot
    axs[i,0].errorbar( bin_phi*180/np.pi, bin_vR, yerr=bin_vR_err, fmt='o', 
        ecolor='Black', marker='o', markerfacecolor='None', 
        markeredgecolor='Black', markersize=5)
    axs[i,1].errorbar( bin_phi*180/np.pi, bin_vT, yerr=bin_vT_err, fmt='o', 
        ecolor='Black', marker='o', markerfacecolor='None', 
        markeredgecolor='Black', markersize=5)

    # Plot the best-fitting amplitude
    trig_phis = np.linspace(phi_lim[0], phi_lim[1], num=100)
    axs[i,0].plot( trig_phis*180/np.pi, 
        lm.b_vR[i]+lm.m_vR[i]*np.sin(2*(trig_phis-lm.phiB)))
    axs[i,1].plot( trig_phis*180/np.pi, 
        lm.b_vT[i]+lm.m_vT[i]*np.cos(2*(trig_phis-lm.phiB)))

    # Add fiducials: bar, 0 line or tangential velocity curve
    axs[i,0].axvline( 25, linestyle='dotted', linewidth=1.0, 
        color='Red')
    axs[i,1].axvline( 25, linestyle='dotted', linewidth=1.0, 
        color='Red')
    axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=1.0 )
    X0, _ = lm._generate_gaussian_prior_m_b('vT', bin_R_cent)
    b0 = X0[0,0]
    axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=1.0 )
    axs[i,1].axhline( b0, linestyle='dashed', color='Black', linewidth=1.0 )
        
    # Annotate
    axs[i,0].annotate( r'$R_{cen}=$'+str(bin_R_cent)+' kpc', 
        xy=(0.05,0.1), xycoords='axes fraction', fontsize=12 )
        
    # Set limits
    # axs[i,0].set_xlim( phi_lim[0], phi_lim[1] )
    # axs[i,1].set_xlim( phi_lim[0], phi_lim[1] )
    
    # Set limits
    axs[i,0].set_xlim( -45, 90 )
    axs[i,1].set_xlim( -45, 90 )
    axs[i,0].set_ylim( -30, 30 )
    axs[i,1].set_ylim( 175, 245 )
    
    # Set the labels
    axs[i,0].set_ylabel(r'$v_{R}$ [km/s]')
    axs[i,1].set_ylabel(r'$v_{T}$ [km/s]')
    ##fi
    if i == lm.n_R_bins-1:
        axs[i,0].set_xlabel(r'$\phi$ [deg]')
        axs[i,1].set_xlabel(r'$\phi$ [deg]')
    ##fi
    else:
        axs[i,0].tick_params(labelbottom='off')
        axs[i,1].tick_params(labelbottom='off')
    ##ie
    

fig.subplots_adjust(hspace=0.05)

fig.savefig('fig5.pdf')