# ----------------------------------------------------------------------------
#
# TITLE -
# AUTHOR - James Lane
# PROJECT -
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Make Figure 6, shows the M and B values for the outter disk
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

plt.rc('font', family='serif', size=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('text', usetex=True)

# ----------------------------------------------------------------------------

## Make the linear models

### Load catalogs
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

plot_kws={'ecolor':'Black','marker':'o','markeredgecolor':'Black','color':'Black','markersize':10,
          'markerfacecolor':'None','capsize':5,'fmt':'o'}

plot_type='errorbar'

# Format for this figure is 2x2
fig = plt.figure( figsize=(12,6) ) 
axs = fig.subplots( nrows=2, ncols=2 )

# Plot
if plot_type == 'errorbar':
    axs[0,0].errorbar( lm.R_bin_cents, lm.b_vR, yerr=lm.b_err_vR, 
        **plot_kws)
    axs[0,1].errorbar( lm.R_bin_cents, lm.m_vR, yerr=lm.m_err_vR, 
        **plot_kws)
    axs[1,0].errorbar( lm.R_bin_cents, lm.b_vT, yerr=lm.b_err_vT, 
        **plot_kws)
    axs[1,1].errorbar( lm.R_bin_cents, lm.m_vT, yerr=lm.m_err_vT, 
        **plot_kws)
elif plot_type == 'plot':
    axs[0,0].scatter( lm.R_bin_cents, lm.b_vR, **plot_kws)
    axs[0,1].scatter( lm.R_bin_cents, lm.m_vR, **plot_kws)
    axs[1,0].scatter( lm.R_bin_cents, lm.b_vT, **plot_kws)
    axs[1,1].scatter( lm.R_bin_cents, lm.m_vT, **plot_kws)
elif plot_type == 'scatter':
    axs[0,0].scatter( lm.R_bin_cents, lm.b_vR, **plot_kws)
    axs[0,1].scatter( lm.R_bin_cents, lm.m_vR, **plot_kws)
    axs[1,0].scatter( lm.R_bin_cents, lm.b_vT, **plot_kws)
    axs[1,1].scatter( lm.R_bin_cents, lm.m_vT, **plot_kws)

# Labels and limits
axs[0,0].set_ylabel(r'v$_{0,R}$ [km/s]')
axs[0,1].set_ylabel(r'A$_{R}$ [km/s]')
axs[1,0].set_ylabel(r'v$_{0,T}$ [km/s]')
axs[1,1].set_ylabel(r'A$_{T}$ [km/s]')
axs[0,0].set_xlabel(r'R [kpc]')
axs[0,1].set_xlabel(r'R [kpc]')
axs[1,0].set_xlabel(r'R [kpc]')
axs[1,1].set_xlabel(r'R [kpc]')
axs[0,0].set_xlim( np.min(lm.R_bin_cents)-0.5, np.max(lm.R_bin_cents)+0.5  )
axs[0,1].set_xlim( np.min(lm.R_bin_cents)-0.5, np.max(lm.R_bin_cents)+0.5  )
axs[1,0].set_xlim( np.min(lm.R_bin_cents)-0.5, np.max(lm.R_bin_cents)+0.5  )
axs[1,1].set_xlim( np.min(lm.R_bin_cents)-0.5, np.max(lm.R_bin_cents)+0.5  )
axs[1,0].set_ylim(190,210)
axs[0,0].set_ylim(-10,10)

# Add fiducials
axs[0,0].axhline(0, linestyle='dashed', color='Black')
axs[0,1].axhline(0, linestyle='dashed', color='Black')
axs[1,1].axhline(0, linestyle='dashed', color='Black')

# Prior
# if lm.vT_prior_type=='df':    
#     axs[1,0].plot( lm.df_prior_R, lm.df_prior_vT,
#         linestyle='dashed', color='Black' )
# if lm.vT_prior_type=='rotcurve':
#     axs[1,0].plot( lm.rotcurve_prior_R, lm.rotcurve_prior_vT,
#         linestyle='dashed', color='Black')
# ##fi

prior_rs, prior_vts = np.load('../../../data/generated/MWPotential2014_DF_vT_data.npy')
where_prior_in_rlims = np.where( (prior_rs < 15) & (prior_rs > 12) )[0]
axs[0,0].plot([], [], linestyle='dashed', color='Black', label='Prior')
axs[1,0].plot(prior_rs[where_prior_in_rlims], prior_vts[where_prior_in_rlims],
              linestyle='dashed', linewidth=1.0, color='Black', label='Prior', 
              zorder=10)
axs[1,0].fill_between( prior_rs[where_prior_in_rlims], 
    prior_vts[where_prior_in_rlims]-5, prior_vts[where_prior_in_rlims]+5, 
    color='Black', alpha=0.2)
axs[0,0].fill_between( np.array([12,15]), np.array([5,5]), np.array([-5,-5]), 
    color='Black', alpha=0.2)

fig.subplots_adjust(hspace=0.3)
fig.savefig('fig6.pdf')