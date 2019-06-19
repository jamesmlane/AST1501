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

phib_lims = [-np.pi/4, np.pi/4]
phib_bin_size = np.pi/45

lm_th_phiB = 0
lm_th = ast1501.linear_model.LinearModel( instantiate_method=2, 
    df_filename='../../../notebooks/8-radial_DF_generation/data/results_b90.npy',
    phib_bin_size=phib_bin_size, phib_lims=phib_lims, vT_prior_type='df', 
    phiB=lm_th_phiB)
    
lm_sa = ast1501.linear_model.LinearModel( instantiate_method=2, 
    df_filename='../../../scripts/generate_spiral_df/base/data.npy',
    phib_bin_size=phib_bin_size, phib_lims=phib_lims, vT_prior_type='df')
    
lm_bar_phiB = 25*np.pi/180
lm_bar = ast1501.linear_model.LinearModel( instantiate_method=2,
    df_filename='../../../scripts/generate_bar_df/2019-04-24/data.npy',
    phib_bin_size=phib_bin_size, phib_lims=phib_lims, vT_prior_type='df',
    phiB=lm_bar_phiB)
    
lm_all_phiB = 0
lm_all = ast1501.linear_model.LinearModel( instantiate_method=2, 
    df_filename='../../../scripts/generate_bar_spiral_triaxial_df/2019-04-21/data_b090.npy',
    phib_bin_size=phib_bin_size, phib_lims=phib_lims, vT_prior_type='df', 
    phiB=lm_all_phiB)

# ----------------------------------------------------------------------------

## Make the figure

fig = plt.figure( figsize=(12,6) ) 
axs = fig.subplots( nrows=2, ncols=2 )

plot_colors = ['Red','DarkOrange','Purple']
plot_labels = ['SA','Bar','TH+BAR+SA']
plot_lms = [lm_sa, lm_bar, lm_all]

for i in range(3):
    
    this_lm = plot_lms[i]
    # pdb.set_trace()
    axs[0,0].plot(this_lm.R_bin_cents, this_lm.b_vR, markeredgecolor='Black',
        markerfacecolor=plot_colors[i], marker='o', color=plot_colors[i], 
         label=plot_labels[i])
    axs[0,1].plot(this_lm.R_bin_cents, this_lm.m_vR, markeredgecolor='Black',
        markerfacecolor=plot_colors[i], marker='o', color=plot_colors[i] )
    axs[1,0].plot(this_lm.R_bin_cents, this_lm.b_vT, markeredgecolor='Black',
        markerfacecolor=plot_colors[i], marker='o', color=plot_colors[i] )
    axs[1,1].plot(this_lm.R_bin_cents, this_lm.m_vT, markeredgecolor='Black',
        markerfacecolor=plot_colors[i], marker='o', color=plot_colors[i] )
    
###i

# Do the triaxial halo
axs[0,0].scatter(lm_th.R_bin_cents, lm_th.b_vR, edgecolor='Black',
    facecolor='DodgerBlue', marker='o', label='TH')
axs[0,1].scatter(lm_th.R_bin_cents, lm_th.m_vR, edgecolor='Black',
    facecolor='DodgerBlue', marker='o' )
axs[1,0].scatter(lm_th.R_bin_cents, lm_th.b_vT, edgecolor='Black',
    facecolor='DodgerBlue', marker='o' )
axs[1,1].scatter(lm_th.R_bin_cents, lm_th.m_vT, edgecolor='Black',
    facecolor='DodgerBlue', marker='o' )

# Do the subtraction
axs[0,0].plot([], [], color='DodgerBlue', linestyle='dashed', label='Subtracted')
axs[0,1].plot(this_lm.R_bin_cents, lm_all.m_vR-lm_bar.m_vR, color='DodgerBlue', 
    linestyle='dashed' )
axs[1,1].plot(this_lm.R_bin_cents, lm_all.m_vT-lm_bar.m_vT, color='DodgerBlue', 
    linestyle='dashed' )

prior_rs, prior_vts = np.load('../../../data/generated/MWPotential2014_DF_vT_data.npy')
where_prior_in_rlims = np.where( (prior_rs < 15) & (prior_rs > 5) )[0]
axs[0,0].plot([], [], linestyle='dashed', color='Black', label='Prior')
axs[1,0].plot(prior_rs[where_prior_in_rlims], prior_vts[where_prior_in_rlims],
              linestyle='dashed', linewidth=1.0, color='Black', label='Prior', 
              zorder=10)
axs[1,0].fill_between( prior_rs[where_prior_in_rlims], 
    prior_vts[where_prior_in_rlims]-5, prior_vts[where_prior_in_rlims]+5, 
    color='Black', alpha=0.2)
axs[0,0].plot([], [], linestyle='dotted', linewidth=1.0, color='Black', 
              label='Bar 2:1 OLR')
axs[0,1].axvline(7.15, linestyle='dotted', linewidth=1.0, color='Black')
axs[1,1].axvline(7.15, linestyle='dotted', linewidth=1.0, color='Black')

axs[0,0].set_ylim(-10,10)
axs[1,0].set_ylim(190,220)
axs[0,0].set_xlabel('R [kpc]')
axs[0,0].set_ylabel(r'v$_{0,R}$ [km/s]')
axs[0,1].set_xlabel('R [kpc]')
axs[0,1].set_ylabel(r'A$_{R}$ [km/s]')
axs[1,0].set_xlabel('R [kpc]')
axs[1,0].set_ylabel(r'v$_{0,T}$ [km/s]')
axs[1,1].set_xlabel('R [kpc]')
axs[1,1].set_ylabel(r'A$_{T}$ [km/s]')

axs[0,0].legend(ncol=2, fontsize=8, handlelength=2.5)

fig.subplots_adjust(hspace=0.3)
fig.savefig('fig3.pdf')

# ----------------------------------------------------------------------------