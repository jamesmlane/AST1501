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
from matplotlib import colors, cm

## Astropy
from astropy import units as apu
from astropy.io import fits
from astropy.table import Table

## Project specific
sys.path.append('../../../src/')
import ast1501.linear_model
import ast1501.potential

# ----------------------------------------------------------------------------

## matplotlibrc

plt.rc('font', family='serif', size=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

# ----------------------------------------------------------------------------

### Load catalogs
gaiadr2_apogee_catalog = '../../data/generated/gaiadr2-apogee_dataset.FIT'
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

## Keywords

# Radial bin range and size
R_lim = [12,15]
R_bin_size = 0.75
R_bin_cents = np.arange( R_lim[0], R_lim[1], R_bin_size ) + R_bin_size/2

# Phi bin range and size
phi_lim = [-np.pi/2, np.pi/2]
phi_bin_size = np.pi/30
phi_bin_cents = np.arange( phi_lim[0], phi_lim[1], phi_bin_size ) + phi_bin_size/2

# Phib bin range and size
phib_lim = [0, np.pi/2]
phib_bin_size = np.pi/60
phib_bin_cents = np.arange( phib_lim[0], phib_lim[1], phib_bin_size ) + phi_bin_size/2

# Make the slave linear model
lm_sl = ast1501.linear_model.LinearModel(instantiate_method=1, gc_R=gc_R, 
    gc_phi=gc_phi, gc_vR=gc_vR, gc_vT=gc_vT, R_lims=R_lim, 
    R_bin_size=R_bin_size, phi_lims=phi_lim, 
    phi_bin_size=phi_bin_size, phib_lims=phib_lim,
    phib_bin_size=phib_bin_size, force_yint_zero_vR=False)

# ----------------------------------------------------------------------------

n_mc_samp = 1000

# Generate the b/a sample
b_a_mc_sample = np.random.uniform(low=0.9, high=1.1, size=n_mc_samp)

# Generate the phiB sample
phiB_mc_sample = np.random.uniform(low=0, high=np.pi, size=n_mc_samp)

mc_sample_arr = []

# Loop over the MC samples
for i in tqdm_notebook(range(n_mc_samp)):
    
    # Generate the MC samples of the Gaia data and parameters
    bs_mc_samp_vR, bs_mc_samp_vT = lm_sl.sample_bootstrap()
    b_a = b_a_mc_sample[i]
    phiB = phiB_mc_sample[i]
    
    # Make the Kuijken model for these parameters
    kt = ast1501.potential.kuijken_potential(b_a=b_a, phib=phiB)
    
    # Make a LinearModel for the Gaia data
    lm_gd = ast1501.linear_model.LinearModel(instantiate_method=3, bs_sample_vR=bs_mc_samp_vR,
                                             bs_sample_vT=bs_mc_samp_vT, phib_lims=phib_lim, 
                                             phib_bin_size=phib_bin_size, use_velocities=['vR'])
    
    bs_kt_samp_vR = []
    bs_kt_samp_vT = []
    
    # Make a LinearModel for the Kuijken data. First make Kuijken look like a bootstrap sample
    for i in range( len(bs_mc_samp_vR) ):
            
        # Basic positions
        R_bin_cent = bs_mc_samp_vR[i][0]
        phi_bin_cents = bs_mc_samp_vR[i][3]

        # Make the KT velocities
        kt_vR = kt.kuijken_vr(R=R_bin_cent, phi=phi_bin_cents, )
        kt_vT = kt.kuijken_vt(R=R_bin_cent, phi=phi_bin_cents, )

        # Mock some errors. Half a km/s is reasonable for velocities
        kt_vR_err = np.ones_like(kt_vR)*0.5
        kt_vT_err = np.ones_like(kt_vT)*0.5

        # Knit this into a bootstrap-like array
        bs_kt_samp_vR.append( [R_bin_cent,kt_vR,kt_vR_err,bs_mc_samp_vR[i][3],bs_mc_samp_vR[i][4]] )
        bs_kt_samp_vT.append( [R_bin_cent,kt_vT,kt_vT_err,bs_mc_samp_vT[i][3],bs_mc_samp_vT[i][4]] )
    
    ###i
            
    lm_kt = ast1501.linear_model.LinearModel(instantiate_method=3, bs_sample_vR=bs_kt_samp_vR, 
                                             bs_sample_vT=bs_kt_samp_vT, phib_lims=phib_lim, 
                                             phib_bin_size=phib_bin_size, use_velocities=['vR'], )
    
    mc_sample_arr.append([b_a,phiB,lm_gd,lm_kt])
    
###i

# ----------------------------------------------------------------------------
