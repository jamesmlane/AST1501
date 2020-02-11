# ----------------------------------------------------------------------------
#
# TITLE -
# AUTHOR - James Lane
# PROJECT -
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Script to bin simulations from Hunt+ 2019 for analysis
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb

from astropy import table

from linear_model import LinearModel2

# ----------------------------------------------------------------------------

filename_lsbar = '/gpfs/fs0/scratch/b/bovy/jash2/mwbubs-models/data/transient_spirals/11-lsb-corrected-angle/lsb-corrected-angle_99.fits'
filename_sfbar = '/gpfs/fs0/scratch/b/bovy/jash2/mwbubs-models/data/transient_spirals/08-three-spirals-sfb/three-spirals-sfb_99.fits'

# ----------------------------------------------------------------------------

# Set limits for analysis
phi_lims = [-np.pi/2,np.pi/2] # In radians
R_lims = [5,15] # In kpc
z_lim = 0.3

# Load the data
data = table.Table.read(filename_lsbar, format="fits")
R,phi,vR,vT,z = data["R", "phi", "vR", "vT", "z"].as_array().view((np.float64, 5)).T
time = data["t"][0]

# Convert to kpc, km/s, and make phi range -np.pi -> np.pi
R *= 8
z *= 8
vR *= 220
vT *= 220
phi[phi > np.pi] -= 2*np.pi

# Mask out the good particles
valid_particle_mask = np.isfinite(R) & np.isfinite(phi) & np.isfinite(vR)\
 & np.isfinite(vT) & (phi > phi_lims[0]) & (phi < phi_lims[1])\
 & (R > R_lims[0]) & (R < R_lims[1]) & (np.abs(z) < z_lim)

R = R[valid_particle_mask]
phi = phi[valid_particle_mask]
vR = vR[valid_particle_mask]
vT = vT[valid_particle_mask]

# Create the bootstrap samples
R_bin_size=1.0
phi_bin_size=np.pi/30
phib_lims = [0,np.pi/2]
phib_bin_size = np.pi/60
force_yint_vR=True
use_velocities=['vR','vT']

lm = LinearModel2(instantiate_method=1, gc_R=R, gc_phi=phi, gc_vT=vT, 
    gc_vR=vR, phi_lims=phi_lims, R_lims=R_lims, phi_bin_size=phi_bin_size, 
    R_bin_size=R_bin_size, phib_lims=phib_lims, phib_bin_size=phib_bin_size, 
    use_velocities=use_velocities, vT_prior_path='./MWPotential2014_df_vT_data.npy')
    

