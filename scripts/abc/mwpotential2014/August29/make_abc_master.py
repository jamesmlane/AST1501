# ----------------------------------------------------------------------------
#
# TITLE - make_abc_samples.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Script to make ABC master LinearModel for the triaxial halo project.

Run on August 29

6-12 kpc, 1kpc bins, vR only, LinearModel2, 
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, importlib, glob, pickle, tqdm

## Plotting
# from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib import colors
# from matplotlib import cm

## Astropy
from astropy.io import fits
from astropy import table
from astropy import units as apu

## galpy
from galpy import potential

## Scipy
from scipy.stats import binned_statistic_2d, binned_statistic
from scipy import stats
from scipy import interpolate

## Add project-specific package. Assume relative location
sys.path.append('../../../../src/')
from ast1501.linear_model import LinearModel
from ast1501.linear_model import LinearModel2
from ast1501.linear_model import LinearModelSolution
import ast1501.abc
import ast1501.potential

# ----------------------------------------------------------------------------

# Load parameters from the YAML file
PARAMETER_FILE = './abc_parameters.yaml'
parameter_dict = ast1501.abc.load_abc_params(PARAMETER_FILE)
locals().update(parameter_dict)

# ----------------------------------------------------------------------------

### Read DR16 data
print('Reading data...')

## Load catalogs
gaiadr2_apogee_catalog = '../../../../data/generated/gaiadr2-apogee_dr16_dataset.FIT'
f = fits.open(gaiadr2_apogee_catalog)
data = f[1].data

## Cut on galactocentric absolute Z < 0.3 kpc
where_low_z = np.where( np.abs(data['Z']) < 0.3 )[0]
data_low_z = data[where_low_z] 
z_select_text = r'$|$Z$_{GC}| < 0.3$ kpc'

## Read catalog values
# ID, RA, Dec, logg, abundances, errors
apid = data_low_z['APOGEE_ID']
locid = data_low_z['LOCATION_ID']
vhelio = data_low_z['VHELIO']
gc_R = data_low_z['R']
gc_phi = data_low_z['PHI']
gc_phi[ np.where(gc_phi > np.pi) ] -= 2*np.pi
gc_z = data_low_z['Z']
gc_vR = data_low_z['VR']
gc_vT = data_low_z['VT']
gc_vz = data_low_z['VZ']

# ----------------------------------------------------------------------------

### Make the master
print('Making master linear model...')
lm_mas = LinearModel2(instantiate_method=1, 
                      gc_R=gc_R, 
                      gc_phi=gc_phi, 
                      gc_vT=gc_vT, 
                      gc_vR=gc_vR, 
                      R_lims=R_LIMS, 
                      R_bin_size=R_BIN_SIZE, 
                      phi_lims=PHI_LIMS, 
                      phi_bin_size=PHI_BIN_SIZE, 
                      phib_lims=PHIB_LIMS,
                      phib_bin_size=PHIB_BIN_SIZE,
                      use_velocities=USE_VELOCITIES,
                      prior_var_arr=PRIOR_VAR_ARR, 
                      vT_prior_type=VT_PRIOR_TYPE,
                      vT_prior_path=VT_PRIOR_PATH,
                      vT_prior_offset=VT_PRIOR_OFFSET,
                      phiB=PHIB,
                      n_iterate=N_ITERATE, 
                      n_bs=N_BS, 
                      fit_yint_vR_constant=FIT_YINT_VR_CONSTANT,
                      force_yint_vR=FORCE_YINT_VR, 
                      force_yint_vR_value=FORCE_YINT_VR_VALUE)

# ----------------------------------------------------------------------------

### Pickle the results
with open('./'+FILENAME+'_master_lm.pickle','wb') as f:
    pickle.dump(lm_mas,f)
##wi
