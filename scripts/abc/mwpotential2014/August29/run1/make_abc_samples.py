# ----------------------------------------------------------------------------
#
# TITLE - make_abc_samples.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Script to make ABC samples for the triaxial halo project.

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
sys.path.append('../../../../../src/')
from ast1501.linear_model import LinearModel
from ast1501.linear_model import LinearModel2
from ast1501.linear_model import LinearModelSolution
import ast1501.abc
import ast1501.potential

# ----------------------------------------------------------------------------

### Load parameters

# Load parameters from the YAML file
PARAMETER_FILE = '../abc_parameters.yaml'
parameter_dict = ast1501.abc.load_abc_params(PARAMETER_FILE)
locals().update(parameter_dict)

VT_PRIOR_PATH = '../'+VT_PRIOR_PATH

# ----------------------------------------------------------------------------

### Get bar data
print('Reading data...')

## Names
bar_model_names = glob.glob('../../../../../data/df/MWPotential2014/bar/*.npy')
n_bar_models = len(bar_model_names)

## Pre-known values for the bar models
bar_model_af_vals = np.array([0.01,0.015,0.02,0.025,0.01,0.015,0.02,0.025,0.01,
                              0.015,0.02,0.025,0.01,0.015,0.02,0.025,0.01,0.015,
                              0.02,0.025,0.01,0.015,0.02,0.025])
bar_model_omegab_vals = np.array([35,35,35,35,40,40,40,40,45,45,45,45,50,50,50,
                                  50,55,55,55,55,60,60,60,60])

## Load the data
bar_models = []
for i in range(len(bar_model_names)):
    bar_model_data_temp = np.load(bar_model_names[i])
    bar_models.append(bar_model_data_temp)
###i
assert n_bar_models == len(bar_model_af_vals) and\
       n_bar_models == len(bar_model_omegab_vals),\
       'Missmatched number of bar parameters'

# for i in range(n_bar_models):
#     print(bar_model_names[i])
#     print(bar_model_af_vals[i])
#     print(bar_model_omegab_vals[i])
#     print('\n')

# ----------------------------------------------------------------------------

### Read in the master
print('Reading the master linear model...')
master_filename = '../'+FILENAME+'_master_lm.pickle'
with open(master_filename,'rb') as f:
    lm_mas = pickle.load(f)
##wi

R_bin_cents_mas, phi_bin_cents_mas = lm_mas.get_bs_sample_positions()
phi_err_mas = lm_mas.get_bs_phi_errors()
vR_mas, vR_err_mas, vT_mas, vT_err_mas = lm_mas.get_bs_velocities()
n_pts_mas = len(R_bin_cents_mas)

# ----------------------------------------------------------------------------

### Run the ABC

# Generate the b/a sample and position angle sample
b_mc_sample = np.random.uniform(low=TH_B_LOW, high=TH_B_HI, size=N_ABC_SAMPLES)
pa_mc_sample = np.random.uniform(low=TH_PA_LOW, high=TH_PA_HI, size=N_ABC_SAMPLES)

# Generate the bar samples
bar_mc_sample = np.random.randint(low=0, high=n_bar_models, size=N_ABC_SAMPLES)

# Array to hold ABC linear models and linear model solutions
lm_arr = []
lm_sol_arr = []

# Setup the Milky Way potential for calculating the circular velocity
mwpot = potential.MWPotential2014

# Generate the linear models
for i in tqdm.tqdm(range(N_ABC_SAMPLES),desc='We waitses'):
    
    # Generate the MC samples of the Gaia data and parameters
    th_b = b_mc_sample[i]
    th_pa = pa_mc_sample[i]
    
    # Make the Kuijken model for these parameters and generate the velocities
    kt = ast1501.potential.kuijken_potential(b_a=th_b, phib=th_pa)
    kt_vR = kt.kuijken_vr(R=R_bin_cents_mas, phi=phi_bin_cents_mas)
    kt_vT = kt.kuijken_vt(R=R_bin_cents_mas, phi=phi_bin_cents_mas)
    
    # Sample from the normal distribution and scale by the known data errors
    kt_vR_pert = np.random.normal(loc=0.0, scale=1.0, size=n_pts_mas) * vR_err_mas
    kt_vT_pert = np.random.normal(loc=0.0, scale=1.0, size=n_pts_mas) * vT_err_mas
    
    # Interpolate on the bar model
    vR_bar, vT_bar = ast1501.abc.interpolate_bar_model(R_bin_cents_mas, 
        phi_bin_cents_mas, bar_models[bar_mc_sample[i]])
    # vT_circ = potential.vcirc(mwpot,R_bin_cents_mas/8)*220
    # vT_bar -= vT_circ
    
    # Apply the noise and bar perturbations to the simulated data
    kt_vR = kt_vR + kt_vR_pert + vR_bar
    kt_vT = kt_vT + kt_vT_pert + vT_bar
            
    # Stitch into a bootstrap sample
    kt_bs_sample_vR, kt_bs_sample_vT = \
        ast1501.linear_model.make_data_like_bootstrap_samples(R_bin_cents_mas,
        phi_bin_cents_mas, kt_vR, kt_vT, vT_err=vR_err_mas, vR_err=vR_err_mas,
        phi_err=phi_err_mas)
    
    # Make the linear model
    lm_kt = LinearModel2(instantiate_method=3, 
        bs_sample_vR=kt_bs_sample_vR, 
        bs_sample_vT=kt_bs_sample_vT, 
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
    
    # Make the linear model solution
    if 'vR' in USE_VELOCITIES and 'vT' in USE_VELOCITIES:
        lm_kt_sol = LinearModelSolution(
            use_velocities=USE_VELOCITIES,
            th_b=th_b,
            th_pa=th_pa,
            bar_omega_b=bar_model_omegab_vals[bar_mc_sample[i]],
            bar_af=bar_model_af_vals[bar_mc_sample[i]],
            b_vR=lm_kt.b_vR,
            m_vR=lm_kt.m_vR,
            b_vT=lm_kt.b_vT,
            m_vT=lm_kt.m_vT,
            b_err_vR=lm_kt.b_err_vR,
            m_err_vR=lm_kt.m_err_vR,
            b_err_vT=lm_kt.b_err_vT,
            m_err_vT=lm_kt.m_err_vT,
            phiB=lm_kt.phiB)
    elif 'vR' in USE_VELOCITIES:
        lm_kt_sol = LinearModelSolution(
            use_velocities=USE_VELOCITIES,
            th_b=th_b,
            th_pa=th_pa,
            bar_omega_b=bar_model_omegab_vals[bar_mc_sample[i]],
            bar_af=bar_model_af_vals[bar_mc_sample[i]],
            b_vR=lm_kt.b_vR,
            m_vR=lm_kt.m_vR,
            b_err_vR=lm_kt.b_err_vR,
            m_err_vR=lm_kt.m_err_vR,
            phiB=lm_kt.phiB)
    elif 'vT' in USE_VELOCITIES:
        lm_kt_sol = LinearModelSolution(
            use_velocities=USE_VELOCITIES,
            th_b=th_b,
            th_pa=th_pa,
            bar_omega_b=bar_model_omegab_vals[bar_mc_sample[i]],
            bar_af=bar_model_af_vals[bar_mc_sample[i]],
            b_vT=lm_kt.b_vT,
            m_vT=lm_kt.m_vT,
            b_err_vT=lm_kt.b_err_vT,
            m_err_vT=lm_kt.m_err_vT,
            phiB=lm_kt.phiB)
    ##ie
    
    # Append the linear model and the solution
    lm_arr.append(lm_kt)
    lm_sol_arr.append(lm_kt_sol)
    
    
###i

# ----------------------------------------------------------------------------

### Pickle the results
print('Saving results...')

# Samples
with open('./'+FILENAME+'_samples_lm.pickle','wb') as f:
    pickle.dump(lm_arr,f)
##wi
# Sample solutions
with open('./'+FILENAME+'_solutions_lm.pickle','wb') as f:
    pickle.dump(lm_sol_arr,f)
##wi

# ----------------------------------------------------------------------------
