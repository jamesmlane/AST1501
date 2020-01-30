# ----------------------------------------------------------------------------
#
# TITLE - make_abc_samples.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Script to analyze and plot results from the ABC analysis.

Run on August 29

6-12 kpc, 1kpc bins, vR only, LinearModel2, 
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, importlib, glob, pickle, tqdm

## Plotting
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
from matplotlib import cm

## Add project-specific package. Assume relative location
sys.path.append('../../../../src/')
from ast1501.linear_model import LinearModel
from ast1501.linear_model import LinearModel2
from ast1501.linear_model import LinearModelSolution
import ast1501.abc
import ast1501.potential

plt.rc('text', usetex=True)

# ----------------------------------------------------------------------------

### Set the parameters for the matching and plotting
EPSILON_N_SIGMA=1.0

# Load parameters from the YAML file
PARAMETER_FILE = './abc_parameters.yaml'
parameter_dict = ast1501.abc.load_abc_params(PARAMETER_FILE)
locals().update(parameter_dict)

# ----------------------------------------------------------------------------

### Load data
print('Reading data...')

## Bar names
bar_model_names = glob.glob('../../../../data/df/MWPotential2014/bar/*.npy')
n_bar_models = len(bar_model_names)

## Pre-known values for the bar models
bar_model_af_vals = np.array([0.01,0.015,0.02,0.025,0.01,0.015,0.02,0.025,0.01,
                              0.015,0.02,0.025,0.01,0.015,0.02,0.025,0.01,0.015,
                              0.02,0.025,0.01,0.015,0.02,0.025])
bar_model_omegab_vals = np.array([35,35,35,35,40,40,40,40,45,45,45,45,50,50,50,
                                  50,55,55,55,55,60,60,60,60])

## Load bar models
bar_models = []
for i in range(len(bar_model_names)):
    bar_model_data_temp = np.load(bar_model_names[i])
    bar_models.append(bar_model_data_temp)
###i
assert n_bar_models == len(bar_model_af_vals) and\
       n_bar_models == len(bar_model_omegab_vals),\
       'Missmatched number of bar parameters'

## Read in the master
print('Reading the master linear model...')
master_filename = './'+FILENAME+'_master_lm.pickle'
with open(master_filename,'rb') as f:
    lm_mas = pickle.load(f)
##wi 

# Assume samples they are contained in directories called run*, contain files 
#that being with FILENAME, and are called _solutions_*
lm_solutions = []
run_directories = glob.glob('run*')
for i in range(len(run_directories)):
    
    # Load the solutions from this directory
    this_directory = run_directories[i]
    solution_files = glob.glob(this_directory+'/'+FILENAME+'_solutions_*.pickle')
    
    # Load each of the solution files
    for j in range(len(solution_files)):
        
        with open(solution_files[j],'rb') as f:
            lm_solution_new = pickle.load(f)
            print('Adding '+str(len(lm_solution_new))+' linear models')
            lm_solutions.extend(lm_solution_new)
        ##wi
    ###j
    
###i

n_abc_solutions = len(lm_solutions)

# ----------------------------------------------------------------------------

### Find the good matches

# Load master model properties based on which velocities are used
if 'vR' in USE_VELOCITIES:
    lm_mas_m_vR = lm_mas.m_vR
    lm_mas_m_vR_err = lm_mas.m_err_vR
##fi

if 'vT' in USE_VELOCITIES:
    lm_mas_m_vT = lm_mas.m_vT
    lm_mas_m_vT_err = lm_mas.m_err_vT
##fi

# Array of matches. Will be indexed if the file is a match
matches = np.zeros(n_abc_solutions)

all_bars = np.zeros(n_abc_solutions)

# Loop over the samples
for i in range( n_abc_solutions ):
    
    lm_sol = lm_solutions[i]
    
    all_bars[i] = lm_sol.bar_omega_b
    
    if 'vR' in USE_VELOCITIES:
        lm_sol_m_vR = lm_sol.m_vR
        lm_sol_m_vR_err = lm_sol.m_err_vR
    ##fi
    
    if 'vT' in USE_VELOCITIES:
        lm_sol_m_vT = lm_sol.m_vT
        lm_sol_m_vT_err = lm_sol.m_err_vT
        if SUBTRACT_MEAN_VT:
            lm_sol_m_vT -= lm_sol.b_vT
        ##fi
    ##fi
        
    good_match = -1
    
    if 'vR' in USE_VELOCITIES and 'vT' in USE_VELOCITIES:
        all_match_vR = np.all( np.abs( lm_sol_m_vR - lm_mas_m_vR ) < \
                               EPSILON_N_SIGMA*lm_mas_m_vR_err )
        all_match_vT = np.all( np.abs( lm_sol_m_vT - lm_mas_m_vT ) < \
                               EPSILON_N_SIGMA*lm_mas_m_vT_err )
        if all_match_vR and all_match_vT: good_match = 1
    elif 'vR' in USE_VELOCITIES:
        all_match_vR = np.all( np.abs( lm_sol_m_vR - lm_mas_m_vR ) < \
                               EPSILON_N_SIGMA*lm_mas_m_vR_err )
        if all_match_vR: good_match = 1
    elif 'vT' in USE_VELOCITIES:
        all_match_vT = np.all( np.abs( lm_sol_m_vT - lm_mas_m_vT ) < \
                               EPSILON_N_SIGMA*lm_mas_m_vT_err )
        if all_match_vR and all_match_vT: good_match = 1
    ##ie
    
    matches[i]=good_match
    
###i

where_good_matches = np.where( matches == 1 )[0]
n_good_matches = len(where_good_matches)
where_bad_matches = np.where( matches == -1 )[0]
n_bad_matches = len(where_bad_matches)

# Fill good match arrays
match_th_b = np.zeros(n_good_matches)
match_th_pa = np.zeros(n_good_matches)
match_bar_omega_b = np.zeros(n_good_matches)
match_bar_af = np.zeros(n_good_matches)

for i in range( n_good_matches ):
    
    match_sol = lm_solutions[where_good_matches[i]]
    match_th_b[i] = match_sol.th_b
    match_th_pa[i] = match_sol.th_pa
    match_bar_omega_b[i] = match_sol.bar_omega_b
    match_bar_af[i] = match_sol.bar_af

###i

# ----------------------------------------------------------------------------

### Plot the amplitudes of each solution

fig = plt.figure( figsize=(12,4) )
axs = fig.subplots( nrows=1, ncols=2 )

R_bin_cents = np.arange(R_LIMS[0],R_LIMS[1],R_BIN_SIZE)+R_BIN_SIZE/2

axs[0].errorbar( R_bin_cents, lm_mas.b_vR, yerr=lm_mas.b_err_vR, 
    **{'markerfacecolor':None,'ecolor':'Black','markeredgecolor':'Black',
       'color':'Black','markersize':10,'marker':'o'} )
axs[1].errorbar( R_bin_cents, lm_mas.m_vR, yerr=lm_mas.m_err_vR, 
    **{'markerfacecolor':None,'ecolor':'Black','markeredgecolor':'Black',
       'color':'Black','markersize':10,'marker':'o'} )
       
axs[0].set_xlabel(r'R [kpc]', fontsize=12)
axs[1].set_xlabel(r'R [kpc]', fontsize=12)
axs[0].set_ylabel(r'$b_{R}$ [km/s]', fontsize=12)
axs[1].set_ylabel(r'$m_{R}$ [km/s]', fontsize=12)

# Add the individual solutions
for i in range( n_good_matches ):
    
    lm_sol = lm_solutions[where_good_matches[i]]
    axs[0].axhline( lm_sol.b_vR[0], color='Red', alpha=0.5 )
    axs[1].plot( R_bin_cents, lm_sol.m_vR, color='Red', alpha=0.25 )
    
###i

fig.savefig('model_amplitudes.pdf')

# ----------------------------------------------------------------------------

### Plot the velocities of the data and solutions

R_bin_cents_mas, phi_bin_cents_mas = lm_mas.get_bs_sample_positions()
unique_R_bin_cents_mas = np.unique(R_bin_cents_mas)

fig, axs = lm_mas.plot_velocity_known_m_b_phi(velocity_type='vR')

# Loop over each solution and plot the data
for i in range( n_good_matches ):
    
    # Load the LinearModelSolution
    lm_sol = lm_solutions[where_good_matches[i]]
    th_b = lm_sol.th_b
    th_pa = lm_sol.th_pa
    soln_bar_omega_b = lm_sol.bar_omega_b
    soln_bar_af = lm_sol.bar_af
    
    where_bar_model = np.where( (bar_model_omegab_vals == soln_bar_omega_b) &
                                (bar_model_af_vals == soln_bar_af) )[0][0]
    
    # Load the Kuijken & Tremaine data
    kt = ast1501.potential.kuijken_potential(b_a=0.8, phib=1.5)
    kt_vR = kt.kuijken_vr(R=R_bin_cents_mas, phi=phi_bin_cents_mas)
    
    # Perturb the solution?
    # kt_vR_pert = np.random.normal(loc=0.0, scale=1.0, size=n_pts_mas) * vR_err_mas
    
    # Interpolate on the bar model
    vR_bar, vT_bar = ast1501.abc.interpolate_bar_model(R_bin_cents_mas, 
        phi_bin_cents_mas, bar_models[where_bar_model])
        
    kt_vR = vR_bar# kt_vR + vR_bar # + kt_vR_pert
    
    # Loop over radii
    for j in range( len(unique_R_bin_cents_mas) ):
        
        this_radius = unique_R_bin_cents_mas[j]
        where_cur_radius = np.where( R_bin_cents_mas == this_radius )
        
        axs[j].plot( phi_bin_cents_mas[where_cur_radius], 
                     kt_vR[where_cur_radius], 
                     color='Red', alpha=0.2  
                    )

fig.savefig('solution_velocities.pdf')

# ----------------------------------------------------------------------------

# ### Plot the velocities of the data and good samples instead of the solutions
# 
# ## Read in the master
# print('Reading good full sample linear models...')
# master_filename = './'+FILENAME+'_samples_good_lm.pickle'
# with open(master_filename,'rb') as f:
#     lm_samples = pickle.load(f)
# ##wi 
# 
# R_bin_cents_mas, phi_bin_cents_mas = lm_mas.get_bs_sample_positions()
# unique_R_bin_cents_mas = np.unique(R_bin_cents_mas)
# 
# fig, axs = lm_mas.plot_velocity_known_m_b_phi(velocity_type='vR')
# 
# # Loop over each solution and plot the data
# for i in range( n_good_matches ):
# 
#     # Load the LinearModelSolution
#     lm_samp = lm_samples[i]
# 
#     pdb.set_trace()
# 
#     # th_b = lm_sol.th_b
#     # th_pa = lm_sol.th_pa
#     # soln_bar_omega_b = lm_sol.bar_omega_b
#     # soln_bar_af = lm_sol.bar_af
# 
#     # where_bar_model = np.where( (bar_model_omegab_vals == soln_bar_omega_b) &
#     #                             (bar_model_af_vals == soln_bar_af) )[0]
#     # 
#     # # Load the Kuijken & Tremaine data
#     # kt = ast1501.potential.kuijken_potential(b_a=th_b, phib=th_pa)
#     # kt_vR = kt.kuijken_vr(R=R_bin_cents_mas, phi=phi_bin_cents_mas)
#     # 
#     # # Perturb the solution?
#     # # kt_vR_pert = np.random.normal(loc=0.0, scale=1.0, size=n_pts_mas) * vR_err_mas
#     # 
#     # # Interpolate on the bar model
#     # vR_bar, vT_bar = ast1501.abc.interpolate_bar_model(R_bin_cents_mas, 
#     #     phi_bin_cents_mas, bar_models[where_bar_model])
#     # 
#     # kt_vR = kt_vR + vR_bar # + kt_vR_pert
# 
#     samp_R,samp_phi = lm_samp.get_bs_sample_positions()
#     samp_vR,_,_,_ = lm_samp.get_bs_velocities()
#     samp_R_unique = np.unique(samp_R)
# 
#     # Loop over radii
#     for j in range( len(samp_R_unique) ):
# 
#         this_radius = samp_R_unique[j]
#         where_cur_radius = np.where( samp_R == this_radius )
#         axs[j].plot( samp_phi[where_cur_radius], 
#                      samp_vR[where_cur_radius], 
#                      color='Red', alpha=0.2  
#                     )
# 
# fig.savefig('solution_sample_velocities.pdf')

# ----------------------------------------------------------------------------

where_shift_th_pa = np.where( match_th_pa > np.pi/2 )
new_th_pa = match_th_pa[where_shift_th_pa] - np.pi

pdb.set_trace()

### Plot the results

# Just plot single PDFs
fig,ax = ast1501.abc.plot_posterior_histogram(match_th_b,bins=20,
    lims=[TH_B_LOW,TH_B_HI])

ax.set_xlabel('Triaxial Halo b/a')
ax.set_ylabel('Probability')
ax.set_xlim(TH_B_LOW,TH_B_HI)
fig.savefig('th_b_pdf.pdf')

fig,ax = ast1501.abc.plot_posterior_histogram(match_th_pa,bins=20,
    lims=[TH_PA_LOW,TH_PA_HI])
ax.set_xlabel(r'Triaxial Halo $\phi_{B}$')
ax.set_ylabel('Probability')
ax.set_xlim(TH_PA_LOW,TH_PA_HI)
fig.savefig('th_pa_pdf.pdf')

fig,ax = ast1501.abc.plot_posterior_discrete(match_bar_omega_b)
ax.set_xlabel(r'Bar $\Omega_{B}$ [km/s/kpc]')
ax.set_ylabel('Probability')
ax.set_xlim(25,65)
fig.savefig('th_bar_omega_b_pdf.pdf')

fig,ax = ast1501.abc.plot_posterior_discrete(match_bar_af)
ax.set_xlabel(r'Bar $A_{f}$')
ax.set_ylabel('Probability')
ax.set_xlim(0.005,0.03)
fig.savefig('th_bar_af_pdf.pdf')

# Plot the combined PDF
staircase_data = np.array([  match_th_b,
                        match_th_pa,
                        #match_bar_omega_b,
                        #match_bar_af
                     ]).T
staircase_labels = [r'$b/a$',r'$\phi_{b}$ [rad]']
                    #r'$\Omega_{b}$ [km/s/kpc]',r'$A_{f}$']
fig, axs = ast1501.abc.staircase_plot_kernel(staircase_data, staircase_labels)

axs[1,0].set_yticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
axs[1,0].set_yticklabels([r'0',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$'])
axs[1,1].set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
axs[1,1].set_xticklabels([r'0',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$'])
axs[1,1].set_xlabel(r'$\phi_{b}$ [rad]')

# axs[0,0].set_xlim(0.69,1.01)
# axs[1,0].set_xlim(0.69,1.01)

fig.subplots_adjust(wspace=0, hspace=0)
fig.set_facecolor('White')
fig.savefig('staircase_plots.pdf', plot_median=True)

b_a_median = np.median( match_th_b )
b_a_upper_68ci = np.sort( match_th_b )[ int((0.5+0.68/2)*n_good_matches) ]
b_a_lower_68ci = np.sort( match_th_b )[ int((0.5-0.68/2)*n_good_matches) ]
b_a_t1_68ci = np.sort( match_th_b )[ int( (1-0.68)*n_good_matches ) ]
b_a_t1_95ci = np.sort( match_th_b )[ int( (1-0.95)*n_good_matches ) ]

print('b/a:')
print('Median: '+str(b_a_median))
print('68% CI: '+str(b_a_lower_68ci)+', '+str(b_a_upper_68ci))