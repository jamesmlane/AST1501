# ----------------------------------------------------------------------------
#
# TITLE - evaluate_triaxial_df_serial.py
# AUTHOR - James Lane
# PROJECT -
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Core script for evaluating a triaxial DF in serial
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, time

## Plotting
from matplotlib import pyplot as plt
plt.ioff()

## Astropy
from astropy import units as apu

# galpy
from galpy import orbit
from galpy import potential
from galpy import df
from galpy import actionAngle

# Project specific
sys.path.append('../../../src')
import ast1501.df
import ast1501.potential

# ----------------------------------------------------------------------------

# Tunable parameters
halo_b = 1.1

r_range = [5,15] # kpc
phi_range = [-np.pi/2,np.pi/2] # radians
delta_r = 1 # kpc
delta_phi = 2 # arc in kpc

velocity_parms = [20,20,8,8]

# ----------------------------------------------------------------------------

# Default parameters
t_evolve, t_form, t_steady = [10,-9,8] # Timing in Gyr
times = -np.array([0,t_evolve]) * apu.Gyr # Times in Gyr
sigma_vR, sigma_vT, sigma_vZ = [30,30,20] # Velocity dispersions

# Make the position arrays
grid_rpoints, grid_phipoints = ast1501.df.generate_grid_radial( r_range, 
    phi_range, delta_r, delta_phi, delta_phi_in_arc=True)

# Make the potential
mwpot = potential.MWPotential2014
halo_a = 1.0
halo_phi = 0.0
halo_c = 1.0
trihalo = ast1501.potential.make_MWPotential2014_triaxialNFW(halo_b=halo_b, 
    halo_phi=halo_phi, halo_c=halo_c)
tripot_grow = ast1501.potential.make_tripot_dsw(trihalo=trihalo, tform=t_form, 
    tsteady=t_steady)
potential.turn_physical_off(tripot_grow)

# Action angle coordinates and the DF
qdf_aA= actionAngle.actionAngleAdiabatic(pot=potential.MWPotential2014, c=True)
qdf = df.quasiisothermaldf( hr= 2*apu.kpc, sr= sigma_vR*(apu.km/apu.s),
                            sz= sigma_vZ*(apu.km/apu.s),
                            hsr= 9.8*(apu.kpc), hsz= 7.6*(apu.kpc),
                            pot= potential.MWPotential2014, aA= qdf_aA)

# ----------------------------------------------------------------------------

# Make the log
logfile = open('./log.txt','w')
t_total_1 = time.time()

# Now run the serial evaluator
results = ast1501.df.evaluate_df_polar_serial(grid_rpoints,grid_phipoints,
    tripot_grow,qdf,velocity_parms,times,sigma_vR=sigma_vR,sigma_vT=sigma_vT,
    evaluator_threshold=0.0001,plot_df=False,coords_in_xy=False,
    logfile=logfile,verbose=0)

# Close the log
t_total_2 = time.time()
logfile.write('Took '+str(round(t_total_2-t_total_1))+' s total')
logfile.close()

# Save results
np.save('results.npy',results)
