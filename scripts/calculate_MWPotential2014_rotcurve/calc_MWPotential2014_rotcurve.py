# ----------------------------------------------------------------------------
#
# TITLE - calculate_MWPotential2014_rotcurve.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Calculate the rotation curve of MWPotential2014 using the quasi-isothermal 
distribution function, which will give a slightly different result than 
calculating the circular velocity curve.
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, time

## Plotting
from matplotlib import pyplot as plt

## Astropy
from astropy import units as apu

## galpy
from galpy import potential, actionAngle, df

## scipy
from scipy import interpolate

# ----------------------------------------------------------------------------

svr, svz = [30, 20] # km/s
r_scale, vr_scale, vz_scale = [2, 9.8, 7.6] # kpc
aA = actionAngle.actionAngleAdiabatic( pot=potential.MWPotential2014, 
                                       c=True)
qdf = df.quasiisothermaldf(hr= r_scale*apu.kpc, 
                           sr= svr*(apu.km/apu.s),
                           sz= svz*(apu.km/apu.s),
                           hsr= vr_scale*(apu.kpc), 
                           hsz= vz_scale*(apu.kpc),
                           pot= potential.MWPotential2014, 
                           aA= aA)

# Declare the radial range and spacing
R_range = [3,17] # in kpc
delta_R = 0.5 # in kpc
Rs = np.arange(R_range[0],R_range[1],delta_R)
vTs = np.zeros_like(Rs)

# Calculate the mean tangential velocity
t1 = time.time()
for i in range( len(Rs) ):
    
    vTs[i] = qdf.meanvT(Rs[i]/8, z=0, gl=True, nsigma=10, ngl=40)*220
    # vTs[i] = qdf.meanvT(Rs[i]/8, z=0)*220
    
###i
t2 = time.time()

print(str(t2-t1)+' s')

# Interpolate on the results
spline = interpolate.interp1d(Rs, vTs, kind='cubic')
new_Rs = np.arange(R_range[0],R_range[1]-1,0.01)
new_vTs = spline(new_Rs)

plt.scatter(Rs, vTs, color='Blue')
plt.plot(new_Rs, new_vTs, color='Red')
plt.axvline(5, linestyle='dashed', color='Black')
plt.axvline(15, linestyle='dashed', color='Black')
plt.xlabel('R [kpc]')
plt.ylabel(r'$v_{T}$ [km/s]')
plt.savefig('df_inferred_rotcurve.png')

# Save the results?
np.save('MWPotential2014_DF_vT_data.npy', np.array([new_Rs,new_vTs]) )