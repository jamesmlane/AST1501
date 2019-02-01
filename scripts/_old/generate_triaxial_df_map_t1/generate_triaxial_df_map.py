# ----------------------------------------------------------------------------
#
# TITLE - generate_triaxial_df_map.py
# AUTHOR - James Lane
# PROJECT - AST 1501
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Make observable maps for a triaxial halos of chosen properties
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, math, time
# import glob
# import subprocess

## Plotting
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
from matplotlib import cm

## Astropy
# from astropy.io import fits
# from astropy.coordinates import SkyCoord
# from astropy import table
from astropy import units as apu
# from astropy import wcs

## galpy
from galpy import orbit
from galpy import potential
from galpy import df
from galpy.actionAngle import actionAngleAdiabatic,actionAngleStaeckel
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv

## Import project-specific modules
sys.path.append('../../src/')
import ast1501.potential
import ast1501.util
import ast1501.df

# ----------------------------------------------------------------------------

## Keywords

# Should decide on the halo properties. Perhaps read these from a file to do 
# it in an automated fashion.

# Otherwise pick properties for the simulation

# Set the grid spacing
# dx = 1 # kpc
# dy = 1 # kpc
dr = 1 # kpc
darc = 3 # kpc

# Can save time by mirroring across the x=0 line
# range_x = [0,15] # kpc
# range_y = [-10,10] # kpc

range_r = [7,15]
range_phi = [0,np.pi]

mirror_x = True # if True then only evaluate positive x and flip it

# Pick the halo parameters
halo_b = 1.2
halo_c = 1
halo_phi = 0

# Pick the time parameters
t_evolve = 10 # Gyr
tform = -9 # Gyr ago
tsteady = 8 # Gyr after tform

# The times over which each orbit will be integrated
times = -np.array([0,t_evolve]) * apu.Gyr

### Make MWPotential2014
mwpot = potential.MWPotential2014

### Make the triaxial halo
trihalo = ast1501.potential.make_MWPotential2014_triaxialNFW(halo_b=halo_b, 
    halo_phi=halo_phi, halo_c=halo_c)

### Make MWPotential2014 with DSW around the halo and triaxial halo
tripot_grow = ast1501.potential.make_tripot_dsw(trihalo=trihalo, tform=tform, tsteady=tsteady)
potential.turn_physical_off(tripot_grow)

# Velocity dispersions in km/s
sigma_vR = 46/1.5
sigma_vT = 40/1.5
sigma_vZ = 28/1.5

# Action angle coordinates and the DF
qdf_aA= actionAngleAdiabatic(pot=potential.MWPotential2014, c=True)
qdf = df.quasiisothermaldf( hr= 2*apu.kpc,
                            sr= sigma_vR*(apu.km/apu.s),
                            sz= sigma_vZ*(apu.km/apu.s),
                            hsr= 9.8*(apu.kpc),
                            hsz= 7.6*(apu.kpc),
                            pot= potential.MWPotential2014, 
                            aA= qdf_aA)

# Set velocity deltas and range
dvT = 20.
dvR = 20.
vR_low = -8*sigma_vR
vR_hi = 8*sigma_vR

# ----------------------------------------------------------------------------

## Read the data if it's going to be done from a file

# ----------------------------------------------------------------------------

# First generate the grid of positions to evaluate
# x_posns = np.arange( range_x[0], range_x[1], dx ) + dx/2
# y_posns = np.arange( range_y[0], range_y[1], dy ) + dy/2

r_posns = np.arange( range_r[0], range_r[1], dr) + dr/2
pdb.set_trace()
# Will have to define the range in arc within the for loop

# ----------------------------------------------------------------------------

logfile = open('data/log.txt','w')

radius_out = np.empty(len(r_posns),dtype='object')

# Loop over radii
for i in range( len( r_posns ) ):
    
    # Determine the range of azimuth. Determine this using the requested arc 
    # resolution
    
    arc_length = np.pi*r_posns[i]
    n_bins = math.ceil(arc_length/darc)
    dphi = (range_phi[1]-range_phi[0])/n_bins
    
    # Choose the phi positions. Goes from -pi/2 to +pi/2. Phi=0 is towards the 
    # sun.
    phi_posns = np.linspace(range_phi[0], range_phi[1], num=n_bins, 
        endpoint=False) + dphi/2 - np.pi/2
    
    # Arrays
    vR_mean = np.zeros(len(phi_posns))
    vR_std = np.zeros(len(phi_posns))
    vT_mean = np.zeros(len(phi_posns))
    vT_std = np.zeros(len(phi_posns))
    
    # Now loop over phi positions:
    for j in range( len( phi_posns ) ):
        
        # Set the tangential velocity to about match the local circular velocity
        vcirc_offset = potential.vcirc(mwpot,r_posns[i]*apu.kpc) * mwpot[0]._vo # in km/s
        vT_low = -8*sigma_vT + vcirc_offset
        vT_hi = 8*sigma_vT + vcirc_offset
        
        # Make the velocity range, and distribution function arrays
        _, dfp, vR_range, vT_range = ast1501.df.gen_vRvT_1D(dvT, dvR, vR_low, vR_hi, vT_low, vT_hi)

        R_z_phi = [r_posns[i],0,phi_posns[j]]

        # Now use the radius and phi position to evaluate the triaxial DF 
        t1 = time.time()
        
        dfp = ast1501.df.evaluate_df_adaptive_vrvt(R_z_phi, times, tripot_grow, 
            qdf, vR_range, vT_range, dfp, threshold=0.0001)
            
        # Calculate the moments
        moments = ast1501.df.calculate_df_vmoments(dfp, vR_range, vT_range, 
            dvR, dvT)
        vR_mean[j], vT_mean[j], vR_std[j], vT_std[j] = moments[1:]
        
        t2 = time.time()
        logfile.write( 'R='+str(round(r_posns[i],2))+' kpc, phi='+str(round(phi_posns[j],2))+' rad, t='+str(round(t2-t1))+' s\n' )

        fig = plt.figure( figsize=(5,5) )
        ax = fig.add_subplot(111)

        # Plot
        fig, ax, cbar = ast1501.df.hist_df(dfp, vR_low, vR_hi, vT_low, vT_hi, fig, ax, log=True)
        ax.set_title('R='+str(round(r_posns[i],2))+' kpc, phi='+str(round(phi_posns[j],2))+' rad')
        fig.savefig( 'data/R-'+str(round(r_posns[i],2))+'_phi-'+str(round(phi_posns[j],2))+'_dfp.pdf' )
        
        print( 'Done R: '+str(round(r_posns[i],2))+' phi: '+str(round(phi_posns[j],2)) )
        
    ###j
    
    # Now save the output for this radius
    radius_out[i] = np.array([r_posns[i],phi_posns,vR_mean,vT_mean,vR_std,vT_std],dtype='object')

###i

np.save('data/data_R'+str(round(r_posns[0],2))+'-'+str(round(r_posns[-1],2))+'.npy',radius_out)

logfile.close()

# ----------------------------------------------------------------------------