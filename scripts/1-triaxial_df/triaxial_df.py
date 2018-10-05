# ----------------------------------------------------------------------------
#
# TITLE - triaxial_df.py
# AUTHOR - James Lane
# PROJECT - AST1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Determine the effects of growing a triaxial halo on the distribution function
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb
import copy
# import glob
# import subprocess

## Plotting
import matplotlib
from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib import colors
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
from galpy.actionAngle import actionAngleAdiabatic
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv

# ----------------------------------------------------------------------------

## Keywords

# Time in Gyr when the simulation will 'begin' (In the past!)
t0 = 10

# The times at which the triaxial halo will begin to grow and end growing
# with respect to t0
t_tri_begin = 1 # 1 Gyr after the simulation starts
t_tri_end = 9 # 9 Gyr after the simulation starts

# Set the velocity dispersions in km/s
sigma_vR = 46
sigma_vT = 40
sigma_vZ = 28

# Solar rotational velocity for grid setup
vrot_sol = 220 # In km/s

# Evaluation radius
r_eval = 8. # In kpc

# Name of the output directory
dir = './run4'


# ----------------------------------------------------------------------------

## Setup directory for storage

# Check if exists, if it does do not overwrite. Then make it.
if os.path.isdir(dir) == True:
    sys.exit('dir already exists, select different name')
##fi
os.mkdir(dir)

# ----------------------------------------------------------------------------

## Setup potentials

# Get MWPotential2014
mwpot = potential.MWPotential2014
mwbulge = copy.deepcopy(mwpot[0])
mwdisk = copy.deepcopy(mwpot[1])
mwhalo = copy.deepcopy(mwpot[2])

# Make the potential
mwhalo_a = mwhalo.a * mwhalo._ro * apu.kpc
mwhalo_amp = mwhalo.dens(mwhalo_a,0) * 16 * mwhalo.a**3 * np.pi * \
             gpconv.mass_in_msol(mwhalo._vo, mwhalo._ro) * apu.M_sun

mwdisk_a = mwdisk._a * mwdisk._ro * apu.kpc
mwdisk_b = mwdisk._b * mwdisk._ro * apu.kpc
mwdisk_amp = mwdisk._amp * gpconv.mass_in_msol(mwdisk._vo, mwdisk._ro) * apu.M_sun

mwbulge_r1 = 1
mwbulge_amp = mwbulge.dens(mwbulge_r1,0) * np.exp((1/mwbulge.rc)**2) * \
              gpconv.dens_in_msolpc3(mwhalo._vo, mwhalo._ro) * apu.M_sun / apu.pc**3
mwbulge_alpha = mwbulge.alpha
mwbulge_rc = mwbulge.rc * mwbulge._ro * apu.kpc

# Generate the scalped potentials
mwbulge = potential.PowerSphericalPotentialwCutoff(amp=mwbulge_amp, alpha=mwbulge_alpha, rc=mwbulge_rc)
mwbulge.turn_physical_off()
mwdisk = potential.MiyamotoNagaiPotential(amp=mwdisk_amp, a=mwdisk_a, b=mwdisk_b)
mwdisk.turn_physical_off()
mwhalo = potential.NFWPotential(amp=mwhalo_amp, a=mwhalo_a)
mwhalo.turn_physical_off()
mwpot = [mwhalo, mwdisk, mwbulge]

# Make the negative amplitude NFW and wrap it in a DehnenSmoothWrapperPotential
mwhalo_rev = potential.NFWPotential(amp=mwhalo_amp*-1,
                                    a=mwhalo_a)
mwhalo_rev.turn_physical_off()
mwhalo_rev_dsw = potential.DehnenSmoothWrapperPotential(pot=mwhalo_rev,
                                        tform= -(t0-t_tri_begin) * apu.Gyr,
                                        tsteady= (t_tri_begin-t_tri_end) * apu.Gyr
                                        )
mwhalo_rev_dsw.turn_physical_off()

# ----------------------------------------------------------------------------

## Setup the DF

# Make the quasi-isothermal distribution function.
# See notebook #3 for the velocity dispersions and scales.
qdf_aA= actionAngleAdiabatic(pot= potential.MWPotential2014,c=True)
qdf = df.quasiisothermaldf(hr= 2*apu.kpc,
                            sr= sigma_vR*(apu.km/apu.s),
                            sz= sigma_vZ*(apu.km/apu.s),
                            hsr= 9.8*(apu.kpc),
                            hsz= 7.6*(apu.kpc),
                            pot= potential.MWPotential2014,
                            aA= qdf_aA)

# ----------------------------------------------------------------------------

## Setup the integration grids and prepare for integration

# Deltas
delta_vR = 10.
delta_vT = 10.

# Set ranges
vR_range = np.arange( -2*sigma_vR, 2*sigma_vR+1, delta_vR )
vT_range = np.arange( -2*sigma_vT, 2*sigma_vT+1, delta_vT )+vrot_sol

# This goes along with a grid of distribution function values
dfp = np.zeros((len(vR_range),len(vT_range)))
df0 = np.zeros((len(vR_range),len(vT_range)))

# Print the number of velocities
print( 'Number of grid velocities: '+str( len(vR_range)*len(vT_range)) )

# Now set the triaxial properties which will be used
tri_b = np.linspace(0.5, 2, num=3)
tri_phi = np.linspace(0, np.pi/2, num=3)

# Set the orbit times
times = -np.array([0,10]) * apu.Gyr

# ----------------------------------------------------------------------------

## Loop over the triaxial properties

# Get the data ready to be stored
df0_out = np.empty( (len(tri_b),len(tri_phi)), dtype='object' )
dfp_out = np.empty( (len(tri_b),len(tri_phi)), dtype='object' )

counter = 1

print('Done 0/'+str(len(tri_b)*len(tri_phi)))

for i in range( len(tri_b) ):
    for j in range( len(tri_phi) ):

# ----------------------------------------------------------------------------

        ## Make the time-dependant potential

        # Generate the triaxial potential and wrap it
        trihalo = potential.TriaxialNFWPotential(amp=mwhalo_amp,
                                                a=mwhalo_a,
                                                b=tri_b[i],
                                                c=1.0,
                                                pa=tri_phi[j])
        trihalo_dsw = potential.DehnenSmoothWrapperPotential(pot=trihalo,
                                    tform= -(t0-t_tri_begin) * apu.Gyr,
                                    tsteady= (t_tri_begin-t_tri_end) * apu.Gyr
                                    )
        trihalo_dsw.turn_physical_off()

        # Add it together
        tripot = [ mwhalo_rev_dsw,
                   mwhalo,
                   trihalo_dsw,
                   mwdisk,
                   mwbulge ]

# ----------------------------------------------------------------------------

        ## Loop over the velocities

        for k in range( len(vR_range) ):
            for l in range( len(vT_range) ):

                # Make the orbit
                o = orbit.Orbit(vxvv=[  r_eval*apu.kpc,
                                        vR_range[k]*apu.km/apu.s,
                                        vT_range[l]*apu.km/apu.s,
                                        0.*apu.kpc,
                                        0.*apu.km/apu.s,
                                        0.*apu.radian])

                # Evaluate the orbit in the qDF
                df0[k,l] = qdf(o)

                # Integrate
                o.integrate(times, tripot)

                # Now evaluate perturbed DR using the qDF and integrated orbit
                dfp[k,l] = qdf(o(times[1]))

            ###l
        ###k

# ----------------------------------------------------------------------------

        ## Now save the results
        df0_out[i,j] = df0
        dfp_out[i,j] = dfp


        print('Done '+str(counter)+'/'+str(len(tri_b)*len(tri_phi)))
        counter += 1
    ###j
###i

# Now output the results
data_out = np.array((tri_b,tri_phi,vR_range,vT_range,df0_out,dfp_out))
np.save(dir+'/results.npy', arr=data_out, allow_pickle=True)

# Now output a .readme
readme = open(dir+'/readme.txt', 'w')
readme.write('nb: '+str(len(tri_b))+', nphi: '+str(len(tri_phi)))
readme.write('\nb: ')
for b in tri_b:
    readme.write(str(b)+', ')
readme.write('\nphi: ')
for phi in tri_phi:
    readme.write(str(phi)+', ')
readme.close()
# ----------------------------------------------------------------------------
