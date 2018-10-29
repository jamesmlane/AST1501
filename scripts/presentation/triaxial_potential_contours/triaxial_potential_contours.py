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
Make two figures of the Milky Way potential. One with an NFW halo and one 
with a triaxial NFW halo
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy
# import glob
# import subprocess

## Plotting
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm

## Astropy
from astropy import units as apu

## galpy
from galpy import orbit
from galpy import potential
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv
from galpy.util import bovy_plot as gpplot

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

trihalo = potential.TriaxialNFWPotential(amp=mwhalo_amp,
                                        a=mwhalo_a,
                                        b=2.,
                                        c=1.0,
                                        pa=0.)
                                        
tripot = [trihalo, mwdisk, mwbulge]

# ----------------------------------------------------------------------------

potential.plotPotentials(tripot,xy=True, rmin=-20, rmax=20, zmin=-20, zmax=20,
                            nrs=100, nzs=100)
fig = plt.gcf()
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig('tripot_contours.pdf')
plt.close('all')

potential.plotPotentials(mwpot,xy=True, rmin=-20, rmax=20, zmin=-20, zmax=20,
                            nrs=100, nzs=100)
fig = plt.gcf()
ax = plt.gca()
ax.set_aspect('equal')
plt.savefig('mwpot_contours.pdf')
plt.close('all')

