# ----------------------------------------------------------------------------
#
# TITLE - potential.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
#   
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Defined functions for the AST 1501 project: Potential utilities
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy
# import glob
# import subprocess

## Astropy
# from astropy.io import fits
# from astropy.coordinates import SkyCoord
# from astropy import table
from astropy import units as apu

## galpy
# from galpy import orbit
from galpy import potential
from galpy.util import bovy_conversion as gpconv
# from galpy.util import bovy_coords as gpcoords

# ----------------------------------------------------------------------------

def get_MWPotential2014():
    '''get_MWPotential2014:
    
    Return the paramters of galpy's MWPotential2014 object as variables
    with astropy units
    
    Args:
        None
        
    Returns:
        parm_arr (numpy array) - array of potential parameters in the form:
            [blg_alpha, blg_rc, blg_amp, dsk_a, dsk_b, dsk_amp, halo_a, halo_amp]
    '''
    
    # Get MWPotential2014, unpack the component potentials, and save copies
    mwpot = potential.MWPotential2014
    mwbulge = copy.deepcopy(mwpot[0])
    mwdisk = copy.deepcopy(mwpot[1])
    mwhalo = copy.deepcopy(mwpot[2])
    
    mwbulge_r1 = 1
    mwbulge_alpha = mwbulge.alpha
    mwbulge_rc = mwbulge.rc * mwbulge._ro * apu.kpc
    mwbulge_amp = mwbulge.dens(mwbulge_r1,0) * np.exp((1/mwbulge.rc)**2) * \
                  gpconv.dens_in_msolpc3(mwhalo._vo, mwhalo._ro) * apu.M_sun / apu.pc**3 
    
    mwdisk_a = mwdisk._a * mwdisk._ro * apu.kpc
    mwdisk_b = mwdisk._b * mwdisk._ro * apu.kpc
    mwdisk_amp = mwdisk._amp * gpconv.mass_in_msol(mwdisk._vo, mwdisk._ro) * apu.M_sun
    
    mwhalo_a = mwhalo.a * mwhalo._ro * apu.kpc
    mwhalo_amp = mwhalo.dens(mwhalo_a,0) * 16 * mwhalo.a**3 * np.pi * \
                 gpconv.mass_in_msol(mwhalo._vo, mwhalo._ro) * apu.M_sun
    
    parm_arr = np.array([   mwbulge_alpha, mwbulge_rc, mwbulge_amp,
                            mwdisk_a, mwdisk_b, mwdisk_amp,
                            mwhalo_a, mwhalo_amp
                        ], dtype='object')

    return parm_arr
#def

def make_MWPotential2014_triaxialNFW(  halo_b=1.0, halo_phi=0.0, halo_c=1.0
                                    halo_amp=None, halo_a=None):
    '''make_MWPotential2014_triaxial:
    
    Generate a triaxial NFW MW halo
    
    Args:
        halo_b (float) - Halo secondary to primary axis ratio (b/a) [1.0]
        halo_phi (float) - Halo primary axis position angle in radians [0.0]
        halo_c (float) - Halo tertiary to primary axis ratio (c/a) [1.0]
        halo_amp (float) - Halo amplitude. If None it will be identical 
            to MWPotential2014 [None]
        halo_a (float) - Halo scale length. If None it will be identical 
            to MWPotential2014 [None]
        
    Returns:
        TriaxialNFW object    
    '''

    _, _, _, _, _, _, mwhalo_a, mwhalo_amp = get_MWPotential2014()
    
    # Check argument choices
    if halo_mass == None:
        use_halo_amp = mwhalo_amp
    else: use_halo_amp = halo_mass
    ##ie
    if halo_a == None:
        use_halo_a = mwhalo_a
    else: use_halo_a = mwhalo_a
    ##ie
    
    return potential.TriaxialNFW(amp=use_mwhalo_amp, a=use_mwhalo_a, b=halo_b, 
                                    pa=halo_phi, c=halo_c)
#def

def make_tripot_dsw(trihalo, tform, tsteady, 
                        mwhalo=None, mwdisk=None, mwbulge=None):
    '''make_tripot_dsw
    
    Generate a triaxial smoothly varying NFW profile that interpolates between 
    a spherical halo and a triaxial halo.
    
    Args:
        halo (galpy Potential object) - The triaxial halo to introduce
        tform (float) - Time of triaxial halo formation in Gyr (No astropy units).
        tsteady (float) - Time of finished triaxial halo formation in Gyr (No astropy units).
        mwhalo (galpy Potential object) - Halo model to slowly remove. If None 
            then use the same as MWPotential2014 [None]
        mwdisk (galpy Potential object) - Disk model to use. If None then use 
            the same as MWPotential2014 [None]
        mwbulge (galpy Potential object) - bulge model to use. If None then use 
            the same as MWPotential2014 [None]
        
        
    Returns:
        tripot (galpy Potential object array) - triaxial time varying potential  
    '''

    # Get MWPotential2014 parameters
    mwbulge_alpha, mwbulge_rc, mwbulge_amp, mwdisk_a, mwdisk_b, mwdisk_amp,\
        mwhalo_a, mwhalo_amp = get_MWPotential2014()
    
    # Check potential arguments
    if mwbulge == None
        use_mwbulge = potential.PowerSphericalPotentialwCutoff(amp=mwbulge_amp, 
            alpha=mwbulge_alpha, rc=mwbulge_rc)
    else: use_mwbulge = mwbulge
    ##ie
    if mwdisk == None
        use_mwdisk = potential.MiyamotoNagaiPotential(amp=mwdisk_amp, 
            a=mwdisk_a, b=mwdisk_b)
    else: use_disk = mwdisk
    ##ie
    if mwhalo == None
        use_mwhalo = potential.NFWPotential(amp=mwhalo_amp, a=mwhalo_a)
    else: use_mwhalo = mwhalo
    ##ie
    
    # Wrap the old halo in a DSW
    mwhalo_decay_dsw = potential.DehnenSmoothWrapperPotential(pot=mwhalo, 
        tform=tform*apu.Gyr, tsteady=tsteady*apu.Gyr, decay=True)
    
    # Wrap the triaxial halo in a DSW:
    trihalo_dsw = potential.DehnenSmoothWrapperPotential(pot=trihalo,
        tform=tform*apu.Gyr, tsteady=tsteady*apu.Gyr)
        
    return [mwbulge, mwdisk, mwhalo, mwhalo_decay_dsw, trihalo_dsw]

#
