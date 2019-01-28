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
'''Defined functions for the AST 1501 project: Potential utilities
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy
# import glob
# import subprocess

##
from matplotlib import pyplot as plt

## Astropy
from astropy import units as apu

## galpy
from galpy import orbit
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

def make_MWPotential2014_triaxialNFW(   halo_b=1.0, 
                                        halo_phi=0.0,
                                        halo_c=1.0,
                                        halo_amp=None, 
                                        halo_a=None):
    '''make_MWPotential2014_triaxialNFW:
    
    Generate a triaxial NFW with the same properties as that in MWPotential2014
    
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
    if halo_amp == None:
        use_halo_amp = mwhalo_amp
    else: use_halo_amp = halo_mass
    ##ie
    if halo_a == None:
        use_halo_a = mwhalo_a
    else: use_halo_a = mwhalo_a
    ##ie
    
    return potential.TriaxialNFWPotential(amp=use_halo_amp, a=use_halo_a, 
                                    b=halo_b, pa=halo_phi, c=halo_c)
#def

def make_tripot_dsw(trihalo, t_form, t_steady, 
                        mwhalo=None, mwdisk=None, mwbulge=None):
    '''make_tripot_dsw
    
    Generate a triaxial smoothly varying NFW profile that interpolates between 
    a spherical halo and a triaxial halo.
    
    Args:
        halo (galpy Potential object) - The triaxial halo to introduce
        t_form (float) - Time of triaxial halo formation in Gyr 
            (No astropy units attached).
        t_steady (float) - Time of finished triaxial halo formation in Gyr 
            (No astropy units attached).
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
    if mwbulge == None:
        use_mwbulge = potential.PowerSphericalPotentialwCutoff(amp=mwbulge_amp, 
            alpha=mwbulge_alpha, rc=mwbulge_rc)
    else: use_mwbulge = mwbulge
    ##ie
    if mwdisk == None:
        use_mwdisk = potential.MiyamotoNagaiPotential(amp=mwdisk_amp, 
            a=mwdisk_a, b=mwdisk_b)
    else: use_disk = mwdisk
    ##ie
    if mwhalo == None:
        use_mwhalo = potential.NFWPotential(amp=mwhalo_amp, a=mwhalo_a)
    else: use_mwhalo = mwhalo
    ##ie
    
    # Wrap the old halo in a DSW
    mwhalo_decay_dsw = potential.DehnenSmoothWrapperPotential(pot=use_mwhalo, 
        tform=t_form*apu.Gyr, tsteady=t_steady*apu.Gyr, decay=True)
    
    # Wrap the triaxial halo in a DSW:
    trihalo_dsw = potential.DehnenSmoothWrapperPotential(pot=trihalo,
        tform=t_form*apu.Gyr, tsteady=t_steady*apu.Gyr)
        
    return [use_mwbulge, use_mwdisk, mwhalo_decay_dsw, trihalo_dsw]

#

def find_closed_orbit(pot,Lz,rtol=0.001,R0=1.0,vR0=0.0,plot_loop=False):
    '''find_closed_orbit:
    
    Calculate a closed orbit for a given angular momentum
    
    Args:
        pot (galpy Potential object) - Potential for which to determine 
            the closed orbit
        Lz (float) - Angular momentum for the closed orbit
        rtol (float) - change in radius marking the end of the search [0.01]
        R0 (float) - Starting radius [1.0]
        vR0 (float) - Starting radial velocity [0.0]
        plot_loop (bool) - Plot the surface during each loop evaluation? [False]
    
    Returns:
        orbit (galpy Orbit object) - Orbit object representing a closed 
            orbit in the given potential for the given angular momentum
    '''
    
    # Turn off physical
    potential.turn_physical_off(pot)
    
    # Initialize starting orbit
    o = orbit.Orbit([R0,vR0,Lz/R0,0.0,0.0,0.0])
    
    # Evaluate the while loop
    loop_counter = 0
    delta_R = rtol*2.
    while delta_R > rtol:
        
        # Evaluate the crossing time so the integration can be performed 
        # for ~ long enough. Integrate for 100 crossing times.
        tdyn = 2*np.pi*(R0/np.abs(potential.evaluateRforces(pot,R0,0.0,phi=0.0)))**0.5
        times = np.linspace(0,100*tdyn,num=10001)
        o.integrate(times,pot)    
        
        # Evaluate all points where the orbit crosses from negative to positive 
        # phi
        phis = o.phi(times) - np.pi
        shift_phis = np.roll(phis,-1)
        where_cross = (phis[:-1] < 0.)*(shift_phis[:-1] > 0.)
        R_cross = o.R(times)[:-1][where_cross]
        vR_cross = o.vR(times)[:-1][where_cross]
        
        # Plot the surface of section as a test, if asked
        if plot_loop:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(R_cross,vR_cross,s=20,color='Black')
            ax.set_xlabel(r'$R$')
            ax.set_ylabel(r'$v_{R}$')
            ax.set_title(   r'R='+str(round(o.R(0),6))+\
                            ', vR='+str(round(o.vR(0),6))+\
                            ', vT='+str(round(o.vT(0),6))
                        )
            fig.savefig('./loop_fig'+str(loop_counter)+'.pdf')
            plt.close('all')
        ##fi
        
        # Calculate the difference in radius
        delta_R = np.abs( o.R(0) - np.average( R_cross ) )
        
        # Update the orbit
        o = orbit.Orbit( [  np.average( R_cross ),
                            np.average( vR_cross ),
                            Lz/np.average( R_cross ),
                            0.0,0.0,0.0] )
    
        # Count
        loop_counter += 1
        
    return o
#def



