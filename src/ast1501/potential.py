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

##
from matplotlib import pyplot as plt

## Astropy
from astropy import units as apu

## galpy
from galpy import orbit
from galpy import potential
from galpy.util import bovy_conversion as gpconv

# ----------------------------------------------------------------------------

def _get_MWPotential2014_params():
    '''_get_MWPotential2014_params:
    
    Return the parameters of galpy's MWPotential2014 object as variables
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

def make_triaxialNFW(halo_b=1.0, halo_phi=0.0, halo_c=1.0, halo_amp=None, 
                        halo_a=None):
    '''make_triaxialNFW:
    
    Generate a triaxial NFW. Normally use the same properties as MWPotential2014
    
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

    _, _, _, _, _, _, mwhalo_a, mwhalo_amp = _get_MWPotential2014_params()
    
    # Check argument choices
    if halo_amp == None:
        use_halo_amp = mwhalo_amp
    else: use_halo_amp = halo_amp
    ##ie
    if halo_a == None:
        use_halo_a = mwhalo_a
    else: use_halo_a = halo_a
    ##ie
    
    return potential.TriaxialNFWPotential(amp=use_halo_amp, a=use_halo_a, 
                                    b=halo_b, pa=halo_phi, c=halo_c)
#def

def make_MWPotential2014_triaxialNFW(halo_b=1.0, halo_phi=0.0, halo_c=1.0, 
                                        halo_amp=None, halo_a=None):
    '''make_MWPotential2014_triaxialNFW:
    
    Return MWPotential2014 with a triaxial halo rather than a spherical NFW 
    halo.
    
    Args:
        halo_b (float) - Halo secondary to primary axis ratio (b/a) [1.0]
        halo_phi (float) - Halo primary axis position angle in radians [0.0]
        halo_c (float) - Halo tertiary to primary axis ratio (c/a) [1.0]
        halo_amp (float) - Halo amplitude. If None it will be identical 
            to MWPotential2014 [None]
        halo_a (float) - Halo scale length. If None it will be identical 
            to MWPotential2014 [None]
            
    Returns:
        Potential object - MWPotential2014 with halo replaced by triaxial halo  
    '''
    
    # Make the triaxial halo
    trihalo = make_triaxialNFW(halo_b=halo_b, halo_phi=halo_phi, halo_c=halo_c, 
                                halo_amp=halo_amp, halo_a=halo_a)
    
    # Get MWPotential2014 parameters
    mwbulge_alpha, mwbulge_rc, mwbulge_amp, mwdisk_a, mwdisk_b, mwdisk_amp,\
        _, _ = _get_MWPotential2014_params()
    
    # Make the bulge and disk
    mwbulge = potential.PowerSphericalPotentialwCutoff(amp=mwbulge_amp, 
                                                        alpha=mwbulge_alpha, 
                                                        rc=mwbulge_rc)
    mwdisk = potential.MiyamotoNagaiPotential(amp=mwdisk_amp, a=mwdisk_a, 
                                                b=mwdisk_b)
    
    return [mwbulge, mwdisk, trihalo]
#def

def make_halo_dsw(new_halo, t_form=-9, t_steady=-8, mwhalo=None, mwdisk=None, 
                    mwbulge=None):
    '''make_halo_dsw
    
    Slowly interpolate between a known MW halo (default is MWPotential2014) to 
    a new MW halo.
    
    Args:
        new_halo (galpy Potential object) - The new halo to introduce
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
        pot_tdep (galpy Potential object array) - time varying potential  
    '''

    # Get MWPotential2014 parameters
    mwbulge_alpha, mwbulge_rc, mwbulge_amp, mwdisk_a, mwdisk_b, mwdisk_amp,\
        mwhalo_a, mwhalo_amp = _get_MWPotential2014_params()
    
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
    
    # Wrap the new halo in a DSW:
    new_halo_grow_dsw = potential.DehnenSmoothWrapperPotential(pot=new_halo,
        tform=t_form*apu.Gyr, tsteady=t_steady*apu.Gyr)
        
    return [use_mwbulge, use_mwdisk, mwhalo_decay_dsw, new_halo_grow_dsw]
#def

def make_triaxialNFW_dsw(halo_b=1.0, halo_phi=0.0, halo_c=1.0, halo_amp=None, 
                    halo_a=None, t_form=-9, t_steady=8, mwhalo=None, 
                    mwdisk=None, mwbulge=None):
    '''make_triaxialNFW_dsw:
    
    Wrapper function that uses make_triaxialNFW and make_halo_dsw to generate a 
    time-dependent potential that varies from the MWPotential2014 halo to a 
    traxial halo.
    
    Args:
        halo_b (float) - Halo secondary to primary axis ratio (b/a) [1.0]
        halo_phi (float) - Halo primary axis position angle in radians [0.0]
        halo_c (float) - Halo tertiary to primary axis ratio (c/a) [1.0]
        halo_amp (float) - Halo amplitude. If None it will be identical 
            to MWPotential2014 [None]
        halo_a (float) - Halo scale length. If None it will be identical 
            to MWPotential2014 [None]
        t_form (float) - Time of triaxial halo formation in Gyr 
            (No astropy units attached) [-9.0].
        t_steady (float) - Time of finished triaxial halo formation in Gyr 
            (No astropy units attached) [8.0].
        mwhalo (galpy Potential object) - Halo model to slowly remove. If None 
            then use the same as MWPotential2014 [None]
        mwdisk (galpy Potential object) - Disk model to use. If None then use 
            the same as MWPotential2014 [None]
        mwbulge (galpy Potential object) - bulge model to use. If None then use 
            the same as MWPotential2014 [None]
        
    Returns:
        pot_tdep (galpy Potential object array) - time varying potential 
    '''
    
    trihalo = make_triaxialNFW(halo_b=halo_b, halo_phi=halo_phi, halo_c=halo_c, 
                                halo_amp=halo_amp, halo_a=halo_a)
                                
    pot_tdep = make_halo_dsw(trihalo, t_form=t_form, t_steady=t_steady, 
                                mwhalo=mwhalo, mwdisk=mwdisk, mwbulge=mwbulge)
                                
    return pot_tdep
#def

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
            
    To Do:
        Should do for a fixed quantity - E
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
        times = np.linspace(0,100*tdyn,num=100001)
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

class kuijken_potential():
    '''kuijken_potential:
    
    Args:
        
        
    Returns:
        
    '''
    
    def __init__(self, b_a=1.0, phib=0, R0=8.0, p=None, alpha=None, 
                 psi_0=None, v_c=None, is2Dinfer=False):
        '''__init__:
        
        Args:
            b_a (float) - Axis ratio in the plane of the disk. [1.0]
            phib (float) - secondary axis angle [0.0]
            R0 (float) - Scale radius [8.0]
            p (float) - Psi amplitude power law index
            alpha (float) - Radial power law index
            psi_0 (float) - Psi amplitude at the scale radius
            phib (float) - Bar angle
            is2Dinfer (bool) - Infer potential properties based on 2D (R and b) 
                or 1D (just b)
        '''
        
        # First calculate all the values using b/a and the profiles fitted 
        # with MWPotential2014
        
        # if p == None:
        #     if b_a >= 1.0:
        #         p = self._offset_power_law(b_a, 0.152, -0.99, 1.169, 0.524)
        #     ##fi
        # 
        #     if b_a < 1.0:
        #         p = self._offset_power_law(b_a, 33.695, -0.178, 0.0034, -33.16)
        #     ##fi
        # ##fi
        # 
        # if alpha == None:
        #     alpha = self._power_law(b_a, 0.279, 0.141, -0.558)
        # ##fi
        # 
        # if psi_0 == None:
        #     psi_0 = self._power_law(b_a, 1730.979, 2.053, -1731.046)
        # ##fi
        # 
        # if v_c == None:
        #     v_c = self._power_law(b_a, 296.477, 0.182, -126.565)
        # ##fi
        
        # Must be set
        self.b_a = b_a
        self.R0 = R0
        self.phib = phib
        self.is2Dinfer = is2Dinfer
    #def
    
    def _double_power_law(self,R,b,A1,A2,k1,k2,d):
        return A1*np.power(R,k1) + A2*np.power(b,k2) + d
    #def
    
    def _power_law(self,x,A,k,d):
        return A*np.power(x,k)+d
    #def
    
    def _offset_power_law(self,x,A,c,k,d):
        return A*np.power(x+c,k)+d
    #def
    
    def _get_v_c(self,R):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb
            # A1, A2, k1, k2, d
            params = [2.69889467e+01,2.95235000e+02,3.26001601e-01,
                      1.88714756e-01,-1.79201675e+02]
            v_c = self._double_power_law(R,self.b_a,*params)
        else:
            # Set parameters from kuijken_fit.ipynb
            # A, k, d
            params = [296.477, 0.182, -126.565]
            v_c = self._power_law(self.b_a,*params)
        return v_c
    #def
    
    def _get_alpha(self,R):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb
            # A1, A2, k1, k2, d
            params = [-1.49984367e+02,2.36875055e-01,3.69210365e-04,
                      1.60606290e-01,1.49588215e+02]
            alpha = self._double_power_law(R,self.b_a,*params)
        else:
            # Set parameters from kuijken_fit.ipynb
            # A, k, d
            params = [0.279, 0.141, -0.558]
            alpha = self._power_law(self.b_a,*params)
        return alpha
    #def
    
    def _get_psi0(self,R):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb
            # A1, A2, k1, k2, d
            params = [-1.77157188e+03,1.78311874e+03,-1.97068863e-03,
                      2.06887401e+00,-1.94272099e+01]
            psi0 = self._double_power_law(R,self.b_a,*params)
        else:
            params = [1730.979,2.053,-1741.046]
            psi0 = self._power_law(self.b_a,*params)
        return psi0
    #def
    
    def _get_p(self,R):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb.
            # A1, A2, k1, k2, d
            params = [-9.55285852e+02,2.55131016e-01,3.56811498e-04,
                      5.81101032e-01,9.55288745e+02]
            p= self._double_power_law(R,self.b_a,*params)
        else:
            # Note this fit is for 1+p, so need to subtract 1
            if self.b_a >= 1.0:
                params = [0.152, -0.99, 1.169, 0.524]
                p = self._offset_power_law(self.b_a, *params)-1
            ##fi
            if self.b_a < 1.0:
                params = [33.695, -0.178, 0.0034, -33.16]
                p = self._offset_power_law(self.b_a, *params)-1
            ##fi
        return p
    #def
    
    def psi(self,R):
        '''psi:
        
        Psi function.
        
        Args:
            R (float) - Galactocentric cylindrical radius
        
        Returns:
            Psi(R)
        '''
        psi0 = self._get_psi0(R)
        p = self._get_p(R)
        return psi0 * np.power( R / self.R0, p )
    #def
    
    def v_circ(self,R):
        '''v_circ:
        
        Calculate the circular velocity
        
        Args:
            R (float) - Galactocentric cylindrical radius
        
        Returns:
            v_circ (float) - Circular velocity
        
        '''
        
        v_c = self._get_v_c(R)
        alpha = self._get_alpha(R)
        return np.power( R / self.R0, alpha ) * v_c
    #def
    
    def epsilon_psi(self,R):
        '''epsilon_psi:
        
        Epsilon function.
        
        Args:
            R (float) - Galactocentric cylindrical radius
        
        Returns:
            epsilon_psi (float) - epsilon function 
            
        '''
        v_circ = self.v_circ(R)
        return 2 * self.psi(R) / ( v_circ**2 )
    #def
    
    def kuijken_vr(self,R,phi):
        '''kuijken_vr:
        
        Args:
            R (float) - Galactocentric cylindrical radius
            phi (float) - Galactocentric cylindrical radius 
        
        Returns:
            vr (float) - Radial velocity fluctuation 
        '''
        e_psi = self.epsilon_psi(R)
        p = self._get_p(R)
        alpha = self._get_alpha(R)
        v_circ = self.v_circ(R)
        return -( (1+0.5*p) / (1-alpha) ) * e_psi * v_circ * np.sin( 2*( phi-self.phib ) )
    #def
    
    def kuijken_vt(self,R,phi):
        '''kuijken_vt:
        
        Args:
            R (float) - Galactocentric cylindrical radius
            phi (float) - Galactocentric cylindrical radius
        
        Returns:
            vt (float) - Tangential velocity fluctuation
        '''
        e_psi = self.epsilon_psi(R)
        p = self._get_p(R)
        alpha = self._get_alpha(R)
        v_circ = self._get_v_c(R)
        return -( ( 1 + 0.25*p*( 1 + alpha ) ) / ( 1 - alpha ) ) * e_psi * v_circ * np.cos( 2*( phi-self.phib ) )
    #def
