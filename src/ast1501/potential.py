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

## Scipy
import scipy.interpolate
import scipy.signal

## galpy
from galpy import orbit
from galpy import potential
from galpy.util import bovy_conversion as gpconv

## Project
from . import coordinates as ast1501_coordinates

# ----------------------------------------------------------------------------

# Define local paths. Always use os.path.join
package_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.realpath(os.path.split(os.path.split(package_directory)[0])[0])

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
                 psi_0=None, v_c=None, is2Dinfer=False, 
                 home_dir=None):
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
                or 1D (just b) [False]
            home_dir (str) - Path to the base of the AST1500 project
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
        
        # coefficients to third order polynomial fit to residuals for 1D 
        # (b only) parameter inference case
        coeffs_vR_1D, coeffs_vT_1D = self._residuals_1D_coeffs()
        self.coeffs_vR_1D = coeffs_vR_1D
        self.coeffs_vT_1D = coeffs_vT_1D
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
    
    def _third_order_poly(self,x,A,B,C,D):
        return A*np.power(x,3)+B*np.power(x,2)+C*x+D
    #def
    
    def _get_v_c(self,R=None):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb
            # A1, A2, k1, k2, d
            params = [ 4.78216823e+04, -8.07681598e+01, 4.05882213e-04, 
                       -2.37986043e-01, -4.75617654e+04]
            if R is None: raise Exception('R must be provided')
            v_c = self._double_power_law(R,self.b_a,*params)
        else:
            # Set parameters from kuijken_fit.ipynb
            # A, B, C, D
            params = [7.05801933,-33.01486635,62.73846391,182.5000245]
            v_c = self._third_order_poly(self.b_a,*params)
        return v_c
    #def
    
    def _get_alpha(self,R=None):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb
            # A1, A2, k1, k2, d
            params = [ -7.41244033e+01, 2.64673977e+02, 4.27004574e-04, 
                       2.59477216e-04, -1.90666816e+02]
            if R is None: raise Exception('R must be provided')
            alpha = self._double_power_law(R,self.b_a,*params)
        else:
            # Set parameters from kuijken_fit.ipynb
            # A, B, C, D
            params = [0.03308258,-0.15579813,0.30111694,-0.28035999]
            alpha = self._third_order_poly(self.b_a,*params)
        return alpha
    #def
    
    def _get_psi0(self,R=None):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb
            # A1, A2, k1, k2, d
            params = [ 7.45313065e-01, 3.10535207e+03, -4.64140481e+01,  
                       1.23030373e+00, -3.10639618e+03]
            if R is None: raise Exception('R must be provided')
            psi0 = self._double_power_law(R,self.b_a,*params)
        else:
            # Set parameters from kuijken_fit.ipynb
            # A, B, C, D
            params = [-1174.4162675,3971.9089045,-684.64660765,-2113.71480391]
            psi0 = self._third_order_poly(self.b_a,*params)
        return psi0
    #def
    
    def _get_p(self,R=None):
        if self.is2Dinfer:
            # Set parameters from kuijken_fit.ipynb.
            # A1, A2, k1, k2, d
            params = [ -0.54815577, -2.74992312, 0.34326494, -0.0691154, 
                       4.31419703]
            if R is None: raise Exception('R must be provided')
            p= self._double_power_law(R,self.b_a,*params)
        else:
            params = [0.06929522,-0.31220909,0.59907812,0.02555314]
            p = self._third_order_poly(self.b_a,*params)
        return p
    #def
    
    def _residuals_1D_coeffs(self):
        '''_residuals_1D:
        
        Determine 3rd order residual coefficients for this b value
        
        Completely hard-coded assuming a 3rd order polynomial fit to the 
        residuals of the 3rd order polynomial fit to the 1D (b only) inferred 
        parameters of the Kuijken model.
        '''
        
        # Load residual fit coefficients
        # coeffs_path = os.path.
        residuals_directory = 'data/generated/DF-kuijken_linear_fit_third_order_residuals.npy'
        read_dir = os.path.join(project_directory,residuals_directory)
        coeffs = np.load(read_dir)
        b_values = coeffs[0]
        vR_coeffs = coeffs[[1,2,3,4]]
        vT_coeffs = coeffs[[5,6,7,8]]
        
        # Interpolate based on the value of b
        coeffs_out_vR = np.zeros(4)
        coeffs_out_vT = np.zeros(4)
        
        # Loop over the 4 coefficients
        for i in range(4):
            
            # Interpolation
            fn_vR = scipy.interpolate.interp1d(b_values, vR_coeffs[i,:], 
                kind='cubic')
            coeffs_out_vR[i] = fn_vR(self.b_a)
            fn_vT = scipy.interpolate.interp1d(b_values, vT_coeffs[i,:], 
                kind='cubic')
            coeffs_out_vT[i] = fn_vT(self.b_a)
            
        return coeffs_out_vR, coeffs_out_vT
    #def
    
    def _calculate_1D_residuals(self, R, which_velocity):
        '''_calculate_1D_residuals:
        
        Calculate residuals for the 1D parameter inference based on the 
        3rd order polynomial fits to the Kuijken parameters, and a 3rd order 
        polynomial fit to the residuals.
        
        Args:
            which_velocity (string) - 'vR' or 'vT'
        '''
        
        vT_residual_amp = self.coeffs_vT_1D[0]*np.power(R,3) + \
                          self.coeffs_vT_1D[1]*np.power(R,2) + \
                          self.coeffs_vT_1D[2]*R + \
                          self.coeffs_vT_1D[3]
        vR_residual_amp = self.coeffs_vR_1D[0]*np.power(R,3) + \
                          self.coeffs_vR_1D[1]*np.power(R,2) + \
                          self.coeffs_vR_1D[2]*R + \
                          self.coeffs_vR_1D[3]
        
        if which_velocity=='vR':
            return vR_residual_amp
        if which_velocity=='vT':
            return vT_residual_amp
        ##fi
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
    
    def kuijken_vr(self, R, phi, apply_correction=True):
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
        vR_amp = -( (1+0.5*p) / (1-alpha) ) * e_psi * v_circ
        if apply_correction:
            vR_amp -= self._calculate_1D_residuals(R,which_velocity='vR')
        return vR_amp * np.sin( 2*( phi-self.phib ) )
    #def
    
    def kuijken_vt(self, R, phi, apply_correction=True):
        '''kuijken_vt:
        
        Args:
            R (float) - Galactocentric cylindrical radius
            phi (float) - Galactocentric cylindrical radius
            apply_correction = 
        
        Returns:
            vt (float) - Tangential velocity fluctuation
        '''
        e_psi = self.epsilon_psi(R)
        p = self._get_p(R)
        alpha = self._get_alpha(R)
        v_circ = self.v_circ(R)
        vT_amp = -( ( 1 + 0.25*p*( 1 + alpha ) ) / ( 1 - alpha ) ) * e_psi * v_circ
        if apply_correction:
            vT_amp += self._calculate_1D_residuals(R,which_velocity='vT')
        return vT_amp * np.cos(2*(phi-self.phib))
    #def
#cls

def make_cos2_power_law(phi0, p, alpha, vc, phib=0*apu.radian, m=2):
    '''make_cos2_power_law:
    
    Make the cosmphi + power law potential that will mimic the explicit form 
    of the Kuijken+Tremaine potential
    
    Args:
        
    
    Returns:
        
    '''
    
    # First equate the properties of the two potentials
    cos2phi_amp = 1.0
    cos2phi_R1 = 8.0*apu.kpc
    cos2phi_Rb = 1.0*apu.kpc
    power_law_alpha = -2*alpha
    power_law_r1 = 8.0*apu.kpc
    power_law_amp = (power_law_r1.value**3) * (vc**2) / (2*alpha)
    
    cos2phi_pot = potential.CosmphiDiskPotential( amp=cos2phi_amp, phio=phi0, 
        phib=phib, m=m, p=p, r1=cos2phi_R1, rb=cos2phi_Rb )
    power_law_pot = potential.PowerSphericalPotential( amp=power_law_amp, 
        alpha=power_law_alpha, r1=power_law_r1 )
    
    return [cos2phi_pot, power_law_pot]
#def

def make_LongSlowBar():
    
    # From Hunt & Bovy 2018 pg. 4 we know:
    # Radius of the bar is 5 kpc
    # Pattern speed of the bar is 1.35 times the local pattern speed
    
    OB=1.35
    RB = 5/8
    Af = 0.02
    
    return potential.DehnenBarPotential(omegab=OB, rb=RB, Af=Af, 
        tsteady=1/gpconv.time_in_Gyr(ro=8,vo=220), 
        tform=2/gpconv.time_in_Gyr(ro=8,vo=220))
    
#def

def make_Sgr_mop(time, ics='gaia_dr2', sgr_halo_m=(14*(10**10))*apu.Msun, 
                 sgr_halo_a=13*apu.kpc, sgr_stlr_m=(6.4*(10**8))*apu.M_sun, 
                 sgr_stlr_a=0.85*apu.kpc, laporte_2018_class=None, 
                 correct_for_mass_loss=False, get_apocenter=None,
                 return_orbit=True, return_time=True):
    '''make_Sgr_mop:
    
    Make a moving object potential corresponding to a Sgr orbit. The default 
    masses and scale parameters are from the H1 model of Laporte 2018
    
    Laporte models are:
    H1 : M=14*10^10 Msol, a=13 kpc
    H2 : M=14*10^10 Msol, a=7 kpc
    L1 : M=8*10^10 Msol, a=13 kpc
    L2 : M=8*10^10 Msol, a=7 kpc
    
    All have stellar component M=6.4*10^8 Msol, a=0.85 kpc
    
    Args:
        time (float) - Lookback time at which to begin the orbit (in Gyr, no apu)
        ics (str) - Initial conditions to use. Only option is gaia_dr2 => the 
            Gaia DR2 astrometry-based kinematics
        sgr_halo_m (float) - Sagittarius halo profile mass
        sgr_halo_a (float) - Sagittarius halo profile scale length
        sgr_stlr_m (float) - Sagittarius stellar profile mass
        sgr_stlr_a (float) - Sagittarius stellar profile scale length
        laporte_2018_class (str) - Model H1, H2, L1, or L2 from Laporte+ 2018
        correct_for_mass_loss (bool) - Apply the correction inferred from 
            Niederste-Ostholt+ 2012 (Figure 11) for the halo mass and stellar 
            mass to mimic present-day Sgr properties
        get_apocenter (int) - Return the orbit with times only up until a 
            specific apocenter in the past: 1, 2, 3 or so depending on masses 
            of Sgr and MW (reached within t_universe). Will only change 
            times output with return_time, not outputed orbit
        return_orbit (bool) - Return the orbit along with the MOP
        return_time (bools) - Return the times along with the orbit

    Returns:
        sgr_mop (Potential) - Moving object potential
        o_sgr (Orbit) - Sgr orbit corresponding to the MOP
        times (array) - Times corresponding to the orbit and MOP (except when 
            using get_apocenter keyword)
    '''
    
    # If a specific apocenter is requested look for it within the age of the 
    # universe
    _GETTING_SGR_APOCENTER=False
    if isinstance(get_apocenter,int) and get_apocenter > 0:
        _GETTING_SGR_APOCENTER = True
    ##fi
    if _GETTING_SGR_APOCENTER:
        time = 13.7
    ##fi
    
    # Select from Laporte 2018 models if selected
    if laporte_2018_class in ['H1','H2','L1','L2']:
        if laporte_2018_class == 'H1':
            sgr_halo_m = 14*(10**10)*apu.M_sun
            sgr_halo_a = 13*apu.kpc
        elif laporte_2018_class == 'H2':
            sgr_halo_m = 14*(10**10)*apu.M_sun
            sgr_halo_a = 7*apu.kpc
        elif laporte_2018_class == 'L1':
            sgr_halo_m = 8*(10**10)*apu.M_sun
            sgr_halo_a = 16*apu.kpc
        elif laporte_2018_class == 'L2':
            sgr_halo_m = 8*(10**10)*apu.M_sun
            sgr_halo_a = 8*apu.kpc
        ##ie
        sgr_stlr_m=(6.4*(10**8))*apu.M_sun
        sgr_stlr_a=0.85*apu.kpc
    ##fi
    
    if correct_for_mass_loss:
        sgr_halo_m *= 0.1
        sgr_stlr_m *= 0.3
    ##fi
    
    helio_vx = -11.1
    helio_vy = 12.24 + 220
    helio_vz = 7.25
    helio_R = 8.2
    helio_z = 0.023
    
    if ics=='gaia_dr2':
        sgr_x, sgr_y, sgr_z = [-25.2,2.5,-6.4]
        sgr_u, sgr_v, sgr_w = [-221.3,-266.5,197.4]

        sgr_x += helio_R
        sgr_z += helio_z
        sgr_vx = sgr_u + helio_vx
        sgr_vy = sgr_v + helio_vy
        sgr_vz = sgr_w + helio_vz

        sgr_R = np.sqrt( np.square(sgr_x) + np.square(sgr_y) )
        sgr_phi = ast1501_coordinates.calculate_galactic_azimuth(sgr_x,sgr_y,cw=True,lh=True)
        sgr_vR = sgr_vx * np.cos(sgr_phi) + sgr_vy * np.sin(sgr_phi)
        sgr_vT = -sgr_vx * np.sin(sgr_phi) + sgr_vy * np.cos(sgr_phi)
        
        # Make the orbit
        sgr_vxvv = [sgr_R*apu.kpc, sgr_vR*apu.km/apu.s, sgr_vT*apu.km/apu.s, 
                    sgr_z*apu.kpc, sgr_vz*apu.km/apu.s, sgr_phi*apu.rad]
        o_sgr = orbit.Orbit(sgr_vxvv)
        o_sgr.turn_physical_on()
    ##fi
    
    print('Setting up potential for Sgr MOP integration...')
    pot = potential.MWPotential2014
    n_snaps = 10000
    times = np.linspace(0, -time, n_snaps) * apu.Gyr
    
    sgr_halo = potential.HernquistPotential(amp=sgr_halo_m, a=sgr_halo_a)
    sgr_stlr = potential.HernquistPotential(amp=sgr_stlr_m, a=sgr_stlr_a) 
    sgr_pot = [sgr_halo,sgr_stlr]
    sgr_dynfric = potential.ChandrasekharDynamicalFrictionForce(
        GMs=sgr_halo_m, rhm=(1+np.sqrt(2))*sgr_halo_a, dens=pot)
    pot_df = [pot,sgr_dynfric]
    
    print('Integrating Sgr MOP orbit...')
    o_sgr.integrate(times, pot_df)
    
    if _GETTING_SGR_APOCENTER:
        rs = o_sgr.r(times).value
        apo_arg = scipy.signal.argrelextrema(rs, np.greater)[get_apocenter-1]
        times = times[apo_arg]
        if not return_time:
            print('Warning: Specific apocenter requested, but is only found in times, set return_time=True')
        ##fi
    ##fi
    
    sgr_mop = potential.MovingObjectPotential(o_sgr,
        pot=sgr_pot)
    
    results = [sgr_mop]
    if return_orbit:
        results.append(o_sgr)
    if return_time:
        results.append(times)
    ##fi
    return results
#def