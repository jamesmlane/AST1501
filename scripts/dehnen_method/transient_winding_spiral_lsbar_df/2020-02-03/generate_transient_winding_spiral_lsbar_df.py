# ----------------------------------------------------------------------------
#
# TITLE - generate_transient_spiral_df.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Script to run parallelized MWPotential2014 DF evaluation with transient
winding spiral arms and a long slow bar.
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, time, copy

## galpy and Astropy
from astropy import units as apu
from galpy import orbit, potential, df, actionAngle
from galpy.util import bovy_conversion as gpconv

## Project specific
sys.path.append('../../../../src')
import ast1501.df
import ast1501.potential
import ast1501.util

# ----------------------------------------------------------------------------

### Parameters

# General
_NCORES = 12                        # Number of cores to use
_VERBOSE = 0                        # Degree of verbosity
_PLOT_DF = False                    # Plot the output DF
_COORD_IN_XY = False                # Input coordinate grid in XY or polar?

# Timing
_T_EVOLVE, _T_FORM, _T_STEADY = [10,-9,8]
_TIMES = -np.array([0,_T_EVOLVE]) * apu.Gyr

# Spatial
_RRANGE = [5,15]                    # Range in galactocentric R
_PHIRANGE = [-np.pi/2,np.pi/2]      # Range in galactocentric phi
_DR = 1.0                           # Bin size in R
_DPHI = 1.0                         # Bin size in Phi (arc in kpc)
_GRIDR, _GRIDPHI = ast1501.df.generate_grid_radial( _RRANGE, 
                                                    _PHIRANGE, 
                                                    _DR, 
                                                    _DPHI, 
                                                    delta_phi_in_arc=True )

# Distribution Function
_VPARMS = [5,5,8,8]   # dvT,dvR,nsigma,nsigma
_SIGMAPARMS = ast1501.df.get_vsigma()
_SIGMA_VR,_SIGMA_VT,_SIGMA_VZ = _SIGMAPARMS
_SCALEPARMS =  ast1501.df.get_scale_lengths()
_RADIAL_SCALE, _SIGMA_VR_SCALE, _SIGMA_VZ_SCALE = _SCALEPARMS
_EVAL_THRESH = 0.0001   # DF evaluation threshold

_N_ARMS = [2,3,4]
_POT_PHI0 = np.array([75,67,64])* apu.deg  #* (np.pi/180) # 
_POT_AMP = 0.136* apu.M_sun / (apu.pc**3) # / gpconv.dens_in_msolpc3(ro=8,vo=220)
_POT_H = 1*apu.kpc #  / 8.
_POT_RS = 2.4*apu.kpc # / 8. 
_POT_ALPHA = 12*apu.deg #  * (np.pi/180)
_POT_RREF = 8*apu.kpc # Probably???  / 8.
_POT_OMEGA = 0*apu.km/apu.s/apu.kpc # / gpconv.freq_in_kmskpc(ro=8,vo=220) # 
_POT_LIFETIME = np.array([0.120,0.120,0.240,0.240])*apu.Gyr #  / gpconv.time_in_Gyr(ro=8,vo=220)
_POT_LIFETIME_PRINT = np.array([0.120,0.120,0.240,0.240])
_POT_SIGMA = _POT_LIFETIME / 5.6
_POT_T0 = np.array([-0.86,-0.29,-0.89,-0.29])*apu.Gyr #  / gpconv.time_in_Gyr(ro=8,vo=220) # 
_POT_T0_PRINT = np.array([-0.86,-0.29,-0.89,-0.29])
_POT_BETA = -0.1

# ----------------------------------------------------------------------------

_AA = actionAngle.actionAngleAdiabatic( pot=potential.MWPotential2014, 
                                        c=True)
_QDF = df.quasiisothermaldf(hr= _RADIAL_SCALE*apu.kpc, 
                            sr= _SIGMA_VR*(apu.km/apu.s),
                            sz= _SIGMA_VZ*(apu.km/apu.s),
                            hsr= _SIGMA_VR_SCALE*(apu.kpc), 
                            hsz= _SIGMA_VZ_SCALE*(apu.kpc),
                            pot= potential.MWPotential2014, 
                            aA= _AA)

_LSBAR_OMEGAB = 1.35
_LSBAR_RB = 5/8
_LSBAR_AF = 0.02
_LSBAR_POT = potential.DehnenBarPotential(omegab=_LSBAR_OMEGAB, rb=_LSBAR_RB, 
    Af=_LSBAR_AF, tsteady=1/gpconv.time_in_Gyr(vo=220,ro=8), 
    tform=-2/gpconv.time_in_Gyr(vo=220,ro=8))

# ----------------------------------------------------------------------------

### Evaluate the DF

# Counter for evaluations
evaluation_counter = 0

for i in range( len(_N_ARMS) ):
    for j in range( len(_POT_LIFETIME) ):

        if i > 0: continue

        # Write the parameters in the log
        output_str = 'N_ARMS_'+str(_N_ARMS[i])+'_LIFETIME_'+str(_POT_LIFETIME_PRINT[j])+'_T0_'+str(_POT_T0_PRINT[j])
        _LOGFILE = open('./log'+output_str+'.txt','w') 
        
        ### Make potentials and DFs
        _SPIRAL_ARM_POT = potential.SpiralArmsPotential(N=_N_ARMS[i], 
            amp=_POT_AMP, phi_ref=_POT_PHI0[i], alpha=_POT_ALPHA, r_ref=_POT_RREF, 
            Rs=_POT_RS, H=_POT_H, omega=_POT_OMEGA, Cs=[1,])
        _SPIRAL_ARM_POT_COROT = potential.CorotatingRotationWrapperPotential(
            pot=_SPIRAL_ARM_POT, vpo=220*apu.km/apu.s, beta=_POT_BETA)
        _SPIRAL_ARM_POT_TDEP = potential.GaussianAmplitudeWrapperPotential(
            pot=_SPIRAL_ARM_POT_COROT, to=_POT_T0[j], sigma=_POT_SIGMA[j])
        _POT = [potential.MWPotential2014,_SPIRAL_ARM_POT_TDEP,_LSBAR_POT]

        _LOGFILE.write(str(len(_GRIDR))+' evaluations\n')
        write_params = [_NCORES,_TIMES,_N_ARMS[i],_POT_PHI0[i],_POT_LIFETIME,
                _POT_SIGMA[j],_POT_T0[j],_POT_AMP,_POT_H,_POT_RS,_POT_ALPHA,
                _POT_RREF,_POT_OMEGA,_POT_BETA,_RRANGE,_PHIRANGE,_DR,_DPHI,
                _VPARMS,_SIGMAPARMS,_SCALEPARMS,_EVAL_THRESH,]
        write_param_names = ['NCORES','TIMES','N_ARMS','PHI0','LIFETIME',
                    'SIGMA','T0','AMP','HSCALE','RSCALE','ALPHA','RREF','OMEGA',
                    'BETA','RRANGE','PHIRANGE','DR','DPHI','VPARMS',
                    'SIGMAPARMS','SCALEPARMS','EVAL_THRESH']
        _LOGFILE = ast1501.util.df_evaluator_write_params(_LOGFILE,write_params,
                                                            write_param_names)

        # Run the program
        t1 = time.time()
        #pdb.set_trace()
        results = ast1501.df.evaluate_df_polar_parallel(_GRIDR, 
                                                        _GRIDPHI, 
                                                        _POT, 
                                                        _QDF, 
                                                        _VPARMS, 
                                                        _TIMES, 
                                                        _NCORES,
                                                        sigma_vR=_SIGMA_VR,
                                                        sigma_vT=_SIGMA_VT,
                                                        evaluator_threshold=_EVAL_THRESH,
                                                        plot_df=_PLOT_DF,
                                                        coords_in_xy=_COORD_IN_XY,
                                                        logfile=_LOGFILE,
                                                        verbose=_VERBOSE)
        t2 = time.time()    
                                
        # Write in the log
        _LOGFILE.write('\n'+str(round(t2-t1))+' s total')
        _LOGFILE.close()

        # Write results to file
        np.save('data'+output_str+'.npy',np.array(results))

        # Count up
        evaluation_counter += 1
    ###j
###i

# ----------------------------------------------------------------------------
