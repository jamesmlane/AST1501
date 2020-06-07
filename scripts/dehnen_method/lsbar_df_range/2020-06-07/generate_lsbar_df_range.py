# ----------------------------------------------------------------------------
#
# TITLE - generate_triaxial_df.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Script to run a parallelized evaluation of the DF of a long-slow bar 
with a range of pattern speeds, but the same radial force fraction
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
_NCORES = 10                        # Number of cores to use
_VERBOSE = 0                        # Degree of verbosity
_PLOT_DF = False                    # Plot the output DF
_COORD_IN_XY = False                # Input coordinate grid in XY or polar?

# Timing
_T_EVOLVE = 3
_TIMES = -np.array([0,_T_EVOLVE]) * apu.Gyr

# Spatial
_RRANGE = [5,15]                    # Range in galactocentric R
_PHIRANGE = [-np.pi/2,np.pi/2]      # Range in galactocentric phi
_DR = 1                             # Bin size in R
_DPHI = 1                           # Bin size in Phi (arc in kpc)
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
# ----------------------------------------------------------------------------

### Make the DF

_AA = actionAngle.actionAngleAdiabatic( pot=potential.MWPotential2014, 
                                        c=True)
_QDF = df.quasiisothermaldf(hr= _RADIAL_SCALE*apu.kpc, 
                            sr= _SIGMA_VR*(apu.km/apu.s),
                            sz= _SIGMA_VZ*(apu.km/apu.s),
                            hsr= _SIGMA_VR_SCALE*(apu.kpc), 
                            hsz= _SIGMA_VZ_SCALE*(apu.kpc),
                            pot= potential.MWPotential2014, 
                            aA= _AA)

# ----------------------------------------------------------------------------

### Potential parameters

_LSBAR_RB = 5/8 
_LSBAR_OMEGAB_RANGE = np.array([34,36,38,40,42,44,46])/(220/8)
_LSBAR_OMEGAB_RANGE_STR = _LSBAR_OMEGAB_RANGE.astype(str)
_LSBAR_AF_RANGE = np.array([0.010])
_LSBAR_AF_RANGE_STR = _LSBAR_AF_RANGE.astype(str)

# ----------------------------------------------------------------------------

### Evaluate the DF

# Counter for evaluations
evaluation_counter = 0


for i in range( len(_LSBAR_OMEGAB_RANGE) ):
    for j in range( len(_LSBAR_AF_RANGE) ):
    
        # Make the log file
        output_str = '_OMEGAB_'+_LSBAR_OMEGAB_RANGE_STR[i]+'_AF_'+_LSBAR_AF_RANGE_STR[j]
        _LOGFILE = open('./log'+output_str+'.txt','w')

        # Make the potential
        _LSBAR_POT = potential.DehnenBarPotential(omegab=_LSBAR_OMEGAB[i], 
            rb=_LSBAR_RB, Af=_LSBAR_AF_RANGE[j], 
            tsteady=1/gpconv.time_in_Gyr(ro=8,vo=220), 
            tform=-2/gpconv.time_in_Gyr(ro=8,vo=220))
        _POT = [potential.MWPotential2014,_LSBAR_POT]
        
        # Write the parameters in the log
        _LOGFILE.write(str(len(_GRIDR))+' evaluations\n')
        write_dict = {'NCORES':_NCORES,
                      'TIMES':_TIMES,
                      'RRANGE':_RRANGE,
                      'PHIRANGE':_PHIRANGE,
                      'DR':_DR,
                      'DPHI':_DPHI,
                      'VPARMS':_VPARMS,
                      'SIGMAPARMS':_SIGMAPARMS,
                      'SCALEPARMS':_SCALEPARMS,
                      'EVAL_THRESH':_EVAL_THRESH,
                      'BAR_OMEGAB':_LSBAR_OMEGAB_RANGE_STR[i],
                      'BAR_AF':_LSBAR_AF_RANGE_STR[j]
                      }
        _LOGFILE = ast1501.util.df_evaluator_write_params(_LOGFILE,
            param_dict=write_dict)
        
        # Run the program
        t1 = time.time()
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
        _LOGFILE.write('\nTook '+str(round(t2-t1))+' s total')
        _LOGFILE.close()
        
        # Write results to file
        np.save('data'+output_str+'.npy',np.array(results))
        
        # Count up
        evaluation_counter += 1
    ###j
###i

# ----------------------------------------------------------------------------