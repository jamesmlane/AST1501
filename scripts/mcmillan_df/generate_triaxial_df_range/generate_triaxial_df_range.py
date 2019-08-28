# ----------------------------------------------------------------------------
#
# TITLE - generate_triaxial_df_range.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Script to run parallelized triaxial DF evaluation with McMillan potential

b/a: 0.8 -> 1.0

Run August 22, 2019

Evaluate the triaxial over a cylindrical grid.

Minor change from other runs: only run over half the galaxy
and ranging from 5 to 15 kpc.
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
sys.path.append('../../../src')
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
_BASE_POT = potential.mwpotentials.McMillan2017 

# Timing
_T_EVOLVE = 10
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

### Make potentials and DFs
_BA_RANGE = np.array([0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,
    0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99])
_BA_RANGE_STR = np.array(['080','081','082','083','084','085','086',
    '087','088','089','090','091','092','093','094','095','096','097',
    '098','099',
])
_AA = actionAngle.actionAngleAdiabatic( pot=_BASE_POT, 
                                        c=True)
_QDF = df.quasiisothermaldf(hr= _RADIAL_SCALE*apu.kpc, 
                            sr= _SIGMA_VR*(apu.km/apu.s),
                            sz= _SIGMA_VZ*(apu.km/apu.s),
                            hsr= _SIGMA_VR_SCALE*(apu.kpc), 
                            hsz= _SIGMA_VZ_SCALE*(apu.kpc),
                            pot= _BASE_POT, 
                            aA= _AA)

# ----------------------------------------------------------------------------

### Evaluate the DF

# Counter for evaluations
evaluation_counter = 0

for i in range( len(_SFBAR_OMEGAB_RANGE) ):
    
    # Make the log file
    output_str = '_BA_'+_SFBAR_OMEGAB_RANGE_STR[i]+'_AF_'+_SFBAR_AF_RANGE_STR[j]
    _LOGFILE = open('./log'+output_str+'.txt','w')

    # Make the potential
    _SFBAR_POT = potential.DehnenBarPotential(omegab=_SFBAR_OMEGAB_RANGE[i], 
        rb=_SFBAR_RB, Af=_SFBAR_AF_RANGE[j], 
        tsteady=1/gpconv.time_in_Gyr(ro=8,vo=220), 
        tform=-2/gpconv.time_in_Gyr(ro=8,vo=220))
    _POT = [potential.MWPotential2014,_SFBAR_POT]

    # Write the parameters in the log
    _LOGFILE.write(str(len(_GRIDR))+' evaluations\n')
    write_params = [_NCORES,_TIMES,_SFBAR_OMEGAB_RANGE[i],_SFBAR_AF_RANGE[j],
                    _RRANGE,_PHIRANGE,_DR,_DPHI,_VPARMS,_SIGMAPARMS,
                    _SCALEPARMS,_EVAL_THRESH,]
    write_param_names = ['NCORES','TIMES','BAR_OMEGAB','BAR_AF','RRANGE',
                         'PHIRANGE','DR','DPHI','VPARMS','SIGMAPARMS',
                         'SCALEPARMS','EVAL_THRESH']
    _LOGFILE = ast1501.util.df_evaluator_write_params(_LOGFILE,write_params,
                                                        write_param_names)

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
    _LOGFILE.write('\n'+str(round(t2-t1))+' s total')
    _LOGFILE.close()

    # Write results to file
    np.save('data'+output_str+'.npy',np.array(results))
    
    # Count up
    evaluation_counter += 1
###i

# ----------------------------------------------------------------------------
