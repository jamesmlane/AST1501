# ----------------------------------------------------------------------------
#
# TITLE - generate_transient_spiral_df.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Script to run parallelized MWPotential2014 DF evaluation Sgr passage. Do 
this for only a few locations so that we can plot the distribution function
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

### Basic Parameters

# General
_NCORES = 12                        # Number of cores to use
_VERBOSE = 0                        # Degree of verbosity
_PLOT_DF = True                     # Plot the output DF
_COORD_IN_XY = False                # Input coordinate grid in XY or polar?

# Timing
_T_EVOLVE, _T_FORM, _T_STEADY = [10,-9,8]
_TIMES = -np.array([0,_T_EVOLVE]) * apu.Gyr

# Spatial
# _RRANGE = [5,15]                    # Range in galactocentric R
# _PHIRANGE = [-np.pi/2,np.pi/2]      # Range in galactocentric phi
# _DR = 1.0                           # Bin size in R
# _DPHI = 1.0                         # Bin size in Phi (arc in kpc)
# _GRIDR, _GRIDPHI = ast1501.df.generate_grid_radial( _RRANGE, 
#                                                     _PHIRANGE, 
#                                                     _DR, 
#                                                     _DPHI, 
#                                                     delta_phi_in_arc=True )

# Manully set the grid for a small number of locations
_GRIDR = [5.5,5.5,6.5,6.5,7.5,7.5,8.5,8.5,9.5,9.5,10.5,10.5,11.5,11.5,
            12.5,12.5,13.5,13.5,14.5,14.5]
_GRIDPHI = [-0.9,0.9,-0.9,0.9,-0.9,0.9,-0.9,0.9,-0.9,0.9,-0.9,0.9,-0.9,0.9,
            -0.9,0.9,-0.9,0.9,-0.9,0.9,]

# Distribution Function
_VPARMS = [5,5,8,8]   # dvT,dvR,nsigma,nsigma
_SIGMAPARMS = ast1501.df.get_vsigma()
_SIGMA_VR,_SIGMA_VT,_SIGMA_VZ = _SIGMAPARMS
_SCALEPARMS =  ast1501.df.get_scale_lengths()
_RADIAL_SCALE, _SIGMA_VR_SCALE, _SIGMA_VZ_SCALE = _SCALEPARMS
_EVAL_THRESH = 0.0001   # DF evaluation threshold

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

### Potentials

_SGR_MOP = ast1501.potential.make_Sgr_mop(_T_EVOLVE)
_POT = [potential.MWPotential2014,_SGR_MOP]

# ----------------------------------------------------------------------------

### Evaluate the DF

# Write the parameters in the log
_LOGFILE = open('./log_sgr_mop.txt','w') 
        
_LOGFILE.write(str(len(_GRIDR))+' evaluations\n')
write_params = [_NCORES,_TIMES,'Manual','Manual','Manual','Manual',
        _VPARMS,_SIGMAPARMS,_SCALEPARMS,_EVAL_THRESH,]
write_param_names = ['NCORES','TIMES','RRANGE','PHIRANGE','DR','DPHI',
            'VPARMS','SIGMAPARMS','SCALEPARMS','EVAL_THRESH']
_LOGFILE = ast1501.util.df_evaluator_write_params(_LOGFILE,write_params,
                                                    write_param_names)
_LOGFILE.write('\nSgr params: M=1.4E11, a=13kpc, Mstlr=6.4E8\n')

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
np.save('data_sgr_mop.npy',np.array(results))

# ----------------------------------------------------------------------------
