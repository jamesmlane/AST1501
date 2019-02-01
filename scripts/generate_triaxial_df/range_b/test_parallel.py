# ----------------------------------------------------------------------------
#
# TITLE - test_parallel.py
# AUTHOR - James Lane
# PROJECT - AST1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Script to run parallelized triaxial DF evaluation. This one will be over a 
range of possible b values
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, time, copy

## Matplotlib, Astropy, & galpy
from matplotlib import pyplot as plt
from astropy import units as apu
from galpy import orbit, potential, df, actionAngle

## Project specific
sys.path.append('../../src')
import ast1501.df
import ast1501.potential

'''This simple script does a mock evaluation of the parallel df evaluator
'''

### Define keywords

# General
_NCORES = 8
_LOGFILE = open('./log.txt','w')
_VERBOSE = 0
_PLOT_DF = False
_COORD_IN_XY = False

# Spatial
_RRANGE = [5,15]
_PHIRANGE = [-np.pi,np.pi]
_DR = 2 # kpc
_DPHI = 2 # arc in kpc

# Halo evolution
_HALO_B_RANGE = [0.9,0.95,0.975,1.025,1.5,1.1]
_HALO_A, _HALO_C, _HALO_PHI = [1.0,1.0,0.0]
_T_EVOLVE, _T_FORM, _T_STEADY = [10,-9,8]

# DF
_VPARMS = [20,20,8,8] # dvT,dvR,nsigma,nsigma
_SIGMA_VR, _SIGMA_VT, _SIGMA_VZ = [30,30,20] # km/s
_EVAL_THRESH = 0.0001

### Make the input
_GRIDR, _GRIDPHI = ast1501.df.generate_grid_radial( _RRANGE, _PHIRANGE, _DR, _DPHI, delta_phi_in_arc=True )
_TIMES = -np.array([0,_T_EVOLVE]) * apu.Gyr
_POT = ast1501.potential.make_triaxialNFW_dsw(halo_b=_HALO_B, halo_phi=_HALO_PHI, 
    halo_c=_HALO_C, t_form=_T_FORM, t_steady=_T_STEADY)
_AA = actionAngle.actionAngleAdiabatic(pot=potential.MWPotential2014, c=True)
_QDF = df.quasiisothermaldf( hr= 2*apu.kpc, sr= _SIGMA_VR*(apu.km/apu.s),
                            sz= _SIGMA_VZ*(apu.km/apu.s),
                            hsr= 9.8*(apu.kpc), hsz= 7.6*(apu.kpc),
                            pot= potential.MWPotential2014, aA= _AA)

print('starting')

for i in range( len( _HALO_B_RANGE ) ):

    ### Keywords
    _LOGFILE = open('./log'+str(i)+'.txt','w')
    _HALO_B = _HALO_B_RANGE[i]
    _POT = ast1501.potential.make_triaxialNFW_dsw(halo_b=_HALO_B, 
        halo_phi=_HALO_PHI, halo_c=_HALO_C, t_form=_T_FORM, t_steady=_T_STEADY)

    # Run
    _LOGFILE.write(str(len(_GRIDR))+' evaluations')
    t1 = time.time()
    results = ast1501.df.evaluate_df_polar_parallel(_GRIDR, 
        _GRIDPHI, _POT, _QDF, _VPARMS, _TIMES, _NCORES, plot_df=_PLOT_DF)
    t2 = time.time()
    _LOGFILE.write('\n'+str(round(t2-t1))+' s total')
    _LOGFILE.close()
        
    # Write
    np.save('data'+str(i)+'.npy',np.array(results))

    print('Done b='+str(_HALO_B))

#def
