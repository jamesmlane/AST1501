# ----------------------------------------------------------------------------
#
# TITLE - generate_transient_spiral_df.py
# AUTHOR - James Lane
# PROJECT - AST 1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Script to run parallelized MWPotential2014 DF evaluation Sgr passage along 
with a Dehnen bar model. Only evaluate the Sgr impact as far back as 1 Gyr, 
which is about the first apocenter in the past for the dGal. Include the 
mass-loss perscription
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
_NCORES = 15                        # Number of cores to use
_VERBOSE = 0                        # Degree of verbosity
_PLOT_DF = False                    # Plot the output DF
_COORD_IN_XY = False                # Input coordinate grid in XY or polar?

# Timing
_T_EVOLVE = 1 # Gyr
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

LAPORTE_MODEL=['H1','H2','L1','L2']
SGR_HALO_M = [ (14*(10**10))*apu.Msun, (14*(10**10))*apu.Msun, 
               (8*(10**10))*apu.Msun,  (8*(10**10))*apu.Msun]
SGR_HALO_A = [ 13*apu.kpc, 7*apu.kpc, 16*apu.kpc, 8*apu.kpc]
SGR_STLR_M = (6.4*(10**8))*apu.M_sun
SGR_STLR_A = 0.85*apu.kpc
    
_SGR_MOP = ast1501.potential.make_Sgr_mop(_T_EVOLVE)
_POT = [potential.MWPotential2014,_SGR_MOP]

# ----------------------------------------------------------------------------

### Evaluate the DF

for i in range(len(LAPORTE_MODEL)):
    
    # Make the potential
    _SGR_MOP = ast1501.potential.make_Sgr_mop(_T_EVOLVE,
        laporte_2018_class=LAPORTE_MODEL[i], return_orbit=False,
        return_time=False, correct_for_mass_loss=True)
    _POT = [potential.MWPotential2014,_SGR_MOP,_LSBAR_POT]
    potential.turn_physical_off(_POT)
    
    MODEL_NAME = 'sgr_mop_'+LAPORTE_MODEL[i]
    _LOGFILE = open('./log_'+MODEL_NAME+'.txt','w')         
    _LOGFILE.write(str(len(_GRIDR))+' evaluations\n')
    write_params = [_NCORES,_TIMES,_RRANGE,_PHIRANGE,_DR,_DPHI,
            _VPARMS,_SIGMAPARMS,_SCALEPARMS,_EVAL_THRESH,LAPORTE_MODEL,
            SGR_HALO_M,SGR_HALO_A,SGR_STLR_M,SGR_STLR_A,_LSBAR_OMEGAB,
            _LSBAR_AF,_LSBAR_RB]
    write_param_names = ['NCORES','TIMES','RRANGE','PHIRANGE','DR','DPHI',
                'VPARMS','SIGMAPARMS','SCALEPARMS','EVAL_THRESH','LAPORTE_MODEL',
                'SGR_HALO_M','SGR_HALO_A','SGR_STLR_M','SGR_STLR_A',
                'LSBAR_OMEGAB','LSBAR_AF','LSBAR_RB']
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
    np.save('data_'+MODEL_NAME+'.npy',np.array(results))
###i

# ----------------------------------------------------------------------------
