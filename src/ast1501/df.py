# ----------------------------------------------------------------------------
#
# TITLE - df.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS: 
# 1. evaluate_df_adaptive_vrvt
# 2. hist_df
# 3. calculate_df_vmoments
# 4. generate_triaxial_df_map_polar
# 5. generate_triaxial_df_map_rect
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Functions to calculate DFs
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy
# import glob
# import subprocess

## Plotting
from matplotlib import pyplot as plt
# from matplotlib.backends.backend_pdf import Pdfages
# from matplotlib import colors
# from matplotlib import cm
# import aplpy

## Astropy
# from astropy.io import fits
# from astropy.coordinates import SkyCoord
# from astropy import table
from astropy import units as apu
# from astropy import wcs

## galpy
from galpy import orbit
from galpy import potential
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv
# from galpy.util import bovy_plot as gpplot

# ----------------------------------------------------------------------------

def evaluate_df_adaptive_vrvt(R_z_phi,
                                times,
                                pot,
                                df_evaluator,
                                vR_range,
                                vT_range,
                                df,
                                compute_unperturbed=False,
                                threshold=0.001,
                                threshold_norm=True):
    '''
    evaluate_df_adaptive:
    
    Evaluate the DF of a potential using the reverse integration technique, and 
    use a simple threshold to avoid calculating the DF in regions where it 
    doesn't need to be calculated. Works at a single physical point. Only 
    accepts grids of radial and tangential velocities.
    
    Args:
        R_z_phi (array) - Position at which to evaluate the DF, in kpc, kpc, rad
        times (array) - Times in Gyr
        pot (galpy pot object) - Potential object that will be used to for 
            integration
        df_evaluator (galpy DF object) - DF object that can accept an orbit as 
            an argument
        vR_range (array) - array of radial velocities
        vT_range (array) - array of tangential velocities
        df (array) - array to hold the DF
        threshold (float) - The threshold DF value, normalized such that the 
            maximum DF value is 1 (see threshold_real) [0.001]
        threshold_norm (bool) - Is the threshold value normalized to the 
            maximum of the DF or is it just a real DF value? 
    
    Returns:
        df (array) - array of DF values
    '''
    
    # First check that the dimensions are correct:
    if (df.shape != (len(vR_range),len(vT_range))):
        raise RuntimeError('DF array not correct size')
    
    # Now 0 all of the elements of the DF arrays
    df[:,:] = 0
    
    # Make an array of values that tell us if we've computed the DF at this 
    # location or not
    computed_df = np.zeros_like(df)
    
    # Unpack coordinates
    R,z,phi = R_z_phi
    
    # Make sure there is no physical output
    pot = copy.copy(pot)
    potential.turn_physical_off(pot)
    
    # Now find the maximum value of the perturbed DF. Assume that we'll start 
    # vR = 0 km/s and vT = 220 km/s, assuming they're in the arrays. Otherwise 
    # start in the middle of the arrays.
    vR_start = 0
    vT_start = 220
    if (vR_start < vR_range[0]) or (vR_start > vR_range[-1]):
        vR_start = vR_range[ int(len(vR_range)/2) ]
    if (vT_start < vT_range[0]) or (vT_start > vT_range[-1]):
        vT_start = vT_range[ int(len(vT_range)/2) ]
    if vR_start not in vR_range:
        vR_start = vR_range[ np.argmin( np.abs( vR_range-vR_start ) ) ]
    if vT_start not in vT_range:
        vT_start = vT_range[ np.argmin( np.abs( vT_range-vT_start ) ) ]
    ##fi
    
    # Now start the search for the maximum DF point. Pick the first point 
    # in the grid
    cur_point = [  np.where(vR_start == vR_range)[0][0], 
                    np.where(vT_start == vT_range)[0][0] ]
    # Initialize checker variable
    found_df_max = False   
    fake_maxima = np.zeros_like(df)
    
    # Loop over the maximum finder
    while found_df_max == False:
        
        # If we have butt up against the edge of the input array consider this 
        # a failed evaluation attempt
        if cur_point[0] == 0 or cur_point[0] == len(vR_range)-1 or cur_point[1] == 1 or cur_point[1] == len(vT_range)-1:
            raise RuntimeError('Attempting to evaluate perturbed DF outside of supplied grid')
        ##fi
        
        if computed_df[ cur_point[0] , cur_point[1] ] == 0:
            # Calculate the orbit
            cur_o = orbit.Orbit(vxvv=[  R*apu.kpc, 
                                        vR_range[cur_point[0]]*apu.km/apu.s, 
                                        vT_range[cur_point[1]]*apu.km/apu.s, 
                                        z*apu.kpc,
                                        0*apu.km/apu.s,
                                        phi*apu.radian
                                    ])
        
            # Should the unperturbed DF be computed? 
            if compute_unperturbed:
                df[ cur_point[0] , cur_point[1] ] = df_evaluator(cur_o)
            # Integrate the orbit and evaluate the perturbed DF
            else:
                cur_o.integrate(times, pot)
                df[ cur_point[0] , cur_point[1] ] = df_evaluator(cur_o(times[-1]))
            ##ie
            computed_df[ cur_point[0] , cur_point[1] ] = 1.
            
        # Now check all points around this point
        check_df_around = np.zeros((3,3))
        grad_df_around = np.zeros((3,3))
        check_df_around[1,1] = df[ cur_point[0] , cur_point[1] ]
        
        for i in range(3):
            for j in range(3):
                
                # Make sure this isn't the chosen point. We know we've already 
                # evaluated the DF there.
                if i == 1 and j == 1:
                    continue
                ##fi
                
                # Determine the location of the subgrid points w.r.t. the 
                # larger grid
                act_i = int(cur_point[0]-1+i)
                act_j = int(cur_point[1]-1+j)
                
                # Check if the DF at this point has already been computed
                if computed_df[ act_i , act_j ] == 1:
                    check_df_around[ i, j ] = df[ act_i, act_j ]
                else:
                    # If it hasn't then run the orbit
                    check_around_o = orbit.Orbit(vxvv=[R*apu.kpc, 
                                          vR_range[act_i]*apu.km/apu.s, 
                                          vT_range[act_j]*apu.km/apu.s, 
                                          z*apu.kpc,
                                          0*apu.km/apu.s,
                                          phi*apu.radian])
                    
                    # Should the unperturbed DF be computed? 
                    if compute_unperturbed:
                        df[ act_i , act_j ] = df_evaluator(check_around_o)                      
                    ##fi
                    else:
                        check_around_o.integrate(times,pot)
                        df[ act_i , act_j ] = df_evaluator(check_around_o(times[-1]))
                    ##ie
                    check_df_around[i,j] = df[ act_i , act_j ]
                    computed_df[ act_i , act_j ] = 1.
                ##ie
                
                # Compute the gradient
                grad_df_around[i,j] = check_df_around[i,j]-check_df_around[1,1]                    
            ###j
        ###i
        
        # Have we found a global maximum?
        if np.max(grad_df_around) == 0:
            
            # If this point is more than double all of the surrounding points
            # then it's probably an imposter point. If we've already been here 
            # it's a genuine maxima.
            where_gt_double_surround = len( np.where( check_df_around/check_df_around[1,1] < 0.5 )[0] )
            if where_gt_double_surround == 8 and fake_maxima[cur_point[0], cur_point[1]] == 0:
                
                # Probably an imposter. Update position towards least negative 
                # gradient, and jump twice the distance so we don't come 
                # back to this point. Record this point
                fake_maxima[ cur_point[0], cur_point[1] ] = 1
            ##fi    
            else:
                # We've found a maximum!
                maximum_location = [ cur_point[0], cur_point[1] ]  
                df_max = df[ cur_point[0], cur_point[1] ]
                found_df_max = True
                break  
            ##ie
        ##fi
        
        # Otherwise update the position of the checking point
        max_grad_loc = np.where( grad_df_around == np.max(grad_df_around) )
        max_grad_i = max_grad_loc[0][0]
        max_grad_j = max_grad_loc[1][0]
        new_cur_i = cur_point[0] + max_grad_i - 1
        new_cur_j = cur_point[1] + max_grad_j - 1
        
        # If we found a fake maximum make a double jump
        if fake_maxima[ cur_point[0], cur_point[1] ] == 1:
            grad_df_around[1,1] = -9999
            # Find the second largest gradient, do a double jump
            max_grad_loc = np.where( grad_df_around == np.max(grad_df_around) )
            max_grad_i = max_grad_loc[0][0]
            max_grad_j = max_grad_loc[1][0]
            new_cur_i = cur_point[0] + 2*max_grad_i
            new_cur_j = cur_point[1] + 2*max_grad_j
        
        cur_point = [ int(new_cur_i), int(new_cur_j) ]
    ##wh

    # Set the starting row to be that of the maximum.
    level_start = [ maximum_location[0], maximum_location[1] ]
    cur_point = [ level_start[0], level_start[1] ]
    
    # Now start at the maximum and search the grid. First go upwards in 'i'
    increase_i_done = False
    while increase_i_done == False:
        
        increase_j_done = False
        first_of_i = True
        # Check increasing 'j'
        while increase_j_done == False:
            
            # Compute the DF at the current point
            # Check if the DF at this point has already been computed
            if computed_df[ cur_point[0] , cur_point[1] ] == 0:
                # If it hasn't then run the orbit
                cur_o = orbit.Orbit(vxvv=[  R*apu.kpc, 
                                            vR_range[ cur_point[0] ]*apu.km/apu.s, 
                                            vT_range[ cur_point[1] ]*apu.km/apu.s, 
                                            z*apu.kpc,
                                            0*apu.km/apu.s,
                                            phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if compute_unperturbed:
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o)  
                else:
                    cur_o.integrate(times,pot)
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times[-1]))
                ##ie
                computed_df[ cur_point[0], cur_point[1] ] = 1.
            ##ie
            
            # Now compute the gradient between this point and one point back in 'j'
            if first_of_i:
                cur_grad = 0
                first_of_i = False
            else:
                cur_grad = df[ cur_point[0], cur_point[1] ] - df[ cur_point[0], cur_point[1]-1 ]
            ##ie
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_df_norm = df[ cur_point[0], cur_point[1] ] / df_max
            else:
                cur_df_norm = df[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_df_norm < threshold:
                increase_j_done = True
            ##fi
            
            # Update the current position
            cur_point = [ cur_point[0], cur_point[1]+1 ]
        ##wh
        
        # Done increasing 'j'. Now decrease 'j'. Set the curret position back 
        # to the maximum at this level
        cur_point = [ level_start[0], level_start[1] ]
        decrease_j_done = False
        # Check decreasing 'j'
        while decrease_j_done == False:
            
            # Compute the DF at the current point
            # Check if the DF at this point has already been computed
            if computed_df[ cur_point[0] , cur_point[1] ] == 0:
                # If it hasn't then run the orbit
                cur_o = orbit.Orbit(vxvv=[  R*apu.kpc, 
                                            vR_range[ cur_point[0] ]*apu.km/apu.s, 
                                            vT_range[ cur_point[1] ]*apu.km/apu.s, 
                                            z*apu.kpc,
                                            0*apu.km/apu.s,
                                            phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if compute_unperturbed:
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o)  
                else:
                    cur_o.integrate(times,pot)
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times[-1]))
                ##ie
                computed_df[ cur_point[0], cur_point[1] ] = 1.
            ##ie
            
            # Now compute the gradient between this point and one point back in 'j'
            cur_grad = df[ cur_point[0], cur_point[1] ] - df[ cur_point[0], cur_point[1]+1 ]
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_df_norm = df[ cur_point[0], cur_point[1] ] / df_max
            else:
                cur_df_norm = df[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_df_norm < threshold:
                decrease_j_done = True
            ##fi
            
            # Update the current position
            cur_point = [ cur_point[0], cur_point[1]-1 ]
        ##wh
        
        # Did this row contribute any values of the perturbed DF above the 
        # threshold?
        if threshold_norm == True:
            cur_i_df_norm = df[ level_start[0], : ] / df_max
        else:
            cur_i_df_norm = df[ level_start[0], : ]
        ##ie
        
        # Check if this row contributed anything 
        if len( np.where( cur_i_df_norm > threshold)[0] ) == 0:
            increase_i_done = True
        ##fi
        
        # Now update to a new value of i. Find the maximum value the DF in this 
        # row and go upwards there
        where_i_max = np.argmax( df[ level_start[0], : ] )
        level_start = [ level_start[0]+1, where_i_max ]
        cur_point = [ level_start[0], level_start[1] ]
        
    ##wh
    
    # Now go down in 'i'. Set the level to be 1 below the max
    level_start = [ maximum_location[0]-1, maximum_location[1] ]
    cur_point = [ level_start[0], level_start[1] ]
    
    decrease_i_done = False
    while decrease_i_done == False:
        
        increase_j_done = False
        first_of_i = True
        # Check increasing 'j'
        while increase_j_done == False:
            
            # Compute the DF at the current point
            # Check if the DF at this point has already been computed
            if computed_df[ cur_point[0] , cur_point[1] ] == 0:
                # If it hasn't then run the orbit
                cur_o = orbit.Orbit(vxvv=[  R*apu.kpc, 
                                            vR_range[ cur_point[0] ]*apu.km/apu.s, 
                                            vT_range[ cur_point[1] ]*apu.km/apu.s, 
                                            z*apu.kpc,
                                            0*apu.km/apu.s,
                                            phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if compute_unperturbed:
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o)  
                else:                  
                    cur_o.integrate(times,pot)
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times[-1]))
                ##ie
                computed_df[ cur_point[0], cur_point[1] ] = 1.
            ##ie
            
            # Now compute the gradient between this point and one point back in 'j'
            if first_of_i:
                cur_grad = 0
                first_of_i = False
            else:
                cur_grad = df[ cur_point[0], cur_point[1] ] - df[ cur_point[0], cur_point[1]-1 ]
            ##ie
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_df_norm = df[ cur_point[0], cur_point[1] ] / df_max
            else:
                cur_df_norm = df[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_df_norm < threshold:
                increase_j_done = True
            ##fi
            
            # Update the current position
            cur_point = [ cur_point[0], cur_point[1]+1 ]
        ##wh
        
        # Done increasing 'j'. Now decrease 'j'. Set the curret position back 
        # to the maximum at this level
        cur_point = [ level_start[0], level_start[1] ]
        decrease_j_done = False
        # Check decreasing 'j'
        while decrease_j_done == False:
            
            # Compute the DF at the current point
            # Check if the DF at this point has already been computed
            if computed_df[ cur_point[0] , cur_point[1] ] == 0:
                # If it hasn't then run the orbit
                cur_o = orbit.Orbit(vxvv=[  R*apu.kpc, 
                                            vR_range[ cur_point[0] ]*apu.km/apu.s, 
                                            vT_range[ cur_point[1] ]*apu.km/apu.s, 
                                            z*apu.kpc,
                                            0*apu.km/apu.s,
                                            phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if compute_unperturbed:
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o)  
                else:
                    cur_o.integrate(times,pot)
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times[-1]))
                ##ie
                computed_df[ cur_point[0], cur_point[1] ] = 1.
            ##ie
        
            # Now compute the gradient between this point and one point back in 'j'
            cur_grad = df[ cur_point[0], cur_point[1] ] - df[ cur_point[0], cur_point[1]+1 ]
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_df_norm = df[ cur_point[0], cur_point[1] ] / df_max
            else:
                cur_df_norm = df[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_df_norm < threshold:
                decrease_j_done = True
            ##fi
            
            # Update the current position
            cur_point = [ cur_point[0], cur_point[1]-1 ]
        ##wh
        
        # Did this row contribute any values of the perturbed DF above the 
        # threshold?
        if threshold_norm == True:
            cur_i_df_norm = df[ level_start[0], : ] / df_max
        else:
            cur_i_df_norm = df[ level_start[0], : ]
        ##ie
        
        # Check if this row contributed anything 
        if len( np.where( cur_i_df_norm > threshold)[0] ) == 0:
            decrease_i_done = True
        ##fi
        
        # Now update to a new value of i. Find the maximum value the DF in this 
        # row and go upwards there
        where_i_max = np.argmax( df[ level_start[0], : ] )
        level_start = [ level_start[0]-1, where_i_max ]
        cur_point = [ level_start[0], level_start[1] ]
        
    ##wh
    
    print('All done')
    
    # All done!
    return df    
#def

# ----------------------------------------------------------------------------

def calculate_df_vmoments(df, vR_range, vT_range, dvR, dvT):
    '''calculate_df_vmoment:
    
    Calculate the velocity moments of a DF.
    
    Args:
        df (array) - The DF value array which goes [vT,vR]
        v_range (array) - The velocity array
        dv (float) - The 
    
    Returns:
        velocity_moments (array) [dens, vR average, vR std deviation, vT average, vT std deviation]
    '''
    
    dens = np.sum(df) * dvR * dvT
    vR_avg = (1/dens) * np.sum( np.sum(df, axis=1) * vR_range ) * dvR * dvT
    vT_avg = (1/dens) * np.sum( np.sum(df, axis=0) * vT_range ) * dvR * dvT
    vR_std = np.sqrt((1/dens) * np.sum( np.sum(df, axis=1) * np.square( vR_avg - vR_range ) ) * dvR * dvT)
    vT_std = np.sqrt((1/dens) * np.sum( np.sum(df, axis=0) * np.square( vT_avg - vT_range ) ) * dvR * dvT)

    return [dens,vR_avg,vT_avg,vR_std,vT_std]
    
#def
    
# ----------------------------------------------------------------------------

def gen_vRvT_1D(dvT, dvR, vR_low, vR_hi, vT_low, vT_hi, verbose=0):
    '''gen_vRvT_1D
    '''
    
    # Generate the velocity range
    vR_range = np.arange( vR_low, vR_hi, dvR )
    vT_range = np.arange( vT_low, vT_hi, dvT )

    # Generate the array of distribution function values
    dfp = np.zeros((len(vR_range),len(vT_range)))
    df0 = np.zeros((len(vR_range),len(vT_range)))

    # Output information
    if verbose > 0:
        print( str(len(vR_range)*len(vT_range))+' independent velocities' )
        print( str(len(vR_range))+' Between vR=['+str(round(np.amin(vR_range)))+','+str(round(np.amax(vR_range)))+']')
        print( str(len(vR_range))+' Between vR=['+str(round(np.amin(vT_range)))+','+str(round(np.amax(vT_range)))+']')
        if verbose > 1:
            print('\n')
            print(vR_range)
            print(vT_range)
            print('\n')
        ##fi
    ##fi
    
    return df0, dfp, vR_range, vT_range
    
#def

# ----------------------------------------------------------------------------

def hist_df(df, vR_low, vR_hi, vT_low, vT_hi, fig, ax, log=False):
    '''hist_df
    '''
    
    ## Make the original distribution function
    img_arr = np.rot90( df/np.max(df) )
    if log:
        img = ax.imshow(np.log10(img_arr), interpolation='nearest',
                            extent=[vR_low, vR_hi, vT_low, vT_hi],
                            cmap='viridis', vmax=0, vmin=-3)
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(r'$\log [f/f_{max}]$', fontsize=16)
    else:
        img = ax.imshow(img_arr, interpolation='nearest',
                            extent=[vR_low, vR_hi, vT_low, vT_hi],
                            cmap='viridis', vmax=1, vmin=0)
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(r'$f/f_{max}$', fontsize=16)
    ##ie

    # Decorate
    ax.set_xlabel(r'$V_{R}$ [km/s]', fontsize=14)
    ax.set_ylabel(r'$V_{\phi}$ [km/s]', fontsize=14)

    return fig, ax, cbar,
#def

# ----------------------------------------------------------------------------

def generate_triaxial_df_map_polar(dr,darc,range_r,velocity_parms,fileout,
                                    range_phi=[0,np.pi],
                                    halo_b=1.0,
                                    halo_c=1.0,
                                    halo_phi=1.0,
                                    mirror_x=True,
                                    t_evolve=10.0,
                                    tform=-9.0,
                                    tsteady=8.0,
                                    ):
    '''generate_triaxial_df_map_polar:
    
    Make a map of triaxial DF values on a polar grid.

    Args:
        dr [float] - radial bin spacing
        darc [float] - arclength bin spacing
        range_r [2-array] - Range of R over which to evaluate (low,high)
        velocity_parms [4-array] - parameters to set tangential and radial
            velocity grid spacing (dv..) and width in units of DF sigmas 
            (n_sigma..) (dvR,dvT,n_sigma_vR,n_sigma_vT)
        fileout [string] - Output filename
        range_phi [2-array] - Range of phi over which to evaluate (low,high)
        halo_b [float] - Halo b/a [1.0]
        halo_c [float] - Halo c/a [1.0]
        halo_phi [float] - Halo position angle in X-Y plane [1.0]
        mirror_x [bool] - Only compute the perturbed density for half the 
            circle given its symmetry, to save time [True]
        t_evolve [float] - Total integration time of the system in Gyr [10.0]
        t_form [float] - Time at which to begin the smooth transformation in 
            Gyr (should be negative) [-9.0]
        t_steady [float] - Duration of the transformation phase in positive 
            Gyr [8.0]
        make_log [bool] - Output a text log? [True]
        
        
    Outputs:
        
    '''
    
    # Make the potential and triaxial halo
    mwpot = potential.MWPotential2014
    trihalo = ast1501.potential.make_MWPotential2014_triaxialNFW(halo_b=halo_b, 
        halo_phi=halo_phi, halo_c=halo_c)
    tripot_grow = ast1501.potential.make_tripot_dsw(trihalo=trihalo, tform=tform, tsteady=tsteady)
    potential.turn_physical_off(tripot_grow)    
    
    # Velocity dispersions in km/s
    sigma_vR = 46/1.5
    sigma_vT = 40/1.5
    sigma_vZ = 28/1.5
    
    # Action angle coordinates and the DF
    qdf_aA= actionAngleAdiabatic(pot=potential.MWPotential2014, c=True)
    qdf = df.quasiisothermaldf( hr= 2*apu.kpc,
                                sr= sigma_vR*(apu.km/apu.s),
                                sz= sigma_vZ*(apu.km/apu.s),
                                hsr= 9.8*(apu.kpc),
                                hsz= 7.6*(apu.kpc),
                                pot= potential.MWPotential2014, 
                                aA= qdf_aA)
                                
    # Set velocity deltas and range. vT range will be set in the for loop
    # so it can be offset by the local circular velocity.
    dvT = velocity_parms[0]
    dvR = velocity_parms[1]
    vR_low = -velocity_parms[2]*sigma_vR
    vR_hi = velocity_parms[2]*sigma_vR
    
    # Set radial grid. azimuthal grid will be set in the loop.
    r_posns = np.arange( range_r[0], range_r[1], dr) + dr/2
    
    # Make a log?
    if make_log:
        logfile = open('./log.txt','w')
    ##fi
    
    radius_out = np.empty(len(r_posns),dtype='object')
    
    # Loop over all radii
    for i in range( len( r_posns ) ):
        
        # Decide on the phi spacing
        arc_length = np.pi*r_posns[i]
        n_bins = math.ceil(arc_length/darc)
        dphi = (range_phi[1]-range_phi[0])/n_bins
    
        # Choose the phi positions. Goes from -pi/2 to +pi/2. Phi=0 is towards 
        # the sun.
        phi_posns = np.linspace(range_phi[0], range_phi[1], num=n_bins, 
            endpoint=False) + dphi/2 - np.pi/2
            
        # Arrays to store velocity moments
        vR_mean = np.zeros(len(phi_posns))
        vR_std = np.zeros(len(phi_posns))
        vT_mean = np.zeros(len(phi_posns))
        vT_std = np.zeros(len(phi_posns))
        
        # Now loop over phi positions:
        for j in range( len( phi_posns ) ):
            
            # Set the tangential velocity about the local circular velocity
            vcirc_offset = potential.vcirc(mwpot,r_posns[i]*apu.kpc) * mwpot[0]._vo # in km/s
            vT_low = -velocity_parms[3]*sigma_vT + vcirc_offset
            vT_hi = velocity_parms[3]*sigma_vT + vcirc_offset
            
            # Make the velocity range, and distribution function arrays
            _, dfp, vR_range, vT_range = ast1501.df.gen_vRvT_1D(dvT, dvR, 
                                                                vR_low, vR_hi, 
                                                                vT_low, vT_hi)

            R_z_phi = [r_posns[i],0,phi_posns[j]]

            # Now use the radius and phi position to evaluate the triaxial DF 
            t1 = time.time()
            
            dfp = ast1501.df.evaluate_df_adaptive_vrvt(R_z_phi, times, tripot_grow, 
                qdf, vR_range, vT_range, dfp, threshold=0.0001)
                
            # Calculate the moments
            moments = ast1501.df.calculate_df_vmoments(dfp, vR_range, vT_range, 
                dvR, dvT)
            vR_mean[j], vT_mean[j], vR_std[j], vT_std[j] = moments[1:]
            
            t2 = time.time()
            logfile.write( 'R='+str(round(r_posns[i],2))+' kpc, phi='+str(round(phi_posns[j],2))+' rad, t='+str(round(t2-t1))+' s\n' )

            fig = plt.figure( figsize=(5,5) )
            ax = fig.add_subplot(111)

            # Plot
            fig, ax, cbar = ast1501.df.hist_df(dfp, vR_low, vR_hi, vT_low, vT_hi, fig, ax, log=True)
            ax.set_title('R='+str(round(r_posns[i],2))+' kpc, phi='+str(round(phi_posns[j],2))+' rad')
            fig.savefig( 'data/R-'+str(round(r_posns[i],2))+'_phi-'+str(round(phi_posns[j],2))+'_dfp.pdf' )
            
            print( 'Done R: '+str(round(r_posns[i],2))+' phi: '+str(round(phi_posns[j],2)) )
            
        ###j
        radius_out[i] = np.array([r_posns[i],phi_posns,vR_mean,vT_mean,vR_std,vT_std],dtype='object')
#def

np.save(fileout,radius_out)

logfile.close()

# ----------------------------------------------------------------------------

def generate_triaxial_df_map_rect():
    '''generate_triaxial_df_map_rect:
    
    Args:
    
    Outputs:
    '''

# ----------------------------------------------------------------------------

def generate_grid_radial(   r_range,
                            phi_range,
                            delta_r,
                            delta_phi,
                            delta_phi_in_arc=True):
    '''generate_grid_radial:
    
    Generate a radial grid pursuant to a radial range, azimuth range, delta 
    in r and azimuth.
    
    Args:
        r_range (float 2-array) - 2 element array of the r range
        phi_range (float 2-array) - 2 element array of the phi range
        delta_r (float) - spacing in the r direction
        delta_phi (float) - spacing in the azimuth direction
        delta_phi_in_arc [bool] - delta_phi is given as a delta in arclength 
            instead of angle [True]
    
    Returns:
        grid_rpoints (float array) - 1-D array of x grid points
        grid_phipoints (float array) - 1-D array of y grid points
    '''
    
    # Make the radial points
    grid_rpoints_core = np.arange( r_range[0], r_range[1], delta_r )
    
    # Make the empty arrays to hold the data
    grid_rpoints = np.array([])
    grid_phipoints = np.array([])
    
    for i in range( len( grid_rpoints_core ) ):
        
        # Determine the azimuth points at this radii. Either fixed angular
        # interval or fixed arc interval.
        if delta_phi_in_arc:
            # Calculate the arc length limits at this radius
            arc_min = phi_range[0] * grid_rpoints_core[i]
            arc_max = phi_range[1] * grid_rpoints_core[i]
            grid_arcpoints_new = np.arange( arc_min, arc_max, delta_phi )
            
            # Center the points to account for an uneven gridding, then center 
            # the points in the spacing window and convert back to angle.
            grid_arcpoints_new -= ( ( arc_max - arc_min ) % delta_phi )
            grid_arcpoints_new += ( delta_phi / 2 )
            grid_phipoints_new = grid_arcpoints_new / grid_rpoints_core[i]
            
            # Append to the total array
            grid_phipoints = np.append( grid_phipoints, new_grid_phipoints )
            
        else:
            # Just do a straight range in azimuth.
            grid_phipoints = np.append( grid_phipoints, 
                np.arrange( phi_range[0], phi_range[1], delta_phi ) )
        ##ie
    
    return grid_rpoints, grid_phipoints
        
    ###i
#def
    
# ----------------------------------------------------------------------------

def generate_grid_rect():
    '''generate_grid_rect:
    
    Generate a rectangular grid pursuant to an x range, a y range and a delta 
    in r and azimuth.
    
    Args:
        x_range (float 2-array) - 2 element array of the X range
        y_range (float 2-array) - 2 element array of the Y range
        delta_x (float) - spacing in the X direction
        delta_y (float) - spacing in the Y direction
        include_endpoints [bool] - Include the end points in the array [False]
    
    Returns:
        grid_xpoints (float array) - 1-D array of x grid points
        grid_ypoints (float array) - 1-D array of y grid points
    '''
    
    # Make the x and y range
    grid_xpoints_core = np.arange(x_range[0], x_range[1], delta_x)
    grid_ypoints_core = np.arange(y_range[0], y_range[1], delta_y)
    
    # Make the x and y grid
    grid_xpoints, grid_ypoints = np.meshgrid(   grid_xpoints_core,
                                                grid_ypoints_core,
                                                indexing='ij')
                                                
    return grid_xpoints, grid_ypoints
#def
    
