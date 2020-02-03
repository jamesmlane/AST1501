# ----------------------------------------------------------------------------
#
# TITLE - df.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS: 
# 1. evaluate_df_adaptive_vRvT
# 2. calculate_df_vmoments_vRvT
# 2. hist_df
# 3. calculate_df_vmoments
# 4. generate_triaxial_df_map_polar
# 5. generate_triaxial_df_map_rect
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Functions to calculate DFs
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy, time
from itertools import repeat
import multiprocessing

## Plotting
from matplotlib import pyplot as plt

## Astropy
from astropy import units as apu

## galpy
from galpy import orbit
from galpy import potential
from galpy import df
from galpy import actionAngle
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv
from galpy.util import multi

# ----------------------------------------------------------------------------

def evaluate_df_adaptive_vRvT(R_z_phi,
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
    evaluate_df_adaptive_vRvT:
    
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
                df[ cur_point[0] , cur_point[1] ] = df_evaluator(cur_o(times))[-1]
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
                        df[ act_i , act_j ] = df_evaluator(check_around_o(times))[-1]
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
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times))[-1]
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
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times))[-1]
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
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times))[-1]
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
                    df[ cur_point[0], cur_point[1] ] = df_evaluator(cur_o(times))[-1]
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
    
    # print('All done')
    
    # All done!
    return df    
#def

# ----------------------------------------------------------------------------

def calculate_df_vmoments_vRvT(df, vR_range, vT_range, dvR, dvT):
    '''calculate_df_vmoments_vRvT:
    
    Calculate the velocity moments of a DF that spans vR and vT.
    
    Args:
        df (array) - The DF value array which goes [vT,vR]
        vR_range (array) - The velocity array in vR
        vT_range (array) - The velocity array in vT
        dvR (float) - The spacing in vR
        dvT (float) - The spacing in vT
    
    Returns:
        velocity_moments (array) [dens, vR_avg, vR_stds, vT_avg, vT_std]
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
    '''gen_vRvT_1D:
    
    Generate 1D arrays of both tangential and radial velocities given deltas
    and ranges.
    
    Args:
        dvT (float) - delta in vT
        dvR (float) - delta in vR
        vR_low (float) - Low end of vR range in km/s
        vR_hi (float) - High end of vR range in km/s
        vT_low (float) - Low end of vT range in km/s
        vT_hi (float) - High end of vT range in km/s
        verbose (int) - integer either 0 or >0 to specify whether velocity range 
            is printed to screen [0]
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
    '''hist_df:
    
    Plot a 2D histogram of a distribution function
    
    Args:
        df (float array) - The distribution function array
        vR_low (float) - Low end of vR range in km/s
        vR_hi (float) - High end of vR range in km/s
        vT_low (float) - Low end of vT range in km/s
        vT_hi (float) - High end of vT range in km/s
        fig (matplotlib Figure) - figure object to plot on
        ax (matplotlib Axis) - axis object to plot on
        log (bool) - Plot the DF values in log instead of linear [False]
    '''
    
    ## Make the original distribution function
    img_arr = np.rot90( df/np.max(df) )
    if log:
        # Set values less than 0 very small so log doesn't crap on them.
        img_arr[np.where(img_arr <= 0)] = 1e-10
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

def evaluate_df_polar(r,phi,pot,df,velocity_parms,times,
                            sigma_vR=30.0,
                            sigma_vT=30.0,
                            evaluator_threshold=0.0001,
                            plot_df=True,
                            coords_in_xy=False,
                            verbose=0
                            ):
    '''evaluate_df_polar:
    
    Given a polar position, evaluate the distribution function across a grid of 
    velocities and return the moments of the distribution function. Takes in 
    a positional argument and an array of velocities
    
    Args:
        r (float) - Galactocentric cylindrical radius in kpc
        phi (float) - azimuth in rad (0 at Sun-GC line, increases CCW from GNP)
        pot (galpy Potential object) - Time dependent potential in which to 
            evaluate DF
        df (galpy df object) - distribution function object
        velocity_parms (4-array) - parameters to set tangential/radial
            velocity grid spacing (dv..), and width in units of DF sigmas 
            (n_sigma..), looks like: (dvR,dvT,n_sigma_vR,n_sigma_vT)
        times (float array) - Array of negative times from 0 to t_evolve. Must 
            be compatible with the evolution timescale for the potential.
            
        sigma_vR (float) - The radial velocity dispersion in km/s. Used to set 
            the range for velocity exploration [30.0]
        sigma_vT (float) - The tangential velocity dispersion in km/s. Used to 
            set the range for velocity exploration [30.0]
        evaluator_threshold (float) - The threshold at which the DF evaluator 
            should stop, relative to the peak value of the DF [0.0001]
        plot_df (bool) - Plot the distribution function [True]
        coords_in_xy (bool) - if True r and phi are given as galactocentric x 
            and y (in kpc) instead of polar coordinates. Note X is positive 
            from GC->Sun and Y is positive towards galactic rotation [False]
        verbose (int) - Verbosity level [0]
                
    Returns:
        output_array (float array) - array of [ r, phi, x, y, vr_mean, vt_mean, 
            vr_std, vt_std]
    
    '''
    
    # Set the correct coordinates
    if coords_in_xy:
        x = copy.copy(r)
        y = copy.copy(phi)
        r = (x**2 + y**2)**0.5
        phi = np.arctan2(y,x)
    else:
        x = np.cos( phi ) * r
        y = np.sin( phi ) * r
    ##ie        
    
    # Set velocity deltas and radial range.
    dvT = velocity_parms[0]
    dvR = velocity_parms[1]
    vR_low = -velocity_parms[2]*sigma_vR
    vR_hi = velocity_parms[2]*sigma_vR
    
    # Determine the tangential velocity at the local radius. Currently forcing 
    # to use MWPotential2014. Use it to set the tangential velocity range.
    mwpot = potential.MWPotential2014
    vcirc_offset = potential.vcirc(mwpot, r*apu.kpc) * mwpot[0]._vo # in km/s
    vT_low = -velocity_parms[3]*sigma_vT + vcirc_offset
    vT_hi = velocity_parms[3]*sigma_vT + vcirc_offset
    
    # Make the velocity range, and distribution function arrays
    _, dfp, vR_range, vT_range = gen_vRvT_1D(dvT, dvR, 
                                            vR_low, vR_hi, 
                                            vT_low, vT_hi)
    
    # Make the position array
    R_z_phi = [r,0,phi]
    
    # Record the time
    t1 = time.time()
    
    # Calculate the distribution function using the adaptive evaluator.
    dfp = evaluate_df_adaptive_vRvT(R_z_phi, times, pot, df, vR_range, 
        vT_range, dfp, threshold=evaluator_threshold)
        
    # Calculate the moments of the DF
    moments = calculate_df_vmoments_vRvT(dfp, vR_range, vT_range, dvR, dvT)
    dens, vR_mean, vT_mean, vR_std, vT_std = moments
    
    # Write the logging text
    t2 = time.time()
    logtext = 'R='+str(round(r,2))+\
              ' kpc, phi='+str(round(phi,2))+\
              ' rad, t='+str(round(t2-t1))+' s\n'
    
    # Plot the distribution function
    if plot_df:
        # Make a figure
        fig = plt.figure( figsize=(5,5) )
        ax = fig.add_subplot(111)
        # Plot
        fig, ax, _ = hist_df(dfp, vR_low, vR_hi, vT_low, vT_hi, fig, ax, log=True)
        ax.set_title('R='+str(round(r,2))+' kpc, phi='+str(round(phi,2))+' rad')
        fig.savefig( './R-'+str(round(r,2))+'_phi-'+str(round(phi,2))+'_dfp.pdf' )
        plt.close(fig)
    ##fi
    
    if verbose > 0:
        print( 'Done R: '+str(round(r,2))+' phi: '+str(round(phi,2)) )
    ##fi
    
    output_array = np.array([r,phi,x,y,vR_mean,vR_std,vT_mean,vT_std,logtext])
    
    return output_array
#def

# ----------------------------------------------------------------------------

def evaluate_df_polar_serial(r,phi,pot,df,velocity_parms,times,
                                sigma_vR=30.0,
                                sigma_vT=30.0,
                                evaluator_threshold=0.0001,
                                plot_df=True,
                                coords_in_xy=False,
                                logfile=None,
                                verbose=0):
    '''evaluate_df_polar_serial:
    
    Calculate the velocity moments of a DF pertaining to a time-varying 
    potential in serial. Function is a wrapper of evaluate_df_polar
    
    Args:
        r (float) - Galactocentric cylindrical radius in kpc
        phi (float) - azimuth in rad (0 at Sun-GC line, increases CCW from GNP)
        pot (galpy Potential object) - Time dependent potential in which to 
            evaluate DF
        df (galpy df object) - distribution function object
        velocity_parms (4-array) - parameters to set tangential/radial
            velocity grid spacing (dv..), and width in units of DF sigmas 
            (n_sigma..), looks like: (dvR,dvT,n_sigma_vR,n_sigma_vT)
        times (float array) - Array of negative times from 0 to t_evolve. Must 
            be compatible with the evolution timescale for the potential.
        
        All other parameters see documentation of evaluate_df_polar
    
    Returns:
        results (object array) - array of results from each evaluation of 
            evaluate_df_polar, which each have the form:
            [ r, phi, x, y, vr_mean, vt_mean, vr_std, vt_std]
    '''
    
    n_calls = len(r)
    
    output = np.array([])
    
    for i in range(n_calls):
    
        results = evaluate_df_polar(r[i],phi[i],pot,df,velocity_parms,times,
                    sigma_vR=sigma_vR,sigma_vT=sigma_vT,
                    evaluator_threshold=evaluator_threshold,plot_df=plot_df,
                    coords_in_xy=coords_in_xy,logfile=logfile,verbose=verbose)
                    
        output = np.append(output,results)
            
    ###i
    
    return output.reshape(n_calls,8)

# ----------------------------------------------------------------------------

def evaluate_df_polar_parallel(r,phi,use_pot,use_df,velocity_parms,times,ncores,
                                sigma_vR=30.0,
                                sigma_vT=30.0,
                                evaluator_threshold=0.0001,
                                plot_df=True,
                                coords_in_xy=False,
                                logfile=None,
                                verbose=0):
    '''evaluate_df_polar_parallel:
    
    Calculate the velocity moments of a DF pertaining to a time-varying 
    potential in parallel. Function is a wrapper of evaluate_df_polar
    
    Args:
        r (float) - Galactocentric cylindrical radius in kpc
        phi (float) - azimuth in rad (0 at Sun-GC line, increases CCW from GNP)
        use_pot (galpy Potential object) - Time dependent potential in which to 
            evaluate DF
        use_df (galpy df object) - distribution function object
        velocity_parms (4-array) - parameters to set tangential/radial
            velocity grid spacing (dv..), and width in units of DF sigmas 
            (n_sigma..), looks like: (dvR,dvT,n_sigma_vR,n_sigma_vT)
        times (float array) - Array of negative times from 0 to t_evolve. Must 
            be compatible with the evolution timescale for the potential.
        ncores (int) - Number of cores to use in parallel.
        
        All other parameters see documentation of evaluate_df_polar
    
    Returns:
        results (object array) - array of results from each evaluation of 
            evaluate_df_polar, which each have the form:
            [ r, phi, x, y, vr_mean, vt_mean, vr_std, vt_std]
    '''
    
    # Number of total iterations of the evaluate_df_polar function
    n_calls = len(r)
    
    # Make a lambda function to pass keywords
    lambda_func = (lambda x: evaluate_df_polar(r[x], phi[x], 
        use_pot, use_df, velocity_parms, times, sigma_vR, sigma_vT, 
        evaluator_threshold, plot_df, coords_in_xy, verbose))
    
    # Evaluate the results in parallel
    results = multi.parallel_map(lambda_func, 
        np.arange(0,n_calls,1,dtype='int'),  
        numcores=ncores)
        
    # Turn the results into a numpy array
    results = np.array(results)
    
    # Now if we're logging then append the log text to the logfile and remove 
    # it from the output
    if logfile != None:
        logfile.write('\n')
        for i in range( results.shape[0] ):
            logfile.write(results[i,-1])
        ###i
    ##fi
    
    # Returned the pared down results array, without the logging text.
    return results[:,:-1]
#def

# ----------------------------------------------------------------------------

def triaxial_df_serial_wrapper(r,phi,halo_parms):
    '''triaxial_df_parallel_wrapper:
    
    Very thin wrapper of other methods that uses defult methods to evaluate 
    the DF of a triaxial halo. Works in serial, i.e. no parallel.
    
    Args:
        r (float) - Galactocentric cylindrical radius in kpc
        phi (float) - azimuth in rad (0 at Sun-GC line, increases CCW from GNP)
        halo_parms (4-array) - 
            
    Return:
        results
    '''
    
    # Defaults for many parameters
    t_evolve, t_form, t_steady = [10,-9,8] # Timing in Gyr
    times = -np.array([0,t_evolve]) * apu.Gyr # Times in Gyr
    sigma_vR, sigma_vT, sigma_vZ = [30,30,20] # Velocity dispersions
    velocity_parms = [20,20,8,8]
    
    # Make the potential
    mwpot = potential.MWPotential2014
    halo_a,halo_b,halo_c,halo_phi = halo_parms
    trihalo = ast1501.potential.make_MWPotential2014_triaxialNFW(halo_b=halo_b, 
        halo_phi=halo_phi, halo_c=halo_c)
    tripot_grow = ast1501.potential.make_tripot_dsw(trihalo=trihalo, 
        tform=t_form, tsteady=t_steady)
    potential.turn_physical_off(tripot_grow)
    
    # Action angle coordinates and the DF
    qdf_aA= actionAngleAdiabatic(pot=potential.MWPotential2014, c=True)
    qdf = df.quasiisothermaldf( hr= 2*apu.kpc, sr= sigma_vR*(apu.km/apu.s),
                                sz= sigma_vZ*(apu.km/apu.s),
                                hsr= 9.8*(apu.kpc), hsz= 7.6*(apu.kpc),
                                pot= potential.MWPotential2014, aA= qdf_aA)
    
    logfile = open('./log.txt','w')
    
    # Now run the serial evaluator
    results = evaluate_df_polar_serial(r,phi,tripot_grow,qdf,velocity_parms,times,
    sigma_vR=sigma_vR,sigma_vT=sigma_vT,evaluator_threshold=0.0001,
    plot_df=True,coords_in_xy=False,logfile=logfile,verbose=1)
    
    logfile.close()
    
    return results
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
    
    # Make times  
    times = -np.array([0,t_evolve]) * apu.Gyr
    
    # Velocity dispersions in km/s
    sigma_vR = 46/1.5
    sigma_vT = 40/1.5
    sigma_vZ = 28/1.5
    
    # Action angle coordinates and the DF
    qdf_aA= actionAngle.actionAngleAdiabatic(pot=potential.MWPotential2014, 
                                                c=True)
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
            _, dfp, vR_range, vT_range = gen_vRvT_1D(dvT, dvR, 
                                                    vR_low, vR_hi, 
                                                    vT_low, vT_hi)

            R_z_phi = [r_posns[i],0,phi_posns[j]]

            # Now use the radius and phi position to evaluate the triaxial DF 
            t1 = time.time()
            
            dfp = evaluate_df_adaptive_vRvT(R_z_phi, times, tripot_grow, 
                qdf, vR_range, vT_range, dfp, threshold=0.0001)
                
            # Calculate the moments
            moments = calculate_df_vmoments(dfp, vR_range, vT_range, 
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
    ###i
    
    # Save results
    np.save(fileout,radius_out)

    if make_log:
        logfile.close()
    ##fi
#def

# ----------------------------------------------------------------------------

def generate_triaxial_df_map_rect():
    '''generate_triaxial_df_map_rect:
    
    Args:
    
    Outputs:
    '''
    pass
#def

# ----------------------------------------------------------------------------

def generate_grid_radial(   r_range,
                            phi_range,
                            delta_r,
                            delta_phi,
                            delta_phi_in_arc=True,
                            return_rect_coords=False):
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
        return_rect_coords [bool] - Return the grid in rectangular coordinates 
            as well as polar coordinates.
    
    Returns:
        grid_rpoints (float array) - 1-D array of r grid points
        grid_phipoints (float array) - 1-D array of phi grid points
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
            grid_arcpoints_new += ( ( arc_max - arc_min ) % delta_phi )/2
            grid_phipoints_new = grid_arcpoints_new / grid_rpoints_core[i]
            
            # Make the new radii points
            grid_rpoints_new = np.ones_like(grid_phipoints_new)*grid_rpoints_core[i] + delta_r/2
            
            # Append to the total array
            grid_phipoints = np.append( grid_phipoints, grid_phipoints_new )
            grid_rpoints = np.append( grid_rpoints, grid_rpoints_new )
            
        else:
            # Just do a straight range in azimuth.
            grid_phipoints_new = np.arrange( phi_range[0], phi_range[1], delta_phi )
            grid_phipoints = np.append( grid_phipoints, grid_phipoints_new )
            grid_rpoints_new = np.ones_like(grid_phipoints_new)*grid_rpoints_core[i] + delta_r/2
            grid_rpoints = np.append( grid_rpoints, 
                np.ones_like(grid_phipoints_new)*grid_rpoints_core[i] )
        ##ie
    ###i
    if return_rect_coords:
        grid_xpoints = np.cos( grid_phipoints ) * grid_rpoints # X positive towards Sun-GC line
        grid_ypoints = np.sin( grid_phipoints ) * grid_rpoints # Y positive towards Gal. Rotation
        return grid_rpoints, grid_phipoints, grid_xpoints, grid_ypoints
    else:
        return grid_rpoints, grid_phipoints
    ##ie    
#def
    
# ----------------------------------------------------------------------------

def generate_grid_rect(x_range, y_range, delta_x, delta_y,
                        return_polar_coords=False):
    '''generate_grid_rect:
    
    Generate a rectangular grid pursuant to an x range, a y range and a delta 
    in r and azimuth.
    
    Args:
        x_range (float 2-array) - 2 element array of the X range
        y_range (float 2-array) - 2 element array of the Y range
        delta_x (float) - spacing in the X direction
        delta_y (float) - spacing in the Y direction
        
        return_polar_coords (bool) - Return the grid in polar coordinates as 
            well as rectangular coordinates? [False]
    
    Returns:
        grid_xpoints (float array) - 1-D array of x grid points
        grid_ypoints (float array) - 1-D array of y grid points
    '''
    
    # Make the x and y range
    grid_xpoints_core = np.arange(x_range[0], x_range[1], delta_x) + delta_x/2
    grid_ypoints_core = np.arange(y_range[0], y_range[1], delta_y) + delta_y/2
    
    # Make the x and y grid
    grid_xpoints, grid_ypoints = np.meshgrid(   grid_xpoints_core,
                                                grid_ypoints_core,
                                                indexing='ij')
    grid_xpoints = grid_xpoints.flatten()
    grid_ypoints = grid_ypoints.flatten()
    
    # Return the polar grid as well as the rectangular grid?
    if return_polar_coords:
        grid_rpoints = np.sqrt( grid_xpoints**2 + grid_ypoints**2)
        grid_phipoints = np.arctan2(grid_ypoints,grid_xpoints) # 0 at Sun-GC line
        return grid_xpoints, grid_ypoints, grid_rpoints, grid_phipoints
    else:
        return grid_xpoints, grid_ypoints
    ##ie
#def

# ----------------------------------------------------------------------------

def get_vsigma(source='default'):
    '''get_vsigma:
    
    Return the velocity dispersions in the radial, tangential, and vertical 
    directions for use in a quasi-isothermal distribution function in km/s. 
    Function defined for continuity across scripts.
    
    Args: 
        source (float) - Where do the values for the velocity dispersion come 
            from. ['default']
    
    Returns:
        vsigma (3-array) - Velocity dispersion in the radial, tangential, and 
            vertical directions at the solar position, in km/s.
    '''
    if source == 'default':
        return [30,20,20]
    ##ie
#def

# ----------------------------------------------------------------------------

def get_scale_lengths(source='default'):
    '''get_heights:
    
    Return the scale lengths for use in a quasi-isothermal distribution, 
    in kpc. Function defined for continuity across scripts.
    
    Args: 
        source (float) - Where do the values for the scale heights come 
            from. ['default']
    
    Returns:
        lengths (3-array) - Scale length for radial, radial velocity dispersion, 
            and vertical velocity dispersion
    '''
    if source == 'default':
        return [2,9.8,7.6]
    ##ie
#def

# ----------------------------------------------------------------------------

def make_default_MWPotential2014_qdf():
    '''make_default_MWPotential2014_qdf:
    
    Generate the default QDF for MWPotential2014 
    
    Args: 
        source (float) - Where do the values for the scale heights come 
            from. ['default']
    
    Returns:
        lengths (3-array) - Scale length for radial, radial velocity dispersion, 
            and vertical velocity dispersion
    '''
    svr,svt,svz = get_vsigma() # By default these are 30, 20, and 20 km/s
    r_scale,vr_scale,vz_scale = get_scale_lengths() # These are 2, 9.8, 7.6 kpc
    aA = actionAngle.actionAngleAdiabatic( pot=potential.MWPotential2014, 
                                           c=True)
    qdf = df.quasiisothermaldf(hr= r_scale*apu.kpc, 
                               sr= svr*(apu.km/apu.s),
                               sz= svz*(apu.km/apu.s),
                               hsr= vr_scale*(apu.kpc), 
                               hsz= vz_scale*(apu.kpc),
                               pot= potential.MWPotential2014, 
                               aA= aA)
    return qdf
#def

# ----------------------------------------------------------------------------
