# ----------------------------------------------------------------------------
#
# TITLE - df.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS:
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
# from matplotlib.backends.backend_pdf import PdfPages
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
                                dfp,
                                df0=None,
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
        dfp (array) - array to hold the perturbed DF
        df0 (array) - array to hold the unperturbed DF. If None then don't 
            compute the unperturbed DF [None]
        threshold (float) - The threshold DF value, normalized such that the 
            maximum DF value is 1 (see threshold_real) [0.001]
        threshold_norm (bool) - Is the threshold value normalized to the 
            maximum of the DF or is it just a real DF value? 
    
    Returns:
        df0 (array) - array of unperturbed DF values
        dfp (array) - array of perturbed DF values
    '''
    
    # Are we computing the unperturbed DF?
    if df0 == None:
        _compute_unperturbed = False
    else:
        _compute_unperturbed = True
    
    # First check that the dimensions are correct:
    if (df0.shape != (len(vR_range),len(vT_range))) and _compute_unperturbed == True:
        raise RuntimeError('Unperturbed DF array not correct size')
    if (dfp.shape != (len(vR_range),len(vT_range))):
        raise RuntimeError('Perturbed DF array not correct size')
    
    # Now 0 all of the elements of the DF arrays
    if _compute_unperturbed:
        df0[:,:] = 0
    dfp[:,:] = 0
    
    # Make an array of values that tell us if we've computed the DF at this 
    # location or not
    computed_df = np.zeros_like(dfp)
    
    # Unpack coordinates
    R,z,phi = R_z_phi
    
    # Get physical conversions
    pot = copy.deepcopy(pot)
    potential.turn_physical_off(pot)
    
    # Now find the maximum value of the perturbed DF. Assume that we'll start 
    # vR = 0 km/s and vT = 220 km/s, assuming they're in the arrays:
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
    
    # Now star the search for the maximum DF point. Pick the first point 
    # in the grid
    cur_point = [  np.where(vR_start == vR_range)[0][0], 
                    np.where(vT_start == vT_range)[0][0] ]
    # Initialize checker variable
    found_df_max = False   
    fake_maxima = np.zeros_like(dfp)
    
    # Loop over the maximum finder
    while found_df_max == False:
        
        # If we have butt up against the edge of the input array consider this 
        # a failed evaluation attempt
        if cur_point[0] == 0 or cur_point[0] == len(vR_range)-1 or cur_point[1] == 1 or cur_point[1] == len(vT_range)-1:
            raise RuntimeError('Attemped to evaluate perturbed DF outside of supplied grid')
        ##fi
        
        if computed_df[ cur_point[0] , cur_point[1] ] == 0
            # Calculate the orbit
            cur_o = orbit.Orbit(vxvv=[R*apu.kpc, 
                                  vR_range[cur_point[0]]*apu.km/apu.s, 
                                  vT_range[cur_point[1]]*apu.km/apu.s, 
                                  z*apu.kpc,
                                  0*apu.km/apu.s,
                                  phi*apu.radian])
        
            # Should the unperturbed DF be computed? 
            if _compute_unperturbed:
                df0[ cur_point[0] , cur_point[1] ] = df_evaluator(cur_o)
            
            # Integrate the orbit and evaluate the perturbed DF
            cur_o.integrate(times, pot)
            dfp[ cur_point[0] , cur_point[1] ] = df_evaluator(cur_o(times[-1]))
            computed_df[ cur_point[0] , cur_point[1] ] = 1.
            
        # Now check all points around this point
        check_df_around = np.zeros((3,3))
        grad_df_around = np.zeros((3,3))
        check_df_around[1,1] = dfp[ cur_point[0] , cur_point[1] ]
        
        for i in range(3):
            for j in range(3):
                
                # Make sure this isn't the chosen point.
                if i == 1 and j == 1:
                    continue
                
                # Determine the location of the subgrid points w.r.t. the 
                # larger grid
                act_i = int(cur_point[0]-1+i)
                act_j = int(cur_point[1]-1+j)
                
                # Check if the DF at this point has already been computed
                if computed_df[ act_i , act_j ] == 1:
                    check_df_around[ i, j ] = dfp[ act_i, act_j ]
                else:
                    # If it hasn't then run the orbit
                    check_around_o = orbit.Orbit(vxvv=[R*apu.kpc, 
                                          vR_range[act_i]*apu.km/apu.s, 
                                          vT_range[act_j]*apu.km/apu.s, 
                                          z*apu.kpc,
                                          0*apu.km/apu.s,
                                          phi*apu.radian])
                    
                    # Should the unperturbed DF be computed? 
                    if _compute_unperturbed:
                        df0[ act_i , act_j ] = df_evaluator(check_around_o)                      
                    
                    # Now integrate
                    check_around_o.integrate(times,pot)
                    dfp[ act_i , act_j ] = df_evaluator(check_around_o(times[-1]))
                    check_df_around[i,j] = dfp[ act_i , act_j ]
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
                dfp_max = dfp[ cur_point[0], cur_point[1] ]
                found_df_max = True
                break  
            ##ie
        ##fi
        
        # Otherwise update the position of the checking point
        max_grad_loc = np.where( grad_df_around == np.max(grad_df_around) )
        max_grad_i = max_grad_loc[0][0]
        max_grad_j = max_grad_loc[1][0]
        new_cur_i = cur_point[0] + max_grad_i
        new_cur_j = cur_point[1] + max_grad_j
        
        # If we found a fake maximum make a double jump
        if fake_maxima[ cur_point[0], cur_point[1] ] = 1:
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
    cur_point[ level_start[0], level_start[1] ]
        
    # Now start at the maximum and search the grid. First go upwards in 'i'
    increase_i_done = False
    while increase_i_done == False:
        
        increase_j_done = False
        first_of_i = True
        # Check increasing 'j'
        while increase_j_done == False:
            
            # Compute the DF at the current point
            # Check if the DF at this point has already been computed
            if computed_df[ cur_point[0] , cur_point[1] ] == 1:
                check_df_around[ i, j ] = dfp[ act_i, act_j ]
            else:
                # If it hasn't then run the orbit
                check_around_o = orbit.Orbit(vxvv=[R*apu.kpc, 
                                      vR_range[act_i]*apu.km/apu.s, 
                                      vT_range[act_j]*apu.km/apu.s, 
                                      z*apu.kpc,
                                      0*apu.km/apu.s,
                                      phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if _compute_unperturbed:
                    df0[ act_i , act_j ] = df_evaluator(check_around_o)                      
                
                # Now integrate
                check_around_o.integrate(times,pot)
                dfp[ act_i , act_j ] = df_evaluator(check_around_o(times[-1]))
                check_df_around[i,j] = dfp[ act_i , act_j ]
                computed_df[ act_i , act_j ] = 1.
            ##ie
            
            # Now compute the gradient between this point and one point back in 'j'
            if first_of_i:
                cur_grad = 0
                first_of_i = False
            else:
                cur_grad = dfp[ cur_point[0], cur_point[1] ] - dfp[ cur_point[0], cur_point[1]-1 ]
            ##ie
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ] / dfp_max
            else:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_dfp_norm < threshold:
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
            if computed_df[ cur_point[0] , cur_point[1] ] == 1:
                check_df_around[ i, j ] = dfp[ act_i, act_j ]
            else:
                # If it hasn't then run the orbit
                check_around_o = orbit.Orbit(vxvv=[R*apu.kpc, 
                                      vR_range[act_i]*apu.km/apu.s, 
                                      vT_range[act_j]*apu.km/apu.s, 
                                      z*apu.kpc,
                                      0*apu.km/apu.s,
                                      phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if _compute_unperturbed:
                    df0[ act_i , act_j ] = df_evaluator(check_around_o)                      
                
                # Now integrate
                check_around_o.integrate(times,pot)
                dfp[ act_i , act_j ] = df_evaluator(check_around_o(times[-1]))
                check_df_around[i,j] = dfp[ act_i , act_j ]
                computed_df[ act_i , act_j ] = 1.
            ##ie
        
            # Now compute the gradient between this point and one point back in 'j'
            cur_grad = dfp[ cur_point[0], cur_point[1] ] - dfp[ cur_point[0], cur_point[1]+1 ]
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ] / dfp_max
            else:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_dfp_norm < threshold:
                decrease_j_done = True
            ##fi
            
            # Update the current position
            cur_point = [ cur_point[0], cur_point[1]-1 ]
        ##wh
        
        # Did this row contribute any values of the perturbed DF above the 
        # threshold?
        if threshold_norm == True:
            cur_i_dfp_norm = dfp[ level_start[0], : ] / dfp_max
        else:
            cur_i_dfp_norm = dfp[ level_start[0], : ]
        ##ie
        
        # Check if this row contributed anything 
        if len( np.where( cur_i_dfp_norm > threshold) ) == 0:
            increase_i_done == True
        ##fi
        
        # Now update to a new value of i. Find the maximum value the DF in this 
        # row and go upwards there
        where_i_max = np.argmax( dfp[ level_start[0], : ] )
        level_start = [ level_start[0]+1, where_i_max ]
        cur_point = [ level_start[0], level_start[1] ]
        
    ##wh
    
    # Now go down in 'i'. Set the level to be 1 below the max
    level_start = [ maximum_location[0]-1, maximum_location[1] ]
    cur_point[ level_start[0], level_start[1] ]
    
    decrease_i_done = False
    while decrease_i_done == False:
        
        increase_j_done = False
        first_of_i = True
        # Check increasing 'j'
        while increase_j_done == False:
            
            # Compute the DF at the current point
            # Check if the DF at this point has already been computed
            if computed_df[ cur_point[0] , cur_point[1] ] == 1:
                check_df_around[ i, j ] = dfp[ act_i, act_j ]
            else:
                # If it hasn't then run the orbit
                check_around_o = orbit.Orbit(vxvv=[R*apu.kpc, 
                                      vR_range[act_i]*apu.km/apu.s, 
                                      vT_range[act_j]*apu.km/apu.s, 
                                      z*apu.kpc,
                                      0*apu.km/apu.s,
                                      phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if _compute_unperturbed:
                    df0[ act_i , act_j ] = df_evaluator(check_around_o)                      
                
                # Now integrate
                check_around_o.integrate(times,pot)
                dfp[ act_i , act_j ] = df_evaluator(check_around_o(times[-1]))
                check_df_around[i,j] = dfp[ act_i , act_j ]
                computed_df[ act_i , act_j ] = 1.
            ##ie
            
            # Now compute the gradient between this point and one point back in 'j'
            if first_of_i:
                cur_grad = 0
                first_of_i = False
            else:
                cur_grad = dfp[ cur_point[0], cur_point[1] ] - dfp[ cur_point[0], cur_point[1]-1 ]
            ##ie
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ] / dfp_max
            else:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_dfp_norm < threshold:
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
            if computed_df[ cur_point[0] , cur_point[1] ] == 1:
                check_df_around[ i, j ] = dfp[ act_i, act_j ]
            else:
                # If it hasn't then run the orbit
                check_around_o = orbit.Orbit(vxvv=[R*apu.kpc, 
                                      vR_range[act_i]*apu.km/apu.s, 
                                      vT_range[act_j]*apu.km/apu.s, 
                                      z*apu.kpc,
                                      0*apu.km/apu.s,
                                      phi*apu.radian])
                
                # Should the unperturbed DF be computed? 
                if _compute_unperturbed:
                    df0[ act_i , act_j ] = df_evaluator(check_around_o)                      
                
                # Now integrate
                check_around_o.integrate(times,pot)
                dfp[ act_i , act_j ] = df_evaluator(check_around_o(times[-1]))
                check_df_around[i,j] = dfp[ act_i , act_j ]
                computed_df[ act_i , act_j ] = 1.
            ##ie
        
            # Now compute the gradient between this point and one point back in 'j'
            cur_grad = dfp[ cur_point[0], cur_point[1] ] - dfp[ cur_point[0], cur_point[1]+1 ]
            
            # If we are decreasing in value of the DF and we are below the
            # threshold then move on
            if threshold_norm == True:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ] / dfp_max
            else:
                cur_dfp_norm = dfp[ cur_point[0], cur_point[1] ]
            ##ie
            
            if cur_grad < 0 and cur_dfp_norm < threshold:
                decrease_j_done = True
            ##fi
            
            # Update the current position
            cur_point = [ cur_point[0], cur_point[1]-1 ]
        ##wh
        
        # Did this row contribute any values of the perturbed DF above the 
        # threshold?
        if threshold_norm == True:
            cur_i_dfp_norm = dfp[ level_start[0], : ] / dfp_max
        else:
            cur_i_dfp_norm = dfp[ level_start[0], : ]
        ##ie
        
        # Check if this row contributed anything 
        if len( np.where( cur_i_dfp_norm > threshold) ) == 0:
            decrease_i_done == True
        ##fi
        
        # Now update to a new value of i. Find the maximum value the DF in this 
        # row and go upwards there
        where_i_max = np.argmax( dfp[ level_start[0], : ] )
        level_start = [ level_start[0]-1, where_i_max ]
        cur_point = [ level_start[0], level_start[1] ]
        
    ##wh
    
    # All done!
    if _compute_unperturbed == True:
        return dfp, df0
    else:
        return dfp
        
#def

# ----------------------------------------------------------------------------
