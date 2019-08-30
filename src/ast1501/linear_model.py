# ----------------------------------------------------------------------------
#
# TITLE - linear_model.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS: 
# LinearModel
# LinearModel2
# LinearModelSolution
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''Functions to calculate aspects of the linear model
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

## Scipy
from scipy.stats import binned_statistic

## AST 1501
from . import df as ast1501_df

# ----------------------------------------------------------------------------

class LinearModel():
    '''LinearModel:
    
    A class encompassing the sinusoidal-linear model of the velocity
    fluctuations in the disk as a function of radius.
    
    There are a few ways to instantiate the linear model. These are triggered 
    by the instantiate_method keyword:
    1 - Expects gc_R, gc_phi, gc_vT, and gc_vR are Gaia data. Will bootstrap 
        on them. Requires R_lims, R_bin_size, phi_lims, phi_bin_size
    2 - Expects df_filename is the path to a DF file that can be read and 
        turned into bootstrap samples
    3 - Expects that bs_sample_vR and bs_sample_vT are already provided
    
    Keywords which are always required are: phib_lims and phib_bin_size
    
    In order to pickle this class you must import like:
    -> from ast1501.linear_model import LinearModel where you are pickling
    and unpickling
    
    ** Co-variance not supported b/c large matrix inversion issues **
    
    Args:
        Required: 
            instantiate_method (int) - How to instantiate the class, see above.
        
        Instantiation method 1:
            gc_{R,phi,vT,vR} (float array) - Gaia star properties
        
        Instantiation method 2:
            df_filename (string) - Name of the filename of DF velocity field
        
        Instantiation method 3:
            bs_sample_{vR,vT} (6-array) - Bootstrap samples
        
        Limits & Running:
            {R,phi,phib}_lims (2-array) - lower and upper limits
            {R,phi,phib}_bin_size (float) - bin size
            use_velocities (2-array) - Array of velocities to use in 
                determination of model properties
        
        Prior: 
            prior_var_arr (4-array) - Array of variances for vT offset, 
                vT amplitudes, vR offset, and vR amplitudes
            vT_prior_type (string) - Type of prior to use for vT: 'df' for 
                distribution function inferred, 'rotcurve' for the rotation curve
                calculated from MWPotential2014
            vT_prior_path (string) - Path to DF file containing vT data for 
                prior. Required if vT_prior_type='df' [None]
            vT_prior_offset (float) - Arbitrary offset applied to the vT prior
        
        Options:
            phiB (float) - Force the value of phiB to a fixed value [None]
            n_iterate (int) - Number of times to iterate the noise model [5]
            n_bs (100) - Number of bootstrap samples
            force_yint_vR (bool) - Force radial velocities to have a constant 
                y-intercept (b value) [True]
            force_yint_vR_value (float) - Value to force the radial velocity 
                y-intercept [0]
    '''
    
    def __init__(self,
                  instantiate_method, 
                   # Method 1 instantiation
                   gc_R=None, 
                   gc_phi=None, 
                   gc_vT=None, 
                   gc_vR=None,
                   # Method 2 instantiation
                   df_filename=None, 
                   # Method 3 instantiation
                   bs_sample_vR=None, 
                   bs_sample_vT=None, 
                   # Limits & Running
                   R_lims=None, 
                   R_bin_size=None, 
                   phi_lims=None,
                   phi_bin_size=None, 
                   phib_lims=None, 
                   phib_bin_size=None, 
                   use_velocities=['vR','vT'], 
                   # Prior
                   prior_var_arr=[25,np.inf,25,np.inf], 
                   vT_prior_type='df',
                   vT_prior_path=None,
                   vT_prior_offset=0,
                   # Options
                   phiB=None, 
                   n_iterate=5, 
                   n_bs=1000, 
                   force_yint_vR=True, 
                   force_yint_vR_value=0
                ):
        
        # First, get the bootstrap samples, one of three ways: calculate from 
        # Gaia data, load from file, or manually specify.
        if instantiate_method == 1:
            
            # Assert that we have the necessary keywords
            assert (gc_R is not None) and (gc_phi is not None) and \
                   (gc_vT is not None) and (gc_vR is not None),\
            'gc_R, gc_phi, gc_vT, gc_vR all expected for instantiate_method=1 but not provided'
            assert (R_lims is not None) and (R_bin_size is not None) and \
                   (phi_lims is not None) and (phi_bin_size is not None),\
            'R_lims, R_bin_size, phi_lims, phi_bin_size all expected for instantiate_method=1 but not provided'
            
            # Create bin center arrays
            R_bin_cents = np.arange( R_lims[0], R_lims[1], R_bin_size )
            R_bin_cents += R_bin_size/2
            phi_bin_cents = np.arange( phi_lims[0], phi_lims[1], phi_bin_size ) 
            phi_bin_cents += phi_bin_size/2
            
            # Assign properties
            self.gc_R = gc_R
            self.gc_phi = gc_phi
            self.gc_vR = gc_vR
            self.gc_vT = gc_vT
            self.R_bin_cents = R_bin_cents
            self.n_R_bins = len(R_bin_cents)
            self.R_bin_size = R_bin_size
            self.phi_bin_cents = phi_bin_cents
            self.phi_bin_size = phi_bin_size
            self.n_bs=n_bs
            
            # Create the bootstrap sample
            bs_sample_vR, bs_sample_vT = self._make_bootstrap_samples()
            self.bs_sample_vR = bs_sample_vR
            self.bs_sample_vT = bs_sample_vT
        
        # Load the bootstrap-like sample from a DF velocity field file
        elif instantiate_method == 2:
            
            # Assert that we have the necessary keywords
            assert df_filename!=None,\
            'df_filename expected for instantiate_method=2 but not provided' 
            
            # Read the data, assume data is of the form: 
            # [R,phi,x,y,vR,vR_disp,vT,vT_disp]
            data = np.load(df_filename).T.astype(float)
            R,phi,_,_,vR,_,vT,_ = data
            self.R_bin_cents = np.sort(np.unique(R))
            self.n_R_bins = len(np.sort(np.unique(R)))
            self.R_bin_size = np.diff( np.sort( np.unique(R) ) )[0]
            
            # Make the bootstrap samples
            bs_sample_vR, bs_sample_vT = \
                self._make_data_like_bootstrap_samples(R, phi, vR, vT)
            self.bs_sample_vR = bs_sample_vR
            self.bs_sample_vT = bs_sample_vT
        
        # Load pre-constructed bootstrap arrays
        elif instantiate_method == 3:
            
            # Assert that we have the necessary keywords
            assert bs_sample_vR!=None and bs_sample_vT!=None,\
            'bs_sample_vR and bs_sample_vT expected for instantiate_method=3 but not provided'
            
            self.bs_sample_vR = bs_sample_vR
            self.bs_sample_vT = bs_sample_vT
            
            # Assume that the bootstrap samples contain the relevent
            # information about the radial profile
            self.n_R_bins = len(bs_sample_vR)
            R_bin_cents = np.zeros(self.n_R_bins)
            for i in range(self.n_R_bins):
                R_bin_cents[i] = bs_sample_vR[i][0]
            ###i
            # Assume the R_bins are uniformly distributed
            self.R_bin_cents = R_bin_cents
            self.n_R_bins = len(R_bin_cents)
            self.R_bin_size = np.diff(R_bin_cents)[0]
            
        else:
        
            raise Exception('instantiate_method must be 1, 2, or 3')
        
        ##ie
        
        # Always needs to have phiB information
        assert phib_lims!=None and phib_bin_size!=None,\
            'phib_lims and phib_bin_size required parameters'
        phib_bin_cents = np.arange( phib_lims[0], phib_lims[1], phib_bin_size )
        phib_bin_cents += phib_bin_size/2
        self.phib_bin_size = phib_bin_size
        self.phib_bin_cents = phib_bin_cents
        self.n_phib_bins = len(phib_bin_cents)
        
        # Always need to have prior covariance information
        assert len(prior_var_arr)==4,'prior_var_arr must have 4 elements'
        var_b_vT, var_m_vT, var_b_vR, var_m_vR = prior_var_arr
        self.var_b_vT = var_b_vT
        self.var_m_vT = var_m_vT
        self.var_b_vR = var_b_vR
        self.var_m_vR = var_m_vR
        
        # Initialize vT prior information based on type
        self.vT_prior_type = vT_prior_type
        self.vT_prior_offset = vT_prior_offset
        if vT_prior_type == 'rotcurve':
            self.rotcurve_prior_R = np.arange(5,15,0.01)
            self.rotcurve_prior_vT = potential.vcirc(potential.MWPotential2014, 
                R=self.rotcurve_prior_R)
        if vT_prior_type == 'df':
            assert vT_prior_path is not None,"vT_prior_type is 'df' but vT_prior_path not supplied"
            self.vT_prior_path = vT_prior_path
            df_prior_R, df_prior_vT = np.load(vT_prior_path)
            self.df_prior_R = df_prior_R
            self.df_prior_vT = df_prior_vT
        ##fi
        
        # Figure out if we're going to use one velocity or two.
        # Needs to be a list or a tuple
        assert type(use_velocities)==list or type(use_velocities)==tuple
        if 'vT' in use_velocities:
            self.use_vT = True
        else:
            self.use_vT = False
        ##ie
        if 'vR' in use_velocities:
            self.use_vR = True
        else:
            self.use_vR = False
        ##ie
        if self.use_vR==False and self.use_vT==False:
            raise Exception('Cannot use neither vR or vT')
        ##fi
        
        # Figure out how many velocities will be used.
        if self.use_vR==False or self.use_vT==False:
            self.n_velocities=1
        else:
            self.n_velocities=2
        ##ie
        
        # Declare single velocity properties based on whether we will use 
        # vR or vT for ease of use throughout the class
        if self.n_velocities==1:
            if self.use_vR:
                self.bs_sample_1v = bs_sample_vR
                self.trig_fn_1v = np.sin
                self.vel_1v = 'vR'
            if self.use_vT:
                self.bs_sample_1v = bs_sample_vT
                self.trig_fn_1v = np.cos
                self.vel_1v = 'vT'
            ##fi
        ##fi
        
        # Declare whether vR will be forced to be a constant value
        self.force_yint_vR=force_yint_vR
        self.force_yint_vR_value=force_yint_vR_value
            
        # Declare the number of times to iterate the noise model
        self.n_iterate=n_iterate
        
        # Declare phiB. If it was None, then it will be calculated during 
        # each step. If it was not none then we will force it to be the same
        self.phiB = phiB
        if self.phiB!=None:
            self.force_phiB=True
        else:
            self.force_phiB=False
        ##ie
            
        # Now run the linear model
        results_arr = self.run_iterating_linear_model(update_results=True)
        
        # Set a few properties
        self.results_arr = results_arr
        latest_results = results_arr[-1]
        if self.n_velocities == 2:
            self.b_vR = latest_results[6][:,1]
            self.m_vR = latest_results[7][:,1]
            self.b_vT = latest_results[6][:,0]
            self.m_vT = latest_results[7][:,0]
            self.b_err_vR = latest_results[8][:,1]
            self.m_err_vR = latest_results[9][:,1]
            self.b_err_vT = latest_results[8][:,0]
            self.m_err_vT = latest_results[9][:,0]
        if self.n_velocities == 1:
            if self.use_vR:
                self.b_vR = latest_results[3]
                self.m_vR = latest_results[4]
                self.b_err_vR = latest_results[5]
                self.m_err_vR = latest_results[6]
            if self.use_vT:
                self.b_vT = latest_results[3]
                self.m_vT = latest_results[4]
                self.b_err_vT = latest_results[5]
                self.m_err_vT = latest_results[6]
            ##fi
        ##fi
        if self.force_phiB == False:
            if self.n_velocities == 2:
                self.phiB = latest_results[5]
            if self.n_velocities == 1:
                self.phiB = latest_results[2]
            ##fi
        ##fi
    #def
    
    # Define getters and setters:
    def get_bs_samples(self):
        '''get_bs_samples:
        
        Return the bootstrap samples.
        '''
        return self.bs_sample_vR, self.bs_sample_vT 
    #def
    
    def get_bs_sample_positions(self):
        '''get_bs_sample_positions:
        
        Return the physical locations where the bootstrap samples were obtained
        for the LinearModel in a pair of single 1-D arrays.
        
        Returns:
            R_posns (float array) - Array of R locations for each point
            phi_posns (float array) - Array of phi locations for each point
        '''
        R_posns = np.array([])
        phi_posns = np.array([])
        # First find each unique radial position
        for i in range(self.n_R_bins):
            these_phi_posns = self.bs_sample_vR[i][3]
            these_R_posns = np.ones_like(these_phi_posns)*self.bs_sample_vR[i][0]
            R_posns = np.append(R_posns,these_R_posns)
            phi_posns = np.append(phi_posns,these_phi_posns)
        ###i
        return R_posns,phi_posns
    #def
    
    def get_bs_velocities(self):
        '''get_bs_velocities:
        
        Return the bootstrap velocities and errors in single 1-D arrays
        
        Returns:
            vR (float array) - Array of vR for each point
            vR_err (float array) - Array of vR errors for each point
            vT (float array) - Array of vT for each point
            vT_err (float array) - Array of vT errors for each point
        '''
        vR = np.array([])
        vR_err = np.array([])
        vT = np.array([])
        vT_err = np.array([])
        # First find each unique radial position
        for i in range(self.n_R_bins):
            these_vR = self.bs_sample_vR[i][1]
            these_vR_err = self.bs_sample_vR[i][2]
            these_vT = self.bs_sample_vT[i][1]
            these_vT_err = self.bs_sample_vT[i][2]
            
            vR = np.append(vR,these_vR)
            vR_err = np.append(vR_err,these_vR_err)
            vT = np.append(vT,these_vT)
            vT_err = np.append(vT_err,these_vT_err)
        ###i    
        return vR,vR_err,vT,vT_err
    #def
    
    def get_bs_phi_errors(self):
        '''get_bs_phi_errs:
        
        Return the bootstrap phi errors
        
        Returns:
            phi_err (float array) - Array of phi errors for each point
        '''
        phi_err = np.array([])
        # First find each unique radial position
        for i in range(self.n_R_bins):
            these_phi_err = self.bs_sample_vR[i][4]
            phi_err = np.append(phi_err,these_phi_err)
        ###i    
        return phi_err
    #def
    
    def _make_bootstrap_samples(self):
        '''make_bootstrap_sample:
        
        Make the bootstrap samples for vR and vT from the data which has 
        already been declared
        
        Args:
            None
            
        Returns:
            bs_sample_vT (N-array) - Array of the vT bootstrap sample results for a 
                single radius. It contains:
                - R_bin_cent (float) - Radial bin center
                - vT (float array) - vT as a function of phi
                - vT_error (float array) - vT uncertainty as a function of phi
                - phi_bin_phi (float array) - phi bin centers
                - phi_bin_phi_err (float array) - phi bin center uncertainty
            bs_sample_vR (N-array) - same but for vR
            
        '''
        
        bs_samples_vT = []
        bs_samples_vR = []
        
        for i in range( len(self.R_bin_cents) ):
            
            # Make the bootstrap sample
            bs_samp = self._bootstrap_in_phi( self.R_bin_cents[i] )                            
            bs_samp_vR = [bs_samp[0], bs_samp[1], bs_samp[2], 
                          bs_samp[5], bs_samp[6]]
            bs_samp_vT = [bs_samp[0], bs_samp[3], bs_samp[4], 
                          bs_samp[5], bs_samp[6]]
            
            bs_samples_vR.append(bs_samp_vR)
            bs_samples_vT.append(bs_samp_vT)
        ###i
        
        return bs_samples_vR, bs_samples_vT
    #def
    
    def _bootstrap_in_phi(self,R_bin_cent):
        '''_bootstrap_in_phi:
        
        Perform a bootstrap determination of the average velocity in phi bins. 
        Returns an array which can be unpacked wherever it is needed.
        
        Args:
            R_bin_cent (float) The radial bin center for this sample
            
        Returns:
            bs_sample (8-array) - Array of the bootstrap sample results for a 
                single radius. It contains:
                - R_bin_cent (float) - Radial bin center
                - R_bin_size (float) - Radial bin size
                - vR (float array) - vR as a function of phi
                - vR_error (float array) - vR uncertainty as a function of phi
                - vT (float array) - vT as a function of phi
                - vT_error (float array) - vT uncertainty as a function of phi
                - phi_bin_phi (float array) - phi bin centers
                - phi_bin_phi_err (float array) - phi bin center uncertainty
        '''
        
        n_phi_bins = len(self.phi_bin_cents)
        
        # Find all the points within this radial bin
        stars_in_R_bin = np.where( ( self.gc_R < R_bin_cent + self.R_bin_size/2 ) & 
                                   ( self.gc_R > R_bin_cent - self.R_bin_size/2 ) )[0]
        n_stars_in_R_bin = len(stars_in_R_bin)
        gc_R_in_R_bin = self.gc_R[stars_in_R_bin]
        gc_phi_in_R_bin = self.gc_phi[stars_in_R_bin]
        gc_vR_in_R_bin = self.gc_vR[stars_in_R_bin]
        gc_vT_in_R_bin = self.gc_vT[stars_in_R_bin]
        
        phi_bin_vR = np.array([])
        phi_bin_vR_err = np.array([])
        phi_bin_vT = np.array([])
        phi_bin_vT_err = np.array([])
        phi_bin_phi = np.array([])
        phi_bin_phi_err = np.array([])

        # Loop over phi bins
        for j in range(n_phi_bins):

            # Find all the points within this phi bin
            stars_in_phi_bin = np.where( ( gc_phi_in_R_bin < self.phi_bin_cents[j] + self.phi_bin_size/2 ) &
                                         ( gc_phi_in_R_bin > self.phi_bin_cents[j] - self.phi_bin_size/2 ) )[0]
            n_stars_in_phi_bin = len(stars_in_phi_bin)
            gc_R_in_phi_bin = gc_R_in_R_bin[stars_in_phi_bin]
            gc_phi_in_phi_bin = gc_phi_in_R_bin[stars_in_phi_bin]
            
            gc_vR_in_phi_bin = gc_vR_in_R_bin[stars_in_phi_bin]
            gc_vT_in_phi_bin = gc_vT_in_R_bin[stars_in_phi_bin]
            
            # If we have more than a certain number of stars then BS
            bs_vR_avg_samps = np.array([])
            bs_vT_avg_samps = np.array([])
            bs_phi_avg_samps = np.array([])
            
            if n_stars_in_phi_bin > 10:

                # Loop over BS samples
                for k in range(self.n_bs):
                    sample = np.random.randint(0,n_stars_in_phi_bin,n_stars_in_phi_bin)
                    bs_vR_avg_samps = np.append( bs_vR_avg_samps, np.average(gc_vR_in_phi_bin[sample]) )
                    bs_vT_avg_samps = np.append( bs_vT_avg_samps, np.average(gc_vT_in_phi_bin[sample]) )
                    bs_phi_avg_samps = np.append( bs_phi_avg_samps, np.average(gc_phi_in_phi_bin[sample]) )
                ###k
        
                # Append the mean to the list of measurements
                phi_bin_vR = np.append( phi_bin_vR, np.mean( bs_vR_avg_samps ) )
                phi_bin_vR_err = np.append( phi_bin_vR_err, np.std( bs_vR_avg_samps ) )
                phi_bin_vT = np.append( phi_bin_vT, np.mean( bs_vT_avg_samps ) )
                phi_bin_vT_err = np.append( phi_bin_vT_err, np.std( bs_vT_avg_samps ) )
                phi_bin_phi = np.append( phi_bin_phi, np.mean( bs_phi_avg_samps ) )
                phi_bin_phi_err = np.append( phi_bin_phi_err, np.std( bs_phi_avg_samps ) )
                
            ##fi
        ###j
        
        return [R_bin_cent, phi_bin_vR, phi_bin_vR_err, phi_bin_vT,
                phi_bin_vT_err, phi_bin_phi, phi_bin_phi_err]
    #def
    
    def _make_data_like_bootstrap_samples(self, R, phi, vR, vT, phi_err=0.01, 
                                            vT_err=0.5, vR_err=0.5):
        '''make_data_like_bootstrap_samples:
        
        Take a series of R/phi data and velocities and knit it into a form that 
        looks like the bootstrap sample arrays, and which is appropriate for using 
        in the linear model functions.
        
        Args:
            R
            phi (float array) - Phi positions
            vT (float array) - Tangential velocities
            vR (float array) - Radial velocities
            phi_err (float array) - Phi position errors [None]
            vT_err (float array) - Tangential velocity errors [None]
            vR_err (float array) - Radial velocity errors [None]
            
        Returns:
            bs_samples_vT (N-array) - Array of the vR bootstrap sample results 
                for a single radius. It contains:
                - R_bin_cent (float) - Radial bin center
                - R_bin_size (float) - Radial bin size
                - vT (float array) - vT as a function of phi
                - vT_error (float array) - vT uncertainty as a function of phi
                - phi_bin_phi (float array) - phi bin centers
                - phi_bin_phi_err (float array) - phi bin center uncertainty
            bs_samples_vR (N-array) - same but for vT
        '''
        
        # Declare the arrays which hold the bootstrap samples
        bs_samples_vT = []
        bs_samples_vR = []
        
        # Loop over each unique radius and extract all the data for that bin
        for i in range(self.n_R_bins):
            
            this_R_bin_cent = self.R_bin_cents[i]
            where_unique_R = np.where(R==this_R_bin_cent)[0]
            this_phi = phi[where_unique_R]
            this_vT = vT[where_unique_R]
            this_vR = vR[where_unique_R]
            
            # Now generate the error arrays. Start of as small numbers but can 
            # be filled. Handles arrays of errors, but also constants.
            if type(phi_err) == float or type(phi_err) == int:
                this_phi_err = np.ones_like(this_phi)*phi_err
            else:
                this_phi_err = phi_err[where_unique_R]
            ##ie
            
            if type(vT_err) == float or type(vT_err) == int:
                this_vT_err = np.ones_like(this_phi)*vT_err
            else:
                this_vT_err = vT_err[where_unique_R]
            ##ie
            
            if type(vR_err) == float or type(vR_err) == int:
                this_vR_err = np.ones_like(this_phi)*vR_err
            else:
                this_vR_err = vR_err[where_unique_R]
            ##ie
            
            # Make the velocity sample
            vT_sample = [this_R_bin_cent, this_vT, this_vT_err, 
                         this_phi, this_phi_err]
            vR_sample = [this_R_bin_cent, this_vR, this_vR_err,
                         this_phi, this_phi_err]
            
            bs_samples_vT.append(vT_sample)
            bs_samples_vR.append(vR_sample)
        ###i
        
        return bs_samples_vR, bs_samples_vT
    #def
    
    def run_iterating_linear_model(self, force_yint_vR=None, 
                                    force_yint_vR_value=None, n_iterate=None,
                                    update_results=False):
        '''run_iterating_linear_model:
        
        Function to iterate over the loop where the linear model is evaluated,
        each time the noise model is updated to apply higher constant noise 
        offsets to radial bins which don't match the overall trends as well.
        
        Args:
            force_yint_vR (bool) - Force the y-intercept to be a constant 
                value [None]
            force_yint_vR_value (float) - Constant value to force the 
                y-intercept [None]
            n_iterate (int) - Number of times to iterate the model. Overwritten 
                by the class n_iterate property if set [None]
            update_results (bool) - Set the results property to be the results
                from this function evaluation [False]
        
        Returns:
        '''
        
        # Set defaults. Warning these changes aren't actually propogated
        # through to the actual functions that evaluate the linear model yet!
        if n_iterate==None:
            n_iterate=self.n_iterate
        ##fi
        if force_yint_vR==None:
            force_yint_vR=self.force_yint_vR
        ##fi
        if force_yint_vR_value==None:
            force_yint_vR_value=self.force_yint_vR_value
        ##fi
        
        # Empty arrays to hold results and errors
        results_arr = []
        
        # Set the size of the extra variance arrays
        if self.n_velocities == 2:
            extra_variance = np.zeros((self.n_R_bins,2))
        else:
            extra_variance = np.zeros(self.n_R_bins)
        ##ie
        
        # Loop over all the times we are supposed to iterate the model
        for i in range( n_iterate ):
            
            # Determine if we are using one velocity or two velocities
            if self.n_velocities==2:
                likelihood_vT, likelihood_vR, prod_likelihood_vT,\
                prod_likelihood_vR, prod_likelihood_both, phib_max_likelihood,\
                bs, ms, bs_err, ms_err, variance_model_data\
                = self._iterate_noise_model_2_velocities(extra_variance)
            else:
                likelihood, prod_likelihood, phib_max_likelihood, bs, ms,\
                bs_err, ms_err, variance_model_data \
                = self._iterate_noise_model_1_velocity(extra_variance)
            ##ie
            
            # Update the variance
            extra_variance = variance_model_data
            
            # Construct the output array
            if self.n_velocities==2:
                output_results = [likelihood_vT, likelihood_vR, 
                    prod_likelihood_vT, prod_likelihood_vR, 
                    prod_likelihood_both, phib_max_likelihood, 
                    bs, ms, bs_err, ms_err, variance_model_data]
            if self.n_velocities==1:
                output_results = [likelihood, prod_likelihood, 
                    phib_max_likelihood, bs, ms, bs_err, ms_err, 
                    variance_model_data]
            ##fi
            results_arr.append(output_results)
        ###i
        
        if update_results:
            self.results_arr=results_arr
        ##fi
        
        return results_arr
    #def
    
    def _generate_gaussian_prior_m_b(self, prior_style, R_bin_cent):
        '''_generate_gaussian_prior_m_b:
        
        Make the parameters of the prior: the mean sample and the inverse variance. 
        For both m and b.
        
        Args:
            prior_style (string) - Either 'vT' or 'vR'
            R_bin_cent (float) - Radius in kpc for the circular velocity curve
        
        Returns:
            X0 (2x1 element array) - Mean of the gaussian prior
            SIGMA_inv (2x2 array) - Inverse of the variance array
        '''
        # Generate the prior
        if prior_style == 'vT':
            if self.vT_prior_type=='df':
                which_bin = np.argmin( np.abs( R_bin_cent-self.df_prior_R ) )
                b0 = self.df_prior_vT[which_bin]
            elif self.vT_prior_type=='rotcurve':
                b0 = potential.vcirc(potential.MWPotential2014, R_bin_cent/8.0)*220.0
            b0 += self.vT_prior_offset
            m0 = 0
            X0 = np.zeros((2,1)) # Make a column vector
            X0[0,0] = b0
            X0[1,0] = m0
            SIGMA_inv = np.array([[1/self.var_b_vT,0],[0,1/self.var_m_vT]])
        elif prior_style == 'vR':
            b0 = 0
            m0 = 0
            X0 = np.zeros((2,1)) # Make a column vector
            X0[0,0] = b0
            X0[1,0] = m0
            SIGMA_inv = np.array([[1/self.var_b_vR,0],[0,1/self.var_m_vR]])
        ##ie
        return X0, SIGMA_inv
    #def
    
    def _calculate_phib_likelihood(self, bs_sample, prior_style, trig_function,
                                   extra_variance=0, force_yint=False, 
                                   force_yint_value=0):
        '''_calculate_phib_likelihood:
        
        Calculate the likelihood as a function of the given phib's for a single 
        radial bin and a series of phi bins.
        
        Args:
            bs_sample (6-array) - 6 element array of bootstrap properties
            prior_style (string) - Style of prior, either 'vR' or 'vT'
            trig_function (func) - Probably either np.cos or np.sin
            extra_variance (float) - Should an extra variance term be added to this 
                radial bin?
        
        Returns:
            Likelihood (float array) - Likelihood as a function of phib
        '''
        
        # Unpack the bootstrap sample
        R_bin_cent, phi_bin_v, phi_bin_v_err, phi_bin_phi, _ = bs_sample
        
        # Make the prior
        X0, SIGMA_inv = self._generate_gaussian_prior_m_b(prior_style, 
                                                          R_bin_cent)

        # Now make the vectors
        n_good_phi_bins = len(phi_bin_v)
        Y = np.zeros((n_good_phi_bins,1))
        C = np.zeros((n_good_phi_bins,n_good_phi_bins))
        Y[:,0] = phi_bin_v
        
        # If the y-intercept is forced then apply that to the Y vector and 
        # shrink the prior vectors
        if force_yint:
            Y[:,0] = Y[:,0] - force_yint_value
            X0 = np.delete(X0,0)
            SIGMA_inv = np.delete(np.delete(SIGMA_inv,0,axis=0),0,axis=1)
            # Reshape axis to maintain original number of dimensions
            X0 = X0.reshape(X0.shape[0],1)
            SIGMA_inv = SIGMA_inv.reshape(SIGMA_inv.shape[0],SIGMA_inv.shape[0])
        ##fi
        
        # Fill the co-variance matrix
        for j in range(n_good_phi_bins):
            C[j,j] = phi_bin_v_err[j]**2 + extra_variance
        ###j
        C_inv = np.linalg.inv(C)

        # Now loop over all possible values of phi B, making the vector 
        # A for each and calculating the likelihood.
        n_phib_bins = len(self.phib_bin_cents)
        likelihood = np.zeros( n_phib_bins )
        
        for j in range(n_phib_bins):    
            if force_yint:
                A = np.ones((n_good_phi_bins,1))
                A[:,0] = trig_function( 2*( phi_bin_phi - self.phib_bin_cents[j] ) )
            else:
                A = np.ones((n_good_phi_bins,2))
                A[:,1] = trig_function( 2*( phi_bin_phi - self.phib_bin_cents[j] ) )
            ##ie
            
            # Now compute the vectors which form the solution
            V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
            V = np.linalg.inv( V_inv )
            W = np.matmul( V , 
                np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
            U = np.linalg.multi_dot( [Y.T,C_inv,Y] ) + np.linalg.multi_dot( [X0.T,SIGMA_inv,X0] ) - np.linalg.multi_dot( [W.T,V_inv,W] )
            likelihood[j] = 0.5*( np.log(V.diagonal()).sum() - np.log(C.diagonal()).sum() ) - U[0,0]/2 
        ###j

        return likelihood

    #def
    
    def _calculate_best_fit_m_b(self, R_bin_cent, phiB, bs_sample, 
                                prior_style, trig_function, 
                                force_yint=False, force_yint_value=0,
                                extra_variance=0):
        '''_calculate_best_fit_m_b:
        
        Calculate the best-fitting m and b values for the linear model
        
        Args:
            bs_sample (6-array) - 6 element array of bootstrap properties
            prior_style (string) - Style of prior, either 'vR' or 'vT'
            trig_function (func) - Either np.cos or np.sin
            force_yint (bool) - Should the y-intercept be forced to be a 
                constant value? [False]
            force_yint_value (float) - What should the y-intercept be forced 
                to be?
            extra_variance (float) - Should an extra variance term be added to 
                this radial bin? [0]
        
        Returns:
            X (2-array) - Best-fitting m and b
            SIG_X (2-array) - Uncertainty in the best-fit
        '''
        
        # Unpack the bootstrap sample
        R_bin_cent, phi_bin_v, phi_bin_v_err, phi_bin_phi, _ = bs_sample
        
        # Make the prior
        X0, SIGMA_inv = self._generate_gaussian_prior_m_b(prior_style, 
                                                          R_bin_cent)

        # Now make the vectors
        n_good_phi_bins = len(phi_bin_v)
        Y = np.zeros((n_good_phi_bins,1))
        C = np.zeros((n_good_phi_bins,n_good_phi_bins))
        Y[:,0] = phi_bin_v
        
        # If forcing y-intercept to be zero then apply the offset to the 
        # Y values
        if force_yint:
            Y[:,0] = Y[:,0] - force_yint_value
            X0 = np.delete(X0,0)
            SIGMA_inv = np.delete(np.delete(SIGMA_inv,0,axis=0),0,axis=1)
            # Reshape axis to maintain original number of dimensions
            X0 = X0.reshape(X0.shape[0],1)
            SIGMA_inv = SIGMA_inv.reshape(SIGMA_inv.shape[0],SIGMA_inv.shape[0])
        ##fi
        
        # Fill the co-variance matrix
        for j in range(n_good_phi_bins):
            C[j,j] = phi_bin_v_err[j]**2 + extra_variance
        ###j
        C_inv = np.linalg.inv(C)
        
        # Check if the y intercept is forced to be 0
        if force_yint:
            A = np.ones((n_good_phi_bins,1))
            A[:,0] = trig_function( 2*( phi_bin_phi - phiB ) )
        else:
            A = np.ones((n_good_phi_bins,2))
            A[:,1] = trig_function( 2*( phi_bin_phi - phiB ) )
        ##ie
        
        V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
        V = np.linalg.inv( V_inv )
        W = np.matmul( V , np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
        return W, V
    #def
    
    def _calculate_variance_data_model(self, bs_sample, m, b, phib, 
                                       trig_function):
        '''calculate_variance_data_model:
        
        Calculate the variance of the differences between the best-fitting model and the 
        data.
        
        Args:
            bs_sample 
            m (float) - Best-fitting m
            b (float) - Best-fitting b
            phib (float) - Best-fitting phib
            trig_function (function) - Either np.sin or np.cos
            
        Returns:
            variance (float) - variance of the difference between the model and data
        
        '''
        
        # Unpack the bootstrap sample
        _, phi_bin_v, _, phi_bin_phi, _ = bs_sample
        
        # Calculate the model at the locations where there is data
        model_v = b + m*trig_function(2*(phi_bin_phi-phib))
        
        # Calculate the standard deviation of the differences between model and data
        sd_data_model = np.std(model_v-phi_bin_v)
        
        return np.square(sd_data_model)
        
    #def
    
    def _iterate_noise_model_1_velocity(self, extra_variance):
        '''_iterate_noise_model_1_velocity:
        
        Iterate over the calculation of the best-fitting linear model using 
        a single velocity only, adding an empirically derived variance to 
        radial bins which do not match the overall trends particularly well.
        
        Args:
            extra_variance (n_R x 2 array) - Extra variance for vR and vT as a 
                function of radius.
        
        Returns:
        
        '''
        
        # First determine the best-fitting phiB
        
        # Likelihood matrix for radial bins and phiB values
        likelihood = np.ones( ( self.n_R_bins, self.n_phib_bins ) )
        
        # If vR is being used then assign the keyword to force the y-intercept 
        # to be a constant value
        force_yint = False
        force_yint_value = 0
        if self.use_vR:
            force_yint = self.force_yint_vR
            force_yint_value = self.force_yint_vR_value
        ##fi
        
        for j in range( self.n_R_bins ):
            
            # Calculate the log likelihood of the chosen velocity
            likelihood[j,:] = \
            self._calculate_phib_likelihood(self.bs_sample_1v[j], self.vel_1v,
                                            self.trig_fn_1v,
                                            extra_variance=extra_variance[j], 
                                            force_yint=force_yint, 
                                            force_yint_value=force_yint_value)
            
        # Marginalize over all radial bins
        prod_likelihood = np.sum(likelihood, axis=0)
    
        # Determine the best-fitting phib
        phib_max_likelihood_arg = np.argmax( prod_likelihood )
        phib_max_likelihood = self.phib_bin_cents[phib_max_likelihood_arg]
        
        # If we are forcing phiB then assign it
        if self.force_phiB:
            use_phiB = self.phiB
        else:
            use_phiB = phib_max_likelihood
        ##ie

        ms = np.zeros( self.n_R_bins )
        bs = np.zeros( self.n_R_bins )
        ms_err = np.zeros( self.n_R_bins )
        bs_err = np.zeros( self.n_R_bins )
        variance_model_data = np.zeros( self.n_R_bins )
            
        # Loop over radial bins, calculate the best-fitting m and b
        for j in range( self.n_R_bins ):

            # Now determine the best-fitting m and b
            X, SIG_X = \
            self._calculate_best_fit_m_b(self.R_bin_cents[j], use_phiB,
                                         self.bs_sample_1v[j], self.vel_1v, 
                                         self.trig_fn_1v, 
                                         extra_variance=extra_variance[j], 
                                         force_yint=force_yint,
                                         force_yint_value=force_yint_value )
            if force_yint:
                bs[j] = force_yint_value
                bs_err[j] = 0
                ms[j] = X[0]
                ms_err[j] = np.sqrt( SIG_X[0,0] )
            else:                 
                bs[j] = X[0]
                bs_err[j] = np.sqrt( SIG_X[0,0] )
                ms[j] = X[1]
                ms_err[j] = np.sqrt( SIG_X[1,1] )
            ##ie
            
            # Now calculate the standard deviation of the difference between the data and the model
            variance_model_data[j] = \
            self._calculate_variance_data_model(self.bs_sample_1v[j], ms[j], 
                                                bs[j], use_phiB, self.trig_fn_1v)
        ###j 
        
        return likelihood, prod_likelihood, phib_max_likelihood, \
               bs, ms, bs_err, ms_err, variance_model_data
        
    def _iterate_noise_model_2_velocities(self, extra_variance):
        '''_iterate_noise_model_2_velocities:
        
        Iterate over the calculation of the best-fitting linear model using 
        both vR and vT velocities, adding an empirically derived variance to 
        radial bins which do not match the overall trends particularly well.
        
        Args:
            extra_variance (n_R x 2 array) - Extra variance for vR and vT as a 
                function of radius.
            
        Returns:
            
        '''
            
        # Make an array to store the log likelihoods
        likelihood_vT = np.ones( ( self.n_R_bins, self.n_phib_bins ) )
        likelihood_vR = np.ones( ( self.n_R_bins, self.n_phib_bins ) )

        # Loop over the radial bins and calculate the likelihood as a function of 
        # phiB for both tangential and radial velocities. 
        for j in range( self.n_R_bins ):

            # Calculate the log likelihood of the tangential and radial
            # velocities as functions of phiB
            likelihood_vT[j,:] = \
            self._calculate_phib_likelihood(self.bs_sample_vT[j], 'vT', np.cos, 
                                            extra_variance=extra_variance[j,0] )
            
            likelihood_vR[j,:] = \
            self._calculate_phib_likelihood(self.bs_sample_vR[j], 'vR', np.sin, 
                                            extra_variance=extra_variance[j,1], 
                                            force_yint=self.force_yint_vR,
                                            force_yint_value=self.force_yint_vR_value)
        ###j

        # Marginalize over all radii
        prod_likelihood_vT = np.sum(likelihood_vT, axis=0)
        prod_likelihood_vR = np.sum(likelihood_vR, axis=0)
        prod_likelihood_both = prod_likelihood_vR + prod_likelihood_vT

        # Determine the best-fitting phib
        phib_max_likelihood_arg = np.argmax( prod_likelihood_both )
        phib_max_likelihood = self.phib_bin_cents[phib_max_likelihood_arg]
        
        # If we are forcing phiB then assign it
        if self.force_phiB:
            use_phiB = self.phiB
        else:
            use_phiB = phib_max_likelihood
        ##ie

        ms = np.zeros( (self.n_R_bins,2) )
        bs = np.zeros( (self.n_R_bins,2) )
        ms_err = np.zeros( (self.n_R_bins,2) )
        bs_err = np.zeros( (self.n_R_bins,2) )
        variance_model_data = np.zeros((self.n_R_bins,2))

        # Loop over radial bins, calculate the best-fitting m and b
        for j in range( self.n_R_bins ):

            # Now determine the best-fitting m and b
            X_vT, SIG_X_vT = \
            self._calculate_best_fit_m_b(self.R_bin_cents[j], use_phiB,
                                         self.bs_sample_vT[j], 'vT', np.cos, 
                                         extra_variance=extra_variance[j,0] )
            X_vR, SIG_X_vR = \
            self._calculate_best_fit_m_b(self.R_bin_cents[j], use_phiB,
                                         self.bs_sample_vR[j], 'vR', np.sin, 
                                         extra_variance=extra_variance[j,1],
                                         force_yint=self.force_yint_vR, 
                                         force_yint_value=self.force_yint_vR_value )
            if self.force_yint_vR:
                bs[j,1] = self.force_yint_vR_value
                bs_err[j,1] = 0
                ms[j,1] = X_vR[0]
                ms_err[j,1] = np.sqrt( SIG_X_vR[0,0] )
            else:
                bs[j,1] = X_vR[0]
                bs_err[j,1] = np.sqrt( SIG_X_vR[0,0] )
                ms[j,1] = X_vR[1]
                ms_err[j,1] = np.sqrt( SIG_X_vR[1,1] )
            ##ie
            
            bs[j,0] = X_vT[0]
            bs_err[j,0] = np.sqrt( SIG_X_vT[0,0] )
            ms[j,0] = X_vT[1]
            ms_err[j,0] = np.sqrt( SIG_X_vT[1,1] )
            
            # Now calculate the standard deviation of the difference between the data and the model
            variance_model_data[j,0] = \
            self._calculate_variance_data_model(self.bs_sample_vT[j], ms[j,0], 
                                                bs[j,0], use_phiB, np.cos)
            variance_model_data[j,1] = \
            self._calculate_variance_data_model(self.bs_sample_vR[j], ms[j,1], 
                                                bs[j,1], use_phiB, np.sin)
        ###j 
        
        return likelihood_vT, likelihood_vR, prod_likelihood_vT, \
               prod_likelihood_vR, prod_likelihood_both, phib_max_likelihood, \
               bs, ms, bs_err, ms_err, variance_model_data
    #def
    
    # Define some plotting routines
    def plot_velocity_known_m_b_phi(self, velocity_type, fig=None, axs=None, 
                                    phi_lim=[-np.pi/2,np.pi/2], 
                                    plot_best_fit=True):
        '''plot_velocity_known_m_b_phi
        
        Plot the velocities as a function of radius for a bootstrap sample.
        overplot the best-fitting solution from the linear model. Note that 
        fig and axs need to be commensurate with the number of radial bins 
        being plotted.
        
        Args:
            velocity_type (string) - Either 'vR' or 'vT': which one to use
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created [None]
            phi_lim (2-array) - The limits of phi to plot
            plot_best_fit (bool) - Include the best fitting m=2 profile
        '''
        
        # Select the right bootstrap sample
        if velocity_type == 'vR':
            bs_samp = self.bs_sample_vR
        if velocity_type == 'vT':
            bs_samp = self.bs_sample_vT
        ##fi
        
        if fig is None and axs is None:
            fig = plt.figure( figsize=(5,self.n_R_bins*2) )
            axs = fig.subplots( nrows=self.n_R_bins, ncols=1 )
        ##fi
        
        # Loop over all radii
        for i in range( self.n_R_bins ):
            
            # Unpack the velocity sample for this radius
            bin_R_cent = bs_samp[i][0]
            bin_v = bs_samp[i][1]
            bin_v_err = bs_samp[i][2]
            bin_phi = bs_samp[i][3]
            
            # Plot
            axs[i].errorbar( bin_phi, bin_v, yerr=bin_v_err, fmt='o', 
                ecolor='Black', marker='o', markerfacecolor='None', 
                markeredgecolor='Black', markersize=5)
        
            # Plot the best-fitting amplitude
            if plot_best_fit:
                trig_phis = np.linspace(phi_lim[0], phi_lim[1], num=100)
                if velocity_type == 'vR':
                    axs[i].plot( trig_phis, 
                        self.b_vR[i]+self.m_vR[i]*np.sin(2*(trig_phis-self.phiB)))
                if velocity_type == 'vT':
                    axs[i].plot( trig_phis, 
                        self.b_vT[i]+self.m_vT[i]*np.cos(2*(trig_phis-self.phiB)))
                ##fi
            ##fi
        
            # Add fiducials: bar, 0 line or tangential velocity curve
            axs[i].axvline( 25*(np.pi/180), linestyle='dotted', linewidth=1.0, 
                color='Red', label='Bar Angle')
            X0, _ = self._generate_gaussian_prior_m_b(velocity_type, bin_R_cent)
            b0 = X0[0,0]
            axs[i].axhline( b0, linestyle='dashed', color='Black', 
                linewidth=1.0, label=r'$V_{circ}$')
                
            # Annotate
            axs[i].annotate( r'$R_{cen}=$'+str(bin_R_cent)+' kpc', 
                xy=(0.05,0.8), xycoords='axes fraction' )
                
            # Set limits
            axs[i].set_xlim( phi_lim[0], phi_lim[1] )
            
            # Set the labels
            if velocity_type == 'vR':
                axs[i].set_ylabel(r'$v_{R}$ [km/s]')
            if velocity_type == 'vT':
                axs[i].set_ylabel(r'$v_{T}$ [km/s]')
            ##fi
            if i == 0:
                axs[i].set_xlabel(r'$\phi$')
            ##fi
        
        axs[0].legend(loc='best')
        fig.subplots_adjust(hspace=0)
        
        return fig, axs
    #def
    
    # Define some plotting routines
    def plot_vRvT_known_m_b_phi(self, fig=None, axs=None,
                                    phi_lim=[-np.pi/2,np.pi/2], label_fs=12,):
        '''plot_vTvR_known_m_b_phi
        
        Plot both velocities as a function of radius for a bootstrap sample.
        overplot the best-fitting solution from the linear model. Note that 
        fig and axs need to be commensurate with the number of radial bins 
        being plotted.
        
        Args:
            velocity_type (string) - Either 'vR' or 'vT': which one to use
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created [None]
            phi_lim (2-array) - The limits of phi to plot
        '''
        
        if fig is None and axs is None:
            fig = plt.figure( figsize=(10,self.n_R_bins*2) )
            axs = fig.subplots( nrows=self.n_R_bins, ncols=2 )
        ##fi
        
        # Loop over all radii
        for i in range( self.n_R_bins ):
            
            # Unpack the velocity sample for this radius
            bin_R_cent = self.bs_sample_vR[i][0]
            bin_vR = self.bs_sample_vR[i][1]
            bin_vR_err = self.bs_sample_vR[i][2]
            bin_phi = self.bs_sample_vR[i][3]
            bin_vT = self.bs_sample_vT[i][1]
            bin_vT_err = self.bs_sample_vT[i][2]
            
            # Plot
            axs[i,0].errorbar( bin_phi, bin_vR, yerr=bin_vR_err, fmt='o', 
                ecolor='Black', marker='o', markerfacecolor='None', 
                markeredgecolor='Black', markersize=5)
            axs[i,1].errorbar( bin_phi, bin_vT, yerr=bin_vT_err, fmt='o', 
                ecolor='Black', marker='o', markerfacecolor='None', 
                markeredgecolor='Black', markersize=5)
        
            # Plot the best-fitting amplitude
            trig_phis = np.linspace(phi_lim[0], phi_lim[1], num=100)
            axs[i,0].plot( trig_phis, 
                self.b_vR[i]+self.m_vR[i]*np.sin(2*(trig_phis-self.phiB)))
            axs[i,1].plot( trig_phis, 
                self.b_vT[i]+self.m_vT[i]*np.cos(2*(trig_phis-self.phiB)))
        
            # Add fiducials: bar, 0 line or tangential velocity curve
            axs[i,0].axvline( 25*(np.pi/180), linestyle='dotted', linewidth=1.0, 
                color='Red')
            axs[i,1].axvline( 25*(np.pi/180), linestyle='dotted', linewidth=1.0, 
                color='Red')
            axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=1.0 )
            X0, _ = self._generate_gaussian_prior_m_b('vT', bin_R_cent)
            b0 = X0[0,0]
            axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=1.0 )
            axs[i,1].axhline( b0, linestyle='dashed', color='Black', linewidth=1.0 )
                
            # Annotate
            axs[i,0].annotate( r'$R_{cen}=$'+str(bin_R_cent)+' kpc', 
                xy=(0.05,0.8), xycoords='axes fraction' )
                
            # Set limits
            axs[i,0].set_xlim( phi_lim[0], phi_lim[1] )
            axs[i,1].set_xlim( phi_lim[0], phi_lim[1] )
            
            # Set the labels
            axs[i,0].set_ylabel(r'$v_{R}$ [km/s]', fontsize=label_fs)
            axs[i,1].set_ylabel(r'$v_{T}$ [km/s]', fontsize=label_fs)
            ##fi
            if i == self.n_R_bins-1:
                axs[i,0].set_xlabel(r'$\phi$', fontsize=label_fs)
                axs[i,1].set_ylabel(r'$\phi$', fontsize=label_fs)
            ##fi
        
        fig.subplots_adjust(hspace=0)
        
        return fig, axs
    #def
    
    def plot_velocity_m_r(self, fig=None, axs=None, plot_type='errorbar', 
                            plot_kws={}, label_fs=12, which_velocity=None):
        '''plot_velocity_m_r:
        
        Plot both m and b as a functions of radius for either vR or vT
        
        Args:
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created. Should be 2 length [None]
            plot_type (string) - What type of plot to make? Either 'errorbar'm 
                'scatter' or 'plot' ['errorbar']
            plot_kws (dict) - Dictionary of properties to be passed to the 
                plotting functions. Make sure commensurate with plot_type! [{}]
        '''
        
        # Format for this figure is single column 2 axis
        if fig is None or axs is None:
            fig = plt.figure( figsize=(12,4) )
            axs = fig.subplots( nrows=1, ncols=2 )
        ##fi
        
        if which_velocity == None:
            if self.n_velocities==2:
                raise Exception('No velocity specified and 2 used.')
            ##fi
            which_velocity = self.vel_1v
        ##fi
        
        if which_velocity == 'vR':
            use_b = self.b_vR
            use_m = self.m_vR
            use_b_err = self.b_err_vR
            use_m_err = self.m_err_vR
        if which_velocity == 'vT':
            use_b = self.b_vT
            use_m = self.m_vT
            use_b_err = self.b_err_vT
            use_m_err = self.m_err_vT
        ##fi
        
        # Plot
        if plot_type == 'errorbar':
            axs[0].errorbar( self.R_bin_cents, use_b, yerr=use_b_err, 
                **plot_kws)
            axs[1].errorbar( self.R_bin_cents, use_m, yerr=use_m_err, 
                **plot_kws)
        elif plot_type == 'plot':
            axs[0].plot( self.R_bin_cents, use_b, **plot_kws)
            axs[1].plot( self.R_bin_cents, use_m, **plot_kws)
        elif plot_type == 'scatter':
            axs[0].scatter( self.R_bin_cents, use_b, **plot_kws)
            axs[1].scatter( self.R_bin_cents, use_m, **plot_kws)
        ##fi
        
        # Labels and limits
        if which_velocity == 'vR':
            axs[0].set_ylabel(r'$b_{R}$ [km/s]', fontsize=label_fs)
            axs[1].set_ylabel(r'$m_{R}$ [km/s]', fontsize=label_fs)
        if which_velocity == 'vT':
            axs[0].set_ylabel(r'$b_{T}$ [km/s]', fontsize=label_fs)
            axs[1].set_ylabel(r'$m_{T}$ [km/s]', fontsize=label_fs)
        ##fi
        axs[0].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[1].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[0].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[1].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        
        # Add fiducials
        axs[1].axhline(0, linestyle='dashed', color='Black')
        if which_velocity == 'vR':
            axs[0].axhline(0, linestyle='dashed', color='Black')
        if which_velocity == 'vT':
            if self.vT_prior_type=='df':    
                axs[0].plot( self.df_prior_R, self.df_prior_vT,
                    linestyle='dashed', color='Black' )
            if self.vT_prior_type=='rotcurve':
                axs[0].plot( self.rotcurve_prior_R, self.rotcurve_prior_vT,
                    linestyle='dashed', color='Black')
            ##fi
        ##fi
        
        return fig, axs
    
    def plot_vRvT_m_r(self, fig=None, axs=None, plot_type='errorbar', 
                        plot_kws={}, label_fs=12):
        '''plot_m_b:
        
        Plot both m and b as functions of radius for vT and vR profiles
        
        Args:
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created. Should be 2x2 [None]
            plot_type (string) - What type of plot to make? Either 'errorbar'
                'scatter' or 'plot' ['errorbar']
            plot_kws (dict) - Dictionary of properties to be passed to the 
                plotting functions. Make sure commensurate with plot_type! [{}]
        '''
        
        # Format for this figure is 2x2
        if fig is None or axs is None:
            fig = plt.figure( figsize=(12,6) ) 
            axs = fig.subplots( nrows=2, ncols=2 )
        ##fi
        
        # Plot
        if plot_type == 'errorbar':
            axs[0,0].errorbar( self.R_bin_cents, self.b_vR, yerr=self.b_err_vR, 
                **plot_kws)
            axs[0,1].errorbar( self.R_bin_cents, self.m_vR, yerr=self.m_err_vR, 
                **plot_kws)
            axs[1,0].errorbar( self.R_bin_cents, self.b_vT, yerr=self.b_err_vT, 
                **plot_kws)
            axs[1,1].errorbar( self.R_bin_cents, self.m_vT, yerr=self.m_err_vT, 
                **plot_kws)
        elif plot_type == 'plot':
            axs[0,0].plot( self.R_bin_cents, self.b_vR, **plot_kws)
            axs[0,1].plot( self.R_bin_cents, self.m_vR, **plot_kws)
            axs[1,0].plot( self.R_bin_cents, self.b_vT, **plot_kws)
            axs[1,1].plot( self.R_bin_cents, self.m_vT, **plot_kws)
        elif plot_type == 'scatter':
            axs[0,0].scatter( self.R_bin_cents, self.b_vR, **plot_kws)
            axs[0,1].scatter( self.R_bin_cents, self.m_vR, **plot_kws)
            axs[1,0].scatter( self.R_bin_cents, self.b_vT, **plot_kws)
            axs[1,1].scatter( self.R_bin_cents, self.m_vT, **plot_kws)
        
        # Labels and limits
        axs[0,0].set_ylabel(r'$b_{R}$ [km/s]', fontsize=label_fs)
        axs[0,1].set_ylabel(r'$m_{R}$ [km/s]', fontsize=label_fs)
        axs[1,0].set_ylabel(r'$b_{T}$ [km/s]', fontsize=label_fs)
        axs[1,1].set_ylabel(r'$m_{T}$ [km/s]', fontsize=label_fs)
        axs[0,0].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[0,1].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[1,0].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[1,1].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[0,0].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[0,1].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[1,0].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[1,1].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        
        # Add fiducials
        axs[0,0].axhline(0, linestyle='dashed', color='Black')
        axs[0,1].axhline(0, linestyle='dashed', color='Black')
        axs[1,1].axhline(0, linestyle='dashed', color='Black')
        
        # Prior
        if self.vT_prior_type=='df':    
            axs[1,0].plot( self.df_prior_R, self.df_prior_vT,
                linestyle='dashed', color='Black' )
        if self.vT_prior_type=='rotcurve':
            axs[1,0].plot( self.rotcurve_prior_R, self.rotcurve_prior_vT,
                linestyle='dashed', color='Black')
        ##fi
        
        return fig, axs
        
    def sample_bootstrap(self, sample_phi=False):
        '''sample_bootstrap:
        
        Sample from the bootstrap assuming its own errors. Return two 
        bootstrap samples of the same shape (as vR and vT) with the 
        same errors
        
        Args:
            sample_phi (bool) - Draw samples from the phi distribution [False]
            
        '''
        
        # Make empty arrays
        bs_mc_samp_vR = []
        bs_mc_samp_vT = []
        
        # Loop over all radii
        for i in range( self.n_R_bins ):
            
            n_phi_bins = len( self.bs_sample_vR[i][1] )
            
            # Loop over each phi bin
            this_R_vR_samp = np.zeros(n_phi_bins)
            this_R_vT_samp = np.zeros(n_phi_bins)
            this_R_phi_samp = np.zeros(n_phi_bins)
            
            for j in range( n_phi_bins ):
                
                # Generate the samples for this phi bin
                vR_samp = np.random.normal(loc=self.bs_sample_vR[i][1][j], 
                    scale=self.bs_sample_vR[i][2][j], size=1).astype(float)[0]
                this_R_vR_samp[j] = vR_samp
                vT_samp = np.random.normal(loc=self.bs_sample_vT[i][1][j], 
                    scale=self.bs_sample_vT[i][2][j], size=1).astype(float)[0]
                this_R_vT_samp[j] = vT_samp
                if sample_phi:
                    phi_samp = np.random.normal(loc=self.bs_sample_vR[i][3][j],
                        scale=self.bs_sample_vR[4][j], size=1).astype(float)[0]
                else:
                    phi_samp = self.bs_sample_vR[i][3][j]
                ##ie
                this_R_phi_samp[j] = phi_samp
            ###j
            
            # Pack results
            this_R = self.bs_sample_vR[i][0]
            this_R_vR_err = self.bs_sample_vR[i][2]
            this_R_vT_err = self.bs_sample_vT[i][2]
            this_R_phi_err = self.bs_sample_vR[i][4]
            bs_mc_samp_vR.append( [this_R,this_R_vR_samp,this_R_vR_err,
                                   this_R_phi_samp,this_R_phi_err] )
            bs_mc_samp_vT.append( [this_R,this_R_vT_samp,this_R_vT_err,
                                   this_R_phi_samp,this_R_phi_err] )
            
        ###i
        return bs_mc_samp_vR, bs_mc_samp_vT
    #def
#cls

class LinearModel2():
    '''LinearModel2:
    
    A class encompassing the sinusoidal-linear model of the velocity
    fluctuations in the disk as a function of radius. This 2nd edition of the 
    linear model class will simul-fit all radial bins at the same time. 
    Ideally this will increase speed, but will also be useful for being able to
    fit for a constant value of b across all radial bins.
    
    Could probably be merged with LinearModel but the manner in which 
    phiB, the final solution, and the iterative noise model are calculated all
    differ substantially. Likely best solution is to make LinearModel and 
    LinearModel2 both children of a base LinearModel class.
        
    There are a few ways to instantiate the linear model. These are triggered 
    by the instantiate_method keyword:
    1 - Expects gc_R, gc_phi, gc_vT, and gc_vR are Gaia data. Will bootstrap 
        on them. Requires R_lims, R_bin_size, phi_lims, phi_bin_size
    2 - Expects df_filename is the path to a DF file that can be read and 
        turned into bootstrap samples
    3 - Expects that bs_sample_vR and bs_sample_vT are already provided
    
    Keywords which are always required are: phib_lims and phib_bin_size
    
    In order to pickle this class you must import like:
    -> from ast1501.linear_model import LinearModel where you are pickling
    and unpickling
    
    ** Co-variance not supported b/c large matrix inversion issues **
    
    Args:
        Required:
            instantiate_method (int) - How to instantiate the class, see above.
            
        Instantiation method 1:
            gc_{R,phi,vT,vR} (float array) - Gaia star properties
        
        Instantiation method 2:
            df_filename (string) - Name of the filename of DF velocity field
            
        Instantiation method 3:
            bs_sample_{vR,vT} (6-array) - Bootstrap samples
        
        Limits & Running:
            {R,phi,phib}_lims (2-array) - lower and upper limits
            {R,phi,phib}_bin_size (float) - bin size
            use_velocities (2-array) - Array of velocities to use in 
                determination of model properties
        
        Prior: 
            prior_var_arr (4-array) - Array of variances for vT offset, 
                vT amplitudes, vR offset and vR amplitudes
            vT_prior_type (string) - Type of prior to use for vT: 'df' for 
                distribution function inferred, 'rotcurve' for the rotation curve
                calculated from MWPotential2014
            vT_prior_path (string) - Path to DF file containing vT data for 
                prior. Required if vT_prior_type='df' [None]
            vT_prior_offset (float) - Arbitrary offset applied to the vT prior
        
        Options:
            phiB (float) - Force the value of phiB to a fixed value [None]
            n_iterate (int) - Number of times to iterate the noise model [5]
            n_bs (100) - Number of bootstrap samples
            force_yint_vR (bool) - Force radial velocities to have a constant 
                y-intercept (b value) [True]
            force_yint_vR_value (float) - Value to force the radial velocity 
                y-intercept [0]
            fit_yint_vR_constant (bool) - Allow the linear model to fit a 
                constant offset to all vR data points [False]
    '''
    
    def __init__(self, 
                  instantiate_method, 
                   # Method 1 instantiation
                   gc_R=None, 
                   gc_phi=None, 
                   gc_vT=None, 
                   gc_vR=None,
                   # Method 2 instantiation
                   df_filename=None, 
                   # Method 3 instantiation
                   bs_sample_vR=None, 
                   bs_sample_vT=None, 
                   # Limits & Running
                   R_lims=None, 
                   R_bin_size=None, 
                   phi_lims=None,
                   phi_bin_size=None, 
                   phib_lims=None, 
                   phib_bin_size=None, 
                   use_velocities=['vR','vT'], 
                   # Prior
                   prior_var_arr=[25,np.inf,25,np.inf], 
                   vT_prior_type='df',
                   vT_prior_path=None,
                   vT_prior_offset=0,
                   # Options
                   phiB=None,
                   n_iterate=5, 
                   n_bs=1000, 
                   force_yint_vR=True, 
                   force_yint_vR_value=0, 
                   fit_yint_vR_constant=False, 
                  ):
        
        # First, get the bootstrap samples, one of three ways: calculate from 
        # Gaia data, load from file, or manually specify.
        if instantiate_method == 1:
            
            # Assert that we have the necessary keywords
            assert (gc_R is not None) and (gc_phi is not None) and \
                   (gc_vT is not None) and (gc_vR is not None),\
            'gc_R, gc_phi, gc_vT, gc_vR all expected for instantiate_method=1 but not provided'
            assert (R_lims is not None) and (R_bin_size is not None) and \
                   (phi_lims is not None) and (phi_bin_size is not None),\
            'R_lims, R_bin_size, phi_lims, phi_bin_size all expected for instantiate_method=1 but not provided'
            
            # Create bin center arrays
            R_bin_cents = np.arange( R_lims[0], R_lims[1], R_bin_size )
            R_bin_cents += R_bin_size/2
            phi_bin_cents = np.arange( phi_lims[0], phi_lims[1], phi_bin_size ) 
            phi_bin_cents += phi_bin_size/2
            
            # Assign properties
            self.gc_R = gc_R
            self.gc_phi = gc_phi
            self.gc_vR = gc_vR
            self.gc_vT = gc_vT
            self.R_bin_cents = R_bin_cents
            self.n_R_bins = len(R_bin_cents)
            self.R_bin_size = R_bin_size
            self.phi_bin_cents = phi_bin_cents
            self.phi_bin_size = phi_bin_size
            self.n_bs=n_bs
            
            # Get the bootstrap sample
            bs_sample_vR, bs_sample_vT = self._make_bootstrap_samples()
            self.bs_sample_vR = bs_sample_vR
            self.bs_sample_vT = bs_sample_vT
        
        # Load the bootstrap-like sample from a DF velocity field file
        elif instantiate_method == 2:
            
            # Assert that we have the necessary keywords
            assert df_filename!=None,\
            'df_filename expected for instantiate_method=2 but not provided' 
            
            # Read the data, assume data is of the form: 
            # [R,phi,x,y,vR,vR_disp,vT,vT_disp]
            data = np.load(df_filename).T.astype(float)
            R,phi,_,_,vR,_,vT,_ = data
            self.R_bin_cents = np.sort(np.unique(R))
            self.n_R_bins = len(np.sort(np.unique(R)))
            self.R_bin_size = np.diff( np.sort( np.unique(R) ) )[0]
            
            # Make the bootstrap samples
            bs_sample_vR, bs_sample_vT = \
                self._make_data_like_bootstrap_samples(R, phi, vR, vT)
            self.bs_sample_vR = bs_sample_vR
            self.bs_sample_vT = bs_sample_vT
        
        # Load pre-constructed
        elif instantiate_method == 3:
            
            # Assert that we have the necessary keywords
            assert bs_sample_vR!=None and bs_sample_vT!=None,\
            'bs_sample_vR and bs_sample_vT expected for instantiate_method=3 but not provided'
            
            self.bs_sample_vR = bs_sample_vR
            self.bs_sample_vT = bs_sample_vT
            
            # Assume that the bootstrap samples contain the relevent
            # information about the radial profile
            self.n_R_bins = len(bs_sample_vR)
            R_bin_cents = np.zeros(self.n_R_bins)
            for i in range(self.n_R_bins):
                R_bin_cents[i] = bs_sample_vR[i][0]
            ###i
            # Assume the R_bins are uniformly distributed
            self.R_bin_cents = R_bin_cents
            self.n_R_bins = len(R_bin_cents)
            self.R_bin_size = np.diff(R_bin_cents)[0]
            
        else:
        
            raise Exception('instantiate_method must be 1, 2, or 3')
        
        ##ie
        
        # Always needs to have phiB information
        assert phib_lims!=None and phib_bin_size!=None,\
            'phib_lims and phib_bin_size required parameters'
        phib_bin_cents = np.arange( phib_lims[0], phib_lims[1], phib_bin_size )
        phib_bin_cents += phib_bin_size/2
        self.phib_bin_size = phib_bin_size
        self.phib_bin_cents = phib_bin_cents
        self.n_phib_bins = len(phib_bin_cents)
        
        # Always need to have prior information
        assert len(prior_var_arr)==4,'prior_var_arr must have 4 elements'
        var_b_vT, var_m_vT, var_b_vR, var_m_vR = prior_var_arr
        self.var_b_vT = var_b_vT
        self.var_m_vT = var_m_vT
        self.var_b_vR = var_b_vR
        self.var_m_vR = var_m_vR
        
        # Initialize vT prior information based on type
        self.vT_prior_type = vT_prior_type
        self.vT_prior_offset = vT_prior_offset
        if vT_prior_type == 'rotcurve':
            self.rotcurve_prior_R = np.arange(5,15,0.01)
            self.rotcurve_prior_vT = potential.vcirc(potential.MWPotential2014, 
                R=self.rotcurve_prior_R)
        if vT_prior_type == 'df':
            assert vT_prior_path is not None,"vT_prior_type is 'df' but vT_prior_path not supplied"
            self.vT_prior_path = vT_prior_path
            df_prior_R, df_prior_vT = np.load(vT_prior_path)
            self.df_prior_R = df_prior_R
            self.df_prior_vT = df_prior_vT
        ##fi
        
        # Figure out if we're going to use one velocity or two.
        # Needs to be a list or a tuple
        assert type(use_velocities)==list or type(use_velocities)==tuple
        if 'vT' in use_velocities:
            self.use_vT = True
        else:
            self.use_vT = False
        ##ie
        if 'vR' in use_velocities:
            self.use_vR = True
        else:
            self.use_vR = False
        ##ie
        if self.use_vR==False and self.use_vT==False:
            raise Exception('Cannot use neither vR or vT')
        ##fi
        
        # Figure out how many velocities will be used.
        if self.use_vR==False or self.use_vT==False:
            self.n_velocities=1
        else:
            self.n_velocities=2
        ##ie
        
        # Declare single velocity properties based on whether we will use 
        # vR or vT for ease of use throughout the class
        if self.n_velocities==1:
            if self.use_vR:
                self.bs_sample_1v = bs_sample_vR
                self.trig_fn_1v = np.sin
                self.vel_1v = 'vR'
            if self.use_vT:
                self.bs_sample_1v = bs_sample_vT
                self.trig_fn_1v = np.cos
                self.vel_1v = 'vT'
            ##fi
        ##fi
        
        # Assign vR and vT trigonometric functions
        self.trig_function_vT = np.cos
        self.trig_function_vR = np.sin
        
        # Declare whether vR will be forced to be a constant value
        self.force_yint_vR=force_yint_vR
        self.force_yint_vR_value=force_yint_vR_value
        
        # Declare whether vR will be fit simultaneously across all bins.
        # If it is then don't force a constant value
        self.fit_yint_vR_constant=fit_yint_vR_constant
        if self.fit_yint_vR_constant and self.force_yint_vR:
            print('Warning, fitting y-intercept constant across all radial bins, ignoring forced y-intercept value...')
            self.force_yint_vR = False
        ##fis
            
        # Declare the number of times to iterate the noise model
        self.n_iterate=n_iterate
        
        # Declare phiB. If it was None, then it will be calculated during 
        # each step. If it was not none then we will force it to be the same
        self.phiB = phiB
        if self.phiB!=None:
            self.force_phiB=True
        else:
            self.force_phiB=False
        ##ie
        
        # Figure out how many total data points there are for easy array
        # construction later
        total_data_vR = 0
        total_data_vT = 0
        for i in range( self.n_R_bins ):
            temp_bs_sample_vR = self.bs_sample_vR[i]
            temp_bs_sample_vT = self.bs_sample_vT[i]
            total_data_vR += len( temp_bs_sample_vR[1] )
            total_data_vT += len( temp_bs_sample_vT[1] )
        ##fi
        assert total_data_vR == total_data_vT,\
            'Total number of vR & vT data points does not match, error?'
        self.n_data_points = total_data_vR
        
        # Now run the linear model
        results_arr = self.run_iterating_linear_model(update_results=True)
        
        # Set a few properties
        self.results_arr = results_arr
        latest_results = results_arr[-1]
        if self.n_velocities == 2:
            self.b_vR = latest_results[6][:,1]
            self.m_vR = latest_results[7][:,1]
            self.b_vT = latest_results[6][:,0]
            self.m_vT = latest_results[7][:,0]
            self.b_err_vR = latest_results[8][:,1]
            self.m_err_vR = latest_results[9][:,1]
            self.b_err_vT = latest_results[8][:,0]
            self.m_err_vT = latest_results[9][:,0]
        if self.n_velocities == 1:
            if self.use_vR:
                self.b_vR = latest_results[3]
                self.m_vR = latest_results[4]
                self.b_err_vR = latest_results[5]
                self.m_err_vR = latest_results[6]
            if self.use_vT:
                self.b_vT = latest_results[3]
                self.m_vT = latest_results[4]
                self.b_err_vT = latest_results[5]
                self.m_err_vT = latest_results[6]
            ##fi
        ##fi
        if self.force_phiB == False:
            if self.n_velocities == 2:
                self.phiB = latest_results[5]
            if self.n_velocities == 1:
                self.phiB = latest_results[2]
            ##fi
        ##fi
    #def
    
    # Define getters and setters:
    def get_bs_samples(self):
        '''get_bs_samples:
        
        Return the bootstrap samples.
        '''
        return self.bs_sample_vR, self.bs_sample_vT 
    #def
    
    def get_bs_sample_positions(self):
        '''get_bs_sample_positions:
        
        Return the physical locations where the bootstrap samples were obtained
        for the LinearModel in a pair of single 1-D arrays.
        
        Returns:
            R_posns (float array) - Array of R locations for each point
            phi_posns (float array) - Array of phi locations for each point
        '''
        R_posns = np.array([])
        phi_posns = np.array([])
        # First find each unique radial position
        for i in range(self.n_R_bins):
            these_phi_posns = self.bs_sample_vR[i][3]
            these_R_posns = np.ones_like(these_phi_posns)*self.bs_sample_vR[i][0]
            R_posns = np.append(R_posns,these_R_posns)
            phi_posns = np.append(phi_posns,these_phi_posns)
        ###i
        return R_posns,phi_posns
    #def
    
    def get_bs_velocities(self):
        '''get_bs_velocities:
        
        Return the bootstrap velocities and errors in single 1-D arrays
        
        Returns:
            vR (float array) - Array of vR for each point
            vR_err (float array) - Array of vR errors for each point
            vT (float array) - Array of vT for each point
            vT_err (float array) - Array of vT errors for each point
        '''
        vR = np.array([])
        vR_err = np.array([])
        vT = np.array([])
        vT_err = np.array([])
        # First find each unique radial position
        for i in range(self.n_R_bins):
            these_vR = self.bs_sample_vR[i][1]
            these_vR_err = self.bs_sample_vR[i][2]
            these_vT = self.bs_sample_vT[i][1]
            these_vT_err = self.bs_sample_vT[i][2]
            
            vR = np.append(vR,these_vR)
            vR_err = np.append(vR_err,these_vR_err)
            vT = np.append(vT,these_vT)
            vT_err = np.append(vT_err,these_vT_err)
        ###i    
        return vR,vR_err,vT,vT_err
    #def
    
    def get_bs_phi_errors(self):
        '''get_bs_phi_errs:
        
        Return the bootstrap phi errors
        
        Returns:
            phi_err (float array) - Array of phi errors for each point
        '''
        phi_err = np.array([])
        # First find each unique radial position
        for i in range(self.n_R_bins):
            these_phi_err = self.bs_sample_vR[i][4]
            phi_err = np.append(phi_err,these_phi_err)
        ###i    
        return phi_err
    #def
    
    def _make_bootstrap_samples(self):
        '''make_bootstrap_sample:
        
        Make the bootstrap samples for vR and vT from the data which has 
        already been declared
        
        Args:
            None
            
        Returns:
            bs_sample_vT (N-array) - Array of the vT bootstrap sample results for a 
                single radius. It contains:
                - R_bin_cent (float) - Radial bin center
                - vT (float array) - vT as a function of phi
                - vT_error (float array) - vT uncertainty as a function of phi
                - phi_bin_phi (float array) - phi bin centers
                - phi_bin_phi_err (float array) - phi bin center uncertainty
            bs_sample_vR (N-array) - same but for vR
            
        '''
        
        bs_samples_vT = []
        bs_samples_vR = []
        
        for i in range( len(self.R_bin_cents) ):
            
            # Make the bootstrap sample
            bs_samp = self._bootstrap_in_phi( self.R_bin_cents[i] )                            
            bs_samp_vR = [bs_samp[0], bs_samp[1], bs_samp[2], 
                          bs_samp[5], bs_samp[6]]
            bs_samp_vT = [bs_samp[0], bs_samp[3], bs_samp[4], 
                          bs_samp[5], bs_samp[6]]
            
            bs_samples_vR.append(bs_samp_vR)
            bs_samples_vT.append(bs_samp_vT)
        ###i
        
        return bs_samples_vR, bs_samples_vT
    #def
    
    def _bootstrap_in_phi(self,R_bin_cent):
        '''_bootstrap_in_phi:
        
        Perform a bootstrap determination of the average velocity in phi bins. 
        Returns an array which can be unpacked wherever it is needed.
        
        Args:
            R_bin_cent (float) The radial bin center for this sample
            
        Returns:
            bs_sample (8-array) - Array of the bootstrap sample results for a 
                single radius. It contains:
                - R_bin_cent (float) - Radial bin center
                - R_bin_size (float) - Radial bin size
                - vR (float array) - vR as a function of phi
                - vR_error (float array) - vR uncertainty as a function of phi
                - vT (float array) - vT as a function of phi
                - vT_error (float array) - vT uncertainty as a function of phi
                - phi_bin_phi (float array) - phi bin centers
                - phi_bin_phi_err (float array) - phi bin center uncertainty
        '''
        
        n_phi_bins = len(self.phi_bin_cents)
        
        # Find all the points within this radial bin
        stars_in_R_bin = np.where( ( self.gc_R < R_bin_cent + self.R_bin_size/2 ) & 
                                   ( self.gc_R > R_bin_cent - self.R_bin_size/2 ) )[0]
        n_stars_in_R_bin = len(stars_in_R_bin)
        gc_R_in_R_bin = self.gc_R[stars_in_R_bin]
        gc_phi_in_R_bin = self.gc_phi[stars_in_R_bin]
        gc_vR_in_R_bin = self.gc_vR[stars_in_R_bin]
        gc_vT_in_R_bin = self.gc_vT[stars_in_R_bin]
        
        phi_bin_vR = np.array([])
        phi_bin_vR_err = np.array([])
        phi_bin_vT = np.array([])
        phi_bin_vT_err = np.array([])
        phi_bin_phi = np.array([])
        phi_bin_phi_err = np.array([])

        # Loop over phi bins
        for j in range(n_phi_bins):

            # Find all the points within this phi bin
            stars_in_phi_bin = np.where( ( gc_phi_in_R_bin < self.phi_bin_cents[j] + self.phi_bin_size/2 ) &
                                         ( gc_phi_in_R_bin > self.phi_bin_cents[j] - self.phi_bin_size/2 ) )[0]
            n_stars_in_phi_bin = len(stars_in_phi_bin)
            gc_R_in_phi_bin = gc_R_in_R_bin[stars_in_phi_bin]
            gc_phi_in_phi_bin = gc_phi_in_R_bin[stars_in_phi_bin]
            
            gc_vR_in_phi_bin = gc_vR_in_R_bin[stars_in_phi_bin]
            gc_vT_in_phi_bin = gc_vT_in_R_bin[stars_in_phi_bin]
            
            # If we have more than a certain number of stars then BS
            bs_vR_avg_samps = np.array([])
            bs_vT_avg_samps = np.array([])
            bs_phi_avg_samps = np.array([])
            
            if n_stars_in_phi_bin > 10:

                # Loop over BS samples
                for k in range(self.n_bs):
                    sample = np.random.randint(0,n_stars_in_phi_bin,n_stars_in_phi_bin)
                    bs_vR_avg_samps = np.append( bs_vR_avg_samps, np.average(gc_vR_in_phi_bin[sample]) )
                    bs_vT_avg_samps = np.append( bs_vT_avg_samps, np.average(gc_vT_in_phi_bin[sample]) )
                    bs_phi_avg_samps = np.append( bs_phi_avg_samps, np.average(gc_phi_in_phi_bin[sample]) )
                ###k
        
                # Append the mean to the list of measurements
                phi_bin_vR = np.append( phi_bin_vR, np.mean( bs_vR_avg_samps ) )
                phi_bin_vR_err = np.append( phi_bin_vR_err, np.std( bs_vR_avg_samps ) )
                phi_bin_vT = np.append( phi_bin_vT, np.mean( bs_vT_avg_samps ) )
                phi_bin_vT_err = np.append( phi_bin_vT_err, np.std( bs_vT_avg_samps ) )
                phi_bin_phi = np.append( phi_bin_phi, np.mean( bs_phi_avg_samps ) )
                phi_bin_phi_err = np.append( phi_bin_phi_err, np.std( bs_phi_avg_samps ) )
                
            ##fi
        ###j
        
        return [R_bin_cent, phi_bin_vR, phi_bin_vR_err, phi_bin_vT,
                phi_bin_vT_err, phi_bin_phi, phi_bin_phi_err]
    #def
    
    def _make_data_like_bootstrap_samples(self, R, phi, vR, vT, phi_err=0.01, 
                                            vT_err=0.5, vR_err=0.5):
        '''make_data_like_bootstrap_samples:
        
        Take a series of R/phi data and velocities and knit it into a form that 
        looks like the bootstrap sample arrays, and which is appropriate for using 
        in the linear model functions.
        
        Args:
            R
            phi (float array) - Phi positions
            vT (float array) - Tangential velocities
            vR (float array) - Radial velocities
            phi_err (float array) - Phi position errors [None]
            vT_err (float array) - Tangential velocity errors [None]
            vR_err (float array) - Radial velocity errors [None]
            
        Returns:
            bs_samples_vT (N-array) - Array of the vR bootstrap sample results 
                for a single radius. It contains:
                - R_bin_cent (float) - Radial bin center
                - R_bin_size (float) - Radial bin size
                - vT (float array) - vT as a function of phi
                - vT_error (float array) - vT uncertainty as a function of phi
                - phi_bin_phi (float array) - phi bin centers
                - phi_bin_phi_err (float array) - phi bin center uncertainty
            bs_samples_vR (N-array) - same but for vT
        '''
        
        # Declare the arrays which hold the bootstrap samples
        bs_samples_vT = []
        bs_samples_vR = []
        
        # Loop over each unique radius and extract all the data for that bin
        for i in range(self.n_R_bins):
            
            this_R_bin_cent = self.R_bin_cents[i]
            where_unique_R = np.where(R==this_R_bin_cent)[0]
            this_phi = phi[where_unique_R]
            this_vT = vT[where_unique_R]
            this_vR = vR[where_unique_R]
            
            # Now generate the error arrays. Start of as small numbers but can 
            # be filled. Handles arrays of errors, but also constants.
            if type(phi_err) == float or type(phi_err) == int:
                this_phi_err = np.ones_like(this_phi)*phi_err
            else:
                this_phi_err = phi_err[where_unique_R]
            ##ie
            
            if type(vT_err) == float or type(vT_err) == int:
                this_vT_err = np.ones_like(this_phi)*vT_err
            else:
                this_vT_err = vT_err[where_unique_R]
            ##ie
            
            if type(vR_err) == float or type(vR_err) == int:
                this_vR_err = np.ones_like(this_phi)*vR_err
            else:
                this_vR_err = vR_err[where_unique_R]
            ##ie
            
            # Make the velocity sample
            vT_sample = [this_R_bin_cent, this_vT, this_vT_err, 
                         this_phi, this_phi_err]
            vR_sample = [this_R_bin_cent, this_vR, this_vR_err,
                         this_phi, this_phi_err]
            
            bs_samples_vT.append(vT_sample)
            bs_samples_vR.append(vR_sample)
        ###i
        
        return bs_samples_vR, bs_samples_vT
    #def
    
    def run_iterating_linear_model(self,update_results=True,
        n_iterate=None):
        '''run_iterating_linear_model:
        
        Function to iterate over the loop where the linear model is evaluated,
        each time the noise model is updated to apply higher constant noise 
        offsets to radial bins which don't match the overall trends as well.
        
        Args:
            update_results (bool) - Should the results property be updated
            n_iterate (int) - Number of times to iterate the noise model [self]
        
        Returns:
            results_arr (list) - List of linear model fitting results
        '''

        if n_iterate is None:
            n_iterate = self.n_iterate
        ##fi

        # Empty arrays to hold results and errors
        results_arr = []
        
        # Set the size of the extra variance arrays
        if self.n_velocities == 2:
            extra_variance = np.zeros((self.n_R_bins,2))
        else:
            extra_variance = np.zeros(self.n_R_bins)
        ##ie
        
        # Loop over all the times we are supposed to iterate the model
        for i in range( n_iterate ):
            
            # Determine if we are using one velocity or two velocities
            if self.n_velocities==1:
                likelihood, prod_likelihood, phib_max_likelihood, bs, ms,\
                bs_err, ms_err, variance_model_data \
                = self._iterate_noise_model_1_velocity(extra_variance)
            elif self.n_velocities==2:
                likelihood_vT, likelihood_vR, prod_likelihood_vT,\
                prod_likelihood_vR, prod_likelihood_both, phib_max_likelihood,\
                bs, ms, bs_err, ms_err, variance_model_data\
                = self._iterate_noise_model_2_velocities(extra_variance)
            ##ie
            
            # Update the variance
            extra_variance = variance_model_data
            
            # Construct the output array
            if self.n_velocities==2:
                output_results = [likelihood_vT, likelihood_vR, 
                    prod_likelihood_vT, prod_likelihood_vR, 
                    prod_likelihood_both, phib_max_likelihood, 
                    bs, ms, bs_err, ms_err, variance_model_data]
            if self.n_velocities==1:
                output_results = [likelihood, prod_likelihood, 
                    phib_max_likelihood, bs, ms, bs_err, ms_err, 
                    variance_model_data]
            ##fi
            results_arr.append(output_results)
        ###i
        
        if update_results:
            self.results_arr=results_arr
        ##fi
        
        return results_arr
    #def
    
    def _generate_gaussian_prior_m_b(self, prior_style, R_bin_cent=None):
        '''_generate_gaussian_prior_m_b:
        
        Make the parameters of the prior: the mean sample and the inverse variance. 
        For both m and b. vR and vT have different values. Since LinearModel2 
        fits b as a constant across all vR bins then the vR prior will 
        take that special form
        
        Args:
            prior_style (string) - Either 'vT' or 'vR'
            R_bin_cent (float) - Radius in kpc for the circular velocity curve
        
        Returns:
            X0 (2x1 element array) - Mean of the gaussian prior
            SIGMA_inv (2x2 array) - Inverse of the variance array
        '''
        # Generate the prior
        if prior_style == 'vT':
            if R_bin_cent == None:
                raise Exception('Must supply radial bin center for vT prior') 
            ##fi
            if self.vT_prior_type=='df':
                which_bin = np.argmin( np.abs( R_bin_cent-self.df_prior_R ) )
                b0 = self.df_prior_vT[which_bin]
            elif self.vT_prior_type=='rotcurve':
                b0 = potential.vcirc(potential.MWPotential2014, R_bin_cent/8.0)*220.0
            b0 += self.vT_prior_offset
            m0 = 0
            X0 = np.zeros((2,1)) # Make a column vector
            X0[0,0] = b0
            X0[1,0] = m0
            SIGMA_inv = np.array([[1/self.var_b_vT,0],[0,1/self.var_m_vT]])
        elif prior_style == 'vR':
            b0 = 0
            m0 = 0
            X0 = np.zeros((1+self.n_R_bins,1)) # Make a column vector
            X0[0,0] = b0
            X0[1:,0] = m0
            SIGMA_inv = np.zeros((1+self.n_R_bins,1+self.n_R_bins))
            SIGMA_inv[0,0] = 1/self.var_b_vR
            SIGMA_cols,SIGMA_rows = np.diag_indices(1+self.n_R_bins)
            SIGMA_inv[SIGMA_cols[1:],SIGMA_cols[1:]] = 1/self.var_m_vR
        return X0, SIGMA_inv
    #def
    
    def _calculate_phib_likelihood_vT(self, bs_sample, extra_variance=0 ):
        '''_calculate_phib_likelihood_vT:
        
        Calculate the likelihood as a function of the given phib's for a single 
        radial bin and a series of phi bins for their tangential velocities.
        
        Args:
            bs_sample (6-array) - 6 element array of bootstrap properties
            extra_variance (float) - Should an extra variance term be added to 
                this radial bin?
        
        Returns:
            Likelihood (float array) - Likelihood as a function of phib
        '''
        
        # Unpack the bootstrap sample
        R_bin_cent, phi_bin_v, phi_bin_v_err, phi_bin_phi, _ = bs_sample
        
        # Make the prior
        X0, SIGMA_inv = self._generate_gaussian_prior_m_b('vT',R_bin_cent)

        # Now make the vectors
        n_good_phi_bins = len(phi_bin_v)
        Y = np.zeros((n_good_phi_bins,1))
        C = np.zeros((n_good_phi_bins,n_good_phi_bins))
        Y[:,0] = phi_bin_v
        
        # Fill the co-variance matrix
        C_rows,C_cols = np.diag_indices(C.shape[0])
        C[C_rows,C_cols] = np.square(phi_bin_v_err) + extra_variance
        C_inv = np.linalg.inv(C)

        # Now loop over all possible values of phi B, making the vector 
        # A for each and calculating the likelihood.
        n_phib_bins = len(self.phib_bin_cents)
        likelihood = np.zeros( n_phib_bins )
        
        for j in range(n_phib_bins):    
            A = np.ones((n_good_phi_bins,2))
            A[:,1] = self.trig_function_vT( 2*( phi_bin_phi - self.phib_bin_cents[j] ) )
            
            # Now compute the vectors which form the solution
            V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
            V = np.linalg.inv( V_inv )
            W = np.matmul( V , 
                np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
            U = np.linalg.multi_dot( [Y.T,C_inv,Y] ) + np.linalg.multi_dot( [X0.T,SIGMA_inv,X0] ) - np.linalg.multi_dot( [W.T,V_inv,W] )
            likelihood[j] = 0.5*( np.log(V.diagonal()).sum() - np.log(C.diagonal()).sum() ) - U[0,0]/2 
        ###j

        return likelihood
    #def
    
    def _calculate_phib_likelihood_vR(self, extra_variance=0):
        '''_calculate_phib_likelihood:
        
        Calculate the likelihood as a function of the given phib's for all
        radial bins simultaneously
        
        Args:
            extra_variance (float) - Should an extra variance term be added to this 
                radial bin?
        
        Returns:
            Likelihood (float array) - Likelihood as a function of phib
        '''
        
        # Make the prior
        X0, SIGMA_inv = self._generate_gaussian_prior_m_b('vR')
        
        # Construct the vectors
        # Y contains all measured data points concatenated together
        # C is square with both dimensions equal to the total number of data 
        # points
        # A is length all measured data points and width 1 + Number of radial 
        # bins (The first is deleted if the fit is forced)
        Y = np.zeros((self.n_data_points,1))
        C = np.zeros((self.n_data_points,self.n_data_points))
        fill_inds = np.arange(0,self.n_data_points,dtype='int') # Linear indices for filling
        
        # Loop over all radial bins and fill the vectors
        for i in range( self.n_R_bins ):
            # Unpack the bootstrap sample
            _, phi_bin_v, phi_bin_v_err, _, _ = self.bs_sample_vR[i]
            
            # Extract the indices for just this bin
            select_inds = np.arange(0,len(phi_bin_v),dtype='int')
            R_inds = fill_inds[ select_inds ]
            
            # Fill arrays. ** Co-variance not supported b/c large matrix inversion issues **
            Y[R_inds,0] = phi_bin_v
            C[R_inds,R_inds] = np.square(phi_bin_v_err) + extra_variance[i]
            
            # Remove the indices which are already filled
            fill_inds = np.delete(fill_inds,select_inds)
        ###i
        
        # Invert the co-variance matrix
        C_inv = np.linalg.inv(C)

        # If the y-intercept is forced then apply that to the Y vector. Also 
        # trim the prior and prior variance arrays
        if self.force_yint_vR:
            Y[:,0] = Y[:,0] - self.force_yint_vR_value
            X0 = np.delete(X0,0)
            SIGMA_inv = np.delete(np.delete(SIGMA_inv,0,axis=0),0,axis=1)
            # Reshape axis to maintain original number of dimensions
            X0 = X0.reshape(X0.shape[0],1)
            SIGMA_inv = SIGMA_inv.reshape(SIGMA_inv.shape[0],SIGMA_inv.shape[0])
        ##fi

        # Now loop over all possible values of phi B, making the vector 
        # A for each and calculating the likelihood.
        n_phib_bins = len(self.phib_bin_cents)
        likelihood = np.zeros( n_phib_bins )
                
        for j in range(n_phib_bins):
            
            # Setup A, the vector containing the positions. It has a shape of 
            # the total number of data points times 1 + the number of radial 
            # bins (or just the number of radial bins if forcing the y-intercept)
            A = np.zeros((self.n_data_points,self.n_R_bins+1))
            fill_inds = np.arange(0,self.n_data_points,dtype='int') 
            
            for i in range(self.n_R_bins):
                _, _, _, phi_bin_phi, _ = self.bs_sample_vR[i]
                select_inds = np.arange(0,len(phi_bin_phi),dtype='int')
                R_inds = fill_inds[ select_inds ]
                A[R_inds,i+1] = self.trig_function_vR( 2*( phi_bin_phi - self.phib_bin_cents[j] ) )
                fill_inds = np.delete(fill_inds,select_inds)
            ##fi
            
            # If we're forcing the y-intercept delete the first column of A
            if self.force_yint_vR:
                A = np.delete(A,0,axis=1)
            ##fi
            
            # Now compute the vectors which form the solution
            V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
            V = np.linalg.inv( V_inv )
            W = np.matmul( V , 
                np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
            U = np.linalg.multi_dot( [Y.T,C_inv,Y] ) + np.linalg.multi_dot( [X0.T,SIGMA_inv,X0] ) - np.linalg.multi_dot( [W.T,V_inv,W] )
            # Calculate the log likelihood, changing data types for numerical stability
            likelihood[j] = 0.5*( np.log(V.diagonal()).sum() - np.log(C.diagonal()).sum() ) - U[0,0]/2 
        ###j 
        
        return likelihood

    #def
    
    def _calculate_best_fit_m_b_vT(self, phiB, bs_sample, extra_variance=0):
        '''_calculate_best_fit_m_b_vT:
        
        Calculate the best-fitting m and b values for the for vT data
        
        Args:
            phiB (float) - phiB
            bs_sample (6-array) - 6 element array of bootstrap properties
            extra_variance (float) - Should an extra variance term be added to 
                this radial bin? [0]
        
        Returns:
            X (2-array) - Best-fitting m and b
            SIG_X (2-array) - Uncertainty in the best-fit
        '''
        
        # Unpack the bootstrap sample
        R_bin_cent, phi_bin_v, phi_bin_v_err, phi_bin_phi, _ = bs_sample
        
        # Make the prior
        X0, SIGMA_inv = self._generate_gaussian_prior_m_b('vT',R_bin_cent)

        # Now make the vectors
        n_good_phi_bins = len(phi_bin_v)
        Y = np.zeros((n_good_phi_bins,1))
        C = np.zeros((n_good_phi_bins,n_good_phi_bins))
        Y[:,0] = phi_bin_v
        
        # Fill the co-variance matrix
        C_rows,C_cols = np.diag_indices(C.shape[0])
        C[C_rows,C_cols] = np.square(phi_bin_v_err) + extra_variance
        C_inv = np.linalg.inv(C)
        
        A = np.ones((n_good_phi_bins,2))
        A[:,1] = self.trig_function_vT( 2*( phi_bin_phi - phiB ) )
        
        V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
        V = np.linalg.inv( V_inv )
        W = np.matmul( V , np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
        return W, V
    #def
    
    def _calculate_best_fit_m_b_vR(self, phiB, extra_variance=0):
        '''_calculate_best_fit_m_b_vR:
        
        Calculate the best-fitting m and b values for the for the vR data 
        for all bins simultaneously.
        
        Args:
            phiB (float) - perturbation angle
            extra_variance (float) - Should an extra variance term be added to 
                this radial bin? [0]
        
        Returns:
            X (N+1-array) - Best-fitting m and b (N is number of radial bins)
            SIG_X (N+1-array) - Uncertainty in the best-fit
        '''
        
        # Make the prior
        X0, SIGMA_inv = self._generate_gaussian_prior_m_b('vR')
        
        # Construct the vectors
        # Y contains all measured data points concatenated together
        # C is square with both dimensions equal to the total number of data 
        # points
        # A is length all measured data points and width 1 + Number of radial 
        # bins (The first is deleted if the fit is forced)
        Y = np.zeros((self.n_data_points,1))
        C = np.zeros((self.n_data_points,self.n_data_points))
        A = np.zeros((self.n_data_points,1+self.n_R_bins))
        A[:,0] = 1
        
        # Linear indices for filling
        fill_inds = np.arange(0,self.n_data_points,dtype='int')
        
        # Loop over all radial bins and fill the vectors
        for i in range( self.n_R_bins ):
            # Unpack the bootstrap sample
            _, phi_bin_v, phi_bin_v_err, phi_bin_phi, _ = self.bs_sample_vR[i]
            
            # Extract the indices for just this bin
            select_inds = np.arange(0,len(phi_bin_v),dtype='int')
            R_inds = fill_inds[ select_inds ]
            
            # Fill arrays
            Y[R_inds,0] = phi_bin_v
            C[R_inds,R_inds] = np.square(phi_bin_v_err) + extra_variance[i]
            A[R_inds,i+1] = self.trig_function_vR( 2*( phi_bin_phi - phiB ) )
            
            # Remove the indices which are already filled
            fill_inds = np.delete(fill_inds,select_inds)
        ###i
        
        # Invert the co-variance matrix
        C_inv = np.linalg.inv(C)
        
        # If we're forcing the y-intercept delete the first column of A and 
        # apply the offset to the Y values. Also trim the prior and prior 
        # variance arrays
        if self.force_yint_vR:
            A = np.delete(A,0,axis=1)
            Y[:,0] = Y[:,0] - self.force_yint_vR_value
            X0 = np.delete(X0,0)
            SIGMA_inv = np.delete(np.delete(SIGMA_inv,0,axis=0),0,axis=1)
            # Reshape axis to maintain original number of dimensions
            X0 = X0.reshape(X0.shape[0],1)
            SIGMA_inv = SIGMA_inv.reshape(SIGMA_inv.shape[0],SIGMA_inv.shape[0])
        ##fi
        
        V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
        V = np.linalg.inv( V_inv )
        W = np.matmul( V , np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
        return W, V
    #def
    
    def _calculate_variance_data_model(self, bs_sample, m, b, phib, 
                                       trig_function):
        '''calculate_variance_data_model:
        
        Calculate the variance of the differences between the best-fitting model and the 
        data.
        
        Args:
            bs_sample 
            m (float) - Best-fitting m
            b (float) - Best-fitting b
            phib (float) - Best-fitting phib
            trig_function (function) - Either np.sin or np.cos
            
        Returns:
            variance (float) - variance of the difference between the model and data
        
        '''
        
        # Unpack the bootstrap sample
        _, phi_bin_v, _, phi_bin_phi, _ = bs_sample
        
        # Calculate the model at the locations where there is data
        model_v = b + m*trig_function(2*(phi_bin_phi-phib))
        
        # Calculate the standard deviation of the differences between model and data
        sd_data_model = np.std(model_v-phi_bin_v)
        
        return np.square(sd_data_model)
        
    #def
    
    def _iterate_noise_model_1_velocity(self, extra_variance):
        '''_iterate_noise_model_1_velocity:
        
        Iterate over the calculation of the best-fitting linear model using 
        a single velocity only, adding an empirically derived variance to 
        radial bins which do not match the overall trends particularly well.
        
        Args:
            extra_variance (n_R x 2 array) - Extra variance for vR and vT as a 
                function of radius.
        
        Returns:
        
        '''
        
        # Calculate the best phiB value
        
        # For vT (loop over all radial bins)
        if self.use_vT:
            # Likelihood matrix for vT radial bins and phiB values
            likelihood = np.ones( ( self.n_R_bins, self.n_phib_bins ) )
            # Loop over the radial bins and calculate the likelihood as a 
            # function of phiB for tangential velocities
            for j in range( self.n_R_bins ):
                likelihood[j,:] = \
                self._calculate_phib_likelihood_vT(self.bs_sample_vT[j], 
                    extra_variance=extra_variance[j] )
            ###j
            prod_likelihood = np.sum(likelihood, axis=0)
        ##fi
        
        # For vR (all radial bins at once)
        if self.use_vR:
            # Calculate the phiB likelihood for all radial bins for vR
            likelihood = \
            self._calculate_phib_likelihood_vR(extra_variance=extra_variance)
            phib_max_likelihood_arg = np.argmax( likelihood )
            prod_likelihood = likelihood
        ##fi
        
        # Determine the maximum likelihood phiB
        phib_max_likelihood_arg = np.argmax( prod_likelihood )
        phib_max_likelihood = self.phib_bin_cents[phib_max_likelihood_arg]
        
        # If we are forcing phiB then assign it
        if self.force_phiB:
            use_phiB = self.phiB
        else:
            use_phiB = phib_max_likelihood
        ##ie

        # Prepare the solution arrays
        ms = np.zeros( self.n_R_bins )
        bs = np.zeros( self.n_R_bins )
        ms_err = np.zeros( self.n_R_bins )
        bs_err = np.zeros( self.n_R_bins )
        variance_model_data = np.zeros( self.n_R_bins )
        
        if self.use_vT:
            # Loop over radial bins, calculate the best-fitting m and b
            for j in range( self.n_R_bins ):

                # Now determine the best-fitting m and b
                X, SIG_X = \
                self._calculate_best_fit_m_b_vT(use_phiB,
                                                self.bs_sample_vT[j], 
                                                extra_variance=extra_variance[j])       
                bs[j] = X[0]
                bs_err[j] = np.sqrt( SIG_X[0,0] )
                ms[j] = X[1]
                ms_err[j] = np.sqrt( SIG_X[1,1] )
            ###j
        ##fi
        
        if self.use_vR:
            X, SIG_X = \
            self._calculate_best_fit_m_b_vR(use_phiB, extra_variance)
            
            # All b values are fit together
            if self.force_yint_vR:
                bs[:] = self.force_yint_vR_value
                bs_err[:] = 0
                ms[:] = X[:,0]
                ms_err[:] = np.sqrt( SIG_X.diagonal() )
            else:                 
                bs[:] = X[0]
                bs_err[:] = np.sqrt( SIG_X[0,0] )
                ms = X[1:]
                SIG_X_rows,SIG_X_cols = np.diag_indices(SIG_X.shape[0])
                ms_err = np.sqrt( SIG_X[SIG_X_rows[1:],SIG_X_cols[1:]] )
            ##fi
        ##fi
        
        # Now calculate the standard deviation of the difference between the 
        # data and model
        for j in range( self.n_R_bins ):
            variance_model_data[j] = \
            self._calculate_variance_data_model(self.bs_sample_1v[j], ms[j], 
                                                bs[j], use_phiB, self.trig_fn_1v)
        ###i
        
        return likelihood, prod_likelihood, phib_max_likelihood, \
               bs, ms, bs_err, ms_err, variance_model_data
    #def
        
    def _iterate_noise_model_2_velocities(self, extra_variance):
        '''_iterate_noise_model_2_velocities:
        
        Iterate over the calculation of the best-fitting linear model using 
        both vR and vT velocities, adding an empirically derived variance to 
        radial bins which do not match the overall trends particularly well.
        
        Args:
            extra_variance (n_R x 2 array) - Extra variance for vR and vT as a 
                function of radius.
            
        Returns:
            
        '''
            
        # Make an array to store the log likelihoods
        likelihood_vT = np.ones( ( self.n_R_bins, self.n_phib_bins ) )
        likelihood_vR = np.ones( ( self.n_R_bins, self.n_phib_bins ) )

        # Loop over the radial bins and calculate the likelihood as a function 
        # of phiB for tangential velocities
        for j in range( self.n_R_bins ):

            # Calculate the log likelihood of the tangential and radial
            # velocities as functions of phiB
            likelihood_vT[j,:] = \
            self._calculate_phib_likelihood_vT(self.bs_sample_vT[j],
                extra_variance=extra_variance[j,0] )
        ###j
        
        # Calculate the likelihood as a function of phiB for radial velocities
        likelihood_vR = \
        self._calculate_phib_likelihood_vR(extra_variance=extra_variance[:,1])

        # Marginalize over all radii for vT, already done for vR
        prod_likelihood_vT = np.sum(likelihood_vT, axis=0)
        prod_likelihood_vR = likelihood_vR
        prod_likelihood_both = prod_likelihood_vR + prod_likelihood_vT

        # Determine the best-fitting phib
        phib_max_likelihood_arg = np.argmax( prod_likelihood_both )
        phib_max_likelihood = self.phib_bin_cents[phib_max_likelihood_arg]
        
        # If we are forcing phiB then assign it
        if self.force_phiB:
            use_phiB = self.phiB
        else:
            use_phiB = phib_max_likelihood
        ##ie

        ms = np.zeros( (self.n_R_bins,2) )
        bs = np.zeros( (self.n_R_bins,2) )
        ms_err = np.zeros( (self.n_R_bins,2) )
        bs_err = np.zeros( (self.n_R_bins,2) )
        variance_model_data = np.zeros((self.n_R_bins,2))

        # Loop over radial bins, calculate the best-fitting m and b for vT
        for j in range( self.n_R_bins ):

            # Now determine the best-fitting m and b for vT
            X_vT, SIG_X_vT = \
            self._calculate_best_fit_m_b_vT(use_phiB, self.bs_sample_vT[j],
                                            extra_variance=extra_variance[j,0])

            bs[j,0] = X_vT[0]
            bs_err[j,0] = np.sqrt( SIG_X_vT[0,0] )
            ms[j,0] = X_vT[1]
            ms_err[j,0] = np.sqrt( SIG_X_vT[1,1] )
        ###j 
        
        # Do it once for vR across all radii
        X_vR, SIG_X_vR = \
        self._calculate_best_fit_m_b_vR(use_phiB, 
            extra_variance=extra_variance[:,1])
        if self.force_yint_vR:
            bs[:,1] = self.force_yint_vR_value
            bs_err[:,1] = 0
            ms[:,1] = X_vR[:,0]
            ms_err[:,1] = np.sqrt( SIG_X_vR.diagonal() )
        else:                 
            bs[:,1] = X_vR[0]
            bs_err[:,1] = np.sqrt( SIG_X_vR[0,0] )
            ms[:,1] = X_vR[1:,0]
            SIG_X_vR_rows,SIG_X_vR_cols = np.diag_indices(SIG_X_vR.shape[0])
            ms_err[:,1] = np.sqrt( SIG_X_vR[SIG_X_vR_rows[1:],SIG_X_vR_cols[1:]] )
        
        for j in range( self.n_R_bins ):
            # Now calculate the standard deviation of the difference between the data and the model
            variance_model_data[j,0] = \
            self._calculate_variance_data_model(self.bs_sample_vT[j], ms[j,0], 
                                                bs[j,0], use_phiB, np.cos)
            variance_model_data[j,1] = \
            self._calculate_variance_data_model(self.bs_sample_vR[j], ms[j,1], 
                                                bs[j,1], use_phiB, np.sin)
        ###j
        
        return likelihood_vT, likelihood_vR, prod_likelihood_vT, \
               prod_likelihood_vR, prod_likelihood_both, phib_max_likelihood, \
               bs, ms, bs_err, ms_err, variance_model_data
    #def
    
    # Define some plotting routines
    def plot_velocity_known_m_b_phi(self, velocity_type, fig=None, axs=None, 
                                    phi_lim=[-np.pi/2,np.pi/2], 
                                    plot_best_fit=True):
        '''plot_velocity_known_m_b_phi
        
        Plot the velocities as a function of radius for a bootstrap sample.
        overplot the best-fitting solution from the linear model. Note that 
        fig and axs need to be commensurate with the number of radial bins 
        being plotted.
        
        Args:
            velocity_type (string) - Either 'vR' or 'vT': which one to use
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created [None]
            phi_lim (2-array) - The limits of phi to plot
            plot_best_fit (bool) - Include the best fitting m=2 profile
        '''
        
        # Select the right bootstrap sample
        if velocity_type == 'vR':
            bs_samp = self.bs_sample_vR
        if velocity_type == 'vT':
            bs_samp = self.bs_sample_vT
        ##fi
        
        if fig is None and axs is None:
            fig = plt.figure( figsize=(5,self.n_R_bins*2) )
            axs = fig.subplots( nrows=self.n_R_bins, ncols=1 )
        ##fi
        
        # Loop over all radii
        for i in range( self.n_R_bins ):
            
            # Unpack the velocity sample for this radius
            bin_R_cent = bs_samp[i][0]
            bin_v = bs_samp[i][1]
            bin_v_err = bs_samp[i][2]
            bin_phi = bs_samp[i][3]
            
            # Plot
            axs[i].errorbar( bin_phi, bin_v, yerr=bin_v_err, fmt='o', 
                ecolor='Black', marker='o', markerfacecolor='None', 
                markeredgecolor='Black', markersize=5)
        
            # Plot the best-fitting amplitude
            if plot_best_fit:
                trig_phis = np.linspace(phi_lim[0], phi_lim[1], num=100)
                if velocity_type == 'vR':
                    axs[i].plot( trig_phis, 
                        self.b_vR[i]+self.m_vR[i]*np.sin(2*(trig_phis-self.phiB)))
                if velocity_type == 'vT':
                    axs[i].plot( trig_phis, 
                        self.b_vT[i]+self.m_vT[i]*np.cos(2*(trig_phis-self.phiB)))
                ##fi
            ##fi
        
            # Add fiducials: bar, 0 line or tangential velocity curve
            axs[i].axvline( 25*(np.pi/180), linestyle='dotted', linewidth=1.0, 
                color='Red', label='Bar Angle')
            X0, _ = self._generate_gaussian_prior_m_b(velocity_type, bin_R_cent)
            b0 = X0[0,0]
            axs[i].axhline( b0, linestyle='dashed', color='Black', 
                linewidth=1.0, label=r'$V_{circ}$')
                
            # Annotate
            axs[i].annotate( r'$R_{cen}=$'+str(bin_R_cent)+' kpc', 
                xy=(0.05,0.8), xycoords='axes fraction' )
                
            # Set limits
            axs[i].set_xlim( phi_lim[0], phi_lim[1] )
            
            # Set the labels
            if velocity_type == 'vR':
                axs[i].set_ylabel(r'$v_{R}$ [km/s]')
            if velocity_type == 'vT':
                axs[i].set_ylabel(r'$v_{T}$ [km/s]')
            ##fi
            if i == 0:
                axs[i].set_xlabel(r'$\phi$')
            ##fi
        
        axs[0].legend(loc='best')
        fig.subplots_adjust(hspace=0)
        
        return fig, axs
    #def
    
    # Define some plotting routines
    def plot_vRvT_known_m_b_phi(self, fig=None, axs=None,
                                    phi_lim=[-np.pi/2,np.pi/2], label_fs=12,):
        '''plot_vTvR_known_m_b_phi
        
        Plot both velocities as a function of radius for a bootstrap sample.
        overplot the best-fitting solution from the linear model. Note that 
        fig and axs need to be commensurate with the number of radial bins 
        being plotted.
        
        Args:
            velocity_type (string) - Either 'vR' or 'vT': which one to use
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created [None]
            phi_lim (2-array) - The limits of phi to plot
        '''
        
        if fig is None and axs is None:
            fig = plt.figure( figsize=(10,self.n_R_bins*2) )
            axs = fig.subplots( nrows=self.n_R_bins, ncols=2 )
        ##fi
        
        # Loop over all radii
        for i in range( self.n_R_bins ):
            
            # Unpack the velocity sample for this radius
            bin_R_cent = self.bs_sample_vR[i][0]
            bin_vR = self.bs_sample_vR[i][1]
            bin_vR_err = self.bs_sample_vR[i][2]
            bin_phi = self.bs_sample_vR[i][3]
            bin_vT = self.bs_sample_vT[i][1]
            bin_vT_err = self.bs_sample_vT[i][2]
            
            # Plot
            axs[i,0].errorbar( bin_phi, bin_vR, yerr=bin_vR_err, fmt='o', 
                ecolor='Black', marker='o', markerfacecolor='None', 
                markeredgecolor='Black', markersize=5)
            axs[i,1].errorbar( bin_phi, bin_vT, yerr=bin_vT_err, fmt='o', 
                ecolor='Black', marker='o', markerfacecolor='None', 
                markeredgecolor='Black', markersize=5)
        
            # Plot the best-fitting amplitude
            trig_phis = np.linspace(phi_lim[0], phi_lim[1], num=100)
            axs[i,0].plot( trig_phis, 
                self.b_vR[i]+self.m_vR[i]*np.sin(2*(trig_phis-self.phiB)))
            axs[i,1].plot( trig_phis, 
                self.b_vT[i]+self.m_vT[i]*np.cos(2*(trig_phis-self.phiB)))
        
            # Add fiducials: bar, 0 line or tangential velocity curve
            axs[i,0].axvline( 25*(np.pi/180), linestyle='dotted', linewidth=1.0, 
                color='Red')
            axs[i,1].axvline( 25*(np.pi/180), linestyle='dotted', linewidth=1.0, 
                color='Red')
            axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=1.0 )
            X0, _ = self._generate_gaussian_prior_m_b('vT', bin_R_cent)
            b0 = X0[0,0]
            axs[i,0].axhline( 0, linestyle='dashed', color='Black', linewidth=1.0 )
            axs[i,1].axhline( b0, linestyle='dashed', color='Black', linewidth=1.0 )
                
            # Annotate
            axs[i,0].annotate( r'$R_{cen}=$'+str(bin_R_cent)+' kpc', 
                xy=(0.05,0.8), xycoords='axes fraction' )
                
            # Set limits
            axs[i,0].set_xlim( phi_lim[0], phi_lim[1] )
            axs[i,1].set_xlim( phi_lim[0], phi_lim[1] )
            
            # Set the labels
            axs[i,0].set_ylabel(r'$v_{R}$ [km/s]', fontsize=label_fs)
            axs[i,1].set_ylabel(r'$v_{T}$ [km/s]', fontsize=label_fs)
            ##fi
            if i == self.n_R_bins-1:
                axs[i,0].set_xlabel(r'$\phi$', fontsize=label_fs)
                axs[i,1].set_ylabel(r'$\phi$', fontsize=label_fs)
            ##fi
        
        fig.subplots_adjust(hspace=0)
        
        return fig, axs
    #def
    
    def plot_velocity_m_r(self, fig=None, axs=None, plot_type='errorbar', 
                            plot_kws={}, label_fs=12, which_velocity=None):
        '''plot_velocity_m_r:
        
        Plot both m and b as a functions of radius for either vR or vT
        
        Args:
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created. Should be 2 length [None]
            plot_type (string) - What type of plot to make? Either 'errorbar'm 
                'scatter' or 'plot' ['errorbar']
            plot_kws (dict) - Dictionary of properties to be passed to the 
                plotting functions. Make sure commensurate with plot_type! [{}]
        '''
        
        # Format for this figure is single column 2 axis
        if fig is None or axs is None:
            fig = plt.figure( figsize=(12,4) )
            axs = fig.subplots( nrows=1, ncols=2 )
        ##fi
        
        if which_velocity == None:
            if self.n_velocities==2:
                raise Exception('No velocity specified and 2 used.')
            ##fi
            which_velocity = self.vel_1v
        ##fi
        
        if which_velocity == 'vR':
            use_b = self.b_vR
            use_m = self.m_vR
            use_b_err = self.b_err_vR
            use_m_err = self.m_err_vR
        if which_velocity == 'vT':
            use_b = self.b_vT
            use_m = self.m_vT
            use_b_err = self.b_err_vT
            use_m_err = self.m_err_vT
        ##fi
        
        # Plot
        if plot_type == 'errorbar':
            axs[0].errorbar( self.R_bin_cents, use_b, yerr=use_b_err, 
                **plot_kws)
            axs[1].errorbar( self.R_bin_cents, use_m, yerr=use_m_err, 
                **plot_kws)
        elif plot_type == 'plot':
            axs[0].plot( self.R_bin_cents, use_b, **plot_kws)
            axs[1].plot( self.R_bin_cents, use_m, **plot_kws)
        elif plot_type == 'scatter':
            axs[0].scatter( self.R_bin_cents, use_b, **plot_kws)
            axs[1].scatter( self.R_bin_cents, use_m, **plot_kws)
        ##fi
        
        # Labels and limits
        if which_velocity == 'vR':
            axs[0].set_ylabel(r'$b_{R}$ [km/s]', fontsize=label_fs)
            axs[1].set_ylabel(r'$m_{R}$ [km/s]', fontsize=label_fs)
        if which_velocity == 'vT':
            axs[0].set_ylabel(r'$b_{T}$ [km/s]', fontsize=label_fs)
            axs[1].set_ylabel(r'$m_{T}$ [km/s]', fontsize=label_fs)
        ##fi
        axs[0].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[1].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[0].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[1].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        
        # Add fiducials
        axs[1].axhline(0, linestyle='dashed', color='Black')
        if which_velocity == 'vR':
            axs[0].axhline(0, linestyle='dashed', color='Black')
        if which_velocity == 'vT':
            if self.vT_prior_type=='df':    
                axs[0].plot( self.df_prior_R, self.df_prior_vT,
                    linestyle='dashed', color='Black' )
            if self.vT_prior_type=='rotcurve':
                axs[0].plot( self.rotcurve_prior_R, self.rotcurve_prior_vT,
                    linestyle='dashed', color='Black')
            ##fi
        ##fi
        
        return fig, axs
    
    def plot_vRvT_m_r(self, fig=None, axs=None, plot_type='errorbar', 
                        plot_kws={}, label_fs=12):
        '''plot_m_b:
        
        Plot both m and b as functions of radius for vT and vR profiles
        
        Args:
            fig (matplotlib figure object) - Figure object to use, if None then 
                one will be created [None]
            axs (matplotlib axs object) - Axs objects to use, if None then they 
                will be created. Should be 2x2 [None]
            plot_type (string) - What type of plot to make? Either 'errorbar'
                'scatter' or 'plot' ['errorbar']
            plot_kws (dict) - Dictionary of properties to be passed to the 
                plotting functions. Make sure commensurate with plot_type! [{}]
        '''
        
        # Format for this figure is 2x2
        if fig is None or axs is None:
            fig = plt.figure( figsize=(12,6) ) 
            axs = fig.subplots( nrows=2, ncols=2 )
        ##fi
        
        # Plot
        if plot_type == 'errorbar':
            axs[0,0].errorbar( self.R_bin_cents, self.b_vR, yerr=self.b_err_vR, 
                **plot_kws)
            axs[0,1].errorbar( self.R_bin_cents, self.m_vR, yerr=self.m_err_vR, 
                **plot_kws)
            axs[1,0].errorbar( self.R_bin_cents, self.b_vT, yerr=self.b_err_vT, 
                **plot_kws)
            axs[1,1].errorbar( self.R_bin_cents, self.m_vT, yerr=self.m_err_vT, 
                **plot_kws)
        elif plot_type == 'plot':
            axs[0,0].plot( self.R_bin_cents, self.b_vR, **plot_kws)
            axs[0,1].plot( self.R_bin_cents, self.m_vR, **plot_kws)
            axs[1,0].plot( self.R_bin_cents, self.b_vT, **plot_kws)
            axs[1,1].plot( self.R_bin_cents, self.m_vT, **plot_kws)
        elif plot_type == 'scatter':
            axs[0,0].scatter( self.R_bin_cents, self.b_vR, **plot_kws)
            axs[0,1].scatter( self.R_bin_cents, self.m_vR, **plot_kws)
            axs[1,0].scatter( self.R_bin_cents, self.b_vT, **plot_kws)
            axs[1,1].scatter( self.R_bin_cents, self.m_vT, **plot_kws)
        
        # Labels and limits
        axs[0,0].set_ylabel(r'$b_{R}$ [km/s]', fontsize=label_fs)
        axs[0,1].set_ylabel(r'$m_{R}$ [km/s]', fontsize=label_fs)
        axs[1,0].set_ylabel(r'$b_{T}$ [km/s]', fontsize=label_fs)
        axs[1,1].set_ylabel(r'$m_{T}$ [km/s]', fontsize=label_fs)
        axs[0,0].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[0,1].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[1,0].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[1,1].set_xlabel(r'R [kpc]', fontsize=label_fs)
        axs[0,0].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[0,1].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[1,0].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        axs[1,1].set_xlim( np.min(self.R_bin_cents)-1, np.max(self.R_bin_cents)+1 )
        
        # Add fiducials
        axs[0,0].axhline(0, linestyle='dashed', color='Black')
        axs[0,1].axhline(0, linestyle='dashed', color='Black')
        axs[1,1].axhline(0, linestyle='dashed', color='Black')
        
        # Prior
        if self.vT_prior_type=='df':    
            axs[1,0].plot( self.df_prior_R, self.df_prior_vT,
                linestyle='dashed', color='Black' )
        if self.vT_prior_type=='rotcurve':
            axs[1,0].plot( self.rotcurve_prior_R, self.rotcurve_prior_vT,
                linestyle='dashed', color='Black')
        ##fi
        
        return fig, axs
        
    def sample_bootstrap(self, sample_phi=False):
        '''sample_bootstrap:
        
        Sample from the bootstrap assuming its own errors. Return two 
        bootstrap samples of the same shape (as vR and vT) with the 
        same errors
        
        Args:
            sample_phi (bool) - Draw samples from the phi distribution [False]
            
        '''
        
        # Make empty arrays
        bs_mc_samp_vR = []
        bs_mc_samp_vT = []
        
        # Loop over all radii
        for i in range( self.n_R_bins ):
            
            n_phi_bins = len( self.bs_sample_vR[i][1] )
            
            # Loop over each phi bin
            this_R_vR_samp = np.zeros(n_phi_bins)
            this_R_vT_samp = np.zeros(n_phi_bins)
            this_R_phi_samp = np.zeros(n_phi_bins)
            
            for j in range( n_phi_bins ):
                
                # Generate the samples for this phi bin
                vR_samp = np.random.normal(loc=self.bs_sample_vR[i][1][j], 
                    scale=self.bs_sample_vR[i][2][j], size=1).astype(float)[0]
                this_R_vR_samp[j] = vR_samp
                vT_samp = np.random.normal(loc=self.bs_sample_vT[i][1][j], 
                    scale=self.bs_sample_vT[i][2][j], size=1).astype(float)[0]
                this_R_vT_samp[j] = vT_samp
                if sample_phi:
                    phi_samp = np.random.normal(loc=self.bs_sample_vR[i][3][j],
                        scale=self.bs_sample_vR[4][j], size=1).astype(float)[0]
                else:
                    phi_samp = self.bs_sample_vR[i][3][j]
                ##ie
                this_R_phi_samp[j] = phi_samp
            ###j
            
            # Pack results
            this_R = self.bs_sample_vR[i][0]
            this_R_vR_err = self.bs_sample_vR[i][2]
            this_R_vT_err = self.bs_sample_vT[i][2]
            this_R_phi_err = self.bs_sample_vR[i][4]
            bs_mc_samp_vR.append( [this_R,this_R_vR_samp,this_R_vR_err,
                                   this_R_phi_samp,this_R_phi_err] )
            bs_mc_samp_vT.append( [this_R,this_R_vT_samp,this_R_vT_err,
                                   this_R_phi_samp,this_R_phi_err] )
            
        ###i
        return bs_mc_samp_vR, bs_mc_samp_vT
    #def
#cls

class LinearModelSolution():
    '''LinearModelSolution:
    
    Class representing the solution to either LinearModel or LinearModel2. 
    Lightweight for easy transport, but doesn't contain any information on the 
    data that was used to obtain the solution
    
    Args:
        Required:
        use_velocities
        
        Model properties:
        th_b - Triaxial halo b/a
        th_pa - Triaxial halo position angle
        bar_omega_b - Bar pattern speed
        bar_af - Bar radial force fraction
        
        Solution parameters:
        b_vR - y-intercept for radial velocities
        m_vR - amplitudes for radial velocities 
        b_vT - y-intercept for tangential velocities
        m_vT - amplitudes for tangential velocities 
        *_err_*
        
        Optional:
        phiB - Phase of the solution
        
        
    '''
    def __init__(self,
                  # Required
                  use_velocities,
                   # Model properties
                   th_b=None,
                   th_pa=None,
                   bar_omega_b=None,
                   bar_af=None,
                   # Solution parameters
                   b_vR=None,
                   m_vR=None,
                   b_vT=None,
                   m_vT=None,
                   b_err_vR=None,
                   m_err_vR=None,
                   b_err_vT=None,
                   m_err_vT=None,
                   # Optional
                   phiB=None,
                  ):
                  
        # Figure out if we're going to use one velocity or two.
        # Needs to be a list or a tuple
        assert type(use_velocities)==list or type(use_velocities)==tuple
        if 'vT' in use_velocities:
            self.use_vT = True
        else:
            self.use_vT = False
        ##ie
        if 'vR' in use_velocities:
            self.use_vR = True
        else:
            self.use_vR = False
        ##ie
        if self.use_vR==False and self.use_vT==False:
            raise Exception('Cannot use neither vR or vT')
        ##fi
        
        # Figure out how many velocities will be used.
        if self.use_vR==False or self.use_vT==False:
            self.n_velocities=1
        else:
            self.n_velocities=2
        ##ie
        
        # Fill model parameters
        self.th_b=th_b
        self.th_pa=th_pa
        self.bar_omega_b=bar_omega_b
        self.bar_af=bar_af
        
        # Is there a triaxial halo
        if self.th_b is not None and self.th_pa is not None:
            self.has_th=True
        else:
            self.has_th=False
        ##ie
        
        # Is there a bar
        if self.bar_omega_b is not None and self.bar_af is not None:
            self.has_bar=True
        else:
            self.has_bar=False
        ##ie
        
        # Fill solution parameters
        if self.use_vT:
            assert (b_vT is not None) and (m_vT is not None) and \
                   (b_err_vT is not None) and (m_err_vT is not None),\
                   'b_vT, m_vT, b_err_vT, m_err_vT expected if using vT'
            self.b_vT=b_vT
            self.m_vT=m_vT
            self.b_err_vT=b_err_vT
            self.m_err_vT=m_err_vT
        ##fi
        if self.use_vR:
            assert (b_vR is not None) and (m_vR is not None) and \
                   (b_err_vR is not None) and (m_err_vR is not None),\
                   'b_vR, m_vR, b_err_vR, m_err_vR expected if using vR'
            self.b_vR=b_vR
            self.m_vR=m_vR
            self.b_err_vR=b_err_vR
            self.m_err_vR=m_err_vR
        ##fi
        
        # Fill optional parameters
        self.phiB=phiB
    #def
    def get_th_properties(self):
        '''get_th_properties:
        
        Return triaxial halo model properties that yielded this solution
        
        Returns:
            th_props (2-arr) - array of triaxial halo properties:
                [b/a,position_angle] in fraction of scale length and radians 
                respectively
        '''
        if self.has_th:
            return [self.th_b,self.th_pa]
        else:
            print('No triaxial halo properties provided')
            pass
        ##ie
    #def
    
    def get_bar_properties(self):
        '''get_bar_properties:
        
        Return bar model properties that yielded this solution
        
        Returns:
            bar_props (2-arr) - array of bar properties: 
                [pattern_speed,radial_force_fraction] in km/s/kpc and fraction
                of radial force at solar circle respectively
        '''
        if self.has_bar:
            return [self.bar_omega_b,self.bar_af]
        else:
            print('No bar properties provided')
            pass
        ##ie
    #def
#cls        

def make_data_like_bootstrap_samples(R, phi, vR, vT, phi_err=0.01, 
                                        vT_err=0.5, vR_err=0.5):
    '''make_data_like_bootstrap_samples:
    
    Take a series of R/phi data and velocities and knit it into a form that 
    looks like the bootstrap sample arrays, and which is appropriate for using 
    in the linear model functions.
    
    Args:
        R
        phi (float array) - Phi positions
        vT (float array) - Tangential velocities
        vR (float array) - Radial velocities
        phi_err (float array) - Phi position errors [None]
        vT_err (float array) - Tangential velocity errors [None]
        vR_err (float array) - Radial velocity errors [None]
        
    Returns:
        bs_samples_vT (N-array) - Array of the vR bootstrap sample results 
            for a single radius. It contains:
            - R_bin_cent (float) - Radial bin center
            - R_bin_size (float) - Radial bin size
            - vT (float array) - vT as a function of phi
            - vT_error (float array) - vT uncertainty as a function of phi
            - phi_bin_phi (float array) - phi bin centers
            - phi_bin_phi_err (float array) - phi bin center uncertainty
        bs_samples_vR (N-array) - same but for vT
    '''
    
    # Declare the arrays which hold the bootstrap samples
    bs_samples_vT = []
    bs_samples_vR = []
    
    # Find the unique bins
    unique_bins = np.unique(R)
    n_R_bins = len(unique_bins)
    
    # Loop over each unique radius and extract all the data for that bin
    for i in range(n_R_bins):
        
        this_R_bin_cent = unique_bins[i]
        where_unique_R = np.where(R==this_R_bin_cent)[0]
        this_phi = phi[where_unique_R]
        this_vT = vT[where_unique_R]
        this_vR = vR[where_unique_R]
        
        # Now generate the error arrays. Start of as small numbers but can 
        # be filled. Handles arrays of errors, but also constants.
        if type(phi_err) == float or type(phi_err) == int:
            this_phi_err = np.ones_like(this_phi)*phi_err
        else:
            this_phi_err = phi_err[where_unique_R]
        ##ie
        
        if type(vT_err) == float or type(vT_err) == int:
            this_vT_err = np.ones_like(this_phi)*vT_err
        else:
            this_vT_err = vT_err[where_unique_R]
        ##ie
        
        if type(vR_err) == float or type(vR_err) == int:
            this_vR_err = np.ones_like(this_phi)*vR_err
        else:
            this_vR_err = vR_err[where_unique_R]
        ##ie
        
        # Make the velocity sample
        vT_sample = [this_R_bin_cent, this_vT, this_vT_err, 
                     this_phi, this_phi_err]
        vR_sample = [this_R_bin_cent, this_vR, this_vR_err,
                     this_phi, this_phi_err]
        
        bs_samples_vT.append(vT_sample)
        bs_samples_vR.append(vR_sample)
    ###i
    
    return bs_samples_vR, bs_samples_vT
#def