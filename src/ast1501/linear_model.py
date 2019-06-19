# ----------------------------------------------------------------------------
#
# TITLE - df.py
# AUTHOR - James Lane
# PROJECT - AST1501
# CONTENTS: 
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
    '''linear_model:
    
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
    
    Args:
        instantiate_method (int) - How to instantiate the class, see above.
        gc_{R,phi,vT,vR} (float array) - Gaia star properties
        df_filename (string) - Name of the filename of DF velocity field
        {R,phi,phib}_lims (2-array) - lower and upper limits
        {R,phi,phib}_bin_size (float) - bin size
        bs_sample_{vR,vT} (6-array) - Bootstrap samples
        prior_var_arr (4-array) - 
        n_iterate (int) - Number of times to iterate the noise model [5]
        n_bs (100) - Number of bootstrap samples
        use_velocities (2-array) - Array of velocities to use in determination 
            of model properties
        force_yint_zero_vR (bool) - Force radial velocities to have 0
            y-intercept (b value) [True]
        vT_prior_type (string) - Type of prior to use for vT: 'df' for 
            distribution function inferred, 'rotcurve' for the rotation curve
            calculated from MWPotential2014
        vT_prior_offset (float) - Arbitrary offset applied to the vT prior
    '''
    
    def __init__(self, instantiate_method, gc_R=None, gc_phi=None, gc_vT=None, 
                 gc_vR=None, df_filename=None, R_lims=None, R_bin_size=None, 
                 phi_lims=None, phi_bin_size=None, phib_lims=None, 
                 phib_bin_size=None, phiB=None, bs_sample_vR=None, 
                 bs_sample_vT=None, prior_var_arr=[25,np.inf,25,np.inf], 
                 n_iterate=5, n_bs=1000, use_velocities=['vR','vT'], 
                 force_yint_zero_vR=True, vT_prior_type='df', 
                 vT_prior_offset=0):
        
        # First, get the bootstrap samples, one of three ways: calculate from 
        # Gaia data, load from file, or manually specify.
        if instantiate_method == 1:
            
            # Assert that we have the necessary keywords
            assert (gc_R is not None) and (gc_phi is not None) and \
                   (gc_vT is not None) and (gc_vR is not None),\
            'gc_R, gc_phi, gc_vT, gc_vR all expected but not provided'
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
        
        # Load the DF prior
        project_dir = '/Users/JamesLane/Science/Projects/PhD/AST1501/'
        df_prior_file_dir = 'data/generated/MWPotential2014_DF_vT_data.npy'
        df_prior_path = project_dir+df_prior_file_dir
        df_prior_R, df_prior_vT = np.load(df_prior_path)
        self.df_prior_R = df_prior_R
        self.df_prior_vT = df_prior_vT
        self.rotcurve_prior_R = np.arange(5,15,0.01)
        self.rotcurve_prior_vT = potential.vcirc(potential.MWPotential2014, 
            R=self.rotcurve_prior_R)
        # Type of prior for vT?
        self.vT_prior_type = vT_prior_type
        # Arbitrary offset for vT
        self.vT_prior_offset = vT_prior_offset
        
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
        
        # Declare whether vR will be forced to be 0
        self.force_yint_zero_vR=force_yint_zero_vR
        
        # Declare the number of n_iterate
        self.n_iterate=n_iterate
        
        # Declare phiB. If it was None, then it will be calculated during 
        # each step. If it was not none then we will force it to be the same
        self.phiB = phiB
        if phiB!=None:
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
                                            vT_err=0.01, vR_err=0.01):
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
    
    def run_iterating_linear_model(self, force_yint_zero_vR=True, 
                                    n_iterate=5, update_results=False):
        '''run_iterating_linear_model:
        
        Things
        
        Args:
            force_yint_zero_vR (bool) - Force the y-intercept to be 0 for the 
                vR velocities [True]
            n_iterate (int) - Number of times to iterate the model. Overwritten 
                by the class n_iterate property if set [5]
            update_results (bool) - Set the results property to be the results
                from this function evaluation [False]
        
        Returns:
        '''

        if self.n_iterate!=None:
            n_iterate=self.n_iterate
        ##fi

        # Empty arrays to hold results and errors
        results_arr = []
        
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
                                   extra_variance=0, force_yint_zero=False):
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
        for j in range(n_good_phi_bins):
            C[j,j] = phi_bin_v_err[j]**2 + extra_variance
        ###j
        C_inv = np.linalg.inv(C)

        # Now loop over all possible values of phi B, making the vector 
        # A for each and calculating the likelihood.
        n_phib_bins = len(self.phib_bin_cents)
        likelihood = np.zeros( n_phib_bins )
        
        for j in range(n_phib_bins):    
            if force_yint_zero:
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
            likelihood[j] = np.sqrt( np.linalg.det(V)/np.linalg.det(C) ) * np.exp( -U/2 )
        ###j

        return likelihood

    #def
    
    def _calculate_best_fit_m_b(self, R_bin_cent, phiB, bs_sample, 
                                prior_style, trig_function, 
                                force_yint_zero=False, 
                                extra_variance=0):
        '''_calculate_best_fit_m_b:
        
        Calculate the best-fitting m and b values for the linear model
        
        Args:
            bs_sample (6-array) - 6 element array of bootstrap properties
            prior_style (string) - Style of prior, either 'vR' or 'vT'
            trig_function (func) - Either np.cos or np.sin
            force_yint_zero (bool) - Should the y intercept be forced to be 0? 
                [False]
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
        for j in range(n_good_phi_bins):
            C[j,j] = phi_bin_v_err[j]**2 + extra_variance
        ###j
        C_inv = np.linalg.inv(C)
        
        # Check if the y intercept is forced to be 0
        if force_yint_zero:
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
        a single velocity only, adding an empiraically derived variance to 
        radial bins which do not match the overall trends particularly well.
        
        Args:
            extra_variance (n_R x 2 array) - Extra variance for vR and vT as a 
                function of radius.
        
        Returns:
        
        '''
        
        likelihood = np.ones( ( self.n_R_bins, self.n_phib_bins ) )
        
        # If vR is being used the assign the keyword to force the y-intercept 
        # to 0
        force_yint_zero = False
        if self.use_vR:
            force_yint_zero = self.force_yint_zero_vR
        ##fi
        
        for j in range( self.n_R_bins ):
            
            # Calculate the log likelihood of the tangential and radial
            # velocities as functions of phiB
            lin_likelihood = \
            self._calculate_phib_likelihood(self.bs_sample_1v[j], self.vel_1v,
                                            self.trig_fn_1v,
                                            extra_variance=extra_variance[j], 
                                            force_yint_zero=force_yint_zero )
            likelihood[j,:] = np.log(lin_likelihood)
            
        # Marginalize over all radii
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
                                         force_yint_zero=force_yint_zero )

            bs[j] = X[0]
            bs_err[j] = np.sqrt( SIG_X[0,0] )
            ms[j] = X[1]
            ms_err[j] = np.sqrt( SIG_X[1,1] )
            
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
            lin_likelihood_vT = \
            self._calculate_phib_likelihood(self.bs_sample_vT[j], 'vT', np.cos, 
                                            extra_variance=extra_variance[j,0] )
            likelihood_vT[j,:] = np.log(lin_likelihood_vT)
            
            lin_likelihood_vR = \
            self._calculate_phib_likelihood(self.bs_sample_vR[j], 'vR', np.sin, 
                                            extra_variance=extra_variance[j,1], 
                                            force_yint_zero=self.force_yint_zero_vR)
            likelihood_vR[j,:] = np.log(lin_likelihood_vR)
            
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
                                         force_yint_zero=self.force_yint_zero_vR )

            bs[j,0] = X_vT[0]
            bs[j,1] = X_vR[0]
            bs_err[j,0] = np.sqrt( SIG_X_vT[0,0] )
            bs_err[j,1] = np.sqrt( SIG_X_vR[0,0] )
            ms[j,0] = X_vT[1]
            ms[j,1] = X_vR[1]
            ms_err[j,0] = np.sqrt( SIG_X_vT[1,1] )
            ms_err[j,1] = np.sqrt( SIG_X_vR[1,1] )
            
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
                                    phi_lim=[-np.pi/2,np.pi/2]):
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
            trig_phis = np.linspace(phi_lim[0], phi_lim[1], num=100)
            if velocity_type == 'vR':
                axs[i].plot( trig_phis, 
                    self.b_vR[i]+self.m_vR[i]*np.sin(2*(trig_phis-self.phiB)))
            if velocity_type == 'vT':
                axs[i].plot( trig_phis, 
                    self.b_vT[i]+self.m_vT[i]*np.cos(2*(trig_phis-self.phiB)))
            ##fi
        
            # Add fiducials: bar, 0 line or tangential velocity curve
            axs[i].axvline( 25*(np.pi/180), linestyle='dotted', linewidth=1.0, 
                color='Red')
            X0, _ = self._generate_gaussian_prior_m_b(velocity_type, bin_R_cent)
            b0 = X0[0,0]
            axs[i].axhline( b0, linestyle='dashed', color='Black', linewidth=1.0 )
                
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
            axs[0].scatter( self.R_bin_cents, use_b, **plot_kws)
            axs[1].scatter( self.R_bin_cents, use_m, **plot_kws)
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

# def bootstrap_in_phi(gc_R, gc_phi, gc_vR, gc_vT, R_bin_cent, R_bin_size, 
#                         phi_bin_cents, phi_bin_size, n_bs):
#     '''bootstrap_in_phi:
# 
#     Perform a bootstrap determination of the average velocity in phi bins. 
#     Returns an array which can be unpacked wherever it is needed.
# 
#     Args:
#         gc_(R/phi/vR/vT) (float array) - Star properties
#         (R/phi)_bin_cent (float) - (Radial/Phi) bin center
#         (R/phi)_bin_size (float) - (Radial/Phi) bin size
#         n_bs (int) - Number of bootstrap samples
# 
#     Returns:
#         bs_sample (8-array) - Array of the bootstrap sample results for a 
#             single radius. It contains:
#             - R_bin_cent (float) - Radial bin center
#             - R_bin_size (float) - Radial bin size
#             - vR (float array) - vR as a function of phi
#             - vR_error (float array) - vR uncertainty as a function of phi
#             - vT (float array) - vT as a function of phi
#             - vT_error (float array) - vT uncertainty as a function of phi
#             - phi_bin_phi (float array) - phi bin centers
#             - phi_bin_phi_err (float array) - phi bin center uncertainty
#     '''
# 
#     n_phi_bins = len(phi_bin_cents)
# 
#     # Find all the points within this radial bin
#     stars_in_R_bin = np.where( ( gc_R < R_bin_cent + R_bin_size/2 ) & 
#                                ( gc_R > R_bin_cent - R_bin_size/2 ) )[0]
#     n_stars_in_R_bin = len(stars_in_R_bin)
#     gc_R_in_R_bin = gc_R[stars_in_R_bin]
#     gc_phi_in_R_bin = gc_phi[stars_in_R_bin]
#     gc_vR_in_R_bin = gc_vR[stars_in_R_bin]
#     gc_vT_in_R_bin = gc_vT[stars_in_R_bin]
# 
#     phi_bin_vR = np.array([])
#     phi_bin_vR_err = np.array([])
#     phi_bin_vT = np.array([])
#     phi_bin_vT_err = np.array([])
#     phi_bin_phi = np.array([])
#     phi_bin_phi_err = np.array([])
# 
#     # Loop over phi bins
#     for j in range(n_phi_bins):
# 
#         # Find all the points within this phi bin
#         stars_in_phi_bin = np.where( ( gc_phi_in_R_bin < phi_bin_cents[j] + phi_bin_size/2 ) &
#                                      ( gc_phi_in_R_bin > phi_bin_cents[j] - phi_bin_size/2 ) )[0]
#         n_stars_in_phi_bin = len(stars_in_phi_bin)
#         gc_R_in_phi_bin = gc_R_in_R_bin[stars_in_phi_bin]
#         gc_phi_in_phi_bin = gc_phi_in_R_bin[stars_in_phi_bin]
# 
#         gc_vR_in_phi_bin = gc_vR_in_R_bin[stars_in_phi_bin]
#         gc_vT_in_phi_bin = gc_vT_in_R_bin[stars_in_phi_bin]
# 
#         # If we have more than a certain number of stars then BS
#         bs_vR_avg_samps = np.array([])
#         bs_vT_avg_samps = np.array([])
#         bs_phi_avg_samps = np.array([])
# 
#         if n_stars_in_phi_bin > 10:
# 
#             # Loop over BS samples
#             for k in range(n_bs):
#                 sample = np.random.randint(0,n_stars_in_phi_bin,n_stars_in_phi_bin)
#                 bs_vR_avg_samps = np.append( bs_vR_avg_samps, np.average(gc_vR_in_phi_bin[sample]) )
#                 bs_vT_avg_samps = np.append( bs_vT_avg_samps, np.average(gc_vT_in_phi_bin[sample]) )
#                 bs_phi_avg_samps = np.append( bs_phi_avg_samps, np.average(gc_phi_in_phi_bin[sample]) )
#             ###k
# 
#             # Append the mean to the list of measurements
#             phi_bin_vR = np.append( phi_bin_vR, np.mean( bs_vR_avg_samps ) )
#             phi_bin_vR_err = np.append( phi_bin_vR_err, np.std( bs_vR_avg_samps ) )
#             phi_bin_vT = np.append( phi_bin_vT, np.mean( bs_vT_avg_samps ) )
#             phi_bin_vT_err = np.append( phi_bin_vT_err, np.std( bs_vT_avg_samps ) )
#             phi_bin_phi = np.append( phi_bin_phi, np.mean( bs_phi_avg_samps ) )
#             phi_bin_phi_err = np.append( phi_bin_phi_err, np.std( bs_phi_avg_samps ) )
# 
#         ##fi
#     ###j
# 
#     return [R_bin_cent, R_bin_size, phi_bin_vR, phi_bin_vR_err, phi_bin_vT,
#             phi_bin_vT_err, phi_bin_phi, phi_bin_phi_err]
# #def
# 
# def make_bootstrap_samples(gc_R, gc_phi, gc_vR, gc_vT, R_bin_cents, R_bin_size, 
#                            phi_bin_cents, phi_bin_size, n_bs, ):
#     '''make_bootstrap_sample:
# 
#     Make the bootstrap dataset
# 
#     Args:
#         gc_(R/phi/vR/vT) (float array) - Star properties
#         (R/phi)_bin_cents (float) - (Radial/Phi) bin centers
#         (R/phi)_bin_size (float) - (Radial/Phi) bin size
#         n_bs (int) - Number of bootstrap samples
# 
#     Returns:
#         bs_samples_vT (N-array) - Array of the vT bootstrap sample results for a 
#             single radius. It contains:
#             - R_bin_cent (float) - Radial bin center
#             - R_bin_size (float) - Radial bin size
#             - vT (float array) - vT as a function of phi
#             - vT_error (float array) - vT uncertainty as a function of phi
#             - phi_bin_phi (float array) - phi bin centers
#             - phi_bin_phi_err (float array) - phi bin center uncertainty
#         bs_samples_vR (N-array) - same but for vR
# 
#     '''
# 
#     bs_samples_vT = []
#     bs_samples_vR = []
# 
#     for i in range( len(R_bin_cents) ):
# 
#         # Make the bootstrap sample
#         bs_samp = bootstrap_in_phi( gc_R, gc_phi, gc_vR, gc_vT, R_bin_cents[i], 
#                                     R_bin_size, phi_bin_cents, phi_bin_size, 
#                                     n_bs)
# 
#         bs_samp_vR = [bs_samp[0], bs_samp[1], bs_samp[2], bs_samp[3], 
#                       bs_samp[6], bs_samp[7]]
#         bs_samp_vT = [bs_samp[0], bs_samp[1], bs_samp[4], bs_samp[5], 
#                       bs_samp[6], bs_samp[7]]
# 
#         bs_samples_vR.append(bs_samp_vR)
#         bs_samples_vT.append(bs_samp_vT)
#     ###i
# 
#     return bs_samples_vR, bs_samples_vT
# #def
# 
# def make_data_like_bootstrap_samples(R_bin_cents, phi, vT, vR, R_bin_size=None,
#                                      phi_err=0.01, vT_err=0.01, vR_err=0.01):
#     '''make_data_like_bootstrap_samples:
# 
#     Take a series of R/phi data and velocities and knit it into a form that 
#     looks like the bootstrap sample arrays, and which is appropriate for using 
#     in the linear model functions.
# 
#     Args:
#         radius (float array) - Array of radial bin centers
#         radius_bin_size (int) - Radial bin size
#         phi (float array) - Phi positions
#         vT (float array) - Tangential velocities
#         vR (float array) - Radial velocities
#         phi_err (float array) - Phi position errors [None]
#         vT_err (float array) - Tangential velocity errors [None]
#         vR_err (float array) - Radial velocity errors [None]
# 
#     Returns:
#         bs_samples_vT (N-array) - Array of the vT bootstrap sample results for a 
#             single radius. It contains:
#             - R_bin_cent (float) - Radial bin center
#             - R_bin_size (float) - Radial bin size
#             - vT (float array) - vT as a function of phi
#             - vT_error (float array) - vT uncertainty as a function of phi
#             - phi_bin_phi (float array) - phi bin centers
#             - phi_bin_phi_err (float array) - phi bin center uncertainty
#         bs_samples_vR (N-array) - same but for vR
#     '''
# 
#     # First find the number of unique radii
#     unique_R = np.unique( R_bin_cents )
#     n_R_bins = len(unique_R)
# 
#     if R_bin_size == None:
#         R_bin_size = np.sort(np.diff(unique_R))[0]
#     ##fi
# 
#     # Declare the arrays which hold the bootstrap samples
#     bs_samples_vT = []
#     bs_samples_vR = []
# 
#     # Loop over each unique radius and extract all the data for that bin
#     for i in range(n_R_bins):
# 
#         where_unique_R = np.where( R_bin_cents == unique_R[i] )[0]
#         this_R_bin_cent = unique_R[i]
#         this_R_bin_size = R_bin_size
#         this_phi = phi[where_unique_R]
#         this_vT = vT[where_unique_R]
#         this_vR = vR[where_unique_R]
# 
#         # Now generate the error arrays. Start of as zeros but can be filled. 
#         # Handles arrays of errors, but also constant errors and 
#         if phi_err == None:
#             this_phi_err = np.zeros_like(this_phi)
#         elif type(phi_err) == float or type(phi_err) == int:
#             this_phi_err = np.ones_like(this_phi)*phi_err
#         else:
#             this_phi_err = phi_err[where_unique_R]
#         ##ie
# 
#         if vT_err == None:
#             this_vT_err = np.zeros_like(this_phi)
#         elif type(vT_err) == float or type(vT_err) == int:
#             this_vT_err = np.ones_like(this_phi)*vT_err
#         else:
#             this_vT_err = vT_err[where_unique_R]
#         ##ie
# 
#         if vR_err == None:
#             this_vR_err = np.zeros_like(this_phi)
#         elif type(phi_err) == float or type(phi_err) == int:
#             this_vR_err = np.ones_like(this_phi)*vR_err
#         else:
#             this_vR_err = vR_err[where_unique_R]
#         ##ie
# 
#         # Make the velocity sample
#         vT_sample = [this_R_bin_cent, this_R_bin_size, this_vT, this_vT_err, 
#                      this_phi, this_phi_err]
#         vR_sample = [this_R_bin_cent, this_R_bin_size, this_vR, this_vR_err,
#                      this_phi, this_phi_err]
# 
#         bs_samples_vT.append(vT_sample)
#         bs_samples_vR.append(vR_sample)
#     ###i
#     return bs_samples_vT, bs_samples_vR
# #def
# 
# 
# def generate_gaussian_prior_m_b(prior_style, R_bin_cent, var_b, var_m, 
#                                 vt_prior_path = '/Users/JamesLane/Science/Projects/PhD/AST1501/data/generated/MWPotential2014_DF_vT_data.npy'):
#     '''generate_gaussian_prior_m_b:
# 
#     Make the parameters of the prior: the mean sample and the inverse variance. 
#     For both m and b.
# 
#     Args:
#         prior_style (string) - Either 'vT' or 'vR'
#         R_bin_cent (float) - Radius in kpc for the circular velocity curve
#         var_b (float) - Variance in m
#         var_m (float) - Variance in b
# 
#     Returns:
#         X0 (2x1 element array) - Mean of the gaussian prior
#         SIGMA_inv (2x2 array) - Inverse of the variance array
#     '''
#     # Generate the prior
#     if prior_style == 'vT':
#         prior_r, prior_vt = np.load(vt_prior_path)
#         where_close_to_bin = np.argmin( np.abs( R_bin_cent-prior_r ) )
#         b0 = prior_vt[where_close_to_bin]
#         # b0 = potential.vcirc(potential.MWPotential2014, R_bin_cent/8.0)*220.0
#         m0 = 0
#         X0 = np.zeros((2,1)) # Make a column vector
#         X0[0,0] = b0
#         X0[1,0] = m0
#         SIGMA_inv = np.array([[1/var_b,0],[0,1/var_m]])
#     elif prior_style == 'vR':
#         b0 = 0
#         m0 = 0
#         X0 = np.zeros((2,1)) # Make a column vector
#         X0[0,0] = b0
#         X0[1,0] = m0
#         SIGMA_inv = np.array([[1/var_b,0],[0,1/var_m]])
#     ##ie
#     return X0, SIGMA_inv
# #def
# 
# def calculate_phib_likelihood(R_bin_cent, R_bin_size, phib_bin_cents, 
#                                 bs_sample, var_b, var_m, prior_style,
#                                 force_yint_zero=False, 
#                                 trig_function=np.cos, 
#                                 extra_variance=0):
#     '''calculate_phib_likelihood:
# 
#     Calculate the likelihood as a function of the given phib's for a single 
#     radial bin and a series of phi bins.
# 
#     Args:
#         R_bin_cent (float) - Radial bin center
#         R_bin_size (float) - Radial bin size
#         phib_bin_cents (float) - Phib bin centers
#         bs_sample (6-array) - 6 element array of bootstrap properties
#         var_(b/m) (float) - Variance of (b/m) for the prior
#         prior_style (string) - Style of prior, either 'vR' or 'vT'
#         force_yint_zero (bool) - Should the y intercept be forced to be 0?
#         trig_function (func) - Probably either np.cos or np.sin
#         extra_variance (float) - Should an extra variance term be added to this 
#             radial bin?
# 
#     Returns:
#         Likelihood (float array) - Likelihood as a function of phib
#     '''
# 
#     # Unpack the bootstrap sample
#     _, _, phi_bin_v, phi_bin_v_err, phi_bin_phi, _ = bs_sample
# 
#     # Make the prior
#     X0, SIGMA_inv = generate_gaussian_prior_m_b(prior_style, R_bin_cent, 
#                                                 var_b, var_m)
# 
#     # Now make the vectors
#     n_good_phi_bins = len(phi_bin_v)
#     Y = np.zeros((n_good_phi_bins,1))
#     C = np.zeros((n_good_phi_bins,n_good_phi_bins))
#     Y[:,0] = phi_bin_v
#     for j in range(n_good_phi_bins):
#         C[j,j] = phi_bin_v_err[j]**2 + extra_variance
#     ###j
#     C_inv = np.linalg.inv(C)
# 
#     # Now loop over all possible values of phi B, making the vector 
#     # A for each and calculating the likelihood.
#     n_phib_bins = len(phib_bin_cents)
#     likelihood = np.zeros( n_phib_bins )
# 
#     for j in range(n_phib_bins):    
#         if force_yint_zero:
#             A = np.ones((n_good_phi_bins,1))
#             A[:,0] = trig_function( 2*( phi_bin_phi - phib_bin_cents[j] ) )
#         else:
#             A = np.ones((n_good_phi_bins,2))
#             A[:,1] = trig_function( 2*( phi_bin_phi - phib_bin_cents[j] ) )
#         ##ie
# 
#         # Now compute the vectors which form the solution
#         V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
#         V = np.linalg.inv( V_inv )
#         W = np.matmul( V , np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
#         U = np.linalg.multi_dot( [Y.T,C_inv,Y] ) + np.linalg.multi_dot( [X0.T,SIGMA_inv,X0] ) - np.linalg.multi_dot( [W.T,V_inv,W] )
#         likelihood[j] = np.sqrt( np.linalg.det(V)/np.linalg.det(C) ) * np.exp( -U/2 )
#     ###j
# 
#     return likelihood
# 
# #def
# 
# def calculate_best_fit_m_b(R_bin_cent, phib, bs_sample, 
#                             var_b, var_m, prior_style, 
#                             force_yint_zero=False, 
#                             trig_function=np.cos, 
#                             extra_variance=0):
#     '''calculate_best_fit_m_b:
# 
#     Calculate the best-fitting m and b values for the linear model
# 
#     Args:
#         R_bin_cent (float) - Radial bin center
#         R_bin_size (float) - Radial bin size
#         phib_bin_cents (float) - Phib bin centers
#         bs_sample (6-array) - 6 element array of bootstrap properties
#         var_(b/m) (float) - Variance of (b/m) for the prior
#         prior_style (string) - Style of prior, either 'vR' or 'vT'
#         force_yint_zero (bool) - Should the y intercept be forced to be 0?
#         trig_function (func) - Probably either np.cos or np.sin
#         extra_variance (float) - Should an extra variance term be added to this radial bin?
# 
#     Returns:
#         X (2-array) - Best-fitting m and b
#         SIG_X (2-array) - Uncertainty in the best-fit
# 
#     '''
# 
#     # Unpack the bootstrap sample
#     _, _, phi_bin_v, phi_bin_v_err, phi_bin_phi, _ = bs_sample
# 
#     # Make the prior
#     X0, SIGMA_inv = generate_gaussian_prior_m_b(prior_style, R_bin_cent, 
#                                                 var_b, var_m)
# 
#     # Now make the vectors
#     n_good_phi_bins = len(phi_bin_v)
#     Y = np.zeros((n_good_phi_bins,1))
#     C = np.zeros((n_good_phi_bins,n_good_phi_bins))
#     Y[:,0] = phi_bin_v
#     for j in range(n_good_phi_bins):
#         C[j,j] = phi_bin_v_err[j]**2 + extra_variance
#     ###j
#     C_inv = np.linalg.inv(C)
# 
#     # Check if the y intercept is forced to be 0
#     if force_yint_zero:
#         A = np.ones((n_good_phi_bins,1))
#         A[:,0] = trig_function( 2*( phi_bin_phi - phib ) )
#     else:
#         A = np.ones((n_good_phi_bins,2))
#         A[:,1] = trig_function( 2*( phi_bin_phi - phib ) )
#     ##ie
# 
#     V_inv = np.linalg.multi_dot( [ A.T, C_inv, A ] ) + SIGMA_inv
#     V = np.linalg.inv( V_inv )
#     W = np.matmul( V , np.linalg.multi_dot( [A.T,C_inv,Y] ) + np.linalg.multi_dot( [SIGMA_inv,X0] ) )
# 
#     X = np.matmul( np.linalg.inv( np.linalg.multi_dot([A.T,np.linalg.inv(C),A]) ), np.linalg.multi_dot([A.T,np.linalg.inv(C),Y]) )
#     SIG_X = np.linalg.inv( np.linalg.multi_dot([A.T,np.linalg.inv(C),A]) )
# 
#     return W, V
# #def
# 
# def calculate_variance_data_model(bs_sample, m, b, phib, trig_function):
#     '''calculate_variance_data_model:
# 
#     Calculate the variance of the differences between the best-fitting model and the 
#     data.
# 
#     Args:
#         bs_sample 
#         m (float) - Best-fitting m
#         b (float) - Best-fitting b
#         phib (float) - Best-fitting phib
#         trig_function (function) - Either np.sin or np.cos
# 
#     Returns:
#         variance (float) - variance of the difference between the model and data
# 
#     '''
# 
#     # Unpack the bootstrap sample
#     _, _, phi_bin_v, _, phi_bin_phi, _ = bs_sample
# 
#     # Calculate the model at the locations where there is data
#     model_v = b + m*trig_function(2*(phi_bin_phi-phib))
# 
#     # Calculate the standard deviation of the differences between model and data
#     sd_data_model = np.std(model_v-phi_bin_v)
# 
#     return np.square(sd_data_model)
# 
# #def
# 
# def radial_velocity_known_m_b_phi(R_bin_cents, R_bin_size, phi_range, 
#                                     phi_bin_size, gc_R, gc_phi, gc_v,
#                                     ms, bs, phib, trig_function, 
#                                     phi_bin_size_in_arc=True, zero_vels=False
#                                ):
#     '''radial_velocity_known_m_b_phi:
# 
#     Plot the velocity trends in the data as a function of radius
# 
#     '''
# 
#     n_R = len(R_bin_cents)
# 
#     # Declare the figure
#     fig = plt.figure( figsize=(15,n_R*3) )
#     axs = fig.subplots(nrows=n_R, ncols=3)
# 
#     # Loop over all radii
#     for i in range( n_R ):
# 
#         # Select the stars in this bin
#         stars_in_bin = np.where( (gc_R > (R_bin_cents[i]-R_bin_size/2) ) & 
#                                  (gc_R < (R_bin_cents[i]+R_bin_size/2) ) )[0]
#         gcR_in_bin = gc_R[stars_in_bin]
#         gcv_in_bin = gc_v[stars_in_bin]
#         gcphi_in_bin = gc_phi[stars_in_bin]
# 
#         if phi_bin_size_in_arc:
#             # Bin the Gaia data in arc
#             arc_min = phi_range[0]*R_bin_cents[i]
#             arc_max = phi_range[1]*R_bin_cents[i]
#             phi_bin_cents = np.arange( arc_min, arc_max, phi_bin_size)
#             phi_bin_cents += ( ( arc_max - arc_min ) % phi_bin_size )/2
#             phi_bin_cents /= R_bin_cents[i]
#         else:
#             phi_bin_cents = np.arange( phi_range[0], phi_range[1], phi_bin_size )
#             phi_bin_cents += ( ( phi_range[1] - phi_range[0] ) % phi_bin_size )/2
#         ##ie
# 
#         # Make the bin edges and bin velocity, R, and number in phi
#         phi_bin_edges = np.append( phi_bin_cents-np.diff(phi_bin_cents)[0], 
#                                    phi_bin_cents[-1]+np.diff(phi_bin_cents)[0] )
#         binned_v, _, _ = binned_statistic(gcphi_in_bin, gcv_in_bin, bins=phi_bin_edges, statistic='mean')
#         binned_R, _, _ = binned_statistic(gcphi_in_bin, gcR_in_bin, bins=phi_bin_edges, statistic='mean')        
# 
#         # Make a number histogram to examine whether there is enough stars for 
#         # a valid measurement
#         binned_n, _, = np.histogram(gcphi_in_bin, bins=phi_bin_edges)
#         binned_n = binned_n.astype('float')
#         min_N = 20
#         where_low_bin_numbers = np.where(binned_n < min_N)
#         binned_v[ where_low_bin_numbers ] = np.nan
#         binned_R[ where_low_bin_numbers ] = np.nan
#         binned_n[ where_low_bin_numbers ] = np.nan
# 
#         # Find where there was data
#         where_data = np.where( np.isfinite(binned_v) )
#         where_no_data = np.where( np.isnan(binned_v) )
#         binned_v[ where_no_data ] = np.nan
#         binned_R[ where_no_data ] = np.nan
#         binned_n[ where_no_data ] = np.nan
# 
#         # Subtract off mean where non-zero
#         if zero_vels:
#             binned_v -= np.nanmean( binned_v[where_data] )
# 
#         axs[i,0].plot( phi_bin_cents, binned_v, linewidth=0.5, color='Black' )
#         axs[i,1].plot( phi_bin_cents, binned_R, linewidth=0.5, color='Black' )
#         axs[i,2].plot( phi_bin_cents, binned_n, linewidth=0.5, color='Black' )
#         axs[i,0].scatter( phi_bin_cents, binned_v, s=5, color='Black' )
#         axs[i,1].scatter( phi_bin_cents, binned_R, s=5, color='Black' )
#         axs[i,2].scatter( phi_bin_cents, binned_n, s=5, color='Black' )
# 
#         axs[i,2].set_yscale("log", nonposy='clip')
# 
#         axs[i,0].annotate( r'$R_{cen}=$'+str(R_bin_cents[i])+' kpc', xy=(0.05,0.8), xycoords='axes fraction' )
#         axs[i,1].set_ylim( R_bin_cents[i]-R_bin_size/2, R_bin_cents[i]+R_bin_size/2 )
# 
#         axs[i,0].set_xlim( phi_range[0], phi_range[1] )
#         axs[i,1].set_xlim( phi_range[0], phi_range[1] )
#         axs[i,2].set_xlim( phi_range[0], phi_range[1] )
#         axs[i,2].set_ylim( 1, 5000)
# 
#         # Add a bar
#         axs[i,0].axvline( 25*(np.pi/180), linestyle='dashed', linewidth=0.5, color='Red' )
#         axs[i,1].axvline( 25*(np.pi/180), linestyle='dashed', linewidth=0.5, color='Red' )
#         axs[i,2].axvline( 25*(np.pi/180), linestyle='dashed', linewidth=0.5, color='Red' )
# 
#         axs[i,0].axhline( np.nanmean( binned_v[where_data] ), linestyle='dashed', linewidth=0.5 )
#         axs[i,1].axhline( R_bin_cents[i], linestyle='dashed', linewidth=0.5 )
#         axs[i,2].axhline( min_N, linestyle='dashed', linewidth=0.5 )
# 
#         axs[i,1].set_ylabel(r'$\bar{R}$ [kpc]')
#         axs[i,2].set_ylabel(r'$N$')
# 
#         axs[i,0].set_xlabel(r'$\phi$')
#         axs[i,1].set_xlabel(r'$\phi$')
#         axs[i,2].set_xlabel(r'$\phi$')
# 
#         # Plot the best-fitting amplitude
#         trig_phis = np.linspace(-np.pi/2, np.pi/2, num=100)
# 
#         axs[i,0].plot( trig_phis, bs[i]+ms[i]*trig_function(2*(trig_phis-phib)) )
# 
#     return fig, axs
# #def
# 
# def iterate_noise_model_2_velocities(R_bin_cents, R_bin_size, phib_bin_cents, 
#                                      bs_samples_vT, bs_samples_vR, var_arr, 
#                                      force_yint_zero_vR, extra_variance, 
#                                      force_phiB=None):
#     '''iterate_noise_model_2_velocities:
# 
#     Iterate over the calculation of the best-fitting linear model, adding an 
#     empiracally derived variance to radial bins which do not match the overall
#     trends particularly well.
# 
#     Args:
# 
#     Returns:
# 
#     '''
# 
#     # Unpack the variances
#     var_b_vT, var_m_vT, var_b_vR, var_m_vR = var_arr
# 
#     n_R_bins = len(R_bin_cents)
#     n_phib_bins = len(phib_bin_cents)
# 
#     # Make an array to store the log likelihoods
#     store_likelihood_vT = np.ones( ( n_R_bins, n_phib_bins ) )
#     store_likelihood_vR = np.ones( ( n_R_bins, n_phib_bins ) )
# 
#     # Loop over the radial bins and calculate the likelihood as a function of 
#     # phiB for both tangential and radial velocities. 
#     for j in range( n_R_bins ):
# 
#         # Calculate the log likelihood of the tangential and radial velocities
#         likelihood_vT = calculate_phib_likelihood(R_bin_cents[j], R_bin_size,  
#                                                   phib_bin_cents, bs_samples_vT[j],
#                                                   var_b_vT, var_m_vT, 'vT', 
#                                                   force_yint_zero=False, 
#                                                   trig_function=np.cos, 
#                                                   extra_variance=extra_variance[j,0] )
#         store_likelihood_vT[j,:] = np.log(likelihood_vT)
#         likelihood_vR = calculate_phib_likelihood(R_bin_cents[j], R_bin_size, 
#                                                   phib_bin_cents, bs_samples_vR[j], 
#                                                   var_b_vR, var_m_vR, 'vR', 
#                                                   force_yint_zero=force_yint_zero_vR, 
#                                                   trig_function=np.sin, 
#                                                   extra_variance=extra_variance[j,1])
#         store_likelihood_vR[j,:] = np.log(likelihood_vR)
# 
#     ###j
# 
#     # Marginalize over all radii
#     prod_likelihood_vT = np.sum(store_likelihood_vT, axis=0)
#     prod_likelihood_vR = np.sum(store_likelihood_vR, axis=0)
#     prod_likelihood_both = prod_likelihood_vR + prod_likelihood_vT
# 
#     # Determine the best-fitting phib
#     phib_max_likelihood_arg = np.argmax( prod_likelihood_both )
#     phib_max_likelihood = phib_bin_cents[phib_max_likelihood_arg]
# 
#     if force_phiB != None:
#         phib_max_likelihood = force_phiB
#     ##fi
# 
#     ms = np.zeros( (n_R_bins,2) )
#     bs = np.zeros( (n_R_bins,2) )
#     ms_err = np.zeros( (n_R_bins,2) )
#     bs_err = np.zeros( (n_R_bins,2) )
#     variance_model_data = np.zeros((n_R_bins,2))
# 
#     # Loop over radial bins, calculate the 
#     for j in range( n_R_bins ):
# 
#         # Now determine the best-fitting m and b
#         X_vT, SIG_X_vT = calculate_best_fit_m_b(R_bin_cents[j], 
#                                                 phib_max_likelihood, 
#                                                 bs_samples_vT[j], var_b_vT,
#                                                 var_m_vT, 'vT',
#                                                 force_yint_zero=False, 
#                                                 trig_function=np.cos, 
#                                                 extra_variance=extra_variance[j,0] )
#         X_vR, SIG_X_vR = calculate_best_fit_m_b(R_bin_cents[j], 
#                                                 phib_max_likelihood, 
#                                                 bs_samples_vR[j], var_b_vR, 
#                                                 var_m_vR, 'vR',
#                                                 force_yint_zero=force_yint_zero_vR, 
#                                                 trig_function=np.sin, 
#                                                 extra_variance=extra_variance[j,1])
# 
#         bs[j,0] = X_vT[0]
#         bs[j,1] = X_vR[0]
#         bs_err[j,0] = np.sqrt( SIG_X_vT[0,0] )
#         bs_err[j,1] = np.sqrt( SIG_X_vR[0,0] )
#         ms[j,0] = X_vT[1]
#         ms[j,1] = X_vR[1]
#         ms_err[j,0] = np.sqrt( SIG_X_vT[1,1] )
#         ms_err[j,1] = np.sqrt( SIG_X_vR[1,1] )
# 
#         # Now calculate the standard deviation of the difference between the data and the model
#         variance_model_data[j,0] = calculate_variance_data_model(bs_samples_vT[j], ms[j,0], bs[j,0], 
#                                                                  phib_max_likelihood, 
#                                                                  trig_function=np.cos)
#         variance_model_data[j,1] = calculate_variance_data_model(bs_samples_vR[j], ms[j,1], bs[j,1], 
#                                                                  phib_max_likelihood, 
#                                                                  trig_function=np.sin)
#     ###j 
# 
#     return store_likelihood_vT, store_likelihood_vR, prod_likelihood_vT, prod_likelihood_vR,\
#             prod_likelihood_both, phib_max_likelihood, bs, ms, bs_err, ms_err, variance_model_data
# #def
# 
# def run_iterating_linear_model(R_bin_cents, R_bin_size, phib_bin_cents, 
#                                 bs_samples_vT, bs_samples_vR, var_arr, 
#                                 force_yint_zero_vR, n_iterate, 
#                                 plot_results=False, force_phiB=None):
# 
#     n_R_bins = len(R_bin_cents)
# 
#     alpha_low = 0.25
#     alpha_increment = (1-alpha_low)/(n_iterate-1)
# 
#     if plot_results:
#         fig1 = plt.figure( figsize=(5,(n_R_bins+1)*2) )
#         axs1 = fig1.subplots( nrows=n_R_bins+1, ncols=1 )
# 
#         fig2 = plt.figure( figsize=(12,6) ) 
#         axs2 = fig2.subplots( nrows=2, ncols=2 )
#     ##fi
# 
#     results_arr = []
#     extra_variance = np.zeros((n_R_bins,2))
# 
#     for i in range( n_iterate ):
# 
#         store_likelihood_vT, store_likelihood_vR, prod_likelihood_vT, prod_likelihood_vR,\
#         prod_likelihood_both, phib_max_likelihood, bs, ms, bs_err, ms_err, variance_model_data\
#         = iterate_noise_model_2_velocities(R_bin_cents, R_bin_size,  
#                                            phib_bin_cents, bs_samples_vT, 
#                                            bs_samples_vR, var_arr, 
#                                            force_yint_zero_vR,
#                                            extra_variance, force_phiB=force_phiB)
# 
#         extra_variance = variance_model_data
# 
#         output_results = [store_likelihood_vT, store_likelihood_vR, 
#             prod_likelihood_vT, prod_likelihood_vR, prod_likelihood_both, 
#             phib_max_likelihood, bs, ms, bs_err, ms_err, variance_model_data]
#         results_arr.append(output_results)
# 
#         if plot_results:
#             for j in range( n_R_bins ):
#                 axs1[j].plot( phib_bin_cents, store_likelihood_vT[j,:], marker='o', markersize=5, color='DodgerBlue', 
#                               alpha=alpha_low + i*alpha_increment )
#                 axs1[j].plot( phib_bin_cents, store_likelihood_vR[j,:], marker='o', markersize=5, color='Red', 
#                               alpha=alpha_low + i*alpha_increment )
#                 axs1[j].annotate('R='+str(R_bin_cents[j])+' kpc', fontsize=14, xy=(0.7,0.85), xycoords='axes fraction')
#                 axs1[j].set_ylabel(r'$\ln \mathcal{L}$', fontsize=16)
#                 axs1[j].axvline( np.pi*25/180, color='Black', linestyle='dashed', alpha=0.5 )
#                 axs1[j].tick_params(labelbottom='off')
#             ###i
# 
#             axs2[0,0].scatter(R_bin_cents, bs[:,0], edgecolor='DodgerBlue', facecolor='DodgerBlue', 
#                             alpha=alpha_low + i*alpha_increment )
#             axs2[0,1].scatter(R_bin_cents, ms[:,0], edgecolor='DodgerBlue', facecolor='DodgerBlue', 
#                             alpha=alpha_low + i*alpha_increment )
#             axs2[1,0].scatter(R_bin_cents, bs[:,1], edgecolor='Red', facecolor='Red', 
#                             alpha=alpha_low + i*alpha_increment )
#             axs2[1,1].scatter(R_bin_cents, ms[:,1], edgecolor='Red', facecolor='Red', 
#                             alpha=alpha_low + i*alpha_increment )
# 
#             if i == n_iterate-1:
#                 axs2[0,0].errorbar(R_bin_cents, bs[:,0], fmt='o', markeredgecolor='Black', markerfacecolor='DodgerBlue', 
#                                    alpha=alpha_low + i*alpha_increment, yerr=bs_err[:,0], ecolor='Black')
#                 axs2[0,1].errorbar(R_bin_cents, ms[:,0], fmt='o', markeredgecolor='Black', markerfacecolor='DodgerBlue', 
#                                    alpha=alpha_low + i*alpha_increment, yerr=ms_err[:,0], ecolor='Black')
#                 axs2[1,0].errorbar(R_bin_cents, bs[:,1], fmt='o', markeredgecolor='Black', markerfacecolor='Red', 
#                                    alpha=alpha_low + i*alpha_increment, yerr=bs_err[:,1], ecolor='Black')
#                 axs2[1,1].errorbar(R_bin_cents, ms[:,1], fmt='o', markeredgecolor='Black', markerfacecolor='Red', 
#                                    alpha=alpha_low + i*alpha_increment, yerr=ms_err[:,1], ecolor='Black')
#             ###i
# 
#             if i == 0:
#                 axs1[0].plot([], [], color='DodgerBlue', label=r'$v_{T}$')
#                 axs1[0].plot([], [], color='Red', label=r'$v_{R}$')
#                 axs1[0].legend(loc=(0.3,0.6))
#                 axs1[-1].annotate('Product', fontsize=14, xy=(0.7,0.85), xycoords='axes fraction')
#                 axs1[-1].set_ylabel(r'$\ln \mathcal{L}$', fontsize=16)
#                 axs1[-1].axvline( np.pi*25/180, color='Black', linestyle='dashed', alpha=0.5 )
#                 axs1[-1].set_xlabel(r'$\Phi_{gc}$ (radians)', fontsize=16)
# 
#                 axs2[0,0].set_xlabel('R [kpc]')
#                 axs2[0,0].set_ylabel(r'b $(v_{c})$ [km/s]')
#                 axs2[0,1].set_xlabel('R [kpc]')
#                 axs2[0,1].set_ylabel(r'|m| (A) [km/s]')
#                 axs2[1,1].set_xlabel('R [kpc]')
#                 axs2[1,1].set_ylabel(r'|m| (A) [km/s]')
#                 axs2[1,0].set_xlabel('R [kpc]')
#                 axs2[1,0].set_ylabel(r'b $(v_{c})$ [km/s]')
#             ###j
# 
#             axs1[-1].plot( phib_bin_cents, prod_likelihood_vT, marker='s', markersize=5, color='DodgerBlue', 
#                            alpha=alpha_low + i*alpha_increment)
#             axs1[-1].plot( phib_bin_cents, prod_likelihood_vR, marker='s', markersize=5, color='Red', 
#                            alpha=alpha_low + i*alpha_increment)
#             axs1[-1].plot( phib_bin_cents, prod_likelihood_both, marker='s', markersize=5, color='Purple', 
#                            alpha=alpha_low + i*alpha_increment)
# 
#             # Overplot the circular velocity curve
#             vt_prior_path = '/Users/JamesLane/Science/Projects/PhD/AST1501/data/generated/MWPotential2014_DF_vT_data.npy'
#             vcirc_Rs, vcirc = np.load(vt_prior_path)
#             # vcirc_Rs = np.linspace( np.min(R_bin_cents) , np.max(R_bin_cents) ,
#             #                         num=100)
#             # vcirc = potential.vcirc(potential.MWPotential2014,R=vcirc_Rs/8)*220 # km/s
#             axs2[0,0].plot(vcirc_Rs, vcirc, linewidth=1.0, color='Black', alpha=0.5, linestyle='dashed')
#         ##fi
# 
#     if plot_results:
#         fig1.subplots_adjust(hspace=0)
#         return results_arr, fig1, axs1, fig2, axs2
#     else:
#         return results_arr
#     ##ie
# #def
# 
# def fit_linear_model_df_data(filename, phib_bin_cents, var_arr, 
#                                 force_yint_zero_vR, n_iterate, force_phiB=None ):
#     '''fit_linear_model_df_data:
# 
#     Convenience function to take the filename of some DF data and determine the 
#     linear fit
# 
#     Args:
#         filename
# 
#     Returns:
#         results_arr
#     '''
# 
#     # Read the data, assume data is of the form: [R,phi,x,y,vR,vR_disp,vT,vT_disp]
#     data = np.load(filename).T.astype(float)
#     R,phi,_,_,vR,_,vT,_ = data
#     R_bin_cents = np.sort(np.unique(R))
#     R_bin_size = np.diff( np.sort( np.unique(R) ) )[0]
# 
#     bs_samples_vT, bs_samples_vR = make_data_like_bootstrap_samples(R,phi,vT,vR)
#     results = run_iterating_linear_model(R_bin_cents, R_bin_size, phib_bin_cents, 
#                                          bs_samples_vT, bs_samples_vR, var_arr, 
#                                          force_yint_zero_vR, n_iterate, 
#                                          force_phiB=force_phiB)
# 
#     return R_bin_cents, results