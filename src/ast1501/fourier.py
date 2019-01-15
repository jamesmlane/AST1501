# ----------------------------------------------------------------------------
#
# TITLE - fourier.py
# AUTHOR - James Lane
# PROJECT - AST1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Routines for Fourier transformation
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy

# ----------------------------------------------------------------------------

def psd(arr,dx,pad=True,return_phase=True):
    '''
    psd:
    
    Calculate the power spectrum of an array
    
    Args:
        arr (array) - 1D data array
        dx (float) - spacing of arr
        pad (bool) - Pad the data with N+1 zeros? [True]
        return_phase (bool) - Return the phase information? [True]
        
    Returns:
        ps (array) - The power spectrum of the array
    '''
    # First copy the data
    arr = copy.deepcopy(arr)
    
    # Remove NaN
    arr[ np.isnan(arr) ] = 0.
    
    # Account for padding
    if pad:
        n= 2*len(arr)+1
    else:
        n= len(arr)
    ##ie
    
    # Take the 1D FFT. n controls whether padding will occur
    arr_fft = np.fft.fftshift(np.fft.fft(arr,n=n))
    
    # Get the phase information
    if return_phase:
        phase = np.angle( arr_fft )
    ##fi
    
    # Then calculate the periodogram estimate of the power spectrum
    ret = np.abs(arr_fft)**2. + np.abs(arr_fft[::-1])**2.
    
    pdb.set_trace()
    
    # Correct the 0 order term and account for the padding by multiplying the 
    # power by 2
    if pad:
        ret[int(len(arr))] *= 0.5
        ret *= 2
    else:
        ret[int(len(arr)/2)] *= 0.5
    ##ie
    
    # Output
    if return_phase:
        return (np.fft.fftshift(np.fft.fftfreq(n,dx)),
                ret/n**2.,
                phase)
    else:
        return (np.fft.fftshift(np.fft.fftfreq(n,dx)),
                ret/n**2.)
    ##ie

def psd2d(image,pad=True):
    '''
    psd2d:
    
    Calculate the 2D fourier transform of an image and determine it's 2D 
    power spectrum using the periodogram estimate.
    
    Args:
        image (ndarray) - The data image
        pad (bool) - Pad the data with N+1 zeros? [True]
    
    Returns:
        ps (ndarray) - The 2D power spectrum    
    '''
    # First copy the data
    image = copy.deepcopy(image)
    
    #Now rm NaN
    image[ np.isnan(image) ]= 0.
    
    # pad N+1 0s
    # if pad:
    #     image = np.pad(image, 
    #         pad_width=( (0,int(image.shape[0]+1)), (0,int(image.shape[1]+1)) ), 
    #         mode='constant', constant_values=0)
    # ##fi
    
    # Compute the FFT
    if pad:
        image_fft= np.fft.fftshift(np.fft.fft2(image,
                                                     s=(2*image.shape[0]+1,
                                                        2*image.shape[1]+1)))
    else:
        image_fft= np.fft.fftshift(np.fft.fft2(image,
                                                     s=(image.shape[0],
                                                        image.shape[1])))
                                                        
    #Calculate the periodogram estimate of the power spectrum
    ret= np.abs(image_fft)**2.\
        +np.abs(image_fft[::-1,::-1])**2.
        
    # Return the results
    if pad:
        ret[image.shape[0],image.shape[1]]*= 0.5 #correct zero order term
        return ret/(2.*image.shape[0]+1)**2./(2.*image.shape[1]+1)**2.*4.
    else:
        ret[image.shape[0]/2,image.shape[1]/2]*= 0.5 #correct zero order term
        return ret/image.shape[0]**2./image.shape[1]**2.*16.
#def

def psd1d(image,dx,binsize=1.,pad=True):
    '''
    psd1d:
    
    Calculate the 1D, azimuthally averaged power spectrum of an image
    
    Args:
        image ( NxM array ) - the data image
        dx (float) - The spacing in the X and Y directions
        binsize (float) - radial binsize in terms of the Nyquist frequency [1.0]
        pad (bool) - Pad the input image with N+1 zeros? [True]
    
    Returns
        ps1d (array) - The 1D power spectrum
    '''
    
    image = copy.deepcopy(image)
    
    #First subtract DC component, taking into account NaN
    image -= np.mean( image[ np.where( ~np.isnan(image) ) ] )
    
    #Calculate the 2D PSD
    ret2d= psd2d(image,pad=pad)
    nr,radii,ret= azimuthalAverage(ret2d,returnradii=False,binsize=binsize,
                                    dx=1./2./dx/image.shape[0],
                                    dy=1./2./dx/image.shape[1],
                                    interpnan=False,
                                    return_nr=True)
    return (radii/2.**pad/dx/(image.shape[0]/2.-0.),ret,ret/np.sqrt(nr))
#def

def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False, dx= None, dy= None,
        binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None,
        mask=None ):
    '''
    azimuthalAverage:
    
    Calculate the azimuthally averaged radial profile.
    
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.
    
    Args:
        image (array) - The data image
        center (2-array) - The [x,y] pixel coordinates used as the center. 
            If None, then use the center of the image (including fractional 
            pixels). [None]
        stddev (bool) - Return the azimuthal standard deviation instead of the 
            average [False]
        returnradii (bool) - return (radii_array,radial_profile) [False]
        return_nr [bool] - return number of pixels per radius *and* 
            radius [False]
        dx,dy (float) - spacing in x and y (must either both be set or not set 
            at all) [None,None]
        binsize (float) - size of the averaging bin.  Can lead to strange 
            results if non-binsize factors are used to specify the center and the 
            binsize is too large [0.5]
        weights (array) - can do a weighted average instead of a simple average 
            if this keyword parameter is set. weights.shape must = image.shape.  
            weighted stddev is undefined, so don't set weights and 
            stddev. [None]
        steps (bool) - if specified, will return a double-length bin array and 
            radial profile so you can plot a step-form radial profile 
            (which more accurately represents what's going on) [False]
        interpnan (bool) - Interpolate over NAN values, i.e. bins where there 
            is no data?
        left,right (float,float) - passed to interpnan; they set the 
            extrapolated values [None,None]
        mask - can supply a mask (boolean array same size as image with True 
            for OK and False for not) to average over only select data. [None]
    '''
    
    # Calculate the indices from the image
    x, y = np.indices(image.shape)
    if not image.shape[0] == image.shape[1]:
        y = y.astype('float')
        y *= (image.shape[0]-1)/(image.shape[1]-1.)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape,dtype='bool')

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)  
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0,maxbin,nbins+1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:]+bins[:-1])/2.0

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    #nr = np.bincount(whichbin)[1:]
    nr = np.histogram(r,bins)[0]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat,bins)
        # This method is still very slow; is there a trick to do this with histograms? 
        radial_prof = np.array([image.flat[mask.flat*(whichbin==b)].std() for b in xrange(1,nbins+1)])
    else: 
        radial_prof = np.histogram(r, bins, range=[0.,r.max()],weights=(image*weights*mask))[0] / np.histogram(r, bins, weights=(mask*weights),range=[0.,r.max()])[0]

    if interpnan:
        radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

    if steps:
        xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
        yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
        return xarr,yarr
    elif returnradii: 
        return bin_centers,radial_prof
    elif return_nr:
        return nr,bin_centers,radial_prof
    else:
        return radial_prof
#def

# ----------------------------------------------------------------------------
