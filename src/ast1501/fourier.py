# ----------------------------------------------------------------------------
#
# TITLE - fourier.py
# AUTHOR - James Lane
# PROJECT - AST1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Routines to make the fourier transform
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy

# ----------------------------------------------------------------------------

# 
def calculate_power_spectrum2d(data, d1, d2):
    '''
    calculate_power_spectrum:
    
    Calculate the Fourier coefficients of a data array. Similar to Bovy+2015?
    
    Args:
        data (NxM float array) - Input data. Must be 2D
        d1 (float) - bin size in the 1st dimension of data
        d2 (float) - bin size in the 2nd dimension of data
    
    Returns:
        pk (float array) - 
    '''

    # Make sure the array is the correct shape
    if len(data.shape) != 2: 
        raise RuntimeError('Input array must be 2D')
    ##fi
    
    # Subtract the DC component
    image -= np.mean(data[True-np.isnan(data)])
    
    # Axis lengths
    n = data.shape[0]
    m = data.shape[1]
    
    # Pad the input array with N+1 and M+1 zeros
    data_pad = np.pad(data, 
        pad_width=( (0,int(n+1)), (0,int(m+1)) ), 
        mode='constant', constant_values=0)
    
    # Calculate the 2D Fourier transform and the power
    ft2d = np.fft.fft2(data_pad)
    p2d = np.abs(ft2d) / ( n * m )**2
    
    # Shift low frequency components to center
    p2d_shift = np.fft.fftshift(p2d)
    
    # Calculate the radially averaged power, first make arrays 
    # with all of the frequencies
    # kx = np.arange(0,n+1,1.0) / (n*d1)
    # ky = np.arange(0,m+1,1.0) / (m*d2)
    kx = np.fft.fftfreq(n,d1)
    ky = np.fft.fftfreq(m,d2)
    kx2d, ky2d = np.meshgrid(kx,ky,indexing='ij')
    
    # Now shift them by the frequency of the middle bin
    kx2d -= kx[int(round(n/2))]
    ky2d -= ky[int(round(m/2))]
    
    # Now calculate the squared sum of the wavenumber
    k2d = np.sqrt(np.square(kx2d) + np.square(ky2d))
    
    # Now do the radial averaging. The minimum k value will be a combination of
    # the two 1st frequencies. The max will be a combination of the maximum 
    # frequencies.
    k_min = 0.1 # Not 0
    k_max = np.sqrt( np.square( np.max(kx) ) + np.square( np.max(ky) ) )
    
    # Now make a range for k
    k_range = np.linspace( k_min, k_max, 10 )
    dk = np.average(np.diff(k_range)) # Will all be the same
    pk = np.zeros(len(k_range))
    
    # pdb.set_trace()
    
    for i in range(len(pk)):
        where_in_k_bin = np.where(  ( (k2d) < k_range[i]+dk/2 ) & 
                                    ( (k2d) > k_range[i]-dk/2 ))
        if len(where_in_k_bin[0]) == 0: continue
        pk[i] = 4*np.pi**2 * np.average( p2d_shift[where_in_k_bin] )
    ###i
            
    return k_range,pk,[k2d,]
#def

def psd2d(image,pad=True):
    """
    NAME:
       psd2d
    PURPOSE:
       Calculate the 2D power spectrum of an image
    INPUT:
       image - the image [NxM]
    OUTPUT:
       the 2D power spectrum using definitions of NR 13.4 eqn. (13.4.5)
    HISTORY:
       2014-06-06 - Written - Bovy (IAS)
    """
    # First copy
    image = copy.deepcopy(image)
    
    #First rm NaN
    image[np.isnan(image)]= 0.
    #First take the 2D FFT
    
    # pad 2n+1 0s
    if pad:
        image = np.pad(image, 
            pad_width=( (0,int(image.shape[0]+1)), (0,int(image.shape[1]+1)) ), 
            mode='constant', constant_values=0)
    ##fi
    
    
    if pad:
        image_fft= np.fft.fftshift(np.fft.fft2(image,
                                                     s=(2*image.shape[0]+1,
                                                        2*image.shape[1]+1)))
    else:
        image_fft= np.fft.fftshift(np.fft.fft2(image,
                                                     s=(image.shape[0],
                                                        image.shape[1])))
    #Then calculate the periodogram estimate of the power spectrum
    ret= np.abs(image_fft)**2.\
        +np.abs(image_fft[::-1,::-1])**2.
    if pad:
        ret[image.shape[0],image.shape[1]]*= 0.5 #correct zero order term
        return ret/(2.*image.shape[0]+1)**2./(2.*image.shape[1]+1)**2.*4.
    else:
        ret[image.shape[0]/2,image.shape[1]/2]*= 0.5 #correct zero order term
        return ret/image.shape[0]**2./image.shape[1]**2.*16.
#def

def psd1d(image,dx,binsize=1.,pad=True):
    """
    NAME:
       psd1d
    PURPOSE:
       Calculate the 1D, azimuthally averaged power spectrum of an image
    INPUT:
       image - the image [NxM]
       dx- spacing in X and Y directions
       binsize= (1) radial binsize in terms of Nyquist frequency
    OUTPUT:
       the 1D power spectrum using definitions of NR 13.4 eqn. (13.4.5)
    HISTORY:
       2014-06-06 - Written - Bovy (IAS)
    """
    
    image = copy.deepcopy(image)
    
    #First subtract DC component
    image -= np.mean(image[ np.where( ~np.isnan(image) ) ])
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
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    dx, dy- spacing in x and y (must either both be set or not set at all)
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    
    """
    # Calculate the indices from the image
    x, y = np.indices(image.shape)
    if not image.shape[0] == image.shape[1]:
        y= y.astype('float')
        y*= (image.shape[0]-1)/(image.shape[1]-1.)

    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape,dtype='bool')
    # obsolete elif len(mask.shape) > 1:
    # obsolete     mask = mask.ravel()

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
