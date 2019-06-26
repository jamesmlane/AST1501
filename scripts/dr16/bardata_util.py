import os, os.path
import pickle
import numpy
import numpy.lib.recfunctions
from astropy.io import fits
import astropy.units as u
from galpy.util import save_pickles
import gaia_tools.xmatch as xmatch
from astropy.coordinates import SkyCoord, Galactocentric, CartesianDifferential
from gaia_tools.load import (_swap_in_astroNN, _add_astroNN_distances, 
                             _elemIndx)
gaia2_matches= None
def distance_adjust_func(d):
    return 0.99\
            +(numpy.pi/2.+numpy.arctan((d-4.35)/0.3))/numpy.pi*0.06\
            +(numpy.pi/2.+numpy.arctan((d-5.5)/0.15))/numpy.pi*0.055
def load_data():
    """Load the astroNN data, match to Gaia, combine"""
    astronn_dr16nn_filename= '/geir_data/scr/henrysky/astroNN_dr16_r12.fits'
    aspcap_filename= '/geir_data/scr/henrysky/sdss_mirror/apogeework/apogee'\
              '/spectro/aspcap/r12/noaspcap/allStar-r12-noaspcap-58358.fits'
    ages_filename= '/geir_data/scr/tmackereth/astroNN_dr16_r12_ages_corrections.fits'
    astronn= fits.getdata(astronn_dr16nn_filename)
    aspcap= fits.getdata(aspcap_filename)
    ages= fits.getdata(ages_filename)
    ages['age']= ages['age_lowess_correct']
    data= _swap_in_astroNN(aspcap,astronn)
    data= _add_astroNN_distances(data,astronn)
    data= _add_astroNN_ages(data,ages,rowmatched=True)
    gaia_match_filename= 'astroNN_gaia_match.pkl'
    global gaia2_matches
    if os.path.exists(gaia_match_filename):
        with open(gaia_match_filename,'rb') as savefile:
            gaia2_matches= pickle.load(savefile)
            matches_indx= pickle.load(savefile)
    else:
        gaia2_matches, matches_indx= xmatch.cds(data,colRA='RA',colDec='DEC',
                           xcat='vizier:I/345/gaia2',gaia_all_columns=True)
        save_pickles(gaia_match_filename,gaia2_matches,matches_indx)
    data= data[matches_indx]
    indx= data['weighted_dist'] > 0
    data= data[indx]
    gaia2_matches= gaia2_matches[indx]
    skyc= SkyCoord(ra=data['RA']*u.deg,
               dec=data['DEC']*u.deg,
               distance=data['weighted_dist']*u.pc,
               pm_ra_cosdec=gaia2_matches['pmra'].data.data*u.mas/u.yr,
               pm_dec=gaia2_matches['pmdec'].data.data*u.mas/u.yr,
               radial_velocity=data['VHELIO_AVG']*u.km/u.s)
    v_sun= CartesianDifferential([11.1,242,7.25]*u.km/u.s)
    gc_frame= Galactocentric(galcen_distance=8.125*u.kpc,
                             z_sun=20.8*u.pc,
                             galcen_v_sun=v_sun)
    gc = skyc.transform_to(gc_frame)
    gc.representation_type = 'cylindrical'
    data= numpy.lib.recfunctions.append_fields(\
                        data,
                        ['R','phi','Z','vR','vT','vz','plx','plx_err'],
                        [gc.rho.to(u.kpc).value,
                         numpy.pi-gc.phi.rad,
                         gc.z.to(u.kpc).value,
                         gc.d_rho.to(u.km/u.s).value,
                         -(gc.d_phi*gc.rho).to(u.km/u.s,
                                equivalencies=u.dimensionless_angles()).value,
                         gc.d_z.to(u.km/u.s).value,
                         gaia2_matches['parallax'],
                         gaia2_matches['parallax_error']],
                         ['f8','f8','f8','f8','f8','f8','f8','f8'],
                                               usemask=False)
    return data

def reload_data(data,distfac=distance_adjust_func,
                R0=8.125,vsun=[11.1,242.,7.25]):
    """Function to 'reload' the data applying (a) systematic distance offset, 
    (b) different R0, and (c) different vsun"""
    if callable(distfac):
        fac= distfac(data['weighted_dist']/1000.)
    else:
        fac= distfac
    skyc= SkyCoord(ra=data['RA']*u.deg,
                   dec=data['DEC']*u.deg,
                   distance=data['weighted_dist']*u.pc*fac,
                   pm_ra_cosdec=gaia2_matches['pmra'].data.data*u.mas/u.yr,
                   pm_dec=gaia2_matches['pmdec'].data.data*u.mas/u.yr,
                   radial_velocity=data['VHELIO_AVG']*u.km/u.s)
    v_sun= CartesianDifferential(vsun*u.km/u.s)
    gc_frame= Galactocentric(galcen_distance=R0*u.kpc,
                             z_sun=20.8*u.pc,
                             galcen_v_sun=v_sun)
    gc = skyc.transform_to(gc_frame)
    gc.representation_type = 'cylindrical'
    data['R']= gc.rho.to(u.kpc).value
    data['phi']= numpy.pi-gc.phi.rad
    data['Z']= gc.z.to(u.kpc).value
    data['vR']= gc.d_rho.to(u.km/u.s).value
    data['vT']= -(gc.d_phi*gc.rho).to(u.km/u.s,
                                equivalencies=u.dimensionless_angles()).value
    data['vz']= gc.d_z.to(u.km/u.s).value
    return data    

def _add_astroNN_ages(data,astroNNAgesdata,rowmatched=False):
    # Edited version, because Ted's new file is different from the previous one
    fields_to_append= ['age','age_total_error','age_model_error']
    if True:
        # Faster way to join structured arrays (see https://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays) 
        newdtype= data.dtype.descr+\
            [(f,'<f8') for f in fields_to_append]
        newdata= numpy.empty(len(data),dtype=newdtype)
        for name in data.dtype.names:
            newdata[name]= data[name]
        for f in fields_to_append:
            if rowmatched:
                newdata[f]= astroNNAgesdata[f]
            else:
                newdata[f]= numpy.zeros(len(data))-9999.
        data= newdata
    else:
        # This, for some reason, is the slow part (see numpy/numpy#7811)
        if rowmatched:
            data= numpy.lib.recfunctions.append_fields(\
                data,
                fields_to_append,
                [astroNNAgesdata[f] for f in fields_to_append],
                [astroNNAgesdata[f].dtype for f in fields_to_append],
                usemask=False)
        else:
            data= numpy.lib.recfunctions.append_fields(\
                data,
                fields_to_append,
                [numpy.zeros(len(data))-9999. for f in fields_to_append],
                usemask=False)
    if rowmatched: return data
