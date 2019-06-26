from astropy.io import fits
from gaia_tools.load import _swap_in_astroNN, _add_astroNN_distances
astronn_dr16nn_filename= '/geir_data/scr/henrysky/astroNN_dr16_r12.fits'
aspcap_filename= '/geir_data/scr/henrysky/sdss_mirror/apogeework/apogee'\
              '/spectro/aspcap/r12/noaspcap/allStar-r12-noaspcap-58358.fits'
astronn= fits.getdata(astronn_dr16nn_filename)
aspcap= fits.getdata(aspcap_filename)
data= _swap_in_astroNN(aspcap,astronn)
data= _add_astroNN_distances(data,astronn)