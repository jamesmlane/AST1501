# ----------------------------------------------------------------------------
#
# TITLE -
# AUTHOR - James Lane
# PROJECT -
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
''' Bin the Hunt2019 data and check moments of the velocity fields
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb

## Plotting
from matplotlib import pyplot as plt

## Astropy
from astropy import table

## Scipy
from scipy.stats import binned_statistic_2d
from scipy.stats import kurtosis, skew

# ----------------------------------------------------------------------------

# Choose what to plot
plot_logN = True
plot_N = False
plot_med_z = False
plot_med_vR = False
plot_med_vT = True
plot_med_vz = False
plot_std_vR = False
plot_std_vT = True
plot_std_vz = True
plot_skew_vR = False
plot_skew_vT = True
plot_skew_vz = False
plot_kurt_vR = False
plot_kurt_vT = False
plot_kurt_vz = True

# ----------------------------------------------------------------------------

filenames = ['/gpfs/fs0/scratch/b/bovy/jash2/mwbubs-models/data/transient_spirals/11-lsb-corrected-angle/lsb-corrected-angle_99.fits',
             '/gpfs/fs0/scratch/b/bovy/jash2/mwbubs-models/data/transient_spirals/08-three-spirals-sfb/three-spirals-sfb_99.fits']
# long-slow bar
# short-fast bar

data_filename = filenames[0] # Use long-slow bar
simtype = 'lsbar'

# ----------------------------------------------------------------------------

# Make / load the sample

force_make_sample=False
if not os.path.exists('good_sample_'+simtype+'.npy') or force_make_sample:
    print('Making sample...')
    # Set limits for analysis
    phi_lims = [-np.pi/2,np.pi/2] # In radians
    R_lims = [0,30] # In kpc
    z_lim = 0.3

    # Load the data
    data = table.Table.read(data_filename, format="fits")
    R,phi,vR,vT,z,vz = data["R", "phi", "vR", "vT", "z", "vz"].as_array().view((np.float64, 6)).T
    time = data["t"][0]

    # Convert to kpc, km/s, and make phi range -np.pi -> np.pi
    R *= 8
    z *= 8
    vR *= 220
    vT *= 220
    vz *= 220
    phi[phi > np.pi] -= 2*np.pi

    # Mask out the good particles to reduce the sample size
    valid_particle_mask = np.isfinite(R) & np.isfinite(phi) & np.isfinite(vR)\
        & np.isfinite(vT) & np.isfinite(z) & np.isfinite(vz) & (phi > phi_lims[0])\
        & (phi < phi_lims[1]) & (R > R_lims[0]) & (R < R_lims[1])\
        & (np.abs(z) < z_lim)

    R = R[valid_particle_mask]
    phi = phi[valid_particle_mask]
    vR = vR[valid_particle_mask]
    vT = vT[valid_particle_mask]
    z = z[valid_particle_mask]
    vz = vz[valid_particle_mask]

    good_sample = np.array([R,vR,vT,z,vz,phi])
    np.save('good_sample_'+simtype+'.npy', good_sample)
elif os.path.exists('good_sample_'+simtype+'.npy'):
    print('Loading sample...')
    good_sample = np.load('good_sample_'+simtype+'.npy')
    R,vR,vT,z,vz,phi = good_sample
else:
    sys.exit('Failed to get sample')
##ie

x = np.cos(phi) * R
y = np.sin(phi) * R

# ----------------------------------------------------------------------------

# Make the plots
def hist_gcxy(x, y, vals, vmin, vmax, stat, cmap):
    
    # Make the histogram using the supplied value and stat. Stat must be 
    # compatible with binned_statistic_2d, either 'median' or np.std
    hist, xedges, yedges, binid = binned_statistic_2d(x, y, values=vals, 
        statistic=stat, bins=n_bins, range=hist_range)

    # Plot the image. Rotate to account for histogram => plotting grid
    img = ax.imshow(np.rot90(hist), interpolation='nearest',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap=cmap, 
        vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = plt.colorbar(img)
    
    # Add the sun and it's orbit
    ax.scatter(x_sun, 0, marker=r'$\odot$', color='Black', s=256)
    orbit_circle = plt.Circle((0, 0), x_sun, edgecolor='Black', facecolor='None')
    disk_radius_circle = plt.Circle((0, 0), 15, edgecolor='Black', facecolor='None')
    bulge_circle = plt.Circle((0, 0), 3.5, edgecolor='Black', facecolor='None')
    ax.add_artist(orbit_circle)
    ax.add_artist(disk_radius_circle)
    ax.add_artist(bulge_circle)
    
    # Decorate
    ax.set_xlabel(r'X$_{GC}$ [kpc]')
    ax.set_ylabel(r'Y$_{GC}$ [kpc]')
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(direction='in', top='on', right='on')
    
    return fig, ax, cbar
#def

# Plot settings
n_bins = 25 # Bins
x_sun = 8.125 # Sun location
x_hi = 20
x_lo = 0
y_hi = 10
y_lo = -10
hist_range = [ [x_lo,x_hi], [y_lo,y_hi] ] # Ranges

## Make particle density map
if plot_logN:
    print('Plotting LogN...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=hist_range)
    # Rotate to account for histogram -> plotting grid
    hist = np.rot90(hist)
    img = ax.imshow(np.log10(hist), interpolation='nearest',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='rainbow', vmin=3, vmax=6.0)
    cbar = plt.colorbar(img)
    ax.scatter(x_sun, 0, marker=r'$\odot$', color='Black', s=256)
    orbit_circle = plt.Circle((0, 0), x_sun, edgecolor='Black', facecolor='None')
    disk_radius_circle = plt.Circle((0, 0), 15, edgecolor='Black', facecolor='None')
    bulge_circle = plt.Circle((0, 0), 4, edgecolor='Black', facecolor='None')
    ax.add_artist(orbit_circle)
    ax.add_artist(disk_radius_circle)
    ax.add_artist(bulge_circle)
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_ylim(y_hi,y_lo)
    cbar.set_label('Log[N]')
    fig.set_facecolor('White')
    plt.tight_layout()
    # plt.show()
    fig.savefig('plots/LogN_hist_H19.pdf')
    plt.close(fig)

if plot_N:
    print('Plotting N...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=hist_range)
    # Rotate to account for histogram -> plotting grid
    hist = np.rot90(hist)
    img = ax.imshow(hist, interpolation='nearest',
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap='viridis', vmin=100000, vmax=1000000)
    cbar = plt.colorbar(img)
    ax.scatter(x_sun, 0, marker=r'$\odot$', color='Black', s=256)
    orbit_circle = plt.Circle((0, 0), x_sun, edgecolor='Black', facecolor='None')
    disk_radius_circle = plt.Circle((0, 0), 15, edgecolor='Black', facecolor='None')
    bulge_circle = plt.Circle((0, 0), 4, edgecolor='Black', facecolor='None')
    ax.add_artist(orbit_circle)
    ax.add_artist(disk_radius_circle)
    ax.add_artist(bulge_circle)
    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_ylim(y_hi,y_lo)
    cbar.set_label('N')
    fig.set_facecolor('White')
    plt.tight_layout()
    # plt.show()
    fig.savefig('plots/N_hist_H19.pdf')
    plt.close(fig)

## Mean height

if plot_med_z:
    print('Plotting Median Z...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, z*1000, -50, 50, 'median', 'RdYlBu_r')
    cbar.set_label(r'Median $Z$ [pc]')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/Z_med_hist_H19.pdf')

### Mean velocity fields

## Radial velocity
if plot_med_vR:
    print('Plotting median vR...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vR, -15, 15, 'median', 'RdYlBu_r')
    cbar.set_label(r'Median V$_{R}$ [km/s]')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VR_med_hist_H19.pdf')

## Tangential velocity
if plot_med_vT:
    print('Plotting median vT...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vT, 190, 220, 'median', 'rainbow')
    cbar.set_label(r'Median V$_{T}$ [km/s]')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VT_med_hist_H19.pdf')

## vertical velocity
if plot_med_vz:
    print('Plotting median vz...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vz, -2.5, 2.5, 'median', 'RdYlBu_r')
    cbar.set_label(r'Median V$_{z}$ [km/s]')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/Vz_med_hist_H19.pdf')

### Dispersions

## Radial velocity
if plot_std_vR:
    print('Plotting std vR...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vR, 10, 65, np.std, 'rainbow')
    cbar.set_label(r'Standard Deviation V$_{R}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VR_std_hist_H19.pdf')

## Tangential velocity
if plot_std_vT:
    print('Plotting std vT...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vT, 10, 45, np.std, 'rainbow')
    cbar.set_label(r'Standard Deviation V$_{T}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VT_std_hist_H19.pdf')

## vertical velocity
if plot_std_vz:
    print('Plotting std vz...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vz, 5, 35, np.std, 'rainbow')
    cbar.set_label(r'Standard Deviation V$_{z}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/Vz_std_hist_H19.pdf')

### Skew

## Radial velocity
if plot_skew_vR:
    print('Plotting skew vR...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vR, -0.5, 1.5, skew, 'rainbow')
    cbar.set_label(r'Skew V$_{R}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VR_skew_hist_H19.pdf')

## Tangential velocity
if plot_skew_vT:
    print('Plotting skew vT...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vT, -4, 0, skew, 'rainbow')
    cbar.set_label(r'Skew V$_{T}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VT_skew_hist_H19.pdf')

## vertical velocity
if plot_skew_vz:
    print('Plotting skew vz...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vz, -0.5, 0.5, skew, 'RdYlBu_r')
    cbar.set_label(r'Skew V$_{z}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/Vz_skew_hist_H19.pdf')

### Kurtosis

## Radial velocity
if plot_kurt_vR:
    print('Plotting kurtosis vR...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vR, -1, 8, kurtosis, 'rainbow')
    cbar.set_label(r'Kurtosis V$_{R}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VR_kurtosis_hist_H19.pdf')

## Tangential velocity
if plot_kurt_vT:
    print('Plotting kurtosis vT...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vT, 0, 10, kurtosis, 'rainbow')
    cbar.set_label(r'Kurtosis V$_{T}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/VT_kurtosis_hist_H19.pdf')

## vertical velocity
if plot_kurt_vz:
    print('Plotting kurtosis vz...')
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    fig, ax, cbar = hist_gcxy(x, y, vz, -1, 1, kurtosis, 'RdYlBu_r')
    cbar.set_label(r'Kurtosis V$_{z}$')
    ax.set_ylim(y_hi,y_lo)
    fig.set_facecolor('White')
    plt.tight_layout()
    fig.savefig('plots/Vz_kurtosis_hist_H19.pdf')