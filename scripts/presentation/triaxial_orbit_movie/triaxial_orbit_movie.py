# ----------------------------------------------------------------------------
#
# TITLE -
# AUTHOR - James Lane
# PROJECT -
# CONTENTS:
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''

'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy
import glob
# import subprocess

## Plotting
from matplotlib import pyplot as plt
from matplotlib import animation
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
from galpy.util import bovy_plot as gpplot

from scipy.signal import argrelextrema

import ophstream.misc

# ----------------------------------------------------------------------------

# Time in Gyr when the simulation will 'begin' (In the past!)
t0 = 3

# The times at which the triaxial halo will begin to grow and end growing
# with respect to t0
t_tri_begin = 1 # 1 Gyr after the simulation starts
t_tri_end = 3 # 4 Gyr after the simulation starts

### Make the potential

# Get MWPotential2014
mwpot = potential.MWPotential2014
mwbulge = copy.deepcopy(mwpot[0])
mwdisk = copy.deepcopy(mwpot[1])
mwhalo = copy.deepcopy(mwpot[2])

# Make the potential
mwhalo_a = mwhalo.a * mwhalo._ro * apu.kpc
mwhalo_amp = mwhalo.dens(mwhalo_a,0) * 16 * mwhalo.a**3 * np.pi * \
             gpconv.mass_in_msol(mwhalo._vo, mwhalo._ro) * apu.M_sun

mwdisk_a = mwdisk._a * mwdisk._ro * apu.kpc
mwdisk_b = mwdisk._b * mwdisk._ro * apu.kpc
mwdisk_amp = mwdisk._amp * gpconv.mass_in_msol(mwdisk._vo, mwdisk._ro) * apu.M_sun

mwbulge_r1 = 1
mwbulge_amp = mwbulge.dens(mwbulge_r1,0) * np.exp((1/mwbulge.rc)**2) * \
              gpconv.dens_in_msolpc3(mwhalo._vo, mwhalo._ro) * apu.M_sun / apu.pc**3
mwbulge_alpha = mwbulge.alpha
mwbulge_rc = mwbulge.rc * mwbulge._ro * apu.kpc

# Generate the scalped potentials
mwbulge = potential.PowerSphericalPotentialwCutoff(amp=mwbulge_amp, alpha=mwbulge_alpha, rc=mwbulge_rc)
# mwbulge.turn_physical_off()
mwdisk = potential.MiyamotoNagaiPotential(amp=mwdisk_amp, a=mwdisk_a, b=mwdisk_b)
# mwdisk.turn_physical_off()
mwhalo = potential.NFWPotential(amp=mwhalo_amp, a=mwhalo_a)
# mwhalo.turn_physical_off()
mwpot = [mwhalo, mwdisk, mwbulge]

# Make the negative amplitude NFW and wrap it in a DehnenSmoothWrapperPotential
mwhalo_rev = potential.NFWPotential(amp=mwhalo_amp*-1,
                                    a=mwhalo_a)
# mwhalo_rev.turn_physical_off()
mwhalo_rev_dsw = potential.DehnenSmoothWrapperPotential(pot=mwhalo_rev,
                                        tform= t_tri_begin * apu.Gyr,
                                        tsteady= t_tri_end * apu.Gyr
                                        )
# mwhalo_rev_dsw.turn_physical_off()

trihalo = potential.TriaxialNFWPotential(amp=mwhalo_amp,
                                        a=mwhalo_a,
                                        b=3.,
                                        c=1.0,
                                        pa=0.)

trihalo_dsw = potential.DehnenSmoothWrapperPotential(pot=trihalo,
                            tform= t_tri_begin * apu.Gyr,
                            tsteady= t_tri_end * apu.Gyr)
                                        
tripot = [ mwhalo_rev_dsw,
           mwhalo,
           trihalo_dsw,
           mwdisk,
           mwbulge ]

# ----------------------------------------------------------------------------

### Make the orbits

times = np.arange(0,t0,0.005) * apu.Gyr

o1 = orbit.Orbit( vxvv=[1.,
                        0.,
                        1.,
                        0.,
                        0.,
                        0.] )
o2 = orbit.Orbit( vxvv=[1.,0.,1.,0.,0.,np.pi] )

o1.integrate(times,tripot)
o2.integrate(times,tripot)

# ----------------------------------------------------------------------------

### Now make an animation

fig = plt.figure( figsize=(6,6) )
ax = fig.add_subplot(111)

xlim = [-1.5,1.5]
ylim = [-1.5,1.5]

### Axis 1: XY
ax.set_xlim(xlim[0],xlim[1])
ax.set_ylim(ylim[0],ylim[1])
ax.set_ylabel('Y [8 kpc]', fontsize=14, labelpad=10.0)
ax.set_xlabel('X [8 kpc]', fontsize=14, labelpad=10.0)
ax.tick_params(right='on', top='on', labelbottom='off', direction='in')

line_color = 'DodgerBlue'
line_color_tri = 'Red'
pt_ec = 'Black'

# Setup lines
line1, = ax.plot([], [], '-', color=line_color, linewidth=1, zorder=1)
pt1 = ax.scatter([], [], color=line_color, s=3, edgecolor=pt_ec, zorder=1)
line2, = ax.plot([], [], '-', color=line_color, linewidth=1, zorder=1)
pt2 = ax.scatter([], [], color=line_color, s=3, edgecolor=pt_ec, zorder=1)
time_text = ax.annotate('', xy=(0.7,0.9), xycoords='axes fraction', fontsize=14)
time_template = '%.2f Gyr'
time_template_tri = '%.2f Gyr\ntriaxiality on'

def init():
    line1.set_data([], [])
    pt1.set_offsets([])
    line2.set_data([], [])
    pt2.set_offsets([])

    time_text.set_text('')

    return  pt1, line1, pt2, line2, time_text
#def

# Animate function, argument is the frame number.
def animate(i):

    # Plot lines and points.

    if times[i].value < t_tri_begin:
        
        line1.set_data( o1.x(times[:i+1]), o1.y(times[:i+1]) )
        pt1.set_offsets( (o1.x(times[i]), o1.y(times[i])) )
        line2.set_data( o2.x(times[:i+1]), o2.y(times[:i+1]) )
        pt2.set_offsets( (o2.x(times[i]), o2.y(times[i])) )
        
        time_text.set_text( time_template % (times[i].value) )
    else:
        
        line1.set_data( o1.x(times[:i+1]), o1.y(times[:i+1]) )
        # line1.set_color('Red')
        pt1.set_offsets( (o1.x(times[i]), o1.y(times[i])) )
        line2.set_data( o2.x(times[:i+1]), o2.y(times[:i+1]) )
        # line2.set_color('Red')
        pt2.set_offsets( (o2.x(times[i]), o2.y(times[i])) )
        
        time_text.set_text( time_template_tri % (times[i].value) )

    return  pt1, line1, pt2, line2, time_text
#def

anim = animation.FuncAnimation(fig, animate, frames=len(times),
                                interval=1)
anim.save('animation.mp4', writer='ffmpeg', fps=25, dpi=300)
