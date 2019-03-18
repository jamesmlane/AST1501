# ----------------------------------------------------------------------------
#
# TITLE - triaxial_orbit_movie.py
# AUTHOR - James Lane
# PROJECT - AST1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Make a movie of some orbits in a growing triaxial potential
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy
import glob

## Plotting
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib import cm

## Astropy
from astropy import units as apu

## galpy
from galpy import orbit
from galpy import potential
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv
from galpy.util import bovy_plot as gpplot

# Project-specific
sys.path.insert(0, os.path.abspath('../../../src') )
import ast1501.potential

# ----------------------------------------------------------------------------

# Keywords

# Simulation length 
t0 = 7 # Gyr

# The times at which the triaxial halo will begin to grow and end growing
# with respect to t0
tform = 1 # 1 Gyr after the simulation starts
tsteady = 4 # 3 Gyr after the simulation starts

# ----------------------------------------------------------------------------

### Make the potential

# Make MWPotential2014
mwpot = potential.MWPotential2014

# Make the triaxial halo
trihalo = ast1501.potential.make_MWPotential2014_triaxialNFW(halo_b=2.0, 
    halo_phi=0.0, halo_c=1.0)

# Make MWPotential2014 with DSW around the halo and triaxial halo
tripot = ast1501.potential.make_tripot_dsw(trihalo=trihalo, tform=tform, 
    tsteady=tsteady)

# ----------------------------------------------------------------------------

### Make the orbits

# First get the kinematics for circular orbits
r = np.array([1,2,4])
vc = potential.vcirc(mwpot, r, 0)

# Times
times = np.arange(0,t0,0.01) * apu.Gyr

# Orbit declaration
o1 = orbit.Orbit( vxvv=[r[0],0.,vc[0],0.,0.,0.] )
o2 = orbit.Orbit( vxvv=[r[1],0.,vc[1],0.,0.,0.] )
o3 = orbit.Orbit( vxvv=[r[2],0.,vc[2],0.,0.,0.] )

# Integrate
print('Integrating')
o1.integrate(times,tripot)
o2.integrate(times,tripot)
o3.integrate(times,tripot)
print('Done integration')

# ----------------------------------------------------------------------------

### Now make an animation

fig = plt.figure( figsize=(6,6) )
ax = fig.add_subplot(111)

xlim = [-5,5]
ylim = [-5,5]

### Axis 1: XY
ax.set_xlim(xlim[0],xlim[1])
ax.set_ylim(ylim[0],ylim[1])
ax.set_ylabel('Y  [8 kpc]', fontsize=14, labelpad=10.0)
ax.set_xlabel('X  [8 kpc]', fontsize=14, labelpad=10.0)
ax.tick_params(right='on', top='on', direction='in')

line1_color = 'DodgerBlue'
line2_color = 'Red'
line3_color = 'Purple'
line_width = 1.5
point_size = 30
pt_ec = 'None'

# Setup lines
line1, = ax.plot([], [], '-', color=line1_color, linewidth=line_width, zorder=1)
pt1 = ax.scatter([], [], color=line1_color, s=point_size, edgecolor=pt_ec, zorder=1)
line2, = ax.plot([], [], '-', color=line2_color, linewidth=line_width, zorder=1)
pt2 = ax.scatter([], [], color=line2_color, s=point_size, edgecolor=pt_ec, zorder=1)
line3, = ax.plot([], [], '-', color=line3_color, linewidth=line_width, zorder=1)
pt3 = ax.scatter([], [], color=line3_color, s=point_size, edgecolor=pt_ec, zorder=1)
time_text = ax.annotate('', xy=(0.7,0.9), xycoords='axes fraction', fontsize=14)
tri_text = ax.annotate('', xy=(0.7,0.95), xycoords='axes fraction', fontsize=14)
time_template = '%.2f Gyr'

# Initialization function
def init():
    line1.set_data([], [])
    pt1.set_offsets([])
    line2.set_data([], [])
    pt2.set_offsets([])
    line3.set_data([], [])
    pt3.set_offsets([])

    time_text.set_text('')
    tri_text.set_text('')

    return  pt1, line1, pt2, line2, pt3, line3, time_text, tri_text
#def

# Animate function, argument is the frame number.
def animate(i):

    # Plot lines and points.
    line1.set_data( o1.x(times[:i+1]), o1.y(times[:i+1]) )
    pt1.set_offsets( (o1.x(times[i]), o1.y(times[i])) )
    line2.set_data( o2.x(times[:i+1]), o2.y(times[:i+1]) )
    pt2.set_offsets( (o2.x(times[i]), o2.y(times[i])) )
    line3.set_data( o3.x(times[:i+1]), o3.y(times[:i+1]) )
    pt3.set_offsets( (o3.x(times[i]), o3.y(times[i])) )
    
    # Check if before triaxiality on
    if times[i].value < tform:
        time_text.set_text( time_template % (times[i].value) )
        tri_text.set_text('')
    else:    
        time_text.set_text( time_template % (times[i].value) )
        tri_text.set_text('Triaxial On')
##ie

    return  pt1, line1, pt2, line2, pt3, line3, time_text, tri_text
#def

# Generate animation
print('Compiling animation')
anim = animation.FuncAnimation(fig, animate, frames=len(times),
                                interval=2)
anim.save('animation.mp4', writer='ffmpeg', fps=45, dpi=300)
