# ----------------------------------------------------------------------------
#
# TITLE - triaxial_orbit_movie.py
# AUTHOR - James Lane
# PROJECT - AST1501
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
Make a movie of some orbits in a growing triaxial potential. Add a few
specific features, including:
 - random selection of orbits with different properties.
 - 
'''
__author__ = "James Lane"

### Imports

## Basic
import numpy as np
import sys, os, pdb, copy, pickle
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
from galpy.df import dehnendf
from galpy.util import bovy_coords as gpcoords
from galpy.util import bovy_conversion as gpconv
from galpy.util import bovy_plot as gpplot

# Project-specific
sys.path.insert(0, os.path.abspath('../../../src') )
import ast1501.potential

# ----------------------------------------------------------------------------

# Keywords

# Make new orbits or load them from file? filename is either saved to or loaded 
# depending of if we are making new orbits.
make_new_orbits = True
filename = 'movie_orbits.p'

# Simulation length 
t0 = 4 # Gyr

# Number of orbits
n_orbits = 25

# Radial range to sample from
R_min = 6
R_max = 18

# The times at which the triaxial halo will begin to grow and end growing
# with respect to t0
t_form = 1 # 1 Gyr after the simulation starts
t_steady = 3 # 2 Gyr after the simulation starts

# Triaxial halo parameters
halo_b = 0.7

# ----------------------------------------------------------------------------

### Make the potential

# Make MWPotential2014
mwpot = potential.MWPotential2014
tripot = ast1501.potential.make_triaxialNFW_dsw(halo_b=5.0, t_form=t_form, 
                                                t_steady=t_steady)

# ----------------------------------------------------------------------------

# Do some sort of sampling of orbit properties. Define a Dehnen disk and pull 
# orbit sample from it. Then evaluate orbits

if make_new_orbits:
    # Make a Dehnen DF and sample a few orbits from between 4 and 15 kpc
    # dfc= dehnendf(beta=0.,profileParams=(1./4.,1.,0.2))
    # orbs = dfc.sample(n=n_orbits, returnOrbit=True, nphi=1, 
    #                     rrange=[R_min*apu.kpc,R_max*apu.kpc])
    
    orbs = []
    for i in range(n_orbits):
        _R = np.random.uniform(low=R_min, high=R_max, size=1)[0]
        _phi = np.random.uniform(low=0, high=2*np.pi, size=1)[0]
        _vc = potential.vcirc( potential.MWPotential2014, R=_R/8.0 )*220.0
        _vT = _vc * np.random.uniform(low=0.9, high=1.1, size=1)[0]
        _vR = 0.0
        orbs.append( orbit.Orbit(vxvv=[ _R*apu.kpc,
                                        _vR*apu.kpc/apu.s,
                                        _vT*apu.km/apu.s,
                                        0.0*apu.kpc,
                                        0.0*apu.km/apu.s,
                                        _phi*apu.radian]) )
    
    times = np.arange(0,t0,0.005) * apu.Gyr
    
    cntr = 0
    for o in orbs:
        print('Integrating '+str(cntr))
        o.integrate(times,tripot)
        print('Done '+str(cntr))
        cntr += 1
    ###o
    # pickle.dump( orbs, open( filename, 'wb' ) )
else:
    pass
    # Assume that we'll be loading orbits
    # orbs = pickle.load( open( filename, 'rb' ) )
    # times = orbs[0].time()
##ie

# Set the radial scale
ro = orbs[0]._ro

# ----------------------------------------------------------------------------

### Plotting keywords

# We would like to color the points by guiding center radius
lc_norm = colors.Normalize( vmin=R_min-1, vmax=R_max+1 )
cmap = cm.jet
lcs = [ cmap(lc_norm( e.rguiding(pot=mwpot)))[0] for e in orbs ]
lw = 1.5
ps = 30
pt_ec = 'None'

# ----------------------------------------------------------------------------

### Now make an animation

fig = plt.figure( figsize=(6,6) )
ax = fig.add_subplot(111)

Rlim = 24
xlim = [-Rlim,Rlim]
ylim = [-Rlim,Rlim]

### Axis 1: XY
ax.set_xlim(xlim[0],xlim[1])
ax.set_ylim(ylim[0],ylim[1])
ax.set_ylabel('Y  [kpc]', fontsize=14, labelpad=10.0)
ax.set_xlabel('X  [kpc]', fontsize=14, labelpad=10.0)
ax.tick_params(right='on', top='on', direction='in')
# for Rad in [8,10,12,14]:
#     circ = plt.Circle((0,0), Rad, edgecolor='Grey', facecolor='None', alpha=0.25)
#     ax.add_artist(circ)
####

# Setup lines

lines = [ ax.plot([],[],'-',color=lc,linewidth=lw,zorder=1,alpha=1.0)[0] for lc,_ in zip(lcs,orbs) ]
pts = [ ax.scatter([],[],color=lc,s=ps,edgecolor=pt_ec,zorder=1,alpha=1.0) for lc,_ in zip(lcs,orbs) ]

time_text = ax.annotate('', xy=(0.55,0.9), xycoords='axes fraction', fontsize=14)
tri_text = ax.annotate('', xy=(0.55,0.95), xycoords='axes fraction', fontsize=14)
time_template = '%.2f Gyr'
texts = [time_text,tri_text]

objs = lines+pts+texts

# Initialization function
def init():
    
    for line in lines:
        line.set_data([],[])
    ####
    for pt in pts:
        pt.set_offsets([])
    ####    
    time_text.set_text('')
    tri_text.set_text('')

    return  objs
#def

# Animate function, argument is the frame number.
def animate(i):

    imin = max(0,i-20)

    # Plot lines and points.
    for j,line in enumerate(lines):
        line.set_data( orbs[j].x(times[imin:i+1]).value,
                       orbs[j].y(times[imin:i+1]).value )
    ###j
    for j,pt in enumerate(pts):
        pt.set_offsets( (orbs[j].x(times[i]).value, 
                         orbs[j].y(times[i]).value) )
    ###i
    
    # Check if before triaxiality on
    if times[i].value < t_form:
        time_text.set_text( time_template % (times[i].value) )
        tri_text.set_text('Spherical Halo')
    elif times[i].value < t_steady:
        time_text.set_text( time_template % (times[i].value) )
        tri_text.set_text('Transforming Halo')
    else:    
        time_text.set_text( time_template % (times[i].value) )
        tri_text.set_text('Triaxial Halo')
    ##ie

    return objs
#def

# Generate animation
print('Compiling animation')
anim = animation.FuncAnimation(fig, animate, frames=int(len(times)),
                                interval=2)
anim.save('animation_v2.mp4', writer='ffmpeg', fps=30, dpi=300)
