# AST1501

James Lane AST 1501 Project at UofT

## Notebooks

### Prepare Dataset

[Notebook](notebooks/1-gaiadr2-apogee_dataset/gaiadr2-apogee_dataset.FIT)

Prepare the dataset. Convert all quantities to galactocentric
coordinates, find all stars without proper motion measurements, and 
apply the cuts recommended by Henry.

### Velocity maps

[Notebook](notebooks/2-velocity_maps/velocity_maps.ipynb)

APOGEE DR14 + Gaia DR2 stars shown in galactocentric coordinates. Plots show
2D density, heliocentric radial velocity, galactic proper motions, and Bovy+
2012 model for comparison.

### Velocity dispersions

[Notebook](notebooks/3-velocity_dispersions/velocity_dispersions.ipynb)

Calculate the velocity dispersions for the asymmetric drift model from
Bovy+2012 and BT08. Fit them with simple profiles.
