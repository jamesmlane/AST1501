# AST1501

James Lane AST 1501 Project at UofT

## Notebooks

### Prepare Dataset

[Notebook](notebooks/1-gaiadr2-apogee_dataset/gaiadr2-apogee_dataset.ipynb)

Prepare the dataset. Convert all quantities to galactocentric coordinates, find all stars without proper motion measurements, and apply the cuts recommended by Henry.

### Velocity maps

[Notebook](notebooks/2-velocity_maps/velocity_maps.ipynb)

APOGEE DR14 + Gaia DR2 stars shown in galactocentric coordinates. Plots show 2D density, galactocentric radial velocity, galactocentric tangential velocity, heliocentric radial velocity, galactic proper motions, and Bovy+ 2012 model for comparison.

### Velocity dispersions

[Notebook](notebooks/3-velocity_dispersions/velocity_dispersions.ipynb)

Calculate the velocity dispersions for the asymmetric drift model from Bovy+2012 and BT08. Fit them with simple profiles.

## Triaxial Potential

[Notebook](notebooks/4-compare_potentials)

Explore the triaxial potential functionality in galpy. Compare the triaxial potential to the MWPotential2014 model in galpy.

## Triaxial Potential Decomposition

[Notebook](notebooks/5-potential_decomposition/potential_decomposition.ipynb)

Decompose the triaxial NFW potential + MWPotential2014 disk and bulge into a power law at various radial slices as well as a sinusoidal component.

## Triaxial DF

[Notebook 1](notebooks/6-triaxial_potential_DF/triaxial_potential_DF.ipynb)

Work through DF calculation of the triaxial halo

[Notebook 2](notebooks/6-triaxial_potential_DF/test_strange_DF_actions.ipynb)

Test the actions of the strange features of the triaxial halo DF that are commonly seen.
