# AST1501

James Lane AST 1501 Project at UofT

## Notebooks

If the notebooks don't render in github then take the URL and paste it 
into [NBViewer](http://nbviewer.jupyter.org)

### 1-Prepare Dataset

[Notebook](notebooks/1-gaiadr2-apogee_dataset/gaiadr2-apogee_dataset.ipynb)

Prepare the dataset. Convert all quantities to galactocentric coordinates, find all stars without proper motion measurements, and apply the cuts recommended by Henry.

### 2-Property Maps

[Notebook 1](notebooks/2-velocity_maps/velocity_maps.ipynb)

APOGEE DR14 + Gaia DR2 stars shown in galactocentric coordinates. Plots show 2D density, galactocentric radial velocity, galactocentric tangential velocity, heliocentric radial velocity, galactic proper motions, and Bovy+ 2012 model for comparison.

[Notebook 2](notebooks/2-velocity_maps/chemistry_maps.ipynb)

Maps and azimuthal gradients of element abundances.

### 3-Velocity dispersions

[Notebook](notebooks/3-velocity_dispersions/velocity_dispersions.ipynb)

Calculate the velocity dispersions for the asymmetric drift model from Bovy+2012 and BT08. Fit them with simple profiles.

## Triaxial Potential

[Notebook](notebooks/4-compare_potentials)

Explore the triaxial potential functionality in galpy. Compare the triaxial potential to the MWPotential2014 model in galpy.

## Triaxial Potential Decomposition

[Notebook](notebooks/5-potential_decomposition/potential_decomposition.ipyn)

Decompose the triaxial NFW potential + MWPotential2014 disk and bulge into a power law at various radial slices as well as a sinusoidal component.

## Triaxial DF

[Notebook 1](notebooks/6-triaxial_potential_DF/triaxial_potential_DF.ipynb)

Calculate the DF of the triaxial halo.

[Notebook 2](notebooks/6-triaxial_potential_DF/test_strange_DF_actions.ipynb)

Test the actions of the strange features of the triaxial halo DF that are commonly seen.

## Power Spectrum

[Notebook](notebooks/7-power_spectrum/power_spectrum.ipynb)

Calculate the power spectrum of velocity fluctuations in the Gaia DR2 data as well as from triaxial halos.
