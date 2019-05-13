# VoidFinder

The VoidFinder algorithm by [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) is based upon the algorithm described by [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E).  It removes all isolated galaxies (described as having the third nearest-neighbor more than ~7 Mpc/h away) using only galaxies with absolute magnitudes M<sub>r</sub> < -20.  After applying a grid to the remaining galaxies, spheres are grown from all empty cells until it is bounded by four galaxies on the surface.  A sphere must have a minimum radius of 10 Mpc/h to be considered part of a void.  If two spheres overlap by more than 10% of their volume, they are considered part of the same void.

## Citation

Please cite [Hoyle & Vogeley (2002)](http://adsabs.harvard.edu/abs/2002ApJ...566..641H) and [El-Ad & Piran (1997)](http://adsabs.harvard.edu/abs/1997ApJ...491..421E) when using this algorithm.




## Building & Running Voidfinder
Voidfinder now includes some cythonized code for speed.  To build this code, enter the 
/python/voidfinder/ directory and run:

cython -a *.pyx

python setup.py build_ext --inplace

then from the /python/scripts/ directory edit the paths in whichever run script desired and run
as normal (example: python SDSS_Voidfinder.py)
