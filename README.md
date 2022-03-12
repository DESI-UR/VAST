# VAST: Void Analysis Software Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4135702.svg)](https://zenodo.org/record/4135702)
![tests](https://github.com/DESI-UR/VAST/actions/workflows/tests.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/vast/badge/?version=latest)](https://vast.readthedocs.io/en/latest/?badge=latest)

The Void Analysis Software Toolkit, or VAST, provides pure Python 
implementations of two popular classes of void-finding algorithms in galaxy 
catalogs:

1. Void identification by growing spherical voids.
1. Void identification using watershed algorithms.


Our docs can be found here: https://vast.readthedocs.io/en/latest/

## VoidFinder

**VoidFinder** is an algorithm which utilizes a sphere-growing method on a grid 
search and a unionization of the sufficiently large spheres.  The `VoidFinder` 
directory contains the package, which includes an efficient Multi-Process 
Cythonized version of VoidFinder (`from vast.voidfinder import find_voids`), as 
well as an OpenGL based visualization for the output of VoidFinder (the 
`vast.voidfinder.viz` package).

See 
[here](https://www.youtube.com/playlist?list=PLCZohAzuOVRK4itOBDQNFMl3w2uvox16a) 
for 3D OpenGL-based visualization of VoidFinder's voids in SDSS DR7!


## V<sup>2</sup>

**V<sup>2</sup>** is a voronoi-tesselation-based algorithm for finding the void 
regions, based on the ZOBOV algorithm.  ZOBOV uses the gradient of the volume of 
adjacent voronoi cells to flow multiple cells together into large void regions.

