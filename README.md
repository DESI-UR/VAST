# VAST

The Void Analysis Software Toolkit, or VAST, provides pure Python implementations of two popular classes of void-finding algorithms in galaxy catalogs:

1. Void identification by growing spherical voids.
1. Void identification using watershed algorithms.

**VoidFinder** is the algorithm which utilizes a sphere-growing method on a grid search and a unionization of the sufficiently large spheres.  The `VoidFinder` directory contains the package, which includes an efficient Multi-Process Cythonized version of VoidFinder (`from vast.voidfinder import find_voids`), as well as an OpenGL based visualization for the output of VoidFinder (the `vast.voidfinder.viz` package).

See [here](https://www.youtube.com/playlist?list=PLCZohAzuOVRK4itOBDQNFMl3w2uvox16a) for 3D OpenGL-based visualization of VoidFinder's voids in SDSS DR7!

**V<sup>2</sup>** is a voronoi-tesselation-based algorithm for finding the void regions, based on the ZOBOV algorithm.  ZOBOV uses the gradient of the volume of adjacent voronoi cells to flow multiple cells together into large void regions.  This code is still under development.

