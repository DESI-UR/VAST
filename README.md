# UR work on voids

There are two void-finding algorithms which live here: VoidFinder and Dylan's version of ZOBOV.

VoidFinder is an algorithm which utilizes a sphere-growing method on a grid search and a unionization of the sufficiently large spheres.  The `VoidFinder/python/voidfinder/` directory contains the package, which includes an efficient Multi-Process Cythonized version of VoidFinder (`from voidfinder import find_voids`), as well as an OpenGL based visualization for the output of VoidFinder (the `voidfinder.viz` package).

Vsquared is a voronoi-tesselation-based algorithm for finding the void regions, based on the ZOBOV algorithm.  ZOBOV uses the gradient of the volume of adjacent voronoi cells to flow multiple cells together into large void regions.  This package includes stuff to fill in later!


