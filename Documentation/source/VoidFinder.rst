VoidFinder
==========

`vast.voidfinder` is a software package containing a Python 3 implementation of 
the VoidFinder algorithm [@El-Ad:1997] that is based on the algorithm's Fortran 
implementation by @Hoyle:2002.  Motivated by the expectation that voids are 
spherical to first order, this algorithm defines voids as the unions of sets of 
spheres grown in the underdense regions of the large-scale structure.

`VoidFinder` begins by removing all isolated tracers from the catalog of 
objects, defined as having significantly ($1.5\sigma$) larger than average 
third-nearest neighbor distances.  The remaining tracers are then placed on a 
grid, and spheres are grown from the centers of the empty cells until they are 
bounded by four tracers on their surfaces.

All spheres larger than a specified radius (typically around 10 Mpc/h) are 
considered possible maximal spheres -- the largest sphere that can fit in a 
given void region.  Filtering through these candidate maximal spheres by order 
of decreasing radius, no maximal sphere can overlap by more than 10% of its 
volume with any other previously identified (larger) maximal sphere.  After the 
maximal spheres are identified, the remaining holes are combined with these 
maximal spheres to enhance the void structure if they overlap exactly one 
maximal sphere by at least 50% of its volume.  The union of a set of spheres 
(one maximal and the remaining smaller holes) defines a void region.

.. autofunction:: vast.voidfinder.find_voids