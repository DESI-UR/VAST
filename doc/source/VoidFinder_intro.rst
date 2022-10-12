############
Introduction
############

**VAST.VoidFinder** is a software package containing a Python 3 implementation 
of the VoidFinder algorithm 
(`El-Ad & Piran, 1997 <https://arxiv.org/abs/astro-ph/9702135>`_) that is based 
on the algorithm's Fortran implementation by 
`Hoyle & Vogeley (2002) <https://arxiv.org/abs/astro-ph/0109357>`_.  Motivated 
by the expectation that voids are spherical to first order, this algorithm 
defines voids as the unions of sets of spheres grown in the underdense regions 
of the large-scale structure.

The **VoidFinder** directory contains the package, which includes an efficient 
Multi-Process Cythonized version of VoidFinder.  Options are available to 
identify voids in both observational galaxy surveys and periodic cosmological 
simulations.  To import the main void-finding function of VoidFinder::
    
    from vast.voidfinder import find_voids

**VoidFinder** begins by removing all isolated tracers from a catalog of 
objects.  The remaining tracers are then placed on a grid, and spheres are grown 
from the centers of the empty cells until they are bounded by four tracers on 
their surfaces.

All spheres larger than a specified radius (typically around 10 Mpc/h) are 
considered possible maximal spheres -- the largest sphere that can fit in a 
given void region.  Filtering through these candidate maximal spheres by order 
of decreasing radius, no maximal sphere can overlap by more than 10% of its 
volume with any other previously identified (larger) maximal sphere.  After the 
maximal spheres are identified, the remaining holes are combined with these 
maximal spheres to enhance the void structure if they overlap exactly one 
maximal sphere by at least 50% of its volume.  The union of a set of spheres 
(one maximal and the remaining smaller holes) defines a void region.
   





Citation
========

Please cite `Douglass et al. (2022) <https://doi.org/10.21105/joss.04033>`_ 
<!--`Hoyle & Vogeley (2002) <https://arxiv.org/abs/astro-ph/0109357>`_--> 
and `El-Ad & Piran (1997) <https://arxiv.org/abs/astro-ph/9702135>`_ when using 
this algorithm.




