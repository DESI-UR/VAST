
.. role:: raw-html(raw)
    :format: html



############
Introduction
############

:raw-html:`<strong>V<sup>2</sup></strong>` is a software package for finding 
voids based on the ZOBOV (ZOnes Bordering On Voidness) algorithm by 
`Neyrinck (2008) <https://arxiv.org/abs/0712.3049>`_, which uses the gradient of 
the volume of adjacent voronoi cells to flow multiple cells into larger void 
regions. Voids can be found both in observational surveys and simulations with 
periodic boundary conditions.

**VAST** contains the package, as well as an example script for automatically 
running all steps of the algorithm. To import the main void-finding class of 
:raw-html:`<strong>V<sup>2</sup></strong>`::

    from vast.vsquared import zobov

**Voronoi tesselation**

:raw-html:`<strong>V<sup>2</sup></strong>` first produces a Voronoi tessellation 
of the catalog of large-scale tracers, and the volumes of the Voronoi cells are 
used to identify local density minima.

**Zones**

Zones are then built from density minima in the distribution of the Voronoi 
cells using a watershed transform, where each cell is linked to its least dense 
neighbor.

**Defining the voids**

Finally, voids are formed from these by identifying low-density boundaries 
between adjacent zones and using them to grow unions of weakly divided zones.  
This list of voids is then typically pruned to remove void candidates unlikely 
to be true voids.  :raw-html:`<strong>V<sup>2</sup></strong>` includes several 
of the different void-pruning methods that exist, including methods from other 
ZOBOV implementations such as `VIDE <http://www.cosmicvoids.net/>`_ and 
`REVOLVER <https://github.com/seshnadathur/Revolver/>`_.



Citation
========

Please cite `Neyrinck (2008) <https://arxiv.org/abs/0712.3049>`_ when using this 
algorithm.



