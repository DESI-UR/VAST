################
Using VoidRender
################


Quick start
===========

A summary checklist for installing and running `VAST.VoidRender`.

 * Clone the `GitHub repository <https://github.com/DESI-UR/VAST>`_
 * Navigate to ``VAST/VoidFender`` (for running `VoidRender` on output from 
   `VoidFinder`) and/or ``VAST/Vsquared`` (for running `VoidRender` on output 
   from `Vsquared`) and run::
    
    python setup.py install
    
   See :ref:`VF-install` or :ref:`V2-install` for installation options.  NOTE: 
   The version of `VoidRender` that corresponds to the void-finding algorithm of 
   choice will be built and installed when that algorithm (`VoidFinder`, for 
   example) is built and installed.
   
 * Navigate to the ``scripts`` directory within your chosen void-finding 
   algorithm and modify ``visualize_voids.py`` if appropriate.  Parameters to 
   edit might include:
   
   * Void file name(s) sent into ``viz.load_hole_data``
   * Galaxy catalog file name sent into ``viz.load_galaxy_data``
   * etc.

 * Run your script (``visualize_voids.py``, in this case) on your machine.





Example scripts
===============

Included in the `VAST/VoidFinder` repository (`VAST/VoidFinder/scripts/`) is 
``visualize_voids.py``, a working example script for how to run `VoidRender` to 
visualize the voids found with `VoidFinder` in a given catalog.

Included in the `VAST/Vsquared` repository (`VAST/Vsquared/scripts/`) is 
``visualize_voids.py``, a working example script for how to run `VoidRender` to 
visualize the voids found with `Vsquared` in a given catalog.





Visualize voids
===============

The easiest way to use `VoidRender` is to create a script that

1. Reads in the output from a **VAST** void-finding algorithm and the 
   corresponding galaxy catalog within which the voids were located.
2. Defines various visualization aesthetics (void color, etc.).
3. Generates an interactive 3D visualization of the voids and galaxies.

Examples of this script are the ``visualize_voids.py`` files, located in the 
``scripts`` directory within each of the void-finding algorithms within 
**VAST**.

**NOTE:** Due to differences in the void structures found by each of the 
different void-finding algorithms implemented in **VAST**, there is an 
implementation of `VoidRender` that corresponds to each algorithm.



1. Reading in the data
----------------------

Generally, the first functions that should be called in a script running 
``VoidRender`` are ``load_galaxy_data`` and ``load_void_data``::

    from vast.voidfinder.viz import load_galaxy_data, load_void_data
    
These functions read a galaxy catalog into memory (as an ``astropy.table.Table`` 
object)





