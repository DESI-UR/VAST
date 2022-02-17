
.. role:: raw-html(raw)
    :format: html


################
Using VoidRender
################


Quick start
===========

A summary checklist for installing and running **VoidRender**.

 * Clone the `GitHub repository <https://github.com/DESI-UR/VAST>`_
 * Navigate to ``VAST`` and run::
    
    python setup.py install
    
   See :ref:`VF-install` or :ref:`V2-install` for installation options.  NOTE: 
   The version of **VoidRender** that corresponds to the void-finding algorithm 
   of choice will be built and installed when **VAST** is built and installed.
   
 * Navigate to the ``example_scripts`` directory and modify 
   ``visualize_voids_[algorithm].py`` if appropriate.  Parameters to edit might 
   include:
   
   * Void file name(s) sent into ``viz.load_void_data``
   * Galaxy catalog file name sent into ``viz.load_galaxy_data``
   * etc.

 * Run your script (``visualize_voids_[algorithm].py``) on your machine.





Example scripts
===============

Included in the **VAST** repository (``VAST/example_scripts/``) is 
``visualize_voids_VoidFinder.py``, a working example script for how to run 
**VoidRender** to visualize the voids found with **VoidFinder** in a given 
catalog.

Also included in the **VAST** repository (``VAST/example_scripts/``) is 
``visualize_voids_V2.py``, a working example script for how to run 
**VoidRender** to visualize the voids found with 
:raw-html:`<strong>V<sup>2</sup></strong>` in a given catalog.  Note: 
:raw-html:`<strong>V<sup>2</sup></strong>` must be run with the ``-v`` option to 
produce output needed for the visualization.





Visualize voids
===============

The easiest way to use **VoidRender** is to create a script that

1. Reads in the output from a **VAST** void-finding algorithm and the 
   corresponding galaxy catalog within which the voids were located.
2. Defines various visualization aesthetics (void color, etc.).
3. Generates an interactive 3D visualization of the voids and galaxies.

Examples of this script are the ``visualize_voids_[algorithm].py`` files, 
located in the ``example_scripts`` directory within **VAST**.  What follows is a 
breakdown of this script, explaining the various options and functions called.

.. note:: Due to differences in the void structures found by each of the different void-finding algorithms implemented in **VAST**, there is an implementation of **VoidRender** that corresponds to each algorithm.



1. Reading in the data
----------------------

Generally, the first functions that should be called in a script running 
**VoidRender** are ``load_galaxy_data`` and ``load_void_data``::

    from vast.voidfinder.viz import load_galaxy_data, load_void_data
    
for **VoidFinder**, or::

    from vast.vsquared.viz import load_galaxy_data, load_void_data
    
for :raw-html:`<strong>V<sup>2</sup></strong>`.  These functions read a galaxy 
catalog and a void catalog into memory (as ``numpy.ndarray`` objects), 
respectively.  These load functions are provided as convenient utilities to 
access the **VAST** outputs.

The output from ``load_galaxy_data`` is a ``numpy.ndarray`` object containing 
the Cartesian coordinates of the objects in the input catalog.

The outputs from ``load_void_data`` are:
 
 * The Cartesian coordinates of the centers of the void holes 
   (**VoidFinder**) or the vertices of triangles making up void edges 
   (:raw-html:`<strong>V<sup>2</sup></strong>`) as a ``numpy.ndarray`` object
 * The radii of the void holes (**VoidFinder**) or the Cartesian components of 
   each void edge triangle's unit normal vector 
   (:raw-html:`<strong>V<sup>2</sup></strong>`)
 * ID values for the void holes (**VoidFinder**) or void ID values for the 
   triangles (:raw-html:`<strong>V<sup>2</sup></strong>`)
   
.. note:: If you want to draw lines connecting the wall galaxies to each other (as shown in :ref:`fig-vfviz`), the field and wall galaxies must be loaded into memory as separate objects.



.. _VR-params:

2. Visualization aesthetics
---------------------------

Void color
^^^^^^^^^^

The default behavior of **VoidRender** is to color all voids the same color 
(blue).  It is possible to change this color and/or assign different voids 
different colors.

To change the colors of the voids, set the ``void_hole_color`` keyword in 
**VoidRender**.  To set all voids to a single color, provide a single 
RGB\ :math:`\alpha` array.  To set different colors for the voids, provide an 
array of shape (:math:`N_{voids}`,4), where :math:`N_{voids}` corresponds to the 
number of unique void IDs in the ``holes_group_IDs`` keyword.  The number of 
holes may be different than the number of voids.


Galaxy color and size
^^^^^^^^^^^^^^^^^^^^^

The default behavior of **VoidRender** is to color all galaxies the same color 
(red).  It is possible to change this color, or to color field and wall galaxies 
differently (in **VoidFinder**).

To change the color of the galaxies (or the field galaxies), set the 
``galaxy_color`` keyword of **VoidRender** to a single RGB\ :math:`\alpha` 
array.  If a separate list of wall galaxy coordinates is provided 
(**VoidFinder** only), their display color can be set in a similar manner using 
the ``wall_galaxy_color`` keyword in **VoidRender**.  The lines connecting the 
wall galaxies will also be drawn in this same color.

The largest size of the galaxy points can be set using the 
``galaxy_display_radius`` keyword in **VoidRender**; the default is 2.  The size 
of the galaxies can be dynamically changed with the mouse scroll wheel while 
in **VoidRender**.


Sphere surface resolution
^^^^^^^^^^^^^^^^^^^^^^^^^

(**VoidFinder** only)

**VoidRender** renders the surfaces of the spheres as a set of triangles.  The 
depth of triangularization can be altered using the 
``SPHERE_TRIANGULARIZATION_DEPTH`` keyword in **VoidRender**.  An increased 
depth will result in a smoother surface, but rendering higher resolutions will 
take longer because the number of triangles increases exponentially with this 
value.  A value of 3 (default) generates 1280 triangles for each sphere; a 
value of 4 would generate 15,360 triangles for each sphere.





3. Visualizing voids
--------------------

To generate the interactive window within which the voids and galaxies are 
displayed, import the ``VoidRender`` class::

    from vast.voidfinder.viz import VoidRender
    
Then, initialize the ``VoidRender`` object with the galaxy array(s), void array, 
and additional parameters (see Section :ref:`VR-params` for details)::

    viz = VoidRender(...)
    
Finally, generate the interactive window::

    viz.run()
    
Now that the interactive window has started, the camera view can be controlled 
using typical WASD-like controls.  For full reference of all keyboard controls, 
see :ref:`VR-VF-docstring` and/or :ref:`VR-V2-docstring`.

    








