
.. role:: raw-html(raw)
    :format: html


#################
Using V\ :sup:`2`
#################



Quick start
===========

A summary checklist for installing and running 
:raw-html:`<strong>V<sup>2</sup></strong>`.

 * Clone the `GitHub repository <https://github.com/DESI-UR/VAST>`_
 * Navigate to ``VAST/Vsquared`` and run::
    
    python setup.py install
    
   See :ref:`V2-install` for installation options.
   
 * Navigate to ``VAST/Vsquared/scripts`` and modify ``DR7_config.ini`` if 
   appropriate.  Fields to edit might include:
   
   * Input catalog and Survey name
   * Output directory
   * redshift and/or magnitude limits
   * Minimum void radius
   * etc.

 * Run ``vsquared.py`` from the ``VAST/Vsquared/scripts`` directory on your 

   machine or using a cluster::
   
    python vsquared.py -c DR7_config.ini
    
   .. note:: Include the ``-v`` option to produce the output required for **VoidRender**.

The output files will be located in the directory specified by the Output 
directory.

 * ``[survey_name]_galviz.dat``
 * ``[survey_name]_galzones.dat``
 * ``[survey_name]_triangles.dat``
 * ``[survey_name]_zobovoids.dat``
 * ``[survey_name]_zonevoids.dat``

See :ref:`V2-output` for a detailed description of each of these files.





Example configuration file
==========================

Included in the :raw-html:`<strong>V<sup>2</sup></strong>` repository 
(``VAST/Vsquared/scripts/``) are a finite selection of example configuration 
files:

 * ``DR7_config.ini`` contains the settings to run 
   :raw-html:`<strong>V<sup>2</sup></strong>` on the SDSS DR7 main galaxy 
   sample.  A volume-limited version of this galaxy catalog is provided with the 
   package 
   (``VAST/Vsquared/data/vollim_dr7_cbp_102709.fits``).

See :ref:`V2-config` for details on the configuration file options.




Finding voids
=============


Script
------

The easiest way to use :raw-html:`<strong>V<sup>2</sup></strong>` is to use the 
``vsquared.py`` script, located in ``VAST/Vsquared/scripts/``. For usage 
information, run::

    python vsquared.py --help


In a Python Shell
-----------------

Finding voids can also be done in a Python shell, using the 
``vast.vsquared.zobov.Zobov`` class and its methods:

1. Create a ``Zobov`` object using the desired configuration file and additional 
   input parameters::

       newZobov = Zobov("DR7_config.ini")
   
   See :ref:`V2-init` for details on the initialization method's arguments.
2. Apply a void-pruning method to the voids found::

       newZobov.sortVoids()
   
   See :ref:`V2-sort` for details on this method's arguments.
3. Save the results to disk (these methods take no additional arguments)::

       newZobov.saveVoids()
       newZobov.saveZones()
       newZobov.preViz() #if intending to visualize results


.. _V2-config:

Configuration File Options
--------------------------

Using :raw-html:`<strong>V<sup>2</sup></strong>` requires a configuration file 
with the following options:

.. list-table:: Configuration file options
   :width: 100%
   :widths: 25 25 25 25 50
   :header-rows: 1

   * - Key
     - Section
     - Data type
     - Unit
     - Comment
   * - ``Input Catalog``
     - Paths
     - string
     - 
     - Path to the input data catalog
   * - ``Survey Name``
     - Paths
     - string
     - 
     - Survey identifier to use in output file names
   * - ``Output Directory``
     - Paths
     - string
     - 
     - Path to the directory where output files will be saved
   * - ``H_0``
     - Cosmology
     - float
     - (km/s)/Mpc
     - Hubble constant of the desired cosmology
   * - ``Omega_m``
     - Cosmology
     - float
     - 
     - Dimensionless matter density parameter of the desired cosmology
   * - ``redshift_min``
     - Settings
     - float
     - 
     - The redshift above which void-finding will be applied
   * - ``redshift_max``
     - Settings
     - float
     - 
     - The redshift below which void-finding will be applied
   * - ``rabsmag_min``
     - Settings
     - float
     - 
     - The minimum magnitude for a galaxy to be used for void-finding
   * - ``radius_min``
     - Settings
     - float
     - Mpc/h
     - The minimum radius for a void candidate to be considered a true void
   * - ``nside``
     - Settings
     - integer
     - 
     - The NSIDE parameter used in the HEALPix pixelization of the survey mask; 
       must be a power of 2
   * - ``redshift_step``
     - Settings
     - float
     - 
     - The step size used to create a comoving-distance-to-redshift lookup table 
   
   
   
   
Input
=====

As :raw-html:`<strong>V<sup>2</sup></strong>` is designed to identify voids in a 
galaxy distribution, it requires a galaxy catalog (or similar) on which to run.

This input data file is specified by the ``Input Catalog`` field in the sample 
``DR7_config.ini`` configuration file.


File format
-----------

Currently supported formats for the input data file include:

 * .fits


Data columns
------------

.. list-table:: Required columns for input file
   :width: 100%
   :widths: 25 25 25 50
   :header-rows: 1

   * - Column name
     - Data type
     - Unit
     - Comment
   * - ra
     - float
     - degrees
     - Right ascension
   * - dec
     - float
     - degrees
     - Declination
   * - redshift
     - float
     - 
     - Redshift
     
.. list-table:: Optional columns for input file
   :width: 5in
   :header-rows: 1
   
   * - Column name
     - Data type
     - Unit
     - Comment
   * - rabsmag
     - float
     - 
     - Absolute magnitude.  Only used if ``rabsmag_min`` is not ``None``.




.. _V2-output:

Output
======

Each void found by :raw-html:`<strong>V<sup>2</sup></strong>` is a set of 
Voronoi cells.  The files that list the identified voids are:

 * ``[survey_name]_galzones.dat`` -- Identifies the zone to which each galaxy 
   belongs.
 * ``[survey_name]_zonevoids.dat`` -- Identifies the void to which each zone 
   belongs.
 * ``[survey_name]_zobovoids.dat`` -- Identifies the coordinates, effective 
   radius, and ellipticity of each void.

Each of these files is described in more detail below.

Additional files that are produced during the process (which may or may not be 
useful to the user post-void-finding) include
 
 * ``[survey_name]_triangles.dat`` -- Identifies the vertices, normal vector,
   and void membership of each triangle making up a void boundary
 * ``[survey_name]_galviz.dat`` -- Identifies the voids to which each galaxy and
   its nearest neighbor belong

.. list-table:: ``_galzones`` output file
   :widths: 25 25 50
   :header-rows: 1
   
   * - Column name
     - Data type
     - Comment
   * - gal
     - integer
     - Unique galaxy identifier
   * - zone
     - integer
     - Unique identifier of the galaxy's containing zone
   * - depth
     - integer
     - Number of adjacent voronoi cells between the galaxy's cell and the edge 
       of its zone
   * - edge
     - integer
     - 1 if the galaxy's voronoi cell extends outside the survey mask, 0 
       otherwise
   * - out
     - integer
     - 1 if the galaxy is located outside the survey mask, 0 otherwise
     
.. list-table:: ``_zonevoids`` output file
   :widths: 25 25 50
   :header-rows: 1
   
   * - Column name
     - Data type
     - Comment
   * - zone
     - integer
     - Unique zone identifier
   * - void0
     - integer
     - Unique identifier of the zone's smallest containing void; -1 if zone is 
       not part of a void
   * - void1
     - integer
     - Unique identifier of the zone's largest containing void; -1 if zone is 
       not part of a void

.. list-table:: ``_zobovoids`` output file
   :widths: 25 25 25 50
   :header-rows: 1

   * - Column name
     - Data type
     - Unit
     - Comment
   * - x
     - float
     - Mpc/h
     - x-coordinate of the weighted center of the void
   * - y
     - float
     - Mpc/h
     - y-coordinate of the weighted center of the void
   * - z
     - float
     - Mpc/h
     - z-coordinate of the weighted center of the void
   * - redshift
     - float
     - 
     - redshift of the weighted center of the void
   * - ra
     - float
     - degrees
     - right ascension of the weighted center of the void
   * - dec
     - float
     - degrees
     - declination of the weighted center of the void
   * - radius
     - float
     - Mpc/h
     - effective radius of the void
   * - x1
     - float
     - 
     - normalized x-component of the void's first ellipsoid axis
   * - y1
     - float
     - 
     - normalized y-component of the void's first ellipsoid axis
   * - z1
     - float
     - 
     - normalized z-component of the void's first ellipsoid axis
   * - x2
     - float
     - 
     - normalized x-component of the void's second ellipsoid axis
   * - y2
     - float
     - 
     - normalized y-component of the void's second ellipsoid axis
   * - z2
     - float
     - 
     - normalized z-component of the void's second ellipsoid axis
   * - x3
     - float
     - 
     - normalized x-component of the void's third ellipsoid axis
   * - y3
     - float
     - 
     - normalized y-component of the void's third ellipsoid axis
   * - z3
     - float
     - 
     - normalized z-component of the void's third ellipsoid axis

.. list-table:: ``_triangles`` output file
   :widths: 25 25 25 50
   :header-rows: 1

   * - Column name
     - Data type
     - Unit
     - Comment
   * - void_id
     - integer
     - 
     - Unique identifier of the triangle's containing void
   * - n_x
     - float
     - 
     - normalized x-component of the triangle's normal vector
   * - n_y
     - float
     - 
     - normalized y-component of the triangle's normal vector
   * - n_z
     - float
     - 
     - normalized z-component of the triangle's normal vector
   * - p1_x
     - float
     - Mpc/h
     - x-coordinate of the triangle's first vertex
   * - p1_y
     - float
     - Mpc/h
     - y-coordinate of the triangle's first vertex
   * - p1_z
     - float
     - Mpc/h
     - z-coordinate of the triangle's first vertex
   * - p2_x
     - float
     - Mpc/h
     - x-coordinate of the triangle's second vertex
   * - p2_y
     - float
     - Mpc/h
     - y-coordinate of the triangle's second vertex
   * - p2_z
     - float
     - Mpc/h
     - z-coordinate of the triangle's second vertex
   * - p3_x
     - float
     - Mpc/h
     - x-coordinate of the triangle's third vertex
   * - p3_y
     - float
     - Mpc/h
     - y-coordinate of the triangle's third vertex
   * - p3_z
     - float
     - Mpc/h
     - z-coordinate of the triangle's third vertex

.. list-table:: ``_galviz`` output file
   :widths: 25 25 50
   :header-rows: 1
   
   * - Column name
     - Data type
     - Comment
   * - gid
     - integer
     - Unique galaxy identifier
   * - g2v
     - integer
     - Unique identifier of the galaxy's containing void
   * - g2v2
     - integer
     - Unique identifier of the containing void of the galaxy's nearest
       neighbor




Using the output
================

Is my object in a void?
-----------------------

Because voids found by :raw-html:`<strong>V<sup>2</sup></strong>` are formed 
from zones, which are unions of objects' voronoi cells, each object's void 
membership is easily determined from the output.  The ``_galzones.dat`` output 
file (see :ref:`V2-output`) contains each object's zone membership, and the 
``_zonevoids.dat`` output file contains each zone's void membership.  If the 
values in the ``void0`` and ``void1`` columns of a zone are ``-1``, the zone 
does not belong to any void, and any objects contained within that zone are not 
in a void.
 
 


