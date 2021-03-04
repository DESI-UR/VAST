#################
Using V\ :sup:`2`
#################



Quick start
===========

A summary to-do list for installing and running `VAST.Vsquared`.

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
   machine or using a cluster.

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

Included in the `VAST.Vsquared` repository (``VAST/Vsquared/scripts/``) are 
a finite selection of example configuration files:

 * ``DR7_config.ini`` contains the settings to run `Vsquared` on the SDSS DR7 
   main galaxy sample.  A volume-limited version of this galaxy catalog is 
   provided with the package 
   (``VAST/Vsquared/data/vollim_dr7_cbp_102709.fits``).
   
   
   
   
Input
=====

As `VAST.Vsquared` is designed to identify voids in a galaxy distribution, it 
requires a galaxy catalog (or similar) on which to run.

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
   * - Rgal
     - float
     - Mpc/h
     - Comoving distance.  Only used if ``dist_metric`` is set to ``comoving``.  
       If this column is not provided, and the distance metric is set to 
       ``comoving``, then the comoving distances will be calculated based on the 
       given cosmological parameters and the redshift column.
   * - rabsmag
     - float
     - 
     - Absolute magnitude.  Only used if ``mag_cut == True``.




.. _V2-output:

Output
======

Each void found by `VAST.Vsquared` is a set of Voronoi tesselations.  The files 
that list the identified voids are:

 * ``[survey_name]_galzones.dat`` -- Identifies the zone to which each galaxy 
   belongs.
 * ``[survey_name]_zonevoids.txt`` -- Identifies the void to which each zone 
   belongs.

Both of these files are described in more detail below.

Additional files that are produced during the process (which may or may not be 
useful to the user post-void-finding) include

 * ``[survey_name]_galviz.dat`` -- 
 * ``[survey_name]_triangles.dat`` -- 
 * ``[survey_name]_zobovoids.dat`` -- 

.. list-table:: ``_galzones`` output file
   :widths: 25 25 25 50
   :header-rows: 1
   
   * - Column name
     - Data type
     - Unit
     - Comment
   * - something
     - something
     - something
     - something
     
.. list-table:: ``_zonevoids`` output file
   :widths: 25 25 25 50
   :header-rows: 1
   
   * - Column name
     - Data type
     - Unit
     - Comment
   * - something
     - something
     - something
     - something
   
   
   
   