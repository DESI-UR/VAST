################
Using VoidFinder
################



Quick start
===========

A summary checklist for installing and running `VAST.VoidFinder`.

 * Clone the `GitHub repository <https://github.com/DESI-UR/VAST>`_
 * Navigate to ``VAST/VoidFinder`` and run::
    
    python setup.py install
    
   See :ref:`VF-install` for installation options.
   
 * Navigate to ``VAST/VoidFinder/scripts`` and modify
   ``SDSS_VoidFinder_dr7.py`` if appropriate.  Variables to edit might include:
   
   * ``in_directory`` and ``out_directory``
   * ``data_filename``
   * redshift and/or magnitude limits
   * Comment out ``dist_metric`` from all functions if you are using comoving distances.
   * etc.

 * Run your script (``SDSS_VoidFinder_dr7.py``, in this case) on your machine or 
   using a cluster.

The output files will be located in the directory specified by ``out_directory``.

 * ``[survey_name]_[comoving/redshift]_holes.txt``
 * ``[survey_name]_[comoving/redshift]_maximal.txt``
 * ``[survey_name]_mask.pickle``
 * ``field_galaxies.txt``
 * ``wall_galaxies.txt``

See :ref:`VF-output` for a detailed description of each of these files.




Example scripts
===============

Included in the `VAST.VoidFinder` repository (``VAST/VoidFinder/scripts/``) are 
a finite selection of example scripts:

 * ``SDSS_VoidFinder_dr7.py`` will run `VoidFinder` on the SDSS DR7 main galaxy 
   sample.  A volume-limited version of this galaxy catalog is provided with the 
   package (``VAST/VoidFinder/vollim_dr7_cbp_102709.dat``).
 * ``classifyEnvironment.py`` takes the output of `VoidFinder` (identified 
   voids) and determines which objects of a given catalog are within the voids 
   ("void" objects), which are outside the voids ("wall" objects), and which are 
   too close to the survey boundary and/or are outside the survey bounds to be 
   classified ("edge" objects).




Finding voids
=============

The easiest way to use `VAST.VoidFinder` is to create a script that

1. Filters and processes the input galaxy catalog so that it is in the correct 
   format for `VoidFinder`
2. Generates a survey mask based on the input galaxy catalog, defining the 
   boundaries within which voids can be found
3. Finds voids within the galaxy catalog

An example of this script is the ``SDSS_VoidFinder_dr7.py`` file, located in 
``VAST/VoidFinder/scripts/``.  What follows is a breakdown of this script, 
explaining the various options and functions called.


1. Preparing the data
---------------------

Pre-processing the data
^^^^^^^^^^^^^^^^^^^^^^^

Generally, the first function that should be called in a script running 
`VoidFinder` is ``file_preprocess``::

    from vast.voidfinder.preprocessing import file_preprocess
    
This function reads the input data file into memory (as an 
``astropy.table.Table`` object), checks the various column names to ensure that 
they match what is expected in other functions, and creates the file names for 
the files produced by `VoidFinder`.  While not strictly necessary to run, it 
encompasses many of the introductory steps necessary to run `VoidFinder`.

The outputs from ``file_preprocess`` are

 * The input galaxy catalog as an ``astropy.table.Table`` object
 * The distance limits within which to locate voids
 * The output file names for the void files (see :ref:`VF-output` for details)

Constructing a volume-limited sample
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is essential that the data used to identify voids is a uniform sample.  If 
one is using a redshift galaxy survey, this often means that the galaxy catalog 
is a magnitude-limited sample (targets with measured redshifts were observed out 
to some maximum magnitude limit, or minimum apparent brightness).  The galaxy 
population changes with redshift for a magnitude-limited survey, since only 
intrinsically brighter galaxies are included as the redshift increases.  
Therefore, it is often necessary to create a **volume-limited sample**, where 
the galaxies included are all drawn from the same absolute magnitude or 
luminosity distribution.

To convert from a magnitude-limited survey to a volume-limited sample, one 
must eliminate all galaxies fainter than some absolute magnitude limit and all 
galaxies beyond the redshift at which this minimum absolute magnitude was 
observed in the survey.  This is done in ``filter_galaxies``::

    from vast.voidfinder import filter_galaxies
    
The outputs from ``filter_galaxies`` are

* ``astropy.table.Table`` objects of the wall and field galaxies from the 
  volume-limited sample constructed based on the input galaxy data.
* The dimensions of the grid shape needed for finding the voids.  All of the 
  wall galaxies will be placed in this grid, and spheres will be grown from each 
  *empty* grid cell.  The side length of each grid cell is defined by the value 
  of ``hole_grid_edge_length``, an input to ``filter_galaxies``.
* The minimum coordinates of the wall galaxy sub-sample.  This defines the 
  coordinates of the corner of the grid described in the previous bullet point.
    

2. Generating a mask
--------------------

To keep voids from extending beyond the survey bounds, we use a mask based on 
the distribution of galaxies from the input galaxy data.  This mask is built 
using the ``generate_mask`` function::

    from vast.voidfinder.multizmask import generate_mask
    
This is an (ra, dec) mask, the resolution of which is based on the furthest 
extent of the galaxy data.  (A finer resolution -- pixel width -- is required 
for data samples extending to higher redshifts, since the comoving volume 
associated with any given pixel increases with distance.)

The value of the mask is a boolean representing whether or not a given (ra, dec) 
position is part of the survey, or outside the survey.  For example, if the 
resolution of the mask is 1 degree, then if ``mask[320,17] == True``, the 
right ascension of 320 degrees and declination of 17 degrees is within the 
survey.

The outputs of ``generate_mask`` are

* The survey mask, a boolean array.  Cells which are True indicate those (ra, 
  dec) locations within the galaxy survey.
* The survey mask resolution, an integer which is used to scale an object's 
  (ra, dec) coordinates to the array index within the mask where it belongs.



3. Finding voids
----------------

The heart of `VoidFinder`, voids are identified in the wall galaxy sample 
outputted from ``filter_galaxies`` in the ``find_voids`` fuunction::

    from vast.voidfinder import find_voids
    
Here, the wall galaxies are placed on in a grid (with a cell length defined by 
``hole_grid_edge_length``).  Spheres are grown from the center of every empty 
cell until they are bounded by four galaxies.  Note that the smallest sphere 
that can be grown has a diameter equal to ``hole_grid_edge_length``.

The resulting spheres are then sorted, duplicates are removed, and the list of 
spheres is iterated through to identify maximal spheres (the largest sphere in 
a void) and the additional void holes (spheres smaller than a void's maximal 
sphere that overlap with their void's maximal sphere by at least 50% of their 
volume).  The union of a maximal sphere and its void's holes defines a void.

The outputs of ``find_voids`` are the output files described in 
:ref:`VF-output`.

Parallelized void-finding
^^^^^^^^^^^^^^^^^^^^^^^^^

``find_voids`` can be run both single- and multi-threaded!  This is 
controlled via the ``num_cpus`` optional input.  The default setting is 
multi-threaded, using the total number of physical cores on the machine being 
used.  The number of cells given to each thread at a given time is set by the 
value in ``batch_size``.

To run ``find_voids`` in a single thread, set ``num_cpus = 1``.

Checkpoint files
^^^^^^^^^^^^^^^^

**Note** This capability is currently only available in the multi-process 
implementation of ``find_voids``.

In addition, the current list of void spheres found can be saved to disk 
periodically, and ``find_voids`` can be restarted from one of these files if the 
process did not complete before, for example, the job timing out on a computing 
cluster.

To generate these files, set ``save_after`` to some integer.  ``find_voids`` 
will save a file every ``save_after`` grid cells are searched.

To start ``find_voids`` from one of these files, set 
``use_start_checkpoint == True``.



.. _VF-input:

Input file
----------

As `VAST.VoidFinder` is designed to identify dynamically distinct cosmic voids 
in a galaxy distribution, it requires a galaxy catalog (or similar) on which to 
run.

This input data file is specified by the ``galaxies_filename`` variable in the 
sample ``SDSS_VoidFinder_dr7.py`` script.  Its location is specified with the 
``in_directory`` variable in the same sample script, so that the file 
``in_directory + galaxies_filename`` is opened in 
``vast.voidfinder.preprocessing.file_preprocess``.


File format
^^^^^^^^^^^

Currently supported formats for the input data file include:

 * ascii commented header (readable by ``astropy.table.Table.read``)
 * .h5


Data columns
^^^^^^^^^^^^

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



.. _VF-output:

Output
------

Each void found by `VAST.VoidFinder` is a union of spheres: one maximal sphere 
(the largest sphere that can fit within that void) and a set of smaller spheres 
(called holes).  The two files that list the identified voids are:

 * ``[survey_name]_[comoving/redshift]_maximal.txt``
 * ``[survey_name]_[comoving/redshift]_holes.txt``

Both of these files are described in more detail below.

Additional files that can be produced during the process (which may or may not 
be useful to the user post-void-finding) include

 * ``[survey_name]_mask.pickle`` -- The sky mask of the survey.  The resolution 
   of the mask is computed to be optimal for void-finding at the highest 
   redshift that voids are found.  See :ref:`VF-mask` for details on the file 
   contents.
 * ``[survey_name]_field_gal_file.txt`` -- A list of the field galaxies removed 
   from the input galaxy file prior to void-finding.
 * ``[survey_name]_wall_gal_file.txt`` -- A list of the wall galaxies that are 
   used to define the non-void regions.

The union of the field and wall galaxy files is the same as the input data file, 
after any redshift and/or magnitude cuts are applied.

.. list-table:: Maximal sphere output file
   :widths: 25 25 25 50
   :header-rows: 1
   
   * - Column name
     - Data type
     - Unit
     - Comment
   * - x
     - float
     - Mpc/h
     - x-coordinate of the center of the maximal sphere
   * - y
     - float
     - Mpc/h
     - y-coordinate of the center of the maximal sphere
   * - z
     - float
     - Mpc/h
     - z-coordinate of the center of the maximal sphere
   * - radius
     - float
     - Mpc/h
     - Radius of the maximal sphere
   * - flag
     - integer
     - 
     - Unique number associated to each void.  With only one maximal sphere per 
       void, this means that each maximal sphere has a different ``flag`` value.
   * - r
     - float
     - Mpc/h
     - Comoving distance to the center of the maximal sphere
   * - ra
     - float
     - degrees
     - Right ascension of the center of the maximal sphere
   * - dec
     - float
     - degrees
     - Declination of the center of the maximal sphere
     
.. list-table:: Holes output file
   :widths: 25 25 25 50
   :header-rows: 1
   
   * - Column name
     - Data type
     - Unit
     - Comment
   * - x
     - float
     - Mpc/h
     - x-coordinate of the center of the hole (sphere)
   * - y
     - float
     - Mpc/h
     - y-coordinate of the center of the hole (sphere)
   * - z
     - float
     - Mpc/h
     - z-coordinate of the center of the hole (sphere)
   * - radius
     - float
     - Mpc/h
     - Radius of the hole (sphere)
   * - flag
     - integer
     - 
     - Unique number associated to each void.  The union of all holes with the 
       same flag value define that void.



Adjustable parameters
---------------------

.. list-table::
   :widths: 25 25 10 10 30
   :header-rows: 1
   
   * - Keyword
     - Function(s)
     - Data type
     - Default value
     - Comment
   * - ``mag_cut``
     - ``file_preprocess``, ``filter_galaxies``
     - boolean
     - True
     - Determines whether or not to apply an absolute magnitude cut to the 
       input galaxy catalog.
   * - ``magnitude_limit``
     - ``filter_galaxies``
     - float
     - -20.09
     - Faintest absolute magnitude permitted in the galaxy catalog in which the 
       voids are going to be identified.  Only used if ``mag_cut == True``.
   * - ``rm_isolated``
     - ``file_preprocess``, ``filter_galaxies``
     - boolean
     - True
     - Determines whether or not to remove isolated galaxies from the input 
       galaxy catalog.  If ``mag_cut == True``, this removal occurs after the 
       magnitude limit is applied.
   * - ``sep_neighbor``
     - ``filter_galaxies``
     - integer
     - 3
     - If ``rm_isolated`` is true, then the distance to the Nth nearest 
       neighbor is used to determine whether or not a galaxy is isolated, where 
       N is defined by the value of this variable.
   * - ``dist_metric``
     - ``file_preprocess``, ``filter_galaxies``, ``generate_mask``
     - string
     - comoving
     - Determines which distance metric to use.  The options are ``comoving`` 
       (calculates the comoving distance to the galaxies based on the given 
       cosmology) or ``redshift`` (scales the distance to the galaxy by 
       :math:`c/H_0`, where :math:`H_0 = 100h`).
   * - ``min_z``, ``max_z``
     - ``file_preprocess``
     - float
     - None
     - The minimum and maximum redshift limits within which to find the voids.  
       If left to ``None``, the minimum and maximum redshift range of the 
       input galaxy catalog is used.
   * - ``dist_limits``
     - ``filter_galaxies``
     - list of floats
     - None
     - The minimum and maximum distance limits within which to find the voids.  
       If none are given, then no distance cut is applied to the input galaxy 
       sample.
   * - ``Omega_m``
     - ``file_preprocess``
     - float
     - 0.3
     - Value of :math:`\Omega_M`.  This is used only when calculating the 
       comoving distance.
   * - ``h``
     - ``file_preprocess``, ``filter_galaxies``, ``generate_mask``
     - float
     - 1
     - Reduced Hubble constant, where :math:`H_0 = 100h`.  With the default 
       value of 1, all distances will be in units of Mpc/h.
   * - ``verbose``
     - ``file_preprocess``, ``filter_galaxies``, ``find_voids``
     - integer
     - 0
     - Determines how much of the print statements are generated.  A value of 0 
       (the default) displays the minimum status statements.
   * - ``print_after``
     - ``find_voids``
     - integer
     - 5
     - Number of seconds to wait between status updates while growing the 
       spheres in ``find_voids``.
   * - ``write_table``
     - ``filter_galaxies``
     - boolean
     - True
     - Determines whether or not to save the 
       ``[survey_name]_field_gal_file.txt`` and 
       ``[survey_name]_wall_gal_file.txt`` files to disk.  If so, these files 
       will be saved to the location specified by ``out_directory``.
   * - ``hole_grid_edge_length``
     - ``filter_galaxies``, ``find_voids``
     - float
     - 5.0
     - The length of the edge of one cell in the grid used to identify where to 
       start growing potential void spheres from.  Units are Mpc/h.
   * - ``galaxy_map_grid_edge_length``
     - ``find_voids``
     - float
     - None
     - The length of the edge of one cell in the grid used to find the nearest 
       galaxies while growing spheres.  If the value is ``None``, this length 
       will equal :math:`3\times` ``hole_grid_edge_length``.  Units are Mpc/h.
   * - ``smooth_mask``
     - ``generate_mask``
     - boolean
     - True
     - If this value is True, small holes in the mask (single cells without any 
       galaxy in them that are surrounded by at least three cells which have 
       galaxies in them) are unmasked.
   * - ``max_hole_mask_overlap``
     - ``find_voids``
     - float
     - 0.1
     - Maximum allowed fraction of a void sphere to extend outside the survey 
       mask.  Note that a void sphere's center *must* be within the survey, so 
       this value can range from (0,0.5).
   * - ``hole_center_iter_dist``
     - ``find_voids``
     - float
     - 1.0
     - Distance to shift the center of a growing sphere on each iteration while 
       finding the four bounding galaxies.  Units are Mpc/h.
   * - ``maximal_spheres_filename``
     - ``find_voids``
     - string
     - maximal_spheres.txt
     - Location to save the maximal spheres.  If ``file_preprocess`` was run, 
       this should be set to ``out1_filename``.
   * - ``void_table_filename``
     - ``find_voids``
     - string
     - voids_table.txt
     - Location to save the list of void spheres.  If ``file_preprocess`` was 
       run, this should be set to ``out2_filename``.
   * - ``potential_voids_filename``
     - ``find_voids``
     - string
     - potential_voids_list.txt
     - Location to save the list of potential voids spheres (all spheres found, 
       no filtering yet implemented on the list).  An ideal name for this file 
       would be ``out_directory + survey_name + 'potential_voids_list.txt'``.
   * - ``num_cpus``
     - ``find_voids``
     - integer
     - None
     - Number of CPUs to use in the multi-threaded implementation of 
       ``find_voids``.  The default value will use the number of physical cores 
       on the machine being used.  If you want to run ``find_voids`` in a 
       single thread, set ``num_cpus = 1``.
   * - ``save_after``
     - ``find_voids``
     - integer
     - None
     - If this is not ``None``, then ``find_voids`` will save a 
       ``VoidFinderCheckpoint.h5`` file after *approximately* every 
       ``save_after`` cells have been processed.  Each new checkpoint file will 
       overwrite the previous file.
   * - ``use_start_checkpoint``
     - ``find_voids``
     - boolean
     - False
     - Determines whether or not to start ``find_voids`` with a previously 
       generated ``VoidFinderCheckpoint.h5`` file.  If ``False``, ``find_voids`` 
       will start growing spheres fresh.
   * - ``batch_size``
     - ``find_voids``
     - integer
     - 10,000
     - Number of potential cells to evaluate at a time.  Only used in the 
       multi-threaded mode of ``find_voids``.




Using the output
================

Is my object in a void?
-----------------------

Because `VAST.VoidFinder` defines voids as a union of spheres, it is relatively 
simple to determine whether or not an object is located within a void: if its 
location falls within one of the spheres listed in the ``_holes.txt`` output 
file (see :ref:`VF-output`), then it is within a void!

Note that only the centers of the maximal spheres are given in both Cartesian 
(x, y, z) and sky coordinates (ra, dec, distance).  Therefore, it is often 
necessary to convert an object's (ra, dec, redshift) coordinates to (x, y, z) 
coordinates to determine whether or not the object lives within a void.  One 
might find useful the ``ra_dec_to_xyz`` function::

    from vast.voidfinder import ra_dec_to_xyz

It is necessary to use the same cosmology and/or distance metric as was used 
when finding the voids for this conversion!  If using comoving distances, 
``ra_dec_to_xyz`` expects the comoving distance to be in a column named 
``Rgal``.  If your object / data file does not already have this column, you can 
compute the comoving distance using the ``z_to_comoving_dist`` function::

    from vast.voidfinder.distance import z_to_comoving_dist
    
See the example script ``classifyEnvironment.py`` (found in the 
``VAST/VoidFinder/scripts/`` directory) for a working example of how to 
determine whether an object is within a void, in the wall, too close to the 
survey boundary to classify, or outside the survey bounds. 
 
 
 
 