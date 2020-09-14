#VoidFinder Function to do just about everything


import numpy as np

from astropy.table import Table

import time

import matplotlib as mpl

import matplotlib.pyplot as plt

from .hole_combine import combine_holes, combine_holes_2

from .voidfinder_functions import mesh_galaxies, \
                                  in_mask, \
                                  not_in_mask, \
                                  in_survey, \
                                  save_maximals, \
                                  mesh_galaxies_dict
                                  #build_mask, \

from .table_functions import add_row, \
                             subtract_row, \
                             to_vector, \
                             to_array, \
                             table_dtype_cast, \
                             table_divide

from .volume_cut import volume_cut, check_hole_bounds

from .avsepcalc import av_sep_calc

from .mag_cutoff_function import mag_cut, field_gal_cut

from ._voidfinder import _hole_finder

from .constants import c

from ._voidfinder_cython import check_mask_overlap


################################################################################
# I made these constants into default parameters in filter_galaxies() and 
# find_voids() because a user may wish to try different grid spacings and such.  
# I left DtoR and RtoD because they will never ever change since they're based 
# on pi.
#-------------------------------------------------------------------------------
#dl = 5           # Cell side length [Mpc/h]
#dr = 1.          # Distance to shift the hole centers
#c = 3e5
DtoR = np.pi/180.
RtoD = 180./np.pi
################################################################################



def filter_galaxies(galaxy_table,
                    survey_name, 
                    mag_cut_flag=True,
                    filter_flux=None, 
                    rm_isolated_flag=True,
                    write_table=True,
                    sep_neighbor=3,
                    distance_metric='comoving', 
                    h=1.0,
                    hole_grid_edge_length=5.0,
                    magnitude_limit=-20.0,
                    verbose=0):
    """
    Description
    ===========
    
    A hodge podge of miscellaneous tasks which need to be done to format the data into
    something the main find_voids() function can use.
    
    1). Optional magnitude cut
    2). Convert from ra-dec-redshift space into xyz space
    3). Calculate the hole search grid shape
    4). Optional remove isolated galaxies by partitioning them into wall (non-isolated)
            and field (isolated) groups
    5). Optionally write out the wall and field galaxies to disk
    
    
    Parameters
    ==========
    
    galaxy_table : astropy.table of shape (N,?)
        variable number of required columns.  If doing magnitude cut, must include
        'rabsmag' column. If distance metric is 'comoving', must include 'Rgal'
        column, otherwise must include 'redshift'.  Also must always include 'ra' 
        and 'dec'
        
    survey_name : str
        Name of the galxy catalog, string value to prepend or append to output names
        
    mag_cut_flag : bool
        whether or not to cut on magnitude, removing galaxies less than
        magnitude_limit
    
    filter_flux : float
        
    magnitude_limit : float
        value at which to perform magnitude cut
        
    rm_isolated_flag : bool
        whether or not to perform Nth neighbor distance calculation, and use it
        to partition the input galaxies into wall and field galaxies
        
    write_table : bool
        use astropy.table.Table.write to write out the wall and field galaxies
        to file
        
    sep_neighbor : int, positive
        if rm_isolated_flag is true, find the Nth galaxy neighbors based on this value
        
    distance_metric : str
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).
        
    hole_grid_edge_length : float
        length in Mpc/h of the edge of one cell of a grid cube for the search grid
        
    verbose : int
        values greater than zero indicate to print output
        
        
    Returns
    =======
    
    wall_gals_xyz : numpy.ndarray of shape (K,3)
        the galaxies which were designated not to be isolated
    
    field_gals_xyz : numpy.ndarray of shape (L,3)
        the galaxies designated as isolated
    
    hole_grid_shape : tuple of 3 integers (i,j,k)
        shape of the hole search grid
    
    coords_min : numpy.ndarray of shape (3,)
        coordinates of the minimum of the survey used for converting from
        xyz space into ijk space
    
    """
    
    
    
    ################################################################################
    # PRE-PROCESS DATA
    # Filter based on magnitude and convert from redshift to radius if necessary
    #
    ################################################################################
    if verbose > 0:
        print('Filter Galaxies Start', flush=True)

    # Remove faint galaxies

    if mag_cut_flag:
        
        galaxy_table = galaxy_table[galaxy_table['rabsmag'] < magnitude_limit]

    coords_xyz = ra_dec_to_xyz(galaxy_table,
                               distance_metric,
                               h)
    
    ################################################################################
    #
    # Grid shape
    #
    ################################################################################
    
    hole_grid_shape, coords_min = calculate_grid(coords_xyz,
                                                 hole_grid_edge_length)
    print('Calculated grid')
    ################################################################################
    #
    #   SEPARATION
    #
    ################################################################################
    
    if rm_isolated_flag:
        
        wall_gals_xyz, field_gals_xyz = wall_field_separation(coords_xyz,
                                                              sep_neighbor=sep_neighbor,
                                                              verbose=verbose)

    else:
        
        wall_gals_xyz = coords_xyz
        
        field_gals_xyz = np.array([])

    ################################################################################
    #
    # Write results to disk if desired
    #
    ################################################################################
    
    if write_table:
    
    
        write_start = time.time()
        
    
        wall_xyz_table = Table(data=wall_gals_xyz, names=["x", "y", "z"])
        
        wall_xyz_table.write(survey_name + 'wall_gal_file.txt', format='ascii.commented_header', overwrite=True)
    
        
        field_xyz_table = Table(data=field_gals_xyz, names=["x", "y", "z"])
    
        field_xyz_table.write(survey_name + 'field_gal_file.txt', format='ascii.commented_header', overwrite=True)
        
        
        if verbose > 0:
            
            print("Time to write field and wall tables: ", time.time() - write_start, flush=True)


    nf =  len(field_gals_xyz)
    
    nwall = len(wall_gals_xyz)
    
    if verbose > 0:
        
        print('Number of field gals:', nf, 'Number of wall gals:', nwall, flush=True)

    return wall_gals_xyz, field_gals_xyz, hole_grid_shape, coords_min




def filter_flux(galaxy_table,
                    survey_name,
                    mag_cut_flag=True,
                    flux_cut=None,
                    rm_isolated_flag=True,
                    write_table=True,
                    sep_neighbor=3,
                    distance_metric='comoving',
                    h=1.0,
                    hole_grid_edge_length=5.0,
                    magnitude_limit=-20.0,
                    verbose=0):
    """
    Description
    ===========

    A hodge podge of miscellaneous tasks which need to be done to format the data into
    something the main find_voids() function can use.

    1). Optional magnitude cut
    2). Convert from ra-dec-redshift space into xyz space
    3). Calculate the hole search grid shape
    4). Optional remove isolated galaxies by partitioning them into wall (non-isolated)
            and field (isolated) groups
    5). Optionally write out the wall and field galaxies to disk
    Parameters
    ==========

    galaxy_table : astropy.table of shape (N,?)
        variable number of required columns.  If doing magnitude cut, must include
        'rabsmag' column. If distance metric is 'comoving', must include 'Rgal'
        column, otherwise must include 'redshift'.  Also must always include 'ra'
        and 'dec'

    survey_name : str
        Name of the galxy catalog, string value to prepend or append to output names

    mag_cut_flag : bool
        whether or not to cut on magnitude, removing galaxies less than
        magnitude_limit

    flux_cut : float

    magnitude_limit : float
        value at which to perform magnitude cut

    rm_isolated_flag : bool
        whether or not to perform Nth neighbor distance calculation, and use it
        to partition the input galaxies into wall and field galaxies

    write_table : bool
        use astropy.table.Table.write to write out the wall and field galaxies
        to file

    sep_neighbor : int, positive
        if rm_isolated_flag is true, find the Nth galaxy neighbors based on this value

    distance_metric : str
        Distance metric to use in calculations.  Options are 'comoving'
        (default; distance dependent on cosmology) and 'redshift' (distance
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where
        H0 = 100h).

    hole_grid_edge_length : float
        length in Mpc/h of the edge of one cell of a grid cube for the search grid

    verbose : int
        values greater than zero indicate to print output

     Returns
    =======

    wall_gals_xyz : numpy.ndarray of shape (K,3)
        the galaxies which were designated not to be isolated

    field_gals_xyz : numpy.ndarray of shape (L,3)
        the galaxies designated as isolated

    hole_grid_shape : tuple of 3 integers (i,j,k)
        shape of the hole search grid

    coords_min : numpy.ndarray of shape (3,)
        coordinates of the minimum of the survey used for converting from
        xyz space into ijk space

    """



    ################################################################################
    # PRE-PROCESS DATA
    # Filter based on magnitude and convert from redshift to radius if necessary
    #
    ################################################################################
    # mag_cut_flag=False #I have added because seems like it doesn't see it.
    if verbose > 0:
        print('Filter Galaxies Start', flush=True)

    # Remove faint galaxies
    print(len(galaxy_table))
    print(mag_cut_flag) 

    if mag_cut_flag:
        galaxy_table = galaxy_table[galaxy_table['rabsmag'] < magnitude_limit]
    
    print(galaxy_table)
    coords_xyz = ra_dec_to_xyz(galaxy_table,
                               distance_metric,
                               h)

    ################################################################################
    #
    #Filter transmission flux rate or Transmitted Flux Density
    #
    ################################################################################
    if verbose > 0:
        print('Filter Transmitted Flux Density Start', flush=True)
    
    print(galaxy_table[0:5])
    print(galaxy_table['rabsmag'][0:5])
    # Remove ?tomographic maps                                                                                                                                                                             
    #Plot the hoistogram for all of the data without cut
    plt.figure()
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    plt.grid(True,ls='-.',alpha=.4)
    plt.title(r'Histogram for Flux Contrast',fontsize=16)
    plt.xlabel(r'Flux Contrast $\delta$',fontsize=14)
    plt.ylabel(r'Number',fontsize=18)

    # plt.hist(galaxy_table['rabsmag'] ,bins=range(min(galaxy_table['rabsmag']), max(galaxy_table['rabsmag']) + 0.1, 0.1), color='teal')
    plt.hist(galaxy_table['rabsmag'])  #, color='teal') 
    #plt.show()
    plt.savefig('galaxy-original.png')
    print(len(galaxy_table))
    
    if flux_cut is None:
        print('No filter is applied on transmission flux rate.')
    else:
        print('Filter is applied on transmission flux rate.')
        print(flux_cut)
        galaxy_table = galaxy_table[galaxy_table['rabsmag'] < flux_cut]
        print(len(galaxy_table))
        print(galaxy_table[0:5])
    
    plt.figure()
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    plt.grid(True,ls='-.',alpha=.4)
    plt.title(r'Histogram for Flux Contrast',fontsize=16)
    plt.xlabel(r'Flux Contrast $\delta$',fontsize=14)
    plt.ylabel(r'Number',fontsize=18)

    # plt.hist(galaxy_table['rabsmag'] ,bins=range(min(galaxy_table['rabsmag']), max(galaxy_table['rabsmag']) + 0.1, 0.1), color='teal')                                                                    
    plt.hist(galaxy_table['rabsmag'])  #, color='teal')                                                                                                                                                     
    #plt.show()                                                                                                                                                                                                
    plt.savefig('galaxy-cut.png')

    coords_xyz = ra_dec_to_xyz(galaxy_table,
                               distance_metric,
                               h)


    ################################################################################
    #
    # Grid shape
    #
    ################################################################################

    hole_grid_shape, coords_min = calculate_grid(coords_xyz,
                                                 hole_grid_edge_length)
    print('Calculated grid')
    ################################################################################
    #
    #   SEPARATION
    #
    ################################################################################

    if rm_isolated_flag:

        wall_gals_xyz, field_gals_xyz = wall_field_separation(coords_xyz,
                                                              sep_neighbor=sep_neighbor,
                                                              verbose=verbose)

    else:

        wall_gals_xyz = coords_xyz

        field_gals_xyz = np.array([])


    ################################################################################
    #
    # Write results to disk if desired
    #
    ################################################################################

    if write_table:


        write_start = time.time()


        wall_xyz_table = Table(data=wall_gals_xyz, names=["x", "y", "z"])

        wall_xyz_table.write(survey_name + 'wall_gal_file.txt', format='ascii.commented_header', overwrite=True)


        field_xyz_table = Table(data=field_gals_xyz, names=["x", "y", "z"])

        field_xyz_table.write(survey_name + 'field_gal_file.txt', format='ascii.commented_header', overwrite=True)


        if verbose > 0:

            print("Time to write field and wall tables: ", time.time() - write_start, flush=True)


    nf =  len(field_gals_xyz)

    nwall = len(wall_gals_xyz)

    if verbose > 0:

        print('Number of field gals:', nf, 'Number of wall gals:', nwall, flush=True)

    return wall_gals_xyz, field_gals_xyz, hole_grid_shape, coords_min









def ra_dec_to_xyz(galaxy_table,
                  distance_metric='comoving',
                  h=1.0,
                  ):
    """
    Description
    ===========
    
    Convert galaxy coordinates from ra-dec-redshift space into xyz space.
    
    
    Parameters
    ==========
    
    galaxy_table : astropy.table of shape (N,?)
        must contain columns 'ra' and 'dec' in degrees, and either 'Rgal' in who knows
        what unit if distance_metric is 'comoving' or 'redshift' for everything else
        
    distance_metric : str
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).
        
    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).
        
        
    Returns
    =======
    
    coords_xyz : numpy.ndarray of shape (N,3)
        values of the galaxies in xyz space
    """
    
    
    if distance_metric == 'comoving':
        
        r_gal = galaxy_table['Rgal'].data
        
    else:
        
        r_gal = c*galaxy_table['z'].data/(100*h)
        
        
    ra = galaxy_table['ra'].data
    
    dec = galaxy_table['dec'].data
    
    ################################################################################
    # Convert from ra-dec-radius space to xyz space
    ################################################################################
    
    ra_radian = ra*DtoR
    
    dec_radian = dec*DtoR
    
    x = r_gal*np.cos(ra_radian)*np.cos(dec_radian)
    
    y = r_gal*np.sin(ra_radian)*np.cos(dec_radian)
    
    z = r_gal*np.sin(dec_radian)
    
    num_gal = x.shape[0]
    
    coords_xyz = np.concatenate((x.reshape(num_gal,1),
                                 y.reshape(num_gal,1),
                                 z.reshape(num_gal,1)), axis=1)
    
    return coords_xyz







def calculate_grid(galaxy_coords_xyz,
                   hole_grid_edge_length):
    """
    Description
    ===========
    
    Given a galaxy survey in xyz/Cartesian coordinates and the length
    of a cubical grid cell, calculate the cubical grid shape which will
    contain the survey.
    
    The cubical grid is obtained by finding the minimum and maximum values in each
    of the 3 dimensions, calculating the distance of the survey in each dimension, and
    dividing each dimension by the cell grid length to get the number of required grid
    cells in each dimension.
        
    In order to transform additional points to their closest grid cell later in VoidFinder,
    a user will need the origin (0,0,0) of the grid, which is the point (min_x, min_y, min_z)
    from the survey, so this function also returns that min value.
    
    
    Parameters
    ==========
    
    galaxy_coords_xyz : numpy.ndarray of shape (N,3)
        coordinates of survey galaxies in xyz space
        
    hole_grid_edge_length : float
        length in xyz space of the edge of 1 cubical cell in the grid
        
        
    Returns
    =======
    
    hole_grid_shape : tuple of ints (i,j,k)
        number of grid cells in each dimension
        
    coords_min : numpy.ndarray of shape (3,)
        the (min_x, min_y, min_z) point which is the (0,0,0) of the grid
    
    """
    
    
    coords_max = np.max(galaxy_coords_xyz, axis=0)
    
    coords_min = np.min(galaxy_coords_xyz, axis=0)
    
    box = coords_max - coords_min

    ngrid = box/hole_grid_edge_length
    
    #print("Ngrid: ", ngrid)
    
    hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
    
    return hole_grid_shape, coords_min
    
    

def wall_field_separation(galaxy_coords_xyz,
                          sep_neighbor=3,
                          verbose=0):
    """
    Description
    ===========
    
    Given a set of galaxy coordinates in xyz space, find all the galaxies whose
    distance to their Nth nearest neighbor is above or below some limit.  Galaxies
    whose Nth nearest neighbor is close (below), will become 'wall' galaxies, and
    galaxies whose Nth nearest neighbor is far (above) will become field/void/isolated
    galaxies.
    
    The distance limit used below is the mean distance to Nth nearest neighbor plus
    1.5 times the standard deviation of the Nth nearest neighbor distance.
    
    
    Parameters
    ==========
    
    galaxy_coords_xyz : numpy.ndarray of shape (N,3)
        coordinates in xyz space of the galaxies
        
    sep_neighbor : int
       Nth neighbor
       
    verbose : int
        whether to print timing output, 0 for off and >= 1 for on
        
        
    Returns
    =======
    
    wall_gals_xyz : ndarray of shape (K, 3)
        xyz coordinate subset of the input corresponding to tightly packed galaxies
        
    field_gals_xyz : ndarray of shape (L, 3)
        xyz coordinate subset of the input corresponding to isolated galaxies
        
    """
    
    
    if verbose > 0:
            
        print('Finding sep',flush=True)
        
    sep_start = time.time()

    dists3 = av_sep_calc(galaxy_coords_xyz, sep_neighbor)
    
    sep_end = time.time()

    if verbose > 0:
        
        print('Time to find sep =',sep_end-sep_start, flush=True)
    
    avsep = np.mean(dists3)
    
    sd = np.std(dists3)
    
    l = avsep + 1.5*sd

    if verbose > 0:
        
        print('Average separation of n=3rd neighbor gal is', avsep, flush=True)
    
        print('The standard deviation is', sd, flush=True)
    
        print('Going to build wall with search value', l, flush=True)

    # l = 5.81637  # s0 = 7.8, gamma = 1.2, void edge = -0.8
    # l = 7.36181  # s0 = 3.5, gamma = 1.4
    # or force l to have a fixed number by setting l = ****

    fw_start = time.time()

    #f_coord_table, w_coord_table = field_gal_cut(coord_in_table, dists3, l)
    
    gal_close_neighbor_index = dists3 < l

    wall_gals_xyz = galaxy_coords_xyz[gal_close_neighbor_index]

    field_gals_xyz = galaxy_coords_xyz[np.logical_not(gal_close_neighbor_index)]
    
    fw_end = time.time()
    
    if verbose > 0:
        
        print('Time to sort field and wall gals =', fw_end-fw_start, flush=True)

    return wall_gals_xyz, field_gals_xyz
    
    






def find_voids(galaxy_coords_xyz,
               dist_limits,
               mask, 
               mask_resolution,
               coords_min,
               hole_grid_shape,
               survey_name,
               max_hole_mask_overlap=0.1,
               hole_grid_edge_length=5.0,
               galaxy_map_grid_edge_length=None,
               hole_center_iter_dist=1.0,
               maximal_spheres_filename="maximal_spheres.txt",
               void_table_filename="voids_table.txt",
               potential_voids_filename="potential_voids_list.txt",
               num_cpus=None,
               save_after=None,
               use_start_checkpoint=False,
               batch_size=10000,
               verbose=1,
               print_after=5.0,
               ):
    """
    Description:
    ============
    
    
    Main entry point for VoidFinder.  
    
    Using the VoidFinder algorithm, this function grows a sphere in each empty 
    grid cell of a grid imposed over the target galaxy distribution.  It then combines 
    these spheres into unique voids, identifying a maximal sphere for each void.

    This algorithm at a high level uses 3 data to find voids
    in the super-galactic structure of the universe:
    
    1).  The galaxy coordinates
    2).  A survey-limiting mask
    3).  A cubic-cell grid of potential void locations
    
    Before running VoidFinder, a preprocessing stage of removing isolated galaxies from
    the target galaxy survey is performed.  Currently this is done by removing galaxies
    whose distance to their 3rd nearest neighbor is greater than 1.5 times the standard
    deviation of 3rd nearest neighbor distance for the survey.  This step should be 
    performed prior to calling this function.
    
    Next, VoidFinder will impose a grid of cubic cells over the remaining non-isolated,
    or "wall" galaxies.  The cell size of this grid should be small enough to allow
    a thorough search, but is also the primary consumer of time in this algorithm.
    
    At each grid cell, VoidFinder will evaluate whether that cubic cell is "empty"
    or "nonempty."  Empty cells contain no galaxies, non-empty cells contain at least 1
    galaxy.  This makes the removal of isolated galaxies in the preprocessing stage
    important.
    
    VoidFinder will proceed to grow a sphere, called a hole, at every Empty grid cell.
    These pre-void holes will be filtered such that the potential voids along the edge of
    the survey will be removed, since any void on the edge of the survey could
    potentially grow unbounded, and there may be galaxies not present which would
    have bounded the void.  After the filtering, these pre-voids will be combined
    into the actual voids based on an analysis of their overlap.
    
    This implementation uses a reference point, 'coords_min', from xyz space, and the 
    'hole_grid_edge_length' to convert between the x,y,z coordinates of a galaxy,
    and the i,j,k coordinates of a cell in the search grid such that:
    
    ijk = ((xyz - coords_min)/hole_grid_edge_length).astype(integer) 
    
    During the sphere growth, VoidFinder also uses a secondary grid to help find
    the bounding galaxies for a sphere.  This secondary grid facilitates nearest-neighbor
    and radius queries, and uses a coordinate space referred to in the code
    as pqr, which uses a similar transformation:
    
    pqr = ((xyz - coords_min)/neighbor_grid_edge_length).astype(integer)
    
    In VoidFinder terminology, a Void is a union of spheres, and a single sphere
    is just a hole.  The Voids are found by taking the set of holes, and ordering them
    based on radius. Starting from the largest found hole, label it a maximal
    sphere, and continue to the next hole.  If the next hole does not overlap with any of the
    previous maximal spheres by some factor, it is also considered a maximal sphere.  This process
    is repeated until there are no more maximal spheres, and all other spheres are joined to the
    maximal spheres.
    
    
    A note on the purpose of VoidFinder - VoidFinder is intended to find distinct, discrete
    void *locations* within the large scale structure of the universe.  This is in contrast
    to finding the large scale void *structure*.  VoidFinder answers the question
    "Where are the voids?" with a concrete "Here is a list of x,y,z coordinates", but it does not
    answer the questions "What do the voids look like?  How are they shaped?  How much do
    they overlap?"  These questions can be partially answered with additional analysis on the 
    output of VoidFinder, but the main algorithm is intended to find discrete, disjoint x-y-z 
    coordinates of the centers of void regions.  If you wanted a local density estimate for a
    given galaxy, you could just use the distance to Nth nearest neighbor, for example.  This
    is not what VoidFinder is for.
    
    
    To do this, VoidFinder makes the following assumptions:
    
    1.  A Void region can be approximated by a sphere.
    
        1.a. the center of the maximal sphere in that void region will yield the x-y-z
        
    2.  Void regions are distinct/discrete - we're not looking for huge tunneling structures
        throughout space, if does happen to be the structure of space (it basically does happen
        to be that way) we want the locations of the biggest rooms
    
    
    
    
    
    
    Parameters
    ==========
    
    galaxy_coords_xyz : numpy.ndarray of shape (num_galaxies, 3)
        coordinates of the galaxies in the survey, units of Mpc/h
        (xyz space)
    
    dist_limits : numpy array of shape (2,)
        minimum and maximum distance limit of the survey in units of Mpc/h
        (xyz space)
        
    mask : numpy.ndarray of shape (N,M) type bool
        represents the survey footprint in scaled ra/dec space.  Value of True 
        indicates that a location is within the survey
        (ra/dec space)

    mask_resolution : integer
        Scale factor of coordinates needed to index mask
        
    coords_min : ndarray of shape (3,) or (1,3)
        coordinates used for converting from xyz space into the grid ijk space
    
    hole_grid_shape : tuple of 3 integers
        ijk dimensions of the 3 grid directions
        
    hole_grid_edge_length : float
        size in Mpc/h of the edge of 1 cube in the search grid, or
        distance between 2 grid cells
        (xyz space)
        
    max_hole_mask_overlap : float in range (0, 0.5)
        when the volume of a hole overlaps the mask by this fraction,
        discard that hole.  Maximum value of 0.5 because a value of 0.5
        means that the hole center will be outside the mask, but more importantly
        because the numpy.roots() function used below won't return a valid
        polynomial root.
        
    survey_name : str
        identifier for the survey running, may be prepended or appended
        to output filenames including the checkpoint filename
        
    galaxy_map_grid_edge_length : float or None
        edge length in Mpc/h for the secondary grid for finding nearest neighbor
        galaxies.  If None, will default to 3*hole_grid_edge_length (which results
        in a cell volume of 3^3 = 27 times larger cube volume).  This parameter
        yields a tradeoff between number of galaxies in a cell, and number of
        cells to search when growing a sphere.  Too large and many redundant galaxies
        may be searched, too small and too many cells will need to be searched.
        (xyz space)
        
    hole_center_iter_dist : float
        distance to move the sphere center each iteration while growing a void
        sphere in units of Mpc/h
        (xyz space)
    
    maximal_spheres_filename : str
        location to save maximal spheres file 
    
    void_table_filename : str
        location to save void table to
    
    potential_voids_filename : str
        location to save potential voids file to
    
    num_cpus : int or None
        number of cpus to use while running the main algorithm.  None will result
        in using number of physical cores on the machine.  Some speedup benefit
        may be obtained from using additional logical cores via Intel Hyperthreading
        but with diminishing returns.  This can safely be set above the number of 
        physical cores without issue if desired.
        
    save_after : int or None
        save a VoidFinderCheckpoint.h5 file after *approximately* every save_after
        cells have been processed.  This will over-write this checkpoint file every
        save_after cells, NOT append to it.  Also, saving the checkpoint file forces
        the worker processes to pause and synchronize with the master process to ensure
        the correct values get written, so choose a good balance between saving
        too often and not often enough if using this parameter.  Note that it is
        an approximate value because it depends on the number of worker processes and
        the provided batch_size value, if you batch size is 10,000 and your save_after
        is 1,000,000 you might actually get a checkpoint at say 1,030,000.
        if None, disables saving the checkpoint file
        
    
    use_start_checkpoint : bool
        Whether to attempt looking for a  VoidFinderCheckpoint.h5 file which can be used to 
        restart the VF run
        if False, VoidFinder will start fresh from 0    
    
        
    batch_size : int
        number of potential void cells to evaluate at a time.  Lower values may be a
        bit slower as it involves some memory allocation overhead, and values which
        are too high may cause the status update printing to take more than print_after
        seconds.  Default value 10,000
        
    verbose : int
        level of verbosity to print during running, 0 indicates off, 1 indicates
        to print after every 'print_after' cells have been processed, and 2 indicates
        to print all debugging statements
        
    print_after : float
        number of seconds to wait before printing a status update
    
    
    Returns
    =======
    
    All output is currently written to disk:
    
    potential voids table, ascii.commented_header format
    
    combined voids table, ascii.commented_header format
    
    maximal spheres table
    
    """
    ############################################################################
    # Calculate the radial value at which we need to check a finished hole
    # for overlap with the mask
    #
    # We're assuming axis-aligned intersection only, so this calculation uses
    # the area of a "spherical cap" https://en.wikipedia.org/wiki/Spherical_cap
    # We set the area of the spherical cap equal to some fraction p times the
    # area of the whole sphere, and if that volume of the sphere is in the mask
    # then we discard the current hole.  Instead of calculating the actual 
    # volume at any point, instead we calculate the percentage of the radius
    # at which that volume is achieved, which is the same for every sphere, then
    # we can check the 6 directions from the center of a hole as a proxy for
    # an actual 10% volume overlap.  This is a good-enough approximation, since
    # the mask is already composed of cubic cells anyway.
    #
    #
    # let l = r - Y, where r is the radius of a sphere, and Y is the distance
    # along the radius at the volume we care about
    # V_cap = pi*l*l*(3r-l)/3 
    # V_sphere = (4/3)*pi*r^3
    # let 'p' be the fraction of the volume we care about (say, 10% or 0.1)
    #
    # pi*(r-Y)*(r-Y)*(3r-r+Y)/3 = p*(4/3)*pi*r^3
    #
    # algebra
    #
    # (Y/r)^3 - 3(Y/r) + (2-4p) = 0
    #
    # Solve for the value (Y/r) given parameter p using the numpy.roots, which
    # solves the roots of polynomials in the form ax^n + bx^(n-1) ... + C = 0
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.roots.html
    # The root we care about will be in the interval (0,1)
    #
    # a = 1
    # b = 0
    # c = -3
    # d = 2 - 4*p
    #
    # DEPRECATED THIS IN FAVOR OF A NEW STRATEGY
    # 
    ############################################################################
    '''
    coeffs = [1.0, 0.0, -3.0, 2.0 - 4.0*max_hole_mask_overlap]
    
    roots = np.roots(coeffs)
    
    radial_mask_check = None
    
    for root in roots:
        
        if root > 0.0 and root < 1.0:
            
            radial_mask_check = root
            
    if radial_mask_check is None:
        
        raise ValueError("Could not calculate appropriate radial check value for input max_hole_mask_overlap "+str(max_hole_mask_overlap))
    
    
    print("For mask volume check of: ", max_hole_mask_overlap, "Using radial hole value of: ", radial_mask_check)
    '''
    ############################################################################
    #
    #   GROW HOLES
    #
    ############################################################################

    if galaxy_map_grid_edge_length is None:
        
        galaxy_map_grid_edge_length = 3.0*hole_grid_edge_length

    tot_hole_start = time.time()

    print('Growing holes', flush=True)

    x_y_z_r_array, n_holes = _hole_finder(hole_grid_shape, 
                                           hole_grid_edge_length, 
                                           hole_center_iter_dist,
                                           galaxy_map_grid_edge_length,
                                           coords_min.reshape(1,3),
                                           mask,
                                           mask_resolution,
                                           dist_limits[0],
                                           dist_limits[1],
                                           galaxy_coords_xyz,
                                           survey_name,
                                           #radial_mask_check,
                                           save_after=save_after,
                                           use_start_checkpoint=use_start_checkpoint,
                                           batch_size=batch_size,
                                           verbose=verbose,
                                           print_after=print_after,
                                           num_cpus=num_cpus)


    print('Found a total of', n_holes, 'potential voids.', flush=True)

    print('Time to find all holes =', time.time() - tot_hole_start, flush=True)
    
    ############################################################################
    # 
    #   CHECK IF 90% OF VOID VOLUME IS WITHIN SURVEY LIMITS
    #
    ############################################################################
    # Cythonized version
    #---------------------------------------------------------------------------
    vol_cut_start = time.time()
    
    print("Starting volume cut", flush=True)
    
    hole_bound_time = time.time()
    
    valid_idx, monte_index = check_hole_bounds(x_y_z_r_array, 
                                              mask.astype(np.uint8), 
                                              mask_resolution, 
                                              dist_limits,
                                              cut_pct=0.1,
                                              pts_per_unit_volume=.01,
                                              num_surf_pts=20,
                                              num_cpus=num_cpus)
    
    print("Found ", np.sum(np.logical_not(valid_idx)), " holes to cut", 
          time.time() - vol_cut_start, flush=True)

    x_y_z_r_array = x_y_z_r_array[valid_idx]
    #---------------------------------------------------------------------------
    '''
    # Pure python version
    #---------------------------------------------------------------------------
    coordinates = np.empty((1,3), dtype=np.float64)
    temp_coordinates = np.empty((1,3), dtype=np.float64)
    mask = mask.astype(np.uint8)
    
    keep_holes = np.ones(x_y_z_r_array.shape[0], dtype=np.bool)
    
    for idx in range(x_y_z_r_array.shape[0]):
    
        hole_radius = x_y_z_r_array[idx,3]
        
        coordinates[0,0] = x_y_z_r_array[idx,0]
        coordinates[0,1] = x_y_z_r_array[idx,1]
        coordinates[0,2] = x_y_z_r_array[idx,2]
    
        discard = check_mask_overlap(coordinates,
                           temp_coordinates,
                           radial_mask_check,
                           hole_radius,
                           mask, 
                           mask_resolution,
                           dist_limits[0],
                           dist_limits[1])
        
        if discard:
            
            keep_holes[idx] = False
    
    x_y_z_r_array = x_y_z_r_array[keep_holes]
    
    
    print("After volume cut, remaining holes: ", x_y_z_r_array.shape[0])
    #---------------------------------------------------------------------------
    '''
    ############################################################################
    #
    #   SORT HOLES BY SIZE
    #
    ############################################################################
    print("Sorting holes based on radius", flush=True)
    
    sort_order = x_y_z_r_array[:,3].argsort()[::-1]
    
    x_y_z_r_array = x_y_z_r_array[sort_order]
    '''
    sort_start = time.time()
    
    print('Sorting holes by size', flush=True)
    
    potential_voids_table = Table(x_y_z_r_array, names=('x','y','z','radius'))

    potential_voids_table.sort('radius')
    
    potential_voids_table.reverse()
    
    sort_end = time.time()

    print('Holes are sorted; Time to sort holes =', sort_end-sort_start, flush=True)
    '''

    ############################################################################
    #
    #   CHECK IF 90% OF VOID VOLUME IS WITHIN SURVEY LIMITS
    #
    ############################################################################
    '''
    print('Removing holes with at least 10% of their volume outside the mask', flush=True)

    mask = mask.astype(np.uint8)


    cut_start = time.time()
    potential_voids_table = volume_cut(potential_voids_table, 
                                       mask, 
                                       mask_resolution, 
                                       dist_limits)

    print("Time to volume-cut holes: ", time.time() - cut_start, flush=True)
    
    print("Num volume-cut holes: ", len(potential_voids_table), flush=True)
    
    potential_voids_table.write(potential_voids_filename, format='ascii.commented_header', overwrite=True)
    '''
    ############################################################################
    #
    #   FILTER AND SORT HOLES INTO UNIQUE VOIDS
    #
    ############################################################################
    print("Combining holes into unique voids", flush=True)
    
    combine_start = time.time()
    
    maximal_spheres_table, myvoids_table = combine_holes_2(x_y_z_r_array)
    
    print("Combine time:", time.time() - combine_start, flush=True)
    
    print('Number of unique voids is', len(maximal_spheres_table), flush=True)
    '''
    combine_start = time.time()

    print('Combining holes into unique voids', flush=True)

    maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table)
    
    print('Number of unique voids is', len(maximal_spheres_table), flush=True)
    
    combine_end = time.time()

    print('Time to combine holes into voids =', combine_end-combine_start, flush=True)
    '''
    
    ############################################################################
    #
    # Save list of all void holes
    #
    ############################################################################
    myvoids_table.write(void_table_filename, format='ascii.commented_header', overwrite=True)

    

    
    ############################################################################
    #
    #   COMPUTE VOLUME OF EACH VOID
    #
    ############################################################################
    '''
    print('Compute void volumes')

    # Initialize void volume array
    void_vol = np.zeros(void_count)

    nran = 10000

    for i in range(void_count):
        nsph = 0
        rad = 4*myvoids_table['radius'][v_index[i]]

        for j in range(nran):
            rand_pos = add_row(np.random.rand(3)*rad, myvoids_table['x','y','z'][v_index[i]]) - 0.5*rad
            
            for k in range(len(myvoids_table)):
                if myvoids_table['flag'][k]:
                    # Calculate difference between particle and sphere
                    sep = sum(to_vector(subtract_row(rand_pos, myvoids_table['x','y','z'][k])))
                    
                    if sep < myvoids_table['radius'][k]**2:
                        # This particle lies in at least one sphere
                        nsph += 1
                        break
        
        void_vol[i] = (rad**3)*nsph/nran
    
    '''
    ############################################################################
    #
    #   IDENTIFY VOID GALAXIES
    #
    ############################################################################
    '''
    print('Assign field galaxies to voids')

    # Count the number of galaxies in each void
    nfield = np.zeros(void_count)

    # Add void field to f_coord
    f_coord['vID'] = -99

    for i in range(nf): # Go through each void galaxy
        for j in range(len(myvoids_table)): # Go through each void
            if np.linalg.norm(to_vector(subtract_row(f_coord[i], myvoids_table['x','y','z'][j]))) < myvoids_table['radius'][j]:
                # Galaxy lives in the void
                nfield[myvoids_table['flag'][j]] += 1

                # Set void ID in f_coord to match void ID
                f_coord['vID'][i] = myvoids_table['flag'][j]

                break

    f_coord.write(voidgals_filename, format='ascii.commented_header')
    '''

    ############################################################################
    #
    #   MAXIMAL HOLE FOR EACH VOID
    #
    ############################################################################

    save_maximals(maximal_spheres_table, maximal_spheres_filename)

    
    ############################################################################
    #
    #   VOID REGION SIZES
    #
    ############################################################################
    '''

    # Initialize
    void_regions = Table()

    void_regions['radius'] = myvoids_table['radius'][v_index]
    void_regions['effective_radius'] = (void_vol*0.75/np.pi)**(1./3.)
    void_regions['volume'] = void_vol
    void_regions['x'] = myvoids_table['x'][v_index]
    void_regions['y'] = myvoids_table['y'][v_index]
    void_regions['z'] = myvoids_table['z'][v_index]
    void_regions['deltap'] = (nfield - N_gal*void_vol/vol)/(N_gal*void_vol/vol)
    void_regions['n_gal'] = nfield
    void_regions['vol_maxHole'] = (4./3.)*np.pi*myvoids_table['radius'][v_index]**3/void_vol

    void_regions.write(out3_filename, format='ascii.commented_header')
    '''

