#VoidFinder Function to do just about everything


import numpy as np

from astropy.table import Table

import time

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

from ._voidfinder_cython_find_next import MaskChecker


################################################################################
DtoR = np.pi/180.
RtoD = 180./np.pi
################################################################################



def filter_galaxies(galaxy_table,
                    survey_name, 
                    out_directory,
                    mag_cut=True, 
                    dist_limits=None,
                    rm_isolated=True,
                    write_table=True,
                    sep_neighbor=3,
                    dist_metric='comoving', 
                    h=1.0,
                    magnitude_limit=-20.09,
                    verbose=0):
    """
    A hodge podge of miscellaneous tasks which need to be done to format the data into
    something the main find_voids() function can use.
    
    1) Optional magnitude cut
    2) Convert from ra-dec-redshift space into xyz space
    3) Calculate the hole search grid shape
    4) Optional remove isolated galaxies by partitioning them into wall (non-isolated)
       and field (isolated) groups
    5) Optionally write out the wall and field galaxies to disk
    
    
    Parameters
    ==========
    
    galaxy_table : astropy.table of shape (N,?)
        variable number of required columns.  If doing magnitude cut, must include
        'rabsmag' column. If distance metric is 'comoving', must include 'Rgal'
        column, otherwise must include 'redshift'.  Also must always include 'ra' 
        and 'dec'
        
    survey_name : str
        Name of the galxy catalog, string value to prepend or append to output names

    out_directory : string
        Directory path for output files
        
    mag_cut : bool
        whether or not to cut on magnitude, removing galaxies less than
        magnitude_limit

    dist_limits : list of length 2
        [Minimum distance, maximum distance] of galaxy sample (in units of Mpc/h)
        
    magnitude_limit : float
        value at which to perform magnitude cut
        
    rm_isolated : bool
        whether or not to perform Nth neighbor distance calculation, and use it
        to partition the input galaxies into wall and field galaxies
        
    write_table : bool
        use astropy.table.Table.write to write out the wall and field galaxies
        to file
        
    sep_neighbor : int, positive
        if rm_isolated_flag is true, find the Nth galaxy neighbors based on this value
        
    dist_metric : str
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).
        
        
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
    
    
    print('Filter Galaxies Start', flush=True)

    ############################################################################
    # PRE-PROCESS DATA
    # Filter based on magnitude and convert from redshift to radius if necessary
    #---------------------------------------------------------------------------
    # Remove faint galaxies
    if mag_cut:
        
        galaxy_table = galaxy_table[galaxy_table['rabsmag'] <= magnitude_limit]

    # Remove galaxies outside redshift range
    if dist_limits is not None:

        if dist_metric == 'comoving':

            distance_boolean = np.logical_and(galaxy_table['Rgal'] >= dist_limits[0], 
                                              galaxy_table['Rgal'] <= dist_limits[1])
        else:

            H0 = 100*h

            distance_boolean = np.logical_and(c*galaxy_table['redshift']/H0 >= dist_limits[0],
                                              c*galaxy_table['redshift']/H0 <= dist_limits[1])

        galaxy_table = galaxy_table[distance_boolean]


    # Convert galaxy coordinates to Cartesian
    coords_xyz = ra_dec_to_xyz(galaxy_table, dist_metric, h)
    ############################################################################

    
    
    


    
    ############################################################################
    # Separation
    #---------------------------------------------------------------------------
    if rm_isolated:
        
        wall_gals_xyz, field_gals_xyz = wall_field_separation(coords_xyz,
                                                              sep_neighbor=sep_neighbor,
                                                              verbose=verbose)

    else:
        
        wall_gals_xyz = coords_xyz
        
        field_gals_xyz = np.array([])
    ############################################################################



    ############################################################################
    # Write results to disk if desired
    #---------------------------------------------------------------------------
    if write_table:
    
        write_start = time.time()
        
    
        wall_xyz_table = Table(data=wall_gals_xyz, names=["x", "y", "z"])
        
        wall_xyz_table.write(out_directory + survey_name + 'wall_gal_file.txt', 
                             format='ascii.commented_header', 
                             overwrite=True)
    
        
        field_xyz_table = Table(data=field_gals_xyz, names=["x", "y", "z"])
    
        field_xyz_table.write(out_directory + survey_name + 'field_gal_file.txt', 
                              format='ascii.commented_header', 
                              overwrite=True)
        
        
        if verbose > 0:
            
            print("Time to write field and wall tables:", 
                  time.time() - write_start, 
                  flush=True)


    nf =  len(field_gals_xyz)
    
    nwall = len(wall_gals_xyz)
    
    print('Number of field gals:', nf, 
          'Number of wall gals:', nwall, 
          flush=True)
    ############################################################################


    return wall_gals_xyz, field_gals_xyz












def ra_dec_to_xyz(galaxy_table,
                  distance_metric='comoving',
                  h=1.0,
                  ):
    """
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
        
        r_gal = c*galaxy_table['redshift'].data/(100*h)
        
        
    ra = galaxy_table['ra'].data
    
    dec = galaxy_table['dec'].data
    
    ############################################################################
    # Convert from ra-dec-radius space to xyz space
    #---------------------------------------------------------------------------
    ra_radian = ra*DtoR
    
    dec_radian = dec*DtoR
    
    x = r_gal*np.cos(ra_radian)*np.cos(dec_radian)
    
    y = r_gal*np.sin(ra_radian)*np.cos(dec_radian)
    
    z = r_gal*np.sin(dec_radian)
    
    num_gal = x.shape[0]
    
    coords_xyz = np.concatenate((x.reshape(num_gal,1),
                                 y.reshape(num_gal,1),
                                 z.reshape(num_gal,1)), axis=1)
    ############################################################################
    
    return coords_xyz







def calculate_grid(galaxy_coords_xyz,
                   hole_grid_edge_length):
    """
    Given a galaxy survey in xyz/Cartesian coordinates and the length of a 
    cubical grid cell, calculate the cubical grid shape which will contain the 
    survey.
    
    The cubical grid is obtained by finding the minimum and maximum values in 
    each of the 3 dimensions, calculating the distance of the survey in each 
    dimension, and dividing each dimension by the cell grid length to get the 
    number of required grid cells in each dimension.
        
    In order to transform additional points to their closest grid cell later in 
    VoidFinder, a user will need the origin (0,0,0) of the grid, which is the 
    point (min_x, min_y, min_z) from the survey, so this function also returns 
    that min value.
    
    
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

    coords_max : numpy.ndarray of shape (3,)
        the (max_x, max_y, max_z) point of the galaxies
    """
    
    
    coords_max = np.max(galaxy_coords_xyz, axis=0)
    
    coords_min = np.min(galaxy_coords_xyz, axis=0)
    
    box = coords_max - coords_min

    ngrid = box/hole_grid_edge_length
    
    #print("Ngrid: ", ngrid)
    
    hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
    
    return hole_grid_shape, coords_min, coords_max
    
    



    

def wall_field_separation(galaxy_coords_xyz,
                          sep_neighbor=3,
                          verbose=0):
    """
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
    
    
    ############################################################################
    # Calculate the average distance to the 3rd nearest neighbor
    #---------------------------------------------------------------------------
    print('Finding isolated galaxy distance',flush=True)
        
    sep_start = time.time()

    dists3 = av_sep_calc(galaxy_coords_xyz, sep_neighbor)
    
    sep_end = time.time()

    print('Time to find isolated galaxy distance:', sep_end-sep_start, flush=True)
    ############################################################################
    


    ############################################################################
    # Calculate the isolated galaxy criterion
    #---------------------------------------------------------------------------
    avsep = np.mean(dists3)
    
    sd = np.std(dists3)
    
    l = avsep + 1.5*sd

    if verbose > 0:
        
        print('Average separation of 3rd neighbor gal is', avsep, flush=True)
    
        print('The standard deviation is', sd, flush=True)
    
    print('Removing all galaxies with 3rd nearest neighbors further than', l, 
          flush=True)
    ############################################################################



    ############################################################################
    # Separate galaxies into field and wall
    #---------------------------------------------------------------------------
    fw_start = time.time()

    #f_coord_table, w_coord_table = field_gal_cut(coord_in_table, dists3, l)
    
    gal_close_neighbor_index = dists3 < l

    wall_gals_xyz = galaxy_coords_xyz[gal_close_neighbor_index]

    field_gals_xyz = galaxy_coords_xyz[np.logical_not(gal_close_neighbor_index)]
    
    fw_end = time.time()
    
    if verbose > 0:
        
        print('Time to sort field and wall gals:', fw_end-fw_start, flush=True)
    ############################################################################

    return wall_gals_xyz, field_gals_xyz
    
    






def find_voids(galaxy_coords_xyz,
               survey_name,
               mask_type='ra_dec_z',
               mask=None, 
               mask_resolution=None,
               dist_limits=None,
               xyz_limits=None,
               check_only_empty_cells=True,
               max_hole_mask_overlap=0.1,
               hole_grid_edge_length=5.0,
               min_maximal_radius=10.0,
               galaxy_map_grid_edge_length=None,
               hole_center_iter_dist=1.0,
               maximal_spheres_filename="maximal_spheres.txt",
               void_table_filename="voids_table.txt",
               potential_voids_filename="potential_voids_list.txt",
               num_cpus=None,
               save_after=None,
               use_start_checkpoint=False,
               batch_size=10000,
               verbose=0,
               print_after=5.0
               ):
    """
    Main entry point for VoidFinder.  
    
    Using the VoidFinder algorithm, this function grows a sphere in each empty 
    grid cell of a grid imposed over the target galaxy distribution.  It then 
    combines these spheres into unique voids, identifying a maximal sphere for 
    each void.

    This algorithm at a high level uses 3 data to find voids in the large-scale 
    structure of the universe:
    
    1)  The galaxy coordinates
    2)  A survey-limiting mask
    3)  A cubic-cell grid of potential void locations
    
    Before running VoidFinder, a preprocessing stage of removing isolated 
    galaxies from the target galaxy survey is performed.  Currently this is done 
    by removing galaxies whose distance to their 3rd nearest neighbor is greater 
    than 1.5 times the standard deviation of 3rd nearest neighbor distance for 
    the survey.  This step should be performed prior to calling this function.
    
    Next, VoidFinder will impose a grid of cubic cells over the remaining 
    non-isolated, or "wall" galaxies.  The cell size of this grid should be 
    small enough to allow a thorough search, but is also the primary consumer of 
    time in this algorithm.
    
    At each grid cell, VoidFinder will evaluate whether that cubic cell is 
    "empty" or "nonempty."  Empty cells contain no galaxies, non-empty cells 
    contain at least 1 galaxy.  This makes the removal of isolated galaxies in 
    the preprocessing stage important.
    
    VoidFinder will proceed to grow a sphere, called a hole, at every Empty grid 
    cell.  These pre-void holes will be filtered such that the potential voids 
    along the edge of the survey will be removed, since any void on the edge of 
    the survey could potentially grow unbounded, and there may be galaxies not 
    present which would have bounded the void.  After the filtering, these 
    pre-voids will be combined into the actual voids based on an analysis of 
    their overlap.
    
    This implementation uses a reference point, 'coords_min', from xyz space, 
    and the 'hole_grid_edge_length' to convert between the x,y,z coordinates of 
    a galaxy, and the i,j,k coordinates of a cell in the search grid such that:
    
    ijk = ((xyz - coords_min)/hole_grid_edge_length).astype(integer) 
    
    During the sphere growth, VoidFinder also uses a secondary grid to help find 
    the bounding galaxies for a sphere.  This secondary grid facilitates 
    nearest-neighbor and radius queries, and uses a coordinate space referred to 
    in the code as pqr, which uses a similar transformation:
    
    pqr = ((xyz - coords_min)/neighbor_grid_edge_length).astype(integer)
    
    In VoidFinder terminology, a Void is a union of spheres, and a single sphere 
    is just a hole.  The Voids are found by taking the set of holes, and 
    ordering them based on radius. Starting from the largest found hole, label 
    it a maximal sphere, and continue to the next hole.  If the next hole does 
    not overlap with any of the previous maximal spheres by some factor, it is 
    also considered a maximal sphere.  This process is repeated until there are 
    no more maximal spheres, and all other spheres are joined to the maximal 
    spheres.
    
    A note on the purpose of VoidFinder - VoidFinder is intended to find 
    distinct, discrete void *locations* within the large scale structure of the 
    universe.  This is in contrast to finding the large scale void *structure*.  
    VoidFinder answers the question "Where are the voids?" with a concrete "Here 
    is a list of x,y,z coordinates", but it does not answer the questions "What 
    do the voids look like?  How are they shaped?  How much do they overlap?"  
    These questions can be partially answered with additional analysis on the 
    output of VoidFinder, but the main algorithm is intended to find discrete, 
    disjoint x-y-z coordinates of the centers of void regions.  If you wanted a 
    local density estimate for a given galaxy, you could just use the distance 
    to Nth nearest neighbor, for example.  This is not what VoidFinder is for.
    
    To do this, VoidFinder makes the following assumptions:
    
    1.  A Void region can be approximated by a union of spheres.  Note: the 
        center of the maximal sphere in that void region will yield the x-y-z 
        coordinate of that void region.
        
    2.  Void regions are distinct/discrete - we are not looking for huge 
        tunneling structures throughout space, if does happen to be the 
        structure of space (it basically does happen to be that way) we want the 
        locations of the biggest rooms
    
    
    Parameters
    ==========
    
    galaxy_coords_xyz : numpy.ndarray of shape (num_galaxies, 3)
        coordinates of the galaxies in the survey, units of Mpc/h 
        (xyz space)
        
    survey_name : str
        identifier for the survey running, may be prepended or appended to 
        output filenames including the checkpoint filename
        
    mask_type : string, one of ['ra_dec_z', 'xyz', 'periodic']
        Determines the mode of mask checking to use and which mask parameters to use.  
        
        'ra_dec_z' means the mask, mask_resolution, and dist_limits parameters
            must be provided.  The 'mask' represents an angular space in Right Ascension and 
            Declination, the corresponding mask_resolution integer represents the scale needed
            to index into the Right Ascension and Declination of the mask, and the dist_limits
            represent the min and max redshift values (as radial distances in xyz space).
        
        'xyz' means that the xyz_limits parameter must be provided which directly encodes
            a bounding box for the survey in xyz space
            
        'periodic' means that the xyz_limits parameter must be provided, which directly
            encodes a bounding box representing the periodic boundary of the survey, and
            the survey will be treated as if its bounding box were tiled to infinity in
            all directions.  Spheres will still only be grown starting from within the
            original bounding box.
        
    mask : numpy.ndarray of shape (N,M) type bool
        represents the survey footprint in scaled ra/dec space.  Value of True 
        indicates that a location is within the survey
        (ra/dec space)

    mask_resolution : integer
        Scale factor of coordinates needed to index mask
        
    dist_limits : numpy array of shape (2,)
        [min_dist, max_dist] in units of Mpc/h (xyz space)
        
    xyz_limits : numpy array of shape (2,3)
        format [x_min, y_min, z_min]
               [x_max, y_max, z_max]
        to be used for checking against the mask when mask_type == 'xyz' or for
        periodic conditions when mask_type == 'periodic'
        
    hole_grid_edge_length : float
        size in Mpc/h of the edge of 1 cube in the search grid, or distance 
        between 2 grid cells
        (xyz space)

    min_maximal_radius : float
        the minimum radius in units of distance for a hole to be considered
        for maximal status.  Default value is 10 Mpc/h.
        
    max_hole_mask_overlap : float in range (0, 0.5)
        when the volume of a hole overlaps the mask by this fraction, discard 
        that hole.  Maximum value of 0.5 because a value of 0.5 means that the 
        hole center will be outside the mask, but more importantly because the 
        numpy.roots() function used below won't return a valid polynomial root.
        
    galaxy_map_grid_edge_length : float or None
        edge length in Mpc/h for the secondary grid for finding nearest neighbor 
        galaxies.  If None, will default to 3*hole_grid_edge_length (which 
        results in a cell volume of 3^3 = 27 times larger cube volume).  This 
        parameter yields a tradeoff between number of galaxies in a cell, and 
        number of cells to search when growing a sphere.  Too large and many 
        redundant galaxies may be searched, too small and too many cells will 
        need to be searched.
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
        number of cpus to use while running the main algorithm.  None will 
        result in using number of physical cores on the machine.  Some speedup 
        benefit may be obtained from using additional logical cores via Intel 
        Hyperthreading but with diminishing returns.  This can safely be set 
        above the number of physical cores without issue if desired.
        
    save_after : int or None
        save a VoidFinderCheckpoint.h5 file after *approximately* every 
        save_after cells have been processed.  This will over-write this 
        checkpoint file every save_after cells, NOT append to it.  Also, saving 
        the checkpoint file forces the worker processes to pause and synchronize 
        with the master process to ensure the correct values get written, so 
        choose a good balance between saving too often and not often enough if 
        using this parameter.  Note that it is an approximate value because it 
        depends on the number of worker processes and the provided batch_size 
        value, if your batch size is 10,000 and your save_after is 1,000,000 you 
        might actually get a checkpoint at say 1,030,000.  If None, disables 
        saving the checkpoint file.

    check_only_empty_cells : bool
        whether or not to start growing a hole in a cell which has galaxies in
        it, aka "non-empty".  If True (default), don't grow holes in these cells.
    
    use_start_checkpoint : bool
        Whether to attempt looking for a VoidFinderCheckpoint.h5 file which can 
        be used to restart the VF run.  If False, VoidFinder will start fresh 
        from 0.
    
    batch_size : int
        number of potential void cells to evaluate at a time.  Lower values may 
        be a bit slower as it involves some memory allocation overhead, and 
        values which are too high may cause the status update printing to take 
        more than print_after seconds.  Default value 10,000
        
    verbose : int
        level of verbosity to print during running, 0 indicates off, 1 indicates 
        to print after every 'print_after' cells have been processed, and 2 
        indicates to print all debugging statements
        
    print_after : float
        number of seconds to wait before printing a status update
    
    
    Returns
    =======
    
    All output is currently written to disk:
    
    potential voids table, ascii.commented_header format
    
    combined voids table, ascii.commented_header format
    
    maximal spheres table
    """
    
    if mask_type == "ra_dec_z":
        mask_mode = 0
    elif mask_type == "xyz":
        mask_mode = 1
    elif mask_type == "periodic":
        mask_mode = 2
    else:
        raise ValueError("mask_type must be 'ra_dec_z', 'xyz', or 'periodic'")
    
    
    if dist_limits is None:
        min_dist = None
        max_dist = None
    else:
        min_dist = dist_limits[0]
        max_dist = dist_limits[1]
    
    
    
    print('Growing holes', flush=True)
        
    ############################################################################
    # GROW HOLES
    #---------------------------------------------------------------------------
    

    tot_hole_start = time.time()

    x_y_z_r_array, n_holes = _hole_finder(galaxy_coords_xyz,
                                          hole_grid_edge_length, 
                                          hole_center_iter_dist,
                                          galaxy_map_grid_edge_length,
                                          survey_name,
                                          mask_mode=mask_mode,
                                          mask=mask,
                                          mask_resolution=mask_resolution,
                                          min_dist=min_dist,
                                          max_dist=max_dist,
                                          xyz_limits=xyz_limits,
                                          check_only_empty_cells=check_only_empty_cells,
                                          save_after=save_after,
                                          use_start_checkpoint=use_start_checkpoint,
                                          batch_size=batch_size,
                                          verbose=verbose,
                                          print_after=print_after,
                                          num_cpus=num_cpus)


    print('Found a total of', n_holes, 'potential voids.', flush=True)

    print('Time to grow all holes:', time.time() - tot_hole_start, flush=True)
    ############################################################################




    if mask_mode == 0:
        mask_checker = MaskChecker(mask_mode,
                                   survey_mask_ra_dec=mask.astype(np.uint8),
                                   n=mask_resolution,
                                   rmin=min_dist,
                                   rmax=max_dist,
                                   )
        
    elif mask_mode in [1,2]:
        mask_checker = MaskChecker(mask_mode,
                                   xyz_limits=xyz_limits)





    
    ############################################################################
    # CHECK IF 90% OF VOID VOLUME IS WITHIN SURVEY LIMITS
    #---------------------------------------------------------------------------
    print("Starting volume cut", flush=True)
    
    vol_cut_start = time.time()
    
    valid_idx, monte_index = check_hole_bounds(x_y_z_r_array, 
                                               mask_checker,
                                               cut_pct=0.1,
                                               pts_per_unit_volume=.01,
                                               num_surf_pts=20,
                                               num_cpus=num_cpus,
                                               verbose=verbose)
    
    print("Found ", np.sum(np.logical_not(valid_idx)), "holes to cut", 
          time.time() - vol_cut_start, flush=True)

    x_y_z_r_array = x_y_z_r_array[valid_idx]
    ############################################################################


    
    ############################################################################
    # SORT HOLES BY SIZE
    #---------------------------------------------------------------------------
    print("Sorting holes based on radius", flush=True)
    
    sort_order = x_y_z_r_array[:,3].argsort()[::-1]
    
    x_y_z_r_array = x_y_z_r_array[sort_order]
    ############################################################################

    

    ############################################################################
    # FILTER AND SORT HOLES INTO UNIQUE VOIDS
    #---------------------------------------------------------------------------
    print("Combining holes into unique voids", flush=True)
    
    combine_start = time.time()
    
    maximal_spheres_table, myvoids_table = combine_holes_2(x_y_z_r_array, 
                                                           min_maximal_radius=min_maximal_radius)
    
    print("Combine time:", time.time() - combine_start, flush=True)
    
    print('Number of unique voids is', len(maximal_spheres_table), flush=True)
    ############################################################################

    
    
    ############################################################################
    # Save list of all void holes
    #---------------------------------------------------------------------------
    myvoids_table.write(void_table_filename, 
                        format='ascii.commented_header', 
                        overwrite=True)
    ############################################################################



    ############################################################################
    # Compute volume of each void
    #---------------------------------------------------------------------------
    ############################################################################


    
    ############################################################################
    # Identify void galaxies
    #---------------------------------------------------------------------------
    ############################################################################


    
    ############################################################################
    # Save list of maximal hole in each void
    #---------------------------------------------------------------------------
    save_maximals(maximal_spheres_table, maximal_spheres_filename)
    ############################################################################



    ############################################################################
    # Void region size
    #---------------------------------------------------------------------------
    ############################################################################
    
    
    return maximal_spheres_table, myvoids_table




