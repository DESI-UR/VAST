#VoidFinder Function to do just about everything


import numpy as np

from astropy.table import Table

import time

from .hole_combine import combine_holes

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

from .volume_cut import volume_cut

from .avsepcalc import av_sep_calc

from .mag_cutoff_function import mag_cut, field_gal_cut

from ._voidfinder import _hole_finder

from .constants import c


######################################################################
# I made these constants into default parameters in filter_galaxies()
# and find_voids() because a user may wish to try different
# grid spacings and such.  I left DtoR and RtoD because they will
# never ever change since they're based on pi.
# dec_offset was unused in the code
######################################################################
#dec_offset = -90
#dl = 5           # Cell side length [Mpc/h]
#dr = 1.          # Distance to shift the hole centers
#c = 3e5
DtoR = np.pi/180.
RtoD = 180./np.pi


def filter_galaxies(galaxy_table, 
                    #mask_array, 
                    #mask_resolution, 
                    survey_name, 
                    mag_cut_flag=True, 
                    rm_isolated_flag=True, 
                    distance_metric='comoving', 
                    h=1.0,
                    search_grid_edge_length=5,
                    ):
    '''
    Filter the input galaxy catalog, removing galaxies fainter than some limit 
    and removing isolated galaxies.


    Parameters:
    ===========

    galaxy_table : astropy table
        List of galaxies and their coordinates (ra, dec, redshift) and magnitudes

    (REFACTORED OUT) mask_array : numpy array of shape (2,n)
        n pairs of RA,dec coordinates that are within the survey limits and are 
        scaled by the mask_resolution.  Oth row is RA; 1st row is dec.

    (REFACTORED OUT) mask_resolution : integer
        Scale factor of coordinates in mask_array

    survey_name : string
        Name of galaxy catalog

    mag_cut_flag : boolean
        Determines whether or not to remove galaxies fainter than Mr = -20.  True 
        (default) will remove the faint galaxies.

    rm_isolated_flag : boolean
        Determines whether or not to remove isolated galaxies.  True (default) 
        will remove the isolated galaxies.

    distance_metric : string
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).
    
    search_grid_edge_length : float
        length in Mpc/h of the edge of one cell of a grid cube
        


    Returns:
    ========

    coord_min_table : astropy table
        Minimum values of the galaxy coordinates in x, y, and z.

    mask : numpy array of shape (n,m)
        Index array of the coordinates that are within the survey footprint

    ngrid[0] : numpy array of shape (3,)
        Number of cells in each cartesian direction.

    '''
    
    ################################################################################
    #
    #   PRE-PROCESS DATA
    #
    ################################################################################
    print('Filter Galaxies Start', flush=True)

    # Remove faint galaxies
    if mag_cut_flag:
        galaxy_table = mag_cut(galaxy_table,-20)

    # Convert galaxy coordinates to Cartesian
    if distance_metric == 'comoving':
        r_gal = galaxy_table['Rgal']
    else:
        r_gal = c*galaxy_table['redshift']/(100*h)
        
    xin = r_gal*np.cos(galaxy_table['ra']*DtoR)*np.cos(galaxy_table['dec']*DtoR)
    yin = r_gal*np.sin(galaxy_table['ra']*DtoR)*np.cos(galaxy_table['dec']*DtoR)
    zin = r_gal*np.sin(galaxy_table['dec']*DtoR)
    coord_in_table = Table([xin, yin, zin], names=('x','y','z'))

    # Cartesian coordinate minima
    coord_min_x = [min(coord_in_table['x'])]
    coord_min_y = [min(coord_in_table['y'])]
    coord_min_z = [min(coord_in_table['z'])]
    coord_min_table = Table([coord_min_x, coord_min_y, coord_min_z], names=('x','y','z'))

    # Cartesian coordinate maxima
    coord_max_x = [max(coord_in_table['x'])]
    coord_max_y = [max(coord_in_table['y'])]
    coord_max_z = [max(coord_in_table['z'])]
    coord_max_table = Table([coord_max_x, coord_max_y, coord_max_z], names=('x','y','z'))

    # Number of galaxies
    N_gal = len(galaxy_table)

    print('x:', coord_min_table['x'][0], coord_max_table['x'][0], flush=True)
    print('y:', coord_min_table['y'][0], coord_max_table['y'][0], flush=True)
    print('z:', coord_min_table['z'][0], coord_max_table['z'][0], flush=True)
    print('There are', N_gal, 'galaxies in this catalog.', flush=True)

    # Convert coord_in, coord_min, coord_max tables to numpy arrays
    coord_in = to_array(coord_in_table)
    coord_min = to_vector(coord_min_table)
    coord_max = to_vector(coord_max_table)


    #print('Reading mask',flush=True)

    #mask = build_mask(mask_array, mask_resolution)

    #print('Read mask',flush=True)

    ################################################################################
    #
    #   PUT THE GALAXIES ON A CHAIN MESH
    #
    ################################################################################


    #dl = box/ngrid # length of each side of the box
    #print('Number of grid cells is', ngrid, dl, box)

    #print('Making the grid')

    #print('coord_min shape:', coord_min.shape)
    #print('coord_max shape:', coord_max.shape)

    # Array of size of survey in x, y, z directions [Mpc/h]
    #box = np.array([coord_max_x[0] - coord_min_x[0], coord_max_y[0] - coord_min_y[0], coord_max_z[0] - coord_min_z[0]])
    box = coord_max - coord_min
    
    print("Box: ", box)
    print("search_grid_edge_length", search_grid_edge_length)

    #print('box shape:', box.shape)

    # Array of number of cells in each direction
    ngrid = box/search_grid_edge_length
    
    print("Ngrid: ", ngrid)
    
    ngrid = np.ceil(ngrid).astype(int)
    
    

    #print('ngrid shape:', ngrid.shape)

    print('Number of grid cells is', ngrid, 'with side lengths of', search_grid_edge_length, 'Mpc/h', flush=True)

    '''
    # Bin the galaxies onto a 3D grid
    #mesh_indices, ngal, chainlist, linklist = mesh_galaxies(coord_in_table, coord_min_table, dl, ngrid)
    #ngal, chainlist, linklist = mesh_galaxies(coord_in_table, coord_min_table, dl, tuple(ngrid))

    #print('Made the grid')

  
    print('Checking the grid')
    grid_good = True

    for i in range(ngrid[0]):
        for j in range(ngrid[1]):
            for k in range(ngrid[2]):
                count = 0
                igal = chainlist[i,j,k]
                while igal != -1:
                    count += 1
                    igal = linklist[igal]
                if count != ngal[i,j,k]:
                    print(i,j,k, count, ngal[i,j,k])
                    grid_good = False
    if grid_good:
        print('Grid construction was successful.')
    '''
    ################################################################################
    #
    #   SEPARATION
    #
    ################################################################################
    
    if rm_isolated_flag:
        sep_start = time.time()

        print('Finding sep',flush=True)

        l, avsep, sd, dists3 = av_sep_calc(coord_in_table)

        print('Average separation of n3rd gal is', avsep, flush=True)
        print('The standard deviation is', sd,flush=True)

        # l = 5.81637  # s0 = 7.8, gamma = 1.2, void edge = -0.8
        # l = 7.36181  # s0 = 3.5, gamma = 1.4
        # or force l to have a fixed number by setting l = ****

        print('Going to build wall with search value', l, flush=True)

        sep_end = time.time()

        print('Time to find sep =',sep_end-sep_start, flush=True)

        fw_start = time.time()

        f_coord_table, w_coord_table = field_gal_cut(coord_in_table, dists3, l)

    else:
        w_coord_table = coord_in_table
        f_coord_table = Table(names=coord_in_table.colnames)

    print(f_coord_table.colnames)
    print(f_coord_table.dtype)

    f_coord_table.write(survey_name + 'field_gal_file.txt', format='ascii.commented_header', overwrite=True)
    w_coord_table.write(survey_name + 'wall_gal_file.txt', format='ascii.commented_header', overwrite=True)


    if rm_isolated_flag:
        fw_end = time.time()

        print('Time to sort field and wall gals =', fw_end-fw_start, flush=True)


    nf =  len(f_coord_table)
    nwall = len(w_coord_table)
    print('Number of field gals:', nf, 'Number of wall gals:', nwall, flush=True)

    return coord_min_table, ngrid[0]



def filter_galaxies_2(galaxy_table,
                      survey_name, 
                      mag_cut_flag=True, 
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
        
        r_gal = c*galaxy_table['redshift'].data/(100*h)
        
        
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
    
    Using the VoidFinder algorithm, this function grows spheres in each empty 
    grid cell of the galaxy distribution.  It then combines these spheres into 
    unique voids, identifying a maximal sphere for each void.

    This algorithm at a high level uses 3 structures to find voids
    in the super-galactic structure of the universe:
    
    1).  The galaxy coordinates
    2).  A survey-limiting mask
    3).  A cell grid of potential void locations
    
    At each grid cell, VoidFinder will grow a maximal sphere until it is
    geometrically bounded by 4 galaxies.  Once every cell location has been searched,
    these pre-voids will be filtered such that the potential voids along the edge of
    the survey will be removed, since any void on the edge of the survey could
    potentially grow unbounded, and there may be galaxies not present which would
    have bounded the void.  After the filtering, these pre-voids will be combined
    into the actual voids based on an analysis of their overlap.
    
    This implementation uses a reference point, 'coords_min', and the 
    'hole_grid_edge_length' to convert between the x,y,z coordinates of a galaxy,
    and the i,j,k coordinates of a cell in the search grid such that:
    
    ijk = ((xyz - coords_min)/hole_grid_edge_length).astype(integer) 
    
    During the sphere growth, VoidFinder also uses a secondary grid to help find
    the bounding galaxies for a sphere.  The grid facilitates nearest-neighbor
    and radius queries, and uses a coordinate space referred to in the code
    as pqr, which uses a similar transformation:
    
    pqr = ((xyz - coords_min)/neighbor_grid_edge_length).astype(integer)
    
    In VoidFinder terminology, a Void is a union of spheres, and a single sphere
    is just a hole.  The Voids are found by taking the set of holes, ordering them
    based on radius, and starting from the largest found hole, label it a maximal
    sphere, and continue to the next hole.  If the next hole does not overlap with any of the
    previous maximal spheres by some factor, it is also considered a maximal sphere.  This process
    is repeated until there are no more maximal spheres, and all other spheres are joined to the
    maximal spheres.
    
    
    Parameters
    ----------
    
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
    #   SORT HOLES BY SIZE
    #
    ############################################################################

    sort_start = time.time()

    print('Sorting holes by size', flush=True)

    potential_voids_table = Table(x_y_z_r_array, names=('x','y','z','radius'))

    potential_voids_table.sort('radius')
    
    potential_voids_table.reverse()

    sort_end = time.time()

    print('Holes are sorted; Time to sort holes =', sort_end-sort_start, flush=True)


    ############################################################################
    #
    #   CHECK IF 90% OF VOID VOLUME IS WITHIN SURVEY LIMITS
    #
    ############################################################################

    print('Removing holes with at least 10% of their volume outside the mask', flush=True)


    potential_voids_table = volume_cut(potential_voids_table, mask, 
                                       mask_resolution, dist_limits)

    potential_voids_table.write(potential_voids_filename, format='ascii.commented_header', overwrite=True)

    ############################################################################
    #
    #   FILTER AND SORT HOLES INTO UNIQUE VOIDS
    #
    ############################################################################

    combine_start = time.time()

    print('Combining holes into unique voids', flush=True)

    maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table)

    print('Number of unique voids is', len(maximal_spheres_table), flush=True)

    # Save list of all void holes
    myvoids_table.write(void_table_filename, format='ascii.commented_header', overwrite=True)

    combine_end = time.time()

    print('Time to combine holes into voids =', combine_end-combine_start, flush=True)

    '''
    ############################################################################
    #
    #   COMPUTE VOLUME OF EACH VOID
    #
    ############################################################################
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
    
    
    ############################################################################
    #
    #   IDENTIFY VOID GALAXIES
    #
    ############################################################################
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

    '''
    ############################################################################
    #
    #   VOID REGION SIZES
    #
    ############################################################################


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

