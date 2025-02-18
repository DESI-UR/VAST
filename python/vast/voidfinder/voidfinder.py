#VoidFinder Function to do just about everything


import numpy as np

from astropy.table import Table

from sklearn import neighbors

import time

import pickle

from .hole_combine import combine_holes_2

from .voidfinder_functions import mesh_galaxies, save_maximals, xyz_to_radecz

from .volume_cut import check_hole_bounds

from .avsepcalc import av_sep_calc

#from .mag_cutoff_function import field_gal_cut

from ._voidfinder import _hole_finder

from .constants import c

from ._voidfinder_cython_find_next import MaskChecker

from .postprocessing import save_output_from_filter_galaxies, save_output_from_wall_field_separation, save_output_from_find_voids



################################################################################
# Debugging imports
################################################################################
#from sklearn.neighbors import KDTree



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
                    capitalize_colnames=False,
                    verbose=0):
    """
    A hodge podge of miscellaneous tasks which need to be done to format the 
    data into something the main find_voids() function can use.
    
    1) Optional magnitude cut
    2) Convert from ra-dec-redshift space into xyz space
    3) Calculate the hole search grid shape
    4) Optional remove isolated galaxies by partitioning them into wall 
       (non-isolated) and field (isolated) groups
    5) Optionally write out the wall and field galaxies to disk
    
    
    Parameters
    ==========
    
    galaxy_table : astropy.table of shape (N,?)
        variable number of required columns.  If doing magnitude cut, must 
        include 'rabsmag' column. If distance metric is 'comoving', must include 
        'Rgal' column, otherwise must include 'redshift'.  Also must always 
        include 'ra' and 'dec'
        
    survey_name : str
        Name of the galaxy catalog, string value to prepend or append to output 
        names

    out_directory : string
        Directory path for output files
        
    mag_cut : bool
        whether or not to cut on magnitude, removing galaxies less than 
        magnitude_limit

    dist_limits : list of length 2
        [Minimum distance, maximum distance] of galaxy sample (in units of 
        Mpc/h)
        
    magnitude_limit : float
        value at which to perform magnitude cut

    rm_isolated : bool
        whether or not to perform Nth neighbor distance calculation, and use it 
        to partition the input galaxies into wall and field galaxies

    write_table : bool
        use astropy.table.Table.write to write out the wall and field galaxies 
        to file

    sep_neighbor : int, positive
        if rm_isolated_flag is true, find the Nth galaxy neighbors based on this 
        value

    dist_metric : str
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where H0 = 
        100h).    
    
    capitalize_colnames : bool
        If True, the column names in the void table outputs are capitalized. 
        Otherwise, the column names are lowercase   

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
        coordinates of the minimum of the survey used for converting from xyz 
        space into ijk space
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
                                                              verbose=verbose,
                                                              survey_name=survey_name, 
                                                              out_directory=out_directory,
                                                              capitalize_colnames=capitalize_colnames)

    else:
        
        wall_gals_xyz = coords_xyz
        
        field_gals_xyz = np.array([])
    ############################################################################



    ############################################################################
    # Write results to disk if desired
    #---------------------------------------------------------------------------
    if write_table:

        write_start = time.time()

        save_output_from_filter_galaxies(
            survey_name, 
            out_directory,
            wall_gals_xyz, 
            field_gals_xyz,
            write_table,
            mag_cut, 
            dist_limits,
            rm_isolated,
            dist_metric, 
            h,
            magnitude_limit,
            capitalize_colnames,
            verbose
        )

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
        must contain columns 'ra' and 'dec' in degrees, and either 'Rgal' in who 
        knows what unit if distance_metric is 'comoving' or 'redshift' for 
        everything else
        
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







def calculate_grid(galaxy_coords,
                   mask_type='ra_dec_z',
                   xyz_limits=None,
                   hole_grid_edge_length=5.0, 
                   galaxy_map_grid_edge_length=None, 
                   verbose=0):
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
    
    galaxy_coords : astropy Table or numpy.ndarray of shape (N,3)
        coordinates of survey galaxies in either sky coordinates or xyz space

    mask_type : string, one of ['ra_dec_z', 'xyz', 'periodic']
        Determines the mode of mask checking to use and which mask parameters to 
        use.  
        
        'ra_dec_z' means the mask, mask_resolution, and dist_limits parameters
            must be provided.  The 'mask' represents an angular space in Right 
            Ascension and Declination, the corresponding mask_resolution integer 
            represents the scale needed to index into the Right Ascension and 
            Declination of the mask, and the dist_limits represent the min and 
            max redshift values (as radial distances in xyz space).
        
        'xyz' means that the xyz_limits parameter must be provided which 
            directly encodes a bounding box for the survey in xyz space
            
        'periodic' means that the xyz_limits parameter must be provided, which 
            directly encodes a bounding box representing the periodic boundary 
            of the survey, and the survey will be treated as if its bounding box 
            were tiled to infinity in all directions.  Spheres will still only 
            be grown starting from within the original bounding box.

    xyz_limits : numpy array of shape (2,3)
        format [x_min, y_min, z_min]
               [x_max, y_max, z_max]
        to be used for checking against the mask when mask_type == 'xyz' or for
        periodic conditions when mask_type == 'periodic'
        
    hole_grid_edge_length : float
        Length in xyz space of the edge of 1 cubical cell in the grid.  Default 
        value is 5 Mpc/h.

    galaxy_map_grid_edge_length : float or None
        Edge length in Mpc/h for the secondary grid for finding nearest neighbor 
        galaxies.  If None, will default to 3*hole_grid_edge_length (which 
        results in a cell volume of 3^3 = 27 times larger cube volume).  This 
        parameter yields a tradeoff between number of galaxies in a cell, and 
        number of cells to search when growing a sphere.  Too large and many 
        redundant galaxies may be searched, too small and too many cells will 
        need to be searched.

    verbose : int
        Level of verbosity to print during running, 0 indicates off, 1 indicates 
        to print after every 'print_after' cells have been processed, and 2 
        indicates to print all debugging statements.  Default is 0.
        
        
    Returns
    =======
    
    hole_grid_shape : tuple of ints (i,j,k)
        number of grid cells in each dimension

    galaxy_map_grid_shape : tuple of ints (i,j,k)
        number of galaxy map grid cells in each dimension
        
    coords_min : numpy.ndarray of shape (3,)
        the (min_x, min_y, min_z) point which is the (0,0,0) of the grid
    """

    ############################################################################
    # Do some sanity checking on the mask modes and various inputs since we have 
    # a lot of None type optional inputs
    #---------------------------------------------------------------------------
    if (mask_type in ['xyz', 'periodic']) and (xyz_limits is None):
        raise ValueError("Mask type is %s but required mask parameter xyz_limits is None" % (mask_type))
    ############################################################################


    ############################################################################
    # Depending on the mask mode, calculate the transform origin and grid cell
    # parameters for our Hole grid and our Galaxy Map grids.
    #
    # For 'ra-dec-z' - just use the min and max of the provided coordinates of 
    #     the survey for the transform
    #
    # For 'xyz' mode - use the required/provided xyz_limits
    #
    # For 'periodic' mode - use the required/provided xyz_limits, but we also 
    #     have to ensure the GalaxyMap cells align with the xyz_limits 
    #     boundaries, so that VoidFinder doesn't accidentally introduce dead 
    #     space into cells because of misalignment.  
    #
    # The important values calculated below are:
    #    'coords_min' - origin for transforming between real and index spaces
    #    'hole_grid_shape' - grid cells for the hole growing grid
    #    'galaxy_map_grid_shape' - grid cells for the galaxy finding grid
    #---------------------------------------------------------------------------
    if mask_type == 'ra_dec_z': #ra-dec-redshift

        # Convert galaxy coords to Cartesian
        galaxy_coords = ra_dec_to_xyz(galaxy_coords)
        
        # Find the maximum coordinate in each direction
        coords_max = np.max(galaxy_coords, axis=0)
        
        # Find the minimum coordinate in each direction
        coords_min = np.min(galaxy_coords, axis=0)
        
        # Size of the grid in each direction
        box = coords_max - coords_min
        
        # Number of grid cells in each direction
        ngrid = box/hole_grid_edge_length
        
        # Shape of the grid
        hole_grid_shape = tuple(np.ceil(ngrid).astype(int))

        # Calculate the size of a galaxy map grid cell
        if galaxy_map_grid_edge_length is None:
            galaxy_map_grid_edge_length = 3.0*hole_grid_edge_length

        # Number of galaxy map grid cells in each direction
        ngrid_galaxymap = box/galaxy_map_grid_edge_length
        
        # Shape of the galaxy map grid
        galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))

        
    elif mask_type == 'xyz': #xyz
        
        # Size of the grid in each direction
        box = xyz_limits[1,:] - xyz_limits[0,:]
        
        # Number of grid cells in each direction
        ngrid = box/hole_grid_edge_length
        
        # Shape of the grid
        hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
        
        # Origin of the grid
        coords_min = xyz_limits[0,:]

        # Calculate the size of a galaxy map grid cell
        if galaxy_map_grid_edge_length is None:
            galaxy_map_grid_edge_length = 3.0*hole_grid_edge_length
        
        # Number of galaxy map grid cells in each direction
        ngrid_galaxymap = box/galaxy_map_grid_edge_length
        
        # Shape of the galaxy map grid
        galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))
        
        
    elif mask_type == 'periodic': #periodic
        
        # Size of the grid in each direction
        box = xyz_limits[1,:] - xyz_limits[0,:]
        
        # Number of grid cells in each direction
        ngrid = box/hole_grid_edge_length
        
        # Shape of the grid
        hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
        
        # Origin of the grid
        coords_min = xyz_limits[0,:]
        
        # Calculate the size of a galaxy map grid cell
        if galaxy_map_grid_edge_length is None:
            
            desired_length = 3.0*hole_grid_edge_length
            
            # Find the common integer divisors of the length dimensions of the 
            # survey limits
            common_divisors = get_common_divisors(box)
            
            if len(common_divisors) == 0 or \
               (len(common_divisors) == 1 and common_divisors[0] == 1):
                
                error_str = """Could not automatically determine meaningful 
                galaxy_map_grid_edge_length from the provided xyz_limits.  In 
                mask_mode==periodic, the survey limits provided by the 
                xyz_limits variable must be divisible by a common integer in all 
                dimensions"""
                
                raise ValueError(error_str)
            
            common_divisors = np.array(common_divisors)
            
            argmin = np.abs(common_divisors - desired_length).argmin()
            
            galaxy_map_grid_edge_length = float(common_divisors[argmin])
            
            # Number of galaxy map grid cells in each direction
            ngrid_galaxymap = box/galaxy_map_grid_edge_length

            # Shape of the galaxy map grid
            galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))
            
        else:
            
            # Number of galaxy map grid cells in each direction
            ngrid_galaxymap = box/galaxy_map_grid_edge_length
            
            rounded = np.rint(ngrid_galaxymap)
            
            print(rounded)
            
            close_to_round = np.isclose(ngrid_galaxymap, rounded)
            
            print(close_to_round)
            
            if np.all(close_to_round):
                # Vals are good, just proceed with given
                galaxy_map_grid_shape = tuple(np.rint(ngrid_galaxymap).astype(int))
            else:
                # Attempt to adjust galaxy_map_grid_edge_length
                error_str = """The provided combination of xyz_limits and 
                galaxy_map_grid_edge length will not work.  In 
                mask_mode==periodic, the edge length must be an integer divisor 
                of all dimensions of the survey as provided by the xyz_limits 
                input."""
    
                raise ValueError(error_str)
        
    # Recast coords_min
    coords_min = coords_min.reshape(1,3).astype(np.float64)
    
    
    if verbose > 0:
        
        print("Hole-growing Grid:", hole_grid_shape, flush=True)
        
        print("Galaxy-searching Grid:", galaxy_map_grid_shape, flush=True)
        
        print("Galaxy-searching edge length:", galaxy_map_grid_edge_length, 
              flush=True)
        
    ############################################################################
    
    '''
    coords_max = np.max(galaxy_coords_xyz, axis=0)
    
    coords_min = np.min(galaxy_coords_xyz, axis=0)
    
    box = coords_max - coords_min

    ngrid = box/hole_grid_edge_length
    
    #print("Ngrid: ", ngrid)
    
    hole_grid_shape = tuple(np.ceil(ngrid).astype(int))
    '''
    return hole_grid_shape, galaxy_map_grid_shape, coords_min#, coords_max
    
    



    

def wall_field_separation(galaxy_coords_xyz,
                          sep_neighbor=3,
                          verbose=0,
                          survey_name = "", 
                          out_directory = "",
                          write_galaxies=False,
                          capitalize_colnames=False):
    """
    Given a set of galaxy coordinates in xyz space, find all the galaxies whose
    distance to their Nth nearest neighbor is above or below some limit.  
    Galaxies whose Nth nearest neighbor is close (below), will become 'wall' 
    galaxies, and galaxies whose Nth nearest neighbor is far (above) will become 
    field/void/isolated galaxies.
    
    The distance limit used below is the mean distance to Nth nearest neighbor 
    plus 1.5 times the standard deviation of the Nth nearest neighbor distance.
    
    
    Parameters
    ==========
    
    galaxy_coords_xyz : numpy.ndarray of shape (N,3)
        coordinates in xyz space of the galaxies
        
    sep_neighbor : int
       Nth neighbor
       
    verbose : int
        whether to print timing output, 0 for off and >= 1 for on

    write_galaxies : bool
        write out the wall and field galaxies to an output file
    
    capitalize_colnames : bool
        If True, the column names in the void table outputs are capitalized. 
        Otherwise, the column names are lowercase       

          
    Returns
    =======
    
    wall_gals_xyz : ndarray of shape (K, 3)
        xyz coordinate subset of the input corresponding to tightly packed 
        galaxies
        
    field_gals_xyz : ndarray of shape (L, 3)
        xyz coordinate subset of the input corresponding to isolated galaxies
        
    """
    
    ############################################################################
    # Check for (and remove) identical galaxies
    #---------------------------------------------------------------------------
    print('Checking for duplicate galaxies', flush=True)
    
    dup_start = time.time()
    
    galaxy_tree = neighbors.KDTree(galaxy_coords_xyz)
    
    distances, indices = galaxy_tree.query(galaxy_coords_xyz, k=2)
    
    duplicates = distances[:,1] == 0.
    
    if np.sum(duplicates) > 0:
        
        galaxy_coords_xyz = galaxy_coords_xyz[~duplicates]
        
        print('Removed', np.sum(duplicates), 'duplicates', flush=True)
        
    dup_end = time.time()
    
    print('Time to check for and remove duplicate galaxies:', dup_end-dup_start, 
          flush=True)
    ############################################################################
    
    
    
    ############################################################################
    # Calculate the average distance to the 3rd nearest neighbor
    #---------------------------------------------------------------------------
    print('Finding isolated galaxy distance', flush=True)
        
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
        
    save_output_from_wall_field_separation(
        survey_name, 
        out_directory,
        wall_gals_xyz,
        field_gals_xyz,
        write_galaxies,
        sep_neighbor,
        avsep,
        sd,
        np.sum(duplicates),
        capitalize_colnames,
        verbose,
    )

    return wall_gals_xyz, field_gals_xyz
    
    






def find_voids(galaxy_coords_xyz,
               survey_name,
               out_directory,
               mask_type='ra_dec_z',
               mask=None, 
               mask_resolution=None,
               dist_limits=None,
               xyz_limits=None,
               check_only_empty_cells=True,
               max_hole_mask_overlap=0.1,
               hole_grid_edge_length=5.0,
               grid_origin=None,
               min_maximal_radius=10.0,
               galaxy_map_grid_edge_length=None,
               pts_per_unit_volume=0.01,
               num_cpus=None,
               save_after=None,
               use_start_checkpoint=False,
               batch_size=10000,
               verbose=0,
               print_after=5.0,
               capitalize_colnames = False,
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
    
    VoidFinder will proceed to grow a sphere (or hole), at every Empty grid 
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
    to Nth nearest neighbor, for example.
    
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
    
    galaxy_coords : numpy.ndarray of shape (num_galaxies, 3)
        coordinates of the galaxies in the survey, units of Mpc/h
        (xyz space)
        
    survey_name : str
        identifier for the survey running, may be prepended or appended to 
        output filenames including the checkpoint filename
    
    out_directory : string
        Directory path for output files
        
    mask_type : string, one of ['ra_dec_z', 'xyz', 'periodic']
        Determines the mode of mask checking to use and which mask parameters to 
        use.  
        
        'ra_dec_z' means the mask, mask_resolution, and dist_limits parameters
            must be provided.  The 'mask' represents an angular space in Right 
            Ascension and Declination, the corresponding mask_resolution integer 
            represents the scale needed to index into the Right Ascension and 
            Declination of the mask, and the dist_limits represent the min and 
            max redshift values (as radial distances in xyz space).
        
        'xyz' means that the xyz_limits parameter must be provided which 
            directly encodes a bounding box for the survey in xyz space
            
        'periodic' means that the xyz_limits parameter must be provided, which 
            directly encodes a bounding box representing the periodic boundary 
            of the survey, and the survey will be treated as if its bounding box 
            were tiled to infinity in all directions.  Spheres will still only 
            be grown starting from within the original bounding box.
        
    mask : numpy.ndarray of shape (N,M) type bool
        Represents the survey footprint in scaled ra/dec space.  Value of True 
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
        Size in Mpc/h of the edge of 1 cube in the search grid, or distance 
        between 2 grid cells
        (xyz space)

    grid_origin : ndarray of shape (3,) or None
        The spatial location to use as (0,0,0) in the search grid.
        if None, will use the numpy.min() function on the provided galaxies
        as the grid origin

    min_maximal_radius : float
        The minimum radius in units of distance for a hole to be considered
        for maximal status.  Default value is 10 Mpc/h.
        
    max_hole_mask_overlap : float in range (0, 0.5)
        When the volume of a hole overlaps the mask by this fraction, discard 
        that hole.  Maximum value of 0.5 because a value of 0.5 means that the 
        hole center will be outside the mask, but more importantly because the 
        numpy.roots() function used below won't return a valid polynomial root.
        
    galaxy_map_grid_edge_length : float or None
        Edge length in Mpc/h for the secondary grid for finding nearest neighbor 
        galaxies.  If None, will default to 3*hole_grid_edge_length (which 
        results in a cell volume of 3^3 = 27 times larger cube volume).  This 
        parameter yields a tradeoff between number of galaxies in a cell, and 
        number of cells to search when growing a sphere.  Too large and many 
        redundant galaxies may be searched, too small and too many cells will 
        need to be searched.
        (xyz space)
        
    hole_center_iter_dist : float
        Distance to move the sphere center each iteration while growing a void
        sphere in units of Mpc/h
        (xyz space)

    pts_per_unit_volume : float
        Number of points per unit volume that are distributed within the holes 
        to calculate the fraction of the hole's volume that falls outside the 
        survey bounds.  Default is 0.01.
    
    num_cpus : int or None
        Number of cpus to use while running the main algorithm.  None will 
        result in using number of physical cores on the machine.  Some speedup 
        benefit may be obtained from using additional logical cores via Intel 
        Hyperthreading but with diminishing returns.  This can safely be set 
        above the number of physical cores without issue if desired.
        
    save_after : int or None
        Save a VoidFinderCheckpoint.h5 file after *approximately* every 
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
        Whether or not to start growing a hole in a cell which has galaxies in
        it, aka "non-empty".  If True (default), don't grow holes in these 
        cells.
    
    use_start_checkpoint : bool
        Whether to attempt looking for a VoidFinderCheckpoint.h5 file which can 
        be used to restart the VF run.  If False, VoidFinder will start fresh 
        from 0.
    
    batch_size : int
        Number of potential void cells to evaluate at a time.  Lower values may 
        be a bit slower as it involves some memory allocation overhead, and 
        values which are too high may cause the status update printing to take 
        more than print_after seconds.  Default value 10,000
        
    verbose : int or bool
        Level of verbosity to print during running, 0 indicates off, 1 indicates 
        to print after every 'print_after' cells have been processed, and 2 
        indicates to print all debugging statements
        
    print_after : float
        Number of seconds to wait before printing a status update

    capitalize_colnames : bool
        If True, the column names in the void table outputs are capitalized. 
        Otherwise, the column names are lowercase
    
    
    Returns
    =======
    
    All output is currently written to disk:
        
    combined voids table, fits format
    
    maximal spheres table, fits format
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
    
    
    
    try:
        verbose = int(verbose)
    except:
        raise ValueError("verbose argument invalid, must be int or bool: "+str(verbose))
    
    ############################################################################
    # GROW HOLES
    #---------------------------------------------------------------------------
    
    if verbose > 0:
        print('Growing holes', flush=True)
        
        print("Input galaxies shape: ", galaxy_coords_xyz.shape)

        tot_hole_start = time.time()

    x_y_z_r_array, n_holes = _hole_finder(galaxy_coords_xyz,
                                          hole_grid_edge_length,
                                          galaxy_map_grid_edge_length,
                                          survey_name,
                                          grid_origin=grid_origin,
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

    if verbose > 0:
        print('Found a total of', n_holes, 'potential voids.', flush=True)
    
        print('Time to grow all holes:', time.time() - tot_hole_start, flush=True)
    ############################################################################




    ############################################################################
    # Debug Section
    ############################################################################
    """
    #print(galaxy_coords_xyz[0].shape)
    #print(x_y_z_r_array.shape)
    
    galaxy_tree = KDTree(galaxy_coords_xyz[0])
    distances, indices = galaxy_tree.query(x_y_z_r_array[:,0:3], k=1)
    #print(distances.shape, indices.shape)
    
    has_bad_hole = False
    
    for row in x_y_z_r_array:
    
        curr_pos = row[0:3].reshape(1,3)
        curr_rad = row[3]
        
        ind, dist = galaxy_tree.query_radius(curr_pos, 0.99*curr_rad, return_distance=True)
    
        ind = ind[0]
        
        if ind.shape[0] > 0:
        
            print("Bad Hole, num galaxies inside: ", ind.shape)
            
            has_bad_hole = True
            
    if has_bad_hole:
        print("ERROR: DEBUG SECTION FOUND HOLE WITH ONE OR MORE GALAXIES INSIDE")
    """
    ############################################################################
    # End Debug Section
    ############################################################################







    ############################################################################
    # Initialize mask object
    #---------------------------------------------------------------------------
    if mask_mode == 0: # sky mask
        mask_checker = MaskChecker(mask_mode,
                                   survey_mask_ra_dec=mask.astype(np.uint8),
                                   n=mask_resolution,
                                   rmin=min_dist,
                                   rmax=max_dist)
        
    elif mask_mode in [1,2]: # Cartesian mask
        mask_checker = MaskChecker(mask_mode,
                                   xyz_limits=xyz_limits)
    ############################################################################




    
    ############################################################################
    # CHECK IF 90% OF VOID VOLUME IS WITHIN SURVEY LIMITS
    #---------------------------------------------------------------------------
    if verbose > 0:
        print("Starting volume cut", flush=True)
    
        vol_cut_start = time.time()
    
    valid_idx, monte_index = check_hole_bounds(x_y_z_r_array, 
                                               mask_checker,
                                               cut_pct=0.1,
                                               pts_per_unit_volume=pts_per_unit_volume,
                                               num_surf_pts=20,
                                               num_cpus=num_cpus,
                                               verbose=verbose)
    
    if verbose > 0:
        print("Found", np.sum(np.logical_not(valid_idx)), "holes to cut", 
              time.time() - vol_cut_start, flush=True)

    x_y_z_r_array = x_y_z_r_array[valid_idx]

    # Array that stores whether or not any part of holes fall outside survey
    boundary_hole = monte_index[valid_idx]
    ############################################################################


    
    ############################################################################
    # SORT HOLES BY SIZE
    #---------------------------------------------------------------------------
    if verbose > 0:
        print("Sorting holes based on radius", flush=True)
    
    sort_order = x_y_z_r_array[:,3].argsort()[::-1]
    
    x_y_z_r_array = x_y_z_r_array[sort_order]
    
    boundary_hole = boundary_hole[sort_order]
    ############################################################################

    

    ############################################################################
    # FILTER AND SORT HOLES INTO UNIQUE VOIDS
    #---------------------------------------------------------------------------
    if verbose > 0:
        print("Combining holes into unique voids", flush=True)
    
        combine_start = time.time()
    
    maximal_spheres_table, myvoids_table = combine_holes_2(x_y_z_r_array, 
                                                           boundary_hole, 
                                                           mask_checker,
                                                           min_maximal_radius=min_maximal_radius,
                                                           verbose=verbose)
    
    if verbose > 0:
        print("Combine time:", time.time() - combine_start, flush=True)
    
        print('Number of unique voids is', len(maximal_spheres_table), flush=True)
    ############################################################################
        

    ############################################################################
    # Save list of all void holes and maximal spheres
    #---------------------------------------------------------------------------

    #format column names for output file
    myvoids_table['x'].unit='Mpc/h'
    myvoids_table['y'].unit='Mpc/h'
    myvoids_table['z'].unit='Mpc/h'
    myvoids_table['radius'].unit='Mpc/h'

    maximal_spheres_table = xyz_to_radecz(maximal_spheres_table)

    maximal_spheres_table['x'].unit='Mpc/h'
    maximal_spheres_table['y'].unit='Mpc/h'
    maximal_spheres_table['z'].unit='Mpc/h'
    maximal_spheres_table['radius'].unit='Mpc/h'
    maximal_spheres_table['r'].unit='Mpc/h'
    maximal_spheres_table['ra'].unit='deg'
    maximal_spheres_table['dec'].unit='deg'

    if capitalize_colnames: 

        for colname in myvoids_table.colnames:
            myvoids_table[colname].name=colname.upper()
            
        for colname in maximal_spheres_table.colnames:
            maximal_spheres_table[colname].name=colname.upper()

    #save output
    save_output_from_find_voids(
        maximal_spheres_table,
        myvoids_table, 
        galaxy_coords_xyz,
        out_directory, 
        survey_name,
        mask_type,
        mask,            
        mask_resolution,
        dist_limits,     
        xyz_limits,     
        check_only_empty_cells,
        max_hole_mask_overlap,  
        hole_grid_edge_length,   
        grid_origin,             
        min_maximal_radius,
        galaxy_map_grid_edge_length,
        pts_per_unit_volume,
        num_cpus,
        batch_size,
        capitalize_colnames,
        verbose=verbose
    )
    
    ############################################################################
    # Compute volume of each void
    #---------------------------------------------------------------------------
    ############################################################################


    ############################################################################
    # Void region size
    #---------------------------------------------------------------------------
    ############################################################################
    
    
    return maximal_spheres_table, myvoids_table




