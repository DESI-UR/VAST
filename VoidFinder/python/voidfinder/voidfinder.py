#VoidFinder Function to do just about everything


import numpy as np

from astropy.table import Table

import time

from .hole_combine import combine_holes

from .voidfinder_functions import build_mask, \
                                  mesh_galaxies, \
                                  in_mask, \
                                  not_in_mask, \
                                  in_survey, \
                                  save_maximals, \
                                  mesh_galaxies_dict

from .table_functions import add_row, \
                             subtract_row, \
                             to_vector, \
                             to_array, \
                             table_dtype_cast, \
                             table_divide

from .volume_cut import volume_cut

from .avsepcalc import av_sep_calc

from .mag_cutoff_function import mag_cut, field_gal_cut

from ._voidfinder import _main_hole_finder


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
                    mask_array, 
                    mask_resolution, 
                    min_dist, 
                    max_dist, 
                    survey_name, 
                    mag_cut_flag, 
                    rm_isolated_flag, 
                    distance_metric, 
                    h,
                    c=3e5,
                    search_grid_edge_length=5,
                    ):
    '''
    Filter the input galaxy catalog, removing galaxies fainter than some limit 
    and removing isolated galaxies.


    Parameters:
    ===========

    galaxy_table : astropy table
        List of galaxies and their coordinates (ra, dec, redshift) and magnitudes

    mask_array : numpy array of shape (2,n)
        n pairs of RA,dec coordinates that are within the survey limits and are 
        scaled by the mask_resolution.  Oth row is RA; 1st row is dec.

    mask_resolution : integer
        Scale factor of coordinates in mask_array

    min_dist : float
        Minimum distance (in Mpc/h) of the galaxy distribution

    max_dist : float
        Maximum distance (in Mpc/h) of the galaxy distribution

    survey_name : string
        Name of galaxy catalog

    mag_cut_flag : boolean
        Determines whether or not to remove galaxies fainter than Mr = -20.  True 
        will remove the faint galaxies.

    rm_isolated_flag : boolean
        Determines whether or not to remove isolated galaxies.  True will remove 
        the isolated galaxies.

    distance_metric : string
        Distance metric to use in calculations.  Options are 'comoving' 
        (distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).
        
    c : Kelly can fill this in
    
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
    print('Pre-processing data', flush=True)

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
    print('There are', N_gal, 'galaxies in this simulation.', flush=True)

    # Convert coord_in, coord_min, coord_max tables to numpy arrays
    coord_in = to_array(coord_in_table)
    coord_min = to_vector(coord_min_table)
    coord_max = to_vector(coord_max_table)


    print('Reading mask',flush=True)

    mask = build_mask(mask_array, mask_resolution)

    print('Read mask',flush=True)

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

    #print('box shape:', box.shape)

    # Array of number of cells in each direction
    ngrid = box/search_grid_edge_length
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


    f_coord_table.write(survey_name + 'field_gal_file.txt', format='ascii.commented_header', overwrite=True)
    w_coord_table.write(survey_name + 'wall_gal_file.txt', format='ascii.commented_header', overwrite=True)


    if rm_isolated_flag:
        fw_end = time.time()

        print('Time to sort field and wall gals =', fw_end-fw_start, flush=True)


    nf =  len(f_coord_table)
    nwall = len(w_coord_table)
    print('Number of field gals:', nf, 'Number of wall gals:', nwall, flush=True)

    return coord_min_table, mask, ngrid[0]












def find_voids(void_grid_shape, 
               min_dist, 
               max_dist,
               galaxy_coords,
               coord_min, 
               mask, 
               mask_resolution,
               void_grid_edge_length=5,
               search_grid_edge_length=None,
               hole_center_iter_dist=1.0,
               maximal_spheres_filename="maximal_spheres.txt",
               void_table_filename="voids_table.txt",
               potential_voids_filename="potential_voids_list.txt",
               num_cpus=None,
               batch_size=10000,
               verbose=1,
               print_after=5.0,
               ):
    """
    Description
    -----------
    
    Main entry point for VoidFinder.  
    
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
    
    This implementation uses a reference point, 'coord_min', and the 
    'void_grid_edge_length' to convert between the x,y,z coordinates of a galaxy,
    and the i,j,k coordinates of a cell in the search grid such that:
    
    ijk = ((xyz - coord_min)/void_grid_edge_length).astype(integer) 
    
    During the sphere growth, VoidFinder also uses a secondary grid to help find
    the bounding galaxies for a sphere.  The grid facilitates nearest-neighbor
    and radius queries, and uses a coordinate space referred to in the code
    as pqr, which uses a similar transformation:
    
    pqr = ((xyz - coord_min)/neighbor_grid_edge_length).astype(integer)
    
    
    Parameters
    ----------
    
    void_grid_shape : array or tuple of length 3
        number of grid cells in each direction to search as potential
        void centers
    
    min_dist : float
        minimum redshift limit of the survey in units of Mpc/h
        (xyz space)
        
    max_dist : float
        maximum redshift limit of the survey in units of Mpc/h
        (xyz space)
        
    galaxy_coords : numpy.ndarray of shape (num_galaxies, 3)
        coordinates of the galaxies in the survey, units of Mpc/h
        (xyz space)
    
    coord_min : ndarray of shape (1,3)
        minimum coordinates of the galaxy survey
        (xyz space)
    
    mask : numpy.ndarray of shape (N,M) type bool
        represents the survey footprint in scaled ra/dec space.  Value of True 
        indicates that a location is within the survey
        (ra/dec space)

    mask_resolution : integer
        Scale factor of coordinates needed to index mask
        
    void_grid_edge_length : float
        size in Mpc/h of the edge of 1 cube in the search grid, or
        distance between 2 grid cells
        (xyz space)
        
    search_grid_edge_length : float or None
        edge length in Mpc/h for the secondary grid for finding nearest neighbor
        galaxies.  If None, will default to 3*void_grid_edge_length (which results
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
    
    potential voids table, ascii.commented_header format
    
    combined voids table, ascii.commented_header format
    
    maximal spheres table
    
    """
    

    ################################################################################
    #
    #   GROW HOLES
    #
    ################################################################################

    if search_grid_edge_length is None:
        
        search_grid_edge_length = 3.0*void_grid_edge_length

    tot_hole_start = time.time()

    print('Growing holes', flush=True)

    x_y_z_r_array, n_holes = _main_hole_finder(void_grid_shape, 
                                               void_grid_edge_length, 
                                               hole_center_iter_dist,
                                               search_grid_edge_length,
                                               coord_min,
                                               mask,
                                               mask_resolution,
                                               min_dist,
                                               max_dist,
                                               galaxy_coords,
                                               batch_size=batch_size,
                                               verbose=verbose,
                                               print_after=print_after,
                                               num_cpus=num_cpus)

    print('Found a total of', n_holes, 'potential voids.', flush=True)

    print('Time to find all holes =', time.time() - tot_hole_start, flush=True)
    
    ################################################################################
    #
    #   SORT HOLES BY SIZE
    #
    ################################################################################

    sort_start = time.time()

    print('Sorting holes by size', flush=True)

    potential_voids_table = Table(x_y_z_r_array, names=('x','y','z','radius'))

    potential_voids_table.sort('radius')
    
    potential_voids_table.reverse()

    sort_end = time.time()

    print('Holes are sorted; Time to sort holes =', sort_end-sort_start, flush=True)

    ################################################################################
    #
    #   CHECK IF 90% OF VOID VOLUME IS WITHIN SURVEY LIMITS
    #
    ################################################################################

    print('Removing holes with at least 10% of their volume outside the mask', flush=True)

    potential_voids_table = volume_cut(potential_voids_table, mask, mask_resolution, [min_dist, max_dist])

    potential_voids_table.write(potential_voids_filename, format='ascii.commented_header', overwrite=True)

    ################################################################################
    #
    #   FILTER AND SORT HOLES INTO UNIQUE VOIDS
    #
    ################################################################################

    combine_start = time.time()

    print('Combining holes into unique voids',flush=True)

    maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table)

    print('Number of unique voids is', len(maximal_spheres_table),flush=True)

    # Save list of all void holes
    myvoids_table.write(void_table_filename, format='ascii.commented_header', overwrite=True)

    combine_end = time.time()

    print('Time to combine holes into voids =', combine_end-combine_start,flush=True)

    '''
    ################################################################################
    #
    #   COMPUTE VOLUME OF EACH VOID
    #
    ################################################################################
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
    
    
    ################################################################################
    #
    #   IDENTIFY VOID GALAXIES
    #
    ################################################################################
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

    ################################################################################
    #
    #   MAXIMAL HOLE FOR EACH VOID
    #
    ################################################################################

    save_maximals(maximal_spheres_table, maximal_spheres_filename)

    '''
    ################################################################################
    #
    #   VOID REGION SIZES
    #
    ################################################################################


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

