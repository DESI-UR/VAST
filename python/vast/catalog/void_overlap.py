
from astropy.table import Table, join
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import time
from sklearn import neighbors

from vast.voidfinder._voidfinder_cython_find_next import MaskChecker
from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder import ra_dec_to_xyz
from vast.voidfinder.voidfinder_functions import not_in_mask
import pickle

# TODO: spellcheck and format documentation
# TODO: Running on large void catalogs becomes memory intensive. Explore ways of reducing memory usage,
# such as breaking up the caclulation into redshift bins (or vertical z bins for cubic volumes)
# TODO: Resolve the VoidFinder maximal sphere point-membership issue (see point_query_VF for details)
# TODO: Revise the find_overlap function to appropriately handle higher values of the variable 'U' 
# (see find_overlap for details)
# TODO: Test level of detail needed for good uncertainties (repetitions of monte carlo)
# TODO: add verbose option
# TODO: Automatic mask generation from vast if the user doesn't provide an input mask
# from vast.voidfinder.multizmask import generate_mask


"""
Authors: Hernan Rincon

Some code has been adopted from the following individuals: [check authors]
# TODO: Add other authors (I believe it's Kelly Douglass and Lorenzo Mendoza, but this should be checked)

"""

# Code for comparing the volume overlap of two void catalogs or the volume filling fraction of a single void catalog

def is_edge_point (x, y, z, mask, mask_resolution, rmin, rmax, edge_buffer):
    
    is_edge = False
    #-------------------------------------------------------------------
    # Is the point within edge_buffer Mpc/h of the survey boundary?
    #-------------------------------------------------------------------
    # Calculate coordinates that are edge_buffer Mpc/h in each Cartesian 
    # direction of the point
    coord_min = np.array([x,y,z]) - edge_buffer
    coord_max = np.array([x,y,z]) + edge_buffer

    # Coordinates to check
    x_coords = [coord_min[0], coord_max[0], x, x, x, x]
    y_coords = [y, y, coord_min[1], coord_max[1], y, y]
    z_coords = [z, z, z, z, coord_min[2], coord_max[2]]
    extreme_coords = np.array([x_coords, y_coords, z_coords]).T

    i = 0
    while is_edge is False and i <= 5:
        # Check to see if any of these are outside the survey
        if not_in_mask(extreme_coords[i].reshape(1,3), mask, mask_resolution, rmin, rmax):
            # Point is within 10 Mpc/h of the survey edge
            is_edge = True
        i += 1
    return is_edge


def calc_volume_boundaries(void_cat_A, void_cat_B):
    """Compute the boundaries of the minimal rectangular volume (parallelpiped)
    that completely contains two void catalogs.
    
    Parameters
    ----------
    void_cat_A : astropy.Table
        Table of void data from first catalog.
    void_cat_B : astropy.Table
        Table of void data from second catalog.
        
    Returns
    -------
    x_min : float
    x_max : float
    y_min : float
    y_max : float
    z_min : float
    z_max : float
    """
    
    x_min = np.minimum(np.min(void_cat_A['x']), np.min(void_cat_B['x']))
    x_max = np.maximum(np.max(void_cat_A['x']), np.max(void_cat_B['x']))
    
    y_min = np.minimum(np.min(void_cat_A['y']), np.min(void_cat_B['y']))
    y_max = np.maximum(np.max(void_cat_A['y']), np.max(void_cat_B['y']))

    z_min = np.minimum(np.min(void_cat_A['z']), np.min(void_cat_B['z']))
    z_max = np.maximum(np.max(void_cat_A['z']), np.max(void_cat_B['z']))

    return x_min, x_max, y_min, y_max, z_min, z_max

def generate_grid_points(x_min, x_max, y_min, y_max, z_min, z_max):
    """Creates a dense rectangular grid of points in 3D for the void volume calculation.
    
    Parameters
    ----------
    x_min : float
    x_max : float
    y_min : float
    y_max : float
    z_min : float
    z_max : float
        The grid boundaries in the x, y, and z dimensions set by calc_volume_boundaries
    
    Returns
    -------
    point_coords : numpy.array with shape (3,N)
        The grid coordinates.
    """
    
    # The default grid spacing is 1 Megaparsec
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    z_range = np.arange(z_min, z_max)
    
    """
    # For debuging purposes, we may wish to use a differnet grid spacing
    num_points_in_range = 100
    x_range = np.linspace(x_min, x_max, num_points_in_range) 
    y_range = np.linspace(y_min, y_max, num_points_in_range)
    z_range = np.linspace(z_min, z_max, num_points_in_range)
    """

    # Creating a meshgrid from the input ranges 
    X,Y,Z = np.meshgrid(x_range,y_range,z_range)

    x_points = np.ravel(X)
    y_points = np.ravel(Y)
    z_points = np.ravel(Z)
    
    point_coords = np.array([x_points, y_points, z_points])
    
    return point_coords

def mask_point_filter(pts, mask, mask_resolution, rmin, rmax, edge_buffer):
    """Determines which grid points generated by generate_grid_points lie in the survey mask
    
    Parameters
    ----------
    pts : numpy.array with shape (3,N)
        The grid points genererated by generate_grid_points
    mask : nmpy.array with shape (N, M)
        The survey mask returned by vast.voidfinder.multizmask.generate_mask
    mask_resolution : int
        The survey mask resolution returend by vast.voidfinder.multizmask.generate_mask
    rmin : float
        The comoving minimum distance within the survey volume
    rmax : float
        The comoving maximum distance within the survey volume
    edge_buffer: float
        The distance from the survey edge at which a point is considered and edge point
    
    Returns
    -------
    points_in_mask : numpy array
        The list of grid points within the survey mask
    """
    # Initalize a boolean array to tag points in the mask
    points_boolean = np.zeros(pts.shape[1], dtype = bool)
    # Declare MaskChecker object from VAST
    mask_checker = MaskChecker(0,
                            mask,
                            mask_resolution,
                            rmin,
                            rmax)
    
    #Flag points that fall outside the mask
    for i in range(pts.shape[1]):
        # The current point
        curr_pt = pts[:,i]
        # Declare if point is not in mask
        not_in_mask = bool(mask_checker.not_in_mask(curr_pt))
        # Invert not_in_mask to tag points in the mask
        if not not_in_mask:
            is_edge = is_edge_point(curr_pt[0], curr_pt[1], curr_pt[2],
                                    mask, mask_resolution, rmin, rmax, edge_buffer)
            points_boolean[i] = not is_edge
        
    # Select all points in the mask
    points_in_mask = pts[:,points_boolean]
    return points_in_mask


def kd_tree(void_cat):
    """Creates a kdtree from the x-y-z coorinates of a void catalog.
    
    Parameters
    ----------
    void_cat: Astropy Table
        The given void catalogue which includes columns 'x', 'y', and 'z'
    
    Returns
    -------
    sphere_tree: sklearn.neighbors._kd_tree.KDTree
        The kdtree object for the void catalog
    """
    cx = void_cat['x']
    cy = void_cat['y']
    cz = void_cat['z']

    # Format the void centers in an array of shape (N, 3)
    sphere_coords = np.array([cx, cy, cz]).T

    sphere_tree = neighbors.KDTree(sphere_coords)
    
    return sphere_tree

def point_query_VF(point_coords, sphere_tree, void_cat, max_sphere_size):
    """Determines which members of a set of grid points are inside VoidFinder voids
    
    Parameters
    ----------
    point_coords : numpy.array with shape (3,N)
        The grid points within the survey volume
    sphere_tree: sklearn.neighbors._kd_tree.KDTree
        The kdtree object for the void catalog genreated by kd_tree
    void_cat: Astropy Table
        The VoidFinder void catalogue which includes a void radius column 'radius'
    max_sphere_size : float
        A currently unused parameter corresponding to the maximum sphere size in the void catalog
    
    Returns
    -------
    true_inside : numpy.array with shape (3,N)
        Indicates which grid points are inside voids
    """
    # Initialize the array that will tag which points are in voids
    true_inside = np.full(point_coords.shape[1], False, dtype=bool)
    
    # k determines the number of nearest neighbors in sphere_tree that are checked for each grid point
    # The default behavior is to check only the single nearest void for void membership
    # It is possible tht more distant voids are very large and enclose a grid point that is not enclsed by
    # its nearest neighbor void, but this sourve of error is currently not accounted for
    # Setting k to a higher value will improve this error at the cost of runtime
    k = min(sphere_tree.data.shape[0], 1) #change 1 to 5 for testing
    
    # Query the sphere tree
    dist, idx = sphere_tree.query(point_coords.T, k = k)
    
    # Check with points are within void interiors
    interiors = dist < void_cat['radius'][idx]
    # Perform an OR operation on the k interiors columns to tag any point inside any of the k voids as
    # belonging to a void
    for i in range (k):
        true_inside += interiors[:,i]
    """
    # Old (possibly non-funtional) code that uses max_sphere_size to check the complete void catalog for each point's void 
    # membership without having to iterate over each void in the catalog
    # This would be the ideal approach for 100% accuracy, but sphere_tree.query_radius returns a jagged list structure,
    # complicating the otherwise time-efficient numpy array caclulations
    # Revisiting this code may be of interest if we can find a way to speed it up
    ind, dist = sphere_tree.query_radius(point_coords.T, max_sphere_size,return_distance=True)
    true_inside = np.array([np.isin(True, void_cat['radius'][ind_ball]<dist_ball) for (ind_ball,dist_ball) in zip(ind, dist)],dtype=bool)
    """
    return true_inside

def point_query_V2(point_coords, sphere_tree, void_cat):
    """Determines which members of a set of grid points are inside V2 voids
    
    Parameters
    ----------
    point_coords : numpy.array with shape (3,N)
        The grid points within the survey volume
    sphere_tree: sklearn.neighbors._kd_tree.KDTree
        The kdtree object for the void catalog genreated by kd_tree
    void_cat: Astropy Table
        The V2 void catalogue which includes a void radius column 'radius'
    
    Returns
    -------
    true_inside : numpy.array with shape (3,N)
        Indicates which grid points are inside voids
    """
    # For V2, k=1 is sufficient to ensure 100% accuracy in void membership
    idx = sphere_tree.query(point_coords.T, k = 1, return_distance=False)
    
    true_inside = void_cat[idx]['in_void']

    return true_inside

def prep_V2_cat(V2_galzones, V2_zonevoids):
    """Formates a V2 catalog for use with the void overlap calclator
    
    Parameters
    ----------
    V2_galzones: Astropy Table
        The V2 galzones catalogue
    V2_zonevoids: Astropy Table
        The V2 zonevoids catalogue
    Returns
    -------
    V2_galzones : Astropy table
        The V2_galzones object formatted for use with teh volume overlap caculator
    """
    
    #goal: add void column to galzones
    
    # Determine galaxy void Membership
    # ----------------------------
    
    #match the galaxy zones with void IDs
    zone_voids = V2_zonevoids['zone','void0']
    zone_voids.add_row([-1, -1])
    V2_galzones['__sort__'] = np.arange(len(V2_galzones))
    void_IDs = join(V2_galzones, zone_voids, keys='zone',  join_type='left')
    void_IDs.sort('__sort__')
    #mark void galaxies and wall galaxies
    V2_galzones['in_void'] = (void_IDs['void0'] != -1)  #* np.isin(void_IDs['void0'], V2_zonevoids['void'])
    
    return V2_galzones

class OverlapCalculator():
    """Compares the voulume overlap of two void catalogs. 
    
    WARNING: This class assumes that V2 catalogs are generated 
    from the same underlying galaxy catalog. The code must be manually edtied if this isn't the case.
    """
    
    def __init__ (self, data_table_V1, data_table_V2, title1, title2, mask_file, rmin, rmax, zone_table_V1 = None, zone_table_V2 = None, use_mask = True, V1_algorithm="VF", V2_algorithm="VF",mask_tuple = None, num_cpus=1):
        """Formates a V2 catalog for use with the void overlap calclator

        Parameters
        ----------
        data_table_V1: Astropy Table
            The first void catalog
        data_table_V2: Astropy Table
            The second void catalog
        title1: string
            The name of the first void catalog
        title2: string
            The name of the second void catalog    
        mask_file: string or None
            The path to the survey mask file
        rmin: float
            The comoving minimum distance within the survey volume
        rmax : float
            The comoving maximum distance within the survey volume
        zone_table_V1: Astropy Table or None
            The zone catalog for the first void catalog
        zone_table_V2: Astropy Table or None
            The zone catalog for the second zone catalog
        use_mask: bool
            If True, the survey mask is applied. Set to False for a simulation box
        V1_algorithm: string, either "VF" or "V2"
            Denotes whether VoidFinder or V^2 was used to derive the first void catalog
        V2_algorithm: string, either "VF" or "V2"
            Denotes whether VoidFinder or V^2 was used to derive the second void catalog
        """
        self.max_sphere_size = 0
        
        # Format V^2 catalogs for ease of use 
        if V1_algorithm=="V2":
            data_table_V1 = prep_V2_cat(data_table_V1, zone_table_V1)
        # Note the maximum sphere size for a VoidFinder catalog
        else:
            self.max_sphere_size = np.max(data_table_V1["radius"])
        
        # Likewise for the second catalog 
        if V2_algorithm=="V2":
            data_table_V2 = prep_V2_cat(data_table_V2, zone_table_V2)
        else:
            self.max_sphere_size = np.maximum(np.max(data_table_V2["radius"]), self.max_sphere_size)
        
        #save information of interest  
        self.data_table_V1 = data_table_V1
        self.data_table_V2 = data_table_V2
        self.title1 = title1
        self.title2 = title2
        self.use_mask = use_mask
        self.V1_algorithm = V1_algorithm
        self.V2_algorithm = V2_algorithm
            
        if use_mask:
            if mask_tuple is not None:
                self.mask, self.mask_resolution = mask_tuple
            else:
                temp_infile = open(mask_file, 'rb')
                self.mask, self.mask_resolution, _ = pickle.load(temp_infile)
                temp_infile.close()

            self.rmin = rmin
            self.rmax = rmax
        
    def find_overlap(self, edge_buffer):
        """Calculates the overlap of two void catalogs
        
        Parameters
        ----------
        edge_buffer: float
            The distance from the survey edge at which a point is considered and edge point
    
        """
        # Calculate the grid boundaries 
        xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(self.data_table_V1, self.data_table_V2)

        # Generate the grid
        pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Case of using a survey mask
        if self.use_mask: 
            points_in_mask = mask_point_filter(pts, 
                                               self.mask, 
                                               self.mask_resolution, 
                                               self.rmin, 
                                               self.rmax,
                                               edge_buffer)
            self.pim = points_in_mask
        # Case of a cubic volume
        else:
            points_in_mask = pts

        (var, self.n_points) = points_in_mask.shape
        
        # U denotes the number of iterations of the volume overlap algorithm to be performed,
        # with each iteration adding a different random offset to the grid points
        # U is currently set to 1 for testing purposes, but should be set higher for
        # monte carlo sampling (U is the number of random points sampled in each grid cell)
        # TODO: add U as a user input, and change the name of U to something more informative
        U = 1
        
        # Variables describing the number of points to be found withing the void of each catalog
        # for each iteration of the algorithm
        self.count_in_V1 = np.zeros(U)
        self.count_out_V1 = np.zeros(U)

        self.count_in_V2 = np.zeros(U)
        self.count_out_V2 = np.zeros(U)

        self.inside_both = np.zeros(U)
        self.inside_neither = np.zeros(U)
        self.inside_V1 = np.zeros(U)
        self.inside_V2 = np.zeros(U)

        points_in_mask_copy = points_in_mask.copy()
        
        # Create kd-trees for the void centers
        kdTree_V1 = kd_tree(self.data_table_V1)
        kdTree_V2 = kd_tree(self.data_table_V2)

        # Perform each iteration of the algorithm, with a different random offset delta applied
        # To the grid points
        for i in range(U):

            delta = np.random.rand(3)

            points_in_mask_copy[0] = points_in_mask[0] + delta[0]
            points_in_mask_copy[1] = points_in_mask[1] + delta[1]
            points_in_mask_copy[2] = points_in_mask[2] + delta[2]
            
            if self.V1_algorithm == "VF":
                true_inside_V1 = point_query_VF(points_in_mask_copy, kdTree_V1, self.data_table_V1, self.max_sphere_size)
                
            elif self.V1_algorithm == "V2":
                true_inside_V1 = point_query_V2(points_in_mask_copy, kdTree_V1, self.data_table_V1).data.T[0]
                
            # Number of points inside and outside of catalog A voids
            self.count_in_V1[i] = np.sum(true_inside_V1)
            self.count_out_V1[i] = np.sum(~true_inside_V1)

            if self.V2_algorithm == "VF":
                true_inside_V2= point_query_VF(points_in_mask_copy, kdTree_V2, self.data_table_V2, self.max_sphere_size)
            elif self.V2_algorithm == "V2":
                true_inside_V2 = point_query_V2(points_in_mask_copy, kdTree_V2, self.data_table_V2).data.T[0]
            
            # Number of points inside and outside of catalog B voids
            self.count_in_V2[i] = np.sum(true_inside_V2)
            self.count_out_V2[i] = np.sum(~true_inside_V2)

            # Number of points inside both A and B voids
            self.inside_V1_and_V2 = np.logical_and(true_inside_V1, true_inside_V2)
            self.inside_both[i] = np.sum(self.inside_V1_and_V2)

            # Number of points inide neither A nor B voids
            self.not_inside_V1_and_V2 = np.logical_and(~true_inside_V1, ~true_inside_V2)
            self.inside_neither[i] = np.sum(self.not_inside_V1_and_V2)
            
            # Number of points in A but not B voids
            self.inside_v1 = np.logical_and(true_inside_V1, ~true_inside_V2)
            self.inside_V1[i] = np.sum(self.inside_v1)

            # Number of points not in A but in B voids
            self.inside_v2 = np.logical_and(~true_inside_V1, true_inside_V2)
            self.inside_V2[i] = np.sum(self.inside_v2)
        
        # The below code is reduntant with the above, but assumes U=1
        # TODO: replace this code with something that joins all of the U trials together
        # such as self.inside_V2 = np.sum(self.inside_V2, axis=0) / U, etc.
        # (the divide by U would normalize the sum, I believe axis=0 is the correct choice, but try axis=1 if not)
        self.inside_V1_and_V2 = np.logical_and(true_inside_V1, true_inside_V2)
        self.not_inside_V1_and_V2 = np.logical_and(~true_inside_V1, ~true_inside_V2)
        self.inside_V1 = np.logical_and(true_inside_V1, ~true_inside_V2)
        self.inside_V2 = np.logical_and(~true_inside_V1, true_inside_V2)
        
        # Volume fractions
        self.r_V1 = self.count_in_V1 / self.n_points
        self.r_V2 = self.count_in_V2 / self.n_points
        self.r_V1_V2 = np.sum(self.inside_V1_and_V2) / self.n_points
        self.r_not_V1_V2 = np.sum(self.not_inside_V1_and_V2) / self.n_points
        self.r_V1_not_V2 = np.sum(self.inside_V1) / self.n_points
        self.r_V2_not_V1 = np.sum(self.inside_V2) / self.n_points
    
    def plot(self):
        """Plots the overlap of two void catalogs after running find_overlap
        
        """
        fig = plt.figure()
        plt.pie(
            [self.r_V1_V2, self.r_not_V1_V2, self.r_V2_not_V1, self.r_V1_not_V2],
            labels=[
                f"Common to {self.title1}/{self.title2} Voids",
                "Exterior to All Voids",
                f"Unique to {self.title2} Voids",
                f"Unique to {self.title1} Voids"
            ]
        )

        plt.title(f"Volume Overlap of {self.title1} and {self.title2} Voids")
        
        return fig
    
    def report(self, do_print=True, do_return = True):
        """Outputs the overlap of two void catalogs after running find_overlap
        
        """
        if do_print:
            print("Number of points used:", self.n_points)
            print(f"Common to {self.title1}/{self.title2} Voids:", self.r_V1_V2)
            print("Exterior to All Voids:", self.r_not_V1_V2)
            print(f"Unique to {self.title2} Voids:", self.r_V2_not_V1)
            print(f"Unique to {self.title1} Voids:", self.r_V1_not_V2)
        if do_return:
            return (self.n_points, 
                    self.n_points*self.r_V1_V2, 
                    self.n_points*self.r_not_V1_V2, 
                    self.n_points*self.r_V2_not_V1, 
                    self.n_points*self.r_V1_not_V2)
        
        
class SingleCalculator():
    """Compares the voulume filling fraction of one void catalog. 
    
    WARNING: This class assumes that V2 catalogs are generated 
    from the same underlying galaxy catalog. The code must be manually edtied if this isn't the case.
    """
    
    def __init__ (self, data_table_V1, title1, mask_file, rmin, rmax, zone_table_V1 = None, use_mask = True, V1_algorithm="VF",mask_tuple = None):
        """Formates a V2 catalog for use with the void overlap calclator

        Parameters
        ----------
        data_table_V1: Astropy Table
            The first void catalog
        data_table_V2: Astropy Table
            The second void catalog
        title1: string
            The name of the first void catalog
        title2: string
            The name of the second void catalog    
        mask_file: string or None
            The path to the survey mask file
        rmin: float
            The comoving minimum distance within the survey volume
        rmax : float
            The comoving maximum distance within the survey volume
        zone_table_V1: Astropy Table or None
            The zone catalog for the first void catalog
        zone_table_V2: Astropy Table or None
            The zone catalog for the second zone catalog
        use_mask: bool
            If True, the survey mask is applied. Set to False for a simulation box
        V1_algorithm: string, either "VF" or "V2"
            Denotes whether VoidFinder or V^2 was used to derive the first void catalog
        V2_algorithm: string, either "VF" or "V2"
            Denotes whether VoidFinder or V^2 was used to derive the second void catalog
        """
        self.max_sphere_size = 0
        
        # Format V^2 catalogs for ease of use 
        if V1_algorithm=="V2":
            data_table_V1 = prep_V2_cat(data_table_V1, zone_table_V1)
        # Note the maximum shpehre size for a VoidFinder catalog
        else:
            self.max_sphere_size = np.max(data_table_V1["radius"])
        
        
        #save information of interest  
        self.data_table_V1 = data_table_V1
        self.title1 = title1
        self.use_mask = use_mask
        self.V1_algorithm = V1_algorithm

        if use_mask:
            if mask_tuple is not None:
                self.mask, self.mask_resolution = mask_tuple
            else:
                temp_infile = open(mask_file, 'rb')
                self.mask, self.mask_resolution, _ = pickle.load(temp_infile)
                temp_infile.close()

            self.rmin = rmin
            self.rmax = rmax
        
    def find_overlap(self, edge_buffer):
        """Calculates the overlap of one void catalog
        
        Parameters
        ----------
        edge_buffer: float
            The distance from the survey edge at which a point is considered and edge point

        """
        # Calculate the grid boundaries 
        xmin, xmax, ymin, ymax, zmin, zmax = calc_volume_boundaries(self.data_table_V1, self.data_table_V1)

        # Generate the grid
        pts = generate_grid_points(xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Case of using a survey mask
        if self.use_mask: 
            points_in_mask = mask_point_filter(pts, 
                                               self.mask, 
                                               self.mask_resolution, 
                                               self.rmin, 
                                               self.rmax,
                                               edge_buffer)
            self.pim = points_in_mask
        # Case of a cubic volume
        else:
            points_in_mask = pts

        (var, self.n_points) = points_in_mask.shape
        
        # U denotes the number of iterations of the volume overlap algorithm to be performed,
        # with each iteration adding a different random offset to the grid points
        # U is currently set to 1 for testing purposes, but should be set higher for
        # monte carlo sampling (U is the number of random points sampled in each grid cell)
        # TODO: add U as a user input, and change the name of U to something more informative
        U = 1
        
        # Variables describing the number of points to be found withing the void of each catalog
        # for each iteration of the algorithm
        self.count_in_V1 = np.zeros(U)
        self.count_out_V1 = np.zeros(U)



        self.inside_V1 = np.zeros(U)

        points_in_mask_copy = points_in_mask.copy()
        
        # Create kd-trees for the void centers
        kdTree_V1 = kd_tree(self.data_table_V1)
        
        # Perform each iteration of the algorithm, with a different random offset delta applied
        # To the grid points
        for i in range(U):

            delta = np.random.rand(3)

            points_in_mask_copy[0] = points_in_mask[0] + delta[0]
            points_in_mask_copy[1] = points_in_mask[1] + delta[1]
            points_in_mask_copy[2] = points_in_mask[2] + delta[2]
            
            if self.V1_algorithm == "VF":
                true_inside_V1 = point_query_VF(points_in_mask_copy, kdTree_V1, self.data_table_V1, self.max_sphere_size)
                
            elif self.V1_algorithm == "V2":
                true_inside_V1 = point_query_V2(points_in_mask_copy, kdTree_V1, self.data_table_V1).data.T[0]
                
            # Number of points inside and outside of catalog A voids
            self.count_in_V1[i] = np.sum(true_inside_V1)
            self.count_out_V1[i] = np.sum(~true_inside_V1)


    
        
        # Volume fractions TODO: replace 0 with something that is actually correct for the multiple loop case
        self.r_V1 = self.count_in_V1[0] / self.n_points
        self.r_not_V1 = self.count_out_V1[0] / self.n_points
    def plot(self):
        """Plots the overlap of two void catalogs after running find_overlap
        
        """
        fig = plt.figure()
        plt.pie(
            [self.r_V1, self.r_not_V1],
            labels=[
                "void",
                "wall",
            ]
        )

        plt.title(f"Volume Overlap of {self.title1} Voids")
        
        return fig
    
    def report(self,do_print=True, do_return=True):
        """Outputs the overlap of two void catalogs after running find_overlap
        
        """
        if do_print:
            print("Number of points used:", self.n_points)
            print("void:", self.r_V1)
            print("wall:", self.r_not_V1)
            
        if do_return:
            return self.n_points, self.r_V1, self.r_not_V1