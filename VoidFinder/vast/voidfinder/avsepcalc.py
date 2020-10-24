

"""
Function to calculate nearest neighbor separation in Python VoidFinder
"""

from sklearn import neighbors

import numpy as np

import time

from .table_functions import to_array


def av_sep_calc(coords_xyz, n):
    """
    Description
    ===========
    
    Find the average distance to 3rd nearest neighbor and some related properties.
    
    Note: Sklearn KDTree implementation outperformed scipy KDTree by a ridiculous
          factor on sim dataset of 37 million galaxies
    
    Parameters
    ==========
    
    coords_xyz : numpy.ndarray of shape (N, 3)
        Cartesian coordinates of the relevant galaxies in xyz space
        
    n : int
        nth nearest neighbor to return distances for (i.e. for 3rd nearest
        neighbor separation n=3)
        
    
    Returns
    =======
    
    all_nth_distances : numpy.ndarray of shape (N,)
        all distances to the nth nearest neighbor

    """
    
    #avsep_start = time.time()

    #gal_array = to_array(GALTABLEXYZ)
    
    galaxy_tree = neighbors.KDTree(coords_xyz)

    ############################################################################
    # Because the galaxies are themselves contained in gal_array, that means
    # the output 'distances[:,0]' will be the input galaxy indices, since
    # their distances from themselves is 0.  So the query output columns look 
    # like:
    #
    # 0     1          2          3         ...
    # self  1st neigh  2nd neigh  3rd neigh ...
    #
    # So for example if we want the 3rd nearest neighbor distances, we use as 
    # input to the KDtree 'k=n+1', and our result will be at column index 'n'
    #---------------------------------------------------------------------------
    distances, indices = galaxy_tree.query(coords_xyz, k=n+1)
    
    all_nth_distances = distances[:,n]

    #avg = np.mean(all_3rd_distances)

    #sd = np.std(all_3rd_distances)

    #dist_lim = avg + 1.5*(sd)
    ############################################################################
    

    #return dist_lim, avg, sd, all_3rd_distances
    
    return all_nth_distances
