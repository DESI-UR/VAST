import numpy as np
from scipy.spatial.distance import cdist

"""
This code has been adopted from the following individuals: Kelly Douglass

The documentation format has been altered by Hernan Rincon
"""


def bounding_volume(x, R):
    """
    
    Returns the volume and corners of a parallelpiped containing the
    N input spheres of interest.
    
    params:
    ---------------------------------------------------------------------------------------------
    x (numpy array of floats of shape (N, 3)): Centers of N input spheres. Each row contains the
        following: (x coordinate, y coordinate, z coordinate)
    
    R (numpy array of floats of shape N): Radii of N input spheres.
    
    returns:
    ---------------------------------------------------------------------------------------------
    vol (float): Volume of parallelpiped.
    
    xmin (numpy array of floats of shape 3): Lower corner of volume.
    
    xmax (numpy array of floats of shape 3): Upper corner of volume.
    """  
    
    # Compute the corners of the bounding parallelpiped containing
    # the group of spheres. Then store the volume.

    xmax = np.max(x.T + R, axis=1)

    xmin = np.min(x.T - R, axis=1)
    
    vol = np.prod(xmax - xmin)
    
    return vol, xmin, xmax

def volume_of_spheres(x, R, nsamples=10000):
    """
    
    Obtains the volume, with uncertainties, of the intersection and union of
    N spherical volumes using Monte Carlo sampling.
        
    params:
    ---------------------------------------------------------------------------------------------
    x (numpy array of floats of shape (N, 3)): Centers of N input spheres. Each row contains the
        following: (x coordinate, y coordinate, z coordinate)
    
    R (numpy array of floats of shape N): Radii of N input spheres.
    
    nsamples (int): Number of Monte Carlo samples to generate.
    
    returns:
    ---------------------------------------------------------------------------------------------
    ivol (float): Volume of intersecting regions of *all* spheres.
    
    idv (float): Uncertainty in intersection volume due to Monte Carlo shot noise.
    
    uvol (float): Volume of union of spheres.
    
    udv (float): Uncertainty in union volume.
    """
    vol, xmin, xmax = bounding_volume(x, R)


    obsd = np.random.uniform(low=xmin, high=xmax, size=(nsamples, 3))

    dist = cdist(obsd, x, metric='euclidean')

    points_in_holes = dist <= R

    # Track union and intersection. Note that intersection
    # looks for the intersecting regions of *all n* spheres,
    # not pairs of spheres.
    
    n_inter = np.count_nonzero(np.all(points_in_holes, axis=1))
    
    n_union = np.count_nonzero(np.any(points_in_holes, axis=1))


    # Calculate intersecting volume and accuracy.
    # Based on binomial probability of point inside intersection.
    izp = n_inter / nsamples
    izq = (nsamples - n_inter) / nsamples

    ivol = vol * izp
    isigma = np.sqrt(izp * izq / nsamples)
    idv = vol * isigma

    # Calculate union volume and accuracy.
    # Based on binomial probability of point inside union.
    uzp = n_union / nsamples
    uzq = (nsamples - n_union) / nsamples

    uvol = vol * uzp
    usigma = np.sqrt(uzp * uzq / nsamples)
    udv = vol * usigma

    return ivol, idv, uvol, udv

