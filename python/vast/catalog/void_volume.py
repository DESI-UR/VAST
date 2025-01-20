import numpy as np

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
    n, d = x.shape
    
    # Compute the corners of the bounding parallelpiped containing
    # the group of spheres. Then store the volume.
    xmin = np.copy(x[-1])
    xmax = np.copy(xmin)

    for i in range(n):
        xmin = np.minimum(xmin, x[i] - R[i])
        xmax = np.maximum(xmax, x[i] + R[i])
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
    n, d = x.shape
    R2 = R**2
    
    vol, xmin, xmax = bounding_volume(x, R)

    # Track unions and joint intersections per spherical volume.
    n_inter, n_union = 0, 0
    
    for iobs in range(nsamples):
        obsd = np.random.uniform(low=xmin, high=xmax)

        ioint = True
        uoint = False

        for i in range(n):
            z2 = np.sum((obsd - x[i])**2)
            internal = z2 <= R2[i]
            
            # Track union and intersection. Note that intersection
            # looks for the intersecting regions of *all n* spheres,
            # not pairs of spheres.
            uoint = uoint or internal
            ioint = ioint and internal

        if ioint:
            n_inter += 1
        if uoint:
            n_union += 1

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

