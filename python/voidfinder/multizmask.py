from astropy.io import fits
from astropy.table import Table

import numpy as np

from .absmag_comovingdist_functions import Distance


def generate_mask(gal_data, dist_metric, h=1, O_m=0.3):
    '''
    Generate sky mask that identifies the footprint of the input galaxy survey.


    Parameters:
    ===========

    galaxy_data : astropy table
        Table of all galaxies in sample

    dist_metric : string
        Distance metric to use in calculations.  Options are 'comoving' 
        (distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).

    O_m : float
        Omega-matter.  Default value is 0.3.


    Returns:
    ========

    mask : numpy array of shape (2,n)
        n pairs of RA,dec coordinates that are within the survey limits and 
        are scaled by the mask_resolution.  Oth row is RA; 1st row is dec.

    mask_resolution : integer
        Scale factor of coordinates in maskfile
    '''


    D2R = np.pi/180

    c = 3e5

    ra  = gal_data['ra']%360
    dec = gal_data['dec']

    if dist_metric == 'comoving':
        r = Distance(gal_data['redshift'], O_m, h)
    else:
        H0 = 100*h
        r = c*gal_data['redshift']/H0

    ang = np.array(list(zip(ra,dec)))


    '''
    ###########################################################################
    # Build variable resolution mask
    #--------------------------------------------------------------------------
    nmax = 1 + int(D2R*np.amax(r)/10.)

    mask = []

    for i in range(1,nmax+1):
        mask.append(list(zip(*(np.unique((i*ang).astype(int),axis=0)))))

    mask = np.array(mask)
    np.save(args.output,mask)
    ###########################################################################
    '''


    ###########################################################################
    # Build highest resolution mask necessary for survey
    #--------------------------------------------------------------------------

    # Mask resolution (inverse of the angular radius of the minimum void at the 
    # maximum distance)
    mask_resolution = 1 + int(D2R*np.amax(r)/10)

    # Scale all coordinates by mask_resolution
    mask = list(zip(*(np.unique((mask_resolution*ang).astype(int), axis=0))))

    # Convert to numpy array
    mask = np.array(mask)

    '''
    # Save scaled survey mask coordinates and mask resolution
    #outfile = open(args.output, 'wb')
    outfile = open(mask_filename, 'wb')
    pickle.dump((mask_resolution, mask), outfile)
    outfile.close()
    '''
    ###########################################################################


    return mask, mask_resolution

