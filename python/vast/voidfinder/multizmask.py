from astropy.io import fits
from astropy.table import Table

import numpy as np
import time
import healpy as hp

from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder.constants import c #speed of light

maskra = 360
maskdec = 180
dec_offset = -90

D2R = np.pi/180.0


def generate_mask(gal_data, 
                  z_max, 
                  dist_metric='comoving',
                  smooth_mask=True,
                  min_maximal_radius=10.0,
                  Omega_M=0.3,
                  h=1.0):
    """
    This function creates a grid of shape (N,M) where the N dimension represents 
    increments of the ra space (0 to 360 degrees) and the M dimension represents 
    increments in the dec space (0 to 180 degrees).  The value of the mask is a 
    boolean representing whether or not that (ra,dec) position is part of the 
    survey, or outside the survey.  For example, if mask[320,17] == True, that 
    indicates the right ascension of 320 degrees and declination of 17 degrees 
    is within the survey.
    
    Note that this mask will be for the ra-dec (right ascension, declination) 
    space of the survey, the radial min/max limits will be need to be checked 
    separately.  
    

    Parameters
    ==========

    gal_data : astropy table
        Table of all galaxies in sample
        Ra and Dec must be given in degrees
        Ra can be in either -180 to 180 or 0 to 360 format
        Dec must be in -90 to 90 format since the code below subtracts
        90 degrees to go to 0 to 180 format

    z_max : float
        Maximum redshift of the volume-limited catalog.

    dist_metric : string
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    smooth_mask : boolean
        If smooth_mask is set to True (default), small holes in the mask (single 
        cells without any galaxy in them that are surrounded by at least 3 cells 
        which have galaxies in them) are unmasked.

    min_maximal_radius : float
        Minimum radius of the maximal spheres.  Default is 10 Mpc/h.  The mask 
        resolution depends on this value.

    Omega_M : float
        Cosmological matter normalized energy density.  Default is 0.3.

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).


    Returns
    =======

    mask : numpy array of shape (N,M)
        Boolean array of the entire sky, with points within the survey limits 
        set to True.  N represents the incremental RA; M represents the 
        incremental dec.

    mask_resolution : integer
        Scale factor of coordinates in maskfile
    """

    print("Generating mask", flush=True)
    
    ############################################################################
    # First, extract the ra (Right Ascension) and dec (Declination) coordinates
    # of our galaxies from the astropy table.  Make a big (N,2) numpy array
    # where each row of the array is the (ra, dec) pair corresponding to a 
    # galaxy in the survey.
    #
    # Also make sure the ra values are in the range [0,360) instead of 
    # [-180, 180)
    #---------------------------------------------------------------------------
    ra  = gal_data['ra'].data % 360.0
    
    dec = gal_data['dec'].data
    
    if dist_metric == 'comoving':
        r_max = z_to_comoving_dist(np.array([z_max], dtype=np.float32), 
                                   Omega_M, 
                                   h)
    else:
        r_max = c*z_max/(100*h)
    
    num_galaxies = ra.shape[0]

    #ang = np.array(list(zip(ra,dec)))
    #this is almost 70x faster than list(zip())
    ang = np.concatenate((ra.reshape(num_galaxies, 1), 
                          dec.reshape(num_galaxies,1)), 
                         axis=1)
    ############################################################################



    ############################################################################
    # Next, we need to calculate the "resolution" of our mask.  Depending on how 
    # deep the survey goes, deeper surveys will require higher angular
    # resolutions in order to appropriately account for galaxy positions in the 
    # ra and dec space, so depending on the radial depth of the survey, 
    # calculate an integer scale factor "mask_resolution" to increase the number 
    # of ra-dec positions in our mask.
    #
    # A "mask_resolution" of 1 indicates our mask will span the range [0,360)
    # and [0,180) in steps of 1.  A mask_resolution of 2, means we span [0,360) 
    # and [0,180) in steps of 0.5, so our actual mask will have a shape of 
    # (360x2, 180x2).  Similarly for a mask_resolution of N, the mask will be 
    # shape (360xN, 180xN), in increments of 1/N degrees
    #
    # Mask resolution (inverse of the angular radius of the minimum void at the 
    # maximum distance).
    #---------------------------------------------------------------------------
    mask_resolution = 1 + int(D2R*r_max/min_maximal_radius) # scalar value despite value of r_max
    ############################################################################

    
    
    ############################################################################
    # Now that we know the mask_resolution scale factor, convert our ra-dec
    # coordinates into this scaled space.  Each coordinate in the scaled space
    # will fall into an integer bucket, so we use .astype(int), and that value
    # will represent that the mask[ra_scaled, dec_scaled] should be set to True.
    #
    #---------------------------------------------------------------------------
    scaled_converted_ang = (mask_resolution*ang).astype(int)
    
    ############################################################################



    ############################################################################
    # Now we create a healpix based boolean pre-mask with approximately the same
    # number of bins as in the mask: (360*N)*(180*N). The actual dimensionality 
    # of the pre-mask is the closest squared multiple of 12, as required by the
    # healpix formalism. This multiple, termed as NSIDE in the healpix
    # algorithm, is used with healpix to select and mark bins in the pre-mask 
    # that contain galaxies.
    #
    # We optionally smooth the premask to reduce patchiness.
    #  
    #---------------------------------------------------------------------------
    
    num_px = maskra * maskdec * mask_resolution ** 2
    hpscale=1
    nside = int(hpscale*np.sqrt(num_px / 12)) #test scale by 4
    healpix_mask = np.zeros(hp.nside2npix(nside), dtype = bool)
    galaxy_pixels = hp.ang2pix(nside, ra, dec, lonlat = True)
    galaxy_pixels = np.unique(galaxy_pixels, axis=0)
    healpix_mask[galaxy_pixels] = 1

    print(num_px)
    print(hp.nside2npix(nside))
    
    if smooth_mask:
        
        neighbors = hp.get_all_neighbours(nside,np.arange(len(healpix_mask)))[::2]
        correct_idxs = np.sum(healpix_mask[neighbors], axis=0)
        healpix_mask[np.where(correct_idxs >= 3)] = 1

        """
        correct_idxs = []
        
        for (i, curr_val) in enumerate(healpix_mask):
            
            neighbors=hp.get_all_neighbours(nside,i)[::2]
            
            if curr_val == 0 and np.sum(healpix_mask[neighbors]) >= 3:
                correct_idxs.append(i)
            
        healpix_mask[correct_idxs] = 1
        """
    
    ############################################################################



    ############################################################################
    # Now we create the actual boolean mask by allocating an array of shape
    # (360*N, 180*N), and iterating through all ra-dec positions on this grid,
    # and using the corresponding location on pre-mask to determine if the bin
    # is in the mask.
    #
    # Since declination is actually measured from the equator, we need to 
    # subtract 90 degrees from the mask index in order to convert from 
    # [0,180) space into [-90,90) space.
    #---------------------------------------------------------------------------

    maskY, maskX = np.meshgrid(
        np.arange(mask_resolution * 180)/mask_resolution + dec_offset,
        np.arange(mask_resolution * 360)/mask_resolution
        )
    grid_pixels = hp.ang2pix(nside, maskX, maskY, lonlat = True)
    mask = healpix_mask[grid_pixels]

    """
    mask = np.zeros((mask_resolution*maskra, mask_resolution*maskdec), 
                    dtype=bool) 
    
    for i in range(0, mask_resolution * 360):

        for j in range (0, mask_resolution * 180):

            pxid = hp.ang2pix(nside, float(i) / mask_resolution,
                              float(j) / mask_resolution + dec_offset, 
                              lonlat = True) 

            mask[i,j] = healpix_mask[pxid]"""
    
    if smooth_mask:

        padding=np.expand_dims(np.ones(mask.shape[0]+2),axis=1)
        neighbors=np.hstack((padding,
                            np.vstack((mask[-1],mask,mask[0])),
                            padding))
        neighbor_sum = neighbors[:-2,1:-1] + neighbors[2:,1:-1] + neighbors[1:-1,:-2] + neighbors[1:-1,2:]
        mask[np.where(neighbor_sum>=3)] = 1
                        
        
        """
        correct_idxs = []
    
        for idx in range(mask.shape[0]):
            
            for jdx in range(mask.shape[1]):
                
                if idx < 1 or jdx < 1 or idx > mask.shape[0] - 2 or jdx > mask.shape[1] - 2:
                    continue
                
                curr_val = mask[idx, jdx]
                
                neigh1 = int(mask[idx-1,jdx])
                neigh2 = int(mask[idx+1,jdx])
                neigh3 = int(mask[idx,jdx-1])
                neigh4 = int(mask[idx,jdx+1])
                
                if curr_val == 0 and neigh1+neigh2+neigh3+neigh4 >= 3:
                    correct_idxs.append((idx,jdx))
        
        for (idx, jdx) in correct_idxs:
            
            mask[idx,jdx] = 1"""
    ############################################################################

    return mask, mask_resolution





    
    



def build_mask(maskfile, mask_resolution):
    '''
    Build the survey mask.  Assumes the coordinates in maskfile have already 
    been scaled to highest resolution necessary.


    Parameters:
    ===========

    maskfile : numpy array of shape (2,n)
        n pairs of RA,dec coordinates that are within the survey limits and are 
        scaled by the mask_resolution.  Oth row is RA; 1st row is dec.

    mask_resolution : integer
        Scale factor of coordinates in maskfile


    Returns:
    ========

    mask : numpy array of shape (N,M)
        Boolean array of the entire sky, with points within the survey limits 
        set to True.  N represents the incremental RA; M represents the 
        incremental dec.
    '''

    '''
    mask = []
    
    for i in range(1, 1+len(maskfile)):
        
        mask.append(np.zeros((i*maskra, i*maskdec), dtype=bool))
        
        for j in range(len(maskfile[i-1][0])):
            
            mask[i-1][maskfile[i-1][0][j]][maskfile[i-1][1][j]-i*dec_offset] = True
            
    mask = np.array(mask)
    '''

    mask = np.zeros((mask_resolution*maskra, mask_resolution*maskdec), dtype=bool)

    for j in range(len(maskfile[0])):

        mask[ maskfile[0,j], maskfile[1,j] - mask_resolution*dec_offset] = True

    return mask





