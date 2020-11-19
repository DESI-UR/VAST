from astropy.io import fits
from astropy.table import Table

import numpy as np
import time

#from .absmag_comovingdist_functions import Distance

from voidfinder.constants import c #speed of light

maskra = 360
maskdec = 180
dec_offset = -90


def generate_mask(gal_data, 
                  #dist_metric='comoving',
                  smooth_mask=True,
                  h=1.0, 
                  O_m=0.3,
                  verbose=0):
    '''
    Description
    ===========
    
    This function creates a grid of shape (N,M) where the N dimension represents 
    increments of the ra space (0 to 360 degrees) and the M dimension represents 
    increments in the dec space (0 to 180 degrees).  The value of the mask is a boolean 
    representing whether or not that (ra,dec) position is part of the survey, or 
    outside the survey.  For example, if mask[320,17] == True, that indicates 
    the right ascension of 320 degrees and declination of 17 degrees is within the survey.
    
    Note that this mask will be for the ra-dec (right ascension, declination) space
    of the survey, the radial min/max limits will be need to be checked separately.  
    

    Parameters:
    ===========

    galaxy_data : astropy table
        Table of all galaxies in sample
        Ra and Dec must be given in degrees
        Ra can be in either -180 to 180 or 0 to 360 format
        Dec must be in -90 to 90 format since the code below subtracts
        90 degrees to go to 0 to 180 format

    UNUSED? dist_metric : string
        Distance metric to use in calculations.  Options are 'comoving' 
        (default; distance dependent on cosmology) and 'redshift' (distance 
        independent of cosmology).

    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).

    O_m : float
        Omega-matter.  Default value is 0.3.


    Returns:
    ========

    mask : numpy array of shape (N,M)
        Boolean array of the entire sky, with points within the survey limits 
        set to True.  N represents the incremental RA; M represents the 
        incremental dec.

    mask_resolution : integer
        Scale factor of coordinates in maskfile
    '''

    if verbose > 0:
        
        print("Generate mask start", flush=True)

    D2R = np.pi/180.0
    
    ###########################################################################
    # First, extract the ra (Right Ascension) and dec (Declination) coordinates
    # of our galaxies from the astropy table.  Make a big (N,2) numpy array
    # where each row of the array is the (ra, dec) pair corresponding to
    # a galaxy in the survey.
    #
    # Also make sure the ra values are in the range [0,360) instead of
    # [-180, 180)
    ###########################################################################

    ra  = gal_data['ra'].data % 360.0
    
    dec = gal_data['dec'].data 
    
    r = gal_data["Rgal"].data 
    
    num_galaxies = ra.shape[0]

    #ang = np.array(list(zip(ra,dec)))
    #this is almost 70x faster than list(zip())
    ang = np.concatenate((ra.reshape(num_galaxies, 1), dec.reshape(num_galaxies,1)), axis=1)



    ###########################################################################
    # Next, we need to calculate the "resolution" of our mask.  Depending on
    # how deep the survey goes, deeper surveys will require higher angular
    # resolutions in order to appropriately account for galaxy positions in
    # the ra and dec space, so depending on the radial depth of the survey, 
    # calculate an integer scale factor "mask_resolution" to increase the number 
    # of ra-dec positions in our mask.
    #
    # A "mask_resolution" of 1 indicates our mask will span the range [0,360)
    # and [0,180) in steps of 1.  A mask_resolution of 2, means we span
    # [0,360) and [0,180) in steps of 0.5, so our actual mask will have
    # a shape of (360x2, 180x2).  Similarly for a mask_resolution of N,
    # the mask will be shape (360xN, 180xN), in increments of 1/N degrees
    #
    # Mask resolution (inverse of the angular radius of the minimum void at the 
    # maximum distance)
    ###########################################################################
    
    mask_resolution = 1 + int(D2R*np.amax(r)/10) #scalar value despite use of amax
    
    ###########################################################################
    # Now that we know the mask_resolution scale factor, convert our ra-dec
    # coordinates into this scaled space.  Each coordinate in the scaled space
    # will fall into an integer bucket, so we use .astype(int), and that value
    # will represent that the mask[ra_scaled, dec_scaled] should be set to
    # True
    #
    # We take the unique rows of the scaled coordinates, since as we converted
    # to integers, many ra-dec pairs will fall into the same integer bucket
    # and we only need to set that integer bucket to "True" once.
    ###########################################################################
    scaled_converted_ang = (mask_resolution*ang).astype(int)
    
    pre_mask = np.unique(scaled_converted_ang, axis=0)
    
    print(pre_mask)
    ###########################################################################
    # Now we create the actual boolean mask by allocating an array of shape
    # (360*N, 180*N), and iterating through all the unique galaxy ra-dec
    # integer buckets and setting those locations to True to represent they
    # are valid locations
    #
    # Since declination is actually measured from the equator, we need to
    # subtract 90 degrees from the scaled value in order to convert from
    # [-90,90) space into [0,180} space.
    ###########################################################################
    mask = np.zeros((mask_resolution*maskra, mask_resolution*maskdec), dtype=bool)
    print(len(mask))

    #for j in range(len(pre_mask[0])):

    #    mask[ maskfile[0,j], maskfile[1,j] - mask_resolution*dec_offset] = True
    
    for row in pre_mask:
        print(row[0], row[1] - mask_resolution*dec_offset) 
        mask[row[0], row[1] - mask_resolution*dec_offset] = True
        
        
    
    if smooth_mask:
        
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
            
            mask[idx,jdx] = 1

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





