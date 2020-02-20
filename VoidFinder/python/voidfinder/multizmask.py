from astropy.io import fits
from astropy.table import Table

import numpy as np
import time

#from .absmag_comovingdist_functions import Distance


maskra = 360
maskdec = 180
dec_offset = -90


def generate_mask(gal_data, dist_metric='comoving', h=1.0, O_m=0.3):
    '''
    Generate sky mask that identifies the footprint of the input galaxy survey.


    Parameters:
    ===========

    galaxy_data : astropy table
        Table of all galaxies in sample

    dist_metric : string
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

    mask : numpy array of shape (2,n)
        n pairs of RA,dec coordinates that are within the survey limits and 
        are scaled by the mask_resolution.  Oth row is RA; 1st row is dec.

    mask_resolution : integer
        Scale factor of coordinates in maskfile
    '''


    print("Generate mask start")

    D2R = np.pi/180.0

    c = 299792.0

    ra  = gal_data['ra'].data % 360
    dec = gal_data['dec'].data 
    r = gal_data["Rgal"].data 
    
    num_galaxies = ra.shape[0]

    #ang = np.array(list(zip(ra,dec)))
    #this is almost 70x faster than list(zip())
    ang = np.concatenate((ra.reshape(num_galaxies, 1), dec.reshape(num_galaxies,1)), axis=1)


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
    
    mask_resolution = 1 + int(D2R*np.amax(r)/10) #scalar value despite use of amax
    

    # Scale all coordinates by mask_resolution
    
    
    #start_time = time.time()
    
    scaled_converted_ang = (mask_resolution*ang).astype(int)
    
    pre_mask = np.unique(scaled_converted_ang, axis=0).T
    
    #print("Numpy only time: ", time.time() - start_time)
    
    #print(unique_vals.dtype, unique_vals.shape)
    #print(unique_vals[0:10])
    
    #start_time = time.time()
    
    #mask = list(zip(*(np.unique((mask_resolution*ang).astype(int), axis=0))))

    # Convert to numpy array
    #mask = np.array(mask)
    
    #print("List zip time: ", time.time() - start_time)
    
    #print(unique_vals.dtype, unique_vals.shape, mask.dtype, mask.shape)
    #print(unique_vals[0:10])
    #print(mask[0:10])
    
    #print(np.all(mask == unique_vals))
    
    mask = build_mask(pre_mask, mask_resolution)
    

    '''
    # Save scaled survey mask coordinates and mask resolution
    #outfile = open(args.output, 'wb')
    outfile = open(mask_filename, 'wb')
    pickle.dump((mask_resolution, mask), outfile)
    outfile.close()
    '''
    ###########################################################################


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





