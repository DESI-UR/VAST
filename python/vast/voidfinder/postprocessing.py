
import numpy as np
import healpy as hp

import pickle

import warnings

from astropy.table import Table
from astropy.io import ascii, fits

from vast.voidfinder.distance import z_to_comoving_dist

import time

import h5py

from vast.voidfinder.constants import c #speed of light

import os


def file_postprocess(maximal_spheres_filename,
                    void_table_filename, 
                    galaxies_filename,
                    mask_filename,
                    wall_field_filename,
                    in_directory,
                    out_directory, 
                    survey_name,
                    mag_cut,
                    magnitude_limit,
                    rm_isolated,
                    dist_metric, 
                    dist_limits,
                    min_z,
                    max_z,
                    Omega_M,
                    h,
                    verbose=0):
    '''
    Set up output file names, calculate distances, etc.
    
    
    PARAMETERS
    ==========
    
    maximal_spheres_filename : string
        Location of saved maximal spheres file 
    
    void_table_filename : string
        Location of saved void table 
    
    galaxies_filename : string
        File name of galaxy catalog.  Should be readable by 
        astropy.table.Table.read as a ascii.commented_header file.  Required 
        columns include 'ra', 'dec', 'z', and absolute magnitude (either 
        'rabsmag' or 'magnitude'.

    mask_filename : string
        File name of survey mask. Should be a pickle file that reads in 
        (mask : (n,m) numpy array, mask_resolution: int, dist_limits: 
        2-element list of floats)

    wall_field_filename : string or None
        File name of the galaxy catalog split into wall and field
        galaxies. Should be a pickle file that reads in
        (wall_coords_xyz: (n,3) numpy array, field_coords_xyz: (n,3) numpy 
        array). If set to None, separate wall and field files are used 
        instead
        
    in_directory : string
        Directory path for input files
    
    out_directory : string
        Directory path for output files

    survey_name : string
        Name of the galaxy catalog, string value to prepend or append to output 
        names
        
    mag_cut : boolean
        Determines whether or not to implement a magnitude cut on the galaxy 
        survey.  Default is True (remove all galaxies fainter than Mr = -20).
    
    magnitude_limit : float
        value at which to perform magnitude cut
        
    rm_isolated : boolean
        Determines whether or not to remove isolated galaxies (defined as those 
        with the distance to their third nearest neighbor greater than the sum 
        of the average third-nearest-neighbor distance and 1.5 times the 
        standard deviation of the third-nearest-neighbor distances).
    
    dist_metric : string
        Description of which distance metric to use.  Options should include 
        'comoving' (default) and 'redshift'.
    
    dist_limits : list of length 2
        [Minimum distance, maximum distance] of galaxy sample (in units of 
        Mpc/h)  

    min_z, max_z : float
        Minimum and maximum redshift range for the survey mask.  Default values 
        are None (determined from galaxy extent).
        
    Omega_M : float
        Value of the matter density of the given cosmology.  Default is 0.3.
        
    h : float
        Value of the Hubble constant.  Default is 1 (so all distances will be in 
        units of h^-1).
    
    verbose : int
        values greater than zero indicate to print output
    
    RETURNS
    =======
    
    galaxy_data_table : astropy table
        Table of all galaxies in catalog.
        
    dist_limits : numpy array of shape (2,)
        Minimum and maximum distances to use for void search.  Units are Mpc/h, 
        in either comoving or redshift coordinates (depending on dist_metric).
        
    out1_filename : string
        File name of maximal sphere output file.
        
    out2_filename : string
        File name of all void holes
    
    '''
    
    # fits output file name
    log_filename = out_directory + survey_name +'VoidFinder_Output.fits'

    # name of file containing input galaxies
    in_filename = in_directory + galaxies_filename
    
    # names of files assembled into the output
    wall_filename = out_directory + survey_name + 'wall_gal_file.txt'
    field_filename = out_directory + survey_name + 'field_gal_file.txt'
    neighbor_filename = out_directory + survey_name + 'neighbor_cut.pickle'

    ############################################################################
    # Helper functions for fits file assembly
    #---------------------------------------------------------------------------

    # Save general voidfinding information to a HDU
    def write_header():

        hdr = fits.Header()

        hdr['input_galaxy_table'] = in_filename
        hdr['dist_metric'] = dist_metric
        hdr['dist_limit_Mpc_min'] = dist_limits[0]
        hdr['dist_limit_Mpc_max'] = dist_limits[1]
        hdr['redshift_limit_min'] = min_z
        hdr['redshift_limit_max'] = max_z
        hdr['matter_density'] = Omega_M
        hdr['hubble_parameter'] = h

        # criteria for separating wall and field galaxies
        if rm_isolated: 
            temp_infile = open(neighbor_filename, 'rb')
            l, sd = pickle.load(temp_infile)
            temp_infile.close()
            hdr['3rd_neighbor_separation_av'] = l
            hdr['3rd_neighbor_separation_sd'] = sd
        
        # absolute magnitude cut info
        hdr['mag_cut']=mag_cut
        if mag_cut:
                hdr['magnitude_limit']=magnitude_limit
        
        hdu = fits.PrimaryHDU(header=hdr)

        return hdu


    # Save mask to a HDU
    def write_mask():

        temp_infile = open(mask_filename, 'rb')
        mask, mask_resolution, dist_limits = pickle.load(temp_infile)
        temp_infile.close()

        hdu = fits.ImageHDU(mask.astype(int))
        hdr = hdu.header
        hdr['mask_resolution'] = mask_resolution
        hdr['mask_dist_limit_Mpc_min'] = dist_limits[0]
        hdr['mask_dist_limit_Mpc_max'] = dist_limits[1]

        # caclulate solid angle sky coverage
        nside = round(np.sqrt(mask.size / 12))
        pix_cells = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat = True) #healpix array containing corresponding ra-dec coords in radians
        pix_cells = (np.floor(pix_cells[0]).astype(int), np.floor(pix_cells[1]+90).astype(int))
        healpix_mask = mask[pix_cells]
        coverage = np.sum(healpix_mask) * hp.nside2pixarea(nside) #solid angle coverage in steradians
        hdr['steradian_sky_coverage'] = coverage

        return hdu

    # Save wall and field galaxy tables to HDUs
    def write_galaxies():

        wallHDU = fits.BinTableHDU()
        fieldHDU = fits.BinTableHDU()

        # Form header
        wallHDU.header['num_wall_galaxies'] = 0
        fieldHDU.header['num_field_galaxies'] = 0

        # Read in wall and field galaxies (may be stored in several differnet file configurations)
        if wall_field_filename is not None and os.path.isfile(wall_field_filename):

            temp_infile = open(wall_field_filename, 'rb')
            wall_coords_xyz, field_coords_xyz = pickle.load(temp_infile)
            temp_infile.close()   

            wall_xyz_table = Table(data=wall_coords_xyz, names=["x", "y", "z"])
            field_xyz_table = Table(data=field_coords_xyz, names=["x", "y", "z"])
            wallHDU.data = fits.BinTableHDU(wall_xyz_table).data
            fieldHDU.data = fits.BinTableHDU(field_xyz_table).data
            wallHDU.header['num_wall_galaxies'] = len(wall_coords_xyz)
            fieldHDU.header['num_field_galaxies'] = len(field_coords_xyz)

        else:
            
            wall_coords_xyz = Table.read(wall_filename,format='ascii.commented_header')
            wallHDU.data = fits.BinTableHDU(wall_coords_xyz).data
            wallHDU.header['num_wall_galaxies'] = len(wall_coords_xyz)
            
            field_coords_xyz = Table.read(field_filename,format='ascii.commented_header')
            fieldHDU.data = fits.BinTableHDU(field_coords_xyz).data
            fieldHDU.header['num_field_galaxies'] = len(field_coords_xyz)

        return wallHDU, fieldHDU
    
    # Record the galaxy number density along with intermediate caclulations
    def galaxy_density(primaryHDU, maskHDU, wallHDU, fieldHDU):

        coverage = maskHDU.header['steradian_sky_coverage']
        primaryHDU.header['steradian_sky_coverage'] = coverage
        d_max = primaryHDU.header['dist_limit_Mpc_max']
        d_min = primaryHDU.header['dist_limit_Mpc_min']
        vol = coverage / 3 * (d_max ** 3 - d_min ** 3) #survey volume = steradians / 3 * delta_d
        primaryHDU.header['survey_volume_Mpc3'] = vol
        num_gals = wallHDU.header['num_wall_galaxies'] + fieldHDU.header['num_field_galaxies']
        primaryHDU.header['num_galaxies'] = num_gals
        primaryHDU.header['galaxy_num_density_Mpc-3'] = num_gals/vol

        
    # Save void tables to HDUs
    def write_voids():

        maximals = Table.read(maximal_spheres_filename,format='ascii.commented_header')
        holes = Table.read(void_table_filename,format='ascii.commented_header')

        maximalHDU = fits.BinTableHDU(maximals)
        holeHDU = fits.BinTableHDU(holes)

        maximalHDU.header['num_voids'] = len(maximals)
        holeHDU.header['num_voids'] = len(maximals)

        return maximalHDU, holeHDU

    ############################################################################
    # Fits file creation
    #---------------------------------------------------------------------------

    print('Assembling full output file', flush=True)

    # Assemble each HDU and create fits object
    with warnings.catch_warnings():

        warnings.filterwarnings("ignore", message=".*contains characters not allowed by the FITS standard")
        primaryHDU = write_header()
        maskHDU = write_mask()
        wallHDU, fieldHDU = write_galaxies()
        galaxy_density(primaryHDU, maskHDU, wallHDU, fieldHDU)
        maximalHDU, holeHDU = write_voids()
        hdul = fits.HDUList([primaryHDU, maskHDU, wallHDU, fieldHDU, maximalHDU, holeHDU])

    # Save output
    hdul.writeto(log_filename, overwrite=True)

    # Clean up unneeded files
    to_delete = [maximal_spheres_filename, void_table_filename, mask_filename,
                 wall_filename, field_filename, neighbor_filename]
    
    if wall_field_filename is not None:
        to_delete.append(wall_field_filename)

    for filename in to_delete:
        if os.path.exists(filename):
            os.remove(filename)
    
    print('Voidfinding output saved to ' + log_filename, flush=True)
    





    


       
        
        
        

