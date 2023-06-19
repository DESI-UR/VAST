
import numpy as np
import healpy as hp

import pickle

from astropy.table import Table
from astropy.io import fits

import os


def file_postprocess(maximal_spheres_filename,
                    void_table_filename, 
                    galaxies_filename,
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
    Set up a single fits file for all output results. Currently only works for 
    masked sky surveys, and not for cubic simulations.
    
    
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
        'rabsmag' or 'magnitude'

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
    
    '''
    
    # fits output file name
    log_filename = out_directory + survey_name +'VoidFinder_Output.fits'

    # name of file containing input galaxies
    in_filename = in_directory + galaxies_filename

    # names of files assembled into the output
    wall_filename = out_directory + survey_name + 'wall_gal_file.txt'
    field_filename = out_directory + survey_name + 'field_gal_file.txt'
    neighbor_filename = out_directory + survey_name + 'neighbor_cut.pickle'
    mask_filename = out_directory + survey_name + 'mask.pickle'

    ############################################################################
    # Helper functions for fits file assembly
    #---------------------------------------------------------------------------

    # Save general voidfinding information to a HDU
    def write_header():

        hdr = fits.Header()

        hdr['INFILE'] = (in_filename, 'Input Galaxy Table')
        hdr['METRIC'] = (dist_metric, 'Distance Metric')
        hdr['DLIML'] = (dist_limits[0], 'Lower Distance Limit (Mpc/h)')
        hdr['DLIMU'] = (dist_limits[1], 'Upper Distance Limit (Mpc/h)')
        hdr['ZLIML'] = (min_z, 'Lower Redshift Limit')
        hdr['ZLIMU'] = (max_z, 'Upper Redshift Limit')
        hdr['OMEGAM'] = (Omega_M,'Matter Density')
        hdr['HP'] = (h, 'Hubble Parameter')

        # criteria for separating wall and field galaxies
        if rm_isolated: 
            temp_infile = open(neighbor_filename, 'rb')
            l, sd = pickle.load(temp_infile)
            temp_infile.close()
            hdr['3NNLA'] = (l,'Average 3rd Neighbor Separation')
            hdr['3NNLS'] = (sd,'STD of 3rd Neighbor Separation')
        
        # absolute magnitude cut info
        hdr['MAGCUT']=(mag_cut,'Magnitude Cut Applied')
        if mag_cut:
                hdr['MAGLIM']=(magnitude_limit, 'Magnitude Limit')
        
        hdu = fits.PrimaryHDU(header=hdr)

        return hdu


    # Save mask to a HDU
    def write_mask():

        temp_infile = open(mask_filename, 'rb')
        mask, mask_resolution = pickle.load(temp_infile) #dist_limits is unneeded
        temp_infile.close()
        
        hdu = fits.ImageHDU(mask.astype(int))
        hdr = hdu.header
        hdr['MSKRES'] = (mask_resolution, 'Mask Resolution')

        # caclulate solid angle sky coverage
        nside = round(np.sqrt(mask.size / 12))
        pix_cells = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat = True) #healpix array containing corresponding ra-dec coords in radians
        pix_cells = (np.floor(pix_cells[0]).astype(int), np.floor(pix_cells[1]+90).astype(int))
        healpix_mask = mask[pix_cells]
        coverage = np.sum(healpix_mask) * hp.nside2pixarea(nside) #solid angle coverage in steradians
        hdr['COVSTR'] = (coverage, 'Sky Coverage (Steradians)')

        return hdu

    # Save wall and field galaxy tables to HDUs
    def write_galaxies():

        wallHDU = fits.BinTableHDU()
        fieldHDU = fits.BinTableHDU()

        # Form header
        wallHDU.header['WALL'] = (0, 'Wall Galaxy Count')
        fieldHDU.header['FIELD'] = (0, 'Field Galaxy Count')

        # Read in wall and field galaxies (may be stored in several differnet file configurations)
        if wall_field_filename is not None and os.path.isfile(wall_field_filename):

            temp_infile = open(wall_field_filename, 'rb')
            wall_coords_xyz, field_coords_xyz = pickle.load(temp_infile)
            temp_infile.close()   

            wall_xyz_table = Table(data=wall_coords_xyz, names=["x", "y", "z"])
            field_xyz_table = Table(data=field_coords_xyz, names=["x", "y", "z"])
            wallHDU.data = fits.BinTableHDU(wall_xyz_table).data
            fieldHDU.data = fits.BinTableHDU(field_xyz_table).data
            wallHDU.header['WALL'] = len(wall_coords_xyz)
            fieldHDU.header['FIELD'] = len(field_coords_xyz)

        else:
            
            wall_coords_xyz = Table.read(wall_filename,format='ascii.commented_header')
            wallHDU.data = fits.BinTableHDU(wall_coords_xyz).data
            wallHDU.header['WALL'] = len(wall_coords_xyz)
            
            field_coords_xyz = Table.read(field_filename,format='ascii.commented_header')
            fieldHDU.data = fits.BinTableHDU(field_coords_xyz).data
            fieldHDU.header['FIELD'] = len(field_coords_xyz)

        return wallHDU, fieldHDU
        
    # Save void tables to HDUs
    def write_voids():

        maximals = Table.read(maximal_spheres_filename,format='ascii.commented_header')
        holes = Table.read(void_table_filename,format='ascii.commented_header')

        maximalHDU = fits.BinTableHDU(maximals)
        holeHDU = fits.BinTableHDU(holes)

        maximalHDU.header['VOID'] = (len(maximals), 'Void Count')
        holeHDU.header['VOID'] = (len(maximals), 'Void Count')

        return maximalHDU, holeHDU
    
    # Calculations dependant on the initial HDU properties
    def post_calculations(primaryHDU, maskHDU, wallHDU, fieldHDU, maximalHDU):

        coverage = maskHDU.header['COVSTR']
        primaryHDU.header['COVSTR'] = (coverage, 'Sky Coverage (Steradians)')
        primaryHDU.header['COVDEG'] = (coverage*(180/np.pi)**2, 'Sky Coverage (Degrees^2)')
        d_max = primaryHDU.header['DLIMU']
        d_min = primaryHDU.header['DLIML']
        vol = coverage / 3 * (d_max ** 3 - d_min ** 3) #survey volume = steradians / 3 * delta_d
        primaryHDU.header['VOLUME'] = (vol, 'Survey Volume (Mpc/h)^3')
        num_gals = wallHDU.header['WALL'] + fieldHDU.header['FIELD']
        primaryHDU.header['GALAXY'] = (num_gals,'Galaxy Count')
        primaryHDU.header['WALL'] = (wallHDU.header['WALL'], 'Wall Galaxy Count')
        primaryHDU.header['FIELD'] = (fieldHDU.header['FIELD'], 'Field Galaxy Count')
        primaryHDU.header['DENSITY'] = (num_gals/vol, 'Galaxy Count Density (Mpc/h)^-3')
        primaryHDU.header['AVSEP'] = (np.power(vol/num_gals, 1/3), 'Average Galaxy Separation (Mpc/h)')
        primaryHDU.header['VOID'] = (maximalHDU.header['VOID'], 'Void Count')

    ############################################################################
    # Fits file creation
    #---------------------------------------------------------------------------

    print('Assembling full output file', flush=True)

    # Assemble each HDU and create fits object

    #Supress Hierarch Card warnings (outdated)
    #with warnings.catch_warnings():
        #warnings.filterwarnings("ignore", message=".*contains characters not allowed by the FITS standard")
        #create HDUs here

    primaryHDU = write_header()
    maskHDU = write_mask()
    wallHDU, fieldHDU = write_galaxies()
    maximalHDU, holeHDU = write_voids()
    post_calculations(primaryHDU, maskHDU, wallHDU, fieldHDU, maximalHDU)
    hdul = fits.HDUList([primaryHDU, maskHDU, wallHDU, fieldHDU, maximalHDU, holeHDU])

    # Save output
    hdul.writeto(log_filename, overwrite=True)

    # Clean up unneeded files
    to_delete = [maximal_spheres_filename, void_table_filename,
                 wall_filename, field_filename, neighbor_filename]
    
    if wall_field_filename is not None:
        to_delete.append(wall_field_filename)

    if mask_filename is not None:
        to_delete.append(mask_filename)

    for filename in to_delete:
        if os.path.exists(filename):
            os.remove(filename)
    
    print('Voidfinding output saved to ' + log_filename, flush=True)
    





    


       
        
        
        

