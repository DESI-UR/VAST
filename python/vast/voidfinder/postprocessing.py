
import numpy as np
import healpy as hp

import pickle

from astropy.table import Table
from astropy.io import fits

import os

# (Make Number) Format floats for headers
def mknum (flt):

    if flt is None:
        return None

    #preserve 3 sig figs for numbers starting with "0."
    if abs(flt) < 1:
        return float(f"{flt:.3g}")
    #otherwise round to two decimal places
    else:
        return float(f"{flt:.2f}")
        
def file_postprocess(maximals,
                    holes, 
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
                    wall_galaxies,
                    field_galaxies=None,
                    verbose=0):
    '''
    Set up a single fits file for all output results. Currently only works for 
    masked sky surveys, and not for cubic simulations.
    
    
    PARAMETERS
    ==========
    
    maximals : Astropy Table
        the table of maximal spheres 
    
    holes : Astropy Table
        the table of hole spheres 
    
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
    
    pass


    
def open_fits_file(
        out_directory, 
        survey_name):
    
    # format directory and file name appropriately
    if len(out_directory) > 0 and out_directory[-1] != '/':
        out_directory += '/'

    if len(survey_name) > 0 and survey_name[-1] != '_':
        survey_name += '_'

    log_filename = out_directory + survey_name +'VoidFinder_Output.fits'
    
    #create the output file if it doesn't already exist
    if not os.path.isfile(log_filename):
        hdul = fits.HDUList([fits.PrimaryHDU(header=fits.Header())])
        hdul.writeto(log_filename)

    #open the output file
    hdul = fits.open(log_filename)

    return hdul, log_filename


def save_output_from_preprocessing(
        galaxies_filename,
        out_directory, 
        survey_name,
        dist_metric,
        dist_limits, 
        min_z,
        max_z,
        Omega_M,
        h,
        verbose=0):
    
    #open the output file
    hdul, log_filename = open_fits_file(out_directory, survey_name)

    #modify output file
    primaryHDU_header = hdul['PRIMARY'].header
    primaryHDU_header['INFILE'] = (galaxies_filename, 'Input Galaxy Table')
    primaryHDU_header['METRIC'] = (dist_metric, 'Distance Metric')
    primaryHDU_header['DLIML'] = (mknum(dist_limits[0]), 'Lower Distance Limit (Mpc/h)')
    primaryHDU_header['DLIMU'] = (mknum(dist_limits[1]), 'Upper Distance Limit (Mpc/h)')
    primaryHDU_header['ZLIML'] = (min_z, 'Lower Redshift Limit')
    primaryHDU_header['ZLIMU'] = (max_z, 'Upper Redshift Limit')
    primaryHDU_header['OMEGAM'] = (Omega_M,'Matter Density')
    primaryHDU_header['HP'] = (h, 'Reduced Hubble Parameter h (((km/s)/Mpc)/100)')

    #save file changes
    hdul.writeto(log_filename, overwrite=True)

def append_wall_field_galaxies(
        hdul,   
        wall_gals_xyz, 
        field_gals_xyz):
    try:
        hdul.index_of('WALL')
    except:
        hdu = fits.BinTableHDU()
        hdu.name = 'WALL'
        hdul.append(hdu)
    try:
        hdul.index_of('FIELD')
    except:
        hdu = fits.BinTableHDU()
        hdu.name = 'FIELD'
        hdul.append(hdu)

    wallHDU = hdul['WALL']
    fieldHDU = hdul['FIELD']
    wall_xyz_table = Table(data=wall_gals_xyz, names=["X", "Y", "Z"], units = ['Mpc/h','Mpc/h','Mpc/h'])
    field_xyz_table = Table(data=field_gals_xyz, names=["X", "Y", "Z"], units = ['Mpc/h','Mpc/h','Mpc/h'])
    wallHDU.data = fits.BinTableHDU(wall_xyz_table).data
    fieldHDU.data = fits.BinTableHDU(field_xyz_table).data
    wallHDU.header['WALLNUM'] = (len(wall_xyz_table), 'Wall Galaxy Count')
    fieldHDU.header['FIELDNUM'] = (len(field_xyz_table), 'Field Galaxy Count')

def save_output_from_filter_galaxies(
        survey_name, 
        out_directory,
        wall_gals_xyz, 
        field_gals_xyz,
        write_galaxies,
        mag_cut, 
        dist_limits,
        rm_isolated,
        dist_metric, 
        h,
        magnitude_limit,
        verbose=0):

    #open the output file
    hdul, log_filename = open_fits_file(out_directory, survey_name)

    #modify output file
    primaryHDU_header = hdul['PRIMARY'].header
    primaryHDU_header['METRIC'] = (dist_metric, 'Distance Metric')
    if dist_limits is not None:
        primaryHDU_header['DLIML'] = (mknum(dist_limits[0]), 'Lower Distance Limit (Mpc/h)')
        primaryHDU_header['DLIMU'] = (mknum(dist_limits[1]), 'Upper Distance Limit (Mpc/h)')
    primaryHDU_header['HP'] = (h, 'Reduced Hubble Parameter h (((km/s)/Mpc)/100)')
    primaryHDU_header['RMISO'] = (rm_isolated, 'Isolated Galaxies Removed')
    primaryHDU_header['MAGCUT']=(mag_cut,'Magnitude Cut Applied')
    if mag_cut:
            primaryHDU_header['MAGLIM']=(magnitude_limit, 'R-Band Magnitude Limit')

    
    if write_galaxies:
        append_wall_field_galaxies(
            hdul,   
            wall_gals_xyz, 
            field_gals_xyz
        )

    #save file changes
    hdul.writeto(log_filename, overwrite=True)

def save_output_from_wall_field_separation(   
    survey_name, 
    out_directory,
    wall_gals_xyz,
    field_gals_xyz,
    write_galaxies,
    sep_neighbor,
    avsep,
    sd,
    verbose=0):
       
    
    #open the output file
    hdul, log_filename = open_fits_file(out_directory, survey_name)

    primaryHDU_header = hdul['PRIMARY'].header
    primaryHDU_header['NNS'] = (sep_neighbor,'Nth Neighbor Used for Wall-Field Separation')
    primaryHDU_header['NNSA'] = (mknum(avsep),'Average Nth Neighbor Separation (Mpc/h)')
    primaryHDU_header['NNSS'] = (mknum(sd),'STD of Nth Neighbor Separation (Mpc/h)')
    
    if write_galaxies:
        append_wall_field_galaxies(
            hdul,   
            wall_gals_xyz, 
            field_gals_xyz
        )

    #save file changes
    hdul.writeto(log_filename, overwrite=True)   

def save_output_from_generate_mask(
        mask,
        mask_resolution,                    
        survey_name,
        out_directory,
        smooth_mask,
        log_smooth_mask=True):
    
    #open the output file
    hdul, log_filename = open_fits_file(out_directory, survey_name)

    append_mask(
        hdul,
        mask,
        mask_resolution,   
        smooth_mask,
        log_smooth_mask)
    
    #save file changes
    hdul.writeto(log_filename, overwrite=True)  


def append_mask(
        hdul,
        mask,
        mask_resolution,   
        smooth_mask=None,
        log_smooth_mask=False):
    
    try:
        hdul.index_of('MASK')
    except:
        hdu = fits.ImageHDU()
        hdu.name = 'MASK'
        hdul.append(hdu)

    maskHDU = hdul['MASK']
    maskHDU.data = fits.ImageHDU(mask.astype(int)).data
    maskHDU.header['MSKRES'] = (mask_resolution, 'Mask Resolution')
    if log_smooth_mask: maskHDU.header['SMMSK'] = (smooth_mask, 'Smooth Mask')

    # caclulate solid angle sky coverage
    nside = round(np.sqrt(mask.size / 12))
    pix_cells = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat = True) #healpix array containing corresponding ra-dec coords in radians
    pix_cells = (np.floor(pix_cells[0]).astype(int), np.floor(pix_cells[1]+90).astype(int))
    healpix_mask = mask[pix_cells]
    coverage = np.sum(healpix_mask) * hp.nside2pixarea(nside) #solid angle coverage in steradians
    maskHDU.header['COVSTR'] = (coverage, 'Sky Coverage (Steradians)')
 



def save_output_from_find_voids(
        maximals,
        holes, 
        galaxy_coords_xyz,
        out_directory, 
        survey_name,
        mask_type,
        mask,            
        mask_resolution,
        dist_limits,     
        xyz_limits,     
        check_only_empty_cells,
        max_hole_mask_overlap,  
        hole_grid_edge_length,   
        grid_origin,             
        min_maximal_radius,
        galaxy_map_grid_edge_length,
        pts_per_unit_volume,
        num_cpus,
        batch_size,
        verbose=0):

    print('Assembling full output file', flush=True)

    #open the output file
    hdul, log_filename = open_fits_file(out_directory, survey_name)

    try:
        hdul.index_of('MAXIMALS')
    except:
        hdu = fits.BinTableHDU()
        hdu.name = 'MAXIMALS'
        hdul.append(hdu)
    try:
        hdul.index_of('HOLES')
    except:
        hdu = fits.BinTableHDU()
        hdu.name = 'HOLES'
        hdul.append(hdu)

    maximalHDU=hdul['MAXIMALS']
    holeHDU=hdul['HOLES']

    maximalHDU.data = fits.BinTableHDU(maximals).data
    holeHDU.data = fits.BinTableHDU(holes).data

    maximalHDU.header['VOID'] = (len(maximals), 'Void Count')
    holeHDU.header['VOID'] = (len(maximals), 'Void Count')
    holeHDU.header['HOLE'] = (len(holes), 'Hole Count')

    if mask is not None:
        append_mask(
            hdul,
            mask,
            mask_resolution) 
    
    try:
        hdul.index_of('FIELD')
    except:
        append_wall_field_galaxies(
            hdul,   
            galaxy_coords_xyz, 
            np.array([]))

    primaryHDU = hdul['PRIMARY']
    wallHDU = hdul['WALL']
    fieldHDU=hdul['FIELD']
    

    if mask_type == 'ra_dec_z':
        maskHDU = hdul['MASK']
        primaryHDU.header['MSKRES'] = (maskHDU.header['MSKRES'], 'Mask Resolution')
        coverage = maskHDU.header['COVSTR'] #preseve copy of COVSTR with more decimal places
        maskHDU.header['COVSTR'] = mknum(coverage)
        primaryHDU.header['COVSTR'] = (maskHDU.header['COVSTR'], 'Sky Coverage (Steradians)')
        primaryHDU.header['COVDEG'] = (mknum(coverage*(180/np.pi)**2), 'Sky Coverage (Degrees^2)')
        
        d_max = primaryHDU.header['DLIMU']
        d_min = primaryHDU.header['DLIML']
        vol = coverage / 3 * (d_max ** 3 - d_min ** 3) #survey volume = steradians / 3 * delta_d
    else:
        delta_x = xyz_limits[1,0]-xyz_limits[0,0]
        delta_y = xyz_limits[1,1]-xyz_limits[0,1]
        delta_z = xyz_limits[1,2]-xyz_limits[0,2]
        vol = delta_x*delta_y*delta_z

    primaryHDU.header['VOLUME'] = (mknum(vol), 'Survey Volume (Mpc/h)^3')
    num_gals = int(wallHDU.header['WALLNUM']) + int(fieldHDU.header['FIELDNUM'])
    primaryHDU.header['GALAXY'] = (num_gals,'Galaxy Count')
    primaryHDU.header['WALLNUM'] = (wallHDU.header['WALLNUM'], 'Wall Galaxy Count')
    primaryHDU.header['FIELDNUM'] = (fieldHDU.header['FIELDNUM'], 'Field Galaxy Count')
    primaryHDU.header['DENSITY'] = (mknum(num_gals/vol), 'Galaxy Count Density (Mpc/h)^-3')
    primaryHDU.header['AVSEP'] = (mknum(np.power(vol/num_gals, 1/3)), 'Average Galaxy Separation (Mpc/h)')
    primaryHDU.header['VOID'] = (maximalHDU.header['VOID'], 'Void Count')
    if dist_limits is not None:
        primaryHDU.header['DLIML'] = (mknum(dist_limits[0]), 'Lower Distance Limit (Mpc/h)')
        primaryHDU.header['DLIMU'] = (mknum(dist_limits[1]), 'Upper Distance Limit (Mpc/h)')

    primaryHDU.header ['CHKE'] = (check_only_empty_cells,'Check Only Empty Cells') 
    primaryHDU.header ['MHMO'] = (mknum(max_hole_mask_overlap),'Max Hole Mask Overlap') 
    primaryHDU.header ['HGEL'] = (mknum(hole_grid_edge_length),'Hole Grid Edge Length') 
    primaryHDU.header ['GOX'] = (mknum(grid_origin[0]),'Grid Origin X Coordinate') 
    primaryHDU.header ['GOY'] = (mknum(grid_origin[1]),'Grid Origin Y Coordinate') 
    primaryHDU.header ['GOZ'] = (mknum(grid_origin[2]),'Grid Origin Z Coordinate')
    primaryHDU.header ['MMR'] = (mknum(min_maximal_radius),'Minimum Allowed Maximal Radius')
    primaryHDU.header ['MMR'] = (mknum(galaxy_map_grid_edge_length), 'Edge Length of NN Grid')
    primaryHDU.header ['PPUV'] = (mknum(pts_per_unit_volume),'Points per Unit Volume')
    primaryHDU.header ['NCPU'] = (num_cpus,'Number of CPUs')
    primaryHDU.header ['BS'] = (batch_size,'Batch Size')

    hdul.writeto(log_filename, overwrite=True)   
    print('Voidfinding output saved to ' + log_filename, flush=True)