
import numpy as np

from astropy.table import Table
from astropy.io import ascii, fits

from vast.voidfinder.distance import z_to_comoving_dist

import time

import h5py

from vast.voidfinder.constants import c #speed of light

import os

from .postprocessing import save_output_from_preprocessing



def load_data_to_Table(input_filepath):
    """
    Load a .dat, .txt, .fits, or .h5 file representing a table of numerical 
    values.  For HDF5 (.h5) files, assumes the "columns" are stored as 
    individual datasets on the root h5py.File object.
    
    If the input filepath ends with a .h5 extension, this will attempt to read 
    it in as the HDF5 format, otherwise for all other extensions it will attempt 
    the normal Table.read()
    
    Parameters
    ==========
    
    input_filepath : str
        path to the input .dat, .txt, .fits, or .h5 file
        
    Returns
    =======
    
    astropy.table.Table
    """
        
    if input_filepath.endswith(".h5"):
        
        infile = h5py.File(input_filepath, 'r')
        
        data_table = Table()
        
        for col in list(infile.keys()):
            
            data_table[col] = infile[col][()]
            
        infile.close()

    elif input_filepath.endswith('.fits') or input_filepath.endswith('.fit'):

        hdu = fits.open(input_filepath)

        data_table = Table(hdu[1].data)

        hdu.close()
        
    else:
        
        if os.path.getsize(input_filepath) < 5e9:
            
            data_table = Table.read(input_filepath, format='ascii.commented_header')

        else:
        
            # Import header line
            data_table_fobj = open(input_filepath, 'r')

            header_line = data_table_fobj.readline()

            data_table_fobj.close()

            # Parse header line
            col_names = header_line.split(' ')

            # Read in the data in 100 Mb chunks
            data_table = ascii.read(input_filepath, 
                                    format='no_header', 
                                    names=col_names[1:], 
                                    guess=False, 
                                    fast_reader={'chunk_size': 100*1000000})
        
    return data_table
    
    
    


def file_preprocess(galaxies_filename, 
                    survey_name,
                    in_directory, 
                    out_directory, 
                    mag_cut=True,
                    rm_isolated=True,
                    dist_metric='comoving', 
                    min_z=None,
                    max_z=None,
                    Omega_M=0.3,
                    h=1.0,
                    verbose=0):
    '''
    Set up output file names, calculate distances, etc.
    
    
    PARAMETERS
    ==========
    
    galaxies_filename : string
        File name of galaxy catalog.  Should be readable by 
        astropy.table.Table.read as a ascii.commented_header file.  Required 
        columns include 'ra', 'dec', 'z', and absolute magnitude (either 
        'rabsmag' or 'magnitude'.

    survey_name : str
        identifier for the survey running, may be prepended or appended to 
        output filenames including the checkpoint filename
        
    in_directory : string
        Directory path for input files
    
    out_directory : string
        Directory path for output files
        
    mag_cut : boolean
        Determines whether or not to implement a magnitude cut on the galaxy 
        survey.  Default is True (remove all galaxies fainter than Mr = -20).
        
    rm_isolated : boolean
        Determines whether or not to remove isolated galaxies (defined as those 
        with the distance to their third nearest neighbor greater than the sum 
        of the average third-nearest-neighbor distance and 1.5 times the 
        standard deviation of the third-nearest-neighbor distances).
    
    dist_metric : string
        Description of which distance metric to use.  Options should include 
        'comoving' (default) and 'redshift'.
        
    min_z, max_z : float
        Minimum and maximum redshift range for the survey mask.  Default values 
        are None (determined from galaxy extent).
        
    Omega_M : float
        Value of the matter density of the given cosmology.  Default is 0.3.
        
    h : float
        Value of the Hubble constant.  Default is 1 (so all distances will be in 
        units of h^-1).
    
    
    RETURNS
    =======
    
    galaxy_data_table : astropy table
        Table of all galaxies in catalog.
        
    dist_limits : numpy array of shape (2,)
        Minimum and maximum distances to use for void search.  Units are Mpc/h, 
        in either comoving or redshift coordinates (depending on dist_metric).
        
    
    '''
    
    
    ############################################################################
    # Open galaxy catalog
    #---------------------------------------------------------------------------
    in_filename = in_directory + galaxies_filename
    
    print("Loading galaxy data table at: ", in_filename, flush=True)
    load_start_time = time.time()
    
    galaxy_data_table = load_data_to_Table(in_filename)
        
    print("Galaxy data table load time: ", time.time() - load_start_time, flush=True)
    ############################################################################
    
    
    ############################################################################
    # Rename columns
    #---------------------------------------------------------------------------
    if mag_cut and ('rabsmag' not in galaxy_data_table.columns):
        galaxy_data_table['magnitude'].name = 'rabsmag'
        
    if 'redshift' not in galaxy_data_table.columns:
        galaxy_data_table['z'].name = 'redshift'
    ############################################################################
    
    
    ############################################################################
    # Determine min and max redshifts if not supplied by user
    #---------------------------------------------------------------------------
    # Minimum distance
    if min_z is None:
        min_z = min(galaxy_data_table['redshift'])

    
    # Maximum distance
    if max_z is None:
        max_z = max(galaxy_data_table['redshift'])
    
    
    if dist_metric == 'comoving':
        # Convert redshift to comoving distance
        dist_limits = z_to_comoving_dist(np.array([min_z, max_z], dtype=np.float32), Omega_M, h)
    else:
        H0 = 100*h
        dist_limits = c*np.array([min_z, max_z])/H0
    ############################################################################
    
    
    ############################################################################
    # Calculate comoving distance
    #---------------------------------------------------------------------------
    if dist_metric == 'comoving': #and 'Rgal' not in galaxy_data_table.columns:
        
        print("Calculating Rgal data table column", flush=True)
        calc_start_time = time.time()
        
        galaxy_data_table['Rgal'] = z_to_comoving_dist(galaxy_data_table['redshift'].data.astype(np.float32), Omega_M, h)
    
        print("Finished Rgal calculation time: ", time.time() - calc_start_time, flush=True)
    ############################################################################
        
    
    save_output_from_preprocessing(
        galaxies_filename,
        out_directory, 
        survey_name,
        dist_metric, 
        dist_limits,
        min_z,
        max_z,
        Omega_M,
        h,
        verbose
    )
    
    return galaxy_data_table, dist_limits



       
        
        
        

