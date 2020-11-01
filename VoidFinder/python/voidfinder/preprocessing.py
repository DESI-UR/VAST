
import numpy as np

from astropy.table import Table

from astropy.io import ascii

from voidfinder.distance import z_to_comoving_dist

import time

import h5py

from voidfinder.constants import c #speed of light

import os

from astropy.io import fits

def load_data_to_Table(input_filepath):
    """
    Description
    ===========
    
    Load a .dat or .h5 file representing a table of numerical values.  For HDF5 (.h5)
    files, assumes the "columns" are stored as individual datasets on the root h5py.File
    object.
    
    If the input filepath ends with a .h5 extension, this will attempt to read it in 
    as the HDF5 format, otherwise for all other extensions it will attempt the normal
    Table.read()
    
    Parameters
    ==========
    
    input_filepath : str
        path to the input .dat or .h5 file
        
    Returns
    =======
    
    astropy.table.Table
    """
    if input_filepath.endswith(".fits"):
        
        data = fits.open(input_filepath)
        data=data[1].data
        data_table = Table()
        ''' 
        print('Inside load table before columns')
        RA=Table.Column(data['RA'][0:1000], name='ra')
        DEC=Table.Column(data['DEC'][0:1000], name='dec')
        Z=Table.Column(data['Z'][0:1000], name='redshift')
        DELTA=Table.Column(data['deltas'][0:1000], name='deltas')
        '''
        print('Inside load table before columns')                                                     
        RA=Table.Column(data['RA'], name='ra')                                                
        DEC=Table.Column(data['DEC'], name='dec')                                             
        Z=Table.Column(data['Z'], name='redshift')                                            
        DELTA=Table.Column(data['deltas'], name='deltas') 
        
        print('Inside load table before adding columns')
        data_table.add_column(RA)
        data_table.add_column(DEC)
        data_table.add_column(Z)
        data_table.add_column(DELTA)
        
        print('Length of data before removing 0s:')
        print(len(data_table))
        data_table=data_table[data_table['deltas']!=0]
        print('Length of data after removing 0s:')
        print(len(data_table))

        print('fits to table done:)')
        

    
    else:    
        if input_filepath.endswith(".h5"):
        
            infile = h5py.File(input_filepath, 'r')
        
            data_table = Table()
        
            for col in list(infile.keys()):
            
                data_table[col] = infile[col][()]
            
                infile.close()
        
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
                    in_directory, 
                    out_directory, 
                    mag_cut=True,
                    flux_cut=None,
                    rm_isolated=True,
                    dist_metric='comoving', 
                    min_z=None,
                    max_z=None,
                    Omega_M=0.3,
                    h=1.0,
                    verbose=0):
    '''
    Set up output file names, calculate distances, etc.
    
    
    PARAMETERS:
    ==========
    
    galaxies_filename : string
        File name of galaxy catalog.  Should be readable by 
        astropy.table.Table.read as a ascii.commented_header file.  Required 
        columns include 'ra', 'dec', 'z', and absolute magnitude (either 
        'rabsmag' or 'magnitude'.
        
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
    
    
    RETURNS:
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
    
    ############################################################################
    # Build output file names
    #---------------------------------------------------------------------------
    if mag_cut and rm_isolated:
        out1_suffix = '_' + dist_metric + '_maximal.txt'
        out2_suffix = '_' + dist_metric + '_holes.txt'
    elif rm_isolated:
        out1_suffix = '_' + dist_metric + '_maximal_noMagCut.txt'
        out2_suffix = '_' + dist_metric + '_holes_noMagCut.txt'
    elif mag_cut:
        out1_suffix = '_' + dist_metric + '_maximal_keepIsolated.txt'
        out2_suffix = '_' + dist_metric + '_holes_keepIsolated.txt'
    else:
        out1_suffix = '_' + dist_metric + '_maximal_noFiltering.txt'
        out2_suffix = '_' + dist_metric + 'holes_noFiltering.txt'
    
    out1_filename = out_directory + galaxies_filename[:-4] + out1_suffix  # List of maximal spheres of each void region: x, y, z, radius, distance, ra, dec
    out2_filename = out_directory + galaxies_filename[:-4] + out2_suffix  # List of holes for all void regions: x, y, z, radius, flag (to which void it belongs)
    #out3_filename = out_directory + 'out3_vollim_dr7.txt'                # List of void region sizes: radius, effective radius, evolume, x, y, z, deltap, nfield, vol_maxhole
    #voidgals_filename = out_directory + 'vollim_voidgals_dr7.txt'        # List of the void galaxies: x, y, z, void region
    ############################################################################
    
    
    ############################################################################
    # Open galaxy catalog
    #---------------------------------------------------------------------------
    in_filename = in_directory + galaxies_filename
    
    if verbose > 0:
        print("Loading galaxy data table at: ", in_filename, flush=True)
        load_start_time = time.time()
    
    
    galaxy_data_table = load_data_to_Table(in_filename)

    if verbose > 0:
        print("Galaxy data table load time: ", time.time() - load_start_time, flush=True)
    ############################################################################
    
    
    ############################################################################
    # Rename columns
    #---------------------------------------------------------------------------
    if mag_cut and 'rabsmag' not in galaxy_data_table.columns:
        galaxy_data_table['magnitude'].name = 'rabsmag'
        
    if 'z' not in galaxy_data_table.columns:
        galaxy_data_table['redshift'].name = 'z'
    ############################################################################

    
    ############################################################################
    # Determine min and max redshifts if not supplied by user
    #---------------------------------------------------------------------------
    # Minimum distance
    if min_z is None:
        min_z = min(galaxy_data_table['z'])

    
    # Maximum distance
    if max_z is None:
        max_z = max(galaxy_data_table['z'])
    
    
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
    if dist_metric == 'comoving' and 'Rgal' not in galaxy_data_table.columns:
        
        if verbose > 0:
            print("Calculating Rgal data table column", flush=True)
            calc_start_time = time.time()
        
        galaxy_data_table['Rgal'] = z_to_comoving_dist(galaxy_data_table['z'].data.astype(np.float32), Omega_M, h)
    
        if verbose > 0:
            print("Finished Rgal calculation time: ", time.time() - calc_start_time, flush=True)
    ############################################################################
    
    print(galaxy_data_table)

    return galaxy_data_table, dist_limits, out1_filename, out2_filename



       
        
        
        

