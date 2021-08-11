
import numpy
import h5py
from astropy.table import Table
import matplotlib
import matplotlib.pyplot as plt

#from vast.voidfinder.absmag_comovingdist_functions import Distance
from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder.preprocessing import load_data_to_Table


# Constants
c = 3e5
DtoR = numpy.pi/180.
RtoD = 180./numpy.pi
distance_metric = 'comoving'
#distance_metric = 'redshift'
Omega_M = 0.3
h = 1.0

if __name__ == "__main__":
    
    infilename1 = "vollim_dr7_cbp_102709_holes.txt"
    infilename2 = "vollim_dr7_cbp_102709_maximal.txt"
    infilename3 = "vollim_dr7_cbp_102709.dat"
    
    ############################################################################
    # load hole locations
    # keys are 'x' 'y' 'z' 'radius' 'flag'
    #---------------------------------------------------------------------------
    holes_data = Table.read(infilename1, format='ascii.commented_header')
    ############################################################################
    
    

    ############################################################################
    # Load galaxy data and convert coordinates to xyz
    #---------------------------------------------------------------------------
    galaxy_data = Table.read(infilename3, format='ascii.commented_header')
    
    if distance_metric == 'comoving':
        
        r_gal = galaxy_data['Rgal']
        
    else:
        
        r_gal = c*galaxy_data['redshift']/(100*h)
    
    xin = r_gal*numpy.cos(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
    
    yin = r_gal*numpy.sin(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
    
    zin = r_gal*numpy.sin(galaxy_data['dec']*DtoR)
    
    xyz_galaxy_data = Table([xin, yin, zin], names=('x','y','z'))
    ############################################################################

    
    
    print(xyz_galaxy_data)
    
    print(holes_data)
    





def load_void_data(infilename):
    '''
    Load voids as formatted for VoidFinder

    Parameters
    ==========

    infilename : string
        path to desired data file

    Returns
    =======

    holes_xyz : numpy.ndarray shape (N,3)
        the xyz centers of the holes

    holes_radii : numpy.ndarray shape (N,)
        the radii of the holes

    hole_flags : numpy.ndarray shape (N,)
        the VoidFinder 'flag' output representing
        which void group a hole belongs to

    '''
    
    holes_data = load_data_to_Table(infilename)
    
    num_rows = len(holes_data)
    
    holes_xyz = numpy.empty((num_rows, 3), dtype=numpy.float64)
    hole_radii = numpy.empty(num_rows, dtype=numpy.float64)
    hole_flags = numpy.empty(num_rows, dtype=numpy.int32)
    
    holes_xyz[:,0] = holes_data['x']
    holes_xyz[:,1] = holes_data['y']
    holes_xyz[:,2] = holes_data['z']
    hole_radii[:] = holes_data["radius"]
    hole_flags[:] = holes_data["flag"]
    
    return holes_xyz, hole_radii, hole_flags


def load_galaxy_data(infilename):
    """
    Load a table of galaxies for use in VoidRender

    Parameters
    ==========

    infilename : string
        path to desired data file
        intended to be an astropy table output from VoidFinder
        with columns 'ra', 'dec', 'redshift', and possibly 'Rgal'

    Returns
    =======

    galaxy_data_xyz : numpy.ndarray shape (N,3)
        xyz coordinates of galaxies from the data table

    """
    
    galaxy_data = load_data_to_Table(infilename)
    
    if distance_metric == 'comoving' and 'Rgal' not in galaxy_data.columns:
        
        r_gal = z_to_comoving_dist(galaxy_data['redshift'].data.astype(numpy.float32), 
                                   Omega_M, 
                                   h)
    
    elif distance_metric == 'comoving':
        
        r_gal = galaxy_data['Rgal']
        
    else:
        
        r_gal = c*galaxy_data['redshift']/(100*h)
    
    xin = r_gal*numpy.cos(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
    
    yin = r_gal*numpy.sin(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
    
    zin = r_gal*numpy.sin(galaxy_data['dec']*DtoR)
    
    #xyz_galaxy_table = Table([xin, yin, zin], names=('x','y','z'))
    
    num_rows = len(galaxy_data)
    
    galaxy_data_xyz = numpy.empty((num_rows, 3), dtype=numpy.float64)
    
    galaxy_data_xyz[:,0] = xin
    galaxy_data_xyz[:,1] = yin
    galaxy_data_xyz[:,2] = zin
    
    return galaxy_data_xyz
    
    
    
    
    
    
    
    


