"""Load and print void and galaxy data.
"""

import numpy
import h5py
from astropy.table import Table
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
from vast.vsquared.util import open_fits_file_V2

# Constants
c = 3e5
DtoR = numpy.pi/180.
RtoD = 180./numpy.pi
#distance_metric = 'comoving'
distance_metric = 'comoving'
Omega_M = 0.3
h = 1.0
Kos = FlatLambdaCDM(h*100.,Omega_M)
"""
if __name__ == "__main__":
    
    infilename1 = "../data/DR7_triangles.dat"
    infilename2 = "../data/vollim_dr7_cbp_102709.fits"
    
    # load hole locations
    # keys are 'x' 'y' 'z' 'radius' 'flag'
    
    voids_data = Table.read(infilename1, format='ascii.commented_header')
    
    # Load galaxy data and convert coordinates to xyz
#    galaxy_data = Table.read(infilename2, format='ascii.commented_header')
#    
#    if distance_metric == 'comoving':
#        
#        r_gal = galaxy_data['Rgal']
#        
#    else:
#        
#        r_gal = c*galaxy_data['redshift']/(100*h)
#    
#    xin = r_gal*numpy.cos(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
#    
#    yin = r_gal*numpy.sin(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
#    
#    zin = r_gal*numpy.sin(galaxy_data['dec']*DtoR)
#    
#    xyz_galaxy_data = Table([xin, yin, zin], names=('x','y','z'))
    
    #print(xyz_galaxy_data)
    
    print(voids_data)
"""

def load_void_data(infilename):
    '''
    Load voids as formatted for Vsquared

    Parameters
    ==========

    infilename : string
        path to desired void catalog

    Returns
    =======

    voids_tri_x, _y, _z : numpy.ndarrays shape (N,3)
        the xyz coordinates of the the vertices of triangles making up void edges

    voids_norm : numpy.ndarray shape (N,3)
        the xyz coordinates of each triangle's unit normal vector

    voids_id : numpy.ndarray shape (N,)
        the void ID of each triangle

    gal_viz : numpy.ndarray shape (N,)
        the void ID of each galaxy

    gal_opp : numpy.ndarray shape (N,)
        the void ID of each galaxy's nearest neighbor

    '''
    #read in voids file
    catalog = open_fits_file_V2(infilename)
    
    voids_data = Table(catalog['TRIANGLE'].data)
    gv_data = Table(catalog['GALVIZ'].data)

    del catalog
    #make column names lowercase
    for colname in voids_data.colnames:
        voids_data[colname].name = colname.lower()
    for colname in gv_data.colnames:
        gv_data[colname].name = colname.lower()
    
    num_rows = len(voids_data)

    # each element of voids_tri_x is the x-components of the three vertices of a triangle on a void surface;
    # similarly for voids_tri_y and voids_tri_z
    voids_tri_x = numpy.empty((num_rows, 3), dtype=numpy.float32)
    voids_tri_y = numpy.empty((num_rows, 3), dtype=numpy.float32)
    voids_tri_z = numpy.empty((num_rows, 3), dtype=numpy.float32)

    # each element of voids_norm is the unit vector normal to the surface of a triangle on a void surface
    voids_norm = numpy.empty((num_rows, 3), dtype=numpy.float32)

    # each element of voids_id is the ID of the void whose surface a triangle belongs to
    voids_id = numpy.empty(num_rows, dtype=numpy.int32)

    voids_tri_x[:,0] = voids_data['p1_x']
    voids_tri_x[:,1] = voids_data['p2_x']
    voids_tri_x[:,2] = voids_data['p3_x']
    voids_tri_y[:,0] = voids_data['p1_y']
    voids_tri_y[:,1] = voids_data['p2_y']
    voids_tri_y[:,2] = voids_data['p3_y']    
    voids_tri_z[:,0] = voids_data['p1_z']
    voids_tri_z[:,1] = voids_data['p2_z']
    voids_tri_z[:,2] = voids_data['p3_z']
    voids_norm[:,0] = voids_data['n_x']
    voids_norm[:,1] = voids_data['n_y']
    voids_norm[:,2] = voids_data['n_z']
    voids_id[:] = voids_data["void"]

    # gal_viz contains the void ID of each galaxy
    gal_viz = gv_data['g2v']

    # gal_opp contains the void IDs of each galaxy's neighbors
    gal_opp = gv_data['g2v2']
    
    return voids_tri_x, voids_tri_y, voids_tri_z, voids_norm, voids_id, gal_viz, gal_opp


def load_galaxy_data(infilename):
    """
    Load a table of galaxies for use in VoidRender

    Parameters
    ==========

    infilename : string
        path to desired data file
        intended to be a fits table
        with columns 'ra', 'dec', and 'z'

    Returns
    =======

    galaxy_data_xyz : numpy.ndarray shape (N,3)
        xyz coordinates of galaxies from the data table

    """
    
    galaxy_data = fits.open(infilename)[1].data
    '''
    if distance_metric == 'comoving' and 'Rgal' not in galaxy_data.columns:
        
        r_gal = z_to_comoving_dist(galaxy_data['redshift'].data.astype(numpy.float32), 
                                   Omega_M, 
                                   h)
    
    elif distance_metric == 'comoving':
        
        r_gal = galaxy_data['Rgal']
    '''
    if distance_metric == 'comoving':
        
        r_gal = Kos.comoving_distance(galaxy_data['z'].astype(numpy.float32))
    else:
        
        r_gal = c*galaxy_data['z']/(100*h)
    
    xin = r_gal*numpy.cos(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
    
    yin = r_gal*numpy.sin(galaxy_data['ra']*DtoR)*numpy.cos(galaxy_data['dec']*DtoR)
    
    zin = r_gal*numpy.sin(galaxy_data['dec']*DtoR)
    
    #xyz_galaxy_table = Table([xin, yin, zin], names=('x','y','z'))
    
    num_rows = len(galaxy_data['z'])
    
    galaxy_data_xyz = numpy.empty((num_rows, 3), dtype=numpy.float64)
    
    galaxy_data_xyz[:,0] = xin
    galaxy_data_xyz[:,1] = yin
    galaxy_data_xyz[:,2] = zin
    
    return galaxy_data_xyz
