import numpy as np
from collections.abc import Iterable
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy import interpolate
from astropy.io import fits
import os
import mmap


c    = 3e5
D2R  = np.pi/180.


def toCoord(z,ra,dec,H0,Om_m):
    """Convert redshift, RA, and Dec to comoving coordinates.

    Parameters
    ----------
    z : list or ndarray
        Object redshift.
    ra : list or ndarray
        Object right ascension, in decimal degrees.
    dec : list or ndarray
        Object declination, in decimal degrees.
    H0 : float
        Hubble's constant in km/s/Mpc.
    Om_m : float
        Value of matter density.

    Returns
    -------
    cs : list
        Comoving xyz-coordinates, assuming input cosmology.
    """
    Kos = FlatLambdaCDM(H0,Om_m)
    r = Kos.comoving_distance(z)
    r = np.array([d.value for d in r])
    #r = c*z/H0
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    return c1,c2,c3


def toSky(cs,H0,Om_m,zstep):
    """Convert redshift, RA, and Dec to comoving coordinates.

    Parameters
    ----------
    cs : ndarray
        Comoving xyz-coordinates table [x,y,z], assuming input cosmology.
    H0 : float
        Hubble's constant in km/s/Mpc.
    Om_m : float
        Value of matter density.
    zstep : float
        Redshift step size for converting distance to redshift.

    Returns
    -------
    z : float
        Object redshift.
    ra : float
        Object right ascension, in decimal degrees.
    dec : float
        Object declination, in decimal degrees.
    """
    Kos = FlatLambdaCDM(H0,Om_m)
    c1 = cs.T[0]
    c2 = cs.T[1]
    c3 = cs.T[2]
    r   = np.sqrt(c1**2.+c2**2.+c3**2.)
    dec = np.arcsin(c3/r)/D2R
    ra  = (np.arccos(c1/np.sqrt(c1**2.+c2**2.))*np.sign(c2)/D2R)%360
    zmn = z_at_value(Kos.comoving_distance, np.amin(r)*u.Mpc, method='bounded')
    zmx = z_at_value(Kos.comoving_distance, np.amax(r)*u.Mpc, method='bounded')
    zmn = zmn-(zstep+zmn%zstep)
    zmx = zmx+(2*zstep-zmx%zstep)
    ct  = np.array([np.linspace(zmn,zmx,int(np.ceil(zmn/zstep))),Kos.comoving_distance(np.linspace(zmn,zmx,int(np.ceil(zmn/zstep)))).value]).T
    r2z = interpolate.pchip(*ct[:,::-1].T)
    z = r2z(r)
    #z = H0*r/c
    return z,ra,dec


def dcut_worker(num_voids,
                index_coordinator,
                file_descriptor,
                vcens,
                vrads,
                coords, 
                vvols,
                minvol,
                periodic, 
                cmin, 
                cmax,
               ):
    """Apply central density cuts to the void catalog in parallel.

    Parameters
    ----------
    num_voids : int
        The total number of voids in the catalog
    index_coordinator : multiprocessing.Value
        Index for coordinating void selection between parallel processes
    file_descriptor : int
        The file descriptor integer used to reference the shared memory for the parallel processes
    vcens : ndarray
        The void centers
    vrads : ndarray
        The void raddii
    coords : ndarray
        The coordinates of the Vornoi cell centers
    vvols : ndarray
        Array of void volumes.
    minvol : float
        The threshold central density (given as a volume) used for cutting voids
    periodic: boolean
        Flag indicating periodic mode
    cmin: array
        Minimum coordinates of survey
    cmin: array
        Maximum coordinates of survey
        
    """
   
                                 
    
    buffer_length = num_voids #bool so 1 bytes per element

    buffer = mmap.mmap(file_descriptor, buffer_length)
    
    dcut = np.frombuffer(buffer, dtype=bool)

    dcut.shape = (num_voids,)
    
    curr_index = 0
    
    while True:
        
        index_coordinator.acquire()
        
        curr_index = index_coordinator.value
        
        index_coordinator.value += 1
        
        index_coordinator.release()
    
        if curr_index >= num_voids:
            break

        vcen = vcens[curr_index]
        vrad = vrads[curr_index]
        vvol = vvols[curr_index]

        # number of galaxies within 1/4th of the void radius divided by volume 4/3*pi*(R/4)^3
        # should be less than the user specified fraction of the mean density
        void_cut = 64.* num_coords_in_sphere(vcen, vrad/4., coords, periodic, cmin, cmax) / vvol <1./ minvol
    
        dcut[curr_index] = void_cut
        

"""
def inSphere(cs, r, coords, periodic, cmin, cmax):
    '''
    Checks if a set of comoving coordinates are within a sphere.

    Parameters
    ==========

    cs : list or ndarray
        Center of sphere.

    r : float
        Sphere volume.

    coords : list or ndarray
        Comoving xyz-coordinates.

    periodic: boolean
        Flag indicating periodic mode

    cmin: array
        Minimum coordinates
        
    cmin: array
        Maximum coordinates

    Returns
    =======

    inSphere : bool array
        True if abs(coords - cs) < r.
    '''
    if not periodic:
        #return np.sum((cs.reshape(3,1) - coords.T)**2., axis=0)<r**2.
        return np.sum((cs - coords)**2., axis=1)<r**2.

    box_size = (cmax - cmin)

    transformed_coords = np.array(coords)

    transformed_coords = transformed_coords - cs + box_size / 2

    transformed_coords = transformed_coords % box_size

    return np.sum((box_size.reshape(3,1) / 2 - transformed_coords.T)**2., axis=0)<r**2.
"""


def num_coords_in_sphere(cs, r, coords, periodic, cmin, cmax):
    """
    Checks if a set of comoving coordinates are within a sphere.

    Parameters
    ==========

    cs : list or ndarray
        Center of sphere.

    r : float
        Sphere volume.

    coords : list or ndarray
        Comoving xyz-coordinates.

    periodic: boolean
        Flag indicating periodic mode

    cmin: array
        Minimum coordinates of the survey
        
    cmin: array
        Maximum coordinates of the survey

    Returns
    =======

    num_in_sphere : int
        The number of coords that meet the condition abs(coords - cs) < r.
    """
    if not periodic:

        #in_sphere = np.sum((cs.reshape(3,1) - coords.T)**2., axis=0)<r**2.
        diff = cs - coords
        
    else:
        box_size = (cmax - cmin)
    
        transformed_coords = np.array(coords)
    
        transformed_coords = transformed_coords - cs + box_size / 2
    
        transformed_coords = transformed_coords % box_size
    
        #in_sphere = np.sum((box_size.reshape(3,1) / 2 - transformed_coords.T)**2., axis=0)<r**2.
        diff = box_size / 2 - transformed_coords
    
    diff = np.power(diff, 2)
    diff = np.sum(diff, axis = 1)
    in_sphere = diff < r**2

    num_in_sphere = np.sum(in_sphere)
    
    return num_in_sphere


'''
def getBuff(cin, idsin, cmin, cmax, buff, n):
    """Identify tracers contained in buffer shell around periodic boundary.

    Parameters
    ==========
    
    cin : ndarray
        Array of tracer positions.
    idsin : ndarray
        Array of tracer IDs.
    cmin : ndarray
        Array of coordinate minima.
    cmax : ndarray
        Array of coordinate maxima.
    buff : float
        Width of buffer shell.
    n : int
        Number of buffer shell.

    Returns
    =======
    
    cout : list
        List of buffer tracer positions.
    idsout : ndarray
        Array of tracer IDs in the original periodic box.
    """
    
    cout = []
    
    idsout = idsin.tolist()
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                
                if i==1 and j==1 and k==1:
                    continue
                
                #create an offset copy of the box (c2)
                c2 = cin+(np.array([i,j,k])-1)*(cmax-cmin)
                c2d = np.amax(np.abs(c2-(cmax+cmin)/2.)-(cmax-cmin)/2.,axis=1)
                cut = c2d<buff*(n+1)
                cut[c2d<=buff*n] = False
                cout.extend(c2[cut].tolist())
                idsout.extend(idsin[:len(cin)][cut].tolist())
                
    return cout, np.array(idsout)
'''


def wCen_worker(num_voids,
                index_coordinator,
                file_descriptor,
                vcuts,
                vols,
                coords, 
                periodic, 
                cmin, 
                cmax,
               ):
    """Find the weighted center of tracers' Voronoi cells in parallel

    Parameters
    ----------
    num_voids : int
        The total number of voids in the catalog
    index_coordinator : multiprocessing.Value
        Index for coordinating void selection between parallel processes
    file_descriptor : int
        The file descriptor integer used to reference the shared memory for the parallel processes
    vcuts : list of lists
        Cuts to select the appopriate coordinates and cell volumes for each void
    vols : ndarray
        Array of Voronoi cell volumes.
    coords : ndarray
        The coordinates of the Vornoi cell centers
    periodic: boolean
        Flag indicating periodic mode
    cmin: array
        Minimum coordinates of survey
    cmin: array
        Maximum coordinates of survey

    """                      
    
    buffer_length = num_voids*8*3 #float64 so 8 bytes per element

    buffer = mmap.mmap(file_descriptor, buffer_length)
    
    vcens = np.frombuffer(buffer, dtype=np.float64)

    vcens.shape = (num_voids, 3)
    
    curr_index = 0
    
    while True:
        
        index_coordinator.acquire()
        
        curr_index = index_coordinator.value
        
        index_coordinator.value += 1
        
        index_coordinator.release()
    
        if curr_index >= num_voids:
            break

        vcut = vcuts[curr_index]
    
        void_center = wCen(vols[vcut],coords[vcut], periodic, cmin, cmax)
        vcens[curr_index] = void_center
    
    
def wCen(vols,coords, periodic, cmin, cmax):
    """Find the weighted center of tracers' Voronoi cells.

    Parameters
    ----------
    vols : ndarray
        Array of Voronoi cell volumes.
    coords : ndarray
        Array of cells' positions.
    periodic: boolean
        Flag indicating periodic mode
    cmin: array
        Minimum coordinates of survey
    cmin: array
        Maximum coordinates of survey
        
    Returns
    -------
    wCen : ndarray
        Weighted center of tracers' Voronoi cells.
    """
    if not periodic:
        return np.sum(vols.reshape(len(vols),1)*coords,axis=0)/np.sum(vols)

    transformed_coords = np.array(coords)

    box_size = (cmax - cmin)

    transformed_coords = transformed_coords - coords[0] + box_size / 2
    transformed_coords = transformed_coords % box_size
    
    center = np.sum(vols.reshape(len(vols),1)*transformed_coords,axis=0)/np.sum(vols)
        
    center = center + coords[0] - box_size / 2
    
    center = center - cmin
    center = center % box_size
    center = center + cmin

    if np.any((np.max(transformed_coords, axis=0) - np.min(transformed_coords, axis=0)) >= box_size/2):
        print('WARNING: A void has been detected in periodic mode that is longer in at least one dimension than half the simulation width. The void center and best fit ellipsoid may not be accurately caclulated.')

    return center


def getSMA_worker(num_voids,
                index_coordinator,
                file_descriptor,
                vrads,
                vcens,
                vcuts,
                coords, 
                periodic, 
                cmin, 
                cmax,
               ):
    """Convert tracers and void effective radius to ellipsoid semi-major axes in parallel.

    Parameters
    ----------
    num_voids : int
        The total number of voids in the catalog
    index_coordinator : multiprocessing.Value
        Index for coordinating void selection between parallel processes
    file_descriptor : int
        The file descriptor integer used to reference the shared memory for the parallel processes
    vrads : ndarray
        The void raddii
    vcens : ndarray
        The void centers
    vcuts : list of lists
        Cuts to select the appopriate coordinates and cell volumes for each void
    coords : ndarray
        The coordinates of the Vornoi cell centers
    periodic: boolean
        Flag indicating periodic mode
    cmin: array
        Minimum coordinates of survey
    cmin: array
        Maximum coordinates of survey

    """
   
                                 
    buffer_length = num_voids*8*3*3 #float64 so 8 bytes per element and 3 by 3 table for each void

    buffer = mmap.mmap(file_descriptor, buffer_length)
    
    ellipses = np.frombuffer(buffer, dtype=np.float64)

    ellipses.shape = (num_voids,3,3)
    
    curr_index = 0
    
    while True:
        
        index_coordinator.acquire()
        
        curr_index = index_coordinator.value
        
        index_coordinator.value += 1
        
        index_coordinator.release()
    
        if curr_index >= num_voids:
            break

        vrad = vrads[curr_index]
        vcut = vcuts[curr_index]
        vcen = vcens[curr_index]
    
        eigenvalue = getSMA(vrad, vcen, coords[vcut], periodic, cmin, cmax)
        ellipses[curr_index] = eigenvalue
        
        
def getSMA(vrad, void_center, coords, periodic, cmin, cmax):
    """Convert tracers and void effective radius to ellipsoid semi-major axes.

    Parameters
    ----------
    vrad : float
        Void radius
    void_center : nfdarray
        The cooordinates of the void center
    coords : ndarray
        Array of void cell center coordinates.
    periodic: boolean
        Flag indicating periodic mode
    cmin: array
        Minimum coordinates of box
    cmin: array
        Maximum coordinates of box

    Returns
    -------
    sma : ndarray
        Ellipsoid semi-major axes for voids.
    """
    
    # Handle zones with < 3 cells
    # This can occur for cells near the edge of the survey mask whose neighbors are all discarded out-of-mask cells,
    # leaving an isolated cell or two cells as consituting a zone. These zones should not be present if an appropriate minimum
    # size cut is used for the voids
    if coords.shape[0] < 3:
        return np.full((3, 3), np.inf)
    
    if periodic:
        
        box_size = (cmax - cmin)

        transformed_coords = np.array(coords)

        transformed_coords = transformed_coords - coords[0] + box_size / 2

        transformed_coords = transformed_coords % box_size

        void_center = void_center - coords[0] + box_size / 2

        void_center = void_center % box_size

        transformed_coords = transformed_coords - void_center
    else:
        transformed_coords = np.array(coords) - void_center
    # tensor components
    comp_Ixx = np.sum(transformed_coords[:,[1,2]]**2)
    comp_Iyy = np.sum(transformed_coords[:,[0,2]]**2)
    comp_Izz = np.sum(transformed_coords[:,[0,1]]**2)
    comp_Ixy = -np.sum(np.prod(transformed_coords[:,[0,1]], axis=1))
    comp_Ixz = -np.sum(np.prod(transformed_coords[:,[0,2]], axis=1))
    comp_Iyz = -np.sum(np.prod(transformed_coords[:,[1,2]], axis=1))
    
    # tensor
    tensor_I = np.array([[comp_Ixx, comp_Ixy, comp_Ixz],[comp_Ixy, comp_Iyy, comp_Iyz],[comp_Ixz, comp_Iyz, comp_Izz]])

    # eigenvalues
    eival,eivec = np.linalg.eig(tensor_I)

    # principal axes of ellipsod
    a = np.sqrt(5/2 * ( - eival[0] + eival[1] + eival[2] ))
    b = np.sqrt(5/2 * (   eival[0] - eival[1] + eival[2] ))
    c = np.sqrt(5/2 * (   eival[0] + eival[1] - eival[2] ))

    eival[0] = a
    eival[1] = b
    eival[2] = c

    # normalize principal axes to unit ellipse and then scale to void size (factors of (4/3 pi)^(1/3) cancel out)
    eival =  vrad * eival/((np.prod(eival))**(1./3))

    # scale axes components (eigenvectors) by axes lengths
    return eival.reshape(3,1)*eivec.T


def P(r):
    """Calculate probability that void is fake.
    
    Parameters
    ----------
    r : float or ndarray
        Void radius or radii.

    Returns
    -------
    prob : float or ndarray
        Probability that void is fake.
    """
    return np.exp(-5.12*(r-1.) - 0.28*((r-1.)**2.8))


def flatten(l):
    """Recursively flattens a list.

    Parameters
    ----------
    l : list
        List to be flattened

    Returns
    -------
    """
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el,(str,bytes)):
            yield from flatten(el)
        else:
            yield el


def open_fits_file_V2(
        log_filename,
        method=None,
        out_directory=None, 
        survey_name=None):
    
    '''
    Reads in a fits file. If the file doesn't exist, a new file is created.
    
    
    PARAMETERS
    ==========
    
    log_filename : string
        The full path to the fits file. If None, then the path is created from
        out_directory and survey_name

    method : int
        0 = VIDE method (arXiv:1406.1191); link zones with density <1/5 mean density, and remove voids with density >1/5 mean density.
        1 = ZOBOV method (arXiv:0712.3049); keep full void hierarchy.
        2 = ZOBOV method; cut voids over a significance threshold.
        3 = not available
        4 = REVOLVER method (arXiv:1904.01030); every zone below mean density is a void.

    out_directory : string
        The folder containing the fits file. Only used if log_filename = None

    survey_name : string
        The name of the survey associated with the fits file. The name of the
        fits file will be (survey_name + '_V2_<method>_Output.fits'). Only used
        if log_filename = None
    
    RETURNS
    =======

    hdul : astropy fits object
        The fits file

    log_filename : string
        The full path to the fits file. Only returned if the log_filename
        input parameter is None

    '''
    # set the method name
    method_name = 'ZOBOV'
    if method == 0:
        method_name = 'VIDE'
    elif method == 4:
        method_name = 'REVOLVER'
    elif method == 5:
        method_name = 'REVOLVER2'
    
    # format directory and file name appropriately
    return_file_path = False
    if log_filename is None:

        return_file_path = True

        if len(out_directory) > 0 and out_directory[-1] != '/':
            out_directory += '/'

        if len(survey_name) > 0 and survey_name[-1] != '_':
            survey_name += '_'

        log_filename = out_directory + survey_name + f'V2_{method_name}_Output.fits'
        
    #create the output file if it doesn't already exist
    if not os.path.isfile(log_filename):
        hdul = fits.HDUList([fits.PrimaryHDU(header=fits.Header())])
        hdul.writeto(log_filename)

    #open the output file
    hdul = fits.open(log_filename)

    if return_file_path:
        return hdul, log_filename
    
    return hdul


# (Make Number) Format floats for headers
def mknumV2 (flt):
    """Formats a float for fits headers
    Parameters
    ----------
    flt : float
        float to be formatted
    Returns
    -------
    float
        Formatted float
    """
    if flt is None:
        return None

    #preserve 3 sig figs for numbers starting with "0."
    if abs(flt) < 1:
        return float(f"{flt:.3g}")
    #otherwise round to two decimal places
    else:
        return float(f"{flt:.2f}")
    

def rotate(p):
    """Rotates polygon into its plane.
    Parameters
    ----------
    p : ndarray
        Array of points making up polygon
    Returns
    -------
    r : ndarray
        Rotated array of points
    """
    p  = p-p[0]
    n1 = p[1]
    n2 = p[2]
    n3 = np.cross(n1,n2)
    n1 = n1/np.sqrt(np.sum(n1**2))
    n3 = n3/np.sqrt(np.sum(n3**2))
    n2 = np.cross(n3,n1)
    m = np.linalg.inv(np.array([n1,n2,n3]).T)
    r = np.matmul(m,p.T)[0:2].T
    return r


def partition_face_vertices(cell):
    """Obtains faces form a multivoro Cell object
    Parameters
    ----------
    cell : multivoro Cell
        Voronoi cell
    Returns
    -------
    faces : list of lists
        List of face coordinate indexes for each cell face
    """
    face_vertices = cell.get_face_vertices()
    faces = []
    start_index=0
    end_index = 0
    while end_index < len(face_vertices):
        num_vertices_in_face = face_vertices[start_index]
        start_index += 1
        end_index = start_index+num_vertices_in_face
        faces.append(face_vertices[start_index:end_index])
        start_index = end_index
    return faces


def galzone_worker(ngal,
                   index_coordinator,
                   zlist_file_descriptor,
                   elist_file_descriptor,
                   zcell,
                   glut,
                   volumes,
                   olist 
                   ):

    """Records zone IDs for the catalog galaxies in parallel.

    Parameters
    ----------
    num_voids : int
        The total number of voids in the catalog
    index_coordinator : multiprocessing.Value
        Index for coordinating void selection between parallel processes
    zlist_file_descriptor : int
        The file descriptor integer used to reference the shared memory for the parallel processes for the zone list
    elist_file_descriptor : int
        The file descriptor integer used to reference the shared memory for the parallel processes for the edge cell list
    zcell : list of list
        For each zone, the list of galaxy indexes belonging to it
    glut : ndarray or list of lists
        For each Voronoi cell, the list of galaxy indexes for galaxies found within it
    volumes : ndarray
        The Voronoi cell volumes
    olist: ndarray
        Boolean mask flagging galaxies that fall outside the survey mask

    """
    
    buffer_length = ngal*4 #int so 4 bytes per element

    buffer = mmap.mmap(zlist_file_descriptor, buffer_length)
    
    zlist = np.frombuffer(buffer, dtype=np.int32)

    zlist.shape = (ngal,)

    buffer = mmap.mmap(elist_file_descriptor, buffer_length)
    
    elist = np.frombuffer(buffer, dtype=np.int32)

    elist.shape = (ngal,)
    
    curr_index = 0
    
    while True:
        
        index_coordinator.acquire()
        
        curr_index = index_coordinator.value
        
        index_coordinator.value += 1
        
        index_coordinator.release()
    
        if curr_index >= len(zcell):
            break

        #each element of zcell is a zone, and the zone is a 
        #list of the galaxy indices belonging to that zone
        cl = zcell[curr_index]

        # glut transfers array index to galaxy ID
        # aka glut gives the indices of galaxies that make pre-tessellation cuts

        # for galaxy index c in zone cl
        for c in cl:
            # record the zone ID of the galaxy
            zlist[glut[c]] = curr_index
            # if galaxy is interior to survey (cell volume != 0) and is inside the mask
            if volumes[c]!=0. or olist[glut[c]].all():
                # mark as non-edge galaxy
                elist[glut[c]] = 0

def scale_volumes_by_randoms(tessellation, catalog, periodic, xyz, cmin, cmax):

    coords = catalog.coord[catalog.nnls==np.arange(len(catalog.nnls))] 

    if periodic or xyz:

        raise ValueError('Randoms are note supported for periodic or xyz mode.')

        """
        # untested code
        
        num_randoms = len(catalog.rand)
        sim_volume = np.prod(cmax - cmin)
        volume_per_random =  sim_volume / num_randoms
        randoms_grid_size = (1000*volume_per_random)

        weights_rand = catalog.weights_rand if hasattr(catalog, 'weights_rand') else None

        # place randoms on grid
        grid_randoms, _ = np.histogramdd(catalog.rand, 
                               bins=(int(np.ceil((cmax[0]-cmin[0])/randoms_grid_size)),
                                     int(np.ceil((cmax[1]-cmin[1])/randoms_grid_size)),
                                     int(np.ceil((cmax[2]-cmin[2])/randoms_grid_size))),
                               weights = weights_rand,
                                 )
    
        grid_norm = np.max(grid_randoms)
        grid_randoms = grid_randoms / grid_norm # setup for downweighting Voronoi cell volumes
        galaxy_grid_indices = np.floor((coords - cmin)/randoms_grid_size).astype(int) # indices of galaxies on grid
        randoms_multiplier = grid_randoms[galaxy_grid_indices[:,0], galaxy_grid_indices[:,1], galaxy_grid_indices[:,2]] #weights for each galaxy from randoms
        """            
    else: 

        num_gals = coords.shape[0]
        survey_volume = catalog.total_volume
        mean_galaxy_separation = (survey_volume / num_gals)**(1/3)
        
        dist = np.sqrt(coords[:,0]*coords[:,0] + coords[:,1]*coords[:,1] + coords[:,2]*coords[:,2])
    
        min_r = np.min(dist)
        max_r = np.max(dist)
        num_r_bins = int((max_r - min_r) / mean_galaxy_separation)
        r_bins = np.linspace(min_r, max_r, num_r_bins)
        print(f'Randoms: {num_r_bins} number density bins between {mknumV2(min_r)} and {mknumV2(max_r)} Mpc/h')
        
        weights_rand = None
            
        if hasattr(catalog, 'weights_rand'):

            weights_rand = catalog.weights_rand 
    
            inverse_weights = True # TODO: make user input (True for DESI LSS catalogs)
            
            if inverse_weights:
                zero_rand = weights_rand==0.
                if np.any(zero_rand):
                    print(f'WARNING: {np.sum(zero_rand)} out of {len(zero_rand)} randoms have a weight of 0. Reassigning weights to 1.')
                    weights_rand[zero_rand] = 1.
                weights_rand = 1 / weights_rand

        grid_norm = 0
        randoms_multiplier = np.zeros(num_gals)

        for r_bin_low, r_bin_high in zip(r_bins[:-1], r_bins[1:]):
            
            select_shell = (r_bin_low<=dist)*(r_bin_high>=dist)

            shell_volume = survey_volume * (r_bin_high**3 - r_bin_low**3) / (r_bins[-1]**3 - r_bins[0]**3)

            shell_number_density = np.sum(select_shell) / shell_volume

            randoms_grid_size = shell_number_density**(-1/3)

            print(f'Randoms grid size of cell length {mknumV2(randoms_grid_size)} from {mknumV2(r_bin_low)} to {mknumV2(r_bin_high)} Mpc/h')

            # place randoms on grid
            grid_randoms, _ = np.histogramdd(catalog.rand, 
                                   bins=(int(np.ceil((cmax[0]-cmin[0])/randoms_grid_size)),
                                         int(np.ceil((cmax[1]-cmin[1])/randoms_grid_size)),
                                         int(np.ceil((cmax[2]-cmin[2])/randoms_grid_size))),
                                   weights = weights_rand,
                                     )
        
            grid_norm = max(grid_norm, np.max(grid_randoms)/(randoms_grid_size**3)) #used for final normalization
            grid_randoms = grid_randoms / (randoms_grid_size**3) # setup for downweighting Voronoi cell volumes
            galaxy_grid_indices = np.floor((coords[select_shell] - cmin)/randoms_grid_size).astype(int) # indices of galaxies on grid
            randoms_multiplier[select_shell] = grid_randoms[galaxy_grid_indices[:,0], galaxy_grid_indices[:,1], galaxy_grid_indices[:,2]] #weights for each galaxy from randoms

        randoms_multiplier = randoms_multiplier / grid_norm # downweight Voronoi cell volumes

    finite_density = tessellation.volumes != 0.
        
    empty_randoms_cells = randoms_multiplier[finite_density]==0.
        
    if np.any(empty_randoms_cells):

        print(f'WARNING: {np.sum(empty_randoms_cells)} of {len(empty_randoms_cells)} galaxies detected without randoms in their grid cells. Their weights will be set to 1.')

        randoms_multiplier[randoms_multiplier==0.] = 1.
        
        #np.save('/global/homes/h/hrincon/BeyondDESIVAST/DESI_Project_543/VAST/empty_cells.npy', coords[finite_density][empty_randoms_cells])

    print (np.min(randoms_multiplier), np.max(randoms_multiplier), np.average(randoms_multiplier)) # Debugging
    print (np.min(randoms_multiplier[finite_density]), np.max(randoms_multiplier[finite_density]), np.average(randoms_multiplier[finite_density])) # Debugging

    tessellation.weights[finite_density] = tessellation.weights[finite_density] / randoms_multiplier[finite_density]
    #tessellation.volumes[finite_density] = tessellation.volumes[finite_density] * randoms_multiplier[finite_density]
        