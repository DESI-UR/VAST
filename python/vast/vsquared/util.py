import numpy as np
from collections.abc import Iterable
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy import interpolate
from astropy.io import fits
import os

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


def inSphere(cs, r, coords):
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

    Returns
    =======

    inSphere : bool
        True if abs(coords - cs) < r.
    """
    return np.sum((cs.reshape(3,1) - coords.T)**2., axis=0)<r**2.


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


def wCen(vols,coords):
    """Find the weighted center of tracers' Voronoi cells.

    Parameters
    ----------
    vols : ndarray
        Array of Voronoi volumes.
    coords : ndarray
        Array of cells' positions.

    Returns
    -------
    wCen : ndarray
        Weighted center of tracers' Voronoi cells.
    """
    return np.sum(vols.reshape(len(vols),1)*coords,axis=0)/np.sum(vols)


def getSMA(vrad,coords):
    """Convert tracers and void effective radius to ellipsoid semi-major axes.

    Parameters
    ----------
    vrad : ndarray
        List of void radii.
    coords : ndarray
        Array of void coordinates.

    Returns
    -------
    sma : ndarray
        Ellipsoid semi-major axes for voids.
    """
    iTen = np.zeros((3,3))
    for p in coords:
        iTen = iTen + np.array([[p[1]**2.+p[2]**2.,0,0],[0,p[0]**2.+p[2]**2.,0],[0,0,p[0]**2.+p[1]**2.]])
        iTen = iTen - np.array([[0,p[0]*p[1],p[0]*p[2]],[p[0]*p[1],0,p[1]*p[2]],[p[0]*p[2],p[1]*p[2],0]])
    eival,eivec = np.linalg.eig(iTen)
    eival = eival**.25
    rfac = vrad/(np.prod(eival)**(1./3))
    eival = eival*rfac
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