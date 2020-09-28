import numpy as np
import collections
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy import interpolate

c    = 3e5
D2R  = np.pi/180.

#changes to comoving coordinates
def toCoord(z,ra,dec,H0,Om_m):
    Kos = FlatLambdaCDM(H0,Om_m)
    r = Kos.comoving_distance(z)
    r = np.array([d.value for d in r])
    #r = c*z/H0
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    return c1,c2,c3

#changes to sky coordinates
def toSky(cs,H0,Om_m,zstep):
    Kos = FlatLambdaCDM(H0,Om_m)
    c1 = cs.T[0]
    c2 = cs.T[1]
    c3 = cs.T[2]
    r   = np.sqrt(c1**2.+c2**2.+c3**2.)
    dec = np.arcsin(c3/r)/D2R
    ra  = (np.arccos(c1/np.sqrt(c1**2.+c2**2.))*np.sign(c2)/D2R)%360
    zmn = z_at_value(Kos.comoving_distance,np.amin(r)*u.Mpc)
    zmx = z_at_value(Kos.comoving_distance,np.amax(r)*u.Mpc)
    zmn = zmn-(zstep+zmn%zstep)
    zmx = zmx+(2*zstep-zmx%zstep)
    ct  = np.array([np.linspace(zmn,zmx,int(np.ceil(zmn/zstep))),Kos.comoving_distance(np.linspace(zmn,zmx,int(np.ceil(zmn/zstep)))).value]).T
    r2z = interpolate.pchip(*ct[:,::-1].T)
    z = r2z(r)
    #z = H0*r/c
    return z,ra,dec

#checks which points are within a sphere
def inSphere(cs,r,coords):
    return np.sum((cs.reshape(3,1)-coords.T)**2.,axis=0)<r**2.

#finds the weighted center of tracers' Voronoi cells
def wCen(vols,coords):
    return np.sum(vols.reshape(len(vols),1)*coords,axis=0)/np.sum(vols)

#converts tracers and void effective radius to ellipsoid semi-major axes
def getSMA(vrad,coords):
    iTen = np.zeros((3,3))
    for p in coords:
        iTen = iTen + np.array([[p[1]**2.+p[2]**2.,0,0],[0,p[0]**2.+p[2]**2.,0],[0,0,p[0]**2.+p[1]**2.]])
        iTen = iTen - np.array([[0,p[0]*p[1],p[0]*p[2]],[p[0]*p[1],0,p[1]*p[2]],[p[0]*p[2],p[1]*p[2],0]])
    eival,eivec = np.linalg.eig(iTen)
    eival = eival**.25
    rfac = vrad/(np.product(eival)**(1./3))
    eival = eival*rfac
    return eival.reshape(3,1)*eivec.T

#probability a void is fake
def P(r):
    return np.exp(-5.12*(r-1.) - 0.28*((r-1.)**2.8))


################################################################################
# Flattens a list
#-------------------------------------------------------------------------------
def flatten(l):
    '''
    Recursivley flattens a list


    PARAMETERS
    ==========

    l : list
        List to be flattened


    RETURNS
    =======


    '''
    for el in l:
        if isinstance(el,collections.Iterable) and not isinstance(el,(str,bytes)):
            yield from flatten(el)
        else:
            yield el
