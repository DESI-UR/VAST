import numpy as np
import collections
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy import interpolate

H0   = 100
Om_m = 0.3
c    = 3e6
zstp = 5e-5
D2R  = np.pi/180.

Kos = FlatLambdaCDM(H0,Om_m)

#changes to comoving coordinates
def toCoord(z,ra,dec):
    r = Kos.comoving_distance(z)
    r = np.array([d.value for d in r])
    #r = c*z/H0
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    return c1,c2,c3

#changes to sky coordinates
def toSky(cs):
    c1 = cs.T[0]
    c2 = cs.T[1]
    c3 = cs.T[2]
    r   = np.sqrt(c1**2.+c2**2.+c3**2.)
    dec = np.arcsin(c3/r)/D2R
    ra  = np.arccos(c1/np.sqrt(c1**2.+c2**2.))/D2R
    zmn = z_at_value(Kos.comoving_distance,np.amin(r)*u.Mpc)
    zmx = z_at_value(Kos.comoving_distance,np.amax(r)*u.Mpc)
    zmn = zmn-(zstp+zmn%zstp)
    zmx = zmx+(2*zstp-zmx%zstp)
    ct  = np.array([np.linspace(zmn,zmx,int(np.ceil(zmn/zstp))),Kos.comoving_distance(np.linspace(zmn,zmx,int(np.ceil(zmn/zstp)))).value]).T
    r2z = interpolate.pchip(*ct[:,::-1].T)
    z = r2z(r)
    return z,ra,dec

#checks which points are within a sphere
def inSphere(cs,r,coords):
    return np.sum((cs.reshape(3,1)-coords.T)**2.,axis=0)<r**2.

#finds the weighted center of tracers' Voronoi cells
def wCen(vols,coords):
    return np.sum(vols.reshape(len(vols),1)*coords,axis=0)/np.sum(vols)

#probability a void is fake
def P(r):
    return np.exp(-5.12*(r-1.) - 0.28*((r-1.)**2.8))

#flattens a list
def flatten(l):
    for el in l:
        if isinstance(el,collections.Iterable) and not isinstance(el,(str,bytes)):
            yield from flatten(el)
        else:
            yield el
