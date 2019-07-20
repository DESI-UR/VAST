import numpy as np
import collections
from astropy.cosmology import FlatLambdaCDM

H0   = 100
Om_m = 0.3
c    = 3e6

#changes to comoving coordinates
def toCoord(z,ra,dec):
    D2R = np.pi/180.
    Kos = FlatLambdaCDM(H0,Om_m)
    r = Kos.comoving_distance(z)
    r = np.array([d.value for d in r])
    #r = c*z/H0
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    return c1,c2,c3

def inSphere(cs,r,coords):
    inx = np.logical_and(coords[0]>cs[0]-r,coords[0]<cs[0]+r)
    iny = np.logical_and(coords[1]>cs[1]-r,coords[1]<cs[1]+r)
    inz = np.logical_and(coords[2]>cs[2]-r,coords[2]<cs[2]+r)
    inc = np.product([inx,iny,inz],dtype=bool,axis=0)
    inS = np.sqrt(np.sum((cs.reshape(len(inc),1)-coords.T[inc].T)**2.,axis=1))>r
    inc[inS] = False
    return inc

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
