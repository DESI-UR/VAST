import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import ConvexHull, Voronoi, Delaunay

#tess  = Tesselation(infile)
#zones = Zones(tess)
#voids = Voids(zones)

class Tesselation:
    def __init__(self,fname):
        hdulist = fits.open(fname)
        print("Extracting coordinates...")
        z   = hdulist[1].data['z']
        ra  = hdulist[1].data['ra']
        dec = hdulist[1].data['dec']
        c1,c2,c3 = toCoord(z,ra,dec)
        coords = np.array([c1,c2,c3]).T
        print("Tesselating...")
        Vor = Voronoi(coords)
        print("Triangulating...")
        Del = Delaunay(coords)
        ver = Vor.vertices
        reg = np.array(Vor.regions)[Vor.point_region]
        print("Computing volumes...")
        vol = np.zeros(len(reg))
        cut = np.array([-1 not in r for r in reg])
        hul = [ConvexHull(ver[r]) for r in reg[cut]]
        vol[cut] = np.array([h.volume for h in hul])
        self.volumes = vol
        sim = Del.simplices
        nei = []
        lut = [[] for _ in range(len(vol))]
        print("Consolidating neighbors...")
        for i in range(len(sim)):
            for j in sim[i]:
                lut[j].append(i)
        for i in range(len(vol)):
            cut = np.array(lut[i])
            nei.append(np.unique(sim[cut]))
        self.neighbors = np.array(nei)

class Zones:
    def __init__(self,tess):
        vol   = tess.volumes
        nei   = tess.neighbors
        print("Sorting cells...")
        srt   = np.argsort(-1.*vol)        
        vol2  = vol[srt]
        nei2  = nei[srt]
        lut   = np.zeros(len(vol),dtype=int)
        zones = []
        zcell = []
        print("Building zones...")
        for i in range(len(vol)):
            ns = nei2[i]
            vs = vol[ns]
            n  = ns[np.argmax(vs)]
            if n == srt[i]:
                lut[n] = len(zones)
                zcell.append([n])
                zones.append(vol[n])
            else:
                lut[srt[i]] = lut[n]
                zcell[lut[n]].append(srt[i])
        self.zcell = np.array(zcell)
        self.zones = np.array(zones)
        zlinks = [[[] for _ in range(len(zones))] for _ in range(2)]       
        print("Linking zones...")
        for i in range(len(vol)):
            ns = nei[i]
            z1 = lut[i]
            for n in ns:
                z2 = lut[n]
                if z1 != z2:
                    if z2 not in zlinks[0][z1]:
                        zlinks[0][z1].append(z2)
                        zlinks[0][z2].append(z1)
                        zlinks[1][z1].append(0.)
                        zlinks[1][z2].append(0.)
                    j  = np.where(zlinks[0][z1] == z2)[0][0]
                    k  = np.where(zlinks[0][z2] == z1)[0][0]
                    ml = np.amax([zlinks[1][z1][j],vol[i],vol[n]])
                    zlinks[1][z1][j] = ml
                    zlinks[1][z2][k] = ml
        self.zlinks = zlinks

class Voids:
    def __init__(self,zon):

#change to cosmology-independent coordinates
def toCoord(z,ra,dec):
    D2R = np.pi/180.
    Kos = FlatLambdaCDM(100,0.3)
    r = Kos.comoving_distance(z)
    r = np.array([d.value for d in r])
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    return c1,c2,c3
