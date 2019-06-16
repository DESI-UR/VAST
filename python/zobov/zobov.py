import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import ConvexHull, Voronoi, Delaunay
from scipy import stats
import collections

#tess  = Tesselation(infile)
#zones = Zones(tess)
#voids = Voids(zones)
#voids_sorted = voids.vSort()

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
        del Vor
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
        zvols = []
        zcell = []
        print("Building zones...")
        for i in range(len(vol)):
            ns = nei2[i]
            vs = vol[ns]
            n  = ns[np.argmax(vs)]
            if n == srt[i]:
                lut[n] = len(zvols)
                zcell.append([n])
                zvols.append(vol[n])
            else:
                lut[srt[i]] = lut[n]
                zcell[lut[n]].append(srt[i])
        self.zcell = np.array(zcell)
        self.zvols = np.array(zvols)
        zlinks = [[[] for _ in range(len(zvols))] for _ in range(2)]       
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
                    nl = np.amin([vol[i],vol[n]])
                    ml = np.amax([zlinks[1][z1][j],nl])
                    zlinks[1][z1][j] = ml
                    zlinks[1][z2][k] = ml
        self.zlinks = zlinks

class Voids:
    def __init__(self,zon):
        zvols  = np.array(zon.zvols)
        zlinks = zon.zlinks
        print("Sorting links...")
        zl1   = np.array(list(flatten(zlinks[1])))
        zlu   = -1.*np.sort(-1.*np.unique(zl1))
        zl0   = np.array(list(flatten(zlinks[0])))
        zlut  = [np.unique(zl0[np.where(zl1==zl)[0]]).tolist() for zl in zlu]
        voids = []
        mvols = []
        ovols = []
        vlut  = np.array(range(len(zvols)))
        mvlut = np.array(zvols)
        ovlut = np.array(zvols)
        print("Expanding voids...")
        for i in range(len(zlu)):
            lvol  = zlu[i]
            mxvls = mvlut[zlut[i]]
            mvarg = np.argmax(mxvls)
            mxvol = mxvls[mvarg]
            for j in zlut[i]:
                if mvlut[j] < mxvol:                
                    voids.append([])
                    ovols.append([])
                    vcomp = np.where(vlut==vlut[j])[0]
                    for ov in -1.*np.sort(-1.*np.unique(ovlut[vcomp])):
                        ocomp = np.where(ovlut[vcomp]==ov)[0]
                        voids[-1].append(vcomp[ocomp].tolist())
                        ovols[-1].append(ov)
                    ovols[-1].append(lvol)
                    mvols.append(mvlut[j])
                    vlut[vcomp]  = vlut[zlut[i]][mvarg]
                    mvlut[vcomp] = mxvol
                    ovlut[vcomp] = lvol
        self.voids = voids
        self.mvols = mvols
        self.ovols = ovols
    def vSort(self,method=2,minsig=2):
        if method==1:
            voids = [[c for q in v for c in q] for v in self.voids]
        elif method==2:
            voids = []
            for i in range(len(self.mvols)):
                vh = self.mvols[i]
                vl = self.ovols[i][-1]
                r  = vh / vl
                p  = P(r)
                if stats.norm.isf(p/2.) >= minsig:
                    voids.append([c for q in self.voids[i] for c in q])
        elif method==3:
            print("Coming soon")
        else:
            print("Choose a valid method")
        return voids

#To Do: change to cosmology-independent coordinates
def toCoord(z,ra,dec):
    D2R = np.pi/180.
    Kos = FlatLambdaCDM(100,0.3)
    r = Kos.comoving_distance(z)
    r = np.array([d.value for d in r])
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    return c1,c2,c3

#probability a void is fake
def P(r):
    return np.exp(-5.12*(r-1.) - 0.28*((r-1.)**2.8))

def flatten(l):
    for el in l:
        if isinstance(el,collections.Iterable) and not isinstance(el,(str,bytes)):
            yield from flatten(el)
        else:
            yield el
