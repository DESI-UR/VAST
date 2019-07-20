import numpy as np
import pickle
from scipy import stats

from util import inSphere, P
from classes import Catalog, Tesselation, Zones, Voids

infile  = "./data/vollim_dr7_cbp_102709.fits"
catname = "DR7"
intloc  = "./intermediate/" + catname
nside   = 64
denscut = 0.2

class Zobov:
    def __init__(self,start=0,end=3,save_intermediate=True):
        if start not in [0,1,2,3,4] or end not in [0,1,2,3,4] or end<start:
            print("Choose valid stages")
            return
        if start<4:
            if start<3:
                if start<2:
                    if start<1:
                        ctlg = Catalog(infile,nside)
                        if save_intermediate:
                            pickle.dump(ctlg,open(intloc+"_ctlg.pkl",'wb'))
                    else:
                        ctlg = pickle.load(open(intloc+"_ctlg.pkl",'rb'))
                    if end>0:
                        tess = Tesselation(ctlg)
                        if save_intermediate:
                            pickle.dump(tess,open(intloc+"_tess.pkl",'wb'))
                else:
                    ctlg = pickle.load(open(intloc+"_ctlg.pkl",'rb'))
                    tess = pickle.load(open(intloc+"_tess.pkl",'rb'))
                if end>1:
                    zones = Zones(tess)
                    if save_intermediate:
                        pickle.dump(zones,open(intloc+"_zones.pkl",'wb'))
            else:
                ctlg  = pickle.load(open(intloc+"_ctlg.pkl",'rb'))
                tess  = pickle.load(open(intloc+"_tess.pkl",'rb'))
                zones = pickle.load(open(intloc+"_zones.pkl",'rb'))
            if end>2:
                voids = Voids(zones)
                if save_intermediate:
                    pickle.dump(voids,open(intloc+"_voids.pkl",'wb'))
        else:
            ctlg  = pickle.load(open(intloc+"_ctlg.pkl",'rb'))
            tess  = pickle.load(open(intloc+"_tess.pkl",'rb'))
            zones = pickle.load(open(intloc+"_zones.pkl",'rb'))
            voids = pickle.load(open(intloc+"_voids.pkl",'rb'))
        self.catalog     = ctlg
        self.tesselation = tess
        self.zones       = zones
        self.prevoids    = voids
    def sortVoids(self,method=0,minsig=2,dc=denscut):
        if not hasattr(self,'prevoids'):
            print("Run all stages of Zobov first")
            return
        if method==0:
            voids  = []
            minvol = np.mean(self.tesselation.volumes[self.tesselation.volumes>0])/dc
            for i in range(len(self.prevoids.ovols)):
                vl = self.prevoids.ovols[i][-1]
                if vl < minvol:
                    break
                voids.append([c for q in self.prevoids.voids[i] for c in q])
        elif method==1:
            voids = [[c for q in v for c in q] for v in self.prevoids.voids]
        elif method==2:
            voids = []
            for i in range(len(self.prevoids.mvols)):
                vh = self.prevoids.mvols[i]
                vl = self.prevoids.ovols[i][-1]
                r  = vh / vl
                p  = P(r)
                if stats.norm.isf(p/2.) >= minsig:
                    voids.append([c for q in self.prevoids.voids[i] for c in q])
        elif method==3:
            print("Coming soon")
        else:
            print("Choose a valid method")
        return voids
