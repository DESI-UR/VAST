"""Utility classes for the ZOBOV algorithm using a voronoi tesselation of an
input catalog.
"""

import numpy as np
import healpy as hp
from astropy.io import fits
from scipy.spatial import ConvexHull, Voronoi, Delaunay, KDTree

from vast.vsquared.util import toCoord, getBuff, flatten

class Catalog:
    """Catalog data for void calculation.
    """

    def __init__(self,catfile,nside,zmin,zmax,maglim=None,H0=100,Om_m=0.3,periodic=False,cmin=None,cmax=None,maskfile=None):
        """Initialize catalog.

        Parameters
        ----------
        catfile: str
            Object catalog file (FITS format).
        nside : int
            HEALPix map nside parameter (2,4,8,16,...,2^k).
        zmin : float
            Minimum redshift boundary.
        zmax : float
            Maximum redshift boundary.
        maglim : float or None
            Catalog object magnitude limit.
        H0 : float
            Hubble parameter, in units of km/s/Mpc.
        Om_m : float
            Matter density.
        maskfile : str or None
            Mask file giving HEALPixels with catalog objects (FITS format).
        periodic : bool
            Use periodic boundary conditions.
        cmin : ndarray or None
            Array of coordinate minima.
        cmax : ndarray or None
            Array of coordinate maxima.
        """
        print("Extracting data...")
        hdulist = fits.open(catfile)
        if periodic:
            self.coord = np.array([hdulist[1].data['x'],hdulist[1].data['y'],hdulist[1].data['z']]).T
            self.cmin = cmin
            self.cmax = cmax
        else:
            z    = hdulist[1].data['z']
            ra   = hdulist[1].data['ra']
            dec  = hdulist[1].data['dec']
            zcut = np.logical_and(z>zmin,z<zmax)
            if not zcut.any():
                print("Choose valid redshift limits")
                return
            scut = zcut
            c1,c2,c3   = toCoord(z,ra,dec,H0,Om_m)
            self.coord = np.array([c1,c2,c3]).T
        nnls = np.arange(len(self.coord))
        if not periodic:
            nnls[zcut<1] = -1
        if maglim is not None:
            print("Applying magnitude cut...")
            mcut = np.logical_and(mag<maglim,zcut)
            if not mcut.any():
                print("Choose valid magnitude limit")
                return
            scut = mcut
            ncut = np.arange(len(self.coord),dtype=int)[zcut][mcut[zcut]<1]
            tree = KDTree(self.coord[mcut])
            lut  = np.arange(len(self.coord),dtype=int)[mcut]
            nnls[ncut] = lut[tree.query(self.coord[ncut])[1]]
            self.mcut = mcut
        self.nnls = nnls
        if not periodic:
            if maskfile is None:
                print("Generating mask...")
                mask = np.zeros(hp.nside2npix(nside),dtype=bool)
                pids = hp.ang2pix(nside,ra[scut],dec[scut],lonlat=True)
                mask[pids] = True
            else:
                mask = (hp.read_map(maskfile)).astype(bool)
            self.mask = mask
            pids = hp.ang2pix(nside,ra,dec,lonlat=True)
            self.imsk = mask[pids]*zcut
        try:
            self.galids = hdulist[1].data['ID']
        except:
            self.galids = np.arange(len(z))

class Tesselation:
    """Implementation of Voronoi tesselation of the catalog.
    """

    def __init__(self,cat,viz=False,periodic=False,buff=5.):
        """Initialize tesselation.

        Parameters
        ----------
        cat : Catalog
            Catalog of objects used to compute the Voronoi tesselation.
        viz : bool
            Compute visualization.
        periodic : bool
            Use periodic boundary conditions.
        buff : float
            Width of incremental buffer shells for periodic computation.
        """
        coords = cat.coord[cat.nnls==np.arange(len(cat.nnls))]
        if periodic:
            print("Triangulating...")
            Del = Delaunay(coords,incremental=True,qhull_options='QJ')
            sim = Del.simplices
            simlen = len(sim)
            cids = np.arange(len(coords))
            print("Finding periodic neighbors...")
            n = 0
            coords2,cids = getBuff(coords,cids,cat.cmin,cat.cmax,buff,n)
            coords3 = coords.tolist()
            coords3.extend(coords2)
            Del.add_points(coords2)
            sim = Del.simplices
            while np.amin(sim[simlen:])<len(coords):
                n = n+1
                simlen = len(sim)
                coords2,cids = getBuff(coords,cids,cat.cmin,cat.cmax,buff,n)
                coords3.extend(coords2)
                Del.add_points(coords2)
                sim = Del.simplices
            for i in range(len(sim)):
                sim[i] = cids[sim[i]]
            print("Tesselating...")
            Vor = Voronoi(coords3)
            ver = Vor.vertices
            reg = np.array(Vor.regions)[Vor.point_region]
            del Vor
            print("Computing volumes...")
            vol = np.zeros(len(coords))
            cut = np.arange(len(coords))
            hul = []
            for r in reg[cut]:
                try:
                    ch = ConvexHull(ver[r])
                except:
                    ch = ConvexHull(ver[r],qhull_options='QJ')
                hul.append(ch)
            #hul = [ConvexHull(ver[r]) for r in reg[cut]]
            vol[cut] = np.array([h.volume for h in hul])
            self.volumes = vol
        else:
            print("Tesselating...")
            Vor = Voronoi(coords)
            ver = Vor.vertices
            
            reg = np.array(Vor.regions, dtype=object)[Vor.point_region]
            
            
            
            
            
            del Vor
            ve2 = ver.T
            vth = np.arctan2(np.sqrt(ve2[0]**2.+ve2[1]**2.),ve2[2])
            vph = np.arctan2(ve2[1],ve2[0])
            vrh = np.array([np.sqrt((v**2.).sum()) for v in ver])
            crh = np.array([np.sqrt((c**2.).sum()) for c in coords])
            rmx = np.amax(crh)
            rmn = np.amin(crh)
            print("Computing volumes...")
            vol = np.zeros(len(reg))
            cu1 = np.array([-1 not in r for r in reg])
            cu2 = np.array([np.product(np.logical_and(vrh[r]>rmn,vrh[r]<rmx),dtype=bool) for r in reg[cu1]]).astype(bool)
            msk = cat.mask
            nsd = hp.npix2nside(len(msk))
            pid = hp.ang2pix(nsd,vth,vph)
            imk = msk[pid]
            cu3 = np.array([np.product(imk[r],dtype=bool) for r in reg[cu1][cu2]]).astype(bool)
            cut = np.arange(len(vol))
            cut = cut[cu1][cu2][cu3]
            hul = []
            for r in reg[cut]:
                try:
                    ch = ConvexHull(ver[r])
                except:
                    ch = ConvexHull(ver[r],qhull_options='QJ')
                hul.append(ch)
            #hul = [ConvexHull(ver[r]) for r in reg[cut]]
            vol[cut] = np.array([h.volume for h in hul])
            self.volumes = vol
            if viz:
                self.vertIDs = reg
                vecut = np.zeros(len(vol),dtype=bool)
                vecut[cut] = True
                self.vecut = vecut
                self.verts = ver
            print("Triangulating...")
            Del = Delaunay(coords,qhull_options='QJ')
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
        self.neighbors = np.array(nei, dtype=object)


class Zones:
    """Partitioning of particles into zones around density minima.
    """

    def __init__(self,tess,viz=False):
        """Implementation of zones: see arXiv:0712.3049 for details.

        Parameters
        ----------
        tess : Tesselation
            Voronoid tesselation of an object catalog.
        viz : bool
            Compute visualization.
        """
        vol   = tess.volumes
        nei   = tess.neighbors

        # Sort the Voronoi cells by their volume
        print("Sorting cells...")

        srt   = np.argsort(-1.*vol)

        vol2  = vol[srt]
        nei2  = nei[srt]

        # Build zones from the cells
        lut   = np.zeros(len(vol), dtype=int)
        depth = np.zeros(len(vol), dtype=int)

        zvols = [0.]
        zcell = [[]]
        zceln = []

        print("Building zones...")

        for i in range(len(vol)):

            if vol2[i] == 0.:
                lut[srt[i]] = -1
                zcell[-1].append(srt[i])
                continue

            ns = nei2[i]
            vs = vol[ns]
            n  = ns[np.argmax(vs)]

            if n == srt[i]:
                # This cell has the largest volume of its neighbors
                lut[n] = len(zvols) - 1
                zcell.insert(-1,[n])
                zvols.insert(-1,vol[n])
                zceln.append([n])
            else:
                # This cell is put into its least-dense neighbor's zone
                lut[srt[i]]   = lut[n]
                depth[srt[i]] = depth[n]+1
                zcell[lut[n]].append(srt[i])
                if len(zceln[lut[n]])<4:
                    for j,o in enumerate(zceln[lut[n]]):
                        if o in ns:
                            if j+1==len(zceln[lut[n]]):
                                zceln[lut[n]].append(srt[i])
                            else:
                                continue
                        else:
                            break

        self.zcell = np.array(zcell, dtype=object)
        self.zvols = np.array(zvols)
        self.depth = depth

        # Identify neighboring zones and the least-dense cells linking them
        zlinks = [[[] for _ in range(len(zvols))] for _ in range(2)]

        if viz:
            zverts = [[] for _ in range(len(zvols))]
            znorms = [[] for _ in range(len(zvols))]

        print("Linking zones...")

        for i in range(len(vol)):
            ns = nei[i]
            z1 = lut[i]
            if z1 == -1:
                continue
            for n in ns:
                z2 = lut[n]
                if z2 == -1:
                    continue
                if z1 != z2:
                    # This neighboring cell is in a different zone
                    if z2 not in zlinks[0][z1]:
                        zlinks[0][z1].append(z2)
                        zlinks[0][z2].append(z1)
                        zlinks[1][z1].append(0.)
                        zlinks[1][z2].append(0.)
                    j  = np.where(zlinks[0][z1] == z2)[0][0]
                    k  = np.where(zlinks[0][z2] == z1)[0][0]
                    # Update maximum link volume if needed
                    nl = np.amin([vol[i],vol[n]])
                    ml = np.amax([zlinks[1][z1][j],nl])
                    zlinks[1][z1][j] = ml
                    zlinks[1][z2][k] = ml
                    if viz and tess.vecut[i]:
                        vts = tess.vertIDs[i].copy()
                        vts.extend(tess.vertIDs[n])
                        vts = np.array(vts)
                        vts = vts[[len(vts[vts==v])==2 for v in vts]]
                        #vts = np.unique(vts[vts!=-1])
                        if len(vts)>2:
                            try:
                                vcs = (tess.verts[vts].T[0:2]).T
                                zverts[z1].append((vts[ConvexHull(vcs).vertices]).tolist())
                            except:
                                vcs = (tess.verts[vts].T[1:3]).T
                                zverts[z1].append((vts[ConvexHull(vcs).vertices]).tolist())
                            znorms[z1].append([i,n])
        self.zlinks = zlinks
        if viz:
            self.zverts = zverts
            self.znorms = znorms


class Voids:
    """Calculation of voids using a set of minimum-density zones.
    """

    def __init__(self,zon):
        """Implementation of void calculation: see arXiv:0712.3049.

        Parameters
        ----------
        zon: Zones
            A group of zones around density minima in an input catalog.
        """
        zvols  = np.array(zon.zvols)
        zlinks = zon.zlinks

        # Sort zone links by volume, identify zones linked at each volume
        print("Sorting links...")

        zl0   = np.array(list(flatten(zlinks[0])))
        zl1   = np.array(list(flatten(zlinks[1])))

        zlu   = -1.*np.sort(-1.*np.unique(zl1))
        zlut  = [np.unique(zl0[np.where(zl1==zl)[0]]).tolist() for zl in zlu]

        voids = []
        mvols = []
        ovols = []
        vlut  = np.arange(len(zvols))
        mvlut = np.array(zvols)
        ovlut = np.array(zvols)

        # For each zone-linking by descending link volume, create void from     
        # all zones and groups of zones linked at this volume except for that   
        # with the highest maximum cell volume (the "shallower" voids flow into 
        # the "deepest" void with which they are linked)
        print("Expanding voids...")

        for i in range(len(zlu)):
            lvol  = zlu[i]
            mxvls = mvlut[zlut[i]]
            mvarg = np.argmax(mxvls)
            mxvol = mxvls[mvarg]
            for j in zlut[i]:
                if mvlut[j] < mxvol:
                    # This is not the "deepest" zone or void being linked
                    voids.append([])
                    ovols.append([])
                    vcomp = np.where(vlut==vlut[j])[0]
                    # Store void's "overflow" volumes, largest max cell volume, constituent zones
                    for ov in -1.*np.sort(-1.*np.unique(ovlut[vcomp])):
                        ocomp = np.where(ovlut[vcomp]==ov)[0]
                        voids[-1].append(vcomp[ocomp].tolist())
                        ovols[-1].append(ov)
                    ovols[-1].append(lvol)
                    mvols.append(mvlut[j])
                    vlut[vcomp]  = vlut[zlut[i]][mvarg]
                    mvlut[vcomp] = mxvol
                    ovlut[vcomp] = lvol
        
        # Include the "deepest" void in the survey and its subvoids
        voids.append([])
        ovols.append([])
        for ov in -1.*np.sort(-1.*np.unique(ovlut)):
            ocomp = np.where(ovlut==ov)[0]
            voids[-1].append(ocomp.tolist())
            ovols[-1].append(ov)
        ovols[-1].append(0.)
        mvols.append(mvlut[0])

        self.voids = voids
        self.mvols = mvols
        self.ovols = ovols
