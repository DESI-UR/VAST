import numpy as np
import healpy as hp
from astropy.io import fits
from scipy.spatial import ConvexHull, Voronoi, Delaunay, KDTree

from vast.vsquared.util import toCoord, flatten

class Catalog:
    def __init__(self,catfile,nside,zmin,zmax,maglim=None,H0=100,Om_m=0.3,maskfile=None):
        print("Extracting data...")
        hdulist = fits.open(catfile)
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
        nnls = np.arange(len(z))
        nnls[zcut<1] = -1
        if maglim is not None:
            mag  = hdulist[1].data['rabsmag']
            print("Applying magnitude cut...")
            mcut = np.logical_and(mag<maglim,zcut)
            if not mcut.any():
                print("Choose valid magnitude limit")
                return
            scut = mcut
            tree = KDTree(self.coord[mcut])
            nnls[zcut][mcut[zcut]<1] = tree.query(self.coord[zcut][mcut[zcut]<1])[1]
            self.mcut = mcut
        self.nnls = nnls
        if maskfile is None:
            print("Generating mask...")
            mask = np.zeros(hp.nside2npix(nside),dtype=bool)
            pids = hp.ang2pix(nside,ra[scut],dec[scut],lonlat=True)
            mask[pids] = True
        self.mask = mask
        pids = hp.ang2pix(nside,ra,dec,lonlat=True)
        self.imsk = mask[pids]*zcut

class Tesselation:
    def __init__(self,cat,viz=False):
        coords = cat.coord[cat.nnls==np.arange(len(cat.nnls))]
        print("Tesselating...")
        Vor = Voronoi(coords)
        ver = Vor.vertices
        reg = np.array(Vor.regions)[Vor.point_region]
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
        cu2 = np.array([np.product(np.logical_and(vrh[r]>rmn,vrh[r]<rmx),dtype=bool) for r in reg[cu1]])
        msk = cat.mask
        nsd = hp.npix2nside(len(msk))
        pid = hp.ang2pix(nsd,vth,vph)
        imk = msk[pid]
        cu3 = np.array([np.product(imk[r],dtype=bool) for r in reg[cu1][cu2]])
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
        self.neighbors = np.array(nei)
################################################################################





################################################################################
#-------------------------------------------------------------------------------
class Zones:
    def __init__(self,tess,viz=False):
        vol   = tess.volumes
        nei   = tess.neighbors

        ########################################################################
        # Sort the Voronoi cells by their volume
        #-----------------------------------------------------------------------
        print("Sorting cells...")

        srt   = np.argsort(-1.*vol)

        vol2  = vol[srt]
        nei2  = nei[srt]
        ########################################################################


        ########################################################################
        # Build zones from the cells
        #-----------------------------------------------------------------------
        lut   = np.zeros(len(vol), dtype=int)
        depth = np.zeros(len(vol), dtype=int)

        zvols = []
        zcell = []

        print("Building zones...")

        for i in range(len(vol)):

            ns = nei2[i]
            vs = vol[ns]
            n  = ns[np.argmax(vs)]

            if n == srt[i]:
                # This cell has the largest volume of its neighbors
                lut[n] = len(zvols)
                zcell.append([n])
                zvols.append(vol[n])
            else:
                # This cell is put into its least-dense neighbor's zone
                lut[srt[i]]   = lut[n]
                depth[srt[i]] = depth[n]+1
                zcell[lut[n]].append(srt[i])

        self.zcell = np.array(zcell)
        self.zvols = np.array(zvols)
        self.depth = depth
        ########################################################################


        ########################################################################
        # Identify neighboring zones and the least-dense cells linking them
        #-----------------------------------------------------------------------
        zlinks = [[[] for _ in range(len(zvols))] for _ in range(2)]

        if viz:
            zverts = [[] for _ in range(len(zvols))]
            znorms = [[] for _ in range(len(zvols))]

        print("Linking zones...")

        for i in range(len(vol)):
            ns = nei[i]
            z1 = lut[i]
            for n in ns:
                z2 = lut[n]
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
                            vcs = (tess.verts[vts].T[0:2]).T
                            zverts[z1].append((vts[ConvexHull(vcs).vertices]).tolist())
                            znorms[z1].append([i,n])
        self.zlinks = zlinks
        if viz:
            self.zverts = zverts
            self.znorms = znorms
        ########################################################################
################################################################################




################################################################################
#-------------------------------------------------------------------------------
class Voids:
    def __init__(self,zon):
        zvols  = np.array(zon.zvols)
        zlinks = zon.zlinks

        ########################################################################
        # Sort zone links by volume, identify zones linked at each volume
        #-----------------------------------------------------------------------
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

        ########################################################################
        # For each zone-linking by descending link volume, create void from     
        # all zones and groups of zones linked at this volume except for that   
        # with the highest maximum cell volume (the "shallower" voids flow into 
        # the "deepest" void with which they are linked)
        #-----------------------------------------------------------------------
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
