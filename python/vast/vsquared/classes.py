"""Utility classes for the ZOBOV algorithm using a voronoi tesselation of an
input catalog.
"""

import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import ConvexHull, Voronoi, Delaunay, KDTree

from vast.vsquared.util import toCoord, getBuff, flatten, mknumV2, rotate
from vast.voidfinder.preprocessing import load_data_to_Table

class Catalog:
    """Catalog data for void calculation.
    """

    def __init__(self,catfile,nside,zmin,zmax,column_names,maglim=None,H0=100,Om_m=0.3,periodic=False,xyz=False,cmin=None,cmax=None,maskfile=None,zobov=None):
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
        column_names : str
            'Galaxy Column Names' section of configuration file, in INI format
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
        xyz : bool
            Use rectangular boundary conditions.
        cmin : ndarray or None
            Array of coordinate minima.
        cmax : ndarray or None
            Array of coordinate maxima.
        """
        print("Extracting data...")
        
        # read in galaxy file
        galaxy_table = load_data_to_Table(catfile)

        # For periodic mode, store the galaxy coordinates, the minimum coordinates,and the maximum coordinates
        if periodic or xyz:
            self.coord = np.array([galaxy_table[column_names['x']],
                                   galaxy_table[column_names['y']],
                                   galaxy_table[column_names['z']]]).T
            self.cmin = cmin
            self.cmax = cmax
        
        # For ra-dec-z mode, convert sky coodinates to cartesian coordinates
        else:
            z    = galaxy_table[column_names['redshift']]
            ra   = galaxy_table[column_names['ra']]
            dec  = galaxy_table[column_names['dec']]
            zcut = np.logical_and(z>zmin,z<zmax)
            if not zcut.any():
                print("Choose valid redshift limits")
                return
            scut = zcut #alias for zcut, unless magnitude limit is used, in which case scut will be later set to mcut
            c1,c2,c3   = toCoord(z,ra,dec,H0,Om_m)
            self.coord = np.array([c1,c2,c3]).T
        
        # Array that will hold -1 for galaxies outside z limits, nearest neighbor galaxy in magcut for remaining galaxies not in magcut, and self identifier for further remaining galaxies
        nnls = np.arange(len(self.coord)) 
        
        # Galaxies outside z limit are marked with -1
        if not periodic and not xyz:
            nnls[zcut<1] = -1

        # Apply magnitude limit
        if maglim is not None:
            print("Applying magnitude cut...")
            mag = galaxy_table[column_names['rabsmag']]
            mcut = np.logical_and(mag<maglim,zcut) # mcut is a subsample of zcut that removes galaxies outside the magnitude limit
            if not mcut.any():
                print("Choose valid magnitude limit")
                return
            scut = mcut # scut is made into an alias for mcut, unless no magnirtude limitis used, in hich case it remains an alias for zcut
            ncut = np.arange(len(self.coord),dtype=int)[zcut][mcut[zcut]<1]  # indexes of galaxies in zcut but not in mcut
            tree = KDTree(self.coord[mcut]) #kdtree of galaxies in mcut
            lut  = np.arange(len(self.coord),dtype=int)[mcut] #indexes of galaxies in mcut
            nnls[ncut] = lut[tree.query(self.coord[ncut])[1]] # the nearest neighbor index for each galaxy in zcut but not in mcut, and where the neighbors are in mcut
            self.mcut = mcut
        
        self.nnls = nnls

        # Apply survey mask
        self.imsk = np.ones(len(nnls),dtype=bool) #simply selects all galaxies for periodic mode, see def. below otherwise
        if not periodic and not xyz:
            if maskfile is None:
                print("Generating mask...")
                mask = np.zeros(hp.nside2npix(nside),dtype=bool) #create a healpix mask with specified nside
                pids = hp.ang2pix(nside,ra[scut],dec[scut],lonlat=True) #convert scut galaxy coordinates to mask coordinates
                mask[pids] = True #mark where galaxies fall in mask
            else:
                mask = (hp.read_map(maskfile)).astype(bool)
            self.mask = mask #mask of all galaxies in scut, where scut might be zcut or mcut depending on if magnitude cut is used
            pids = hp.ang2pix(nside,ra,dec,lonlat=True) #convert all galaxy coordinates to mask coordinates
            # mask pixel bool value for every galaxy (aka is galaxy in same mask bin as a galaxy in scut), multiplied by zcut
            # this is used to select galaxies located outside the survey mask in the 'out' column of the galzones HDU
            self.imsk = mask[pids]*zcut 

            #record mask information
            maskHDU = fits.ImageHDU(mask.astype(int))
            maskHDU.name = 'MASK'

            coverage = np.sum(mask) * hp.nside2pixarea(nside) #solid angle coverage in steradians
            coverage_deg = coverage*(180/np.pi)**2 #solid angle coverage in deg^2
            maskHDU.header['COVSTR'] = (mknumV2(coverage), 'Sky Coverage (Steradians)')
            zobov.hdu.header['COVSTR'] = (mknumV2(coverage), 'Sky Coverage (Steradians)')
            zobov.hdu.header['COVDEG'] = (mknumV2(coverage_deg), 'Sky Coverage (Degrees^2)')

            d_max = zobov.hdu.header['DLIMU']
            d_min = zobov.hdu.header['DLIML']
            vol = coverage / 3 * (d_max ** 3 - d_min ** 3) # volume calculation (A sphere subtends 4*pi steradians)
            zobov.maskHDU = maskHDU
        else:
            delta_x = self.cmax[0]-self.cmin[0]
            delta_y = self.cmax[1]-self.cmin[1]
            delta_z = self.cmax[2]-self.cmin[2]
            vol = delta_x*delta_y*delta_z

        zobov.hdu.header['VOLUME'] = (mknumV2(vol), 'Survey Volume (Mpc/h)^3')
        masked_gal_count = np.sum(nnls==np.arange(len(nnls))) #this selects every galaxy in zcut if no magcut is used and selects galaxies in mcut if magcut is used
        zobov.hdu.header['MSKGAL'] = (masked_gal_count, 'Number of Galaxies in Tesselation')
        zobov.hdu.header['MSKDEN'] = (mknumV2(masked_gal_count/vol), 'Galaxy Count Density (Mpc/h)^-3') 
        zobov.hdu.header['MSKSEP'] = (mknumV2(np.power(vol/masked_gal_count, 1/3)), 'Average Galaxy Separation (Mpc/h)')
        
        # create galaxy IDs and optionally get catalog target IDs
        self.galids = np.arange(len(galaxy_table))
        galaxy_ID_name = column_names['ID']
        if galaxy_ID_name != 'None':
            self.tarids = galaxy_table[galaxy_ID_name]
        

class Tesselation:
    """Implementation of Voronoi tesselation of the catalog.
    """

    def __init__(self,cat,viz=False,periodic=False,xyz=False,buff=5.):
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
        coords = cat.coord[cat.nnls==np.arange(len(cat.nnls))] #this selects every galaxy in zcut if no magcut is used and selects galaxies in mcut if magcut is used
        if periodic:
            print("Triangulating...")
            # create delaunay triangulation
            Del = Delaunay(coords,incremental=True,qhull_options='QJ')
            sim = Del.simplices
            simlen = len(sim)
            cids = np.arange(len(coords))
            print("Finding periodic neighbors...")
            n = 0
            # add a buffer of galaxies around the simulation edges
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
            ver = Vor.vertices #array of vertices in the tesselelation
            
            reg = np.array(Vor.regions, dtype=object)[Vor.point_region] #for each galaxy, indexes of vertices forming the voronoi cell
            
            
            
            
            
            del Vor
            ve2 = ver.T
            if not xyz:
                vth = np.arctan2(np.sqrt(ve2[0]**2.+ve2[1]**2.),ve2[2]) #spherical coordinates for voronoi vertices
                vph = np.arctan2(ve2[1],ve2[0])
                vrh = np.array([np.sqrt((v**2.).sum()) for v in ver])
                crh = np.array([np.sqrt((c**2.).sum()) for c in coords]) #radial distance to galaxies
                rmx = np.amax(crh)
                rmn = np.amin(crh)
            print("Computing volumes...")
            vol = np.zeros(len(reg))
            cut = np.arange(len(vol))
            if xyz:
                cu1 = np.array([-1 not in r for r in reg]) #cut selecting galaxies with finite voronoi cells
                cu2 = np.array([np.product(np.logical_and(ve2[0][r]>cat.cmin[0],ve2[0][r]<cat.cmax[0]),dtype=bool) for r in reg[cu1]]).astype(bool)
                cu3 = np.array([np.product(np.logical_and(ve2[1][r]>cat.cmin[1],ve2[1][r]<cat.cmax[1]),dtype=bool) for r in reg[cu1][cu2]]).astype(bool)
                cu4 = np.array([np.product(np.logical_and(ve2[2][r]>cat.cmin[2],ve2[2][r]<cat.cmax[2]),dtype=bool) for r in reg[cu1][cu2][cu3]]).astype(bool)
                cut = cut[cu1][cu2][cu3][cu4]
            else:
                cu1 = np.array([-1 not in r for r in reg]) #cut selecting galaxies with finite voronoi cells
                cu2 = np.array([np.product(np.logical_and(vrh[r]>rmn,vrh[r]<rmx),dtype=bool) for r in reg[cu1]]).astype(bool) #cut selecting galaxes whose voronoi coordinates are finite and are within the survey distance limits
                msk = cat.mask
                nsd = hp.npix2nside(len(msk))
                pid = hp.ang2pix(nsd,vth,vph)
                imk = msk[pid] #cut selecting voronoi vertexes inside survey mask
                cu3 = np.array([np.product(imk[r],dtype=bool) for r in reg[cu1][cu2]]).astype(bool) #cut selecting galaxies with finite voronoi coords that are within the total survvey bounds
                cut = cut[cu1][cu2][cu3] #shortcut for cu3 but w/o having to stack [cu1][cu2][cu3] each time
            
            hul = [] #list of convex hull objects 
            for r in reg[cut]:
                try:
                    ch = ConvexHull(ver[r])
                except:
                    ch = ConvexHull(ver[r],qhull_options='QJ')
                hul.append(ch)
            #hul = [ConvexHull(ver[r]) for r in reg[cut]]
            vol[cut] = np.array([h.volume for h in hul]) #write the volumes from the convex hulls to vol
            self.volumes = vol
            if viz: #save vertices, regions,and the cut to select regions contained by the surey bounds
                self.vertIDs = reg
                vecut = np.zeros(len(vol),dtype=bool)
                vecut[cut] = True
                self.vecut = vecut
                self.verts = ver
            print("Triangulating...")
            Del = Delaunay(coords,qhull_options='QJ')
            sim = Del.simplices #tetrahedra cells between galaxies (whereas voronoi cells were arbitrary shapes around galaxies)

        nei = [] #for each galaxy, list of indexes of neighbor galaxies in the same tetrahedra objects that it's part of
        lut = [[] for _ in range(len(vol))] #same as nei but duplicate galaxy entries may appear
        hzn = np.zeros(len(vol),dtype=bool) # for each galaxy, flags if teh galaxy is along edge of survey
        print("Consolidating neighbors...")
        for i in range(len(sim)):
            for j in sim[i]:
                lut[j].append(i)
        for i in range(len(vol)):
            cut = np.array(lut[i])
            nei.append(np.unique(sim[cut]))
            if 0. in vol[nei[i]]: # if any of cell's neighbors have a volume of 0 (meaning cell extends outside survey bounds)
                hzn[i] = True # denote edge cell
        self.neighbors = np.array(nei, dtype=object) #for each galaxy, list of neighbor galaxy coordinates (see above explanation)
        self.hzn       = hzn # selects edge cells (see above explanation)

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
        hzn   = tess.hzn

        # Sort the Voronoi cells by their volume
        print("Sorting cells...")

        srt   = np.argsort(-1.*vol) # srt[5] gives the index in vol of the 5th largest volume (lowest density)

        vol2  = vol[srt] # cell volumes sorted from largest to smallest (aka least dense to most dense region)
        nei2  = nei[srt] # coordinates of tetrahedra that include each galaxy, sorted from least dense to most dense region

        # Build zones from the cells
        lut   = np.zeros(len(vol), dtype=int) #for each galaxy, the ID if the zone it belongs to
        depth = np.zeros(len(vol), dtype=int) #for each galaxy, the number of adjacent cells between it and the largest cell in its zone

        zvols = [0.] # the volume of the largest cell in each zone?
        zcell = [[]] #list of zones, where each zone is a list of cells
        zhzn  = [1]

        print("Building zones...")

        for i in range(len(vol)):

            if vol2[i] == 0.:
                lut[srt[i]] = -1
                zcell[-1].append(srt[i])
                continue

            ns = nei2[i] # indexes of galaxies neigboring curent galaxy
            vs = vol[ns] # volumes of cells neighboring current cell
            n  = ns[np.argmax(vs)] #index of neigboring galaxy with largest volume

            if n == srt[i]: # if current cell is larger than all it's neighbors (aka the center of a zone)
                # This cell has the largest volume of its neighbors
                lut[n] = len(zvols) - 1 # the galaxy in this cell is given a new zone ID 
                zcell.insert(-1,[n]) # create a new zone
                zvols.insert(-1,vol[n]) # note the volume of the largest cell in the zone
                zhzn.insert(-1,int(hzn[n])) #note whether the largest cell in teh zone is an edge cell

            else:
                # This cell is put into its least-dense neighbor's zone
                lut[srt[i]]   = lut[n] #the galaxy in this cell is given the zone ID of it's least dense neighbor
                depth[srt[i]] = depth[n]+1 #the galaxy's depth = its least dense neighbor's depth + 1
                zcell[lut[n]].append(srt[i]) #the galaxy is added to it's least dense neighbor's zone
                zhzn[lut[n]] += int(hzn[srt[i]]) #increment the zone's edge flag if an edge cell is found (0 = no edge cells)


        self.zcell = np.array(zcell, dtype=object)
        self.zvols = np.array(zvols)
        self.zhzn  = np.array(zhzn)
        self.depth = depth
        
        # For each zone i and its neighbors j
        # Identify neighboring zones (zlinks[0][i] has j in once it for every cell on their border?)
        # and the least-dense cells linking them (zlinks[1][i] has j copies of the maximum link volume between the zones?)
        zlinks = [[[] for _ in range(len(zvols))] for _ in range(2)] 

        if viz:
            zverts = [[] for _ in range(len(zvols))]
            znorms = [[] for _ in range(len(zvols))]
            zarea_0 = np.zeros(len(zvols))
            zarea_t = np.zeros(len(zvols))
            zarea_s = [[] for _ in range(len(zvols))]

        print("Linking zones...")

        for i in range(len(vol)): #iterate through every cell
            ns = nei[i] #get cells neighboring current cell
            z1 = lut[i] #get the zone ID for current cell
            if z1 == -1: #if cell isn't part of a void, then nothing further needs to be done
                continue
            for n in ns: #iterate though neighboring cells
                z2 = lut[n] #get zone ID of current neighbor
                if z2 == -1: #if current neighbor isn't in a void
                    if viz:
                        vts = tess.vertIDs[i].copy() #get indexes of vertices forming original neighbor
                        vts.extend(tess.vertIDs[n]) #add indexes of vertices forming current neighbor
                        vts = np.array(vts)
                        vts = vts[[len(vts[vts==v])==2 for v in vts]] #select only vertices that are shared by the two cells?
                        if len(vts)>2: #If there are 3 vertices shared between teh cells (a triangle)
                            vcs = rotate(tess.verts[vts]) #rotate the triangle to be flat on the "ground"
                            chv = ConvexHull(vcs).volume #get the triangle's area: TODO: this can surely be replaced with a simple area formula
                            zarea_0[z1] += chv
                            zarea_t[z1] += chv
                    else:
                        continue
                if z1 != z2:
                    # This neighboring cell is in a different zone
                    if z2 not in zlinks[0][z1]:
                        zlinks[0][z1].append(z2)
                        zlinks[0][z2].append(z1)
                        zlinks[1][z1].append(0.)
                        zlinks[1][z2].append(0.)
                        if viz:
                            zarea_s[z1].append(0.)
                            zarea_s[z2].append(0.)
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
                            vcs = rotate(tess.verts[vts])
                            chv = ConvexHull(vcs).volume
                            zarea_t[z1] += chv
                            zarea_s[z1][j] += chv
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
            self.zverts  = zverts
            self.znorms  = znorms
            self.zarea_0 = zarea_0
            self.zarea_t = zarea_t
            self.zarea_s = zarea_s


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
        
        # For each zone i and its neighbors j
        # Identify neighboring zones (zlinks[0][i] has j in once it for every cell on their border?)
        # and the least-dense cells linking them (zlinks[1][i] has j copies of the maximum link volume between the zones?)
        zl0   = np.array(list(flatten(zlinks[0])))
        zl1   = np.array(list(flatten(zlinks[1])))

        zlu   = -1.*np.sort(-1.*np.unique(zl1)) # the maximum cell volumes linking all adjacent zones, in descending order
        zlut  = [np.unique(zl0[np.where(zl1==zl)[0]]).tolist() for zl in zlu] #the zone ids for each linking volume?

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
        
        #for every zone link
        for i in range(len(zlu)):
            lvol  = zlu[i] #get the maximum volume along the link surface
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
