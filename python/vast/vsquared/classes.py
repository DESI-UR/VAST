"""Utility classes for the ZOBOV algorithm using a voronoi tesselation of an
input catalog.
"""

import numpy as np
import healpy as hp
import time
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import ConvexHull, Voronoi, Delaunay, KDTree

from vast.vsquared.util import toCoord, getBuff, flatten, mknumV2
from vast.voidfinder.preprocessing import load_data_to_Table

from vast.vsquared.class_utils import calculate_region_volume

import os
import mmap
import tempfile
import multiprocessing
from multiprocessing import Process, Value

from ctypes import c_int64

class Catalog:
    """Catalog data for void calculation.
    """

    def __init__(self,
                 catfile,
                 nside,
                 zmin,
                 zmax,
                 column_names,
                 maglim=None,
                 H0=100,
                 Om_m=0.3,
                 periodic=False,
                 cmin=None,
                 cmax=None,
                 maskfile=None,
                 zobov=None,
                 verbose=0):
        """
        Description
        ===========
        Given a number of input physical parameters and data files, repackage
        important bits of information for later use as attributes of this class.
        
        Also:
            converts ra-dec-redshift into xyz coords
            creates a mask for the survey using HEALPix
          
        Parameters
        ==========
        
        catfile: str
            Object catalog file (FITS format).
            
        nside : int
            HEALPix map `nside` parameter (2,4,8,16,...,2^k).  This value represents
            how many subdivisions of the Hierarchical Equal Area isoLatitude PIXelization 
            of the sphere there are, where 
            nside==1 -> 12 sphere regions
            nside==2 -> 48 sphere regions
            nside==4 -> 192 sphere regions, etc
            see: https://healpix.jpl.nasa.gov/index.shtml
            
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
            
        cmin : ndarray or None
            Array of coordinate minima. (Only necessary in Periodic mode?)
            
        cmax : ndarray or None
            Array of coordinate maxima. (Only necessary in Periodic mode?)
            
        verbose : int
            used to enable (>=1) or disable (0) print messages
            
        
        Outputs
        =======
        
        self.coord : ndarray shape (N,3)
            xyz coordinates of the galaxies
            
        self.nnls : ndarray shape (N,)
            integer representing validity of the given galaxy in
            self.coord, -1 means invalid
            
        """
        
        
        
        
        ################################################################################
        # read in galaxy file to an Astropy Table
        ################################################################################
        if verbose > 0:
            print("Extracting data...")
        
        galaxy_table = load_data_to_Table(catfile)
        
        if verbose > 0:
            print("Read in galaxy data (rows, cols): ", len(galaxy_table), len(galaxy_table.columns))
            print(galaxy_table.columns)
        
        
        
        ################################################################################
        # This section is actually doing 2 things:
        #   1. Getting the coordinates of the galaxies in xyz format
        #      a. in periodic mode, they need to already by in xyz
        #      b. in non-periodic mode, they need to be in ra-dec-redshift and then
        #         we convert them to xyz
        #   2. Checking redshift (z) validity
        #
        # Right now, these two things are mangled together and mangled with
        # the periodic mode parameter - I'm (QuiteAFoxtrot) leaving them mangled 
        # for the moment even though they could be independent from the periodic 
        # mode parameter to avoid huge changes to the API/behavior without further 
        # discussion and since it is likely that people will provide XYZ coords for 
        # periodic/synthetic surveys and ra-dec-redshift for real (aka non-periodic
        # surveys
        # 
        # self.coord is a shape (N,3) array of the xyz coordinates
        ################################################################################
        if periodic:
            
            self.coord = np.array([galaxy_table[column_names['x']],
                                   galaxy_table[column_names['y']],
                                   galaxy_table[column_names['z']]]).T
                                   
            self.cmin = cmin
            
            self.cmax = cmax
            
        else:
            
            z    = galaxy_table[column_names['redshift']]
            
            ra   = galaxy_table[column_names['ra']]
            
            dec  = galaxy_table[column_names['dec']]
            
            zcut = np.logical_and(z > zmin, z < zmax) # 1 if gal is in the zlims 0 if not
            
            if not zcut.any():
                print("Choose valid redshift limits", z.min(), z.max())
                return
            
            #alias for zcut, unless magnitude limit is used, in which case scut will be later set to mcut
            # `scut` a boolean array to identify desired galaxies
            scut = zcut 
            
            c1, c2, c3 = toCoord(z, ra, dec, H0, Om_m)
            
            self.coord = np.array([c1, c2, c3]).T
            
            
        
        ################################################################################
        # This array will be used for selecting the subset of galaxies which are
        # valid, by combining information from redshift cuts and magnitude limit
        # cuts.  It is initialized to be the index of the galaxy within the coords
        # array, and then that index is modified 
        # Array that will hold -1 for galaxies outside z limits, nearest neighbor 
        # galaxy in magcut for remaining galaxies not in magcut, and self identifier 
        # for further remaining galaxies
        # NNLS = "Nearest Neighbor Lookup Something"?
        ################################################################################
        num_gals = len(self.coord)
        
        nnls = np.arange(num_gals) 
        
        # Galaxies outside z/redshift limit are marked with -1
        # non-periodic mode only since periodic assume rectangular region
        # and infinite universe
        if not periodic:
            nnls[zcut < 1] = -1

        ################################################################################
        # Apply magnitude limit (aka cut dim galaxies) by updating the nnls index
        # This section can also only be applied in non-periodic mode
        ################################################################################
        if maglim is not None:
            
            if verbose > 0:
                print("Applying magnitude cut...")
                
            mag = galaxy_table[column_names['rabsmag']]
            
            mcut = np.logical_and(mag < maglim, zcut) # mcut is a subsample of zcut that removes galaxies outside the magnitude limit
            
            if not mcut.any():
                print("Choose valid magnitude limit")
                return
            
            # scut is made into an alias for mcut, unless no magnitude limit 
            # is used, in which case it remains an alias for zcut
            # `scut` a boolean array to identify desired galaxies
            scut = mcut 
            
            ncut = np.arange(num_gals, dtype=int)[zcut][mcut[zcut]<1]  # indexes of galaxies in zcut but not in mcut
            
            # These neighbor indices do not appear to be used anywhere so
            # for now, offsetting this code block to not run by default since
            # a KDTree is computationally expensive
            calc_neighbors = False 
            if calc_neighbors:
                
                tree = KDTree(self.coord[mcut]) #kdtree of galaxies in mcut
                
                lut  = np.arange(num_gals, dtype=int)[mcut] #indexes of galaxies in mcut
                
                # the nearest neighbor index for each galaxy in zcut but not in mcut, 
                # and where the neighbors are in mcut
                #
                # For galaxies where it is within the redshift limits (zcut) but not within
                # the magnitude limits (mcut), set the value in nnls to the index of its
                # nearest neighbor galaxy who is in the magnitude limits (since the
                # tree is only built on magnitude cut galaxies)
                neigh_idxs_from_mcut = tree.query(self.coord[ncut])[1]
                nnls[ncut] = lut[neigh_idxs_from_mcut] 
            else:
                nnls[ncut] = -1
            
            self.mcut = mcut
        
        self.nnls = nnls
        
        #print("SCUT==nnls?: ", np.all((self.nnls > -1) == scut)) #True lol...

        ################################################################################
        # Apply survey mask
        ################################################################################
        if not periodic:
            
            if maskfile is None:
                
                if verbose > 0:
                    print("Generating mask...")
                    
                #create a healpix mask with specified nside - the healpix
                #mask is just an array of a specific length corresponding to the
                #`nside` parameter (1->len 12, 2-> len 48, 3-> len 192, etc)
                #which healpy can convert to and from angular space into 
                #indices into that pixel space
                # We initialize our mask with 0's
                mask = np.zeros(hp.nside2npix(nside), dtype=bool) 
                
                #healpy can now give us the integer indices of each of the
                #galaxies we have identified with `scut`
                pix_idxs = hp.ang2pix(nside, ra[scut], dec[scut], lonlat=True) #convert scut galaxy coordinates to mask coordinates
                
                #mark where galaxies fall in mask
                mask[pix_idxs] = True 
                
            else:
                mask = (hp.read_map(maskfile)).astype(bool)
                
            self.mask = mask #mask of all galaxies in scut, where scut might be zcut or mcut depending on if magnitude cut is used
            
            pix_idxs = hp.ang2pix(nside, ra, dec, lonlat=True) #convert all galaxy coordinates to mask coordinates
            
            # mask pixel bool value for every galaxy (aka is galaxy in same mask bin as a galaxy in scut), multiplied by zcut
            # this is used to select galaxies located outside the survey mask in the 'out' column of the galzones HDU
            # `imsk` "in mask" -> bool array of length num_gals where 1 means in the mask and 0 means not in the mask
            self.imsk = mask[pix_idxs]*zcut 
            
            #print("Mask shape: ", mask.shape, "Zcut shape: ", zcut.shape, "IMSK shape: ", self.imsk.shape)

            

            coverage = np.sum(mask) * hp.nside2pixarea(nside) #solid angle coverage in steradians
            
            coverage_deg = coverage*(180/np.pi)**2 #solid angle coverage in deg^2
            
            d_max = zobov.hdu.header['DLIMU']
            
            d_min = zobov.hdu.header['DLIML']
            
            #Use the sky angle coverage with the inner and outer radii of the survey
            #to calculate the approx volume of the survey
            vol = coverage / 3 * (d_max ** 3 - d_min ** 3) # volume calculation (A sphere subtends 4*pi steradians)
            
            
            #record mask information
            maskHDU = fits.ImageHDU(mask.astype(int))
            maskHDU.name = 'MASK'
            maskHDU.header['COVSTR'] = (mknumV2(coverage), 'Sky Coverage (Steradians)')
            zobov.hdu.header['COVSTR'] = (mknumV2(coverage), 'Sky Coverage (Steradians)')
            zobov.hdu.header['COVDEG'] = (mknumV2(coverage_deg), 'Sky Coverage (Degrees^2)')
            zobov.maskHDU = maskHDU
        
        ################################################################################
        # In Periodic mode,we have rectangular geometry so
        # can just do x*y*z for volume
        ################################################################################
        else:
            
            delta_x = self.cmax[0] - self.cmin[0]
            
            delta_y = self.cmax[1] - self.cmin[1]
            
            delta_z = self.cmax[2] - self.cmin[2]
            
            vol = delta_x*delta_y*delta_z



        #this selects every galaxy in zcut if no magcut is used and selects galaxies
        #in mcut if magcut is used
        masked_gal_count = np.sum(nnls==np.arange(len(nnls))) 

        zobov.hdu.header['VOLUME'] = (mknumV2(vol), 'Survey Volume (Mpc/h)^3')
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

    def __init__(self,
                 cat,
                 nside,
                 viz=False,
                 periodic=False,
                 num_cpus=1,
                 buff=5.0,
                 verbose=0):
        """Initialize tesselation.

        Parameters
        ==========
        
        cat : Catalog
            Catalog of objects used to compute the Voronoi tesselation.
            
        viz : bool
            Compute visualization.
            
        periodic : bool
            Use periodic boundary conditions.
            
        buff : float
            Width of incremental buffer shells for periodic computation.
            
        verbose : int
            used to enable (>=1) or disable (0) print messages
            
            
        Outputs
        =======
        
        self.volumes : ndarray of shape (num_galaxies,)
            volume of the voronoi cell for each input galaxy
        
        self.neighbors : list of lists
            for each galaxy, a list of the indices of the neighbor galaxies
            which belong to the same
        
        
        self.vecut
        self.verts
        self.vertIDs
        """
        
        self.num_cpus = num_cpus
        
        #the catalog.nnls index has been computed in the Catalog class such that
        #this selects the subset of galaxies in `zcut` if no magcut is used and 
        #selects galaxies in `mcut` if magcut is used
        coords = cat.coord[cat.nnls==np.arange(len(cat.nnls))] 
        
        self.num_gals = coords.shape[0]
        
        if periodic:
            
            if verbose > 0:
                print("Triangulating...")
                
            # create an initial delaunay triangulation
            Del = Delaunay(coords, incremental=True, qhull_options='QJ')
            sim = Del.simplices
            simlen = len(sim)
            cids = np.arange(len(coords))
            
            
            if verbose > 0:
                print("Finding periodic neighbors...")
            n = 0
            
            # add a buffer of galaxies around the simulation edges
            coords2, cids = getBuff(coords,cids,cat.cmin,cat.cmax,buff,n)
            
            coords3 = coords.tolist()
            
            coords3.extend(coords2)
            
            #Update the delaunay triangulation with these new points
            Del.add_points(coords2)
            
            sim = Del.simplices
            
            #Keep adding points to the delaunay triangulation until?
            while np.amin(sim[simlen:]) < len(coords):
                n = n + 1
                simlen = len(sim)
                coords2, cids = getBuff(coords, cids, cat.cmin, cat.cmax, buff, n)
                coords3.extend(coords2)
                Del.add_points(coords2)
                sim = Del.simplices
                
            for i in range(len(sim)):
                sim[i] = cids[sim[i]]
                
            if verbose > 0:
                print("Tesselating...")
            Vor = Voronoi(coords3)
            ver = Vor.vertices
            reg = np.array(Vor.regions)[Vor.point_region]
            del Vor
            
            
            if verbose > 0:
                print("Computing volumes...")
            '''
            #See comments below - replacing this section with parallelized stuff
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
            '''
                
            ################################################################################
            # Tweaking this section to use the new parallelized infrastructure for
            # volume calculation 
            # use dummy radii of 1.0 and max/min of 2.0/0.0 so all
            # radii fall within r_max and r_min
            ################################################################################
            vrh = np.ones(len(Vor.point_region), dtype=np.float64)
            r_max = 2.0
            r_min = 0.0
            verticies_in_mask_uint8 = np.ones(len(Vor.point_region), dtype=np.uint8)
                
            output_volumes = self.calculate_region_volumes(Vor,
                                                           ver.astype(np.float64),
                                                           r_max,
                                                           r_min,
                                                           vrh,
                                                           verticies_in_mask_uint8,
                                                           )
            
            self.volumes = output_volumes
            
        else:
            
            if verbose > 0:
                print("Tesselating...")
                
            #Just for debugging
            #derp_time = time.time()
            #derp1 = Delaunay(coords, qhull_options='QJ')
            #print("Derp1 time: ", time.time() - derp_time)
                
            
                
            voronoi_time = time.time()
            
            voronoi_graph = Voronoi(coords)
            
            print("Voronoi time: ", time.time() - voronoi_time)
            
            
            other_time = time.time()
            
            ################################################################################
            # We will need to know whether the verticies are within the mask to include 
            # those volumes or not, so calculate the sky angles of the vertex locations
            # and throw them into the healpix utility function to get the mask values
            # corresponding to those locations
            ################################################################################
            mask = cat.mask
            
            vertices = voronoi_graph.vertices
            
            vertices_theta = np.arctan2(np.sqrt(vertices[:,0]**2. + vertices[:,1]**2.), vertices[:,2]) 
            
            verticies_phi = np.arctan2(vertices[:,1], vertices[:,0])
            
            pix_ids = hp.ang2pix(nside, vertices_theta, verticies_phi) 
            
            verticies_in_mask = mask[pix_ids]
            
            verticies_in_mask_uint8 = verticies_in_mask.astype(np.uint8)
            
            ################################################################################
            # We will also need some radial information about the verticies and galaxies
            ################################################################################
            vrh = np.linalg.norm(vertices, axis=1).astype(np.float64)
            
            crh = np.linalg.norm(coords, axis=1).astype(np.float64)
            
            r_max = np.max(crh) 
            
            r_min = np.min(crh) 
            
            verticies64 = voronoi_graph.vertices.astype(np.float64)
            
            output_volumes = self.calculate_region_volumes(voronoi_graph,
                                                           #output_volume,
                                                           verticies64,
                                                           r_max,
                                                           r_min,
                                                           vrh,
                                                           verticies_in_mask_uint8,
                                                           )
            
            self.volumes = output_volumes
            
            print("Cut+Convex Hull time: ", time.time() - other_time)
            
            
            if verbose > 0:
                print("Triangulating...")
            delaunay_time = time.time()
            Del = Delaunay(coords, qhull_options='QJ')
            print("Delaunay time: ", time.time() - delaunay_time)
            sim = Del.simplices #tetrahedra cells between galaxies (whereas voronoi cells were arbitrary shapes around galaxies)
        
        
        ################################################################################
        #
        ################################################################################
        
        neigh_time = time.time()
        
        nei = [] #for each galaxy, list of neighbor galaxy coordinates in the same tetrahedra objects that it's part of
        lut = [[] for _ in range(len(self.volumes))] #for each galaxy, indexes (in tetreheda list sim) of tetrahedra that it's part of
        if verbose > 0:
            print("Consolidating neighbors...")
            
        for i in range(len(sim)):
            for j in sim[i]:
                lut[j].append(i)
                
        for i in range(len(self.volumes)):
            cut = np.array(lut[i])
            nei.append(np.unique(sim[cut]))
            
            
        #for each galaxy, list of neighbor galaxy coordinates (see above explanation)
        self.neighbors = np.array(nei, dtype=object) 
        
        print("Neighbor time: ", time.time() - neigh_time)

        #save vertices, regions,and the cut to select regions contained by the survey bounds
        if viz: 
            self.vertIDs = reg
            vecut = np.zeros(len(vol), dtype=bool)
            vecut[cut] = True
            self.vecut = vecut
            self.verts = ver

    def calculate_region_volumes(self, 
                                 voronoi_graph,
                                 #output_volume,
                                 verticies64,
                                 r_max,
                                 r_min,
                                 vrh,
                                 in_mask_uint8,
                                 ):
        """
        This function essentially serves as a switch between single process
        and multiprocess calculation for calculating the region volume
        """
        
        
        
        if self.num_cpus == 1:
            
            output_volumes = np.zeros(self.num_gals, dtype=np.float64)
            
            for idx, region_idx in enumerate(voronoi_graph.point_region):
                
                region = voronoi_graph.regions[region_idx]
                
                calculate_region_volume(idx,
                                        region,
                                        output_volumes,
                                        verticies64,
                                        r_max,
                                        r_min,
                                        vrh,
                                        in_mask_uint8)
                
            
        elif self.num_cpus > 1:
            
            # We're going to use a very simply multiprocessing scheme here
            # since the voronoi graph has already been calculated and we can
            # essentially treat it as read-only
            num_indices = len(voronoi_graph.point_region)
            
            index_coordinator = Value(c_int64, 0, lock=True)
            
            
            volumes_fd, VOLUMES_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared", 
                                                               dir="/dev/shm", 
                                                               text=False)
            
            volumes_buffer_length = self.num_gals*8
            
            os.ftruncate(volumes_fd, volumes_buffer_length)
            
            volumes_buffer = mmap.mmap(volumes_fd, 0)
            
            os.unlink(VOLUMES_BUFFER_PATH)
            
            output_volumes = np.frombuffer(volumes_buffer, dtype=np.float64)
    
            output_volumes.shape = (self.num_gals,)
            
            
            
            
            
            
            startup_context = multiprocessing.get_context("fork")
                
            processes = []
            
            for proc_idx in range(self.num_cpus):
            #for proc_idx in range(1):
                
                #p = startup_context.Process(target=_hole_finder_worker_profile, 
                p = startup_context.Process(target=self.volume_calculation_worker, 
                                            args=(num_indices, 
                                                  index_coordinator, 
                                                  voronoi_graph,
                                                  volumes_fd,
                                                  verticies64,
                                                  r_max,
                                                  r_min,
                                                  vrh,
                                                  in_mask_uint8
                                                  ))
                
                p.start()
                
                processes.append(p)
                
            
            for p in processes:
            
                p.join(None) #block till join
        
        return output_volumes
        
        
    def volume_calculation_worker(self, 
                                  max_indicies,
                                  index_coordinator,
                                  voronoi_graph,
                                  volumes_fd,
                                  verticies64,
                                  r_max,
                                  r_min,
                                  vrh,
                                  in_mask_uint8
                                  ):
        
        #max_indices and num_gals are the same thing
        volumes_buffer_length = max_indicies*8 #float64 so 8 bytes per element
    
        volumes_buffer = mmap.mmap(volumes_fd, volumes_buffer_length)
        
        output_volumes = np.frombuffer(volumes_buffer, dtype=np.float64)
    
        output_volumes.shape = (self.num_gals,)
        
        
        curr_index = 0
        
        while True:
            
            index_coordinator.acquire()
            
            curr_index = index_coordinator.value
            
            index_coordinator.value += 1
            
            index_coordinator.release()
        
            if curr_index >= max_indicies:
                break
        
            #print("Working index: ", curr_index, " of: ", max_indicies)
        
            region_idx = voronoi_graph.point_region[curr_index]
                
            region = voronoi_graph.regions[region_idx]
            
            calculate_region_volume(curr_index,
                                    region,
                                    output_volumes,
                                    verticies64,
                                    r_max,
                                    r_min,
                                    vrh,
                                    in_mask_uint8)
        
        return None
        
        


class Zones:
    """Partitioning of particles into zones around density minima.
    """

    def __init__(self,
                 tess,
                 viz=False,
                 verbose=0):
        """Implementation of zones: see arXiv:0712.3049 for details.

        Parameters
        ----------
        tess : Tesselation
            Voronoid tesselation of an object catalog.
        viz : bool
            Compute visualization.
        verbose : int
            used to enable (>=1) or disable (0) print messages
        """
        vol   = tess.volumes
        nei   = tess.neighbors

        # Sort the Voronoi cells by their volume
        if verbose > 0:
            print("Sorting cells...")

        #srt   = np.argsort(-1.*vol) # srt[5] gives the index in vol of the 5th largest volume (lowest density)
        srt = np.argsort(vol)[::-1] #largest to smallest

        vol2  = vol[srt] # cell volumes sorted from largest to smallest (aka least dense to most dense region)
        nei2  = nei[srt] # coordinates of tetrahedra that include each galaxy, sorted from least dense to most dense region

        # Build zones from the cells
        lut   = np.zeros(len(vol), dtype=int) #for each galaxy, the ID if the zone it belongs to
        depth = np.zeros(len(vol), dtype=int) #for each galaxy, the number of adjacent cells between it and the largest cell in its zone

        zvols = [0.] # the volume of the largest cell in each zone?
        zcell = [[]] #list of zones, where each zone is a list of cells

        if verbose > 0:
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
            else:
                # This cell is put into its least-dense neighbor's zone
                lut[srt[i]]   = lut[n] #the galaxy in this cell is given the zone ID of it's least dense neighbor
                depth[srt[i]] = depth[n]+1 #the galaxy's depth = its least dense neighbor's depth + 1
                zcell[lut[n]].append(srt[i]) #the galaxy is added to it's least dense neighbor's zone

        self.zcell = np.array(zcell, dtype=object)
        self.zvols = np.array(zvols)
        self.depth = depth

        # Identify neighboring zones and the least-dense cells linking them
        zlinks = [[[] for _ in range(len(zvols))] for _ in range(2)] 

        if viz:
            zverts = [[] for _ in range(len(zvols))]
            znorms = [[] for _ in range(len(zvols))]

        if verbose > 0:
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

    def __init__(self, zones, verbose=0):
        """Implementation of void calculation: see arXiv:0712.3049.

        Parameters
        ----------
        zones: Zones
            A group of zones around density minima in an input catalog.
        verbose : int
            used to enable (>=1) or disable (0) print messages
        """
        zvols  = np.array(zones.zvols)
        zlinks = zones.zlinks

        # Sort zone links by volume, identify zones linked at each volume
        if verbose > 0:
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
        if verbose > 0:
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
