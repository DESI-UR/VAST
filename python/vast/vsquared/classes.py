"""Utility classes for the ZOBOV algorithm using a voronoi tesselation of an
input catalog.
"""

import numpy as np
import healpy as hp
import time
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import ConvexHull, Voronoi, Delaunay, KDTree

from vast.vsquared.util import toCoord, getBuff, flatten, mknumV2, rotate, partition_face_vertices
from vast.voidfinder.preprocessing import load_data_to_Table

from vast.vsquared.class_utils import calculate_region_volume

from multivoro import compute_voronoi

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
                 custom_galaxy_table=None,
                 maglim=None,
                 H0=100,
                 Om_m=0.3,
                 periodic=False,
                 xyz=False,
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

        custom_galaxy_table : astropy table
            If not None, the provided galaxy table is used for the voidfinding rather than 
            the catfile path. This is a convenience feature meant for testing V2 in live 
            environments without first needing to save the galaxy input to a file. Final 
            runs should use the catfile path instead.
            
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
        
    
    
        self.cmin = None
        self.cmax = None
        
        ################################################################################
        # read in galaxy file to an Astropy Table
        ################################################################################
        if verbose > 0:
            print("Extracting data...")

        if custom_galaxy_table is None:
            galaxy_table = load_data_to_Table(catfile)
        else:
            galaxy_table = custom_galaxy_table
        
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
        if periodic or xyz:
            
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
        if not periodic and not xyz:
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
            
            #self.mcut = mcut #No need for this? It's never used again...

        
        self.nnls = nnls
        
        #print("SCUT==nnls?: ", np.all((self.nnls > -1) == scut)) #True lol...

        ################################################################################
        # Apply survey mask
        ################################################################################
        if not periodic and not xyz:
            
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
                #read in exisitng mask
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
            
            #all galaxies are in mask for simulations
            self.imsk = np.ones(len(nnls),dtype=bool)
            
        # ------------------------------------------------------------------------------------------------------
        # Save metadata
        # ------------------------------------------------------------------------------------------------------



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
                 xyz=False,
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
            
        num_cpus : int
            number of CPUs to use for computation
            
        verbose : int
            used to enable (>=1) or disable (0) print messages
            
            
        Outputs
        =======
        
        self.volumes : ndarray of shape (num_galaxies,)
            volume of the voronoi cell for each input galaxy
        
        self.cells : list of multivoro cells
            The cells of the Voronoi tessellation 
        
        
        """
        
        self.num_cpus = num_cpus
        
        #the catalog.nnls index has been computed in the Catalog class such that
        #this selects the subset of galaxies in `zcut` if no magcut is used and 
        #selects galaxies in `mcut` if magcut is used
        coords = cat.coord[cat.nnls==np.arange(len(cat.nnls))] 
        
        self.num_gals = coords.shape[0]
        
        
        if verbose > 0:
            print("Tesselating...")
            
                    
        print("Starting multivoro")
                        
        
        multivoro_start = time.time()
        
        
        radii = 1*np.ones(coords.shape[0], dtype=np.float32)

        if periodic:

            periodic_boundaries = (True, True, True)
            limits = np.array([self.cmin, self.cmax])

        else:

            periodic_boundaries = (False, False, False)
        
            lower_min = coords.min(axis=0) - 100.0
            
            upper_max = coords.max(axis=0) + 100.0
            
            print("Lower min: ", lower_min)
            print("Upper max: ", upper_max)
            
            
            limits = np.empty((2,3), dtype=np.float32)
            limits[0,0] = lower_min[0]
            limits[0,1] = lower_min[1]
            limits[0,2] = lower_min[2]
            limits[1,0] = upper_max[0]
            limits[1,1] = upper_max[1]
            limits[1,2] = upper_max[2]
        
        
        print("Radii: ", radii)
        print("Limits: ", limits)

        
        cells = compute_voronoi(
                                points=coords,
                                radii=radii,
                                limits=limits,
                                n_threads=num_cpus,
                                periodic_boundaries=periodic_boundaries,
                                )
        self.cells = cells
        
        print("Multivoro time: ", time.time() - multivoro_start)
        
        print("Num cells: ", len(cells))

        
        volume_time = time.time()
        
        ################################################################################
        # Get information about survey mask
        ################################################################################
        
        crh = np.linalg.norm(coords, axis=1).astype(np.float64)
    
        r_max = np.max(crh) 
        
        r_min = np.min(crh) 

        if xyz or periodic:
            # mask is not used, so set it to a one-element array
            mask_uint8=np.ones((1,), dtype=np.uint8)
        else:
            mask = cat.mask
            mask_uint8 = mask.astype(np.uint8)

        ################################################################################
        # Calculate volumes of cells
        ################################################################################
        
        output_volumes = self.calculate_region_volumes(self.cells,
                                                       r_max,
                                                       r_min,
                                                       mask_uint8,
                                                       xyz,
                                                       cat.cmin,
                                                       cat.cmax,
                                                       nside
                                                       )
        
        self.volumes = output_volumes
        
        print("Cut+Convex Hull time: ", time.time() - volume_time)
        

        ################################################################################
        # Flag galaxies along the survey edges
        ################################################################################

        neigh_time = time.time()
        
        self.hzn = np.zeros(self.num_gals, dtype=bool) # for each galaxy, flags if the galaxy is along edge of survey
        
        for idx in range(self.num_gals):
            
            neigh_indices = self.cells[idx].get_neighbors()
            
            # if any of cell's neighbors have a volume of 0 (meaning cell extends outside survey bounds)
            if np.any(self.volumes[neigh_indices] == 0.0):
                
                self.hzn[idx] = True
        
        print("Neighbor time: ", time.time() - neigh_time)

        

    def calculate_region_volumes(self, 
                                 cells,
                                 r_max,
                                 r_min,
                                 mask_uint8,
                                 xyz_mode,
                                 cmin,
                                 cmax,
                                 nside,
                                 ):
        """
        This function essentially serves as a switch between single process
        and multiprocess calculation for calculating the region volume
        """

        
        if self.num_cpus == 1:
            
            output_volumes = np.zeros(self.num_gals, dtype=np.float64)

            for idx, cell in enumerate(cells):

    
                vertices = cell.get_vertices()
    
                ################################################################################
                # We will need some radial information about the verticies to know whether to
                # include those galaxies or not
                ################################################################################
                
                vrh = np.linalg.norm(vertices, axis=1).astype(np.float64)
                
                #using <= and >= since original code inversely checked just > and <
                if np.any(vrh <= r_min) or np.any(vrh >= r_max):
                    continue
    
                ################################################################################
                # We will also need to know whether the verticies are within the mask to include 
                # those volumes or not, so calculate the sky angles of the vertex locations
                # and throw them into the healpix utility function to get the mask values
                # corresponding to those locations
                ################################################################################
        
                vertices_theta = np.arctan2(np.sqrt(vertices[:,0]**2. + vertices[:,1]**2.), vertices[:,2]) 
                        
                verticies_phi = np.arctan2(vertices[:,1], vertices[:,0])
                        
                pix_ids = hp.ang2pix(nside, vertices_theta, verticies_phi) 
                        
                verticies_in_mask = mask_uint8[pix_ids]
        
                if np.any(verticies_in_mask==0):
                    continue

                ################################################################################
                # Calculate the region volume
                ################################################################################
                                    
                calculate_region_volume(idx,
                                        vertices,
                                        output_volumes,
                                        r_max,
                                        r_min,
                                        xyz_mode,
                                        cmin,
                                        cmax)
                
            
        elif self.num_cpus > 1:
            
            # We're going to use a very simply multiprocessing scheme here
            # since the voronoi graph has already been calculated and we can
            # essentially treat it as read-only
            num_indices = len(cells)
            
            index_coordinator = Value(c_int64, 0, lock=True)
            
            
            volumes_fd, VOLUMES_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared", 
                                                               dir="/dev/shm", 
                                                               text=False)
            
            volumes_buffer_length = self.num_gals*8
            
            os.ftruncate(volumes_fd, volumes_buffer_length)
            
            volumes_buffer = mmap.mmap(volumes_fd, 0)
            
            os.unlink(VOLUMES_BUFFER_PATH)
            
            output_volumes = np.frombuffer(volumes_buffer, dtype=np.float64)
            
            output_volumes[:] = 0
    
            output_volumes.shape = (self.num_gals,)
            
            
            
            
            
            
            startup_context = multiprocessing.get_context("fork")
                
            processes = []
            
            for proc_idx in range(self.num_cpus):
            #for proc_idx in range(1):
                
                #p = startup_context.Process(target=_hole_finder_worker_profile, 
                p = startup_context.Process(target=self.volume_calculation_worker, 
                                            args=(num_indices, 
                                                  index_coordinator, 
                                                  cells,
                                                  volumes_fd,
                                                  r_max,
                                                  r_min,
                                                  mask_uint8,
                                                  xyz_mode,
                                                  cmin,
                                                  cmax,
                                                  nside
                                                  ))
                
                p.start()
                
                processes.append(p)
                
            
            for p in processes:
            
                p.join(None) #block till join
        
        return output_volumes
        
        
    def volume_calculation_worker(self, 
                                  max_indicies,
                                  index_coordinator,
                                  cells,
                                  volumes_fd,
                                  r_max,
                                  r_min,
                                  mask_uint8,
                                  xyz_mode,
                                  cmin,
                                  cmax,
                                  nside
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
        
            cell = cells[curr_index]

            vertices = cell.get_vertices()

            ################################################################################
            # We will need some radial information about the verticies to know whether to
            # include those galaxies or not
            ################################################################################
            
            vrh = np.linalg.norm(vertices, axis=1).astype(np.float64)
            
            #using <= and >= since original code inversely checked just > and <
            if np.any(vrh <= r_min) or np.any(vrh >= r_max):
                continue

            ################################################################################
            # We will also need to know whether the verticies are within the mask to include 
            # those volumes or not, so calculate the sky angles of the vertex locations
            # and throw them into the healpix utility function to get the mask values
            # corresponding to those locations
            ################################################################################
    
            vertices_theta = np.arctan2(np.sqrt(vertices[:,0]**2. + vertices[:,1]**2.), vertices[:,2]) 
                    
            verticies_phi = np.arctan2(vertices[:,1], vertices[:,0])
                    
            pix_ids = hp.ang2pix(nside, vertices_theta, verticies_phi) 
                    
            verticies_in_mask = mask_uint8[pix_ids]
    
            if np.any(verticies_in_mask==0):
                continue

            ################################################################################
            # Calculate the region volume
            ################################################################################
            
            calculate_region_volume(curr_index,
                                    vertices,
                                    output_volumes,
                                    r_max,
                                    r_min,
                                    xyz_mode,
                                    cmin,
                                    cmax)
        
        return None
        
        
class Zones:
    """Partitioning of particles into zones around density minima.
    """

    def __init__(self,
                 tess,
                 viz=False,
                 catalog=None,
                 verbose=0):
        """Implementation of zones: see arXiv:0712.3049 for details.

        Parameters
        ==========
        
        tess : Tesselation
            Voronoid tesselation of an object catalog.
            
        viz : bool
            Compute visualization.
            
        verbose : int
            used to enable (>=1) or disable (0) print messages
        """

        coords = catalog.coord[catalog.nnls==np.arange(len(catalog.nnls))] 
        
        vol = tess.volumes
        
        cells = tess.cells
        
        num_gals = tess.num_gals
        
        hzn = tess.hzn
        

        # Sort the Voronoi cells by their volume
        if verbose > 0:
            print("Sorting cells...")

        sort_order = np.argsort(vol)[::-1] #largest to smallest volume (aka lease dense to most dense)

        #vol2  = vol[srt] # cell volumes sorted from largest to smallest (aka least dense to most dense region)
        #nei2  = nei[srt] # coordinates of tetrahedra that include each galaxy, sorted from least dense to most dense region

        # Build zones from the cells
        
        #for each galaxy, the ID of the zone it belongs to
        gal_zone_IDs = np.zeros(len(vol), dtype=int) 
        
        #for each galaxy, the number of adjacent cells between it and the largest cell in its zone
        depth = np.zeros(len(vol), dtype=int) 

        
        
        #each element of zcell is a zone, and the zone is a 
        #list of the galaxy indices belonging to that zone
        zcell = [[]] 
        # the volume of the largest cell in each zone?
        # this is initialized with a 0 and zcell with an empty
        # list to capture any cells with 0 volume
        zvols = [0.] 
        zhzn = [1]
        

        if verbose > 0:
            print("Building zones...")

        build_time = time.time()
        
        for srt_i in sort_order:

            # Maybe keep some separate lists for the 0-vol 
            # cells since we have to check it explicitly anyway
            if vol[srt_i] == 0.:
                gal_zone_IDs[srt_i] = -1
                zcell[-1].append(srt_i)
                continue

            #ns = nei2[i] # indexes of galaxies neigboring curent galaxy
            curr_neigh_idxs = cells[srt_i].get_neighbors()
            #ns = np.append([srt_i], ns) #inefficient but just for testing
            
            neigh_vols = vol[curr_neigh_idxs] # volumes of cells neighboring current cell
            
            largest_neigh_vol_idx = curr_neigh_idxs[np.argmax(neigh_vols)] #index of neigboring galaxy with largest volume

            # if current cell is larger than all it's neighbors (aka the center of a zone)
            if vol[srt_i] > vol[largest_neigh_vol_idx]:
                # Current cell has the largest volume of its neighbors
                gal_zone_IDs[srt_i] = len(zvols) - 1 # the galaxy in this cell is given a new zone ID 
                
                # create a new zone
                # using insert(-1,...) instead of append to keep these lists
                # sorted from largest zone to smallest zone
                zcell.insert(-1, [srt_i]) 
                
                # note the volume of the largest cell in the zone
                zvols.insert(-1, vol[srt_i]) 
                
                zhzn.insert(-1, int(hzn[srt_i])) #note whether the largest cell in the zone is an edge cell
            
            else:
                # This cell is put into its least-dense neighbor's zone
                gal_zone_IDs[srt_i] = gal_zone_IDs[largest_neigh_vol_idx] #the galaxy in this cell is given the zone ID of it's least dense neighbor
                
                depth[srt_i] = depth[largest_neigh_vol_idx]+1 #the galaxy's depth = its least dense neighbor's depth + 1
                
                zcell[gal_zone_IDs[largest_neigh_vol_idx]].append(srt_i) #the galaxy is added to it's least dense neighbor's zone
                
                zhzn[gal_zone_IDs[largest_neigh_vol_idx]] += int(hzn[srt_i]) #increment the zone's edge flag if an edge cell is found (0 = no edge cells)

        
        print("Zone building time: ", time.time() - build_time)

        self.zcell = np.array(zcell, dtype=object)
        self.zvols = np.array(zvols)
        self.zhzn  = np.array(zhzn)
        self.depth = depth
        
        

        # Identify neighboring zones and the least-dense cells linking them
        # shape (2, num_zones, X)
        # neighbor_zone_IDs = zone_links[curr_zone_ID]
        # zlinks[0,...] is zone IDs
        # zlinks[1,...] is linkage volumes - watershed breakpoint for the
        # boundary between current zone and neighbor zone
        # For each zone i and its neighbors j
        # Identify neighboring zones (zlinks[0][i] has j in once it for every cell on their border?)
        # and the least-dense cells linking them (zlinks[1][i] has j copies of the maximum link volume between the zones?)
        zlinks = [[[] for _ in range(len(zvols))] for _ in range(2)] 

        if viz:
            #zverts = [[] for _ in range(len(zvols))]
            #znorms = [[] for _ in range(len(zvols))]
            zarea_0 = np.zeros(len(zvols)) # zone edge surface areas
            zarea_t = np.zeros(len(zvols)) # zone total surface areas
            zarea_s = [[] for _ in range(len(zvols))] # shared zone surfaces areas for each zone link
            
            # zone triangle data
            triangle_norms = []
            triangles_verts = []
            triangle_zones = []
            triangle_cells = []
            

        if verbose > 0:
            print("Linking zones...")
            
        link_time = time.time()
        
    	#loop through cells
        for i in range(len(vol)):
            #ns = nei[i]
            curr_neigh_idxs = cells[i].get_neighbors()
            
            z1 = gal_zone_IDs[i]
            
            if z1 == -1:
                continue

            # get cell info
            cell = cells[i]
            vertices = cell.get_vertices()
            faces = partition_face_vertices(cell)
            
            #loop though neighbor cells
            for n, face in zip(curr_neigh_idxs, faces):
                
                z2 = gal_zone_IDs[n]
                
                #Calculate edge area for cells on survey edges (z2==-1)
                if z2 == -1:

                    if viz:
        
                        # record the surface area and triangle data of the boundary formed by the vertices
                        if len(face)>2: #If there are at least 3 vertices in teh face (>=1 triangles)
    
                            face_vertices = vertices[face]
    
                            ##rotate the cell boundary into the x-y plane and obtain the boundary's triangles
                            simplices = Delaunay(rotate(face_vertices)).simplices
    
                            triangles = face[simplices]
                                
                            #calculate the triangle area
                            cell_center = coords[i] - vertices[triangles][:,0,:] #coordinates of voronoi cell center
                            edge_1 = vertices[triangles][:,1,:] - vertices[triangles][:,0,:] # triangle edges
                            edge_2 = vertices[triangles][:,2,:] - vertices[triangles][:,0,:]
                            cross = np.cross(edge_1, edge_2)
                            area = 0.5 * np.linalg.norm(cross,axis=1) #triangle area
                            normal = cross / np.expand_dims(area,axis=1) #triangle's normal vector
                            normal *= np.expand_dims(np.sign(np.diag(np.dot(cell_center, normal.T))), axis=1) # flip normals as needed
                            area_summed = np.sum(area) # ridge area where cells meet
                 
                            zarea_0[z1] += area_summed #add area to zone edge area
                            zarea_t[z1] += area_summed #add area to zone total area
                            
                            #  record triangles info
                            for normal_i, triangle_i in zip (normal, triangles):
                                triangle_norms.append(normal_i)
                                triangles_verts.append(triangle_i)
                                triangle_zones.append(z1)
                                triangle_cells.append(i)
                            
                    continue
                
                #Ensure neighboring cell is from a different zone
                if z1 == z2:
                    continue
                
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
                nl = np.amin([vol[i], vol[n]])
                ml = np.amax([zlinks[1][z1][j], nl])
                
                
                zlinks[1][z1][j] = ml
                zlinks[1][z2][k] = ml
                
                
                if viz and vol[i] > 0:            

                    # record the surface area and triangle data of the boundary formed by the vertices
                    if len(face)>2: #If there are at least 3 vertices shared between the cells (>=1 triangles)

                        face_vertices = vertices[face]

                        ##rotate the cell boundary into the x-y plane and obtain the boundary's triangles
                        simplices = Delaunay(rotate(face_vertices)).simplices

                        triangles = face[simplices]

			            #calculate the triangle area
                        cell_center = coords[i] - vertices[triangles][:,0,:] #coordinates of voronoi cell center
                        edge_1 = vertices[triangles][:,1,:] - vertices[triangles][:,0,:] # triangle edges
                        edge_2 = vertices[triangles][:,2,:] - vertices[triangles][:,0,:]
                        cross = np.cross(edge_1, edge_2)
                        area = 0.5 * np.linalg.norm(cross,axis=1) #triangle area
                        normal = cross / np.expand_dims(area,axis=1) #triangle's normal vector
                        normal *= np.expand_dims(np.sign(np.diag(np.dot(cell_center, normal.T))), axis=1) # flip normals as needed
                        area_summed = np.sum(area) # ridge area where cells meet
                    

                        zarea_t[z1] += area_summed #add ridge area to total zone surface area
                        zarea_s[z1][j] += area_summed # add ridge area to shared z1 z2 surface area
                        
                        # record triangles info
                        for normal_i, triangle_i in zip (normal, triangles):

                            triangle_norms.append(normal_i)
                            triangles_verts.append(triangle_i)
                            triangle_zones.append(z1)
                            triangle_cells.append(i)
        


        self.zlinks = zlinks
        
        print("Zone linking time: ", time.time() - link_time)
        
        ################################################################################
        # New implementation for zlinks
        ################################################################################
        '''
        zone_links = {}
        
        zone_link_volumes = np.zeros(len(zvols))
        
        for idx in range(num_gals):
            
            curr_neigh_idxs = cells[idx].get_neighbors()
            
            curr_zone_ID = gal_zone_IDs[idx]
            
            if curr_zone_ID == -1:
                continue
            
            for neigh_idx in curr_neigh_idxs:
                
                neigh_zone_ID = gal_zone_IDs[neigh_idx]
                
                #Ensure zone2 is valid
                if neigh_zone_ID == -1:
                    continue
                
                #Ensure neighboring cell is from a different zone
                if curr_zone_ID == neigh_zone_ID:
                    continue
        
                if curr_zone_ID not in zone_links:
                    zone_links[curr_zone_ID] = []
                    
                if neigh_zone_ID not in zone_links:
                    zone_links[neigh_zone_ID] = []
        
                zone_links[curr_zone_ID].append(neigh_zone_ID)
                zone_links[neigh_zone_ID].append(curr_zone_ID)
        
        
        '''
        
        #print("zlinks: ", len(zlinks), len(zlinks[0]), len(zlinks[1]), len(zlinks[0][0]), len(zlinks[0][1]))
        #print(zlinks[0][0])
        #print(zlinks[1][0])
        
        
        
        if viz:
            #self.zverts = zverts
            #self.znorms = znorms
            self.zarea_0 = zarea_0
            self.zarea_t = zarea_t
            self.zarea_s = zarea_s
            self.triangle_norms = np.array(triangle_norms)
            self.triangles = np.array(triangles_verts)	
            self.triangle_zones = np.array(triangle_zones) 
            self.triangle_cells = triangle_cells


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
        
        
        zvols  = np.array(zones.zvols) #Really the Zone Core(largest) Volumes
        zlinks = zones.zlinks

        # Sort zone links by volume, identify zones linked at each volume
        if verbose > 0:
            print("Sorting links...")

        # For each zone i and its neighbors j
        # Identify neighboring zones (zlinks[0][i] has j in once it for every cell on their border?)
        # and the least-dense cells linking them (zlinks[1][i] has j copies of the maximum link volume between the zones?)
        zl0   = np.array(list(flatten(zlinks[0])))
        zl1   = np.array(list(flatten(zlinks[1])))
        
        #print("Zl0")
        #print(zl0.shape)
        #print("Zl1")
        #print(zl1.shape)

        #zlu   = -1.*np.sort(-1.*np.unique(zl1))
        #largest to smallest zone max link volume
        #these are essentially the breakpoints for the watershed algorithm
        #for more dense zones to join into less dense zones
        zlu = np.sort(np.unique(zl1))[::-1] 
        #print("ZLU: ", zlu.shape)
        
        #At each breakpoint, a list of the unique zone IDs which will
        #begin to flow into someone else
        zlut  = [ np.unique( zl0[np.where(zl1==zl)[0]] ).tolist() for zl in zlu ]
        
        
        voids = []
        mvols = []
        ovols = []
        vlut  = np.arange(len(zvols))
        #mvlut = np.array(zvols)
        #ovlut = np.array(zvols)
        mvlut = zvols.copy()
        ovlut = zvols.copy()

        # For each zone-linking by descending link volume, create void from     
        # all zones and groups of zones linked at this volume except for that   
        # with the highest maximum cell volume (the "shallower" voids flow into 
        # the "deepest" void with which they are linked)
        if verbose > 0:
            print("Expanding voids...")

        #At each watershed breakpoint
        for i in range(len(zlu)):
            
            #Get the breakpoint volume
            lvol  = zlu[i]
            
            #For each zone which flows at this breakpoint, get the 
            #zone's largest cell volume
            mxvls = mvlut[zlut[i]]
            
            #Of the flowing zones, get the one with the largest
            #core volume
            mvarg = np.argmax(mxvls)
            
            mxvol = mxvls[mvarg]
            
            #For each zone which flows at this breakpoint
            for j in zlut[i]:
                
                # This is not the "deepest" zone or void being linked
                # aka not the largest core voronoi volume
                if mvlut[j] < mxvol:
                    
                    voids.append([])
                    ovols.append([])
                    
                    #Places where volumes match the flowing zone
                    #volume comparison?
                    vcomp = np.where(vlut==vlut[j])[0]
                    
                    # largest to smallest, the unique zone core volumes
                    # which match the current flowing zone
                    something = np.sort(np.unique(ovlut[vcomp]))[::-1]
                    
                    # Store void's "overflow" volumes, largest max cell volume, constituent zones
                    for ov in something:
                        
                        #places where zone 
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
        for ov in np.sort(np.unique(ovlut))[::-1]:
            ocomp = np.where(ovlut==ov)[0]
            voids[-1].append(ocomp.tolist())
            ovols[-1].append(ov)
        ovols[-1].append(0.)
        mvols.append(mvlut[0])

        '''
        ################################################################################
        # New implementation 
        ################################################################################
        num_breakpoints = len(zlu)
        
        for idx in range(num_breakpoints):
            
            breakpoint_vol = zlu[idx]

            flowing_zone_IDs = zlut[idx]

            flowing_zone_core_vols = mvlut[flowing_zone_IDs]
            
            largest_core_vol_idx = np.argmax(flowing_zone_core_vols)
            
            largest_flowing_core_vol = flowing_zone_core_vols[largest_core_vol_idx]


            for flowing_zone_ID in flowing_zone_IDs:

                if mvlut[j] < largest_flowing_core_vol:
                    pass
        '''



        self.voids = voids
        self.mvols = mvols
        self.ovols = ovols
        
        #print(len(voids))
        #print(len(mvols))
        #print(len(ovols))
        #print(voids[0:10])
        #print(mvols[0:10])
        #print(ovols[0:10])
        
        
        
        
        
        
