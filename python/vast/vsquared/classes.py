"""Utility classes for the ZOBOV algorithm using a voronoi tesselation of an
input catalog.
"""

import numpy as np
import healpy as hp
import time
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import ConvexHull, Voronoi, Delaunay, KDTree

from vast.vsquared.util import toCoord, mknumV2, rotate, partition_face_vertices
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
                 randfile,
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

        randfile: str
            Randoms catalog file (FITS format).
            
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
        
        self.weights = None if column_names['weight'] == "None" else galaxy_table[column_names['weight']]

        if randfile is not None:
            randoms_table = load_data_to_Table(randfile)
            if verbose > 0:
                print("Read in randoms (rows, cols): ", len(randoms_table), len(randoms_table.columns))
                print(randoms_table.columns)

            if column_names['weight'] != "None" and column_names['weight'] in randoms_table:
                self.weights_rand = randoms_table[column_names['weight']]
        
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

            # create array of galaxies
            self.coord = np.array([galaxy_table[column_names['x']],
                                   galaxy_table[column_names['y']],
                                   galaxy_table[column_names['z']]]).T
                                   
            self.cmin = cmin
            
            self.cmax = cmax

            # create array of randoms
            if randfile is not None:
                self.rand = np.array([randoms_table[column_names['x']],
                                      randoms_table[column_names['y']],
                                      randoms_table[column_names['z']]]).T
            
        else:

            # convert sky coordinates of galaxies to cartesian cooridnates
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

            # convert sky coordinates of randoms to cartesian cooridnates
            if randfile is not None:

                z_rand    = randoms_table[column_names['redshift']]
            
                ra_rand   = randoms_table[column_names['ra']]
                
                dec_rand  = randoms_table[column_names['dec']]
                
                zcut_rand = np.logical_and(z_rand > zmin, z_rand < zmax) # 1 if gal is in the zlims 0 if not
                
                if not zcut_rand.any():
                    print("Choose valid redshift limits for randoms", z_rand.min(), z_rand.max())
                    return
                
                #alias for zcut, unless magnitude limit is used, in which case scut will be later set to mcut
                # `scut` a boolean array to identify desired galaxies
                scut_rand = zcut_rand
                
                c1, c2, c3 = toCoord(z_rand, ra_rand, dec_rand, H0, Om_m)
                
                self.rand = np.array([c1, c2, c3]).T
    
                    
            
            
        
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

            if randfile is not None:
                
                mag = randoms_table[column_names['rabsmag']]
            
                mcut_rand = np.logical_and(mag < maglim, zcut_rand) # mcut is a subsample of zcut that removes galaxies outside the magnitude limit
                
                if not mcut_rand.any():
                    print("Choose valid magnitude limit for randoms")
                    return
                
                # scut is made into an alias for mcut, unless no magnitude limit 
                # is used, in which case it remains an alias for zcut
                # `scut` a boolean array to identify desired galaxies
                scut_rand = mcut_rand
                                
                
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

        if randfile is not None:
            self.rand = self.rand[scut_rand]
            if hasattr(self, 'weights_rand'):
                self.weights_rand = self.weights_rand[scut_rand]
        
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
                #read in existing mask
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
                 randoms_grid_size = 1.,
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

        randoms_grid_size : float
            The grid cell length for binning randoms in Mpc/h. Defaults to 1.
            
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
        
        # set mutlivoro radii values to a common value
        radii = 1*np.ones(coords.shape[0], dtype=np.float32)

        if periodic:
            # periodic mode
            periodic_boundaries = (True, True, True)
            limits = np.array([cat.cmin, cat.cmax])
            cmin = cat.cmin
            cmax = cat.cmax

        else:
            # survey and xyz mode
            # multivoro needs xyz limits for tessellation, so draw a box around the survey
            periodic_boundaries = (False, False, False)

            cmin = coords.min(axis=0)
            lower_min = cmin - 100.0

            cmax = coords.max(axis=0)
            upper_max = cmax + 100.0
            
            print("Lower min of bounding box: ", lower_min)
            print("Upper max of bounding box: ", upper_max)
            
            
            limits = np.empty((2,3), dtype=np.float32)
            limits[0,0] = lower_min[0]
            limits[0,1] = lower_min[1]
            limits[0,2] = lower_min[2]
            limits[1,0] = upper_max[0]
            limits[1,1] = upper_max[1]
            limits[1,2] = upper_max[2]
        
        
        #print("Radii: ", radii)
        #print("Limits: ", limits)

        ################################################################################
        # Compute Voronoi tessellation
        ################################################################################
        
        cells = compute_voronoi(
                                points=coords,
                                radii=radii,
                                limits=limits,
                                n_threads=num_cpus,
                                periodic_boundaries=periodic_boundaries,
                                )
        self.cells = cells
        
        print("Multivoro time: ", time.time() - multivoro_start)
        
        print("Number of cells: ", len(cells))

        
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

        output_volumes, edge_cells = self.calculate_region_volumes(self.cells,
                                                       r_max,
                                                       r_min,
                                                       mask_uint8,
                                                       xyz,
                                                       periodic,
                                                       cat.cmin,
                                                       cat.cmax,
                                                       nside
                                                       )
        
        self.volumes = output_volumes
        self.edge_cells = edge_cells
        
        if cat.weights is not None:
            weights = cat.weights[cat.nnls==np.arange(len(cat.nnls))]
            finite_density = self.volumes != 0.
            self.volumes[finite_density] = self.volumes[finite_density] / weights[finite_density]

        
        if hasattr(cat, "rand"):
            
            weights_rand = cat.weights_rand if hasattr(cat, 'weights_rand') else None

            # place randoms on grid
            grid_randoms, _ = np.histogramdd(cat.rand, 
                                   bins=(int(np.ceil((cmax[0]-cmin[0])/randoms_grid_size)),
                                         int(np.ceil((cmax[1]-cmin[1])/randoms_grid_size)),
                                         int(np.ceil((cmax[2]-cmin[2])/randoms_grid_size))),
                                   weights = weights_rand,
                                     )
            
            grid_randoms = grid_randoms / np.max(grid_randoms) # setup for upweighting Voronoi cell volumes
            galaxy_grid_indices = np.floor((coords - cmin)/randoms_grid_size).astype(int) # indices of galaxies on grid
            randoms_multiplier = grid_randoms[galaxy_grid_indices[:,0], galaxy_grid_indices[:,1], galaxy_grid_indices[:,2]] #weights for each galaxy from randoms
                
            finite_density = self.volumes != 0.
            if np.any(randoms_multiplier[finite_density]==0.):
                raise ValueError ('Provided randoms do not fill all grid cells. Try a larger randoms_grid_size value')
            self.volumes[finite_density] = self.volumes[finite_density] / randoms_multiplier[finite_density]
            
        
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
                                 periodic_mode,
                                 cmin,
                                 cmax,
                                 nside
                                 ):
        """
        This function essentially serves as a switch between single process
        and multiprocess calculation for calculating the region volume
        """

        
        if self.num_cpus == 1:
            
            output_volumes = np.zeros(self.num_gals, dtype=np.float64)
            edge_cells = np.zeros(self.num_gals, dtype=np.bool)

            for idx, cell in enumerate(cells):

    
                vertices = cell.get_vertices()

                if not xyz_mode and not periodic_mode:
    
                    ################################################################################
                    # We will need some radial information about the verticies to know whether to
                    # include those galaxies or not
                    ################################################################################
                    
                    vrh = np.linalg.norm(vertices, axis=1).astype(np.float64)
                    
                    #using <= and >= since original code inversely checked just > and <
                    if np.any(vrh <= r_min) or np.any(vrh >= r_max):
                        edge_cells[idx] = True
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
                        edge_cells[idx] = True
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
            
            edge_file_descriptor, EDGE_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared_edge", 
                                                               dir="/dev/shm", 
                                                               text=False)
            
            edge_buffer_length = self.num_gals
            
            os.ftruncate(edge_file_descriptor, edge_buffer_length)
            
            edge_buffer = mmap.mmap(edge_file_descriptor, 0)
            
            os.unlink(EDGE_BUFFER_PATH)
            
            edge_cells = np.frombuffer(edge_buffer, dtype=np.bool)

            edge_cells[:] = False
            
            edge_cells.shape = (self.num_gals,)
            
            
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
                                                  edge_file_descriptor,
                                                  r_max,
                                                  r_min,
                                                  mask_uint8,
                                                  xyz_mode,
                                                  periodic_mode,
                                                  cmin,
                                                  cmax,
                                                  nside
                                                  ))
                
                p.start()
                
                processes.append(p)
                
            
            for p in processes:
            
                p.join(None) #block till join
        
        return output_volumes, edge_cells
        
        
    def volume_calculation_worker(self, 
                                  max_indicies,
                                  index_coordinator,
                                  cells,
                                  volumes_fd,
                                  edge_file_descriptor,
                                  r_max,
                                  r_min,
                                  mask_uint8,
                                  xyz_mode,
                                  periodic_mode,
                                  cmin,
                                  cmax,
                                  nside
                                 ):
        
        #max_indices and num_gals are the same thing
        volumes_buffer_length = max_indicies*8 #float64 so 8 bytes per element
    
        volumes_buffer = mmap.mmap(volumes_fd, volumes_buffer_length)
        
        output_volumes = np.frombuffer(volumes_buffer, dtype=np.float64)
    
        output_volumes.shape = (self.num_gals,)

        edge_buffer_length = max_indicies # bool for 1 byte per element

        edge_buffer = mmap.mmap(edge_file_descriptor, edge_buffer_length)

        edge_cells = np.frombuffer(edge_buffer, dtype=np.bool)
    
        edge_cells.shape = (self.num_gals,)
        
        
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

            if not xyz_mode and not periodic_mode:

                ################################################################################
                # We will need some radial information about the verticies to know whether to
                # include those galaxies or not
                ################################################################################
                
                vrh = np.linalg.norm(vertices, axis=1).astype(np.float64)
                
                #using <= and >= since original code inversely checked just > and <
                if np.any(vrh <= r_min) or np.any(vrh >= r_max):
                    edge_cells[curr_index] = True
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
                    edge_cells[curr_index] = True
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
        
        ################################################################################
        # Unpack the results of the previous Catalog and Tesselation stages
        ################################################################################
        
        #Select the coordinates of the galaxies who pass the conditions
        #captured by the array 'nnls'
        # Not actually used in Zones, so lets skip the == computation
        #coords = catalog.coord[catalog.nnls==np.arange(len(catalog.nnls))] 
        
        # Array of shape (num_gals,) dtype float volume of that galaxy's voronoi cell
        gal_cell_vols = tess.volumes

        #Array of edge cell flags
        edge_cells = tess.edge_cells
        
        # List of length num_gals multivoro cell objects with methods describing the current
        # galaxy's cell
        cells = tess.cells
        
        #In Tess this value is filtered by the nnls array
        num_gals = tess.num_gals
        
        # Array of shape (num_gals,) dtype bool indicating whether that galaxy's cell is
        # an edge cell
        hzn = tess.hzn
        
        ################################################################################
        # Sort the Voronoi cells by their volume
        ################################################################################
        if verbose > 0:
            print("Sorting cells...")

        sort_order = np.argsort(gal_cell_vols)[::-1] #largest to smallest volume (aka lease dense to most dense)

        #vol2  = vol[srt] # cell volumes sorted from largest to smallest (aka least dense to most dense region)
        #nei2  = nei[srt] # coordinates of tetrahedra that include each galaxy, sorted from least dense to most dense region

        if viz:
            
            zarea_0 = {}
            zarea_t = {}
            zarea_s = {}
            
            
            # zone triangle data
            triangle_norms = []
            triangles_verts = []
            triangle_zones = []
            triangle_zone_links = []
        
        
        
        ################################################################################
        # Build zones from the cells - rewrite to use more concise data structure
        # Hopefully clearer data structures will make the downstream code easier to
        # modify/parallelize/etc, so for now, updating this and adding a dummy
        # section to convert the zone_info back into its older-version counterparts
        # until later
        #
        # Galaxies with their cells need to be grouped into Zones based on the volume
        # of their cell and their neighbor's cells
        # for each galaxy, track 
        #  - the ID of the zone it belongs to
        #  - its depth, the number of adjacent cells between it and the largest cell 
        #      in its zone
        #  - the grouping of galaxy indices which form a zone (zone_info[zone_ID]["galaxy_indices"])
        #  - the volume of the largest cell in the zone (zone_info[zone_ID]["largest_cell_volume"])
        #  - number of edge cells in the current zone (zone_info[zone_ID]["edge_cell_count"])
        #
        # Also repackage the multivoro cell neighbor information into arrays similar
        # to the scipy output so we can avoid calls to the get_neighbors() methods
        # and replace them with array accesses later
        #
        # Also keep track of the subset of galaxies/cells which actually touch other
        # zones so we dont have to go through as many later
        ################################################################################
        build_time = time.time()
        
        if verbose > 0:
            print("Building zones...")

        
        gal_zone_IDs = np.empty(num_gals, dtype=np.int32) 
        gal_zone_IDs.fill(-2) #init to -2, not 0
        
        depth = np.zeros(num_gals, dtype=int) 
        
        next_zone_ID = 0
        
        zone_info = {}
        
        zone_linkage_info = {}
        
        zone_link_volumes = {}
        
        degenerate_gal_cells = []
        
        for gal_idx in sort_order:

            if edge_cells[gal_idx]:
                gal_zone_IDs[gal_idx] = -1 
                #zone_info[-1].append(gal_idx)
                continue
            elif gal_cell_vols[gal_idx] == 0.:
                degenerate_gal_cells.append(gal_idx)
                continue

            curr_neigh_idxs = cells[gal_idx].get_neighbors()
            
            neigh_vols = gal_cell_vols[curr_neigh_idxs] 
            
            largest_neigh_vol_idx = curr_neigh_idxs[np.argmax(neigh_vols)] 

            
            # for void visualization,
            # get current galaxy's verticies and faces
            
            curr_vertices = cells[gal_idx].get_vertices()
            
            curr_faces = partition_face_vertices(cells[gal_idx])
            
            ################################################################################
            # if current cell is larger than all it's neighbors (aka the center of a zone)
            # it starts a new zone
            # Otherwise put this galaxy/cell into its least-dense neighbor's zone
            ################################################################################
            if gal_cell_vols[gal_idx] > gal_cell_vols[largest_neigh_vol_idx]:
                
                zone_ID = next_zone_ID
                next_zone_ID += 1
                
                gal_zone_IDs[gal_idx] = zone_ID 
                
                zone_info[zone_ID] = {}
                zone_info[zone_ID]["linked_zones"] = {}
                zone_info[zone_ID]["galaxy_indices"] = [gal_idx]
                zone_info[zone_ID]["largest_cell_volume"] = gal_cell_vols[gal_idx]
                zone_info[zone_ID]["edge_cell_count"] = hzn[gal_idx]
                
            else:
                
                zone_ID = gal_zone_IDs[largest_neigh_vol_idx] 
                
                gal_zone_IDs[gal_idx] = zone_ID
                
                depth[gal_idx] = depth[largest_neigh_vol_idx] + 1  #the galaxy's depth = its least dense neighbor's depth + 1
                
                zone_info[zone_ID]["galaxy_indices"].append(gal_idx)
                
                zone_info[zone_ID]["edge_cell_count"] += int(hzn[gal_idx])
                
            ################################################################################
            # Keep track of zone linkage volume as we build the zones
            ################################################################################
            neigh_zone_IDs = gal_zone_IDs[curr_neigh_idxs]
            
            for ndx, (neigh_zone_ID, neigh_face, neigh_idx) in enumerate(zip(neigh_zone_IDs, curr_faces, curr_neigh_idxs)):

                # Neighbor is outside the survey 
                if edge_cells[neigh_idx]:
                    
                    if viz:
        
                        # record the surface area and triangle data of the boundary formed by the vertices
                        if len(neigh_face)>2: #If there are at least 3 vertices in teh face (>=1 triangles)
    
                            # ordered face vertices
                            face_vertices = curr_vertices[neigh_face]
    
                            #calculate surfacearea and normal
                            normal_vector = np.sum(np.cross(face_vertices, np.roll(face_vertices, 1, axis=0)), axis=0)
                            normal_mag = np.linalg.norm(normal_vector)
                            area = 0.5 * normal_mag
                            normal_vector=normal_vector/normal_mag
    
                            zarea_0[zone_ID] = zarea_0.get(zone_ID, 0) + area #add area to zone edge area
                            zarea_t[zone_ID] = zarea_t.get(zone_ID, 0) + area #add area to zone total area
    
                            # get list of triangles
                            for tri_idx in range(1, len(face_vertices) - 1):
                                triangle = face_vertices[[0,tri_idx,tri_idx+1]]
    
                                triangle_norms.append(normal_vector)
                                triangles_verts.append(triangle)
                                triangle_zones.append(zone_ID)
                                triangle_zone_links.append(neigh_zone_ID)
                                
                    # Was an edge cell, so continue to the next neighbor
                    
                    continue

                # This neighbor hasn't been processed yet or is in the same zone. 
                if neigh_zone_ID == -2 or neigh_zone_ID == zone_ID:
                    continue

                neigh_idx = curr_neigh_idxs[ndx]
                
                key_lower = min(zone_ID, neigh_zone_ID)
                key_upper = max(zone_ID, neigh_zone_ID)
                
                zone_pair = (int(key_lower), int(key_upper))
                
                zone_linkage_info[zone_pair] = 1
                
                zone_info[zone_ID]["linked_zones"][neigh_zone_ID] = 1
                zone_info[neigh_zone_ID]["linked_zones"][zone_ID] = 1
                
                if viz:
                    zarea_s.setdefault((key_lower, key_upper), 0.)
                
                # if the chosen cell is less dense than the current least dense cell connecting the two zones
                # update the least dense cell connecting the two zones
                gal_volume = gal_cell_vols[gal_idx]
                neigh_volume = gal_cell_vols[neigh_idx]
                
                if gal_volume == neigh_volume:
                    
                    link_volume = gal_volume
                    link_gal = min(gal_idx, neigh_idx)
                    
                elif gal_volume < neigh_volume:
                    
                    link_volume = gal_volume
                    link_gal = gal_idx
                    
                else:
                    
                    link_volume = neigh_volume
                    link_gal = neigh_idx
                
                
                if zone_pair not in zone_link_volumes:
                    
                    zone_link_volumes[zone_pair] = 0.0
                
                if link_volume > zone_link_volumes[zone_pair]:
                    
                    zone_link_volumes[zone_pair] = link_volume

                
                if viz and gal_cell_vols[gal_idx] > 0:            

                    # record the surface area and triangle data of the boundary formed by the vertices
                    if len(neigh_face)>2: #If there are at least 3 vertices shared between the cells (>=1 triangles)

                        # ordered face vertices
                        face_vertices = curr_vertices[neigh_face]

                        #calculate surfacearea and normal
                        normal_vector = np.sum(np.cross(face_vertices, np.roll(face_vertices, 1, axis=0)), axis=0)
                        normal_mag = np.linalg.norm(normal_vector)
                        area = 0.5 * normal_mag
                        normal_vector=normal_vector/normal_mag

                        zarea_t[zone_ID] = zarea_t.get(zone_ID, 0) + area #add ridge area to total zone surface area
                        zarea_t[neigh_zone_ID] = zarea_t.get(neigh_zone_ID, 0) + area #add ridge area to neighbor's total zone surface area
                        zarea_s[(key_lower, key_upper)] += area # add ridge area to shared z1 z2 surface area
                        
                        # get list of triangles
                        for tri_idx in range(1, len(face_vertices) - 1):
                            triangle = face_vertices[[0,tri_idx,tri_idx+1]]

                            # add triangle for galaxy
                            triangle_norms.append(normal_vector)
                            triangles_verts.append(triangle)
                            triangle_zones.append(zone_ID)
                            triangle_zone_links.append(neigh_zone_ID)

                            # add triangle for neighbor
                            triangle_norms.append(-normal_vector)
                            triangles_verts.append(triangle)
                            triangle_zones.append(neigh_zone_ID)
                            triangle_zone_links.append(zone_ID)
            
        print("Zone building time: ", time.time() - build_time)
        
        if len(gal_zone_IDs[gal_zone_IDs==-2]) != 0:
            print('WARNING:', len(gal_zone_IDs[gal_zone_IDs==-2]), 'galaxies not processed by zone-building stage')
            
        if len(degenerate_gal_cells) != 0:
            print('WARNING:', len(degenerate_gal_cells), 'denerate galaxies detected')
            

        zone_IDs = np.arange(next_zone_ID)
        
        self.zone_info = zone_info
        self.zone_IDs = zone_IDs
        self.num_zones = next_zone_ID
        self.zone_link_volumes = zone_link_volumes
        self.depth = depth  
        
        
        if viz:
            self.zarea_0 = zarea_0 #np.array(list(zarea_0.values()))
            self.zarea_t = zarea_t # np.array(list(zarea_t.values()))
            self.zarea_s = zarea_s
            self.triangle_norms = np.array(triangle_norms)
            self.triangles = np.array(triangles_verts)	
            self.triangle_zones = np.array(triangle_zones) 
            self.triangle_zone_links = np.array(triangle_zone_links)




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
        
        
        zvols  = np.array([zones.zone_info[zone_ID]["largest_cell_volume"] for zone_ID in zones.zone_info.keys()]) #largest cell volume for each zone
        
        # zone_link_volumes has one entry for each zone link, each entry is a tuple of zones (keys)
        # and the least dense cell volume connecting them (values)
        zone_link_volumes = zones.zone_link_volumes

         # Sort zone links by volume, identify zones linked at each volume
        if verbose > 0:
            print("Sorting links...")

        # ---------------------------------------------------------------------------------
        # Construct the zone_links list. Each entry in zone_links is a watershed breakpoint
        # containing a list of the unique zone IDs which border the breakpoint
        # There may be more than two zones at a breakpoint.
        # ---------------------------------------------------------------------------------

        #largest to smallest zone linking volume
        link_volumes = np.sort(np.unique(list(zone_link_volumes.values())))[::-1] 
        
        #At each breakpoint, a list of the unique zone IDs which border the breakpoint
        link_volumes_dict = {}

        for pair, watershed_break in zone_link_volumes.items():
        
            if watershed_break not in link_volumes_dict:
                link_volumes_dict[watershed_break] = [] #explicitly creates a new list object for each watershed break
        
            if pair[0] not in link_volumes_dict[watershed_break]:
                link_volumes_dict[watershed_break].append(pair[0])
        
            if pair[1] not in link_volumes_dict[watershed_break]:
                link_volumes_dict[watershed_break].append(pair[1])  
                
        zone_links = list(link_volumes_dict.values())
        
        #print('lv1',len(zlinks[0]))
        voids = []
        mvols = []
        ovols = []
        vlut  = np.arange(len(zvols))
        mvlut = zvols.copy()
        ovlut = zvols.copy()

        # For each zone-linking by descending link volume, create void from     
        # all zones and groups of zones linked at this volume except for that   
        # with the highest maximum cell volume (the "shallower" voids flow into 
        # the "deepest" void with which they are linked)
        if verbose > 0:
            print("Expanding voids...")

        # At each watershed breakpoint in order of increasing density
        for i, link_volume in enumerate(link_volumes):
            
            #For each child void which borders this breakpoint, get the 
            # child's core volume
            mxvls = mvlut[zone_links[i]]
            
            #Of the selected children, get the one with the largest
            #core volume
            mvarg = np.argmax(mxvls)
            
            mxvol = mxvls[mvarg]
            
            #For each child which borders this breakpoint
            for j in zone_links[i]:
                
                # if the child doesn't have the largest core volume of the children
                if mvlut[j] < mxvol:

                    # create a new void in the hierarchy
                    voids.append([])
                    ovols.append([])

                    # for the current water height, get all zones that can be sequentially 
                    # linked to reach the current break point
                    vcomp = np.where(vlut==vlut[j])[0]
                    
                    # ordered largest to smallest, the linking volumes used to form each child void
                    # (or the core density in the case of an unlinked zone) 
                    overflow_volumes = np.sort(np.unique(ovlut[vcomp]))[::-1]
                    
                    # For each overflow volume
                    for overflow_volume in overflow_volumes:
                        
                        #select the zones that constitute the child void and add them as a list to the parent void
                        ocomp = np.where(ovlut[vcomp]==overflow_volume)[0]
                        
                        voids[-1].append(vcomp[ocomp].tolist())
                        # add the child's overflow volume to the parent's list of overflow volumes
                        ovols[-1].append(overflow_volume)
                        
                    ovols[-1].append(link_volume)
                    mvols.append(mvlut[j])
                    vlut[vcomp]  = vlut[zone_links[i]][mvarg]
                    mvlut[vcomp] = mxvol
                    ovlut[vcomp] = link_volume
        
        """
        # TODO: there are a few zones (e.g. 5 out of 600) that have no
        # zone links. These zones are discarded by VIDE but are made into 
        # voids by REVOLVER. Do we want to change this behavior at all?

        # Update (3/18/26): The zone-building stage has been redesigned
        # so it will need to be tested whether this differnece between 
        # REVOLVER and VIDE pruning persists. The below code for bringing
        # the two methods into agreement will likely no longer work with 
        # the new zone design
        
        # isolated voids
        for i in zone_link_volumes.keys():
            if len(zone_link_volumes[i])==0:
                if zvols[i] > 0:
                    pass
        """
                
        
        # Include the "deepest" void in the survey and its subvoids
        voids.append([])
        ovols.append([])
        for overflow_volume in np.sort(np.unique(ovlut))[::-1]:
            ocomp = np.where(ovlut==overflow_volume)[0]
            voids[-1].append(ocomp.tolist())
            ovols[-1].append(overflow_volume)
        ovols[-1].append(0.)
        mvols.append(mvlut[0])





        ################################################################################
        # Output
        ################################################################################

        self.voids = voids
        self.mvols = mvols
        self.ovols = ovols
        
        #print(len(voids))
        #print(len(mvols))
        #print(len(ovols))
        #print(voids[0:10])
        #print(mvols[0:10])
        #print(ovols[0:10])
        
        
        
        
        
        
