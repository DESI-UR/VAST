"""Implementation of the ZOnes Bordering on Voids (ZOBOV) algorithm.
"""

import numpy as np
import pickle
import configparser
from scipy import stats
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import time
import os
import mmap
import tempfile
import multiprocessing
from multiprocessing import Value, Process# Pool, shared_memory
from ctypes import c_int64

#from itertools import repeat

from vast.vsquared.util import toSky, \
                               num_coords_in_sphere, dcut_worker,\
                               wCen, wCen_worker,\
                               getSMA, getSMA_worker,\
                               P, \
                               galzone_worker, \
                               flatten, \
                               open_fits_file_V2, \
                               mknumV2
                               
from vast.vsquared.classes import Catalog, \
                                  Tesselation, \
                                  Zones, \
                                  Voids

class Zobov:
    """
    Description
    ===========
    Entrypoint to V^2.  Currently this class encapsulates the entirety
    of V^2 from loading a config file and data files, through tessellating, 
    watershed, pruning, and saving results to disk.  The primary computational
    methods are __init__() and sortVoids().
    
    """
    
    def __init__(self,
                 configfile,
                 #start=0,
                 #end=3,
                 custom_galaxy_table = None,
                 custom_cat_name=None,
                 stages=[0,1,2,3],
                 save_intermediate=True,
                 visualize=False,
                 periodic=False, 
                 num_cpus=1,
                 xyz=False,
                 capitalize_colnames=False,
                 verbose=0):
        """
        Description
        ===========
        Initialization of the ZOnes Bordering on Voids (ZOBOV) algorithm.
        This __init__ method does not really initalize a class so much as
        actually run the whole V^2 pipeline given `start` and `end` parameters
        representing the starting and ending stages to run.
        

        Parameters
        ==========
        
        configfile : str
            Configuration file path, for a config file in INI format.
        
        custom_galaxy_table : astropy table
            If not None, the provided galaxy table is used for the voidfinding
            rather than the galaxy input file in configfile. This is a 
            convenience feature meant for testing V2 in live environments 
            without first needing to save the galaxy input to a file. Final 
            runs should use the configfile settings instead.
            
        custom_cat_name : str
            If not none, a custom survey name is used, overrriding the value
            set in configfile. This is a convenience feature meant for testing 
            V2 in live environments without first needing to save the galaxy 
            input to a file. Final runs should use the configfile settings 
            instead.
            
        stages : list of integers
            0=generate catalog, 
            1=generate tesselation, 
            2=generate zones, 
            3=generate voids, 
            Example: stages=[1,2,3] will attempt to load a previously pickled
            catalog object from a previous run with save_intermediate=True, and
            then run the tesselation, zones creation, and voids creation stages
            stages=[2,3] will attempt to load a previously pickled tesselation
            object and then run the zones and voids creation
            Default stages=[0,1,2,3] to run all four stages.
            
        save_intermediate : bool
            If true, pickle and save intermediate outputs.
            
        visualize : bool
            If True, tell the Zones class to create the output data
            necessary to visualize the V^2 output using the VAST/VoidRender
            OpenGL tool.
            
        periodic : bool
            Use periodic boundary conditions.
            In Periodic mode, galaxy coordinates currently must be provided in 
            cartesian/xyz format.  In non-periodic mode, provide them in
            ra/dec/redshift
            
        num_cpus : int
            number of cpus to leverage for computations
            
         xyz : bool
            Use rectangular boundary conditions.

        capitalize_colnames : bool
            If True, column names in ouput file are capitalized. If False, column names are lowercase
        """
        
        self.verbose = verbose
        
        self.num_cpus = num_cpus
        
        ################################################################################
        # Some basic parameter sanity checks
        # make sure start and end stages are within bounds
        # make sure `visualize` is set correctly for periodic mode
        ################################################################################
        
        #if start not in [0,1,2,3,4] or end not in [0,1,2,3,4] or end<start:
        #    print("Choose valid stages")
        #    return
        
        if visualize*periodic:
            print("Visualization not implemented for periodic boundary conditions: changing to false")
            self.visualize = False
        else:
            self.visualize = visualize
            
        self.periodic = periodic
        self.xyz = False if periodic*xyz or not xyz else True

        ################################################################################
        # Load the config INI file from disk
        ################################################################################

        
        config = configparser.ConfigParser()
        
        config.read(configfile)


        ################################################################################
        # Extract some values from the config INI file 
        ################################################################################
        self.infile  = config['Paths']['Input Catalog'] if custom_cat_name is None else 'None'

        self.randfile  = config['Paths']['Input Randoms']
        if self.randfile == "None": self.randfile = None

        self.catname = config['Paths']['Survey Name'] if custom_cat_name is None else custom_cat_name
        
        self.outdir  = config['Paths']['Output Directory']
        
        self.intloc  = self.outdir +"/intermediate/" + self.catname

        self.maskfile  = config['Paths']['Mask File']
        if self.maskfile == "None": self.maskfile = None
        
        self.H0   = float(config['Cosmology']['H_0'])
        
        self.Om_m = float(config['Cosmology']['Omega_m'])
        
        self.Kos = FlatLambdaCDM(self.H0, self.Om_m)
        
        self.zmin   = float(config['Settings']['redshift_min'])
        
        self.zmax   = float(config['Settings']['redshift_max'])
        
        self.minrad = float(config['Settings']['radius_min'])
        
        self.zstep  = float(config['Settings']['redshift_step'])
        
        self.nside  = int(config['Settings']['nside'])
        
        self.maglim = config['Settings']['rabsmag_min']
        self.maglim = None if self.maglim == "None" else float(self.maglim)
        
        self.cmin = np.array([float(config['Settings']['x_min']),float(config['Settings']['y_min']),float(config['Settings']['z_min'])])
        
        self.cmax = np.array([float(config['Settings']['x_max']),float(config['Settings']['y_max']),float(config['Settings']['z_max'])])
        
        self.buff = float(config['Settings']['buffer'])
        
        self.column_names = config['Galaxy Column Names']
        
        
        
        ################################################################################
        # Some additional sanity checks
        ################################################################################
        if self.periodic and self.maglim is not None:
            #Right now maglim uses zcut which is only produced in
            #non-periodic mode
            print("WARNING: using maglim in periodic mode which utilizes redshift information")
        
        ################################################################################
        # Now that we've got the necessary values extracted from the config file, we
        # can initialize the FITS output file information
        # HDU = Header+Data Unit
        # HDUH = Header+Data Unit Header
        ################################################################################
        hdu = fits.PrimaryHDU(header=fits.Header())
        
        hduh = hdu.header
        
        self.hdu = hdu
        
        self.initialize_fits_hdu_header(hduh)




        run_stage_0 = 0 in stages
        self.create_catalog(run_stage_0, save_intermediate, custom_galaxy_table=custom_galaxy_table)
        
        run_stage_1 = 1 in stages
        self.create_tessellation(run_stage_1, save_intermediate)
        
        run_stage_2 = 2 in stages
        self.create_zones(run_stage_2, save_intermediate)
        
        run_stage_3 = 3 in stages
        self.create_prevoids(run_stage_3, save_intermediate)
        
        ################################################################################
        #
        ################################################################################
        #self.catalog = ctlg
        
        #if end>0:
        #    self.tesselation = tess
        #if end>1:
        #    self.zones       = zones
        #if end>2:
        #    self.prevoids    = voids
            
        self.capitalize = capitalize_colnames


    def create_catalog(self, run_stage, save_intermediate=False, custom_galaxy_table=None):
        """
        Description
        ===========
        
        Given an indicator whether we're running this stage or loading
        results from this stage from disk, do the appropriate running
        or loading of data, and if running, potentially save the
        output as an intermediate result
        """
        
        if run_stage:
            
            if self.verbose > 0:
                start_time = time.time()
            
            ctlg = Catalog(catfile=self.infile,
                           randfile = self.randfile,
                           nside=self.nside,
                           zmin=self.zmin,
                           zmax=self.zmax,
                           column_names=self.column_names, 
                           custom_galaxy_table=custom_galaxy_table,
                           maglim=self.maglim,
                           H0=self.H0,
                           Om_m=self.Om_m,
                           periodic=self.periodic,
                           xyz=self.xyz,
                           cmin=self.cmin,
                           cmax=self.cmax, 
                           zobov=self,
                           maskfile = self.maskfile,
                           verbose=self.verbose)
            
            if self.verbose > 0:
                print("Catalog creation time: ", time.time() - start_time)
            
            self.catalog = ctlg
            
            if save_intermediate:
                pickle.dump(ctlg,open(self.intloc+"_ctlg.pkl",'wb'))
        else:
            self.catalog = pickle.load(open(self.intloc+"_ctlg.pkl",'rb'))
        
        return None
        
        
    def create_tessellation(self, run_stage, save_intermediate=False):
        """
        Description
        ===========
        
        Given an indicator whether we're running this stage or loading
        results from this stage from disk, do the appropriate running
        or loading of data, and if running, potentially save the
        output as an intermediate result
        """
        
        if run_stage:
            
            if self.verbose > 0:
                start_time = time.time()
            
            tess = Tesselation(self.catalog,
                               self.nside,
                               viz=self.visualize,
                               periodic=self.periodic,
                               xyz=self.xyz,
                               num_cpus=self.num_cpus, 
                               buff=self.buff,
                               verbose=self.verbose)
            
            if self.verbose > 0:
                print("Tesselation creation time: ", time.time() - start_time)
            
            self.tessellation = tess
            
            if save_intermediate:
                pickle.dump(tess,open(self.intloc+"_tess.pkl",'wb'))
        else:
            self.tessellation = pickle.load(open(self.intloc+"_tess.pkl",'rb'))
        
        return None


    def create_zones(self, run_stage, save_intermediate=False):
        """
        Description
        ===========
        
        Given an indicator whether we're running this stage or loading
        results from this stage from disk, do the appropriate running
        or loading of data, and if running, potentially save the
        output as an intermediate result
        """
        
        if run_stage:
            
            if self.verbose > 0:
                start_time = time.time()
            
            zones = Zones(self.tessellation, 
                          viz=self.visualize,
                          catalog = self.catalog,
                          verbose=self.verbose)
            
            if self.verbose > 0:
                print("Zones creation time: ", time.time() - start_time)
            
            self.zones = zones
            
            if save_intermediate:
                pickle.dump(zones,open(self.intloc+"_zones.pkl",'wb'))
        else:
            self.zones = pickle.load(open(self.intloc+"_zones.pkl",'rb'))
        
        return None
    
    
    def create_prevoids(self, run_stage, save_intermediate=False):
        """
        Description
        ===========
        
        Given an indicator whether we're running this stage or loading
        results from this stage from disk, do the appropriate running
        or loading of data, and if running, potentially save the
        output as an intermediate result
        """
        
        if run_stage:
            
            if self.verbose > 0:
                start_time = time.time()
                
            voids = Voids(self.zones,
                          verbose=self.verbose)
            
            if self.verbose > 0:
                print("Prevoids creation time: ", time.time() - start_time)
            
            self.prevoids = voids
            
            if save_intermediate:
                pickle.dump(voids,open(self.intloc+"_voids.pkl",'wb'))
        else:
            self.prevoids = pickle.load(open(self.intloc+"_voids.pkl",'rb'))
        
        return None
    


    def initialize_fits_hdu_header(self, hduh):
        """
        Description
        ===========
        
        Initialize the FITS Header+Data Unit Header given the values
        from the config INI file which have been extracted to class
        members on this object
        """
        
        
        hduh['INFILE'] = (self.infile.split('/')[-1], 'Input Galaxy Table') #split directories by '/' and take the filename at the end

        if self.randfile is not None:
            hduh['RANDFILE'] = (self.randfile.split('/')[-1], 'Input Randoms Catalog')

        if self.maskfile is not None:
            hduh['MASKFILE'] = (self.maskfile.split('/')[-1], 'Input Angular Mask')
        
        hduh['HP'] = (self.H0/100, 'Reduced Hubble Parameter h (((km/s)/Mpc)/100)')
        
        hduh['OMEGAM'] = (self.Om_m,'Matter Density')
        
        hduh['ZLIML'] = (mknumV2(self.zmin), 'Lower Redshift Limit')
        
        hduh['ZLIMU'] = (mknumV2(self.zmax), 'Upper Redshift Limit')
        
        hduh['DLIML'] =  (self.Kos.comoving_distance(self.zmin).value, 'Lower Distance Limit (Mpc/h)')
        
        hduh['DLIMU'] =  (self.Kos.comoving_distance(self.zmax).value, 'Upper Distance Limit (Mpc/h)')
        
        hduh['MINR'] = (mknumV2(self.minrad), ' Minimum Void Radius (Mpc/h)')
        
        hduh['ZSTEP'] = (mknumV2(self.zstep), 'Step Size for r-to-z Lookup Table')
        
        hduh['NSIDE'] = (self.nside, 'NSIDE for HEALPix Pixelization')
        
        hduh['MAGLIM'] = (mknumV2(self.maglim), 'Magnitude Limit (dex)')
        
        hduh['PXMIN'] = (mknumV2(self.cmin[0]), 'Lower X-limit for Periodic Boundary Conditions')
        hduh['PYMIN'] = (mknumV2(self.cmin[1]), 'Lower Y-limit for Periodic Boundary Conditions')
        hduh['PZMIN'] = (mknumV2(self.cmin[2]), 'Lower Z-limit for Periodic Boundary Conditions')
        
        
        hduh['PXMAX'] = (mknumV2(self.cmax[0]), 'Upper X-limit for Periodic Boundary Conditions')
        hduh['PYMAX'] = (mknumV2(self.cmax[1]), 'Upper Y-limit for Periodic Boundary Conditions')
        hduh['PZMAX'] = (mknumV2(self.cmax[2]), 'Upper Z-limit for Periodic Boundary Conditions')
        
        hduh['BUFFER'] = (mknumV2(self.buff), 'Periodic Buffer Shell Width (Mpc/h)')
        
        return None
        
        
        

    def sortVoids(self, 
                  method=0, 
                  minsig=2, 
                  zone_linking_cut=0.2, 
                  central_density_cut=None,
                  apply_mgs_cut = False,
                  apply_median_radius_cut = False
                 ):
        """
        Description
        ===========
        
        Sort voids according to one of several methods.

        Parameters
        ==========

        method : int or string
            0 or VIDE or vide = VIDE method (arXiv:1406.1191); link zones with density less than zone_linking_cut * mean density
            1 or ZOBOV or zobov = ZOBOV method (arXiv:0712.3049); keep full void hierarchy.
            2 or ZOBOV2 or zobov2 = ZOBOV method; cut voids over a significance threshold.
            3 = not available
            4 or REVOLVER or revolver = REVOLVER method (arXiv:1904.01030); every zone below mean density is a void.
        
        minsig : float
            Minimum significance threshold for selecting voids. This value is only used when method=2

        zone_linking_cut : float
            Density cut for linking zones using VIDE method. This value is only used when method=0. When used,
            zone_linking_cut should be set to a value between 0 and 1, representing the fraction of the mean density
            used for the zone-linking threshold. A value of 0.2 may be used to match the VIDE pruning choices found 
            in arXiv:1406.1191 and arXiv:2202.01226
            
        central_density_cut : float or None
            Density cut for filtering voids from the final catalog. If set to None (default value), no cut is applied.
            If set to a float between 0 and 1, central_density_cut represents the fraction of the mean density used as 
            the threshold for filtering voids. Voids whose central densities are more dense than the threshold will be 
            cut. A value of 0.2 may be used to match the VIDE pruning choices found in arXiv:1406.1191 and 
            arXiv:2202.01226

        apply_mgs_cut : bool
            If True, voids with radii smaller than the mean galaxy separation are cut from the catalog.
            Defaults to False, meaning no cut is applied. Setting the cut to True will match the VIDE pruning choices 
            found in arXiv:1406.1191 and arXiv:2202.01226

        apply_median_radius_cut : bool
            If True, only the 50% largest voids returned. Defaults to False. Setting the cut to True will match the 
            analysis choices made in arXiv:1904.01030 and the REVOVLER pruning definiton used in arXiv:2202.01226
        """

        # ------------------------------------------------------------------------------------------------------
        # Ensure appropriate user settings
        # ------------------------------------------------------------------------------------------------------
       
        if isinstance(method, str):
            try:
                method = int(method)
            except:
                if method == 'VIDE' or method == 'vide':
                    method = 0
                if method == 'ZOBOV' or method == 'zobov':
                    method = 1
                if method == 'ZOBOV2' or method == 'zobov2':
                    method = 2
                if method == 'REVOLVER' or method == 'revolver':
                    method = 4

        if not hasattr(self, 'prevoids'):
            if method != 4:
                print("Run all stages of Zobov first")
                return
            else:
                if not hasattr(self, 'zones'):
                    print("Run all stages of Zobov first")
                    return

        # Selecting void candidates
        if self.verbose > 0:
            print("Selecting void candidates...")
            start_time = time.time()
        
        # mean cell volume 
        # TODO: for sky surveys, make this a function of the radial dnesity profile
        # rahter than a fixed value
        minvol = np.mean(self.tessellation.volumes[self.tessellation.volumes>0])

        if method == 0: #VIDE
            # zone-linking theshold
            minvol_scaled = minvol/zone_linking_cut
            
            voids  = []
            #print('lv0',len(self.prevoids.ovols))
            # for each void candidate
            for i in range(len(self.prevoids.ovols)):
                vl = self.prevoids.ovols[i]
                vbuff = []
                # for each child void bordering the link
                for j in range(len(vl)-1):
                    # add the deepest child to the void and any other
                    # children that meet the threshold condition
                    if j > 0 and vl[j] < minvol_scaled:
                        break
                    vbuff.extend(self.prevoids.voids[i][j])
                voids.append(vbuff)
            
        elif method == 1: #ZOBOV
            
            voids = [[c for q in v for c in q] for v in self.prevoids.voids]

        elif method == 2: #ZOBOV2
            
            voids = []
            for i in range(len(self.prevoids.mvols)):
                vh = self.prevoids.mvols[i]
                vl = self.prevoids.ovols[i][-1]

                r  = vh / vl
                p  = P(r)

                if stats.norm.isf(p/2.) >= minsig:
                    voids.append([c for q in self.prevoids.voids[i] for c in q])

        elif method==3: #UNKNOWN
            #
            # Method 3 is documented as not available
            # will need to consult with dveyrat or others on what
            # this section means.  Commenting out for now.
            #
            #raise NotImplementedError
            
            voids = []
            for i in range(len(self.prevoids.mvols)):
                vh = self.prevoids.mvols[i]
                vl = np.amax(self.zones.zlinks[1][self.prevoids.voids[i][0][0]])
                r  = vh / vl
                p1 = P(r)
                for j in range(len(self.prevoids.voids[i])):
                    if j == len(self.prevoids.voids[i])-1:
                        voids.append([c for q in self.prevoids.voids[i] for c in q])
                    else:
                        vl = self.prevoids.ovols[i][j+2]
                        r  = vh / vl
                        p2 = P(r)
                        p3 = 1.
                        for zid in self.prevoids.voids[i][j+1]:
                            vhz = np.amax(self.zones.zone_info[zid]["largest_cell_volume"])
                            vlz = np.amax(self.zones.zlinks[1][zid])
                            rz  = vhz / vlz
                            p3  = p3 * P(rz)
                        if p2 > p1*p3:
                            voids.append([c for q in self.prevoids.voids[i][:j+1] for c in q])
                            break
                        else:
                            p1 = p2
            
        
        elif method == 4: #REVOLVER
            
            voids = np.arange(len(self.zones.zone_info)).reshape(len(self.zones.zone_info),1).tolist()
            
        else:
            print("Choose a valid method")
            return

        if self.verbose > 0:
            print('Void candidates selected...')

        
        #for every void in hierarchy (VIDE) or for every zone (REVOLVER), the cells that compose it
        zcell = np.array([self.zones.zone_info[zone_ID]["galaxy_indices"] for zone_ID in self.zones.zone_info.keys()], dtype=object)
        vcuts = [list(flatten(zcell[v])) for v in voids]

        gcut  = np.arange(len(self.catalog.coord))[self.catalog.nnls==np.arange(len(self.catalog.nnls))]
        
        cutco = self.catalog.coord[gcut]

        # Build array of void volumes
        vvols = np.array([np.sum(self.tessellation.volumes[vcut]) for vcut in vcuts])

        # Calculate effective radii of the voids
        vrads = (vvols*3/(4*np.pi))**(1/3)
        if self.verbose > 0:
            print('Effective void radius calculated')

        # ------------------------------------------------------------------------------------------------------
        # User-defined cuts on void radii
        # ------------------------------------------------------------------------------------------------------
       
        # Cut all voids with radii smaller than set minimum  
        # note: if self.minrad = 0, then one zone will still be cut, corresponding to galaxies with 0 cell volume
        # (aka edge galaxies that are not placed in voids). This behavior is intended.
        rcut  = vrads > self.minrad

        # optionally cut on median radius
        if apply_median_radius_cut:
            rcut *= vrads > np.median(vrads)

        # optionally remove voids smaller than the mean cell size
        if apply_mgs_cut:
            rcut *= vrads>(minvol)**(1./3)
        
        # apply radial cuts
        
        voids = np.array(voids, dtype=object)[rcut]
        vcuts = [vcuts[i] for i in np.arange(len(rcut))[rcut]] # vcuts is a list
        vvols = vvols[rcut]
        vrads = vrads[rcut]
        
        if self.verbose > 0:
            print('Removed voids smaller than', self.minrad, 'Mpc/h')

        # ------------------------------------------------------------------------------------------------------
        # Identify void centers.
        # ------------------------------------------------------------------------------------------------------
       
        if self.verbose > 0:
            print("Finding void centers...")
        if self.num_cpus == 1:
            vcens = np.array([wCen(self.tessellation.volumes[vcut], cutco[vcut], self.periodic, self.cmin, self.cmax) for vcut in vcuts])
        else:
            #parallel version

            # set up shared memory for parallel processes and then run processes
                
            num_voids = len(vcuts)
            
            index_coordinator = Value(c_int64, 0, lock=True)

            file_descriptor, ARRAY_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared_vol", 
                                                               dir="/dev/shm", 
                                                               text=False)
            
            buffer_length = num_voids*8*3
            
            os.ftruncate(file_descriptor, buffer_length)
            
            array_buffer = mmap.mmap(file_descriptor, 0)
            
            os.unlink(ARRAY_BUFFER_PATH)
            
            vcens = np.frombuffer(array_buffer, dtype=np.float64)
            
            vcens[:] = 0
    
            vcens.shape = (num_voids, 3)
            
            startup_context = multiprocessing.get_context("fork")
                
            processes = []
            
            for proc_idx in range(self.num_cpus):

                p = startup_context.Process(target=wCen_worker, 
                                            args=(num_voids, 
                                                  index_coordinator, 
                                                  file_descriptor,
                                                  vcuts,
                                                  self.tessellation.volumes, 
                                                  cutco, 
                                                  self.periodic, 
                                                  self.cmin, 
                                                  self.cmax
                                                  ))
                
                p.start()
                
                processes.append(p)
                
            
            for p in processes:
            
                p.join(None) #block till join
        
        # ------------------------------------------------------------------------------------------------------
        # Apply central density cut
        # ------------------------------------------------------------------------------------------------------
        
        if central_density_cut is not None:
            # central density threshold 
            minvol_scaled = minvol / central_density_cut

            if self.verbose > 0:
                print("Cutting on central density...")
            
            # Apply central density cut 
            # -----------------------
            if self.num_cpus == 1:
                # number of galaxies within 1/4th of the void radius divided by volume 4/3*pi*(R/4)^3
                # should be less than the user specified fraction of the mean density
                dcut = np.array([64.*num_coords_in_sphere(vcens[i], vrads[i]/4., cutco, self.periodic, self.cmin, self.cmax)/vvols[i] for i in range(len(vrads))])<1./minvol_scaled
            else:
                #parallel version

                # set up shared memory for parallel processes and then run processes
                
                num_voids = len(vrads)
                
                index_coordinator = Value(c_int64, 0, lock=True)
    
                file_descriptor, ARRAY_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared_dcut", 
                                                                   dir="/dev/shm", 
                                                                   text=False)
                
                buffer_length = num_voids # 1 byte bool
                
                os.ftruncate(file_descriptor, buffer_length)
                
                array_buffer = mmap.mmap(file_descriptor, 0)
                
                os.unlink(ARRAY_BUFFER_PATH)
                
                dcut = np.frombuffer(array_buffer, dtype=bool)
                
                dcut[:] = 0
        
                dcut.shape = (num_voids,)
                
                startup_context = multiprocessing.get_context("fork")
                    
                processes = []
                
                for proc_idx in range(self.num_cpus):
    
                    p = startup_context.Process(target=dcut_worker, 
                                                args=(num_voids, 
                                                      index_coordinator, 
                                                      file_descriptor,
                                                      vcens,
                                                      vrads,
                                                      cutco,
                                                      vvols,
                                                      minvol_scaled,
                                                      self.periodic, 
                                                      self.cmin, 
                                                      self.cmax
                                                      ))
                    
                    p.start()
                    
                    processes.append(p)
                    
                
                for p in processes:
                
                    p.join(None) #block till join

            vcuts = [vcuts[i] for i in np.arange(len(dcut))[dcut]] # vcuts is a list
            vrads = vrads[dcut]
            vcens = vcens[dcut]
            voids = voids[dcut]
            del vvols # vvols is not needed anymore so delete it rather than propogating cuts
        
        # ------------------------------------------------------------------------------------------------------
        # Edge-void calculations
        # ------------------------------------------------------------------------------------------------------
        
        if self.verbose > 0:
            print("Determining edge voids...")
        
        if self.visualize:
            varea_0 = [np.sum([self.zones.zarea_0.get(zone_ID, 0) for zone_ID in voi]) for voi in voids]
            varea_t = [np.sum([self.zones.zarea_t[zone_ID] for zone_ID in voi]) for voi in voids]
            varea_s = np.zeros(len(voids))
            for i in range(len(voids)):
                if len(voids[i])==1:
                    continue
                for j in range(len(voids[i])-1):
                    z1 = voids[i][j]
                    for k in range(j+1,len(voids[i])):
                        z2 = voids[i][k]
                        if z2 in self.zones.zone_info[z1]["linked_zones"]:
                            key_lower = min(z1, z2)
                            key_upper = max(z1, z2)
                            varea_s[i] += self.zones.zarea_s[(key_lower, key_upper)]
        else:
            zhzn = np.array([self.zones.zone_info[zone_ID]["edge_cell_count"] for zone_ID in self.zones.zone_info.keys()])
            vhzn = [np.sum(zhzn[np.array(voi, dtype=int)]) for voi in voids]

        # ------------------------------------------------------------------------------------------------------
        # Identify eigenvectors of best-fit ellipsoid for each void.
        # ------------------------------------------------------------------------------------------------------
        
        if self.verbose > 0:
            print("Calculating ellipsoid axes...")

        if self.num_cpus == 1:
            vaxes = np.array([getSMA(vrads[i], vcens[i], cutco[vcuts[i]], self.periodic, self.cmin, self.cmax) for i in range(len(vrads))])
        else:
            #parallel version

            # set up shared memory for parallel processes and then run processes
                
            num_voids = len(vrads)
            
            index_coordinator = Value(c_int64, 0, lock=True)

            file_descriptor, ARRAY_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared_ell", 
                                                               dir="/dev/shm", 
                                                               text=False)
            
            buffer_length = num_voids*8*3*3
            
            os.ftruncate(file_descriptor, buffer_length)
            
            array_buffer = mmap.mmap(file_descriptor, 0)
            
            os.unlink(ARRAY_BUFFER_PATH)
            
            vaxes = np.frombuffer(array_buffer, dtype=np.float64)
            
            vaxes[:] = 0
    
            vaxes.shape = (num_voids, 3,3)
            
            startup_context = multiprocessing.get_context("fork")
                
            processes = []
            
            for proc_idx in range(self.num_cpus):

                p = startup_context.Process(target=getSMA_worker, 
                                            args=(num_voids,
                                                index_coordinator,
                                                file_descriptor,
                                                vrads,
                                                vcens,
                                                vcuts,
                                                cutco, 
                                                self.periodic, 
                                                self.cmin, 
                                                self.cmax,
                                               ))
                
                p.start()
                
                processes.append(p)
                
            
            for p in processes:
            
                p.join(None) #block till join

        # ------------------------------------------------------------------------------------------------------
        # Calculate zone information
        # ------------------------------------------------------------------------------------------------------

        if self.verbose > 0:
            print("Calculating zone information...")

        # zvoid holds smallest parent void in void hierarchy and largest parent void in void hierarchy
        # for each zone
        zvoid = [[-1,-1] for _ in range(len(self.zones.zone_info))]
        
        #iterate over voids
        for i in range(len(voids)):
            
            #iterate over zones in void
            for j in voids[i]:
                
                # if the zone is marked as in a void
                if zvoid[j][0] > -0.5:
                    
                    # if the current void has fewer zones than the marked void
                    if len(voids[i]) < len(voids[zvoid[j][0]]):
                        
                        #update the lowest level void in the hierarchy that contains the zone
                        zvoid[j][0] = i
                        
                    # if the current void has more zones than the marked void
                    elif len(voids[i]) > len(voids[zvoid[j][1]]):
                        
                        #update the highest level void in the hierarchy that contains the zone
                        zvoid[j][1] = i
                
                # if the zone not is marked as in a void, update both entries in zvoid
                else:
                    zvoid[j][0] = i
                    zvoid[j][1] = i

        # record the calculated info
        self.vrads = vrads
        self.vcens = vcens
        self.vaxes = vaxes
        self.zvoid = np.array(zvoid)
        self.method = method
        
        
        if self.verbose > 0:
            print("SortVoids time: ", time.time() - start_time)

        if self.visualize:
            self.varea_0 = np.array(varea_0)
            self.varea_t = np.array(varea_t)-varea_s
        else:
            self.vhzn = (np.array(vhzn)).astype(bool)



    def saveVoids(self):
        """
        Description
        ===========
        Output calculated voids to a FITS file 
        [catalogname]_V2_[pruning method]_Output.fits
        """
        
        if self.verbose > 0:
            start_time = time.time()
        
        if not hasattr(self,'vcens'):
            print("Sort voids first")
            return
        
        vcen = self.vcens.T
        vax1 = np.array([vx[0] for vx in self.vaxes]).T
        vax2 = np.array([vx[1] for vx in self.vaxes]).T
        vax3 = np.array([vx[2] for vx in self.vaxes]).T

        # format output tables
        if self.periodic:
            names = ['void','x','y','z','radius', 'x1','y1','z1','x2','y2','z2','x3','y3','z3']
            if self.capitalize:
                names = [name.upper() for name in names]
            vT = Table([np.arange(len(self.vrads)),vcen[0],vcen[1],vcen[2],self.vrads,vax1[0],vax1[1],vax1[2],vax2[0],vax2[1],vax2[2],vax3[0],vax3[1],vax3[2]],
                    names = names,
                    units = ['','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h'])
        else:
            vz,vra,vdec = toSky(self.vcens,self.H0,self.Om_m,self.zstep)
            columns = [np.arange(len(self.vrads)), vcen[0], vcen[1], vcen[2], 
                       vz, vra, vdec, self.vrads,  
                       vax1[0], vax1[1], vax1[2], vax2[0], vax2[1], vax2[2], vax3[0], vax3[1], vax3[2]]
            names = ['void','x','y','z',
                     'redshift','ra','dec','radius','x1','y1','z1','x2','y2','z2','x3','y3','z3']
            units = ['','Mpc/h','Mpc/h','Mpc/h','','deg','deg','Mpc/h', 'Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h']

            if self.visualize:
                columns += [self.varea_t,self.varea_0]
                names += ['tot_area','edge_area']
                units += ['(Mpc/h)^2','(Mpc/h)^2']
            else:
                columns.append((self.vhzn).astype(int))
                names.append('edge')
                units.append('')
            if self.capitalize:
                names = [name.upper() for name in names]
            
            vT = Table(columns, names = names, units = units)
        
        names = ['zone','void0','void1']
        if self.capitalize:
            names = [name.upper() for name in names]
        vZ = Table([np.array(range(len(self.zvoid))),(self.zvoid).T[0],(self.zvoid).T[1]], names=names)
        
        # read in the ouptput file
        hdul, log_filename = open_fits_file_V2(None, self.method, self.outdir, self.catname) 

        # write to the output file
        hdul['PRIMARY'].header = self.hdu.header
        if not self.periodic and not self.xyz:
            hdul.append(self.maskHDU)

        hdu = fits.BinTableHDU()
        hdu.name = 'VOIDS'
        hdul.append(hdu)
        voids = hdul['VOIDS']
        voids.header['VOID'] = (len(vT), 'Void Count')
        voids.data = fits.BinTableHDU(vT).data

        hdu = fits.BinTableHDU()
        hdu.name = 'ZONEVOID'
        hdul.append(hdu)
        zones = hdul['ZONEVOID']
        zones.header['COUNT'] = (len(vZ), 'Zone Count')
        zones.data = fits.BinTableHDU(vZ).data
        
        #save file changes
        hdul.writeto(log_filename, overwrite=True)
        
        if self.verbose > 0:
            print('V2 void output saved to', log_filename)

        if self.verbose > 0:
            print("SaveVoids time: ", time.time() - start_time)


    def saveZones(self, record_cell_volumes = False):
        """
        Description
        ===========
        
        Output calculated zones to a FITS file 
        [catalogname]_V2_[pruning method]_Output.fits

        Parameters
        ==========
        
        record_cell_volumes : bool
            If True, the tessellation cell volumes are added to the output.
            Defaults to False.
        """

        if self.verbose > 0:
            start_time = time.time()

        if not hasattr(self,'zones'):
            print("Build zones first")
            return
        
        ngal  = len(self.catalog.coord)
        glist = np.arange(ngal)
        # indices of galaxies that make pre-tessellation cuts
        glut1 = glist[self.catalog.nnls==glist]
        # for each cell center, all galaxies in its cell
        glut2 = [[] for _ in glut1]
        dlist = -1 * np.ones(ngal,dtype=int)
        

        if len(glut1) == ngal:
            # case of no cuts on galaxies
            glut2 = glut1
            dlist = self.zones.depth
        else:
            print('Warning: Due to redshift and/or magntiude cuts on the galaxy sample, the zone-saving stage may be time-intensive. Rerun with a galaxy input file that that has already applied these cuts for a faster runtime.')
            # Warning: time-instensive for large data sets
            # Idea: replace with kdtree?
            for i,l in enumerate(glut2):
                # for current cell, add all galaxy IDs of contained galaxies to to glut2
                l.extend((glist[self.catalog.nnls==glut1[i]]).tolist())
                dlist[l] = self.zones.depth[i]
         
        #each element of zcell is a zone, and the zone is a 
        #list of the galaxy indices belonging to that zone
        zcell = np.array([self.zones.zone_info[zone_ID]["galaxy_indices"] for zone_ID in self.zones.zone_info.keys()], dtype=object)
        # inverted imsk, 1 means galaxy outside survey mask, 0 means galaxy in survey mask
        olist = 1-np.array(self.catalog.imsk,dtype=int)
        
        if self.num_cpus == 1:
            
            # list of zone IDs for each galaxy, initalized to -1
            zlist = -1 * np.ones(ngal,dtype=int)
            # list of edge flags, initialized to 1 (True, edge void)
            elist = np.ones(ngal,dtype=int)
            # loop through zone IDs and galaxy IDs in zones
            for i,cl in enumerate(zcell):
                # loop through galaxy IDs in current zone
                for c in cl:
                    # write the zone ID for the current galaxy
                    zlist[glut2[c]] = i
                    # if galaxy is interior to survey (cell volume != 0) and is inside the mask
                    if self.tessellation.volumes[c]!=0. or olist[glut2[c]].all():
                        # mark as non-edge galaxy
                        elist[glut2[c]] = 0
        else:
            
            #parallel version

            # set up shared memory for parallel processes and then run processes
                           
            index_coordinator = Value(c_int64, 0, lock=True)

            zlist_file_descriptor, ARRAY_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared_zlist", 
                                                               dir="/dev/shm", 
                                                               text=False)
            
            zlist_buffer_length = ngal*4
            
            os.ftruncate(zlist_file_descriptor, zlist_buffer_length)
            
            array_buffer = mmap.mmap(zlist_file_descriptor, 0)
            
            os.unlink(ARRAY_BUFFER_PATH)
            
            zlist = np.frombuffer(array_buffer, dtype=np.int32)
            
            zlist[:] = -1
    
            zlist.shape = (ngal,)

            elist_file_descriptor, ARRAY_BUFFER_PATH = tempfile.mkstemp(prefix="vsquared_elist", 
                                                               dir="/dev/shm", 
                                                               text=False)
            
            elist_buffer_length = ngal*4
            
            os.ftruncate(elist_file_descriptor, elist_buffer_length)
            
            array_buffer = mmap.mmap(elist_file_descriptor, 0)
            
            os.unlink(ARRAY_BUFFER_PATH)
            
            elist = np.frombuffer(array_buffer, dtype=np.int32)
            
            elist[:] = 1
    
            zlist.shape = (ngal,)
            
            startup_context = multiprocessing.get_context("fork")
                
            processes = []
            
            for proc_idx in range(self.num_cpus):

                p = startup_context.Process(target=galzone_worker, 
                                            args=(ngal,
                                                index_coordinator,
                                                zlist_file_descriptor,
                                                elist_file_descriptor,
                                                zcell,
                                                glut2,
                                                self.tessellation.volumes,
                                                olist 
                                               ))
                
                p.start()
                
                processes.append(p)
                
            
            for p in processes:
            
                p.join(None) #block till join
                
        elist[np.array(olist,dtype=bool)] = 0
            
        # format output tables
        names = ['gal', 'x', 'y', 'z', 'zone', 'depth', 'edge', 'out']
        columns = [self.catalog.galids, self.catalog.coord[:,0], self.catalog.coord[:,1], self.catalog.coord[:,2], zlist,dlist,elist,olist]
        units = ['','Mpc/h','Mpc/h','Mpc/h','','','','']
        
        if hasattr(self.catalog, 'tarids'):
            names.insert(1, 'target')
            columns.insert(1, self.catalog.tarids)
            units.insert(1, '')
            
        if self.capitalize:
            names = [name.upper() for name in names]

        zT = Table(columns, names=names, units=units)
        
        # read in the ouptput file
        hdul, log_filename = open_fits_file_V2(None, self.method, self.outdir, self.catname) 
        
        # write to the output file
        hdu = fits.BinTableHDU()
        hdu.name = 'GALZONE'
        hdul.append(hdu)
        galaxies = hdul['GALZONE']
        galaxies.header['COUNT'] = (len(zT), 'Galaxy Count')
        galaxies.data = fits.BinTableHDU(zT).data

        # Save cell volume information
        if record_cell_volumes:

            columns = [glut1, self.tessellation.volumes]

            cell_table = Table(columns, names=['gal','volume'], units=['','(Mpc/h)^2'])
        
            hdu = fits.BinTableHDU()
            hdu.name = 'CELLZONE'
            hdul.append(hdu)
            cells = hdul['CELLZONE']
            cells.header['COUNT'] = (len(cell_table), 'Cell Count')
            cells.data = fits.BinTableHDU(cell_table).data
        
        #save file changes
        hdul.writeto(log_filename, overwrite=True)
        if self.verbose > 0:
            print('V2 zone output saved to', log_filename)
        
        if self.verbose > 0:
            print("SaveZones time: ", time.time() - start_time)
            
        
        


    def preViz(self):
        """
        Description
        ===========
        
        Pre-computations needed for zone and void visualizations. Outputs to
        a FITS file [catalogname]_V2_[pruning method]_Output.fits
        """
        
        if self.verbose > 0:
            start_time = time.time()
        
        if not self.visualize:
            print("Rerun with visualize=True")
            return
        
        if not hasattr(self,'vcens'):
            print("Sort voids first")
            return


        from sklearn import neighbors

        galaxy_coords = self.catalog.coord
        
        triangle_norms = self.zones.triangle_norms 
        vertices = self.zones.triangles
        triangle_zones = self.zones.triangle_zones
        triangle_zone_links = self.zones.triangle_zone_links
        
        # read in the ouptput file
        hdul, log_filename = open_fits_file_V2(None, self.method, self.outdir, self.catname)

        # generate the void ID (vid) for each triangle

        zones = hdul['ZONEVOID'].data['zone']
        containing_void = hdul['ZONEVOID'].data['void1'] 
        zones_to_voids = dict(zip(zones, containing_void))
        zones_to_voids[-1]=-1

        vid = np.vectorize(zones_to_voids.get)(triangle_zones) 
        triangle_neighbor_voids = np.vectorize(zones_to_voids.get)(triangle_zone_links) 

        # cut down triangle data to match void prunning
        # triangles are in a valid void and do not border a zone in the same void
        select_voids = (vid != -1) * (vid != triangle_neighbor_voids)
        vid = vid[select_voids]
        vertices = vertices[select_voids]
        triangle_norms = triangle_norms[select_voids]
        if len(vid)==0:
            print("Error: largest void found encompasses entire survey (try using a method other than 1 or 2)")
            return
        
        # Generate a lookup table (g2v) for converting galaxies to their containting voids

        galaxies_to_zones = hdul['GALZONE'].data['zone']
        zones_to_voids = np.concatenate((containing_void, [-1]))
        g2v = zones_to_voids[galaxies_to_zones]

        # Generate a lookup table (g2v2) for converting galaxies to the 
        # containting voids of their nearest neighbor galaxies

        galaxy_tree = neighbors.KDTree(galaxy_coords)
        indices = galaxy_tree.query(galaxy_coords, k=2, return_distance=False)
        neighbor_galaxies = indices[:,1]
        g2v2 = zones_to_voids[galaxies_to_zones[neighbor_galaxies]]


        # format data to an astropy table

        names = ['void','n_x','n_y','n_z','p1_x','p1_y','p1_z','p2_x','p2_y','p2_z','p3_x','p3_y','p3_z']
        if self.capitalize:
            names = [name.upper() for name in names]

        vizT = Table([vid, triangle_norms[:,0], triangle_norms[:,1], triangle_norms[:,2],
              vertices[:,0,0], vertices[:,0,1], vertices[:,0,2],
              vertices[:,1,0], vertices[:,1,1], vertices[:,1,2],
              vertices[:,2,0], vertices[:,2,1], vertices[:,2,2]
             ],
             names=names,
             units = ['','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h'])


        names = ['gid','g2v','g2v2']
        if self.capitalize:
            names = [name.upper() for name in names]
        g2vT = Table([np.arange(len(g2v)),g2v,g2v2],names=names)
        

        # write to the output file
        hdu = fits.BinTableHDU()
        hdu.name = 'TRIANGLE'
        hdul.append(hdu)
        triangles = hdul['TRIANGLE']
        triangles.header['COUNT'] = (len(vizT), 'Triangle Count')
        triangles.data = fits.BinTableHDU(vizT).data

        hdu = fits.BinTableHDU()
        hdu.name = 'GALVIZ'
        hdul.append(hdu)
        galaxies = hdul['GALVIZ']
        galaxies.header['COUNT'] = (len(g2vT), 'Galaxy Count')
        galaxies.data = fits.BinTableHDU(g2vT).data
        
        #save file changes
        hdul.writeto(log_filename, overwrite=True)
        
        if self.verbose > 0:
            print('V2 visualization output saved to', log_filename)
        
        if self.verbose > 0:
            print("PreViz time: ", time.time() - start_time)
        
        
        
