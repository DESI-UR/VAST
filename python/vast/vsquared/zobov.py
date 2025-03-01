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


from vast.vsquared.util import toSky, \
                               inSphere, \
                               wCen, \
                               getSMA, \
                               P, \
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
                 stages=[0,1,2,3],
                 save_intermediate=True,
                 visualize=False,
                 periodic=False, 
                 num_cpus=1,
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

        ################################################################################
        # Load the config INI file from disk
        ################################################################################
        
        config = configparser.ConfigParser()
        
        config.read(configfile)


        ################################################################################
        # Extract some values from the config INI file 
        ################################################################################
        self.infile  = config['Paths']['Input Catalog']
        
        self.catname = config['Paths']['Survey Name']
        
        self.outdir  = config['Paths']['Output Directory']
        
        self.intloc  = "../../intermediate/" + self.catname
        
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

        ################################################################################
        # Eeegads Holmes, I do believe we're running the various pipeline stages
        # now
        ################################################################################
        '''
        if start<4:
            if start<3:
                if start<2:
                    
                    
                    if start<1:
                        ctlg = Catalog(catfile=self.infile,
                                       nside=self.nside,
                                       zmin=self.zmin,
                                       zmax=self.zmax,
                                       column_names=config['Galaxy Column Names'], 
                                       maglim=self.maglim,
                                       H0=self.H0,
                                       Om_m=self.Om_m,
                                       periodic=self.periodic, 
                                       cmin=self.cmin,
                                       cmax=self.cmax, 
                                       zobov=self)
                        if save_intermediate:
                            pickle.dump(ctlg,open(self.intloc+"_ctlg.pkl",'wb'))
                            
                    else:
                        ctlg = pickle.load(open(self.intloc+"_ctlg.pkl",'rb'))
                        
                    if end>0:
                        tess = Tesselation(ctlg,
                                           viz=self.visualize,
                                           periodic=self.periodic,
                                           buff=self.buff)
                        if save_intermediate:
                            pickle.dump(tess,open(self.intloc+"_tess.pkl",'wb'))
                            
                            
                else:
                    ctlg = pickle.load(open(self.intloc+"_ctlg.pkl",'rb'))
                    tess = pickle.load(open(self.intloc+"_tess.pkl",'rb'))
                    
                if end>1:
                    zones = Zones(tess, viz=visualize)
                    if save_intermediate:
                        pickle.dump(zones,open(self.intloc+"_zones.pkl",'wb'))
                        
            else:
                ctlg  = pickle.load(open(self.intloc+"_ctlg.pkl",'rb'))
                tess  = pickle.load(open(self.intloc+"_tess.pkl",'rb'))
                zones = pickle.load(open(self.intloc+"_zones.pkl",'rb'))
                
            if end>2:
                voids = Voids(zones)
                if save_intermediate:
                    pickle.dump(voids,open(self.intloc+"_voids.pkl",'wb'))
                    
        else:
            ctlg  = pickle.load(open(self.intloc+"_ctlg.pkl",'rb'))
            tess  = pickle.load(open(self.intloc+"_tess.pkl",'rb'))
            zones = pickle.load(open(self.intloc+"_zones.pkl",'rb'))
            voids = pickle.load(open(self.intloc+"_voids.pkl",'rb'))
        '''
        
        run_stage_0 = 0 in stages
        self.create_catalog(run_stage_0, save_intermediate)
        
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


    def create_catalog(self, run_stage, save_intermediate=False):
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
                           nside=self.nside,
                           zmin=self.zmin,
                           zmax=self.zmax,
                           column_names=self.column_names, 
                           maglim=self.maglim,
                           H0=self.H0,
                           Om_m=self.Om_m,
                           periodic=self.periodic, 
                           cmin=self.cmin,
                           cmax=self.cmax, 
                           zobov=self,
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
            
            zones = Zones(self.tessellation, viz=self.visualize)
            
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
                
            voids = Voids(self.zones)
            
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
        
        
        
        
        


    def sortVoids(self, method=0, minsig=2, dc=0.2):
        """
        Sort voids according to one of several methods.

        Parameters
        ==========

        method : int or string
            0 or VIDE or vide = VIDE method (arXiv:1406.1191); link zones with density <1/5 mean density, and remove voids with density >1/5 mean density.
            1 or ZOBOV or zobov = ZOBOV method (arXiv:0712.3049); keep full void hierarchy.
            2 or ZOBOV2 or zobov2 = ZOBOV method; cut voids over a significance threshold.
            3 = not available
            4 or REVOLVER or revolver = REVOLVER method (arXiv:1904.01030); every zone below mean density is a void.
        
        minsig : float
            Minimum significance threshold for selecting voids.

        dc : float
            Density cut for linking zones using VIDE method.
        """

        #format method
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

        if method == 0: #VIDE
            
            voids  = []
            minvol = np.mean(self.tessellation.volumes[self.tessellation.volumes>0])/dc
            for i in range(len(self.prevoids.ovols)):
                vl = self.prevoids.ovols[i]
                vbuff = []

                for j in range(len(vl)-1):
                    if j > 0 and vl[j] < minvol:
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
                            vhz = np.amax(self.zones.zvols[zid])
                            vlz = np.amax(self.zones.zlinks[1][zid])
                            rz  = vhz / vlz
                            p3  = p3 * P(rz)
                        if p2 > p1*p3:
                            voids.append([c for q in self.prevoids.voids[i][:j+1] for c in q])
                            break
                        else:
                            p1 = p2
            
        
        elif method == 4: #REVOLVER
            #print('Method 4')
            voids = np.arange(len(self.zones.zvols)).reshape(len(self.zones.zvols),1).tolist()

        else:
            print("Choose a valid method")
            return

        if self.verbose > 0:
            print('Void candidates selected...')

        vcuts = [list(flatten(self.zones.zcell[v])) for v in voids]

        gcut  = np.arange(len(self.catalog.coord))[self.catalog.nnls==np.arange(len(self.catalog.nnls))]
        
        cutco = self.catalog.coord[gcut]

        # Build array of void volumes
        vvols = np.array([np.sum(self.tessellation.volumes[vcut]) for vcut in vcuts])

        # Calculate effective radius of voids
        vrads = (vvols*3/(4*np.pi))**(1/3)
        if self.verbose > 0:
            print('Effective void radius calculated')

        # Locate all voids with radii smaller than set minimum
        if method==4:
            self.minrad = np.median(vrads)
        rcut  = vrads > self.minrad
        
        voids = np.array(voids, dtype=object)[rcut]

        vcuts = [vcuts[i] for i in np.arange(len(rcut))[rcut]]
        vvols = vvols[rcut]
        vrads = vrads[rcut]
        if self.verbose > 0:
            print('Removed voids smaller than', self.minrad, 'Mpc/h')

        # Identify void centers.
        if self.verbose > 0:
            print("Finding void centers...")
        vcens = np.array([wCen(self.tessellation.volumes[vcut],cutco[vcut]) for vcut in vcuts])
        if method==0:
            dcut  = np.array([64.*len(cutco[inSphere(vcens[i],vrads[i]/4.,cutco)])/vvols[i] for i in range(len(vrads))])<1./minvol
            vrads = vrads[dcut]
            rcut  = vrads>(minvol*dc)**(1./3)
            vrads = vrads[rcut]
            vcens = vcens[dcut][rcut]
            voids = (voids[dcut])[rcut]

        # Identify eigenvectors of best-fit ellipsoid for each void.
        if self.verbose > 0:
            print("Calculating ellipsoid axes...")

        vaxes = np.array([getSMA(vrads[i],cutco[vcuts[i]]) for i in range(len(vrads))])

        zvoid = [[-1,-1] for _ in range(len(self.zones.zvols))]

        for i in range(len(voids)):
            for j in voids[i]:
                if zvoid[j][0] > -0.5:
                    if len(voids[i]) < len(voids[zvoid[j][0]]):
                        zvoid[j][0] = i
                    elif len(voids[i]) > len(voids[zvoid[j][1]]):
                        zvoid[j][1] = i
                else:
                    zvoid[j][0] = i
                    zvoid[j][1] = i

        self.vrads = vrads
        self.vcens = vcens
        self.vaxes = vaxes
        self.zvoid = np.array(zvoid)
        self.method = method
        
        if self.verbose > 0:
            print("SortVoids time: ", time.time() - start_time)


    def saveVoids(self):
        """Output calculated voids to an ASCII file [catalogname]_zonevoids.dat.
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
            names = ['x','y','z','radius','x1','y1','z1','x2','y2','z2','x3','y3','z3']
            if self.capitalize:
                names = [name.upper() for name in names]
            vT = Table([vcen[0],vcen[1],vcen[2],self.vrads,vax1[0],vax1[1],vax1[2],vax2[0],vax2[1],vax2[2],vax3[0],vax3[1],vax3[2]],
                    names = names,
                    units = ['Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h'])
        else:
            names = ['x','y','z','redshift','ra','dec','radius','x1','y1','z1','x2','y2','z2','x3','y3','z3']
            if self.capitalize:
                names = [name.upper() for name in names]
            vz,vra,vdec = toSky(self.vcens,self.H0,self.Om_m,self.zstep)
            vT = Table([vcen[0],vcen[1],vcen[2],vz,vra,vdec,self.vrads,vax1[0],vax1[1],vax1[2],vax2[0],vax2[1],vax2[2],vax3[0],vax3[1],vax3[2]],
                    names = names,
                    units = ['Mpc/h','Mpc/h','Mpc/h','','deg','deg','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h'])
        
        names = ['zone','void0','void1']
        if self.capitalize:
            names = [name.upper() for name in names]
        vZ = Table([np.array(range(len(self.zvoid))),(self.zvoid).T[0],(self.zvoid).T[1]], names=names)
        
        # read in the ouptput file
        hdul, log_filename = open_fits_file_V2(None, self.method, self.outdir, self.catname) 

        # write to the output file
        hdul['PRIMARY'].header = self.hdu.header
        if not self.periodic:
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


    def saveZones(self):
        """Output calculated zones to a fits file
        V2_{method_name}_Output.fits
        """

        if self.verbose > 0:
            start_time = time.time()

        if not hasattr(self,'zones'):
            print("Build zones first")
            return

        ngal  = len(self.catalog.coord)
        glist = np.arange(ngal)
        glut1 = glist[self.catalog.nnls==glist]
        glut2 = [[] for _ in glut1]
        dlist = -1 * np.ones(ngal,dtype=int)

        for i,l in enumerate(glut2):
            l.extend((glist[self.catalog.nnls==glut1[i]]).tolist())
            dlist[l] = self.zones.depth[i]

        zlist = -1 * np.ones(ngal,dtype=int)
        zcell = self.zones.zcell

        olist = 1-np.array(self.catalog.imsk,dtype=int)
        elist = np.zeros(ngal,dtype=int)

        for i,cl in enumerate(zcell):
            for c in cl:
                zlist[glut2[c]] = i
                if self.tessellation.volumes[c]==0. and not olist[glut2[c]].all():
                    elist[glut2[c]] = 1
        elist[np.array(olist,dtype=bool)] = 0

        # format output tables
        names = ['gal','zone','depth','edge','out']
        columns = [self.catalog.galids,zlist,dlist,elist,olist]
        
        if hasattr(self.catalog, 'tarids'):
            names.insert(1, 'target')
            columns.insert(1, self.catalog.tarids)
            
        if self.capitalize:
            names = [name.upper() for name in names]

        zT = Table(columns, names=names)
        
        # read in the ouptput file
        hdul, log_filename = open_fits_file_V2(None, self.method, self.outdir, self.catname) 

        # write to the output file
        hdu = fits.BinTableHDU()
        hdu.name = 'GALZONE'
        hdul.append(hdu)
        galaxies = hdul['GALZONE']
        galaxies.header['COUNT'] = (len(zT), 'Galaxy Count')
        galaxies.data = fits.BinTableHDU(zT).data
        
        #save file changes
        hdul.writeto(log_filename, overwrite=True)
        if self.verbose > 0:
            print('V2 zone output saved to', log_filename)
        
        if self.verbose > 0:
            print("SaveZones time: ", time.time() - start_time)
        


    def preViz(self):
        """Pre-computations needed for zone and void visualizations. Produces
        an ASCII file [catalogname]_galviz.dat.
        """
        
        if self.verbose > 0:
            start_time = time.time()
        
        if not self.visualize:
            print("Rerun with visualize=True")
            return
        
        if not hasattr(self,'vcens'):
            print("Sort voids first")
            return

        galc = self.catalog.coord[self.catalog.nnls==np.arange(len(self.catalog.coord))]
        gids = np.arange(len(self.catalog.coord))
        gids = gids[self.catalog.nnls==gids]
        g2v = -1*np.ones(len(self.catalog.coord),dtype=int)
        g2v2 = -1*np.ones(len(self.catalog.coord),dtype=int)
        verc = self.tessellation.verts
        zverts = self.zones.zverts
        znorms = self.zones.znorms
        z2v = self.zvoid.T[1]
        z2v3 = np.unique(z2v[z2v!=-1])
        z2v2 = np.array([np.where(z2v==z2)[0] for z2 in z2v3])
        zcut = [np.prod([np.prod(self.tessellation.vecut[self.zones.zcell[z]])>0 for z in z2])>0 for z2 in z2v2]

        tri1 = []
        tri2 = []
        tri3 = []
        norm = []
        vid  = []

        for k,v in enumerate(z2v2[zcut]):
            for z in v:
                for i in range(len(znorms[z])):
                    p = znorms[z][i]
                    n = galc[p[1]] - galc[p[0]]
                    n = n/np.sqrt(np.sum(n**2.))
                    polids = zverts[z][i]
                    trids = [[polids[0],polids[j],polids[j+1]] for j in range(1,len(polids)-1)]
                    for t in trids:
                        tri1.append(verc[t[0]])
                        tri2.append(verc[t[1]])
                        tri3.append(verc[t[2]])
                        norm.append(n)
                        vid.append(z2v3[zcut][k])
                    g2v[gids[p[0]]] = z2v3[zcut][k]
        for k,v in enumerate(z2v2[zcut]):
            for z in v:
                for i in range(len(znorms[z])):
                    if g2v[gids[p[1]]] != -1:
                        g2v2[gids[p[1]]] = z2v3[zcut][k]

        if len(vid)==0:
            print("Error: largest void found encompasses entire survey (try using a method other than 1 or 2)")
            return

        tri1 = np.array(tri1).T
        tri2 = np.array(tri2).T
        tri3 = np.array(tri3).T
        norm = np.array(norm).T
        vid = np.array(vid)

        # format output tables
        names = ['void_id','n_x','n_y','n_z','p1_x','p1_y','p1_z','p2_x','p2_y','p2_z','p3_x','p3_y','p3_z']
        if self.capitalize:
            names = [name.upper() for name in names]
        vizT = Table([vid,norm[0],norm[1],norm[2],tri1[0],tri1[1],tri1[2],tri2[0],tri2[1],tri2[2],tri3[0],tri3[1],tri3[2]],
                     names=names,
                     units = ['','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h','Mpc/h'])
        
        names = ['gid','g2v','g2v2']
        if self.capitalize:
            names = [name.upper() for name in names]
        g2vT = Table([np.arange(len(g2v)),g2v,g2v2],names=names)

        # read in the ouptput file
        hdul, log_filename = open_fits_file_V2(None, self.method, self.outdir, self.catname) 

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
        
        
        
