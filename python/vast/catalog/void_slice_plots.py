import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from matplotlib.lines import Line2D
from matplotlib import colors as clr
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import FixedLocator, MaxNLocator, DictFormatter

from astropy.table import Table, join
from astropy.table import vstack as avstack
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, z_at_value

from scipy.spatial import cKDTree, ConvexHull

from sympy import solve_poly_system, im
from sympy.abc import x,y

from vast.voidfinder.distance import z_to_comoving_dist
from vast.voidfinder.voidfinder_functions import xyz_to_radecz

#matplotlib.rcParams.update({'font.size': 38})

"""
Authors: Hernan Rincon

The main body of this code has been adopted from the following individuals: Dahlia Veyrat
"""

D2R = np.pi/180.

def toSky(cs):
    '''
    Convert Cartesian coordinates to sky coordinates
    
    params:
    ----------------------------------------------------------
    cs (numpy array of shape (3,N)): The cartesian coordinates 
        of the catalog in Mpc/h
        
    returns:
    ------------------------------------------------
    r, ra, dec (1-D numpy arrays): the sky coordiantes of the
        catalog, with r in Mpc/h and ra and dec in degrees
    '''
    c1  = cs.T[0]
    c2  = cs.T[1]
    c3  = cs.T[2]
    
    r   = np.sqrt(c1**2. + c2**2. + c3**2.)
    dec = np.arcsin(c3/r)/D2R
    ra  = (np.arccos(c1/np.sqrt(c1**2. + c2**2.))*np.sign(c2)/D2R)%360
    
    return r,ra,dec



def toCoord(r,ra,dec):
    '''
    Convert sky coordinates to Cartesian coordinates
    
    params:
    ----------------------------------------------------------
    r, ra, dec (1-D numpy arrays): the sky coordiantes of the
        catalog, with r in Mpc/h and ra and dec in degrees
        
    returns:
    ----------------------------------------------------------
    c1, c2, c3 (1-D numpy arrays): The cartesian coordinates 
        of the catalog in Mpc/h
    
    '''
    c1 = r*np.cos(ra*D2R)*np.cos(dec*D2R)
    c2 = r*np.sin(ra*D2R)*np.cos(dec*D2R)
    c3 = r*np.sin(dec*D2R)
    
    return c1,c2,c3




class VoidMapVF():
    """
    Class used for plotting VoidFinder void catalogs
    """
    

    def __init__(self, gdata, maximals, holes):
        """
        Initialize the VoidMapVF class with void catalog data
        
        params:
        ----------------------------------------------------------
        gdata (numpy array): the galaxy data. Should contain 
            either sky coordiantes ("r", "ra", "dec") or cartesian
            cooridnates ("x", "y", "z"). May optionally contain 
            galaxy magnitudes ("rabsmag").
            
        maximals (numpy array): the voidfinder maximals sphere 
            table. Should be loaded with lowercase column names.
            
        holes (numpy array): the voidfinder holes table. Should be 
            loaded with lowercase column names.
        """
        self.load_data(gdata,maximals,holes)

    def load_data(self,gdata,maximals,holes):
        """
        Load void catalog data for plotting
        
        params:
        ----------------------------------------------------------
        gdata (numpy array): the galaxy data. Should contain 
            either sky coordiantes ("r", "ra", "dec") or cartesian
            cooridnates ("x", "y", "z"). May optionally contain 
            galaxy magnitudes ("rabsmag").
            
        maximals (numpy array): the voidfinder maximals sphere 
            table. Should be loaded with lowercase column names.
            
        holes (numpy array): the voidfinder holes table. Should be 
            loaded with lowercase column names.
        """   
        self.gdata=gdata
        self.maximals=maximals
        self.holes=holes
        #making the data from galaxies more accessible/easier to manipulate?
        if np.isin('rabsmag', gdata.colnames):
            self.rmag = gdata['rabsmag']
        radecr = 0
        if np.isin('r', gdata.colnames):
            self.gr = gdata['r']
            radecr+=1
        elif np.isin('redshift', gdata.colnames):
            self.gr=  z_to_comoving_dist(np.array(gdata['redshift'],dtype=np.float32),0.315,1)
            radecr+=1
        if np.isin('ra', gdata.colnames):
            self.gra  = gdata['ra']
            radecr+=1
        if np.isin('dec', gdata.colnames):
            self.gdec = gdata['dec']
            radecr+=1
        #else:
        #    create ra, dec, r columns from x, y, z, data
        if radecr<3:
            self.gx,self.gy,self.gz = gdata['x'], gdata['y'], gdata['z']
        #Converting galaxy data to cartesian
        #cKDTree finds nearest neighbors to data point
        else:
            self.gx,self.gy,self.gz = toCoord(self.gr,self.gra,self.gdec)
        self.kdt = cKDTree(np.array([self.gx,self.gy,self.gz]).T)

        #Simplifying VoidFinder maximal sphere coordinates and converting them into RA,DEC,DIS
        self.vfx = maximals['x'] 
        self.vfy = maximals['y']
        self.vfz = maximals['z']
        self.vfr,self.vfra,self.vfdec = toSky(np.array([self.vfx,self.vfy,self.vfz]).T)
        self.vfrad = maximals['radius']
        self.vfedge = maximals['edge']

        #Same coordinate conversion, but for holes
        self.vflag = holes['void']
        self.vfx2 = holes['x']
        self.vfy2 = holes['y']
        self.vfz2 = holes['z']
        self.vfr1,self.vfra1,self.vfdec1 = toSky(np.array([self.vfx2,self.vfy2,self.vfz2]).T)
        self.vfrad1 = holes['radius']

        #group hole coordinates based on their void IDs
        self.vfx4   = [self.vfx2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfy4   = [self.vfy2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfz4   = [self.vfz2[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfr2   = [self.vfr1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfra2  = [self.vfra1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfdec2 = [self.vfdec1[self.vflag==vfl] for vfl in np.unique(self.vflag)]
        self.vfrad2 = [self.vfrad1[self.vflag==vfl] for vfl in np.unique(self.vflag)]

        #Determine which galaxies are inside voids
        self.gflag_vf = np.zeros(len(self.gx),dtype=bool)

        for vfl in np.unique(self.vflag):
            self.vfx3 = self.vfx2[self.vflag==vfl]
            self.vfy3 = self.vfy2[self.vflag==vfl]
            self.vfz3 = self.vfz2[self.vflag==vfl]
            self.vfrad3 = self.vfrad1[self.vflag==vfl]
            for i in range(len(self.vfx3)):
                galinds = self.kdt.query_ball_point([self.vfx3[i],self.vfy3[i],self.vfz3[i]],self.vfrad3[i])
                self.gflag_vf[galinds] = True

        #Determine which galaxies are outside voids
        self.wflag_vf = (1-self.gflag_vf).astype(bool)



    def cint2(self, dec):
        '''
        Calculate radii of hole-slice intersections for survey 
        geometry
        
        params:
        ----------------------------------------------------------
        dec (float): the declianation of the cross-secitonal plane
        
        returns:
        ----------------------------------------------------------
        cr (list of lists): The radii of the hole-slice 
            intersections for each void
        '''
        cr = []
        #loop through voids
        for i in range(len(self.vfr2)):
            cr.append([])
            #loop through hoels in current void
            for j in range(len(self.vfr2[i])):
                
                #distance between hole center and plane
                dtd = np.abs(self.vfr2[i][j]*np.sin((self.vfdec2[i][j]-dec)*D2R))
                #case that hole does intersect plane
                if dtd>self.vfrad2[i][j]:
                    cr[i].append(0.)
                #case that hole does intersect plane 
                else:
                    #add intersetion radius
                    cr[i].append(np.sqrt(self.vfrad2[i][j]**2.-dtd**2.))
        return cr
    
    def cint2xyz(self, plane_height, vfn4):
        '''
        Calculate radii of hole-slice intersections for cubic
        box geometry
        
        params:
        ----------------------------------------------------------
        plane_height (float): the height cross-secitonal plane
        
        vfn4 (list): one of self.vfx4, self.vfy4, or self.vfz4, 
            corresponding to the dimension of the cross-section
            
        returns:
        ----------------------------------------------------------
        cr (list of lists): The radii of the hole-slice 
            intersections for each void
        '''

        #list of <list of radii of hole-plane intersections> for each void
        cr = []
        for i in range(len(self.vfr2)):
            #create new cr entry for current void
            cr.append([])
            #iterate through holes in void
            for j in range(len(self.vfr2[i])):
                #minimum distance from hole center to the plane
                dtd = np.abs(vfn4[i][j] - plane_height)
                #case that hole doesn't intersect plane
                if dtd>self.vfrad2[i][j]:
                    cr[i].append(0.)
                #case that hole does intersect plane
                else:
                    #add inters
                    cr[i].append(np.sqrt(self.vfrad2[i][j]**2.-dtd**2.))
        return cr


    def gcp2(self, cc1,cc2,crad,npt,chkdpth,ra_dec_z=True):
        '''
        Convert circles' coordinates to ordered boundary
        
        params:
        ----------------------------------------------------------
        cc1 (numpy array of floats): The comoving radial 
            coordinates of the holes consituting a void
            
        cc2 (numpy array of floats): The right ascensions
            coordinates of the holes consituting a void
        
        crad (list): The radii of the hole-slice 
            intersections for a void
            
        npt (int): 
        
        chkdpth (int):
        
        ra_dec_z (bool)
            
        returns:
        ----------------------------------------------------------
        cr (numpy array): The radii of the hole-slice 
            intersections
        '''
        if ra_dec_z:
            ccx = cc1*np.cos(cc2*D2R)
            ccy = cc1*np.sin(cc2*D2R)
        else:
            ccx,ccy = cc1,cc2

        Cx = [np.linspace(0.,2*np.pi,int(npt*crad[k]/10)) for k in range(len(ccx))]
        Cy = [np.linspace(0.,2*np.pi,int(npt*crad[k]/10)) for k in range(len(ccx))]
        
        Cx = [np.cos(Cx[k])*crad[k]+ccx[k] for k in range(len(ccx))]
        Cy = [np.sin(Cy[k])*crad[k]+ccy[k] for k in range(len(ccx))]
        
        #cut out segments of holes that are inside of other holes
        for i in range(len(ccx)):
            for j in range(len(ccx)):
                if i==j:
                    continue
                cut = (Cx[j]-ccx[i])**2.+(Cy[j]-ccy[i])**2.>crad[i]**2.
                Cx[j] = Cx[j][cut]
                Cy[j] = Cy[j][cut]

        Cp = []

        for i in range(len(ccx)):
            Cp.extend(np.array([Cx[i],Cy[i]]).T.tolist())

        Cp = np.array(Cp)

        kdt = cKDTree(Cp)

        Cpi = [0]

        while len(Cpi)<len(Cp):
            if len(Cpi)==1:
                nid = kdt.query(Cp[Cpi[-1]],2)[1][1]
            else:
                nids = kdt.query(Cp[Cpi[-1]],chkdpth+1)[1][1:]
                for k in range(chkdpth):
                    if nids[k] not in Cpi[(-1*(chkdpth+1)):-1]:
                        nid = nids[k]
                        break
                nids = kdt.query(Cp[Cpi[-1]],7)[1][1:]
            Cpi.append(nid)

        #Cpi.append(0)
        if ra_dec_z:
            C1 = np.sqrt(Cp[Cpi].T[0]**2.+Cp[Cpi].T[1]**2.)
            C2 = (np.sign(Cp[Cpi].T[1])*np.arccos(Cp[Cpi].T[0]/C1)+np.pi*(1.-np.sign(Cp[Cpi].T[1])))/D2R

            return C1,C2
        else:
            return Cp[Cpi].T[0], Cp[Cpi].T[1]


    def plot_survey(self,dec,wdth,npc,chkdpth, 
             ra0, ra1, cz0, cz1, title, graph = None, zlimits = True, rot = 0, 
             colors = ["blue","blue","blue"], gal_colors = ["black","red"], include_gals=True, include_voids=True, alpha=0.2, border_alpha = 1,
             horiz_legend_offset=.8, plot_sdss = True, sdss_lim=332.38626, sdss_color='magenta', 
             mag_limit = None, galaxy_point_size = 1, return_plot_data=False):
        '''
        Plot VoidFinder voids within a survey volume.
        
        params:
        ----------------------------------------------------------
        dec (float): declination of slice
        
        wdth (float): Distance from declination plane in Mpc/h 
            within which galaxies are plotted
            
        npc ():
        
        chkdpth ():
        
        ra0 (float): the minimum ra corresponding the leftmost 
            plot boundary
        
        ra1 (float): the maximum ra corresponding the rightmost 
            plot boundary
            
        cz0 (float): the minimum distance corresponding the inner 
            plot boundary. Given as a redshift if zlimit = True, 
            otherwise given in Mpc/h
        
        cz1 (float): the maximum distance corresponding the outer 
            plot boundary. Given as a redshift if zlimit = True, 
            otherwise given in Mpc/h
            
        title (string): the plot title
        
        graph (list): list containing a pyplot figure (graph[0]), 
            a floating_axes.FloatingSubplot object (graph[1]), 
            and floating subplot axes given by
            floating_axes.FloatingSubplot.get_aux_axes (graph[2]).
            Used for plotting voids over an already existing plot.
            Defaults to none.
        
        zlimits (bool): if True, the distance limits (cz0, cz1) are 
            given as redshifts. Otherwise, they are given in Mpc/h.
            Defaults to True.
            
        rot (float): The rotation of the plot. Changes teh relative 
            positions of the ra values.
            
        colors (list): a three element list of strings specifying 
            matplotlib colors for plotting edge voids (colors[0]), 
            near-edge voids (colors[1]), and interior voids 
            (colors[2]). Defaults to ["blue","blue","blue"]
            
        gal_colors (list): a two element list of strings specifying 
            matplotlib colors for plotting wall galaxies 
            (gal_colors[0]) and void galaxies (gal_colors[1])
            
        include_gals (bool): Demtermines if galaxies are plotted
        
        include_voids (bool): Demtermines if voids are plotted
        
        alpha (float): The alpha value for the filled interiors of 
            the plotted voids
            
        border_alpha: the alpha value for the borders of the
            plotted voids
            
        horiz_legend_offset (float): the horizontal positioning of 
            the graph legend.
            
        plot_sdss (bool): Determines if the SDSS DR7 void catalog
            distance limit is plotted
            
        sdss_lim (float): SDSS DR7 void catalog distance limit 
        
        sdss_color (string): The color used to plot the SDSS DR7 
            void catalog distance limit 
        
        mag_limit (float): The magnitude limit used for selecting 
            plotted galaxies. Defaults to None, in which case all
            galaxies are plotted
            
        galaxy_point_size (float): The size of the markers for the
            plotted galaxies
        
        '''
        if zlimits:
            cz0 = z_to_comoving_dist(np.array([cz0],dtype=np.float32),.315,1)[0]
            cz1 = z_to_comoving_dist(np.array([cz1],dtype=np.float32),.315,1)[0]

        set_bottom_invisible = False if cz0 > 0 else True #Don't illustrate the bottom axis if the redshift extends to 0

        if graph is None:
            fig = plt.figure(1, figsize=(1600/96,800/96))

            ax3, aux_ax3 = setup_axes3(fig, 111, ra0, ra1, cz0, cz1, rot, set_bottom_invisible)

            #aux_ax3.set_aspect(1) #This is causing errors

            plt.title(f"{title} $\delta$ = {dec}$^\circ$", loc='left',)
        else:
            fig, ax3, aux_ax3 = graph[0], graph[1], graph[2]
            
        if return_plot_data:
            plot_data = []
        
        if include_voids:
            Cr = self.cint2(dec)

            for i in range(len(self.vfr)):

                if np.sum(Cr[i])>0:

                    Cr2, Cra2 = self.gcp2(self.vfr2[i], self.vfra2[i], Cr[i], npc, chkdpth)

                    if self.vfedge[i] == 1:
                        vcolor = colors[0]#'gold'
                    elif self.vfedge[i] == 2:
                        vcolor = colors[1]#'darkgoldenrod'
                    else:
                        vcolor = colors[2]

                    #vcolor = 'blue'

                    aux_ax3.plot(Cra2, Cr2, color=vcolor, alpha = border_alpha)
                    aux_ax3.fill(Cra2, Cr2, color=vcolor, alpha=alpha)
                    
                    if return_plot_data:
                        plot_data.append([Cra2, Cr2])
                

            #Cr2,Cra2 = gcp3(vfx4[i],vfy4[i],vfz4[i],vfr2[i],vfrad2[i],vfdec2[i],dec,npc,chkdpth)
            #if len(Cr2)>0:
            #    aux_ax3.plot(Cra2,Cr2,color='blue')
            #    aux_ax3.fill(Cra2,Cr2,alpha=0.5,color='blue')
        if include_gals:
            # Edge galaxies
            #gdcut = (gr[eflag_vf]*np.sin((gdec[eflag_vf] - dec)*D2R))**2 < wdth**2
            #aux_ax3.scatter(gra[eflag_vf][gdcut], gr[eflag_vf][gdcut], color='g', s=1, alpha=0.5)

            # Wall galaxies
            gdcut = (self.gr[self.wflag_vf]*np.sin((self.gdec[self.wflag_vf] - dec)*D2R))**2. < wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[self.wflag_vf] < mag_limit
            aux_ax3.scatter(self.gra[self.wflag_vf][gdcut], self.gr[self.wflag_vf][gdcut], color=gal_colors[0], s=galaxy_point_size, edgecolor=None)
            
            if return_plot_data:
                plot_data.append([self.gra[self.wflag_vf][gdcut], self.gr[self.wflag_vf][gdcut]])
                
            # Void galaxies
            gdcut = (self.gr[self.gflag_vf]*np.sin((self.gdec[self.gflag_vf] - dec)*D2R))**2. < wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[self.gflag_vf] < mag_limit
            aux_ax3.scatter(self.gra[self.gflag_vf][gdcut], self.gr[self.gflag_vf][gdcut], color=gal_colors[1], s=galaxy_point_size, edgecolor=None)
            
            if return_plot_data:
                plot_data.append([self.gra[self.gflag_vf][gdcut], self.gr[self.gflag_vf][gdcut]])
        
        if plot_sdss:
            cc=plt.Circle((0, 0), sdss_lim, color=sdss_color,fill=False,linewidth=3)
            ax3.add_artist( cc )
            
        #create legend
        void_legend_handles=[]
        if include_voids:
            legend_color=clr.to_rgba(colors[2])
            legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
            void_legend_handles.append(
                Line2D([0], [0], label='voids', marker='o', markersize=15, 
                    markeredgecolor=colors[2], markerfacecolor=legend_color, linestyle=''
                )
            )
            if colors[1]!=colors[2]:
                legend_color=clr.to_rgba(colors[1])
                legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
                void_legend_handles.append(
                    Line2D([0], [0], label='near-edge voids', marker='o', markersize=15, 
                        markeredgecolor=colors[1], markerfacecolor=legend_color, linestyle='')
                )
            if colors[0]!=colors[2]:
                legend_color=clr.to_rgba(colors[0])
                legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
                void_legend_handles.append(
                    Line2D([0], [0], label='edge voids', marker='o', markersize=15, 
                        markeredgecolor=colors[0], markerfacecolor=legend_color, linestyle='')
                )
        if include_gals:
            
            void_legend_handles.append(
                Line2D([0], [0], label="wall galaxies", marker='o', markersize=9, 
                    markeredgecolor=gal_colors[0], markerfacecolor=gal_colors[0], linestyle='')
            )
            void_legend_handles.append(
                Line2D([0], [0], label="void galaxies", marker='o', markersize=9, 
                    markeredgecolor=gal_colors[1], markerfacecolor=gal_colors[1], linestyle='')
            )
        if plot_sdss:
            void_legend_handles.append(
                Line2D([0], [0], label='SDSS limit', color=sdss_color, linewidth=5)
            )
            
        aux_ax3.legend(handles=void_legend_handles, loc="upper left", bbox_to_anchor=(horiz_legend_offset,1))
        
        self.graph = [fig, ax3, aux_ax3]
        
        if return_plot_data:
            return self.graph, plot_data

        return self.graph
    
    # Plot VoidFinder Voids from a Cubic Simulation (Version 2)
    def plot_xyz(self,plane_height,wdth,npc,chkdpth, 
                title, h="x",v="y",n="z", h_range = (0,50), v_range = (0,50), graph = None, 
                colors = ["blue","blue","blue"],gal_colors = ["black","red"],include_gals=True,alpha=0.2, border_alpha = 1,scale=1):
            '''
            Plot VoidFinder voids wihtin a cubic simulation volume.
            '''
            axes = {"x":self.vfx4, "y":self.vfy4, "z":self.vfz4}
            gal_axes = {"x":self.gx, "y":self.gy, "z":self.gz}
            vfh4 = axes[h]
            vfv4 = axes[v]
            vfn4 = axes[n]
            gh = gal_axes[h]
            gv = gal_axes[v]
            gn = gal_axes[n]
                
            if graph is None:

                v_scale = (v_range[1]-v_range[0])/(h_range[1]-h_range[0])

                fig, ax = plt.subplots(num=1, figsize=(scale*800/96, scale*v_scale*800/96))

                ax.set_xlabel(h)
                ax.set_ylabel(v)

                ax.set_xlim(h_range)
                ax.set_ylim(v_range)
                
                ax.set_aspect('equal', adjustable='box')

                plt.title(f"{title} ${n}$ = {plane_height} [Mpc h$^{-1}$]")
            else:
                fig, ax = graph[0], graph[1]

            Cr = self.cint2xyz(plane_height, vfn4)
            
            for i in range(len(self.vfr)):
                if np.sum(Cr[i])>0:

                    Cr2, Cra2 = self.gcp2(vfh4[i], vfv4[i], Cr[i], npc, chkdpth, ra_dec_z=False)

                    if self.vfedge[i] == 1:
                        vcolor = colors[0]#'gold'
                    elif self.vfedge[i] == 2:
                        vcolor = colors[1]#'darkgoldenrod'
                    else:
                        vcolor = colors[2]

                    #vcolor = 'blue'

                    ax.plot(Cr2, Cra2, color=vcolor, alpha = border_alpha)
                    ax.fill(Cr2, Cra2, color=vcolor, alpha=alpha)
                    #plt.scatter(vfh4[i], vfv4[i], color='orange',s=1) #debugging hole locations

                #Cr2,Cra2 = gcp3(vfx4[i],vfy4[i],vfz4[i],vfr2[i],vfrad2[i],vfdec2[i],dec,npc,chkdpth)
                #if len(Cr2)>0:
                #    aux_ax3.plot(Cra2,Cr2,color='blue')
                #    aux_ax3.fill(Cra2,Cr2,alpha=0.5,color='blue')
            if include_gals:
                # Edge galaxies
                #gdcut = (gr[eflag_vf]*np.sin((gdec[eflag_vf] - dec)*D2R))**2 < wdth**2
                #aux_ax3.scatter(gra[eflag_vf][gdcut], gr[eflag_vf][gdcut], color='g', s=1, alpha=0.5)

                # Wall galaxies
                gdcut = (gn[self.wflag_vf] - plane_height)**2. < wdth**2.
                ax.scatter(gh[self.wflag_vf][gdcut], gv[self.wflag_vf][gdcut], color=gal_colors[0], s=1)

                # Void galaxies
                gdcut = (gn[self.gflag_vf] - plane_height)**2. < wdth**2.
                ax.scatter(gh[self.gflag_vf][gdcut], gv[self.gflag_vf][gdcut], color=gal_colors[1], s=1)

            
            self.graph = [fig, ax]

            return self.graph


class VoidMapV2():
    """
    Class for plotting V2 voids
    """

    def __init__(self,tridata, gzdata, zvdata, zbdata, gdata, edge_threshold = 0.1):
        """
        Initialize the VoidMapV2 class with void catalog data
        
        params:
        ----------------------------------------------------------
        tridata (numpy array): the V2 triangles table. Should be 
            loaded with lowercase column names.
            
        gzdata (numpy array): the V2 galzones table. Should be 
            loaded with lowercase column names.
            
        zvdata (numpy array): the V2 zonevoids table. Should be 
            loaded with lowercase column names.
            
        zbdata (numpy array): the V2 zobovoids table. Should be 
            loaded with lowercase column names.
            
        gdata (numpy array): the galaxy data. Should contain 
            either sky coordiantes ("r", "ra", "dec") or 
            cartesian cooridnates ("x", "y", "z"). May optionally 
            contain galaxy magnitudes ("rabsmag")
            
        edge_threshold (float): the threshold ratio of 
            edge_area/surface_area above which V2 voids are 
            considered edge voids. Defaults to 0.1
        
        """   
        self.load_data(tridata, gzdata, zvdata, zbdata, gdata, edge_threshold)

    def load_data(self, tridata, gzdata, zvdata, zbdata, gdata, edge_threshold):
        '''
        Load void catalog data for plotting
        
        params:
        ----------------------------------------------------------
        tridata (numpy array): the V2 triangles table. Should be 
            loaded with lowercase column names.
            
        gzdata (numpy array): the V2 galzones table. Should be 
            loaded with lowercase column names.
            
        zvdata (numpy array): the V2 zonevoids table. Should be 
            loaded with lowercase column names.
            
        zbdata (numpy array): the V2 zobovoids table. Should be 
            loaded with lowercase column names.
            
        gdata (numpy array): the galaxy data. Should contain 
            either sky coordiantes ("r", "ra", "dec") or 
            cartesian cooridnates ("x", "y", "z"). May optionally 
            contain galaxy magnitudes ("rabsmag")
            
        edge_threshold (float): the threshold ratio of 
            edge_area/surface_area above which V2 voids are 
            considered edge voids. Defaults to 0.1
        '''
        

        #gr   = gdata[1].data['Rgal']
        if np.isin('rabsmag', gdata.colnames):
            self.rmag = gdata['rabsmag']
            
        self.gr   = z_to_comoving_dist(gdata['redshift'].astype(np.float32), 0.315, 1)
        self.gra  = gdata['ra']
        self.gdec = gdata['dec']

        self.gx,self.gy,self.gz = toCoord(self.gr,self.gra,self.gdec)
        
        # Determine galaxy void Membership
        # ----------------------------
        #match the galaxy table with the galzones table
        kdt = cKDTree(np.array([gzdata['x'],gzdata['y'],gzdata['z']]).T)#KDTree of zone centers
        _, indexes = kdt.query(np.array([self.gx,self.gy,self.gz]).T) #get the indexes of closest zone for each galaxy
        galaxy_zones = gzdata[indexes]['zone',] #sort the zones to match the galaxies
        #match the galaxy zones with void IDs
        zone_voids = zvdata['zone','void0']
        zone_voids.add_row([-1, -1])
        galaxy_zones['__sort__'] = np.arange(len(galaxy_zones))
        void_IDs = join(galaxy_zones, zone_voids, keys='zone',  join_type='left')
        void_IDs.sort('__sort__')
        #mark void galaxies and wall galaxies
        self.gflag_v2 = (void_IDs['void0'] != -1)  * np.isin(void_IDs['void0'], zbdata['void'])
        self.wflag_v2 = ~ self.gflag_v2
        
        # Cut out triangles for voids that aren't included in the catalog
        tridata = tridata[np.isin(tridata['void'], zbdata['void'])]
        
        self.p1_r,self.p1_ra,self.p1_dec = toSky(np.array([tridata['p1_x'],tridata['p1_y'],tridata['p1_z']]).T)
        self.p2_r,self.p2_ra,self.p2_dec = toSky(np.array([tridata['p2_x'],tridata['p2_y'],tridata['p2_z']]).T)
        self.p3_r,self.p3_ra,self.p3_dec = toSky(np.array([tridata['p3_x'],tridata['p3_y'],tridata['p3_z']]).T)

        self.p1_x = tridata['p1_x']
        self.p1_y = tridata['p1_y']
        self.p1_z = tridata['p1_z']

        self.p2_x = tridata['p2_x']
        self.p2_y = tridata['p2_y']
        self.p2_z = tridata['p2_z']

        self.p3_x = tridata['p3_x']
        self.p3_y = tridata['p3_y']
        self.p3_z = tridata['p3_z']

        trivids = np.array(tridata['void'])
        
        self.edge = {}
        for row in zbdata:
            #self.edge[row['void']] = row["edge_area"]/row['tot_area'] > edge_threshold
            self.edge[row['void']] = False

        self.tridat= tridata
        self.trivids = trivids


    def getinx(self, xx,aa,yy,bb,zz,cc,dd):
        # get the cross section of a line within a plane
        # the first point is <xx, yy, zz> and the second point minus the first point is <aa, bb, cc>
        # the declination of the plane is dd = 1 / tan (dec)
        negb = -1.*aa*xx-bb*yy+cc*dd*dd*zz
        sqto = 0.5*np.sqrt((2.*aa*xx+2.*bb*yy-2.*cc*dd*dd*zz)**2.-4.*(aa**2.+bb**2.-cc*cc*dd*dd)*(xx**2.+yy**2.-zz*zz*dd*dd))
        twa = aa**2.+bb**2.-cc*cc*dd*dd
        tt = (negb+sqto)/twa
        if tt>0 and tt<1:
            tt = tt
        else:
            tt = (negb-sqto)/twa
        return xx+aa*tt,yy+bb*tt,zz+cc*tt
    
    def isin2(self, p,ps):
        nc = 1
        for i in range(len(ps)-1):
            p1 = ps[i]
            p2 = ps[i+1]
            if p1[0]<p[0] and p2[0]<p[0]:
                continue
            elif (p1[1]-p[1])*(p2[1]-p[1])>0:
                continue
            elif p1[0]>p[0] and p2[0]>p[0]:
                nc = nc+1
            elif ((p2[1]-p1[1])/(p2[0]-p1[0]))*((p1[1]-p[1])-((p2[1]-p1[1])/(p2[0]-p1[0]))*(p1[0]-p[0]))<1:
                nc = nc+1
        return nc%2==0

    def trint2(self, dec):
        
        # get triangles coordinates
        p1_dec,  p2_dec,  p3_dec = self.p1_dec, self.p2_dec, self.p3_dec
        p1_x,p1_y,p1_z = self.p1_x,self.p1_y,self.p1_z
        p2_x,p2_y,p2_z = self.p2_x,self.p2_y,self.p2_z
        p3_x,p3_y,p3_z = self.p3_x,self.p3_y,self.p3_z
        trivids = self.trivids
        getinx = self.getinx
        # table of flags for if/how triangles cross dec slice
        decsum = np.array([(p1_dec>dec).astype(int),(p2_dec>dec).astype(int),(p3_dec>dec).astype(int)]).T
        intr  = [[] for _ in range(np.amax(trivids)+1)]
        intra = [[] for _ in range(np.amax(trivids)+1)]
        #loop through every traingle and add its cross section (two points in ra-r space) to the intr, intra tables
        for i in range(len(trivids)):
            if np.sum(decsum[i])==0:
                continue
            if np.sum(decsum[i])==3:
                continue
            cv = trivids[i]
            if np.sum(decsum[i])==1:
                if decsum[i][0]==1:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                elif decsum[i][1]==1:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
                elif decsum[i][2]==1:
                    sss = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
            elif np.sum(decsum[i])==2:
                if decsum[i][0]==0:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                elif decsum[i][1]==0:
                    sss = getinx(p1_x[i],p2_x[i]-p1_x[i],p1_y[i],p2_y[i]-p1_y[i],p1_z[i],p2_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
                elif decsum[i][2]==0:
                    sss = getinx(p1_x[i],p3_x[i]-p1_x[i],p1_y[i],p3_y[i]-p1_y[i],p1_z[i],p3_z[i]-p1_z[i],1./np.tan(dec*D2R))
                    sst = getinx(p3_x[i],p2_x[i]-p3_x[i],p3_y[i],p2_y[i]-p3_y[i],p3_z[i],p2_z[i]-p3_z[i],1./np.tan(dec*D2R))
            intr[cv].append(np.sqrt(np.sum(np.array(sss)**2.)))
            intr[cv].append(np.sqrt(np.sum(np.array(sst)**2.)))
            intra[cv].append((np.arccos(sss[0]/np.sqrt(sss[0]**2.+sss[1]**2.))*np.sign(sss[1])/D2R)%360)
            intra[cv].append((np.arccos(sst[0]/np.sqrt(sst[0]**2.+sst[1]**2.))*np.sign(sst[1])/D2R)%360)
        return intr,intra

    def getorder(self, xs,ys):
        chains = []
        scut = np.zeros(len(xs),dtype=bool)
        for i in range(len(xs)):
            if len(xs[xs==xs[i]])==1:
                scut[i] = True
            elif len(xs[xs==xs[i]])>2:
                print("0",end='',flush=True)
        dists = []
        pairs = []
        for i in range(len(xs)):
            if scut[i]:
                for j in range(i+1,len(xs)):
                    if scut[j]:
                        dists.append((xs[i]-xs[j])**2.+(ys[i]-ys[j])**2.)
                        pairs.append([i,j])
        pairs = np.array(pairs)[np.argsort(dists)]
        paird = scut
        xs2 = xs.tolist()
        ys2 = ys.tolist()
        cmp = np.arange(len(xs)).tolist()
        for i in range(len(pairs)):
            if paird[pairs[i][0]] and paird[pairs[i][1]]:
                paird[pairs[i][0]] = False
                paird[pairs[i][1]] = False
                xs2.extend([xs[pairs[i][0]],xs[pairs[i][1]]])
                ys2.extend([ys[pairs[i][0]],ys[pairs[i][1]]])
                cmp.extend([pairs[i][0],pairs[i][1]])
        xs2 = np.array(xs2)
        ys2 = np.array(ys2)
        lcut = np.ones(len(xs2),dtype=bool)
        for i in range(len(xs2)):
            if lcut[i]:
                chains.append([])
                chains[-1].append(cmp[i])
                lcut[i] = False
                j = i + 1 - 2*(i%2)
                while xs2[j] != xs2[i]:
                    lcut[j] = False
                    k = np.where(xs2==xs2[j])[0]
                    k = k[k != j][0]
                    chains[-1].append(cmp[k])
                    lcut[k] = False
                    j = k + 1 - 2*(k%2)
                if chains[-1][0] != chains[-1][-1]:
                    chains[-1].append(chains[-1][0])
        return chains

    def convint3(self, intr,intra):
        intx = np.array(intr)*np.cos(np.array(intra)*D2R)
        inty = np.array(intr)*np.sin(np.array(intra)*D2R)
        chkl = []
        ccut = np.ones(len(intr),dtype=bool)
        for i in range(int(len(intr)/2)):
            chkl.append(intx[2*i]+intx[2*i+1])
        chkl = np.array(chkl)
        for i in range(len(chkl)):
            if len(chkl[chkl==chkl[i]])>1:
                ccut[2*i] = False
                ccut[2*i+1] = False
        intx = intx[ccut]
        inty = inty[ccut]
        ocut = self.getorder(intx,inty)
        icut = np.zeros(len(ocut),dtype=bool)
        lens = np.zeros(len(ocut))
        for i in range(len(ocut)):
            for j in range(len(ocut[i])-1):
                lens[i] = lens[i] + np.sqrt((intx[ocut[i][j+1]]-intx[ocut[i][j]])**2.+(inty[ocut[i][j+1]]-inty[ocut[i][j]])**2.)
        mlh = np.amax(lens)
        for i in range(len(ocut)):
            if lens[i]==mlh:
                continue
            o = ocut[i]
            P = np.array([intx[o][0],inty[o][0]])
            for j in range(len(ocut)):
                if j==i:
                    continue
                o1 = ocut[j]
                Ps = np.array([intx[o1],inty[o1]]).T
                if self.isin2(P,Ps):
                    icut[i] = True
                    break
        return [[np.array(intr)[ccut][o].tolist(),np.array(intra)[ccut][o].tolist()] for o in ocut],icut 

    def plot_survey(self,dec,wdth,
             ra0, ra1, cz0, cz1, title, graph = None, zlimits = True, rot = 0, 
             colors = ["blue","blue"],include_gals=True,alpha=0.2, border_alpha = 1,
             horiz_legend_offset=.8, plot_sdss = True, sdss_lim=332.38626, sdss_color='magenta', 
             mag_limit = None, galaxy_point_size=1, return_plot_data=False):
        '''
        Plot Vsquared voids within a survey volume
        
        params:
        ----------------------------------------------------------
        dec (float): declination of slice
        
        wdth (float): Distance from declination plane in Mpc/h 
            within which galaxies are plotted
            
        npc ():
        
        chkdpth ():
        
        ra0 (float): the minimum ra corresponding the leftmost 
            plot boundary
        
        ra1 (float): the maximum ra corresponding the rightmost 
            plot boundary
            
        cz0 (float): the minimum distance corresponding the inner 
            plot boundary. Given as a redshift if zlimit = True, 
            otherwise given in Mpc/h
        
        cz1 (float): the maximum distance corresponding the outer 
            plot boundary. Given as a redshift if zlimit = True, 
            otherwise given in Mpc/h
            
        title (string): the plot title
        
        graph (list): list containing a pyplot figure (graph[0]), 
            a floating_axes.FloatingSubplot object (graph[1]), 
            and floating subplot axes given by
            floating_axes.FloatingSubplot.get_aux_axes (graph[2]).
            Used for plotting voids over an already existing plot.
            Defaults to none.
        
        zlimits (bool): if True, the distance limits (cz0, cz1) are 
            given as redshifts. Otherwise, they are given in Mpc/h.
            Defaults to True.
            
        rot (float): The rotation of the plot. Changes teh relative 
            positions of the ra values.
            
        colors (list): a three element list of strings specifying 
            matplotlib colors for plotting edge voids (colors[0]), 
            near-edge voids (colors[1]), and interior voids 
            (colors[2]). Defaults to ["blue","blue","blue"]
            
        gal_colors (list): a two element list of strings specifying 
            matplotlib colors for plotting wall galaxies 
            (gal_colors[0]) and void galaxies (gal_colors[1])
            
        include_gals (bool): Demtermines if galaxies are plotted
        
        include_voids (bool): Demtermines if voids are plotted
        
        alpha (float): The alpha value for the filled interiors of 
            the plotted voids
            
        border_alpha: the alpha value for the borders of the
            plotted voids
            
        horiz_legend_offset (float): the horizontal positioning of 
            the graph legend.
            
        plot_sdss (bool): Determines if the SDSS DR7 void catalog
            distance limit is plotted
            
        sdss_lim (float): SDSS DR7 void catalog distance limit 
        
        sdss_color (string): The color used to plot the SDSS DR7 
            void catalog distance limit 
        
        mag_limit (float): The magnitude limit used for selecting 
            plotted galaxies. Defaults to None, in which case all
            galaxies are plotted
            
        galaxy_point_size (float): The size of the markers for the
            plotted galaxies
        
        '''
        #set up axes
        #----------------
        if zlimits:
            cz0 = z_to_comoving_dist(np.array([cz0],dtype=np.float32),.315,1)[0]
            cz1 = z_to_comoving_dist(np.array([cz1],dtype=np.float32),.315,1)[0]

        set_bottom_invisible = False if cz0 > 0 else True #Don't illustrate the bottom axis if the redshift extends to 0
            
        if graph is None:
            fig = plt.figure(1, figsize=(1600/96,800/96))

            ax3, aux_ax3 = setup_axes3(fig, 111, ra0, ra1, cz0, cz1, rot, set_bottom_invisible)

            #aux_ax3.set_aspect(1) #Not included in original V2 code?

            plt.title(f"{title} $\delta$ = {dec}$^\circ$", loc='left')
        else:
            fig, ax3, aux_ax3 = graph[0], graph[1], graph[2]
        
        if return_plot_data:
            plot_data = []
            
        # plot voids on axes
        #----------------
        Intr,Intra = self.trint2(dec)
        #for every unique void ID
        for i in np.unique(self.trivids):
            if len(Intr[i])>0:
                #Intr2 = convint(Intr[i])
                #Intra2 = convint(Intra[i])
                #Intr2,Intra2 = convint2(Intr[i],Intra[i])
                Intc2,Icut = self.convint3(Intr[i],Intra[i])
                Intr2 = [Intc[0] for Intc in Intc2]
                Intra2 = [Intc[1] for Intc in Intc2]
                for j in range(len(Intr2)):
                    #if Icut[j]:
                    #    continue
                    
                    color = colors[0] if self.edge[i]==1 else colors[1]
                    aux_ax3.plot(Intra2[j],Intr2[j],alpha=border_alpha,color=color)
                    aux_ax3.fill(Intra2[j],Intr2[j],alpha=alpha,color=color)
                    
                    if return_plot_data:
                        plot_data.append([Intra2[j],Intr2[j]])
                #for j in range(len(Intr2)):
                #    if Icut[j]:
                #        aux_ax3.plot(Intra2[j],Intr2[j],color='blue')
                #        aux_ax3.fill(Intra2[j],Intr2[j],color='white')
        if include_gals:
            gflag_v2, wflag_v2 = self.gflag_v2, self.wflag_v2
            gra, gdec, gr  = self.gra, self.gdec, self.gr
            gdcut = (gr[wflag_v2]*np.sin((gdec[wflag_v2]-dec)*D2R))**2.<wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[wflag_v2] < mag_limit
            aux_ax3.scatter(gra[wflag_v2][gdcut],gr[wflag_v2][gdcut],color='k',s=galaxy_point_size, edgecolor=None)
            if return_plot_data:
                plot_data.append([gra[wflag_v2][gdcut],gr[wflag_v2][gdcut]])
                
            gdcut = (gr[gflag_v2]*np.sin((gdec[gflag_v2]-dec)*D2R))**2.<wdth**2.
            if mag_limit is not None:
                gdcut = gdcut * self.rmag[gflag_v2] < mag_limit
            aux_ax3.scatter(gra[gflag_v2][gdcut],gr[gflag_v2][gdcut],color='red',s=galaxy_point_size, edgecolor=None)
            if return_plot_data:
                plot_data.append([gra[gflag_v2][gdcut],gr[gflag_v2][gdcut]])
                
        if plot_sdss:
            cc=plt.Circle((0, 0), sdss_lim, color=sdss_color,fill=False,linewidth=3)
            ax3.add_artist( cc )
            
        #create legend
        void_legend_handles=[]
        legend_color=clr.to_rgba(colors[1])
        legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
        void_legend_handles.append(
            Line2D([0], [0], label='voids', marker='o', markersize=15, 
                markeredgecolor=colors[1], markerfacecolor=legend_color, linestyle=''
            )
        )
        if colors[0]!=colors[1]:
            legend_color=clr.to_rgba(colors[0])
            legend_color = (legend_color[0],legend_color[1],legend_color[2],alpha)
            void_legend_handles.append(
                Line2D([0], [0], label='edge voids', marker='o', markersize=15, 
                    markeredgecolor=colors[0], markerfacecolor=legend_color, linestyle='')
            )
        if include_gals:
            void_legend_handles.append(
                Line2D([0], [0], label="wall galaxies", marker='o', markersize=9, 
                    markeredgecolor='k', markerfacecolor='k', linestyle='')
            )
            void_legend_handles.append(
                Line2D([0], [0], label="void galaxies", marker='o', markersize=9, 
                    markeredgecolor='r', markerfacecolor='r', linestyle='')
            )
        if plot_sdss:
            void_legend_handles.append(
                Line2D([0], [0], label='SDSS limit', color=sdss_color, linewidth=5)
            )
            
        aux_ax3.legend(handles=void_legend_handles, loc="upper left", bbox_to_anchor=(horiz_legend_offset,1))
        
        
        self.graph = [fig, ax3, aux_ax3]

        if return_plot_data:
            return self.graph, plot_data
        
        return self.graph


def setup_axes3(fig, rect, ra0, ra1, cz0, cz1, rot, set_bottom_invisible):
    """
    Sometimes, things like axis_direction need to be adjusted.
    """

    # rotate a bit for better orientation
    tr_rotate = Affine2D().translate(rot, 0)

    # scale degree to radians
    tr_scale = Affine2D().scale(np.pi/180., 1.)

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_locator1 = angle_helper.LocatorDMS(4)
    tick_formatter1 = FormatterModDMS()

    grid_locator2 = MaxNLocator(3)

    #ra0, ra1 = 195, 236
    #cz0, cz1 = 0., 335. # 306.

    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                        extremes=(ra0, ra1, cz0, cz1),
                                        grid_locator1=grid_locator1,
                                        grid_locator2=grid_locator2,
                                        tick_formatter1=tick_formatter1,
                                        tick_formatter2=None,
                                        )

    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax1)

    # adjust axis
    ax1.axis["left"].set_axis_direction("bottom")
    ax1.axis["right"].set_axis_direction("top")

    if set_bottom_invisible:
        ax1.axis["bottom"].set_visible(False)
    else:
        ax1.axis["bottom"].set_axis_direction("top")
        #ax1.axis["bottom"].major_ticklabels.set_axis_direction("bottom") #This would duplicate the axes lables on the bottom
        ax1.axis["bottom"].toggle(ticklabels=False)
    
    ax1.axis["top"].set_axis_direction("bottom")
    ax1.axis["top"].toggle(ticklabels=True, label=True)
    ax1.axis["top"].major_ticklabels.set_axis_direction("top")
    ax1.axis["top"].label.set_axis_direction("top")

    ax1.axis["left"].label.set_text(r"r [Mpc h$^{-1}$]")
    ax1.axis["top"].label.set_text(r"$\alpha$")

    #print(ax1.axis["top"].major_ticklabels._grid_info)
    #print(ax1.axis["left"].major_ticklabels)
    #ax1.axis["top"].major_ticklabels.set_visible(False)
    #ax1.axis["left"].major_ticklabels.set_visible(False)
    #ax1.set_xticks([1, 2, 3])
    #ax1.set_yticks([10, 20, 30])
    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)

    aux_ax.patch = ax1.patch # for aux_ax to have a clip path as in ax
    ax1.patch.zorder=0.8 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.
    aux_ax.set_facecolor("white")

    return ax1, aux_ax

class FormatterModDMS(angle_helper.FormatterDMS):


    def __call__(self, direction, factor, values):

        values1 = [v%360 for v in values]
        
        return super().__call__(direction, factor, values1)
