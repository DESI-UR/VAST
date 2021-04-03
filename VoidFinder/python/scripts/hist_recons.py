import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from matplotlib.transforms import Affine2D
from scipy.integrate import quad as quad
import sys
sys.path.insert(0, "/scratch/ierez/IGMCosmo/VoidFinder/python/")
#import voidfinder
#from voidfinder import distance
#from voidfinder.distance import z_to_comoving_dist
from astropy import constants as const
from astropy.table import Table

import os
from astropy.io import fits
in_directory='/scratch/ierez/IGMCosmo/VoidFinder/outputs/delta_runs/'
os.chdir(in_directory)

recons_filename = '/scratch/sbenzvi_lab/boss/dr16/reconstructed_maps/data_reconstructed_random_without0s_shifted90.fits' #delta_fields
quasars_filename='/scratch/sbenzvi_lab/boss/dr16/delta_fields/quasars.fits'
#void_filename = '/Users/kellydouglass/Documents/Research/Voids/VoidFinder/void_catalogs/SDSS/python_implementation/vollim_dr7_cbp_102709_comoving_maximal.txt'
#void_filename = '/scratch/ierez/IGMCosmo/VoidFinder/outputs/delta_runsdeltafields_added90_fixed._comoving_maximal_noMagCut.txt'

recons = fits.open(recons_filename)  
recons=Table(recons[1].data)
recons['z'].name='redshift'

quasars = fits.open(quasars_filename)  
quasars=Table(quasars[1].data)

dpi=500
mpl.rcParams['figure.dpi']= dpi
plt.figure()                                                                                        
plt.rc('text', usetex=False)                                                                        
plt.rc('font', family='serif')                                                                      
                                                                                                    
plt.grid(True,ls='-.',alpha=.4)                                                                     
plt.title(r'Histogram for redshifts of reconstructed maps for S82', fontsize=16)                                         
plt.xlabel(r'Redshift',fontsize=14)                                                                       
plt.ylabel(r'Number',fontsize=18)                                                                   
                                                                                                    
plt.hist(recons['redshift'] , bins=30, color='teal')#,bins=range(int(min(galaxies['cz'])), int(max(galaxies['cz'])) + 0.1, 0.1), color='teal')                                                                   \
                                                                                                    
#plt.hist(2*np.pi*data[1].data['ra'], color='teal')                                                  
plt.show()                                                                                          
                                                                                                    
plt.savefig('recons_redshift_histogram.png')  

dpi=500
mpl.rcParams['figure.dpi']= dpi
plt.figure()                                                                                        
plt.rc('text', usetex=False)                                                                        
plt.rc('font', family='serif')                                                                      
                                                                                                    
plt.grid(True,ls='-.',alpha=.4)                                                                     
plt.title(r'Histogram for redshifts of reconstructed maps for S82', fontsize=16)                                         
plt.xlabel(r'Redshift',fontsize=14)                                                                       
plt.ylabel(r'Number',fontsize=18)                                                                   
                                                                                                    
plt.hist(recons['delta'] , bins=30, color='teal')#,bins=range(int(min(galaxies['cz'])), int(max(galaxies['cz'])) + 0.1, 0.1), color='teal')                                                                   \
                                                                                                    
#plt.hist(2*np.pi*data[1].data['ra'], color='teal')                                                  
plt.show()                                                                                          
                                                                                                    
plt.savefig('recons_delta_histogram.png')  

import seaborn as sns
sns.set_theme(style="whitegrid")
ax = sns.violinplot(x=recons["delta"])
plt.savefig('recons_delta_violin.png')
