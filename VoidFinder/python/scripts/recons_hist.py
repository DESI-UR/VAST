'''
Print a histogram of delta values of reconstructed maps, along with distrbution for redshift.
'''

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
in_directory='/scratch/ierez/IGMCosmo/VoidFinder/outputs/'
os.chdir(in_directory)

recons_filename = '/scratch/sbenzvi_lab/boss/dr16/reconstructed_maps/data_reconstructed_random_without0s_shifted90.fits' 

recons = fits.open(recons_filename)  
recons=Table(recons[1].data)
recons['z'].name='redshift'

#Histogram of redshift
dpi=500
mpl.rcParams['figure.dpi']= dpi
plt.figure()                                                                                        
plt.rc('text', usetex=False)                                                                        
plt.rc('font', family='serif')                                                                     
plt.grid(True,ls='-.',alpha=.4)                                                                     
plt.title(r'Histogram for redshifts of reconstructed maps for S82', fontsize=16)                                         
plt.xlabel(r'Redshift',fontsize=14)                                                               
plt.ylabel(r'Number',fontsize=18)                                                                  
plt.hist(recons['redshift'] , bins=30, color='teal')
plt.savefig('recons_redshift_distn.png')  
#max(quasars['comoving'])

#Histogram of redshift                                                                             
dpi=500
mpl.rcParams['figure.dpi']= dpi
plt.figure()                                                                                       
plt.rc('text', usetex=False)                                                                       
plt.rc('font', family='serif')
plt.grid(True,ls='-.',alpha=.4)                                                                    
plt.title(r'Histogram for redshifts of reconstructed maps for S82', fontsize=16)                   
plt.xlabel(r'Redshift',fontsize=14)                                                               

plt.ylabel(r'Number',fontsize=18)                                                                 

plt.hist(recons['redshift'] , bins=100, color='teal')#,bins=range(int(min(galaxies['cz'])), int(max(
#plt.hist(2*np.pi*data[1].data['ra'], color='teal')                                                
#plt.show()
plt.savefig('recons_redshift_distn2.png')
#max(quasars['comoving'])   

import seaborn as sns
#sns.set_theme(style="whitegrid")
#tips = sns.load_dataset("tips")
#ax = sns.violinplot(x=galaxies["delta"])
plt.xlabel(r'Redshift',fontsize=14)
plt.ylabel(r'Number',fontsize=18)
plt.title(r'Violin plot for redshifts of reconstructed maps for S82', fontsize=16)
ax = sns.violinplot(x=recons["redshift"])
plt.savefig('recons_redshift_distn_violin.png')
#Histogram of redshift                                                                            
dpi=500
mpl.rcParams['figure.dpi']= dpi
plt.figure()                                                                                       
plt.rc('text', usetex=False)                                                                      

plt.rc('font', family='serif')
plt.grid(True,ls='-.',alpha=.4)                                                                    
plt.title(r'Histogram for delta values of reconstructed maps for S82', fontsize=16)                  

plt.xlabel(r'$\delta$',fontsize=14)                                                               

plt.ylabel(r'Number',fontsize=18)                                                  

plt.hist(recons['delta'] , bins=30, color='teal')
plt.savefig('recons_delta_distn.png')
#max(quasars['comoving'])                                                                          

#Histogram of redshift                                                               
dpi=500
mpl.rcParams['figure.dpi']= dpi
plt.figure()
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.grid(True,ls='-.',alpha=.4)
plt.title(r'Histogram for delta values of reconstructed maps for S82', fontsize=16)
plt.xlabel(r'$\delta$',fontsize=14)

plt.ylabel(r'Number',fontsize=18)

plt.hist(recons['delta'] , bins=100, color='teal')
plt.savefig('recons_delta_distn2.png')
                                                  

import seaborn as sns
plt.xlabel(r'$\delta$',fontsize=14)
plt.ylabel(r'Number',fontsize=18)
plt.title(r'Violin plot for delta values of reconstructed maps for S82', fontsize=16)
ax = sns.violinplot(x=recons["delta"])
plt.savefig('recons_delta_distn_violin.png')

print('Done:)')
