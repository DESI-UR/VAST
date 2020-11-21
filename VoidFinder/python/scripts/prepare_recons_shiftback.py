'''
Shift back dec of delta fields file by 90.
Fix naming from deltas to delta
'''

print('Fix reconstructed maps file shifted by 90.')

from astropy.table import Table
from astropy.io import fits
import os
import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib

#in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/delta_fields/'
in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/'
os.chdir(in_directory)
out_directory="/scratch/ierez/IGMCosmo/VoidFinder/outputs/"

filename='data_reconstructed_random_without0s_shifted90.fits'

data = fits.open(filename)
print(data[1].data[0:5])

prepared=Table(data[1].data)
'''
prepared['dec']=prepared['dec']-90

prepared['deltas'].name = 'delta'
print(len(prepared['ra']))
print(prepared[0:5])



prepared.write('data_reconstructed_random_without0s_shifted90.fits', format='fits', overwrite=True)
'''

print(len(prepared))
plt.figure()                                                                                       
plt.rc('text', usetex=False)                                                                        
plt.rc('font', family='serif')                                                                      
plt.grid(True,ls='-.',alpha=.4)                                                                     
plt.title(r'RA vs DEC of Reconstructed Maps in S82 with shifted RA given to VF',fontsize=16)        
plt.xlabel(r'RA',fontsize=14)                                                                       
plt.ylabel(r'DEC',fontsize=18)                                                                      
plt.scatter(prepared['ra'],prepared['dec'], color='teal', s=5, label='Stripe 82')                           
plt.show()                                                                                          
plt.savefig(out_directory+'ravsdec_shifted_VF_recons.png')   

