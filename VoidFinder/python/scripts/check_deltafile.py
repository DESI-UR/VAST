'''
Check prepared.firs file
'''
print('Check prepared.fits file')

from astropy.table import Table
from astropy.io import fits
import os
import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use("TkAgg")                                                                                               


from os import listdir
from os.path import isfile, join

in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/delta_fields/'
os.chdir(in_directory)

filename='prepared.fits'

data = fits.open(filename)
print(data.info())
#print(data[0].header)
#print(data[1].data['ra'])
print('This is the length of the merged S82 file.')
print(len(data[1].data['ra']))


out_directory="/scratch/ierez/IGMCosmo/VoidFinder/outputs/"

ra=data[1].data['ra']

print(len(ra))

print(ra[5])

print(np.pi)

for i in range(len(ra)):
    if ra[i] > np.pi:
        ra[i] = (ra[i]-2*np.pi)*(180/np.pi) 
    
dec=data[1].data['dec']

for j in range(len(ra)):
    dec[j] = dec[j]*(180/np.pi)





plt.figure()
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

plt.grid(True,ls='-.',alpha=.4)
plt.title(r'RA vs DEC of Delta Fields in S82',fontsize=16)
plt.xlabel(r'RA',fontsize=14)
plt.ylabel(r'DEC',fontsize=18)

plt.scatter(ra,dec, color='teal', s=5, label='Stripe 82')
plt.show()

plt.savefig(out_directory+'ravsdec_adapted.png')


