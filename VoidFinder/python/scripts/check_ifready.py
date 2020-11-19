'''
Check prepared.fits file
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

filename='deltafields_added90_fixed.fits'

data = fits.open(filename)
print(data.info())
print('This is the length of the merged S82 file.')
print(len(data[1].data['ra']))


out_directory="/scratch/ierez/IGMCosmo/VoidFinder/outputs/"

data=data[1].data


plt.figure()
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

plt.grid(True,ls='-.',alpha=.4)
plt.title(r'RA vs DEC of Delta Fields in S82 with shifted RA given to VF',fontsize=16)
plt.xlabel(r'RA',fontsize=14)
plt.ylabel(r'DEC',fontsize=18)

plt.scatter(data['ra'],data['dec'], color='teal', s=5, label='Stripe 82')
plt.show()

plt.savefig(out_directory+'ravsdec_shifted_VF.png')


