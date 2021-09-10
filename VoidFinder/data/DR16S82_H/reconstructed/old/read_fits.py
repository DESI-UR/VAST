print('Read data_reconstructed.fits file')

from astropy.table import Table
from astropy.io import fits
import os
import numpy as np

in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/'
os.chdir(in_directory)

filename='data_reconstructed.fits'

data = fits.open(filename)
print(data.info())

print(data[0].data)

print(data[1].data['RA'])

main= data[1]

print(main)

print(main.data['deltas'])

print(len(main.data['deltas']))

main=np.random.shuffle(main)

