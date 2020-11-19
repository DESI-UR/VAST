from astropy.table import Table
from astropy.io import fits
import os
import numpy as np
np.random.seed(15)

in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/delta_fields/'
os.chdir(in_directory)

filename='deltafields_added90.fits'
data = fits.open(filename)
main= data[1].data

print(main)
