'''
Prepare mini version of deltafields file shifted by 90.
'''

print('Prepare mini version of deltafields file shifted by 90.')

from astropy.table import Table
from astropy.io import fits
import os
import numpy as np
in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/delta_fields/'
os.chdir(in_directory)

filename='deltafields_added90.fits'

data = fits.open(filename)
print(len(data[1].data['ra']))

prepared=Table(data[1].data[0:1000])

prepared['deltas'].name='delta'
print(len(prepared['ra']))

prepared.write('deltafields_added90_mini.fits', format='fits', overwrite=True)
