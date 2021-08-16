###################################################
#Run tomographic maps data on VoidFinder
#Data format: x,y,z, delta
#Run on comoving coordinates
###################################################


import numpy as np
from vast.voidfinder import find_voids, calculate_grid
from vast.voidfinder.preprocessing import load_data_to_Table
from astropy.io import fits
from astropy.table import Table

out_directory = '/scratch/ierez/IGMCosmo/VoidFinder/outputs/recons_runs/'

data =fits.open('mini_reconstructed.fits')

print('Can read the file')

data=Table(data[1].data)

print(len(data))

#data=data[data['y']<20]

#data=data[data['z']<20]

#data.write('mini_reconstructed.fits', format='fits', overwrite=True)

