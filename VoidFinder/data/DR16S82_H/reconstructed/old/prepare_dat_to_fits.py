'''

map_reconstructed.dat ---> map_reconstructed.fits

Read map_reconstructed.dat data
Remove the delta readings equal to 0 because they were set to zero 
by the tomographic map reconstruction algorithm.
Randomize the row order to make the separation calculations faster.

'''

import sys
sys.path.insert(0, "/scratch/ierez/IGMCosmo/VoidFinder/python/")

from voidfinder.preprocessing import load_data_to_Table
import time
from astropy.io import ascii

from astropy.table import Table
from astropy.io import fits
import os
import numpy as np
np.random.seed(15)

in_directory = '/scratch/sbenzvi_lab/boss/dr16/reconstructed_maps/' 
in_filename = in_directory + 'map_reconstructed.dat'

out_filename = in_directory + 'map_reconstructed.fits'

print("Loading data table at: ", in_filename, flush=True)

load_start_time = time.time()

data_table = load_data_to_Table(in_filename)

print("Data table in dat format  load time: ", time.time() - load_start_time, flush=True)

print(data_table.colnames)

data_table['delta\n'].name='delta'

print(len(data_table))

new_data_table=data_table[data_table['delta']!=0]

print(len(new_data_table))

print('Removed 0s.')

np.random.shuffle(new_data_table)

print('Randomized the row order.')

new_data_table.write('map_reconstructed.fits', format='fits', overwrite=True)
                                                                                                                                                                               
print('Generated the map_reconstructed.fits file.')

