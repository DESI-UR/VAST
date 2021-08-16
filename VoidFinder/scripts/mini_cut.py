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

data =fits.open("/scratch/sbenzvi_lab/boss/dr16/reconstructed_maps/map_reconstructed.fits")

print('Can read the file')

data=Table(data[1].data)[0:1000]


#data=data[data['x']<20]

#data=data[data['y']<20]

#data=data[data['z']<20]

print(max(data['x']))

print(min(data['x']))

print(max(data['y']))

print(min(data['y']))

print(max(data['z']))

print(min(data['z']))

#data=data[data['x']<max(data['x'])*0.01]

#data=data[data['y']<max(data['y'])*0.0001]

#data=data[data['z']<max(data['z'])*0.0001]  

data.write('mini_cut.fits', format='fits', overwrite=True)

