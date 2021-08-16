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

data =fits.open("mini_reconstructed.fits")

data=Table(data[1].data)

print('Can read the file')


x = data['x']
y = data['y']
z = data['z']
delta=data['delta']


num_data = len(delta)

print(num_data)

'''
wall_coords_xyz = np.concatenate((x.reshape(num_gal,1),
                                  y.reshape(num_gal,1),
                                  z.reshape(num_gal,1)), axis=1)

hole_grid_edge_length = 5.0

hole_grid_shape, coords_min, coords_max = calculate_grid(wall_coords_xyz,
                                                         hole_grid_edge_length)


xyz_limits = np.concatenate((coords_min.reshape(1,3),
                             coords_max.reshape(1,3)), axis=0)



survey_name = "DR16S82_reconstructed_"

out1_filename = "maximals.txt"

out2_filename = "voids.txt"


find_voids(wall_coords_xyz,
           coords_min,
           hole_grid_shape,
           survey_name,
           mask_type='xyz',
           mask=None, 
           mask_resolution=None,
           min_dist=None,
           max_dist=None,
           xyz_limits=xyz_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           #hole_grid_edge_length=hole_grid_edge_length,
           #galaxy_map_grid_edge_length=None,
           #hole_center_iter_dist=1.0,
           maximal_spheres_filename=out_directory+out1_filename,
           void_table_filename=out_directory+out2_filename,
           potential_voids_filename=out_directory+survey_name+'potential_voids_list.txt',
           num_cpus=4,
           batch_size=10000,
           verbose=1,
           print_after=5.0)



'''




