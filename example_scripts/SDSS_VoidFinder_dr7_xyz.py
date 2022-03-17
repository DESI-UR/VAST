




import numpy as np
from vast.voidfinder import find_voids, calculate_grid
from vast.voidfinder.preprocessing import load_data_to_Table



wall_coords_xyz = load_data_to_Table("SDSS_dr7_wall_gal_file.txt")

x = wall_coords_xyz['x']
y = wall_coords_xyz['y']
z = wall_coords_xyz['z']

num_gal = x.shape[0]

wall_coords_xyz = np.concatenate((x.reshape(num_gal,1),
                                  y.reshape(num_gal,1),
                                  z.reshape(num_gal,1)), axis=1)

hole_grid_edge_length = 5.0

hole_grid_shape, coords_min, coords_max = calculate_grid(wall_coords_xyz,
                                                         hole_grid_edge_length)


xyz_limits = np.concatenate((coords_min.reshape(1,3),
                             coords_max.reshape(1,3)), axis=0)


survey_name = "SDSS_dr7_"

out1_filename = "maximals.txt"

out2_filename = "voids.txt"


find_voids(wall_coords_xyz,
           coords_min,
           hole_grid_shape,
           survey_name,
           mask_type='xyz',
           mask=None, 
           mask_resolution=None,
           dist_limits=None,
           xyz_limits=xyz_limits,
           #save_after=50000,
           #use_start_checkpoint=True,
           hole_grid_edge_length=hole_grid_edge_length,
           galaxy_map_grid_edge_length=None,
           hole_center_iter_dist=1.0,
           maximal_spheres_filename=out1_filename,
           void_table_filename=out2_filename,
           potential_voids_filename=survey_name+'potential_voids_list.txt',
           num_cpus=4,
           batch_size=10000,
           verbose=1,
           print_after=5.0)








