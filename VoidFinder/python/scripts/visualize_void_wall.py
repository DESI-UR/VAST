'''Runs VoidRender'''

################################################################################
#
# IMPORT MODULES
#
################################################################################


from voidfinder.viz import VoidRender, load_hole_data, load_galaxy_data

import numpy as np

from vispy.color import Colormap

from astropy.table import Table


################################################################################
#
# LOAD DATA
#
################################################################################


holes_xyz, holes_radii, holes_flags = load_hole_data("../../data/SDSS/vollim_dr7_cbp_102709_comoving_holes.txt")


wall_gal_infile = "../../data/SDSS/SDSS_dr7_wall_gal_file.txt"

wall_gal_table = Table.read(wall_gal_infile, format="ascii.commented_header")

#print(wall_gal_table)

num_wall = len(wall_gal_table)


wall_galaxy_data = np.concatenate((wall_gal_table['x'].data.reshape(num_wall,1), 
                              wall_gal_table['y'].data.reshape(num_wall,1), 
                              wall_gal_table['z'].data.reshape(num_wall,1)), axis=1)



field_galaxy_infile = "../../data/SDSS/SDSS_dr7_field_gal_file.txt"

field_gal_table = Table.read(field_galaxy_infile, format="ascii.commented_header")

num_field = len(field_gal_table)

field_galaxy_data = np.concatenate((field_gal_table['x'].data.reshape(num_field,1), 
                              field_gal_table['y'].data.reshape(num_field,1), 
                              field_gal_table['z'].data.reshape(num_field,1)), axis=1)




print("Galaxies: ", wall_galaxy_data.shape, field_galaxy_data.shape)
print("Holes: ", holes_xyz.shape, holes_radii.shape, holes_flags.shape)
################################################################################
#
# VOID COLORING
#
################################################################################


hole_IDs = np.unique(holes_flags)

num_hole_groups = len(hole_IDs)

cm = Colormap(['#880000',
               '#EEEE00',
               "#008800",
               '#EE00EE',
               '#000088',
               '#EE00EE'])

hole_color_vals = cm.map(np.linspace(0, 1.0, num_hole_groups))

print(hole_color_vals.shape)

void_hole_colors = np.empty((holes_xyz.shape[0],4), dtype=np.float32)

for idx in range(void_hole_colors.shape[0]):
    
    hole_group = holes_flags[idx] 
    
    #print(hole_group)
    
    void_hole_colors[idx,:] = hole_color_vals[hole_group-1] #uhg you used 1-based indexing WHY? :D
        

################################################################################
#
# DRAW VOIDS!
#
################################################################################


viz = VoidRender(holes_xyz=holes_xyz, 
                 holes_radii=holes_radii,
                 holes_group_IDs=holes_flags,
                 galaxy_xyz=field_galaxy_data,
                 galaxy_color=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
                 wall_galaxy_xyz=wall_galaxy_data,
                 wall_distance=5.95,
                 wall_galaxy_color=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                 galaxy_display_radius=10.0,
                 remove_void_intersects=1,
                 void_hole_color=void_hole_colors,
                 SPHERE_TRIANGULARIZATION_DEPTH=3,
                 canvas_size=(1600,1200))

viz.run()



