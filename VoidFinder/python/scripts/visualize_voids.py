'''Runs VoidRender'''

################################################################################
#
# IMPORT MODULES
#
################################################################################


from voidfinder.viz import VoidRender, load_hole_data, load_galaxy_data

import numpy as np

from vispy.color import Colormap


################################################################################
#
# LOAD DATA
#
################################################################################


holes_xyz, holes_radii, holes_flags = load_hole_data("../../data/SDSS/vollim_dr7_cbp_102709_comoving_holes.txt")

#galaxy_data = load_galaxy_data('/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt')
#galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")
#galaxy_data = load_galaxy_data('kias1033_5.dat')
#galaxy_data = load_galaxy_data("dr12n.dat")
galaxy_data = load_galaxy_data("../../data/tao3043.dat")

print("Galaxies: ", galaxy_data.shape)
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
                 galaxy_xyz=galaxy_data,
                 galaxy_display_radius=10,
                 remove_void_intersects=2,
                 #void_hole_color=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
                 void_hole_color=void_hole_colors,
                 SPHERE_TRIANGULARIZATION_DEPTH=3,
                 canvas_size=(1600,1200))

viz.run()