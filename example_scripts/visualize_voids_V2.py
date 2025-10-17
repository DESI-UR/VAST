'''Runs VoidRender on Vsquared output'''


################################################################################
# Import modules
#-------------------------------------------------------------------------------
from vast.vsquared.viz import VoidRender, load_void_data, load_galaxy_data

import numpy as np

from vispy.color import Colormap
################################################################################



################################################################################
# Load data
#-------------------------------------------------------------------------------
voids_tri_x, voids_tri_y, voids_tri_z, voids_norm, voids_id, gal_viz, gal_opp = load_void_data("DR7_V2_VIDE_Output.fits")

galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.fits")

print("Galaxies:", galaxy_data.shape)
print("Voids:", voids_tri_x.shape, voids_tri_y.shape, voids_tri_z.shape, voids_norm.shape, voids_id.shape)
################################################################################



################################################################################
# Void coloring
#-------------------------------------------------------------------------------
num_voids = len(np.unique(voids_id))

cm = Colormap(['#880000',
               '#EEEE00',
               "#008800",
               '#EE00EE',
               '#000088',
               '#EE00EE'])

void_color_vals = cm.map(np.linspace(0, 1.0, num_voids))

print(void_color_vals.shape)

void_colors = np.empty((num_voids,4), dtype=np.float32)

for idx in range(void_colors.shape[0]):
    
    void_id = idx
    
    void_colors[idx,:] = void_color_vals[void_id]
################################################################################



################################################################################
# Draw voids!
#-------------------------------------------------------------------------------
viz = VoidRender(voids_tri_x=voids_tri_x,
                 voids_tri_y=voids_tri_y,
                 voids_tri_z=voids_tri_z,
                 voids_norm=voids_norm,
                 voids_id=voids_id,
                 galaxy_xyz=galaxy_data,
                 galaxy_display_radius=4,
                 gal_viz = gal_viz,
                 gal_opp = gal_opp,
                 void_color=void_colors,
                 canvas_size=(1600,1200))

viz.run()
################################################################################



