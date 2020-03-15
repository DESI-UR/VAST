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
################################################################################

infile1 = open("/home/moose/VoidFinder/VoidFinder/doc/profiling/baseline_2020_03/alg_comparison/VF_DEBUG1.txt", 'r')

results_original = list(infile1)

infile1.close()

infile2 = open("/home/moose/VoidFinder/VoidFinder/doc/profiling/baseline_2020_03/alg_comparison/VF_DEBUG_new.txt", 'r')

results_new = list(infile2)

infile2.close()

print(len(results_original), len(results_new))


map_original = {}

for idx, row in enumerate(results_original):

    parts = row.split(" ")

    cell_ID = parts[0]

    result_name = parts[1].strip()

    if result_name == "hole":
        
        hole_data = parts[2]
        
    else:
        
        hole_data = None

    map_original[cell_ID] = [result_name, hole_data]


hole_to_nim = []
nim_to_hole = []

for idx, row in enumerate(results_new):

    parts = row.split(" ")

    cell_ID = parts[0]

    new_result_name = parts[1].strip()
    
    if new_result_name == "hole":
        new_hole_data = parts[2]

    old_result_name = map_original[cell_ID][0]
    old_result_data = map_original[cell_ID][1]


    if old_result_name == "hole" and "nim" in new_result_name:
        
        old_hole_data = old_result_data.strip().split(",")
        
        x = float(old_hole_data[0])
        y = float(old_hole_data[1])
        z = float(old_hole_data[2])
        r = float(old_hole_data[3])
        
        hole_to_nim.append((x,y,z,r))
        
        
    elif "nim" in old_result_name and "hole" == new_result_name:
        
        
        new_data = new_hole_data.strip().split(",")

        x = float(new_data[0])
        y = float(new_data[1])
        z = float(new_data[2])
        r = float(new_data[3])
        
        nim_to_hole.append((x,y,z,r))





print(len(hole_to_nim), len(nim_to_hole))



hole_to_nim = np.array(hole_to_nim)
nim_to_hole = np.array(nim_to_hole)


num_hole_to_nim = hole_to_nim.shape[0]
num_nim_to_hole = nim_to_hole.shape[0]


holes_xyz = np.concatenate((hole_to_nim[:,0:3], nim_to_hole[:,0:3]), axis=0)
holes_radii = np.concatenate((hole_to_nim[:,3], nim_to_hole[:,3]), axis=0)
holes_flags = np.ones(num_hole_to_nim+num_nim_to_hole, dtype=np.int32)
holes_flags[0:num_hole_to_nim] = 0


################################################################################
#
# LOAD DATA
#
################################################################################


#holes_xyz, holes_radii, holes_flags = load_hole_data("../../data/SDSS/vollim_dr7_cbp_102709_comoving_holes.txt")
#holes_xyz, holes_radii, holes_flags = load_hole_data("../../data/SDSS/vollim_dr7_cbp_102709_holes.txt")

#galaxy_data = load_galaxy_data('/Users/kellydouglass/Documents/Drexel/Research/Data/kias1033_5_P-MJD-F_MPAJHU_ZdustOS_stellarMass_BPT_SFR_NSA_correctVflag_Voronoi_CMD.txt')
galaxy_data = load_galaxy_data("../../data/SDSS/vollim_dr7_cbp_102709.dat")
#galaxy_data = load_galaxy_data("vollim_dr7_cbp_102709.dat")
#galaxy_data = load_galaxy_data('kias1033_5.dat')
#galaxy_data = load_galaxy_data("dr12n.dat")
#galaxy_data = load_galaxy_data("../../data/tao3043.dat")

print("Galaxies: ", galaxy_data.shape)
print("Holes: ", holes_xyz.shape, holes_radii.shape, holes_flags.shape)
################################################################################
#
# VOID COLORING
#
################################################################################


hole_IDs = np.unique(holes_flags)

num_hole_groups = len(hole_IDs)

'''
cm = Colormap(['#880000',
               '#EEEE00',
               "#008800",
               '#EE00EE',
               '#000088',
               '#EE00EE'])

hole_color_vals = cm.map(np.linspace(0, 1.0, num_hole_groups))

print(hole_color_vals.shape)
'''

hole_color_vals = np.array([[0.0, 1.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 1.0]], dtype=np.float32)

void_hole_colors = np.empty((holes_xyz.shape[0],4), dtype=np.float32)

for idx in range(void_hole_colors.shape[0]):
    
    hole_group = holes_flags[idx] 
    
    void_hole_colors[idx,:] = hole_color_vals[hole_group]
        

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
                 remove_void_intersects=1,
                 #void_hole_color=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
                 void_hole_color=void_hole_colors,
                 SPHERE_TRIANGULARIZATION_DEPTH=3,
                 canvas_size=(1600,1200))

viz.run()