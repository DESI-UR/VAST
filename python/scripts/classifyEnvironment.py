'''Identify galaxies as being in a void or not.'''


################################################################################
#
#   IMPORT LIBRARIES
#
################################################################################


import numpy as np

from astropy.table import QTable, Table

import sys
sys.path.insert(1, '/Users/kellydouglass/Documents/Research/VoidFinder/python')
from voidfinder.vflag import determine_vflag


################################################################################
#
#   CONSTANTS
#
################################################################################


c = 3e5 # km/s
h = 1
H = 100*h

DtoR = np.pi/180


################################################################################
#
#   IMPORT DATA
#
################################################################################


#voids = Table.read('SDSSdr7/vollim_dr7_cbp_102709_holes.txt', format='ascii.commented_header')
voids = Table.read('../voidfinder/data/vollim_dr7_cbp_102709_holes.txt', format='ascii.commented_header')
'''
voids['x'] == x-coordinate of center of void (in Mpc/h)
voids['y'] == y-coordinate of center of void (in Mpc/h)
voids['z'] == z-coordinate of center of void (in Mpc/h)
voids['R'] == radius of void (in Mpc/h)
voids['voidID'] == index number identifying to which void the sphere belongs
'''

#galaxy_file = input('Galaxy data file (with extension): ')
galaxy_file = '/Users/kellydouglass/Documents/Research/Rotation_curves/RotationCurves/master_file.txt'

galaxies = QTable.read(galaxy_file, format='ascii.ecsv')


################################################################################
#
#   CONVERT GALAXY ra,dec,z TO x,y,z
#
################################################################################
'''Conversions are from http://www.physics.drexel.edu/~pan/VoidCatalog/README'''


# Convert redshift to distance
galaxies_r = c*galaxies['NSA_redshift']/H

# Calculate x-coordinates
galaxies_x = galaxies_r*np.cos(galaxies['NSA_DEC']*DtoR)*np.cos(galaxies['NSA_RA']*DtoR)

# Calculate y-coordinates
galaxies_y = galaxies_r*np.cos(galaxies['NSA_DEC']*DtoR)*np.sin(galaxies['NSA_RA']*DtoR)

# Calculate z-coordinates
galaxies_z = galaxies_r*np.sin(galaxies['NSA_DEC']*DtoR)


################################################################################
#
#   IDENTIFY AS IN VOID OR NO
#
################################################################################


for i in range(len(galaxies)):

    #print('Galaxy #', galaxies['NSA_index'][i])
    
    galaxies['vflag'][i] = determine_vflag(galaxies_x[i],galaxies_y[i],galaxies_z[i], voids)


################################################################################
#
#   SAVE RESULTS
#
################################################################################


# Output file name
outfile = galaxy_file[:-4] + '_vflag.txt'

galaxies.write(outfile, format='ascii.ecsv', overwrite=True)