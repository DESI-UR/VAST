'''Identify galaxies as being in a void or not.'''


################################################################################
# IMPORT LIBRARIES
#-------------------------------------------------------------------------------
import os

import numpy as np

from astropy.table import QTable, Table

import pickle

#import sys
#sys.path.insert(1, '/local/path/VAST/VoidFinder/vast/voidfinder/')
from vast.voidfinder.vflag import determine_vflag
from vast.voidfinder.distance import z_to_comoving_dist
################################################################################





################################################################################
# USER INPUT
#-------------------------------------------------------------------------------
# FILE OF VOID HOLES
#-------------------------------------------------------------------------------
void_filename = './vollim_dr7_cbp_102709_comoving_holes.txt'

dist_metric = 'comoving'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# SURVEY MASK FILE
#-------------------------------------------------------------------------------
mask_filename = './vollim_dr7_cbp_102709_mask.pickle'
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# FILE OF OBJECTS TO BE CLASSIFIED
#-------------------------------------------------------------------------------
galaxy_filename = './vollim_dr7_cbp_102709.dat'

galaxy_file_format = 'commented_header'
#-------------------------------------------------------------------------------
################################################################################





################################################################################
# CONSTANTS
#-------------------------------------------------------------------------------
c = 3e5 # km/s

h = 1
H = 100*h

Omega_M = 0.26

DtoR = np.pi/180
################################################################################





################################################################################
# IMPORT DATA
#-------------------------------------------------------------------------------
print('Importing data')

# Read in list of void holes
voids = Table.read(void_filename, format='ascii.commented_header')
'''
voids['x'] == x-coordinate of center of void (in Mpc/h)
voids['y'] == y-coordinate of center of void (in Mpc/h)
voids['z'] == z-coordinate of center of void (in Mpc/h)
voids['R'] == radius of void (in Mpc/h)
voids['voidID'] == index number identifying to which void the sphere belongs
'''


# Read in list of objects to be classified
if galaxy_file_format == 'ecsv':
    galaxies = QTable.read( galaxy_filename, format='ascii.ecsv')
    DtoR = 1.
else:
    galaxies = Table.read( galaxy_filename, format='ascii.' + galaxy_file_format)


# Read in survey mask
mask_infile = open(mask_filename, 'rb')
mask, mask_resolution, dist_limits = pickle.load(mask_infile)
mask_infile.close()

print('Data and mask imported')
################################################################################




################################################################################
# CONVERT GALAXY ra,dec,z TO x,y,z
#
# Conversions are from http://www.physics.drexel.edu/~pan/VoidCatalog/README
#-------------------------------------------------------------------------------
print('Converting coordinate system')

# Convert redshift to distance
if dist_metric == 'comoving':
    if 'Rgal' not in galaxies.columns:
        galaxies['Rgal'] = z_to_comoving_dist(galaxies['redshift'].data.astype(np.float32), Omega_M, h)
    galaxies_r = galaxies['Rgal']
    
else:
    galaxies_r = c*galaxies['redshift']/H


# Calculate x-coordinates
galaxies_x = galaxies_r*np.cos(galaxies['dec']*DtoR)*np.cos(galaxies['ra']*DtoR)

# Calculate y-coordinates
galaxies_y = galaxies_r*np.cos(galaxies['dec']*DtoR)*np.sin(galaxies['ra']*DtoR)

# Calculate z-coordinates
galaxies_z = galaxies_r*np.sin(galaxies['dec']*DtoR)

print('Coordinates converted')
################################################################################





################################################################################
# IDENTIFY LARGE-SCALE ENVIRONMENT
#-------------------------------------------------------------------------------
print('Identifying environment')

galaxies['vflag'] = -9

for i in range(len(galaxies)):

    #print('Galaxy #', galaxies['NSA_index'][i])
    
    galaxies['vflag'][i] = determine_vflag(galaxies_x[i], 
                                           galaxies_y[i], 
                                           galaxies_z[i], 
                                           voids, 
                                           mask, 
                                           mask_resolution, 
                                           dist_limits[0], 
                                           dist_limits[1])

print('Environments identified')
################################################################################





################################################################################
# SAVE RESULTS
#-------------------------------------------------------------------------------
# Output file name
galaxy_file_name, _ = os.path.splitext(galaxy_filename)
outfile = galaxy_file_name + '_vflag_' + dist_metric + '.txt'


if galaxy_file_format == 'ecsv':
    galaxies.write(outfile, format='ascii.ecsv', overwrite=True)
else:
    galaxies.write(outfile, format='ascii.' + galaxy_file_format, overwrite=True)
################################################################################





