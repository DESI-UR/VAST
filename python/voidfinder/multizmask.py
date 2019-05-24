from astropy.io import fits
from astropy.table import Table

import numpy as np
#import argparse
#import pickle

from .absmag_comovingdist_functions import Distance

'''
parser = argparse.ArgumentParser(description='make mask')
parser.add_argument('--input', '-i',
                    help    = 'input file, .fits',
                    dest    = 'input',
                    default = None,
                    type    = str)
parser.add_argument('--output', '-o',
                    help    = 'output file, .pickle',
                    dest    = 'output',
                    default = None,
                    type    = str)
parser.add_argument('--H0','-H',
                    help    = 'H_0',
                    dest    = 'H0',
                    default = 100.,
                    type    = float)
parser.add_argument('--OmegaM','-m',
                    help    = 'Omega_m',
                    dest    = 'Om',
                    default = 0.3,
                    type    = float)
args = parser.parse_args()


################################################################################
# Read in galaxy file
################################################################################
#gal_file = fits.open(args.input) 
#gal_data = Table(gal_file[1].data)

gal_data = Table.read(args.input, format="ascii.commented_header")
################################################################################
'''


def generate_mask(gal_data, H_0=100., O_m=0.3):
     '''
     Generate sky mask that identifies the footprint of the input galaxy survey.


     Parameters:
     ===========

     galaxy_filename : string
          Name of file containing list of galaxies.

     mask_filename : string
          Name of file within which to save mask.

     H_0 : float
          Hubble's constant.  Default value is 100h.

     O_m : float
          Omega-matter.  Default value is 0.3.


     Returns:
     ========

     mask : numpy array of shape (2,n)
          n pairs of RA,dec coordinates that are within the survey limits and 
          are scaled by the mask_resolution.  Oth row is RA; 1st row is dec.

     mask_resolution : integer
          Scale factor of coordinates in maskfile
     '''


     #H_0 = args.H0
     #O_m = args.Om
     D2R = np.pi/180.

     ra  = gal_data['ra']%360
     dec = gal_data['dec']
     r   = Distance(gal_data['redshift'], O_m, H_0/100.)
     ang = np.array(list(zip(ra,dec)))


     '''
     ###########################################################################
     # Build variable resolution mask
     #--------------------------------------------------------------------------
     nmax = 1 + int(D2R*np.amax(r)/10.)

     mask = []

     for i in range(1,nmax+1):
         mask.append(list(zip(*(np.unique((i*ang).astype(int),axis=0)))))

     mask = np.array(mask)
     np.save(args.output,mask)
     ###########################################################################
     '''


     ###########################################################################
     # Build highest resolution mask necessary for survey
     #--------------------------------------------------------------------------

     # Mask resolution (inverse of the angular radius of the minimum void at the 
     # maximum distance)
     mask_resolution = 1 + int(D2R*np.amax(r)/10)

     # Scale all coordinates by mask_resolution
     mask = list(zip(*(np.unique((mask_resolution*ang).astype(int), axis=0))))

     # Convert to numpy array
     mask = np.array(mask)

     '''
     # Save scaled survey mask coordinates and mask resolution
     #outfile = open(args.output, 'wb')
     outfile = open(mask_filename, 'wb')
     pickle.dump((mask_resolution, mask), outfile)
     outfile.close()
     '''
     ###########################################################################


     return mask, mask_resolution

