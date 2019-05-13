from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from absmag_comovingdist_functions import Distance

parser = argparse.ArgumentParser(description='make mask')
parser.add_argument('--input', '-i',
                    help    = 'input file, .fits',
                    dest    = 'input',
                    default = None,
                    type    = str)
parser.add_argument('--output', '-o',
                    help    = 'output file, .npy',
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




#################################################################
# Mucking around with the input
#################################################################
#gal_file = fits.open(args.input) 
#gal_data = Table(gal_file[1].data)

gal_data = Table.read(args.input, format="ascii.commented_header")

#################################################################
# 
#################################################################



H_0 = args.H0
O_m = args.Om
D2R = np.pi/180.

ra  = gal_data['ra']%360
dec = gal_data['dec']
r   = Distance(gal_data['redshift'], O_m, H_0/100.)
ang = np.array(list(zip(ra,dec)))


'''
################################################################################
# Build variable resolution mask
#-------------------------------------------------------------------------------
nmax = 1 + int(D2R*np.amax(r)/10.)

mask = []

for i in range(1,nmax+1):
    mask.append(list(zip(*(np.unique((i*ang).astype(int),axis=0)))))

mask = np.array(mask)
np.save(args.output,mask)
################################################################################
'''


################################################################################
# Build highest resolution mask necessary for survey
#-------------------------------------------------------------------------------

# Mask resolution (inverse of the angular radius of the minimum void at the 
# maximum distance)
mask_resolution = 1 + int(D2R*np.amax(r)/10)

# Scale all coordinates by mask_resolution
mask = list(zip(*(np.unique((mask_resolution*ang).astype(int), axis=0))))

# Convert to numpy array
mask = np.array(mask)

# Save scaled survey mask coordinates and mask resolution
outfile = open(args.output, 'wb')
pickle.dump((mask_resolution, mask), outfile)
outfile.close()
################################################################################



'''fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(ra,dec,s = 1,c='green')

ax.set_title('DESI Data Challenge Survey Boundaries')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')

plt.show()'''
