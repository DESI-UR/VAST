from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import argparse
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

gal_file = fits.open(args.input) 
gal_data = Table(gal_file[1].data)

H_0 = args.H0
O_m = args.Om
D2R = np.pi/180.

ra  = gal_data['ra']%360
dec = gal_data['dec']
r   = Distance(gal_data['z'], O_m, H_0/100.)
ang = np.array(list(zip(ra,dec)))

'''fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(ra,dec,s = 1,c='green')

ax.set_title('DESI Data Challenge Survey Boundaries')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')

plt.show()'''

nmax = 1 + int(D2R*np.amax(r)/10.)

mask = []

for i in range(1,nmax+1):
    mask.append(list(zip(*(np.unique((i*ang).astype(int),axis=0)))))

mask = np.array(mask)
np.save(args.output,mask)
