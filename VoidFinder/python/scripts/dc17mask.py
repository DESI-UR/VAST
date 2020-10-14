#code to create DESI data challenge mask for VoidFinder

from astropy.io import fits 
import matplotlib.pyplot as plt
from astropy.table import Table, unique
import numpy as np

gal_file = fits.open('ztargets_galaxiesonly_modified.fits') 
gal_data = Table(gal_file[1].data)


ra = gal_data['RA']
dec = gal_data['DEC']

'''fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(ra,dec,s = 1,c='green')

ax.set_title('DESI Data Challenge Survey Boundaries')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')

plt.show()'''


#Sorting NGC and SGC
ngcbool_ra = np.logical_and(ra<=300,ra>=80)
ngcbool_dec = np.logical_and(dec<=80,dec>=-15)
ngcbool = np.logical_and(ngcbool_ra,ngcbool_dec)
ngc = gal_data[ngcbool]
sgc = gal_data[np.logical_not(ngcbool)]

ngc['RA'] = ngc['RA'].astype(int)
ngc['DEC'] = ngc['DEC'].astype(int)
ngc.remove_column('Z')
ngc.remove_column('FLUX_R')

sgc['RA'] = sgc['RA'].astype(int)
sgc['DEC'] = sgc['DEC'].astype(int)
sgc.remove_column('Z')
sgc.remove_column('FLUX_R')

ngc = unique(ngc)
sgc = unique(sgc)

'''ngc.write('dc17ngcmask.dat',format='ascii.no_header')
sgc.write('dc17sgcmask.dat',format='ascii.no_header')'''

fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(ngc['RA'],ngc['DEC'],s = 1,c='blue')
ax.scatter(sgc['RA'],sgc['DEC'],s = 1,c='red')
ax.set_title('DESI Data Challenge Survey Boundaries')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')

plt.show()

