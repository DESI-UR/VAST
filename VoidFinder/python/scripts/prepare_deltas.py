'''                                                                                                                                                                                                         
Calculate ra, dec, z and rabsmag for  one of the fits files of DR16 delta fields
and store that information in the very last HDU to be read by the VoidFinder algorithm.

'''

print('Prepare one of the fits files for DR16 delta fields and store that in the last HDU.')

from astropy.table import Table
from astropy.io import fits
import os
import numpy as np

in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/delta_fields/'
os.chdir(in_directory)

filename='delta-100.fits'

data = fits.open(filename)
print(data.info())

#lambda_obs=list() #observed wavelength. This is what we measure.
#lambda_rf=list()  #wavelength in the rest frame, different from reference frame.
#This is the wavelength in the rest frame of the observer so it depends on where the quasar is.
#It shouldn't be important for me now.
#lambda_ref   #wavelength in the reference frame. This is the absorption spectrum for H that we know.

ra=list()
dec=list()
z=list()
delta=list()
prepared=Table()

for hdu_num in range(1,36):
    lambda_obs=10**(Table(data[hdu_num].data)['LOGLAM'])
    #lambda_rf=lambda_obs/((Table(data[0].data)['Z'][i]+1)
    lambda_ref= 1215.67 #angstrom, reference wavelength. This is the Lyman-alpha spectral line for H.
    z_add=(lambda_obs-lambda_ref)/lambda_ref
    z.extend(z_add)
    delta.extend(Table(data[hdu_num].data)['DELTA'])
    ra.extend(data[hdu_num].header['RA']*np.ones(len(z_add)))
    dec.extend(data[hdu_num].header['DEC']*np.ones(len(z_add)))

print(len(ra))
print(len(dec))
print(len(z))
print(len(delta))

RA=Table.Column(ra, name='ra')
DEC=Table.Column(dec, name='dec')
Z=Table.Column(z, name='z')
DELTA=Table.Column(delta, name='rabsmag')

prepared.add_column(RA)
prepared.add_column(DEC)
prepared.add_column(Z)
prepared.add_column(DELTA)

prepared = fits.BinTableHDU.from_columns([fits.Column(name='ra',  array=ra), fits.Column(name='dec', array=dec)])

print('Necessary data calculated.')

#Add the prepared data Table as the very last HDU to the original data.
prepared.write('prepared.fits', format='fits',colnames=('ra','dec','z','rabsmag'), overwrite=True)

print('I have written the file.')

filename='prepared.fits'

data = fits.open(filename)
print(data.info())

print(data[1].data)
