'''                                                                                                                                                                                                         
Read the fits files of DR16 delta fields for Stripe 82
Calculate ra, dec, z and deltas 
Store that data in a new file to be read by the VoidFinder algorithm.

'''

print('Prepare the fits file for DR16 delta fields')

from astropy.table import Table
from astropy.io import fits
import os
import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use("TkAgg")


from os import listdir
from os.path import isfile, join

in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/delta_fields/'
os.chdir(in_directory)
#############################################################################
#read_fits function reads the fits file, returns an astropy.Table with columns                                                                                                                             #ra, dec, z, deltas if the data is in the Stripe 82.
#############################################################################

def read_fits(namelist=('delta-100.fits')):


    ra=list()
    dec=list()
    z=list()
    delta=list()
    calculated=Table()

    ra_min=-43
    ra_max=45
    dec_min=-1.25 # check that
    dec_max=1.25
    lambda_ref= 1215.67 #angstrom, reference wavelength. This is the Lyman-alpha spectral line for H. 
    i=0

    for filename in namelist:
        data = fits.open(filename)
        #i=i+1
        #print(i)
        
        #print(len(data[1].data))

        for hdu_num in range(1,len(data)-1):
            if data[hdu_num].header['RA']*(180/np.pi) > 180:
                data[hdu_num].header['RA']=data[hdu_num].header['RA']*(180/np.pi)-360
            else:
                data[hdu_num].header['RA']=data[hdu_num].header['RA']*(180/np.pi)
            if data[hdu_num].header['DEC']*(180/np.pi) > 180:
                data[hdu_num].header['DEC']=data[hdu_num].header['DEC']*(180/np.pi)-360
            else:
                data[hdu_num].header['DEC']=data[hdu_num].header['DEC']*(180/np.pi)
            
            if ra_min <= data[hdu_num].header['RA'] <= ra_max and dec_min <= data[hdu_num].header['DEC'] <= dec_max:

                lambda_obs=10**(Table(data[hdu_num].data)['LOGLAM'])
                #lambda_rf=lambda_obs/((Table(data[0].data)['Z'][i]+1)                                                                        
                data[hdu_num].header['RA']=data[hdu_num].header['RA']+90 #to make life easier for VF :)
                data[hdu_num].header['DEC']=data[hdu_num].header['DEC']+90 #to make life happier for VF :)

                z_add=(lambda_obs-lambda_ref)/lambda_ref
                z.extend(z_add)
                delta.extend(Table(data[hdu_num].data)['DELTA'])
                ra.extend(data[hdu_num].header['RA']*np.ones(len(z_add)))
                dec.extend(data[hdu_num].header['DEC']*np.ones(len(z_add)))
        
    RA=Table.Column(ra, name='ra')
    DEC=Table.Column(dec, name='dec')
    Z=Table.Column(z, name='z')
    DELTA=Table.Column(delta, name='deltas')

    calculated.add_column(RA)
    calculated.add_column(DEC)
    calculated.add_column(Z)
    calculated.add_column(DELTA)

    return calculated




#if satisfies the condition write to the file
#lambda_obs=list() #observed wavelength. This is what we measure.
#lambda_rf=list()  #wavelength in the rest frame, different from reference frame.
#This is the wavelength in the rest frame of the observer so it depends on where the quasar is.
#It shouldn't be important for me now.
#lambda_ref   #wavelength in the reference frame. This is the absorption spectrum for H that we know.

#for hdu_num in range(1,len(data)-2):
#    print((data[hdu_num].header['PLATE'])/(data[hdu_num+1].header['PLATE']))
#keep list of the hdus that are in S82 and give to the next for loop 

'''
ra=list()
dec=list()
z=list()
delta=list()
prepared=Table()


for hdu_num in range(1,3):
    lambda_obs=10**(Table(data[hdu_num].data)['LOGLAM'])
    #lambda_rf=lambda_obs/((Table(data[0].data)['Z'][i]+1)
    lambda_ref= 1215.67 #angstrom, reference wavelength. This is the Lyman-alpha spectral line for H.
    z_add=(lambda_obs-lambda_ref)/lambda_ref
    z.extend(z_add)
    delta.extend(Table(data[hdu_num].data)['DELTA'])
    ra.extend(data[hdu_num].header['RA']*np.ones(len(z_add)))
    dec.extend(data[hdu_num].header['DEC']*np.ones(len(z_add)))

data = fits.open(filename)
print(data.info())
print(data[0].data)
#print(len(data.info()))                                                                                                                                                                                    
print(len(data))

#print(data[0].header)                                                                                                                                                                                      
print(data[1].header['PLATE'])

print(len(ra))
print(len(dec))
print(len(z))
print(len(delta))

RA=Table.Column(ra, name='ra')
DEC=Table.Column(dec, name='dec')
Z=Table.Column(z, name='z')
DELTA=Table.Column(delta, name='deltas')

prepared.add_column(RA)
prepared.add_column(DEC)
prepared.add_column(Z)
prepared.add_column(DELTA)

#prepared = fits.BinTableHDU.from_columns([fits.Column(name='ra',  array=ra), fits.Column(name='dec', array=dec)])
'''




onlyfiles = [f for f in listdir(in_directory) if isfile(join(in_directory, f))]

print(len(onlyfiles))
onlyfiles.remove('prepared.fits')
onlyfiles.remove('alldeltas.fits')
print(len(onlyfiles))


#print(onlyfiles[0:2])

prepared=read_fits(onlyfiles)
    
#For vstack, I am worried about possible additional [ ]
#I am trying with my method instead.

 
print('Necessary data calculated.')

prepared.write('deltafields_added90.fits', format='fits', overwrite=True)

print('I have written the file.')


filename='deltafields_added90.fits'

data = fits.open(filename)
print(data.info())
print(data[0].header)  
print(data[1].data['ra'])
print('This is the length of the merged S82 file.')
print(len(data[1].data['ra']))


out_directory="/scratch/ierez/IGMCosmo/VoidFinder/outputs/"

plt.figure()
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

plt.grid(True,ls='-.',alpha=.4)
plt.title(r'Histogram for RAs of Delta Fields',fontsize=16)
plt.xlabel(r'RA',fontsize=14)
plt.ylabel(r'Number',fontsize=18)

#plt.hist(galaxy_table['rabsmag'] ,bins=range(min(galaxy_table['rabsmag']), max(galaxy_table['rabsmag']) + 0.1, 0.1), color='teal')                                                                         
plt.hist(data[1].data['ra'], color='teal')
plt.show()

plt.savefig(out_directory+'ra_distn_giventoVF.png')  


plt.figure()
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

plt.grid(True,ls='-.',alpha=.4)
plt.title(r'Histogram for DECs of Delta Fields',fontsize=16)
plt.xlabel(r'DEC',fontsize=14)
plt.ylabel(r'Number',fontsize=18)

#plt.hist(galaxy_table['rabsmag'] ,bins=range(min(galaxy_table['rabsmag']), max(galaxy_table['rabsmag']) + 0.1, 0.1), color='teal')                                                                        \
                                                                                                                                                                                                            
plt.hist(data[1].data['dec'], color='teal')
plt.show()

plt.savefig(out_directory+'dec_distn_giventoVF.png')


plt.figure()
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

plt.grid(True,ls='-.',alpha=.4)
plt.title(r'RA vs DEC of Delta Fields in S82',fontsize=16)
plt.xlabel(r'RA',fontsize=14)
plt.ylabel(r'DEC',fontsize=18)

plt.scatter(data[1].data['ra'],data[1].data['dec'], color='teal', s=5, label='Stripe 82')
plt.show()

plt.savefig(out_directory+'ravsdec_giventoVF.png')






#vstack to merge tables
#stripe 82

