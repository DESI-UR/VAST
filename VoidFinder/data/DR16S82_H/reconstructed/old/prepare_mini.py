from astropy.table import Table
from astropy.io import fits
import os
import numpy as np

print("Prepare a mini version of the main data set with its first 1000 lines.")

in_directory='/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/'
os.chdir(in_directory)

filename='data_reconstructed_random_without0s.fits'

out_file='data_reconstructed_random_without0s_mini.fits'

print('Remove the zeros in the randomized reconstructed maps so they can be directly given to the VF.')
data = fits.open(filename)
print(data[1].data[0:5])
data=data[1].data
data_table = Table()

print(data[0:5])

print('Inside load table before columns')
RA=Table.Column(data['RA'], name='ra')
DEC=Table.Column(data['DEC'], name='dec')
Z=Table.Column(data['Z'], name='redshift')
DELTA=Table.Column(data['deltas'], name='deltas')

#print(data['Z'][0:5])

print('Inside load table before adding columns')
data_table.add_column(RA)
data_table.add_column(DEC)
data_table.add_column(Z)
data_table.add_column(DELTA)

print('Length of data before removing 0s:')
print(len(data_table))

data_table=data_table[data_table[0:1000]]
print('Length of data after removing 0s:')
print(len(data_table))
'''                                                                                                                            
print('fits to table done:)')                                                                                                  
                                                                                                                               
print('Necessary data calculated.')                                                                                            
'''
data_table.write(out_file, format='fits', overwrite=True)

print('I have written the file.')
