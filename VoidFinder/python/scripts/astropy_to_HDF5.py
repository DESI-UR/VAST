


import h5py

from astropy.table import Table

"""
Description
===========

This is a helper script to convert the astropy Table .dat files into a more
efficient HDF5-based file type.  For large .dat files like the ~3GB tao3043.dat
sim dataset, it takes on the order of 60-70 seconds to read in the file using
astropy.Table.read() but only on the order of 1 second to read the data from an
HDF5 file and put it into an empty astropy.Table object.

If you use this script on your .dat file, the resulting .h5 file can be provided
as an input to the VoidFinder file_preprocess() method and can use the faster
HDF5 formatted file instead.  Just make sure your output .h5 file has the .h5
file suffix as part of its filename.


Usage
=====

Point the in_filepath and out_filepath variables below to your input and output
locations respectively.  
"""


in_filepath = '/home/moose/VoidFinder/VoidFinder/data/tao3043.dat'

out_filepath = '/home/moose/VoidFinder/VoidFinder/data/tao3043.h5'


galaxy_data_table = Table.read(in_filepath, format='ascii.commented_header')

col_names = galaxy_data_table.columns


outfile = h5py.File(out_filepath, 'w')

for col in col_names:
    
    print("Working: ", col)
    
    outfile.create_dataset(col, data=galaxy_data_table[col].data)
    
outfile.close()
    