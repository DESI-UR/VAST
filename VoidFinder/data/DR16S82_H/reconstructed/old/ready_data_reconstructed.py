'''
Remove the delta readings equal to 0 because they were set to zero 
by the tomographic map reconstruction algorithm.
'''
import sys
sys.path.insert(0, "/scratch/ierez/IGMCosmo/VoidFinder/python/")

from voidfinder.preprocessing import load_data_to_Table
import time
from astropy.io import ascii

in_directory = '/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/' 
in_filename = in_directory + 'data_reconstructed.dat'

out_filename = in_directory + 'data_reconstructed_removed.dat'

print("Loading data table at: ", in_filename, flush=True)

load_start_time = time.time()

data_table = load_data_to_Table(in_filename)

print("Data table load time: ", time.time() - load_start_time, flush=True)

print(data_table.colnames)

print(data_table[0:2])

print(len(data_table))

print(data_table['ra'][0:2])

print(data_table['delta'][0:2])

new_data_table=data_table[data_table['delta']!=0]

#new_data_table = data_table[np.where(data_table['rabsmag'] != 0)[0]] 

print(len(new_data_table))
print('I can remove 0s.')

ascii.write(new_data_table, out_filename, overwrite=True)

print("Job time: ", time.time() - load_start_time, flush=True)

print('Done:)')

