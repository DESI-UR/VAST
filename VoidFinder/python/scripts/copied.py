'''
Remove the delta readings equal to 0 because they were set to zero 
by the tomographic map reconstruction algorithm.
'''
from voidfinder.preprocessing import load_data_to_Table

import sys
sys.path.insert(0, "/scratch/ierez/IGMCosmo/VoidFinder/python/")

in_directory = '/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/' 

in_filename = in_directory + 'mini_data_reconstructed.dat'

out_filename = in_directory + 'mini_data_recontructed_removed.dat'

print("Loading data table at: ", in_filename, flush=True)

load_start_time = time.time()

data_table = load_data_to_Table(in_filename)

print("Data table load time: ", time.time() - load_start_time, flush=True)

new_data_table=data_table[data_table['rabsmag']!=0]


file = open(out_filename,"w")
file.write(new_data_table)
file.close()
print('Done:)')
