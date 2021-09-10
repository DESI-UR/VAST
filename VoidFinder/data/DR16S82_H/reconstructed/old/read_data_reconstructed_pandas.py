'''                                                                                                                                                                                                         
Remove the delta readings equal to 0 because they were set to zero                                                                                                                                          
by the tomographic map reconstruction algorithm. Use the dataframe method.
                                                                                                                                                            
'''

import pandas as pd
import numpy as np

in_directory = '/scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/'
file_name = in_directory + 'mini_data_reconstructed.dat'
out_file = in_directory + 'trial2.dat'

df = pd.read_table(file_name, sep="\s+", usecols=['ra','dec','z','rabsmag'])
print('Read the file')

new_df = np.fromstring(df.rabsmag.astype('|S7').tobytes().replace(b'0',b''), dtype='|S6')
print('Removed 0s')

file = open(out_file,"w")
file.write(new_df)
file.close()
print('Done:)')
