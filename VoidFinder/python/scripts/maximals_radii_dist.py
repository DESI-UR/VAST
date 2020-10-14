

from astropy.table import Table

import matplotlib.pyplot as plt
import numpy





def read_file(input_filepath):
    
    infile = open(input_filepath, 'r')
    
    data_rows = list(infile)
    
    infile.close()
    
    out_radii = []
    
    for idx, row in enumerate(data_rows):
        
        if idx == 0:
            continue
        
        else:
            
            parts = row.split(" ")
            
            radius = parts[3]
            
            out_radii.append(float(radius))
            
    return numpy.array(out_radii)
        
        




input_filepaths = ["/home/moose/VoidFinder/VoidFinder/data/SDSS/maximals_grid_5.0.txt",
                   "../../data/SDSS/maximals_grid_2.5.txt"]


colors = ['r', 'b']

labels = ["grid=5.0", "grid=2.5"]

radius_cols = []

for input_filepath in input_filepaths:
    
    print(input_filepath)
    
    curr_table = Table.read(input_filepath, format='ascii.commented_header')

    radii = curr_table['radius']
    
    #radii = read_file(input_filepath)
    
    radius_cols.append(radii)
    
all_radii = numpy.concatenate(radius_cols)

hist, bins = numpy.histogram(all_radii, bins=50)

fig = plt.figure(figsize=(14,8.75))

axes = fig.add_axes([.1, .1, .8, .8])


for radii, label, color in zip(radius_cols, labels, colors):

    axes.hist(radii, bins=bins, color=color, ec='k', alpha=0.5, label=label)
    
axes.legend()

axes.set_title("VoidFinder DR7")

axes.set_xlabel("Maximal Sphere radii")

axes.set_ylabel("Count")

#plt.show()
plt.savefig("DR7_maximals_radii.png")





