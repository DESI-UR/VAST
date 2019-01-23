
import numpy as np
from sklearn import neighbors
from astropy.table import Table
from test_voidfinder import diff_voids
import matplotlib.pyplot as plt
import matplotlib
from table_functions import to_array


tablename2 = 'o1.dat'
tablename1 = 'maximal_spheres_test.txt'

table_1, table_2 = diff_voids(tablename1,tablename2)

table_2_array = to_array(table_2)
table_1_array= to_array(table_1)

max_spheres_tree = neighbors.KDTree(table_2_array)

dist, indices = galaxy_tree.query(table_1_array)

num_bins = 10

n, bins, patches = ax.hist(dist, num_bins, density=1)

ax.plot(bins)
ax.set_xlabel('Distance')
ax.set_ylabel('Probability')
ax.set_title(r'Histogram of Distance between Python and Fortran Different Voids')

