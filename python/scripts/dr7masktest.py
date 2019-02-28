#code to plot dr7 survey mask 

from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from test_voidfinder import diff_voids

RtoD = 180./np.pi

mask_table = Table.read('cbpdr7mask.dat', format="ascii.no_header")
#out_table = Table.read('', format="ascii.commented_header")

t1_ra = mask_table.columns[0]
t1_dec = mask_table.columns[1]


#t2_ra = out_table.columns[5][:8]
#t2_dec = out_table.columns[6][:8]'''

tablename2 = 'o1.dat'
tablename1 = 'maximal_spheres_test.txt'

table_1, table_2 = diff_voids(tablename1,tablename2)

t3_ra = table_1.columns[6]
t3_dec = table_1.columns[7]

t4_ra = table_2.columns[5]
t4_dec = table_2.columns[6]


fig = plt.figure()

ax = fig.add_subplot(111)
ax.scatter(t1_ra,t1_dec,s = 5,c='mediumvioletred',label='Mock 1 Mask')
#ax.scatter(t2_ra,t2_dec,s= 1,c='red', label='HUGE spheres')
#ax.scatter(t3_ra,t3_dec,s=5,c='mediumvioletred',label='Python') 
#ax.scatter(t4_ra,t4_dec,s=5,c='darkorange',label='Fortran') 

'''ax.add_patch(
	patches.Rectangle((122.93, 0), 127.02, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((119.91, 5), 128.24, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((117.43, 10), 133.93, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((115.56, 15), 141.03, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((114.20, 20), 144.14, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((112.91, 25), 148.66, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((113.07, 30), 148.58, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((110.56, 35), 148.22, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((110, 40), 145.47, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((112.1, 45), 139.85, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((116.16, 50), 130.63, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((119.97, 55), 120.55, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((126.96, 60), 105.52, 5, color = 'green', alpha = 0.4))
ax.add_patch(
	patches.Rectangle((136.04, 65), 80.66, 5, color = 'green', alpha = 0.4))'''
ax.set_title('Mock 1 Survey Boundaries')
ax.set_xlabel('RA')
ax.set_ylabel('DEC')
#ax.set_ylim(-10,10)
#ax.set_xlim(80,100)
plt.legend()
plt.show()



