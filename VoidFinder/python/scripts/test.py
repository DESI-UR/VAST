'''
Test plotting 
'''
import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")

data=range(1000)

plt.figure()
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

plt.grid(True,ls='-.',alpha=.4)
plt.title(r'Histogram for Flux Contrast',fontsize=16)
plt.xlabel(r'Flux Contrast $\delta$',fontsize=14)
plt.ylabel(r'Number',fontsize=18)

#plt.hist(galaxy_table['rabsmag'] ,bins=range(min(galaxy_table['rabsmag']), max(galaxy_table['rabsmag']) + 0.1, 0.1), color='teal')                                                                    
plt.hist(data, color='teal')
plt.show()                                                                                                                                                                                             

#plt.savefig('test-hist.png')
