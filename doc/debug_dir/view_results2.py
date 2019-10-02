

import matplotlib.pyplot as plt
import numpy
import pickle















infile = open("/home/moose/VoidFinder/doc/profiling/Cell_Processing_Times_SingleThreadCython.pickle", 'rb')
single_data = pickle.load(infile)
infile.close()

'''
infile = open("/home/moose/VoidFinder/doc/profiling/Cell_Processing_Times_MultiThreadCython.pickle", 'rb')
multi_data = pickle.load(infile)
infile.close()
'''


print(single_data[0:10])
#print(multi_data[0:10])
print(single_data.min())
#print(multi_data.min())


hist, bins = numpy.histogram(single_data, bins=50)

plt.hist(single_data, bins=bins, color='r', label="single")
#plt.hist(multi_data, bins=bins, color='b', label='multi')
plt.show()