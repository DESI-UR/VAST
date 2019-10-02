

import pickle
import numpy

import matplotlib.pyplot as plt









total_cycles = 0

for idx in range(8):
    
    infile = open("multi_gen_times_"+str(idx)+".pickle", 'rb')
    worker_gen_times, worker_main_times = pickle.load(infile)
    infile.close()
    
    print(len(worker_gen_times), len(worker_main_times))
    
    total_cycles += len(worker_gen_times)
    
    
    
    
    worker_gen_times = numpy.array(worker_gen_times)
    
    worker_cell_ID_gen_times = (worker_gen_times[:,0] - worker_gen_times[:,1])/(1.0e9)
    
    plt.hist(worker_cell_ID_gen_times, bins=50)
    plt.title("Worker: "+str(idx))
    plt.show()
    plt.close()
    
    
print(total_cycles, total_cycles*10000)