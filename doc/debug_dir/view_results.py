
import matplotlib.pyplot as plt




import numpy


import pickle



try:
    infile = open("single_thread_profile.pickle", 'rb')
    single_times, single_samples = numpy.array(pickle.load(infile))
    infile.close()
except:
    single_times = None
    single_samples = None

try:
    infile = open("multi_thread_profile.pickle", 'rb')
    multi_times, multi_samples = numpy.array(pickle.load(infile))
    infile.close()
except:
    multi_times = None
    multi_samples = None



def plot_abs(single_times, single_samples, multi_times, multi_samples):
    
    min_y = 1000000000000.0
    max_y = 0.0
    
    min_x = 1000000000000.0
    max_x = 0.0
    
    if single_times is not None:
        
        plt.plot(single_times, single_samples, color='r', label="single")
        
        min_y = min(min(single_samples), min_y)
        
        max_y = max(max(single_samples), max_y)
        
        min_x = min(min(single_times), min_x)
        
        max_x = max(max(single_times), max_x)
        
        
    if multi_times is not None:
        
        plt.plot(multi_times, multi_samples, color='b', label="multi")
        
        min_y = min(min(multi_samples), min_y)
        
        max_y = max(max(multi_samples), max_y)
        
        min_x = min(min(multi_times), min_x)
        
        max_x = max(max(multi_times), max_x)
        
    plt.legend(loc=4)
    
    
    
    
    y_range = max_y - min_y
    x_range = max_x - min_x
    
    plt.ylim(min_y - .05*y_range, max_y + .05*y_range)
    plt.xlim(min_x - .05*x_range, max_x + .05*x_range)
    
    plt.ylabel("Cells Completed")
    plt.xlabel("Time [s]")
    
    plt.show()
    
def plot_rate(single_times, single_samples, multi_times, multi_samples):
    
    min_y = 1000000000000.0
    max_y = 0.0
    
    min_x = 1000000000000.0
    max_x = 0.0
    
    if single_times is not None:
        
        rates = numpy.diff(single_samples)/numpy.diff(single_times)
        
        rates[rates<1.0] = 1.0
        
        plt.semilogy(single_times[1:], rates, color='r', label="single")
        
        min_y = min(min(rates), min_y)
        
        max_y = max(max(rates), max_y)
        
        min_x = min(min(single_times), min_x)
        
        max_x = max(max(single_times), max_x)
        
        
    if multi_times is not None:
        
        rates = numpy.diff(multi_samples)/numpy.diff(multi_times)
        
        rates[rates<1.0] = 1.0
        
        plt.semilogy(multi_times[1:], rates, color='b', label="multi")
        
        min_y = min(min(rates), min_y)
        
        max_y = max(max(rates), max_y)
        
        min_x = min(min(multi_times), min_x)
        
        max_x = max(max(multi_times), max_x)
        
    plt.legend()
    
    y_range = max_y - min_y
    x_range = max_x - min_x
    
    #plt.ylim(min_y - .05*y_range, max_y + .05*y_range)
    plt.xlim(min_x - .05*x_range, max_x + .05*x_range)
    
    plt.ylabel("Cells Per Second")
    plt.xlabel("Time [s]")
    
    plt.show()
    
plot_abs(single_times, single_samples, multi_times, multi_samples)

plot_rate(single_times, single_samples, multi_times, multi_samples)
  
    