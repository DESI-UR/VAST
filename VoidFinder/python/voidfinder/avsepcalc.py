#function to calculate nearest neighbor separation in Python VoidFinder

from sklearn import neighbors
import numpy as np
import time

from .table_functions import to_array

def av_sep_calc(GALTABLEXYZ):
	"""
	Note: Sklearn KDTree implementation outperformed scipy KDTree by a ridiculous
	      factor on sim dataset of 37 million galaxies
	"""
	
	avsep_start = time.time()

	gal_array = to_array(GALTABLEXYZ)
	
	galaxy_tree = neighbors.KDTree(gal_array)

	distances, indices = galaxy_tree.query(gal_array,k=4)
	
	all_3rd_distances = distances[:,3]

	avg = np.mean(all_3rd_distances)

	sd = np.std(all_3rd_distances)

	dist_lim = avg + 1.5*(sd)

	return dist_lim, avg, sd, all_3rd_distances
