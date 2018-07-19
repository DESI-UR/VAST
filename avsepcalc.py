#function to calculate nearest neighbor separation in Python VoidFinder


def av_sep_calc(GALTABLEXYZ):
	from scipy import spatial
	import numpy as np
	gal_array = np.array([GALTABLEXYZ['x'], GALTABLEXYZ['y'], GALTABLEXYZ['z']]).T
	galaxy_tree = spatial.KDTree(gal_array)
	#all_3rd_distances = []
	distances, indices = galaxy_tree.query(gal_array,k=4)
	all_3rd_distances = distances[:,3]
	avg = np.mean(all_3rd_distances)
	sd = np.std(all_3rd_distances)
	dist_lim = avg + 1.5*(sd)
	return dist_lim, avg, sd, all_3rd_distances
