'''code to write a preprocessing function which eliminates
magnitudes that are fainter than -20 (magnitudes are backwards so
bright ones are more negative than -20, that's what we want to keep)'''

'''code to write a preprocessing function which eliminates field
galaxies with less than 3 'nearby' neighbors'''

import numpy as np

from astropy.table import Table


def mag_cut(T,mag_limit):
	
	bool_mag = T['rabsmag'] < mag_limit #bool array which is true for bright galaxies
	
	bright_data = T[bool_data] #create new table which only has bright galaxies
	
	return bright_data


def field_data_cut(T,dists,l):
	
	bool_data = dists < l
	
	wall_data = T[bool_data]
	
	field_data = T[np.logical_not(bool_data)]
	
	
	return field_data, wall_data

	
