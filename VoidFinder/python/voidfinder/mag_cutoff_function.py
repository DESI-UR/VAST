'''code to write a preprocessing function which eliminates
magnitudes that are fainter than -20 (magnitudes are backwards so
bright ones are more negative than -20, that's what we want to keep)'''

'''code to write a preprocessing function which eliminates field
galaxies with less than 3 'nearby' neighbors'''

import numpy as np

from astropy.table import Table


def mag_cut(T,mag_limit):
	
	bool_mag = T['rabsmag'] < mag_limit #bool array which is true for bright galaxies
	
	bright_gal = T[bool_mag] #create new table which only has bright galaxies
	
	return bright_gal


def field_gal_cut(T,dists,l):
	
	bool_gal = dists < l
	
	wall_gals = T[bool_gal]
	
	field_gals = T[np.logical_not(bool_gal)]
	
	
	return field_gals, wall_gals

	
