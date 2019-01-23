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
	'''#convert RA,DEC to x,y,z first
	RA_rad = np.radians(T['ra'])
	DEC_rad = np.radians(T['dec'])
	T_x = T['Rgal']*np.cos(DEC_rad)*np.cos(RA_rad)
	T_y = T['Rgal']*np.cos(DEC_rad)*np.sin(RA_rad)
	T_z = T['Rgal']*np.sin(DEC_rad)
	T_cart = Table([T_x, T_y, T_z], names=['x', 'y', 'z'])
	field_gal_table = Table(np.zeros(3),names=('x','y','z'))
	for i,gal in enumerate(T_cart):
		xmin = gal['x']-dist_lim
		xmax = gal['x']+dist_lim
		bool_x = np.logical_and(T_cart['x']>=xmin,T_cart['x']<=xmax)
		subsample = T_cart[bool_x]
		ymin = gal['y']-dist_lim
		ymax = gal['y']+dist_lim
		bool_y = np.logical_and(subsample['y']>=ymin,subsample['y']<=ymax)
		subsubsample = subsample[bool_y]
		zmin = gal['z']-dist_lim
		zmax = gal['z']+dist_lim
		bool_z = np.logical_and(subsubsample['z']>=zmin,subsubsample['z']<=zmax)
		subsubsubsample = subsubsample[bool_z]
		num_gal = len(subsubsubsample)
		if num_gal > 3:
			dist = np.sqrt(((T_cart['x']-gal['x'])**2)+((T_cart['y']-gal['y'])**2)+((T_cart['z']-gal['z'])**2))
			dist.sort()
			if dist[3] >= dist_lim:
				field_gal_table.add_row(T_cart[:][i])
				T.remove_row(i)
				T_cart.remove_row(i)
		else:
			field_gal_table.add_row(T_cart[:][i])
			T.remove_row(i)
			T_cart.remove_row(i)
	field_gal_table.remove_row(0)
	return field_gal_table, T_cart'''
	return field_gals, wall_gals

	
