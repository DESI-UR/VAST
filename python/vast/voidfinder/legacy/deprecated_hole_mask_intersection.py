

#
# This strategy was briefly used in VoidFinder to check whether a given hole overlapped
# too much outside of the survey mask in order to discard it.  We refactored this process
# into a monte-carlo method which replaced this strategy.
#





############################################################################
# Calculate the radial value at which we need to check a finished hole
# for overlap with the mask
#
# We're assuming axis-aligned intersection only, so this calculation uses
# the area of a "spherical cap" https://en.wikipedia.org/wiki/Spherical_cap
# We set the area of the spherical cap equal to some fraction p times the
# area of the whole sphere, and if that volume of the sphere is in the mask
# then we discard the current hole.  Instead of calculating the actual 
# volume at any point, instead we calculate the percentage of the radius
# at which that volume is achieved, which is the same for every sphere, then
# we can check the 6 directions from the center of a hole as a proxy for
# an actual 10% volume overlap.  This is a good-enough approximation, since
# the mask is already composed of cubic cells anyway.
#
#
# let l = r - Y, where r is the radius of a sphere, and Y is the distance
# along the radius at the volume we care about
# V_cap = pi*l*l*(3r-l)/3 
# V_sphere = (4/3)*pi*r^3
# let 'p' be the fraction of the volume we care about (say, 10% or 0.1)
#
# pi*(r-Y)*(r-Y)*(3r-r+Y)/3 = p*(4/3)*pi*r^3
#
# algebra
#
# (Y/r)^3 - 3(Y/r) + (2-4p) = 0
#
# Solve for the value (Y/r) given parameter p using the numpy.roots, which
# solves the roots of polynomials in the form ax^n + bx^(n-1) ... + C = 0
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.roots.html
# The root we care about will be in the interval (0,1)
#
# a = 1
# b = 0
# c = -3
# d = 2 - 4*p
#
# DEPRECATED THIS IN FAVOR OF A NEW STRATEGY
# 
############################################################################

coeffs = [1.0, 0.0, -3.0, 2.0 - 4.0*max_hole_mask_overlap]

roots = np.roots(coeffs)

radial_mask_check = None

for root in roots:
    
    if root > 0.0 and root < 1.0:
        
        radial_mask_check = root
        
if radial_mask_check is None:
    
    raise ValueError("Could not calculate appropriate radial check value for input max_hole_mask_overlap "+str(max_hole_mask_overlap))


print("For mask volume check of: ", max_hole_mask_overlap, "Using radial hole value of: ", radial_mask_check)









#---------------------------------------------------------------------------
#
# from voidfinder.find_voids, after the _hole_finder has been run
'''
# Pure python version
#---------------------------------------------------------------------------
coordinates = np.empty((1,3), dtype=np.float64)
temp_coordinates = np.empty((1,3), dtype=np.float64)
mask = mask.astype(np.uint8)

keep_holes = np.ones(x_y_z_r_array.shape[0], dtype=np.bool)

for idx in range(x_y_z_r_array.shape[0]):

    hole_radius = x_y_z_r_array[idx,3]
    
    coordinates[0,0] = x_y_z_r_array[idx,0]
    coordinates[0,1] = x_y_z_r_array[idx,1]
    coordinates[0,2] = x_y_z_r_array[idx,2]

    discard = check_mask_overlap(coordinates,
                       temp_coordinates,
                       radial_mask_check,
                       hole_radius,
                       mask, 
                       mask_resolution,
                       dist_limits[0],
                       dist_limits[1])
    
    if discard:
        
        keep_holes[idx] = False

x_y_z_r_array = x_y_z_r_array[keep_holes]


print("After volume cut, remaining holes: ", x_y_z_r_array.shape[0])
#---------------------------------------------------------------------------
'''



#---------------------------------------------------------------------------
#
# from voidfinder.find_voids, after the _hole_finder has been run

'''
sort_start = time.time()

print('Sorting holes by size', flush=True)

potential_voids_table = Table(x_y_z_r_array, names=('x','y','z','radius'))

potential_voids_table.sort('radius')

potential_voids_table.reverse()

sort_end = time.time()

print('Holes are sorted; Time to sort holes =', sort_end-sort_start, flush=True)
'''



#---------------------------------------------------------------------------
#
# from voidfinder.find_voids, was replaced by 
# voidfinder.volume_cut.check_hole_bounds()
'''
print('Removing holes with at least 10% of their volume outside the mask', flush=True)

mask = mask.astype(np.uint8)


cut_start = time.time()
potential_voids_table = volume_cut(potential_voids_table, 
                                   mask, 
                                   mask_resolution, 
                                   dist_limits)

print("Time to volume-cut holes: ", time.time() - cut_start, flush=True)

print("Num volume-cut holes: ", len(potential_voids_table), flush=True)

potential_voids_table.write(potential_voids_filename, format='ascii.commented_header', overwrite=True)
'''




#---------------------------------------------------------------------------
#
# from voidfinder.find_voids, don't think this code ever actually worked
# to calculate void volumes - combinatorial problem with sphere overlap
# volumes
'''
print('Compute void volumes')

# Initialize void volume array
void_vol = np.zeros(void_count)

nran = 10000

for i in range(void_count):
    nsph = 0
    rad = 4*myvoids_table['radius'][v_index[i]]

    for j in range(nran):
        rand_pos = add_row(np.random.rand(3)*rad, myvoids_table['x','y','z'][v_index[i]]) - 0.5*rad
        
        for k in range(len(myvoids_table)):
            if myvoids_table['flag'][k]:
                # Calculate difference between particle and sphere
                sep = sum(to_vector(subtract_row(rand_pos, myvoids_table['x','y','z'][k])))
                
                if sep < myvoids_table['radius'][k]**2:
                    # This particle lies in at least one sphere
                    nsph += 1
                    break
    
    void_vol[i] = (rad**3)*nsph/nran

'''



#---------------------------------------------------------------------------
#
# from voidfinder.find_voids, assigning field galaxies to the voids can
# easily be done elsewhere or in 'postprocessing'
'''
print('Assign field galaxies to voids')

# Count the number of galaxies in each void
nfield = np.zeros(void_count)

# Add void field to f_coord
f_coord['vID'] = -99

for i in range(nf): # Go through each void galaxy
    for j in range(len(myvoids_table)): # Go through each void
        if np.linalg.norm(to_vector(subtract_row(f_coord[i], myvoids_table['x','y','z'][j]))) < myvoids_table['radius'][j]:
            # Galaxy lives in the void
            nfield[myvoids_table['flag'][j]] += 1

            # Set void ID in f_coord to match void ID
            f_coord['vID'][i] = myvoids_table['flag'][j]

            break

f_coord.write(voidgals_filename, format='ascii.commented_header')
'''





#---------------------------------------------------------------------------
#
# from voidfinder.find_voids, assigning field galaxies to the voids can
# easily be done elsewhere or in 'postprocessing'
'''

# Initialize
void_regions = Table()

void_regions['radius'] = myvoids_table['radius'][v_index]
void_regions['effective_radius'] = (void_vol*0.75/np.pi)**(1./3.)
void_regions['volume'] = void_vol
void_regions['x'] = myvoids_table['x'][v_index]
void_regions['y'] = myvoids_table['y'][v_index]
void_regions['z'] = myvoids_table['z'][v_index]
void_regions['deltap'] = (nfield - N_gal*void_vol/vol)/(N_gal*void_vol/vol)
void_regions['n_gal'] = nfield
void_regions['vol_maxHole'] = (4./3.)*np.pi*myvoids_table['radius'][v_index]**3/void_vol

void_regions.write(out3_filename, format='ascii.commented_header')
'''




#---------------------------------------------------------------------------
#
# from voidfinder.find_voids, older slower way of combining holes into
# voids

'''
combine_start = time.time()

print('Combining holes into unique voids', flush=True)

maximal_spheres_table, myvoids_table = combine_holes(potential_voids_table)

print('Number of unique voids is', len(maximal_spheres_table), flush=True)

combine_end = time.time()

print('Time to combine holes into voids =', combine_end-combine_start, flush=True)
'''




