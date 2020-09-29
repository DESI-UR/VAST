

import numpy as np
import pickle
from astropy.table import Table

import matplotlib.pyplot as plt


from vast.voidfinder._voidfinder_cython import check_mask_overlap
from vast.voidfinder.volume_cut import volume_cut, check_hole_bounds


################################################################################
#
################################################################################

temp_infile = open("/home/moose/VoidFinder/VoidFinder/python/scripts/SDSS_dr7_mask.pickle", 'rb')
mask, mask_resolution = pickle.load(temp_infile)
temp_infile.close()

mask = mask.astype(np.uint8)

dist_limits = [0.0, 300.3472]

DtoR = np.pi/180.
RtoD = 180./np.pi


plt.imshow(mask)
plt.show()
plt.close()


def correct_mask(mask):
    
    correct_idxs = []
    
    for idx in range(mask.shape[0]):
        for jdx in range(mask.shape[1]):
            
            if idx < 1 or jdx < 1 or idx > mask.shape[0] - 2 or jdx > mask.shape[1] - 2:
                continue
            
            curr_val = mask[idx, jdx]
            
            neigh1 = mask[idx-1,jdx]
            neigh2 = mask[idx+1,jdx]
            neigh3 = mask[idx,jdx-1]
            neigh4 = mask[idx,jdx+1]
            
            if curr_val == 0 and neigh1+neigh2+neigh3+neigh4 >= 3:
                correct_idxs.append((idx,jdx))
                
    for (idx, jdx) in correct_idxs:
        
        mask[idx,jdx] = 1


correct_mask(mask)

plt.imshow(mask)
plt.show()
plt.close()



################################################################################
# calculate radial dist
################################################################################
max_hole_mask_overlap = 0.1

coeffs = [1.0, 0.0, -3.0, 2.0 - 4.0*max_hole_mask_overlap]
    
roots = np.roots(coeffs)

radial_mask_check = None

for root in roots:
    
    if root > 0.0 and root < 1.0:
        
        radial_mask_check = root
        
if radial_mask_check is None:
    
    raise ValueError("Could not calculate appropriate radial check value for input max_hole_mask_overlap "+str(max_hole_mask_overlap))

print("For mask volume check of: ", max_hole_mask_overlap, "Using radial hole value of: ", radial_mask_check)


################################################################################
# New method
################################################################################

def new_method(test_position, radial_mask_check, test_radius, mask, mask_resolution, dist_limits):

    temp_coordinates = np.empty((1,3), dtype=np.float64)
    
    discard = check_mask_overlap(test_position,
                                 temp_coordinates,
                                 radial_mask_check,
                                 test_radius,
                                 mask, 
                                 mask_resolution,
                                 dist_limits[0], 
                                 dist_limits[1])
    
    if discard:
        too_much_overlap = True
    else:
        too_much_overlap = False
        
    return too_much_overlap


################################################################################
# Old method
################################################################################
def old_method(test_position, radial_mask_check, test_radius, mask, mask_resolution, dist_limits):

    temp = np.empty((1,4), dtype=np.float64)
    temp[0,0] = test_position[0,0]
    temp[0,1] = test_position[0,1]
    temp[0,2] = test_position[0,2]
    temp[0,3] = test_radius
    
    table_test_position = Table(temp, names=('x','y','z','radius'))
    
    results = volume_cut(table_test_position, mask, mask_resolution, dist_limits)
    
    if len(results) == 0:
        too_much_overlap = True
    else:
        too_much_overlap = False
        
    return too_much_overlap





################################################################################
# new method 2
################################################################################
def new_method_2(test_position, radial_mask_check, test_radius, mask, mask_resolution, dist_limits):

    temp = np.empty((1,4), dtype=np.float64)
    temp[0,0] = test_position[0,0]
    temp[0,1] = test_position[0,1]
    temp[0,2] = test_position[0,2]
    temp[0,3] = test_radius
    
    
    
    
    valid_idx = check_hole_bounds(temp, 
                                  mask, 
                                  mask_resolution, 
                                  dist_limits,
                                  cut_pct=0.1,
                                  pts_per_unit_volume=3,
                                  num_surf_pts=20,
                                  num_cpus=1)
    
    
    
    if valid_idx[0] == 0:
        too_much_overlap = True
    else:
        too_much_overlap = False
        
    return too_much_overlap


################################################################################
# Helper
################################################################################
def ra_dec_to_xyz(ra, dec, radius, h=1.0):
    """
    Description
    ===========
    
    Convert galaxy coordinates from ra-dec-redshift space into xyz space.
    
    
    Parameters
    ==========
        
    h : float
        Fractional value of Hubble's constant.  Default value is 1 (where 
        H0 = 100h).
        
        
    Returns
    =======
    
    coords_xyz : numpy.ndarray of shape (N,3)
        values of the galaxies in xyz space
    """
    
    
    ################################################################################
    # Convert from ra-dec-radius space to xyz space
    ################################################################################
    
    ra_radian = ra*DtoR
    
    dec_radian = dec*DtoR
    
    x = radius*np.cos(ra_radian)*np.cos(dec_radian)
    
    y = radius*np.sin(ra_radian)*np.cos(dec_radian)
    
    z = radius*np.sin(dec_radian)
    
    num_gal = x.shape[0]
    
    coords_xyz = np.concatenate((x.reshape(num_gal,1),
                                 y.reshape(num_gal,1),
                                 z.reshape(num_gal,1)), axis=1)
    
    return coords_xyz


n_test = 100

#ra = np.linspace(150.0, 170.0, n_test)
ra = np.linspace(107.0, 263.0, n_test)
dec = np.linspace(-10.0, 72.0, n_test)


xx, yy = np.meshgrid(ra, dec)

print(xx.shape)
print(yy.shape)

#exit()
ra = xx.ravel()
dec = yy.ravel()
radial_pos = np.array([200.0]*n_test*n_test, dtype=np.float64)

test_positions = ra_dec_to_xyz(ra, dec, radial_pos)

test_radii = np.array([20.0]*n_test*n_test, dtype=np.float64)




plt.imshow(mask)
plt.scatter(dec+90.0, ra)
plt.show()
plt.close()


new_results = []
old_results = []

for idx, (test_pos, test_radius) in enumerate(zip(test_positions, test_radii)):
    
    if idx%1000 == 0:
        print(idx)
    
    
    new_too_much_overlap = new_method(test_pos.reshape(1,3), 
                                       radial_mask_check, 
                                       test_radius, 
                                       mask, 
                                       mask_resolution, 
                                       dist_limits)
    
    old_too_much_overlap = old_method(test_pos.reshape(1,3), 
                                       radial_mask_check, 
                                       test_radius, 
                                       mask, 
                                       mask_resolution, 
                                       dist_limits)
    
    
    new_results.append(new_too_much_overlap)
    old_results.append(old_too_much_overlap)
    
#print(new_results)
#print(old_results)

#plot new results


input = np.concatenate((test_positions, test_radii.reshape(n_test*n_test,1)), axis=1)

new_2_results, monte_index = check_hole_bounds(input, 
                                  mask, 
                                  mask_resolution, 
                                  dist_limits,
                                  cut_pct=0.1,
                                  pts_per_unit_volume=.01,
                                  num_surf_pts=100,
                                  num_cpus=1)


outfile = open("temp.pickle", 'wb')
pickle.dump((new_2_results, monte_index), outfile)
outfile.close()


'''
infile = open("temp.pickle", 'rb')
new_2_results, monte_index = pickle.load(infile)
infile.close()
'''
print("Monte: ", np.sum(monte_index))

new_results = np.logical_not(new_2_results)



#new_results = np.array(new_results)
old_results = np.array(old_results)


################################################################################
#
################################################################################

keep_idx = np.logical_not(new_results)

not_monte_idx = np.logical_not(monte_index)

print(np.sum(keep_idx))

pop1 = np.logical_and(new_results, not_monte_idx)
pop2 = np.logical_and(new_results, monte_index)
pop3 = np.logical_and(keep_idx, not_monte_idx)
pop4 = np.logical_and(keep_idx, monte_index)
keep_idx2 = np.logical_and(keep_idx, monte_index)
keep_idx3 = np.logical_and(keep_idx, np.logical_not(monte_index))


print(np.sum(keep_idx2))
print(np.sum(keep_idx3))



plt.imshow(mask.T)
plt.title("New Results, blue=too much overlap")
plt.scatter(ra[pop1], dec[pop1]+90.0, color='b')
plt.scatter(ra[pop2], dec[pop2]+90.0, color='g')
plt.scatter(ra[pop3], dec[pop3]+90.0, color='r')
plt.scatter(ra[pop4], dec[pop4]+90.0, color='k')
#plt.scatter(ra[monte_index], dec[monte_index]+90.0, color='g')
#plt.scatter(ra[keep_idx3], dec[keep_idx3]+90.0, color='r')
plt.show()
plt.close()






################################################################################
#
################################################################################

keep_idx = np.logical_not(old_results)

plt.imshow(mask.T)
plt.title("Old Results, blue=too much overlap")
plt.scatter(ra[old_results], dec[old_results]+90.0, color='b')
plt.scatter(ra[keep_idx], dec[keep_idx]+90.0, color='r')
plt.show()
plt.close()






################################################################################
#
################################################################################
mismatches = new_results != old_results

if np.any(mismatches):
    print(np.sum(mismatches), "Mismatches")
    
    idx = np.where(mismatches)[0]
    
    sub_idx = np.where(np.logical_and(new_results, mismatches))[0]
    
    print(new_results[idx])
    print(old_results[idx])
    
    print(ra[idx])
    print(dec[idx])
    print(test_positions[idx])
    
    plt.imshow(mask)
    plt.scatter(dec[idx]+90.0, ra[idx], color='b')
    plt.scatter(dec[sub_idx]+90.0, ra[sub_idx], color='r')
    plt.show()
    plt.close()
    
    
else:
    print("All match")
















