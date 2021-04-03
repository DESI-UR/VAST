
import numpy as np
#from astropy.table import Table

from .voidfinder_functions import not_in_mask



maskra = 360
maskdec = 180
dec_offset = -90


# Distance limits (units of Mpc/h, taken from voids_sdss.py)
rmin = 0
rmax = 300


################################################################################
#
#   FUNCTION - DETERMINE_VFLAG
#
################################################################################


def determine_vflag(x, y, z, voids, mask, mask_resolution):
    '''
    Determines whether or not a galaxy is a void, wall, edge, or unclassifiable
    galaxy.


    Parameters:
    ===========

    x : float
        x-position of object in units of Mpc/h

    y : float
        y-position of object in units of Mpc/h

    z : float
        z-position of object in units of Mpc/h

    voids : astropy table
        List of holes defining the void regions.  Columns consist of the center 
        of each hole (x,y,z) in units of Mpc/h, the hole radii, and the void 
        identifier for each hole.

    mask : numpy boolean array of shape (n,m)
        True values correspond to ra,dec coordinates which lie within the 
        survey footprint.


    Returns:
    ========

    vflag : integer
        0 = wall galaxy
        1 = void galaxy
        2 = edge galaxy (too close to survey boundary to determine)
        9 = outside survey footprint
    '''


    ############################################################################
    #   INTRO CALCULATIONS, INITIALIZATIONS
    ############################################################################

    # Distance from galaxy to center of all voids
    distance_to_center = np.sqrt((voids['x'] - x)**2 + (voids['y'] - y)**2 + (voids['z'] - z)**2)
    
    # Boolean to find which void surrounds the galaxy, if any
    boolean = distance_to_center < voids['radius']
    
    
    ############################################################################
    #   VOID GALAXIES
    ############################################################################
    
    if any(boolean):
        # The galaxy resides in at least one void
        vflag = 1
        
        
    ############################################################################
    #   WALL GALAXIES
    ############################################################################
        
    else:
        # The galaxy does not live in any voids
        
        ########################################################################
        # Is the galaxy outside the survey boundary?
        ########################################################################

        coord_array = np.array([[x,y,z]])

        # Check to see if the galaxy is within the survey
        if not_in_mask(coord_array, mask, mask_resolution, rmin, rmax):
            # Galaxy is outside the survey mask
            vflag = 9

        else:
            # Galaxy is within the survey mask, but is not within a void
            vflag = 0

            ####################################################################
            # Is the galaxy within 10 Mpc/h of the survey boundary?
            ####################################################################

            # Calculate coordinates that are 10 Mpc/h in each Cartesian 
            # direction of the galaxy
            coord_min = np.array([x,y,z]) - 10
            coord_max = np.array([x,y,z]) + 10

            # Coordinates to check
            x_coords = [coord_min[0], coord_max[0], x, x, x, x]
            y_coords = [y, y, coord_min[1], coord_max[1], y, y]
            z_coords = [z, z, z, z, coord_min[2], coord_max[2]]
            extreme_coords = np.array([x_coords, y_coords, z_coords]).T

            i = 0
            while vflag == 0 and i <= 5:
                # Check to see if any of these are outside the survey
                if not_in_mask(extreme_coords[i].reshape(1,3), mask, mask_resolution, rmin, rmax):
                    # Galaxy is within 10 Mpc/h of the survey edge
                    vflag = 2
                i += 1
        
        
    ############################################################################
    #   FUNCTION OUTPUT
    ############################################################################
    
    return vflag
