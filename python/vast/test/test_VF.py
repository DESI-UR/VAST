# -*- coding: utf-8 -*-
import unittest

import os
import numpy as np
from sklearn import neighbors
from astropy.table import Table, setdiff, vstack
from astropy.io import fits

from vast.voidfinder.constants import c
from vast.voidfinder import find_voids, filter_galaxies
from vast.voidfinder.table_functions import to_array
from vast.voidfinder.multizmask import generate_mask
from vast.voidfinder.preprocessing import file_preprocess
from vast.voidfinder._voidfinder_cython import grow_spheres
from vast.voidfinder._voidfinder_cython_find_next import SpatialMap, \
                                                         Cell_ID_Memory, \
                                                         GalaxyMapCustomDict, \
                                                         HoleGridCustomDict, \
                                                         NeighborMemory, \
                                                         MaskChecker, \
                                                         find_next_prime, \
                                                         _query_first


from vast.voidfinder._hole_combine_cython import cap_height, spherical_cap_volume




from vast.voidfinder.viz import VoidRender, \
                                format_galaxy_data, \
                                load_void_data






class TestVoidFinder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Set up some global variables available to all test cases.
        '''
        TestVoidFinder.wall = None
        TestVoidFinder.dist_limits = None
        TestVoidFinder.mask = None
        TestVoidFinder.mask_resolution = None
        TestVoidFinder.grid_shape = None
        TestVoidFinder.galaxy_map = None

    def setUp(self):
        '''
        Set up a dummy survey that can be used to test I/O, preprocessing, and 
        void-finding.
        '''
        self.ra_range = np.arange(10, 30, 0.5)
        self.dec_range = np.arange(-10, 10, 0.5)
        self.redshift_range = np.arange(0.0005, 0.011, 0.0001) # 0.0005

        RA, DEC, REDSHIFT = np.meshgrid(self.ra_range, 
                                        self.dec_range, 
                                        self.redshift_range)

        self.galaxies_table = Table()
        self.galaxies_table['ra'] = np.ravel(RA)
        self.galaxies_table['dec'] = np.ravel(DEC)
        self.galaxies_table['redshift'] = np.ravel(REDSHIFT)
        self.galaxies_table.add_row([20., 0., 0.])

        # Shuffle the table (so that the KDtree does not die)
        self.rng = np.random.default_rng()
        self.galaxies_shuffled = Table(self.rng.permutation(self.galaxies_table))
        self.galaxies_shuffled['Rgal'] = c*self.galaxies_shuffled['redshift']/100.
        N_galaxies = len(self.galaxies_shuffled)

        # All galaxies will be brighter than the magnitude limit, so that none
        # of them are removed
        self.galaxies_shuffled['rabsmag'] = -21.0

        # Create test_galaxies file
        self.galaxies_filename = 'test_galaxies.txt'
        self.galaxies_shuffled.write(self.galaxies_filename,
                                     format='ascii.commented_header',
                                     overwrite=True)

        # Convert galaxy coordinates to Cartesian
        self.gal = np.zeros((N_galaxies,3))
        self.gal[:,0] = self.galaxies_shuffled['Rgal']*np.cos(self.galaxies_shuffled['ra']*np.pi/180.)*np.cos(self.galaxies_shuffled['dec']*np.pi/180.)
        self.gal[:,1] = self.galaxies_shuffled['Rgal']*np.sin(self.galaxies_shuffled['ra']*np.pi/180.)*np.cos(self.galaxies_shuffled['dec']*np.pi/180.)
        self.gal[:,2] = self.galaxies_shuffled['Rgal']*np.sin(self.galaxies_shuffled['dec']*np.pi/180.)

        # Calculate coordinate extremes
        self.coords_min = np.min(self.gal, axis=0)
        self.coords_max = np.max(self.gal, axis=0)

        # Minimum maximal sphere radius
        self.min_maximal_radius = 1. # Mpc/h

        # Define maximal spheres
        self.maximals = Table()
        self.maximals['x'] = [25., 10.]
        self.maximals['y'] = [8., 3.]
        self.maximals['z'] = [0., -1.]
        self.maximals['r'] = [2.5, 1.5]
        self.maximals['void'] = [0, 1]

        # Define holes
        holes = Table()
        holes['x'] = [24., 10.5]
        holes['y'] = [7.9, 3.2]
        holes['z'] = [0.1, -0.5]
        holes['r'] = [2., 0.5]
        holes['void'] = [0, 1]
        self.holes = vstack([holes, self.maximals])

        # Remove points which fall inside holes
        remove_boolean = np.zeros(len(self.gal), dtype=bool)
        for i in range(len(self.holes)):
            d = (self.holes['x'][i] - self.gal[:,0])**2 + (self.holes['y'][i] - self.gal[:,1])**2 + (self.holes['z'][i] - self.gal[:,2])**2
            remove_boolean = remove_boolean | (d < self.holes['r'][i]**2)
        TestVoidFinder.wall = self.gal[~remove_boolean]
        TestVoidFinder.field = self.gal[remove_boolean]

        # Build KDTree of wall galaxies
        self.wall_tree = neighbors.KDTree(TestVoidFinder.wall)

        # Define resource directory
        self.RESOURCE_DIR = "/dev/shm"
        if not os.path.isdir(self.RESOURCE_DIR):
            self.RESOURCE_DIR = "/tmp"

        # Define mask mode
        self.mask_mode = 0

        # Define hole grid parameters
        self.hole_grid_edge_length = 1.0

        self.cell_ID_mem = Cell_ID_Memory(10)




    def test_1_file_preprocess(self):
        """
        Take a galaxy data file and return a data table, compute the redshift 
        range in comoving coordinates
        """
        f_galaxy_table, f_dist_limits = \
            file_preprocess(self.galaxies_filename, 'test_', '', '', dist_metric='redshift')

        # Check the galaxy table
        self.assertEqual(len(setdiff(f_galaxy_table, self.galaxies_shuffled)), 0)

        # Check the distance limits
        TestVoidFinder.dist_limits = np.zeros(2)
        TestVoidFinder.dist_limits[1] = c*self.redshift_range[-1]/100.
        self.assertTrue(np.isclose(f_dist_limits, TestVoidFinder.dist_limits).all())

        # file_preprocess now outputs a data table to a fits file. This could be a source of further checks 



    
    def test_2_generate_mask(self):
        """
        Take a table of galaxy coordinates and maximum redshift and return a 
        boolean mask + resolution
        """
        f_mask, f_mask_resolution = generate_mask(self.galaxies_shuffled, 
                                                  self.redshift_range[-1],
                                                  'test_','',
                                                  dist_metric='redshift', 
                                                  min_maximal_radius=self.min_maximal_radius)

        # Check the mask
        TestVoidFinder.mask = np.zeros((360,180), dtype=bool)
        for i in range(int(self.ra_range[0]), int(self.ra_range[-1]+1)):
            for j in range(int(self.dec_range[0] + 90), int(self.dec_range[-1] + 90)+1):
                TestVoidFinder.mask[i, j] = True
        self.assertTrue((f_mask == TestVoidFinder.mask).all())

        # Check the mask resolution
        TestVoidFinder.mask_resolution = 1
        self.assertTrue(np.isclose(f_mask_resolution, TestVoidFinder.mask_resolution))

        # generate_mask now saves its output to a fits file. This could be a source of further checks
    



    
    def test_3_filter_galaxies(self):
        """
        Filter galaxies.
        
        Update 3/14/2022 - filter_galaxies is no longer returning grid_shape and
        coords_min parameters - instead they are calculated inside the main body 
        of find_voids() for consistency among the 3 mask_mode types
        
        """
        # Take a table of galaxy coordinates, the name of the survey, and the
        # output directory and returns astropy tables of the Cartesian
        # coordinates of the wall and field galaxies as well as the shape 
        # of the grid on which the galaxies will be placed and the coordinates
        # of the lower left corner of the grid.
        
        f_wall, f_field = filter_galaxies(self.galaxies_shuffled, 
                                          'test_', 
                                          '', 
                                          dist_metric='redshift', 
                                          )

        # Check the wall galaxy coordinates
        gal_tree = neighbors.KDTree(self.gal)
        distances, indices = gal_tree.query(self.gal, k=4)
        dist3 = distances[:,3]
        wall = self.gal[dist3 < (np.mean(dist3) + 1.5*np.std(dist3))]
        self.assertTrue(np.isclose(f_wall, wall).all())

        # Check the field galaxy coordinates
        field = self.gal[dist3 >= (np.mean(dist3) + 1.5*np.std(dist3))]
        self.assertTrue(np.isclose(f_field, field).all())

        #These tests are deprecated as the 3/14/2022 update
        #leaving the commented versions here in case we need to roll back
        # Check the grid shape
        #n_cells = (np.max(self.gal, axis=0) - np.min(self.gal, axis=0))
        #TestVoidFinder.grid_shape = tuple(np.ceil(n_cells).astype(int))
        #self.assertEqual(f_grid_shape, TestVoidFinder.grid_shape)

        # Check the minimum coordinates
        #self.assertTrue(np.isclose(f_min, np.min(self.gal, axis=0)).all())




    def test_4_query_first(self):
        '''
        Find the nearest galaxy to a point
        '''

        ########################################################################
        # Set up the GalaxyMap
        #-----------------------------------------------------------------------
        box = self.coords_max - self.coords_min

        galaxy_map_grid_edge_length = 15.

        ngrid_galaxymap = box/galaxy_map_grid_edge_length

        galaxy_map_grid_shape = tuple(np.ceil(ngrid_galaxymap).astype(int))
        #print("galaxy_map_grid_shape: ", galaxy_map_grid_shape)

        mesh_indices = ((TestVoidFinder.wall - self.coords_min)/galaxy_map_grid_edge_length).astype(np.int64)
    
        pre_galaxy_map = {}

        for idx in range(mesh_indices.shape[0]):

            bin_ID_pqr = tuple(mesh_indices[idx])
            
            if bin_ID_pqr not in pre_galaxy_map:
                
                pre_galaxy_map[bin_ID_pqr] = []
            
            pre_galaxy_map[bin_ID_pqr].append(idx)
            
        del mesh_indices

        num_in_galaxy_map = len(pre_galaxy_map)

        galaxy_search_cell_dict = GalaxyMapCustomDict(galaxy_map_grid_shape,
                                                      self.RESOURCE_DIR)

        offset = 0

        galaxy_map_list = []

        for key in pre_galaxy_map:
            
            indices = np.array(pre_galaxy_map[key], dtype=np.int64)
            
            num_elements = indices.shape[0]
            
            galaxy_map_list.append(indices)
            
            galaxy_search_cell_dict.setitem(*key, offset, num_elements)
            
            offset += num_elements

        galaxy_map_array = np.concatenate(galaxy_map_list)

        del galaxy_map_list

        num_galaxy_map_elements = len(galaxy_search_cell_dict)

        TestVoidFinder.galaxy_map = SpatialMap(self.RESOURCE_DIR,
                                               self.mask_mode,
                                               TestVoidFinder.wall,
                                               self.hole_grid_edge_length,
                                               self.coords_min, 
                                               galaxy_map_grid_edge_length,
                                               galaxy_search_cell_dict,
                                               galaxy_map_array)
        ########################################################################


        ########################################################################
        # Test query_first against a KDTree
        #-----------------------------------------------------------------------
        

        tree_results = []
        vf_results = []

        #Get some random points in the fake galaxy survey to check against
        check_points = np.zeros((100,3))
        check_points_ra = self.rng.uniform(np.min(self.ra_range), 
                                           np.max(self.ra_range), 
                                           check_points.shape[0])
        check_points_dec = self.rng.uniform(np.min(self.dec_range), 
                                            np.max(self.dec_range), 
                                            check_points.shape[0])
        check_points_r = 0.01*c*self.rng.uniform(np.min(self.redshift_range), 
                                                 np.max(self.redshift_range), 
                                                 check_points.shape[0])
        check_points[:,0] = check_points_r*np.cos(check_points_ra*np.pi/180)*np.cos(check_points_dec*np.pi/180)
        check_points[:,1] = check_points_r*np.sin(check_points_ra*np.pi/180)*np.cos(check_points_dec*np.pi/180)
        check_points[:,2] = check_points_r*np.sin(check_points_dec*np.pi/180)

        for idx in range(check_points.shape[0]):
        
            curr_point = check_points[idx:idx+1, :]
            
            tree_dist, tree_idx = self.wall_tree.query(curr_point, 1)

            tree_results.append(tree_idx[0][0])
            
            distidxpair = _query_first(TestVoidFinder.galaxy_map.reference_point_ijk,
                                       TestVoidFinder.galaxy_map.grid_origin,
                                       TestVoidFinder.galaxy_map.dl,
                                       TestVoidFinder.galaxy_map.shell_boundaries_xyz,
                                       TestVoidFinder.galaxy_map.cell_center_xyz,
                                       TestVoidFinder.galaxy_map,
                                       self.cell_ID_mem,
                                       curr_point[0,:].astype(np.float64)
                                       )
            

            vf_idx = distidxpair['idx']
            vf_dist = distidxpair['dist']

            vf_results.append(vf_idx)
            '''
            if tree_idx[0][0] != vf_idx:
                print("KDTree:", tree_dist[0][0], tree_idx[0][0], TestVoidFinder.wall[tree_idx[0][0]])
                print("_query_first:", vf_dist, vf_idx, TestVoidFinder.wall[vf_idx])
            '''

        tree_results = np.array(tree_results)
        vf_results = np.array(vf_results)

        self.assertTrue((tree_results == vf_results).all())

        


    """
    
    def test_5_main_algorithm(self):
        '''
        Grow some holes
        '''

        neighbor_mem = NeighborMemory(50)

        # Number of points to test
        N_test = 10

        # Convert hole centers to numpy arrays
        hole_centers = to_array(self.holes)

        # Generate array of points from which to grow spheres
        #hole_cells = (hole_centers - self.galaxy_map.coord_min)/self.hole_grid_edge_length
        n_cells = (self.coords_max - self.galaxy_map.coord_min)/self.hole_grid_edge_length
        i_j_k_array = self.rng.integers([0, 0, 0], 
                                        high=n_cells.astype(int), 
                                        size=(N_test,3), 
                                        dtype=np.int64)

        # Define the mask_checker object
        mask_checker = MaskChecker(self.mask_mode,
                                   survey_mask_ra_dec=TestVoidFinder.mask,
                                   n=TestVoidFinder.mask_resolution,
                                   rmin=TestVoidFinder.dist_limits[0],
                                   rmax=TestVoidFinder.dist_limits[1],
                                   )

        # Initialize the return_arrays
        f_return_array = np.empty((N_test,4), dtype=np.float64)
        return_array = np.empty((N_test,4), dtype=np.float64)

        # Run main_algorithm
        main_algorithm(i_j_k_array, 
                       TestVoidFinder.galaxy_map, 
                       self.hole_grid_edge_length, 
                       self.hole_center_iter_dist, 
                       TestVoidFinder.galaxy_map.coord_min, 
                       mask_checker, 
                       f_return_array, 
                       self.cell_ID_mem, 
                       neighbor_mem, 
                       0)

        # Find first galaxy for each cell center
        cell_centers = (i_j_k_array + 0.5)*self.hole_grid_edge_length + self.galaxy_map.coord_min
        r1, idx1 = self.wall_tree.query(cell_centers, 1)
        idx1 = idx1[0]

        # Calculate unit vector (pointing from initial center to first galaxy)
        A_minusCenter = TestVoidFinder.wall[idx1] - cell_centers
        unit = A_minusCenter/r1

        for i in range(N_test):

            galaxies_to_search = np.ones(TestVoidFinder.wall.shape[0], 
                                         dtype=bool)

            galaxies_to_search[idx1[i]] = False

            ####################################################################
            # Find second galaxy
            #-------------------------------------------------------------------
            candidate_minusA = TestVoidFinder.wall[galaxies_to_search] - TestVoidFinder.wall[idx1[i]]
            candidate_minusCenter = TestVoidFinder.wall[galaxies_to_search] - cell_centers[i]

            bottom = 2*np.dot(-candidate_minusA, unit[i])
            top = np.linalg.norm(candidate_minusA, axis=1)

            x = top**2/bottom

            # Galaxy #2 is that with the smallest positive value of x
            idx2 = np.where(x > 0, x, np.nan).nanargmin()
            x_min = x[idx2]
            ####################################################################


            ####################################################################
            # Move center based on location of second galaxy
            #-------------------------------------------------------------------
            new_center = cell_centers[i] + x_min*unit[i]

            # Check that the new center is still within the mask
            if mask_checker.not_in_mask(new_center):
                return_array[i,:] = np.nan
                continue
            else:
                if idx2 >= idx1[i]:
                    idx2 += 1
                
                galaxies_to_search[idx2] = False

                AB_midpoint = 0.5*(TestVoidFinder.wall[idx1[i]] + TestVoidFinder.wall[idx2])
                unit = (new_center - AB_midpoint)/np.linalg.norm(new_center - AB_midpoint)

                B_minusCenter = TestVoidFinder.wall[idx2] - new_center
            ####################################################################


            ####################################################################
            # Find third galaxy
            #-------------------------------------------------------------------
            candidate_minusA = TestVoidFinder.wall[galaxies_to_search] - TestVoidFinder.wall[idx1[i]]
            candidate_minusCenter = TestVoidFinder.wall[galaxies_to_search] - new_center

            bottom = 2*np.dot(candidate_minusA, unit)
            top = np.linalg.norm(candidate_minusCenter, axis=1)**2 - np.linalg.norm(B_minusCenter, axis=1)**2

            x = top/bottom

            # Galaxy #3 is that with the smallest positive value of x
            idx3 = np.where(x > 0, x, np.nan).nanargmin()
            x_min = x[idx3]
            ####################################################################


            ####################################################################
            # Move center based on location of third galaxy
            #-------------------------------------------------------------------
            new_center = new_center + x_min*unit
            B_minusCenter = TestVoidFinder.wall[idx2] - new_center

            # Check that the new center is still within the mask
            if mask_checker.not_in_mask(new_center):
                return_array[i,:] = np.nan
                continue
            else:
                if idx3 >= idx1[i]:
                    idx3 += 1
                if idx3 >= idx2:
                    idx3 += 1

                galaxies_to_search[idx3] = False

                AB = TestVoidFinder.wall[idx1[i]] - TestVoidFinder.wall[idx2]
                BC = TestVoidFinder.wall[idx2] - TestVoidFinder.wall[idx3]

                ABcrossBC = np.cross(AB, BC)
                unit = ABcrossBC/np.norm(ABcrossBC)

                center_minusA = new_center - TestVoidFinder.wall[idx1[i]]
                if np.dot(center_minusA, unit) < 0:
                    unit *= -1

                check_both = False
                if np.dot(center_minusA, unit) == 0:
                    check_both = True
            ####################################################################


            ####################################################################
            # Find fourth galaxy
            #-------------------------------------------------------------------
            candidate_minusA = TestVoidFinder.wall[galaxies_to_search] - TestVoidFinder.wall[idx1[i]]
            candidate_minusCenter = TestVoidFinder.wall[galaxies_to_search] - new_center

            bottom = 2*np.dot(candidate_minusA, unit)
            top = np.linalg.norm(candidate_minusCenter, axis=1)**2 - np.linalg.norm(B_minusCenter, axis=1)**2

            x = top/bottom

            # Galaxy #4 is that with the smallest positive value of x
            idx4 = np.where(x > 0, x, np.nan).nanargmin()
            x_min = x[idx4]

            if check_both:
                # We need to flip the unit vector and check the other direction
                bottom = 2*np.dot(candidate_minusA, -unit)

                x = top/bottom

                idx4b = np.where(x > 0, x, np.nan).nanargmin()
                x_min_4b = x[idx4b]

                closer_4b = x_min_4b < x_min
            ####################################################################


            ####################################################################
            # Calculate the final center
            #-------------------------------------------------------------------
            final_center = new_center + x_min*unit
            if check_both:
                final_center_4b = new_center - x_min_4b*unit

            # Check that the center is within the mask
            if check_both and not mask_checker.not_in_mask(final_center) and not closer_4b:
                return_array[i,0] = final_center[0]
                return_array[i,1] = final_center[1]
                return_array[i,2] = final_center[2]
                return_array[i,3] = np.norm(final_center - TestVoidFinder.wall[idx1[i]])
            elif check_both and not mask_checker.not_in_mask(final_center_4b):
                return_array[i,0] = final_center_4b[0]
                return_array[i,1] = final_center_4b[1]
                return_array[i,2] = final_center_4b[2]
                return_array[i,3] = np.norm(final_center_4b - TestVoidFinder.wall[idx1[i]])
            elif not mask_checker.not_in_mask(final_center):
                return_array[i,0] = final_center[0]
                return_array[i,1] = final_center[1]
                return_array[i,2] = final_center[2]
                return_array[i,3] = np.norm(final_center - TestVoidFinder.wall[idx1[i]])
            else:
                return_array[i,:] = np.nan
            ####################################################################

            break

        #self.assertTrue(np.isclose(f_return_array, return_array).all())


    """




    



    
    def test_6_find_voids(self):
        """
        Identify maximal spheres and holes in the galaxy distribution
        """
        
        coords_min = np.min(np.concatenate([TestVoidFinder.wall, TestVoidFinder.field]), axis=0)
        

        find_voids(TestVoidFinder.wall, 
                   'test_', '',
                   grid_origin=coords_min,
                   mask=TestVoidFinder.mask, 
                   mask_resolution=1,
                   dist_limits=TestVoidFinder.dist_limits,
                   hole_grid_edge_length=self.hole_grid_edge_length,
                   min_maximal_radius=self.min_maximal_radius, 
                   num_cpus=1, 
                   pts_per_unit_volume=0.01, # 5
                   )


        '''
        holes_xyz, holes_radii, holes_flags, field_galaxy_data, wall_galaxy_data = load_void_data('test_VoidFinder_Output.fits')
        #holes_xyz, holes_radii, holes_flags = load_void_data('test_VoidFinder_Output.fits')
    
        viz = VoidRender(holes_xyz,
                     holes_radii,
                     holes_flags,
                     galaxy_xyz=TestVoidFinder.field,
                     wall_galaxy_xyz=TestVoidFinder.wall,
                     wall_distance=None,
                     galaxy_display_radius=10.0,
                     remove_void_intersects=1,
                     SPHERE_TRIANGULARIZATION_DEPTH=2
                     )
    
        viz.run()           
        '''

        #Load voids
        with fits.open('test_VoidFinder_Output.fits') as file:
            f_maximals = Table(file['MAXIMALS'].data)
            f_holes = Table(file['HOLES'].data)
        
        # Check maximal spheres
        #maximals_truth = Table.read('python/vast/voidfinder/tests/test_galaxies_redshift_maximal_truth.txt', 
        #                            format='ascii.commented_header')
        
        
        #For the maximals, make sure we have found the same number of
        #maximal spheres
        num_truth_rows = len(self.maximals)
        num_test_rows = len(f_maximals)
        
        self.assertTrue(num_truth_rows == num_test_rows)
        
        
        print("Truth Maximals data: ")
        print(self.maximals)
        print("VoidFinder Detected Maximals: ")
        f_maximals.pprint_all()
        #print("Old VoidFinder Detected Maximals: ")
        #maximals_truth.pprint_all()
        
        #for name in f_maximals.dtype.names:
        #    if not np.allclose(f_maximals[name], maximals_truth[name]):
        #        print(f_maximals[name])
        #        print(maximals_truth[name])
        
        truth_maximal_radii = np.array(self.maximals['r'])
        found_maximal_radii = np.array(f_maximals['radius'])
        
        #Since VoidFinder sorts its maximal holes from largest to smallest,
        #we can and should directly compare row by row
        #Astropy tables are essentially Fortran matrix order so call out the
        #data by column instead of row
        #for truth_row, test_row in zip(self.maximals, f_maximals):
        #    print(truth_row, test_row)
        x_diffs = self.maximals['x'] - f_maximals['x']
        y_diffs = self.maximals['y'] - f_maximals['y']
        z_diffs = self.maximals['z'] - f_maximals['z']
        
        dists_sq = np.array(x_diffs*x_diffs + y_diffs*y_diffs + z_diffs*z_diffs)
        
        maximal_center_dist_diffs = np.sqrt(dists_sq)
        print("Maximal center dists: ", maximal_center_dist_diffs)
        
        """
        if False:
        
            x_diffs = self.maximals['x'] - maximals_truth['x']
            y_diffs = self.maximals['y'] - maximals_truth['y']
            z_diffs = self.maximals['z'] - maximals_truth['z']
            
            dists_sq = np.array(x_diffs*x_diffs + y_diffs*y_diffs + z_diffs*z_diffs)
            
            maximal_center_dist_diffs = np.sqrt(dists_sq)
            print("Old Maximal center dists: ", maximal_center_dist_diffs)
        """
        
        
        
        
        #We decided that maximal hole center positions should be within 10% of the truth
        #radius, and found maximal radii should be within 5% of the truth radius
        #self.assertTrue(all([np.allclose(f_maximals[name], maximals_truth[name]) for name in f_maximals.dtype.names]))
        
        self.assertTrue(np.all(maximal_center_dist_diffs <= 0.10*truth_maximal_radii))
        
        self.assertTrue(np.all(np.abs((truth_maximal_radii - found_maximal_radii)) <= 0.05*truth_maximal_radii))
        
        
        
        
        
        #self.assertEqual(len(setdiff(f_maximals, maximals_truth)), 0)

        # Check holes
        #holes_truth = Table.read('python/vast/voidfinder/tests/test_galaxies_redshift_holes_truth.txt', 
        #                         format='ascii.commented_header')

        # Sort both tables by void flag, radius, x, y, z columns
        f_holes.sort(['void', 'radius', 'x', 'y', 'z'])
        #holes_truth.sort(['void', 'radius', 'x', 'y', 'z'])
        '''
        for name in f_holes.dtype.names:
            if not np.allclose(f_holes[name], holes_truth[name]):
                diffs = np.isclose(f_holes[name], holes_truth[name])
                print(f_holes[name][~diffs], holes_truth[name][~diffs])
        '''
        
        
        
        print("Truth Holes data: ", len(self.holes))
        print(self.holes)
        print("VoidFinder Detected Holes: ", len(f_holes))
        f_holes.pprint_all()
        #print("Old VoidFinder Detected Holes: ", len(holes_truth))
        #holes_truth.pprint_all()
        
        
        #Changing the Holes test to make sure that they overlap their
        #identified maximal sphere by at least 50% in volume
        void_IDs = np.unique(f_holes['void'])
        
        maximal_positions = to_array(self.maximals) #pulls out x y z
        
        for idx, void_ID in enumerate(void_IDs):
            
            mask = f_holes['void'] == void_ID
            
            sub_holes = f_holes[mask]
            
            #mask = holes_truth['void'] == void_ID
            #sub_holes = holes_truth[mask]
            
            sub_holes_pos = to_array(sub_holes) #pulls out x y z
            
            sub_holes_radii = np.array(sub_holes['radius'])
        
            curr_maximal_pos = maximal_positions[idx]
            
            curr_maximal_radius = truth_maximal_radii[idx]
            
            
            for jdx in range(len(sub_holes)):
            
                hole_pos = sub_holes_pos[jdx]
                curr_radius = sub_holes_radii[jdx]
                
                separation = np.sqrt(np.sum((hole_pos - curr_maximal_pos)**2))
                
                
                #print(separation, curr_radius, curr_maximal_radius)
                
                #Hole does not overlap the maximal at all
                if separation > (curr_radius + curr_maximal_radius):
                    self.assertTrue(False)
            
                curr_cap_height = cap_height(curr_radius, curr_maximal_radius, separation)
                    
                maximal_cap_height = cap_height(curr_maximal_radius, curr_radius, separation)
            
                overlap_volume = spherical_cap_volume(curr_radius, curr_cap_height) + spherical_cap_volume(curr_maximal_radius, maximal_cap_height)
            
                hole_vol = (4.0/3.0)*np.pi*(curr_radius**3)
                
                print("Overlap pct: ", overlap_volume/hole_vol)
                
                self.assertTrue(overlap_volume/hole_vol >= 0.85)
            
            
            
        
        #self.assertTrue(all([np.allclose(f_holes[name], holes_truth[name]) for name in f_holes.dtype.names]))
        #self.assertEqual(len(setdiff(holes_truth, f_holes)), 0)
        



    def tearDown(self):
        """
        Delete files produced for the unit tests.
        """
        if os.path.exists(self.galaxies_filename):
            os.remove(self.galaxies_filename)

        files = [ 'test_field_gal_file.txt',
                  'test_galaxies_redshift_maximal.txt', 
                  'test_galaxies_redshift_holes.txt',
                  'test_wall_gal_file.txt' ]

        for f in files:
            if os.path.exists(f):
                os.remove(f)





if __name__ == '__main__':

    my_run = TestVoidFinder()

    my_run.setUp()

    print('Testing file_preprocess')
    my_run.test_1_file_preprocess()

    print('Testing generate_mask')
    my_run.test_2_generate_mask()

    print('Testing query_first')
    my_run.test_4_query_first()

    print('Testing main_algorithm')
    my_run.test_5_main_algorithm()





