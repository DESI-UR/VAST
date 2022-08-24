# -*- coding: utf-8 -*-
import unittest

import os
import numpy as np
from sklearn import neighbors
from astropy.table import Table, setdiff, vstack

from vast.voidfinder.constants import c
from vast.voidfinder import find_voids, filter_galaxies
from vast.voidfinder.multizmask import generate_mask
from vast.voidfinder.preprocessing import file_preprocess

class TestVoidFinder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Set up some global variables available to all test cases.
        '''
        TestVoidFinder.wall = None
        TestVoidFinder.dist_limits = None
        TestVoidFinder.mask = None
        TestVoidFinder.grid_shape = None

    def setUp(self):
        '''
        Set up a dummy survey that can be used to test I/O, preprocessing, and 
        void-finding.
        '''
        self.ra_range = np.arange(10, 30, 0.5)
        self.dec_range = np.arange(-10, 10, 0.5)
        self.redshift_range = np.arange(0, 0.011, 0.0005) # 0.0001

        RA, DEC, REDSHIFT = np.meshgrid(self.ra_range, 
                                        self.dec_range, 
                                        self.redshift_range)

        self.galaxies_table = Table()
        self.galaxies_table['ra'] = np.ravel(RA)
        self.galaxies_table['dec'] = np.ravel(DEC)
        self.galaxies_table['redshift'] = np.ravel(REDSHIFT)

        # Shuffle the table (so that the KDtree does not die)
        rng = np.random.default_rng()
        self.galaxies_shuffled = Table(rng.permutation(self.galaxies_table))
        self.galaxies_shuffled['Rgal'] = c*self.galaxies_shuffled['redshift']/100.
        N_galaxies = len(self.galaxies_shuffled)

        # All galaxies will be brighter than the magnitude limit, so that none
        # of them are removed
        self.galaxies_shuffled['rabsmag'] = 5*np.random.rand(N_galaxies) - 25.1

        self.galaxies_filename = 'test_galaxies.txt'
        self.galaxies_shuffled.write(self.galaxies_filename,
                                     format='ascii.commented_header',
                                     overwrite=True)

        self.gal = np.zeros((N_galaxies,3))
        self.gal[:,0] = self.galaxies_shuffled['Rgal']*np.cos(self.galaxies_shuffled['ra']*np.pi/180.)*np.cos(self.galaxies_shuffled['dec']*np.pi/180.)
        self.gal[:,1] = self.galaxies_shuffled['Rgal']*np.sin(self.galaxies_shuffled['ra']*np.pi/180.)*np.cos(self.galaxies_shuffled['dec']*np.pi/180.)
        self.gal[:,2] = self.galaxies_shuffled['Rgal']*np.sin(self.galaxies_shuffled['dec']*np.pi/180.)

        # Minimum maximal sphere radius
        self.min_maximal_radius = 1. # Mpc/h

    def test_1_file_preprocess(self):
        """
        Take a galaxy data file and return a data table, compute the redshift 
        range in comoving coordinates, and generate output filename.
        """
        f_galaxy_table, f_dist_limits, f_out1_filename, f_out2_filename = \
            file_preprocess(self.galaxies_filename, '', '', dist_metric='redshift')

        # Check the galaxy table
        self.assertEqual(len(setdiff(f_galaxy_table, self.galaxies_shuffled)), 0)

        # Check the distance limits
        TestVoidFinder.dist_limits = np.zeros(2)
        TestVoidFinder.dist_limits[1] = c*self.redshift_range[-1]/100.
        self.assertTrue(np.isclose(f_dist_limits, TestVoidFinder.dist_limits).all())

        # Check the first output file name
        self.assertEqual(f_out1_filename, 'test_galaxies_redshift_maximal.txt')

        # Check the second output file name
        self.assertEqual(f_out2_filename, 'test_galaxies_redshift_holes.txt')



    
    def test_2_generate_mask(self):
        """
        Take a table of galaxy coordinates and maximum redshift and return a 
        boolean mask + resolution
        """
        f_mask, f_mask_resolution = generate_mask(self.galaxies_shuffled, 
                                                  self.redshift_range[-1], 
                                                  dist_metric='redshift', 
                                                  min_maximal_radius=self.min_maximal_radius)

        # Check the mask
        TestVoidFinder.mask = np.zeros((360,180), dtype=bool)
        for i in range(int(self.ra_range[0]), int(self.ra_range[-1]+1)):
            for j in range(int(self.dec_range[0] + 90), int(self.dec_range[-1] + 90)+1):
                TestVoidFinder.mask[i, j] = True
        self.assertTrue((f_mask == TestVoidFinder.mask).all())

        # Check the mask resolution
        self.assertTrue(np.isclose(f_mask_resolution, 1))
    



    
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
        TestVoidFinder.wall = self.gal[dist3 < (np.mean(dist3) + 1.5*np.std(dist3))]
        self.assertTrue(np.isclose(f_wall, TestVoidFinder.wall).all())

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
    



    
    def test_4_find_voids(self):
        """
        Identify maximal spheres and holes in the galaxy distribution
        """
        maximals = Table()
        maximals['x'] = [25., 10.]
        maximals['y'] = [8., 3.]
        maximals['z'] = [0., -1.]
        maximals['r'] = [2.5, 1.5]
        maximals['flag'] = [0, 1]

        holes = Table()
        holes['x'] = [24., 10.5]
        holes['y'] = [7.9, 3.2]
        holes['z'] = [0.1, -0.5]
        holes['r'] = [2., 0.5]
        holes['flag'] = [0, 1]
        holes = vstack([holes, maximals])

        # Remove points which fall inside holes
        remove_boolean = np.zeros(len(TestVoidFinder.wall), dtype=bool)
        for i in range(len(holes)):
            d = (holes['x'][i] - TestVoidFinder.wall[:,0])**2 + (holes['y'][i] - TestVoidFinder.wall[:,1])**2 + (holes['z'][i] - TestVoidFinder.wall[:,2])**2
            remove_boolean = remove_boolean | (d < holes['r'][i]**2)

        find_voids([TestVoidFinder.wall[~remove_boolean], 
                    np.concatenate([TestVoidFinder.field, 
                                    TestVoidFinder.wall[remove_boolean]])], 
                   'test_', 
                   mask=TestVoidFinder.mask, 
                   mask_resolution=1,
                   dist_limits=TestVoidFinder.dist_limits,
                   hole_grid_edge_length=1.0,
                   hole_center_iter_dist=0.2, 
                   min_maximal_radius=self.min_maximal_radius, 
                   num_cpus=1, 
                   pts_per_unit_volume=0.01, # 5
                   void_table_filename='test_galaxies_redshift_holes.txt', 
                   maximal_spheres_filename='test_galaxies_redshift_maximal.txt')

        # Check maximal spheres
        f_maximals = Table.read('test_galaxies_redshift_maximal.txt', 
                                format='ascii.commented_header')
        maximals_truth = Table.read('python/vast/voidfinder/tests/test_galaxies_redshift_maximal_truth.txt', 
                                    format='ascii.commented_header')
        self.assertEqual(len(setdiff(f_maximals, maximals_truth)), 0)

        # Check holes
        f_holes = Table.read('test_galaxies_redshift_holes.txt', 
                             format='ascii.commented_header')
        holes_truth = Table.read('python/vast/voidfinder/tests/test_galaxies_redshift_holes_truth.txt', 
                                 format='ascii.commented_header')
        self.assertEqual(len(setdiff(holes_truth, f_holes)), 0)
    

    def tearDown(self):
        """Delete files produced for the unit tests.
        """
        if os.path.exists(self.galaxies_filename):
            os.remove(self.galaxies_filename)

        files = [ 'test_field_gal_file.txt',
                  #'test_galaxies_redshift_maximal.txt', 
                  #'test_galaxies_redshift_holes.txt',
                  'test_wall_gal_file.txt' ]

        for f in files:
            if os.path.exists(f):
                os.remove(f)

