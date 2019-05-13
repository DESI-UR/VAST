from unittest import TestCase

import numpy as np
from astropy.table import Table
from voidfinder.table_functions import to_array

class TestTableFunctions(TestCase):
    def test_to_array(self):
        x = [0, 1, 0, 30, 55, -18, 72, 0]
        y = [0, 0, -18, 0, 0, 0, 0, 100]
        radius = [20, 11, 15, 16, 18, 9, 8, 7]
        table = Table([x, y, radius], names=('x','y','radius'))
        table['z'] = 0
        fake_array = np.asarray([table['x'], table['y'], table['z']]).T
        self.assertTrue(np.array_equal(to_array(table), fake_array))
