'''
voidfinder.table_functions
==========================

General astropy table functions.
'''

import numpy as np
from astropy.table import Table

def add_row(table, row):
    '''Column-wise add table row to table'''

    out_table = Table()

    for name in row.colnames:
        out_table[name] = table[name] + row[name]

    return out_table

def subtract_row(table, row):
    '''Column-wise subtract table row from table'''

    out_table = Table()

    for name in row.colnames:
        out_table[name] = table[name] - row[name]

    return out_table

def table_divide(table, scalar):
    '''Scale values in table'''

    out_table = Table()

    for name in table.colnames:
        out_table[name] = table[name]/scalar

    return out_table

def table_dtype_cast(table, dtype):
    '''Cast table dtype to given dtype'''

    for name in table.colnames:
        table[name] = table[name].astype(dtype)

    return table

def row_cross(row1, row2):
    '''Calculate cross-product of two rows'''

    crossed_row = np.cross(to_vector(row1), to_vector(row2))

    return Table(crossed_row, names=['x','y','z'])

def row_dot(row1, row2):
    '''Calculate the dot-product of two rows'''

    dotted_row = np.dot(to_vector(row1), to_vector(row2))

    return dotted_row

def to_vector(row):
    '''Convert table row to numpy array'''

    vector = np.array([row['x'], row['y'], row['z']])
    vector = vector.T

    return vector

def to_array(table):
    '''Convert table to numpy array'''

    array = np.array([table['x'], table['y'], table['z']])
    array = array.T

    return array

