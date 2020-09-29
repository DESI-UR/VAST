import numpy as np
from vast.vsquared import util as util

def test_prob_false_voids():
    radius = np.array([1.1, 2., 3., 10.])
    prob = np.array([0.599029897, 0.00451658094, 5.08084353e-06, 7.30041296e-78])
    assert(np.isclose(prob, util.P(radius)).all())

def test_toCoord():
    z = np.array([0.1, 0.2, 0.5, 1.])
    ra = np.array([0., 0., 0., 0.])
    dec = np.array([0., 0., 0., 0.])
    dist = np.array([418.45448763,  816.72323905, 1888.62539593, 3303.82880589])
    c1,c2,c3 = util.toCoord(z, ra, dec, 70., 0.3)
    assert(np.isclose(c1, dist).all())
