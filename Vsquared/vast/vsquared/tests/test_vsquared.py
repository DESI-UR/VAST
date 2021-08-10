import numpy as np
from vast.vsquared import util, classes, zobov


class testUtil():

    def test_toCoord(self):
        z = np.array([0.1, 0.2, 0.5, 1.])
        ra = np.array([0., 0., 0., 0.])
        dec = np.array([0., 0., 0., 0.])
        dist = np.array([418.45448763, 816.72323905, 1888.62539593, 3303.82880589])
        c1,c2,c3 = util.toCoord(z, ra, dec, 70., 0.3)
        assert(np.isclose(c1, dist).all())

    def test_toSky(self):
        c1 = np.array([500., 1000., 1500., 0.])
        c2 = np.array([0., 0., 0., 1000.])
        c3 = np.array([0., 0., 0., 1000.])
        coord = np.array([c1,c2,c3]).T
        redshift = np.array([0.12006335, 0.2478459 , 0.38520913, 0.36086924])
        z,ra,dec = util.toSky(coord, 70., 0.3, 0.001)
        assert(np.isclose(z, redshift).all())

    def test_inSphere(self):
        cs = np.array([1., 0., -1.])
        c1 = np.array([1., 2., 1.5, 2.])
        c2 = np.array([0., 0., 0.5, 1.])
        c3 = np.array([-1., -1., -0.5, 0.])
        coord = np.array([c1,c2,c3]).T
        insph = util.inSphere(cs, 1., coord)
        assert((insph == np.array([True, False, True, False])).all())

    def test_wCen(self):
        vols = np.array([1., 2., 3., 4.])
        c1 = np.array([0., 100., 0., -300.])
        c2 = np.array([0., 0., 100., 100.])
        c3 = np.array([0., 0., -100., 0.])
        coord = np.array([1000., 1000., 1000.]) + np.array([c1,c2,c3]).T
        wc = np.array([900., 1070., 970.])
        wcen = util.wCen(vols, coord)
        assert(np.isclose(wcen, wc).all())

    def test_SMA(self):
        c1 = np.array([0., 100., 0., -300.])
        c2 = np.array([0., 0., 100., 100.])
        c3 = np.array([0., 0., -100., 0.])
        coord = np.array([1000., 1000., 1000.]) + np.array([c1,c2,c3]).T
        ax2 = np.array([-12.3314653, 7.39154934, 4.1289785])
        SMA = util.getSMA(10., coord)
        assert(np.isclose(SMA[1],ax2).all())

    def test_prob_false_voids(self):
        radius = np.array([1.1, 2., 3., 10.])
        prob = np.array([0.599029897, 0.00451658094, 5.08084353e-06, 7.30041296e-78])
        assert(np.isclose(prob, util.P(radius)).all())

    def test_flatten(self):
        l = [[[0.,[1.,2.]],[3.,4.,5.],[6.],[7.,8.],[9.,10.,[11.,12.,13.],14.]], \
             [[15.,16.,17.],[18.]],[[19.],[20.],[21.],22.]]
        flat = np.arange(23)
        l2 = np.array(list(util.flatten(l)))
        assert(np.isclose(l2, flat).all())


class testCatalog():

    def __init__(self):
        self.cat = classes.Catalog(None,1,0.2,0.9,-20.,test=True)

    def test_coord(self):
        tcoord = np.array([[ 783.9834049 , 1309.17621023,  786.65493169], \
                           [ 251.57906769,  215.05470304, 1659.8600681 ], \
                           [-953.01273258,  551.00968422, 1343.34965151], \
                           [ 654.24618327,  109.69279205,  320.15252084], \
                           [ 466.44534345,  -49.68217317, 1020.76670875], \
                           [-504.01798857, 1444.59430258, 1628.78672599], \
                           [1517.98012095, -326.67329107,  288.60190882], \
                           [-935.32678971,  167.47281955,  487.43010419], \
                           [ -16.48003108,   11.75078132,  370.69222288], \
                           [1127.89393595, 1385.89505997, -172.34543953]])
        assert(np.isclose(self.cat.coord,tcoord).all())

    def test_mcut(self):
        tmcut = np.array([True, True, True, True, False, False, True, True, False, True])
        assert((self.cat.mcut==tmcut).all())

    def test_nnls(self):
        tnnls = np.array([0, 1, 2, 3, 1, -1, 6, 7, -1, 9])
        assert((self.cat.nnls==tnnls).all())

    def test_mask(self):
        tmask = np.array([True, True, False, False, True, False, True, False, True, False, False, False])
        assert((self.cat.mask==tmask).all())

    def test_imsk(self):
        timsk = np.array([True, True, True, True, False, False, True, True, False, True])
        assert((self.cat.imsk==timsk).all())
