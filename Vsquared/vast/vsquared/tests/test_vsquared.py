import numpy as np
import os
import copy
from vast.vsquared import util, classes, zobov
v2path = zobov.__file__[:-22]


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
        self.cat = classes.Catalog(v2path+"data/test_data.fits",16,0.03,0.1)

    def test_coord(self):
        mcoord = np.array([-158.0472400951847,-19.01100010666949,94.40978960900837])
        assert(np.isclose(np.mean(self.cat.coord.T[0]),mcoord[0]))
        assert(np.isclose(np.mean(self.cat.coord.T[1]),mcoord[1]))
        assert(np.isclose(np.mean(self.cat.coord.T[2]),mcoord[2]))

    def test_nnls(self):
        assert(len(self.cat.nnls[self.cat.nnls==-1])==1800)
        assert(len(self.cat.nnls[self.cat.nnls==np.arange(10000)])==8200)

    def test_mask(self):
        assert(len(self.cat.mask[self.cat.mask])==576)

    def test_imsk(self):
        assert(len(self.cat.imsk[self.cat.imsk])==8200)


class testTesselation():

    def __init__(self):
        cat = classes.Catalog(v2path+"data/test_data.fits",16,0.03,0.1)
        self.tess = classes.Tesselation(cat)

    def test_volumes(self):
        assert(np.isclose(np.mean(self.tess.volumes),1600.190056988941))

    def test_neighbors(self):
        assert(np.isclose(np.mean([len(nn) for nn in self.tess.neighbors]),16.06))


class testZones():

    def __init__(self):
        cat  = classes.Catalog(v2path+"data/test_data.fits",16,0.03,0.1)
        tess = classes.Tesselation(cat)
        self.zones = classes.Zones(tess)

    def test_zcell(self):
        assert(np.isclose(np.mean([len(zc) for zc in self.zones.zcell]),82.82828282828282))

    def test_zvols(self):
        assert(np.isclose(np.mean(self.zones.zvols),6549.395680389604))

    def test_zlinks(self):
        assert(np.isclose(np.mean([len(zl0) for zl0 in self.zones.zlinks[0]]),12.626262626262626))
        assert(np.isclose(np.mean([np.mean(zl1) for zl1 in self.zones.zlinks[1]]),2345.915247204872))


class testVoids():

    def __init__(self):
        cat   = classes.Catalog(v2path+"data/test_data.fits",16,0.03,0.1)
        tess  = classes.Tesselation(cat)
        zones = classes.Zones(tess)
        self.voids = classes.Voids(zone)

    def test_voids(self):
        assert(np.isclose(np.mean([len(v) for v in self.voids.voids]),1.9191919191919191))
        assert(np.isclose(np.mean([np.mean([len(vv) for vv in v]) for v in self.voids.voids]),1.0578692450350204))

    def test_mvols(self):
        assert(np.isclose(np.mean(self.voids.mvols),6549.3956803896035))

    def test_ovols(self):
        assert(np.isclose(np.mean([np.mean(ov) for ov in self.voids.ovols]),5519.078018084154))


class testZobov():

    def __init__(self):
        self.zobov = zobov.Zobov(v2path+"data/test_config.ini",save_intermediate=False)

    def test_sortVoids(self):
        zobov1 = copy.deepcopy(self.zobov)
        zobov2 = copy.deepcopy(self.zobov)
        zobov4 = copy.deepcopy(self.zobov)
        self.zobov.sortVoids()
        zobov1.sortVoids(1)
        zobov2.sortVoids(2)
        zobov4.sortVoids(4)
        assert(len(self.zobov.vrads) == 62)
        assert(len(zobov1.vrads) == 87)
        assert(len(zobov2.vrads) == 14)
        assert(len(zobov4.vrads) == 47)

    def test_saveVoids(self):
        if not hasattr(self.zobov,'vrads'):
            print("Run test_sortVoids first")
        self.zobov.saveVoids()
        assert(os.path.exists(v2path+"data/TEST_zobovoids.dat"))
        os.remove(v2path+"data/TEST_zobovoids.dat")
        assert(os.path.exists(v2path+"data/TEST_zonevoids.dat"))
        os.remove(v2path+"data/TEST_zonevoids.dat")

    def test_saveZones(self):
        self.zobov.saveZones()
        assert(os.path.exists(v2path+"data/TEST_galzones.dat"))
        os.remove(v2path+"data/TEST_galzones.dat")
