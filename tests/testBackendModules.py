#!/bin/env python3

import sys
sys.path.append('./')

import doctest
import unittest
import numpy as np
import warnings

import optrace as ot


class BackendModuleTests(unittest.TestCase):


    # Szrface Function missing
    # TODO checks Masks and PlottingMeshes
    def test_Surface(self):

        # check if init is working without exceptionss
        S =  [ot.Surface("Circle", r=4),
              ot.Surface("Sphere", r=5, rho=1/10),
              ot.Surface("Point"),
              ot.Surface("Line", ang=45),
              ot.Surface("Asphere", r=3.2, rho=-1/7, k=2),
              ot.Surface("Rectangle", dim=[5, 6]),
              ot.Surface("Data", r=2, Data=np.array([[1, 2, 3], [-1, 0, -1], [5, 3, 0]])),
              ot.Surface("Ring", r=3, ri=0.2)]

        # check if exceptions in init work
        self.assertRaises(TypeError, ot.Surface)  # no type
        self.assertRaises(ValueError, ot.Surface, "ABC")  # invalid type
        self.assertRaises(ValueError, ot.Surface, "Asphere", r=3.2, rho=-1/7, k=5) # r too large for surface
        self.assertRaises(ValueError, ot.Surface, "Sphere", r=-1)  # size negative
        self.assertRaises(ValueError, ot.Surface, "Sphere", rho=0)  # rho zero
        self.assertRaises(ValueError, ot.Surface, "Ring", r=1, ri=-0.2)  # size negative
        self.assertRaises(ValueError, ot.Surface, "Ring", r=1, ri=1.4)  # ri larger than r
        self.assertRaises(ValueError, ot.Surface, "Rectangle", dim=[-5, 5])  # size negative
        self.assertRaises(TypeError, ot.Surface, "Function", func=lambda x, y: x + y)  # func is no ot.SurfaceFunction

        x = np.linspace(-0.7, 5.3, 1000)
        y = np.linspace(-8, -2, 1000)
        p = np.random.uniform(-2, -1, size=(100, 3))
        s = np.random.uniform(-1, 1, size=(100, 3))
        s /= np.linalg.norm(s, axis=1)[:, np.newaxis]  # normalize
        s[:, 2] = np.abs(s[:, 2])  # only positive s_z

        # check if other functions throw
        for i, Si in enumerate(S):
            with self.subTest(i=i, Surface=Si.surface_type):
                Si.getExtent()
                Si.copy()

                # moving working correctly
                pos_new = np.array([2.3, -5, 10])
                Si.moveTo(pos_new)
                self.assertTrue(np.all(pos_new == Si.pos))

                if Si.surface_type not in ["Point", "Line"]:
                    z = Si.getValues(x, y)
                    # print(z)
                    Si.getMask(x, y)
                    Si.getEdge(nc=100)
                    Si.getPlottingMesh(N=100)
                    Si.getNormals(x, y)

                    # check if moving actually moved the origin of the function
                    self.assertEqual(Si.getValues(np.array([pos_new[0]]), np.array([pos_new[1]]))[0], pos_new[2])

                    # check values
                    if Si.isPlanar():
                        self.assertTrue(np.all(z == pos_new[2]))
                    elif Si is S[1]:  # Sphere
                        self.assertAlmostEqual(z[0], 10.94461486)
                        self.assertAlmostEqual(z[500], 10.0000009)
                    elif Si is S[4]: # Asphere
                        self.assertAlmostEqual(z[0], 10.)  # == maxz since it's outside
                        self.assertAlmostEqual(z[300], 9.78499742)
                    elif Si is S[6]:                    
                        self.assertAlmostEqual(z[0], 15.)  # == maxz since it's outside
                        self.assertAlmostEqual(z[200], 10.8993994)

                if Si.hasHitFinding():
                    Si.findHit(p, s)

                if Si.isPlanar():
                    Si.getRandomPositions(N=1000)

    def test_SurfaceFunction(self):

        # turn off warnings temporarily, since not specifying maxz, minz in __init__ leads to one
        warnings.simplefilter("ignore")

        func = lambda x, y: 1 + x - y
        der = lambda x, y: tuple(np.ones_like(x), -np.ones_like(x))
        hits = lambda p, s: p*s  # not correct

        S0 = ot.SurfaceFunction(func, r=5, derivative=der, hits=hits)

        self.assertTrue(S0.hasHits())
        self.assertTrue(S0.hasDerivative())

        # max and min values are +-5*sqrt(2), since the offset of 1 is removed internally
        self.assertAlmostEqual(S0.maxz, 5*np.sqrt(2), delta=1e-6)
        self.assertAlmostEqual(S0.minz, -5*np.sqrt(2), delta=1e-6)

        # restore warnings
        warnings.simplefilter("default")

    def test_Filter(self):
        pass

    def test_Detector(self):
        pass

    def test_Lens(self):
        pass

    def test_RaySource(self):
        pass

    def test_Image(self):
        pass

    def test_Raytracer(self):
        pass

    def test_RefractionIndex(self):

        func = lambda wl: 2.0 - wl/500/5 

        # check __init__
        R = [ot.RefractionIndex("Constant", n=1.2),
             ot.RefractionIndex("Cauchy", coeff=[1.2, 0.004]),
             ot.RefractionIndex("Function", func=func)]

        # check presets
        for material in ot.presets_n:
            material(550)
            # print(material.desc, material.getAbbeNumber())

        # check exceptions
        self.assertRaises(ValueError, ot.RefractionIndex, "ABC")  # invalid type
        n2 = ot.RefractionIndex("Function")
        self.assertRaises(RuntimeError, n2, 550)  # func missing
        self.assertRaises(ValueError, ot.RefractionIndex, "Constant", n=0.99)  # n < 1
        # self.assertRaises(ValueError, ot.RefractionIndex, "Cauchy", Cauchy=[0.99])  # A < 1

        # check values

        wl = np.array([450., 580.])
        Rval = np.array([[1.2, 1.2],
                         [1.219753086, 1.211890606],
                         [1.82, 1.768]])

        for i, Ri in enumerate(R):
            for j, wlj in enumerate(wl):
                self.assertAlmostEqual(Ri(wlj), Rval[i, j], places=5)

        # check if equal operator is working
        self.assertEqual(ot.preset_n_SF10, ot.preset_n_SF10)
        self.assertEqual(R[1], R[1])
        self.assertEqual(ot.RefractionIndex("Function", func=func), ot.RefractionIndex("Function", func=func))

        
    def test_Color(self):
        doctest.testmod(ot.tracer.Color)

    def test_Misc(self):
        doctest.testmod(ot.tracer.Misc)


if __name__ == '__main__':
    unittest.main()

