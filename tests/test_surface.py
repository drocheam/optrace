#!/bin/env python3

import unittest
import numpy as np
import pytest

import optrace.tracer.misc as misc
import optrace as ot
from optrace.tracer.geometry.surface import Surface



class SurfaceTests(unittest.TestCase):

    def test_surface_init_type_errors(self):
        self.assertRaises(TypeError, ot.FunctionSurface2D, 3, func=2)  # func is no function
        self.assertRaises(TypeError, ot.TiltedSurface, r=2, normal=True)  # invalid normal type
        self.assertRaises(TypeError, Surface, r=[])  # invalid r type
        self.assertRaises(TypeError, ot.RingSurface, r=2, ri=[])  # invalid ri type
        self.assertRaises(TypeError, ot.ConicSurface, r=1, R=10, k=[])  # invalid k type
        self.assertRaises(TypeError, ot.AsphericSurface, r=1, R=10, k=[], coeff=[0.])  # invalid k type
        self.assertRaises(TypeError, ot.AsphericSurface, r=1, R=10, k=2, coeff=0)  # invalid coeff type
        self.assertRaises(TypeError, ot.SphericalSurface, r=1, R=[])  # invalid R type
        self.assertRaises(TypeError, ot.ConicSurface, r=1, R=[], k=0)  # invalid R type
        self.assertRaises(TypeError, ot.RectangularSurface, dim=True)  # invalid dim type
        self.assertRaises(TypeError, ot.RectangularSurface, dim=[1, 2, 3])  # invalid dim shape
        self.assertRaises(TypeError, ot.SlitSurface, dim=[1, 1], dimi=True)  # invalid dimi type
        self.assertRaises(TypeError, ot.SlitSurface, dim=[1, 1], dimi=True)  # invalid dimi type
        self.assertRaises(TypeError, ot.SlitSurface, dim=[1, 1], dimi=[0.1, 0.1, 0.1])  # invalid dimi shape
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=lambda x, y: x, z_min=[], z_max=2)
        # ^-- invalid z_min type
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=lambda x, y: x, z_min=0, z_max=[])
        # ^-- invalid z_max type
        self.assertRaises(TypeError, ot.DataSurface2D, r=3, data=True)  # invalid data type
        self.assertRaises(TypeError, ot.DataSurface2D, r=3, data=np.ones((100, 100)), parax_roc=[])
        # ^-- invalid curvature type
        self.assertRaises(TypeError, ot.DataSurface1D, r=3, data=True)  # invalid data type
        self.assertRaises(TypeError, ot.DataSurface1D, r=[22], data=np.linspace(0, 1, 100))  # invalid r type
        self.assertRaises(TypeError, ot.DataSurface1D, r=2, data=np.ones(100), parax_roc=[])  # invalid curvature type
        self.assertRaises(TypeError, ot.TiltedSurface, r=3, normal_sph=3)  # invalid normal_sph type

    def test_surface_init_value_errors(self):
        self.assertRaises(ValueError, ot.TiltedSurface, r=2, normal=[0.5, 0, -np.sqrt(1-0.5**2)])  # n_z negative
        self.assertRaises(ValueError, ot.TiltedSurface, r=2, normal=[1, 0, 0])  # n_z zero
        self.assertRaises(ValueError, ot.ConicSurface, r=3.2, R=-7, k=5)  # r too large for surface
        self.assertRaises(ValueError, ot.SphericalSurface, r=-1, R=20)  # size negative
        self.assertRaises(ValueError, ot.SphericalSurface, r=4, R=0)  # R zero
        self.assertRaises(ValueError, ot.ConicSurface, r=4, R=0, k=2)  # R zero
        self.assertRaises(ValueError, ot.AsphericSurface, r=4, R=0, k=2, coeff=[0, 2])  # R zero
        self.assertRaises(ValueError, ot.SphericalSurface, r=2, R=np.inf)  # R inf
        self.assertRaises(ValueError, ot.SphericalSurface, r=1, R=-np.inf)  # R -inf
        self.assertRaises(ValueError, ot.RingSurface, r=1, ri=-0.2)  # size negative
        self.assertRaises(ValueError, ot.RingSurface, r=1, ri=0)  # ri zero
        self.assertRaises(ValueError, Surface, r=0)  # r zero
        self.assertRaises(ValueError, ot.RingSurface, r=1, ri=1.4)  # ri larger than r
        self.assertRaises(ValueError, ot.RectangularSurface, dim=[-5, 5])  # size negative
        self.assertRaises(ValueError, ot.RectangularSurface, dim=[5, np.inf])  # size non finite
        self.assertRaises(ValueError, ot.SlitSurface, dim=[-5, 5], dimi=[0.1, 0.1])  # size negative
        self.assertRaises(ValueError, ot.SlitSurface, dim=[5, np.inf], dimi=[0.1, 0.1])  # size non finite
        self.assertRaises(ValueError, ot.SlitSurface, dim=[5, 5], dimi=[-3, 3])  # size negative
        self.assertRaises(ValueError, ot.SlitSurface, dim=[5, 5], dimi=[1, np.inf])  # size non finite
        self.assertRaises(ValueError, ot.SlitSurface, dim=[5, 5], dimi=[6, 1])  # slit too large x
        self.assertRaises(ValueError, ot.SlitSurface, dim=[5, 5], dimi=[1, 6])  # slit too large y
        self.assertRaises(ValueError, ot.DataSurface2D, r=2, data=np.array([[1, 2], [3, 4]]))
        # ^-- data needs to be at least 50x5
        self.assertRaises(ValueError, ot.DataSurface2D, r=2, data=np.ones((100, 101)))  # matrix not square
        self.assertRaises(ValueError, ot.DataSurface2D, r=2, data=np.array([[1, np.nan], [4, 5]]))  # nan in data
        self.assertRaises(ValueError, ot.DataSurface2D, r=2, data=np.ones((100, 100, 100)))  # data needs to be 2D
        self.assertRaises(ValueError, ot.DataSurface1D, r=2, data=np.ones(10))  # not enough values
        self.assertRaises(ValueError, ot.DataSurface1D, r=100, data=np.ones(100)*np.nan)  # nan in data
        self.assertRaises(ValueError, ot.DataSurface1D, r=5, data=np.mgrid[-1:1:100j, -1:1:100j][0])
        # ^-- wrong number of dimensions
        self.assertRaises(ValueError, ot.AsphericSurface, r=2, k=2, R=50, coeff=[4, np.nan])  # nan in coeffs
        self.assertRaises(ValueError, ot.AsphericSurface, r=2, k=2, R=50, coeff=[])  # empty coeff
       
    def test_surface_init_runtime_errors(self):
        self.assertRaises(RuntimeError, ot.TiltedSurface, r=3)  # normal or normal_sph missing

    def _test_surfaces(self):

        # SurfaceFunction with all funcs defined
        sfunc =  ot.FunctionSurface2D(func = lambda x, y: np.ones_like(x),
                                      deriv_func=lambda x, y: (np.zeros_like(x), np.zeros_like(x)),
                                      mask_func=lambda x, y: x+y < 2, r=3)

        S = [ot.CircularSurface(r=4),
             ot.SphericalSurface(r=5, R=10),
             ot.ConicSurface(r=3.2, R=-7, k=2),
             ot.AsphericSurface(r=3, R=-20, k=1, coeff=[0.02, 0.0034, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5]),
             ot.RectangularSurface(dim=[5, 6]),
             ot.SlitSurface(dim=[5, 6], dimi=[0.1, 0.2]),
             ot.FunctionSurface2D(func=lambda x, y: x, r=2),
             ot.FunctionSurface2D(func=lambda x, y: -x, r=2),
             ot.FunctionSurface1D(func=lambda r: r**2, r=2),
             ot.FunctionSurface1D(func=lambda r: r**2 + 0.001*r**4, r=3),
             sfunc,
             ot.TiltedSurface(r=4, normal=[0.5, 0, np.sqrt(1-0.5**2)]),
             ot.DataSurface2D(r=2, data=1+np.random.sample((50, 50))),
             ot.DataSurface1D(r=2, data=1+np.random.sample(50)),
             ot.RingSurface(r=3, ri=0.2),  # ring with ri < 0.5r
             ot.RingSurface(r=3, ri=2)]  # ring with ri > 0.5r gets plotted differently

        return S

    def test_surface_geometry(self):

        for i, Si in enumerate(self._test_surfaces()):
            with self.subTest(i=i, Surface=type(Si).__name__):
                # thickness expressions
                self.assertAlmostEqual(Si.ds, Si.dn + Si.dp, 8)

                # check that surface center is at pos[2]
                self.assertAlmostEqual(Si.values(np.array([Si.pos[0]]), np.array([Si.pos[1]]))[0], Si.pos[2])

                # moving working correctly
                pos_new = np.array([2.3, -5, 10])
                Si.move_to(pos_new)
                self.assertTrue(np.all(pos_new == Si.pos))

                # get info
                Si.info

                # check if moving actually moved the origin of the function
                self.assertEqual(Si.values(np.array([pos_new[0]]), np.array([pos_new[1]]))[0], pos_new[2])

    def test_surface_normals(self):
       
        # test test_surfaces
        for i, Si in enumerate(self._test_surfaces()):
            with self.subTest(i=i, Surface=type(Si).__name__):
       
                # values on and beyond the surface
                x = np.linspace(Si.extent[0]-1, Si.extent[1]+2, 1000)
                y = np.linspace(Si.extent[2]-1, Si.extent[3]+2, 1000)
                
                n = Si.normals(x, y)

                # z component needs to be positive
                self.assertTrue(np.all(n[:, 2] > 0))

                # length is 1
                self.assertTrue(np.allclose(n[:, 0]**2 + n[:, 1]**2 + n[:, 2]**2 - 1, 0))

                # points outside the surface have a normal of [0, 0, 1]
                m = Si.mask(x, y)
                self.assertTrue(np.all(n[~m] == [0, 0, 1]))
                
                # flat surfaces have n = [0, 0, 1] everywhere
                if Si.is_flat():
                    self.assertTrue(np.all(n[:, 2] == 1))

                # for a tilted surface n = Si.normal if on the surface
                if isinstance(Si, ot.TiltedSurface):
                    self.assertTrue(np.all(n[m] == Si.normal))


        # list of non-flat surfaces and normal values for the x, y vectors below
        iter_ = zip([ot.SphericalSurface(r=5, R=-10),
                     ot.ConicSurface(r=5, R=12, k=3),
                     ot.ConicSurface(r=5, R=-12, k=3),
                     ot.AsphericSurface(r=5, R=12, k=3, coeff=[0, 1e-4, 1e-8]),
                     ot.TiltedSurface(r=2, normal_sph=[20, 50]),
                     ot.DataSurface1D(r=3, data=-1+np.linspace(0, 3, 200)**1.25),
                     ot.FunctionSurface2D(r=4, func=lambda x, y: x ** 2 + y ** 2 / 2),
                     ot.DataSurface2D(r=3, data=1+np.mgrid[-3:3:100j, -3:3:100j][1] + np.linspace(-3, 3, 100)**2),
                     ot.FunctionSurface1D(r=4, func=lambda r: r**2)],
                    [[0.1,        0.05,       0.99373035],
                     [-0.08444007, -0.04222003,  0.9955337],
                     [0.08444007, 0.04222003, 0.9955337],
                     [-0.08493345, -0.04246673,  0.99548123],
                     [0.21984631, 0.26200263, 0.93969262],
                     [-0.70594412, -0.35297206,  0.61404693],
                     [-0.87287156, -0.21821789,  0.43643578],
                     [0, -0.894427191, 0.447213595]])

        # some offset
        x0, y0, z0 = 1.24, -5.8, 0.01

        # check values
        for surf, n in iter_:
            surf.move_to([x0, y0, z0])

            x = np.array([x0+1])
            y = np.array([y0+0.5])

            na = surf.normals(x, y)
            self.assertTrue(np.allclose(n - na, 0))

    def test_surface_values(self):
        
        x = np.linspace(-0.7, 5.3, 1000)
        y = np.linspace(-8, -2, 1000)

        # check flat surfaces 
        for Si in self._test_surfaces():
            if Si.is_flat():
                pos_new = np.array([2.3, -5, 10])
                Si.move_to(pos_new)
                z = Si.values(x, y)

                self.assertTrue(np.all(z == Si.pos[2]))

        # list of non-flat surfaces and values for the x, y vectors below
        iter_ = zip([ot.SphericalSurface(r=5, R=-10),
                     ot.ConicSurface(r=5, R=12, k=3),
                     ot.AsphericSurface(r=5, R=12, k=3, coeff=[0, 1e-4, 1e-8]),
                     ot.TiltedSurface(r=2, normal_sph=[20, 50]),
                     ot.DataSurface1D(r=3, data=-1+np.linspace(0, 3, 200)**1.25),
                     ot.FunctionSurface2D(r=4, func=lambda x, y: x ** 2 + y ** 2 / 2),
                     ot.DataSurface2D(r=3, data=1+np.mgrid[-3:3:100j, -3:3:100j][1] + np.linspace(-3, 3, 100)**2),
                     ot.FunctionSurface1D(r=4, func=lambda r: r + 0.1*r**2)],
                    [[-0.06269654, -0.01250782],
                    [0.05254347, 0.01043481],
                    [0.05269974, 0.01044106],
                    [-0.37336424,  0.13940869],
                    [1.14965824, 0.42044821],
                    [1.125, 0.125],
                    [0.75, -0.25],
                    [1.24303399, 0.525]])

        # some offset
        x0, y0, z0 = 1.24, -5.8, 0.01

        # check values
        for surf, z in iter_:
            surf.move_to([x0, y0, z0])

            x = np.array([x0+1, x0])
            y = np.array([y0+0.5, y0-0.5])

            za = surf.values(x, y) - z0
            
            for zi1, zi2 in zip(z, za):
                self.assertAlmostEqual(zi1, zi2)

    @pytest.mark.os
    def test_surface_hit_finding(self):

        p = np.random.uniform(-2, -1, size=(10000, 3))
        s = np.random.uniform(-1, 1, size=(10000, 3))
        s /= np.linalg.norm(s, axis=1)[:, np.newaxis]  # normalize
        s[:, 2] = np.abs(s[:, 2])  # only positive s_z
        
        for i, Si in enumerate(self._test_surfaces()):
            with self.subTest(i=i, Surface=type(Si).__name__):
                p_hit, is_hit, _ = Si.find_hit(p, s)

                # check that all hitting rays have the correct z-value at the surface
                z_hit = Si.values(p_hit[is_hit, 0], p_hit[is_hit, 1])
                self.assertTrue(np.allclose(p_hit[is_hit, 2] - z_hit, 0, rtol=0, atol=Surface.C_EPS))
              
                # rays missing, but with valid x,y coordinates inside extent must always be behind the surface
                zs = Si.values(p_hit[~is_hit, 0], p_hit[~is_hit, 1])
                ms = Si.mask(p_hit[~is_hit, 0], p_hit[~is_hit, 1])
                self.assertTrue(np.all(p_hit[~is_hit, 2][ms] > zs[ms]))

                # check if p_hit is actually a valid position on the ray
                t = (p_hit[:, 2] - p[:, 2])/s[:, 2]
                p_ray = p + s*t[:, np.newaxis]
                self.assertTrue(np.allclose(p_ray[:, 0] - p_hit[:, 0], 0, atol=Si.C_EPS))
                self.assertTrue(np.allclose(p_ray[:, 1] - p_hit[:, 1], 0, atol=Si.C_EPS))
                self.assertTrue(np.allclose(p_ray[:, 2] - p_hit[:, 2], 0, atol=Si.C_EPS))

                # when rays start behind the surface their hit position is just the start position
                p_hit[:, 2] = Si.z_max + 2
                p_hit2, is_hit, _ = Si.find_hit(p_hit, s)
                self.assertTrue(np.allclose(p_hit2 - p_hit, 0))
                self.assertFalse(np.any(is_hit))

    @pytest.mark.os
    def test_surface_random_positions(self):

        for i, Si in enumerate(self._test_surfaces()):
            with self.subTest(i=i, Surface=type(Si).__name__):
                # random positions only for planar types
                if isinstance(Si, ot.RectangularSurface | ot.CircularSurface | ot.RingSurface)\
                        and not isinstance(Si, ot.SlitSurface):
                    # check that generated positions are all inside surface and have correct values
                    p_ = Si.random_positions(N=10000)
                    m_ = Si.mask(p_[:, 0], p_[:, 1])
                    z_ = Si.values(p_[:, 0], p_[:, 1])
                    self.assertTrue(np.all(m_))
                    self.assertTrue(np.allclose(z_-p_[:, 2], 0))

    def test_surface_mask(self):

        for i, Si in enumerate(self._test_surfaces()):
            with self.subTest(i=i, Surface=type(Si).__name__):
                
                # check mask values. Especially surface edge, which by definition is valid
                if isinstance(Si, ot.RectangularSurface):  # includes SlitSurface

                    if not isinstance(Si, ot.SlitSurface):
                        self.assertTrue(Si.mask([Si.pos[0]], [Si.pos[1]])[0])
                    else:
                        self.assertFalse(Si.mask(np.array([Si.pos[0]]), np.array([Si.pos[1]]))[0])
                        self.assertTrue(Si.mask(np.array([Si.pos[0] + Si.dimi[0]/2]), np.array([Si.pos[1]]))[0])

                    self.assertTrue(Si.mask([Si.extent[0]], [Si.extent[2]])[0])
                    self.assertTrue(Si.mask([Si.extent[0]], [Si.extent[3]])[0])
                    self.assertTrue(Si.mask([Si.extent[1]], [Si.extent[3]])[0])
                    self.assertTrue(Si.mask([Si.extent[1]], [Si.extent[3]])[0])
                    self.assertFalse(Si.mask([Si.extent[1] + 1], [Si.pos[1]])[0])
                else:
                    if isinstance(Si, ot.FunctionSurface2D) and Si.mask_func is not None:
                        # special check for one of the SurfaceFunction Surfaces
                        self.assertTrue(Si.mask(np.array([Si.pos[0]]), np.array([Si.pos[1]]))[0])
                        self.assertFalse(Si.mask(np.array([Si.pos[0] + 1.5]), np.array([Si.pos[1] + 1.5]))[0])
                    else:
                        self.assertTrue(Si.mask(np.array([Si.extent[1]]), np.array([Si.pos[1]]))[0])
                        self.assertFalse(Si.mask(np.array([Si.extent[1] + 1]), np.array([Si.pos[1]]))[0])

                    if isinstance(Si, ot.RingSurface):
                        self.assertFalse(Si.mask(np.array([Si.pos[0]]), np.array([Si.pos[1]]))[0])
                        self.assertTrue(Si.mask(np.array([Si.pos[0] + Si.ri]), np.array([Si.pos[1]]))[0])

    def test_surface_get_edge(self):

        for i, Si in enumerate(self._test_surfaces()):
            with self.subTest(i=i, Surface=type(Si).__name__):
                for nc in [22, 23, 51, 100, 10001]:
                    x_, y_, z_ = Si.edge(nc)
                    
                    # check shapes
                    self.assertEqual(x_.shape[0], nc)
                    self.assertEqual(y_.shape[0], nc)
                    self.assertEqual(z_.shape[0], nc)

                    # check values
                    z2 = Si.values(x_, y_)
                    self.assertTrue(np.allclose(z_-z2, 0))

    def test_surface_plotting_mesh(self):

        for i, Si in enumerate(self._test_surfaces()):
            with self.subTest(i=i, Surface=type(Si).__name__):
                for N in [10, 11, 25, 200]:
                    X, Y, Z = Si.plotting_mesh(N)
                    
                    # check shapes (Slit and Rectangle Surface don't use N parameter)
                    if not isinstance(Si, ot.RectangularSurface):  # includes SlitSurface
                        self.assertEqual(X.shape, (N, N))
                        self.assertEqual(Y.shape, (N, N))
                        self.assertEqual(Z.shape, (N, N))
                    
                    # check if values are masked correctly
                    m2 = Si.mask(X.flatten(), Y.flatten())
                    self.assertTrue(np.all(m2 == np.isfinite(Z.flatten())))

                    # check if values are calculated correctly
                    z2 = Si.values(X.flatten(), Y.flatten())
                    valid = np.isfinite(Z.flatten())
                    self.assertTrue(np.allclose(Z.flatten()[valid]-z2[valid], 0))
                   
                    # check if values are in relevant area of circle
                    if not isinstance(Si, ot.RectangularSurface):  # also excludes SlitSurface
                        # no points outside circle area
                        R2 = (X-Si.pos[0])**2 + (Y-Si.pos[1])**2
                        self.assertTrue(np.all(R2 < (Si.r + 10*Si.N_EPS)**2))

                        # no points in empty region of ring
                        if isinstance(Si, ot.RingSurface):
                            self.assertTrue(np.all(R2 > (Si.ri - 10*Si.N_EPS)**2))

    def test_surface_misc_exceptions(self):

        # test object lock
        S = ot.CircularSurface(r=2)
        self.assertRaises(AttributeError, S.__setattr__, "r46546", 4)  # _new_lock active
        self.assertRaises(RuntimeError, S.__setattr__, "r", 4)  # object locked

        # test array lock
        self.assertRaises(RuntimeError, S.__setattr__, "pos", np.array([4, 5]))  # object locked
        
        def setArrayElement():
            ot.CircularSurface(r=3).pos[0] = 1.
        self.assertRaises(ValueError, setArrayElement)  # array elements read-only

        self.assertRaises(ValueError, ot.ConicSurface(r=3, R=-10, k=1).plotting_mesh, 5)  # N < 10
        self.assertRaises(ValueError, ot.ConicSurface(r=3, R=-10, k=1).edge, 5)  # nc < 20

    def test_surface_paraxial_radius_of_curvature(self):

        # spheric and aspheric
        self.assertAlmostEqual(ot.SphericalSurface(r=3, R=10).parax_roc, 10)
        self.assertAlmostEqual(ot.ConicSurface(r=3, k=1, R=-5.897).parax_roc, -5.897)
        self.assertAlmostEqual(ot.AsphericSurface(r=3, k=1, R=5.897, coeff=[0]).parax_roc, 5.897)
        self.assertAlmostEqual(ot.AsphericSurface(r=3, k=1, R=5.897, coeff=[-1e-3]).parax_roc, 5.967379271)
        # ^-- r^2 of polynomial contributes to curvature

        # flat
        self.assertAlmostEqual(ot.CircularSurface(r=2).parax_roc, np.inf)
        self.assertAlmostEqual(ot.RingSurface(r=2, ri=0.5).parax_roc, np.inf)
        self.assertAlmostEqual(ot.RectangularSurface(dim=[1, 1]).parax_roc, np.inf)
        self.assertAlmostEqual(ot.SlitSurface(dim=[1, 1], dimi=[0.1, 0.1]).parax_roc, np.inf)

        # data and function surfaces
        self.assertEqual(ot.DataSurface2D(data=np.ones((50, 50)), r=3).parax_roc, None)
        self.assertAlmostEqual(ot.DataSurface2D(data=np.ones((50, 50)), r=3, parax_roc=5).parax_roc, 5)
        # ^-- (factually wrong)
        self.assertAlmostEqual(ot.DataSurface1D(data=np.ones(50), r=3, parax_roc=5).parax_roc, 5)  # (factually wrong)
        func = lambda x, y: (x**2 + y**2)/1000
        self.assertEqual(ot.FunctionSurface2D(func=func, r=2).parax_roc, None)
        self.assertEqual(ot.FunctionSurface2D(func=func, r=2, parax_roc=5).parax_roc, 5)  # (factually wrong)
        func = lambda r: r**2/1000
        self.assertEqual(ot.FunctionSurface1D(func=func, r=2).parax_roc, None)
        self.assertEqual(ot.FunctionSurface1D(func=func, r=2, parax_roc=5).parax_roc, 5)  # (factually wrong)

    def test_surface_sphere_projection_quadrants(self):
        
        # test that coordinates on sphere get projected in the correct quadrant relative to its center
        # see sign convention notes in Surface.sphere_projection
        # and check that center always gets projected to the center
        
        # actual projection behavior gets checked in test_tracer tests

        # test different offsets, curvatures and curvatures signs
        for x0, y0, z0 in zip([0, 5, -1], [0, -3, 2], [0, 2, -10]):
            for R in [0.01, 1, 100]:
                for sign in [-1, 1]:
                    # generate surface
                    surf = ot.SphericalSurface(r=0.999*R, R=R*sign)
                    surf.move_to([x0, y0, z0])

                    # 3D offsets and position
                    p0 = np.tile(surf.pos, (5, 1))
                    p1 = 0.9*R*np.array([[0, 0, 0], [+1, +1, 0], [+1, -1, 0], [-1, +1, 0], [-1, -1, 0]])
                    p = p0 + p1

                    # get correct z values
                    p[:, 2] = surf.values(p[:, 0], p[:, 1])

                    # test all projection methods
                    for projm in ot.SphericalSurface.sphere_projection_methods:
                        p1m = p1 if projm != "Orthographic" else p1 + [x0, y0, z0]  
                        # ^--  in orthographic mode we need to add the position, since it uses absolute coordinates
                        pp = surf.sphere_projection(p, projm)
                        self.assertTrue(np.all(np.sign(pp[1:, :2]) == np.sign(p1m[1:, :2])))  # correct quadrant
                        self.assertTrue(np.allclose(pp[0, :2] - p1m[0, :2], 0, atol=1e-9, rtol=0))
                        # ^-- projection at center

    def test_surface_zmin_zmax_values(self):

        r_ = np.linspace(0, 3, 1000)
        Y, X = np.mgrid[-3:3:501j, -3:3:501j]
        R = np.sqrt(X**2 + Y**2)

        for r, DataSurface in zip([r_, R], [ot.DataSurface1D, ot.DataSurface2D]):

            # smooth function with zero offset
            surf1 = DataSurface(r=3, data=r**2)
            self.assertAlmostEqual(surf1.z_min, 0)
            self.assertAlmostEqual(surf1.z_max, 9)
            
            # smooth function with non zero offset
            surf1 = DataSurface(r=3, data=1000+r**2)
            self.assertAlmostEqual(surf1.z_min, 0)
            self.assertAlmostEqual(surf1.z_max, 9)
           
            # function with non central peak at r=0.5 with z = 1000
            surf1 = DataSurface(r=3, data=-50+(1-np.exp(-np.abs(r)/0.01))*1000*np.exp(-(r-0.5)**2/0.1**2))
            self.assertAlmostEqual(surf1.z_min, 0)
            self.assertAlmostEqual(surf1.z_max, 1000, delta=3e-3)
       
        # FunctionSurface 2D
        #######################

        # smooth function with zero offset
        surf1 = ot.FunctionSurface2D(r=3, func=lambda x, y: x ** 2 + y ** 2)
        self.assertAlmostEqual(surf1.z_min, 0)
        self.assertAlmostEqual(surf1.z_max, 9)
        
        # smooth function with non zero offset
        surf1 = ot.FunctionSurface2D(r=3, func=lambda x, y: 1000 + x ** 2 + y ** 2)
        self.assertAlmostEqual(surf1.z_min, 0)
        self.assertAlmostEqual(surf1.z_max, 9)
       
        def func(x, y):
            r = np.sqrt(x**2 + y**2)
            return -50+(1-np.exp(-np.abs(r)/0.01))*1000*np.exp(-(r-0.5)**2/0.1**2)

        # function with non central peak at r=0.5 with z = 1000
        surf1 = ot.FunctionSurface2D(r=3, func=func)
        self.assertAlmostEqual(surf1.z_min, 0)
        self.assertAlmostEqual(surf1.z_max, 1000, delta=3e-3)
        
        # FunctionSurface 1D
        #######################

        # smooth function with zero offset
        surf1 = ot.FunctionSurface1D(r=3, func=lambda r: r**2)
        self.assertAlmostEqual(surf1.z_min, 0)
        self.assertAlmostEqual(surf1.z_max, 9)
        
        # smooth function with non zero offset
        surf1 = ot.FunctionSurface1D(r=3, func=lambda r: 1000 + r**2)
        self.assertAlmostEqual(surf1.z_min, 0)
        self.assertAlmostEqual(surf1.z_max, 9)
       
        def func(r):
            return -50+(1-np.exp(-np.abs(r)/0.01))*1000*np.exp(-(r-0.5)**2/0.1**2)

        # function with non central peak at r=0.5 with z = 1000
        surf1 = ot.FunctionSurface1D(r=3, func=func)
        self.assertAlmostEqual(surf1.z_min, 0)
        self.assertAlmostEqual(surf1.z_max, 1000, delta=3e-3)

    def test_surface_zmin_zmax_cases(self):

        sfunc = lambda x, y: x
        
        # test z_max, z_min, parax_roc parameter handling
        self.assertRaises(ValueError, ot.FunctionSurface2D, r=2, func=sfunc, z_max=5)
        # ^-- both z_min z_max need to be provided
        self.assertRaises(ValueError, ot.FunctionSurface2D, r=2, func=sfunc, z_min=5)
        # ^-- both z_min, z_max need to be provided

        # z_max, z_min info messages for "Function"
        sfunc = lambda x, y: x**2 + y**2/10
        z_min, z_max = ot.FunctionSurface2D(func=sfunc, r=3).extent[4:]
        surf = ot.FunctionSurface2D(func=sfunc, r=3, z_min=z_min, z_max=z_max)  # a valid call
        self.assertAlmostEqual(z_min, surf.z_min)  # value has been assigned
        self.assertAlmostEqual(z_max, surf.z_max)
        surf = ot.FunctionSurface2D(func=sfunc, r=3, z_min=z_min - 1e-9, z_max=z_max + 1e-9)  # also valid call
        self.assertAlmostEqual(z_min-1e-9, surf.z_min)  # value has been assigned
        self.assertAlmostEqual(z_max+1e-9, surf.z_max)
        surf = ot.FunctionSurface2D(func=sfunc, r=3, z_min=z_min + 1e-3, z_max=z_max - 1e-3)
        # ^-- range warning, values don't get set
        self.assertNotAlmostEqual(z_min+1e-3, surf.z_min)  # value has not been assigned
        self.assertNotAlmostEqual(z_max-1e-3, surf.z_max)
        surf = ot.FunctionSurface2D(func=sfunc, r=3, z_min=z_min + 1e-3, z_max=z_max + 1e-3)
        # ^-- range valid, +offset -> warning, don't set values
        self.assertNotAlmostEqual(z_min+1e-3, surf.z_min)  # value has not been assigned
        self.assertNotAlmostEqual(z_max+1e-3, surf.z_max)
        surf = ot.FunctionSurface2D(func=sfunc, r=3, z_min=z_min - 1e-3, z_max=z_max - 1e-3)
        # ^-- range valid, -offset -> warning, don't set values
        self.assertNotAlmostEqual(z_min-1e-3, surf.z_min)  # value has not been assigned
        self.assertNotAlmostEqual(z_max-1e-3, surf.z_max)
        surf = ot.FunctionSurface2D(func=sfunc, r=3, z_min=z_min, z_max=z_max + 5)
        # ^-- range much larger than measured -> warning, set values
        self.assertAlmostEqual(z_min, surf.z_min)  # value has been assigned
        self.assertAlmostEqual(z_max+5, surf.z_max)

        # messages for type "Data"

        # surface height change due to n=4 polynomial interpolation
        # especially noticeable with noisy data
        data = np.random.sample((300, 300))
        ot.DataSurface2D(data=data, r=3)  
        
        # surface height change due to n=4 polynomial interpolation
        # especially noticeable with noisy data
        data = np.random.sample(300)
        ot.DataSurface1D(data=data, r=3)  
        
        # since the center is not included in the data set (200 is even), the surface height increases a little bit
        Y, X = np.mgrid[-1:1:200j, -1:1:200j]
        data = X**2 + Y**2
        ot.DataSurface2D(data=data, r=1)

    def test_surface_misc(self):

        # sphere projection
        self.assertRaises(ValueError, ot.SphericalSurface(r=2, R=10).sphere_projection, np.ones((3, 3)),
                          projection_method="ABC")
        # ^-- invalid projection method

        # check conversion of normal_sph to normal for a TiltedSurface
        theta = 30
        phi = 12
        tilt = ot.TiltedSurface(r=3, normal_sph=[theta, phi])
        thet, ph = np.deg2rad(theta), np.deg2rad(phi)
        self.assertTrue(np.allclose(tilt.normal - [np.sin(thet)*np.cos(ph), np.sin(thet)*np.sin(ph), np.cos(thet)], 0))

        # coverage tests

        # coverage test: intersection s=[0, 0, 1] with a parabola
        surf = ot.ConicSurface(R=1, k=-1, r=3)
        p = np.random.uniform(-3, 3, size=(1000, 3))
        p[:, 2] = 0
        s = np.tile([0., 0., 1.], (1000, 1)) 
        surf.find_hit(p, s)
        
        # get values of surface, but no points are actually on the surface
        surf = ot.SphericalSurface(r=0.5, R=10)
        x, y = np.ones(1000), np.ones(1000)
        surf.values(x, y)  # no points are on the surface

        # __find_bounds with masked value at center
        sfunc = lambda x, y: np.ones_like(x)
        mask_func = lambda x, y: x**2 + y**2 > 0.5
        ot.FunctionSurface2D(func=sfunc, r=3, mask_func=mask_func)

        # coverage test: case rectangle and slit, too few points for edge
        rect = ot.RectangularSurface(dim=[2, 3])
        self.assertRaises(ValueError, rect.edge, nc=1)
        slit = ot.SlitSurface(dim=[2, 3], dimi=[0.1, 0.1])
        self.assertRaises(ValueError, slit.edge, nc=1)

    def test_surface_function_values(self):

        # these functions do the same as tilted surface class,
        # where we exactly now the behaviour

        def func(x, y, normal):
            mx = -normal[0]/normal[2]
            my = -normal[1]/normal[2]
            return x*mx + y*my
    
        def deriv(x, y, normal):
            mx = -normal[0]/normal[2]
            my = -normal[1]/normal[2]
            return np.full_like(x, mx), np.full_like(y, my)

        def mask(x, y, a):
            return x < a

        normal0 = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
        func0 = lambda x, y: func(x, y, normal0)
        deriv0 = lambda x, y: deriv(x, y, normal0)
        mask0 = lambda x, y: mask(x, y, 1000)

        # call with just the function
        S0 = ot.FunctionSurface2D(r=1, func=func0)

        # call with all function properties provided
        S1 = ot.FunctionSurface2D(r=5, func=func0, deriv_func=deriv0, mask_func=mask0)

        # check function pass-through
        a = 4
        pars = (np.array([1, 0, 0.5]), np.array([0, -1., 0]))
        pars2 = (np.array([[0, -1, -0.5]]), np.array([[0, 0, 1]]))
        self.assertTrue(np.allclose(S1.values(*pars) - func0(*pars), 0))
        self.assertTrue(np.all(S1.mask(*pars) == mask0(*pars)))
        self.assertTrue(np.allclose(S1.normals(*pars) - normal0, 0))

        for pos in [[0, 0, 0], [-1, 2, 0.5]]:

            normal = [1/np.sqrt(2), 0 , 1/np.sqrt(2)]
            S2 = ot.FunctionSurface2D(r=5, func=func, func_args=dict(normal=normal), deriv_func=deriv,
                                      deriv_args=dict(normal=normal), mask_func=mask, mask_args=dict(a=a+pos[2]))
            S2.move_to(pos)

            # relative coordinates from center
            pars0 = (pars[0]-pos[0], pars[1]-pos[1])
            pars20 = (pars2[0]-pos, pars2[1])

            # check function pass-through
            self.assertTrue(np.allclose(S2.values(*pars) - func(*pars0, normal) - pos[2], 0))
            self.assertTrue(np.all(S2.mask(*pars) == mask(*pars0, a)))
            self.assertTrue(np.allclose(S2.normals(*pars) - normal, 0))

    def test_surface_function_exceptions(self):
       
        func = lambda x, y: x

        # check type checks in init
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=np.array([1, 2]))  # invalid func type
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=None)  # func can't be none
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=func, mask_func=1)  # invalid type
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=func, deriv_func=1)  # invalid type
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=func, func_args=1)  # invalid type
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=func, deriv_args=1)  # invalid type
        self.assertRaises(TypeError, ot.FunctionSurface2D, r=2, func=func, mask_args=1)  # invalid type

        # check assertion errors when wrong types are returned from user provided functions
        x = np.linspace(-1, 1, 1000)
        y = np.linspace(-1, 1, 1000)
        self.assertRaises(RuntimeError, ot.FunctionSurface2D, r=2, func=lambda x, y: x > y)  # must return floats
        self.assertRaises(RuntimeError, ot.FunctionSurface2D, r=2, func=lambda x, y: 1)  # must return np.ndarray
        self.assertRaises(RuntimeError, ot.FunctionSurface2D, r=2, func=lambda x, y: x, mask_func=lambda x, y: x + y)
        # ^-- must return bools
        self.assertRaises(RuntimeError, ot.FunctionSurface2D, r=2, func=lambda x, y: x, mask_func=lambda x, y: 1)
        # ^-- must return np.ndarray
        self.assertRaises(RuntimeError, ot.FunctionSurface2D(r=2, func=lambda x, y: x,
                                                             deriv_func=lambda x, y: (x, 1)).normals, x, y)
        # ^-- must return two np.ndarray
        self.assertRaises(RuntimeError, ot.FunctionSurface2D(r=2, func=lambda x, y: x,
                                                             deriv_func=lambda x, y: (1, y)).normals, x, y)
        # ^-- must return two np.ndarray
        self.assertRaises(RuntimeError, ot.FunctionSurface2D(r=2, func=lambda x, y: x,
                                                             deriv_func=lambda x, y: (x > y, y)).normals, x, y)
        # ^-- must return two np.ndarray with float values
        self.assertRaises(RuntimeError, ot.FunctionSurface2D(r=2, func=lambda x, y: x,
                                                             deriv_func=lambda x, y: (y, x > y)).normals, x, y)
        # ^-- must return two np.ndarray with float values

        # FunctionSurface1D
        self.assertRaises(RuntimeError, ot.FunctionSurface1D, r=2, func=lambda r: x > y)  # must return floats
        self.assertRaises(RuntimeError, ot.FunctionSurface1D, r=2, func=lambda r: 1)  # must return np.ndarray
        self.assertRaises(RuntimeError, ot.FunctionSurface1D, r=2, func=lambda r: r, mask_func=lambda r: r)
        # ^-- must return bools
        self.assertRaises(RuntimeError, ot.FunctionSurface1D, r=2, func=lambda r: r, mask_func=lambda r: 1)
        # ^-- must return np.ndarray
        self.assertRaises(RuntimeError, ot.FunctionSurface1D(r=2, func=lambda r: r,
                                                             deriv_func=lambda r: 0).normals, x, y)
        # ^-- must return a np.ndarray
        self.assertRaises(RuntimeError, ot.FunctionSurface1D(r=2, func=lambda r: r,
                                                             deriv_func=lambda r: r > 0).normals, x, y)
        # ^-- must return a np.ndarray with float values


    @pytest.mark.os
    def test_surface_numerical_precision(self):

        # this function checks the numerical precision

        surf_f_1 = lambda x, y: z0 + (x**2 + y**2)/R/2  # parabola = conic section with k=-1
        surf_f_2 = lambda x, y: z0 + R - np.sqrt(R**2 - (x**2 + y**2))  # sphere, k=0

        # actual derivative values
        def n_conic(x, y, rho, k):
            # see surface class for details on calculation

            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            n_r = -rho * r  / np.sqrt(1 - k* rho**2 * r**2 )

            n = np.zeros((x.shape[0], 3), dtype=np.float64, order='F')
            n[:, 0] = n_r*np.cos(phi)  
            n[:, 1] = n_r*np.sin(phi)  
            n[:, 2] = np.sqrt(1 - n_r**2)

            return n

        # check precision for different surfaces, radii, offsets and regions
        for k, surf_f in zip([-1, 0], [surf_f_1, surf_f_2]):
            for R in [0.1, 10, 10000]:
                for z0 in [0, -80, -10000]:
                    for fr in [0.002, 0.7]:

                        r = fr*R

                        # compare estimated normals with actual, FunctionSurface2D
                        surf1 = ot.FunctionSurface2D(func=surf_f, r=r)
                        x = np.linspace(0, r, 100)
                        y = np.zeros_like(x)
                        n_is = surf1.normals(x, y)
                        n_should = n_conic(x, y, 1/R, k)
                        self.assertTrue(np.allclose(n_is - n_should, 0, atol=1e-7, rtol=0))
                        
                        # compare estimated normals with actual, FunctionSurface1D
                        surf11 = ot.FunctionSurface1D(func=lambda r: surf_f(r, np.zeros_like(r)), r=r,)
                        x = np.linspace(0, r, 100)
                        y = np.zeros_like(x)
                        n_is = surf11.normals(x, y)
                        n_should = n_conic(x, y, 1/R, k)
                        self.assertTrue(np.allclose(n_is - n_should, 0, atol=1e-7, rtol=0))

                        # type "Data" for different sample sizes
                        for N in [50, 51, 400, 401]:
                            for i in range(2):
                                if not i:  # 2D data surface
                                    Y, X = np.mgrid[-r:r:N*1j, -r:r:N*1j]
                                    data = surf_f(X.flatten(), Y.flatten()).reshape(X.shape)
                                    surf2 = ot.DataSurface2D(data=data, r=r)
                                else:  # 1D data surface
                                    x2 = np.linspace(0, r, N)
                                    y2 = np.zeros_like(x2)
                                    data = surf_f(x2, y2)
                                    surf2 = ot.DataSurface1D(data=data, r=r)
                           
                                # compare numeric normals to actual ones
                                # data normals are more unprecise, since enough samples are needed to define the shape
                                n_is = surf2.normals(x, y)
                                n_should = n_conic(x, y, 1/R, k)
                                self.assertTrue(np.allclose(n_is - n_should, 0, atol=1e-4, rtol=0))

    def test_flip_surface(self):

        Y, X = np.mgrid[-1:1:100j, -1:1:100j]

        # different surface types
        Ss = [
              # surface planar types
              ot.CircularSurface(r=2),
              ot.RingSurface(r=3, ri=0.2),
              ot.RectangularSurface(dim=[2, 3]),
              ot.SlitSurface(dim=[2, 3], dimi=[0.01, 0.1]),
              ot.DataSurface2D(r=2, data=np.zeros((100, 100))),
              ot.FunctionSurface2D(r=2, func=lambda x, y: np.zeros_like(x)),
              # in built analytical types
              ot.SphericalSurface(r=2, R=10),
              ot.SphericalSurface(r=2, R=-10),
              ot.ConicSurface(r=2, R=10, k=0.5),
              ot.ConicSurface(r=2, R=-10, k=-2),
              ot.TiltedSurface(r=2, normal=[0, np.cos(np.pi/5), np.sin(np.pi/5)]),
              ot.TiltedSurface(r=3, normal=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
              ot.AsphericSurface(r=2, R=10, k=0.5, coeff=[0, 0.002, 0.00001]),
              ot.AsphericSurface(r=2, R=-10, k=0.5, coeff=[-0.02, 0.00001]),
              # function surfaces 2D
              ot.FunctionSurface2D(r=5, func=lambda x, y: x ** 2 + y ** 2),
              ot.FunctionSurface2D(r=5, func=lambda x, y: x + 2 * y,
                                   deriv_func=lambda x, y: (np.ones_like(x), np.full_like(y, 2))),
              ot.FunctionSurface2D(r=5, func=lambda x, y: 1 - x ** 2 + y ** 2),
              ot.FunctionSurface2D(r=5, func=lambda x, y: 1 - x ** 2 + 0.001 * y ** 3),
              ot.FunctionSurface2D(r=5, func=lambda x, y: x ** 2 + y ** 2, parax_roc=2),
              ot.FunctionSurface2D(r=4, func=lambda x, y, a: a * x + y ** 2, func_args=dict(a=2)),
              # function surfaces 1D
              ot.FunctionSurface1D(r=5, func=lambda r: r**2),
              ot.FunctionSurface1D(r=5, func=lambda r: r, deriv_func=lambda r: np.ones_like(r)),
              ot.FunctionSurface1D(r=5, func=lambda r: r**2, parax_roc=0.5),
              ot.FunctionSurface1D(r=4, func=lambda r, a: a * r** 2, func_args=dict(a=2)),
              # Data Surfaces
              ot.DataSurface1D(r=3, data=0.5 + np.linspace(0, 1, 100)**2 - np.linspace(0, 1, 100)),
              ot.DataSurface1D(r=3, data=0.5 + np.linspace(0, 1, 100)**2 - np.linspace(0, 1, 100), parax_roc=-1),
              ot.DataSurface2D(r=3, data=X**2 - Y),
              ot.DataSurface2D(r=3, data=1+0.001*Y/(X+10), parax_roc=0.3),
             ]

        pos = np.array([9, 7, 0.12])
        Y, X = np.mgrid[-1:1:10j, -1:1:10j]
        x, y = X.flatten(), Y.flatten()

        for i, S in enumerate(Ss):

            S.move_to(pos)
            Sr = S.copy()
            Sr.flip()
            Srr = Sr.copy()
            Srr.flip()  # now double flip
        
            self.assertTrue(np.allclose(Sr.pos - S.pos, 0, atol=1e-10))
            self.assertTrue(np.allclose(Srr.pos - S.pos, 0, atol=1e-10))

            val0 = S.values(x + pos[0], y + pos[1]) - pos[2]
            norm0 = S.normals(x + pos[0], y + pos[1])
            val1 = Sr.values(x+pos[0], pos[1]-y) - pos[2] # the y minus is important
            norm1 = Sr.normals(x+pos[0], pos[1]-y)  # the minus sign is important
            val2 = Srr.values(x+pos[0], y+pos[1]) - pos[2] # double flipped
            self.assertTrue(np.allclose(val0 + val1, 0))  # check flipped values
            self.assertTrue(np.allclose(val0 - val2, 0))  # double reversion leads to initial values
            self.assertTrue(np.allclose(norm0 - norm1 * np.array([-1, 1, 1])[None, None, :], 0)) 
            # ^-- normal x component gets negated
            # [x, y, z] rotated around [1, 0, 0] by pi is [x, -y, -z],
            # this negated (=pointing towards +z) is [-x, y, z]

            # check paraxial curvature circles
            if not S.is_flat() and S.rotational_symmetry and S.parax_roc is not None:
                self.assertEqual(S.parax_roc, -Sr.parax_roc)
                self.assertEqual(S.parax_roc, Srr.parax_roc)

            # check that z_min, z_max were assigned correctly
            _, _, Z = Sr.plotting_mesh(100)
            self.assertTrue(np.nanmin(Z) >= Sr.z_min - 10*Sr.C_EPS)
            self.assertTrue(np.nanmax(Z) <= Sr.z_max + 10*Sr.C_EPS)

        # flip slit and rectangle
        rect = ot.RectangularSurface(dim=[5, 4])
        slit = ot.SlitSurface(dim=[5, 4], dimi=[0.2, 0.3])
        for obj in [rect, slit]:
            obj.rotate(-78)
            obj2 = obj.copy()
            obj.flip()
            self.assertEqual(-obj2._angle, obj._angle)

    @staticmethod
    def relative_rotate(x, y, angle):
        xr = x*np.cos(angle) - y*np.sin(angle)
        yr = x*np.sin(angle) + y*np.cos(angle)
        return xr, yr

    def test_rotate_symmetric_surfaces(self):

        # rotating surfaces, that have rotational symmetry should change absolutely nothing in the object

        Ss = [
              # surfaces
              ot.CircularSurface(r=2),
              ot.RingSurface(r=3, ri=0.2),
              ot.SphericalSurface(r=2, R=10),
              ot.SphericalSurface(r=2, R=-10),
              ot.ConicSurface(r=2, R=10, k=0.5),
              ot.ConicSurface(r=2, R=-10, k=-2),
              ot.DataSurface1D(r=2, data=np.linspace(0, 1, 100)),
              ot.FunctionSurface1D(r=2, func=lambda r: r**2),
              ot.AsphericSurface(r=2, R=10, k=0.5, coeff=[0, 0.002, 0.00001]),
              ot.AsphericSurface(r=2, R=-10, k=0.5, coeff=[-0.02, 0.00001]),
             ]

        for SSi in Ss:
            for i in range(2):
                cr0 = SSi.crepr()
                SSi.rotate(20.132)
                cr1 = SSi.crepr()
                self.assertEqual(cr0, cr1)
                SSi.move_to([456, -2.567, 0.22])

    def test_rotate_non_symmetrical_surfaces(self):

        def func(x, y, normal):
            mx = -normal[0]/normal[2]
            my = -normal[1]/normal[2]
            return x*mx + y*my
        
        normal = [0, 1/np.sqrt(2), 1/np.sqrt(2)]

        pos0 = np.array([-1.5, 0.2, 5])
        angle = 27.1328
        
        Ss = [
                ot.FunctionSurface2D(r=3, func=lambda x, y: (x ** 2 + y ** 2 / 3) / 10),
                ot.FunctionSurface2D(r=3, func=lambda x, y: (x ** 2 + y ** 2 / 3) / 10, mask_func=lambda x, y: y < 0.5),
                ot.FunctionSurface2D(r=3, func=func, func_args=dict(normal=normal), ),
                ot.FunctionSurface2D(r=3, func=func, func_args=dict(normal=normal), mask_func=lambda x, y: y < 0.5),
                ot.TiltedSurface(r=3, normal=[0.5, 1, 0.9]),
                ot.RectangularSurface(dim=[5, 4]),
                ot.DataSurface2D(r=3, data=np.random.uniform(0, 1, (200, 200))),
            ]

        x = np.random.uniform(-1, 1, 100) + pos0[0]
        y = np.random.uniform(-1, 1, 100) + pos0[1]

        s = np.random.uniform(-0.05, 0.05, (100, 3))
        s[:, 2] = 1
        s = misc.normalize(s)
        p = np.column_stack((x, y, np.full_like(x, -20)))

        for S in Ss:
            for i in range(3):  # rotate three times and flip two times
                S.move_to(pos0)
                z0 = S.values(x, y)
                n0 = S.normals(x, y)
                m0 = S.mask(x, y)
                ph0, ish0, _ = S.find_hit(p, s)

                S.rotate(angle)
                xr, yr = self.relative_rotate(x-pos0[0], y-pos0[1], np.deg2rad(angle))  # rotate coordinates
                x2, y2 = xr+pos0[0], yr+pos0[1]
                z1 = S.values(x2, y2)
                n1 = S.normals(x2, y2)
                n0r = n0.copy()
                n0r[:, 0], n0r[:, 1] = self.relative_rotate(n0[:, 0], n0[:, 1], np.deg2rad(angle))
                m1 = S.mask(x2, y2)
                self.assertTrue(np.allclose(z0 - z1, 0))
                self.assertTrue(np.allclose(n0r - n1, 0))
                self.assertTrue(np.all(m0 == m1))

                p2 = p.copy()
                p2[:, 0], p2[:, 1] = self.relative_rotate(p2[:, 0] - pos0[0], p2[:, 1] - pos0[1], np.deg2rad(angle))
                p2[:, 0] += pos0[0]
                p2[:, 1] += pos0[1]

                s2 = s.copy()
                s2[:, 0], s2[:, 1] = self.relative_rotate(s2[:, 0], s2[:, 1], np.deg2rad(angle))

                ph1, ish1, _ = S.find_hit(p2, s2)
                ph1[:, 0], ph1[:, 1] = self.relative_rotate(ph1[:, 0] - pos0[0], ph1[:, 1] - pos0[1],
                                                            -np.deg2rad(angle))
                ph1[:, 0] += pos0[0]
                ph1[:, 1] += pos0[1]
                
                self.assertTrue(np.all(ish0 == ish1))
                self.assertTrue(np.allclose(ph0 - ph1, 0))

                S.flip()

    def test_rotate_rectangular_slit_surface(self):

        rect = ot.RectangularSurface(dim=[2, 3])
        slit = ot.SlitSurface(dim=[2, 3], dimi=[0.05, 0.15])
        
        for obj0 in [rect, slit]:

            obj = obj0.copy()
            pos0 = [5, -3, 12]
            obj.move_to(pos0)

            x = np.random.uniform(-2, 2, 100)
            y = np.random.uniform(-2, 2, 100)

            # edge, mesh and mask or an unrotated rectangle/slit
            X0, Y0, Z0 = obj.plotting_mesh(100)
            x0, y0, z0 = obj.edge(100)
            m0 = obj.mask(x, y)

            angle = -82.46
            obj.rotate(angle)

            # get mesh, rotate mesh back
            X1, Y1, Z1 = obj.plotting_mesh(100)
            X1r, Y1r = self.relative_rotate(X1 - pos0[0], Y1 - pos0[1], -np.deg2rad(angle))
            X1r, Y1r = X1r + pos0[0], Y1r + pos0[1]

            # get edge, rotate edge back
            x1, y1, z1 = obj.edge(100)
            x1r, y1r = self.relative_rotate(x1 - pos0[0], y1 - pos0[1], -np.deg2rad(angle))
            x1r, y1r = x1r + pos0[0], y1r + pos0[1]

            # rotate coordinates to math rotation, get mask
            xr, yr = self.relative_rotate(x - pos0[0], y - pos0[1], -np.deg2rad(angle))
            m1 = obj.mask(xr + pos0[0], yr + pos0[1])

            # compare values
            self.assertTrue(np.allclose(x1r - x0 + y1r - y0, 0)) 
            self.assertTrue(np.allclose(X1r - X0 + Y1r - Y0, 0)) 
            self.assertTrue(np.all(m1 == m0))

            # check extent
            obj = obj0
            self.assertTrue(np.allclose(np.array(obj.extent) - [-1, 1, -1.5, 1.5, 0, 0], 0))
            obj.rotate(90)
            self.assertTrue(np.allclose(np.array(obj.extent) - [-1.5, 1.5, -1, 1, 0, 0], 0))
            obj.rotate(-180)
            self.assertTrue(np.allclose(np.array(obj.extent) - [-1.5, 1.5, -1, 1, 0, 0], 0))
            obj.rotate(45)
            isq = 1/np.sqrt(2)
            self.assertTrue(np.allclose(np.array(obj.extent) - [-2.5*isq, 2.5*isq, -2.5*isq, 2.5*isq, 0, 0], 0))

if __name__ == '__main__':
    unittest.main()
