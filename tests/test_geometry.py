#!/bin/env python3

import sys
sys.path.append('.')

import doctest
import unittest
import numpy as np
import numexpr as ne
import warnings
import pytest

import optrace.tracer.misc as misc
import optrace.tracer.color as color

import optrace as ot
from optrace.tracer.geometry.element import Element
from optrace.tracer.base_class import BaseClass as BaseClass

import contextlib  # redirect stdout



class GeometryTests(unittest.TestCase):

    def test_point(self):
        # init
        p = ot.Point()

        # test behavior
        for pos in [[1, -5, 10], [0, 0, 7]]:
            # check moving
            a = np.array(pos)
            p.move_to(a)
            self.assertTrue(np.allclose(p.pos - a, 0))

            # check extent
            self.assertTrue(np.allclose(p.extent - a.repeat(2), 0))
            
            # check random positions
            for N in [1, 10, 107]:
                rpos = p.get_random_positions(N)
                self.assertEqual(rpos.shape[0], N)
                self.assertTrue(np.allclose(rpos - p.pos, 0))
        
        # test lock
        self.assertRaises(AttributeError, p.__setattr__, "r46546", 4)  # _new_lock active
        self.assertRaises(RuntimeError, p.__setattr__, "pos", [0, 1, 0])  # object locked
        def setArrayElement():
            ot.Point().pos[0] = 1.
        self.assertRaises(ValueError, setArrayElement)  # array elements read-only
    
    def test_line(self):

        # test inits
        self.assertRaises(ValueError, ot.Line, r=-1)  # radius must be > 0
        self.assertRaises(ValueError, ot.Line, r=0) # radius must be > 0
        self.assertRaises(TypeError, ot.Line, r=[])  # invalid radius type
        self.assertRaises(TypeError, ot.Line, angle=[])  # invalid angle type

        # test behavior
        for r, ang in zip([0.1, 10, 17.8992], [0, 10, 45, 70, 89.9, 90, 359.9]):
            l = ot.Line(r=r, angle=ang)

            for pos in [[1, -5, 10], [0, 0, 7]]:
                # check moving
                a = np.array(pos)
                l.move_to(a)
                self.assertTrue(np.allclose(l.pos - a, 0))

                # check extent
                ang_ = ang/180*np.pi
                ext = a.repeat(2) + [-r*np.cos(ang_), r*np.cos(ang_), -r*np.sin(ang_), r*np.sin(ang_), 0, 0]
                self.assertTrue(np.allclose(l.extent - ext, 0))
                
                # check random positions
                for N in [1, 10, 107]:
                    rpos = l.get_random_positions(N)
                    self.assertEqual(rpos.shape[0], N)

                    # radial position correct
                    self.assertTrue(np.all(rpos[:, 2] == l.z_max))
                    rr = np.sqrt((rpos[:, 0] - pos[0])**2 + (rpos[:, 1] - pos[1])**2)
                    self.assertTrue(np.all(rr <= r + 1e-11))

                    # angle correct
                    # angle should be ang or ang+pi, check this by comparing sin^2, which is pi periodic
                    rang = np.arctan2(rpos[:, 1] - pos[1], rpos[:, 0] - pos[0])
                    self.assertTrue(np.allclose(np.sin(rang)**2 - np.sin(ang_)**2, 0))
        
        # test lock
        self.assertRaises(AttributeError, l.__setattr__, "r46546", 4)  # _new_lock active
        self.assertRaises(RuntimeError, l.__setattr__, "pos", [0, 1, 0])  # object locked
        def setArrayElement():
            ot.Line().pos[0] = 1.
        self.assertRaises(ValueError, setArrayElement)  # array elements read-only

    def test_marker(self):
        
        # init exceptions
        self.assertRaises(TypeError, ot.Marker, 1, [0, 0, 0])  # invalid desc
        self.assertRaises(TypeError, ot.Marker, "Test", 1)  # invalid pos type
        self.assertRaises(ValueError, ot.Marker, "Test", [np.inf, 0, 0])  # inf in pos
        self.assertRaises(TypeError, ot.Marker, "Test", [0, 0, 0], marker_factor="yes")  # invalid factor type
        self.assertRaises(TypeError, ot.Marker, "Test", [0, 0, 0], text_factor="yes")  # invalid factor type

        # check init assignments and extent
        for desc, pos in zip(["a", "", "afhajk"], [[0, 1, 0], [5, 0, 0], [7, 8, 9]]):
            M = ot.Marker(desc, pos)
            self.assertEqual(M.desc, desc)
            self.assertTrue(np.all(M.pos == pos))
            self.assertTrue(np.all(M.pos.repeat(2) == M.extent))

        # moving and setting pos does the same
        pos2, pos3 = [5, -7, 9], [12, 0, 8]
        M.move_to(pos2)
        self.assertTrue(np.all(M.pos == pos2))
        M.move_to(pos3)
        self.assertTrue(np.all(M.pos == pos3))

        # test lock
        self.assertRaises(AttributeError, M.__setattr__, "r46546", 4)  # _new_lock active

    def test_surface_init_exceptions(self):

        # type errors
        self.assertRaises(TypeError, ot.Surface)  # no type
        self.assertRaises(TypeError, ot.Surface, "Function", func=lambda x, y: x + y)  # func is no ot.SurfaceFunction
        self.assertRaises(TypeError, ot.Surface, "Circle", r=2, normal=True)  # invalid normal type
        self.assertRaises(TypeError, ot.Surface, 4654)  # invalid surface_type type
        self.assertRaises(TypeError, ot.Surface, r=[])  # invalid r type
        self.assertRaises(TypeError, ot.Surface, ri=[])  # invalid ri type
        self.assertRaises(TypeError, ot.Surface, pos=True)  # invalid pos type
        self.assertRaises(TypeError, ot.Surface, k=[])  # invalid k type
        self.assertRaises(TypeError, ot.Surface, R=[])  # invalid R type
        self.assertRaises(TypeError, ot.Surface, dim=True)  # invalid dim type
        self.assertRaises(TypeError, ot.Surface, z_min=[])  # invalid z_min type
        self.assertRaises(TypeError, ot.Surface, z_max=[])  # invalid z_max type
        self.assertRaises(TypeError, ot.Surface, data=True)  # invalid data type
        self.assertRaises(TypeError, ot.Surface, curvature_circle=[])  # invalid curvature type

        # value errors
        self.assertRaises(ValueError, ot.Surface, "Circle", r=2, normal=[0.5, 0, -np.sqrt(1-0.5**2)])  # n_z negative
        self.assertRaises(ValueError, ot.Surface, "Circle", r=2, normal=[1, 0, 0])  # n_z zero
        self.assertRaises(ValueError, ot.Surface, "ABC")  # invalid type
        self.assertRaises(ValueError, ot.Surface, "Conic", r=3.2, R=-7, k=5)  # r too large for surface
        self.assertRaises(ValueError, ot.Surface, "Sphere", r=-1)  # size negative
        self.assertRaises(ValueError, ot.Surface, "Sphere", R=0)  # R zero
        self.assertRaises(ValueError, ot.Surface, "Sphere", R=np.inf)  # R inf
        self.assertRaises(ValueError, ot.Surface, "Sphere", R=-np.inf)  # R -inf
        self.assertRaises(ValueError, ot.Surface, "Ring", r=1, ri=-0.2)  # size negative
        self.assertRaises(ValueError, ot.Surface, "Ring", r=1, ri=0)  # ri zero
        self.assertRaises(ValueError, ot.Surface, "Circle", r=0)  # r zero
        self.assertRaises(ValueError, ot.Surface, "Ring", r=1, ri=1.4)  # ri larger than r
        self.assertRaises(ValueError, ot.Surface, "Rectangle", dim=[-5, 5])  # size negative
        self.assertRaises(ValueError, ot.Surface, "Rectangle", dim=[5, np.inf])  # size non finite
        self.assertRaises(ValueError, ot.Surface, "Data", data=np.array([[1, 2], [3, 4]]))  # data needs to be at least 50x5
        self.assertRaises(ValueError, ot.Surface, "ABC")  # invalid surface type
        self.assertRaises(ValueError, ot.Surface, "Data", data=np.array([[1, 2, 3], [4, 5, 6]]))  # matrix not square
        self.assertRaises(ValueError, ot.Surface, "Data", data=np.array([[1, np.nan], [4, 5]]))  # nan in data
        
        # runtime errors
        self.assertRaises(RuntimeError, ot.Surface, "Sphere", r=2, normal=[0.5, 0, np.sqrt(1-0.5**2)])  # normal for Circles only

    def example_sfunc(self):
        # SurfaceFunction with all funcs defined
        # note the offset of 1, this is important to test the coordinate transformations
        def hit_func(p, s):
            t = (1 - p[:, 2])/s[:, 2]
            return p + s*t[:, np.newaxis]

        return ot.SurfaceFunction(lambda x, y: np.ones_like(x), hit_func=hit_func,
                                   deriv_func=lambda x, y: (np.zeros_like(x), np.zeros_like(x)), 
                                   mask_func=lambda x, y: x+y < 2)

    def test_surface_behavior(self):

        sfunc = self.example_sfunc()

        S = [ot.Surface("Circle", r=4),
             ot.Surface("Sphere", r=5, R=10),
             ot.Surface("Conic", r=3.2, R=-7, k=2),
             ot.Surface("Rectangle", dim=[5, 6]),
             ot.Surface("Function", func=ot.SurfaceFunction(lambda x, y: x), r=2),  # func without some properties
             ot.Surface("Function", func=sfunc, r=2),  # func with all properties
             ot.Surface("Circle", r=4, normal=[0.5, 0, np.sqrt(1-0.5**2)]),
             ot.Surface("Data", r=2, data=1+np.random.sample((50, 50)), silent=True),
             ot.Surface("Ring", r=3, ri=0.2),  # ring with ri < 0.5r
             ot.Surface("Ring", r=3, ri=2)]  # ring with ri > 0.5r gets plotted differently

        x = np.linspace(-0.7, 5.3, 1000)
        y = np.linspace(-8, -2, 1000)
        p = np.random.uniform(-2, -1, size=(1000, 3))
        s = np.random.uniform(-1, 1, size=(1000, 3))
        s /= np.linalg.norm(s, axis=1)[:, np.newaxis]  # normalize
        s[:, 2] = np.abs(s[:, 2])  # only positive s_z

        # check if other functions throw
        for i, Si in enumerate(S):
            with self.subTest(i=i, Surface=Si.surface_type):
                
                self.assertTrue(Si.extent[0] < Si.extent[1] and Si.extent[2] < Si.extent[3]\
                        and Si.extent[4] <= Si.extent[5])

                if Si.has_rotational_symmetry():
                    Si.curvature_circle

                # thickness expressions
                self.assertAlmostEqual(Si.d, Si.dn + Si.dp, 8)

                # check that surface center is at pos[2]
                self.assertAlmostEqual(Si.get_values(np.array([Si.pos[0]]), np.array([Si.pos[1]]))[0], Si.pos[2])

                # moving working correctly
                pos_new = np.array([2.3, -5, 10])
                Si.move_to(pos_new)
                self.assertTrue(np.all(pos_new == Si.pos))

                z = Si.get_values(x, y)
                Si.get_mask(x, y)
                Si.get_edge(nc=100)
                Si.get_normals(x, y)

                # check if moving actually moved the origin of the function
                self.assertEqual(Si.get_values(np.array([pos_new[0]]), np.array([pos_new[1]]))[0], pos_new[2])

                # check values
                if Si.no_z_extent():
                    self.assertTrue(np.all(z == pos_new[2]))
                elif Si is S[1]:  # Sphere
                    self.assertAlmostEqual(z[0], 10.94461486)
                    self.assertAlmostEqual(z[500], 10.0000009)
                elif Si is S[2]:  # Conic
                    self.assertAlmostEqual(z[0], 10.)  # == z_max since it's outside
                    self.assertAlmostEqual(z[300], 9.78499742)

                if Si.has_hit_finding():
                    p_hit, is_hit = Si.find_hit(p, s)

                    # check that all hitting rays have the correct z-value at the surface
                    z_hit = Si.get_values(p_hit[is_hit, 0], p_hit[is_hit, 1])
                    self.assertTrue(np.allclose(p_hit[is_hit, 2] - z_hit, 0))
                    
                    # check that not hitting rays have z = z_max assigned
                    self.assertTrue(np.allclose(p_hit[~is_hit, 2], Si.z_max))

                    # check if p_hit is actually a valid position on the ray
                    t = (p_hit[:, 2] - p[:, 2])/s[:, 2]
                    p_ray = p + s*t[:, np.newaxis]
                    self.assertTrue(np.allclose(p_ray[:, 0] - p_hit[:, 0], 0))
                    self.assertTrue(np.allclose(p_ray[:, 1] - p_hit[:, 1], 0))
                    self.assertTrue(np.allclose(p_ray[:, 2] - p_hit[:, 2], 0))

                # random positions only for planar types
                if Si.no_z_extent():
                    # check that generated positions are all inside surface and have correct values
                    p_ = Si.get_random_positions(N=10000)
                    m_ = Si.get_mask(p_[:, 0], p_[:, 1])
                    z_ = Si.get_values(p_[:, 0], p_[:, 1])
                    self.assertTrue(np.all(m_))
                    self.assertTrue(np.allclose(z_-p_[:, 2], 0))

                # check mask values. Especially surface edge, which by definition is valid
                if Si.surface_type == "Rectangle":
                    self.assertTrue(Si.get_mask([Si.pos[0]], [Si.pos[1]])[0])
                    self.assertTrue(Si.get_mask([Si.extent[0]], [Si.extent[2]])[0])
                    self.assertTrue(Si.get_mask([Si.extent[0]], [Si.extent[3]])[0])
                    self.assertTrue(Si.get_mask([Si.extent[1]], [Si.extent[3]])[0])
                    self.assertTrue(Si.get_mask([Si.extent[1]], [Si.extent[3]])[0])
                    self.assertFalse(Si.get_mask([Si.extent[1]+1], [Si.pos[1]])[0])
                else:
                    if Si.surface_type == "Function" and Si.func.mask_func is not None:
                        # special check for one of the SurfaceFunction Surfaces
                        self.assertTrue(Si.get_mask(np.array([Si.pos[0]]), np.array([Si.pos[1]]))[0])
                        self.assertFalse(Si.get_mask(np.array([Si.pos[0]+1.5]), np.array([Si.pos[1]+1.5]))[0])
                    else:
                        self.assertTrue(Si.get_mask(np.array([Si.extent[1]]), np.array([Si.pos[1]]))[0])
                        self.assertFalse(Si.get_mask(np.array([Si.extent[1]+1]), np.array([Si.pos[1]]))[0])

                    if Si.surface_type == "Ring":
                        self.assertFalse(Si.get_mask(np.array([Si.pos[0]]), np.array([Si.pos[1]]))[0])
                        self.assertTrue(Si.get_mask(np.array([Si.pos[0]+Si.ri]), np.array([Si.pos[1]]))[0])

                # check get_edge
                for nc in [22, 23, 51, 100, 10001]:
                    x_, y_, z_ = Si.get_edge(nc)
                    
                    # check shapes
                    self.assertEqual(x_.shape[0], nc)
                    self.assertEqual(y_.shape[0], nc)
                    self.assertEqual(z_.shape[0], nc)

                    # check values
                    z2 = Si.get_values(x_, y_)
                    self.assertTrue(np.allclose(z_-z2, 0))

                # check get_plotting_mesh
                for N in [10, 11, 25, 200]:
                    X, Y, Z = Si.get_plotting_mesh(N)
                    
                    # check shapes
                    self.assertEqual(X.shape, (N, N))
                    self.assertEqual(Y.shape, (N, N))
                    self.assertEqual(Z.shape, (N, N))
                    
                    # check if values are masked correctly
                    m2 = Si.get_mask(X.flatten(), Y.flatten())
                    self.assertTrue(np.all(m2 == np.isfinite(Z.flatten())))

                    # check if values are calculated correctly
                    z2 = Si.get_values(X.flatten(), Y.flatten())
                    valid = np.isfinite(Z.flatten())
                    self.assertTrue(np.allclose(Z.flatten()[valid]-z2[valid], 0))
                   
                    # check if values are in relevant area of circle
                    if Si.surface_type != "Rectangle":
                        # no points outside circle area
                        R2 = (X-Si.pos[0])**2 + (Y-Si.pos[1])**2
                        self.assertTrue(np.all(R2 < (Si.r + 10*Si.N_EPS)**2))

                        # no points in empty region of ring
                        if Si.surface_type == "Ring":
                            self.assertTrue(np.all(R2 > (Si.ri - 10*Si.N_EPS)**2))

    def test_surface_misc_exceptions(self):
        # no random positions defined for this type 
        self.assertRaises(AssertionError, ot.Surface("Sphere", r=1).get_random_positions, 100)  

        # test object lock
        S = ot.Surface("Circle")
        self.assertRaises(AttributeError, S.__setattr__, "r46546", 4)  # _new_lock active
        self.assertRaises(RuntimeError, S.__setattr__, "r", 4)  # object locked
        self.assertRaises(RuntimeError, S.__setattr__, "dim", np.array([4, 5]))  # object locked

        # test array lock
        self.assertRaises(RuntimeError, S.__setattr__, "pos", np.array([4, 5]))  # object locked
        
        def setArrayElement():
            ot.Surface("Circle").pos[0] = 1.
        self.assertRaises(ValueError, setArrayElement)  # array elements read-only

        self.assertRaises(ValueError, ot.Surface("Conic", r=3, R=-10, k=1).get_plotting_mesh, 5)  # N < 10
        self.assertRaises(ValueError, ot.Surface("Conic", r=3, R=-10, k=1).get_edge, 5)  # nc < 20

    def test_surface_curvature_circle(self):
        # curvature_circle tests
        self.assertAlmostEqual(ot.Surface("Sphere", R=10).curvature_circle, 10)
        self.assertAlmostEqual(ot.Surface("Conic", R=-5.897).curvature_circle, -5.897)
        self.assertAlmostEqual(ot.Surface("Circle").curvature_circle, np.inf)
        self.assertAlmostEqual(ot.Surface("Ring").curvature_circle, np.inf)
        self.assertAlmostEqual(ot.Surface("Rectangle").curvature_circle, np.inf)
        self.assertEqual(ot.Surface("Data", data=np.ones((50, 50))).curvature_circle, None)
        self.assertAlmostEqual(ot.Surface("Data", data=np.ones((50, 50)), curvature_circle=5).curvature_circle, 5)  # (factually wrong)
        surf_f = ot.SurfaceFunction(func=lambda x, y: (x**2 + y**2)/1000)
        self.assertEqual(ot.Surface("Function", func=surf_f, r=2).curvature_circle, None)
        self.assertEqual(ot.Surface("Function", func=surf_f, r=2, curvature_circle=5).curvature_circle, 5)  # (factually wrong)

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
                    surf = ot.Surface("Sphere", r=0.999*R, R=R*sign)
                    surf.move_to([x0, y0, z0])

                    # 3D offsets and position
                    p0 = np.tile(surf.pos, (5, 1))
                    p1 = 0.9*R*np.array([[0, 0, 0], [+1, +1, 0], [+1, -1, 0], [-1, +1, 0], [-1, -1, 0]])
                    p = p0 + p1

                    # get correct z values
                    p[:, 2] = surf.get_values(p[:, 0], p[:, 1])

                    # test all projection methods
                    for projm in ot.Surface.sphere_projection_methods:
                        pp = surf.sphere_projection(p, projm)
                        self.assertTrue(np.all(np.sign(pp[1:, :2]) == np.sign(p1[1:, :2])))  # correct quadrant
                        self.assertTrue(np.allclose(pp[0, :2] - p1[0, :2], 0, atol=1e-9))  # projection at center

    def test_surface_zmin_zmax_special_types(self):

        sfunc = self.example_sfunc()
        
        # test z_max, z_min, curvature_circle parameter handling
        self.assertRaises(ValueError, ot.Surface, "Circle", z_min=5)  # z_min can only be provided for type "Function"
        self.assertRaises(ValueError, ot.Surface, "Circle", z_max=5)  # z_max can only be provided for type "Function"
        self.assertRaises(ValueError, ot.Surface, "Circle", curvature_circle=5)  
        # ^-- can only be provided for type "Function" and "Data"
        self.assertRaises(ValueError, ot.Surface, "Function", func=sfunc, z_max=5)  
        # ^-- both z_min z_max need to be provided
        self.assertRaises(ValueError, ot.Surface, "Function", func=sfunc, z_min=5)  
        # ^-- both z_min, z_max need to be provided

        # z_max, z_min info messages for "Function"
        sfunc = ot.SurfaceFunction(lambda x, y: x**2 + y**2/10)
        z_min, z_max = ot.Surface("Function", func=sfunc, r=3).extent[4:]
        surf = ot.Surface("Function", func=sfunc, r=3, z_min=z_min, z_max=z_max)  # a valid call
        self.assertAlmostEqual(z_min, surf.z_min)  # value has been assigned
        self.assertAlmostEqual(z_max, surf.z_max)
        surf = ot.Surface("Function", func=sfunc, r=3, z_min=z_min-1e-9, z_max=z_max+1e-9)  # also valid call
        self.assertAlmostEqual(z_min-1e-9, surf.z_min)  # value has been assigned
        self.assertAlmostEqual(z_max+1e-9, surf.z_max)
        surf = ot.Surface("Function", func=sfunc, r=3, z_min=z_min+1e-3, z_max=z_max-1e-3)  
        # ^-- range warning, values don't get set
        self.assertNotAlmostEqual(z_min+1e-3, surf.z_min)  # value has not been assigned
        self.assertNotAlmostEqual(z_max-1e-3, surf.z_max)
        surf = ot.Surface("Function", func=sfunc, r=3, z_min=z_min+1e-3, z_max=z_max+1e-3)  
        # ^-- range valid, +offset -> warning, don't set values
        self.assertNotAlmostEqual(z_min+1e-3, surf.z_min)  # value has not been assigned
        self.assertNotAlmostEqual(z_max+1e-3, surf.z_max)
        surf = ot.Surface("Function", func=sfunc, r=3, z_min=z_min-1e-3, z_max=z_max-1e-3)  
        # ^-- range valid, -offset -> warning, don't set values
        self.assertNotAlmostEqual(z_min-1e-3, surf.z_min)  # value has not been assigned
        self.assertNotAlmostEqual(z_max-1e-3, surf.z_max)
        surf = ot.Surface("Function", func=sfunc, r=3, z_min=z_min, z_max=z_max+5)
        # ^-- range much larger than measured -> warning, set values
        self.assertAlmostEqual(z_min, surf.z_min)  # value has been assigned
        self.assertAlmostEqual(z_max+5, surf.z_max)

        # messages for type "Data"

        # surface height change due to n=4 polynomial interpolation
        # especially noticeable with noisy data
        data = np.random.sample((300, 300))
        ot.Surface("Data", data=data, r=3)  
        
        # since the center is not included in the data set (200 is even), the surface height increases a little bit
        X, Y = np.mgrid[-1:1:200j, -1:1:200j]
        data = X**2 + Y**2
        ot.Surface("Data", data=data, r=1)

    def test_surface_misc(self):

        sfunc = self.example_sfunc()

        # sphere projection
        self.assertRaises(RuntimeError, ot.Surface("Circle").sphere_projection, np.ones((3, 3)))  # surface not a sphere
        self.assertRaises(ValueError, ot.Surface("Sphere").sphere_projection, np.ones((3, 3)), projection_method="ABC")
        # ^-- invalid projection method

        # coverage tests

        # coverage test: intersection s=[0, 0, 1] with a parabola
        surf = ot.Surface("Conic", k=-1, r=3)
        p = np.random.uniform(-3, 3, size=(1000, 3))
        p[:, 2] = 0
        s = np.tile([0., 0., 1.], (1000, 1)) 
        surf.find_hit(p, s)
        
        # get values of surface, but no points are actually on the surface
        surf = ot.Surface("Sphere", r=0.5)
        x, y = np.ones(1000), np.ones(1000)
        surf.get_values(x, y)  # no points are on the surface

        # __find_bounds with masked value at center
        sfunc = ot.SurfaceFunction(lambda x, y: np.ones_like(x), mask_func=lambda x, y: x**2 + y**2 > 0.5)
        ot.Surface("Function", func=sfunc, r=3)

        # throw when no hit finding defined
        self.assertRaises(RuntimeError, ot.Surface("Function", func=sfunc).find_hit, 
                          np.random.sample((5, 3)), np.random.sample((5, 3)))  # no hit finding defined

    def test_surface_function(self):

        # define some functions
        func = lambda x, y, a: 1 + x - y + a
        der = lambda x, y, a: (a*np.ones_like(x), -np.ones_like(x))
        mask = lambda x, y, a: y > a
        hits = lambda p, s, a: a*p  # not correct, just some function
        func0 = lambda x, y: func(x, y, 0)
        der0 = lambda x, y: der(x, y, 1)
        mask0 = lambda x, y: mask(x, y, 5)
        hits0 = lambda x, y: hits(x, y, 1)
        
        # check type checks in init
        self.assertRaises(TypeError, ot.SurfaceFunction, np.array([1, 2]))  # invalid func type
        self.assertRaises(TypeError, ot.SurfaceFunction, None)  # func can't be none
        self.assertRaises(TypeError, ot.SurfaceFunction, func, hit_func=1)  # invalid type
        self.assertRaises(TypeError, ot.SurfaceFunction, func, mask_func=1)  # invalid type
        self.assertRaises(TypeError, ot.SurfaceFunction, func, deriv_func=1)  # invalid type
        self.assertRaises(TypeError, ot.SurfaceFunction, func, func_args=1)  # invalid type
        self.assertRaises(TypeError, ot.SurfaceFunction, func, hit_args=1)  # invalid type
        self.assertRaises(TypeError, ot.SurfaceFunction, func, deriv_args=1)  # invalid type
        self.assertRaises(TypeError, ot.SurfaceFunction, func, mask_args=1)  # invalid type

        # call with just the function
        S0 = ot.SurfaceFunction(func0)

        # call with all function properties provided
        S1 = ot.SurfaceFunction(func0, deriv_func=der0, hit_func=hits0, mask_func=mask0)

        # check function pass-through
        self.assertAlmostEqual(S1.get_values(1, 1), func0(1, 1))
        self.assertAlmostEqual(S1.get_mask(1, 1), mask0(1, 1))
        self.assertAlmostEqual(S1.get_derivative(1, 1), der0(1, 1))
        self.assertAlmostEqual(S1.get_hits(1, 1), hits0(1, 1))

        # call with all function properties and arg dicts provided
        for a in [0, -1, 2]:
            S2 = ot.SurfaceFunction(func, func_args=dict(a=a), deriv_func=der, deriv_args=dict(a=a), 
                                    hit_func=hits, hit_args=dict(a=a), mask_func=mask, mask_args=dict(a=a))

            # check function pass-through
            self.assertAlmostEqual(S2.get_values(1, 1), func(1, 1, a))
            self.assertAlmostEqual(S2.get_mask(1, 1), mask(1, 1, a))
            self.assertAlmostEqual(S2.get_derivative(1, 1), der(1, 1, a))
            self.assertAlmostEqual(S2.get_hits(1, 1), hits(1, 1, a))

    @pytest.mark.slow
    def test_surface_numerical_precision(self):

        # this function checks the numerical precision

        surf_f_1 = lambda x, y: z0 + (x**2 + y**2)/R/2  # parabola = conic section with k=-1
        surf_f_2 = lambda x, y: z0 + R - np.sqrt(R**2 - (x**2 + y**2))  # sphere, k=0

        # actual derivative values
        def n_conic(x, y, rho, k):
            # see surface class for details on calculation

            r = ne.evaluate("sqrt(x**2 + y**2)")
            phi = ne.evaluate("arctan2(y, x)")
            n_r = ne.evaluate("-rho * r  / sqrt(1 - k* rho**2 * r**2 )")

            n = np.zeros((x.shape[0], 3), dtype=np.float64, order='F')
            ne.evaluate("n_r*cos(phi)",     out=n[:, 0])
            ne.evaluate("n_r*sin(phi)",     out=n[:, 1])
            ne.evaluate("sqrt(1 - n_r**2)", out=n[:, 2])

            return n

        # check precision for different surfaces, radii, offsets and regions
        for k, surf_f in zip([-1, 0], [surf_f_1, surf_f_2]):

            for R in [0.1, 10, 10000]:

                for z0 in [0, -10, -1000, -10000]:
                    for fr in [0.002, 0.5, 0.7]:
                        r = fr*R

                        # type "Function"
                        f = ot.SurfaceFunction(func=surf_f)
                        surf1 = ot.Surface("Function", func=f, r=r, silent=True)
                           
                        # compare estimated normals with actual
                        x = np.linspace(0, r, 100)
                        y = np.zeros_like(x)
                        n_is = surf1.get_normals(x, y)
                        n_should = n_conic(x, y, 1/R, k)
                        self.assertTrue(np.allclose(n_is - n_should, 0, atol=1e-7))

                        # type "Data" for different sample sizes
                        for N in [50, 51, 400, 401]:
                            X, Y = np.mgrid[-r:r:N*1j, -r:r:N*1j]
                            data = surf_f(X.flatten(), Y.flatten()).reshape(X.shape)
                            surf2 = ot.Surface("Data", data=data, r=r, silent=True)
                           
                            # compare numeric normals to actual ones
                            # data normals are more unprecise, since enough samples are needed to define the shape
                            n_is = surf2.get_normals(x, y)
                            n_should = n_conic(x, y, 1/R, k)
                            self.assertTrue(np.allclose(n_is - n_should, 0, atol=1e-4))

    def test_filter(self):

        # init exceptions
        self.assertRaises(TypeError, ot.Filter, ot.Surface("Circle"), spectrum=ot.presets.spectrum.x, 
                          pos=[0, 0, 0])  # spectrum is not a TransmissionSpectrum
        self.assertRaises(TypeError, ot.Filter, ot.Point(), spectrum=ot.presets.light_spectrum.d65, 
                          pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(TypeError, ot.Filter, ot.Line(), spectrum=ot.presets.light_spectrum.d65, 
                          pos=[0, 0, 0])  # non 2D surface

        # exemplary filter
        F = ot.Filter(ot.Surface("Circle"), spectrum=ot.TransmissionSpectrum("Gaussian"), pos=[0, 0, 0])

        # test color
        self.assertEqual(F.get_color(), F.spectrum.get_color())  # filter color is just spectrum color

        # test call
        wl = np.random.uniform(color.WL_MIN, color.WL_MAX, 1000)
        self.assertTrue(np.all(F(wl) == F.spectrum(wl)))  # call to filter is just call to its spectrum

        # _new_lock active after init
        self.assertRaises(AttributeError, F.__setattr__, "aaa", 1)

    def test_detector(self):

        # test inits
        self.assertRaises(TypeError, ot.Detector, ot.Point(), pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(TypeError, ot.Detector, ot.Line(), pos=[0, 0, 0])  # non 2D surface

        # _new_lock active
        self.assertRaises(AttributeError, ot.Detector(ot.Surface("Circle"), pos=[0, 0, 0]).__setattr__, "aaa", 1)
        
        # misc exceptions
        
        # surface has no hit finding
        surf = ot.Surface("Rectangle", dim=[3, 3])
        self.assertRaises(RuntimeError, ot.Detector, ot.Surface("Data", data=np.ones((50, 50))), pos=[0, 0, 0])  
        
        # invalid detector type
        self.assertRaises(RuntimeError, ot.Detector, ot.Surface("Conic", R=12, k=-5), pos=[0, 0, 0])  

    def test_s_object(self):

        front = ot.Surface("Sphere")
        back = ot.Surface("Circle")
        pos = [0, -1, 5]

        # type errors in init
        self.assertRaises(TypeError, Element, front, pos, back, d1=[], d2=2)  # invalid d1 type
        self.assertRaises(TypeError, Element, front, pos, back, d2=[], d1=1)  # invalid d2 type
        self.assertRaises(TypeError, Element, front, pos, 5, 1, 1)  # invalid BackSurface type
        self.assertRaises(TypeError, Element, 5, pos, back, 1, 1)  # invalid FrontSurface type
        self.assertRaises(TypeError, Element, front, 5, back, 1, 1)  # invalid pos type
        
        # value errors in init
        self.assertRaises(ValueError, Element, front, pos, back)  # no d1, d2
        self.assertRaises(ValueError, Element, front, [0, 0, np.inf])  # invalid pos
        self.assertRaises(ValueError, Element, front, pos, back, d1=1)  # d2 missing
        self.assertRaises(ValueError, Element, front, pos, back, d2=1)  # d1 missing
        self.assertRaises(ValueError, Element, front, pos, back, d1=-1, d2=1)  # d1 negative
        self.assertRaises(ValueError, Element, front, [5, 5], back, 1, 1)  # wrong pos shape

        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Sphere"), n=ot.RefractionIndex("Constant", n=1.1), 
                    pos=[0, -1, 5], d=2)

        self.assertTrue(L.has_back())
        self.assertTrue(np.all(L.pos == np.array([0, -1, 5])))

        self.assertRaises(RuntimeError, L.__setattr__, "surface", ot.Surface("Circle")) 
        # ^-- surface needs to be set in a different way
        self.assertRaises(RuntimeError, L.__setattr__, "pos", [0, 1, 0]) 
        # ^-- pos needs to be set in a different way

        L.move_to([5, 6, 8])
        self.assertTrue(np.all(L.pos == np.array([5, 6, 8])))

        # only Elements with one surface can be set
        self.assertRaises(RuntimeError, L.set_surface, ot.Surface("Circle"))

        AP = ot.Aperture(ot.Surface("Ring"), pos=[0, 1, 2])
        AP.set_surface(ot.Surface("Conic"))
        self.assertEqual(AP.surface.surface_type, "Conic")

        # getDesc for Element with one and two Surfaces
        Element(ot.Surface("Circle"), [0, 0, 0], ot.Surface("Circle"), d1=0.1, d2=0.1).get_desc()
        Element(ot.Surface("Circle"), [0, 0, 0], d1=0.1, d2=0.1).get_desc()

    def test_s_object_collisions(self):

        # check collisions

        # get geometry
        geom = ot.presets.geometry.arizona_eye()

        # z overlap but no x-y overlap -> no collision
        geom[1].move_to(geom[0].front.pos + [0, 20, 0.01])
        self.assertFalse(Element.check_collision(geom[0].front, geom[1].front)[0])
        
        # no z-overlap -> no collision
        self.assertFalse(Element.check_collision(geom[0].front, geom[2].front)[0])
        # z-overlap but no collision
        self.assertFalse(Element.check_collision(geom[0].front, geom[0].back)[0])
        
        # collision
        geom[1].move_to(geom[0].front.pos + [0, 0, 0.01])  # cornea is now inside empty area of pupil
        coll, x, y = Element.check_collision(geom[0].front, geom[1].front)
        self.assertTrue(coll)
        self.assertTrue(np.all(geom[0].front.get_values(x, y) >= geom[1].front.get_values(x, y)))

        # another collision
        geom = ot.presets.geometry.arizona_eye()
        geom[2].move_to(geom[0].front.pos + [0, 0, 0.2])
        coll, x, y = Element.check_collision(geom[0].front, geom[2].front)
        self.assertTrue(coll)
        self.assertTrue(np.all(geom[0].front.get_values(x, y) >= geom[2].front.get_values(x, y)))

        # collision point - surface

        surf = ot.Surface("Sphere", r=1, R=-5)
        point = ot.Point()

        # point in front of surface, only "hit" when order is reversed
        point.move_to([0, 0, -1])
        hit, _, _ = Element.check_collision(point, surf)
        self.assertFalse(hit)
        hit, _, _ = Element.check_collision(surf, point)
        self.assertTrue(hit)

        # point in behind surface, only "hit" when order is reversed
        point.move_to([0, 0, 1])
        hit, _, _ = Element.check_collision(point, surf)
        self.assertTrue(hit)
        hit, _, _ = Element.check_collision(surf, point)
        self.assertFalse(hit)
        
        # collision line - surface

        surf = ot.Surface("Sphere", r=4, R=-5)
        line = ot.Line(r=0.1)

        # line in front of surface, only "hit" when order is reversed
        line.move_to([0, 0, -10])
        hit, _, _ = Element.check_collision(line, surf)
        self.assertFalse(hit)
        hit, _, _ = Element.check_collision(surf, line)
        self.assertTrue(hit)

        # line behind surface, only "hit" when order is reversed
        line.move_to([0, 0, 1])
        hit, _, _ = Element.check_collision(line, surf)
        self.assertTrue(hit)
        hit, _, _ = Element.check_collision(surf, line)
        self.assertFalse(hit)
        
        # Line no intersects surface, so regarless of order we get a collision
        line = ot.Line(r=5)
        line.move_to([0, 0, -0.1])
        hit, _, _ = Element.check_collision(line, surf)
        self.assertTrue(hit)
        hit, _, _ = Element.check_collision(surf, line)
        self.assertTrue(hit)

        # Coverage

        # type errors
        self.assertRaises(TypeError, Element.check_collision, ot.Point(), ot.Point())
        self.assertRaises(TypeError, Element.check_collision, ot.Line(), ot.Line())

        # surface collision on Element creation
        surf1 = ot.Surface("Sphere", r=3, R=4)
        surf2 = ot.Surface("Circle", r=3)
        self.assertRaises(RuntimeError, Element, surf1, [0, 0, 0], surf2, d1=0.1, d2=0.1)

    def test_lens(self):

        # define media and surfaces
        front = ot.Surface("Sphere", r=3, R=10)  # height 10 - sqrt(91)
        back = ot.Surface("Conic", r=2, k=-5, R=-20)  # height sqrt(26) - 5
        n = ot.presets.refraction_index.SF10
        n2 = ot.RefractionIndex("Constant", n=1.05)
        
        # test init exceptions
        self.assertRaises(TypeError, ot.Lens, front, back, None, pos=[0, 0, 0], d=1)  # invalid RefractionIndex
        self.assertRaises(TypeError, ot.Lens, front, back, n, n2=front, pos=[0, 0, 0], d=1)  # invalid RefractionIndex
        self.assertRaises(ValueError, ot.Lens, front, back, n, pos=[0, 0, 0], d1=1)  # d2 missing
        self.assertRaises(ValueError, ot.Lens, front, back, n, pos=[0, 0, 0], d2=1)  # d1 missing
        self.assertRaises(TypeError, ot.Lens, front, ot.Point(), n, pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(TypeError, ot.Lens, front, ot.Line(), n, pos=[0, 0, 0])  # non 2D surface
        
        # define some lenses
        L = [ot.Lens(front, back, n, pos=[0, 0, 0], de=0.),
             ot.Lens(front, back, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(front, back, n, pos=[0, 0, 0], d=0.8),
             ot.Lens(back, front, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(back, front, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(front, back, n, pos=[0, 0, 0], d1=0.5, d2=0.3, n2=n2),
             ot.Lens(front, front, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], de=0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], de=-0.1),
             ot.Lens(front, front, n, pos=[0, 0, 0], d1=0.1, d2=0.1),
             ot.Lens(back, back, n, pos=[0, 0, 0], d=0.1),
             ot.Lens(back, back, n, pos=[0, 0, 0], d1=0.1, d2=0.1)]

        # exact thicknesses at lens center
        d = [10-np.sqrt(91) + np.sqrt(26)-5, 10-np.sqrt(91) + np.sqrt(26)-5 + 0.1,
             0.8, 0.1, 0.1, 0.8, 0.1, 10-np.sqrt(91) + 0.1, 0.1, 0.2, 0.1, 0.2]

        # check thicknesses
        for di, Li in zip(d, L):
            # three different expressions for the thickness d
            self.assertAlmostEqual(di, Li.d1+Li.d2, 8)
            self.assertAlmostEqual(di, Li.d, 8)
            self.assertAlmostEqual(di, Li.de + Li.front.dp + Li.back.dn, 8)

        # misc tests
        self.assertRaises(AttributeError, L[0].__setattr__, "aaa", 2)  # _new_lock active

    @pytest.mark.slow
    def test_ray_source_behavior(self):

        def or_func(x, y):
            s = np.column_stack((-x, -y, np.ones_like(x)*5))
            ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
            s /= ab[:, np.newaxis]
            return s

        # use power other than default of 1
        # use position other than default of [0, 0, 0]
        # use s other than [0, 0, 1]
        rargs = dict(spectrum=ot.presets.light_spectrum.d50, or_func=or_func, pos=[0.5, -2, 3], 
                     power=2.5, s=[0, 0.5, 1], pol_angle=0.5, div_func=lambda e: np.cos(e))
       
        # possible surface types
        Surfaces = [ot.Point(), ot.Line(), ot.Surface("Circle"), 
                    ot.Surface("Rectangle"), ot.Surface("Ring")]

        # check most RaySource combinations
        # also checks validity of createRays()
        # the loop also has the nice side effect of checking of all image presets

        for Surf in Surfaces:
            for dir_type in ot.RaySource.divergences:
                for or_type in ot.RaySource.orientations:
                    for pol_type in ot.RaySource.polarizations:
                        for Im in [None, ot.presets.image.color_checker]:

                            # only check RectangleSurface with Image being active/set
                            if Im is not None and (not isinstance(Surf, ot.Surface) or Surf.surface_type != "Rectangle"):
                                continue

                            RS = ot.RaySource(Surf, divergence=dir_type, orientation=or_type, 
                                              polarization=pol_type, image=Im, **rargs)
                            RS.get_color()
                            p, s, pols, weights, wavelengths = RS.create_rays(10000)

                            self.assertGreater(np.min(s[:, 2]), 0)  # ray direction in positive direction
                            self.assertGreater(np.min(weights), 0)  # no zero weight rays
                            self.assertEqual(np.sum(weights), rargs["power"])  # rays amount to power
                            self.assertGreaterEqual(np.min(wavelengths), color.WL_MIN)  # inside visible range
                            self.assertLessEqual(np.max(wavelengths), color.WL_MAX)  # inside visible range

                            # check positions
                            self.assertTrue(np.all(p[:, 2] == rargs["pos"][2]))  # rays start at correct z-position
                            self.assertGreaterEqual(np.min(p[:, 0]), RS.surface.extent[0])
                            self.assertLessEqual(np.max(p[:, 0]), RS.surface.extent[1])
                            self.assertGreaterEqual(np.min(p[:, 1]), RS.surface.extent[2])
                            self.assertLessEqual(np.max(p[:, 1]), RS.surface.extent[3])

                            # s needs to be a unity vector
                            ss = s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2
                            self.assertTrue(np.allclose(ss, 1, atol=0.00001))

                            # pol needs to be a unity vector
                            polss = pols[:, 0]**2 + pols[:, 1]**2 + pols[:, 2]**2
                            self.assertTrue(np.allclose(polss, 1, atol=0.00001))

        # special image shapes

        # ray source with one pixel image
        image = np.array([[[0., 1., 0.]]])
        RSS = ot.Surface("Rectangle")
        RS = ot.RaySource(RSS, divergence="Lambertian", image=image,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RS.create_rays(10000)

        # ray source with one width pixel image
        image = np.array([[[0., 1., 0.], [1., 1., 0.]]])
        RSS = ot.Surface("Rectangle")
        RS = ot.RaySource(RSS, divergence="Lambertian", image=image,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RS.create_rays(10000)
        
        # ray source with one height pixel image
        image = np.array([[[0., 1., 0.]], [[1., 1., 0.]]])
        RSS = ot.Surface("Rectangle")
        RS = ot.RaySource(RSS, divergence="Lambertian", image=image,
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RS.create_rays(10000)

    def test_ray_source_init_exceptions(self):

        rsargs = [ot.Surface("Rectangle"), [0, 0, 0]]
    
        # type errors
        self.assertRaises(TypeError, ot.RaySource, *rsargs, spectrum=1)  # invalid spectrum_type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, or_func=1)  # invalid or_func
        self.assertRaises(TypeError, ot.RaySource, *rsargs, power=None)  # invalid power type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, div_angle=None)  # invalid div_angle type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, pol_angle=None)  # invalid pol_angle type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, orientation=1)  # invalid orientation
        self.assertRaises(TypeError, ot.RaySource, *rsargs, divergence=1)  # invalid divergence
        self.assertRaises(TypeError, ot.RaySource, *rsargs, polarization=1)  # invalid polarization
        self.assertRaises(TypeError, ot.RaySource, *rsargs, s=1)  # invalid s type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, s=[1, 3])  # invalid s type
        
        # value errors
        self.assertRaises(ValueError, ot.RaySource, *rsargs, div_angle=0)  # pdiv_angle needs to be above 0
        self.assertRaises(ValueError, ot.RaySource, *rsargs, orientation="A")  # invalid orientation
        self.assertRaises(ValueError, ot.RaySource, *rsargs, divergence="A")  # invalid divergence
        self.assertRaises(ValueError, ot.RaySource, *rsargs,  polarization="A")  # invalid polarization
        self.assertRaises(ValueError, ot.RaySource, *rsargs, power=0)  # power needs to be above 0
        self.assertRaises(ValueError, ot.RaySource, *rsargs, s=[0, 0, np.inf])  # non finite s
        self.assertRaises(ValueError, ot.RaySource, ot.Surface("Sphere"), pos=[0, 0, 0], s=[1, 3]) 
        # ^-- currently only planar surfaces are supported
        
        # image error handling
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=1)  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=[1, 2])  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=[[1, 2], [3, 4]])  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=np.ones((2, 2, 3, 2)))  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=np.ones((0, 2, 3)))  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=np.ones((2, 0, 3)))  # invalid image type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, image=np.ones((2, 2, 2)))  # invalid image type
        self.assertRaises(ValueError, ot.RaySource, *rsargs, image=np.zeros((2, 2, 3)))  # image completely black
        self.assertRaises(ValueError, ot.RaySource, *rsargs, image=-np.ones((2, 2, 3)))  # invalid image values
        self.assertRaises(ValueError, ot.RaySource, *rsargs, image=2*np.ones((2, 2, 3)))  # invalid image values
        self.assertRaises(ValueError, ot.RaySource, *rsargs, image=np.nan*np.ones((2, 2, 3)))  # invalid image values
        self.assertRaises(RuntimeError, ot.RaySource, *rsargs, 
                          image=np.ones((int(1.1*ot.RaySource._max_image_px), 1, 3)))  # image too large

    def test_ray_source_misc(self):
        RS = ot.RaySource(ot.Surface("Rectangle"), [0, 0, 0])

        self.assertRaises(AttributeError, RS.__setattr__, "aaa", 2)  # _new_lock active

        self.assertRaises(RuntimeError, RS.create_rays, 1000)  # spectrum missing
        self.assertRaises(RuntimeError, RS.get_color)  # spectrum missing


        # image can only be used with rectangle Surface
        self.assertRaises(RuntimeError, ot.RaySource(ot.Surface("Circle"), pos=[0, 0, 0],
                                                     image=ot.presets.image.ETDRS_chart_inverted).create_rays, 10000)

        # no or_func specified
        self.assertRaises(RuntimeError, ot.RaySource(ot.Surface("Circle"), pos=[0, 0, 0], 
                                                     spectrum=ot.presets.light_spectrum.d65,
                                                     orientation="Function").create_rays, 10000)  
        
        # no div_func specified
        self.assertRaises(RuntimeError, ot.RaySource(ot.Surface("Circle"), pos=[0, 0, 0], 
                                                     spectrum=ot.presets.light_spectrum.d65,
                                                     divergence="Function").create_rays, 10000)  

        # some rays with negative component in z direction
        self.assertRaises(RuntimeError, ot.RaySource(ot.Surface("Circle"), divergence="Isotropic", pos=[0, 0, 0],
                                                     spectrum=ot.presets.light_spectrum.d65,
                                                     div_angle=80, s=[0, 1, 0.1]).create_rays, 10000) 

        # additional coverage tests
        ot.RaySource(ot.Surface("Circle"), pos=[0, 0, 0], 
                     spectrum=ot.presets.light_spectrum.d65).create_rays(100000, no_pol=True)  # no_pol=True is tested

if __name__ == '__main__':
    unittest.main()
