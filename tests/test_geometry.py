#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import pytest

import optrace.tracer.misc as misc
import optrace.tracer.color as color

import optrace as ot
from optrace.tracer.geometry.element import Element
from optrace.tracer.base_class import BaseClass as BaseClass

from optrace.tracer.geometry.surface import Surface



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
                rpos = p.random_positions(N)
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
                    rpos = l.random_positions(N)
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

    def test_point_marker(self):
        
        # init exceptions
        self.assertRaises(TypeError, ot.PointMarker, 1, [0, 0, 0])  # invalid desc
        self.assertRaises(TypeError, ot.PointMarker, "Test", 1)  # invalid pos type
        self.assertRaises(ValueError, ot.PointMarker, "Test", [np.inf, 0, 0])  # inf in pos
        self.assertRaises(TypeError, ot.PointMarker, "Test", [0, 0, 0], marker_factor="yes")  # invalid factor type
        self.assertRaises(TypeError, ot.PointMarker, "Test", [0, 0, 0], text_factor="yes")  # invalid factor type
        self.assertRaises(TypeError, ot.PointMarker, "Test", [0, 0, 0], label_only="yes")  # invalid label_only type

        # check init assignments and extent
        for desc, pos in zip(["a", "", "afhajk"], [[0, 1, 0], [5, 0, 0], [7, 8, 9]]):
            M = ot.PointMarker(desc, pos)
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
    
    def test_line_marker(self):
        
        # init exceptions
        self.assertRaises(TypeError, ot.LineMarker, r=3, desc="Test", pos=[0, 0, 0], line_factor="yes")  # invalid factor type
        self.assertRaises(TypeError, ot.LineMarker, r=3, desc="Test", pos=[0, 0, 0], text_factor="yes")  # invalid factor type

        LM = ot.LineMarker(r=5, angle=12, desc="hjkhkJ", pos=[5, 6, 7])
        self.assertTrue(np.allclose(LM.pos - [5, 6, 7], 0))

        # flip and rotate
        lmx0 = LM.front.angle
        LM.flip()
        lmx1 = LM.front.angle
        self.assertNotAlmostEqual(lmx0, lmx1)
        LM.rotate(45)
        self.assertNotAlmostEqual(lmx1, LM.front.angle)
        
        # test lock
        self.assertRaises(AttributeError, LM.__setattr__, "r46546", 4)  # _new_lock active

    def test_filter(self):

        # init exceptions
        self.assertRaises(TypeError, ot.Filter, ot.CircularSurface(r=3), spectrum=ot.presets.spectrum.x, 
                          pos=[0, 0, 0])  # spectrum is not a TransmissionSpectrum
        self.assertRaises(TypeError, ot.Filter, ot.Point(), spectrum=ot.presets.light_spectrum.d65, 
                          pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(TypeError, ot.Filter, ot.Line(), spectrum=ot.presets.light_spectrum.d65, 
                          pos=[0, 0, 0])  # non 2D surface

        # exemplary filter
        F = ot.Filter(ot.CircularSurface(r=3), spectrum=ot.TransmissionSpectrum("Gaussian"), pos=[0, 0, 0])

        # test color
        self.assertEqual(F.color(), F.spectrum.color())  # filter color is just spectrum color

        # test call
        wl = np.random.uniform(*ot.global_options.wavelength_range, 1000)
        self.assertTrue(np.all(F(wl) == F.spectrum(wl)))  # call to filter is just call to its spectrum

        # _new_lock active after init
        self.assertRaises(AttributeError, F.__setattr__, "aaa", 1)

    def test_detector(self):

        # test inits
        self.assertRaises(TypeError, ot.Detector, ot.Point(), pos=[0, 0, 0])  # non 2D surface
        self.assertRaises(TypeError, ot.Detector, ot.Line(), pos=[0, 0, 0])  # non 2D surface

        # _new_lock active
        self.assertRaises(AttributeError, ot.Detector(ot.CircularSurface(r=3), pos=[0, 0, 0]).__setattr__, "aaa", 1)
        
        # misc exceptions
        
        # surface has no hit finding
        surf = ot.RectangularSurface(dim=[3, 3])
        self.assertRaises(RuntimeError, ot.Detector, ot.DataSurface2D(data=np.ones((50, 50)), r=3), pos=[0, 0, 0])  
        
        # invalid detector type
        self.assertRaises(RuntimeError, ot.Detector, ot.AsphericSurface(r=3, R=12, k=-5, coeff=[0, 0.001]), pos=[0, 0, 0])  

    def test_element(self):

        front = ot.SphericalSurface(r=3, R=10)
        back = ot.CircularSurface(r=3)
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

        L = ot.Lens(ot.CircularSurface(r=3), ot.SphericalSurface(r=3, R=10), n=ot.RefractionIndex("Constant", n=1.1), 
                    pos=[0, -1, 5], d=2)

        self.assertTrue(L.has_back())
        self.assertTrue(np.all(L.pos == np.array([0, -1, 5])))

        self.assertRaises(RuntimeError, L.__setattr__, "surface", ot.SphericalSurface(r=3, R=10)) 
        # ^-- surface needs to be set in a different way
        self.assertRaises(RuntimeError, L.__setattr__, "pos", [0, 1, 0]) 
        # ^-- pos needs to be set in a different way

        L.move_to([5, 6, 8])
        self.assertTrue(np.all(L.pos == np.array([5, 6, 8])))

        # only Elements with one surface can be set
        self.assertRaises(RuntimeError, L.set_surface, ot.SphericalSurface(r=3, R=10))

        AP = ot.Aperture(ot.RingSurface(r=3, ri=0.1), pos=[0, 1, 2])
        AP.set_surface(ot.ConicSurface(r=3, R=10, k=-1))
        self.assertTrue(isinstance(AP.surface, ot.ConicSurface))

        # getDesc for Element with one and two Surfaces
        Element(ot.CircularSurface(r=3), [0, 0, 0], ot.CircularSurface(r=2), d1=0.1, d2=0.1).get_desc()
        Element(ot.CircularSurface(r=3), [0, 0, 0], d1=0.1, d2=0.1).get_desc()

    def test_group_empty(self):

        G = ot.Group()
        self.assertTrue(np.allclose(G.pos - np.array([0, 0, 0]), 0))
        self.assertEqual(len(G.elements), 0)
        self.assertTrue(np.allclose(G.extent - np.array([0, 0, 0, 0, 0, 0]), 0))
        G.move_to([0, 5, 10])
        self.assertTrue(np.allclose(G.tma().abcd - np.eye(2), 0))

    def test_group(self):
        
        # check descriptions
        G = ot.Group(desc="ahfk", long_desc="abc")
        self.assertEqual(G.desc, "ahfk")
        self.assertEqual(G.long_desc, "abc")

        # check tma
        eye = ot.presets.geometry.arizona_eye()
        tma0 = ot.TMA(eye.lenses)
        tma1 = eye.tma()
        self.assertTrue(np.allclose(tma0.abcd - tma1.abcd, 0))

        # change ambient index, abcd should now differ
        eye2 = eye.copy()
        eye2.n0 = ot.RefractionIndex("Constant", n=1.2)
        tma2 = eye2.tma()
        self.assertFalse(np.allclose(tma0.abcd - tma2.abcd, 0))

        # position of Group is just position of first element
        self.assertTrue(np.allclose(eye.lenses[0].pos - eye.pos, 0))

        # test if extent correct. In the eye model the eye volume has the highest xy extent
        # first values of z-extent is given by the first lens (cornea)
        ext = np.array(eye.volumes[0].extent)
        ext[4] = eye.lenses[0].extent[4]
        self.assertTrue(np.allclose(np.array(eye.extent) - ext, 0))

        # check moving
        # new position of group is the new position of first element
        # all others are move accordingly, the abcd matrix stays the same
        # (alternatively we could compare if the relative distances stayed the same)
        pos = [0.12, 5, -8]
        eye.move_to(pos)
        tma2 = eye.tma()
        self.assertTrue(np.allclose(eye.pos - pos, 0))
        self.assertTrue(np.allclose(tma1.abcd - tma2.abcd, 0))

        # check types
        self.assertRaises(TypeError, ot.Group, [], 1)  # invalid refraction index type

    def test_group_geometry_actions(self):

        G = ot.Group()

        RS = ot.RaySource(ot.Point(), pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.led_b1)
        F = ot.Filter(ot.CircularSurface(r=3, desc="a"), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5])
        DET = ot.Detector(ot.CircularSurface(r=3), pos=[0, 0, 10])
        AP = ot.Aperture(ot.RingSurface(r=3, ri=0.2, desc="a"), pos=[0, 0, 10])
        M = ot.PointMarker("Test", pos=[0, 0, 10])
        IL = ot.IdealLens(r=3, D=20, pos=[0, 0, 10])
        L = ot.Lens(ot.CircularSurface(r=2, desc="a"), ot.CircularSurface(r=3, desc="a"),
                       n=ot.RefractionIndex("Constant", n=1.2), pos=[0, 0, 10])
        
        # assign desc, needed later
        IL.front.desc = "a"
        IL.back.desc = "a"

        # test actions for different element types
        for el, list_ in zip([RS, L, IL, AP, DET, F, M], 
                             [G.ray_sources, G.lenses, G.lenses, G.apertures, G.detectors, G.filters, G.markers]):
        
            el2 = el.copy()

            # adding works
            G.add(el)
            self.assertEqual(len(list_), 1)
            G.add(el2)
            self.assertEqual(len(list_), 2)

            # removal works
            succ = G.remove(el2)
            self.assertTrue(succ)
            self.assertEqual(len(list_), 1)
            self.assertTrue(list_[0] is el)
            
            # removing it a second time fails
            succ = G.remove(el2)
            self.assertFalse(succ)
            
            # adding the same element a second time does not work
            G.add(el2)
            assert len(list_) == 2
            G.add(el2)
            self.assertEqual(len(list_), 2)
            G.remove(el2)
            
            # has() works
            self.assertTrue(G.has(el))
            self.assertFalse(G.has(el2))
            
            # clear actually clears
            G.add(el2)
            G.clear()
            self.assertEqual(len(list_), 0)

            # adding as list
            G.add([el, el2])
            self.assertEqual(len(list_), 2)
            self.assertTrue(list_[0] is el)
            self.assertTrue(list_[1] is el2)
            
            # removing as list
            suc = G.remove([el, el2])
            self.assertEqual(len(list_), 0)
            self.assertTrue(suc)

            # adding as Group
            G.add(ot.Group([el, el2]))
            self.assertEqual(len(list_), 2)
            self.assertTrue(list_[0] is el)
            self.assertTrue(list_[1] is el2)
            
            # removing as Group
            suc = G.remove(ot.Group([el, el2]))
            self.assertEqual(len(list_), 0)
            self.assertTrue(suc)
    
        # removal of whole G list, make sure remove makes a copy of a given list
        # otherwise G.filters would change for each deleted element and therefore also the list iterator,
        # leading to not all elements being actually deleted
        assert len(G.filters) == 0
        for i in np.arange(5):
            G.add(ot.Filter(ot.CircularSurface(r=2), spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 5]))
        G.remove(G.filters)
        self.assertFalse(len(G.filters))

        # check type checking for adding
        self.assertRaises(TypeError, G.add, ot.Point())  # Surface is a invalid element type

        # move exceptions
        self.assertRaises(ValueError, G.move_to, [1, 2])
        self.assertRaises(ValueError, G.move_to, [1, np.nan, 3])
        self.assertRaises(TypeError, G.move_to, 3)

        # get tracing surfaces
        IL.move_to([0, 0, 5])  # move so list is unordered in z-position of the elements
        L.move_to([0, 0, -5])
        G = ot.Group([RS, L, IL, DET, AP, F])
        surf = G.tracing_surfaces

        # check length
        self.assertEqual(len(surf), 5)

        # check ordering and that correct surfaces were chosen (that with desc="a")
        for i, el in enumerate(surf):
            if i+1 < len(surf):
                self.assertTrue(surf[i].pos[2] <= surf[i+1].pos[2])
            self.assertEqual(el.desc, "a")

        # check ambient index overwrite when adding two groups together
        G1 = ot.Group(None, n0=ot.RefractionIndex("Constant", n=2))
        G2 = ot.Group(None, n0=ot.RefractionIndex("Constant", n=1))
        G3 = G1.copy()
        G1.add(G2)
        self.assertEqual(G1.n0(500)[()], 1)
        G2.add(G3)
        self.assertEqual(G2.n0(500)[()], 2)
        
    def test_ideal_lens(self):

        # exceptions
        self.assertRaises(TypeError, ot.IdealLens, r=2, pos=[0, 0, 0], D=[])  # invalid optical power type
        self.assertRaises(ValueError, ot.IdealLens, r=2, pos=[0, 0, 1], D=np.inf)  # invalid optical power value
        self.assertRaises(ValueError, ot.IdealLens, r=2, pos=[0, 0, 1], D=0)  # invalid optical power value

        # check extent of IdealLens
        L = ot.IdealLens(r=4.2, D=15, pos=[0, 0, 1])
        self.assertAlmostEqual(L.extent[1]-L.extent[0], 8.4)
        self.assertAlmostEqual(L.extent[3]-L.extent[2], 8.4)
        self.assertEqual(L.d, 0)

        # check surface class
        self.assertTrue(isinstance(L.front, ot.CircularSurface))
        self.assertTrue(isinstance(L.back, ot.CircularSurface))

        # is_ideal correctly set
        self.assertTrue(L.is_ideal)

    def test_lens(self):

        # define media and surfaces
        front = ot.SphericalSurface(r=3, R=10)  # height 10 - sqrt(91)
        back = ot.ConicSurface(r=2, k=-5, R=-20)  # height sqrt(26) - 5
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

        # check cylinder surface
        for Li in L:
            N = int(np.random.uniform(40, 200))
            X, Y, Z = Li.cylinder_surface(N)
            self.assertTrue(X.shape[0] == Y.shape[0] == Z.shape[0] == N)
            de = np.abs(Li.front.edge(20)[2][0] - Li.back.edge(20)[2][0])
            self.assertTrue(np.allclose(np.abs(Z[:, 0] - Z[:, 1]), de))

        # coverage: test tma. actual values will be checked in test_tma.py
        L[0].tma()
        L[0].tma(550)
        L[0].tma(550, n0=ot.RefractionIndex("Abbe", n=1.2, V=50))

        # is_ideal correctly set
        self.assertFalse(L[0].is_ideal)
        
        # misc tests
        self.assertRaises(AttributeError, L[0].__setattr__, "aaa", 2)  # _new_lock active

    @pytest.mark.slow
    def test_ray_source_behavior(self):

        def or_func(x, y):
            s = np.column_stack((-x, -y, np.ones_like(x)*5))
            ab = (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2) ** 0.5
            s /= ab[:, np.newaxis]
            return s

        # use powers other than default of 1
        # use position other than default of [0, 0, 0]
        # use s other than [0, 0, 1]
        rargs = dict(spectrum=ot.presets.light_spectrum.d50, or_func=or_func, pos=[0.5, -2, 3], 
                     power=2.5, s=[0, 0.5, 1], pol_angle=0.5, div_func=lambda e: np.cos(e),
                     pol_func=lambda x: x, pol_angles=[10, 30], pol_probs=[1, 2], conv_pos=[1, 2, 10])
       
        # possible surface types
        Surfaces = [ot.Point(), ot.Line(), ot.CircularSurface(r=2), ot.presets.image.color_checker([2, 2]), 
                    ot.RectangularSurface(dim=[2, 2]), ot.RingSurface(r=2, ri=0.2), ot.presets.psf.airy(1000)]

        # check most RaySource combinations
        # also checks validity of createRays()
        # the loop also has the nice side effect of checking of all image presets

        for Surf in Surfaces:
            for dir_type in ot.RaySource.divergences:
                for div_2d in [False, True]:
                    for or_type in ot.RaySource.orientations:
                        for pol_type in ot.RaySource.polarizations:

                            RS = ot.RaySource(Surf, divergence=dir_type, orientation=or_type, div_2d=div_2d,
                                              polarization=pol_type, **rargs)
                            RS.color()
                            p, s, pols, weights, wavelengths = RS.create_rays(8000)

                            self.assertGreater(np.min(s[:, 2]), 0)  # ray direction in positive direction
                            self.assertGreater(np.min(weights), 0)  # no zero weight rays
                            self.assertAlmostEqual(np.sum(weights), rargs["power"])  # rays amount to power
                            self.assertGreaterEqual(np.min(wavelengths), ot.global_options.wavelength_range[0])  # inside visible range
                            self.assertLessEqual(np.max(wavelengths), ot.global_options.wavelength_range[1])  # inside visible range

                            # check positions
                            self.assertTrue(np.all(p[:, 2] == rargs["pos"][2]))  # rays start at correct z-position
                            self.assertGreaterEqual(np.min(p[:, 0]), RS.surface.extent[0])
                            self.assertLessEqual(np.max(p[:, 0]), RS.surface.extent[1])
                            self.assertGreaterEqual(np.min(p[:, 1]), RS.surface.extent[2])
                            self.assertLessEqual(np.max(p[:, 1]), RS.surface.extent[3])

                            # s needs to be a unity vector
                            ss = s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2
                            self.assertTrue(np.allclose(ss, 1, atol=0.00002, rtol=0))

                            # pol needs to be a unity vector
                            polss = pols[:, 0]**2 + pols[:, 1]**2 + pols[:, 2]**2
                            self.assertTrue(np.allclose(polss, 1, atol=0.00002, rtol=0))

        # special image shapes

        # ray source with one pixel image
        RSS = ot.RGBImage(np.array([[[0., 1., 0.]]]), [2, 2])
        RS = ot.RaySource(RSS, divergence="Lambertian",
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RS.create_rays(10000)

        # ray source with one width pixel image
        RSS = ot.RGBImage(np.array([[[0., 1., 0.], [1., 1., 0.]]]), [2, 2])
        RS = ot.RaySource(RSS, divergence="Lambertian",
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RS.create_rays(10000)
        
        # ray source with one height pixel image
        RS = ot.RGBImage(np.array([[[0., 1., 0.]], [[1., 1., 0.]]]), [2, 2])
        RS = ot.RaySource(RSS, divergence="Lambertian",
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RS.create_rays(10000)

    def test_ray_source_init_exceptions(self):

        rsargs = [ot.RectangularSurface(dim=[2, 2]), [0, 0, 0]]
    
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
        self.assertRaises(TypeError, ot.RaySource, *rsargs, s_sph=5)  # invalid ss type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, pol_func=5)  # invalid pol_func type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, pol_angles=5)  # invalid pol_angles type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, pol_probs=5)  # invalid pol_probs type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, conv_pos=5)  # invalid conv_pos type
        self.assertRaises(TypeError, ot.RaySource, *rsargs, conv_pos=[2, 5])  # invalid conv_pos type
        
        # value errors
        self.assertRaises(ValueError, ot.RaySource, *rsargs, div_angle=0)  # pdiv_angle needs to be above 0
        self.assertRaises(ValueError, ot.RaySource, *rsargs, orientation="A")  # invalid orientation
        self.assertRaises(ValueError, ot.RaySource, *rsargs, divergence="A")  # invalid divergence
        self.assertRaises(ValueError, ot.RaySource, *rsargs,  polarization="A")  # invalid polarization
        self.assertRaises(ValueError, ot.RaySource, *rsargs, power=0)  # power needs to be above 0
        self.assertRaises(ValueError, ot.RaySource, *rsargs, s=[0, 0, np.inf])  # non finite s
        self.assertRaises(ValueError, ot.RaySource, ot.SphericalSurface(r=3, R=-5), pos=[0, 0, 0], s=[1, 3]) 
        # ^-- currently only planar surfaces are supported
        
        # image error handling
        self.assertRaises(RuntimeError, ot.RaySource, ot.RGBImage(np.ones((int(1.1*ot.RaySource._max_image_px), 1, 3)),
                                                               [2, 2]))  # image too large
        self.assertRaises(ValueError, ot.RaySource, ot.RGBImage(np.zeros((100, 100, 3)), [2, 2]))  # image completely black

    def test_ray_source_polarization(self):
            
        N = 100000
        sx = np.tile([1, 0, 0], (N, 1))
       
        # constant polarization angles
        for pmode, pangle in zip(["x", "y", "Constant", "Constant", "Constant"], [0, np.pi/2, 0.2, -0.5, 3.5]):
            RS = ot.RaySource(ot.CircularSurface(r=2), polarization=pmode, pol_angle=np.degrees(pangle))
            _, _, pols, _, _ = RS.create_rays(N)
            psx = misc.rdot(pols, sx)  # for light parallel to optical axis cos(pangle) is the scalar product between pol and x-axis
            self.assertTrue(np.allclose(psx - np.cos(pangle), 0))  # check if near each other

        # list polarization angles
        for pmode, pangles, pprobs in zip(["xy", "List", "List"], [[0, np.pi/2], [0, 0.2], [0, 0.2]], [[1, 1], [1, 1], [1, 2]]):
            RS = ot.RaySource(ot.CircularSurface(r=2), polarization=pmode, pol_angles=np.degrees(pangles), pol_probs=pprobs)
            _, _, pols, _, _ = RS.create_rays(N)
            psx = misc.rdot(pols, sx)  # for light parallel to optical axis cos(pangle) is the scalar product between pol and x-axis
            self.assertTrue(np.all((np.abs(psx - np.cos(pangles[0])) < 1e-5) | (np.abs(psx - np.cos(pangles[1])) < 1e-5)))

            # check if expectation value matches
            mean = np.sum(np.array(pangles)*np.array(pprobs)) / np.sum(pprobs)
            mean2 = np.mean(np.arccos(psx))
            self.assertAlmostEqual(mean, mean2, delta=0.001)

        # check if standard deviation for uniform polarization matches
        RS = ot.RaySource(ot.CircularSurface(r=2), polarization="Uniform")
        _, _, pols, _, _ = RS.create_rays(N)
        psx = misc.rdot(pols, sx)  
        self.assertAlmostEqual(np.degrees(np.std(np.arccos(psx))), 180/np.sqrt(12), delta=0.0001)
       
        # function mode
        # check if standard deviation for uniform polarization matches
        RS = ot.RaySource(ot.CircularSurface(r=2), polarization="Function", pol_func=lambda x: np.exp(-(x-0.5)**2/2/0.03**2))
        _, _, pols, _, _ = RS.create_rays(N)
        psx = misc.rdot(pols, sx)  
        self.assertAlmostEqual(np.degrees(np.std(np.arccos(psx))), 0.03, delta=0.001)

    def test_ray_source_misc(self):
        RS = ot.RaySource(ot.RectangularSurface(dim=[2, 2]), [0, 0, 0])

        self.assertRaises(AttributeError, RS.__setattr__, "aaa", 2)  # _new_lock active

        # no or_func specified
        self.assertRaises(TypeError, ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 0], 
                                                  spectrum=ot.presets.light_spectrum.d65,
                                                  orientation="Function").create_rays, 10000)  
        
        # no div_func specified
        self.assertRaises(TypeError, ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 0], 
                                                  spectrum=ot.presets.light_spectrum.d65,
                                                  divergence="Function").create_rays, 10000)  
        
        # no pol_func specified
        self.assertRaises(TypeError, ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 0], 
                                                  spectrum=ot.presets.light_spectrum.d65,
                                                  polarization="Function").create_rays, 10000)  
        
        # no pol_angles specified
        self.assertRaises(TypeError, ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 0], 
                                                  spectrum=ot.presets.light_spectrum.d65,
                                                  polarization="List").create_rays, 10000)  
        
        # some rays with negative component in z direction
        self.assertRaises(RuntimeError, ot.RaySource(ot.CircularSurface(r=3), divergence="Isotropic", pos=[0, 0, 0],
                                                     spectrum=ot.presets.light_spectrum.d65,
                                                     div_angle=80, s=[0, 1, 0.1]).create_rays, 10000) 

        # additional coverage tests
        ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 0], 
                     spectrum=ot.presets.light_spectrum.d65).create_rays(100000, no_pol=True)  # no_pol=True is tested
        
        # coverage: pol_angles specified, but pol_probs not
        ot.RaySource(ot.CircularSurface(r=3), pos=[0, 0, 0], spectrum=ot.presets.light_spectrum.d65,
                    polarization="List", pol_angles=[0, 10]).create_rays(10000)  

        # s in spherical coordinates gets mapped correctly to s
        RS = ot.RaySource(ot.RectangularSurface(dim=[2, 2]), pos=[0, 0, 0], s_sph=[45, 90])
        self.assertTrue(np.allclose(RS.s - [0, 1/np.sqrt(2), 1/np.sqrt(2)], 0))

    def test_flip_point_line(self):

        # point flip
        P = ot.Point()
        P.move_to([2, 3, 15.789])
        P2 = P.copy()
        P2.flip()
        self.assertTrue(np.allclose(P2.pos - P.pos, 0, atol=1e-10))
        
        # line flip
        L1 = ot.Line(r=5, angle=20)
        L1.move_to([2, 3, 15.789])
        L2 = L1.copy()
        L2.flip()
        self.assertTrue(np.allclose(L2.pos - L1.pos, 0, atol=1e-10))
        self.assertAlmostEqual(-L2.angle, L1.angle)

    def test_flip_elements(self):

        # marker flip
        M1 = ot.PointMarker("bla", [12, -3, 5])
        M2 = M1.copy()
        M2.flip()
        self.assertTrue(np.allclose(M1.pos - M2.pos, 0, atol=1e-10))

        # element flip, single surface
        pos = np.array([7, 8, 9])
        F1 = ot.Filter(ot.SphericalSurface(r=3, R=5), spectrum=ot.TransmissionSpectrum("Gaussian", desc="abc"), pos=pos)
        F2 = F1.copy()
        F2.flip()
        F3 = F2.copy()
        F3.flip()
        self.assertTrue(np.allclose(F1.pos - F2.pos, 0, atol=1e-10))  # same position
        z1 = F1.surface.values(np.array([7, 8]), np.array([8.5, 9.2]))
        z2 = F2.surface.values(np.array([7, 8]), np.array([8.5, 9.2]))
        z3 = F3.surface.values(np.array([7, 8]), np.array([8.5, 9.2]))
        self.assertTrue(np.allclose(z1 - 2*pos[2] + z2, 0, atol=1e-10))  # surface flipped
        self.assertTrue(np.allclose(z1 - z3, 0, atol=1e-10))  # double reversion leads to initial values

        # element flip, double surface
        pos = np.array([7, 8, 9])
        S1 = ot.SphericalSurface(r=3, R=5)
        S2 = ot.CircularSurface(r=3)
        L1 = ot.Lens(S1, S2, pos=pos, n=ot.presets.refraction_index.SF10)
        L2 = L1.copy()
        L2.flip()
        L3 = L2.copy()
        L3.flip()
        self.assertTrue(np.all(L1.pos == L2.pos))  # same position
        self.assertTrue(np.all(L3.pos == L2.pos))  # same position

        # front flipped
        z1 = L1.front.values(np.array([7, 8]), np.array([8.5, 9.2]))
        z2 = L2.back.values(np.array([7, 8]), np.array([8.5, 9.2]))
        z3 = L3.front.values(np.array([7, 8]), np.array([8.5, 9.2]))
        self.assertTrue(np.allclose(z1 - 2*pos[2] + z2, 0, atol=1e-10))  # surface flipped
        self.assertTrue(np.allclose(z1 - z3, 0, atol=1e-8))  # double reversion leads to initial element

        # back flipped
        z1 = L1.back.values(np.array([7, 8]), np.array([8.5, 9.2]))
        z2 = L2.front.values(np.array([7, 8]), np.array([8.5, 9.2]))
        z3 = L3.back.values(np.array([7, 8]), np.array([8.5, 9.2]))
        self.assertTrue(np.allclose(z1 - 2*pos[2] + z2, 0, atol=1e-10))  # surface flipped
        self.assertTrue(np.allclose(z1 - z3, 0, atol=1e-8))  # double reversion leads to initial element

    def test_flip_group(self):

        n = ot.presets.refraction_index.SF10
        n2 = ot.presets.refraction_index.F2
        n3 = ot.presets.refraction_index.K5
        
        # group with ascending order of desc strings according to z position
        G = ot.Group([
            ot.RaySource(ot.CircularSurface(r=3), pos=[0, 1, -1], desc="0"),
            ot.Filter(ot.CircularSurface(r=3), spectrum=ot.TransmissionSpectrum("Constant", val=0.5), pos=[0, 0, 2], desc="1"),
            ot.Lens(ot.CircularSurface(r=5), ot.CircularSurface(r=3), n=n, pos=[0, 0, 5], d=1, n2=n, desc="2"),
            ot.Lens(ot.CircularSurface(r=5), ot.CircularSurface(r=3), n=n, pos=[0, 0, 7], d=1, n2=n2, desc="3"),
            ot.Lens(ot.CircularSurface(r=5), ot.CircularSurface(r=3), n=n, pos=[0, 0, 9], n2=n3, d=1, desc="4"),
            ot.Detector(ot.CircularSurface(r=3), pos=[0, 0, 11], desc="5"),
            ot.Aperture(ot.RingSurface(r=3, ri=1), pos=[0, 0, 13], desc="6"),
            ot.PointMarker("7", pos=[0, 0, 15]),
            ot.CylinderVolume(r=3, length=5, pos=[0, 0, 20], desc="8")
        ])

        # check that after reversing the desc are in reversed order
        Gr = G.copy()
        Gr.flip()
        Grel = Gr.elements
        for i, el in enumerate(Grel):
            self.assertEqual(int(el.desc), len(Grel)-1-i)

        # new position consists of x position of last element, z-position of formerly first element
        # and a y-shift which is the negative former y-distance of the last element from relative to the first
        pos = G.elements[-1].pos.copy()
        y0 = (G.extent[2] + G.extent[3]) / 2
        z0 = (G.extent[4] + G.extent[5]) / 2
        pos[1] = y0 - (G.elements[-1].pos[1] - y0)
        pos[2] = z0 - (G.elements[-1].pos[2] - z0)
        self.assertTrue(np.allclose(Gr.pos - pos, 0))

        # lens sides switched
        for L, Lr in zip(G.lenses, Gr.lenses):
            self.assertEqual(L.front.r, Lr.back.r)
            self.assertEqual(L.back.r, Lr.front.r)

        # check assignment of ambient n
        self.assertEqual(Gr.lenses[0].n2, n2)
        self.assertEqual(Gr.lenses[1].n2, n)
        self.assertEqual(Gr.lenses[2].n2, G.n0)
        self.assertEqual(Gr.n0, n3)

        # xy extent stayed the same
        self.assertTrue(np.allclose(np.array(G.extent[:4]) - Gr.extent[:4], 0))
        
        # difference between start and end of z-extent stayed the same
        self.assertAlmostEqual(G.extent[5]-G.extent[4], Gr.extent[5]-Gr.extent[4])

        # flip group around a user specified axis
        M1 = ot.PointMarker("", pos=[2, 3, 5])
        M2 = ot.PointMarker("", pos=[-1, 0, 2])
        G3 = ot.Group([M1, M2])
        G4 = G.copy()
        G3.flip(y0=1, z0=2)
        self.assertTrue(np.allclose(G3.markers[0].pos - [2, -1, -1], 0))
        self.assertTrue(np.allclose(G3.markers[1].pos - [-1, 2, 2], 0))

        # coverage test: flip empty group
        ot.Group().flip()

    def test_rotate_symmetric(self):

        # rotating elements and points, that have rotational symmetry should change absolutely nothing in the object

        Ss = [
              # Point
              ot.Point(),
              # detector and lens with rotational symmetry
              ot.presets.geometry.arizona_eye().lenses[-2],
              ot.presets.geometry.arizona_eye().detectors[-1],
              ot.PointMarker("defg", [0, 0, 0]),
              ot.SphereVolume(R=5, pos=[0, 1, 23])
             ]

        for SSi in Ss:
            for i in range(2):
                cr0 = SSi.crepr()
                SSi.rotate(20.132)
                cr1 = SSi.crepr()
                self.assertEqual(cr0, cr1)
                SSi.move_to([456, -2.567, 0.22])

    def test_rotate_line(self):

        # rotating a line adds to its angle

        ang0 = 7.987
        sang = ang0

        L = ot.Line(r=2, angle=ang0)
        L.move_to([8, 8, -78])
        pos0 = L.pos

        for dang in np.random.uniform(0, 360, 10):
            L.rotate(dang)
            sang += dang
            # check that angle is the cumulative angle
            self.assertAlmostEqual(L.angle, sang)
            # position is the same as before
            self.assertTrue(np.allclose(L.pos - pos0, 0))
    
    def test_rotate_element(self):

        S0 = ot.FunctionSurface2D(r=3, func=lambda x, y: x ** 2 + y ** 2 / 0.95)
        S1 = ot.TiltedSurface(r=4, normal=[0, 0.5, 1])

        # rotate a lens, this is the same as rotate both its surfaces
        L = ot.Lens(S0, S1, n=ot.presets.refraction_index.SF10, de=0, pos=[0, 5, 6])
        S02 = L.front.copy()
        S12 = L.back.copy()
        ang = -4.879
        L.rotate(ang)
        S02.rotate(ang)
        S12.rotate(ang)
        self.assertEqual(L.front.crepr(), S02.crepr())
        self.assertEqual(L.back.crepr(), S12.crepr())

        # rotate a detector, this is the same as rotation its surface
        D = ot.Detector(S1, pos=[-8, 9., 12])
        S13 = D.front.copy()
        ang = -4.879
        D.rotate(ang)
        S13.rotate(ang)
        self.assertEqual(D.front.crepr(), S13.crepr())

    def test_rotate_group(self):

        surf =  ot.RectangularSurface(dim=[3., 2])
        det = ot.Detector(surf, pos=[0, 0, 0])
        mark = ot.PointMarker("abc", pos=[0, 0, 5])

        G = ot.Group([det, mark])

        # rotate around default axis
        G.rotate(90)
        self.assertAlmostEqual(det.surface._angle, np.pi/2)

        # rotate around user axis
        G.rotate(90, x0=2, y0=1)
        self.assertTrue(np.allclose(mark.pos - [3, -1, 5], 0))
        self.assertTrue(np.allclose(det.pos - [3, -1, 0], 0))
        self.assertAlmostEqual(det.surface._angle, np.pi)

        # coverage: rotate empty group
        ot.Group().rotate(5)

    def test_volumes(self):

        # default calls
        ot.SphereVolume(R=5, pos=[-8, 9, 10])
        ot.BoxVolume(dim=[10, 20], length=5, pos=[-8, 9, 10])
        ot.CylinderVolume(r=5, length=3, pos=[-8, 9, 10])
        
        # calls with opacity and color
        ot.SphereVolume(R=5, pos=[-8, 9, 10], opacity=0.9, color=(1, 0.5, 0.1))
        ot.BoxVolume(dim=[10, 20], length=5, pos=[-8, 9, 10], opacity=0.9, color=(1, 0.5, 0.1))
        ot.CylinderVolume(r=5, length=3, pos=[-8, 9, 10], opacity=0.9, color=(1, 0.5, 0.1))

        # Volume exceptions, tested with the help of a sphere volume
        self.assertRaises(ValueError, ot.SphereVolume, pos=[1, 2, 3], R=5, opacity=-1)  # opacity negative
        self.assertRaises(ValueError, ot.SphereVolume, pos=[1, 2, 3], R=5, opacity=2)  # opacity > 1
        self.assertRaises(TypeError, ot.SphereVolume, pos=[1, 2, 3], R=5, opacity=[])  # invalid opacity type
        self.assertRaises(TypeError, ot.SphereVolume, pos=[1, 2, 3], R=5, color=2)  # invalid color type
        self.assertRaises(ValueError, ot.SphereVolume, pos=[1, 2, 3], R=5, color=[1, 2])  # invalid color length

        # sphere volume pos defines sphere center
        sphv = ot.SphereVolume(R=3, pos=[1, 2, 3])
        self.assertAlmostEqual(sphv.extent[4], 0)
        self.assertAlmostEqual(sphv.extent[5], 6)
        
        # cylinder volume pos defines center of front surface
        cylv = ot.CylinderVolume(r=3, length=5, pos=[1, 2, 3])
        self.assertAlmostEqual(cylv.extent[4], 3)
        self.assertAlmostEqual(cylv.extent[5], 8)
        
        # box volume pos defines center of front surface
        boxv = ot.BoxVolume(dim=[2, 3], length=6, pos=[1, 2, 3])
        self.assertAlmostEqual(boxv.extent[4], 3)
        self.assertAlmostEqual(boxv.extent[5], 9)
    
    def test_geometry_eye_presets(self):

        def base_RT():
            RT = ot.Raytracer(outline=[-30, 30, -30, 30, -50, 200])
            RSS = ot.CircularSurface(r=0.5)
            RS = ot.RaySource(RSS, pos=[0, 0, -50], spectrum=ot.presets.light_spectrum.d65)
            RT.add(RS)
            return RT

        # test example geometries without parameters
        for geom_func in ot.presets.geometry.eye_models:
            RT = base_RT()
            geom = geom_func()
            RT.add(geom)
            RT.trace(50000)

        # test parameter of arizona eye
        for A, r_det, P, pos in zip([0, 1, 2, 3, 4], [5, 8, 9, 10], [2, 3, 5, 5.7],
                                    [[0, 0, 0], [-1, 0, 5], [10, 7, -4], [8, 9, 10]]):
            RT = base_RT()
            geom = ot.presets.geometry.arizona_eye(adaptation=A, pupil=P, r_det=r_det, pos=pos)
            RT.add(geom)

            RT.trace(50000)

            self.assertFalse(RT.geometry_error)
            self.assertEqual(RT.detectors[-1].surface.r, r_det)
            self.assertEqual(RT.apertures[-1].surface.ri, P / 2)
            self.assertTrue(np.allclose(RT.lenses[0].pos - pos, 0))

        # test parameter of legrand eye
        for r_det, P, pos in zip([5, 8, 9, 10], [2, 3, 5, 5.7],
                                 [[0, 0, 0], [-1, 0, 5], [10, 7, -4], [8, 9, 10]]):
            RT = base_RT()
            geom = ot.presets.geometry.legrand_eye(pupil=P, r_det=r_det, pos=pos)
            RT.add(geom)
            RT.trace(50000)

            self.assertFalse(RT.geometry_error)
            self.assertEqual(RT.detectors[-1].surface.r, r_det)
            self.assertEqual(RT.apertures[-1].surface.ri, P / 2)
            self.assertTrue(np.allclose(RT.lenses[0].pos - pos - [0, 0, 0.25], 0))


    def test_geometry_camera_preset(self):
        """check that ideal camera preset gets calculated correctly"""

        RT = ot.Raytracer(outline=[-20, 20, -20, 20, -2000, 2000])

        for z_g in [-1000, -52, -3.59]:  # different object distances
            for b in [2, 125, 369]:  # different image distances
                for cam_pos in [[0, 0, 0], [-3, 2, 50]]:  # different camera positions

                    # create a point source
                    r_det = 10
                    div_angle = np.degrees(np.arctan(r_det/(cam_pos[2] - z_g))/2)
                    RS = ot.RaySource(ot.Point(), pos=[cam_pos[0], cam_pos[1], z_g],
                                      div_angle=div_angle, divergence="Isotropic")
                    RT.add(RS)

                    # add camera and aperture at image plane
                    RT.add(ot.presets.geometry.ideal_camera(z_g=z_g, cam_pos=cam_pos, b=b, r_det=r_det, r=r_det))
                    RT.add(ot.Aperture(ot.CircularSurface(r=8), pos=RT.detectors[0].pos))

                    RT.trace(10000)

                    # check that image distance is correct and that the image is a point
                    self.assertAlmostEqual(np.std(RT.rays.p_list[:, -1, :2] - cam_pos[:2]), 0, delta=1e-10)
                    self.assertAlmostEqual(RT.detectors[0].pos[2] - RT.lenses[0].pos[2], b)
                    
                    RT.clear()

        # check assignment of r_det and r
        G = ot.presets.geometry.ideal_camera(z_g=-1000, cam_pos=[0, 0, 5], b=10, r=6.3, r_det=5.4)
        self.assertEqual(G.lenses[0].front.r, 6.3)
        self.assertEqual(G.detectors[0].front.dim[0], 2*5.4)

        # for g = -inf, D of the lens is just 1/b (because of 1/f = 1/b + 1/g)
        G = ot.presets.geometry.ideal_camera(z_g=-np.inf, b=5, cam_pos=[0, 0, 0])
        self.assertAlmostEqual(G.lenses[0].D, 1000/5)

        # check if negative b and g are handled correctly
        self.assertRaises(ValueError, ot.presets.geometry.ideal_camera, z_g=5, b=5, cam_pos=[0, 0, 0])
        self.assertRaises(ValueError, ot.presets.geometry.ideal_camera, z_g=-5, b=-5, cam_pos=[0, 0, 0])

if __name__ == '__main__':
    unittest.main()
