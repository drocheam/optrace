#!/bin/env python3

import sys
sys.path.append('.')

import doctest
import unittest
import numpy as np
import warnings
import pytest

import optrace as ot
from optrace.tracer.transfer_matrix_analysis import TMA as TMA


class TMATests(unittest.TestCase):

    def test_tma_single_lens_efl(self):

        R1 = 100
        R2 = 80
        d = 0.5
        wl = 550
        wl2 = 650
        n = ot.RefractionIndex("Abbe", n=1.5, V=60)
        nc = ot.RefractionIndex("Constant", n=1)
        n1 = ot.RefractionIndex("Abbe", n=1.3, V=60)
        n2 = ot.RefractionIndex("Abbe", n=1.1, V=60)
        spa = ot.Surface("Sphere", r=3, R=R1)
        spb = ot.Surface("Sphere", r=3, R=-R2)
        aspa = ot.Surface("Conic", r=3, R=R1)
        aspb = ot.Surface("Conic", r=3, R=-R2)
        circ = ot.Surface("Circle", r=3)

        # lens maker equation
        def getf(R1, R2, n, n1, n2, d):
            D = (n-n1)/n2/R1 - (n-n2)/R2/n2 + (n-n1)/(n*R1) * (n-n2)/(n2*R2) * d 
            return 1 / D if D else np.inf 
      
        # check focal length for different surface constellations
        largs = dict(d=d, n=n, pos=[0, 0, 0])
        self.assertAlmostEqual(ot.Lens(spa, spb, **largs).tma(wl=wl).efl, getf(R1, -R2, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spb, spa, **largs).tma(wl=wl).efl, getf(-R2, R1, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spa, circ, **largs).tma(wl=wl).efl, getf(R1, np.inf, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(circ, spb, **largs).tma(wl=wl).efl, getf(np.inf, -R2, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(circ, spa, **largs).tma(wl=wl).efl, getf(np.inf, R1, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spb, circ, **largs).tma(wl=wl).efl, getf(-R2, np.inf, n(wl), 1, 1, d))
        self.assertTrue(np.isnan(ot.Lens(circ, circ, **largs).tma(wl=wl).efl))
        self.assertAlmostEqual(ot.Lens(spa, spa, **largs).tma(wl=wl).efl, getf(R1, R1, n(wl), 1, 1, d))
        self.assertAlmostEqual(ot.Lens(spb, spb, **largs).tma(wl=wl).efl, getf(-R2, -R2, n(wl), 1, 1, d))

        # ambient n on both sides
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=n2, **largs).tma(wl=wl, n0=n2).efl, 
                               getf(R1, -R2, n(wl), n2(wl), n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=n2, **largs).tma(wl=wl, n0=n2).efl, 
                               getf(R1, -R2, n(wl), n2(wl), n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(aspa, aspb, n2=n2, **largs).tma(wl=wl2, n0=n2).efl, 
                               getf(R1, -R2, n(wl2), n2(wl2), n2(wl2), d))  # different wl

        # ambient n on one side
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=n2, **largs).tma(wl=wl, ).efl, 
                               getf(R1, -R2, n(wl), 1, n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=nc, **largs).tma(wl=wl, n0=n2).efl, 
                               getf(R1, -R2, n(wl), n2(wl), 1, d))  # ambient n
        self.assertAlmostEqual(ot.Lens(aspa, aspb, n2=n2, **largs).tma(wl=wl2, n0=n1).efl, 
                               getf(R1, -R2, n(wl2), n1(wl2), n2(wl2), d))  # different wl

    def test_tma_single_lens_cardinal(self):
        R1 = 76
        R2 = 35
        wl = 450
        n = ot.RefractionIndex("Constant", n=1.5)
        nc = ot.RefractionIndex("Constant", n=1)
        n1 = ot.RefractionIndex("Constant", n=1.3)
        n2 = ot.RefractionIndex("Constant", n=1.1)
        spa = ot.Surface("Sphere", r=3, R=R1)
        spb = ot.Surface("Sphere", r=3, R=-R2)
        spc = ot.Surface("Sphere", r=3, R=-R1)
        spd = ot.Surface("Sphere", r=3, R=R2)
        circ = ot.Surface("Circle", r=3)

        def cmp_props(L, list2, n0=ot.RefractionIndex("Constant", n=1)):
            tma = L.tma(n0=n0)
            # edmund optics defines the nodal point shift NPS as sum of focal lengths
            list1 = [tma.efl_n, tma.bfl, tma.ffl,
                     tma.principal_point[0] - tma.vertex_point[0],
                     tma.principal_point[1] - tma.vertex_point[1],
                     tma.focal_length[0] + tma.focal_length[1]]

            for el1, el2 in zip(list1, list2):
                self.assertAlmostEqual(el1, el2, delta=0.0002)

        # values from
        # https://www.edmundoptics.com/knowledge-center/tech-tools/focal-length/

        # edmund optics list: efl, bfl, ffl, P, P'', NPS
        # P and P'' are lengths between principal points and vertext points
       
        # standard biconvex lens
        L = ot.Lens(spa, spb, pos=[0, 0, 0], d=0.71, n=n1)
        cmp_props(L, [79.9980, 79.8255, -79.6235, 0.3745, -0.1725, 0])

        # standard iconcave lens
        L = ot.Lens(spb, spa, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [-47.8992, -47.9904, 47.9412, 0.0420, -0.0912, 0])

        # negative meniscus, R < 0
        L = ot.Lens(spb, spc, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [-129.9674, -130.2150, 129.8534, -0.1140, -0.2476, 0])

        # positive meniscus, R < 0
        L = ot.Lens(spc, spb, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [129.5455, 129.6591, -129.2987, 0.2468, 0.1136, 0])

        # negative meniscus, R > 0
        L = ot.Lens(spa, spd, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [-129.9674, -129.8534, 130.2150, 0.2476, 0.1140, 0])

        # positive meniscus, R > 0
        L = ot.Lens(spd, spa, pos=[0, 0, 0], d=0.2, n=n)
        cmp_props(L, [129.5455, 129.2987, -129.6591, -0.1136, -0.2468, 0])

        # plano-convex, R > 0
        L = ot.Lens(spd, circ, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [70, 69.5267, -70, 0, -0.4733, 0])
        
        # plano-convex, R < 0
        L = ot.Lens(circ, spc, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [152, 152, -151.5267, 0.4733, 0, 0])

        # plano-concave, R < 0
        L = ot.Lens(spc, circ, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [-152, -152.4733, 152, 0, -0.4733, 0])
        
        # plano-concave, R > 0
        L = ot.Lens(circ, spd, pos=[0, 0, 0], d=0.71, n=n)
        cmp_props(L, [-70, -70, 70.4733, 0.4733, 0, 0])

        # negative meniscus, R > 0, different media
        L = ot.Lens(spa, spd, pos=[0, 0, 0], d=0.2, n=n, n2=n2)
        cmp_props(L, [-113.7271, -125.0559, 148.0705, 0.2253, 0.0439, 22.7454], n1)

        # both surfaces disc
        tma = ot.Lens(circ, circ, pos=[0, 0, 0], d=0.2, n=n, n2=n2).tma()
        list1 = [tma.efl_n, tma.bfl, tma.ffl,
                 tma.principal_point[0] - tma.vertex_point[0],
                 tma.principal_point[1] - tma.vertex_point[1],
                 tma.focal_length[0] + tma.focal_length[1]]
        self.assertTrue(np.all(np.isnan(list1)))

    def test_raytracer_tma_cardinal_values(self):

        # check if RT throws with no rotational symmetry
        RT = ot.Raytracer([-5, 6, -5, 6, -12, 500])
        RT.add(ot.presets.geometry.arizona_eye())
        RT.lenses[0].move_to([0, 5, -12])
        self.assertRaises(RuntimeError, RT.tma)  # no rotational symmetry
       
        # compare paraxial properties of LeGrand paraxial eye
        # values are from Field Guide to Visual and Ophthalmic Optics, Jim Schwiegerling, page 15
        # and Handbook of visual optics, Volume 1, Pablo, Artal, table 16.1
        # for some reason the the principal points differ between these two,
        # we find the one from Artal to be correct

        RT = ot.Raytracer([-6, 6, -6, 6, -12, 50])
        RT.add(ot.presets.geometry.legrand_eye())
        
        tma = RT.tma()
        self.assertAlmostEqual(tma.focal_point[0], -15.089, delta=0.001)
        self.assertAlmostEqual(tma.focal_point[1], 24.197, delta=0.001)
        self.assertAlmostEqual(tma.nodal_point[0], 7.200, delta=0.001)
        self.assertAlmostEqual(tma.nodal_point[1], 7.513, delta=0.001)
        self.assertAlmostEqual(tma.principal_point[0], 1.595, delta=0.001)
        self.assertAlmostEqual(tma.principal_point[1], 1.908, delta=0.001)
        self.assertAlmostEqual(tma.power_n[1], 59.940, delta=0.001)
        self.assertAlmostEqual(tma.d, 7.600, delta=0.001)
        self.assertAlmostEqual(RT.lenses[0].tma().power_n[1], 42.356, delta=0.001)
        self.assertAlmostEqual(RT.lenses[1].tma(n0=RT.lenses[0].n2).power_n[1], 21.779, delta=0.001)

        # additional checks for derived properties
        self.assertAlmostEqual(tma.vertex_point[0], RT.lenses[0].front.pos[2])
        self.assertAlmostEqual(tma.vertex_point[1], RT.lenses[-1].back.pos[2])
        self.assertAlmostEqual(tma.n1, RT.n0(550))  # wavelength independent, so value unimportant
        self.assertAlmostEqual(tma.n2, RT.lenses[-1].n2(550))
        self.assertAlmostEqual(tma.power[0], -59.940/tma.n1, delta=0.001)
        self.assertAlmostEqual(tma.power[1], 59.940/tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.efl_n, 1000/59.940, delta=0.001)
        self.assertAlmostEqual(tma.efl, 1000/59.940*tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.focal_length_n[0], -1000 / 59.940, delta=0.001)
        self.assertAlmostEqual(tma.focal_length_n[1], 1000 / 59.940, delta=0.001)
        self.assertAlmostEqual(tma.focal_length[0], -1000 / 59.940 * tma.n1, delta=0.001)
        self.assertAlmostEqual(tma.focal_length[1], 1000 / 59.940 * tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.ffl, tma.focal_point[0] - tma.vertex_point[0])
        self.assertAlmostEqual(tma.bfl, tma.focal_point[1] - tma.vertex_point[1])

    def test_tma_image_object_distances(self):

        L = ot.Lens(ot.Surface("Sphere", R=100), ot.Surface("Sphere", R=-200), de=0, pos=[0, 0, 2],
                    n=ot.RefractionIndex("Constant", n=1.5))
        Ltma = L.tma()
        Lf = 2/(1/100 + 1/200)
        self.assertAlmostEqual(Ltma.efl, Lf, places=1)
        
        def check_image_pos(g_z): 
            g = L.pos[2] - g_z
            f = L.tma().efl
            image_pos_approx = L.pos[2] + f*g/(g-f)
            image_pos = L.tma().image_position(g_z)

            self.assertAlmostEqual(image_pos, image_pos_approx, delta=0.5)

        for g_z in [-2*Lf, -Lf/2, 0, Lf/2, 2*Lf]:
            check_image_pos(g_z)
        
        def check_object_pos(b_z): 
            b = b_z - L.pos[2]
            f = L.tma().efl
            object_pos_approx = L.pos[2] - f*b/(b-f)
            object_pos = L.tma().object_position(b_z)

            self.assertAlmostEqual(object_pos, object_pos_approx, delta=0.5)

        for b_z in [-2*Lf, -Lf/2, 0, Lf/2, 2*Lf]:
            check_object_pos(b_z)

        RT = ot.Raytracer([-6, 6, -6, 6, -12, 50])
        RT.add(ot.presets.geometry.legrand_eye())
        tma = RT.tma()

        # image/object distances special cases
        # object at -inf
        self.assertAlmostEqual(tma.image_position(-np.inf), tma.focal_point[1])
        self.assertAlmostEqual(tma.image_position(-100000), tma.focal_point[1], delta=0.005)
        # image at inf
        self.assertAlmostEqual(tma.object_position(np.inf), tma.focal_point[0])
        self.assertAlmostEqual(tma.object_position(100000), tma.focal_point[0], delta=0.005)
        # object at focal point 0
        self.assertAlmostEqual(1 / tma.image_position(tma.focal_point[0] - 1e-12), 0) # 1/b approaches zero
        self.assertTrue(1 / tma.image_position(tma.focal_point[0] - 1e-12) > 0)  # positive infinity
        # image at focal point 1
        self.assertAlmostEqual(1 / tma.object_position(tma.focal_point[1] + 1e-12), 0)
        self.assertTrue(1 / tma.object_position(tma.focal_point[1] + 1e-12) < 0)  # negative infinity

    def test_tma_error_cases(self):

        L = ot.Lens(ot.Surface("Circle"), ot.Surface("Circle"), pos=[0, 0, 0], n=ot.presets.refraction_index.SF10)
        
        self.assertRaises(TypeError, TMA, [L], wl=None)  # invalid wl type
        self.assertRaises(TypeError, TMA, L, wl=None)  # invalid lenses type
        self.assertRaises(TypeError, TMA, [L], n0=1.2)  # invalid n0 type
        self.assertRaises(ValueError, TMA, [L], wl=10)  # wl to small
        self.assertRaises(ValueError, TMA, [L], wl=1000)  # wl to large

        self.assertRaises(ValueError, TMA, [])  # empty lens list

        # check if only symmetric lenses
        self.assertRaises(RuntimeError, ot.Lens(ot.Surface("Circle"), ot.Surface("Circle", normal=[0, 1, 1]),
                          n=ot.presets.refraction_index.SF10, pos=[0, 0, 0]).tma)

        # check surface collision
        LL = [ot.presets.geometry.legrand_eye().elements[0], ot.presets.geometry.legrand_eye().elements[2]]
        LL[1].move_to([0, 0, 0])
        self.assertRaises(RuntimeError, TMA, LL)  # surface collision
        
        # check sharing of same axis
        LL = [ot.presets.geometry.legrand_eye().elements[0], ot.presets.geometry.legrand_eye().elements[2]]
        LL[1].move_to([0, 1, 10])
        self.assertRaises(RuntimeError, TMA, LL)

        # positions inside z-range of lens setup
        LL = [ot.presets.geometry.legrand_eye().elements[0], ot.presets.geometry.legrand_eye().elements[2]]
        self.assertRaises(ValueError, TMA(LL).image_position, 0.25)  # object inside lens setup
        self.assertRaises(ValueError, TMA(LL).object_position, 0.25)  # image inside lens setup

        # check lock
        tma = TMA(LL)
        self.assertRaises(AttributeError, tma.__setattr__, "n78", ot.presets.refraction_index.LAK8)  # invalid property
        self.assertRaises(RuntimeError, tma.__setattr__, "wl", 560)  # object locked

    def test_tma_misc(self):

        # test tma.trace
        L = ot.Lens(ot.Surface("Sphere", R=100), ot.Surface("Sphere", R=-200), de=0, pos=[0, 0, 2],
                    n=ot.RefractionIndex("Constant", n=1.5))
        Ltma = L.tma()
        Ltma.trace([1.2, -0.7])  # list input
        res = Ltma.trace(np.array([1.2, -0.7]))  # 1D array input
        res2 = Ltma.trace(np.array([[1.2, -0.7], [5, 3]]))  # 2D array input
        self.assertTrue(np.allclose(res2[0], res))  # rows in input lead to rows in output

        # test tma.matrix_at
        ######
        # image object positions for imaging
        z_g = Ltma.focal_point[0] - 5
        z_b = Ltma.image_position(z_g)
        # calculate matrix for this image-object distance combination
        abcd = Ltma.matrix_at(z_g, z_b)
        # for imaging element B of the ABCD matrix must be zero, check if this is the case
        self.assertAlmostEqual(abcd[0, 1], 0)

if __name__ == '__main__':
    unittest.main()
