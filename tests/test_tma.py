#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import pytest

import optrace as ot
from optrace.tracer.transfer_matrix_analysis import TMA as TMA


class TMATests(unittest.TestCase):

    def test_tma_empty(self):

        tma = TMA([])

        # abcd matrix is unity matrix
        self.assertTrue(np.allclose(tma.abcd - np.eye(2), 0))

        # all other properties are set to nan
        for val in [tma.ffl, tma.bfl, tma.efl, tma.efl_n, tma.powers, tma.powers_n, tma.focal_lengths, tma.vertex_point,
                    tma.focal_points, tma.nodal_points, tma.principal_points, tma.d, tma.focal_lengths_n]:
            self.assertTrue(np.all(np.isnan(val)))

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
        spa = ot.SphericalSurface(r=3, R=R1)
        spb = ot.SphericalSurface(r=3, R=-R2)
        aspa = ot.ConicSurface(r=3, R=R1, k=-0.4564)
        aspb = ot.ConicSurface(r=3, R=-R2, k=0.5)
        circ = ot.CircularSurface(r=3)

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
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=n2, **largs).tma(wl=wl).efl, 
                               getf(R1, -R2, n(wl), 1, n2(wl), d))  # ambient n
        self.assertAlmostEqual(ot.Lens(spa, spb, n2=nc, **largs).tma(wl=wl, n0=n2).efl, 
                               getf(R1, -R2, n(wl), n2(wl), 1, d))  # ambient n
        self.assertAlmostEqual(ot.Lens(aspa, aspb, n2=n2, **largs).tma(wl=wl2, n0=n1).efl, 
                               getf(R1, -R2, n(wl2), n1(wl2), n2(wl2), d))  # different wl

    @pytest.mark.os
    def test_tma_single_lens_cardinal(self):
        R1 = 76
        R2 = 35
        wl = 450
        n = ot.RefractionIndex("Constant", n=1.5)
        nc = ot.RefractionIndex("Constant", n=1)
        n1 = ot.RefractionIndex("Constant", n=1.3)
        n2 = ot.RefractionIndex("Constant", n=1.1)
        spa = ot.SphericalSurface(r=3, R=R1)
        spb = ot.SphericalSurface(r=3, R=-R2)
        spc = ot.SphericalSurface(r=3, R=-R1)
        spd = ot.SphericalSurface(r=3, R=R2)
        circ = ot.CircularSurface(r=3)

        def cmp_props(L, list2, n0=ot.RefractionIndex("Constant", n=1)):
            tma = L.tma(n0=n0)
            # edmund optics defines the nodal point shift NPS as sum of focal lengths
            list1 = [tma.efl_n, tma.bfl, tma.ffl,
                     tma.principal_points[0] - tma.vertex_point[0],
                     tma.principal_points[1] - tma.vertex_point[1],
                     tma.focal_lengths[0] + tma.focal_lengths[1]]

            for el1, el2 in zip(list1, list2):
                self.assertAlmostEqual(el1, el2, delta=0.0002)
               
            # check determinant
            self.assertAlmostEqual(np.linalg.det(tma.abcd), tma.n1/tma.n2)

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
                 tma.principal_points[0] - tma.vertex_point[0],
                 tma.principal_points[1] - tma.vertex_point[1],
                 tma.focal_lengths[0] + tma.focal_lengths[1]]
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
        self.assertAlmostEqual(tma.focal_points[0], -15.089, delta=0.001)
        self.assertAlmostEqual(tma.focal_points[1], 24.197, delta=0.001)
        self.assertAlmostEqual(tma.nodal_points[0], 7.200, delta=0.001)
        self.assertAlmostEqual(tma.nodal_points[1], 7.513, delta=0.001)
        self.assertAlmostEqual(tma.principal_points[0], 1.595, delta=0.001)
        self.assertAlmostEqual(tma.principal_points[1], 1.908, delta=0.001)
        self.assertAlmostEqual(tma.powers_n[1], 59.940, delta=0.001)
        self.assertAlmostEqual(tma.d, 7.600, delta=0.001)
        self.assertAlmostEqual(RT.lenses[0].tma().powers_n[1], 42.356, delta=0.001)
        self.assertAlmostEqual(RT.lenses[1].tma(n0=RT.lenses[0].n2).powers_n[1], 21.779, delta=0.001)

        # additional checks for derived properties
        self.assertAlmostEqual(tma.vertex_point[0], RT.lenses[0].front.pos[2])
        self.assertAlmostEqual(tma.vertex_point[1], RT.lenses[-1].back.pos[2])
        self.assertAlmostEqual(tma.n1, RT.n0(550))  # wavelength independent, so value unimportant
        self.assertAlmostEqual(tma.n2, RT.lenses[-1].n2(550))
        self.assertAlmostEqual(tma.powers[0], -59.940 / tma.n1, delta=0.001)
        self.assertAlmostEqual(tma.powers[1], 59.940 / tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.efl_n, 1000/59.940, delta=0.001)
        self.assertAlmostEqual(tma.efl, 1000/59.940*tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.focal_lengths_n[0], -1000 / 59.940, delta=0.001)
        self.assertAlmostEqual(tma.focal_lengths_n[1], 1000 / 59.940, delta=0.001)
        self.assertAlmostEqual(tma.focal_lengths[0], -1000 / 59.940 * tma.n1, delta=0.001)
        self.assertAlmostEqual(tma.focal_lengths[1], 1000 / 59.940 * tma.n2, delta=0.001)
        self.assertAlmostEqual(tma.ffl, tma.focal_points[0] - tma.vertex_point[0])
        self.assertAlmostEqual(tma.bfl, tma.focal_points[1] - tma.vertex_point[1])
        self.assertAlmostEqual(np.linalg.det(tma.abcd), tma.n1/tma.n2)

    def test_tma_image_object_distances(self):

        L = ot.Lens(ot.SphericalSurface(r=1, R=100), ot.SphericalSurface(r=2, R=-200), de=0, pos=[0, 0, 2],
                    n=ot.RefractionIndex("Constant", n=1.5))
        Ltma = L.tma()
        Lf = 2/(1/100 + 1/200)
        self.assertAlmostEqual(Ltma.efl, Lf, places=1)
        
        def check_image_pos(g_z): 
            tma = L.tma()
            g = L.pos[2] - g_z
            f = tma.efl
            image_pos_approx = L.pos[2] + f*g/(g-f)
            image_pos = tma.image_position(g_z)

            self.assertAlmostEqual(image_pos, image_pos_approx, delta=0.5)

            # check if magnification nearly matches b/g
            b = image_pos - tma.principal_points[1]
            m = tma.image_magnification(g_z)
            self.assertAlmostEqual(-b/g, m, delta=0.005)  # minus because image is inverted

        for g_z in [-2*Lf, -Lf/2, 0, Lf/2, 2*Lf]:
            check_image_pos(g_z)
        
        def check_object_pos(b_z): 
            tma = L.tma()
            b = b_z - L.pos[2]
            f = tma.efl
            object_pos_approx = L.pos[2] - f*b/(b-f)
            object_pos = tma.object_position(b_z)

            self.assertAlmostEqual(object_pos, object_pos_approx, delta=0.5)
            
            # check if magnification nearly matches b/g
            g = tma.principal_points[0] - object_pos
            m = tma.object_magnification(b_z)
            self.assertAlmostEqual(-b/g, m, delta=0.005)  # minus because image is inverted

        for b_z in [-2*Lf, -Lf/2, 0, Lf/2, 2*Lf]:
            check_object_pos(b_z)

        RT = ot.Raytracer([-6, 6, -6, 6, -12, 50])
        RT.add(ot.presets.geometry.legrand_eye())
        tma = RT.tma()

        # image/object distances special cases
        # object at -inf
        self.assertAlmostEqual(tma.image_position(-np.inf), tma.focal_points[1])
        self.assertAlmostEqual(tma.image_position(-100000), tma.focal_points[1], delta=0.005)
        self.assertAlmostEqual(tma.image_magnification(-np.inf), 0, delta=1e-9)
        self.assertAlmostEqual(tma.image_magnification(-10000000), 0, delta=1e-5)

        # image at inf
        self.assertAlmostEqual(tma.object_position(np.inf), tma.focal_points[0])
        self.assertAlmostEqual(tma.object_position(100000), tma.focal_points[0], delta=0.005)
        self.assertTrue(np.isnan(tma.object_magnification(np.inf)))
        self.assertTrue(np.abs(tma.object_magnification(10000000)) > 10000)

        # object at focal point 0
        self.assertAlmostEqual(1 / tma.image_position(tma.focal_points[0] - 1e-12), 0) # 1/b approaches zero
        self.assertTrue(1 / tma.image_position(tma.focal_points[0] - 1e-12) > 0)  # positive infinity
        mz = tma.image_magnification(tma.focal_points[0])
        self.assertTrue(np.isnan(mz) or mz > 10000)  # large magnification

        # image at focal point 1
        self.assertAlmostEqual(1 / tma.object_position(tma.focal_points[1] + 1e-12), 0)
        self.assertTrue(1 / tma.object_position(tma.focal_points[1] + 1e-12) < 0)  # negative infinity
        mz = tma.object_magnification(tma.focal_points[1] + 1e-12)
        self.assertAlmostEqual(mz, 0, delta=1e-9)

        # check object/image distance reversal for different front and back medium 
        G = ot.presets.geometry.legrand_eye()
        for i in range(2):
            tma = G.tma()
            zg = -600
            zi = tma.image_position(zg)
            zg2 = tma.object_position(zi)
            
            self.assertAlmostEqual(zg, zg2)
            G.flip()

    def test_tma_error_cases(self):

        L = ot.Lens(ot.CircularSurface(r=1), ot.CircularSurface(r=1), pos=[0, 0, 0], n=ot.presets.refraction_index.SF10)
        
        self.assertRaises(TypeError, TMA, [L], wl=None)  # invalid wl type
        self.assertRaises(TypeError, TMA, L, wl=None)  # invalid lenses type
        self.assertRaises(TypeError, TMA, [L], n0=1.2)  # invalid n0 type
        self.assertRaises(ValueError, TMA, [L], wl=10)  # wl to small
        self.assertRaises(ValueError, TMA, [L], wl=1000)  # wl to large

        # check if only symmetric lenses
        self.assertRaises(RuntimeError, ot.Lens(ot.CircularSurface(r=1), ot.TiltedSurface(r=1, normal=[0, 1, 1]),
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

    def test_tma_ideal_lens(self):

        n01 = ot.RefractionIndex("Constant", n=2)
        n02 = ot.RefractionIndex("Constant", n=1.5)

        for D in [30.4897, -12.5]:  # different optical powers

            L0 = ot.IdealLens(r=2, D=D, pos=[0, 0, 1])
            L1 = ot.IdealLens(r=2, D=D, n2=n01, pos=[0, 0, 1])

            RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 60])
            RT.add(L0)

            LL = [ot.IdealLens(r=2, D=D*2/3, pos=[0, 0, 0]), ot.IdealLens(r=2, D=D/3, pos=[0, 0, 1e-9])]

            for tma in [L0.tma(), L1.tma(), RT.tma(), ot.TMA(LL, n0=n02)]:
                abcd1 = tma.abcd
                abcd = np.array([[1, 0], [-D/1000, tma.n1/tma.n2]])  # 30dpt (1/m) ->  0.03 (1/mm)
                self.assertTrue(np.allclose(abcd1-abcd, 0))
                self.assertAlmostEqual(tma.powers[1], D)
                self.assertAlmostEqual(np.linalg.det(tma.abcd), tma.n1/tma.n2)  # check determinant

    def test_tma_pupils(self):

        for z0 in [0, -10, 12.56]:

            # empty group
            # entrance and exit pupil are the stop itself
            tma = ot.Group().tma()
            zs = z0 + 7.89
            zp = tma.pupil_position(zs)
            mp = tma.pupil_magnification(zs)
            self.assertEqual(zp[0], zs)
            self.assertEqual(zp[1], zs)
            self.assertEqual(mp[0], 1)
            self.assertEqual(mp[1], 1)

            # single lens checks with different optical powers
            for D in [12, -12, 100]:

                # stop before the lens
                # -> stop is just imaged normally by the lens for the exit pupil
                # entrance pupil = aperture stop
                IL = ot.IdealLens(r=2, D=D, pos=[0, 0, z0])
                tma = IL.tma()
                zs = z0-5
                zi = tma.image_position(zs)
                mi = tma.image_magnification(zs)
                zp = tma.pupil_position(zs)
                mp = tma.pupil_magnification(zs)
                self.assertAlmostEqual(zp[1], zi)
                self.assertAlmostEqual(mp[1], mi)
                self.assertAlmostEqual(zp[0], zs)
                self.assertAlmostEqual(mp[0], 1)
                self.assertTrue(((zp[1] < zs) and D > 0) or ((zp[1] > zs) and D < 0))
                self.assertTrue(((mp[1] > 1) and D > 0) or ((mp[1] < 1) and D < 0))
            
                # stop after the lens
                # -> stop is just imaged normally by the lens for the entrace pupil
                # exit pupil = aperture stop
                IL = ot.IdealLens(r=2, D=D, pos=[0, 0, z0])
                tma = IL.tma()
                zs = z0+5
                zi = z0 - (tma.image_position(z0 - (zs - z0))-z0)
                mi = tma.image_magnification(z0 - (zs - z0))  # inverse since we want g/b
                zp = tma.pupil_position(zs)
                mp = tma.pupil_magnification(zs)
                self.assertAlmostEqual(zp[0], zi)
                self.assertAlmostEqual(mp[0], mi)
                self.assertAlmostEqual(zp[1], zs)
                self.assertAlmostEqual(mp[1], 1)
                self.assertTrue(((zp[0] > zs) and D > 0) or ((zp[0] < zs) and D < 0))
                self.assertTrue(((mp[0] > 1) and D > 0) or ((mp[0] < 1) and D < 0))

            # example from https://wp.optics.arizona.edu/jgreivenkamp/wp-content/uploads/sites/11/2019/08/502-09-Stops-and-Pupils.pdf
            # slide 9-23 - 9-25
            IL = ot.IdealLens(r=2, D=10, pos=[0, 0, z0])
            IL2 = ot.IdealLens(r=2, D=1/0.075, pos=[0, 0, z0+50])
            G = ot.Group([IL, IL2])

            for zil in [-5, 2, 8, 150]:
                tma = G.tma()
                zp = z0+25
                zp1, zp2 = tma.pupil_position(zp)
                m1, m2 = tma.pupil_magnification(zp)
                self.assertAlmostEqual(zp1, z0+100/3, delta=1e-5)
                self.assertAlmostEqual(zp2, z0+12.5, delta=1e-5)
                self.assertAlmostEqual(m1, 4/3, delta=1e-5)
                self.assertAlmostEqual(m2, 1.5, delta=1e-5)

                # add some lenses that do nearly nothing but change the mat and ds list
                # zp and mp should stay the same
                G.add(ot.IdealLens(r=2, D=1e-6, pos=[0, 0, z0+zil]))


            # Example from Pedrotti Introduction to Optics, page58
            IL = ot.IdealLens(r=3, D=1/0.06, pos=[0, 0, z0])
            IL2 = ot.IdealLens(r=3, D=-1/0.1, pos=[0, 0, z0+40])
            tma = ot.Group([IL, IL2]).tma()
            zs = z0-30
            zp = tma.pupil_position(zs)
            mp = tma.pupil_magnification(zs)
            self.assertAlmostEqual(zp[0], zs)
            self.assertAlmostEqual(zp[1], z0-10)
            self.assertAlmostEqual(mp[0], 1)
            self.assertAlmostEqual(mp[1], 1)
            
            # example from above but stop is 30mm behind last lens
            IL = ot.IdealLens(r=3, D=1/0.06, pos=[0, 0, z0])
            IL2 = ot.IdealLens(r=3, D=-1/0.1, pos=[0, 0, z0+40])
            tma = ot.Group([IL, IL2]).tma()
            zs = z0+70
            zp = tma.pupil_position(zs)
            mp = tma.pupil_magnification(zs)
            zi = tma.object_position(zs)
            mi = 1/tma.object_magnification(zs)  # inverse since object and image are swapped
            self.assertAlmostEqual(zp[0], zi)
            self.assertAlmostEqual(zp[1], zs)
            self.assertAlmostEqual(mp[0], mi)
            self.assertAlmostEqual(mp[1], 1)

            # Example 1 of Schröder, Technische Optik, page 64
            IL = ot.IdealLens(r=3.5, D=30, pos=[0, 0, z0 + 100])
            tma = IL.tma()
            zs = z0 + 90
            zp = tma.pupil_position(zs)
            mp = tma.pupil_magnification(zs)
            self.assertAlmostEqual(zp[1], IL.pos[2]-14.2857142857)
            self.assertAlmostEqual(mp[1], 1.42857142857)

        # pupil positions of LeGrandEye, see
        # Schwiegerling J. Field Guide to Visual and Ophthalmic Optics. SPIE Publications: 2004.
        eye = ot.presets.geometry.legrand_eye()
        zs0 = eye.apertures[0].pos[2]
        ze = eye.tma().pupil_position(eye.apertures[0].pos[2])
        self.assertAlmostEqual(ze[0], 3.038, delta=0.0005)
        self.assertAlmostEqual(ze[1], 3.682, delta=0.0005)

        # flip the eye and check again
        eye.flip()
        zs = eye.apertures[0].pos[2]
        ze = eye.tma().pupil_position(zs)
        self.assertAlmostEqual(zs0+zs-ze[1], 3.038, delta=0.0005)
        self.assertAlmostEqual(zs0+zs-ze[0], 3.682, delta=0.0005)

    def test_tma_misc(self):

        # some lens
        L = ot.Lens(ot.SphericalSurface(r=1, R=100), ot.SphericalSurface(r=1, R=-200), de=0, pos=[0, 0, 2],
                    n=ot.RefractionIndex("Constant", n=1.5))
        Ltma = L.tma()

        # test tma.matrix_at
        ######
        # image object positions for imaging
        z_g = Ltma.focal_points[0] - 5
        z_b = Ltma.image_position(z_g)
        # calculate matrix for this image-object distance combination
        abcd = Ltma.matrix_at(z_g, z_b)
        # for imaging element B of the ABCD matrix must be zero, check if this is the case
        self.assertAlmostEqual(abcd[0, 1], 0)

    def test_tma_optical_center(self):

        # glass block -> undefined optical center
        L = ot.Lens(ot.CircularSurface(r=3), ot.CircularSurface(r=3), pos=[0, 0, 0], n=ot.presets.refraction_index.SF10)
        tma = L.tma()
        self.assertTrue(np.isnan(tma.optical_center))
        
        # ideal lens -> optical center at its position
        L = ot.Lens(ot.CircularSurface(r=3), ot.CircularSurface(r=3), pos=[0, 0, 0], n=ot.presets.refraction_index.SF10)
        IL = ot.IdealLens(r=3, D=15, pos=[0, 0, 1])
        tma = IL.tma()
        self.assertEqual(tma.optical_center, 1)

        # symmetrically biconvex lens -> optical center at center
        front = ot.SphericalSurface(r=3, R=8)
        back = ot.SphericalSurface(r=3, R=-8)
        nL1 = ot.RefractionIndex("Constant", n=1.5)
        L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
        oc = L1.tma().optical_center
        self.assertAlmostEqual(oc, L1.pos[2])
        
        # asymmetrically biconvex lens -> optical center at V1 + (V2 - V1)/(1 - R2/R1)
        front2 = ot.SphericalSurface(r=3, R=8)
        back2 = ot.SphericalSurface(r=3, R=-12)
        nL1 = ot.RefractionIndex("Constant", n=1.5)
        L1 = ot.Lens(front2, back2, de=0.1, pos=[0, 0, 12], n=nL1)
        tma = L1.tma()
        oc = tma.optical_center
        V1, V2 = tma.vertex_point
        self.assertAlmostEqual(oc, V1 + (V2-V1)/(1 - back2.R/front2.R))

        # meniscus lens
        L2 = ot.Lens(back2, back, d=0.5, pos=[0, 0, 12], n=nL1)
        tma = L2.tma()
        oc = tma.optical_center
        V1, V2 = tma.vertex_point
        self.assertAlmostEqual(oc, V1 + (V2-V1)/(1 - back.R/back2.R))
        
        # concave lens
        L3 = ot.Lens(back2, front, d=0.3, pos=[0, 0, 12], n=nL1)
        tma = L3.tma()
        oc = tma.optical_center
        V1, V2 = tma.vertex_point
        self.assertAlmostEqual(oc, V1 + (V2-V1)/(1 - front.R/back2.R))
      
        # for a setup with the same n1 and n2 the optical center is wavelength independent 
        for wl in [380, 550, 780]:
            self.assertAlmostEqual(L3.tma(wl).optical_center, oc)

        # Example from "Correctly making panoramic imagery and the meaning of optical center", DOI:10.1117/12.805489
        front4 = ot.SphericalSurface(r=3, R=20)
        back4 = ot.SphericalSurface(r=3, R=5)
        nL2 = ot.RefractionIndex("Constant", n=1.5)
        L4 = ot.Lens(front4, back4, d=8, pos=[0, 0, 0], n=nL2)
        tma = L4.tma()
        V2 = tma.vertex_point[1]
        OC = tma.optical_center
        N1, N2 = tma.nodal_points
        self.assertTrue(OC > N2 > N1 > V2)

if __name__ == '__main__':
    unittest.main()
