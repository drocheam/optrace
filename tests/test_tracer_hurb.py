#!/bin/env python3

import pytest
import unittest
import numpy as np
import os

import optrace as ot
import optrace.tracer.misc as misc

from hurb_geometry import *
from tracing_geometry import tracing_geometry

# Tests for special cases in Raytracing

class TracerHurbTests(unittest.TestCase):


    def test_hurb_std_dev_pinhole(self):
        """
        Test the diffraction profile of a pinhole for different parameters,
        including index, diameter, wavelength and distance
        """
        for n, ri, wl, zd in [[1, 0.02, 550, 20], [1.33, 0.012, 380, 30], 
                              [1.5, 0.005, 780, 23], [1.1, 0.01, 480, 20]]:
                
            r, imgi, imgr = hurb_pinhole(n=n, ri=ri, wl=wl, zd=zd, 
                                         N=2000000, N_px=315, dim_ext_fact=3, use_hurb=True, hurb_factor=2)

            std_i = np.average(r**2, weights=imgi)**0.5
            std_r = np.average(r**2, weights=imgr)**0.5
            self.assertAlmostEqual(std_i/std_r, 0.95, delta=0.04)

    def test_hurb_std_dev_slit(self):
        """
        Test slit, different aspect ratios, and orientation.
        Also check for different media, wavelengths and distances.
        """
        for n, d1, d2, wl, zd, ang in [[1, 0.02, 0.1, 550, 20, 0], [1.33, 0.012, 0.05, 380, 30, 10], 
                              [1.5, 0.005, 0.005, 780, 23., -30], [1.1, 0.01, 0.1, 480, 20, 45]]:

            r, _, imgi1, imgi2, imgr1, imgr2 = hurb_slit(n=n, d1=d1, d2=d2, wl=wl, zd=zd, 
                                      N=3000000, N_px=945, angle=ang, dim_ext_fact=5, use_hurb=True, hurb_factor=2)

            std_i1 = np.average(r**2, weights=imgi1)**0.5
            std_r1 = np.average(r**2, weights=imgr1)**0.5
            self.assertAlmostEqual(std_i1/std_r1, 1.11, delta=0.05, msg=f"{n=}, {d1=}, {d2=}, {wl=}, {zd=}, {ang=}")
            
            std_i2 = np.average(r**2, weights=imgi2)**0.5
            std_r2 = np.average(r**2, weights=imgr2)**0.5
            self.assertAlmostEqual(std_i2/std_r2, 1.11, delta=0.09, msg=f"{n=}, {d1=}, {d2=}, {wl=}, {zd=}, {ang=}")
    
    def test_hurb_std_dev_lens(self):
        """
        Test airy pattern from a pupil before an ideal lens.
        Tests include different media, radii, wavelengths and distances
        """
        for n, ri, wl, zd in [[1, 1, 550, 20], [1.33, 3, 380, 30], 
                              [1.5, 5, 780, 23], [1.1, 2.7, 480, 20]]:
                
            r, imgi, imgr = hurb_lens(n=n, ri=ri, wl=wl, zd=zd, 
                                      N=1000000, N_px=315, dim_ext_fact=3, use_hurb=True, hurb_factor=2)

            std_i = np.average(r**2, weights=imgi)**0.5
            std_r = np.average(r**2, weights=imgr)**0.5
            self.assertAlmostEqual(std_i/std_r, 0.95, delta=0.04)
    

    def test_hurb_setting(self):
        """
        Test toggling and setting of the use_hurb parameter
        """
        # turning HURB off leads to an ideal focus
        r, imgi, imgr = hurb_lens(n=1.1, ri=2, wl=550, zd=20, 
                                  N=100000, N_px=945, dim_ext_fact=3, use_hurb=False)
        std_i = np.average(r**2, weights=imgi)**0.5
        self.assertAlmostEqual(std_i, 0.0, delta=1e-10)

        # hurb is disabled by default
        RT = ot.Raytracer([-1, 1, -1, 1, -1, 1])
        self.assertFalse(RT.use_hurb)
        
        # type checking for use_hurb
        self.assertRaises(TypeError, ot.Raytracer, [-1, 1, -1, 1, -1, 1], use_hurb=[2, 3])

    def test_hurb_factor(self):
        """
        Test the influence of the HURB scaling factor, with the example of a slit
        """
        facts = [1, np.sqrt(2), 2, 3]
        stds = []

        for fact in facts:
            r, _, imgi, _, imgr, _ = hurb_slit(n=1.1, d1=0.05, d2=0.50, wl=550, zd=20, 
                                       N=1000000, N_px=945, dim_ext_fact=6, use_hurb=True, hurb_factor=fact)
            
            stds.append(np.average(r**2, weights=imgi)**0.5)

        # std dev seems to shrink by 1/sqrt(fact) for a slit (why not by 1 / fact ?)
        stdf = np.array(stds)*np.sqrt(facts)
        std_norm = np.std(stdf/np.mean(stdf))
        self.assertAlmostEqual(std_norm, 0, delta=0.05)

    def test_hurb_in_snapshot(self):
        """
        Test the detection of use_hurb and HURB_FACTOR for tracing snapshots
        """
        # create a raytracer
        RT = tracing_geometry()
        snap = RT.tracing_snapshot()

        # check that setting HURB leads to changes in the snapshot
        RT.use_hurb = True
        snap2 = RT.tracing_snapshot()
        diff = RT.compare_property_snapshot(snap, snap2)
        self.assertTrue(diff["Any"])
        self.assertTrue(diff["TraceSettings"])
        
        # check that setting HURB_FACTOR leads to changes in the snapshot
        RT.HURB_FACTOR = 1.389
        snap3 = RT.tracing_snapshot()
        diff2 = RT.compare_property_snapshot(snap2, snap3)
        self.assertTrue(diff2["Any"])
        self.assertTrue(diff2["TraceSettings"])

    def test_hurb_unsupported_aperture(self):

        # create raytracer with point source and spherical aperture
        RT = ot.Raytracer([-5, 5, -5, 5, 0, 10], use_hurb=True)
        point = ot.Point()
        RS = ot.RaySource(point, pos=[0, 0, 0], s=[0, 0, 1])
        RT.add(RS)
        sph = ot.SphericalSurface(r=3, R=12)
        ap = ot.Aperture(sph, pos=[0, 0, 2])
        RT.add(ap)

        # doesn't trace, sets geometry error
        RT.trace(10000)
        self.assertTrue(RT.geometry_error)

    def test_hurb_negative_sz(self):
        """
        When a tilted ray is diffracted at an aperture, 
        there can be ray directions that have a negative component in z direction.
        These should be handled by absorbing them and printing a message.
        """
        # create raytracer with point source and spherical aperture
        RT = ot.Raytracer([-5, 5, -5, 5, 0, 10], use_hurb=True)
        point = ot.Point()
        RS = ot.RaySource(point, pos=[0, -5, 0], s=[0, 0, 1], orientation="Converging", conv_pos=[0, 0, 0.01])
        RT.add(RS)
        ring = ot.RingSurface(r=3, ri=0.001)
        ap = ot.Aperture(ring, pos=[0, 0, 0.01])
        RT.add(ap)

        # trace, there will be rays with negative directions that should be absorbed
        # check all ray directions
        # and also for the info message
        RT.trace(100000)
        self.assertTrue(RT._msgs[RT.INFOS.HURB_NEG_DIR, 1] > 0)
        self.assertFalse(np.any(RT.rays.direction_vectors()[:, :, 2] < 0))
    

    def test_hurb_aperture_projection(self):
        """
        A tilted ray sees the aperture projected.
        These leads to smaller distances to the edge, resulting in a higher uncertainty.
        Test the cos(angle) dependency for a slit.
        """
        deg_std_cos = []

        # check multiple input angles to aperture
        for ang0 in [5, 40, 60, 80]:

            ang = np.deg2rad(ang0)
            za = 5/np.tan(ang)

            # create raytracer with point source and slit aperture
            RT = ot.Raytracer([-5, 5, -5, 5, 0, 10*za], use_hurb=True)
            point = ot.Point()
            RS = ot.RaySource(point, pos=[-5, 0, 0],  orientation="Converging", conv_pos=[0, 0, za])
            RT.add(RS)
            slit = ot.SlitSurface(dim=[2, 2], dimi=[0.03, 1.9])  # d2 much larger, so only diffraction in x is relevant
            ap = ot.Aperture(slit, pos=[0, 0, za])
            RT.add(ap)

            # trace
            RT.trace(1000000)

            # get direction vectors and calculate std dev of angle between new direction and input direction
            s = RT.rays.direction_vectors(normalize=True)
            angs = np.acos(misc.rdot(s[:, 0], s[:, 1]))
            deg_std = np.rad2deg(np.nanstd(angs))

            # slit width scales with cos(ang) -> direction uncertainty with 1/cos(ang)
            # so product of divergence and cos(ang) should stay constant for all input angles between ray and aperture
            deg_std_cos.append(deg_std*np.cos(ang))

        # normalize products and ensure there all almost equal
        dev_std = np.std(np.array(deg_std_cos)/deg_std_cos[0])
        self.assertAlmostEqual(dev_std, 0, delta=0.001)


if __name__ == '__main__':
    unittest.main()

