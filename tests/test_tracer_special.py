#!/bin/env python3

import pytest
import unittest
import numpy as np
import os

import optrace as ot
import optrace.tracer.misc as misc

from test_tracer import lens_maker
from tracing_geometry import tracing_geometry

# Tests for special cases in Raytracing

class TracerSpecialTests(unittest.TestCase):

    def test_sphere_detector_range_hits(self):
        """
        this function checks if the detector hit finding correctly handles:
        * rays starting after the detector
        * rays ending before the detector
        * rays starting inside of detector extent
        for different number of threads
        """

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -100, 100])
        RS = ot.RaySource(ot.CircularSurface(r=0.5), pos=[0, 0, 0], divergence="None")
        RT.add(RS)

        aps = ot.RingSurface(r=2, ri=1)
        ap = ot.Aperture(aps, pos=[0, 0, 3])
        RT.add(ap)

        dets = ot.SphericalSurface(r=5, R=-6)
        det = ot.Detector(dets, pos=[0, 0, -10])
        RT.add(det)

        # angular extent for projection_method = "Equidistant"
        ext0 = np.arcsin(RS.surface.r/np.abs(det.surface.R))
        ext = np.array(ext0).repeat(4)*[-1, 1, -1, 1]

        RT.trace(400000)
        cpus = misc.cpu_count()

        for z in [RT.outline[4], RS.pos[2]+1, ap.pos[2]-1, ap.pos[2]+1, RS.pos[2]+1+det.surface.R,
                  RT.outline[5]-RT.N_EPS, RT.outline[5] + 1]:
            det.move_to([0, 0, z])
            for N_th in [1, 3, 4]:
                os.environ["PYTHON_CPU_COUNT"] = str(N_th)
                img = RT.detector_image(projection_method="Equidistant")
                
                if RT.outline[5] > z > RS.surface.pos[2]:
                    self.assertAlmostEqual(img.power(), RS.power)  # correctly lit
                    self.assertTrue(np.allclose(img.extent-ext, 0, atol=1e-2, rtol=0))  # extent correct
                else:
                    self.assertAlmostEqual(img.power(), 0)
        
        os.environ["PYTHON_CPU_COUNT"] = str(cpus)

    def test_offset_system_equality(self):
        """
        this function tests if the same geometry behaves the same if it is shifted by a vector
        also checks numeric precision for some larger offset values
        """

        im = None
        abcd = None

        for pos0 in [(0, 0, 0), (5.789, 0.123, -45.6), (0, 16546.789789, -4654), (1e8, -1e10, 15)]:

            x0, y0, z0 = (*pos0,)
            pos0 = np.array(pos0)
            
            RT = ot.Raytracer(outline=[-5+x0, 5+x0, -5+y0, 5+y0, -10+z0, 50+z0])

            RSS = ot.CircularSurface(r=0.2)
            RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), 
                              divergence="None", pos=pos0+[0, 0, -3])
            RT.add(RS)

            front = ot.SphericalSurface(r=3, R=50)
            back = ot.ConicSurface(r=3, R=-50, k=-1.5)
            L0 = ot.Lens(front, back, n=ot.presets.refraction_index.SF10, pos=pos0+[0, 0.01, 0])
            RT.add(L0)

            front = ot.FunctionSurface1D(r=3,
                                         func=lambda r: r**2/50 + r**2/5000,
                                         parax_roc=25)
            back = ot.CircularSurface(r=2)
            L1 = ot.Lens(front, back, n=ot.presets.refraction_index.SF10, pos=pos0+[0, 0.01, 10])
            RT.add(L1)

            Y, X = np.mgrid[-1:1:100j, -1:1:100j]
            data = 3 - (X**2 + Y**2)
            front = ot.DataSurface2D(data=data, r=4)
            back = ot.TiltedSurface(r=4, normal=[0, 0.01, 1])
            L2 = ot.Lens(front, back, n=ot.presets.refraction_index.K5, pos=pos0+[0, 0, 20])
            RT.add(L2)

            rect = ot.RectangularSurface(dim=[10, 10])
            Det = ot.Detector(rect, pos=pos0+[0., 0., 45])
            RT.add(Det)

            # render image, so we can compare side length and power
            RT.trace(100000)
            im2 = RT.detector_image()

            # remove not symmetric element and compare abcd matrix
            RT.remove(L2)
            abcd2 = RT.tma().abcd

            # first run: save reference values
            if im is None:
                abcd = abcd2
                im = im2.copy()
            # run > 1: compare values
            if im is not None:
                self.assertTrue(np.allclose(im.extent-im2.extent+pos0[:2].repeat(2), 0, atol=0.001, rtol=0))
                self.assertAlmostEqual(im.extent[1]-im.extent[0], im2.extent[1]-im2.extent[0], delta=0.001)
                self.assertAlmostEqual(im.extent[3]-im.extent[2], im2.extent[3]-im2.extent[2], delta=0.001)
                self.assertAlmostEqual(im.power(), im2.power(), places=4)
                self.assertTrue(np.allclose(abcd-abcd2, 0, atol=0.0001, rtol=0))

    @pytest.mark.os
    def test_numeric_tracing_surface_hit_special_cases(self):

        # in the next part "outside surface" relates to an x,y value outside the x,y value range of the surface
        # cases: 0. ray starting above surface and hitting
        #        1. ray starting outside surface and not hitting
        #        2, ray starting outside and hitting lens center
        #        3. ray starting outside surface, not hitting, while overflying surface xy extent
        #        4. ray starting above surface and not hitting
        #        5. ray starting outside surface and hitting surface, while it also could hit edge
        #        6. ray starting outside surface and hitting only edge
        #        7. ray starting inside surface, hitting lens, but extension to zmax of surface would not hit

        # make raytracer
        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

        # Ray Sources
        for x, sx in zip([0, -4, -8, -8, 0, -8, -8, 2.5], [0, 0, 8, 16, 4.5, 13, 6, 0.65]):
            RSS = ot.Point()
            RS = ot.RaySource(RSS, divergence="None", spectrum=ot.presets.light_spectrum.FDC,
                              pos=[x, 0, 0], s=[sx, 0, 10])
            RT.add(RS)

        # Data
        Y, X = np.mgrid[-3:3:200j, -3:3:200j]
        Z = -(X**2 + Y**2)/5

        # add Lens 1
        front = ot.DataSurface2D(r=3, data=Z)
        back = ot.DataSurface2D(r=3, data=Z)
        nL1 = ot.RefractionIndex("Constant", n=1)
        L1 = ot.Lens(front, back, d=1.5, pos=[0, 0, 10], n=nL1)
        RT.add(L1)

        # one ray for each source
        cnt = len(RT.ray_sources)
        RT.trace(cnt)

        # rays missing are absorbed
        for w in RT.rays.w_list[[1, 3, 4, 6], 2]:
            assert w == 0

        # check rays hitting
        mask = L1.front.mask(RT.rays.p_list[[0, 2, 5, 7], 1, 0], RT.rays.p_list[[0, 2, 5, 7], 1, 1])
        assert np.all(mask)

    @pytest.mark.slow
    def test_same_surface_behavior(self):
        """check if different surface modes describing the same surface shape actually behave the same"""

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 500], no_pol=True)

        RSS = ot.CircularSurface(r=0.2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        # illuminated area is especially important for numeric surface_type="Data"
        for RS_r in [0.001, 0.01, 0.1, 0.5]:

            RS.set_surface(ot.CircularSurface(r=RS_r))

            # check different curvatures
            for R_ in [3, 10.2, 100]:
                
                r = 2

                conic = ot.ConicSurface(R=R_, k=-1, r=r)
                sph = ot.SphericalSurface(R=R_, r=r)

                func = lambda x, y, R: 1/2/R*(x**2 + y**2)
                func2 = lambda x, y: 0.78785 + 1/2/R_*(x**2 + y**2)  # some offset that needs to be removed

                surff1 = ot.FunctionSurface2D(func=func2, r=r)
                surff2 = ot.FunctionSurface2D(func=func, r=r, z_min=0, z_max=func(0, r, R_), func_args=dict(R=R_))

                asph1 = ot.AsphericSurface(R=R_, r=r, k=-1, coeff=[0.])  # same as conic
                asph2 = ot.AsphericSurface(R=1e9, r=r, k=-1, coeff=[1/2/R_])
                # conic can be neglected, only polynomial part

                # type "Data" with different resolutions and offsets
                # and odd number defines a point in the center of the lens,
                # for "even" the center is outside the grid
                surfs = []
                for N in [900, 200, 50, 901, 201, 51]:
                    # 2D surface
                    Y, X = np.mgrid[-r:r:N*1j, -r:r:N*1j]
                    data = 4.657165 + 1/2/R_ * (X**2 + Y**2)  # some random offset
                    surf_ = ot.DataSurface2D(r=r, data=data)
                    surfs.append(surf_)

                    # 1D surface
                    r_ = np.linspace(0, r, N)
                    data = 4.657165 + 1/2/R_ * r_**2  # some random offset
                    surf_ = ot.DataSurface1D(r=r, data=data)
                    surfs.append(surf_)

                surf_circ = ot.CircularSurface(r=r)

                n = 1.5
                d = 0.1 + func(0, r, R_)
                f = lens_maker(R_, np.inf, n, 1, d)
                f_list = []

                # create lens, trace and find focus
                for surf in [sph, conic, surff1, surff2, *surfs, asph1, asph2]:

                    L = ot.Lens(surf, surf_circ, n=ot.RefractionIndex("Constant", n=n),
                                pos=[0, 0, +d/2], d=d)
                    RT.add(L)

                    RT.trace(100000)
                    res, _ = RT.focus_search(RT.focus_search_methods[0], 5)
                    f_list.append(res.x)

                    RT.remove(L)
            
                self.assertAlmostEqual(f, f_list[1], delta=0.2)  # f and conic almost equal
                self.assertAlmostEqual(f_list[0], f_list[1], delta=0.2)  # sphere and conic almost equal

                # other surfaces almost equal
                # since those use the exact same function, they should be nearly identical
                f_list2 = f_list[1:]
                for i, fl in enumerate(f_list2):
                    if i + 1 < len(f_list2):
                        self.assertAlmostEqual(fl, f_list2[i+1], delta=0.001)

    def test_abnormal_rays(self):
        """
        rays that hit the lens cylinder edge in any way are absorbed
        case 1: ray hits lens front, but not back
        case 2: ray misses front, but hits back
        case 3: both aren't hit -> already handled by case 1
        """
        N = 10000
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50])

        RSS = ot.CircularSurface(r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        surf1 = ot.CircularSurface(r=3)
        surf2 = ot.CircularSurface(r=1e-6)
        
        # rays don't hit front surface
        L = ot.Lens(surf2, surf1, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.ABSORB_MISSING, 1] / N, places=3)

        # rays don't hit back surface
        RT.lenses[0] = ot.Lens(surf1, surf2, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.ABSORB_MISSING, 2] / N, places=3)
       
    def test_ray_reach(self):

        RT = tracing_geometry()

        # blocking aperture
        RT.apertures[0] = ot.Aperture(ot.CircularSurface(r=5), pos=RT.apertures[0].pos)

        # see if it is handled
        RT.trace(10000)

    def test_detector_ill_conditioned(self):
        """test handling of ill-conditioned rays in find_hit and for detector image/spectrum rendering"""

        # make raytracer
        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -8, 12])

        # create source
        RS0 = ot.RaySource(ot.CircularSurface(r=0.05), divergence="None", pos=[0, 0.5, -4])
        RT.add(RS0)

        # add tilted Detector with incorrect z_max
        # this leads to no roots for regula falsi hit finding inside search region 
        surf = ot.TiltedSurface(r=0.2, normal=[0, -np.sin(1.0), np.cos(1.0)])
        surf._lock = False
        surf.z_max = 0.1
        Det = ot.Detector(surf, pos=[0, 0, 12])
        RT.add(Det)

        # trace
        RT.trace(1000)

        # _hit_detector should set ill_count to N
        _, _, _, _, _, _, ill_count = RT._hit_detector("Detector Image", 0, None, None, "Equidistant")
        self.assertEqual(RT.rays.N, ill_count)

        # coverage: test both detector functions, should print warnings
        RT.detector_image()
        RT.detector_spectrum()

    def test_tilted_plane_different_surface_types(self):
        """
        similar to brewster polarizer example
        checks bound fixing in numerical RT.find_surface_hit
        """

        n = ot.RefractionIndex("Constant", n=1.55)
        b_ang = np.arctan(1.55/1)

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -8, 12])

        # source parameters
        RSS = ot.CircularSurface(r=0.05)
        spectrum = ot.LightSpectrum("Monochromatic", wl=550.)
        s = [0, np.sin(b_ang), np.cos(b_ang)]

        # create sources
        RS0 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0.5, 0, -4], 
                          polarization="x", desc="x-pol")
        RS1 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0, 0, -4], 
                          polarization="y", desc="y-pol")
        RS2 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[-0.5, 0, -4],
                          polarization="Uniform", desc="no pol")
        RT.add(RS0)
        RT.add(RS1)
        RT.add(RS2)

        surf_f = lambda x, y: np.tan(b_ang)*x
        surf1 = ot.FunctionSurface2D(func=surf_f, r=0.7)

        surf2 = ot.TiltedSurface(r=0.7, normal=[-np.sin(b_ang), 0, np.cos(b_ang)])

        Y, X = np.mgrid[-0.7:0.7:100j, -0.7:0.7:100j]
        Z = np.tan(b_ang)*Y
        surf3 = ot.DataSurface2D(r=0.7, data=Z)

        for surf in [surf1, surf2, surf3]:
            L = ot.Lens(surf, surf, d=0.2, pos=[0, 0, 0.5], n=n)
            RT.add(L)

            RT.trace(10000)
            RT.remove(L)

            # check if all rays are straight and get parallel shifted
            p, s, _, _, _, _, _ = RT.rays.rays_by_mask(np.ones(RT.rays.N, dtype=bool), ret=[1, 1, 0, 0, 0, 0, 0])
            self.assertAlmostEqual(np.mean(s[:, -2, 2]), 1)  # still straight
            self.assertTrue(np.allclose(p[:, -2, 0] - p[:, -3, 0], -0.0531867, atol=0.00001, rtol=0))  # parallel shift

    def test_non_sequental_surface_extent(self):
        """
        one surface has a larger z-extent and second surface starts at a lower z-value
        or second case, first surface ends at larger z-extent
        make sure we get no tracing or geometry check errors and cylinder ray detection works
        """

        RT = ot.Raytracer(outline=[-15, 15, -15, 15, -20, 50])

        RSS = ot.CircularSurface(r=4)
        RS = ot.RaySource(RSS, pos=[0, 0, -20])
        RT.add(RS)

        # )) - lens, second surface embracing first
        R1, R2 = 5, 10
        front = ot.SphericalSurface(r=0.999*R1, R=-R1)
        back = ot.SphericalSurface(r=0.999*R2, R=-R2)
        L = ot.Lens(front, back, pos=[0, 0, 0], n=ot.RefractionIndex(), d=0.5)
        RT.add(L)

        FS = ot.CircularSurface(r=3)
        F = ot.Filter(FS, spectrum=ot.TransmissionSpectrum("Constant", val=1), pos=[0, 0, 20])
        RT.add(F)

        N = 100000
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] > 0))
        
        # (( - lens, first surface embracing second
        RT.remove(L)
        front = ot.SphericalSurface(r=0.999*R2, R=R2)
        back = ot.SphericalSurface(r=0.999*R1, R=R1)
        L = ot.Lens(front, back, pos=[0, 0, 0], n=ot.RefractionIndex(), d=0.5)
        RT.add(L)
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] > 0))

        # )), but this time hitting space between surfaces (=edge)
        RT.remove(RS)
        RS = ot.RaySource(ot.RingSurface(r=7, ri=6), pos=[0, 0, -20])
        RT.add(RS)
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] == 0))
        
        # ((, but this time hitting space between surfaces (=edge)
        RT.remove(L)
        front = ot.SphericalSurface(r=0.999*R1, R=-R1)
        back = ot.SphericalSurface(r=0.999*R2, R=-R2)
        L = ot.Lens(front, back, pos=[0, 0, 0], n=ot.RefractionIndex(), d=0.5)
        RT.add(L)
        RT.trace(N)
        self.assertTrue(np.all(RT.rays.w_list[:, -2] == 0))

    def test_few_rays_action(self):
        """test how the raytracer handles few or no rays"""

        RT = tracing_geometry()
        AP = RT.apertures[0]

        # add ideal lens
        RT.add(ot.IdealLens(r=3, D=1, pos=[0, 0, RT.outline[5]-1]))

        # remove apertures and filter so no rays are filtered
        for el in [*RT.apertures, *RT.filters]:
            RT.remove(el)

        for i in range(2):  # without and with rays filtered
            for N in [70, 3, 1]:  # ray numbers

                RT.trace(N)

                # test renders
                RT.source_image()
                RT.detector_image()
                RT.source_spectrum()
                RT.detector_spectrum()
                RT.iterative_render(N)

                # test focus_search
                for fm in RT.focus_search_methods:
                    RT.focus_search(fm, z_start=30)

                # ray storage
                RT.rays.source_sections()
                RT.rays.source_sections(0)
                RT.rays.source_sections(1)
                RT.rays.rays_by_mask(np.full_like(RT.rays.wl_list, 0, dtype=bool))
                RT.rays.rays_by_mask(np.full_like(RT.rays.wl_list, 1, dtype=bool))

            # add aperture and move it so that no rays reach detector or focus_search region
            RT.add(AP)
            RT.apertures[0].move_to(AP.pos + [0.2, 0.2, 0])
      
    def test_hit_dector_many_surfaces_different_detector_surfaces(self):

        # the following is an part of the microscope example

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 300])

        RSS = ot.presets.image.cell([200e-3, 200e-3])
        RS = ot.RaySource(RSS, divergence="Lambertian",
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=27, desc="Cell")
        RT.add(RS)

        # objective doublet properties
        n1, n2 = ot.presets.refraction_index.LAK8, ot.presets.refraction_index.SF10
        R1, R2 = 7.74, -7.29

        # objective group
        objective = ot.Group(desc="Objective")

        # Lens 1 of doublet
        front = ot.CircularSurface(r=5.5)
        back = ot.SphericalSurface(r=5.5, R=-R2)
        L01 = ot.Lens(front, back, d1=0.5, d2=0, pos=[0, 0, 0], n=n2, n2=n2)
        objective.add(L01)

        # Lens 2 of doublet
        front = ot.SphericalSurface(r=5.5, R=-R2)
        back = ot.ConicSurface(r=5.5, R=-R1, k=-0.55)
        L02 = ot.Lens(front, back, d1=0, d2=5.3, pos=[0, 0, 0.0001], n=n1)
        objective.add(L02)

        # move objective lens so that its focal point is 0.6mm behind object
        L0f0 = objective.tma().focal_points[0]
        objective.move_to([0, 0, L01.pos[2] - (L0f0 - RS.pos[2] - 0.6)])

        # add group to raytracer
        RT.add(objective)

        tilt = ot.TiltedSurface(r=3.5, normal=[2, 0, 1])
        det = ot.Detector(tilt, pos=[0, 0, 12.125])
        RT.add(det)

        # detector extent should be larger than that objective
        assert det.extent[5] > L02.extent[5]
        assert det.extent[4] < L01.extent[4]

        RT.trace(200000)

        # tilted surface intersects with 4 surfaces, check if correctly renders image
        img = RT.detector_image()
        self.assertTrue(img.power() > 0.55)

        # check ring detector
        RT.add(ot.Detector(ot.RingSurface(r=3.5, ri=0.3), pos=[0, 0, 12]))
        img = RT.detector_image(detector_index=1)
        self.assertTrue(img.power() > 0.4)
        ny, nx = img._data.shape[:2]
        self.assertEqual(img._data[ny // 2, nx // 2, 3], 0)
        # ^-- due to hole in detector there is no light detected in its center
        
        # check circle detector
        RT.add(ot.Detector(ot.CircularSurface(r=3.5), pos=[0, 0, 12]))
        img = RT.detector_image(detector_index=2)
        self.assertTrue(img.power() > 0.4)
        
        # check conic surface detector
        RT.add(ot.Detector(ot.ConicSurface(r=3.5, R=-10, k=2), pos=[0, 0, 12]))
        img = RT.detector_image(detector_index=3)
        self.assertTrue(img.power() > 0.4)
    
    def test_ray_storage_misc(self):
        # coverage tests

        # warns that there are sources without rays
        RT = tracing_geometry()
        RT.ray_sources[0].power = 0.000001
        RT.ray_sources[1].power = 1
        RT.trace(10000)

    def test_raytracer_output_threading_nopol(self):
        """test actions without multithreading, progressbar and tracing with no_pol"""
        RT = tracing_geometry()
        ot.global_options.multithreading = False
        ot.global_options.show_progress_bar = False

        # raytracer not silent -> outputs messages and progress bar for actions
        RT.trace(10000)
        RT.focus_search(RT.focus_search_methods[0], 12)
        RT.source_image()
        RT.detector_image()
        RT.source_spectrum()
        RT.detector_spectrum()
        RT.iterative_render(100000)

        # show all tracing messages
        for i in np.arange(len(RT.INFOS)):
            RT._msgs = np.zeros((len(RT.INFOS), 2), dtype=int)
            RT._msgs[i, 0] = 1

            # some infos throw
            try:
                RT._show_messages(1000)
            except Exception as err:
                print(err)

        # simulate with no_pol
        RT.no_pol = True
        RT.trace(10000)
        
        ot.global_options.multithreading = True
        ot.global_options.show_progress_bar = True
    
    def test_absorb_missing(self):
        """
        infinitely small lens -> almost all rays miss and are set to absorbed
        """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 50])

        RSS = ot.CircularSurface(r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None", pos=[0, 0, -3])
        RT.add(RS)

        surf = ot.CircularSurface(r=1e-6)
        L = ot.Lens(surf, surf, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)

        N = 10000
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.ABSORB_MISSING, 1] / N, places=3)
        self.assertTrue(np.all(RT.rays.p_list[:, -1, 2] < RT.outline[5] - 1))  # absorbed before hitting outline
    
    def test_tir(self):
        """
        unrealistically high refractive index, a slight angle leads to TIR at the next surface
        """

        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 50], 
                          n0=ot.RefractionIndex("Constant", 100))

        RSS = ot.CircularSurface(r=2)
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="None",
                          pos=[0, 0, -3], s=[0, 0.1, 0.99])
        RT.add(RS)

        surf1 = ot.CircularSurface(r=10)
        surf2 = ot.CircularSurface(r=10)
        L = ot.Lens(surf2, surf1, n=ot.RefractionIndex("Constant", n=1.5), pos=[0, 0, 0], d=0.1)
        RT.add(L)

        N = 10000
        RT.trace(N)
        self.assertEqual(RT._msgs[RT.INFOS.TIR, 0], N)
   
    def test_outline_intersection(self):
        """
        strongly diverging rays, but tracing geometry is a long corridor
        -> almost all rays are absorbed by outline 
        """

        RT = ot.Raytracer(outline=[-3, 3, -3, 3, -10, 5000])

        RSS = ot.Point()
        RS = ot.RaySource(RSS, spectrum=ot.LightSpectrum("Monochromatic", wl=555), divergence="Isotropic", 
                          div_angle=80, pos=[0, 0, -3])
        RT.add(RS)

        N = 10000
        RT.trace(N)
        self.assertAlmostEqual(1, RT._msgs[RT.INFOS.OUTLINE_INTERSECTION, 0] / N, places=3)
    
    def test_brewster_and_fresnel_transmission(self):
        """ test polarization and transmission for a setup with brewster angle"""

        RT = ot.Raytracer(outline=[-5, 5, -5, 5, -10, 10])

        # source parameters
        n = ot.RefractionIndex("Constant", n=1.55)
        b_ang = np.arctan(1.55/1)  # brewster angle
        RSS = ot.CircularSurface(r=0.05)
        spectrum = ot.LightSpectrum("Monochromatic", wl=550.)
        s = [0, np.sin(b_ang), np.cos(b_ang)]

        # create sources
        RS0 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[2, -2.5, -2], s=s, 
                          polarization="x", desc="x-pol", power=1)
        RS1 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[0, -2.5, -2], s=s, 
                          polarization="y", desc="y-pol", power=1)
        RS2 = ot.RaySource(RSS, divergence="None", spectrum=spectrum, pos=[-2, -2.5, -2], s=s, 
                          polarization="Uniform", desc="no pol", power=10)
        RT.add(RS0)
        RT.add(RS1)
        RT.add(RS2)

        # add refraction index step
        rect = ot.RectangularSurface(dim=[10, 10])
        L1 = ot.Lens(rect, rect, de=0.5, pos=[0, 0, 0], n=n, n2=n)
        RT.add(L1)

        RT.trace(100000)

        def get_s_w_pol_source(index):

            mask = np.zeros(RT.rays.N, dtype=bool)
            Ns, Ne = RT.rays.B_list[index:index + 2]
            mask[Ns:Ne] = True

            p, s, pol, w, wl, _, _ = RT.rays.rays_by_mask(mask, slice(None))
            return s, pol, w

        sx, polx, wx = get_s_w_pol_source(0)
        sy, poly, wy = get_s_w_pol_source(1)
        sn, poln, wn = get_s_w_pol_source(2)

        # check power after transmission
        # values for n=1.55, n0=1
        # see https://de.wikipedia.org/wiki/Brewster-Winkel
        w0 = wn[0, 0]
        self.assertTrue(np.allclose(wx[:, 1]/w0, 0.8301, atol=0.001, rtol=0))
        self.assertTrue(np.allclose(wy[:, 1]/w0, 1.0000, atol=0.001, rtol=0))
        self.assertAlmostEqual(np.mean(wn[:, 1])/w0, 0.915, delta=0.001)

        # check pol projection values
        self.assertTrue(np.allclose(polx[:, 0, 1]**2 + polx[:, 0, 2]**2, 0, atol=0.00001, rtol=0))
        self.assertTrue(np.allclose(polx[:, 1, 1]**2 + polx[:, 1, 2]**2, 0, atol=0.00001, rtol=0))
        self.assertTrue(np.allclose(poly[:, 0, 1]**2 + poly[:, 0, 2]**2, 1, atol=0.00001, rtol=0))
        self.assertTrue(np.allclose(poly[:, 1, 1]**2 + poly[:, 1, 2]**2, 1, atol=0.00001, rtol=0))

        mean_yz_proj = lambda pol: np.mean(np.sqrt(pol[:, 1]**2 + pol[:, 2]**2)) 
        self.assertAlmostEqual(mean_yz_proj(poln[:, 0]), 2/np.pi, delta=0.005)
        self.assertAlmostEqual(mean_yz_proj(poln[:, 1]), 2/np.pi, delta=0.005)

        for pols, ss in zip([polx[:, 0], polx[:, 1], poly[:, 0], poly[:, 1], poln[:, 0], poln[:, 1]],
                            [sx[:, 0], sx[:, 1], sy[:, 0], sy[:, 1], sn[:, 0], sn[:, 1]]):
            # pols stays unity vector
            polss = pols[:, 0]**2 + pols[:, 1]**2 + pols[:, 2]**2
            self.assertTrue(np.allclose(polss, 1, atol=0.0001, rtol=0))

            # pol and s are always perpendicular
            cross = misc.cross(pols, ss)
            crosss = cross[:, 0]**2 + cross[:, 1]**2 + cross[:, 2]**2
            self.assertTrue(np.allclose(crosss, 1, atol=0.0001, rtol=0))
    
    def test_object_collision(self):

        # make raytracer
        RT = ot.Raytracer(outline=[-10, 10, -10, 10, -10, 60])

        # add Raysource
        RSS = ot.Point()
        RS = ot.RaySource(RSS, divergence="Isotropic",
                          pos=[0, 0, 0], s=[0, 0, 1], div_angle=75)
        RT.add(RS)
        
        geom = ot.presets.geometry.arizona_eye()
        geom.elements[2].move_to([0, 0, 0.15])
        RT.add(geom)

        RT.trace(1000)
        self.assertTrue(RT.geometry_error)  # object collision
        self.assertEqual(RT.rays.N, 0)  # not traced

if __name__ == '__main__':
    unittest.main()
