#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import pytest
import scipy.interpolate

import optrace as ot
from optrace.tracer import color



# TODO
class ConvolutionTests(unittest.TestCase):

    def test_exceptions(self):

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf, _ = ot.presets.psf.halo()
        s_psf = [0.1, 0.3]

        # check that call is correct, also tests threading
        for threading in [True, False]:
            ot.convolve(img, s_img, psf, s_psf, silent=True, threading=threading)

        # check invalid types
        self.assertRaises(TypeError, ot.convolve, 5, s_img, psf, s_psf)  # invalid img
        self.assertRaises(TypeError, ot.convolve, img, 5, psf, s_psf)  # invalid s_img
        self.assertRaises(TypeError, ot.convolve, img, s_img, 5, s_psf)  # invalid psf
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, 5)  # invalid s_psf
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, k=[])  # invalid k
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, silent=[])  # invalid silent
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, threading=[])  # invalid threading
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, m=[])  # invalid magnification
        
        # value errors
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf=[0, 1])  # s_psf x value 0
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf=[1, 0])  # s_psf y value 0
        self.assertRaises(ValueError, ot.convolve, img, [0, 1], psf, s_psf)  # s_img x value 0
        self.assertRaises(ValueError, ot.convolve, img, [1, 0], psf, s_psf)  # s_img y value 0
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf, m=0)  # m can't be zero
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf, k=0)  # k <= 0
        

        # resolution tests
        ###########################################

        # low resolution psf
        psf2 = np.ones((5, 5, 3))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # low resolution image
        img2 = np.ones((5, 5, 3))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)
        
        # huge resolution image
        img2 = np.ones((3000, 3000, 3))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)
        
        # huge resolution psf
        psf2 = np.ones((3000, 3000, 3))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # psf much larger than image
        s_psf2 = [s_img[0]*3, s_img[1]*3]
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf2)
       
        # shape tests
        ###############################################

        # invalid psf shape case 1
        psf2 = np.ones((300, 300, 2))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # invalid psf shape case 2
        psf2 = np.ones((300, 2))
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf2, s_psf)
        
        # invalid image shape case 1
        img2 = np.ones((300, 300, 2))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)
        
        # invalid image shape case 2
        img2 = np.ones((300, 2))
        self.assertRaises(ValueError, ot.convolve, img2, s_img, psf, s_psf)

        # color test
        #############################################

        # psf can't be a colored sRGB
        psf2 = ot.presets.image.color_checker
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf2, s_psf)
       
        # coverage tests
        #############################################
       
        # low resolution psf warning
        psf2 = np.ones((90, 90))
        for sil in [True, False]:
            ot.convolve(img, s_img, psf2, s_psf, silent=sil)
        
        # low resolution image warning
        img2 = np.ones((90, 90, 3))
        for sil in [True, False]:
            ot.convolve(img2, s_img, psf, s_psf, silent=sil)
        

    def test_point_psf(self):
        """test if convolution with a point produces the same image"""

        image = ot.presets.image.cell
        s_img = [1, 1]
        s_psf = [1e-9, 1e-9]

        # image with point

        # odd and even pixel size
        for psf_s in [(200, 200), (201, 201)]:
            psf = np.ones(psf_s)

            img2, s2, dbg = ot.convolve(image, s_img, psf, s_psf, k=3, silent=True)

            # compare
            mdiff = np.max(img2[2:-2, 2:-2] - dbg["img"])
            self.assertTrue(mdiff**2.2 < 2e-5)  # should stay the same, but small numerical errors


        # point with point

        psf = np.zeros((301, 301, 3))
        psf[151, 151] = 1
        img = psf.copy()
        psf = psf[:, :, 0]

        img2, s2, dbg = ot.convolve(img, s_img, psf, s_psf)

        # point stays at center
        self.assertAlmostEqual(img2[img2.shape[0]//2+1, img2.shape[1]//2+1, 1], 1)

    def test_behavior_basic(self):
        # test odd and even pixel counts, different pixel ratios
        pass


    def test_0intensity(self):


        psf, s_psf = ot.presets.psf.gaussian(d=1)
        img, s_img = ot.presets.psf.gaussian(d=1)

        img0 = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img = color.srgb_linear_to_srgb(img0)

        # img = np.ones_like(psf)
        # img[:, :, :] *= 0
        # psf[:, :] *= 0
        # img[200, 200] = 1
        # psf[200, 200] = 1
        # s_psf = [5e-4, 5e-4]
        # psf = psf **1000000
        # img = img **1000000

        img2, s2, dbg = ot.convolve(img, s_img, psf, s_psf, k=3)


        img2 = color.srgb_to_srgb_linear(img2)
        img2 = img2[:, :, 0]


        import scipy.signal
        img22 = scipy.signal.fftconvolve(img0[:, :, 0], psf, mode="same")
        img22 /= np.max(img22)
        # input()

        # sx2 = (s2[0] + s2[0]/img.shape[1]/1.5)/2
        # sy2 = (s2[1] + s2[1]/img.shape[0]/1.5)/2
        X, Y = np.mgrid[-s2[1]/2:s2[1]/2:img2.shape[0]*1j, -s2[0]/2:s2[0]/2:img2.shape[1]*1j]
        # X, Y = np.mgrid[-s2[1]/2:sy2:img2.shape[0]*1j, -s2[0]/2:sx2:img2.shape[1]*1j]

        # print(s_img, s2)

        R2 = X**2 + Y**2
        # Z = np.exp(-3.265306122e6*(R2))
        Z = np.exp(-8.163265306e6*(R2))
        # Z /= np.max(Z)


        return

        # TODO why is there a difference?
        # for some reasons one of the images is not centered

        diff = img2 - Z

        print(s2, img2.shape, s_img, img.shape)

        import matplotlib.pyplot as plt

        import optrace.plots as otp
        otp.image_plot(img, s_img)
        otp.image_plot(img2, s2)
        # otp.image_plot(psf, s_psf)
        otp.convolve_debug_plots(img2, s2, dbg)

        plt.figure()
        plt.imshow(diff)
        plt.show(block=False)
        plt.figure()
        plt.imshow(img2)
        plt.show(block=False)
        plt.figure()
        plt.imshow(Z)
        plt.show(block=False)

        where = np.where(diff == np.max(diff))
        print(where, img2[where[0][0], where[1][0]], Z[where[0][0], where[1][0]])

        print(np.max(img2), np.max(Z), np.max(img2 - Z))

        input()


    @pytest.mark.slow
    def test_tracing_consistency(self):
        """
        checks that psf convolution produces the same result as tracing.
        This function also checks the behavior and correctness of a colored PSF
        """

        # i == 0: colored psf and bw image
        # i == 1: bw psf and colored image
        # i == 2: both bw
        for i in range(3):

            # raytracer and raysource
            RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40], no_pol=True, silent=True)
            RS = ot.RaySource(ot.Point(), divergence="Lambertian", div_angle=2, s=[0, 0, 1], pos=[0, 0, 0])
            RT.add(RS)

            # add Lens
            front = ot.SphericalSurface(r=3, R=8)
            back = ot.SphericalSurface(r=3, R=-8)
            nL1 = ot.RefractionIndex("Abbe", n=1.5, V=120) if i == 0 else ot.RefractionIndex("Constant", n=1.5)
            L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
            RT.add(L1)

            # add Detector
            DetS = ot.RectangularSurface(dim=[10, 10])
            Det = ot.Detector(DetS, pos=[0, 0, 36.95])
            Det.move_to([0, 0, 38.571])
            RT.add(Det)

            # render PSF
            RT.trace(2e6)
            img = RT.detector_image(945, extent=[-0.07, 0.07, -0.07, 0.07])
            psf = img.copy()
            s_psf = [img.extent[1]-img.extent[0], img.extent[3]-img.extent[2]]

            # image
            img = ot.presets.image.ETDRS_chart_inverted if i == 0 or i == 2 else ot.presets.image.color_checker
            s_img = [0.07, 0.07]

            # swap old source for image source
            RT.remove(RS)
            RSS = ot.RectangularSurface(dim=s_img) 
            RS = ot.RaySource(RSS, divergence="Lambertian", image=img, div_angle=2, s=[0, 0, 1], pos=[0, 0, 0])
            RT.add(RS)

            # rendered image
            _, det_im = RT.iterative_render(30e6, 189, extent=[-0.12, 0.12, -0.12, 0.12],  no_sources=True)
            img_ren0 = det_im[0]
            img_ren = img_ren0.get("sRGB (Absolute RI)")
            s_img_ren = [img_ren0.extent[1]-img_ren0.extent[0], img_ren0.extent[3]-img_ren0.extent[2]]

            # convolution image
            mag = 2.232751
            img_conv, s_img_conv, _ = ot.convolve(img, [s_img[0]*mag, s_img[1]*mag], psf, s_psf, silent=True)

            # spacing vectors
            x = np.linspace(-s_img_conv[0]/2, s_img_conv[0]/2, img_conv.shape[1])
            y = np.linspace(-s_img_conv[1]/2, s_img_conv[1]/2, img_conv.shape[0])
            xi = np.linspace(-s_img_ren[0]/2, s_img_ren[0]/2, img_ren.shape[1])
            yi = np.linspace(-s_img_ren[1]/2, s_img_ren[1]/2, img_ren.shape[0])

            # interpolate on convoluted image and subtract from rendered one channel-wise
            im_diff = np.fliplr(np.flipud(img_ren)).copy()
            im_diff[:, :, 0] -= scipy.interpolate.RectBivariateSpline(x, y, img_conv[:, :, 0].T, kx=2, ky=2)(xi, yi).T
            im_diff[:, :, 1] -= scipy.interpolate.RectBivariateSpline(x, y, img_conv[:, :, 1].T, kx=2, ky=2)(xi, yi).T
            im_diff[:, :, 2] -= scipy.interpolate.RectBivariateSpline(x, y, img_conv[:, :, 2].T, kx=2, ky=2)(xi, yi).T
            im_diff -= np.mean(im_diff)

            # import optrace.plots as otp

            # otp.image_plot(img_ren, s_img_ren, flip=True)
            # otp.image_plot(img_conv, s_img_conv)
            # otp.image_plot(np.abs(im_diff), s_img_ren, block=True)

            # check that mean difference is small
            # we still have some deviations due to noise, incorrect magnification and not-linear aberrations
            # print(np.mean(np.abs(im_diff)))
            self.assertAlmostEqual(np.mean(np.abs(im_diff)), 0, delta=0.02)

    @pytest.mark.slow
    def test_size_consistency(self):

        img = ot.presets.image.ETDRS_chart_inverted
        s_img = [1, 1]

        for s_psf in [[0.001, 0.001], [0.01, 0.01], [0.1, 0.1], [0.99, 0.99]]:
            
            img2o = None
            
            for i, res in enumerate([2000, 400, 51]):

                psf, _ = ot.presets.psf.gaussian()
                img2, s, dbg = ot.convolve(img, s_img, psf, s_psf, k=5, silent=True)

                if i == 0:
                    img2o = img2.copy()
                else:
                    dev = np.mean(np.abs(img2-img2o))
                
                    if i == 1:
                        self.assertAlmostEqual(dev, 0.0, delta=1e-3)
                    elif i == 2:
                        self.assertAlmostEqual(dev, 0.0, delta=5e-3)

        # test that different image/psf resolutions produce approximately the same result

    def test_psf_presets(self):

        psf = ot.presets.psf
        psts = [psf.circle(), psf.gaussian(), psf.halo(), psf.glare(), psf.airy()]

        img = ot.presets.image.ETDRS_chart_inverted
        ilen = [3, 3]

        for pst in psts:

            # small psfs, units in mms
            self.assertTrue(pst[1][0] < 3e-2)
            self.assertTrue(pst[1][1] < 3e-2)

            # square psf
            self.assertTrue(pst[0].shape[0] == pst[0].shape[1])

            # range 0-1
            self.assertAlmostEqual(np.min(pst[0]), 0)
            self.assertAlmostEqual(np.max(pst[0]), 1)

            # convolution doesn't fail
            ot.convolve(img, ilen, *pst, silent=True)

        # TODO how to check shape/curve?

    def test_presets_exceptions(self):

        self.assertRaises(ValueError, ot.presets.psf.airy, -1)  # invalid d
        self.assertRaises(ValueError, ot.presets.psf.circle, -1)  # invalid d
        self.assertRaises(ValueError, ot.presets.psf.gaussian, -1)  # invalid d

        self.assertRaises(ValueError, ot.presets.psf.glare, -1)  # invalid d1
        self.assertRaises(ValueError, ot.presets.psf.glare, 1, -2)  # invalid d2
        self.assertRaises(ValueError, ot.presets.psf.glare, 1, 2, -1)  # invalid a
        self.assertRaises(ValueError, ot.presets.psf.glare, 2, 1)  # d2 < d1

        self.assertRaises(ValueError, ot.presets.psf.halo, -1)  # invalid d1
        self.assertRaises(ValueError, ot.presets.psf.halo, 1, -2)  # invalid d2
        self.assertRaises(ValueError, ot.presets.psf.halo, 1, 2, -1)  # invalid s
        self.assertRaises(ValueError, ot.presets.psf.halo, 1, 2, 1, -1)  # invalid a
        self.assertRaises(ValueError, ot.presets.psf.halo, 2, 1)  # d2 < d1

    def test_zero_image(self):

        img = np.zeros((200, 200, 3))
        s_img = [5, 5]
        psf, s_psf = ot.presets.psf.glare()

        img2, s2, _ = ot.convolve(img, s_img, psf, s_psf)
        self.assertEqual(np.max(img2), 0)
    
    def test_zero_psf(self):

        img = np.zeros((200, 200, 3))
        img = ot.presets.image.color_checker
        s_img = [5, 5]
        psf = np.zeros((200, 200))
        s_psf = [1, 1]

        img2, s2, _ = ot.convolve(img, s_img, psf, s_psf)
        self.assertEqual(np.max(img2), 0)

    def test_coverage(self):
        # test silent, threading, inter_p_k parameter
        pass


    def test_m_behavior(self):
        """test the behavior of the magnification m regarding sign and value"""

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf, _ = ot.presets.psf.halo()
        s_psf = [1, 0.3]

        # m = -1 means flip the output image
        img2, s2, dbg = ot.convolve(img, s_img, psf, s_psf, m=1)
        img2_f, s2_f, dbg = ot.convolve(img, s_img, psf, s_psf, m=-1)
        self.assertEqual(s2, s2_f)
        self.assertTrue(np.allclose(img2 - np.fliplr(np.flipud(img2_f)), 0))

        # a m = 2 is the same as scaling s_img by 2
        img2, s2, dbg = ot.convolve(img, [2*s_img[0], 2*s_img[1]], psf, s_psf, m=1)
        img2_s, s2_s, dbg = ot.convolve(img, s_img, psf, s_psf, m=2)
        self.assertEqual(s2, s2_s)
        self.assertTrue(np.allclose(img2 - img2_s, 0))

if __name__ == '__main__':
    unittest.main()
