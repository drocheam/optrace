#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import numexpr as ne
import pytest
import scipy.interpolate

import optrace as ot
from optrace.tracer import color
import optrace.tracer.misc as misc



class ConvolutionTests(unittest.TestCase):

    def test_exceptions(self):

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf, _ = ot.presets.psf.halo()
        s_psf = [0.1, 0.3]

        # check invalid types
        self.assertRaises(TypeError, ot.convolve, 5, s_img, psf, s_psf)  # invalid img
        self.assertRaises(TypeError, ot.convolve, img, 5, psf, s_psf)  # invalid s_img
        self.assertRaises(TypeError, ot.convolve, img, s_img, 5, s_psf)  # invalid psf
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, 5)  # invalid s_psf
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, rendering_intent=[])  
        # ^-- invalid rendering_intent
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, silent=[])  # invalid silent
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, threading=[])  # invalid threading
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf, s_psf, m=[])  # invalid magnification
        
        # value errors
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf=[0, 1])  # s_psf x value 0
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf=[1, 0])  # s_psf y value 0
        self.assertRaises(ValueError, ot.convolve, img, [0, 1], psf, s_psf)  # s_img x value 0
        self.assertRaises(ValueError, ot.convolve, img, [1, 0], psf, s_psf)  # s_img y value 0
        self.assertRaises(ValueError, ot.convolve, img, s_img, psf, s_psf, m=0)  # m can't be zero
        
    def test_resolution_exceptions(self):

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf, _ = ot.presets.psf.halo()
        s_psf = [0.1, 0.3]

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
       
    def test_shape_exceptions(self):

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf, _ = ot.presets.psf.halo()
        s_psf = [0.1, 0.3]

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

    def test_coverage(self):

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf, _ = ot.presets.psf.halo()
        s_psf = [0.1, 0.3]
        
        # check that call is correct, also tests threading
        for threading in [True, False]:
            ot.convolve(img, s_img, psf, s_psf, silent=True, threading=threading)

        # color test
        #############################################

        # psf can't be a colored sRGB
        psf2 = ot.presets.image.color_checker
        self.assertRaises(TypeError, ot.convolve, img, s_img, psf2, s_psf)
       
        # coverage tests
        #############################################
      
        # pixels are strongly non-square
        for sil in [True, False]:
            ot.convolve(img, s_img, psf, [1e-9, 1e-8], silent=sil)
            ot.convolve(img, [1, 10], psf, s_psf, silent=sil)
        
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

            img2, s2 = ot.convolve(image, s_img, psf, s_psf, silent=True)

            # compare
            img = misc.load_image(image)
            dsx = (img2.shape[1] - img.shape[1]) // 2
            dsy = (img2.shape[0] - img.shape[0]) // 2
            mdiff = np.max(img2[dsy:-dsy, dsx:-dsx] - img)
            self.assertTrue(mdiff**2.2 < 2e-5)  # should stay the same, but small numerical errors

    def test_point_image_point_psf(self):

        psf = np.zeros((301, 301, 3))
        psf[151, 151] = 1
        img = psf.copy()
        psf = psf[:, :, 0]
        s_img = [1, 1]
        s_psf = [1e-9, 1e-9]

        img2, s2 = ot.convolve(img, s_img, psf, s_psf)

        # point stays at center
        self.assertAlmostEqual(img2[img2.shape[0]//2+1, img2.shape[1]//2+1, 1], 1)

    def test_coordinate_value_correctness(self):
        """
        this function checks that convolution produces the mathematical correct result
        by convolving two gaussians and comparing it to the correct result
        Check the image bounds and check odd and even, low and large pixel numbers
        """
        def _gaussian(d, sz):
            sig = 0.175  # sigma so approximating zeroth order of airy disk
            ds = 5*sig  # plot 5 sigma

            X, Y = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
            Z = ne.evaluate("exp(-(X**2 + Y**2) / 2 / sig**2)")

            return Z, [2*ds*d/1000, 2*ds*d/1000]  # scale size with d

        # check for different image sizes
        for sz, delta in zip([400, 401, 1999, 2000], [5e-7, 5e-5, 1e-5, 1e-7]):

            # two gaussians
            psf, s_psf = _gaussian(1, sz)
            img, s_img = _gaussian(2, sz)

            # make sRGB image
            img0 = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            img = color.srgb_linear_to_srgb(img0)

            # convolution
            img2, s2 = ot.convolve(img, s_img, psf, s_psf)

            # convert back to intensities
            img2 = color.srgb_to_srgb_linear(img2)
            img2 = img2[:, :, 0]

            # resulting gaussian with convolution
            X, Y = np.mgrid[-s2[1]/2:s2[1]/2:img2.shape[0]*1j, -s2[0]/2:s2[0]/2:img2.shape[1]*1j]
            R2 = X**2 + Y**2
            Z = np.exp(-3.265306122449e6*(R2))

            # normalize both images
            Z /= np.max(Z)
            img2 /= np.max(img2)

            # compare
            diff = img2 - Z
            # print(np.mean(np.abs(diff)))
            self.assertAlmostEqual(np.mean(np.abs(diff)), 0, delta=delta)  # difference small
            self.assertAlmostEqual(np.mean(np.abs(diff-diff.T)), 0, delta=1e-12)  # rotationally symmetric

            # TODO there seems to be some remaining error
            # similar to an integration error that becomes smaller and smaller for a larger resolution

            # visualize
            # import matplotlib.pyplot as plt
            # import optrace.plots as otp

            # plt.figure()
            # plt.imshow(diff)
            # plt.show(block=False)

    @pytest.mark.slow
    def test_tracing_consistency(self):
        """
        checks that psf convolution produces the same result as tracing.
        This function also checks the behavior and correctness of a colored PSF
        """

        # i == 0: colored psf and bw image
        # i == 1: bw psf and colored image
        # i == 2: both bw
        # i == 3: both colored
        for i in range(4):

            # raytracer and raysource
            RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40], no_pol=True, silent=False)
            RS = ot.RaySource(ot.Point(), divergence="Lambertian", div_angle=2, s=[0, 0, 1], pos=[0, 0, 0])
            RT.add(RS)

            # add Lens
            front = ot.SphericalSurface(r=3, R=8)
            back = ot.SphericalSurface(r=3, R=-8)
            nL1 = ot.RefractionIndex("Abbe", n=1.5, V=120) if (i == 0 or i == 3)\
                    else ot.RefractionIndex("Constant", n=1.5)
            L1 = ot.Lens(front, back, de=0.1, pos=[0, 0, 12], n=nL1)
            RT.add(L1)

            # add Detector
            DetS = ot.RectangularSurface(dim=[10, 10])
            Det = ot.Detector(DetS, pos=[0, 0, 36.95])
            Det.move_to([0, 0, 38.571])
            RT.add(Det)

            # render PSF
            RT.trace(2e6)
            img = RT.detector_image(345, extent=[-0.07, 0.07, -0.07, 0.07])
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
            mag = 2.222751
            img_conv, s_img_conv = ot.convolve(img, [s_img[0]*mag, s_img[1]*mag], psf, s_psf, silent=True)

            # spacing vectors
            x = np.linspace(-s_img_conv[0]/2, s_img_conv[0]/2, img_conv.shape[1])
            y = np.linspace(-s_img_conv[1]/2, s_img_conv[1]/2, img_conv.shape[0])
            xi = np.linspace(-s_img_ren[0]/2, s_img_ren[0]/2, img_ren.shape[1])
            yi = np.linspace(-s_img_ren[1]/2, s_img_ren[1]/2, img_ren.shape[0])

            # interpolate on convoluted image and subtract from rendered one channel-wise
            img_ren = color.srgb_to_srgb_linear(img_ren)
            img_conv = color.srgb_to_srgb_linear(img_conv)
            im_diff = np.fliplr(np.flipud(img_ren)).copy()
            im_diff[:, :, 0] -= scipy.interpolate.RectBivariateSpline(x, y, img_conv[:, :, 0].T, kx=1, ky=1)(xi, yi).T
            im_diff[:, :, 1] -= scipy.interpolate.RectBivariateSpline(x, y, img_conv[:, :, 1].T, kx=1, ky=1)(xi, yi).T
            im_diff[:, :, 2] -= scipy.interpolate.RectBivariateSpline(x, y, img_conv[:, :, 2].T, kx=1, ky=1)(xi, yi).T
            im_diff -= np.mean(im_diff)


            # check that mean difference is small
            # we still have some deviations due to noise, incorrect magnification and not-linear aberrations
            print(np.mean(np.abs(im_diff)))
            delta = 0.0075 if i not in [1, 3] else 0.025  # larger errors for bright colored color checker image due too not enough rays
            # self.assertAlmostEqual(np.mean(np.abs(im_diff)), 0, delta=delta)
            
            # import optrace.plots as otp
            # otp.image_plot(img_ren, s_img_ren, flip=True)
            # otp.image_plot(img_conv, s_img_conv)
            # otp.image_plot(np.abs(im_diff), s_img_ren, block=True)

    def test_white_balance(self):
        """
        when different color components have different frequencies, 
        in a naive case higher frequencies of one color would be deleted due to 
        a finite resolution of image compared to the PSF. But this should be handled when convolving
        """

        # generate color image with high frequency red components
        rimg = ot.RImage([-1e-6, 1e-6, -1e-6, 1e-6])
        rimg.render(945)
        rimg._img[:, :] = 0.5
        rimg._img[:2, :2, 2] = 1
        rimg._img[-2:, -2:, 2] = 0
        assert(np.std(np.mean(rimg._img[:, :, :3], axis=(0, 1))) < 1e-6)  # mean color is white
        rimg._img[:, :, :3] = color.srgb_linear_to_xyz(rimg._img[:, :, :3])  # convert to XYZ

        # convolve with white image
        img = ot.presets.image.ETDRS_chart
        s_img = [1, 1]
        s_psf = [2*rimg.extent[1], 2*rimg.extent[3]]

        # convolve and convert to linear sRGB
        img2, s2 = ot.convolve(img, s_img, rimg, s_psf)
        img2l = color.srgb_to_srgb_linear(img2)

        # mean color of new image should also be white
        cerr = np.std(np.mean(img2l, axis=(0, 1)))
        self.assertAlmostEqual(cerr, 0, delta=1e-6)

    @pytest.mark.slow
    def test_size_consistency(self):
        """test that different image/psf resolutions produce approximately the same result"""

        img = ot.presets.image.ETDRS_chart_inverted
        s_img = [1, 1]

        for s_psf in [[0.001, 0.001], [0.01, 0.01], [0.1, 0.1], [0.99, 0.99]]:
            
            img2o = None
           
            for psf_func in [ot.presets.psf.gaussian, ot.presets.psf.halo]:
                for i, res in enumerate([2000, 1999, 400, 399, 52, 51]):

                    psf, _ = psf_func()
                    img2, s = ot.convolve(img, s_img, psf, s_psf, silent=True)

                    if i == 0:
                        img2o = img2.copy()
                    else:
                        dev = np.mean(np.abs(img2-img2o))
                    
                        if i == 1:
                            self.assertAlmostEqual(dev, 0.0, delta=1e-3)
                        elif i == 2:
                            self.assertAlmostEqual(dev, 0.0, delta=5e-3)

    def test_psf_presets(self):
        """check preset units, shape and value range"""

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

    def test_presets_exceptions(self):
        """check handling of incorrect parameters to psf presets"""

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
        """no warnings/exceptions with zero image"""

        img = np.zeros((200, 200, 3))
        s_img = [5, 5]
        psf, s_psf = ot.presets.psf.glare()

        img2, s2 = ot.convolve(img, s_img, psf, s_psf)
        self.assertEqual(np.max(img2), 0)
    
    def test_zero_psf(self):
        """no warnings/exceptions with zero psf"""

        img = np.zeros((200, 200, 3))
        img = ot.presets.image.color_checker
        s_img = [5, 5]
        psf = np.zeros((200, 200))
        s_psf = [1, 1]

        img2, s2 = ot.convolve(img, s_img, psf, s_psf)
        self.assertEqual(np.max(img2), 0)

    def test_m_behavior(self):
        """test the behavior of the magnification m regarding sign and value"""

        img = ot.presets.image.color_checker
        s_img = [5, 6]
        psf, _ = ot.presets.psf.halo()
        s_psf = [1, 0.3]

        # m = -1 means flip the output image
        img2, s2 = ot.convolve(img, s_img, psf, s_psf, m=1)
        img2_f, s2_f = ot.convolve(img, s_img, psf, s_psf, m=-1)
        self.assertEqual(s2, s2_f)
        self.assertTrue(np.allclose(img2 - np.fliplr(np.flipud(img2_f)), 0))

        # a m = 2 is the same as scaling s_img by 2
        img2, s2 = ot.convolve(img, [2*s_img[0], 2*s_img[1]], psf, s_psf, m=1)
        img2_s, s2_s = ot.convolve(img, s_img, psf, s_psf, m=2)
        self.assertEqual(s2, s2_s)
        self.assertTrue(np.allclose(img2 - img2_s, 0))

    def test_channel_orthogonality(self):
        """
        Checks that a red image convolved with a PSF missing red produces a zero image.
        Also tests that normalize=False leads to small values
        """

        # generate PSF with missing red component
        rimg = ot.RImage([-1e-6, 1e-6, -1e-6, 1e-6])
        rimg.render(945)
        rimg._img[:, :] = 1
        rimg._img[:, :, 0] = 0
        rimg._img[:, :, :3] = color.srgb_linear_to_xyz(rimg._img[:, :, :3])  # convert to XYZ

        # convolve with red noise image
        img = np.random.sample((1000, 1000, 3))
        img[:, :, 1:] = 0
        s_img = [1, 1]
        s_psf = [2*rimg.extent[1], 2*rimg.extent[3]]

        # convolve while not normalizing the output values
        img2, s2 = ot.convolve(img, s_img, rimg, s_psf, normalize=False)

        # check that image is almost empty (besides numerical errors)
        self.assertAlmostEqual(np.mean(img2), 0, delta=1e-06)

if __name__ == '__main__':
    unittest.main()
