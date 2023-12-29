#!/bin/env python3

import sys
sys.path.append('.')

import unittest
import numpy as np
import pytest
import scipy.interpolate
import scipy.ndimage

import optrace as ot
from optrace.tracer import color
import optrace.tracer.misc as misc


# test slice_ and padding properties

class ConvolutionTests(unittest.TestCase):

    def test_exceptions(self):

        img = ot.presets.image.color_checker([5, 6])
        psf = ot.presets.psf.halo()
        psf.s = [0.1, 0.3]

        # check invalid types
        self.assertRaises(TypeError, ot.convolve, 5, psf)  # invalid img
        self.assertRaises(TypeError, ot.convolve, img, 5)  # invalid psf
        self.assertRaises(TypeError, ot.convolve, img, psf, m=[])  # invalid magnification
        self.assertRaises(TypeError, ot.convolve, img, psf, slice_=[])  # invalid slice_
        self.assertRaises(TypeError, ot.convolve, img, psf, cargs=[])  # invalid cargs
        self.assertRaises(TypeError, ot.convolve, img, psf, padding_mode=[])  # invalid padding_mode
        self.assertRaises(TypeError, ot.convolve, img, psf, padding_value=2)  # invalid padding_value
        
        # value errors
        self.assertRaises(ValueError, ot.convolve, img, psf, m=0)  # m can't be zero
        self.assertRaises(ValueError, ot.convolve, img, psf, padding_value=[0, 0])  # invalid padding_value shape
        self.assertRaises(ValueError, ot.convolve, img, psf, padding_value=[[0, 0, 0]])  # invalid padding_value shape
        
    def test_resolution_exceptions(self):

        img = ot.presets.image.color_checker([5, 6])
        psf = ot.presets.psf.halo()
        psf.s = [0.1, 0.3]
        s_psf = psf.s
        s_img = img.s

        # resolution tests
        ###########################################

        # low resolution psf
        psf2 = ot.Image(np.ones((5, 5, 3)), s_psf)
        self.assertRaises(ValueError, ot.convolve, img, psf2)
        
        # low resolution image
        img2 = ot.Image(np.ones((5, 5, 3)), s_img)
        self.assertRaises(ValueError, ot.convolve, img2, psf)
        
        # huge resolution image
        img2 = ot.Image(np.ones((3000, 3000, 3)), s_img)
        self.assertRaises(ValueError, ot.convolve, img2, psf)
        
        # huge resolution psf
        psf2 = ot.Image(np.ones((3000, 3000, 3)), s_psf)
        self.assertRaises(ValueError, ot.convolve, img, psf2)
        
        # psf much larger than image
        psf3 = ot.Image(psf2.data, [s_img[0]*3, s_img[1]*3])
        self.assertRaises(ValueError, ot.convolve, img, psf3)
       
    def test_coverage(self):

        img = ot.presets.image.color_checker([5, 6])
        psf = ot.presets.psf.halo()
        psf.s = s_psf = [0.1, 0.3]
        s_img = img.s
        
        # check that call is correct, also tests threading
        for threading in [False, True]:
            ot.global_options.multithreading = threading
            ot.convolve(img, psf)

        # color test
        #############################################

        # psf can't be a colored sRGB
        psf2 = ot.presets.image.color_checker(s_psf)
        self.assertRaises(ValueError, ot.convolve, img, psf2)
       
        # coverage tests
        #############################################
        img = ot.presets.image.color_checker([1, 1]).data

        # pixels are strongly non-square
        ot.convolve(ot.Image(img, s_img), ot.Image(psf.data, [1e-9, 1e-8]))
        ot.convolve(ot.Image(img, [1, 10]), ot.Image(psf.data, s_psf))
        
        # low resolution psf warning
        psf2 = np.ones((90, 90, 3))
        ot.convolve(ot.Image(img, s_img), ot.Image(psf2, s_psf))
        
        # low resolution image warning
        img2 = np.ones((90, 90, 3))
        ot.convolve(ot.Image(img2, s_img), psf)
        
    def test_point_psf(self):
        """test if convolution with a point produces the same image"""

        s_img = [1, 1]
        image = ot.presets.image.cell(s_img)
        s_psf = [1e-9, 1e-9]

        # image with point

        # odd and even pixel size
        for padding in ["constant", "edge"]:
            for slice_ in [False, True]:
                for psf_s in [(200, 200, 3), (201, 201, 3)]:
                    psf = np.ones(psf_s)

                    img2_ = ot.convolve(image, ot.Image(psf, s_psf), slice_=slice_, padding_mode=padding)
                    img2, s2 = img2_.data, img2_.s

                    # compare
                    dsx = (img2.shape[1] - image.shape[1]) // 2
                    dsy = (img2.shape[0] - image.shape[0]) // 2
                    img2p = img2[dsy:-dsy, dsx:-dsx] if not slice_ else img2
                    mdiff = np.max(img2p - image.data)
                    self.assertTrue(mdiff**2.2 < 2e-5)  # should stay the same, but small numerical errors

    def test_point_image_point_psf(self):

        psf = np.zeros((301, 301, 3))
        psf[151, 151] = 1
        img = psf.copy()
        s_img = [1, 1]
        s_psf = [1e-9, 1e-9]

        for padding in ["constant", "edge"]:  
        # with default settings ("constant" and padding_value=0) no 
        # padding takes place, as scipy.fftconvolve automatically pads with black
            for slice_ in [False, True]:
                img2_ = ot.convolve(ot.Image(img, s_img), ot.Image(psf, s_psf), slice_=slice_, padding_mode=padding)
                img2, s2 = img2_.data, img2_.s

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

            Y, X = np.mgrid[-ds:ds:sz*1j, -ds:ds:sz*1j]
            Z = np.exp(-(X**2 + Y**2) / 2 / sig**2)
            Z = np.repeat(Z[:, :, np.newaxis], 3, axis=2)

            return Z, [2*ds*d/1000, 2*ds*d/1000]  # scale size with d

        # check for different image sizes
        for sz in [400, 401, 1999, 2000]:

            # two gaussians
            psf, s_psf = _gaussian(1, sz)
            img, s_img = _gaussian(2, sz)

            # make sRGB image
            img = color.srgb_linear_to_srgb(img)

            # convolution
            img2_ = ot.convolve(ot.Image(img, s_img), ot.Image(psf, s_psf))
            img2, s2 = img2_.data, img2_.s

            # convert back to intensities
            img2 = color.srgb_to_srgb_linear(img2)
            img2 = img2[:, :, 0]

            # resulting gaussian with convolution
            # s2 += np.array([s2[0]/img2.shape[1], s2[1]/img2.shape[0]])
            Y, X = np.mgrid[-s2[0]/2:s2[0]/2:img2.shape[1]*1j, -s2[1]/2:s2[1]/2:img2.shape[0]*1j]
            R2 = X**2 + Y**2
            Z = np.exp(-3.265306122449e6*(R2))

            # normalize both images
            Z /= np.max(Z)
            img2 /= np.max(img2)

            # compare
            diff = img2 - Z
            self.assertAlmostEqual(np.mean(np.abs(diff)), 0, delta=5e-05)  # difference small
            self.assertAlmostEqual(np.mean(np.abs(diff-diff.T)), 0, delta=1e-12)  # rotationally symmetric

            # TODO there seems to be some remaining error
            # similar to an integration error that becomes smaller and smaller for a larger resolution

            # visualize
            # import matplotlib.pyplot as plt
            # import optrace.plots as otp

            # plt.figure()
            # plt.imshow(diff)
            # plt.show(block=True)

    @pytest.mark.slow
    @pytest.mark.timeout(600)
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
            RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 40], no_pol=True)
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
            RT.trace(2000000)
            img = RT.detector_image(345, extent=[-0.07, 0.07, -0.07, 0.07])
            psf = img.copy()

            # image
            img = ot.presets.image.ETDRS_chart_inverted if i == 0 or i == 2 else ot.presets.image.color_checker
            s_img = [0.07, 0.07]

            # swap old source for image source
            RT.remove(RS)
            RS = ot.RaySource(img(s_img), divergence="Lambertian", div_angle=2, s=[0, 0, 1], pos=[0, 0, 0])
            RT.add(RS)

            # rendered image
            _, det_im = RT.iterative_render(30e6, 189, extent=[-0.12, 0.12, -0.12, 0.12],  no_sources=True)
            img_ren0 = det_im[0]
            img_ren = img_ren0.get("sRGB (Absolute RI)")
            s_img_ren = [img_ren0.extent[1]-img_ren0.extent[0], img_ren0.extent[3]-img_ren0.extent[2]]

            # convolution image
            mag = 2.232751
            mag = 2.222751
            img_conv_ = ot.convolve(img([s_img[0]*mag, s_img[1]*mag]), psf)
            img_conv, s_img_conv  = img_conv_.data, img_conv_.s

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
            # print(np.mean(np.abs(im_diff)))
            delta = 0.0075 if i not in [1, 3] else 0.03  # larger errors for bright colored color checker image due too not enough rays
            # print(delta, np.mean(np.abs(im_diff)))
            self.assertAlmostEqual(np.mean(np.abs(im_diff)), 0, delta=delta)
     
    def test_white_balance(self):
        """
        when different color components have different frequencies, 
        in a naive case higher frequencies of one color would be deleted due to 
        a finite resolution of image compared to the PSF. But this should be handled when convolving
        """
        # make sure psf normalization has no effect
        for normalize in [True, False]:

            # generate color image with high frequency red components
            rimg = ot.RImage([-1e-6, 1e-6, -1e-6, 1e-6])
            rimg.render(945)
            rimg._img[:, :] = 0.5
            rimg._img[:2, :2, 2] = 1
            rimg._img[-2:, -2:, 2] = 0
            assert(np.std(np.mean(rimg._img[:, :, :3], axis=(0, 1))) < 1e-6)  # mean color is white
            rimg._img[:, :, :3] = color.srgb_linear_to_xyz(rimg._img[:, :, :3])  # convert to XYZ

            # convolve with white image
            s_img = [1, 1]
            img = ot.presets.image.ETDRS_chart(s_img)

            # convolve and convert to linear sRGB
            img2_ = ot.convolve(img, rimg, cargs=dict(normalize=normalize))
            img2, s2 = img2_.data, img2_.s
            img2l = color.srgb_to_srgb_linear(img2)

            # mean color of new image should also be white
            cerr = np.std(np.mean(img2l, axis=(0, 1)))
            self.assertAlmostEqual(cerr, 0, delta=1e-6)
           
            ### White balance colored image

            # convolve with colored image
            img = ot.presets.image.group_photo(s_img)

            # convolve and convert to linear sRGB
            img2_ = ot.convolve(img, rimg, cargs=dict(normalize=normalize))
            img2, s2 = img2_.data, img2_.s

            # mean color of new image should be almost the same (small difference because of image padding)
            # sum of all color channels should be roughly the same before and after convolution
            csum = np.sum(np.sum(color.srgb_to_srgb_linear(img.data), axis=0), axis=0)
            csum2 = np.sum(np.sum(color.srgb_to_srgb_linear(img2), axis=0), axis=0)
            # compare std. dev. of ratio so normalization has no impact
            self.assertAlmostEqual(np.std(csum2/csum), 0, delta=1e-6)

    @pytest.mark.slow
    def test_size_consistency(self):
        """test that different image/psf resolutions produce approximately the same result"""

        img = ot.presets.image.ETDRS_chart_inverted([1, 1])

        for s_psf in [[0.001, 0.001], [0.01, 0.01], [0.1, 0.1], [0.99, 0.99]]:
            
            img2o = None
           
            for psf_func in [ot.presets.psf.gaussian, ot.presets.psf.halo]:
                for i, res in enumerate([2000, 1999, 400, 399, 52, 51]):

                    psf = psf_func()
                    img2_ = ot.convolve(img, psf)
                    img2, s2 = img2_.data, img2_.s

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
            self.assertTrue(pst.s[0] < 6e-2)
            self.assertTrue(pst.s[1] < 6e-2)

            # square psf
            self.assertTrue(pst.shape[0] == pst.shape[1])

            # range 0-1
            self.assertAlmostEqual(np.min(pst.data), 0)
            self.assertAlmostEqual(np.max(pst.data), 1)

            # convolution doesn't fail
            ot.convolve(img(ilen), pst)

    def test_presets_exceptions(self):
        """check handling of incorrect parameters to psf presets"""

        self.assertRaises(ValueError, ot.presets.psf.airy, -1)  # invalid d
        self.assertRaises(ValueError, ot.presets.psf.circle, -1)  # invalid d
        self.assertRaises(ValueError, ot.presets.psf.gaussian, -1)  # invalid sig

        self.assertRaises(ValueError, ot.presets.psf.glare, -1)  # invalid sig1
        self.assertRaises(ValueError, ot.presets.psf.glare, 1, -2)  # invalid sig2
        self.assertRaises(ValueError, ot.presets.psf.glare, 1, 2, -1)  # invalid a
        self.assertRaises(ValueError, ot.presets.psf.glare, 2, 1)  # sig2 < sig1

        self.assertRaises(ValueError, ot.presets.psf.halo, -1)  # invalid sig1
        self.assertRaises(ValueError, ot.presets.psf.halo, 1, -2)  # invalid sig2
        self.assertRaises(ValueError, ot.presets.psf.halo, 1, 2, -1)  # invalid r
        self.assertRaises(ValueError, ot.presets.psf.halo, 1, 2, 1, -1)  # invalid a

    def test_zero_image(self):
        """no warnings/exceptions with zero image"""

        img = ot.Image(np.zeros((200, 200, 3)), [5, 5])
        psf = ot.presets.psf.glare()

        img2 = ot.convolve(img, psf)
        self.assertEqual(np.max(img2.data), 0)
    
    def test_zero_psf(self):
        """no warnings/exceptions with zero psf"""

        img = ot.presets.image.color_checker([5, 5])
        psf = ot.Image(np.zeros((200, 200, 3)), [1, 1])

        img2 = ot.convolve(img, psf)
        self.assertEqual(np.max(img2.data), 0)

    def test_m_behavior(self):
        """test the behavior of the magnification m regarding sign and value"""

        img = ot.presets.image.color_checker([5, 6])
        psf = ot.presets.psf.halo()
        psf.s = [1, 0.3]
        s_psf = psf.s
        s_img = img.s

        # m = -1 means flip the output image
        img2_ = ot.convolve(img, psf, m=1)
        img2, s2 = img2_.data, img2_.s
        img2_f_ = ot.convolve(img, psf, m=-1)
        img2_f, s2_f = img2_f_.data, img2_f_.s
        self.assertTrue(np.all(s2 == s2_f))
        self.assertTrue(np.allclose(img2 - np.fliplr(np.flipud(img2_f)), 0))

        # a m = 2 is the same as scaling s_img by 2
        img2_ = ot.convolve(ot.Image(img.data, [2*s_img[0], 2*s_img[1]]), psf, m=1)
        img2, s2 = img2_.data, img2_.s
        img2_s_ = ot.convolve(img, psf, m=2)
        img2_s, s2_s = img2_s_.data, img2_s_.s
        self.assertTrue(np.all(s2 == s2_s))
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
        img0 = np.random.sample((1000, 1000, 3))
        img0[:, :, 1:] = 0
        img = ot.Image(img0, [1, 1])

        # convolve while not normalizing the output values
        img2 = ot.convolve(img, rimg, cargs=dict(normalize=False))

        # check that image is almost empty (besides numerical errors)
        self.assertAlmostEqual(np.mean(img2.data), 0, delta=1e-06)

    def test_slicing(self):

        # regardless of the other padding option and evenness of input
        # after slicing the resulting image should have the same side lengths and shape
        for padding in ["constant", "edge"]:
            for sp in [1.1e-2, 1e-2]:
                for sh0, sh1 in zip([100, 100, 101, 101], [100, 101, 100, 101]):
                    
                    img0 = np.zeros((201, 201, 3))
                    img0[100:150, 120:140] = 1
                    img = ot.Image(img0, [1, 1])
                    psf = ot.Image(np.ones((sh0, sh1, 3)), [sp, sp])

                    img2 = ot.convolve(img, psf, slice_=True, padding_mode=padding)
                    
                    # check shape and side lengths
                    self.assertTrue(np.all(img2.s == img.s))
                    self.assertTrue(np.all(img2.shape == img.shape))

                    # check center of mass. Shouldn't deviate by more than half a pixel in some cases
                    cm = scipy.ndimage.center_of_mass(img.data)
                    cm2 = scipy.ndimage.center_of_mass(img2.data)
                    self.assertAlmostEqual(cm[0], cm2[0], delta=0.7)
                    self.assertAlmostEqual(cm[1], cm2[1], delta=0.7)

    def test_padding(self):

        for slice_ in [True, False]:

            img = ot.Image(np.ones((100, 100, 3)), [1, 1])
            psf = ot.presets.psf.circle(60)

            # padding with black leaves a decreasing edge
            img2 = ot.convolve(img, psf, slice_=slice_, padding_mode="constant", padding_value=[0, 0, 0])
            val = (np.mean(img2.data[0, :]) + np.mean(img2.data[-1, :])\
                    + np.mean(img2.data[:, 0]) + np.mean(img2.data[:, -1]))/4
            self.assertNotAlmostEqual(val, 1, delta=0.00001)
            
            # padding with white keeps white edge
            img3 = ot.convolve(img, psf, slice_=slice_, padding_mode="constant", padding_value=[1, 1, 1])
            val = (np.mean(img3.data[0, :]) + np.mean(img3.data[-1, :])\
                    + np.mean(img3.data[:, 0]) + np.mean(img3.data[:, -1]))/4
            self.assertAlmostEqual(val, 1, delta=0.00001)
            
            # padding with "edge" fills color correctly
            # check with different channel values, so we now each channel gets edge padded correctly
            img4 = ot.Image(np.full((100, 100, 3), 0.3), [1, 1])
            img4._data[:, :, 0] = 0
            img5 = ot.convolve(img4, psf, slice_=slice_, padding_mode="edge", cargs=dict(normalize=False))
            val0 = (np.mean(img5.data[0, :, 0]) + np.mean(img5.data[-1, :, 0])\
                    + np.mean(img5.data[:, 0, 0]) + np.mean(img5.data[:, -1, 0]))/4
            self.assertAlmostEqual(val0, 0, delta=0.00001)
            val = (np.mean(img5.data[0, :, 1:]) + np.mean(img5.data[-1, :, 1:])\
                    + np.mean(img5.data[:, 0, 1:]) + np.mean(img5.data[:, -1, 1:]))/4
            self.assertAlmostEqual(val, 0.3, delta=0.00001)
            
    def test_color_steadiness(self):
       
        # test convolving without normalization of a single colored image leads to the same colored images
        # this means that the psf is correctly normalized

        for rgb in [[0, 1, 0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.1]]:

            img4 = ot.Image(np.full((100, 100, 3), 0.3), [1, 1])
            img4._data[:, :, 0] = rgb[0]
            img4._data[:, :, 1] = rgb[1]
            img4._data[:, :, 2] = rgb[2]

            psf = ot.presets.psf.circle(60)
            
            img5 = ot.convolve(img4, psf, slice_=True, padding_mode="edge", cargs=dict(normalize=False))

            for i in range(3):
                val = (np.mean(img5.data[0, :, i]) + np.mean(img5.data[-1, :, i])\
                        + np.mean(img5.data[:, 0, i]) + np.mean(img5.data[:, -1, i]))/4
                self.assertAlmostEqual(val, rgb[i], delta=0.00001)

if __name__ == '__main__':
    unittest.main()
