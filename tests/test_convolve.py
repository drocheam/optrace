#!/bin/env python3

import unittest
import numpy as np
import pytest
import scipy.interpolate
import scipy.ndimage
import cv2

import optrace as ot
from optrace.tracer import color
import optrace.tracer.misc as misc



class ConvolutionTests(unittest.TestCase):

    def test_exceptions(self):
        """test value and type errors for the convolve() function"""

        img = ot.presets.image.color_checker([5, 6])
        img2 = ot.presets.psf.airy(1000)
        psf = ot.presets.psf.halo()
        psf.extent = [-0.05, 0.05, -0.15, 0.15]

        # check invalid types
        self.assertRaises(TypeError, ot.convolve, 5, psf)  # invalid img
        self.assertRaises(TypeError, ot.convolve, img, 5)  # invalid psf
        self.assertRaises(TypeError, ot.convolve, img, psf, m=[])  # invalid magnification
        self.assertRaises(TypeError, ot.convolve, img, psf, keep_size=[])  # invalid keep_size
        self.assertRaises(TypeError, ot.convolve, img, psf, cargs=[])  # invalid cargs
        self.assertRaises(TypeError, ot.convolve, img, psf, padding_mode=[])  # invalid padding_mode
        self.assertRaises(TypeError, ot.convolve, img, psf, padding_value=2)  # invalid padding_value for RGBImage
        self.assertRaises(TypeError, ot.convolve, img2, psf, padding_value=[1, 2])  
        # ^-- invalid padding_value for GrayscaleImage
        
        # value errors
        self.assertRaises(ValueError, ot.convolve, img, psf, m=0)  # m can't be zero
        self.assertRaises(ValueError, ot.convolve, img, psf, padding_value=[0, 0])  # invalid padding_value shape
        self.assertRaises(ValueError, ot.convolve, img, psf, padding_value=[[0, 0, 0]])  # invalid padding_value shape
        self.assertRaises(ValueError, ot.convolve, img, psf, padding_value=[0, 0, -1])  
        # ^-- invalid padding_value for RGBImage
        self.assertRaises(ValueError, ot.convolve, img2, psf, padding_value=-2)  
        # ^-- invalid padding_value for GrayscaleImage

        # test exceptions for colored image and colored psf
        rimg = ot.RenderImage([-1, 1, -1, 1])
        rimg.render()
        rimg2 = ot.RenderImage([-1, 1, 0, 1])
        rimg2.render()
        self.assertRaises(TypeError, ot.convolve, img, rimg)  # color img and psf require R, G, B psf list
        self.assertRaises(TypeError, ot.convolve, img, [rimg, rimg])  # see above, but must be explicitly 3 elements
        self.assertRaises(TypeError, ot.convolve, img, [rimg, img2, rimg])  # one element has the wrong type
        self.assertRaises(ValueError, ot.convolve, img, [rimg, rimg2, rimg])  # extents are not the same
        self.assertRaises(TypeError, ot.convolve, img2, [rimg, rimg, rimg]) 
        # ^-- a single RenderImage PSF sufficient for grayscale image
            
    def test_resolution_exceptions(self):

        img = ot.presets.image.color_checker([5, 6])
        psf = ot.presets.psf.halo()
        psf.extent = [-0.05, 0.05, -0.15, 0.15]
        s_psf = psf.s
        s_img = img.s

        # resolution tests
        ###########################################

        # low resolution psf
        psf2 = ot.GrayscaleImage(np.ones((5, 5)), s_psf)
        self.assertRaises(ValueError, ot.convolve, img, psf2)
        
        # low resolution image
        img2 = ot.RGBImage(np.ones((5, 5, 3)), s_img)
        self.assertRaises(ValueError, ot.convolve, img2, psf)
        
        # huge resolution image
        img2 = ot.RGBImage(np.ones((3000, 3000, 3)), s_img)
        self.assertRaises(ValueError, ot.convolve, img2, psf)
        
        # huge resolution psf
        psf2 = ot.GrayscaleImage(np.ones((3000, 3000)), s_psf)
        self.assertRaises(ValueError, ot.convolve, img, psf2)
        
        # psf much larger than image
        psf3 = ot.GrayscaleImage(psf2.data, [s_img[0]*3, s_img[1]*3])
        self.assertRaises(ValueError, ot.convolve, img, psf3)
       
    def test_coverage(self):
        """Some additional coverage testing"""

        # check different threading and progressbar constellations
        # needs to test simple and RGB psf

        img = ot.presets.image.color_checker([5, 6])
        psf = ot.presets.psf.halo()
        
        rimg = ot.RenderImage([-2.3, 1, -1, 1])
        rimg.render()
        rimg._data = np.random.sample(rimg.shape)
        psf2 = [rimg, rimg, rimg]

        psf.extent = [-0.05, 0.05, -0.15, 0.15]
        s_img = img.s
        s_psf = psf.s
        
        # check that call is correct, also tests threading and progressbar
        for bar in [False, True]:
            for threading in [False, True]:
                for psfi in [psf, psf2]:
                    ot.global_options.show_progress_bar = bar
                    ot.global_options.multithreading = threading
                    ot.convolve(img, psfi)

        # additional coverage tests
        #############################################
        img = ot.presets.image.color_checker([1, 1]).data

        # pixels are strongly non-square
        ot.convolve(ot.RGBImage(img, s_img), ot.GrayscaleImage(psf.data, [1e-9, 1e-8]))
        ot.convolve(ot.RGBImage(img, [1, 10]), ot.GrayscaleImage(psf.data, s_psf))
        
        # low resolution psf warning
        psf2 = np.ones((90, 90))
        ot.convolve(ot.RGBImage(img, s_img), ot.GrayscaleImage(psf2, s_psf))
        
        # low resolution image warning
        img2 = np.ones((90, 90, 3))
        ot.convolve(ot.RGBImage(img2, s_img), psf)
        
    def test_point_psf(self):
        """test if convolution with a point produces the same image"""

        s_img = [1, 1]
        image = ot.presets.image.cell(s_img)
        s_psf = [1e-9, 1e-9]

        # image with point

        # odd and even pixel size
        for padding in ["constant", "edge"]:
            for slice_ in [False, True]:
                for psf_s in [(200, 200), (201, 201)]:
                    psf = np.ones(psf_s)

                    img2_ = ot.convolve(image, ot.GrayscaleImage(psf, s_psf), keep_size=slice_, padding_mode=padding)
                    img2, s2 = img2_.data, img2_.s

                    # compare
                    dsx = (img2.shape[1] - image.shape[1]) // 2
                    dsy = (img2.shape[0] - image.shape[0]) // 2
                    img2p = img2[dsy:-dsy, dsx:-dsx] if not slice_ else img2
                    mdiff = np.max(img2p - image.data)
                    self.assertTrue(mdiff**2.2 < 2e-5)  # should stay the same, but small numerical errors

    def test_point_image_point_psf(self):
        """
        Point convolved with a point is also a point. 
        We also use a special case, where only one pixel of psf and image is set.
        """
        psf = np.zeros((301, 301))
        psf[151, 151] = 1
        img = np.repeat(psf[:, :, np.newaxis], 3, axis=2)
        s_img = [1, 1]
        s_psf = [1e-9, 1e-9]

        for padding in ["constant", "edge"]:  
        # with default settings ("constant" and padding_value=0) no 
        # padding takes place, as scipy.fftconvolve automatically pads with black
            for slice_ in [False, True]:
                img2_ = ot.convolve(ot.RGBImage(img, s_img), ot.GrayscaleImage(psf, s_psf), 
                                    keep_size=slice_, padding_mode=padding)
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
            psf = color.srgb_linear_to_srgb(psf)[:, :, 0]
            img, s_img = _gaussian(2, sz)

            # make sRGB image
            img = color.srgb_linear_to_srgb(img)

            # convolution
            img2_ = ot.convolve(ot.RGBImage(img, s_img), ot.GrayscaleImage(psf, s_psf))
            img2, s2 = img2_.data, img2_.s

            # convert back to intensities
            img2 = color.srgb_to_srgb_linear(img2)
            img2 = img2[:, :, 0]

            # resulting gaussian with convolution
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

            # NOTE there seems to be some remaining error
            # similar to an integration error that becomes smaller and smaller for a larger resolution


    @pytest.mark.conv
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
        # i == 4: colored psf and linear image
        for i in range(5):
            with self.subTest(i=i):

                # raytracer and raysource
                RT = ot.Raytracer(outline=[-5, 5, -5, 5, 0, 100], no_pol=True)
                RS_args = dict(surface=ot.Point(), divergence="Isotropic", div_angle=2, s=[0, 0, 1], pos=[0, 0, 0]) 

                if i in [1, 2, 3]:
                    for j, spec in enumerate([ot.presets.light_spectrum.srgb_r, ot.presets.light_spectrum.srgb_g, 
                                              ot.presets.light_spectrum.srgb_b]):
                        RS = ot.RaySource(**RS_args, spectrum=spec, power=color.SRGB_PRIMARY_POWER_FACTORS[j])
                        RT.add(RS)
                else:
                    RS = ot.RaySource(**RS_args)
                    RT.add(RS)

                # add Lens
                front = ot.SphericalSurface(r=3, R=8)
                back = ot.SphericalSurface(r=3, R=-8)
                nL1 = ot.RefractionIndex("Abbe", n=1.5, V=80) if i in [0, 3, 4]\
                        else ot.RefractionIndex("Constant", n=1.5)
                L1 = ot.Lens(front, back, de=0.1, pos=[0.2, -0.3, 12], n=nL1)  # some asymmetry
                RT.add(L1)

                # more asymmetry
                RT.add(ot.Aperture(ot.RingSurface(r=0.3, ri=0.2), pos=[0.1, 0, L1.back.z_max]))

                # add Detector
                DetS = ot.RectangularSurface(dim=[10, 10])
                Det = ot.Detector(DetS, pos=[0, 0, 36.95])
                Det.move_to([0, 0, RT.tma().image_position(0)])
                RT.add(Det)

                # render PSF
                RT.trace(1000000)
                psf = RT.detector_image()
                s_img = np.array(psf.s)

                if i in [1, 2]:
                    psf = ot.GrayscaleImage(np.mean(psf.get("sRGB (Absolute RI)").data, axis=2), extent=psf.extent)

                if i == 3:
                    psf_r = RT.detector_image(source_index=0, extent=psf.extent)
                    psf_g = RT.detector_image(source_index=1, extent=psf.extent)
                    psf_b = RT.detector_image(source_index=2, extent=psf.extent)
                    psf = [psf_r, psf_g, psf_b]
                
                # image
                if i in [0, 2]:
                    img = ot.presets.image.ETDRS_chart_inverted 
                elif i == 4:
                    img = lambda x: ot.presets.psf.airy(x[0]*1000/5/2)
                else: 
                    img = ot.presets.image.tv_testcard2

                # swap old source for image source
                # make ray cone converge to lens center, so we are still somewhat in the paraxial case
                RT.remove(RT.ray_sources)
                RS = ot.RaySource(img(s_img), divergence="Isotropic", div_angle=2, pos=[0, 0, 0], 
                                  orientation="Converging", conv_pos=[0, 0, 12])
                RT.add(RS)

                # convolution image
                mag = RT.tma().image_magnification(0)
                img_conv_ = ot.convolve(img(s_img), psf, m=mag)
                img_conv, s_img_conv = img_conv_.data, img_conv_.s
                if isinstance(img_conv_, ot.GrayscaleImage):
                    img_conv = np.repeat(img_conv[:, :, np.newaxis], 3, axis=2)

                # rendered image
                det_im = RT.iterative_render(30e6, extent=img_conv_.extent)
                img_ren0 = det_im[0]
                img_ren = img_ren0.get("sRGB (Absolute RI)", 189).data

                # interpolate on convolved image and subtract from rendered one channel-wise
                img_ren = color.srgb_to_srgb_linear(img_ren)
                img_conv = color.srgb_to_srgb_linear(img_conv)
                img_conv = cv2.resize(img_conv, [img_ren.shape[1], img_ren.shape[0]], interpolation=cv2.INTER_AREA)
                img_conv /= np.mean(img_conv)
                img_ren /= np.mean(img_ren)
                im_diff = img_ren - img_conv
                im_diff -= np.mean(im_diff)
                res = np.mean(np.abs(im_diff/np.max(img_conv)))  # mean absolute deviation
                mean_ratio = np.mean(img_ren, axis=(0, 1)) / np.mean(img_conv, axis=(0, 1)) # ratio of mean rgb values

                # import optrace.plots as otp
                # text = f"{res:.2g} f{np.max(np.abs(mean_ratio-1)):.2g}"
                # otp.image_plot(ot.RGBImage(img_ren/np.max(img_ren), [1, 1]), title="rendered")
                # otp.image_plot(ot.RGBImage(img_conv/np.max(img_conv), [1, 1]), title="conv")
                # otp.image_plot(ot.RGBImage(np.abs(img_conv/np.max(img_conv)-img_ren/np.max(img_ren)), [1, 1]), 
                               # title=text)
                # otp.block()

                # different error tolerances for each case, derived/estimated empirically
                err_mean = [0.0018, 0.013, 0.007, 0.0085, 0.0004]
                err_color = [0.008, 0.0015, 0.0015, 0.0015, 0.005]

                # check that mean difference is small
                # we still have some deviations due to noise, incorrect magnification and not-linear aberrations
                self.assertAlmostEqual(res, 0, delta=err_mean[i], msg=f"case={i}")
                self.assertAlmostEqual(np.max(np.abs(mean_ratio-1)), 0, delta=err_color[i], msg=f"case={i}")
     
    def test_white_balance(self):
        """
        when different color components have different frequencies, 
        in a naive case higher frequencies of one color would be deleted due to 
        a finite resolution of image compared to the PSF. But this should be handled when convolving
        """
        # make sure psf normalization has no effect
        for normalize in [True, False]:

            # generate color image with high frequency red components
            rimg = ot.RenderImage([-1e-6, 1e-6, -1e-6, 1e-6])
            rimg.render()
            rimg._data[:, :] = 0.5
            rimg._data[:2, :2, 2] = 1
            rimg._data[-2:, -2:, 2] = 0
            assert(np.std(np.mean(rimg._data[:, :, :3], axis=(0, 1))) < 1e-6)  # mean color is white
            rimg._data[:, :, :3] = color.srgb_linear_to_xyz(rimg._data[:, :, :3])  # convert to XYZ

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
            img2_ = ot.convolve(img, [rimg, rimg, rimg], cargs=dict(normalize=normalize), keep_size=True)
            img2, s2 = img2_.data, img2_.s
            img2l = color.srgb_to_srgb_linear(img2)

            # every r, g, b psf produces white -> overall image must also be black and white regardless of mean color
            cerr = np.std(np.mean(img2l, axis=(0, 1)))
            self.assertAlmostEqual(cerr, 0, delta=1e-6)

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
            self.assertAlmostEqual(np.max(pst.data), 1)  # sums to one

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

        img = ot.RGBImage(np.zeros((200, 200, 3)), [5, 5])
        psf = ot.presets.psf.glare()

        img2 = ot.convolve(img, psf)
        self.assertEqual(np.max(img2.data), 0)
    
    def test_zero_psf(self):
        """no warnings/exceptions with zero psf"""
        # zero grayscale_image
        img = ot.presets.image.color_checker([5, 5])
        psf = ot.GrayscaleImage(np.zeros((200, 200)), [1, 1])
        img2 = ot.convolve(img, psf)
        self.assertEqual(np.max(img2.data), 0)
        
        # zero render_image
        img = ot.presets.image.siemens_star([5, 5])
        psf = ot.RenderImage([-1, 1, -1, 1])
        psf.render()
        img2 = ot.convolve(img, psf)
        self.assertEqual(np.max(img2.data), 0)
        
        # zero three render_images
        img = ot.presets.image.color_checker([5, 5])
        psf = ot.RenderImage([-1, 1, -1, 1])
        psf.render()
        img2 = ot.convolve(img, [psf, psf, psf])
        self.assertEqual(np.max(img2.data), 0)

    def test_m_behavior(self):
        """test the behavior of the magnification m regarding sign and value"""

        img = ot.presets.image.color_checker([5, 6])
        psf = ot.presets.psf.halo()
        psf.extent = [-0.05, 0.05, -0.15, 0.15]
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
        img2_ = ot.convolve(ot.RGBImage(img.data, [2*s_img[0], 2*s_img[1]]), psf, m=1)
        img2, s2 = img2_.data, img2_.s
        img2_s_ = ot.convolve(img, psf, m=2)
        img2_s, s2_s = img2_s_.data, img2_s_.s
        self.assertTrue(np.all(s2 == s2_s))
        self.assertTrue(np.allclose(img2 - img2_s, 0))

    def test_channel_orthogonality(self):
        """
        Checks that a red image convolved with RGB PSFs all missing red produces a zero image.
        Checks that there is no "bleeding" from the other channels into red through color conversions etc.
        Also tests that normalize=False does not normalize the resultant small values.
        """
        # G and B render image PSF are equal and miss red component
        # R PSF is zero
        rimg = ot.RenderImage([-1e-6, 1e-6, -1e-6, 1e-6])
        rimg.render()
        rimg._data[:, :] = 1
        rimg._data[:, :, 0] = 0
        rimg._data[:, :, :3] = color.srgb_linear_to_xyz(rimg._data[:, :, :3])  # convert to XYZ
        rimg_g = rimg_b = rimg.copy()
        rimg._data *= 0
        rimg_r = rimg
        
        # create red noise image
        img0 = np.random.sample((1000, 1000, 3))
        img0[:, :, 1:] = 0
        img = ot.RGBImage(img0, [1, 1])

        # convolve while not normalizing the output values
        img2 = ot.convolve(img, [rimg_r, rimg_g, rimg_b], cargs=dict(normalize=False))

        # check that image is almost empty (besides numerical errors)
        self.assertAlmostEqual(np.mean(img2.data), 0, delta=1e-06)

    def test_slicing(self):
        """
        Check that image is correctly sliced regardless of padding mode, shape, 
        and image type (GrayscaleImage, RGBImage).
        """
        img0 = np.zeros((201, 201, 3))
        img0[100:150, 120:140] = 1
        img1 = ot.RGBImage(img0, [1, 1])
        img2 = ot.GrayscaleImage(img0[:, :, 0], [1, 1])

        # regardless of the other padding option and evenness of input
        # after slicing the resulting image should have the same side lengths and shape
        for img in [img1, img2]:
            for padding in ["constant", "edge"]:
                for sp in [1.1e-2, 1e-2]:
                    for sh0, sh1 in zip([100, 100, 101, 101], [100, 101, 100, 101]):
                        
                        psf = ot.GrayscaleImage(np.ones((sh0, sh1)), [sp, sp])

                        img2 = ot.convolve(img, psf, keep_size=True, padding_mode=padding)
                        
                        # check shape and side lengths
                        self.assertTrue(np.all(img2.s == img.s))
                        self.assertTrue(np.all(img2.shape == img.shape))

                        # check center of mass. Shouldn't deviate by more than half a pixel in some cases
                        cm = scipy.ndimage.center_of_mass(img.data)
                        cm2 = scipy.ndimage.center_of_mass(img2.data)
                        self.assertAlmostEqual(cm[0], cm2[0], delta=0.7)
                        self.assertAlmostEqual(cm[1], cm2[1], delta=0.7)

    def test_padding(self):
        """
        Make sure padding works. Test padding_modes and padding_value, both for GrayscaleImage and RGBImage.
        """ 
        img1 = ot.RGBImage(np.ones((100, 100, 3)), [1, 1])
        img2 = ot.GrayscaleImage(np.ones((100, 100)), [1, 1])
        psf = ot.presets.psf.circle(60)

        for img, pval1, pval2 in zip([img1, img2], [[0, 0, 0], 0], [[1, 1, 1], 1]):
            for slice_ in [True, False]:

                # padding with black leaves a decreasing edge
                img2 = ot.convolve(img, psf, keep_size=slice_, padding_mode="constant", padding_value=pval1)
                val = (np.mean(img2.data[0, :]) + np.mean(img2.data[-1, :])\
                        + np.mean(img2.data[:, 0]) + np.mean(img2.data[:, -1]))/4
                self.assertNotAlmostEqual(val, 1, delta=0.00001)
                
                # padding with white keeps white edge
                img3 = ot.convolve(img, psf, keep_size=slice_, padding_mode="constant", padding_value=pval2)
                val = (np.mean(img3.data[0, :]) + np.mean(img3.data[-1, :])\
                        + np.mean(img3.data[:, 0]) + np.mean(img3.data[:, -1]))/4
                self.assertAlmostEqual(val, 1, delta=0.00001)
               
                if isinstance(img, ot.RGBImage):
                    # padding with "edge" fills color correctly
                    # check with different channel values, so we now each channel gets edge padded correctly
                    img4 = ot.RGBImage(np.full((100, 100, 3), 0.3), [1, 1])
                    img4._data[:, :, 0] = 0
                    img5 = ot.convolve(img4, psf, keep_size=slice_, padding_mode="edge", cargs=dict(normalize=False))
                    val0 = (np.mean(img5.data[0, :, 0]) + np.mean(img5.data[-1, :, 0])\
                            + np.mean(img5.data[:, 0, 0]) + np.mean(img5.data[:, -1, 0]))/4
                    self.assertAlmostEqual(val0, 0, delta=0.00001)
                    val = (np.mean(img5.data[0, :, 1:]) + np.mean(img5.data[-1, :, 1:])\
                            + np.mean(img5.data[:, 0, 1:]) + np.mean(img5.data[:, -1, 1:]))/4
                    self.assertAlmostEqual(val, 0.3, delta=0.00001)

                else:
                    img4 = ot.GrayscaleImage(np.full((100, 100), 0.3), [1, 1])
                    img5 = ot.convolve(img4, psf, keep_size=slice_, padding_mode="edge", cargs=dict(normalize=False))
                    val = (np.mean(img5.data[0, :]) + np.mean(img5.data[-1, :])\
                            + np.mean(img5.data[:, 0]) + np.mean(img5.data[:, -1]))/4
                    self.assertAlmostEqual(val, 0.3, delta=0.00001)

    def test_extent_shifting(self):
        """test that image extent/position is handled correctly when PSF or image (or both) are shifted"""

        img = ot.presets.image.color_checker([2, 3])
        psf = ot.presets.psf.airy(50)

        img2 = ot.convolve(img, psf)

        # image2 is centered
        self.assertEqual(img2.extent[1] + img2.extent[0], 0)
        self.assertEqual(img2.extent[2] + img2.extent[3], 0)

        # shift image and psf in opposite directions
        img.extent += [-6, -6, 3, 3]
        psf.extent -= [-6, -6, 3, 3]
        img2 = ot.convolve(img, psf)

        # image2 is still centered
        self.assertEqual(img2.extent[1] + img2.extent[0], 0)
        self.assertEqual(img2.extent[2] + img2.extent[3], 0)

        # shift psf back
        psf.extent += [-6, -6, 3, 3]
        img2 = ot.convolve(img, psf)

        # image2 is at center of img
        self.assertEqual((img2.extent[1] + img2.extent[0])/2, -6)
        self.assertEqual((img2.extent[2] + img2.extent[3])/2, 3)

    def test_image_flip(self):
        """
        check that image flip (magnification parameter m < 0) flips the image correctly.
        
        This is checked with an asymmetric PSF and image.
        In the first step we compare the PSF and an image created by a Point. 
        They should be equal and their center of mass should be in the same region.
        Secondly, image two points. The points in the resulting image should be flipped
        center of mass in third quadrant goes to first quadrant
        """

        RT = ot.Raytracer([-10, 10, -10, 10, -100, 400])

        RS = ot.RaySource(ot.Point(), divergence="Lambertian", div_angle=3, pos=[0, 0, -20])
        RT.add(RS)

        # create a shifted sphere lens with R=5
        n = ot.RefractionIndex("Constant", n=1.3)
        front = ot.SphericalSurface(r=4.99999999, R=5)
        back = ot.SphericalSurface(r=4.99999999, R=-5)
        L = ot.Lens(front, back, d=10, pos=[0, -0.9, 0], n=n)
        RT.add(L)

        # add rectangular detector
        DetS = ot.RectangularSurface([10, 10])
        Det = ot.Detector(DetS, pos=[0, 0, 18.987], desc="Retina")
        RT.add(Det)

        # render psf
        RT.trace(500000)
        dimg = RT.detector_image()
        
        data = np.zeros((1001, 1001))
        data[500, 500] = 1
        img = ot.GrayscaleImage(data, [0.5, 0.5])
        img2 = ot.convolve(img, dimg, m=-1, keep_size=True)

        # test that center of mass is above center
        cm = np.array(scipy.ndimage.center_of_mass(img2.data[:, :, 0])) / img2.shape[0]
        self.assertTrue(cm[0] > 0.53)

        # set second bright pixel in lower left of the image
        data[0, 0] = 1
        img = ot.GrayscaleImage(data, [0.5, 0.5])
        img2 = ot.convolve(img, dimg, m=-1)
        cm = np.array(scipy.ndimage.center_of_mass(img2.data[:, :, 0])) / img2.shape[0]

        # test that center of mass is in first quadrant
        self.assertTrue(cm[0] > 0.7)
        self.assertTrue(cm[1] > 0.7)

    @pytest.mark.norm
    def test_unnormalized_color_and_grayscale(self):
        """
        test convolving without normalization of a single colored image leads to the same colored images
        this means that the psf is correctly normalized
        """
        # color image and greyscale PSF
        for rgb in [[0, 1, 0], [0.2, 0.3, 0.5], [0.1, 0.1, 0.1]]:
            data = np.tile(rgb, (100, 100, 1))
            img4 = ot.RGBImage(data, [1, 1])

            psf = ot.presets.psf.circle(60)
            img5 = ot.convolve(img4, psf, keep_size=True, padding_mode="edge", cargs=dict(normalize=False))

            for i in range(3):
                val = np.mean(img5.data[:, :, i])
                self.assertAlmostEqual(val, img4.data[0, 0, i], delta=0.00001)
            
            img4_lin = color.srgb_to_srgb_linear(img4.data)
            img5_lin = color.srgb_to_srgb_linear(img5.data)
            self.assertAlmostEqual(np.sum(img4_lin), np.sum(img5_lin), delta=0.01, msg=str(rgb))
       
        # greyscale image and greyscale PSF
        for gv in [0.3, 0., 1.0, 0.297]:

            img4 = ot.GrayscaleImage(np.full((100, 100), gv), [1, 1])
            psf = ot.presets.psf.circle(60)
            
            img5 = ot.convolve(img4, psf, keep_size=True, padding_mode="edge", cargs=dict(normalize=False))
            self.assertAlmostEqual(np.mean(img5.data), gv, delta=0.00001)

if __name__ == '__main__':
    unittest.main()
