#!/bin/env python3

import sys
sys.path.append('.')

from pathlib import Path
import os
import unittest
import numpy as np
import pytest
import cv2

import optrace.tracer.color as color

import optrace as ot


class RImageTests(unittest.TestCase):

    def test_r_image_misc(self):

        # init exceptions
        self.assertRaises(TypeError, ot.RImage, 5)  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 7])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, np.inf])  # invalid extent
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], 5)  # invalid projection type
        self.assertRaises(TypeError, ot.RImage([5, 6, 7, 8]).render, limit=[5])  # invalid limit type
        self.assertRaises(ValueError, ot.RImage([5, 6, 7, 8]).render, limit=-1)  # invalid limit value

        img = ot.RImage([-1, 1, -2, 2])
        self.assertFalse(img.has_image())  # no image yet
        
        # create some image
        img, P, L = self.gen_r_image()
       
        # now has an image
        self.assertTrue(img.has_image())
        
        # check display_modes
        for dm in ot.RImage.display_modes:
            img_ = img.get(dm)
            
            # all values should be positive
            self.assertTrue(np.min(img_) > -1e-9)
            
            # in srgb modes the values should be <= 1
            if dm.find("sRGB") != -1:
                self.assertTrue(np.max(img_) < 1+1e-9)

        self.assertRaises(ValueError, img.get, "hjkhjkhjk")  # invalid mode

        # rgb with log and rendering intent options
        for ri in color.SRGB_RENDERING_INTENTS:
            for log in [True, False]:
                im = img.rgb(log=log, rendering_intent=ri)
                self.assertTrue(np.min(im) > 0-1e-9)
                self.assertTrue(np.max(im) < 1+1e-9)

        # check if empty image throws
        RIm = ot.RImage(extent=[-1, 1, -1, 1])
        for mode in ot.RImage.display_modes:
            self.assertRaises(RuntimeError, RIm.get, mode)
    
    @pytest.mark.slow
    def test_r_image_cut(self):
       
        for ratio in [1/6, 0.9, 3, 5]:  # different side ratios
            for Npx in ot.RImage.SIZES:  # different pixel numbers

                img, P, L = self.gen_r_image(ratio=ratio, N_px=Npx)
            
                for p in [-0.1, 0, 0.163485, 1, 1.01]: # different relative position inside image
                    for cut_ in ["x", "y"]:
                        ext = img.extent[:2] if cut_ == "x" else img.extent[2:]
                        ext2 = img.extent[2:] if cut_ == "x" else img.extent[:2]
                        kwargs = {cut_: ext[0] + p*(ext[1] - ext[0])}

                        for dm in ["sRGB (Absolute RI)", "Irradiance"]:  # modes with three and one channel

                            if p < 0 or p > 1:
                                self.assertRaises(ValueError, img.cut, dm, **kwargs)
                            else:
                                rgb_mode = dm.find("sRGB (") != -1
                                bine, vals = img.cut(dm, **kwargs)

                                # check properties
                                self.assertEqual(len(vals), 3 if rgb_mode else 1)  # channel count
                                self.assertEqual(len(bine), len(vals[0])+1)  # lens of bin edges and values
                                self.assertAlmostEqual(bine[0], ext2[0])  # first bin starts at extent
                                self.assertAlmostEqual(bine[-1], ext2[-1])  # last bin ends at extent

                                # stack rgb channels in depth dimension
                                if rgb_mode:
                                    vals = np.dstack((vals[0], vals[1], vals[2]))

                                # make cut through image, multiply with 1-1e-12 so p = 1 gets mapped into last bin
                                if cut_ == "x":
                                    img_ = img.get(dm)[:, int(p * (1 - 1e-12) * img.shape[1])]
                                else:
                                    img_ = img.get(dm)[int(p * (1 - 1e-12) * img.shape[0]), :]

                                # compare with cut from before
                                self.assertTrue(np.allclose(img_ - vals, 0))


        # coverage: x and y not provided
        self.assertRaises(ValueError, img.cut, "Irradiance")

        # coverage: additional kwargs
        img.cut("sRGB (Perceptual RI)", x=0.5, L_th=0.01, log=True)

    @pytest.mark.slow
    def test_r_image_render_and_rescaling(self):

        for limit in [None, 20]:  # different resolution limits

            for i, ratio in enumerate([1/6, 0.38, 1, 5]):  # different side ratios

                ot.global_options.multithreading = bool(i % 2)  # toggle threading
                img, P, L = self.gen_r_image(ratio=ratio, limit=limit)

                # check power sum
                P0 = img.power()
                L0 = img.luminous_power()
                self.assertAlmostEqual(P0/P, 1, places=6)
                self.assertAlmostEqual(L0/L, 1, places=6)

                ratio_act = img.shape[1] / img.shape[0]
                sx0, sy0 = img.s

                for i, Npx in enumerate([*ot.RImage.SIZES, *np.random.randint(1, 2*ot.RImage.SIZES[-1], 3)]):  # different side lengths
                    img.rescale(Npx)
                    self.assertEqual(img.N, min(*img.shape))

                    # check that nearest matching side length was chosen
                    siz = ot.RImage.SIZES
                    near_siz = siz[np.argmin(np.abs(Npx - np.array(siz)))]
                    cmp_siz1 = img.shape[1] if ratio_act < 1 else img.shape[0]
                    self.assertEqual(near_siz, cmp_siz1)

                    self.assertAlmostEqual(P0/img.power(), 1, places=6)  # overall power stays the same even after rescaling
                    self.assertAlmostEqual(ratio_act, img.shape[1]/img.shape[0])  # ratio stayed the same
                    self.assertEqual(sx0, img.s[0])  # side length stayed the same
                    self.assertEqual(sy0, img.s[1])  # side length stayed the same
                    self.assertAlmostEqual(img.s[0]*img.s[1]/img.shape[0]/img.shape[1], img.Apx)  # check pixel area

        ot.global_options.multithreading = True

        # test exceptions render
        self.assertRaises(ValueError, img.render, N=ot.RImage.MAX_IMAGE_SIDE*1.2)  # pixel number too large
        self.assertRaises(ValueError, img.render, N=-2)  # pixel number negative
        self.assertRaises(ValueError, img.render, N=0)  # pixel number zero

        # test exceptions rescale
        self.assertRaises(ValueError, img.rescale, 0.5)  # N < 1
        self.assertRaises(ValueError, img.rescale, -1)  # N negative

        # image with too small extent
        zero_ext = [0, ot.RImage.EPS/2, 0, ot.RImage.EPS/2]
        RIm = ot.RImage(extent=zero_ext)
        Im_zero_ext = RIm.extent.copy()
        RIm.render()
        self.assertFalse(np.all(RIm.extent == Im_zero_ext))
        
        # image with high aspect ratio in y 
        hasp_ext = [0, 1, 0, 1.2*ot.RImage.MAX_IMAGE_RATIO]
        RIm = ot.RImage(extent=hasp_ext)
        Im_hasp_ext = RIm.extent.copy()
        RIm.render()
        self.assertFalse(np.all(RIm.extent == Im_hasp_ext))
        
        # image with high aspect ratio in x
        hasp_ext = [0, 1.2*ot.RImage.MAX_IMAGE_RATIO, 0, 1]
        RIm = ot.RImage(extent=hasp_ext)
        Im_hasp_ext = RIm.extent.copy()
        RIm.render()
        self.assertFalse(np.all(RIm.extent == Im_hasp_ext))

        # coverage: _dont_rescale = True
        ext = [0, 1, 0, 1]
        RIm = ot.RImage(extent=ext)
        RIm.render(_dont_rescale=True)

    def gen_r_image(self, N=10000, N_px=ot.RImage.SIZES[6], ratio=1, limit=None):
        # create some image
        img = ot.RImage([-1, 1, -1*ratio, 1*ratio])
        p = np.zeros((N, 2))
        p[:, 0] = np.random.uniform(img.extent[0], img.extent[1], N)
        p[:, 1] = np.random.uniform(img.extent[2], img.extent[3], N)
        w = np.random.uniform(1e-9, 1, N)
        wl = np.random.uniform(*ot.global_options.wavelength_range, N)
        img.render(N_px, p, w, wl, limit=limit)
        return img, np.sum(w), np.sum(w * color.y_observer(wl) * 683)  # 683 lm/W conversion factor

    @pytest.mark.os
    def test_r_image_saving_loading(self):
       
        # create some image
        img = self.gen_r_image()[0]

        # saving and loading for valid path
        path = "test_img.npz"
        img.save(path)
        img2 = ot.RImage.load(path)
        os.remove(path)

        # test if saved and loaded images are equal
        self.assertTrue(img2.limit is None)  # None limit is None after loading
        self.assertTrue(np.all(img2._img == img._img))
        self.assertTrue(img2.N == img.N)
        self.assertTrue(img2.s[0] == img.s[0])
        self.assertTrue(img2.s[1] == img.s[1])
        self.assertTrue(img2.projection is None)

        # coverage: don't provide file type ending
        path2 = "test_img"
        img.save(path2)
        os.remove(path)

        # throw IOError if invalid file path
        self.assertRaises(IOError, img.save, "./hsajkfhajkfhjk/hajfhsajkfhajksfhjk/ashjsafhkj")

    @pytest.mark.os
    def test_r_image_export(self):

        path = "export_image.png"
        path2 = "export_image.jpg"

        # check display modes and channels
        for ratio in [0.746, 1, 2.45]:
            img = self.gen_r_image(ratio=ratio)[0]
            img.rescale(90)
                
            for imm in ot.RImage.display_modes:
                img.export(path, imm, log=True, flip=False)
                img.export(path, imm, log=False, flip=True)
                im3 = ot.Image(path, [1, 1]).data  # load from disk

        # check shapes
        for ratio in [0.746, 1, 2.45]:
            img = self.gen_r_image(ratio=ratio)[0]

            for Npx in [1, 9, 90, 1000]:

                img.rescale(Npx)
                img.export(path, imm, log=False, flip=True)
                im3 = ot.Image(path, [1, 1]).data  # load from disk
                self.assertAlmostEqual(ratio, im3.shape[0]/im3.shape[1], delta=2/Npx)  # check ratio
        
        # check cv2 imwrite parameters and export as jpg
        img.export(path2, "Lightness (CIELUV)", [cv2.IMWRITE_JPEG_QUALITY, 3])
        
        # check handling of incorrect path
        self.assertRaises(IOError, img.export, "hjkhjkhjkhjk.pg", "Irradiance")  # invalid format
        self.assertRaises(IOError, img.export, str(Path("hjkhjk") / Path("hjkhjkhjkhjk.png")), "Irradiance")  # invalid path

        os.remove(path)
        os.remove(path2)

    def test_r_image_filter(self):
    
        # render a point
        # airy resolution limit filter makes an airy disc from this
        # check if standard deviation is approximately that of a gaussian with comparable core

        # rays at center of image
        p = np.zeros((1000, 3))
        w = np.ones(1000)
        wl = np.full(1000, 500)

        for k in [1, 0.3, 5]:  # different side ratios
        
            r0 = 1e-4  # smaller than resolution limit
            img = ot.RImage([-r0, r0, -k/2*r0, k/2*r0])

            for limit in [0.3, 20]:  # different limits

                img.render(900, p, w, wl, limit=limit)
                irr = img.img[:, :, 3]  # irradiance image

                # radial distance in image
                ny, nx = irr.shape[:2]
                x = np.linspace(0, img.extent[1], nx//2+1)

                # check centroid
                # for factor 1.10861 see https://www.wolframalpha.com/input?i=%28integrate+x*%282*J1%28x%29%2Fx%29%5E2+from+0+to+10.1735%29+%2F+%28integrate+%282*J1%28x%29%2Fx%29%5E2+from+0+to+10.1735%29%29
                irr = irr[irr.shape[0]//2+1, irr.shape[1]//2:]
                centroid = 3.8317*np.sum(x * irr)/np.sum(irr)

                # compare to limit
                self.assertAlmostEqual(centroid/1.10861, limit/1000, delta=0.0002)
   
        # coverage: RuntimeError when image is a projection and filter is applied
        img = ot.RImage([-1, 1, -1, 1], projection="abc")
        self.assertRaises(RuntimeError, img.render, limit=1)

    @pytest.mark.os
    def test_image_presets(self) -> None:

        for imgi in ot.presets.image.all_presets:
            RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6])
            RSS = imgi([6, 6])
            RS = ot.RaySource(RSS, pos=[0, 0, 0])
            RT.add(RS)
            RT.trace(500000)
            img = RT.source_image(256)

    def test_rimage_to_image(self) -> None:

        rimg, P, L = self.gen_r_image()

        for mode in ot.RImage.display_modes:
            for size in [1, 9, 25, 1000]:
                rimg.rescale(size)

                rimg_mode = rimg.get(mode)
                if (max_ := np.max(rimg_mode)):
                    rimg_mode /= max_

                if rimg_mode.ndim == 2:
                    rimg_mode = np.repeat(rimg_mode[:, :, np.newaxis], 3, axis=2)

                img = rimg.to_image(mode)

                self.assertTrue(np.allclose(rimg_mode - img.data, 0))
                self.assertTrue(np.all(img.s == rimg.s))
                
        # coverage: black image

        rimg.img *= 0 
        rimg_mode = rimg.get("Irradiance")

        if (max_ := np.max(rimg_mode)):
            rimg_mode /= max_

        if rimg_mode.ndim == 2:
            rimg_mode = np.repeat(rimg_mode[:, :, np.newaxis], 3, axis=2)

        img = rimg.to_image(mode)
        self.assertTrue(np.allclose(rimg_mode - img.data, 0))
        self.assertTrue(np.all(img.s == rimg.s))

    @pytest.mark.os
    def test_image_inits(self):

        # type errors
        self.assertRaises(TypeError, ot.Image, 1, [2, 2])  # invalid image type
        self.assertRaises(TypeError, ot.Image, np.ones((100, 100, 3)), 2)  # invalid s type

        # value errors
        self.assertRaises(ValueError, ot.Image, np.ones((100, 100, 3)), [2, 2, 2])  # invalid s length
        self.assertRaises(ValueError, ot.Image, np.ones((100, 100)), [2, 2])  # invalid data shape
        self.assertRaises(ValueError, ot.Image, np.ones((100, 100, 3))*1.001, [2, 2])  # data above 1
        self.assertRaises(ValueError, ot.Image, -np.ones((100, 100, 3))*0.001, [2, 2])  # data below 0
        self.assertRaises(ValueError, ot.Image, np.full((100, 100, 3), np.nan), [2, 2])  # data not finite
        self.assertRaises(ValueError, ot.Image, np.full((100, 100, 3), np.inf), [2, 2])  # data not finite

        # IOError
        self.assertRaises(IOError, ot.Image, "hjkhjkhjkhjk.png", [2, 2])  # invalid path
        assert os.path.exists("setup.py")
        self.assertRaises(IOError, ot.Image, "setup.py", [2, 2])  # invalid format
    
    @pytest.mark.os
    def test_image_saving_loading(self):

        for path in ["test.png", "test.jpg", "test.bmp"]:
            for i, img in enumerate([ot.presets.image.cell([1, 1]), 
                                     ot.Image(np.random.sample((30, 30, 3)), [2, 3]),
                                     ot.Image(np.random.sample((1, 1, 3)), [5, 1])]):

                img.save(path)
                img2 = ot.Image(path, [1, 1])  # check if loads correctly
                os.remove(path)
                
                if not i:  # only check for first, square image. 
                    # Otherwise we can't compare because interpolation took place
                    self.assertEqual(img.data[0, 0, 0], img2.data[0, 0, 0]) # check if nothing has been flipped

        # coverage: cv2.imwrite params and flip option
        img.save("test.png", [cv2.IMWRITE_PNG_COMPRESSION, 1], flip=True)
        os.remove("test.png")

        # IOError, invalid type
        self.assertRaises(IOError, img.save, "hjkhjkhjkhjk.ggg")  # invalid format
        self.assertRaises(IOError, img.save, str(Path("hjkhjk") / Path("hjkhjkhjkhjk.png")))  # invalid path

    def test_0image_orientation(self):
        """check that pixel and image positions are consistent for image, rendered image and ray starting positions"""

        # create image with single bright pixel of all
        im_data = np.zeros((5, 5, 3))
        im_data[0, 0] = 1
        img = ot.Image(im_data, [1, 1])

        RT = ot.Raytracer([-2, 2, -2, 2, -10, 10])

        RS = ot.RaySource(img, pos=[0, 0, 0], s=[0, 0, 1], divergence="None")
        RT.add(RS)

        det = ot.Detector(ot.RectangularSurface([1, 1, ]), pos=[0, 0, 1])
        RT.add(det)

        RT.trace(10000)

        # render source and detector images. image pixels should map 1:1
        simg = RT.source_image(5).get("sRGB (Absolute RI)")
        dimg = RT.detector_image(5, extent=img.extent).get("sRGB (Absolute RI)")

        # white element stayed in the same place for source and detector image
        self.assertAlmostEqual(np.mean(simg[0, 0]), 1, delta=0.001)
        self.assertAlmostEqual(np.mean(dimg[0, 0]), 1, delta=0.001)

        # recreating the image from rendered data still produces the element at the same position
        self.assertAlmostEqual(np.mean(ot.Image(dimg, [1, 1]).data[0, 0]), 1, delta=0.001)

        # saving and loading produces the same pixel value
        img.save("test.png")
        img2 = ot.Image("test.png", [1, 1])
        os.remove("test.png")
        self.assertAlmostEqual(np.mean(img2.data[0, 0]), 1, delta=0.01)

        # check that rays start at negative x, y, corner
        self.assertAlmostEqual(np.mean(RT.rays.p_list[:, 0, 0]), -0.5 + 1/5/2, delta=0.0001)
        self.assertAlmostEqual(np.mean(RT.rays.p_list[:, 0, 1]), -0.5 + 1/5/2, delta=0.0001)

if __name__ == '__main__':
    unittest.main()
