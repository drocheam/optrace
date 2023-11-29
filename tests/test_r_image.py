#!/bin/env python3

import sys
sys.path.append('.')

from pathlib import Path
import os
import unittest
import numpy as np
import pytest

from PIL import Image as PILImage  # image loading
import matplotlib.pyplot as plt
import optrace.tracer.color as color

import optrace as ot
import optrace.plots as otp


class RImageTests(unittest.TestCase):

    @pytest.mark.slow
    def test_r_image_misc(self):

        # init exceptions
        self.assertRaises(TypeError, ot.RImage, 5)  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 7])  # invalid extent
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, np.inf])  # invalid extent
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], 5)  # invalid projection type
        self.assertRaises(TypeError, ot.RImage, [5, 6, 8, 9], limit=[5])  # invalid limit type
        self.assertRaises(ValueError, ot.RImage, [5, 6, 8, 9], limit=-1)  # invalid limit value

        img = ot.RImage([-1, 1, -2, 2], silent=True)
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

        # test refilter
        img.limit = 5
        img.refilter()

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
                                    img_ = img.get(dm)[:, int(p * (1 - 1e-12) * img.Nx)]
                                else:
                                    img_ = img.get(dm)[int(p * (1 - 1e-12) * img.Ny), :]

                                # compare with cut from before
                                self.assertTrue(np.allclose(img_ - vals, 0))

    @pytest.mark.slow
    def test_r_image_render_and_rescaling(self):

        for limit in [None, 20]:  # different resolution limits

            for i, ratio in enumerate([1/6, 0.38, 1, 5]):  # different side ratios

                img, P, L = self.gen_r_image(ratio=ratio, limit=limit, threading=bool(i % 2))  # toggle threading

                # check power sum
                P0 = img.power()
                L0 = img.luminous_power()
                self.assertAlmostEqual(P0, P)
                self.assertAlmostEqual(L0, L)

                ratio_act = img.Nx / img.Ny
                sx0, sy0 = img.sx, img.sy

                for i, Npx in enumerate([*ot.RImage.SIZES, *np.random.randint(1, 2*ot.RImage.SIZES[-1], 3)]):  # different side lengths
                    img.rescale(Npx)
                    self.assertEqual(img.N, min(img.Nx, img.Ny))

                    # check that nearest matching side length was chosen
                    siz = ot.RImage.SIZES
                    near_siz = siz[np.argmin(np.abs(Npx - np.array(siz)))]
                    cmp_siz1 = img.Nx if ratio_act < 1 else img.Ny
                    self.assertEqual(near_siz, cmp_siz1)

                    self.assertAlmostEqual(P0, img.power())  # overall power stays the same even after rescaling
                    self.assertAlmostEqual(ratio_act, img.Nx/img.Ny)  # ratio stayed the same
                    self.assertEqual(sx0, img.sx)  # side length stayed the same
                    self.assertEqual(sy0, img.sy)  # side length stayed the same
                    self.assertAlmostEqual(img.sx*img.sy/img.Nx/img.Ny, img.Apx)  # check pixel area

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
        RIm.render(keep_extent=False)
        self.assertFalse(np.all(RIm.extent == Im_zero_ext))
        RIm = ot.RImage(extent=zero_ext)
        RIm.render(keep_extent=True)
        self.assertTrue(np.all(RIm.extent == Im_zero_ext))
        
        # image with high aspect ratio in y 
        hasp_ext = [0, 1, 0, 1.2*ot.RImage.MAX_IMAGE_RATIO]
        RIm = ot.RImage(extent=hasp_ext)
        Im_hasp_ext = RIm.extent.copy()
        RIm.render(keep_extent=False)
        self.assertFalse(np.all(RIm.extent == Im_hasp_ext))
        RIm = ot.RImage(extent=hasp_ext)
        RIm.render(keep_extent=True)
        self.assertTrue(np.all(RIm.extent == Im_hasp_ext))
        
        # image with high aspect ratio in x
        hasp_ext = [0, 1.2*ot.RImage.MAX_IMAGE_RATIO, 0, 1]
        RIm = ot.RImage(extent=hasp_ext)
        Im_hasp_ext = RIm.extent.copy()
        RIm.render(keep_extent=False)
        self.assertFalse(np.all(RIm.extent == Im_hasp_ext))
        RIm = ot.RImage(extent=hasp_ext)
        RIm.render(keep_extent=True)
        self.assertTrue(np.all(RIm.extent == Im_hasp_ext))

        # coverage: _dont_rescale = True
        ext = [0, 1, 0, 1]
        RIm = ot.RImage(extent=ext)
        RIm.render(_dont_rescale=True)

    def gen_r_image(self, N=10000, N_px=ot.RImage.SIZES[6], ratio=1, limit=None, threading=True):
        # create some image
        img = ot.RImage([-1, 1, -1*ratio, 1*ratio], silent=True, limit=limit, threading=threading)
        p = np.zeros((N, 2))
        p[:, 0] = np.random.uniform(img.extent[0], img.extent[1], N)
        p[:, 1] = np.random.uniform(img.extent[2], img.extent[3], N)
        w = np.random.uniform(1e-9, 1, N)
        wl = np.random.uniform(*color.WL_BOUNDS, N)
        img.render(N_px, p, w, wl)
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
        self.assertTrue(img2.sx == img.sx)
        self.assertTrue(img2.sy == img.sy)
        self.assertTrue(img2.projection is None)

        # coverage: don't provide file type ending
        path2 = "test_img"
        img.save(path2)
        os.remove(path)

        # throw IOError if invalid file path
        self.assertRaises(IOError, img.save, "./hsajkfhajkfhjk/hajfhsajkfhajksfhjk/ashjsafhkj")

    @pytest.mark.slow
    @pytest.mark.os
    def test_r_image_export(self):

        path = "export_image.png"
        path2 = "export_image"

        # check display modes and channels
        for ratio in [0.746, 1, 2.45]:
            img = self.gen_r_image(ratio=ratio)[0]
            img.silent = True
            img.rescale(90)
                
            for imm in ot.RImage.display_modes:
                img.export_png(path, imm, log=True, flip=False)
                img.export_png(path, imm, log=False, flip=True)
                im3 = np.asarray(PILImage.open(path), dtype=np.float64)  # load from disk
                self.assertTrue(im3.ndim == (2 if not imm.startswith("sRGB") else 3))  # check number of channels

        # coverage: non-default resample parameter
        img.export_png(path, imm, log=True, flip=False, resample=0)

        # coverage: file ending not provided
        img.export_png(path2, imm, log=True, flip=False)

        # check shapes
        for ratio in [0.746, 1, 2.45]:
            img = self.gen_r_image(ratio=ratio)[0]
            img.silent = True

            for Npx in [9, 90, 1000]:

                img.rescale(Npx)
                
                for Ns in [256, 512, 984]:
                    img.export_png(path, imm, log=False, flip=True, size=Ns)
                    im3 = np.asarray(PILImage.open(path), dtype=np.float64)  # load from disc
                    self.assertAlmostEqual(ratio, im3.shape[0]/im3.shape[1], delta=2/Ns)  # check ratio

        os.remove(path)

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

                img.limit = limit
                img.render(900, p, w, wl)
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
        img = ot.RImage([-1, 1, -1, 1], projection="abc", limit=1)
        self.assertRaises(RuntimeError, img.render)
        self.assertRaises(RuntimeError, img.refilter)
        self.assertRaises(RuntimeError, img.rescale, 315)

    @pytest.mark.os
    @pytest.mark.slow
    def test_image_presets(self) -> None:

        for imgi in ot.presets.image.all_presets:
            RT = ot.Raytracer(outline=[-3, 3, -3, 3, 0, 6], silent=True)
            RSS = ot.RectangularSurface(dim=[6, 6])
            RS = ot.RaySource(RSS, pos=[0, 0, 0], image=imgi)
            RT.add(RS)
            RT.trace(500000)
            img = RT.source_image(256)
            
            otp.r_image_plot(img, "sRGB (Absolute RI)")

        plt.close('all')


if __name__ == '__main__':
    unittest.main()
