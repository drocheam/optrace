#!/bin/env python3

import unittest
import numpy as np

import optrace.tracer.color as color
import optrace as ot


class RefractionIndexTests(unittest.TestCase):

    def test_refraction_index_base(self):

        func = lambda wl: 2.0 - wl/500/5 
        func2 = lambda wl, a: a - wl/500/5 
        wlf = color.wavelengths(1000)
        funcf = func(wlf)
        
        n_list = [

        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/archer.agf
        (ot.RefractionIndex("Schott", 
                            coeff=[2.417473, -0.008685888, 0.01396835, 0.0006180845, -5.274288e-05, 3.679696e-06],
                            desc="S-BAL3M"), 1.568151, 52.737315),

        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/birefringent.agf
        (ot.RefractionIndex("Sellmeier1", 
                            coeff=[1.29899, 0.0089232927, 43.17364, 1188.531, 0.0, 0.0],
                            desc="ADP"), 1.523454, 52.25678),

        (ot.RefractionIndex("Sellmeier4", 
                            coeff=[3.1399, 1.3786, 0.02941, 3.861, 225.9009], 
                            desc="ALN"), 2.154291, 50.733102),

        (ot.RefractionIndex("Handbook of Optics 1", 
                            coeff=[2.7405, 0.0184, 0.0179, 0.0155], 
                            desc="BBO"), 1.670737, 52.593907),

        (ot.RefractionIndex("Sellmeier3", 
                            coeff=[0.8559, 0.00345744, 0.8391, 0.019881, 0.0009, 0.038809, 0.6845, 49.070025], 
                            desc="CALCITE"), 1.658643, 48.541403),

        (ot.RefractionIndex("Sellmeier5", 
                            coeff=[0.8559, 0.00345744, 0.8391, 0.019881, 0.0009, 0.038809, 0.6845, 49.070025, 0, 0], 
                            desc="CALCITE"), 1.658643, 48.541403),

        (ot.RefractionIndex("Handbook of Optics 2", 
                            coeff=[2.81418, 0.87968, 0.09253764, 0.00711], 
                            desc="ZNO"), 2.003385, 12.424016),

        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/hikari.agf
        (ot.RefractionIndex("Extended3", 
                            coeff=[3.22566311, -0.0126719158, -0.000122584245, 0.0306900263, 
                                   0.000649958511, 1.0629994e-05, 1.20774149e-06, 0.0, 0.0],
                            desc="Q-LASFH19S"), 1.82098, 42.656702),

        (ot.RefractionIndex("Extended2", 
                            coeff=[2.54662904, -0.0122972332, 0.0187464623, 0.000460296583, 7.85351066e-07,
                                  1.72720972e-06, -0.000133476806, 0.0], 
                            desc="E-KZFH1"), 1.61266, 44.461379),

        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/infrared.agf
        (ot.RefractionIndex("Herzberger", 
                            coeff=[2.2596285, 0.0311097853, 0.0010251756, -0.000594355286, 
                                   4.49796618e-07, -4.12852834e-09], 
                            desc="CLRTR_OLD"), 2.367678, 15.282309),

        # https://raw.githubusercontent.com/nzhagen/zemaxglass/master/AGF_files/lightpath.agf
        (ot.RefractionIndex("Conrady", 
                            coeff=[1.47444837, 0.0103147698, 0.00026742387], 
                            desc="NICHIA_MELT1"), 1.493724, 64.358478),

        # https://refractiveindex.info/?shelf=glass&book=OHARA-PHM&page=PHM51
        (ot.RefractionIndex("Extended", 
                            coeff=[2.5759016, -0.010553544, 0.013895937, 0.00026498331, 
                                   -1.9680543e-06, 1.0989977e-07, 0, 0], 
                            desc="PHM"), 1.617,  62.8008),

        # elements from presets
        (ot.presets.refraction_index.COC, 1.5324098, 56.0522),

        # different modes
        (ot.RefractionIndex("Abbe", n=1.578, V=70), 1.578, 70),
        (ot.RefractionIndex("Constant", n=1.2546), 1.2546, np.inf),
        (ot.RefractionIndex("Function", func=func), 1.764976, 11.2404),
        (ot.RefractionIndex("Function", func=func2, func_args=dict(a=2)), 1.764976, 11.2404),
        (ot.RefractionIndex("Data", wls=wlf, vals=funcf), 1.764976, 11.2404),
        ]

        for nl_ in n_list:
            n, nc, V = nl_

            n(color.wavelengths(1000))  # call with array

            self.assertEqual(n.is_dispersive(), n.spectrum_type != "Constant")
            self.assertAlmostEqual(n(np.array(ot.presets.spectral_lines.d)), nc, delta=5e-5)
            self.assertAlmostEqual(n.abbe_number(), V, delta=0.3)

        # check if equal operator is working
        self.assertEqual(ot.presets.refraction_index.SF10, ot.presets.refraction_index.SF10)
        self.assertEqual(n_list[1][0], n_list[1][0])
        self.assertNotEqual(n_list[1][0], n_list[2][0])
        assert n_list[-1][0].spectrum_type == "Data"
        self.assertEqual(n_list[-1][0], n_list[-1][0])  # comparision of Data type
        self.assertEqual(ot.RefractionIndex("Function", func=func), ot.RefractionIndex("Function", func=func))

    def test_refraction_index_abbe_mode(self):

        for lines in [*ot.presets.spectral_lines.all_line_combinations, None]:
            for nc in [1.01, 1.15, 1.34, 1.56, 1.913, 2.678]:
                for V in [15, 37.56, 78, 156]:
                  n = ot.RefractionIndex("Abbe", n=nc, V=V, lines=lines)
                  self.assertAlmostEqual(nc, n(n.lines[1]), delta=1e-4)
                  self.assertAlmostEqual(V, n.abbe_number(lines), delta=1e-2)  # enforce lines...
                  self.assertAlmostEqual(V, n.abbe_number(), delta=1e-2)  # ...but should correct one anyway

    def test_refraction_index_exceptions(self):

        # value exceptions
        self.assertRaises(ValueError, ot.RefractionIndex, "ABC")  # invalid type
        self.assertRaises(ValueError, ot.RefractionIndex, "Constant", n=0.99)  # n < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Constant", n=np.inf)  # n not finite
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=0)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=-1)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=np.inf)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=np.nan)  # invalid V
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 480])  
        # ^-- lines need to have 3 elements
        self.assertRaises(ValueError, ot.RefractionIndex, "Abbe", V=50, lines=[380, 780, 480])  
        # ^-- lines need to be ascending
        self.assertRaises(ValueError, ot.RefractionIndex, "Function", func=lambda wl: 0.5 - wl/wl)  # func < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Data", wls=[380, 780], vals=[1.5, 0.9])  # vals < 1
        self.assertRaises(ValueError, ot.RefractionIndex, "Cauchy", coeff=[1, 0, 0, 0, 0])  # too many coeff

        # type errors
        self.assertRaises(TypeError, ot.RefractionIndex, "Cauchy", coeff=1)  # invalid coeff type
        self.assertRaises(TypeError, ot.RefractionIndex, "Abbe", V=[1])  # invalid V type
        self.assertRaises(TypeError, ot.RefractionIndex, "Constant", n=[5])  # n not a float
        n2 = ot.RefractionIndex("Function")
        self.assertRaises(TypeError, n2, 550)  # func missing
        self.assertRaises(TypeError, ot.RefractionIndex("Cauchy"), 550)  # coeffs not specified
        
        # misc
        self.assertRaises(AttributeError, n2.__setattr__, "aaa", 1)  # _new_lock active
        self.assertRaises(RuntimeError, ot.RefractionIndex("Cauchy", coeff=[2, -1, 0, 0]), np.array([380., 780.]))  
        # n < 1 on runtime
        
        # check exceptions when wavelengths are outside data range
        wl0, wl1 = ot.global_options.wavelength_range
        ot.global_options.wavelength_range[1] = wl1 + 100
        self.assertRaises(RuntimeError, ot.presets.refraction_index.PET, color.wavelengths(1000))
        ot.global_options.wavelength_range[0] = wl0-100
        ot.global_options.wavelength_range[1] = wl1
        self.assertRaises(RuntimeError, ot.presets.refraction_index.PET, color.wavelengths(1000))

        # reset color bounds
        ot.global_options.wavelength_range[:] = [wl0, wl1]

    def test_refraction_index_equality(self):

        # equal operator
        self.assertTrue(ot.RefractionIndex("Constant") == ot.RefractionIndex("Constant"))
        self.assertFalse(ot.RefractionIndex("Constant", n=2) == ot.RefractionIndex("Constant", n=1))
        self.assertFalse(ot.RefractionIndex("Constant", n=2) == 1)  # comparison between different types
        self.assertTrue(ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 2]) ==\
                        ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 2]))
        self.assertFalse(ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 2]) ==\
                         ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 1]))
        self.assertFalse(ot.RefractionIndex("Data", wls=[400, 500], vals=[1, 2]) ==\
                         ot.RefractionIndex("Data", wls=[400, 501], vals=[1, 2]))

    def test_refraction_index_presets(self):
        """check that all refraction index presets are callable 
        and their values and abbe numbers are in a reasonable range.
        All presets also should have descriptions"""

        wl = color.wavelengths(1000)

        # check presets
        for material in ot.presets.refraction_index.all_presets:
            n = material(wl)
            A0 = material.abbe_number()

            self.assertTrue(np.all((1 <= n) & (n <= 2.5)))  # sane refraction index everywhere
            self.assertTrue(A0 == np.inf or A0 < 150)  # sane Abbe Number

            # should have descriptions
            self.assertNotEqual(material.desc, "")
            self.assertNotEqual(material.long_desc, "")

            # real dispersive materials have a declining n
            if material.is_dispersive() and material.desc not in ["PEI", "PVC"]:  # exclude because of noisy data
                self.assertTrue(np.all(np.diff(n) < 0)) # steadily declining


if __name__ == '__main__':
    unittest.main()
