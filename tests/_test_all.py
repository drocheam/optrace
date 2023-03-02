#!/bin/env python3

# additional underscore in file name so pytest does not load this file automatically, but only the other sub-test files


import unittest
from test_color import ColorTests
from test_examples import ExampleTests
from test_geometry import GeometryTests
from test_gui import GUITests
from test_convolve import ConvolutionTests
from test_load import LoadTests
from test_plots import PlotTests
from test_presets import PresetTests
from test_scope import ScopeTests
from test_spectrum import SpectrumTests
from test_surface import SurfaceTests
from test_tma import TMATests
from test_trace_misc import TracerMiscTests
from test_tracer import TracerTests
from test_tracer_special import TracerSpecialTests


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ColorTests))
    suite.addTest(unittest.makeSuite(ExampleTests))
    suite.addTest(unittest.makeSuite(GeometryTests))
    suite.addTest(unittest.makeSuite(GUITests))
    suite.addTest(unittest.makeSuite(LoadTests))
    suite.addTest(unittest.makeSuite(PlotTests))
    suite.addTest(unittest.makeSuite(ConvolutionTests))
    suite.addTest(unittest.makeSuite(PresetTests))
    suite.addTest(unittest.makeSuite(ScopeTests))
    suite.addTest(unittest.makeSuite(SpectrumTests))
    suite.addTest(unittest.makeSuite(SurfaceTests))
    suite.addTest(unittest.makeSuite(TMATests))
    suite.addTest(unittest.makeSuite(TracerMiscTests))
    suite.addTest(unittest.makeSuite(TracerSpecialTests))
    suite.addTest(unittest.makeSuite(TracerTests))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
