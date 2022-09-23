#!/bin/env python3

# additional underscore in file name so pytest does not load this file automatically, but only the other sub-test files

import unittest
from test_gui import GUITests
from test_examples import ExampleTests
from test_trace_modules import TraceTests
from test_geometry import GeometryTests
from test_spectrum import SpectrumTests
from test_plots import PlotTests
from test_tracer import TracerTests
from test_presets import PresetTests


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SpectrumTests))
    suite.addTest(unittest.makeSuite(GeometryTests))
    suite.addTest(unittest.makeSuite(TraceTests))
    suite.addTest(unittest.makeSuite(PresetTests))
    suite.addTest(unittest.makeSuite(TracerTests))
    suite.addTest(unittest.makeSuite(PlotTests))
    suite.addTest(unittest.makeSuite(GUITests))
    suite.addTest(unittest.makeSuite(ExampleTests))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
