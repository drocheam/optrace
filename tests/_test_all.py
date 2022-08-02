#!/bin/env python3

# additional underscore in file name so pytest does not load this file automatically, but only the other sub-test files

import unittest
from test_frontend import FrontendTests
from test_examples import ExampleTests
from test_backend_modules import BackendModuleTests
from test_plots import PlotTests
from test_tracer import TracerTests


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BackendModuleTests))
    suite.addTest(unittest.makeSuite(TracerTests))
    suite.addTest(unittest.makeSuite(PlotTests))
    suite.addTest(unittest.makeSuite(FrontendTests))
    suite.addTest(unittest.makeSuite(ExampleTests))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
