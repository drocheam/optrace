#!/bin/env python3

import unittest
from testFrontend import FrontendTests
from testExamples import ExampleTests
from testBackendModules import BackendModuleTests
from testPlots import PlotTests


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BackendModuleTests))
    suite.addTest(unittest.makeSuite(PlotTests))
    suite.addTest(unittest.makeSuite(FrontendTests))
    suite.addTest(unittest.makeSuite(ExampleTests))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
