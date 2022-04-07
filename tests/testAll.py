#!/bin/env python3

import unittest
from testExamples import ExampleTests
from testBackendModules import BackendModuleTests
from testFrontend import FrontendTests


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BackendModuleTests))
    suite.addTest(unittest.makeSuite(ExampleTests))
    suite.addTest(unittest.makeSuite(FrontendTests))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
