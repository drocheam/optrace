#!/bin/env python3

import sys
sys.path.append('.')

# add PYTHONPATH to env, so the examples can find optrace
import os

if "TOX_ENV_DIR" not in os.environ:
    os.environ["PYTHONPATH"] = "."

import unittest
import subprocess
import pytest

# Test that the library does not load internal classes or external libraries into its or the global namespace

# loading optrace should not load optrace.gui or optrace.plots, 
# optrace.plots should not load optrace.gui
# no external libraries like numpy, scipy should be in the global namespace
# internal classes like Surface, Element, RayStorage, ... are not in the same namespace


class ScopeTests(unittest.TestCase):

    def _run_command(self, command, timeout=40):
        """run a subprocess and get exit code"""

        # we need to run a seperate process so that the script is in a default state without libraries loaded
        process = subprocess.Popen(["python", "-c", command], stdout=subprocess.DEVNULL, env=os.environ)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

        return process.returncode

    def test_scope_partial_load(self):
        """test that importing optrace does not import optrace.plots and optrace.gui by default"""
        
        # check that importing works correctly, code == 0
        code = self._run_command("import optrace as ot")
        self.assertEqual(code, 0)
        
        # plots not loaded, code != 0
        code = self._run_command("import optrace as ot; ot.plots")
        self.assertNotEqual(code, 0)
        
        # plots gets loaded, code == 0
        code = self._run_command("import optrace as ot; import optrace.plots; ot.plots")
        self.assertEqual(code, 0)
        
        # gui not loaded, code != 0
        code = self._run_command("import optrace as ot; ot.gui")
        self.assertNotEqual(code, 0)
        
        # gui gets loaded, code == 0
        code = self._run_command("import optrace as ot; import optrace.gui; ot.gui")
        self.assertEqual(code, 0)

    # needs to be run first, otherwise ot.plots and ot.gui are already loaded
    def test_scope_main(self):
        """test that internal/unneeded things in optrace.tracer and optrace itself are not loaded"""
        
        import optrace as ot
        self.assertRaises(AttributeError, eval, "ot.Element", locals())
        self.assertRaises(AttributeError, eval, "ot.BaseClass", locals())
        self.assertRaises(AttributeError, eval, "ot.BaseImage", locals())
        self.assertRaises(AttributeError, eval, "ot.misc", locals())
        self.assertRaises(AttributeError, eval, "ot.Surface", locals())
        self.assertRaises(AttributeError, eval, "ot.ray_storage", locals())
        self._test_scope_ext_libs(locals())

    def _test_scope_ext_libs(self, loc):
        """test that external or standard libraries don't get loaded into the global namespace"""
        
        self.assertRaises(NameError, eval, "numpy", loc)
        self.assertRaises(NameError, eval, "np", loc)
        self.assertRaises(NameError, eval, "scipy", loc)
        self.assertRaises(NameError, eval, "chardet", loc)
        self.assertRaises(NameError, eval, "matplotlib", loc)
        self.assertRaises(NameError, eval, "plt", loc)
        self.assertRaises(NameError, eval, "os", loc)
        self.assertRaises(NameError, eval, "sys", loc)
        self.assertRaises(NameError, eval, "typing", loc)
        self.assertRaises(NameError, eval, "mayavi", loc)
        self.assertRaises(NameError, eval, "pyface", loc)
        self.assertRaises(NameError, eval, "traits", loc)
        self.assertRaises(NameError, eval, "traitsui", loc)
        self.assertRaises(NameError, eval, "warnings", loc)
        self.assertRaises(NameError, eval, "time", loc)
        self.assertRaises(NameError, eval, "contextlib", loc)
        self.assertRaises(NameError, eval, "threading", loc)
        self.assertRaises(NameError, eval, "enum", loc)
        self.assertRaises(NameError, eval, "tqdm", loc)
        self.assertRaises(NameError, eval, "cv2", loc)
        self.assertRaises(NameError, eval, "functools", loc)
        self.assertRaises(NameError, eval, "pathlib", loc)
        self.assertRaises(NameError, eval, "qdarktheme", loc)
    
    def test_scope_plots(self):
        """test that internal/unneeded things in optrace.plots are not loaded"""
        
        import optrace.plots as otp
        self.assertRaises(AttributeError, eval, "otp._check_types", locals())
        self.assertRaises(AttributeError, eval, "otp._check_labels", locals())
        self.assertRaises(AttributeError, eval, "otp._show_grid", locals())
        self.assertRaises(AttributeError, eval, "otp._spectrum_plot", locals())
        self.assertRaises(AttributeError, eval, "otp._chromacity_plot", locals())
        self.assertRaises(AttributeError, eval, "otp._red_xyz", locals())
        self.assertRaises(AttributeError, eval, "otp._CONV_XYZ_NORM", locals())
        self._test_scope_ext_libs(locals())
    
    def test_scope_gui(self):
        """test that internal/unneeded things in optrace.gui are not loaded"""

        import optrace.gui as otg
        self.assertRaises(AttributeError, eval, "otg.PropertyBrowser", locals())
        self.assertRaises(AttributeError, eval, "otg.CommandWindow", locals())
        self._test_scope_ext_libs(locals())

    def test_init_order(self):

        eval_str = "import optrace as ot; import optrace.plots as otp; import optrace.gui as otg"
        code = self._run_command(eval_str)
        self.assertEqual(code, 0)
        
        eval_str = "import optrace as ot; import optrace.gui as otg; import optrace.plots as otp"
        code = self._run_command(eval_str)
        self.assertEqual(code, 0)
        
        eval_str = "import optrace.gui as otg; import optrace.plots as otp; import optrace as ot"
        code = self._run_command(eval_str)
        self.assertEqual(code, 0)
        
        eval_str = "import optrace.plots as otp; import optrace.gui as otg; import optrace as ot"
        code = self._run_command(eval_str)
        self.assertEqual(code, 0)

if __name__ == '__main__':
    unittest.main()

