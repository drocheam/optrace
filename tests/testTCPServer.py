#!/bin/env python3

import unittest
import subprocess
import warnings

from testFrontend import RT_Example

import optrace as ot
from optrace.gui import TraceGUI
import optrace.gui.TCPServer as TCPServer
import time

import socket 


# the twisted reactor is not restartable
# therefore we can only start and end the reactor once for all tests
# that is why the server is only run in one test

class TCPTests(unittest.TestCase):

    def setUp(self) -> None:
        # deactivate warnings
        warnings.simplefilter("ignore")
    
    def tearDown(self) -> None:
        # reset warnings
        warnings.simplefilter("default")
    
    def send(self, cmd):

        s = socket.socket()
        s.connect(('localhost', TCPServer.TCP_port))
        
        s.send(cmd.encode())

        s.close()

    def test_Server(self):
        RT = RT_Example()
        sim = TraceGUI(RT, silent=True)

        def interact(sim):
            sim._waitForIdle()

            state = RT.Rays.crepr()
            self.send("GUI.replot()")
            sim._waitForIdle()
            self.assertFalse(state == RT.Rays.crepr()) # check if raytraced

            self.send("GUI.showDetectorImage()")
            sim._waitForIdle()
            self.assertTrue(sim.lastDetImage is not None) # check if raytraced
           
            # create optrace objects
            LLlen = len(RT.LensList)
            self.send("a = Surface(\"Circle\");"
                       "b=Surface(\"Sphere\", rho=-1/10);"
                       "L = Lens(a, b, n=presets.RefractionIndex.SF10, de=0.2, pos=[0, 0, 8]);"
                       "RT.add(L)")
            sim._waitForIdle()
            self.assertEqual(len(RT.LensList), LLlen+1)

            # numpy and time available
            self.send("a = np.array([1, 2, 3]);"
                      "time.time()")
            sim._waitForIdle()

            state = RT.Rays.crepr()
            self.send("RT.remove(APL[0])")
            sim._waitForIdle()
            self.assertEqual(len(RT.ApertureList), 0) # check if ApertureList empty after removal
            self.assertEqual(len(sim._AperturePlotObjects), 0) # check if aperture plot is removed
            self.assertFalse(state == RT.Rays.crepr()) # check if raytraced
           
            # send command to command elements in UI and execute
            self.assertEqual(sim.Command_History, "")
            self.assertEqual(len(RT.LensList), 5)
            sim._setInMain("_Cmd", "RT.remove(LL[0])")
            sim._doInMain(sim.sendCmd)
            sim._waitForIdle()
            self.assertEqual(len(RT.LensList), 4)
            self.assertNotEqual(sim.Command_History, "")  # history changed

            sim._doInMain(sim.close)

        sim.debug(_func=interact, silent=True, no_server=False, _args=(sim,))


if __name__ == '__main__':
    # deactivate warnings temporarily
    warnings.simplefilter("ignore")
    unittest.main()
    warnings.simplefilter("default")

