#!/bin/env python3

import unittest
import subprocess
import warnings


from testFrontend import RT_Example

import optrace as ot
from optrace.gui import TraceGUI
import optrace.gui.TCPServer as TCPServer
import time


# TODO more tests
# TODO make command cross platform
class TCPTests(unittest.TestCase):

    def setUp(self) -> None:
        # deactivate warnings
        warnings.simplefilter("ignore")
    
    def tearDown(self) -> None:
        # reset warnings
        warnings.simplefilter("default")
    
    # execute, but kill after timeout, since everything should be automated
    # higher timeout for a human viewer to see if everythings working
    def send(self, cmds, timeout=20):

        cmd = "{ sleep 1 "
        for cmdi in cmds:
            cmd += f'; echo "{cmdi}"; sleep 1 '
        cmd += ' ; echo "quit()"; } '
        cmd += f"| telnet localhost {TCPServer.TCP_port}"

        # start process. ignore warnings and redirect stdout to null
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    def loadScene(self):
        RT = RT_Example()
        sim = TraceGUI(RT, silent=True)

        def interact(sim):
            sim.waitForIdle()

            state = RT.Rays.crepr()
            self.send(["GUI.replot()"])
            sim.waitForIdle()
            self.assertFalse(state == RT.Rays.crepr()) # check if raytraced

            state = RT.Rays.crepr()
            self.send(["RT.remove(AL[0])"])
            sim.waitForIdle()
            self.assertEqual(len(RT.ApertureList), 0) # check if ApertureList empty after removal
            self.assertEqual(len(sim.AperturePlotObjects), 0) # check if aperture plot is removed
            self.assertFalse(state == RT.Rays.crepr()) # check if raytraced
            
            self.send(["GUI.close()"])
            time.sleep(2)

        sim.run(_func=interact, silent=True, _args=(sim,))

    def test_0(self):
        self.loadScene()


if __name__ == '__main__':
    # deactivate warnings temporarily
    warnings.simplefilter("ignore")
    unittest.main()
    warnings.simplefilter("default")

