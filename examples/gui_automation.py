#!/usr/bin/env python3

import time
import numpy as np

import optrace as ot
from optrace.gui import TraceGUI

# this example shows the automation capabilities of the GUI
# the position and size of a source are varied and the scene and rays are updated after each step
# The automation function can be rerun by pressing the button in the "Custom" GUI tab.

RT = ot.Raytracer(outline=[-10, 10, -10, 10, -25, 40])

# Line Source emitting parallel white light
RSS0 = ot.Line(r=1, angle=90)
RS0 = ot.RaySource(RSS0, divergence="None", spectrum=ot.presets.light_spectrum.d65,
                  pos=[0, 0, -10], s=[0, 0, 1])
RT.add(RS0)

# create a sphere lens with R=5
n = ot.RefractionIndex("Constant", n=1.3)
front = ot.SphericalSurface(r=4.99999999, R=5)
back = ot.SphericalSurface(r=4.99999999, R=-5)
L = ot.Lens(front, back, d=10, pos=[0, 0, 0], n=n)
RT.add(L)

# thing to automate
def automated(GUI):

    # time between changes
    sleeping_time = 1
    
    # change settings (but these could also be set initializing TraceGUI())
    GUI.minimalistic_view = True
    GUI.hide_labels = True

    # zoom in to the relevant part
    GUI.set_camera(center=[0, 0, 4], height=10)
   
    # GUI properties were set, but the changes need to be processed
    GUI.process()

    # default state, needed to rerun this function
    with GUI.smart_replot():
        RT.ray_sources[0].set_surface(ot.Line(r=1, angle=90))
        RT.ray_sources[0].move_to([0, 0, -15])

    # vary the lateral source position
    for yp in np.linspace(1, 4, 4):

        # replot/retrace things that changed automaticalyy
        with GUI.smart_replot():

            time.sleep(sleeping_time)
            RT.ray_sources[0].move_to([0, yp, -15])
        
    # reset
    RT.ray_sources[0].move_to([0, 0, -15])
    
    # vary the source size
    for ri in np.linspace(0.5, 5, 5):

        with GUI.smart_replot():

            time.sleep(sleeping_time)
            RT.ray_sources[0].set_surface(ot.Line(r=ri, angle=90))

# Note that the automation function is run in the main thread (as user input would also be)
# so the interaction with the scene in this time will be limited

# create the GUI and provide the automation function to TraceGUI.control()
sim = TraceGUI(RT)
sim.add_custom_button("Rerun", lambda: automated(sim))
sim.control(func=automated, args=(sim,))
