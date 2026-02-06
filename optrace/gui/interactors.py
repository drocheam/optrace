from __future__ import annotations
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


class Picker:

    def __init__(self, 
                 scene:         QtInteractor, 
                 callback:      Callable, 
                 right_button:  bool = False, 
                 tolerance:     float = 0.0025)\
            -> None:
        """
        a mouse picker that calls it callback regardless of something has been hit or not
        and only calls it when the mouse has not moved due to dragging etc.
        Comparable to:
        https://github.com/enthought/mayavi/blob/2cf91a40fb4d584a454e3b9f1c9fb6a7d9fed2e5/mayavi/core/mouse_pick_dispatcher.py
        """
        self._mm = 0
        self._scene = scene
        self._mmb = False
        self._tolerance = tolerance
        self._callback = callback
        self._right_button = right_button

    def _on_mouse_move(self, camera: pyvista.plotting.camera.Camera, event: str) -> None:
        """on mouse move, set move variable"""
        self._mm = True

    def _on_button_press(self, iren: vtkmodules.vtkRenderingUI.vtkGenericRenderWindowInteractor, event: str) -> None:
        """on button press, reset mouse move variable and set correct button variable"""
        self._mm = False
        self._mmb = True

    def _on_button_release(self, iren: vtkmodules.vtkRenderingUI.vtkGenericRenderWindowInteractor, event: str) -> None:
        """pick if correct button and mouse not moved"""
        if not self._mm:
            if self._mmb:
                x, y = iren.GetEventPosition()
                self._scene.picker.Pick(x, y, 0, self._scene.renderer)

    def _on_pick(self, picker: vtkmodules.vtkRenderingCore.vtkPointPicker, event: str) -> None:
        """run callback if picked and corrected button used"""

        if self._mmb:
            self._callback(picker, event)

        self._mm = False
        self._mmb = False

    def activate(self) -> None:
        """activate picker"""

        self._scene.camera.AddObserver("ModifiedEvent", self._on_mouse_move)
        self._scene.interactor.AddObserver("EndInteractionEvent", self._on_button_release)
        self._scene.interactor.AddObserver("LeftButtonPressEvent" if not self._right_button\
                                                                  else "RightButtonPressEvent", self._on_button_press)
        self._scene.picker.AddObserver("EndPickEvent", self._on_pick)
        self._scene.picker.tolerance = self._tolerance


class CameraOrientationWidgetFixes:
        
    def __init__(self, 
                 plotter:           ScenePlotting, 
                 scene:             QtInteractor, 
                 orientation_axes:  vtkmodules.vtkInteractionWidgets.vtkCameraOrientationWidget)\
            -> None:
        """
        Apply the following fixes to the Camera Orienation Widget:
        - Clicking the +Y or -Y handle shows the X axis in up or down direction
        - Clicking the handles does not change the zoom or camera position
        - No rendering of the view while applying these properties
        """
        self._plotter = plotter
        self._scene = scene
        self._orientation_axes = orientation_axes
        self.__oa_cam_props = None

    def _on_orientation_change_start(self, 
                                     widget:    vtkmodules.vtkInteractionWidgets.vtkCameraOrientationWidget, 
                                     event:     str)\
            -> None:
        """store camera props and disable render in the case that a handle is selected"""
        self._scene.interactor.EnableRenderOff()
        self.__oa_cam_props = self._plotter.get_camera()
   
    def _on_orientation_change_interact(self,
                                        widget: vtkmodules.vtkInteractionWidgets.vtkCameraOrientationWidget,
                                        event:  str)\
            -> None:
        """reenable rendering while interacting with the widget"""
        self._scene.interactor.EnableRenderOn()
    
    def _on_orientation_change_end(self,
                                   widget: vtkmodules.vtkInteractionWidgets.vtkCameraOrientationWidget,
                                   event:  str)\
            -> None:
        """restore camera center and scale and handle +y and -y views so x points up/downwards"""
        if widget.GetRepresentation().IsAnyHandleSelected():
            normal = self._plotter.get_camera()[2]
            roll = 90 if abs(normal[1]) == 1 else 0
            self._plotter.set_camera(*self.__oa_cam_props[:2], roll=roll)
            self._scene.interactor.EnableRenderOn()

    def activate(self) -> None:
        """apply fixes"""
        self._orientation_axes.AddObserver("StartInteractionEvent", self._on_orientation_change_start)
        self._orientation_axes.AddObserver("InteractionEvent", self._on_orientation_change_interact)
        self._orientation_axes.AddObserver("EndInteractionEvent", self._on_orientation_change_end)


class KeyboardShortcuts:

    def __init__(self, plot: ScenePlotting, ui: TraceGUI, scene: QtInteractor) -> None:
        """remove old shortcuts from vtk and pyvista and apply new ones"""
        self._ui = ui
        self._plot = plot
        self._scene = scene

    def keyrelease(self, vtk_obj: vtkmodules.vtkRenderingUI.vtkGenericRenderWindowInteractor, event: str) -> None:
        """callback on key_release"""

        match vtk_obj.GetKeyCode():

            case "i":  # reset view
                self._plot.set_initial_camera()
                self._scene.render()

            case "h":  # hide/show side menu and toolbar
                self._ui.maximize_scene = bool(self._ui._scene_not_maximized)  # toggle value

            case "v":  # toggle minimalistic_view
                self._ui.minimalistic_view = not bool(self._ui.minimalistic_view)
            
            case "c":  # high_contrast
                self._ui.high_contrast = not bool(self._ui.high_contrast)

            case "b":  # toggle label visibility
                self._ui.hide_labels = not bool(self._ui.hide_labels)
            
            case "d":  # render DetectorImage
                self._ui.detector_image()
           
            case "0":  # close all pyplots
                plt.close("all")

            case "n":  # reselect and replot rays
                self._ui.replot_rays()
            
            case "+":  # zoom in
                self._scene.camera.zoom(1.1)
                self._scene.render()

            case "-":  # zoom out
                self._scene.camera.zoom(1/1.1)
                self._scene.render()
            
            # move camera (no shift) or rotate view (with shift)
            case _ if (dir_ := vtk_obj.GetKeySym()) in ("Up", "Down", "Left", "Right"):
                
                cam = self._scene.camera
                right = np.cross(cam.GetDirectionOfProjection(), cam.up)

                if self._scene.interactor.shift_key:

                    if dir_ == "Up":
                        cam.elevation = cam.elevation + 5

                    elif dir_ == "Down":
                        cam.elevation = cam.elevation - 5

                    elif dir_ == "Left":
                        cam.azimuth = cam.azimuth + 5

                    elif dir_ == "Right":
                        cam.azimuth = cam.azimuth - 5

                else:

                    if dir_ == "Up":
                        dvec = np.array(cam.up)*cam.parallel_scale/20
                    
                    elif dir_ == "Down":
                        dvec = -np.array(cam.up)*cam.parallel_scale/20
                    
                    elif dir_ == "Right":
                        dvec = np.array(right)*cam.parallel_scale/15
                    
                    elif dir_ == "Left":
                        dvec = -np.array(right)*cam.parallel_scale/15

                    cam.position = cam.position + dvec
                    cam.focal_point = cam.focal_point + dvec

                # recalculate up view and reset camera clipping range
                cam.up = np.cross(cam.GetDirectionOfProjection(), -right)
                cam.reset_clipping_range()
                self._scene.render()

    def activate(self) -> None:
        """Override old shortcuts and apply new ones"""

        # remove default shortcuts from vtk and pyvistaqt
        self._scene.iren.interactor.RemoveObservers('CharEvent')
        self._scene.iren.interactor.RemoveObservers('KeyPressEvent')
        self._scene.iren.interactor.RemoveObservers('KeyReleaseEvent')

        # add own
        self._scene.iren.interactor.AddObserver('KeyPressEvent', self.keyrelease)

