from __future__ import annotations
from contextlib import contextmanager  # context manager for _no_trait_action()
from typing import assert_never

import numpy as np
import copy
import matplotlib.pyplot as plt 

import pyvista as pv
import vtk

from ..tracer.geometry import Surface, Element
from ..tracer import *
from ..warnings import warning
from ..global_options import global_options as go
from ..tracer.misc import masked_assign

from .interactors import Picker, CameraOrientationWidgetFixes, KeyboardShortcuts


class ScenePlotting:
    """
    This class provides the functionality for plotting elements and rays of the raytracer inside a mayavi scene.
    It uses properties and settings from a TraceGUI.
    """

    SURFACE_RES: int = 150
    """Surface sampling count in each dimension"""

    MAX_RAYS_SHOWN: int = 50000
    """Maximum of rays shown in visualization"""

    LABEL_STYLE: dict = dict(font_size=12, bold=True, font_family="courier", shadow=True, italic=False)
    """Standard Text Style. Used for object labels, legends and axes"""

    INFO_STYLE: dict = dict(font_size=14, bold=True, font_family="courier", shadow=True, italic=False)
    """Info Text Style. Used for status messages and interaction overlay"""

    ##########

    def __init__(self, 
                 ui:                TraceGUI, 
                 raytracer:         Raytracer,
                 initial_camera:    dict = {})\
            -> None:
        """
        Init the ScenePlotting object

        :param ui: TraceGUI
        :param raytracer: raytracer (the same that the TraceGUI uses)
        :param initial_camera: keyword dictionary for set_camera()
        """
        self.ui = ui
        self.scene = self.ui.scene
        self.raytracer = raytracer

        # plot object lists
        self._lens_plots = []
        self._axes_labels = []
        self._filter_plots = []
        self._aperture_plots = []
        self._detector_plots = []
        self._ray_source_plots = []
        self._point_marker_plots = []
        self._line_marker_plots = []
        self._index_box_plots = []
        self._volume_plots = []
        self._fault_markers = []
        self._ray_plot = None
        self._crosshair = None
        self._outline = None
        self._orientation_axes = None
        self._ray_highlight_plot = None

        # initial camera settings
        self._initial_camera = dict(direction=[1, 0, 0]) | initial_camera

        # texts 
        self._status_text = None
        self._ray_text = None

        # ray properties
        self.__ray_property_dict = {}  # properties of shown rays, set while tracing
        self._ray_property_dict = {}  # properties of shown rays, set after tracing
        self.ray_selection = None
        
    # Helper Functions
    ###################################################################################################################

    def __remove_objects(self, objs: list) -> None:
        """remove visual objects from raytracer geometry"""

        for obj in objs:
            for obji in obj[:4]:
                if obji is not None:
                    self.scene.remove_actor(obji, render=False)

        objs[:] = []
    
    @staticmethod
    def apply_prop(prop, **kwargs):
        """apply properties to object prop"""
        for key, val in kwargs.items():
            if hasattr(prop, key):
                setattr(prop, key, val)
   
    def screenshot(self, path: str = None, **kwargs) -> np.ndarray:
        """
        Save a screenshot of the scene. Passes the parameters down to the pyvista.Plotter.screenshot function.
        The function returns a numpy array for the image, which also can be saved by providing the path parameter.
        """
        self._status_text.SetVisibility(False)  # temporarily hide status text so it won't be on the screenshot
        arr = self.scene.screenshot(path, **kwargs)
        self._status_text.SetVisibility(True)
        return arr

    def get_camera(self) -> tuple[np.ndarray, float, np.ndarray, float]:
        """
        Get the camera parameters that can be passed down to set_camera()

        :return: Return the current camera parameters (center, height, direction, roll)
        """
        return self.scene.camera.focal_point,\
               self.scene.camera.parallel_scale,\
               np.array(self.scene.camera.GetDirectionOfProjection()),\
               self.scene.camera.roll\

    def set_camera(self, 
                   center:          np.ndarray = None, 
                   height:          float = None,
                   direction:       list = None,
                   roll:            float = None)\
            -> None:
        """
        Sets the camera view.
        Not all parameters must be defined, setting single properties is also allowed.
        
        :param center: 3D coordinates of center of view in mm
        :param height: half of vertical height in mm
        :param direction: camera view direction vector 
        (direction of vector perpendicular to your monitor and in your viewing direction)
        :param roll: absolute camera roll angle in degrees
        """
        # force parallel projection
        self.scene.parallel_projection = True
       
        # calculate vector between camera and focal point
        cam = self.scene.camera
        normal = np.asarray(direction, dtype=np.float64) if direction is not None\
                else np.array(cam.GetDirectionOfProjection())
        dist_vec = cam.distance * normal / (np.linalg.norm(normal) + 1e-12) 

        # old cross product of direction and view_up
        right = np.cross(cam.GetDirectionOfProjection(), cam.up)

        if center is not None:
            cam.focal_point = center

        if height is not None:
            cam.parallel_scale = height

        # set/update camera position (since either/both focal_point or direction changed)
        cam.position = cam.focal_point - dist_vec

        # update view_up from new normal and old cross product of direction and view_up 
        cam.up = np.cross(cam.GetDirectionOfProjection(), -right)

        # absolute roll angle
        if roll is not None:
            cam.roll = roll
        
        # reset clipping
        cam.reset_clipping_range()

    def set_initial_camera(self) -> None:
        """
        sets the initial camera view

        When parameter initial_camera is not set, it is a y-side view with all elements inside the viewable range
        When it is set, the camera properties are applied.
        """
        self.scene.view_vector((-1, 0, 0), viewup=(0, 1, 0), render=False)
        self.scene.camera.parallel_scale *= 0.90
        self.set_camera(**self._initial_camera)

    # Element Plotting
    ###################################################################################################################

    def plot_lenses(self) -> None:
        """replot all lenses from raytracer"""
        self.__remove_objects(self._lens_plots)

        for num, L in enumerate(self.raytracer.lenses):
            t = self.plot_element(L, num, self._lens_color[:3], self._lens_alpha)
            self._lens_plots.append(t)

    def plot_apertures(self) -> None:
        """replot all apertures from raytracer"""
        self.__remove_objects(self._aperture_plots)

        for num, AP in enumerate(self.raytracer.apertures):
            t = self.plot_element(AP, num, self._aperture_color, 1)
            self._aperture_plots.append(t)

    def plot_filters(self) -> None:
        """replot all filter from raytracer"""
        self.__remove_objects(self._filter_plots)

        for num, F in enumerate(self.raytracer.filters):

            fcolor = F.color()
            alpha = 0.1 + 0.899*fcolor[3]  # offset in both directions, ensures visibility and see-through

            t = self.plot_element(F, num, fcolor[:3] if not self.ui.high_contrast else self._foreground_color, alpha)
            self._filter_plots.append(t)

    def plot_detectors(self) -> None:
        """replot all detectors from taytracer"""
        self.__remove_objects(self._detector_plots)

        for num, Det in enumerate(self.raytracer.detectors):
            t = self.plot_element(Det, num, self._detector_color[:3], self._detector_alpha)
            self._detector_plots.append(t)

    def plot_ray_sources(self) -> None:
        """replot all ray sources from raytracr"""
        self.__remove_objects(self._ray_source_plots)

        for num, RS in enumerate(self.raytracer.ray_sources):
            t = self.plot_element(RS, num, (1., 1., 1.), self._raysource_alpha, spec=False, light=False)
            self._ray_source_plots.append(t)

    def plot_outline(self) -> None:
        """replot the raytracer outline"""

        if self._outline is not None:
            self.scene.remove_actor(self._outline, render=False)

        outline_mesh = pv.Box(bounds=self.raytracer.outline.copy())
        self._outline = self.scene.add_mesh(outline_mesh, color=self._outline_color, style='wireframe', lighting=False,
                                            name="Outline", pickable=False, line_width=1, render=False)

    def plot_orientation_axes(self) -> None:
        """plot orientation axes"""

        if self._orientation_axes is not None:
            return

        # create Orientation Widget and apply fixes
        self._orientation_axes = self.scene.add_camera_orientation_widget(animate=False, n_frames=0)
        fixes = CameraOrientationWidgetFixes(self, self.scene, self._orientation_axes)
        fixes.activate()

        self._orientation_axes.GetRepresentation().SetSize(90, 90)
        self._orientation_axes.GetRepresentation().AnchorToLowerLeft()
        self._orientation_axes.GetRepresentation().SetVisibility(not bool(self.ui.minimalistic_view))
        
        # optional: adapt colors. Only supported from vtk>=9.6
        # if hasattr(self._plot._orientation.GetRepresentation(), "SetXAxisColor"):
            # self._plot._orientation.GetRepresentation().SetXAxisColor(0.7, 0., 0.)
            # self._plot._orientation.GetRepresentation().SetYAxisColor(0., 0.7, 0.)
            # self._plot._orientation.GetRepresentation().SetZAxisColor(0., 0., 0.7)

    @staticmethod
    def calculate_labels(x0: float, x1: float, min_s: int, max_s: int) -> np.ndarray:

        # NOTE min_s and max_s should be in the range 3-50

        # desired spacings sp and subjective weights spw (e.g. spacing of 0.25 is more natural than 0.4)
        sp = np.array([0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 1, 1.25, 1.5, 2, 2.5, 3, 3.75, 4, 5, 6, 6.25, 
                       7.5, 8, 10, 12.5, 15, 20, 25, 40])
        spw = np.array([1, 2, 0.8, 2, 0.5, 0.6, 2, 1, 1, 2, 1, 0.8, 0.5, 1, 2, 0.5, 0.5, 0.5, 1, 2, 1, 1, 2, 2, 1])
        
        # range normalization so (x1-x0)*norm is in range [10, 100]
        norm = 10 ** -np.floor(np.log10(x1-x0)-1)
       
        # get label counts of normalized numbers x0*norm and x1*norm
        lf0 = np.ceil(x0*norm/sp).astype(int)  # label factor for smallest element
        lf1 = np.floor(x1*norm/sp).astype(int)  # label factor for largest element
        ln = lf1 - lf0  # label count

        # product of spacing weights and label counts in valid range by [min_s, max_s]
        w = ((ln >= min_s) & (ln <= max_s))*spw 

        # find element with highest weight
        if np.any(w > 0):
            ind = np.argmax(w)
            return np.arange(lf0[ind], lf1[ind]+1)*sp[ind]/norm

        # return linspace of full range with maximum label count
        return np.linspace(x0, x1, max_s+1)


    def plot_axes(self) -> None:
        """plot cartesian axes"""

   
        def add_point_labels(x, y, name="", orientation=0, **kwargs):
            i = 0
            el = []
            for xi, yi in zip(x, y):
                ax = self.scene.add_point_labels([xi], [yi], name=f"{name}{i}", **kwargs)
                ax.GetMapper().Update()  # update so we can set text properties
                ax.GetMapper().GetInputDataObject(0, 0).GetTextProperty().SetOrientation(orientation)
                ax.visibility = not self.ui.minimalistic_view
                el.append(ax)
                i += 1
            
            return el
        
        ext = self.raytracer.outline
        args = dict(font_size=11, bold=True, font_family="courier", pickable=False, fill_shape=False, render=False, 
                    justification_horizontal="center", justification_vertical="center", always_visible=True, 
                    text_color=self._axis_color, show_points=False, shadow=not self.ui.high_contrast, shape_opacity=0.)

        # X
        lx = self.calculate_labels(ext[0], ext[1], 4, 16)
        Lx = np.vstack((lx, np.repeat(ext[2], len(lx)), np.repeat(ext[4], len(lx)))).T
        Lxs = [f"{lxi:.6g} –" for lxi in lx]
        axes_x = add_point_labels(Lx, Lxs, name=f"Labels x", **(args | dict(justification_horizontal="right")))
        axes_x2 = add_point_labels([[(ext[0]+ext[1])/2, ext[2], ext[4]]], ["\nx / mm" + len(Lxs[len(Lxs)//2])*2*" "],
                                   name=f"Title x", **(args | dict(justification_horizontal="right")))
       
        # Y
        ly = self.calculate_labels(ext[2], ext[3], 4, 16)
        Ly = np.vstack((np.repeat(ext[0], len(ly)), ly, np.repeat(ext[4], len(ly)))).T
        Lys = [f"{lyi:.6g} –" for lyi in ly] 
        axes_y = add_point_labels(Ly, Lys, name=f"Labels y", **(args | dict(justification_horizontal="right")))
        axes_y2 = add_point_labels([[ext[0], (ext[2]+ext[3])/2, ext[4]]], ["y / mm" + len(Lys[len(Lys)//2])*2*" "], 
                                   name=f"Title y",  **(args | dict(justification_horizontal="right")))
        
        # Z
        lz = self.calculate_labels(ext[4], ext[5], 5, 24)
        Lz = np.vstack((np.repeat(ext[0], len(lz)), np.repeat(ext[2], len(lz)), lz)).T
        Lzs = [f"\n\n{lzi:.6g}" for lzi in lz] 
        axes_z = add_point_labels(Lz, Lzs, name=f"Labels z", **args)
        axes_z2 = add_point_labels(Lz, ["–"]*len(Lzs), name=f"Ticks z", orientation=90, 
                                   **(args | dict(justification_horizontal="right")))
        axes_z3 = add_point_labels([[ext[0], ext[2], (ext[4]+ext[5])/2]], ["\n\nz / mm"], name=f"Title z", 
                       **(args | dict(justification_vertical="top")))
        
        self._axes_labels = [*axes_x, *axes_x2, *axes_y, *axes_y2, *axes_z, *axes_z2, *axes_z3]


    def plot_index_boxes(self) -> None:
        """plot outlines for ambient refraction index regions"""

        self.__remove_objects(self._index_box_plots)

        # sort Element list in z order
        Lenses = sorted(self.raytracer.lenses, key=lambda element: element.pos[2])

        # create n list and z-boundary list
        nList = [self.raytracer.n0] + [element.n2 for element in Lenses] + [self.raytracer.n0]
        BoundList = [self.raytracer.outline[[4, 4]]] +\
                    [(np.mean(element.front.extent[4:]), np.mean(element.back.extent[4:])) for element in Lenses] +\
                    [self.raytracer.outline[[5, 5]]]

        # replace None values of Lenses n2 with the ambient n0
        nList = [self.raytracer.n0 if ni is None else ni for ni in nList]

        # delete boxes with too small extent
        i = 0
        while i < len(nList)-2:
            # delete if negative or small positive value
            if BoundList[i+1][0] - BoundList[i][1] < 5e-4:
                del nList[i], BoundList[i]
            else:
                i += 1
        
        # join boxes with the same refraction index
        i = 0
        while i < len(nList)-2:
            if nList[i] == nList[i+1]:
                del nList[i+1], BoundList[i+1]
            else:
                i += 1

        # skip box plotting if n=1 everywhere
        if len(BoundList) == 2 and nList[0] == RefractionIndex():
            return

        # plot boxes
        for i in range(len(BoundList)-1):

            # plot outline
            outline_mesh = pv.Box(bounds=[*self.raytracer.outline[:4], BoundList[i][1], BoundList[i + 1][0]])
            outline_corners = outline_mesh.outline_corners(factor=0.1)
            outline = self.scene.add_mesh(outline_corners, color=self._outline_color, style='wireframe', lighting=False,
                                          name=f"Refraction Index Outline {i}", pickable=False, 
                                          line_width=1, render=False)

            # label position
            y_pos = self.raytracer.outline[2] + (self.raytracer.outline[3] - self.raytracer.outline[2]) * 0.05
            x_pos = np.mean(self.raytracer.outline[:2])
            z_pos = np.mean([BoundList[i][1], BoundList[i+1][0]])

            # plot label
            label = (f"ambient\n" if not self.ui.minimalistic_view else "")  + "n=" + nList[i].get_desc()
            text = self._plot_label([x_pos, y_pos, z_pos], label, f"Refraction Index Outline Label {i}",
                                    text_color=self._axis_color, shadow=not self.ui.high_contrast, bold=False,
                                    background_opacity=0,
                                    map_args=dict(frame_color=self._subtle_color, show_frame=True))

            # append plot objects
            self._index_box_plots.append((outline, None, None, text, None))

    def plot_element(self,
                     obj:           Element,
                     num:           int,
                     color:         tuple,
                     alpha:         float,
                     spec:          bool = True,
                     light:         bool = True,
                     no_label:      bool = False)\
            -> tuple:
        """plotting of a Element. Gets called from plotting for Lens, Filter, Detector, RaySource. """

        def plot(C, surf_type):

            grid = pv.StructuredGrid(C[0], C[1], C[2])
            name = f"{type(obj).__name__} {num} {surf_type} surface"
            
            # Determine material properties
            p_specular = 0.25 if spec else 0.0
            p_ambient = 0.4 if spec else 0.1
            
            actor = self.scene.add_mesh(grid, color=color, opacity=alpha, name=name, render=False, 
                                        show_scalar_bar=False, pickable=False, lighting=light, 
                                        specular=p_specular, ambient=p_ambient, style='surface')
            
            return actor

        # decide what to plot
        plotFront = isinstance(obj.front, Surface)
        plotCyl = plotFront
        plotBack = obj.back is not None and isinstance(obj.back, Surface)

        # plot front
        nres = self.SURFACE_RES if not plotFront or not isinstance(obj.front, RectangularSurface) else 10
        a = plot(obj.front.plotting_mesh(N=nres), "front") if plotFront else None

        # # cylinder between surfaces
        b = plot(obj.cylinder_surface(nc=2 * self.SURFACE_RES), "cylinder") if plotCyl else None

        # # adapt cylinder opacity
        if plotCyl and isinstance(obj, Lens):
            b.prop.opacity = self._cylinder_opacity

        # # use wireframe mode, so edge is always visible, even if it is infinitesimal small
        if plotCyl and (not obj.has_back() or (obj.extent[5] - obj.extent[4] < 0.05)):
            self.apply_prop(b.prop, style='wireframe', lighting=False, line_width=1.75, opacity=alpha/1.5)

        # # calculate middle center z-position
        if obj.has_back():
            zl = (obj.front.values(np.array([obj.front.pos[0]]), np.array([obj.front.extent[3]]))[0] \
                  + obj.back.values(np.array([obj.back.pos[0]]), np.array([obj.back.extent[3]]))[0]) / 2
        else:
            # 0/40 values in [0, 2pi] -> z - value at 0 deg relative to x axis in xy plane
            zl = obj.front.edge(40)[2][0] if isinstance(obj.front, Surface) else obj.pos[2]

        # Handle Labels
        text_actor = None

        if not no_label:
            label_str = f"{obj.abbr}{num}"
            if obj.desc != "" and not bool(self.ui.minimalistic_view):
                label_str += f": {obj.desc}"

            # Using add_point_labels for 3D placement that stays visible
            text_pos = [obj.pos[0], obj.extent[3], zl]
            text_actor = self._plot_label(text_pos, label_str, f"Label_{obj.abbr}{num}", 
                                          text_color=self._foreground_color,
                                          background_opacity=self._info_opacity, 
                                          background_color=self._info_frame_color)

        # plot BackSurface if one exists
        c = plot(obj.back.plotting_mesh(N=self.SURFACE_RES), "back") if plotBack else None

        return a, b, c, text_actor, obj

    @staticmethod
    def mesh_from_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                         u: np.ndarray, v: np.ndarray, w: np.ndarray, s: np.ndarray) -> pyvista.core.pointset.PolyData:
        """create PolyData from point coordinates"""
        
        # point array
        nodes = np.empty((2 * len(x), 3))
        nodes[0::2] = np.column_stack((x, y, z))
        nodes[1::2] = np.column_stack((x + u, y + v, z + w))

        # line connectivity
        connectivity = np.empty(3 * len(x), dtype=np.int32)
        connectivity[0::3] = 2
        connectivity[1::3] = np.arange(0, 2 * len(x), 2)
        connectivity[2::3] = np.arange(1, 2 * len(x), 2)

        # construct PolyData
        ray_mesh = pv.PolyData(nodes, lines=connectivity)
        ray_mesh.point_data["scalars"] = np.repeat(s, 2)
        return ray_mesh

    def plot_rays(self, 
                  x:    np.ndarray, 
                  y:    np.ndarray, 
                  z:    np.ndarray,
                  u:    np.ndarray, 
                  v:    np.ndarray, 
                  w:    np.ndarray, 
                  s:    np.ndarray)\
            -> None:
        """plot a subset of traced rays"""

        if self._ray_plot is not None:
            self.scene.remove_actor(self._ray_plot, render=False)

        ray_mesh = self.mesh_from_points(x, y, z, u, v, w, s)

        # toggle between points and lines
        style = 'points' if self.ui.plotting_mode == 'Points' else 'wireframe'
       
        # add to scene
        self._ray_plot = self.scene.add_mesh(ray_mesh, scalars="scalars", cmap="Greys", opacity=self.ui.ray_opacity,
                                             line_width=self.ui.ray_width, point_size=self.ui.ray_width, 
                                             style=style, lighting=False, name=f"Rays", render_points_as_spheres=True, 
                                             pickable=True, render=False, scalar_bar_args=dict(render=False))

    def set_ray_highlight(self, index: int) -> None:
        """
        Highlight the ray with index 'index'.
        Assigns the positions to the _ray_highlight_plot and makes it visible
        """
        p_ = self.__ray_property_dict["p"]
        s_un = self.__ray_property_dict["s_un"]

        x, y, z = p_[index, :, 0].flatten(), p_[index, :, 1].flatten(), p_[index, :, 2].flatten()
        u, v, w = s_un[index, :, 0].flatten(), s_un[index, :, 1].flatten(), s_un[index, :, 2].flatten()

        if self._ray_highlight_plot is not None:
            self.scene.remove_actor(self._ray_highlight_plot, render=False)

        lt = pv.LookupTable(values=[np.array([*self._crosshair_color, 1.])*255])
        ray_mesh = self.mesh_from_points(x, y, z, u, v, w, np.ones(x.shape[:2]))

        # add to scene
        self._ray_highlight_plot = self.scene.add_mesh(ray_mesh, scalars="scalars", cmap=lt, opacity=1,
                                                       line_width=self.ui.ray_width, point_size=self.ui.ray_width, 
                                                       style="wireframe", render=False, lighting=False, 
                                                       name="Ray Highlight", render_points_as_spheres=True, 
                                                       pickable=False, show_scalar_bar=False)

    def plot_point_markers(self) -> None:
        """plot point markers inside the scene"""
        self.__remove_objects(self._point_marker_plots)

        for num, mark in enumerate(self.raytracer.markers):

            if not isinstance(mark, PointMarker):
                continue
                
            dy, dx = 0.2 * mark.marker_factor, 0

            if not mark.label_only:
                actor = self.scene.add_point_labels([mark.pos], ["+"], font_family="times",
                                                    name=f"Marker Cross {num}", render=False, pickable=False,
                                                    font_size=int(15*mark.marker_factor),
                                                    text_color=self._marker_color, show_points=False, shape_opacity=0,
                                                    justification_horizontal="center", justification_vertical="center",
                                                    always_visible=True)
            else:
                actor = None

            text_actor = self._plot_label([mark.pos[0]+dx, mark.pos[1]+dy, mark.pos[2]], mark.desc, 
                                          f"Marker Label {num}", text_color=self._foreground_color,
                                          background_opacity=self._info_opacity,
                                          background_color=self._info_frame_color, font_size=int(10*mark.text_factor))
                
            self._point_marker_plots.append((actor, None, None, text_actor, mark))
   
    def _plot_label(self, pos: list, label: str, name: str, map_args: dict = None, **kwargs)\
            -> vtkmodules.vtkRenderingCore.vtkActor2D:
        """plot a scene label that reacts to changes of hide_label, minimal_scene, vertical_labels and high_contrast"""    
        pargs = dict(render=False, text_color=self._foreground_color, show_points=False, 
                     shape_opacity=0, background_opacity=self._info_opacity, 
                     background_color=self._info_frame_color, always_visible=True)

        text_actor = self.scene.add_point_labels([pos], [label], name=name, **(pargs | self.LABEL_STYLE | kwargs))
            
        text_actor.GetMapper().Update()  # update so we can set text properties
        tprop = text_actor.mapper.GetInputDataObject(0, 0).text_property

        if self.ui.vertical_labels:
            self.apply_prop(tprop, orientation=90, justification_horizontal="left", 
                            justification_vertical="center")
        else:
            self.apply_prop(tprop, justification_horizontal="center", justification_vertical="bottom")

        if map_args is not None:
            self.apply_prop(tprop, **map_args)

        if self.ui.hide_labels:
            text_actor.visibility = False

        return text_actor

    def plot_line_markers(self) -> None:
        """plot line markers inside the scene"""
        self.__remove_objects(self._line_marker_plots)

        for num, mark in enumerate(self.raytracer.markers):
            if isinstance(mark, LineMarker):

                drx = mark.front.r * np.cos(np.radians(mark.front.angle))
                dry = mark.front.r * np.sin(np.radians(mark.front.angle))
                dx, dy = mark.pos[0]+drx, mark.pos[1]+dry

                grid = pv.PolyData(np.array([[mark.pos[0]-drx, mark.pos[1]-dry, mark.pos[2]],
                                             [mark.pos[0]+drx, mark.pos[1]+dry, mark.pos[2]]]), lines=[2, 0, 1])

                m = self.scene.add_mesh(grid, color=self._line_marker_color, name=f"Line Marker {num}", render=False, 
                                        show_scalar_bar=False, pickable=False, lighting=False, style='wireframe',
                                        line_width=mark.line_factor)

                text = self._plot_label([mark.pos[0]+dx, mark.pos[1]+dy, mark.pos[2]], mark.desc, 
                        f"Line Marker Label {num}", text_color=self._foreground_color,
                                        background_opacity=self._info_opacity,
                                        background_color=self._info_frame_color, font_size=int(10*mark.text_factor))

                self._line_marker_plots.append((m, None, None, text, mark))

    def plot_volumes(self) -> None:
        """plot volumes inside the scene"""
        self.__remove_objects(self._volume_plots)

        for num, V in enumerate(self.raytracer.volumes):
            color = self._volume_color if (V.color is None or self.ui.high_contrast) else np.array(V.color)
            t = self.plot_element(V, num, color, V.opacity, no_label=True, spec=False)
            self._volume_plots.append(t)

    # Initialization
    ###################################################################################################################

    def set_colors(self) -> None:
        """initialize or change colors depending on high_contrast setting"""

        high_contrast = self.ui.high_contrast

        self._lens_alpha =          0.35
        self._detector_alpha =      0.99
        self._raysource_alpha =     0.55
        self._info_opacity =        0.2
        self._aperture_color =      (0., 0., 0.)
        self._crosshair_color =     (1., 0., 0.)
        self._background_color =    (0.2, 0.2, 0.2)      if not high_contrast else (1., 1., 1.)
        self._foreground_color =    (1., 1., 1.)         if not high_contrast else (0., 0., 0.)
        self._lens_color =          (0.63, 0.79, 1.00)   if not high_contrast else self._foreground_color
        self._detector_color =      (0.0, 0.0, 0.0)      if not high_contrast else self._foreground_color
        self._subtle_color =        (0.3, 0.3, 0.3)      if not high_contrast else (0.7, 0.7, 0.7)
        self._marker_color =        (0., 1., 0.)         if not high_contrast else self._foreground_color
        self._line_marker_color =   (0.8, 0, 0.8)        if not high_contrast else self._foreground_color
        self._outline_color =       (0.5, 0.5, 0.5)      if not high_contrast else (0.8, 0.8, 0.8)
        self._axis_color =          (0.5, 0.5, 0.5)      if not high_contrast else (0.7, 0.7, 0.7)
        self._info_frame_color =    (0., 0., 0.)         if not high_contrast else (1., 1., 1.)
        self._volume_color =        (0.45, 0.45, 0.45)   if not high_contrast else (1., 1., 1.)
        self._cylinder_opacity =    self._lens_alpha     if not high_contrast else 0.6

    def init_keyboard_shortcuts(self) -> None:
        """init keyboard shortcut detection inside the scene"""

        shortcuts = KeyboardShortcuts(self, self.ui, self.scene)
        shortcuts.activate()

    def init_ray_info(self) -> None:
        """init detection of ray point clicks and the info display"""

        picker = Picker(self.scene, self._on_ray_pick, right_button=False)
        picker.activate()

        self._ray_text = self.scene.add_text("", position="upper_left", name="Ray Info Text", render=False)
        
        self._ray_text.SetMinimumFontSize(self.INFO_STYLE["font_size"])
        self._ray_text.SetMaximumFontSize(self.INFO_STYLE["font_size"])
        self.apply_prop(self._ray_text.prop, **self.INFO_STYLE)
        self.apply_prop(self._ray_text.prop, background_opacity=self._info_opacity,
                        opacity=1, color=self._foreground_color, vertical_justification="top",
                        background_color=self._info_frame_color)

    def init_status_info(self) -> None:
        """init GUI status text display"""

        picker = Picker(self.scene, self._on_space_pick, right_button=True)
        picker.activate()

        self._status_text = self.scene.add_text("", position="lower_right", name="Status Info Text", render=False)
        
        self._status_text.SetMinimumFontSize(self.INFO_STYLE["font_size"])
        self._status_text.SetMaximumFontSize(self.INFO_STYLE["font_size"])
        self.apply_prop(self._status_text.prop, **(self.INFO_STYLE | dict(color=self._foreground_color)))

    # Scene Changes
    ###################################################################################################################

    def change_label_orientation(self) -> None:
        """Set labels of Elements to vertical or horizontal depending on option vertical_labels"""
       
        if self.ui.vertical_labels:
            opts = dict(justification_horizontal="left", orientation=90, justification_vertical="center")
        else:
            opts = dict(justification_horizontal="center", justification_vertical="bottom", orientation=0) \

        for objs in [self._lens_plots, self._detector_plots, self._aperture_plots, self._filter_plots,
                     self._index_box_plots, self._point_marker_plots, self._line_marker_plots, self._ray_source_plots,
                     self._index_box_plots]:
            for obj in objs:
                if obj[3] is not None:
                    self.apply_prop(obj[3].mapper.GetInputDataObject(0, 0).text_property, **opts)

    def change_minimalistic_view(self) -> None:
        """Hide long labels, orientation axes and normal axes depending on if option minimalistic_view is set"""

        show = not bool(self.ui.minimalistic_view)

        # hide orientation axes
        if self._orientation_axes is not None:
            self._orientation_axes.GetRepresentation().SetVisibility(show)

        # shorten index plot description
        for rio in self._index_box_plots:
            if rio[3] is not None:
                label = rio[3].mapper.GetInputDataObject(0, 0).labels
                text = label.GetValue(0)
                label.SetValue(0, text.replace("ambient\n", "") if not show else ("ambient\n" + text))

        # remove descriptions from labels in minimalistic_view
        for Objects in [self._ray_source_plots, self._lens_plots, self._volume_plots, 
                        self._filter_plots, self._aperture_plots, self._detector_plots]:
            for num, obj in enumerate(Objects):
                if obj[3] is not None and obj[4] is not None:
                    label = f"{obj[4].abbr}{num}"
                    label = label if obj[4].desc == "" or not show else label + ": " + obj[4].desc
                    obj[3].mapper.GetInputDataObject(0, 0).labels.SetValue(0, label)

        for ax in self._axes_labels:
            ax.visibility = show
           
    def change_label_visibility(self) -> None:
        """Hide/show labels depending on value of option 'hide_labels'"""

        # remove descriptions from labels in minimalistic_view
        for Objects in [self._ray_source_plots, self._lens_plots, self._point_marker_plots, self._volume_plots,
                        self._index_box_plots, self._line_marker_plots, self._filter_plots,
                        self._aperture_plots, self._detector_plots]:
            for num, obj in enumerate(Objects):
                if obj[3] is not None:
                    obj[3].visibility = not bool(self.ui.hide_labels)

    def change_contrast(self) -> None:
        """Normal or high contrast scene mode depending on state of option 'high_contrast'"""

        self.set_colors()
        self.scene.set_background(self._background_color)
        high_contrast = self.ui.high_contrast

        # update filter color
        for F in self._filter_plots:
            for Fi in F[:2]:
                if Fi is not None and F[4] is not None:
                    Fi.prop.color = np.array(F[4].color()[:3]) if not high_contrast else self._foreground_color
        
        # update volume colors
        for V in self._volume_plots:
            for Vi in V[:3]:
                Vi.prop.color = self._volume_color if V[4].color is None or high_contrast else np.array(V[4].color)

        # update misc color
        for color, objs in zip([self._lens_color, self._detector_color, self._aperture_color, 
                                self._line_marker_color, self._outline_color],
                               [self._lens_plots, self._detector_plots, self._aperture_plots,
                                self._line_marker_plots, [[self._outline]]]):
            for obj in objs:
                for el in obj[:3]:
                    if el is not None and hasattr(el, "prop"):
                        el.prop.color = color

        # special case: point marker plots are just labels
        for el in self._point_marker_plots:
            if el[0] is not None:
                el[0].mapper.GetInputDataObject(0, 0).text_property.color = self._marker_color

        # update background colors of labels
        for objs in [self._lens_plots, self._detector_plots, self._aperture_plots, self._filter_plots,
                     self._point_marker_plots, self._line_marker_plots, self._volume_plots, self._ray_source_plots]:
            for obj in objs:
                if len(obj) > 3 and obj[3] is not None:
                    obj[3].mapper.GetInputDataObject(0, 0).text_property.background_color = self._background_color
                    obj[3].mapper.GetInputDataObject(0, 0).text_property.color = self._foreground_color

        # change lens cylinder visibility
        for lens in self._lens_plots:
            if lens[1] is not None:
                lens[1].prop.opacity = self._cylinder_opacity

        # update axes color
        for ax in self._axes_labels:
            ax.mapper.GetInputDataObject(0, 0).text_property.color = self._axis_color
            ax.mapper.GetInputDataObject(0, 0).text_property.SetShadow(not high_contrast)

        # change index plot objects
        for obj in self._index_box_plots:
    
            if obj[3] is not None:
                tprop = obj[3].mapper.GetInputDataObject(0, 0).text_property
                tprop.frame_color = self._subtle_color
                tprop.color = self._axis_color
                tprop.SetShadow(not high_contrast)

            if obj[0] is not None:
                obj[0].prop.color = self._outline_color

        # reassign ray source colors
        self.color_ray_sources()

        # in coloring type Plain the ray color is changed from white to a bright orange
        if self.ui.coloring_mode == "Plain" and self._ray_plot is not None:
            vals = [255, 255, 255, 255] if not bool(self.ui.high_contrast) else [255, 132, 0, 255] 
            self._ray_plot.mapper.lookup_table.SetTable(pv.convert_array(np.array([vals], dtype=np.uint8)))

        # update status and ray text
        for text in [self._status_text, self._ray_text]:
            if text is not None:
                text.prop.background_color = self._info_frame_color
                text.prop.color = self._foreground_color

        # change scalar bar
        if len(self.scene.scalar_bars):
            self.scene.scalar_bar.GetTitleTextProperty().color = self._foreground_color
            self.scene.scalar_bar.GetLabelTextProperty().color = self._foreground_color

    def set_ray_opacity(self) -> None:
        """change the ray opacity"""
        if self._ray_plot is not None:
            self._ray_plot.prop.opacity = self.ui.ray_opacity
    
    def set_ray_representation(self) -> None:
        """change the ray representation between 'points' and 'surface'"""
        if self._ray_plot is not None:
            self._ray_plot.prop.style = 'points' if self.ui.plotting_mode == 'Points' else 'wireframe'

    def set_ray_width(self) -> None:
        """change the ray width"""
        if self._ray_plot is not None:
            self._ray_plot.prop.line_width = self.ui.ray_width
            self._ray_plot.prop.point_size = self.ui.ray_width

    def set_status(self, _status: dict[str]) -> None:
        """sets the status in the status text depending on the _status dictionary"""

        msgs = {"RunningCommand": "Running Command",
                "Tracing": "Raytracing",
                "Focussing": "Finding Focus",
                "ChangingDetector": "Changing Detector",
                "DetectorImage": "Rendering Detector Image",
                "SourceImage": "Rendering Source Image",
                "SourceSpectrum": "Rendering Source Spectrum",
                "DetectorSpectrum": "Rendering Detector Spectrum",
                "Drawing": "Updating Scene",
                "Screenshot": "Saving a Screenshot"}
       
        # print messages to scene
        if not _status["InitScene"] and self._status_text is not None:
            text = ""

            for key, val in msgs.items():
                if _status[key]:
                    text += msgs[key] + "...\n"
                    
            self._status_text.SetText(1, text)

    def set_fault_markers(self) -> None:
        """calculate and plot fault markers marking geometry collisions"""
        # remove old markers, generate and plot new ones
        # chose some random fault positions, maximum 5
        pfault = self.raytracer.fault_pos
        ch = min(5, pfault.shape[0])

        if ch:
            f_ind = np.random.choice(np.arange(pfault.shape[0]), size=ch, replace=False)

            self.raytracer.remove(self._fault_markers)
            self._fault_markers = [PointMarker("COLLISION", pos=pfault[ind], text_factor=1.5, 
                                               marker_factor=1.5) for ind in f_ind]
            self.raytracer.add(self._fault_markers)
            self.plot_point_markers()

    def remove_fault_markers(self) -> None:
        """remove the fault markers from the scene and raytracer"""
        if self._fault_markers:
            self.raytracer.remove(self._fault_markers)
            self._fault_markers = []
            self.plot_point_markers()

    def move_detector_diff(self, ind: int, diff: float) -> None:
        """
        move the detector plot with index 'ind' differentially in z-direction

        :param ind: detector index
        :param movement: moving distance
        """
        if ind < len(self._detector_plots):
            det = self._detector_plots[ind]

            # move surfaces
            det[0].position = np.array(det[0].position) + [0, 0, diff]
            det[1].position = np.array(det[1].position) + [0, 0, diff]

            # move label
            dataset = det[3].GetMapper().GetInputAlgorithm().GetInput()
            vtk_pts = dataset.GetPoints()
            vtk_pts.SetPoint(0, np.array(vtk_pts.GetPoint(0))+[0, 0, diff])
            det[3].GetMapper().Modified()  # notify for replot

    def clear_ray_text(self) -> None:
        """clear the ray info text"""
        self._ray_text.SetText(2, "")

    def hide_crosshair(self) -> None:
        """Hide the crosshair if it still/already exists"""
        if self._crosshair is not None:
            self._crosshair.visibility = False

    def hide_ray_highlight(self) -> None:
        """Hide the ray highlight plot if it still/already exists"""
        if self._ray_highlight_plot is not None:
            self._ray_highlight_plot.visibility = False

    def set_crosshair(self, pos: np.ndarray) -> None:
        """
        Set the crosshair position if it still/already exists
        
        :param pos: array with three elements (x, y, z)
        """
        self._crosshair = self.scene.add_point_labels(pos, ["+"], name=f"Crosshair", font_size=32, bold=True, 
                                                      font_family="times", render=False, always_visible=True, 
                                                      justification_horizontal="center", shape_opacity=0.,
                                                      justification_vertical="center", pickable=False,
                                                      text_color=self._crosshair_color, show_points=False)

    # Ray and RaySource plotting
    ###################################################################################################################

    def color_rays(self) -> None:
        """color the plotted rays in the scene depending on the coloring mode"""

        if self._ray_plot is None or not len(self.scene.scalar_bars):
            return

        rp = self._ray_property_dict
        pol_, w_, wl_, snum_, n_ = rp["pol"], rp["w"], rp["wl"], rp["snum"], rp["n"]
        N, nt, nc = pol_.shape

        # set plotting properties depending on plotting mode
        match self.ui.coloring_mode:

            case 'Power':
                s = w_.ravel()*1e6
                cm = "gnuplot"
                title = "Ray Power\n in µW\n"

            case 'Source':
                s = np.broadcast_to(snum_[:, np.newaxis], (snum_.shape[0], nt))
                cm = "spring"
                title = "Ray Source\nNumber"

            case 'Wavelength':
                s = np.broadcast_to(wl_[:, np.newaxis], (wl_.shape[0], nt))
                cm = "nipy_spectral"
                title = "Wavelength\n in nm\n"

            case 'Refractive Index':
                s = n_.ravel()
                cm = "gnuplot"
                title = "Refractive\nIndex"

            case ('Polarization xz' | 'Polarization yz'):

                if self.raytracer.no_pol:
                    warning("Polarization calculation turned off in raytracer, "
                            "reverting to a different mode")

                    self.ui.coloring_mode = "Power"
                    return

                if self.ui.coloring_mode == "Polarization yz":
                    # projection of unity vector onto yz plane is the pythagorean sum of the y and z component
                    s = np.hypot(pol_[:, :, 1], pol_[:, :, 2]).ravel()
                    title = "Polarization\n projection\n on yz-plane"

                else:
                    # projection of unity vector onto xz plane is the pythagorean sum of the x and z component
                    s = np.hypot(pol_[:, :, 0], pol_[:, :, 2]).ravel()
                    title = "Polarization\n projection\n on xz-plane"

                cm = "gnuplot"

            case _:  # Plain
                s = np.ones_like(w_)
                cm = "Greys" if not self.ui.high_contrast else "Wistia"
                title = "None"

        self._ray_plot.mapper.dataset.point_data['scalars'] = np.repeat(s, 2)  # or 'cell_data' depending on your mesh
        lutm = self._ray_plot.mapper.lookup_table
        bar = self.scene.scalar_bar
        self._ray_plot.mapper.SetScalarRange(np.min(s), np.max(s))
        lutm.apply_cmap(cm)
        table = pv.pyvista_ndarray(lutm.GetTable())

        self.apply_prop(bar, title=title, orientation=1, number_of_labels=11, label_format="%-6.3f", 
                        visibility=self.ui.coloring_mode != "Plain")
        text_style = self.INFO_STYLE | dict(font_family=1, color=self._foreground_color) 
        self.apply_prop(bar.GetTitleTextProperty(), **text_style)
        self.apply_prop(bar.GetLabelTextProperty(), **text_style)

        # right edge anchor at half height
        right_anchor = vtk.vtkCoordinate()
        right_anchor.SetCoordinateSystemToNormalizedViewport()
        right_anchor.SetValue(1.0, 0.5)

        # set distance to edge in pixels relative to anchor
        pos = bar.GetPositionCoordinate()
        pos.SetReferenceCoordinate(right_anchor)
        pos.SetCoordinateSystemToDisplay()
        pos.SetValue(-120, -250)

        # set bar width and height in pixels
        pos2 = bar.GetPosition2Coordinate()
        pos2.SetCoordinateSystemToDisplay()
        pos2.SetValue(80, 500)

        # apply colormap options
        match self.ui.coloring_mode:

            case 'Wavelength':
                spectral_colormap = go.spectral_colormap if go.spectral_colormap is not None\
                                    else color.spectral_colormap
                table = 255*spectral_colormap(color.wavelengths(255))
                lutm.SetTable(pv.convert_array(table.astype(np.uint8)))
                self._ray_plot.mapper.SetScalarRange(go.wavelength_range)
                self.scene.scalar_bar.label_format = "%-6.0f"

            case ('Polarization xz' | "Polarization yz"):
                self._ray_plot.mapper.SetScalarRange(0.0, 1.0)

            case 'Source':
                bar.number_of_labels = len(self.raytracer.ray_sources)
                bar.label_format = "%-6.0f"
                if len(self.raytracer.ray_sources) > 1:
                    table = 255*color.spectral_colormap(np.linspace(440, 620, bar.number_of_labels))
                    lutm.SetTable(pv.convert_array(table.astype(np.uint8)))

            case 'Plain':
                vals = [255, 255, 255, 255] if not bool(self.ui.high_contrast) else [255, 132, 0, 255] 
                lutm.SetTable(pv.convert_array(np.array([vals], dtype=np.uint8)))
            
            case ('Power' | 'Refractive Index'):
                pass

            case _:
                assert_never(self.ui.coloring_mode)

    def color_ray_sources(self) -> None:
        """sets colors of ray sources"""

        if self._ray_plot is None:
            return

        lutm = self._ray_plot.mapper.lookup_table
        table = pv.pyvista_ndarray(lutm.GetTable())

        match self.ui.coloring_mode:

            case ("Plain" | "Refractive Index"):
                RSColor = [self._foreground_color for RSp in self._ray_source_plots]

            case 'Wavelength':
                RSColor = [RS.color(rendering_intent="Absolute", clip=True) for RS in self.raytracer.ray_sources]

            case ('Polarization xz' | "Polarization yz"):
                # color from polarization projection on yz-plane
                RSColor = []
                for RS in self.raytracer.ray_sources:
                    
                    if RS.polarization in ["x", "y", "Constant"]:

                        match RS.polarization:
                            case "x":
                                pol_ang = 0
                            case "y":
                                pol_ang = np.pi/2
                            case "Constant":
                                pol_ang = np.deg2rad(RS.pol_angle)
                            case _:
                                assert_never(RS.polarization)
                    
                        proj = np.sin(pol_ang) if self.ui.coloring_mode == "Polarization yz" else np.cos(pol_ang)
                        col = np.array(table[int(proj*255)])
                        RSColor.append(col[:3]/255.)
                    else:
                        RSColor.append(np.ones(3))

            case 'Source':
                RSColor = [np.array(table[i][:3]) / 255. for i, _ in enumerate(self._ray_source_plots)]

            case 'Power':
                # set to maximum ray power, this is the same for all sources
                RSColor = [np.array(table[-1][:3]) / 255. for RSp in self._ray_source_plots]

            case _:
                assert_never(self.ui.coloring_mode)

        if len(self.raytracer.ray_sources) == len(self._ray_source_plots):
            for color, RSp in zip(RSColor, self._ray_source_plots):
                for RSpi in RSp[:2]:
                    if RSpi is not None:
                        RSpi.prop.color = tuple(color)
        else:
            warning("Number of RaySourcePlots differs from actual Sources. "
                    "Maybe the GUI was not updated properly?")

    def random_ray_selection(self):
        """
        select 'TraceGUI.rays_visible' rays from raytracer.rays.N random rays.
        Assigns the ScenePlotting.ray_selection boolean array.
        """
        # chose random elements
        N = self.raytracer.rays.N
        set_size = min(N, self.ui.rays_visible)
        rindex = np.random.choice(N, size=set_size, replace=False)  # random choice

        # make bool array with chosen rays set to true
        self.ray_selection = np.zeros(N, dtype=bool)
        self.ray_selection[rindex] = True

    def get_rays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """assign traced and selected rays into a ray dictionary"""
        
        p_, s_, pol_, w_, wl_, snum_, n_ = self.raytracer.rays.rays_by_mask(self.ray_selection, normalize=True)
        l_ = self.raytracer.rays.ray_lengths(self.ray_selection)
        ol_ = l_*n_  # in this situation faster than calling self.raytracer.rays.optical_lengths
        
        _, s_un, _, _, _, _, _ = self.raytracer.rays.rays_by_mask(self.ray_selection, normalize=False,
                                                                  ret=[0, 1, 0, 0, 0, 0, 0])

        # force copies
        self.__ray_property_dict.update(p=p_.copy(), s=s_.copy(), pol=pol_.copy(), w=w_.copy(), wl=wl_.copy(),
                                        snum=snum_.copy(), n=n_.copy(), index=np.where(self.ray_selection)[0],
                                        l=l_.copy(), ol=ol_.copy(), s_un=s_un.copy())
       
        # use flatten instead of ravel so we have guaranteed copies
        x, y, z = p_[:, :, 0].flatten(), p_[:, :, 1].flatten(), p_[:, :, 2].flatten()
        u, v, w = s_un[:, :, 0].flatten(), s_un[:, :, 1].flatten(), s_un[:, :, 2].flatten()
        s = np.ones_like(z)

        return x, y, z, u, v, w, s

    def assign_ray_properties(self) -> None:
        """safely copies the ray property dicts"""
        # set _ray_property_dict, that is used by other methods
        # other methods can't use __ray_property_dict, since this would require locks in the main thread
        self._ray_property_dict = copy.deepcopy(self.__ray_property_dict)
    
    def remove_rays(self) -> None:
        """remove ray properties and ray plot object"""
        self._ray_property_dict = {}

        if self._ray_plot is not None:
            self.scene.remove_actor(self._ray_plot, render=False)
            self._ray_plot = None

    def select_rays(self, mask: np.ndarray, max_show: int = None) -> None:
        """
        Apply a specific selection of rays for display.
        If the number is too large it is either limited by the 'max_show' parameter or a predefined limit.

        :param mask: boolean array for which rays to display. Shape must equal the number of currently traced rays.
        :param max_show: maximum number of rays to display
        """

        if mask.shape[0] != self.raytracer.rays.N:
            raise ValueError(f"Shape mismatch between mask ({mask.shape[0]}) "
                             f"and number of rays ({self.raytracer.rays.N}).")

        if mask.ndim != 1:
            raise ValueError(f"Mask must have only a single dimension, but has {mask.ndim}.")
        
        if max_show is not None and max_show <= 0:
            raise ValueError(f"Parameter 'max_show' must be above zero, but is {max_show}.")

        if mask.dtype != "bool":
            raise ValueError(f"Mask must be boolean, but is {mask.dtype}.")

        true_val = np.count_nonzero(mask)

        if not true_val:
            raise ValueError("No elements are set inside the mask.")

        if max_show is not None or true_val > self.MAX_RAYS_SHOWN:

            if max_show is None and true_val > self.MAX_RAYS_SHOWN:
                warning(f"Limited the number of displayed rays to {self.MAX_RAYS_SHOWN}, as the requested value "
                        f"of {true_val} was above the limit.")
                bound = self.MAX_RAYS_SHOWN
            
            elif max_show is not None and max_show > self.MAX_RAYS_SHOWN:
                warning(f"Limited the number of displayed rays to {self.MAX_RAYS_SHOWN}, as the requested value "
                        f"of {max_show} was above the limit.")
                bound = min(true_val, self.MAX_RAYS_SHOWN)

            else:
                bound = min(max_show, true_val)

            sub = np.zeros(true_val, dtype=bool)
            indices = np.random.choice(true_val, size=bound, replace=False)
            sub[indices] = True
            self.ray_selection = masked_assign(mask, sub)

        else:
            self.ray_selection = mask


    # Picking Handler
    ###################################################################################################################

    def _on_space_pick(self, picker_obj: vtkmodules.vtkRenderingCore.vtkPointPicker, event: str) -> None:
        """
        3D Space Clicking Handler. Shows Click Coordinates or moves Detector to this position when Shift is pressed.

        :param picker_obj:
        """
        pos = self.scene.picker.GetPickPosition()

        # differentiate between Click and Shift+Click
        if self.scene.interactor.shift_key:
            if self.raytracer.detectors:
                # set outside position to inside of outline
                pos_z = max(self.raytracer.outline[4], pos[2])
                pos_z = min(self.raytracer.outline[5], pos_z)

                self.ui.z_det = pos_z  # move detector
                self.reset_picking()
        else:
            r, ang = np.hypot(pos[0], pos[1]), np.rad2deg(np.arctan2(pos[1], pos[0]))
            self._ray_text.SetText(2, f"Pick Position (x, y, z):    ({pos[0]:>9.6g} mm, {pos[1]:>9.6g} mm, "\
                                  f"{pos[2]:>9.6g} mm)\n"\
                                  f"Relative to Axis (r, phi):  ({r:>9.6g} mm, {ang:>9.3f} °)")
            
            self.hide_ray_highlight()
            self.set_crosshair(pos)

    def _on_ray_pick(self, picker_obj: vtkmodules.vtkRenderingCore.vtkPointPicker, event: str) -> None:
        """Ray Picking Handler. Shows ray properties in the scene."""
        if self._ray_text is None:
            return

        if (b := self.scene.picker.GetPointId()) >= 0:

            # calculate ray index (i0) and section index (i1)
            a = self._ray_property_dict["p"].shape[1]  # number of points per ray plotted
            i0, i1 = np.divmod(b, 2*a)
            i1 = min(1 + (i1-1)//2, a - 1)

            self.pick_ray_section(i0, i1, self.scene.interactor.shift_key)

        else:
            self.reset_picking()
    
    def reset_picking(self) -> None:
        """hide ray text, ray highlight and crosshair""" 
        self.clear_ray_text()
        self.hide_ray_highlight()
        self.hide_crosshair()

    def pick_ray_section(self, 
                         index:     int, 
                         section:   int = None, 
                         detailed:  bool = False)\
            -> None:
        """
        From the ray index and section:
            
        1. Highlight the selected ray
        2. set the crosshair to the intersection position (if 'section' is not None)
        3. set ray info text (if 'section' is not None)

        :param index: is the index of the displayed rays.
        :param section: intersection index (starting position is zero)
        :param detailed: If 'detailed' is set, a more detailed ray info text is shown
        """
        if not len(self._ray_property_dict):
            raise RuntimeError("No rays available.")

        # count of displayed rays and sections
        N = self._ray_property_dict["p"].shape[0]
        Nt = self._ray_property_dict["p"].shape[1]

        if not ( 0 <= index < N):
            raise ValueError(f"Only {N} rays are displayed, {index} is an invalid index.")

        if section is not None and not (0 <= section < Nt):
            raise ValueError(f"Rays have only {Nt} sections, {section} is an invalid section number.")

        # with self.constant_camera():
        self.set_ray_highlight(index)
        
        if section is None:
            return
        
        i0, i1 = index, section

        # get properties of this ray section
        rp = self._ray_property_dict

        # choose surface of intersection. Surface undefined (=None) for last absorption at outline
        surfs = [self.raytracer.ray_sources[rp["snum"][i0]].front] + self.raytracer.tracing_surfaces + [None]
        surf = surfs[i1]

        # show surface properties?
        surf_props = isinstance(surf, Surface) and surf.mask(rp["p"][i0, i1, 0, None], rp["p"][i0, i1, 1, None])[0]

        # assign properties, nan means not defined/applicable
        p     = rp["p"][i0, i1]
        s     = rp["s"][i0, i1]                if i1 < len(surfs)-1  else [np.nan, np.nan, np.nan]
        pw    = rp["w"][i0, i1]
        pols  = rp["pol"][i0, i1]              if i1 < len(surfs)-1  else [np.nan, np.nan, np.nan]
        n     = rp["n"][i0, i1]                if i1 < len(surfs)-1  else np.nan
        wv    = rp["wl"][i0]
        snum  = rp["snum"][i0]
        index = rp["index"][i0]
        pw0   = rp["w"][i0, i1-1]              if i1                 else np.nan
        s0    = rp["s"][i0, i1-1]              if i1                 else [np.nan, np.nan, np.nan]
        pols0 = rp["pol"][i0, i1-1]            if i1                 else [np.nan, np.nan, np.nan]
        l0    = rp["l"][i0, i1 - 1]            if i1                 else np.nan
        l1    = rp["l"][i0, i1]                if i1 < len(surfs)-1  else np.nan
        ol0   = rp["ol"][i0, i1 - 1]           if i1                 else np.nan
        ol1   = rp["ol"][i0, i1]               if i1 < len(surfs)-1  else np.nan
        n0    = rp["n"][i0, i1-1]              if i1                 else np.nan
        pl    = (pw0-pw)/pw0                   if i1 and pw0         else np.nan
        normal = surf.normals(p[0, None], 
                              p[1, None])[0]   if surf_props         else [np.nan, np.nan, np.nan]

        def to_sph_coords(s):
            theta = np.rad2deg(np.arccos(s[2]))
            phi = np.rad2deg(np.arctan2(s[1], s[0]))
            return np.array([theta, phi])

        # coordinates in spherical coordinates
        s_sph = to_sph_coords(s)
        s0_sph = to_sph_coords(s0)
        pols_sph = to_sph_coords(pols)
        pols0_sph = to_sph_coords(pols0)
        normal_sph = to_sph_coords(normal)

        # differentiate between Click and Shift+Click
        if detailed:

            text =  f"Ray {index} from RS{snum} "
            text += f"at surface {i1}\n\n" if i1 else "at ray source\n\n"

            text += f"Intersection Position: ({p[0]:>10.6g} mm, {p[1]:>10.6g} mm, {p[2]:>10.6g} mm)\n\n"

            text += f"Vectors:                        Cartesian (x, y, z)                   "\
                     "Spherical (theta, phi)\n"
            text += f"Direction Before:      ({s0[0]:>10.5f}, {s0[1]:>10.5f}, {s0[2]:>10.5f})"\
                    f"         ({s0_sph[0]:>10.5f}°, {s0_sph[1]:>10.5f}°)\n"
            text += f"Direction After:       ({s[0]:>10.5f}, {s[1]:>10.5f}, {s[2]:>10.5f})"\
                    f"         ({s_sph[0]:>10.5f}°, {s_sph[1]:>10.5f}°)\n"
            text += f"Polarization Before:   ({pols0[0]:>10.5f}, {pols0[1]:>10.5f}, {pols0[2]:>10.5f})"\
                    f"         ({pols0_sph[0]:>10.5f}°, {pols0_sph[1]:>10.5f}°)\n"
            text += f"Polarization After:    ({pols[0]:>10.5f}, {pols[1]:>10.5f}, {pols[2]:>10.5f})"\
                    f"         ({pols_sph[0]:>10.5f}°, {pols_sph[1]:>10.5f}°)\n"
            text += f"Surface Normal:        ({normal[0]:>10.5f}, {normal[1]:>10.5f}, {normal[2]:>10.5f})"\
                    f"         ({normal_sph[0]:>10.5f}°, {normal_sph[1]:>10.5f}°)\n\n"

            text += f"Wavelength:               {wv:>10.2f} nm\n"
            text += f"Refraction Index Before:  {n0:>10.4f}"\
                    f"           Distance to Last Intersection:          {l0:>10.5g} mm\n"
            text += f"Refraction Index After:   {n:>10.4f}"\
                    f"           Distance to Next Intersection:          {l1:>10.5g} mm\n"
            text += f"Ray Power Before:         {pw0*1e6:>10.5g} µW"\
                    f"        Optical Distance to Last Intersection:  {ol0:>10.5g} mm\n"
            text += f"Ray Power After:          {pw*1e6:>10.5g} µW"\
                    f"        Optical Distance to Next Intersection:  {ol1:>10.5g} mm\n"
            text += f"Power Loss on Surface:    {pl*100:>10.5g} %\n\n"

            if surf_props:
                text += "Surface Information:\n" + surf.info

        else:
            text =  f"Ray {index} from Source {snum} "
            text += f"at surface {i1}\n" if i1 else "at ray source\n"
            text += f"Intersection Position: ({p[0]:>10.5g} mm, {p[1]:>10.5g} mm, {p[2]:>10.5g} mm)\n"
            text += f"Direction After:       ({s[0]:>10.5f},    {s[1]:>10.5f},    {s[2]:>10.5f}   )\n"
            text += f"Polarization After:    ({pols[0]:>10.5f},    {pols[1]:>10.5f},    {pols[2]:>10.5f}   )\n"
            text += f"Wavelength:             {wv:>10.2f} nm\n"
            text += f"Ray Power After:        {pw*1e6:>10.5g} µW\n"
            text += f"Pick using Shift+Left Mouse Button for more info"

        # apply text
        self._ray_text.SetText(2, text.replace("nan", " - "))
        self.apply_prop(self._ray_text.prop, **self.INFO_STYLE)
        self.apply_prop(self._ray_text.prop, background_opacity=self._info_opacity,
                        opacity=1, color=self._foreground_color)  # these parts required?

        self.set_crosshair(p)
