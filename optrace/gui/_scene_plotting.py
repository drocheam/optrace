from __future__ import annotations
from contextlib import contextmanager  # context manager for _no_trait_action()

import numpy as np  # calculations
import copy
import matplotlib.pyplot as plt 
import mayavi.modules.text  # Text type
from mayavi.sources.parametric_surface import ParametricSurface  # provides outline and axes

from ..tracer.geometry import Surface, Element
from ..tracer import *
from ..warnings import warning
from ..global_options import global_options as go

from ..property_checker import PropertyChecker as pc
from ..tracer.misc import masked_assign



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
        :param initial_camera: keyword dictionary for set_camer()
        """
        self.ui = ui
        self._scene_size = np.array(ui._scene_size0)
        self._scene_size0 = self._scene_size.copy()
        self.scene = ui.scene
        self.raytracer = raytracer

        # plot object lists
        self._lens_plots = []
        self._axis_plots = []
        self._filter_plots = []
        self._aperture_plots = []
        self._detector_plots = []
        self._ray_source_plots = []
        self._marker_plots = []
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
        
        # pickers
        self._ray_picker = None
        self._space_picker = None

        # assign scene background color
        self.set_colors()

    # Helper Functions
    ###################################################################################################################

    def __remove_objects(self, objs: list) -> None:
        """remove visual objects from raytracer geometry"""

        # try to delete objects in pipeline while iterating over its parents and grandparents
        for obj in objs:
            for obji in obj[:4]:
                if obji is not None:
                    for i in range(4):
                        if obji.parent in self.scene.mayavi_scene.children:
                            obji.parent.remove()
                        obji = obji.parent

        objs[:] = []
    
    @contextmanager
    def constant_camera(self, *args, **kwargs) -> None:
        """context manager the saves and restores the camera view"""
        cc_traits_org = {}

        if self.scene is not None and self.scene.camera is not None:
            cc_traits_org = self.scene.camera.trait_get("position", "focal_point", "view_up", "view_angle",
                                                        "clipping_range", "parallel_scale")
        try:
            yield

        finally:
            if self.scene is not None and self.scene.camera is not None:
                self.scene.camera.trait_set(**cc_traits_org)
    
    def screenshot(self, path: str, **kwargs) -> None:
        """
        Save a screenshot of the scene. Passes the parameters down to the mlab.savefig function.
        See `https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig` for parameters.
        """
        self._status_text.visible = False  # temporarily hide status text so it won't be on the screenshot
        self.scene.mlab.savefig(path, **kwargs)
        self._status_text.visible = True

    def get_camera(self) -> tuple[np.ndarray, float, np.ndarray, float]:
        """
        Get the camera parameters that can be passed down to set_camera()

        :return: Return the current camera parameters (center, height, direction, roll)
        """
        cam = self.scene.scene.camera
        normal = (cam.focal_point - cam.position) / cam.distance 
    
        return cam.focal_point,\
               cam.parallel_scale,\
               normal,\
               self.scene.mlab.roll()\

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
        cam = self.scene.scene.camera
        normal = np.asarray(direction, dtype=np.float64) if direction is not None\
                else (cam.focal_point - cam.position) / cam.distance
        dist_vec = cam.distance * normal / (np.linalg.norm(normal) + 1e-12) 

        # old cross product of direction and view_up
        rev = np.cross(cam.focal_point - cam.position, cam.view_up)

        if center is not None:
            cam.focal_point = center

        if height is not None:
            cam.parallel_scale = height

        # set/update camera position (since either/both focal_point or direction changed)
        cam.position = cam.focal_point - dist_vec

        # update view_up from new normal and old cross product of direction and view_up 
        cam.view_up = np.cross(cam.focal_point - cam.position, -rev)

        # absolute roll angle
        if roll is not None:
            self.scene.mlab.roll(roll)
        
        # render
        cam.compute_view_plane_normal()
        self.scene.scene._renderer.reset_camera_clipping_range()
        self.scene.scene.render()


    def set_initial_camera(self) -> None:
        """
        sets the initial camera view

        When parameter initial_camera is not set, it is a y-side view with all elements inside the viewable range
        When it is set, the camera properties are applied.
        """
        self.scene.parallel_projection = True
        self.scene._def_pos = 1  # for some reason it is set to None
        self.scene.y_plus_view()
        self.scene.scene_editor.camera.parallel_scale *= 0.85
        
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

            t = self.plot_element(F, num, fcolor[:3] if not self.ui.high_contrast else self.scene.foreground, alpha)
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
            t = self.plot_element(RS, num, (1, 1, 1), self._raysource_alpha, spec=False, light=False)
            self._ray_source_plots.append(t)

    def plot_outline(self) -> None:
        """replot the raytracer outline"""

        if self._outline is not None:
            self._outline.remove()

        self.scene.engine.add_source(ParametricSurface(name="Outline"), self.scene)
        self._outline = self.scene.mlab.outline(extent=self.raytracer.outline.copy(), color=self._outline_color)
        self._outline.actor.actor.pickable = False  # only rays should be pickable

    def plot_orientation_axes(self) -> None:
        """plot orientation axes"""

        if self._orientation_axes is not None:
            self._orientation_axes.remove()

        self.scene.engine.add_source(ParametricSurface(name="Orientation Axes"), self.scene)

        # show axes indicator
        self._orientation_axes = self.scene.mlab.orientation_axes()
        self._orientation_axes.text_property.trait_set(**self.LABEL_STYLE)
        self._orientation_axes.marker.interactive = 0  # make orientation axes non-interactive
        self._orientation_axes.widgets[0].viewport = [0, 0, 0.1, 0.15]
        self._orientation_axes.axes.visibility = not bool(self.ui.minimalistic_view)

        # turn of text scaling of orientation_axes
        mark = self._orientation_axes.widgets[0].orientation_marker
        mark.x_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        mark.y_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        mark.z_axis_caption_actor2d.text_actor.text_scale_mode = 'none'

    def plot_axes(self) -> None:
        """plot cartesian axes"""

        # save old font factor. This is the one we adapted constantly in self._resizeSceneElements()
        ff_old = self._axis_plots[0][0].axes.font_factor if self._axis_plots else 0.75

        self.__remove_objects(self._axis_plots)

        # find label number for axis so that step size is an int*10^k
        # or any of [0.25, 0.5, 0.75, 1.25, 2.5]*10^k with k being an integer
        # label number needs to be in range [min_s, max_s]
        def get_label_num(num: float, min_s: int, max_s: int) -> int:

            norm = 10 ** -np.floor(np.log10(num)-1)
            num_norm = num*norm  # normalize num so that 10 <= num < 100

            for i in np.arange(max_s, min_s-1, -1):
                if num_norm/i in [0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 1,
                                  1.25, 1.5, 2, 2.5, 3, 3.75, 4, 5, 6, 6.25, 7.5, 8, 10]:
                    return i+1  # increment since there are n+1 labels for n steps

            return max_s+1

        def draw_axis(objs, ext: list, lnum: int, name: str, lform: str,
                      vis_x: bool, vis_y: bool, vis_z: bool):

            self.scene.engine.add_source(ParametricSurface(name=f"{name}-Axis"), self.scene)

            a = self.scene.mlab.axes(extent=ext, nb_labels=lnum, x_axis_visibility=vis_x,
                                     y_axis_visibility=vis_y, z_axis_visibility=vis_z, color=self._outline_color)

            label = f"{name} / mm"
            a.axes.trait_set(font_factor=ff_old, fly_mode='none', label_format=lform, x_label=label,
                             y_label=label, z_label=label, layer_number=1)

            a.title_text_property.trait_set(**self.LABEL_STYLE, color=self._axis_color, opacity=self._axis_alpha)
            a.label_text_property.trait_set(**self.LABEL_STYLE, color=self._axis_color, opacity=self._axis_alpha)
            a.visible = not bool(self.ui.minimalistic_view)
            a.actors[0].pickable = False

            objs.append((a,))

        # place axes at outline
        ext = self.raytracer.outline

        # enforce placement of x- and z-axis at y=ys (=RT.outline[2]) by shrinking extent
        ext_ys = ext.copy()
        ext_ys[3] = ext_ys[2]

        # X-Axis
        lnum = get_label_num(ext[1] - ext[0], 5, 16)
        draw_axis(self._axis_plots, ext_ys, lnum, "x", '%-#.4g', True, False, False)

        # Y-Axis
        lnum = get_label_num(ext[3] - ext[2], 5, 16)
        draw_axis(self._axis_plots, ext.copy(), lnum, "y", '%-#.4g', False, True, False)

        # Z-Axis
        lnum = get_label_num(ext[5] - ext[4], 5, 24)
        draw_axis(self._axis_plots, ext_ys, lnum, "z", '%-#.5g', False, False, True)

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
            self.scene.engine.add_source(ParametricSurface(name=f"Refraction Index Outline {i}"), self.scene)
            outline = self.scene.mlab.outline(extent=[*self.raytracer.outline[:4], BoundList[i][1], BoundList[i + 1][0]],
                                              color=self._outline_color)
            outline.outline_mode = 'cornered'
            outline.actor.actor.pickable = False  # only rays should be pickable
            outline.outline_filter.corner_factor = 0.1

            # label position
            y_pos = self.raytracer.outline[2] + (self.raytracer.outline[3] - self.raytracer.outline[2]) * 0.05
            x_pos = np.mean(self.raytracer.outline[:2])
            z_pos = np.mean([BoundList[i][1], BoundList[i+1][0]])

            # plot label
            label = (f"ambient\n" if not self.ui.minimalistic_view else "")  + "n=" + nList[i].get_desc()
            text = self.scene.mlab.text(x_pos, y_pos, z=z_pos, text=label, name="Label")
            text.actor.text_scale_mode = 'none'
            text.visible = not bool(self.ui.hide_labels)
            text.property.trait_set(**self.LABEL_STYLE, frame=True, frame_color=self._subtle_color, 
                                    color=self._axis_color, opacity=self._axis_alpha)

            if not self.ui.vertical_labels:
                text.property.trait_set(justification="center", bold=False)
            else:
                text.property.trait_set(justification="left", orientation=90, bold=False,
                                        vertical_justification="center")

            # append plot objects
            self._index_box_plots.append((outline, text, None))

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
            # plot surface
            a = self.scene.mlab.mesh(C[0], C[1], C[2], color=color, opacity=alpha,
                                     name=f"{type(obj).__name__} {num} {surf_type} surface")

            # make non-pickable, so it does not interfere with our ray picker
            a.actor.actor.pickable = False
            a.actor.property.representation = "surface"

            a.actor.actor.property.lighting = light
            if spec:
                a.actor.property.trait_set(specular=0.5, ambient=0.25)
                if self.ui.high_contrast:
                    a.actor.property.trait_set(specular_color=(0.15, 0.15, 0.15),
                                               diffuse_color=(0.12, 0.12, 0.12))
                a.actor.property.color = color  # reassign color

            return a

        # decide what to plot
        plotFront = isinstance(obj.front, Surface)
        plotCyl = plotFront
        plotBack = obj.back is not None and isinstance(obj.back, Surface)

        # Element consisting only of Point or Line: nothing plotted except label -> add parent object
        if not (plotFront or plotCyl or plotBack):
            self.scene.engine.add_source(ParametricSurface(name=f"{type(obj).__name__} {num}"), self.scene)

        # plot front
        nres = self.SURFACE_RES if not plotFront or not isinstance(obj.front, RectangularSurface) else 10
        a = plot(obj.front.plotting_mesh(N=nres), "front") if plotFront else None

        # cylinder between surfaces
        b = plot(obj.cylinder_surface(nc=2 * self.SURFACE_RES), "cylinder") if plotCyl else None

        # adapt cylinder opacity
        if plotCyl and isinstance(obj, Lens):
            b.actor.property.trait_set(opacity=self._cylinder_opacity)

        # use wireframe mode, so edge is always visible, even if it is infinitesimal small
        if plotCyl and (not obj.has_back() or (obj.extent[5] - obj.extent[4] < 0.05)):
            b.actor.property.trait_set(representation="wireframe", lighting=False, line_width=1.75, opacity=alpha/1.5)

        # calculate middle center z-position
        if obj.has_back():
            zl = (obj.front.values(np.array([obj.front.pos[0]]), np.array([obj.front.extent[3]]))[0] \
                  + obj.back.values(np.array([obj.back.pos[0]]), np.array([obj.back.extent[3]]))[0]) / 2
        else:
            # 0/40 values in [0, 2pi] -> z - value at 0 deg relative to x axis in xy plane
            zl = obj.front.edge(40)[2][0] if isinstance(obj.front, Surface) else obj.pos[2]

        if not no_label:
            # object label
            label = f"{obj.abbr}{num}"

            # add description if any exists. But only if we are not in "minimalistic_view" displaying mode
            label = label if obj.desc == "" or bool(self.ui.minimalistic_view) else label + ": " + obj.desc
            text = self.scene.mlab.text(x=obj.pos[0], y=obj.extent[3], z=zl, text=label, name="Label")
            text.actor.text_scale_mode = 'none'
            text.visible = not bool(self.ui.hide_labels)

            if not self.ui.vertical_labels:
                text.property.trait_set(**self.LABEL_STYLE, justification="center", vertical_justification="bottom", 
                                        orientation=0, background_opacity=0.5, background_color=self.scene.background)
            else:
                text.property.trait_set(**self.LABEL_STYLE, justification="left", orientation=90, 
                                        vertical_justification="center", background_opacity=0.5, 
                                        background_color=self.scene.background)
        else:
            text = None

        # plot BackSurface if one exists
        c = plot(obj.back.plotting_mesh(N=self.SURFACE_RES), "back") if plotBack else None

        return a, b, c, text, obj

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
            self._ray_plot.parent.parent.remove()

        self._ray_plot = self.scene.mlab.quiver3d(x, y, z, u, v, w, scalars=s,
                                                  scale_mode="vector", scale_factor=1, colormap="Greys", mode="2ddash")

        self._ray_plot.actor.actor.property.trait_set(lighting=False, render_points_as_spheres=True,
                                                      opacity=self.ui.ray_opacity)

        self._ray_plot.actor.property.trait_set(line_width=self.ui.ray_width, point_size=self.ui.ray_width)

        self._ray_plot.glyph.color_mode = "color_by_scalar"
        self._ray_plot.parent.parent.name = "Rays"
        self._ray_plot.actor.property.representation = 'points' if self.ui.plotting_mode == 'Points' else 'surface'

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
            self._ray_highlight_plot.parent.parent.remove()

        self._ray_highlight_plot = self.scene.mlab.quiver3d(x, y, z, u, v, w,
                                                            scale_mode="vector", scale_factor=1, 
                                                            colormap="Reds", mode="2ddash", line_width=2)
        
        self._ray_highlight_plot.actor.actor.pickable = False  # only rays should be pickable
        self._ray_highlight_plot.actor.property.color = self._crosshair_color
        self._ray_highlight_plot.actor.actor.property.trait_set(lighting=False, render_points_as_spheres=True)

        self._ray_highlight_plot.glyph.color_mode = "color_by_scalar"
        self._ray_highlight_plot.parent.parent.name = "Ray Highlight"
        self._ray_highlight_plot.visible = True

    def plot_point_markers(self) -> None:
        """plot point markers inside the scene"""
        self.__remove_objects(self._marker_plots)

        for num, mark in enumerate(self.raytracer.markers):
            if isinstance(mark, PointMarker):

                dy, dx = 0.2 * mark.marker_factor, 0

                m = self.scene.mlab.points3d(*mark.pos, mode="axes", color=self._marker_color)
                m.actor.actor.property.trait_set(lighting=False, line_width=5, representation="wireframe")

                m.visible = not mark.label_only
                m.glyph.glyph.scale_factor = dy
                m.parent.parent.name = f"Marker {num}"
                m.actor.actor.trait_set(pickable=False, force_translucent=True)
            
                text = self.scene.mlab.text(x=mark.pos[0]+dx, y=mark.pos[1]+dy, z=mark.pos[2], 
                                            text=mark.desc, name="Label")

                tprop = dict(justification="center") if not self.ui.vertical_labels\
                        else dict(justification="left", orientation=90, vertical_justification="center")

                text.property.trait_set(**self.LABEL_STYLE, background_opacity=0.5, 
                                        background_color=self.scene.background, **tprop)
                text.property.font_size = int(8.4 * mark.text_factor)
                text.actor.text_scale_mode = 'none'
                text.visible = not bool(self.ui.hide_labels)

                self._marker_plots.append((m, None, None, text, mark))
    
    def plot_line_markers(self) -> None:
        """plot line markers inside the scene"""
        self.__remove_objects(self._line_marker_plots)

        for num, mark in enumerate(self.raytracer.markers):
            if isinstance(mark, LineMarker):

                drx = mark.front.r * np.cos(np.radians(mark.front.angle))
                dry = mark.front.r * np.sin(np.radians(mark.front.angle))

                dx, dy = mark.pos[0]+drx, mark.pos[1]+dry
                m = self.scene.mlab.plot3d([mark.pos[0]-drx, mark.pos[0]+drx], [mark.pos[1]-dry, mark.pos[1]+dry],
                                           [mark.pos[2], mark.pos[2]], tube_radius=0, color=self._line_marker_color)

                m.parent.parent.parent.parent.name = f"Marker {num}"
                m.actor.actor.trait_set(pickable=False, force_translucent=True)
                m.actor.actor.property.trait_set(lighting=False, line_width=mark.line_factor,
                                                 representation="wireframe")
            
                text = self.scene.mlab.text(x=mark.pos[0]+dx, y=mark.pos[1]+dy, z=mark.pos[2], 
                                            text=mark.desc, name="Label")

                tprop = dict(justification="center") if not self.ui.vertical_labels\
                        else dict(justification="left", orientation=90, vertical_justification="center")

                text.property.trait_set(**self.LABEL_STYLE, background_opacity=0.5, 
                                        background_color=self.scene.background, **tprop)
                text.property.font_size = int(8.4 * mark.text_factor)
                text.actor.text_scale_mode = 'none'
                text.visible = not bool(self.ui.hide_labels)

                self._line_marker_plots.append((m, None, None, text, mark))

    def plot_volumes(self) -> None:
        """plot volumes inside the scene"""
        self.__remove_objects(self._volume_plots)

        for num, V in enumerate(self.raytracer.volumes):
            color = self._volume_color if (V.color is None or self.ui.high_contrast) else V.color
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
        self._aperture_color =      (0, 0, 0)
        self._crosshair_color =     (1, 0, 0)
        self.scene.background =     (0.2, 0.2, 0.2)      if not high_contrast else (1, 1, 1)
        self.scene.foreground =     (1, 1, 1)            if not high_contrast else (0, 0, 0)
        self._lens_color =          (0.63, 0.79, 1.00)   if not high_contrast else self.scene.foreground
        self._detector_color =      (0.0, 0.0, 0.0)      if not high_contrast else self.scene.foreground
        self._subtle_color =        (0.3, 0.3, 0.3)      if not high_contrast else (0.7, 0.7, 0.7)
        self._marker_color =        (0, 1, 0)            if not high_contrast else self.scene.foreground
        self._line_marker_color =   (0.8, 0, 0.8)        if not high_contrast else self.scene.foreground
        self._outline_color =       (0.5, 0.5, 0.5)      if not high_contrast else (0.8, 0.8, 0.8)
        self._axis_color =          (1, 1, 1)            if not high_contrast else (0.5, 0.5, 0.5)
        self._info_frame_color =    (0, 0, 0)            if not high_contrast else (1, 1, 1)
        self._volume_color =        (0.45, 0.45, 0.45)   if not high_contrast else (1, 1, 1)
        self._axis_alpha =          0.55                 if not high_contrast else 0.35
        self._cylinder_opacity =    self._lens_alpha     if not high_contrast else 0.6

    def init_keyboard_shortcuts(self) -> None:
        """init keyboard shortcut detection inside the scene"""

        # also see already available shortcuts
        # https://docs.enthought.com/mayavi/mayavi/application.html#keyboard-interaction

        def keyrelease(vtk_obj, event):

            match vtk_obj.GetKeyCode():

                # it seems like pressing "m" in the scene does something,
                # although i can't find any documentation on this
                # a side effect is deactivating the mouse pickers, to reactivate we need to press "m" another time
                case "m":
                    warning("Avoid pressing 'm' in the scene because it interferes with mouse picking handlers.")
                
                case "a":
                    warning("Avoid pressing 'a' as it is a mayavi shortcut for actor mode, where rays can be moved.")

                case "i":  # reset view
                    self.set_initial_camera()

                case "h":  # hide/show side menu and toolbar
                    self.ui.maximize_scene = bool(self.ui._scene_not_maximized)  # toggle value

                case "v":  # toggle minimalistic_view
                    self.ui.minimalistic_view = not bool(self.ui.minimalistic_view)
                
                case "c":  # high_contrast
                    self.ui.high_contrast = not bool(self.ui.high_contrast)

                case "b":  # toggle label visibility
                    self.ui.hide_labels = not bool(self.ui.hide_labels)
                
                case "d":  # render DetectorImage
                    self.ui.detector_image()
               
                # q does not seem to be registered anymore. Is it already used by vtk for something? 
                # Use 0 as key instead
                case "0":  # close all pyplots
                    plt.close("all")

                case "n":  # reselect and replot rays
                    self.scene.disable_render = True
                    self.ui.replot_rays()
                    self.scene.disable_render = False

                # Unfortunately this does not work as special keys don't seem to be correctly checked
                # after pressing F11 then Esc also triggers this
                # case "" if vtk_obj.GetKeySym() == "F11":
                    # self.scene.scene_editor._full_screen_fired()
        
        self.scene.interactor.add_observer('KeyPressEvent', keyrelease)  # calls keyrelease() in main thread

    def init_crosshair(self) -> None:
        """init a crosshair for the picker"""

        self.scene.engine.add_source(ParametricSurface(name="Crosshair"), self.scene)

        self._crosshair = self.scene.mlab.text(x=0, y=0, z=0, text="+", name="Label")
        self._crosshair.actor.text_scale_mode = 'none'
        self._crosshair.visible = False
        self._crosshair.property.trait_set(font_size=32, justification="center", vertical_justification="center", 
                                           orientation=0, bold=True, font_family="times", 
                                           color=self._crosshair_color, use_tight_bounding_box=True)

    def init_ray_info(self) -> None:
        """init detection of ray point clicks and the info display"""

        self._ray_picker = self.scene.mlab.gcf().on_mouse_pick(self._on_ray_pick, button='Left')
        self._ray_picker.tolerance = 0.0025  # it seems tolerance can only be set for all pickers at once

        # add ray info text
        self.scene.engine.add_source(ParametricSurface(name="Ray Info Text"), self.scene)
        self._ray_text = self.scene.mlab.text(0.02, 0.97, "")
        self._ray_text.actor.text_scale_mode = 'none'
        self._ray_text.property.trait_set(**self.INFO_STYLE, background_opacity=self._info_opacity,
                                          opacity=1, color=self.scene.foreground, vertical_justification="top",
                                          background_color=self._info_frame_color)

    def init_status_info(self) -> None:
        """init GUI status text display"""

        self._space_picker = self.scene.mlab.gcf().on_mouse_pick(self._on_space_pick, button='Right')
        self._space_picker.tolerance = 0.0025  # it seems tolerance can only be set for all pickers at once

        # add status text
        self.scene.engine.add_source(ParametricSurface(name="Status Info Text"), self.scene)
        self._status_text = self.scene.mlab.text(0.97, 0.01, "")
        self._status_text.actor.text_scale_mode = 'none'
        self._status_text.property.trait_set(**self.INFO_STYLE, justification="right")

    # Scene Changes
    ###################################################################################################################

    def change_label_orientation(self) -> None:
        """Set labels of Elements to vertical or horizontal depending on option vertical_labels"""
       
        opts = dict(justification="center", vertical_justification="bottom", orientation=0) \
            if not self.ui.vertical_labels\
            else dict(justification="left", orientation=90, vertical_justification="center")

        for objs in [self._lens_plots, self._detector_plots, self._aperture_plots, self._filter_plots,
                     self._marker_plots, self._line_marker_plots, self._index_box_plots, self._ray_source_plots]:

            for obj in objs:
                for obji in obj:
                    if isinstance(obji, mayavi.modules.text.Text):
                        obji.property.trait_set(**opts)

    def change_minimalistic_view(self) -> None:
        """Hide long labels, orientation axes and normal axes depending on if option minimalistic_view is set"""

        show = not bool(self.ui.minimalistic_view)

        if self._orientation_axes is not None:
            self._orientation_axes.axes.visibility = show

        for rio in self._index_box_plots:
            if rio[1] is not None:
                rio[1].text = rio[1].text.replace("ambient\n", "") if not show else ("ambient\n" + rio[1].text)

        # remove descriptions from labels in minimalistic_view
        for Objects in [self._ray_source_plots, self._lens_plots, self._volume_plots, 
                        self._filter_plots, self._aperture_plots, self._detector_plots]:
            for num, obj in enumerate(Objects):
                if obj[3] is not None and obj[4] is not None:
                    label = f"{obj[4].abbr}{num}"
                    label = label if obj[4].desc == "" or not show else label + ": " + obj[4].desc
                    obj[3].text = label

        for ax in self._axis_plots:
            if ax[0] is not None:
                ax[0].visible = show
           

    def change_label_visibility(self) -> None:
        """Hide/show labels depending on value of option 'hide_labels'"""
        
        show = not bool(self.ui.hide_labels)

        for rio in self._index_box_plots:
            if rio[1] is not None:
                rio[1].visible = show
        
        # remove descriptions from labels in minimalistic_view
        for Objects in [self._ray_source_plots, self._lens_plots, self._marker_plots, self._volume_plots, 
                        self._line_marker_plots, self._filter_plots, self._aperture_plots, self._detector_plots]:
            for num, obj in enumerate(Objects):
                if obj[3] is not None:
                    obj[3].visible = show


    def change_contrast(self) -> None:
        """Normal or high contrast scene mode depending on state of option 'high_contrast'"""

        self.set_colors()
        high_contrast = self.ui.high_contrast

        # update filter color
        for F in self._filter_plots:
            for Fi in F[:2]:
                if Fi is not None and F[4] is not None:
                    Fi.actor.property.color = F[4].color()[:3] if not high_contrast else self.scene.foreground
        
        # update volume colors
        for V in self._volume_plots:
            for Vi in V[:3]:
                Vi.actor.property.color = self._volume_color if V[4].color is None or high_contrast else V[4].color

        # update misc color
        for color, objs in zip([self._lens_color, self._detector_color, self._aperture_color, self._marker_color,
                                self._line_marker_color, self._outline_color],
                               [self._lens_plots, self._detector_plots, self._aperture_plots,
                                self._marker_plots, self._line_marker_plots, [[self._outline]]]):
            for obj in objs:
                for el in obj[:3]:
                    if el is not None:
                        el.actor.property.trait_set(color=color)
                        if high_contrast and el is not self._outline:
                            el.actor.property.trait_set(specular_color=(0.15, 0.15, 0.15),
                                                        diffuse_color=(0.12, 0.12, 0.12))

        # update background colors of labels
        for objs in [self._lens_plots, self._detector_plots, self._aperture_plots, self._filter_plots, 
                     self._marker_plots, self._line_marker_plots, self._volume_plots, self._ray_source_plots]:
            for obj in objs:
                if len(obj) > 3 and obj[3] is not None:
                    obj[3].property.background_color = self.scene.background

        if self._crosshair is not None:
            self._crosshair.property.color = self._crosshair_color

        # change lens cylinder visibility
        for lens in self._lens_plots:
            if lens[1] is not None:
                lens[1].actor.property.trait_set(opacity=self._cylinder_opacity)

        # update axes color
        for ax in self._axis_plots:
            for el in ax:
                if el is not None:
                    el.axes.property.color = self._outline_color
                    el.label_text_property.opacity = self._axis_alpha
                    el.title_text_property.opacity = self._axis_alpha

        # change index plot objects
        for obj in self._index_box_plots:

            if obj[1] is not None:
                obj[1].property.color = self._axis_color
                obj[1].property.frame_color = self._subtle_color

            if obj[0] is not None:
                obj[0].actor.property.color = self._outline_color

        # reassign ray source colors
        self.color_ray_sources()

        if self._ray_highlight_plot is not None:
            self._ray_highlight_plot.actor.property.color = self._crosshair_color

        # in coloring type Plain the ray color is changed from white to a bright orange
        if self.ui.coloring_mode == "Plain" and self._ray_plot is not None:
            self._ray_plot.parent.scalar_lut_manager.lut_mode = "Greys" if not high_contrast else "Wistia"
            self._ray_plot.parent.scalar_lut_manager.reverse_lut = bool(high_contrast)

        if self._ray_text is not None:
            self._ray_text.property.background_color = self._info_frame_color
            
    def resize_scene_elements(self) -> None:
        """
        When resizing the scene window, some object sizes change. This function compensates for these changes.
        It is called from within the TraceGUI
        """

        if self.scene._renwin is not None and self.scene.scene_editor._interactor is not None:
            scene_size = self.scene.scene_editor._interactor.size

            if scene_size[0]*scene_size[1] > 0 and\
                (self._scene_size[0] != scene_size[0] or self._scene_size[1] != scene_size[1]):

                # average of x and y size for former and current scene size
                ch1 = (self._scene_size[0] + self._scene_size[1]) / 2
                ch2 = (scene_size[0] + scene_size[1]) / 2

                # update font factor so font size stays the same
                for ax in self._axis_plots:
                    ax[0].axes.font_factor *= ch1/ch2

                # rescale orientation axes
                if self._orientation_axes is not None:
                    self._orientation_axes.widgets[0].zoom *= ch1 / ch2

                if self._ray_plot is not None:
                    bar = self._ray_plot.parent.scalar_lut_manager.scalar_bar_representation
                    bar.position2 = bar.position2 * self._scene_size / scene_size
                    bar.position = [bar.position[0], (1-bar.position2[1])/2]

                # set current window size
                self._scene_size = scene_size.copy()

    def set_ray_opacity(self) -> None:
        """change the ray opacity"""
        if self._ray_plot is not None:
            self._ray_plot.actor.property.opacity = self.ui.ray_opacity
    
    def set_ray_representation(self) -> None:
        """change the ray representation between 'points' and 'surface'"""
        if self._ray_plot is not None:
            self._ray_plot.actor.property.representation = 'points' if self.ui.plotting_mode == 'Points' else 'surface'

    def set_ray_width(self) -> None:
        """change the ray width"""
        if self._ray_plot is not None:
            self._ray_plot.actor.property.trait_set(line_width=self.ui.ray_width, point_size=self.ui.ray_width)

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
            self._status_text.text = ""

            for key, val in msgs.items():
                if _status[key]:
                    self._status_text.text += msgs[key] + "...\n"

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
        move the detector plot with index 'ind' differentially

        :param ind: detector index
        :param movement: moving distance
        """
        if ind < len(self._detector_plots):
            # change relative actor position without touching the underlying mlab_source array
            # some time ago we change the data with mlab_source.z, but this lead to issues
            # for flat surfaces that weren't moved at all for some reason
            det = self._detector_plots[ind]
            # v--- don't use += here, as changes won't be correctly detected
            det[0].actor.actor.position = det[0].actor.actor.position + [0, 0, diff]
            det[1].actor.actor.position = det[1].actor.actor.position + [0, 0, diff]
            self._detector_plots[ind][3].z_position += diff

    def clear_ray_text(self) -> None:
        """clear the ray info text"""
        self._ray_text.text = ""

    def hide_crosshair(self) -> None:
        """Hide the crosshair if it still/already exists"""
        if self._crosshair is not None:
            self._crosshair.visible = False

    def hide_ray_highlight(self) -> None:
        """Hide the ray highlight plot if it still/already exists"""
        if self._ray_highlight_plot is not None:
            self._ray_highlight_plot.visible = False

    def set_crosshair(self, pos: np.ndarray) -> None:
        """
        Set the crosshair position if it still/already exists
        
        :param pos: array with three elements (x, y, z)
        """
        if self._crosshair is not None:
            self._crosshair.trait_set(x_position=pos[0], y_position=pos[1], 
                                      z_position=pos[2], visible=True)

    # Ray and RaySource plotting
    ###################################################################################################################

    def color_rays(self) -> None:
        """color the plotted rays in the scene depending on the coloring mode"""

        if self._ray_plot is None:
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

        self._ray_plot.mlab_source.trait_set(scalars=s)

        # lut legend settings
        lutm = self._ray_plot.parent.scalar_lut_manager
        lutm.trait_set(use_default_range=True, show_scalar_bar=True, use_default_name=False,
                       show_legend=self.ui.coloring_mode != "Plain", lut_mode=cm, reverse_lut=False)

        # lut visibility and title
        lutm.scalar_bar.trait_set(title=title, unconstrained_font_size=True)
        lutm.label_text_property.trait_set(**self.LABEL_STYLE)
        lutm.title_text_property.trait_set(**self.INFO_STYLE)

        # lut position and size
        hr, vr = tuple(self._scene_size0/self._scene_size)  # horizontal and vertical size ratio
        lutm.scalar_bar_representation.position = np.array([0.92, (1-0.6*vr)/2])
        lutm.scalar_bar_representation.position2 = np.array([0.06*hr, 0.6*vr])

        # misc lutm props
        lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-#6.3g"
        lutm.scalar_bar_representation.border_thickness = 0  # no ugly borders
        lutm.scalar_bar_widget.process_events = False  # make non-interactive
        lutm.number_of_labels = 11

        match self.ui.coloring_mode:

            case 'Wavelength':
                spectral_colormap = go.spectral_colormap if go.spectral_colormap is not None\
                                    else color.spectral_colormap
                lutm.lut.table = 255*spectral_colormap(color.wavelengths(255))
                lutm.data_range = go.wavelength_range 
                lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-6.0f"

            case ('Polarization xz' | "Polarization yz"):
                lutm.data_range = [0.0, 1.0]

            case 'Source':
                lutm.number_of_labels = len(self.raytracer.ray_sources)
                lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-6.0f"
                if lutm.number_of_labels > 1:  # pragma: no branch
                    lutm.lut.table = 255*color.spectral_colormap(np.linspace(440, 620, lutm.number_of_labels))

            case 'Plain':
                lutm.reverse_lut = bool(self.ui.high_contrast)

    def color_ray_sources(self) -> None:
        """sets colors of ray sources"""

        if self._ray_plot is None:
            return

        lutm = self._ray_plot.parent.scalar_lut_manager

        match self.ui.coloring_mode:

            case ("Plain" | "Refractive Index"):
                RSColor = [self.scene.foreground for RSp in self._ray_source_plots]

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
                            case "Constant":  # pragma: no branch
                                pol_ang = np.deg2rad(RS.pol_angle)
                    
                        proj = np.sin(pol_ang) if self.ui.coloring_mode == "Polarization yz" else np.cos(pol_ang)
                        col = np.array(lutm.lut.table[int(proj*255)])
                        RSColor.append(col[:3]/255.)
                    else:
                        RSColor.append(np.ones(3))

            case 'Source':
                RSColor = [np.array(lutm.lut.table[i][:3]) / 255. for i, _ in enumerate(self._ray_source_plots)]

            case 'Power':  # pragma: no branch
                # set to maximum ray power, this is the same for all sources
                RSColor = [np.array(lutm.lut.table[-1][:3]) / 255. for RSp in self._ray_source_plots]

        if len(self.raytracer.ray_sources) == len(self._ray_source_plots):
            for color, RSp in zip(RSColor, self._ray_source_plots):
                for RSpi in RSp[:2]:
                    if RSpi is not None:
                        RSpi.actor.actor.property.trait_set(color=tuple(color))
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
            with self.constant_camera():
                self._ray_plot.parent.parent.remove()
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

    def _on_space_pick(self, picker_obj: tvtk.tvtk_classes.point_picker.PointPicker) -> None:
        """
        3D Space Clicking Handler. Shows Click Coordinates or moves Detector to this position when Shift is pressed.

        :param picker_obj:
        """

        pos = picker_obj.pick_position

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
            self._ray_text.text = f"Pick Position (x, y, z):    ({pos[0]:>9.6g} mm, {pos[1]:>9.6g} mm, "\
                                  f"{pos[2]:>9.6g} mm)\n"\
                                  f"Relative to Axis (r, phi):  ({r:>9.6g} mm, {ang:>9.3f} °)"

            self.set_crosshair(pos)

    def _on_ray_pick(self, picker_obj: tvtk.tvtk_classes.point_picker.PointPicker = None) -> None:
        """
        Ray Picking Handler. Shows ray properties in the scene.

        :param picker_obj:
        """
        if self._ray_text is None:
            return

        # it seems that picker_obj.point_id is only present for the first element in the picked list,
        # so we only can use it when the RayPlot is first in the list
        # see https://github.com/enthought/mayavi/issues/906
        if picker_obj is not None and len(picker_obj.actors) != 0 \
           and picker_obj.actors[0]._vtk_obj is self._ray_plot.actor.actor._vtk_obj:

            a = self._ray_property_dict["p"].shape[1]  # number of points per ray plotted
            b = picker_obj.point_id  # point id of the ray point

            # calculate ray index (i0) and section index (i1)
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

        with self.constant_camera():
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
        self._ray_text.text = text.replace("nan", " - ")
        self._ray_text.property.trait_set(**self.INFO_STYLE, background_opacity=self._info_opacity,
                                          opacity=1, color=self.scene.foreground)

        self.set_crosshair(p)
