

"""
Frontend for the Raytracer class.
Shows a 3D representation of the geometry that is navigable.
Allows for the recalculation of the rays as well as visual settings for rays or points.
The detector is movable along the optical axis.
Interaction to source image and detector image plotting and its properties.
Autofocus functionality with different modes.

"""


from traits.api import HasTraits, Range, Instance, on_trait_change, Str, Button, Enum, List, Dict, Bool, Unicode
from traitsui.api import View, Item, HSplit, Group, CheckListEditor
from pyface.api import GUI as pyfaceGUI
from pyface.qt import QtCore, QtGui

from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.modules.surface import Surface as mayaviSurface
from mayavi.sources.parametric_surface import ParametricSurface

from Backend import *
from Backend.Misc import timer as timer
from Frontend.ImagePlots import DetectorPlot, SourcePlot
from Frontend.DebuggingPlots import AutoFocusDebugPlot

import numpy as np
from threading import Thread
from typing import Callable
from pynput.keyboard import Key, Listener

import time




#TODO aufräumen mehrere Detektoren, bessere Variablennamen etc

class GUI(HasTraits):

    ####################################################################################################################
    # UI objects

    DETECTOR_COLOR: tuple[float, float, float] = (0.1, 0.1, 0.1) 
    """RGB Color for the Detector visualization"""

    DETECTOR_ALPHA: float = 0.8
    """Alpha value for the Detector visualization"""

    BACKGROUND_COLOR: tuple[float, float, float] = (0.408, 0.4, 0.4)
    """RGB Color for the Scene background"""

    SUBTLE_COLOR: tuple[float, float, float] = (0.65, 0.65, 0.65)
    """RGB Color for subtle text"""

    LENS_COLOR: tuple[float, float, float] = (0.37, 0.69, 1.00)
    """RGB Color for the Lens surface visualization"""

    LENS_ALPHA: float = 0.32
    """Alpha value for the Lens visualization"""

    RAYSOURCE_COLOR: tuple[float, float, float] = (1.00, 1.00, 1.00)
    """RGB Color for the Lens surface visualization"""

    RAYSOURCE_ALPHA: float = 0.5
    """Alpha value for the RaySource visualization"""

    RAY_ALPHA: float = 0.01
    """Alpha value for the ray visualization"""

    FILTER_ALPHA: float = 0.9
    """Alpha value for the Filter visualization"""

    OUTLINE_ALPHA: float = 0.25
    """ Alpha value for outline visualization """

    FONT_STYLE: str = 'courier'
    """Text font style """

    FONT_SHADOW: bool = True
    """Activate/deactivate font shadow"""

    SURFACE_RES: int = 100
    """Surface sampling count in each dimension"""

    D_VIS: float = 0.125
    """Visualization thickness for side view of planar elements"""

    MAX_RAYS_SHOWN: int = 10000
    """Maximum of rays shown in visualization"""
    
    ##########
    # The mayavi scene.
    scene = Instance(MlabSceneModel, args=())

    # Ranges

    Rays = Range(5, 10000000, 200000, desc='Number of Rays for Simulation', enter_set=True,
                 auto_set=False, label="N_rays", mode='text')

    Rays_s = Range(-5.0, -1.0, -2, desc='Percentage of N Which is Drawn', enter_set=True,
                   auto_set=False, label="N_vis (log)", mode='slider')

    Pos_Det = Range(low='Pos_Det_min', high='Pos_Det_max', value='Pos_Det_max', mode='slider',
                    desc='z-Position of the Detector', enter_set=True, auto_set=True, label="z_pos")

    Ray_alpha = Range(-5.0, 0.0, np.log10(RAY_ALPHA), desc='Opacity of the Rays/Points', enter_set=True,
                      auto_set=True, label="Alpha (log)", mode='slider')

    Ray_width = Range(1.0, 20.0, 1, desc='Ray Linewidth or Point Size', enter_set=True,
                      auto_set=True, label="Width")

    ImagePixels = Range(1, 1000, 100, desc='Detector Image Pixels in Smaller of x or y Dimension', enter_set=True,
                        auto_set=True, label="Pixels_xy", mode='text')

    # Checklists (with capitalization workaround from https://stackoverflow.com/a/23783351)
    AbsorbMissing = List(editor=CheckListEditor(values=["Absorb Rays Missing Lens"], format_func=lambda x: x),
                         desc="if Rays are Absorbed when not Hitting a Lens")
    LogImage = List(editor=CheckListEditor(values=['Logarithmic Scaling'], format_func=lambda x: x),
                    desc="if Logarithmic Values are Plotted")

    FlipImage = List(editor=CheckListEditor(values=['Flip Image'], format_func=lambda x: x),
                    desc="if imaged should be rotated by 180 degrees")

    FocusDebugPlot = List(editor=CheckListEditor(values=['Show Cost Function'], format_func=lambda x: x),
                    desc="if cost function is shown")

    # Enums
    PlottingType        = Enum('Rays', 'Points', 'None', desc="Ray Representation")
    ColoringType        = Enum('White', 'Power', 'Wavelength', 'Polarization', 'Source', desc="Ray Property to Color the Rays With")
    ImageType           = Enum('Irradiance', 'Illuminance', 'sRGB', desc="Image Type Presented")
    FocusType           = Enum('Position Variance', 'Airy Disc Weighting', 'Irradiance Variance', desc="Method for Autofocus")
    SourceSelection     = Enum(values='SourceNames', desc="Source Selection for Source Image")
    DetectorSelection   = Enum(values='DetectorNames', desc="Detector Selection")

    # Buttons
    DetectorImageButton = Button(label="Show Detector Image", desc="Generating a Detector Image")
    SourceImageButton   = Button(label="Show Source Image", desc="Generating a Source Image of the Chosen Source")
    AutoFocusButton     = Button(label="Find Focus", desc="Finding the Focus Between the Lenses in Front and Behind the Detector")

    # Labels
    Whitespace_Label    = Str('\n')
    Separator           = Item("Whitespace_Label", style='readonly', show_label=False)

    Status=Dict(Unicode())

    ####################################################################################################################
    # UI view creation

    view = View(
                HSplit(
                    # add mayavi scene
                    Group(
                        Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                             height=850, width=1100, show_label=False)
                        ),
                    # add UI Elements
                    Group(
                        Separator,
                        Item('Rays'),
                        Item('AbsorbMissing', style="custom", show_label=False),
                        Separator,
                        Item("PlottingType", label='Plotting'),
                        Item("ColoringType", label='Coloring', visible_when="PlottingType != 'None'"),
                        Item('Rays_s', visible_when="PlottingType != 'None'"),
                        Item('Ray_alpha', visible_when="PlottingType != 'None'"),
                        Item('Ray_width', visible_when="PlottingType != 'None'"),
                        Separator,
                        Item('DetectorSelection', label="Detector", visible_when="len(DetectorNames) > 1"),
                        Item('Pos_Det', label="Det_z"),
                        Item('FocusType', label='AF Mode'),
                        Item('FocusDebugPlot', style="custom", show_label=False),
                        Item('AutoFocusButton', show_label=False),
                        Separator,
                        Item('ImageType', label='Image'),
                        Item('ImagePixels'),
                        Item('LogImage', style="custom", show_label=False),
                        Item('FlipImage', style="custom", show_label=False),
                        Item('DetectorImageButton', show_label=False),
                        Item('SourceSelection', label="Source", visible_when="len(SourceNames) > 1"),
                        Item('SourceImageButton', show_label=False)
                        ),
                    # add spacing right of UI Elements
                    Group(
                        Item("Whitespace_Label", style='readonly', show_label=False, width=20)
                        )
                    ),
                resizable=True
                )

    ####################################################################################################################
    # Class constructor

    def __init__(self, RT: Raytracer, AbsorbMissing: bool=True, *args, **kwargs) -> None:
        """

        :param RT:
        :param args:
        :param kwargs:
        """
        self.Raytracer = RT

        # ray properties
        self.RaysPlotScatter = None  # plot for scalar ray values
        self.RaysPlot = None  # plot for visualization of rays/points
        self.RayText = None  # textbox for ray information
        self.subset = np.array([])  # indices for subset of all rays in Raytracer

        # Detector properties
        self.DetectorPlot = []  # visual representation of the Detectord
        self.DetectorCylinderPlot = [] # visual representation of the Detectors' Cylinder
        self.DetText = []
        self.SourceNames = []
        self.DetectorNames = []
        self.DetInd = 0
        
        # minimal/maximal z-positions of Detector_obj
        self.Pos_Det_min = self.Raytracer.outline[4]
        self.Pos_Det_max = self.Raytracer.outline[5]

        # holds scene size, gets set on scene.activated
        self.scene_size = [0, 0]

        # hold axes objects
        self.ax_x = None
        self.ax_y = None
        self.ax_z = None

        # shift press indicator
        self.ShiftPressed = False

        # workaround to exit trait functions
        self.set_only = False
        
        # set AbsorbMissing parameter from bool to list
        self.AbsorbMissing = self.AbsorbMissing if AbsorbMissing else []

        # define Status dict
        self.Status.update(dict(Init=True, Tracing=False, Drawing=False, Focussing=False,
                                DetectorImage=False, SourceImage=False))

        super().__init__(*args, **kwargs)
        

    def __call__(self, exit=False) -> None:
        """
        start the GUI
        """
        self.exit = exit
        self.configure_traits()

    ####################################################################################################################
    # Plotting functions

    def plotLens(self, Lens: Lens, num: int) -> None:
        """

        :param Lens:
        :param num:
        """
        self.plotSObject(Lens, num, "Lens", "L", self.LENS_COLOR, self.LENS_ALPHA)

    def plotFilter(self, filter_: Filter, num: int) -> None:
        """

        :param filter_:
        :param num:
        """
        Fcolor = filter_.getColor()
        self.plotSObject(filter_, num, "Filter", "F", Fcolor, self.FILTER_ALPHA)    

    def plotDetector(self, Det: Detector, num: int) -> None:
        """

        :return:
        """

        a, b, c, text = self.plotSObject(Det, num, "Detector", "DET", 
                                                  self.DETECTOR_COLOR, self.DETECTOR_ALPHA)    

        self.DetectorPlot.append(a)
        self.DetectorCylinderPlot.append(b)
        self.DetText.append(text)

    def plotRaySource(self, RS: RaySource, num: int) -> None:
        """

        :param RS:
        :param num:
        """

        self.plotSObject(RS, num, "RaySource", "RS", RS.getColor(), self.RAYSOURCE_ALPHA, 
                                     d=-self.D_VIS, spec=False, light=False)   

        if RS.light_type in ["RGB_Image", "BW_Image"]:
            RS_Image_Text = self.scene.mlab.text3d(RS.pos[0]-0.5, RS.pos[1], RS.pos[2], "Image", color=(0, 0, 0),
                                                   orient_to_camera=False, scale=0.25, name=f"Image Text")
            RS_Image_Text.actor.actor.pickable = False
            RS_Image_Text.actor.actor.property.lighting = False

    def plotOutline(self) -> None:
        """plot the raytracer outline"""

        self.scene.engine.add_source(ParametricSurface(name="Outline"), self.scene)
        a = self.scene.mlab.outline(extent=self.Raytracer.outline.copy(), opacity=self.OUTLINE_ALPHA)
        a.actor.actor.pickable = False  # only rays should be pickablw

    def plotOrientationAxes(self) -> None:
        """plot orientation axes"""

        self.scene.engine.add_source(ParametricSurface(name="Orientation Axes"), self.scene)
        
        # show axes indicator
        self.oaxes = self.scene.mlab.orientation_axes()
        self.oaxes.text_property.trait_set(font_family=self.FONT_STYLE, shadow=self.FONT_SHADOW)
        self.oaxes.marker.interactive = 0  # make orientation axes non-interactive

        # turn of text scaling of oaxes
        self.oaxes.widgets[0].orientation_marker.x_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self.oaxes.widgets[0].orientation_marker.y_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self.oaxes.widgets[0].orientation_marker.z_axis_caption_actor2d.text_actor.text_scale_mode = 'none'

    def plotAxes(self) -> None:
        """plot cartesian axes"""

        # find label number for axis so that step size is an int*10^k 
        # or any of [0.25, 0.5, 0.75, 1.25, 2.5]*10^k with k being an integer
        # label number needs to be in range [min_s, max_s]
        def getLabelNum(num: float, min_s: int, max_s: int) -> int:
            norm = 10 ** -np.floor(np.log10(num)-1)
            num_norm = num*norm  # normalize num so that 10 <= num < 100
            
            for i in np.arange(max_s, min_s-1, -1):
                if num_norm/i in [0.2, 0.25, 0.4, 0.5, 0.75, 0.8, 1,
                                  1.25, 1.5, 2, 2.5, 3, 3.75, 4, 5, 6, 6.25, 7.5, 8, 10]:
                    return i+1  # increment since there are n+1 labels for n steps

            return max_s+1

        def drawAxis(obj, ext: list, lnum: int, name: str, lform: str, vis_x: bool, vis_y: bool, vis_z: bool):

            self.scene.engine.add_source(ParametricSurface(name=f"{name}-Axis"), self.scene)

            obj = self.scene.mlab.axes(extent=ext, nb_labels=lnum, x_axis_visibility=vis_x, 
                                       y_axis_visibility=vis_y, z_axis_visibility=vis_z)

            label=f"{name} / mm"
            obj.axes.trait_set(font_factor=0.7, fly_mode='none', label_format=lform, x_label=label, 
                               y_label=label, z_label=label, layer_number=1)

            obj.property.trait_set(display_location='background', opacity=0.5)
            obj.title_text_property.trait_set(font_family=self.FONT_STYLE, shadow=self.FONT_SHADOW)
            obj.label_text_property.trait_set(font_family=self.FONT_STYLE, shadow=self.FONT_SHADOW)
            obj.actors[0].pickable = False

        # place axes at outline
        ext = self.Raytracer.outline

        # enforce placement of x- and z-axis at y=ys (=RT.outline[2]) by shrinking extent
        ext_ys = ext.copy()
        ext_ys[3] = ext_ys[2]

        # X-Axis
        lnum = getLabelNum(ext[1] - ext[0], 5, 16)
        drawAxis(self.ax_x, ext_ys, lnum, "x", '%-#.4g', True, False, False)

        # Y-Axis
        lnum = getLabelNum(ext[3] - ext[2], 5, 16)
        drawAxis(self.ax_y, ext.copy(), lnum, "y", '%-#.4g', False, True, False)

        # Z-Axis
        lnum = getLabelNum(ext[5] - ext[4], 5, 24)
        drawAxis(self.ax_z, ext_ys, lnum, "z", '%-#.5g', False, False, True)

    
    def plotRefractionIndexBoxes(self) -> None:
        """plot outlines for ambient refraction index regions"""

        # sort Element list in z order
        Lenses = sorted(self.Raytracer.LensList, key=lambda Element: Element.pos[2])

        # create n list and z-boundary list
        nList = [self.Raytracer.n0] + [Element.n2 for Element in Lenses] + [self.Raytracer.n0]
        BoundList = [self.Raytracer.outline[4]] + [Element.pos[2] for Element in Lenses] + [self.Raytracer.outline[5]]

        # replace None values of Lenses n2 with the ambient n0
        nList = [self.Raytracer.n0 if ni is None else ni for ni in nList]

        # join boxes with the same refraction index
        i = 0
        while i < len(nList)-2:
            if nList[i] == nList[i+1]:
                del nList[i+1]
                del BoundList[i+1]
            else:
                i += 1

        # skip box plotting if n=1 everywhere
        if len(BoundList) == 2 and nList[0] == RefractionIndex("Constant", n=1):
            return

        # plot boxes
        for i in range(len(BoundList)-1):

            # plot outline
            self.scene.engine.add_source(ParametricSurface(name=f"Refraction Index Outline {i}"), self.scene)
            outline = self.scene.mlab.outline(extent=[*self.Raytracer.outline[:4], BoundList[i], BoundList[i+1]],
                                              opacity=self.OUTLINE_ALPHA)
            outline.outline_mode = 'cornered'
            outline.actor.actor.pickable = False  # only rays should be pickablw

            # plot label
            label = str(nList[i].n) if nList[i].n_type == "Constant" else "function"
            x_pos = self.Raytracer.outline[0] + (self.Raytracer.outline[1]-self.Raytracer.outline[0])*0.05
            z_pos = (BoundList[i+1]+BoundList[i])/2
            text  = self.scene.mlab.text(x_pos, 0, z=z_pos, text=f"ambient\nn={label}", name=f"Label")

            text.actor.text_scale_mode = 'none'
            text.property.trait_set(font_size=11, font_family=self.FONT_STYLE, shadow=self.FONT_SHADOW, 
                                    justification="center", frame=True, frame_color=self.SUBTLE_COLOR)

    def plotSObject(self, obj, num, name, shortname, color, alpha, d=D_VIS, spec=True, light=True):
        """plotting of a SObject. Gets called from plotting for Lens, Filter, Detector, RaySource."""
        # plot surface
        X, Y, Z = obj.FrontSurface.getPlottingMesh(N=self.SURFACE_RES)
        a = self.scene.mlab.mesh(X, Y, Z, color=color, opacity=alpha,
                                 name=f"{name} {num} front surface")

        # make non pickable so it does not interfere with our ray picker
        a.actor.actor.pickable = False
        a.actor.actor.property.lighting = light
        if spec:
            a.actor.property.trait_set(specular=0.5, ambient=0.25)

        # if surface is a plane, plot a cylinder for side viewing
        if obj.hasBackSurface() or obj.Surface.isPlanar():
            Xj, Yj, Zj = obj.getCylinderSurface(nc=self.SURFACE_RES, d=d)
            b = self.scene.mlab.mesh(Xj, Yj, Zj, color=color, opacity=alpha,
                                     name=f"{name} {num} cylinder")
            b.actor.actor.pickable = False
            b.actor.actor.property.lighting = light
            if spec:
                b.actor.property.trait_set(specular=0.5, ambient=0.25)
        else:
            b = None

        # calculate middle center z-position
        zl = (obj.extent[4] + obj.extent[5])/2 if not obj.hasBackSurface() else obj.pos[2]
        zl = zl + self.D_VIS/2 if obj.Surface.isPlanar() else zl

        # filter label
        text = self.scene.mlab.text(obj.extent[1], 0, z=zl, text=f"{shortname}{num}", name=f"Label")
        text.actor.text_scale_mode = 'none'
        text.property.trait_set(font_size=11, font_family=self.FONT_STYLE, shadow=self.FONT_SHADOW,
                                justification="center")
        
        if obj.hasBackSurface():
            X2, Y2, Z2 = obj.BackSurface.getPlottingMesh(N=self.SURFACE_RES)
            c = self.scene.mlab.mesh(X2, Y2, Z2, color=color,  opacity=alpha,
                                     name=f"{name} {num} back surface")
            c.actor.actor.pickable = False
            c.actor.actor.property.lighting = light
            if spec:
                c.actor.property.trait_set(specular=0.5, ambient=0.25)
        else:
            c = None

        return a, b, c, text

    def plotRays(self) -> None:
        """Ray/Point Plotting with a specified scalar coloring"""
        p_, _, pol_, w_, wl_, snum_ = self.Raytracer.RaySources.getRaysByMask(self.subset, 
                                                                ret=[True, False, True, True, True, True])

        # get flattened list of the coordinates
        x_cor = p_[:, :, 0].ravel()
        y_cor = p_[:, :, 1].ravel()
        z_cor = p_[:, :, 2].ravel()

        # connections are a list of lines, each line is a list of indices (of points that are connected)
        #
        # connections = [ [0, 1, 2, .., nt],                with nt: number of points per ray (equal for all rays)
        #                 [nt+1, nt+2, ..., 2*nt],                N: number of rays
        #                 ...                                    nc: coordinate dimensions (=3)
        #                 [(N-1)*nt+1, ...,  N*nt]]
        N, nt, nc = p_.shape
        connections = np.reshape(np.arange(N*nt), (N, nt))

        # set plotting properties depending on plotting mode
        match self.ColoringType:

            case 'Power':
                s = w_.ravel()*1e6
                cm = "gnuplot"
                title = "Ray Power\n in µW\n"

            case 'Source':
                s = snum_.repeat(nt)
                cm = "spring"
                title = "Ray Source\nNumber"

            case 'Wavelength':
                s = wl_.repeat(nt)
                cm = "nipy_spectral"
                title = "Wavelength\n in nm\n"

            case 'Polarization':
                s = np.sqrt(pol_[:, :, 1]**2 + pol_[:, :, 2]**2).ravel()
                cm = "gnuplot"
                title = "Polarization\n projection\n on yz-plane"

            case _:
                s = np.ones_like(z_cor)
                cm = "Greys"
                title = "None"

        # no-color-smoothing-workaround
        if self.PlottingType == "Rays" and self.ColoringType != "White":

            # we now want to assign scalar values to the ray coordinates (scalar can be ray intensity, color, ...)
            # mayavi automatically blends the transition between scalar values, to circumvent this
            # we duplicate all points inside the ray and set both ray colors to the same coordinate.
            # Doing so, the blending region gets infinitesimal small and we achieve sharp transitions

            # create bool array with twice the columns, but first and last element in row set to false
            pos_ins = np.concatenate((np.arange(N)*nt*2, np.arange(1, N+1)*nt*2-1))
            bool_m = np.ones(N*nt*2, dtype=bool)
            bool_m[pos_ins] = False

            # duplicate all coordinates inside the ray, excluding start and beginning
            x_cor = np.repeat(x_cor, 2)[bool_m]
            y_cor = np.repeat(y_cor, 2)[bool_m]
            z_cor = np.repeat(z_cor, 2)[bool_m]

            # recreate the connection list
            connections = np.reshape(np.arange(2*N*(nt-1)), (N, 2*(nt-1)))

            # repeat all scalar values except last in ray
            bool_m2 = np.ones(N*nt, dtype=bool)
            bool_m2[np.arange(1, N+1)*nt - 1] = False
            s = np.repeat(s[bool_m2], 2)

        # Assign scalars to plot
        self.RaysPlotScatter = self.scene.mlab.pipeline.scalar_scatter(x_cor, y_cor, z_cor, s, name="Rays")

        # assign which points are connected
        self.RaysPlotScatter.mlab_source.dataset.lines = connections

        # plot the lines between points
        self.RaysPlot = self.scene.mlab.pipeline.surface(self.RaysPlotScatter, colormap=cm,
                                                         line_width=self.Ray_width, opacity=10**self.Ray_alpha)
        self.RaysPlot.actor.actor.property.trait_set(lighting=False, render_points_as_spheres=True)

        # lut legend settings
        lutm = self.RaysPlot.parent.scalar_lut_manager
        lutm.trait_set(use_default_range=True, show_scalar_bar=True, use_default_name=False,
                       show_legend=self.ColoringType!="White")

        # lut visibility and title
        lutm.scalar_bar.trait_set(title=title, unconstrained_font_size=True)
        lutm.label_text_property.trait_set(font_family=self.FONT_STYLE, shadow=self.FONT_SHADOW, font_size=12, bold=True)
        lutm.title_text_property.trait_set(font_family=self.FONT_STYLE, shadow=self.FONT_SHADOW, font_size=14)

        # lut position and size
        lutm.scalar_bar_representation.position         = np.array([0.9, 0.1])
        lutm.scalar_bar_representation.position2        = np.array([0.09, 0.8])
        lutm.scalar_bar_widget.process_events           = False  # make non-interactive
        lutm.scalar_bar_representation.border_thickness = 0  # no ugly borders 

        # custom lut for wavelengths
        match self.ColoringType:

            case 'Wavelength':
                lutm.lut.table = Color.spectralCM(255)
                lutm.data_range = [380, 780]
                lutm.number_of_labels = 11

            case 'Polarization':
                if self.Raytracer.no_pol:
                    raise RuntimeError("WARNING: Polarization calculation turned off in Raytracer.")

                lrange = lutm.data_range[1]-lutm.data_range[0]

                if lrange > 0.8 and 1 - lutm.data_range[1] < 0.05:
                    lutm.data_range = [lutm.data_range[0], 1]

                if lutm.data_range[0]/lrange < 0.05: 
                    lutm.data_range = [0, lutm.data_range[1]]

                lutm.number_of_labels = 11
            
            case 'Source':
                lutm.number_of_labels = len(self.Raytracer.RaySources.List)
                lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-#6.f"
                if lutm.number_of_labels > 1:
                    # set color steps, smaller 420-650nm range for  larger color contrast
                    lutm.lut.table = Color.spectralCM(lutm.number_of_labels, 420, 650)

            case 'Power':
                if lutm.data_range[0]/(lutm.data_range[1]-lutm.data_range[0]) < 0.05: 
                    lutm.data_range = [0, lutm.data_range[1]]

        # set visual ray properties
        self.RaysPlot.actor.property.representation = 'points' if self.PlottingType == 'Points' else 'surface'
        self.RaysPlot.actor.property.trait_set(line_width=self.Ray_width, point_size=self.Ray_width)
        self.RaysPlot.actor.actor.force_translucent = True

    def plotGeometry(self) -> None:
        """plot Background, Objects (Lenses, RaySources, Detectors, Filters), Outlines and Axes"""
        # plot RaySources
        [self.plotRaySource(RS, num+1) for num, RS in enumerate(self.Raytracer.RaySources.List)]

        # plot lenses
        [self.plotLens(L, num+1) for num, L in enumerate(self.Raytracer.LensList)]

        # plot filters
        [self.plotFilter(F, num+1) for num, F in enumerate(self.Raytracer.FilterList)]

        # plot detectors
        [self.plotDetector(D, num+1) for num, D in enumerate(self.Raytracer.DetectorList)]

        # assign detector position
        if not self.Raytracer.DetectorList:
            self.Pos_Det = [self.Raytracer.DetectorList[0].pos[2]]
        else:
            self.Pos_Det = self.Raytracer.outline[5]

        # plot axes
        self.plotAxes()
        self.plotOrientationAxes()

        self.plotOutline()

        # plot refraction index boxes
        self.plotRefractionIndexBoxes()

        # set background color
        self.scene.background = self.BACKGROUND_COLOR

    def initRayInfoText(self) -> None:
        """init detection of ray point clicks and the info display"""
        # init point pickers
        self.scene.mlab.gcf().on_mouse_pick(self.onRayPick, button='Left')
        self.scene.mlab.gcf().on_mouse_pick(self.onSpacePick, button='Right')

        # add ray info text
        self.scene.engine.add_source(ParametricSurface(name="Ray Info Text"), self.scene)
        self.onRayPick()  # call to set default text

    def initStatusText(self) -> None:
        """init GUI status text display"""
        # add status text
        self.scene.engine.add_source(ParametricSurface(name="Status Info Text"), self.scene)
        self.StatusText = self.scene.mlab.text(0.97, 0.01, "Status Text")
        self.StatusText.actor.text_scale_mode = 'none'
        self.StatusText.property.trait_set(font_size=13, color=(1, 1, 1), shadow=self.FONT_SHADOW, bold=True,
                                           font_family=self.FONT_STYLE, vertical_justification=0, justification=2)
    
    def initSourceList(self) -> None:
        """generate a descriptive list of RaySource names"""
        self.SourceNames = [f"RS{num+1}: {RS.Surface.surface_type} at {RS.pos}"
                            for num, RS in enumerate(self.Raytracer.RaySources.List) ]

    def initDetectorList(self) -> None:
        """generate a descriptive list of Detector names"""
        self.DetectorNames = [f"DET{num+1}: {Det.Surface.surface_type} at z={Det.pos[2]}" 
                              for num, Det in enumerate(self.Raytracer.DetectorList) ]

    def initCamera(self) -> None:
        """initialize the scene view and the camera"""
        # set camera position
        self.scene.mlab.view(0, 0)
        dz = self.Raytracer.outline[5] - self.Raytracer.outline[4]
        self.scene.scene.camera.position       = [0.0, dz, self.Raytracer.outline[4] + dz/2]
        self.scene.scene.camera.focal_point    = [0.0, 0, self.Raytracer.outline[4] + dz/2]
        self.scene.scene.camera.view_up        = [1.0, 0.0, 0.0]
        self.scene.scene.camera.parallel_scale = dz/2

        # set parallel projection
        self.scene.scene.parallel_projection = True

        # re-render
        self.scene.renderer.reset_camera()
        self.scene.scene.render()


    def onSpacePick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker') -> None:
        """
        3D Space Clicking Handler. 
        Shows Click Coordinates or moves Detector to this position when Shift is pressed.
        :param picker_obj:
        """

        pos = picker_obj.pick_position

        if self.ShiftPressed:
            # set outside position to inside of outline
            pos_z = max(self.Pos_Det_min, pos[2])
            pos_z = min(self.Pos_Det_max, pos_z)
            
            # assign detector position, calls moveDetector()
            self.Pos_Det = pos_z
            
            # call with no parameter resets text
            self.onRayPick()
        else:
            self.RayText.text = f"Pick Position: ({pos[0]:>9.6g}, {pos[1]:>9.6g}, {pos[2]:>9.6g})"
            self.RayText.property.trait_set(color=(1, 1, 1), bold=True, background_opacity=0.2)

    def onRayPick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker'=None) -> None:
        """
        Ray Picking Handler.
        Shows ray properties on screen.
        :param picker_obj:
        """
        if not self.RayText:
            self.RayText = self.scene.mlab.text(0.02, 0.97, "") 

        # it seems that picker_obj.point_id is only present for the first element in the picked list,
        # so we only can use it when the RayPlot is first in the list
        # see https://github.com/enthought/mayavi/issues/906
        if picker_obj is not None and len(picker_obj.actors) != 0 \
           and picker_obj.actors[0]._vtk_obj is self.RaysPlot.actor.actor._vtk_obj:
            
            a = self.Raytracer.RaySources.nt # number of points per ray plotted
            b = picker_obj.point_id # point id of the ray point

            # get ray number and ray section
            if self.PlottingType == "Rays" and self.ColoringType != "White":
                # in some cases we changed the number of points per ray using the workaround in self.plotRays()
                n_, n2_ = np.divmod(b, (a-1)*2)
                if n2_ > 0:
                    n2_ = 1 + (n2_-1)//2
            else:
                n_, n2_ = np.divmod(b, a)

            pos = np.nonzero(self.subset)[0][n_]

            # get properties of this ray section
            p_, s_, pols_, pw_, wv, snum = self.Raytracer.RaySources.getRay(pos)
            p, s, pols, pw = p_[n2_], s_[n2_], pols_[n2_], pw_[n2_]
            pl = (pw_[0]-pw)/pw_[0]  # power loss

            # generate text
            text = f"Ray {pos}" +  \
                   f" from Source {snum}" +  \
                   (f" at surface {n2_}\n" if n2_> 0 else f" at ray source\n") + \
                   f"Position:     ({p[0]:>10.5g}, {p[1]:>10.5g}, {p[2]:>10.5g})\n" + \
                   f"Direction     ({s[0]:>10.5f}, {s[1]:>10.5f}, {s[2]:>10.5f})\n" + \
                   f"Polarization: ({pols[0]:>10.5f}, {pols[1]:>10.5f}, {pols[2]:>10.5f})\n" + \
                   f"Wavelength:  {wv:>10.2f}nm\n" + \
                   f"Ray Power:   {pw*1e6:>10.5g}µW\n" + \
                   f"Power Loss:  {pl*100:>10.5g}%"
            
            self.RayText.text = text
            self.RayText.property.trait_set(color=(1, 1, 1), bold=True, background_opacity=0.2)
        else:
            self.RayText.text = "Left Click on a Ray-Surface Intersection to Show Ray Properties.\n" + \
                                "Right Click Anywhere to Show 3D Position.\n" + \
                                "Shift + Right Click to Move the Detector to this z-Position"
            self.RayText.property.trait_set(color=self.SUBTLE_COLOR, bold=False, background_opacity=0)

        # text settings
        self.RayText.actor.text_scale_mode = 'none'
        self.RayText.property.trait_set(font_family=self.FONT_STYLE, vertical_justification=2, font_size=13,
                                        shadow=self.FONT_SHADOW)

    def initShiftPressDetector(self) -> None:
        """Init shift key press detection"""
        self.ShiftPressed = False

        def on_press(key):
            if key == Key.shift:
                self.ShiftPressed = True

        def on_release(key):
            if key == Key.shift:
                self.ShiftPressed = False

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()


    def close(self) -> None:
        """close the GUI"""
        # close all ui frames
        app = QtGui.QApplication.instance()
        for widget in app.topLevelWidgets():
            widget.close()

        # exit application
        QtCore.QCoreApplication.quit()

    # wait while 'key' in StatusList is set
    def waitWhile(self, key: str):
        """
        wait while the GUI is busy doing 'key'
        :param key: string key from GUI.Status
        """
        time.sleep(0.001)  # wait for flags to be set
        # wait for tasks to finish
        while self.Status[key]:
            time.sleep(0.02)

    ####################################################################################################################
    # Trait handlers.

    @on_trait_change('Rays_s')
    def filterRays(self) -> None:
        """chose a subset of all Raytracer rays and plot them with :obj:`GUI.drawRays`"""

        # don's use self.Rays as ray number, since the raytracer could still be raytracing
        # instead use the last saved number
        N = self.Raytracer.RaySources.N

        if self.PlottingType == "None" or N == 0:
            return

        # currently busy, settings will be applied after current task
        if self.Status["Init"] or self.Status["Tracing"]:
            return

        self.Status["Drawing"] = True

        set_size = int(1 + 10**self.Rays_s*N)  # number of shown rays
        sindex = np.random.choice(N, size=set_size, replace=False)  # random choice

        # make bool array with chosen rays set to true
        self.subset = np.zeros((N,), dtype=bool)
        self.subset[sindex] = True

        self.drawRays()

    @on_trait_change('ColoringType')
    def drawRays(self) -> None:
        """remove old ray view, call :obj:`GUI.plotRays` and restore scene view"""

        if not self.Raytracer.RaySources.hasSources():
            print("WARNING: RaySource Missing")

        # currently busy, but next plotting will use the current settings
        elif self.Status["Init"] or self.Status["Tracing"]:
            return

        # only draw if source has rays and scene renderer is initialized
        elif self.Raytracer.RaySources.hasRays() and not self.Status["Init"]:

            self.Status["Drawing"] = True

            # we could set the ray traits instead of removing and creating a new object,
            # but unfortunately the allocated numpy arrays (x, y, z, ..) of the objects are fixed in size
            if self.RaysPlot:
                self.RaysPlot.remove()
            if self.RaysPlotScatter:
                self.RaysPlotScatter.remove()

            # save camera
            cc_traits_org = self.scene.camera.trait_get("position", "focal_point", "view_up", "view_angle",
                                                        "clipping_range", "parallel_scale")
            self.plotRays()
          
            # reset camera
            self.scene.camera.trait_set(**cc_traits_org)
           
            # show picker text 
            self.onRayPick()

            self.Status["Drawing"] = False


    @on_trait_change('Rays, AbsorbMissing')
    def setRays(self) -> None:
        """raytrace in separate thread, after that call :obj:`GUI.filterRays`"""

        # return if value should only be set
        if self.set_only or self.Status["Init"]:
            return

        if self.Raytracer.RaySources.hasSources():

            for key in self.Status.keys():
                if self.Status[key] and key != "InitTracing":
                    self.StatusText.text += "Can't trace while other background tasks are running.\n"
                    return

            self.Raytracer.AbsorbMissing = self.AbsorbMissing

            self.Status["Tracing"] = True

            # run this in background thread
            def background(RT: Raytracer, N: int, func: Callable) -> None:
                # trace if not already
                if not RT.RaySources.hasRays() or self.Rays != RT.RaySources.N:
                    RT.trace(N=N)
                pyfaceGUI.invoke_later(func)

            # execute this after thread has finished
            def after() -> None:

                self.Status["Tracing"] = False

                # lower the ratio of rays shown so that the GUI does not become to slow
                if 10**self.Rays_s*self.Rays > self.MAX_RAYS_SHOWN:
                    self.Rays_s = np.log10(self.MAX_RAYS_SHOWN/self.Rays)
                else: # in the other case self.filterRays() is called because of the self.Rays_s change
                    self.filterRays()

                # set parameters to current value, since they could be changed by the user while tracing
                self.set_only = True
                self.AbsorbMissing = ["Absorb Rays Missing Lens"] if self.Raytracer.AbsorbMissing else []
                self.Rays = self.Raytracer.RaySources.N
                self.set_only = False

                if self.exit:
                    self.close()

            action = Thread(target=background, args=(self.Raytracer, self.Rays, after))
            action.start()
        else:
            print("WARNING: RaySource Missing")

    @on_trait_change('Ray_alpha')
    def setRayAlpha(self) -> None:
        """change opacity of visible rays"""

        if self.RaysPlot:
            self.Status["Drawing"] = True
            self.RaysPlot.actor.property.trait_set(opacity=10**self.Ray_alpha)
            self.Status["Drawing"] = False

    @on_trait_change('Pos_Det')
    def setDetector(self) -> None:
        """move and replot chosen Detector"""

        if self.set_only:
            return

        if self.DetectorPlot:
            xp, yp, zp = self.Raytracer.DetectorList[self.DetInd].pos
            self.Raytracer.DetectorList[self.DetInd].moveTo([xp, yp, self.Pos_Det])
            
            Surf = self.Raytracer.DetectorList[self.DetInd].Surface.getPlottingMesh(N=self.SURFACE_RES)[2]
            self.DetectorPlot[self.DetInd].mlab_source.trait_set(z=Surf)

            if self.DetectorCylinderPlot[self.DetInd]:
                Cyl = self.Raytracer.DetectorList[self.DetInd].getCylinderSurface(nc=self.SURFACE_RES)[2] 
                self.DetectorCylinderPlot[self.DetInd].mlab_source.trait_set(z=Cyl)

            self.DetText[self.DetInd].z_position = self.Pos_Det

        elif not self.Raytracer or not self.Raytracer.DetectorList:
            print("WARNING: Detector missing.")

    @on_trait_change('DetectorImageButton')
    def showDetectorImage(self) -> None:
        """render a DetectorImage at the chosen Detector, uses a separate thread"""

        # prevent race conditions
        if self.Status["Tracing"]:
            self.StatusText.text += "Can't do this while raytracing.\n"
            return

        self.Status["DetectorImage"] = True

        ret = [0]

        def background(RT: Raytracer, N: int, ret: list, func: Callable) -> None:
            ret[0] = RT.DetectorImage(N=N, ind=self.DetInd)
            pyfaceGUI.invoke_later(func)

        def after() -> None:
            DetectorPlot(ret[0], log=self.LogImage, flip=self.FlipImage, mode=self.ImageType)
            self.Status["DetectorImage"] = False

        action = Thread(target=background, args=(self.Raytracer, self.ImagePixels, ret, after))
        action.start()

    @on_trait_change('SourceImageButton')
    def showSourceImage(self) -> None:
        """render a SourceImage for the chosen Source, uses a separate thread"""

        # prevent race conditions
        if self.Status["Tracing"]:
            self.StatusText.text += "Can't do this while raytracing.\n"
            return

        # surface number for selected source
        snum = int(self.SourceSelection.split(":", 1)[0].split("RS")[1])

        self.Status["SourceImage"] = True

        ret = [0]

        def background(RT: Raytracer, N: int, snum: int, ret: list, func: Callable) -> None:
            ret[0] = RT.SourceImage(N=N, sindex=snum-1)
            pyfaceGUI.invoke_later(func)

        def after() -> None:
            SourcePlot(ret[0], log=self.LogImage, flip=self.FlipImage, mode=self.ImageType)
            self.Status["SourceImage"] = False

        action = Thread(target=background, args=(self.Raytracer, self.ImagePixels, snum, ret, after))
        action.start()

    @on_trait_change('AutoFocusButton')
    def moveToFocus(self) -> None:
        """
        Find a Focus.
        The chosen Detector defines the search range for focus finding. 
        Searches are always between lenses aor the next outline. 
        Search takes place in a separate thread, after that the Detector is moved to the focus 
        """

        # prevent race conditions
        if self.Status["Tracing"]:
            self.StatusText.text += "Can't do this while raytracing.\n"
            return
            
        self.Status["Focussing"] = True
        
        r = [0]
        vals = [0]

        # run this in background thread
        def background(RT: Raytracer, z: float, method: str, r: list, vals: list, func: Callable) -> None:
            zf, r[0], vals[0] = RT.autofocus(method, z, ret_cost=self.FocusDebugPlot)
            RT.DetectorList[self.DetInd].moveTo([*RT.DetectorList[self.DetInd].pos[:2], zf])
            pyfaceGUI.invoke_later(func)

        # execute this function after thread has finished
        def after() -> None:
            if self.FocusDebugPlot:
                AutoFocusDebugPlot(r[0], vals[0])
            self.Pos_Det = self.Raytracer.DetectorList[self.DetInd].pos[2]  # setDetector() is called
            self.Status["Focussing"] = False
       
        action = Thread(target=background, args=(self.Raytracer, self.Pos_Det, self.FocusType, r, vals, after))
        action.start()

    @on_trait_change('scene.activated')
    def plotScene(self) -> None:
        """Initialize the GUI. Inits a variety of things."""

        self.plotGeometry()

        self.initCamera()
        self.initSourceList()
        self.initDetectorList()

        self.initRayInfoText()
        self.initStatusText()

        self.initShiftPressDetector()

        self.Status["Init"] = False

        # set absorb rays missing to true by default, side effect is calling setRays()
        self.AbsorbMissing = ["Absorb Rays Missing Lens"]
        # self.setRays()


    @on_trait_change('Status[]')
    def changeStatus(self) -> None:
        """Update the status info text."""

        if self.Status["Init"]:
            return

        text = ""
        if self.Status["Tracing"]:
            text += "Raytracing...\n"
        if self.Status["Focussing"]:
            text += "Focussing...\n"
        if self.Status["DetectorImage"]:
            text += "Generating Detector Image...\n"
        if self.Status["SourceImage"]:
            text += "Generating Source Image...\n"
        if self.Status["Drawing"]:
            text += "Drawing Rays...\n"
        self.StatusText.text = text

    @on_trait_change('Ray_width')
    def changeRayWidth(self) -> None:
        """ set the ray width for the visible rays."""

        if self.RaysPlot:
            self.Status["Drawing"] = True
            self.RaysPlot.actor.property.trait_set(line_width=self.Ray_width, point_size=self.Ray_width)
            self.Status["Drawing"] = False


    @on_trait_change('PlottingType')
    def changePlottingType(self) -> None:
        """Change the PlottingType. Removes the visual rays/points and replots using :obj:`GUI.drawRays`"""

        # remove ray/point plot
        if self.PlottingType == "None" and self.RaysPlot is not None:
            self.Status["Drawing"] = True
            self.RaysPlot.remove()
            self.RaysPlotScatter.remove()
            self.RaysPlot = None
            self.RaysPlotScatter = None
            self.Status["Drawing"] = False

        # replot rays/points
        else:
            self.drawRays()

    @on_trait_change('DetectorSelection')
    def changeDetector(self) -> None:
        """Updates the Detector Selection and the corresponding properties"""

        # surface number for selected source
        self.DetInd = int(self.DetectorSelection.split(":", 1)[0].split("DET")[1]) - 1
        self.set_only = True
        self.Pos_Det = self.Raytracer.DetectorList[self.DetInd].pos[2]
        self.set_only = False

    # better lighting on y+ starting view
    # @on_trait_change('scene:scene_editor:light_manager')
    # def better_lighting(self) -> None:
        # self.scene.scene_editor.light_manager.lights[0].trait_set(azimuth=10, elevation=10)

    # rescale axes texts when the window was resized
    # for some reasons these are the only ones having no text_scaling_mode = 'none' option
    @on_trait_change('scene:scene_editor:interactor:size')
    def size_change(self) -> None:
        """Handles GUI window size changes. Fixes incorrect scaling by mayavi"""

        scene_size = self.scene.scene_editor.interactor.size

        # init scene size
        if not self.scene_size[0] and not self.scene_size[1]:
            self.scene_size = scene_size

        elif scene_size[0] != self.scene_size[0] or scene_size[1] != self.scene_size[1]:

            # average of x and y size
            ch1 = (self.scene_size[0]+self.scene_size[1])/2
            ch2 = (scene_size[0]+scene_size[1])/2
           
            # update font factor so font size stays the same
            self.ax_x.axes.trait_set(font_factor=self.ax_x.axes.font_factor*ch1/ch2)
            self.ax_y.axes.trait_set(font_factor=self.ax_y.axes.font_factor*ch1/ch2)
            self.ax_z.axes.trait_set(font_factor=self.ax_z.axes.font_factor*ch1/ch2)

            # rescale orientation axes
            self.oaxes.widgets[0].zoom = self.oaxes.widgets[0].zoom * ch1/ch2

            # set current window size
            self.scene_size = scene_size

