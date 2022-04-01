

"""
Frontend for the Raytracer class.
Shows a 3D representation of the geometry that is navigable.
Allows for the recalculation of the rays as well as visual settings for rays or points.
The detector is movable along the optical axis.
Interaction to source image and detector image plotting and its properties.
Autofocus functionality with different modes.

"""


from traits.api import HasTraits, Range, Instance, on_trait_change, Str, Button, Enum, List, Dict
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
import traceback  # for printing traceback
import readline  # enables cursor movements in input()

# TODO Farben der RaySources passen sich ColoringType an
# Whites -> RS sind White, Power -> maximale Farbe der Skala, Wavelength -> RS.getColor()
# Polarization -> Weiß wenn zufällig, sonst nach Farbskala. Source -> Farbe der Quelle

# TODO Filter-opacity ergibt sich nach durchschnittlichem tau. Farbe wie bekannt

class GUI(HasTraits):

    ####################################################################################################################
    # UI objects

    DETECTOR_COLOR: tuple[float, float, float] = (0.10, 0.10, 0.10) 
    """RGB Color for the Detector visualization"""

    DETECTOR_ALPHA: float = 0.8
    """Alpha value for the Detector visualization"""

    BACKGROUND_COLOR: tuple[float, float, float] = (0.41, 0.40, 0.40)
    """RGB Color for the Scene background"""

    LENS_COLOR: tuple[float, float, float] = (0.63, 0.79, 1.00)
    """RGB Color for the Lens surface visualization"""

    LENS_ALPHA: float = 0.32
    """Alpha value for the Lens visualization"""

    RAYSOURCE_ALPHA: float = 0.5
    """Alpha value for the RaySource visualization"""

    RAY_ALPHA: float = 0.01
    """Alpha value for the ray visualization"""

    OUTLINE_ALPHA: float = 0.25
    """ Alpha value for outline visualization """

    SURFACE_RES: int = 100
    """Surface sampling count in each dimension"""

    D_VIS: float = 0.125
    """Visualization thickness for side view of planar elements"""

    MAX_RAYS_SHOWN: int = 10000
    """Maximum of rays shown in visualization"""
   
    TEXT_STYLE: dict = dict(font_size=11, color=(1, 1, 1), font_family="courier", shadow=True)
    """Standard Text Style. Used for object labels, legends and axes"""

    INFO_STYLE: dict = dict(font_size=13, bold=True, color=(1, 1, 1), font_family="courier", shadow=True)
    """Info Text Style. Used for status messages and interaction overlay"""

    SUBTLE_INFO_STYLE: dict = dict(font_size=13, bold=False, color=(0.65, 0.65, 0.65), font_family="courier", shadow=True)
    """Style for hidden info text. The color is used for the refraction index boxes frame"""

    ##########
    # The mayavi scene.
    scene: Instance = Instance(MlabSceneModel, args=())
    """the mayavi scene"""

    # Ranges

    Rays: Range = Range(5, 10000000, 200000, desc='Number of Rays for Simulation', enter_set=True,
                 auto_set=False, label="N_rays", mode='text')
    """Number of rays for raytracing"""

    Rays_s: Range = Range(-5.0, -1.0, -2, desc='Percentage of N Which is Drawn', enter_set=True,
                   auto_set=False, label="N_vis (log)", mode='slider')
    """Number of rays shown in mayavi scenen. log10 value."""

    Pos_Det: Range = Range(low='Pos_Det_min', high='Pos_Det_max', value='Pos_Det_max', mode='slider',
                    desc='z-Position of the Detector', enter_set=True, auto_set=True, label="z_pos")
    """z-Position of Detector. Lies inside z-range of :obj:`Backend.Raytracer.outline`"""

    Ray_alpha: Range = Range(-5.0, 0.0, np.log10(RAY_ALPHA), desc='Opacity of the Rays/Points', enter_set=True,
                      auto_set=True, label="Alpha (log)", mode='slider')
    """opacity of shown rays. log10 values"""

    Ray_width: Range = Range(1.0, 20.0, 1, desc='Ray Linewidth or Point Size', enter_set=True,
                      auto_set=True, label="Width")
    """Width of rays shown."""

    ImagePixels: Range = Range(1, 1000, 100, desc='Detector Image Pixels in Smaller of x or y Dimension', enter_set=True,
                        auto_set=True, label="Pixels_xy", mode='text')
    """Image Pixel value for Source/Detector Image. This the number of pixels for the smaller image side."""

    # Checklists (with capitalization workaround from https://stackoverflow.com/a/23783351)

    AbsorbMissing: List = List(editor=CheckListEditor(values=["Absorb Rays Missing Lens"], format_func=lambda x: x),
                         desc="if Rays are Absorbed when not Hitting a Lens")
    """Boolean value for absorbing rays missing Lens. Packed as Checkbox into a :obj:`List`"""

    LogImage: List = List(editor=CheckListEditor(values=['Logarithmic Scaling'], format_func=lambda x: x),
                    desc="if Logarithmic Values are Plotted")
    """Boolean value for a logarithmic image visualization. Packed as Checkbox into a :obj:`List`"""

    FlipImage: List = List(editor=CheckListEditor(values=['Flip Image'], format_func=lambda x: x),
                    desc="if imaged should be rotated by 180 degrees")
    """Boolean value for flipping the image (rotating it by 180°). Packed as Checkbox into a :obj:`List`"""

    FocusDebugPlot = List(editor=CheckListEditor(values=['Show Cost Function'], format_func=lambda x: x),
                    desc="if cost function is shown")

    # Enums

    PlottingType: Enum = Enum('Rays', 'Points', 'None', desc="Ray Representation")
    """Ray Plotting Type. One of ['Rays', 'Points', 'None']"""

    ColoringType: Enum = Enum('White', 'Power', 'Wavelength', 'Polarization', 'Source', desc="Ray Property to Color the Rays With")
    """Ray Coloring Mode. One of ['White', 'Power', 'Wavelength', 'Polarization', 'Source']"""

    ImageType: Enum = Enum('Irradiance', 'Illuminance', 'sRGB', desc="Image Type Presented")
    """Image Type. One of ['Irradiance', 'Illuminance', 'sRGB']"""

    FocusType: Enum = Enum('Position Variance', 'Airy Disc Weighting', 'Irradiance Variance', desc="Method for Autofocus")
    """Focus Finding Mode. One of ['Position Variance', 'Airy Disc Weighting', 'Irradiance Variance']"""

    SourceSelection: Enum = Enum(values='SourceNames', desc="Source Selection for Source Image")
    """Source Selection. Holds the name of one of the Sources."""

    DetectorSelection: Enum = Enum(values='DetectorNames', desc="Detector Selection")
    """Detector Selection. Holds the name of one of the Detectors."""

    # Buttons
    DetectorImageButton: Button = Button(label="Show Detector Image", desc="Generating a Detector Image")
    SourceImageButton:   Button = Button(label="Show Source Image", desc="Generating a Source Image of the Chosen Source")
    AutoFocusButton:     Button = Button(label="Find Focus", desc="Finding the Focus Between the Lenses in Front and Behind the Detector")

    # Labels
    Whitespace_Label    = Str('\n')
    Separator           = Item("Whitespace_Label", style='readonly', show_label=False)

    Status: Dict=Dict(Str())
    """ 
    Status dictionary. Consisting of key string and bool value.
         * InitScene: Set when Initializing mlab scene. Excludes displaying of rays and initial tracing
         * DisplayingGUI: Set if pyqt GUI is not fully loaded
         * Drawing: Set if Rays/Points/Detector etc. is updated
         * Tracing, Focussing, DetectorImage, SourceImage: Set if operation is active 
    """

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
    """the UI view"""


    ####################################################################################################################
    # Class constructor

    def __init__(self, RT: Raytracer, AbsorbMissing: bool=True, silent: bool=False, *args, **kwargs) -> None:
        """

        :param RT:
        :param args:
        :param kwargs:
        """
        self.Raytracer: Backend.Raytracer = RT
        """reference to the Raytracer"""

        # ray properties
        self.RaysPlotScatter = None  # plot for scalar ray values
        self.RaysPlot = None  # plot for visualization of rays/points
        self.RayText = None  # textbox for ray information
        self.subset = np.array([])  # indices for subset of all rays in Raytracer

        # Detector properties
        self.DetectorPlot = []  # visual representation of the Detector
        self.DetectorCylinderPlot = [] # visual representation of the Detectors' Cylinder
        self.DetText = []
        self.SourceNames = []
        self.DetectorNames = []
        self.DetInd = 0
       
        self.RaySourcePlots = []

        # minimal/maximal z-positions of Detector_obj
        self.Pos_Det_min = self.Raytracer.outline[4]
        self.Pos_Det_max = self.Raytracer.outline[5]

        # holds scene size, gets set on scene.activated
        self.scene_size = [0, 0]

        # hold axes objects
        self.axes = [None, None, None]

        # shift press indicator
        self.ShiftPressed = False

        # workaround to exit trait functions
        self.set_only = False
        
        # show info and warning messages
        self.silent = silent

        # set AbsorbMissing parameter from bool to list
        self.AbsorbMissing = self.AbsorbMissing if AbsorbMissing else []

        # define Status dict
        self.Status.update(dict(InitScene=True, DisplayingGUI=True, Tracing=False, Drawing=False, Focussing=False,
                                DetectorImage=False, SourceImage=False))

        self.threads = []

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
        alpha = 0.1 + 0.899*Fcolor[1]  # offset in both directions, ensures visibility and see-through
        self.plotSObject(filter_, num, "Filter", "F", Fcolor[0], alpha)    

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

        a, b, _, _ = self.plotSObject(RS, num, "RaySource", "RS", (1, 1, 1), self.RAYSOURCE_ALPHA, 
                                     d=-self.D_VIS, spec=False, light=False)   

        self.RaySourcePlots.append((a, b))

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
        self.oaxes.text_property.trait_set(**self.TEXT_STYLE)
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

        def drawAxis(objs, ind: int, ext: list, lnum: int, name: str, lform: str, vis_x: bool, vis_y: bool, vis_z: bool):

            self.scene.engine.add_source(ParametricSurface(name=f"{name}-Axis"), self.scene)

            objs[ind] = self.scene.mlab.axes(extent=ext, nb_labels=lnum, x_axis_visibility=vis_x, 
                                       y_axis_visibility=vis_y, z_axis_visibility=vis_z)

            label=f"{name} / mm"
            objs[ind].axes.trait_set(font_factor=0.7, fly_mode='none', label_format=lform, x_label=label, 
                               y_label=label, z_label=label, layer_number=1)

            objs[ind].property.trait_set(display_location='background', opacity=0.5)
            objs[ind].title_text_property.trait_set(**self.TEXT_STYLE)
            objs[ind].label_text_property.trait_set(**self.TEXT_STYLE)
            objs[ind].actors[0].pickable = False

        # place axes at outline
        ext = self.Raytracer.outline

        # enforce placement of x- and z-axis at y=ys (=RT.outline[2]) by shrinking extent
        ext_ys = ext.copy()
        ext_ys[3] = ext_ys[2]

        # X-Axis
        lnum = getLabelNum(ext[1] - ext[0], 5, 16)
        drawAxis(self.axes, 0, ext_ys, lnum, "x", '%-#.4g', True, False, False)

        # Y-Axis
        lnum = getLabelNum(ext[3] - ext[2], 5, 16)
        drawAxis(self.axes, 1, ext.copy(), lnum, "y", '%-#.4g', False, True, False)

        # Z-Axis
        lnum = getLabelNum(ext[5] - ext[4], 5, 24)
        drawAxis(self.axes, 2, ext_ys, lnum, "z", '%-#.5g', False, False, True)

    
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
            text.property.trait_set(**self.TEXT_STYLE, justification="center", frame=True, frame_color=self.SUBTLE_INFO_STYLE["color"])

    def plotSObject(self, obj, num, name, shortname, color, alpha, d=D_VIS, spec=True, light=True):
        """plotting of a SObject. Gets called from plotting for Lens, Filter, Detector, RaySource."""

        def plot(C, surf_type):
            # plot surface
            a = self.scene.mlab.mesh(C[0], C[1], C[2], color=color, opacity=alpha, name=f"{name} {num} {surf_type} surface")

            # make non pickable so it does not interfere with our ray picker
            a.actor.actor.pickable = False

            a.actor.actor.property.lighting = light
            if spec:
                a.actor.property.trait_set(specular=0.5, ambient=0.25)
            return a

        # decide what to plot
        plotFront = obj.FrontSurface.surface_type not in ["Point", "Line"]
        plotCyl = obj.hasBackSurface() or obj.Surface.isPlanar()
        plotBack  = obj.BackSurface is not None and obj.BackSurface.surface_type not in ["Point", "Line"]

        a = plot(obj.FrontSurface.getPlottingMesh(N=self.SURFACE_RES), "front") if plotFront else None

        # if surface is a plane or object a lens, plot a cylinder for side viewing
        b = plot(obj.getCylinderSurface(nc=self.SURFACE_RES, d=d), "cylinder") if plotCyl else None

        # calculate middle center z-position
        zl = (obj.extent[4] + obj.extent[5])/2 if not obj.hasBackSurface() else obj.pos[2]
        zl = zl + self.D_VIS/2 if obj.Surface.isPlanar() and not obj.hasBackSurface() else zl

        # object label
        text = self.scene.mlab.text(x=obj.extent[1], y=0., z=zl, text=f"{shortname}{num}", name=f"Label")
        text.property.trait_set(**self.TEXT_STYLE, justification="center")
        text.actor.text_scale_mode = 'none'
       
        # plot BackSurface if one exists
        c = plot(obj.BackSurface.getPlottingMesh(N=self.SURFACE_RES), "back") if plotBack else None

        return a, b, c, text

    def plotRays(self) -> None:
        """Ray/Point Plotting with a specified scalar coloring"""
        p_, _, pol_, w_, wl_, snum_ = self.Raytracer.Rays.getRaysByMask(self.subset, 
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
        lutm.label_text_property.trait_set(**self.TEXT_STYLE)
        lutm.title_text_property.trait_set(**self.INFO_STYLE)

        # lut position and size
        lutm.scalar_bar_representation.position         = np.array([0.9, 0.1])
        lutm.scalar_bar_representation.position2        = np.array([0.09, 0.8])
        lutm.scalar_bar_widget.process_events           = False  # make non-interactive
        lutm.scalar_bar_representation.border_thickness = 0  # no ugly borders 

        if len(self.Raytracer.RaySourceList) != len(self.RaySourcePlots):
            raise RuntimeError("Number of RaySourcePlots differs from actual Sources. Maybe the GUI was not updated properly?")
        
        # custom lut for wavelengths
        match self.ColoringType:

            case "White":
                RSColor = [(1, 1, 1) for RSp in self.RaySourcePlots]

            case 'Wavelength':
                lutm.lut.table = Color.spectralCM(255)
                lutm.data_range = [Color.WL_MIN, Color.WL_MAX]
                lutm.number_of_labels = 11

                RSColor = [RS.getColor() for RS in self.Raytracer.RaySourceList]

            case 'Polarization':
                if self.Raytracer.no_pol:
                    raise RuntimeError("WARNING: Polarization calculation turned off in Raytracer.")

                lrange = lutm.data_range[1]-lutm.data_range[0]

                if lrange > 0.8 and 1 - lutm.data_range[1] < 0.05:
                    lutm.data_range = [lutm.data_range[0], 1]

                if lutm.data_range[0]/lrange < 0.05: 
                    lutm.data_range = [0, lutm.data_range[1]]

                lutm.number_of_labels = 11
          
                # Color from polarization projection on yz-plane
                RSColor = []
                for RS in self.Raytracer.RaySourceList:
                    match RS.polarization_type:
                        case "x":
                            col = lutm.lut.table[0]
                        case "y":
                            col = lutm.lut.table[-1]
                        case "Angle":
                            yzpr = (np.sin(np.deg2rad(RS.pol_ang)) - lutm.data_range[0]) / lrange
                            col = lutm.lut.table[int(yzpr*255)]
                        case _:  # no single axis -> show white
                            col = (255, 255, 255)

                    RSColor.append(np.array(col[:3])/255.)

            case 'Source':
                lutm.number_of_labels = len(self.Raytracer.RaySourceList)
                lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-#6.f"
                if lutm.number_of_labels > 1:
                    # set color steps, smaller 420-650nm range for  larger color contrast
                    lutm.lut.table = Color.spectralCM(lutm.number_of_labels, 420, 650)
        
                RSColor = [np.array(lutm.lut.table[i][:3])/255. for i, _ in enumerate(self.RaySourcePlots)]

            case 'Power':
                if lutm.data_range[0]/(lutm.data_range[1]-lutm.data_range[0]) < 0.05: 
                    lutm.data_range = [0, lutm.data_range[1]]

                # set to maximum ray power, this is the same for all sources
                RSColor = [np.array(lutm.lut.table[-1][:3])/255. for RSp in self.RaySourcePlots]

        # set RaySource Colors
        for i, RSp in enumerate(self.RaySourcePlots):
            for RSi in RSp:
                if RSi:
                    RSi.actor.actor.property.trait_set(color=tuple(RSColor[i]))

        # set visual ray properties
        self.RaysPlot.actor.property.representation = 'points' if self.PlottingType == 'Points' else 'surface'
        self.RaysPlot.actor.property.trait_set(line_width=self.Ray_width, point_size=self.Ray_width)
        self.RaysPlot.actor.actor.force_translucent = True

    def plotGeometry(self) -> None:
        """plot Background, Objects (Lenses, RaySources, Detectors, Filters), Outlines and Axes"""
        
        # plot SObjects
        [self.plotRaySource(RS, num+1) for num, RS in enumerate(self.Raytracer.RaySourceList)]
        [self.plotLens(L, num+1) for num, L in enumerate(self.Raytracer.LensList)]
        [self.plotFilter(F, num+1) for num, F in enumerate(self.Raytracer.FilterList)]
        [self.plotDetector(D, num+1) for num, D in enumerate(self.Raytracer.DetectorList)]

        # assign detector position
        if self.Raytracer.DetectorList:
            self.Pos_Det = [self.Raytracer.DetectorList[0].pos[2]]
        else:
            self.Pos_Det = self.Raytracer.outline[5]

        # plot axes and outlines
        self.plotAxes()
        self.plotOrientationAxes()
        self.plotOutline()
        self.plotRefractionIndexBoxes()

        # set background color
        self.scene.background = self.BACKGROUND_COLOR

    def initRayInfoText(self) -> None:
        """init detection of ray point clicks and the info display"""
        # init point pickers
        self.scene.mlab.gcf().on_mouse_pick(self.onRayPick, button='Left')
        self.scene.mlab.gcf().on_mouse_pick(self.onSpacePick, button='Right')

        # add ray info text
        self.RayTextParent = self.scene.engine.add_source(ParametricSurface(name="Ray Info Text"), self.scene)
        self.RayText = self.scene.mlab.text(0.02, 0.97, "") 

    def initStatusText(self) -> None:
        """init GUI status text display"""
        # add status text
        self.StatusTextParent = self.scene.engine.add_source(ParametricSurface(name="Status Info Text"), self.scene)
        self.StatusText = self.scene.mlab.text(0.97, 0.01, "Status Text")
        self.StatusText.property.trait_set(**self.INFO_STYLE, justification="right")
        self.StatusText.actor.text_scale_mode = 'none'
    
    def initSourceList(self) -> None:
        """generate a descriptive list of RaySource names"""
        self.SourceNames = [f"RS{num+1}: {RS.Surface.surface_type} at {RS.pos}"
                            for num, RS in enumerate(self.Raytracer.RaySourceList) ]

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
            self.RayText.property.trait_set(**self.INFO_STYLE, background_opacity=0.2)

    def onRayPick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker'=None) -> None:
        """
        Ray Picking Handler.
        Shows ray properties on screen.
        :param picker_obj:
        """

        # it seems that picker_obj.point_id is only present for the first element in the picked list,
        # so we only can use it when the RayPlot is first in the list
        # see https://github.com/enthought/mayavi/issues/906
        if picker_obj is not None and len(picker_obj.actors) != 0 \
           and picker_obj.actors[0]._vtk_obj is self.RaysPlot.actor.actor._vtk_obj:
            
            a = self.Raytracer.Rays.nt # number of points per ray plotted
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
            p_, s_, pols_, pw_, wv, snum = self.Raytracer.Rays.getRay(pos)
            p, s, pols, pw = p_[n2_], s_[n2_], pols_[n2_], pw_[n2_]

            pw0 = pw_[n2_-1] if n2_ > 0 else None
            s0 = s_[n2_-1] if n2_ > 0 else None
            pols0 = pols_[n2_-1] if n2_ > 0 else None
            pl = (pw0-pw)/pw0 if n2_ > 0 else None  # power loss

            sh = self.ShiftPressed

            def toSphCoords(s):
                return np.array([np.rad2deg(np.arctan2(s[1], s[0])), np.rad2deg(np.arccos(s[2])), 1])

            s_sph = toSphCoords(s)
            s0_sph = toSphCoords(s0) if n2_ > 0 else None
            pols_sph = toSphCoords(pols)
            pols0_sph = toSphCoords(pols0) if n2_ > 0 else None

            # generate text
            text =  f"Ray {pos}" +  \
                    f" from Source {snum}" +  \
                   (f" at surface {n2_}\n" if n2_> 0 else f" at ray source\n") + \
                    f"Position:             ({p[0]:>10.5g}, {p[1]:>10.5g}, {p[2]:>10.5g})\n" + \
                   (f"Direction Before:     ({s0[0]:>10.5f}, {s0[1]:>10.5f}, {s0[2]:>10.5f})" if n2_ and sh else "") + \
                   (f"      ({s0_sph[0]:>10.5f}°, {s0_sph[1]:>10.5f}°, {s0_sph[2]:>10.5f})\n" if n2_ and sh else "") + \
                    f"Direction After:      ({s[0]:>10.5f}, {s[1]:>10.5f}, {s[2]:>10.5f})" + \
                   (f"      ({s_sph[0]:>10.5f}°, {s_sph[1]:>10.5f}°, {s_sph[2]:>10.5f})\n" if sh else "\n") + \
                   (f"Polarization Before:  ({pols0[0]:>10.5f}, {pols0[1]:>10.5f}, {pols0[2]:>10.5f})" if n2_ and sh else "") + \
                   (f"      ({pols0_sph[0]:>10.5f}°, {pols0_sph[1]:>10.5f}°, {pols0_sph[2]:>10.5f})\n" if n2_ and sh else "") + \
                    f"Polarization After:   ({pols[0]:>10.5f}, {pols[1]:>10.5f}, {pols[2]:>10.5f})" + \
                   (f"      ({pols_sph[0]:>10.5f}°, {pols_sph[1]:>10.5f}°, {pols_sph[2]:>10.5f})\n" if sh else "\n") + \
                    f"Wavelength:            {wv:>10.2f} nm\n" + \
                   (f"Ray Power Before:      {pw0*1e6:>10.5g} µW\n" if n2_ > 0 and sh else "") + \
                    f"Ray Power After:       {pw*1e6:>10.5g} µW\n" + \
                   (f"Power Loss on Surface: {pl*100:>10.5g} %" if n2_ > 0 and sh else "") + \
                   (f"Pick using Shift+Left Mouse Button for more info" if not sh else "")
            
            self.RayText.text = text
            self.RayText.property.trait_set(**self.INFO_STYLE, background_opacity=0.2)
        else:
            self.RayText.text = "Left Click on a Ray-Surface Intersection to Show Ray Properties.\n" + \
                                "Right Click Anywhere to Show 3D Position.\n" + \
                                "Shift + Right Click to Move the Detector to this z-Position"
            self.RayText.property.trait_set(**self.SUBTLE_INFO_STYLE, background_opacity=0)

        # text settings
        self.RayText.property.trait_set(vertical_justification="top")
        self.RayText.actor.text_scale_mode = 'none'

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

    def setGUILoaded(self) -> None:
        """sets GUILoaded Status. Exits GUI if self.exit set."""

        self.Status["DisplayingGUI"] = False
        
        if self.exit:
            self.close()

    def hasDetector(self) -> bool:
        """
        Prints message if Detectors are missing.
        :return: if the Raytracer has Detectors
        """
        if not self.Raytracer.DetectorList:
            if not self.silent:
                print("Detector Missing, skipping this action.")
            return False
        return True

    def hasRaySource(self) -> bool:
        """
        Prints message if RaySources are missing.
        :return: if the Raytracer has RaySources
        """
        if not self.Raytracer.RaySourceList:
            if not self.silent:
                print("RaySource Missing, skipping this action.")
            return False
        return True

    def noRaceCondition(self) -> bool:
        """
        Prints a message if the raytracer is tracing and therefore a race condition would occur.
        :return: if the Raytracer is tracing
        """
        if self.Status["Tracing"]:
            self.StatusText.text += "Can't do this while raytracing.\n"
            if not self.silent:
                print("Raytracing in progress, skipping this action because of a race condition risk.")
            return False
        return True

    ###############################################################################################################
    # Interface functions
    ###############################################################################################################

    def replot(self):
        self.scene.scene.disable_render  = True
        self.Status["InitScene"] = True
        cc_traits_org = self.scene.camera.trait_get("position", "focal_point", "view_up", "view_angle",
                                                    "clipping_range", "parallel_scale")

        # ray properties
        self.RaysPlotScatter = None  # plot for scalar ray values
        self.RaysPlot = None  # plot for visualization of rays/points

        self.RaySourcePlots = []

        # Detector properties
        self.DetectorPlot = []  # visual representation of the Detector
        self.DetectorCylinderPlot = [] # visual representation of the Detectors' Cylinder
        self.DetText = []
        self.DetInd = 0
        
        # minimal/maximal z-positions of Detector_obj
        self.Pos_Det_min = self.Raytracer.outline[4]
        self.Pos_Det_max = self.Raytracer.outline[5]

        # we cant use mlab.clf() since the on_mouse_pick would be unusable
        # see https://stackoverflow.com/a/23476093
        # instead delete every object, but don't delete StatusText and RayText
        for child in self.scene.mayavi_scene.children.copy():
            obj = child.children[0].children[0]
            if obj not in [self.RayText, self.StatusText]:
                child.remove()
                pyfaceGUI().process_events()

        self.plotGeometry()

        self.initSourceList()
        self.initDetectorList()

        self.scene.camera.trait_set(**cc_traits_org)

        self.Status["InitScene"] = False

        self.setRays()
        
        self.scene.scene.disable_render = False
        self.scene.scene.render()
        pyfaceGUI().process_events()


    def close(self) -> None:
        """close the GUI after it is loaded"""

        pyfaceGUI().process_events()

        # close all ui frames
        app = QtGui.QApplication.instance()
        for widget in app.topLevelWidgets():
            widget.close()

        # exit application
        QtCore.QCoreApplication.quit()
        time.sleep(0.1)  # wait for it actually being closed

    def waitForIdle(self) -> None:
        """wait until the GUI is Idle. Only call this from another thread"""
       
        for th in self.threads:
            th.join()
        
        time.sleep(0.001)  # wait for flags to be set

        while True:
            busy = False
            [busy := True for key in self.Status.keys() if self.Status[key]]

            if not busy:
                break

            time.sleep(0.02)

        # fixes a pyface bug where the GUI is shown as busy
        # see https://github.com/enthought/mayavi/issues/59
        pyfaceGUI().process_events()
        pyfaceGUI().set_busy(False)

    # wait while 'key' in StatusList is set
    # def waitWhile(self, key: str):
        # """
        # wait while the GUI is busy doing 'key'.
        # Only call this from another thread.
        # :param key: string key from GUI.Status
        # """
        # time.sleep(0.001)  # wait for flags to be set
        # # wait for tasks to finish
        # while self.Status[key]:
            # time.sleep(0.02)

    def interact(self, func=None, args=tuple()):

        if func is None:
            def func(self):
                self.waitForIdle()
                while True:

                    cmd = input(">>")
                    
                    try:
                        exec(cmd)
                        self.waitForIdle()
                    except Exception:
                        traceback.print_exc()

            args=(self,)

        th = Thread(target=func, args=args)
        th.start()
        self()
        th.join()

    ####################################################################################################################
    # Trait handlers.

    @on_trait_change('Rays_s')
    def filterRays(self) -> None:
        """chose a subset of all Raytracer rays and plot them with :obj:`GUI.drawRays`"""

        if self.hasRaySource():
            # don's use self.Rays as ray number, since the raytracer could still be raytracing
            # instead use the last saved number
            N = self.Raytracer.Rays.N

            if self.PlottingType == "None" or N == 0:
                return

            # currently busy, settings will be applied after current task
            if self.Status["InitScene"] or self.Status["Tracing"]:
                return

            self.Status["Drawing"] = True

            set_size = int(1 + 10**self.Rays_s*N)  # number of shown rays
            sindex = np.random.choice(N, size=set_size, replace=False)  # random choice

            # make bool array with chosen rays set to true
            self.subset = np.zeros(N, dtype=bool)
            self.subset[sindex] = True

            self.drawRays()

            self.Status["Drawing"] = False

    @on_trait_change('ColoringType')
    def drawRays(self) -> None:
        """remove old ray view, call :obj:`GUI.plotRays` and restore scene view"""

        if self.hasRaySource():

            # currently busy, but next plotting will use the current settings
            if self.Status["InitScene"] or self.Status["Tracing"]:
                return

            # only draw if source has rays and scene renderer is initialized
            elif self.Raytracer.Rays.hasRays() and not self.Status["InitScene"]:

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
        if self.set_only or self.Status["InitScene"]:
            return

        if self.hasRaySource():

            for key in self.Status.keys():
                if self.Status[key] and key != "DisplayingGUI":
                    self.StatusText.text += "Can't trace while other background tasks are running.\n"
                    return

            self.Status["Tracing"] = True

            # run this in background thread
            def background(RT: Raytracer, N: int) -> None:
                RT.AbsorbMissing = self.AbsorbMissing
                RT.trace(N=N)

                # execute this after thread has finished
                def on_finish() -> None:

                    self.Status["Tracing"] = False

                    # lower the ratio of rays shown so that the GUI does not become to slow
                    if 10**self.Rays_s*self.Rays > self.MAX_RAYS_SHOWN:
                        self.Rays_s = np.log10(self.MAX_RAYS_SHOWN/self.Rays)
                    else: # in the other case self.filterRays() is called because of the self.Rays_s change
                        self.filterRays()

                    # set parameters to current value, since they could be changed by the user while tracing
                    self.set_only = True
                    self.AbsorbMissing = ["Absorb Rays Missing Lens"] if self.Raytracer.AbsorbMissing else []
                    self.Rays = self.Raytracer.Rays.N
                    self.set_only = False

                    pyfaceGUI.invoke_later(self.setGUILoaded)

                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background, args=(self.Raytracer, self.Rays))
            action.start()
            self.threads.append(action)
        else:
            pyfaceGUI.invoke_later(self.setGUILoaded)
             
    @on_trait_change('Ray_alpha')
    def setRayAlpha(self) -> None:
        """change opacity of visible rays"""

        if self.hasRaySource() and self.RaysPlot:
            self.Status["Drawing"] = True
            self.RaysPlot.actor.property.trait_set(opacity=10**self.Ray_alpha)
            self.Status["Drawing"] = False

    @on_trait_change('Pos_Det')
    def setDetector(self) -> None:
        """move and replot chosen Detector"""

        if self.hasDetector() and self.DetectorPlot and not self.set_only:

            xp, yp, zp = self.Raytracer.DetectorList[self.DetInd].pos
            self.Raytracer.DetectorList[self.DetInd].moveTo([xp, yp, self.Pos_Det])
            
            Surf = self.Raytracer.DetectorList[self.DetInd].Surface.getPlottingMesh(N=self.SURFACE_RES)[2]
            self.DetectorPlot[self.DetInd].mlab_source.trait_set(z=Surf)

            if self.DetectorCylinderPlot[self.DetInd]:
                Cyl = self.Raytracer.DetectorList[self.DetInd].getCylinderSurface(nc=self.SURFACE_RES)[2] 
                self.DetectorCylinderPlot[self.DetInd].mlab_source.trait_set(z=Cyl)

            self.DetText[self.DetInd].z_position = self.Pos_Det

    @on_trait_change('DetectorImageButton')
    def showDetectorImage(self) -> None:
        """render a DetectorImage at the chosen Detector, uses a separate thread"""

        if self.hasDetector() and self.noRaceCondition():

            self.Status["DetectorImage"] = True

            def background(RT: Raytracer, N: int) -> None:
                res = RT.DetectorImage(N=N, ind=self.DetInd)

                def on_finish() -> None:
                    DetectorPlot(res, log=self.LogImage, flip=self.FlipImage, mode=self.ImageType)
                    self.Status["DetectorImage"] = False
                
                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background, args=(self.Raytracer, self.ImagePixels))
            action.start()
            self.threads.append(action)

    @on_trait_change('SourceImageButton')
    def showSourceImage(self) -> None:
        """render a SourceImage for the chosen Source, uses a separate thread"""

        if self.hasRaySource() and self.noRaceCondition():
            # surface number for selected source
            snum = int(self.SourceSelection.split(":", 1)[0].split("RS")[1])

            self.Status["SourceImage"] = True

            def background(RT: Raytracer, N: int, snum: int) -> None:
                res = RT.SourceImage(N=N, sindex=snum-1)

                def on_finish() -> None:
                    SourcePlot(res, log=self.LogImage, flip=self.FlipImage, mode=self.ImageType)
                    self.Status["SourceImage"] = False
                
                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background, args=(self.Raytracer, self.ImagePixels, snum))
            action.start()
            self.threads.append(action)

    @on_trait_change('AutoFocusButton')
    def moveToFocus(self) -> None:
        """
        Find a Focus.
        The chosen Detector defines the search range for focus finding. 
        Searches are always between lenses aor the next outline. 
        Search takes place in a separate thread, after that the Detector is moved to the focus 
        """
        if self.hasDetector() and self.hasRaySource() and self.noRaceCondition():
                
            self.Status["Focussing"] = True

            # run this in background thread
            def background(RT: Raytracer, z: float, method: str) -> None:
                zf, zff, r, vals = RT.autofocus(method, z, ret_cost=self.FocusDebugPlot)
                RT.DetectorList[self.DetInd].moveTo([*RT.DetectorList[self.DetInd].pos[:2], zf])

                # execute this function after thread has finished
                def on_finish() -> None:
                    if self.FocusDebugPlot:
                        AutoFocusDebugPlot(r, vals, zf, zff, f"Focus Finding Using the {method} Method")
                    self.Pos_Det = self.Raytracer.DetectorList[self.DetInd].pos[2]  # setDetector() is called
                    self.Status["Focussing"] = False
                
                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background, args=(self.Raytracer, self.Pos_Det, self.FocusType))
            action.start()
            self.threads.append(action)

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

        self.Status["InitScene"] = False

        # set absorb rays missing to true by default, side effect is calling setRays()
        self.AbsorbMissing = ["Absorb Rays Missing Lens"]
        # self.setRays()


    @on_trait_change('Status[]')
    def changeStatus(self) -> None:
        """Update the status info text."""

        if self.Status["InitScene"]:
            return

        text = ""
        if self.Status["Tracing"]:          text += "Raytracing...\n"
        if self.Status["Focussing"]:        text += "Focussing...\n"
        if self.Status["DetectorImage"]:    text += "Generating Detector Image...\n"
        if self.Status["SourceImage"]:      text += "Generating Source Image...\n"
        if self.Status["Drawing"]:          text += "Drawing Rays...\n"

        self.StatusText.text = text

    @on_trait_change('Ray_width')
    def changeRayWidth(self) -> None:
        """ set the ray width for the visible rays."""

        if self.hasRaySource() and self.RaysPlot:
            self.Status["Drawing"] = True
            self.RaysPlot.actor.property.trait_set(line_width=self.Ray_width, point_size=self.Ray_width)
            self.Status["Drawing"] = False


    @on_trait_change('PlottingType')
    def changePlottingType(self) -> None:
        """Change the PlottingType. Removes the visual rays/points and replots using :obj:`GUI.drawRays`"""

        if self.hasRaySource():
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
        if self.hasDetector():
            # surface number for selected source
            self.DetInd = int(self.DetectorSelection.split(":", 1)[0].split("DET")[1]) - 1
            self.set_only = True
            self.Pos_Det = self.Raytracer.DetectorList[self.DetInd].pos[2]
            self.set_only = False

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
            self.axes[0].axes.trait_set(font_factor=self.axes[0].axes.font_factor*ch1/ch2)
            self.axes[1].axes.trait_set(font_factor=self.axes[1].axes.font_factor*ch1/ch2)
            self.axes[2].axes.trait_set(font_factor=self.axes[2].axes.font_factor*ch1/ch2)

            # rescale orientation axes
            self.oaxes.widgets[0].zoom = self.oaxes.widgets[0].zoom * ch1/ch2

            # set current window size
            self.scene_size = scene_size

