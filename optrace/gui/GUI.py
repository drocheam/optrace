

"""
Frontend for the Raytracer class.
Shows a 3D representation of the geometry that is navigable.
Allows for the recalculation of the rays as well as visual settings for rays or points.
The detector is movable along the optical axis.
Interaction to source image and detector image plotting and its properties.
Autofocus functionality with different modes.

"""

import wx

import sys
import time
import numpy as np
from threading import Thread
from typing import Callable

from pynput.keyboard import Key, Listener

from pyface.qt import QtCore, QtGui
from pyface.api import GUI as pyfaceGUI
from traitsui.api import View, Item, HSplit, Group, CheckListEditor
from traits.api import HasTraits, Range, Instance, on_trait_change, Str, Button, Enum, List, Dict

from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.modules.surface import Surface as mayaviSurface
from mayavi.sources.parametric_surface import ParametricSurface

from optrace.tracer import *
from optrace.tracer.Misc import timer as timer
from optrace.plots.ImagePlots import ImagePlot
from optrace.plots.DebugPlots import AutoFocusDebugPlot

import optrace.gui.TCPServer as TCPServer
from twisted.internet import reactor
from twisted.python import log, util


class GUI(HasTraits):

    ####################################################################################################################
    # UI objects

    DETECTOR_COLOR: tuple[float, float, float] = (0.10, 0.10, 0.10, 0.80) 
    """RGB + Alpha tuple for the Detector visualization"""

    LENS_COLOR: tuple[float, float, float] = (0.63, 0.79, 1.00, 0.30)
    """RGB + Alpha Tuple for the Lens surface visualization"""

    BACKGROUND_COLOR: tuple[float, float, float] = (0.32, 0.30, 0.30)
    """RGB Color for the Scene background"""
    
    RAYSOURCE_ALPHA: float = 0.5
    """Alpha value for the RaySource visualization"""

    OUTLINE_ALPHA: float = 0.25
    """ Alpha value for outline visualization """

    SURFACE_RES: int = 100
    """Surface sampling count in each dimension"""

    D_VIS: float = 0.125
    """Visualization thickness for side view of planar elements"""

    MAX_RAYS_SHOWN: int = 10000
    """Maximum of rays shown in visualization"""
   
    LABEL_STYLE: dict = dict(font_size=11, color=(1, 1, 1), bold=True, font_family="courier", shadow=True)
    """Standard Text Style. Used for object labels, legends and axes"""
    
    TEXT_STYLE: dict = dict(font_size=11, color=(1, 1, 1), font_family="courier", shadow=True, italic=False, bold=False)
    """Standard Text Style. Used for object labels, legends and axes"""

    INFO_STYLE: dict = dict(font_size=13, bold=True, color=(1, 1, 1), font_family="courier", shadow=True, italic=False)
    """Info Text Style. Used for status messages and interaction overlay"""

    SUBTLE_INFO_STYLE: dict = dict(font_size=13, bold=False, color=(0.501, 0.501, 0.501), font_family="courier", shadow=True)
    """Style for hidden info text. The color is used for the refraction index boxes frame"""
    
    AXIS_STYLE: dict = dict(font_size=11, bold=False, italic=False, color=(1, 1, 1), font_family="courier", shadow=True)
    
    ##########
    Scene: Instance = Instance(MlabSceneModel, args=())
    """the mayavi scene"""

    # Ranges

    RayCount: Range = Range(5, 10000000, 200000, desc='Number of Rays for Simulation', enter_set=True,
                 auto_set=False, label="N_rays", mode='text')
    """Number of rays for raytracing"""

    RayAmountShown: Range = Range(-5.0, -1.0, -2, desc='Percentage of N Which is Drawn', enter_set=True,
                   auto_set=False, label="N_vis (log)", mode='slider')
    """Number of rays shown in mayavi scenen. log10 value."""

    PosDet: Range = Range(low='PosDetMin', high='PosDetMax', value='PosDetMax', mode='slider',
                    desc='z-Position of the Detector', enter_set=True, auto_set=True, label="z_pos")
    """z-Position of Detector. Lies inside z-range of :obj:`Backend.Raytracer.outline`"""

    RayAlpha: Range = Range(-5.0, 0.0, -2.0, desc='Opacity of the Rays/Points', enter_set=True,
                      auto_set=True, label="Alpha (log)", mode='slider')
    """opacity of shown rays. log10 values"""

    RayWidth: Range = Range(1.0, 20.0, 1, desc='Ray Linewidth or Point Size', enter_set=True,
                      auto_set=True, label="Width")
    """Width of rays shown."""

    ImagePixels: Range = Range(1, 1000, 100, desc='Detector Image Pixels in Smaller of x or y Dimension', enter_set=True,
                        auto_set=True, label="Pixels_xy", mode='text')
    """Image Pixel value for Source/Detector Image. This the number of pixels for the smaller image side."""

    # Checklists (with capitalization workaround from https://stackoverflow.com/a/23783351)
    # this should basically be bool values, but because we want to have checkboxes with text right of them
    # we need to embed a CheckListEditor in a List. 
    # To assign bool values we need the workaround function self.__CheckValFromBool

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

    CleanerView = List(editor=CheckListEditor(values=['Cleaner Scene View'], format_func=lambda x: x),
                    desc="if some scene elements should be hidden")

    AFOneSource = List(editor=CheckListEditor(values=['Rays From Selected Source Only'], format_func=lambda x: x),
                    desc="if autofocus only uses currently selected source")
    
    DetImageOneSource = List(editor=CheckListEditor(values=['Rays From Selected Source Only'], format_func=lambda x: x),
                    desc="if DetectorImage only uses currently selected source")
    # Enums

    PlottingTypes: list = ['Rays', 'Points']
    PlottingType: Enum = Enum(*PlottingTypes, desc="Ray Representation")
    """Ray Plotting Type"""

    ColoringTypes: list = ['White', 'Power', 'Wavelength', 'Polarization', 'Source']
    ColoringType: Enum = Enum(*ColoringTypes, desc="Ray Property to Color the Rays With")
    """Ray Coloring Mode"""

    ImageType: Enum = Enum(*Image.display_modes, desc="Image Type Presented")
    """Image Type"""

    FocusType: Enum = Enum(*Raytracer.AutofocusModes, desc="Method for Autofocus")
    """Focus Finding Mode from Raytracer.AutofocusModes"""

    SourceNames = List()
    SourceSelection: Enum = Enum(values='SourceNames', desc="Source Selection for Source Image")
    """Source Selection. Holds the name of one of the Sources."""

    DetectorNames = List()
    DetectorSelection: Enum = Enum(values='DetectorNames', desc="Detector Selection")
    """Detector Selection. Holds the name of one of the Detectors."""

    # Buttons
    DetectorImageButton: Button = Button(label="Show Detector Image", desc="Generating a Detector Image")
    SourceImageButton:   Button = Button(label="Show Source Image", desc="Generating a Source Image of the Chosen Source")
    AutoFocusButton:     Button = Button(label="Find Focus", desc="Finding the Focus Between the Lenses in Front and Behind the Detector")

    # Labels
    WhitespaceLabel     = Str('')
    Separator           = Item("WhitespaceLabel", style='readonly', show_label=False)

    Status: Dict = Dict(Str())
    """ 
    Status dictionary. Consisting of key string and bool value.
         * InitScene: Set when Initializing mlab scene. Excludes displaying of rays and initial tracing
         * DisplayingGUI: Set if pyqt GUI is not fully loaded
         * Drawing: Set if Rays/Points/Detector etc. is updated
         * Tracing, Focussing, DetectorImage, SourceImage: Set if operation is active 
    """

    # wx App. Needed for the tcp server
    App = wx.App(False)

    # size of the mlab scene
    SceneSize0 = [1150, 850] # default size
    # with the window bar this should still fit on a 900p screen
    SceneSize = SceneSize0.copy() # will hold current size
    
    ####################################################################################################################
    # UI view creation

    view = View(
                HSplit(
                    # add mayavi scene
                    Group(
                        Item('Scene', editor=SceneEditor(scene_class=MayaviScene),
                             height=SceneSize[1], width=SceneSize[0], show_label=False)
                        ),
                    # add UI Elements
                    Group(
                        Separator,
                        Item('RayCount'),
                        Item('AbsorbMissing', style="custom", show_label=False),
                        Separator,
                        Item("PlottingType", label='Plotting'),
                        Item("ColoringType", label='Coloring'),
                        Item('RayAmountShown'),
                        Item('RayAlpha'),
                        Item('RayWidth'),
                        Item('CleanerView', style="custom", show_label=False),
                        Separator,
                        Item('SourceSelection', label="Source"),
                        Item('DetectorSelection', label="Detector"),
                        Item('PosDet', label="Det_z"),
                        Separator,
                        Item('FocusType', label='AF Mode'),
                        Item('AFOneSource', style="custom", show_label=False),
                        Item('FocusDebugPlot', style="custom", show_label=False),
                        Item('AutoFocusButton', show_label=False),
                        Separator,
                        Item('ImageType', label='Image'),
                        Item('ImagePixels'),
                        Item('LogImage', style="custom", show_label=False),
                        Item('FlipImage', style="custom", show_label=False),
                        Item('DetImageOneSource', style="custom", show_label=False),
                        Item('SourceImageButton', show_label=False),
                        Item('DetectorImageButton', show_label=False),
                        ),
                    # add spacing right of UI Elements
                    Group(
                        Item("WhitespaceLabel", style='readonly', show_label=False, width=20)
                        )
                    ),
                resizable=True
                )
    """the UI view"""


    ####################################################################################################################
    # Class constructor

    def __init__(self, RT: Raytracer, AbsorbMissing: bool=True, CleanerView: bool=False, LogImage: bool=False,
                 FlipImage: bool=False, FocusDebugPlot: bool=False, AFOneSource: bool=False,
                 DetImageOneSource: bool=False, **kwargs) -> None:
        """
        The extra bool parameters are needed to assign the List-Checklisteditor to bool values in the GUI.__init__. 
        Otherwise the user would assign bool values to a string list.

        :param RT:
        :param args:
        :param kwargs:
        """

        self.Raytracer: Backend.Raytracer = RT
        """reference to the Raytracer"""

        # ray properties
        self.RaySelection = np.array([])  # indices for subset of all rays in Raytracer
        self.RaysPlotScatter = None  # plot for scalar ray values
        self.RaysPlot = None  # plot for visualization of rays/points
        self.RayText = None  # textbox for ray information

        self.LensPlotObjects = []
        self.AxisPlotObjects = []
        self.FilterPlotObjects = []
        self.OutlinePlotObjects = []
        self.AperturePlotObjects = []
        self.DetectorPlotObjects = []
        self.RaySourcePlotObjects = []
        self.RefractionIndexPlotObjects = []
        self.OrientationAxes = None

        # minimal/maximal z-positions of Detector_obj
        self.PosDetMin = self.Raytracer.outline[4]
        self.PosDetMax = self.Raytracer.outline[5]

        self.DetInd = 0
        self.SourceInd = 0

        # shift press indicator
        self.ShiftPressed   = False
        self.silent         = False
        self._exit          = False
        self._set_only      = False

        # convert bool parameters to values of ChecklistEditor
        self._set_only = True
        self.AbsorbMissing     = self.__CheckValFromBool(self.AbsorbMissing,     AbsorbMissing)
        self.CleanerView       = self.__CheckValFromBool(self.CleanerView,       CleanerView)
        self.LogImage          = self.__CheckValFromBool(self.LogImage,          LogImage)
        self.FlipImage         = self.__CheckValFromBool(self.FlipImage,         FlipImage)
        self.FocusDebugPlot    = self.__CheckValFromBool(self.FocusDebugPlot,    FocusDebugPlot)
        self.AFOneSource       = self.__CheckValFromBool(self.AFOneSource,       AFOneSource)
        self.DetImageOneSource = self.__CheckValFromBool(self.DetImageOneSource, DetImageOneSource)
        self._set_only = False

        self.lastDetSnap = None
        self.lastDetImage = None
        self.lastSourceSnap = None
        self.lastSourceImage = None

        # define Status dict
        self.StatusText = None
        self.Status.update(dict(InitScene=True, DisplayingGUI=True, Tracing=False, Drawing=False, Focussing=False,
                                DetectorImage=False, SourceImage=False))

        super().__init__(**kwargs)

        # wx Frame, see https://docs.enthought.com/mayavi/mayavi/auto/example_wx_mayavi_embed_in_notebook.html#example-wx-mayavi-embed-in-notebook
        self.wxF = wx.Frame(None, wx.ID_ANY, 'Mayavi with Wx')

    ####################################################################################################################
    # Helpers

    def __CheckValFromBool(self, obj, bool_): 
        return [obj.trait._metadata["editor"].values[0]] if bool_ else []

    def __removeObjects(self, objs):
        for obj in objs:
            for obji in obj:
                if obji is not None:
                    if obji.parent.parent.parent in self.Scene.mayavi_scene.children:
                        obji.parent.parent.parent.remove()
                    if obji.parent.parent in self.Scene.mayavi_scene.children:
                        obji.parent.parent.remove()

        objs[:] = [] 

    ####################################################################################################################
    # Plotting functions

    def plotLenses(self) -> None:
        """"""
        self.__removeObjects(self.LensPlotObjects)
      
        for num, L in enumerate(self.Raytracer.LensList):
            t = self.plotSObject(L, num, self.LENS_COLOR[:3], self.LENS_COLOR[3])
            self.LensPlotObjects.append(t)

    def plotApertures(self) -> None:
        """"""
        self.__removeObjects(self.AperturePlotObjects)
        
        for num, AP in enumerate(self.Raytracer.ApertureList):
            t = self.plotSObject(AP, num, (0, 0, 0), 1)
            self.AperturePlotObjects.append(t)

    def plotFilters(self) -> None:
        """"""
        self.__removeObjects(self.FilterPlotObjects)
        
        for num, F in enumerate(self.Raytracer.FilterList):
            Fcolor = F.getColor()
            alpha = 0.1 + 0.899*Fcolor[1]  # offset in both directions, ensures visibility and see-through
            t = self.plotSObject(F, num, Fcolor[:3], alpha)    
            self.FilterPlotObjects.append(t)

    def plotDetectors(self) -> None:
        """"""
        self.__removeObjects(self.DetectorPlotObjects)

        for num, Det in enumerate(self.Raytracer.DetectorList):
            t = self.plotSObject(Det, num, self.DETECTOR_COLOR[:3], self.DETECTOR_COLOR[3])    
            self.DetectorPlotObjects.append(t)

    def plotRaySources(self) -> None:
        """"""
        self.__removeObjects(self.RaySourcePlotObjects)

        for num, RS in enumerate(self.Raytracer.RaySourceList):
            t = self.plotSObject(RS, num, (1, 1, 1), self.RAYSOURCE_ALPHA, 
                                         d=-self.D_VIS, spec=False, light=False)   
            self.RaySourcePlotObjects.append(t)

    def plotOutline(self) -> None:
        """plot the raytracer outline"""

        self.__removeObjects(self.OutlinePlotObjects)

        self.Scene.engine.add_source(ParametricSurface(name="Outline"), self.Scene)
        a = self.Scene.mlab.outline(extent=self.Raytracer.outline.copy(), opacity=self.OUTLINE_ALPHA)
        a.actor.actor.pickable = False  # only rays should be pickablw

        self.OutlinePlotObjects.append((a,))

    def plotOrientationAxes(self) -> None:
        """plot orientation axes"""

        if self.OrientationAxes is not None:
            oaxes.remove()

        self.Scene.engine.add_source(ParametricSurface(name="Orientation Axes"), self.Scene)
        
        # show axes indicator
        self.OrientationAxes = self.Scene.mlab.orientation_axes()
        self.OrientationAxes.text_property.trait_set(**self.TEXT_STYLE)
        self.OrientationAxes.marker.interactive = 0  # make orientation axes non-interactive
        self.OrientationAxes.visible = not bool(self.CleanerView)
        self.OrientationAxes.widgets[0].viewport = [0, 0, 0.1, 0.15]

        # turn of text scaling of oaxes
        self.OrientationAxes.widgets[0].orientation_marker.x_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self.OrientationAxes.widgets[0].orientation_marker.y_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self.OrientationAxes.widgets[0].orientation_marker.z_axis_caption_actor2d.text_actor.text_scale_mode = 'none'

    def plotAxes(self) -> None:
        """plot cartesian axes"""

        # save old font factor. This is the one we adapted constantly in self.size_change()
        ff_old = self.AxisPlotObjects[0][0].axes.font_factor if self.AxisPlotObjects else 0.65

        self.__removeObjects(self.AxisPlotObjects)

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

            self.Scene.engine.add_source(ParametricSurface(name=f"{name}-Axis"), self.Scene)

            a = self.Scene.mlab.axes(extent=ext, nb_labels=lnum, x_axis_visibility=vis_x, 
                                       y_axis_visibility=vis_y, z_axis_visibility=vis_z)

            label=f"{name} / mm"
            a.axes.trait_set(font_factor=ff_old, fly_mode='none', label_format=lform, x_label=label, 
                               y_label=label, z_label=label, layer_number=1)

            # a.property.trait_set(display_location='background')
            a.title_text_property.trait_set(**self.AXIS_STYLE)
            a.label_text_property.trait_set(**self.AXIS_STYLE)
            a.actors[0].pickable = False
            a.visible = not bool(self.CleanerView)

            objs.append((a,))

        # place axes at outline
        ext = self.Raytracer.outline

        # enforce placement of x- and z-axis at y=ys (=RT.outline[2]) by shrinking extent
        ext_ys = ext.copy()
        ext_ys[3] = ext_ys[2]

        # X-Axis
        lnum = getLabelNum(ext[1] - ext[0], 5, 16)
        drawAxis(self.AxisPlotObjects, 0, ext_ys, lnum, "x", '%-#.4g', True, False, False)

        # Y-Axis
        lnum = getLabelNum(ext[3] - ext[2], 5, 16)
        drawAxis(self.AxisPlotObjects, 1, ext.copy(), lnum, "y", '%-#.4g', False, True, False)

        # Z-Axis
        lnum = getLabelNum(ext[5] - ext[4], 5, 24)
        drawAxis(self.AxisPlotObjects, 2, ext_ys, lnum, "z", '%-#.5g', False, False, True)

    def plotRefractionIndexBoxes(self) -> None:
        """plot outlines for ambient refraction index regions"""

        self.__removeObjects(self.RefractionIndexPlotObjects)

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
        if len(BoundList) == 2 and nList[0] == RefractionIndex():
            return

        # plot boxes
        for i in range(len(BoundList)-1):

            # plot outline
            self.Scene.engine.add_source(ParametricSurface(name=f"Refraction Index Outline {i}"), self.Scene)
            outline = self.Scene.mlab.outline(extent=[*self.Raytracer.outline[:4], BoundList[i], BoundList[i+1]],
                                              opacity=self.OUTLINE_ALPHA)
            outline.outline_mode = 'cornered'
            outline.actor.actor.pickable = False  # only rays should be pickablw

            # plot label
            label = nList[i].getDesc()
            x_pos = self.Raytracer.outline[0] + (self.Raytracer.outline[1]-self.Raytracer.outline[0])*0.05
            y_pos = (self.Raytracer.outline[2] + self.Raytracer.outline[3])/2
            z_pos = (BoundList[i+1]+BoundList[i])/2
            text_ = f"ambient\nn={label}" if not self.CleanerView else f"n={label}"
            text  = self.Scene.mlab.text(x_pos, y_pos, z=z_pos, text=text_, name=f"Label")

            text.actor.text_scale_mode = 'none'
            text.property.trait_set(**self.TEXT_STYLE, justification="center", frame=True, frame_color=self.SUBTLE_INFO_STYLE["color"])

            points = None

            self.RefractionIndexPlotObjects.append((outline, text, points))

    def plotSObject(self, obj, num, color, alpha, d=D_VIS, spec=True, light=True):
        """plotting of a SObject. Gets called from plotting for Lens, Filter, Detector, RaySource."""

        def plot(C, surf_type):
            # plot surface
            a = self.Scene.mlab.mesh(C[0], C[1], C[2], color=color, opacity=alpha, 
                                     name=f"{type(obj).__name__} {num} {surf_type} surface")

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
        label = f"{obj.abbr}{num}"
        # add desciption if any exists. But only if we are not in "CleanerView" displaying mode
        label = label if obj.desc == "" or bool(self.CleanerView) else label + ": " + obj.desc
        text = self.Scene.mlab.text(x=obj.extent[1], y=obj.pos[1], z=zl, text=label, name=f"Label")
        text.property.trait_set(**self.LABEL_STYLE, justification="center")
        text.actor.text_scale_mode = 'none'
       
        # plot BackSurface if one exists
        c = plot(obj.BackSurface.getPlottingMesh(N=self.SURFACE_RES), "back") if plotBack else None

        return a, b, c, text, obj

    def plotRays(self) -> None:
        """Ray/Point Plotting with a specified scalar coloring"""
        p_, _, pol_, w_, wl_, snum_ =\
                self.Raytracer.Rays.getRaysByMask(self.RaySelection, ret=[1, 0, 1, 1, 1, 1])

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
                if self.Raytracer.no_pol:
                    if not self.silent:
                        print("WARNING: Polarization calculation turned off in Raytracer.")
                    self.RaysPlot = None
                    self.RaysPlotScatter = None
                    return
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
        self.RaysPlotScatter = self.Scene.mlab.pipeline.scalar_scatter(x_cor, y_cor, z_cor, s, name="Rays")

        # assign which points are connected
        self.RaysPlotScatter.mlab_source.dataset.lines = connections

        # plot the lines between points
        self.RaysPlot = self.Scene.mlab.pipeline.surface(self.RaysPlotScatter, colormap=cm,
                                                         line_width=self.RayWidth, opacity=10**self.RayAlpha)
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
        hr = self.SceneSize0[0]/self.SceneSize[0] # horizontal ratio 
        vr = self.SceneSize0[1]/self.SceneSize[1] # vertical ratio
        lutm.scalar_bar_representation.position         = np.array([0.92, (1-0.6*vr)/2])
        lutm.scalar_bar_representation.position2        = np.array([0.06*hr, 0.6*vr])
        lutm.scalar_bar_widget.process_events           = False  # make non-interactive
        lutm.scalar_bar_representation.border_thickness = 0  # no ugly borders 

        if len(self.Raytracer.RaySourceList) != len(self.RaySourcePlotObjects):
            raise RuntimeError("Number of RaySourcePlots differs from actual Sources. Maybe the GUI was not updated properly?")
        
        # custom lut for wavelengths
        match self.ColoringType:

            case "White":
                RSColor = [(1, 1, 1) for RSp in self.RaySourcePlotObjects]

            case 'Wavelength':
                lutm.lut.table = Color.spectralCM(255)
                lutm.data_range = [Color.WL_MIN, Color.WL_MAX]
                lutm.number_of_labels = 11

                RSColor = [RS.getColor()[:3] for RS in self.Raytracer.RaySourceList]

            case 'Polarization':

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
        
                RSColor = [np.array(lutm.lut.table[i][:3])/255. for i, _ in enumerate(self.RaySourcePlotObjects)]

            case 'Power':
                if lutm.data_range[0]/(lutm.data_range[1]-lutm.data_range[0]) < 0.05: 
                    lutm.data_range = [0, lutm.data_range[1]]

                # set to maximum ray power, this is the same for all sources
                RSColor = [np.array(lutm.lut.table[-1][:3])/255. for RSp in self.RaySourcePlotObjects]

        # set RaySource Colors
        for i, RSp in enumerate(self.RaySourcePlotObjects):
            RSp[0].actor.actor.property.trait_set(color=tuple(RSColor[i]))
            if RSp[1] is not None:
                RSp[1].actor.actor.property.trait_set(color=tuple(RSColor[i]))

        # set visual ray properties
        self.RaysPlot.actor.actor.force_translucent = True
        self.RaysPlot.actor.property.representation = 'points' if self.PlottingType == 'Points' else 'surface'
        self.RaysPlot.actor.property.trait_set(line_width=self.RayWidth, 
                                               point_size=self.RayWidth if self.PlottingType == "Points" else 0.1)

    def initRayInfoText(self) -> None:
        """init detection of ray point clicks and the info display"""
        self.Scene.mlab.gcf().on_mouse_pick(self.onRayPick, button='Left')

        # add ray info text
        self.RayTextParent = self.Scene.engine.add_source(ParametricSurface(name="Ray Info Text"), self.Scene)
        self.RayText = self.Scene.mlab.text(0.02, 0.97, "") 

    def initStatusText(self) -> None:
        """init GUI status text display"""
        self.Scene.mlab.gcf().on_mouse_pick(self.onSpacePick, button='Right')
        
        # add status text
        self.StatusTextParent = self.Scene.engine.add_source(ParametricSurface(name="Status Info Text"), self.Scene)
        self.StatusText = self.Scene.mlab.text(0.97, 0.01, "Status Text")
        self.StatusText.property.trait_set(**self.INFO_STYLE, justification="right")
        self.StatusText.actor.text_scale_mode = 'none'
   
    def initSourceList(self) -> None:
        """generate a descriptive list of RaySource names"""
        self.SourceNames = [f"{RaySource.abbr}{num}: {RS.getDesc()}"[:35] 
                            for num, RS in enumerate(self.Raytracer.RaySourceList)]
        # don't append and delete directly on elements of self.SourceNames, 
        # because issues with trait updates with type "List"

    def initDetectorList(self) -> None:
        """generate a descriptive list of Detector names"""
        self.DetectorNames = [f"{Detector.abbr}{num}: {Det.getDesc()}"[:35] 
                              for num, Det in enumerate(self.Raytracer.DetectorList)]
        # don't append and delete directly on elements of self.DetectorNames, 
        # because issues with trait updates with type "List"

    def defaultCameraView(self) -> None:
        """set scene camera view. This should be called after the objects are added, otherwise the clipping range is incorrect"""
        self.Scene.scene.parallel_projection = True
        self.Scene.scene._def_pos = 1  # for some reason it is set to None
        self.Scene.y_plus_view()

    def onSpacePick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker') -> None:
        """
        3D Space Clicking Handler. 
        Shows Click Coordinates or moves Detector to this position when Shift is pressed.
        :param picker_obj:
        """

        pos = picker_obj.pick_position

        if self.ShiftPressed:
            if self.hasDetector():
                # set outside position to inside of outline
                pos_z = max(self.PosDetMin, pos[2])
                pos_z = min(self.PosDetMax, pos_z)
                
                # assign detector position, calls moveDetector()
                self.PosDet = pos_z
                
                # call with no parameter resets text
                self.onRayPick()
        else:
            self.RayText.text = f"Pick Position: ({pos[0]:>9.6g}, {pos[1]:>9.6g}, {pos[2]:>9.6g})"
            self.RayText.property.trait_set(**self.INFO_STYLE, background_opacity=0.2, opacity=1)

    def onRayPick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker'=None) -> None:
        """
        Ray Picking Handler.
        Shows ray properties on screen.
        :param picker_obj:
        """

        if not self.RayText:
            return

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

            N_shown = np.count_nonzero(self.RaySelection)
            RayMask = Misc.partMask(self.RaySelection, np.arange(N_shown) == n_)
            pos = np.nonzero(RayMask)[0][0]

            # get properties of this ray section
            p_, s_, pols_, pw_, wv, snum = self.Raytracer.Rays.getRaysByMask(RayMask)
            p_, s_, pols_, pw_, wv, snum = p_[0], s_[0], pols_[0], pw_[0], wv[0], snum[0]  # choose only ray
            p, s, pols, pw = p_[n2_], s_[n2_], pols_[n2_], pw_[n2_]

            pw0 = pw_[n2_-1] if n2_ else None
            s0 = s_[n2_-1] if n2_ else None
            pols0 = pols_[n2_-1] if n2_ else None
            pl = (pw0-pw)/pw0 if n2_ else None  # power loss

            def toSphCoords(s):
                return np.array([np.rad2deg(np.arctan2(s[1], s[0])), np.rad2deg(np.arccos(s[2])), 1])

            s_sph = toSphCoords(s)
            s0_sph = toSphCoords(s0) if n2_ else None
            pols_sph = toSphCoords(pols)
            pols0_sph = toSphCoords(pols0) if n2_ else None

            Elements = self.Raytracer._makeElementList()
            lastn = self.Raytracer.n0
            nList = [lastn]
            SList = [self.Raytracer.RaySourceList[snum].Surface]

            for El in Elements:
                SList.append(El.FrontSurface)
                if isinstance(El, Lens):
                    nList.append(El.n)
                    lastn = El.n2 if El.n2 is not None else self.Raytracer.n0
                    SList.append(El.BackSurface)
                nList.append(lastn)

            n = nList[n2_](wv)
            n0 = nList[n2_-1](wv) if n2_ else None

            normal = SList[n2_].getNormals(np.array([p[0]]), np.array([p[1]]))[0]
            normal_sph = toSphCoords(normal)

            if self.ShiftPressed:
                text =  f"Ray {pos}" +  \
                        f" from Source {snum}" +  \
                       (f" at surface {n2_}\n\n" if n2_ else f" at ray source\n\n") + \
                        f"Intersection Position: ({p[0]:>10.5g}, {p[1]:>10.5g}, {p[2]:>10.5g})\n\n" + \
                        f"                       Cartesian (x, y, z)                       Spherical (phi, theta, r)\n" + \
                       (f"Direction Before:      ({s0[0]:>10.5f}, {s0[1]:>10.5f}, {s0[2]:>10.5f})" if n2_ else "") + \
                       (f"      ({s0_sph[0]:>10.5f}°, {s0_sph[1]:>10.5f}°, {s0_sph[2]:>10.5f})\n" if n2_ else "") + \
                        f"Direction After:       ({s[0]:>10.5f}, {s[1]:>10.5f}, {s[2]:>10.5f})" + \
                        f"      ({s_sph[0]:>10.5f}°, {s_sph[1]:>10.5f}°, {s_sph[2]:>10.5f})\n" + \
                       (f"Polarization Before:   ({pols0[0]:>10.5f}, {pols0[1]:>10.5f}, {pols0[2]:>10.5f})" if n2_ else "") + \
                       (f"      ({pols0_sph[0]:>10.5f}°, {pols0_sph[1]:>10.5f}°, {pols0_sph[2]:>10.5f})\n" if n2_ else "") + \
                        f"Polarization After:    ({pols[0]:>10.5f}, {pols[1]:>10.5f}, {pols[2]:>10.5f})" + \
                        f"      ({pols_sph[0]:>10.5f}°, {pols_sph[1]:>10.5f}°, {pols_sph[2]:>10.5f})\n" + \
                       (f"Surface Normal:        ({normal[0]:>10.5f}, {normal[1]:>10.5f}, {normal[2]:>10.5f})" if pw > 0 else "") + \
                       (f"      ({normal_sph[0]:>10.5f}°, {normal_sph[1]:>10.5f}°, {normal_sph[2]:>10.5f})\n\n" if pw > 0 else "\n")+ \
                        f"Wavelength:               {wv:>10.2f} nm\n" + \
                       (f"Refraction Index Before:  {n0:>10.4f}\n" if n2_ else "") + \
                        f"Refraction Index After:   {n:>10.4f}\n" + \
                       (f"Ray Power Before:         {pw0*1e6:>10.5g} µW\n" if n2_ else "") + \
                        f"Ray Power After:          {pw*1e6:>10.5g} µW" + ("\n" if n2_ else "") + \
                       (f"Power Loss on Surface:    {pl*100:>10.5g} %" if n2_ else "")
            else:
                text =  f"Ray {pos}" +  \
                        f" from Source {snum}" +  \
                       (f" at surface {n2_}\n" if n2_ else f" at ray source\n") + \
                        f"Intersection Position: ({p[0]:>10.5g}, {p[1]:>10.5g}, {p[2]:>10.5g})\n" + \
                        f"Direction After:       ({s[0]:>10.5f}, {s[1]:>10.5f}, {s[2]:>10.5f})\n" + \
                        f"Polarization After:    ({pols[0]:>10.5f}, {pols[1]:>10.5f}, {pols[2]:>10.5f})\n" + \
                        f"Wavelength:             {wv:>10.2f} nm\n" + \
                        f"Ray Power After:        {pw*1e6:>10.5g} µW\n" + \
                       (f"Pick using Shift+Left Mouse Button for more info")

            self.RayText.text = text
            self.RayText.property.trait_set(**self.INFO_STYLE, background_opacity=0.2, opacity=1)
        else:
            self.RayText.text = "Left Click on a Ray-Surface Intersection to Show Ray Properties.\n" + \
                                "Right Click Anywhere to Show 3D Position.\n" + \
                                "Shift + Right Click to Move the Detector to this z-Position"
            self.RayText.property.trait_set(**self.SUBTLE_INFO_STYLE, background_opacity=0, opacity=0 if bool(self.CleanerView) else 1)

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
       
        if self._exit:
            self.close()

    def close(self):

        def widgetClose():
            app = QtGui.QApplication.instance()
            for widget in app.topLevelWidgets():
                widget.close()
            time.sleep(0.2)

        pyfaceGUI().process_events()
        pyfaceGUI().invoke_later(self.Scene.close)
        pyfaceGUI().invoke_later(widgetClose)

    def serve(self):

        sdict = dict(mlab = self.Scene.mlab,
                     engine = self.Scene.engine,
                     scene = self.Scene.scene,
                     camera = self.Scene.scene.camera,
                     close = self.close,
                     GUI = self,
                     RT = self.Raytracer,
                     LL = self.Raytracer.LensList,
                     FL = self.Raytracer.FilterList,
                     AL = self.Raytracer.ApertureList,
                     RSL = self.Raytracer.RaySourceList,
                     DL = self.Raytracer.DetectorList)

        TCPServer.serve_tcp(self, dict_=sdict)

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
            if self.StatusText is not None:
                self.StatusText.text += "Can't do this while raytracing.\n"
            if not self.silent:
                print("Raytracing in progress, skipping this action because of a race condition risk.")
            return False
        return True

    ###############################################################################################################
    # Interface functions
    ###############################################################################################################

    def replot(self, change=None):
       
        all_ = change is None
        
        if not all_ and not change["Any"]:
            return

        self.Status["Drawing"] = True
        pyfaceGUI().process_events()
        self.Scene.scene.disable_render = True
        cc_traits_org = self.Scene.camera.trait_get("position", "focal_point", "view_up", "view_angle",
                                                    "clipping_range", "parallel_scale")

        # RaySources need to be plotted first, since plotRays() colors them
        if all_ or change["RaySources"]:
            self.plotRaySources()
            self.initSourceList()
            
            # restore selected RaySource. If it is missing, default to 0.
            if self.SourceInd >= len(self.Raytracer.RaySourceList):
                self.SourceInd = 0
            if len(self.Raytracer.RaySourceList):
                self.SourceSelection = self.SourceNames[self.SourceInd]
       
        if all_ or change["TraceSettings"]:
            # reassign AbsorbMissing if it has changed
            if self.Raytracer.AbsorbMissing != bool(self.AbsorbMissing):
                self._set_only = True
                self.AbsorbMissing = self.__CheckValFromBool(self.AbsorbMissing, self.Raytracer.AbsorbMissing)
                self._set_only = False

        rdh = False  # if Drawing Status should be reset in this function
        if self.Raytracer.RaySourceList:
            if all_ or change["Filters"] or change["Lenses"] or change ["Apertures"] or change["Ambient"]\
                    or change["RaySources"] or change["TraceSettings"]:
                self.trace()
            elif change["Rays"]:
                self.chooseRays()
            else: # change["
                rdh = True
        else:
            rdh = True

        if all_ or change["Filters"]:
            self.plotFilters()
        
        if all_ or change["Lenses"]:
            self.plotLenses()

        if all_ or change["Apertures"]:
            self.plotApertures()

        if all_ or change["Detectors"]:
            self.plotDetectors()
            self.initDetectorList()

            # restore selected Detector. If it is missing, default to 0.
            if self.DetInd >= len(self.Raytracer.DetectorList):
                self.DetInd = 0
            if len(self.Raytracer.DetectorList):
                self.DetectorSelection = self.DetectorNames[self.DetInd]

        self.Scene.camera.trait_set(**cc_traits_org)
        
        if all_ or change["Ambient"]:
            self.plotRefractionIndexBoxes()
            self.plotAxes()
            self.plotOutline()

            # minimal/maximal z-positions of Detector_obj
            self.PosDetMin = self.Raytracer.outline[4]
            self.PosDetMax = self.Raytracer.outline[5]

        self.Scene.scene.disable_render = False
        self.Scene.camera.trait_set(**cc_traits_org)
        pyfaceGUI().process_events()
        
        if rdh:
            self.Status["Drawing"] = False

            # this only gets invoked after raytracing, but with no sources we need to do it here
            if not self.Raytracer.RaySourceList:
                pyfaceGUI().invoke_later(self.setGUILoaded)

    def waitForIdle(self) -> None:
        """wait until the GUI is Idle. Only call this from another thread"""
        time.sleep(0.01)  # wait for flags to be set
        while self.busy:
            time.sleep(0.02)
        
    @property
    def busy(self):
        busy = False
        [busy := True for key in self.Status.keys() if self.Status[key]]
        return busy
    
    def run(self, silent=False, no_server=False, _exit=False, _func=None, _args=tuple()):

        self._exit = _exit
        self.silent = silent
        self.Raytracer.silent = silent
        
        if not self.silent:
            log.startLogging(sys.stdout)

        if _func is not None:
            th = Thread(target=_func, args=_args, daemon=True)
            th.start()

        if no_server or _exit:
            self.configure_traits()
        else:
            self.edit_traits()
            self.serve()

    ####################################################################################################################
    # Trait handlers.

    @on_trait_change('RayAmountShown')
    def chooseRays(self) -> None:
        """chose a subset of all Raytracer rays and plot them with :obj:`GUI.drawRays`"""

        # don't do while init or tracing, since it will be done afterwards anyway
        if self.hasRaySource() and self.Raytracer.Rays.N and not self._set_only\
                and not self.Status["InitScene"] and not self.Status["Tracing"]:
            # don's use self.RayCount as ray number, since the raytracer could still be raytracing
            # instead use the last saved number
            N = self.Raytracer.Rays.N

            self.Status["Drawing"] = True
            pyfaceGUI().process_events()

            set_size = int(1 + 10**self.RayAmountShown*N)  # number of shown rays
            sindex = np.random.choice(N, size=set_size, replace=False)  # random choice

            # make bool array with chosen rays set to true
            self.RaySelection = np.zeros(N, dtype=bool)
            self.RaySelection[sindex] = True

            self.replotRays()

            self.Status["Drawing"] = False

    @on_trait_change('ColoringType, PlottingType')
    def replotRays(self) -> None:
        """remove old ray view, call :obj:`GUI.plotRays` and restore scene view"""

        # this gets called after InitScene and Tracing anyway
        if self.hasRaySource() and self.Raytracer.Rays.N\
                and not self.Status["InitScene"] and not self.Status["Tracing"]:

            self.Status["Drawing"] = True

            # set parameters to current value, since they could be changed by the user while tracing
            self._set_only = True
            self.AbsorbMissing = self.__CheckValFromBool(self.AbsorbMissing, self.Raytracer.AbsorbMissing)
            self.RayCount = self.Raytracer.Rays.N
            self._set_only = False

            # we could set the ray traits instead of removing and creating a new object,
            # but unfortunately the allocated numpy arrays (x, y, z, ..) of the objects are fixed in size
            for el in [self.RaysPlot, self.RaysPlotScatter]:
                if el:
                    el.remove()

            # save camera
            cc_traits_org = self.Scene.camera.trait_get("position", "focal_point", "view_up", "view_angle",
                                                        "clipping_range", "parallel_scale")
            self.plotRays()
          
            # reset camera
            self.Scene.camera.trait_set(**cc_traits_org)
           
            # show picker text 
            self.onRayPick()

            self.Status["Drawing"] = False

    @on_trait_change('RayCount, AbsorbMissing')
    def trace(self) -> None:
        """raytrace in separate thread, after that call :obj:`GUI.filterRays`"""

        if self._set_only or self.Status["Tracing"]:
            return

        if self.hasRaySource():

            for key in self.Status.keys():
                if self.Status[key] and key not in ["DisplayingGUI", "Drawing", "InitScene"]:
                    if self.StatusText is not None:
                        self.StatusText.text += "Can't trace while other background tasks are running.\n"
                    return

            self.Status["Tracing"] = True

            # run this in background thread
            def background() -> None:
                self.Raytracer.AbsorbMissing = bool(self.AbsorbMissing)
                self.Raytracer.trace(N=self.RayCount)

                # execute this after thread has finished
                def on_finish() -> None:
                    self.Status["Tracing"] = False

                    # lower the ratio of rays shown so that the GUI does not become to slow
                    if self.RayCount * 10**self.RayAmountShown > self.MAX_RAYS_SHOWN:
                        self._set_only = True
                        self.RayAmountShown = np.log10(self.MAX_RAYS_SHOWN/self.RayCount)
                        self._set_only = False

                    self.chooseRays()
                    pyfaceGUI.invoke_later(self.setGUILoaded)

                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()
        else:
            pyfaceGUI.invoke_later(self.setGUILoaded)
             
    @on_trait_change('RayAlpha')
    def setRayAlpha(self) -> None:
        """change opacity of visible rays"""
        if self.hasRaySource() and self.RaysPlot:
            self.Status["Drawing"] = True
            self.RaysPlot.actor.property.trait_set(opacity=10**self.RayAlpha)
            self.Status["Drawing"] = False

    @on_trait_change('PosDet')
    def setDetector(self) -> None:
        """move and replot chosen Detector"""

        if not self._set_only and self.hasDetector() and self.DetectorPlotObjects:

            xp, yp, zp = self.Raytracer.DetectorList[self.DetInd].pos
            self.Raytracer.DetectorList[self.DetInd].moveTo([xp, yp, self.PosDet])
            
            Surf = self.Raytracer.DetectorList[self.DetInd].Surface.getPlottingMesh(N=self.SURFACE_RES)[2]
            self.DetectorPlotObjects[self.DetInd][0].mlab_source.trait_set(z=Surf)

            if self.DetectorPlotObjects[self.DetInd][1]:
                Cyl = self.Raytracer.DetectorList[self.DetInd].getCylinderSurface(nc=self.SURFACE_RES)[2] 
                self.DetectorPlotObjects[self.DetInd][1].mlab_source.trait_set(z=Cyl)
                zl = self.PosDet + self.D_VIS/2
            else:            
                obj = self.Raytracer.DetectorList[self.DetInd]
                zl = (obj.extent[4] + obj.extent[5])/2 

            self.DetectorPlotObjects[self.DetInd][3].z_position = zl

            # reinit DetectorList and Selection to update z_pos in detector name
            self.initDetectorList()
            self.DetectorSelection = self.DetectorNames[self.DetInd]

    @on_trait_change('DetectorImageButton')
    def showDetectorImage(self) -> None:
        """render a DetectorImage at the chosen Detector, uses a separate thread"""

        if self.hasDetector() and self.noRaceCondition():

            self.Status["DetectorImage"] = True
            snum = None if not self.DetImageOneSource else self.SourceInd

            def background() -> None:
                # only calculate DetectorImage if Raytracer Snapshot, selected Detector or ImagePixels changed
                # otherwise we can replot the old Image with the new visual settings
                snap = str(self.Raytracer.PropertySnapshot()) + self.DetectorSelection + str(self.ImagePixels) + str(snum)
                if snap != self.lastDetSnap or self.lastDetImage is None:
                    self.lastDetImage = self.Raytracer.DetectorImage(N=self.ImagePixels, ind=self.DetInd,
                                                                     snum=snum, max_res=True)
                    self.lastDetSnap = snap

                Imc = self.lastDetImage.getByDisplayMode(self.ImageType, log=self.LogImage)

                def on_finish() -> None:
                    ImagePlot(self.lastDetImage, log=self.LogImage, flip=self.FlipImage, mode=self.ImageType, Imc=Imc)
                    self.Status["DetectorImage"] = False
                
                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()

    @on_trait_change('Scene.closing')
    def stop_tcp(self):
        if reactor.running:
            reactor.stop()

    @on_trait_change('SourceImageButton')
    def showSourceImage(self) -> None:
        """render a SourceImage for the chosen Source, uses a separate thread"""

        if self.hasRaySource() and self.noRaceCondition():

            self.Status["SourceImage"] = True

            def background() -> None:
                # only calculate SourceImage if Raytracer Snapshot, selected Source or ImagePixels changed
                # otherwise we can replot the old Image with the new visual settings
                snap = str(self.Raytracer.PropertySnapshot()) + self.SourceSelection + str(self.ImagePixels)

                if snap != self.lastSourceSnap or self.lastSourceImage is None:
                    self.lastSourceImage = self.Raytracer.SourceImage(N=self.ImagePixels,\
                                                                      sindex=self.SourceInd, max_res=True)
                    self.lastSourceSnap = snap
                
                Imc = self.lastSourceImage.getByDisplayMode(self.ImageType, log=self.LogImage)

                def on_finish() -> None:
                    ImagePlot(self.lastSourceImage, log=self.LogImage, flip=self.FlipImage, mode=self.ImageType, Imc=Imc)
                    self.Status["SourceImage"] = False
                
                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()

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
            def background() -> None:

                snum = None if not self.AFOneSource else self.SourceInd
                zf, zff, r, vals = self.Raytracer.autofocus(self.FocusType, self.PosDet,\
                                                            ret_cost=self.FocusDebugPlot, snum=snum)
                self.Raytracer.DetectorList[self.DetInd].moveTo([*self.Raytracer.DetectorList[self.DetInd].pos[:2], zf])

                # execute this function after thread has finished
                def on_finish() -> None:
                    if self.FocusDebugPlot:
                        AutoFocusDebugPlot(r, vals, zf, zff, f"Focus Finding Using the {self.FocusType} Method")
                    self.PosDet = self.Raytracer.DetectorList[self.DetInd].pos[2]  # setDetector() is called
                    self.Status["Focussing"] = False
                
                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()

    @on_trait_change('Scene.activated')
    def plotScene(self) -> None:
        """Initialize the GUI. Inits a variety of things."""
        self.initRayInfoText()
        self.initStatusText()
        self.initShiftPressDetector()
        self.plotOrientationAxes()
        
        self.Scene.background = self.BACKGROUND_COLOR
        
        self.replot()

        self.defaultCameraView()  # this need to be called after replot, which defines the visual scope
        self.Status["InitScene"] = False

    @on_trait_change('Status[]')
    def changeStatus(self) -> None:
        """Update the status info text."""

        if not self.Status["InitScene"] and self.StatusText is not None:
            text = ""
            if self.Status["Tracing"]:          text += "Raytracing...\n"
            if self.Status["Focussing"]:        text += "Focussing...\n"
            if self.Status["DetectorImage"]:    text += "Generating Detector Image...\n"
            if self.Status["SourceImage"]:      text += "Generating Source Image...\n"
            if self.Status["Drawing"]:          text += "Drawing...\n"

            self.StatusText.text = text

    @on_trait_change('CleanerView')
    def toggleCleanerView(self):

        if not self.Status["InitScene"]:

            show = not bool(self.CleanerView)

            if self.OrientationAxes is not None:
                self.OrientationAxes.visible = show

            for rio in self.RefractionIndexPlotObjects:
                if rio[1] is not None:
                    rio[1].text = rio[1].text.replace("ambient\n", "") if not show else ("ambient\n" + rio[1].text)

            # remove descriptions from labels in CleanerView
            for Objects in [self.RaySourcePlotObjects, self.LensPlotObjects,
                            self.FilterPlotObjects, self.AperturePlotObjects, self.DetectorPlotObjects]:
                for num, obj in enumerate(Objects):
                    label = f"{obj[4].abbr}{num}"
                    label = label if obj[4].desc == "" or not show else label + ": " + obj[4].desc
                    obj[3].text = label

            # replot Ray Pick Text, opacity of default text gets set
            self.onRayPick()

            for ax in self.AxisPlotObjects:
                if ax[0] is not None:
                    ax[0].visible = show
    
    @on_trait_change('RayWidth')
    def changeRayWidth(self) -> None:
        """ set the ray width for the visible rays."""
        if self.hasRaySource() and self.RaysPlot:
            self.Status["Drawing"] = True
            self.RaysPlot.actor.property.trait_set(line_width=self.RayWidth, point_size=self.RayWidth)
            self.Status["Drawing"] = False

    @on_trait_change('DetectorSelection')
    def changeDetector(self) -> None:
        """Updates the Detector Selection and the corresponding properties"""
        if self.hasDetector():
            # surface number for selected source
            self.DetInd = int(self.DetectorSelection.split(":", 1)[0].split("DET")[1])
            self.PosDet = self.Raytracer.DetectorList[self.DetInd].pos[2]

    @on_trait_change('SourceSelection')
    def changeRaySource(self) -> None:
        """Updates the Detector Selection and the corresponding properties"""
        if self.hasRaySource():
            self.SourceInd = int(self.SourceSelection.split(":", 1)[0].split("RS")[1])

    # rescale axes texts when the window was resized
    # for some reasons these are the only ones having no text_scaling_mode = 'none' option
    @on_trait_change('Scene:scene_editor:interactor:size')
    def size_change(self) -> None:
        """Handles GUI window size changes. Fixes incorrect scaling by mayavi"""

        SceneSize = self.Scene.scene_editor.interactor.size

        if SceneSize[0] != self.SceneSize[0] or SceneSize[1] != self.SceneSize[1]:

            # average  of x and y size for former and current scene size
            ch1 = (self.SceneSize[0] + self.SceneSize[1]) / 2
            ch2 = (SceneSize[0] + SceneSize[1]) / 2

            # update font factor so font size stays the same
            for ax in self.AxisPlotObjects:
                ax[0].axes.trait_set(font_factor=ax[0].axes.font_factor*ch1/ch2)

            # rescale orientation axes
            self.OrientationAxes.widgets[0].zoom = self.OrientationAxes.widgets[0].zoom * ch1/ch2
  
            if self.RaysPlot is not None:
                bar = self.RaysPlot.parent.scalar_lut_manager.scalar_bar_representation
                bar.position2 = np.array([bar.position2[0]*self.SceneSize[0]/SceneSize[0], 
                                          bar.position2[1]*self.SceneSize[1]/SceneSize[1]])
                bar.position = np.array([bar.position[0], (1-bar.position2[1])/2])

            # set current window size
            self.SceneSize = SceneSize

