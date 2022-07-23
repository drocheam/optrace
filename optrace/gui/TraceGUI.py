

"""
Frontend for the Raytracer class.
Shows a 3D representation of the geometry that is navigable.
Allows for the recalculation of the rays as well as visual settings for rays or points.
The detector is movable along the optical axis.
Interaction to source image and detector image plotting and its properties.
Autofocus functionality with different modes.

"""

import wx  # provides the creation of a wx app

import sys  # logging to sys.stdout
import time  # provides sleeping
import numpy as np  # calculations
from threading import Thread, Lock, Event  # threading
from typing import Callable, Any  # callable typing hints

import socket

from pynput.keyboard import Key, Listener  # detects key presses for events

from pyface.qt import QtGui  # closing UI elements
from pyface.api import GUI as pyfaceGUI  # invoke_later() method

# traits types and UI elements
from traitsui.api import View, Item, HSplit, VSplit, VFold, Group, CheckListEditor, TextEditor
from traits.api import HasTraits, Range, Instance, observe, Str, Button, Enum, List, Dict, String
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi.sources.parametric_surface import ParametricSurface  # provides outline and axes

# provides types and plotting functionality, most of these are also imported for the TCP protocol scope
from optrace.tracer import Lens, Filter, Aperture, RaySource, Detector, Raytracer, RImage, RefractionIndex,\
                           Surface, SurfaceFunction, Spectrum, LightSpectrum, TransmissionSpectrum
from optrace.plots.RImagePlots import RImagePlot  # plot RImages
from optrace.plots.DebugPlots import AutoFocusDebugPlot  # debugging of autofocus functionality
import optrace.tracer.presets as presets  # for server scope

import optrace.tracer.Color as Color  # for visible wavelength range
import optrace.gui.TCPServer as TCPServer  # TraceGUI TCP server
import optrace.tracer.Misc as Misc  # for partMask function

from twisted.internet import reactor  # networking functionality for tcp protocol
from twisted.python import log  # logging

from contextlib import contextmanager  # context manager for _no_trait_action()


# threading rules:
# + some ideas from: https://traits-futures.readthedocs.io/en/latest/guide/threading.html
# * GUI changes only in main thread
# * the GUI must be responsive at all costs
# * simplified RayStorage access model: only one thread can read and write (_RayAccessLock)
# * no locks in main thread, waiting for locks only in worker threads 
#
#
# Locks:
# * _DetectorImageLock: locks saved detector image for reading and writing
# * _SourceImageLock: locks saved source image for reading and writing
# * _DetectorLock: locks Detector in place (moveToFocus(), DetectorImage(), changeSelectedDetector(),), 
#                   changes of DetectorSelection and PosDet are intercepted and prevented in the main thread
# * _RayAccessLock: locks the RayStorage for reading or writing. Needed for moveToFocus(), DetectorImage(), 
#                   SourceImage(), trace() and replotRays()
#                   actually also needed for the picker, but we don't want to block the main thread, 
#                   so we prevent picking in this case


class TraceGUI(HasTraits):

    ####################################################################################################################
    # UI objects

    DETECTOR_COLOR: tuple[float, float, float] = (0.10, 0.10, 0.10, 0.80) 
    """RGB + Alpha tuple for the Detector visualization"""

    LENS_COLOR: tuple[float, float, float] = (0.63, 0.79, 1.00, 0.35)
    """RGB + Alpha Tuple for the Lens surface visualization"""

    BACKGROUND_COLOR: tuple[float, float, float] = (0.27, 0.25, 0.24)
    """RGB Color for the Scene background"""
    
    RAYSOURCE_ALPHA: float = 0.55
    """Alpha value for the RaySource visualization"""

    OUTLINE_ALPHA: float = 0.25
    """ Alpha value for outline visualization """

    SURFACE_RES: int = 100
    """Surface sampling count in each dimension"""

    D_VIS: float = 0.125
    """Visualization thickness for the side view of planar elements"""

    MAX_RAYS_SHOWN: int = 10000
    """Maximum of rays shown in visualization"""
   
    LABEL_STYLE: dict = dict(font_size=11, color=(1, 1, 1), bold=True, font_family="courier", shadow=True)
    """Standard Text Style. Used for object labels, legends and axes"""
    
    TEXT_STYLE: dict = dict(font_size=11, color=(1, 1, 1), font_family="courier", shadow=True, italic=False, bold=False)
    """Standard Text Style. Used for object labels, legends and axes"""

    INFO_STYLE: dict = dict(font_size=13, bold=True, color=(1, 1, 1), font_family="courier", shadow=True, italic=False)
    """Info Text Style. Used for status messages and interaction overlay"""

    SUBTLE_INFO_STYLE: dict = dict(font_size=13, bold=False, color=(0.42, 0.42, 0.42),
                                   font_family="courier", shadow=False)
    """Style for hidden info text. The color is used for the refraction index boxes frame"""
    
    ##########
    Scene: Instance = Instance(MlabSceneModel, args=())
    """the mayavi scene"""

    # Ranges

    RayCount: Range = Range(1000, Raytracer.MAX_RAYS, 200000, desc='Number of Rays for Simulation', enter_set=True,
                            auto_set=False, label="N_rays", mode='text')
    """Number of rays for raytracing"""

    RayAmountShown: Range = Range(-5.0, -1.0, -2, desc='Percentage of N Which is Drawn', enter_set=False,
                                  auto_set=False, label="N_vis (log)", mode='slider')
    """Number of rays shown in mayavi scenen. log10 value."""

    PosDet: Range = Range(low='_PosDetMin', high='_PosDetMax', value='_PosDetMax', mode='slider',
                          desc='z-Position of the Detector', enter_set=True, auto_set=True, label="z_pos")
    """z-Position of Detector. Lies inside z-range of :obj:`Backend.Raytracer.outline`"""

    RayAlpha: Range = Range(-5.0, 0.0, -2.0, desc='Opacity of the Rays/Points', enter_set=True,
                            auto_set=True, label="Alpha (log)", mode='slider')
    """opacity of shown rays. log10 values"""

    RayWidth: Range = Range(1.0, 20.0, 1, desc='Ray Linewidth or Point Size', enter_set=True,
                            auto_set=True, label="Width")
    """Width of rays shown."""

    ImagePixels: Range = Range(1, RImage.MAX_IMAGE_SIDE, 256, desc='Detector Image Pixels in Smaller of x or y Dimension',
                               enter_set=True, auto_set=True, label="Pixels_xy", mode='text')
    """Image Pixel value for Source/Detector Image. This the number of pixels for the smaller image side."""

    # Checklists (with capitalization workaround from https://stackoverflow.com/a/23783351)
    # this should basically be bool values, but because we want to have checkboxes with text right of them
    # we need to embed a CheckListEditor in a List. 
    # To assign bool values we need the workaround function self._CheckValFromBool

    AbsorbMissing: List = List(editor=CheckListEditor(values=["Absorb Rays Missing Lens"], format_func=lambda x: x),
                               desc="if Rays are Absorbed when not Hitting a Lens")
    """Boolean value for absorbing rays missing Lens. Packed as Checkbox into a :obj:`List`"""

    LogImage: List = List(editor=CheckListEditor(values=['Logarithmic Scaling'], format_func=lambda x: x),
                          desc="if Logarithmic Values are Plotted")
    """Boolean value for a logarithmic image visualization. Packed as Checkbox into a :obj:`List`"""

    FlipDetImage: List = List(editor=CheckListEditor(values=['Flip Detector Image'], format_func=lambda x: x),
                              desc="if detector image should be rotated by 180 degrees")
    """Boolean value for flipping the image (rotating it by 180°). Packed as Checkbox into a :obj:`List`"""

    FocusDebugPlot: List = List(editor=CheckListEditor(values=['Show Cost Function'], format_func=lambda x: x),
                                desc="if cost function is shown")
    """Show a plot of the optimization cost function for the focus finding"""

    CleanerView: List = List(editor=CheckListEditor(values=['Cleaner Scene View'], format_func=lambda x: x),
                             desc="if some scene elements should be hidden")
    """More minimalistic scene view. Axes are hidden and labels shortened."""

    AFOneSource: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'], format_func=lambda x: x),
                             desc="if autofocus only uses currently selected source")
    """Use only rays from selected source for focus finding"""
    
    DetImageOneSource: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'], 
                                   format_func=lambda x: x),
                                   desc="if DetectorImage only uses currently selected source")
    """Use only rays from selected source for the detector image"""

    # Enums

    PlottingTypes: list = ['Rays', 'Points']
    PlottingType: Enum = Enum(*PlottingTypes, desc="Ray Representation")
    """Ray Plotting Type"""

    ColoringTypes: list = ['White', 'Power', 'Wavelength', 'Polarization', 'Source']
    ColoringType: Enum = Enum(*ColoringTypes, desc="Ray Property to Color the Rays With")
    """Ray Coloring Mode"""

    ImageType: Enum = Enum(*RImage.display_modes, desc="Image Type Presented")
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
    RunButton:           Button = Button(label="Run", desc="runs the specified command")
    SourceImageButton:   Button = Button(label="Show Source Image",
                                         desc="Generating a Source Image of the Chosen Source")
    AutoFocusButton:     Button = Button(label="Find Focus",
                                         desc="Finding the Focus Between the Lenses in Front and Behind the Detector")

    _Cmd: String = String()

    # Labels
    Execute_Label = Str('Command:')
    History_Label = Str('History:')
    Command_History = Str('')
    WhitespaceLabel = Str('')
    Separator = Item("WhitespaceLabel", style='readonly', show_label=False)

    _Status: Dict = Dict(Str())
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
    _SceneSize0 = [1150, 850] # default size
    # with the window bar this should still fit on a 900p screen
    _SceneSize = _SceneSize0.copy() # will hold current size
    
    ####################################################################################################################
    # UI view creation

    view = View(
                HSplit(
                     # add mayavi scene
                    Group(
                        Group(
                            Item('Scene', editor=SceneEditor(scene_class=MayaviScene),
                                 height=_SceneSize0[1], width=_SceneSize0[0], show_label=False),
                             ),
                        layout="split",
                    ),
                    # add UI Elements
                    Group(
                        Group(
                            # Separator,
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
                            Item('FlipDetImage', style="custom", show_label=False),
                            Item('DetImageOneSource', style="custom", show_label=False),
                            Item('SourceImageButton', show_label=False),
                            Item('DetectorImageButton', show_label=False),
                            label="UI Actions"
                            ),
                        Group(
                            Item("Execute_Label", style='readonly', show_label=False),
                            Item('_Cmd', editor=TextEditor(auto_set=False, enter_set=True), show_label=False),
                            Item("RunButton", show_label=False),
                            Separator,
                            Separator,
                            Item("History_Label", style='readonly', show_label=False),
                            Item("Command_History", show_label=False, style="custom"),
                            label="Execute"
                            ),
                        layout="tabbed",
                        ),
                    ),
                resizable=True,
                title="Optrace"  # window title
                )
    """the UI view"""


    ####################################################################################################################
    # Class constructor

    def __init__(self, RT: Raytracer, **kwargs) -> None:
        """
        The extra bool parameters are needed to assign the List-Checklisteditor to bool values in the GUI.__init__. 
        Otherwise the user would assign bool values to a string list.

        :param RT:
        :param kwargs:
        """
       
        self.__DetectorLock = Lock()
        self.__DetImageLock = Lock()
        self.__SourceImageLock = Lock()
        self.__RayAccessLock = Lock()

        self.Raytracer: Raytracer = RT
        """reference to the Raytracer"""

        # ray properties
        self._RaySelection = np.array([])  # indices for subset of all rays in Raytracer
        self._RaysPlotScatter = None  # plot for scalar ray values
        self._RaysPlot = None  # plot for visualization of rays/points
        self._RayText = None  # textbox for ray information
        self._RayTextParent = None
        self._RayPicker = None
        self._SpacePicker = None
        self._Crosshair = None  # Crosshair used for picking visualization

        self._LensPlotObjects = []
        self._AxisPlotObjects = []
        self._FilterPlotObjects = []
        self._OutlinePlotObjects = []
        self._AperturePlotObjects = []
        self._DetectorPlotObjects = []
        self._RaySourcePlotObjects = []
        self._RefractionIndexPlotObjects = []
        self._OrientationAxes = None

        # minimal/maximal z-positions of Detector_obj
        self._PosDetMin = self.Raytracer.outline[4]
        self._PosDetMax = self.Raytracer.outline[5]

        self._DetInd = 0
        if len(self.Raytracer.DetectorList):
            self.PosDet = self.Raytracer.DetectorList[self._DetInd].pos[2]
        self._SourceInd = 0

        self._Socket = None

        # shift press indicator
        self._ShiftPressed  = False
        self.silent = False
        self._exit = False

        # set the properties after this without leading to a re-draw
        self._no_trait_action_flag = False
       
        with self._no_trait_action():

            # add true default parameter
            if "AbsorbMissing" not in kwargs:
                kwargs["AbsorbMissing"] = True

            # convert bool values to list entries for List()
            for key, val in kwargs.items():  
                if key in ["AbsorbMissing", "CleanerView", "LogImage", "FlipDetImage", 
                           "FocusDebugPlot", "AFOneSource", "DetImageOneSource"] and isinstance(val, bool):
                    kwargs[key] = self._CheckValFromBool(key, val)
           
            self._lastDetSnap = None
            self.lastDetImage = None
            self._lastSourceSnap = None
            self.lastSourceImage = None

            # define Status dict
            self._StatusText = None
            self._StatusTextParent = None
            self._Status.update(dict(InitScene=1, DisplayingGUI=1, Tracing=0, Drawing=0, Focussing=0,
                                    DetectorImage=0, SourceImage=0, ChangingDetector=0))

            super().__init__(**kwargs)

        # wx Frame, see https://docs.enthought.com/mayavi/mayavi/auto/example_wx_mayavi_embed_in_notebook.html#example-wx-mayavi-embed-in-notebook
        self.wxF = wx.Frame(None, wx.ID_ANY, 'Optrace Wx Frame')

    ####################################################################################################################
    # Helpers

    # convert bool value to list entry for Checkbox Lists
    def _CheckValFromBool(self, name, bool_): 
        return [self._trait(name, 0).editor.values[0]] if bool_ else []

    # remove visual objects from raytracer geometry
    def __removeObjects(self, objs):
        for obj in objs:
            for obji in obj[:4]:
                if obji is not None:
                    if obji.parent.parent.parent in self.Scene.mayavi_scene.children:
                        obji.parent.parent.parent.remove()
                    if obji.parent.parent in self.Scene.mayavi_scene.children:
                        obji.parent.parent.remove()

        objs[:] = [] 

    # workaround so we can set bool values to some List settings
    def __setattr__(self, key, val):
        if key in ["AbsorbMissing", "CleanerView", "LogImage", "FlipDetImage", "FocusDebugPlot",
                   "AFOneSource", "DetImageOneSource"]:
            if isinstance(val, bool):
                val = self._CheckValFromBool(key, val)

        super().__setattr__(key, val)

    @contextmanager
    def _no_trait_action(self):
        self._no_trait_action_flag = True
        try:
            yield
        finally:
            self._no_trait_action_flag = False
    
    @contextmanager
    def _constant_camera(self):
        cc_traits_org = self.Scene.camera.trait_get("position", "focal_point", "view_up", "view_angle",
                                                    "clipping_range", "parallel_scale")
        try:
            yield
        finally:
            self.Scene.camera.trait_set(**cc_traits_org)
   
    def _doInMain(self, f, *args, **kw):
        pyfaceGUI().invoke_later(f, *args, **kw)
        pyfaceGUI().process_events()

    def _setInMain(self, trait, val):
        pyfaceGUI().set_trait_later(self, trait, val)
        pyfaceGUI().process_events()

    def _process_events(self):
        pyfaceGUI().process_events()

    ####################################################################################################################
    # Plotting functions

    def plotLenses(self) -> None:
        """"""
        self.__removeObjects(self._LensPlotObjects)
      
        for num, L in enumerate(self.Raytracer.LensList):
            t = self._plotSObject(L, num, self.LENS_COLOR[:3], self.LENS_COLOR[3])
            self._LensPlotObjects.append(t)

    def plotApertures(self) -> None:
        """"""
        self.__removeObjects(self._AperturePlotObjects)
        
        for num, AP in enumerate(self.Raytracer.ApertureList):
            t = self._plotSObject(AP, num, (0, 0, 0), 1)
            self._AperturePlotObjects.append(t)

    def plotFilters(self) -> None:
        """"""
        self.__removeObjects(self._FilterPlotObjects)
        
        for num, F in enumerate(self.Raytracer.FilterList):
            Fcolor = F.getColor()
            alpha = 0.1 + 0.899*Fcolor[1]  # offset in both directions, ensures visibility and see-through
            t = self._plotSObject(F, num, Fcolor[:3], alpha)    
            self._FilterPlotObjects.append(t)

    def plotDetectors(self) -> None:
        """"""
        self.__removeObjects(self._DetectorPlotObjects)

        for num, Det in enumerate(self.Raytracer.DetectorList):
            t = self._plotSObject(Det, num, self.DETECTOR_COLOR[:3], self.DETECTOR_COLOR[3])    
            self._DetectorPlotObjects.append(t)

    def plotRaySources(self) -> None:
        """"""
        self.__removeObjects(self._RaySourcePlotObjects)

        for num, RS in enumerate(self.Raytracer.RaySourceList):
            t = self._plotSObject(RS, num, (1, 1, 1), self.RAYSOURCE_ALPHA, d=-self.D_VIS, spec=False, light=False)
            self._RaySourcePlotObjects.append(t)

    def plotOutline(self) -> None:
        """plot the raytracer outline"""
        self.__removeObjects(self._OutlinePlotObjects)

        self.Scene.engine.add_source(ParametricSurface(name="Outline"), self.Scene)
        a = self.Scene.mlab.outline(extent=self.Raytracer.outline.copy(), opacity=self.OUTLINE_ALPHA)
        a.actor.actor.pickable = False  # only rays should be pickable

        self._OutlinePlotObjects.append((a,))

    def plotOrientationAxes(self) -> None:
        """plot orientation axes"""

        if self._OrientationAxes is not None:
            self._OrientationAxes.remove()

        self.Scene.engine.add_source(ParametricSurface(name="Orientation Axes"), self.Scene)
        
        # show axes indicator
        self._OrientationAxes = self.Scene.mlab.orientation_axes()
        self._OrientationAxes.text_property.trait_set(**self.TEXT_STYLE)
        self._OrientationAxes.marker.interactive = 0  # make orientation axes non-interactive
        self._OrientationAxes.visible = not bool(self.CleanerView)
        self._OrientationAxes.widgets[0].viewport = [0, 0, 0.1, 0.15]

        # turn of text scaling of oaxes
        self._OrientationAxes.widgets[0].orientation_marker.x_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self._OrientationAxes.widgets[0].orientation_marker.y_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self._OrientationAxes.widgets[0].orientation_marker.z_axis_caption_actor2d.text_actor.text_scale_mode = 'none'

    def plotAxes(self) -> None:
        """plot cartesian axes"""

        # save old font factor. This is the one we adapted constantly in self._resizeSceneElements()
        ff_old = self._AxisPlotObjects[0][0].axes.font_factor if self._AxisPlotObjects else 0.65

        self.__removeObjects(self._AxisPlotObjects)

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

        def drawAxis(objs, ind: int, ext: list, lnum: int, name: str, lform: str,
                     vis_x: bool, vis_y: bool, vis_z: bool):

            self.Scene.engine.add_source(ParametricSurface(name=f"{name}-Axis"), self.Scene)

            a = self.Scene.mlab.axes(extent=ext, nb_labels=lnum, x_axis_visibility=vis_x, 
                                     y_axis_visibility=vis_y, z_axis_visibility=vis_z)

            label=f"{name} / mm"
            a.axes.trait_set(font_factor=ff_old, fly_mode='none', label_format=lform, x_label=label, 
                               y_label=label, z_label=label, layer_number=1)

            # a.property.trait_set(display_location='background')
            a.title_text_property.trait_set(**self.TEXT_STYLE)
            a.label_text_property.trait_set(**self.TEXT_STYLE)
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
        drawAxis(self._AxisPlotObjects, 0, ext_ys, lnum, "x", '%-#.4g', True, False, False)

        # Y-Axis
        lnum = getLabelNum(ext[3] - ext[2], 5, 16)
        drawAxis(self._AxisPlotObjects, 1, ext.copy(), lnum, "y", '%-#.4g', False, True, False)

        # Z-Axis
        lnum = getLabelNum(ext[5] - ext[4], 5, 24)
        drawAxis(self._AxisPlotObjects, 2, ext_ys, lnum, "z", '%-#.5g', False, False, True)

    def plotRefractionIndexBoxes(self) -> None:
        """plot outlines for ambient refraction index regions"""

        self.__removeObjects(self._RefractionIndexPlotObjects)

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
            text = self.Scene.mlab.text(x_pos, y_pos, z=z_pos, text=text_, name=f"Label")

            text.actor.text_scale_mode = 'none'
            text.property.trait_set(**self.TEXT_STYLE, justification="center", frame=True,
                                    frame_color=self.SUBTLE_INFO_STYLE["color"])

            points = None

            self._RefractionIndexPlotObjects.append((outline, text, points))

    def _plotSObject(self, obj, num, color, alpha, d=D_VIS, spec: bool=True, light: bool=True):
        """plotting of a SObject. Gets called from plotting for Lens, Filter, Detector, RaySource.

        :param obj:
        :param num:
        :param color:
        :param alpha:
        :param d:
        :param spec:
        :param light:
        :return:
        """

        def plot(C, surf_type):
            # plot surface
            a = self.Scene.mlab.mesh(C[0], C[1], C[2], color=color, opacity=alpha, 
                                     name=f"{type(obj).__name__} {num} {surf_type} surface")

            # make non-pickable so it does not interfere with our ray picker
            a.actor.actor.pickable = False

            a.actor.actor.property.lighting = light
            if spec:
                a.actor.property.trait_set(specular=0.5, ambient=0.25)
            return a

        # decide what to plot
        plotFront = obj.FrontSurface.surface_type not in ["Point", "Line"]
        plotCyl = (obj.hasBackSurface() or obj.Surface.isPlanar()) and plotFront
        plotBack = obj.BackSurface is not None and obj.BackSurface.surface_type not in ["Point", "Line"]

        a = plot(obj.FrontSurface.getPlottingMesh(N=self.SURFACE_RES), "front") if plotFront else None

        # if surface is a plane or object a lens, plot a cylinder for side viewing
        b = plot(obj.getCylinderSurface(nc=self.SURFACE_RES, d=d), "cylinder") if plotCyl else None

        # calculate middle center z-position
        zl = (obj.extent[4] + obj.extent[5])/2 if not obj.hasBackSurface() else obj.pos[2]
        zl = zl + self.D_VIS/2 if obj.Surface.isPlanar() and not obj.hasBackSurface() else zl

        # object label
        label = f"{obj.abbr}{num}"
        # add description if any exists. But only if we are not in "CleanerView" displaying mode
        label = label if obj.desc == "" or bool(self.CleanerView) else label + ": " + obj.desc
        text = self.Scene.mlab.text(x=obj.extent[1], y=obj.pos[1], z=zl, text=label, name=f"Label")
        text.property.trait_set(**self.LABEL_STYLE, justification="center")
        text.actor.text_scale_mode = 'none'
       
        # plot BackSurface if one exists
        c = plot(obj.BackSurface.getPlottingMesh(N=self.SURFACE_RES), "back") if plotBack else None

        return a, b, c, text, obj

    def _getRays(self) -> None:
        """Ray/Point Plotting with a specified scalar coloring"""

        p_, s_, pol_, w_, wl_, snum_ = self.Raytracer.Rays.getRaysByMask(self._RaySelection, normalize=False)

        # get flattened list of the coordinates
        x, y, z = p_[:, :, 0].ravel(), p_[:, :, 1].ravel(), p_[:, :, 2].ravel()
        u, v, w = s_[:, :, 0].ravel(), s_[:, :, 1].ravel(), s_[:, :, 2].ravel()

        self._RayProperties = [pol_, w_, wl_, snum_]
        s = np.ones_like(z)

        return x, y, z, u, v, w, s

    def _plotRays(self, x, y, z, u, v, w, s):
        if self._RaysPlot is not None:
            self._RaysPlot.parent.parent.remove()

        self._RaysPlot = self.Scene.mlab.quiver3d(x, y, z, u, v, w, scalars=s, 
                                                 scale_mode="vector", scale_factor=1, colormap="Greys", mode="2ddash")
        self._RaysPlot.glyph.trait_set(color_mode="color_by_scalar")
        self._RaysPlot.actor.actor.property.trait_set(lighting=False, render_points_as_spheres=True,
                                                     opacity=10**self.RayAlpha)
        self._RaysPlot.actor.property.trait_set(line_width=self.RayWidth, 
                                               point_size=self.RayWidth if self.PlottingType == "Points" else 0.1)
        self._RaysPlot.parent.parent.name = "Rays"

    def _assignRayColors(self):

        pol_, w_, wl_, snum_ = self._RayProperties
        N, nt, nc = pol_.shape
        
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
                    self._RaysPlot = None
                    return
                s = np.sqrt(pol_[:, :, 1]**2 + pol_[:, :, 2]**2).ravel()
                cm = "gnuplot"
                title = "Polarization\n projection\n on yz-plane"

            case _:
                s = np.ones_like(w_)
                cm = "Greys"
                title = "None"

        self._RaysPlot.mlab_source.trait_set(scalars=s)

        # lut legend settings
        lutm = self._RaysPlot.parent.scalar_lut_manager
        lutm.trait_set(use_default_range=True, show_scalar_bar=True, use_default_name=False,
                       show_legend=self.ColoringType!="White", lut_mode=cm)

        # lut visibility and title
        lutm.scalar_bar.trait_set(title=title, unconstrained_font_size=True)
        lutm.label_text_property.trait_set(**self.TEXT_STYLE)
        lutm.title_text_property.trait_set(**self.INFO_STYLE)

        # lut position and size
        hr = self._SceneSize0[0]/self._SceneSize[0]  # horizontal ratio
        vr = self._SceneSize0[1]/self._SceneSize[1]  # vertical ratio
        lutm.scalar_bar_representation.position = np.array([0.92, (1-0.6*vr)/2])
        lutm.scalar_bar_representation.position2 = np.array([0.06*hr, 0.6*vr])
        lutm.scalar_bar_widget.process_events = False  # make non-interactive
        lutm.scalar_bar_representation.border_thickness = 0  # no ugly borders 

        if len(self.Raytracer.RaySourceList) != len(self._RaySourcePlotObjects):
            raise RuntimeError("Number of RaySourcePlots differs from actual Sources. "
                               "Maybe the GUI was not updated properly?")
        
        match self.ColoringType:

            case "White":
                RSColor = [(1, 1, 1) for RSp in self._RaySourcePlotObjects]

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
                    match RS.polarization:
                        case "x":
                            col = lutm.lut.table[0]
                        case "y":
                            col = lutm.lut.table[-1]
                        case "Angle":
                            yzpr = (np.sin(np.deg2rad(RS.pol_angle)) - lutm.data_range[0]) / lrange
                            col = lutm.lut.table[int(yzpr*255)]
                        case _:  # no single axis -> show white
                            col = (255, 255, 255)

                    RSColor.append(np.array(col[:3])/255)

            case 'Source':
                lutm.number_of_labels = len(self.Raytracer.RaySourceList)
                lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-#6.f"
                if lutm.number_of_labels > 1:
                    # set color steps, smaller 420-650nm range for  larger color contrast
                    lutm.lut.table = Color.spectralCM(lutm.number_of_labels, 420, 650)
        
                RSColor = [np.array(lutm.lut.table[i][:3])/255. for i, _ in enumerate(self._RaySourcePlotObjects)]

            case 'Power':
                if lutm.data_range[0]/(lutm.data_range[1]-lutm.data_range[0]) < 0.05: 
                    lutm.data_range = [0, lutm.data_range[1]]

                # set to maximum ray power, this is the same for all sources
                RSColor = [np.array(lutm.lut.table[-1][:3])/255. for RSp in self._RaySourcePlotObjects]

        # set RaySource Colors
        for i, RSp in enumerate(self._RaySourcePlotObjects):
            if RSp[0] is not None:
                RSp[0].actor.actor.property.trait_set(color=tuple(RSColor[i]))
            if RSp[1] is not None:
                RSp[1].actor.actor.property.trait_set(color=tuple(RSColor[i]))


    def _initRayInfoText(self) -> None:
        """init detection of ray point clicks and the info display"""
        self._RayPicker = self.Scene.mlab.gcf().on_mouse_pick(self._onRayPick, button='Left')

        # add ray info text
        self._RayTextParent = self.Scene.engine.add_source(ParametricSurface(name="Ray Info Text"), self.Scene)
        self._RayText = self.Scene.mlab.text(0.02, 0.97, "") 

    def _initStatusText(self) -> None:
        """init GUI status text display"""
        self._SpacePicker = self.Scene.mlab.gcf().on_mouse_pick(self._onSpacePick, button='Right')
        
        # add status text
        self._StatusTextParent = self.Scene.engine.add_source(ParametricSurface(name="Status Info Text"), self.Scene)
        self._StatusText = self.Scene.mlab.text(0.97, 0.01, "Status Text")
        self._StatusText.property.trait_set(**self.INFO_STYLE, justification="right")
        self._StatusText.actor.text_scale_mode = 'none'
   
    def _initSourceList(self) -> None:
        """generate a descriptive list of RaySource names"""
        self.SourceNames = [f"{RaySource.abbr}{num}: {RS.getDesc()}"[:35] 
                             for num, RS in enumerate(self.Raytracer.RaySourceList)]
        # don't append and delete directly on elements of self.SourceNames, 
        # because issues with trait updates with type "List"

    def _initDetectorList(self) -> None:
        """generate a descriptive list of Detector names"""
        self.DetectorNames = [f"{Detector.abbr}{num}: {Det.getDesc()}"[:35] 
                               for num, Det in enumerate(self.Raytracer.DetectorList)]
        # don't append and delete directly on elements of self.DetectorNames, 
        # because issues with trait updates with type "List"

    def _initCrosshair(self) -> None:
        """init a crosshair for the picker"""
        self._Crosshair = self.Scene.mlab.points3d([0.], [0.], [0.], mode="axes", color=(1, 0, 0))
        self._Crosshair.parent.parent.name = "Crosshair"
        self._Crosshair.actor.actor.property.trait_set(lighting=False)
        self._Crosshair.actor.actor.pickable = False
        self._Crosshair.glyph.glyph.scale_factor = 1.5
        self.CrosshairVisibility(False)

    def moveCrosshair(self, pos: list[float]) -> None:
        """move the crosshair for the picker
        :param pos:
        """
        if self._Crosshair is not None:
            x_cor, y_cor, z_cor = [pos[0]], [pos[1]], [pos[2]]
            self._Crosshair.mlab_source.trait_set(x=x_cor, y=y_cor, z=z_cor)

    def CrosshairVisibility(self, visible: bool, global_override: bool = None) -> None:
        """change crosshair visibility
        :param visible:
        :param global_override:
        """
        if self._Crosshair is not None:
            self._Crosshair.visible = global_override if global_override is not None else visible

    def defaultCameraView(self) -> None:
        """set scene camera view. This should be called after the objects are added,
           otherwise the clipping range is incorrect"""
        self.Scene.scene.parallel_projection = True
        self.Scene.scene._def_pos = 1  # for some reason it is set to None
        self.Scene.y_plus_view()

    def _onSpacePick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker') -> None:
        """
        3D Space Clicking Handler. 
        Shows Click Coordinates or moves Detector to this position when Shift is pressed.
        :param picker_obj:
        """

        pos = picker_obj.pick_position

        if self._ShiftPressed:
            self.CrosshairVisibility(False)
            if self.Raytracer.DetectorList:
                # set outside position to inside of outline
                pos_z = max(self._PosDetMin, pos[2])
                pos_z = min(self._PosDetMax, pos_z)
                
                # assign detector position, calls moveDetector()
                self.PosDet = pos_z
                
                # call with no parameter resets text
                self._onRayPick()
        else:
            self._RayText.text = f"Pick Position: ({pos[0]:>9.6g}, {pos[1]:>9.6g}, {pos[2]:>9.6g})"
            self._RayText.property.trait_set(**self.INFO_STYLE, background_opacity=0.2, opacity=1)
            self.moveCrosshair(pos)
            self.CrosshairVisibility(True)

    def _onRayPick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker'=None) -> None:
        """
        Ray Picking Handler.
        Shows ray properties on screen.
        :param picker_obj:
        """

        if not self._RayText:
            return

        # it seems that picker_obj.point_id is only present for the first element in the picked list,
        # so we only can use it when the RayPlot is first in the list
        # see https://github.com/enthought/mayavi/issues/906
        if picker_obj is not None and len(picker_obj.actors) != 0 \
           and picker_obj.actors[0]._vtk_obj is self._RaysPlot.actor.actor._vtk_obj:
         
            # don't pick while plotted rays or Rays in Raytracer could change
            # using the RayAccessLock wouldn't be wise, because it also used by Focussing
            if self._Status["Tracing"] or self._Status["Drawing"]:
                if not self.silent:
                    print("Can't pick while tracing or drawing.")
                return

            a = self.Raytracer.Rays.nt # number of points per ray plotted
            b = picker_obj.point_id # point id of the ray point

            n_, n2_ = np.divmod(b, 2*a)
            if n2_ > 0:
                n2_ = min(1 + (n2_-1)//2, a - 1)  # in some cases n2_ would is above a-1, but why?

            N_shown = np.count_nonzero(self._RaySelection)
            RayMask = Misc.partMask(self._RaySelection, np.arange(N_shown) == n_)
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

            if self._ShiftPressed:
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
                        f"Pick using Shift+Left Mouse Button for more info"

            self._RayText.text = text
            self._RayText.property.trait_set(**self.INFO_STYLE, background_opacity=0.2, opacity=1)
            self.moveCrosshair(p)
            self.CrosshairVisibility(True)
        else:
            self._RayText.text = "Left Click on a Ray-Surface Intersection to Show Ray Properties.\n" + \
                                 "Right Click Anywhere to Show 3D Position.\n" + \
                                 "Shift + Right Click to Move the Detector to this z-Position"
            self._RayText.property.trait_set(**self.SUBTLE_INFO_STYLE, background_opacity=0,
                                            opacity=0 if bool(self.CleanerView) else 1)
            self.CrosshairVisibility(False)

        # text settings
        self._RayText.property.trait_set(vertical_justification="top")
        self._RayText.actor.text_scale_mode = 'none'

    # TODO documentation
    def _initKeyboardShortcuts(self) -> None:
        """Init shift key press detection"""

        self._ShiftPressed = False

        # check if focus is in scene. otherwise we would react on key presses from different windows or dialogs
        def sceneFocused() -> bool:
            if self.Scene is None or self.Scene.scene_editor is None or self.Scene.scene_editor._content is None:
                return False
            return self.Scene.scene_editor._content.hasFocus()
        
        def on_press(key):
            if key == Key.shift:
                self._ShiftPressed = True

        def on_release(key):
            if key == Key.shift:
                self._ShiftPressed = False

            elif hasattr(key, 'char') and sceneFocused():
                if key.char == "y":
                    self.defaultCameraView()

                elif key.char == "c":
                    self.CleanerView = not bool(self.CleanerView)

                elif key.char == "r":
                    # cycle plotting types
                    ptypes = self.PlottingTypes
                    self.PlottingType = ptypes[(ptypes.index(self.PlottingType)+1) % len(ptypes)]

                elif key.char == "d":
                    self.showDetectorImage()

                elif key.char == "n":
                    self.Scene.scene.disable_render = True
                    self.replotRays()
                    self.Scene.scene.disable_render = False

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def _setGUILoaded(self) -> None:
        """sets GUILoaded Status. Exits GUI if self.exit set."""

        self._Status["DisplayingGUI"] = 0
      
        if self._exit:  # wait 300ms so the user has time to see the scene 
            pyfaceGUI().invoke_after(300, self.close)

    def close(self):
        """close the whole application"""
        pyfaceGUI().invoke_later(QtGui.QApplication.closeAllWindows)

    def serve(self):

        # dictionary for command scope
        sdict = dict(# GUI and scene
                     mlab=self.Scene.mlab, engine=self.Scene.engine, scene=self.Scene.scene,
                     camera=self.Scene.scene.camera, GUI=self, 

                     # libs
                     np=np, time=time, sys=sys,

                     # tracer classes
                     Lens=Lens, Surface=Surface, Aperture=Aperture, Filter=Filter, Spectrum=Spectrum, 
                     TransmissionSpectrum=TransmissionSpectrum, LightSpectrum=LightSpectrum, 
                     RefractionIndex=RefractionIndex, Detector=Detector, SurfaceFunction=SurfaceFunction, 
                     RImage=RImage, presets=presets, RImagePlot=RImagePlot,

                     # abbreviations for raytracer and object lists
                     RT=self.Raytracer, LL=self.Raytracer.LensList, FL=self.Raytracer.FilterList, 
                     APL=self.Raytracer.ApertureList, RSL=self.Raytracer.RaySourceList,
                     DL=self.Raytracer.DetectorList)

        TCPServer.serve_tcp(self, dict_=sdict)

    ####################################################################################################################
    # Interface functions
    ####################################################################################################################

    def replot(self, change=None):
       
        all_ = change is None
        
        if not all_ and not change["Any"]:
            return

        self._Status["Drawing"] += 1
        pyfaceGUI().process_events()
        self.Scene.scene.disable_render = True

        with self._constant_camera():

            # RaySources need to be plotted first, since plotRays() colors them
            if all_ or change["RaySources"]:
                self.plotRaySources()
                self._initSourceList()
                
                # restore selected RaySource. If it is missing, default to 0.
                if self._SourceInd >= len(self.Raytracer.RaySourceList):
                    self._SourceInd = 0
                if len(self.Raytracer.RaySourceList):
                    self.SourceSelection = self.SourceNames[self._SourceInd]
           
            if all_ or change["TraceSettings"]:
                # reassign AbsorbMissing if it has changed
                if self.Raytracer.AbsorbMissing != bool(self.AbsorbMissing):
                    with self._no_trait_action():
                        self.AbsorbMissing = self._CheckValFromBool("AbsorbMissing", self.Raytracer.AbsorbMissing)

            rdh = False  # if Drawing Status should be reset in this function
            if self.Raytracer.RaySourceList:
                if all_ or change["Filters"] or change["Lenses"] or change ["Apertures"] or change["Ambient"]\
                        or change["RaySources"] or change["TraceSettings"]:
                    self.trace()
                elif change["Rays"]:
                    self.replotRays()
                else:
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
                # no threading and __DetectorLock here, since we're only updating the list
            
                self.plotDetectors()
                self._initDetectorList()

                # restore selected Detector. If it is missing, default to 0.
                if self._DetInd >= len(self.Raytracer.DetectorList):
                    self._DetInd = 0

                if len(self.Raytracer.DetectorList):
                    self.DetectorSelection = self.DetectorNames[self._DetInd]

            if all_ or change["Ambient"]:
                self.plotRefractionIndexBoxes()
                self.plotAxes()
                self.plotOutline()

                # minimal/maximal z-positions of Detector_obj
                self._PosDetMin = self.Raytracer.outline[4]
                self._PosDetMax = self.Raytracer.outline[5]

        self.Scene.scene.disable_render = False
        pyfaceGUI().process_events()
        
        if rdh:
            # this only gets invoked after raytracing, but with no sources we need to do it here
            if not self.Raytracer.RaySourceList:
                pyfaceGUI().invoke_later(self._setGUILoaded)
            
        self._Status["Drawing"] -= 1

    def _waitForIdle(self) -> None:
        """wait until the GUI is Idle. Only call this from another thread"""
        time.sleep(0.1)  # wait for flags to be set, this could also be trait handlers, which could take longer
        # unfortunately we can't trust self.busy either
        while self.busy:
            time.sleep(0.05)
        
    @property
    def busy(self):
        # currently I see no way to check if the TraceGUI is actually idle
        # it could also be in some trait_handlers or functions from other classes
        busy = False
        [busy := True for key in self._Status.keys() if self._Status[key]]
        return busy or self.Scene.busy
  
    def debug(self, silent=False, no_server=True, _exit=False, _func=None, _args: tuple=None):

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

    def run(self, silent=False):

        self.silent = silent
        self.Raytracer.silent = silent
        
        if not self.silent:
            log.startLogging(sys.stdout)

        self.edit_traits()
        self.serve()

    ####################################################################################################################
    # Trait handlers.

    @observe('RayAmountShown')
    def replotRays(self, event=None) -> None:
        """chose a subset of all Raytracer rays and plot them with :obj:`GUI.drawRays`"""

        # don't do while init or tracing, since it will be done afterwards anyway
        if self.Raytracer.RaySourceList and self.Raytracer.Rays.N and not self._no_trait_action_flag\
                and not self._Status["InitScene"] and not self._Status["Tracing"]:
            # don't do while tracing, because RayStorage rays are still being generated
            
            self._Status["Drawing"] += 1 
            pyfaceGUI().process_events()

            def background():
                with self.__RayAccessLock:
                    N = self.Raytracer.Rays.N
                    set_size = int(1 + 10**self.RayAmountShown*N)  # number of shown rays
                    sindex = np.random.choice(N, size=set_size, replace=False)  # random choice

                    # make bool array with chosen rays set to true
                    self._RaySelection = np.zeros(N, dtype=bool)
                    self._RaySelection[sindex] = True
                    
                    res = self._getRays()
                
                def on_finish():

                    with self._constant_camera():
                        self._plotRays(*res)
                        self.changeRayColors()
                        self.changeRayRepresentation()
                   
                    # reset picker text 
                    self._onRayPick()

                    self._Status["Drawing"] -= 1
                   
                    # gets unset on first run
                    if self._Status["DisplayingGUI"]:
                        pyfaceGUI.invoke_later(self._setGUILoaded)

                pyfaceGUI.invoke_later(on_finish)
            
            action = Thread(target=background)
            action.start()

    @observe('RayCount, AbsorbMissing')
    def trace(self, event=None) -> None:
        """raytrace in separate thread, after that call :obj:`GUI.filterRays`"""

        if self._no_trait_action_flag:
            return
        
        if self.Raytracer.RaySourceList:

            self._Status["Tracing"] += 1

            # run this in background thread
            def background() -> None:

                AM, N = bool(self.AbsorbMissing), self.RayCount
                self.Raytracer.AbsorbMissing = AM

                with self.__RayAccessLock:
                    self.Raytracer.trace(N=N)

                # execute this after thread has finished
                def on_finish() -> None:
                    with self._no_trait_action():
                        # reset parameters that could have changed during tracing
                        self.AbsorbMissing = self._CheckValFromBool("AbsorbMissing", self.Raytracer.AbsorbMissing)
                        self.RayCount = N
                        
                        # lower the ratio of rays shown so that the GUI does not become to slow
                        if N * 10**self.RayAmountShown > self.MAX_RAYS_SHOWN:
                            self.RayAmountShown = np.log10(self.MAX_RAYS_SHOWN/N)

                    self._Status["Tracing"] -= 1
                    self.replotRays()

                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()
        else:
            pyfaceGUI.invoke_later(self._setGUILoaded)
             
    @observe('RayAlpha')
    def changeRayAlpha(self, event=None) -> None:
        """change opacity of visible rays"""
        if self._RaysPlot is not None:
            self._RaysPlot.actor.property.trait_set(opacity=10**self.RayAlpha)

    @observe("ColoringType")
    def changeRayColors(self, event=None):
        if self._RaysPlot is not None:
            self._assignRayColors()

    @observe("PlottingType")
    def changeRayRepresentation(self, event=None):
        if self._RaysPlot is not None:
            self._RaysPlot.actor.property.representation = 'points' if self.PlottingType == 'Points' else 'surface'
  
    @observe('PosDet, DetectorSelection')
    def changeDetector(self, event=None) -> None:
        """move and replot chosen Detector"""

        if not self._no_trait_action_flag and self.Raytracer.DetectorList and self._DetectorPlotObjects:

            self._Status["ChangingDetector"] += 1

            def background():
                
                def on_finish():

                    DetInd = int(self.DetectorSelection.split(":", 1)[0].split("DET")[1])
                    if self._DetInd != DetInd:
                        self._DetInd = DetInd
                        with self._no_trait_action():
                            self.PosDet = self.Raytracer.DetectorList[self._DetInd].pos[2]

                    xp, yp, zp = self.Raytracer.DetectorList[self._DetInd].pos
                    self.Raytracer.DetectorList[self._DetInd].moveTo([xp, yp, self.PosDet])
                    
                    Surf = self.Raytracer.DetectorList[self._DetInd].Surface.getPlottingMesh(N=self.SURFACE_RES)[2]
                    self._DetectorPlotObjects[self._DetInd][0].mlab_source.trait_set(z=Surf)

                    if self._DetectorPlotObjects[self._DetInd][1]:
                        Cyl = self.Raytracer.DetectorList[self._DetInd].getCylinderSurface(nc=self.SURFACE_RES)[2] 
                        self._DetectorPlotObjects[self._DetInd][1].mlab_source.trait_set(z=Cyl)
                        zl = self.PosDet + self.D_VIS/2
                    else:            
                        obj = self.Raytracer.DetectorList[self._DetInd]
                        zl = (obj.extent[4] + obj.extent[5])/2 

                    self._DetectorPlotObjects[self._DetInd][3].z_position = zl

                    # reinit DetectorList and Selection to update z_pos in detector name
                    self._initDetectorList()
                    with self._no_trait_action():
                        self.DetectorSelection = self.DetectorNames[self._DetInd]  # __DetectorLock gets set another time
                    self.__DetectorLock.release()
                    self._Status["ChangingDetector"] -= 1
                
                self.__DetectorLock.acquire()
                pyfaceGUI().invoke_later(on_finish)

            action = Thread(target=background)
            action.start()

    @observe('DetectorImageButton')
    def showDetectorImage(self, event=None) -> None:
        """render a DetectorImage at the chosen Detector, uses a separate thread"""

        if self.Raytracer.DetectorList:

            self._Status["DetectorImage"] += 1

            def background() -> None:
              
                with self.__DetImageLock:
                    with self.__DetectorLock:
                        with self.__RayAccessLock:
                            # only calculate Image if Raytracer Snapshot, selected Source or ImagePixels changed
                            # otherwise we can replot the old Image with the new visual settings
                            snum = None if not self.DetImageOneSource else self._SourceInd
                            snap = str(self.Raytracer.PropertySnapshot()) + self.DetectorSelection + str(snum)
                            rerender =  snap != self._lastDetSnap or self.lastDetImage is None
                            log, mode, px, dindex, flip = bool(self.LogImage), self.ImageType, self.ImagePixels,\
                                                          self._DetInd, bool(self.FlipDetImage)
                            
                            if rerender:
                                DImg = self.Raytracer.DetectorImage(N=px, ind=dindex, snum=snum)

                    if rerender:
                        self.lastDetImage = DImg
                        self._lastDetSnap = snap
                    else:
                        DImg = self.lastDetImage.copy()

                DImg.rescale(px)
                Imc = DImg.getByDisplayMode(mode, log=log)

                def on_finish() -> None:
                    RImagePlot(DImg, log=log, mode=mode, flip=flip, Imc=Imc)
                    self._Status["DetectorImage"] -= 1

                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()

    @observe('Scene:closing')
    def stop_server(self, event=None):
        if reactor.running:
            reactor.stop()
        if self._Socket is not None:
            self._Socket.close()

    @observe('SourceImageButton')
    def showSourceImage(self, event=None) -> None:
        """render a SourceImage for the chosen Source, uses a separate thread"""

        if self.Raytracer.RaySourceList:

            self._Status["SourceImage"] += 1

            def background() -> None:
    
                with self.__SourceImageLock:
                    with self.__RayAccessLock:
                        # only calculate Image if Raytracer Snapshot, selected Source or ImagePixels changed
                        # otherwise we can replot the old Image with the new visual settings
                        snap = str(self.Raytracer.PropertySnapshot()) + self.SourceSelection
                        rerender =  snap != self._lastSourceSnap or self.lastSourceImage is None
                        log, mode, px, sindex = bool(self.LogImage), self.ImageType, self.ImagePixels, self._SourceInd
                        
                        if rerender:
                            SImg = self.Raytracer.SourceImage(N=px, sindex=sindex)

                    if rerender:
                        self.lastSourceImage = SImg
                        self._lastSourceSnap = snap
                    else:
                        SImg = self.lastSourceImage.copy()

                SImg.rescale(px)
                Imc = SImg.getByDisplayMode(mode, log=log)

                def on_finish() -> None:
                    RImagePlot(SImg, log=log, mode=mode, Imc=Imc)
                    self._Status["SourceImage"] -= 1
                
                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()

    @observe('AutoFocusButton')
    def moveToFocus(self, event=None) -> None:
        """
        Find a Focus.
        The chosen Detector defines the search range for focus finding. 
        Searches are always between lenses or the next outline. 
        Search takes place in a separate thread, after that the Detector is moved to the focus 
        """

        if self.Raytracer.DetectorList and self.Raytracer.RaySourceList:
                
            self._Status["Focussing"] += 1

            def background() -> None:
                snum = None if not self.AFOneSource else self._SourceInd
                mode, PosDet, ret_cost, DetInd = self.FocusType, self.PosDet, bool(self.FocusDebugPlot), self._DetInd
             
                with self.__DetectorLock:
                    with self.__RayAccessLock:
                        zf, zff, r, vals = self.Raytracer.autofocus(mode, PosDet, ret_cost=ret_cost, snum=snum)
                        self.Raytracer.DetectorList[DetInd].moveTo([*self.Raytracer.DetectorList[DetInd].pos[:2], zf])

                # execute this function after thread has finished
                def on_finish() -> None:

                    # update PosDet if the the detector did not change in the meantime 
                    # (due to different threads changing it)
                    if self._DetInd == DetInd:
                        self.PosDet = self.Raytracer.DetectorList[DetInd].pos[2]

                    if self.FocusDebugPlot:
                        AutoFocusDebugPlot(r, vals, zf, zff, f"Focus Finding Using the {mode} Method")
                    
                    self._Status["Focussing"] -= 1

                pyfaceGUI.invoke_later(on_finish)

            action = Thread(target=background)
            action.start()

    @observe('Scene:activated')
    def _plotScene(self, event=None) -> None:
        """Initialize the GUI. Inits a variety of things."""
        self._initRayInfoText()
        self._initStatusText()
        self._initCrosshair()
        self._initKeyboardShortcuts()
        self.plotOrientationAxes()
        
        self.Scene.background = self.BACKGROUND_COLOR

        self.replot()

        self.defaultCameraView()  # this needs to be called after replot, which defines the visual scope
        self._Status["InitScene"] = 0

    @observe('_Status.items')
    def changeStatus(self, event=None) -> None:
        """Update the status info text."""

        if not self._Status["InitScene"] and self._StatusText is not None:
            text = ""
            if self._Status["Tracing"]:          text += "Raytracing...\n"
            if self._Status["Focussing"]:        text += "Focussing...\n"
            if self._Status["DetectorImage"]:    text += "Generating Detector Image...\n"
            if self._Status["ChangingDetector"]: text += "Changing Detector...\n"
            if self._Status["SourceImage"]:      text += "Generating Source Image...\n"
            if self._Status["Drawing"]:          text += "Drawing...\n"

            self._StatusText.text = text

    @observe('CleanerView')
    def changeCleanerView(self, event=None):

        if not self._Status["InitScene"]:
            show = not bool(self.CleanerView)

            if self._OrientationAxes is not None:
                self._OrientationAxes.visible = show

            for rio in self._RefractionIndexPlotObjects:
                if rio[1] is not None:
                    rio[1].text = rio[1].text.replace("ambient\n", "") if not show else ("ambient\n" + rio[1].text)

            # remove descriptions from labels in CleanerView
            for Objects in [self._RaySourcePlotObjects, self._LensPlotObjects,
                            self._FilterPlotObjects, self._AperturePlotObjects, self._DetectorPlotObjects]:
                for num, obj in enumerate(Objects):
                    if obj[3] is not None and obj[4] is not None:
                        label = f"{obj[4].abbr}{num}"
                        label = label if obj[4].desc == "" or not show else label + ": " + obj[4].desc
                        obj[3].text = label

            # replot Ray Pick Text, opacity of default text gets set
            self._onRayPick()

            for ax in self._AxisPlotObjects:
                if ax[0] is not None:
                    ax[0].visible = show
    
    @observe('RayWidth')
    def changeRayWidth(self, event=None) -> None:
        """ set the ray width for the visible rays."""
        if self._RaysPlot is not None:
            self._RaysPlot.actor.property.trait_set(line_width=self.RayWidth, point_size=self.RayWidth)

    @observe('SourceSelection')
    def changeSelectedRaySource(self, event=None) -> None:
        """Updates the Detector Selection and the corresponding properties"""
        if self.Raytracer.RaySourceList:
            self._SourceInd = int(self.SourceSelection.split(":", 1)[0].split("RS")[1])
   
    # run on button press, since observe('Cmd') would only register if the string changes
    @observe('RunButton')
    def sendCmd(self, event=None) -> None:
        """Updates the Detector Selection and the corresponding properties"""
        if self._Socket is None:  # create on demand
            self._Socket = socket.socket()
            self._Socket.connect(('localhost', TCPServer.TCP_port))

        if self._Cmd != "":
            self.Command_History += self._Cmd + "\n"
            self._Socket.send(self._Cmd.encode())

    # rescale axes texts when the window was resized
    # for some reasons these are the only ones having no text_scaling_mode = 'none' option
    # TODO this is the only trait so far that does not work with @observe, like:
    # @observe("Scene:scene_editor:interactor:size:items:value")
    # since size is a numpy array and not a list
    @observe("Scene:scene_editor:busy")
    def _resizeSceneElements(self, event) -> None:
        """Handles GUI window size changes. Fixes incorrect scaling by mayavi"""

        if self.Scene.scene_editor.busy and self.Scene.scene_editor.interactor is not None:
            SceneSize = self.Scene.scene_editor.interactor.size

            if self._SceneSize[0] != SceneSize[0] or self._SceneSize[1] != SceneSize[1]:
                if SceneSize[0] and SceneSize[1]:  # size coordinates non-zero

                    # average  of x and y size for former and current scene size
                    ch1 = (self._SceneSize[0] + self._SceneSize[1]) / 2
                    ch2 = (SceneSize[0] + SceneSize[1]) / 2

                    # update font factor so font size stays the same
                    for ax in self._AxisPlotObjects:
                        ax[0].axes.font_factor *= ch1/ch2

                    # rescale orientation axes
                    if self._OrientationAxes is not None:
                        self._OrientationAxes.widgets[0].zoom *= ch1/ch2
                         
                    if self._RaysPlot is not None:
                        bar = self._RaysPlot.parent.scalar_lut_manager.scalar_bar_representation
                        bar.position2 = np.array([bar.position2[0]*self._SceneSize[0]/SceneSize[0], 
                                                  bar.position2[1]*self._SceneSize[1]/SceneSize[1]])
                        bar.position = np.array([bar.position[0], (1-bar.position2[1])/2])

                    # set current window size
                    self._SceneSize = SceneSize

