import gc  # garbage collector stats
import time  # provides sleeping
import warnings  # show warnings
import traceback  # traceback printing
from threading import Thread, Lock  # threading
from typing import Callable, Any  # typing types
from contextlib import contextmanager  # context manager for _no_trait_action()

import numpy as np  # calculations
import matplotlib.pyplot as plt  # closing plot windows

# enforce qt backend
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt'

from pyface.qt import QtGui  # closing UI elements
from pyface.api import GUI as pyface_gui  # invoke_later() method

# traits types and UI elements
from traitsui.api import View, Item, HSplit, CheckListEditor, TextEditor, CodeEditor
from traitsui.api import Group as TGroup
from traits.api import HasTraits, Range, Instance, observe, Str, Button, Enum, List, Dict, Float, Bool

from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi.sources.parametric_surface import ParametricSurface  # provides outline and axes

# provides types and plotting functionality, most of these are also imported for the TCP protocol scope
from ..tracer import Lens, Filter, Aperture, RaySource, Detector, Raytracer, RImage, RefractionIndex,\
                     Surface, SphericalSurface, DataSurface, FunctionSurface, ConicSurface, RectangularSurface,\
                     RingSurface, CircularSurface, TiltedSurface, Spectrum, LightSpectrum, TransmissionSpectrum,\
                     Point, Line, TMA, Group, Marker, AsphericSurface, IdealLens
from ..tracer.geometry.element import Element  # Element type
from ..plots import r_image_plot, r_image_cut_plot, autofocus_cost_plot, spectrum_plot  # different plots

from ..tracer import presets  # for send_cmd scope
from ..tracer import color  # for visible wavelength range
from ..tracer import misc  # for partMask function

from .property_browser import PropertyBrowser  # dictionary browser

from ..__metadata__ import __version__

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
#                   changes of detector_selection and det_pos are intercepted and prevented in the main thread
# * _RayAccessLock: locks the RayStorage for reading or writing. Needed for moveToFocus(), DetectorImage(),
#                   SourceImage(), trace() and replotRays()
#                   actually also needed for the picker, but we don't want to block the main thread,
#                   so we prevent picking in this case


class TraceGUI(HasTraits):

    ####################################################################################################################
    # UI objects

    SURFACE_RES: int = 150
    """Surface sampling count in each dimension"""

    MAX_RAYS_SHOWN: int = 10000
    """Maximum of rays shown in visualization"""

    LABEL_STYLE: dict = dict(font_size=11, bold=True, font_family="courier", shadow=True)
    """Standard Text Style. Used for object labels, legends and axes"""

    TEXT_STYLE: dict = dict(font_size=11, font_family="courier", italic=False, bold=True)
    """Standard Text Style. Used for object labels, legends and axes"""

    INFO_STYLE: dict = dict(font_size=13, bold=True, font_family="courier", shadow=True, italic=False)
    """Info Text Style. Used for status messages and interaction overlay"""

    ##########
    scene: Instance = Instance(MlabSceneModel, args=())
    """the mayavi scene"""

    # Ranges

    ray_count: Range = Range(1, Raytracer.MAX_RAYS, 200000, desc='Number of rays for Simulation', enter_set=True,
                             auto_set=False, label="Rays", mode='text')
    """Number of rays for raytracing"""
    
    filter_constant: Range = Range(0.5, 20., 1., desc='filter constant', enter_set=True,
                             auto_set=False, label="Limit (µm)", mode='text')
    """gaussian filter constant standard deviation"""

    ray_amount_shown: Range = Range(1, 10000, 2000, desc='Percentage of N Which is Drawn', enter_set=False,
                                    auto_set=False, label="Rays Shown", mode='logslider')
    """Number of rays shown in mayavi scenen"""

    det_pos: Range = Range(low='_det_pos_min', high='_det_pos_max', value='_det_pos_max', mode='slider',
                           desc='z-Position of the Detector', enter_set=True, auto_set=True, label="z_pos")
    """z-Position of Detector. Lies inside z-range of :obj:`Backend.raytracer.outline`"""

    ray_opacity: Range = Range(1e-5, 1, 0.01, desc='Opacity of the rays/Points', enter_set=True,
                             auto_set=True, label="Opacity", mode='logslider')
    """opacity of shown ray"""

    ray_width: Range = Range(1.0, 20.0, 1, desc='Ray Linewidth or Point Size', enter_set=True,
                             auto_set=True, label="Width")
    """Width of rays shown."""

    cut_value: Float = Float(0, desc='value for specified cut dimension',
                             enter_set=False, label="value", mode='text')
    """numeric cut value for chosen image cut dimension"""

    # Checklists (with capitalization workaround from https://stackoverflow.com/a/23783351)
    # this should basically be bool values, but because we want to have checkboxes with text right of them
    # we need to embed a CheckListEditor in a List.
    # To assign bool values we need a workaround in __setattr__

    absorb_missing: List = List(editor=CheckListEditor(values=["Absorb Rays Missing Lens"], format_func=lambda x: x),
                                desc="if Rays are Absorbed when not Hitting a Lens")
    """Boolean value for absorbing rays missing Lens. Packed as Checkbox into a :obj:`List`"""

    log_image: List = List(editor=CheckListEditor(values=['Logarithmic Scaling'], format_func=lambda x: x),
                           desc="if Logarithmic Values are Plotted")
    """Boolean value for a logarithmic image visualization. Packed as Checkbox into a :obj:`List`"""

    flip_det_image: List = List(editor=CheckListEditor(values=['Flip Detector Image'], format_func=lambda x: x),
                                desc="if detector image should be rotated by 180 degrees")
    """Boolean value for flipping the image (rotating it by 180°). Packed as Checkbox into a :obj:`List`"""

    focus_cost_plot: List = List(editor=CheckListEditor(values=['Plot Cost Function'], format_func=lambda x: x),
                                 desc="if cost function is shown")
    """Show a plot of the optimization cost function for the focus finding"""

    minimalistic_view: List = List(editor=CheckListEditor(values=['More Minimalistic Scene'], format_func=lambda x: x),
                                   desc="if some scene elements should be hidden")
    """More minimalistic scene view. Axes are hidden and labels shortened."""

    af_one_source: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'],
                                                      format_func=lambda x: x),
                               desc="if autofocus only uses currently selected source")
    """Use only rays from selected source for focus finding"""

    det_image_one_source: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'],
                                                             format_func=lambda x: x),
                                      desc="if DetectorImage only uses currently selected source")
    """Use only rays from selected source for the detector image"""

    det_spectrum_one_source: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'],
                                                                format_func=lambda x: x),
                                         desc="if Detector Spectrum only uses currently selected source")
    """Use only rays from selected source for the detector spectrum"""

    show_all_warnings: List = List(editor=CheckListEditor(values=['Show All Warnings'], format_func=lambda x: x),
                                   desc="if all warnings are shown")
    """show debugging messages"""

    activate_filter: List = List(editor=CheckListEditor(values=['Activate Filter'], format_func=lambda x: x),
                                   desc="if gaussian filter is applied")
    """gaussian blur filter estimating resolution limit"""

    wireframe_surfaces: \
        List = List(editor=CheckListEditor(values=['Show Surfaces as Wireframe'], format_func=lambda x: x),
                    desc="if surfaces are shown as wireframe")
    """Sets the surface representation to normal or wireframe view"""

    garbage_collector_stats: \
        List = List(editor=CheckListEditor(values=['Show Garbage Collector Stats'], format_func=lambda x: x),
                    desc="if stats from garbage collector are shown")
    """Show stats from garbage collector"""

    raytracer_single_thread: \
        List = List(editor=CheckListEditor(values=['No Multithreading in Raytracer'], format_func=lambda x: x),
                    desc="if raytracer backend uses only one thread")
    """limit raytracer backend operation to one thread (excludes the TraceGUI)"""
    
    command_dont_wait: \
        List = List(editor=CheckListEditor(values=["Don't Wait for Idle State"], format_func=lambda x: x),
                    desc="don't wait for idle state while running a command. Can lead to race conditions.")
    """don't wait for idle state before running an user command.
    Useful for debugging, but can lead to race conditions"""
    
    command_dont_replot: \
        List = List(editor=CheckListEditor(values=["No Automatic Scene Replotting"], format_func=lambda x: x),
                    desc="Don't replot scene after geometry change")
    """don't replot scene after raytracer geometry change"""

    maximize_scene: \
        List = List(editor=CheckListEditor(values=["Maximize Scene (Press h in Scene)"], format_func=lambda x: x),
                    desc="Maximizes Scene, hides side menu and toolbar")
    """maximize mode. Menu and toolbar hidden"""
    
    high_contrast: \
        List = List(editor=CheckListEditor(values=["High Contrast Mode"], format_func=lambda x: x),
                    desc="show objects in black on white background.")
    """high contrast mode without object colors and with white background"""
    
    # Enums

    plotting_types: list = ['Rays', 'Points']  #: available ray representations
    plotting_type: Enum = Enum(*plotting_types, desc="Ray Representation")
    """Ray Plotting Type"""

    coloring_types: list = ['Plain', 'Power', 'Wavelength', 'Source', 'Polarization xz', 'Polarization yz']  
    """available ray coloring modes"""
    coloring_type: Enum = Enum(*coloring_types, desc="Ray Property to color the Rays With")
    """Ray Coloring Mode"""

    image_type: Enum = Enum(*RImage.display_modes, desc="Image Type Presented")
    """Image Type"""
   
    projection_method_enabled: Bool = False  #: if projection method is selectable
    projection_method: Enum = Enum(*SphericalSurface.sphere_projection_methods, desc="Projection Method for spherical detectors")
    """sphere surface projection method"""

    focus_type: Enum = Enum(*Raytracer.autofocus_methods, desc="Method for Autofocus")
    """Focus Finding Mode from raytracer.AutofocusModes"""

    cut_dimension: Enum = Enum(["y", "x"], desc="image cut dimension")
    """dimension for image cut"""

    source_names: List = List()  #: short names for raytracer ray sources
    source_selection: Enum = Enum(values='source_names', desc="Source Selection for Source Image")
    """Source Selection. Holds the name of one of the Sources."""

    detector_names: List  = List()  #: short names for raytracer detectors
    detector_selection: Enum = Enum(values='detector_names', desc="Detector Selection")
    """Detector Selection. Holds the name of one of the Detectors."""

    image_pixels: Enum = Enum(RImage.SIZES,
                                desc='Detector Image Pixels in Smaller of x or y Dimension',
                                label="Pixels_xy")
    """Image Pixel value for Source/Detector Image. This the number of pixels for the smaller image side."""


    # Buttons

    _detector_image_button:      Button = Button(label="Detector Image", desc="Generating a Detector Image")
    _run_button:                 Button = Button(label="Run", desc="runs the specified command")
    _clear_button:               Button = Button(label="Clear", desc="clear command history")
    _property_browser_button:    Button = Button(label="Open Property Browser", desc="property browser is opened")
    _source_spectrum_button:     Button = Button(label="Source Spectrum", desc="sources spectrum is shown")
    _detector_spectrum_button:   Button = Button(label="Detector Spectrum", desc="detector spectrum is shown")
    _source_cut_button:          Button = Button(label="Source Image Cut", desc="source image cut is shown")
    _detector_cut_button:        Button = Button(label="Detector Image Cut", desc="detector image cut is shown")
    _source_image_button:        Button = Button(label="Source Image", desc="Source Image of the Chosen Source")
    _auto_focus_button:          Button = Button(label="Find Focus", desc="Finding the Focus Between the Lenses"
                                                                         " around the Detector Position")

    # Strings and Labels

    _cmd:                        Str = Str()
    _autofocus_information:      Str = Str()
    _execute_label:              Str = Str('Command:')
    _history_label:              Str = Str('History:')
    _command_history:            Str = Str('')
    _debug_options_label:        Str = Str('Debugging Options:')
    _debug_command_label:        Str = Str('Command Running Options:')
    _spectrum_label:             Str = Str('Generate Spectrum:')
    _image_label:                Str = Str('Render Image:')
    _geometry_label:             Str = Str('Geometry:')
    _autofocus_label:            Str = Str('Autofocus:')
    _property_label:             Str = Str('Properties:')
    _trace_label:                Str = Str('Trace Settings:')
    _plotting_label:             Str = Str('Ray Plotting Modes:')
    _ray_visual_label:           Str = Str('Ray Visual Settings:')
    _ui_visual_label:            Str = Str('Scene and UI Settings:')
    _autofocus_output_label:     Str = Str('Optimization Output:')
    _filter_label:               Str = Str('Resolution Filter (Gaussian):')
    _whitespace_label:           Str = Str('')

    _separator: Item = Item("_whitespace_label", style='readonly', show_label=False, width=210)

    _scene_not_maximized: Bool = Bool(True)

    _status: Dict = Dict(Str())

    # size of the mlab scene
    _scene_size0 = [1100, 800]  # default size
    # with the window bar this should still fit on a 900p screen
    _scene_size = _scene_size0.copy()  # will hold current size

    ####################################################################################################################
    # UI view creation

    view = View(
                HSplit(
                    TGroup(
                        TGroup(
                            Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                                 height=_scene_size0[1], width=_scene_size0[0], show_label=False),
                             ),
                        layout="split",
                    ),
                    TGroup(
                        TGroup(
                            _separator,
                            Item("_trace_label", style='readonly', show_label=False, emphasized=True),
                            Item('ray_count'),
                            Item('absorb_missing', style="custom", show_label=False),
                            _separator,
                            _separator,
                            Item("_plotting_label", style='readonly', show_label=False, emphasized=True),
                            Item("plotting_type", label='Plotting'),
                            Item("coloring_type", label='Coloring'),
                            _separator,
                            _separator,
                            Item("_ray_visual_label", style='readonly', show_label=False, emphasized=True),
                            Item('ray_amount_shown'),
                            Item('ray_opacity'),
                            Item('ray_width'),
                            _separator,
                            _separator,
                            Item("_ui_visual_label", style='readonly', show_label=False, emphasized=True),
                            Item('minimalistic_view', style="custom", show_label=False),
                            Item('maximize_scene', style="custom", show_label=False),
                            Item('high_contrast', style="custom", show_label=False),
                            _separator,
                            _separator,
                            Item("_property_label", style='readonly', show_label=False, emphasized=True),
                            Item('_property_browser_button', show_label=False),
                            _separator,
                            label="Main"
                            ),
                        TGroup(
                            _separator,
                            Item("_geometry_label", style='readonly', show_label=False, emphasized=True),
                            Item('source_selection', label="Source"),
                            Item('detector_selection', label="Detector"),
                            Item('det_pos', label="Det_z"),
                            _separator,
                            Item("_image_label", style='readonly', show_label=False, emphasized=True),
                            Item('image_type', label='Mode'),
                            Item('projection_method', label='Projection', enabled_when="projection_method_enabled"),
                            Item('image_pixels'),
                            Item('log_image', style="custom", show_label=False),
                            Item('flip_det_image', style="custom", show_label=False),
                            Item('det_image_one_source', style="custom", show_label=False),
                            Item('_source_image_button', show_label=False),
                            Item('_detector_image_button', show_label=False),
                            _separator,
                            Item("cut_dimension", label="Cut at"),
                            Item("cut_value", label="Value"),
                            Item('_source_cut_button', show_label=False),
                            Item('_detector_cut_button', show_label=False),
                            _separator,
                            Item("_filter_label", style='readonly', show_label=False, emphasized=True),
                            Item('activate_filter', style="custom", show_label=False),
                            Item('filter_constant'),
                            label="Image",
                            ),
                        TGroup(
                            _separator,
                            Item("_geometry_label", style='readonly', show_label=False, emphasized=True),
                            Item('source_selection', label="Source"),
                            Item('detector_selection', label="Detector"),
                            Item('det_pos', label="Det_z"),
                            _separator,
                            _separator,
                            Item("_spectrum_label", style='readonly', show_label=False, emphasized=True),
                            Item('_source_spectrum_button', show_label=False),
                            Item('det_spectrum_one_source', style="custom", show_label=False),
                            Item('_detector_spectrum_button', show_label=False),
                            label="Spectrum",
                            ),
                        TGroup(
                            _separator,
                            Item("_geometry_label", style='readonly', show_label=False, emphasized=True),
                            Item('source_selection', label="Source"),
                            Item('detector_selection', label="Detector"),
                            Item('det_pos', label="Det_z"),
                            _separator,
                            _separator,
                            Item("_autofocus_label", style='readonly', show_label=False, emphasized=True),
                            Item('focus_type', label='Mode'),
                            Item('af_one_source', style="custom", show_label=False),
                            Item('focus_cost_plot', style="custom", show_label=False),
                            Item('_auto_focus_button', show_label=False),
                            _separator,
                            Item("_autofocus_output_label", style='readonly', show_label=False),
                            Item("_autofocus_information", show_label=False, style="custom"),
                            _separator,
                            label="Focus",
                            ),
                        TGroup(
                            TGroup(
                                Item("_execute_label", style='readonly', show_label=False, emphasized=True),
                                Item('_cmd', editor=CodeEditor(), show_label=False, style="custom"), 
                                Item("_run_button", show_label=False),
                                style_sheet="*{max-height:200px; max-width:320px}"
                            ),
                            _separator,
                            TGroup(
                                Item("_history_label", style='readonly', show_label=False, emphasized=True),
                                Item("_command_history", editor=CodeEditor(), show_label=False, style="custom"),
                                Item("_clear_button", show_label=False),
                                style_sheet="*{max-width:320px; max-height:500px}"
                            ),
                            label="Run"
                            ),
                        HSplit(  # hsplit with whitespace group so we have a left margin in this tab
                            TGroup(Item("_whitespace_label", style='readonly', show_label=False, width=50)),
                            TGroup(
                                _separator,
                                _separator,
                                Item("_debug_options_label", style='readonly', show_label=False, emphasized=True),
                                Item('raytracer_single_thread', style="custom", show_label=False),
                                Item('show_all_warnings', style="custom", show_label=False),
                                Item('garbage_collector_stats', style="custom", show_label=False),
                                Item('wireframe_surfaces', style="custom", show_label=False),
                                _separator,
                                _separator,
                                Item("_debug_command_label", style='readonly', show_label=False, emphasized=True),
                                Item('command_dont_replot', style="custom", show_label=False),
                                Item('command_dont_wait', style="custom", show_label=False),
                                _separator),
                            label="Debug",
                            ),
                        layout="tabbed",
                        visible_when="_scene_not_maximized",
                        ),
                    ),
                resizable=True,
                title=f"Optrace {__version__}"
                )
    """the UI view"""

    ####################################################################################################################
    # Class constructor

    def __init__(self, raytracer: Raytracer, **kwargs) -> None:
        """
        The extra bool parameters are needed to assign the List-Checklisteditor to bool values in the GUI.__init__.
        Otherwise the user would assign bool values to a string list.

        :param RT:
        :param kwargs:
        """
        # lock object for multithreading
        self.__detector_lock = Lock()  # when detector is changed, moved or used for rendering
        self.__det_image_lock = Lock()  # when last detector image changed
        self.__source_image_lock = Lock()  # when last source image changed
        self.__ray_access_lock = Lock()  # when rays read or written

        self.raytracer: Raytracer = raytracer
        """reference to the raytracer"""

        # ray properties
        self._ray_selection = np.array([])  # indices for subset of all rays in raytracer
        self._ray_property_dict = {}  # properties of shown rays
        self._rays_plot_scatter = None  # plot for scalar ray values
        self._rays_plot = None  # plot for visualization of rays/points
        self._ray_text = None  # textbox for ray information
        self._ray_text_parent = None
        self._ray_picker = None
        self._space_picker = None

        # plot object lists
        self._lens_plots = []
        self._axis_plots = []
        self._filter_plots = []
        self._outline_plots = []
        self._aperture_plots = []
        self._detector_plots = []
        self._ray_source_plots = []
        self._marker_plots = []
        self._index_box_plots = []
        self._crosshair = None
        self._orientation_axes = None

        # minimal/maximal z-positions of Detector_obj
        self._det_pos_min = self.raytracer.outline[4]
        self._det_pos_max = self.raytracer.outline[5]

        self._det_ind = 0
        if len(self.raytracer.detectors):
            self.det_pos = self.raytracer.detectors[self._det_ind].pos[2]
        self._source_ind = 0

        self.silent = False
        self._exit = False

        self._last_det_snap = None
        self.last_det_image = None
        self._last_source_snap = None
        self.last_source_image = None

        # default value for image_pixels
        self.image_pixels = 189

        # assign scene background color
        self._set_colors()
        # self.scene.background = self.BACKGROUND_COLOR
        
        # set the properties after this without leading to a re-draw
        self._no_trait_action_flag = False
       
        # these properties should not be set, since they have to be initialized first
        # and setting as inits them would lead to issues
        forbidden = ["source_selection", "detector_selection", "det_pos"]
        if any(key in kwargs for key in forbidden):
            raise RuntimeError(f"Assigning an initial value for properties '{forbidden}' is not supported")

        with self._no_trait_action():

            # add true default parameters
            if "absorb_missing" not in kwargs:
                kwargs["absorb_missing"] = True

            # convert bool values to list entries for List()
            for key, val in kwargs.items():
                if key in ["absorb_missing", "minimalistic_view", "log_image", "flip_det_image", "wireframe_surfaces",
                           "focus_cost_plot", "af_one_source", "det_image_one_source", "det_spectrum_one_source",
                           "raytracer_single_thread", "show_all_warnings", "garbage_collector_stats", "high_contrast",
                           "command_dont_wait", "command_dont_replot", "maximize_scene", "activate_filter"]\
                                   and isinstance(val, bool):
                    kwargs[key] = [self._trait(key, 0).editor.values[0]] if val else []

            # define Status dict
            self._status_text = None
            self._status_text_parent = None
            self._status.update(dict(InitScene=1, DisplayingGUI=1, Tracing=0, Drawing=0, Focussing=0,
                                     DetectorImage=0, SourceImage=0, ChangingDetector=0, SourceSpectrum=0,
                                     DetectorSpectrum=0, RunningCommand=0))

            super().__init__(**kwargs)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        # workaround so we can set bool values to some List settings
        if isinstance(val, bool) and\
            key in ["absorb_missing", "minimalistic_view", "log_image", "flip_det_image", "focus_cost_plot",
                    "det_spectrum_one_source", "af_one_source", "det_image_one_source", "raytracer_single_thread",
                    "show_all_warnings", "wireframe_surfaces", "garbage_collector_stats", "high_contrast",
                    "command_dont_wait", "command_dont_replot", "maximize_scene", "activate_filter"]:
            val = [self._trait(key, 0).editor.values[0]] if val else []

        # the detector can only be changed, if it isn't locked (__detetector_lock)
        # but while waiting for the release, the detector position and therefore its label can change
        # making the old name invalid
        # workaround: get index from name and use updated name
        if key == "detector_selection":
            det_ind = int(val.split(":", 1)[0].split("DET")[1])
            val = self.detector_names[det_ind]

        super().__setattr__(key, val)

    ####################################################################################################################
    # Helpers

    @contextmanager
    def _no_trait_action(self, *args, **kwargs) -> None:
        """context manager that sets and resets the _no_trait_action_flag"""
        self._no_trait_action_flag = True
        try:
            yield
        finally:
            self._no_trait_action_flag = False

    @contextmanager
    def _try(self, *args, **kwargs) -> bool:
        """
        execute an environment in a try block, print exceptions and return if exceptions have occurred
        example usage:

        with self._try() as error:
            something()
        if error:
            print("error occured, quiting")
            quit()

        :return: if an exception has occured
        """
        
        error = []  # evals to false

        try:
            yield error  # return reference to list

        except Exception as err:

            if not self.silent:
                traceback.print_exc()  # print traceback

            error[:] = [True]  # change reference, now evals to true
    
    @contextmanager
    def _constant_camera(self, *args, **kwargs) -> None:
        """context manager the saves and restores the camera view"""
        cc_traits_org = self.scene.camera.trait_get("position", "focal_point", "view_up", "view_angle",
                                                    "clipping_range", "parallel_scale")
        try:
            yield
        finally:
            self.scene.camera.trait_set(**cc_traits_org)

    def _do_in_main(self, f: Callable, *args, **kw) -> None:
        """execute a function in the GUI main thread"""
        pyface_gui.invoke_later(f, *args, **kw)
        pyface_gui.process_events()

    def _set_in_main(self, trait: str, val: Any) -> None:
        """assign a property in the GUI main thread"""
        pyface_gui.set_trait_later(self, trait, val)
        pyface_gui.process_events()

    ####################################################################################################################
    # Plotting functions

    # remove visual objects from raytracer geometry
    def __remove_objects(self, objs) -> None:
        for obj in objs:
            for obji in obj[:4]:
                if obji is not None:
                    if obji.parent.parent.parent in self.scene.mayavi_scene.children:
                        obji.parent.parent.parent.remove()
                    if obji.parent.parent in self.scene.mayavi_scene.children:
                        obji.parent.parent.remove()

        objs[:] = []
    
    def _plot_lenses(self) -> None:
        """replot all lenses from raytracer"""
        self.__remove_objects(self._lens_plots)

        for num, L in enumerate(self.raytracer.lenses):
            t = self._plot_element(L, num, self._lens_color[:3], self._lens_alpha)
            self._lens_plots.append(t)

    def _plot_apertures(self) -> None:
        """replot all apertures from raytracer"""
        self.__remove_objects(self._aperture_plots)

        for num, AP in enumerate(self.raytracer.apertures):
            t = self._plot_element(AP, num, self._aperture_color, 1)
            self._aperture_plots.append(t)

    def _plot_filters(self) -> None:
        """replot all filter from raytracer"""
        self.__remove_objects(self._filter_plots)

        for num, F in enumerate(self.raytracer.filters):
            fcolor = F.get_color()
            alpha = 0.1 + 0.899*fcolor[3]  # offset in both directions, ensures visibility and see-through
            t = self._plot_element(F, num, fcolor[:3] if not self.high_contrast else self.scene.foreground, alpha)
            self._filter_plots.append(t)

    def _plot_detectors(self) -> None:
        """replot all detectors from taytracer"""
        self.__remove_objects(self._detector_plots)

        for num, Det in enumerate(self.raytracer.detectors):
            t = self._plot_element(Det, num, self._detector_color[:3], self._detector_alpha)
            self._detector_plots.append(t)

    def _plot_ray_sources(self) -> None:
        """replot all ray sources from raytracr"""
        self.__remove_objects(self._ray_source_plots)

        for num, RS in enumerate(self.raytracer.ray_sources):
            t = self._plot_element(RS, num, (1, 1, 1), self._raysource_alpha, spec=False, light=False)
            self._ray_source_plots.append(t)

    def _plot_outline(self) -> None:
        """replot the raytracer outline"""
        self.__remove_objects(self._outline_plots)

        self.scene.engine.add_source(ParametricSurface(name="Outline"), self.scene)
        a = self.scene.mlab.outline(extent=self.raytracer.outline.copy(), color=self._outline_color)
        a.actor.actor.pickable = False  # only rays should be pickable

        self._outline_plots.append((a,))

    def _plot_orientation_axes(self) -> None:
        """plot orientation axes"""

        if self._orientation_axes is not None:
            self._orientation_axes.remove()

        self.scene.engine.add_source(ParametricSurface(name="Orientation Axes"), self.scene)

        # show axes indicator
        self._orientation_axes = self.scene.mlab.orientation_axes()
        self._orientation_axes.text_property.trait_set(**self.TEXT_STYLE)
        self._orientation_axes.marker.interactive = 0  # make orientation axes non-interactive
        self._orientation_axes.visible = not bool(self.minimalistic_view)
        self._orientation_axes.widgets[0].viewport = [0, 0, 0.1, 0.15]

        # turn of text scaling of orientation_axes
        self._orientation_axes.widgets[0].orientation_marker.x_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self._orientation_axes.widgets[0].orientation_marker.y_axis_caption_actor2d.text_actor.text_scale_mode = 'none'
        self._orientation_axes.widgets[0].orientation_marker.z_axis_caption_actor2d.text_actor.text_scale_mode = 'none'

    def _plot_axes(self) -> None:
        """plot cartesian axes"""

        # save old font factor. This is the one we adapted constantly in self._resizeSceneElements()
        ff_old = self._axis_plots[0][0].axes.font_factor if self._axis_plots else 0.65

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

            # a.property.trait_set(display_location='background')
            a.title_text_property.trait_set(**self.TEXT_STYLE, color=self._axis_color, opacity=self._axis_alpha)
            a.label_text_property.trait_set(**self.TEXT_STYLE, color=self._axis_color, opacity=self._axis_alpha)
            a.actors[0].pickable = False
            a.visible = not bool(self.minimalistic_view)

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

    def _plot_index_boxes(self) -> None:
        """plot outlines for ambient refraction index regions"""

        self.__remove_objects(self._index_box_plots)

        # TODO don't plot any boxes if relevant boxes would be too small

        # sort Element list in z order
        Lenses = sorted(self.raytracer.lenses, key=lambda Element: Element.pos[2])

        # create n list and z-boundary list
        nList = [self.raytracer.n0] + [Element.n2 for Element in Lenses] + [self.raytracer.n0]
        BoundList = [self.raytracer.outline[[4, 4]]] +\
                    [(np.mean(Element.front.extent[4:]), np.mean(Element.back.extent[4:])) for Element in Lenses]\
                    + [self.raytracer.outline[[5, 5]]]

        # replace None values of Lenses n2 with the ambient n0
        nList = [self.raytracer.n0 if ni is None else ni for ni in nList]

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

            # skip plotting if bound difference is small (e.g. for lenses in contact, like achromatic one)
            bdiff = (BoundList[i+1][0] - BoundList[i][1])
            if bdiff < 1e-3 or np.abs(bdiff) < 1e-3:
                continue

            # plot outline
            self.scene.engine.add_source(ParametricSurface(name=f"Refraction Index Outline {i}"), self.scene)
            outline = self.scene.mlab.outline(extent=[*self.raytracer.outline[:4], BoundList[i][1], BoundList[i + 1][0]],
                                              color=self._outline_color)
            outline.outline_mode = 'cornered'
            outline.actor.actor.pickable = False  # only rays should be pickable
            outline.outline_filter.corner_factor = 0.1

            # label position
            x_pos = self.raytracer.outline[0] + (self.raytracer.outline[1] - self.raytracer.outline[0]) * 0.05
            y_pos = np.mean(self.raytracer.outline[2:4])
            z_pos = np.mean([BoundList[i][1], BoundList[i+1][0]])

            # plot label
            label = nList[i].get_desc()
            text_ = f"ambient\nn={label}" if not self.minimalistic_view else f"n={label}"
            text = self.scene.mlab.text(x_pos, y_pos, z=z_pos, text=text_, name="Label")
            text.actor.text_scale_mode = 'none'
            text.property.trait_set(**self.TEXT_STYLE, justification="center", frame=True,
                                    frame_color=self._subtle_color, color=self._axis_color, opacity=self._axis_alpha)
            text.property.bold = False
            # append plot objects
            self._index_box_plots.append((outline, text, None))

    def _set_colors(self) -> None:

        if self.scene is not None:
            self.scene.background = (0.205, 0.19, 0.19) if not self.high_contrast else (1, 1, 1)
            self.scene.foreground = (1, 1, 1) if not self.high_contrast else (0, 0, 0)

        self._lens_color = (0.63, 0.79, 1.00) if not self.high_contrast else self.scene.foreground
        self._lens_alpha = 0.35
        self._aperture_color = (0, 0, 0)
        self._detector_color = (0.10, 0.10, 0.10) if not self.high_contrast else self.scene.foreground
        self._detector_alpha = 0.9
        self._subtle_color = (0.3, 0.3, 0.3) if not self.high_contrast else (0.7, 0.7, 0.7)
        self._crosshair_color = (1, 0, 0)
        self._marker_color = (0, 1, 0) if not self.high_contrast else self.scene.foreground
        self._outline_color = (0, 0, 0)
        self._raysource_alpha = 0.55
        self._axis_color = (1, 1, 1) if not self.high_contrast else (0, 0, 0)
        self._axis_alpha = 0.55
        self._info_opacity = 0.2

    def _plot_element(self,
                      obj:      Element,
                      num:      int,
                      color:    tuple,
                      alpha:    float,
                      spec:     bool = True,
                      light:    bool = True)\
            -> tuple:
        """plotting of a Element. Gets called from plotting for Lens, Filter, Detector, RaySource.

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
            a = self.scene.mlab.mesh(C[0], C[1], C[2], color=color, opacity=alpha,
                                     name=f"{type(obj).__name__} {num} {surf_type} surface")

            # make non-pickable, so it does not interfere with our ray picker
            a.actor.actor.pickable = False
            a.actor.property.representation = "wireframe" if self.wireframe_surfaces else "surface"

            a.actor.actor.property.lighting = light
            if spec:
                a.actor.property.trait_set(specular=0.5, ambient=0.25)
            return a

        # decide what to plot
        plotFront = isinstance(obj.front, Surface)
        plotCyl = plotFront
        plotBack = obj.back is not None and isinstance(obj.back, Surface)

        # Element consisting only of Point or Line: nothing plotted except label -> add parent object
        if not (plotFront or plotCyl or plotBack):
            self.scene.engine.add_source(ParametricSurface(name=f"{type(obj).__name__} {num}"), self.scene)

        # plot front
        a = plot(obj.front.get_plotting_mesh(N=self.SURFACE_RES), "front") if plotFront else None

        # cylinder between surfaces
        b = plot(obj.get_cylinder_surface(nc=2*self.SURFACE_RES), "cylinder") if plotCyl else None

        # use wireframe mode, so edge is always visible, even if it is infinitesimal small
        if plotCyl and (not obj.has_back() or (obj.extent[5] - obj.extent[4] < 0.1)):
            b.actor.property.trait_set(representation="wireframe", lighting=False, line_width=1.5, opacity=alpha/2)

        # calculate middle center z-position
        if obj.has_back():
            zl = (obj.front.get_values(np.array([obj.front.extent[1]]), np.array([obj.front.pos[1]]))[0] \
                  + obj.back.get_values(np.array([obj.back.extent[1]]), np.array([obj.back.pos[1]]))[0]) / 2
        else:
            # 10/40 values in [0, 2pi] -> z - value at 90 deg relative to x axis in xy plane
            zl = obj.front.get_edge(40)[2][10] if isinstance(obj.front, Surface) else obj.pos[2]

        # object label
        label = f"{obj.abbr}{num}"
        # add description if any exists. But only if we are not in "minimalistic_view" displaying mode
        label = label if obj.desc == "" or bool(self.minimalistic_view) else label + ": " + obj.desc
        text = self.scene.mlab.text(x=obj.extent[1], y=obj.pos[1], z=zl, text=label, name="Label")
        text.property.trait_set(**self.LABEL_STYLE, justification="center")
        text.actor.text_scale_mode = 'none'

        # plot BackSurface if one exists
        c = plot(obj.back.get_plotting_mesh(N=self.SURFACE_RES), "back") if plotBack else None

        return a, b, c, text, obj

    def _get_rays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ray/Point Plotting with a specified scalar coloring
        :return:
        """

        p_, s_, pol_, w_, wl_, snum_, n_ = self.raytracer.rays.rays_by_mask(self._ray_selection, normalize=True)
        self._ray_property_dict.update(p=p_, s=s_, pol=pol_, w=w_, wl=wl_, snum=snum_, n=n_,
                                       index=np.where(self._ray_selection)[0])

        # get flattened list of the coordinates
        _, s_un, _, _, _, _, _ = self.raytracer.rays.rays_by_mask(self._ray_selection, normalize=False,
                                                               ret=[0, 1, 0, 0, 0, 0, 0])
        x, y, z = p_[:, :, 0].ravel(), p_[:, :, 1].ravel(), p_[:, :, 2].ravel()
        u, v, w = s_un[:, :, 0].ravel(), s_un[:, :, 1].ravel(), s_un[:, :, 2].ravel()
        s = np.ones_like(z)

        return x, y, z, u, v, w, s

    def _plot_rays(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                   u: np.ndarray, v: np.ndarray, w: np.ndarray, s: np.ndarray)\
            -> None:

        if self._rays_plot is not None:
            self._rays_plot.parent.parent.remove()

        self._rays_plot = self.scene.mlab.quiver3d(x, y, z, u, v, w, scalars=s,
                                                   scale_mode="vector", scale_factor=1, colormap="Greys", mode="2ddash")
        self._rays_plot.glyph.trait_set(color_mode="color_by_scalar")
        self._rays_plot.actor.actor.property.trait_set(lighting=False, render_points_as_spheres=True,
                                                       opacity=self.ray_opacity)
        self._rays_plot.actor.property.trait_set(line_width=self.ray_width,
                                                 point_size=self.ray_width if self.plotting_type == "Points" else 0.1)
        self._rays_plot.parent.parent.name = "Rays"

    def _assign_ray_colors(self) -> None:
        """color the ray representation and the ray source"""

        pol_, w_, wl_, snum_ = self._ray_property_dict["pol"], self._ray_property_dict["w"], \
            self._ray_property_dict["wl"], self._ray_property_dict["snum"]
        N, nt, nc = pol_.shape

        # set plotting properties depending on plotting mode
        match self.coloring_type:

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

            case ('Polarization xz' | 'Polarization yz'):
                if self.raytracer.no_pol:
                    if not self.silent:
                        print("WARNING: Polarization calculation turned off in raytracer, "
                              "reverting to a different mode")
                    self.coloring_type = "Power"
                    return
                if self.coloring_type == "Polarization yz":
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
                cm = "Greys"
                title = "None"

        self._rays_plot.mlab_source.trait_set(scalars=s)

        # lut legend settings
        lutm = self._rays_plot.parent.scalar_lut_manager
        lutm.trait_set(use_default_range=True, show_scalar_bar=True, use_default_name=False,
                       show_legend=self.coloring_type != "Plain", lut_mode=cm, reverse_lut=False)

        # lut visibility and title
        lutm.scalar_bar.trait_set(title=title, unconstrained_font_size=True)
        lutm.label_text_property.trait_set(**self.TEXT_STYLE)
        lutm.title_text_property.trait_set(**self.INFO_STYLE)

        # lut position and size
        hr, vr = tuple(self._scene_size0/self._scene_size)  # horizontal and vertical size ratio
        lutm.scalar_bar_representation.position = np.array([0.92, (1-0.6*vr)/2])
        lutm.scalar_bar_representation.position2 = np.array([0.06*hr, 0.6*vr])
        lutm.scalar_bar_widget.process_events = False  # make non-interactive
        lutm.scalar_bar_representation.border_thickness = 0  # no ugly borders
        lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-#6.3g"
        lutm.number_of_labels = 11

        match self.coloring_type:

            case 'Wavelength':
                lutm.lut.table = color.spectral_colormap(255)
                lutm.data_range = color.WL_BOUNDS 
                lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-6.0f"

            case ('Polarization xz' | "Polarization yz"):
                lutm.data_range = [0.0, 1.0]

            case 'Source':
                lutm.number_of_labels = len(self.raytracer.ray_sources)
                lutm.scalar_bar_widget.scalar_bar_actor.label_format = "%-6.0f"
                if lutm.number_of_labels > 1:  # pragma: no branch
                    lutm.lut.table = color.spectral_colormap(lutm.number_of_labels, 440, 620)

            case 'Plain':
                lutm.reverse_lut = bool(self.high_contrast)

    def _assign_ray_source_colors(self) -> None:
        """sets colors of ray sources"""

        if self._rays_plot is None:
            return

        lutm = self._rays_plot.parent.scalar_lut_manager

        match self.coloring_type:

            case "Plain":
                RSColor = [self.scene.foreground for RSp in self._ray_source_plots]

            case 'Wavelength':
                RSColor = [RS.get_color(rendering_intent="Absolute") for RS in self.raytracer.ray_sources]

            case ('Polarization xz' | "Polarization yz"):
                # color from polarization projection on yz-plane
                RSColor = []
                for RS in self.raytracer.ray_sources:
                    
                    if RS.polarization in ["x", "y", "Angle"]:

                        match RS.polarization:
                            case "x":
                                pol_ang = 0
                            case "y":
                                pol_ang = np.pi/2
                            case "Angle":  # pragma: no branch
                                pol_ang = np.deg2rad(RS.pol_angle)
                    
                        proj = np.sin(pol_ang) if self.coloring_type == "Polarization yz" else np.cos(pol_ang)
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
            if not self.silent:
                print("Number of RaySourcePlots differs from actual Sources. "
                      "Maybe the GUI was not updated properly?")

    def _init_ray_info_text(self) -> None:
        """init detection of ray point clicks and the info display"""
        self._ray_picker = self.scene.mlab.gcf().on_mouse_pick(self._on_ray_pick, button='Left')

        # add ray info text
        self._ray_text_parent = self.scene.engine.add_source(ParametricSurface(name="Ray Info Text"), self.scene)
        self._ray_text = self.scene.mlab.text(0.02, 0.97, "")
        self._on_ray_pick()  # call to set default text

    def _init_status_text(self) -> None:
        """init GUI status text display"""
        self._space_picker = self.scene.mlab.gcf().on_mouse_pick(self._on_space_pick, button='Right')

        # add status text
        self._status_text_parent = self.scene.engine.add_source(ParametricSurface(name="Status Info Text"), self.scene)
        self._status_text = self.scene.mlab.text(0.97, 0.01, "Status Text")
        self._status_text.property.trait_set(**self.INFO_STYLE, justification="right")
        self._status_text.actor.text_scale_mode = 'none'

    def _init_source_list(self) -> None:
        """generate a descriptive list of RaySource names"""
        self.source_names = [f"{RaySource.abbr}{num}: {RS.get_desc()}"[:35]
                             for num, RS in enumerate(self.raytracer.ray_sources)]
        # don't append and delete directly on elements of self.source_names,
        # because issues with trait updates with type List

    def _init_detector_list(self) -> None:
        """generate a descriptive list of Detector names"""
        self.detector_names = [f"{Detector.abbr}{num}: {Det.get_desc()}"[:35]
                               for num, Det in enumerate(self.raytracer.detectors)]
        # don't append and delete directly on elements of self.detector_names,
        # because issues with trait updates with type "List"
    
    def _plot_markers(self) -> None:
        
        self.__remove_objects(self._marker_plots)

        for num, mark in enumerate(self.raytracer.markers):
            m = self.scene.mlab.points3d(*mark.pos, mode="axes", color=self._marker_color)
            m.parent.parent.name = f"Marker {num}"
            m.actor.actor.property.trait_set(lighting=False, line_width=5)
            m.actor.actor.trait_set(pickable=False, force_translucent=True)
            radius  = 0.2 * mark.marker_factor
            m.glyph.glyph.scale_factor = radius
            m.visible = not mark.label_only
        
            text = self.scene.mlab.text(x=mark.pos[0]+radius, y=mark.pos[1], z=mark.pos[2], text=mark.desc, name="Label")
            text.property.trait_set(**self.LABEL_STYLE, justification="center")
            text.actor.text_scale_mode = 'none'
            text.property.font_size = int(8 * mark.text_factor)

            self._marker_plots.append((m, None, None, text, mark))

    def _init_crosshair(self) -> None:
        """init a crosshair for the picker"""
        self._crosshair = self.scene.mlab.points3d([0.], [0.], [0.], mode="axes", color=self._crosshair_color)
        self._crosshair.parent.parent.name = "Crosshair"
        self._crosshair.actor.actor.property.trait_set(lighting=False)
        self._crosshair.actor.actor.pickable = False
        self._crosshair.glyph.glyph.scale_factor = 1.5
        self._crosshair.visible = False

    def default_camera_view(self) -> None:
        """set scene camera view. This should be called after the objects are added,
           otherwise the clipping range is incorrect"""
        self.scene.parallel_projection = True
        self.scene._def_pos = 1  # for some reason it is set to None
        self.scene.y_plus_view()
        self.scene.scene_editor.camera.parallel_scale *= 0.85

    def _on_space_pick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker') -> None:
        """
        3D Space Clicking Handler.
        Shows Click Coordinates or moves Detector to this position when Shift is pressed.
        :param picker_obj:
        """

        pos = picker_obj.pick_position

        # differentiate between Click and Shift+Click
        if self.scene.interactor.shift_key:
            if self._crosshair is not None:
                self._crosshair.visible = False
            if self.raytracer.detectors:
                # set outside position to inside of outline
                pos_z = max(self._det_pos_min, pos[2])
                pos_z = min(self._det_pos_max, pos_z)

                # assign detector position, calls moveDetector()
                self.det_pos = pos_z

                # call with no parameter resets text
                self._on_ray_pick()
        else:
            self._ray_text.text = f"Pick Position: ({pos[0]:>9.6g} mm, {pos[1]:>9.6g} mm, {pos[2]:>9.6g} mm)"
            self._ray_text.property.trait_set(**self.INFO_STYLE, background_opacity=self._info_opacity,
                                              opacity=1, color=self.scene.foreground)
            if self._crosshair is not None:
                self._crosshair.mlab_source.trait_set(x=[pos[0]], y=[pos[1]], z=[pos[2]])
                self._crosshair.visible = True

    def _on_ray_pick(self, picker_obj: 'tvtk.tvtk_classes.point_picker.PointPicker' = None) -> None:
        """
        Ray Picking Handler.
        Shows ray properties on screen.
        :param picker_obj:
        """

        if not self._ray_text:
            return

        # it seems that picker_obj.point_id is only present for the first element in the picked list,
        # so we only can use it when the RayPlot is first in the list
        # see https://github.com/enthought/mayavi/issues/906
        if picker_obj is not None and len(picker_obj.actors) != 0 \
           and picker_obj.actors[0]._vtk_obj is self._rays_plot.actor.actor._vtk_obj:

            # don't pick while plotted rays or rays in raytracer could change
            # using the RayAccessLock wouldn't be wise, because it also used by Focussing
            if self._status["Tracing"] or self._status["Drawing"]:
                if not self.silent:
                    print("Can't pick while tracing or drawing.")
                return

            a = self.raytracer.rays.Nt  # number of points per ray plotted
            b = picker_obj.point_id  # point id of the ray point

            n_, n2_ = np.divmod(b, 2*a)
            n2_ = min(1 + (n2_-1)//2, a - 1)

            N_shown = np.count_nonzero(self._ray_selection)
            RayMask = misc.part_mask(self._ray_selection, np.arange(N_shown) == n_)
            pos = np.nonzero(RayMask)[0][0]

            # get properties of this ray section
            p_, s_, pols_, pw_, wv, snum, n_ = self.raytracer.rays.rays_by_mask(RayMask)
            p_, s_, pols_, pw_, wv, snum, n_ = p_[0], s_[0], pols_[0], pw_[0], wv[0], snum[0], n_[0]  # choose only ray
            p, s, pols, pw = p_[n2_], s_[n2_], pols_[n2_], pw_[n2_]

            pw0 = pw_[n2_-1] if n2_ else None
            s0 = s_[n2_-1] if n2_ else None
            pols0 = pols_[n2_-1] if n2_ else None
            pl = ((pw0-pw)/pw0 if pw0 else 0) if n2_ else None  # power loss

            def to_sph_coords(s):
                # theta, phi
                return np.array([np.rad2deg(np.arccos(s[2])), np.rad2deg(np.arctan2(s[1], s[0]))])

            s_sph = to_sph_coords(s)
            s0_sph = to_sph_coords(s0) if n2_ else None
            pols_sph = to_sph_coords(pols)
            pols0_sph = to_sph_coords(pols0) if n2_ else None

            n = n_[n2_]
            n0 = n_[n2_-1] if n2_ else None

            surfs = self.raytracer.tracing_surfaces

            if 0 < n2_ <= len(surfs):
                surf = surfs[n2_-1]
            elif not n2_:
                surf = self.raytracer.ray_sources[snum].front
            else:
                surf = None

            is_surf = isinstance(surf, Surface)
            if is_surf:
                normal = surf.get_normals(np.array([p[0]]), np.array([p[1]]))[0]
                normal_sph = to_sph_coords(normal)

            l = self.raytracer.rays.ray_lengths(RayMask)[0]
            ol = self.raytracer.rays.optical_lengths(RayMask)[0] 
            l0 = l[n2_ - 1] if n2_ > 0 else None
            l1 = l[n2_] if n2_ < l.shape[0] else None
            ol0 = ol[n2_ - 1] if n2_ > 0 else None
            ol1 = ol[n2_] if n2_ < ol.shape[0] else None

            # differentiate between Click and Shift+Click
            if self.scene.interactor.shift_key:
                text = f"Ray {pos}" +  \
                    f" from Source {snum}" +  \
                    (f" at surface {n2_}\n\n" if n2_ else " at ray source\n\n") + \
                    f"Intersection Position: ({p[0]:>10.6g} mm, {p[1]:>10.6g} mm, {p[2]:>10.6g} mm)\n\n" + \
                    "Vectors:                        Cartesian (x, y, z)                   " + \
                    "Spherical (theta, phi)\n" + \
                    (f"Direction Before:      ({s0[0]:>10.5f}, {s0[1]:>10.5f}, {s0[2]:>10.5f})" if n2_ else "") + \
                    (f"         ({s0_sph[0]:>10.5f}°, {s0_sph[1]:>10.5f}°)\n" if n2_ else "") + \
                    f"Direction After:       ({s[0]:>10.5f}, {s[1]:>10.5f}, {s[2]:>10.5f})" + \
                    f"         ({s_sph[0]:>10.5f}°, {s_sph[1]:>10.5f}°)\n" + \
                    (f"Polarization Before:   ({pols0[0]:>10.5f}, {pols0[1]:>10.5f}, {pols0[2]:>10.5f})" if n2_
                    else "") + \
                    (f"         ({pols0_sph[0]:>10.5f}°, {pols0_sph[1]:>10.5f}°)\n" if n2_
                    else "") + \
                    f"Polarization After:    ({pols[0]:>10.5f}, {pols[1]:>10.5f}, {pols[2]:>10.5f})" + \
                    f"         ({pols_sph[0]:>10.5f}°, {pols_sph[1]:>10.5f}°)\n" + \
                    (f"Surface Normal:        ({normal[0]:>10.5f}, {normal[1]:>10.5f}, {normal[2]:>10.5f})"
                    if pw > 0  and is_surf else "") + \
                    (f"         ({normal_sph[0]:>10.5f}°, {normal_sph[1]:>10.5f}°)\n\n"
                    if pw > 0 and is_surf else "\n") + \
                    f"Wavelength:               {wv:>10.2f} nm" + \
                    (f"\nRefraction Index Before:  {n0:>10.4f}" if n2_ else "") + \
                    (f"           Distance to Last Intersection:          {l0:>10.5g} mm" if l0 else "") + \
                    f"\nRefraction Index After:   {n:>10.4f}" + \
                    (f"           Distance to Next Intersection:          {l1:>10.5g} mm" if l1 else "") + \
                    (f"\nRay Power Before:         {pw0*1e6:>10.5g} µW" if n2_ else "") + \
                    (f"        Optical Distance to Last Intersection:  {ol0:>10.5g} mm" if ol0 else "") + \
                    f"\nRay Power After:          {pw*1e6:>10.5g} µW" + \
                    (f"        Optical Distance to Next Intersection:  {ol1:>10.5g} mm" if ol1 else "") + \
                    (f"\nPower Loss on Surface:    {pl*100:>10.5g} %" if n2_ else "") + \
                    ("\n\nSurface Information:\n" if surf is not None else "") + \
                    (surf.info if surf is not None else "")
            else:
                text = f"Ray {pos}" +  \
                    f" from Source {snum}" +  \
                    (f" at surface {n2_}\n" if n2_ else " at ray source\n") + \
                    f"Intersection Position: ({p[0]:>10.5g} mm, {p[1]:>10.5g} mm, {p[2]:>10.5g} mm)\n" + \
                    f"Direction After:       ({s[0]:>10.5f},    {s[1]:>10.5f},    {s[2]:>10.5f}   )\n" + \
                    f"Polarization After:    ({pols[0]:>10.5f},    {pols[1]:>10.5f},    {pols[2]:>10.5f}   )\n" + \
                    f"Wavelength:             {wv:>10.2f} nm\n" + \
                    f"Ray Power After:        {pw*1e6:>10.5g} µW\n" + \
                    "Pick using Shift+Left Mouse Button for more info"

            self._ray_text.text = text
            self._ray_text.property.trait_set(**self.INFO_STYLE, background_opacity=self._info_opacity,
                                              opacity=1, color=self.scene.foreground)
            if self._crosshair is not None:
                self._crosshair.mlab_source.trait_set(x=[p[0]], y=[p[1]], z=[p[2]])
                self._crosshair.visible = True
        else:
            self._ray_text.text = ""
            if self._crosshair is not None:
                self._crosshair.visible = False

        # text settings
        self._ray_text.property.trait_set(vertical_justification="top")
        self._ray_text.actor.text_scale_mode = 'none'

    def _init_keyboard_shortcuts(self) -> None:
        """Init shift key press detection"""

        # also see already available shortcuts
        # https://docs.enthought.com/mayavi/mayavi/application.html#keyboard-interaction

        def keyrelease(vtk_obj, event):

            match vtk_obj.GetKeyCode():

                # it seems like pressing "m" in the scene does something,
                #  although i can't find any documentation on this
                # a side effect is deactivating the mouse pickers, to reactivate we need to press "m" another time
                case "m":
                    if not self.silent:
                        print("Avoid pressing 'm' in the scene because it interferes with mouse picking handlers.")

                case "y":  # reset view
                    self.default_camera_view()

                case "h":  # hide/show side menu and toolbar
                    self.maximize_scene = bool(self._scene_not_maximized)  # toggle value

                case "v":  # toggle minimalistic_view
                    self.minimalistic_view = not bool(self.minimalistic_view)
                
                case "c":  # high_contrast
                    self.high_contrast = not bool(self.high_contrast)

                case "r":  # cycle plotting types (Point or Rays)
                    ptypes = self.plotting_types
                    self.plotting_type = ptypes[(ptypes.index(self.plotting_type) + 1) % len(ptypes)]

                case "d":  # render DetectorImage
                    self.show_detector_image()

                case "n":  # reselect and replot rays
                    self.scene.disable_render = True
                    self.replot_rays()
                    self.scene.disable_render = False

        self.scene.interactor.add_observer('KeyReleaseEvent', keyrelease)  # calls keyrelease() in main thread

    def _set_gui_loaded(self) -> None:
        """sets GUILoaded Status. Exits GUI if self.exit set."""

        self._status["DisplayingGUI"] = 0

        if self._exit:  # wait 200ms so the user has time to see the scene
            pyface_gui.invoke_after(200, self.close)

    ####################################################################################################################
    # Interface functions
    ####################################################################################################################

    def replot(self, change: dict = None) -> None:
        """
        replot all or part of the geometry and scene.
        If the change dictionary is not provided, everything is replotted according to the raytracer geometry
        :param change: optional change dict from raytracer.compare_property_snapshot
        """

        all_ = change is None

        if not all_ and not change["Any"]:
            return

        self._status["Drawing"] += 1
        pyface_gui.process_events()
        self.scene.disable_render = True

        with self._constant_camera():

            # RaySources need to be plotted first, since plotRays() colors them
            if all_ or change["RaySources"]:
                self._plot_ray_sources()
                self._init_source_list()

                # restore selected RaySource. If it is missing, default to 0.
                if self._source_ind >= len(self.raytracer.ray_sources):
                    self._source_ind = 0
                if len(self.raytracer.ray_sources):
                    self.source_selection = self.source_names[self._source_ind]

            if all_ or change["TraceSettings"]:
                # reassign absorb_missing if it has changed
                if self.raytracer.absorb_missing != bool(self.absorb_missing):
                    with self._no_trait_action():
                        self.absorb_missing = self.raytracer.absorb_missing

            rdh = False  # if GUILoaded Status should be reset in this function
            if self.raytracer.ray_sources:
                if all_ or change["Filters"] or change["Lenses"] or change["Apertures"] or change["Ambient"]\
                        or change["RaySources"] or change["TraceSettings"]:
                    self.retrace()
                elif change["Rays"]:
                    self.replot_rays()
            else:
                rdh = True

            if all_ or change["Filters"]:
                self._plot_filters()

            if all_ or change["Lenses"]:
                self._plot_lenses()

            if all_ or change["Apertures"]:
                self._plot_apertures()

            if all_ or change["Markers"]:
                self._plot_markers()
            
            if all_ or change["Detectors"]:
                # no threading and __detector_lock here, since we're only updating the list
                self._plot_detectors()
                self._init_detector_list()

                # restore selected Detector. If it is missing, default to 0.
                if self._det_ind >= len(self.raytracer.detectors):
                    self._det_ind = 0

                if len(self.raytracer.detectors):
                    self.detector_selection = self.detector_names[self._det_ind]

            if all_ or change["Ambient"]:
                self._plot_index_boxes()
                self._plot_axes()
                self._plot_outline()

                # minimal/maximal z-positions of Detector_obj
                self._det_pos_min = self.raytracer.outline[4]
                self._det_pos_max = self.raytracer.outline[5]

        self.scene.disable_render = False
        self.scene.render()
        pyface_gui.process_events()

        if rdh and self._status["DisplayingGUI"]:  # reset initial loading flag
            # this only gets invoked after raytracing, but with no sources we need to do it here
            pyface_gui.invoke_later(self._set_gui_loaded)

        self._status["Drawing"] -= 1

    def _wait_for_idle(self) -> None:
        """wait until the GUI is Idle. Only call this from another thread"""
        time.sleep(0.3)  # wait for flags to be set, this could also be trait handlers, which could take longer
        while self.busy:
            time.sleep(0.05)

    @property
    def busy(self) -> bool:
        """busy state flag of TraceGUI"""
        # currently I see no way to check with 100% certainty if the TraceGUI is actually idle
        # it could also be in some trait_handlers or functions from other classes
        return self.scene.busy or any(val > 0 for val in self._status.values())

    def debug(self, silent: bool = False, _exit: bool = False, _func: Callable = None, _args: tuple = None) -> None:
        """
        Run in debugging mode
        :param silent:
        :param _exit:
        :param _func:
        :param _args:
        """
        self._exit = _exit
        self.silent = silent
        self.raytracer.silent = silent

        if _func is not None:
            th = Thread(target=_func, args=_args, daemon=True)
            th.start()

        self.configure_traits()

    def run(self, silent: bool = False) -> None:
        """
        Run the TraceGUI
        :param silent:
        """
        self.silent = silent
        self.raytracer.silent = silent
        self.configure_traits()

    @observe('scene:closing')  # needed so this gets also called when clicking x on main window
    def close(self, event=None) -> None:
        """
        close the whole application, including plot windows
        :param event: optional event from traits observe decorator
        """
        pyface_gui.invoke_later(plt.close, "all")  # close plots
        pyface_gui.invoke_later(QtGui.QApplication.closeAllWindows)  # close Qt windows

    ####################################################################################################################
    # Trait handlers.

    @observe('high_contrast')
    def _change_contrast(self, event=None) -> None:
        """
        change high contrast mode
        :param event: optional event from traits observe decorator
        """

        self._set_colors()

        # update filter color
        for F in self._filter_plots:
            for Fi in F[:2]:
                if Fi is not None and F[4] is not None:
                    Fi.actor.property.color = F[4].get_color()[:3] if not self.high_contrast else self.scene.foreground

        # update misc color
        for color, objs in zip([self._lens_color, self._detector_color, self._aperture_color, self._marker_color,
                                self._crosshair_color, self._outline_color],
                               [self._lens_plots, self._detector_plots, self._aperture_plots,
                                self._marker_plots, [[self._crosshair]], self._outline_plots]):
            for obj in objs:
                for el in obj[:3]:
                    if el is not None:
                        el.actor.property.color = color

        # update axes color
        for ax in self._axis_plots:
            for el in ax:
                if el is not None:
                    el.axes.property.color = self._outline_color

        # change index plot objects
        for obj in self._index_box_plots:
            if obj[1] is not None:
                obj[1].property.color = self._axis_color
                obj[1].property.frame_color = self._subtle_color
            if obj[0] is not None:
                obj[0].actor.property.color = self._outline_color

        # reassign ray source colors
        self._assign_ray_source_colors()

        # invert rays in coloring mode white
        if self.coloring_type == "Plain" and self._rays_plot is not None:
            self._rays_plot.parent.scalar_lut_manager.reverse_lut = bool(self.high_contrast)

    @observe('ray_amount_shown')
    def replot_rays(self, event=None) -> None:
        """
        choose a subset of all raytracer rays and plot them with :obj:`TraceGUI._plot_rays`.
        :param event: optional event from traits observe decorator
        """

        # don't do while init or tracing, since it will be done afterwards anyway
        if self.raytracer.ray_sources and self.raytracer.rays.N and not self._no_trait_action_flag\
             and not self._status["Tracing"]: # don't do while tracing, RayStorage rays are still being generated

            self._status["Drawing"] += 1
            pyface_gui.process_events()

            def background():
                with self.__ray_access_lock:
                    N = self.raytracer.rays.N
                    set_size = min(N, self.ray_amount_shown)
                    source_index = np.random.choice(N, size=set_size, replace=False)  # random choice

                    # make bool array with chosen rays set to true
                    self._ray_selection = np.zeros(N, dtype=bool)
                    self._ray_selection[source_index] = True

                    res = self._get_rays()

                def on_finish():

                    with self._constant_camera():
                        self._plot_rays(*res)
                        self._change_ray_colors()
                        self._change_ray_representation()

                    # reset picker text
                    self._on_ray_pick()

                    self._status["Drawing"] -= 1

                    # gets unset on first run
                    if self._status["DisplayingGUI"]:
                        pyface_gui.invoke_later(self._set_gui_loaded)

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('ray_count, absorb_missing')
    def retrace(self, event=None) -> None:
        """
        raytrace in separate thread, after that call :obj:`TraceGUI.replot_rays`.
        :param event: optional event from traits observe decorator
        """

        if self._no_trait_action_flag:
            return

        elif self.raytracer.ray_sources:

            self._status["Tracing"] += 1

            # run this in background thread
            def background() -> None:

                self.raytracer.absorb_missing = bool(self.absorb_missing)

                with self.__ray_access_lock:
                    with self._try() as error:
                        self.raytracer.trace(N=self.ray_count)
                    if error or not self.raytracer.rays.N:  # error or raytracing failed (= no rays)
                        self._status["Tracing"] -= 1
                        # gets unset on first run
                        if self._status["DisplayingGUI"]:
                            pyface_gui.invoke_later(self._set_gui_loaded)
                        return

                # execute this after thread has finished
                def on_finish() -> None:
                    with self._no_trait_action():
                        # reset parameter to value of raytracer.absorb_missing, since it can refuse this value
                        self.absorb_missing = self.raytracer.absorb_missing

                    self._status["Tracing"] -= 1
                    self.replot_rays()

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()
        else:
            pyface_gui.invoke_later(self._set_gui_loaded)

    @observe('ray_opacity')
    def _change_ray_opacity(self, event=None) -> None:
        """
        change opacity of visible rays.
        :param event: optional event from traits observe decorator
        """
        if self._rays_plot is not None:
            self._rays_plot.actor.property.trait_set(opacity=self.ray_opacity)

    @observe("coloring_type")
    def _change_ray_colors(self, event=None) -> None:
        """
        change ray coloring mode.
        :param event: optional event from traits observe decorator
        """
        if self._rays_plot is not None:
            self._assign_ray_colors()
            self._assign_ray_source_colors()

    @observe("plotting_type")
    def _change_ray_representation(self, event=None) -> None:
        """
        change ray view to connected lines or points only.
        :param event: optional event from traits observe decorator
        """
        if self._rays_plot is not None:
            self._rays_plot.actor.property.representation = 'points' if self.plotting_type == 'Points' else 'surface'

    @observe('detector_selection')
    def _change_detector(self, event=None) -> None:
        """
        change detector selection
        :param event: optional event from traits observe decorator
        """

        if not self._no_trait_action_flag and self.raytracer.detectors and self._detector_plots:

            self._status["ChangingDetector"] += 1

            def background():

                def on_finish():

                    self._det_ind = int(self.detector_selection.split(":", 1)[0].split("DET")[1])
                    with self._no_trait_action():
                        self.det_pos = self.raytracer.detectors[self._det_ind].pos[2]

                    self.__detector_lock.release()
                    self.projection_method_enabled = \
                        isinstance(self.raytracer.detectors[self._det_ind].surface, SphericalSurface)
                    self._status["ChangingDetector"] -= 1

                self.__detector_lock.acquire()
                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('det_pos')
    def _move_detector(self, event=None) -> None:
        """
        move chosen Detector.
        :param event: optional event from traits observe decorator
        """

        if not self._no_trait_action_flag and self.raytracer.detectors and self._detector_plots:

            self._status["ChangingDetector"] += 1

            def background():

                def on_finish():

                    # move detector object
                    xp, yp, zp = self.raytracer.detectors[self._det_ind].pos
                    self.raytracer.detectors[self._det_ind].move_to([xp, yp, self.det_pos])

                    # move surface, cylinder and label
                    self._detector_plots[self._det_ind][0].mlab_source.z += event.new - event.old
                    self._detector_plots[self._det_ind][1].mlab_source.z += event.new - event.old
                    self._detector_plots[self._det_ind][3].z_position += event.new - event.old

                    # reinit detectors and Selection to update z_pos in detector name
                    self._init_detector_list()
                    with self._no_trait_action():
                        self.detector_selection = self.detector_names[self._det_ind]

                    self.__detector_lock.release()
                    self._status["ChangingDetector"] -= 1

                self.__detector_lock.acquire()
                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    def show_detector_cut(self) -> None:
        """
        detector image cut.
        :param event: optional event from traits observe decorator
        """
        self.show_detector_image(cut=True)

    @observe('_detector_image_button, _detector_cut_button')
    def show_detector_image(self, event=None, cut=False) -> None:
        """
        render a DetectorImage at the chosen Detector, uses a separate thread.
        :param event: optional event from traits observe decorator
        """

        if self.raytracer.detectors and self.raytracer.rays.N:

            self._status["DetectorImage"] += 1

            def background() -> None:

                with self.__det_image_lock:
                    with self.__detector_lock:
                        with self.__ray_access_lock:
                            # only calculate Image if raytracer Snapshot, selected Source or image_pixels changed
                            # otherwise we can replot the old Image with the new visual settings
                            source_index = None if not self.det_image_one_source else self._source_ind
                            snap = str(self.raytracer.property_snapshot()) + self.detector_selection\
                                + str(source_index) + self.projection_method\
                                + str(len(self.activate_filter)) + str(self.filter_constant)
                            rerender = snap != self._last_det_snap or self.last_det_image is None

                            log, mode, px, dindex, flip, pm = bool(self.log_image), self.image_type,\
                                                              int(self.image_pixels), self._det_ind, \
                                                              bool(self.flip_det_image), self.projection_method
                            limit = None if not self.activate_filter else self.filter_constant

                            cut_args = dict(x=self.cut_value) if self.cut_dimension == "x"\
                                else dict(y=self.cut_value)

                            if rerender:
                                with self._try() as error:
                                    DImg = self.raytracer.detector_image(N=px, detector_index=dindex,
                                                                         source_index=source_index, 
                                                                         projection_method=pm,
                                                                         limit=limit)
                                if error:
                                    self._status["DetectorImage"] -= 1
                                    return

                    if rerender:
                        self.last_det_image = DImg
                        self._last_det_snap = snap
                    else:
                        DImg = self.last_det_image.copy()

                DImg.rescale(px)
                Imc = DImg.get_by_display_mode(mode, log=log)

                def on_finish() -> None:
                    with self._try():
                        if (event is None and not cut) or (event is not None and event.name == "_detector_image_button"):
                            r_image_plot(DImg, log=log, mode=mode, flip=flip, imc=Imc)
                        else:
                            r_image_cut_plot(DImg, log=log, mode=mode, flip=flip, imc=Imc, **cut_args)

                    self._status["DetectorImage"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('_detector_spectrum_button')
    def show_detector_spectrum(self, event=None) -> None:
        """
        render a Detector Spectrum for the chosen Source, uses a separate thread.
        :param event: optional event from traits observe decorator
        """

        if self.raytracer.detectors and self.raytracer.rays.N:

            self._status["DetectorSpectrum"] += 1

            def background() -> None:

                with self.__detector_lock:
                    with self.__ray_access_lock:
                        source_index = None if not self.det_spectrum_one_source else self._source_ind
                        with self._try() as error:
                            Det_Spec = self.raytracer.detector_spectrum(detector_index=self._det_ind,
                                                                        source_index=source_index)
                        if error:
                            self._status["DetectorSpectrum"] -= 1
                            return

                def on_finish() -> None:
                    with self._try():
                        spectrum_plot(Det_Spec)
                    self._status["DetectorSpectrum"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    def show_source_cut(self) -> None:
        """
        source image cut.
        :param event: optional event from traits observe decorator
        """
        self.show_source_image(cut=True)

    @observe('_source_spectrum_button')
    def show_source_spectrum(self, event=None) -> None:
        """
        render a Source Spectrum for the chosen Source, uses a separate thread.
        :param event: optional event from traits observe decorator
        """

        if self.raytracer.ray_sources and self.raytracer.rays.N:

            self._status["SourceSpectrum"] += 1

            def background() -> None:

                with self.__ray_access_lock:
                    with self._try() as error:
                        RS_Spec = self.raytracer.source_spectrum(source_index=self._source_ind)
                    if error:
                        self._status["SourceSpectrum"] -= 1
                        return

                def on_finish() -> None:
                    with self._try():
                        spectrum_plot(RS_Spec)
                    self._status["SourceSpectrum"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('_source_image_button, _source_cut_button')
    def show_source_image(self, event=None, cut=False) -> None:
        """
        render a source image for the chosen Source, uses a separate thread
        :param event: optional event from traits observe decorator
        :param cut:
        """

        if self.raytracer.ray_sources and self.raytracer.rays.N:

            self._status["SourceImage"] += 1

            def background() -> None:

                with self.__source_image_lock:
                    with self.__ray_access_lock:
                        # only calculate Image if raytracer Snapshot, selected Source or image_pixels changed
                        # otherwise we can replot the old Image with the new visual settings
                        snap = str(self.raytracer.property_snapshot()) + self.source_selection\
                               + str(len(self.activate_filter)) + str(self.filter_constant)

                        rerender = snap != self._last_source_snap or self.last_source_image is None
                        log, mode, px, source_index = bool(self.log_image), self.image_type, int(self.image_pixels), \
                            self._source_ind
                        limit = None if not self.activate_filter else self.filter_constant
                        cut_args = dict(x=self.cut_value) if self.cut_dimension == "x"\
                            else dict(y=self.cut_value)

                        if rerender:
                            with self._try() as error:
                                SImg = self.raytracer.source_image(N=px, source_index=source_index, limit=limit)
                            if error:
                                self._status["SourceImage"] -= 1
                                return

                    if rerender:
                        self.last_source_image = SImg
                        self._last_source_snap = snap
                    else:
                        SImg = self.last_source_image.copy()

                SImg.rescale(px)
                Imc = SImg.get_by_display_mode(mode, log=log)

                def on_finish() -> None:
                    with self._try():
                        if (event is None and not cut) or (event is not None and event.name == "_source_image_button"):
                            r_image_plot(SImg, log=log, mode=mode, imc=Imc)
                        else:
                            r_image_cut_plot(SImg, log=log, mode=mode, imc=Imc, **cut_args)

                    self._status["SourceImage"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('_auto_focus_button')
    def move_to_focus(self, event=None) -> None:
        """
        Find a Focus.
        The chosen Detector defines the search range for focus finding.
        Searches are always between lenses or the next outline.
        Search takes place in a separate thread, after that the Detector is moved to the focus
        :param event: optional event from traits observe decorator
        """

        if self.raytracer.detectors and self.raytracer.ray_sources and self.raytracer.rays.N:

            self._status["Focussing"] += 1
            self._autofocus_information = ""

            def background() -> None:
                source_index = None if not self.af_one_source else self._source_ind
                mode, det_pos, ret_cost, det_ind = self.focus_type, self.det_pos, bool(self.focus_cost_plot), \
                    self._det_ind

                with self.__ray_access_lock:
                    with self._try() as error:
                        res, afdict = self.raytracer.autofocus(mode, det_pos, return_cost=ret_cost,
                                                                        source_index=source_index)
                    if error:
                        self._status["Focussing"] -= 1
                        return

                with self.__detector_lock:
                    if det_ind < len(self.raytracer.detectors):  # pragma: no branch
                        self.raytracer.detectors[det_ind].move_to([*self.raytracer.detectors[det_ind].pos[:2],
                                                                   res.x])

                # execute this function after thread has finished
                def on_finish() -> None:

                    bounds, pos, N = afdict["bounds"], afdict["pos"], afdict["N"]
                    
                    if det_ind < len(self.raytracer.detectors):  # pragma: no branch
                        self.det_pos = self.raytracer.detectors[det_ind].pos[2]

                    if self.focus_cost_plot:
                        with self._try():
                            autofocus_cost_plot(res, afdict, f"{mode} Cost Function\nMinimum at z={res.x:.5g}mm")

                    self._autofocus_information = \
                        f"Found 3D position: [{pos[0]:.7g}mm, {pos[1]:.7g}mm, {pos[2]:.7g}mm]\n"\
                        f"Search Region: z = [{bounds[0]:.7g}mm, {bounds[1]:.7g}mm]\n"\
                        f"Used {N} Rays for Autofocus\n"\
                        f"Ignoring Filters and Apertures\n\nOptimizeResult:\n{res}"

                    self._status["Focussing"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('scene:activated')
    def _plot_scene(self, event=None) -> None:
        """
        Initialize the GUI. Inits a variety of things.
        :param event: optional event from traits observe decorator
        """
        self._init_ray_info_text()
        self._init_status_text()
        self._init_crosshair()
        self._init_keyboard_shortcuts()

        self._plot_orientation_axes()
        self.replot()
        self.default_camera_view()  # this needs to be called after replot, which defines the visual scope
        self._status["InitScene"] = 0
        
    @observe('_status:items')
    def _change_status(self, event=None) -> None:
        """
        Update the status info text.
        :param event: optional event from traits observe decorator
        """
        msgs = {"RunningCommand": "Running Command",
                "Tracing": "Raytracing",
                "Focussing": "Focussing",
                "ChangingDetector": "Changing Detector",
                "DetectorImage": "Generating Detector Image",
                "SourceImage": "Generating Source Image",
                "SourceSpectrum": "Generating Source Spectrum",
                "DetectorSpectrum": "Generating Detector Spectrum",
                "Drawing": "Drawing"}
       
        # print messages to scene
        if not self._status["InitScene"] and self._status_text is not None:
            self._status_text.text = ""
            for key, val in msgs.items():
                if self._status[key]:
                    self._status_text.text += msgs[key] + "...\n"

    @observe('minimalistic_view')
    def _change_minimalistic_view(self, event=None) -> None:
        """
        change the minimalistic ui view option.
        :param event: optional event from traits observe decorator
        """
        if not self._status["InitScene"]:

            self._status["Drawing"] += 1
            show = not bool(self.minimalistic_view)

            if self._orientation_axes is not None:
                self._orientation_axes.visible = show

            for rio in self._index_box_plots:
                if rio[1] is not None:
                    rio[1].text = rio[1].text.replace("ambient\n", "") if not show else ("ambient\n" + rio[1].text)

            # remove descriptions from labels in minimalistic_view
            for Objects in [self._ray_source_plots, self._lens_plots,
                            self._filter_plots, self._aperture_plots, self._detector_plots]:
                for num, obj in enumerate(Objects):
                    if obj[3] is not None and obj[4] is not None:
                        label = f"{obj[4].abbr}{num}"
                        label = label if obj[4].desc == "" or not show else label + ": " + obj[4].desc
                        obj[3].text = label

            for ax in self._axis_plots:
                if ax[0] is not None:
                    ax[0].visible = show
            
            self._status["Drawing"] -= 1

    @observe("maximize_scene")
    def _change_maximize_scene(self, event=None) -> None:
        """
        change "maximize scene, hide side menu and toolbar"
        :param event: optional event from traits observe decorator
        """
        self._scene_not_maximized = not bool(self.maximize_scene)
        self.scene.scene_editor._tool_bar.setVisible(self._scene_not_maximized)

    @observe('ray_width')
    def _change_ray_width(self, event=None) -> None:
        """
        set the ray width for the visible rays.
        :param event: optional event from traits observe decorator
        """
        if self._rays_plot is not None:
            self._rays_plot.actor.property.trait_set(line_width=self.ray_width, point_size=self.ray_width)

    @observe('source_selection')
    def _change_selected_ray_source(self, event=None) -> None:
        """
        Updates the Detector Selection and the corresponding properties.
        :param event: optional event from traits observe decorator
        """
        if self.raytracer.ray_sources:
            self._source_ind = int(self.source_selection.split(":", 1)[0].split("RS")[1])

    @observe('raytracer_single_thread')
    def _change_raytracer_threading(self, event=None) -> None:
        """
        change the raytracer multithreading option. Useful for debugging.
        :param event: optional event from traits observe decorator
        """
        self.raytracer.threading = not bool(self.raytracer_single_thread)

    @observe('garbage_collector_stats')
    def _change_garbage_collector_stats(self, event=None) -> None:
        """
        show garbage collector stats
        :param event: optional event from traits observe decorator
        """
        gc.set_debug(gc.DEBUG_STATS if self.garbage_collector_stats else 0)

    @observe('show_all_warnings')
    def _change_warnings(self, event=None) -> None:
        """
        change warning visibility.
        :param event: optional event from traits observe decorator
        """
        if bool(self.show_all_warnings):
            warnings.simplefilter("always")
        else:
            warnings.resetwarnings()

    @observe('_property_browser_button')
    def open_property_browser(self, event=None) -> None:
        """
        Open a property browser for the gui, the scene, the raytracer and the shown rays.
        :param event: optional event from traits observe decorator
        """
        gdb = PropertyBrowser(self, self.scene, self.raytracer, self._ray_property_dict)
        gdb.edit_traits()

    @observe('wireframe_surfaces')
    def _change_surface_mode(self, event=None) -> None:
        """
        change surface representation to normal or wireframe view.
        :param event: optional event from traits observe decorator
        """
        repr_ = "wireframe" if bool(self.wireframe_surfaces) else "surface"

        for obj in [*self._ray_source_plots, *self._lens_plots,
                    *self._filter_plots, *self._aperture_plots, *self._detector_plots]:
            for obji in obj[:3]:
                if obji is not None:
                    obji.actor.property.representation = repr_

    @observe('_clear_button')
    def clear_history(self, event=None) -> None:
        """
        clear command history
        :param event: optional event from traits observe decorator
        """
        self._command_history = ""
    
    @observe('_run_button')
    def send_cmd(self, event=None) -> None:
        """
        Updates the Detector Selection and the corresponding properties.
        :param event: optional event from traits observe decorator
        """
        dict_ = dict(  # GUI and scene
                     mlab=self.scene.mlab, engine=self.scene.engine, scene=self.scene,
                     camera=self.scene.camera, GUI=self,

                     # abbreviations for raytracer and object lists
                     RT=self.raytracer, LL=self.raytracer.lenses, FL=self.raytracer.filters,
                     APL=self.raytracer.apertures, RSL=self.raytracer.ray_sources,
                     DL=self.raytracer.detectors, ML=self.raytracer.markers)

        def background():

            cmd = self._cmd

            # wait until all other actions have finished
            if not self.command_dont_wait:
                while sum(self._status.values()) != self._status["RunningCommand"]:
                    time.sleep(0.05)

            def on_finish():

                self._command_history += ("\n" if self._command_history else "") + cmd

                with self._try():
                    hs = self.raytracer.property_snapshot()
                    exec(cmd, locals() | dict_, globals())

                    if not self.command_dont_replot:
                        hs2 = self.raytracer.property_snapshot()
                        cmp = self.raytracer.compare_property_snapshot(hs, hs2)
                        self.replot(cmp)

                self._status["RunningCommand"] -= 1

            pyface_gui.invoke_later(on_finish)

        if self._cmd != "":
            self._status["RunningCommand"] += 1
            action = Thread(target=background, daemon=True)
            action.start()

    # rescale axes texts when the window was resized
    # for some reasons these are the only ones having no text_scaling_mode = 'none' option
    # this is the only trait so far that does not work with @observe, like:
    # @observe("scene:scene_editor:interactor:size:items:value")
    @observe("scene:scene_editor:busy")
    def _resize_scene_elements(self, event) -> None:
        """
        Handles GUI window size changes. Fixes incorrect scaling by mayavi.
        :param event: event from traits observe decorator
        """

        if self.scene.scene_editor.busy and self.scene.scene_editor.interactor is not None:
            scene_size = self.scene.scene_editor.interactor.size

            if self._scene_size[0] != scene_size[0] or self._scene_size[1] != scene_size[1]:
                # average  of x and y size for former and current scene size
                ch1 = (self._scene_size[0] + self._scene_size[1]) / 2
                ch2 = (scene_size[0] + scene_size[1]) / 2

                # update font factor so font size stays the same
                for ax in self._axis_plots:
                    ax[0].axes.font_factor *= ch1/ch2

                # rescale orientation axes
                if self._orientation_axes is not None:
                    self._orientation_axes.widgets[0].zoom *= ch1 / ch2

                if self._rays_plot is not None:
                    bar = self._rays_plot.parent.scalar_lut_manager.scalar_bar_representation
                    bar.position2 = bar.position2 * self._scene_size / scene_size
                    bar.position = [bar.position[0], (1-bar.position2[1])/2]

                # set current window size
                self._scene_size = scene_size
