
import time  # provides sleeping
import warnings  # show warnings
import traceback  # traceback printing
import gc  # garbage collector stats
from threading import Thread, Lock  # threading
from typing import Callable, Any  # typing types
from contextlib import contextmanager  # context manager for _no_trait_action()

# enforce qt backend
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt'

from pyface.api import GUI as pyface_gui  # invoke_later() method
from pyface.qt import QtGui  # closing UI elements

# traits types and UI elements
from traitsui.api import Group as TGroup
from traitsui.api import View, Item, HSplit, CheckListEditor, TextEditor, RangeEditor
from traits.api import HasTraits, Range, Instance, observe, Str, Button, Enum, List, Dict, Float, Bool

from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

import matplotlib.pyplot as plt  # closing plot windows

# provides types and plotting functionality, most of these are also imported for the TCP protocol scope
from ..tracer import *
from ..plots import r_image_plot, r_image_cut_plot, autofocus_cost_plot, spectrum_plot  # different plots

from .property_browser import PropertyBrowser  # dictionary browser
from .command_window import CommandWindow
from ._scene_plotting import ScenePlotting

from ..__metadata__ import __version__



class TraceGUI(HasTraits):

    scene: Instance = Instance(MlabSceneModel, args=())
    """the mayavi scene"""

    # Ranges

    ray_count: Range = Range(1, Raytracer.MAX_RAYS, 200000, desc='Number of rays for Simulation', enter_set=True,
                             auto_set=False, label="Rays", mode='text')
    """Number of rays for raytracing"""
    
    filter_constant: Range = Range(0.3, 40., 1., desc='filter constant', enter_set=True,
                             auto_set=False, label="Limit (µm)", mode='text')
    """gaussian filter constant standard deviation"""

    ray_amount_shown: Range = Range(1, 10000, 2000, desc='the number of rays which is drawn', enter_set=False,
                                    auto_set=False, label="Count", mode='logslider')
    """Number of rays shown in mayavi scene"""

    det_pos: Range = Range(low='_det_pos_min', high='_det_pos_max', value='_det_pos_max', mode='text',
                           desc='z-Position of the Detector', enter_set=True, auto_set=True, label="z_det")
    """z-Position of Detector. Lies inside z-range of :obj:`Backend.raytracer.outline`"""

    ray_opacity: Range = Range(1e-5, 1, 0.01, desc='Opacity of the rays/Points', enter_set=True,
                               auto_set=True, label="Opacity", editor=RangeEditor(format_str="%.4g", low=1e-5, 
                                                                                  high=1, mode="logslider"))
    """opacity of shown ray"""

    ray_width: Range = Range(1.0, 20.0, 1, desc='Ray Linewidth or Point Size', enter_set=True,
                             auto_set=True, label="Width", editor=RangeEditor(format_str="%.4g", low=1.0, 
                                                                              high=20.0, mode="logslider"))
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

    vertical_labels: List = List(editor=CheckListEditor(values=['Vertical Labels'], format_func=lambda x: x),
                                   desc="if element labels are displayed vertically in scene")
    """in geometries with tight geometry and long descriptions one could want to display the labels vertically"""

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
        List = List(editor=CheckListEditor(values=['No Raytracer Multithreading'], format_func=lambda x: x),
                    desc="if raytracer backend uses only one thread")
    """limit raytracer backend operation to one thread (excludes the TraceGUI)"""
    
    command_dont_skip: \
        List = List(editor=CheckListEditor(values=["Don't Wait for Idle State"], format_func=lambda x: x),
                    desc="don't skip when other actions are running. Can lead to race conditions.")
    """Run command even if background tasks are active..
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

    coloring_types: list = ['Plain', 'Power', 'Wavelength', 'Source', 'Polarization xz',\
                            'Polarization yz', 'Refractive Index']  
    """available ray coloring modes"""
    coloring_type: Enum = Enum(*coloring_types, desc="Ray Property to color the Rays With")
    """Ray Coloring Mode"""

    image_type: Enum = Enum(*RImage.display_modes, desc="Image Type Presented")
    """Image Type"""
   
    projection_method_enabled: Bool = False  #: if projection method is selectable, only the case for spherical detectors
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

    image_pixels: Enum = Enum(RImage.SIZES, label="Pixels_xy",
                              desc='Detector Image Pixels in Smaller of x or y Dimension')
    """Image Pixel value for Source/Detector Image. This the number of pixels for the smaller image side."""


    # Buttons

    _detector_image_button:      Button = Button(label="Detector Image", desc="Generating a Detector Image")
    _property_browser_button:    Button = Button(label="Open Property Browser", desc="property browser is opened")
    _command_window_button:      Button = Button(label="Open Command Window", desc="command window is opened")
    _source_spectrum_button:     Button = Button(label="Source Spectrum", desc="sources spectrum is shown")
    _detector_spectrum_button:   Button = Button(label="Detector Spectrum", desc="detector spectrum is shown")
    _source_cut_button:          Button = Button(label="Source Image Cut", desc="source image cut is shown")
    _detector_cut_button:        Button = Button(label="Detector Image Cut", desc="detector image cut is shown")
    _source_image_button:        Button = Button(label="Source Image", desc="Source Image of the Chosen Source")
    _auto_focus_button:          Button = Button(label="Find Focus", desc="Finding the Focus Between the Lenses"
                                                                         " around the Detector Position")

    # Strings and Labels

    _autofocus_information:      Str = Str()
    _spectrum_information:       Str = Str()
    _debug_options_label:        Str = Str('Debugging Options:')
    _debug_command_label:        Str = Str('Command Running Options:')
    _spectrum_label:             Str = Str('Generate Spectrum:')
    _image_label:                Str = Str('Render Image:')
    _geometry_label:             Str = Str('Geometry:')
    _autofocus_label:            Str = Str('Autofocus:')
    _property_label:             Str = Str('Additional Windows:')
    _trace_label:                Str = Str('Trace Settings:')
    _plotting_label:             Str = Str('Ray Plotting Modes:')
    _ray_visual_label:           Str = Str('Ray Visual Settings:')
    _ui_visual_label:            Str = Str('Scene and UI Settings:')
    _autofocus_output_label:     Str = Str('Optimization Output:')
    _spectrum_output_label:      Str = Str('Spectrum Properties:')
    _filter_label:               Str = Str('Resolution Filter (Gaussian):')
    _whitespace_label:           Str = Str('')

    _separator: Item = Item("_whitespace_label", style='readonly', show_label=False, width=210)

    _scene_not_maximized: Bool = Bool(True)

    _status: Dict = Dict(Str())

    # size of the mlab scene
    _scene_size0 = [1100, 800]  # default size

    _bool_list_elements = []

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
                            Item('vertical_labels', style="custom", show_label=False),
                            _separator,
                            _separator,
                            Item("_property_label", style='readonly', show_label=False, emphasized=True),
                            Item('_property_browser_button', show_label=False),
                            Item('_command_window_button', show_label=False),
                            _separator,
                            label="Main"
                            ),
                        TGroup(
                            _separator,
                            Item("_geometry_label", style='readonly', show_label=False, emphasized=True),
                            Item('source_selection', label="Source"),
                            Item('detector_selection', label="Detector"),
                            Item('det_pos'),
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
                            Item('det_pos'),
                            _separator,
                            _separator,
                            Item("_spectrum_label", style='readonly', show_label=False, emphasized=True),
                            Item('_source_spectrum_button', show_label=False),
                            Item('det_spectrum_one_source', style="custom", show_label=False),
                            Item('_detector_spectrum_button', show_label=False),
                            _separator,
                            _separator,
                            Item("_spectrum_output_label", style='readonly', show_label=False, emphasized=True),
                            Item("_spectrum_information", show_label=False, style="custom"),
                            label="Spectrum",
                            ),
                        TGroup(
                            _separator,
                            Item("_geometry_label", style='readonly', show_label=False, emphasized=True),
                            Item('source_selection', label="Source"),
                            Item('detector_selection', label="Detector"),
                            Item('det_pos'),
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
                        HSplit(  # hsplit with whitespace group so we have a left margin in this tab
                            TGroup(Item("_whitespace_label", style='readonly', show_label=False, width=50)),
                            TGroup(
                                _separator,
                                _separator,
                                _separator,
                                Item("_debug_options_label", style='readonly', show_label=False, emphasized=True),
                                Item('raytracer_single_thread', style="custom", show_label=False),
                                Item('show_all_warnings', style="custom", show_label=False),
                                Item('garbage_collector_stats', style="custom", show_label=False),
                                Item('wireframe_surfaces', style="custom", show_label=False),
                                _separator,
                                _separator,
                                _separator,
                                Item("_debug_command_label", style='readonly', show_label=False, emphasized=True),
                                Item('command_dont_replot', style="custom", show_label=False),
                                Item('command_dont_skip', style="custom", show_label=False),
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

    def __init__(self, raytracer: Raytracer, ui_style=None, **kwargs) -> None:
        """
        Create a new TraceGUI with the assigned Raytracer.

        :param RT: raytracer
        :param ui_style: UI style string for Qt
        :param kwargs: additional arguments, options, traits and parameters
        """
        # set UI theme. Unfortunately this does not work dynamically (yet)
        if ui_style is not None:
            QtGui.QApplication.setStyle(ui_style)
      
        self._plot = ScenePlotting(self, raytracer)

        # lock object for multithreading
        self.__detector_lock = Lock()  # when detector is changed, moved or used for rendering
        self.__det_image_lock = Lock()  # when last detector image changed
        self.__source_image_lock = Lock()  # when last source image changed
        self.__ray_access_lock = Lock()  # when rays read or written

        self.raytracer: Raytracer = raytracer
        """reference to the raytracer"""

        self._cdb = None

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

        # set the properties after this without leading to a re-draw
        self._no_trait_action_flag = False
       
        # these properties should not be set, since they have to be initialized first
        # and setting as inits them would lead to issues
        forbidden = ["source_selection", "detector_selection", "det_pos"]
        if any(key in kwargs for key in forbidden):
            raise RuntimeError(f"Assigning an initial value for properties '{forbidden}' is not supported")

        # one element list traits that are treated as bool values
        self._bool_list_elements = [el for el in self.trait_get() if (self.trait(el).default_kind == "list")\
                                    and self._trait(el, 0).editor is not None and\
                                    (len(self._trait(el, 0).editor.values) == 1)]

        with self._no_trait_action():

            # add true default parameters
            if "absorb_missing" not in kwargs:
                kwargs["absorb_missing"] = True

            # convert bool values to list entries for List()
            for key, val in kwargs.items():
                if key in self._bool_list_elements and isinstance(val, bool):
                    kwargs[key] = [self._trait(key, 0).editor.values[0]] if val else []

            # define Status dict
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
        if isinstance(val, bool) and key in self._bool_list_elements:
            val = [self._trait(key, 0).editor.values[0]] if val else []

        # the detector can only be changed, if it isn't locked (__detector_lock)
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
    
    def _do_in_main(self, f: Callable, *args, **kw) -> None:
        """execute a function in the GUI main thread"""
        pyface_gui.invoke_later(f, *args, **kw)
        pyface_gui.process_events()

    def _set_in_main(self, trait: str, val: Any) -> None:
        """assign a property in the GUI main thread"""
        pyface_gui.set_trait_later(self, trait, val)
        pyface_gui.process_events()

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
    
    def _set_gui_loaded(self) -> None:
        """sets GUILoaded Status. Exits GUI if self.exit set."""

        self._status["DisplayingGUI"] = 0

        if self._exit:  # wait 400ms so the user has time to see the scene
            pyface_gui.invoke_after(400, self.close)

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

        with self._plot.constant_camera():

            # RaySources need to be plotted first, since plotRays() colors them
            if all_ or change["RaySources"]:
                self._plot.plot_ray_sources()
                self._init_source_list()

                # restore selected RaySource. If it is missing, default to 0.
                if self._source_ind >= len(self.raytracer.ray_sources):
                    self._source_ind = 0
                if len(self.raytracer.ray_sources):
                    self.source_selection = self.source_names[self._source_ind]

            # if GUILoaded Status should be reset in this function
            rdh = not bool(self.raytracer.ray_sources)
            
            if (all_ or change["Filters"] or change["Lenses"] or change["Apertures"] or change["Ambient"]\
                    or change["RaySources"] or change["TraceSettings"]) and not rdh:
                self.retrace()
            elif all_ or change["Rays"]:
                self.replot_rays()

            if all_ or change["Filters"]:
                self._plot.plot_filters()

            if all_ or change["Lenses"]:
                self._plot.plot_lenses()

            if all_ or change["Apertures"]:
                self._plot.plot_apertures()
            
            if all_ or change["Volumes"]:
                self._plot.plot_volumes()

            if all_ or change["Markers"]:
                self._plot.plot_point_markers()
                self._plot.plot_line_markers()
            
            if all_ or change["Detectors"]:
                # no threading and __detector_lock here, since we're only updating the list
                self._plot.plot_detectors()
                self._init_detector_list()

                # restore selected Detector. If it is missing, default to 0.
                if self._det_ind >= len(self.raytracer.detectors):
                    self._det_ind = 0

                if len(self.raytracer.detectors):
                    self.detector_selection = self.detector_names[self._det_ind]

            if all_ or change["Ambient"]:
                self._plot.plot_index_boxes()
                self._plot.plot_axes()
                self._plot.plot_outline()

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

    def _wait_for_idle(self, timeout=30) -> None:
        """wait until the GUI is Idle. Only call this from another thread"""

        def raise_timeout(keys):
            raise TimeoutError(f"Timeout while waiting for other actions to finish. Blocking actions: {keys}")

        tsum = 1
        time.sleep(1)  # wait for flags to be set, this could also be trait handlers, which could take longer
        while self.busy:
            time.sleep(0.05)
            tsum += 0.05

            if tsum > timeout:
                keys = [key for key, val in self._status.items() if val]
                pyface_gui.invoke_later(raise_timeout, keys)
                return

    @property
    def busy(self) -> bool:
        """busy state flag of TraceGUI"""
        # currently I see no way to check with 100% certainty if the TraceGUI is actually idle
        # it could also be in some trait_handlers or functions from other classes
        return self.scene.busy or any(val > 0 for val in self._status.values())

    def debug(self, silent: bool = False, _exit: bool = False, _func: Callable = None, _args: tuple = None) -> None:
        """
        Run in debugging mode

        :param silent: if all standard output should be muted
        :param _exit: if exit directly after initializing and plotting the GUI
        :param _func: thread function to execute while the GUI is active
        :param _args: arguments for _func
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

        :param silent: if all standard output should be muted
        """
        self.silent = silent
        self.raytracer.silent = silent
        self.configure_traits()

    @observe('scene:closing', dispatch="ui")  # needed so this gets also called when clicking x on main window
    def close(self, event=None) -> None:
        """
        close the whole application, including plot windows

        :param event: optional event from traits observe decorator
        """
        pyface_gui.invoke_later(plt.close, "all")  # close plots
        pyface_gui.invoke_later(QtGui.QApplication.closeAllWindows)  # close Qt windows

    ####################################################################################################################
    # Trait handlers.

    @observe('high_contrast', dispatch="ui")
    def _change_contrast(self, event=None) -> None:
        """
        change the high contrast mode

        :param event: optional event from traits observe decorator
        """
        self._status["Drawing"] += 1
        self._plot.change_contrast()
        self._status["Drawing"] -= 1

    @observe('ray_amount_shown', dispatch="ui")
    def replot_rays(self, event=None) -> None:
        """
        choose a subset of all raytracer rays and plot them with :obj:`TraceGUI._plot_rays`.

        :param event: optional event from traits observe decorator
        """

        if not self._no_trait_action_flag and not self._status["Tracing"]: 
            # don't do while tracing, RayStorage rays are still being generated

            self._status["Drawing"] += 1
            pyface_gui.process_events()

            if not self.raytracer.ray_sources or not self.raytracer.rays.N:
                self._plot.remove_rays()
                self._status["Drawing"] -= 1
                return

            def background():
                with self.__ray_access_lock:
                    res = self._plot.get_rays()

                def on_finish():

                    self._plot.assign_ray_props()

                    with self._plot.constant_camera():
                        self._plot.plot_rays(*res)
                        self._change_ray_and_source_colors()

                    # reset picker text
                    self._plot.reset_ray_text()

                    self._status["Drawing"] -= 1

                    # gets unset on first run
                    if self._status["DisplayingGUI"]:
                        pyface_gui.invoke_later(self._set_gui_loaded)

                pyface_gui.invoke_later(on_finish)
             
            action = Thread(target=background, daemon=True)
            action.start()

    @observe('ray_count, absorb_missing', dispatch="ui")
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

                error_state = False
                self.raytracer.absorb_missing = bool(self.absorb_missing)

                with self.__ray_access_lock:
                    with self._try() as error:
                        self.raytracer.trace(N=self.ray_count)
                        
                    # error or raytracing failed (= no rays)
                    if error or self.raytracer.geometry_error or not self.raytracer.rays.N:  
                        error_state = True

                # execute this after thread has finished
                def on_finish() -> None:

                    if error_state:
                        self._plot.remove_rays()
                           
                        # plot fault positions if they are provided
                        if self.raytracer.geometry_error and self.raytracer.fault_pos.shape[0]:
                            self._plot.set_fault_markers()

                        self._status["Tracing"] -= 1

                        # gets unset on first run
                        if self._status["DisplayingGUI"]:
                            pyface_gui.invoke_later(self._set_gui_loaded)

                    else:
                        self._plot.remove_fault_markers()

                        self._status["Tracing"] -= 1
                        self.replot_rays()

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()
        else:
            pyface_gui.invoke_later(self._set_gui_loaded)

    @observe('ray_opacity', dispatch="ui")
    def _change_ray_opacity(self, event=None) -> None:
        """
        change opacity of visible rays.

        :param event: optional event from traits observe decorator
        """
        self._plot.set_ray_opacity()

    @observe("coloring_type", dispatch="ui")
    def _change_ray_and_source_colors(self, event=None) -> None:
        """
        change ray coloring mode.

        :param event: optional event from traits observe decorator
        """
        self._plot.assign_ray_colors()
        self._plot.assign_ray_source_colors()

    @observe("plotting_type", dispatch="ui")
    def _change_ray_representation(self, event=None) -> None:
        """
        change ray view to connected lines or points only.

        :param event: optional event from traits observe decorator
        """
        self._plot.set_ray_repr()

    @observe('detector_selection', dispatch="ui")
    def _change_detector(self, event=None) -> None:
        """
        change detector selection

        :param event: optional event from traits observe decorator
        """

        if not self._no_trait_action_flag and self.raytracer.detectors:

            self._status["ChangingDetector"] += 1

            def background():

                def on_finish():

                    with self._no_trait_action():
                        self._det_ind = int(self.detector_selection.split(":", 1)[0].split("DET")[1])
                        self.det_pos = self.raytracer.detectors[self._det_ind].pos[2]

                    self.__detector_lock.release()
                    self.projection_method_enabled = \
                        isinstance(self.raytracer.detectors[self._det_ind].surface, SphericalSurface)
                    self._status["ChangingDetector"] -= 1

                self.__detector_lock.acquire()
                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('det_pos', dispatch="ui")
    def _move_detector(self, event=None) -> None:
        """
        move chosen detector.

        :param event: optional event from traits observe decorator
        """

        if not self._no_trait_action_flag and self.raytracer.detectors:

            self._status["ChangingDetector"] += 1

            def background():

                def on_finish():

                    # move detector object
                    xp, yp, zp = self.raytracer.detectors[self._det_ind].pos
                    self.raytracer.detectors[self._det_ind].move_to([xp, yp, self.det_pos])
                    self._plot.move_detector_diff(self._det_ind, event.new - event.old)

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
        Plot a detector image cut.

        :param event: optional event from traits observe decorator
        """
        self.show_detector_image(cut=True)

    @observe('_detector_image_button, _detector_cut_button', dispatch="ui")
    def show_detector_image(self, event=None, cut=False) -> None:
        """
        render a detector image at the chosen Detector, uses a separate thread.

        :param event: optional event from traits observe decorator
        :param cut: if a RImage cut image is plotted
        """

        if self.raytracer.detectors and self.raytracer.rays.N:

            self._status["DetectorImage"] += 1

            def background() -> None:
            
                error = False

                with self.__det_image_lock:
                    with self.__detector_lock:
                        with self.__ray_access_lock:
                            # only calculate Image if raytracer Snapshot, selected Source or image_pixels changed
                            # otherwise we can replot the old Image with the new visual settings
                            source_index = None if not self.det_image_one_source else self._source_ind
                            snap = [self.raytracer.property_snapshot(), self.detector_selection, source_index, 
                                    self.projection_method, self.activate_filter, self.filter_constant]
                            rerender = snap != self._last_det_snap or self.last_det_image is None

                            log, mode, px, dindex, flip, pm = bool(self.log_image), self.image_type,\
                                int(self.image_pixels), self._det_ind, bool(self.flip_det_image), self.projection_method
                            limit = None if not self.activate_filter else self.filter_constant
                            cut_args = {self.cut_dimension : self.cut_value}

                            if rerender:
                                with self._try() as error:
                                    DImg = self.raytracer.detector_image(N=px, detector_index=dindex,
                                                                         source_index=source_index, 
                                                                         projection_method=pm, limit=limit)
                    if not error:

                        if rerender:
                            self.last_det_image = DImg
                            self._last_det_snap = snap
                        else:
                            DImg = self.last_det_image.copy()
                            DImg.rescale(px)

                        Imc = DImg.get(mode, log=log)

                def on_finish() -> None:
                    if not error:
                        with self._try():
                            if (event is None and not cut) or (event is not None\
                                    and event.name == "_detector_image_button"):
                                r_image_plot(DImg, log=log, mode=mode, flip=flip, imc=Imc)
                            else:
                                r_image_cut_plot(DImg, log=log, mode=mode, flip=flip, imc=Imc, **cut_args)

                    self._status["DetectorImage"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('_detector_spectrum_button', dispatch="ui")
    def show_detector_spectrum(self, event=None) -> None:
        """
        render a Detector Spectrum for the chosen Source, uses a separate thread.
        :param event: optional event from traits observe decorator
        """

        if self.raytracer.detectors and self.raytracer.rays.N:

            self._status["DetectorSpectrum"] += 1
            self._spectrum_information = ""

            def background() -> None:

                error = False

                with self.__detector_lock:
                    with self.__ray_access_lock:
                        source_index = None if not self.det_spectrum_one_source else self._source_ind
                        with self._try() as error:
                            Det_Spec = self.raytracer.detector_spectrum(detector_index=self._det_ind,
                                                                        source_index=source_index)

                def on_finish() -> None:
                    if not error:
                        with self._try():
                            spectrum_plot(Det_Spec)
                        self._spectrum_information = self._get_spectrum_information(Det_Spec)

                    self._status["DetectorSpectrum"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    def show_source_cut(self) -> None:
        """show a source image cut plot"""
        self.show_source_image(cut=True)
    
    def _get_spectrum_information(self, spec) -> str:
        return\
            f"{spec.get_long_desc()}\n\n"\
            f"Power: {spec.power():.6g} W\n"\
            f"Luminous Power: {spec.luminous_power():.6g} lm\n\n"\
            f"Peak: {spec.peak():.6g} {spec.unit}\n"\
            f"Peak Wavelength: {spec.peak_wavelength():.3f} nm\n"\
            f"Centroid Wavelength: {spec.centroid_wavelength():.3f} nm\n"\
            f"FWHM: {spec.fwhm():.3f} nm\n\n"\
            f"Dominant Wavelength: {spec.dominant_wavelength():.3f} nm\n"\
            f"Complementary Wavelength: {spec.complementary_wavelength():.3f} nm\n"\

    @observe('_source_spectrum_button', dispatch="ui")
    def show_source_spectrum(self, event=None) -> None:
        """
        render a Source Spectrum for the chosen Source, uses a separate thread.
        :param event: optional event from traits observe decorator
        """

        if self.raytracer.ray_sources and self.raytracer.rays.N:

            self._status["SourceSpectrum"] += 1
            self._spectrum_information = ""

            def background() -> None:

                error = False

                with self.__ray_access_lock:
                    with self._try() as error:
                        RS_Spec = self.raytracer.source_spectrum(source_index=self._source_ind)

                def on_finish() -> None:
                    if not error:
                        with self._try():
                            spectrum_plot(RS_Spec)
                        self._spectrum_information = self._get_spectrum_information(RS_Spec)

                    self._status["SourceSpectrum"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('_source_image_button, _source_cut_button', dispatch="ui")
    def show_source_image(self, event=None, cut=False) -> None:
        """
        render a source image for the chosen Source, uses a separate thread
        :param event: optional event from traits observe decorator
        :param cut: if an RImage cut plot is plotted
        """

        if self.raytracer.ray_sources and self.raytracer.rays.N:

            self._status["SourceImage"] += 1

            def background() -> None:

                error = False

                with self.__source_image_lock:
                    with self.__ray_access_lock:
                        # only calculate Image if raytracer Snapshot, selected Source or image_pixels changed
                        # otherwise we can replot the old Image with the new visual settings
                        snap = [self.raytracer.property_snapshot(), self.source_selection,
                                self.activate_filter, self.filter_constant]

                        rerender = snap != self._last_source_snap or self.last_source_image is None
                        log, mode, px, source_index = bool(self.log_image), self.image_type, int(self.image_pixels), \
                            self._source_ind
                        limit = None if not self.activate_filter else self.filter_constant
                        cut_args = {self.cut_dimension : self.cut_value}

                        if rerender:
                            with self._try() as error:
                                SImg = self.raytracer.source_image(N=px, source_index=source_index, limit=limit)

                    if not error:

                        if rerender:
                            self.last_source_image = SImg
                            self._last_source_snap = snap
                        else:
                            SImg = self.last_source_image.copy()
                            SImg.rescale(px)

                        Imc = SImg.get(mode, log=log)

                def on_finish() -> None:
                    if not error:
                        with self._try():
                            if (event is None and not cut) or (event is not None and event.name == "_source_image_button"):
                                r_image_plot(SImg, log=log, mode=mode, imc=Imc)
                            else:
                                r_image_cut_plot(SImg, log=log, mode=mode, imc=Imc, **cut_args)

                    self._status["SourceImage"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('_auto_focus_button', dispatch="ui")
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

                error = False

                source_index = None if not self.af_one_source else self._source_ind
                mode, det_pos, ret_cost, det_ind = self.focus_type, self.det_pos, bool(self.focus_cost_plot), \
                    self._det_ind

                with self.__ray_access_lock:
                    with self._try() as error:
                        res, afdict = self.raytracer.autofocus(mode, det_pos, return_cost=ret_cost,
                                                               source_index=source_index)

                if not error:
                    with self.__detector_lock:
                        if det_ind < len(self.raytracer.detectors):  # pragma: no branch
                            self.raytracer.detectors[det_ind].move_to([*self.raytracer.detectors[det_ind].pos[:2], 
                                                                       res.x])

                # execute this function after thread has finished
                def on_finish() -> None:

                    if not error:
                        bounds, pos, N = afdict["bounds"], afdict["pos"], afdict["N"]
                        
                        if det_ind < len(self.raytracer.detectors):  # pragma: no branch
                            self.det_pos = self.raytracer.detectors[det_ind].pos[2]

                        if self.focus_cost_plot:
                            with self._try():
                                autofocus_cost_plot(res, afdict, f"{mode} Cost Function\nMinimum at z={res.x:.5g}mm")

                        self._autofocus_information = \
                            f"Found 3D position: [{pos[0]:.7g}mm, {pos[1]:.7g}mm, {pos[2]:.7g}mm]\n"\
                            f"Search Region: z = [{bounds[0]:.7g}mm, {bounds[1]:.7g}mm]\n"\
                            f"Method: {mode}\n"\
                            f"Used {N} Rays for Autofocus\n"\
                            f"Ignoring Filters and Apertures\n\nOptimizeResult:\n{res}"

                    self._status["Focussing"] -= 1

                pyface_gui.invoke_later(on_finish)

            action = Thread(target=background, daemon=True)
            action.start()

    @observe('scene:activated', dispatch="ui")
    def _plot_scene(self, event=None) -> None:
        """
        Initialize the GUI. Inits a variety of things.

        :param event: optional event from traits observe decorator
        """
        self._plot.init_crosshair()
        self._plot.init_ray_info_text()
        self._plot.init_status_text()
        self._plot.init_keyboard_shortcuts()

        self._plot.plot_orientation_axes()
        self.replot()
        self._plot.default_camera_view()  # this needs to be called after replot, which defines the visual scope
        self._status["InitScene"] = 0
        
    @observe('_status:items', dispatch="ui")
    def _change_status(self, event=None) -> None:
        """
        Update the status info text.

        :param event: optional event from traits observe decorator
        """
        self._plot.set_status(self._status)

    @observe('minimalistic_view', dispatch="ui")
    def _change_minimalistic_view(self, event=None) -> None:
        """
        change the minimalistic ui view option.

        :param event: optional event from traits observe decorator
        """
        if not self._status["InitScene"]:
            self._status["Drawing"] += 1
            self._plot.change_minimalistic_view()
            self._status["Drawing"] -= 1
    
    @observe('vertical_labels')
    def change_label_orientation(self, event=None):
        """
        Make element labels horizontal or vertical

        :param event: optional event from traits observe decorator
        """
        self._status["Drawing"] += 1
        self._plot.change_label_orientation()
        self._status["Drawing"] -= 1

    @observe("maximize_scene", dispatch="ui")
    def _change_maximize_scene(self, event=None) -> None:
        """
        change maximize scene, hide side menu and toolbar"

        :param event: optional event from traits observe decorator
        """
        self._scene_not_maximized = not bool(self.maximize_scene)
        self.scene.scene_editor._tool_bar.setVisible(self._scene_not_maximized)

    @observe('ray_width', dispatch="ui")
    def _change_ray_width(self, event=None) -> None:
        """
        set the ray width for the visible rays.

        :param event: optional event from traits observe decorator
        """
        self._plot.set_ray_width()

    @observe('source_selection', dispatch="ui")
    def _change_selected_ray_source(self, event=None) -> None:
        """
        Updates the Detector Selection and the corresponding properties.

        :param event: optional event from traits observe decorator
        """
        if self.raytracer.ray_sources:
            self._source_ind = int(self.source_selection.split(":", 1)[0].split("RS")[1])

    @observe('raytracer_single_thread', dispatch="ui")
    def _change_raytracer_threading(self, event=None) -> None:
        """
        change the raytracer multithreading option. Useful for debugging.

        :param event: optional event from traits observe decorator
        """
        self.raytracer.threading = not bool(self.raytracer_single_thread)

    @observe('garbage_collector_stats', dispatch="ui")
    def _change_garbage_collector_stats(self, event=None) -> None:
        """
        show garbage collector stats

        :param event: optional event from traits observe decorator
        """
        gc.set_debug(gc.DEBUG_STATS if self.garbage_collector_stats else 0)

    @observe('show_all_warnings', dispatch="ui")
    def _change_warnings(self, event=None) -> None:
        """
        change warning visibility.

        :param event: optional event from traits observe decorator
        """
        if bool(self.show_all_warnings):
            warnings.simplefilter("always")
        else:
            warnings.resetwarnings()
    
    @observe('_command_window_button', dispatch="ui")
    def open_command_window(self, event=None) -> None:
        """
        Open the command window for executing code.
        
        :param event: optional event from traits observe decorator
        """
        if self._cdb is None:
            self._cdb = CommandWindow(self, self.silent)
        self._cdb.edit_traits()

    @observe('_property_browser_button', dispatch="ui")
    def open_property_browser(self, event=None) -> None:
        """
        Open a property browser for the gui, the scene, the raytracer, shown rays and cardinal points.

        :param event: optional event from traits observe decorator
        """
        gdb = PropertyBrowser(self)
        gdb.edit_traits()

    @observe('wireframe_surfaces', dispatch="ui")
    def _change_surface_mode(self, event=None) -> None:
        """
        change surface representation to normal or wireframe view.

        :param event: optional event from traits observe decorator
        """
        self._status["Drawing"] += 1
        self._plot.change_surface_mode()
        self._status["Drawing"] -= 1

    def send_cmd(self, cmd) -> None:
        """
        send/execute a command
        """

        if cmd != "":
            
            if self.busy and not self.command_dont_skip:
                if not self.silent:
                    print("Other actions running, try again when the program is idle.")
                return False

            self._status["RunningCommand"] += 1
        
            dict_ = dict(  # GUI and scene
                         mlab=self.scene.mlab, engine=self.scene.engine, scene=self.scene,
                         camera=self.scene.camera, GUI=self,

                         # abbreviations for raytracer and object lists
                         RT=self.raytracer, LL=self.raytracer.lenses, FL=self.raytracer.filters,
                         APL=self.raytracer.apertures, RSL=self.raytracer.ray_sources,
                         DL=self.raytracer.detectors, ML=self.raytracer.markers)
            
            pyface_gui.process_events()

            with self._try():
                hs = self.raytracer.property_snapshot()

                exec(cmd, locals() | dict_, globals())
        
                if not self.command_dont_replot:
                    hs2 = self.raytracer.property_snapshot()
                    cmp = self.raytracer.compare_property_snapshot(hs, hs2)
                    self.replot(cmp)

            self._status["RunningCommand"] -= 1
            return True

        return False

    # rescale axes texts when the window was resized
    # for some reasons these are the only ones having no text_scaling_mode = 'none' option
    # this is the only trait so far that does not work with @observe, like:
    # @observe("scene:scene_editor:interactor:size:items:value")
    @observe("scene:scene_editor:busy", dispatch="ui")
    def _resize_scene_elements(self, event) -> None:
        """
        Handles GUI window size changes. Fixes incorrect scaling by mayavi.

        :param event: event from traits observe decorator
        """
        self._plot.resize_scene_elements()
