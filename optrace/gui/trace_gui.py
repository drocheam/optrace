
import time  # provides sleeping
import traceback  # traceback printing
from threading import Thread, Lock  # threading
from typing import Callable, Any  # typing types
from contextlib import contextmanager  # context managers
import numpy as np
                
from pyface.api import GUI as pyface_gui  # invoke_later() method
from pyface.qt import QtGui, QtCore  # closing UI elements

# traits types and UI elements
from traitsui.api import Group as TGroup
from traitsui.api import View, Item, HSplit, CheckListEditor, TextEditor, RangeEditor
from traits.api import HasTraits, Range, Instance, observe, Str, Button, Enum, List, Dict, Float, Bool

from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

# provides types and plotting functionality
from ..tracer import *
from ..plots import image_plot, image_cut_plot, autofocus_cost_plot, spectrum_plot  # different plots

from .property_browser import PropertyBrowser  # dictionary browser
from .command_window import CommandWindow
from ._scene_plotting import ScenePlotting

from ..tracer.misc import PropertyChecker as pc
from ..warnings import warning
from ..global_options import global_options
from .. import metadata



class TraceGUI(HasTraits):

    scene: Instance = Instance(MlabSceneModel, args=())
    """the mayavi scene"""

    # Ranges

    ray_count: Range = Range(1, 50000000, 200000, desc='Number of rays for Simulation', enter_set=True,
                             auto_set=False, label="Rays", mode='spinner')
    """Number of rays for raytracing"""
    
    filter_constant: Range = Range(0.3, 50., 1., desc='filter constant', enter_set=True,
                             auto_set=False, label="Limit (µm)", mode='text')
    """gaussian filter constant standard deviation"""

    rays_visible: Range = Range(1, ScenePlotting.MAX_RAYS_SHOWN, 2000, desc='the number of rays which is drawn', enter_set=False,
                                auto_set=False, label="Count", mode='logslider')
    """Number of rays shown in mayavi scene"""

    z_det: Range = Range(low='_z_det_min', high='_z_det_max', value='_z_det_max', mode='text',
                         desc='z-Position of the Detector', enter_set=True, auto_set=True, label="z_det")
    """z-Position of Detector. Lies inside z-range of `Raytracer.outline`"""

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

    log_image: List = List(editor=CheckListEditor(values=['Logarithmic Scaling'], format_func=lambda x: x),
                           desc="if Logarithmic Values are Plotted")
    """Boolean value for a logarithmic image visualization. Packed as Checkbox into a :obj:`List`"""

    flip_detector_image: List = List(editor=CheckListEditor(values=['Flip Detector Image'], format_func=lambda x: x),
                                     desc="if detector image should be rotated by 180 degrees")
    """Boolean value for flipping the image (rotating it by 180°). Packed as Checkbox into a :obj:`List`"""

    cost_function_plot: List = List(editor=CheckListEditor(values=['Plot Cost Function'], format_func=lambda x: x),
                                    desc="if cost function is shown")
    """Show a plot of the optimization cost function for the focus finding"""

    minimalistic_view: List = List(editor=CheckListEditor(values=['More Minimalistic Scene'], format_func=lambda x: x),
                                   desc="if some scene elements should be hidden")
    """More minimalistic scene view. Axes are hidden and labels shortened."""
    
    hide_labels: List = List(editor=CheckListEditor(values=['Hide Labels'], format_func=lambda x: x),
                                   desc="if object labels should be hidden")
    """Shows or hides object labels"""

    autofocus_single_source: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'],
                                                                format_func=lambda x: x),
                                         desc="if autofocus only uses currently selected source")
    """Use only rays from selected source for focus finding"""

    detector_image_single_source: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'],
                                                                     format_func=lambda x: x),
                                              desc="if DetectorImage only uses currently selected source")
    """Use only rays from selected source for the detector image"""

    detector_spectrum_single_source: List = List(editor=CheckListEditor(values=['Rays From Selected Source Only'],
                                                                        format_func=lambda x: x),
                                                 desc="if Detector Spectrum only uses currently selected source")
    """Use only rays from selected source for the detector spectrum"""

    vertical_labels: List = List(editor=CheckListEditor(values=['Vertical Labels'], format_func=lambda x: x),
                                   desc="if element labels are displayed vertically in scene")
    """in geometries with tight geometry and long descriptions one could want to display the labels vertically"""

    activate_filter: List = List(editor=CheckListEditor(values=['Activate Filter'], format_func=lambda x: x),
                                   desc="if gaussian filter is applied")
    """gaussian blur filter estimating resolution limit"""

    maximize_scene: \
        List = List(editor=CheckListEditor(values=["Maximize Scene (Press h in Scene)"], format_func=lambda x: x),
                    desc="Maximizes Scene, hides side menu and toolbar")
    """maximize mode. Menu and toolbar hidden"""
    
    high_contrast: \
        List = List(editor=CheckListEditor(values=["High Contrast Mode"], format_func=lambda x: x),
                    desc="show objects in black on white background.")
    """high contrast mode without object colors and with white background"""
    
    # Enums

    plotting_modes: list = ['Rays', 'Points']  #: available ray representations
    plotting_mode: Enum = Enum(*plotting_modes, desc="Ray Representation")
    """Ray Plotting Type"""

    coloring_modes: list = ['Plain', 'Power', 'Wavelength', 'Source', 'Polarization xz',\
                            'Polarization yz', 'Refractive Index']  
    """available ray coloring modes"""
    coloring_mode: Enum = Enum(*coloring_modes, desc="Ray Property to color the Rays With")
    """Ray Coloring Mode"""

    image_mode: Enum = Enum(*RenderImage.image_modes, desc="Image Type Presented")
    """Image Type"""
   
    projection_method_enabled: Bool = False  #: if method is selectable, only the case for spherical detectors
    projection_method: Enum = Enum(*SphericalSurface.sphere_projection_methods,
                                   desc="Projection Method for spherical detectors")
    """sphere surface projection method"""

    autofocus_method: Enum = Enum(*Raytracer.autofocus_methods, desc="Method for Autofocus")
    """Focus Finding Mode from raytracer.AutofocusModes"""

    cut_dimension: Enum = Enum(["y", "x"], desc="image cut dimension")
    """dimension for image cut"""

    source_names: List = List()  #: short names for raytracer ray sources
    source_selection: Enum = Enum(values='source_names', desc="Source Selection for Source Image")
    """Source Selection. Holds the name of one of the Sources."""

    detector_names: List  = List()  #: short names for raytracer detectors
    detector_selection: Enum = Enum(values='detector_names', desc="Detector Selection")
    """Detector Selection. Holds the name of one of the Detectors."""

    image_pixels: Enum = Enum(RenderImage.SIZES, label="Pixels_xy",
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
    _filter_label:               Str = Str('Resolution Limit Filter:')
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
                    TGroup(  # additional group so scene-sidebar ratio stays the same on windows resizing
                        TGroup(
                            Item('scene', editor=SceneEditor(scene_class=MayaviScene), resizable=True,
                                 height=_scene_size0[1], width=_scene_size0[0], show_label=False),
                             ),
                        layout="split",
                    ),
                    TGroup(
                        TGroup(
                            _separator,
                            Item("_trace_label", style='readonly', show_label=False, emphasized=True),
                            Item('ray_count'),
                            _separator, _separator,
                            Item("_plotting_label", style='readonly', show_label=False, emphasized=True),
                            Item("plotting_mode", label='Plotting'),
                            Item("coloring_mode", label='Coloring'),
                            _separator, _separator,
                            Item("_ray_visual_label", style='readonly', show_label=False, emphasized=True),
                            Item('rays_visible'),
                            Item('ray_opacity'),
                            Item('ray_width'),
                            _separator, _separator,
                            Item("_ui_visual_label", style='readonly', show_label=False, emphasized=True),
                            Item('minimalistic_view', style="custom", show_label=False),
                            Item('maximize_scene', style="custom", show_label=False),
                            Item('high_contrast', style="custom", show_label=False),
                            Item('vertical_labels', style="custom", show_label=False),
                            Item('hide_labels', style="custom", show_label=False),
                            _separator, _separator,
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
                            Item('z_det'),
                            _separator,
                            Item("_image_label", style='readonly', show_label=False, emphasized=True),
                            Item('image_mode', label='Mode'),
                            Item('projection_method', label='Projection', enabled_when="projection_method_enabled"),
                            Item('image_pixels'),
                            Item('log_image', style="custom", show_label=False),
                            Item('flip_detector_image', style="custom", show_label=False),
                            Item('detector_image_single_source', style="custom", show_label=False),
                            Item('_source_image_button', show_label=False),
                            Item('_detector_image_button', show_label=False),
                            _separator,
                            Item("cut_dimension", label="Cut at"),
                            Item("cut_value", label="Value"),
                            Item('_source_cut_button', show_label=False),
                            Item('_detector_cut_button', show_label=False),
                            _separator,
                            Item("_filter_label", style='readonly', show_label=False, emphasized=True,
                                 enabled_when="not projection_method_enabled"),
                            Item('activate_filter', style="custom", show_label=False,
                                 enabled_when="not projection_method_enabled"),
                            Item('filter_constant', enabled_when="not projection_method_enabled"),
                            label="Image",
                            ),
                        TGroup(
                            _separator,
                            Item("_geometry_label", style='readonly', show_label=False, emphasized=True),
                            Item('source_selection', label="Source"),
                            Item('detector_selection', label="Detector"),
                            Item('z_det'),
                            _separator, _separator,
                            Item("_spectrum_label", style='readonly', show_label=False, emphasized=True),
                            Item('_source_spectrum_button', show_label=False),
                            Item('detector_spectrum_single_source', style="custom", show_label=False),
                            Item('_detector_spectrum_button', show_label=False),
                            _separator, _separator,
                            Item("_spectrum_output_label", style='readonly', show_label=False, emphasized=True),
                            Item("_spectrum_information", show_label=False, style="custom"),
                            label="Spectrum",
                            ),
                        TGroup(
                            _separator,
                            Item("_geometry_label", style='readonly', show_label=False, emphasized=True),
                            Item('source_selection', label="Source"),
                            Item('detector_selection', label="Detector"),
                            Item('z_det'),
                            _separator, _separator,
                            Item("_autofocus_label", style='readonly', show_label=False, emphasized=True),
                            Item('autofocus_method', label='Mode'),
                            Item('autofocus_single_source', style="custom", show_label=False),
                            Item('cost_function_plot', style="custom", show_label=False),
                            Item('_auto_focus_button', show_label=False),
                            _separator,
                            Item("_autofocus_output_label", style='readonly', show_label=False),
                            Item("_autofocus_information", show_label=False, style="custom"),
                            _separator,
                            label="Focus",
                            ),
                        layout="tabbed",
                        visible_when="_scene_not_maximized",
                        ),
                    ),
                resizable=True,
                title=f"Optrace {metadata.version}",
                # icon=""  # TODO create an icon and set the path here and in sub-windows
                )
    """the UI view"""

    ####################################################################################################################
    # Class constructor

    def __init__(self, 
                 raytracer:         Raytracer, 
                 initial_camera:    dict = {},
                 **kwargs) -> None:
        """
        Create a new TraceGUI with the assigned Raytracer.

        :param RT: raytracer
        :param initial_camera: parameter dictionary for set_camera
        :param kwargs: additional arguments, options, traits and parameters
        """
        pc.check_type("raytracer", raytracer, Raytracer)
        pc.check_type("initial_camera", initial_camera, dict)

        # allow SIGINT interrupts
        pyface_gui.allow_interrupt()

        self._plot = ScenePlotting(self, raytracer, initial_camera)

        # lock object for multithreading
        self.__detector_lock = Lock()  # when detector is changed, moved or used for rendering
        self.__det_image_lock = Lock()  # when last detector image changed
        self.__source_image_lock = Lock()  # when last source image changed
        self.__ray_access_lock = Lock()  # when rays read or written

        self.raytracer: Raytracer = raytracer
        """reference to the raytracer"""

        self.property_browser = PropertyBrowser(self)
        self.command_window = CommandWindow(self)
        self._property_browser_view = None
        self._command_window_view = None
        
        self._sequential = False
 
        # minimal/maximal z-positions of Detector_obj
        self._z_det_min = self.raytracer.outline[4]
        self._z_det_max = self.raytracer.outline[5]

        self._det_ind = 0
        if len(self.raytracer.detectors):
            self.z_det = self.raytracer.detectors[self._det_ind].pos[2]
        self._source_ind = 0

        self._exit = False

        self._last_det_snap = None
        self.last_det_image = None
        self._last_source_snap = None
        self.last_source_image = None

        self._init_forced_ray_count = "ray_count" in kwargs

        # default value for image_pixels
        self.image_pixels = 189

        # set the properties after this without leading to a re-draw
        self._no_trait_action_flag = False
       
        # these properties should not be set, since they have to be initialized first
        # and setting as inits them would lead to issues
        forbidden = ["source_selection", "detector_selection", "z_det"]
        if any(key in kwargs for key in forbidden):
            raise RuntimeError(f"Assigning an initial value for properties '{forbidden}' is not supported")

        # one element list traits that are treated as bool values
        self._bool_list_elements = [el for el in self.trait_get() if (self.trait(el).default_kind == "list")\
                                    and self._trait(el, 0).editor is not None and\
                                    (len(self._trait(el, 0).editor.values) == 1)]

        with self._no_trait_action():

            # convert bool values to list entries for List()
            for key, val in kwargs.items():
                if key in self._bool_list_elements and isinstance(val, bool):
                    kwargs[key] = [self._trait(key, 0).editor.values[0]] if val else []

            # define Status dict
            self._status.update(dict(InitScene=1, DisplayingGUI=1, Tracing=0, Drawing=0, Focussing=0,
                                     DetectorImage=0, SourceImage=0, ChangingDetector=0, SourceSpectrum=0,
                                     DetectorSpectrum=0, Screenshot=0, RunningCommand=0))

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
    def _no_trait_action(self) -> None:
        """context manager that sets and resets the _no_trait_action_flag"""
        self._no_trait_action_flag = True
        try:
            yield
        finally:
            self._no_trait_action_flag = False
    
    @contextmanager
    def smart_replot(self, automatic_replot: bool = True) -> None:
        """
        This contextmanager makes a property snapshot of the raytracer before and after a specific action.
        If changes are detected. the corresponding elements are updated.

        :param automatic_replot: if automatic replotting should be executed
        """
        if automatic_replot:
            hs = self.raytracer.property_snapshot()
        
        try:
            yield
        
        finally:
            if automatic_replot:
                hs2 = self.raytracer.property_snapshot()
                cmp = self.raytracer.compare_property_snapshot(hs, hs2)
                self.replot(cmp)

    def process(self) -> None:
        """
        Force an update to traits and unprocessed events.
        """
        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 10000)

    def _start_action(self, func, args = ()) -> None:
        """execute an action. If _sequential is set, is is run in the main thread, in a different thread otherwise"""
        if self._sequential:
            func(*args)
        else:
            action = Thread(target=func, args=args, daemon=True)
            action.start()

    @contextmanager
    def _try(self) -> bool:
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

            warning(traceback.format_exc())  # print traceback
            error[:] = [True]  # change reference, now evals to true
    
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

        self.process()
        self._status["DisplayingGUI"] = 0

        if self._exit:
            pyface_gui.invoke_later(self.close)

    ####################################################################################################################
    # Interface functions
    ####################################################################################################################

    def screenshot(self, path: str, **kwargs) -> None:
        """
        Save a screenshot of the scene. Passes the parameters down to the mlab.savefig function.
        See `https://docs.enthought.com/mayavi/mayavi/auto/mlab_figure.html#savefig` for parameters.
        """
        pc.check_type("path", path, str)
       
        self._status["Screenshot"] += 1
        self._plot.screenshot(path, **kwargs)
        self._status["Screenshot"] -= 1

    def set_camera(self, 
                   center:      np.ndarray = None, 
                   height:      float = None, 
                   direction:   list = None,
                   roll:        float = None)\
            -> None:
        """
        Sets the camera view.
        Not all parameters must be defined, setting single properties is also allowed.
        
        :param focal_point: 3D position of camera focal point, also defines the center of the view
        :param parallel_scale: view scaling. Defines half the size of the view in vertical direction.
        :param center: 3D coordinates of center of view
        :param height: half of vertical height in mm
        :param direction: camera view direction vector
         (direction of vector perpendicular to your monitor and in your viewing direction)
        :param roll: absolute camera roll angle
        """
        pc.check_type("center", center, list | np.ndarray | None)
        pc.check_type("height", height, float | int | None)
        pc.check_type("direction", direction, list | np.ndarray | None)
        pc.check_type("roll", roll, float | int | None)
        
        if height is not None:
            pc.check_above("height", height, 0)

        self._status["Drawing"] += 1
        self._plot.set_camera(center, height, direction, roll)
        self._status["Drawing"] -= 1
    
    def get_camera(self) -> tuple[np.ndarray, float, np.ndarray, float]:
        """
        Get the camera parameters that can be passed down to set_camera()

        :return: Return the current camera parameters (center, height, direction, roll)
        """
        return self._plot.get_camera()

    def pick_ray(self, index: int) -> None:
        """
        Pick a ray with index 'index'.
        The selected ray is highlighted.

        :param index: is the index of the displayed rays.
        """
        self.pick_ray_section(index, None, False)

    def pick_ray_section(self, 
                         index:     int, 
                         section:   int, 
                         detailed:  bool = False)\
            -> None:
        """
        Pick a ray with index 'index' at intersection number 'section'.
        A ray highlight plot, crosshair and ray info text are shown.

        :param index: is the index of the displayed rays.
        :param section: intersection index (starting position is zero)
        :param detailed: If 'detailed' is set, a more detailed ray info text is shown
        """
        pc.check_type("index", index, int | np.int64)
        pc.check_type("section", section, int | np.int64 | None)

        self._status["Drawing"] += 1
        with self._try():
            self._plot.pick_ray_section(index, section, detailed)
        self._status["Drawing"] -= 1
    
    def reset_picking(self) -> None:
        """
        Hide/reset the displayed picker objects.
        Hides the ray info text, ray highlight and crosshair that have been created
        with TraceGUI.pick_ray or TraceGUI.pick_ray_section
        """
        self._status["Drawing"] += 1
        self._plot.reset_picking()
        self._status["Drawing"] -= 1

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
        self.process()
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
                self._z_det_min = self.raytracer.outline[4]
                self._z_det_max = self.raytracer.outline[5]
            
            if (all_ or change["Filters"] or change["Lenses"] or change["Apertures"] or change["Ambient"]\
                    or change["RaySources"] or change["TraceSettings"]) and not rdh:

                # when initializing the scene, don't retrace when the raytracer already has rays stored, 
                # there are no geometry errors and the ray_count parameter for the TraceGUI is not explicitely set
                # and the scene did not change since the last trace
                if self._status["DisplayingGUI"] and self.raytracer.rays.N and not self._init_forced_ray_count\
                        and not self.raytracer.geometry_error and self.raytracer.check_if_rays_are_current():
                    with self._no_trait_action():
                        self.ray_count = self.raytracer.rays.N
                    self.replot_rays()
                else:
                    self.retrace()

            elif all_ or change["Rays"]:
                self.replot_rays()

        self.scene.disable_render = False
        self.scene.render()
        self.process()

        if rdh and self._status["DisplayingGUI"]:  # reset initial loading flag
            # this only gets invoked after raytracing, but with no sources we need to do it here
            pyface_gui.invoke_later(self._set_gui_loaded)

        self._status["Drawing"] -= 1
        self.process()  # needed so rays are displayed in replot() 
        # with _sequential=True (e.g. in TraceGUI.control())

    @property
    def busy(self) -> bool:
        """busy state flag of TraceGUI"""
        # currently I see no way to check with 100% certainty if the TraceGUI is actually idle
        # it could also be in some trait_handlers or functions from other classes
        return self.scene.busy or any(val > 0 for val in self._status.values())

    def debug(self, func: Callable, args: tuple = (), kwargs: dict = {}) -> None:
        """
        Control the GUI from a separate thread.

        :param func: thread function to execute while the GUI is active
        :param args: positional arguments for func
        :param kwargs: keyword arguments for func
        """
        th = Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        th.start()

        self.configure_traits()
    
    def control(self, func: Callable, args: tuple = (), kwargs: dict = {}) -> None:
        """
        Interact with the TraceGUI.
        Function 'func' is executed in the main thread after the TraceGUI is loaded.

        :param func: function to execute
        :param args: positional arguments for func
        :param kwargs: keyword arguments for func
        """
        pc.check_type("func", func, Callable)
        pc.check_type("args", args, tuple)
        pc.check_type("kwargs", kwargs, dict)

        # in "control" mode everything is handled sequentially, so actions don't start
        # in separate threads which could lead to runtime issues
        def func2(*args, **kwargs):

            while self.busy:
                time.sleep(0.05)

            def func3(*args, **kwargs):
                self._sequential = True
                func(*args, **kwargs)
                self._sequential = False
            
            pyface_gui.invoke_later(func3, *args, **kwargs)

        th = Thread(target=func2, args=args, kwargs=kwargs, daemon=True)
        th.start()

        self.configure_traits()

    def run(self) -> None:
        """Run the TraceGUI"""
        self.configure_traits()

    @observe('scene:closing', dispatch="ui")  # needed so this gets also called when clicking x on main window
    def close(self, event=None) -> None:
        """
        Close the whole application.
        
        :param event: optional event from traits observe decorator
        """
        if self._property_browser_view is not None and self._property_browser_view.control is not None:
            self._property_browser_view.control.window().close()

        if self._command_window_view is not None and self._command_window_view.control is not None:
            self._command_window_view.control.window().close()
        
        pyface_gui.invoke_later(QtGui.QApplication.closeAllWindows)  # close remaining Qt windows

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

    @observe('rays_visible', dispatch="ui")
    def replot_rays(self, event=None, mask: np.ndarray = None, max_show: int = None) -> None:
        """
        choose a subset of all raytracer rays and plot them with :`TraceGUI._plot_rays`.

        By default, a random subset is selected.
        When mask and max_show (both optional) are provided, a specific subset is displayed.
        See TraceGUI.select_rays.

        :param event: optional event from traits observe decorator
        :param mask: boolean array for which rays to display. Shape must equal the number of currently traced rays.
        :param max_show: maximum number of rays to display
        """

        if not self._no_trait_action_flag and not self._status["Tracing"]: 
            # don't do while tracing, RayStorage rays are still being generated

            self._status["Drawing"] += 1
            self.process()

            if not self.raytracer.ray_sources or not self.raytracer.rays.N:
                self._plot.remove_rays()
                self._status["Drawing"] -= 1
                return

            def background():
                with self._try() as error:
                    with self.__ray_access_lock:
                        if mask is None:
                            self._plot.random_ray_selection()
                        else:
                            self._plot.select_rays(mask, max_show)
                        res = self._plot.get_rays()

                def on_finish():

                    if not error:

                        self._plot.assign_ray_properties()

                        with self._plot.constant_camera():
                            self._plot.plot_rays(*res)
                            self._change_ray_and_source_colors()

                        # reset picker text, highlight plot and crosshair
                        self._plot.clear_ray_text()
                        self._plot.hide_crosshair()
                        self._plot.hide_ray_highlight()
                   
                        # if a custom selection was provided, we need to update the number of displayed rays
                        if mask is not None:
                            with self._no_trait_action():
                                self.rays_visible = np.count_nonzero(self.ray_selection)

                    self._status["Drawing"] -= 1

                    # gets unset on first run
                    if self._status["DisplayingGUI"]:
                        pyface_gui.invoke_later(self._set_gui_loaded)
            
                pyface_gui.invoke_later(on_finish)
            
            self._start_action(background)

    def select_rays(self, mask: np.ndarray, max_show: int = None) -> None:
        """
        Apply a specific selection of rays for display.
        If the number is too large it is either limited by the 'max_show' parameter or a predefined limit.

        :param mask: boolean array for which rays to display. Shape must equal the number of currently traced rays.
        :param max_show: maximum number of rays to display
        """
        pc.check_type("mask", mask, np.ndarray)
        pc.check_type("max_show", max_show, int | None)

        if not self._status["Tracing"]:
            self.replot_rays(None, mask, max_show)
        else:
            warning(f"Did not apply the ray selection as the raytracer is still tracing.")

    @property
    def ray_selection(self) -> np.ndarray:
        """:return: boolean array for which of the traced rays in TraceGUI.raytracer.rays are displayed currently"""
        return self._plot.ray_selection

    @observe('ray_count', dispatch="ui")
    def retrace(self, event=None) -> None:
        """
        raytrace in separate thread, after that call `TraceGUI.replot_rays`.

        :param event: optional event from traits observe decorator
        """

        if self._no_trait_action_flag:
            return

        elif self.raytracer.ray_sources:

            nt = len(self.raytracer.tracing_surfaces) + 2
            if self.raytracer.rays.storage_size(self.ray_count, nt, self.raytracer.rays.no_pol)\
                    > self.raytracer.MAX_RAY_STORAGE_RAM:
                warning(f"Resetting the ray_count number since more than {self.raytracer.MAX_RAY_STORAGE_RAM*1e-9:.1f}"
                        " GB RAM requested. Either decrease the number of rays, surfaces or do an iterative render. "
                        "If your system can handle"
                        " more RAM usage, increase the Raytracer.MAX_RAY_STORAGE_RAM parameter.")
                pyface_gui.set_trait_later(self, "ray_count", event.old)
                return

            self._status["Tracing"] += 1

            # run this in background thread
            def background() -> None:

                error_state = False

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

            self._start_action(background)
        else:
            pyface_gui.invoke_later(self._set_gui_loaded)

    @observe('ray_opacity', dispatch="ui")
    def _change_ray_opacity(self, event=None) -> None:
        """
        change opacity of visible rays.

        :param event: optional event from traits observe decorator
        """
        self._plot.set_ray_opacity()

    @observe("coloring_mode", dispatch="ui")
    def _change_ray_and_source_colors(self, event=None) -> None:
        """
        change ray coloring mode.

        :param event: optional event from traits observe decorator
        """
        self._plot.color_rays()
        self._plot.color_ray_sources()

    @observe("plotting_mode", dispatch="ui")
    def _change_ray_representation(self, event=None) -> None:
        """
        change ray view to connected lines or points only.

        :param event: optional event from traits observe decorator
        """
        self._plot.set_ray_representation()

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
                        self.z_det = self.raytracer.detectors[self._det_ind].pos[2]

                    self.projection_method_enabled = \
                        isinstance(self.raytracer.detectors[self._det_ind].surface, SphericalSurface)
                    self.__detector_lock.release()
                    self._status["ChangingDetector"] -= 1

                self.__detector_lock.acquire()
                pyface_gui.invoke_later(on_finish)

            self._start_action(background)

    @observe('z_det', dispatch="ui")
    def _move_detector(self, event=None) -> None:
        """
        Move current detector.

        :param event: optional event from traits observe decorator
        """
        if not self._no_trait_action_flag and self.raytracer.detectors:

            self._status["ChangingDetector"] += 1

            def background():

                def on_finish():

                    # move detector object
                    xp, yp, zp = self.raytracer.detectors[self._det_ind].pos
                    self.raytracer.detectors[self._det_ind].move_to([xp, yp, self.z_det])
                    self._plot.move_detector_diff(self._det_ind, event.new - event.old)

                    # reinit detectors and Selection to update z_pos in detector name
                    self._init_detector_list()
                    with self._no_trait_action():
                        self.detector_selection = self.detector_names[self._det_ind]

                    self.__detector_lock.release()
                    self._status["ChangingDetector"] -= 1

                self.__detector_lock.acquire()
                pyface_gui.invoke_later(on_finish)

            self._start_action(background)

    @observe('_detector_cut_button', dispatch="ui")
    def detector_cut(self, event = None, extent: list | np.ndarray = None, **kwargs) -> None:
        """
        Plot a detector image cut.

        :param event: optional event from traits observe decorator
        :param extent: image extent, see Raytracer.detector_image()
        :param kwargs: additional keyword arguments for r_image_cut_plot
        """
        self.detector_image(event, cut=True, extent=extent, **kwargs)

    @observe('_detector_image_button', dispatch="ui")
    def detector_image(self, 
                       event        = None, 
                       cut:         bool = False, 
                       extent:      list | np.ndarray = None, 
                       **kwargs)\
            -> None:
        """
        Render a detector image at the chosen Detector, uses a separate thread.

        :param event: optional event from traits observe decorator
        :param cut: if a Image cut image is plotted
        :param extent: image extent, see Raytracer.detector_image()
        :param kwargs: additional keyword arguments for r_image_plot
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
                            source_index = None if not self.detector_image_single_source else self._source_ind
                            limit = self.filter_constant if self.activate_filter \
                                                            and not self.projection_method_enabled else None
                            
                            snap = [self.raytracer.property_snapshot(), *self.detector_selection, source_index, 
                                    *self.projection_method, limit, extent]
                            rerender = snap != self._last_det_snap or self.last_det_image is None

                            if rerender:
                                with self._try() as error:
                                    DImg = self.raytracer.detector_image(detector_index=self._det_ind, extent=extent,
                                                                         source_index=source_index, limit=limit,
                                                                         projection_method=self.projection_method)
                    if not error:

                        if rerender:
                            self.last_det_image = DImg
                            self._last_det_snap = snap
                        else:
                            DImg = self.last_det_image.copy()

                        img = DImg.get(self.image_mode, int(self.image_pixels))

                def on_finish() -> None:

                    if not error:

                        with self._try():

                            if (event is None and not cut) or (event is not None\
                                    and event.name == "_detector_image_button"):
                                image_plot(img, log=bool(self.log_image), flip=bool(self.flip_detector_image), **kwargs)

                            else:
                                cut_args = {self.cut_dimension : self.cut_value}
                                image_cut_plot(img, log=bool(self.log_image),
                                               flip=bool(self.flip_detector_image), **cut_args, **kwargs)

                    self._status["DetectorImage"] -= 1

                pyface_gui.invoke_later(on_finish)

            self._start_action(background)

    @observe('_detector_spectrum_button', dispatch="ui")
    def detector_spectrum(self, event=None, extent: list | np.ndarray = None, **kwargs) -> None:
        """
        Render a Detector Spectrum for the chosen Source, uses a separate thread.

        :param event: optional event from traits observe decorator
        :param extent: image extent, see Raytracer.detector_image()
        :param kwargs: additional keyword arguments for spectrum_plot
        """

        if self.raytracer.detectors and self.raytracer.rays.N:

            self._status["DetectorSpectrum"] += 1
            self._spectrum_information = ""

            def background() -> None:

                error = False

                with self.__detector_lock:
                    with self.__ray_access_lock:
                        source_index = None if not self.detector_spectrum_single_source else self._source_ind
                        with self._try() as error:
                            Det_Spec = self.raytracer.detector_spectrum(detector_index=self._det_ind, extent=extent,
                                                                        source_index=source_index)

                def on_finish() -> None:
                    if not error:
                        with self._try():
                            spectrum_plot(Det_Spec, **kwargs)
                        self._spectrum_information = self._get_spectrum_information(Det_Spec)

                    self._status["DetectorSpectrum"] -= 1

                pyface_gui.invoke_later(on_finish)

            self._start_action(background)

    @observe('_source_cut_button', dispatch="ui")
    def source_cut(self, event = None, **kwargs) -> None:
        """
        Plot a source image cut.

        :param event: optional event from traits observe decorator
        :param kwargs: additional keyword arguments for r_image_cut_plot
        """
        self.source_image(event, cut=True, **kwargs)
    
    def _get_spectrum_information(self, spec) -> str:

        stats = f"{spec.get_long_desc()}\n\n"\
            f"Power: {spec.power():.6g} W\n"\
            f"Luminous Power: {spec.luminous_power():.6g} lm\n\n"\
            f"Peak: {spec.peak():.6g} {spec.unit}\n"\
            f"Peak Wavelength: {spec.peak_wavelength():.3f} nm\n"\
            f"Centroid Wavelength: {spec.centroid_wavelength():.3f} nm\n"\
            f"FWHM: {spec.fwhm():.3f} nm\n\n"\

        dom = spec.dominant_wavelength()
        stats += "Dominant Wavelength: " + (f"{dom:.3f} nm" if np.isfinite(dom) else "None")

        comp = spec.complementary_wavelength()
        stats += "\nComplementary Wavelength: " + (f"{comp:.3f} nm" if np.isfinite(comp) else "None")

        return stats

    @observe('_source_spectrum_button', dispatch="ui")
    def source_spectrum(self, event=None, **kwargs) -> None:
        """
        render a Source Spectrum for the chosen Source, uses a separate thread.

        :param event: optional event from traits observe decorator
        :param kwargs: additional keyword arguments for spectrum_plot
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
                            spectrum_plot(RS_Spec, **kwargs)
                        self._spectrum_information = self._get_spectrum_information(RS_Spec)

                    self._status["SourceSpectrum"] -= 1

                pyface_gui.invoke_later(on_finish)

            self._start_action(background)

    @observe('_source_image_button', dispatch="ui")
    def source_image(self, event = None, cut: bool = False, **kwargs) -> None:
        """
        Render a source image for the chosen Source, uses a separate thread

        :param event: optional event from traits observe decorator
        :param cut: if an Image cut plot is plotted
        :param kwargs: additional keyword arguments for image_plot
        """

        if self.raytracer.ray_sources and self.raytracer.rays.N:

            self._status["SourceImage"] += 1

            def background() -> None:

                error = False

                with self.__source_image_lock:
                    with self.__ray_access_lock:
                        limit = None if not self.activate_filter else self.filter_constant
                        
                        # only calculate Image if raytracer Snapshot, selected Source or image_pixels changed
                        # otherwise we can replot the old Image with the new visual settings
                        snap = [self.raytracer.property_snapshot(), *self.source_selection, limit]
                        rerender = snap != self._last_source_snap or self.last_source_image is None

                        if rerender:
                            with self._try() as error:
                                SImg = self.raytracer.source_image(source_index=self._source_ind, limit=limit)

                    if not error:

                        if rerender:
                            self.last_source_image = SImg
                            self._last_source_snap = snap
                        else:
                            SImg = self.last_source_image.copy()

                        img = SImg.get(self.image_mode, int(self.image_pixels))

                def on_finish() -> None:
                    if not error:
                        with self._try():
                            if (event is None and not cut) or \
                                    (event is not None and event.name == "_source_image_button"):
                                image_plot(img, log=bool(self.log_image), **kwargs)
                            else:
                                cut_args = {self.cut_dimension : self.cut_value}
                                image_cut_plot(img, log=bool(self.log_image), **cut_args, **kwargs)

                    self._status["SourceImage"] -= 1

                pyface_gui.invoke_later(on_finish)

            self._start_action(background)

    @observe('_auto_focus_button', dispatch="ui")
    def move_to_focus(self, event=None, **kwargs) -> None:
        """
        Find a Focus.
        The chosen Detector defines the search range for focus finding.
        Searches are always between lenses or the next outline.
        Search takes place in a separate thread, after that the Detector is moved to the focus

        :param event: optional event from traits observe decorator
        :param kwargs: additional keyword arguments for autofocus_cost_plot
        """

        if self.raytracer.detectors and self.raytracer.ray_sources and self.raytracer.rays.N:

            self._status["Focussing"] += 1
            self._autofocus_information = ""

            def background() -> None:

                error = False

                source_index = None if not self.autofocus_single_source else self._source_ind
                mode, z_det, ret_cost, det_ind = self.autofocus_method, self.z_det, bool(self.cost_function_plot), \
                    self._det_ind

                with self.__ray_access_lock:
                    with self._try() as error:
                        res, afdict = self.raytracer.autofocus(mode, z_det, return_cost=ret_cost,
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
                            self.z_det = self.raytracer.detectors[det_ind].pos[2]

                        if self.cost_function_plot:
                            with self._try():
                                autofocus_cost_plot(res, afdict, f"{mode} Cost Function\nMinimum at z={res.x:.5g}mm",
                                                    **kwargs)

                        self._autofocus_information = \
                            f"Found 3D position: [{pos[0]:.7g}mm, {pos[1]:.7g}mm, {pos[2]:.7g}mm]\n"\
                            f"Search Region: z = [{bounds[0]:.7g}mm, {bounds[1]:.7g}mm]\n"\
                            f"Method: {mode}\n"\
                            f"Used {N} Rays for Autofocus\n"\
                            f"Ignoring Filters and Apertures\n\nOptimizeResult:\n{res}"

                    self._status["Focussing"] -= 1

                pyface_gui.invoke_later(on_finish)

            self._start_action(background)

    @observe('scene:activated', dispatch="ui")
    def _plot_scene(self, event=None) -> None:
        """
        Initialize the GUI. Inits a variety of things.

        :param event: optional event from traits observe decorator
        """
        self._plot.init_crosshair()
        self._plot.init_ray_info()
        self._plot.init_status_info()
        self._plot.init_keyboard_shortcuts()

        self._plot.plot_orientation_axes()
        self.replot()
        self._plot.set_initial_camera()  # this needs to be called after replot, which defines the visual scope
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
    
    @observe('hide_labels', dispatch="ui")
    def _change_hide_labels(self, event=None) -> None:
        """
        Hide or show object labels.

        :param event: optional event from traits observe decorator
        """
        if not self._status["InitScene"]:
            self._status["Drawing"] += 1
            self._plot.change_label_visibility()
            self._status["Drawing"] -= 1

    @observe('vertical_labels', dispatch="ui")
    def _change_label_orientation(self, event=None) -> None:
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

    @observe('_command_window_button', dispatch="ui")
    def open_command_window(self, event=None) -> None:
        """
        Open the command window for executing code.
        
        :param event: optional event from traits observe decorator
        """
        if self._command_window_view is None or self._command_window_view.destroyed:
            self._command_window_view = self.command_window.edit_traits()
        else:
            self._command_window_view.control.window().raise_()
            
        # code editor does not work correctly with dark mode, so we need to change the design
        bgcolor = "#aaa" if global_options.ui_dark_mode else "white"
        self._command_window_view.control.window().\
                setStyleSheet(f"QAbstractScrollArea{{font-family: monospace; background-color: {bgcolor}; color: black}}")

    @observe('_property_browser_button', dispatch="ui")
    def open_property_browser(self, event=None) -> None:
        """
        Open a property browser for the gui, the scene, the raytracer, shown rays and cardinal points.

        :param event: optional event from traits observe decorator
        """

        if self._property_browser_view is None or self._property_browser_view.destroyed:
            self._property_browser_view = self.property_browser.edit_traits()
            self.property_browser.update_dict()
        else:
            self._property_browser_view.control.window().raise_()

    # rename
    def run_command(self, cmd: str) -> None:
        """
        send/execute a command
        """
            
        if self.busy:
            warning("Other actions running, try again when the program is idle.")
            return

        self._sequential = True
        self._status["RunningCommand"] += 1

        # make this dict also available to control()
        dict_ = dict(mlab=self.scene.mlab, engine=self.scene.engine, scene=self.scene,
                     camera=self.scene.camera, GUI=self, RT=self.raytracer, LL=self.raytracer.lenses, 
                     FL=self.raytracer.filters, APL=self.raytracer.apertures, RSL=self.raytracer.ray_sources,
                     DL=self.raytracer.detectors, ML=self.raytracer.markers, VL=self.raytracer.volumes)

        with self.smart_replot(self.command_window.automatic_replot):
            with self._try():
                exec(cmd, locals() | dict_, globals())

        self._status["RunningCommand"] -= 1
        self._sequential = False

    # rescale axes texts when the window was resized
    # for some reasons these are the only ones having no text_scaling_mode = 'none' option
    # this is the only trait so far that does not work with @observe, like:
    # @observe("scene:scene_editor:interactor:size:items:value")
    @observe("scene:scene_editor:busy", dispatch="ui")
    def _resize_scene_elements(self, event=None) -> None:
        """
        Handles GUI window size changes. Fixes incorrect scaling by mayavi.

        :param event: event from traits observe decorator
        """
        self._plot.resize_scene_elements()
