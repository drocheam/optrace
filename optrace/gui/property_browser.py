from __future__ import annotations
from typing import Any
import numpy as np

from traitsui.api import View, Item, ValueEditor, Group
from traits.api import HasTraits, observe, Button, Dict, Str
from traits.observation.api import TraitChangeEvent

from ..tracer.presets import spectral_lines as spec_lines
from ..tracer.base_class import BaseClass  # BaseClass type

from ..tracer import presets as otp


class PropertyBrowser(HasTraits):

    _update_button: Button = Button(label="Update", desc="updates the browser")

    _unit_label:   Str = Str("Distances in mm, optical powers in dpt")
    _legend_title: Str = Str("Legend")

    _ray_legend: Str = Str("p:      position                      s:    unity direction vector           "
                           "s_un:   direction vector\n"
                           "l:      ray length to next point      ol:   optical length to next point     "
                           "pol:    polarization unity vector\n"
                           "w:      power                         wv:   wavelength                       "
                           "snum:   source number\n"
                           "index:  ray index                     n:    ambient refractive index")

    _tma_legend: Str = Str("abcd:  ABCD Matrix                bfl:  back focal length       d:  thickness\n"
                           "efl:   effective focal length     ffl:  front focal length\n"
                           "n1:    index before setup         n2:   index after setup")


    _scene_dict:     Dict = Dict()
    _trace_gui_dict: Dict = Dict()
    _raytracer_dict: Dict = Dict()
    _ray_dict:       Dict = Dict()
    _card_dict:      Dict = Dict()
    _preset_dict:    Dict = Dict()

    view = View(
                Item('_update_button', label=" Update Dictionaries"),
                Group(
                    Group(
                        Item('_ray_dict', editor=ValueEditor(), show_label=False),
                        Item("_legend_title", show_label=False, style="readonly", emphasized=True),
                        Item("_ray_legend", show_label=False, style="readonly", style_sheet="*{font-family: monospace}"),
                        label="Shown Rays",
                    ),
                    Group(
                        Item("_unit_label", show_label=False, style="readonly"),
                        Item('_card_dict', editor=ValueEditor(), show_label=False),
                        Item("_legend_title", show_label=False, style="readonly", emphasized=True),
                        Item("_tma_legend", show_label=False, style="readonly", style_sheet="*{font-family: monospace}"),
                        label="Cardinal Points",
                    ),
                    Group(
                        Item('_raytracer_dict', editor=ValueEditor(), show_label=False),
                        label="Raytracer",
                    ),
                    Group(
                        Item('_preset_dict', editor=ValueEditor(), show_label=False),
                        label="Presets",
                    ),
                    Group(
                        Item('_trace_gui_dict', editor=ValueEditor(), show_label=False),
                        label="TraceGUI",
                    ),
                    Group(
                        Item('_scene_dict', editor=ValueEditor(), show_label=False),
                        label="TraceGUI.scene",
                    ),
                    layout="tabbed",
                ),
                resizable=True,
                width=800,
                height=800,
                title="Property Browser")

    def __init__(self, gui: TraceGUI) -> None:
        """
        Initialize a PropertyBrowser

        :param gui: reference to the TraceGUI
        """
        self.gui = gui
        self.scene = gui.scene
        self.raytracer = gui.raytracer

        super().__init__()

    @observe('_update_button')
    def update_dict(self, event: TraitChangeEvent = None) -> None:
        """
        Update the dictionaries in all ValueEditor()

        :param event: optional trait change event
        """
        self._raytracer_dict = self._gen_dict_repr(self.raytracer.__dict__)
        self._ray_dict = self._gen_dict_repr(self.gui._plot._ray_property_dict)
        self._scene_dict = self._gen_dict_repr(self.scene.trait_get())
        self._trace_gui_dict = self._gen_dict_repr(self.gui.trait_get())
        self._card_dict = self._gen_dict_repr(self._gen_cardinals())
        self._preset_dict = self._gen_dict_repr(self._gen_pdict())

    def _gen_dict_repr(self, val: Any) -> dict:
        """generate a dictionary containing representable elements for ValueEditor()"""

        # some elements are not copyable or should not be copied or the application will hang
        # int64 and float32 are shown as hexadecimal values in the ValueEditor,
        # convert to float64 to show them correctly

        if isinstance(val, None | bool | int | float | str | BaseClass):
            return val

        elif isinstance(val, np.ndarray):
            # unpack arrays with only one element
            if val.size == 1:
                return self._gen_dict_repr(val[()])

            # force convert to float, but only if size is not gigantic
            return np.array(val, dtype=np.float64) if val.size < 1e5 else val

        elif isinstance(val, list):
            return [self._gen_dict_repr(el) for el in val]

        elif isinstance(val, tuple):
            return tuple([self._gen_dict_repr(el) for el in val])

        elif isinstance(val, dict):
            return {key: self._gen_dict_repr(val_) for key, val_ in val.items()}

        else:
            return str(val)

    def _gen_pdict(self) -> dict:
        """generate a dictionary of optrace presets"""

        pdict = {"presets.image": otp.image.__dict__,
                 "presets.light_spectrum": otp.light_spectrum.__dict__,
                 "presets.refraction_index": otp.refraction_index.__dict__,
                 "presets.spectrum": otp.spectrum.__dict__,
                 "presets.psf": otp.psf.__dict__,
                 "presets.spectral_lines": otp.spectral_lines.__dict__,}

        return {key0:  {key: val for key, val in val0.items() if not key.startswith("__") and\
                        not "module" in str(val) and not "class" in str(val)}\
                for key0, val0 in pdict.items()}

    def _gen_cardinals(self) -> dict:
        """generate a dictionary of cardinal points / ray transfer matrix analysis"""

        # get properties for lens
        def set_cdict(w, cdict, name):
            cdict[name] = dict()

            # get properties for wavelengths
            for wl in spec_lines.FdC:
                tma = w.tma(wl=wl)
                cdict[name][f"{wl:.4g}nm"] = \
                    dict(nodal_points=tma.nodal_points, d=tma.d, n1=tma.n1, n2=tma.n2,
                         focal_points=tma.focal_points, focal_lengths=tma.focal_lengths,
                         focal_lengths_n=tma.focal_lengths_n, principal_points=tma.principal_points,
                         vertex_points=tma.vertex_points, abcd=tma.abcd, efl=tma.efl, efl_n=tma.efl_n,
                         powers=tma.powers, powers_n=tma.powers_n, bfl=tma.bfl, ffl=tma.ffl,
                         optical_center=tma.optical_center)

        try:
            cdict = dict()
        
            # overall system
            set_cdict(self.raytracer, cdict, "System")

            # each lens
            for i, L in enumerate(self.raytracer.lenses):
                set_cdict(L, cdict, f"Lens {i}")

            return cdict

        # throws when geometry is invalid or no rotational symmetry
        except Exception as e:
            return dict(exception=e)

