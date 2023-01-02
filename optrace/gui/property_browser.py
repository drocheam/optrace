
from typing import Any  # Any type
import numpy as np  # numpy ndarray type

from traitsui.api import View, Item, ValueEditor, Group
from traits.api import HasTraits, observe, Button, Dict, Str

from ..tracer.presets import spectral_lines as spec_lines
from ..tracer.base_class import BaseClass  # BaseClass type


class PropertyBrowser(HasTraits):

    update_button: Button = Button(label="Update", desc="updates the browser")

    unit_label:   Str = Str("Distances in mm, optical power in dpt")
    legend_title: Str = Str("Legend")

    ray_legend: Str = Str("p:      Position                      s:    unity direction vector           s_un:  direction vector\n"
                          "l:      ray length to next point      ol:   optical length to next point     pol:   polarization unity vector\n"
                          "w:      power                         wv:   wavelength                       snum:  source number\n"
                          "index:  ray index                     n:    ambient refractive index")

    tma_legend: Str = Str("abcd:  ABCD Matrix                bfl:  back focal length       d:  thickness\n"
                          "efl:   effective focal length     ffl:  front focal length\n"
                          "n1:    index before setup         n2:   index after setup")


    scene_dict:     Dict = Dict()
    trace_gui_dict: Dict = Dict()
    raytracer_dict: Dict = Dict()
    ray_dict:       Dict = Dict()
    card_dict:      Dict = Dict()

    view = View(
                Item('update_button', label=" Update Dictionaries"),
                Group(
                    Group(
                        Item('ray_dict', editor=ValueEditor(), show_label=False),
                        Item("legend_title", show_label=False, style="readonly", emphasized=True),
                        Item("ray_legend", show_label=False, style="readonly", style_sheet="*{font-family: monospace}"),
                        label="Shown Rays",
                    ),
                    Group(
                        Item("unit_label", show_label=False, style="readonly"),
                        Item('card_dict', editor=ValueEditor(), show_label=False),
                        Item("legend_title", show_label=False, style="readonly", emphasized=True),
                        Item("tma_legend", show_label=False, style="readonly", style_sheet="*{font-family: monospace}"),
                        label="Cardinal Points",
                    ),
                    Group(
                        Item('raytracer_dict', editor=ValueEditor(), show_label=False),
                        label="Raytracer",
                    ),
                    Group(
                        Item('trace_gui_dict', editor=ValueEditor(), show_label=False),
                        label="TraceGUI",
                    ),
                    Group(
                        Item('scene_dict', editor=ValueEditor(), show_label=False),
                        label="TraceGUI.scene",
                    ),
                    layout="tabbed",
                ),
                resizable=True,
                width=800,
                height=800,
                title="Property Browser")

    def __init__(self, gui, scene, raytracer) -> None:
        """

        :param gui:
        :param scene:
        :param raytracer:
        """
        self.gui = gui
        self.scene = scene
        self.raytracer = raytracer

        super().__init__()
        self.update_dict()

    @observe('update_button')
    def update_dict(self, event=None) -> None:
        """

        :param event:
        """
        # some elements are not copyable or should not be copied or the application will hang
        # int64 and float32 are shown as hexadecimal values in the ValueEditor,
        # convert to float64 to show them correctly
        def repr_(val: Any) -> Any:

            if isinstance(val, None | bool | int | float | str | BaseClass):
                return val

            elif isinstance(val, np.ndarray):
                # unpack arrays with only one element
                if val.size == 1:
                    return repr_(val[()])

                # force convert to float, but only if size is not gigantic
                return np.array(val, dtype=np.float64) if val.size < 1e5 else val

            elif isinstance(val, list):
                return [repr_(el) for el in val]

            elif isinstance(val, tuple):
                return tuple([repr_(el) for el in val])

            elif isinstance(val, dict):
                return {key: repr_(val_) for key, val_ in val.items()}

            else:
                return str(val)

        self.raytracer_dict = repr_(self.raytracer.__dict__)
        self.ray_dict = repr_(self.gui._ray_property_dict)
        self.scene_dict = repr_(self.scene.trait_get())
        self.trace_gui_dict = repr_(self.gui.trait_get())
        self.card_dict = repr_(self.gen_cardinals())

    def gen_cardinals(self) -> dict:
        """

        :return:
        """
        # get properties for lens
        def set_cdict(w, cdict, name):
            cdict[name] = dict()

            # get properties for wavelengths
            for wl in spec_lines.FdC:
                tma = w.tma(wl=wl)
                cdict[name][f"{wl:.4g}nm"] = \
                    dict(nodal_points=tma.nodal_point, d=tma.d, n1=tma.n1, n2=tma.n2,
                         focal_points=tma.focal_point, focal_lengths=tma.focal_length,
                         focal_lengths_n=tma.focal_length_n, principal_points=tma.principal_point,
                         vertex_points=tma.vertex_point, abcd=tma.abcd, efl=tma.efl, efl_n=tma.efl_n,
                         power=tma.power, power_n=tma.power_n, bfl=tma.bfl, ffl=tma.ffl)

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

