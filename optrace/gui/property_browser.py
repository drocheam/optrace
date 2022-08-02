
from traitsui.api import View, Item, ValueEditor, Group
from traits.api import HasTraits, observe, Button, Dict
import numpy as np  # numpy ndarray type


class PropertyBrowser(HasTraits):

    UpdateButton: Button = Button(label="Update", desc="updates the browser")

    Scene_dict = Dict()
    TraceGUI_dict = Dict()
    Raytracer_dict = Dict()
    Ray_dict = Dict()

    view = View(
                Group(
                    Group(
                        Item('TraceGUI_dict', editor=ValueEditor(), width=900, height=700, show_label=False),
                        label="TraceGUI",
                    ),
                    Group(
                        Item('Scene_dict', editor=ValueEditor(), width=900, height=700, show_label=False),
                        label="TraceGUI.Scene",
                    ),
                    Group(
                        Item('Raytracer_dict', editor=ValueEditor(), width=900, height=700, show_label=False),
                        label="Raytracer",
                    ),
                    Group(
                        Item('Ray_dict', editor=ValueEditor(), width=900, height=700, show_label=False),
                        label="Shown Rays",
                    ),
                    layout="tabbed",
                ),
                Item('UpdateButton', show_label=False),
                resizable=True, 
                title="Property Browser")

    def __init__(self, gui, scene, raytracer, rays):
        self.gui = gui
        self.scene = scene
        self.raytracer = raytracer
        self.rays = rays

        super().__init__()
        self.update_dict()
    
    @observe('UpdateButton')
    def update_dict(self, event=None):

        self.Scene_dict = {}
        self.TraceGUI_dict = {}
        self.Raytracer_dict = {}
        self.Ray_dict = {}
       
        # some elements are not copyable or should not be copied or the application will hang
        # also convert numpy array formats       
        # int64 and float32 are shown as hexadecimal values in the ValueEditor,
        # convert to float64 to show them correctly
        def repr_(val):
            
            if isinstance(val, None | int | float | str):    return val 
            elif isinstance(val, np.ndarray):                return np.array(val, dtype=np.float64)
            elif isinstance(val, list):                      return [repr_(el) for el in val]
            elif isinstance(val, tuple):                     return tuple([repr_(el) for el in val])
            elif isinstance(val, dict):                      return {key: repr_(val_) for key, val_ in val.items()}
            else:                                            return str(val)
        
        self.Raytracer_dict = self.raytracer.__dict__
        self.Ray_dict = repr_(self.rays)  # only call so array values get converted
        self.Scene_dict = repr_(self.scene.trait_get())
        self.TraceGUI_dict = repr_(self.gui.trait_get())

    # filtering like https://stackoverflow.com/a/15024168
    # would be nice, but some dictionary elements can't be copied
    # (or can't be copied without creating new UI elements
