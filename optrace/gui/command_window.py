from __future__ import annotations
from typing import Any

from traitsui.api import View, Item, ValueEditor, Group, CodeEditor, ListStrEditor, CheckListEditor, HSplit
from traits.api import HasTraits, observe, Button, Dict, Str, List, Enum
from pyface.qt import QtGui  # copying to clipboard 
from traits.observation.api import TraitChangeEvent

from ..warnings import warning



class CommandWindow(HasTraits):

    cmd:                 Str = Str()  #: command to run
    history:             List = List()  #: command history

    _execute_label:       Str = Str('Command:')
    _history_label:       Str = Str('History:')
    _whitespace_label:    Str = Str()

    _run_button:          Button = Button(label="Run", desc="runs the specified command")
    _replot_button:       Button = Button(label="Replot/Retrace", desc="replots and retraces the scene")
    _clear_button:        Button = Button(label="Clear History", desc="clear command history")
    _clipboard_button:    Button = Button(label="Copy History to Clipboard",
                                          desc="copies the full history to the clipboard")
    
    automatic_replot: List = List(editor=CheckListEditor(values=['Replot and retrace automatically'], 
                                                         format_func=lambda x: x),
                                   desc="if the scene is automatically reploted and traced on changes")
    """Option for automatic replotting/recalculating the geometry after running the command"""

    # code_editor = 
    view = View(
                HSplit(
                    Group(
                        Group(
                            Item("_execute_label", style='readonly', show_label=False, emphasized=True),
                            Item('cmd', editor=CodeEditor(), show_label=False, style="custom"),
                            ),
                        Item("_whitespace_label", style='readonly', show_label=False, width=563),
                        Group(
                            Item("_history_label", style='readonly', show_label=False, emphasized=True),
                            Item("history", editor=ListStrEditor(horizontal_lines=True), show_label=False, height=220),
                            ),
                        ),
                    Group(
                        Group(
                            Item("_whitespace_label", style='readonly', show_label=False, width=10),
                            Item("automatic_replot", style="custom", show_label=False),
                            Item("_run_button", show_label=False),
                            Item("_replot_button", show_label=False),
                            Item("_whitespace_label", style='readonly', show_label=False, width=10, height=358),
                            Item("_clear_button", show_label=False),
                            Item("_clipboard_button", show_label=False),
                        ),
                    ),
                ),
                resizable=True,
                width=800,
                height=700,
                title="Command Window")
    
    def __init__(self, gui: TraceGUI) -> None:
        """
        Initialize the command window

        :param gui: parent TraceGUI
        """
        self.gui = gui
        self.automatic_replot = True
        super().__init__()

    def __setattr__(self, key: str, val: Any) -> None:
        """
        assigns the value of an attribute
        :param key: attribute name
        :param val: value to assign
        """
        # workaround so we can set bool values to some List settings
        if isinstance(val, bool) and key == "automatic_replot":
            val = [self._trait(key, 0).editor.values[0]] if val else []

        super().__setattr__(key, val)

    @observe('_clear_button', dispatch="ui")
    def clear_history(self, event: TraitChangeEvent = None) -> None:
        """
        clear command history
        :param event: optional trait change event
        """
        self.history = []
    
    @observe('_clipboard_button', dispatch="ui")
    def copy_history(self, event: TraitChangeEvent = None) -> None:
        """
        copy the full history to the clipboard
        :param event: optional trait change event
        """
        output = ""
        for el in self.history:
            output += el + "\n"

        clipboard = QtGui.QApplication.clipboard()
        clipboard.setText(output, mode=QtGui.QClipboard.Clipboard)
 
        # check if actually copied
        if clipboard.text(mode=QtGui.QClipboard.Clipboard) != output:  
            # can't test these, because it seems to fail only on Windows
            warning("Copying to clipboard failed. This can be an library or system issue.\n")  # pragma: no cover
    
    @observe('_replot_button', dispatch="ui")
    def _replot(self, event: TraitChangeEvent = None) -> None:
        """
        Replots the TraceGUI

        :param event: optional trait change event
        """
        self.gui.replot()

    @observe('_run_button', dispatch="ui")
    def send_command(self, event: TraitChangeEvent = None) -> None:
        """
        Execute the entered command in the TraceGUI

        :param event: optional trait change event
        """
        if self.cmd:
            self.gui.run_command(self.cmd)

            # add to history if something happened and if the command differs from the last one
            if not self.history or self.cmd != self.history[-1]:
                self.history = self.history + [self.cmd]

