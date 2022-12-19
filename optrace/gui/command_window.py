
from typing import Any  # Any type

from traitsui.api import View, Item, ValueEditor, Group, CodeEditor
from traits.api import HasTraits, observe, Button, Dict, Str


class CommandWindow(HasTraits):

    _cmd:                        Str = Str()
    _execute_label:              Str = Str('Command:')
    _history_label:              Str = Str('History:')
    _command_history:            Str = Str('')
    _whitespace_label:           Str = Str('')
    _status:                     Str = Str('')
    
    _run_button:                 Button = Button(label="Run", desc="runs the specified command")
    _clear_button:               Button = Button(label="Clear", desc="clear command history")
    

    view = View(
                Group(
                    Group(
                        Item("_execute_label", style='readonly', show_label=False, emphasized=True),
                        Item('_cmd', editor=CodeEditor(), show_label=False, style="custom"), 
                        Item("_run_button", show_label=False),
                        style_sheet="*{max-height:200px}"
                    ),
                    Item("_whitespace_label", style='readonly', show_label=False, width=210),
                    Group(
                        Item("_history_label", style='readonly', show_label=False, emphasized=True),
                        Item("_command_history", editor=CodeEditor(), show_label=False, style="custom"),
                        Item("_clear_button", show_label=False),
                        style_sheet="*{max-height:600px}"
                    ),
                    Item("_status", style="readonly", show_label=False)
                    ),
                resizable=True,
                width=650,
                height=800,
                title="Command Window")

    def __init__(self, gui) -> None:
        """

        :param gui:
        """
        self.gui = gui
        super().__init__()

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
        :param event: optional event from traits observe decorator
        """
        succ = self.gui.send_cmd(self._cmd)

        if succ:
            self._command_history += ("\n" if self._command_history else "") + self._cmd

