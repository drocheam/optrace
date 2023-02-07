
from traitsui.api import View, Item, ValueEditor, Group, CodeEditor, ListStrEditor
from traits.api import HasTraits, observe, Button, Dict, Str, List
from pyface.qt import QtGui  # copying to clipboard 


class CommandWindow(HasTraits):

    _cmd:                 Str = Str()
    _execute_label:       Str = Str('Command:')
    _history_label:       Str = Str('History:')
    _whitespace_label:    Str = Str()
    _status:              Str = Str()

    _history:             List = List()
    
    _run_button:          Button = Button(label="Run", desc="runs the specified command")
    _clear_button:        Button = Button(label="Clear", desc="clear command history")
    _clipboard_button:    Button = Button(label="Copy History to Clipboard", desc="copies the full history to the clipboard")
    

    view = View(
                Group(
                    Group(
                        Item("_execute_label", style='readonly', show_label=False, emphasized=True),
                        # custom stylesheet because colored or dark themes mess up the syntax highlighting
                        # of the code editor
                        Item('_cmd', editor=CodeEditor(), show_label=False, style="custom", 
                             style_sheet="*{background-color: white; color: black}"), 
                        Item("_run_button", show_label=False),
                    ),
                    Item("_whitespace_label", style='readonly', show_label=False, width=210),
                    Group(
                        Item("_history_label", style='readonly', show_label=False, emphasized=True),
                        Item("_history", editor=ListStrEditor(horizontal_lines=True), show_label=False,
                             height=220, style_sheet="*{font-family: monospace}"),
                        Item("_clear_button", show_label=False),
                        Item("_clipboard_button", show_label=False),
                    ),
                    Item("_status", style="readonly", show_label=False),
                    ),
                resizable=True,
                width=700,
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
        self._history = []
    
    @observe('_clipboard_button')
    def copy_history(self, event=None) -> None:
        """
        copy the full history to the clipboard
        :param event: optional event from traits observe decorator
        """
        output = ""
        for el in self._history:
            output += el + "\n"

        clipboard = QtGui.QApplication.clipboard()
        clipboard.clear()  # needed to work on Windows?
        clipboard.setText(output, mode=clipboard.Clipboard)
    
    @observe('_run_button')
    def send_cmd(self, event=None) -> None:
        """
        :param event: optional event from traits observe decorator
        """
        if self._cmd:
            ret = self.gui.send_cmd(self._cmd)

            # add to history if something happened and if the command differs from the last one
            if ret and (not self._history or self._cmd != self._history[-1]):
                self._history = self._history + [self._cmd]

