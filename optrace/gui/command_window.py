
from traitsui.api import View, Item, ValueEditor, Group, CodeEditor, ListStrEditor
from traits.api import HasTraits, observe, Button, Dict, Str, List
from pyface.qt import QtGui  # copying to clipboard 


class CommandWindow(HasTraits):

    _cmd:                 Str = Str()
    _history:             List = List()

    _execute_label:       Str = Str('Command:')
    _history_label:       Str = Str('History:')
    _whitespace_label:    Str = Str()

    _run_button:          Button = Button(label="Run", desc="runs the specified command")
    _clear_button:        Button = Button(label="Clear", desc="clear command history")
    _clipboard_button:    Button = Button(label="Copy History to Clipboard",
                                          desc="copies the full history to the clipboard")

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
                    ),
                resizable=True,
                width=700,
                height=800,
                title="Command Window")

    def __init__(self, gui, silent: bool = False) -> None:
        """
        Initialize the command window from a gui

        :param gui: parent TraceGUI
        :param silent: if standard output should be omitted
        """
        self.gui = gui
        self.silent = silent
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
        clipboard.clear(mode=clipboard.Clipboard)  
        clipboard.setText(output, mode=clipboard.Clipboard)
 
        # check if actually copied
        if clipboard.text(mode=clipboard.Clipboard) != output:  
            # can't test these, because it seems to fail only on Windows
            if not self.silent:  # pragma: no cover
                print(output + "\n\n")  # pragma: no cover
                print("Copying to clipboard failed. This can be an library or system issue.\n"  # pragma: no cover
                      "The history was instead output to the terminal, you can copy it from there.")  # pragma: no cover

    @observe('_run_button')
    def send_cmd(self, event=None) -> None:
        """
        Execute a command in the TraceGUI

        :param event: optional event from traits observe decorator
        """
        if self._cmd:
            ret = self.gui.send_cmd(self._cmd)

            # add to history if something happened and if the command differs from the last one
            if ret and (not self._history or self._cmd != self._history[-1]):
                self._history = self._history + [self._cmd]
