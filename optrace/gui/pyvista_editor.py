
from pyface.qt import QtGui
from traits.api import Instance
from traitsui.api import Editor, BasicEditorFactory

from pyvistaqt import QtInteractor


class _PyVistaEditor(Editor):
    """
    Qt-based editor that embeds PyVistaQt.
    """
    def init(self, parent):
        """
        Create the QtInteractor control and link it to the TraitsUI system.
        """
        # Create the PyVistaQt widget
        self.control = QtInteractor(parent.parentWidget(), auto_update=False)
        
        # Assign the created plotter to the Trait on the TraceGUI instance
        self.value = self.control

        # close all windows if the scene gets destroyed
        self.control.destroyed.connect(QtGui.QApplication.closeAllWindows)

    def set_size_policy(self, *args):
        pass

class PyVistaEditor(BasicEditorFactory):
    """
    The Factory class used in the View definition: editor=PyVistaEditor()
    """
    klass = _PyVistaEditor

SceneInstance = Instance(QtInteractor)
