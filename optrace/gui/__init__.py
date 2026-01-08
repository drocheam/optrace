# fix for MacOS Qt issue
# see https://gitlab.kitware.com/vtk/vtk/-/issues/19841
import vtkmodules.qt
vtkmodules.qt.QVTKRWIBase = "QOpenGLWidget"

from .trace_gui import TraceGUI
