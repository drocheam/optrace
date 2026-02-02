
# enforce qt backend  
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt'

# import os 
# os.environ["QT_API"] = "pyside6"

from .global_options import global_options as global_options
from .tracer.__init__ import *

from . import metadata

from .warnings import OptraceWarning

