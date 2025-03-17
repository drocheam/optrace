
# enforce qt backend  
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt'

from .global_options import global_options as global_options
from .tracer.__init__ import *

from . import metadata

from .warnings import OptraceWarning

