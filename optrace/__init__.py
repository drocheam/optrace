
import warnings

def simplified_warning(message, category, filename, lineno, file=None, line=None):
    return "Warning: " + str(message) + "\n" 

warnings.formatwarning = simplified_warning


from .__metadata__ import *

from . import global_options as global_options

from .tracer.__init__ import *

