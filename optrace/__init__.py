
from .__metadata__ import *


# TODO is this the right place for this?
import warnings

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):

    return "Warning: " + str(message) + "\n" 
    # return ' %s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)

warnings.formatwarning = warning_on_one_line

from . import global_options as global_options

from .tracer.__init__ import *

