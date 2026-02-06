
import sys
import signal

def signal_handler(sig, frame):
    print()
    sys.exit(0)

# global SIGINT handler
# required so pyplot only scripts (such as IOL_imaging_example) don't raise KeyboardInterrupts
signal.signal(signal.SIGINT, signal_handler)

# enforce qt backend  
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt'

from .global_options import global_options as global_options
from .tracer.__init__ import *

from . import metadata

from .warnings import OptraceWarning

