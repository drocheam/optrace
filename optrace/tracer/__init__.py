"""
Backend module
"""

# load modules
from optrace.tracer.geometry.SurfaceFunction import SurfaceFunction
from optrace.tracer.geometry.Surface import Surface 
from optrace.tracer.geometry.Lens import Lens
from optrace.tracer.geometry.Aperture import Aperture
from optrace.tracer.geometry.Filter import Filter
from optrace.tracer.geometry.Detector import Detector

from optrace.tracer.RayStorage import RaySource
from optrace.tracer.RImage import RImage
from optrace.tracer.Raytracer import Raytracer

from optrace.tracer.spectrum.RefractionIndex import RefractionIndex
from optrace.tracer.spectrum.Spectrum import Spectrum
from optrace.tracer.spectrum.LightSpectrum import LightSpectrum
from optrace.tracer.spectrum.TransmissionSpectrum import TransmissionSpectrum

from optrace.tracer.presets.Lines import *
from optrace.tracer.presets.Spectrum import *
from optrace.tracer.presets.RefractionIndex import *
from optrace.tracer.presets.Image import *


