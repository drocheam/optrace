
# presets
from . import presets

# tracer classes
from .refraction_index import RefractionIndex
from .ray_storage import RaySource
from .r_image import RImage
from .raytracer import Raytracer
from .transfer_matrix_analysis import TMA

# spectra
from .spectrum.spectrum import Spectrum
from .spectrum.light_spectrum import LightSpectrum
from .spectrum.transmission_spectrum import TransmissionSpectrum

# surfaces
from .geometry.surface.data_surface_1d import DataSurface1D
from .geometry.surface.data_surface_2d import DataSurface2D
from .geometry.surface.function_surface_2d import FunctionSurface2D
from .geometry.surface.function_surface_1d import FunctionSurface1D
from .geometry.surface.circular_surface import CircularSurface
from .geometry.surface.rectangular_surface import RectangularSurface
from .geometry.surface.ring_surface import RingSurface
from .geometry.surface.conic_surface import ConicSurface
from .geometry.surface.tilted_surface import TiltedSurface
from .geometry.surface.spherical_surface import SphericalSurface
from .geometry.surface.aspheric_surface import AsphericSurface

# misc base shapes
from .geometry.point import Point
from .geometry.line import Line

# elements
from .geometry.lens import Lens
from .geometry.ideal_lens import IdealLens
from .geometry.aperture import Aperture
from .geometry.filter import Filter
from .geometry.detector import Detector

# markers
from .geometry.marker.line_marker import LineMarker
from .geometry.marker.point_marker import PointMarker

# volumes
from .geometry.volume.sphere_volume import SphereVolume
from .geometry.volume.box_volume import BoxVolume
from .geometry.volume.cylinder_volume import CylinderVolume
from .geometry.volume.volume import Volume

# misc geometry
from .geometry.group import Group

# tools
from . import load
from .color.tools import WL_BOUNDS
from .convolve import convolve
from . import color
