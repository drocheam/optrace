import matplotlib

# set font globally
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# increase dpi if in inline backend, otherwise the resolution is way too low in spyder etc.
if "inline" in matplotlib.rcParams["backend"]:
    matplotlib.rcParams["figure.dpi"] = 200

from .r_image_plots import r_image_plot, r_image_cut_plot
from .misc_plots import autofocus_cost_plot, abbe_plot, surface_profile_plot, convolve_debug_plots, image_plot
from .spectrum_plots import spectrum_plot, refraction_index_plot
from .chromaticity_plots import chromaticity_norms, chromaticities_cie_1931, chromaticities_cie_1976
