
from optrace.tracer.spectrum.Spectrum import *
from optrace.tracer.spectrum.LightSpectrum import *

import optrace.tracer.Color as Color
import optrace.tracer.presets.Spectrum as presets_spectrum

class TransmissionSpectrum(Spectrum):

    # don't allow all types of class Spectrum, especially not types "Lines" and "Monochromatic"
    spectrum_types = ["Constant", "Data", "Rectangle", "Gaussian", "Function"]

    quantity = "Transmission T"
    unit = ""

    def getXYZ(self):

        # illuminate the filter with daylight, get color of resulting spectrum
        func1 = lambda wl: Color.Illuminant(wl, "D65") * self(wl)
        lspec = LightSpectrum("Function", func=func1)

        return lspec.getXYZ()

    def getColor(self):

        XYZ = self.getXYZ()
        Y0 = presets_spectrum.preset_spec_D65.getXYZ()[0, 0, 1]

        # 1 - Yc/Y0 is the ratio of visble ambient light coming through the filter
        # gamma correct for non-linear human vision
        alpha = (1 - XYZ[0, 0, 1]/Y0) ** (1/2.2)
        XYZ /= Y0

        RGB = Color.XYZ_to_sRGB(XYZ)[0, 0]

        return RGB[0], RGB[1], RGB[2], alpha

