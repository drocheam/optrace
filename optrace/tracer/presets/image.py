import pathlib  # loading files in relative path
import numpy as np  # np.ndarray type
import scipy.misc  # for a racoon face amd the "ascent" image


# path of the image library folder
image_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "ressources" / "images"


ascent: np.ndarray = np.repeat(np.array(scipy.misc.ascent(), dtype=np.float64)[:, :, np.newaxis] / 255, 3, axis=2)
"""Ascent Image from scipy.misc, 
see https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.face.html#scipy.misc.ascent"""


bacteria: str = str(image_dir / "bacteria.png")
"""Colored rings on white background. Own creation."""


checkerboard: np.ndarray = np.zeros((8, 8, 3), dtype=np.float64)
"""black and white checkerboard pattern image with 8 boxes in each dimension"""
checkerboard[::2, 1::2] = 1.0
checkerboard[1::2, ::2] = 1.0


color_checker: str = str(image_dir / "ColorChecker.jpg")
"""Color checker chart
Public domain image from
https://commons.wikimedia.org/wiki/File:X-rite_color_checker,_SahiFa_Braunschweig,_AP3Q0026_edit.jpg """


ETDRS_chart: str = str(image_dir / "ETDRS_Chart.png")
"""ETDRS Chart standard
Public Domain Image from https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg """


ETDRS_chart_inverted: str = str(image_dir / "ETDRS_Chart_inverted.png")
"""ETDRS Chart inverted
edited version of Public Domain Image from https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg """


racoon: np.ndarray = np.array(scipy.misc.face(), dtype=np.float64) / 255
"""Racoon Image from scipy.misc, 
see https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.face.html#scipy.misc.face"""


resolution_chart: str = str(image_dir / "EIA_Resolution_Chart_1956.png")
"""EIA 1956 resolution chart
Public Domain image from https://commons.wikimedia.org/wiki/File:EIA_Resolution_Chart_1956.svg """


test_screen: str = str(image_dir / "TestScreen_square.png")
"""TV test screen
Public Domain Image from  https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg """


all_presets: list = [ascent, bacteria, checkerboard, color_checker, ETDRS_chart, ETDRS_chart_inverted, racoon,
                     resolution_chart, test_screen]
"""list with all Image presets"""
