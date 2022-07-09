
import pathlib  # loading files in relative path

# directory of this source file
this_dir = pathlib.Path(__file__).resolve().parent


# https://commons.wikimedia.org/wiki/File:X-rite_color_checker,_SahiFa_Braunschweig,_AP3Q0026_edit.jpg
# Permission: Public Domain
preset_image_color_checker: str = str(this_dir / "images" / "ColorChecker.jpg")
"""Color checker chart"""

# https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg
# Permission: Public Domain
preset_image_test_screen: str = str(this_dir / "images" / "TestScreen_square.png")
"""TV test screen"""

# https://commons.wikimedia.org/wiki/File:EIA_Resolution_Chart_1956.svg
# Permission: Public Domain
preset_image_resolution_chart: str = str(this_dir / "images" / "EIA_Resolution_Chart_1956.png")
"""EIA 1956 resolution chart"""

# https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg
# Permission: Public Domain
preset_image_ETDRS_chart: str = str(this_dir / "images" / "ETDRS_Chart.png")
"""ETDRS Chart standard"""
preset_image_ETDRS_chart_inverted: str = str(this_dir / "images" / "ETDRS_Chart_inverted.png")
"""ETDRS Chart inverted"""


presets_image: list = [preset_image_color_checker, preset_image_test_screen, preset_image_resolution_chart,
                       preset_image_ETDRS_chart, preset_image_ETDRS_chart_inverted]
"""list with all Image presets"""

