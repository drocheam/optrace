import pathlib  # loading files in relative path

# path of the image library folder
image_dir = pathlib.Path(__file__).resolve().parent / "images"


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


resolution_chart: str = str(image_dir / "EIA_Resolution_Chart_1956.png")
"""EIA 1956 resolution chart
Public Domain image from https://commons.wikimedia.org/wiki/File:EIA_Resolution_Chart_1956.svg """


test_screen: str = str(image_dir / "TestScreen_square.png")
"""TV test screen
Public Domain Image from  https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg """


all_presets: list = [color_checker, ETDRS_chart, ETDRS_chart_inverted,
                     resolution_chart, test_screen]
"""list with all Image presets"""
