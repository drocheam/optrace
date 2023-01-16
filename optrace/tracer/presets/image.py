import pathlib  # loading files in relative path
import numpy as np  # np.ndarray type


# path of the image library folder
image_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "ressources" / "images"


# Scene Images
#######################################################################################################################

cell: str = str(image_dir / "cell.webp")
"""Stable Diffusion image from https://lexica.art/prompt/960d8351-f474-4cc0-b84b-4e9521754064"""

group_photo: str = str(image_dir / "group_photo.jpg")
"""Stable Diffusion image from https://lexica.art/prompt/06ba5ac6-7bfd-4ce6-8002-9d0e487b36b2"""

interior: str = str(image_dir / "interior.jpg")
"""Stable Diffusion image from https://lexica.art/prompt/44d7e1fe-ba3b-4e73-972c-a30b95897434"""

landscape: str = str(image_dir / "landscape.jpg")
"""Stable Diffusion image from https://lexica.art/prompt/0da3a592-465e-46d6-8ee6-dfe17ddea386"""

scenes: list = [cell, group_photo, interior, landscape]
"""photography-like images for viewing natural scenes"""


# Test Images
#######################################################################################################################

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

test_screen: str = str(image_dir / "TestScreen_square.png")
"""TV test screen
Public Domain Image from  https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg """

test_images: list = [checkerboard, color_checker, ETDRS_chart, ETDRS_chart_inverted, test_screen]
"""test images for color, resolution or distortion"""


# All Images
#######################################################################################################################

all_presets: list = [*test_images, *scenes]
"""list with all Image presets"""
