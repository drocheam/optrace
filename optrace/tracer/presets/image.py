import pathlib  # loading files in relative path
import numpy as np  # np.ndarray type


# path of the image library folder
image_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "ressources" / "images"


# Scene Images
#######################################################################################################################

cell: str = str(image_dir / "cell.webp")
"""Stable Diffusion image from https://lexica.art/prompt/960d8351-f474-4cc0-b84b-4e9521754064"""

documents: str = str(image_dir / "documents.webp")
"""Photo of a keyboard and documents on a desk. Source: https://www.pexels.com/photo/documents-on-wooden-surface-95916/ """

fruits: str = str(image_dir / "fruits.webp")
"""Photo of different fruits on a tray. Source: https://www.pexels.com/photo/sliced-fruits-on-tray-1132047/ """

group_photo: str = str(image_dir / "group_photo.webp")
"""Photo of a group of people in front of a blackboard. Source: https://www.pexels.com/photo/photo-of-people-standing-near-blackboard-3184393/ """

hong_kong: str = str(image_dir / "hong_kong.webp")
"""Photo of a Hong Kong street at night. Source: https://www.pexels.com/photo/cars-on-street-during-night-time-3158562/ """

interior: str = str(image_dir / "interior.webp")
"""Green sofa in an interior room. Source: https://www.pexels.com/photo/green-2-seat-sofa-1918291/ """

landscape: str = str(image_dir / "landscape.webp")
"""Landscape image of a mountain and water scene. Source: https://www.pexels.com/photo/green-island-in-the-middle-of-the-lake-during-daytime-724963/ """

scenes: list = [cell, documents, fruits, group_photo, hong_kong, interior, landscape]
"""photography-like images for viewing natural scenes"""


# Test Images
#######################################################################################################################

checkerboard: np.ndarray = np.zeros((8, 8, 3), dtype=np.float64)
"""black and white checkerboard pattern image with 8 boxes in each dimension"""
checkerboard[::2, 1::2] = 1.0
checkerboard[1::2, ::2] = 1.0

color_checker: str = str(image_dir / "color_checker.webp")
"""Color checker chart
Public domain image from
https://commons.wikimedia.org/wiki/File:X-rite_color_checker,_SahiFa_Braunschweig,_AP3Q0026_edit.jpg """

ETDRS_chart: str = str(image_dir / "ETDRS_chart.png")
"""ETDRS Chart standard
Public Domain Image from https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg """

ETDRS_chart_inverted: str = str(image_dir / "ETDRS_chart_inverted.png")
"""ETDRS Chart inverted
edited version of Public Domain Image from https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg """

eye_test_vintage: str = str(image_dir / "eye_test_vintage.webp")
"""Photo of a vintage eye test chart. Source: https://www.publicdomainpictures.net/en/view-image.php?image=284944&picture=eye-test-chart-vintage """

test_screen: str = str(image_dir / "test_screen.png")
"""TV test screen
Public Domain Image from  https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg """

test_images: list = [checkerboard, color_checker, ETDRS_chart, ETDRS_chart_inverted, eye_test_vintage, test_screen]
"""test images for color, resolution or distortion"""


# All Images
#######################################################################################################################

all_presets: list = [*test_images, *scenes]
"""list with all Image presets"""
