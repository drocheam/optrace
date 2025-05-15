import pathlib  # loading files in relative path
import numpy as np  # np.ndarray | list type
from ..image import RGBImage, GrayscaleImage

# path of the image library folder
image_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "resources" / "images"

# for information on image sources and licenses see SOURCE.txt in optrace/resources/images/

# Scene Images
#######################################################################################################################

def cell(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Stable Diffusion image from https://lexica.art/prompt/960d8351-f474-4cc0-b84b-4e9521754064

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :param kwargs: additional keyword arguments for the creation of the RGBImage object
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "cell.webp"), s, extent, desc="Cell")

def documents(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Photo of a keyboard and documents on a desk. 
    Source: https://www.pexels.com/photo/documents-on-wooden-surface-95916/ 

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "documents.webp"), s, extent, desc="Documents")

def fruits(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Photo of fruits on a tray.
    Source: https://www.pexels.com/photo/sliced-fruits-on-tray-1132047/

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "fruits.webp"), s, extent, desc="Fruits")

def group_photo(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Photo of a group of people in front of a blackboard. 
    Source: https://www.pexels.com/photo/photo-of-people-standing-near-blackboard-3184393/

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "group_photo.webp"), s, extent, desc="Group Photo")

def hong_kong(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Photo of a Hong Kong street at night. 
    Source: https://www.pexels.com/photo/cars-on-street-during-night-time-3158562/

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "hong_kong.webp"), s, extent, desc="Hong Kong")

def interior(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Green sofa in an interior room. 
    Source: https://www.pexels.com/photo/green-2-seat-sofa-1918291/

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "interior.webp"), s, extent, desc="Interior")

def landscape(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Landscape image of a mountain and water scene. 
    Source: https://www.pexels.com/photo/green-island-in-the-middle-of-the-lake-during-daytime-724963/

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "landscape.webp"), s, extent, desc="Landscape")

scenes: list = [cell, documents, fruits, group_photo, hong_kong, interior, landscape]
"""photography-like images for viewing natural scenes"""


# Test Images
#######################################################################################################################

def color_checker(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Color checker chart.
    Source: https://commons.wikimedia.org/wiki/File:X-rite_color_checker,_SahiFa_Braunschweig,_AP3Q0026_edit.jpg

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "color_checker.webp"), s, extent, desc="Color Checker Chart")

def ETDRS_chart(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> GrayscaleImage:
    """
    ETDRS Chart standard. 
    Source: https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: GrayscaleImage object
    """
    return RGBImage(str(image_dir / "ETDRS_chart.png"), s, extent, desc="ETDRS Chart").to_grayscale_image()

def ETDRS_chart_inverted(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    ETDRS Chart inverted.
    Edited version of: https://commons.wikimedia.org/wiki/File:ETDRS_Chart_2.svg

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: GrayscaleImage object
    """
    return RGBImage(str(image_dir / "ETDRS_chart_inverted.png"), 
                    s, extent, desc="ETDRS Chart Inverted").to_grayscale_image()

def eye_test_vintage(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    Photo of a vintage eye test chart. 
    Source: https://www.publicdomainpictures.net/en/view-image.php?image=284944&picture=eye-test-chart-vintage

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "eye_test_vintage.webp"), s, extent, desc="Eye Test Vintage")

def grid(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> GrayscaleImage:
    """
    White grid on black background with 10x10 cells.
    Useful for distortion characterization.
    Source: Own creation.

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: GrayscaleImage object
    """
    grid = np.zeros((301, 301))
    grid[::30] = 1
    grid[:, ::30] = 1
    return GrayscaleImage(grid, s, extent, desc="Grid")

def siemens_star(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> GrayscaleImage:
    """
    Siemens Star Image.
    Source: Own creation.

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: GrayscaleImage object
    """
    # creating a Siemens Star is easy mathematically, but the Anti-Aliasing is the hard part
    # load a finished image instead
    return RGBImage(str(image_dir / "siemens_star.png"), s, extent, desc="Siemens Star").to_grayscale_image()

def tv_testcard1(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    TV test card 1.
    Source: https://commons.wikimedia.org/wiki/File:TestScreen_square_more_colors.svg

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "tv_testcard1.png"), s, extent, desc="TV Testcard 1")

def tv_testcard2(s: np.ndarray | list = None, extent: list | np.ndarray = None) -> RGBImage:
    """
    TV test card 2.
    Source: https://commons.wikimedia.org/wiki/File:Bulgarian_colour_testcard.png

    :param s: image side lengths list in mm (x length, y length)
    :param extent: image extent in the form [xs, xe, ys, ye]
    :return: RGBImage object
    """
    return RGBImage(str(image_dir / "tv_testcard2.png"), s, extent, desc="TV Testcard 2")

test_images: list = [color_checker, ETDRS_chart, ETDRS_chart_inverted, 
                     eye_test_vintage, grid, siemens_star, tv_testcard1, tv_testcard2]
"""test images for color, resolution or distortion"""


# All Images
#######################################################################################################################

all_presets: list = [*test_images, *scenes]
"""list with all Image presets"""
