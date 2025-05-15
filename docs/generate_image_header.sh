#!/bin/bash

# Define input image paths
IMAGE_ROW1_COL1="./docs/source/images/example_spherical_aberration2.png"
IMAGE_ROW1_COL2="./docs/source/images/example_rgb_render4.webp"
IMAGE_ROW1_COL3="./docs/source/images/example_legrand2.png"
IMAGE_ROW1_COL4="./docs/source/images/example_keratoconus_4.webp"
IMAGE_ROW1_COL5="./docs/source/images/example_brewster.png"
IMAGE_ROW2_COL1="./docs/source/images/example_gui_automation_1.png"
IMAGE_ROW2_COL2="./docs/source/images/LED_illuminants.svg"
IMAGE_ROW2_COL3="./docs/source/images/example_double_gauss_2.png"
IMAGE_ROW2_COL4="./docs/source/images/rgb_render_srgb1.webp"
IMAGE_ROW2_COL5="./docs/source/images/example_cosine_surfaces1.png"

# Define height spacing (in pixels)
HEIGHT=200
HORIZONTAL_SPACING=25
VERTICAL_SPACING=15

# Define output file name (using .png for transparency support)
OUTPUT_IMAGE="./docs/source/images/header.webp"

# Resize images while maintaining aspect ratio
magick "$IMAGE_ROW1_COL1" -filter Lanczos -resize x${HEIGHT}x result_row1_col1.png
magick "$IMAGE_ROW1_COL2" -filter Lanczos -resize x${HEIGHT}x result_row1_col2.png
magick "$IMAGE_ROW1_COL3" -filter Lanczos -resize x${HEIGHT}x result_row1_col3.png
magick "$IMAGE_ROW1_COL4" -filter Lanczos -resize x${HEIGHT}x result_row1_col4.png
magick "$IMAGE_ROW1_COL5" -filter Lanczos -resize x${HEIGHT}x result_row1_col5.png
magick "$IMAGE_ROW2_COL1" -filter Lanczos -resize x${HEIGHT}x result_row2_col1.png
magick "$IMAGE_ROW2_COL2" -filter Lanczos -resize x${HEIGHT}x result_row2_col2.png
magick "$IMAGE_ROW2_COL3" -filter Lanczos -resize x${HEIGHT}x result_row2_col3.png
magick "$IMAGE_ROW2_COL4" -filter Lanczos -resize x${HEIGHT}x result_row2_col4.png
magick "$IMAGE_ROW2_COL5" -filter Lanczos -resize x${HEIGHT}x result_row2_col5.png

# Create the combined image using montage with spacing and transparency
# -tile 5x2 specifies 5 columns and 2 rows
# -geometry defines the spacing: WidthxHeight+xOffset+yOffset
# -background transparent sets the background color for spacing
montage \
    result_row1_col1.png \
    result_row1_col2.png \
    result_row1_col3.png \
    result_row1_col4.png \
    result_row1_col5.png \
    result_row2_col1.png \
    result_row2_col2.png \
    result_row2_col3.png \
    result_row2_col4.png \
    result_row2_col5.png \
    -tile 5x2 \
    -geometry +${HORIZONTAL_SPACING}+${VERTICAL_SPACING} \
    -background transparent \
    -quality 95\
    "$OUTPUT_IMAGE"

# Clean up temporary resized images
rm result_row1_col1.png \
   result_row1_col2.png \
   result_row1_col3.png \
   result_row1_col4.png \
   result_row1_col5.png \
   result_row2_col1.png \
   result_row2_col2.png \
   result_row2_col3.png \
   result_row2_col4.png \
   result_row2_col5.png

echo "Combined image with spacing created: $OUTPUT_IMAGE"
