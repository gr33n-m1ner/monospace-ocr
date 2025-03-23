import cv2
import numpy

# Calculate background area % of sub-image.
def calculate_background_area(pixels, row_l, row_r, col_l, col_r, background):
    total_area = (row_r - row_l + 1) * (col_r - col_l + 1)
    background_area = 0

    for row in range(row_l, row_r + 1):
        for col in range(col_l, col_r + 1):
            if pixels[row][col] == background:
                background_area = background_area + 1

    return background_area / total_area

# Extract sub-image from image
def get_sub_image(pixels, row_l, row_r, col_l, col_r):
    h = row_r - row_l + 1
    w = col_r - col_l + 1
    sub_pixels = numpy.zeros([h, w])
    for i in range(h):
        for j in range(w):
            sub_pixels[i][j] = pixels[i + row_l][j + col_l]
    return sub_pixels

# Get bounding box of symbol
def shrink(
        pixels, row_l, row_r, col_l, col_r, background,
        shrink_top = True,
        shrink_bottom = True,
        shrink_left = True,
        shrink_right = True
):
    while True:
        end_shrink = True

        if shrink_top and calculate_background_area(pixels, row_l, row_l, col_l, col_r, background) == 1:
            row_l = row_l + 1
            end_shrink = False
        if shrink_bottom and calculate_background_area(pixels, row_r, row_r, col_l, col_r, background) == 1:
            row_r = row_r - 1
            end_shrink = False
        if shrink_left and calculate_background_area(pixels, row_l, row_r, col_l, col_l, background) == 1:
            col_l = col_l + 1
            end_shrink = False
        if shrink_right and calculate_background_area(pixels, row_l, row_r, col_r, col_r, background) == 1:
            col_r = col_r - 1
            end_shrink = False

        if end_shrink:
            break
    return get_sub_image(pixels, row_l, row_r, col_l, col_r)

# Convert to black-and-white via thresholding
def to_black_and_white(image):
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)

    return image