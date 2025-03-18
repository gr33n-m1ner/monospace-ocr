import cv2
import math
import numpy

import utils
from PIL import Image

POOL_SIZE = 4

# Allowed non-background elements in background row/col
BACKGROUND_CONCENTRATION_ROW = 1.00
BACKGROUND_CONCENTRATION_COL = 0.95
BACKGROUND_CONCENTRATION_AREA = 1.00

# Map of trained letters to quadrant areas
QUADRANT_AREAS = {}

# Train character mapping
TRAIN_MAPPING = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', '/', '@', '*', '(', ')', '[', ']', ';', ':', '"', '<', '>',
                 '{', '}', '.', ',', '+', '=', '0', '\\']

def is_background_row(image, row, background):
    count = 0
    width = image.shape[1]
    for col in range(width):
        if image[row][col] == background:
            count = count + 1
    return count / width >= BACKGROUND_CONCENTRATION_ROW

def is_background_col(image, col, background):
    count = 0
    height = image.shape[0]
    for row in range(height):
        if image[row][col] == background:
            count = count + 1
    return count / height >= BACKGROUND_CONCENTRATION_COL

def find_split(is_background):
    sz = len(is_background)
    for step in range(1, sz):
        for start in range(sz):
            ok = True
            if not is_background[start]:
                break
            for curr in range(start, sz, step):
                if not is_background[curr]:
                    ok = False
                    break
            if ok:
                return [start, step]

def is_partial_background_row(image, row, col_l, col_r, background):
    for col in range(col_l, col_r + 1):
        if image[row][col] != background:
            return False
    return True

def is_partial_background_col(image, col, row_l, row_r, background):
    for row in range(row_l, row_r + 1):
        if image[row][col] != background:
            return False
    return True

def is_empty(image, row_l, row_r, col_l, col_r, background):
    height = image.shape[0]
    width = image.shape[1]

    row_r = min(row_r, height - 1)
    col_r = min(col_r, width - 1)

    count = 0
    total = 0
    for row in range(row_l, row_r + 1):
        for col in range(col_l, col_r + 1):
            if image[row][col] == background:
                count = count + 1
            total = total + 1
    return count / total >= BACKGROUND_CONCENTRATION_AREA

def find_quadrant_areas(image, row_l, row_r, col_l, col_r, background, index = None, mode = None):
    height = image.shape[0]
    width = image.shape[1]

    row_r = min(row_r, height - 1)
    col_r = min(col_r, width - 1)

    if is_empty(image, row_l, row_r, col_l, col_r, background):
        return tuple(0 for _ in range(POOL_SIZE * POOL_SIZE))

    while is_partial_background_row(image, row_l, col_l, col_r, background):
        row_l = row_l + 1
    while is_partial_background_row(image, row_r, col_l, col_r, background):
        row_r = row_r - 1
    while is_partial_background_col(image, col_l, row_l, row_r, background):
        col_l = col_l + 1
    while is_partial_background_col(image, col_r, row_l, row_r, background):
        col_r = col_r - 1

    areas = [0 for _ in range(POOL_SIZE * POOL_SIZE)]
    row_step = (row_r - row_l + 1) / POOL_SIZE
    col_step = (col_r - col_l + 1) / POOL_SIZE
    for row in range(row_l, row_r + 1):
        for col in range(col_l, col_r + 1):
            if image[row][col] == background:
                continue
            row_steps = math.floor((row - row_l) / row_step)
            col_steps = math.floor((col - col_l) / col_step)
            areas[row_steps * POOL_SIZE + col_steps] = areas[row_steps * POOL_SIZE + col_steps] + 1
    for i in range(len(areas)):
        areas[i] = round((areas[i] / ((row_r - row_l + 1) * (col_r - col_l + 1))) * 10000)
    return tuple(areas)

# Take sub-image from image
def sub_image(data, row_l, row_r, col_l, col_r):
    h = row_r - row_l + 1
    w = col_r - col_l + 1
    pixels = numpy.zeros([h, w, 3], dtype = numpy.uint8)
    for i in range(h):
        for j in range(w):
            pixels[i][j] = data[i + row_l][j + col_l]
    return pixels

# Debug: output sub-image as image
def output(data, row_l, row_r, col_l, col_r, index):
    pixels = sub_image(data, row_l, row_r, col_l, col_r)
    image = Image.fromarray(pixels, 'RGB')
    image.save(str(index) + '.png')

def loss(area_a, area_b):
    result = 0
    for i in range(len(area_a)):
        delta = area_a[i] - area_b[i]
        result = result + delta * delta
    return result

# Parses text from image
def parse(path, mode = None):
    if mode == 'train':
        QUADRANT_AREAS[' '] = tuple([0 for _ in range(POOL_SIZE * POOL_SIZE)])

    image = cv2.imread(path)
    image = utils.preprocess(image)

    # For simplicity's sake
    background = image[0][0]

    height = image.shape[0]
    width = image.shape[1]

    is_row_background = [is_background_row(image, row, background) for row in range(height)]
    is_col_background = [is_background_col(image, col, background) for col in range(width)]

    row_split = find_split(is_row_background)
    col_split = find_split(is_col_background)

    if mode != 'train':
        col_split = [6, 9]

    index = 0
    result = ''
    for row in range(row_split[0], height, row_split[1]):
        for col in range(col_split[0], width, col_split[1]):
            if mode == 'train' and is_empty(image, row, row + row_split[1] - 1, col, col + col_split[1] - 1, background):
                # output(image, row, row + row_split[1] - 1, col, col + col_split[1] - 1, index)
                continue

            areas = find_quadrant_areas(image, row, row + row_split[1] - 1, col, col + col_split[1] - 1, background, index, 'test')
            if mode == 'train':
                c = TRAIN_MAPPING[index]
                QUADRANT_AREAS[c] = areas
            else:
                best_loss = 999_999_999
                best_char = None
                for key in QUADRANT_AREAS.keys():
                    curr_loss = loss(QUADRANT_AREAS[key], areas)
                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        best_char = key
                result = result + best_char
            index = index + 1
        result = result + '\n'

    return result
