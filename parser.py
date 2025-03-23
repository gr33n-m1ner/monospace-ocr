from operator import truediv

import cv2
import math
import numpy

import utils
from PIL import Image

index = 0

POOL_SIZE = 3

# Map of trained letters to pooling areas
POOLING_AREAS = {}

img = {}

# Train character mapping
TRAIN_MAPPING = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F',
                 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z', '[', ']', '(', ')', '{', '}', '<', '>', '\'', '"', '=', '+',
                 '-', '*', '/', '.', ',', ':', ';', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Debug: extract sub-image from image
def sub_image(data, row_l, row_r, col_l, col_r):
    h = row_r - row_l + 1
    w = col_r - col_l + 1
    pixels = numpy.zeros([h, w, 3], dtype = numpy.uint8)
    for i in range(h):
        for j in range(w):
            pixels[i][j] = data[i + row_l][j + col_l]
    return pixels

# Debug: output sub-image as image
def output(data, row_l, row_r, col_l, col_r, filename):
    pixels = sub_image(data, row_l, row_r, col_l, col_r)
    image = Image.fromarray(pixels, 'RGB')
    image.save(filename + '.png')

# Determines whether sub-image is empty (could be single row or column)
def is_background(image, row_l, row_r, col_l, col_r, background):
    for row in range(row_l, row_r + 1):
        for col in range(col_l, col_r + 1):
            if image[row][col] != background:
                return False
    return True

def is_empty(image, row_l, row_r, col_l, col_r, background):
    bad = 0
    for row in range(row_l, row_r + 1):
        for col in range(col_l, col_r + 1):
            if image[row][col] != background:
                bad = bad + 1
    return bad <= 3

def coeff(a, b):

    aa = cv2.resize(a, (32, 32))
    bb = cv2.resize(b, (32, 32))
    _, aa = cv2.threshold(aa, 127, 255, cv2.THRESH_BINARY)
    _, bb = cv2.threshold(bb, 127, 255, cv2.THRESH_BINARY)
    res = 0
    for i in range(32):
        for j in range(32):
            if aa[i][j] == bb[i][j]:
                res = res + 1
    return res

def find_split(background_flag):
    sz = len(background_flag)
    for step in range(1, sz):
        for start in range(sz):
            mistakes = 0
            total = 0
            if not background_flag[start]:
                break
            for curr in range(start, sz, step):
                if not background_flag[curr]:
                    mistakes = mistakes + 1
                total = total + 1
            if (mistakes / total) <= 0.15:
                return [start, step]

def find_pooling_areas(image, row_l, row_r, col_l, col_r, background, mode):
    height = image.shape[0]
    width = image.shape[1]

    row_r = min(row_r, height - 1)
    col_r = min(col_r, width - 1)

    if is_background(image, row_l, row_r, col_l, col_r, background):
       return tuple(0 for _ in range(POOL_SIZE * POOL_SIZE))

    # Fit to bounding box
    while is_background(image, row_l, row_l, col_l, col_r, background):
        row_l = row_l + 1
    while is_background(image, row_r, row_r, col_l, col_r, background):
        row_r = row_r - 1
    while is_background(image, row_l, row_r, col_l, col_l, background):
        col_l = col_l + 1
    while is_background(image, row_l, row_r, col_r, col_r, background):
        col_r = col_r - 1

    if mode == 'train':
        c = TRAIN_MAPPING[index]
        img[c] = sub_image(image, row_l, row_r, col_l, col_r)

    areas = [0 for _ in range(POOL_SIZE * POOL_SIZE)]
    row_step = (row_r - row_l + 1) / POOL_SIZE
    col_step = (col_r - col_l + 1) / POOL_SIZE
    for row in range(row_l, row_r - 1):
        for col in range(col_l, col_r - 1):
            if image[row][col] == background:
                continue
            row_steps = math.floor((row - row_l) / row_step)
            col_steps = math.floor((col - col_l) / col_step)
            areas[row_steps * POOL_SIZE + col_steps] = areas[row_steps * POOL_SIZE + col_steps] + 1
    # for i in range(len(areas)):
        # areas[i] = round((areas[i] / ((row_r - row_l + 1) * (col_r - col_l + 1))) * 10000)
    return tuple(areas)

def loss(area_a, area_b):
    result = 0
    for i in range(len(area_a)):
        delta = area_a[i] - area_b[i]
        result = result + delta * delta
    return result

def parse_cell(image, row_l, row_r, col_l, col_r, background, mode):
    global index
    if is_empty(image, row_l, row_r, col_l, col_r, background):
        return ' '
    # output(image, row_l, row_r, col_l, col_r, str(index))
    pooling_areas = find_pooling_areas(image, row_l, row_r, col_l, col_r, background, mode)
    if mode == 'train':
        c = TRAIN_MAPPING[index]
        POOLING_AREAS[c] = pooling_areas
        print(c, pooling_areas)
        index = index + 1
        return c
    else:
        while is_background(image, row_l, row_l, col_l, col_r, background):
            row_l = row_l + 1
        while is_background(image, row_r, row_r, col_l, col_r, background):
            row_r = row_r - 1
        while is_background(image, row_l, row_r, col_l, col_l, background):
            col_l = col_l + 1
        while is_background(image, row_l, row_r, col_r, col_r, background):
            col_r = col_r - 1
        im = sub_image(image, row_l, row_r, col_l, col_r)
        best_loss = 32 * 32
        best_char = ' '
        for c in POOLING_AREAS.keys():
            curr_coeff = coeff(img[c], im)
            if curr_coeff < best_loss:
                best_loss = curr_coeff
                best_char = c
        index = index + 1
        return best_char


# Parses image row
def parse_row(image, row_l, row_r, background, mode):
    # output(image, row_l, row_r, 0, image.shape[1] - 1, str(index))
    width = image.shape[1]

    is_background_col = [is_background(image, row_l, row_r, i, i, background) for i in range(width)]
    col_split = find_split(is_background_col)

    # print(col_split)

    result = ''
    for col in range(col_split[0], width, col_split[1]):
        col_end = min(col + col_split[1] - 1, width - 1)
        result = result + parse_cell(image, row_l, row_r, col, col_end, background, mode)
    return result

# Parses text from image
def parse(path, mode):
    if mode == 'train':
        POOLING_AREAS[' '] = [0 for _ in range(POOL_SIZE * POOL_SIZE)]
    image = cv2.imread(path)
    image = utils.preprocess(image)

    # For simplicity's sake
    background = image[0][0]

    if mode == 'train':
        img[' '] = sub_image(image, 0, 0, 0, 0)

    height = image.shape[0]
    width = image.shape[1]

    is_row_background = [is_background(image, row, row, 0, width - 1, background) for row in range(height)]
    row_split = find_split(is_row_background)

    result = ''
    for row in range(row_split[0], height, row_split[1]):
        row_end = min(row + row_split[1] - 1, height - 1)
        result = result + parse_row(image, row, row_end, background, mode) + '\n'
    return result
