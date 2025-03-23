from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from io import BytesIO

import cv2
import numpy
import os
import utils

# Height & width/height proportion of resulting dataset elements.
CELL_HEIGHT = 16
CELL_PROPORTION = 0.5
# Character which will be used to measure width of symbol
WIDE_CHAR = 'w'

# Count files in directory
def file_cnt(path):
    path = Path(path)
    return sum(1 for entry in path.iterdir() if entry.is_file())

# Create missing subdirectories for character and font
def create_missing_subdirectories(ordinal, font):
    # Create subdirectories if not exists
    if not os.path.exists('./generated'):
        os.mkdir('generated')
    if not os.path.exists('./generated/' + ordinal):
        os.mkdir('generated/' + ordinal)
    if not os.path.exists('./generated/' + ordinal + '/' + font):
        os.mkdir('generated/' + ordinal + '/' + font)

def generate(char, font_ref, cell_width, cell_height):
    image = Image.new('L', (cell_width, cell_height), 255)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, fill = 0, font = font_ref)
    pixels = numpy.asarray(image)
    pixels = utils.to_black_and_white(pixels)
    return pixels

def generate_dataset(font, cell_height):
    font_file = open('fonts/' + font, "rb")
    bytes_font = BytesIO(font_file.read())
    font_ref = ImageFont.truetype(bytes_font, cell_height)

    cell_width = round(cell_height * CELL_PROPORTION)
    wide_char = generate(WIDE_CHAR, font_ref, cell_width, cell_height)
    wide_char = utils.shrink(
        wide_char, 0, cell_height - 1, 0, cell_width - 1, 255,
        shrink_top = False, shrink_bottom = False
    )
    width = wide_char.shape[1]

    # Alphanumeric + basic symbols
    for i in range(33, 127):
        symbol = generate(chr(i), font_ref, cell_width, cell_height)
        symbol = utils.get_sub_image(symbol, 0, cell_height - 1, 0, width - 1)
        symbol = cv2.resize(symbol, (round(CELL_HEIGHT * CELL_PROPORTION), CELL_HEIGHT))

        # can we avoid this?
        symbol = cv2.resize(symbol, (CELL_HEIGHT, CELL_HEIGHT))

        symbol = symbol.astype("uint8")
        symbol = utils.to_black_and_white(symbol)

        create_missing_subdirectories(str(i), font)
        index = file_cnt('generated/' + str(i) + '/' + font)
        cv2.imwrite('generated/' + str(i) + '/' + font + '/' + str(index) + '.png', symbol)

def generate_all():
    fonts = Path('./fonts')
    for font in fonts.iterdir():
        for height in range(CELL_HEIGHT, CELL_HEIGHT * 2):
            generate_dataset(str(font).replace('fonts\\', ''), height)

generate_all()