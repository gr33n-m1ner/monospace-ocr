import utils, cv2, numpy as np

img = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (8, 16))
img  = utils.to_black_and_white(img)
cv2.imwrite('3.png', img)