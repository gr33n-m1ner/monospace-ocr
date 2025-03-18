import cv2
import numpy

def preprocess(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sharpening kernel
    kernel = numpy.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)

    # Convert to black-and-white via thresholding
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    cv2.imwrite('new.png', image)

    return image