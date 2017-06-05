import math
import numpy
from PIL import Image


def cosine_image(width, height, freq = lambda w: 8.0/w * math.pi):
    """
    Generates an image of size width x height that represents the values of a
    cosine function. Returns this image as an array of shape = (width, height).

    Inputs:
        width - width of output image
        height - height of output image
        freq - function that takes the width and generates the frequency of the
            cosine function. [Default: 8.0/w * PI]

    Outputs:
        arr - and array of shape = (width, height) that is filled with the
            values of -0.5*math.cos(i*freq(width))+0.5.
    """
    arr = numpy.zeros((height, width))
    for i in range(width):
        arr[:,i] = -0.5*math.cos(i*freq(width))+0.5
    return arr


def save_float_image(arr, fname):
    """
    Save the given array of floating-point values to a image of unsigned bytes.
    """
    arr = (arr * 255).round().astype("uint8")
    img = Image.fromarray(arr).transpose(Image.FLIP_TOP_BOTTOM)
    img.save(fname)

# Save a example cosine texture
if __name__ == "__main__":
    save_float_image(cosine_image(1024, 1024), "cosine.png")
