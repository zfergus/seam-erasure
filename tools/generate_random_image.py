import numpy
from PIL import Image


def random_image(width, height, depth):
    """ Generates a random numpy.array of size (height, width, depth) """
    return numpy.random.rand(width, height, depth)


def save_float_image(arr, fname):
    """
    Save the given array of floating-point values to a image of unsigned bytes.
    """
    arr = (arr * 255).round().astype("uint8")
    img = Image.fromarray(arr).transpose(Image.FLIP_TOP_BOTTOM)
    img.save(fname)

# Save a example cosine texture
if __name__ == "__main__":
    save_float_image(random_image(100, 100, 3), "random.png")
