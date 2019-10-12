#!/usr/bin/env python
"""
Reads and writes weight data files.

!!! Weight data files must be in Image row ordering (0, 0) in the top-left. !!!
"""

from __future__ import print_function, division

import sys
import gzip
import logging

import numpy


def read_tex_from_file(ioFile):
    """
    Read a .data file into memory.

    Inputs:
        ioFile: a file for the .data file

    Returns:
        width-by-height-by-#channels numpy float32 array of data
        width-by-height numpy boolean array where True values correspond to
            values where weights are zero in all channels.
    """
    f = gzip.GzipFile(fileobj=ioFile, mode="rb")

    # fromfile() is a numpy function
    # UPDATE: We can't use fromfile() on a gzip file object. We have to read
    #         it first and then use frombuffer().
    #         Reference: https://bit.ly/2II4HM6
    # NOTE: I should make a dtype("")
    header = f.read(3 * numpy.uint32().itemsize)
    width, height, channels = numpy.frombuffer(header, numpy.uint32, 3)

    # Make a mask.
    # Since every pixel in the model should have some weight, the mask can
    # be True if any non-zero weight every appears for a pixel.
    mask = numpy.zeros((width, height), dtype=bool)

    # This is inefficient. We could read it at once, but I don't want to think
    # about making sure the channel-wise memory layout is what numpy wants.
    result = numpy.zeros((width, height, channels), dtype=numpy.float32)
    for chan in range(channels):
        data = f.read(width * height * numpy.float32().itemsize)
        data = numpy.frombuffer(data, numpy.float32,
                                width * height).reshape(width, height)
        # Update the mask with any nonzero entries.
        mask = numpy.logical_or(mask, data != 0)
        result[:, :, chan] = data
    result = result[::-1]

    return result, mask


def read_tex_from_path(path):
    """
    Read a .data file into memory.

    Inputs:
        path: a path to the .data file

    Returns
    -------
        width-by-height-by-#channels numpy float32 array of data
        width-by-height numpy boolean array where True values correspond to
            values where weights are zero in all channels.

    """
    logging.info("Loading: %s" % path)
    with open(path, "rb") as f:
        result, mask = read_tex_from_file(f)
    return result, mask


def write_tex_to_file(ioFile, data):
    """
    Save a .data to the given file.

    Inputs:
        ioFile: a File at which to save the .data file
        data: width-by-height-by-#channels numpy float32 array of data
    """
    data = data[::-1]

    f = gzip.GzipFile(fileobj=ioFile, mode="wb")

    header = numpy.zeros(3, dtype=numpy.uint32)
    header[:] = data.shape

    f.write(memoryview(header))

    channel = numpy.zeros((data.shape[0], data.shape[1]), dtype=numpy.float32)
    for ch in range(data.shape[2]):
        channel[:] = data[:, :, ch]
        f.write(memoryview(channel))


def write_tex_to_path(path, data):
    """
    Saves a .data to disk.

    Inputs:
        path: a path at which to save the .data file
        data: width-by-height-by-#channels numpy float32 array of data
    """

    logging.info("Saving: %s" % path)
    with open(path, "wb") as f:
        write_tex_to_file(f, data)


def normalize_data(data, mask=None):
    """
    Normalize the width-by-height-by-#channels array `data`, optionally
    ignoring values for which `mask` is True. Modifies `data` in place and
    returns None.
    """

    if mask is None:
        data /= data.sum(axis=2)[:, :, numpy.newaxis]
    else:
        assert mask.shape == data.shape[:2]
        data[mask] /= data.sum(axis=2)[mask][..., numpy.newaxis]


if __name__ == "__main__":
    def usage():
        print("Usage: %s path/to/tex1.data path/to/tex2.data" % sys.argv[0],
              file=sys.stderr)
        sys.exit(-1)
    logging.basicConfig(
        format="[%(levelname)s] [%(asctime)s] %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO)

    if len(sys.argv) != 3:
        usage()

    path1, path2 = sys.argv[1:]

    tex1, mask1 = read_tex_from_path(path1)
    tex2, mask2 = read_tex_from_path(path2)

    assert tex1.shape == tex2.shape
    assert mask1.shape == mask2.shape

    assert (mask1 == mask2).all()

    tex1 = tex1[mask1]
    tex2 = tex2[mask2]

    # This is pretty memory intensive, so let's be efficient.

    # diff:
    # diff = tex1 - tex2
    diff = tex1
    numpy.subtract(tex1, tex2, diff)
    # Don't use tex1 anymore, it's been reused as diff.
    del tex1

    # absolute difference:
    # abs_diff = abs(tex1-tex2)
    abs_diff = diff
    numpy.absolute(diff, abs_diff)
    # Don't use diff anymore, it's been reused as abs_diff.
    del diff

    total_abs_diff = abs_diff.sum()
    logging.info("Total absolute difference: %s" % total_abs_diff)
    logging.info("Average absolute difference: %s" %
                 (total_abs_diff / numpy.prod(abs_diff.shape)))
    logging.info("Median absolute difference: %s" % numpy.median(abs_diff))
    logging.info("Maximum absolute difference: %s" % abs_diff.max())
    logging.info("Minimum absolute difference: %s" % abs_diff.min())

    # difference, squared:
    # abs_diff2 = abs_diff**2
    abs_diff2 = abs_diff
    numpy.square(abs_diff, abs_diff2)
    # Don't use abs_diff anymore, it's been reused as abs_diff2.
    del abs_diff

    avg_abs_diff2 = numpy.average(abs_diff2)

    logging.info("Mean squared error: %s" % avg_abs_diff2)
    logging.info("Root mean squared error: %s" % numpy.sqrt(avg_abs_diff2))
