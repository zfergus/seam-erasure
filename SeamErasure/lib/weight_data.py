#!/usr/bin/env python

"""
Reads and writes weight data files.
!!! Weight data files must be in Image row ordering (0, 0) in the top-left. !!!
"""

from __future__ import print_function, division

from numpy import *
import gzip


def read_tex_from_file(ioFile):
    '''
    Reads a .data file into memory.

    Inputs:
        ioFile: a file for the .data file

    Returns:
        width-by-height-by-#channels numpy float32 array of data
        width-by-height numpy boolean array where True values correspond to
            values where weights are zero in all channels.
    '''
    f = gzip.GzipFile(fileobj=ioFile, mode='rb')

    # fromfile() is a numpy function
    # UPDATE: We can't use fromfile() on a gzip file object. We have to read
    #         it first and then use frombuffer().
    #         http://stackoverflow.com/questions/15966335/efficient-numpy-
    #         fromfile-on-zipped-files
    # NOTE: I should make a dtype('')
    header = f.read(3 * uint32().itemsize)
    width, height, channels = frombuffer(header, uint32, 3)

    # Make a mask.
    # Since every pixel in the model should have some weight, the mask can
    # be True if any non-zero weight every appears for a pixel.
    mask = zeros((width, height), dtype = bool)

    # This is inefficient. We could read it at once, but I don't want to think
    # about making sure the channel-wise memory layout is what numpy wants.
    result = zeros((width, height, channels), dtype = float32)
    for chan in range(channels):
        data = f.read(width * height * float32().itemsize)
        data = frombuffer(data, float32, width * height).reshape(width, height)
        # Update the mask with any nonzero entries.
        mask = logical_or(mask, data != 0)
        result[:, :, chan] = data
    result = result[::-1]

    return result, mask


def read_tex_from_path(path):
    '''
    Reads a .data file into memory.

    Inputs:
        path: a path to the .data file

    Returns:
        width-by-height-by-#channels numpy float32 array of data
        width-by-height numpy boolean array where True values correspond to
            values where weights are zero in all channels.
    '''

    print('+ Loading:', path)

    with file(path, 'rb') as f:
        result, mask = read_tex_from_path(f)

    print('- Loaded:', path)

    return result, mask


def write_tex_to_file(ioFile, data):
    '''
    Saves a .data to the given file.

    Inputs:
        ioFile: a File at which to save the .data file
        data: width-by-height-by-#channels numpy float32 array of data
    '''

    data = data[::-1]

    f = gzip.GzipFile(fileobj=ioFile, mode='wb')

    header = zeros(3, dtype = uint32)
    header[:] = data.shape

    f.write(getbuffer(header))

    channel = zeros((data.shape[0], data.shape[1]), dtype = float32)
    for ch in range(data.shape[2]):
        channel[:] = data[:, :, ch]
        f.write(getbuffer(channel))


def write_tex_to_path(path, data):
    '''
    Saves a .data to disk.

    Inputs:
        path: a path at which to save the .data file
        data: width-by-height-by-#channels numpy float32 array of data
    '''

    print('+ Saving:', path)

    with file(path, 'wb') as f:
        write_tex_to_file(f, data)

    print('- Saved:', path)


def normalize_data(data, mask = None):
    '''
    Normalize the width-by-height-by-#channels array `data`, optionally
    ignoring values for which `mask` is True. Modifies `data` in place and
    returns None.
    '''

    if mask is None:
        data /= data.sum(axis = 2)[:, :, newaxis]
    else:
        assert mask.shape == data.shape[:2]
        data[mask] /= data.sum(axis = 2)[mask][..., newaxis]

if __name__ == '__main__':
    import sys

    def usage():
        print("Usage:", sys.argv[0], "path/to/tex1.data path/to/tex2.data",
              file = sys.stderr)
        sys.exit(-1)

    if len(sys.argv) != 3:
        usage()

    path1, path2 = sys.argv[1:]

    tex1, mask1 = read_tex_from_path(path1)
    tex2, mask2 = read_tex_from_path(path2)

    assert tex1.shape == tex2.shape
    assert mask1.shape == mask2.shape

    assert all(mask1 == mask2)

    tex1 = tex1[mask1]
    tex2 = tex2[mask2]

    # This is pretty memory intensive, so let's be efficient.

    # diff:
    # diff = tex1 - tex2
    diff = tex1
    subtract(tex1, tex2, diff)
    # Don't use tex1 anymore, it's been reused as diff.
    del tex1

    # absolute difference:
    # abs_diff = abs(tex1-tex2)
    abs_diff = diff
    absolute(diff, abs_diff)
    # Don't use diff anymore, it's been reused as abs_diff.
    del diff

    total_abs_diff = abs_diff.sum()
    print('Total absolute difference:', total_abs_diff)
    print('Average absolute difference:',
          total_abs_diff / prod(abs_diff.shape))
    print('Median absolute difference:', median(abs_diff))
    print('Maximum absolute difference:', abs_diff.max())
    print('Minimum absolute difference:', abs_diff.min())

    # difference, squared:
    # abs_diff2 = abs_diff**2
    abs_diff2 = abs_diff
    square(abs_diff, abs_diff2)
    # Don't use abs_diff anymore, it's been reused as abs_diff2.
    del abs_diff

    avg_abs_diff2 = average(abs_diff2)

    print('Mean squared error:', avg_abs_diff2)
    print('Root mean squared error:', sqrt(avg_abs_diff2))
