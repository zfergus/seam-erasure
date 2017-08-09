"""
Utility functions for testing values.

Written by Zachary Ferguson
"""

from bcolors import bcolors


def result_str(value):
    """ Create a colored string for if the value is PASS or FAIL """
    return "%s%s" % ((bcolors.OKGREEN + "PASS") if value
        else (bcolors.FAIL + "Fail"), bcolors.ENDC)


def display_results(value, format_str = "%s"):
    """ Prints the result_str of the given value. """
    print(format_str % result_str(value))


def test_equal_f(x0, x1, epsilon = 1.0e-10):
    """ Test for floating point precision within epsilon. """
    return abs(x0 - x1) < epsilon


def test_equal_i(x0, x1):
    """ Test for exact equality. """
    return x0 == x1


def test_equal(x0, x1):
    """ Generic test for equality. """
    if type(x0) is float or type(x1) is float:
        return test_equal_f(x0, x1)
    else:
        return test_equal_i(x0, x1)
