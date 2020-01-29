# -*- coding: utf-8 -*-
"""seam_erasure: Seamlessly erase seams from your favorite 3D models."""

from . import accumulate_coo
from . import bilerp_energy
from . import dirichlet
from . import find_seam
from . import lsq_constraints
from . import mask
from . import obj_reader
from . import points_in_triangle
from . import seam_erasure
from . import seam_gradient
from . import seam_intervals
from . import seam_value_energy_lerp
from . import seam_value_energy_texture
from . import texture
from . import util
from . import weight_data


__all__ = [
    "accumulate_coo", "bilerp_energy", "dirichlet", "find_seam",
    "lsq_constraints", "mask", "obj_reader", "points_in_triangle",
    "seam_erasure", "seam_gradient", "seam_intervals",
    "seam_value_energy_lerp", "seam_value_energy_texture", "texture", "util",
    "weight_data"
]

__author__ = "Zachary Ferguson"
__copyright__ = "Copyright 2016, Zachary Ferguson"
__credits__ = "Zachary Ferguson, Yotam Gingold, Songrun Liu, Alec Jacobson"
__license__ = "MIT"
__version__ = "1.0.5"
__maintainer__ = "Zachary Ferguson"
__email__ = "zfergus@nyu.edu"
__status__ = "Production"
