"""
Create a boolean mask for if the pixels of a texture is constrained, True, or
free, False.

Written by Zachary Ferguson
"""

import itertools
import logging

import numpy
import scipy
import scipy.sparse
from tqdm import tqdm

from .seam_intervals import compute_edge_intervals
from .points_in_triangle import points_in_triangle
from .util import UV_to_XY, pairwise, lerp_UV, surrounding_pixels


def get_all_surrounding_pixels(edges, width, height):
    """
    Get a set of all pixels surrounding the given edges.

    Input:
        edges  - unsorted list of UV edges
        width  - width of the texture
        height - height of the texture
    Output:
        Returns a set of (X, Y) coordinates for the surrounding pixels.
    """
    # TODO: This could be improved for better performance
    pixels = set()
    for edge in edges:
        interval = sorted(list(compute_edge_intervals(edge, width, height)))

        # Find all pixels along the seam
        for a, b in pairwise(interval):
            uv_mid = lerp_UV((a + b) / 2.0, edge[0], edge[1])
            pixels |= set(surrounding_pixels(
                uv_mid, width, height, as_tuple=True))
    return pixels


def mask_seam(mesh, seam_edges, width, height, seam_pixels=None):
    """
    Create a boolean mask for if the pixels of a texture is constrained (True)
    or free (False). Pixels along the seam are not constrained.

    Steps:
        1. Find all pixels along the seam_loops using find_all_seam_pixels()
        2. Mark these pixels in the mask as False.

    Inputs:
        mesh - a OBJ recordclass
        seam_edges - an unsorted list of edges along the seam
        width  - width of texture/mask
        height - height of texture/mask
    Output:
        Constructs a width x height array of booleans. True if the pixel
        is part of the mask/foreground. False if the pixel is part of the
        background.
    """
    # Store the surrounding pixels XY coords in pixels
    seam_pixels = get_all_surrounding_pixels(seam_edges, width, height)

    vals = numpy.full(len(seam_pixels), True, dtype=bool)
    coords = numpy.array(list(seam_pixels))
    mask = scipy.sparse.coo_matrix((vals, (coords[:, 1], coords[:, 0])),
                                   shape=(height, width)).A.astype(bool)

    return ~mask


def mask_inside_seam(mesh, seam_edges, width, height):
    """
    Create a boolean mask for if the pixels of a texture is constrained (True)
    or free (False).

    Steps:
        1. Find all pixels along the seam_loops using find_all_seam_pixels()
        2. Test these pixels against the triangles of the mesh in UV-space.
            a. If pixel is outside of all the triangles, mark it as free(false)
            b. Else if the pixel is inside at least one triangle mark as
               constrained (true)
    Inputs:
        mesh - a OBJ recordclass
        seam_edges - an unsorted list of edges along the seam
        width  - width of texture/mask
        height - height of texture/mask
    Output:
        Constructs a width x height array of booleans. True if the pixel
        is part of the mask/foreground. False if the pixel is part of the
        background.
    """
    # Store the surrounding pixels XY coords in pixels
    seam_pixels = get_all_surrounding_pixels(seam_edges, width, height)

    # Create a list of the UV faces in Pixel space
    faces = [
        numpy.array([UV_to_XY(mesh.vt[fv.vt], width, height) for fv in face])
        for face in mesh.f
    ]

    # This mask should be small enough for a dense matrix
    mask = numpy.zeros((height, width), dtype=bool)

    # Constrain all the pixels in seam_pixels that are inside a face
    pts = numpy.array(list(seam_pixels))
    disable_pbar = logging.getLogger().getEffectiveLevel() > logging.INFO
    for i, face in enumerate(tqdm(faces, unit="faces", disable=disable_pbar,
                                  desc="Building Least Squares Constraints")):
        # Create a bounding box for the face
        ll = numpy.array([face[:, 0].min(), face[:, 1].min()])
        ur = numpy.array([face[:, 0].max(), face[:, 1].max()])

        # Intersect the bounding box with the seam pixels
        inidx = numpy.all(numpy.logical_and(ll <= pts, pts <= ur), axis=1)
        inbox = pts[inidx]

        # Only test seam pixels inside the bounding_box
        if(inbox.shape[0] > 0):
            mask[inbox[:, 1], inbox[:, 0]] |= points_in_triangle(face, inbox)

    # Mask is False if pixels inside (this needs to be inverted).
    vals = numpy.full(len(seam_pixels), True, dtype=bool)
    coords = numpy.array(list(seam_pixels))
    full = scipy.sparse.coo_matrix((vals, (coords[:, 1], coords[:, 0])),
                                   shape=mask.shape).A
    mask = full ^ mask
    return ~mask


def mask_inside_faces(mesh, width, height, init_mask=None):
    """
    Create a boolean mask for if the pixels of a texture is constrained (True)
    or free (False). For all pixels mask the pixel as false if the pixel is
    inside all triangles.

    Inputs:
        mesh - a OBJ recordclass
        width  - width of texture/mask
        height - height of texture/mask
        init_mask - a mask of size height x width to start from.
            (Default: None -> initial mask of all False)
    Output:
        Constructs a width x height array of booleans. True if the pixel
        is part of the mask/foreground. False if the pixel is part of the
        background.
    """
    # Create a list of the UV faces in Pixel space
    faces = [
        numpy.array([UV_to_XY(mesh.vt[fv.vt], width, height) for fv in face])
        for face in mesh.f
    ]

    # This mask should be small enough for a dense matrix
    mask = init_mask
    if(mask is None):
        mask = numpy.zeros((height, width), dtype=bool)

    disable_pbar = logging.getLogger().getEffectiveLevel() > logging.INFO
    for i, face in enumerate(tqdm(faces, unit="faces", disable=disable_pbar,
                                  desc="Building Dirichlet Energy")):
        # Bounding box for the face to get surrounding pixels.
        ll = numpy.array([face[:, 0].min(), face[:, 1].min()])
        ur = numpy.array([face[:, 0].max(), face[:, 1].max()])
        bbox = numpy.vstack([ll, ur])
        xRange = range(
            max(0, int(bbox[0][0])),
            min(width,
                int(numpy.ceil(bbox[1][0])) + 1))
        yRange = range(
            max(0, int(bbox[0][1])),
            min(height,
                int(numpy.ceil(bbox[1][1])) + 1))
        inbox = numpy.array(list(itertools.product(xRange, yRange)))

        # Test inside face for all pixels in the bounding box
        mask[inbox[:, 1], inbox[:, 0]] |= points_in_triangle(face, inbox)

    return ~mask


if __name__ == "__main__":
    import obj_reader
    import texture
    from find_seam import find_seam, seam_to_UV

    mesh = obj_reader.quads_to_triangles(
        obj_reader.load_obj('../models/cow.obj'))
    texture = numpy.array(
        texture.load_texture("../textures/cow/Cow_Monster_N.png"))

    height, width, depth = (texture.shape + (1, ))[:3]
    N = width * height
    textureVec = texture.reshape(N, -1)

    logging.info("\nFinding seam of model")
    seam, boundary, foldovers = find_seam(mesh)
    uv_seam, uv_boundary, uv_foldovers = seam_to_UV(
        mesh, seam, boundary, foldovers)
    logging.info("Done\n")

    logging.info("Number of edges along the seam: %d" % (len(seam) * 2))
    logging.info("Number of edges along the boundary: %d" % len(boundary))
    logging.info("Number of foldover edges: %d\n" % len(foldovers))

    # Find all of the seam loops
    bag_of_edges = ([edge for edgepair in uv_seam
                     for edge in edgepair] + uv_boundary + uv_foldovers)

    # Constrain the values
    logging.info("Mask Inside Seam")
    lsq_mask = mask_inside_seam(mesh, bag_of_edges, width, height)

    logging.info("Mask Seam")
    lsq1_mask = mask_seam(mesh, bag_of_edges, width, height)

    logging.info("XOR")
    lsq2_mask = lsq_mask ^ lsq1_mask

    # Construct a dirichlet energy for the texture.
    logging.info("Mark Inside Faces")
    dirichlet_mask = mask_inside_faces(
        mesh, width, height, init_mask=~lsq_mask)
