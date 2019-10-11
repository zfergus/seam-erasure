"""
Find the edges on the seam of a 3D model.

Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
        Zachary Ferguson <zfergus2@gmu.edu>

License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)
"""

from __future__ import division

import logging

import numpy

from .util import pairwise_loop


def edges_equal_in_UV(mesh, forwards, backwards):
    """Determine if the UV edges are equal."""
    faceEdge = (mesh.f[forwards[0]][forwards[1][0]],
                mesh.f[forwards[0]][forwards[1][1]])
    uvEdge0 = [mesh.vt[faceV.vt] for faceV in faceEdge]

    faceEdge = (mesh.f[backwards[0]][backwards[1][1]],
                mesh.f[backwards[0]][backwards[1][0]])
    uvEdge1 = [mesh.vt[faceV.vt] for faceV in faceEdge]

    return uvEdge0 == uvEdge1


def orientation(a, b, c):
    """Compute the orientation of c relative to the line AB."""
    M = [[a[0] - c[0], a[1] - c[1]], [b[0] - c[0], b[1] - c[1]]]
    # Return det(M)
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


def edge_is_foldover(mesh, forwards, backwards):
    """
    Check for foldovers in UV space.

    Given two triangles (a,b,c) and (a,b,d) the goal is to determine if c and d
    lie on opposite sides of the line passing through ab. This is known as the
    3-point "orientation test".
    REF: https://www.cs.cmu.edu/~quake/robust.html
    Parameters:
        mesh - the mesh in obj_reader.OBJ recordclass format
        forward, backwards - the forwards and backwards edge s.t.
            forward == backwards[::-1]
    """
    possible_indices = {0, 1, 2}

    # Get all points a, b, c, d that make up the two triangular faces
    a = mesh.vt[mesh.f[forwards[0]][forwards[1][0]].vt]
    b = mesh.vt[mesh.f[forwards[0]][forwards[1][1]].vt]
    missing_index = list(possible_indices ^ set(forwards[1]))[0]
    c = mesh.vt[mesh.f[forwards[0]][missing_index].vt]
    missing_index = list(possible_indices ^ set(backwards[1]))[0]
    d = mesh.vt[mesh.f[backwards[0]][missing_index].vt]

    orientation1 = orientation(a, b, c)
    orientation2 = orientation(a, b, d)

    # If both points c and d are on the same side of line AB, then AB is a
    # foldover edge.
    return numpy.sign(orientation1) == numpy.sign(orientation2)


def find_seam(mesh):
    """
    Find the seam of the given mesh.

    Takes a mesh object as returned by obj_reader.load_obj().
    Returns edges as pairs of indices into face index arrays.
    Returns two lists, non-matching edges and true boundary edges.

    Let the position or texcoord indices of faces be turned into arrays:
        face_position_indices = asarray( [ [ fv.v for fv in face ]
            for face in mesh.f ], dtype = int )
        face_texcoord_indices = asarray( [ [ fv.vt for fv in face ]
            for face in mesh.f ], dtype = int )

    Then each returned non-matching edge is a 2-tuple of "index bundles", one
    for the edge (i,j) and one for the edge (j,i), that can be used to index
    into `face_position_indices` or `face_texcoord_indices`.
    For examples, let `edges` be a returned sequence of seam edges. Then:
        for edge_both_directions in non_matching_edges:
            ## The edge appears twice, once forwards and once backwards.
            forwards, backwards = edge_both_directions
            ## They can be used to find the start and end position indices:
            forwards_position_start, forwards_position_end =
                face_position_indices[ forwards ]
            backwards_position_start, backwards_position_end =
                face_position_indices[ backwards ]
            ## Or the texcoords:
            forwards_texcoord_start, forwards_texcoord_end =
                face_texcoord_indices[ forwards ]
            backwards_texcoord_start, backwards_texcoord_end =
                face_texcoord_indices[ backwards ]
            ## The positions will match, but not the texcoords:
            assert forwards_position_start, forwards_position_end ==
                backwards_position_end, backwards_position_start
            assert forwards_texcoord_start, forwards_texcoord_end !=
                backwards_texcoord_end, backwards_texcoord_start

            ## You can also use an index bundle to index into the original
            ## mesh.f list:
            forwards_face_vertex_start =
                mesh.f[ forwards[0] ][ forwards[1][0] ]
            forwards_face_vertex_end =
                mesh.f[ forwards[0] ][ forwards[1][1] ]
            backwards_face_vertex_start =
                mesh.f[ backwards[0] ][ backwards[1][0] ]
            backwards_face_vertex_end =
                mesh.f[ backwards[0] ][ backwards[1][1] ]

    True boundary edges are simply oriented index bundles (not 2-tuples),
    since a true boundary means that both orientations of the edge do not
    appear.
    for edge in true_boundary_edges:
        ## The edge can be used to find the start and end position indices:
        position_start, position_end = face_position_indices[ edge ]
        ## Or the texcoords:
        texcoord_start, texcoord_end = face_texcoord_indices[ edge ]

        ## You can use the index bundle to index into the original mesh.f list:
        face_vertex_start = mesh.f[ edge[0] ][ edge[1][0] ]
        face_vertex_end = mesh.f[ edge[0] ][ edge[1][1] ]
    """
    face_position_indices = numpy.asarray(
        [[fv.v for fv in face] for face in mesh.f], dtype=int)
    face_texcoord_indices = numpy.asarray(
        [[fv.vt for fv in face] for face in mesh.f], dtype=int)

    # Assume triangles
    assert face_position_indices.shape == face_texcoord_indices.shape
    assert face_position_indices.shape[1] == 3

    # A map from a pair of vertex indices to the index (face and endpoints)
    # into face_position_indices.
    # The following should be true for every key, value pair:
    #    key == face_position_indices[ value ]
    # This gives us a "reverse map" so that we can look up other face
    # attributes based on position edges.
    # The value are written in the format returned by numpy.where(),
    # which stores multi-dimensional indices such as array[a0,b0], array[a1,b1]
    # as ( (a0,a1), (b0,b1) ).
    directed_position_edge2face_position_index = {}
    for fi, face in enumerate(face_position_indices):
        for i, j in pairwise_loop((0, 1, 2)):
            directed_position_edge2face_position_index[(face[i], face[j])] = \
                (fi, (i, j))

    # First find all undirected position edges (collect a canonical orientation
    # of the directed edges).
    undirected_position_edges = (
        set((min(i, j), max(i, j)) for i, j in
            directed_position_edge2face_position_index.keys()))

    # Now we will iterate over all position edges.
    # Seam edges are the edges whose two opposite directed edges have different
    # texcoord indices (or one doesn't exist at all in the case of a mesh
    # boundary).
    seam_non_matching = []
    seam_mesh_boundary = []
    seam_foldover_edges = []
    for vp_edge in undirected_position_edges:
        # We should only see the canonical ordering, where the first vertex
        # index is smaller.
        assert vp_edge[0] < vp_edge[1]

        # If it and its opposite exist as directed edges, check if their
        # texture coordinate indices match.
        if (vp_edge in directed_position_edge2face_position_index and
                vp_edge[::-1] in directed_position_edge2face_position_index):
            forwards = \
                directed_position_edge2face_position_index[vp_edge]
            backwards = \
                directed_position_edge2face_position_index[vp_edge[::-1]]

            # NOTE: They should never be equal.
            assert forwards != backwards

            # If the texcoord indices are similarly flipped or the edges are
            # equivilant in UV-space.
            if ((face_texcoord_indices[forwards] ==
                 face_texcoord_indices[backwards][::-1]).all()
                    or edges_equal_in_UV(mesh, forwards, backwards)):
                # Check for foldovers in UV space.
                if (edge_is_foldover(mesh, forwards, backwards)):
                    # Add to a third list of uv foldover edges.
                    seam_foldover_edges.append(forwards)
                continue

            # Otherwise, we have a non-matching seam edge.
            seam_non_matching.append((forwards, backwards))

        # Otherwise, the edge and its opposite aren't both in the directed
        # edges. One of them should be.
        elif vp_edge in directed_position_edge2face_position_index:
            seam_mesh_boundary.append(
                directed_position_edge2face_position_index[vp_edge])
        elif vp_edge[::-1] in directed_position_edge2face_position_index:
            seam_mesh_boundary.append(
                directed_position_edge2face_position_index[vp_edge[::-1]])
        else:
            # This should never happen! One of these two must have been seen.
            assert (
                vp_edge in directed_position_edge2face_position_index
                or vp_edge[::-1] in directed_position_edge2face_position_index)

    return seam_non_matching, seam_mesh_boundary, seam_foldover_edges


def seam_to_UV(mesh, seam, boundary, foldovers):
    """
    Convert the seam, boundary, and foldovers to be in UV-space.

    Input Format:
        seam      = [[[fi, [fvi0, fvi1]], [fi', [fvi0', fvi1']]], ...]
        boundary  = [ [fi0, [fvi0, fvi1]], ...]
        foldovers = [ [fi0, [fvi0, fvi1]], ...]
    Return Format:
        ( [[[uv0, uv1], [uv0', uv1']], ...],
          [ [uv0, uv1], ...],
          [ [uv0, uv1], ...] )
    """
    uv_seam = [[[mesh.vt[mesh.f[edge[0]][i].vt] for i in edge[1]]
                for edge in edgePair] for edgePair in seam]
    uv_boundary = [[mesh.vt[mesh.f[edge[0]][i].vt] for i in edge[1]]
                   for edge in boundary]
    uv_foldovers = [[mesh.vt[mesh.f[edge[0]][i].vt] for i in edge[1]]
                    for edge in foldovers]
    return uv_seam, uv_boundary, uv_foldovers


if __name__ == "__main__":
    import obj_reader
    mesh = obj_reader.load_obj("../models/cube.obj")
    seam_fast = find_seam(mesh)
    logging.info("find_seam_fast.find_seam():")
    logging.info(seam_fast)
