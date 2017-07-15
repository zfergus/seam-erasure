/*
Find the edges on the seam of a 3D model.

Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
        Zachary Ferguson <zfergus2@gmu.edu>

License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)
*/
"use strict";

var FindSeam = function FindSeam(){};


FindSeam.edges_equal_in_UV = function edges_equal_in_UV(mesh, forwards, backwards){
    /* Determine if the UV edges are equal. */
    var faceEdge = [mesh.f[forwards[0]][forwards[1][0]], mesh.f[forwards[0]][forwards[1][1]]];
    var uvEdge0 = faceEdge.map(fv => mesh.vt[fv.vt]);

    faceEdge = [mesh.f[backwards[0]][backwards[1][1]], mesh.f[backwards[0]][backwards[1][0]]];
    var uvEdge1 = faceEdge.map(fv => mesh.vt[fv.vt]);

    return numeric.equal(uvEdge0, uvEdge1);
}


FindSeam.orientation = function orientation(a, b, c){
    /* Computes the orientation of c relative to the line AB */
    var M = [[a[0] - c[0], a[1] - c[1]],
         [b[0] - c[0], b[1] - c[1]]];
    // Return det(M)
    return M[0][0] * M[1][1] - M[0][1] * M[1][0];
}


FindSeam.edge_is_foldover = function edge_is_foldover(mesh, forwards, backwards){
    /*
    Check for foldovers in UV space.
    Given two triangles (a,b,c) and (a,b,d) the goal is to determine if c and d
    lie on opposite sides of the line passing through ab. This is known as the
    3-point "orientation test".
    REF: https://www.cs.cmu.edu/~quake/robust.html
    Parameters:
        mesh - the mesh in obj_reader.OBJ recordclass format
        forward, backwards - the forwards and backwards edge s.t.
            forward == backwards[::-1]
    */
    var missing_index = function(edge){
        var possible_indices = [0, 1, 2];
        for(var i = 0; i < possible_indices.length; i++){
            if(edge.indexOf(possible_indices[i]) < 0){
                return possible_indices[i];
            }
        }
        return -1;
    }

    // Get all points a, b, c, d that make up the two triangular faces
    var a = mesh.vt[mesh.f[forwards[0]][forwards[1][0]].vt]
    var b = mesh.vt[mesh.f[forwards[0]][forwards[1][1]].vt]
    var c = mesh.vt[mesh.f[forwards[0]][missing_index(forwards[1])].vt]
    var d = mesh.vt[mesh.f[backwards[0]][missing_index(backwards[1])].vt]

    var orientation1 = FindSeam.orientation(a, b, c)
    var orientation2 = FindSeam.orientation(a, b, d)

    // If both points c and d are on the same side of line AB, then AB is a
    // foldover edge.
    return Math.abs(Math.sign(orientation1) - Math.sign(orientation2)) < 1e-8
}


FindSeam.find_seam = function find_seam(mesh){
    /*
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
            //// The edge appears twice, once forwards and once backwards.
            forwards, backwards = edge_both_directions
            //// They can be used to find the start and end position indices:
            forwards_position_start, forwards_position_end =
                face_position_indices[ forwards ]
            backwards_position_start, backwards_position_end =
                face_position_indices[ backwards ]
            //// Or the texcoords:
            forwards_texcoord_start, forwards_texcoord_end =
                face_texcoord_indices[ forwards ]
            backwards_texcoord_start, backwards_texcoord_end =
                face_texcoord_indices[ backwards ]
            //// The positions will match, but not the texcoords:
            assert forwards_position_start, forwards_position_end ==
                backwards_position_end, backwards_position_start
            assert forwards_texcoord_start, forwards_texcoord_end !=
                backwards_texcoord_end, backwards_texcoord_start

            //// You can also use an index bundle to index into the original
            //// mesh.f list:
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
        //// The edge can be used to find the start and end position indices:
        position_start, position_end = face_position_indices[ edge ]
        //// Or the texcoords:
        texcoord_start, texcoord_end = face_texcoord_indices[ edge ]

        //// You can use the index bundle to index into the original mesh.f list:
        face_vertex_start = mesh.f[ edge[0] ][ edge[1][0] ]
        face_vertex_end = mesh.f[ edge[0] ][ edge[1][1] ]
    */

    var face_position_indices = mesh.f.map(face => face.map(fv => fv.v));

    var face_texcoord_indices = mesh.f.map(face => face.map(fv => fv.vt));

    var vp = mesh.v;
    var vt = mesh.vt;

    // A map from a pair of vertex indices to the index (face and endpoints)
    // into face_position_indices.
    // The following should be true for every key, value pair:
    //    key == face_position_indices[ value ]
    // This gives us a "reverse map" so that we can look up other face
    // attributes based on position edges.
    // The value are written in the format returned by numpy.where(),
    // which stores multi-dimensional indices such as array[a0,b0], array[a1,b1]
    // as ( (a0,a1), (b0,b1) ).
    var directed_position_edge2face_position_index = {};
    for(var fi = 0; fi < face_position_indices.length; fi++){
        var face = face_position_indices[fi];
        directed_position_edge2face_position_index[JSON.stringify([face[0], face[1]])] = [fi, [0, 1]];
        directed_position_edge2face_position_index[JSON.stringify([face[1], face[2]])] = [fi, [1, 2]];
        directed_position_edge2face_position_index[JSON.stringify([face[2], face[0]])] = [fi, [2, 0]];
    }

    // First find all undirected position edges (collect a canonical orientation
    // of the directed edges).
    var keys = Object.keys(directed_position_edge2face_position_index);
    var undirected_position_edges = new Set();
    for(var i = 0; i < keys.length; i++){
        var ij = JSON.parse(keys[i]);
        var minIJ = Math.min(ij[0], ij[1]);
        var maxIJ = Math.max(ij[0], ij[1]);
        undirected_position_edges.add(JSON.stringify([minIJ, maxIJ]));
    }
    undirected_position_edges = Array.from(undirected_position_edges).map(JSON.parse);

    // Now we will iterate over all position edges.
    // Seam edges are the edges whose two opposite directed edges have different
    // texcoord indices (or one doesn't exist at all in the case of a mesh
    // boundary).
    var seam_non_matching = [];
    var seam_mesh_boundary = [];
    var seam_foldover_edges = [];
    for(var i = 0; i < undirected_position_edges.length; i++){
        var vp_edge = undirected_position_edges[i];
        // If it and its opposite exist as directed edges, check if their
        // texture coordinate indices match.
        if(directed_position_edge2face_position_index.hasOwnProperty(JSON.stringify(vp_edge)) &&
            directed_position_edge2face_position_index.hasOwnProperty(JSON.stringify(vp_edge.slice().reverse()))){
            var forwards = directed_position_edge2face_position_index[JSON.stringify(vp_edge)]
            var backwards = directed_position_edge2face_position_index[JSON.stringify(vp_edge.slice().reverse())]

            // If the texcoord indices are similarly flipped or the edges are
            // equivilant in UV-space.
            var forwards_texcoord_indices = [
                face_texcoord_indices[forwards[0]][forwards[1][0]],
                face_texcoord_indices[forwards[0]][forwards[1][1]]];
            var backwards_texcoord_indices = [
                face_texcoord_indices[backwards[0]][backwards[1][0]],
                face_texcoord_indices[backwards[0]][backwards[1][1]]];
            if(numeric.equal(forwards_texcoord_indices,
                backwards_texcoord_indices.slice().reverse()) ||
                FindSeam.edges_equal_in_UV(mesh, forwards, backwards)){
                // Check for foldovers in UV space.
                if(FindSeam.edge_is_foldover(mesh, forwards, backwards)){
                    // Add to a third list of uv foldover edges.
                    seam_foldover_edges.push(forwards);
                }
                continue;
            }

            // Otherwise, we have a non-matching seam edge.
            seam_non_matching.push([forwards, backwards]);
        }
        // Otherwise, the edge and its opposite aren't both in the directed
        // edges. One of them should be.
        else if(directed_position_edge2face_position_index.hasOwnProperty(JSON.stringify(vp_edge))){
            seam_mesh_boundary.push(directed_position_edge2face_position_index[vp_edge]);
        }
        else if(directed_position_edge2face_position_index.hasOwnProperty(JSON.stringify(vp_edge.slice().reverse()))){
            seam_mesh_boundary.push(directed_position_edge2face_position_index[vp_edge.slice().reverse()]);
        }
    }

    return [seam_non_matching, seam_mesh_boundary, seam_foldover_edges];
}


FindSeam.seam_to_UV = function seam_to_UV(mesh, seam, boundary, foldovers){
    /*
    Convert the seam, boundary, and foldovers to be in UV-space.
    Input  Format:
        seam      = [[[fi, [fvi0, fvi1]], [fi', [fvi0', fvi1']]], ...]
        boundary  = [ [fi0, [fvi0, fvi1]], ...]
        foldovers = [ [fi0, [fvi0, fvi1]], ...]
    Return Format:
        ( [[[uv0, uv1], [uv0', uv1']], ...],
          [ [uv0, uv1], ...],
          [ [uv0, uv1], ...] )
    */

    var uv_seam = seam.map(edgePair => edgePair.map(edge => edge[1].map(
        i => mesh.vt[mesh.f[edge[0]][i].vt])));

    var uv_boundary = boundary.map(edge => edge[1].map(
        i => mesh.vt[mesh.f[edge[0]][i].vt]));

    var uv_foldovers = foldovers.map(edge => edge[1].map(
        i => mesh.vt[mesh.f[edge[0]][i].vt]))

    return [uv_seam, uv_boundary, uv_foldovers];
}
