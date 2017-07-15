/*
Create a boolean mask for if the pixels of a texture is constrained, True, or
free, False.

Written by Zachary Ferguson
*/
"use strict";

var Mask = function Mask(){};

Mask.get_all_surrounding_pixels = function get_all_surrounding_pixels(edges, width, height){
    /*
    Get a set of all pixels surrounding the given edges.
    Input:
        edges  - unsorted list of UV edges
        width  - width of the texture
        height - height of the texture
    Output:
        Returns a set of (X, Y) coordinates for the surrounding pixels.
    */
    var pixels = new Set();
    for(var i = 0; i < edges.length; i++){
        var edge = edges[i];
        var xy0 = UV_to_XY(edge[0], width, height);
        var xy1 = UV_to_XY(edge[1], width, height);
        var intervals = Array.from(compute_edge_intervals(edge, width, height)).sort();
        // Find all pixels along the seam
        for(var j = 0; j < intervals.length - 1; j++){
            var a = intervals[j], b = intervals[j + 1];
            var uv_mid = lerp_UV((a + b) / 2.0, edge[0], edge[1])
            surrounding_pixels(uv_mid, width, height, "array").forEach(
                p => pixels.add(JSON.stringify(p)));
        }
    }

    return Array.from(pixels).map(JSON.parse);
}


Mask.mask_seam = function mask_seam(mesh, seam_edges, width, height){
    /*
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
    */
    // Store the surrounding pixels XY coords in pixels
    var seam_pixels = Mask.get_all_surrounding_pixels(seam_edges, width, height);

    var vals = numeric.ones([seam_pixels.length]);
    var rows = numeric.slice(seam_pixels, [':', 1]); // Y-values of the pixels
    var cols = numeric.slice(seam_pixels, [':', 0]); // X-values of the pixels
    var mask = numeric.ccsFull(numeric.ccsScatterShaped([rows, cols, vals], height, width));

    return numeric.not(mask);
}


Mask.mask_inside_seam = function mask_inside_seam(mesh, seam_edges, width, height){
    /*
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
    */
    // Store the surrounding pixels XY coords in pixels
    var seam_pixels = Mask.get_all_surrounding_pixels(seam_edges, width, height);

    // Create a list of the UV faces in Pixel space
    var faces = mesh.f.map(face => face.map(
        fv => (p => [p.x, p.y])(UV_to_XY(mesh.vt[fv.vt], width, height))));

    // This mask should be small enough for a dense matrix
    var mask = numeric.zeros([height, width])

    // Constrain all the pixels in seam_pixels that are inside a face
    var pts = seam_pixels;
    for(var i = 0; i < faces.length; i++){
        var face = faces[i];
        print_progress(i / faces.length);

        // Create a bounding box for the face
        var ll = [Math.min(...numeric.slice(face, [':', 0])),
            Math.min(...numeric.slice(face, [':', 1]))];
        var ur = [Math.max(...numeric.slice(face, [':', 0])),
            Math.max(...numeric.slice(face, [':', 1]))];

        // Intersect the bounding box with the seam pixels
        var inidx = numeric.all(numeric.and(numeric.leq(ll, pts), numeric.leq(pts, ur)), 1);
        var inbox = numeric.mask(pts, inidx);

        // Only test seam pixels inside the bounding_box
        if(inbox.length > 0){
            // TODO: This could be done better with Numeric
            var are_inside = points_in_triangle(face, inbox);
            for(var j = 0; j < inbox.length; j++){
                mask[inbox[j][1]][inbox[j][0]] =
                    mask[inbox[j][1]][inbox[j][0]] || are_inside[j];
            }
        }
    }
    // Mask is False if pixels inside (this needs to be inverted).
    var vals = numeric.ones([seam_pixels.length]);
    var rows = numeric.slice(seam_pixels, [':', 1]); // Y-values of the pixels
    var cols = numeric.slice(seam_pixels, [':', 0]); // X-values of the pixels
    var full = numeric.ccsFull(numeric.ccsScatterShaped([rows, cols, vals], height, width));
    mask = numeric.sub(full, mask);

    print_progress(1.0);
    return numeric.not(mask)
}


Mask.mask_inside_faces = function mask_inside_faces(mesh, width, height, init_mask){
    /*
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
    */
    // Create a list of the UV faces in Pixel space
    var faces = mesh.f.map(face => face.map(
        fv => (p => [p.x, p.y])(UV_to_XY(mesh.vt[fv.vt], width, height))));

    // This mask should be small enough for a dense matrix
    var mask = init_mask;
    if(mask === undefined){
        mask = numeric.zeros([height, width]);
    }

    for(var i = 0; i < faces.length; i++){
        var face = faces[i];
        print_progress(i / faces.length); // Progress tracker

        // Bounding box for the face to get surrounding pixels.
        var ll = [Math.min(...numeric.slice(face, [':', 0])),
            Math.min(...numeric.slice(face, [':', 1]))];
        var ur = [Math.max(...numeric.slice(face, [':', 0])),
            Math.max(...numeric.slice(face, [':', 1]))];

        var bbox = [ll, ur];

        var xRange = numeric.range(
            Math.max(0,     Math.floor(bbox[0][0])),
            Math.min(width, Math.ceil(bbox[1][0]) + 1));
        var yRange = numeric.range(
            Math.max(0,      Math.floor(bbox[0][1])),
            Math.min(height, Math.ceil(bbox[1][1]) + 1));

        var inbox = product(xRange, yRange);

        // Test inside face for all pixels in the bounding box
        if(inbox.length > 0){
            // TODO: This could be done better with Numeric
            // mask[inbox[:, 1], inbox[:, 0]] |= points_in_triangle(face, inbox)
            var are_inside = points_in_triangle(face, inbox);
            for(var j = 0; j < inbox.length; j++){
                mask[inbox[j][1]][inbox[j][0]] =
                    mask[inbox[j][1]][inbox[j][0]] || are_inside[j];
            }
        }
    }

    print_progress(1.0);
    return numeric.not(mask);
}
