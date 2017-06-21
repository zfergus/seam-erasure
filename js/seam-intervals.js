/*
Computes the intersection of seam edges with pixels of a width x height
texture.

Written by Zachary Ferguson
*/
"use strict";


function compute_edge_intervals(edge, width, height){
    /* Compute the intervals along an edge. */
    var intervals = new Set([0, 1]);

    var v0 = UV_to_XY(edge[0], width, height);
    var v1 = UV_to_XY(edge[1], width, height);

    // Create expressions for converting to t values
    var x1_x0 = v1.x - v0.x;
    var y1_y0 = v1.y - v0.y;

    var x_to_t = function(x){ return round((x - v0.x) / (x1_x0), 10); };
    var y_to_t = function(y){ return round((y - v0.y) / (y1_y0), 10); };

    // Add whole number pixels to t values
    range_min_max(Math.ceil(v0.x), Math.ceil(v1.x)).forEach(function(x){
        intervals.add(x_to_t(x))
    });

    range_min_max(Math.ceil(v0.y), Math.ceil(v1.y)).forEach(function(y){
        intervals.add(y_to_t(y))
    });

    return intervals;
}


function compute_edgePair_intervals(edgePair, width, height){
    var intervals = new Set();
    for(var i = 0; i < edgePair.length; i++){
        var edge = edgePair[i];
        compute_edge_intervals(edge, width, height).forEach(function(x){
            intervals.add(x);
        });
    }
    return Array.from(intervals).sort();
}


function compute_seam_intervals(uv_seam, width, height){
    /* Computes all intervals from 0 to 1 on the seam's edge pairs. */
    return uv_seam.map(
        edgePair => compute_edgePair_intervals(edgePair, width, height));
}
