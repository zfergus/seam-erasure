/*
Utility file for testing if points are in a given triangle.

Written by Zachary Ferguson
*/
"use strict";


function points_in_triangle(tri, points, tol){
    /*
    Test if the points are inside the triangle.
    Input:
        tri - the triangle as a matrix where the rows are the xy points.
        points - the points as a matrix where the rows are the xy points.
    Returns a vector of boolean values.
    */
    if(tol === undefined){
        tol = 1e-8;
    }
    // B is the transformation from xy to barycentric coordinates
    var B = numeric.vstack([numeric.transpose(tri), numeric.ones([1, 3])]);
    if(Math.abs(numeric.det(B)) < 1e-10){ // Degenerate triangle
        return numeric.rep([npoints], false);
    }

    var nPoints = points.length;
    var vecs = numeric.vstack([numeric.transpose(points),
        numeric.ones([1, nPoints])]);

    // Convert the grid from XY locations to barycentric coordinates.
    // This will only fail of the triangle is degenerate.
    var coords = numeric.solveMatrix(B, vecs);

    return numeric.all(numeric.geq(coords, -tol), 0);
}
