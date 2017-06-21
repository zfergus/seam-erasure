/*
Compute the Least Squares Constraints for the seams.

Written by Zachary Ferguson
*/
"use strict";

var LSQConstraints = function LSQConstraints(){};


LSQConstraints.constrain_values = function constrain_values(mask, textureVec){
    /*
        Constrain the values inside the triangles
        Inputs:
            mask - a height by width matrix of boolean values for if
                pixel r,c is to be constrained
            textureVec - a vector for the texture pixel values
        Output:
            A least square constrained matrices in the form (C, d)
    */
    var dims = numeric.dim(mask);
    var height = dims[0], width = dims[1];
    var N = width * height;
    dims = numeric.dim(textureVec);
    var depth = dims.length > 1 ? dims[1] : 1;

    var values = numeric.asNumbers(flatten2D(mask));

    var C = numeric.ccsDiag(values, N, N);
    var d = numeric.ccsDot(C, numeric.ccsSparseShaped(textureVec));

    var norm = numeric.sum(mask)
    if(norm !== 0){
        norm = 1.0 / norm;
    }

    // The matrix should be C.T*C, but C is a diagonal binary matrix so it
    // doesn't matter.
    return QuadEnergy(numeric.ccsmul(norm, C), numeric.ccsmul(-norm, d),
        numeric.ccsmul(norm, numeric.ccsDot(numeric.ccsTranspose(d), d)));
}
