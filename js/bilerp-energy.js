/*
Solves for the energy coefficient matrix, A in x^T*A*x+Bx+C. Energy formula for
bilinear interpolation.

Written by Zachary Ferguson
*/
"use sctrict";

var BilerpEnergy = function BilerpEnergy(){};

BilerpEnergy.A_Mat = function A_Mat(uv0, uv1, p00, p10, p01, p11, nPixels){
    // Create a cooefficent A matrix.
    var c1 = (uv1.u - uv0.u) * (uv1.v - uv0.v);
    var coeffs = numeric.zeros([1, nPixels]);
    coeffs[0][p00] =  c1;
    coeffs[0][p10] = -c1;
    coeffs[0][p01] = -c1;
    coeffs[0][p11] =  c1;
    return coeffs;
}


BilerpEnergy.B_Mat = function B_Mat(uv0, uv1, p00, p10, p01, p11, nPixels){
    // Create a cooefficent B matrix.
    var c1 = (uv1.u - uv0.u) * uv0.v + uv0.u * (uv1.v - uv0.v);
    var c2 = uv1.u - uv0.u;
    var c3 = uv1.v - uv0.v;
    var coeffs = numeric.zeros([1, nPixels]);
    coeffs[0][p00] =  c1 - c2 - c3;
    coeffs[0][p10] = -c1 + c2;
    coeffs[0][p01] = -c1      + c3;
    coeffs[0][p11] =  c1;
    return coeffs
}


BilerpEnergy.C_Mat = function C_Mat(uv0, uv1, p00, p10, p01, p11, nPixels){
    // Create a cooefficent C matrix.
    var c1 = uv0.u * uv0.v;
    var c2 = uv0.u;
    var c3 = uv0.v;
    var c4 = 1;
    var coeffs = numeric.zeros([1, nPixels]);
    coeffs[0][p00] =  c1 - c2 - c3 + c4;
    coeffs[0][p10] = -c1 + c2;
    coeffs[0][p01] = -c1      + c3;
    coeffs[0][p11] =  c1;
    return coeffs;
}


BilerpEnergy.bilerp_coeffMats = function bilerp_coeffMats(uv0, uv1, p00, p10, p01, p11, nPixels){
    // Compute the matrix coefficients for a bilinear interpolation.

    // Compute A, AP
    var A  = BilerpEnergy.A_Mat(uv0, uv1, p00, p10, p01, p11, nPixels);

    // Compute B, Bp
    var B  = BilerpEnergy.B_Mat(uv0, uv1, p00, p10, p01, p11, nPixels);

    // Compute C, Cp
    var C  = BilerpEnergy.C_Mat(uv0, uv1, p00, p10, p01, p11, nPixels);

    return [A, B, C];
}


BilerpEnergy.E_ab = function E_ab(a, b, edgePair, width, height){
    /*
    Calculate the energy in the inverval a to b.
    Parameters:
        a, b - interval to integrate over
        edgePair - the edge pair to interpolate
        width, height - texture's dimensions
    Returns: Energy matrix for the interval
    */

    // Get the UV lilrdinates of the edge pair, swaping endpoints of one edge
    var uv0 = edgePair[0][0], uv1 = edgePair[0][1], uv0p = edgePair[1][0],
        uv1p = edgePair[1][1];

    // Determine the midpoint of the interval in UV-space
    var mid_uv   = lerp_UV((a + b) / 2., uv0, uv1);
    var mid_uv_p = lerp_UV((a + b) / 2., uv0p, uv1p);

    // Determine surrounding pixel indices
    var s_pixels = surrounding_pixels(mid_uv, width, height, "index");
    var p00 = s_pixels[0], p10 = s_pixels[1], p01 = s_pixels[2],
        p11 = s_pixels[3];
    s_pixels = surrounding_pixels(mid_uv_p, width, height, "index");
    var p00p = s_pixels[0], p10p = s_pixels[1], p01p = s_pixels[2],
        p11p = s_pixels[3];

    var nPixels = width * height;

    var luv = globalEdge_to_local(uv0, uv1, p00, width, height);
    var luv0 = luv[0], luv1 = luv[1];
    var luvp = globalEdge_to_local(uv0p, uv1p, p00p, width, height);
    var luv0p = luvp[0], luv1p = luvp[1];

    // Compute the coefficient matrix for the interval
    var coeffs = BilerpEnergy.bilerp_coeffMats(luv0, luv1, 0, 1, 2, 3, 8);
    var A = coeffs[0], B = coeffs[1], C = coeffs[2];
    coeffs = BilerpEnergy.bilerp_coeffMats(luv0p, luv1p, 4, 5, 6, 7, 8);
    var Ap = coeffs[0], Bp = coeffs[1], Cp = coeffs[2];

    // Each of the A, Ap, B, Bp, C, Cp are 1xN matrices.
    // E is Nx1 * 1xN = NxN
    var term = function(M, n){
        /*
        Compute the integral term with constant matrix (M) and power n after
        integration.
        */
        return numeric.div(M, n * (b**n - a**n));
    };

    // Difference of matrices (1x8)
    var Adiff = numeric.sub(A, Ap);
    var Bdiff = numeric.sub(B, Bp);
    var Cdiff = numeric.sub(C, Cp);

    // Product of differences (8x8)
    var AA = numeric.dot(numeric.transpose(Adiff), Adiff);
    var BB = numeric.dot(numeric.transpose(Bdiff), Bdiff);
    var CC = numeric.dot(numeric.transpose(Cdiff), Cdiff);
    var AB = numeric.dot(numeric.transpose(Adiff), Bdiff);
    var AC = numeric.dot(numeric.transpose(Adiff), Cdiff);
    var BC = numeric.dot(numeric.transpose(Bdiff), Cdiff);

    var values = numeric.add(term(AA, 5.),
        term(numeric.add(AB, numeric.transpose(AB)), 4.),
        term(numeric.add(AC, numeric.transpose(AC), BB), 3.),
        term(numeric.add(BC, numeric.transpose(BC)), 2.),
        term(CC, 1.));

    // Cartisian Product of (indx X indx)
    var indx = [p00, p10, p01, p11, p00p, p10p, p01p, p11p];
    var ijs = numeric.transpose(product(indx, indx));

    values = flatten2D(values);
    return numeric.ccsScatterShaped([ijs[0], ijs[1], values], nPixels, nPixels);
}


BilerpEnergy.E_edgePair = function E_edgePair(edgePair, width, height, edge_len){
    /* Compute the energy coefficient matrix over a single edge pair. */
    var intervals = compute_edgePair_intervals(edgePair, width, height);

    var N = width * height;

    // Space for the matrix.
    // E_edge = scipy.sparse.coo_matrix((N, N))
    var E_edge = new AccumulateCOO();

    // Solve for the energy coeff matrix over the edge pair
    // for a, b in pairwise(intervals):
    for(var i = 0; i < (intervals.length - 1); i++){
        // Add intervals energy to total Energy
        var a = intervals[i], b = intervals[i + 1];
        E_edge.add(BilerpEnergy.E_ab(a, b, edgePair, width, height));
    }

    // Multiply by the length of the edge in 3D
    return numeric.ccsmul(E_edge.total(N, N), edge_len);
}

BilerpEnergy.E_total = function E_total(seam, width, height, depth, edge_lens){
    /* Calculate the energy coeff matrix for a width x height texture. */
    log_output("Building Bilinear Energy Matrix:");

    // Sum up the energy coefficient matrices for all the edge pairs
    var N = width * height;

    // E = scipy.sparse.coo_matrix((N, N))
    var E = new AccumulateCOO();

    var sum_edge_lens = 0.0;
    for(var i = 0; i < seam.length; i++){
        var edgePair = seam[i], edge_len = edge_lens[i];
        print_progress(i / seam.length);
        sum_edge_lens += edge_len;
        E.add(BilerpEnergy.E_edgePair(edgePair, width, height, edge_len));
    }
    E = E.total(N, N)

    print_progress(1.0);

    // Divide by the total edge length in 3D
    return QuadEnergy(numeric.ccsdiv(E, sum_edge_lens),
        numeric.ccsZeros(N, depth),
        numeric.ccsZeros(depth, depth));
}
