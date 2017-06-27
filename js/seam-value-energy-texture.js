/*
Solves for the energy coefficient matrix, A in x^T*A*x+Bx+C. Energy formula for
seam value energy.

Written by Zachary Ferguson
*/
"use strict";

var SVETexture = function SVETexture(){};


SVETexture.E_ab = function E_ab(a, b, edge, width, height){
    /*
    Calculate the energy in the inverval a to b.
    Parameters:
        a, b - interval to integrate over
        edge - the edge in UV-space to interpolate
        width, height - texture's dimensions
    Returns: Energy matrix for the interval
    */

    // Get the UV coordinates of the edge pair, swaping endpoints of one edge
    var uv0 = edge[0], uv1 = edge[1];

    // Determine the midpoint of the interval in UV-space
    var mid_uv = lerp_UV((a + b) / 2., uv0, uv1);

    // Determine surrounding pixel indices
    var s_pixels = surrounding_pixels(mid_uv, width, height, "index");
    var p00 = s_pixels[0], p10 = s_pixels[1], p01 = s_pixels[2],
        p11 = s_pixels[3];

    var nPixels = width * height;

    var luv = globalEdge_to_local(uv0, uv1, p00, width, height);
    var luv0 = luv[0], luv1 = luv[1];

    // Compute the coefficient matrix for the interval
    var coeffs = BilerpEnergy.bilerp_coeffMats(luv0, luv1, 0, 1, 2, 3, 4);
    var A = coeffs[0], B = coeffs[1], C = coeffs[2];

    // Each of the A, Ap, B, Bp, C, Cp are 1xN matrices.
    // Q is Nx1 * 1xN = NxN
    var term = function(M, n){return numeric.div(M, n * (b**n - a**n));};

    // Product of cooefficents (4x4)
    var AA = numeeric.dot(numeric.transpose(A), A);
    var AB = numeeric.dot(numeric.transpose(A), B);
    var AC = numeeric.dot(numeric.transpose(A), C);
    var BB = numeeric.dot(numeric.transpose(B), B);
    var BC = numeeric.dot(numeric.transpose(B), C);
    var CC = numeeric.dot(numeric.transpose(C), C);

    var values = numeric.add(term(AA, 5.),
        term(numeric.add(AB, numeric.transpose(AB)), 4.),
        term(numeric.add(AC, numeric.transpose(AC), BB), 3.),
        term(numeric.add(BC, numeric.transpose(BC)), 2.),
        term(CC, 1.));

    var indx = [p00, p10, p01, p11, p00p, p10p, p01p, p11p];
    var ijs = numeric.transpose(product(indx, indx));

    values = flatten2D(values);
    return numeric.ccsScatterShaped([ijs[0], ijs[1], values], nPixels, nPixels);
}


SVETexture.E_edge = function E_edge(edge, width, height, edge_len){
    /* Compute the energy coefficient matrix over a single edge pair. */
    var intervals = Array.from(compute_edge_intervals(uv_edge, width, height)).sort();

    var N = width * height;

    // Space for the matrix.
    // E_edge = scipy.sparse.coo_matrix((N, N))
    var E_edge = new AccumulateCOO();

    // Solve for the energy coeff matrix over the edge pair
    for(var i = 0; i < (intervals.length - 1); i++){
        // Add intervals energy to total Energy
        var a = intervals[i], b = intervals[i + 1];
        E_edge.add(E_ab(a, b, edge, width, height));
    }
    // Finally accumulate the total.
    E_edge = E_edge.total(N, N);

    // Multiply by the length of the edge in 3D
    return numeric.mul(E_edge, edge_len);
}


SVETexture.E_total = function E_total(mesh, edges, width, height, textureVec){
    /*
    Calculate the energy coeff matrix for a width x height texture.
    Inputs:
        mesh - the model in OBJ format
        edges - edges of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        textureVec - (width*height)x(depth) vector for the texture
    Output:
        QuadEnergy for the total seam value energy.
    */
    log_output("Building Seam Value Energy Matrix:");

    // Sum up the energy coefficient matrices for all the edge pairs
    var N = width * height;

    // E = scipy.sparse.coo_matrix((N, N))
    var E = new AccumulateCOO();

    var sum_edge_lens = 0.0;
    for(var i = 0; i < edges.length; i++){
        var edge = edges[i];
        print_progress(i / float(len(edges)))
        var face = mesh.f[edge[0]];
        // Calculate the 3D edge length
        var verts = edge[1].map(i => Object.values(mesh.v[mesh.f[edge[0]][i].v]));
        var edge_len = numeric.norm2(numeric.sub(verts[1], verts[0]));
        sum_edge_lens += edge_len;

        // Convert to UV edge
        var uv_edge = edge[1].map(i => mesh.vt[face[i].vt]);
        // Calculate the energy of the edge
        E.add(E_edge(uv_edge, width, height, edge_len));
    }
    E = E.total(N, N);

    print_progress(1.0);
    log_output("\n");

    // Divide by the total edge length in 3D
    var Q = numeric.ccsdiv(E, sum_edge_lens);
    var p0 = numeric.ccsSparseShaped(textureVec);
    var L = numeric.ccsmul(-1, numeric.ccsDot(SV, p0));
    var C = numeric.ccsDot(numeric.ccsTranspose(p0), numeric.ccsDot(SV, p0));
    return QuadEnergy(Q, L, C);
}
