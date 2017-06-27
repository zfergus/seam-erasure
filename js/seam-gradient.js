/*
Seam gradient energy to get a better gradient energy across the seam.

Written by Zachary Ferguson
*/
"use strict";

var SeamGradient = function SeamGradient(){};


SeamGradient.A_Mat = function A_Mat(st_edge, gamma_perp, p00, p10, p01, p11, nPixels){
    /* Create a cooefficent matrix A for the equation ApT + Bp. */
    var c = gamma_perp[1] * (st_edge[1].u - st_edge[0].u) +
        gamma_perp[0] * (st_edge[1].v - st_edge[0].v);
    var coeffs = numeric.zeros([1, nPixels]);
    coeffs[0][p00] =  c;
    coeffs[0][p10] = -c;
    coeffs[0][p01] = -c;
    coeffs[0][p11] =  c;
    return coeffs;
}

SeamGradient.B_Mat = function B_Mat(st_edge, gamma_perp, p00, p10, p01, p11, nPixels){
    /* Create a cooefficent matrix B for the equation ApT + Bp. */
    var c1 = gamma_perp[1] * st_edge[0].u + gamma_perp[0] * st_edge[0].v;
    var c2 = gamma_perp[0];
    var c3 = gamma_perp[1];
    var coeffs = numeric.zeros([1, nPixels]);
    coeffs[0][p00] =  c1 - c2 - c3;
    coeffs[0][p10] = -c1 + c2;
    coeffs[0][p01] = -c1      + c3;
    coeffs[0][p11] =  c1;
    return coeffs;
}


SeamGradient.inside_perpendicular_vector = function inside_perpendicular_vector(mesh, edge){
    /*
    Returns the normalized vector in the perpendicular inside directions.
    Inputs:
        mesh - the model in OBJ format
        edge - the edge in (fi, (fv0, fv1)) format
    Output:
        Returns the appropriate perpendicular vector pointing inside the UV
        face.
    */
    var p0 = mesh.vt[mesh.f[edge[0]][edge[1][0]].vt];
    p0 = [p0.u, p0.v];
    var p1 = mesh.vt[mesh.f[edge[0]][edge[1][1]].vt];
    p1 = [p1.u, p1.v];
    var vec = numeric.sub(p1, p0);
    var triUV = mesh.f[edge[0]].map(fv => mesh.vt[fv.vt]);
    var perp;
    if(is_counterclockwise(...triUV)){
        perp = [-vec[1], vec[0]];
    }
    else{
        perp = [vec[1], -vec[0]];
    }
    var length = numeric.norm2(perp);
    return length > 1e-8 ? numeric.div(perp, length):perp;
}

SeamGradient.E_ab = function E_ab(a, b, mesh, edgePair, width, height){
    /*
    Calculate the Energy in the inverval a to b.
    Inputs:
        mesh - the model in OBJ format
        edgePair - the edgePair of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
    Output:
        Returns the energy coefficient matrix for the interval.
    */

    // Get the UV coordinates of the edge pair, swaping endpoints of one edge
    var uv_edgePair = edgePair.map(edge => edge[1].map(
        i => mesh.vt[mesh.f[edge[0]][i].vt]))
    var uv0 = uv_edgePair[0][0], uv1 = uv_edgePair[0][1],
        uv0p = uv_edgePair[1][0], uv1p = uv_edgePair[1][1];

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

    var st_edge   = globalEdge_to_local(uv0, uv1, p00, width, height);
    var st_edge_p = globalEdge_to_local(uv0p, uv1p, p00p, width, height);

    var perp_edge = SeamGradient.inside_perpendicular_vector(mesh, edgePair[0]);
    var A = SeamGradient.A_Mat(st_edge, perp_edge, 0, 1, 2, 3, 8);
    var B = SeamGradient.B_Mat(st_edge, perp_edge, 0, 1, 2, 3, 8);

    var perp_edge_p = SeamGradient.inside_perpendicular_vector(mesh, edgePair[1]);
    var Ap = SeamGradient.A_Mat(st_edge_p, perp_edge_p, 4, 5, 6, 7, 8);
    var Bp = SeamGradient.B_Mat(st_edge_p, perp_edge_p, 4, 5, 6, 7, 8);

    // Each of the A, Ap, B, Bp are 1xN matrices.
    // E is Nx1 * 1xN = NxN
    var term = function(M, n){
        /*
        Compute the integral term with constant matrix (M) and
        power n after integration.
        */
        return numeric.div(M, n * (b**n - a**n));
    };

    // Sum of matrices (1x8)
    var Asum = numeric.add(A, Ap)
    var Bsum = numeric.add(B, Bp)

    // Product of sums (8x8)
    var AA = numeric.dot(numeric.transpose(Asum), Asum);
    var BB = numeric.dot(numeric.transpose(Bsum), Bsum);
    var AB = numeric.dot(numeric.transpose(Asum), Bsum);

    var values = numeric.add(term(AA, 3.),
        term(numeric.add(AB, numeric.transpose(AB)), 2.), term(BB, 1.));

    // Cartisian Product of (indx X indx)
    var indx = [p00, p10, p01, p11, p00p, p10p, p01p, p11p];
    var ijs = numeric.transpose(product(indx, indx));

    values = flatten2D(values);
    return numeric.ccsScatterShaped([ijs[0], ijs[1], values], nPixels, nPixels);
}


SeamGradient.E_edgePair = function E_edgePair(mesh, edgePair, width, height, edge_len){
    /*
    Compute the energy coefficient matrix over a single edge pair.
    Inputs:
        mesh - the model in OBJ format
        edgePair - the edgePair of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        edge_len - the length of the edge in 3D space
    Output:
        Returns the energy coefficient matrix over a single edge pair.
    */
    var uv_edgePair = edgePair.map(edge => edge[1].map(
        i => mesh.vt[mesh.f[edge[0]][i].vt]));

    var intervals = compute_edgePair_intervals(uv_edgePair, width, height);

    var N = width * height;

    // Space for the matrix.
    // E_edge = scipy.sparse.coo_matrix((N, N))
    var E_edge = new AccumulateCOO();

    // Solve for the energy coeff matrix over the edge pair
    for(var i = 0; i < (intervals.length - 1); i++){
        // Add intervals energy to total Energy
        var a = intervals[i], b = intervals[i + 1];
        E_edge.add(SeamGradient.E_ab(a, b, mesh, edgePair, width, height));
    }

    // Multiply by the length of the edge in 3D
    return numeric.ccsmul(E_edge.total(N, N), edge_len);
}


SeamGradient.E_total = function E_total(mesh, seam, width, height, depth, edge_lens){
    /*
    Calculate the energy coeff matrix for a width x height texture.
    Inputs:
        mesh - the model in OBJ format
        seam - the seam of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        edge_lens - a list containing the lengths of each edge in 3D space.
    Output:
        Returns the quadtratic term matrix for the seam gradient.
    */
    log_output("Building Seam Gradient Matrix:");

    // Sum up the energy coefficient matrices for all the edge pairs
    var N = width * height;

    // E = scipy.sparse.coo_matrix((N, N))
    var E = new AccumulateCOO();

    var sum_edge_lens = 0.0;
    for(var i = 0; i < seam.length; i++){
        var edgePair = seam[i], edge_len = edge_lens[i];
        print_progress(i / seam.length);
        sum_edge_lens += edge_len;
        E.add(SeamGradient.E_edgePair(mesh, edgePair, width, height, edge_len));
    }
    E = E.total(N, N);

    print_progress(1.0);

    // Divide by the total edge length in 3D
    return QuadEnergy(numeric.ccsdiv(E, sum_edge_lens),
        numeric.ccsZeros(N, depth),
        numeric.ccsZeros(depth, depth));
}
