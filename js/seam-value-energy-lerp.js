/*
Solves for the energy coefficient matrix, A in x^T*A*x+Bx+C. Energy formula for
seam value energy.

Written by Zachary Ferguson
*/
"use strict";

var SVELerp = function SVELerp(){};


SVELerp.E_ab = function E_ab(a, b, mesh, edge, width, height){
    /*
    Calculate the energy in the inverval a to b.
    Parameters:
        a, b - interval to integrate over
        mesh - 3D model in OBJ format
        edge - the edge in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
    Returns: Energy matrix for the interval
    */
    // Get the UV coordinates of the edge pair, swaping endpoints of one edge
    var uv0 = mesh.vt[mesh.f[edge[0]][edge[1][0]].vt], uv1 = mesh.vt[mesh.f[edge[0]][edge[1][1]].vt];
    var x0 = mesh.vc[mesh.f[edge[0]][edge[1][0]].v], x1 = mesh.vc[mesh.f[edge[0]][edge[1][1]].v];
    x0 = numeric.reshape(x0, [1, x0.length]);
    x1 = numeric.reshape(x1, [1, x1.length]);

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



    // Difference in endpoints
    var x1_x0 = numeric.sub(x1, x0);

    // A, B, C are 1xN and x0, x1 are 1xD
    // L is Nx1 * 1xD = NxD
    values = numeric.add(
        term(numeric.dot(numeric.transpose(A), x1_x0), 4.0),
        term(numeric.add(numeric.dot(numeric.transpose(A), x0),
                         numeric.dot(numeric.transpose(B), x1_x0)), 3.0),
        term(numeric.add(numeric.dot(numeric.transpose(B), x0),
                         numeric.dot(numeric.transpose(C), x1_x0)), 2.0),
        term(numeric.dot(numeric.transpose(C), x0), 1.0));

    var ijs = numeric.transpose(product([p00, p10, p01, p11], numeric.range(numeric.dim(x0)[1])));
    rows = ijs[0];
    cols = ijs[1];

    var L = numeric.ccsScatterShaped([rows, cols, values], nPixels, numeric.dim(x0)[1]);

    // x0, x1 are 1xD
    // C is Dx1 * 1xD = DxD
    var x1_x0x0 = numeric.dot(numeric.transpose(x1_x0), x0);

    values = numeric.add(
        term(numeric.dot(numeric.transpose(x1_x0), x1_x0), 3.0),
        term(numeric.add(x1_x0x0, numeric.transpose(x1_x0x0)), 2.0),
        term(numeric.dot(numeric.transpose(x0), x0), 1.0));

    var C = numeric.ccsSparseShaped(values);

    return [Q, L, C];
}


SVELerp.E_edge = function E_edge(mesh, edge, width, height, edge_len){
    /*
    Compute the energy coefficient matrix over a single edge.
    Inputs:
        mesh - the model in OBJ format
        edge - the edge in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
        edge_len - the length of the edge in 3D space
    Output:
        Returns the energy coefficient matrix over a single edge.
    */
    var uv_edge = edge[1].map(i => mesh.vt[mesh.f[edge[0]][i].vt]);
    var intervals = Array.from(compute_edge_intervals(uv_edge, width, height)).sort();

    var N = width * height;
    var depth = mesh.vc[0].length

    var Q_edge = new AccumulateCOO(), L_edge = new AccumulateCOO(),
        C_edge = new AccumulateCOO();

    // Solve for the energy coeff matrix over the edge pair
    for(var i = 0; i < (intervals.length - 1); i++){
        // Add intervals energy to total Energy
        var a = intervals[i], b = intervals[i + 1];
        var QLC = E_ab(a, b, mesh, edge, width, height);
        Q_edge.add(QLC[0]);
        L_edge.add(QLC[1]);
        C_edge.add(QLC[2]);
    }

    Q_edge = numeric.ccsmul(edge_len, Q_edge.total(N, N));
    L_edge = numeric.ccsmul(edge_len, L_edge.total(N, depth));
    C_edge = numeric.ccsmuk(edge_len, C_edge.total(depth, depth));

    // Multiply by the length of the edge in 3D
    return [Q_edge, L_edge, C_edge];
}


SVELerp.E_total = function E_total(mesh, edges, width, height){
    /*
    Calculate the energy coeff matrix for a width x height texture.
    Inputs:
        mesh - the model in OBJ format
        edges - edges of the model in (fi, (fv0, fv1)) format
        width, height - texture's dimensions
    Assume:
        depth == len(mesh.vc[0])
    Output:
        Returns the quadtratic term matrix for the seam value energy.
    */
    log_output("Building Seam Value of Lerp Energy Matrix:");

    // Check the model contains vertex colors.
    if(len(mesh.vc) != len(mesh.v)){
        console.error("Mesh does not contain an equal number vertex " +
            "colors and vertices.")
        return;
    }

    // Sum up the energy coefficient matrices for all the edge pairs
    var N = width * height;
    var depth = mesh.vc[0].length;

    var Q = new AccumulateCOO();
    var L = new AccumulateCOO();
    var C = new AccumulateCOO();

    var sum_edge_lens = 0.0;
    for(var i = 0; i < edges.length; i++){
        var edge = edges[i];
        print_progress(i / edges.length);
        // Calculate the 3D edge length
        var verts = edge[1].map(i => Object.values(mesh.v[mesh.f[edge[0]][i].v]));
        var edge_len = numeric.norm2(numeric.sub(verts[1], verts[0]));
        sum_edge_lens += edge_len;

        // Compute the QuadEnergy of the edge.
        var QeLeCe = E_edge(mesh, edge, width, height, edge_len);
        Q.add(QeLeCe[0]);
        L.add(QeLeCe[1]);
        C.add(QeLeCe[2]);
    }

    Q = numeric.ccsdiv(Q.total(N, N), sum_edge_lens);
    L = numeric.ccsdiv(L.total(N, depth), sum_edge_lens);
    C = numeric.ccsdic(C.total(depth, depth), sum_edge_lens);

    print_progress(1.0);
    log_output("\n");

    // Divide by the total edge length in 3D
    return QuadEnergy(Q, L, C);
}
