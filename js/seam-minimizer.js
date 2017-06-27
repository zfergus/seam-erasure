/*
Minimize the energy difference of a textures "outside" pixels.

Written by Zachary Ferguson
*/
"use strict";

var SeamMinimizer = function SeamMinimizer(){};


// energies_str = "BLE, SV, SG, LSQ, L"
var Energies = function(BLE, SV, SG, LSQ, L){
    return {"BLE": BLE, "SV": SV, "SG": SG, "LSQ": LSQ, "L": L};
}


var SeamValueMethod = function(){};

/* Enum for the seam value methods. */
SeamValueMethod.NONE = 0;
SeamValueMethod.TEXTURE = 1;
SeamValueMethod.LERP = 2;

SeamValueMethod.compute_energy = function compute_energy(method, mesh, edges, width, height, textureVec){
    if(method === SeamValueMethod.NONE){
        log_output("=== Not using Seam Value Energy ===\n");
        return undefined;
    }
    else if(method === SeamValueMethod.TEXTURE){
        return SVETexture.E_total(mesh, edges, width, height, textureVec);
    }
    else if(method === SeamValueMethod.LERP){
        return SVELerp.E_total(mesh, edges, width, height);
    }
    // else{
    //     raise ValueError("Invalid method of computing Seam Value Energy.")
}


// def display_quadratic_energy(coeffs, x0, x, name, out=sys.stdout):
//     """ Compute the energy of a solution given the coefficents. """
//     print("%s Before After" % name)
//     E0 = x0.T.dot(coeffs.Q.dot(x0)) + 2.0 * x0.T.dot(coeffs.L.A) + coeffs.C.A
//     E = x.T.dot(coeffs.Q.dot(x)) + 2.0 * x.T.dot(coeffs.L.A) + coeffs.C.A
//     depth = (x.shape + (1,))[1]
//     for i in range(depth):
//         print("%d %g %g" % (i, E0[i] if depth < 2 else E0[i, i],
//             E[i] if depth < 2 else E[i, i]))
//
//
// def display_energies(energies, x0, x, out=sys.stdout):
//     """
//     Display the bilinear and Dirichlet energies.
//     Inputs:
//         energies - an Energies object for the coefficents of the quadratic
//             energies.
//         x0 - original vector
//         x - solution vector
//     """
//     // LSQ = QuadEnergy(2 * energies.LSQ.Q, energies.LSQ.L, energies.LSQ.C)
//     names = ["Bilinear_Energy", "Seam_Value_Energy", "Seam_Gradient_Energy",
//         "Least_Squares_Energy", "Dirichlet_Energy"]
//     coeffs = [energies.BLE, energies.SV, energies.SG, energies.LSQ, energies.L]
//     for name, energy in zip(names, coeffs):
//         if(energy):
//             display_quadratic_energy(energy, x0, x, name, out=out)
//         else:
//             print("%s\nN/a" % name)
//         print("")


function compute_seam_lengths(mesh, seam){
    /* Calculate the length, in 3D, of all edges on the seam. */
    return seam.map(function(edgePair){
        var fi = edgePair[0][0], fv0 = edgePair[0][1][0], fv1 = edgePair[0][1][1];
        var v0 = Object.values(mesh.v[mesh.f[fi][fv0].v]);
        var v1 = Object.values(mesh.v[mesh.f[fi][fv1].v]);
        return numeric.norm2(numeric.sub(v1, v0));
    });
}

SeamMinimizer.compute_energies = function compute_energies(mesh, texture, sv_method){
    /*
        Minimize the difference between the values of cooresponding edges,
        edge pairs.
        Parameters:
            mesh - a OBJ recordclass for the mesh
            texture - a height x width x depth numpy array of texture values
        Returns:
            Returns a Energies object containing the coefficents for the
            quadtratic energies.
    */
    if(sv_method === undefined){
        sv_method = SeamValueMethod.NONE;
    }

    var dims = numeric.dim(texture);
    var height = dims[0], width = dims[1], depth = dims.length > 2 ? dims[2]:1;
    var N = width * height;
    var textureVec = [];
    // var textureVec = numeric.reshape(texture, [N, depth]); // SLOW!!!
    for(var i = 0; i < texture.length; i++){
        for(var j = 0; j < texture[i].length; j++){
            textureVec.push(texture[i][j]);
        }
    }

    log_output("Finding seam of model");
    var sbf = FindSeam.find_seam(mesh);
    var seam = sbf[0], boundary = sbf[1], foldovers = sbf[2];
    var uv_sbf = FindSeam.seam_to_UV(mesh, seam, boundary, foldovers);
    var uv_seam = uv_sbf[0], uv_boundary = uv_sbf[1], uv_foldovers = uv_sbf[2];
    log_output("Done\n")

    log_output("Number of edges along the seam: " + (seam.length * 2));
    log_output("Number of edges along the boundary: " + boundary.length);
    log_output("Number of foldover edges: " + foldovers.length + "\n");

    log_output("Computing seam edge lengths");
    var edge_lens = compute_seam_lengths(mesh, seam);
    log_output("Done\n");

    // Calculate the energy coeff matrix
    var BLE = BilerpEnergy.E_total(uv_seam, width, height, depth, edge_lens);
    log_output("");

    var SG = SeamGradient.E_total(mesh, seam, width, height, depth, edge_lens);
    log_output("");

    var bag_of_F_edges = [];
    seam.forEach(edgePair => edgePair.forEach(
        edge => bag_of_F_edges.push(edge)));
    bag_of_F_edges = bag_of_F_edges.concat(boundary, foldovers);

    var SV = SeamValueMethod.compute_energy(sv_method, mesh, bag_of_F_edges,
        width, height, textureVec);

    // All edges unsorted
    var bag_of_UV_edges = [];
    uv_seam.forEach(edgePair => edgePair.forEach(
        edge => bag_of_UV_edges.push(edge)));
    bag_of_UV_edges = bag_of_UV_edges.concat(uv_boundary, uv_foldovers);

    // Constrain the values
    log_output("Building Least Squares Constraints:");
    var lsq_mask = Mask.mask_inside_seam(mesh, bag_of_UV_edges, width, height);
    var LSQ = LSQConstraints.constrain_values(lsq_mask, textureVec);
    log_output("");

    // Construct a dirichlet energy for the texture.
    log_output("Building Dirichlet Energy Mask:");
    var dirichlet_mask = Mask.mask_inside_faces(mesh, width, height,
        numeric.not(lsq_mask));
    log_output("\nBuilding Dirichlet Energy:");
    var L = Dirichlet.dirichlet_energy(height, width, textureVec,
        numeric.not(dirichlet_mask), lsq_mask);
    log_output("");

    return Energies(BLE, SV, SG, LSQ, L);
}


SeamMinimizer.solve_seam = function solve_seam(mesh, texture, sv_method, do_global){
    /*
    Solves for the minimized seam values.
    Returns the minimized texture as a numpy array, shape = (N, depth)
    */
    if(sv_method === undefined){
        sv_method = SeamValueMethod.NONE;
    }
    if(do_global === undefined){
        do_global = false;
    }

    var dims = numeric.dim(texture);
    var height = dims[0], width = dims[1], depth = dims.length > 2 ? dims[2]:1;
    var N = width * height;

    // Get the coefficients for the quadratic energies.
    var energies = SeamMinimizer.compute_energies(mesh, texture, sv_method);
    // BLE, SV, SG, LSQ, LSQ1, LSQ2, L = energies // WARNING: Do not change order
    var BLE = energies.BLE, SV = energies.SV, SG = energies.SG,
        LSQ = energies.LSQ, L = energies.L;

    log_output("Solving for minimal energy solution");

    // Minimize energy with constraints (Quad * x = lin)
    // Weights in the order [bleW, svW, sgW, lsqW, diriW]
    var weights = do_global ?
        [1e10, 1e2, 1e2, 1e2, 1e0]:[1e10, 1e2, 1e2, 1e4, 1e0];

    var quad = new AccumulateCOO(); // Quadratic term
    var lin = new AccumulateCOO(); // Linear term
    for(var i = 0; i < weights.length; i++){
        var weight = weights[i], E = [BLE, SV, SG, LSQ, L][i];
        if(E !== undefined){
            quad.add(numeric.ccsmul(weight, E.Q));
            lin.add(numeric.ccsmul(weight, E.L));
        }
    }
    quad = quad.total(N, N);
    lin = lin.total(N, depth);

    var solution = numeric.ccsSolveMatrix(quad, numeric.ccsmul(-1, lin));

    log_output("Done\n");

    return numeric.ccsFull(solution);
}
