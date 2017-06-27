/*
Author: Yotam Gingold <yotam (strudel) yotamgingold.com>
License: Public Domain [CC0](http://creativecommons.org/publicdomain/zero/1.0/)

Let the energy E = | G*x - U |^2_M
where:
    x is a row-by-col image flattened into a vector with row-by-col entries.
    G is the gradient operator,
    U are the target gradients (possibly U = G*y for some known function y),
    and M is the diagonal mass matrix which is the norm of each derivative in
    the summation.
Then E = (G*x - U).T * M * (G*x - U) = x.T*G.T*M*G*x + U.T*M*U -
            2*x.T*G.T*M*U
Let L = G.T*M*G.
Note that G has a row for each row and column derivative in the image.
Let S be a matrix that zeros gradients resulting from the application of G
    (e.g. G*y).
Formally, let S be an identity matrix whose dimensions are the same as G has
rows, modified to have entries corresponding to the derivatives involving some
grid positions to be zero. Note that S = S*S and S.T = S and S*M = M*S.
Let U = S*G*y for some known other image y, and S has zeros for values in y we
don't trust.
Then
E = x.T*L*x + y.T*G.T*S.T*M*S*G*y - 2*x.T*G.T*M*S*G*y
  = x.T*L*x + y.T*G.T*S*M*S*G*y - 2*x.T*G.T*S*M*S*G*y
  = x.T*L*x + y.T*L'*y - 2*x.T*L'*y
where L' = G.T*S*M*S*G = G.T*M*S*G = G.T*S*M*G.
The derivative of 1/2 E with respect to x is:
    0.5 * dE/dx = L*x - L'*y
so the minimizing x could be found by solving
    L*x = L'*y

The G, M, and S in the above description are exactly what is returned by
grad_and_mass() in this module.
mask should be True for every grid location we care about (want a solution to).
skip should be True for every grid location we have a known good value for.

Translated by Zachary Ferguson
*/
"use sctrict";

var Dirichlet = function Dirichlet(){};


Dirichlet.grad_and_mass = function grad_and_mass(rows, cols, mask, skip){
    /*
    Returns a gradient operator matrix G for a grid of dimensions
        'rows' by 'cols',
    a corresponding mass matrix M such that L = G.T*M*G is a
    Laplacian operator matrix that is symmetric and normalized such that the
    diagonal values of interior vertices equal 1,
    and a skip matrix S such that the gradient entries of S*G*x are zero for
    (i,j) such that skip[i,j] is False. If skip is None, all entries are
    assumed to be True and S will be the identity matrix.

    Optional parameter `mask` will result in a gradient operator that entirely
    ignores (i,j) such that mask[i,j] is False.

    In other words, `mask` should be True for every grid location
    you care about (want a solution to via L = G.T*M*G).
    `skip` should be True for every grid location you have a known good value
    for.

    Matrices returned are scipy.sparse matrices.
    */

    print_progress(0)

    // The number of derivatives in the +row direction is cols * (rows - 1),
    // because the bottom row doesn't have them.
    var num_Grow = cols * (rows - 1);
    // The number of derivatives in the +col direction is rows * (cols - 1),
    // because the right-most column doesn't have them.
    var num_Gcol = rows * (cols - 1);

    // Gradient matrix
    var gOnes = numeric.ones([num_Grow + num_Gcol]);
    var vals = numeric.mul(-1, gOnes).concat(gOnes)

    var gColRange = numeric.range(rows * cols);
    gColRange = numeric.mask(gColRange, numeric.not(numeric.eq(
        numeric.mod(gColRange, cols), cols - 1)))
    var colJ = numeric.range(num_Grow).concat(gColRange,
        numeric.range(cols, num_Grow + cols), numeric.add(gColRange, 1))

    // Skip matrix
    if(skip !== undefined){
        var S_diag = flatten2D(numeric.and(numeric.slice(skip, [':-1']),
            numeric.slice(skip, ['1:'])));
        S_diag = S_diag.concat(flatten2D(numeric.and(
            numeric.slice(skip, [':', ':-1']),
            numeric.slice(skip, [':', '1:']))));
        S_diag = numeric.asNumbers(S_diag);
    }
    else{
        var S_diag = numeric.ones([num_Grow + num_Gcol]);
    }

    // Mass diagonal matrix
    if(mask !== undefined){
        let m = numeric.zeros([rows - 1, cols]);
        for(var i = 0; i < rows-1; i++){
            for(var j = 0; j < cols; j++){
                if(j > 0 && mask[i][j - 1] && mask[i + 1][j - 1]){
                    m[i, j] += 0.125;
                }
                if(j < cols - 1 && mask[i][j + 1] && mask[i + 1][j + 1]){
                    m[i, j] += 0.125;
                }
            }
        }
        mass = flatten2D(m);
        m = numeric.zeros([rows, cols - 1]);
        for(var i = 0; i < rows; i++){
            for(var j = 0; j < cols-1; j++){
                if(i > 0 && mask[i - 1][j] && mask[i - 1][j + 1]){
                    m[i, j] += 0.125;
                }
                if(i < rows - 1 && mask[i + 1][j] && mask[i + 1][j + 1]){
                    m[i, j] += 0.125;
                }
            }
        }
        mass = mass.concat(flatten2D(m));

        let keep_rows = flatten2D(numeric.and(numeric.slice(mask, [':-1']),
            numeric.slice(mask, ['1:'])))
        keep_rows = keep_rows.concat(flatten2D(numeric.and(
            numeric.slice(mask, [':', ':-1']), numeric.slice(mask, [':', '1:']))))
        tiled_keep_rows = keep_rows.concat(keep_rows)
        vals = numeric.mask(vals, tiled_keep_rows)
        colJ = numeric.mask(colJ, tiled_keep_rows)
        S_diag = numeric.mask(S_diag, keep_rows)
        mass = numeric.mask(mass, keep_rows)
        var output_row = numeric.sum(keep_rows)
    }
    else{
        m = numeric.hstack([numeric.rep([rows - 1, 1], 0.125),
                          numeric.rep([rows - 1, cols - 2], 0.25),
                          numeric.rep([rows - 1, 1], 0.125)]);
        mass = flatten2D(m);
        m = numeric.vstack([numeric.rep([1, cols - 1], 0.125),
                          numeric.rep([rows - 2, cols - 1], 0.25),
                          numeric.rep([1, cols - 1], 0.125)]);
        mass = mass.concat(flatten2D(m));
        var output_row = num_Grow + num_Gcol;
    }

    // rowI is dependent on the number of output rows.
    rowI = numeric.range(output_row)
    rowI = rowI.concat(rowI);

    G = numeric.ccsScatterShaped([rowI, colJ, vals], output_row, rows * cols);

    M = numeric.ccsDiag(mass, output_row, output_row);

    S = numeric.ccsDiag(S_diag, output_row, output_row);

    print_progress(1.0);

    return [G, M, S];
}


Dirichlet.gen_symmetric_grid_laplacian = function gen_symmetric_grid_laplacian(rows, cols, mask){
    /*
    Returns a Laplacian operator matrix for a grid of dimensions 'rows' by
    'cols'. The matrix is symmetric and normalized such that the diagonal
    values of interior vertices equal 1.

    Matrices returned are scipy.sparse matrices.
    */

    G_M_S = Dirichlet.grad_and_mass(rows, cols, mask)
    return numeric.ccsDot(numeric.ccsTranspose(G), numeric.ccsDot(M, G));
}


Dirichlet.dirichlet_energy = function dirichlet_energy(rows, cols, y, mask, skip){
    /*
    Builds the quadratic energy for the Dirichlet.

        E  = x.T*L*x + y.T*G.T*S.T*M*S*G*y - 2*x.T*G.T*M*S*G*y
           = x.T*L*x + y.T*G.T*S*M*S*G*y - 2*x.T*G.T*S*M*S*G*y
           = x.T*L*x + y.T*L'*y - 2*x.T*L'*y
        L  = G.T*M*G
        L' = G.T*S*M*G

    Inputs:
        (rows, cols) - deminsions of texture/2D grid
        y - Vector of original value (y above).
        mask - Ignore values cooresponding to False
        skip - True for good prexisting values
    Output:
        A QuadEnergy object containing the values for the quadratic, linear,
        and constant terms.
    */
    var G_M_S = Dirichlet.grad_and_mass(rows, cols, mask, skip)
    var G = G_M_S[0], M = G_M_S[1], S = G_M_S[2];
    var GT = numeric.ccsTranspose(G);

    // Divide by the sum of the Mass matrix
    // TODO: Should multiply all edges by N
    // M_sum = float(M.sum())
    var L  = numeric.ccsDot(GT, numeric.ccsDot(M, G));
    var Lp = numeric.ccsDot(GT, numeric.ccsDot(S, numeric.ccsDot(M, G)));

    var sparseY = numeric.ccsSparseShaped(y);
    var sparseYT = numeric.ccsTranspose(sparseY);
    return new QuadEnergy(L,
        numeric.ccsmul(-1, numeric.ccsDot(Lp, sparseY)),
        numeric.ccsDot(sparseYT, numeric.ccsDot(Lp, y)));
}
