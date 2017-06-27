/*
Commonly used Numpy function not implemented instandard Numeric.js
Written by Zachary Ferguson
*/
"use strict";


numeric.ccsZeros = function ccsZeros(nRows, nCols){
    /* Returns an empty ccsSparse of size nRows x nCols. */
    return numeric.ccsScatter([[nRows-1], [nCols-1], [0]])
}


numeric.ccsScatterAdd = function ccsScatterAdd(ijv){
    /*
    Acts the same as ccsScatter but any duplicate indecies are summed together.
    */
    var rows = ijv[0], cols = ijv[1], vals = ijv[2];
    var summedRows = [], summedCols = [], summedVals = [];
    var seenHash = {};
    for(var i = 0; i < rows.length; i++)
    {
        var index = seenHash[[rows[i], cols[i]]];
        if(index === undefined)
        {
            seenHash[[rows[i], cols[i]]] = summedRows.length;
            summedRows.push(rows[i]);
            summedCols.push(cols[i]);
            summedVals.push(vals[i]);
        }
        else{
            summedVals[index] += vals[i];
        }
    }

    return numeric.ccsScatter([summedRows, summedCols, summedVals]);
}


numeric.ccsScatterShaped = function ccsScatterShaped(ijv, nRows, nCols){
    /*
    Acts as ccsScatterAdd but expands the rows and cols to the shape provided.
    */
    if(nRows !== undefined && nCols !== undefined){
        var rows = ijv[0], cols = ijv[1], vals = ijv[2];
        if(rows.indexOf(nRows-1) < 0 || cols.indexOf(nCols-1) < 0){
            // Expand matrix if necessary
            rows.push(nRows-1);
            cols.push(nCols-1);
            vals.push(0);
        }
    }

    return numeric.ccsScatterAdd([rows, cols, vals]);
}


numeric.ccsSparseShaped = function ccsSparseShaped(A, dims){
    if(dims === undefined){ // By default preserve the dims of A
        dims = numeric.dim(A);
    }
    var spA = numeric.ccsSparse(A); // Convert to Sparse
    if((""+numeric.ccsDim(spA)) !== (""+dims)){ // Expand if necessary
        spA[0][spA[0].length - 1] += 1;
        spA[1].push(dims[0]-1);
        spA[2].push(0);
    }
    return spA;
}


numeric.ccsDiag = function ccsDiag(vals, nRows, nCols){
    var indices = numeric.range(vals.length);
    return numeric.ccsScatterShaped([indices, indices, vals], nRows, nCols);
}

numeric.ccsTranspose = function ccsTranspose(M){
    var ijv = numeric.ccsGather(M);
    return numeric.ccsScatter([ijv[1], ijv[0], ijv[2]]);
}

numeric.equal = function equal(x, y, tol){
    if(tol === undefined){
        tol = 1e-8;
    }

    return numeric.norm1(numeric.sub(x, y)) < tol;
}

numeric.asNumbers = function asNumbers(x){
    return x.map(Number);
}

numeric.solveMatrix = function solveMatrix(A, B){
    var LU = numeric.LU(A);
    var dims = numeric.dim(B);
    var nCols = dims.length > 1 ? dims[1] : 1;
    var sol = undefined;
    for(var j = 0; j < nCols; j++){
        var solCol = numeric.LUsolve(LU, numeric.slice(B, [':', j]));
        if(sol === undefined){
            sol = numeric.reshape(solCol, [dims[0], 1]);
        }
        else{
            sol = numeric.hstack([sol, numeric.reshape(solCol, [dims[0], 1])])
        }
    }
    return sol;
}

numeric.ccsGetCol = function ccsGetCol(M, j){
    var nRows = numeric.ccsDim(M)[0];
    var col = numeric.ccsGetBlock(M, undefined, j);
    col[0] = [0, nRows];
    if(col[1].indexOf(nRows-1) < 0){
        col[1].push(nRows-1);
        col[2].push(0);
    }
    return col;
}

numeric.ccsHStack = function ccsHStack(A, B, nRows, nColsA, nColsB){
    var gatheredA = numeric.ccsGather(A);
    var gatheredB = numeric.ccsGather(B);

    var rows = gatheredA[0].concat(gatheredB[0]);
    var cols = gatheredA[1].concat(numeric.add(nColsA, gatheredB[1]));
    var vals = gatheredA[2].concat(gatheredB[2]);
    return numeric.ccsScatterShaped([rows, cols, vals], nRows, nColsA + nColsB);
}

numeric.ccsSolveMatrix = function ccsSolveMatrix(A, B){
    var solving = true;
    let foo = function(){
        if(solving){
            console.log('.');
            setTimeout(foo, 100);
        }
    }
    foo();
    var LUP = numeric.ccsLUP(A);
    var dims = numeric.ccsDim(B);
    var nRows = dims[0], nCols = dims.length > 1 ? dims[1] : 1;
    var sol = undefined;
    for(var j = 0; j < nCols; j++){
        var solCol = numeric.ccsLUPSolve(LUP, numeric.ccsGetCol(B, j))
        if(sol === undefined){
            sol = solCol;
        }
        else{
            sol = numeric.ccsHStack(sol, solCol, nRows, j, 1);
        }
    }
    solving = false;
    return sol;
}
