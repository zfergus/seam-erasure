/*
Faster way of accumulating sparse COO matricies. Avoids expensive loop of adds.
Written by Yotam Gingold
*/
"use strict";


var AccumulateCOO = function(){
    /*
    Class for accumulating additions of COO matricies. Does not sum matrices
    until total() is called.
    */
    this.rows = [];
    this.cols = [];
    this.vals = [];
}

AccumulateCOO.prototype.add = function(A){
    /*
    Add a coo_matrix to this matrix. Does not perform the addition until
    total() is called.
    Input:
        A - A ccs sparse matrix to add to this matrix
    */
    var rows_cols_vals = numeric.ccsGather(A);
    this.rows.push(rows_cols_vals[0]);
    this.cols.push(rows_cols_vals[1]);
    this.vals.push(rows_cols_vals[2]);
}

AccumulateCOO.prototype.total = function(nRows, nCols){
    /*
    Constructs a coo_matrix from the accumulated values.
    Input:
        shape - shape of the output matrix
    Output:
        Return a coo_matrix of the accumulated values.
    */
    this.rows = flatten2D(this.rows);
    this.cols = flatten2D(this.cols);
    this.vals = flatten2D(this.vals);

    return numeric.ccsScatterShaped([this.rows, this.cols, this.vals],
        nRows, nCols);
}
