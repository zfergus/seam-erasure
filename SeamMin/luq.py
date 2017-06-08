"""
Compute the LUQ decomposition of a sparse square matrix.

Based on Pawel Kowal's MatLab code.
Written by: Zachary Ferguson
"""

import numpy
import scipy.sparse
import scipy.sparse.linalg


def luq(A, do_pivot, tol = 1e-8):
    """
    PURPOSE: calculates the following decomposition

        A = L |Ubar  0 | Q
              |0     0 |

        where Ubar is a square invertible matrix
        and matrices L, Q are invertible.

    USAGE: [L,U,Q] = luq(A,do_pivot,tol)
    INPUT:
        A - a sparse matrix
        do_pivot = 1 with column pivoting
                 = 0 without column pivoting
        tol - uses the tolerance tol in separating zero and nonzero values

    OUTPUT:
        L,U,Q          matrices

    COMMENTS:
        This method is based on lu decomposition,
            https://en.wikipedia.org/wiki/LU_decomposition.

    Based on LREM_SOLVE:
    Copyright  (c) Pawel Kowal (2006)
    All rights reserved
    LREM_SOLVE toolbox is available free for noncommercial academic use only.
    pkowal3@sgh.waw.pl
    """

    n, m = A.shape

    # Test if A is a sparse matrix
    # if ~issparse(A)
    #     A = sparse(A)
    # end

    ###########################################################################
    # SPECIAL CASES
    ###########################################################################
    if(n == 0 or m == 0):
        # Return (L, U, Q) = (I(nxn), A, I(mxm))
        return (scipy.sparse.identity(n), A, scipy.sparse.identity(m))

    ###########################################################################
    # LU DECOMPOSITION
    ###########################################################################
    # Perform a LU decomposition on A.
    # Returns a scipy.sparse.linalg.SuperLU
    LUDecomp = scipy.sparse.linalg.splu(A)
    L = LUDecomp.L
    U = LUDecomp.U
    P = scipy.sparse.csc_matrix((n, n))
    P[numpy.arange(m), LUDecomp.perm_r] = 1
    if do_pivot:
        Q = scipy.sparse.csc_matrix((m, m))
        Q[numpy.arange(m), LUDecomp.perm_c] = 1
        Q = Q.T if do_pivot else scipy.sparse.identity(m)
    else:
        Q = scipy.sparse.identity(m)

    # import pdb; pdb.set_trace()
    p  = n - L.shape[1]
    LL = scipy.sparse.csc_matrix((n - p, p))
    if(p != 0):
        LL = scipy.sparse.vstack([LL, scipy.sparse.identity(p).tocsc()])
    L  = scipy.sparse.hstack([P.T.dot(L), P[(n - p):n, :].T])
    if(p != 0):
        U  = scipy.sparse.vstack([U, scipy.sparse.csc_matrix((p, m))])

    ###########################################################################
    # FINDS ROWS WITH ZERO AND NONZERO ELEMENTS ON THE DIAGONAL
    ###########################################################################
    if(U.shape[0] == 1 or U.shape[1] == 1):
        S = scipy.sparse.csc_matrix(U[0, 0])
    else:
        S = scipy.sparse.dia_matrix((U.diagonal(), [0]), shape=U.shape)

    # I = find(abs(S)>tol)
    I = (abs(S) > tol).nonzero()
    # Jl = (1:n)'
    Jl = numpy.arange(0, n).reshape((1, n)).T
    # Jl(I) = []
    Jl = numpy.delete(Jl, I[0])
    # Jq = (1:m)'
    Jq = numpy.arange(0, m).reshape((1, m)).T
    # Jq(I) = []
    Jq = numpy.delete(Jq, I)

    # Ubar1 = U(I,I)
    Ubar1 = U[I]
    # Ubar2 = U(Jl,Jq)
    Ubar2 = U[Jl.flatten(), Jq.flatten()]
    # Qbar1 = Q(I,:)
    Qbar1 = Q[I[0], :]
    # Lbar1 = L(:,I)
    Lbar1 = L[:, I[1]]

    ###########################################################################
    # ELIMINATES NONZEZO ELEMENTS BELOW AND ON THE RIGHT OF THE
    # INVERTIBLE BLOCK OF THE MATRIX U
    #
    # UPDATES MATRICES L, Q
    ###########################################################################
    # if ~isempty(I)
    import pdb
    pdb.set_trace()
    if(I[0].shape[0] != 0):
        # Utmp = U(I,Jq)
        Utmp = U[I[0], Jq]
        # X = Ubar1'\U(Jl,I)'
        X = scipy.sparse.linalg.spsolve(Ubar1.T, U[Jl, I].T)
        # Ubar2 = Ubar2-X'*Utmp
        Ubar2 = Ubar2 - X.T.dot(Utmp)
        # Lbar1 = Lbar1+L(:,Jl)*X'
        Lbar1 = Lbar1 + L[:, Jl].dot(X.T)

        # X = Ubar1\Utmp
        X = scipy.sparse.linalg.spsolve(Ubar1, Utmp)
        # Qbar1 = Qbar1+X*Q(Jq,:)
        Qbar1 = Qbar1 + X.dot(Q[Jq, :])
        # Utmp = []
        Utmp = numpy.empty(1)
        # X = []
        X = numpy.empty(1)
    # end

    ###########################################################################
    # FINDS ROWS AND COLUMNS WITH ONLY ZERO ELEMENTS
    ###########################################################################
    # I2 = find(max(abs(Ubar2),[],2)>tol)
    I2 = ((abs(Ubar2)).max(1) > tol).nonzero()
    # I5 = find(max(abs(Ubar2),[],1)>tol)
    I5 = ((abs(Ubar2)).max(0) > tol).nonzero()
    # I3 = Jl(I2)
    I3 = Jl[I2]
    # I4 = Jq(I5)
    I4 = Jq[I5]
    # Jq(I5) = []
    Jq[I5] = numpy.empty(1)
    # Jl(I2) = []
    J1[I2] = numpy.empty(1)
    # U = []
    U = numpy.empty(1)

    ###########################################################################
    # FINDS A PART OF THE MATRIX U WHICH IS NOT IN THE REQIRED FORM
    ###########################################################################
    # A = Ubar2(I2,I5)
    A = Ubar[I2, I5]

    ###########################################################################
    # PERFORMS LUQ DECOMPOSITION OF THE MATRIX A
    ###########################################################################
    # [L1,U1,Q1] = luq(A,do_pivot,tol)
    L1, U1, Q1 = luq(A, do_pivot, tol)

    ###########################################################################
    # UPDATES MATRICES L, U, Q
    ###########################################################################
    # Lbar2 = L(:,I3)*L1
    Lbar2 = L[:, I3].dot(L1)
    # Qbar2 = Q1*Q(I4,:)
    Qbar2 = Q1.dot(Q[I4, :])
    # L = [Lbar1 Lbar2 L(:,Jl)]
    L = scipy.sparse.hstack([Lbar1, Lbar2, L[:, Jl]])
    # Q = [Qbar1; Qbar2; Q(Jq,:)]
    Q = scipy.sparse.vstack([Qbar1, Qbar2, Q[Jq, :]])

    # n1 = length(I)
    n1 = I.shape[0]
    # n2 = length(I3)
    n2 = I3.shape[0]
    # m2 = length(I4)
    m2 = I4.shape[0]
    # U = [Ubar1 sparse(n1,m-n1);sparse(n2,n1) U1 sparse(n2,m-n1-m2);
    # sparse(n-n1-n2,m)]
    U = scipy.sparse.vstack([
        scipy.sparse.hstack([Ubar1, scipy.sparse.csc_matrix(
            shape = (n1, m - n1))]),
        scipy.sparse.hstack([scipy.sparse.csc_matrix(
            shape = (n2, n1)), U1, scipy.sparse.csc_matrix(
                shape = (n2, m - n1 - m2))]),
        scipy.sparse.csc_matrix(n - n1 - n2, m)
    ])

    return L, U, Q

if __name__ == "__main__":
    # A = scipy.sparse.csc_matrix(numpy.ones((4, 4)))
    A = scipy.sparse.identity(4).tocsc()
    L, U, Q = luq(A, True)
    print("L:\n%s" % L)
    print("U:\n%s" % U)
    print("Q:\n%s" % Q)
    print("A = L*U*Q:\n%s" % L.dot(U).dot(Q))
