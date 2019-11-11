import numpy as np
from scipy import sparse


def laplaceBeltrami(X, F):
    # Adapted from the tutorial of Gabriel Peyre:
    # http://www.numerical-tours.com/matlab/meshproc_7_geodesic_poisson/

    n = X.shape[1]
    m = F.shape[1]

    # Callback to get the coordinates of all the
    # vertex of index i=1,2,3 in all faces
    XF = lambda i: X[:, F[i, :]]

    # Compute un-normalized normal through the
    # formula e1xe2 where ei are the edges.
    a = XF(1) - XF(0)
    b = XF(2) - XF(0)
    Na = np.cross(XF(1) - XF(0), XF(2) - XF(0), axis=0)

    # Compute the area of each face as half the norm of the cross product.
    amplitude = lambda x: np.sqrt(np.sum(x**2, axis=0))
    A = amplitude(Na) / 2.

    # Compute the set of unit-norm normals to each face.
    normalize = lambda x: x / np.tile(amplitude(x), (3, 1))
    N = normalize(Na)

    # Populate the sparse entries of the matrices for the operator implementing
    # opposite edge e_i indexes
    s = np.mod(np.arange(3) + 1, 3)
    t = np.mod(np.arange(3) + 2, 3)

    # indexes to build the sparse matrices
    I = np.tile(np.arange(0, m), 3)
    J = F.flatten()
    # vector N_f^e_i
    V = np.cross(XF(t) - XF(s), N, axis=0).reshape(3, -1)

    # Sparse matrix with entries 1/(2Af)
    dA = sparse.spdiags(1. / (2 * A[:]), 0, m, m)

    # Compute gradient.
    gradMat = [
        dA * sparse.csr_matrix((V[i], (I, J)), shape=(m, n)) for i in range(3)
    ]

    # Compute divergence matrices as transposed of
    # grad for the face area inner product.
    dAf = sparse.spdiags(2 * A[:], 0, m, m)
    divMat = [gradMat[i].T * dAf for i in range(3)]

    # Laplacian operator as the composition of grad and div.
    delta = divMat[0] * gradMat[0] + divMat[1] * gradMat[1] + divMat[2] * gradMat[2]

    return delta
