from scipy import io as sio
import os
import numpy as np
import sys
from scipy import sparse
from scipy.sparse.linalg import lsqr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import laplaceBeltrami

DIR = '/Users/shin/Documents/MATLAB/3DMMasSTN'

# Load original BFM
BFM = sio.loadmat(os.path.join(DIR, 'models/PublicMM1/01_MorphableModel.mat'))

# Load modified FW expression model
FWexprModel = sio.loadmat(
    os.path.join(DIR, 'models/3DDFA_Release/Matlab/Model_Expression.mat'))

# Load FW to BFM mapping
FWtoBFMmap = sio.loadmat(os.path.join(DIR, 'util/map_tddfa_to_basel.mat'))

model = {'shapePC': np.zeros((3 * 53490, 10), dtype=np.float)}
model['shapePC'][:, 0:
                 5] = BFM['shapePC'][:, 0:5] * BFM['shapeEV'][0:5].squeeze()

# Extrapolate expressions to mouth interior via Laplace-Beltrami
vertices = BFM['shapeMU'].reshape(53490, 3).astype(np.double)
L = laplaceBeltrami.laplaceBeltrami(vertices.T, BFM['tl'].T - 1)

# Selection matrix for BFM vertices to FW
S = sparse.csr_matrix(
    (np.ones_like(FWtoBFMmap['map_tddfa_to_basel'].squeeze()),
     (np.arange(len(FWtoBFMmap['map_tddfa_to_basel'].squeeze())),
      FWtoBFMmap['map_tddfa_to_basel'].squeeze())),
    shape=(len(FWtoBFMmap['map_tddfa_to_basel'][0]), 53490))

wExp = FWexprModel['w_exp'][:, 0].reshape(53215, 3).T
numer = sparse.vstack([L, S])
print([numer])
print([
    sparse.vstack([
        L * vertices,
        S * vertices + FWexprModel['w_exp'][:, 0].reshape(53215, 3)
    ])
])
for i in range(0, 5):
    expression = lsqr(
        numer,
        sparse.vstack([
            L * vertices,
            S * vertices + FWexprModel['w_exp'][:, i].reshape(53215, 3)
        ]).toarray()) - vertices
    model['shapePC'][:, i + 5] = expression.T[:]
    print('Loop', i)

