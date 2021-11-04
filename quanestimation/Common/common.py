# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csc_matrix
from sympy import Matrix, GramSchmidt
        
def mat_vec_convert(A):
    if A.shape[1] == 1:
        dim = int(np.sqrt(len(A)))
        return A.reshape([dim, dim])
    else:
        return A.reshape([len(A)**2,1])
    
def suN_generator(n):
    U, V, W = [], [], []
    for i in range(n-1):
        for j in range(i+1, n):
            U_tp = csc_matrix(([1.,1.], ([i,j],[j,i])), shape=(n, n)).toarray()
            V_tp = csc_matrix(([-1.j,1.j], ([i,j],[j,i])), shape=(n, n)).toarray()
            U.append(U_tp)
            V.append(V_tp)

    diag = []
    for i in range(n-1):
        mat_tp = csc_matrix(([1,-1], ([i,i+1],[i,i+1])), shape=(n, n)).toarray()
        diag_tp = np.diagonal(mat_tp)
        diag.append(Matrix(diag_tp)) 
    W_gs = GramSchmidt(diag,True)
    for k in range(len(W_gs)):
        W_tp = np.fromiter(W_gs[k], dtype=complex)
        W.append(np.sqrt(2)*np.diag(W_tp))

    return U+V+W
