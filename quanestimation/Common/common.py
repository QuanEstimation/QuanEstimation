# -*- coding: utf-8 -*-

import numpy as np
import os
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

def gramschmidt(A):
    dim = len(A)
    n = len(A[0])
    Q = [np.zeros(n, dtype=np.complex128) for i in range(dim)]
    for j in range(0, dim):
        q = A[j]
        for i in range(0, j):
            rij = np.vdot(Q[i], q)
            q = q - rij*Q[i]
        rjj = np.linalg.norm(q, ord=2)
        Q[j] = q/rjj
    return Q

def get_basis(dim, index):
    x = np.zeros(dim)
    x[index] = 1.0
    return x.reshape(dim,1)
    
def sic_povm(fiducial):
    """
    Generate a set of POVMs by applying the $d^2$ Weyl-Heisenberg displacement operators to a
    fiducial state. 
    The Weyl-Heisenberg displacement operators are constructioned by Fuchs et al. in the article
    https://doi.org/10.3390/axioms6030021 and it is realized in QBism.

    """

    d = fiducial.shape[0]
    w = np.exp(2.0*np.pi*1.0j/d)
    Z = np.diag([w**i for i in range(d)])
    X = np.zeros((d,d),dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            if j != (d-1):
                X += np.dot(get_basis(d, j+1), get_basis(d,j).transpose().conj())
            else:
                X += np.dot(get_basis(d, 0), get_basis(d,j).transpose().conj())
    X = X/d
    
    D = [[[] for i in range(d)] for j in range(d)]
    for a in range(d):
        for b in range(d):
            X_a = np.linalg.matrix_power(X,a)
            Z_b = np.linalg.matrix_power(Z,b)
            D[a][b] = (-np.exp(1.0j*np.pi/d))**(a*b)*np.dot(X_a, Z_b)
     
    res = []
    for m in range(d):
        for n in range(d):
            res_tp = np.dot(D[m][n], fiducial)
            res_tp = res_tp/np.linalg.norm(res_tp)
            basis_res = np.dot(res_tp, res_tp.conj().transpose())/d
            res.append(basis_res)
    return res

def load_M(dim):
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'sic_fiducial_vectors/d%d.txt'%(dim))
    data = np.loadtxt(file_path)
    fiducial = data[:,0] + data[:,1]*1.0j
    fiducial = np.array(fiducial).reshape(len(fiducial),1) 
    M = sic_povm(fiducial)
    return M

def extract_ele(element, n):
    if n:
        for x in element: 
            yield from extract_ele(x,n-1)
    else: 
        yield element
        