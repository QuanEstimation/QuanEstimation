import numpy as np
import os
import copy
from scipy.sparse import csc_matrix
from sympy import Matrix, GramSchmidt
from itertools import product

def mat_vec_convert(A):
    if A.shape[1] == 1:
        dim = int(np.sqrt(len(A)))
        return A.reshape([dim, dim])
    else:
        return A.reshape([len(A)**2,1])
    
def suN_unsorted(n):
    U, V, W = [], [], []
    for i in range(1, n):
        for j in range(0, i):
            U_tp = csc_matrix(([1.,1.], ([i,j],[j,i])), shape=(n, n)).toarray()
            V_tp = csc_matrix(([1.j,-1.j], ([i,j],[j,i])), shape=(n, n)).toarray()
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

    return U, V, W

def suN_generator(n):
    symm, anti_symm, diag = suN_unsorted(n)
    if n == 2:
        return [symm[0], anti_symm[0], diag[0]]
    else:
        Lambda = [0. for i in range(len(symm+anti_symm+diag))]

        Lambda[0], Lambda[1], Lambda[2] = symm[0], anti_symm[0], diag[0]

        repeat_times = 2
        m1, n1, k1 = 0, 3, 1
        while True:
            m1 += n1
            j,l = 0,0
            for i in range(repeat_times):
                Lambda[m1+j] = symm[k1]
                Lambda[m1+j+1] = anti_symm[k1]
                j += 2
                k1 += 1

            repeat_times += 1
            n1 = n1 + 2
            if k1 == len(symm):
                break
    
        m2, n2, k2 = 2, 5, 1
        while True:
            m2 += n2
            Lambda[m2] = diag[k2]
            n2 = n2 + 2
            k2 = k2 + 1
            if k2 == len(diag):
                break
        return Lambda
        
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

def SIC(dim):
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

def brgd(n):
    if n==1:
        return ["0","1"]
    L0 = brgd(n-1)
    L1 = copy.deepcopy(L0)
    L1.reverse()
    L0 = ["0" + i for i in L0]
    L1 = ["1" + j for j in L1]
    res = L0 + L1
    return res

def AdaptiveInput(x, func, dfunc, channel="dynamics"):
    para_num = len(x)
    size = [len(x[i]) for i in range(len(x))]
    x_all = product(*x) 
    if channel == "dynamics":
        dim = len(func([0 for i in range(para_num)]))
        H_list, dH_list = [], []
        for xi in x_all:
            H_list.append(func([i for i in xi]))
            dH_list.append(dfunc([i for i in xi]))
        H_res = np.reshape(H_list, [*size,*[dim,dim]])
        dH_res = np.reshape(dH_list, [*size,*[para_num,dim,dim]])
        return H_res, dH_res
    elif channel == "kraus":
        k_num = len(func([0 for i in range(para_num)]))
        dim = len(func([0 for i in range(para_num)])[0])
        K_list, dK_list = [], []
        for xi in x_all:
            K_list.append(func([i for i in xi]))
            dK_list.append(dfunc([i for i in xi]))
        K_res = np.reshape(K_list, [*size,*[k_num,dim,dim]])
        dK_res = np.reshape(dK_list, [*size,*[para_num,k_num,dim,dim]])
        return K_res, dK_res
    else:
        raise ValueError("{!r} is not a valid value for channel, supported values are 'dynamics' and 'kraus'.".format(channel))
        
        