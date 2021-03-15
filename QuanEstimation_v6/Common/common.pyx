# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:19:07 2020

@author: JL MZ
"""
import numpy as np
from scipy.sparse import csc_matrix
from sympy import Matrix,GramSchmidt

def Liouville_commu(double complex [:,:] A):
    cdef Py_ssize_t dim = A.shape[0]
    
    result = np.zeros((dim*dim, dim*dim), dtype=np.complex128)
    cdef double complex [:, :] result_view = result
    
    cdef int ni, nj, nk
    cdef Py_ssize_t bi, bj, bk

    for bi in range(0,dim):
        for bj in range(0,dim):
             for bk in range(0,dim):
                ni = dim*bi+bj
                nj = dim*bk+bj
                nk = dim*bi+bk

                result_view[ni,nj] = A[bi,bk]
                result_view[ni,nk] = -A[bk,bj]
                result_view[ni,ni] = A[bi,bi]-A[bj,bj]
    res = np.asarray(result_view)
    return res

def Liouville_dissip(double complex[:,:] A):

    cdef Py_ssize_t dim = A.shape[0]
    
    result = np.zeros((dim*dim, dim*dim), dtype=np.complex128)
    cdef double complex [:, :] result_view = result
    
    cdef double complex L_temp
    cdef int ni, nj
    cdef Py_ssize_t bi, bj, bk, bl, bp
    
    for bi in range(dim):
        for bj in range(dim):
            ni = dim*bi+bj
            for bk in range(dim):
                for bl in range(dim):
                    nj = dim*bk+bl
                    L_temp = A[bi,bk]*np.conj(A[bj,bl])
                    for bp in range(dim):
                        L_temp = L_temp-0.5*float(bk==bi)*A[bp,bj]*np.conj(A[bp,bl])\
                                   -0.5*float(bl==bj)*A[bp,bk]*np.conj(A[bp,bi])
                    result_view[ni,nj] = L_temp

    res = np.asarray(result_view)
    res[np.abs(res) < 1e-10] = 0.
    return res

def dydt(d_rho, Liouv_tot, A): 
    drho = np.dot(Liouv_tot, d_rho) + A
    return drho

def dRHO(d_rho, Liouv_tot, A, h): 
    k1 = h * dydt(d_rho, Liouv_tot, A) 
    k2 = h * dydt(d_rho+0.5*k1, Liouv_tot, A) 
    k3 = h * dydt(d_rho+0.5*k2, Liouv_tot, A) 
    k4 = h * dydt(d_rho+k3, Liouv_tot, A) 
    d_rho = d_rho + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
    return d_rho
        
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

def Adam(gt, t, para, m_t, v_t, alpha=0.01, beta1=0.90, beta2=0.99, epsilon=1e-8):
    t = t+1
    m_t = beta1*m_t + (1-beta1)*gt
    v_t = beta2*v_t + (1-beta2)*(gt*gt)
    m_cap = m_t/(1-(beta1**t))
    v_cap = v_t/(1-(beta2**t))
    para = para +(alpha*m_cap)/(np.sqrt(v_cap)+epsilon)
    return para, m_t, v_t
    