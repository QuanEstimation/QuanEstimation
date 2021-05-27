# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:19:07 2020

@author: JL MZ
"""
import numpy as np
from scipy.sparse import csc_matrix
from sympy import Matrix,GramSchmidt

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
    para = para+(alpha*m_cap)/(np.sqrt(v_cap)+epsilon)
    return para, m_t, v_t
    