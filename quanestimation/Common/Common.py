import numpy as np
import os
import copy
from scipy.sparse import csc_matrix, csr_matrix
from sympy import Matrix, GramSchmidt
from itertools import product
import juliacall

def load_julia():
    """
    Load Julia.
    """

    jl = juliacall.newmodule("QuanEstimation")
    jl.Main.seval("using QuanEstimation, PythonCall")
    return jl.Main.QuanEstimation

def mat_vec_convert(A):
    if A.shape[1] == 1:
        dim = int(np.sqrt(len(A)))
        return A.reshape([dim, dim])
    else:
        return A.reshape([len(A) ** 2, 1])


def suN_unsorted(n):
    U, V, W = [], [], []
    for i in range(1, n):
        for j in range(0, i):
            U_tp = csc_matrix(([1.0, 1.0], ([i, j], [j, i])), shape=(n, n)).toarray()
            V_tp = csc_matrix(([1.0j, -1.0j], ([i, j], [j, i])), shape=(n, n)).toarray()
            U.append(U_tp)
            V.append(V_tp)

    diag = []
    for i in range(n - 1):
        mat_tp = csc_matrix(([1, -1], ([i, i + 1], [i, i + 1])), shape=(n, n)).toarray()
        diag_tp = np.diagonal(mat_tp)
        diag.append(Matrix(diag_tp))
    W_gs = GramSchmidt(diag, True)
    for k in range(len(W_gs)):
        W_tp = np.fromiter(W_gs[k], dtype=complex)
        W.append(np.sqrt(2) * np.diag(W_tp))

    return U, V, W


def suN_generator(n):
    r"""
    Generation of the SU($N$) generators with $N$ the dimension of the system.

    Parameters
    ----------
    > **n:** `int` 
        -- The dimension of the system.

    Returns
    ----------
    SU($N$) generators.
    """

    symm, anti_symm, diag = suN_unsorted(n)
    if n == 2:
        return [symm[0], anti_symm[0], diag[0]]
    else:
        Lambda = [0.0 for i in range(len(symm + anti_symm + diag))]

        Lambda[0], Lambda[1], Lambda[2] = symm[0], anti_symm[0], diag[0]

        repeat_times = 2
        m1, n1, k1 = 0, 3, 1
        while True:
            m1 += n1
            j, l = 0, 0
            for i in range(repeat_times):
                Lambda[m1 + j] = symm[k1]
                Lambda[m1 + j + 1] = anti_symm[k1]
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
            q = q - rij * Q[i]
        rjj = np.linalg.norm(q, ord=2)
        Q[j] = q / rjj
    return Q


def basis(dim, index):
    x = np.zeros(dim)
    x[index] = 1.0
    return x.reshape(dim, 1)


def sic_povm(fiducial):
    """
    Generate a set of POVMs by applying the $d^2$ Weyl-Heisenberg displacement operators to a
    fiducial state.
    The Weyl-Heisenberg displacement operators are constructioned by Fuchs et al. in the article
    https://doi.org/10.3390/axioms6030021 and it is realized in QBism.
    """

    d = fiducial.shape[0]
    w = np.exp(2.0 * np.pi * 1.0j / d)
    Z = np.diag([w**i for i in range(d)])
    X = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            if j != (d - 1):
                X += np.dot(basis(d, j + 1), basis(d, j).transpose().conj())
            else:
                X += np.dot(basis(d, 0), basis(d, j).transpose().conj())
    X = X / d

    D = [[[] for i in range(d)] for j in range(d)]
    for a in range(d):
        for b in range(d):
            X_a = np.linalg.matrix_power(X, a)
            Z_b = np.linalg.matrix_power(Z, b)
            D[a][b] = (-np.exp(1.0j * np.pi / d)) ** (a * b) * np.dot(X_a, Z_b)

    res = []
    for m in range(d):
        for n in range(d):
            res_tp = np.dot(D[m][n], fiducial)
            res_tp = res_tp / np.linalg.norm(res_tp)
            basis_res = np.dot(res_tp, res_tp.conj().transpose()) / d
            res.append(basis_res)
    return res


def SIC(dim):
    """
    Generation of a set of rank-one symmetric informationally complete 
    positive operator-valued measure (SIC-POVM).

    Parameters
    ----------
    > **dim:** `int` 
        -- The dimension of the system.

    Returns
    ----------
    A set of SCI-POVM.

    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    if dim <= 151:
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "sic_fiducial_vectors/d%d.txt" % (dim),
        )
        data = np.loadtxt(file_path)
        fiducial = data[:, 0] + data[:, 1] * 1.0j
        fiducial = np.array(fiducial).reshape(len(fiducial), 1)
        M = sic_povm(fiducial)
        return M
    else:
        raise ValueError("The dimension of the space should be less or equal to 151.")


def extract_ele(element, n):
    if n:
        for x in element:
            yield from extract_ele(x, n - 1)
    else:
        yield element


def annihilation(n):
    data = np.sqrt(np.arange(1, n, dtype=complex))
    indices = np.arange(1, n)
    indptr = np.arange(n + 1)
    indptr[-1] = n - 1
    a = csr_matrix((data, indices, indptr), shape=(n, n)).todense()
    return a


def brgd(n):
    if n == 1:
        return ["0", "1"]
    L0 = brgd(n - 1)
    L1 = copy.deepcopy(L0)
    L1.reverse()
    L0 = ["0" + i for i in L0]
    L1 = ["1" + j for j in L1]
    res = L0 + L1
    return res


def BayesInput(x, func, dfunc, channel="dynamics"):
    """
    Generation of the input variables H, dH (or K, dK).

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **func:** `list`
        -- Function defined by the users which returns H or K.

    > **dfunc:** `list`
        -- Function defined by the users which returns dH or dK.

    > **channel:** `string`
        -- Seeting the output of this function. Options are:  
        "dynamics" (default) --  The output of this function is H and dH.  
        "Kraus" (default) --  The output of this function is K and dHK.

    Returns
    ----------
    H, dH (or K, dK).
    """

    para_num = len(x)
    size = [len(x[i]) for i in range(len(x))]
    x_all = product(*x)
    if channel == "dynamics":
        dim = len(func([0 for i in range(para_num)]))
        H_list, dH_list = [], []
        for xi in x_all:
            H_list.append(func([i for i in xi]))
            dH_list.append(dfunc([i for i in xi]))
        # H_res = np.reshape(H_list, [*size, *[dim, dim]])
        # dH_res = np.reshape(dH_list, [*size, *[para_num, dim, dim]])
        return H_list, dH_list
    elif channel == "Kraus":
        k_num = len(func([0 for i in range(para_num)]))
        dim = len(func([0 for i in range(para_num)])[0])
        K_list, dK_list = [], []
        if para_num == 1:
            for xi in x_all:
                K_list.append(func([i for i in xi]))
            #     dK_list.append(dfunc([i for i in xi]))
            # K_res = np.reshape(K_list, [*size, *[k_num, dim, dim]])
            # dK_res = np.reshape(dK_list, [*size, *[para_num, k_num, dim, dim]])
        else:
            for xi in x_all:
                K_list.append(func([i for i in xi]))
                dK_list.append(dfunc([i for i in xi]))
            # K_res = np.reshape(K_list, [*size, *[k_num, dim, dim]])
            # dK_res = np.reshape(dK_list, [*size, *[k_num, para_num, dim, dim]])
        return K_list, dK_list
    else:
        raise ValueError(
            "{!r} is not a valid value for channel, supported values are 'dynamics' and 'Kraus'.".format(
                channel
            )
        )
