import numpy as np
import os
import copy
from scipy.sparse import csc_matrix, csr_matrix
from sympy import Matrix, GramSchmidt
from itertools import product
import juliacall
from scipy.linalg import sqrtm


def load_julia():
    """
    Load Julia and initialize QuanEstimation module.
    
    Returns:
        jl.Main.QuanEstimation: Julia module for quantum estimation
    """
    jl = juliacall.newmodule("QuanEstimation")
    jl.Main.seval("using QuanEstimation, PythonCall")
    return jl.Main.QuanEstimation


def mat_vec_convert(A):
    """
    Convert between matrix and vector representations.
    
    Args:
        A (np.array): 
            Input matrix or vector. 
        
    Returns:
        (np.array): 
            Converted matrix or vector.
    """
    if A.shape[1] == 1:
        dim = int(np.sqrt(len(A)))
        return A.reshape([dim, dim])
    else:
        return A.reshape([len(A) ** 2, 1])


def suN_unsorted(n):
    """
    Generate unsorted SU(N) generators.
    
    Args:
        n (float): 
            Dimension of the system.
        
    Returns:
        (tuple): 
            Tuple of symmetric, antisymmetric, and diagonal generators.
    """
    U, V, W = [], [], []
    for i in range(1, n):
        for j in range(0, i):
            U_tp = csc_matrix(
                ([1.0, 1.0], ([i, j], [j, i])), 
                shape=(n, n)
            ).toarray()
            V_tp = csc_matrix(
                ([1.0j, -1.0j], ([i, j], [j, i])), 
                shape=(n, n)
            ).toarray()
            U.append(U_tp)
            V.append(V_tp)

    diag = []
    for i in range(n - 1):
        mat_tp = csc_matrix(
            ([1, -1], ([i, i + 1], [i, i + 1])), 
            shape=(n, n)
        ).toarray()
        diag_tp = np.diagonal(mat_tp)
        diag.append(Matrix(diag_tp))
        
    W_gs = GramSchmidt(diag, True)
    for k in range(len(W_gs)):
        W_tp = np.fromiter(W_gs[k], dtype=complex)
        W.append(np.sqrt(2) * np.diag(W_tp))

    return U, V, W


def suN_generator(n):
    """
    Generate sorted SU(N) generators.
    
    Args:
        n (float): 
            Dimension of the system. 
        
    Returns:
        (list): 
            List of SU(N) generators.
    """
    symm, anti_symm, diag = suN_unsorted(n)
    if n == 2:
        return [symm[0], anti_symm[0], diag[0]]
    else:
        Lambda = [0.0] * len(symm + anti_symm + diag)

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
    """
    Perform Gram-Schmidt orthogonalization.
    
    Args:
        A (list): 
            List of vectors to orthogonalize.
        
    Returns:
        (list): 
            List of orthonormal vectors.
    """
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
    """
    Generate basis vector.
    
    Args:
        dim (float): 
            Dimension of Hilbert space.
        index (float): 
            Index of basis vector.
        
    Returns:
        (np.array): 
            Basis vector as column vector.
    """
    x = np.zeros(dim)
    x[index] = 1.0
    return x.reshape(dim, 1)


def sic_povm(fiducial):
    """
    Generate SIC-POVM from fiducial state.
    
    Args:
        fiducial (np.array): 
            the fiducial state vector.
        
    Returns:
        (list): 
            List of POVM elements.
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
            phase = (-np.exp(1.0j * np.pi / d)) ** (a * b)
            D[a][b] = phase * np.dot(X_a, Z_b)

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
    Generate SIC-POVM for given dimension.
    
    Args:
        dim (float): 
            Dimension of the system.
        
    Returns:
        (list): 
            List of SIC-POVM elements.
        
    Raises:
        ValueError: 
            If dimension > 151.
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
        raise ValueError(
            "The dimension of the space should be less or equal to 151."
        )


def extract_ele(element, n):
    """
    Recursively extract elements.
    
    Args:
        element (np.array/list): 
            Input element
        n (int/float): 
            Depth of extraction.
    """
    if n:
        for x in element:
            yield from extract_ele(x, n - 1)
    else:
        yield element


def annihilation(n):
    """
    Create annihilation operator.
    
    Args:
        n (int/float): 
            Dimension of space.
        
    Returns:
        (np.array): 
            Annihilation operator matrix.
    """
    data = np.sqrt(np.arange(1, n, dtype=complex))
    indices = np.arange(1, n)
    indptr = np.arange(n + 1)
    indptr[-1] = n - 1
    a = csr_matrix((data, indices, indptr), shape=(n, n)).todense()
    return a


def brgd(n):
    """
    Generate binary reflected Gray code.
    
    Args:
        n (int/float): 
            Number of bits
        
    Returns:
        (list): 
            List of Gray code sequences
    """
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
    Generate input variables for Bayesian estimation.
    
    Args:
        x (np.array): 
            Parameter regimes
        func (callable): 
            Function returning H or K
        dfunc (callable): 
            Function returning dH or dK
        channel (str, optional): 
            "dynamics" or "Kraus" (default: "dynamics")
        
    Returns:
        (tuple): 
            Tuple of (H_list, dH_list) or (K_list, dK_list)
        
    Raises:
        ValueError: 
            For invalid channel.
    """
    x_all = product(*x)
    if channel == "dynamics":
        H_list, dH_list = [], []
        for xi in x_all:
            H_list.append(func(*xi))
            dH_list.append(dfunc(*xi))
        return H_list, dH_list
    elif channel == "Kraus":
        K_list, dK_list = [], []
        for xi in x_all:
            K_list.append(func(*xi))
            dK_list.append(dfunc(*xi))
        return K_list, dK_list
    else:
        raise ValueError(
            "{!r} is not a valid channel. Supported values: "
            "'dynamics' or 'Kraus'.".format(channel)
        )

def fidelity(input1, input2):
    """
    Compute the fidelity between two quantum states.
    
    For state vectors (1D arrays), the fidelity is defined as |<ψ|φ>|².
    For density matrices (2D arrays), the fidelity is defined as [tr(√(√ρ σ √ρ))]².
    
    Args:
        input1 (np.ndarray): First quantum state (vector or density matrix)
        input2 (np.ndarray): Second quantum state (vector or density matrix)
        
    Returns:
        float: Fidelity between the two states
        
    Raises:
        TypeError: If inputs are not numpy arrays
        ValueError: If inputs are not both vectors or both matrices
        RuntimeError: If the matrix product is not positive semidefinite
    """
    if not (isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray)):
        raise TypeError("Inputs must be numpy arrays")
    
    if input1.ndim == 1 and input2.ndim == 1:
        # Vector case
        overlap = np.vdot(input1, input2)
        return np.abs(overlap) ** 2
    
    elif input1.ndim != 1:
        # Matrix case
        if input1.shape != input2.shape:
            raise ValueError("Input matrices must have the same shape")
        
        rho_sqrt = sqrtm(input1)
        product = rho_sqrt @ input2 @ rho_sqrt            
        fidelity_sqrt = np.trace(sqrtm(product))

        return np.real(fidelity_sqrt) ** 2
    else:
        raise ValueError("Inputs must be either both vectors or both matrices")
