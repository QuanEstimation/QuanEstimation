import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm, schur, eigvals
from quanestimation.Common.Common import SIC, suN_generator
from scipy.integrate import quad
from scipy.stats import norm, poisson, rayleigh, gamma

def CFIM(rho, drho, M=[], eps=1e-8):
    r"""
    Calculation of the classical Fisher information (CFI) and classical Fisher 
    information matrix (CFIM) for a density matrix. The entry of CFIM $\mathcal{I}$
    is defined as
    \begin{align}
    \mathcal{I}_{ab}=\sum_y\frac{1}{p(y|\textbf{x})}[\partial_a p(y|\textbf{x})][\partial_b p(y|\textbf{x})],
    \end{align}

    where $p(y|\textbf{x})=\mathrm{Tr}(\rho\Pi_y)$ with $\rho$ the parameterized 
    density matrix.

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        -- Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **M:** `list of matrices`
        -- A set of positive operator-valued measure (POVM). The default measurement 
        is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **CFI (CFIM):** `float or matrix` 
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is CFI and for multiparameter estimation (the length of drho 
        is more than one), it returns CFIM.
    
    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    if M == []:
        M = SIC(len(rho[0]))
    else:
        if type(M) != list:
            raise TypeError("Please make sure M is a list!")

    m_num = len(M)
    para_num = len(drho)
    CFIM_res = np.zeros([para_num, para_num])
    for pi in range(0, m_num):
        Mp = M[pi]
        p = np.real(np.trace(np.dot(rho, Mp)))
        Cadd = np.zeros([para_num, para_num])
        if p > eps:
            for para_i in range(0, para_num):
                drho_i = drho[para_i]
                dp_i = np.real(np.trace(np.dot(drho_i, Mp)))
                for para_j in range(para_i, para_num):
                    drho_j = drho[para_j]
                    dp_j = np.real(np.trace(np.dot(drho_j, Mp)))
                    Cadd[para_i][para_j] = np.real(dp_i * dp_j / p)
                    Cadd[para_j][para_i] = np.real(dp_i * dp_j / p)
        CFIM_res += Cadd

    if para_num == 1:
        return CFIM_res[0][0]
    else:
        return CFIM_res


def FIM(p, dp, eps=1e-8):
    r"""
    Calculation of the classical Fisher information (CFI) and classical Fisher 
    information matrix (CFIM) for classical scenarios. The entry of FIM $I$
    is defined as
    \begin{align}
    I_{ab}=\sum_{y}\frac{1}{p_y}[\partial_a p_y][\partial_b p_y],
    \end{align}

    where $\{p_y\}$ is a set of the discrete probability distribution.

    Parameters
    ----------
    > **p:** `array` 
        -- The probability distribution.

    > **dp:** `list`
        -- Derivatives of the probability distribution on the unknown parameters to 
        be estimated. For example, dp[0] is the derivative vector on the first 
        parameter.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **CFI (CFIM):** `float or matrix` 
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is CFI and for multiparameter estimation (the length of drho 
        is more than one), it returns CFIM.
    """

    para_num = len(dp[0])
    m_num = len(p)
    FIM_res = np.zeros([para_num, para_num])
    for pi in range(0, m_num):
        p_tp = p[pi]
        Cadd = np.zeros([para_num, para_num])
        if p_tp > eps:
            for para_i in range(0, para_num):
                dp_i = dp[pi][para_i]
                for para_j in range(para_i, para_num):
                    dp_j = dp[pi][para_j]
                    Cadd[para_i][para_j] = np.real(dp_i * dp_j / p_tp)
                    Cadd[para_j][para_i] = np.real(dp_i * dp_j / p_tp)
        FIM_res += Cadd

    if para_num == 1:
        return FIM_res[0][0]
    else:
        return FIM_res

def FI_Expt(y1, y2, dx, ftype="norm"):
    r"""
    Calculation of the classical Fisher information (CFI) based on the experiment data. 

    Parameters
    ----------
    > **y1:** `array` 
        -- Experimental data obtained at the truth value (x).

    > **y2:** `list`
        -- Experimental data obtained at x+dx.

    > **dx:** `float`
        -- A known small drift of the parameter.

    > **ftype:** `string`
        -- The distribution the data follows. Options are:  
        "norm" (default) -- normal distribution.  
        "gamma" -- gamma distribution.
        "rayleigh" -- rayleigh distribution.
        "poisson" -- poisson distribution.

    Returns
    ----------
    **CFI:** `float or matrix` 
    """
    fidelity = 0.0
    if ftype == "norm":
        mu1, std1 = norm.fit(y1)
        mu2, std2 = norm.fit(y2)
        f_func = lambda x: np.sqrt(norm.pdf(x, mu1, std1)*norm.pdf(x, mu2, std2))
        fidelity, err = quad(f_func, -np.inf, np.inf)
    elif ftype == "gamma":
        a1, alpha1, beta1 = gamma.fit(y1)
        a2, alpha2, beta2 = gamma.fit(y2)
        f_func = lambda x: np.sqrt(gamma.pdf(x, a1, alpha1, beta1)*gamma.pdf(x, a2, alpha2, beta2))
        fidelity, err = quad(f_func, 0., np.inf)
    elif ftype == "rayleigh":
        mean1, var1 = rayleigh.fit(y1)
        mean2, var2 = rayleigh.fit(y2)
        f_func = lambda x: np.sqrt(rayleigh.pdf(x, mean1, var1)*rayleigh.pdf(x, mean2, var2))
        fidelity, err = quad(f_func, -np.inf, np.inf)
    elif ftype == "poisson":
        k1 = np.arange(max(y1)+1)
        k2 = np.arange(max(y2)+1)
        p1_pois = poisson.pmf(k1, np.mean(y1))
        p2_pois = poisson.pmf(k2, np.mean(y2))
        p1_pois, p2_pois = p1_pois/sum(p1_pois), p2_pois/sum(p2_pois)
        fidelity = sum([np.sqrt(p1_pois[i]*p2_pois[i]) for i in range(len(p1_pois))])
    else:
        raise ValueError("{!r} is not a valid value for ftype, supported values are 'norm', 'poisson', 'gamma' and 'rayleigh'.".format(ftype))
    Fc = 8*(1-fidelity)/dx**2
    return Fc


def SLD(rho, drho, rep="original", eps=1e-8):
    r"""
    Calculation of the symmetric logarithmic derivative (SLD) for a density matrix.
    The SLD operator $L_a$ is determined by
    \begin{align}
    \partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)
    \end{align}

    with $\rho$ the parameterized density matrix. The entries of SLD can be calculated
    as 
    \begin{align}
    \langle\lambda_i|L_{a}|\lambda_j\rangle=\frac{2\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle}{\lambda_i+\lambda_j}
    \end{align}

    for $\lambda_i~(\lambda_j) \neq 0$. If $\lambda_i=\lambda_j=0$, the entry of SLD is set to be zero.

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        -- Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **rep:** `string`
        -- The basis for the SLDs. Options are:  
        "original" (default) -- it means the basis is the same with the input density 
        matrix (rho).  
        "eigen" -- it means the basis is the same with theeigenspace of the density
        matrix (rho).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **SLD(s):** `matrix or list`
        --For single parameter estimation (the length of drho is equal to one), the
        output is a matrix and for multiparameter estimation (the length of drho 
        is more than one), it returns a list.
    """

    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    para_num = len(drho)
    dim = len(rho)
    SLD = [[] for i in range(0, para_num)]

    purity = np.trace(np.dot(rho, rho))

    if np.abs(1 - purity) < eps:
        SLD_org = [[] for i in range(0, para_num)]
        for para_i in range(0, para_num):
            SLD_org[para_i] = 2 * drho[para_i]

            if rep == "original":
                SLD[para_i] = SLD_org[para_i]
            elif rep == "eigen":
                val, vec = np.linalg.eig(rho)
                val = np.real(val)
                SLD[para_i] = np.dot(
                    vec.conj().transpose(), np.dot(SLD_org[para_i], vec)
                )
            else:
                raise ValueError("{!r} is not a valid value for rep, supported values are 'original' and 'eigen'.".format(rep))
        if para_num == 1:
            return SLD[0]
        else:
            return SLD
    else:
        val, vec = np.linalg.eig(rho)
        val = np.real(val)
        for para_i in range(0, para_num):
            SLD_eig = np.array(
                [[0.0 + 0.0 * 1.0j for i in range(0, dim)] for i in range(0, dim)]
            )
            for fi in range(0, dim):
                for fj in range(0, dim):
                    if val[fi] + val[fj] > eps:
                        SLD_eig[fi][fj] = (
                            2
                            * np.dot(
                                vec[:, fi].conj().transpose(),
                                np.dot(drho[para_i], vec[:, fj]),
                            )
                            / (val[fi] + val[fj])
                        )
            SLD_eig[SLD_eig == np.inf] = 0.0

            if rep == "original":
                SLD[para_i] = np.dot(vec, np.dot(SLD_eig, vec.conj().transpose()))
            elif rep == "eigen":
                SLD[para_i] = SLD_eig
            else:
                raise ValueError("{!r} is not a valid value for rep, supported values are 'original' and 'eigen'.".format(rep))

        if para_num == 1:
            return SLD[0]
        else:
            return SLD


def RLD(rho, drho, rep="original", eps=1e-8):
    r"""
    Calculation of the right logarithmic derivative (RLD) for a density matrix.
    The RLD operator defined by $\partial_{a}\rho=\rho \mathcal{R}_a$
    with $\rho$ the parameterized density matrix. 
    \begin{align}
    \langle\lambda_i| \mathcal{R}_{a} |\lambda_j\rangle=\frac{1}{\lambda_i}\langle\lambda_i| 
    \partial_a\rho |\lambda_j\rangle 
    \end{align}

    for $\lambda_i\neq 0$ is the $ij$th entry of RLD.

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        -- Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **rep:** `string`
        -- The basis for the RLD(s). Options are:  
        "original" (default) -- it means the basis is the same with the input density 
        matrix (rho).  
        "eigen" -- it means the basis is the same with the eigenspace of the density 
        matrix (rho).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **RLD(s):** `matrix or list`
        -- For single parameter estimation (the length of drho is equal to one), the output 
        is a matrix and for multiparameter estimation (the length of drho is more than one), 
        it returns a list.
    """
    
    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    para_num = len(drho)
    dim = len(rho)
    RLD = [[] for i in range(0, para_num)]

    val, vec = np.linalg.eig(rho)
    val = np.real(val)
    for para_i in range(0, para_num):
        RLD_eig = np.array(
            [[0.0 + 0.0 * 1.0j for i in range(0, dim)] for i in range(0, dim)]
        )
        for fi in range(0, dim):
            for fj in range(0, dim):
                term_tp = np.dot(vec[:, fi].conj().transpose(),np.dot(drho[para_i], vec[:, fj]))
                if np.abs(val[fi]) > eps:
                    RLD_eig[fi][fj] = (term_tp/val[fi])
                else:
                    if np.abs(term_tp) < eps:
                        raise ValueError("The RLD does not exist. It only exist when the support of drho is contained in the support of rho.",
            )
        RLD_eig[RLD_eig == np.inf] = 0.0

        if rep == "original":
            RLD[para_i] = np.dot(vec, np.dot(RLD_eig, vec.conj().transpose()))
        elif rep == "eigen":
            RLD[para_i] = RLD_eig
        else:
            raise ValueError("{!r} is not a valid value for rep, supported values are 'original' and 'eigen'.".format(rep))
    if para_num == 1:
        return RLD[0]
    else:
        return RLD


def LLD(rho, drho, rep="original", eps=1e-8):
    r"""
    Calculation of the left logarithmic derivative (LLD) for a density matrix $\rho$.
    The LLD operator is defined by $\partial_{a}\rho=\mathcal{R}_a^{\dagger}\rho$. 
    The entries of LLD can be calculated as 
    \begin{align}
    \langle\lambda_i| \mathcal{R}_{a}^{\dagger} |\lambda_j\rangle=\frac{1}{\lambda_j}\langle\lambda_i| 
    \partial_a\rho |\lambda_j\rangle 
    \end{align}

    for $\lambda_j\neq 0$.

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        -- Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **rep:** `string`
        -- The basis for the LLD(s). Options are:  
        "original" (default) -- it means the basis is the same with the input density 
        matrix (rho).  
        "eigen" -- it means the basis is the same with the eigenspace of the density 
        matrix (rho).

    > **eps:** float
        -- Machine epsilon.

    Returns
    ----------
    **LLD(s):** `matrix or list`
        -- For single parameter estimation (the length of drho is equal to one), the output 
        is a matrix and for multiparameter estimation (the length of drho is more than one), 
        it returns a list.
    """

    if type(drho) != list:
        raise TypeError("Please make sure drho is a list!")

    para_num = len(drho)
    dim = len(rho)
    LLD = [[] for i in range(0, para_num)]

    val, vec = np.linalg.eig(rho)
    val = np.real(val)
    for para_i in range(0, para_num):
        LLD_eig = np.array(
            [[0.0 + 0.0 * 1.0j for i in range(0, dim)] for i in range(0, dim)]
        )
        for fi in range(0, dim):
            for fj in range(0, dim):
                term_tp = np.dot(vec[:, fi].conj().transpose(), np.dot(drho[para_i], vec[:, fj]),)
                if np.abs(val[fj]) > eps:
                    LLD_eig_tp = (term_tp/val[fj])
                    LLD_eig[fj][fi] = LLD_eig_tp.conj()
                else: 
                    if np.abs(term_tp) < eps:
                        raise ValueError("The LLD does not exist. It only exist when the support of drho is contained in the support of rho.",
            )
        LLD_eig[LLD_eig == np.inf] = 0.0

        if rep == "original":
            LLD[para_i] = np.dot(vec, np.dot(LLD_eig, vec.conj().transpose()))
        elif rep == "eigen":
            LLD[para_i] = LLD_eig
        else:
            raise ValueError("{!r} is not a valid value for rep, supported values are 'original' and 'eigen'.".format(rep))

    if para_num == 1:
        return LLD[0]
    else:
        return LLD


def QFIM(rho, drho, LDtype="SLD", exportLD=False, eps=1e-8):
    r"""
    Calculation of the quantum Fisher information (QFI) and quantum Fisher 
    information matrix (QFIM) for all types. The entry of QFIM $\mathcal{F}$
    is defined as
    \begin{align}
    \mathcal{F}_{ab}=\frac{1}{2}\mathrm{Tr}(\rho\{L_a, L_b\})
    \end{align}

    with $L_a, L_b$ are SLD operators and 

    and 
    \begin{align}
    \mathcal{F}_{ab}=\mathrm{Tr}(\rho \mathcal{R}_a \mathcal{R}^{\dagger}_b)
    \end{align}

    with $\mathcal{R}_a$ the RLD or LLD operator.

    Parameters
    ----------
    > **rho:** `matrix`
        -- Density matrix.

    > **drho:** `list`
        Derivatives of the density matrix on the unknown parameters to be 
        estimated. For example, drho[0] is the derivative vector on the first 
        parameter.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **exportLD:** `bool`
        -- Whether or not to export the values of logarithmic derivatives. If set True
        then the the values of logarithmic derivatives will be exported.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM:** `float or matrix` 
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    if type(drho) != list:
        raise TypeError("Please make sure drho is a list")

    para_num = len(drho)

    # single parameter estimation
    if para_num == 1:
        if LDtype == "SLD":
            LD_tp = SLD(rho, drho, eps=eps)
            SLD_ac = np.dot(LD_tp, LD_tp) + np.dot(LD_tp, LD_tp)
            QFIM_res = np.real(0.5 * np.trace(np.dot(rho, SLD_ac)))
        elif LDtype == "RLD":
            LD_tp = RLD(rho, drho, eps=eps)
            QFIM_res = np.real(
                np.trace(np.dot(rho, np.dot(LD_tp, LD_tp.conj().transpose())))
            )
        elif LDtype == "LLD":
            LD_tp = LLD(rho, drho, eps=eps)
            QFIM_res = np.real(
                np.trace(np.dot(rho, np.dot(LD_tp, LD_tp.conj().transpose())))
            )
        else:
            raise ValueError("{!r} is not a valid value for LDtype, supported values are 'SLD', 'RLD' and 'LLD'.".format(LDtype))

    # multiparameter estimation
    else:
        if LDtype == "SLD":
            QFIM_res = np.zeros([para_num, para_num])
            LD_tp = SLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    SLD_ac = np.dot(LD_tp[para_i], LD_tp[para_j]) + np.dot(
                        LD_tp[para_j], LD_tp[para_i]
                    )
                    QFIM_res[para_i][para_j] = np.real(
                        0.5 * np.trace(np.dot(rho, SLD_ac))
                    )
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
        elif LDtype == "RLD":
            QFIM_res = np.zeros((para_num, para_num), dtype=np.complex128)
            LD_tp = RLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.trace(
                            np.dot(
                                rho,
                                np.dot(LD_tp[para_i], LD_tp[para_j].conj().transpose()),
                            )
                        )
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j].conj()
        elif LDtype == "LLD":
            QFIM_res = np.zeros((para_num, para_num), dtype=np.complex128)
            LD_tp = LLD(rho, drho, eps=eps)
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.trace(
                            np.dot(
                                rho,
                                np.dot(LD_tp[para_i], LD_tp[para_j].conj().transpose()),
                            )
                        )
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j].conj()
        else:
            raise ValueError("{!r} is not a valid value for LDtype, supported values are 'SLD', 'RLD' and 'LLD'.".format(LDtype))

    if exportLD == False:
        return QFIM_res
    else:
        return QFIM_res, LD_tp


def QFIM_Kraus(rho0, K, dK, LDtype="SLD", exportLD=False, eps=1e-8):
    """
    Calculation of the quantum Fisher information (QFI) and quantum Fisher 
    information matrix (QFIM) with Kraus operator(s) for all types.

    Parameters
    ----------
    > **rho0:** `matrix`
        -- Initial state (density matrix).

    > **K:** `list`
        -- Kraus operator(s).

    > **dK:** `list` 
        -- Derivatives of the Kraus operator(s) on the unknown parameters to be 
        estimated.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **exportLD:** `bool`
        -- Whether or not to export the values of logarithmic derivatives. If set True
        then the the values of logarithmic derivatives will be exported.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM:** `float or matrix`
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    dK = [[dK[i][j] for i in range(len(K))] for j in range(len(dK[0]))]
    rho = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K])
    drho = [
                sum(
                    [
                        (
                            np.dot(dKi, np.dot(rho0, Ki.conj().T))
                            + np.dot(Ki, np.dot(rho0, dKi.conj().T))
                        )
                        for (Ki, dKi) in zip(K, dKj)
                    ]
                )
                for dKj in dK
            ]
    return QFIM(rho, drho, LDtype=LDtype, exportLD=exportLD, eps=eps)


def QFIM_Bloch(r, dr, eps=1e-8):
    """
    Calculation of the SLD based quantum Fisher information (QFI) and quantum  
    Fisher information matrix (QFIM) in Bloch representation.

    Parameters
    ----------
    > **r:** `list`
        -- Parameterized Bloch vector.

    > **dr:** `list `
        -- Derivatives of the Bloch vector on the unknown parameters to be 
        estimated. For example, dr[0] is the derivative vector on the first 
        parameter.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM in Bloch representation:** `float or matrix`
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    if type(dr) != list:
        raise TypeError("Please make sure dr is a list")

    para_num = len(dr)
    QFIM_res = np.zeros([para_num, para_num])

    dim = int(np.sqrt(len(r) + 1))
    Lambda = suN_generator(dim)

    if dim == 2:
        #### single-qubit system ####
        r_norm = np.linalg.norm(r) ** 2
        if np.abs(r_norm - 1.0) < eps:
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.real(np.inner(dr[para_i], dr[para_j]))
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
        else:
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    QFIM_res[para_i][para_j] = np.real(
                        np.inner(dr[para_i], dr[para_j])
                        + np.inner(r, dr[para_i])
                        * np.inner(r, dr[para_j])
                        / (1 - r_norm)
                    )
                    QFIM_res[para_j][para_i] = QFIM_res[para_i][para_j]
    else:
        rho = np.identity(dim, dtype=np.complex128) / dim
        for di in range(dim**2 - 1):
            rho += np.sqrt(dim * (dim - 1) / 2) * r[di] * Lambda[di] / dim

        G = np.zeros((dim**2 - 1, dim**2 - 1), dtype=np.complex128)
        for row_i in range(dim**2 - 1):
            for col_j in range(row_i, dim**2 - 1):
                anti_commu = np.dot(Lambda[row_i], Lambda[col_j]) + np.dot(
                    Lambda[col_j], Lambda[row_i]
                )
                G[row_i][col_j] = 0.5 * np.trace(np.dot(rho, anti_commu))
                G[col_j][row_i] = G[row_i][col_j]

        mat_tp = G * dim / (2 * (dim - 1)) - np.dot(
            np.array(r).reshape(len(r), 1), np.array(r).reshape(1, len(r))
        )
        mat_inv = inv(mat_tp)

        for para_m in range(0, para_num):
            for para_n in range(para_m, para_num):
                QFIM_res[para_m][para_n] = np.real(
                    np.dot(
                        np.array(dr[para_n]).reshape(1, len(r)),
                        np.dot(mat_inv, np.array(dr[para_m]).reshape(len(r), 1)),
                    )[0][0]
                )
                QFIM_res[para_n][para_m] = QFIM_res[para_m][para_n]

    if para_num == 1:
        return QFIM_res[0][0]
    else:
        return QFIM_res


def QFIM_Gauss(R, dR, D, dD):
    """
    Calculation of the SLD based quantum Fisher information (QFI) and quantum 
    Fisher information matrix (QFIM) with gaussian states.

    Parameters
    ----------
    > **R:** `array` 
        -- First-order moment.

    > **dR:** `list`
        -- Derivatives of the first-order moment on the unknown parameters to be 
        estimated. For example, dR[0] is the derivative vector on the first 
        parameter.

    > **D:** `matrix`
        -- Second-order moment.

    > **dD:** `list`
        -- Derivatives of the second-order moment on the unknown parameters to be 
        estimated. For example, dD[0] is the derivative vector on the first 
        parameter.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QFI or QFIM with gaussian states:** `float or matrix`
        -- For single parameter estimation (the length of drho is equal to one), 
        the output is QFI and for multiparameter estimation (the length of drho 
        is more than one), it returns QFIM.
    """

    para_num = len(dR)
    m = int(len(R) / 2)
    QFIM_res = np.zeros([para_num, para_num])

    C = np.array(
        [
            [D[i][j] - R[i] * R[j] for j in range(2 * m)]
            for i in range(2 * m)
        ]
    )
    dC = [
        np.array(
            [
                [
                    dD[k][i][j] - dR[k][i] * R[j] - R[i] * dR[k][j]
                    for j in range(2 * m)
                ]
                for i in range(2 * m)
            ]
        )
        for k in range(para_num)
    ]

    C_sqrt = sqrtm(C)
    J = np.kron([[0, 1], [-1, 0]], np.eye(m))
    B = C_sqrt @ J @ C_sqrt
    P = np.eye(2 * m)
    P = np.vstack([P[:][::2], P[:][1::2]])
    T, Q = schur(B)
    vals = eigvals(B)
    c = vals[::2].imag
    Diag = np.diagflat(c**-0.5)
    S = inv(J @ C_sqrt @ Q @ P @ np.kron([[0, 1], [-1, 0]], -Diag)).T @ P.T

    sx = np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = np.array([[1.0, 0.0], [0.0, -1.0]])
    a_Gauss = [1j * sy, sz, np.eye(2), sx]

    es = [
        [np.eye(1, m**2, m * i + j).reshape(m, m) for j in range(m)] for i in range(m)
    ]

    As = [[np.kron(s, a_Gauss[i]) / np.sqrt(2) for s in es] for i in range(4)]
    gs = [
        [[[np.trace(inv(S) @ dC @ inv(S.T) @ aa.T) for aa in a] for a in A] for A in As]
        for dC in dC
    ]
    G = [np.zeros((2 * m, 2 * m)).astype(np.longdouble) for _ in range(para_num)]

    for i in range(para_num):
        for j in range(m):
            for k in range(m):
                for l in range(4):
                    G[i] += np.real(
                        gs[i][l][j][k]
                        / (4 * c[j] * c[k] + (-1) ** (l + 1))
                        * inv(S.T)
                        @ As[l][j][k]
                        @ inv(S)
                    )

    QFIM_res += np.real(
        [
            [np.trace(G[i] @ dC[j]) + dR[i] @ inv(C) @ dR[j] for j in range(para_num)]
            for i in range(para_num)
        ]
    )

    if para_num == 1:
        return QFIM_res[0][0]
    else:
        return QFIM_res
