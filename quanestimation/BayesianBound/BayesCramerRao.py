import numpy as np
from scipy import interpolate
from scipy.integrate import simpson, solve_bvp
from itertools import product
from quanestimation.AsymptoticBound.CramerRao import CFIM, QFIM
from quanestimation.Common.Common import SIC, extract_ele


def BCFIM(x, p, rho, drho, M=[], eps=1e-8):
    r"""
    Calculation of the Bayesian classical Fisher information (BCFI) and the 
    Bayesian classical Fisher information matrix (BCFIM) of the form
    \begin{align}
    \mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}
    \end{align}
    
    with $\mathcal{I}$ the CFIM and $p(\textbf{x})$ the prior distribution.

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **drho:** `multidimensional list`
        -- Derivatives of the parameterized density matrix (rho) with respect to the unknown
        parameters to be estimated.

    > **M:** `list of matrices`
        -- A set of positive operator-valued measure (POVM). The default measurement 
        is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **BCFI or BCFIM:** `float or matrix`
        -- For single parameter estimation (the length of x is equal to one), the output 
        is BCFI and for multiparameter estimation (the length of x is more than one), 
        it returns BCFIM.

    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    para_num = len(x)
    if para_num == 1:
        #### single parameter scenario ####
        if M == []:
            M = SIC(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        p_num = len(p)
        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        p_num = len(p)
        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        arr = [p[i] * F_tp[i] for i in range(p_num)]
        return simpson(arr, x[0])
    else:
        #### multiparameter scenario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)

        dim = len(rho_list[0])
        if M == []:
            M = SIC(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        F_list = [
            [[0.0 for i in range(len(p_list))] for j in range(para_num)]
            for k in range(para_num)
        ]
        for i in range(len(p_list)):
            F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]

        BCFIM_res = np.zeros([para_num, para_num])
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                arr = p * F_ij
                for si in reversed(range(para_num)):
                    arr = simpson(arr, x[si])
                BCFIM_res[para_i][para_j] = arr
                BCFIM_res[para_j][para_i] = arr
        return BCFIM_res


def BQFIM(x, p, rho, drho, LDtype="SLD", eps=1e-8):
    r"""
    Calculation of the Bayesian quantum Fisher information (BQFI) and the 
    Bayesian quantum Fisher information matrix (BQFIM) of the form
    \begin{align}
    \mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x}
    \end{align}
    
    with $\mathcal{F}$ the QFIM of all types and $p(\textbf{x})$ the prior distribution.

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **drho:** `multidimensional list`
        -- Derivatives of the parameterized density matrix (rho) with respect to the unknown
        parameters to be estimated.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **BQFI or BQFIM:** `float or matrix`
        -- For single parameter estimation (the length of x is equal to one), the output 
        is BQFI and for multiparameter estimation (the length of x is more than one), 
        it returns BQFIM.
    """

    para_num = len(x)
    if para_num == 1:
        #### single parameter scenario ####
        p_num = len(p)
        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], LDtype=LDtype, eps=eps)
        arr = [p[i] * F_tp[i] for i in range(p_num)]
        return simpson(arr, x[0])
    else:
        #### multiparameter scenario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)

        F_list = [
            [[0.0 for i in range(len(p_list))] for j in range(para_num)]
            for k in range(para_num)
        ]
        for i in range(len(p_list)):
            F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]

        BQFIM_res = np.zeros([para_num, para_num])
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                arr = p * F_ij
                for si in reversed(range(para_num)):
                    arr = simpson(arr, x[si])
                BQFIM_res[para_i][para_j] = arr
                BQFIM_res[para_j][para_i] = arr
        return BQFIM_res


def BCRB(x, p, dp, rho, drho, M=[], b=[], db=[], btype=1, eps=1e-8):
    r"""
    Calculation of the Bayesian Cramer-Rao bound (BCRB). The covariance matrix 
    with a prior distribution $p(\textbf{x})$ is defined as
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\int p(\textbf{x})\sum_y\mathrm{Tr}
    (\rho\Pi_y)(\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
    \mathrm{d}\textbf{x}
    \end{align}

    where $\textbf{x}=(x_0,x_1,\dots)^{\mathrm{T}}$ are the unknown parameters to be estimated
    and the integral $\int\mathrm{d}\textbf{x}:=\iiint\mathrm{d}x_0\mathrm{d}x_1\cdots$.
    $\{\Pi_y\}$ is a set of positive operator-valued measure (POVM) and $\rho$ represents 
    the parameterized density matrix.

    This function calculates three types BCRB. The first one is 
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})\left(B\mathcal{I}^{-1}B
    +\textbf{b}\textbf{b}^{\mathrm{T}}\right)\mathrm{d}\textbf{x},
    \end{align}

    where $\textbf{b}$ and $\textbf{b}'$ are the vectors of biase and its derivatives on parameters.
    $B$ is a diagonal matrix with the $i$th entry $B_{ii}=1+[\textbf{b}']_{i}$ and $\mathcal{I}$ 
    is the CFIM.

    The second one is
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \mathcal{B}\,\mathcal{I}_{\mathrm{Bayes}}^{-1}\,
    \mathcal{B}+\int p(\textbf{x})\textbf{b}\textbf{b}^{\mathrm{T}}\mathrm{d}\textbf{x},
    \end{align}

    where $\mathcal{B}=\int p(\textbf{x})B\mathrm{d}\textbf{x}$ is the average of $B$ and 
    $\mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}$ is 
    the average CFIM.

    The third one is
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})
    \mathcal{G}\left(\mathcal{I}_p+\mathcal{I}\right)^{-1}\mathcal{G}^{\mathrm{T}}\mathrm{d}\textbf{x}
    \end{align}

    with $[\mathcal{I}_{p}]_{ab}:=[\partial_a \ln p(\textbf{x})][\partial_b \ln p(\textbf{x})]$ and
    $\mathcal{G}_{ab}:=[\partial_b\ln p(\textbf{x})][\textbf{b}]_a+B_{aa}\delta_{ab}$.

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **drho:** `multidimensional list`
        -- Derivatives of the parameterized density matrix (rho) with respect to the unknown
        parameters to be estimated.

    > **M:** `list of matrices`
        -- A set of positive operator-valued measure (POVM). The default measurement 
        is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

    > **b:** `list`
        -- Vector of biases of the form $\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}$.
        
    > **db:** `list`
        -- Derivatives of b with respect to the unknown parameters to be estimated, It should be 
        expressed as $\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}$.

    > **btype:** `int (1, 2, 3)`
        -- Types of the BCRB. Options are:  
        1 (default) -- It means to calculate the first type of the BCRB.  
        2 -- It means to calculate the second type of the BCRB.
        3 -- It means to calculate the third type of the BCRB.

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **BCRB:** `float or matrix`
        -- For single parameter estimation (the length of x is equal to one), the 
        output is a float and for multiparameter estimation (the length of x is 
        more than one), it returns a matrix.

    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    para_num = len(x)
    if para_num == 1:
        #### single parameter scenario ####
        p_num = len(p)
        if not b:
            b = np.zeros(p_num)
            db = np.zeros(p_num)
        elif not db:
            db = np.zeros(p_num)

        if M == []:
            M = SIC(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(b[0]) == list or type(b[0]) == np.ndarray:
            b = b[0]
        if type(db[0]) == list or type(db[0]) == np.ndarray:
            db = db[0]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)

        if btype == 1:
            arr = [
                p[i] * ((1 + db[i]) ** 2 / F_tp[i] + b[i] ** 2) for i in range(p_num)
            ]
            F = simpson(arr, x[0])
            return F
        elif btype == 2:
            arr = [p[i] * F_tp[i] for i in range(p_num)]
            F1 = simpson(arr, x[0])
            arr2 = [p[j] * (1 + db[j]) for j in range(p_num)]
            B = simpson(arr2, x[0])
            arr3 = [p[k] * b[k] ** 2 for k in range(p_num)]
            bb = simpson(arr3, x[0])
            F = B**2 / F1 + bb
            return F
        elif btype == 3:
            I_tp = [np.real(dp[i] * dp[i] / p[i] ** 2) for i in range(p_num)]
            arr = [p[j]*(dp[j]*b[j]/p[j]+(1 + db[j]))**2 / (I_tp[j] + F_tp[j]) for j in range(p_num)]
            F = simpson(arr, x[0])
            return F
        else:
            raise NameError("NameError: btype should be choosen in {1, 2, 3}.")
    else:
        #### multiparameter scenario ####
        if not b:
            b, db = [], []
            for i in range(para_num):
                b.append(np.zeros(len(x[i])))
                db.append(np.zeros(len(x[i])))
        elif not db:
            db = []
            for i in range(para_num):
                db.append(np.zeros(len(x[i])))

        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        dp_ext = extract_ele(dp, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)
        b_pro = product(*b)
        db_pro = product(*db)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)
        dp_list = [dpi for dpi in dp_ext]

        b_list, db_list = [], []
        for b_ele, db_ele in zip(b_pro, db_pro):
            b_list.append([b_ele[i] for i in range(para_num)])
            db_list.append([db_ele[j] for j in range(para_num)])

        dim = len(rho_list[0])
        if M == []:
            M = SIC(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")
        if btype == 1:
            F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                F_inv = np.linalg.pinv(F_tp)
                B = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                term1 = np.dot(B, np.dot(F_inv, B))
                term2 = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = term1[pj][pk] + term2[pj][pk]

            res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p * F_ij
                    for si in reversed(range(para_num)):
                        arr = simpson(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        elif btype == 2:
            F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            B_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            bb_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                B_tp = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                bb_tp = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        B_list[pj][pk][i] = B_tp[pj][pk]
                        bb_list[pj][pk][i] = bb_tp[pj][pk]

            F_res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p * F_ij
                    for si in reversed(range(para_num)):
                        arr = simpson(arr, x[si])
                    F_res[para_i][para_j] = arr
                    F_res[para_j][para_i] = arr
            B_res = np.zeros([para_num, para_num])
            bb_res = np.zeros([para_num, para_num])
            for para_m in range(para_num):
                for para_n in range(para_num):
                    B_mn = np.array(B_list[para_m][para_n]).reshape(p_shape)
                    bb_mn = np.array(bb_list[para_m][para_n]).reshape(p_shape)
                    arr2 = p * B_mn
                    arr3 = p * bb_mn
                    for sj in reversed(range(para_num)):
                        arr2 = simpson(arr2, x[sj])
                        arr3 = simpson(arr3, x[sj])
                    B_res[para_m][para_n] = arr2
                    bb_res[para_m][para_n] = arr3
            res = np.dot(B_res, np.dot(np.linalg.pinv(F_res), B_res)) + bb_res
            return res
        elif btype == 3:
            F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            for i in range(len(p_list)):
                F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
                I_tp = np.zeros((para_num, para_num))
                G_tp = np.zeros((para_num, para_num))
                for pm in range(para_num):
                    for pn in range(para_num):
                        if pm == pn:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]+(1.0 + db_list[i][pm])
                        else:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]
                        I_tp[pm][pn] = dp_list[i][pm] * dp_list[i][pn] / p_list[i] ** 2

                F_tot = np.dot(G_tp, np.dot(np.linalg.pinv(F_tp + I_tp), G_tp.T))
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tot[pj][pk]

            res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p * F_ij
                    for si in reversed(range(para_num)):
                        arr = simpson(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        else:
            raise NameError("NameError: btype should be choosen in {1, 2, 3}.")


def BQCRB(x, p, dp, rho, drho, b=[], db=[], btype=1, LDtype="SLD", eps=1e-8):
    r"""
    Calculation of the Bayesian quantum Cramer-Rao bound (BQCRB). The covariance matrix 
    with a prior distribution $p(\textbf{x})$ is defined as
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\int p(\textbf{x})\sum_y\mathrm{Tr}
    (\rho\Pi_y)(\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
    \mathrm{d}\textbf{x}
    \end{align}

    where $\textbf{x}=(x_0,x_1,\dots)^{\mathrm{T}}$ are the unknown parameters to be estimated
    and the integral $\int\mathrm{d}\textbf{x}:=\iiint\mathrm{d}x_0\mathrm{d}x_1\cdots$.
    $\{\Pi_y\}$ is a set of positive operator-valued measure (POVM) and $\rho$ represent 
    the parameterized density matrix.

    This function calculates three types of the BQCRB. The first one is
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq\int p(\textbf{x})\left(B\mathcal{F}^{-1}B
    +\textbf{b}\textbf{b}^{\mathrm{T}}\right)\mathrm{d}\textbf{x},
    \end{align}
        
    where $\textbf{b}$ and $\textbf{b}'$ are the vectors of biase and its derivatives on parameters.
    $B$ is a diagonal matrix with the $i$th entry $B_{ii}=1+[\textbf{b}']_{i}$ and $\mathcal{F}$
    is the QFIM for all types.

    The second one is
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \mathcal{B}\,\mathcal{F}_{\mathrm{Bayes}}^{-1}\,
    \mathcal{B}+\int p(\textbf{x})\textbf{b}\textbf{b}^{\mathrm{T}}\mathrm{d}\textbf{x},
    \end{align}

    where $\mathcal{B}=\int p(\textbf{x})B\mathrm{d}\textbf{x}$ is the average of $B$ and 
    $\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x}$ is 
    the average QFIM.

    The third one is
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})
    \mathcal{G}\left(\mathcal{I}_p+\mathcal{F}\right)^{-1}\mathcal{G}^{\mathrm{T}}\mathrm{d}\textbf{x}
    \end{align}

    with $[\mathcal{I}_{p}]_{ab}:=[\partial_a \ln p(\textbf{x})][\partial_b \ln p(\textbf{x})]$ and
    $\mathcal{G}_{ab}:=[\partial_b\ln p(\textbf{x})][\textbf{b}]_a+B_{aa}\delta_{ab}$.

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **drho:** `multidimensional list`
        -- Derivatives of the parameterized density matrix (rho) with respect to the unknown
        parameters to be estimated.

    > **b:** `list`
        -- Vector of biases of the form $\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}$.
        
    > **db:** `list`
        -- Derivatives of b with respect to the unknown parameters to be estimated, It should be 
        expressed as $\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}$.

    > **btype:** `int (1, 2, 3)`
        -- Types of the BQCRB. Options are:  
        1 (default) -- It means to calculate the first type of the BQCRB.  
        2 -- It means to calculate the second type of the BQCRB.
        3 -- It means to calculate the third type of the BCRB.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **BQCRB:** `float or matrix`
        -- For single parameter estimation (the length of x is equal to one), the 
        output is a float and for multiparameter estimation (the length of x is 
        more than one), it returns a matrix.
    """

    para_num = len(x)

    if para_num == 1:
        #### single parameter scenario ####
        p_num = len(p)

        if not b:
            b = np.zeros(p_num)
            db = np.zeros(p_num)
        elif not db:
            db = np.zeros(p_num)

        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(b[0]) == list or type(b[0]) == np.ndarray:
            b = b[0]
        if type(db[0]) == list or type(db[0]) == np.ndarray:
            db = db[0]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], LDtype=LDtype, eps=eps)

        if btype == 1:
            arr = [
                p[i] * ((1 + db[i]) ** 2 / F_tp[i] + b[i] ** 2) for i in range(p_num)
            ]
            F = simpson(arr, x[0])
            return F
        elif btype == 2:
            arr2 = [p[i] * F_tp[i] for i in range(p_num)]
            F2 = simpson(arr2, x[0])
            arr2 = [p[j] * (1 + db[j]) for j in range(p_num)]
            B = simpson(arr2, x[0])
            arr3 = [p[k] * b[k] ** 2 for k in range(p_num)]
            bb = simpson(arr3, x[0])
            F = B**2 / F2 + bb
            return F
        elif btype == 3:
            I_tp = [np.real(dp[i] * dp[i] / p[i] ** 2) for i in range(p_num)]
            arr = [p[j]*(dp[j]*b[j]/p[j]+(1 + db[j]))**2 / (I_tp[j] + F_tp[j]) for j in range(p_num)]
            F = simpson(arr, x[0])
            return F
        else:
            raise NameError("NameError: btype should be choosen in {1, 2, 3}.")
    else:
        #### multiparameter scenario ####
        if not b:
            b, db = [], []
            for i in range(para_num):
                b.append(np.zeros(len(x[i])))
                db.append(np.zeros(len(x[i])))
        elif not db:
            db = []
            for i in range(para_num):
                db.append(np.zeros(len(x[i])))

        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        dp_ext = extract_ele(dp, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)
        b_pro = product(*b)
        db_pro = product(*db)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)
        dp_list = [dpi for dpi in dp_ext]

        b_list, db_list = [], []
        for b_ele, db_ele in zip(b_pro, db_pro):
            b_list.append([b_ele[i] for i in range(para_num)])
            db_list.append([db_ele[j] for j in range(para_num)])

        if btype == 1:
            F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
                F_inv = np.linalg.pinv(F_tp)
                B = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                term1 = np.dot(B, np.dot(F_inv, B))
                term2 = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = term1[pj][pk] + term2[pj][pk]

            res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p * F_ij
                    for si in reversed(range(para_num)):
                        arr = simpson(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        elif btype == 2:
            F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            B_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            bb_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
                B_tp = np.diag([(1.0 + db_list[i][j]) for j in range(para_num)])
                bb_tp = np.dot(
                    np.array(b_list[i]).reshape(para_num, 1),
                    np.array(b_list[i]).reshape(1, para_num),
                )
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tp[pj][pk]
                        B_list[pj][pk][i] = B_tp[pj][pk]
                        bb_list[pj][pk][i] = bb_tp[pj][pk]

            F_res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p * F_ij
                    for si in reversed(range(para_num)):
                        arr = simpson(arr, x[si])
                    F_res[para_i][para_j] = arr
                    F_res[para_j][para_i] = arr
            B_res = np.zeros([para_num, para_num])
            bb_res = np.zeros([para_num, para_num])
            for para_m in range(para_num):
                for para_n in range(para_num):
                    B_mn = np.array(B_list[para_m][para_n]).reshape(p_shape)
                    bb_mn = np.array(bb_list[para_m][para_n]).reshape(p_shape)
                    arr2 = p * B_mn
                    arr3 = p * bb_mn
                    for sj in reversed(range(para_num)):
                        arr2 = simpson(arr2, x[sj])
                        arr3 = simpson(arr3, x[sj])
                    B_res[para_m][para_n] = arr2
                    bb_res[para_m][para_n] = arr3
            res = np.dot(B_res, np.dot(np.linalg.pinv(F_res), B_res)) + bb_res
            return res
        elif btype == 3:
            F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
            for i in range(len(p_list)):
                F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
                I_tp = np.zeros((para_num, para_num))
                G_tp = np.zeros((para_num, para_num))
                for pm in range(para_num):
                    for pn in range(para_num):
                        if pm == pn:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]+(1.0 + db_list[i][pm])
                        else:
                            G_tp[pm][pn] = dp_list[i][pn]*b_list[i][pm]/p_list[i]
                        I_tp[pm][pn] = dp_list[i][pm] * dp_list[i][pn] / p_list[i] ** 2

                F_tot = np.dot(G_tp, np.dot(np.linalg.pinv(F_tp + I_tp), G_tp.T))
                for pj in range(para_num):
                    for pk in range(para_num):
                        F_list[pj][pk][i] = F_tot[pj][pk]

            res = np.zeros([para_num, para_num])
            for para_i in range(0, para_num):
                for para_j in range(para_i, para_num):
                    F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                    arr = p * F_ij
                    for si in reversed(range(para_num)):
                        arr = simpson(arr, x[si])
                    res[para_i][para_j] = arr
                    res[para_j][para_i] = arr
            return res
        else:
            raise NameError("NameError: btype should be choosen in {1, 2, 3}.")


def VTB(x, p, dp, rho, drho, M=[], eps=1e-8):
    r"""
    Calculation of the Bayesian version of Cramer-Rao bound introduced by
    Van Trees (VTB). The covariance matrix with a prior distribution $p(\textbf{x})$ 
    is defined as
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\int p(\textbf{x})\sum_y\mathrm{Tr}
    (\rho\Pi_y)(\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
    \mathrm{d}\textbf{x}
    \end{align}

    where $\textbf{x}=(x_0,x_1,\dots)^{\mathrm{T}}$ are the unknown parameters to be estimated
    and the integral $\int\mathrm{d}\textbf{x}:=\iiint\mathrm{d}x_0\mathrm{d}x_1\cdots$.
    $\{\Pi_y\}$ is a set of positive operator-valued measure (POVM) and $\rho$ represent 
    the parameterized density matrix.

    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \left(\mathcal{I}_{\mathrm{prior}}
    +\mathcal{I}_{\mathrm{Bayes}}\right)^{-1},
    \end{align}

    where $\mathcal{I}_{\mathrm{prior}}=\int p(\textbf{x})\mathcal{I}_{p}\mathrm{d}\textbf{x}$ 
    is the CFIM for $p(\textbf{x})$ and 
    $\mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}$ is the 
    average CFIM.

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **dp:** `list`
        -- Derivatives of the prior distribution with respect to the unknown parameters 
        to be estimated. For example, dp[0] is the derivative vector with respect to the first 
        parameter.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **drho:** `multidimensional list`
        -- Derivatives of the parameterized density matrix (rho) with respect to the 
        unknown parameters to be estimated.

    > **M:** `list of matrices`
        -- A set of positive operator-valued measure (POVM). The default measurement 
        is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **VTB:** `float or matrix`
        -- For single parameter estimation (the length of x is equal to one), the 
        output is a float and for multiparameter estimation (the length of x is 
        more than one), it returns a matrix.

    **Note:** 
        SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
        which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
        solutions.html).
    """

    para_num = len(x)
    p_num = len(p)

    if para_num == 1:
        #### single parameter scenario ####
        if M == []:
            M = SIC(len(rho[0]))
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")

        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(dp[0]) == list or type(dp[0]) == np.ndarray:
            dp = [dp[i][0] for i in range(p_num)]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = CFIM(rho[m], [drho[m]], M=M, eps=eps)


        arr1 = [np.real(dp[i] * dp[i] / p[i]) for i in range(p_num)]
        I = simpson(arr1, x[0])
        arr2 = [np.real(F_tp[j] * p[j]) for j in range(p_num)]
        F = simpson(arr2, x[0])
        return 1.0 / (I + F)
    else:
        #### multiparameter scenario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        dp_ext = extract_ele(dp, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)
        dp_list = [dpi for dpi in dp_ext]

        dim = len(rho_list[0])
        if M == []:
            M = SIC(dim)
        else:
            if type(M) != list:
                raise TypeError("Please make sure M is a list!")
        
        F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
        I_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
        for i in range(len(p_list)):
            F_tp = CFIM(rho_list[i], drho_list[i], M=M, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]
                    I_list[pj][pk][i] = (
                            dp_list[i][pj] * dp_list[i][pk] / p_list[i] ** 2
                        )

        F_res = np.zeros([para_num, para_num])
        I_res = np.zeros([para_num, para_num])
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                I_ij = np.array(I_list[para_i][para_j]).reshape(p_shape)
                arr1 = p * F_ij
                arr2 = p * I_ij
                for si in reversed(range(para_num)):
                    arr1 = simpson(arr1, x[si])
                    arr2 = simpson(arr2, x[si])
                F_res[para_i][para_j] = arr1
                F_res[para_j][para_i] = arr1
                I_res[para_i][para_j] = arr2
                I_res[para_j][para_i] = arr2
        return np.linalg.pinv(F_res + I_res)

def QVTB(x, p, dp, rho, drho, LDtype="SLD", eps=1e-8):
    r"""
    Calculation of the Bayesian version of quantum Cramer-Rao bound introduced 
    by Van Trees (QVTB). The covariance matrix with a prior distribution p(\textbf{x}) 
    is defined as
    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\int p(\textbf{x})\sum_y\mathrm{Tr}
    (\rho\Pi_y)(\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}
    \mathrm{d}\textbf{x}
    \end{align}

    where $\textbf{x}=(x_0,x_1,\dots)^{\mathrm{T}}$ are the unknown parameters to be estimated
    and the integral $\int\mathrm{d}\textbf{x}:=\iiint\mathrm{d}x_0\mathrm{d}x_1\cdots$.
    $\{\Pi_y\}$ is a set of positive operator-valued measure (POVM) and $\rho$ represent 
    the parameterized density matrix.

    \begin{align}
    \mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \left(\mathcal{I}_{\mathrm{prior}}
    +\mathcal{F}_{\mathrm{Bayes}}\right)^{-1},
    \end{align}

    where $\mathcal{I}_{\mathrm{prior}}=\int p(\textbf{x})\mathcal{I}_{p}\mathrm{d}\textbf{x}$ is 
    the CFIM for $p(\textbf{x})$ and $\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}
    \mathrm{d}\textbf{x}$ is the average QFIM of all types.

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** multidimensional array
        -- The prior distribution.

    > **dp:** `list`
        -- Derivatives of the prior distribution with respect to the unknown parameters to to
        estimated. For example, dp[0] is the derivative vector with respect to the first 
        parameter.

    > **rho:** `multidimensional list`
        -- Parameterized density matrix.

    > **drho:** `multidimensional list`
        -- Derivatives of the parameterized density matrix (rho) with respect to the unknown
        parameters to be estimated.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QVTB:** `float or matrix`
        -- For single parameter estimation (the length of x is equal to one), the 
        output is a float and for multiparameter estimation (the length of x is 
        more than one), it returns a matrix.
    """
    para_num = len(x)
    p_num = len(p)

    if para_num == 1:
        if type(drho[0]) == list:
            drho = [drho[i][0] for i in range(p_num)]
        if type(dp[0]) == list or type(dp[0]) == np.ndarray:
            dp = [dp[i][0] for i in range(p_num)]

        F_tp = np.zeros(p_num)
        for m in range(p_num):
            F_tp[m] = QFIM(rho[m], [drho[m]], LDtype=LDtype, eps=eps)

        arr1 = [np.real(dp[i] * dp[i] / p[i]) for i in range(p_num)]
        I = simpson(arr1, x[0])
        arr2 = [np.real(F_tp[j] * p[j]) for j in range(p_num)]
        F = simpson(arr2, x[0])
        return 1.0 / (I + F)
    else:
        #### multiparameter scenario ####
        p_shape = np.shape(p)
        p_ext = extract_ele(p, para_num)
        dp_ext = extract_ele(dp, para_num)
        rho_ext = extract_ele(rho, para_num)
        drho_ext = extract_ele(drho, para_num)

        p_list, rho_list, drho_list = [], [], []
        for p_ele, rho_ele, drho_ele in zip(p_ext, rho_ext, drho_ext):
            p_list.append(p_ele)
            rho_list.append(rho_ele)
            drho_list.append(drho_ele)
        dp_list = [dpi for dpi in dp_ext]

        F_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
        I_list = [
                [[0.0 for i in range(len(p_list))] for j in range(para_num)]
                for k in range(para_num)
            ]
        for i in range(len(p_list)):
            F_tp = QFIM(rho_list[i], drho_list[i], LDtype=LDtype, eps=eps)
            for pj in range(para_num):
                for pk in range(para_num):
                    F_list[pj][pk][i] = F_tp[pj][pk]
                    I_list[pj][pk][i] = (
                            dp_list[i][pj] * dp_list[i][pk] / p_list[i] ** 2
                        )

        F_res = np.zeros([para_num, para_num])
        I_res = np.zeros([para_num, para_num])
        for para_i in range(0, para_num):
            for para_j in range(para_i, para_num):
                F_ij = np.array(F_list[para_i][para_j]).reshape(p_shape)
                I_ij = np.array(I_list[para_i][para_j]).reshape(p_shape)
                arr1 = p * F_ij
                arr2 = p * I_ij
                for si in reversed(range(para_num)):
                    arr1 = simpson(arr1, x[si])
                    arr2 = simpson(arr2, x[si])
                F_res[para_i][para_j] = arr1
                F_res[para_j][para_i] = arr1
                I_res[para_i][para_j] = arr2
                I_res[para_j][para_i] = arr2
        return np.linalg.pinv(F_res + I_res)


def OBB_func(x, y, t, J, F):
    interp_J = interpolate.interp1d(t, (J))
    interp_F = interpolate.interp1d(t, (F))
    J_tp, F_tp = interp_J(x), interp_F(x)
    return np.vstack((y[1], -J_tp * y[1] + F_tp * y[0] - J_tp))


def boundary_condition(ya, yb):
    return np.array([ya[1] + 1.0, yb[1] + 1.0])


def OBB(x, p, dp, rho, drho, d2rho, LDtype="SLD", eps=1e-8):
    r"""
    Calculation of the optimal biased bound based on the first type of the BQCRB 
    in the case of single parameter estimation. The expression of OBB with a 
    prior distribution $p(x)$ is
    \begin{align}
    \mathrm{var}(\hat{x},\{\Pi_y\})\geq\int p(x)\left(\frac{(1+b')^2}{F}
    +b^2\right)\mathrm{d}x,
    \end{align}
        
    where $b$ and $b'$ are the vector of biase and its derivative on $x$. 
    $F$ is the QFI for all types.

    Parameters
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `array`
        -- The prior distribution.

    > **dp:** `list`
        -- Derivatives of the prior distribution with respect to the unknown parameters to to
        estimated. For example, dp[0] is the derivative vector with respect to the first 
        parameter.

    > **rho:** `list`
        -- Parameterized density matrix.

    > **drho:** `list`
        -- Derivatives of the parameterized density matrix (rho) with respect to the unknown
        parameters to be estimated.

    > **drho:** `list`
        -- Second order Derivatives of the parameterized density matrix (rho) with respect to the 
        unknown parameters to be estimated.

    > **LDtype:** `string`
        -- Types of QFI (QFIM) can be set as the objective function. Options are:  
        "SLD" (default) -- QFI (QFIM) based on symmetric logarithmic derivative (SLD).  
        "RLD" -- QFI (QFIM) based on right logarithmic derivative (RLD).  
        "LLD" -- QFI (QFIM) based on left logarithmic derivative (LLD).

    > **eps:** `float`
        -- Machine epsilon.

    Returns
    ----------
    **QVTB:** `float or matrix`
        -- For single parameter estimation (the length of x is equal to one), the 
        output is a float and for multiparameter estimation (the length of x is 
        more than one), it returns a matrix.
    """

    #### single parameter scenario ####
    p_num = len(p)

    if type(drho[0]) == list:
        drho = [drho[i][0] for i in range(p_num)]
    if type(d2rho[0]) == list:
        d2rho = [d2rho[i][0] for i in range(p_num)]
    if type(dp[0]) == list or type(dp[0]) == np.ndarray:
        dp = [dp[i][0] for i in range(p_num)]
    if type(x[0]) != float or type(x[0]) != int:
        x = x[0]

    F, J = np.zeros(p_num), np.zeros(p_num)
    bias, dbias = np.zeros(p_num), np.zeros(p_num)
    for m in range(p_num):
        f, LD = QFIM(rho[m], [drho[m]], LDtype=LDtype, exportLD=True, eps=eps)
        F[m] = f
        term1 = np.dot(d2rho[m], LD)
        term2 = np.dot(d2rho[m], LD.conj().T)
        term3 = np.dot(np.dot(LD, LD), drho[m])
        dF = np.real(np.trace(term1 + term2 - term3))
        J[m] = dp[m] / p[m] - dF / f

    y_guess = np.zeros((2, x.size))
    fun = lambda m, n: OBB_func(m, n, x, J, F)
    result = solve_bvp(fun, boundary_condition, x, y_guess)
    res = result.sol(x)
    bias, dbias = res[0], res[1]

    value = [p[i] * ((1 + dbias[i]) ** 2 / F[i] + bias[i] ** 2) for i in range(p_num)]
    return simpson(value, x)
