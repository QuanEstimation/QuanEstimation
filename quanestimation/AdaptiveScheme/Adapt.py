import numpy as np
from scipy.integrate import simpson
from itertools import product

from quanestimation.Common.Common import extract_ele, SIC
from quanestimation.MeasurementOpt.MeasurementStruct import MeasurementOpt
from quanestimation.Parameterization.GeneralDynamics import Lindblad
from quanestimation.AsymptoticBound.CramerRao import QFIM, CFIM


class Adapt:
    """
    Attributes
    ----------
    > **x:** `list`
        -- The regimes of the parameters for the integral.

    > **p:** `multidimensional array`
        -- The prior distribution.

    > **rho0:** `matrix`
        -- Initial state (density matrix).

    > **method:** `string`
        -- Choose the method for updating the tunable parameters (u). Options are:  
        "FOP" (default) -- Fix optimal point.  
        "MI" -- mutual information.
        
    > **savefile:** `bool`
        -- Whether or not to save all the posterior distributions.  
        If set `True` then three files "pout.npy", "xout.npy" and "y.npy" will be 
        generated including the posterior distributions, the estimated values, and
        the experimental results in the iterations. If set `False` the posterior 
        distribution in the final iteration, the estimated values and the experimental 
        results in all iterations will be saved in "pout.npy", "xout.npy" and "y.npy". 
        
    > **max_episode:** `int`
        -- The number of episodes.

    > **eps:** `float`
        -- Machine epsilon.
    """

    def __init__(self, x, p, rho0, method="FOP", savefile=False, max_episode=1000, eps=1e-8):

        self.x = x
        self.p = p
        self.rho0 = np.array(rho0, dtype=np.complex128)
        self.max_episode = max_episode
        self.eps = eps
        self.para_num = len(x)
        self.savefile = savefile
        self.method = method

    def dynamics(self, tspan, H, dH, Hc=[], ctrl=[], decay=[], dyn_method="expm"):
        r"""
        Dynamics of the density matrix of the form 
        
        \begin{align}
        \partial_t\rho &=\mathcal{L}\rho \nonumber \\
        &=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}
        \left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right),
        \end{align} 

        where $\rho$ is the evolved density matrix, H is the Hamiltonian of the 
        system, $\Gamma_i$ and $\gamma_i$ are the $i\mathrm{th}$ decay 
        operator and decay rate.

        Parameters
        ----------
        > **tspan:** `array`
            -- Time length for the evolution.

        > **H0:** `multidimensional list`
            -- Free Hamiltonian with respect to the values in x.

        > **dH:** `multidimensional list`
            -- Derivatives of the free Hamiltonian with respect to the unknown parameters 
            to be estimated.

        > **Hc:** `list`
            -- Control Hamiltonians.

        > **ctrl:** `list`
            -- Control coefficients.

        > **decay:** `list`
            -- Decay operators and the corresponding decay rates. Its input rule is 
            `decay=[[Gamma1, gamma1], [Gamma2,gamma2],...]`, where `Gamma1 (Gamma2)` 
            represents the decay operator and `gamma1 (gamma2)` is the corresponding 
            decay rate.

        > **dyn_method:** `string`
            -- Setting the method for solving the Lindblad dynamics. Options are:  
            "expm" (default) -- Matrix exponential.  
            "ode" -- Solving the differential equations directly.
        """

        self.tspan = tspan
        self.H = H
        self.dH = dH
        self.Hc = Hc
        self.ctrl = ctrl
        self.decay = decay

        self.dynamic_type = "dynamics"
        self.dyn_method = dyn_method

    def Kraus(self, K, dK):
        r"""
        Dynamics of the density matrix of the form 
        \begin{align}
        \rho=\sum_i K_i\rho_0K_i^{\dagger}
        \end{align}

        where $\rho$ is the evolved density matrix, $K_i$ is the Kraus operator.

        Parameters
        ----------
        > **K:** `multidimensional list`
            -- Kraus operator(s) with respect to the values in x.

        > **dK:** `multidimensional list`
            -- Derivatives of the Kraus operator(s) with respect to the unknown parameters 
            to be estimated.
        """

        self.K = K
        self.dK = dK

        self.dynamic_type = "Kraus"

    def CFIM(self, M=[], W=[]):
        r"""
        Choose CFI or $\mathrm{Tr}(WI^{-1})$ as the objective function. 
        In single parameter estimation the objective function is CFI and 
        in multiparameter estimation it will be $\mathrm{Tr}(WI^{-1})$.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.

        > **M:** `list of matrices`
            -- A set of positive operator-valued measure (POVM). The default measurement 
            is a set of rank-one symmetric informationally complete POVM (SIC-POVM).

        **Note:** 
            SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state 
            which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/
            solutions.html).
        """

        if M == []:
            M = SIC(len(self.rho0))
        if W == []:
            W = np.eye(len(self.x))
        self.W = W

        if self.dynamic_type == "dynamics":
            adaptive_dynamics(
                self.x,
                self.p,
                M,
                self.tspan,
                self.rho0,
                self.H,
                self.dH,
                self.decay,
                self.Hc,
                self.ctrl,
                W,
                self.max_episode,
                self.eps,
                self.savefile,
                self.method,
                dyn_method=self.dyn_method,
            )
        elif self.dynamic_type == "Kraus":
            adaptive_Kraus(
                self.x,
                self.p,
                M,
                self.rho0,
                self.K,
                self.dK,
                W,
                self.max_episode,
                self.eps,
                self.savefile,
                self.method
            )
        else:
            raise ValueError(
                "{!r} is not a valid value for type of dynamics, supported values are 'dynamics' and 'Kraus'.".format(
                    self.dynamic_type
                )
            )

    def Mopt(self, W=[]):
        r"""
        Measurement optimization for the optimal x.

        Parameters
        ----------
        > **W:** `matrix`
            -- Weight matrix.
        """

        if W == []:
            W = np.identity(self.para_num)
        else:
            W = W

        if self.dynamic_type == "dynamics":
            if self.para_num == 1:
                F = []
                if self.dyn_method == "expm":
                    for i in range(len(self.H)):
                        dynamics = Lindblad(
                            self.tspan,
                            self.rho0,
                            self.H[i],
                            self.dH[i],
                            decay=self.decay,
                            Hc=self.Hc,
                            ctrl=self.ctrl,
                        )
                        rho_tp, drho_tp = dynamics.expm()
                        rho, drho = rho_tp[-1], drho_tp[-1]
                        F_tp = QFIM(rho, drho)
                        F.append(F_tp)
                elif self.dyn_method == "ode":
                    for i in range(len(self.H)):
                        dynamics = Lindblad(
                            self.tspan,
                            self.rho0,
                            self.H[i],
                            self.dH[i],
                            decay=self.decay,
                            Hc=self.Hc,
                            ctrl=self.ctrl,
                        )
                        rho_tp, drho_tp = dynamics.ode()
                        rho, drho = rho_tp[-1], drho_tp[-1]
                        F_tp = QFIM(rho, drho)
                        F.append(F_tp)
                idx = np.argmax(F)
                H_res, dH_res = self.H[idx], self.dH[idx]
            else:
                p_ext = extract_ele(self.p, self.para_num)
                H_ext = extract_ele(self.H, self.para_num)
                dH_ext = extract_ele(self.dH, self.para_num)

                p_list, H_list, dH_list = [], [], []
                for p_ele, H_ele, dH_ele in zip(p_ext, H_ext, dH_ext):
                    p_list.append(p_ele)
                    H_list.append(H_ele)
                    dH_list.append(dH_ele)

                F = []
                if self.dyn_method == "expm":
                    for i in range(len(p_list)):
                        dynamics = Lindblad(
                            self.tspan,
                            self.rho0,
                            self.H_list[i],
                            self.dH_list[i],
                            decay=self.decay,
                            Hc=self.Hc,
                            ctrl=self.ctrl,
                        )
                        rho_tp, drho_tp = dynamics.expm()
                        rho, drho = rho_tp[-1], drho_tp[-1]
                        F_tp = QFIM(rho, drho)
                        if np.linalg.det(F_tp) < self.eps:
                            F.append(self.eps)
                        else:
                            F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
                elif self.dyn_method == "ode":
                    for i in range(len(p_list)):
                        dynamics = Lindblad(
                            self.tspan,
                            self.rho0,
                            self.H_list[i],
                            self.dH_list[i],
                            decay=self.decay,
                            Hc=self.Hc,
                            ctrl=self.ctrl,
                        )
                        rho_tp, drho_tp = dynamics.ode()
                        rho, drho = rho_tp[-1], drho_tp[-1]
                        F_tp = QFIM(rho, drho)
                        if np.linalg.det(F_tp) < self.eps:
                            F.append(self.eps)
                        else:
                            F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
                idx = np.argmax(F)
                H_res, dH_res = self.H_list[idx], self.dH_list[idx]
            m = MeasurementOpt(mtype="projection", minput=[], method="DE")
            m.dynamics(
                self.tspan,
                self.rho0,
                H_res,
                dH_res,
                Hc=self.Hc,
                ctrl=self.ctrl,
                decay=self.decay,
                dyn_method=self.dyn_method,
            )
            m.CFIM(W=W)
        elif self.dynamic_type == "Kraus":
            if self.para_num == 1:
                F = []
                for hi in range(len(self.K)):
                    rho_tp = sum(
                        [np.dot(Ki, np.dot(self.rho0, Ki.conj().T)) for Ki in self.K[hi]]
                    )
                    drho_tp = sum(
                        [
                            np.dot(dKi, np.dot(self.rho0, Ki.conj().T))
                            + np.dot(Ki, np.dot(self.rho0, dKi.conj().T))
                            for (Ki, dKi) in zip(self.K[hi], self.dK[hi])
                        ]
                    )
                    F_tp = QFIM(rho_tp, drho_tp)
                    F.append(F_tp)

                idx = np.argmax(F)
                K_res, dK_res = self.K[idx], self.dK[idx]
            else:
                p_shape = np.shape(self.p)

                p_ext = extract_ele(self.p, self.para_num)
                K_ext = extract_ele(self.K, self.para_num)
                dK_ext = extract_ele(self.dK, self.para_num)

                p_list, K_list, dK_list = [], [], []
                for K_ele, dK_ele in zip(K_ext, dK_ext):
                    p_list.append(p_ele)
                    K_list.append(K_ele)
                    dK_list.append(dK_ele)
                F = []
                for hi in range(len(p_list)):
                    rho_tp = sum(
                        [np.dot(Ki, np.dot(self.rho0, Ki.conj().T)) for Ki in K_list[hi]]
                    )
                    dK_reshape = [
                        [dK_list[hi][i][j] for i in range(self.k_num)]
                        for j in range(self.para_num)
                    ]
                    drho_tp = [
                        sum(
                            [
                                np.dot(dKi, np.dot(self.rho0, Ki.conj().T))
                                + np.dot(Ki, np.dot(self.rho0, dKi.conj().T))
                                for (Ki, dKi) in zip(K_list[hi], dKj)
                            ]
                        )
                        for dKj in dK_reshape
                    ]
                    F_tp = QFIM(rho_tp, drho_tp)
                    if np.linalg.det(F_tp) < self.eps:
                        F.append(self.eps)
                    else:
                        F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
                F = np.array(F).reshape(p_shape)
                idx = np.where(np.array(F) == np.max(np.array(F)))
                K_res, dK_res = self.K_list[idx], self.dK_list[idx]
            m = MeasurementOpt(mtype="projection", minput=[], method="DE")
            m.Kraus(self.rho0, K_res, dK_res, decay=self.decay)
            m.CFIM(W=W)
        else:
            raise ValueError(
                "{!r} is not a valid value for type of dynamics, supported values are 'dynamics' and 'Kraus'.".format(
                    self.dynamic_type
                )
            )
    
def adaptive_dynamics(x, p, M, tspan, rho0, H, dH, decay, Hc, ctrl, W, max_episode, eps, savefile, method, dyn_method="expm"):

    para_num = len(x)
    dim = np.shape(rho0)[0]
    if para_num == 1:
        #### singleparameter senario ####
        p_num = len(p)

        F = []
        rho_all = []
        if dyn_method == "expm":
            for hi in range(p_num):
                dynamics = Lindblad(tspan, rho0, H[hi], dH[hi], decay=decay, Hc=Hc, ctrl=ctrl)
                rho_tp, drho_tp = dynamics.expm()
                F_tp = CFIM(rho_tp[-1], drho_tp[-1], M)
                F.append(F_tp)
                rho_all.append(rho_tp[-1])
        elif dyn_method == "ode":
            for hi in range(p_num):
                dynamics = Lindblad(tspan, rho0, H[hi], dH[hi], decay=decay, Hc=Hc, ctrl=ctrl)
                rho_tp, drho_tp = dynamics.ode()
                F_tp = CFIM(rho_tp[-1], drho_tp[-1], M)
                F.append(F_tp)
                rho_all.append(rho_tp[-1])
        
        u = 0.0
        if method == "FOP":
            idx = np.argmax(F)
            x_opt = x[0][idx]
            print("The optimal parameter is %f" % x_opt)
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    savefile_true([np.array(p)], x_out, res_exp)
        elif method == "MI":
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    savefile_true([np.array(p)], x_out, res_exp)
    else:
        #### miltiparameter senario ####
        p_shape = np.shape(p)
        x_list = []
        for x_tp in product(*x):
            x_list.append([x_tp[i] for i in range(para_num)])

        p_ext = extract_ele(p, para_num)
        H_ext = extract_ele(H, para_num)
        dH_ext = extract_ele(dH, para_num)

        p_list, H_list, dH_list = [], [], []
        for p_ele, H_ele, dH_ele in zip(p_ext, H_ext, dH_ext):
            p_list.append(p_ele)
            H_list.append(H_ele)
            dH_list.append(dH_ele)

        p_num = len(p_list)
        F = []
        rho_all = []
        if dyn_method == "expm":
            for hi in range(p_num):
                dynamics = Lindblad(tspan, rho0, H_list[hi], dH_list[hi], decay=decay, Hc=Hc, ctrl=ctrl)
                rho_tp, drho_tp = dynamics.expm()
                F_tp = CFIM(rho_tp[-1], drho_tp[-1], M)
                if np.linalg.det(F_tp) < eps:
                    F.append(eps)
                else:
                    F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
                rho_all.append(rho_tp[-1])
        elif dyn_method == "ode":
            for hi in range(p_num):
                dynamics = Lindblad(tspan, rho0, H_list[hi], dH_list[hi], decay=decay, Hc=Hc, ctrl=ctrl)
                rho_tp, drho_tp = dynamics.ode()
                F_tp = CFIM(rho_tp[-1], drho_tp[-1], M)
                if np.linalg.det(F_tp) < eps:
                    F.append(eps)
                else:
                    F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
                rho_all.append(rho_tp[-1])

        u = [0.0 for i in range(para_num)]
        if method == "FOP":
            F = np.array(F).reshape(p_shape)
            idx = np.unravel_index(F.argmax(), F.shape)
            x_opt = [x[i][idx[i]] for i in range(para_num)]
            print("The optimal parameter are %s" % (x_opt))
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, x_opt, ei, p_shape)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, x_opt, ei, p_shape)
                    savefile_true(np.array(p), x_out, res_exp)
        elif method == "MI":
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, ei, p_shape)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, ei, p_shape)
                    savefile_true(np.array(p), x_out, res_exp)

def adaptive_Kraus(x, p, M, rho0, K, dK, W, max_episode, eps, savefile, method):
    para_num = len(x)
    dim = np.shape(rho0)[0]
    if para_num == 1:
        #### singleparameter senario ####
        p_num = len(p)
        F = []
        rho_all = []
        for hi in range(p_num):
            rho_tp = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K[hi]])
            drho_tp = [sum([(np.dot(dKi, np.dot(rho0, Ki.conj().T)) + np.dot(Ki, np.dot(rho0, dKi.conj().T))) for (Ki, dKi) in zip(K[hi], dKj)]) for dKj in dK[hi]]
            F_tp = CFIM(rho_tp, drho_tp, M)
            F.append(F_tp)
            rho_all.append(rho_tp)

        u = 0.0
        if method == "FOP":
            idx = np.argmax(F)
            x_opt = x[0][idx]
            print("The optimal parameter is %s" % x_opt)
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    savefile_true([np.array(p)], x_out, res_exp)
        elif method == "MI":
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    savefile_true([np.array(p)], x_out, res_exp)
    else:
        #### miltiparameter senario ####
        p_shape = np.shape(p)
        x_list = []
        for x_tp in product(*x):
            x_list.append([x_tp[i] for i in range(para_num)])

        p_ext = extract_ele(p, para_num)
        K_ext = extract_ele(K, para_num)
        dK_ext = extract_ele(dK, para_num)

        p_list, K_list, dK_list = [], [], []
        for p_ele, K_ele, dK_ele in zip(p_ext, K_ext, dK_ext):
            p_list.append(p_ele)
            K_list.append(K_ele)
            dK_list.append(dK_ele)
        k_num = len(K_list[0])
        p_num = len(p_list)
        F = []
        rho_all = []
        for hi in range(p_num):
            rho_tp = sum([np.dot(Ki, np.dot(rho0, Ki.conj().T)) for Ki in K_list[hi]])
            dK_reshape = [[dK_list[hi][i][j] for i in range(k_num)] for j in range(para_num)]
            drho_tp = [sum([np.dot(dKi, np.dot(rho0, Ki.conj().T))+ np.dot(Ki, np.dot(rho0, dKi.conj().T)) for (Ki, dKi) in zip(K_list[hi], dKj)]) for dKj in dK_reshape]
            F_tp = CFIM(rho_tp, drho_tp, M)
            if np.linalg.det(F_tp) < eps:
                F.append(eps)
            else:
                F.append(1.0 / np.trace(np.dot(W, np.linalg.inv(F_tp))))
            rho_all.append(rho_tp)

        if method == "FOP":
            F = np.array(F).reshape(p_shape)
            idx = np.unravel_index(F.argmax(), F.shape)
            x_opt = [x[i][idx[i]] for i in range(para_num)]
            print("The optimal parameter is %s" % (x_opt))
            u = [0.0 for i in range(para_num)]
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, x_opt, ei, p_shape)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, x_opt, ei, p_shape)
                    savefile_true(np.array(p), x_out, res_exp)
        elif method == "MI":
            if savefile == False:
                y, xout = [], []
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, ei, p_shape)
                    xout.append(x_out)
                    y.append(res_exp)
                savefile_false(p, xout, y)
            else:
                for ei in range(max_episode):
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, ei, p_shape)
                    savefile_true(np.array(p), x_out, res_exp)

def iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei):
    rho = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
    for hj in range(p_num):
        x_idx = np.argmin(np.abs(x[0] - (x[0][hj] + u)))
        rho[hj] = rho_all[x_idx]
    print("The tunable parameter is %f" % u)
    res_exp = input("Please enter the experimental result: ")
    res_exp = int(res_exp)
    pyx = np.zeros(p_num)
    for xi in range(p_num):
        pyx[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))

    arr = np.array([pyx[m] * p[m] for m in range(p_num)])
    py = simpson(arr, x[0])
    p_update = pyx * p / py
    
    for i in range(p_num):
        if x[0][0] < (x[0][i] + u) < x[0][-1]:
            p[i] = p_update[i]
        else:
            p[i] = 0.0

    p_idx = np.argmax(p)
    x_out = x[0][p_idx]
    print("The estimator is %s (%d episodes)" % (x_out, ei))
    u = x_opt - x_out

    if (ei + 1) % 50 == 0:
        if (x_out + u) > x[0][-1] and (x_out + u) < x[0][0]:
            raise ValueError("please increase the regime of the parameters.")
    return p, x_out, res_exp, u

def iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, x_opt, ei, p_shape):
    rho = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
    for hj in range(p_num):
        idx_list = [np.argmin(np.abs(x[i] - (x_list[hj][i] + u[i]))) for i in range(para_num)]
        x_idx = int(sum([idx_list[i] * np.prod(np.array(p_shape[(i + 1) :])) for i in range(para_num)]))
        rho[hj] = rho_all[x_idx]
    print("The tunable parameter are %s" % (u))
    res_exp = input("Please enter the experimental result: ")
    res_exp = int(res_exp)
    pyx_list = np.zeros(p_num)
    for xi in range(p_num):
        pyx_list[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
    pyx = pyx_list.reshape(p_shape)
    arr = p * pyx
    for si in reversed(range(para_num)):
        arr = simpson(arr, x[si])
    py = arr
    p_update = p * pyx / py
    
    p_lis = np.zeros(p_num)
    p_ext = extract_ele(p_update, para_num)
    p_up_lis = []
    for p_ele in p_ext:
        p_up_lis.append(p_ele)
    for i in range(p_num):
        res = [x_list[0][ri] < (x_list[i][ri] + u[ri]) < x_list[-1][ri] for ri in range(para_num)]
        if all(res):
            p_lis[i] =  p_up_lis[i]

    p = p_lis.reshape(p_shape)
    
    p_idx = np.unravel_index(p.argmax(), p.shape)
    x_out = [x[i][p_idx[i]] for i in range(para_num)]

    print("The estimator is %s (%d episodes)" % (x_out, ei))
    u = np.array(x_opt) - np.array(x_out)

    if (ei + 1) % 50 == 0:
        for un in range(para_num):
            if (x_out[un] + u[un]) > x[un][-1] and (x_out[un] + u[un]) < x[un][0]:
                raise ValueError("please increase the regime of the parameters.")
    return p, x_out, res_exp, u

def iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei):
    rho = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
    for hj in range(p_num):
        x_idx = np.argmin(np.abs(x[0] - (x[0][hj] + u)))
        rho[hj] = rho_all[x_idx]
    print("The tunable parameter is %f" % u)

    res_exp = input("Please enter the experimental result: ")
    res_exp = int(res_exp)
    pyx = np.zeros(p_num)
    for xi in range(p_num):
        pyx[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))

    arr = np.array([pyx[m] * p[m] for m in range(p_num)])
    py = simpson(arr, x[0])
    p_update = pyx * p / py
    
    for i in range(p_num):
        if x[0][0] < (x[0][i] + u) < x[0][-1]:
            p[i] = p_update[i]
        else:
            p[i] = 0.0

    p_idx = np.argmax(p)
    x_out = x[0][p_idx]
    print("The estimator is %s (%d episodes)" % (x_out, ei))

    MI = np.zeros(p_num)
    for ui in range(p_num):
        rho_u = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
        for hk in range(p_num):
            x_idx = np.argmin(np.abs(x[0] - (x[0][hk] + x[0][ui])))
            rho_u[hk] = rho_all[x_idx]
        value_tp = np.zeros(p_num)
        for mi in range(len(M)):
            pyx_tp = np.array([np.real(np.trace(np.dot(rho_u[xi], M[mi]))) for xi in range(p_num)])
            mean_tp = simpson(np.array([pyx_tp[i] * p[i] for i in range(p_num)]), x[0])
            value_tp += pyx_tp*np.log2(pyx_tp/mean_tp)
        # arr = np.array([value_tp[i] * p[i] for i in range(p_num)])
        arr = np.zeros(p_num)
        for i in range(p_num):
            if x[0][0] < (x[0][i] + x[0][ui]) < x[0][-1]:
                arr[i] = value_tp[i] * p[i]
        MI[ui] = simpson(arr, x[0])
    u = x[0][np.argmax(MI)]

    if (ei + 1) % 50 == 0:
        if (x_out + u) > x[0][-1] and (x_out + u) < x[0][0]:
            raise ValueError("please increase the regime of the parameters.")
    return p, x_out, res_exp, u

def iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, dim, ei, p_shape):
    rho = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
    for hj in range(p_num):
        idx_list = [np.argmin(np.abs(x[i] - (x_list[hj][i] + u[i]))) for i in range(para_num)]
        x_idx = int(sum([idx_list[i] * np.prod(np.array(p_shape[(i + 1) :])) for i in range(para_num)]))
        rho[hj] = rho_all[x_idx]
    print("The tunable parameter are %s" % (u))
    res_exp = input("Please enter the experimental result: ")
    res_exp = int(res_exp)
    pyx_list = np.zeros(p_num)
    for xi in range(p_num):
        pyx_list[xi] = np.real(np.trace(np.dot(rho[xi], M[res_exp])))
    pyx = pyx_list.reshape(p_shape)
    arr = p * pyx
    for si in reversed(range(para_num)):
        arr = simpson(arr, x[si])
    py = arr
    p_update = p * pyx / py
    
    p_lis = np.zeros(p_num)
    p_ext = extract_ele(p_update, para_num)
    p_up_lis = []
    for p_ele in p_ext:
        p_up_lis.append(p_ele)
    for i in range(p_num):
        res = [x_list[0][ri] < (x_list[i][ri] + u[ri]) < x_list[-1][ri] for ri in range(para_num)]
        if all(res):
            p_lis[i] =  p_up_lis[i]

    p = p_lis.reshape(p_shape)

    p_idx = np.unravel_index(p.argmax(), p.shape)
    x_out = [x[i][p_idx[i]] for i in range(para_num)]
    print("The estimator is %s (%d episodes)" % (x_out, ei))

    MI = np.zeros(p_num)
    for ui in range(p_num):
        rho_u = [np.zeros((dim, dim), dtype=np.complex128) for i in range(p_num)]
        for hj in range(p_num):
            idx_list = [np.argmin(np.abs(x[i] - (x_list[hj][i] + x_list[ui][i]))) for i in range(para_num)]
            x_idx = int(sum([idx_list[i] * np.prod(np.array(p_shape[(i + 1) :])) for i in range(para_num)]))
            rho_u[hj] = rho_all[x_idx]
        value_tp = np.zeros(p_shape)
        for mi in range(len(M)):
            pyx_list_tp = np.array([np.real(np.trace(np.dot(rho_u[xi], M[mi]))) for xi in range(p_num)])
            pyx_tp = pyx_list_tp.reshape(p_shape)
            mean_tp = p * pyx_tp
            for si in reversed(range(para_num)):
                mean_tp = simpson(mean_tp, x[si])
            value_tp += pyx_tp*np.log2(pyx_tp/mean_tp) 

        # value_int = p * value_tp
        # for sj in reversed(range(para_num)):
        #    value_int = simpson(value_int, x[sj])  

        arr = np.zeros(p_num)
        p_ext = extract_ele(p, para_num)
        value_ext = extract_ele(value_tp, para_num)
        p_lis, value_lis = [], []
        for p_ele, value_ele in zip(p_ext, value_ext):
            p_lis.append(p_ele)
            value_lis.append(value_ele)
        for hj in range(p_num):
            res = [x_list[0][ri] < (x_list[hj][ri] + x_list[ui][ri]) < x_list[-1][ri] for ri in range(para_num)]
            if all(res):
                arr[hj] =  p_lis[hj] * value_lis[hj]
        value_int = arr.reshape(p_shape)
        for sj in reversed(range(para_num)):
           value_int = simpson(value_int, x[sj])     

        MI[ui] = value_int
    p_idx = np.unravel_index(MI.argmax(), p.shape)
    u = [x[i][p_idx[i]] for i in range(para_num)]

    if (ei + 1) % 50 == 0:
        for un in range(para_num):
            if (x_out[un] + u[un]) > x[un][-1] and (x_out[un] + u[un]) < x[un][0]:
                raise ValueError("please increase the regime of the parameters.")
    return p, x_out, res_exp, u

def savefile_true(p, xout, y):
    fp = open('pout.csv','a')
    fp.write('\n')
    np.savetxt(fp, p)
    fp.close()

    fx = open('xout.csv','a')
    fx.write('\n')
    np.savetxt(fx, [xout])
    fx.close()

    fy = open('y.csv','a')
    fy.write('\n')
    np.savetxt(fy, [y])
    fy.close()

def savefile_false(p, xout, y):
    fp = open('pout.csv','a')
    fp.write('\n')
    np.savetxt(fp, np.array(p))
    fp.close()

    fx = open('xout.csv','a')
    fx.write('\n')
    np.savetxt(fx, np.array(xout))
    fx.close()

    fy = open('y.csv','a')
    fy.write('\n')
    np.savetxt(fy, np.array(y))
    fy.close()
