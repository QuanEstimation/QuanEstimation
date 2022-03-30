import numpy as np
from scipy.interpolate import interp1d
import os
import math
import warnings
from julia import Main
import quanestimation.StateOpt as stateoptimize
from quanestimation.Common.common import SIC


class StateSystem:
    def __init__(self, savefile, psi0, seed, load, eps):

        """
        ----------
        Inputs
        ----------
        savefile:
            --description: True: save the states and the value of target function for each episode .
                           False: save the states and the value of target function for the last episode.
            --type: bool

        psi0:
            --description: initial guess of states (kets).
            --type: array

        seed:
            --description: random seed.
            --type: int

        eps:
            --description: machine eps.
            --type: float
        """
        self.savefile = savefile
        self.psi0 = psi0
        self.psi = psi0
        self.eps = eps
        self.seed = seed

        if load == True:
            if os.path.exists("states.csv"):
                self.psi0 = np.genfromtxt("states.csv", dtype=np.complex128)

    def load_save(self):
        if os.path.exists("states.csv"):
            file_load = open("states.csv", "r")
            file_load = "".join([i for i in file_load]).replace("im", "j")
            file_load = "".join([i for i in file_load]).replace(" ", "")
            file_save = open("states.csv", "w")
            file_save.writelines(file_load)
            file_save.close()
        else:
            pass

    def dynamics(self, tspan, H0, dH, Hc=[], ctrl=[], decay=[]):

        """
        ----------
        Inputs
        ----------
        tspan:
           --description: time series.
           --type: array

        H0:
           --description: free Hamiltonian.
           --type: matrix (a list of matrix)

        dH:
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)

        Hc:
           --description: control Hamiltonian.
           --type: list (of matrix)

        ctrl:
            --description: control coefficients.
            --type: list (of vector)

        decay:
           --description: decay operators and the corresponding decay rates.
                          decay[0][0] represent the first decay operator and
                          decay[0][1] represent the corresponding decay rate.
           --type: list

        ctrl_bound:
           --description: lower and upper bounds of the control coefficients.
                          ctrl_bound[0] represent the lower bound of the control coefficients and
                          ctrl_bound[1] represent the upper bound of the control coefficients.
           --type: list
        """
        self.tspan = tspan

        if Hc == [] or ctrl == []:
            if type(H0) == np.ndarray:
                self.freeHamiltonian = np.array(H0, dtype=np.complex128)
                self.dim = len(self.freeHamiltonian)
            else:
                self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]
                self.dim = len(self.freeHamiltonian[0])
        else:
            ctrl_num = len(ctrl)
            Hc_num = len(Hc)
            if Hc_num < ctrl_num:
                raise TypeError(
                    "There are %d control Hamiltonians but %d coefficients sequences: \
                                 too many coefficients sequences."
                    % (Hc_num, ctrl_num)
                )
            elif Hc_num > ctrl_num:
                warnings.warn(
                    "Not enough coefficients sequences: there are %d control Hamiltonians \
                               but %d coefficients sequences. The rest of the control sequences are\
                               set to be 0."
                    % (Hc_num, ctrl_num),
                    DeprecationWarning,
                )
                for i in range(Hc_num - ctrl_num):
                    ctrl = np.concatenate((ctrl, np.zeros(len(ctrl[0]))))
            else: pass

            if len(ctrl[0]) == 1:
                if type(H0) == np.ndarray:
                    H0 = np.array(H0, dtype=np.complex128)
                    Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                    Htot = H0 + sum([Hc[i] * ctrl[i][0] for i in range(ctrl_num)])
                    self.freeHamiltonian = np.array(Htot, dtype=np.complex128)
                    self.dim = len(self.freeHamiltonian)
                else:
                    H0 = [np.array(x, dtype=np.complex128) for x in H0]
                    Htot = []
                    for i in range(len(H0)):
                        Htot.append(
                            H0[i] + sum([Hc[i] * ctrl[i][0] for i in range(ctrl_num)])
                        )
                    self.freeHamiltonian = [
                        np.array(x, dtype=np.complex128) for x in Htot
                    ]
                    self.dim = len(self.freeHamiltonian[0])
            else:
                if type(H0) != np.ndarray:
                    #### linear interpolation  ####
                    f = interp1d(self.tspan, H0, axis=0)
                else: pass
                number = math.ceil((len(self.tspan) - 1) / len(ctrl[0]))
                if len(self.tspan) - 1 % len(ctrl[0]) != 0:
                    tnum = number * len(ctrl[0])
                    self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum + 1)
                    if type(H0) != np.ndarray:
                        H0_inter = f(self.tspan)
                        H0 = [np.array(x, dtype=np.complex128) for x in H0_inter]
                    else: pass
                else: pass

                if type(H0) == np.ndarray:
                    H0 = np.array(H0, dtype=np.complex128)
                    Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                    ctrl = [np.array(ctrl[i]).repeat(number) for i in range(len(Hc))]
                    Htot = []
                    for i in range(len(ctrl[0])):
                        S_ctrl = sum([Hc[j] * ctrl[j][i] for j in range(len(ctrl))])
                        Htot.append(H0 + S_ctrl)
                    self.freeHamiltonian = [
                        np.array(x, dtype=np.complex128) for x in Htot
                    ]
                    self.dim = len(self.freeHamiltonian)
                else:
                    H0 = [np.array(x, dtype=np.complex128) for x in H0]
                    Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                    ctrl = [np.array(ctrl[i]).repeat(number) for i in range(len(Hc))]
                    Htot = []
                    for i in range(len(ctrl[0])):
                        S_ctrl = sum([Hc[j] * ctrl[j][i] for j in range(len(ctrl))])
                        Htot.append(H0[i] + S_ctrl)
                    self.freeHamiltonian = [
                        np.array(x, dtype=np.complex128) for x in Htot
                    ]
                    self.dim = len(self.freeHamiltonian[0])
        if self.psi0 == []:
            np.random.seed(self.seed)
            r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
            r = r_ini / np.linalg.norm(r_ini)
            phi = 2 * np.pi * np.random.random(self.dim)
            psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)

        if self.psi == []:
            self.psi = [self.psi0]
        else:
            self.psi = [np.array(psi, dtype=np.complex128) for psi in self.psi]

        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((len(self.psi0), len(self.psi0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]

        if decay == []:
            decay_opt = [np.zeros((len(self.psi0), len(self.psi0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        self.opt = Main.QuanEstimation.StateOpt(self.psi0)
        if any(self.gamma):
            self.dynamic = Main.QuanEstimation.Lindblad(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                self.psi0,
                self.tspan,
                self.decay_opt,
                self.gamma,
            )
        else:
            self.dynamic = Main.QuanEstimation.Lindblad(
                self.freeHamiltonian,
                self.Hamiltonian_derivative,
                self.psi0,
                self.tspan,
            )
        self.output = Main.QuanEstimation.Output(self.opt, self.savefile)

        self.dynamics_type = "dynamics"
        if len(self.Hamiltonian_derivative) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

    def kraus(self, K, dK):

        k_num = len(K)
        para_num = len(dK[0])
        self.K = [np.array(x, dtype=np.complex128) for x in K]
        self.dK = [
            [np.array(dK[i][j], dtype=np.complex128) for i in range(k_num)]
            for j in range(para_num)
        ]

        self.dim = len(self.K[0])

        if self.psi0 == []:
            np.random.seed(self.seed)
            r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
            r = r_ini / np.linalg.norm(r_ini)
            phi = 2 * np.pi * np.random.random(self.dim)
            psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)
        else:
            self.psi0 = np.array(self.psi0[0], dtype=np.complex128)

        if self.psi == []:
            self.psi = [self.psi0]
        else:
            self.psi = [np.array(psi, dtype=np.complex128) for psi in self.psi]

        self.opt = Main.QuanEstimation.StateOpt(self.psi0)
        self.dynamic = Main.QuanEstimation.Kraus(self.K, self.dK, self.psi0)
        self.output = Main.QuanEstimation.Output(self.opt, self.savefile)

        self.dynamics_type = "kraus"
        if len(self.dK) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

    def QFIM(self, W=[], LDtype="SLD"):
        """
        Description: use autodifferential algorithm to search the optimal initial state that maximize the
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """

        if LDtype != "SLD" and LDtype != "RLD" and LDtype != "LLD":
            raise ValueError(
                "{!r} is not a valid value for LDtype, supported values are 'SLD', 'RLD' and 'LLD'.".format(
                    LDtype
                )
            )

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W
        else:
            pass

        self.obj = Main.QuanEstimation.QFIM_Obj(
            self.W, self.eps, self.para_type, LDtype
        )
        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system)

        self.load_save()

    def CFIM(self, M=[], W=[]):
        """
        Description: use autodifferential algorithm to search the optimal initial state that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        M:
            --description: a set of POVM.
            --type: list of matrix

        W:
            --description: weight matrix.
            --type: matrix
        """
        if M == []:
            M = SIC(len(self.psi0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W

        self.obj = Main.QuanEstimation.CFIM_Obj(M, self.W, self.eps, self.para_type)
        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system)

        self.load_save()

    def HCRB(self, W=[]):
        if self.dynamics_type == "dynamics":
            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W
            if len(self.Hamiltonian_derivative) == 1:
                raise ValueError(
                    "In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the target function for control optimization",
                )
            else: pass

        elif self.dynamics_type == "kraus":
            if W == []:
                W = np.eye(len(self.dK))
            self.W = W
            if len(self.dK) == 1:
                raise ValueError(
                    "In single parameter scenario, HCRB is equivalent to QFI. Please choose QFIM as the target function for control optimization",
                )
            else: pass
        else:
            raise ValueError(
                "Supported type of dynamics are Lindblad and Kraus."
                )

        self.obj = Main.QuanEstimation.HCRB_Obj(self.W, self.eps, self.para_type)
        system = Main.QuanEstimation.QuanEstSystem(
                self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system)

        self.load_save()


def StateOpt(savefile=False, method="AD", **kwargs):

    if method == "AD":
        return stateoptimize.AD_Sopt(savefile=savefile, **kwargs)
    elif method == "PSO":
        return stateoptimize.PSO_Sopt(savefile=savefile, **kwargs)
    elif method == "DE":
        return stateoptimize.DE_Sopt(savefile=savefile, **kwargs)
    elif method == "DDPG":
        return stateoptimize.DDPG_Sopt(savefile=savefile, **kwargs)
    elif method == "NM":
        return stateoptimize.NM_Sopt(savefile=savefile, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values are 'AD', 'PSO', 'DE', 'NM', 'DDPG'.".format(
                method
            )
        )


def csv2npy_states(states, num=1):
    S_save = []
    N = int(len(states) / num)
    for si in range(N):
        S_tp = states[si * num : (si + 1) * num]
        S_save.append(S_tp)
    np.save("states", S_save)
