import numpy as np
import os
import math
import warnings
import quanestimation.StateOpt as stateoptimize


class StateSystem:
    def __init__(self, psi0, seed, load, eps):

        """
        ----------
        Inputs
        ----------
        W:
            --description: weight matrix.
            --type: matrix

        eps:
            --description: machine eps.
            --type: float
        """

        self.psi0 = psi0
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

        psi0:
            --description: initial guess of states (kets).
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
                                too many coefficients sequences"
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
            else:
                pass

            if len(ctrl[0]) == 1:
                H0 = np.array(H0, dtype=np.complex128)
                Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                Htot = H0 + sum([Hc[i] * ctrl[i][0] for i in range(ctrl_num)])
                self.freeHamiltonian = np.array(Htot, dtype=np.complex128)
                self.dim = len(self.freeHamiltonian)
            else:
                number = math.ceil((len(self.tspan) - 1) / len(ctrl[0]))
                if len(self.tspan) - 1 % len(ctrl[0]) != 0:
                    tnum = number * len(ctrl[0])
                    self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum + 1)
                else:
                    pass

                H0 = np.array(H0, dtype=np.complex128)
                Hc = [np.array(x, dtype=np.complex128) for x in Hc]
                Htot = []
                for i in range(len(ctrl[0])):
                    S_ctrl = sum([Hc[j] * ctrl[j][i] for j in range(len(ctrl))])
                    Htot.append(H0 + S_ctrl)
                self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in Htot]
                self.dim = len(self.freeHamiltonian[0])

        if self.psi0 == []:
            np.random.seed(self.seed)
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)

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

        self.dynamics_type = "dynamics"

    def kraus(self, K, dK):
        # TODO: initialize K, dK

        self.K = K
        self.dK = dK
        self.dim = len(self.K)

        if self.psi0 == []:
            np.random.seed(self.seed)
            for i in range(self.dim):
                r_ini = 2 * np.random.random(self.dim) - np.ones(self.dim)
                r = r_ini / np.linalg.norm(r_ini)
                phi = 2 * np.pi * np.random.random(self.dim)
                psi0 = [r[i] * np.exp(1.0j * phi[i]) for i in range(self.dim)]
            self.psi0 = np.array(psi0)

        self.dynamics_type = "kraus"


def StateOpt(method="AD", **kwargs):

    if method == "AD":
        return stateoptimize.AD_Sopt(**kwargs)
    elif method == "PSO":
        return stateoptimize.PSO_Sopt(**kwargs)
    elif method == "DE":
        return stateoptimize.DE_Sopt(**kwargs)
    elif method == "DDPG":
        return stateoptimize.DDPG_Sopt(**kwargs)
    elif method == "NM":
        return stateoptimize.NM_Sopt(**kwargs)
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
