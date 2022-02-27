import numpy as np
import os
import math
import warnings
import quanestimation.MeasurementOpt as Measure
from quanestimation.Common.common import gramschmidt, sic_povm


class MeasurementSystem:
    def __init__(self, mtype, minput, measurement0, seed, load, eps):

        """
        ----------
        Inputs
        ----------
        tspan:
           --description: time series.
           --type: array

        rho0:
           --description: initial state (density matrix).
           --type: matrix

        H0:
           --description: free Hamiltonian.
           --type: matrix or a list of matrix

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

        measurement0:
           --description: a set of POVMs.
           --type: list (of vector)

        W:
            --description: weight matrix.
            --type: matrix

        eps:
            --description: calculation eps.
            --type: float

        notes: the Weyl-Heisenberg covariant SIC-POVM fiducial state of dimension $d$
               are download from http://www.physics.umb.edu/Research/QBism/solutions.html.

        """
        self.mtype = mtype
        self.minput = minput
        self.eps = eps
        self.seed = seed
        self.load = load

        if self.mtype == "projection":
            if measurement0 == []:
                np.random.seed(self.seed)
                M = [[] for i in range(len(self.rho0))]
                for i in range(len(self.rho0)):
                    r_ini = 2 * np.random.random(len(self.rho0)) - np.ones(
                        len(self.rho0)
                    )
                    r = r_ini / np.linalg.norm(r_ini)
                    phi = 2 * np.pi * np.random.random(len(self.rho0))
                    M[i] = [r[i] * np.exp(1.0j * phi[i]) for i in range(len(self.rho0))]
                self.M = gramschmidt(np.array(M))
            elif len(measurement0) >= 1:
                self.M = [measurement0[0][i] for i in range(len(self.rho0))]

        elif self.mtype == "input":
            if minput[0] == "LC":
                ## optimize the combination of a set of SIC-POVM
                if minput[1] == []:
                    file_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "sic_fiducial_vectors/d%d.txt" % (len(self.rho0)),
                    )
                    data = np.loadtxt(file_path)
                    fiducial = data[:, 0] + data[:, 1] * 1.0j
                    fiducial = np.array(fiducial).reshape(len(fiducial), 1)
                    self.povm_basis = sic_povm(fiducial)
                    self.M_num = minput[2]
                else:
                    ## optimize the combination of a set of given POVMs
                    if type(minput[1]) != list:
                        raise TypeError("The given POVMs should be a list!")
                    else:
                        accu = len(str(int(1 / self.eps))) - 1
                        for i in range(len(minput[1])):
                            val, vec = np.linalg.eig(minput[1])
                            if np.all(val.round(accu) >= 0):
                                pass
                            else:
                                raise TypeError(
                                    "The given POVMs should be semidefinite!"
                                )
                        M = np.zeros(
                            (len(self.rho0), len(self.rho0)), dtype=np.complex128
                        )
                        for i in range(len(minput[1])):
                            M += minput[1][i]
                        if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                            pass
                        else:
                            raise TypeError(
                                "The sum of the given POVMs should be identity matrix!"
                            )
                        self.povm_basis = [
                            np.array(x, dtype=np.complex128) for x in minput[1]
                        ]
                        self.M_num = minput[2]
            elif minput[0] == "rotation":
                ## optimize the coefficients of the rotation matrix
                if type(minput[1]) != list:
                    raise TypeError("The given POVMs should be a list!")
                else:
                    if minput[1] == []:
                        raise TypeError("The initial POVM should not be empty!")
                    accu = len(str(int(1 / self.eps))) - 1
                    for i in range(len(minput[1])):
                        val, vec = np.linalg.eig(minput[1])
                        if np.all(val.round(accu) >= 0):
                            pass
                        else:
                            raise TypeError("The given POVMs should be semidefinite!")
                    M = np.zeros((len(self.rho0), len(self.rho0)), dtype=np.complex128)
                    for i in range(len(minput[1])):
                        M += minput[1][i]
                    if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                        pass
                    else:
                        raise TypeError(
                            "The sum of the given POVMs should be identity matrix!"
                        )
                    self.povm_basis = [
                        np.array(x, dtype=np.complex128) for x in minput[1]
                    ]
                    self.mtype = "rotation"
            else:
                raise ValueError(
                    "{!r} is not a valid value for the first input of minput, supported values are 'LC' and 'rotation'.".format(
                        self.minput[0]
                    )
                )
        else:
            raise ValueError(
                "{!r} is not a valid value for mtype, supported values are 'projection' and 'input'.".format(
                    self.mtype
                )
            )

    def load_save(self):
        if os.path.exists("measurements.csv"):
            file_load = open("measurements.csv", "r")
            file_load = "".join([i for i in file_load]).replace("im", "j")
            file_load = "".join([i for i in file_load]).replace(" ", "")
            file_save = open("measurements.csv", "w")
            file_save.writelines(file_load)
            file_save.close()
        else:
            pass

    def dynamics(self, tspan, rho0, H0, dH, Hc=[], ctrl=[], decay=[]):

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
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if Hc == [] or ctrl == []:
            if type(H0) == np.ndarray:
                self.freeHamiltonian = np.array(H0, dtype=np.complex128)
            else:
                self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]
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

        if type(dH) != list:
            raise TypeError("The derivative of Hamiltonian should be a list!")

        if dH == []:
            dH = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.Hamiltonian_derivative = [np.array(x, dtype=np.complex128) for x in dH]

        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        self.dynamics_type = "dynamics"

    def kraus(rho0, K, dK):
        self.rho0 = np.array(rho0, dtype=np.complex128)
        self.K = K
        self.dK = dK

        self.dynamics_type = "kraus"


def MeasurementOpt(mtype="projection", minput=[], method="DE", **kwargs):

    if method == "AD":
        return Measure.AD_Mopt(mtype, minput, **kwargs)
    elif method == "PSO":
        return Measure.PSO_Mopt(mtype, minput, **kwargs)
    elif method == "DE":
        return Measure.DE_Mopt(mtype, minput, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values are 'AD', 'PSO' and 'DE'.".format(
                method
            )
        )


def csv2npy_measurements(M, num):
    n = int(np.sqrt(len(M[0])))
    N = int(len(M) / num)
    M_save = []
    for mi in range(N):
        M_tp = M[mi * num : (mi + 1) * num]
        M = [M_tp[i].reshape(n, n).T for i in range(num)]
        M_save.append(M)
    np.save("measurements", M_save)


def load_measurements(M, num, indx=-1):
    n = int(np.sqrt(len(M[0])))
    N = int(len(M) / num)
    M_save = []
    for mi in range(N):
        M_tp = M[mi * num : (mi + 1) * num]
        M = [M_tp[i].reshape(n, n).T for i in range(num)]
        M_save.append(M)
    return M_save[indx]
