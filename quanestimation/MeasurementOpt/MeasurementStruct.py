from re import S
import numpy as np
import os
import math
import warnings
import quanestimation.MeasurementOpt as Measure
from quanestimation.Common.common import gramschmidt, sic_povm


class MeasurementSystem:
    def __init__(self, mtype, minput, save_file, measurement0, seed, load, eps):

        """
         ----------
         Inputs
         ----------
        save_file:
             --description: True: save the measurements and the value of the target function
                                  for each episode.
                            False: save the measurements and all the value of the target
                                   function for the last episode.
             --type: bool

         measurement0:
            --description: a set of POVMs.
            --type: list (of vector)

         eps:
             --description: calculation eps.
             --type: float

         notes: the Weyl-Heisenberg covariant SIC-POVM fiducial state of dimension $d$
                are download from http://www.physics.umb.edu/Research/QBism/solutions.html.

        """
        self.mtype = mtype
        self.minput = minput
        self.save_file = save_file
        self.eps = eps
        self.seed = seed
        self.load = load
        self.measurement0 = measurement0

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

        if self.mtype == "projection":
            if self.measurement0 == []:
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
            elif len(self.measurement0) >= 1:
                self.M = [self.measurement0[0][i] for i in range(len(self.rho0))]

            self.opt = Main.QuanEstimation.Mopt_Projection(self.measurement0)

        elif self.mtype == "input":
            if self.minput[0] == "LC":
                ## optimize the combination of a set of SIC-POVM
                if self.minput[1] == []:
                    file_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "sic_fiducial_vectors/d%d.txt" % (len(self.rho0)),
                    )
                    data = np.loadtxt(file_path)
                    fiducial = data[:, 0] + data[:, 1] * 1.0j
                    fiducial = np.array(fiducial).reshape(len(fiducial), 1)
                    self.povm_basis = sic_povm(fiducial)
                    self.M_num = self.minput[2]
                else:
                    ## optimize the combination of a set of given POVMs
                    if type(self.minput[1]) != list:
                        raise TypeError("The given POVMs should be a list!")
                    else:
                        accu = len(str(int(1 / self.eps))) - 1
                        for i in range(len(self.minput[1])):
                            val, vec = np.linalg.eig(self.minput[1])
                            if np.all(val.round(accu) >= 0):
                                pass
                            else:
                                raise TypeError(
                                    "The given POVMs should be semidefinite!"
                                )
                        M = np.zeros(
                            (len(self.rho0), len(self.rho0)), dtype=np.complex128
                        )
                        for i in range(len(self.minput[1])):
                            M += self.minput[1][i]
                        if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                            pass
                        else:
                            raise TypeError(
                                "The sum of the given POVMs should be identity matrix!"
                            )
                        self.povm_basis = [
                            np.array(x, dtype=np.complex128) for x in self.minput[1]
                        ]
                        self.M_num = self.minput[2]

                if self.measurement0 == []:
                    self.B = [
                        np.random.random(len(self.povm_basis))
                        for i in np.arange(self.M_num)
                    ]
                elif len(self.measurement0) >= 1:
                    self.B = [
                        self.measurement0[0][i] for i in range(len(self.povm_basis))
                    ]

                self.opt = Main.QuanEstimation.Mopt_LinearComb(
                    self.B, self.povm_basis, self.measurement0
                )

            elif self.minput[0] == "rotation":
                ## optimize the coefficients of the rotation matrix
                if type(self.minput[1]) != list:
                    raise TypeError("The given POVMs should be a list!")
                else:
                    if self.minput[1] == []:
                        raise TypeError("The initial POVM should not be empty!")
                    accu = len(str(int(1 / self.eps))) - 1
                    for i in range(len(self.minput[1])):
                        val, vec = np.linalg.eig(self.minput[1])
                        if np.all(val.round(accu) >= 0):
                            pass
                        else:
                            raise TypeError("The given POVMs should be semidefinite!")
                    M = np.zeros((len(self.rho0), len(self.rho0)), dtype=np.complex128)
                    for i in range(len(self.minput[1])):
                        M += self.minput[1][i]
                    if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                        pass
                    else:
                        raise TypeError(
                            "The sum of the given POVMs should be identity matrix!"
                        )
                    self.povm_basis = [
                        np.array(x, dtype=np.complex128) for x in self.minput[1]
                    ]
                    self.mtype = "rotation"

                if self.measurement0 == []:
                    self.s = np.random.random(len(self.rho0) ** 2)
                elif len(self.measurement0) >= 1:
                    self.s = [
                        self.measurement0[0][i]
                        for i in range(len(self.rho0) * len(self.rho0))
                    ]

                self.opt = Main.QuanEstimation.Mopt_Rotation(
                    self.s, self.povm_basis, self.measurement0
                )

            else:
                raise ValueError(
                    "{!r} is not a valid value for the first input of minput, \
                                 supported values are 'LC' and 'rotation'.".format(
                        self.minput[0]
                    )
                )
        else:
            raise ValueError(
                "{!r} is not a valid value for mtype, supported values are 'projection' \
                              and 'input'.".format(
                    self.mtype
                )
            )

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

        if len(dH) == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        if decay == []:
            decay_opt = [np.zeros((len(self.rho0), len(self.rho0)))]
            self.gamma = [0.0]
        else:
            decay_opt = [decay[i][0] for i in range(len(decay))]
            self.gamma = [decay[i][1] for i in range(len(decay))]
        self.decay_opt = [np.array(x, dtype=np.complex128) for x in decay_opt]

        self.dynamic = Main.QuanEstimation.Lindblad(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
        )
        self.output = Main.QuanEstimation.Output(self.opt, self.save_file)

        self.dynamics_type = "dynamics"

    def kraus(self, rho0, K, dK):
        k_num = len(K)
        para_num = len(dK[0])
        dK_tp = [
            [np.array(dK[i][j], dtype=np.complex128) for i in range(k_num)]
            for j in range(para_num)
        ]
        self.rho0 = np.array(rho0, dtype=np.complex128)
        self.K = [np.array(x, dtype=np.complex128) for x in K]
        self.dK = dK_tp

        if para_num == 1:
            self.para_type = "single_para"
        else:
            self.para_type = "multi_para"

        if self.mtype == "projection":
            if self.measurement0 == []:
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
            elif len(self.measurement0) >= 1:
                self.M = [self.measurement0[0][i] for i in range(len(self.rho0))]

            self.opt = Main.QuanEstimation.Mopt_Projection(self.measurement0)

        elif self.mtype == "input":
            if self.minput[0] == "LC":
                ## optimize the combination of a set of SIC-POVM
                if self.minput[1] == []:
                    file_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "sic_fiducial_vectors/d%d.txt" % (len(self.rho0)),
                    )
                    data = np.loadtxt(file_path)
                    fiducial = data[:, 0] + data[:, 1] * 1.0j
                    fiducial = np.array(fiducial).reshape(len(fiducial), 1)
                    self.povm_basis = sic_povm(fiducial)
                    self.M_num = self.minput[2]
                else:
                    ## optimize the combination of a set of given POVMs
                    if type(self.minput[1]) != list:
                        raise TypeError("The given POVMs should be a list!")
                    else:
                        accu = len(str(int(1 / self.eps))) - 1
                        for i in range(len(self.minput[1])):
                            val, vec = np.linalg.eig(self.minput[1])
                            if np.all(val.round(accu) >= 0):
                                pass
                            else:
                                raise TypeError(
                                    "The given POVMs should be semidefinite!"
                                )
                        M = np.zeros(
                            (len(self.rho0), len(self.rho0)), dtype=np.complex128
                        )
                        for i in range(len(self.minput[1])):
                            M += self.minput[1][i]
                        if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                            pass
                        else:
                            raise TypeError(
                                "The sum of the given POVMs should be identity matrix!"
                            )
                        self.povm_basis = [
                            np.array(x, dtype=np.complex128) for x in self.minput[1]
                        ]
                        self.M_num = self.minput[2]

                self.opt = Main.QuanEstimation.Mopt_LinearComb(
                    self.B, self.povm_basis, self.measurement0
                )

            elif self.minput[0] == "rotation":
                ## optimize the coefficients of the rotation matrix
                if type(self.minput[1]) != list:
                    raise TypeError("The given POVMs should be a list!")
                else:
                    if self.minput[1] == []:
                        raise TypeError("The initial POVM should not be empty!")
                    accu = len(str(int(1 / self.eps))) - 1
                    for i in range(len(self.minput[1])):
                        val, vec = np.linalg.eig(self.minput[1])
                        if np.all(val.round(accu) >= 0):
                            pass
                        else:
                            raise TypeError("The given POVMs should be semidefinite!")
                    M = np.zeros((len(self.rho0), len(self.rho0)), dtype=np.complex128)
                    for i in range(len(self.minput[1])):
                        M += self.minput[1][i]
                    if np.all(M.round(accu) - np.identity(len(self.rho0)) == 0):
                        pass
                    else:
                        raise TypeError(
                            "The sum of the given POVMs should be identity matrix!"
                        )
                    self.povm_basis = [
                        np.array(x, dtype=np.complex128) for x in self.minput[1]
                    ]
                    self.mtype = "rotation"
                    self.opt = Main.QuanEstimation.Mopt_Rotation(
                        self.s, self.povm_basis, self.measurement0
                    )
            else:
                raise ValueError(
                    "{!r} is not a valid value for the first input of minput, \
                                  supported values are 'LC' and 'rotation'.".format(
                        self.minput[0]
                    )
                )
        else:
            raise ValueError(
                "{!r} is not a valid value for mtype, supported values are \
                             'projection' and 'input'.".format(
                    self.mtype
                )
            )

        self.dynamic = Main.QuanEstimation.Kraus(
            self.K,
            self.dK,
            self.rho0,
        )
        self.output = Main.QuanEstimation.Output(self.opt, self.save_file)

        self.dynamics_type = "kraus"

    def CFIM(self, W=[]):
        """
        Description: use differential evolution algorithm to update the measurements that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """
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


def MeasurementOpt(
    mtype="projection", minput=[], save_file=False, method="DE", **kwargs
):

    if method == "AD":
        return Measure.AD_Mopt(mtype, minput, save_file=save_file, **kwargs)
    elif method == "PSO":
        return Measure.PSO_Mopt(mtype, minput, save_file=save_file, **kwargs)
    elif method == "DE":
        return Measure.DE_Mopt(mtype, minput, save_file=save_file, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values \
                          are 'AD', 'PSO' and 'DE'.".format(
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
