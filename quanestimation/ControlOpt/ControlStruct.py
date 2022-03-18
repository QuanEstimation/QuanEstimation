import numpy as np
import warnings
import math
import os
import quanestimation.ControlOpt as ctrl
from julia import Main
from quanestimation.Common.common import SIC


class ControlSystem:
    def __init__(self, savefile, ctrl0, load, eps):

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

        Hc:
           --description: control Hamiltonian.
           --type: list (of matrix)

        dH:
           --description: derivatives of Hamiltonian on all parameters to
                          be estimated. For example, dH[0] is the derivative
                          vector on the first parameter.
           --type: list (of matrix)

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

        savefile:
            --description: True: save the control coefficients and the value of the target function
                                 for each episode.
                           False: save the control coefficients and all the value of the target
                                  function for the last episode.
            --type: bool

        ctrl0:
            --description: initial control coefficients.
            --type: list (of vector)

        eps:
            --description: calculation eps.
            --type: float

        """
        self.savefile = savefile
        self.ctrl0 = ctrl0
        self.eps = eps
        self.load = load

    def dynamics(self, tspan, rho0, H0, dH, Hc, decay=[], ctrl_bound=[]):
        self.tspan = tspan
        self.rho0 = np.array(rho0, dtype=np.complex128)

        if type(H0) == np.ndarray:
            self.freeHamiltonian = np.array(H0, dtype=np.complex128)
        else:
            self.freeHamiltonian = [np.array(x, dtype=np.complex128) for x in H0]

        if Hc == []:
            Hc = [np.zeros((len(self.rho0), len(self.rho0)))]
        self.control_Hamiltonian = [np.array(x, dtype=np.complex128) for x in Hc]

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

        ctrl_bound = [float(ctrl_bound[0]), float(ctrl_bound[1])]
        if ctrl_bound == []:
            ctrl_bound = [-np.inf, np.inf]
        self.ctrl_bound = ctrl_bound

        if self.ctrl0 == []:
            if ctrl_bound == []:
                ctrl0 = [
                    2 * np.random.random(len(self.tspan) - 1)
                    - np.ones(len(self.tspan) - 1)
                    for i in range(len(self.control_Hamiltonian))
                ]
                self.control_coefficients = ctrl0
                self.ctrl0 = [np.array(ctrl0)]
            else:
                a = ctrl_bound[0]
                b = ctrl_bound[1]
                ctrl0 = [
                    (b - a) * np.random.random(len(self.tspan) - 1)
                    + a * np.ones(len(self.tspan) - 1)
                    for i in range(len(self.control_Hamiltonian))
                ]
            self.control_coefficients = ctrl0
            self.ctrl0 = [np.array(ctrl0)]
        elif len(self.ctrl0) >= 1:
            self.control_coefficients = [
                self.ctrl0[0][i] for i in range(len(self.control_Hamiltonian))
            ]

        if self.load == True:
            if os.path.exists("controls.csv"):
                data = np.genfromtxt("controls.csv")[-len(self.control_Hamiltonian) :]
                self.control_coefficients = [data[i] for i in range(len(data))]

        ctrl_num = len(self.control_coefficients)
        Hc_num = len(self.control_Hamiltonian)
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
                self.control_coefficients = np.concatenate(
                    (
                        self.control_coefficients,
                        np.zeros(len(self.control_coefficients[0])),
                    )
                )
        else:
            pass

        number = math.ceil((len(self.tspan) - 1) / len(self.control_coefficients[0]))
        if len(self.tspan) - 1 % len(self.control_coefficients[0]) != 0:
            tnum = number * len(self.control_coefficients[0])
            self.tspan = np.linspace(self.tspan[0], self.tspan[-1], tnum + 1)
        else:
            pass

        self.opt = Main.QuanEstimation.ControlOpt(
            self.control_coefficients, self.ctrl_bound
        )
        self.dynamic = Main.QuanEstimation.Lindblad(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
        )
        self.output = Main.QuanEstimation.Output(self.opt, self.savefile)

        self.dynamics_type = "lindblad"

    def QFIM(self, W=[], LDtype="SLD"):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
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

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.obj = Main.QuanEstimation.QFIM_Obj(
            self.W, self.eps, self.para_type, LDtype
        )
        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system)

    def CFIM(self, M=[], W=[]):

        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
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
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        self.obj = Main.QuanEstimation.CFIM_Obj(M, self.W, self.eps, self.para_type)
        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.run(system)

    def HCRB(self, W=[]):
        """
        Description: use differential evolution algorithm to update the control coefficients that maximize the
                     HCRB.

        ---------
        Inputs
        ---------
        W:
            --description: weight matrix.
            --type: matrix
        """
        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        if len(self.Hamiltonian_derivative) == 1:
            warnings.warn(
                "In single parameter scenario, HCRB is equivalent to QFI. \
                           Please choose QFIM as the target function for control optimization",
                DeprecationWarning,
            )
        else:

            if W == []:
                W = np.eye(len(self.Hamiltonian_derivative))
            self.W = W

            self.obj = Main.QuanEstimation.HCRB_Obj(self.W, self.eps, self.para_type)
            system = Main.QuanEstimation.QuanEstSystem(
                self.opt, self.alg, self.obj, self.dynamic, self.output
            )
            Main.QuanEstimation.run(system)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", LDtype="SLD"):
        if not (method == "binary" or method == "forward"):
            raise ValueError(
                "{!r} is not a valid value for method, supported \
                             values are 'binary' and 'forward'.".format(
                    method
                )
            )

        if self.dynamics_type != "lindblad":
            raise ValueError(
                "{!r} is not a valid type for dynamics, supported type is \
                             Lindblad dynamics.".format(
                    self.dynamics_type
                )
            )

        if len(self.Hamiltonian_derivative) > 1:
            f = 1 / f

        if M == []:
            M = SIC(len(self.rho0))
        M = [np.array(x, dtype=np.complex128) for x in M]

        if W == []:
            W = np.eye(len(self.Hamiltonian_derivative))
        self.W = W

        if M != []:
            self.obj = Main.QuanEstimation.CFIM_Obj(M, self.W, self.eps, self.para_type)
        else:
            if target == "HCRB":
                if self.para_type == "single_para":
                    warnings.warn(
                        "In single parameter scenario, HCRB is equivalent to QFI. Please \
                                   choose QFIM as the target function for control optimization",
                        DeprecationWarning,
                    )
                self.obj = Main.QuanEstimation.HCRB_Obj(
                    self.W, self.eps, self.para_type
                )
            elif target == "QFIM" or (
                LDtype == "SLD" and LDtype == "LLD" and LDtype == "RLD"
            ):
                self.obj = Main.QuanEstimation.QFIM_Obj(
                    self.W, self.eps, self.para_type, LDtype
                )
            else:
                raise ValueError(
                    "Please enter the correct values for target and LDtype.\
                                  Supported target are 'QFIM', 'CFIM' and 'HCRB',  \
                                  supported LDtype are 'SLD', 'RLD' and 'LLD'."
                )

        system = Main.QuanEstimation.QuanEstSystem(
            self.opt, self.alg, self.obj, self.dynamic, self.output
        )
        Main.QuanEstimation.mintime(method, f, system)


def ControlOpt(savefile=False, method="auto-GRAPE", **kwargs):

    if method == "auto-GRAPE":
        return ctrl.GRAPE_Copt(savefile=savefile, **kwargs, auto=True)
    elif method == "GRAPE":
        return ctrl.GRAPE_Copt(savefile=savefile, **kwargs, auto=False)
    elif method == "PSO":
        return ctrl.PSO_Copt(savefile=savefile, **kwargs)
    elif method == "DE":
        return ctrl.DE_Copt(savefile=savefile, **kwargs)
    elif method == "DDPG":
        return ctrl.DDPG_Copt(savefile=savefile, **kwargs)
    else:
        raise ValueError(
            "{!r} is not a valid value for method, supported values are 'auto-GRAPE', \
                         'GRAPE', 'PSO', 'DE', 'DDPG'.".format(
                method
            )
        )


def csv2npy_controls(controls, num):
    C_save = []
    N = int(len(controls) / num)
    for ci in range(N):
        C_tp = controls[ci * num : (ci + 1) * num]
        C_save.append(C_tp)
    np.save("controls", C_save)
