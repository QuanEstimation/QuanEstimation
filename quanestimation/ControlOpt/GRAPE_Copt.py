import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control


class GRAPE_Copt(Control.ControlSystem):
    def __init__(
        self,
        tspan,
        rho0,
        H0,
        dH,
        Hc,
        decay=[],
        ctrl_bound=[],
        W=[],
        auto=True,
        Adam=True,
        ctrl0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        load=False,
    ):

        Control.ControlSystem.__init__(
            self, tspan, rho0, H0, Hc, dH, decay, ctrl_bound, W, ctrl0, load, eps=1e-8
        )

        """
        ----------
        Inputs
        ----------
        auto:
            --description: True: use autodifferential to calculate the gradient.
                                  False: calculate the gradient with analytical method.
            --type: bool (True or False)

        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

        ctrl0:
           --description: initial guess of controls.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int

        epsilon:
            --description: learning rate.
            --type: float

        beta1:
            --description: the exponential decay rate for the first moment estimates .
            --type: float

        beta2:
            --description: the exponential decay rate for the second moment estimates .
            --type: float

        """

        self.auto = auto
        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0

    def QFIM(self, save_file=False):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to update the control coefficients that maximize the
                     QFI (1/Tr(WF^{-1} with F the QFIM).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and QFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the QFI (Tr(WF^{-1})).
            --type: bool

        """
        grape = Main.QuanEstimation.GRAPE_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.mt,
            self.vt,
            self.epsilon,
            self.beta1,
            self.beta2,
            self.eps,
        )
        if self.auto == True:
            Main.QuanEstimation.QFIM_autoGRAPE_Copt(
                grape, self.max_episode, self.Adam, save_file
            )
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn(
                    "GRAPE is not available when the length of each control is not equal to the \
                               length of time, and is replaced by auto-GRAPE.",
                    DeprecationWarning,
                )
                Main.QuanEstimation.QFIM_autoGRAPE_Copt(
                    grape, self.max_episode, self.Adam, save_file
                )
            else:
                Main.QuanEstimation.QFIM_GRAPE_Copt(
                    grape, self.max_episode, self.Adam, save_file
                )

    def CFIM(self, M, save_file=False):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to update the control coefficients that maximize the
                     CFI (1/Tr(WF^{-1} with F the CFIM).
        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and CFI (Tr(WF^{-1})).
                           False: save the control coefficients for the last episode and all the CFI (Tr(WF^{-1})).
            --type: bool

        """
        M = [np.array(x, dtype=np.complex128) for x in M]
        grape = Main.QuanEstimation.GRAPE_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.mt,
            self.vt,
            self.epsilon,
            self.beta1,
            self.beta2,
            self.eps,
        )
        if self.auto == True:
            Main.QuanEstimation.CFIM_autoGRAPE_Copt(
                M, grape, self.max_episode, self.Adam, save_file
            )
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn(
                    "GRAPE is not available when the length of each control is not equal to the length of time, \
                               and is replaced by auto-GRAPE.",
                    DeprecationWarning,
                )
                Main.QuanEstimation.CFIM_autoGRAPE_Copt(
                    M, grape, self.max_episode, self.Adam, save_file
                )
            else:
                Main.QuanEstimation.CFIM_GRAPE_Copt(
                    M, grape, self.max_episode, self.Adam, save_file
                )

    def HCRB(self, save_file=False):
        warnings.warn(
            "GRAPE is not available when the objective function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
            DeprecationWarning,
        )

    def mintime(self, f, target="QFIM", W=[], M=[]):
        M = [np.array(x, dtype=np.complex128) for x in M]
        grape = Main.QuanEstimation.GRAPE_Copt(
            self.freeHamiltonian,
            self.Hamiltonian_derivative,
            self.rho0,
            self.tspan,
            self.decay_opt,
            self.gamma,
            self.control_Hamiltonian,
            self.control_coefficients,
            self.ctrl_bound,
            self.W,
            self.mt,
            self.vt,
            self.epsilon,
            self.beta1,
            self.beta2,
            self.eps,
        )
        if target == "QFIM":
            if self.auto:
                Main.QuanEstimation.mintime_binary(
                    "QFIM_autoGRAPE_Copt",
                    grape,
                    f,
                    self.max_episode,
                    self.Adam,
                )
            else:
                Main.QuanEstimation.mintime_binary(
                    "QFIM_GRAPE_Copt",
                    grape,
                    f,
                    self.max_episode,
                    self.Adam,
                )
        elif target == "CFIM":
            if self.auto:
                Main.QuanEstimation.mintime_binary(
                    "CFIM_autoGRAPE_Copt",
                    grape,
                    f,
                    M,
                    self.max_episode,
                    self.Adam,
                )
            else:
                Main.QuanEstimation.mintime_binary(
                    "CFIM_GRAPE_Copt",
                    grape,
                    f,
                    M,
                    self.max_episode,
                    self.Adam,
                )
        else:
            warnings.warn(
                "GRAPE is not available with the objective function {!r}. Supported methods are 'PSO', 'DE' and 'DDPG'.".format(
                    target
                ),
                DeprecationWarning,
            )
