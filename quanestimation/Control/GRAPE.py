import numpy as np
import warnings
from julia import Main
import quanestimation.Control.Control as Control

class GRAPE(Control.ControlSystem):
    def __init__(self, tspan, rho0, H0, Hc=[], dH=[], ctrl_0=[], Decay=[], ctrl_bound=[], W=[], \
                 auto=True, Adam=True, max_episode=300, lr=0.01, beta1=0.90, beta2=0.99, precision=1e-6):

        Control.ControlSystem.__init__(self, tspan, rho0, H0, Hc, dH, ctrl_0, Decay, ctrl_bound, W)

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

        max_episode:
            --description: max number of training episodes.
            --type: int

        lr:
            --description: learning rate.
            --type: float

        beta1:
            --description: the exponential decay rate for the first moment estimates .
            --type: float

        beta2:
            --description: the exponential decay rate for the second moment estimates .
            --type: float

        precision:
            --description: calculation precision.
            --type: float

        """
        self.auto = auto
        self.Adam = Adam
        self.max_episode = max_episode
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.precision = precision

    def QFIM(self, save_file=False):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to calculate the gradient of QFIM (QFI) and
                     update the control coefficients that maximize the CFI or 1/Tr(WF^{-1}).

        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and QFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the QFI or Tr(WF^{-1}).
            --type: bool

        """
        grape = Main.QuanEstimation.Gradient(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, \
                self.tspan, self.Decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, \
                self.ctrl_bound, self.W, self.mt, self.vt, self.lr, self.beta1, self.beta2, self.precision)
        if self.auto == True:
            Main.QuanEstimation.auto_GRAPE_QFIM(grape, self.precision, self.max_episode, self.Adam, save_file)
        else:
            if len(self.tspan) != len(self.control_coefficients[0]):
                warnings.warn('GRAPE does not support the case when the length of each control is not equal to the \
                               length of time, and is replaced by auto-GRAPE.', DeprecationWarning)
                Main.QuanEstimation.auto_GRAPE_QFIM(grape, self.precision, self.max_episode, self.Adam, save_file)
            else:
                Main.QuanEstimation.GRAPE_QFIM(grape, self.precision, self.max_episode, self.Adam, save_file)

    def CFIM(self, Measurement, save_file=False):
        """
        Description: use auto-GRAPE (GRAPE) algorithm to calculate the gradient of CFIM (CFI) and
                     update the control coefficients that maximize the CFI or 1/Tr(WF^{-1}).
        ---------
        Inputs
        ---------
        save_file:
            --description: True: save all the control coefficients and CFI or Tr(WF^{-1}).
                           False: save the control coefficients for the last episode and all the CFI or Tr(WF^{-1}).
            --type: bool

        """

        grape = Main.QuanEstimation.Gradient(self.freeHamiltonian, self.Hamiltonian_derivative, self.rho0, self.tspan, \
                        self.Decay_opt, self.gamma, self.control_Hamiltonian, self.control_coefficients, self.ctrl_bound,\
                        self.W, self.mt, self.vt, self.lr, self.beta1, self.beta2, self.precision)
        if self.auto == True:
            Main.QuanEstimation.auto_GRAPE_CFIM(Measurement, grape, self.precision, self.max_episode, self.Adam, save_file)
        else:
            if len(self.tspan) != len(self.control_coefficients[0]):
                warnings.warn('GRAPE does not support the case when the length of each control is not equal to the length of time, \
                               and is replaced by auto-GRAPE.', DeprecationWarning)
                Main.QuanEstimation.auto_GRAPE_QFIM(grape, self.precision, self.max_episode, self.Adam, save_file)
            else:
                Main.QuanEstimation.GRAPE_CFIM(Measurement, grape, self.precision, self.max_episode, self.Adam, save_file)
