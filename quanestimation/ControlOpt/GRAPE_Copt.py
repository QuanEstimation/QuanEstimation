import numpy as np
import warnings
from julia import Main
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC


class GRAPE_Copt(Control.ControlSystem):
    def __init__(
        self,
        savefile=False,
        Adam=True,
        ctrl0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        load=False,
        eps=1e-8,
        auto=True,
    ):

        Control.ControlSystem.__init__(self, savefile, ctrl0, load, eps)

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

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0
        self.auto = auto

    def QFIM(self, W=[], LDtype="SLD"):
        if self.auto:
            if self.Adam:
                self.alg = Main.QuanEstimation.autoGRAPE(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = Main.QuanEstimation.autoGRAPE(self.max_episode, self.epsilon)
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn("GRAPE is not available when the length of each control is not \
                               equal to the length of time, and is replaced by auto-GRAPE.",
                               DeprecationWarning)
                #### call autoGRAPE automatically ####
                if self.Adam:
                    self.alg = Main.QuanEstimation.autoGRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
                else:
                    self.alg = Main.QuanEstimation.autoGRAPE(self.max_episode, self.epsilon)
            else:
                if LDtype == "SLD":
                    if self.Adam:
                        self.alg = Main.QuanEstimation.GRAPE(
                            self.max_episode, self.epsilon, self.beta1, self.beta2
                            )
                    else:
                        self.alg = Main.QuanEstimation.GRAPE(self.max_episode, self.epsilon)
                else:
                    raise ValueError("GRAPE is only available when LDtype is SLD.")

        super().QFIM(W, LDtype)

    def CFIM(self, M=[], W=[]):
        if self.auto:
            if self.Adam:
                self.alg = Main.QuanEstimation.autoGRAPE(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = Main.QuanEstimation.autoGRAPE(self.max_episode, self.epsilon)
        else:
            if (len(self.tspan) - 1) != len(self.control_coefficients[0]):
                warnings.warn("GRAPE is not available when the length of each control is not \
                               equal to the length of time, and is replaced by auto-GRAPE.",
                               DeprecationWarning)
                #### call autoGRAPE automatically ####
                if self.Adam:
                    self.alg = Main.QuanEstimation.autoGRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
                else:
                    self.alg = Main.QuanEstimation.autoGRAPE(self.max_episode, self.epsilon)
            else:    
                if self.Adam:
                    self.alg = Main.QuanEstimation.GRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
                else:
                    self.alg = Main.QuanEstimation.GRAPE(self.max_episode, self.epsilon)

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        raise ValueError(
            "GRAPE is not available when the target function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
        )

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", LDtype="SLD"):
        if target == "HCRB":
            raise ValueError(
                "GRAPE is not available when the target function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
            )
        if self.auto:
            if self.Adam:
                self.alg = Main.QuanEstimation.autoGRAPE(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = Main.QuanEstimation.autoGRAPE(self.max_episode, self.epsilon)
        else:
            
            if self.Adam:
                self.alg = Main.QuanEstimation.GRAPE(
                        self.max_episode, self.epsilon, self.beta1, self.beta2
                    )
            else:
                self.alg = Main.QuanEstimation.GRAPE(self.max_episode, self.epsilon)

        super().mintime(f, W, M, method, target, LDtype)
