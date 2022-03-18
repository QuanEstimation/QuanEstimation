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

        if self.auto:
            if self.Adam:
                self.alg = Main.QuanEstimation.AD(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = Main.QuanEstimation.AD(self.max_episode, self.epsilon)
        else:
            if self.Adam:
                self.alg = Main.QuanEstimation.GRAPE(
                    self.max_episode, self.epsilon, self.beta1, self.beta2
                )
            else:
                self.alg = Main.QuanEstimation.GRAPE(self.max_episode, self.epsilon)

    def QFIM(self, W=[], LDtype="SLD"):
        super().QFIM(W, LDtype)

    def CFIM(self, M=[], W=[]):
        super().CFIM(M, W)

    def HCRB(self, W=[]):
        warnings.warn(
            "GRAPE is not available when the target function is HCRB. \
                       Supported methods are 'PSO', 'DE' and 'DDPG'.",
            DeprecationWarning,
        )

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", LDtype="SLD"):
        if target == "HCRB":
            warnings.warn(
                "GRAPE is not available when the target function is HCRB.Supported methods are 'PSO', 'DE' and 'DDPG'.",
                DeprecationWarning,
            )

        super().mintime(f, W, M, method, target, LDtype)
