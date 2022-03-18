from julia import Main
import warnings
import numpy as np
import quanestimation.StateOpt.StateStruct as State
from quanestimation.Common.common import SIC


class AD_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        Adam=False,
        psi0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)

        """
        ----------
        Inputs
        ----------
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

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

        eps:
            --description: calculation eps.
            --type: float

        """

        self.Adam = Adam
        self.max_episode = max_episode
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.mt = 0.0
        self.vt = 0.0

        if self.Adam:
            self.alg = Main.QuanEstimation.AD(
                self.max_episode, self.epsilon, self.beta1, self.beta2
            )
        else:
            self.alg = Main.QuanEstimation.AD(self.max_episode, self.epsilon)

    def QFIM(self, W=[], dtype="SLD"):
        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        super().CFIM(M, W)

    def HCRB(self, W=[]):
        super().HCRB(W)
