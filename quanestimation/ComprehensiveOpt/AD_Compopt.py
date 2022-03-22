import numpy as np
from julia import Main
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp
from quanestimation.Common.common import SIC


class AD_Compopt(Comp.ComprehensiveSystem):
    def __init__(
        self,
        savefile=False,
        Adam=False,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        max_episode=300,
        epsilon=0.01,
        beta1=0.90,
        beta2=0.99,
        seed=1234,
        eps=1e-8,
    ):

        Comp.ComprehensiveSystem.__init__(
            self, psi0, ctrl0, measurement0, savefile, seed, eps
        )

        """
        ----------
        Inputs
        ----------
        savefile:
            --description: True: save the states (or controls, measurements) and the value of the 
                                 target function for each episode.
                           False: save the states (or controls, measurements) and all the value 
                                   of the target function for the last episode.
            --type: bool 
            
        Adam:
            --description: whether to use Adam to update the controls.
            --type: bool (True or False)

        psi0:
           --description: initial guesses of states (kets).
           --type: array
           
        ctrl0:
            --description: initial control coefficients.
            --type: list (of vector)
            
        measurement0:
           --description: a set of POVMs.
           --type: list (of vector)

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
        self.seed = seed

    def SC(self, W=[], M=[], target="QFIM", LDtype="SLD"):
        if M != []:
            raise ValueError(
                "AD is not available when target is 'CFIM'. Supported methods are 'PSO' and 'DE'.",
            )
        elif target == "HCRB":
            raise ValueError(
                "AD is not available when the target function is HCRB. Supported methods are 'PSO', 'DE' and 'DDPG'.",
            )

        if self.Adam:
            self.alg = Main.QuanEstimation.AD(
                self.max_episode, self.epsilon, self.beta1, self.beta2
            )
        else:
            self.alg = Main.QuanEstimation.AD(self.max_episode, self.epsilon)

        super().SC(W, M, target, LDtype)
