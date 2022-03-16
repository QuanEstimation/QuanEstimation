from julia import Main
import warnings
import numpy as np
import quanestimation.StateOpt.StateStruct as State
from quanestimation.Common.common import SIC

class NM_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        state_num=10,
        psi0=[],
        max_episode=1000,
        ar=1.0,
        ae=2.0,
        ac=0.5,
        as0=0.5,
        seed=1234,
        load=False,
        eps=1e-8):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)

        """
        --------
        inputs
        --------
        state_num:
           --description: the number of input states.
           --type: int
        
        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int
        
        ar:
            --description: reflection constant.
            --type: float

        ae:
            --description: expansion constant.
            --type: float

        ac:
            --description: constraction constant.
            --type: float

        as0:
            --description: shrink constant.
            --type: float
        
        seed:
            --description: random seed.
            --type: int

        eps:
            --description: calculation eps.
            --type: float
        
        """

        self.state_num = state_num
        self.max_episode = max_episode
        self.ar = ar
        self.ae = ae
        self.ac = ac
        self.as0 = as0
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        ini_state = Main.vec(self.psi)
        self.alg = Main.QuanEstimation.NM(
            self.max_episode,
            self.state_num,
            ini_state,
            self.ar,
            self.ae,
            self.ac,
            self.as0,
            self.seed
        )

        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        ini_state = Main.vec(self.psi)
        self.alg = Main.QuanEstimation.NM(
            self.max_episode,
            self.state_num,
            ini_state,
            self.ar,
            self.ae,
            self.ac,
            self.as0,
            self.seed
        )

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        ini_state = Main.vec(self.psi)
        self.alg = Main.QuanEstimation.NM(
            self.max_episode,
            self.state_num,
            ini_state,
            self.ar,
            self.ae,
            self.ac,
            self.as0,
            self.seed
        )

        super().HCRB(W)
