from julia import Main
import warnings
import numpy as np
import quanestimation.StateOpt.StateStruct as State
from quanestimation.Common.common import SIC


class DE_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        popsize=10,
        psi0=[],
        max_episode=1000,
        c=1.0,
        cr=0.5,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)

        """
        --------
        inputs
        --------
        popsize:
           --description: the number of populations.
           --type: int

        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int
        
        c:
            --description: mutation constant.
            --type: float

        cr:
            --description: crossover constant.
            --type: float
        
        seed:
            --description: random seed.
            --type: int
        
        """

        self.p_num = popsize
        self.max_episode = max_episode
        self.c = c
        self.cr = cr
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        ini_population = ([self.psi],)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        ini_population = ([self.psi],)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        ini_population = ([self.psi],)
        self.alg = Main.QuanEstimation.DE(
            self.max_episode,
            self.p_num,
            ini_population,
            self.c,
            self.cr,
            self.seed,
        )

        super().HCRB(W)
