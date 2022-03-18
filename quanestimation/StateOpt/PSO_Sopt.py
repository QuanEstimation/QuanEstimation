from julia import Main
import warnings
import quanestimation.StateOpt.StateStruct as State
import numpy as np
from quanestimation.Common.common import SIC


class PSO_Sopt(State.StateSystem):
    def __init__(
        self,
        save_file=False,
        particle_num=10,
        psi0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        State.StateSystem.__init__(self, save_file, psi0, seed, load, eps)

        """
        --------
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int

        psi0:
           --description: initial guesses of states (kets).
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int

        c0:
            --description: damping factor that assists convergence.
            --type: float

        c1:
            --description: exploitation weight that attract the particle to its best previous position.
            --type: float

        c2:
            --description: exploitation weight that attract the particle to the best position in the neighborhood.
            --type: float

        seed:
            --description: random seed.
            --type: int

        """

        self.max_episode = max_episode
        self.p_num = particle_num
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        ini_particle = ([self.psi],)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        ini_particle = ([self.psi],)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        ini_particle = ([self.psi],)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().HCRB(W)
