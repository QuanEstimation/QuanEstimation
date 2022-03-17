import numpy as np
from julia import Main
import warnings
import quanestimation.ControlOpt.ControlStruct as Control
from quanestimation.Common.common import SIC


class PSO_Copt(Control.ControlSystem):
    def __init__(
        self,
        save_file=False,
        particle_num=10,
        ctrl0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        load=False,
        eps=1e-8,
    ):

        Control.ControlSystem.__init__(self, save_file, ctrl0, load, eps)

        """
        -------- 
        inputs
        --------
        particle_num:
           --description: the number of particles.
           --type: int
        
        ctrl0:
           --description: initial guesses of controls.
           --type: array

        max_episode:
            --description: max number of the training episodes.
            --type: int or array
        
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
        self.particle_num = particle_num
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def QFIM(self, W=[], dtype="SLD"):
        ini_particle = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.particle_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().QFIM(W, dtype)

    def CFIM(self, M=[], W=[]):
        ini_particle = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.particle_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().CFIM(M, W)

    def HCRB(self, W=[]):
        ini_particle = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.particle_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )
        
        super().HCRB(W)

    def mintime(self, f, W=[], M=[], method="binary", target="QFIM", dtype="SLD"):
        ini_particle = ([self.ctrl0],)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.particle_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().mintime(f,W,M,method,target,dtype)