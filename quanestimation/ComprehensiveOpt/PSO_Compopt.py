import numpy as np
from julia import Main
import quanestimation.ComprehensiveOpt.ComprehensiveStruct as Comp
from quanestimation.Common.common import SIC


class PSO_Compopt(Comp.ComprehensiveSystem):
    def __init__(
        self,
        savefile=False,
        particle_num=10,
        psi0=[],
        ctrl0=[],
        measurement0=[],
        max_episode=[1000, 100],
        c0=1.0,
        c1=2.0,
        c2=2.0,
        seed=1234,
        eps=1e-8,
    ):

        Comp.ComprehensiveSystem.__init__(
            self, psi0, ctrl0, measurement0, savefile, seed, eps
        )

        """
        --------
        inputs
        --------
        savefile:
            --description: True: save the states (or controls, measurements) and the value of the 
                                 target function for each episode.
                           False: save the states (or controls, measurements) and all the value 
                                   of the target function for the last episode.
            --type: bool 
        particle_num:
           --description: the number of particles.
           --type: int
        
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

        self.p_num = particle_num
        self.max_episode = max_episode
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.seed = seed

    def SC(self, W=[], M=[], target="QFIM", LDtype="SLD"):
        ini_particle = (self.psi0, self.ctrl0)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().SC(W, M, target, LDtype)

    def CM(self, rho0, W=[]):
        ini_particle = (self.ctrl0, self.measurement0)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().CM(rho0, W)

    def SM(self, W=[]):
        ini_particle = (self.psi0, self.measurement0)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().SM(W)

    def SCM(self, W=[]):
        ini_particle = (self.psi0, self.ctrl0, self.measurement0)
        self.alg = Main.QuanEstimation.PSO(
            self.max_episode,
            self.p_num,
            ini_particle,
            self.c0,
            self.c1,
            self.c2,
            self.seed,
        )

        super().SCM(W)
